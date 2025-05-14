import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import pandas as pd

# Hyperparameters
BUFFER_SIZE = int(1e5)  # Replay buffer size
BATCH_SIZE = 64         # Minibatch size
GAMMA = 0.99            # Discount factor
LR = 5e-4               # Learning rate
TARGET_UPDATE_FREQUENCY = 100 # How often to update the target network
EPS_START = 0.9         # Starting value of epsilon
EPS_END = 0.05          # Minimum value of epsilon
EPS_DECAY = 200         # Epsilon decay rate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN_CNN(nn.Module):
    """1D CNN for processing time-series data with 2 features."""
    def __init__(self, sequence_length, action_size, num_features=2, seed=42):
        super(DQN_CNN, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the flattened size after conv and pool layers
        # Input: (batch, num_features, sequence_length)
        # After conv1: (batch, 16, sequence_length)
        # After pool1: (batch, 16, sequence_length // 2)
        # After conv2: (batch, 32, sequence_length // 2)
        # After pool2: (batch, 32, (sequence_length // 2) // 2)
        flattened_size_after_conv = 32 * ((sequence_length // 2) // 2)
        
        self.fc1 = nn.Linear(flattened_size_after_conv, 128)
        self.fc_out = nn.Linear(128, action_size)

    def forward(self, state):
        # Expect state to be of shape (batch_size, num_features, sequence_length)
        x = F.relu(self.conv1(state))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        return self.fc_out(x)

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed=42):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        np.random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        # States are now expected to be (num_features, sequence_length)
        # We stack them along a new batch dimension: (batch_size, num_features, sequence_length)
        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        # Next_states also need to be stacked correctly
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

class Agent:
    WINDOW_SIZE = 100
    NUM_FEATURES = 4 # log_returns, volatility, close_price, volume

    def __init__(self, initial_capital=10000, lambda_reward=0.75, seed=42):
        self.initial_capital = initial_capital
        self.lambda_reward = lambda_reward
        self.actions = {"short": 0, "long": 1, "hold": 2}
        self.action_size = len(self.actions)
        
        # The input to the CNN will be a sequence of (WINDOW_SIZE - 1) steps, each with NUM_FEATURES
        self.sequence_length = self.WINDOW_SIZE - 1
        # self.state_size is no longer a flat dimension but rather (NUM_FEATURES, self.sequence_length)
        # The DQN_CNN will take sequence_length and num_features as parameters

        self.qnetwork_local = DQN_CNN(self.sequence_length, self.action_size, self.NUM_FEATURES, seed).to(device)
        self.qnetwork_target = DQN_CNN(self.sequence_length, self.action_size, self.NUM_FEATURES, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        self.memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
        self.t_step = 0
        self.steps_done = 0
        self.evaluation_mode = False # Added for evaluation mode

        # Initialize trade-specific state
        self.trade_units = 0

        self.reset()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def reset(self):
        self.capital = self.initial_capital
        self.current_position = "none" # Explicitly "none", "long", or "short"
        self.entry_price = 0
        self.holding_period = 0
        self.total_reward = 0
        self.trade_units = 0 # Reset trade units

    def preprocess_state(self, state_window_df):
        """ 
        Processes a window of data containing 'log_returns', 'volatility_30_period', 'close', and 'volume'.
        Expected input is a pandas DataFrame with WINDOW_SIZE rows.
        Output shape: (NUM_FEATURES, WINDOW_SIZE - 1)
        """
        required_cols_for_state = {'log_returns', 'volatility_30_period', 'close', 'volume'}
        if not isinstance(state_window_df, pd.DataFrame) or not required_cols_for_state.issubset(state_window_df.columns):
            # Return a zero state with the correct shape
            return np.zeros((self.NUM_FEATURES, self.sequence_length), dtype=np.float32)

        if len(state_window_df) != self.WINDOW_SIZE:
            # Return a zero state with the correct shape
            return np.zeros((self.NUM_FEATURES, self.sequence_length), dtype=np.float32)

        log_returns = state_window_df['log_returns'].values[1:] # Shape (WINDOW_SIZE-1,)
        volatility = state_window_df['volatility_30_period'].values[1:] # Shape (WINDOW_SIZE-1,)
        close_prices = state_window_df['close'].values[1:] # Shape (WINDOW_SIZE-1,)
        volumes = state_window_df['volume'].values[1:] # Shape (WINDOW_SIZE-1,)


        log_returns = np.nan_to_num(log_returns, nan=0.0)
        volatility = np.nan_to_num(volatility, nan=0.0)
        
        # Normalize close_prices within the window to [0, 1]
        min_price = np.min(close_prices)
        max_price = np.max(close_prices)
        if max_price > min_price:
            close_prices_normalized = (close_prices - min_price) / (max_price - min_price)
        else:
            close_prices_normalized = np.zeros_like(close_prices) # All prices in window are the same

        # Normalize volumes within the window to [0, 1]
        min_volume = np.min(volumes)
        max_volume = np.max(volumes)
        if max_volume > min_volume:
            volumes_normalized = (volumes - min_volume) / (max_volume - min_volume)
        else:
            volumes_normalized = np.zeros_like(volumes) # All volumes in window are the same
        
        # Stack features to create shape (NUM_FEATURES, sequence_length)
        # The order of stacking matters for how the CNN will interpret the channels.
        processed_state = np.stack((log_returns, volatility, close_prices_normalized, volumes_normalized), axis=0) 
        
        return processed_state.astype(np.float32)

    def select_action(self, state_np, add_noise=True): # add_noise is effectively deprecated by evaluation_mode
        # state_np is expected to be (NUM_FEATURES, sequence_length)
        
        if self.evaluation_mode: # If in evaluation mode, no exploration
            state = torch.from_numpy(state_np).float().unsqueeze(0).to(device)
            self.qnetwork_local.eval() # Ensure model is in eval mode
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            # self.qnetwork_local.train() # No need to switch back to train here, managed by set_evaluation_mode
            return np.argmax(action_values.cpu().data.numpy())

        # Training mode: proceed with epsilon-greedy
        epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        
        if random.random() > epsilon:
            state = torch.from_numpy(state_np).float().unsqueeze(0).to(device)
            self.qnetwork_local.eval() # Temporarily set to eval for inference
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train() # Set back to train mode
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.t_step = (self.t_step + 1) % TARGET_UPDATE_FREQUENCY
        if self.t_step == 0:
            self.update_target_network()
            
    def update_target_network(self):
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def set_evaluation_mode(self, is_eval_mode: bool):
        """Sets the agent to evaluation or training mode."""
        self.evaluation_mode = is_eval_mode
        if self.evaluation_mode:
            self.qnetwork_local.eval()
            self.qnetwork_target.eval() # Also set target network to eval mode
        else:
            self.qnetwork_local.train()
            self.qnetwork_target.train() # Also set target network to train mode

    def step(self, action_idx, current_price_for_decision, next_price_for_evaluation, current_timestamp):
        action_str = [k for k, v in self.actions.items() if v == action_idx][0]
        position_before_action = self.current_position # Store position before action

        realized_pnl_this_step = 0
        # Note: unrealized_pnl variable here is for info dict, distinct from uPnL_after_action used in reward
        unrealized_pnl_info = 0 

        if self.current_position != "none":
            self.holding_period += 1
        else:
            # self.holding_period is reset in open_position or close_position
            # but if agent chooses 'hold' while having no position, it should remain 0.
            self.holding_period = 0 

        if action_str == "long":
            if self.current_position == "short": # Close short, then go long
                realized_pnl_this_step = (self.entry_price - current_price_for_decision) * self.trade_units
                self.capital += realized_pnl_this_step
                self.open_position("long", current_price_for_decision)
            elif self.current_position == "none": # Go long
                self.open_position("long", current_price_for_decision)
            # If already long, action "long" implies holding; no PnL realized.

        elif action_str == "short":
            if self.current_position == "long": # Close long, then go short
                realized_pnl_this_step = (current_price_for_decision - self.entry_price) * self.trade_units
                self.capital += realized_pnl_this_step
                self.open_position("short", current_price_for_decision)
            elif self.current_position == "none": # Go short
                self.open_position("short", current_price_for_decision)
            # If already short, action "short" implies holding; no PnL realized.

        elif action_str == "hold": # 'Hold' action implies closing the position if one exists
            if self.current_position == "long": # Close long position
                realized_pnl_this_step = (current_price_for_decision - self.entry_price) * self.trade_units
                self.capital += realized_pnl_this_step
                self.close_position()
            elif self.current_position == "short": # Close short position
                realized_pnl_this_step = (self.entry_price - current_price_for_decision) * self.trade_units
                self.capital += realized_pnl_this_step
                self.close_position()
            # If no position, "hold" does nothing. realized_pnl_this_step remains 0.

        # Calculate uPnL for the reward component (based on position *after* action, at *next* price)
        uPnL_for_reward = 0
        if self.current_position == "long": # Position is currently long (either newly opened or was long and action wasn't close)
            uPnL_for_reward = (next_price_for_evaluation - self.entry_price) * self.trade_units
            unrealized_pnl_info = uPnL_for_reward # For info dict
        elif self.current_position == "short": # Position is currently short
            uPnL_for_reward = (self.entry_price - next_price_for_evaluation) * self.trade_units
            unrealized_pnl_info = uPnL_for_reward # For info dict
        
        # --- MODIFIED REWARD CALCULATION START ---
        # Define reward modification parameters (these can be tuned or made configurable)
        LOSS_AVERSION_MULTIPLIER = 1.5  # Penalize losses more
        CAPITAL_PRESERVATION_PENALTY_AMOUNT = 25 # Fixed penalty amount if capital is below initial

        current_step_reward = 0

        # 1. Apply realized PnL, with extra penalty for losses
        if realized_pnl_this_step < 0:
            current_step_reward += realized_pnl_this_step * LOSS_AVERSION_MULTIPLIER
        else:
            current_step_reward += realized_pnl_this_step

        # 2. Add discounted unrealized PnL if a position is active (as before)
        if self.current_position != "none":
            current_step_reward += uPnL_for_reward * (self.lambda_reward ** self.holding_period)

        # 3. Apply penalty if capital is below initial capital
        if self.capital < self.initial_capital:
            current_step_reward -= CAPITAL_PRESERVATION_PENALTY_AMOUNT
            # Optional: Could make this penalty proportional to the deficit, e.g.:
            # deficit = self.initial_capital - self.capital
            # current_step_reward -= (deficit / self.initial_capital) * SOME_SCALING_FACTOR 
        # --- MODIFIED REWARD CALCULATION END ---
        
        self.total_reward += current_step_reward

        info = {
            "action_taken": action_str,
            "position_before_action": position_before_action,
            "position_after_action": self.current_position, # current_position is updated by open/close
            "realized_pnl_this_step": realized_pnl_this_step, 
            "unrealized_pnl_at_step_end": unrealized_pnl_info, 
            "holding_period_after_step": self.holding_period, 
            "entry_price_if_in_position": self.entry_price,
            "trade_units": self.trade_units,
            "capital_after_step": self.capital, # Add capital to info
            "current_price_for_decision": current_price_for_decision, # Add price to info
            "data_timestamp": current_timestamp # Add timestamp to info
        }
        return current_step_reward, self.capital, info

    def open_position(self, position_type, price):
        self.current_position = position_type
        self.entry_price = price
        self.holding_period = 1 # Start holding period
        if price != 0:
            self.trade_units = self.initial_capital / price
        else:
            self.trade_units = 0 # Avoid division by zero, though price shouldn't be 0

    def close_position(self):
        self.current_position = "none"
        self.entry_price = 0
        self.holding_period = 0
        self.trade_units = 0 # Reset trade units when position is closed

if __name__ == '__main__':
    agent = Agent(initial_capital=10000, lambda_reward=0.75)

    dummy_numpy_data_window = np.random.rand(100, 3) 
    dummy_numpy_data_window[:, 1] = np.cumsum(np.random.randn(100)) + 50000 
    
    state_from_numpy = agent.preprocess_state(dummy_numpy_data_window[:, 1])
    print(f"Sample state from NumPy (price changes): {state_from_numpy[:5]}...")

    action = agent.select_action(state_from_numpy)
    print(f"Selected action: {action}")

    price_for_decision = dummy_numpy_data_window[-1, 1]
    price_for_evaluation = price_for_decision + (np.random.randn() * 10)

    print(f"Agent initial capital: {agent.capital:.2f}, Position: {agent.current_position}")
    print(f"Taking action {action} at decision price {price_for_decision:.2f}, evaluation price {price_for_evaluation:.2f}")
    
    reward, capital, info = agent.step(action, price_for_decision, price_for_evaluation, current_timestamp=None)
    
    print(f"After step -> Reward: {reward:.2f}, Capital: {capital:.2f}")
    print(f"Agent position: {info['position_after_action']}, Entry: {info['entry_price_if_in_position']:.2f}, Holding time: {info['holding_period_after_step']}")
