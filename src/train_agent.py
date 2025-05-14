import pandas as pd
import numpy as np
from rl.agent import Agent # Assuming rl_agent.py is in the same directory
import os
import json
from datetime import datetime

# Configuration
DATA_FILE_PATH = "/Users/melihkarakose/Desktop/EC 581/data/btc_hourly_data_with_volatility.csv" # <--- CHANGED TO USE THE FILE WITH VOLATILITY
INITIAL_CAPITAL = 10000.0
WINDOW_SIZE = 100
LAMBDA_VAL = 0.75
NUM_EPISODES = 10 # Number of times to run through the data (or part of it)
LOG_EPISODES = True
EPISODES_LOG_DIR = "episodes"

def load_data(file_path):
    """Loads and preprocesses the financial data."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return None
    
    # Ensure required columns are present
    required_cols = {'timestamp', 'close', 'volume', 'log_returns', 'volatility_30_period'}
    if not required_cols.issubset(df.columns):
        print(f"Error: CSV file must contain {required_cols} columns. Found: {df.columns}")
        return None
        
    if len(df) < WINDOW_SIZE + 1: # Need at least one window and one next step
        print(f"Error: Data has {len(df)} rows, not enough for window size {WINDOW_SIZE} + 1 next step.")
        return None
        
    return df

def run_episode(agent, data_df, episode_num):
    """Runs a single episode of training/simulation."""
    print(f"--- Starting Episode {episode_num + 1} ---")
    agent.reset() # Use agent's reset method

    total_reward_episode = 0.0
    
    episode_log_data = {
        "episode_number": episode_num + 1,
        "initial_capital": agent.initial_capital,
        "lambda_reward": agent.lambda_reward,
        "window_size": agent.WINDOW_SIZE,
        "steps_data": []
    }
    # Create a reverse mapping for actions for logging
    action_map_rev = {v: k for k, v in agent.actions.items()}

    current_script_window_size = WINDOW_SIZE 

    start_index = current_script_window_size - 1
    end_index = len(data_df) - 2

    if start_index > end_index:
        print("Not enough data to run an episode after considering window and next step.")
        return

    for i in range(start_index, end_index + 1):
        # 1. Prepare the data window for the current state
        current_window_df_slice = data_df.iloc[i - agent.WINDOW_SIZE + 1 : i + 1]
        
        # 2. Get current state (processed numpy array)
        # Pass the relevant columns from the slice to preprocess_state
        state_np = agent.preprocess_state(current_window_df_slice[['log_returns', 'volatility_30_period', 'close', 'volume']])

        # 3. Agent selects an action
        action = agent.select_action(state_np)

        # 4. Determine prices and timestamp for agent's step method
        current_price_for_decision = data_df['close'].iloc[i]
        next_price_for_evaluation = data_df['close'].iloc[i+1]
        current_timestamp = data_df['timestamp'].iloc[i]

        # 5. Agent takes a step - rl_agent.py's step returns: reward, capital, info_dict
        reward, capital_after_step, step_info = agent.step(
            action, current_price_for_decision, next_price_for_evaluation, current_timestamp
        )
        
        total_reward_episode += reward

        # Extract necessary info from the dictionary
        current_pos_after_step = step_info['position']
        holding_period_after_step = step_info['holding_period']
        realized_pnl_step = step_info['realized_pnl']

        if LOG_EPISODES:
            log_step_data = {
                "step_in_episode": i - start_index + 1,
                "data_timestamp": str(current_timestamp), # Ensure serializable
                "current_price_for_decision": float(current_price_for_decision), # Added for price chart
                "action_int": int(action), # Store original action int
                "action_str": action_map_rev.get(action, "unknown"),
                "reward": float(reward),
                "capital_after_step": float(capital_after_step),
                "position_after_step": str(current_pos_after_step),
                "holding_period_after_step": int(holding_period_after_step),
                "realized_pnl_this_step": float(realized_pnl_step),
                "unrealized_pnl_at_step_end": float(step_info.get('unrealized_pnl', 0)),
                "entry_price_if_in_position": float(step_info.get('entry_price', 0)),
                "trade_units": float(step_info.get('trade_units', 0))
            }
            episode_log_data["steps_data"].append(log_step_data)

        # 6. Prepare next_state_np for the experience tuple
        next_event_idx = i + 1 
        
        done_for_buffer = (i == end_index) 

        if i == end_index: # If it's the last possible step in the episode data
            next_state_np = np.zeros_like(state_np) # No subsequent state
        else:
            next_window_start_idx = next_event_idx - agent.WINDOW_SIZE + 1
            next_window_end_idx = next_event_idx + 1
            
            if next_window_start_idx < 0 or next_window_end_idx > len(data_df):
                print(f"Warning: Cannot form full next_state window at step {i}. Using zeros for next_state.")
                next_state_np = np.zeros_like(state_np)
            else:
                next_window_data_slice = data_df.iloc[next_window_start_idx : next_window_end_idx]
                if len(next_window_data_slice) < agent.WINDOW_SIZE:
                    print(f"Warning: Next state window is shorter ({len(next_window_data_slice)}) than WINDOW_SIZE at step {i}. Using zeros for next_state.")
                    next_state_np = np.zeros_like(state_np)
                else:
                    # Pass the relevant columns for the next state
                    next_state_np = agent.preprocess_state(next_window_data_slice[['log_returns', 'volatility_30_period', 'close', 'volume']])
        
        # 7. Store experience in replay buffer
        agent.store_experience(state_np, action, reward, next_state_np, done_for_buffer)

        # 8. Agent learns from experiences
        agent.learn()

        if (i - start_index) % 100 == 0:
            current_volume = data_df['volume'].iloc[i]
            current_volatility = data_df['volatility_30_period'].iloc[i]
            print(f"Step {i - start_index + 1}: Action: {action_map_rev.get(action, 'unknown')}({action}), "
                  f"Close: {current_price_for_decision:.2f}, Vol: {current_volume:.2f}, Volatility: {current_volatility:.4f}, "
                  f"Reward: {reward:.2f}, Capital: {capital_after_step:.2f}, Position: {current_pos_after_step}, "
                  f"Holding: {holding_period_after_step}, Realized PnL: {realized_pnl_step:.2f}")

        if capital_after_step <= 0: # Example condition for ending episode
             print(f"Episode ended early at step {i - start_index + 1} due to insufficient capital.")
             if LOG_EPISODES:
                 episode_log_data["summary"] = {
                    "status": "Ended early - insufficient capital",
                    "total_reward_episode": float(total_reward_episode),
                    "final_capital": float(capital_after_step),
                    "final_position": str(current_pos_after_step),
                    "final_entry_price": float(step_info.get('entry_price', 0)),
                    "final_holding_period": int(holding_period_after_step),
                    "total_steps_executed": i - start_index + 1
                 }
             break
            
    print(f"--- Episode {episode_num + 1} Finished ---")
    # Use the last known capital and position details
    final_capital = capital_after_step
    final_position = current_pos_after_step
    final_entry_price = step_info.get('entry_price', 0) # Get from info if available
    final_holding_period = holding_period_after_step

    print(f"Total Reward: {total_reward_episode:.2f}, Final Capital: {final_capital:.2f}")
    print(f"Final Position: {final_position}, Entry: {final_entry_price:.2f}, Holding Time: {final_holding_period}")

    if LOG_EPISODES and "summary" not in episode_log_data: # If not ended early
        episode_log_data["summary"] = {
            "status": "Completed",
            "total_reward_episode": float(total_reward_episode),
            "final_capital": float(final_capital),
            "final_position": str(final_position),
            "final_entry_price": float(final_entry_price),
            "final_holding_period": int(final_holding_period),
            "total_steps_executed": end_index - start_index + 1
        }
    
    if LOG_EPISODES:
        if not os.path.exists(EPISODES_LOG_DIR):
            os.makedirs(EPISODES_LOG_DIR)
        
        log_filename = datetime.now().strftime("%d%m%Y-%H%M%S") + f"_ep{episode_num + 1}.json"
        log_filepath = os.path.join(EPISODES_LOG_DIR, log_filename)
        
        try:
            with open(log_filepath, 'w') as f:
                json.dump(episode_log_data, f, indent=4)
            print(f"Episode log saved to: {log_filepath}")
        except Exception as e:
            print(f"Error saving episode log: {e}")

def main():
    data = load_data(DATA_FILE_PATH)

    if data is not None:
        trading_agent = Agent(
            initial_capital=INITIAL_CAPITAL,
            lambda_reward=LAMBDA_VAL 
        )

        for episode in range(NUM_EPISODES):
            run_episode(trading_agent, data, episode)
    else:
        print("Could not load data. Exiting.")

if __name__ == "__main__":
    main()

