import pandas as pd
import numpy as np
from rl.agent import Agent # Assuming rl_agent.py is in the same directory
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split # Import for data splitting

# Configuration
DATA_FILE_PATH = "/Users/melihkarakose/Desktop/EC 581/data/btc_hourly_data_with_volatility.csv" # <--- CHANGED TO USE THE FILE WITH VOLATILITY
INITIAL_CAPITAL = 10000.0
WINDOW_SIZE = 100
LAMBDA_VAL = 0.75
NUM_EPISODES = 10 # Number of times to run through the data (or part of it)
LOG_EPISODES = True
EPISODES_LOG_DIR = "episodes"

# Global variable to hold the current run's log directory
CURRENT_RUN_LOG_DIR = None

def calculate_sharpe_ratio(capital_over_time, risk_free_rate=0.0):
    """Calculates the Sharpe ratio for a series of capital values."""
    if len(capital_over_time) < 2:
        return 0.0
    
    capital_series = pd.Series(capital_over_time)
    returns = capital_series.pct_change().dropna()
    
    if len(returns) < 1 or np.std(returns) == 0:
        return 0.0
        
    excess_returns = returns - risk_free_rate # Assuming risk_free_rate is per-period
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
    return sharpe_ratio

def run_buy_and_hold_strategy(data_df, initial_capital):
    """Simulates a Buy and Hold strategy."""
    if data_df.empty or len(data_df) < 2:
        print("Not enough data for Buy and Hold strategy.")
        return initial_capital, 0.0, [initial_capital]

    buy_price = data_df['close'].iloc[0]
    
    capital_over_time_bh = [initial_capital]
    current_capital = initial_capital
    peak_capital = initial_capital
    max_drawdown_bh = 0.0

    for i in range(1, len(data_df)):
        current_price = data_df['close'].iloc[i]
        current_capital = (current_price / buy_price) * initial_capital
        capital_over_time_bh.append(current_capital)

        peak_capital = max(peak_capital, current_capital)
        drawdown = (peak_capital - current_capital) / peak_capital
        max_drawdown_bh = max(max_drawdown_bh, drawdown)
        
    final_capital_bh = capital_over_time_bh[-1]
    return final_capital_bh, max_drawdown_bh, capital_over_time_bh

def load_data(file_path, test_size=0.3, validation_size=0.0): # Updated default test_size to 0.3
    """Loads and preprocesses the financial data, then splits it."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return None, None, None
    
    required_cols = {'timestamp', 'close', 'volume', 'log_returns', 'volatility_30_period'}
    if not required_cols.issubset(df.columns):
        print(f"Error: CSV file must contain {required_cols} columns. Found: {df.columns}")
        return None, None, None
        
    if len(df) < WINDOW_SIZE + 1: # Keep this check
        print(f"Error: Data has {len(df)} rows, not enough for window size {WINDOW_SIZE} + 1 next step.")
        return None, None, None # Return three values as expected by caller
        
    # Simplified split: train and test only
    if test_size >= 1.0 or test_size < 0:
        print("Error: test_size must be between 0.0 and 1.0")
        return None, None, None # Return three values

    train_df, test_df = train_test_split(df, test_size=test_size, shuffle=False)
    
    # Validation dataframe is now empty
    val_df = pd.DataFrame()

    print(f"Data loaded. Train size: {len(train_df)}, Test size: {len(test_df)}")
    return train_df, val_df, test_df # val_df is now empty

def run_episode(agent, data_df, episode_num, dataset_name="Training", evaluation_mode=False, log_dir=None):
    """Runs a single episode of training/simulation."""
    print(f"--- Starting Episode {episode_num + 1} on {dataset_name} Data {'(Evaluation)' if evaluation_mode else ''} ---")
    agent.reset()
    # Set agent's evaluation mode based on the parameter
    if hasattr(agent, 'set_evaluation_mode'):
        agent.set_evaluation_mode(evaluation_mode)
        if evaluation_mode:
            print("Agent correctly set to evaluation mode for this episode.")
        else:
            print("Agent correctly set to training mode for this episode.")
    else:
        print("Warning: Agent class does not have 'set_evaluation_mode' method. Cannot dynamically switch modes.")

    total_reward_episode = 0.0
    capital_over_time = [agent.initial_capital]
    
    peak_capital = agent.initial_capital
    max_drawdown_episode = 0.0
    winning_trades = 0
    losing_trades = 0
    
    previous_position_str = "NEUTRAL"

    episode_log_data = {
        "episode_number": episode_num + 1,
        "initial_capital": agent.initial_capital,
        "lambda_reward": agent.lambda_reward,
        "window_size": agent.WINDOW_SIZE,
        "steps_data": []
    }
    action_map_rev = {v: k for k, v in agent.actions.items()}

    current_script_window_size = WINDOW_SIZE 

    start_index = current_script_window_size - 1
    end_index = len(data_df) - 2

    if start_index > end_index:
        print("Not enough data to run an episode after considering window and next step.")
        # Ensure summary is logged even if episode doesn't run
        if LOG_EPISODES:
            episode_log_data["summary"] = {
                "status": "Not enough data",
                "dataset_name": dataset_name,
                "evaluation_mode": evaluation_mode,
                "total_reward_episode": 0,
                "final_capital": agent.initial_capital,
                "max_drawdown": 0,
                "winning_trades": 0,
                "losing_trades": 0
            }
            log_filename_suffix = f"_ep{episode_num + 1}"
            if evaluation_mode:
                log_filename_suffix += f"_{dataset_name}_eval_nodata"
            else: # Training mode
                log_filename_suffix += f"_{dataset_name}_train_nodata" # Added _train and dataset_name
            log_filename = datetime.now().strftime("%d%m%Y-%H%M%S") + log_filename_suffix + ".json"
            log_dir_to_use = log_dir if log_dir is not None else EPISODES_LOG_DIR
            if not os.path.exists(log_dir_to_use):
                os.makedirs(log_dir_to_use)
            log_filepath = os.path.join(log_dir_to_use, log_filename)
            try:
                with open(log_filepath, 'w') as f:
                    json.dump(episode_log_data, f, indent=4)
                print(f"Episode log (no data) saved to: {log_filepath}")
            except Exception as e:
                print(f"Error saving episode log (no data): {e}")
        return agent.initial_capital, 0.0, 0.0, 0, 0, [agent.initial_capital]

    for i in range(start_index, end_index + 1):
        current_window_df_slice = data_df.iloc[i - agent.WINDOW_SIZE + 1 : i + 1]
        state_np = agent.preprocess_state(current_window_df_slice[['log_returns', 'volatility_30_period', 'close', 'volume']])

        action = agent.select_action(state_np) # select_action now respects agent.evaluation_mode

        current_price_for_decision = data_df['close'].iloc[i]
        next_price_for_evaluation = data_df['close'].iloc[i+1]
        current_timestamp = data_df['timestamp'].iloc[i]

        reward, capital_after_step, step_info = agent.step(
            action, current_price_for_decision, next_price_for_evaluation, current_timestamp
        )
        
        total_reward_episode += reward
        capital_over_time.append(capital_after_step)

        peak_capital = max(peak_capital, capital_after_step)
        if peak_capital > 0:
            drawdown = (peak_capital - capital_after_step) / peak_capital
            max_drawdown_episode = max(max_drawdown_episode, drawdown)

        realized_pnl_this_step = step_info.get('realized_pnl_this_step', 0.0)

        trade_closed_or_flipped = False
        # Use position_before_action and position_after_action for win/loss logic
        pos_before = step_info.get("position_before_action", "none")
        pos_after = step_info.get("position_after_action", "none")

        if pos_before != "none" and pos_after == "none": # Position was closed
            trade_closed_or_flipped = True
        elif (pos_before == "long" and pos_after == "short") or \
             (pos_before == "short" and pos_after == "long"): # Position was flipped
            trade_closed_or_flipped = True

        if trade_closed_or_flipped and realized_pnl_this_step != 0:
            if realized_pnl_this_step > 0:
                winning_trades += 1
            elif realized_pnl_this_step < 0:
                losing_trades += 1
        
        if LOG_EPISODES: # Changed: Log steps for both training and evaluation
            log_step_data = {
                "step_in_episode": i - start_index + 1,
                "data_timestamp": str(current_timestamp),
                "current_price_for_decision": float(current_price_for_decision),
                "action_int": int(action),
                "action_str": step_info.get("action_taken", action_map_rev.get(action, "unknown")), # Use from step_info
                "reward": float(reward),
                "capital_after_step": float(capital_after_step),
                "position_before_action": str(pos_before),
                "position_after_step": str(pos_after),
                "holding_period_after_step": int(step_info.get('holding_period_after_step', 0)),
                "realized_pnl_this_step": float(realized_pnl_this_step),
                "unrealized_pnl_at_step_end": float(step_info.get('unrealized_pnl_at_step_end', 0)),
                "entry_price_if_in_position": float(step_info.get('entry_price_if_in_position', 0)),
                "trade_units": float(step_info.get('trade_units', 0))
            }
            episode_log_data["steps_data"].append(log_step_data)

        if not evaluation_mode:
            next_event_idx = i + 1
            done_for_buffer = (i == end_index)
            if i == end_index: 
                next_state_np = np.zeros_like(state_np)
            else:
                next_window_start_idx = next_event_idx - agent.WINDOW_SIZE + 1
                next_window_end_idx = next_event_idx + 1
                if next_window_start_idx < 0 or next_window_end_idx > len(data_df):
                    next_state_np = np.zeros_like(state_np)
                else:
                    next_window_data_slice = data_df.iloc[next_window_start_idx : next_window_end_idx]
                    if len(next_window_data_slice) < agent.WINDOW_SIZE:
                        next_state_np = np.zeros_like(state_np)
                    else:
                        next_state_np = agent.preprocess_state(next_window_data_slice[['log_returns', 'volatility_30_period', 'close', 'volume']])

            agent.store_experience(state_np, action, reward, next_state_np, done_for_buffer)
            agent.learn()

        if (i - start_index) % 100 == 0:
            current_volume = data_df['volume'].iloc[i]
            current_volatility = data_df['volatility_30_period'].iloc[i]
            print(f"Step {i - start_index + 1}: Action: {step_info.get('action_taken', action_map_rev.get(action, 'unknown'))}({action}), "
                  f"Close: {current_price_for_decision:.2f}, Vol: {current_volume:.2f}, Volatility: {current_volatility:.4f}, "
                  f"Reward: {reward:.2f}, Capital: {capital_after_step:.2f}, Position: {pos_after}, "
                  f"Realized PnL: {realized_pnl_this_step:.2f}")

        if capital_after_step <= 0:
             print(f"Episode ended early at step {i - start_index + 1} due to insufficient capital.")
             if LOG_EPISODES:
                 episode_log_data["summary"] = {
                    "status": "Ended early - insufficient capital",
                    "total_reward_episode": float(total_reward_episode),
                    "final_capital": float(capital_after_step),
                    "final_position": str(pos_after),
                    "final_entry_price": float(step_info.get('entry_price_if_in_position', 0)),
                    "total_steps_executed": i - start_index + 1
                 }
             break
            
    print(f"--- Episode {episode_num + 1} on {dataset_name} Finished {'(Evaluation)' if evaluation_mode else ''} ---")
    final_capital = capital_after_step
    final_position = pos_after
    final_entry_price = step_info.get('entry_price_if_in_position', 0)

    print(f"Total Reward: {total_reward_episode:.2f}, Final Capital: {final_capital:.2f}")
    print(f"Final Position: {final_position}, Entry: {final_entry_price:.2f}")

    if LOG_EPISODES:
        summary_status = "Completed"
        if capital_after_step <= 0 and "summary" not in episode_log_data : # Ensure status is set if loop broke early
             summary_status = "Ended early - insufficient capital"
        elif "summary" in episode_log_data and "status" in episode_log_data["summary"]:
             summary_status = episode_log_data["summary"]["status"]
        
        episode_log_data["summary"] = {
            "status": summary_status,
            "dataset_name": dataset_name,
            "evaluation_mode": evaluation_mode,
            "total_reward_episode": float(total_reward_episode),
            "final_capital": float(capital_after_step if 'capital_after_step' in locals() else agent.initial_capital),
            "max_drawdown": float(max_drawdown_episode),
            "winning_trades": winning_trades,
            "losing_trades": losing_trades
        }
        log_filename_suffix = f"_ep{episode_num + 1}"
        if evaluation_mode:
            log_filename_suffix += f"_{dataset_name}_eval"
        else: # Training mode
            log_filename_suffix += f"_{dataset_name}_train"
        log_filename = datetime.now().strftime("%d%m%Y-%H%M%S") + log_filename_suffix + ".json"
        log_dir_to_use = log_dir if log_dir is not None else EPISODES_LOG_DIR
        if not os.path.exists(log_dir_to_use):
            os.makedirs(log_dir_to_use)
        log_filepath = os.path.join(log_dir_to_use, log_filename)
        try:
            with open(log_filepath, 'w') as f:
                json.dump(episode_log_data, f, indent=4)
            print(f"Episode log saved to: {log_filepath}")
        except Exception as e:
            print(f"Error saving episode log: {e}")
            
    return (capital_after_step if 'capital_after_step' in locals() else agent.initial_capital), \
           total_reward_episode, max_drawdown_episode, \
           winning_trades, losing_trades, capital_over_time

def main():
    global CURRENT_RUN_LOG_DIR
    # Create a unique subfolder for this run
    run_timestamp = datetime.now().strftime("%d%m%Y-%H%M%S")
    CURRENT_RUN_LOG_DIR = os.path.join(EPISODES_LOG_DIR, run_timestamp)
    if not os.path.exists(CURRENT_RUN_LOG_DIR):
        os.makedirs(CURRENT_RUN_LOG_DIR)

    train_data, val_data, test_data = load_data(DATA_FILE_PATH)

    if train_data is not None and not train_data.empty:
        trading_agent = Agent(
            initial_capital=INITIAL_CAPITAL,
            lambda_reward=LAMBDA_VAL 
        )

        print("\n=== Training Phase ===")
        for episode in range(NUM_EPISODES):
            if len(train_data) < WINDOW_SIZE + 2: # Ensure enough data for at least one step
                print(f"Not enough training data for episode {episode + 1}. Skipping.")
                continue
            run_episode(trading_agent, train_data, episode, dataset_name="Training", evaluation_mode=False, log_dir=CURRENT_RUN_LOG_DIR)
        
        print("\n=== Evaluation Phase ===")
        if test_data is not None and not test_data.empty:
            if len(test_data) < WINDOW_SIZE + 2: # Ensure enough data for at least one step
                print("Not enough test data for evaluation. Skipping.")
            else:
                print("\n--- Evaluating Agent on Test Data ---")
                agent_final_capital, agent_total_reward, agent_max_dd, \
                agent_wins, agent_losses, agent_capital_over_time = run_episode(
                    trading_agent, test_data, 0, dataset_name="Test", evaluation_mode=True, log_dir=CURRENT_RUN_LOG_DIR
                )
                agent_sharpe = calculate_sharpe_ratio(agent_capital_over_time)
                agent_wl_ratio = agent_wins / agent_losses if agent_losses > 0 else float('inf') if agent_wins > 0 else 0

                print("\n--- Evaluating Buy and Hold on Test Data ---")
                bh_final_capital, bh_max_dd, bh_capital_over_time = run_buy_and_hold_strategy(test_data, INITIAL_CAPITAL)
                bh_sharpe = calculate_sharpe_ratio(bh_capital_over_time)

                print("\n=== Performance Summary on Test Data ===")
                print(f"{'Metric':<25} {'Agent':<15} {'Buy & Hold':<15}")
                print("-" * 55)
                print(f"{'Final Capital':<25} {agent_final_capital:<15.2f} {bh_final_capital:<15.2f}")
                print(f"{'Max Drawdown':<25} {agent_max_dd:<15.4f} {bh_max_dd:<15.4f}")
                print(f"{'Sharpe Ratio':<25} {agent_sharpe:<15.4f} {bh_sharpe:<15.4f}")
                print(f"{'Winning Trades':<25} {agent_wins:<15} {'N/A':<15}")
                print(f"{'Losing Trades':<25} {agent_losses:<15} {'N/A':<15}")
                print(f"{'Win/Loss Ratio':<25} {agent_wl_ratio:<15.2f} {'N/A':<15}")
        else:
            print("No test data to evaluate.")
    else:
        print("Could not load or split data. Exiting.")

if __name__ == "__main__":
    main()

