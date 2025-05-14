# Reinforcement Learning for Cryptocurrency Trading

This project aims to develop a reinforcement learning agent for trading cryptocurrencies, specifically focusing on BTC/USDT.

## Project Goals

- Download historical cryptocurrency data (BTC/USDT hourly data from Binance).
- Preprocess and prepare the data for a reinforcement learning environment.
- Design and implement a reinforcement learning agent.
- Train the agent on historical data.
- Backtest and evaluate the agent's trading performance.

## Current Status

- Historical hourly BTC/USDT data from 2022-01-01 to 2025-01-01 has been downloaded using the `ccxt` library and saved to `btc_hourly_data_2022_2025.csv`.
- A Python script (`src/download_data.py`) is available to perform this download.
- A Python script (`src/filter_existing_csv.py`) is available to filter the downloaded data to include only timestamp, close, and volume, saving it to `btc_hourly_data_filtered_from_existing.csv`.
- A Python script (`src/calculate_volatility.py`) is available to calculate log returns and 30-period volatility, saving the output to `btc_hourly_data_with_volatility.csv`.
- A `todo.md` file tracks the project's progress and next steps.

## Current Status (As of May 14, 2025)

*   **Data Pipeline**: Scripts are in place to download raw BTC hourly data, filter it, and calculate log returns and 30-period volatility (`src/calculate_volatility.py` producing `data/btc_hourly_data_with_volatility.csv`).
*   **RL Agent (`src/rl/agent.py`)**: Defines an `Agent` class with:
    *   Initialization of capital, observation window size, and a lambda for the reward function.
    *   Defined action space (long, short, hold).
    *   A state representation based on a 100-hour sliding window, now using the last 99 periods of **log returns, 30-period volatility, window-normalized close prices, and window-normalized volumes** as input features for the DQN.
    *   A PnL and capital update mechanism within its `step` method.
*   **Deep Q-Network (DQN) Implementation (Updated May 14, 2025):**
    *   The `Agent` now includes an `evaluation_mode` flag and a `set_evaluation_mode(is_eval_mode: bool)` method to switch between training (with exploration) and evaluation (deterministic) behavior. This also sets the PyTorch models to `train()` or `eval()` mode accordingly.
    *   The `Agent` in `src/rl_agent.py` now incorporates a DQN using a **1D CNN architecture** (built with PyTorch).
    *   The input to the DQN is based on a window of **log returns, 30-period volatility, window-normalized close prices, and window-normalized volumes**. Close prices and volumes are normalized within each window to the [0,1] range.
    *   It uses a Replay Buffer to store experiences (state, action, reward, next_state, done).
    *   Action selection is managed by an epsilon-greedy strategy, with epsilon decaying over time to balance exploration and exploitation.
    *   The agent includes a `learn()` method to update the DQN's weights using sampled experiences from the replay buffer.
    *   A target network is used for stable Q-learning updates.
*   **Training Environment (`src/train_agent.py`)**: Provides a simulation loop:
    *   Loads historical data.
    *   Manages episodes, resetting the agent for each.
    *   Feeds data to the agent ensuring no future data leakage.
    *   Calls the agent's `step` method, which now internally handles PnL and capital updates.
    *   The training loop now calls `agent.store_experience()` to save experiences and `agent.learn()` to update the DQN.
    *   `run_episode` function now correctly calls `agent.set_evaluation_mode()` based on whether the episode is for training or evaluation.
    *   Utilizes the more detailed `step_info` from `agent.step()` for accurate metric calculation (e.g., Win/Loss Ratio) and logging.\
*   **Data Splitting (Implemented May 14, 2025):**
    *   The `load_data` function in `src/train_agent.py` now splits data into 70% training, 15% validation, and 15% test sets while maintaining chronological order.
*   **Enhanced Evaluation Framework (Implemented May 14, 2025):**
    *   `src/train_agent.py` now calculates Sharpe Ratio, Maximum Drawdown, and Win/Loss Ratio.
    *   A "Buy and Hold" baseline strategy has been implemented for comparison.
    *   The agent is evaluated on the test set against the baseline, and a performance summary is printed.
    *   These metrics are also included in the JSON episode logs.
*   **Episode Logging & Visualization (Updated May 14, 2025):**
    *   `src/train_agent.py` (renamed from `train_rl_agent.py`) logs detailed episode information.
    *   `visualization/episode_visualizer.html` now features:
        *   Larger, more readable charts.
        *   A combined Price and Capital chart with dual Y-axes.
*   **Runnable Simulation**: The project can be run using `python3 src/train_agent.py`. The agent will train, and then its performance (along with a Buy & Hold baseline) will be evaluated on the test set. Logs will be generated in the `episodes/` directory.
*   **Data Splitting & Evaluation Metrics (Updated May 14, 2025):**
    *   Historical data is now split into training, validation, and test sets to ensure robust evaluation and prevent overfitting.
    *   Evaluation metrics include Sharpe ratio, Sortino ratio, and maximum drawdown for comprehensive performance analysis.
\
## Episode Log File Naming (Updated May 14, 2025)

Episode logs generated by `src/train_agent.py` are now saved with filenames that clearly indicate whether they are from training or evaluation runs:

- Training episodes: `DATETIME_ep<N>_<DATASET_NAME>_train.json` (e.g., `14052025-170000_ep1_Training_train.json`)
- Evaluation episodes: `DATETIME_ep<N>_<DATASET_NAME>_eval.json` (e.g., `14052025-173000_ep1_Test_eval.json`)
- If an episode could not run due to insufficient data: `_train_nodata` or `_eval_nodata` is appended before `.json`.

This makes it easy to distinguish between training and evaluation logs in the `episodes/` directory.

## Next Steps

Refer to `todo.md` for the latest task list.

Key upcoming tasks include:

1.  **Agent Class Modifications (`src/rl/agent.py`):** (Completed)
    *   Implemented `set_evaluation_mode(self, is_eval_mode: bool)` method.
    *   `step_info` returned by `agent.step()` now provides consistent and detailed information.
2.  **Further Evaluation Enhancements:**
    *   Implement a Random Agent baseline for comparison.
    *   Use the validation set (`val_data`) for hyperparameter tuning.
    *   Address items from `todo.md` such as transaction costs and additional technical indicators (RSI, MACD, Bollinger Bands).
3.  **Incorporate Realism**:
    *   Add transaction costs (e.g., a percentage of trade value) to the PnL calculation.
    *   Consider slippage effects.
4.  **Experimentation**:
    *   Try different reward functions.
    *   Experiment with different action spaces (e.g., varying position sizes).

## Setup & Usage

1.  **Prerequisites:**
    *   Python 3.x
    *   `ccxt` library (`pip install ccxt`)
    *   `pandas` library (`pip install pandas`)
    *   `numpy` library (`pip install numpy`)
    *   `torch` library (`pip install torch`)
2.  **Download Data:**
    *   Run `python src/download_data.py` to fetch the latest hourly BTC/USDT data from Binance.
3.  **Filter Data (Optional):**
    *   Run `python src/filter_existing_csv.py` to create a filtered version of the data with only timestamp, close, and volume columns.
4.  **Calculate Volatility & Log Returns:**
    *   Run `python src/calculate_volatility.py` to process `data/btc_hourly_data_filtered_from_existing.csv` and create `data/btc_hourly_data_with_volatility.csv` which includes `log_returns` and `volatility_30_period`.
5.  **Train Agent:**
    *   Run `python3 src/train_agent.py`. This will train the agent (which now uses `data/btc_hourly_data_with_volatility.csv`), evaluate it on the test set, and save episode logs in the `episodes/` directory.
6.  **Visualize Episodes:**
    *   Open `visualization/episode_visualizer.html` in a web browser.
    *   Use the file input to select one or more JSON log files from the `episodes/` directory to visualize the trading performance.

## Scripts

*   `src/download_data.py`: Downloads BTC hourly data from Binance (2022-01-01 to 2025-01-01) using the `ccxt` library and saves it as `data/btc_hourly_data_2022_2025.csv`.
*   `src/filter_existing_csv.py`: Filters an existing CSV file (e.g., `data/btc_hourly_data_2022_2025.csv`) to include only the 'timestamp', 'close', and 'volume' columns, saving the output to `data/btc_hourly_data_filtered_from_existing.csv`.
*   `src/calculate_volatility.py`: Reads `data/btc_hourly_data_filtered_from_existing.csv`, calculates `log_returns` and `volatility_30_period` (30-hour rolling standard deviation of log returns), and saves the enriched data to `data/btc_hourly_data_with_volatility.csv`.
*   `src/rl/agent.py`: Contains the Reinforcement Learning trading agent. The agent uses a Deep Q-Network (DQN) with a **1D CNN architecture** (PyTorch) to make trading decisions (long, short, or hold). Its state is based on a 100-hour window of past market data, specifically using the last 99 periods of **log returns, 30-period volatility, window-normalized close prices, and window-normalized volumes**. It includes a reward function, replay buffer for experience storage, and an epsilon-greedy strategy for action selection. The agent learns by sampling experiences and updating its Q-network.
*   `src/train_agent.py`: Provides the training environment for the `src/rl/agent.py`. It loads historical price data (now from `data/btc_hourly_data_with_volatility.csv`), runs episodes where the agent makes decisions step-by-step, stores experiences, triggers learning, and logs detailed episode data to JSON files.
*   `visualization/episode_visualizer.html`: An HTML/CSS/JS tool to load and visualize the JSON episode logs, showing charts for price, capital, rewards, and a detailed trade log.

### `src/rl/agent.py`

This script defines the `Agent` class which is trained using Deep Reinforcement Learning (DQN).

**Key Features:**
*   Initializes with a starting capital (e.g., $10,000).
*   Observes a sliding window of the past 100 hours of data, processed into 99 periods of **log returns, 30-period volatility, window-normalized close prices, and window-normalized volumes**.
*   Decides to go long, short, or hold/do nothing for the next hour using a DQN.
*   Reward function: Based on realized PnL and discounted unrealized PnL of the current position.
*   **DQN Architecture**: **1D CNN-based** (PyTorch).
*   **Learning**: Uses a Replay Buffer and learns via Q-learning updates with a target network.
*   **Exploration**: Employs an epsilon-greedy strategy.

**Key Features (Updated May 14, 2025):**
*   Includes `set_evaluation_mode(is_eval_mode: bool)` to manage exploration and model state (train/eval).
*   `step()` method returns a detailed `step_info` dictionary with fields like `action_taken`, `position_before_action`, `position_after_action`, `realized_pnl_this_step`, etc.

**TODO (related to `src/rl/agent.py`):**
*   Potentially refine reward function further based on training performance.
*   Consider alternative normalization strategies for input features if needed.

### `src/train_agent.py`

This script sets up and runs the trading simulation for the RL agent.

**Key Functions (Updated May 14, 2025):**
*   Loads historical data and splits into train, validation, and test sets.
*   Initializes the `Agent` from `src/rl/agent.py`.
*   Runs a specified number of episodes. In each episode:
    *   The agent's state (capital, position) is reset.
    *   The script iterates through the historical data, providing a sliding window of `WINDOW_SIZE` (e.g., 100 hours) to the agent.
    *   The agent selects an action based on the current window using its DQN.
    *   The agent's `step` method is called, and its detailed `step_info` is used for metric calculation and logging.
    *   Experiences are stored, and learning is triggered during training episodes.
    *   `agent.set_evaluation_mode()` is called appropriately for training vs. evaluation episodes.
*   Logs detailed episode data, including refined metrics and step information.
*   Evaluates the trained agent on a test set and compares its performance against a Buy & Hold baseline.

**TODO (related to `src/train_agent.py`):**
*   Implement use of the validation set (`val_data`) for hyperparameter tuning or early stopping.

