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
*   **RL Agent Base**: `src/rl_agent.py` defines an `Agent` class with:
    *   Initialization of capital, observation window size, and a lambda for the reward function.
    *   Defined action space (long, short, hold).
    *   A state representation based on a 100-hour sliding window, now using the last 99 periods of **log returns, 30-period volatility, window-normalized close prices, and window-normalized volumes** as input features for the DQN.
    *   A PnL and capital update mechanism within its `step` method.
*   **Deep Q-Network (DQN) Implementation (Updated May 14, 2025):**
    *   The `Agent` in `src/rl_agent.py` now incorporates a DQN using a **1D CNN architecture** (built with PyTorch).
    *   The input to the DQN is based on a window of **log returns, 30-period volatility, window-normalized close prices, and window-normalized volumes**. Close prices and volumes are normalized within each window to the [0,1] range.
    *   It uses a Replay Buffer to store experiences (state, action, reward, next_state, done).
    *   Action selection is managed by an epsilon-greedy strategy, with epsilon decaying over time to balance exploration and exploitation.
    *   The agent includes a `learn()` method to update the DQN's weights using sampled experiences from the replay buffer.
    *   A target network is used for stable Q-learning updates.
*   **Training Environment**: `src/train_rl_agent.py` provides a simulation loop:
    *   Loads historical data.
    *   Manages episodes, resetting the agent for each.
    *   Feeds data to the agent ensuring no future data leakage.
    *   Calls the agent's `step` method, which now internally handles PnL and capital updates.
    *   The training loop now calls `agent.store_experience()` to save experiences and `agent.learn()` to update the DQN.
*   **Episode Logging & Visualization (New as of May 13, 2025):**
    *   `src/train_rl_agent.py` now logs detailed information for each episode (parameters, step-by-step data including actions, rewards, capital, PnL, and prices) into timestamped JSON files in an `episodes/` directory.
    *   A `visualization/episode_visualizer.html` file has been created. This tool allows users to load one or more episode JSON files and visualizes:
        *   Episode summary.
        *   Charts for Price, Capital Over Time, and Reward Per Step/Cumulative Reward.
        *   A detailed Trade Log table.
*   **Runnable Simulation**: The project can be run using `python3 train_rl_agent.py`. The agent will learn and logs will be generated.

## Next Steps

Refer to `todo.md` for the latest task list.

Key upcoming tasks include:

1.  ~~**Refine Training Loop (`train_rl_agent.py`)**:~~ (Completed)
    *   ~~Integrate the DQN's learning process: call `store_experience` after each step and `learn` periodically or after each episode.~~
2.  ~~**Transition to 1D CNN**: Update the DQN model in `src/rl_agent.py` from the current MLP to a 1D Convolutional Neural Network to better capture temporal patterns in the price window.~~ (Completed)
3.  **Implement a Learning Algorithm**:
    *   Replace the random action selection in `src/rl_agent.py` with a proper RL algorithm (e.g., Q-learning, Deep Q-Network (DQN), Proximal Policy Optimization (PPO)).
    *   Add a learning mechanism to the agent (e.g., updating Q-tables or neural network weights).
4.  **Incorporate Realism**:
    *   Add transaction costs (e.g., a percentage of trade value) to the PnL calculation.
    *   Consider slippage effects.
5.  **Enhance State/Feature Engineering**:
    *   The state representation now includes **log returns, 30-period volatility, window-normalized close prices, and window-normalized volumes**.
    *   Further enhancements could include other technical indicators (RSI, MACD, Bollinger Bands) or other relevant market features.
6.  **Robust Evaluation**:
    *   Split the historical data into training, validation, and test sets to prevent overfitting and get a true measure of performance on unseen data.
    *   Implement more comprehensive backtesting metrics (e.g., Sharpe ratio, Sortino ratio, max drawdown).
7.  **Hyperparameter Tuning**:
    *   Tune the parameters of the RL algorithm (learning rate, discount factor, exploration rate, network architecture if using NNs) and the agent (lambda in reward, window size).
8.  **Experimentation**:
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
    *   Run `python3 train_rl_agent.py`. This will train the agent (which now uses `data/btc_hourly_data_with_volatility.csv`) and save episode logs in the `episodes/` directory.
6.  **Visualize Episodes:**
    *   Open `visualization/episode_visualizer.html` in a web browser.
    *   Use the file input to select one or more JSON log files from the `episodes/` directory to visualize the trading performance.

## Scripts

*   `src/download_data.py`: Downloads BTC hourly data from Binance (2022-01-01 to 2025-01-01) using the `ccxt` library and saves it as `data/btc_hourly_data_2022_2025.csv`.
*   `src/filter_existing_csv.py`: Filters an existing CSV file (e.g., `data/btc_hourly_data_2022_2025.csv`) to include only the 'timestamp', 'close', and 'volume' columns, saving the output to `data/btc_hourly_data_filtered_from_existing.csv`.
*   `src/calculate_volatility.py`: Reads `data/btc_hourly_data_filtered_from_existing.csv`, calculates `log_returns` and `volatility_30_period` (30-hour rolling standard deviation of log returns), and saves the enriched data to `data/btc_hourly_data_with_volatility.csv`.
*   `src/rl_agent.py`: Contains the Reinforcement Learning trading agent. The agent uses a Deep Q-Network (DQN) with a **1D CNN architecture** (PyTorch) to make trading decisions (long, short, or hold). Its state is based on a 100-hour window of past market data, specifically using the last 99 periods of **log returns, 30-period volatility, window-normalized close prices, and window-normalized volumes**. It includes a reward function, replay buffer for experience storage, and an epsilon-greedy strategy for action selection. The agent learns by sampling experiences and updating its Q-network.
*   `src/train_rl_agent.py`: Provides the training environment for the `src/rl_agent.py`. It loads historical price data (now from `data/btc_hourly_data_with_volatility.csv`), runs episodes where the agent makes decisions step-by-step, stores experiences, triggers learning, and logs detailed episode data to JSON files.
*   `visualization/episode_visualizer.html`: An HTML/CSS/JS tool to load and visualize the JSON episode logs, showing charts for price, capital, rewards, and a detailed trade log.

### `src/rl_agent.py`

This script defines the `Agent` class which is trained using Deep Reinforcement Learning (DQN).

**Key Features:**
*   Initializes with a starting capital (e.g., $10,000).
*   Observes a sliding window of the past 100 hours of data, processed into 99 periods of **log returns, 30-period volatility, window-normalized close prices, and window-normalized volumes**.
*   Decides to go long, short, or hold/do nothing for the next hour using a DQN.
*   Reward function: Based on realized PnL and discounted unrealized PnL of the current position.
*   **DQN Architecture**: **1D CNN-based** (PyTorch).
*   **Learning**: Uses a Replay Buffer and learns via Q-learning updates with a target network.
*   **Exploration**: Employs an epsilon-greedy strategy.

**TODO (related to `src/rl_agent.py`):**
*   ~~Transition DQN model from MLP to 1D CNN.~~ (Completed)
*   Potentially refine reward function further based on training performance.
*   Consider alternative normalization strategies for input features if needed.

### `src/train_rl_agent.py`

This script sets up and runs the trading simulation for the RL agent.

**Key Functions:**
*   Loads the filtered historical price data (e.g., from `btc_hourly_data_with_volatility.csv`).
*   Initializes the `Agent` from `src/rl_agent.py`.
*   Runs a specified number of episodes. In each episode:
    *   The agent's state (capital, position) is reset.
    *   The script iterates through the historical data, providing a sliding window of `WINDOW_SIZE` (e.g., 100 hours) to the agent.
    *   The agent selects an action based on the current window using its DQN.
    *   The agent's `step` method is called with the chosen action, the current price for decision-making, and the next price for evaluation.
    *   Experiences `(state, action, reward, next_state, done)` are stored in the agent's replay buffer via `agent.store_experience()`.
    *   The agent's learning process is triggered by calling `agent.learn()`.
    *   Rewards and changes in capital are tracked.
*   Logs detailed information about each episode to a JSON file in the `episodes/` directory.

**TODO (related to `src/train_rl_agent.py`):**
*   ~~Modify the training loop to call `agent.store_experience()` with (state, action, reward, next_state, done) after each step.~~ (Completed)
*   ~~Call `agent.learn()` at appropriate intervals (e.g., after every step or every few steps, once the replay buffer has enough samples).~~ (Completed)

