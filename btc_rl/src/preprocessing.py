#!/usr/bin/env python3
"""
Create train / test .npz datasets of rolling 100-hour windows
for the BTC RL project.

Usage
-----
python -m btc_rl.src.preprocessing --csv /btc_rl/data/BTC_hourly.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "log_returns",
    "volatility_30_period",
    "RSI14",
    "SMA50",
]

WINDOW_SIZE = 100  # Total period for one sample (observation + execution period)
OBS_FEATURE_LENGTH = WINDOW_SIZE - 1  # Length of the observation window (e.g., 99 hours)


def minmax_scale(window: np.ndarray) -> np.ndarray:
    """Min-max scale each column of a (100, 9) window -> (100, 9)."""
    w_min = window.min(axis=0, keepdims=True)
    w_max = window.max(axis=0, keepdims=True)
    denom = np.where(w_max - w_min == 0, 1.0, w_max - w_min)
    return (window - w_min) / denom


def build_windows(arr: np.ndarray, win: int = 100):
    """
    arr shape  -> (T, features)
    returns    -> (T - win + 1, win, features)
    """
    from numpy.lib.stride_tricks import sliding_window_view

    return sliding_window_view(arr, (win, arr.shape[1]))[:, 0, ...]


def main(csv_path: Path):
    # 1) read & clean
    df = (
        pd.read_csv(csv_path)
        .dropna(subset=FEATURES)  # drop rows with any NaN in feature set
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # 2) slice features, convert to numpy
    data_all_features = df[FEATURES].to_numpy(dtype=np.float32)
    prices_all_open = df["open"].to_numpy(dtype=np.float32)  # execution price candidates

    # 3) create rolling windows for the full period (obs + potential exec period)
    # all_windows[k] = data_all_features[k : k+WINDOW_SIZE]
    all_windows = build_windows(data_all_features, win=WINDOW_SIZE)
    # N_samples = data_all_features.shape[0] - WINDOW_SIZE + 1

    # 4) extract observation part from these windows
    # observation_windows[k] = data_all_features[k : k+OBS_FEATURE_LENGTH]
    observation_windows = all_windows[:, :OBS_FEATURE_LENGTH, :]

    # 5) align execution prices: price at the start of the hour *after* the observation window ends
    # For observation_windows[k] (data from original index k to k+OBS_FEATURE_LENGTH-1),
    # the execution price is prices_all_open[k+OBS_FEATURE_LENGTH].
    # Since all_windows starts at index 0 of data_all_features,
    # and observation_windows[k] corresponds to data from k to k+OBS_FEATURE_LENGTH-1,
    # the corresponding execution price is prices_all_open[k + OBS_FEATURE_LENGTH].
    # The first execution price will be prices_all_open[OBS_FEATURE_LENGTH].
    # The number of samples is len(observation_windows).
    exec_prices = prices_all_open[OBS_FEATURE_LENGTH : OBS_FEATURE_LENGTH + len(observation_windows)]

    # 6) scale each observation window individually
    obs_windows_scaled = np.stack([minmax_scale(w) for w in observation_windows])

    # 7) train / test chronological split (70 / 30)
    split_idx = int(len(obs_windows_scaled) * 0.7)
    X_train, X_test = obs_windows_scaled[:split_idx], obs_windows_scaled[split_idx:]
    p_train, p_test = exec_prices[:split_idx], exec_prices[split_idx:]

    # 8) save
    Path("btc_rl/data").mkdir(exist_ok=True)
    np.savez_compressed("btc_rl/data/train_data.npz", X=X_train, prices=p_train)
    np.savez_compressed("btc_rl/data/test_data.npz", X=X_test, prices=p_test)

    print(
        f"✅ Saved: train_data.npz ({X_train.shape[0]} samples, obs shape {X_train.shape[1:]}) • "
        f"test_data.npz ({X_test.shape[0]} samples, obs shape {X_test.shape[1:]})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path, help="Path to BTC_hourly.csv")
    args = parser.parse_args()
    main(args.csv)