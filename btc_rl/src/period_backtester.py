#!/usr/bin/env python3
"""
时间段回测工具，用于对BTC交易模型进行特定时间段的回测
"""

import os
import numpy as np
import pandas as pd
import logging
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import queue

from stable_baselines3 import SAC
from gymnasium.wrappers import TimeLimit

from btc_rl.src.env import BtcTradingEnv
from btc_rl.src.config import get_config

logger = logging.getLogger(__name__)

def prepare_period_data(
    start_date: str, 
    end_date: str, 
    data_file: str = None,
    exchange: str = "binance",
    symbol: str = "BTC/USDT",
    timeframe: str = "1h"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    准备指定时间段的回测数据
    
    Args:
        start_date (str): 开始日期，格式为 YYYY-MM-DD
        end_date (str): 结束日期，格式为 YYYY-MM-DD
        data_file (str): 数据文件路径，如果为None则使用配置中的默认数据
        exchange (str): 交易所，用于获取历史数据
        symbol (str): 交易对，用于获取历史数据
        timeframe (str): 时间周期，用于获取历史数据
        
    Returns:
        tuple: (windows, prices) 用于回测的数据
    """
    # 我们将直接使用已经预处理好的test_data.npz文件
    # 这样可以确保与其他回测方法使用相同的数据
    test_data_path = "btc_rl/data/test_data.npz"
    
    if not os.path.exists(test_data_path):
        logger.error(f"找不到测试数据文件: {test_data_path}")
        return None, None
    
    logger.info(f"使用预处理数据: {test_data_path}")
    
    try:
        # 加载.npz文件
        data = np.load(test_data_path)
        windows = data["X"]  # 这是特征窗口
        prices = data["prices"]  # 这是相应的价格
        
        logger.info(f"成功加载测试数据，窗口形状: {windows.shape}, 价格数量: {prices.shape}")
        
        # 模拟日期范围过滤 - 由于我们没有实际的日期映射，将按比例假设日期范围
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # 获取CSV文件中的日期数据，用于映射
        # 获取交易对的简化名称，用于CSV文件命名
        symbol_name = symbol.replace("/", "")
        
        # 优先检查最新的CSV文件
        csv_path_latest = f"btc_rl/data/{symbol_name}_{timeframe}_2025-03-08_2025-06-09.csv"
        csv_path_default = f"btc_rl/data/{symbol_name}_{timeframe}.csv"
        
        # 准备特定日期范围的CSV文件名
        start_dt_obj = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt_obj = datetime.strptime(end_date, "%Y-%m-%d")
        date_range_csv = f"btc_rl/data/{symbol_name}_{timeframe}_{start_date}_{end_date}.csv"
        
        # 首先查找特定日期范围的CSV文件
        has_csv_mapping = False
        csv_path = None
        
        # 1. 优先查找特定日期范围的文件
        if os.path.exists(date_range_csv):
            csv_path = date_range_csv
            logger.info(f"找到特定日期范围的CSV文件: {date_range_csv}")
        # 2. 其次查找最新的CSV文件
        elif os.path.exists(csv_path_latest):
            csv_path = csv_path_latest
            logger.info(f"找到最新的CSV文件: {csv_path_latest}")
        # 3. 最后查找默认CSV文件
        elif os.path.exists(csv_path_default):
            csv_path = csv_path_default
            logger.info(f"找到默认的CSV文件: {csv_path_default}")
        
        # 如果找不到任何CSV文件，下载指定日期范围的数据
        if csv_path is None:
            logger.info(f"没有找到合适的CSV文件，将下载日期范围 {start_date} 到 {end_date} 的数据")
            try:
                from btc_rl.src.data_fetcher import fetch_historical_data, preprocess_data
                
                # 使用传入的参数下载数据
                logger.info(f"从 {exchange} 下载 {symbol} {timeframe} 数据 ({start_date} 到 {end_date})...")
                df = fetch_historical_data(
                    exchange_id=exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # 确保数据成功获取
                if df is not None and not df.empty:
                    # 预处理数据
                    df = preprocess_data(df)
                    
                    # 创建数据目录（如果不存在）
                    os.makedirs(os.path.dirname(date_range_csv), exist_ok=True)
                    
                    # 保存到CSV
                    df.to_csv(date_range_csv, index=False)
                    logger.info(f"已将下载的数据保存到: {date_range_csv}")
                    
                    # 使用新生成的CSV文件
                    csv_path = date_range_csv
                else:
                    logger.error("无法从交易所API获取数据")
            except Exception as e:
                logger.error(f"下载数据时出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # 尝试从CSV文件进行日期映射
        if csv_path and os.path.exists(csv_path):
            try:
                logger.info(f"使用CSV文件进行日期映射: {csv_path}")
                df = pd.read_csv(csv_path)
                if 'timestamp' in df.columns:
                    # 确保时间戳列是日期时间类型
                    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    # 过滤满足日期范围的记录
                    filtered_df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]
                    
                    # 如果有足够的数据点，使用过滤后的数据比例
                    if len(filtered_df) > 0:
                        start_ratio = filtered_df.index[0] / len(df)
                        end_ratio = (filtered_df.index[-1] + 1) / len(df)
                        
                        # 应用相同的比例到测试数据上
                        start_idx = int(len(windows) * start_ratio)
                        end_idx = int(len(windows) * end_ratio)
                        
                        # 确保索引有效
                        start_idx = max(0, start_idx)
                        end_idx = min(len(windows), end_idx)
                        
                        windows = windows[start_idx:end_idx]
                        prices = prices[start_idx:end_idx]
                        
                        has_csv_mapping = True
                        logger.info(f"根据日期范围过滤数据: {start_date} 到 {end_date}")
                        logger.info(f"过滤后数据点: {len(windows)}")
            except Exception as e:
                logger.error(f"尝试基于CSV文件过滤数据时出错: {e}")
        
        # 如果没有CSV映射，使用简单的线性时间假设
        if not has_csv_mapping:
            # 假设test_data.npz涵盖的整个时间范围是从2年前到现在
            total_days = 365 * 2
            now = datetime.now()
            data_start = now - timedelta(days=total_days)
            
            # 计算过滤百分比
            if start_dt > data_start:
                start_percent = (start_dt - data_start).days / total_days
            else:
                start_percent = 0
                
            if end_dt < now:
                end_percent = (end_dt - data_start).days / total_days
            else:
                end_percent = 1.0
            
            # 确保百分比在有效范围内
            start_percent = max(0.0, min(1.0, start_percent))
            end_percent = max(start_percent, min(1.0, end_percent))
            
            # 应用到数据索引
            start_idx = int(len(windows) * start_percent)
            end_idx = int(len(windows) * end_percent)
            
            # 过滤数据
            windows = windows[start_idx:end_idx]
            prices = prices[start_idx:end_idx]
            
            logger.info(f"未找到CSV日期映射，使用线性时间假设过滤数据")
            logger.info(f"假设时间范围: {start_date} ({start_percent:.2%}) 到 {end_date} ({end_percent:.2%})")
            logger.info(f"过滤后数据点: {len(windows)}")
        
        return windows, prices
        
    except Exception as e:
        logger.error(f"加载测试数据时出错: {e}")
        return None, None

def backtest_model_in_period(
    model_path: str, 
    start_date: str, 
    end_date: str, 
    market_type: str = "未知市场",
    exchange: str = "binance",
    symbol: str = "BTC/USDT",
    timeframe: str = "1h"
) -> Dict:
    """
    在特定时间段内回测模型性能
    
    Args:
        model_path (str): 模型路径
        start_date (str): 开始日期
        end_date (str): 结束日期
        market_type (str): 市场类型描述
        exchange (str): 交易所，用于获取历史数据
        symbol (str): 交易对，用于获取历史数据
        timeframe (str): 时间周期，用于获取历史数据
        
    Returns:
        dict: 回测结果
    """
    logger.info(f"开始在{market_type}条件下回测模型 ({start_date} 到 {end_date})...")
    
    # 检查模型路径是否有效
    if model_path is None or not isinstance(model_path, str):
        logger.error(f"无效的模型路径: {model_path}")
        return None
    
    # 根据市场类型调整数据筛选
    windows, prices = prepare_period_data(
        start_date=start_date, 
        end_date=end_date,
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe
    )
    
    if windows is None or prices is None:
        logger.error(f"无法获取时间段 {start_date} 到 {end_date} 的数据")
        return None
        
    # 根据市场类型进行数据筛选或调整，模拟不同市场环境
    # 创建副本以免修改原始数据
    windows_copy = np.copy(windows)
    prices_copy = np.copy(prices)
    
    data_length = len(windows_copy)
    if data_length > 0:
        if market_type == "上涨趋势环境":
            # 使用偏前部分的数据（假设前部是上涨趋势）
            quarter_point = data_length // 4
            windows_copy = windows_copy[:data_length - quarter_point]
            prices_copy = prices_copy[:data_length - quarter_point]
            logger.info(f"为上涨趋势环境选择了 {len(windows_copy)} 个数据点")
        elif market_type == "下跌趋势环境":
            # 使用偏后部分的数据（假设后部是下跌趋势）
            quarter_point = data_length // 4
            windows_copy = windows_copy[quarter_point:]
            prices_copy = prices_copy[quarter_point:]
            logger.info(f"为下跌趋势环境选择了 {len(windows_copy)} 个数据点")
        elif market_type == "震荡盘整环境":
            # 使用中间部分的数据（假设中间部分是震荡市场）
            sixth_point = data_length // 6
            windows_copy = windows_copy[sixth_point:data_length - sixth_point]
            prices_copy = prices_copy[sixth_point:data_length - sixth_point]
            logger.info(f"为震荡盘整环境选择了 {len(windows_copy)} 个数据点")
        elif market_type == "反弹行情环境":
            # 间隔采样（增加数据变化率）
            step = 2
            windows_copy = windows_copy[::step]
            prices_copy = prices_copy[::step]
            logger.info(f"为反弹行情环境选择了 {len(windows_copy)} 个数据点")
            
    # 使用处理后的数据副本
    windows = windows_copy
    prices = prices_copy
    
    logger.info(f"获取了 {len(windows)} 个数据点进行回测")
    
    try:
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            return None
        
        # 加载模型
        logger.info(f"正在加载模型: {model_path}")
        model = SAC.load(model_path)
        
        # 创建回测环境
        test_queue = queue.Queue(maxsize=1000)
        
        # 使用准备的数据创建环境
        env = BtcTradingEnv(
            windows, 
            prices, 
            websocket_queue=test_queue,
            risk_capital_source="margin_equity",
            risk_fraction_per_trade=0.02,
            max_leverage=1.0
        )
        env = TimeLimit(env, max_episode_steps=len(windows) + 1)
        
        # 初始化历史数据记录
        history_data = []
        
        # 运行回测
        obs, _ = env.reset()
        done = False
        step = 0
        
        while not done:
            # 使用模型预测动作
            action, _ = model.predict(obs, deterministic=True)
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            
            # 保存数据，每2步记录一次
            if step % 2 == 0:
                info_copy = info.copy() if isinstance(info, dict) else {}
                data_point = {
                    "step": step,
                    "action": float(action[0]) if hasattr(action, '__len__') else float(action),
                    "cash_balance": info_copy.get("cash_balance", 10000.0),
                    "margin_equity": info_copy.get("margin_equity", 10000.0),
                    "buy_and_hold_equity": info_copy.get("buy_and_hold_equity", 10000.0),
                    "upnl": info_copy.get("upnl", 0.0),
                    "reward": float(reward) if reward is not None else 0.0,
                    "price": info_copy.get("price", 0.0),
                    "position_btc": info_copy.get("position_btc", 0.0),
                    "total_fee": info_copy.get("total_fee", 0.0),
                    "was_liquidated_this_step": bool(info_copy.get("was_liquidated_this_step", False)),
                    "termination_reason": info_copy.get("termination_reason", None),
                }
                history_data.append(data_point)
        
        # 计算模型统计指标
        if not history_data:
            logger.error("回测没有产生任何历史数据")
            return None
        
        # 计算基本指标
        initial_equity = history_data[0].get("margin_equity", 10000.0)
        final_equity = history_data[-1].get("margin_equity", initial_equity)
        
        # 计算回报率
        total_return = (final_equity - initial_equity) / initial_equity if initial_equity > 0 else 0
        
        # 计算最大回撤
        peak = initial_equity
        max_drawdown = 0
        for data in history_data:
            equity = data.get("margin_equity", 0)
            if equity > peak:
                peak = equity
            else:
                drawdown = (peak - equity) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
        
        # 计算夏普比率和索提诺比率 (简化版，假设无风险利率为0)
        returns = []
        for i in range(1, len(history_data)):
            prev_equity = history_data[i-1].get("margin_equity", 0)
            curr_equity = history_data[i].get("margin_equity", 0)
            if prev_equity > 0:
                ret = (curr_equity - prev_equity) / prev_equity
                returns.append(ret)
        
        mean_return = np.mean(returns) if returns else 0
        std_return = np.std(returns) if returns and len(returns) > 1 else 1e-10
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252*24) if std_return > 0 else 0
        
        # 计算索提诺比率 (仅考虑负收益的标准差)
        negative_returns = [r for r in returns if r < 0]
        downside_std = np.std(negative_returns) if negative_returns and len(negative_returns) > 1 else 1e-10
        sortino_ratio = (mean_return / downside_std) * np.sqrt(252*24) if downside_std > 0 else 0
        
        # 计算胜率
        position_changes = []
        for i in range(1, len(history_data)):
            prev_pos = history_data[i-1].get("position_btc", 0)
            curr_pos = history_data[i].get("position_btc", 0)
            if abs(curr_pos - prev_pos) > 0.000001:  # 检测仓位变化
                entry_price = history_data[i].get("price", 0)
                position_changes.append((i, curr_pos - prev_pos, entry_price))
        
        # 计算交易盈亏
        profitable_trades = 0
        total_trades = 0
        profitable_amount = 0.0
        loss_amount = 0.0
        
        for i, (entry_idx, size_change, entry_price) in enumerate(position_changes):
            if i < len(position_changes) - 1:
                exit_idx = position_changes[i+1][0]
                exit_price = history_data[exit_idx].get("price", 0)
                
                if size_change > 0:  # 做多
                    profit = (exit_price - entry_price) * abs(size_change)
                else:  # 做空
                    profit = (entry_price - exit_price) * abs(size_change)
                
                if profit > 0:
                    profitable_trades += 1
                    profitable_amount += profit
                else:
                    loss_amount += abs(profit)
                total_trades += 1
        
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        avg_profit = profitable_amount / profitable_trades if profitable_trades > 0 else 0
        avg_loss = loss_amount / (total_trades - profitable_trades) if (total_trades - profitable_trades) > 0 else 1
        profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0
        
        # 收集最终结果
        result = {
            "model_name": os.path.basename(model_path).split('.')[0],
            "market_type": market_type,
            "test_period": f"{start_date} 到 {end_date}",
            "initial_equity": initial_equity,
            "final_equity": final_equity,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "profitable_trades_count": profitable_trades,
            "losing_trades_count": total_trades - profitable_trades,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "profit_loss_ratio": profit_loss_ratio,
            "total_fees": history_data[-1].get("total_fee", 0),
            "history_length": len(history_data)
        }
        
        return result
        
    except Exception as e:
        logger.error(f"回测过程中出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None
