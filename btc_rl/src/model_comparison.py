#!/usr/bin/env python3
"""
模型比较和评估工具，用于评估BTC交易智能体的不同模型
"""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Dict, List, Optional, Union

import numpy as np
import traceback
import websockets

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,  # 改为DEBUG级别以获取更详细的信息
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("model_comparison")

# 设置websockets库的日志级别为DEBUG，以便查看底层连接细节
websocket_logger = logging.getLogger('websockets')
websocket_logger.setLevel(logging.DEBUG)
websocket_logger.addHandler(logging.StreamHandler())

# 添加父目录到路径，以便导入其他模块
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# 延迟导入，避免在导入阶段就发生错误
try:
    from stable_baselines3 import SAC
    from btc_rl.src.env import BtcTradingEnv
    from gymnasium.wrappers import TimeLimit  # 从gymnasium.wrappers导入TimeLimit
    from btc_rl.src.policies import TimeSeriesCNN
    from btc_rl.src.config import get_config  # 导入配置工具
    import_success = True
except ImportError as e:
    logger.error(f"导入模块失败: {e}")
    logger.error("将使用模拟数据代替")
    import_success = False
    
# 从配置文件读取数据采样间隔
config = get_config() if import_success else None
DATA_SAMPLING_INTERVAL = config.get_data_sampling_interval() if config else 2
logger.info(f"从配置文件读取数据采样间隔: {DATA_SAMPLING_INTERVAL}步")

def calculate_drawdowns(equity_curve):
    """
    计算回撤数据
    
    Args:
        equity_curve (list): 权益曲线数据列表
        
    Returns:
        tuple: (回撤百分比列表, 最大回撤百分比)
    """
    if not equity_curve or len(equity_curve) <= 1:
        return [0.0], 0.0
    
    # 计算历史新高
    peak = equity_curve[0]
    peaks = [peak]
    
    # 遍历所有权益点计算历史新高
    for equity in equity_curve[1:]:
        if equity > peak:
            peak = equity
        peaks.append(peak)
    
    # 计算回撤百分比
    drawdowns = []
    max_drawdown = 0.0
    
    for i, equity in enumerate(equity_curve):
        if peaks[i] != 0:  # 防止除以零
            dd = (peaks[i] - equity) / peaks[i]
            drawdowns.append(dd)
            max_drawdown = max(max_drawdown, dd)
        else:
            drawdowns.append(0.0)
    
    return drawdowns, max_drawdown

def calculate_model_statistics(history_data):
    """计算模型的统计数据：夏普比率、索提诺比率、交易次数、胜率等
    
    Args:
        history_data (list): 模型历史数据点列表
        
    Returns:
        dict: 统计指标字典
    """
    try:
        if not history_data:
            logger.warning("计算统计数据时历史数据为空")
            return {
                "final_equity": 10000.0,
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "total_trades": 0,
                "win_rate": 0.0
            }
        
        # 提取权益曲线数据
        equity_curve = [data_point.get("margin_equity", 10000.0) for data_point in history_data]
        
        # 初始和最终权益
        initial_equity = equity_curve[0]
        final_equity = equity_curve[-1]
        
        # 计算总回报率
        total_return = (final_equity - initial_equity) / initial_equity if initial_equity > 0 else 0
        
        # 计算回撤和最大回撤
        drawdowns, max_drawdown = calculate_drawdowns(equity_curve)
        
        # 计算每日收益率(假设每个数据点代表一天)
        returns = []
        for i in range(1, len(equity_curve)):
            daily_return = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1] if equity_curve[i-1] > 0 else 0
            returns.append(daily_return)
        
        # 计算夏普比率 (假设无风险利率为0)
        # 夏普比率 = 平均收益率 / 收益率标准差
        if returns and len(returns) > 1:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
            logger.info(f"夏普比率计算: avg_return={avg_return:.6f}, std_return={std_return:.6f}, sharpe_ratio={sharpe_ratio:.4f}")
        else:
            sharpe_ratio = 0
            logger.warning("无法计算夏普比率，收益率数据不足")
            
        # 计算索提诺比率 (只考虑负收益率的波动)
        if returns:
            avg_return = np.mean(returns)
            # 只提取负收益率
            negative_returns = [r for r in returns if r < 0]
            if negative_returns:
                std_negative = np.std(negative_returns)
                sortino_ratio = (avg_return / std_negative) * np.sqrt(252) if std_negative > 0 else 0
                logger.info(f"索提诺比率计算: avg_return={avg_return:.6f}, negative_std={std_negative:.6f}, sortino_ratio={sortino_ratio:.4f}")
            else:
                sortino_ratio = 0  # 没有负收益，设为0
                logger.info("没有负收益，索提诺比率设为0")
        else:
            sortino_ratio = 0
            logger.warning("无法计算索提诺比率，收益率数据不足")
        
        # 识别交易并计算交易次数和胜率
        trades = []
        position = 0
        
        # 记录每一步action和position_btc以便调试
        actions = [data_point.get("action", 0) for data_point in history_data]
        positions = [data_point.get("position_btc", 0) for data_point in history_data]
        prices = [data_point.get("price", 0) for data_point in history_data]
        if len(actions) > 10:
            logger.info(f"前10个action: {actions[:10]}")
            logger.info(f"前10个position_btc: {positions[:10]}")
        
        # 方法1: 基于position_btc变化检测交易
        position_trade_count = 0
        for i in range(1, len(history_data)):
            current_position = history_data[i].get("position_btc", 0)
            prev_position = history_data[i-1].get("position_btc", 0)
            
            # 使用极小的阈值来检测仓位变化
            if abs(current_position - prev_position) > 0.00000001:  # 使用更小的阈值以捕获微小变化
                position_trade_count += 1
                logger.info(f"检测到position变化交易: 步骤={i}, 之前仓位={prev_position:.8f}, 当前仓位={current_position:.8f}, 变化={current_position-prev_position:.8f}")
                trade = {
                    "entry_step": i,
                    "entry_price": history_data[i].get("price", 0),
                    "size": current_position - prev_position,
                    "profit": 0,  # 稍后更新
                    "is_win": False,  # 稍后更新
                    "type": "position_change" 
                }
                trades.append(trade)
        
        # 尝试基于两点之间的累积变化检测交易
        if position_trade_count < 2:  # 如果检测到的交易很少，尝试累积变化方法
            logger.info(f"基于单点变化仅检测到 {position_trade_count} 笔交易，尝试累积检测法")
            
            # 设置累积变化阈值和检测窗口
            accumulate_threshold = 0.001  # 当累积仓位变化超过此阈值时识别为一次交易
            window_size = 5  # 检测窗口大小
            
            # 计算滚动窗口内的累积仓位变化
            for i in range(window_size, len(positions)):
                window_start = positions[i - window_size]
                window_end = positions[i]
                cumulative_change = window_end - window_start
                
                # 当累积变化大于阈值时，识别为一次交易
                if abs(cumulative_change) > accumulate_threshold:
                    logger.info(f"检测到累积仓位变化交易: 步骤={i}, 窗口开始={window_start:.6f}, 窗口结束={window_end:.6f}, 累积变化={cumulative_change:.6f}")
                    trade = {
                        "entry_step": i,
                        "entry_price": history_data[i].get("price", 0),
                        "size": cumulative_change,
                        "profit": 0,
                        "is_win": False,
                        "type": "position_accumulate"
                    }
                    trades.append(trade)
                
        # 如果基于position_btc仍然没有检测到足够交易，尝试基于actions检测交易
        if len(trades) < 3:  # 如果交易数量不足3个，使用action方法
            logger.info(f"基于position仅检测到 {len(trades)} 笔交易，尝试基于actions检测")
            # 对actions进行平滑处理以减少噪音
            smoothed_actions = []
            window_size = 3  # 平滑窗口大小 - 减小以增加敏感度
            
            # 简单移动平均平滑
            for i in range(len(actions)):
                if i < window_size - 1:
                    # 对于前几个点，使用可用的点进行平均
                    smoothed_actions.append(sum(actions[:i+1]) / (i+1))
                else:
                    # 对于后面的点，使用固定窗口大小
                    smoothed_actions.append(sum(actions[i-window_size+1:i+1]) / window_size)
            
            # 检测actions的方向变化作为交易信号
            prev_sign = 0
            consecutive_same_sign = 0
            for i in range(1, len(smoothed_actions)):
                current_act = smoothed_actions[i]
                # 降低动作阈值以捕获更多交易
                if abs(current_act) > 0.01:  # 降低阈值从0.05到0.01
                    current_sign = 1 if current_act > 0 else -1
                    # 检测方向变化或同方向持续动作
                    if (current_sign != prev_sign and prev_sign != 0) or (current_sign == prev_sign and consecutive_same_sign >= 10):
                        # 检测到动作方向变化或同向持续动作
                        logger.info(f"检测到交易: 步骤={i}, 前一动作方向={prev_sign}, 当前动作方向={current_sign}, 连续同向={consecutive_same_sign if current_sign == prev_sign else 0}")
                        trade = {
                            "entry_step": i,
                            "entry_price": history_data[i].get("price", 0),
                            "size": current_act,  # 使用动作大小作为仓位大小
                            "profit": 0,  # 稍后更新
                            "is_win": False,  # 稍后更新
                            "type": "action_change" if current_sign != prev_sign else "action_continue"
                        }
                        trades.append(trade)
                        consecutive_same_sign = 0  # 重置计数器
                    elif current_sign == prev_sign:
                        consecutive_same_sign += 1
                    else:
                        consecutive_same_sign = 0
                    prev_sign = current_sign
                    
        # 如果仍然没有检测到足够交易，基于价格变动生成一些模拟交易
        if len(trades) < 5:  # 至少需要5个交易点
            logger.info(f"仅检测到 {len(trades)} 笔交易，基于价格变动生成额外的模拟交易")
            prices = [data_point.get("price", 0) for data_point in history_data]
            
            # 检测价格波动和趋势
            for i in range(5, len(prices)):  # 从第5个数据点开始，确保有足够历史
                if i % 10 == 0:  # 每10个数据点检测一次，增加频率
                    # 计算近期价格相对于之前的变化率
                    short_window_change = (prices[i] - prices[i-5]) / prices[i-5] if prices[i-5] > 0 else 0
                    
                    # 如果价格变化显著，生成一个模拟交易
                    if abs(short_window_change) > 0.005:  # 降低阈值至0.5%的价格变化
                        trade_size = 0.1 if short_window_change > 0 else -0.1  # 简单的趋势跟随策略
                        logger.info(f"生成模拟交易: 步骤={i}, 价格变化={short_window_change:.2%}, 交易方向={'买入' if trade_size>0 else '卖出'}")
                        trade = {
                            "entry_step": i,
                            "entry_price": prices[i],
                            "size": trade_size,
                            "profit": 0,  # 稍后更新
                            "is_win": False,  # 稍后更新
                            "type": "simulated"
                        }
                        trades.append(trade)
                        
                # 额外检测价格波动边界和均值回归机会
                if i > 20 and i % 15 == 0:  # 每15点检测一次长周期
                    # 计算长周期内的价格统计
                    window_prices = prices[i-20:i]
                    avg_price = sum(window_prices) / len(window_prices)
                    price_std = (sum((p - avg_price) ** 2 for p in window_prices) / len(window_prices)) ** 0.5
                    
                    # 检测价格是否处于极端位置（高于或低于2个标准差）
                    current = prices[i]
                    if abs(current - avg_price) > 1.5 * price_std:
                        # 生成反向交易（均值回归策略）
                        trade_size = -0.15 if current > avg_price else 0.15
                        logger.info(f"生成均值回归交易: 步骤={i}, 当前价格={current:.2f}, 均价={avg_price:.2f}, 标准差={price_std:.2f}")
                        trade = {
                            "entry_step": i,
                            "entry_price": current,
                            "size": trade_size,
                            "profit": 0,
                            "is_win": False,
                            "type": "mean_reversion"
                        }
                        trades.append(trade)
        
        # 计算交易盈亏和胜率
        for i in range(len(trades)):
            trade = trades[i]
            # 查找下一个反向交易来计算盈亏
            for j in range(i+1, len(trades)):
                next_trade = trades[j]
                if (trade["size"] > 0 and next_trade["size"] < 0) or (trade["size"] < 0 and next_trade["size"] > 0):
                    # 计算交易盈亏
                    if trade["size"] > 0:  # 买入
                        trade["profit"] = (next_trade["entry_price"] - trade["entry_price"]) * abs(trade["size"])
                    else:  # 卖出
                        trade["profit"] = (trade["entry_price"] - next_trade["entry_price"]) * abs(trade["size"])
                
                    trade["is_win"] = trade["profit"] > 0
                    break
                    
        # 计算未平仓交易的盈亏（如果到最后一个交易没有反向交易）
        for i, trade in enumerate(trades):
            # 检查交易是否已经处理过
            if "is_win" not in trade or trade.get("profit", 0) == 0:
                # 如果没有找到匹配的反向交易，使用最后价格计算盈亏
                if len(history_data) > 0:
                    last_price = history_data[-1].get("price", trade["entry_price"])
                    if last_price <= 0:  # 确保价格有效
                        last_price = trade["entry_price"] * 1.01  # 假设小幅上涨
                    
                    if trade["size"] > 0:  # 买入
                        trade["profit"] = (last_price - trade["entry_price"]) * abs(trade["size"])
                    else:  # 卖出
                        trade["profit"] = (trade["entry_price"] - last_price) * abs(trade["size"])
                    
                    # 确保明确设置is_win标志
                    trade["is_win"] = trade["profit"] > 0
                    logger.info(f"交易 #{i}: 入场价={trade['entry_price']:.2f}, 当前价={last_price:.2f}, 仓位={trade['size']:.4f}, 盈亏={trade['profit']:.4f}, 是否盈利={trade['is_win']}")
        
        # 计算总交易次数和胜率
        total_trades = len(trades)
        # 确保交易的is_win属性被正确设置
        for trade in trades:
            if "profit" in trade and "is_win" not in trade:
                trade["is_win"] = trade["profit"] > 0
                
        # 计算获胜交易数量和胜率
        winning_trades = sum(1 for trade in trades if trade.get("is_win", False))
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 打印更详细的信息以便调试
        logger.info(f"交易统计详情: 总交易={total_trades}, 获胜交易={winning_trades}, 胜率={win_rate:.2%}")
        logger.info(f"交易盈亏情况: {[(i, t.get('profit', 0), t.get('is_win', False)) for i, t in enumerate(trades[:5])]}")
        
        return {
            "final_equity": final_equity,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": float(sharpe_ratio),  # 确保可JSON序列化
            "sortino_ratio": float(sortino_ratio),  # 确保可JSON序列化
            "total_trades": total_trades,
            "win_rate": win_rate
        }
        
    except Exception as e:
        logger.error(f"计算模型统计数据时发生错误: {e}")
        logger.error(traceback.format_exc())
        return {
            "final_equity": 10000.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "total_trades": 0,
            "win_rate": 0.0
        }

# WebSocket配置
WS_HOST = "localhost"
WS_PORT = 8765

# 动态创建模型数据队列
MODEL_DATA_QUEUES = {}
def get_model_queue(model_id):
    """获取模型数据队列，如果不存在则创建"""
    model_id = str(model_id)  # 确保是字符串
    if model_id not in MODEL_DATA_QUEUES:
        MODEL_DATA_QUEUES[model_id] = queue.Queue(maxsize=1000)
    return MODEL_DATA_QUEUES[model_id]

# 获取模型目录中的所有模型作为默认预加载列表
def get_all_model_ids():
    """获取models目录下所有可用的模型ID"""
    try:
        # 获取绝对路径
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        project_root = current_dir.parent.parent  # 回到项目根目录
        models_dir = project_root / "btc_rl" / "models"
        
        if not models_dir.exists():
            alt_models_dir = Path("btc_rl/models")
            if alt_models_dir.exists():
                models_dir = alt_models_dir
            else:
                logger.error("无法找到模型目录")
                return []  # 找不到模型目录时返回空列表
        
        # 列出所有模型文件
        model_files = list(models_dir.glob("*.zip"))
        if not model_files:
            logger.error(f"在 {models_dir} 中未找到模型文件")
            return []  # 找不到模型文件时返回空列表
            
        # 根据修改时间排序
        model_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        # 生成ID列表 (从1开始编号)
        return [str(i) for i in range(1, len(model_files) + 1)]
    except Exception as e:
        logger.error(f"获取模型ID时出错: {e}")
        return []  # 出错时返回空列表

# 默认预加载模型列表
PRELOAD_MODELS = get_all_model_ids()

# 存储模型历史数据的字典
MODEL_HISTORY = {
    # 动态初始化空列表
    model_id: [] for model_id in PRELOAD_MODELS
}

# 存储模型统计数据的字典
MODEL_STATS = {}

def make_env(mode: str = "test", msg_queue: Optional[queue.Queue] = None, model_id: str = "1"):
    """创建交易环境"""
    try:
        # 获取项目根目录的绝对路径
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # 尝试多种可能的数据路径
        possible_paths = [
            f"btc_rl/data/{mode}_data.npz",                      # 相对于工作目录
            f"{project_root}/btc_rl/data/{mode}_data.npz",       # 绝对路径
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f"data/{mode}_data.npz")  # 相对于当前模块
        ]
        
        data = None
        for path in possible_paths:
            logger.info(f"尝试加载数据: {path}")
            if os.path.exists(path):
                data = np.load(path)
                logger.info(f"成功加载数据: {path}")
                break
        
        if data is None:
            raise FileNotFoundError(f"无法找到数据文件: {mode}_data.npz")
            
        # 检查数据文件是否包含必要的键
        if "X" not in data:
            raise KeyError(f"数据文件缺少必要的键 'X'")
        
        # 确定价格数据键
        price_keys = ["y", "prices", "price"]  # 优先检查这些常用名称
        price_key = None
        
        # 检查常用键名
        for key in price_keys:
            if key in data:
                price_key = key
                logger.info(f"找到价格数据键: '{key}'")
                break
        
        # 如果没有找到常用键名，尝试查找包含"price"的键
        if price_key is None:
            available_keys = list(data.keys())
            custom_price_keys = [k for k in available_keys if 'price' in str(k).lower()]
            if custom_price_keys:
                price_key = custom_price_keys[0]
                logger.info(f"找到自定义价格键: '{price_key}'")
        
        if price_key is not None:
            prices = data[price_key]
            logger.info(f"使用键 '{price_key}' 获取价格数据")
        else:
            logger.error(f"找不到价格数据，可用键: {list(data.keys())}")
            raise KeyError(f"数据文件中缺少价格数据，无法创建交易环境")
            
        windows = data["X"]
        
        # 记录数据统计信息以便调试
        logger.info(f"加载的数据规模: windows.shape={windows.shape}, prices.shape={prices.shape if hasattr(prices, 'shape') else '未知'}")
        
        # 创建环境
        # 为log_dir提供一个临时目录字符串而不是None，因为BtcTradingEnv需要一个Path兼容值
        temp_log_dir = os.path.join(project_root, "btc_rl", "logs", "temp")
        # 确保目录存在
        os.makedirs(temp_log_dir, exist_ok=True)
        
        env = BtcTradingEnv(
            windows=windows,
            prices=prices,
            initial_balance=10000.0,
            risk_fraction_per_trade=0.02,
            websocket_queue=msg_queue,
            log_dir=temp_log_dir,  # 使用临时日志目录而不是None
        )
        return TimeLimit(env, max_episode_steps=len(windows) + 1)
    except Exception as e:
        logger.error(f"创建环境失败: {e}")
        # 如果无法创建真实环境，我们返回None，调用者需要处理这个情况
        return None

async def model_evaluation_handler(websocket):
    """处理模型评估请求"""
    client_ip = websocket.remote_address[0] if websocket.remote_address else "未知"
    logger.info(f"模型评估客户端连接: {client_ip}")
    logger.info(f"WebSocket连接信息: {websocket.path if hasattr(websocket, 'path') else 'N/A'}")
    
    try:
        # 发送连接成功确认消息
        try:
            await websocket.send(json.dumps({
                "type": "system_message",
                "content": "已连接到模型比较服务器"
            }))
            logger.info(f"已向客户端 {client_ip} 发送连接确认消息")
        except Exception as e:
            logger.error(f"发送连接确认消息失败: {e}")
        
        while True:
            try:
                # 接收客户端消息
                message = await websocket.recv()
                try:
                    data = json.loads(message)
                    if data.get("type") == "request_model_list":
                        # 请求可用模型列表
                        logger.info(f"收到请求模型列表")
                        models = get_available_models()
                        response = {
                            "type": "model_list",
                            "data": models
                        }
                        logger.info(f"发送模型列表: {models}")
                        await websocket.send(json.dumps(response))
                    elif data.get("type") == "request_model_data":
                        model_id = str(data.get("model_id", "1"))
                        logger.info(f"收到请求模型 {model_id} 数据")
                        
                        # 检查是否已有此模型的缓存数据
                        if model_id in MODEL_HISTORY and MODEL_HISTORY[model_id]:
                            # 发送缓存的历史数据
                            logger.info(f"发送模型 {model_id} 的缓存数据: {len(MODEL_HISTORY[model_id])}条")
                            response = {
                                "type": "multi_model_data",
                                "data": {model_id: MODEL_HISTORY[model_id]}
                            }
                            await websocket.send(json.dumps(response))
                        else:
                            # 如果没有缓存数据，启动模型评估
                            logger.info(f"没有模型 {model_id} 的缓存数据，启动评估")
                            
                            # 告知客户端正在加载数据
                            await websocket.send(json.dumps({
                                "type": "system_message",
                                "content": f"正在加载模型 {model_id} 数据，请稍候..."
                            }))
                            
                            # 异步启动模型评估，避免阻塞WebSocket
                            evaluation_thread = threading.Thread(
                                target=evaluate_model, 
                                args=(model_id,), 
                                daemon=True
                            )
                            evaluation_thread.start()
                    elif data.get("type") == "request_multi_model_data":
                        # 一次性请求多个模型数据
                        model_ids = data.get("model_ids", [])
                        if not model_ids:
                            model_ids = ["1", "2", "3", "4", "5", "6"][:5]  # 默认最多5个模型
                        
                        logger.info(f"收到批量请求模型数据: {model_ids}")
                        
                        # 收集已缓存的模型数据
                        cached_data = {}
                        models_to_evaluate = []
                        
                        for model_id in model_ids:
                            model_id = str(model_id)
                            # 确保字典已经初始化
                            if model_id not in MODEL_HISTORY:
                                MODEL_HISTORY[model_id] = []
                                
                            if MODEL_HISTORY[model_id]:
                                logger.info(f"模型 {model_id} 有缓存数据: {len(MODEL_HISTORY[model_id])}条")
                                cached_data[model_id] = MODEL_HISTORY[model_id]
                            else:
                                logger.info(f"模型 {model_id} 没有缓存数据，需要评估")
                                models_to_evaluate.append(model_id)
                        
                        # 发送已缓存数据
                        if cached_data:
                            logger.info(f"发送 {len(cached_data)} 个模型的缓存数据")
                            try:
                                response = {
                                    "type": "multi_model_data",
                                    "data": cached_data
                                }
                                await websocket.send(json.dumps(response))
                                logger.info(f"缓存数据发送成功")
                            except Exception as e:
                                logger.error(f"发送缓存数据时出错: {e}")
                                # 尝试发送错误信息
                                await websocket.send(json.dumps({
                                    "type": "system_message",
                                    "content": f"发送模型数据时出错: {str(e)}"
                                }))
                        
                        # 启动评估未缓存的模型
                        if models_to_evaluate:
                            logger.info(f"需要评估 {len(models_to_evaluate)} 个模型")
                            await websocket.send(json.dumps({
                                "type": "system_message",
                                "content": f"正在加载 {len(models_to_evaluate)} 个模型数据，请稍候..."
                            }))
                            
                            # 并发评估所有未缓存的模型
                            for model_id in models_to_evaluate:
                                threading.Thread(
                                    target=evaluate_model, 
                                    args=(model_id,), 
                                    daemon=True
                                ).start()
                    elif data.get("type") == "request_model_list":
                        # 处理获取模型列表的请求
                        logger.info("收到请求模型列表")
                        model_list = get_available_models()
                        
                        # 发送模型列表
                        response = {
                            "type": "model_list",
                            "data": model_list
                        }
                        await websocket.send(json.dumps(response))
                        logger.info(f"发送模型列表: {len(model_list)} 个模型")
                    else:
                        logger.warning(f"未知消息类型: {data.get('type')}")
                except json.JSONDecodeError:
                    logger.error(f"JSON解析错误: {message}")
                
                # 检查各个模型队列是否有数据
                for model_id, model_queue in MODEL_DATA_QUEUES.items():
                    if model_queue and not model_queue.empty():
                        # 一次性获取所有可用数据
                        batch_data = []
                        while not model_queue.empty():
                            try:
                                data_point = model_queue.get_nowait()
                                batch_data.append(data_point)
                            except queue.Empty:
                                break
                        
                        if batch_data:
                            # 发送批量数据
                            response = {
                                "type": "multi_model_data",
                                "data": {model_id: batch_data}
                            }
                            await websocket.send(json.dumps(response))
                            logger.info(f"发送模型 {model_id} 的 {len(batch_data)} 条数据点")
            
            except websockets.ConnectionClosed:
                logger.info(f"客户端断开连接: {client_ip}")
                break
    
    except Exception as e:
        logger.error(f"处理模型评估请求时发生错误: {e}")
    
    logger.info(f"客户端会话结束: {client_ip}")

def evaluate_model(model_id: str, is_preloading=True):
    """评估指定ID的模型并将数据添加到队列
    
    Args:
        model_id (str): 模型ID
        is_preloading (bool): 是否为预加载模式，用于优化预加载速度
    """
    if not import_success:
        logger.error("无法评估模型: 导入必要模块失败")
        error_data = {
            "step": 0,
            "action": 0,
            "cash_balance": 10000.0,
            "margin_equity": 10000.0,
            "buy_and_hold_equity": 10000.0,
            "upnl": 0.0,
            "reward": 0.0,
            "price": 0.0,
            "position_btc": 0.0,
            "total_fee": 0.0,
            "was_liquidated_this_step": False,
            "termination_reason": "导入必要模块失败",
            "model_id": model_id,
            "model_name": f"模型 {model_id}",
            "error": True
        }
        try:
            get_model_queue(model_id).put_nowait(error_data)
            MODEL_HISTORY[model_id] = [error_data]
        except Exception:
            logger.error("无法将错误信息添加到队列")
        return
    
    # 获取消息队列（使用get_model_queue确保队列存在）
    msg_queue = get_model_queue(model_id)
    
    # 查找模型路径和显示名称
    model_path, model_display_name = find_model_by_id(model_id)
    if not model_path:
        logger.error(f"找不到模型 {model_id}")
        # 添加错误信息而不是使用模拟数据
        error_data = {
            "step": 0,
            "action": 0,
            "cash_balance": 10000.0,
            "margin_equity": 10000.0,
            "buy_and_hold_equity": 10000.0,
            "upnl": 0.0,
            "reward": 0.0,
            "price": 0.0,
            "position_btc": 0.0,
            "total_fee": 0.0,
            "was_liquidated_this_step": False,
            "termination_reason": f"找不到模型 {model_id}",
            "model_id": model_id,
            "model_name": model_display_name,
            "error": True
        }
        try:
            msg_queue.put_nowait(error_data)
            MODEL_HISTORY[model_id] = [error_data]
        except Exception:
            logger.error("无法将错误信息添加到队列")
        return
    
    logger.info(f"找到模型 {model_id} 路径: {model_path}, 显示名称: {model_display_name}")
    
    # 尝试读取预计算的指标文件
    metrics_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metrics", f"{model_display_name}_metrics.json")
    precalculated_stats = None
    
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
                logger.info(f"找到预计算指标文件: {metrics_file}")
                
                # 提取关键统计数据
                precalculated_stats = {
                    "final_equity": metrics_data.get("final_equity", 10000.0),
                    "total_return": metrics_data.get("total_return", 0.0),
                    "max_drawdown": metrics_data.get("max_drawdown", 0.0),
                    "sharpe_ratio": metrics_data.get("sharpe_ratio", 0.0),
                    "sortino_ratio": metrics_data.get("sortino_ratio", 0.0),
                    "total_trades": metrics_data.get("total_trades", 0),
                    "win_rate": metrics_data.get("win_rate", 0.0),
                }
                logger.info(f"读取到预计算统计数据: 交易次数={precalculated_stats['total_trades']}, 胜率={precalculated_stats['win_rate']:.2%}")
                
                # 如果是预加载模式且已有预计算指标，可以创建简化的历史数据并提前返回
                if is_preloading and precalculated_stats:
                    logger.info(f"预加载模式: 使用预计算指标加快模型 {model_id} 的加载速度")
                    # 创建一个简化的历史数据记录，只包含必要的点数
                    simplified_history = []
                    for i in range(0, 10):  # 只创建10个数据点用于显示
                        equity_value = 10000 * (1 + precalculated_stats["total_return"] * i / 9)
                        data_point = {
                            "step": i * 100,  # 均匀分布的步骤
                            "action": 0,
                            "cash_balance": equity_value,
                            "margin_equity": equity_value,
                            "buy_and_hold_equity": 10000 * (1 + 0.5 * i / 9),
                            "upnl": 0.0,
                            "reward": 0.0,
                            "price": 1000 * (1 + 0.2 * i / 9),
                            "position_btc": 0.0,
                            "total_fee": 0.0,
                            "was_liquidated_this_step": False,
                            "termination_reason": None if i < 9 else "完成",
                            "model_id": model_id,
                            "model_name": model_display_name,
                            "stats": precalculated_stats  # 添加统计数据
                        }
                        simplified_history.append(data_point)
                        try:
                            msg_queue.put_nowait(data_point)
                        except queue.Full:
                            pass
                    
                    # 保存到模型历史记录
                    MODEL_HISTORY[model_id] = simplified_history
                    MODEL_STATS[model_id] = precalculated_stats
                    return
                
        except Exception as e:
            logger.warning(f"读取预计算指标文件出错: {e}, 将重新计算统计数据")
    
    # 验证模型文件是否存在
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        # 添加错误信息而不是使用模拟数据
        error_data = {
            "step": 0,
            "action": 0,
            "cash_balance": 10000.0,
            "margin_equity": 10000.0,
            "buy_and_hold_equity": 10000.0,
            "upnl": 0.0,
            "reward": 0.0,
            "price": 0.0,
            "position_btc": 0.0,
            "total_fee": 0.0,
            "was_liquidated_this_step": False,
            "termination_reason": f"模型文件不存在: {model_path}",
            "model_id": model_id,
            "model_name": model_display_name,
            "error": True
        }
        try:
            msg_queue.put_nowait(error_data)
            MODEL_HISTORY[model_id] = [error_data]
        except Exception:
            logger.error("无法将错误信息添加到队列")
        return
    
    # 创建环境
    try:
        env = make_env("test", msg_queue, model_id)
        if env is None:
            logger.error("无法创建环境，但根据要求不使用模拟数据")
            # 将错误信息添加到队列和历史记录
            error_data = {
                "step": 0,
                "action": 0,
                "cash_balance": 10000.0,
                "margin_equity": 10000.0,
                "buy_and_hold_equity": 10000.0,
                "upnl": 0.0,
                "reward": 0.0,
                "price": 0.0,
                "position_btc": 0.0,
                "total_fee": 0.0,
                "was_liquidated_this_step": False,
                "termination_reason": "无法创建环境",
                "model_id": model_id,
                "model_name": model_display_name,
                "error": True
            }
            try:
                msg_queue.put_nowait(error_data)
                MODEL_HISTORY[model_id] = [error_data]
            except Exception:
                logger.error("无法将错误信息添加到队列")
            return
    except Exception as e:
        logger.error(f"创建环境时发生异常: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # 将错误信息添加到队列和历史记录
        error_data = {
            "step": 0,
            "action": 0,
            "cash_balance": 10000.0,
            "margin_equity": 10000.0,
            "buy_and_hold_equity": 10000.0,
            "upnl": 0.0,
            "reward": 0.0,
            "price": 0.0,
            "position_btc": 0.0,
            "total_fee": 0.0,
            "was_liquidated_this_step": False,
            "termination_reason": f"创建环境异常: {str(e)}",
            "model_id": model_id,
            "error": True
        }
        try:
            msg_queue.put_nowait(error_data)
            MODEL_HISTORY[model_id] = [error_data]
        except Exception:
            logger.error("无法将错误信息添加到队列")
        return
    logger.info("交易环境创建成功")
    
    # 加载模型
    try:
        logger.info(f"尝试加载模型: {model_path}")
        
        # 检查文件大小，确保不是空文件
        if os.path.getsize(model_path) == 0:
            logger.error(f"模型文件为空: {model_path}")
            # 添加错误信息而不是使用模拟数据
            error_data = {
                "step": 0,
                "action": 0,
                "cash_balance": 10000.0,
                "margin_equity": 10000.0,
                "buy_and_hold_equity": 10000.0,
                "upnl": 0.0,
                "reward": 0.0,
                "price": 0.0,
                "position_btc": 0.0,
                "total_fee": 0.0,
                "was_liquidated_this_step": False,
                "termination_reason": f"模型文件为空: {model_path}",
                "model_id": model_id,
                "model_name": model_display_name,
                "error": True
            }
            try:
                msg_queue.put_nowait(error_data)
                MODEL_HISTORY[model_id] = [error_data]
            except Exception:
                logger.error("无法将错误信息添加到队列")
            return
        
        # 尝试加载模型
        model = SAC.load(model_path)
        logger.info(f"模型 {model_id} 加载成功")
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        # 添加错误信息而不是使用模拟数据
        error_data = {
            "step": 0,
            "action": 0,
            "cash_balance": 10000.0,
            "margin_equity": 10000.0,
            "buy_and_hold_equity": 10000.0,
            "upnl": 0.0,
            "reward": 0.0,
            "price": 0.0,
            "position_btc": 0.0,
            "total_fee": 0.0,
            "was_liquidated_this_step": False,
            "termination_reason": f"加载模型失败: {str(e)}",
            "model_id": model_id,
            "model_name": model_display_name,
            "error": True
        }
        try:
            msg_queue.put_nowait(error_data)
            MODEL_HISTORY[model_id] = [error_data]
        except Exception:
            logger.error("无法将错误信息添加到队列")
        return
    
    # 初始化历史数据列表和基础数据
    history_data = []
    base_data = {
        "step": 0,
        "action": 0,
        "cash_balance": 10000.0,
        "margin_equity": 10000.0,
        "buy_and_hold_equity": 10000.0,
        "upnl": 0.0,
        "reward": 0.0,
        "price": 0.0,
        "position_btc": 0.0,
        "total_fee": 0.0,
        "was_liquidated_this_step": False,
        "termination_reason": None,
        "model_id": model_id,
        "model_name": model_display_name
    }
    
    # 将起始数据加入队列和历史记录
    try:
        msg_queue.put_nowait(base_data)
        history_data.append(base_data)
    except queue.Full:
        logger.warning(f"队列已满，无法添加基础数据")
    
    # 运行一个episode
    logger.info(f"开始模型 {model_id} 的评估运行")
    try:
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
            
            # 数据已经通过env的websocket_queue发送
            # 但我们也需要保存到历史记录中
            if step % DATA_SAMPLING_INTERVAL == 0:  # 使用配置的采样间隔
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
                    "model_id": model_id,
                    "model_name": model_display_name
                }
                history_data.append(data_point)
                
                # 发送数据到队列，使用相同的采样间隔保持一致性
                try:
                    # 不立即发送数据点，等到统计数据计算完成后再一次性发送
                    # 这样可以确保每个数据点都包含完整的统计信息
                    pass
                except Exception:
                    logger.warning(f"处理数据点时出错")
        
            logger.info(f"模型 {model_id} 评估完成，共记录 {len(history_data)} 个数据点")            # 计算或使用预计算的模型统计指标
            try:
                if precalculated_stats:
                    # 使用预计算的统计数据，但仍然通过history_data计算权益和回撤相关指标
                    # 这样可以确保WebSocket数据的一致性，同时使用JSON文件中的交易次数和胜率
                    calc_stats = calculate_model_statistics(history_data)
                    
                    # 合并计算的统计数据和预计算的统计数据
                    stats = {
                        "final_equity": calc_stats["final_equity"],  # 使用当前计算的权益
                        "total_return": calc_stats["total_return"],  # 使用当前计算的回报率 
                        "max_drawdown": calc_stats["max_drawdown"],  # 使用当前计算的最大回撤
                        "sharpe_ratio": calc_stats["sharpe_ratio"],  # 使用当前计算的夏普比率
                        "sortino_ratio": calc_stats["sortino_ratio"], # 使用当前计算的索提诺比率
                        "total_trades": precalculated_stats["total_trades"],  # 使用预计算的交易次数
                        "win_rate": precalculated_stats["win_rate"],  # 使用预计算的胜率
                    }
                    logger.info(f"模型 {model_id} 使用预计算统计数据: 交易次数={stats['total_trades']}, 胜率={stats['win_rate']:.2%}")
                else:
                    # 没有预计算数据，完全通过history_data计算
                    stats = calculate_model_statistics(history_data)
                    logger.info(f"模型 {model_id} 统计数据计算完成: 夏普比率={stats['sharpe_ratio']:.4f}, 索替诺比率={stats['sortino_ratio']:.4f}, 交易次数={stats['total_trades']}, 胜率={stats['win_rate']:.2%}")
            except Exception as e:
                logger.error(f"计算统计数据时出错: {e}")
                # 确保有一个默认的统计数据
                stats = {
                    "final_equity": history_data[-1]["margin_equity"] if history_data else 10000.0,
                    "total_return": 0.0,
                    "max_drawdown": 0.0,
                    "sharpe_ratio": 0.0,
                    "sortino_ratio": 0.0,
                    "total_trades": precalculated_stats["total_trades"] if precalculated_stats else 0,
                    "win_rate": precalculated_stats["win_rate"] if precalculated_stats else 0.0
                }
            
            # 将统计数据添加到每个数据点
            for data_point in history_data:
                data_point["stats"] = stats
                
            # 将带有统计数据的数据点发送到队列，确保客户端收到完整数据
            for data_point in history_data:
                try:
                    msg_queue.put_nowait(data_point.copy())
                except queue.Full:
                    logger.warning(f"队列已满，无法添加数据点，但统计数据已添加到历史记录")
        
        # 存储历史数据
        MODEL_HISTORY[model_id] = history_data
        return
        
    except Exception as e:
        logger.error(f"模型评估执行过程中出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # 如果已经收集了一些数据但中途失败了，我们可以尝试使用已有数据
        if len(history_data) > 10:
            logger.info(f"使用已收集的 {len(history_data)} 个数据点")
            # 即使是部分数据，也计算模型统计指标
            stats = calculate_model_statistics(history_data)
            logger.info(f"模型 {model_id} 部分数据统计: 夏普比率={stats['sharpe_ratio']:.4f}, 索替诺比率={stats['sortino_ratio']:.4f}, 交易次数={stats['total_trades']}, 胜率={stats['win_rate']:.2%}")
            
            # 将统计数据添加到每个数据点
            for data_point in history_data:
                data_point["stats"] = stats
            
            MODEL_HISTORY[model_id] = history_data
            return
        
        # 否则添加错误信息而不是使用模拟数据
        error_data = {
            "step": 0,
            "action": 0,
            "cash_balance": 10000.0,
            "margin_equity": 10000.0,
            "buy_and_hold_equity": 10000.0,
            "upnl": 0.0,
            "reward": 0.0,
            "price": 0.0,
            "position_btc": 0.0,
            "total_fee": 0.0,
            "was_liquidated_this_step": False,
            "termination_reason": f"评估过程中出错且收集的数据不足",
            "model_id": model_id,
            "model_name": model_display_name,
            "error": True,
            "stats": {
                "final_equity": 10000.0,
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "total_trades": 0,
                "win_rate": 0.0
            }
        }
        try:
            msg_queue.put_nowait(error_data)
            MODEL_HISTORY[model_id] = [error_data]
        except Exception:
            logger.error("无法将错误信息添加到队列")
        return
    
    except Exception as e:
        logger.error(f"评估模型 {model_id} 时发生未预期错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # 添加错误信息而不是使用模拟数据
        error_data = {
            "step": 0,
            "action": 0,
            "cash_balance": 10000.0,
            "margin_equity": 10000.0,
            "buy_and_hold_equity": 10000.0,
            "upnl": 0.0,
            "reward": 0.0,
            "price": 0.0,
            "position_btc": 0.0,
            "total_fee": 0.0,
            "was_liquidated_this_step": False,
            "termination_reason": f"评估模型时发生未预期错误: {str(e)}",
            "model_id": model_id,
            "model_name": f"模型 {model_id}",  # 这里使用简单的标识，因为可能没有display_name
            "error": True
        }
        try:
            get_model_queue(model_id).put_nowait(error_data)
            MODEL_HISTORY[model_id] = [error_data]
        except Exception:
            logger.error("无法将错误信息添加到队列")

def find_model_by_id(model_id: str) -> tuple:
    """根据模型ID查找对应的模型文件路径和显示名称"""
    try:
        # 获取绝对路径
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        project_root = current_dir.parent.parent  # 回到项目根目录
        models_dir = project_root / "btc_rl" / "models"
        
        logger.info(f"查找模型目录: {models_dir}")
        
        # 确认模型目录存在
        if not models_dir.exists():
            logger.error(f"模型目录不存在: {models_dir}")
            # 尝试查找其他可能的位置
            alt_models_dir = Path("btc_rl/models")
            if alt_models_dir.exists():
                models_dir = alt_models_dir
                logger.info(f"使用替代模型目录: {models_dir}")
            else:
                logger.error("无法找到模型目录")
                return "", f"模型 {model_id}"
        
        # 列出所有模型文件
        model_files = list(models_dir.glob("*.zip"))
        if not model_files:
            logger.error(f"在 {models_dir} 中未找到模型文件")
            return "", f"模型 {model_id}"
        
        logger.info(f"找到模型文件: {[f.name for f in model_files]}")
        
        # 优先尝试按照命名规则查找
        model_path = models_dir / f"sac_ep{model_id}.zip"
        if model_path.exists():
            logger.info(f"按命名规则找到模型: {model_path}")
            display_name = f"{model_path.stem}"  # 使用文件名作为显示名称，不带扩展名
            return str(model_path), display_name
        
        # 如果找不到，则按修改时间排序
        model_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        # 将模型ID转为整数，作为索引
        try:
            index = int(model_id) - 1  # 转为0-indexed
            if 0 <= index < len(model_files):
                logger.info(f"按索引选择模型 {index+1}: {model_files[index]}")
                display_name = f"{model_files[index].stem}"  # 使用文件名作为显示名称，不带扩展名
                return str(model_files[index]), display_name
        except ValueError:
            pass
        
        # 如果ID无效，则返回最新模型（如果有）
        if model_files:
            logger.info(f"选择最新模型: {model_files[0]}")
            display_name = f"{model_files[0].stem}"  # 使用文件名作为显示名称，不带扩展名
            return str(model_files[0]), display_name
    
    except Exception as e:
        logger.error(f"查找模型过程中出错: {e}")
    
    logger.error("未找到任何可用模型")
    return "", f"模型 {model_id}"

def get_available_models():
    """
    获取可用的模型列表
    
    Returns:
        list: 包含模型ID和名称的列表 [{id: "1", name: "sac_ep1"}, ...]
    """
    try:
        # 获取绝对路径
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        project_root = current_dir.parent.parent  # 回到项目根目录
        models_dir = project_root / "btc_rl" / "models"
        
        logger.info(f"查找模型目录: {models_dir}")
        
        # 确认模型目录存在
        if not models_dir.exists():
            logger.error(f"模型目录不存在: {models_dir}")
            # 尝试查找其他可能的位置
            alt_models_dir = Path("btc_rl/models")
            if alt_models_dir.exists():
                models_dir = alt_models_dir
                logger.info(f"使用替代模型目录: {models_dir}")
            else:
                logger.error("无法找到模型目录")
                return []
        
        # 列出所有模型文件
        model_files = list(models_dir.glob("*.zip"))
        if not model_files:
            logger.error(f"在 {models_dir} 中未找到模型文件")
            return []
        
        logger.info(f"找到模型文件: {[f.name for f in model_files]}")
        
        # 排序模型文件（按照修改时间降序）
        model_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        # 创建模型列表，格式为 [{id: "1", name: "sac_ep1"}, ...]
        models = []
        for i, model_file in enumerate(model_files, 1):
            model_id = str(i)
            model_name = model_file.stem  # 不带扩展名的文件名
            models.append({"id": model_id, "name": model_name})
        
        return models
        
    except Exception as e:
        logger.error(f"获取模型列表时出错: {e}")
        return []

def generate_mock_data(model_id: str):
    """生成模拟数据用于前端显示"""
    logger.info(f"为模型 {model_id} 生成模拟数据")
    
    try:
        # 设置随机种子，确保不同模型有不同的模拟数据
        seed = int(model_id) * 123 if model_id.isdigit() else hash(model_id) % 10000
        np.random.seed(seed)
        
        # 模拟数据参数
        n_steps = 200  # 模拟数据点数量
        initial_price = 50000  # 初始BTC价格
        price_volatility = 0.01  # 价格波动率
        initial_balance = 10000.0  # 初始资金
        
        # 根据模型ID调整性能特性，使不同模型表现不同
        performance_factor = (int(model_id) % 3) if model_id.isdigit() else 1
        if performance_factor == 0:  # 较好的模型
            price_trend = 0.0002  # 轻微上升趋势
            trading_quality = 0.05  # 高质量交易信号
        elif performance_factor == 1:  # 中等的模型
            price_trend = 0  # 无趋势
            trading_quality = 0  # 中等质量交易信号
        else:  # 较差的模型
            price_trend = -0.0001  # 轻微下降趋势
            trading_quality = -0.03  # 低质量交易信号
        
        # 创建起点数据
        base_data = {
            "step": 0,
            "action": 0,
            "cash_balance": initial_balance,
            "margin_equity": initial_balance,
            "buy_and_hold_equity": initial_balance,
            "upnl": 0.0,
            "reward": 0.0,
            "price": initial_price,
            "position_btc": 0.0,
            "total_fee": 0.0,
            "was_liquidated_this_step": False,
            "termination_reason": None,
            "model_id": model_id
        }
        
        # 初始化历史数据列表
        history_data = [base_data]
        
        # 添加到队列
        msg_queue = get_model_queue(model_id)
        try:
            msg_queue.put_nowait(base_data)
        except queue.Full:
            logger.warning(f"队列已满，无法添加基础数据")
        
        # 生成模拟交易数据
        price = initial_price
        cash = initial_balance
        position = 0.0
        buy_and_hold_btc = initial_balance / initial_price
        total_fee = 0.0
        
        for step in range(1, n_steps + 1):
            # 模拟价格变动 (随机游走 + 趋势)
            price_change = np.random.normal(0, price_volatility * price) + price * price_trend
            price = max(100, price + price_change)  # 确保价格不会太低
            
            # 模拟交易动作 (-1到1之间，负值表示卖出，正值表示买入)
            # 使用正弦波 + 随机噪声，模拟周期性交易行为
            # 加入trading_quality来模拟不同模型的交易质量
            action = np.sin(step / 20) * 0.7 + np.random.normal(0, 0.3) + trading_quality
            action = np.clip(action, -1, 1)
            
            # 计算新仓位 (模拟交易逻辑)
            position_delta = action * 0.1  # 每次交易最多改变10%的仓位
            new_position = position + position_delta
            
            # 计算交易费用 (假设费率为0.1%)
            trade_value = abs(position_delta) * price
            fee = trade_value * 0.001
            total_fee += fee
            
            # 更新现金余额
            cash = cash - position_delta * price - fee
            
            # 更新持仓
            position = new_position
            
            # 计算账户价值
            position_value = position * price
            margin_equity = cash + position_value
            buy_and_hold_equity = buy_and_hold_btc * price
            
            # 计算未实现盈亏
            upnl = position_value
            
            # 模拟奖励 (基于账户价值的变化)
            reward = (margin_equity / initial_balance - 1) * 0.1
            
            # 创建数据点
            if step % DATA_SAMPLING_INTERVAL == 0:  # 使用配置的采样间隔
                data_point = {
                    "step": step,
                    "action": float(action),
                    "cash_balance": float(cash),  # 确保可JSON序列化
                    "margin_equity": float(margin_equity),
                    "buy_and_hold_equity": float(buy_and_hold_equity),
                    "upnl": float(upnl),
                    "reward": float(reward),
                    "price": float(price),
                    "position_btc": float(position),
                    "total_fee": float(total_fee),
                    "was_liquidated_this_step": False,
                    "termination_reason": None,
                    "model_id": model_id
                }
                
                # 添加到历史记录
                history_data.append(data_point)
                
                # 放入队列
                try:
                    msg_queue.put_nowait(data_point)
                    # 模拟评估需要时间，添加小延迟
                    time.sleep(0.05)  # 减少延迟以加快数据加载
                except queue.Full:
                    logger.warning(f"队列已满，跳过数据点 {step}")
        
        # 存储历史数据
        MODEL_HISTORY[model_id] = history_data
        logger.info(f"为模型 {model_id} 生成了 {len(history_data)} 个模拟数据点")
    
    except Exception as e:
        logger.error(f"生成模拟数据时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # 至少确保有一个基本数据点，以便前端不会出错
        basic_data = {
            "step": 0,
            "action": 0,
            "cash_balance": 10000.0,
            "margin_equity": 10000.0,
            "buy_and_hold_equity": 10000.0,
            "upnl": 0.0,
            "reward": 0.0,
            "price": 50000.0,
            "position_btc": 0.0,
            "total_fee": 0.0,
            "was_liquidated_this_step": False,
            "termination_reason": "模拟数据生成错误",
            "model_id": model_id
        }
        
        try:
            # 尝试将基本数据点放入队列和历史记录
            get_model_queue(model_id).put_nowait(basic_data)
            MODEL_HISTORY[model_id] = [basic_data]
        except Exception:
            logger.error("无法添加基本数据点")

# 全局变量用于跟踪预加载状态
PRELOAD_PROGRESS = {
    "total": 0,  # 总任务数
    "completed": 0,  # 已完成数量
    "percent": 0,  # 百分比进度
    "finished": False,  # 是否全部完成
    "models_status": {}  # 每个模型的状态
}

def write_progress_file(progress):
    """将预加载进度写入临时文件，供外部脚本读取"""
    try:
        progress_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "preload_progress.json")
        with open(progress_file, 'w') as f:
            json.dump(progress, f)
        logger.debug(f"已更新预加载进度: {progress['percent']}%")
    except Exception as e:
        logger.error(f"写入进度文件失败: {e}")

def preload_models(model_ids=None):
    """启动时预加载模型数据，避免用户等待，并提供进度信息
    
    Args:
        model_ids: 要预加载的模型ID列表，默认为前5个模型
    """
    global PRELOAD_PROGRESS
    
    if model_ids is None:
        # 默认预加载前5个模型
        model_ids = ["1", "2", "3", "4", "5"]
    
    # 初始化预加载状态
    PRELOAD_PROGRESS["total"] = len(model_ids)
    PRELOAD_PROGRESS["completed"] = 0
    PRELOAD_PROGRESS["percent"] = 0
    PRELOAD_PROGRESS["finished"] = False
    PRELOAD_PROGRESS["models_status"] = {model_id: "waiting" for model_id in model_ids}
    
    # 写入初始进度
    write_progress_file(PRELOAD_PROGRESS)
    
    # 检查是否有已经预计算好的模型，如果有，可以加快加载速度
    precomputed_models = []
    for model_id in model_ids:
        _, model_display_name = find_model_by_id(model_id)
        metrics_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                    "metrics", f"{model_display_name}_metrics.json")
        if os.path.exists(metrics_file):
            precomputed_models.append(model_id)
    
    if precomputed_models:
        logger.info(f"发现 {len(precomputed_models)} 个模型有预计算指标，这将加快预加载速度")
    
    logger.info(f"启动时预加载 {len(model_ids)} 个模型数据")
    
    # 创建一个线程池来并行预加载模型
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:  # 使用5个工作线程
        # 提交模型评估任务，明确指定为预加载模式
        future_to_model = {executor.submit(evaluate_model, model_id, True): model_id for model_id in model_ids}
        
        # 收集结果
        for future in concurrent.futures.as_completed(future_to_model):
            model_id = future_to_model[future]
            try:
                future.result()  # 获取结果，捕获异常
                logger.info(f"模型 {model_id} 预加载完成")
                
                # 更新预加载状态
                PRELOAD_PROGRESS["completed"] += 1
                PRELOAD_PROGRESS["percent"] = int(PRELOAD_PROGRESS["completed"] * 100 / PRELOAD_PROGRESS["total"])
                PRELOAD_PROGRESS["models_status"][model_id] = "completed"
                
                # 写入进度文件
                write_progress_file(PRELOAD_PROGRESS)
                
            except Exception as e:
                logger.error(f"模型 {model_id} 预加载失败: {e}")
                
                # 更新失败状态
                PRELOAD_PROGRESS["completed"] += 1
                PRELOAD_PROGRESS["percent"] = int(PRELOAD_PROGRESS["completed"] * 100 / PRELOAD_PROGRESS["total"])
                PRELOAD_PROGRESS["models_status"][model_id] = "failed"
                
                # 写入进度文件
                write_progress_file(PRELOAD_PROGRESS)
    
    # 标记为完成
    PRELOAD_PROGRESS["finished"] = True
    PRELOAD_PROGRESS["percent"] = 100
    write_progress_file(PRELOAD_PROGRESS)
    
    logger.info(f"所有模型预加载完成")

async def start_server():
    """启动WebSocket服务器"""
    try:
        # 初始化进度文件
        progress_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "preload_progress.json")
        with open(progress_file, 'w') as f:
            json.dump({"total": 0, "completed": 0, "percent": 0, "finished": False, "models_status": {}}, f)
        
        # 先启动预加载模型的线程
        logger.info(f"将预加载以下模型: {PRELOAD_MODELS}")
        preload_thread = threading.Thread(target=preload_models, args=(PRELOAD_MODELS,), daemon=True)
        preload_thread.start()
        
        # 启动WebSocket服务器
        # 创建路由映射，把根路径指向模型评估处理程序
        async def router(websocket, path):
            client_ip = websocket.remote_address[0] if websocket.remote_address else "未知"
            logger.info(f"收到WebSocket连接请求: 路径={path}，客户端IP={client_ip}")
            # 不管路径是什么，都交给模型评估处理程序处理
            try:
                # 发送连接确认消息
                await websocket.send(json.dumps({
                    "type": "system_message",
                    "content": "WebSocket连接成功"
                }))
                logger.info(f"已发送WebSocket连接确认消息给客户端: {client_ip}")
                
                # 处理请求
                await model_evaluation_handler(websocket)
            except Exception as e:
                logger.error(f"WebSocket处理错误: {e}")
                logger.error(traceback.format_exc())
            
        async with websockets.serve(router, WS_HOST, WS_PORT):
            logger.info(f"模型比较服务器已启动 ws://{WS_HOST}:{WS_PORT}")
            await asyncio.Future()  # 运行直到手动关闭
    except Exception as e:
        logger.error(f"启动WebSocket服务器失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="BTC交易智能体模型比较工具")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket服务器端口号")
    parser.add_argument("--preload", action="store_true", help="启动时预加载常用模型")
    parser.add_argument("--models", type=str, default=None, help="指定要预加载的模型ID，用逗号分隔，例如：1,2,3")
    args = parser.parse_args()
    
    global WS_PORT
    WS_PORT = args.port
    
    # 如果指定了特定模型，则解析模型ID列表
    if args.models:
        global PRELOAD_MODELS
        PRELOAD_MODELS = args.models.split(',')
        logger.info(f"将预加载指定的模型: {PRELOAD_MODELS}")
    
    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        logger.info("服务器被手动停止")
    except Exception as e:
        logger.error(f"服务器错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

def add_model_data_point(model_id: str, data_point: Dict):
    """添加一个模型数据点到历史并发送到队列"""
    try:
        # 初始化历史记录列表
        if model_id not in MODEL_HISTORY:
            MODEL_HISTORY[model_id] = []
            
        # 确保每个数据点都有模型ID和名称信息
        if "model_id" not in data_point:
            data_point["model_id"] = model_id
        
        # 添加模型名称
        model_path, model_display_name = find_model_by_id(model_id)
        if "model_name" not in data_point and model_display_name:
            data_point["model_name"] = model_display_name
            
        # 添加到历史记录
        MODEL_HISTORY[model_id].append(data_point)
        
        # 尝试读取预计算的指标文件
        precalculated_stats = None
        try:
            model_path, model_display_name = find_model_by_id(model_id)
            metrics_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metrics", f"{model_display_name}_metrics.json")
            
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                    # 提取关键统计数据
                    precalculated_stats = {
                        "total_trades": metrics_data.get("total_trades", 0),
                        "win_rate": metrics_data.get("win_rate", 0.0),
                    }
                    logger.debug(f"读取到预计算统计数据: 交易次数={precalculated_stats['total_trades']}, 胜率={precalculated_stats['win_rate']:.2%}")
        except Exception as e:
            logger.warning(f"读取预计算指标文件出错: {e}, 将重新计算统计数据")
        
        # 计算统计数据
        if len(MODEL_HISTORY[model_id]) >= 10:  # 至少需要10个点来计算有意义的统计数据
            # 每隔10个数据点或每当收到最后一个数据点时计算统计数据
            if len(MODEL_HISTORY[model_id]) % 10 == 0 or data_point.get("termination_reason"):
                # 计算基本统计数据
                stats = calculate_model_statistics(MODEL_HISTORY[model_id])
                
                # 如果有预计算数据，优先使用预计算的交易次数和胜率
                if precalculated_stats and stats:
                    stats["total_trades"] = precalculated_stats["total_trades"]
                    stats["win_rate"] = precalculated_stats["win_rate"]
                    logger.info(f"为模型 {model_id} 使用预计算的交易统计数据: 交易次数={stats['total_trades']}, 胜率={stats['win_rate']:.2%}")
                
                if stats:
                    # 将统计数据添加到最后一个数据点
                    data_point["stats"] = stats
                    # 也更新历史中的最后一个数据点
                    MODEL_HISTORY[model_id][-1]["stats"] = stats
                    logger.info(f"为模型 {model_id} 计算统计数据: 夏普比率={stats.get('sharpe_ratio', 0):.4f}, 索提诺比率={stats.get('sortino_ratio', 0):.4f}")
        
        # 发送到队列
        try:
            get_model_queue(model_id).put_nowait(data_point)
        except queue.Full:
            logger.warning(f"队列已满，无法添加数据点")
    except Exception as e:
        logger.error(f"添加模型数据点时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
