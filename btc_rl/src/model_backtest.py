#!/usr/bin/env python3
"""
模型回测工具，专门用于已训练模型的多环境、多时段回测分析
"""

import os
import sys
import json
import numpy as np
import logging
import datetime
import queue
import traceback
from tabulate import tabulate
from typing import Dict, List, Optional, Tuple

# 导入模型训练相关模块
from btc_rl.src.config import get_config
from btc_rl.src.train_sac import evaluate_model_with_metrics
from btc_rl.src.model_comparison import get_best_model_by_golden_rule, calculate_model_statistics, DATA_SAMPLING_INTERVAL

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("model_backtest")

def format_percent(value):
    """格式化百分比显示"""
    return f"{value * 100:.2f}%" if isinstance(value, (int, float)) else value

def format_currency(value):
    """格式化货币显示"""
    return f"${value:.2f}" if isinstance(value, (int, float)) else value

def get_best_model():
    """获取黄金法则评估的最佳模型"""
    try:
        best_model_info = get_best_model_by_golden_rule()
        if not best_model_info:
            logger.error("无法获取最佳模型信息")
            return None
        
        logger.info(f"黄金法则评估选出的最佳模型是: {best_model_info['model_name']} (评分: {best_model_info['golden_rule_score']:.4f})")
        return best_model_info['model_path']
    except Exception as e:
        logger.error(f"获取最佳模型时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def backtest_model_in_environment(model_path: str, market_type: str = "默认环境"):
    """
    在指定环境下回测模型
    
    Args:
        model_path (str): 模型路径
        market_type (str): 市场环境类型描述
        
    Returns:
        dict: 回测结果
    """
    logger.info(f"开始在{market_type}环境下回测模型: {os.path.basename(model_path)}")
    
    # 加载测试数据
    test_data_path = "btc_rl/data/test_data.npz"
    if not os.path.exists(test_data_path):
        logger.error(f"找不到测试数据文件: {test_data_path}")
        return None
    
    try:
        # 加载数据
        data = np.load(test_data_path)
        windows_orig = data["X"]  # 特征窗口
        prices_orig = data["prices"]  # 价格数据
        
        logger.info(f"原始测试数据形状: 窗口={windows_orig.shape}, 价格={prices_orig.shape}")
        
        # 计算价格变化率，用于识别不同的市场环境
        price_changes = []
        for i in range(1, len(prices_orig)):
            change = (prices_orig[i] - prices_orig[i-1]) / prices_orig[i-1]
            price_changes.append(change)
        
        # 计算移动平均趋势，用于判断上涨、下跌和震荡区间
        window_size = 20  # 使用20个数据点的窗口计算趋势
        trends = []
        
        for i in range(len(price_changes)):
            if i < window_size:
                # 前window_size个点无法计算完整的移动平均，使用到目前为止的数据
                trend = np.mean(price_changes[:i+1]) if i > 0 else 0
            else:
                # 计算过去window_size个点的移动平均趋势
                trend = np.mean(price_changes[i-window_size+1:i+1])
            trends.append(trend)
        
        # 根据市场类型选择合适的数据段
        windows = None
        prices = None
        
        if market_type == "上涨趋势环境":
            # 寻找持续上涨的区间
            up_trends = [(i, trend) for i, trend in enumerate(trends) if trend > 0.001]  # 正趋势阈值
            if up_trends:
                # 选择趋势最强的区间开始位置
                strongest_trend_idx = max(up_trends, key=lambda x: x[1])[0]
                # 确保有足够的数据点
                slice_start = max(0, strongest_trend_idx - 10)
                slice_end = min(len(windows_orig), slice_start + int(len(windows_orig) * 0.3))
                windows = windows_orig[slice_start:slice_end]
                prices = prices_orig[slice_start:slice_end]
                logger.info(f"选择上涨趋势数据: 从索引 {slice_start} 到 {slice_end}，共 {len(windows)} 个窗口")
            else:
                # 如果没有明显上涨趋势，使用初始30%的数据
                slice_end = int(len(prices_orig) * 0.3)
                windows = windows_orig[:slice_end]
                prices = prices_orig[:slice_end]
                logger.info(f"未找到明显上涨趋势，使用前 {slice_end} 个数据点")
        
        elif market_type == "震荡盘整环境":
            # 寻找波动较小的区间
            # 计算每个窗口的价格波动率
            volatilities = []
            for i in range(window_size, len(prices_orig)):
                window_prices = prices_orig[i-window_size:i]
                volatility = np.std(window_prices) / np.mean(window_prices)
                volatilities.append((i, volatility))
            
            # 选择波动率中等的区间
            sorted_volatilities = sorted(volatilities, key=lambda x: x[1])
            middle_index = len(sorted_volatilities) // 2
            
            # 选择这个区间周围的数据
            center_idx = sorted_volatilities[middle_index][0]
            slice_start = max(0, center_idx - int(len(windows_orig) * 0.15))
            slice_end = min(len(windows_orig), center_idx + int(len(windows_orig) * 0.15))
            
            windows = windows_orig[slice_start:slice_end]
            prices = prices_orig[slice_start:slice_end]
            logger.info(f"选择震荡盘整数据: 从索引 {slice_start} 到 {slice_end}，共 {len(windows)} 个窗口")
        
        elif market_type == "下跌趋势环境":
            # 寻找持续下跌的区间
            down_trends = [(i, trend) for i, trend in enumerate(trends) if trend < -0.001]  # 负趋势阈值
            if down_trends:
                # 选择趋势最强的区间开始位置
                strongest_down_idx = min(down_trends, key=lambda x: x[1])[0]
                # 确保有足够的数据点
                slice_start = max(0, strongest_down_idx - 10)
                slice_end = min(len(windows_orig), slice_start + int(len(windows_orig) * 0.3))
                windows = windows_orig[slice_start:slice_end]
                prices = prices_orig[slice_start:slice_end]
                logger.info(f"选择下跌趋势数据: 从索引 {slice_start} 到 {slice_end}，共 {len(windows)} 个窗口")
            else:
                # 如果没有明显下跌趋势，使用最后30%的数据
                slice_start = int(len(prices_orig) * 0.7)
                windows = windows_orig[slice_start:]
                prices = prices_orig[slice_start:]
                logger.info(f"未找到明显下跌趋势，使用最后 {len(windows)} 个数据点")
        
        elif market_type == "反弹行情环境":
            # 寻找先下跌后上涨的区间
            # 首先查找大幅下跌点
            large_drops = []
            for i in range(window_size, len(prices_orig)-window_size):
                pre_window = prices_orig[i-window_size:i]
                post_window = prices_orig[i:i+window_size]
                
                # 计算下跌和反弹幅度
                pre_drop = (min(pre_window) - max(pre_window)) / max(pre_window)
                post_rise = (max(post_window) - min(post_window)) / min(post_window)
                
                # 如果前窗口有下跌且后窗口有上涨
                if pre_drop < -0.05 and post_rise > 0.05:
                    large_drops.append((i, pre_drop, post_rise))
            
            if large_drops:
                # 选择下跌后反弹最明显的点
                best_rebound_idx = max(large_drops, key=lambda x: x[2])[0]
                # 选择该点前后的数据
                slice_start = max(0, best_rebound_idx - int(window_size * 1.5))
                slice_end = min(len(windows_orig), best_rebound_idx + int(window_size * 1.5))
                
                windows = windows_orig[slice_start:slice_end]
                prices = prices_orig[slice_start:slice_end]
                logger.info(f"选择反弹行情数据: 从索引 {slice_start} 到 {slice_end}，共 {len(windows)} 个窗口")
            else:
                # 如果没有找到明显的反弹，创建人工反弹序列
                # 先取一段下跌，再取一段上涨
                mid_point = len(prices_orig) // 2
                down_slice = int(len(prices_orig) * 0.2)
                up_slice = int(len(prices_orig) * 0.2)
                
                windows_down = windows_orig[mid_point:mid_point+down_slice]
                prices_down = prices_orig[mid_point:mid_point+down_slice]
                
                windows_up = windows_orig[:up_slice]
                prices_up = prices_orig[:up_slice]
                
                # 平滑连接点
                # 计算价格连接点的比例
                if len(prices_down) > 0 and len(prices_up) > 0:
                    connect_ratio = prices_up[0] / prices_down[-1]
                    
                    # 调整上涨段的价格和特征，使连接更平滑
                    for i in range(len(prices_up)):
                        adjustment_factor = 1.0 - (i / len(prices_up)) * (1.0 - connect_ratio)
                        prices_up[i] *= adjustment_factor
                        
                        # 同时调整窗口特征中的价格相关数据
                        if i < len(windows_up):
                            for j in range(len(windows_up[i])):
                                for k in range(4):  # 假设前4个特征是OHLC
                                    if k < windows_up[i][j].shape[0]:
                                        windows_up[i][j][k] *= adjustment_factor
                
                # 拼接数据模拟反弹
                windows = np.concatenate([windows_down, windows_up])
                prices = np.concatenate([prices_down, prices_up])
                logger.info(f"创建人工反弹行情数据: 共 {len(windows)} 个窗口")
        
        # 从模型路径中提取模型名称
        model_name = os.path.basename(model_path).split('.')[0]
        from gymnasium.wrappers import TimeLimit
        from stable_baselines3 import SAC
        from btc_rl.src.env import BtcTradingEnv
        
        # 加载模型
        logger.info(f"加载模型: {model_path}")
        model = SAC.load(model_path)
        
        # 创建环境
        test_queue = queue.Queue(maxsize=1000)
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
        
        # 运行一个episode
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
            
            # 保存数据
            if step % DATA_SAMPLING_INTERVAL == 0:
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
        
        # 计算统计指标
        result = calculate_model_statistics(history_data)
        
        # 添加模型信息
        result["model_name"] = model_name
        result["market_type"] = market_type
        result["data_size"] = len(windows)
        
        # 计算额外的详细指标
        # 1. 分析交易数据，计算平均收益和亏损
        position_changes = []
        for i in range(1, len(history_data)):
            prev_pos = history_data[i-1].get("position_btc", 0)
            curr_pos = history_data[i].get("position_btc", 0)
            if abs(curr_pos - prev_pos) > 0.000001:  # 检测仓位变化
                entry_price = history_data[i].get("price", 0)
                entry_time = i
                position_changes.append((entry_time, curr_pos - prev_pos, entry_price))
        
        # 计算交易盈亏详情
        profitable_trades = []
        losing_trades = []
        trade_durations = []
        position_sizes = []
        
        for i, (entry_idx, size_change, entry_price) in enumerate(position_changes):
            if i < len(position_changes) - 1:
                exit_idx = position_changes[i+1][0]
                exit_price = history_data[exit_idx].get("price", 0)
                
                # 计算持仓时间（以数据点数量为单位）
                duration = exit_idx - entry_idx
                trade_durations.append(duration)
                
                # 记录仓位大小
                position_sizes.append(abs(size_change))
                
                if size_change > 0:  # 做多
                    profit = (exit_price - entry_price) * abs(size_change)
                    profit_pct = (exit_price - entry_price) / entry_price
                else:  # 做空
                    profit = (entry_price - exit_price) * abs(size_change)
                    profit_pct = (entry_price - exit_price) / entry_price
                
                if profit > 0:
                    profitable_trades.append(profit_pct)
                else:
                    losing_trades.append(profit_pct)
        
        # 计算平均收益和亏损
        avg_profit = np.mean(profitable_trades) if profitable_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        avg_trade_duration = np.mean(trade_durations) if trade_durations else 0
        avg_position_size = np.mean(position_sizes) if position_sizes else 0
        
        # 添加额外指标
        result["avg_profit"] = avg_profit
        result["avg_loss"] = avg_loss
        result["avg_trade_duration"] = avg_trade_duration
        result["avg_position_size"] = avg_position_size
        result["profitable_trades_count"] = len(profitable_trades)
        result["losing_trades_count"] = len(losing_trades)
        
        # 计算平均交易周期（假设每个数据点为1小时）
        if avg_trade_duration > 0:
            result["avg_trade_duration_hours"] = avg_trade_duration
        
        # 计算最大连续亏损和最大连续盈利交易次数
        if position_changes and len(position_changes) > 1:
            current_win_streak = 0
            current_loss_streak = 0
            max_win_streak = 0
            max_loss_streak = 0
            
            for i, (entry_idx, size_change, entry_price) in enumerate(position_changes):
                if i < len(position_changes) - 1:
                    exit_idx = position_changes[i+1][0]
                    exit_price = history_data[exit_idx].get("price", 0)
                    
                    if size_change > 0:  # 做多
                        profit = exit_price - entry_price
                    else:  # 做空
                        profit = entry_price - exit_price
                    
                    if profit > 0:
                        current_win_streak += 1
                        current_loss_streak = 0
                        max_win_streak = max(max_win_streak, current_win_streak)
                    else:
                        current_loss_streak += 1
                        current_win_streak = 0
                        max_loss_streak = max(max_loss_streak, current_loss_streak)
            
            result["max_win_streak"] = max_win_streak
            result["max_loss_streak"] = max_loss_streak
        
        return result
    
    except Exception as e:
        logger.error(f"在{market_type}环境下回测出错: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def run_multi_environment_backtest(model_path: str = None):
    """
    执行多环境回测
    
    Args:
        model_path (str, optional): 模型路径，如果为None则使用黄金法则选出的最佳模型
        
    Returns:
        list: 所有环境的回测结果
    """
    # 如果未指定模型，使用黄金法则选择的最佳模型
    if model_path is None:
        model_path = get_best_model()
    
    if not model_path or not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return None
    
    logger.info(f"使用模型: {model_path}")
    
    # 定义不同市场环境
    market_environments = [
        "上涨趋势环境",
        "震荡盘整环境",
        "下跌趋势环境", 
        "反弹行情环境"
    ]
    
    # 保存回测结果
    results = []
    
    # 执行多环境回测
    for env in market_environments:
        result = backtest_model_in_environment(model_path, env)
        if result:
            results.append(result)
    
    return results

def analyze_backtest_results(results):
    """
    分析多环境回测结果
    
    Args:
        results (list): 回测结果列表
        
    Returns:
        dict: 分析结果
    """
    if not results:
        logger.error("没有有效的回测结果可分析")
        return None
    
    # 提取各指标
    returns = [r.get('total_return', 0) for r in results]
    drawdowns = [r.get('max_drawdown', 0) for r in results]
    sharpes = [r.get('sharpe_ratio', 0) for r in results]
    win_rates = [r.get('win_rate', 0) for r in results]
    
    # 计算指标的一致性（用标准差衡量）
    return_std = np.std(returns) if len(returns) > 1 else 0
    drawdown_std = np.std(drawdowns) if len(drawdowns) > 1 else 0
    sharpe_std = np.std(sharpes) if len(sharpes) > 1 else 0
    win_rate_std = np.std(win_rates) if len(win_rates) > 1 else 0
    
    # 计算市场适应性得分
    # 得分越高越好，表示模型在不同环境下表现的一致性越高
    adaptability_score = 0
    if len(results) > 1:
        # 归一化标准差（使用1减去归一化标准差，使得值越小得分越高）
        norm_return_std = return_std / max(0.01, np.mean(returns)) if np.mean(returns) > 0 else 1
        norm_drawdown_std = drawdown_std / max(0.01, np.mean(drawdowns)) if np.mean(drawdowns) > 0 else 1
        norm_sharpe_std = sharpe_std / max(0.01, np.mean(sharpes)) if np.mean(sharpes) > 0 else 1
        norm_win_rate_std = win_rate_std / max(0.01, np.mean(win_rates)) if np.mean(win_rates) > 0 else 1
        
        # 计算综合得分（加权平均）
        adaptability_score = (
            (1 - norm_return_std) * 0.4 +  # 回报一致性
            (1 - norm_drawdown_std) * 0.3 +  # 风险一致性
            (1 - norm_sharpe_std) * 0.2 +  # 风险调整回报一致性
            (1 - norm_win_rate_std) * 0.1  # 交易质量一致性
        )
    
    # 找出最佳和最差环境
    if len(returns) > 0:
        best_idx = np.argmax(returns)
        worst_idx = np.argmin(returns)
        
        best_env = results[best_idx].get('market_type', 'unknown')
        worst_env = results[worst_idx].get('market_type', 'unknown')
    else:
        best_env = worst_env = "无法确定"
    
    # 返回分析结果
    return {
        "returns": returns,
        "drawdowns": drawdowns,
        "sharpes": sharpes,
        "win_rates": win_rates,
        "return_std": return_std,
        "drawdown_std": drawdown_std,
        "sharpe_std": sharpe_std,
        "win_rate_std": win_rate_std,
        "adaptability_score": adaptability_score,
        "best_env": best_env,
        "worst_env": worst_env
    }

def display_backtest_results(results, analysis):
    """
    显示回测结果和分析
    
    Args:
        results (list): 回测结果列表
        analysis (dict): 分析结果
    """
    if not results or not analysis:
        logger.error("没有有效的结果可显示")
        return
    
    # 获取模型名称
    model_name = results[0].get('model_name', 'unknown_model')
    
    # 打印表头
    print("\n" + "="*100)
    print(" "*30 + "BTC交易模型多市场环境回测结果")
    print("="*100 + "\n")
    
    print(f"模型: {model_name}")
    print(f"回测时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 准备表格数据
    headers = [
        "市场环境", 
        "数据量", 
        "最终权益", 
        "总回报率", 
        "最大回撤", 
        "夏普比率", 
        "索提诺比率", 
        "胜率", 
        "交易次数", 
        "平均收益/亏损"
    ]
    rows = []
    
    for result in results:
        # 计算平均收益/亏损比(如果有)
        avg_profit = result.get('avg_profit', 0) 
        avg_loss = result.get('avg_loss', 0)
        profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0
        
        total_trades = result.get('profitable_trades_count', 0) + result.get('losing_trades_count', 0)
        
        rows.append([
            result.get('market_type', ''),
            result.get('data_size', result.get('history_length', 0)),
            format_currency(result.get('final_equity', 0)),
            format_percent(result.get('total_return', 0)),
            format_percent(result.get('max_drawdown', 0)),
            f"{result.get('sharpe_ratio', 0):.2f}",
            f"{result.get('sortino_ratio', 0):.2f}",
            format_percent(result.get('win_rate', 0)),
            total_trades,
            f"{profit_loss_ratio:.2f}"
        ])
    
    # 输出表格
    print(tabulate(rows, headers=headers, tablefmt="pretty"))
    
    # 输出市场环境适应性分析
    print("\n市场环境适应性分析:")
    print(f"回报率标准差: {format_percent(analysis['return_std'])} (波动性指标)")
    print(f"最大回撤标准差: {format_percent(analysis['drawdown_std'])} (风险一致性)")
    print(f"夏普比率标准差: {analysis['sharpe_std']:.2f} (风险调整回报一致性)")
    print(f"胜率标准差: {format_percent(analysis['win_rate_std'])} (交易质量一致性)")
    
    print(f"\n市场环境适应性得分: {analysis['adaptability_score']:.4f} (0-1，越高越好)")
    
    # 提供简单分析
    if analysis['adaptability_score'] > 0.75:
        print("结论: 模型在各市场环境中表现稳定，具有较强的市场适应性")
    elif analysis['adaptability_score'] > 0.5:
        print("结论: 模型市场适应性一般，在某些市场环境中表现更佳")
    else:
        print("结论: 模型市场适应性较弱，对特定市场环境依赖性较强")
    
    # 打印各环境性能比较
    returns = analysis['returns']
    env_names = [r.get('market_type', '') for r in results]
    
    if len(returns) > 0 and len(env_names) > 0:
        best_idx = np.argmax(returns)
        worst_idx = np.argmin(returns)
        
        if best_idx < len(env_names) and worst_idx < len(env_names):
            print(f"\n最佳适应环境: {env_names[best_idx]}")
            print(f"  - 回报率: {format_percent(returns[best_idx])}")
            print(f"  - 夏普比率: {analysis['sharpes'][best_idx]:.2f}")
            print(f"  - 胜率: {format_percent(analysis['win_rates'][best_idx])}")
            
            print(f"\n最弱适应环境: {env_names[worst_idx]}")
            print(f"  - 回报率: {format_percent(returns[worst_idx])}")
            print(f"  - 夏普比率: {analysis['sharpes'][worst_idx]:.2f}")
            print(f"  - 胜率: {format_percent(analysis['win_rates'][worst_idx])}")
    
    # 提供优化建议
    print("\n优化建议:")
    
    # 根据回测结果提供具体建议
    if analysis['adaptability_score'] < 0.6:
        print("1. 考虑增加模型对不同市场环境的适应能力，可通过多环境训练实现")
        print("2. 针对表现较弱的市场环境（如下跌趋势或高波动环境）增加特定训练")
        print("3. 优化风险管理策略，特别是在模型表现不佳的环境中")
    
    # 检查胜率是否过低
    if min(analysis['win_rates']) < 0.3:
        print(f"4. 提高低胜率环境（{env_names[np.argmin(analysis['win_rates'])]}，胜率: {format_percent(min(analysis['win_rates']))})的交易质量")
    
    # 检查最大回撤是否过大
    if max(analysis['drawdowns']) > 0.2:
        print(f"5. 减少高回撤环境（{env_names[np.argmax(analysis['drawdowns'])]}，回撤: {format_percent(max(analysis['drawdowns']))})的风险敞口")
    
    print("\n" + "="*100)
    print(" "*15 + "市场适应性测试完成，可根据结果优化模型稳健性")
    print("="*100 + "\n")

def backtest_model_in_timeframe(model_path: str, start_date: str, end_date: str, exchange: str = "binance", 
                          symbol: str = "BTC/USDT", timeframe: str = "1h", skip_data_fetch: bool = False):
    """在指定的时间范围内回测模型
    
    Args:
        model_path (str): 模型路径，如果为None则使用黄金法则选出的最佳模型
        start_date (str): 开始日期，格式为YYYY-MM-DD
        end_date (str): 结束日期，格式为YYYY-MM-DD
        exchange (str): 交易所，用于获取历史数据
        symbol (str): 交易对，用于获取历史数据
        timeframe (str): 时间周期，用于获取历史数据
        skip_data_fetch (bool): 是否跳过数据获取，使用可用数据
        
    Returns:
        list: 回测结果列表
    """
    try:
        from btc_rl.src.period_backtester import backtest_model_in_period
        
        # 如果没有指定模型路径，使用黄金法则选出的最佳模型
        if model_path is None:
            logger.info("未指定模型路径，使用黄金法则选出的最佳模型")
            model_path = get_best_model()
            if model_path is None:
                logger.error("无法获取最佳模型，无法继续回测")
                return []
            logger.info(f"使用最佳模型路径: {model_path}")
        
        # 导入period_backtester中的功能
        logger.info(f"开始在时间段 {start_date} 到 {end_date} 回测模型")
        logger.info(f"数据源配置: 交易所={exchange}, 交易对={symbol}, 时间周期={timeframe}")
        
        # 如果未跳过数据获取，且没有特定日期范围的数据文件，可以在这里直接获取数据
        if not skip_data_fetch:
            date_range_csv = f"btc_rl/data/BTC_1hour_{start_date}_{end_date}.csv"
            if not os.path.exists(date_range_csv):
                logger.info(f"准备从 {exchange} 获取 {start_date} 到 {end_date} 的 {symbol} 数据...")
                try:
                    from btc_rl.src.data_fetcher import fetch_historical_data, preprocess_data
                    
                    # 下载特定日期范围的数据
                    df = fetch_historical_data(
                        exchange_id=exchange,
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if df is not None and not df.empty:
                        # 预处理数据
                        df = preprocess_data(df)
                        
                        # 创建数据目录
                        os.makedirs(os.path.dirname(date_range_csv), exist_ok=True)
                        
                        # 保存处理后的数据
                        df.to_csv(date_range_csv, index=False)
                        logger.info(f"已保存获取的数据到 {date_range_csv}")
                    else:
                        logger.warning("无法获取指定日期范围的数据，将尝试使用可用数据")
                except Exception as e:
                    logger.error(f"获取数据时出错: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
        
        # 不同市场类型的标签
        market_types = [
            "上涨趋势环境",
            "下跌趋势环境",
            "震荡盘整环境",
            "反弹行情环境"
        ]
        
        # 存储各环境的回测结果
        results = []
        
        # 对每种市场类型进行回测
        for market_type in market_types:
            result = backtest_model_in_period(
                model_path, 
                start_date, 
                end_date, 
                market_type, 
                exchange=exchange, 
                symbol=symbol, 
                timeframe=timeframe
            )
            if result:
                results.append(result)
                logger.info(f"{market_type}回测完成")
            else:
                logger.error(f"{market_type}回测失败")
        
        return results
        
    except Exception as e:
        logger.error(f"在时间范围回测时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="BTC交易模型多环境回测工具")
    parser.add_argument("--model", help="指定要回测的模型路径，默认使用黄金法则选出的最佳模型")
    parser.add_argument("--start-date", help="回测起始日期 (格式: YYYY-MM-DD)")
    parser.add_argument("--end-date", help="回测结束日期 (格式: YYYY-MM-DD)")
    parser.add_argument("--exchange", default="binance", help="指定获取数据的交易所 (默认: binance)")
    parser.add_argument("--symbol", default="BTC/USDT", help="指定交易对 (默认: BTC/USDT)")
    parser.add_argument("--timeframe", default="1h", help="指定时间周期 (默认: 1h)")
    parser.add_argument("--skip-data-fetch", action="store_true", help="跳过数据获取，使用可用数据")
    args = parser.parse_args()
    
    try:
        # 检查是否指定了日期范围
        if args.start_date and args.end_date:
            # 使用指定的日期范围进行回测
            logger.info(f"使用指定的时间段进行回测: {args.start_date} 至 {args.end_date}")
            results = backtest_model_in_timeframe(
                args.model, 
                args.start_date, 
                args.end_date,
                exchange=args.exchange,
                symbol=args.symbol,
                timeframe=args.timeframe,
                skip_data_fetch=args.skip_data_fetch
            )
        else:
            # 执行标准的多环境回测
            logger.info("未指定日期范围，进行标准多环境回测")
            results = run_multi_environment_backtest(args.model)
        
        if not results:
            logger.error("没有获取到有效的回测结果")
            return 1
        
        # 分析回测结果
        analysis = analyze_backtest_results(results)
        
        # 显示回测结果
        display_backtest_results(results, analysis)
        
        return 0
        
    except Exception as e:
        logger.error(f"回测过程中出错: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
