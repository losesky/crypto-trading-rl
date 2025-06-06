#!/usr/bin/env python3
"""
Train SAC on the BTC environment with the custom CNN extractor.
"""

import asyncio
import json
import queue
import threading
import os
import datetime
from pathlib import Path

import numpy as np
import torch as th
import websockets
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from btc_rl.src.env import BtcTradingEnv
from btc_rl.src.policies import TimeSeriesCNN
# 导入模型评估和统计相关模块
from btc_rl.src.model_comparison import DATA_SAMPLING_INTERVAL, calculate_model_statistics, calculate_drawdowns
from btc_rl.src.config import get_config

# --- WebSocket Server Setup ---
WEBSOCKET_CLIENTS = set()
MESSAGE_QUEUE = queue.Queue(maxsize=100)  # Thread-safe queue

async def websocket_handler(websocket):  # Removed 'path' argument
    """Handles new WebSocket connections and keeps them alive."""
    WEBSOCKET_CLIENTS.add(websocket)
    print(f"[WebSocket] Client connected: {websocket.remote_address}")
    try:
        await websocket.wait_closed()
    finally:
        print(f"[WebSocket] Client disconnected: {websocket.remote_address}")
        WEBSOCKET_CLIENTS.remove(websocket)

async def broadcast_messages():
    """Continuously checks the queue and broadcasts messages to clients."""
    current_loop = asyncio.get_running_loop() # Get the loop this coroutine is running on
    while True:
        try:
            message = await current_loop.run_in_executor(None, MESSAGE_QUEUE.get) # Blocking get in executor
            if WEBSOCKET_CLIENTS:
                # If websockets.broadcast(...) is returning None, call it directly.
                # This assumes it's a fire-and-forget function or handles its own async execution.
                websockets.broadcast(WEBSOCKET_CLIENTS, message)
        except queue.Empty: # Should not happen with blocking get, but as a safeguard
            await asyncio.sleep(0.01)
        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                print("[WebSocket] Broadcast: Event loop closed, stopping.")
                break
            print(f"[WebSocket] Error broadcasting message (RuntimeError): {e}")
            await asyncio.sleep(0.1)
        except Exception as e:
            print(f"[WebSocket] Error broadcasting message: {e}")
            await asyncio.sleep(0.1) # Avoid busy-looping on persistent errors
    print("[WebSocket] Broadcast messages loop finished.")

def start_websocket_server_thread():
    """Runs the WebSocket server in a separate thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _server_main_coroutine():
        """The main coroutine for the WebSocket server thread."""
        print("[WebSocket] Server coroutine starting...")
        async with websockets.serve(websocket_handler, "localhost", 8765) as server:
            print(f"[WebSocket] Server started on {server.sockets[0].getsockname()}")
            await broadcast_messages() 
        print("[WebSocket] Server coroutine finished.")

    try:
        print("[WebSocket] Starting WebSocket server event loop in new thread...")
        loop.run_until_complete(_server_main_coroutine())
    except KeyboardInterrupt:
        print("[WebSocket] Server thread event loop interrupted by KeyboardInterrupt.")
    except Exception as e:
        print(f"[WebSocket] Server thread event loop exited with exception: {e}")
    finally:
        print("[WebSocket] Cleaning up server thread event loop...")
        if not loop.is_closed():
            for task in asyncio.all_tasks(loop=loop):
                if not task.done():
                    task.cancel()
            
            async def wait_for_tasks_cancellation():
                tasks = [t for t in asyncio.all_tasks(loop=loop) if t is not asyncio.current_task(loop=loop)]
                if tasks:
                    print(f"[WebSocket] Waiting for {len(tasks)} tasks to cancel...")
                    await asyncio.gather(*tasks, return_exceptions=True)
                print("[WebSocket] All tasks cancelled.")

            try:
                if not loop.is_closed():
                    loop.run_until_complete(wait_for_tasks_cancellation())
            except RuntimeError as e:
                print(f"[WebSocket] Minor error during task cancellation: {e}")
            finally:
                if not loop.is_closed():
                    loop.close()
                    print("[WebSocket] Server thread event loop closed.")
        else:
            print("[WebSocket] Server thread event loop was already closed.")
    print("[WebSocket] Server thread finished execution.")

# --- End WebSocket Server Setup ---


def make_env(split: str, msg_queue: queue.Queue):
    data = np.load(f"btc_rl/data/{split}_data.npz")
    windows, prices = data["X"], data["prices"]
    print(f"{split} set windows.shape = {windows.shape}") 
    env = BtcTradingEnv(windows, 
    prices, 
    websocket_queue=msg_queue,
    risk_capital_source="margin_equity",
    risk_fraction_per_trade=0.02,
    max_leverage=1.0,
    )
    # add a very generous TimeLimit just so SB3 knows an episode can finish
    return TimeLimit(env, max_episode_steps=len(windows) + 1)


def evaluate_model_with_metrics(model_path, save_metrics=True):
    """
    评估模型性能并保存指标到对应的JSON文件
    
    Args:
        model_path (str): 模型文件路径
        save_metrics (bool): 是否保存指标到文件
        
    Returns:
        dict: 包含模型性能指标的字典
    """
    print(f"[评估] 正在评估模型: {model_path}")
    config = get_config()
    
    try:
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"[评估] 错误: 模型文件不存在: {model_path}")
            return None
            
        # 加载模型
        print(f"[评估] 正在加载模型: {model_path}")
        model = SAC.load(model_path)
        
        # 创建测试环境
        test_queue = queue.Queue(maxsize=1000)
        test_env = make_env("test", test_queue)
        
        # 初始化历史数据记录
        history_data = []
        
        # 运行一个episode
        obs, _ = test_env.reset()
        done = False
        step = 0
        
        while not done:
            # 使用模型预测动作
            action, _ = model.predict(obs, deterministic=True)
            
            # 执行动作
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            step += 1
            
            # 保存数据
            if step % DATA_SAMPLING_INTERVAL == 0:  # 每2步记录一次，增加数据点密度
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
        stats = calculate_model_statistics(history_data)
        
        # 确保胜率和交易次数有效
        if stats["win_rate"] == 0 and stats["total_trades"] > 0:
            print(f"[警告] 检测到异常情况：交易次数为 {stats['total_trades']} 但胜率为 0，尝试重新计算...")
            # 手动计算交易收益
            position_changes = []
            for i in range(1, len(history_data)):
                prev_pos = history_data[i-1].get("position_btc", 0)
                curr_pos = history_data[i].get("position_btc", 0)
                if abs(curr_pos - prev_pos) > 0.000001:  # 检测仓位变化
                    entry_price = history_data[i].get("price", 0)
                    position_changes.append((i, curr_pos - prev_pos, entry_price))
            
            # 计算交易盈亏
            profitable_trades = 0
            total_effective_trades = 0
            
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
                    total_effective_trades += 1
            
            # 更新胜率
            if total_effective_trades > 0:
                stats["win_rate"] = profitable_trades / total_effective_trades
                print(f"[修复] 重新计算胜率：交易次数 {total_effective_trades}, 盈利交易 {profitable_trades}, 胜率 {stats['win_rate']:.2%}")
        
        # 获取模型文件名（不带扩展名）
        model_name = os.path.basename(model_path).split('.')[0]
        
        # 保存指标到JSON文件
        if save_metrics:
            # 创建指标文件路径
            metrics_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metrics")
            os.makedirs(metrics_dir, exist_ok=True)
            metrics_path = os.path.join(metrics_dir, f"{model_name}_metrics.json")
            
            # 准备指标数据
            metrics_data = {
                "model_name": model_name,
                "model_path": os.path.abspath(model_path),  # 添加绝对路径以便更容易关联
                "evaluation_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "final_equity": float(stats["final_equity"]),
                "total_return": float(stats["total_return"]),
                "max_drawdown": float(stats["max_drawdown"]),
                "sharpe_ratio": float(stats["sharpe_ratio"]),
                "sortino_ratio": float(stats["sortino_ratio"]),
                "total_trades": int(stats["total_trades"]),
                "win_rate": float(stats["win_rate"]),
                "history": history_data[-config.get_history_save_count():]  # 只保存配置指定数量的数据点[-config.get_history_save_count():]  # 只保存配置指定数量的数据点
            }
            
            # 保存到文件
            with open(metrics_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            print(f"[评估] 模型指标已保存到: {metrics_path}")
        
        return stats
    
    except Exception as e:
        print(f"[评估] 评估模型时出错: {e}")
        import traceback
        print(traceback.format_exc())
        return {
            "final_equity": 10000.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "total_trades": 0,
            "win_rate": 0.0
        }

def main(episodes: int = 10):
    # Start WebSocket server in a separate thread
    print("[Main] Starting WebSocket server thread...")
    server_thread = threading.Thread(target=start_websocket_server_thread, daemon=True)
    server_thread.start()
    print("[Main] WebSocket server thread started.")

    # 1) env & policy
    env = DummyVecEnv([lambda: make_env("train", MESSAGE_QUEUE)])

    policy_kwargs = dict(
        features_extractor_class=TimeSeriesCNN,
        features_extractor_kwargs={},
    )

    # ---> buffer trimmed to 100k to speed sample / RAM
    model = SAC(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        buffer_size=100_000,
        verbose=1,
        tensorboard_log="btc_rl/logs/tb/",
        device="auto",
    )

    # compute steps from file, not env wrapper magic
    n_timesteps = np.load("btc_rl/data/train_data.npz")["X"].shape[0]

    # 创建指标摘要文件
    summary_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metrics", "models_summary.json")
    summary_data = {"models": []}
    
    for ep in range(episodes):
        model.learn(total_timesteps=n_timesteps, reset_num_timesteps=False, progress_bar=True)
        model_path = f"btc_rl/models/sac_ep{ep+1}.zip"
        model.save(model_path)
        
        # 评估模型并保存指标
        print(f"[训练] 评估模型 {ep+1}/{episodes}...")
        stats = evaluate_model_with_metrics(model_path)
        
        # 添加到摘要
        summary_data["models"].append({
            "model_name": f"sac_ep{ep+1}",
            "final_equity": float(stats["final_equity"]),
            "total_return": float(stats["total_return"]),
            "max_drawdown": float(stats["max_drawdown"]),
            "sharpe_ratio": float(stats["sharpe_ratio"]),
            "sortino_ratio": float(stats["sortino_ratio"]),
            "total_trades": int(stats["total_trades"]),
            "win_rate": float(stats["win_rate"]),
            "total_fees": float(stats.get("total_fees", 0.0)),
        })
        
        # 保存更新的摘要
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"[训练] 已更新模型摘要: {summary_path}")


if __name__ == "__main__":
    main(episodes=10)