#!/usr/bin/env python3
"""
Train SAC on the BTC environment with the custom CNN extractor.
"""

import asyncio
import json
import queue
import threading
from pathlib import Path

import numpy as np
import torch as th
import websockets
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from btc_rl.src.env import BtcTradingEnv
from btc_rl.src.policies import TimeSeriesCNN

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

    for ep in range(episodes):
        model.learn(total_timesteps=n_timesteps, reset_num_timesteps=False, progress_bar=True)
        model.save(f"btc_rl/models/sac_ep{ep+1}.zip")


if __name__ == "__main__":
    main(episodes=10)