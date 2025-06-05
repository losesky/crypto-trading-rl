# btc_rl/src/env.py
"""
BTC scalper environment — continuous actions, equity-scaled trade size,
and a reward shaped for frequent, low-risk realisations.
"""

from __future__ import annotations
import json # Added for serializing data for WebSocket
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import queue # Added for type hinting

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class BtcTradingEnv(gym.Env):
    """
    Observation : (99, 9)  ➜ last 99 hourly bars
    Action      : scalar a ∈ [-1, 1]
                  USD delta based on specified risk capital and fraction.
    Episode     : runs until data exhausted, equity ≤ 0, or liquidation.
    """

    metadata = {"render_modes": ["human"]}

    # ------------------------ init ------------------------
    def __init__(
        self,
        windows: np.ndarray,  # These are now the observation windows (e.g., 99, 9)
        prices: np.ndarray,   # These are the execution prices for the step *after* the observation
        *,
        initial_balance: float = 10_000.0,
        reward_weights: Tuple[float, float, float, float] = (
            1.0,   # α  (uPnL)
            1.0,   # β  (ΔMarginEquity)
            10.0,  # γ  (drawdown) - 增加对回撤的惩罚
            0.5,   # δ  (holding)
        ),
        risk_capital_source: str = "initial_balance", # "initial_balance", "cash_balance", or "margin_equity"
        risk_fraction_per_trade: float = 0.02, # 降低到2%的风险资本比例，减少每笔交易的风险
        max_leverage: float = 3.0, # 设置为配置文件中的值(3.0)
        fee_rate: float = 0.0002,          # 2 bp per trade notional
        maintenance_margin_rate: float = 0.05, # 5% maintenance margin
        liquidation_penalty_rate: float = 0.05, # 增加到5%，提高清算惩罚
        max_abs_position_btc_cap: float = 100.0, # Max absolute BTC position
        funding_rate_hourly: float = 0.0001 / 24, # Example: 0.01% daily / 24 hours
        flat_bonus: float = 0.02,          # reward for being flat & above EMA
        log_dir: str | Path = "btc_rl/logs/episodes",
        websocket_queue: queue.Queue | None = None, # Added websocket_queue
    ):
        super().__init__()
        assert windows.ndim == 3 and windows.shape[2] == 9 # (N_samples, OBS_FEATURE_LENGTH, num_features)
        assert windows.shape[0] == prices.shape[0]
        assert risk_capital_source in ["initial_balance", "cash_balance", "margin_equity"], \
            "risk_capital_source must be 'initial_balance', 'cash_balance', or 'margin_equity'"

        self.windows = windows.astype(np.float32)
        self.prices = prices.astype(np.float32) # Execution prices
        self.N = len(self.windows)

        self.observation_space = spaces.Box(0.0, 1.0, shape=(windows.shape[1], windows.shape[2]), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        self.initial_balance = initial_balance
        self.α, self.β, self.γ, self.δ = reward_weights
        
        self.risk_capital_source = risk_capital_source
        self.risk_fraction_per_trade = risk_fraction_per_trade
        self.max_leverage = max_leverage
        self.fee_rate = fee_rate
        self.maintenance_margin_rate = maintenance_margin_rate
        self.liquidation_penalty_rate = liquidation_penalty_rate
        self.max_abs_position_btc_cap = max_abs_position_btc_cap
        self.funding_rate_hourly = funding_rate_hourly
        self.flat_bonus = flat_bonus

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.websocket_queue = websocket_queue # Store the queue

        #— runtime state vars (set in reset)
        self.cash_balance: float = 0.0  # Renamed from balance
        self.margin_equity: float = 0.0 # cash_balance + upnl
        self.position_btc: float = 0.0
        self.entry_price: float = 0.0
        self.upnl: float = 0.0
        self.hold_hours: int = 0
        self.peak_margin_equity: float = 0.0 # Renamed from peak_equity
        self.idx: int = 0 # Index for current observation window and corresponding execution price
        self.ema24: float = 0.0          # 24-hour EMA of margin_equity
        self.episode_log: List[Dict] = []
        self.was_liquidated_this_step: bool = False

        # Add B&H state vars
        self.initial_price_for_buy_and_hold: float = 0.0
        self.buy_and_hold_btc_amount: float = 0.0
        self.buy_and_hold_equity: float = 0.0

    # ------------------------ core API ------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.cash_balance = float(self.initial_balance)
        self.position_btc = 0.0
        self.entry_price = 0.0
        self.upnl = 0.0
        self.margin_equity = self.cash_balance # Initially, margin_equity is cash_balance
        self.hold_hours = 0
        self.peak_margin_equity = self.margin_equity
        self.idx = 0 # Index for current observation window and corresponding execution price
        self.ema24 = self.margin_equity
        self.episode_log = []
        self.was_liquidated_this_step = False
        # cumulative fees paid during the episode
        self.total_fee_paid = 0.0

        # Initialize B&H strategy
        if self.N > 0 and self.prices.size > 0:
            self.initial_price_for_buy_and_hold = self.prices[0]
            if self.initial_price_for_buy_and_hold > 1e-9: # Avoid division by zero or tiny numbers
                self.buy_and_hold_btc_amount = self.initial_balance / self.initial_price_for_buy_and_hold
            else:
                self.buy_and_hold_btc_amount = 0.0
        else:
            self.initial_price_for_buy_and_hold = 0.0
            self.buy_and_hold_btc_amount = 0.0

        self.buy_and_hold_equity = self.initial_balance # At reset, B&H equity is initial balance

        return self.windows[0], {}

    def step(self, action: np.ndarray):
        act = float(np.clip(action[0], -1.0, 1.0))
        
        # 检查索引是否在有效范围内
        if self.idx >= len(self.prices):
            # 已经达到数据末尾，返回模拟结束信号
            return np.zeros_like(self.windows[0]), 0.0, True, False, {
                "margin_equity": self.margin_equity,
                "cash_balance": self.cash_balance,
                "bankrupt": False,
                "liquidated": False,
                "end_of_data": True
            }
            
        price = float(self.prices[self.idx])
        step_fee_cost = 0.0  # fees incurred in this step
        self.was_liquidated_this_step = False

        margin_equity_at_step_start = self.cash_balance + self.upnl

        # 0. Update Buy-and-Hold equity
        if self.initial_price_for_buy_and_hold > 1e-9:
            self.buy_and_hold_equity = self.buy_and_hold_btc_amount * price
        else:
            self.buy_and_hold_equity = self.initial_balance
        
        # 1. Mark-to-Market existing position
        if self.position_btc != 0:
            self.upnl = self.position_btc * (price - self.entry_price)
        else:
            self.upnl = 0.0
        self.margin_equity = self.cash_balance + self.upnl

        # 2. Apply Funding Cost
        if self.position_btc != 0:
            funding_cost = abs(self.position_btc * price) * self.funding_rate_hourly
            self.cash_balance -= funding_cost
            self.margin_equity -= funding_cost
            self.total_fee_paid += funding_cost
            step_fee_cost      += funding_cost

        # 3. Maintenance Margin Check & Potential Liquidation
        if self.position_btc != 0:
            current_position_value = abs(self.position_btc * price)
            maintenance_margin_requirement = current_position_value * self.maintenance_margin_rate
            if self.margin_equity < maintenance_margin_requirement:
                self.was_liquidated_this_step = True
                
                self.cash_balance += self.upnl 
                
                liquidation_penalty_amount = current_position_value * self.liquidation_penalty_rate
                self.cash_balance -= liquidation_penalty_amount
                
                closing_fee = current_position_value * self.fee_rate
                self.cash_balance -= closing_fee
                self.total_fee_paid += closing_fee
                step_fee_cost      += closing_fee
                
                self.position_btc = 0.0
                self.entry_price = 0.0
                self.upnl = 0.0
                self.hold_hours = 0
                self.margin_equity = self.cash_balance
                delta_usd = 0.0
                fee = closing_fee + liquidation_penalty_amount
            else:
                delta_usd, fee = self._calculate_trade_and_fee(act, price, margin_equity_at_step_start)
        else:
            delta_usd, fee = self._calculate_trade_and_fee(act, price, margin_equity_at_step_start)

        if not self.was_liquidated_this_step and abs(delta_usd) > 1e-8:
            self.cash_balance -= fee
            self.total_fee_paid += fee
            step_fee_cost      += fee
            
            delta_btc = delta_usd / price
            realised_pnl_from_trade = 0.0
            
            if self.position_btc != 0 and np.sign(delta_btc) != np.sign(self.position_btc) and abs(delta_btc) > 1e-9:
                closed_btc_amount = min(abs(delta_btc), abs(self.position_btc))
                realised_pnl_from_trade += closed_btc_amount * (price - self.entry_price) * np.sign(self.position_btc)
            
            self.cash_balance -= delta_usd

            if abs(self.position_btc + delta_btc) < 1e-9:
                self.position_btc = 0.0
                self.entry_price = 0.0
                self.hold_hours = 0
            else:
                if self.position_btc == 0 or np.sign(delta_btc) == np.sign(self.position_btc) or abs(delta_btc) > abs(self.position_btc):
                    if self.position_btc != 0 and np.sign(delta_btc) == np.sign(self.position_btc):
                        self.entry_price = (abs(self.position_btc) * self.entry_price + abs(delta_btc) * price) / (abs(self.position_btc) + abs(delta_btc))
                    else:
                        self.entry_price = price
                        self.hold_hours = 0
                self.position_btc += delta_btc
                if self.position_btc != 0:
                     self.hold_hours +=1

        if self.position_btc != 0:
            self.upnl = self.position_btc * (price - self.entry_price)
        else:
            self.upnl = 0.0
        self.margin_equity = self.cash_balance + self.upnl
        
        self.peak_margin_equity = max(self.peak_margin_equity, self.margin_equity)
        drawdown = (self.peak_margin_equity - self.margin_equity) / self.peak_margin_equity if self.peak_margin_equity > 1e-9 else 0.0
        
        Δ_margin_equity_numeric = self.margin_equity - margin_equity_at_step_start
        Δ_margin_equity_ratio = Δ_margin_equity_numeric / margin_equity_at_step_start if abs(margin_equity_at_step_start) > 1e-9 else 0.0
        
        uPnL_norm = self.upnl / margin_equity_at_step_start if abs(margin_equity_at_step_start) > 1e-9 else 0.0
        hold_norm = min(self.hold_hours / 240, 1)

        κ = 1 + max(0, -uPnL_norm)

        total_costs_this_step = step_fee_cost

        # 添加对接近破产状态的额外惩罚
        bankruptcy_risk = 0.0
        if self.margin_equity > 0:
            # 当保证金权益低于初始资金的10%时，开始施加越来越强的惩罚
            bankruptcy_threshold = self.initial_balance * 0.1
            if self.margin_equity < bankruptcy_threshold:
                bankruptcy_risk_factor = (bankruptcy_threshold - self.margin_equity) / bankruptcy_threshold
                bankruptcy_risk = 15.0 * bankruptcy_risk_factor**2  # 指数惩罚
                
        # 添加对单次大亏损的惩罚
        large_loss_penalty = 0.0
        if Δ_margin_equity_numeric < 0:
            # 计算亏损占账户总值的比例
            loss_ratio = abs(Δ_margin_equity_numeric) / margin_equity_at_step_start if margin_equity_at_step_start > 1e-9 else 0.0
            # 当单次亏损超过5%时开始惩罚，亏损越大惩罚越重
            if loss_ratio > 0.05:
                large_loss_penalty = 5.0 * (loss_ratio - 0.05)**2

        reward = (
            self.α * uPnL_norm
            + self.β * Δ_margin_equity_ratio
            - self.γ * drawdown**2
            - self.δ * hold_norm * κ
            - bankruptcy_risk  # 添加破产风险惩罚
            - large_loss_penalty  # 添加大亏损惩罚
            - (total_costs_this_step / margin_equity_at_step_start if abs(margin_equity_at_step_start) > 1e-9 else 0.0)
        )

        if self.position_btc == 0 and self.margin_equity > self.ema24:
            reward += self.flat_bonus

        self.ema24 = 0.92 * self.ema24 + 0.08 * self.margin_equity

        # Move termination checks before using them
        terminated_by_data = self.idx >= self.N
        terminated_by_bankruptcy = self.margin_equity <= 0
        terminated_by_liquidation = self.was_liquidated_this_step
        done = terminated_by_data or terminated_by_bankruptcy or terminated_by_liquidation

        termination_reason = (
            "data"        if terminated_by_data else
            "bankrupt"    if terminated_by_bankruptcy else
            "liquidation" if terminated_by_liquidation else
            "?"
        )

        # Send data to WebSocket if queue is available
        if self.websocket_queue is not None:
            live_data = {
                "step": int(self.idx),
                "action": act, # The action taken by the agent
                "cash_balance": self.cash_balance,
                "upnl": self.upnl,
                "margin_equity": self.margin_equity,
                "buy_and_hold_equity": self.buy_and_hold_equity,
                "reward": reward,
                "total_fee": self.total_fee_paid,
                "was_liquidated_this_step": self.was_liquidated_this_step,
                "termination_reason": termination_reason,
                "price": price, # Current market price used in this step
                "position_btc": self.position_btc, # Agent's current BTC position
                # Add any other data points you want to visualize live
            }
            try:
                self.websocket_queue.put_nowait(json.dumps(live_data))
            except queue.Full:
                # Optionally, log that the queue was full, but avoid print statements here
                # as they can slow down training significantly.
                # print("[Env] WebSocket message queue is full. Skipping message.")
                pass
            except Exception as e:
                # print(f"[Env] Error putting message in WebSocket queue: {e}")
                pass

        if self.idx % 100 == 0 or done:
            print(f"[{self.idx}] act={act:+.2f} cash={self.cash_balance:,.1f} upnl={self.upnl:,.1f} MEq={self.margin_equity:,.1f} bh_eq={self.buy_and_hold_equity:,.1f} R={reward:+.4f} Liq={self.was_liquidated_this_step} term={termination_reason}")

        self.episode_log.append(
            dict(
                step=int(self.idx),
                cash_bal=self.cash_balance,
                upnl=self.upnl,
                margin_eq=self.margin_equity,
                pos_btc=self.position_btc,
                price=price,
                rew=reward,
                bh_eq=self.buy_and_hold_equity,
                liquidated=self.was_liquidated_this_step,
                fee_paid=self.total_fee_paid,
            )
        )
        
        info = {
            "margin_equity": self.margin_equity,
            "cash_balance": self.cash_balance,
            "upnl": self.upnl,
            "price": price,
            "position_btc": self.position_btc, 
            "buy_and_hold_equity": self.buy_and_hold_equity,
            "total_fee": self.total_fee_paid,
            "was_liquidated_this_step": self.was_liquidated_this_step,
            "termination_reason": termination_reason,
            "bankrupt": terminated_by_bankruptcy and not terminated_by_liquidation,
            "liquidated": terminated_by_liquidation,
        }

        if done:
            self._dump_episode()
            return np.zeros_like(self.windows[0]), reward, True, False, info
        
        # 增加索引前先保存当前窗口，以便在返回前确保索引在有效范围内
        current_window = self.windows[self.idx]
        self.idx += 1
        
        return current_window, reward, False, False, info


    def _calculate_trade_and_fee(self, act: float, price: float, current_margin_equity: float) -> Tuple[float, float]:
        delta_usd = 0.0
        fee = 0.0

        if self.risk_capital_source == "initial_balance":
            risk_capital = self.initial_balance              # sabit
        elif self.risk_capital_source == "cash_balance":
            risk_capital = self.cash_balance                 # dinamik
        elif self.risk_capital_source == "margin_equity":    # ✅ yeni, tavsiye edilen
            risk_capital = self.margin_equity                # dinamik + unrealised PnL
        else:
            raise ValueError("Unsupported risk_capital_source")

        risk_capital = max(risk_capital, 1.0)                # güvenlik tamponu
        
        proposed_delta_usd = act * risk_capital * self.risk_fraction_per_trade

        current_pos_btc_val_signed = self.position_btc * price
        
        if price > 1e-9:
            proposed_delta_btc = proposed_delta_usd / price
            potential_new_pos_btc = self.position_btc + proposed_delta_btc

            if abs(potential_new_pos_btc) > self.max_abs_position_btc_cap:
                if potential_new_pos_btc > self.max_abs_position_btc_cap:
                    allowed_delta_btc = self.max_abs_position_btc_cap - self.position_btc
                else:
                    allowed_delta_btc = -self.max_abs_position_btc_cap - self.position_btc
                proposed_delta_usd = allowed_delta_btc * price
        else:
            proposed_delta_usd = 0.0

        max_allowed_notional_total = current_margin_equity * self.max_leverage
        
        if abs(proposed_delta_usd) > 1e-8:
            current_notional_abs = abs(current_pos_btc_val_signed)
            proposed_new_notional_abs = abs(current_pos_btc_val_signed + proposed_delta_usd)

            if proposed_new_notional_abs > max_allowed_notional_total:
                current_signed_notional = self.position_btc * price
                potential_final_notional = current_signed_notional + proposed_delta_usd
                if potential_final_notional > max_allowed_notional_total:
                    proposed_delta_usd = max_allowed_notional_total - current_signed_notional
                elif potential_final_notional < -max_allowed_notional_total:
                    proposed_delta_usd = -max_allowed_notional_total - current_signed_notional
                if abs(proposed_delta_usd) < 1e-8 : proposed_delta_usd = 0.0

        delta_usd = proposed_delta_usd
        if abs(delta_usd) > 1e-8:
            fee = abs(delta_usd) * self.fee_rate
        else:
            delta_usd = 0.0
            fee = 0.0
            
        return delta_usd, fee

    def _dump_episode(self):
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        out = Path(self.log_dir) / f"{ts}.json"
        out.write_text(json.dumps(self.episode_log, separators=(",", ":")))