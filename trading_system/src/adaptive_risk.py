"""
自适应风险控制模块 - 根据市场状态动态调整风险参数
"""
import numpy as np
import logging
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

class AdaptiveRiskController:
    """
    自适应风险控制器，根据市场状态动态调整交易风险参数
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化风险控制器
        
        参数:
        - config: 配置字典
        """
        self.logger = logging.getLogger("AdaptiveRiskController")
        self.config = config
        
        # 从配置加载基本参数
        self.trading_config = config.get('trading', {})
        self.base_risk_per_trade = self.trading_config.get('risk_per_trade_pct', 0.02)
        self.base_leverage = self.trading_config.get('max_leverage', 3)
        self.base_fee_rate = self.trading_config.get('fee_rate', 0.0002)
        
        # 从训练环境参数加载
        env_risk_fraction = 0.01  # 默认值，与训练环境保持一致
        
        # 市场状态跟踪
        self.market_volatility = 0.0
        self.trend_strength = 0.0
        self.market_regime = "neutral"  # 可能的值: "trending", "ranging", "neutral", "volatile"
        self.volatility_history = []
        self.price_history = []
        
        # 缓存上次更新时间
        self.last_update_time = datetime.now() - timedelta(hours=1)
        self.update_interval = timedelta(minutes=15)  # 每15分钟重新评估市场状态
        
        self.logger.info("自适应风险控制器初始化完成")
        
    def update_market_state(self, market_data: Dict[str, Any]) -> None:
        """
        更新市场状态
        
        参数:
        - market_data: 市场数据，包含OHLCV等信息
        """
        current_time = datetime.now()
        # 仅在达到更新间隔时更新市场状态
        if (current_time - self.last_update_time) < self.update_interval:
            return
            
        self.last_update_time = current_time
        
        # 提取价格
        current_price = market_data.get('close', 0)
        if current_price <= 0:
            self.logger.warning("无效价格数据，跳过市场状态更新")
            return
        
        # 更新价格历史
        self.price_history.append(current_price)
        if len(self.price_history) > 48:  # 保留最近48小时的数据
            self.price_history.pop(0)
        
        # 计算当前波动率 (基于最近价格变化的标准差)
        if len(self.price_history) > 5:
            returns = np.diff(self.price_history) / self.price_history[:-1]
            current_volatility = np.std(returns) * np.sqrt(24)  # 年化到日波动率
            
            self.volatility_history.append(current_volatility)
            if len(self.volatility_history) > 24:  # 保留最近24小时的波动率数据
                self.volatility_history.pop(0)
            
            # 计算平均波动率
            self.market_volatility = np.mean(self.volatility_history)
            
            # 计算趋势强度 (基于价格方向的一致性)
            if len(returns) > 0:
                direction_consistency = np.sum(np.sign(returns)) / len(returns)
                self.trend_strength = abs(direction_consistency)
                
                # 确定市场状态
                if self.market_volatility > 0.03:  # 高波动率
                    if self.trend_strength > 0.6:
                        self.market_regime = "trending"
                    else:
                        self.market_regime = "volatile"
                else:  # 低波动率
                    if self.trend_strength > 0.7:
                        self.market_regime = "trending"
                    else:
                        self.market_regime = "ranging"
            
            self.logger.debug(f"市场状态更新: 状态={self.market_regime}, 波动率={self.market_volatility:.4f}, 趋势强度={self.trend_strength:.4f}")
    
    def get_adjusted_risk_parameters(self) -> Dict[str, Any]:
        """
        获取根据市场状态调整后的风险参数
        
        返回:
        - 调整后的风险参数字典
        """
        # 默认使用基础参数
        adjusted_risk = self.base_risk_per_trade
        adjusted_leverage = self.base_leverage
        
        # 根据市场状态调整参数
        if self.market_regime == "volatile":
            # 高波动市场减少风险
            adjusted_risk = max(0.005, self.base_risk_per_trade * 0.5)
            adjusted_leverage = max(1, self.base_leverage * 0.5)
        elif self.market_regime == "trending":
            # 强趋势市场可以略微增加风险
            adjusted_risk = min(0.03, self.base_risk_per_trade * 1.2)
        elif self.market_regime == "ranging":
            # 区间震荡市场保持中等风险
            adjusted_risk = self.base_risk_per_trade * 0.7
        
        return {
            "risk_per_trade_pct": adjusted_risk,
            "max_leverage": adjusted_leverage,
            "fee_rate": self.base_fee_rate,  # 费率保持不变
            "market_regime": self.market_regime,
            "volatility": self.market_volatility,
            "trend_strength": self.trend_strength
        }
