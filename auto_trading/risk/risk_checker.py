"""
风险检查模块
负责检查交易风险并执行风险控制策略
"""
import logging
import yaml
import time
import os
import json
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import numpy as np

class RiskChecker:
    """
    风险检查类
    检查交易风险并执行风险控制策略
    """
    
    def __init__(self, position_manager, config_path: str = None):
        """
        初始化风险检查器
        
        Args:
            position_manager: 仓位管理器
            config_path: 配置文件路径
        """
        self.logger = logging.getLogger('RiskChecker')
        self.position_manager = position_manager
        
        # 使用传入的配置路径或默认值
        if config_path is None:
            import os as os_module
            from pathlib import Path
            # 使用当前文件的绝对路径找到配置文件
            current_dir = Path(__file__).parent.parent
            config_path = os_module.path.join(current_dir, 'config', 'risk_config.yaml')
            
        self._load_config(config_path)
        
        # 风险状态
        self.risk_status = {
            'circuit_breaker_active': False,
            'circuit_breaker_until': None,
            'consecutive_losses': 0,
            'daily_pnl': {},
            'total_risk_exposure': 0.0,
            'volatility_alerts': {},
            'is_high_volatility': False,
            'last_check_time': {}
        }
        
        # 创建风险日志目录
        import os as os_module
        from pathlib import Path
        current_dir = Path(__file__).parent.parent
        self.risk_log_path = os_module.path.join(current_dir, "logs", "risk.json")
        os_module.makedirs(os_module.path.dirname(self.risk_log_path), exist_ok=True)
    
    def _load_config(self, config_path: str) -> None:
        """
        加载配置
        
        Args:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            risk_config = config.get('risk', {})
            
            # 资金安全配置
            capital_config = risk_config.get('capital', {})
            self.total_capital_limit = capital_config.get('total_capital_limit', 0.5)
            
            # 交易风控配置
            trade_limits = risk_config.get('trade_limits', {})
            self.max_single_loss = trade_limits.get('max_single_loss', 0.03)
            self.max_daily_drawdown = trade_limits.get('max_daily_drawdown', 0.08)
            self.max_open_positions = trade_limits.get('max_open_positions', 3)
            
            # 波动率控制配置
            volatility_config = risk_config.get('volatility', {})
            self.max_acceptable_volatility = volatility_config.get('max_acceptable_volatility', 0.05)
            self.volatility_window = volatility_config.get('volatility_window', 24)
            self.volatility_scaling = volatility_config.get('volatility_scaling', True)
            
            # 熔断机制配置
            circuit_breaker = risk_config.get('circuit_breaker', {})
            self.circuit_breaker_enabled = circuit_breaker.get('enabled', True)
            self.consecutive_loss_limit = circuit_breaker.get('consecutive_loss_limit', 3)
            self.loss_percentage_24h = circuit_breaker.get('loss_percentage_24h', 0.15)
            self.cooling_period = circuit_breaker.get('cooling_period', 24)  # 小时
            
            self.logger.info("成功加载风险控制配置")
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            raise
    
    def check_trading_allowed(self, symbol: str) -> bool:
        """
        检查是否允许交易
        
        Args:
            symbol: 交易对符号
            
        Returns:
            bool: 是否允许交易
        """
        # 检查熔断机制是否激活
        if self.risk_status['circuit_breaker_active']:
            # 如果已过冷却期，则重置熔断状态
            if (self.risk_status['circuit_breaker_until'] and 
                datetime.now() > self.risk_status['circuit_breaker_until']):
                self.risk_status['circuit_breaker_active'] = False
                self.risk_status['circuit_breaker_until'] = None
                self.logger.info("熔断机制已解除，恢复交易")
            else:
                # 如果熔断机制仍在生效期内
                until_str = self.risk_status['circuit_breaker_until'].strftime('%Y-%m-%d %H:%M:%S') if self.risk_status['circuit_breaker_until'] else "无限期"
                self.logger.warning(f"由于熔断机制激活，禁止交易，直到: {until_str}")
                return False
        
        # 检查波动率
        if self.risk_status['is_high_volatility']:
            self.logger.warning(f"{symbol} 市场波动率过高，限制交易")
            return False
        
        # 检查开仓数量是否超过限制
        open_positions = self.position_manager.get_open_positions()
        if len(open_positions) >= self.max_open_positions:
            self.logger.warning(f"已达到最大持仓数量 ({len(open_positions)}/{self.max_open_positions})，禁止开新仓")
            return False
        
        return True
        
    def check_trade_risk(self, symbol: str, action_probas: Dict[int, float], confidence: float, 
                  current_position: float = 0, target_position: float = 0,
                  min_confidence: float = 0.5, trade_type: str = None, 
                  price: float = 0, quantity: float = 0,
                  stop_price: float = None) -> Dict[str, Any]:
        """
        检查具体交易的风险
        
        Args:
            symbol: 交易对符号
            action_probas: 动作概率分布，用于评估模型的置信度 {0: sell_prob, 1: hold_prob, 2: buy_prob}
            confidence: 模型置信度
            current_position: 当前仓位大小
            target_position: 目标仓位大小
            min_confidence: 最低置信度要求
            trade_type: 交易类型 (BUY/SELL)，如果指定了则使用这个
            price: 交易价格
            quantity: 交易数量
            stop_price: 止损价格，如果有的话
            
        Returns:
            Dict[str, Any]: 结果字典，包含'allowed'键表示是否允许交易，'reason'键表示原因
        """
        try:
            self.logger.info(f"检查交易风险: {symbol}, 动作概率={action_probas}, 置信度={confidence}")
            
            # 获取最高概率的动作
            if action_probas:
                highest_prob_action = max(action_probas.items(), key=lambda x: x[1])[0]
                self.logger.debug(f"最高概率动作: {highest_prob_action}")
            
            # 基本规则：置信度必须高于阈值
            if confidence < min_confidence:
                return {
                    "allowed": False,
                    "reason": f"置信度不足: {confidence:.4f} < {min_confidence:.4f}"
                }
                
            # 检查是否允许交易
            if not self.check_trading_allowed(symbol):
                return {"allowed": False, "reason": "交易系统当前不允许交易"}
            
            # 基于模型置信度的风险检查
            if confidence is not None:
                # 如果未提供min_confidence参数，则使用配置文件中的值或默认值
                if min_confidence is None:
                    min_confidence = self.config.get('min_confidence', 0.6)
                
                self.logger.info(f"检查模型置信度: {confidence:.2f}，要求置信度: {min_confidence}")
                if confidence < min_confidence:
                    return {"allowed": False, "reason": f"模型置信度不足 ({confidence:.2f} < {min_confidence})"}
            
            # 如果提供了action_probas，进行基于概率的决策检查
            if action_probas:
                # 检查概率分布是否过于平均（表示模型不确定）
                max_proba = max(action_probas.values()) if action_probas else 0
                if max_proba < 0.5:
                    return {"allowed": False, "reason": f"模型决策不明确，最大概率仅为 {max_proba:.2f}"}
            
            # 如果提供了传统交易参数，进行传统风险检查
            if price and quantity:
                trade_value = price * quantity
                
                # 检查单笔交易损失是否超过限制
                if stop_price and trade_type == "BUY":
                    potential_loss = (price - stop_price) * quantity
                    potential_loss_ratio = potential_loss / trade_value if trade_value > 0 else 0
                    
                    if potential_loss_ratio > self.max_single_loss:
                        return {"allowed": False, "reason": f"潜在损失 ({potential_loss_ratio:.2%}) 超过单笔最大损失限制 ({self.max_single_loss:.2%})"}
                
                # 检查日内回撤是否超过限制
                today = datetime.now().date().isoformat()
                daily_pnl = self.risk_status['daily_pnl'].get(today, 0)
                daily_drawdown = abs(min(0, daily_pnl)) / trade_value if trade_value > 0 else 0
                
                if daily_drawdown > self.max_daily_drawdown:
                    return {"allowed": False, "reason": f"日内回撤 ({daily_drawdown:.2%}) 超过最大日回撤限制 ({self.max_daily_drawdown:.2%})"}
            
            # 检查仓位变化
            if current_position is not None and target_position is not None:
                position_change = abs(target_position - current_position)
                # 如果仓位变化过大，可能风险过高
                if position_change > 0.5 and current_position > 0:  # 如果变化超过50%
                    return {"allowed": False, "reason": f"仓位变化过大 ({position_change:.2%})，超过安全阈值"}
                
                # 如果波动率较高，限制最大仓位
                if self.risk_status.get('is_high_volatility', False):
                    max_position_size = 0.5  # 高波动率时最大仓位降低
                    if target_position > max_position_size:
                        return {"allowed": False, "reason": f"高波动环境下，目标仓位 ({target_position:.2f}) 超过最大限制 ({max_position_size})"}
            
            return {"allowed": True, "reason": "交易风险检查通过"}
            
        except Exception as e:
            self.logger.error(f"检查交易风险时出错: {e}")
            return {"allowed": False, "reason": f"风险检查异常: {str(e)}"}
    
    def _check_risk_exposure(self) -> bool:
        """
        检查风险敞口
        
        Returns:
            bool: 是否在允许范围内
        """
        try:
            # 获取仓位摘要
            position_summary = self.position_manager.get_position_summary()
            
            # 计算当前使用的资金比例
            total_margin_balance = position_summary['total_margin_balance']
            total_position_value = position_summary['total_position_value']
            
            if total_margin_balance <= 0:
                return False
                
            current_exposure = total_position_value / total_margin_balance
            self.risk_status['total_risk_exposure'] = current_exposure
            
            # 检查是否超过限制
            if current_exposure > self.total_capital_limit:
                self.logger.info(f"风险敞口超过限制: {current_exposure:.2f} > {self.total_capital_limit}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"检查风险敞口失败: {e}")
            return False
    
    def _check_volatility(self, symbol: str) -> bool:
        """
        检查市场波动性
        
        Args:
            symbol: 交易对
            
        Returns:
            bool: 波动性是否在允许范围内
        """
        try:
            # 这里需要接入市场数据，获取波动率
            # 简化实现，假设使用仓位信息中的波动率
            position = self.position_manager.get_position(symbol)
            
            # 如果没有波动率数据，默认允许交易
            if 'volatility' not in position:
                return True
                
            volatility = position.get('volatility', 0)
            
            # 更新风险状态
            self.risk_status['volatility_alerts'][symbol] = volatility
            
            # 检查是否超过最大可接受波动率
            if volatility > self.max_acceptable_volatility:
                self.logger.info(f"波动率过高: {symbol}, {volatility:.4f} > {self.max_acceptable_volatility}")
                self.risk_status['is_high_volatility'] = True
                return False
            
            self.risk_status['is_high_volatility'] = False
            return True
            
        except Exception as e:
            self.logger.error(f"检查波动性失败: {symbol}: {e}")
            return True  # 出错时默认允许交易
    
    def _check_daily_drawdown(self) -> bool:
        """
        检查当日回撤
        
        Returns:
            bool: 是否在允许范围内
        """
        try:
            today = datetime.now().date().isoformat()
            
            # 获取当日已实现和未实现盈亏
            daily_pnl = self.risk_status['daily_pnl'].get(today, 0)
            
            # 获取账户总余额
            position_summary = self.position_manager.get_position_summary()
            total_margin_balance = position_summary['total_margin_balance']
            
            # 添加未实现盈亏
            daily_pnl += position_summary['total_unrealized_pnl']
            
            # 更新当日盈亏
            self.risk_status['daily_pnl'][today] = daily_pnl
            
            # 计算当日回撤比例
            if daily_pnl < 0 and total_margin_balance > 0:
                drawdown_ratio = abs(daily_pnl) / total_margin_balance
                
                # 检查是否超过最大日回撤
                if drawdown_ratio > self.max_daily_drawdown:
                    self.logger.warning(f"当日回撤超过限制: {drawdown_ratio:.4f} > {self.max_daily_drawdown}")
                    
                    # 如果亏损超过一定比例，激活熔断机制
                    if drawdown_ratio > self.loss_percentage_24h and self.circuit_breaker_enabled:
                        self._activate_circuit_breaker(f"当日回撤过大: {drawdown_ratio:.4f}")
                    
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"检查当日回撤失败: {e}")
            return True  # 出错时默认允许交易
    
    def _check_position_limits(self) -> bool:
        """
        检查持仓限制
        
        Returns:
            bool: 是否在允许范围内
        """
        try:
            # 获取当前持仓数量
            position_summary = self.position_manager.get_position_summary()
            position_count = position_summary['position_count']
            
            # 检查是否超过最大持仓数量
            if position_count >= self.max_open_positions:
                self.logger.info(f"已达到最大持仓数量: {position_count}/{self.max_open_positions}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"检查持仓限制失败: {e}")
            return True  # 出错时默认允许交易
    
    def update_trade_result(self, symbol: str, profit: float, is_closed: bool = True) -> None:
        """
        更新交易结果
        
        Args:
            symbol: 交易对
            profit: 盈亏金额
            is_closed: 是否已平仓
        """
        try:
            today = datetime.now().date().isoformat()
            
            # 更新当日盈亏
            if today not in self.risk_status['daily_pnl']:
                self.risk_status['daily_pnl'][today] = 0
                
            self.risk_status['daily_pnl'][today] += profit
            
            # 更新连续亏损计数
            if profit < 0:
                self.risk_status['consecutive_losses'] += 1
                
                # 检查是否触发熔断机制
                if (self.circuit_breaker_enabled and 
                    self.risk_status['consecutive_losses'] >= self.consecutive_loss_limit):
                    self._activate_circuit_breaker(f"连续亏损: {self.risk_status['consecutive_losses']}")
            else:
                self.risk_status['consecutive_losses'] = 0
            
            # 记录风险日志
            self._log_risk_status()
            
        except Exception as e:
            self.logger.error(f"更新交易结果失败: {symbol}: {e}")
    
    def _activate_circuit_breaker(self, reason: str) -> None:
        """
        激活熔断机制
        
        Args:
            reason: 激活原因
        """
        if not self.circuit_breaker_enabled:
            return
            
        self.logger.warning(f"激活熔断机制: {reason}")
        
        self.risk_status['circuit_breaker_active'] = True
        self.risk_status['circuit_breaker_until'] = datetime.now() + timedelta(hours=self.cooling_period)
        
        # 记录风险日志
        self._log_risk_status()
    
    def _log_risk_status(self) -> None:
        """记录风险状态"""
        try:
            # 读取现有日志
            risk_logs = []
            if os.path.exists(self.risk_log_path):
                try:
                    with open(self.risk_log_path, 'r') as f:
                        risk_logs = json.load(f)
                except json.JSONDecodeError:
                    risk_logs = []
            
            # 准备记录内容
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'circuit_breaker_active': self.risk_status['circuit_breaker_active'],
                'circuit_breaker_until': self.risk_status['circuit_breaker_until'].isoformat()
                    if self.risk_status['circuit_breaker_until'] else None,
                'consecutive_losses': self.risk_status['consecutive_losses'],
                'total_risk_exposure': self.risk_status['total_risk_exposure'],
                'is_high_volatility': self.risk_status['is_high_volatility'],
                'daily_pnl': self.risk_status['daily_pnl']
            }
            
            risk_logs.append(log_entry)
            
            # 如果日志太长，移除最早的记录
            max_log_entries = 1000
            while len(risk_logs) > max_log_entries:
                risk_logs.pop(0)
            
            # 写入日志文件
            with open(self.risk_log_path, 'w') as f:
                json.dump(risk_logs, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"记录风险状态失败: {e}")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """
        获取风险摘要
        
        Returns:
            Dict[str, Any]: 风险摘要
        """
        today = datetime.now().date().isoformat()
        
        return {
            'circuit_breaker_active': self.risk_status['circuit_breaker_active'],
            'circuit_breaker_until': self.risk_status['circuit_breaker_until'].isoformat() 
                if self.risk_status['circuit_breaker_until'] else None,
            'consecutive_losses': self.risk_status['consecutive_losses'],
            'daily_pnl': self.risk_status['daily_pnl'].get(today, 0),
            'total_risk_exposure': self.risk_status['total_risk_exposure'],
            'is_high_volatility': self.risk_status['is_high_volatility'],
            'timestamp': datetime.now().isoformat()
        }
