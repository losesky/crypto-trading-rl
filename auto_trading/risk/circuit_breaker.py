"""
熔断机制模块
负责检测并执行交易系统的熔断机制
"""
import logging
import yaml
import os
import json
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import numpy as np

class CircuitBreaker:
    """
    熔断机制类
    检测系统异常情况并执行熔断，暂停交易以避免连续亏损
    """
    
    def __init__(self, config_path: str = "../config/risk_config.yaml"):
        """
        初始化熔断机制
        
        Args:
            config_path: 风控配置文件路径
        """
        self.logger = logging.getLogger('CircuitBreaker')
        self._load_config(config_path)
        
        # 熔断状态
        self.circuit_status = {
            'is_active': False,  # 熔断是否激活
            'activated_at': None,  # 熔断激活时间
            'cooling_until': None,  # 冷却期结束时间
            'trigger_reason': None,  # 触发原因
            'trade_history': [],  # 最近交易历史，用于检查连续亏损
            'daily_loss': 0.0,  # 24小时内亏损金额
            'daily_trades': 0,  # 24小时内交易次数
            'alert_level': 'normal',  # normal, warning, danger
            'status_changes': []  # 状态变更历史
        }
        
        # 创建熔断日志目录
        self.circuit_log_path = "../logs/circuit_breaker.json"
        os.makedirs(os.path.dirname(self.circuit_log_path), exist_ok=True)
        
        # 加载历史熔断状态(如果存在)
        self._load_circuit_status()
    
    def initialize(self) -> None:
        """
        初始化熔断机制，在系统启动时调用
        检查配置是否有效，重新加载熔断状态，并准备开始监控
        """
        self.logger.info("初始化熔断机制")
        
        # 重新加载熔断状态
        self._load_circuit_status()
        
        # 检查是否仍在冷却期内
        if self.is_breaker_active():
            cooling_time = datetime.fromisoformat(self.circuit_status['cooling_until'])
            self.logger.warning(f"交易系统启动时熔断机制处于激活状态，冷却期至 {cooling_time.isoformat()}")
        
        # 记录初始化状态
        status_change = {
            'timestamp': datetime.now().isoformat(),
            'action': 'initialize',
            'reason': 'system_startup'
        }
        self.circuit_status['status_changes'].append(status_change)
        
        # 更新和保存状态
        self._update_daily_stats()
        self._update_alert_level()
        self._save_circuit_status()
        
        self.logger.info(f"熔断机制初始化完成，当前状态: {'激活' if self.is_breaker_active() else '正常'}")
    
    def _load_config(self, config_path: str) -> None:
        """
        加载风控配置
        
        Args:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 加载熔断机制配置
            risk_config = config.get('risk', {})
            self.circuit_config = risk_config.get('circuit_breaker', {})
            
            self.logger.info("熔断机制配置加载成功")
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            # 设置默认值
            self.circuit_config = {
                'enabled': True,
                'consecutive_loss_limit': 3,
                'loss_percentage_24h': 0.15,
                'cooling_period': 24
            }
    
    def _load_circuit_status(self) -> None:
        """加载历史熔断状态"""
        try:
            if os.path.exists(self.circuit_log_path):
                with open(self.circuit_log_path, 'r', encoding='utf-8') as f:
                    saved_status = json.load(f)
                
                # 检查冷却期是否仍然有效
                if saved_status.get('cooling_until'):
                    cooling_time = datetime.fromisoformat(saved_status['cooling_until'])
                    if cooling_time > datetime.now():
                        self.circuit_status['is_active'] = True
                        self.circuit_status['cooling_until'] = saved_status['cooling_until']
                        self.circuit_status['activated_at'] = saved_status['activated_at']
                        self.circuit_status['trigger_reason'] = saved_status['trigger_reason']
                        
                        self.logger.warning(f"加载熔断状态: 熔断仍然激活，冷却期至 {cooling_time.isoformat()}")
                
                # 加载交易历史和其他状态
                if 'trade_history' in saved_status:
                    # 只保留最近24小时的交易记录
                    recent_trades = []
                    for trade in saved_status['trade_history']:
                        if datetime.fromisoformat(trade['timestamp']) > datetime.now() - timedelta(hours=24):
                            recent_trades.append(trade)
                    self.circuit_status['trade_history'] = recent_trades
                
                self.logger.info("熔断状态加载成功")
        except Exception as e:
            self.logger.error(f"加载熔断状态失败: {e}")
    
    def is_breaker_active(self) -> bool:
        """
        检查熔断机制是否处于激活状态
        
        Returns:
            bool: 熔断是否激活
        """
        if not self.circuit_status['is_active']:
            return False
            
        # 检查冷却期是否已过
        if self.circuit_status['cooling_until']:
            cooling_until = datetime.fromisoformat(self.circuit_status['cooling_until'])
            if datetime.now() > cooling_until:
                # 冷却期已过，重置熔断状态
                self._reset_circuit()
                return False
                
        return True
    
    def _reset_circuit(self) -> None:
        """重置熔断状态"""
        self.circuit_status['is_active'] = False
        self.circuit_status['activated_at'] = None
        self.circuit_status['cooling_until'] = None
        self.circuit_status['trigger_reason'] = None
        self.circuit_status['alert_level'] = 'normal'
        
        # 记录状态变更
        status_change = {
            'timestamp': datetime.now().isoformat(),
            'action': 'reset',
            'reason': 'cooling_period_ended'
        }
        self.circuit_status['status_changes'].append(status_change)
        
        # 保存状态
        self._save_circuit_status()
    
    def record_trade(self, trade_info: Dict) -> None:
        """
        记录交易信息，用于检查连续亏损和日亏损
        
        Args:
            trade_info: 交易信息，包含symbol, side, amount, entry_price, exit_price, pnl等
        """
        # 添加时间戳
        trade_info['timestamp'] = datetime.now().isoformat()
        
        # 更新交易历史
        self.circuit_status['trade_history'].append(trade_info)
        
        # 只保留最近100条交易记录
        if len(self.circuit_status['trade_history']) > 100:
            self.circuit_status['trade_history'] = self.circuit_status['trade_history'][-100:]
        
        # 更新日亏损和交易次数
        self._update_daily_stats()
        
        # 检查是否需要触发熔断
        self._check_circuit_breaker_conditions()
        
        # 保存熔断状态
        self._save_circuit_status()
    
    def _update_daily_stats(self) -> None:
        """更新日亏损和交易次数"""
        now = datetime.now()
        daily_loss = 0.0
        daily_trades = 0
        
        # 清理过期交易记录并计算24小时总亏损
        valid_trades = []
        for trade in self.circuit_status['trade_history']:
            trade_time = datetime.fromisoformat(trade['timestamp'])
            if now - trade_time <= timedelta(hours=24):
                valid_trades.append(trade)
                daily_trades += 1
                pnl = trade.get('pnl', 0)
                if pnl < 0:
                    daily_loss += abs(pnl)
                    
        # 更新状态
        self.circuit_status['trade_history'] = valid_trades
        self.circuit_status['daily_loss'] = daily_loss
        self.circuit_status['daily_trades'] = daily_trades
    
    def _check_circuit_breaker_conditions(self) -> bool:
        """
        检查是否需要触发熔断
        
        Returns:
            bool: 是否触发熔断
        """
        # 熔断已激活，无需再次检查
        if self.circuit_status['is_active']:
            return True
        
        # 检查连续亏损
        consecutive_losses = self._check_consecutive_losses()
        if (consecutive_losses >= self.circuit_config.get('consecutive_loss_limit', 3) and 
            consecutive_losses > 0):
            self._activate_circuit_breaker(f"连续亏损 {consecutive_losses} 次")
            return True
        
        # 检查日亏损比例
        if self._check_daily_loss_percentage():
            self._activate_circuit_breaker(f"24小时内亏损超过阈值 {self.circuit_config.get('loss_percentage_24h', 0.15) * 100}%")
            return True
        
        # 更新告警级别
        self._update_alert_level()
        
        return False
    
    def _check_consecutive_losses(self) -> int:
        """
        检查连续亏损次数
        
        Returns:
            int: 连续亏损次数
        """
        count = 0
        # 倒序遍历交易历史
        for trade in reversed(self.circuit_status['trade_history']):
            pnl = trade.get('pnl', 0)
            if pnl < 0:
                count += 1
            else:
                break  # 遇到盈利交易，终止计数
        return count
    
    def _check_daily_loss_percentage(self) -> bool:
        """
        检查日亏损比例是否超过阈值
        
        Returns:
            bool: 是否超过阈值
        """
        # 从交易记录中获取总资金
        total_capital = 0
        for trade in self.circuit_status['trade_history']:
            if 'total_capital' in trade:
                total_capital = trade['total_capital']
                break
        
        # 如果没有总资金信息，无法计算比例
        if total_capital == 0:
            return False
        
        # 计算亏损比例
        loss_percentage = self.circuit_status['daily_loss'] / total_capital
        
        # 检查是否超过阈值
        threshold = self.circuit_config.get('loss_percentage_24h', 0.15)
        if loss_percentage >= threshold:
            self.logger.warning(f"24小时亏损比例 {loss_percentage:.4f} 超过阈值 {threshold}")
            return True
        
        return False
    
    def _activate_circuit_breaker(self, reason: str) -> None:
        """
        激活熔断
        
        Args:
            reason: 触发原因
        """
        now = datetime.now()
        cooling_hours = self.circuit_config.get('cooling_period', 24)
        cooling_until = now + timedelta(hours=cooling_hours)
        
        self.circuit_status['is_active'] = True
        self.circuit_status['activated_at'] = now.isoformat()
        self.circuit_status['cooling_until'] = cooling_until.isoformat()
        self.circuit_status['trigger_reason'] = reason
        self.circuit_status['alert_level'] = 'danger'
        
        status_change = {
            'timestamp': now.isoformat(),
            'action': 'activate',
            'reason': reason,
            'cooling_until': cooling_until.isoformat()
        }
        self.circuit_status['status_changes'].append(status_change)
        
        self.logger.warning(f"熔断机制已激活: 原因={reason}, 冷却期结束时间={cooling_until.isoformat()}")
        
        # 保存熔断状态
        self._save_circuit_status()
    
    def _update_alert_level(self) -> None:
        """更新告警级别"""
        consecutive_losses = self._check_consecutive_losses()
        loss_limit = self.circuit_config.get('consecutive_loss_limit', 3)
        
        # 设置告警级别
        if consecutive_losses >= loss_limit - 1:
            self.circuit_status['alert_level'] = 'danger'
        elif consecutive_losses >= loss_limit / 2:
            self.circuit_status['alert_level'] = 'warning'
        else:
            self.circuit_status['alert_level'] = 'normal'
    
    def get_status(self) -> Dict:
        """
        获取当前熔断状态
        
        Returns:
            Dict: 当前熔断状态
        """
        return {
            'is_active': self.circuit_status['is_active'],
            'activated_at': self.circuit_status['activated_at'],
            'cooling_until': self.circuit_status['cooling_until'],
            'trigger_reason': self.circuit_status['trigger_reason'],
            'daily_loss': self.circuit_status['daily_loss'],
            'daily_trades': self.circuit_status['daily_trades'],
            'alert_level': self.circuit_status['alert_level'],
            'consecutive_losses': self._check_consecutive_losses()
        }
    
    def manual_reset(self) -> bool:
        """
        手动重置熔断状态
        
        Returns:
            bool: 是否重置成功
        """
        if not self.circuit_status['is_active']:
            self.logger.info("熔断未激活，无需重置")
            return False
        
        self._reset_circuit()
        self.logger.info("熔断状态已手动重置")
        return True
    
    def _save_circuit_status(self) -> None:
        """保存熔断状态到日志文件"""
        try:
            # 添加时间戳
            status_with_time = self.circuit_status.copy()
            status_with_time['last_updated'] = datetime.now().isoformat()
            
            with open(self.circuit_log_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(status_with_time, indent=2))
        except Exception as e:
            self.logger.error(f"保存熔断状态失败: {e}")
    
    def check_status(self) -> bool:
        """
        检查熔断状态，判断是否允许交易
        
        Returns:
            bool: 如果返回True则表示可以交易，False表示熔断中不可交易
        """
        # 检查熔断是否激活
        if self.is_breaker_active():
            cooling_time = datetime.fromisoformat(self.circuit_status['cooling_until'])
            self.logger.warning(f"熔断状态检查: 熔断激活中，冷却期至 {cooling_time.isoformat()}")
            return False
        
        return True
    
    def check_trade(self, symbol: str, position_delta: float, positions: Dict[str, Any] = None) -> bool:
        """
        检查特定交易是否可以执行，根据熔断状态和交易特征判断
        
        Args:
            symbol: 交易对
            position_delta: 仓位变化量（正值表示开仓，负值表示平仓）
            positions: 当前持仓信息
            
        Returns:
            bool: 如果返回True则表示允许交易，False表示不允许交易
        """
        # 首先检查整体熔断状态
        if not self.check_status():
            self.logger.warning(f"交易检查: {symbol} 熔断激活中，不允许交易")
            return False
        
        # 检查交易对特定风险
        if self._is_symbol_restricted(symbol):
            self.logger.warning(f"交易检查: {symbol} 在受限交易对列表中")
            return False
        
        # 检查仓位大小是否超过限制
        if position_delta > 0:  # 开仓
            max_single_position = self.circuit_config.get('max_single_position', 0.2)
            total_capital = self.circuit_status.get('total_capital', 10000)  # 默认值，应该从其他地方获取实际值
            
            if position_delta > total_capital * max_single_position:
                self.logger.warning(f"交易检查: {symbol} 开仓量 {position_delta} 超过单笔最大限制 {total_capital * max_single_position}")
                return False
        
        # 检查持仓数量是否已达上限
        if positions and position_delta > 0:  # 如果是开仓
            active_positions = sum(1 for pos in positions.values() if pos.get('amount', 0) > 0)
            max_positions = self.circuit_config.get('max_positions', 5)
            
            if active_positions >= max_positions:
                self.logger.warning(f"交易检查: 当前持仓数 {active_positions} 已达上限 {max_positions}")
                return False
        
        # 市场剧烈波动时的额外检查
        if self._is_market_volatile(symbol) and position_delta > 0:
            self.logger.warning(f"交易检查: {symbol} 市场波动剧烈，降低开仓量或禁止开仓")
            # 这里可以根据情况返回False，或者在调用方减少仓位大小
            return False
        
        # 通过所有检查，允许交易
        return True
    
    def _is_symbol_restricted(self, symbol: str) -> bool:
        """
        检查交易对是否在受限列表中
        
        Args:
            symbol: 交易对
            
        Returns:
            bool: 如果返回True则表示受限，False表示不受限
        """
        restricted_symbols = self.circuit_config.get('restricted_symbols', [])
        return symbol in restricted_symbols
    
    def _is_market_volatile(self, symbol: str) -> bool:
        """
        检查市场是否剧烈波动
        
        Args:
            symbol: 交易对
            
        Returns:
            bool: 如果返回True则表示市场剧烈波动，False表示市场正常
        """
        # 这里应该实现检查市场波动性的逻辑
        # 可以使用外部数据源或者通过计算短期价格变化率来判断
        # 简单实现：如果警报级别是"danger"，认为市场剧烈波动
        return self.circuit_status.get('alert_level', 'normal') == 'danger'
