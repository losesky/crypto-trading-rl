"""
资金管理模块
负责交易资金分配和风险敞口管理
"""
import logging
import yaml
import os
import json
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import numpy as np

class MoneyManager:
    """
    资金管理类
    负责交易资金分配和仓位规模计算
    """
    
    def __init__(self, config_path: str = "../config/risk_config.yaml"):
        """
        初始化资金管理器
        
        Args:
            config_path: 风控配置文件路径
        """
        self.logger = logging.getLogger('MoneyManager')
        self._load_config(config_path)
        
        # 资金状态
        self.capital_status = {
            'total_capital': 0.0,  # 总资金
            'available_capital': 0.0,  # 可用资金
            'allocated_capital': 0.0,  # 已分配资金
            'reserved_capital': 0.0,  # 预留资金（不参与交易）
            'daily_pnl': 0.0,  # 每日盈亏
            'total_pnl': 0.0,  # 总盈亏
            'drawdown': 0.0,  # 当前回撤
            'max_drawdown': 0.0,  # 最大回撤
            'last_update_time': None  # 最后更新时间
        }
        
        # 创建资金状态日志目录
        self.capital_log_path = "../logs/capital_status.json"
        os.makedirs(os.path.dirname(self.capital_log_path), exist_ok=True)
        
        # 尝试加载保存的资金状态
        self._load_capital_status()
    
    def _load_config(self, config_path: str) -> None:
        """
        加载风控配置
        
        Args:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 加载资金管理配置
            risk_config = config.get('risk', {})
            self.capital_config = risk_config.get('capital', {})
            self.trade_limits = risk_config.get('trade_limits', {})
            self.position_config = risk_config.get('position_management', {})
            
            self.logger.info("资金管理配置加载成功")
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            # 设置默认值
            self.capital_config = {
                'total_capital_limit': 0.5,  # 最大使用总资金的50%
                'initial_position_size': 0.02,  # 初始仓位大小为总资金的2%
                'max_position_size': 0.1  # 最大仓位大小为总资金的10%
            }
            self.trade_limits = {
                'max_single_loss': 0.03,  # 单次交易最大亏损限制(总资金的3%)
                'max_daily_drawdown': 0.08,  # 日最大回撤限制(总资金的8%)
                'min_trade_interval': 3600,  # 最小交易间隔(秒)
                'max_daily_trades': 24,  # 24小时最大交易次数
                'max_open_positions': 3  # 最大同时持仓数量
            }
            self.position_config = {
                'max_holding_time': 72,  # 最大持仓时间(小时)
                'pyramid_scaling': True,  # 是否启用金字塔加仓
                'scale_in_steps': 3,  # 加仓步数
                'scale_out_steps': 2  # 减仓步数
            }

    def initialize_capital(self, total_capital: float) -> None:
        """
        初始化资金状态
        
        Args:
            total_capital: 总资金
        """
        # 初始化资金状态
        self.capital_status['total_capital'] = total_capital
        self.capital_status['available_capital'] = total_capital * self.capital_config['total_capital_limit']
        self.capital_status['reserved_capital'] = total_capital * (1 - self.capital_config['total_capital_limit'])
        self.capital_status['allocated_capital'] = 0.0
        self.capital_status['last_update_time'] = datetime.now().isoformat()
        
        self.logger.info(f"资金状态初始化成功: 总资金 {total_capital}, 可用资金 {self.capital_status['available_capital']}")
        self._save_capital_status()
    
    def set_capital(self, total_capital: float, available_capital: Optional[float] = None) -> None:
        """
        设置资金状态，用于从账户信息中更新资金
        
        Args:
            total_capital: 总资金
            available_capital: 可用资金，如果为None，则根据配置比例计算
        """
        if total_capital <= 0:
            self.logger.warning(f"设置资金状态失败: 总资金必须大于0，收到: {total_capital}")
            # 设置一个默认值，避免后续除零错误
            total_capital = 10000.0
            available_capital = 10000.0 * self.capital_config.get('total_capital_limit', 0.5)
        
        # 设置总资金
        self.capital_status['total_capital'] = total_capital
        
        # 设置可用资金
        if available_capital is None:
            available_capital = total_capital * self.capital_config.get('total_capital_limit', 0.5)
        
        self.capital_status['available_capital'] = available_capital
        self.capital_status['reserved_capital'] = total_capital - available_capital
        self.capital_status['last_update_time'] = datetime.now().isoformat()
        
        self.logger.info(f"资金状态设置成功: 总资金 {total_capital}, 可用资金 {available_capital}")
        self._save_capital_status()
    
    def update_dashboard(self, dashboard) -> None:
        """
        将资金管理器的状态更新到监控面板
        
        Args:
            dashboard: Dashboard实例
        """
        try:
            # 更新资金分配信息
            capital_data = {
                'total': self.capital_status['total_capital'],
                'available': self.capital_status['available_capital'],
                'allocated': self.capital_status['allocated_capital'],
                'reserved': self.capital_status['reserved_capital']
            }
            
            # 更新到Dashboard
            dashboard.update_capital_allocation(capital_data)
            
            # 更新绩效指标
            performance_data = {
                'daily_pnl': self.capital_status['daily_pnl'],
                'total_pnl': self.capital_status['total_pnl'],
                'drawdown': self.capital_status['drawdown'],
                'max_drawdown': self.capital_status['max_drawdown']
            }
            
            dashboard.update_performance_metrics(performance_data)
            
            self.logger.debug("已更新资金状态到监控面板")
        except Exception as e:
            self.logger.error(f"更新资金状态到监控面板失败: {e}")
    
    def _save_capital_status(self) -> None:
        """
        将当前资金状态保存到文件
        方便系统重启时恢复状态
        """
        try:
            data_to_save = self.capital_status.copy()
            data_to_save['saved_time'] = datetime.now().isoformat()
            
            with open(self.capital_log_path, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2)
                
            self.logger.debug(f"资金状态已保存到 {self.capital_log_path}")
        except Exception as e:
            self.logger.error(f"保存资金状态失败: {e}")
    
    def _load_capital_status(self) -> None:
        """
        从文件加载保存的资金状态
        """
        try:
            if os.path.exists(self.capital_log_path):
                with open(self.capital_log_path, 'r', encoding='utf-8') as f:
                    saved_status = json.load(f)
                    
                # 更新状态，但保留当前的时间戳
                last_update_time = self.capital_status['last_update_time']
                self.capital_status.update(saved_status)
                self.capital_status['last_update_time'] = last_update_time
                
                self.logger.info(f"已从文件加载资金状态: 总资金 {self.capital_status['total_capital']}, "
                                f"可用资金 {self.capital_status['available_capital']}")
        except Exception as e:
            self.logger.warning(f"加载资金状态失败: {e}")
    
    def calculate_position_size(self, symbol: str, confidence: float, volatility: float = None, suggested_size: float = None) -> Tuple[float, float]:
        """
        计算最佳仓位规模
        
        Args:
            symbol: 交易对
            confidence: 模型置信度(0-1)
            volatility: 市场波动率
            suggested_size: 建议仓位大小（由模型给出）
            
        Returns:
            Tuple[float, float]: (仓位大小占总资金比例, 仓位大小(USDT))
        """
        # 基础仓位大小
        base_size = self.capital_config['initial_position_size']
        
        # 根据置信度调整仓位大小
        conf_factor = self._calculate_confidence_factor(confidence)
        
        # 根据波动率调整仓位大小
        vol_factor = 1.0
        if volatility is not None:
            vol_factor = self._calculate_volatility_factor(volatility)
        
        # 计算最终仓位比例
        position_ratio = min(base_size * conf_factor * vol_factor, self.capital_config['max_position_size'])
        
        # 确保不超过可用资金
        available_ratio = (self.capital_status['available_capital'] - self.capital_status['allocated_capital']) / self.capital_status['total_capital']
        position_ratio = min(position_ratio, available_ratio)
        
        # 计算仓位大小(USDT)
        position_size_usdt = position_ratio * self.capital_status['total_capital']
        
        # 如果有模型建议的仓位大小，考虑其影响
        if suggested_size is not None:
            # 将模型建议值作为参考，与我们计算的值取加权平均
            # 可以根据需要调整权重
            model_weight = 0.3  # 给模型建议的权重
            position_size_usdt = (1 - model_weight) * position_size_usdt + model_weight * suggested_size
            
            # 重新计算仓位比例
            position_ratio = position_size_usdt / self.capital_status['total_capital']
            
            self.logger.info(f"结合模型建议调整仓位: 模型建议={suggested_size:.2f}, 最终仓位={position_size_usdt:.2f} USDT")
        
        self.logger.info(f"计算仓位大小: 符号={symbol}, 置信度={confidence:.4f}, 波动率={volatility}, "
                        f"仓位比例={position_ratio:.4f}, 仓位大小={position_size_usdt:.2f} USDT")
        
        return position_ratio, position_size_usdt
    
    def _calculate_confidence_factor(self, confidence: float) -> float:
        """
        根据模型置信度计算仓位因子
        
        Args:
            confidence: 模型置信度(0-1)
            
        Returns:
            float: 置信度因子(0.1-1.5)
        """
        # 低于0.6的置信度会减少仓位，高于0.8的会增加仓位
        if confidence < 0.6:
            return 0.1 + 1.5 * confidence  # 0.1 - 1.0
        elif confidence > 0.8:
            return 1.0 + 0.5 * (confidence - 0.8) / 0.2  # 1.0 - 1.5
        else:
            return 1.0  # 0.6-0.8时使用标准仓位
    
    def _calculate_volatility_factor(self, volatility: float) -> float:
        """
        根据波动率计算仓位因子
        
        Args:
            volatility: 市场波动率
            
        Returns:
            float: 波动率因子(0.5-1.0)
        """
        # 波动率越高，仓位越小
        max_volatility = self.capital_config.get('max_volatility', 0.05)
        if volatility > max_volatility:
            return 0.5
        else:
            # 线性映射: 波动率越高，返回值越小
            return 1.0 - 0.5 * (volatility / max_volatility)
    
    def calculate_pyramid_sizes(self, initial_size: float, direction: str, steps: int = None) -> List[float]:
        """
        计算金字塔加仓/减仓策略的仓位大小序列
        
        Args:
            initial_size: 初始仓位大小(USDT)
            direction: 'scale_in'表示加仓, 'scale_out'表示减仓
            steps: 步数，默认使用配置文件中的值
            
        Returns:
            List[float]: 仓位大小序列(USDT)
        """
        if direction == 'scale_in':
            steps = steps or self.position_config['scale_in_steps']
            # 加仓策略：递减比例 (如: 100% -> 60% -> 40%)
            ratios = [1.0]
            for i in range(1, steps):
                ratios.append(ratios[-1] * 0.6)  # 每次加仓为前一次的60%
            
            # 归一化确保总和不超过最大仓位
            total = sum(ratios)
            normalized_ratios = [r / total for r in ratios]
            
            # 计算每步仓位大小
            return [initial_size * ratio for ratio in normalized_ratios]
            
        elif direction == 'scale_out':
            steps = steps or self.position_config['scale_out_steps']
            # 减仓策略：先减少大部分，再减少小部分
            if steps <= 1:
                return [initial_size]
            
            # 例如：先减70%，再减30%
            base_ratio = 0.7
            first_step = initial_size * base_ratio
            remain = initial_size - first_step
            
            if steps == 2:
                return [first_step, remain]
            else:
                # 如果步数大于2，则将剩余部分均分
                result = [first_step]
                for i in range(steps - 1):
                    result.append(remain / (steps - 1))
                return result
        else:
            self.logger.error(f"无效的方向: {direction}, 必须是'scale_in'或'scale_out'")
            return [initial_size]  # 默认返回初始大小

    def update_after_trade(self, trade_pnl: float, allocated_capital_change: float) -> None:
        """
        交易后更新资金状态
        
        Args:
            trade_pnl: 交易盈亏(USDT)
            allocated_capital_change: 已分配资金变化(USDT)
        """
        # 更新资金状态
        self.capital_status['daily_pnl'] += trade_pnl
        self.capital_status['total_pnl'] += trade_pnl
        
        # 更新总资金
        self.capital_status['total_capital'] += trade_pnl
        
        # 更新已分配资金
        self.capital_status['allocated_capital'] += allocated_capital_change
        
        # 更新可用资金
        self.capital_status['available_capital'] = (self.capital_status['total_capital'] * 
                                                 self.capital_config['total_capital_limit'] - 
                                                 self.capital_status['allocated_capital'])
        
        # 计算回撤
        peak_capital = max(self.capital_status['total_capital'], 
                          self.capital_status['total_capital'] - self.capital_status['total_pnl'])
        
        current_drawdown = (peak_capital - self.capital_status['total_capital']) / peak_capital if peak_capital > 0 else 0
        self.capital_status['drawdown'] = current_drawdown
        self.capital_status['max_drawdown'] = max(self.capital_status['max_drawdown'], current_drawdown)
        
        self.capital_status['last_update_time'] = datetime.now().isoformat()
        
        self.logger.info(f"资金状态更新: 交易盈亏={trade_pnl:.2f} USDT, "
                       f"已分配资金变化={allocated_capital_change:.2f} USDT, "
                       f"总资金={self.capital_status['total_capital']:.2f} USDT, "
                       f"可用资金={self.capital_status['available_capital']:.2f} USDT")
        
        # 保存资金状态
        self._save_capital_status()
    
    def daily_reset(self) -> None:
        """重置每日统计数据"""
        self.capital_status['daily_pnl'] = 0.0
        self.logger.info("重置每日资金统计数据")
        self._save_capital_status()
    
    def get_capital_status(self) -> Dict:
        """获取当前资金状态"""
        return self.capital_status.copy()
    
    def check_max_drawdown(self) -> bool:
        """
        检查是否超过最大回撤限制
        
        Returns:
            bool: True如果未超过限制，False如果已超过限制
        """
        if self.capital_status['drawdown'] > self.trade_limits['max_daily_drawdown']:
            self.logger.warning(f"超过日最大回撤限制: {self.capital_status['drawdown']:.4f} > {self.trade_limits['max_daily_drawdown']}")
            return False
        return True
