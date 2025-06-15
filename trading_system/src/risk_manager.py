"""
风险管理模块 - 负责实时监控和管理交易风险
"""
import logging
import time
import threading
import json
from datetime import datetime, timedelta
import numpy as np

class RiskManager:
    """
    风险管理器，负责监控和管理交易风险
    包括资金分配、最大回撤控制、连续亏损检测等
    """
    
    def __init__(self, config, trading_env=None):
        """
        初始化风险管理器
        
        参数:
        - config: 配置字典
        - trading_env: 交易环境实例
        """
        self.logger = logging.getLogger("RiskManager")
        self.config = config
        self.trading_env = trading_env
        
        # 风险参数
        self.max_drawdown_pct = 0.15  # 最大回撤百分比
        self.max_daily_loss_pct = 0.05  # 最大每日亏损百分比
        self.max_position_size_pct = 0.3  # 单一仓位占总资金的最大比例
        self.max_consecutive_losses = 5  # 最大连续亏损次数
        self.min_trade_interval_seconds = 300  # 最小交易间隔时间
        self.leverage_scaling_enabled = True  # 是否启用自适应杠杆调整
        
        # 当前状态
        self.is_active = True
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.highest_equity = 0
        self.current_drawdown_pct = 0
        self.last_trade_time = 0
        self.trade_count_today = 0
        self.active_alerts = []
        
        # 市场数据
        self.market_data = {}
        
        # 过去绩效记录
        self.pnl_history = []
        self.drawdown_history = []
        self.position_history = []
        self.risk_events = []
        
        # 加载自定义风险配置
        self._load_risk_config()
        
        # 启动监控线程
        self.is_monitoring = False
        self.monitor_thread = None
        
    def _load_risk_config(self):
        """加载风险管理配置"""
        try:
            trading_config = self.config.get('trading', {})
            
            # 从配置中加载风险参数
            self.max_drawdown_pct = trading_config.get('max_drawdown_pct', 0.15)
            self.max_daily_loss_pct = trading_config.get('max_daily_loss_pct', 0.05)
            self.max_position_size_pct = trading_config.get('max_position_pct', 0.3)
            self.max_consecutive_losses = trading_config.get('max_consecutive_losses', 5)
            self.min_trade_interval_seconds = trading_config.get('min_trade_interval', 300)
            self.leverage_scaling_enabled = trading_config.get('leverage_scaling', True)
            
            self.logger.info(f"风险参数加载完成: 最大回撤={self.max_drawdown_pct:.1%}, "
                        f"最大日亏损={self.max_daily_loss_pct:.1%}, "
                        f"最大仓位比例={self.max_position_size_pct:.1%}")
        except Exception as e:
            self.logger.error(f"加载风险配置失败: {e}")
    
    def start_monitoring(self, interval=60):
        """
        启动风险监控
        
        参数:
        - interval: 监控间隔，单位秒
        """
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.logger.info(f"启动风险监控，间隔: {interval}秒")
        
        def monitor_loop():
            while self.is_monitoring:
                try:
                    self.check_all_risk_factors()
                    time.sleep(interval)
                except Exception as e:
                    self.logger.error(f"风险监控错误: {e}")
                    time.sleep(5)
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止风险监控"""
        self.is_monitoring = False
        self.logger.info("风险监控已停止")
    
    def check_all_risk_factors(self):
        """检查所有风险因素"""
        if not self.trading_env:
            return
            
        try:
            # 获取最新状态
            state = self.trading_env.get_state()
            position = state['position']
            account = state['account']
            market_data = state['market_data']
            
            # 检查资金状态
            equity = account['balance'] + position.get('unrealized_pnl', 0)
            
            # 更新最高权益和回撤
            if equity > self.highest_equity:
                self.highest_equity = equity
            
            if self.highest_equity > 0:
                self.current_drawdown_pct = 1 - (equity / self.highest_equity)
            
            # 记录历史数据
            timestamp = datetime.now().isoformat()
            self.drawdown_history.append({
                'timestamp': timestamp,
                'equity': equity,
                'drawdown': self.current_drawdown_pct
            })
            
            # 检查各项风险指标
            if self.check_max_drawdown():
                self.logger.warning(f"检测到最大回撤风险: {self.current_drawdown_pct:.2%}")
                self.add_risk_event('MAX_DRAWDOWN', f"检测到最大回撤风险: {self.current_drawdown_pct:.2%}")
            
            if self.check_daily_loss():
                self.logger.warning(f"检测到每日最大亏损风险: {self.daily_pnl:.2f} USDT")
                self.add_risk_event('DAILY_LOSS', f"检测到每日最大亏损风险: {self.daily_pnl:.2f} USDT")
            
            if position.get('size', 0) > 0 and self.check_position_size(position):
                self.logger.warning(f"检测到仓位过大风险: {position.get('size', 0)}")
                self.add_risk_event('POSITION_SIZE', f"检测到仓位过大风险: {position.get('size', 0)}")
            
            return True
        except Exception as e:
            self.logger.error(f"风险检查失败: {e}")
            return False
    
    def check_max_drawdown(self):
        """检查最大回撤是否超过限制"""
        return self.current_drawdown_pct > self.max_drawdown_pct
    
    def check_daily_loss(self):
        """检查每日亏损是否超过限制"""
        if not self.trading_env:
            return False
            
        try:
            # 获取账户信息
            state = self.trading_env.get_state()
            account = state['account']
            
            # 计算今日收益
            now = datetime.now()
            today_start = datetime(now.year, now.month, now.day).timestamp()
            
            # 过滤今日的交易记录
            today_trades = [t for t in self.trading_env.trade_history 
                           if t.get('timestamp') and datetime.fromisoformat(t['timestamp']).timestamp() >= today_start]
            
            # 计算今日已实现盈亏
            realized_pnl = sum([t.get('realized_pnl', 0) for t in today_trades if 'realized_pnl' in t])
            
            # 加上未实现盈亏
            unrealized_pnl = state['position'].get('unrealized_pnl', 0)
            
            self.daily_pnl = realized_pnl + unrealized_pnl
            
            # 检查是否超过每日亏损限制
            return self.daily_pnl < (-account['balance'] * self.max_daily_loss_pct)
        except Exception as e:
            self.logger.error(f"检查每日亏损失败: {e}")
            return False
    
    def check_position_size(self, position):
        """检查仓位大小是否合理"""
        if not self.trading_env:
            return False
            
        try:
            # 获取账户信息
            state = self.trading_env.get_state()
            account = state['account']
            market_data = state['market_data']
            
            # 计算仓位价值
            position_size = position.get('size', 0)
            current_price = market_data.get('close', 0)
            position_value = position_size * current_price
            
            # 计算仓位占账户资金的百分比
            position_pct = position_value / account['balance'] if account['balance'] > 0 else 0
            
            # 记录仓位历史
            self.position_history.append({
                'timestamp': datetime.now().isoformat(),
                'position_value': position_value,
                'position_pct': position_pct
            })
            
            # 检查是否超过最大仓位限制
            return position_pct > self.max_position_size_pct
        except Exception as e:
            self.logger.error(f"检查仓位大小失败: {e}")
            return False
    
    def check_consecutive_losses(self):
        """检查连续亏损次数是否超过限制"""
        return self.consecutive_losses >= self.max_consecutive_losses
    
    def check_trade_frequency(self):
        """检查交易频率是否过高"""
        current_time = time.time()
        time_since_last_trade = current_time - self.last_trade_time
        return time_since_last_trade < self.min_trade_interval_seconds
    
    def update_trade_result(self, trade_result):
        """
        更新交易结果，用于跟踪连续亏损等
        
        参数:
        - trade_result: 交易结果字典
        """
        try:
            # 更新最后交易时间
            self.last_trade_time = time.time()
            
            # 增加当天交易计数
            self.trade_count_today += 1
            
            # 处理盈亏
            pnl = trade_result.get('realized_pnl', 0)
            self.pnl_history.append({
                'timestamp': datetime.now().isoformat(),
                'pnl': pnl,
                'trade_id': trade_result.get('order_id', '')
            })
            
            # 更新连续亏损计数
            if pnl < 0:
                self.consecutive_losses += 1
                
                # 检查连续亏损风险
                if self.check_consecutive_losses():
                    self.logger.warning(f"检测到连续亏损风险: {self.consecutive_losses}次")
                    self.add_risk_event('CONSECUTIVE_LOSSES', 
                                       f"检测到连续亏损风险: {self.consecutive_losses}次连续亏损")
            else:
                self.consecutive_losses = 0
        except Exception as e:
            self.logger.error(f"更新交易结果失败: {e}")
    
    def adjust_position_size(self, original_size):
        """
        根据风险因素调整仓位大小
        
        参数:
        - original_size: 原始仓位大小
        
        返回:
        - 调整后的仓位大小
        """
        try:
            # 基本调整因子，初始为1（不调整）
            adjustment_factor = 1.0
            
            # 根据回撤调整
            if self.current_drawdown_pct > self.max_drawdown_pct * 0.5:
                # 回撤接近最大限制时减少仓位
                dd_factor = 1.0 - (self.current_drawdown_pct / self.max_drawdown_pct)
                adjustment_factor *= max(0.2, dd_factor)  # 最低减少到原来的20%
            
            # 根据连续亏损调整
            if self.consecutive_losses > 0:
                loss_factor = 1.0 - (self.consecutive_losses / self.max_consecutive_losses * 0.8)
                adjustment_factor *= max(0.3, loss_factor)  # 最低减少到原来的30%
            
            # 根据每日亏损调整
            if self.daily_pnl < 0 and self.trading_env:
                state = self.trading_env.get_state()
                daily_loss_limit = state['account']['balance'] * self.max_daily_loss_pct
                if abs(self.daily_pnl) > daily_loss_limit * 0.5:
                    # 每日亏损接近限制时减少仓位
                    daily_factor = 1.0 - (abs(self.daily_pnl) / daily_loss_limit)
                    adjustment_factor *= max(0.1, daily_factor)  # 最低减少到原来的10%
            
            # 应用调整因子
            adjusted_size = original_size * adjustment_factor
            
            self.logger.debug(f"仓位调整: {original_size} -> {adjusted_size} (因子: {adjustment_factor:.2f})")
            return adjusted_size
        except Exception as e:
            self.logger.error(f"调整仓位大小失败: {e}")
            return original_size
    
    def suggest_leverage(self, base_leverage, volatility=None):
        """
        根据市场波动性和风险状况建议杠杆倍数
        
        参数:
        - base_leverage: 基础杠杆倍数
        - volatility: 市场波动率(可选)
        
        返回:
        - 建议的杠杆倍数
        """
        try:
            if not self.leverage_scaling_enabled:
                return base_leverage
                
            # 基本调整因子，初始为1（不调整）
            adjustment_factor = 1.0
            
            # 根据回撤调整
            if self.current_drawdown_pct > self.max_drawdown_pct * 0.3:
                # 回撤较大时降低杠杆
                dd_factor = 1.0 - (self.current_drawdown_pct / self.max_drawdown_pct)
                adjustment_factor *= max(0.3, dd_factor)
            
            # 根据波动率调整(如果提供了波动率)
            if volatility is not None and volatility > 0:
                # 波动率越大，杠杆越低
                vol_factor = 1.0 / (1.0 + volatility)
                adjustment_factor *= vol_factor
            
            # 根据连续亏损调整
            if self.consecutive_losses > 1:
                loss_factor = 1.0 - (self.consecutive_losses / self.max_consecutive_losses * 0.5)
                adjustment_factor *= max(0.5, loss_factor)
            
            # 应用调整因子，确保不超过基础杠杆
            suggested_leverage = min(base_leverage, int(base_leverage * adjustment_factor))
            
            # 确保杠杆至少为1
            suggested_leverage = max(1, suggested_leverage)
            
            self.logger.debug(f"杠杆建议: {base_leverage} -> {suggested_leverage} (因子: {adjustment_factor:.2f})")
            return suggested_leverage
        except Exception as e:
            self.logger.error(f"计算建议杠杆失败: {e}")
            return base_leverage
    
    def add_risk_event(self, event_type, description):
        """
        添加风险事件记录
        
        参数:
        - event_type: 事件类型
        - description: 事件描述
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'description': description
        }
        self.risk_events.append(event)
        
        # 添加到活跃警报
        if event not in self.active_alerts:
            self.active_alerts.append(event)
    
    def clear_alerts(self, event_type=None):
        """
        清除警报
        
        参数:
        - event_type: 事件类型(可选)，如果提供则只清除该类型的警报
        """
        if event_type:
            self.active_alerts = [a for a in self.active_alerts if a['type'] != event_type]
        else:
            self.active_alerts = []
    
    def get_active_alerts(self):
        """获取当前活跃警报"""
        return self.active_alerts
    
    def get_risk_status(self):
        """获取当前风险状态摘要"""
        return {
            'drawdown_pct': self.current_drawdown_pct,
            'daily_pnl': self.daily_pnl,
            'consecutive_losses': self.consecutive_losses,
            'active_alerts': len(self.active_alerts),
            'risk_level': self._calculate_risk_level(),
            'trade_frequency': {
                'today_count': self.trade_count_today,
                'time_since_last': time.time() - self.last_trade_time
            }
        }
    
    def _calculate_risk_level(self):
        """计算当前整体风险等级（0-5，0最低，5最高）"""
        risk_level = 0
        
        # 根据回撤增加风险等级
        if self.current_drawdown_pct > self.max_drawdown_pct * 0.8:
            risk_level += 2
        elif self.current_drawdown_pct > self.max_drawdown_pct * 0.5:
            risk_level += 1
        
        # 根据每日亏损增加风险等级
        if self.trading_env:
            state = self.trading_env.get_state()
            daily_loss_limit = state['account']['balance'] * self.max_daily_loss_pct
            if self.daily_pnl < 0:
                if abs(self.daily_pnl) > daily_loss_limit * 0.8:
                    risk_level += 2
                elif abs(self.daily_pnl) > daily_loss_limit * 0.5:
                    risk_level += 1
        
        # 根据连续亏损增加风险等级
        if self.consecutive_losses > self.max_consecutive_losses * 0.8:
            risk_level += 1
        
        return min(risk_level, 5)  # 最高风险等级为5
    
    def update_market_data(self, market_data):
        """
        接收并更新市场数据
        
        参数:
        - market_data: 市场数据字典，包含价格、交易量等信息
        """
        try:
            if not market_data:
                return
                
            self.market_data = market_data
            
            # 如果正在监控，可以立即进行风险评估
            if self.is_monitoring and self.trading_env:
                self.check_all_risk_factors()
                
            return True
        except Exception as e:
            self.logger.error(f"更新市场数据失败: {e}")
            return False
    
    def can_trade(self, trade_size=None, side=None):
        """
        判断是否允许当前交易，根据风险参数和市场状况
        
        参数:
        - trade_size: 交易尺寸(可选)
        - side: 交易方向，'buy'或'sell'(可选)
        
        返回:
        - (bool, str): (是否允许交易，原因)
        """
        try:
            if not self.is_active:
                return False, "风险管理器未激活"
            
            # 检查是否有有效的交易环境
            if not self.trading_env:
                return False, "无法获取交易环境"
            
            # 检查交易频率是否过高
            if self.check_trade_frequency():
                time_since_last = time.time() - self.last_trade_time
                return False, f"交易频率过高，距上次交易仅{time_since_last:.1f}秒"
            
            # 检查风险限制
            if self.check_max_drawdown():
                return False, f"已达最大回撤限制: {self.current_drawdown_pct:.2%}"
            
            if self.check_daily_loss():
                return False, f"已达每日最大亏损限制: {self.daily_pnl:.2f} USDT"
            
            if self.check_consecutive_losses():
                return False, f"连续亏损次数过多: {self.consecutive_losses}次"
            
            # 如果提供了交易尺寸，检查仓位大小是否合理
            if trade_size and side:
                state = self.trading_env.get_state()
                position = state['position']
                account = state['account']
                
                # 计算交易后的仓位大小
                current_size = position.get('size', 0)
                if side == 'buy':
                    new_size = current_size + trade_size
                else:  # sell
                    new_size = current_size - trade_size
                
                # 检查仓位大小是否在限制内
                price = self.market_data.get('close', 0)
                if price <= 0:
                    return False, "无法获取有效价格"
                
                position_value = abs(new_size) * price
                position_pct = position_value / account['balance'] if account['balance'] > 0 else 0
                
                if position_pct > self.max_position_size_pct:
                    return False, f"交易后仓位过大: {position_pct:.2%}，超过限制: {self.max_position_size_pct:.2%}"
            
            # 所有检查都通过
            return True, "交易允许"
            
        except Exception as e:
            self.logger.error(f"检查交易权限失败: {e}")
            return False, f"风险评估错误: {str(e)}"
    
    def get_status(self):
        """
        获取风险管理器的状态信息
        
        返回:
        - dict: 包含状态信息的字典，包括风险统计、警报等
        """
        try:
            # 构建基本状态对象
            status = {
                "status": "active" if self.is_active else "inactive",
                "message": "风险管理正常运行" if self.is_active else "风险管理未激活",
                "is_monitoring": self.is_monitoring,
                "risk_metrics": {
                    "current_drawdown_pct": self.current_drawdown_pct,
                    "daily_pnl": self.daily_pnl,
                    "consecutive_losses": self.consecutive_losses,
                    "highest_equity": self.highest_equity,
                    "trade_count_today": self.trade_count_today
                },
                "risk_limits": {
                    "max_drawdown_pct": self.max_drawdown_pct,
                    "max_daily_loss_pct": self.max_daily_loss_pct,
                    "max_position_size_pct": self.max_position_size_pct,
                    "max_consecutive_losses": self.max_consecutive_losses,
                    "min_trade_interval_seconds": self.min_trade_interval_seconds
                },
                "alerts": self.active_alerts
            }
            
            # 添加风险评估结果
            risk_assessment = []
            
            if self.check_max_drawdown():
                risk_assessment.append({
                    "type": "drawdown",
                    "level": "high",
                    "message": f"当前回撤({self.current_drawdown_pct:.2%})超过最大限制({self.max_drawdown_pct:.2%})"
                })
                
            if self.check_daily_loss():
                risk_assessment.append({
                    "type": "daily_loss",
                    "level": "high",
                    "message": f"当日亏损({self.daily_pnl:.2f})超过限制"
                })
            
            if self.check_consecutive_losses():
                risk_assessment.append({
                    "type": "consecutive_losses",
                    "level": "medium",
                    "message": f"连续亏损({self.consecutive_losses})超过限制({self.max_consecutive_losses})"
                })
            
            status["risk_assessment"] = risk_assessment
            
            # 添加风险事件历史
            status["recent_risk_events"] = self.risk_events[-5:] if self.risk_events else []
            
            return status
            
        except Exception as e:
            self.logger.error(f"获取风险状态信息失败: {e}")
            return {
                "status": "error",
                "message": f"获取风险状态失败: {str(e)}",
                "is_monitoring": self.is_monitoring
            }
    
    def check_trade_risk(self, signal, market_data):
        """
        检查交易风险，决定是否允许交易
        
        参数:
        - signal: 交易信号，包含类型和大小
        - market_data: 当前市场数据
        
        返回:
        - bool: 是否允许交易
        """
        try:
            # 更新市场数据
            self.update_market_data(market_data)
            
            # 提取交易信息
            trade_size = signal.get('size', 0)
            trade_side = signal.get('type', '')  # BUY/SELL
            
            # 调用can_trade方法进行风险评估
            can_trade, reason = self.can_trade(trade_size=trade_size, side=trade_side.lower())
            
            if not can_trade:
                self.logger.warning(f"交易风险检查不通过: {reason}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"交易风险检查失败: {e}")
            return False