"""
仓位追踪器模块 - 负责跟踪仓位状态和表现
"""
import time
import logging
import json
import threading
from datetime import datetime, timedelta
from collections import deque

class PositionTracker:
    """仓位追踪器，负责跟踪和记录仓位状态"""
    
    def __init__(self, trading_env=None):
        """
        初始化仓位追踪器
        
        参数:
        - trading_env: 交易环境实例
        """
        self.logger = logging.getLogger("PositionTracker")
        self.trading_env = trading_env
        
        # 仓位记录
        self.current_position = {
            'symbol': None,
            'side': None,
            'size': 0,
            'entry_price': 0,
            'liquidation_price': 0,
            'margin': 0,
            'leverage': 1,
            'unrealized_pnl': 0,
            'update_time': None
        }
        
        # 历史仓位记录
        self.position_history = deque(maxlen=1000)  # 限制最大记录数
        
        # 性能统计
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'break_even_trades': 0,
            'largest_profit': 0,
            'largest_loss': 0,
            'avg_profit': 0,
            'avg_loss': 0,
            'total_profit': 0,
            'total_loss': 0,
            'max_drawdown': 0,
            'max_drawdown_pct': 0,
            'peak_equity': 0,
            'current_drawdown': 0,
            'current_drawdown_pct': 0
        }
        
        # 已实现盈亏记录
        self.pnl_history = []
        
        # 回调函数
        self.on_position_change = None
        self.on_position_close = None
        
        # 监控设置
        self.is_monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 10  # 秒
    
    def set_trading_env(self, trading_env):
        """设置交易环境"""
        self.trading_env = trading_env
    
    def update_position(self, position_data=None):
        """
        更新当前仓位数据
        
        参数:
        - position_data: 仓位数据字典(可选)，如果不提供则从交易环境获取
        
        返回:
        - 更新后的仓位数据
        """
        try:
            if not position_data and self.trading_env:
                state = self.trading_env.get_state()
                position_data = state.get('position', {})
            
            if not position_data:
                self.logger.warning("未提供仓位数据且未设置交易环境")
                return self.current_position
            
            # 检查仓位是否发生变化
            old_size = self.current_position.get('size', 0)
            old_side = self.current_position.get('side')
            
            new_size = position_data.get('size', 0)
            new_side = position_data.get('side')
            
            # 更新当前仓位数据
            self.current_position.update({
                'symbol': position_data.get('symbol', self.current_position.get('symbol')),
                'side': new_side,
                'size': new_size,
                'entry_price': position_data.get('entry_price', 0),
                'liquidation_price': position_data.get('liquidation_price', 0),
                'margin': position_data.get('isolated_margin', 0),
                'leverage': position_data.get('leverage', 1),
                'unrealized_pnl': position_data.get('unrealized_pnl', 0),
                'update_time': datetime.now().isoformat()
            })
            
            # 添加到历史记录
            position_record = self.current_position.copy()
            position_record['record_time'] = datetime.now().isoformat()
            
            # 如果价格信息可用，添加到记录中
            if self.trading_env:
                market_data = self.trading_env.market_data
                position_record['market_price'] = market_data.get('close', 0)
            
            self.position_history.append(position_record)
            
            # 检查仓位是否已关闭
            if old_size > 0 and new_size == 0:
                self._handle_position_close(old_side, position_data)
            
            # 检查仓位是否有变化
            if old_size != new_size or old_side != new_side:
                # 触发回调
                if self.on_position_change:
                    self.on_position_change(self.current_position)
            
            return self.current_position
            
        except Exception as e:
            self.logger.error(f"更新仓位数据失败: {e}")
            return self.current_position
    
    def _handle_position_close(self, side, position_data=None):
        """处理仓位关闭事件"""
        try:
            # 获取上一个仓位记录
            if len(self.position_history) > 1:
                prev_position = list(self.position_history)[-2]
            else:
                prev_position = None
            
            if not prev_position:
                return
            
            # 计算实现盈亏
            if position_data and 'realized_pnl' in position_data:
                realized_pnl = position_data.get('realized_pnl', 0)
            else:
                # 没有直接提供的实现盈亏，尝试估算
                entry_price = prev_position.get('entry_price', 0)
                exit_price = 0
                
                if self.trading_env:
                    market_data = self.trading_env.market_data
                    exit_price = market_data.get('close', 0)
                elif position_data and 'exit_price' in position_data:
                    exit_price = position_data.get('exit_price', 0)
                
                size = prev_position.get('size', 0)
                side_multiplier = 1 if side == 'BUY' else -1
                
                if entry_price > 0 and exit_price > 0 and size > 0:
                    realized_pnl = side_multiplier * (exit_price - entry_price) * size
                else:
                    realized_pnl = 0
            
            # 记录实现盈亏
            pnl_record = {
                'timestamp': datetime.now().isoformat(),
                'symbol': prev_position.get('symbol'),
                'side': side,
                'size': prev_position.get('size', 0),
                'entry_price': prev_position.get('entry_price', 0),
                'exit_price': position_data.get('exit_price', 0),
                'realized_pnl': realized_pnl,
                'trade_duration': 0,  # 将在下面计算
                'leverage': prev_position.get('leverage', 1)
            }
            
            # 计算交易持续时间
            if prev_position.get('update_time'):
                entry_time = datetime.fromisoformat(prev_position.get('update_time'))
                exit_time = datetime.now()
                duration = (exit_time - entry_time).total_seconds() / 60  # 转换为分钟
                pnl_record['trade_duration'] = duration
            
            self.pnl_history.append(pnl_record)
            
            # 更新统计数据
            self.stats['total_trades'] += 1
            
            if realized_pnl > 0:
                self.stats['winning_trades'] += 1
                self.stats['total_profit'] += realized_pnl
                self.stats['largest_profit'] = max(self.stats['largest_profit'], realized_pnl)
            elif realized_pnl < 0:
                self.stats['losing_trades'] += 1
                self.stats['total_loss'] += abs(realized_pnl)
                self.stats['largest_loss'] = max(self.stats['largest_loss'], abs(realized_pnl))
            else:
                self.stats['break_even_trades'] += 1
            
            # 计算平均盈亏
            if self.stats['winning_trades'] > 0:
                self.stats['avg_profit'] = self.stats['total_profit'] / self.stats['winning_trades']
            
            if self.stats['losing_trades'] > 0:
                self.stats['avg_loss'] = self.stats['total_loss'] / self.stats['losing_trades']
            
            # 触发回调
            if self.on_position_close:
                self.on_position_close(pnl_record)
            
            self.logger.info(f"仓位已关闭: {side} {prev_position.get('size', 0)} @ {pnl_record.get('exit_price', 0)}, 实现盈亏: {realized_pnl}")
            
        except Exception as e:
            self.logger.error(f"处理仓位关闭事件失败: {e}")
    
    def calculate_drawdown(self):
        """计算最大回撤"""
        try:
            if not self.pnl_history:
                return 0, 0
                
            # 计算累计盈亏曲线
            cumulative_pnl = []
            current_pnl = 0
            
            for record in self.pnl_history:
                current_pnl += record.get('realized_pnl', 0)
                cumulative_pnl.append(current_pnl)
            
            # 计算最大回撤
            peak = cumulative_pnl[0]
            max_drawdown = 0
            max_drawdown_pct = 0
            
            for pnl in cumulative_pnl:
                if pnl > peak:
                    peak = pnl
                    self.stats['peak_equity'] = peak
                
                drawdown = peak - pnl
                drawdown_pct = drawdown / peak if peak > 0 else 0
                
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                    max_drawdown_pct = drawdown_pct
                
                # 更新当前回撤
                self.stats['current_drawdown'] = drawdown
                self.stats['current_drawdown_pct'] = drawdown_pct
            
            # 更新统计数据
            self.stats['max_drawdown'] = max_drawdown
            self.stats['max_drawdown_pct'] = max_drawdown_pct
            
            return max_drawdown, max_drawdown_pct
            
        except Exception as e:
            self.logger.error(f"计算最大回撤失败: {e}")
            return 0, 0
    
    def calculate_performance_metrics(self):
        """计算绩效指标"""
        try:
            # 计算胜率
            if self.stats['total_trades'] > 0:
                win_rate = self.stats['winning_trades'] / self.stats['total_trades']
            else:
                win_rate = 0
                
            # 计算盈亏比
            profit_factor = self.stats['avg_profit'] / self.stats['avg_loss'] if self.stats['avg_loss'] > 0 else 0
            
            # 计算期望值
            expectancy = (win_rate * self.stats['avg_profit']) - ((1 - win_rate) * self.stats['avg_loss'])
            
            # 计算净盈亏
            net_pnl = self.stats['total_profit'] - self.stats['total_loss']
            
            # 计算最大回撤
            max_drawdown, max_drawdown_pct = self.calculate_drawdown()
            
            # 返回绩效指标
            return {
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'expectancy': expectancy,
                'net_pnl': net_pnl,
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown_pct,
                'total_trades': self.stats['total_trades']
            }
            
        except Exception as e:
            self.logger.error(f"计算绩效指标失败: {e}")
            return {}
    
    def get_current_position(self):
        """获取当前仓位数据"""
        return self.current_position
    
    def get_position_history(self, limit=100):
        """
        获取仓位历史记录
        
        参数:
        - limit: 返回的最大记录数
        
        返回:
        - 仓位历史记录列表
        """
        return list(self.position_history)[-limit:]
    
    def get_pnl_history(self, limit=100):
        """
        获取盈亏历史记录
        
        参数:
        - limit: 返回的最大记录数
        
        返回:
        - 盈亏历史记录列表
        """
        return self.pnl_history[-limit:]
    
    def get_stats(self):
        """获取统计数据"""
        # 计算最新的绩效指标
        metrics = self.calculate_performance_metrics()
        
        # 更新统计数据
        self.stats.update(metrics)
        
        return self.stats
    
    def start_monitoring(self, interval=10):
        """
        启动仓位监控
        
        参数:
        - interval: 监控间隔，单位秒
        """
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitor_interval = interval
        self.logger.info(f"启动仓位监控，间隔: {interval}秒")
        
        def monitor_loop():
            while self.is_monitoring:
                try:
                    if self.trading_env:
                        self.update_position()
                    time.sleep(interval)
                except Exception as e:
                    self.logger.error(f"仓位监控错误: {e}")
                    time.sleep(5)
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止仓位监控"""
        self.is_monitoring = False
        self.logger.info("仓位监控已停止")
    
    def get_daily_pnl(self):
        """获取每日盈亏统计"""
        try:
            # 按日期分组盈亏记录
            daily_pnl = {}
            
            for record in self.pnl_history:
                timestamp = record.get('timestamp')
                if timestamp:
                    date = datetime.fromisoformat(timestamp).date().isoformat()
                    
                    if date not in daily_pnl:
                        daily_pnl[date] = {
                            'realized_pnl': 0,
                            'trades': 0,
                            'winning_trades': 0,
                            'losing_trades': 0
                        }
                    
                    pnl = record.get('realized_pnl', 0)
                    daily_pnl[date]['realized_pnl'] += pnl
                    daily_pnl[date]['trades'] += 1
                    
                    if pnl > 0:
                        daily_pnl[date]['winning_trades'] += 1
                    elif pnl < 0:
                        daily_pnl[date]['losing_trades'] += 1
            
            # 转换为列表格式
            result = []
            for date, data in daily_pnl.items():
                result.append({
                    'date': date,
                    'realized_pnl': data['realized_pnl'],
                    'trades': data['trades'],
                    'winning_trades': data['winning_trades'],
                    'losing_trades': data['losing_trades'],
                    'win_rate': data['winning_trades'] / data['trades'] if data['trades'] > 0 else 0
                })
            
            # 按日期排序
            result.sort(key=lambda x: x['date'])
            
            return result
            
        except Exception as e:
            self.logger.error(f"获取每日盈亏统计失败: {e}")
            return []
    
    def export_trade_history(self, format='json'):
        """
        导出交易历史记录
        
        参数:
        - format: 导出格式 ('json', 'csv')
        
        返回:
        - 导出的数据
        """
        try:
            data = []
            
            for record in self.pnl_history:
                data.append({
                    'timestamp': record.get('timestamp', ''),
                    'symbol': record.get('symbol', ''),
                    'side': record.get('side', ''),
                    'size': record.get('size', 0),
                    'entry_price': record.get('entry_price', 0),
                    'exit_price': record.get('exit_price', 0),
                    'realized_pnl': record.get('realized_pnl', 0),
                    'leverage': record.get('leverage', 1),
                    'duration_minutes': record.get('trade_duration', 0)
                })
            
            if format == 'json':
                return json.dumps(data)
            elif format == 'csv':
                if not data:
                    return "timestamp,symbol,side,size,entry_price,exit_price,realized_pnl,leverage,duration_minutes\n"
                
                csv_data = "timestamp,symbol,side,size,entry_price,exit_price,realized_pnl,leverage,duration_minutes\n"
                for record in data:
                    csv_data += f"{record['timestamp']},{record['symbol']},{record['side']},{record['size']}," \
                               f"{record['entry_price']},{record['exit_price']},{record['realized_pnl']}," \
                               f"{record['leverage']},{record['duration_minutes']}\n"
                return csv_data
            else:
                self.logger.error(f"不支持的导出格式: {format}")
                return None
                
        except Exception as e:
            self.logger.error(f"导出交易历史记录失败: {e}")
            return None