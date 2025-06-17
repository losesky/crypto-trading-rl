"""
监控面板模块
提供交易系统的实时监控和状态展示
"""
import logging
import yaml
import os
import json
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import threading
import pandas as pd
import numpy as np

class Dashboard:
    """
    监控面板类
    提供交易系统的实时状态监控和数据可视化
    """
    
    def __init__(self, config_path: str = None):
        """
        初始化监控面板
        
        Args:
            config_path: 监控配置文件路径
        """
        self.logger = logging.getLogger('Dashboard')
        
        # 如果没有指定配置路径，自动检测可能的路径
        if config_path is None:
            # 尝试多种可能的路径
            possible_paths = [
                "../config/monitoring_config.yaml",              # 相对于monitor目录
                "./config/monitoring_config.yaml",               # 相对于auto_trading目录
                "../auto_trading/config/monitoring_config.yaml", # 相对于项目根目录
                "/home/losesky/crypto-trading-rl/auto_trading/config/monitoring_config.yaml" # 绝对路径
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
            
            # 如果所有路径都不存在，使用默认路径
            if config_path is None:
                config_path = "../config/monitoring_config.yaml"
        
        self._load_config(config_path)
        
        # 初始化状态存储
        self.dashboard_data = {
            'system_status': {
                'start_time': datetime.now().isoformat(),
                'uptime': 0,
                'status': 'initializing',
                'last_update': datetime.now().isoformat(),
                'errors': [],
                'warnings': []
            },
            'trading_status': {
                'active': False,
                'mode': 'backtest',  # backtest, paper_trading, live_trading
                'current_positions': [],
                'open_orders': [],
                'trade_history': [],
                'daily_pnl': 0.0,
                'total_pnl': 0.0
            },
            'model_status': {
                'loaded_models': [],
                'predictions': [],
                'last_prediction_time': None,
                'model_confidence': 0.0
            },
            'market_status': {
                'current_price': {},
                'indicators': {},
                'volatility': {},
                'trading_volume': {}
            },
            'risk_status': {
                'circuit_breaker': {
                    'is_active': False,
                    'trigger_reason': None,
                    'cooling_until': None
                },
                'capital_allocation': {
                    'total': 0.0,
                    'available': 0.0,
                    'allocated': 0.0
                },
                'drawdown': {
                    'current': 0.0,
                    'max': 0.0
                }
            },
            'performance_metrics': {
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }
        }
        
        # 创建仪表盘数据目录（使用绝对路径）
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dashboard_data_path = os.path.join(base_dir, "logs", "dashboard_data.json")
        self.logger.debug(f"仪表盘数据路径: {self.dashboard_data_path}")
        os.makedirs(os.path.dirname(self.dashboard_data_path), exist_ok=True)
        
        # 启动数据更新线程
        self.update_interval = 5  # 秒
        self.running = False
        self.update_thread = None
    
    def initialize(self) -> None:
        """
        初始化监控面板，在系统启动时调用
        
        准备监控面板的初始状态并加载历史数据（如果存在）
        """
        self.logger.info("初始化监控面板")
        
        # 尝试加载之前保存的仪表盘数据
        try:
            if os.path.exists(self.dashboard_data_path):
                try:
                    with open(self.dashboard_data_path, 'r', encoding='utf-8') as f:
                        saved_data = json.load(f)
                        # 合并历史数据，但保持当前系统状态
                        current_system_status = self.dashboard_data['system_status'].copy()
                        # 保留历史交易数据
                        if 'trading_status' in saved_data:
                            # 只合并数组类型的数据
                            for key in ['current_positions', 'trade_history']:
                                if key in saved_data['trading_status']:
                                    self.dashboard_data['trading_status'][key] = saved_data['trading_status'][key]
                        
                        # 保留风险状态历史信息
                        if 'risk_status' in saved_data:
                            self.dashboard_data['risk_status']['drawdown']['max'] = saved_data['risk_status'].get('drawdown', {}).get('max', 0.0)
                        
                        # 保留性能指标历史数据
                        if 'performance_metrics' in saved_data:
                            self.dashboard_data['performance_metrics'] = saved_data['performance_metrics']
                        
                        self.logger.info("已加载历史仪表盘数据")
                except json.JSONDecodeError as je:
                    self.logger.warning(f"仪表盘数据文件格式无效: {je}，将创建新的数据文件")
                    # 文件格式无效，备份并删除
                    backup_path = f"{self.dashboard_data_path}.bak.{int(time.time())}"
                    try:
                        os.rename(self.dashboard_data_path, backup_path)
                        self.logger.info(f"已将损坏的数据文件备份为: {backup_path}")
                    except Exception as rename_e:
                        self.logger.warning(f"备份损坏的数据文件失败: {rename_e}，将直接覆盖")
        except Exception as e:
            self.logger.warning(f"加载历史仪表盘数据失败: {e}")
        
        # 更新系统状态
        now = datetime.now()
        self.dashboard_data['system_status']['start_time'] = now.isoformat()
        self.dashboard_data['system_status']['status'] = 'initialized'
        self.dashboard_data['system_status']['last_update'] = now.isoformat()
        
        # 保存初始化后的状态
        self._save_dashboard_data()
        
        self.logger.info("监控面板初始化完成")
    
    def _load_config(self, config_path: str) -> None:
        """
        加载监控配置
        
        Args:
            config_path: 配置文件路径
        """
        try:
            # 记录正在尝试加载的配置路径
            abs_path = os.path.abspath(config_path)
            self.logger.debug(f"尝试加载配置文件: {abs_path}")
            
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                # 加载配置参数
                monitoring_config = config.get('monitoring', {})
                self.dashboard_config = monitoring_config.get('dashboard', {})
                
                self.logger.info(f"监控面板配置加载成功: {os.path.basename(config_path)}")
            else:
                self.logger.warning(f"配置文件不存在: {config_path}, 使用默认配置")
                # 设置默认配置
                self.dashboard_config = {
                    'update_interval': 5,
                    'history_length': 100,
                    'enable_notifications': True
                }
                
                # 尝试创建默认配置文件
                try:
                    # 确保目录存在
                    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                    
                    # 创建默认配置
                    default_config = {
                        'monitoring': {
                            'dashboard': self.dashboard_config,
                            'performance': {
                                'monitor_resources': True,
                                'cpu_alert_threshold': 80,
                                'memory_alert_threshold': 85
                            },
                            'notifications': {
                                'send_error_notifications': True,
                                'send_performance_alerts': True,
                                'cooldown_period': 300
                            }
                        }
                    }
                    
                    # 写入配置文件
                    with open(abs_path, 'w', encoding='utf-8') as f:
                        yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
                    
                    self.logger.info(f"已创建默认配置文件: {abs_path}")
                except Exception as write_err:
                    self.logger.warning(f"无法创建默认配置文件: {write_err}")
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            # 设置默认配置
            self.dashboard_config = {
                'update_interval': 5,
                'history_length': 100,
                'enable_notifications': True
            }
    
    def start(self) -> None:
        """启动监控面板"""
        if self.running:
            self.logger.warning("监控面板已经在运行")
            return
        
        self.running = True
        self.dashboard_data['system_status']['status'] = 'running'
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        self.logger.info("监控面板已启动")
    
    def stop(self) -> None:
        """停止监控面板"""
        if not self.running:
            return
        
        self.running = False
        self.dashboard_data['system_status']['status'] = 'stopped'
        if self.update_thread:
            self.update_thread.join(timeout=5.0)
        
        self.logger.info("监控面板已停止")
        self._save_dashboard_data()
    
    def _update_loop(self) -> None:
        """数据更新循环"""
        while self.running:
            try:
                self._update_dashboard_data()
                self._save_dashboard_data()
                time.sleep(self.update_interval)
            except Exception as e:
                self.logger.error(f"仪表盘更新失败: {e}")
                time.sleep(self.update_interval * 2)  # 出错后延长等待时间
    
    def _update_dashboard_data(self) -> None:
        """更新仪表盘数据"""
        now = datetime.now()
        
        # 更新系统状态
        self.dashboard_data['system_status']['uptime'] = (now - datetime.fromisoformat(self.dashboard_data['system_status']['start_time'])).total_seconds()
        self.dashboard_data['system_status']['last_update'] = now.isoformat()
        
        # 其他数据会通过update_xxx方法更新
        # 这里主要是更新一些时间相关的统计信息
    
    def update(self) -> None:
        """
        更新监控面板的状态信息
        在交易系统主循环中定期调用
        """
        try:
            # 更新时间
            now = datetime.now()
            self.dashboard_data['system_status']['last_update'] = now.isoformat()
            
            # 刷新运行时间
            start_time = datetime.fromisoformat(self.dashboard_data['system_status']['start_time'])
            self.dashboard_data['system_status']['uptime'] = (now - start_time).total_seconds()
            
            # 这里我们不获取其他模块的信息，因为那些应该通过各自的update_xxx方法更新
            # 系统其他部分会调用dashboard.update_trading_status()、update_positions()等方法
            
            # 保存更新后的数据
            self._save_dashboard_data()
            
            self.logger.debug("仪表板状态已更新")
        except Exception as e:
            self.logger.error(f"更新仪表板状态失败: {e}")
    
    def _save_dashboard_data(self) -> None:
        """保存仪表盘数据到文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.dashboard_data_path), exist_ok=True)
            
            # 先写入临时文件，然后重命名，避免写入中断导致文件损坏
            temp_path = f"{self.dashboard_data_path}.tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                data_to_save = self.dashboard_data.copy()
                data_to_save['_save_time'] = datetime.now().isoformat()
                json.dump(data_to_save, f, indent=2, default=str)  # default=str 处理无法序列化的对象
            
            # 重命名临时文件，这是一个原子操作，不会留下损坏的文件
            os.replace(temp_path, self.dashboard_data_path)
        except Exception as e:
            self.logger.error(f"保存仪表盘数据失败: {e}")
    
    def update_system_status(self, status: Dict) -> None:
        """
        更新系统状态信息
        
        Args:
            status: 系统状态信息
        """
        self.dashboard_data['system_status'].update(status)
    
    def add_error(self, error_msg: str, source: str = "system") -> None:
        """
        添加错误信息
        
        Args:
            error_msg: 错误信息
            source: 错误来源
        """
        error_entry = {
            "time": datetime.now().isoformat(),
            "message": error_msg,
            "source": source
        }
        
        self.dashboard_data['system_status']['errors'].append(error_entry)
        # 保留最近的100条错误记录
        self.dashboard_data['system_status']['errors'] = self.dashboard_data['system_status']['errors'][-100:]
        
        self.logger.error(f"Error [{source}]: {error_msg}")
    
    def add_warning(self, warning_msg: str, source: str = "system") -> None:
        """
        添加警告信息
        
        Args:
            warning_msg: 警告信息
            source: 警告来源
        """
        warning_entry = {
            "time": datetime.now().isoformat(),
            "message": warning_msg,
            "source": source
        }
        
        self.dashboard_data['system_status']['warnings'].append(warning_entry)
        # 保留最近的100条警告记录
        self.dashboard_data['system_status']['warnings'] = self.dashboard_data['system_status']['warnings'][-100:]
        
        self.logger.warning(f"Warning [{source}]: {warning_msg}")
    
    def update_trading_status(self, status: Dict) -> None:
        """
        更新交易状态
        
        Args:
            status: 交易状态信息
        """
        self.dashboard_data['trading_status'].update(status)
    
    def update_trade_status(self, symbol: str, action: str, quantity: float, 
                           price: float = 0, status: str = "PENDING", note: str = "") -> None:
        """
        更新单笔交易状态
        
        Args:
            symbol: 交易对
            action: 交易动作 (OPEN_LONG, CLOSE_LONG, OPEN_SHORT, CLOSE_SHORT等)
            quantity: 交易数量
            price: 交易价格
            status: 交易状态 (SUCCESS, FAILED, ERROR, PENDING)
            note: 备注信息
        """
        try:
            timestamp = datetime.now()
            
            # 创建交易记录
            trade_record = {
                'timestamp': timestamp.isoformat(),
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': price,
                'status': status,
                'note': note,
                'value': quantity * price if price > 0 else 0
            }
            
            # 确保trades列表存在
            if 'trades' not in self.dashboard_data:
                self.dashboard_data['trades'] = []
            
            # 添加交易记录
            self.dashboard_data['trades'].append(trade_record)
            
            # 限制交易记录数量，只保留最近的1000条
            if len(self.dashboard_data['trades']) > 1000:
                self.dashboard_data['trades'] = self.dashboard_data['trades'][-1000:]
            
            # 更新交易统计
            self._update_trade_statistics(trade_record)
            
            # 保存到文件
            self._save_dashboard_data()
            
            self.logger.debug(f"已记录交易状态: {symbol} {action} {status}")
            
        except Exception as e:
            self.logger.error(f"更新交易状态失败: {e}")
    
    def _update_trade_statistics(self, trade_record: Dict[str, Any]) -> None:
        """
        更新交易统计信息
        
        Args:
            trade_record: 交易记录
        """
        try:
            # 确保dashboard_data中有statistics字段
            if 'statistics' not in self.dashboard_data:
                self.dashboard_data['statistics'] = {
                    'total_trades': 0,
                    'successful_trades': 0,
                    'failed_trades': 0,
                    'total_volume': 0.0,
                    'profit_trades': 0,
                    'loss_trades': 0,
                    'avg_trade_size': 0.0,
                    'last_trade_time': None,
                    'daily_trade_count': 0,
                    'weekly_trade_count': 0,
                    'daily_volume': 0.0,
                    'weekly_volume': 0.0
                }
            
            stats = self.dashboard_data['statistics']
            
            # 更新总交易次数
            stats['total_trades'] += 1
            
            # 根据交易状态更新成功/失败次数
            if trade_record['status'] == 'SUCCESS':
                stats['successful_trades'] += 1
            elif trade_record['status'] in ['FAILED', 'ERROR']:
                stats['failed_trades'] += 1
            
            # 更新交易量
            value = trade_record.get('value', 0)
            if value > 0:
                stats['total_volume'] += value
                
            # 更新平均交易大小
            if stats['total_trades'] > 0:
                stats['avg_trade_size'] = stats['total_volume'] / stats['successful_trades'] if stats['successful_trades'] > 0 else 0
                
            # 更新最后交易时间
            stats['last_trade_time'] = trade_record['timestamp']
            
            # 更新每日和每周统计
            now = datetime.now()
            today = now.replace(hour=0, minute=0, second=0, microsecond=0)
            week_start = today - timedelta(days=today.weekday())
            
            # 清理过期的每日和每周计数
            if 'daily_stats_date' not in stats or stats['daily_stats_date'] != today.isoformat():
                stats['daily_trade_count'] = 0
                stats['daily_volume'] = 0.0
                stats['daily_stats_date'] = today.isoformat()
                
            if 'weekly_stats_date' not in stats or stats['weekly_stats_date'] != week_start.isoformat():
                stats['weekly_trade_count'] = 0
                stats['weekly_volume'] = 0.0
                stats['weekly_stats_date'] = week_start.isoformat()
                
            # 更新每日和每周交易次数与交易量
            if trade_record['status'] == 'SUCCESS':
                stats['daily_trade_count'] += 1
                stats['weekly_trade_count'] += 1
                stats['daily_volume'] += value
                stats['weekly_volume'] += value
                
            # 按交易对更新统计
            symbol = trade_record['symbol']
            if 'symbol_stats' not in self.dashboard_data:
                self.dashboard_data['symbol_stats'] = {}
                
            if symbol not in self.dashboard_data['symbol_stats']:
                self.dashboard_data['symbol_stats'][symbol] = {
                    'total_trades': 0,
                    'successful_trades': 0,
                    'failed_trades': 0,
                    'total_volume': 0.0,
                    'last_trade_time': None
                }
                
            symbol_stats = self.dashboard_data['symbol_stats'][symbol]
            symbol_stats['total_trades'] += 1
            if trade_record['status'] == 'SUCCESS':
                symbol_stats['successful_trades'] += 1
                symbol_stats['total_volume'] += value
            elif trade_record['status'] in ['FAILED', 'ERROR']:
                symbol_stats['failed_trades'] += 1
            
            symbol_stats['last_trade_time'] = trade_record['timestamp']
            
            self.logger.debug(f"已更新交易统计信息")
                
        except Exception as e:
            self.logger.error(f"更新交易统计信息失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def update_positions(self, positions: List[Dict]) -> None:
        """
        更新当前持仓信息
        
        Args:
            positions: 持仓列表
        """
        self.dashboard_data['trading_status']['current_positions'] = positions
    
    def update_orders(self, orders: List[Dict]) -> None:
        """
        更新当前订单信息
        
        Args:
            orders: 订单列表
        """
        self.dashboard_data['trading_status']['open_orders'] = orders
    
    def add_trade(self, trade: Dict) -> None:
        """
        添加交易记录
        
        Args:
            trade: 交易记录
        """
        # 添加时间戳如果没有
        if 'time' not in trade:
            trade['time'] = datetime.now().isoformat()
        
        self.dashboard_data['trading_status']['trade_history'].append(trade)
        # 保留最近的交易记录
        history_length = self.dashboard_config.get('history_length', 100)
        self.dashboard_data['trading_status']['trade_history'] = self.dashboard_data['trading_status']['trade_history'][-history_length:]
        
        # 更新每日盈亏
        if 'pnl' in trade:
            self.dashboard_data['trading_status']['daily_pnl'] += trade['pnl']
            self.dashboard_data['trading_status']['total_pnl'] += trade['pnl']
    
    def update_model_status(self, status: Dict) -> None:
        """
        更新模型状态
        
        Args:
            status: 模型状态信息
        """
        self.dashboard_data['model_status'].update(status)
    
    def add_prediction(self, prediction: Dict) -> None:
        """
        添加模型预测结果
        
        Args:
            prediction: 预测结果
        """
        # 添加时间戳如果没有
        if 'time' not in prediction:
            prediction['time'] = datetime.now().isoformat()
        
        self.dashboard_data['model_status']['predictions'].append(prediction)
        self.dashboard_data['model_status']['last_prediction_time'] = prediction['time']
        
        # 保留最近的预测记录
        history_length = self.dashboard_config.get('history_length', 100)
        self.dashboard_data['model_status']['predictions'] = self.dashboard_data['model_status']['predictions'][-history_length:]
    
    def update_prediction(self, symbol: str, timestamp: datetime, action_probas: Dict, position_size: float, confidence: float) -> None:
        """
        更新模型预测结果
        
        Args:
            symbol: 交易对符号
            timestamp: 预测时间
            action_probas: 行动概率字典
            position_size: 建议仓位大小
            confidence: 预测置信度
        """
        # 创建预测记录
        prediction = {
            'symbol': symbol,
            'time': timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
            'action_probas': action_probas,
            'position_size': position_size,
            'confidence': confidence
        }
        
        # 将预测添加到列表中
        self.add_prediction(prediction)
        
        # 更新最新预测的置信度
        self.dashboard_data['model_status']['model_confidence'] = confidence
        
        self.logger.debug(f"已更新 {symbol} 的预测结果，信心度: {confidence}")
    
    def get_latest_prediction(self, symbol: str) -> Dict:
        """
        获取最新的预测结果
        
        Args:
            symbol: 交易对符号
            
        Returns:
            Dict: 最新的预测结果，如果不存在则返回None
        """
        try:
            predictions = self.dashboard_data['model_status']['predictions']
            
            # 从最新到最旧遍历预测
            for prediction in reversed(predictions):
                if prediction.get('symbol') == symbol:
                    return prediction
                    
            # 如果没有找到匹配的预测
            return None
        except Exception as e:
            self.logger.error(f"获取最新预测失败: {e}")
            return None
    
    def update_market_data(self, symbol: str, timeframe: str, kline_df: pd.DataFrame, features_df: pd.DataFrame) -> None:
        """
        更新市场数据和特征数据
        
        Args:
            symbol: 交易对符号
            timeframe: 时间周期
            kline_df: K线数据DataFrame
            features_df: 特征数据DataFrame
        """
        try:
            # 初始化市场数据结构
            if 'data' not in self.dashboard_data['market_status']:
                self.dashboard_data['market_status']['data'] = {}
                
            # 初始化交易对和时间周期的数据结构
            pair_key = f"{symbol}_{timeframe}"
            if pair_key not in self.dashboard_data['market_status']['data']:
                self.dashboard_data['market_status']['data'][pair_key] = {}
                
            # 更新K线数据和特征数据
            self.dashboard_data['market_status']['data'][pair_key]['kline'] = kline_df.tail(100).to_dict('records')
            self.dashboard_data['market_status']['data'][pair_key]['features'] = features_df.tail(100).to_dict('records')
            
            # 更新最新价格
            if not kline_df.empty:
                latest_price = kline_df.iloc[-1]['close']
                self.dashboard_data['market_status']['current_price'][symbol] = latest_price
                
            self.logger.debug(f"已更新 {symbol} {timeframe} 市场数据")
        except Exception as e:
            self.logger.error(f"更新市场数据失败: {e}")
    
    def get_features_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        获取特定交易对和时间周期的特征数据
        
        Args:
            symbol: 交易对符号
            timeframe: 时间周期
            
        Returns:
            pd.DataFrame: 特征数据DataFrame，如果不存在则返回None
        """
        try:
            pair_key = f"{symbol}_{timeframe}"
            
            # 检查数据是否存在
            if 'data' not in self.dashboard_data['market_status'] or \
               pair_key not in self.dashboard_data['market_status'].get('data', {}) or \
               'features' not in self.dashboard_data['market_status']['data'][pair_key]:
                self.logger.warning(f"无法找到 {symbol} {timeframe} 的特征数据")
                return None
                
            # 获取特征数据并转换为DataFrame
            features_data = self.dashboard_data['market_status']['data'][pair_key]['features']
            if not features_data:
                return None
                
            return pd.DataFrame(features_data)
        except Exception as e:
            self.logger.error(f"获取特征数据失败: {e}")
            return None
    
    def update_performance_metrics(self, metrics: Dict) -> None:
        """
        更新性能指标
        
        Args:
            metrics: 性能指标
        """
        self.dashboard_data['performance_metrics'].update(metrics)
    
    def update_capital_allocation(self, capital_data: Dict) -> None:
        """
        更新资金分配信息
        
        Args:
            capital_data: 资金分配数据，包含 total, available, allocated, reserved 等字段
        """
        try:
            # 更新资金状态到仪表盘
            self.dashboard_data['risk_status']['capital_allocation'].update(capital_data)
            self.logger.debug(f"已更新资金分配信息: 总资金={capital_data.get('total', 0):.2f}, 可用资金={capital_data.get('available', 0):.2f}")
        except Exception as e:
            self.logger.error(f"更新资金分配信息失败: {e}")
    
    def calculate_performance_metrics(self) -> Dict:
        """
        计算交易性能指标
        
        Returns:
            Dict: 性能指标
        """
        trades = self.dashboard_data['trading_status'].get('trade_history', [])
        if not trades:
            return self.dashboard_data['performance_metrics']
        
        # 计算基本指标
        profits = [trade['pnl'] for trade in trades if trade.get('pnl', 0) > 0]
        losses = [trade['pnl'] for trade in trades if trade.get('pnl', 0) < 0]
        
        total_trades = len(trades)
        win_trades = len(profits)
        loss_trades = len(losses)
        
        # 计算胜率
        win_rate = win_trades / total_trades if total_trades > 0 else 0
        
        # 计算平均收益和亏损
        avg_profit = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # 计算利润因子
        total_profit = sum(profits) if profits else 0
        total_loss = abs(sum(losses)) if losses else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # 计算最大回撤
        balance_curve = []
        balance = 0
        for trade in trades:
            balance += trade.get('pnl', 0)
            balance_curve.append(balance)
        
        if balance_curve:
            # 计算最大回撤
            max_balance = 0
            max_drawdown = 0
            
            for balance in balance_curve:
                max_balance = max(max_balance, balance)
                drawdown = (max_balance - balance) / max_balance if max_balance > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
        else:
            max_drawdown = 0
        
        # 更新性能指标
        metrics = {
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'profitable_trades': win_trades,
            'loss_trades': loss_trades
        }
        
        # 更新仪表盘数据
        self.dashboard_data['performance_metrics'] = metrics
        
        return metrics
    
    def get_dashboard_data(self) -> Dict:
        """
        获取完整的仪表盘数据
        
        Returns:
            Dict: 仪表盘数据
        """
        return self.dashboard_data
    
    def get_summary(self) -> str:
        """
        获取系统状态摘要
        
        Returns:
            str: 状态摘要字符串
        """
        data = self.dashboard_data
        
        # 格式化摘要信息
        summary = []
        summary.append(f"系统状态: {data['system_status']['status']}")
        summary.append(f"运行时间: {timedelta(seconds=int(data['system_status']['uptime']))}")
        
        # 交易状态摘要
        trading = data['trading_status']
        summary.append(f"交易模式: {trading['mode']}")
        summary.append(f"当前持仓: {len(trading['current_positions'])}")
        summary.append(f"未平仓订单: {len(trading['open_orders'])}")
        summary.append(f"日盈亏: {trading['daily_pnl']:.2f} USDT")
        summary.append(f"总盈亏: {trading['total_pnl']:.2f} USDT")
        
        # 风控状态摘要
        risk = data['risk_status']
        if risk['circuit_breaker']['is_active']:
            cooling_until = datetime.fromisoformat(risk['circuit_breaker']['cooling_until']) if risk['circuit_breaker']['cooling_until'] else None
            cooling_time = f"直到 {cooling_until.strftime('%Y-%m-%d %H:%M:%S')}" if cooling_until else "无结束时间"
            summary.append(f"熔断状态: 已激活 ({cooling_time})")
            summary.append(f"熔断原因: {risk['circuit_breaker']['trigger_reason']}")
        else:
            summary.append("熔断状态: 未激活")
        
        # 资金状态
        capital = risk['capital_allocation']
        total = capital.get('total', 0)
        available = capital.get('available', 0)
        summary.append(f"总资金: {total:.2f} USDT")
        
        # 避免除零错误
        if total > 0:
            available_percent = (available / total) * 100
        else:
            available_percent = 0
            
        summary.append(f"可用资金: {available:.2f} USDT ({available_percent:.1f}% 可用)")
        
        # 性能指标
        perf = data['performance_metrics']
        summary.append(f"胜率: {perf['win_rate']*100:.1f}%")
        summary.append(f"最大回撤: {perf['max_drawdown']*100:.1f}%")
        
        return "\n".join(summary)
    
    def generate_summary(self) -> None:
        """
        生成系统运行的最终摘要报告
        
        在系统关闭时调用，计算最终性能指标，
        并将摘要信息保存到日志和文件中
        """
        self.logger.info("正在生成系统运行摘要报告...")
        
        # 计算最终性能指标
        metrics = self.calculate_performance_metrics()
        
        # 获取摘要字符串
        summary_text = self.get_summary()
        
        # 记录摘要到日志
        for line in summary_text.split("\n"):
            self.logger.info(f"摘要: {line}")
        
        # 将摘要保存到文件
        summary_file = "../logs/trading_summary.txt"
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                # 添加时间戳
                f.write(f"交易系统摘要报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*60 + "\n\n")
                f.write(summary_text)
                f.write("\n\n")
                
                # 添加详细的性能指标
                f.write("详细性能指标\n")
                f.write("-"*40 + "\n")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.4f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
            
            self.logger.info(f"摘要报告已保存到 {summary_file}")
        except Exception as e:
            self.logger.error(f"保存摘要报告失败: {e}")
        
        # 保存最终的仪表盘数据
        self._save_dashboard_data()
        
        self.logger.info("系统运行摘要报告生成完成")
