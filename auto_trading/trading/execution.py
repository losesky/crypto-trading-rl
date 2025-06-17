"""
交易执行模块
负责执行交易决策，将模型预测转化为实际交易操作
"""
import logging
import yaml
import time
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import threading

from data.binance_api import BinanceAPI
from data.data_processor import DataProcessor
from models.model_loader import ModelLoader
from models.prediction import PredictionProcessor
from trading.order_manager import OrderManager, OrderSide
from trading.position_manager import PositionManager, PositionStatus
from risk.risk_checker import RiskChecker

class TradingExecutor:
    """
    交易执行类
    将模型预测转化为实际交易操作
    """
    
    def __init__(self, config_path: str = None, api: BinanceAPI = None):
        """
        初始化交易执行器
        
        Args:
            config_path: 风控配置文件路径
            api: BinanceAPI实例
        """
        self.logger = logging.getLogger('TradingExecutor')
        
        # 使用传入的配置路径或默认值
        if config_path is None:
            import os
            from pathlib import Path
            # 使用当前文件的相对路径找到配置文件
            current_dir = Path(__file__).parent.parent
            config_path = os.path.join(current_dir, 'config', 'risk_config.yaml')
            
        self._load_config(config_path)
        
        # 初始化各模块
        self.binance_api = api if api is not None else BinanceAPI()
        self.order_manager = OrderManager()
        self.position_manager = PositionManager(self.order_manager)
        
        # 找到模型配置文件路径
        from pathlib import Path
        import os as os_module
        current_dir = Path(__file__).parent.parent
        model_config_path = os_module.path.join(current_dir, 'config', 'model_config.yaml')
        
        # 使用绝对路径初始化
        self.data_processor = DataProcessor(config_path=model_config_path, api=self.binance_api)
        self.model_loader = ModelLoader(config_path=model_config_path)
        self.prediction_processor = PredictionProcessor(config_path=model_config_path)
        
        # 使用传递的风控配置路径（这是风控相关的配置）
        self.risk_checker = RiskChecker(self.position_manager, config_path=config_path)
        
        # 加载模型
        self.models = self.model_loader.initialize_models()
        
        # 交易日志路径 - 使用绝对路径
        from pathlib import Path
        current_dir = Path(__file__).parent.parent
        self.trade_log_path = os_module.path.join(current_dir, "logs", "trades.json")
        os_module.makedirs(os_module.path.dirname(self.trade_log_path), exist_ok=True)
        
        # 交易执行锁，防止并发交易冲突
        self.trade_lock = threading.Lock()
        
        # 交易执行状态
        self.running = False
        self.last_check_time = {}
        
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
            trade_limits = risk_config.get('trade_limits', {})
            
            self.min_trade_interval = trade_limits.get('min_trade_interval', 3600)  # 最小交易间隔(秒)
            self.max_daily_trades = trade_limits.get('max_daily_trades', 24)  # 24小时最大交易次数
            
            self.logger.info("成功加载交易执行配置")
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            raise
    
    def check_trading_conditions(self, symbol: str) -> bool:
        """
        检查是否满足交易条件
        
        Args:
            symbol: 交易对
            
        Returns:
            bool: 是否满足交易条件
        """
        try:
            # 检查交易间隔
            current_time = time.time()
            if symbol in self.last_check_time:
                elapsed = current_time - self.last_check_time[symbol]
                if elapsed < self.min_trade_interval:
                    self.logger.debug(f"交易间隔未满足: {symbol}, 经过{elapsed:.0f}秒/{self.min_trade_interval}秒")
                    return False
            
            # 检查当日交易次数
            today = datetime.now().date()
            trade_count = self._count_trades_today(symbol)
            if trade_count >= self.max_daily_trades:
                self.logger.info(f"已达到每日最大交易次数: {symbol}, {trade_count}/{self.max_daily_trades}")
                return False
            
            # 检查风险限制
            if not self.risk_checker.check_trading_allowed(symbol):
                self.logger.info(f"风险检查未通过，暂停交易: {symbol}")
                return False
            
            return True
        
        except Exception as e:
            self.logger.error(f"检查交易条件失败: {e}")
            return False
    
    def _count_trades_today(self, symbol: str) -> int:
        """
        统计当日交易次数
        
        Args:
            symbol: 交易对
            
        Returns:
            int: 当日交易次数
        """
        try:
            if not os.path.exists(self.trade_log_path):
                return 0
                
            with open(self.trade_log_path, 'r') as f:
                trades = json.load(f)
            
            today = datetime.now().date()
            count = 0
            
            for trade in trades:
                trade_time = datetime.fromisoformat(trade.get('timestamp', '2000-01-01T00:00:00'))
                if trade_time.date() == today and trade.get('symbol') == symbol:
                    count += 1
            
            return count
            
        except Exception as e:
            self.logger.error(f"统计当日交易次数失败: {e}")
            return 0
    
    def get_market_data(self, symbol: str, interval: str = '1h', lookback: int = 100) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        获取市场数据并准备模型输入
        
        Args:
            symbol: 交易对
            interval: K线间隔
            lookback: 历史数据量
            
        Returns:
            Tuple[pd.DataFrame, np.ndarray]: 处理后的数据和模型输入
        """
        try:
            # 获取最新市场数据
            df, model_input = self.data_processor.get_latest_data(
                symbol=symbol,
                interval=interval,
                lookback_bars=lookback
            )
            
            self.logger.debug(f"获取市场数据成功: {symbol}, 数据形状: {model_input.shape if model_input is not None else 'None'}")
            return df, model_input
            
        except Exception as e:
            self.logger.error(f"获取市场数据失败: {symbol}: {e}")
            raise
    
    def make_prediction(self, symbol: str, model_input: np.ndarray) -> Dict[str, Any]:
        """
        使用模型进行预测
        
        Args:
            symbol: 交易对
            model_input: 模型输入数据
            
        Returns:
            Dict[str, Any]: 预测结果
        """
        try:
            # 确保有模型输入数据
            if model_input is None or len(model_input) == 0:
                self.logger.error(f"没有有效的模型输入数据: {symbol}")
                return {
                    "action": 1,  # 默认动作：不操作
                    "confidence": 0,
                    "is_confident": False,
                    "error": "No valid model input data"
                }
            
            # 使用最后一个时间步的数据作为当前观测
            observation = model_input[-1:] 
            
            # 集成模型预测
            prediction = self.prediction_processor.ensemble_predictions(
                observation=observation, 
                models=self.models
            )
            
            # 记录预测结果
            self.prediction_processor.log_prediction_result(prediction, symbol)
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"模型预测失败: {symbol}: {e}")
            return {
                "action": 1,  # 默认动作：不操作
                "confidence": 0,
                "is_confident": False,
                "error": str(e)
            }
    
    def execute_trade_decision(self, symbol: str, prediction: Dict[str, Any], 
                            market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        执行交易决策
        
        Args:
            symbol: 交易对
            prediction: 预测结果
            market_data: 市场数据
            
        Returns:
            Dict[str, Any]: 交易执行结果
        """
        # 使用锁确保一次只处理一个交易
        with self.trade_lock:
            try:
                # 获取当前持仓状态
                position = self.position_manager.get_position(symbol)
                position_status = position.get('status', PositionStatus.NONE)
                
                # 根据持仓状态确定当前仓位
                current_position = 0  # 不持仓
                if position_status == PositionStatus.LONG:
                    current_position = 1  # 多仓
                elif position_status == PositionStatus.SHORT:
                    current_position = -1  # 空仓
                
                # 解释预测动作
                action_interpretation = self.prediction_processor.interpret_action(
                    prediction['action'], current_position)
                
                trade_direction = action_interpretation['trade_direction']
                self.logger.info(f"交易决策: {symbol} - {trade_direction} (置信度: {prediction['confidence']:.4f})")
                
                # 如果没有足够的置信度，不执行交易
                if not prediction.get('is_confident', False):
                    self.logger.info(f"置信度不足，不执行交易: {symbol}")
                    return {
                        "executed": False,
                        "reason": "Insufficient confidence",
                        "prediction": prediction,
                        "action_interpretation": action_interpretation
                    }
                
                # 检查交易条件
                if not self.check_trading_conditions(symbol):
                    return {
                        "executed": False,
                        "reason": "Trading conditions not met",
                        "prediction": prediction,
                        "action_interpretation": action_interpretation
                    }
                
                # 获取当前价格和波动率
                current_price = market_data['close'].iloc[-1]
                volatility = market_data['return_volatility_24'].iloc[-1] if 'return_volatility_24' in market_data.columns else 0.02
                
                # 获取账户信息
                account_info = self.binance_api.get_account_info()
                available_balance = float(account_info['availableBalance'])
                
                # 计算仓位大小
                base_position_size = self.position_manager.initial_position_size
                position_size_pct = self.prediction_processor.calculate_position_size(
                    prediction, volatility, base_position_size)
                
                # 计算实际数量
                account_balance = float(account_info['totalWalletBalance'])
                position_value = account_balance * position_size_pct
                quantity = position_value / current_price
                
                # 执行对应的交易操作
                result = None
                
                if trade_direction == "OPEN_LONG":
                    # 开多仓
                    result = self.position_manager.open_position(
                        symbol=symbol,
                        side='LONG',
                        quantity=quantity
                    )
                    
                elif trade_direction == "OPEN_SHORT":
                    # 开空仓
                    result = self.position_manager.open_position(
                        symbol=symbol,
                        side='SHORT',
                        quantity=quantity
                    )
                    
                elif trade_direction == "CLOSE_LONG":
                    # 平多仓
                    result = self.position_manager.close_position(
                        symbol=symbol
                    )
                    
                elif trade_direction == "CLOSE_SHORT":
                    # 平空仓
                    result = self.position_manager.close_position(
                        symbol=symbol
                    )
                    
                elif trade_direction == "INCREASE_LONG":
                    # 加多仓
                    result = self.position_manager.scale_in_position(
                        symbol=symbol,
                        additional_amount=quantity
                    )
                    
                elif trade_direction == "INCREASE_SHORT":
                    # 加空仓
                    result = self.position_manager.scale_in_position(
                        symbol=symbol,
                        additional_amount=quantity
                    )
                    
                else:
                    # 不执行交易
                    self.logger.info(f"不需要执行交易: {symbol} - {trade_direction}")
                    return {
                        "executed": False,
                        "reason": "No action required",
                        "prediction": prediction,
                        "action_interpretation": action_interpretation
                    }
                
                # 记录本次交易时间
                self.last_check_time[symbol] = time.time()
                
                # 记录交易日志
                self._log_trade(symbol, trade_direction, prediction, result)
                
                return {
                    "executed": True,
                    "trade_direction": trade_direction,
                    "prediction": prediction,
                    "action_interpretation": action_interpretation,
                    "result": result
                }
                
            except Exception as e:
                self.logger.error(f"执行交易决策失败: {symbol}: {e}")
                return {
                    "executed": False,
                    "error": str(e),
                    "prediction": prediction
                }
    
    def check_and_execute_trading(self, symbol: str, interval: str = '1h') -> Dict[str, Any]:
        """
        检查并执行交易
        
        Args:
            symbol: 交易对
            interval: K线间隔
            
        Returns:
            Dict[str, Any]: 交易执行结果
        """
        try:
            # 1. 获取市场数据
            market_data, model_input = self.get_market_data(symbol, interval)
            
            # 2. 模型预测
            prediction = self.make_prediction(symbol, model_input)
            
            # 3. 执行交易决策
            result = self.execute_trade_decision(symbol, prediction, market_data)
            
            return result
            
        except Exception as e:
            self.logger.error(f"检查并执行交易失败: {symbol}: {e}")
            return {"error": str(e)}
    
    def start_trading_loop(self, symbols: List[str], interval: str = '1h', 
                         check_interval: int = 300) -> None:
        """
        启动交易循环
        
        Args:
            symbols: 交易对列表
            interval: K线间隔
            check_interval: 检查间隔(秒)
        """
        self.running = True
        
        def trading_loop():
            while self.running:
                try:
                    for symbol in symbols:
                        self.logger.info(f"检查交易对: {symbol}")
                        self.check_and_execute_trading(symbol, interval)
                        
                    # 检查持仓止损和持仓时间
                    for symbol in symbols:
                        self.position_manager.check_trailing_stop(symbol)
                        self.position_manager.check_holding_time(symbol)
                        
                    # 更新所有订单状态
                    self.order_manager.update_all_orders()
                    
                    # 等待下一次检查
                    time.sleep(check_interval)
                    
                except Exception as e:
                    self.logger.error(f"交易循环异常: {e}")
                    time.sleep(60)  # 发生异常时等待时间
        
        # 启动交易线程
        self.trading_thread = threading.Thread(target=trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()
        
        self.logger.info(f"交易循环已启动，交易对: {symbols}, 间隔: {check_interval}秒")
    
    def stop_trading_loop(self) -> None:
        """停止交易循环"""
        self.running = False
        if hasattr(self, 'trading_thread') and self.trading_thread.is_alive():
            self.trading_thread.join(timeout=5.0)
        self.logger.info("交易循环已停止")
    
    def _log_trade(self, symbol: str, trade_direction: str, prediction: Dict[str, Any], 
                 result: Dict[str, Any]) -> None:
        """
        记录交易日志
        
        Args:
            symbol: 交易对
            trade_direction: 交易方向
            prediction: 预测结果
            result: 交易结果
        """
        try:
            # 读取现有日志
            trades = []
            if os.path.exists(self.trade_log_path):
                try:
                    with open(self.trade_log_path, 'r') as f:
                        trades = json.load(f)
                except json.JSONDecodeError:
                    trades = []
            
            # 添加新交易记录
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'trade_direction': trade_direction,
                'confidence': prediction.get('confidence', 0),
                'action': prediction.get('action', 1),
                'result': {
                    'executed': True,
                    'order_id': result.get('order', {}).get('client_order_id', '')
                    if result and 'order' in result else ''
                }
            }
            
            trades.append(trade_record)
            
            # 如果日志太长，移除最早的记录
            max_log_entries = 1000
            while len(trades) > max_log_entries:
                trades.pop(0)
            
            # 写入日志文件
            with open(self.trade_log_path, 'w') as f:
                json.dump(trades, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"记录交易日志失败: {e}")
    
    def get_trading_summary(self) -> Dict[str, Any]:
        """
        获取交易摘要信息
        
        Returns:
            Dict[str, Any]: 交易摘要
        """
        try:
            # 读取交易日志
            trades = []
            if os.path.exists(self.trade_log_path):
                with open(self.trade_log_path, 'r') as f:
                    trades = json.load(f)
            
            # 计算统计数据
            total_trades = len(trades)
            
            today = datetime.now().date()
            today_trades = [t for t in trades if datetime.fromisoformat(t['timestamp']).date() == today]
            today_count = len(today_trades)
            
            # 按交易对分组
            symbol_counts = {}
            for trade in trades:
                symbol = trade.get('symbol', 'unknown')
                if symbol not in symbol_counts:
                    symbol_counts[symbol] = 0
                symbol_counts[symbol] += 1
            
            # 获取仓位摘要
            position_summary = self.position_manager.get_position_summary()
            
            return {
                'total_trades': total_trades,
                'today_trades': today_count,
                'symbol_counts': symbol_counts,
                'position_summary': position_summary,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"获取交易摘要失败: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
