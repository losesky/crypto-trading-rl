"""
实时交易环境模块 - 用于实际交易场景，连接模型与币安API
"""
import time
import logging
import json
import threading
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_DOWN

from binance_client import BinanceClient

class TradingEnv:
    """实时交易环境，连接模型与交易所API"""
    
    def __init__(self, config, binance_client=None):
        """
        初始化交易环境
        
        参数:
        - config: 配置字典
        - binance_client: 币安客户端实例（如果已经存在）
        """
        self.logger = logging.getLogger("TradingEnv")
        self.config = config
        
        # 交易参数
        self.symbol = config['general']['symbol']
        self.timeframe = config['general']['timeframe']
        self.leverage = config['trading']['max_leverage']
        self.max_position_size_usd = config['trading']['max_position_size_usd']
        self.fee_rate = config['trading']['fee_rate']
        self.stop_loss_pct = config['trading']['stop_loss_pct']
        self.take_profit_pct = config['trading']['take_profit_pct']
        self.risk_per_trade_pct = config['trading']['risk_per_trade_pct']
        
        # 创建或使用现有的币安客户端
        self.binance_client = binance_client or BinanceClient(
            api_key=config['binance']['api_key'],
            api_secret=config['binance']['api_secret'],
            test_net=config['binance']['test_net']
        )
        
        # 交易状态
        self.current_position = {'size': 0, 'side': None, 'entry_price': 0, 'unrealized_pnl': 0}
        self.account_balance = 0
        self.margin_balance = 0
        self.market_data = {}
        self.order_history = []
        self.trade_history = []
        
        # 系统状态
        self.is_running = False
        self.last_update_time = 0
        self.data_buffer = []  # 存储最近的K线数据
        self.max_buffer_size = 100
        self.price_precision = 2
        self.qty_precision = 3
        self.internal_qty_precision = 6  # 内部使用更高精度
        
        # 回调函数
        self.on_position_update = None
        self.on_market_update = None
        self.on_trade_executed = None
        self.on_error = None
        
        # 初始化环境
        self.init_environment()
        
    def init_environment(self):
        """初始化交易环境"""
        try:
            self.logger.info("初始化交易环境...")
            
            # 获取交易对信息
            symbol_info = self.binance_client.get_exchange_info(self.symbol)
            if symbol_info:
                for filter_item in symbol_info.get('filters', []):
                    if filter_item['filterType'] == 'PRICE_FILTER':
                        self.price_precision = len(filter_item['tickSize'].rstrip('0').split('.')[1])
                    elif filter_item['filterType'] == 'LOT_SIZE':
                        self.qty_precision = len(filter_item['stepSize'].rstrip('0').split('.')[1])
                
                self.logger.info(f"交易对信息: 价格精度={self.price_precision}, 数量精度={self.qty_precision}")
            
            # 设置杠杆
            self.binance_client.change_leverage(self.symbol, self.leverage)
            self.logger.info(f"已设置杠杆倍数: {self.leverage}x")
            
            # 设置保证金类型为隔离
            self.binance_client.change_margin_type(self.symbol, "ISOLATED")
            self.logger.info("已设置保证金类型为隔离")
            
            # 获取账户信息
            self.update_account_info()
            
            # 获取当前市场数据
            self.update_market_data()
            
            self.logger.info("交易环境初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"初始化交易环境失败: {e}")
            if self.on_error:
                self.on_error(f"初始化失败: {str(e)}")
            return False
    
    def update_account_info(self):
        """更新账户和持仓信息"""
        try:
            # 获取账户余额
            balance_info = self.binance_client.get_balance()
            if balance_info and 'USDT' in balance_info['total']:
                self.account_balance = float(balance_info['total']['USDT'])
                self.margin_balance = float(balance_info['free']['USDT'])
                self.logger.debug(f"账户更新: 总余额={self.account_balance}, 可用余额={self.margin_balance}")
            
            # 获取持仓信息
            positions = self.binance_client.get_positions(self.symbol)
            if positions:
                position = positions[0]
                position_amt = float(position['positionAmt'])
                position_side = "BUY" if position_amt > 0 else "SELL" if position_amt < 0 else None
                entry_price = float(position['entryPrice']) if float(position['entryPrice']) > 0 else 0
                
                # 创建持仓信息字典，处理可能不存在的字段
                self.current_position = {
                    'size': abs(position_amt),
                    'side': position_side,
                    'entry_price': entry_price,
                    'leverage': float(position.get('leverage', 3)),
                    # 处理可能不存在的isolatedMargin字段
                    'isolated_margin': float(position.get('isolatedMargin', 0)),
                    'unrealized_pnl': float(position.get('unRealizedProfit', 0)),
                    'liquidation_price': float(position.get('liquidationPrice', 0)),
                    'margin_type': position.get('marginType', 'ISOLATED')
                }
                
                if self.on_position_update:
                    self.on_position_update(self.current_position)
                
                self.logger.debug(f"持仓更新: {self.current_position}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"更新账户信息失败: {e}")
            if self.on_error:
                self.on_error(f"更新账户失败: {str(e)}")
            return False
    
    def update_market_data(self):
        """更新市场数据"""
        try:
            # 获取K线数据
            klines = self.binance_client.get_historical_klines(
                self.symbol, 
                self.timeframe,
                limit=100
            )
            
            if klines is not None and not klines.empty:
                latest = klines.iloc[-1]
                
                # 更新市场数据
                self.market_data = {
                    'timestamp': latest['timestamp'],
                    'open': float(latest['open']),
                    'high': float(latest['high']),
                    'low': float(latest['low']),
                    'close': float(latest['close']),
                    'volume': float(latest['volume'])
                }
                
                # 更新数据缓冲区
                self.data_buffer.append(self.market_data)
                if len(self.data_buffer) > self.max_buffer_size:
                    self.data_buffer.pop(0)
                
                self.last_update_time = time.time()
                
                if self.on_market_update:
                    self.on_market_update(self.market_data)
                
                self.logger.debug(f"市场数据更新: 价格={self.market_data['close']}, 时间={self.market_data['timestamp']}")
                
                # 每次市场数据更新后，检查是否需要执行风险管理
                self.check_risk_management()
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"更新市场数据失败: {e}")
            if self.on_error:
                self.on_error(f"更新市场数据失败: {str(e)}")
            return False
    
    def execute_trade(self, action_type, size_pct):
        """
        执行交易
        
        参数:
        - action_type: 交易类型 ("BUY", "SELL", "HOLD")
        - size_pct: 仓位大小百分比 (-1.0 到 1.0 之间)
        
        返回:
        - 交易结果字典
        """
        if action_type == "HOLD" or abs(size_pct) < 0.01:
            return {"success": True, "action": "HOLD", "message": "不执行任何交易"}
        
        try:
            # 更新账户和市场数据
            self.update_account_info()
            self.update_market_data()
            
            current_price = self.market_data['close']
            
            # 计算目标仓位大小
            target_position_value_usd = abs(size_pct) * self.max_position_size_usd
            # 内部计算使用更高精度
            internal_position_size = self.round_step_size(target_position_value_usd / current_price, self.qty_precision, use_internal_precision=True)
            # 下单时使用交易所要求的精度
            target_position_size = self.round_step_size(internal_position_size, self.qty_precision)
            
            # 如果目标仓位太小，则不交易
            if target_position_size < 0.001:
                return {"success": True, "action": "HOLD", "message": "目标仓位太小，不执行交易"}
            
            # 获取当前持仓
            current_size = self.current_position['size']
            current_side = self.current_position['side']
            
            # 定义订单方向和大小
            order_side = None
            order_size = 0
            
            # 判断是否需要平仓再开新仓
            if current_side and (current_side != action_type or abs(target_position_size - current_size) > 0.001):
                # 先平掉当前仓位
                if current_size > 0:
                    close_side = "SELL" if current_side == "BUY" else "BUY"
                    self.logger.info(f"平仓: {close_side} {current_size}")
                    
                    result = self.binance_client.place_order(
                        symbol=self.symbol,
                        side=close_side,
                        order_type="MARKET",
                        quantity=current_size,
                        reduce_only=True
                    )
                    
                    if not result:
                        self.logger.error("平仓失败")
                        if self.on_error:
                            self.on_error("平仓失败")
                        return {"success": False, "action": close_side, "message": "平仓失败"}
                    
                    # 记录交易
                    trade_record = {
                        "timestamp": datetime.now().isoformat(),
                        "action": close_side,
                        "size": current_size,
                        "price": current_price,
                        "order_id": result.get('id', ''),
                        "type": "CLOSE"
                    }
                    self.trade_history.append(trade_record)
                    
                    if self.on_trade_executed:
                        self.on_trade_executed(trade_record)
                    
                    # 更新持仓信息
                    time.sleep(1)  # 等待订单执行
                    self.update_account_info()
            
            # 开新仓
            if action_type in ["BUY", "SELL"] and target_position_size > 0:
                self.logger.info(f"开仓: {action_type} {target_position_size}")
                
                result = self.binance_client.place_order(
                    symbol=self.symbol,
                    side=action_type,
                    order_type="MARKET",
                    quantity=target_position_size
                )
                
                if not result:
                    self.logger.error("开仓失败")
                    if self.on_error:
                        self.on_error("开仓失败")
                    return {"success": False, "action": action_type, "message": "开仓失败"}
                
                # 记录交易
                trade_record = {
                    "timestamp": datetime.now().isoformat(),
                    "action": action_type,
                    "size": target_position_size,
                    "price": current_price,
                    "order_id": result.get('id', ''),
                    "type": "OPEN"
                }
                self.trade_history.append(trade_record)
                
                if self.on_trade_executed:
                    self.on_trade_executed(trade_record)
                
                # 设置止损和止盈
                self.set_stop_loss_take_profit(action_type, target_position_size, current_price)
                
                # 更新持仓信息
                time.sleep(1)  # 等待订单执行
                self.update_account_info()
                
                return {"success": True, "action": action_type, "message": f"成功执行{action_type}交易，大小: {target_position_size}"}
            
            return {"success": True, "action": "HOLD", "message": "不需要执行交易"}
            
        except Exception as e:
            self.logger.error(f"执行交易失败: {e}")
            if self.on_error:
                self.on_error(f"执行交易失败: {str(e)}")
            return {"success": False, "action": action_type, "message": f"交易失败: {str(e)}"}
    
    def set_stop_loss_take_profit(self, side, size, entry_price):
        """设置止损和止盈订单"""
        try:
            stop_loss_price = 0
            take_profit_price = 0
            
            if side == "BUY":
                stop_loss_price = self.round_price(entry_price * (1 - self.stop_loss_pct))
                take_profit_price = self.round_price(entry_price * (1 + self.take_profit_pct))
            else:  # SELL
                stop_loss_price = self.round_price(entry_price * (1 + self.stop_loss_pct))
                take_profit_price = self.round_price(entry_price * (1 - self.take_profit_pct))
            
            # 止损订单
            if stop_loss_price > 0:
                stop_loss_result = self.binance_client.place_order(
                    symbol=self.symbol,
                    side="SELL" if side == "BUY" else "BUY",
                    order_type="STOP_MARKET",
                    quantity=size,
                    stop_price=stop_loss_price,
                    reduce_only=True
                )
                
                if stop_loss_result:
                    self.logger.info(f"已设置止损单: {'SELL' if side == 'BUY' else 'BUY'} {size} @ {stop_loss_price}")
                else:
                    self.logger.error("设置止损单失败")
            
            # 止盈订单
            if take_profit_price > 0:
                take_profit_result = self.binance_client.place_order(
                    symbol=self.symbol,
                    side="SELL" if side == "BUY" else "BUY",
                    order_type="TAKE_PROFIT_MARKET",
                    quantity=size,
                    stop_price=take_profit_price,
                    reduce_only=True
                )
                
                if take_profit_result:
                    self.logger.info(f"已设置止盈单: {'SELL' if side == 'BUY' else 'BUY'} {size} @ {take_profit_price}")
                else:
                    self.logger.error("设置止盈单失败")
                    
            return True
            
        except Exception as e:
            self.logger.error(f"设置止损止盈失败: {e}")
            return False
    
    def check_risk_management(self):
        """检查风险管理条件"""
        if not self.current_position['side'] or self.current_position['size'] <= 0:
            return False
        
        try:
            current_price = self.market_data['close']
            entry_price = self.current_position['entry_price']
            side = self.current_position['side']
            
            # 计算当前盈亏百分比
            pnl_pct = 0
            if side == "BUY":
                pnl_pct = (current_price / entry_price) - 1
            else:  # SELL
                pnl_pct = 1 - (current_price / entry_price)
            
            # 检查是否达到强制平仓条件
            if pnl_pct <= -self.stop_loss_pct * 0.8:  # 提前触发，防止滑点
                self.logger.warning(f"触发风险管理: 当前亏损 {pnl_pct:.2%}, 超过止损阈值 {self.stop_loss_pct:.2%}")
                
                # 执行平仓
                close_side = "SELL" if side == "BUY" else "BUY"
                
                result = self.binance_client.place_order(
                    symbol=self.symbol,
                    side=close_side,
                    order_type="MARKET",
                    quantity=self.current_position['size'],
                    reduce_only=True
                )
                
                if result:
                    self.logger.info(f"风险管理平仓成功: {close_side} {self.current_position['size']} @ {current_price}")
                    
                    # 记录交易
                    trade_record = {
                        "timestamp": datetime.now().isoformat(),
                        "action": close_side,
                        "size": self.current_position['size'],
                        "price": current_price,
                        "order_id": result.get('id', ''),
                        "type": "RISK_MANAGEMENT"
                    }
                    self.trade_history.append(trade_record)
                    
                    if self.on_trade_executed:
                        self.on_trade_executed(trade_record)
                    
                    # 更新持仓信息
                    time.sleep(1)  # 等待订单执行
                    self.update_account_info()
                    
                    return True
                else:
                    self.logger.error("风险管理平仓失败")
                    if self.on_error:
                        self.on_error("风险管理平仓失败")
            
            return False
            
        except Exception as e:
            self.logger.error(f"风险管理检查失败: {e}")
            return False
    
    def start(self, update_interval=60):
        """
        启动交易环境，定期更新数据
        
        参数:
        - update_interval: 更新间隔，单位秒
        """
        if self.is_running:
            return False
        
        self.is_running = True
        self.logger.info(f"启动交易环境，更新间隔: {update_interval}秒")
        
        def update_loop():
            while self.is_running:
                try:
                    self.update_account_info()
                    self.update_market_data()
                    time.sleep(update_interval)
                except Exception as e:
                    self.logger.error(f"更新循环错误: {e}")
                    if self.on_error:
                        self.on_error(f"更新循环错误: {str(e)}")
                    time.sleep(5)
        
        # 启动更新线程
        update_thread = threading.Thread(target=update_loop)
        update_thread.daemon = True
        update_thread.start()
        
        return True
    
    def stop(self):
        """停止交易环境"""
        self.is_running = False
        self.logger.info("交易环境已停止")
        return True
    
    def get_state(self):
        """获取当前环境状态"""
        return {
            'market_data': self.market_data,
            'position': self.current_position,
            'account': {
                'balance': self.account_balance,
                'margin': self.margin_balance
            },
            'last_update': self.last_update_time
        }
    
    def round_step_size(self, value, precision, use_internal_precision=False):
        """
        根据精度四舍五入数值
        
        参数:
        - value: 要舍入的值
        - precision: 精度 (小数位数)
        - use_internal_precision: 是否使用内部更高精度
        
        返回:
        - 舍入后的值
        """
        if use_internal_precision:
            # 内部计算使用更高精度
            step = Decimal('0.1') ** self.internal_qty_precision
        else:
            # 与交易所交互时使用标准精度
            step = Decimal('0.1') ** precision
            
        value = Decimal(str(value))
        return float(value.quantize(step, rounding=ROUND_DOWN))
    
    def round_price(self, price):
        """根据价格精度四舍五入价格"""
        return self.round_step_size(price, self.price_precision)
    
    def get_latest_market_data(self):
        """
        获取最新的市场数据
        
        返回:
        - market_data: 市场数据字典，包含OHLCV等信息
        """
        try:
            # 尝试获取最新K线数据
            klines_df = self.binance_client.get_historical_klines(self.symbol, self.timeframe, limit=1)
            
            if klines_df is not None and not klines_df.empty:
                # 从DataFrame转换为所需的数据格式
                kline = klines_df.iloc[0]
                timestamp_ms = int(kline['timestamp'].timestamp() * 1000)
                
                self.market_data = {
                    'timestamp': timestamp_ms,
                    'open': float(kline['open']),
                    'high': float(kline['high']),
                    'low': float(kline['low']),
                    'close': float(kline['close']),
                    'volume': float(kline['volume']),
                    'close_time': timestamp_ms + 3600000,  # 假设是小时K线，收盘时间比开盘晚1小时
                    'quote_volume': float(kline['volume']) * float(kline['close']),  # 估算quote volume
                    'trades': 0,  # 没有交易次数信息，使用默认值
                    'taker_buy_base': 0.0,  # 没有taker买入基础资产量信息，使用默认值
                    'taker_buy_quote': 0.0   # 没有taker买入报价资产量信息，使用默认值
                }
                
                # 触发市场数据更新回调
                if hasattr(self, 'on_market_update') and self.on_market_update:
                    self.on_market_update(self.market_data)
                    
                return self.market_data
            else:
                self.logger.warning("获取最新K线数据失败，返回缓存数据")
                return self.market_data or {'close': 0, 'open': 0, 'high': 0, 'low': 0, 'volume': 0, 'timestamp': int(time.time() * 1000)}
                
        except Exception as e:
            self.logger.error(f"获取最新市场数据失败: {e}")
            # 返回缓存的数据或空数据
            return self.market_data or {'close': 0, 'open': 0, 'high': 0, 'low': 0, 'volume': 0, 'timestamp': int(time.time() * 1000)}
    
    def start_market_data_stream(self):
        """
        启动市场数据流，定期获取市场数据
        """
        def update_market_data():
            while self._market_data_active:
                try:
                    self.get_latest_market_data()
                except Exception as e:
                    self.logger.error(f"更新市场数据失败: {e}")
                finally:
                    time.sleep(self.config['system']['data_update_interval'])
        
        self._market_data_active = True
        self._market_data_thread = threading.Thread(target=update_market_data)
        self._market_data_thread.daemon = True
        self._market_data_thread.start()
        self.logger.info(f"市场数据流已启动，更新间隔: {self.config['system']['data_update_interval']}秒")
        
    def stop_market_data_stream(self):
        """
        停止市场数据流
        """
        self._market_data_active = False
        if hasattr(self, '_market_data_thread') and self._market_data_thread.is_alive():
            # 等待线程结束
            self._market_data_thread.join(timeout=5)
        self.logger.info("市场数据流已停止")