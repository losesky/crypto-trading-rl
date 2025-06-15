#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
交易服务模块 - 整合所有组件，提供完整的交易服务
"""
import os
import sys
import time
import logging
import json
import threading
import signal
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# 统一使用绝对导入
from binance_client import BinanceClient
from model_wrapper import ModelWrapper
from trading_env import TradingEnv
from order_manager import OrderManager
from position_tracker import PositionTracker
from risk_manager import RiskManager
from data_recorder import DataRecorder
from system_monitor import SystemMonitor
from data_sender import get_instance as get_data_sender

class TradingService:
    """
    交易服务类 - 集成所有交易组件，提供完整的交易功能
    """
    
    def __init__(self, config):
        """
        初始化交易服务
        
        参数:
        - config: 配置文件路径或配置字典
        """
        # 设置日志系统
        self._setup_logging()
        self.logger = logging.getLogger("TradingService")
        self.logger.info("正在初始化交易服务...")
        
        # 加载配置
        if isinstance(config, dict):
            self.config = config
            self.logger.info("使用提供的配置字典")
        else:
            self.config = self._load_config(config)
            self.logger.info(f"已加载配置文件: {config}")
        
        # 设置交易模式（测试/生产）
        self.mode = self.config['general']['mode']
        self.symbol = self.config['general']['symbol']
        self.timeframe = self.config['general']['timeframe']
        
        # 初始化组件
        self.logger.info("正在初始化交易系统组件...")
        self._init_components()
        
        # 服务状态
        self.is_running = False
        self.is_trading_paused = False
        self.start_time = None
        self.trade_count = 0
        self.last_action_time = None
        self.last_model_prediction = None
        
        # 停止信号处理
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
        self.logger.info("交易服务初始化完成")
    
    def _setup_logging(self):
        """设置日志系统"""
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # 配置日志格式和级别
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _load_config(self, config_path):
        """加载配置文件"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # 设置日志级别
            log_level = getattr(logging, config['general']['log_level'])
            logging.getLogger().setLevel(log_level)
            
            return config
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            sys.exit(1)
    
    def _init_components(self):
        """初始化所有交易组件"""
        try:
            # 初始化币安客户端
            self.binance_client = BinanceClient(
                api_key=self.config['binance']['api_key'],
                api_secret=self.config['binance']['api_secret'],
                test_net=self.config['binance']['test_net']
            )
            self.logger.info("币安客户端初始化完成")
            
            # 初始化交易环境
            self.trading_env = TradingEnv(
                config=self.config,
                binance_client=self.binance_client
            )
            self.logger.info("交易环境初始化完成")
            
            # 初始化模型包装器
            try:
                self.model_wrapper = ModelWrapper(
                    model_path=self.config['general']['model_path'],
                    config=self.config
                )
                self.logger.info("模型包装器初始化完成")
            except Exception as e:
                self.logger.error(f"模型加载失败，但将继续启动其他组件: {e}")
                self.model_wrapper = None
                # 如果是测试模式，可以继续运行；如果是生产模式，则失败
                if self.mode != 'test':
                    raise
            
            # 初始化订单管理器
            self.order_manager = OrderManager(binance_client=self.binance_client)
            self.logger.info("订单管理器初始化完成")
            
            # 初始化仓位追踪器
            self.position_tracker = PositionTracker(trading_env=self.trading_env)
            self.logger.info("仓位追踪器初始化完成")
            
            # 初始化风险管理器
            self.risk_manager = RiskManager(
                config=self.config,
                trading_env=self.trading_env
            )
            self.logger.info("风险管理器初始化完成")
            
            # 初始化数据记录器
            self.data_recorder = DataRecorder(
                config=self.config
            )
            # 设置数据记录器的引用
            self.data_recorder.trading_env = self.trading_env
            self.data_recorder.position_tracker = self.position_tracker
            self.logger.info("数据记录器初始化完成")
            
            # 初始化系统监控
            self.system_monitor = SystemMonitor(
                config=self.config
            )
            # 手动设置监控组件
            
            # 初始化数据发送器
            self.data_sender = get_data_sender(self.config)
            self.logger.info("数据发送器初始化完成")
            self.system_monitor.trading_env = self.trading_env
            self.system_monitor.binance_client = self.binance_client
            self.system_monitor.order_manager = self.order_manager
            self.system_monitor.position_tracker = self.position_tracker
            self.system_monitor.risk_manager = self.risk_manager
            self.system_monitor.data_recorder = self.data_recorder
            self.system_monitor.component_status['model_loaded'] = (self.model_wrapper is not None)
            self.logger.info("系统监控初始化完成")
            
            # 设置组件间的关联
            self._setup_component_relations()
            
        except Exception as e:
            self.logger.error(f"初始化组件失败: {e}")
            sys.exit(1)
    
    def _setup_component_relations(self):
        """设置组件间的相互关联"""
        # 设置订单管理器回调
        self.order_manager.on_order_filled = self.position_tracker.update_position
        
        # 设置风险管理器回调
        self.risk_manager.on_risk_triggered = self._handle_risk_event
        
        # 设置交易环境数据更新回调
        self.trading_env.on_market_update = self._handle_market_update
        
        # 设置系统监控报警回调
        self.system_monitor.on_alert = self._handle_system_alert
    
    def start(self):
        """启动交易服务"""
        if self.is_running:
            self.logger.warning("交易服务已经在运行中")
            return
        
        self.logger.info(f"正在启动交易服务... 模式: {self.mode}, 交易对: {self.symbol}")
        
        try:
            # 启动WebSocket服务器 - 允许失败但仍继续
            try:
                self._start_websocket_server()
                self.logger.info("WebSocket服务器启动请求已发送")
            except Exception as e:
                self.logger.warning(f"启动WebSocket服务器失败: {e}")
            
            # 检查API连接 - 允许失败但仍继续
            try:
                account_info = self.binance_client.get_account_info()
                self.logger.info(f"账户连接成功, 可用余额: {account_info['availableBalance'] if account_info else 'Unknown'} USDT")
            except Exception as e:
                self.logger.warning(f"账户信息获取失败: {e}")
                account_info = None
            
            # 设置杠杆 - 允许失败但仍继续
            try:
                self.binance_client.set_leverage(
                    symbol=self.symbol,
                    leverage=self.config['trading']['max_leverage']
                )
                self.logger.info(f"已设置杠杆倍数: {self.config['trading']['max_leverage']}x")
            except Exception as e:
                self.logger.warning(f"设置杠杆失败: {e}，使用默认值")
            
            # 设置持仓模式（双向）- 允许失败但仍继续
            try:
                self.binance_client.set_position_mode(dual_side_position=False)
                self.logger.info("已设置单向持仓模式")
            except Exception as e:
                self.logger.warning(f"设置持仓模式失败: {e}，使用默认模式")
            
            # 开始市场数据流 - 允许失败但仍继续
            try:
                self.trading_env.start_market_data_stream()
                self.logger.info("市场数据流已启动")
            except Exception as e:
                self.logger.warning(f"启动市场数据流失败: {e}")
            
            # 启动订单监控 - 允许失败但仍继续
            try:
                self.order_manager.start_monitor()
                self.logger.info("订单监控已启动")
            except Exception as e:
                self.logger.warning(f"启动订单监控失败: {e}")
            
            # 启动风险监控 - 允许失败但仍继续
            try:
                self.risk_manager.start_monitoring()
                self.logger.info("风险监控已启动")
            except Exception as e:
                self.logger.warning(f"启动风险监控失败: {e}")
            
            # 启动系统监控 - 允许失败但仍继续
            try:
                self.system_monitor.start_monitoring()
                self.logger.info("系统监控已启动")
            except Exception as e:
                self.logger.warning(f"启动系统监控失败: {e}")
            
            # 启动数据记录 - 允许失败但仍继续
            try:
                self.data_recorder.start_recording()
                self.logger.info("数据记录已启动")
            except Exception as e:
                self.logger.warning(f"启动数据记录失败: {e}")
                
            # 启动数据发送服务 - 允许失败但仍继续
            try:
                self.data_sender.start()
                self.logger.info("数据发送服务已启动")
            except Exception as e:
                self.logger.warning(f"启动数据发送服务失败: {e}")
            
            # 启动交易循环
            self.is_running = True
            self.start_time = datetime.now()
            self._trading_loop()
            
            self.logger.info("交易服务已启动")
            
        except Exception as e:
            self.logger.error(f"启动交易服务失败: {e}")
            # 即使出现错误，也不会完全停止，以便UI部分仍然可以运行
            self.is_running = False
            self.logger.warning("交易服务已经停止")
    
    def _trading_loop(self):
        """交易主循环，在单独的线程中执行"""
        def run_loop():
            self.logger.info("交易循环已启动")
            
            while self.is_running:
                try:
                    # 如果交易被暂停，则等待
                    if self.is_trading_paused:
                        time.sleep(5)
                        continue
                        
                    # 获取当前市场数据和仓位数据
                    market_data = self.trading_env.get_latest_market_data()
                    position_data = self.position_tracker.get_current_position()
                    
                    # 检查是否有足够的市场数据
                    if not market_data or 'close' not in market_data:
                        self.logger.warning("市场数据不足，等待...")
                        time.sleep(5)
                        continue
                    
                    # 检查是否允许交易
                    if not self._can_trade():
                        time.sleep(5)
                        continue
                    
                    # 获取模型预测
                    if self.model_wrapper:
                        try:
                            prediction = self.model_wrapper.get_trade_decision(
                                market_data=market_data,
                                position_data=position_data,
                                risk_limit=self.config['trading']['risk_per_trade_pct']
                            )
                            self.last_model_prediction = prediction
                        except Exception as e:
                            self.logger.error(f"模型预测失败: {e}")
                            prediction = {'action': 'HOLD', 'confidence': 0.0}
                            self.last_model_prediction = prediction
                    else:
                        # 如果模型不可用，默认保持当前仓位
                        prediction = {'action': 'HOLD', 'confidence': 0.0}
                        self.last_model_prediction = prediction
                        self.logger.warning("模型不可用，使用默认HOLD策略")
                    
                    # 根据模型预测生成交易信号
                    signal = self._generate_trade_signal(prediction, market_data, position_data)
                    
                    # 执行交易信号
                    if signal:
                        self._execute_trade_signal(signal, market_data)
                        self.last_action_time = datetime.now()
                    
                    # 更新数据记录
                    self.data_recorder.record_model_prediction(prediction)
                    
                    # 定期检查处理
                    self._periodic_tasks()
                    
                    # 交易间隔
                    time.sleep(self.config['system']['data_update_interval'])
                    
                except Exception as e:
                    self.logger.error(f"交易循环发生错误: {e}")
                    time.sleep(self.config['system']['error_retry_delay'])
            
            self.logger.info("交易循环已停止")
        
        # 启动交易循环线程
        self.trading_thread = threading.Thread(target=run_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()
    
    def _can_trade(self):
        """检查是否允许交易"""
        # 检查是否通过风险控制
        can_trade, reason = self.risk_manager.can_trade(trade_size=None, side=None)
        if not can_trade:
            self.logger.info(f"风险控制阻止了交易: {reason}")
            return False
        
        # 测试模式下，WebSocket状态不影响交易决策
        is_test_mode = self.config['general']['mode'] == 'test'
        
        # 检查系统状态（在测试模式下，只检查关键错误）
        system_status = self.system_monitor.get_status()
        if not system_status:
            self.logger.warning("无法获取系统状态")
            return False
        
        if system_status.get('status') != 'healthy' and not is_test_mode:
            error_message = system_status.get('message', '未知状态问题')
            self.logger.warning(f"系统状态不佳: {error_message}")
            return False
        
        # 检查交易所API状态 - 在测试模式下用特殊逻辑处理
        if not is_test_mode:
            if not self.binance_client.check_api_status():
                self.logger.warning("交易所API状态异常")
                return False
        else:
            # 测试模式下只检查基本API可用性，忽略某些特定错误
            try:
                self.binance_client.check_api_status()
            except Exception as e:
                self.logger.warning(f"测试环境API检查错误，但将继续运行: {e}")
                # 在测试模式下，即使API检查失败也继续运行
        
        return True
    
    def _generate_trade_signal(self, prediction, market_data, position_data):
        """
        根据模型预测生成交易信号
        
        参数:
        - prediction: 模型预测结果
        - market_data: 市场数据
        - position_data: 仓位数据
        
        返回:
        - signal: 交易信号字典，如果不需要交易则返回None
        """
        current_price = market_data['close']
        action_type = prediction['action_type']
        action_value = prediction['action_value']
        confidence = prediction['confidence']
        
        # 当前持仓状态
        current_position_size = position_data['size']
        current_position_side = position_data['side']
        
        # 如果预测结果是保持不变
        if action_type == "HOLD":
            return None
        
        # 计算目标仓位大小
        target_position_value = abs(action_value) * self.config['trading']['max_position_size_usd']
        target_position_size = target_position_value / current_price
        
        # 如果变化太小，则不交易
        min_change_threshold = 0.1  # 10%的仓位变化阈值
        
        if current_position_side == action_type and abs(current_position_size - target_position_size) / max(current_position_size, 0.001) < min_change_threshold:
            self.logger.debug(f"仓位变化太小，不交易: {current_position_size:.4f} -> {target_position_size:.4f}")
            return None
        
        # 创建交易信号
        signal = {
            'type': action_type,
            'size': target_position_size,
            'price': current_price,
            'confidence': confidence,
            'timestamp': datetime.now().timestamp(),
            'reason': f"Model prediction: {action_type} with confidence {confidence:.2f}"
        }
        
        self.logger.info(f"生成交易信号: {signal['type']}, 大小: {signal['size']:.4f}, 价格: {signal['price']:.2f}")
        return signal
    
    def _execute_trade_signal(self, signal, market_data):
        """
        执行交易信号
        
        参数:
        - signal: 交易信号
        - market_data: 市场数据
        """
        try:
            symbol = self.symbol
            order_type = "MARKET"  # 使用市场单
            side = signal['type']  # BUY/SELL
            quantity = signal['size']
            
            # 检查风险控制
            if not self.risk_manager.check_trade_risk(signal, market_data):
                self.logger.warning("交易被风险控制拒绝")
                return
            
            # 确保数量大于0
            if quantity <= 0:
                self.logger.warning(f"交易数量太小: {quantity}，已调整为最小值0.001")
                quantity = 0.001
            
            # 创建订单
            order_data = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': quantity
            }
            self.logger.info(f"准备下单: {symbol} {side} {quantity}")
            order_result = self.order_manager.create_order(order_data)
            
            # order_manager.create_order返回(order_id, order_data)元组
            if order_result and isinstance(order_result, tuple) and len(order_result) >= 2:
                order_id, order_info = order_result
                if order_id:
                    self.logger.info(f"订单已提交: ID={order_id}, 类型={side}, 数量={quantity}")
                    self.trade_count += 1
                    
                    # 将订单数据发送到WebSocket服务器，供UI展示
                    try:
                        order_data_to_send = {
                            'order_id': order_id,
                            'symbol': symbol,
                            'side': side,
                            'type': order_type,
                            'price': market_data.get('close', 0),
                            'quantity': quantity,
                            'timestamp': time.time() * 1000
                        }
                        self.data_sender.add_order(order_data_to_send)
                    except Exception as e:
                        self.logger.error(f"发送订单数据到WebSocket服务器失败: {e}")
                else:
                    self.logger.error("创建订单失败: 无效的订单ID")
            else:
                self.logger.error("创建订单失败")
                
        except Exception as e:
            self.logger.error(f"执行交易信号失败: {e}")
    
    def _periodic_tasks(self):
        """执行定期任务"""
        # 检查止损和止盈
        self._check_stop_loss_take_profit()
        
        # 更新仓位数据
        self.position_tracker.update_position()  # 使用正确的方法名
        
        # 记录系统状态
        self.data_recorder.record_system_status(self.get_status())
    
    def _check_stop_loss_take_profit(self):
        """检查是否需要触发止损或止盈"""
        position = self.position_tracker.get_current_position()
        
        # 如果没有持仓，则不需要检查
        if position['size'] <= 0:
            return
        
        market_data = self.trading_env.get_latest_market_data()
        current_price = market_data['close']
        
        # 计算盈亏比例
        entry_price = position['entry_price']
        pnl_pct = 0
        
        if entry_price > 0:
            if position['side'] == 'BUY':
                pnl_pct = (current_price / entry_price) - 1
            else:
                pnl_pct = 1 - (current_price / entry_price)
        
        # 检查止损
        if pnl_pct < -self.config['trading']['stop_loss_pct']:
            self.logger.info(f"触发止损: 盈亏={pnl_pct:.2%}")
            
            # 平仓
            self._close_position(position, "触发止损")
        
        # 检查止盈
        if pnl_pct > self.config['trading']['take_profit_pct']:
            self.logger.info(f"触发止盈: 盈亏={pnl_pct:.2%}")
            
            # 平仓
            self._close_position(position, "触发止盈")
    
    def _close_position(self, position, reason=""):
        """
        平仓当前持仓
        
        参数:
        - position: 持仓信息
        - reason: 平仓原因
        """
        if position['size'] <= 0:
            return
        
        try:
            # 创建相反方向的订单来平仓
            side = "SELL" if position['side'] == 'BUY' else "BUY"
            
            order_data = {
                'symbol': self.symbol,
                'side': side,
                'type': "MARKET",
                'quantity': position['size'],
                'reduce_only': True
            }
            order = self.order_manager.create_order(order_data)
            
            if order:
                self.logger.info(f"平仓订单已提交: ID={order['orderId']}, 原因={reason}")
            else:
                self.logger.error("创建平仓订单失败")
                
        except Exception as e:
            self.logger.error(f"平仓失败: {e}")
    
    def _handle_risk_event(self, event_type, data):
        """
        处理风险事件
        
        参数:
        - event_type: 风险事件类型
        - data: 风险事件数据
        """
        self.logger.warning(f"风险事件: {event_type}, 数据: {data}")
        
        if event_type == 'max_drawdown_reached' or event_type == 'daily_loss_limit_reached':
            # 平掉所有仓位
            position = self.position_tracker.get_current_position()
            self._close_position(position, f"风险控制: {event_type}")
            
            # 记录事件
            self.data_recorder.record_risk_event({
                'type': event_type,
                'data': data,
                'timestamp': datetime.now().timestamp()
            })
    
    def _handle_market_update(self, market_data):
        """
        处理市场数据更新
        
        参数:
        - market_data: 市场数据
        """
        # 更新风险管理器
        self.risk_manager.update_market_data(market_data)
        
        # 记录市场数据
        self.data_recorder.record_market_data(market_data)
        
        # 将市场数据发送到WebSocket服务器，供UI展示
        try:
            self.data_sender.update_market_data(market_data)
        except Exception as e:
            self.logger.error(f"发送市场数据到WebSocket服务器失败: {e}")
    
    def _handle_system_alert(self, alert):
        """
        处理系统警报
        
        参数:
        - alert: 警报信息
        """
        self.logger.warning(f"系统警报: {alert['type']}, {alert['message']}")
        
        # 记录系统警报
        self.data_recorder.record_alert(alert)
        
        # 如果是严重警报，可能需要停止交易
        if alert.get('level', '') == 'error' or alert.get('severity', '') == 'critical':
            self.logger.error("收到严重警报，暂停交易")
            # 暂停交易，但不停止服务
            self._pause_trading()
    
    def _pause_trading(self):
        """暂停交易但不关闭服务"""
        self.is_trading_paused = True
        self.logger.info("交易已暂停")
        
        # 记录状态变更
        self.data_recorder.record_system_status({
            'status': 'paused',
            'message': 'Trading paused due to critical alert',
            'timestamp': datetime.now().timestamp()
        })
    
    def _resume_trading(self):
        """恢复交易"""
        self.is_trading_paused = False
        self.logger.info("交易已恢复")
        
        # 记录状态变更
        self.data_recorder.record_system_status({
            'status': 'active',
            'message': 'Trading resumed',
            'timestamp': datetime.now().timestamp()
        })
    
    def stop(self):
        """停止交易服务"""
        if not self.is_running:
            self.logger.warning("交易服务已经停止")
            return
        
        self.logger.info("正在停止交易服务...")
        self.is_running = False
        
        try:
            # 停止数据流
            self.trading_env.stop_market_data_stream()
            
            # 停止订单监控
            self.order_manager.stop_monitor()
            
            # 停止风险监控
            self.risk_manager.stop_monitoring()
            
            # 停止系统监控
            self.system_monitor.stop_monitoring()
            
            # 停止数据记录
            self.data_recorder.stop_recording()
            
            # 停止数据发送服务
            self.data_sender.stop()
            
            # 停止WebSocket服务器 - 尝试优雅关闭
            self._stop_websocket_server()
            
            # 等待交易线程结束
            if hasattr(self, 'trading_thread') and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=10)
                
            self.logger.info("交易服务已停止")
            
        except Exception as e:
            self.logger.error(f"停止交易服务时发生错误: {e}")
    
    def _start_websocket_server(self):
        """启动WebSocket服务器，用于与UI组件通信"""
        try:
            # 从配置文件中获取WebSocket端口
            ws_port = self.config['ui'].get('ws_port', 8095)  # 使用配置文件中的端口值
            
            # 尝试连接端口，如果连接成功，说明WebSocket已在运行
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', ws_port))
            sock.close()
            
            if result == 0:
                self.logger.info(f"WebSocket服务器已在运行，端口:{ws_port}")
                return True
            
            # 启动WebSocket服务器
            self.logger.info(f"正在启动WebSocket服务器，端口:{ws_port}")
            
            # 使用subprocess启动WebSocket服务器
            websocket_server_path = str(Path(__file__).parent / "websocket_server.py")
            cmd = [sys.executable, websocket_server_path, '--port', str(ws_port)]
            try:
                self.websocket_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE
                )
                self.logger.info(f"WebSocket服务器已启动，PID: {self.websocket_process.pid}")
                
                # 等待一小段时间让服务器完成启动
                time.sleep(1)
                
                # 直接通知系统监控WebSocket服务器已启动
                if hasattr(self, 'system_monitor') and self.system_monitor:
                    self.system_monitor.set_component_status('websocket_server', True)
                    self.logger.info("已通知系统监控WebSocket服务器已启动")
                
                return True
            except Exception as e:
                self.logger.error(f"启动WebSocket服务器失败: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"启动WebSocket服务器过程中出错: {e}")
            return False
    
    def _stop_websocket_server(self):
        """停止WebSocket服务器"""
        try:
            if hasattr(self, 'websocket_process') and self.websocket_process:
                self.logger.info("正在停止WebSocket服务器...")
                
                # 尝试正常终止进程
                self.websocket_process.terminate()
                
                # 给进程一些时间来清理
                try:
                    self.websocket_process.wait(timeout=3)
                    self.logger.info("WebSocket服务器已正常终止")
                except subprocess.TimeoutExpired:
                    # 如果超时，强制终止
                    self.websocket_process.kill()
                    self.logger.warning("WebSocket服务器被强制终止")
            else:
                # 尝试使用系统命令终止可能正在运行的WebSocket服务器
                try:
                    subprocess.run(["pkill", "-f", "python.*websocket_server"], 
                                  stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                    self.logger.info("WebSocket服务器已通过系统命令停止")
                except Exception as e:
                    self.logger.warning(f"尝试通过系统命令停止WebSocket服务器时出错: {e}")
                
        except Exception as e:
            self.logger.error(f"停止WebSocket服务器时出错: {e}")
            return False
        
        return True
    
    def get_status(self):
        """
        获取交易服务状态
        
        返回:
        - status: 状态信息字典
        """
        # 计算运行时间
        uptime = None
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds()
        
        # 获取账户信息
        account_info = {}
        try:
            account_info = self.binance_client.get_account_info() or {}
        except Exception as e:
            self.logger.error(f"获取账户信息失败: {e}")
        
        # 获取当前仓位
        position = self.position_tracker.get_current_position()
        
        return {
            'is_running': self.is_running,
            'is_trading_paused': self.is_trading_paused,
            'mode': self.mode,
            'symbol': self.symbol,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime': uptime,
            'trade_count': self.trade_count,
            'last_action_time': self.last_action_time.isoformat() if self.last_action_time else None,
            'account_info': account_info,
            'current_position': position,
            'last_prediction': self.last_model_prediction,
            'system_status': self.system_monitor.get_status(),
            'risk_status': self.risk_manager.get_status()
        }
    
    def _handle_shutdown(self, signum, frame):
        """处理程序关闭信号"""
        self.logger.info(f"收到关闭信号: {signum}")
        self.stop()
        sys.exit(0)
    
    def wait_for_termination(self):
        """等待程序终止的阻塞方法，用于保持主线程运行"""
        try:
            # 保持主线程运行，直到收到中断信号
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("收到键盘中断，正在关闭...")
            self.stop()

# 命令行入口
def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='交易系统服务')
    parser.add_argument('--config', '-c', type=str, required=True, help='配置文件路径')
    args = parser.parse_args()
    
    # 创建并启动交易服务
    service = TradingService(args.config)
    service.start()
    service.wait_for_termination()

if __name__ == '__main__':
    main()