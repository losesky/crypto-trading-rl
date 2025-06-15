#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据发送器模块 - 将交易系统的实时数据发送到前端
"""

import json
import logging
import threading
import time
import requests
from datetime import datetime

# 尝试导入WebSocket代理
try:
    from websocket_proxy import get_instance as get_websocket_proxy
    USE_WEBSOCKET_PROXY = True
except ImportError:
    USE_WEBSOCKET_PROXY = False

class DataSender:
    """数据发送器，负责将交易系统的实时数据发送到前端"""
    
    def __init__(self, config):
        """
        初始化数据发送器
        
        参数:
        - config: 配置字典
        """
        self.logger = logging.getLogger("DataSender")
        self.config = config
        
        # 从配置中获取WebSocket服务器信息
        ws_port = self.config['ui'].get('ws_port', 8095)
        
        # WebSocket代理模式
        if USE_WEBSOCKET_PROXY:
            self.logger.info("使用WebSocket代理进行数据发送")
            self.ws_proxy = get_websocket_proxy(port=ws_port)
            self.ws_proxy.start()
        # HTTP模式
        else:
            self.logger.info("使用HTTP请求进行数据发送")
            http_port = ws_port + 1  # HTTP端口是WebSocket端口+1
            self.server_url = f"http://localhost:{http_port}"
        
        # 状态变量
        self.is_sending = False
        self.send_thread = None
        self.send_interval = self.config['ui'].get('update_interval', 1000) / 1000.0
        
        # 数据缓存
        self.latest_market_data = None
        self.latest_position_data = None
        self.latest_prediction_data = None
        self.order_data = []
        self.system_status = None
    
    def start(self):
        """启动数据发送服务"""
        if self.is_sending:
            self.logger.warning("数据发送服务已启动")
            return False
        
        self.is_sending = True
        self.logger.info("启动数据发送服务...")
        
        # 启动发送线程
        self.send_thread = threading.Thread(target=self._send_loop)
        self.send_thread.daemon = True
        self.send_thread.start()
        
        return True
    
    def stop(self):
        """停止数据发送服务"""
        if not self.is_sending:
            return
        
        self.is_sending = False
        self.logger.info("停止数据发送服务")
        
        if self.send_thread:
            self.send_thread.join(timeout=2)
    
    def _send_loop(self):
        """发送数据的主循环"""
        heartbeat_counter = 0
        while self.is_sending:
            try:
                # 发送最新的市场数据
                if self.latest_market_data:
                    self._send_data("market_update", self.latest_market_data)
                
                # 发送最新的持仓数据
                if self.latest_position_data:
                    self._send_data("position_update", self.latest_position_data)
                
                # 发送最新的预测数据
                if self.latest_prediction_data:
                    self._send_data("prediction_update", self.latest_prediction_data)
                
                # 发送系统状态
                if self.system_status:
                    self._send_data("status_update", self.system_status)
                
                # 每10次循环发送一次心跳
                heartbeat_counter += 1
                if heartbeat_counter >= 10:
                    self._send_data("heartbeat", {"timestamp": int(datetime.now().timestamp() * 1000)})
                    heartbeat_counter = 0
                    
            except Exception as e:
                self.logger.error(f"发送数据时出错: {e}")
            
            # 等待一段时间
            time.sleep(self.send_interval)
    
    def _send_data(self, event_type, data):
        """发送数据到前端"""
        # 确保数据有时间戳
        if isinstance(data, dict) and 'timestamp' not in data:
            data['timestamp'] = int(datetime.now().timestamp() * 1000)
            
        # WebSocket代理模式
        if USE_WEBSOCKET_PROXY:
            return self.ws_proxy.send_data(event_type, data)
        
        # HTTP模式
        try:
            # 为HTTP请求增加事件类型
            data_to_send = data.copy()
            data_to_send['type'] = event_type
            
            url = f"{self.server_url}"
            headers = {'Content-Type': 'application/json'}
            
            response = requests.post(url, json=data_to_send, headers=headers, timeout=2)
            
            if response.status_code != 200:
                self.logger.warning(f"发送数据失败，状态码: {response.status_code}")
                self.logger.debug(f"响应内容: {response.text}")
                return False
            
            return True
        except requests.exceptions.RequestException as e:
            self.logger.error(f"HTTP请求错误: {e}")
            return False
    
    def update_market_data(self, market_data):
        """更新市场数据"""
        if not market_data:
            return
        
        data_to_send = {
            "symbol": market_data.get('symbol', 'BTCUSDT'),
            "price": market_data.get('close', market_data.get('price', 0)),
            "volume": market_data.get('volume', 0),
            "timestamp": market_data.get('timestamp', int(datetime.now().timestamp() * 1000)),
            "open": market_data.get('open', 0),
            "high": market_data.get('high', 0),
            "low": market_data.get('low', 0),
            "close": market_data.get('close', market_data.get('price', 0))
        }
        
        self.latest_market_data = data_to_send
        self._send_data("market_update", data_to_send)
    
    def update_position_data(self, position_data):
        """更新持仓数据"""
        if not position_data:
            return
        
        data_to_send = {
            "size": position_data.get('size', 0),
            "side": position_data.get('side', 'NONE'),
            "entry_price": position_data.get('entry_price', 0),
            "current_price": position_data.get('current_price', 0),
            "leverage": position_data.get('leverage', 1),
            "margin": position_data.get('margin', 0),
            "liquidation_price": position_data.get('liquidation_price', 0),
            "unrealized_pnl": position_data.get('unrealized_pnl', 0),
            "roe": position_data.get('roe', 0),
            "timestamp": position_data.get('timestamp', int(datetime.now().timestamp() * 1000))
        }
        
        self.latest_position_data = data_to_send
        self._send_data("position_update", data_to_send)
    
    def update_prediction_data(self, prediction_data):
        """更新预测数据"""
        if not prediction_data:
            return
        
        data_to_send = {
            "action": prediction_data.get('action_type', prediction_data.get('action', 'HOLD')),
            "confidence": prediction_data.get('confidence', 0.5),
            "timestamp": prediction_data.get('timestamp', int(datetime.now().timestamp() * 1000)),
            "values": {
                "buy": prediction_data.get('buy_value', prediction_data.get('values', {}).get('buy', 0.33)),
                "sell": prediction_data.get('sell_value', prediction_data.get('values', {}).get('sell', 0.33)),
                "hold": prediction_data.get('hold_value', prediction_data.get('values', {}).get('hold', 0.34))
            }
        }
        
        self.latest_prediction_data = data_to_send
        self._send_data("prediction_update", data_to_send)
    
    def add_order(self, order_data):
        """添加订单数据"""
        if not order_data:
            return
        
        data_to_send = {
            "order_id": order_data.get('order_id', ''),
            "symbol": order_data.get('symbol', 'BTCUSDT'),
            "side": order_data.get('side', 'UNKNOWN'),
            "price": order_data.get('price', 0),
            "size": order_data.get('quantity', order_data.get('size', 0)),
            "timestamp": order_data.get('timestamp', int(datetime.now().timestamp() * 1000)),
            "status": order_data.get('status', 'NEW'),
            "order_type": order_data.get('type', 'MARKET')
        }
        
        # 添加到订单历史并限制长度
        self.order_data.insert(0, data_to_send)
        if len(self.order_data) > 20:
            self.order_data.pop()
        
        self._send_data("order_update", data_to_send)
    
    def update_system_status(self, system_status):
        """更新系统状态"""
        if not system_status:
            return
        
        data_to_send = {
            "type": "status_update",
            "is_running": system_status.get('is_running', True),
            "is_paused": system_status.get('is_paused', False),
            "start_time": system_status.get('start_time', int((datetime.now() - datetime.timedelta(hours=1)).timestamp() * 1000)),
            "account_info": {
                "available_balance": system_status.get('account_info', {}).get('available_balance', 0),
                "margin_balance": system_status.get('account_info', {}).get('margin_balance', 0),
                "daily_pnl": system_status.get('account_info', {}).get('daily_pnl', 0),
                "total_pnl": system_status.get('account_info', {}).get('total_pnl', 0)
            },
            "trade_count": system_status.get('trade_count', 0),
            "last_trade_time": system_status.get('last_trade_time', 0)
        }
        
        self.system_status = data_to_send
        self._send_data("status_update", data_to_send)

# 单例模式
_instance = None

def get_instance(config=None):
    """获取DataSender单例实例"""
    global _instance
    if _instance is None and config is not None:
        _instance = DataSender(config)
    return _instance
