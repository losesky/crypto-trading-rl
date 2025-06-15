#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WebSocket代理服务器 - 用于将系统数据实时推送到前端
"""
import json
import logging
import threading
import time
from datetime import datetime
from flask import Flask, json
from flask_socketio import SocketIO
from flask_cors import CORS

class WebSocketProxy:
    """WebSocket代理服务，负责处理实时数据推送"""
    
    def __init__(self, port=8095):
        """初始化WebSocket代理服务"""
        self.logger = logging.getLogger("WebSocketProxy")
        self.port = port
        
        # 创建Flask应用
        self.app = Flask(__name__)
        CORS(self.app)
        
        # 初始化SocketIO
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode="threading")
        
        # 状态变量
        self.is_running = False
        self.server_thread = None
        
        # 接收到的最新数据
        self.latest_data = {}
        self.connected_clients = 0
        
        # 设置事件处理
        @self.socketio.on('connect')
        def handle_connect():
            self.connected_clients += 1
            self.logger.info(f"客户端连接，当前连接数：{self.connected_clients}")
            # 连接后立即发送一些最新数据
            self._send_latest_data_to_client()
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.connected_clients -= 1
            self.logger.info(f"客户端断开连接，当前连接数：{self.connected_clients}")
        
        @self.socketio.on('ping')
        def handle_ping():
            self.socketio.emit('pong', {'timestamp': int(datetime.now().timestamp() * 1000)})
            
        @self.app.route('/health')
        def health_check():
            return json.dumps({"status": "ok", "clients": self.connected_clients})
    
    def _send_latest_data_to_client(self):
        """向新连接的客户端发送最新数据"""
        for data_type, data in self.latest_data.items():
            try:
                self.socketio.emit(data_type, data)
            except Exception as e:
                self.logger.error(f"发送最新数据失败: {e}")
    
    def start(self):
        """启动WebSocket代理服务"""
        if self.is_running:
            self.logger.warning("WebSocket代理服务已在运行")
            return
        
        self.is_running = True
        self.logger.info(f"启动WebSocket代理服务，端口：{self.port}")
        
        # 在单独的线程中启动服务器
        self.server_thread = threading.Thread(
            target=self.socketio.run,
            args=(self.app,),
            kwargs={
                'host': '0.0.0.0', 
                'port': self.port, 
                'debug': False, 
                'use_reloader': False,
                'allow_unsafe_werkzeug': True  # 允许在开发环境中使用Werkzeug
            }
        )
        self.server_thread.daemon = True
        self.server_thread.start()
    
    def stop(self):
        """停止WebSocket代理服务"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.logger.info("停止WebSocket代理服务")
        
        # 关闭所有连接
        self.socketio.stop()
    
    def send_data(self, event_type, data):
        """发送数据到所有连接的客户端"""
        if not self.is_running:
            return
        
        try:
            # 确保数据有时间戳
            if isinstance(data, dict) and 'timestamp' not in data:
                data['timestamp'] = int(datetime.now().timestamp() * 1000)
            
            # 保存最新数据
            self.latest_data[event_type] = data
            
            # 发送数据
            self.socketio.emit(event_type, data)
            return True
        except Exception as e:
            self.logger.error(f"发送数据失败: {e}")
            return False

# 单例模式
_instance = None

def get_instance(port=8095):
    """获取WebSocketProxy实例（单例模式）"""
    global _instance
    if _instance is None:
        _instance = WebSocketProxy(port)
    return _instance

# 命令行入口
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("WebSocketProxyMain")
    
    # 创建并启动代理服务
    proxy = get_instance()
    proxy.start()
    
    try:
        logger.info("WebSocket代理服务已启动，按Ctrl+C停止...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("接收到停止信号，关闭服务...")
    finally:
        proxy.stop()
