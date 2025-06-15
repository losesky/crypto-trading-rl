#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UI服务器模块 - 提供Web界面的后端服务
"""
import os
import sys
import time
import logging
import json
import threading
import argparse
import socket
from pathlib import Path
from datetime import datetime, timedelta
from functools import wraps
import numpy as np
import pandas as pd

# Web框架
from flask import Flask, jsonify, request, render_template, send_from_directory
try:
    from flask_cors import CORS
except ImportError:
    print("警告: flask_cors 模块未找到。将禁用跨域资源共享。请运行 'pip install flask-cors' 安装此模块。")
    CORS = lambda app: None  # 如果导入失败，提供一个空函数
import websocket
from flask_socketio import SocketIO, emit

class UIServer:
    """
    UI服务器类 - 提供Web界面的后端服务
    """
    
    def __init__(self, config, trading_service=None):
        """
        初始化UI服务器
        
        参数:
        - config: 配置字典或配置文件路径
        - trading_service: 交易服务实例（可选）
        """
        self.logger = logging.getLogger("UIServer")
        
        # 加载配置
        if isinstance(config, str):
            with open(config, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = config
        
        # 保存交易服务实例
        self.trading_service = trading_service
        
        # 设置UI参数
        self.http_port = self.config['ui']['http_port']
        self.ws_port = self.config['ui'].get('ws_port', self.http_port)
        self.update_interval = self.config['ui'].get('update_interval', 1000)
        self.max_data_points = self.config['ui'].get('max_data_points', 200)
        
        # UI数据缓存
        self.market_data_cache = []
        self.position_data_cache = []
        self.order_data_cache = []
        self.prediction_data_cache = []
        self.alert_data_cache = []
        
        # 创建Flask应用
        self.app = Flask(
            __name__, 
            static_folder=str(Path(__file__).parent.parent / "ui"),
            template_folder=str(Path(__file__).parent.parent / "ui")
        )
        CORS(self.app)  # 启用跨域请求
        
        # 初始化SocketIO
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # 设置API路由
        self._setup_routes()
        
        # 订阅数据更新
        if trading_service:
            self._setup_data_subscriptions()
        
        self.update_thread = None
        self.is_running = False
        
        self.logger.info("UI服务器初始化完成")
    
    def _setup_routes(self):
        """设置API路由"""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/<path:path>')
        def all_files(path):
            return send_from_directory(self.app.static_folder, path)
        
        @self.app.route('/api/status')
        def get_status():
            if self.trading_service:
                return jsonify(self.trading_service.get_status())
            return jsonify({"error": "Trading service not available"})
        
        @self.app.route('/api/market_data')
        def get_market_data():
            return jsonify(self.market_data_cache)
        
        @self.app.route('/api/position_data')
        def get_position_data():
            return jsonify(self.position_data_cache)
        
        @self.app.route('/api/orders')
        def get_orders():
            return jsonify(self.order_data_cache)
        
        @self.app.route('/api/predictions')
        def get_predictions():
            return jsonify(self.prediction_data_cache)
        
        @self.app.route('/api/alerts')
        def get_alerts():
            return jsonify(self.alert_data_cache)
        
        @self.app.route('/api/command', methods=['POST'])
        def execute_command():
            if not self.trading_service:
                return jsonify({"error": "Trading service not available"})
            
            try:
                command = request.json.get('command')
                params = request.json.get('params', {})
                
                # 处理各种命令
                if command == 'start_trading':
                    self.trading_service.start()
                    return jsonify({"success": True, "message": "Trading started"})
                
                elif command == 'stop_trading':
                    self.trading_service.stop()
                    return jsonify({"success": True, "message": "Trading stopped"})
                
                elif command == 'pause_trading':
                    self.trading_service._pause_trading()
                    return jsonify({"success": True, "message": "Trading paused"})
                
                elif command == 'resume_trading':
                    self.trading_service._resume_trading()
                    return jsonify({"success": True, "message": "Trading resumed"})
                
                elif command == 'close_position':
                    position = self.trading_service.position_tracker.get_current_position()
                    self.trading_service._close_position(position, "Manual close from UI")
                    return jsonify({"success": True, "message": "Position closing initiated"})
                
                else:
                    return jsonify({"error": f"Unknown command: {command}"})
                
            except Exception as e:
                self.logger.error(f"执行命令出错: {e}")
                return jsonify({"error": str(e)})
        
        # SocketIO事件处理
        @self.socketio.on('connect')
        def handle_connect():
            self.logger.info(f"新的SocketIO客户端连接")
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.logger.info(f"SocketIO客户端断开连接")
    
    def _setup_data_subscriptions(self):
        """设置数据订阅，以获取实时更新"""
        # 订阅市场数据
        if hasattr(self.trading_service.trading_env, 'on_market_data'):
            self.trading_service.trading_env.on_market_data = self._on_market_data_update
        
        # 订阅仓位数据
        if hasattr(self.trading_service.position_tracker, 'on_position_update'):
            self.trading_service.position_tracker.on_position_update = self._on_position_update
        
        # 订阅订单数据
        if hasattr(self.trading_service.order_manager, 'on_order_update'):
            self.trading_service.order_manager.on_order_update = self._on_order_update
        
        # 订阅预测数据
        if hasattr(self.trading_service.data_recorder, 'on_model_prediction'):
            self.trading_service.data_recorder.on_model_prediction = self._on_prediction_update
        
        # 订阅警报数据
        if hasattr(self.trading_service.system_monitor, 'on_alert'):
            # 保存原始回调函数
            original_on_alert = self.trading_service.system_monitor.on_alert
            
            # 创建新的回调，先处理警报，然后更新UI
            @wraps(original_on_alert)
            def wrapped_on_alert(alert):
                # 调用原始回调
                if original_on_alert:
                    original_on_alert(alert)
                # 然后更新UI
                self._on_alert_update(alert)
            
            # 替换回调函数
            self.trading_service.system_monitor.on_alert = wrapped_on_alert
    
    def _on_market_data_update(self, data):
        """处理市场数据更新"""
        # 添加时间戳
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        
        # 添加到缓存
        self.market_data_cache.append(data)
        
        # 限制缓存大小
        while len(self.market_data_cache) > self.max_data_points:
            self.market_data_cache.pop(0)
        
        # 通过SocketIO发送更新
        self.socketio.emit('market_update', data)
    
    def _on_position_update(self, data):
        """处理仓位数据更新"""
        # 添加时间戳
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        
        # 添加到缓存
        self.position_data_cache.append(data)
        
        # 限制缓存大小
        while len(self.position_data_cache) > self.max_data_points:
            self.position_data_cache.pop(0)
        
        # 通过SocketIO发送更新
        self.socketio.emit('position_update', data)
    
    def _on_order_update(self, data):
        """处理订单数据更新"""
        # 添加时间戳
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        
        # 添加到缓存
        self.order_data_cache.append(data)
        
        # 限制缓存大小
        while len(self.order_data_cache) > 100:  # 订单缓存可以稍小一些
            self.order_data_cache.pop(0)
        
        # 通过SocketIO发送更新
        self.socketio.emit('order_update', data)
    
    def _on_prediction_update(self, data):
        """处理模型预测更新"""
        # 添加时间戳
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        
        # 添加到缓存
        self.prediction_data_cache.append(data)
        
        # 限制缓存大小
        while len(self.prediction_data_cache) > self.max_data_points:
            self.prediction_data_cache.pop(0)
        
        # 通过SocketIO发送更新
        self.socketio.emit('prediction_update', data)
    
    def _on_alert_update(self, data):
        """处理警报更新"""
        # 添加时间戳
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().timestamp()
        
        # 添加到缓存
        self.alert_data_cache.append(data)
        
        # 限制缓存大小
        while len(self.alert_data_cache) > 50:  # 警报缓存可以较小
            self.alert_data_cache.pop(0)
        
        # 通过SocketIO发送更新
        self.socketio.emit('alert', data)
    
    def start_update_thread(self):
        """启动定期更新线程"""
        def update_loop():
            while self.is_running:
                try:
                    # 如果交易服务可用，获取并发送最新状态
                    if self.trading_service:
                        status = self.trading_service.get_status()
                        self.socketio.emit('status_update', status)
                    
                    # 睡眠至下一次更新
                    time.sleep(self.update_interval / 1000)
                    
                except Exception as e:
                    self.logger.error(f"更新线程出错: {e}")
                    time.sleep(5)  # 出错后等待一段时间
        
        # 创建并启动线程
        self.update_thread = threading.Thread(target=update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def start(self):
        """启动UI服务器"""
        if self.is_running:
            self.logger.warning("UI服务器已经在运行中")
            return
        
        self.logger.info(f"正在启动UI服务器，端口: {self.http_port}")
        
        try:
            # 检查端口是否可用
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', self.http_port))
            sock.close()
            
            if result == 0:
                self.logger.warning(f"端口 {self.http_port} 已被占用，UI服务器可能无法启动")
            
            # 启动定期更新线程
            self.is_running = True
            self.start_update_thread()
            
            # 启动Flask服务
            self.socketio.run(self.app, host='0.0.0.0', port=self.http_port, debug=False)
            
        except Exception as e:
            self.logger.error(f"启动UI服务器失败: {e}")
            self.is_running = False
    
    def stop(self):
        """停止UI服务器"""
        if not self.is_running:
            return
        
        self.logger.info("正在停止UI服务器...")
        self.is_running = False
        
        # 停止SocketIO服务
        # 注意：Flask开发服务器没有优雅的停止方法
        # 在生产环境中，应该使用正确的WSGI服务器（如gunicorn）
        # 并通过进程管理来管理它

# 命令行入口
def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='交易系统UI服务')
    parser.add_argument('--config', '-c', type=str, required=True, help='配置文件路径')
    args = parser.parse_args()
    
    # 创建并启动UI服务
    server = UIServer(args.config)
    server.start()

if __name__ == '__main__':
    main()
