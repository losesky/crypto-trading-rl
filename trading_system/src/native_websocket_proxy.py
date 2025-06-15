#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
原生WebSocket代理服务器 - 直接使用原生WebSocket协议与前端通信
"""
import json
import logging
import threading
import time
import os
import traceback
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
import websockets

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NativeWebSocketProxy")

# 存储所有连接的客户端
connected_clients = set()
# 最新数据存储
latest_data = {}

async def handle_client(websocket, path):
    """处理单个WebSocket客户端连接"""
    global connected_clients
    
    try:
        # 注册新客户端
        connected_clients.add(websocket)
        client_id = id(websocket)
        logger.info(f"新客户端连接 {client_id}, 当前连接数: {len(connected_clients)}")
        
        # 发送初始数据
        if latest_data:
            for event_type, data in latest_data.items():
                await websocket.send(json.dumps({"event": event_type, "data": data}))
        
        # 发送欢迎消息
        await websocket.send(json.dumps({
            "event": "system", 
            "data": {
                "message": "已连接到交易系统", 
                "timestamp": int(datetime.now().timestamp() * 1000)
            }
        }))
        
        # 处理接收的消息
        async for message in websocket:
            try:
                data = json.loads(message)
                
                # 处理心跳
                if "ping" in data:
                    await websocket.send(json.dumps({
                        "event": "pong", 
                        "data": {"timestamp": int(datetime.now().timestamp() * 1000)}
                    }))
            except json.JSONDecodeError:
                logger.warning(f"客户端 {client_id} 发送的非JSON消息: {message}")
            except Exception as e:
                logger.error(f"处理客户端 {client_id} 消息时出错: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"客户端 {client_id} 连接关闭")
    except Exception as e:
        logger.error(f"处理WebSocket连接时出错: {e}")
        logger.debug(traceback.format_exc())
    finally:
        # 移除已断开的客户端
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        logger.info(f"客户端断开连接, 当前连接数: {len(connected_clients)}")

async def broadcast_data(event_type, data):
    """广播数据到所有连接的客户端"""
    if not connected_clients:
        return
    
    # 确保数据有时间戳
    if isinstance(data, dict) and 'timestamp' not in data:
        data['timestamp'] = int(datetime.now().timestamp() * 1000)
    
    # 保存最新数据
    latest_data[event_type] = data
    
    # 构建消息
    message = json.dumps({"event": event_type, "data": data})
    
    # 广播给所有客户端
    disconnected_clients = set()
    for client in connected_clients:
        try:
            await client.send(message)
        except Exception as e:
            logger.error(f"发送数据到客户端时出错: {e}")
            disconnected_clients.add(client)
    
    # 清理断开连接的客户端
    for client in disconnected_clients:
        if client in connected_clients:
            connected_clients.remove(client)

# REST API服务，用于健康检查和数据发送
app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "running",
        "connections": len(connected_clients),
        "name": "Native WebSocket Proxy",
        "timestamp": int(datetime.now().timestamp() * 1000)
    })

# 提供API端点，生成模拟数据
@app.route('/api/market_data', methods=['GET'])
def api_market_data():
    """市场数据API"""
    data = {
        "symbol": "BTCUSDT",
        "price": 40000 + (time.time() % 1000),
        "volume": 100 + (time.time() % 50),
        "timestamp": int(datetime.now().timestamp() * 1000)
    }
    # 通过WebSocket广播同样的数据
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(broadcast_data("market_update", data))
    return jsonify(data)

@app.route('/api/predictions', methods=['GET'])
def api_predictions():
    """预测数据API"""
    data = {
        "symbol": "BTCUSDT",
        "prediction": 0.7 if time.time() % 2 > 1 else -0.3,
        "confidence": 0.8,
        "timestamp": int(datetime.now().timestamp() * 1000)
    }
    # 通过WebSocket广播同样的数据
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(broadcast_data("prediction_update", data))
    return jsonify(data)

@app.route('/api/positions', methods=['GET'])
def api_positions():
    """持仓数据API"""
    data = {
        "symbol": "BTCUSDT",
        "amount": 0.5,
        "entry_price": 38000,
        "current_price": 40000 + (time.time() % 1000),
        "pnl": 1000,
        "timestamp": int(datetime.now().timestamp() * 1000)
    }
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(broadcast_data("position_update", data))
    return jsonify(data)

@app.route('/api/orders', methods=['GET'])
def api_orders():
    """订单数据API"""
    data = {
        "orders": [
            {
                "id": "ord123456",
                "symbol": "BTCUSDT",
                "side": "buy",
                "price": 39500,
                "amount": 0.2,
                "status": "filled",
                "timestamp": int((datetime.now().timestamp() - 300) * 1000)
            }
        ],
        "timestamp": int(datetime.now().timestamp() * 1000)
    }
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(broadcast_data("order_update", data))
    return jsonify(data)

@app.route('/api/status', methods=['GET'])
def api_status():
    """状态数据API"""
    data = {
        "trading_enabled": True,
        "connection_status": "online",
        "last_update": int(datetime.now().timestamp() * 1000),
        "timestamp": int(datetime.now().timestamp() * 1000)
    }
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(broadcast_data("status_update", data))
    return jsonify(data)

@app.route('/send', methods=['POST'])
def send_data():
    """接收外部数据并转发到WebSocket客户端"""
    try:
        data = request.json
        if not data or 'event' not in data or 'data' not in data:
            return jsonify({"error": "Invalid data format"}), 400
        
        event_type = data['event']
        payload = data['data']
        
        # 使用asyncio运行广播任务
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(broadcast_data(event_type, payload))
        
        return jsonify({"success": True, "clients": len(connected_clients)}), 200
    except Exception as e:
        logger.error(f"处理数据发送请求时出错: {e}")
        return jsonify({"error": str(e)}), 500

class NativeWebSocketProxy:
    """原生WebSocket代理服务类，兼容现有API"""
    
    def __init__(self, port=8095, rest_port=8096):
        """初始化WebSocket代理服务"""
        self.logger = logging.getLogger("NativeWebSocketProxy")
        self.ws_port = port
        self.rest_port = rest_port
        self.is_running = False
        self.server = None
        self.rest_server = None
        self.ws_thread = None
        self.rest_thread = None
    
    def start(self):
        """启动WebSocket代理服务"""
        if self.is_running:
            self.logger.warning("WebSocket代理服务已在运行")
            return
        
        self.is_running = True
        self.logger.info(f"启动原生WebSocket代理服务，端口：{self.ws_port}")
        
        # 启动WebSocket服务器
        self.ws_thread = threading.Thread(target=self._run_websocket_server)
        self.ws_thread.daemon = True
        self.ws_thread.start()
        
        # 启动REST API服务器
        self.rest_thread = threading.Thread(target=self._run_rest_server)
        self.rest_thread.daemon = True
        self.rest_thread.start()
    
    def _run_websocket_server(self):
        """在单独线程中运行WebSocket服务器"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            start_server = websockets.serve(
                handle_client, 
                '0.0.0.0', 
                self.ws_port
            )
            
            self.server = loop.run_until_complete(start_server)
            self.logger.info(f"WebSocket服务器启动成功，监听端口 {self.ws_port}")
            
            loop.run_forever()
        except Exception as e:
            self.logger.error(f"启动WebSocket服务器时出错: {e}")
            self.logger.error(traceback.format_exc())
            self.is_running = False
    
    def _run_rest_server(self):
        """在单独线程中运行REST API服务器"""
        try:
            self.logger.info(f"REST API服务器启动中，端口 {self.rest_port}")
            app.run(host='0.0.0.0', port=self.rest_port, threaded=True)
        except Exception as e:
            self.logger.error(f"启动REST API服务器时出错: {e}")
            self.is_running = False
    
    def stop(self):
        """停止WebSocket代理服务"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.logger.info("停止WebSocket代理服务")
        
        # 关闭WebSocket服务器
        if self.server:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.server.close())
            self.server = None
        
        # REST API服务器无法优雅关闭，需要通过外部终止进程

    def send_data(self, event_type, data):
        """发送数据到所有连接的客户端（兼容旧API）"""
        if not self.is_running:
            return False
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(broadcast_data(event_type, data))
            return True
        except Exception as e:
            self.logger.error(f"发送数据失败: {e}")
            return False

# 单例模式
_instance = None

def get_instance(port=8095, rest_port=8096):
    """获取WebSocketProxy实例（单例模式）"""
    global _instance
    if _instance is None:
        _instance = NativeWebSocketProxy(port, rest_port)
    return _instance

# 命令行入口
if __name__ == "__main__":
    # 创建并启动代理服务
    proxy = get_instance(port=8095, rest_port=8096)
    proxy.start()
    
    try:
        logger.info("WebSocket代理服务已启动，按Ctrl+C停止...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("接收到停止信号，关闭服务...")
    finally:
        proxy.stop()
