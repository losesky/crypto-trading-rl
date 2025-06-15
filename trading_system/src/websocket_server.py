#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebSocket服务器 - 为UI提供实时数据更新
"""

import asyncio
import json
import logging
import websockets
import threading
import time
import random
from datetime import datetime, timedelta
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("WebSocketServer")

# 全局变量
WEBSOCKET_CLIENTS = set()
is_running = True
stop_event = threading.Event()

# 默认WebSocket服务器配置
WS_HOST = "0.0.0.0"  # 监听所有网络接口
WS_PORT = 8095  # 默认端口，可通过命令行参数修改

# 存储最新数据
latest_data = {
    "market_data": None,
    "position_data": None,
    "prediction_data": None,
    "order_data": [],
    "system_status": None
}

async def handler(websocket, path):
    """处理新的WebSocket连接"""
    client_ip = websocket.remote_address[0] if hasattr(websocket, 'remote_address') else "未知"
    logger.info(f"新的WebSocket连接: {client_ip}")
    
    # 添加到客户端集合
    WEBSOCKET_CLIENTS.add(websocket)
    
    try:
        # 发送连接确认
        await websocket.send(json.dumps({
            "type": "system_message",
            "message": "已连接到交易系统WebSocket服务器"
        }))
        
        # 循环接收消息
        while True:
            try:
                message = await websocket.recv()
                logger.info(f"接收到消息: {message[:100]}" + ("..." if len(message) > 100 else ""))
                
                # 处理收到的消息
                await process_message(message, websocket)
                
            except websockets.ConnectionClosed:
                logger.info(f"客户端连接已关闭: {client_ip}")
                break
    finally:
        WEBSOCKET_CLIENTS.remove(websocket)
        logger.info(f"客户端断开连接: {client_ip}")

async def process_message(message_str, sender):
    """处理收到的消息"""
    try:
        message = json.loads(message_str)
        
        # 根据消息类型处理
        message_type = message.get("type", "unknown")
        
        if message_type == "ping":
            # 心跳检测
            await sender.send(json.dumps({"type": "pong", "time": time.time() * 1000}))
        
        elif message_type == "subscribe":
            # 订阅特定数据流
            topics = message.get("topics", [])
            logger.info(f"客户端订阅: {topics}")
            
        elif message_type == "command":
            # 处理命令
            command = message.get("command")
            logger.info(f"收到命令: {command}")
            
            # 这里可以添加命令处理逻辑
            
        else:
            logger.warning(f"未知消息类型: {message_type}")
            
    except json.JSONDecodeError:
        logger.error("无效的JSON消息")
    except Exception as e:
        logger.error(f"处理消息时出错: {e}")

async def broadcast(message):
    """广播消息给所有连接的客户端"""
    if not WEBSOCKET_CLIENTS:
        return
        
    message_json = json.dumps(message)
    disconnected = set()
    
    for client in WEBSOCKET_CLIENTS:
        try:
            await client.send(message_json)
        except websockets.ConnectionClosed:
            disconnected.add(client)
        except Exception as e:
            logger.error(f"向客户端发送消息时出错: {e}")
            disconnected.add(client)
    
    # 移除断开连接的客户端
    for client in disconnected:
        WEBSOCKET_CLIENTS.remove(client)

async def generate_sample_data():
    """不再生成样例数据，仅保持服务器运行"""
    logger.info("WebSocket服务器已启动，等待真实数据...")
    
    while not stop_event.is_set():
        # 等待真实数据流，不再生成模拟数据
        await asyncio.sleep(10)  # 每10秒检查一次是否应该停止
    
    logger.info("WebSocket服务器数据处理已停止")

async def start_server():
    """启动WebSocket服务器"""
    server = await websockets.serve(handler, WS_HOST, WS_PORT)
    logger.info(f"WebSocket服务器已启动: ws://{WS_HOST}:{WS_PORT}")
    
    # 启动HTTP服务器接收真实数据 (使用WS_PORT+1作为HTTP端口)
    http_server = start_http_server(WS_PORT + 1)
    if http_server:
        logger.info(f"HTTP数据接收服务器已启动: http://{WS_HOST}:{WS_PORT + 1}")
    
    # 启动保持活跃任务
    asyncio.create_task(generate_sample_data())
    
    # 保持服务器运行
    await asyncio.Future()

def run_server():
    """在主线程中运行服务器"""
    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        logger.info("服务器停止: 收到键盘中断")
    except Exception as e:
        logger.error(f"服务器错误: {e}")
    finally:
        stop_event.set()
        logger.info("WebSocket服务器已关闭")

class DataHandler(http.server.SimpleHTTPRequestHandler):
    """处理来自交易系统的数据请求"""
    
    def do_POST(self):
        """处理POST请求，接收交易系统发送的数据"""
        global latest_data
        
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        
        try:
            data = json.loads(post_data)
            
            # 处理不同类型的数据
            if 'type' in data:
                data_type = data['type']
                
                if data_type == 'market_data':
                    latest_data['market_data'] = data
                    asyncio.run_coroutine_threadsafe(broadcast(data), asyncio.get_event_loop())
                    logger.info(f"已接收并广播市场数据: price={data.get('price', 'unknown')}")
                    
                elif data_type == 'position_data':
                    latest_data['position_data'] = data
                    asyncio.run_coroutine_threadsafe(broadcast(data), asyncio.get_event_loop())
                    logger.info(f"已接收并广播持仓数据")
                    
                elif data_type == 'prediction_data':
                    latest_data['prediction_data'] = data
                    asyncio.run_coroutine_threadsafe(broadcast(data), asyncio.get_event_loop())
                    logger.info(f"已接收并广播预测数据: action={data.get('action', 'unknown')}")
                    
                elif data_type == 'order_data':
                    # 添加到订单列表开头，限制长度
                    latest_data['order_data'].insert(0, data)
                    if len(latest_data['order_data']) > 20:
                        latest_data['order_data'].pop()
                    asyncio.run_coroutine_threadsafe(broadcast(data), asyncio.get_event_loop())
                    logger.info(f"已接收并广播订单数据")
                    
                elif data_type == 'system_status':
                    latest_data['system_status'] = data
                    asyncio.run_coroutine_threadsafe(broadcast(data), asyncio.get_event_loop())
                    logger.info(f"已接收并广播系统状态更新")
                    
            # 返回成功响应
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success"}).encode('utf-8'))
            
        except json.JSONDecodeError as e:
            # 处理JSON解析错误
            logger.error(f"JSON解析错误: {e}")
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "error", "message": "Invalid JSON"}).encode('utf-8'))
        except Exception as e:
            # 处理其他错误
            logger.error(f"处理POST请求时发生错误: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode('utf-8'))
    
    def do_GET(self):
        """处理GET请求，提供最新数据访问"""
        parsed_path = urlparse(self.path)
        
        # 返回数据的API端点
        if parsed_path.path == '/api/latest_data':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(latest_data).encode('utf-8'))
            return
        
        # 默认返回404
        self.send_response(404)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({"status": "error", "message": "Not found"}).encode('utf-8'))

def start_http_server(port=8096):
    """启动HTTP服务器接收数据"""
    try:
        http_server = socketserver.ThreadingTCPServer(("0.0.0.0", port), DataHandler)
        logger.info(f"HTTP数据接收服务器已启动在端口 {port}")
        
        # 在线程中运行HTTP服务器
        http_thread = threading.Thread(target=http_server.serve_forever)
        http_thread.daemon = True
        http_thread.start()
        
        return http_server
    except Exception as e:
        logger.error(f"启动HTTP服务器失败: {e}")
        return None

if __name__ == "__main__":
    import argparse
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='WebSocket服务器')
    parser.add_argument('--port', type=int, default=WS_PORT, help='WebSocket服务器端口')
    parser.add_argument('--host', type=str, default=WS_HOST, help='WebSocket服务器主机')
    args = parser.parse_args()
    
    # 更新全局配置
    if args.port:
        WS_PORT = args.port
        logger.info(f"使用命令行指定的端口: {WS_PORT}")
    if args.host:
        WS_HOST = args.host
        logger.info(f"使用命令行指定的主机: {WS_HOST}")
    
    try:
        # 启动HTTP服务器
        start_http_server(8096)
        
        run_server()
    except KeyboardInterrupt:
        logger.info("程序停止: 收到键盘中断")
        stop_event.set()
