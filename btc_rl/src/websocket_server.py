#!/usr/bin/env python3
"""
WebSocket服务器，用于连接RL训练系统和可视化前端
"""

import asyncio
import json
import queue
import logging
from pathlib import Path
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

import websockets

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("websocket_server")

# WebSocket配置
WS_HOST = "localhost"
WS_PORT = 8765
HTTP_PORT = 8766  # 为HTTP服务使用不同的端口
WEBSOCKET_CLIENTS = set()
MESSAGE_QUEUE = queue.Queue(maxsize=100)  # 线程安全队列

async def websocket_handler(websocket):
    """处理新的WebSocket连接并保持连接活跃"""
    client_ip = websocket.remote_address[0] if websocket.remote_address else "未知"
    WEBSOCKET_CLIENTS.add(websocket)
    logger.info(f"客户端连接: {client_ip}, 当前连接数: {len(WEBSOCKET_CLIENTS)}")
    
    try:
        # 发送欢迎消息
        welcome_message = json.dumps({
            "type": "system_message",
            "content": "已连接到BTC交易强化学习可视化服务器"
        })
        await websocket.send(welcome_message)
        logger.info(f"已向客户端 {client_ip} 发送欢迎消息")
        
        # 保持连接并处理可能的消息
        while True:
            try:
                message = await websocket.recv()
                logger.info(f"收到客户端消息: {message[:50]}..." if len(message) > 50 else f"收到客户端消息: {message}")
                # 这里可以添加处理客户端发送的消息的逻辑
            except websockets.ConnectionClosed:
                break
    except websockets.ConnectionClosed:
        logger.info(f"连接关闭: {client_ip}")
    except Exception as e:
        logger.error(f"处理WebSocket连接时发生错误: {e}")
    finally:
        WEBSOCKET_CLIENTS.remove(websocket)
        logger.info(f"客户端断开: {client_ip}, 剩余连接数: {len(WEBSOCKET_CLIENTS)}")

async def broadcast_messages():
    """持续检查队列并广播消息给所有客户端"""
    current_loop = asyncio.get_running_loop()
    while True:
        try:
            if not MESSAGE_QUEUE.empty():
                # 获取队列中的消息
                message = await current_loop.run_in_executor(None, MESSAGE_QUEUE.get)
                
                # 消息日志记录
                log_message = message
                if isinstance(log_message, str) and len(log_message) > 100:
                    log_message = log_message[:100] + "..."
                logger.info(f"从队列取出消息: {log_message}")
                
                # 检查是否有连接的客户端
                if WEBSOCKET_CLIENTS:
                    logger.info(f"广播消息至 {len(WEBSOCKET_CLIENTS)} 个客户端")
                    
                    # 确保消息是字符串形式
                    if not isinstance(message, str):
                        try:
                            message = json.dumps(message)
                        except Exception as e:
                            logger.error(f"无法将消息转换为JSON: {e}")
                            message = json.dumps({"error": "无法序列化消息", "message": str(message)})
                    
                    # 检查消息是否为有效的JSON
                    try:
                        # 尝试解析确保这是有效的JSON
                        json.loads(message)
                    except json.JSONDecodeError as e:
                        logger.error(f"广播的消息不是有效的JSON: {e}")
                        message = json.dumps({"error": "无效JSON格式", "raw": message[:500]})
                    
                    # 尝试向每个客户端发送消息
                    failed_clients = []
                    for client in WEBSOCKET_CLIENTS:
                        try:
                            await client.send(message)
                        except Exception as e:
                            logger.error(f"向客户端 {client.remote_address if hasattr(client, 'remote_address') else '未知'} 发送消息失败: {e}")
                            failed_clients.append(client)
                    
                    # 移除发送失败的客户端
                    for client in failed_clients:
                        if client in WEBSOCKET_CLIENTS:
                            WEBSOCKET_CLIENTS.remove(client)
                            logger.warning(f"已移除无响应的客户端，剩余 {len(WEBSOCKET_CLIENTS)} 个连接")
                else:
                    logger.warning("没有连接的WebSocket客户端，消息无法广播")
            else:
                # 队列为空，短暂休眠
                await asyncio.sleep(0.01)
        except Exception as e:
            logger.error(f"广播出错: {e}")
            # 遇到错误时稍微延长休眠
            await asyncio.sleep(0.1)  # 避免在持续错误时CPU占用过高

async def start_server():
    """启动WebSocket服务器"""
    async with websockets.serve(websocket_handler, WS_HOST, WS_PORT):
        logger.info(f"WebSocket服务器已启动 ws://{WS_HOST}:{WS_PORT}")
        # 启动消息广播协程
        broadcast_task = asyncio.create_task(broadcast_messages())
        # 保持服务器运行
        await asyncio.Future()

def add_message_to_queue(message_data):
    """添加消息到广播队列，供外部调用"""
    try:
        # 标准化消息格式以确保前端能够正确解析
        if isinstance(message_data, dict):
            # 已经是JSON对象，直接序列化
            message = json.dumps(message_data)
        elif isinstance(message_data, str):
            # 如果是字符串，尝试解析为JSON对象再序列化
            # 这样可以确保我们有一致的单层JSON格式
            try:
                parsed = json.loads(message_data)
                message = json.dumps(parsed)
                logger.debug(f"成功解析字符串为JSON")
            except json.JSONDecodeError:
                # 如果不是有效的JSON，作为字符串包装在JSON对象中
                message = json.dumps({"type": "raw_message", "content": message_data})
                logger.warning(f"收到非JSON字符串，已包装为JSON对象")
        else:
            # 其他类型，包装为JSON对象
            message = json.dumps({"type": "unknown", "content": str(message_data)})
            logger.warning(f"收到未知类型数据，已转换为字符串并包装")
        
        # 记录完整的消息用于调试
        log_message = message
        if len(log_message) > 200:
            log_message = log_message[:200] + "..."
        logger.info(f"消息已添加到队列: {log_message}")
        
        # 将消息添加到队列
        MESSAGE_QUEUE.put(message)
        return True
    except queue.Full:
        logger.warning("消息队列已满，丢弃消息")
        return False
    except Exception as e:
        logger.error(f"处理消息时出错: {e}")
        return False

# HTTP消息接收器
class MessageHandler(BaseHTTPRequestHandler):
    """处理来自训练进程的HTTP消息请求"""
    
    # 存储最新的状态数据，用于HTTP API访问
    latest_data = {"step": 0, "price": 0, "margin_equity": 0, "action": 0, 
                   "position_btc": 0, "cash_balance": 0, "buy_and_hold_equity": 0,
                   "type": "status", "content": "没有数据"}
    
    def _send_cors_headers(self):
        """发送CORS头部"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
    
    def do_OPTIONS(self):
        """处理OPTIONS请求"""
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()
    
    def do_GET(self):
        """处理GET请求，返回最新的状态数据"""
        if self.path == "/status":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(self.latest_data).encode('utf-8'))
            logger.debug(f"返回状态数据: step={self.latest_data.get('step', 0)}")
            return
        else:
            # 不支持的路径
            self.send_response(404)
            self.send_header('Content-Type', 'application/json')
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"status": "error", "message": "Not found"}).encode('utf-8'))
    
    def do_POST(self):
        """处理POST请求，从训练进程接收消息并添加到队列"""
        if self.path == "/send_message":
            # 处理POST请求
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            
            try:
                # 解析请求JSON
                data = json.loads(post_data)
                
                # 详细记录请求内容以便调试
                logger.info(f"接收到POST请求数据: {post_data[:200]}..." if len(post_data) > 200 else f"接收到POST请求数据: {post_data}")
                
                if "message" in data:
                    message = data["message"]
                    logger.info(f"提取的消息内容: {str(message)[:200]}..." if len(str(message)) > 200 else f"提取的消息内容: {str(message)}")
                    
                    # 尝试解析消息并更新最新数据
                    try:
                        if isinstance(message, str):
                            parsed_message = json.loads(message)
                            # 如果是训练数据，更新状态
                            if isinstance(parsed_message, dict) and "step" in parsed_message:
                                MessageHandler.latest_data = parsed_message
                                logger.debug(f"更新状态数据: step={parsed_message.get('step', 0)}")
                    except Exception as e:
                        logger.error(f"解析消息并更新状态时出错: {e}")
                    
                    # 始终通过add_message_to_queue处理消息，让该函数处理格式化
                    add_message_to_queue(message)
                    
                    # 返回成功响应
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self._send_cors_headers()
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "success"}).encode('utf-8'))
                    return
                else:
                    # 指出特定的错误：缺少message字段
                    logger.warning(f"请求中缺少'message'字段: {post_data}")
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self._send_cors_headers()
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "error", "message": "Missing 'message' field"}).encode('utf-8'))
                    return
            except json.JSONDecodeError as e:
                # JSON解析错误，记录详细信息
                logger.error(f"JSON解析错误: {e}, 原始数据: {post_data}")
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": f"JSON解析错误: {str(e)}"}).encode('utf-8'))
                return
                
        # 如果到达这里，说明请求格式不正确
        self.send_response(400)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"status": "error", "message": "Invalid request format"}).encode('utf-8'))

def run_http_server(host="localhost", port=None):
    """在单独的线程中运行HTTP服务器"""
    global HTTP_PORT  # 在使用HTTP_PORT之前声明为全局变量
    
    if port is None:
        port = HTTP_PORT
    
    # 尝试最多3次，如果指定端口被占用则尝试下一个端口
    for attempt in range(3):
        try:
            current_port = port + attempt
            server = HTTPServer((host, current_port), MessageHandler)
            logger.info(f"HTTP消息接收器已启动 http://{host}:{current_port}/send_message")
            
            # 如果使用了不同的端口，更新全局HTTP_PORT
            if current_port != HTTP_PORT:
                HTTP_PORT = current_port
                logger.info(f"更新HTTP端口为: {HTTP_PORT}")
                
            # 启动服务器
            server.serve_forever()
            return server
        except OSError as e:
            if "Address already in use" in str(e):
                logger.warning(f"端口 {current_port} 已被占用，尝试下一个端口...")
            else:
                logger.error(f"启动HTTP服务器时发生错误: {e}")
                return None
    
    # 所有尝试失败
    logger.error(f"无法启动HTTP服务器，所有尝试的端口都被占用")
    return None

if __name__ == "__main__":
    try:
        # # 启动HTTP服务器线程
        # http_server_thread = threading.Thread(target=run_http_server, daemon=True)
        # http_server_thread.start()
        
        # 启动WebSocket服务器
        asyncio.run(start_server())
    except KeyboardInterrupt:
        logger.info("服务器被手动停止")
    except Exception as e:
        logger.error(f"服务器错误: {e}")
