#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
API代理服务器 - 为前端提供REST API和静态文件服务
"""
import json
import logging
import threading
import time
import os
from datetime import datetime
from flask import Flask, request, jsonify, Response, send_from_directory, send_file, abort
from flask_cors import CORS
import requests

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("APIProxyServer")

# 请求缓存和时间戳
cache = {}
cache_timestamps = {}
cache_ttl = 5  # 缓存有效期（秒）

# 创建Flask应用
app = Flask(__name__)
CORS(app)

# 默认为主交易系统的地址
TRADING_SYSTEM_URL = os.environ.get("TRADING_SYSTEM_URL", "http://localhost:8091")

# 获取WebSocket代理地址
WS_PROXY_URL = os.environ.get("WS_PROXY_URL", "http://localhost:8096")

def get_cached_or_fetch(url, cache_key, default_value=None):
    """获取缓存数据或从远程获取"""
    now = time.time()
    
    # 检查缓存是否有效
    if cache_key in cache and cache_timestamps.get(cache_key, 0) + cache_ttl > now:
        logger.debug(f"使用缓存数据: {cache_key}")
        return cache[cache_key]
    
    try:
        # 从远程服务获取数据
        logger.debug(f"从远程获取数据: {url}")
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            try:
                data = response.json()
                # 更新缓存
                cache[cache_key] = data
                cache_timestamps[cache_key] = now
                return data
            except json.JSONDecodeError:
                logger.error(f"无法解析响应JSON: {response.text[:100]}")
        else:
            logger.warning(f"请求失败: {url}, 状态码: {response.status_code}")
    
    except requests.RequestException as e:
        logger.warning(f"请求异常: {url}, {str(e)}")
    
    # 如果远程获取失败，返回缓存的旧数据（如果有）
    if cache_key in cache:
        logger.debug(f"使用过期缓存数据: {cache_key}")
        return cache[cache_key]
    
    # 都没有时返回默认值
    return default_value

def forward_to_websocket(event_type, data):
    """将数据转发到WebSocket代理"""
    try:
        # 设置较长的超时时间，避免过早放弃
        response = requests.post(f"{WS_PROXY_URL}/send", 
                     json={"event": event_type, "data": data}, 
                     timeout=5)
        
        # 检查响应，记录任何错误
        if response.status_code != 200:
            logger.warning(f"WebSocket转发返回非成功状态码: {response.status_code}, 响应: {response.text[:100]}")
            
    except requests.RequestException as e:
        logger.warning(f"转发到WebSocket失败: {str(e)}")
        # 在连接失败的情况下，尝试不同的URL格式
        try:
            alternate_url = WS_PROXY_URL.replace("http://", "ws://")
            logger.info(f"尝试替代URL: {alternate_url}/send")
            requests.post(f"{alternate_url}/send",
                         json={"event": event_type, "data": data},
                         timeout=5)
        except:
            pass  # 如果替代尝试也失败，继续执行

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "running",
        "api_proxy": True,
        "timestamp": int(datetime.now().timestamp() * 1000)
    })

# 静态文件服务，用于提供UI文件
@app.route('/', defaults={'path': 'index.html'})
@app.route('/<path:path>')
def serve_static(path):
    """提供UI静态文件"""
    # 基于相对路径查找UI目录
    ui_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../ui'))
    # 用于调试
    logger.debug(f"请求文件: {path}, UI目录: {ui_dir}")
    
    try:
        # 检查文件是否存在
        if os.path.isfile(os.path.join(ui_dir, path)):
            return send_from_directory(ui_dir, path)
        elif path.endswith('/') and os.path.isfile(os.path.join(ui_dir, f"{path}index.html")):
            return send_from_directory(ui_dir, f"{path}index.html")
        elif os.path.isfile(os.path.join(ui_dir, "index.html")) and '.' not in path:
            # 对于不含扩展名的路径，可能是SPA应用的路由
            return send_from_directory(ui_dir, "index.html")
        else:
            logger.warning(f"文件未找到: {path}")
            return jsonify({"error": "文件未找到"}), 404
    except Exception as e:
        logger.error(f"提供静态文件时出错: {path}, {str(e)}")
        return jsonify({"error": "服务器错误"}), 500

@app.route('/api/status', methods=['GET'])
def api_status():
    """系统状态API，尝试从主系统获取，如获取失败则返回基本状态"""
    try:
        # 尝试从主交易系统获取状态
        status_data = get_cached_or_fetch(
            f"{TRADING_SYSTEM_URL}/api/status", 
            "status", 
            {
                "trading_enabled": True,
                "connection_status": "online",
                "last_update": int(datetime.now().timestamp() * 1000),
                "timestamp": int(datetime.now().timestamp() * 1000)
            }
        )
        
        # 转发到WebSocket
        forward_to_websocket("status_update", status_data)
        
        return jsonify(status_data)
    except Exception as e:
        logger.error(f"获取状态数据出错: {e}")
        # 返回基本状态
        return jsonify({
            "trading_enabled": True,
            "connection_status": "online",
            "last_update": int(datetime.now().timestamp() * 1000),
            "timestamp": int(datetime.now().timestamp() * 1000)
        })

@app.route('/api/market_data', methods=['GET'])
def api_market_data():
    """市场数据API，从主交易系统获取实际数据"""
    try:
        # 从主交易系统获取市场数据
        market_data = get_cached_or_fetch(
            f"{TRADING_SYSTEM_URL}/api/market_data", 
            "market_data",
            {
                "symbol": "BTCUSDT",
                "price": 40000,
                "timestamp": int(datetime.now().timestamp() * 1000)
            }
        )
        
        # 确保时间戳是最新的
        if "timestamp" not in market_data:
            market_data["timestamp"] = int(datetime.now().timestamp() * 1000)
        
        # 转发到WebSocket
        forward_to_websocket("market_update", market_data)
        
        return jsonify(market_data)
    except Exception as e:
        logger.error(f"获取市场数据出错: {e}")
        return jsonify({
            "symbol": "BTCUSDT",
            "price": 40000,
            "timestamp": int(datetime.now().timestamp() * 1000)
        })

@app.route('/api/predictions', methods=['GET'])
def api_predictions():
    """预测数据API，从主交易系统获取实际预测"""
    try:
        # 从主交易系统获取预测数据
        prediction_data = get_cached_or_fetch(
            f"{TRADING_SYSTEM_URL}/api/predictions", 
            "predictions",
            {
                "action": "HOLD",
                "confidence": 0.5,
                "timestamp": int(datetime.now().timestamp() * 1000)
            }
        )
        
        # 确保时间戳是最新的
        if "timestamp" not in prediction_data:
            prediction_data["timestamp"] = int(datetime.now().timestamp() * 1000)
        
        # 转发到WebSocket
        forward_to_websocket("prediction_update", prediction_data)
        
        return jsonify(prediction_data)
    except Exception as e:
        logger.error(f"获取预测数据出错: {e}")
        return jsonify({
            "action": "HOLD",
            "confidence": 0.5,
            "timestamp": int(datetime.now().timestamp() * 1000)
        })

@app.route('/api/positions', methods=['GET'])
@app.route('/api/position_data', methods=['GET'])
def api_positions():
    """持仓数据API，从主交易系统获取实际持仓"""
    try:
        # 尝试从两个可能的API端点获取数据
        position_data = None
        
        # 首先尝试 position_data 端点
        try:
            position_data = get_cached_or_fetch(
                f"{TRADING_SYSTEM_URL}/api/position_data", 
                "position_data",
                None
            )
        except:
            pass
            
        # 如果失败，尝试 positions 端点
        if not position_data:
            position_data = get_cached_or_fetch(
                f"{TRADING_SYSTEM_URL}/api/positions", 
                "positions",
                {
                    "symbol": "BTCUSDT",
                    "amount": 0,
                    "entry_price": 0,
                    "current_price": 0,
                    "pnl": 0,
                    "timestamp": int(datetime.now().timestamp() * 1000)
                }
            )
        
        # 确保时间戳是最新的
        if "timestamp" not in position_data:
            position_data["timestamp"] = int(datetime.now().timestamp() * 1000)
        
        # 转发到WebSocket
        forward_to_websocket("position_update", position_data)
        
        return jsonify(position_data)
    except Exception as e:
        logger.error(f"获取持仓数据出错: {e}")
        return jsonify({
            "symbol": "BTCUSDT",
            "amount": 0,
            "timestamp": int(datetime.now().timestamp() * 1000)
        })

@app.route('/api/orders', methods=['GET'])
def api_orders():
    """订单数据API，从主交易系统获取实际订单"""
    try:
        orders_data = get_cached_or_fetch(
            f"{TRADING_SYSTEM_URL}/api/orders", 
            "orders",
            {
                "orders": [],
                "timestamp": int(datetime.now().timestamp() * 1000)
            }
        )
        
        # 确保时间戳是最新的
        if "timestamp" not in orders_data:
            orders_data["timestamp"] = int(datetime.now().timestamp() * 1000)
        
        # 转发到WebSocket
        forward_to_websocket("order_update", orders_data)
        
        return jsonify(orders_data)
    except Exception as e:
        logger.error(f"获取订单数据出错: {e}")
        return jsonify({
            "orders": [],
            "timestamp": int(datetime.now().timestamp() * 1000)
        })

@app.route('/api/alerts', methods=['GET'])
def api_alerts():
    """告警数据API，从主交易系统获取实际告警"""
    try:
        alerts_data = get_cached_or_fetch(
            f"{TRADING_SYSTEM_URL}/api/alerts", 
            "alerts",
            {
                "alerts": [],
                "timestamp": int(datetime.now().timestamp() * 1000)
            }
        )
        
        # 确保时间戳是最新的
        if "timestamp" not in alerts_data:
            alerts_data["timestamp"] = int(datetime.now().timestamp() * 1000)
        
        # 转发到WebSocket
        forward_to_websocket("alert", alerts_data)
        
        return jsonify(alerts_data)
    except Exception as e:
        logger.error(f"获取告警数据出错: {e}")
        return jsonify({
            "alerts": [],
            "timestamp": int(datetime.now().timestamp() * 1000)
        })

# 转发所有其他API请求
@app.route('/api/<path:subpath>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def forward_api(subpath):
    """将所有其他API请求转发到主系统"""
    url = f"{TRADING_SYSTEM_URL}/api/{subpath}"
    
    try:
        if request.method == 'GET':
            response = requests.get(url, params=request.args, timeout=10)
        elif request.method == 'POST':
            response = requests.post(url, json=request.get_json(), timeout=10)
        elif request.method == 'PUT':
            response = requests.put(url, json=request.get_json(), timeout=10)
        elif request.method == 'DELETE':
            response = requests.delete(url, params=request.args, timeout=10)
            
        return Response(
            response.content,
            status=response.status_code,
            content_type=response.headers['Content-Type']
        )
    except requests.RequestException as e:
        logger.error(f"转发API请求失败: {url}, {str(e)}")
        return jsonify({"error": "无法连接到交易系统"}), 502

class APIProxyServer:
    def __init__(self, port=8090, trading_url=None, ws_url=None):
        self.port = port
        self.is_running = False
        self.server_thread = None
        
        # 更新全局变量
        global TRADING_SYSTEM_URL, WS_PROXY_URL
        if trading_url:
            TRADING_SYSTEM_URL = trading_url
        if ws_url:
            WS_PROXY_URL = ws_url
            
    def start(self):
        """启动API代理服务器"""
        if self.is_running:
            logger.warning("API代理服务器已在运行")
            return
        
        self.is_running = True
        logger.info(f"启动API代理服务器，端口：{self.port}")
        
        # 在单独的线程中启动服务器
        self.server_thread = threading.Thread(
            target=lambda: app.run(
                host='0.0.0.0',
                port=self.port,
                debug=False,
                threaded=True,
                use_reloader=False
            )
        )
        self.server_thread.daemon = True
        self.server_thread.start()
        
    def stop(self):
        """停止API代理服务器"""
        if not self.is_running:
            return
            
        self.is_running = False
        logger.info("停止API代理服务器")
        # Flask服务器无法优雅关闭，需要通过外部终止进程

# 单例模式
_instance = None

def get_instance(port=8090, trading_url=None, ws_url=None):
    """获取APIProxyServer实例（单例模式）"""
    global _instance
    if _instance is None:
        _instance = APIProxyServer(port, trading_url, ws_url)
    return _instance

# 命令行入口
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="API代理服务器")
    parser.add_argument("--port", type=int, default=8090, help="服务器端口")
    parser.add_argument("--trading-url", default="http://localhost:8091", help="交易系统URL")
    parser.add_argument("--ws-url", default="http://localhost:8096", help="WebSocket代理URL")
    
    args = parser.parse_args()
    
    # 创建并启动代理服务
    proxy = get_instance(args.port, args.trading_url, args.ws_url)
    proxy.start()
    
    try:
        logger.info("API代理服务器已启动，按Ctrl+C停止...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("接收到停止信号，关闭服务...")
    finally:
        proxy.stop()