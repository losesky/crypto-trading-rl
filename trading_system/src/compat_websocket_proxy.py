#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WebSocket代理服务器兼容启动器 - 支持多个版本的Flask-SocketIO
"""
import os
import sys
import time
import logging
import traceback
import importlib.util
import pkg_resources

# 确保正确设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))  # 项目根目录
sys.path.insert(0, parent_dir)

# 创建日志目录
logs_dir = os.path.join(os.path.dirname(current_dir), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(logs_dir, 'websocket_proxy.log'))
    ]
)
logger = logging.getLogger("WebSocketProxyCompatStarter")

def get_socketio_version():
    """获取Flask-SocketIO的版本"""
    try:
        version = pkg_resources.get_distribution("flask-socketio").version
        logger.info(f"Flask-SocketIO版本: {version}")
        return version
    except Exception as e:
        logger.warning(f"无法确定Flask-SocketIO版本: {e}")
        return None

def run_socketio_server(app, port=8095):
    """根据不同版本的Flask-SocketIO启动服务器"""
    from flask_socketio import SocketIO
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
    
    version = get_socketio_version()
    
    # 兼容不同版本的参数
    kwargs = {
        'host': '0.0.0.0',
        'port': port,
        'debug': False,
        'use_reloader': False
    }
    
    # Flask-SocketIO 5.x 及以上版本需要 allow_unsafe_werkzeug 参数
    if version and int(version.split('.')[0]) >= 5:
        logger.info("使用Flask-SocketIO 5.x+参数启动")
        kwargs['allow_unsafe_werkzeug'] = True
    else:
        logger.info("使用Flask-SocketIO旧版参数启动")
    
    try:
        socketio.run(app, **kwargs)
    except TypeError as e:
        # 如果发现TypeError，可能是参数不兼容，尝试移除新参数
        logger.warning(f"启动错误，尝试兼容模式: {e}")
        if 'allow_unsafe_werkzeug' in kwargs:
            del kwargs['allow_unsafe_werkzeug']
            logger.info("去掉allow_unsafe_werkzeug参数重试")
            socketio.run(app, **kwargs)
        else:
            raise
    
    return socketio

def create_standalone_app():
    """创建一个独立的Flask应用"""
    from flask import Flask, jsonify
    from flask_cors import CORS
    
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/health')
    def health_check():
        return jsonify({"status": "ok", "standalone": True})
    
    return app

if __name__ == "__main__":
    try:
        # 尝试导入WebSocket代理
        logger.info("尝试启动WebSocket代理...")
        try:
            # 先尝试标准导入路径
            from trading_system.src.websocket_proxy import get_instance
            proxy = get_instance()
            proxy.start()
            logger.info("WebSocket代理已启动")
            
            # 保持运行
            try:
                logger.info("WebSocket代理服务已启动，按Ctrl+C停止...")
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("接收到停止信号，关闭服务...")
                proxy.stop()
                
        except (ImportError, RuntimeError) as e:
            logger.warning(f"标准导入方式失败: {e}, 尝试兼容模式...")
            
            # 如果标准方式失败，尝试创建独立应用和服务器
            app = create_standalone_app()
            
            logger.info("使用兼容模式启动WebSocket服务器...")
            socketio = run_socketio_server(app)
            
            # 服务器会阻塞在这里直到终止
            
    except Exception as e:
        logger.error(f"启动WebSocket代理时出错: {e}")
        traceback.print_exc()
        sys.exit(1)
