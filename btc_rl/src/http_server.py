#!/usr/bin/env python3
"""
HTTP服务器，用于提供静态文件服务（可视化前端）
"""

import os
import logging
from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools import partial

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("http_server")

# HTTP服务器配置
HTTP_HOST = "0.0.0.0"  # 修改为0.0.0.0以允许所有网络接口连接
HTTP_PORT = 8080
# 更新路径，因为文件位置变了
VISUALIZER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "visualizer")

class VisualizerHTTPRequestHandler(SimpleHTTPRequestHandler):
    """自定义HTTP请求处理器，提供静态文件服务"""
    
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)
    
    def log_message(self, format, *args):
        """重写日志方法，使用我们配置的logger"""
        logger.info(f"{self.address_string()} - {format % args}")

def run_server():
    """运行HTTP服务器"""
    os.chdir(VISUALIZER_DIR)  # 更改工作目录到可视化目录
    handler = partial(VisualizerHTTPRequestHandler, directory=VISUALIZER_DIR)
    httpd = HTTPServer((HTTP_HOST, HTTP_PORT), handler)
    
    logger.info(f"HTTP服务器已启动，监听所有接口 {HTTP_HOST}:{HTTP_PORT}")
    logger.info(f"您可以在浏览器中访问: ")
    logger.info(f"  - 本地访问: http://localhost:{HTTP_PORT}/index.html")
    logger.info(f"  - 局域网/远程访问: http://[您的IP地址]:{HTTP_PORT}/index.html")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("服务器被手动停止")
    finally:
        httpd.server_close()
        logger.info("HTTP服务器已关闭")

if __name__ == "__main__":
    run_server()
