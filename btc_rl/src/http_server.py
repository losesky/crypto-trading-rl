#!/usr/bin/env python3
"""
HTTP服务器，用于提供静态文件服务（可视化前端）
"""

import os
import logging
import sys
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
    # 检查可视化目录是否存在
    if not os.path.exists(VISUALIZER_DIR):
        logger.error(f"可视化目录不存在: {VISUALIZER_DIR}")
        logger.info(f"当前工作目录: {os.getcwd()}")
        logger.info(f"尝试创建可视化目录...")
        try:
            os.makedirs(VISUALIZER_DIR, exist_ok=True)
        except Exception as e:
            logger.error(f"创建目录失败: {e}")
            return
    
    os.chdir(VISUALIZER_DIR)  # 更改工作目录到可视化目录
    handler = partial(VisualizerHTTPRequestHandler, directory=VISUALIZER_DIR)
    
    # 尝试绑定不同端口
    httpd = None
    for port_offset in range(5):  # 尝试5个不同的端口
        try:
            current_port = HTTP_PORT + port_offset
            httpd = HTTPServer((HTTP_HOST, current_port), handler)
            logger.info(f"HTTP服务器已启动，监听所有接口 {HTTP_HOST}:{current_port}")
            logger.info(f"您可以在浏览器中访问: ")
            logger.info(f"  - 本地访问: http://localhost:{current_port}/index.html")
            logger.info(f"  - 局域网/远程访问: http://[您的IP地址]:{current_port}/index.html")
            break
        except OSError as e:
            logger.warning(f"端口 {current_port} 已被占用，尝试下一个端口... ({e})")
            if port_offset == 4:  # 最后一次尝试
                logger.error("无法启动HTTP服务器，所有尝试的端口都被占用")
                return
    
    if httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("服务器被手动停止")
        finally:
            httpd.server_close()
            logger.info("HTTP服务器已关闭")

def main():
    """作为模块导入时的入口点"""
    try:
        run_server()
    except Exception as e:
        logger.error(f"启动HTTP服务器时发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
