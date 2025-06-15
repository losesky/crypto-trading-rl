#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
原生WebSocket代理启动器脚本
"""
import os
import sys
import logging
import traceback

# 确保正确设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))  # 项目根目录
sys.path.insert(0, parent_dir)

# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NativeWebSocketStarter")

try:
    logger.info("启动原生WebSocket代理服务器...")
    
    # 检查所需模块
    required_packages = ['websockets']
    missing_packages = []
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing_packages.append(pkg)
    
    if missing_packages:
        logger.error(f"缺少必要的包: {', '.join(missing_packages)}")
        logger.info("正在安装缺失的包...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            logger.info("依赖安装完成，继续启动...")
        except Exception as e:
            logger.error(f"安装依赖失败: {str(e)}")
            sys.exit(1)
    
    # 引入原生WebSocket代理
    from native_websocket_proxy import get_instance
    
    # 获取端口信息，默认使用8095和8096
    ws_port = int(os.environ.get("WS_PORT", 8095))
    rest_port = int(os.environ.get("WS_REST_PORT", 8096))
    
    # 创建并启动代理
    proxy = get_instance(port=ws_port, rest_port=rest_port)
    proxy.start()
    
    logger.info(f"WebSocket代理服务已启动，WebSocket端口: {ws_port}, REST API端口: {rest_port}")
    
    # 保持运行状态
    import time
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    logger.info("收到停止信号，正在关闭...")
    if 'proxy' in locals():
        proxy.stop()
except Exception as e:
    logger.error(f"启动WebSocket代理时出错: {str(e)}")
    traceback.print_exc()
    sys.exit(1)
