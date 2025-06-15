#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主入口模块 - 用于启动交易系统
"""
import os
import sys
import time
import logging
import json
import threading
import argparse
import signal
from pathlib import Path

def setup_logger():
    """设置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('trading_system.log')
        ]
    )
    return logging.getLogger('TradingSystem')

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="比特币自动交易系统")
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--mode', default='test', choices=['test', 'prod'], help='运行模式: test或prod')
    parser.add_argument('--ui-only', action='store_true', help='只启动UI界面')
    return parser.parse_args()

def check_dependencies():
    """检查并安装必要的依赖"""
    logger = logging.getLogger("TradingSystem")
    required_packages = {
        "flask_cors": "flask-cors",
        "flask_socketio": "flask-socketio",
        "stable_baselines3": "stable-baselines3[extra]", 
        "ccxt": "ccxt",
        "pandas": "pandas",
        "numpy": "numpy",
        "torch": "torch"
    }
    
    missing_packages = []
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"检测到缺少依赖项: {', '.join(missing_packages)}")
        try:
            import subprocess
            for package in missing_packages:
                logger.info(f"正在安装 {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
            logger.info("所有依赖项安装完成")
        except Exception as e:
            logger.error(f"安装依赖项失败: {e}")
            return False
    return True

def main():
    """主函数"""
    logger = setup_logger()
    args = parse_arguments()

    # 检查依赖项
    if not check_dependencies():
        logger.error("缺少必要的依赖项，退出")
        sys.exit(1)

    # 加载配置
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        logger.info(f"已加载配置: {args.config}")
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        sys.exit(1)

    # 设置模式
    config['general']['mode'] = args.mode
    logger.info(f"运行模式: {args.mode}")

    # 导入所需模块
    try:
        # 确保当前目录在导入路径中
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            
        if not args.ui_only:
            # 导入交易服务
            from trading_service import TradingService
            # 传递配置字典
            trading_service = TradingService(config)
        # 导入UI服务器
        from ui_server import UIServer
    except ImportError as e:
        logger.error(f"导入模块失败: {e}")
        logger.error("请确认已安装所有必要依赖")
        sys.exit(1)

    # 启动UI服务器
    ui_server = UIServer(config, trading_service)
    ui_thread = threading.Thread(target=ui_server.start)
    ui_thread.daemon = True
    ui_thread.start()
    logger.info("UI服务已启动")

    # 如果不是仅UI模式，启动交易服务
    if not args.ui_only:
        try:
            logger.info("启动交易服务...")
            trading_service.start()
            logger.info("交易服务已启动")
        except Exception as e:
            logger.error(f"启动交易服务失败: {e}")
            sys.exit(1)

    # 设置信号处理
    def signal_handler(sig, frame):
        logger.info("接收到关闭信号，正在关闭交易系统...")
        if not args.ui_only:
            trading_service.stop()
        ui_server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 保持主线程运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("用户中断，关闭交易系统...")
        if not args.ui_only:
            trading_service.stop()
        ui_server.stop()

if __name__ == "__main__":
    main()
