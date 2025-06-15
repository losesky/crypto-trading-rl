#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WebSocket代理服务器启动器 - 更健壮的启动方式
"""
import os
import sys
import time
import logging
import traceback

# 确保正确设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))  # 项目根目录
sys.path.insert(0, parent_dir)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(current_dir), 'logs', 'websocket_proxy.log'))
    ]
)
logger = logging.getLogger("WebSocketProxyStarter")

def start_proxy():
    """启动WebSocket代理服务"""
    try:
        logger.info("开始导入WebSocket代理模块...")
        from trading_system.src.websocket_proxy import get_instance
        
        logger.info("成功导入WebSocket代理模块")
        logger.info("创建WebSocket代理实例...")
        proxy = get_instance()
        
        logger.info("启动WebSocket代理...")
        proxy.start()
        
        logger.info("WebSocket代理已启动并运行中...")
        return proxy
    except ImportError as e:
        logger.error(f"导入WebSocket代理模块失败: {e}")
        logger.error(f"当前工作目录: {os.getcwd()}")
        logger.error(f"Python路径: {sys.path}")
        traceback.print_exc()
    except Exception as e:
        logger.error(f"启动WebSocket代理时出错: {e}")
        traceback.print_exc()
    
    return None

if __name__ == "__main__":
    proxy = start_proxy()
    
    if proxy:
        try:
            logger.info("WebSocket代理服务已启动，按Ctrl+C停止...")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("接收到停止信号，关闭服务...")
        except Exception as e:
            logger.error(f"运行时错误: {e}")
            traceback.print_exc()
        finally:
            try:
                logger.info("正在停止WebSocket代理...")
                proxy.stop()
                logger.info("WebSocket代理已停止")
            except Exception as e:
                logger.error(f"停止WebSocket代理时出错: {e}")
    else:
        logger.error("WebSocket代理启动失败")
        sys.exit(1)
