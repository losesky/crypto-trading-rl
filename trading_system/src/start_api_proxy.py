#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
API代理服务器启动脚本
"""
import os
import sys
import argparse

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="启动API代理服务器")
    parser.add_argument("--ws-url", help="WebSocket代理URL", default="http://localhost:8096")
    parser.add_argument("--trading-url", help="交易系统URL", default="http://localhost:8091")
    parser.add_argument("--port", type=int, help="HTTP端口", default=8090)
    
    args = parser.parse_args()
    
    try:
        # 动态导入，避免在不支持的环境中出错
        from api_proxy_server import get_instance
        
        # 获取并启动API代理实例
        proxy = get_instance(
            port=args.port, 
            trading_url=args.trading_url, 
            ws_url=args.ws_url
        )
        # 启动前先输出配置信息
        print(f"启动API代理服务器，配置如下:")
        print(f"- 端口: {args.port}")
        print(f"- WebSocket URL: {args.ws_url}")
        print(f"- 交易系统URL: {args.trading_url}")
        proxy.start()
        
        # 保持运行
        print(f"API代理服务器已启动在端口 {args.port}")
        print(f"WebSocket代理URL: {args.ws_url}")
        print(f"交易系统URL: {args.trading_url}")
        
        # 阻塞主线程
        import time
        while True:
            time.sleep(1)
            
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保api_proxy_server.py文件存在于当前目录")
        sys.exit(1)
    except KeyboardInterrupt:
        print("接收到关闭信号，退出中...")
    except Exception as e:
        print(f"启动过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()