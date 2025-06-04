#!/usr/bin/env python3
"""
HTTP服务器包装器，用于抑制错误信息
"""

import os
import sys
import signal
import importlib

def signal_handler(sig, frame):
    """处理中断信号"""
    os._exit(0)

if __name__ == "__main__":
    # 注册信号处理
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # 避免破管异常
    
    # 允许一些必要的输出，但仍然过滤大部分异常
    def excepthook(exctype, value, traceback):
        if exctype != KeyboardInterrupt:
            print(f"HTTP服务器出现问题: {exctype.__name__}") 
    
    sys.excepthook = excepthook
    
    # 只过滤警告
    import warnings
    warnings.filterwarnings('ignore')
    
    # 保留标准错误输出
    
    try:
        # 导入并运行HTTP服务器
        module = importlib.import_module('btc_rl.src.http_server')
        # 调用run_server()函数启动HTTP服务器
        module.run_server()
    except Exception as e:
        # 打印错误信息，但继续执行
        print(f"💡 HTTP服务器启动失败: {str(e)}")
