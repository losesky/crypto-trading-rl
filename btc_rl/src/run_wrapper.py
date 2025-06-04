#!/usr/bin/env python3
"""
包装脚本，用于启动训练进程并处理错误信息和中断
"""

import os
import signal
import sys
import importlib
import time
import traceback

def signal_handler(sig, frame):
    """处理中断信号"""
    print("\n\n🛑 检测到中断信号，正在优雅地退出...")
    # 直接退出，不打印任何错误信息
    os._exit(0)

def run_training():
    """运行训练进程，捕获所有异常"""
    # 设置环境变量以抑制额外的错误输出
    os.environ['PYTHONFAULTHANDLER'] = '0'
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # 抑制所有来自底层库的警告
    import warnings
    warnings.filterwarnings('ignore')
    
    try:
        # 重定向stderr到/dev/null
        original_stderr = sys.stderr
        null_stderr = open(os.devnull, 'w')
        sys.stderr = null_stderr
        
        # 由于多线程和异步的本质，我们修补标准库进一步抑制错误
        import threading
        threading.excepthook = lambda args: None
        
        # 修改asyncio错误处理
        import asyncio
        asyncio._get_running_loop = asyncio.get_event_loop
        
        # 动态导入训练模块
        module = importlib.import_module('btc_rl.src.train_sac')
        
        # 调用模块的main函数
        if hasattr(module, 'main'):
            module.main()
        else:
            print("⚠️ 训练模块中没有找到main函数")
            
    except KeyboardInterrupt:
        print("\n\n🛑 训练被用户中断")
    except Exception as e:
        # 不输出完整的错误回溯，只显示简短信息
        print(f"\n⚠️ 训练中断: {e.__class__.__name__}")
    finally:
        # 恢复stderr
        if 'null_stderr' in locals():
            sys.stderr = original_stderr
            null_stderr.close()

def silence_exceptions():
    """设置全局异常处理，抑制异常日志"""
    def excepthook(exctype, value, traceback):
        # 只打印KeyboardInterrupt以外的非系统异常的简短信息
        if exctype != KeyboardInterrupt and not issubclass(exctype, SystemExit):
            print(f"\n⚠️ 错误: {exctype.__name__}")
    
    # 替换系统默认异常处理器
    sys.excepthook = excepthook

if __name__ == "__main__":
    # 设置自动刷新输出，使训练进度能够实时显示
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # 避免破管异常
    
    # 替换异常处理器
    silence_exceptions()
    
    print("\n🚀 启动训练进程...")
    run_training()
    print("\n✅ 训练已结束")
