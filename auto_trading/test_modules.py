#!/usr/bin/env python3
import traceback
import sys
import os

# 确保当前目录在模块搜索路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    print("尝试导入OrderManager...")
    from trading.order_manager import OrderManager
    
    print("尝试初始化OrderManager...")
    om = OrderManager()
    print('OrderManager初始化成功！')
except Exception as e:
    print(f"错误类型: {type(e)}")
    print(f"错误信息: {e}")
    traceback.print_exc()
