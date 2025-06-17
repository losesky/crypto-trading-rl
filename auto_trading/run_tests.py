#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
运行所有测试的脚本
"""
import os
import sys
import unittest
import logging
from pathlib import Path

# 配置GPU内存使用和TensorFlow日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # 动态分配GPU内存，避免占用全部GPU内存

# 添加项目根目录到路径
# 自动检测脚本运行位置，确保无论从哪个目录运行都能正确找到项目根目录
script_path = Path(__file__).resolve()
script_dir = script_path.parent

# 判断当前目录是否为auto_trading目录
if script_dir.name == "auto_trading":
    PROJECT_ROOT = script_dir.parent  # 项目根目录是auto_trading的父目录
else:
    PROJECT_ROOT = script_dir  # 假设当前已经在项目根目录

# 输出当前配置信息以便调试
print(f"脚本路径: {script_path}")
print(f"项目根目录: {PROJECT_ROOT}")
sys.path.append(str(PROJECT_ROOT))

# 添加btc_rl模块和auto_trading目录到Python路径
btc_rl_path = os.path.join(PROJECT_ROOT, "btc_rl")
btc_rl_src_path = os.path.join(btc_rl_path, "src")
auto_trading_path = os.path.join(PROJECT_ROOT, "auto_trading")

sys.path.insert(0, str(btc_rl_src_path))  # btc_rl/src
sys.path.insert(0, str(btc_rl_path))      # btc_rl 
sys.path.insert(0, str(auto_trading_path)) # auto_trading

if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    
    print("\n=============================================")
    print("开始运行自动交易系统的单元和集成测试")
    print("=============================================\n")
    
    # 自动发现并运行所有测试
    test_loader = unittest.TestLoader()
    
    # 定义测试目录（使用绝对路径）
    # 首先确定auto_trading目录的路径，然后在其下找tests目录
    auto_trading_dir = os.path.join(PROJECT_ROOT, 'auto_trading') if PROJECT_ROOT.name != 'auto_trading' else PROJECT_ROOT
    
    # 检查auto_trading路径是否存在
    if not os.path.exists(auto_trading_dir):
        print(f"错误：找不到auto_trading目录: {auto_trading_dir}")
        sys.exit(1)
    
    # 设置测试目录路径
    tests_dir = os.path.join(auto_trading_dir, 'tests')
    unit_tests_dir = os.path.join(tests_dir, 'unit')
    integration_tests_dir = os.path.join(tests_dir, 'integration')
    
    print(f"测试根目录: {tests_dir}")
    print(f"单元测试目录: {unit_tests_dir}")
    print(f"集成测试目录: {integration_tests_dir}")
    
    # 添加测试目录到路径的最前面，确保优先导入测试目录下的模块
    sys.path.insert(0, tests_dir)
    
    # 加载单元测试
    unit_suite = None
    if os.path.exists(unit_tests_dir) and os.path.isdir(unit_tests_dir):
        try:
            print(f"尝试加载单元测试，目录: {unit_tests_dir}")
            unit_suite = test_loader.discover(unit_tests_dir, pattern="test_*.py")
            print(f"成功加载单元测试")
        except ImportError as e:
            print(f"导入单元测试失败: {e}")
            unit_suite = unittest.TestSuite()  # 空测试套件
    else:
        print(f"单元测试目录不存在: {unit_tests_dir}")
        unit_suite = unittest.TestSuite()
    
    # 加载集成测试
    integration_suite = None
    if os.path.exists(integration_tests_dir) and os.path.isdir(integration_tests_dir):
        try:
            print(f"尝试加载集成测试，目录: {integration_tests_dir}")
            integration_suite = test_loader.discover(integration_tests_dir, pattern="test_*.py")
            print(f"成功加载集成测试")
        except ImportError as e:
            print(f"导入集成测试失败: {e}")
            integration_suite = unittest.TestSuite()  # 空测试套件
    else:
        print(f"集成测试目录不存在: {integration_tests_dir}")
        integration_suite = unittest.TestSuite()
    
    # 运行单元测试
    print("\n=============================================")
    print("运行单元测试...")
    print("=============================================")
    unit_result = unittest.TextTestRunner(verbosity=2).run(unit_suite)
    
    # 运行集成测试
    print("\n=============================================")
    print("运行集成测试...")
    print("=============================================")
    integration_result = unittest.TextTestRunner(verbosity=2).run(integration_suite)
    
    # 汇总测试结果
    total_tests = unit_result.testsRun + integration_result.testsRun
    total_errors = len(unit_result.errors) + len(integration_result.errors)
    total_failures = len(unit_result.failures) + len(integration_result.failures)
    
    # 输出汇总信息
    print("\n=============================================")
    print("测试执行摘要")
    print("=============================================")
    print(f"运行测试总数: {total_tests}")
    if total_errors == 0 and total_failures == 0:
        print("\033[92m✓ 所有测试通过!\033[0m")  # 绿色文本
    else:
        print(f"\033[91m✗ 测试失败: {total_failures} 个失败, {total_errors} 个错误\033[0m")  # 红色文本
    print("=============================================\n")
    
    # 如果有任何错误或失败，返回非零退出码
    sys.exit(1 if total_errors > 0 or total_failures > 0 else 0)
