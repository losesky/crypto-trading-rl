#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试MoneyManager类的状态持久化和恢复功能
"""

import os
import sys
import unittest
import json
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from auto_trading.risk.money_manager import MoneyManager


class TestMoneyManager(unittest.TestCase):
    """测试资金管理器的功能"""
    
    def setUp(self):
        """每个测试前的准备工作"""
        # 使用测试专用的配置文件路径
        self.config_path = "./test_risk_config.yaml"
        
        # 创建测试配置
        with open(self.config_path, 'w') as f:
            f.write("""
risk:
  capital:
    total_capital_limit: 0.5
    initial_position_size: 0.02
    max_position_size: 0.1
  trade_limits:
    max_single_loss: 0.03
    max_daily_drawdown: 0.08
    min_trade_interval: 3600
    max_daily_trades: 24
    max_open_positions: 3
  position_management:
    max_holding_time: 72
    pyramid_scaling: true
    scale_in_steps: 3
    scale_out_steps: 2
            """)
        
        # 使用临时日志路径
        self.log_path = "./test_capital_status.json"
        
        # 如果测试日志存在，先删除
        if os.path.exists(self.log_path):
            os.remove(self.log_path)
        
    def tearDown(self):
        """每个测试后的清理工作"""
        # 删除测试配置文件
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        
        # 删除测试日志
        if os.path.exists(self.log_path):
            os.remove(self.log_path)
    
    def test_save_and_load_capital_status(self):
        """测试资金状态的保存和加载"""
        # 创建第一个实例并设置资金
        money_manager1 = MoneyManager(self.config_path)
        money_manager1.capital_log_path = self.log_path  # 使用测试日志路径
        
        # 设置资金
        total_capital = 15000.0
        available_capital = 7500.0
        money_manager1.set_capital(total_capital, available_capital)
        
        # 添加一些交易记录
        money_manager1.capital_status['daily_pnl'] = 500.0
        money_manager1.capital_status['total_pnl'] = 1200.0
        money_manager1._save_capital_status()
        
        # 确认日志文件已创建
        self.assertTrue(os.path.exists(self.log_path))
        
        # 读取文件内容，确认数据正确
        with open(self.log_path, 'r') as f:
            saved_data = json.load(f)
            self.assertEqual(saved_data['total_capital'], total_capital)
            self.assertEqual(saved_data['available_capital'], available_capital)
            self.assertEqual(saved_data['daily_pnl'], 500.0)
            self.assertEqual(saved_data['total_pnl'], 1200.0)
        
        # 创建第二个实例，测试加载功能
        money_manager2 = MoneyManager(self.config_path)
        money_manager2.capital_log_path = self.log_path  # 使用相同的测试日志路径
        money_manager2._load_capital_status()
        
        # 确认数据已正确加载
        self.assertEqual(money_manager2.capital_status['total_capital'], total_capital)
        self.assertEqual(money_manager2.capital_status['available_capital'], available_capital)
        self.assertEqual(money_manager2.capital_status['daily_pnl'], 500.0)
        self.assertEqual(money_manager2.capital_status['total_pnl'], 1200.0)
        
    def test_position_calculation(self):
        """测试仓位计算功能"""
        money_manager = MoneyManager(self.config_path)
        
        # 设置资金
        total_capital = 10000.0
        money_manager.set_capital(total_capital)
        
        # 计算仓位大小
        symbol = "BTCUSDT"
        confidence = 0.8  # 高置信度
        position_ratio, position_size = money_manager.calculate_position_size(symbol, confidence)
        
        # 验证结果是合理的
        self.assertGreater(position_ratio, 0)
        self.assertLessEqual(position_ratio, money_manager.capital_config['max_position_size'])
        self.assertGreater(position_size, 0)
        
        # 测试低置信度
        confidence = 0.3
        position_ratio_low, position_size_low = money_manager.calculate_position_size(symbol, confidence)
        
        # 验证低置信度仓位比高置信度小
        self.assertLess(position_ratio_low, position_ratio)
        self.assertLess(position_size_low, position_size)


if __name__ == '__main__':
    unittest.main()
