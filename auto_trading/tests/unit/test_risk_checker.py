#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch, MagicMock
import yaml
import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到路径，确保可以导入所有模块
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.append(str(PROJECT_ROOT))

from risk.risk_checker import RiskChecker


class TestRiskChecker(unittest.TestCase):
    """风险检查模块测试类"""

    def setUp(self):
        """准备测试环境"""
        # 创建测试配置
        self.test_config = {
            'risk': {
                'max_position_size': 0.5,
                'max_leverage': 5,
                'max_drawdown': 0.1,
                'max_daily_trades': 20,
                'max_daily_loss': 0.05,
                'min_trade_interval_seconds': 300,
                'market_conditions': {
                    'max_volatility': 0.1,
                    'min_volume': 1000000,
                    'max_spread': 0.005
                },
                'circuit_breaker': {
                    'enabled': True,
                    'consecutive_losses': 5,
                    'max_loss_percent': 0.15,
                    'cooldown_minutes': 60
                }
            }
        }
        
        # 创建临时配置文件
        self.config_path = "/tmp/test_risk_config.yaml"
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
        
        # 模拟数据处理器
        self.mock_data_processor = MagicMock()
        
        # 模拟仓位管理器
        self.mock_position_manager = MagicMock()
        self.mock_position_manager.get_position_value.return_value = 5000
        self.mock_position_manager.get_account_balance.return_value = 10000
        
        # 创建风险检查器实例
        self.risk_checker = RiskChecker(
            config_path=self.config_path,
            data_processor=self.mock_data_processor,
            position_manager=self.mock_position_manager
        )
        
    def tearDown(self):
        """清理测试环境"""
        # 删除临时配置文件
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.risk_checker.max_position_size, 0.5)
        self.assertEqual(self.risk_checker.max_leverage, 5)
        self.assertEqual(self.risk_checker.max_drawdown, 0.1)
        self.assertEqual(self.risk_checker.max_daily_trades, 20)
        self.assertEqual(self.risk_checker.max_daily_loss, 0.05)
        self.assertEqual(self.risk_checker.min_trade_interval_seconds, 300)
        self.assertEqual(self.risk_checker.max_volatility, 0.1)
        self.assertEqual(self.risk_checker.min_volume, 1000000)
        self.assertEqual(self.risk_checker.max_spread, 0.005)
    
    def test_check_position_size(self):
        """测试检查仓位大小"""
        # 测试有效仓位
        result = self.risk_checker.check_position_size("BTCUSDT", 0.3)
        self.assertTrue(result["allowed"])
        
        # 测试超出最大仓位
        result = self.risk_checker.check_position_size("BTCUSDT", 0.6)
        self.assertFalse(result["allowed"])
    
    def test_check_leverage(self):
        """测试检查杠杆倍数"""
        # 测试有效杠杆
        result = self.risk_checker.check_leverage("BTCUSDT", 3)
        self.assertTrue(result["allowed"])
        
        # 测试超出最大杠杆
        result = self.risk_checker.check_leverage("BTCUSDT", 10)
        self.assertFalse(result["allowed"])
    
    def test_check_drawdown(self):
        """测试检查回撤"""
        # 模拟当前资产值和峰值
        self.risk_checker.peak_balance = 11000
        
        # 测试可接受回撤
        result = self.risk_checker.check_drawdown()
        self.assertTrue(result["allowed"])
        
        # 测试超出最大回撤
        self.risk_checker.peak_balance = 12000  # 当前回撤约为 16.7%
        result = self.risk_checker.check_drawdown()
        self.assertFalse(result["allowed"])
    
    def test_check_trade_frequency(self):
        """测试检查交易频率"""
        # 清空交易记录
        self.risk_checker.trades = []
        
        # 添加几笔交易
        today = datetime.now().date()
        self.risk_checker.trades = [
            {"time": datetime.combine(today, datetime.min.time()) + timedelta(hours=i), "symbol": "BTCUSDT"}
            for i in range(10)
        ]
        
        # 测试没有超出最大交易次数
        result = self.risk_checker.check_trade_frequency("BTCUSDT")
        self.assertTrue(result["allowed"])
        
        # 添加更多交易，超出限制
        self.risk_checker.trades.extend([
            {"time": datetime.combine(today, datetime.min.time()) + timedelta(hours=i+10), "symbol": "BTCUSDT"}
            for i in range(15)
        ])
        
        # 测试超出最大交易次数
        result = self.risk_checker.check_trade_frequency("BTCUSDT")
        self.assertFalse(result["allowed"])
    
    def test_check_market_volatility(self):
        """测试检查市场波动性"""
        # 模拟市场数据
        market_data = pd.DataFrame({
            'close': [35000, 35200, 35100, 35300, 35400],
            'high': [35200, 35400, 35300, 35500, 35600],
            'low': [34800, 35000, 34900, 35100, 35200],
            'volume': [2000000, 1800000, 1900000, 2100000, 2200000]
        })
        
        # 模拟数据处理器返回
        self.mock_data_processor.fetch_klines.return_value = market_data
        
        # 测试正常波动性
        result = self.risk_checker.check_market_volatility("BTCUSDT")
        self.assertTrue(result["allowed"])
        
        # 模拟高波动性市场
        high_volatility_data = pd.DataFrame({
            'close': [35000, 38000, 34000, 39000, 33000],
            'high': [35200, 38500, 34500, 39500, 33500],
            'low': [34800, 37500, 33500, 38500, 32500],
            'volume': [2000000, 1800000, 1900000, 2100000, 2200000]
        })
        self.mock_data_processor.fetch_klines.return_value = high_volatility_data
        
        # 测试高波动性
        result = self.risk_checker.check_market_volatility("BTCUSDT")
        self.assertFalse(result["allowed"])
    
    def test_check_trade_interval(self):
        """测试检查交易间隔"""
        # 清空交易记录
        self.risk_checker.trades = []
        
        # 添加一个最近的交易
        self.risk_checker.trades.append({
            "time": datetime.now() - timedelta(seconds=200),
            "symbol": "BTCUSDT"
        })
        
        # 测试交易间隔太短
        result = self.risk_checker.check_trade_interval("BTCUSDT")
        self.assertFalse(result["allowed"])
        
        # 模拟足够长的交易间隔
        self.risk_checker.trades = [{
            "time": datetime.now() - timedelta(seconds=400),
            "symbol": "BTCUSDT"
        }]
        
        # 测试足够的交易间隔
        result = self.risk_checker.check_trade_interval("BTCUSDT")
        self.assertTrue(result["allowed"])
    
    def test_check_all_risks(self):
        """测试检查所有风险"""
        # 设置模拟环境，使所有检查通过
        self.risk_checker.trades = []  # 清空交易记录
        self.risk_checker.peak_balance = 10500  # 设置峰值余额
        
        # 模拟市场数据
        market_data = pd.DataFrame({
            'close': [35000, 35200, 35100, 35300, 35400],
            'high': [35200, 35400, 35300, 35500, 35600],
            'low': [34800, 35000, 34900, 35100, 35200],
            'volume': [2000000, 1800000, 1900000, 2100000, 2200000]
        })
        self.mock_data_processor.fetch_klines.return_value = market_data
        
        # 测试所有风险检查
        result = self.risk_checker.check_all_risks("BTCUSDT", 0.3, 2)
        
        # 验证结果
        self.assertTrue(result["allowed"])
        self.assertEqual(len(result["checks"]), 6)  # 6个风险检查
        
        # 设置超出仓位限制的情况
        result = self.risk_checker.check_all_risks("BTCUSDT", 0.6, 2)
        
        # 验证结果
        self.assertFalse(result["allowed"])


if __name__ == '__main__':
    unittest.main()
