#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import yaml
import os
import sys
from pathlib import Path

# 添加项目根目录到路径，确保可以导入所有模块
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.append(str(PROJECT_ROOT))

from data.data_processor import DataProcessor


class TestDataProcessor(unittest.TestCase):
    """数据处理器测试类"""

    def setUp(self):
        """准备测试环境"""
        # 创建测试配置
        self.test_config = {
            'model': {
                'features': {
                    'lookback_window': 24,
                    'use_technical_indicators': True,
                    'use_market_sentiment': True
                }
            }
        }
        
        # 创建临时配置文件
        self.config_path = "/tmp/test_model_config.yaml"
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
        
        # 模拟BinanceAPI
        self.mock_binance_api_patcher = patch('data.data_processor.BinanceAPI')
        self.mock_binance_api = self.mock_binance_api_patcher.start()
        
        # 创建数据处理器实例
        self.data_processor = DataProcessor(config_path=self.config_path)
        
        # 准备测试数据
        self.test_klines = [
            # [开盘时间, 开盘价, 最高价, 最低价, 收盘价, 成交量, 收盘时间, 成交额, 成交笔数, 主动买入成交量, 主动买入成交额, ignore]
            [1625097600000, "35000", "35500", "34800", "35200", "100", 1625101199999, "3500000", 500, "60", "2100000", "0"],
            [1625101200000, "35200", "35700", "35100", "35600", "120", 1625104799999, "4200000", 600, "80", "2800000", "0"],
            [1625104800000, "35600", "36000", "35400", "35900", "150", 1625108399999, "5400000", 700, "90", "3200000", "0"],
        ]
        
    def tearDown(self):
        """清理测试环境"""
        # 删除临时配置文件
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        # 停止模拟
        self.mock_binance_api_patcher.stop()
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.data_processor.lookback_window, 24)
        self.assertTrue(self.data_processor.use_technical_indicators)
        self.assertTrue(self.data_processor.use_market_sentiment)
    
    def test_fetch_klines(self):
        """测试获取K线数据"""
        # 模拟BinanceAPI的get_klines方法
        mock_api_instance = self.mock_binance_api.return_value
        mock_api_instance.get_klines.return_value = self.test_klines
        
        # 调用fetch_klines方法
        df = self.data_processor.fetch_klines("BTCUSDT", "1h", limit=3)
        
        # 验证调用了正确的API方法
        mock_api_instance.get_klines.assert_called_once_with(
            symbol="BTCUSDT", 
            interval="1h", 
            limit=3,
            start_time=None,
            end_time=None
        )
        
        # 验证返回的DataFrame
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertEqual(df.index.name, 'open_time')
        self.assertTrue('close' in df.columns)
    
    def test_calculate_technical_indicators(self):
        """测试计算技术指标"""
        # 创建测试数据
        test_df = pd.DataFrame({
            'open': [35000, 35200, 35600, 35900, 36200, 36500, 36800, 37100],
            'high': [35500, 35700, 36000, 36300, 36600, 36900, 37200, 37500],
            'low': [34800, 35100, 35400, 35700, 36000, 36300, 36600, 36900],
            'close': [35200, 35600, 35900, 36200, 36500, 36800, 37100, 37400],
            'volume': [100, 120, 150, 180, 200, 220, 240, 260]
        })
        
        # 添加时间戳索引
        test_df.index = pd.date_range(start='2021-07-01', periods=8, freq='H')
        test_df.index.name = 'open_time'
        
        # 计算技术指标
        result_df = self.data_processor.calculate_technical_indicators(test_df)
        
        # 验证是否添加了技术指标
        self.assertTrue('sma_7' in result_df.columns)
        self.assertTrue('rsi_14' in result_df.columns)
        self.assertTrue('bb_upper' in result_df.columns)
        self.assertTrue('macd' in result_df.columns)
    
    def test_prepare_model_input(self):
        """测试准备模型输入数据"""
        # 创建测试数据
        test_df = pd.DataFrame({
            'close': np.random.random(30),
            'open': np.random.random(30),
            'high': np.random.random(30),
            'low': np.random.random(30),
            'volume': np.random.random(30) * 100
        })
        
        # 添加时间戳索引
        test_df.index = pd.date_range(start='2021-07-01', periods=30, freq='H')
        test_df.index.name = 'open_time'
        
        # 准备模型输入
        lookback = 5
        X = self.data_processor.prepare_model_input(test_df, lookback_window=lookback)
        
        # 验证输出形状
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual(X.shape[0], 30 - lookback)  # 样本数量
        self.assertEqual(X.shape[1], lookback)       # 回溯窗口大小
        self.assertEqual(X.shape[2], 5)              # 特征数量
    
    def test_validate_data(self):
        """测试数据验证"""
        # 创建有效的测试数据
        valid_df = pd.DataFrame({
            'close': np.random.random(30),
            'open': np.random.random(30),
            'high': np.random.random(30),
            'low': np.random.random(30),
            'volume': np.random.random(30) * 100
        })
        valid_df.index = pd.date_range(start='2021-07-01', periods=30, freq='H')
        valid_df.index.name = 'open_time'
        
        # 创建无效的测试数据 (行数不足)
        invalid_df = pd.DataFrame({
            'close': np.random.random(5),
            'open': np.random.random(5),
            'high': np.random.random(5),
            'low': np.random.random(5),
            'volume': np.random.random(5) * 100
        })
        invalid_df.index = pd.date_range(start='2021-07-01', periods=5, freq='H')
        invalid_df.index.name = 'open_time'
        
        # 暂时设置lookback_window
        original_lookback = self.data_processor.lookback_window
        self.data_processor.lookback_window = 10
        
        # 测试有效数据
        self.assertTrue(self.data_processor.validate_data(valid_df))
        
        # 测试无效数据
        self.assertFalse(self.data_processor.validate_data(invalid_df))
        
        # 恢复原始lookback_window
        self.data_processor.lookback_window = original_lookback


if __name__ == '__main__':
    unittest.main()
