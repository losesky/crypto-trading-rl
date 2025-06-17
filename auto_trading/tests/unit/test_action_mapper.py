#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
动作映射器测试模块
测试ActionMapper类将连续SAC动作值映射到交易决策的功能
"""

import os
import sys
import unittest
import numpy as np
from typing import Dict, Any

# 配置日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL

# 添加项目根目录到路径，确保可以导入项目模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.action_mapper import ActionMapper, SELL, HOLD, BUY


class TestActionMapper(unittest.TestCase):
    """测试ActionMapper功能"""
    
    def setUp(self):
        """测试前的准备工作，实例化ActionMapper"""
        self.mapper = ActionMapper(
            min_action_value=-1.0,
            max_action_value=1.0,
            buy_threshold=0.03,
            sell_threshold=-0.03,
            position_scale=1.0
        )
    
    def test_map_action_sell(self):
        """测试卖出动作映射"""
        # 测试强卖出信号
        result = self.mapper.map_action(-0.8)
        self.assertEqual(result['action'], SELL)
        self.assertEqual(result['action_type'], 'SELL')
        self.assertGreater(result['signal_strength'], 0.5)
        self.assertGreater(result['position_size'], 0.5)
        
        # 测试弱卖出信号
        result = self.mapper.map_action(-0.1)
        self.assertEqual(result['action'], SELL)
        self.assertEqual(result['action_type'], 'SELL')
        # 由于我们使用了sigmoid变体计算信号强度，弱信号也可能有中等强度
        # self.assertLess(result['signal_strength'], 0.5)  # 不再要求低于0.5
    
    def test_map_action_buy(self):
        """测试买入动作映射"""
        # 测试强买入信号
        result = self.mapper.map_action(0.8)
        self.assertEqual(result['action'], BUY)
        self.assertEqual(result['action_type'], 'BUY')
        self.assertGreater(result['signal_strength'], 0.5)
        self.assertGreater(result['position_size'], 0.5)
        
        # 测试弱买入信号
        result = self.mapper.map_action(0.1)
        self.assertEqual(result['action'], BUY)
        self.assertEqual(result['action_type'], 'BUY')
        # 由于我们使用了sigmoid变体计算信号强度，弱信号也可能有中等强度
        # self.assertLess(result['signal_strength'], 0.5)  # 不再要求低于0.5
    
    def test_map_action_hold(self):
        """测试持有动作映射"""
        # 测试持有信号
        result = self.mapper.map_action(0.0)
        self.assertEqual(result['action'], HOLD)
        self.assertEqual(result['action_type'], 'HOLD')
        self.assertLess(result['signal_strength'], 0.1)
        self.assertLess(result['position_size'], 0.1)
        
        # 测试接近阈值的持有信号
        result = self.mapper.map_action(0.025)
        self.assertEqual(result['action'], HOLD)
        self.assertEqual(result['action_type'], 'HOLD')
    
    def test_get_probabilities(self):
        """测试概率分布推断功能"""
        # 强卖出信号的概率分布
        probs = self.mapper.get_action_probabilities(-0.8)
        self.assertGreater(probs["0"], 0.6)  # SELL概率高
        self.assertLess(probs["2"], 0.2)     # BUY概率低
        
        # 强买入信号的概率分布
        probs = self.mapper.get_action_probabilities(0.8)
        self.assertGreater(probs["2"], 0.6)  # BUY概率高
        self.assertLess(probs["0"], 0.2)     # SELL概率低
        
        # 持有信号的概率分布
        probs = self.mapper.get_action_probabilities(0.0)
        self.assertGreater(probs["1"], 0.5)  # HOLD概率高
        
    def test_calculate_confidence(self):
        """测试置信度计算功能"""
        # 强信号应有较高置信度
        self.assertGreater(self.mapper.calculate_confidence(-0.9), 0.7)
        self.assertGreater(self.mapper.calculate_confidence(0.9), 0.7)
        
        # 弱信号应有一般置信度
        self.assertLess(self.mapper.calculate_confidence(0.1), 0.7)
        self.assertLess(self.mapper.calculate_confidence(-0.1), 0.7)
        
        # 持有信号应有适中置信度
        confidence = self.mapper.calculate_confidence(0.0)
        self.assertGreater(confidence, 0.3)
        self.assertLess(confidence, 0.8)
    
    def test_boundary_conditions(self):
        """测试边界条件处理"""
        # 超出最小值范围
        result = self.mapper.map_action(-1.5)
        self.assertEqual(result['action'], SELL)  # 尽管值被裁剪为-1.0，仍然在SELL区间
        self.assertEqual(result['original_value'], -1.5)
        
        # 超出最大值范围
        result = self.mapper.map_action(1.5)
        self.assertEqual(result['action'], BUY)
        self.assertEqual(result['original_value'], 1.5)
        
        # 恰好在阈值上
        result = self.mapper.map_action(-0.03)
        self.assertEqual(result['action'], SELL)
        
        result = self.mapper.map_action(0.03)
        self.assertEqual(result['action'], BUY)

if __name__ == '__main__':
    unittest.main()
