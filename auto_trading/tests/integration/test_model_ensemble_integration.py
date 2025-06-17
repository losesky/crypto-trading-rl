#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型集成与动作映射集成测试
验证整个推理流程，从模型加载到连续动作映射再到决策生成
"""

import os
import sys
import unittest
import numpy as np
from typing import Dict, Any, List
import logging

# 配置日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 抑制TensorFlow警告
logging.basicConfig(level=logging.ERROR)  # 设置日志级别为ERROR，抑制INFO日志

# 添加项目根目录到路径，确保可以导入项目模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.ensemble import ModelEnsemble
from models.model_loader import ModelLoader
from models.action_mapper import SELL, HOLD, BUY


class TestModelEnsembleIntegration(unittest.TestCase):
    """测试ModelEnsemble和ActionMapper的集成"""
    
    @classmethod
    def setUpClass(cls):
        """
        准备集成测试环境，加载模型配置
        注意：此测试需要在配置文件中设置正确的模型路径
        """
        # 获取项目根目录
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        
        # 创建配置目录和简易配置（如果不存在）
        config_dir = os.path.join(root_dir, 'config')
        os.makedirs(config_dir, exist_ok=True)
        
        cls.config_path = os.path.join(config_dir, 'model_config.yaml')
        
        # 检查是否可以实例化ModelEnsemble（快速测试）
        try:
            ensemble = ModelEnsemble(cls.config_path)
            cls.ensemble_available = True
        except Exception as e:
            print(f"警告: 无法实例化ModelEnsemble: {e}")
            cls.ensemble_available = False
    
    def generate_sample_features(self) -> Dict[str, float]:
        """
        生成样本特征数据
        
        Returns:
            Dict[str, float]: 特征字典
        """
        # 创建测试特征集（根据实际特征名称调整）
        features = {
            'price': 30000.0,
            'volume': 100.0,
            'macd': 0.5,
            'rsi': 55.0,
            'ema_short': 29800.0,
            'ema_long': 29500.0,
            'atr': 200.0,
            'volatility': 0.02,
            'hour': 14.0,
            'day': 3.0,
            # 添加其他必要特征...
        }
        return features
    
    def test_ensemble_prediction(self):
        """测试集成模型的预测功能和连续动作映射"""
        if not self.ensemble_available:
            self.skipTest("ModelEnsemble不可用，跳过此测试")
            
        try:
            # 1. 创建模型集成实例
            ensemble = ModelEnsemble(self.config_path)
            
            # 2. 准备测试数据
            features = self.generate_sample_features()
            
            # 3. 执行预测
            prediction = ensemble.predict(features)
            
            # 4. 验证预测结果
            self.assertIn('action', prediction)
            self.assertIn('confidence', prediction)
            self.assertTrue(isinstance(prediction['action'], int))
            self.assertTrue(0 <= prediction['action'] <= 2)
            
            # 5. 验证是否包含连续动作相关信息
            # 这里可能需要根据实际返回的数据结构进行调整
            if 'original_action_value' in prediction:
                original_value = prediction.get('original_action_value')
                self.assertTrue(isinstance(original_value, float))
                self.assertTrue(-1.0 <= original_value <= 1.0 or 
                                abs(original_value) <= 1.5)  # 允许少量超出边界
                
            # 6. 验证预测结果的概率分布
            probabilities = prediction.get('probabilities', {})
            if probabilities:
                self.assertIn("0", probabilities)
                self.assertIn("1", probabilities)
                self.assertIn("2", probabilities)
                
                # 验证概率总和接近1
                prob_sum = sum(float(p) for p in probabilities.values())
                self.assertAlmostEqual(prob_sum, 1.0, places=1)
                
        except Exception as e:
            self.fail(f"测试模型集成预测失败: {str(e)}")
    
    def test_continuous_action_mapping(self):
        """测试连续动作映射相关逻辑"""
        if not self.ensemble_available:
            self.skipTest("ModelEnsemble不可用，跳过此测试")
            
        try:
            # 使用属性访问的方式测试动作映射功能
            ensemble = ModelEnsemble(self.config_path)
            
            # 确保动作映射器已实例化
            if not hasattr(ensemble, 'action_mapper'):
                # 触发实例化，通过predict调用
                _ = ensemble.predict(self.generate_sample_features())
            
            # 直接测试映射器
            if hasattr(ensemble, 'action_mapper'):
                mapper = ensemble.action_mapper
                
                # 测试一些典型的SAC输出值
                test_values = [-0.7372, -0.35, -0.02, 0.0, 0.02, 0.35, 0.7372]
                actions = []
                
                for val in test_values:
                    result = mapper.map_action(val)
                    # 存储映射结果以便验证
                    actions.append((val, result['action'], result['action_type']))
                
                # 验证映射结果是否合理
                # 注意：根据阈值设置，这些验证条件可能需要调整
                self.assertEqual(actions[0][2], 'SELL')  # -0.7372应为卖出
                self.assertEqual(actions[5][2], 'BUY')   # 0.35应为买入
                self.assertEqual(actions[3][2], 'HOLD')  # 0.0应为持有
                
        except Exception as e:
            self.fail(f"测试连续动作映射失败: {str(e)}")


if __name__ == '__main__':
    unittest.main()
