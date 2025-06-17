#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch, MagicMock, mock_open
import yaml
import os
import sys
import json
from pathlib import Path

# 添加项目根目录到路径，确保可以导入所有模块
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.append(str(PROJECT_ROOT))

from models.model_loader import ModelLoader


class TestModelLoader(unittest.TestCase):
    """模型加载器测试类"""

    def setUp(self):
        """准备测试环境"""
        # 创建测试配置
        self.test_config = {
            'model': {
                'base_path': '/tmp/models',
                'best_model_path': '/tmp/models/best_model',
                'model_type': 'sac',
                'load_latest': True,
                'ensemble': {
                    'enabled': True,
                    'method': 'weighted_voting',
                    'models': [
                        {'name': 'model1', 'weight': 1.0, 'threshold': 0.5},
                        {'name': 'model2', 'weight': 0.8, 'threshold': 0.6}
                    ]
                }
            }
        }
        
        # 创建临时配置文件
        self.config_path = "/tmp/test_model_loader_config.yaml"
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
            
        # 模拟 stable_baselines3 模块
        self.mock_sac = MagicMock()
        self.mock_ppo = MagicMock()
        self.mock_a2c = MagicMock()
        
        # 设置模型类的模拟
        self.patches = [
            patch('models.model_loader.SAC', self.mock_sac),
            patch('models.model_loader.PPO', self.mock_ppo),
            patch('models.model_loader.A2C', self.mock_a2c),
            patch('os.path.exists', return_value=True),
            patch('glob.glob')
        ]
        
        for p in self.patches:
            p.start()
            
        # 设置glob.glob的返回值
        import glob
        glob.glob.return_value = ['/tmp/models/model1.zip', '/tmp/models/model2.zip']
        
        # 创建模型加载器实例
        self.model_loader = ModelLoader(config_path=self.config_path)
        
    def tearDown(self):
        """清理测试环境"""
        # 删除临时配置文件
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        
        # 停止所有模拟
        for p in self.patches:
            p.stop()
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.model_loader.base_path, '/tmp/models')
        self.assertEqual(self.model_loader.best_model_path, '/tmp/models/best_model')
        self.assertEqual(self.model_loader.model_type, 'sac')
        self.assertTrue(self.model_loader.load_latest)
        self.assertTrue(self.model_loader.ensemble_enabled)
        self.assertEqual(self.model_loader.ensemble_method, 'weighted_voting')
        self.assertEqual(len(self.model_loader.ensemble_models), 2)
    
    def test_get_model_class(self):
        """测试获取模型类"""
        # 测试支持的模型类型
        sac_class = self.model_loader._get_model_class('sac')
        ppo_class = self.model_loader._get_model_class('ppo')
        a2c_class = self.model_loader._get_model_class('a2c')
        
        self.assertEqual(sac_class, self.mock_sac)
        self.assertEqual(ppo_class, self.mock_ppo)
        self.assertEqual(a2c_class, self.mock_a2c)
        
        # 测试不支持的模型类型
        with self.assertRaises(ValueError):
            self.model_loader._get_model_class('unsupported_type')
    
    def test_load_model(self):
        """测试加载单个模型"""
        # 设置模型加载的返回值
        mock_model = MagicMock()
        self.mock_sac.load.return_value = mock_model
        
        # 加载模型
        model = self.model_loader.load_model('/tmp/models/test_model.zip')
        
        # 验证模型加载调用
        self.mock_sac.load.assert_called_once_with('/tmp/models/test_model.zip')
        
        # 验证返回的模型
        self.assertEqual(model, mock_model)
    
    @patch('os.path.isdir', return_value=True)
    @patch('os.path.getmtime')
    def test_load_best_model(self, mock_getmtime, mock_isdir):
        """测试加载最佳模型"""
        # 设置文件修改时间，使得model2.zip是最新的
        mock_getmtime.side_effect = lambda f: 200 if f == '/tmp/models/best_model/model2.zip' else 100
        
        # 更改glob.glob的返回值
        import glob
        glob.glob.side_effect = lambda path: [
            '/tmp/models/best_model/model1.zip', 
            '/tmp/models/best_model/model2.zip'
        ] if 'best_model' in path else ['/tmp/models/model1.zip', '/tmp/models/model2.zip']
        
        # 设置模型加载的返回值
        mock_model = MagicMock()
        self.mock_sac.load.return_value = mock_model
        
        # 加载最佳模型
        model = self.model_loader.load_best_model()
        
        # 验证模型加载调用
        self.mock_sac.load.assert_called_once_with('/tmp/models/best_model/model2.zip')
        
        # 验证返回的模型
        self.assertEqual(model, mock_model)
    
    @patch('os.path.getmtime')
    def test_load_latest_model(self, mock_getmtime):
        """测试加载最新模型"""
        # 设置文件修改时间，使得model2.zip是最新的
        mock_getmtime.side_effect = lambda f: 200 if f == '/tmp/models/model2.zip' else 100
        
        # 设置模型加载的返回值
        mock_model = MagicMock()
        self.mock_sac.load.return_value = mock_model
        
        # 加载最新模型
        model = self.model_loader.load_latest_model()
        
        # 验证模型加载调用
        self.mock_sac.load.assert_called_once_with('/tmp/models/model2.zip')
        
        # 验证返回的模型
        self.assertEqual(model, mock_model)
    
    @patch('os.path.exists')
    def test_load_ensemble_models(self, mock_exists):
        """测试加载集成模型"""
        # 模拟文件存在性
        mock_exists.side_effect = lambda path: 'model1.zip' in path or 'model2.zip' in path
        
        # 设置模型加载的返回值
        mock_model1 = MagicMock()
        mock_model2 = MagicMock()
        self.mock_sac.load.side_effect = [mock_model1, mock_model2]
        
        # 加载集成模型
        ensemble_models = self.model_loader.load_ensemble_models()
        
        # 验证返回的模型字典
        self.assertIn('model1', ensemble_models)
        self.assertIn('model2', ensemble_models)
        self.assertEqual(ensemble_models['model1']['model'], mock_model1)
        self.assertEqual(ensemble_models['model2']['model'], mock_model2)
        self.assertEqual(ensemble_models['model1']['weight'], 1.0)
        self.assertEqual(ensemble_models['model2']['weight'], 0.8)


if __name__ == '__main__':
    unittest.main()
