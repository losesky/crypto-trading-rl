"""
btc_rl 模块初始化文件
此模块包含加密货币强化学习训练代码和环境
"""
import sys
import os

# 确保src目录在路径中
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# 导入基本模块
try:
    # 使用相对导入确保模块正确加载
    from . import src
    from .src import policies
    from .src import env
    from .src import config
    
    # 显式导入TimeSeriesCNN类，并将其添加到当前模块的命名空间
    from .src.policies import TimeSeriesCNN
    
    # 确保可以从btc_rl直接访问TimeSeriesCNN
    __all__ = ['TimeSeriesCNN', 'policies', 'env', 'config']
    
    # 输出导入成功信息
    print(f"成功导入btc_rl模块及TimeSeriesCNN类")
except ImportError as e:
    print(f"警告：btc_rl模块导入错误: {e}")
    
    # 记录模块搜索路径，以便诊断
    print(f"当前sys.path: {sys.path}")
    print(f"尝试从以下位置导入: {os.path.dirname(os.path.abspath(__file__))}")
    
    # 如果无法导入，尝试直接从文件定义TimeSeriesCNN
    try:
        import torch as th
        import torch.nn as nn
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
        import gymnasium
        
        class TimeSeriesCNN(BaseFeaturesExtractor):
            """
            Input  : (batch, 9, 100)   # we permute the obs for channels_first
            Output : flat 256-dim vector fed to actor/critic MLPs
            """
            def __init__(self, observation_space: gymnasium.spaces.Box):
                super().__init__(observation_space, features_dim=256)
                self.cnn = nn.Sequential(
                    nn.Conv1d(in_channels=9, out_channels=32, kernel_size=8, stride=2),
                    nn.ReLU(),
                    nn.Conv1d(32, 64, kernel_size=5, stride=2),
                    nn.ReLU(),
                    nn.Conv1d(64, 64, kernel_size=3, stride=1),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                with th.no_grad():
                    dummy_input_for_cnn = th.as_tensor(observation_space.sample()[None]).float().permute(0, 2, 1)
                    n_flat = self.cnn(dummy_input_for_cnn).shape[1]
                self.fc = nn.Sequential(
                    nn.Linear(n_flat, 256),
                    nn.ReLU(),
                )
            
            def forward(self, obs: th.Tensor) -> th.Tensor:
                x = obs.permute(0, 2, 1)
                x = self.cnn(x)
                return self.fc(x)
                
        # 将类添加到policies模块
        if 'policies' in sys.modules and hasattr(sys.modules['btc_rl.src'], 'policies'):
            sys.modules['btc_rl.src.policies'].TimeSeriesCNN = TimeSeriesCNN
            print("已创建TimeSeriesCNN类并添加到policies模块")
    except Exception as inner_e:
        print(f"无法创建TimeSeriesCNN类: {inner_e}")