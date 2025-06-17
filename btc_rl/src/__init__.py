"""
btc_rl.src 模块初始化文件
这个模块包含所有强化学习算法的具体实现
"""
import sys

# 导出常用模块
try:
    # 导入所有子模块
    from . import config
    from . import env
    from . import policies
    from . import preprocessing
    
    # 确保TimeSeriesCNN类可以通过btc_rl.src.policies直接访问
    # 检查policies模块是否有TimeSeriesCNN属性
    if hasattr(policies, 'TimeSeriesCNN'):
        from .policies import TimeSeriesCNN
        print(f"从.policies导入了TimeSeriesCNN类")
    else:
        print(f"警告: policies模块中没有TimeSeriesCNN类")
        # 尝试动态加载TimeSeriesCNN类
        import os
        import importlib.util
        
        # 确定policies.py的路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        policies_path = os.path.join(current_dir, 'policies.py')
        
        if os.path.exists(policies_path):
            print(f"找到policies.py文件，尝试加载TimeSeriesCNN类")
            
            # 尝试从文件中加载TimeSeriesCNN类
            try:
                # 导入必要的依赖
                import torch as th
                import torch.nn as nn
                import gymnasium
                from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
                
                # 导入policies模块
                spec = importlib.util.spec_from_file_location("policies", policies_path)
                policies_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(policies_module)
                
                # 获取TimeSeriesCNN类
                if hasattr(policies_module, 'TimeSeriesCNN'):
                    TimeSeriesCNN = policies_module.TimeSeriesCNN
                    print(f"成功从文件加载TimeSeriesCNN类")
                    
                    # 将类添加到当前模块
                    policies.TimeSeriesCNN = TimeSeriesCNN
                    globals()['TimeSeriesCNN'] = TimeSeriesCNN
                else:
                    print(f"警告: policies.py中没有找到TimeSeriesCNN类")
            except Exception as load_error:
                print(f"加载TimeSeriesCNN类失败: {load_error}")
        else:
            print(f"警告: policies.py文件不存在: {policies_path}")
    
    # 将TimeSeriesCNN类添加到导出列表
    __all__ = ['config', 'env', 'policies', 'preprocessing', 'TimeSeriesCNN']
    
    # 注册TimeSeriesCNN类到全局命名空间，确保pickle可以反序列化
    if 'TimeSeriesCNN' in globals() and 'btc_rl.src.policies' in sys.modules:
        sys.modules['btc_rl.src.policies'].TimeSeriesCNN = globals()['TimeSeriesCNN']
        print("已将TimeSeriesCNN类注册到btc_rl.src.policies模块")
except ImportError as e:
    print(f"btc_rl.src模块导入错误: {e}")
    pass