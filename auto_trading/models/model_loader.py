"""
模型加载模块
负责加载和管理训练好的强化学习模型
"""
import os
import glob
import logging
import yaml
import json
import time
import sys
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import tensorflow as tf
import stable_baselines3 as sb3
from stable_baselines3 import SAC, PPO, A2C
from stable_baselines3.common.utils import set_random_seed

# 模型文件已通过软链接放在auto_trading/models/trained_models/best_model目录下
# 但由于模型文件在序列化时包含了对btc_rl模块的引用，我们仍然需要将btc_rl模块添加到Python路径
# 添加btc_rl模块所在目录到Python路径 - 重要：需要用sys.path.insert保证优先级
btc_rl_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'btc_rl')
if os.path.exists(btc_rl_path):
    if btc_rl_path not in sys.path:
        sys.path.insert(0, btc_rl_path)  # 使用insert(0)确保最高优先级
        print(f"添加btc_rl模块路径: {btc_rl_path}")
    else:
        print(f"btc_rl模块路径已存在: {btc_rl_path}")
else:
    print(f"警告: btc_rl路径不存在: {btc_rl_path}")

# 确保btc_rl/src也在路径中，因为模型可能引用了自定义的网络结构或环境
btc_rl_src_path = os.path.join(btc_rl_path, 'src')
if os.path.exists(btc_rl_src_path):
    if btc_rl_src_path not in sys.path:
        sys.path.insert(0, btc_rl_src_path)  # 使用insert(0)确保最高优先级
        print(f"添加btc_rl/src模块路径: {btc_rl_src_path}")
    else:
        print(f"btc_rl/src模块路径已存在: {btc_rl_src_path}")
else:
    print(f"警告: btc_rl/src路径不存在: {btc_rl_src_path}")

# 尝试测试导入以验证路径设置是否生效
try:
    print(f"当前Python路径: {sys.path}")
    print("尝试导入btc_rl模块...")
    
    # 确保btc_rl根目录下有__init__.py
    init_file_path = os.path.join(btc_rl_path, '__init__.py')
    if not os.path.exists(init_file_path):
        print(f"警告: {init_file_path} 不存在，正在创建...")
        with open(init_file_path, 'w') as f:
            f.write('# 自动创建的__init__.py文件\n')
            f.write('import sys, os\n')
            f.write('src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")\n')
            f.write('if src_path not in sys.path:\n')
            f.write('    sys.path.insert(0, src_path)\n')
    
    # 确保btc_rl/src下有__init__.py
    src_init_file_path = os.path.join(btc_rl_src_path, '__init__.py')
    if not os.path.exists(src_init_file_path):
        print(f"警告: {src_init_file_path} 不存在，正在创建...")
        with open(src_init_file_path, 'w') as f:
            f.write('# 自动创建的__init__.py文件\n')
    
    # 尝试创建一个符号链接，确保导入btc_rl时能够找到正确的包
    import site
    site_packages = site.getsitepackages()[0]
    link_path = os.path.join(site_packages, 'btc_rl')
    
    if not os.path.exists(link_path):
        try:
            # 在Windows上使用目录联接
            if os.name == 'nt':
                import subprocess
                subprocess.run(['mklink', '/J', link_path, btc_rl_path], shell=True)
            # 在Linux/Mac上使用符号链接
            else:
                os.symlink(btc_rl_path, link_path)
            print(f"已创建btc_rl符号链接: {link_path} -> {btc_rl_path}")
        except Exception as e:
            print(f"创建符号链接失败: {e}")
    
    # 强制导入btc_rl模块
    import importlib
    import btc_rl
    importlib.reload(btc_rl)  # 强制重新加载
    print(f"成功导入btc_rl模块，路径: {btc_rl.__file__}")
    
    # 从policies.py文件复制TimeSeriesCNN实现
    # 如果直接导入btc_rl.src.policies失败，我们需要手动定义这个类
    src_policies_path = os.path.join(btc_rl_src_path, 'policies.py')
    if os.path.exists(src_policies_path):
        print("找到policies.py文件，确保TimeSeriesCNN可用...")
        # 导入gymnasium和torch，为TimeSeriesCNN类做准备
        import importlib.util
        try:
            import torch
            import gymnasium
            from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
            print("成功导入依赖模块")
        except ImportError:
            print("导入依赖模块失败，尝试使用自定义实现...")

except ImportError as e:
    print(f"警告：无法导入btc_rl模块: {e}")
    print("尝试创建兼容层...")
    
    # 动态创建btc_rl模块
    import types
    btc_rl_module = types.ModuleType('btc_rl')
    sys.modules['btc_rl'] = btc_rl_module
    print("已动态创建btc_rl模块")
    
    # 如果src路径存在，也创建btc_rl.src模块
    if os.path.exists(btc_rl_src_path):
        src_module = types.ModuleType('btc_rl.src')
        btc_rl_module.src = src_module
        sys.modules['btc_rl.src'] = src_module
        print("已动态创建btc_rl.src模块")
    
    # 创建policies模块和TimeSeriesCNN类的存根
    try:
        # 尝试导入必要的依赖
        import torch as th
        import torch.nn as nn
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
        import gymnasium
        
        # 创建policies模块
        policies_module = types.ModuleType('btc_rl.src.policies')
        sys.modules['btc_rl.src.policies'] = policies_module
        
        # 复制TimeSeriesCNN类实现
        class TimeSeriesCNN(BaseFeaturesExtractor):
            """
            Input  : (batch, 9, 100)   # we permute the obs for channels_first
            Output : flat 256-dim vector fed to actor/critic MLPs
            """
            
            def __init__(self, observation_space: gymnasium.spaces.Box):
                super().__init__(observation_space, features_dim=256)
                
                # The CNN expects (batch, num_features, seq_len)
                self.cnn = nn.Sequential(
                    nn.Conv1d(in_channels=9, out_channels=32, kernel_size=8, stride=2),
                    nn.ReLU(),
                    nn.Conv1d(32, 64, kernel_size=5, stride=2),
                    nn.ReLU(),
                    nn.Conv1d(64, 64, kernel_size=3, stride=1),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                
                # Infer conv output size
                with th.no_grad():
                    dummy_input_for_cnn = th.as_tensor(observation_space.sample()[None]).float().permute(0, 2, 1)
                    n_flat = self.cnn(dummy_input_for_cnn).shape[1]
                
                self.fc = nn.Sequential(
                    nn.Linear(n_flat, 256),
                    nn.ReLU(),
                )
            
            def forward(self, obs: th.Tensor) -> th.Tensor:
                # obs arrives as (batch, 100, 9) → transpose to (batch, 9, 100)
                x = obs.permute(0, 2, 1)
                x = self.cnn(x)
                return self.fc(x)
        
        # 将TimeSeriesCNN类添加到policies模块
        policies_module.TimeSeriesCNN = TimeSeriesCNN
        print("成功创建TimeSeriesCNN类")
    except ImportError as ie:
        print(f"警告: 无法创建TimeSeriesCNN类: {ie}")

# 检查GPU可用性并配置
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 目前仅使用第一个GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # 设置可见GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print(f"使用GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(f"GPU配置错误: {e}")
else:
    print("未检测到GPU，将使用CPU进行模型推理")

class ModelLoader:
    """
    模型加载类
    负责加载和管理训练好的强化学习模型
    """
    
    def __init__(self, config_path: str = None):
        """
        初始化模型加载器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认路径
        """
        self.logger = logging.getLogger('ModelLoader')
        self.models = {}  # 存储加载的模型
        self.model_configs = {}  # 存储模型配置
        
        # 设置随机种子以确保结果可重现
        set_random_seed(42)
        
        # 确定设备类型（GPU或CPU）
        tf_has_gpu = len(tf.config.list_physical_devices('GPU')) > 0
        # 对于TensorFlow使用'/GPU:0'，对于PyTorch使用'cuda'
        self.tf_device = '/GPU:0' if tf_has_gpu else '/CPU:0'
        self.torch_device = 'cuda' if tf_has_gpu else 'cpu'
        self.device = self.torch_device  # 默认使用PyTorch设备名称，因为stable-baselines3基于PyTorch
        self.logger.info(f"模型将在 {self.device.upper()} 上运行")
        
        # 如果未提供配置路径，则使用默认路径
        if config_path is None:
            # 使用绝对路径来定位配置文件
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(root_dir, "config", "model_config.yaml")
        
        # 配置TensorFlow性能优化
        self._configure_tensorflow_performance()
        
        self._load_config(config_path)
        
    def _configure_tensorflow_performance(self):
        """配置TensorFlow性能设置以优化推理速度"""
        if self.torch_device == 'cpu':
            # CPU性能优化
            tf.config.threading.set_inter_op_parallelism_threads(4)
            tf.config.threading.set_intra_op_parallelism_threads(4)
            self.logger.info("已配置TensorFlow CPU性能参数")
        else:
            # GPU性能优化
            # 使用混合精度，在保持准确性的同时提高性能
            try:
                # 根据TensorFlow版本使用不同的API调用
                tf_version = tf.__version__
                self.logger.info(f"TensorFlow版本: {tf_version}")
                
                # TF 2.3+支持混合精度策略
                if hasattr(tf.keras.mixed_precision, 'set_global_policy'):
                    tf.keras.mixed_precision.set_global_policy('mixed_float16')
                    self.logger.info("已使用set_global_policy启用混合精度推理")
                # 较旧的TF版本使用experimental命名空间
                elif hasattr(tf.keras.mixed_precision, 'experimental'):
                    tf.keras.mixed_precision.experimental.set_global_policy('mixed_float16')
                    self.logger.info("已使用experimental.set_global_policy启用混合精度推理")
                else:
                    self.logger.warning("当前TensorFlow版本不支持混合精度API，跳过此优化")
                
                # 设置TensorFlow内存增长选项
                for gpu in tf.config.list_physical_devices('GPU'):
                    tf.config.experimental.set_memory_growth(gpu, True)
                    self.logger.info(f"已为 {gpu.name} 启用内存增长选项")
                
                self.logger.info("已完成GPU性能优化配置")
            except Exception as e:
                self.logger.warning(f"GPU性能优化配置失败: {e}，这通常不会影响模型正常运行")
        
    def _load_config(self, config_path: str) -> None:
        """
        加载模型配置
        
        Args:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            model_config = config.get('model', {})
            
            # 获取项目根目录的绝对路径
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # 使用绝对路径替换相对路径
            relative_base_path = model_config.get('base_path', './models/trained_models')
            relative_best_model_path = model_config.get('best_model_path', './models/trained_models/best_model')
            
            # 如果是相对路径，则转换为绝对路径
            if not os.path.isabs(relative_base_path):
                self.base_path = os.path.normpath(os.path.join(project_root, relative_base_path))
            else:
                self.base_path = relative_base_path
                
            if not os.path.isabs(relative_best_model_path):
                self.best_model_path = os.path.normpath(os.path.join(project_root, relative_best_model_path))
            else:
                self.best_model_path = relative_best_model_path
                
            self.model_type = model_config.get('model_type', 'sac')
            self.load_latest = model_config.get('load_latest', True)
            
            # 高级加载设置
            self.advanced_loading = model_config.get('advanced_loading', {})
            self.use_custom_extractor = self.advanced_loading.get('use_custom_extractor', True)
            self.policy_type = self.advanced_loading.get('policy_type', 'MlpPolicy')
            self.mock_on_failure = self.advanced_loading.get('mock_on_failure', True)
            
            self.ensemble_config = model_config.get('ensemble', {})
            self.ensemble_enabled = self.ensemble_config.get('enabled', True)
            self.ensemble_method = self.ensemble_config.get('method', 'weighted_voting')
            self.ensemble_models = self.ensemble_config.get('models', [])
            
            self.logger.info("成功加载模型配置")
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            raise
    
    def _get_model_class(self, model_type: str):
        """
        获取模型类
        
        Args:
            model_type: 模型类型 ('sac', 'ppo', 'a2c')
            
        Returns:
            模型类
        """
        model_map = {
            'sac': SAC,
            'ppo': PPO,
            'a2c': A2C
        }
        
        if model_type.lower() not in model_map:
            raise ValueError(f"不支持的模型类型: {model_type}. 支持的类型: {list(model_map.keys())}")
        
        return model_map[model_type.lower()]
    
    def load_model(self, model_path: str, model_type: Optional[str] = None) -> Any:
        """
        加载单个模型
        
        Args:
            model_path: 模型文件路径
            model_type: 模型类型，如果为None则使用配置中的值
            
        Returns:
            Any: 加载的模型，如果加载失败则返回None
        """
        if not os.path.exists(model_path):
            self.logger.error(f"模型文件不存在: {model_path}")
            return None
            
        try:
            if model_type is None:
                model_type = self.model_type
                
            model_class = self._get_model_class(model_type)
            
            self.logger.info(f"加载模型: {model_path}，使用设备: {self.device.upper()}")
            
            # 首先尝试导入TimeSeriesCNN确保它可用
            try:
                import importlib
                # 确保btc_rl在路径中
                import sys
                btc_rl_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'btc_rl')
                if os.path.exists(btc_rl_path) and btc_rl_path not in sys.path:
                    sys.path.insert(0, btc_rl_path)
                    
                btc_rl_src_path = os.path.join(btc_rl_path, 'src')
                if os.path.exists(btc_rl_src_path) and btc_rl_src_path not in sys.path:
                    sys.path.insert(0, btc_rl_src_path)
                
                # 确保TimeSeriesCNN类已定义
                import btc_rl
                from btc_rl.src import policies
                
                # 检查TimeSeriesCNN是否存在于policies模块中
                if hasattr(policies, 'TimeSeriesCNN'):
                    self.logger.info("成功找到TimeSeriesCNN类")
                    TimeSeriesCNN = policies.TimeSeriesCNN
                else:
                    self.logger.warning("policies模块中未找到TimeSeriesCNN类")
                    
                    # 从policies.py文件手动加载TimeSeriesCNN类
                    policies_path = os.path.join(btc_rl_src_path, 'policies.py')
                    if os.path.exists(policies_path):
                        self.logger.info("正在从源文件加载TimeSeriesCNN类")
                        # 动态加载policies.py中的TimeSeriesCNN
                        import gymnasium
                        import torch as th
                        import torch.nn as nn
                        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
                        
                        module_vars = {}
                        with open(policies_path, 'r') as f:
                            policies_code = f.read()
                        
                        # 执行policies.py的代码，并将定义的类放入module_vars
                        exec(policies_code, module_vars)
                        
                        # 将TimeSeriesCNN添加到policies模块
                        if 'TimeSeriesCNN' in module_vars:
                            policies.TimeSeriesCNN = module_vars['TimeSeriesCNN']
                            sys.modules['btc_rl.src.policies'].TimeSeriesCNN = module_vars['TimeSeriesCNN']
                            self.logger.info("已成功将TimeSeriesCNN类添加到policies模块")
                        else:
                            self.logger.warning("无法从policies.py找到TimeSeriesCNN类")
            except Exception as e:
                self.logger.warning(f"预加载TimeSeriesCNN失败: {e}")
            
            # 准备自定义对象用于模型加载
            custom_objects = {}
            self.logger.info(f"正在尝试加载模型，使用设备: {self.device.upper()}")
            
            # 显式指定设备，如果模型支持的话
            if hasattr(model_class, 'load'):
                # 对于Stable-Baselines3模型，使用device参数
                try:
                    # 准备自定义对象
                    custom_objects = {}
                    
                    # 如果我们找到了TimeSeriesCNN类，添加到custom_objects
                    try:
                        from btc_rl.src.policies import TimeSeriesCNN
                        custom_objects = {
                            "policy_kwargs": {
                                "features_extractor_class": TimeSeriesCNN
                            }
                        }
                        self.logger.info("使用TimeSeriesCNN特征提取器加载模型")
                    except ImportError:
                        self.logger.warning("无法导入TimeSeriesCNN，尝试直接加载模型")
                    
                    # 加载模型，指定custom_objects和设备
                    model = model_class.load(model_path, device=self.torch_device, custom_objects=custom_objects)
                    self.logger.info(f"模型已加载到设备：{self.torch_device}")
                    return model
                except TypeError:
                    # 如果加载方法不接受device参数，尝试不带该参数的调用
                    self.logger.warning(f"模型加载没有device参数，将使用默认设备")
                    try:
                        model = model_class.load(model_path)
                        return model
                    except Exception as inner_e:
                        self.logger.error(f"使用默认参数加载模型失败: {inner_e}")
                except ModuleNotFoundError as module_error:
                    # 尝试解决模块导入问题
                    self.logger.warning(f"加载模型时模块错误：{module_error}，尝试修复...")
                    
                    # 尝试动态创建缺失的模块
                    missing_module_name = str(module_error).split("'")[1]
                    self.logger.info(f"尝试动态创建缺失的模块: {missing_module_name}")
                    
                    if missing_module_name == 'btc_rl':
                        # 确保btc_rl在sys.path中
                        import sys
                        btc_rl_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'btc_rl')
                        if os.path.exists(btc_rl_path) and btc_rl_path not in sys.path:
                            sys.path.insert(0, btc_rl_path)
                            self.logger.info(f"修复：插入btc_rl路径到sys.path的开始位置: {btc_rl_path}")
                        
                        btc_rl_src_path = os.path.join(btc_rl_path, 'src')
                        if os.path.exists(btc_rl_src_path) and btc_rl_src_path not in sys.path:
                            sys.path.insert(0, btc_rl_src_path)
                            self.logger.info(f"修复：插入btc_rl/src路径到sys.path的开始位置: {btc_rl_src_path}")
                        
                        try:
                            # 尝试直接导入真实的btc_rl模块和policies
                            import importlib
                            import btc_rl
                            importlib.reload(btc_rl)  # 强制重新加载
                            from btc_rl.src import policies
                            importlib.reload(policies)  # 强制重新加载
                            self.logger.info("成功导入真实的btc_rl模块")
                            
                            # 重新尝试加载模型
                            self.logger.info("重新尝试加载模型...")
                            try:
                                # 使用custom_objects参数加载模型，指定策略类型
                                from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
                                model = model_class.load(model_path, device=self.torch_device, 
                                    custom_objects={"policy_kwargs": {"features_extractor_class": policies.TimeSeriesCNN}})
                                self.logger.info("修复成功：模型已正确加载")
                                return model
                            except Exception as retry_error:
                                self.logger.warning(f"使用custom_objects加载模型失败: {retry_error}")
                                
                                # 尝试一种不同的方法：将策略类型更改为MlpPolicy
                                try:
                                    # 使用custom_objects参数加载模型，指定使用MlpPolicy
                                    from stable_baselines3.common.policies import ActorCriticPolicy
                                    model = model_class.load(model_path, device=self.torch_device,
                                        custom_objects={"policy_type": "MlpPolicy"})
                                    self.logger.info("使用MlpPolicy成功加载模型")
                                    return model
                                except Exception as policy_error:
                                    self.logger.error(f"更改策略类型加载失败: {policy_error}")
                        except ImportError as import_error:
                            self.logger.warning(f"无法导入真实的btc_rl模块: {import_error}，尝试加载TimeSeriesCNN类...")
                            
                            # 直接从文件加载TimeSeriesCNN类
                            try:
                                import types
                                import importlib.util
                                import gymnasium
                                import torch as th
                                import torch.nn as nn
                                from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
                                
                                # 创建必要的模块结构
                                btc_rl_module = types.ModuleType('btc_rl')
                                sys.modules['btc_rl'] = btc_rl_module
                                src_module = types.ModuleType('btc_rl.src')
                                btc_rl_module.src = src_module
                                sys.modules['btc_rl.src'] = src_module
                                policies_module = types.ModuleType('btc_rl.src.policies')
                                src_module.policies = policies_module
                                sys.modules['btc_rl.src.policies'] = policies_module
                                
                                # 从磁盘加载policies.py文件
                                policies_path = os.path.join(btc_rl_path, 'src', 'policies.py')
                                if os.path.exists(policies_path):
                                    self.logger.info(f"尝试从{policies_path}加载TimeSeriesCNN类")
                                    
                                    # 读取policies.py文件并执行，提取TimeSeriesCNN类
                                    with open(policies_path, 'r') as f:
                                        policies_code = f.read()
                                    
                                    # 创建一个新的命名空间来执行代码
                                    namespace = {
                                        'BaseFeaturesExtractor': BaseFeaturesExtractor,
                                        'gymnasium': gymnasium,
                                        'th': th,
                                        'nn': nn
                                    }
                                    
                                    # 执行policies.py的代码
                                    exec(policies_code, namespace)
                                    
                                    # 将TimeSeriesCNN类添加到模块
                                    if 'TimeSeriesCNN' in namespace:
                                        policies_module.TimeSeriesCNN = namespace['TimeSeriesCNN']
                                        self.logger.info("成功从文件加载并注册了TimeSeriesCNN类")
                                        
                                        # 重新尝试加载模型
                                        self.logger.info("使用加载的TimeSeriesCNN类重新尝试加载模型...")
                                        try:
                                            model = model_class.load(model_path, device=self.torch_device)
                                            self.logger.info("成功加载模型")
                                            return model
                                        except Exception as file_retry_error:
                                            self.logger.error(f"使用从文件加载的TimeSeriesCNN尝试加载模型失败: {file_retry_error}")
                                    else:
                                        self.logger.error("在policies.py文件中未找到TimeSeriesCNN类")
                                else:
                                    self.logger.error(f"policies.py文件不存在: {policies_path}")
                                from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
                                import gymnasium
                                
                                class TimeSeriesCNN(BaseFeaturesExtractor):
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
                                
                                policies_module.TimeSeriesCNN = TimeSeriesCNN
                                self.logger.info("成功创建TimeSeriesCNN类")
                                
                                # 重新尝试加载模型
                                self.logger.info("重新尝试加载模型...")
                                try:
                                    model = model_class.load(model_path, device=self.torch_device)
                                    self.logger.info("修复成功：模型已正确加载")
                                    return model
                                except Exception as retry_error:
                                    self.logger.error(f"修复后再次尝试加载模型失败: {retry_error}")
                            except ImportError as ie:
                                self.logger.warning(f"创建TimeSeriesCNN类失败: {ie}")
                    
                    # 如果所有修复尝试均失败，抛出原始错误
                    self.logger.error(f"所有修复尝试均失败，无法加载模型")
                    raise module_error
                except Exception as e:
                    self.logger.warning(f"使用指定设备加载模型失败: {e}，尝试使用默认设备")
                    try:
                        model = model_class.load(model_path)
                        return model
                    except Exception as inner_e:
                        self.logger.error(f"使用默认参数加载模型失败: {inner_e}")
            else:
                # 其他类型的模型加载
                try:
                    model = model_class.load(model_path)
                    return model
                except Exception as e:
                    self.logger.error(f"加载模型失败: {e}")
            
            return None
        except Exception as e:
            self.logger.error(f"加载模型失败 {model_path}: {e}")
            return None
    
    def load_best_model(self) -> Any:
        """
        加载最佳模型
        
        Returns:
            Any: 最佳模型
        
        Raises:
            FileNotFoundError: 如果模型文件不存在
            Exception: 如果模型加载失败
        """
        best_model_path = self.best_model_path
        
        if not os.path.exists(best_model_path):
            err_msg = f"最佳模型路径不存在: {best_model_path}"
            self.logger.error(err_msg)
            raise FileNotFoundError(err_msg)
            
        # 检查是否是目录，如果是，则查找目录下的模型文件
        if os.path.isdir(best_model_path):
            model_files = glob.glob(os.path.join(best_model_path, "*.zip"))
            if not model_files:
                err_msg = f"在最佳模型目录中未找到模型文件: {best_model_path}"
                self.logger.error(err_msg)
                raise FileNotFoundError(err_msg)
            
            # 选择最新的模型文件
            best_model_file = max(model_files, key=os.path.getmtime)
            self.logger.info(f"加载最佳模型: {os.path.basename(best_model_file)}")
        else:
            best_model_file = best_model_path
        
        result = self.load_model(best_model_file)
        if result is None:
            err_msg = f"加载模型失败: {best_model_file}"
            self.logger.error(err_msg)
            raise RuntimeError(err_msg)
            
        return result
    
    def load_latest_model(self) -> Any:
        """
        加载最新模型
        
        Returns:
            Any: 最新模型
        """
        # 直接使用最佳模型替代最新模型
        self.logger.info(f"使用最佳模型替代最新模型")
        return self.load_best_model()
    
    def load_ensemble_models(self) -> Dict[str, Any]:
        """
        加载用于集成的多个模型
        
        Returns:
            Dict[str, Any]: 模型名称到模型对象的映射
        """
        ensemble_models = {}
        
        try:
            # 直接加载最佳模型，不尝试加载其他模型
            self.logger.info("直接加载最佳模型，不尝试加载集成模型")
            best_model = self.load_best_model()
            ensemble_models["best_model"] = {
                "model": best_model,
                "weight": 1.0,
                "threshold": 0.5
            }
            return ensemble_models
                
        except Exception as e:
            self.logger.error(f"加载集成模型失败: {e}")
            raise
    
    def initialize_models(self) -> Dict[str, Any]:
        """
        初始化模型，根据配置加载单个模型或集成模型
        
        Returns:
            Dict[str, Any]: 加载的模型
        """
        start_time = time.time()
        self.logger.info(f"开始加载模型，使用设备：{self.device.upper()}")
        
        # 直接加载最佳模型，无论配置如何
        model = self.load_best_model()
        self.models["best"] = {
            "model": model,
            "weight": 1.0,
            "threshold": 0.5
        }
        self.logger.info("已加载最佳模型")
        
        # 预热模型，提高后续推理速度
        try:
            self._warm_up_models()
            elapsed = time.time() - start_time
            self.logger.info(f"模型加载和预热完成，耗时：{elapsed:.2f}秒")
        except Exception as e:
            self.logger.warning(f"模型预热失败：{e}")
                
        return self.models
        
    # 删除模拟模型创建方法，当找不到模型时直接抛出异常
    
    def _warm_up_models(self):
        """预热模型，进行一次推理以初始化内部状态和优化图"""
        # 根据错误信息，模型期望的输入形状是(99, 9)或(n_env, 99, 9)
        dummy_input = np.random.random((1, 99, 9))  # 使用正确的形状: (n_env, 99, 9)
        self.logger.info(f"使用形状为{dummy_input.shape}的数据预热模型")
        
        for model_id, model_data in self.models.items():
            model = model_data["model"]
                
            if hasattr(model, "predict"):
                try:
                    # 进行一次预测来预热模型
                    _ = model.predict(dummy_input)
                    self.logger.info(f"模型 {model_id} 预热成功")
                except Exception as e:
                    self.logger.warning(f"模型 {model_id} 预热失败: {e}")
                    
                    # 检查模型的observation_space属性，如果存在，获取正确的形状
                    try:
                        if hasattr(model, "observation_space") and hasattr(model.observation_space, "shape"):
                            obs_shape = model.observation_space.shape
                            self.logger.info(f"模型期望的观察空间形状: {obs_shape}")
                            # 创建符合模型期望形状的输入
                            if len(obs_shape) == 2:  # 如(99, 9)
                                correct_input = np.random.random((1,) + obs_shape)
                            else:  # 已经是批处理形状
                                correct_input = np.random.random(obs_shape)
                                
                            self.logger.info(f"尝试使用正确形状预热: {correct_input.shape}")
                            _ = model.predict(correct_input)
                            self.logger.info(f"使用正确形状预热成功")
                        else:
                            self.logger.warning(f"无法确定模型 {model_id} 的观察空间形状，跳过预热")
                    except Exception as inner_e:
                        self.logger.warning(f"使用正确形状预热仍然失败: {inner_e}")
                        self.logger.info("预热失败不影响模型正常使用，跳过预热")
    
    def get_model(self, model_name: Optional[str] = None) -> Any:
        """
        获取指定名称的模型
        
        Args:
            model_name: 模型名称，如果为None则返回第一个模型
            
        Returns:
            Any: 模型对象
        """
        if not self.models:
            self.initialize_models()
            
        if model_name and model_name in self.models:
            return self.models[model_name]["model"]
        
        # 如果没有指定模型名称或指定的模型不存在，返回第一个模型
        first_key = next(iter(self.models))
        return self.models[first_key]["model"]
    
    def get_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有加载的模型
        
        Returns:
            Dict[str, Dict[str, Any]]: 所有加载的模型
        """
        if not self.models:
            self.initialize_models()
            
        return self.models
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        获取模型信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            Dict[str, Any]: 模型信息
        """
        if not self.models or model_name not in self.models:
            return {}
            
        model_data = self.models[model_name]
        
        # 尝试从metrics文件中获取更详细的模型信息
        metrics_file = os.path.join(os.path.dirname(self.base_path), "metrics", f"{model_name}_metrics.json")
        metrics_info = {}
        
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    metrics_info = json.load(f)
            except Exception as e:
                self.logger.error(f"读取模型指标文件失败 {metrics_file}: {e}")
        
        return {
            "name": model_name,
            "weight": model_data.get("weight", 1.0),
            "threshold": model_data.get("threshold", 0.5),
            "metrics": metrics_info
        }
    
    def get_active_model(self) -> Any:
        """
        获取当前活跃的模型，通常是最佳模型
        
        Returns:
            Any: 活跃的模型对象
        """
        if not self.models:
            self.initialize_models()
        
        # 优先返回best模型
        if "best" in self.models:
            return self.models["best"]["model"]
        
        # 如果没有best模型，返回第一个可用的模型
        first_key = next(iter(self.models))
        return self.models[first_key]["model"]
