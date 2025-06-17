"""
模型集成模块
提供多种集成学习方法，结合多个模型的预测结果
"""
import os
import numpy as np
import pandas as pd
import logging
import yaml
from typing import Dict, List, Optional, Union, Any, Tuple
from collections import Counter
from datetime import datetime

# 导入常用的深度学习库
try:
    import torch
except ImportError:
    torch = None

# 导入动作映射模块
from .action_mapper import ActionMapper

class ModelEnsemble:
    """
    模型集成类
    提供不同的策略来组合多个模型的预测结果
    """
    
    def __init__(self, config_path: str = None):
        """
        初始化模型集成器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认路径
        """
        self.logger = logging.getLogger('ModelEnsemble')
        
        # 设置PyTorch设备
        try:
            import torch
            self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            self.torch_device = 'cpu'
        
        # 如果未提供配置路径，则使用绝对路径找到配置文件
        if config_path is None:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(root_dir, "config", "model_config.yaml")
            
        self._load_config(config_path)
        
    def _load_config(self, config_path: str) -> None:
        """
        加载配置
        
        Args:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            ensemble_config = config.get('model', {}).get('ensemble', {})
            self.enabled = ensemble_config.get('enabled', True)
            self.method = ensemble_config.get('method', 'weighted_voting')
            self.models_config = ensemble_config.get('models', [])
            
            self.logger.info(f"成功加载集成配置，方法: {self.method}")
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            raise
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        对输入特征进行预测
        
        Args:
            features: 特征字典，键为特征名，值为特征值
            
        Returns:
            Dict[str, Any]: 预测结果
        """
        # 确保导入关键库到局部作用域
        import numpy as np
        import torch
        
        self.logger.info(f"执行模型集成预测，特征数量：{len(features)}")
        self.logger.info(f"使用 PyTorch 设备: {self.torch_device}")
        
        # 记录PyTorch设备信息
        self.logger.info(f"使用 PyTorch 设备: {self.torch_device}")
        
        # 检查特征的数据类型，特别关注时间戳特征
        timestamp_features = []
        non_numeric_features = []
        for name, value in features.items():
            if isinstance(value, pd.Timestamp):
                timestamp_features.append(name)
            elif not isinstance(value, (int, float, np.number)):
                non_numeric_features.append(f"{name}:{type(value)}")
                
        if timestamp_features:
            self.logger.debug(f"检测到时间戳特征: {', '.join(timestamp_features[:5])}" + 
                             (f"...等{len(timestamp_features)}个" if len(timestamp_features) > 5 else ""))
            
        if non_numeric_features:
            self.logger.debug(f"检测到非数值特征: {', '.join(non_numeric_features[:5])}" + 
                             (f"...等{len(non_numeric_features)}个" if len(non_numeric_features) > 5 else ""))
        
        try:
            # 导入和实例化 ModelLoader - 确保路径配置正确
            from . import model_loader
            # 传递None作为配置路径，让ModelLoader使用绝对路径查找配置文件
            model_loader_instance = model_loader.ModelLoader(config_path=None)
            # 加载当前活跃的模型
            active_model = model_loader_instance.get_active_model()
            
            # 将特征转换为模型可接受的输入格式
            # 模型期望的形状是(1, 99, 9)，我们需要将特征重塑为此形式
            
            # 设置并记录PyTorch设备
            import torch
            self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.logger.info(f"使用 PyTorch 设备: {self.torch_device}")
            
            # 这里我们需要使用更严格的特征转换逻辑，以确保与训练时的格式一致
            # 特征排序很重要，因为神经网络对输入张量中值的位置非常敏感
            sorted_feature_names = sorted(features.keys())
            self.logger.info(f"按名称排序特征以确保一致性，共有 {len(sorted_feature_names)} 个特征")
            
            # 确保numpy已被导入
            import numpy as np
            
            # 首先创建一个预定义大小的零矩阵
            model_input = np.zeros((1, 99, 9))
            
            # 记录主要特征组
            feature_groups = {
                "价格特征": ["open", "high", "low", "close", "volume"],
                "技术指标": ["sma_7", "sma_25", "rsi_14", "macd", "bb_upper", "bb_middle", "bb_lower"],
                "震荡指标": ["momentum", "price_velocity_5", "price_acceleration_5"],
                "波动率特征": ["atr_14", "return_volatility_12", "return_volatility_24"]
            }
            
            # 为了调试记录找到的特征组
            found_features = {group: [] for group in feature_groups}
            for name in sorted_feature_names:
                for group, patterns in feature_groups.items():
                    for pattern in patterns:
                        if pattern in name.lower():
                            found_features[group].append(name)
                            break
            
            # 打印找到的特征组
            for group, names in found_features.items():
                if names:
                    self.logger.info(f"找到{group}特征: {', '.join(names[:5])}{' 等' if len(names) > 5 else ''}")
            
            # 转换所有特征为浮点数数组，处理不同类型
            feature_values = []
            for name in sorted_feature_names:
                value = features[name]
                try:
                    if isinstance(value, pd.Timestamp):
                        # 将时间戳转换为Unix时间戳
                        feature_values.append(value.timestamp())
                    elif isinstance(value, (str, bool)):
                        # 字符串和布尔转为0或1
                        feature_values.append(float(1 if value else 0))
                    else:
                        # 其他尝试转为浮点数
                        feature_values.append(float(value))
                except (TypeError, ValueError) as e:
                    self.logger.warning(f"特征 '{name}' 值 '{value}' 转换失败: {e}")
                    feature_values.append(0.0)  # 转换失败用0填充
            
            # 确保我们有正确数量的特征值
            num_features = len(feature_values)
            self.logger.info(f"总计准备 {num_features} 个特征值用于模型输入")
            
            # 填充模型输入数组
            # 我们使用标准方式布局: 行优先，每行9个特征，最多99行
            rows_to_fill = min(99, (num_features + 8) // 9)
            
            # 安全填充特征
            for i in range(min(rows_to_fill * 9, num_features)):
                row = i // 9
                col = i % 9
                if row < 99 and col < 9:
                    model_input[0, row, col] = feature_values[i]
            
            # 确保numpy已被导入
            import numpy as np
            
            # 获取当前模型输入形状和统计信息
            self.logger.info(f"转换后的输入形状: {model_input.shape}")
            
            # 添加额外的调试信息
            non_zero_count = np.count_nonzero(model_input)
            min_val = np.min(model_input)
            max_val = np.max(model_input)
            self.logger.debug(f"模型输入统计: 非零元素数量={non_zero_count}, 最小值={min_val}, 最大值={max_val}")
            
            if active_model is None:
                self.logger.error("没有活跃的模型可用于预测")
                # 返回保守的预测，避免交易
                return {
                    "action": 1,  # HOLD
                    "confidence": 0.0,
                    "is_confident": False,
                    "probabilities": {
                        "0": 0.0,  # SELL
                        "1": 1.0,  # HOLD (保守地持有)
                        "2": 0.0   # BUY
                    },
                    "timestamp": datetime.now().isoformat(),
                    "features_used": list(features.keys())
                }
                
            # 检查并规范化模型输入数据
            # 数值太大可能导致模型预测不稳定，尝试归一化数据
            model_input_normalized = model_input.copy()
            
            # 确保numpy已被导入
            import numpy as np
            
            # 检测异常值并记录
            mean_val = np.mean(model_input)
            std_val = np.std(model_input)
            self.logger.info(f"模型输入原始统计：均值={mean_val:.4f}, 标准差={std_val:.4f}, 最大值={np.max(model_input):.4f}, 最小值={np.min(model_input):.4f}")
            
            # 如果数据方差过大，进行标准化处理
            if std_val > 10000:  # 标准差过大，可能需要标准化
                self.logger.info("检测到数据方差过大，应用标准化处理")
                # 对每个特征通道分别标准化
                for col in range(model_input.shape[2]):
                    channel_data = model_input[0, :, col]
                    if np.std(channel_data) > 0:
                        # Z-score标准化，但保留0值
                        non_zero_mask = channel_data != 0
                        if np.any(non_zero_mask):
                            channel_mean = np.mean(channel_data[non_zero_mask])
                            channel_std = np.std(channel_data[non_zero_mask])
                            if channel_std > 0:
                                model_input_normalized[0, :, col][non_zero_mask] = (channel_data[non_zero_mask] - channel_mean) / channel_std
            
            self.logger.info(f"处理后的模型输入统计：均值={np.mean(model_input_normalized):.4f}, 标准差={np.std(model_input_normalized):.4f}, 最大值={np.max(model_input_normalized):.4f}, 最小值={np.min(model_input_normalized):.4f}")
            
            # 使用模型进行预测
            # 确保numpy已导入到局部作用域
            import numpy as np
            
            # 添加异常捕获以便更好地诊断预测失败
            try:
                # 在调用模型前确保输入形状和数据类型正确
                self.logger.info(f"确保模型输入形状和数据类型正确: {model_input_normalized.shape}")
                
                # 明确记录输入形状和统计信息
                self.logger.info(f"模型输入张量统计：均值={np.mean(model_input_normalized):.4f}, " +
                               f"方差={np.var(model_input_normalized):.4f}, " + 
                               f"最大值={np.max(model_input_normalized):.4f}, " + 
                               f"最小值={np.min(model_input_normalized):.4f}")
                
                # 确保active_model在正确的设备上
                import torch
                if hasattr(active_model, 'device'):
                    model_device = active_model.device
                    self.logger.info(f"模型当前设备：{model_device}")
                    if str(model_device) != self.torch_device:
                        self.logger.warning(f"模型设备({model_device})与系统设备({self.torch_device})不匹配")
                        # 尝试将模型移动到系统设备
                        try:
                            active_model.to(self.torch_device)
                            self.logger.info(f"已将模型移动到设备：{self.torch_device}")
                        except Exception as e:
                            self.logger.warning(f"移动模型到新设备失败：{e}")
                
                # 判断是否为SAC模型
                is_sac = False
                if hasattr(active_model, '_policy'):
                    is_sac = 'SACPolicy' in str(active_model._policy.__class__)
                elif hasattr(active_model, 'policy'):
                    is_sac = 'SACPolicy' in str(active_model.policy.__class__)
                
                # 记录模型类型
                if is_sac:
                    self.logger.info("检测到SAC策略模型，将应用连续到离散的动作映射")
                    
                # 执行预测
                action, _states = active_model.predict(model_input_normalized, deterministic=False)
                
                # 确保numpy已被导入
                import numpy as np
                
                # 处理潜在的异常动作值（如-1或小数值）
                if isinstance(action, np.ndarray):
                    action_value = action.item() if action.size == 1 else action[0].item()
                else:
                    action_value = action
                    
                # 记录原始连续动作值
                original_action_value = action_value
                # 保存为类属性，以便外部访问
                self.last_original_action_value = original_action_value
                
                # 使用高级动作映射器处理连续动作值
                action_detail = {}
                try:
                    # 实例化动作映射器（如果尚未创建）- 使用与train_sac.py一致的映射参数
                    if not hasattr(self, 'action_mapper'):
                        # SAC模型的输出范围是[-1, 1]（由tanh激活函数决定）
                        # 阈值设置接近0，以更好地匹配训练时的连续到离散映射逻辑
                        self.action_mapper = ActionMapper(
                            min_action_value=-1.0,
                            max_action_value=1.0,
                            buy_threshold=0.03,   # >0.03为买入 - 降低阈值使其更敏感
                            sell_threshold=-0.03, # <-0.03为卖出 - 降低阈值使其更敏感
                            position_scale=1.0    # 控制仓位大小缩放
                        )
                    
                    # 映射原始动作值到交易决策和仓位信息
                    action_detail = self.action_mapper.map_action(original_action_value)
                    
                    # 提取映射后的离散动作ID
                    action_value = action_detail['action']
                    
                    # 从映射器中获取动作概率分布和置信度
                    probabilities = self.action_mapper.get_action_probabilities(original_action_value)
                    confidence = self.action_mapper.calculate_confidence(original_action_value)
                    
                    # 保存详细信息，供后续使用
                    self.last_action_detail = action_detail
                    self.last_action_probabilities = probabilities
                    self.last_action_confidence = confidence
                    
                    self.logger.info(
                        f"使用ActionMapper: 原始值={original_action_value:.6f} -> "
                        f"动作={action_value}({action_detail['action_type']}), "
                        f"信号强度={action_detail['signal_strength']:.4f}, "
                        f"建议仓位={action_detail['position_size']:.4f}, "
                        f"置信度={confidence:.4f}"
                    )
                    
                except Exception as mapper_err:
                    self.logger.warning(f"使用ActionMapper失败: {mapper_err}，回退到基本映射")
                    
                    # 回退到基本映射逻辑
                    if isinstance(original_action_value, float):
                        # 使用简单的阈值映射
                        if original_action_value < -0.03:  # 使用与ActionMapper一致的阈值
                            action_value = 0  # SELL
                        elif original_action_value > 0.03:  # 使用与ActionMapper一致的阈值
                            action_value = 2  # BUY
                        else:
                            action_value = 1  # HOLD
                        self.logger.info(f"连续动作值 {original_action_value} 映射到离散动作: {action_value}")
                
                # 最终确认动作值是有效的整数ID
                if not isinstance(action_value, int) or action_value not in [0, 1, 2]:
                    self.logger.warning(f"映射后仍然无效的动作值: {action_value}，将强制设为HOLD(1)")
                    action_value = 1  # 设置为HOLD作为安全默认值
                    action_value = 1  # 默认HOLD作为保守策略
                
                self.logger.info(f"模型预测成功，原始动作: {action}, 处理后动作: {action_value}")
                action = action_value  # 使用处理后的动作值
            except Exception as predict_err:
                self.logger.error(f"模型预测失败: {predict_err}")
                self.logger.info("使用默认动作1(HOLD)替代")
                action = 1  # 默认动作为持有（最保守）
            
            # 获取动作概率分布
            probabilities = {}
            confidence = 0.0
            
            # 尝试多种方法获取动作概率
            probability_extraction_methods = [
                "sac_policy_probs",     # 专为SAC模型添加的方法
                "policy_distribution",  # 标准SB3方法
                "action_probabilities", # 可能的替代方法
                "policy_eval",          # 尝试评估策略
                "compute_actions"       # 直接计算所有动作的概率
            ]
            
            for method in probability_extraction_methods:
                self.logger.debug(f"尝试使用 {method} 方法获取动作概率")
                
                try:
                    # 专门为SAC模型添加的概率提取方法
                    if method == "sac_policy_probs" and hasattr(active_model, 'policy'):
                        try:
                            import torch
                            
                            # 确保使用正确的设备
                            device = torch.device(self.torch_device)
                            self.logger.debug(f"SAC概率提取使用设备: {device}")
                            
                            # 确保策略在正确的设备上
                            if hasattr(active_model.policy, 'to'):
                                active_model.policy.to(device)
                            
                            # 对于SAC，我们知道动作是连续的，但可以通过计算Q值来获取离散动作的"概率"
                            # 映射离散动作到连续空间中的代表值
                            discrete_to_continuous = {
                                0: -1.0,  # SELL对应负值
                                1: 0.0,   # HOLD对应0
                                2: 1.0    # BUY对应正值
                            }
                            
                            with torch.no_grad():
                                # 将观测转换为张量并放到正确的设备上
                                obs_tensor = torch.FloatTensor(model_input_normalized).to(device)
                                
                                # 获取每个离散动作的Q值
                                q_values = {}
                                for discrete_action, cont_action in discrete_to_continuous.items():
                                    # 创建连续动作张量
                                    action_tensor = torch.FloatTensor([[cont_action]]).to(device)
                                    
                                    # 获取Q值（需要策略有q_net属性）
                                    if hasattr(active_model.policy, 'q1_target'):
                                        q1 = active_model.policy.q1_target(obs_tensor, action_tensor)
                                        q_values[discrete_action] = q1.item()
                                    elif hasattr(active_model.policy, 'q1'):
                                        q1 = active_model.policy.q1(obs_tensor, action_tensor)
                                        q_values[discrete_action] = q1.item()
                                    elif hasattr(active_model.policy, 'critic') and hasattr(active_model.policy.critic, 'q1'):
                                        q1 = active_model.policy.critic.q1(obs_tensor, action_tensor)
                                        q_values[discrete_action] = q1.item()
                                
                                # 如果成功获取了所有Q值，转换为概率
                                if len(q_values) == 3:
                                    # 确保numpy已导入到局部作用域
                                    import numpy as np
                                    
                                    # 使用softmax转换Q值为概率
                                    q_array = np.array([q_values[0], q_values[1], q_values[2]])
                                    q_exp = np.exp(q_array - np.max(q_array))  # 防止数值溢出
                                    probs = q_exp / q_exp.sum()
                                    
                                    self.logger.info(f"SAC Q值: {q_values}, 转换为概率: {probs}")
                                    
                                    # 填充概率字典
                                    for i, prob in enumerate(probs):
                                        probabilities[str(i)] = float(prob)
                                    
                                    # 计算信心度
                                    sorted_probs = sorted(probs)
                                    if len(sorted_probs) > 1:
                                        confidence = sorted_probs[-1] - sorted_probs[-2]
                                    else:
                                        confidence = sorted_probs[-1]
                                    break  # 成功获取概率，跳出循环
                        except Exception as e:
                            self.logger.warning(f"SAC概率提取方法失败: {e}")
                            pass  # 继续尝试其他方法
                    
                    elif method == "policy_distribution" and hasattr(active_model, 'policy') and hasattr(active_model.policy, 'get_distribution'):
                        # 标准SB3方法
                        try:
                            # 为防止设备不一致，确保输入和模型在同一设备上
                            import torch
                            device = torch.device(self.torch_device)
                            
                            if hasattr(active_model.policy, 'to'):
                                active_model.policy.to(device)
                                
                            # 使用with torch.no_grad()包装关键操作
                            with torch.no_grad():
                                # 尝试将输入放到同一设备上
                                if hasattr(model_input_normalized, 'to') and hasattr(model_input_normalized, 'device'):
                                    if model_input_normalized.device != device:
                                        model_input_normalized = model_input_normalized.to(device)
                                
                                # 获取分布
                                dist = active_model.policy.get_distribution(model_input_normalized)
                                
                                if hasattr(dist, 'distribution') and hasattr(dist.distribution, 'probs'):
                                    # 确保保持在同一设备上
                                    probs = dist.distribution.probs
                                    if probs.device != device:
                                        probs = probs.to(device)
                                        
                                    # 转到CPU以进行numpy转换
                                    action_probs = probs.cpu().numpy()[0]
                                    self.logger.info(f"通过policy.get_distribution获取的动作概率: {action_probs}")
                                    
                                    # 创建概率字典
                                    for i, prob in enumerate(action_probs):
                                        probabilities[str(i)] = float(prob)
                                    
                                    # 计算信心度 - 最高概率与次高概率之差
                                    sorted_probs = sorted(action_probs)
                                    if len(sorted_probs) > 1:
                                        confidence = sorted_probs[-1] - sorted_probs[-2]
                                        self.logger.info(f"信心度计算成功: {confidence}")
                                    else:
                                        confidence = sorted_probs[-1]
                                    break  # 成功获取概率，跳出循环
                                else:
                                    # 尝试其他可能的分布格式
                                    if hasattr(dist, 'probs'):
                                        probs = dist.probs
                                        if probs.device != device:
                                            probs = probs.to(device)
                                        action_probs = probs.cpu().numpy()[0]
                                        
                                        self.logger.info(f"通过dist.probs获取的动作概率: {action_probs}")
                                        
                                        # 创建概率字典
                                        for i, prob in enumerate(action_probs):
                                            probabilities[str(i)] = float(prob)
                                        
                                        # 计算信心度
                                        sorted_probs = sorted(action_probs)
                                        if len(sorted_probs) > 1:
                                            confidence = sorted_probs[-1] - sorted_probs[-2]
                                        else:
                                            confidence = sorted_probs[-1]
                                        break  # 成功获取概率，跳出循环
                                    else:
                                        self.logger.warning("分布对象没有预期的probs属性，尝试下一个方法")
                        except Exception as e:
                            self.logger.warning(f"通过policy.get_distribution获取概率失败: {e}")
                    
                    elif method == "action_probabilities" and hasattr(active_model, 'action_probability'):
                        # 一些模型可能有action_probability方法
                        all_actions = np.array([0, 1, 2])  # 所有可能的动作
                        action_probs = active_model.action_probability(model_input, actions=all_actions)
                        self.logger.info(f"通过action_probability获取的动作概率: {action_probs}")
                        
                        # 创建概率字典
                        for i, prob in enumerate(action_probs):
                            probabilities[str(i)] = float(prob)
                        
                        # 计算信心度
                        sorted_probs = sorted(action_probs)
                        if len(sorted_probs) > 1:
                            confidence = sorted_probs[-1] - sorted_probs[-2]
                        else:
                            confidence = sorted_probs[-1]
                        break  # 成功获取概率，跳出循环
                        
                    elif method == "policy_eval" and hasattr(active_model, 'policy') and hasattr(active_model.policy, 'evaluate_actions'):
                        # 尝试为每个动作评估值
                        try:
                            import torch
                            # 确保使用正确的设备
                            device = torch.device(self.torch_device if torch.cuda.is_available() else "cpu")
                            self.logger.debug(f"策略评估使用设备: {device}")
                            
                            # 先确保策略在正确的设备上
                            if hasattr(active_model.policy, 'to'):
                                try:
                                    active_model.policy.to(device)
                                    self.logger.debug(f"已将策略移动到设备: {device}")
                                except Exception as device_err:
                                    self.logger.warning(f"无法将策略移动到设备 {device}: {device_err}")
                            
                            # 将输入和操作都放在同一设备上
                            tensor_input = torch.tensor(model_input_normalized, dtype=torch.float32, device=device)
                            values = []
                            
                            # 确保所有张量操作在同一设备上
                            with torch.no_grad():  # 避免梯度计算
                                for act in [0, 1, 2]:
                                    # 确保动作张量也在同一设备上
                                    action_tensor = torch.tensor([[act]], dtype=torch.long, device=device)
                                    try:
                                        value, _ = active_model.policy.evaluate_actions(tensor_input, action_tensor)
                                        # 确保值张量在同一设备上
                                        if value.device != device:
                                            value = value.to(device)
                                        values.append(float(value.detach().cpu().item()))
                                    except Exception as act_err:
                                        self.logger.warning(f"评估动作{act}时出错: {act_err}")
                                        values.append(0.0)  # 出错时使用默认值
                            
                            # 通过softmax转换为概率
                            if values:
                                values = np.array(values)
                                # 应用softmax函数
                                exp_values = np.exp(values - np.max(values))  # 减去最大值以避免数值溢出
                                probs = exp_values / exp_values.sum()
                                
                                self.logger.info(f"通过policy.evaluate_actions获取的动作概率: {probs}")
                                
                                # 创建概率字典
                                for i, prob in enumerate(probs):
                                    probabilities[str(i)] = float(prob)
                                    
                                # 计算信心度
                                sorted_probs = sorted(probs)
                                if len(sorted_probs) > 1:
                                    confidence = sorted_probs[-1] - sorted_probs[-2]
                                else:
                                    confidence = sorted_probs[-1]
                                break  # 成功获取概率，跳出循环
                        except Exception as e:
                            self.logger.warning(f"策略评估方法整体失败: {e}")
                        
                    elif method == "compute_actions" and hasattr(active_model, 'policy'):
                        # 如果上述方法都失败，尝试直接从模型获取所有动作的logits
                        self.logger.debug("尝试直接从模型策略获取logits或概率")
                        try:
                            import torch
                            # 确保使用正确的设备
                            device = torch.device(self.torch_device if torch.cuda.is_available() else "cpu")
                            self.logger.debug(f"计算动作使用设备: {device}")
                            
                            # 将输入放在正确的设备上
                            tensor_input = torch.tensor(model_input_normalized, dtype=torch.float32, device=device)
                            
                            if hasattr(active_model.policy, 'forward'):
                                # 尝试将模型策略移到相同的设备上
                                try:
                                    active_model.policy.to(device)
                                    self.logger.debug(f"策略已移至设备: {device}")
                                except Exception as device_err:
                                    self.logger.warning(f"无法将策略移动到设备{device}: {device_err}")
                                
                                # 确保网络模块也在同一设备上
                                if hasattr(active_model.policy, 'features_extractor') and hasattr(active_model.policy.features_extractor, 'to'):
                                    try:
                                        active_model.policy.features_extractor.to(device)
                                        self.logger.debug(f"特征提取器已移至设备: {device}")
                                    except Exception as fe_err:
                                        self.logger.warning(f"无法将特征提取器移动到设备{device}: {fe_err}")
                                
                                # 获取策略前向传播结果
                                try:
                                    with torch.no_grad():  # 避免梯度计算
                                        # 确保所有子模块都知道我们使用的设备
                                        if hasattr(active_model.policy, 'cnn') and hasattr(active_model.policy.cnn, 'to'):
                                            active_model.policy.cnn.to(device)
                                        if hasattr(active_model.policy, 'fc') and hasattr(active_model.policy.fc, 'to'):
                                            active_model.policy.fc.to(device)
                                        if hasattr(active_model.policy, 'mlp_extractor') and hasattr(active_model.policy.mlp_extractor, 'to'):
                                            active_model.policy.mlp_extractor.to(device)
                                        
                                        policy_output = active_model.policy.forward(tensor_input)
                                    
                                    if isinstance(policy_output, tuple) and len(policy_output) > 0:
                                        # 确保输出张量也在同一设备上
                                        logits = policy_output[0]
                                        if logits.device != device:
                                            logits = logits.to(device)
                                        
                                        # 现在将输出转移到CPU进行后续处理
                                        logits_cpu = logits.detach().cpu().numpy()
                                        # 确保numpy已导入到局部作用域
                                        import numpy as np
                                        
                                        # 计算softmax获取概率
                                        exp_logits = np.exp(logits_cpu - np.max(logits_cpu))
                                        probs = exp_logits / exp_logits.sum()
                                        
                                        self.logger.info(f"通过policy.forward获取的动作概率: {probs}")
                                        
                                        # 创建概率字典
                                        for i, prob in enumerate(probs[0]):
                                            probabilities[str(i)] = float(prob)
                                            
                                        # 计算信心度
                                        probs_list = probs[0].tolist()
                                        sorted_probs = sorted(probs_list)
                                        if len(sorted_probs) > 1:
                                            confidence = sorted_probs[-1] - sorted_probs[-2]
                                        else:
                                            confidence = sorted_probs[-1]
                                        break  # 成功获取概率，跳出循环
                                except Exception as forward_err:
                                    self.logger.warning(f"策略前向传播失败: {forward_err}")
                        except Exception as e:
                            self.logger.warning(f"尝试从策略直接获取概率失败: {e}")
                
                except Exception as e:
                    self.logger.warning(f"使用 {method} 方法获取动作概率时出错: {e}")
                    
            # 如果所有方法都失败，或者动作值无效，使用基于动作的简单方法
            if not probabilities or action not in [0, 1, 2]:
                if not probabilities:
                    self.logger.info("SAC模型不提供原生概率分布，将使用ActionMapper生成概率分布")
                
                # 如果先前已经在动作映射阶段计算过概率和置信度，则直接使用保存的结果
                if hasattr(self, 'last_action_probabilities') and hasattr(self, 'last_action_confidence'):
                    probabilities = self.last_action_probabilities
                    confidence = self.last_action_confidence
                    self.logger.debug(f"使用先前计算的概率分布: {probabilities}, 置信度: {confidence:.4f}")
                    self.logger.info(f"模型动作 {action} 的推断概率: SELL={probabilities['0']:.4f}, HOLD={probabilities['1']:.4f}, BUY={probabilities['2']:.4f}")
                
                # 否则，使用ActionMapper重新生成概率分布
                elif hasattr(self, 'action_mapper'):
                    orig_val = getattr(self, 'last_original_action_value', None)
                    if orig_val is not None and isinstance(orig_val, float):
                        try:
                            # 使用ActionMapper生成概率分布
                            probabilities = self.action_mapper.get_action_probabilities(orig_val)
                            
                            # 计算置信度
                            confidence = self.action_mapper.calculate_confidence(orig_val)
                            
                            self.logger.info(f"根据原始连续动作值 {orig_val} 生成概率分布: {probabilities}, 置信度: {confidence:.4f}")
                        except Exception as e:
                            self.logger.warning(f"使用ActionMapper生成概率失败: {e}，回退到基本方法")
                            # 基于离散动作ID的默认概率
                            if action == 0:  # SELL
                                probabilities = {"0": 0.7, "1": 0.2, "2": 0.1}
                            elif action == 2:  # BUY
                                probabilities = {"0": 0.1, "1": 0.2, "2": 0.7}
                            else:  # HOLD
                                probabilities = {"0": 0.15, "1": 0.7, "2": 0.15}
                            confidence = 0.5
                    else:
                        # 基于离散动作ID的默认概率
                        if action == 0:  # SELL
                            probabilities = {"0": 0.7, "1": 0.2, "2": 0.1}
                        elif action == 2:  # BUY
                            probabilities = {"0": 0.1, "1": 0.2, "2": 0.7}
                        else:  # HOLD
                            probabilities = {"0": 0.15, "1": 0.7, "2": 0.15}
                        confidence = 0.5
                else:
                    # ActionMapper不可用，基于离散动作ID的默认概率
                    if action == 0:  # SELL
                        probabilities = {"0": 0.7, "1": 0.2, "2": 0.1}
                    elif action == 2:  # BUY
                        probabilities = {"0": 0.1, "1": 0.2, "2": 0.7}
                    else:  # HOLD
                        probabilities = {"0": 0.15, "1": 0.7, "2": 0.15}
                    confidence = 0.5
                    
                    self.logger.info(f"使用动作ID {action} 生成默认概率分布: {probabilities}, 置信度: {confidence}")
                    
                # 传统的备用机制 - 当上述方法都失败时使用
                if not probabilities and isinstance(action_value, float):
                    # 使用简单的传统方法生成概率分布
                    action_range = 1.0  # 假设SAC输出范围通常在[-1, 1]
                    orig_val = getattr(self, 'last_original_action_value', None)
                    
                    if orig_val is not None and isinstance(orig_val, float):
                        # 基于连续值的简单线性映射
                        if orig_val < -0.05:  # 卖出区域
                            # 负值越大（绝对值越大），卖出概率越高
                            sell_strength = min(0.9, 0.6 + abs(orig_val) * 0.3)
                            probabilities = {
                                "0": sell_strength,
                                "1": (1 - sell_strength) * 0.8,
                                "2": (1 - sell_strength) * 0.2
                            }
                            confidence = max(0.4, min(0.7, sell_strength - 0.2))
                            
                        elif orig_val > 0.05:  # 买入区域
                            # 正值越大，买入概率越高
                            buy_strength = min(0.9, 0.6 + orig_val * 0.3)
                            probabilities = {
                                "0": (1 - buy_strength) * 0.2,
                                "1": (1 - buy_strength) * 0.8,
                                "2": buy_strength
                            }
                            confidence = max(0.4, min(0.7, buy_strength - 0.2))
                            
                        else:  # 持有区域
                            # 越接近0，持有概率越高
                            hold_strength = max(0.6, 0.8 - abs(orig_val) * 2)
                            remaining = 1 - hold_strength
                            
                            # 根据值的符号略微偏向买入或卖出
                            if orig_val > 0:
                                probabilities = {
                                    "0": remaining * 0.3,
                                    "1": hold_strength,
                                    "2": remaining * 0.7
                                }
                            else:
                                # 偏向卖出的概率分布
                                probabilities = {
                                    "0": remaining * 0.7,
                                    "1": hold_strength,
                                    "2": remaining * 0.3
                                }
                            confidence = max(0.4, min(0.65, hold_strength - 0.1))
                            
                            self.logger.info(
                                f"使用传统方法基于原始值 {orig_val} 计算概率: {probabilities}, "
                                f"置信度: {confidence:.2f}"
                            )
                                
                    # 下面是回退方案的基础概率                
                    else:
                        # 基于动作ID推断的简单方法
                        if action == 0:  # SELL
                            probabilities = {"0": 0.7, "1": 0.2, "2": 0.1}
                        elif action == 2:  # BUY
                            probabilities = {"0": 0.1, "1": 0.2, "2": 0.7}
                        else:  # HOLD
                            probabilities = {"0": 0.2, "1": 0.6, "2": 0.2}
                        
                        # 设置中等信心度
                        confidence = 0.4
            
            # 验证获取的动作概率是否有效
            
            # 对从ActionMapper获取的概率不重复记录日志，减少冗余信息
            if not hasattr(self, 'last_action_probabilities') or probabilities != self.last_action_probabilities:
                self.logger.info(f"模型动作 {action} 的推断概率: SELL={probabilities.get('0', 0):.4f}, HOLD={probabilities.get('1', 0):.4f}, BUY={probabilities.get('2', 0):.4f}")
            
            # 确保所有需要的键都存在
            action_keys = ["0", "1", "2"]
            for action_key in action_keys:
                if action_key not in probabilities:
                    self.logger.debug(f"概率分布缺少键 {action_key}，添加默认值")
                    probabilities[action_key] = 0.1  # 设置一个基础概率
            
            # 验证概率总和
            total_prob = sum(float(p) for p in probabilities.values())
            if abs(total_prob - 1.0) > 0.01:  # 允许轻微的浮点误差
                self.logger.warning(f"动作概率之和不为1: {total_prob}，尝试归一化")
                # 归一化概率
                normalizer = total_prob if total_prob > 0 else 1.0
                for k in probabilities:
                    probabilities[k] = float(probabilities[k]) / normalizer
                
            # 确保概率不会小于最小阈值
            min_prob_threshold = 0.05
            for k in probabilities:
                if float(probabilities[k]) < min_prob_threshold:
                    self.logger.debug(f"将键 {k} 的概率从 {probabilities[k]} 提升到最小阈值 {min_prob_threshold}")
                    probabilities[k] = min_prob_threshold
            
            # 再次归一化
            total_prob = sum(float(p) for p in probabilities.values())
            if abs(total_prob - 1.0) > 0.01:
                for k in probabilities:
                    probabilities[k] = float(probabilities[k]) / total_prob
                    
            # 确保概率和信心度在有效范围内
            for k in probabilities:
                probabilities[k] = max(0.0, min(1.0, float(probabilities[k])))
            confidence = max(0.0, min(1.0, float(confidence)))
                
            self.logger.info(f"最终动作概率: {probabilities}, 信心度: {confidence}")
            
            # 返回预测结果
            return {
                "action": int(action),
                "confidence": float(confidence),
                "is_confident": confidence >= 0.65,  # 使用阈值判断是否有足够的信心
                "probabilities": probabilities,
                "timestamp": datetime.now().isoformat(),
                "features_used": list(features.keys())
            }
        
        except Exception as e:
            self.logger.error(f"预测失败: {e}")
            # 返回默认预测结果
            return {
                "action": 1,  # 默认为HOLD
                "confidence": 0.0,
                "is_confident": False,
                "error": str(e)
            }
    
    def majority_voting(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        多数投票集成方法
        
        Args:
            predictions: 多个模型的预测结果列表
            
        Returns:
            Dict[str, Any]: 集成后的预测结果
        """
        if not predictions:
            return {
                "action": 1,  # 默认动作 (HOLD)
                "confidence": 0.0,
                "is_confident": False
            }
        
        # 提取所有模型的动作
        actions = [p.get("action", 1) for p in predictions]
        
        # 计算每个动作的票数
        action_counts = Counter(actions)
        
        # 获取票数最多的动作
        most_common_action, count = action_counts.most_common(1)[0]
        
        # 计算置信度 (票数占比)
        confidence = count / len(actions)
        
        # 获取所有投给最终动作的预测的平均置信度
        confident_preds = [p for p in predictions if p.get("action", 1) == most_common_action]
        avg_confidence = sum(p.get("confidence", 0) for p in confident_preds) / len(confident_preds) if confident_preds else 0
        
        # 最终置信度 = 投票一致性 * 平均置信度
        final_confidence = confidence * avg_confidence
        
        return {
            "action": int(most_common_action),
            "confidence": float(final_confidence),
            "is_confident": final_confidence > 0.6,  # 可调整阈值
            "vote_ratio": confidence,
            "avg_prediction_confidence": avg_confidence
        }
    
    def weighted_voting(self, predictions: List[Dict[str, Any]], 
                     weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        加权投票集成方法
        
        Args:
            predictions: 多个模型的预测结果列表
            weights: 对应的权重列表，如果为None则使用配置中的权重
            
        Returns:
            Dict[str, Any]: 集成后的预测结果
        """
        if not predictions:
            return {
                "action": 1,  # 默认动作 (HOLD)
                "confidence": 0.0,
                "is_confident": False
            }
        
        # 使用默认权重或配置权重
        if weights is None:
            weights = [1.0] * len(predictions)
            # 如果有配置的模型权重，则使用
            if len(self.models_config) == len(predictions):
                weights = [model.get("weight", 1.0) for model in self.models_config]
        
        # 确保权重长度与预测结果长度相同
        weights = weights[:len(predictions)]
        while len(weights) < len(predictions):
            weights.append(1.0)
        
        # 计算每个动作的权重和
        action_weights = {}
        total_weight = sum(weights)
        
        for i, pred in enumerate(predictions):
            action = pred.get("action", 1)
            if action not in action_weights:
                action_weights[action] = 0
            
            # 在这里，我们可以选择乘以模型的置信度
            pred_confidence = pred.get("confidence", 0.5)
            action_weights[action] += weights[i] * pred_confidence
        
        # 获取权重最高的动作
        max_weight = -float('inf')
        final_action = 1  # 默认动作
        
        for action, weight in action_weights.items():
            if weight > max_weight:
                max_weight = weight
                final_action = action
        
        # 计算置信度 (占比 * 模型平均置信度)
        action_preds = [p for i, p in enumerate(predictions) if p.get("action", 1) == final_action]
        action_weights_sum = sum(weights[i] for i, p in enumerate(predictions) if p.get("action", 1) == final_action)
        
        if action_preds and action_weights_sum > 0:
            avg_confidence = sum(p.get("confidence", 0) for p in action_preds) / len(action_preds)
            weight_ratio = action_weights_sum / total_weight
            confidence = weight_ratio * avg_confidence
        else:
            avg_confidence = 0
            weight_ratio = 0
            confidence = 0
        
        return {
            "action": int(final_action),
            "confidence": float(confidence),
            "is_confident": confidence > 0.6,  # 可调整阈值
            "weight_ratio": weight_ratio,
            "avg_prediction_confidence": avg_confidence
        }
    
    def average_predictions(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        平均预测值集成方法
        
        Args:
            predictions: 多个模型的预测结果列表
            
        Returns:
            Dict[str, Any]: 集成后的预测结果
        """
        if not predictions:
            return {
                "action": 1,  # 默认动作 (HOLD)
                "confidence": 0.0,
                "is_confident": False
            }
        
        # 按动作分组并计算平均值
        action_values = {}
        action_counts = {}
        
        for pred in predictions:
            action = pred.get("action", 1)
            value = pred.get("action_value", 0)
            
            if action not in action_values:
                action_values[action] = 0
                action_counts[action] = 0
            
            action_values[action] += value
            action_counts[action] += 1
        
        # 计算每个动作的平均值
        avg_values = {}
        for action, total_value in action_values.items():
            avg_values[action] = total_value / action_counts[action]
        
        # 找出平均值最高的动作
        max_avg = -float('inf')
        final_action = 1  # 默认动作
        
        for action, avg_value in avg_values.items():
            if avg_value > max_avg:
                max_avg = avg_value
                final_action = action
        
        # 计算置信度 - 最高平均值和次高平均值的差距
        sorted_values = sorted(avg_values.values())
        if len(sorted_values) > 1:
            # 归一化置信度 (0-1之间)
            confidence = (sorted_values[-1] - sorted_values[-2]) / sorted_values[-1] if sorted_values[-1] > 0 else 0
        else:
            confidence = 0.5  # 默认置信度
        
        return {
            "action": int(final_action),
            "confidence": float(confidence),
            "is_confident": confidence > 0.6,  # 可调整阈值
            "action_values": avg_values
        }
    
    def bayesian_ensemble(self, predictions: List[Dict[str, Any]], 
                      prior: Optional[Dict[int, float]] = None) -> Dict[str, Any]:
        """
        贝叶斯集成方法，考虑模型权重和先验概率
        
        Args:
            predictions: 多个模型的预测结果列表
            prior: 动作的先验概率 {action: probability}
            
        Returns:
            Dict[str, Any]: 集成后的预测结果
        """
        if not predictions:
            return {
                "action": 1,  # 默认动作 (HOLD)
                "confidence": 0.0,
                "is_confident": False
            }
        
        # 如果没有提供先验，使用均匀分布
        if prior is None:
            unique_actions = set()
            for pred in predictions:
                unique_actions.add(pred.get("action", 1))
            
            # 默认先验概率
            prior = {action: 1.0 / len(unique_actions) for action in unique_actions}
            
            # 如果没有定义所有可能的动作，设置默认值
            for action in [0, 1, 2]:  # 假设动作空间为 [0, 1, 2]
                if action not in prior:
                    prior[action] = 0.01  # 小概率
        
        # 计算后验概率
        posterior = prior.copy()
        
        for pred in predictions:
            action = pred.get("action", 1)
            confidence = pred.get("confidence", 0.5)
            
            # 更新该动作的后验概率
            for a in posterior:
                if a == action:
                    # 如果是预测的动作，增强其概率
                    posterior[a] *= confidence
                else:
                    # 如果不是预测的动作，降低其概率
                    posterior[a] *= (1 - confidence) / (len(posterior) - 1)
        
        # 归一化后验概率
        total_prob = sum(posterior.values())
        if total_prob > 0:
            for a in posterior:
                posterior[a] /= total_prob
        
        # 找出后验概率最高的动作
        final_action = max(posterior, key=posterior.get)
        max_prob = posterior[final_action]
        
        # 计算置信度 - 最高概率和次高概率的差
        sorted_probs = sorted(posterior.values())
        if len(sorted_probs) > 1:
            confidence = sorted_probs[-1] - sorted_probs[-2]
        else:
            confidence = sorted_probs[-1]
        
        return {
            "action": int(final_action),
            "confidence": float(confidence),
            "is_confident": confidence > 0.6,  # 可调整阈值
            "probability": max_prob,
            "posterior": posterior
        }
    
    def ensemble_predictions(self, predictions: List[Dict[str, Any]], 
                         method: Optional[str] = None,
                         weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        根据指定的方法集成多个预测结果
        
        Args:
            predictions: 多个模型的预测结果列表
            method: 集成方法，如果为None则使用配置中的方法
            weights: 权重列表，用于加权投票
            
        Returns:
            Dict[str, Any]: 集成后的预测结果
        """
        if method is None:
            method = self.method
        
        result = None
        
        if method == "voting":
            result = self.majority_voting(predictions)
        elif method == "weighted_voting":
            result = self.weighted_voting(predictions, weights)
        elif method == "average":
            result = self.average_predictions(predictions)
        elif method == "bayesian":
            result = self.bayesian_ensemble(predictions)
        else:
            self.logger.warning(f"未知的集成方法: {method}，使用加权投票")
            result = self.weighted_voting(predictions, weights)
        
        return result
