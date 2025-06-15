"""
模型包装器模块 - 负责加载和使用训练好的强化学习模型
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
import json
from pathlib import Path
import importlib

# 添加项目根目录到路径中，以便导入btc_rl模块
root_path = str(Path(__file__).parent.parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

# 导入项目中的模型相关模块
from stable_baselines3 import SAC
from btc_rl.src.policies import TimeSeriesCNN
from btc_rl.src.env import BtcTradingEnv as TradingEnv

class ModelWrapper:
    """RL模型包装器，用于加载和使用训练好的模型进行交易决策"""
    
    def __init__(self, model_path, config=None):
        """
        初始化模型包装器
        
        参数:
        - model_path: 模型文件路径
        - config: 配置字典，包含模型参数
        """
        self.logger = logging.getLogger("ModelWrapper")
        self.model_path = model_path
        self.config = config or {}
        self.policy = None
        self.env = None
        self.feature_extractor = None
        self.state_size = None
        self.action_size = None
        
        # 加载模型
        self._load_policy()
        self.logger.info(f"成功加载模型: {model_path}")
    
    def _load_policy(self):
        """加载策略网络"""
        try:
            # 直接加载预训练模型
            self.logger.info(f"正在加载模型: {self.model_path}")
            self.policy = SAC.load(self.model_path)
            
            # 获取模型的观察空间和动作空间
            self.env = self.policy.get_env()
            if self.env is None:
                # 如果模型没有绑定环境，创建一个临时环境
                self.logger.warning("模型没有绑定环境，创建临时环境")
                # 创建空的观察窗口和价格数据
                empty_windows = np.zeros((10, 99, 9))  # 创建10个观察窗口，每个窗口99行9列
                empty_prices = np.ones(10)  # 10个价格点
                
                self.env = TradingEnv(
                    windows=empty_windows,
                    prices=empty_prices,
                    initial_balance=10000,
                    max_leverage=3.0
                )
            
            # 获取状态和动作空间大小
            # 获取观察空间形状
            observation_shape = self.env.observation_space.shape
            
            if len(observation_shape) >= 2:
                # 如果是多维观察空间 (如 99x9)
                self.state_shape = observation_shape
                self.state_size = np.prod(self.state_shape)  # 总元素数
            else:
                # 单维观察空间
                self.state_shape = observation_shape
                self.state_size = observation_shape[0]
                
            self.action_size = self.env.action_space.shape[0]
            self.logger.info(f"策略网络加载成功，观察空间维度: {self.state_shape}")
            
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            raise
    
    def preprocess_state(self, market_data, position_data):
        """
        预处理状态数据，将市场数据和持仓数据转换为模型输入格式
        
        参数:
        - market_data: 市场数据 (OHLCV等)
        - position_data: 持仓数据 (包含大小、方向、盈亏等)
        
        返回:
        - 模型输入状态，形状为(99, 9)或需要的观察空间形状
        """
        # 提取市场特征
        price = market_data.get('close', 0)
        open_price = market_data.get('open', 0)
        high = market_data.get('high', 0)
        low = market_data.get('low', 0)
        volume = market_data.get('volume', 0)
        
        # 提取持仓特征
        position_size = position_data.get('size', 0)
        position_side = 1 if position_data.get('side', '') == 'BUY' else -1 if position_data.get('side', '') == 'SELL' else 0
        upnl = position_data.get('unrealized_pnl', 0)
        entry_price = position_data.get('entry_price', 0)
        leverage = position_data.get('leverage', 1)
        
        # 计算其他特征
        position_value = abs(position_size * price)
        equity = position_data.get('margin', 0) + upnl
        
        # 标准化特征
        norm_position = position_size * position_side / 100  # 标准化持仓大小
        norm_upnl = upnl / (equity or 1)  # 标准化未实现盈亏
        price_change = (price / open_price) - 1 if open_price else 0  # 价格变动百分比
        
        # 构建基本特征向量
        feature_vector = np.array([
            price / 1000,  # 标准化价格
            high / 1000,   # 标准化最高价
            low / 1000,    # 标准化最低价
            volume / 1e6,  # 标准化交易量
            price_change,  # 价格变动
            norm_position, # 标准化持仓
            norm_upnl,     # 标准化未实现盈亏
            leverage / 10, # 标准化杠杆
            equity / 10000 # 标准化权益
        ])
        
        # 检查环境观察空间的形状并相应地构造输入
        if hasattr(self, 'state_shape') and len(self.state_shape) >= 2:
            # 如果观察空间是2D或更高维度(如99x9)
            rows, cols = self.state_shape[0], self.state_shape[1]
            
            # 创建符合形状的观察数据
            observation = np.zeros(self.state_shape)
            
            # 将特征向量复制到每一行，形成时间序列数据的模拟
            # 这种方法适用于模型期望的形状是(99,9)的情况
            for i in range(min(rows, 99)):
                observation[i, :min(cols, len(feature_vector))] = feature_vector[:min(cols, len(feature_vector))]
            
            return observation
        else:
            # 如果是一维观察空间，使用原来的处理方法
            padded_state = np.zeros(self.state_size)
            padded_state[:min(len(feature_vector), self.state_size)] = feature_vector[:min(len(feature_vector), self.state_size)]
            return padded_state
    
    def predict_action(self, state):
        """
        预测动作
        
        参数:
        - state: 模型输入状态
        
        返回:
        - 动作: 范围在[-1, 1]之间的值，正值表示做多，负值表示做空，绝对值表示仓位大小
        """
        if self.policy is None:
            self.logger.error("模型未加载，无法预测")
            return 0.0
            
        try:
            # 根据模型的观察空间形状，确保输入形状正确
            if hasattr(self, 'state_shape') and len(self.state_shape) >= 2:
                # 对于多维观察空间，检查状态形状
                if state.shape != self.state_shape:
                    self.logger.warning(f"状态形状不匹配: 期望{self.state_shape}，实际{state.shape}")
                    # 如果形状不匹配，尝试重塑
                    try:
                        state = state.reshape(self.state_shape)
                    except Exception as reshape_err:
                        self.logger.error(f"重塑状态失败: {reshape_err}")
                
                # 增加批次维度（如果需要）
                if len(state.shape) == len(self.state_shape):  # 没有批次维度
                    model_state = np.expand_dims(state, 0)  # 添加批次维度
                else:
                    model_state = state  # 假设已有批次维度
                    
                # 对于形状(99,9)的观察空间，确保输入是(1,99,9)
                if self.state_shape == (99, 9) and model_state.shape != (1, 99, 9):
                    self.logger.warning(f"状态需要批次维度调整: {model_state.shape} -> (1, 99, 9)")
                    # 如果数据已经正确但缺少批次维度，添加它
                    if model_state.shape == (99, 9):
                        model_state = np.expand_dims(model_state, 0)
                    # 如果数据完全不匹配，记录错误
                    elif model_state.shape != (1, 99, 9):
                        self.logger.error(f"无法调整状态形状到所需的(1, 99, 9)")
            else:
                # 一维观察空间处理
                model_state = np.array(state).reshape(1, -1)
            
            # 记录实际输入模型的形状
            self.logger.debug(f"输入模型的状态形状: {model_state.shape}")
            
            # 获取模型预测的动作
            action, _ = self.policy.predict(model_state, deterministic=True)
            
            # 将动作值裁剪到[-1, 1]范围
            action_value = np.clip(action[0], -1.0, 1.0)
            
            self.logger.debug(f"模型预测动作: {action_value}")
            return float(action_value)
            
        except Exception as e:
            self.logger.error(f"模型预测失败: {e}")
            self.logger.error(f"错误详情: {str(e)}")
            self.logger.error(f"状态形状: {state.shape if hasattr(state, 'shape') else '未知'}")
            return 0.0  # 默认返回0，表示不做任何交易
    
    def get_trade_decision(self, market_data, position_data, risk_limit=0.02):
        """
        获取交易决策
        
        参数:
        - market_data: 市场数据
        - position_data: 持仓数据
        - risk_limit: 风险限制系数
        
        返回:
        - action_type: 动作类型 ("BUY", "SELL", "HOLD")
        - action_value: 动作值，表示仓位大小的百分比
        - confidence: 置信度（0-1之间）
        """
        # 预处理状态数据
        state = self.preprocess_state(market_data, position_data)
        
        # 获取模型预测的动作
        action_value = self.predict_action(state)
        
        # 动作方向
        action_type = "BUY" if action_value > 0.05 else "SELL" if action_value < -0.05 else "HOLD"
        
        # 计算动作的绝对值和置信度
        abs_action = abs(action_value)
        confidence = min(abs_action * 1.5, 1.0)  # 简单的置信度计算，可以根据需要调整
        
        # 应用风险限制
        action_value = action_value * risk_limit / max(0.01, abs_action) if abs_action > risk_limit else action_value
        
        return {
            "action_type": action_type,
            "action_value": action_value,
            "confidence": confidence,
            "raw_action": action_value
        }