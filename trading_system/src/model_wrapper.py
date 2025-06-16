"""
模型包装器模块 - 负责加载和使用训练好的强化学习模型
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
import json
import shutil
import time
from pathlib import Path
import importlib
from collections import deque
from datetime import datetime, timedelta
import threading
from typing import Dict, List, Any, Tuple, Optional

# 添加项目根目录到路径中，以便导入btc_rl模块
root_path = str(Path(__file__).parent.parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

# 导入项目中的模型相关模块
from stable_baselines3 import SAC
from btc_rl.src.policies import TimeSeriesCNN
from btc_rl.src.env import BtcTradingEnv as TradingEnv
from subprocess import Popen, PIPE, STDOUT
# 导入自适应风险控制模块
from trading_system.src.adaptive_risk import AdaptiveRiskController
# 导入预测错误处理模块
from trading_system.src.prediction_error_handler import PredictionErrorHandler

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
        
        # 历史数据缓冲区，用于存储最近的市场和持仓数据
        self.history_buffer_size = 99  # 与模型预期的观察窗口大小相同
        self.market_history = deque(maxlen=self.history_buffer_size)
        self.position_history = deque(maxlen=self.history_buffer_size)
        self.last_update_time = None
        
        # 初始化市场特征缓存
        self.market_features_cache = {
            'volatility': 0.0,
            'trend': 0.0,
            'regime': 'neutral',
            'last_calculated': datetime.now() - timedelta(hours=1)
        }
        
        # 初始化自适应风险控制器
        self.risk_controller = AdaptiveRiskController(config or {})
        
        # 初始化预测错误处理器
        self.error_handler = PredictionErrorHandler(config or {})
        
        # 数据收集相关
        # 确保config是一个字典
        if not isinstance(config, dict):
            config = {}
        
        # 获取数据目录路径，支持相对路径和绝对路径
        data_dir = config.get('general', {}).get('data_dir', 'trading_system/data')
        if not os.path.isabs(data_dir):
            # 如果是相对路径，相对于项目根目录
            data_dir = os.path.join(root_path, data_dir)
        
        self.data_dir = Path(data_dir)
        self.data_collection_dir = self.data_dir / "collected_data"
        
        # 确保目录存在
        try:
            self.data_collection_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"数据收集目录: {self.data_collection_dir}")
        except Exception as e:
            self.logger.error(f"创建数据收集目录失败: {e}")
            # 使用备用目录
            self.data_collection_dir = Path(os.path.expanduser("~/crypto_trading_data"))
            self.data_collection_dir.mkdir(parents=True, exist_ok=True)
            self.logger.warning(f"使用备用数据收集目录: {self.data_collection_dir}")
        
        # 创建子目录
        for subdir in ["experiences", "trades", "model_backups"]:
            (self.data_collection_dir / subdir).mkdir(exist_ok=True)
            
        self.experiences_buffer = deque(maxlen=10000)  # 存储交易经验
        self.experiences_count = 0
        self.last_model_update_check = datetime.now()
        self.model_update_interval = timedelta(hours=24)  # 每24小时检查是否需要更新模型
        self.model_update_thread = None
        self.stop_update_thread = False
        
        # 性能统计
        self.performance_metrics = {
            'trades_count': 0,
            'profitable_trades': 0,
            'loss_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'model_version': Path(model_path).stem if model_path else "unknown"
        }
        
        # 加载模型
        self._load_policy()
        self.logger.info(f"成功加载模型: {model_path}")
        
        # 尝试加载历史数据
        self._load_history_data()
        
        # 启动模型更新检查线程
        self._start_model_update_thread()
    
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
    
    def _load_history_data(self):
        """加载保存的历史数据"""
        try:
            history_file = Path(self.config.get('general', {}).get('data_dir', 'trading_system/data/market')) / "market_history.npz"
            if history_file.exists():
                data = np.load(history_file)
                market_history = data['market_history']
                position_history = data['position_history']
                
                # 将加载的数据添加到缓冲区
                for i in range(min(len(market_history), self.history_buffer_size)):
                    self.market_history.append(market_history[i])
                    self.position_history.append(position_history[i])
                
                self.logger.info(f"成功加载历史数据: {len(self.market_history)}条记录")
            else:
                self.logger.info("未找到历史数据文件，将使用空缓冲区")
        except Exception as e:
            self.logger.warning(f"加载历史数据失败: {e}")
    
    def _save_history_data(self):
        """保存历史数据"""
        try:
            # 确保目录存在
            data_dir = Path(self.config.get('general', {}).get('data_dir', 'trading_system/data/market'))
            data_dir.mkdir(parents=True, exist_ok=True)
            
            history_file = data_dir / "market_history.npz"
            
            # 将缓冲区转换为numpy数组
            market_history = np.array(list(self.market_history))
            position_history = np.array(list(self.position_history))
            
            # 保存数据
            np.savez(history_file, market_history=market_history, position_history=position_history)
            self.logger.debug(f"历史数据已保存: {len(self.market_history)}条记录")
        except Exception as e:
            self.logger.warning(f"保存历史数据失败: {e}")
    
    def _update_history(self, market_data, position_data):
        """
        更新历史数据缓冲区
        
        参数:
        - market_data: 市场数据
        - position_data: 持仓数据
        """
        # 检查是否需要更新（避免频繁更新，每分钟更新一次）
        current_time = datetime.now()
        if self.last_update_time is not None:
            time_diff = (current_time - self.last_update_time).total_seconds()
            if time_diff < 60:  # 每60秒更新一次
                return
        
        # 提取重要特征并存储
        market_features = {
            'price': market_data.get('close', 0),
            'open': market_data.get('open', 0),
            'high': market_data.get('high', 0),
            'low': market_data.get('low', 0),
            'volume': market_data.get('volume', 0),
            'timestamp': market_data.get('timestamp', int(current_time.timestamp() * 1000))
        }
        
        position_features = {
            'size': position_data.get('size', 0),
            'side': position_data.get('side', ''),
            'entry_price': position_data.get('entry_price', 0),
            'unrealized_pnl': position_data.get('unrealized_pnl', 0),
            'leverage': position_data.get('leverage', 1),
            'margin': position_data.get('margin', 0),
            'timestamp': position_data.get('timestamp', int(current_time.timestamp() * 1000))
        }
        
        # 添加到缓冲区
        self.market_history.append(market_features)
        self.position_history.append(position_features)
        
        # 更新市场状态特征
        self._update_market_features(market_data)
        
        # 更新风险控制器的市场状态
        self.risk_controller.update_market_state(market_data)
        
        # 更新最后更新时间
        self.last_update_time = current_time
        
        # 每10次更新保存一次历史数据
        if len(self.market_history) % 10 == 0:
            self._save_history_data()
            
    def _update_market_features(self, market_data):
        """
        计算和更新扩展市场特征
        
        参数:
        - market_data: 当前市场数据
        """
        # 仅在距离上次计算超过5分钟时更新
        current_time = datetime.now()
        time_since_calculation = (current_time - self.market_features_cache['last_calculated']).total_seconds()
        if time_since_calculation < 300:  # 5分钟
            return
            
        self.market_features_cache['last_calculated'] = current_time
        
        # 如果历史数据不足，不进行计算
        if len(self.market_history) < 24:  # 至少需要24个数据点
            return
            
        try:
            # 提取价格序列
            prices = [entry.get('price', 0) for entry in self.market_history if entry.get('price', 0) > 0]
            if not prices or len(prices) < 5:
                return
                
            # 计算波动率 (基于最近价格变化的标准差)
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) * np.sqrt(24)  # 年化到日波动率
            
            # 计算趋势强度 (基于价格方向的一致性)
            direction_consistency = np.sum(np.sign(returns)) / len(returns) if len(returns) > 0 else 0
            trend_strength = abs(direction_consistency)
            
            # 确定市场状态
            if volatility > 0.03:  # 高波动率
                if trend_strength > 0.6:
                    regime = "trending"
                else:
                    regime = "volatile"
            else:  # 低波动率
                if trend_strength > 0.7:
                    regime = "trending"
                else:
                    regime = "ranging"
            
            # 更新缓存
            self.market_features_cache['volatility'] = float(volatility)
            self.market_features_cache['trend'] = float(trend_strength)
            self.market_features_cache['regime'] = regime
            
            self.logger.debug(f"更新市场特征: 波动率={volatility:.4f}, 趋势={trend_strength:.4f}, 状态={regime}")
            
        except Exception as e:
            self.logger.error(f"计算市场特征时出错: {e}")
            # 出错时保持原有值不变
    
    def preprocess_state(self, market_data, position_data):
        """
        预处理状态数据，将市场数据和持仓数据转换为模型输入格式
        
        参数:
        - market_data: 市场数据 (OHLCV等)
        - position_data: 持仓数据 (包含大小、方向、盈亏等)
        
        返回:
        - 模型输入状态，形状为(99, 9)或需要的观察空间形状
        """
        # 更新历史数据缓冲区
        self._update_history(market_data, position_data)
        
        # 提取当前市场特征
        price = market_data.get('close', 0)
        open_price = market_data.get('open', 0)
        high = market_data.get('high', 0)
        low = market_data.get('low', 0)
        volume = market_data.get('volume', 0)
        
        # 提取当前持仓特征
        position_size = position_data.get('size', 0)
        position_side = 1 if position_data.get('side', '') == 'BUY' else -1 if position_data.get('side', '') == 'SELL' else 0
        upnl = position_data.get('unrealized_pnl', 0)
        entry_price = position_data.get('entry_price', 0)
        leverage = position_data.get('leverage', 1)
        
        # 计算其他特征
        equity = position_data.get('margin', 0) + upnl
        
        # 标准化特征
        norm_position = position_size * position_side / 100  # 标准化持仓大小
        norm_upnl = upnl / (equity or 1)  # 标准化未实现盈亏
        price_change = (price / open_price) - 1 if open_price else 0  # 价格变动百分比
        
        # 获取增强的市场特征
        volatility = self.market_features_cache.get('volatility', 0.0)
        trend_strength = self.market_features_cache.get('trend', 0.0)
        
        # 市场状态编码 (one-hot)
        regime = self.market_features_cache.get('regime', 'neutral')
        regime_encoding = {
            'trending': [1, 0, 0, 0],
            'ranging': [0, 1, 0, 0],
            'volatile': [0, 0, 1, 0],
            'neutral': [0, 0, 0, 1],
        }.get(regime, [0, 0, 0, 1])
        
        # 构建当前特征向量 (保持原始特征数量为9，以兼容模型)
        current_feature = np.array([
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
        
        # 如果有增强特征可用，但我们保持特征数量不变以兼容模型
        # 增强特征如volatility和trend_strength可以在这里使用
        # 但由于我们需要保持特征向量与原始模型兼容，所以不直接添加
        
        # 检查环境观察空间的形状并相应地构造输入
        if hasattr(self, 'state_shape') and len(self.state_shape) >= 2:
            rows, cols = self.state_shape[0], self.state_shape[1]
            observation = np.zeros(self.state_shape)
            
            # 如果历史缓冲区有足够数据，使用历史数据构建状态
            if len(self.market_history) >= rows:
                for i in range(rows):
                    # 从历史中获取数据（最新的数据在末尾）
                    idx = -(rows - i)
                    hist_market = self.market_history[idx]
                    hist_position = self.position_history[idx]
                    
                    # 提取历史特征
                    hist_price = hist_market.get('price', 0)
                    hist_open = hist_market.get('open', 0)
                    hist_high = hist_market.get('high', 0)
                    hist_low = hist_market.get('low', 0)
                    hist_volume = hist_market.get('volume', 0)
                    
                    hist_position_size = hist_position.get('size', 0)
                    hist_position_side = 1 if hist_position.get('side', '') == 'BUY' else -1 if hist_position.get('side', '') == 'SELL' else 0
                    hist_upnl = hist_position.get('unrealized_pnl', 0)
                    hist_leverage = hist_position.get('leverage', 1)
                    hist_margin = hist_position.get('margin', 0)
                    
                    # 计算历史特征
                    hist_equity = hist_margin + hist_upnl
                    hist_norm_position = hist_position_size * hist_position_side / 100
                    hist_norm_upnl = hist_upnl / (hist_equity or 1)
                    hist_price_change = (hist_price / hist_open) - 1 if hist_open else 0
                    
                    # 创建历史特征向量
                    hist_feature = np.array([
                        hist_price / 1000,
                        hist_high / 1000,
                        hist_low / 1000,
                        hist_volume / 1e6,
                        hist_price_change,
                        hist_norm_position,
                        hist_norm_upnl,
                        hist_leverage / 10,
                        hist_equity / 10000
                    ])
                    
                    # 将历史特征添加到观察空间
                    observation[i, :min(cols, len(hist_feature))] = hist_feature[:min(cols, len(hist_feature))]
            else:
                # 历史数据不足，使用当前数据填充
                for i in range(rows):
                    observation[i, :min(cols, len(current_feature))] = current_feature[:min(cols, len(current_feature))]
                self.logger.debug(f"历史数据不足，使用当前数据填充: {len(self.market_history)}/{rows}")
            
            return observation
        else:
            # 如果是一维观察空间，使用原来的处理方法
            padded_state = np.zeros(self.state_size)
            padded_state[:min(len(current_feature), self.state_size)] = current_feature[:min(len(current_feature), self.state_size)]
            return padded_state
    
    def predict_action(self, state, market_data=None, position_data=None):
        """
        预测动作
        
        参数:
        - state: 模型输入状态
        - market_data: 市场数据，用于回退策略（可选）
        - position_data: 持仓数据，用于回退策略（可选）
        
        返回:
        - 动作: 范围在[-1, 1]之间的值，正值表示做多，负值表示做空，绝对值表示仓位大小
        - 回退信息: 包含回退策略信息的字典
        """
        if self.policy is None:
            self.logger.error("模型未加载，无法预测")
            return 0.0, {'used_fallback': True, 'reason': "模型未加载"}
        
        # 创建回退策略字典
        fallback_info = {
            'used_fallback': False,
            'reason': None,
            'original_action': None
        }
            
        try:
            # 验证状态数据
            if state is None or (hasattr(state, 'size') and state.size == 0):
                self.logger.error("无效的状态数据：状态为空")
                return self.error_handler.generate_fallback_action("空状态数据", market_data, position_data)
            
            # 使用错误处理器处理NaN和Inf值
            state, nan_info = self.error_handler.handle_nan_values(state)
            if nan_info['fixed']:
                self.logger.warning(f"修复了状态数据中的{nan_info['nan_count']}个NaN和{nan_info['inf_count']}个Inf值")
                fallback_info['reason'] = f"修复了NaN/Inf值: {nan_info['nan_count']}个NaN, {nan_info['inf_count']}个Inf"
            
            # 根据模型的观察空间形状，确保输入形状正确
            if hasattr(self, 'state_shape') and len(self.state_shape) >= 2:
                # 使用错误处理器处理形状问题
                if len(self.state_shape) > 1:
                    # 多维观察空间的预期形状
                    batch_shape = (1,) + self.state_shape
                    model_state, shape_info = self.error_handler.handle_shape_error(state, self.state_shape)
                    
                    # 如果无法修复形状，使用回退策略
                    if model_state is None:
                        self.logger.error(f"无法修复状态形状: {shape_info.get('error', '未知错误')}")
                        return self.error_handler.generate_fallback_action("形状错误", market_data, position_data)
                    
                    # 增加批次维度（如果需要）
                    if len(model_state.shape) == len(self.state_shape):  # 没有批次维度
                        model_state = np.expand_dims(model_state, 0)  # 添加批次维度
                    
                    # 对于形状(99,9)的观察空间，确保输入是(1,99,9)
                    if self.state_shape == (99, 9) and model_state.shape != (1, 99, 9):
                        if model_state.shape == (99, 9):
                            model_state = np.expand_dims(model_state, 0)
                        else:
                            # 处理完全不匹配的情况
                            self.logger.error(f"无法调整状态形状到所需的(1, 99, 9), 当前形状: {model_state.shape}")
                            return self.error_handler.generate_fallback_action("形状不兼容", market_data, position_data)
                else:
                    # 对于无形状问题，简单添加批次维度
                    model_state = np.expand_dims(state, 0) if len(state.shape) == len(self.state_shape) else state
            else:
                # 一维观察空间处理
                try:
                    model_state = np.array(state).reshape(1, -1)
                except Exception as reshape_err:
                    self.logger.error(f"一维状态重塑失败: {reshape_err}")
                    return self.error_handler.generate_fallback_action("形状重塑失败", market_data, position_data)
            
            # 记录实际输入模型的形状
            self.logger.debug(f"输入模型的状态形状: {model_state.shape}")
            
            # 获取模型预测的动作
            try:
                action, _ = self.policy.predict(model_state, deterministic=True)
                # 记录成功的预测
                self.error_handler.record_successful_prediction()
            except Exception as predict_err:
                self.logger.error(f"模型预测失败: {predict_err}")
                return self.error_handler.generate_fallback_action("预测错误", market_data, position_data)
            
            # 验证动作
            if action is None or not isinstance(action, np.ndarray) or action.size == 0:
                self.logger.error("模型返回了无效的动作")
                return self.error_handler.generate_fallback_action("无效动作", market_data, position_data)
            
            # 检查动作是否包含NaN或Inf
            if np.isnan(action).any() or np.isinf(action).any():
                self.logger.error("模型返回了NaN或Inf动作值")
                # 尝试修复动作
                fixed_action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
                if np.all(fixed_action == 0):
                    # 如果修复后全是零，使用回退策略
                    return self.error_handler.generate_fallback_action("NaN/Inf动作值", market_data, position_data)
                else:
                    # 使用修复后的动作，但标记使用了回退
                    action = fixed_action
                    fallback_info['reason'] = "修复了动作中的NaN/Inf值"
                    fallback_info['used_fallback'] = True
            
            # 将动作值裁剪到[-1, 1]范围
            original_action = action[0]
            action_value = np.clip(original_action, -1.0, 1.0)
            
            # 检查是否进行了裁剪
            if action_value != original_action:
                self.logger.warning(f"动作值被裁剪: {original_action} -> {action_value}")
                fallback_info['reason'] = f"动作值被裁剪: {original_action} -> {action_value}"
            
            self.logger.debug(f"模型预测动作: {action_value}")
            fallback_info['original_action'] = float(original_action)
            
            return float(action_value), fallback_info
            
        except Exception as e:
            self.logger.error(f"模型预测失败: {e}")
            self.logger.error(f"错误详情: {str(e)}")
            self.logger.error(f"状态形状: {state.shape if hasattr(state, 'shape') else '未知'}")
            import traceback
            self.logger.error(f"异常跟踪: {traceback.format_exc()}")
            
            # 使用错误处理器生成回退动作
            return self.error_handler.generate_fallback_action("未处理的异常", market_data, position_data)
    
    def get_trade_decision(self, market_data, position_data, risk_limit=None):
        """
        获取交易决策
        
        参数:
        - market_data: 市场数据
        - position_data: 持仓数据
        - risk_limit: 风险限制系数，如果为None则使用自适应风险控制
        
        返回:
        - action_type: 动作类型 ("BUY", "SELL", "HOLD")
        - action_value: 动作值，表示仓位大小的百分比
        - confidence: 置信度（0-1之间）
        """
        # 预处理状态数据
        state = self.preprocess_state(market_data, position_data)
        
        # 获取模型预测的动作和错误处理信息
        # 传递完整的市场和持仓数据，用于智能回退策略
        action_value, fallback_info = self.predict_action(state, market_data, position_data)
        
        # 动作方向
        action_type = "BUY" if action_value > 0.05 else "SELL" if action_value < -0.05 else "HOLD"
        
        # 计算动作的绝对值和置信度
        abs_action = abs(action_value)
        
        # 如果使用了回退策略，降低置信度
        if fallback_info.get('used_fallback', False):
            confidence = 0.3  # 使用回退策略时，置信度固定为低值
            self.logger.warning(f"使用回退策略生成决策: {fallback_info.get('reason', '未知原因')}")
        else:
            confidence = min(abs_action * 1.5, 1.0)  # 正常情况下的置信度计算
        
        # 获取自适应风险参数
        risk_params = self.risk_controller.get_adjusted_risk_parameters()
        adaptive_risk = risk_params['risk_per_trade_pct']
        
        # 如果指定了风险限制，优先使用指定值；否则使用自适应风险
        risk_limit = risk_limit if risk_limit is not None else adaptive_risk
        
        # 应用风险限制
        adjusted_action = action_value * risk_limit / max(0.01, abs_action) if abs_action > risk_limit else action_value
        
        # 计算预期的奖励（简单估计），用于数据收集
        # 这是一个非常粗略的估计，实际奖励应当在交易后根据真实结果计算
        estimated_reward = 0
        position_size = position_data.get('size', 0)
        position_side = 1 if position_data.get('side', '') == 'BUY' else -1 if position_data.get('side', '') == 'SELL' else 0
        current_position = position_size * position_side
        
        # 仅在做出明确交易决策时记录交易经验
        if abs_action > 0.05:
            # 计算估计奖励，考虑简单的动量因素
            price_change_pct = (market_data.get('close', 0) / market_data.get('open', 0) - 1) if market_data.get('open', 0) else 0
            if (action_value > 0 and price_change_pct > 0) or (action_value < 0 and price_change_pct < 0):
                # 顺势交易
                estimated_reward = abs(action_value) * abs(price_change_pct) * 10  # 正向奖励
            else:
                # 逆势交易
                estimated_reward = -abs(action_value) * abs(price_change_pct) * 5   # 负向奖励
            
            # 记录交易经验供后续分析和模型更新
            self.record_trading_experience(
                market_data=market_data,
                position_data=position_data,
                action_value=action_value,
                reward=estimated_reward
            )
        
        # 构建决策结果
        decision = {
            "action_type": action_type,
            "action_value": adjusted_action,
            "confidence": confidence,
            "raw_action": action_value,
            "fallback_used": fallback_info.get('used_fallback', False),
            "fallback_reason": fallback_info.get('reason', None),
            "timestamp": int(datetime.now().timestamp() * 1000),
            "risk_limit_used": risk_limit,
            "market_regime": risk_params['market_regime'],
            "volatility": risk_params['volatility'],
            "trend_strength": risk_params['trend_strength'],
            "error_stats": self.error_handler.get_error_stats() if fallback_info.get('used_fallback', False) else None
        }
        
        return decision
    
    def record_trading_experience(self, market_data: dict, position_data: dict, 
                              action_value: float, reward: float, next_market_data: dict = None):
        """
        记录交易经验，用于后续模型更新
        
        参数:
        - market_data: 当前市场数据
        - position_data: 当前持仓数据
        - action_value: 模型预测的动作值
        - reward: 该动作获得的奖励（可以是预估的，也可以是实际的）
        - next_market_data: 下一个时间点的市场数据（可选）
        """
        try:
            # 构建经验数据
            timestamp = int(datetime.now().timestamp() * 1000)
            experience = {
                'timestamp': timestamp,
                'market_data': {k: market_data.get(k, 0) for k in ['close', 'open', 'high', 'low', 'volume']},
                'position_data': {
                    'size': position_data.get('size', 0),
                    'side': position_data.get('side', ''),
                    'entry_price': position_data.get('entry_price', 0),
                    'unrealized_pnl': position_data.get('unrealized_pnl', 0)
                },
                'action': action_value,
                'reward': reward
            }
            
            if next_market_data:
                experience['next_market_data'] = {
                    k: next_market_data.get(k, 0) for k in ['close', 'open', 'high', 'low', 'volume']
                }
            
            # 添加到缓冲区
            self.experiences_buffer.append(experience)
            self.experiences_count += 1
            
            # 每积累100个经验保存一次
            if self.experiences_count % 100 == 0:
                self._save_experiences()
                self.logger.info(f"已保存{self.experiences_count}条交易经验")
                
            # 更新性能指标
            if reward > 0:
                self.performance_metrics['profitable_trades'] += 1
                self.performance_metrics['total_profit'] += reward
            elif reward < 0:
                self.performance_metrics['loss_trades'] += 1
                self.performance_metrics['total_loss'] += abs(reward)
            self.performance_metrics['trades_count'] += 1
                
        except Exception as e:
            self.logger.warning(f"记录交易经验失败: {e}")
    
    def _save_experiences(self):
        """保存交易经验到文件"""
        try:
            # 确保目录存在
            experiences_dir = self.data_collection_dir / "experiences"
            experiences_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建文件名，包含时间戳以避免覆盖
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = experiences_dir / f"exp_batch_{timestamp}.json"
            
            # 保存到JSON文件
            with open(file_path, 'w') as f:
                json.dump(list(self.experiences_buffer), f, indent=2)
                
        except Exception as e:
            self.logger.error(f"保存交易经验失败: {e}")
    
    def update_model_performance(self, trade_result: dict):
        """
        更新模型性能统计信息
        
        参数:
        - trade_result: 交易结果，包含profit_pct, is_profitable等信息
        """
        try:
            # 更新交易统计
            is_profitable = trade_result.get('is_profitable', False)
            profit_pct = trade_result.get('profit_pct', 0.0)
            absolute_profit = trade_result.get('absolute_profit', 0.0)
            
            # 更新性能指标
            metrics_file = self.data_collection_dir / "model_metrics.json"
            
            # 读取现有指标（如果存在）
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r') as f:
                        self.performance_metrics = json.load(f)
                except json.JSONDecodeError:
                    pass  # 使用默认初始化的指标
            
            # 更新指标
            self.performance_metrics['trades_count'] += 1
            if is_profitable:
                self.performance_metrics['profitable_trades'] += 1
                self.performance_metrics['total_profit'] += absolute_profit
            else:
                self.performance_metrics['loss_trades'] += 1
                self.performance_metrics['total_loss'] += abs(absolute_profit)
            
            # 计算胜率和盈亏比
            if self.performance_metrics['trades_count'] > 0:
                self.performance_metrics['win_rate'] = self.performance_metrics['profitable_trades'] / self.performance_metrics['trades_count']
            
            if self.performance_metrics['loss_trades'] > 0 and self.performance_metrics['total_loss'] > 0:
                pl_ratio = (self.performance_metrics['total_profit'] / max(1, self.performance_metrics['profitable_trades'])) / \
                        (self.performance_metrics['total_loss'] / max(1, self.performance_metrics['loss_trades']))
                self.performance_metrics['profit_loss_ratio'] = pl_ratio
            
            # 保存更新后的指标
            with open(metrics_file, 'w') as f:
                json.dump(self.performance_metrics, f, indent=2)
                
            self.logger.debug(f"已更新模型性能统计: 总交易{self.performance_metrics['trades_count']}笔, "
                           f"胜率{self.performance_metrics.get('win_rate', 0):.2%}")
                
        except Exception as e:
            self.logger.warning(f"更新模型性能统计失败: {e}")
    
    def _start_model_update_thread(self):
        """启动模型更新检查线程"""
        if self.model_update_thread is None:
            self.stop_update_thread = False
            self.model_update_thread = threading.Thread(
                target=self._model_update_monitor,
                daemon=True,
                name="ModelUpdateMonitor"
            )
            self.model_update_thread.start()
            self.logger.info("已启动模型更新监控线程")
    
    def _stop_model_update_thread(self):
        """停止模型更新线程"""
        self.stop_update_thread = True
        if self.model_update_thread and self.model_update_thread.is_alive():
            self.model_update_thread.join(timeout=5)
            self.model_update_thread = None
    
    def _model_update_monitor(self):
        """模型更新监控线程的主函数"""
        self.logger.info("模型更新监控线程启动")
        
        while not self.stop_update_thread:
            try:
                # 检查是否需要更新模型
                current_time = datetime.now()
                time_since_last_check = current_time - self.last_model_update_check
                
                if time_since_last_check > self.model_update_interval:
                    self.logger.info("执行定期模型更新检查")
                    self.check_and_update_model()
                    self.last_model_update_check = current_time
                
                # 休眠一段时间（每小时检查一次）
                for _ in range(60):  # 分步休眠，以便能够更快地响应停止信号
                    if self.stop_update_thread:
                        break
                    time.sleep(60)  # 每分钟检查一次停止信号
                    
            except Exception as e:
                self.logger.error(f"模型更新监控线程异常: {e}")
                time.sleep(300)  # 发生错误后等待5分钟再继续
    
    def check_and_update_model(self):
        """检查是否有新的更好模型，并进行更新"""
        try:
            # 1. 检查是否有足够的交易经验数据
            experiences_dir = self.data_collection_dir / "experiences"
            if not experiences_dir.exists() or len(list(experiences_dir.glob("exp_batch_*.json"))) < 5:
                self.logger.info("交易经验数据不足，暂不更新模型")
                return False
            
            # 2. 计算当前模型的性能指标
            metrics_file = self.data_collection_dir / "model_metrics.json"
            if not metrics_file.exists():
                self.logger.info("没有模型性能数据，暂不更新模型")
                return False
            
            # 3. 运行select_best_model.py脚本，获取最新的最佳模型
            self.logger.info("运行模型选择脚本，查找最佳模型...")
            best_model_cmd = [
                sys.executable,
                str(Path(root_path) / "select_best_model.py")
            ]
            
            process = Popen(best_model_cmd, stdout=PIPE, stderr=STDOUT, text=True)
            output, _ = process.communicate()
            
            if process.returncode != 0:
                self.logger.warning(f"模型选择脚本执行失败: {output}")
                return False
                
            # 4. 解析输出，获取最佳模型路径
            best_model_path = None
            for line in output.split('\n'):
                if "模型路径:" in line:
                    best_model_path = line.split("模型路径:")[1].strip()
                    break
            
            if not best_model_path:
                self.logger.warning("未能从脚本输出中解析出最佳模型路径")
                return False
                
            # 5. 检查是否与当前模型不同
            current_model_path = self.model_path
            if best_model_path == current_model_path:
                self.logger.info(f"当前模型已是最佳模型，无需更新: {current_model_path}")
                return False
                
            # 6. 加载新模型
            self.logger.info(f"发现更好的模型: {best_model_path}，准备更新...")
            
            # 创建模型备份
            backup_dir = self.data_dir / "model_backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            current_model_name = Path(current_model_path).name
            backup_path = backup_dir / f"{current_model_name}.bak.{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            try:
                shutil.copy2(current_model_path, backup_path)
                self.logger.info(f"已备份当前模型: {backup_path}")
            except Exception as e:
                self.logger.warning(f"模型备份失败: {e}")
            
            # 更新模型路径并重新加载
            self.model_path = best_model_path
            self._load_policy()
            
            self.logger.info(f"模型更新成功: {best_model_path}")
            
            # 更新模型版本信息
            self.performance_metrics['model_version'] = Path(best_model_path).stem
            with open(metrics_file, 'w') as f:
                json.dump(self.performance_metrics, f, indent=2)
            
            return True
                
        except Exception as e:
            self.logger.error(f"模型更新检查失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def record_trade_result(self, trade_result: Dict[str, Any]):
        """
        记录实际交易结果并用于性能评估
        
        参数:
        - trade_result: 交易结果字典，应包含以下字段：
          - trade_id: 交易ID
          - entry_time: 入场时间
          - exit_time: 出场时间
          - side: 交易方向 (BUY/SELL)
          - entry_price: 入场价格
          - exit_price: 出场价格
          - size: 交易大小
          - profit_pct: 收益百分比
          - absolute_profit: 绝对收益金额
          - is_profitable: 是否盈利
        """
        try:
            # 更新模型性能统计
            self.update_model_performance(trade_result)
            
            # 保存详细的交易结果记录
            trades_dir = self.data_collection_dir / "trades"
            trades_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建文件名，使用交易ID或时间戳
            trade_id = trade_result.get('trade_id', str(int(time.time())))
            file_path = trades_dir / f"trade_{trade_id}.json"
            
            # 添加额外信息
            trade_data = trade_result.copy()
            trade_data['model_version'] = self.performance_metrics.get('model_version', 'unknown')
            trade_data['record_time'] = datetime.now().isoformat()
            
            # 保存到JSON文件
            with open(file_path, 'w') as f:
                json.dump(trade_data, f, indent=2)
            
            self.logger.info(f"交易结果已记录: ID={trade_id}, 盈亏={trade_result.get('profit_pct', 0):.2%}")
            
            # 尝试触发定期模型评估
            current_time = datetime.now()
            if (current_time - self.last_model_update_check).total_seconds() > 86400:  # 每天至少检查一次
                threading.Thread(
                    target=self.check_and_update_model,
                    daemon=True,
                    name="ModelUpdateCheck"
                ).start()
                self.last_model_update_check = current_time
            
        except Exception as e:
            self.logger.error(f"记录交易结果失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def __del__(self):
        """析构函数，确保资源正确释放"""
        try:
            # 保存所有待保存的数据
            if hasattr(self, 'experiences_buffer') and self.experiences_buffer:
                self._save_experiences()
                
            # 停止更新线程
            if hasattr(self, '_stop_model_update_thread'):
                self._stop_model_update_thread()
                
            self.logger.info("ModelWrapper资源已正确释放")
        except Exception as e:
            # 在析构函数中不要抛出异常
            if hasattr(self, 'logger'):
                self.logger.warning(f"ModelWrapper析构过程中出错: {e}")
            else:
                print(f"ModelWrapper析构过程中出错: {e}")