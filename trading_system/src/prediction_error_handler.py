"""
预测错误处理模块 - 提供增强的错误处理和预测修复功能
"""
import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import traceback

class PredictionErrorHandler:
    """
    增强的预测错误处理器，处理模型预测过程中的各类异常
    并提供回退策略和数据修复功能
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化错误处理器
        
        参数:
        - config: 配置字典
        """
        self.logger = logging.getLogger("PredictionErrorHandler")
        self.config = config or {}
        
        # 错误统计
        self.error_stats = {
            'total_errors': 0,
            'nan_errors': 0,
            'shape_errors': 0,
            'prediction_errors': 0,
            'other_errors': 0,
            'last_error_time': None,
            'consecutive_errors': 0
        }
        
        # 回退策略跟踪
        self.fallback_history = []
        self.max_fallback_history = 100
        
        self.logger.info("预测错误处理器初始化完成")
        
    def handle_nan_values(self, state: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        处理状态数据中的NaN或Inf值
        
        参数:
        - state: 输入状态数据
        
        返回:
        - 修复后的状态数据
        - 信息字典
        """
        info = {'fixed': False, 'nan_count': 0, 'inf_count': 0}
        
        if not hasattr(state, 'dtype') or not np.issubdtype(state.dtype, np.number):
            return state, info
        
        # 检测NaN和Inf的数量
        nan_mask = np.isnan(state)
        inf_mask = np.isinf(state)
        
        nan_count = np.sum(nan_mask)
        inf_count = np.sum(inf_mask)
        
        if nan_count > 0 or inf_count > 0:
            # 更新统计信息
            self.error_stats['nan_errors'] += 1
            self.error_stats['total_errors'] += 1
            
            # 记录问题数量
            info['fixed'] = True
            info['nan_count'] = int(nan_count)
            info['inf_count'] = int(inf_count)
            
            # 修复数据
            fixed_state = np.copy(state)
            
            # 如果NaN/Inf比例过高（超过20%），使用零填充
            total_elements = state.size
            problem_ratio = (nan_count + inf_count) / total_elements
            
            if problem_ratio > 0.2:
                # 大量问题数据，整体替换为零
                self.logger.warning(f"严重的数据问题: {problem_ratio:.1%}的元素是NaN/Inf，使用零填充所有值")
                fixed_state = np.zeros_like(state)
                info['severe_issue'] = True
            else:
                # 只替换问题数据
                fixed_state[nan_mask] = 0.0
                fixed_state[inf_mask] = 0.0
                self.logger.debug(f"修复了数据中的{nan_count}个NaN和{inf_count}个Inf值")
            
            return fixed_state, info
        
        return state, info
    
    def handle_shape_error(self, state: np.ndarray, expected_shape: Tuple[int, ...]) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        处理状态数据形状不匹配的问题
        
        参数:
        - state: 输入状态数据
        - expected_shape: 期望的状态形状
        
        返回:
        - 修复后的状态数据，如果无法修复则返回None
        - 信息字典
        """
        info = {'fixed': False, 'original_shape': state.shape if hasattr(state, 'shape') else 'unknown'}
        
        # 检查状态是否为空
        if state is None or (hasattr(state, 'size') and state.size == 0):
            self.error_stats['shape_errors'] += 1
            self.error_stats['total_errors'] += 1
            info['error'] = "空状态数据"
            return None, info
        
        # 如果形状已经匹配，无需修复
        if hasattr(state, 'shape') and state.shape == expected_shape:
            return state, info
        
        # 形状不匹配，尝试修复
        try:
            # 对于多维观察空间
            if len(expected_shape) > 1:
                # 如果维度数量不匹配但总元素数量匹配，尝试重塑
                if hasattr(state, 'size') and state.size == np.prod(expected_shape):
                    reshaped_state = state.reshape(expected_shape)
                    info['fixed'] = True
                    info['method'] = 'reshape'
                    return reshaped_state, info
                
                # 如果只是缺少批次维度，添加它
                if hasattr(state, 'shape') and len(state.shape) == len(expected_shape) - 1:
                    if state.shape == expected_shape[1:]:
                        expanded_state = np.expand_dims(state, 0)
                        info['fixed'] = True
                        info['method'] = 'expand_dims'
                        return expanded_state, info
                
                # 尝试创建适当大小的零数组，并复制尽可能多的数据
                zero_state = np.zeros(expected_shape)
                
                # 如果输入是一维的，可能需要特殊处理
                if len(state.shape) == 1:
                    # 填充第一行
                    min_length = min(state.shape[0], expected_shape[1])
                    zero_state[0, :min_length] = state[:min_length]
                else:
                    # 尝试复制尽可能多的二维数据
                    rows = min(state.shape[0] if len(state.shape) > 0 else 0, expected_shape[0])
                    cols = min(state.shape[1] if len(state.shape) > 1 else 0, expected_shape[1])
                    if rows > 0 and cols > 0:
                        zero_state[:rows, :cols] = state[:rows, :cols]
                
                info['fixed'] = True
                info['method'] = 'zero_pad'
                info['partial_data'] = True
                self.logger.warning(f"形状不兼容，使用零填充创建了合适的形状: {state.shape} -> {expected_shape}")
                
                # 更新统计
                self.error_stats['shape_errors'] += 1
                self.error_stats['total_errors'] += 1
                
                return zero_state, info
            
            else:
                # 一维观察空间处理
                reshaped_state = np.array(state).reshape(expected_shape)
                info['fixed'] = True
                info['method'] = 'reshape_1d'
                return reshaped_state, info
                
        except Exception as e:
            self.error_stats['shape_errors'] += 1
            self.error_stats['total_errors'] += 1
            info['error'] = str(e)
            info['fixed'] = False
            self.logger.error(f"无法修复状态形状: {e}")
            return None, info
    
    def generate_fallback_action(self, 
                                error_type: str, 
                                market_data: Optional[Dict[str, Any]] = None, 
                                position_data: Optional[Dict[str, Any]] = None) -> Tuple[float, Dict[str, Any]]:
        """
        根据错误类型和市场数据生成回退动作
        
        参数:
        - error_type: 错误类型
        - market_data: 市场数据（可选）
        - position_data: 持仓数据（可选）
        
        返回:
        - 回退动作值
        - 信息字典
        """
        fallback_info = {
            'used_fallback': True,
            'reason': error_type,
            'original_action': None,
            'strategy': 'default'
        }
        
        # 默认回退动作是不做任何交易
        action = 0.0
        
        # 如果有足够的市场和持仓数据，可以使用更智能的回退策略
        if market_data and position_data:
            # 检查是否已有持仓
            position_size = position_data.get('size', 0)
            position_side = 1 if position_data.get('side', '') == 'BUY' else -1 if position_data.get('side', '') == 'SELL' else 0
            current_position = position_size * position_side
            
            # 价格变动
            price_change = None
            open_price = market_data.get('open', 0)
            close_price = market_data.get('close', 0)
            if open_price and close_price:
                price_change = (close_price / open_price) - 1
            
            # 简单的趋势跟踪策略
            if price_change:
                # 如果已有持仓，考虑减仓
                if abs(current_position) > 0:
                    # 如果趋势方向与持仓方向相反，逐步减仓
                    if (current_position > 0 and price_change < -0.01) or (current_position < 0 and price_change > 0.01):
                        action = -0.3 * np.sign(current_position)  # 逐步减仓
                        fallback_info['strategy'] = 'reduce_position'
                    else:
                        action = 0.0  # 保持当前持仓
                        fallback_info['strategy'] = 'hold_position'
                # 如果无持仓，可以考虑开仓
                elif self.error_stats['consecutive_errors'] <= 3:  # 仅当连续错误较少时考虑开仓
                    if abs(price_change) > 0.02:  # 有明显的价格趋势
                        action = 0.2 * np.sign(price_change)  # 小仓位顺势交易
                        fallback_info['strategy'] = 'small_trend_position'
        
        # 记录回退历史
        self.fallback_history.append({
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'action': action,
            'strategy': fallback_info['strategy']
        })
        
        # 保持历史记录在一定大小范围内
        if len(self.fallback_history) > self.max_fallback_history:
            self.fallback_history.pop(0)
        
        # 更新统计
        self.error_stats['last_error_time'] = datetime.now()
        self.error_stats['consecutive_errors'] += 1
        
        return action, fallback_info
    
    def record_successful_prediction(self):
        """
        记录成功的预测，重置连续错误计数
        """
        self.error_stats['consecutive_errors'] = 0
    
    def get_error_stats(self) -> Dict[str, Any]:
        """
        获取错误统计信息
        
        返回:
        - 错误统计字典
        """
        return self.error_stats.copy()
        
    def reset_error_stats(self):
        """
        重置错误统计信息
        """
        for key in self.error_stats:
            if key != 'last_error_time':
                self.error_stats[key] = 0
