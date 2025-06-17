"""
动作映射模块
提供SAC连续动作值到具体交易指令的高级映射功能
"""
import numpy as np
import logging
from typing import Dict, Any, Tuple, Union, Optional

# 动作ID定义
SELL = 0
HOLD = 1
BUY = 2

# 为日志模块创建记录器
logger = logging.getLogger("ActionMapper")

class ActionMapper:
    """
    动作映射类
    将SAC模型输出的连续动作值映射到交易决策和仓位大小
    """
    
    def __init__(self, 
                 min_action_value: float = -1.0, 
                 max_action_value: float = 1.0,
                 buy_threshold: float = 0.05,
                 sell_threshold: float = -0.05,
                 position_scale: float = 1.0):
        """
        初始化动作映射器
        
        Args:
            min_action_value: SAC模型可能输出的最小连续动作值，默认为-1.0
            max_action_value: SAC模型可能输出的最大连续动作值，默认为1.0
            buy_threshold: 大于此阈值被视为买入信号，默认为0.05
            sell_threshold: 小于此阈值被视为卖出信号，默认为-0.05
            position_scale: 仓位缩放因子，控制最大仓位大小，默认为1.0
        """
        self.min_action_value = min_action_value
        self.max_action_value = max_action_value
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.position_scale = position_scale
        
        logger.info(
            f"初始化ActionMapper: 有效动作范围=[{min_action_value}, {max_action_value}], "
            f"买入阈值={buy_threshold}, 卖出阈值={sell_threshold}, 仓位缩放={position_scale}"
        )
    
    def map_action(self, raw_action_value: float) -> Dict[str, Any]:
        """
        将原始连续动作值映射到交易决策和仓位大小
        
        Args:
            raw_action_value: SAC模型输出的原始连续动作值
            
        Returns:
            Dict包含映射后的交易决策信息:
            - action: 动作ID（0=卖出,1=持有,2=买入）
            - action_type: 动作类型的字符串表示
            - position_size: 建议的仓位大小（相对值，范围0-1）
            - signal_strength: 信号强度（0-1之间的浮点数）
            - original_value: 原始动作值
        """
        # 1. 规范化动作值到有效范围
        normalized_action = self._normalize_action_value(raw_action_value)
        
        # 2. 确定动作类型
        action_id, action_type = self._determine_action_type(normalized_action)
        
        # 3. 计算信号强度和仓位大小
        signal_strength, position_size = self._calculate_position_size(
            normalized_action, action_id)
        
        # 4. 构建结果
        result = {
            "action": action_id,
            "action_type": action_type,
            "position_size": position_size,
            "signal_strength": signal_strength,
            "original_value": raw_action_value
        }
        
        logger.debug(
            f"映射动作: 原始值={raw_action_value:.6f}, 标准化值={normalized_action:.6f}, "
            f"动作={action_id}({action_type}), 信号强度={signal_strength:.4f}, "
            f"仓位大小={position_size:.4f}"
        )
        
        return result
    
    def get_action_probabilities(self, raw_action_value: float) -> Dict[str, float]:
        """
        根据连续动作值估计离散动作的概率分布
        采用更接近SAC训练时的连续到离散映射逻辑
        
        Args:
            raw_action_value: 原始连续动作值
            
        Returns:
            Dict[str, float]: 各动作的概率分布
        """
        # 1. 规范化动作值到有效范围
        normalized_action = self._normalize_action_value(raw_action_value)
        
        # 2. 采用softmax风格的概率分布计算
        # 基于与训练时相同的假设：连续值的正负和大小直接反映了交易决策的概率
        
        # 计算基础得分 - 基于动作值相对于决策阈值的位置
        # 使用e^x风格的激活函数使概率分布更符合统计学特征
        if normalized_action < self.sell_threshold:
            # 卖出区间 - 值越小卖出概率越高
            sell_score = np.exp(min(5.0, 1.0 + abs(normalized_action) * 2))
            hold_score = np.exp(0.5)
            buy_score = np.exp(0.1)
        elif normalized_action > self.buy_threshold:
            # 买入区间 - 值越大买入概率越高
            buy_score = np.exp(min(5.0, 1.0 + normalized_action * 2))
            hold_score = np.exp(0.5)
            sell_score = np.exp(0.1)
        else:
            # 持有区间 - 越接近零持有概率越高
            center_distance = abs(normalized_action)
            hold_score = np.exp(1.0 - center_distance * 5)
            
            # 即使在持有区间，也根据值的符号略微倾向买卖一方
            if normalized_action > 0:
                buy_score = np.exp(0.3 + normalized_action)
                sell_score = np.exp(0.2)
            else:
                sell_score = np.exp(0.3 + abs(normalized_action))
                buy_score = np.exp(0.2)
        
        # 3. 归一化为概率
        total_score = sell_score + hold_score + buy_score
        probabilities = {
            "0": sell_score / total_score,  # SELL
            "1": hold_score / total_score,  # HOLD
            "2": buy_score / total_score    # BUY
        }
        
        # 4. 确保概率分布合理 - 避免极端值
        for action_id in probabilities:
            probabilities[action_id] = max(0.01, min(0.98, probabilities[action_id]))
            
        # 5. 重新归一化
        total = sum(probabilities.values())
        for action_id in probabilities:
            probabilities[action_id] /= total
        
        logger.debug(
            f"动作概率推断: 原始值={raw_action_value:.6f}, "
            f"SELL={probabilities['0']:.4f}, HOLD={probabilities['1']:.4f}, BUY={probabilities['2']:.4f}"
        )
        
        return probabilities
    
    def calculate_confidence(self, raw_action_value: float) -> float:
        """
        计算模型决策的置信度
        基于SAC模型输出值的绝对大小和离决策边界的距离
        
        Args:
            raw_action_value: 原始连续动作值
            
        Returns:
            float: 置信度，范围0-1
        """
        # 1. 规范化动作值到有效范围
        normalized_action = self._normalize_action_value(raw_action_value)
        
        # 2. 基于概率分布的置信度计算 (主要方法)
        probs = self.get_action_probabilities(raw_action_value)
        prob_values = [float(probs[str(i)]) for i in range(3)]
        sorted_probs = sorted(prob_values)
        
        if len(sorted_probs) > 1:
            # 最高概率与次高概率的差距越大，置信度越高
            prob_confidence = (sorted_probs[-1] - sorted_probs[-2]) * 1.2
        else:
            prob_confidence = 0.5
        
        # 3. 基于连续值特征的置信度计算 (补充方法)
        # 当动作值远离阈值区域时，置信度更高
        value_confidence = 0.0
        
        if normalized_action < self.sell_threshold:
            # 卖出区域 - 负值越大，置信度越高
            value_distance = abs(normalized_action - self.sell_threshold) / abs(self.min_action_value - self.sell_threshold)
            value_confidence = min(0.95, 0.4 + value_distance * 0.6)
        elif normalized_action > self.buy_threshold:
            # 买入区域 - 正值越大，置信度越高
            value_distance = abs(normalized_action - self.buy_threshold) / abs(self.max_action_value - self.buy_threshold)
            value_confidence = min(0.95, 0.4 + value_distance * 0.6)
        else:
            # 持有区域 - 越接近0，对持有的置信度越高
            # 在阈值边界附近，置信度较低
            center_ratio = 1.0 - abs(normalized_action) / self.buy_threshold
            value_confidence = min(0.8, 0.3 + center_ratio * 0.5)
        
        # 4. 结合两种置信度计算方法
        # 在极端动作值区域，更信任value_confidence；在中间区域，更信任prob_confidence
        if abs(normalized_action) > 0.5:
            confidence = 0.7 * value_confidence + 0.3 * prob_confidence
        else:
            confidence = 0.4 * value_confidence + 0.6 * prob_confidence
        
        # 5. 确保置信度在合理范围内
        confidence = max(0.3, min(0.95, confidence))
        
        logger.debug(f"计算置信度: 原始值={raw_action_value:.6f}, 置信度={confidence:.4f}")
        return confidence
    
    def _normalize_action_value(self, value: float) -> float:
        """规范化动作值到有效范围内"""
        # 确保在指定范围内
        if value < self.min_action_value:
            logger.warning(f"动作值 {value} 小于最小值 {self.min_action_value}，将被裁剪")
            return self.min_action_value
        elif value > self.max_action_value:
            logger.warning(f"动作值 {value} 大于最大值 {self.max_action_value}，将被裁剪")
            return self.max_action_value
        return value
    
    def _determine_action_type(self, normalized_value: float) -> Tuple[int, str]:
        """根据标准化后的动作值确定动作类型"""
        if normalized_value <= self.sell_threshold:  # 使用<=，使得恰好在阈值上的也算
            return SELL, "SELL"
        elif normalized_value >= self.buy_threshold:  # 使用>=，使得恰好在阈值上的也算
            return BUY, "BUY"
        else:
            return HOLD, "HOLD"
    
    def _calculate_position_size(self, normalized_value: float, action_id: int) -> Tuple[float, float]:
        """
        计算信号强度和建议的仓位大小
        模仿train_sac.py中的处理方式，保持连续性和直接关系
        """
        if action_id == HOLD:
            # 持有，仓位维持不变，信号强度很低
            # 返回一个微小的信号强度和仓位值而非零值，以便在边界情况下有连续性过渡
            return 0.05, 0.01
            
        # 直接使用原始连续动作值作为信号强度计算基础
        # 这与BtcTradingEnv中计算delta_usd = act * risk_capital * risk_fraction_per_trade的逻辑一致
        if action_id == BUY:
            # 买入信号，值越大越强
            # 根据一个平滑的非线性映射函数，确保即使是接近阈值的值也能得到合理的强度
            base_strength = (normalized_value - self.buy_threshold) / (self.max_action_value - self.buy_threshold)
            # 使用sigmoid变种以获得更平滑的过渡，同时保持连续性
            signal_strength = min(0.99, max(0.1, 0.5 + 0.5 * np.tanh(2 * base_strength)))
        else:  # SELL
            # 卖出信号，负值越大(绝对值越大)越强
            base_strength = (self.sell_threshold - normalized_value) / (self.sell_threshold - self.min_action_value)
            signal_strength = min(0.99, max(0.1, 0.5 + 0.5 * np.tanh(2 * base_strength)))
        
        # 计算仓位大小 - 与训练环境中的计算方式类似
        # 当连续动作值接近边界时，仓位应该更大
        position_size = abs(normalized_value) * self.position_scale
        
        # 为避免过于剧烈的仓位变化，将仓位大小映射到合理范围
        # 这里的参数可以根据实际风险偏好进行调整
        position_size = min(0.95, position_size * 0.8 + 0.1)
        
        return signal_strength, position_size
