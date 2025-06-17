"""
模型预测模块
负责从加载的模型中获取预测结果，并处理结果以供交易决策使用
"""
import os
import numpy as np
import pandas as pd
import logging
import yaml
import json
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import time

class PredictionProcessor:
    """
    预测处理类
    处理模型预测结果，计算置信度，执行集成决策
    """
    
    def __init__(self, config_path: str = None):
        """
        初始化预测处理器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认路径
        """
        self.logger = logging.getLogger('PredictionProcessor')
        
        # 如果未提供配置路径，则使用绝对路径找到配置文件
        if config_path is None:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(root_dir, "config", "model_config.yaml")
            
        self._load_config(config_path)
        
    def _load_config(self, config_path: str) -> None:
        """
        加载模型配置
        
        Args:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            prediction_config = config.get('model', {}).get('prediction', {})
            self.confidence_threshold = prediction_config.get('confidence_threshold', 0.65)
            self.holding_period = prediction_config.get('holding_period', 12)  # 默认持仓时间(小时)
            self.cooldown_period = prediction_config.get('cooldown_period', 4)  # 交易冷却时间(小时)
            
            ensemble_config = config.get('model', {}).get('ensemble', {})
            self.ensemble_method = ensemble_config.get('method', 'weighted_voting')
            
            self.log_predictions = prediction_config.get('save_predictions', True)
            self.prediction_log_path = prediction_config.get('prediction_log_path', '../logs/predictions')
            
            # 确保日志目录存在
            if self.log_predictions and not os.path.exists(self.prediction_log_path):
                os.makedirs(self.prediction_log_path, exist_ok=True)
            
            self.logger.info("成功加载预测配置")
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            raise
    
    def process_single_prediction(self, 
                               observation: np.ndarray, 
                               model,
                               threshold: float = None) -> Dict[str, Any]:
        """
        处理单个模型的预测结果
        
        Args:
            observation: 模型输入观测值
            model: 模型对象
            threshold: 置信度阈值，如果为None则使用配置中的值
            
        Returns:
            Dict[str, Any]: 处理后的预测结果
        """
        if threshold is None:
            threshold = self.confidence_threshold
        
        try:
            # 获取模型动作和值估计
            action, _states = model.predict(observation, deterministic=False)
            
            # 获取动作概率分布
            if hasattr(model, 'policy') and hasattr(model.policy, 'get_distribution'):
                dist = model.policy.get_distribution(observation)
                action_probs = dist.distribution.probs.cpu().numpy()
                
                # 获取最高概率和对应的动作
                max_prob = np.max(action_probs)
                max_action = np.argmax(action_probs)
                
                # 计算置信度 - 最高概率与次高概率之差
                sorted_probs = np.sort(action_probs)
                if len(sorted_probs) > 1:
                    confidence = sorted_probs[-1] - sorted_probs[-2]
                else:
                    confidence = sorted_probs[-1]
            else:
                # 如果无法获取概率分布，使用近似方法
                value, _ = model.policy.evaluate_actions(
                    model.policy.obs_to_tensor(observation)[0],
                    model.policy.obs_to_tensor(np.array([action]))[0]
                )
                max_prob = float(value.cpu().numpy())
                max_action = action
                confidence = 0.5  # 默认置信度
            
            # 置信度是否超过阈值
            is_confident = confidence > threshold
            
            return {
                "action": int(action),
                "action_value": float(max_prob),
                "confidence": float(confidence),
                "is_confident": bool(is_confident)
            }
        except Exception as e:
            self.logger.error(f"处理预测结果失败: {e}")
            return {
                "action": 1,  # 默认动作 (HOLD)
                "confidence": 0.0,
                "is_confident": False
            }
    
    def process(self, prediction: Dict[str, Any], 
             current_position: int = 0, 
             volatility: float = 0.0) -> Tuple[Dict[int, float], float, float]:
        """
        处理预测结果，生成行动概率、仓位大小和信心度
        
        Args:
            prediction: 预测结果字典，包含action、confidence等
            current_position: 当前持仓状态，默认为0（未持仓）
            volatility: 市场波动率，用于调整仓位大小
            
        Returns:
            Tuple[Dict[int, float], float, float]: 
                - 行动概率字典 {0: prob_sell, 1: prob_hold, 2: prob_buy}
                - 建议仓位大小 (0-1之间的浮点数)
                - 信心度 (0-1之间的浮点数)
        """
        try:
            self.logger.info(f"开始处理预测结果: {prediction}")
            
            # 获取预测信息
            action = prediction.get("action", 1)  # 默认为HOLD
            
            # 确保action是有效的整数值(0,1,2)
            if not isinstance(action, int) or action not in [0, 1, 2]:
                self.logger.warning(f"检测到无效动作值: {action}，类型: {type(action)}，将调整为HOLD(1)")
                action = 1
                prediction["action"] = action
                
            # 确保概率分布存在且完整
            if "probabilities" not in prediction or not prediction["probabilities"]:
                self.logger.warning("缺少概率分布，创建默认分布")
                if action == 0:
                    prediction["probabilities"] = {"0": 0.7, "1": 0.2, "2": 0.1}
                elif action == 2:
                    prediction["probabilities"] = {"0": 0.1, "1": 0.2, "2": 0.7}
                else:
                    prediction["probabilities"] = {"0": 0.2, "1": 0.6, "2": 0.2}
            else:
                # 确保所有键都存在且值有效
                for key in ["0", "1", "2"]:
                    if key not in prediction["probabilities"]:
                        self.logger.warning(f"概率分布缺少键 {key}，添加默认值")
                        prediction["probabilities"][key] = 0.1
                        
                # 验证概率总和是否为1
                total_prob = sum(float(v) for v in prediction["probabilities"].values())
                if abs(total_prob - 1.0) > 0.01:  # 允许小误差
                    self.logger.warning(f"概率总和为 {total_prob}，需要归一化")
                    # 归一化
                    for k in prediction["probabilities"]:
                        prediction["probabilities"][k] = float(prediction["probabilities"][k]) / total_prob
            
            confidence = prediction.get("confidence", 0.0)
            is_confident = prediction.get("is_confident", False)
            
            # 直接使用模型返回的概率分布，如果存在的话
            if "probabilities" in prediction and isinstance(prediction["probabilities"], dict):
                # 检查概率值是否有效
                has_valid_probs = False
                prob_sum = 0.0
                
                # 首先检查是否包含所有必要的动作键(0,1,2)
                required_keys = {"0", "1", "2"}
                has_all_keys = all(str(k) in prediction["probabilities"] for k in required_keys)
                if not has_all_keys:
                    self.logger.warning(f"模型返回的概率分布缺少必要的动作键，当前键: {list(prediction['probabilities'].keys())}")
                
                # 从字符串键转换为整数键，同时验证值的有效性
                temp_action_probas = {}
                for k, v in prediction["probabilities"].items():
                    try:
                        action_key = int(k)
                        prob_value = float(v)
                        # 确保概率在有效范围内
                        if 0 <= prob_value <= 1.0:
                            temp_action_probas[action_key] = prob_value
                            prob_sum += prob_value
                        else:
                            self.logger.warning(f"概率值超出范围[0,1]: key={k}, value={v}")
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"无法转换概率键值: key={k}, value={v}, error={str(e)}")
                
                # 确保所有必要的动作键都存在
                for key in [0, 1, 2]:
                    if key not in temp_action_probas:
                        self.logger.warning(f"缺少动作{key}的概率，添加默认值0.1")
                        temp_action_probas[key] = 0.1
                        prob_sum += 0.1
                
                # 检查是否获取到有效的概率分布
                if len(temp_action_probas) > 0 and prob_sum > 0:
                    action_probas = temp_action_probas
                    
                    # 如果需要，归一化概率
                    if abs(prob_sum - 1.0) > 0.01:  # 允许1%的误差
                        self.logger.info(f"归一化概率分布，当前总和: {prob_sum}")
                        for k in action_probas:
                            action_probas[k] /= prob_sum
                            
                    has_valid_probs = True
                    self.logger.info(f"使用模型返回的有效概率分布: {action_probas}")
                    
                if not has_valid_probs:
                    self.logger.warning(f"模型返回的概率分布无效: {prediction['probabilities']}")
                    # 使用基于动作和置信度的概率分布
                    action_probas = self._create_action_probabilities(action, confidence)
            else:
                self.logger.info("预测结果中没有概率分布，创建基于动作的概率分布")
                # 使用基于动作和置信度的概率分布
                action_probas = self._create_action_probabilities(action, confidence)
            
            # 计算仓位大小
            position_size = 0.0
            if is_confident and action == 2:  # 只有在BUY动作时才设置正向仓位
                if "probabilities" in prediction:
                    # 如果有 BUY 的概率，用它作为仓位大小的基础
                    buy_prob = float(prediction["probabilities"].get("2", 0))
                    if buy_prob > self.confidence_threshold:
                        position_size = buy_prob * 0.5  # 将概率转换为仓位大小，最大0.5
                else:
                    # 否则使用置信度计算
                    position_size = confidence * 0.5  # 最大仓位为0.5
            elif is_confident and action == 0:  # SELL动作时设置负向仓位（做空）
                if "probabilities" in prediction:
                    sell_prob = float(prediction["probabilities"].get("0", 0))
                    if sell_prob > self.confidence_threshold:
                        position_size = -sell_prob * 0.5  # 负值表示做空
                else:
                    position_size = -confidence * 0.5
                    
            # 持有动作(action=1)不产生新仓位，保持position_size=0
            
            # 解释动作（用于日志记录）
            action_info = self.interpret_action(action, current_position)
            
            # 记录预测结果
            if self.log_predictions:
                log_data = {
                    "timestamp": datetime.now().isoformat(),
                    "prediction": prediction,
                    "action_info": action_info,
                    "position_size": position_size,
                    "confidence": confidence,
                    "is_confident": is_confident,
                    "action_probas": action_probas
                }
                self._log_prediction(log_data)
            
            return action_probas, position_size, confidence
            
        except Exception as e:
            self.logger.error(f"处理预测结果失败: {e}")
            # 出错时返回默认值
            return {0: 0.33, 1: 0.34, 2: 0.33}, 0.0, 0.0
            
    def _create_action_probabilities(self, action, confidence):
        """
        根据动作和置信度创建动作概率分布
        
        Args:
            action: 预测动作
            confidence: 置信度
            
        Returns:
            Dict[int, float]: 动作概率字典
        """
        # 如果置信度很低，使用接近均匀的分布
        if confidence < 0.1:
            action_probas = {
                0: 0.30 + (action == 0) * 0.1,
                1: 0.35 + (action == 1) * 0.1,
                2: 0.30 + (action == 2) * 0.1
            }
        else:
            # 正常情况下，根据置信度创建分布
            main_prob = min(0.95, 0.6 + 0.35 * confidence)  # 最高0.95
            remaining = 1.0 - main_prob
            
            # 其他动作的概率分布 - 不是完全均匀
            if action == 0:  # SELL
                action_probas = {0: main_prob, 1: remaining * 0.7, 2: remaining * 0.3}
            elif action == 1:  # HOLD
                action_probas = {0: remaining * 0.45, 1: main_prob, 2: remaining * 0.55}
            else:  # BUY
                action_probas = {0: remaining * 0.3, 1: remaining * 0.7, 2: main_prob}
            
            # 归一化确保总和为1
            total = sum(action_probas.values())
            for k in action_probas:
                action_probas[k] /= total
                
        self.logger.info(f"创建基于动作{action}和置信度{confidence}的概率分布: {action_probas}")
        return action_probas
    
    def ensemble_predictions(self, model_predictions: Dict[str, Dict], method: str = None) -> Dict[str, Any]:
        """
        集成多个模型的预测结果
        
        Args:
            model_predictions: 模型预测结果字典 {model_name: {"action": action, "confidence": confidence, ...}}
            method: 集成方法，可选值有 "voting", "weighted_voting", "average" 等，默认为 None
            
        Returns:
            Dict[str, Any]: 集成后的预测结果
        """
        if not model_predictions:
            self.logger.warning("没有可用模型进行集成预测")
            return {
                "action": 1,  # 默认动作（持仓不变）
                "confidence": 0.0,
                "is_confident": False,
                "individual_predictions": {}
            }
        
        # 获取每个模型的预测结果
        all_actions = []
        all_weights = []
        model_confidences = []
        
        for model_name, prediction in model_predictions.items():
            action = prediction.get("action", 1)
            confidence = prediction.get("confidence", 0.0)
            weight = prediction.get("weight", 1.0)  # 默认权重为1.0
            
            all_actions.append(action)
            all_weights.append(weight)
            model_confidences.append(confidence)
        
        # 根据集成方法处理结果
        if method == "voting":
            # 简单多数投票
            action_counts = {}
            for action in all_actions:
                action_counts[action] = action_counts.get(action, 0) + 1
            
            # 找出票数最多的动作
            max_votes = 0
            final_action = 1  # 默认动作
            
            for action, count in action_counts.items():
                if count > max_votes:
                    max_votes = count
                    final_action = action
            
            # 计算置信度 - 票数占比
            confidence = max_votes / len(all_actions)
            
        elif method == "weighted_voting":
            # 加权投票
            action_weights = {}
            total_weight = sum(all_weights)
            
            for i, action in enumerate(all_actions):
                action_weights[action] = action_weights.get(action, 0) + all_weights[i]
            
            # 找出权重最高的动作
            max_weight = 0
            final_action = 1  # 默认动作
            
            for action, weight in action_weights.items():
                if weight > max_weight:
                    max_weight = weight
                    final_action = action
            
            # 计算置信度 - 权重占比
            confidence = max_weight / total_weight if total_weight > 0 else 0
            
        elif method == "average":
            # 平均预测值
            action_values = {}
            action_counts = {}
            
            for model_name, prediction in model_predictions.items():
                action = prediction["action"]
                value = prediction["action_value"]
                
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
            
            # 所有平均值的最大值与次大值的差作为置信度
            sorted_values = sorted(avg_values.values())
            if len(sorted_values) > 1:
                confidence = (sorted_values[-1] - sorted_values[-2]) / sorted_values[-1]
            else:
                confidence = 0.5  # 默认置信度
        
        else:
            self.logger.warning(f"未知的集成方法: {method}，使用第一个模型的预测")
            first_model = next(iter(model_predictions.values()))
            final_action = first_model["action"]
            confidence = first_model["confidence"]
        
        # 判断置信度是否足够
        is_confident = confidence > self.confidence_threshold
        
        return {
            "action": int(final_action),
            "confidence": float(confidence),
            "is_confident": bool(is_confident),
            "individual_predictions": model_predictions
        }
    
    def log_prediction_result(self, prediction: Dict[str, Any], 
                           symbol: str, timestamp: Optional[datetime] = None) -> None:
        """
        记录预测结果
        
        Args:
            prediction: 预测结果
            symbol: 交易对符号
            timestamp: 时间戳，如果为None则使用当前时间
        """
        if not self.log_predictions:
            return
            
        try:
            if timestamp is None:
                timestamp = datetime.now()
                
            # 创建日志文件名
            date_str = timestamp.strftime('%Y%m%d')
            log_filename = os.path.join(self.prediction_log_path, f"predictions_{symbol}_{date_str}.json")
            
            # 准备记录数据
            log_entry = {
                "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                "symbol": symbol,
                "action": prediction["action"],
                "confidence": prediction["confidence"],
                "is_confident": prediction["is_confident"]
            }
            
            # 如果有单个模型的预测结果，也记录
            if "individual_predictions" in prediction:
                ind_preds = {}
                for model_name, pred in prediction["individual_predictions"].items():
                    ind_preds[model_name] = {
                        "action": pred["action"],
                        "confidence": pred["confidence"],
                        "is_confident": pred["is_confident"]
                    }
                log_entry["individual_predictions"] = ind_preds
            
            # 追加到日志文件
            log_entries = []
            if os.path.exists(log_filename):
                with open(log_filename, 'r') as f:
                    try:
                        log_entries = json.load(f)
                        if not isinstance(log_entries, list):
                            log_entries = []
                    except:
                        log_entries = []
            
            log_entries.append(log_entry)
            
            with open(log_filename, 'w') as f:
                json.dump(log_entries, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"记录预测结果失败: {e}")
    
    def _log_prediction(self, log_data: Dict[str, Any]) -> None:
        """
        内部方法，记录预测结果到文件
        """
        try:
            # 确保日志目录存在
            if not os.path.exists(self.prediction_log_path):
                os.makedirs(self.prediction_log_path, exist_ok=True)
            
            # 创建日志文件名 (按日期)
            date_str = datetime.now().strftime('%Y%m%d')
            log_filename = os.path.join(self.prediction_log_path, f"prediction_log_{date_str}.json")
            
            # 读取现有日志
            existing_logs = []
            if os.path.exists(log_filename):
                try:
                    with open(log_filename, 'r') as f:
                        existing_logs = json.load(f)
                except Exception:
                    existing_logs = []
            
            # 添加新日志
            existing_logs.append(log_data)
            
            # 写入文件
            with open(log_filename, 'w') as f:
                json.dump(existing_logs, f, default=str)
                
        except Exception as e:
            self.logger.error(f"记录预测日志失败: {e}")
    
    def interpret_action(self, action: int, current_position: float = 0) -> Dict[str, Any]:
        """
        解释模型预测的动作
        
        Args:
            action: 预测的动作ID (0=卖出, 1=持有, 2=买入)
            current_position: 当前持仓数量
            
        Returns:
            Dict[str, Any]: 包含动作解释和建议的字典
        """
        action_map = {
            0: "SELL",
            1: "HOLD",
            2: "BUY"
        }
        
        action_name = action_map.get(action, "UNKNOWN")
        description = ""
        recommendation = ""
        
        if action == 0:  # 卖出
            description = "模型预测价格将下跌"
            if current_position > 0:
                recommendation = "建议减仓或平仓"
            elif current_position < 0:
                recommendation = "建议继续持有空仓或加空"
            else:
                recommendation = "建议开空仓"
        elif action == 1:  # 持有
            description = "模型预测价格将横盘震荡"
            if current_position != 0:
                recommendation = "建议继续持有现有仓位"
            else:
                recommendation = "建议观望不入场"
        elif action == 2:  # 买入
            description = "模型预测价格将上涨"
            if current_position > 0:
                recommendation = "建议继续持有多仓或加多"
            elif current_position < 0:
                recommendation = "建议减仓或平仓"
            else:
                recommendation = "建议开多仓"
                
        return {
            "action_id": action,
            "action_name": action_name,
            "description": description,
            "recommendation": recommendation,
            "current_position": current_position
        }
    
    def calculate_position_size(self, prediction: Dict[str, Any], 
                             volatility: float, 
                             base_position_size: float) -> float:
        """
        基于预测置信度和市场波动率计算仓位大小
        
        Args:
            prediction: 预测结果
            volatility: 市场波动率，以小数表示
            base_position_size: 基础仓位大小，以小数表示
            
        Returns:
            float: 计算后的仓位大小，以小数表示
        """
        confidence = prediction.get("confidence", 0)
        
        # 检查置信度是否足够
        if not prediction.get("is_confident", False):
            return 0.0
        
        # 基于波动率调整仓位 - 高波动率降低仓位
        volatility_factor = 1.0
        if volatility > 0:
            # 当波动率超过5%时开始降低仓位
            if volatility > 0.05:
                volatility_factor = 0.05 / volatility
                # 确保不低于0.2
                volatility_factor = max(0.2, volatility_factor)
        
        # 基于置信度调整仓位
        confidence_factor = 0.5 + 0.5 * confidence  # 置信度为0.5时不变，更高时增加
        
        # 计算最终仓位大小
        position_size = base_position_size * confidence_factor * volatility_factor
        
        # 确保仓位在合理范围内
        position_size = min(1.0, max(0.0, position_size))
        
        return position_size
    
    def process(self, prediction: Dict[str, Any], current_position: int = 0, 
               volatility: float = 0.02, base_position_size: float = 0.5) -> Tuple[Dict[int, float], float, float]:
        """
        处理集成模型的预测结果
        
        Args:
            prediction: 集成模型的预测结果
            current_position: 当前持仓状态 (-1:空仓, 0:不持仓, 1:多仓)
            volatility: 市场波动率
            base_position_size: 基础仓位大小
            
        Returns:
            Tuple:
                - Dict[int, float]: 动作概率字典，键为动作ID，值为概率
                - float: 建议的仓位大小
                - float: 预测置信度
        """
        try:
            # 默认动作概率（全部平均）
            default_probas = {0: 0.33, 1: 0.34, 2: 0.33}
            
            # 获取预测动作和置信度
            action = prediction.get("action", 1)  # 默认为HOLD
            confidence = prediction.get("confidence", 0.0)
            is_confident = prediction.get("is_confident", False)
            
            # 解释动作
            action_info = self.interpret_action(action, current_position)
            
            # 计算仓位大小
            position_size = self.calculate_position_size(
                prediction, volatility, base_position_size
            ) if is_confident else 0.0
            
            # 根据预测构建动作概率字典
            action_probas = default_probas.copy()
            
            # 更新预测动作的概率
            # 这里简单地将预测动作的概率设为0.6+，其他平分剩余概率
            if confidence > 0:
                main_prob = 0.6 + 0.3 * confidence  # 最高0.9
                remaining = 1.0 - main_prob
                
                # 设置主要动作的概率
                action_probas[action] = main_prob
                
                # 其他动作平分剩余概率
                other_actions = [a for a in action_probas.keys() if a != action]
                for other in other_actions:
                    action_probas[other] = remaining / len(other_actions)
            
            # 记录预测结果
            if self.log_predictions:
                log_data = {
                    "timestamp": datetime.now().isoformat(),
                    "prediction": prediction,
                    "action_info": action_info,
                    "position_size": position_size,
                    "confidence": confidence,
                    "is_confident": is_confident,
                    "action_probas": action_probas
                }
                self._log_prediction(log_data)
            
            return action_probas, position_size, confidence
            
        except Exception as e:
            self.logger.error(f"处理预测结果失败: {e}")
            # 出错时返回默认值
            return {0: 0.33, 1: 0.34, 2: 0.33}, 0.0, 0.0
