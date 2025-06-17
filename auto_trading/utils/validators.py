import re
import json
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import logging
import os
from typing import Dict, List, Any, Tuple, Optional, Union

logger = logging.getLogger(__name__)

class Validators:
    """
    数据验证工具类，用于验证各种数据的有效性
    包括配置验证、API响应验证、交易数据验证等
    """
    
    @staticmethod
    def validate_config_file(file_path: str, required_fields: List[str] = None) -> Tuple[bool, Dict]:
        """
        验证YAML配置文件的有效性
        
        Args:
            file_path: 配置文件路径
            required_fields: 必需的字段列表
            
        Returns:
            Tuple[bool, Dict]: (是否有效, 配置数据)
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"配置文件不存在: {file_path}")
                return False, {}
                
            with open(file_path, 'r') as file:
                config = yaml.safe_load(file)
                
            if config is None:
                logger.error(f"配置文件为空: {file_path}")
                return False, {}
                
            if required_fields:
                missing_fields = [field for field in required_fields if field not in config]
                if missing_fields:
                    logger.error(f"配置文件缺少必需字段: {missing_fields}")
                    return False, config
                    
            return True, config
            
        except yaml.YAMLError as e:
            logger.error(f"配置文件格式错误: {e}")
            return False, {}
        except Exception as e:
            logger.error(f"验证配置文件失败: {e}")
            return False, {}
            
    @staticmethod
    def validate_api_response(response: Dict, required_fields: List[str] = None) -> bool:
        """
        验证API响应的有效性
        
        Args:
            response: API响应数据
            required_fields: 必需的字段列表
            
        Returns:
            bool: 响应是否有效
        """
        try:
            # 检查是否为空
            if not response:
                logger.error("API响应为空")
                return False
                
            # 检查是否为错误响应
            if 'error' in response or 'code' in response and response.get('code') not in [200, '200', 0, '0']:
                error_code = response.get('code', 'unknown')
                error_msg = response.get('msg', response.get('message', response.get('error', 'Unknown error')))
                logger.error(f"API错误: [{error_code}] {error_msg}")
                return False
                
            # 检查必需字段
            if required_fields:
                missing_fields = [field for field in required_fields if field not in response]
                if missing_fields:
                    logger.error(f"API响应缺少必需字段: {missing_fields}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"验证API响应失败: {e}")
            return False
            
    @staticmethod
    def validate_kline_data(data: List[List], min_length: int = 10) -> Tuple[bool, pd.DataFrame]:
        """
        验证K线数据的有效性并转换为DataFrame
        
        Args:
            data: K线数据列表
            min_length: 最小数据长度
            
        Returns:
            Tuple[bool, pd.DataFrame]: (是否有效, K线数据DataFrame)
        """
        try:
            # 检查是否为空
            if not data:
                logger.error("K线数据为空")
                return False, pd.DataFrame()
                
            # 检查数据长度
            if len(data) < min_length:
                logger.warning(f"K线数据过少: {len(data)} < {min_length}")
                # 针对少量数据，依然处理但发出警告
                
            # 定义K线数据的列名
            columns = [
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades_count', 
                'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
            ]
            
            # 转换为DataFrame
            df = pd.DataFrame(data, columns=columns)
            
            # 检查数据类型和转换
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            # 转换价格和数量数据为浮点型
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 
                              'taker_buy_base_volume', 'taker_buy_quote_volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # 检查是否有空值
            null_counts = df[numeric_columns].isna().sum()
            if null_counts.sum() > 0:
                logger.warning(f"K线数据包含空值: {null_counts[null_counts > 0].to_dict()}")
            
            # 检查价格合理性
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if df[col].min() <= 0:
                    logger.error(f"K线包含无效价格: {col} 最小值为 {df[col].min()}")
                    return False, df
                    
            # 检查数据是否按时间排序
            if not df['open_time'].is_monotonic_increasing:
                logger.warning("K线数据未按时间排序，正在排序...")
                df = df.sort_values('open_time')
            
            # 检查时间间隔是否一致
            time_diffs = df['open_time'].diff().dropna()
            if time_diffs.nunique() > 1:
                logger.warning(f"K线时间间隔不一致: {time_diffs.value_counts().to_dict()}")
                
            # 高低价检查
            if not (df['high'] >= df['low']).all():
                invalid_rows = df[~(df['high'] >= df['low'])].index.tolist()
                logger.error(f"存在最高价低于最低价的K线: 行索引 {invalid_rows}")
                return False, df
                
            return True, df
            
        except Exception as e:
            logger.error(f"验证K线数据失败: {e}")
            return False, pd.DataFrame()
            
    @staticmethod
    def validate_order_params(symbol: str, side: str, order_type: str, 
                            quantity: float = None, price: float = None,
                            time_in_force: str = None) -> bool:
        """
        验证订单参数的有效性
        
        Args:
            symbol: 交易对
            side: 订单方向 (BUY/SELL)
            order_type: 订单类型 (LIMIT/MARKET/STOP_LOSS/STOP_LOSS_LIMIT/TAKE_PROFIT/TAKE_PROFIT_LIMIT)
            quantity: 数量
            price: 价格
            time_in_force: 订单有效期 (GTC/IOC/FOK)
            
        Returns:
            bool: 参数是否有效
        """
        try:
            # 验证交易对
            if not symbol or not isinstance(symbol, str):
                logger.error(f"无效的交易对: {symbol}")
                return False
                
            # 检查订单方向
            valid_sides = ['BUY', 'SELL']
            if side not in valid_sides:
                logger.error(f"无效的订单方向: {side}，有效值: {valid_sides}")
                return False
                
            # 检查订单类型
            valid_types = ['LIMIT', 'MARKET', 'STOP_LOSS', 'STOP_LOSS_LIMIT', 'TAKE_PROFIT', 'TAKE_PROFIT_LIMIT']
            if order_type not in valid_types:
                logger.error(f"无效的订单类型: {order_type}，有效值: {valid_types}")
                return False
                
            # 检查数量
            if quantity is not None:
                try:
                    quantity = float(quantity)
                    if quantity <= 0:
                        logger.error(f"无效的订单数量: {quantity}")
                        return False
                except ValueError:
                    logger.error(f"订单数量无法转换为浮点数: {quantity}")
                    return False
                    
            # 检查价格
            if order_type in ['LIMIT', 'STOP_LOSS_LIMIT', 'TAKE_PROFIT_LIMIT']:
                if price is None:
                    logger.error(f"限价单需要指定价格")
                    return False
                    
                try:
                    price = float(price)
                    if price <= 0:
                        logger.error(f"无效的订单价格: {price}")
                        return False
                except ValueError:
                    logger.error(f"订单价格无法转换为浮点数: {price}")
                    return False
                    
            # 检查订单有效期
            if order_type in ['LIMIT', 'STOP_LOSS_LIMIT', 'TAKE_PROFIT_LIMIT']:
                valid_tif = ['GTC', 'IOC', 'FOK']
                if time_in_force is not None and time_in_force not in valid_tif:
                    logger.error(f"无效的订单有效期: {time_in_force}，有效值: {valid_tif}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"验证订单参数失败: {e}")
            return False
            
    @staticmethod
    def validate_model_input(features: np.ndarray, expected_shape: Tuple = None) -> bool:
        """
        验证模型输入数据的有效性
        
        Args:
            features: 特征数据
            expected_shape: 期望的数据形状
            
        Returns:
            bool: 输入是否有效
        """
        try:
            # 检查是否为空
            if features is None or (isinstance(features, np.ndarray) and features.size == 0):
                logger.error("模型输入为空")
                return False
                
            # 检查数据类型
            if not isinstance(features, np.ndarray):
                logger.error(f"模型输入类型错误: 期望numpy.ndarray，实际{type(features)}")
                return False
                
            # 检查数据形状
            if expected_shape is not None:
                # 检查维度数量
                if len(features.shape) != len(expected_shape):
                    logger.error(f"模型输入维度错误: 期望{len(expected_shape)}维，实际{len(features.shape)}维")
                    return False
                    
                # 检查每个维度的大小
                for i, (actual, expected) in enumerate(zip(features.shape, expected_shape)):
                    # 如果期望维度为None，表示该维度可以是任意大小
                    if expected is not None and actual != expected:
                        logger.error(f"模型输入第{i}维大小错误: 期望{expected}，实际{actual}")
                        return False
                        
            # 检查数据是否包含NaN或无穷大
            if np.isnan(features).any():
                logger.error("模型输入包含NaN值")
                return False
                
            if np.isinf(features).any():
                logger.error("模型输入包含无穷大值")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"验证模型输入失败: {e}")
            return False
            
    @staticmethod
    def validate_timestamp(timestamp: Union[int, float, str, datetime], 
                          min_time: datetime = None, 
                          max_time: datetime = None) -> Tuple[bool, datetime]:
        """
        验证时间戳的有效性并转换为datetime对象
        
        Args:
            timestamp: 时间戳（整数毫秒、浮点秒、字符串或datetime对象）
            min_time: 最小有效时间
            max_time: 最大有效时间
            
        Returns:
            Tuple[bool, datetime]: (是否有效, datetime对象)
        """
        try:
            dt = None
            
            # 转换不同类型的时间戳
            if isinstance(timestamp, datetime):
                dt = timestamp
            elif isinstance(timestamp, (int, float)):
                # 判断是秒级还是毫秒级时间戳
                if timestamp > 1e11:  # 毫秒级时间戳
                    dt = datetime.fromtimestamp(timestamp / 1000, timezone.utc)
                else:  # 秒级时间戳
                    dt = datetime.fromtimestamp(timestamp, timezone.utc)
            elif isinstance(timestamp, str):
                try:
                    # 尝试多种格式解析日期字符串
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y%m%d', '%Y%m%d %H:%M:%S']:
                        try:
                            dt = datetime.strptime(timestamp, fmt)
                            break
                        except ValueError:
                            continue
                    
                    if dt is None:
                        # 尝试ISO格式
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except ValueError:
                    logger.error(f"无法解析日期字符串: {timestamp}")
                    return False, None
            else:
                logger.error(f"不支持的时间戳类型: {type(timestamp)}")
                return False, None
                
            # 检查是否在有效范围内
            if min_time and dt < min_time:
                logger.error(f"时间戳早于最小有效时间: {dt} < {min_time}")
                return False, dt
                
            if max_time and dt > max_time:
                logger.error(f"时间戳晚于最大有效时间: {dt} > {max_time}")
                return False, dt
                
            return True, dt
            
        except Exception as e:
            logger.error(f"验证时间戳失败: {e}")
            return False, None
            
    @staticmethod
    def validate_numeric_range(value: Union[int, float], 
                              min_value: Union[int, float] = None, 
                              max_value: Union[int, float] = None,
                              allow_zero: bool = True,
                              allow_negative: bool = False) -> bool:
        """
        验证数值是否在指定范围内
        
        Args:
            value: 要验证的数值
            min_value: 最小有效值
            max_value: 最大有效值
            allow_zero: 是否允许为零
            allow_negative: 是否允许为负数
            
        Returns:
            bool: 数值是否有效
        """
        try:
            # 转换为数值类型
            try:
                num_value = float(value)
            except (ValueError, TypeError):
                logger.error(f"无法将{value}转换为数值")
                return False
                
            # 检查零值
            if not allow_zero and num_value == 0:
                logger.error(f"数值不允许为零: {value}")
                return False
                
            # 检查负值
            if not allow_negative and num_value < 0:
                logger.error(f"数值不允许为负: {value}")
                return False
                
            # 检查最小值
            if min_value is not None and num_value < min_value:
                logger.error(f"数值小于最小有效值: {num_value} < {min_value}")
                return False
                
            # 检查最大值
            if max_value is not None and num_value > max_value:
                logger.error(f"数值大于最大有效值: {num_value} > {max_value}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"验证数值范围失败: {e}")
            return False
