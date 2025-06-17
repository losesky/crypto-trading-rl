"""
特征工程模块
负责为交易系统创建高级特征和市场指标
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import yaml

class FeatureEngineer:
    """
    特征工程类
    创建用于交易决策的高级特征
    """
    
    def __init__(self, config_path: str = "../config/model_config.yaml"):
        """
        初始化特征工程器
        
        Args:
            config_path: 配置文件路径
        """
        self.logger = logging.getLogger('FeatureEngineer')
        self._load_config(config_path)
        
    def calculate_features(self, df: pd.DataFrame, external_data: Optional[Dict] = None) -> pd.DataFrame:
        """
        计算所有特征 - main.py中使用的接口方法
        
        Args:
            df: 原始K线数据DataFrame
            external_data: 可选的外部市场数据
            
        Returns:
            pd.DataFrame: 包含所有特征的DataFrame
        """
        return self.create_all_features(df, external_data)
        
    def _load_config(self, config_path: str) -> None:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            features_config = config.get('model', {}).get('features', {})
            self.lookback_window = features_config.get('lookback_window', 24)
            
            self.logger.info("成功加载特征工程配置")
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            raise
    
    def calculate_price_derivatives(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算价格导数特征，包括速度、加速度和动量
        
        Args:
            df: 价格数据DataFrame
            
        Returns:
            pd.DataFrame: 添加了价格导数特征的DataFrame
        """
        try:
            # 创建副本以避免修改原始数据
            result_df = df.copy()
            
            # 价格变化（速度）- 5个周期
            result_df['price_velocity_5'] = result_df['close'].pct_change(periods=5)
            
            # 价格变化的变化（加速度）- 5个周期
            result_df['price_acceleration_5'] = result_df['price_velocity_5'].diff(5)
            
            # 价格动量 - 不同周期
            result_df['price_momentum_12'] = result_df['close'] / result_df['close'].shift(12) - 1
            result_df['price_momentum_24'] = result_df['close'] / result_df['close'].shift(24) - 1
            
            # 价格波动率 - 标准差除以均值
            result_df['price_volatility_24'] = result_df['close'].rolling(24).std() / result_df['close'].rolling(24).mean()
            
            # 价格趋势强度 - 通过移动平均线偏离度衡量
            if 'sma_7' in result_df.columns and 'sma_25' in result_df.columns:
                result_df['trend_strength'] = (result_df['sma_7'] - result_df['sma_25']) / result_df['sma_25']
            
            # 价格范围比率 - 当前K线实体与整体范围的比例
            result_df['candle_body_ratio'] = abs(result_df['close'] - result_df['open']) / (result_df['high'] - result_df['low'])
            
            # 填充NaN值
            for col in ['price_velocity_5', 'price_acceleration_5', 'price_momentum_12', 
                       'price_momentum_24', 'price_volatility_24', 'trend_strength', 'candle_body_ratio']:
                if col in result_df.columns:
                    result_df[col] = result_df[col].replace([np.inf, -np.inf], np.nan)
                    result_df[col] = result_df[col].fillna(0)
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"计算价格导数特征失败: {e}")
            raise
    
    def calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算交易量相关特征
        
        Args:
            df: 价格和交易量数据DataFrame
            
        Returns:
            pd.DataFrame: 添加了交易量特征的DataFrame
        """
        try:
            # 创建副本以避免修改原始数据
            result_df = df.copy()
            
            # 交易量变化率
            result_df['volume_change_ratio'] = result_df['volume'].pct_change()
            
            # 相对交易量 - 当前交易量相对于N周期平均交易量
            result_df['relative_volume_10'] = result_df['volume'] / result_df['volume'].rolling(10).mean()
            result_df['relative_volume_24'] = result_df['volume'] / result_df['volume'].rolling(24).mean()
            
            # 交易量加权价格 (VWAP)
            result_df['vwap_24'] = (result_df['volume'] * result_df['close']).rolling(24).sum() / result_df['volume'].rolling(24).sum()
            
            # 买卖压力指标（基于买卖方交易量）
            result_df['buy_pressure'] = result_df['taker_buy_quote_asset_volume'] / result_df['quote_asset_volume']
            result_df['sell_pressure'] = 1 - result_df['buy_pressure']
            
            # 交易量振荡器 - 类似于价格振荡器
            result_df['volume_oscillator'] = (result_df['volume'].rolling(5).mean() / 
                                           result_df['volume'].rolling(20).mean() - 1) * 100
            
            # 交易量趋势 - 交易量短期均线和长期均线的比值
            result_df['volume_trend'] = (result_df['volume'].rolling(5).mean() / 
                                      result_df['volume'].rolling(20).mean())
            
            # 填充NaN值
            for col in ['volume_change_ratio', 'relative_volume_10', 'relative_volume_24', 
                       'vwap_24', 'buy_pressure', 'sell_pressure', 'volume_oscillator', 'volume_trend']:
                if col in result_df.columns:
                    result_df[col] = result_df[col].replace([np.inf, -np.inf], np.nan)
                    result_df[col] = result_df[col].fillna(0)
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"计算交易量特征失败: {e}")
            raise
    
    def calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算波动率相关特征
        
        Args:
            df: 价格数据DataFrame
            
        Returns:
            pd.DataFrame: 添加了波动率特征的DataFrame
        """
        try:
            # 创建副本以避免修改原始数据
            result_df = df.copy()
            
            # 高低价差波动率
            result_df['hl_volatility'] = (result_df['high'] - result_df['low']) / result_df['close']
            
            # 基于对数回报的波动率
            log_returns = np.log(result_df['close'] / result_df['close'].shift(1))
            result_df['return_volatility_12'] = log_returns.rolling(12).std()
            result_df['return_volatility_24'] = log_returns.rolling(24).std()
            
            # Garman-Klass波动率估计 (基于开高低收)
            result_df['garman_klass_vol'] = 0.5 * np.log(result_df['high'] / result_df['low'])**2 - (2*np.log(2)-1) * np.log(result_df['close'] / result_df['open'])**2
            
            # 波动率变化率
            if 'atr_14' in result_df.columns:
                result_df['atr_change'] = result_df['atr_14'].pct_change()
            
            # 相对真实波幅 (ATR/价格)
            if 'atr_14' in result_df.columns:
                result_df['relative_atr'] = result_df['atr_14'] / result_df['close']
            
            # 波动率趋势 (短期波动率/长期波动率)
            result_df['volatility_trend'] = result_df['return_volatility_12'] / result_df['return_volatility_24']
            
            # 填充NaN值
            for col in ['hl_volatility', 'return_volatility_12', 'return_volatility_24', 
                       'garman_klass_vol', 'atr_change', 'relative_atr', 'volatility_trend']:
                if col in result_df.columns:
                    result_df[col] = result_df[col].replace([np.inf, -np.inf], np.nan)
                    result_df[col] = result_df[col].fillna(0)
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"计算波动率特征失败: {e}")
            raise
    
    def calculate_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算价格模式特征
        
        Args:
            df: 价格数据DataFrame
            
        Returns:
            pd.DataFrame: 添加了模式特征的DataFrame
        """
        try:
            # 创建副本以避免修改原始数据
            result_df = df.copy()
            
            # 价格交叉特征 - 短期均线是否穿过长期均线
            if 'sma_7' in result_df.columns and 'sma_25' in result_df.columns:
                result_df['sma_crossover'] = ((result_df['sma_7'].shift(1) <= result_df['sma_25'].shift(1)) & 
                                           (result_df['sma_7'] > result_df['sma_25'])).astype(int)
                result_df['sma_crossunder'] = ((result_df['sma_7'].shift(1) >= result_df['sma_25'].shift(1)) & 
                                            (result_df['sma_7'] < result_df['sma_25'])).astype(int)
            
            # 布林带相对位置
            if all(col in result_df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
                # 价格相对于布林带的位置 (0-1之间，0表示在下轨，0.5表示在中轨，1表示在上轨)
                result_df['bb_position'] = (result_df['close'] - result_df['bb_lower']) / (result_df['bb_upper'] - result_df['bb_lower'])
                # 布林带宽度 (标准化)
                result_df['bb_width'] = (result_df['bb_upper'] - result_df['bb_lower']) / result_df['bb_middle']
            
            # RSI超买超卖
            if 'rsi_14' in result_df.columns:
                result_df['rsi_overbought'] = (result_df['rsi_14'] > 70).astype(int)
                result_df['rsi_oversold'] = (result_df['rsi_14'] < 30).astype(int)
            
            # 价格突破
            result_df['high_breakout'] = (result_df['high'] > result_df['high'].rolling(20).max().shift(1)).astype(int)
            result_df['low_breakdown'] = (result_df['low'] < result_df['low'].rolling(20).min().shift(1)).astype(int)
            
            # 蜡烛图模式识别 (简化版)
            # 十字星 (Doji) - 开盘和收盘价非常接近
            doji_threshold = 0.001
            result_df['doji'] = (abs(result_df['close'] - result_df['open']) / result_df['open'] < doji_threshold).astype(int)
            
            # 捉腰带线 (Hammer) - 下影线长，上影线短，实体小
            body = abs(result_df['close'] - result_df['open'])
            lower_wick = np.minimum(result_df['open'], result_df['close']) - result_df['low']
            upper_wick = result_df['high'] - np.maximum(result_df['open'], result_df['close'])
            result_df['hammer'] = ((lower_wick > 2 * body) & (upper_wick < 0.2 * body)).astype(int)
            
            # 吊颈线 (Hanging Man) - 与锤子相似，但在上升趋势末端
            price_trend = result_df['close'].rolling(5).mean() > result_df['close'].rolling(20).mean()
            result_df['hanging_man'] = (result_df['hammer'] & price_trend).astype(int)
            
            # 填充NaN值
            pattern_cols = ['sma_crossover', 'sma_crossunder', 'bb_position', 'bb_width',
                          'rsi_overbought', 'rsi_oversold', 'high_breakout', 'low_breakdown',
                          'doji', 'hammer', 'hanging_man']
            for col in pattern_cols:
                if col in result_df.columns:
                    result_df[col] = result_df[col].fillna(0)
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"计算价格模式特征失败: {e}")
            raise
    
    def calculate_cycles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算周期性特征
        
        Args:
            df: 价格数据DataFrame
            
        Returns:
            pd.DataFrame: 添加了周期特征的DataFrame
        """
        try:
            # 创建副本以避免修改原始数据
            result_df = df.copy()
            
            # 添加时间特征（基于索引中的时间戳）
            if result_df.index.name == 'open_time':
                # 小时特征
                result_df['hour_sin'] = np.sin(2 * np.pi * result_df.index.hour / 24)
                result_df['hour_cos'] = np.cos(2 * np.pi * result_df.index.hour / 24)
                
                # 日内特征
                result_df['day_sin'] = np.sin(2 * np.pi * result_df.index.dayofweek / 7)
                result_df['day_cos'] = np.cos(2 * np.pi * result_df.index.dayofweek / 7)
                
                # 月内特征
                result_df['month_sin'] = np.sin(2 * np.pi * result_df.index.day / 30)
                result_df['month_cos'] = np.cos(2 * np.pi * result_df.index.day / 30)
            
            # 价格周期性分析 - 使用简单的快速傅里叶变换方法
            # 注意：此处为简化实现，实际应用中可以使用更复杂的方法
            if len(result_df) >= 32:  # 确保有足够的数据点
                try:
                    # 获取最近32个点进行FFT
                    price_series = result_df['close'].iloc[-32:].values
                    price_series = price_series - np.mean(price_series)  # 去均值
                    fft_result = np.abs(np.fft.fft(price_series))
                    
                    # 找出主要频率
                    main_freq_idx = np.argsort(fft_result[1:16])[-3:] + 1  # 排除零频和高频噪声
                    
                    # 添加三个主要周期的强度
                    for i, idx in enumerate(main_freq_idx):
                        result_df.loc[result_df.index[-1], f'cycle_strength_{i+1}'] = fft_result[idx]
                    
                    # 填充其他行
                    for i in range(1, 4):
                        result_df[f'cycle_strength_{i}'] = result_df[f'cycle_strength_{i}'].fillna(method='ffill')
                except Exception as e:
                    self.logger.warning(f"计算FFT特征失败: {e}")
                    for i in range(1, 4):
                        result_df[f'cycle_strength_{i}'] = 0
                
            return result_df
            
        except Exception as e:
            self.logger.error(f"计算周期特征失败: {e}")
            raise
    
    def calculate_sentiment_features(self, df: pd.DataFrame, external_data: Optional[Dict] = None) -> pd.DataFrame:
        """
        计算情绪特征，可以结合外部市场数据
        
        Args:
            df: 基础数据DataFrame
            external_data: 可选的外部市场数据
            
        Returns:
            pd.DataFrame: 添加了情绪特征的DataFrame
        """
        try:
            # 创建副本以避免修改原始数据
            result_df = df.copy()
            
            # 基于内部数据计算情绪指标
            
            # 1. 多空比 - 基于布林带位置
            if 'bb_position' in result_df.columns:
                result_df['bull_bear_ratio'] = result_df['bb_position'].rolling(14).mean()
            
            # 2. 价格强度 - 基于上涨/下跌K线的占比
            price_change = result_df['close'] - result_df['open']
            result_df['bull_candles_ratio'] = (price_change > 0).astype(int).rolling(14).mean()
            
            # 3. 极端情绪指标 - 基于价格与布林带的关系
            if all(col in result_df.columns for col in ['bb_upper', 'bb_lower']):
                # 价格高于上轨或低于下轨的程度
                result_df['extreme_sentiment'] = np.where(
                    result_df['close'] > result_df['bb_upper'],
                    (result_df['close'] - result_df['bb_upper']) / result_df['bb_upper'],
                    np.where(
                        result_df['close'] < result_df['bb_lower'],
                        (result_df['close'] - result_df['bb_lower']) / result_df['bb_lower'],
                        0
                    )
                )
            
            # 4. 动量情绪 - 基于多个时间窗口的价格变化
            result_df['momentum_sentiment'] = (
                0.5 * (result_df['close'] / result_df['close'].shift(1) - 1) +
                0.3 * (result_df['close'] / result_df['close'].shift(3) - 1) +
                0.2 * (result_df['close'] / result_df['close'].shift(7) - 1)
            )
            
            # 5. 波动率情绪 - 波动率变化对未来价格的指示
            if 'return_volatility_24' in result_df.columns:
                result_df['volatility_sentiment'] = result_df['return_volatility_24'].pct_change(3)
            
            # 结合外部市场数据（如果提供）
            if external_data:
                # 示例：添加外部恐慌指数
                if 'fear_greed_index' in external_data:
                    fear_greed = external_data['fear_greed_index']
                    # 假设fear_greed是一个时间序列，需要对齐索引
                    if isinstance(fear_greed, pd.Series):
                        # 重新索引到当前DataFrame的时间戳
                        fear_greed = fear_greed.reindex(result_df.index, method='ffill')
                        result_df['fear_greed_index'] = fear_greed
            
            # 填充NaN值
            sentiment_cols = ['bull_bear_ratio', 'bull_candles_ratio', 'extreme_sentiment',
                            'momentum_sentiment', 'volatility_sentiment', 'fear_greed_index']
            for col in sentiment_cols:
                if col in result_df.columns:
                    result_df[col] = result_df[col].replace([np.inf, -np.inf], np.nan)
                    result_df[col] = result_df[col].fillna(0)
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"计算情绪特征失败: {e}")
            raise
    
    def create_all_features(self, df: pd.DataFrame, external_data: Optional[Dict] = None) -> pd.DataFrame:
        """
        创建所有特征
        
        Args:
            df: 基础数据DataFrame
            external_data: 可选的外部市场数据
            
        Returns:
            pd.DataFrame: 包含所有特征的DataFrame
        """
        try:
            # 逐步添加所有特征
            df_with_features = self.calculate_price_derivatives(df)
            df_with_features = self.calculate_volume_features(df_with_features)
            df_with_features = self.calculate_volatility_features(df_with_features)
            df_with_features = self.calculate_pattern_features(df_with_features)
            df_with_features = self.calculate_cycles(df_with_features)
            df_with_features = self.calculate_sentiment_features(df_with_features, external_data)
            
            return df_with_features
            
        except Exception as e:
            self.logger.error(f"创建所有特征失败: {e}")
            raise
    
    def get_feature_importance(self, feature_names: List[str], importance_values: List[float]) -> Dict[str, float]:
        """
        获取特征重要性
        
        Args:
            feature_names: 特征名称列表
            importance_values: 特征重要性值列表
            
        Returns:
            Dict[str, float]: 特征名称到重要性值的映射
        """
        if len(feature_names) != len(importance_values):
            self.logger.error("特征名称和重要性值列表长度不匹配")
            return {}
        
        return dict(zip(feature_names, importance_values))
    
    def select_features(self, df: pd.DataFrame, top_n: int = 30) -> List[str]:
        """
        基于简单统计方法选择最重要的特征
        
        Args:
            df: 包含所有特征的DataFrame
            top_n: 选择的特征数量
            
        Returns:
            List[str]: 选择的特征名称列表
        """
        try:
            # 选择数值特征
            numeric_df = df.select_dtypes(include=[np.number])
            
            # 计算每个特征与目标变量(收盘价)的相关性
            target = numeric_df['close']
            correlations = numeric_df.corrwith(target).abs().sort_values(ascending=False)
            
            # 排除目标变量本身
            correlations = correlations.drop('close')
            
            # 获取相关性最高的特征
            top_features = correlations.head(top_n).index.tolist()
            
            return top_features
            
        except Exception as e:
            self.logger.error(f"特征选择失败: {e}")
            raise
