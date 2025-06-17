"""
数据预处理模块
负责处理从币安API获取的原始数据，进行清洗、标准化和特征工程
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple
import time
from datetime import datetime, timedelta
import yaml

from data.binance_api import BinanceAPI

class DataProcessor:
    """
    数据预处理类
    处理从交易所获取的原始数据，转换为模型可用的格式
    """
    
    def __init__(self, config_path: str = None, api: Optional[BinanceAPI] = None):
        """
        初始化数据处理器
        
        Args:
            config_path: 配置文件路径
            api: 可选的BinanceAPI实例，如果不提供则创建新实例
        """
        self.logger = logging.getLogger('DataProcessor')
        
        # 如果没有提供配置路径，则使用默认路径
        if config_path is None:
            import os
            from pathlib import Path
            # 使用当前文件的绝对路径找到配置文件
            current_dir = Path(__file__).parent.parent
            config_path = os.path.join(current_dir, 'config', 'model_config.yaml')
            
        self._load_config(config_path)
        self.binance_api = api if api is not None else BinanceAPI()
        
    def _load_config(self, config_path: str) -> None:
        """
        加载模型配置
        
        Args:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            features_config = config.get('model', {}).get('features', {})
            self.lookback_window = features_config.get('lookback_window', 24)
            self.use_technical_indicators = features_config.get('use_technical_indicators', True)
            self.use_market_sentiment = features_config.get('use_market_sentiment', True)
            
            self.logger.info("成功加载数据处理配置")
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            raise
    
    def process_klines(self, klines: List[List]) -> pd.DataFrame:
        """
        处理从API获取的K线数据，转换为DataFrame格式
        
        Args:
            klines: 从API获取的K线数据列表
            
        Returns:
            pd.DataFrame: 处理后的K线数据DataFrame
        """
        try:
            # 转换为DataFrame
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # 转换数据类型
            numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                              'quote_asset_volume', 'taker_buy_base_asset_volume',
                              'taker_buy_quote_asset_volume']
            
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])
                
            # 转换时间戳为datetime
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            # 设置索引
            df.set_index('open_time', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"处理K线数据失败: {e}")
            raise
    
    def fetch_klines(self, symbol: str, interval: str, limit: int = 500,
                   start_time: Optional[int] = None, end_time: Optional[int] = None) -> pd.DataFrame:
        """
        获取K线数据并转换为DataFrame
        
        Args:
            symbol: 交易对 (例如 'BTCUSDT')
            interval: K线间隔 ('1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d')
            limit: 返回的数据量
            start_time: 开始时间（毫秒时间戳）
            end_time: 结束时间（毫秒时间戳）
            
        Returns:
            pd.DataFrame: K线数据DataFrame
        """
        try:
            # 获取K线数据
            klines = self.binance_api.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit,
                start_time=start_time,
                end_time=end_time
            )
            
            # 转换为DataFrame
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # 转换数据类型
            numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                              'quote_asset_volume', 'taker_buy_base_asset_volume',
                              'taker_buy_quote_asset_volume']
            
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])
                
            # 转换时间戳为datetime
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            # 设置索引
            df.set_index('open_time', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"获取K线数据失败: {e}")
            raise
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            df: K线数据DataFrame
            
        Returns:
            pd.DataFrame: 添加了技术指标的DataFrame
        """
        if not self.use_technical_indicators:
            return df
            
        try:
            # 创建副本以避免修改原始数据
            df_with_indicators = df.copy()
            
            # 添加简单移动平均线 (SMA)
            df_with_indicators['sma_7'] = df_with_indicators['close'].rolling(window=7).mean()
            df_with_indicators['sma_25'] = df_with_indicators['close'].rolling(window=25).mean()
            df_with_indicators['sma_99'] = df_with_indicators['close'].rolling(window=99).mean()
            
            # 添加指数移动平均线 (EMA)
            df_with_indicators['ema_9'] = df_with_indicators['close'].ewm(span=9, adjust=False).mean()
            df_with_indicators['ema_21'] = df_with_indicators['close'].ewm(span=21, adjust=False).mean()
            
            # 相对强弱指标 (RSI)
            delta = df_with_indicators['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_with_indicators['rsi_14'] = 100 - (100 / (1 + rs))
            
            # 布林带 (Bollinger Bands)
            sma_20 = df_with_indicators['close'].rolling(window=20).mean()
            std_20 = df_with_indicators['close'].rolling(window=20).std()
            df_with_indicators['bb_upper'] = sma_20 + (std_20 * 2)
            df_with_indicators['bb_middle'] = sma_20
            df_with_indicators['bb_lower'] = sma_20 - (std_20 * 2)
            
            # MACD (Moving Average Convergence Divergence)
            ema_12 = df_with_indicators['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df_with_indicators['close'].ewm(span=26, adjust=False).mean()
            df_with_indicators['macd'] = ema_12 - ema_26
            df_with_indicators['macd_signal'] = df_with_indicators['macd'].ewm(span=9, adjust=False).mean()
            df_with_indicators['macd_hist'] = df_with_indicators['macd'] - df_with_indicators['macd_signal']
            
            # 随机振荡器 (Stochastic Oscillator)
            low_14 = df_with_indicators['low'].rolling(window=14).min()
            high_14 = df_with_indicators['high'].rolling(window=14).max()
            df_with_indicators['stoch_k'] = 100 * ((df_with_indicators['close'] - low_14) / (high_14 - low_14))
            df_with_indicators['stoch_d'] = df_with_indicators['stoch_k'].rolling(window=3).mean()
            
            # 平均真实范围 (ATR)
            high_low = df_with_indicators['high'] - df_with_indicators['low']
            high_close_prev = abs(df_with_indicators['high'] - df_with_indicators['close'].shift(1))
            low_close_prev = abs(df_with_indicators['low'] - df_with_indicators['close'].shift(1))
            tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            df_with_indicators['atr_14'] = tr.rolling(window=14).mean()
            
            # 成交量变化
            df_with_indicators['volume_change'] = df_with_indicators['volume'].pct_change()
            
            # 价格动量
            df_with_indicators['momentum_14'] = df_with_indicators['close'] - df_with_indicators['close'].shift(14)
            
            # 价格变化率
            df_with_indicators['roc_10'] = ((df_with_indicators['close'] / df_with_indicators['close'].shift(10)) - 1) * 100
            
            return df_with_indicators
            
        except Exception as e:
            self.logger.error(f"计算技术指标失败: {e}")
            raise
    
    def add_market_sentiment(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        添加市场情绪指标
        
        Args:
            df: 带有技术指标的DataFrame
            symbol: 交易对
            
        Returns:
            pd.DataFrame: 添加了市场情绪的DataFrame
        """
        if not self.use_market_sentiment:
            return df
            
        try:
            # 创建副本以避免修改原始数据
            df_with_sentiment = df.copy()
            
            # 计算波动性指标: 真实波幅的标准差
            df_with_sentiment['volatility'] = df_with_sentiment['atr_14'].rolling(window=24).std() / df_with_sentiment['close'] * 100
            
            # 计算市场趋势强度指标：计算价格与长期移动平均线的偏离程度
            if 'sma_99' in df_with_sentiment.columns:
                df_with_sentiment['trend_strength'] = (df_with_sentiment['close'] - df_with_sentiment['sma_99']) / df_with_sentiment['sma_99'] * 100
            
            # 买卖压力比：通过买方和卖方成交量计算
            buy_volume = df_with_sentiment['taker_buy_quote_asset_volume']
            sell_volume = df_with_sentiment['quote_asset_volume'] - df_with_sentiment['taker_buy_quote_asset_volume']
            df_with_sentiment['buy_sell_ratio'] = buy_volume / sell_volume
            
            # 平均买卖比例 (14周期)
            df_with_sentiment['avg_buy_sell_ratio'] = df_with_sentiment['buy_sell_ratio'].rolling(window=14).mean()
            
            # 计算交易活跃度 (相对于最近100个时段的平均值)
            df_with_sentiment['trading_activity'] = df_with_sentiment['volume'] / df_with_sentiment['volume'].rolling(window=100).mean()
            
            # 价格偏离度 (相对于20周期移动平均线)
            if 'sma_25' in df_with_sentiment.columns:
                df_with_sentiment['price_deviation'] = (df_with_sentiment['close'] - df_with_sentiment['sma_25']) / df_with_sentiment['sma_25'] * 100
            
            # 计算过去24小时价格变动幅度
            hours_24 = 24 // int(''.join(filter(str.isdigit, df_with_sentiment.index.freq.name))) if df_with_sentiment.index.freq else 24
            df_with_sentiment['price_change_24h'] = df_with_sentiment['close'].pct_change(periods=hours_24) * 100
            
            return df_with_sentiment
            
        except Exception as e:
            self.logger.error(f"添加市场情绪指标失败: {e}")
            raise
    
    def prepare_model_input(self, df: pd.DataFrame, lookback_window: Optional[int] = None) -> np.ndarray:
        """
        准备模型输入数据
        
        Args:
            df: 包含所有特征的DataFrame
            lookback_window: 回溯窗口大小，如果为None则使用配置值
            
        Returns:
            np.ndarray: 模型输入数据，形状为 (n_samples, lookback_window, n_features)
        """
        try:
            if lookback_window is None:
                lookback_window = self.lookback_window
                
            # 删除含有NaN的行
            df_clean = df.dropna()
            
            if len(df_clean) <= lookback_window:
                self.logger.warning(f"数据不足，需要至少 {lookback_window+1} 行有效数据")
                return None
                
            # 选择数值特征列
            feature_columns = [col for col in df_clean.columns if col != 'close_time' and df_clean[col].dtype in [np.float64, np.int64]]
            
            # 标准化特征
            df_norm = (df_clean[feature_columns] - df_clean[feature_columns].mean()) / df_clean[feature_columns].std()
            df_norm.fillna(0, inplace=True)  # 替换可能的NaN值
            
            # 创建输入序列
            n_samples = len(df_norm) - lookback_window
            n_features = len(feature_columns)
            
            X = np.zeros((n_samples, lookback_window, n_features))
            
            for i in range(n_samples):
                X[i] = df_norm.iloc[i:i+lookback_window].values
                
            return X
            
        except Exception as e:
            self.logger.error(f"准备模型输入数据失败: {e}")
            raise
    
    def get_latest_data(self, symbol: str, interval: str, lookback_bars: int = 200) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        获取最新数据并准备模型输入
        
        Args:
            symbol: 交易对 (例如 'BTCUSDT')
            interval: K线间隔
            lookback_bars: 获取的历史K线数量
            
        Returns:
            Tuple[pd.DataFrame, np.ndarray]: 处理后的DataFrame和模型输入数据
        """
        # 获取K线数据
        raw_df = self.fetch_klines(symbol=symbol, interval=interval, limit=lookback_bars + 100)  # 多获取一些数据以确保足够
        
        # 计算技术指标
        df_with_indicators = self.calculate_technical_indicators(raw_df)
        
        # 添加市场情绪
        full_df = self.add_market_sentiment(df_with_indicators, symbol)
        
        # 准备模型输入
        model_input = self.prepare_model_input(full_df)
        
        return full_df, model_input
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """
        获取特征名称列表
        
        Args:
            df: 包含所有特征的DataFrame
            
        Returns:
            List[str]: 特征名称列表
        """
        # 排除非数值特征
        feature_columns = [col for col in df.columns if col != 'close_time' and df[col].dtype in [np.float64, np.int64]]
        return feature_columns
        
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        验证数据完整性和质量
        
        Args:
            df: 待验证的DataFrame
            
        Returns:
            bool: 数据是否有效
        """
        # 检查是否有足够的数据行
        if len(df) < self.lookback_window:
            self.logger.warning(f"数据行不足: {len(df)} < {self.lookback_window}")
            return False
            
        # 检查是否有缺失值
        missing_percentage = df.isnull().mean() * 100
        high_missing = missing_percentage[missing_percentage > 5].index.tolist()
        if high_missing:
            self.logger.warning(f"以下特征缺失值比例高于5%: {high_missing}")
            
        # 检查数据的时间连续性 (针对时间序列数据)
        if df.index.name == 'open_time' and len(df) > 1:
            time_diffs = df.index.to_series().diff()[1:]
            expected_diff = pd.Timedelta(hours=1)  # 假设是1小时K线
            if interval_str := df.index.freq:
                if 'm' in interval_str:
                    minutes = int(''.join(filter(str.isdigit, interval_str)))
                    expected_diff = pd.Timedelta(minutes=minutes)
                elif 'h' in interval_str:
                    hours = int(''.join(filter(str.isdigit, interval_str)))
                    expected_diff = pd.Timedelta(hours=hours)
                elif 'd' in interval_str:
                    days = int(''.join(filter(str.isdigit, interval_str)))
                    expected_diff = pd.Timedelta(days=days)
            
            irregular_intervals = time_diffs[time_diffs != expected_diff]
            if not irregular_intervals.empty:
                self.logger.warning(f"数据时间间隔不一致，有 {len(irregular_intervals)} 个异常间隔")
                
        # 检查异常值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            outliers = df[(df[col] > mean + 3 * std) | (df[col] < mean - 3 * std)]
            if len(outliers) > 0:
                self.logger.info(f"特征 '{col}' 中有 {len(outliers)} 个异常值 (超过3个标准差)")
                
        # 所有检查通过
        return True
