#!/usr/bin/env python3
"""
Fetch BTC historical data from cryptocurrency exchanges and
save to CSV in the required format for the BTC RL project.

This script supports fetching data from multiple exchanges,
with different time ranges, timeframes, and merging them into a unified dataset.

Usage
-----
python -m btc_rl.src.data_fetcher --exchange binance --start_date 2020-01-01 --end_date 2025-01-01 --timeframe 1h
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import time
import random

import ccxt
import pandas as pd
import numpy as np
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("data_fetcher")

# 支持的交易所列表
SUPPORTED_EXCHANGES = [
    'binance', 'coinbase', 'kraken', 'huobi', 
    'kucoin', 'bitfinex', 'okx', 'bybit'
]

# 支持的时间周期
SUPPORTED_TIMEFRAMES = [
    '1m', '5m', '15m', '30m', '1h', '4h', '1d'
]

def fetch_historical_data(
    exchange_id: str,
    symbol: str = 'BTC/USDT',
    timeframe: str = '1h',
    start_date: str = None,
    end_date: str = None,
    max_retries: int = 5,
    retry_delay: int = 10,
    exponential_backoff: bool = True
) -> pd.DataFrame:
    """
    从指定交易所获取历史数据
    
    Parameters
    ----------
    exchange_id : str
        交易所ID (例如 'binance', 'coinbase', 等)
    symbol : str
        交易对 (默认 'BTC/USDT')
    timeframe : str
        时间周期 (默认 '1h' 表示1小时)
    start_date : str
        开始日期，格式为 'YYYY-MM-DD'
    end_date : str
        结束日期，格式为 'YYYY-MM-DD'
    max_retries : int
        API请求失败时最大重试次数
    retry_delay : int
        重试延迟基础秒数
    exponential_backoff : bool
        是否使用指数退避算法
        
    Returns
    -------
    pd.DataFrame
        包含历史数据的DataFrame
    """
    if exchange_id.lower() not in SUPPORTED_EXCHANGES:
        raise ValueError(f"不支持的交易所: {exchange_id}。支持的交易所: {', '.join(SUPPORTED_EXCHANGES)}")
        
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"不支持的时间周期: {timeframe}。支持的时间周期: {', '.join(SUPPORTED_TIMEFRAMES)}")
    
    # 创建交易所实例
    exchange_class = getattr(ccxt, exchange_id.lower())
    exchange = exchange_class({
        'enableRateLimit': True,  # 尊重API速率限制
    })
    
    # 检查交易所是否有获取历史数据的方法
    if not exchange.has['fetchOHLCV']:
        raise Exception(f"{exchange_id} 不支持获取历史K线数据")
    
    # 转换日期格式
    since = None
    if start_date:
        since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    
    end_timestamp = None
    if end_date:
        end_timestamp = datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000
    
    # 设置结果列表
    all_candles = []
    
    # 获取时间周期的毫秒数
    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
    
    # 当前时间戳
    current = since
    
    # 获取数据的最大限制（不同交易所可能不同）
    limit = 1000  # 一般默认值
    
    # 使用进度条获取数据
    logger.info(f"从 {exchange_id} 获取 {symbol} {timeframe} 数据...")
    pbar = tqdm(desc=f"{exchange_id} 数据", unit="条")
    
    # 循环获取数据直到达到结束日期
    consecutive_errors = 0
    
    while True:
        try:
            # 获取历史K线数据
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=current, limit=limit)
            
            if not candles or len(candles) == 0:
                if consecutive_errors > 0:
                    logger.warning(f"返回空数据，但之前有错误。等待后重试...")
                    time.sleep(retry_delay)
                    consecutive_errors += 1
                    if consecutive_errors >= max_retries:
                        logger.error(f"连续 {max_retries} 次获取空数据，可能已到达数据末尾")
                        break
                    continue
                else:
                    logger.info("没有更多数据，已到达数据末尾")
                    break
                
            # 重置连续错误计数
            consecutive_errors = 0
                
            # 更新进度条
            pbar.update(len(candles))
            
            # 添加到结果列表
            all_candles.extend(candles)
            
            # 更新当前时间戳
            current = candles[-1][0] + timeframe_ms
            
            # 如果达到结束日期，则退出循环
            if end_timestamp and current >= end_timestamp:
                break
                
            # 为避免API速率限制，添加短暂延迟和一些随机性
            base_delay = exchange.rateLimit / 1000  # 转换为秒
            jitter = random.uniform(0.1, 0.3) * base_delay  # 添加随机波动
            time.sleep(base_delay + jitter)
            
        except Exception as e:
            consecutive_errors += 1
            
            # 计算延迟时间（指数退避）
            if exponential_backoff:
                delay = retry_delay * (2 ** (consecutive_errors - 1)) + random.uniform(1, 5)
            else:
                delay = retry_delay
                
            err_msg = str(e).lower()
            
            if "rate limit" in err_msg or "too many requests" in err_msg:
                logger.warning(f"触发API速率限制 (尝试 #{consecutive_errors}/{max_retries})，等待 {delay:.1f} 秒后重试...")
                time.sleep(delay)
            elif "connection" in err_msg or "timeout" in err_msg or "network" in err_msg:
                logger.warning(f"网络错误 (尝试 #{consecutive_errors}/{max_retries})，等待 {delay:.1f} 秒后重试: {e}")
                time.sleep(delay)
            else:
                logger.error(f"获取数据时出错 (尝试 #{consecutive_errors}/{max_retries}): {e}")
                time.sleep(delay)
                
            # 达到最大重试次数
            if consecutive_errors >= max_retries:
                logger.error(f"达到最大重试次数 ({max_retries})，放弃该时间段")
                # 尝试移动到下一个时间段而不是完全放弃
                if current and timeframe_ms:
                    # 移动到下一个可能的数据块
                    next_period = timeframe_ms * limit
                    current += next_period
                    logger.info(f"跳过当前数据块，尝试获取下一个时间段的数据...")
                    consecutive_errors = 0
                else:
                    break
    
    pbar.close()
    
    # 如果没有获取到数据，返回空DataFrame
    if not all_candles:
        logger.warning(f"未获取到任何数据")
        return pd.DataFrame()
    
    # 创建DataFrame
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # 转换时间戳为datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # 删除重复数据
    df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    
    # 如果有结束日期限制，应用过滤
    if end_date:
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
        df = df[df['timestamp'] < end_date_dt].reset_index(drop=True)
    
    # 按时间戳排序
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 验证数据质量和连续性
    validate_data_quality(df, timeframe)
    
    return df

def validate_data_quality(df: pd.DataFrame, timeframe: str) -> None:
    """
    验证获取的数据的质量和连续性
    
    Parameters
    ----------
    df : pd.DataFrame
        需要验证的数据
    timeframe : str
        时间周期
    """
    if df.empty:
        logger.warning("数据为空，无法进行验证")
        return
    
    # 检查缺失值
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        logger.warning(f"数据中包含 {missing_count} 个缺失值")
        
    # 检查数据点数量是否合理
    expected_points = {
        '1m': 1440,  # 一天约1440个1分钟K线
        '5m': 288,   # 一天约288个5分钟K线
        '15m': 96,   # 一天约96个15分钟K线
        '30m': 48,   # 一天约48个30分钟K线
        '1h': 24,    # 一天约24个1小时K线
        '4h': 6,     # 一天约6个4小时K线
        '1d': 1      # 一天1个日K线
    }
    
    # 计算时间跨度
    time_range = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
    days = time_range / (24 * 60 * 60)
    
    if timeframe in expected_points:
        expected_total = int(days * expected_points[timeframe])
        actual_total = len(df)
        missing_ratio = 1 - (actual_total / expected_total) if expected_total > 0 else 0
        
        if missing_ratio > 0.1:  # 超过10%的数据点缺失
            logger.warning(f"数据可能不完整: 预期约 {expected_total} 个数据点，实际获取 {actual_total} 个 (缺失率: {missing_ratio:.1%})")
        else:
            logger.info(f"数据完整性检查通过: 预期约 {expected_total} 个数据点，实际获取 {actual_total} 个")
    
    # 检查异常值
    price_std = df['close'].std()
    price_mean = df['close'].mean()
    outliers = df[(df['close'] > price_mean + 3 * price_std) | (df['close'] < price_mean - 3 * price_std)]
    
    if len(outliers) > 0:
        logger.warning(f"检测到 {len(outliers)} 个可能的价格异常值")
        
    # 检查时间戳连续性
    if len(df) > 1:
        # 根据时间周期计算预期的时间间隔
        expected_interval = {
            '1m': pd.Timedelta(minutes=1),
            '5m': pd.Timedelta(minutes=5),
            '15m': pd.Timedelta(minutes=15),
            '30m': pd.Timedelta(minutes=30),
            '1h': pd.Timedelta(hours=1),
            '4h': pd.Timedelta(hours=4),
            '1d': pd.Timedelta(days=1)
        }.get(timeframe)
        
        if expected_interval:
            df = df.copy()
            df['time_diff'] = df['timestamp'].diff()
            gaps = df[df['time_diff'] > expected_interval * 1.5]
            
            if len(gaps) > 0:
                logger.warning(f"检测到 {len(gaps)} 个时间间隔异常，可能存在数据缺失")

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    预处理数据，添加指标如对数收益率、波动率、RSI和SMA
    
    Parameters
    ----------
    df : pd.DataFrame
        原始数据
    
    Returns
    -------
    pd.DataFrame
        处理后的数据
    """
    # 确保时间戳是索引
    df = df.copy()
    
    # 计算对数收益
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # 计算30周期波动率
    df['volatility_30_period'] = df['log_returns'].rolling(window=30).std()
    
    # 计算RSI-14
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI14'] = 100 - (100 / (1 + rs))
    
    # 计算SMA-50（50周期简单移动平均线）
    df['SMA50'] = df['close'].rolling(window=50).mean()
    
    # 删除NaN值
    df = df.dropna().reset_index(drop=True)
    
    return df

def main(args):
    # 获取数据
    df = fetch_historical_data(
        exchange_id=args.exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        exponential_backoff=not args.no_backoff
    )
    
    if df.empty:
        logger.error("未获取到有效数据，退出程序")
        return
    
    # 预处理数据
    df_processed = preprocess_data(df)
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 生成输出文件名
    timeframe_str = args.timeframe.replace('m', 'min').replace('h', 'hour').replace('d', 'day')
    output_file = output_dir / f"BTC_{timeframe_str}_{args.start_date}_{args.end_date}.csv"
    
    # 保存到CSV
    df_processed.to_csv(output_file, index=False)
    logger.info(f"数据已保存至 {output_file}")
    logger.info(f"总行数: {len(df_processed)}")
    
    # 返回路径字符串用于后续处理
    return str(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="获取加密货币历史数据")
    parser.add_argument("--exchange", type=str, default="binance", help=f"交易所名称, 支持: {', '.join(SUPPORTED_EXCHANGES)}")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="交易对 (默认: BTC/USDT)")
    parser.add_argument("--timeframe", type=str, default="1h", help=f"时间周期, 支持: {', '.join(SUPPORTED_TIMEFRAMES)}")
    parser.add_argument("--start_date", type=str, required=True, help="开始日期 (格式: YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, required=True, help="结束日期 (格式: YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="btc_rl/data", help="输出目录 (默认: btc_rl/data)")
    parser.add_argument("--max_retries", type=int, default=5, help="API请求失败时的最大重试次数 (默认: 5)")
    parser.add_argument("--retry_delay", type=int, default=10, help="重试延迟基础秒数 (默认: 10)")
    parser.add_argument("--no_backoff", action="store_true", help="禁用指数退避算法 (默认启用)")
    
    args = parser.parse_args()
    main(args)
