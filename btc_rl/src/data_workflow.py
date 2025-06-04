#!/usr/bin/env python3
"""
集成工作流脚本 - 从获取数据到生成训练测试数据的完整流程

此脚本执行以下步骤:
1. 从指定交易所API获取BTC历史数据
2. 预处理数据并生成所需的BTC_hourly.csv或其他时间精度的CSV文件
3. 运行preprocessing.py脚本生成训练和测试数据集

Usage
-----
python -m btc_rl.src.data_workflow --exchange binance --start_date 2020-01-01 --end_date 2025-01-01 --timeframe 1h
"""

import argparse
import os
import subprocess
import sys
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# 导入自定义模块
from btc_rl.src.data_fetcher import fetch_historical_data, preprocess_data, SUPPORTED_TIMEFRAMES
from btc_rl.src.config import config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("data_workflow")

def convert_to_hourly(df, source_timeframe):
    """
    将非小时级别的数据转换为小时级别
    
    Parameters
    ----------
    df : pd.DataFrame
        原始数据框架
    source_timeframe : str
        源数据的时间周期
        
    Returns
    -------
    pd.DataFrame
        转换后的小时级别数据
    """
    if source_timeframe == '1h':
        return df
    
    # 如果数据精度高于1小时，需要重采样
    if source_timeframe in ['1m', '5m', '15m', '30m']:
        # 将timestamp设为索引
        df = df.set_index('timestamp')
        
        # 重采样到小时级别
        hourly = df.resample('1H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # 重设索引
        hourly = hourly.reset_index()
        return hourly
        
    # 如果数据精度低于1小时，无法直接转换
    elif source_timeframe in ['4h', '1d']:
        logger.warning(f"时间周期 {source_timeframe} 粒度低于1小时，无法准确转换为小时数据，将保留原始粒度")
        return df
    else:
        logger.warning(f"不支持的时间周期: {source_timeframe}，将保留原始数据")
        return df

def run_preprocessing(csv_path, target_timeframe='1h'):
    """
    运行预处理脚本生成训练和测试数据
    
    Parameters
    ----------
    csv_path : str
        CSV文件的路径
    target_timeframe : str
        目标时间周期，默认为1h
        
    Returns
    -------
    bool
        处理是否成功
    """
    try:
        logger.info(f"运行数据预处理脚本...")
        
        # 如果时间周期不是1h，需要进行特殊处理
        if target_timeframe != '1h' and '_hourly_' not in csv_path:
            # 读取原始数据
            df = pd.read_csv(csv_path)
            
            # 转换时间戳列
            if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 转换为小时级别
            df_hourly = convert_to_hourly(df, target_timeframe)
            
            # 保存为临时小时级别文件
            hourly_path = csv_path.replace(f"_{target_timeframe.replace('m', 'min').replace('h', 'hour').replace('d', 'day')}_", "_hourly_")
            df_hourly.to_csv(hourly_path, index=False)
            logger.info(f"已将 {target_timeframe} 数据转换为小时级别并保存至 {hourly_path}")
            csv_path = hourly_path
        
        # 构建预处理脚本命令
        cmd = [sys.executable, "-m", "btc_rl.src.preprocessing", "--csv", csv_path]
        
        # 执行命令
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            logger.info("预处理完成。")
            logger.info(stdout)
        else:
            logger.error(f"预处理失败: {stderr}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"运行预处理时出错: {e}")
        return False

def copy_to_latest(csv_path, timeframe='1h'):
    """
    将生成的CSV文件复制为标准BTC数据文件
    
    Parameters
    ----------
    csv_path : str
        生成的CSV文件路径
    timeframe : str
        时间周期
        
    Returns
    -------
    str
        标准文件路径
    """
    try:
        source_path = Path(csv_path)
        
        # 根据时间周期生成合适的目标文件名
        timeframe_str = timeframe.replace('m', 'min').replace('h', 'hour').replace('d', 'day')
        dest_path = source_path.parent / f"BTC_{timeframe_str}.csv"
        
        # 如果目标文件已存在，先备份
        if dest_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            backup_path = dest_path.parent / f"BTC_{timeframe_str}.{timestamp}.bak"
            dest_path.rename(backup_path)
            logger.info(f"已备份原文件至 {backup_path}")
        
        # 复制文件
        import shutil
        shutil.copy2(source_path, dest_path)
        logger.info(f"已将 {source_path} 复制为 {dest_path}")
        
        # 如果不是小时级别数据，还需要生成标准小时级别文件用于模型训练
        if timeframe != '1h':
            # 从配置中读取的标准小时级别文件路径
            hourly_path = source_path.parent / "BTC_hourly.csv"
            
            # 读取原始数据
            df = pd.read_csv(source_path)
            if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 转换为小时级别
            df_hourly = convert_to_hourly(df, timeframe)
            
            # 如果小时级别文件已存在，先备份
            if hourly_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                hourly_backup_path = hourly_path.parent / f"BTC_hourly.{timestamp}.bak"
                hourly_path.rename(hourly_backup_path)
                logger.info(f"已备份原小时级别文件至 {hourly_backup_path}")
            
            # 保存小时级别数据
            df_hourly.to_csv(hourly_path, index=False)
            logger.info(f"已生成标准小时级别数据文件: {hourly_path}")
            
            # 返回小时级别文件路径用于预处理
            return str(hourly_path)
        
        return str(dest_path)
    except Exception as e:
        logger.error(f"复制文件时出错: {e}")
        return None

def validate_timeframe_data(csv_path, timeframe):
    """
    验证指定时间精度的数据文件是否合格
    
    Parameters
    ----------
    csv_path : str
        CSV文件路径
    timeframe : str
        时间周期
        
    Returns
    -------
    bool
        数据是否合格
    """
    try:
        # 读取数据
        df = pd.read_csv(csv_path)
        if df.empty:
            logger.error(f"数据文件 {csv_path} 为空")
            return False
        
        # 检查必要的列
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"数据文件缺少必要的列: {col}")
                return False
        
        # 转换时间戳列
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
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
        
        # 计算日期范围
        date_range = (df['timestamp'].max() - df['timestamp'].min()).days
        if date_range < 1:
            date_range = 1
        
        # 检查数据点计数
        if timeframe in expected_points:
            expected = date_range * expected_points[timeframe] * 0.8  # 允许20%缺失
            if len(df) < expected:
                logger.warning(f"数据点不足: 预期至少 {expected:.0f} 个点，实际 {len(df)} 个点")
                
        # 检查缺失值
        missing = df[required_cols].isnull().sum().sum()
        if missing > 0:
            logger.warning(f"数据中包含 {missing} 个缺失值")
            
        # 根据时间周期检查连续性
        if len(df) > 1:
            df = df.sort_values('timestamp')
            df['time_diff'] = df['timestamp'].diff()
            
            # 计算预期的时间差
            expected_diff = {
                '1m': pd.Timedelta(minutes=1),
                '5m': pd.Timedelta(minutes=5),
                '15m': pd.Timedelta(minutes=15),
                '30m': pd.Timedelta(minutes=30),
                '1h': pd.Timedelta(hours=1),
                '4h': pd.Timedelta(hours=4),
                '1d': pd.Timedelta(days=1)
            }.get(timeframe)
            
            if expected_diff:
                # 允许一些偏差(1.5倍)
                gap_threshold = expected_diff * 1.5
                gaps = df[df['time_diff'] > gap_threshold]
                gap_count = len(gaps)
                
                if gap_count > 0:
                    gap_percent = (gap_count / len(df)) * 100
                    logger.warning(f"数据中检测到 {gap_count} 个大于预期的时间间隔 ({gap_percent:.1f}%)")
                    
                    # 如果超过10%的数据点有间隔问题，数据可能不可靠
                    if gap_percent > 10:
                        logger.error(f"数据间隔问题超过10%，可能不可靠")
                        return False
        
        # 数据基本满足要求
        logger.info(f"数据验证通过: {csv_path} ({len(df)} 个数据点)")
        return True
        
    except Exception as e:
        logger.error(f"验证数据时出错: {e}")
        return False

def fetch_data(args):
    """
    获取历史数据并保存为CSV
    
    Parameters
    ----------
    args : argparse.Namespace
        命令行参数
    
    Returns
    -------
    str
        CSV文件路径
    """
    try:
        # 从配置文件获取API参数
        max_retries = config.get_api_max_retries()
        retry_delay = config.get_api_retry_delay()
        use_backoff = config.get_api_use_exponential_backoff()
        
        logger.info(f"从 {args.exchange} 获取 {args.symbol} {args.timeframe} 数据...")
        
        # 获取数据
        df = fetch_historical_data(
            exchange_id=args.exchange,
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date,
            max_retries=max_retries,
            retry_delay=retry_delay,
            exponential_backoff=use_backoff
        )
        
        if df.empty:
            logger.error("未获取到有效数据")
            return None
        
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
        
        # 验证数据质量
        if validate_timeframe_data(str(output_file), args.timeframe):
            logger.info("数据质量验证通过")
        else:
            logger.warning("数据质量验证未通过，但将继续使用该数据")
            
        return str(output_file)
        
    except Exception as e:
        logger.error(f"获取数据时出错: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="BTC交易智能体数据准备工作流")
    parser.add_argument("--exchange", type=str, default=config.get_default_exchange(), help="交易所名称")
    parser.add_argument("--symbol", type=str, default=config.get_default_symbol(), help="交易对 (默认: BTC/USDT)")
    parser.add_argument("--timeframe", type=str, default=config.get_default_timeframe(), 
                       help=f"时间周期，支持: {', '.join(SUPPORTED_TIMEFRAMES)}")
    parser.add_argument("--start_date", type=str, required=True, help="开始日期 (格式: YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, required=True, help="结束日期 (格式: YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default=config.get_data_dir(), help="输出目录")
    parser.add_argument("--skip-fetch", action="store_true", help="跳过数据获取步骤，直接使用现有CSV文件")
    parser.add_argument("--csv-path", type=str, help="如果跳过数据获取，指定CSV文件路径")
    parser.add_argument("--skip-validation", action="store_true", help="跳过数据验证步骤")
    
    args = parser.parse_args()
    
    try:
        # 步骤1: 获取数据并生成CSV (除非跳过)
        csv_path = None
        if args.skip_fetch:
            if not args.csv_path:
                logger.error("启用--skip-fetch时必须指定--csv-path")
                return
            csv_path = args.csv_path
            logger.info(f"跳过数据获取，使用现有CSV文件: {csv_path}")
            
            # 即使跳过获取，也需要验证数据质量
            if not args.skip_validation and not validate_timeframe_data(csv_path, args.timeframe):
                logger.warning("现有数据质量验证未通过，但将继续使用该数据")
        else:
            logger.info("开始获取BTC历史数据...")
            csv_path = fetch_data(args)
            if not csv_path:
                logger.error("数据获取失败，退出工作流")
                return
        
        # 步骤2: 复制为标准文件名
        std_csv_path = copy_to_latest(csv_path, args.timeframe)
        if not std_csv_path:
            logger.error("复制文件失败，退出工作流")
            return
            
        # 步骤3: 运行预处理生成训练和测试数据
        success = run_preprocessing(std_csv_path, args.timeframe)
        if success:
            logger.info("✅ 整个工作流程已成功完成!")
            logger.info(f"数据源: {args.exchange}")
            logger.info(f"时间周期: {args.timeframe}")
            logger.info(f"时间范围: {args.start_date} 至 {args.end_date}")
            logger.info("训练和测试数据已生成，可以用于模型训练")
        else:
            logger.error("工作流程未能完成")
    
    except Exception as e:
        logger.error(f"工作流执行出错: {e}")

if __name__ == "__main__":
    main()
