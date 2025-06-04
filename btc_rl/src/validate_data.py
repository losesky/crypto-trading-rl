#!/usr/bin/env python3
"""
数据质量验证工具 - 检查获取的BTC历史数据质量

此脚本用于验证BTC历史数据的质量，检查以下方面：
1. 数据完整性 - 是否有足够的数据点，是否覆盖整个时间范围
2. 数据连续性 - 时间序列是否有缺口或异常间隔
3. 数据异常值 - 检测价格、交易量等异常值
4. 数据一致性 - 验证不同时间精度数据之间的一致性

Usage
-----
python -m btc_rl.src.validate_data --csv btc_rl/data/BTC_hourly.csv
python -m btc_rl.src.validate_data --csv btc_rl/data/BTC_1min.csv --timeframe 1m
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("validate_data")

# 支持的时间精度映射到预期每日数据点
TIMEFRAME_POINTS = {
    '1m': 1440,   # 一天约1440个1分钟K线
    '5m': 288,    # 一天约288个5分钟K线
    '15m': 96,    # 一天约96个15分钟K线
    '30m': 48,    # 一天约48个30分钟K线
    '1h': 24,     # 一天约24个1小时K线
    '4h': 6,      # 一天约6个4小时K线
    '1d': 1       # 一天1个日K线
}

# 时间周期转换为timedelta
TIMEFRAME_DELTA = {
    '1m': timedelta(minutes=1),
    '5m': timedelta(minutes=5),
    '15m': timedelta(minutes=15),
    '30m': timedelta(minutes=30),
    '1h': timedelta(hours=1),
    '4h': timedelta(hours=4),
    '1d': timedelta(days=1)
}

def load_data(csv_path):
    """
    加载数据并进行基本清理
    
    Parameters
    ----------
    csv_path : str
        CSV文件路径
    
    Returns
    -------
    pd.DataFrame
        加载并清理的数据框架
    """
    try:
        df = pd.read_csv(csv_path)
        
        if df.empty:
            logger.error(f"数据文件 {csv_path} 为空")
            return None
        
        # 检查必要的列
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"数据文件缺少必要的列: {', '.join(missing_cols)}")
            return None
        
        # 转换时间戳列
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 按时间排序
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    except Exception as e:
        logger.error(f"加载数据时出错: {e}")
        return None

def check_data_completeness(df, timeframe):
    """
    检查数据完整性
    
    Parameters
    ----------
    df : pd.DataFrame
        数据框架
    timeframe : str
        时间周期
    
    Returns
    -------
    tuple
        (完整性得分, 问题描述列表)
    """
    problems = []
    
    # 计算时间范围
    date_range = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / (24 * 60 * 60)
    if date_range < 1:
        date_range = 1
        
    # 估计预期数据点数
    expected_points = date_range * TIMEFRAME_POINTS.get(timeframe, 24)  # 默认使用1h的点数
    actual_points = len(df)
    
    completeness_score = min(actual_points / expected_points, 1.0) if expected_points > 0 else 0
    
    if completeness_score < 0.95:
        problems.append(f"数据不完整: 预期约 {expected_points:.0f} 个数据点，实际 {actual_points} 个，完整度 {completeness_score:.1%}")
    
    # 检查数据起始和结束时间
    start_time = df['timestamp'].min()
    end_time = df['timestamp'].max()
    
    # 计算日期覆盖范围
    days_covered = (end_time - start_time).total_seconds() / (24 * 60 * 60)
    
    if days_covered < 30:
        problems.append(f"数据覆盖天数较少: {days_covered:.1f} 天，可能不足以训练出健壮的模型")
    
    # 检查数据是否覆盖近期
    now = datetime.now()
    days_since_last = (now - end_time).total_seconds() / (24 * 60 * 60)
    
    if days_since_last > 7:
        problems.append(f"数据不是最新的: 最后一个数据点是 {days_since_last:.1f} 天前")
    
    return completeness_score, problems

def check_data_continuity(df, timeframe):
    """
    检查数据连续性
    
    Parameters
    ----------
    df : pd.DataFrame
        数据框架
    timeframe : str
        时间周期
    
    Returns
    -------
    tuple
        (连续性得分, 问题描述列表)
    """
    problems = []
    
    # 计算时间间隔
    df = df.copy()
    df['time_diff'] = df['timestamp'].diff()
    
    # 获取预期的时间间隔
    expected_diff = TIMEFRAME_DELTA.get(timeframe, timedelta(hours=1))
    
    # 检测异常间隔
    # 允许1.5倍的偏差
    gap_threshold = expected_diff * 1.5
    gaps = df[df['time_diff'] > gap_threshold]
    
    if len(gaps) > 0:
        # 分析缺口
        gap_sizes = []
        for _, row in gaps.iterrows():
            gap_size = row['time_diff'] / expected_diff
            gap_sizes.append(gap_size)
            
            # 记录大缺口
            if gap_size > 5:
                problems.append(f"检测到大缺口: {row['timestamp']} 前有约 {gap_size:.1f} 个缺失数据点")
        
        # 计算连续性分数
        # 连续性 = 1 - (缺口总数 / 总数据点数)
        continuity_score = 1 - (len(gaps) / len(df))
        
        if len(gaps) > len(df) * 0.05:  # 超过5%的数据点有间隔问题
            problems.append(f"数据连续性较差: 检测到 {len(gaps)} 个异常间隔 ({len(gaps)/len(df):.1%})")
    else:
        continuity_score = 1.0
    
    return continuity_score, problems

def check_data_anomalies(df):
    """
    检查数据异常值
    
    Parameters
    ----------
    df : pd.DataFrame
        数据框架
    
    Returns
    -------
    tuple
        (无异常值得分, 问题描述列表)
    """
    problems = []
    
    # 检查价格异常值
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            # 使用Z-score方法检测异常值
            mean = df[col].mean()
            std = df[col].std()
            
            if std == 0:
                problems.append(f"{col} 列标准差为0，可能是常数")
                continue
                
            z_scores = np.abs((df[col] - mean) / std)
            anomalies = df[z_scores > 3]  # 超过3个标准差
            
            anomaly_ratio = len(anomalies) / len(df) if len(df) > 0 else 0
            
            if anomaly_ratio > 0.01:  # 超过1%的数据点是异常值
                problems.append(f"{col} 列中检测到 {len(anomalies)} 个可能的异常值 ({anomaly_ratio:.1%})")
    
    # 检查交易量异常值
    if 'volume' in df.columns:
        mean_vol = df['volume'].mean()
        std_vol = df['volume'].std()
        
        if std_vol > 0:
            z_scores_vol = np.abs((df['volume'] - mean_vol) / std_vol)
            vol_anomalies = df[z_scores_vol > 5]  # 交易量波动较大，使用更高的阈值
            
            vol_anomaly_ratio = len(vol_anomalies) / len(df) if len(df) > 0 else 0
            
            if vol_anomaly_ratio > 0.02:  # 超过2%的交易量数据点是异常值
                problems.append(f"交易量列中检测到 {len(vol_anomalies)} 个可能的异常值 ({vol_anomaly_ratio:.1%})")
    
    # 检查OHLC顺序关系是否正确
    invalid_ohlc = df[(df['high'] < df['low']) | (df['open'] > df['high']) | (df['open'] < df['low']) | 
                     (df['close'] > df['high']) | (df['close'] < df['low'])]
    
    if len(invalid_ohlc) > 0:
        problems.append(f"检测到 {len(invalid_ohlc)} 条OHLC数据违反价格关系 (high < low 或其他不合理关系)")
        
    # 检查负值
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                problems.append(f"{col} 列中有 {neg_count} 个负值，这是不合理的")
    
    # 计算无异常值得分
    all_columns = ['open', 'high', 'low', 'close', 'volume']
    available_columns = [col for col in all_columns if col in df.columns]
    
    if not available_columns:
        return 0, ["无法评估异常值：缺少必要的列"]
        
    anomaly_free_score = 1.0
    if problems:
        # 每个问题降低得分
        anomaly_free_score = max(0, 1.0 - 0.2 * len(problems))
    
    return anomaly_free_score, problems

def generate_quality_report(df, timeframe, output_dir=None):
    """
    生成数据质量报告
    
    Parameters
    ----------
    df : pd.DataFrame
        数据框架
    timeframe : str
        时间周期
    output_dir : str, optional
        输出目录
    
    Returns
    -------
    dict
        质量报告
    """
    if df is None:
        return {
            "status": "error",
            "message": "无法加载数据"
        }
    
    report = {
        "status": "success",
        "data_points": len(df),
        "date_range": {
            "start": df['timestamp'].min().strftime("%Y-%m-%d %H:%M:%S"),
            "end": df['timestamp'].max().strftime("%Y-%m-%d %H:%M:%S"),
            "days": (df['timestamp'].max() - df['timestamp'].min()).days
        },
        "timeframe": timeframe,
        "quality_scores": {},
        "problems": {}
    }
    
    # 1. 检查数据完整性
    completeness_score, completeness_problems = check_data_completeness(df, timeframe)
    report["quality_scores"]["completeness"] = completeness_score
    report["problems"]["completeness"] = completeness_problems
    
    # 2. 检查数据连续性
    continuity_score, continuity_problems = check_data_continuity(df, timeframe)
    report["quality_scores"]["continuity"] = continuity_score
    report["problems"]["continuity"] = continuity_problems
    
    # 3. 检查数据异常值
    anomaly_free_score, anomaly_problems = check_data_anomalies(df)
    report["quality_scores"]["anomaly_free"] = anomaly_free_score
    report["problems"]["anomalies"] = anomaly_problems
    
    # 计算总体质量分数
    weights = {
        "completeness": 0.4,
        "continuity": 0.4,
        "anomaly_free": 0.2
    }
    
    overall_score = sum(score * weights[key] for key, score in report["quality_scores"].items())
    report["overall_quality_score"] = overall_score
    
    # 根据总体分数评定数据质量
    if overall_score >= 0.9:
        report["quality_rating"] = "优秀"
    elif overall_score >= 0.75:
        report["quality_rating"] = "良好"
    elif overall_score >= 0.6:
        report["quality_rating"] = "一般"
    else:
        report["quality_rating"] = "较差"
    
    # 如果提供了输出目录，生成可视化图表
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # 1. 价格走势图
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # 价格图
        df.plot(x='timestamp', y=['open', 'high', 'low', 'close'], ax=axes[0])
        axes[0].set_title(f"BTC 价格 ({report['date_range']['start']} - {report['date_range']['end']})")
        axes[0].set_ylabel("价格 (USDT)")
        axes[0].grid(True)
        
        # 交易量图
        df.plot(x='timestamp', y='volume', ax=axes[1], color='green', alpha=0.7)
        axes[1].set_title("交易量")
        axes[1].set_ylabel("交易量")
        axes[1].grid(True)
        
        # 时间间隔图
        if 'time_diff' not in df.columns:
            df['time_diff'] = df['timestamp'].diff().dt.total_seconds() / 60  # 转为分钟
        
        df.plot(x='timestamp', y='time_diff', ax=axes[2], color='red', alpha=0.5)
        axes[2].set_title("数据点时间间隔 (分钟)")
        axes[2].set_ylabel("间隔 (分钟)")
        axes[2].axhline(y=TIMEFRAME_DELTA[timeframe].total_seconds()/60, color='blue', linestyle='--', 
                        label=f"预期间隔 ({timeframe})")
        axes[2].grid(True)
        axes[2].legend()
        
        plt.tight_layout()
        chart_path = output_path / f"btc_data_quality_{timeframe.replace('m', 'min').replace('h', 'hour').replace('d', 'day')}.png"
        plt.savefig(chart_path)
        plt.close()
        
        report["visualization_chart"] = str(chart_path)
    
    # 记录质量报告
    all_problems = []
    for category, problems in report["problems"].items():
        all_problems.extend(problems)
    
    if all_problems:
        logger.warning("检测到数据质量问题:")
        for i, problem in enumerate(all_problems, 1):
            logger.warning(f" {i}. {problem}")
    
    logger.info(f"数据质量评分: {overall_score:.2f}/1.00 ({report['quality_rating']})")
    logger.info(f"完整性: {completeness_score:.2f}/1.00")
    logger.info(f"连续性: {continuity_score:.2f}/1.00")
    logger.info(f"无异常值: {anomaly_free_score:.2f}/1.00")
    
    return report

def main():
    parser = argparse.ArgumentParser(description="BTC历史数据质量验证")
    parser.add_argument("--csv", required=True, type=str, help="CSV文件路径")
    parser.add_argument("--timeframe", type=str, default="1h", choices=list(TIMEFRAME_POINTS.keys()),
                       help="数据时间周期")
    parser.add_argument("--output-dir", type=str, default="btc_rl/logs/data_quality",
                       help="输出目录，用于保存报告和图表")
    parser.add_argument("--strict", action="store_true", help="严格模式：如果数据质量较差则返回非零退出代码")
    
    args = parser.parse_args()
    
    try:
        # 加载数据
        logger.info(f"正在加载数据文件: {args.csv}")
        df = load_data(args.csv)
        
        if df is None:
            logger.error("无法加载数据，退出")
            sys.exit(1)
        
        # 生成质量报告
        report = generate_quality_report(df, args.timeframe, args.output_dir)
        
        # 如果是严格模式且质量较差，返回非零退出代码
        if args.strict and report["overall_quality_score"] < 0.6:
            logger.error(f"数据质量低于阈值: {report['overall_quality_score']:.2f}/1.00")
            sys.exit(2)
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"数据验证过程出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
