#!/usr/bin/env python3
"""
分析收集的交易数据和模型性能
此脚本分析交易系统收集的实际交易经验和结果，评估模型性能，并提供改进建议
"""

import argparse
import json
import logging
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any

# 将项目路径添加到系统路径
script_dir = Path(__file__).absolute().parent
root_path = script_dir.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("analyze_trading_data")

def load_trades_data(trades_dir: Path) -> pd.DataFrame:
    """加载交易数据"""
    trades = []
    
    if not trades_dir.exists():
        logger.warning(f"交易数据目录不存在: {trades_dir}")
        return pd.DataFrame()
    
    trade_files = list(trades_dir.glob("trade_*.json"))
    if not trade_files:
        logger.warning("没有找到交易数据文件")
        return pd.DataFrame()
    
    logger.info(f"发现{len(trade_files)}个交易数据文件")
    
    for file_path in trade_files:
        try:
            with open(file_path, 'r') as f:
                trade_data = json.load(f)
                trades.append(trade_data)
        except Exception as e:
            logger.warning(f"无法读取交易数据文件 {file_path.name}: {e}")
    
    if not trades:
        return pd.DataFrame()
    
    # 转换为DataFrame
    df = pd.DataFrame(trades)
    
    # 转换时间列
    time_columns = ['entry_time', 'exit_time', 'record_time']
    for col in time_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                logger.warning(f"无法转换{col}列为datetime")
    
    # 排序
    if 'entry_time' in df.columns:
        df = df.sort_values('entry_time')
    
    return df

def load_experiences_data(experiences_dir: Path) -> pd.DataFrame:
    """加载交易经验数据"""
    experiences = []
    
    if not experiences_dir.exists():
        logger.warning(f"交易经验目录不存在: {experiences_dir}")
        return pd.DataFrame()
    
    exp_files = list(experiences_dir.glob("exp_batch_*.json"))
    if not exp_files:
        logger.warning("没有找到交易经验文件")
        return pd.DataFrame()
    
    logger.info(f"发现{len(exp_files)}个交易经验批次文件")
    
    for file_path in exp_files:
        try:
            with open(file_path, 'r') as f:
                exp_batch = json.load(f)
                experiences.extend(exp_batch)
        except Exception as e:
            logger.warning(f"无法读取交易经验文件 {file_path.name}: {e}")
    
    if not experiences:
        return pd.DataFrame()
    
    # 转换为DataFrame
    # 注意：经验数据结构可能很复杂，这里进行扁平化处理
    flattened_experiences = []
    for exp in experiences:
        flat_exp = {
            'timestamp': exp.get('timestamp'),
            'action': exp.get('action'),
            'reward': exp.get('reward'),
            'price': exp.get('market_data', {}).get('close', 0),
            'volume': exp.get('market_data', {}).get('volume', 0),
            'position_size': exp.get('position_data', {}).get('size', 0),
            'position_side': exp.get('position_data', {}).get('side', ''),
            'unrealized_pnl': exp.get('position_data', {}).get('unrealized_pnl', 0)
        }
        flattened_experiences.append(flat_exp)
    
    df = pd.DataFrame(flattened_experiences)
    
    # 转换时间列
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        except:
            logger.warning("无法转换timestamp列为datetime")
    
    # 排序
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
    
    return df

def calculate_metrics(trades_df: pd.DataFrame) -> Dict[str, Any]:
    """计算交易性能指标"""
    if trades_df.empty:
        return {}
    
    # 基本指标
    metrics = {}
    
    # 计算总交易次数
    metrics['total_trades'] = len(trades_df)
    
    # 计算盈利和亏损交易次数
    if 'is_profitable' in trades_df.columns:
        profitable_trades = trades_df[trades_df['is_profitable'] == True]
        loss_trades = trades_df[trades_df['is_profitable'] == False]
        
        metrics['profitable_trades'] = len(profitable_trades)
        metrics['loss_trades'] = len(loss_trades)
        metrics['win_rate'] = len(profitable_trades) / len(trades_df) if len(trades_df) > 0 else 0
    
    # 计算平均收益和亏损
    if 'profit_pct' in trades_df.columns:
        metrics['avg_profit_pct'] = trades_df['profit_pct'].mean() if 'profit_pct' in trades_df.columns else 0
        
        if 'is_profitable' in trades_df.columns:
            metrics['avg_win_pct'] = profitable_trades['profit_pct'].mean() if not profitable_trades.empty else 0
            metrics['avg_loss_pct'] = loss_trades['profit_pct'].mean() if not loss_trades.empty else 0
            
            # 计算盈亏比
            if metrics['avg_loss_pct'] != 0 and len(loss_trades) > 0:
                metrics['profit_loss_ratio'] = abs(metrics['avg_win_pct'] / metrics['avg_loss_pct']) if metrics['avg_loss_pct'] != 0 else 0
    
    # 计算累计收益
    if 'profit_pct' in trades_df.columns:
        metrics['cumulative_return'] = (1 + trades_df['profit_pct']).prod() - 1
    
    # 计算最大回撤
    if 'profit_pct' in trades_df.columns:
        # 计算权益曲线
        equity_curve = (1 + trades_df['profit_pct']).cumprod()
        
        # 计算最大回撤
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax
        metrics['max_drawdown'] = abs(drawdown.min()) if not drawdown.empty else 0
    
    # 计算夏普比率（简化版）
    if 'profit_pct' in trades_df.columns:
        returns = trades_df['profit_pct']
        metrics['sharpe_ratio'] = returns.mean() / returns.std() * (252 ** 0.5) if returns.std() != 0 else 0
    
    # 计算卡玛比率
    if 'cumulative_return' in metrics and 'max_drawdown' in metrics and metrics['max_drawdown'] > 0:
        metrics['calmar_ratio'] = metrics['cumulative_return'] / metrics['max_drawdown'] if metrics['max_drawdown'] > 0 else 0
    
    return metrics

def analyze_model_performance(metrics: Dict[str, Any], trades_df: pd.DataFrame, experiences_df: pd.DataFrame) -> List[str]:
    """分析模型性能并提供改进建议"""
    suggestions = []
    
    # 检查胜率
    if 'win_rate' in metrics:
        win_rate = metrics['win_rate']
        if win_rate < 0.4:
            suggestions.append(f"模型胜率较低({win_rate:.2%})，建议：")
            suggestions.append("- 增加训练中的风险惩罚，提高对错误交易的敏感度")
            suggestions.append("- 考虑在环境中增加止损机制，降低单次亏损金额")
        elif win_rate > 0.6:
            suggestions.append(f"模型胜率较高({win_rate:.2%})，但请检查：")
            suggestions.append("- 平均盈利是否远小于平均亏损（可能是'啄米'策略）")
            suggestions.append("- 交易策略是否有足够的样本量，避免过拟合")
    
    # 检查盈亏比
    if 'profit_loss_ratio' in metrics:
        pl_ratio = metrics['profit_loss_ratio']
        if pl_ratio < 0.8:
            suggestions.append(f"盈亏比较低({pl_ratio:.2f})，建议：")
            suggestions.append("- 调整止盈策略，让盈利可以跑得更远")
            suggestions.append("- 考虑使用追踪止损机制")
        elif pl_ratio > 2.0:
            suggestions.append(f"盈亏比较高({pl_ratio:.2f})，很好，但需要检查：")
            suggestions.append("- 止损是否过早触发，导致胜率过低")
            suggestions.append("- 大额盈利是否来自少量幸运交易（尾部效应）")
    
    # 检查总回报率
    if 'cumulative_return' in metrics:
        cum_return = metrics['cumulative_return']
        if cum_return < 0:
            suggestions.append(f"总回报为负({cum_return:.2%})，建议：")
            suggestions.append("- 重新检查模型训练参数和奖励函数设置")
            suggestions.append("- 考虑增加仓位管理和风险控制机制")
        elif cum_return > 0 and cum_return < 0.05:
            suggestions.append(f"总回报较低({cum_return:.2%})，建议：")
            suggestions.append("- 增加模型对市场趋势的敏感度")
            suggestions.append("- 考虑在训练中增加对回报率的权重")
    
    # 检查最大回撤
    if 'max_drawdown' in metrics:
        max_dd = metrics['max_drawdown']
        if max_dd > 0.2:
            suggestions.append(f"最大回撤较大({max_dd:.2%})，建议：")
            suggestions.append("- 改进资金管理策略，减少单次交易风险")
            suggestions.append("- 在训练中增加对回撤的惩罚")
            suggestions.append("- 考虑增加对市场风险状态的识别特征")
    
    # 检查交易频率（如果有时间信息）
    if not trades_df.empty and 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
        try:
            # 计算交易持续时间
            trades_df['duration'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 3600  # 小时
            avg_duration = trades_df['duration'].mean()
            
            if avg_duration < 2:  # 小于2小时
                suggestions.append(f"平均持仓时间较短({avg_duration:.2f}小时)，建议：")
                suggestions.append("- 检查模型是否过度交易或过度拟合噪声")
                suggestions.append("- 考虑增加交易成本在奖励函数中的权重")
            elif avg_duration > 48:  # 大于48小时
                suggestions.append(f"平均持仓时间较长({avg_duration:.2f}小时)，建议：")
                suggestions.append("- 检查模型是否对市场变化反应不够敏感")
                suggestions.append("- 考虑增加短期信号在特征中的权重")
        except:
            pass
    
    # 基于交易经验的分析
    if not experiences_df.empty:
        # 分析模型决策与奖励的关系
        try:
            action_reward_corr = experiences_df[['action', 'reward']].corr().iloc[0, 1]
            if abs(action_reward_corr) < 0.2:
                suggestions.append(f"模型动作与奖励相关性较低({action_reward_corr:.2f})，建议：")
                suggestions.append("- 检查奖励函数设计是否合理")
                suggestions.append("- 考虑增加特征工程，提高模型对市场的理解能力")
        except:
            pass
    
    # 如果没有特定问题，添加一般性建议
    if not suggestions:
        suggestions.append("模型表现良好，一般改进建议：")
        suggestions.append("- 持续收集交易数据，定期重新训练模型")
        suggestions.append("- 考虑增加更多市场微观结构特征")
        suggestions.append("- 实验不同的风险参数设置，寻找最优组合")
    
    return suggestions

def plot_performance(trades_df: pd.DataFrame, save_dir: Path = None):
    """绘制交易性能图表"""
    if trades_df.empty:
        logger.warning("没有交易数据，无法绘制图表")
        return
    
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('ggplot')
    
    # 设置matplotlib中文显示
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        logger.warning("设置中文显示失败，图表中的中文可能无法正常显示")
    
    # 图1：权益曲线
    if 'profit_pct' in trades_df.columns:
        plt.figure(figsize=(12, 6))
        
        # 计算累计权益曲线
        equity_curve = (1 + trades_df['profit_pct']).cumprod()
        
        plt.plot(range(len(equity_curve)), equity_curve, linewidth=2)
        plt.title('交易系统权益曲线', fontsize=14)
        plt.xlabel('交易次数', fontsize=12)
        plt.ylabel('权益倍数', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 添加初始值和最终值标记
        plt.scatter(0, equity_curve.iloc[0], c='green', s=50, zorder=5)
        plt.scatter(len(equity_curve) - 1, equity_curve.iloc[-1], c='red', s=50, zorder=5)
        
        # 添加初始值和最终值文本
        plt.annotate(f'起始: {equity_curve.iloc[0]:.2f}', 
                    (0, equity_curve.iloc[0]), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
        
        plt.annotate(f'最终: {equity_curve.iloc[-1]:.2f}', 
                    (len(equity_curve) - 1, equity_curve.iloc[-1]), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
        
        if save_dir:
            plt.savefig(save_dir / 'equity_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 图2：盈亏比例饼图
    if 'is_profitable' in trades_df.columns:
        plt.figure(figsize=(8, 8))
        
        profitable_count = len(trades_df[trades_df['is_profitable'] == True])
        loss_count = len(trades_df[trades_df['is_profitable'] == False])
        
        labels = ['盈利交易', '亏损交易']
        sizes = [profitable_count, loss_count]
        colors = ['#66b3ff', '#ff9999']
        explode = (0.1, 0)  # 突出第一个扇形
        
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        plt.axis('equal')  # 保证饼图为正圆
        plt.title('交易盈亏比例', fontsize=14)
        
        if save_dir:
            plt.savefig(save_dir / 'profit_loss_pie.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 图3：收益分布直方图
    if 'profit_pct' in trades_df.columns:
        plt.figure(figsize=(12, 6))
        
        plt.hist(trades_df['profit_pct'], bins=20, edgecolor='black', alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('交易收益分布', fontsize=14)
        plt.xlabel('收益率', fontsize=12)
        plt.ylabel('频数', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if save_dir:
            plt.savefig(save_dir / 'profit_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

def check_directory_structure(trades_dir: Path, experiences_dir: Path) -> List[str]:
    """检查目录结构完整性"""
    missing_dirs = []
    if not trades_dir.exists():
        missing_dirs.append(str(trades_dir))
    if not experiences_dir.exists():
        missing_dirs.append(str(experiences_dir))
    return missing_dirs

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="分析交易数据和模型性能")
    parser.add_argument("--data-dir", type=str, default="/home/losesky/crypto-trading-rl/trading_system/data/collected_data",
                      help="数据收集目录路径")
    parser.add_argument("--reports-dir", type=str, default="/home/losesky/crypto-trading-rl/trading_system/reports",
                      help="报告保存目录")
    parser.add_argument("--generate-plots", action="store_true", help="是否生成图表")
    parser.add_argument("--initialize", action="store_true", help="初始化目录结构")
    
    args = parser.parse_args()
    
    # 转换为Path对象
    data_dir = Path(args.data_dir)
    reports_dir = Path(args.reports_dir)
    
    # 创建报告目录
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化数据收集目录结构（如果指定了--initialize标志）
    if args.initialize:
        logger.info("正在初始化数据收集目录结构...")
        # 创建数据收集相关目录
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "trades").mkdir(exist_ok=True)
        (data_dir / "experiences").mkdir(exist_ok=True)
        (data_dir / "model_backups").mkdir(exist_ok=True)
        
        # 创建一个空的模型性能文件以初始化结构
        metrics_file = data_dir / "model_metrics.json"
        if not metrics_file.exists():
            with open(metrics_file, 'w') as f:
                json.dump({
                    "model_version": "初始化",
                    "trades_count": 0,
                    "profitable_trades": 0,
                    "loss_trades": 0,
                    "total_profit": 0,
                    "total_loss": 0
                }, f, indent=2)
            logger.info(f"创建了初始模型指标文件: {metrics_file}")
        
        logger.info("数据收集目录结构已初始化，请开始交易系统以收集真实交易数据")
        return 0
    
    # 数据子目录
    trades_dir = data_dir / "trades"
    experiences_dir = data_dir / "experiences"
    
    # 检查数据目录是否存在
    if not data_dir.exists():
        logger.warning(f"数据收集目录不存在: {data_dir}")
        logger.warning("请先运行交易系统收集真实交易数据，或者使用 --initialize 初始化目录结构")
        return 1
    
    # 检查必要的子目录是否存在
    missing_dirs = check_directory_structure(trades_dir, experiences_dir)
    if missing_dirs:
        logger.warning(f"以下必要的数据目录不存在: {', '.join(missing_dirs)}")
        logger.warning("请先运行交易系统收集真实交易数据，或者使用 --initialize 初始化目录结构")
        
        # 创建报告说明如何开始数据收集
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        guide_file = reports_dir / f"setup_guide_{timestamp}.txt"
        
        with open(guide_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(" "*20 + "交易数据收集指南\n")
            f.write("="*80 + "\n\n")
            
            f.write("无法找到交易数据。要开始收集真实交易数据，请按照以下步骤操作：\n\n")
            
            f.write("1. 初始化数据收集目录结构：\n")
            f.write("   ./trading_system/scripts/analyze_model_performance.sh --initialize\n\n")
            
            f.write("2. 启动测试交易系统开始收集数据：\n")
            f.write("   ./trading_system/scripts/start_test_trading.sh\n\n")
            
            f.write("3. 让系统运行一段时间（至少完成10-20笔交易）以收集足够的数据\n\n")
            
            f.write("4. 再次运行分析脚本查看结果：\n")
            f.write("   ./trading_system/scripts/analyze_model_performance.sh\n\n")
            
            f.write("注意：系统需要收集真实的交易数据才能进行有意义的分析。请确保交易系统正常运行并产生交易决策。\n")
        
        logger.info(f"已生成设置指南: {guide_file}")
        return 1
    
    # 加载交易数据
    logger.info("正在加载交易数据...")
    trades_df = load_trades_data(trades_dir)
    if trades_df.empty:
        logger.warning("未找到有效的交易数据，目录存在但可能为空")
    else:
        logger.info(f"成功加载{len(trades_df)}条交易记录")
    
    logger.info("正在加载交易经验数据...")
    experiences_df = load_experiences_data(experiences_dir)
    if experiences_df.empty:
        logger.warning("未找到有效的交易经验数据，目录存在但可能为空")
    else:
        logger.info(f"成功加载{len(experiences_df)}条交易经验记录")
    
    # 加载模型性能指标（如果存在）
    metrics_file = data_dir / "model_metrics.json"
    model_metrics = {}
    if metrics_file.exists():
        try:
            with open(metrics_file, 'r') as f:
                model_metrics = json.load(f)
            logger.info("成功加载模型性能指标")
        except Exception as e:
            logger.warning(f"无法加载模型性能指标文件: {e}")
    
    # 计算交易指标
    if not trades_df.empty:
        logger.info("计算交易性能指标...")
        trade_metrics = calculate_metrics(trades_df)
        
        # 合并指标
        metrics = {**model_metrics, **trade_metrics}
    else:
        metrics = model_metrics
    
    # 检查是否有足够的数据进行分析
    has_data = not trades_df.empty or not experiences_df.empty or metrics_file.exists()
    
    if not has_data:
        logger.warning("没有找到任何数据用于分析")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        no_data_report = reports_dir / f"no_data_report_{timestamp}.txt"
        
        with open(no_data_report, 'w') as f:
            f.write("="*80 + "\n")
            f.write(" "*20 + "交易数据缺失报告\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("未能找到任何交易数据或模型性能指标。请按照以下步骤开始数据收集：\n\n")
            
            f.write("1. 确保交易系统配置正确：\n")
            f.write("   - 检查 test_config.json 中的各项设置\n")
            f.write("   - 确保 model_update_config.json 中启用了数据收集功能\n\n")
            
            f.write("2. 启动测试交易系统：\n")
            f.write("   ./trading_system/scripts/start_test_trading.sh\n\n")
            
            f.write("3. 运行足够长的时间（建议至少12-24小时）\n\n")
            
            f.write("4. 再次运行此分析脚本\n\n")
            
            f.write("如果问题持续存在，请检查:\n")
            f.write("1. 交易系统日志是否有错误信息\n")
            f.write("2. model_wrapper.py中的数据收集功能是否正常工作\n")
            f.write("3. 是否有足够的市场波动触发交易决策\n")
        
        logger.info(f"已生成数据缺失报告: {no_data_report}")
        return 1
    
    # 分析模型性能并提供改进建议
    logger.info("分析模型性能并生成改进建议...")
    suggestions = analyze_model_performance(metrics, trades_df, experiences_df)
    
    # 生成报告
    logger.info("生成分析报告...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = reports_dir / f"trading_analysis_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(" "*20 + "交易系统模型性能分析报告\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 写入模型信息
        f.write("模型信息:\n")
        f.write(f"  模型版本: {model_metrics.get('model_version', '未知')}\n")
        f.write("\n")
        
        # 写入交易统计
        f.write("交易统计:\n")
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, float):
                    # 如果是百分比类型的指标
                    if key in ['win_rate', 'avg_profit_pct', 'avg_win_pct', 'avg_loss_pct', 'cumulative_return', 'max_drawdown']:
                        f.write(f"  {key}: {value:.2%}\n")
                    else:
                        f.write(f"  {key}: {value:.4f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
        else:
            f.write("  无可用交易统计数据\n")
        f.write("\n")
        
        # 写入改进建议
        f.write("模型改进建议:\n")
        for suggestion in suggestions:
            f.write(f"{suggestion}\n")
        f.write("\n")
        
        # 写入数据统计
        f.write("数据统计:\n")
        f.write(f"  交易记录数: {len(trades_df) if not trades_df.empty else 0}\n")
        f.write(f"  交易经验记录数: {len(experiences_df) if not experiences_df.empty else 0}\n")
    
    logger.info(f"分析报告已保存至: {report_file}")
    
    # 如果需要，生成图表
    if args.generate_plots and not trades_df.empty:
        logger.info("生成性能图表...")
        plots_dir = reports_dir / f"plots_{timestamp}"
        plot_performance(trades_df, plots_dir)
        logger.info(f"图表已保存至: {plots_dir}")
    
    logger.info("分析完成!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
