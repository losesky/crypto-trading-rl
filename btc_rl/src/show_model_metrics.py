#!/usr/bin/env python3
"""
显示所有模型的统计指标，包括交易次数和胜率

此脚本可以:
- 显示已有的模型指标
- 重新评估模型并生成指标
- 可视化模型性能对比
"""

import os
import json
import argparse
from pathlib import Path
import tabulate
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import sys

# 导入模型评估功能
# 添加项目根目录到sys.path，确保能正确导入项目中的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
sys.path.append(os.path.dirname(current_dir))  # 添加btc_rl/src目录

try:
    # 使用相对导入方式，更可靠
    from btc_rl.src.train_sac import evaluate_model_with_metrics
except ImportError:
    try:
        # 尝试从当前目录导入
        from train_sac import evaluate_model_with_metrics
    except ImportError:
        print("警告: 无法导入evaluate_model_with_metrics函数，--evaluate选项将不可用")
        evaluate_model_with_metrics = None

def format_percent(value):
    """格式化百分比显示"""
    return f"{value * 100:.2f}%"

def format_currency(value):
    """格式化货币显示"""
    return f"${value:.2f}"

def load_model_metrics(metrics_dir=None):
    """加载所有模型的指标数据"""
    if metrics_dir is None:
        # 获取指标目录路径
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        project_root = current_dir.parent.parent
        metrics_dir = project_root / "btc_rl" / "metrics"
    else:
        metrics_dir = Path(metrics_dir)
    
    # 确保目录存在
    os.makedirs(metrics_dir, exist_ok=True)
    
    # 优先检查models_summary.json文件，它通常包含最准确的汇总指标
    summary_path = metrics_dir / "models_summary.json"
    summary_data = {}
    if summary_path.exists():
        try:
            with open(summary_path, 'r') as f:
                summary_data = json.load(f)
                print(f"已加载模型汇总数据: {summary_path}")
        except Exception as e:
            print(f"无法加载汇总文件 {summary_path}: {e}")
    
    # 如果是第一次运行并且没有指标文件，先验证模型文件存在
    model_files = list((project_root / "btc_rl" / "models").glob("*.zip"))
    if len(model_files) > 0 and not any(metrics_dir.glob("*_metrics.json")):
        print(f"发现 {len(model_files)} 个模型文件，但没有对应的指标文件。")
        print("您可能需要运行评估来生成指标文件。")
    
    # 加载所有指标文件
    metrics_data = []
    for file_path in metrics_dir.glob("*_metrics.json"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                model_name = data.get("model_name", "")
                
                # 如果文件中没有模型路径，添加可能的模型路径
                if "model_path" not in data:
                    possible_model_path = project_root / "btc_rl" / "models" / f"{model_name}.zip"
                    if possible_model_path.exists():
                        data["model_path"] = str(possible_model_path)
                
                # 确保交易次数和胜率是一致的：优先使用汇总文件中的数据
                if summary_data and "models" in summary_data:
                    for model_summary in summary_data["models"]:
                        if model_summary.get("model_name") == model_name:
                            # 更新交易次数和胜率，保持与汇总文件一致
                            if "total_trades" in model_summary:
                                data["total_trades"] = model_summary["total_trades"]
                            if "win_rate" in model_summary:
                                data["win_rate"] = model_summary["win_rate"]
                            print(f"已同步模型 {model_name} 的交易数据与汇总文件: 交易次数={model_summary.get('total_trades', 0)}, 胜率={model_summary.get('win_rate', 0)*100:.2f}%")
                            break
                
                metrics_data.append(data)
        except Exception as e:
            print(f"无法加载指标文件 {file_path}: {e}")
    
    # 按模型名称排序
    metrics_data.sort(key=lambda x: x.get("model_name", ""))
    
    return metrics_data

def show_metrics_table(metrics_data, show_full=False, max_dd=0.05, min_sortino=25.0, min_sharpe=12.0):
    """以表格形式显示所有模型的指标"""
    if not metrics_data:
        print("没有找到任何模型指标数据")
        return
    
    # 准备表格数据
    headers = ["模型", "最终权益", "总回报率", "最大回撤", "夏普比率", "索提诺比率", "交易次数", "胜率"]
    rows = []
    
    for data in metrics_data:
        model_name = data.get("model_name", "未知")
        final_equity = format_currency(data.get("final_equity", 0))
        total_return = format_percent(data.get("total_return", 0))
        max_drawdown = format_percent(data.get("max_drawdown", 0))
        sharpe_ratio = f"{data.get('sharpe_ratio', 0):.2f}"
        sortino_ratio = f"{data.get('sortino_ratio', 0):.2f}"
        total_trades = data.get("total_trades", 0)
        win_rate = format_percent(data.get("win_rate", 0))
        
        rows.append([
            model_name, final_equity, total_return, max_drawdown, 
            sharpe_ratio, sortino_ratio, total_trades, win_rate
        ])
    
    # 打印表格
    print(tabulate.tabulate(rows, headers=headers, tablefmt="pretty"))
    
    # 核心风控指标阈值
    MAX_DRAWDOWN_THRESHOLD = max_dd
    MIN_SORTINO_THRESHOLD = min_sortino
    MIN_SHARPE_THRESHOLD = min_sharpe
    
    # 筛选满足风控指标的模型
    risk_controlled_models = []
    for model in metrics_data:
        max_dd_value = model.get("max_drawdown", 1.0)
        sortino = model.get("sortino_ratio", 0.0)
        sharpe = model.get("sharpe_ratio", 0.0)
        
        # 检查是否满足所有风控指标要求
        if max_dd_value <= MAX_DRAWDOWN_THRESHOLD and sortino >= MIN_SORTINO_THRESHOLD and sharpe >= MIN_SHARPE_THRESHOLD:
            risk_controlled_models.append(model)
    
    # 显示最佳模型（不考虑风控指标）
    best_model = max(metrics_data, key=lambda x: x.get("total_return", 0))
    best_model_name = best_model.get("model_name", "未知")
    best_return = best_model.get("total_return", 0)
    print(f"\n最佳模型(仅按回报率): {best_model_name}，总回报率: {format_percent(best_return)}")
    
    # 显示满足风控指标的最佳模型
    if risk_controlled_models:
        rc_best_model = max(risk_controlled_models, key=lambda x: x.get("total_return", 0))
        rc_best_name = rc_best_model.get("model_name", "未知")
        rc_best_return = rc_best_model.get("total_return", 0)
        rc_max_dd = rc_best_model.get("max_drawdown", 0)
        rc_sortino = rc_best_model.get("sortino_ratio", 0)
        rc_sharpe = rc_best_model.get("sharpe_ratio", 0)
        
        print(f"\n符合风控要求的最佳模型: {rc_best_name}")
        print(f"  - 总回报率: {format_percent(rc_best_return)}")
        print(f"  - 最大回撤: {format_percent(rc_max_dd)} (阈值: ≤{MAX_DRAWDOWN_THRESHOLD*100:.1f}%)")
        print(f"  - 索提诺比率: {rc_sortino:.2f} (阈值: ≥{MIN_SORTINO_THRESHOLD:.1f})")
        print(f"  - 夏普比率: {rc_sharpe:.2f} (阈值: ≥{MIN_SHARPE_THRESHOLD:.1f})")
    else:
        print("\n没有模型满足所有风控指标要求:")
        print(f"  - 最大回撤阈值: ≤{MAX_DRAWDOWN_THRESHOLD*100:.1f}%")
        print(f"  - 索提诺比率阈值: ≥{MIN_SORTINO_THRESHOLD:.1f}")
        print(f"  - 夏普比率阈值: ≥{MIN_SHARPE_THRESHOLD:.1f}")
        
        # 寻找接近阈值的模型
        closest_model = min(metrics_data, key=lambda x: 
            abs(x.get("max_drawdown", 1.0) - MAX_DRAWDOWN_THRESHOLD) + 
            abs(MIN_SORTINO_THRESHOLD - x.get("sortino_ratio", 0.0))/25 + 
            abs(MIN_SHARPE_THRESHOLD - x.get("sharpe_ratio", 0.0))/12
        )
        
        print(f"\n最接近风控要求的模型: {closest_model.get('model_name', '未知')}")
        print(f"  - 总回报率: {format_percent(closest_model.get('total_return', 0))}")
        print(f"  - 最大回撤: {format_percent(closest_model.get('max_drawdown', 0))} (阈值: ≤{MAX_DRAWDOWN_THRESHOLD*100:.1f}%)")
        print(f"  - 索提诺比率: {closest_model.get('sortino_ratio', 0):.2f} (阈值: ≥{MIN_SORTINO_THRESHOLD:.1f})")
        print(f"  - 夏普比率: {closest_model.get('sharpe_ratio', 0):.2f} (阈值: ≥{MIN_SHARPE_THRESHOLD:.1f})")
    
    
    # 如果需要显示完整信息，打印每个模型的详细指标
    if show_full:
        for data in metrics_data:
            model_name = data.get("model_name", "未知")
            print(f"\n=== {model_name} 详细指标 ===")
            print(f"评估时间: {data.get('evaluation_time', '未知')}")
            print(f"最终权益: {format_currency(data.get('final_equity', 0))}")
            print(f"总回报率: {format_percent(data.get('total_return', 0))}")
            print(f"最大回撤: {format_percent(data.get('max_drawdown', 0))}")
            print(f"夏普比率: {data.get('sharpe_ratio', 0):.4f}")
            print(f"索提诺比率: {data.get('sortino_ratio', 0):.4f}")
            print(f"交易次数: {data.get('total_trades', 0)}")
            print(f"胜率: {format_percent(data.get('win_rate', 0))}")
            
            # 显示对应的模型文件路径
            model_path = data.get("model_path", "")
            if model_path:
                print(f"模型文件路径: {model_path}")
            else:
                possible_path = f"btc_rl/models/{model_name}.zip"
                print(f"推测模型路径: {possible_path}")
            
            # 显示最近的交易历史（如果有）
            if "history" in data and data["history"]:
                print("\n最近的交易历史:")
                for i, point in enumerate(data["history"], 1):
                    print(f"{i}. 步骤: {point.get('step', 0)}, "
                          f"价格: {point.get('price', 0):.2f}, "
                          f"仓位: {point.get('position_btc', 0):.6f}, "
                          f"权益: {point.get('margin_equity', 0):.2f}")

def plot_metrics(metrics_data):
    """绘制模型指标对比图"""
    if not metrics_data:
        print("没有找到任何模型指标数据")
        return
    
    # 提取数据
    models = [data.get("model_name", "未知") for data in metrics_data]
    returns = [data.get("total_return", 0) * 100 for data in metrics_data]  # 转换为百分比
    drawdowns = [data.get("max_drawdown", 0) * 100 for data in metrics_data]  # 转换为百分比
    sharpes = [data.get("sharpe_ratio", 0) for data in metrics_data]
    sortinos = [data.get("sortino_ratio", 0) for data in metrics_data]
    trades = [data.get("total_trades", 0) for data in metrics_data]
    win_rates = [data.get("win_rate", 0) * 100 for data in metrics_data]  # 转换为百分比
    
    # 创建2x3的子图布局
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("模型性能指标对比", fontsize=16)
    
    # 1. 总回报率
    axes[0, 0].bar(models, returns, color='green')
    axes[0, 0].set_title("总回报率 (%)")
    axes[0, 0].set_ylabel("百分比")
    axes[0, 0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(returns):
        axes[0, 0].text(i, v + 1, f"{v:.1f}%", ha='center')
    
    # 2. 最大回撤
    axes[0, 1].bar(models, drawdowns, color='red')
    axes[0, 1].set_title("最大回撤 (%)")
    axes[0, 1].set_ylabel("百分比")
    axes[0, 1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(drawdowns):
        axes[0, 1].text(i, v + 0.5, f"{v:.1f}%", ha='center')
    
    # 3. 夏普比率
    axes[0, 2].bar(models, sharpes, color='blue')
    axes[0, 2].set_title("夏普比率")
    axes[0, 2].tick_params(axis='x', rotation=45)
    for i, v in enumerate(sharpes):
        axes[0, 2].text(i, v + 0.5, f"{v:.2f}", ha='center')
    
    # 4. 索提诺比率
    axes[1, 0].bar(models, sortinos, color='purple')
    axes[1, 0].set_title("索提诺比率")
    axes[1, 0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(sortinos):
        axes[1, 0].text(i, v + 0.5, f"{v:.2f}", ha='center')
    
    # 5. 交易次数
    axes[1, 1].bar(models, trades, color='orange')
    axes[1, 1].set_title("交易次数")
    axes[1, 1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(trades):
        axes[1, 1].text(i, v + 0.5, f"{v}", ha='center')
    
    # 6. 胜率
    axes[1, 2].bar(models, win_rates, color='teal')
    axes[1, 2].set_title("胜率 (%)")
    axes[1, 2].set_ylabel("百分比")
    axes[1, 2].tick_params(axis='x', rotation=45)
    for i, v in enumerate(win_rates):
        axes[1, 2].text(i, v + 2, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # 保存图片
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    project_root = current_dir.parent.parent
    plots_dir = project_root / "btc_rl" / "metrics" / "plots"
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = plots_dir / f"model_metrics_{timestamp}.png"
    plt.savefig(plot_path)
    print(f"图表已保存到: {plot_path}")
    
    # 显示图表
    plt.show()

def evaluate_all_models(models_dir=None):
    """重新评估所有模型并生成指标
    
    注意: 这个函数计算的交易次数和胜率将作为系统中的标准指标数据。
    其他部分(如model_comparison.py通过WebSocket计算的数据)将使用这里生成的数据以确保一致性。
    """
    global evaluate_model_with_metrics
    
    # 检查评估函数是否可用
    if evaluate_model_with_metrics is None:
        print("错误: 评估函数未正确加载，无法进行评估")
        print("尝试重新导入评估函数...")
        
        try:
            # 再次尝试导入
            from btc_rl.src.train_sac import evaluate_model_with_metrics
        except ImportError:
            try:
                from train_sac import evaluate_model_with_metrics
            except ImportError:
                print("错误: 无法导入evaluate_model_with_metrics函数，无法进行评估")
                return False
    
    if evaluate_model_with_metrics is None:
        print("错误: 评估函数仍然不可用，无法继续")
        return False
    
    # 查找模型目录
    if models_dir is None:
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        project_root = current_dir.parent.parent
        models_dir = project_root / "btc_rl" / "models"
    else:
        models_dir = Path(models_dir)
    
    # 确保目录存在
    if not models_dir.exists():
        print(f"错误: 模型目录不存在: {models_dir}")
        return False
    
    # 查找所有模型文件
    model_files = list(models_dir.glob("*.zip"))
    if not model_files:
        print(f"错误: 在 {models_dir} 中找不到模型文件")
        return False
    
    print(f"找到 {len(model_files)} 个模型文件，开始评估...")
    
    # 逐个评估模型
    success_count = 0
    for model_path in sorted(model_files):
        try:
            print(f"\n评估模型: {model_path.name}")
            stats = evaluate_model_with_metrics(str(model_path), save_metrics=True)
            if stats:
                print(f"模型 {model_path.name} 评估完成:")
                print(f"- 总回报率: {format_percent(stats.get('total_return', 0))}")
                print(f"- 交易次数: {stats.get('total_trades', 0)}")
                print(f"- 胜率: {format_percent(stats.get('win_rate', 0))}")
                success_count += 1
        except Exception as e:
            print(f"评估模型 {model_path.name} 时出错: {e}")
    
    print(f"\n评估完成，成功评估 {success_count}/{len(model_files)} 个模型")
    return success_count > 0

def fix_winrates_in_metrics_files(metrics_dir=None):
    """修复所有指标文件中的胜率数据
    
    注意: 此函数将修复所有指标文件中的胜率数据，并确保它们与models_summary.json一致。
    这是保证系统中各处显示的交易统计数据一致性的关键功能。
    """
    global evaluate_model_with_metrics
    
    # 检查评估函数是否可用
    if evaluate_model_with_metrics is None:
        print("错误: 评估函数未正确加载，无法修复胜率数据")
        print("尝试重新导入评估函数...")
        
        try:
            # 再次尝试导入
            from btc_rl.src.train_sac import evaluate_model_with_metrics
        except ImportError:
            try:
                from train_sac import evaluate_model_with_metrics
            except ImportError:
                print("错误: 无法导入evaluate_model_with_metrics函数，无法进行胜率修复")
                return False
    
    if evaluate_model_with_metrics is None:
        print("错误: 评估函数仍然不可用，无法继续")
        return False
        
    # 查找指标目录
    if metrics_dir is None:
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        project_root = current_dir.parent.parent
        metrics_dir = project_root / "btc_rl" / "metrics"
        models_dir = project_root / "btc_rl" / "models"
    else:
        metrics_dir = Path(metrics_dir)
        models_dir = Path(os.path.dirname(metrics_dir)) / "models"
    
    # 确保目录存在
    if not metrics_dir.exists():
        print(f"错误: 指标目录不存在: {metrics_dir}")
        return False
        
    if not models_dir.exists():
        print(f"错误: 模型目录不存在: {models_dir}")
        return False
    
    # 查找所有指标文件
    metrics_files = list(metrics_dir.glob("*_metrics.json"))
    if not metrics_files:
        print(f"错误: 在 {metrics_dir} 中找不到指标文件")
        return False
        
    print(f"找到 {len(metrics_files)} 个指标文件，开始修复胜率数据...")
    
    # 逐个修复指标文件
    success_count = 0
    for metrics_file in sorted(metrics_files):
        try:
            # 提取模型名称
            model_name = metrics_file.stem.replace("_metrics", "")
            
            # 加载指标数据
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
            
            # 查找对应的模型文件
            model_path = models_dir / f"{model_name}.zip"
            if not model_path.exists():
                print(f"警告: 找不到模型文件 {model_path}")
                # 尝试使用指标文件中的模型路径
                if "model_path" in metrics_data and os.path.exists(metrics_data["model_path"]):
                    model_path = metrics_data["model_path"]
                else:
                    print(f"跳过 {model_name}，找不到有效的模型文件")
                    continue
            
            print(f"\n修复模型 {model_name} 的胜率数据...")
            
            # 重新评估模型以获取正确的胜率
            stats = evaluate_model_with_metrics(model_path, save_metrics=False)
            if not stats:
                print(f"无法评估模型 {model_name}")
                continue
            
            # 更新胜率数据
            metrics_data["win_rate"] = stats["win_rate"]
            metrics_data["total_trades"] = stats["total_trades"]
            
            # 保存更新的指标数据
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            print(f"模型 {model_name} 胜率数据已更新: 交易次数={stats['total_trades']}, 胜率={stats['win_rate']*100:.2f}%")
            success_count += 1
            
            # 同时更新models_summary.json文件以保持一致性
            summary_path = os.path.join(os.path.dirname(metrics_file), "models_summary.json")
            if os.path.exists(summary_path):
                try:
                    # 读取当前汇总
                    with open(summary_path, 'r') as f:
                        summary_data = json.load(f)
                    
                    # 更新相应模型的数据
                    if "models" in summary_data:
                        for model_entry in summary_data["models"]:
                            if model_entry.get("model_name") == model_name:
                                model_entry["total_trades"] = stats["total_trades"]
                                model_entry["win_rate"] = stats["win_rate"]
                                print(f"已更新汇总文件中 {model_name} 的交易数据")
                                break
                    
                    # 保存更新后的汇总
                    with open(summary_path, 'w') as f:
                        json.dump(summary_data, f, indent=2)
                    print(f"模型汇总文件 {summary_path} 已更新")
                except Exception as e:
                    print(f"更新模型汇总文件时出错: {e}")
            
        except Exception as e:
            print(f"修复模型 {metrics_file.stem} 胜率数据时出错: {e}")
    
    print(f"\n胜率数据修复完成，成功修复 {success_count}/{len(metrics_files)} 个模型")
    return success_count > 0

def main():
    parser = argparse.ArgumentParser(description="显示所有模型的统计指标")
    parser.add_argument("--full", action="store_true", help="显示完整的模型指标详情")
    parser.add_argument("--plot", action="store_true", help="绘制模型指标对比图")
    parser.add_argument("--dir", type=str, help="指标文件所在目录", default=None)
    parser.add_argument("--evaluate", action="store_true", help="重新评估所有模型并生成指标")
    parser.add_argument("--fix-winrate", action="store_true", help="专门修复指标文件中的胜率数据")
    
    # 风控指标阈值参数
    parser.add_argument("--max-dd", type=float, default=0.05, help="最大回撤阈值（默认：0.05，即5%）")
    parser.add_argument("--min-sortino", type=float, default=25.0, help="索提诺比率最小阈值（默认：25）")
    parser.add_argument("--min-sharpe", type=float, default=12.0, help="夏普比率最小阈值（默认：12）")
    
    args = parser.parse_args()
    
    # 如果请求修复胜率，先执行修复
    if args.fix_winrate:
        print("开始修复所有模型的胜率数据...")
        if fix_winrates_in_metrics_files():
            print("胜率数据修复完成，将显示更新后的指标")
        else:
            print("胜率数据修复失败或部分失败")
    
    # 如果请求评估，执行评估
    if args.evaluate:
        print("开始重新评估所有模型...")
        if evaluate_all_models():
            print("评估完成，将显示最新指标")
        else:
            print("评估失败或部分失败，将显示现有指标")
    
    # 修复胜率数据
    if args.fix_winrate:
        print("开始修复胜率数据...")
        fix_winrates_in_metrics_files()
    
    # 读取并显示指标
    metrics_data = load_model_metrics(args.dir)
    show_metrics_table(metrics_data, args.full, args.max_dd, args.min_sortino, args.min_sharpe)
    
    if args.plot:
        try:
            plot_metrics(metrics_data)
        except Exception as e:
            print(f"绘图失败: {e}")
            import traceback
            print(traceback.format_exc())

if __name__ == "__main__":
    main()
