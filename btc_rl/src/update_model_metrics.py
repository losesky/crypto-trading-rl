#!/usr/bin/env python3
"""
历史兼容工具: 更新旧版模型指标文件，添加缺失的交易数据

注意:
    自2025年6月11日起，模型评估功能已升级，会自动保存交易数据。
    本脚本仅作为历史兼容工具，用于更新在此日期之前生成的旧版指标文件。
    对于新评估的模型，不再需要手动运行此脚本。

使用方法:
    python btc_rl/src/update_model_metrics.py --model MODEL_NAME
    
    可选参数:
    --force: 强制更新，即使文件已包含交易数据
"""

import os
import sys
import json
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("update_model_metrics")

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

def update_metrics_file(model_name):
    """
    更新指定模型的指标文件，添加缺失的交易数据
    
    Args:
        model_name (str): 模型名称
        
    注意:
        新版模型评估已经自动保存交易数据，此脚本仅用于更新旧版模型指标文件。
        如果您正在使用新版评估系统，无需手动运行此脚本。
    """
    logger.info("注意: 新版模型评估已经自动保存交易数据，此脚本仅用于更新旧版指标文件")
    
    metrics_file = f"btc_rl/metrics/{model_name}_metrics.json"
    
    try:
        # 检查文件是否存在
        if not os.path.exists(metrics_file):
            logger.error(f"指标文件不存在: {metrics_file}")
            return False
        
        # 读取指标文件
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)
        
        # 检查是否已有交易数据
        if 'trades' in metrics_data and len(metrics_data['trades']) > 0:
            logger.info(f"模型 {model_name} 已有交易数据 ({len(metrics_data['trades'])} 笔交易)")
            return True
            
        # 从历史数据中生成交易记录
        history_data = metrics_data.get('history', [])
        
        if not history_data:
            logger.warning(f"模型 {model_name} 没有历史数据")
            return False
            
        # 提取交易记录
        position_changes = []
        for i in range(1, len(history_data)):
            prev_pos = history_data[i-1].get("position_btc", 0)
            curr_pos = history_data[i].get("position_btc", 0)
            
            # 检测仓位变化
            if abs(curr_pos - prev_pos) > 0.000001:
                entry_price = history_data[i].get("price", 0)
                entry_time = history_data[i].get("timestamp", i)  # 使用时间戳或索引
                position_changes.append({
                    "index": i,
                    "size_change": curr_pos - prev_pos,
                    "price": entry_price,
                    "time": entry_time
                })
        
        # 生成交易记录
        trades = []
        for i, change in enumerate(position_changes):
            # 跳过最后一个仓位变化，因为没有对应的平仓
            if i >= len(position_changes) - 1:
                continue
                
            # 获取开仓信息
            entry_idx = change["index"]
            entry_price = change["price"]
            size_change = change["size_change"]
            side = "long" if size_change > 0 else "short"
            
            # 获取平仓信息
            exit_change = position_changes[i+1]
            exit_idx = exit_change["index"]
            exit_price = exit_change["price"]
            
            # 计算持仓时间（以小时为单位，假设每个数据点间隔1小时）
            duration = exit_idx - entry_idx
            
            # 计算利润
            if side == "long":  # 做多
                profit = (exit_price - entry_price) * abs(size_change)
                profit_pct = (exit_price - entry_price) / entry_price
            else:  # 做空
                profit = (entry_price - exit_price) * abs(size_change)
                profit_pct = (entry_price - exit_price) / entry_price
                
            # 创建交易记录
            trade = {
                "open_time": entry_idx * 3600,  # 假设每小时一个数据点
                "close_time": exit_idx * 3600,
                "open_price": entry_price,
                "close_price": exit_price,
                "side": side,
                "size": abs(size_change),
                "profit": profit,
                "return_pct": profit_pct,
                "duration": duration
            }
            
            trades.append(trade)
            
        # 更新指标文件
        if trades:
            logger.info(f"为模型 {model_name} 生成了 {len(trades)} 笔交易记录")
            metrics_data["trades"] = trades
            
            # 计算权益曲线
            if "equity_curve" not in metrics_data or not metrics_data["equity_curve"]:
                equity_curve = [point.get("margin_equity", 10000.0) for point in history_data]
                metrics_data["equity_curve"] = equity_curve
                logger.info(f"为模型 {model_name} 生成了 {len(equity_curve)} 点权益曲线")
            
            # 计算回撤
            if "drawdowns" not in metrics_data or not metrics_data["drawdowns"]:
                from btc_rl.src.model_comparison import calculate_drawdowns
                equity_curve = metrics_data.get("equity_curve", [])
                if equity_curve:
                    drawdowns, max_dd = calculate_drawdowns(equity_curve)
                    metrics_data["drawdowns"] = drawdowns
                    metrics_data["max_drawdown"] = max_dd
                    logger.info(f"为模型 {model_name} 生成了 {len(drawdowns)} 点回撤数据")
            
            # 保存更新后的文件
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
            logger.info(f"已更新模型 {model_name} 的指标文件")
            return True
        else:
            logger.warning(f"模型 {model_name} 没有检测到交易")
            return False
            
    except Exception as e:
        logger.error(f"更新指标文件时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description="更新旧版模型指标文件 (添加交易数据)")
    parser.add_argument("--model", required=True, help="模型名称")
    parser.add_argument("--force", action="store_true", help="强制更新，即使文件已包含交易数据")
    args = parser.parse_args()
    
    # 检查指标文件是否已包含交易数据
    metrics_file = f"btc_rl/metrics/{args.model}_metrics.json"
    if os.path.exists(metrics_file) and not args.force:
        try:
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
                
            if 'trades' in metrics_data and len(metrics_data['trades']) > 0:
                logger.info(f"指标文件 {metrics_file} 已包含 {len(metrics_data['trades'])} 笔交易数据")
                logger.info("如果要强制更新，请使用 --force 参数")
                return 0
        except Exception as e:
            logger.warning(f"读取文件时出错: {e}")
    
    # 执行更新
    if update_metrics_file(args.model):
        logger.info(f"成功更新指标文件: {metrics_file}")
        return 0
    else:
        logger.error(f"更新指标文件失败: {metrics_file}")
        return 1

if __name__ == "__main__":
    logger.info("=====================================================")
    logger.info("  交易数据自动提取工具 (仅用于更新旧版指标文件)")
    logger.info("  注意: 新版评估系统会自动保存交易数据，无需此工具")
    logger.info("=====================================================")
    sys.exit(main())
