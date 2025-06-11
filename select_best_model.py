#!/usr/bin/env python3
"""
使用黄金法则评估标准选择最佳BTC交易模型
单一指标权重≤20%，结合回撤持续时间、盈亏比、策略容量三维验证
"""

import json
import logging
import os
import sys
from tabulate import tabulate

from btc_rl.src.model_comparison import get_best_model_by_golden_rule
from btc_rl.src.config import get_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("select_best_model")

def format_percent(value):
    """格式化百分比显示"""
    return f"{value * 100:.2f}%" if isinstance(value, (int, float)) else value

def format_currency(value):
    """格式化货币显示"""
    return f"${value:.2f}" if isinstance(value, (int, float)) else value

def main():
    """主函数"""
    try:
        # 获取最佳模型
        best_model_info = get_best_model_by_golden_rule()
        
        if not best_model_info:
            logger.warning("没有找到满足条件的最佳模型，请检查筛选条件或模型质量")
            print("\n" + "="*80)
            print(" "*20 + "警告: 没有找到满足条件的最佳模型")
            print("="*80 + "\n")
            print("可能的原因:")
            print("1. 模型质量不符合筛选条件")
            print("2. 筛选条件设置过严（检查config.ini中的model_selection部分）")
            print("3. 训练未完全完成或出现错误")
            print("\n建议操作:")
            print("1. 检查日志文件确认训练是否完成")
            print("2. 适当放宽筛选条件（如降低minimum_win_rate或minimum_sharpe）")
            print("3. 重新运行训练或仅重新评估现有模型")
            return 1
        
        # 打印模型评估结果
        print("\n" + "="*80)
        print(" "*25 + "BTC交易模型黄金法则评估结果")
        print("="*80 + "\n")
        
        # 打印最佳模型信息
        print(f"最佳模型: {best_model_info['model_name']}")
        print(f"模型路径: {best_model_info['model_path']}")
        print(f"综合评分: {best_model_info['golden_rule_score']:.4f}")
        print()
        
        # 打印维度评分
        print("维度评分明细:")
        dimension_scores = best_model_info['dimensions']
        for dimension, score in dimension_scores.items():
            print(f"  {dimension:<10}: {score:.4f}")
        print()
        
        # 打印关键指标
        print("关键指标:")
        print(f"  总回报率: {format_percent(best_model_info['total_return'])}")
        print(f"  最大回撤: {format_percent(best_model_info['max_drawdown'])}")
        print(f"  夏普比率: {best_model_info['sharpe_ratio']:.2f}")
        print(f"  卡玛比率: {best_model_info['calmar_ratio']:.2f}")
        print(f"  盈亏比  : {best_model_info['profit_loss_ratio']:.2f}")
        print(f"  胜率    : {format_percent(best_model_info['win_rate'])}")
        print(f"  最终权益: {format_currency(best_model_info['final_equity'] if 'final_equity' in best_model_info else 0)}")
        print()
        
        # 打印模型排名表格
        print("模型排名（按黄金法则综合评分排序）:")
        all_models = best_model_info['ranked_models']
        
        # 定义表头，与analyze_metrics.sh输出格式一致
        headers = ["模型", "最终权益", "总回报率", "最大回撤", "夏普比率", "索提诺比率", "总费用", "胜率", "综合评分"]
        
        # 准备表格数据
        rows = []
        # 使用从model_comparison.py返回的已排序模型列表（已按黄金法则评分排序）
        sorted_models = all_models
        
        for model in sorted_models:
            rows.append([
                model.get('model_name', ''),
                format_currency(model.get('final_equity', 0)),
                format_percent(model.get('total_return', 0)),
                format_percent(model.get('max_drawdown', 0)),
                f"{model.get('sharpe_ratio', 0):.2f}",
                f"{model.get('sortino_ratio', 0):.2f}",
                format_currency(model.get('total_fees', 0)),
                format_percent(model.get('win_rate', 0)),
                f"{model.get('golden_rule_score', 0):.4f}"
            ])
        
        # 使用tabulate生成漂亮的表格
        print(tabulate(rows, headers=headers, tablefmt="pretty"))
        
        # 显示筛选条件
        config = get_config()
        print("\n" + "="*80)
        print(" "*10 + "模型筛选条件")
        print("="*80)
        print(f"最低总回报率: ≥ {config.getfloat('model_selection', 'minimum_return', fallback=0.5)*100:.0f}%")
        print(f"最大可接受回撤: ≤ {config.getfloat('model_selection', 'maximum_drawdown', fallback=0.20)*100:.0f}%")
        print(f"最低夏普比率: ≥ {config.getfloat('model_selection', 'minimum_sharpe', fallback=4.0):.1f}")
        print(f"最低胜率(硬性要求): ≥ {config.getfloat('model_selection', 'minimum_win_rate', fallback=0.30)*100:.0f}%")
        
        print("\n" + "="*80)
        print(" "*10 + "黄金法则: 单一指标权重≤20%，结合多维度全面评估交易策略")
        print("="*80 + "\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"执行时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
