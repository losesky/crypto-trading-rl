#!/usr/bin/env python3
"""
指标同步工具：确保不同系统的评估结果一致性

此脚本可以:
- 同步所有模型的指标文件
- 确保 model_comparison.py 和 show_model_metrics.py 使用相同的指标数据
"""

import os
import json
from pathlib import Path
import sys
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("metrics_sync")

# 添加项目根目录到sys.path，确保能正确导入项目中的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
sys.path.append(os.path.dirname(current_dir))  # 添加btc_rl/src目录

# 导入配置工具
from btc_rl.src.config import get_config

# 导入评估函数
try:
    from btc_rl.src.train_sac import evaluate_model_with_metrics
except ImportError:
    logger.error("无法导入evaluate_model_with_metrics函数")
    evaluate_model_with_metrics = None

def find_models():
    """获取models目录下所有可用的模型路径"""
    models_dir = Path(project_root) / "btc_rl" / "models"
    return list(models_dir.glob("*.zip"))

def synchronize_metrics():
    """同步所有模型指标文件，确保一致性"""
    logger.info("开始同步所有模型指标文件...")
    models = find_models()
    logger.info(f"找到 {len(models)} 个模型文件")
    
    # 从统一配置获取指标摘要文件路径
    config = get_config()
    metrics_summary_file = config.get_metrics_summary_file()
    
    metrics_dir = Path(project_root) / os.path.dirname(metrics_summary_file)
    metrics_dir.mkdir(exist_ok=True)
    
    # 创建模型摘要文件
    summary_file = Path(project_root) / metrics_summary_file
    summary_data = {
        "models": [],
        "sync_time": None,
        "total_models": len(models)
    }
    
    for model_path in models:
        model_name = model_path.stem
        logger.info(f"处理模型: {model_name}")
        
        # 构建指标文件路径
        metrics_file = metrics_dir / f"{model_name}_metrics.json"
        
        # 检查指标文件是否存在
        metrics_data = None
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                logger.info(f"已加载现有指标文件: {metrics_file}")
            except Exception as e:
                logger.error(f"读取指标文件失败: {e}")
        
        # 如果指标文件不存在或需要重新评估
        if metrics_data is None and evaluate_model_with_metrics is not None:
            logger.info(f"重新评估模型: {model_path}")
            try:
                # 调用评估函数，这将自动保存指标文件
                result = evaluate_model_with_metrics(str(model_path), save_metrics=True)
                if result:
                    logger.info(f"模型 {model_name} 评估成功: 夏普比率={result.get('sharpe_ratio', 0):.2f}, 胜率={result.get('win_rate', 0):.2%}")
                    
                    # 重新加载刚保存的指标文件
                    with open(metrics_file, 'r') as f:
                        metrics_data = json.load(f)
                else:
                    logger.error(f"模型 {model_name} 评估失败")
                    continue
            except Exception as e:
                logger.error(f"评估模型时出错: {e}")
                continue
        elif metrics_data is None:
            logger.error(f"缺少指标文件且无法评估: {model_name}")
            continue
        
        # 更新摘要数据
        model_summary = {
            "model_name": model_name,
            "model_path": str(model_path),
            "final_equity": metrics_data.get("final_equity", 10000.0),
            "total_return": metrics_data.get("total_return", 0.0),
            "max_drawdown": metrics_data.get("max_drawdown", 0.0),
            "sharpe_ratio": metrics_data.get("sharpe_ratio", 0.0),
            "sortino_ratio": metrics_data.get("sortino_ratio", 0.0),
            "total_trades": metrics_data.get("total_trades", 0),
            "win_rate": metrics_data.get("win_rate", 0.0)
        }
        summary_data["models"].append(model_summary)
    
    # 更新同步时间
    import datetime
    summary_data["sync_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 保存摘要文件
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"已更新模型摘要文件: {summary_file}")
    
    logger.info("模型指标同步完成")
    return len(summary_data["models"])

def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description="模型指标同步工具")
    parser.add_argument("--force", action="store_true", help="强制重新评估所有模型")
    args = parser.parse_args()
    
    if args.force:
        logger.info("强制模式: 将重新评估所有模型")
        # 如果是强制模式，删除所有现有指标文件
        metrics_dir = Path(project_root) / "btc_rl" / "metrics"
        for metrics_file in metrics_dir.glob("*_metrics.json"):
            os.remove(metrics_file)
            logger.info(f"已删除: {metrics_file}")
    
    # 同步指标
    model_count = synchronize_metrics()
    logger.info(f"成功同步 {model_count} 个模型的指标")

if __name__ == "__main__":
    main()
