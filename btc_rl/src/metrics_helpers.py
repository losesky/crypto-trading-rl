"""
确保指标评估一致性的配置补丁文件
在执行主脚本时导入这个模块
"""
import os
import json
import logging
from btc_rl.src.config import get_config

logger = logging.getLogger("metrics_sync")

def load_metrics_config():
    """加载指标配置，决定是否使用同步的指标数据"""
    # 使用统一配置文件
    config = get_config()
    
    metrics_config = {
        "use_synchronized_metrics": config.get_use_synchronized_metrics(),
        "prefer_metrics_file": config.get_prefer_metrics_file(),
        "metrics_summary_file": config.get_metrics_summary_file()
    }
    
    logger.info(f"已加载指标配置: {metrics_config}")
    return metrics_config

def load_summary_metrics():
    """加载汇总指标文件，获取所有模型的关键统计数据"""
    # 确定项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    config = load_metrics_config()
    if not config.get("use_synchronized_metrics", True):
        logger.info("未启用同步指标功能，跳过加载汇总文件")
        return {}
    
    summary_file = os.path.join(project_root, config.get("metrics_summary_file", "btc_rl/metrics/models_summary.json"))
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                logger.info(f"已加载模型汇总数据: {summary_file}")
                return summary
        except Exception as e:
            logger.warning(f"无法加载汇总指标文件: {e}")
    else:
        logger.warning(f"汇总指标文件不存在: {summary_file}")
        
    return {}

def get_model_metrics_from_summary(model_name, summary_data=None):
    """从汇总文件中获取特定模型的指标"""
    if summary_data is None:
        summary_data = load_summary_metrics()
    
    if not summary_data or "models" not in summary_data:
        return None
    
    for model in summary_data.get("models", []):
        if model.get("model_name") == model_name:
            return {
                "final_equity": model.get("final_equity", 10000.0),
                "total_return": model.get("total_return", 0.0),
                "max_drawdown": model.get("max_drawdown", 0.0),
                "sharpe_ratio": model.get("sharpe_ratio", 0.0),
                "sortino_ratio": model.get("sortino_ratio", 0.0),
                "total_trades": model.get("total_trades", 0),
                "win_rate": model.get("win_rate", 0.0),
                "total_fees": model.get("total_fees", 0.0),
            }
    
    return None
