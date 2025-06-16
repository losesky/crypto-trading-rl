#!/usr/bin/env python3
"""
模型适应性分析和调整工具 - 分析模型在不同环境中的表现并自动调整参数
"""
import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# 添加项目根目录到路径中
root_path = str(Path(__file__).parent.parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

# 导入项目相关模块
from trading_system.src.adaptive_risk import AdaptiveRiskController
from trading_system.src.model_wrapper import ModelWrapper

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("trading_system/logs", f"model_adaptation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    ]
)
logger = logging.getLogger("model_adaptation")

def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    参数:
    - config_path: 配置文件路径
    
    返回:
    - 配置字典
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"已加载配置文件: {config_path}")
        return config
    except Exception as e:
        logger.error(f"无法加载配置文件: {e}")
        sys.exit(1)

def load_trade_data(data_dir: str) -> pd.DataFrame:
    """
    加载交易数据
    
    参数:
    - data_dir: 数据目录
    
    返回:
    - 交易数据DataFrame
    """
    try:
        trades_dir = Path(data_dir) / "collected_data/trades"
        if not trades_dir.exists():
            logger.warning(f"交易数据目录不存在: {trades_dir}")
            return pd.DataFrame()
        
        # 加载所有交易文件
        trade_files = list(trades_dir.glob("trade_*.json"))
        if not trade_files:
            logger.warning("未找到交易记录文件")
            return pd.DataFrame()
        
        # 合并所有交易数据
        trades_data = []
        for file_path in trade_files:
            try:
                with open(file_path, 'r') as f:
                    trade_data = json.load(f)
                trades_data.append(trade_data)
            except Exception as e:
                logger.warning(f"无法加载交易文件 {file_path.name}: {e}")
        
        if not trades_data:
            logger.warning("未能成功加载任何交易数据")
            return pd.DataFrame()
        
        # 转换为DataFrame
        df = pd.DataFrame(trades_data)
        logger.info(f"成功加载了 {len(df)} 条交易记录")
        return df
    except Exception as e:
        logger.error(f"加载交易数据时发生错误: {e}")
        return pd.DataFrame()

def load_experiences_data(data_dir: str) -> pd.DataFrame:
    """
    加载交易经验数据
    
    参数:
    - data_dir: 数据目录
    
    返回:
    - 经验数据DataFrame
    """
    try:
        experiences_dir = Path(data_dir) / "collected_data/experiences"
        if not experiences_dir.exists():
            logger.warning(f"经验数据目录不存在: {experiences_dir}")
            return pd.DataFrame()
        
        # 加载所有经验文件
        exp_files = list(experiences_dir.glob("exp_batch_*.json"))
        if not exp_files:
            logger.warning("未找到经验记录文件")
            return pd.DataFrame()
        
        # 合并所有经验数据
        all_experiences = []
        for file_path in exp_files:
            try:
                with open(file_path, 'r') as f:
                    experiences = json.load(f)
                all_experiences.extend(experiences)
            except Exception as e:
                logger.warning(f"无法加载经验文件 {file_path.name}: {e}")
        
        if not all_experiences:
            logger.warning("未能成功加载任何经验数据")
            return pd.DataFrame()
        
        # 转换为DataFrame
        df = pd.DataFrame(all_experiences)
        logger.info(f"成功加载了 {len(df)} 条经验记录")
        return df
    except Exception as e:
        logger.error(f"加载经验数据时发生错误: {e}")
        return pd.DataFrame()

def load_model_metrics(data_dir: str) -> Dict[str, Any]:
    """
    加载模型指标
    
    参数:
    - data_dir: 数据目录
    
    返回:
    - 模型指标字典
    """
    try:
        metrics_file = Path(data_dir) / "collected_data/model_metrics.json"
        if not metrics_file.exists():
            logger.warning(f"模型指标文件不存在: {metrics_file}")
            return {}
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        logger.info(f"已加载模型指标: {metrics['model_version']}")
        return metrics
    except Exception as e:
        logger.error(f"加载模型指标时发生错误: {e}")
        return {}

def analyze_environment_compatibility(trades_df: pd.DataFrame, experiences_df: pd.DataFrame) -> Dict[str, Any]:
    """
    分析模型与环境的兼容性
    
    参数:
    - trades_df: 交易数据
    - experiences_df: 经验数据
    
    返回:
    - 兼容性分析结果字典
    """
    result = {
        "compatibility_score": 0.0,
        "environment_regime": "unknown",
        "suggestions": [],
        "risk_adjustments": {},
        "preprocessing_adjustments": {}
    }
    
    # 如果数据不足，无法进行分析
    if trades_df.empty or len(trades_df) < 10:
        result["suggestions"].append("交易数据不足，无法进行详细分析")
        return result
    
    try:
        # 计算胜率
        if "is_profitable" in trades_df.columns:
            win_rate = trades_df["is_profitable"].mean()
            result["win_rate"] = win_rate
            
            # 基于胜率的兼容性初步评估
            if win_rate >= 0.6:
                result["compatibility_score"] = 0.8
                result["suggestions"].append("模型在当前环境中表现良好，胜率超过60%")
            elif win_rate >= 0.5:
                result["compatibility_score"] = 0.6
                result["suggestions"].append("模型在当前环境中表现尚可，胜率超过50%")
            elif win_rate >= 0.4:
                result["compatibility_score"] = 0.4
                result["suggestions"].append("模型在当前环境中表现一般，考虑适当调整风险参数")
            else:
                result["compatibility_score"] = 0.2
                result["suggestions"].append("模型在当前环境中表现不佳，建议显著降低风险或重新训练")
        
        # 分析交易结果与市场环境的关系
        if "entry_time" in trades_df.columns and "exit_time" in trades_df.columns:
            # 将时间字符串转换为时间戳
            if isinstance(trades_df["entry_time"].iloc[0], str):
                trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
            if isinstance(trades_df["exit_time"].iloc[0], str):
                trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])
            
            # 计算持仓时长
            trades_df["duration"] = (trades_df["exit_time"] - trades_df["entry_time"]).dt.total_seconds() / 3600  # 小时
            
            # 分析持仓时长与盈利的关系
            if "profit_pct" in trades_df.columns:
                # 计算持仓时长与利润的相关性
                duration_profit_corr = trades_df[["duration", "profit_pct"]].corr().iloc[0, 1]
                result["duration_profit_correlation"] = duration_profit_corr
                
                if abs(duration_profit_corr) >= 0.3:
                    if duration_profit_corr > 0:
                        result["suggestions"].append("长时间持仓往往带来更多利润，考虑延长持仓时间")
                        result["preprocessing_adjustments"]["trend_weight"] = 1.2  # 增加趋势因子权重
                    else:
                        result["suggestions"].append("短时间持仓往往带来更多利润，考虑缩短持仓时间")
                        result["preprocessing_adjustments"]["volatility_weight"] = 1.2  # 增加波动因子权重
        
        # 分析市场环境
        if not experiences_df.empty and "market_data" in experiences_df.columns:
            # 提取价格数据
            try:
                # 如果market_data是字典或JSON字符串
                if isinstance(experiences_df["market_data"].iloc[0], dict):
                    prices = experiences_df["market_data"].apply(lambda x: x.get("close", 0)).values
                elif isinstance(experiences_df["market_data"].iloc[0], str):
                    prices = experiences_df["market_data"].apply(lambda x: json.loads(x).get("close", 0)).values
                else:
                    prices = []
                
                if len(prices) > 10:
                    # 计算价格变化
                    price_changes = np.diff(prices) / prices[:-1]
                    
                    # 计算波动率
                    volatility = np.std(price_changes) * np.sqrt(24)  # 年化到日波动率
                    
                    # 计算价格变化方向一致性
                    direction_consistency = np.sum(np.sign(price_changes)) / len(price_changes)
                    trend_strength = abs(direction_consistency)
                    
                    result["market_volatility"] = float(volatility)
                    result["trend_strength"] = float(trend_strength)
                    
                    # 确定市场环境
                    if volatility > 0.03:  # 高波动率
                        if trend_strength > 0.6:
                            result["environment_regime"] = "trending_volatile"
                            result["suggestions"].append("市场呈现高波动强趋势特征，建议适度增加风险偏好")
                            result["risk_adjustments"]["risk_per_trade_pct"] = 1.1  # 增加单笔风险
                        else:
                            result["environment_regime"] = "volatile"
                            result["suggestions"].append("市场呈现高波动特征，建议降低风险偏好")
                            result["risk_adjustments"]["risk_per_trade_pct"] = 0.8  # 降低单笔风险
                    else:  # 低波动率
                        if trend_strength > 0.7:
                            result["environment_regime"] = "trending"
                            result["suggestions"].append("市场呈现低波动强趋势特征，建议保持当前风险偏好")
                            result["risk_adjustments"]["risk_per_trade_pct"] = 1.0  # 保持当前风险
                        else:
                            result["environment_regime"] = "ranging"
                            result["suggestions"].append("市场呈现低波动震荡特征，建议降低风险偏好并减小持仓规模")
                            result["risk_adjustments"]["risk_per_trade_pct"] = 0.9  # 稍微降低风险
                            result["risk_adjustments"]["max_leverage"] = 0.9  # 降低杠杆
            except Exception as e:
                logger.warning(f"分析市场环境时出错: {e}")
        
        # 分析模型预测与实际结果的一致性
        if "action" in experiences_df.columns and "reward" in experiences_df.columns:
            # 计算动作与奖励的相关性
            action_reward_corr = experiences_df[["action", "reward"]].corr().iloc[0, 1]
            result["action_reward_correlation"] = action_reward_corr
            
            if action_reward_corr < 0.1:
                result["suggestions"].append("模型的动作与实际奖励相关性较弱，建议重新检查奖励计算方式")
                result["preprocessing_adjustments"]["reward_scaling"] = 1.5  # 增加奖励缩放以强化学习信号
    except Exception as e:
        logger.error(f"分析环境兼容性时出错: {e}")
        result["suggestions"].append(f"分析过程中发生错误: {str(e)}")
    
    return result

def adjust_model_parameters(config: Dict[str, Any], analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据分析结果调整模型参数
    
    参数:
    - config: 原始配置
    - analysis_result: 分析结果
    
    返回:
    - 调整后的配置
    """
    adjusted_config = config.copy()
    
    # 调整风险参数
    risk_adjustments = analysis_result.get("risk_adjustments", {})
    if risk_adjustments and "trading" in adjusted_config:
        trading_config = adjusted_config["trading"]
        
        # 调整风险参数
        if "risk_per_trade_pct" in risk_adjustments:
            adjustment = risk_adjustments["risk_per_trade_pct"]
            current_risk = trading_config.get("risk_per_trade_pct", 0.02)
            trading_config["risk_per_trade_pct"] = max(0.005, min(0.05, current_risk * adjustment))
            logger.info(f"调整风险参数: {current_risk} -> {trading_config['risk_per_trade_pct']}")
        
        # 调整杠杆
        if "max_leverage" in risk_adjustments:
            adjustment = risk_adjustments["max_leverage"]
            current_leverage = trading_config.get("max_leverage", 3)
            trading_config["max_leverage"] = max(1, min(5, current_leverage * adjustment))
            logger.info(f"调整最大杠杆: {current_leverage} -> {trading_config['max_leverage']}")
    
    # 添加预处理调整参数
    preprocessing_adjustments = analysis_result.get("preprocessing_adjustments", {})
    if preprocessing_adjustments:
        if "preprocessing" not in adjusted_config:
            adjusted_config["preprocessing"] = {}
        
        # 复制调整参数
        for key, value in preprocessing_adjustments.items():
            adjusted_config["preprocessing"][key] = value
            logger.info(f"设置预处理参数: {key} = {value}")
    
    # 添加市场环境信息
    if "environment_regime" in analysis_result:
        if "market_analysis" not in adjusted_config:
            adjusted_config["market_analysis"] = {}
        adjusted_config["market_analysis"]["current_regime"] = analysis_result["environment_regime"]
        
        # 添加最后更新时间
        adjusted_config["market_analysis"]["last_updated"] = datetime.now().isoformat()
    
    return adjusted_config

def save_adjusted_config(config: Dict[str, Any], config_path: str) -> None:
    """
    保存调整后的配置
    
    参数:
    - config: 调整后的配置
    - config_path: 配置文件路径
    """
    # 备份原始配置
    backup_path = config_path + f".bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        import shutil
        shutil.copy2(config_path, backup_path)
        logger.info(f"已备份原始配置到: {backup_path}")
    except Exception as e:
        logger.warning(f"备份原始配置失败: {e}")
    
    # 保存新配置
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"已保存调整后的配置到: {config_path}")
    except Exception as e:
        logger.error(f"保存调整后的配置失败: {e}")

def create_adaptation_report(analysis_result: Dict[str, Any], trades_df: pd.DataFrame, output_path: str) -> None:
    """
    创建适应性分析报告
    
    参数:
    - analysis_result: 分析结果
    - trades_df: 交易数据
    - output_path: 输出路径
    """
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 创建报告HTML
        with open(output_path, 'w') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>模型适应性分析报告</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
        }}
        .section {{
            margin-bottom: 30px;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }}
        .metric {{
            margin-bottom: 15px;
        }}
        .metric-title {{
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 1.2em;
        }}
        .suggestions {{
            background-color: #e8f4f8;
            padding: 15px;
            border-left: 4px solid #3498db;
        }}
        .suggestion-item {{
            margin-bottom: 10px;
        }}
        .good {{
            color: #27ae60;
        }}
        .medium {{
            color: #f39c12;
        }}
        .poor {{
            color: #c0392b;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>模型适应性分析报告</h1>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="section">
            <h2>兼容性概述</h2>
            <div class="metric">
                <div class="metric-title">兼容性评分:</div>
                <div class="metric-value {get_score_class(analysis_result.get('compatibility_score', 0))}">{analysis_result.get('compatibility_score', 0):.2f} / 1.0</div>
            </div>
            <div class="metric">
                <div class="metric-title">当前市场环境:</div>
                <div class="metric-value">{analysis_result.get('environment_regime', 'unknown').replace('_', ' ').title()}</div>
            </div>
            <div class="metric">
                <div class="metric-title">交易胜率:</div>
                <div class="metric-value {get_win_rate_class(analysis_result.get('win_rate', 0))}">{analysis_result.get('win_rate', 0):.2%}</div>
            </div>
        </div>
        
        <div class="section">
            <h2>建议调整</h2>
            <div class="suggestions">
                {"".join(f'<div class="suggestion-item">• {s}</div>' for s in analysis_result.get('suggestions', ['无调整建议']))}
            </div>
        </div>
        
        <div class="section">
            <h2>参数调整</h2>
            <h3>风险参数</h3>
            <table>
                <tr>
                    <th>参数</th>
                    <th>调整系数</th>
                    <th>说明</th>
                </tr>
                {"".join(generate_adjustment_rows(analysis_result.get('risk_adjustments', {})))}
            </table>
            
            <h3>预处理参数</h3>
            <table>
                <tr>
                    <th>参数</th>
                    <th>调整值</th>
                    <th>说明</th>
                </tr>
                {"".join(generate_adjustment_rows(analysis_result.get('preprocessing_adjustments', {})))}
            </table>
        </div>
        
        <div class="section">
            <h2>市场分析</h2>
            <div class="metric">
                <div class="metric-title">市场波动率:</div>
                <div class="metric-value">{analysis_result.get('market_volatility', 0):.4f}</div>
            </div>
            <div class="metric">
                <div class="metric-title">趋势强度:</div>
                <div class="metric-value">{analysis_result.get('trend_strength', 0):.4f}</div>
            </div>
            <div class="metric">
                <div class="metric-title">动作-奖励相关性:</div>
                <div class="metric-value">{analysis_result.get('action_reward_correlation', 0):.4f}</div>
            </div>
        </div>
        
        <div class="section">
            <h2>交易摘要</h2>
            <table>
                <tr>
                    <th>指标</th>
                    <th>值</th>
                </tr>
                <tr>
                    <td>总交易次数</td>
                    <td>{len(trades_df)}</td>
                </tr>
                <tr>
                    <td>盈利交易次数</td>
                    <td>{trades_df['is_profitable'].sum() if 'is_profitable' in trades_df.columns else 'N/A'}</td>
                </tr>
                <tr>
                    <td>亏损交易次数</td>
                    <td>{len(trades_df) - trades_df['is_profitable'].sum() if 'is_profitable' in trades_df.columns else 'N/A'}</td>
                </tr>
                <tr>
                    <td>平均盈利</td>
                    <td>{trades_df[trades_df['is_profitable'] == True]['profit_pct'].mean() if 'is_profitable' in trades_df.columns and 'profit_pct' in trades_df.columns else 'N/A':.2%}</td>
                </tr>
                <tr>
                    <td>平均亏损</td>
                    <td>{trades_df[trades_df['is_profitable'] == False]['profit_pct'].mean() if 'is_profitable' in trades_df.columns and 'profit_pct' in trades_df.columns else 'N/A':.2%}</td>
                </tr>
                <tr>
                    <td>平均持仓时间</td>
                    <td>{trades_df['duration'].mean() if 'duration' in trades_df.columns else 'N/A':.2f} 小时</td>
                </tr>
            </table>
        </div>
    </div>
</body>
</html>
""")
        
        logger.info(f"已生成适应性分析报告: {output_path}")
    except Exception as e:
        logger.error(f"生成报告失败: {e}")

def get_score_class(score):
    """根据分数返回CSS类"""
    if score >= 0.7:
        return "good"
    elif score >= 0.4:
        return "medium"
    else:
        return "poor"

def get_win_rate_class(win_rate):
    """根据胜率返回CSS类"""
    if win_rate >= 0.55:
        return "good"
    elif win_rate >= 0.45:
        return "medium"
    else:
        return "poor"

def generate_adjustment_rows(adjustments):
    """生成调整表格行"""
    if not adjustments:
        return '<tr><td colspan="3">无调整参数</td></tr>'
    
    rows = []
    for param, value in adjustments.items():
        explanation = get_adjustment_explanation(param, value)
        rows.append(f'<tr><td>{param}</td><td>{value}</td><td>{explanation}</td></tr>')
    
    return ''.join(rows)

def get_adjustment_explanation(param, value):
    """获取参数调整的解释"""
    explanations = {
        "risk_per_trade_pct": "每笔交易风险比例",
        "max_leverage": "最大杠杆倍数",
        "trend_weight": "趋势因子权重",
        "volatility_weight": "波动因子权重",
        "reward_scaling": "奖励缩放系数"
    }
    
    base_explanation = explanations.get(param, "参数调整")
    
    if isinstance(value, (int, float)):
        if value > 1:
            return f"{base_explanation} (增加 {(value-1)*100:.0f}%)"
        elif value < 1:
            return f"{base_explanation} (减少 {(1-value)*100:.0f}%)"
        else:
            return f"{base_explanation} (保持不变)"
    else:
        return base_explanation

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="模型适应性分析和调整工具")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--data-dir", type=str, help="数据目录")
    parser.add_argument("--output", type=str, help="报告输出路径")
    parser.add_argument("--apply", action="store_true", help="是否应用调整")
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 确定数据目录
    data_dir = args.data_dir
    if not data_dir:
        data_dir = config.get('general', {}).get('data_dir', 'trading_system/data')
    
    # 确定输出路径
    output_path = args.output
    if not output_path:
        reports_dir = Path(data_dir).parent / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(reports_dir / f"model_adaptation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    
    # 加载数据
    trades_df = load_trade_data(data_dir)
    experiences_df = load_experiences_data(data_dir)
    metrics = load_model_metrics(data_dir)
    
    # 分析环境兼容性
    logger.info("开始分析模型与环境的兼容性...")
    analysis_result = analyze_environment_compatibility(trades_df, experiences_df)
    
    # 调整模型参数
    if args.apply:
        logger.info("根据分析结果调整模型参数...")
        adjusted_config = adjust_model_parameters(config, analysis_result)
        
        # 保存调整后的配置
        save_adjusted_config(adjusted_config, args.config)
        
        logger.info("参数调整已应用")
    else:
        logger.info("生成调整建议，但不应用（使用--apply选项应用调整）")
        
    # 创建报告
    logger.info("生成适应性分析报告...")
    create_adaptation_report(analysis_result, trades_df, output_path)
    
    logger.info(f"""
模型适应性分析完成:
- 兼容性评分: {analysis_result.get('compatibility_score', 0):.2f}/1.0
- 市场环境: {analysis_result.get('environment_regime', 'unknown')}
- 建议数量: {len(analysis_result.get('suggestions', []))}
- 报告已保存至: {output_path}
""")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
