#!/usr/bin/env python3
"""
模型监控报告生成脚本 - 根据收集的交易和系统数据，生成自适应性能监控报告
"""
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import base64
from io import BytesIO

# 添加项目根目录到路径中
root_path = str(Path(__file__).parent.parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("trading_system/logs", f"report_generator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    ]
)
logger = logging.getLogger("report_generator")

class ModelMonitorReportGenerator:
    """模型监控报告生成器"""
    
    def __init__(self, config_path: str, data_dir: str, output_dir: str):
        """
        初始化报告生成器
        
        参数:
        - config_path: 配置文件路径
        - data_dir: 数据目录路径
        - output_dir: 输出目录路径
        """
        self.logger = logger
        self.config_path = config_path
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 加载模板
        self.template_path = Path(root_path) / "trading_system/templates/model_monitor_report_template.html"
        if not self.template_path.exists():
            self.logger.error(f"报告模板不存在: {self.template_path}")
            raise FileNotFoundError(f"找不到报告模板: {self.template_path}")
        
        # 存储上次报告的数据，用于计算变化
        self.previous_data = self._load_previous_report_data()
        
        # 当前数据
        self.current_data = {}
        
        self.logger.info("报告生成器初始化完成")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.logger.info(f"已加载配置文件: {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            return {}
    
    def _load_previous_report_data(self) -> Dict[str, Any]:
        """加载上一次报告的数据"""
        try:
            # 查找最新的报告数据文件
            data_files = list(self.output_dir.glob("report_data_*.json"))
            if not data_files:
                self.logger.info("未找到以前的报告数据")
                return {}
            
            # 按修改时间排序，获取最新的文件
            latest_file = sorted(data_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
            
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            self.logger.info(f"已加载上次报告数据: {latest_file}")
            return data
        except Exception as e:
            self.logger.warning(f"加载上次报告数据失败: {e}")
            return {}
    
    def _load_trades_data(self) -> pd.DataFrame:
        """加载交易数据"""
        try:
            trades_dir = self.data_dir / "collected_data/trades"
            if not trades_dir.exists():
                self.logger.warning(f"交易数据目录不存在: {trades_dir}")
                return pd.DataFrame()
            
            # 加载所有交易文件
            trade_files = list(trades_dir.glob("trade_*.json"))
            if not trade_files:
                self.logger.warning("未找到交易记录文件")
                return pd.DataFrame()
            
            # 合并所有交易数据
            trades_data = []
            for file_path in trade_files:
                try:
                    with open(file_path, 'r') as f:
                        trade_data = json.load(f)
                    trades_data.append(trade_data)
                except Exception as e:
                    self.logger.warning(f"无法加载交易文件 {file_path.name}: {e}")
            
            if not trades_data:
                self.logger.warning("未能成功加载任何交易数据")
                return pd.DataFrame()
            
            # 转换为DataFrame
            df = pd.DataFrame(trades_data)
            
            # 转换日期列
            if 'entry_time' in df.columns and isinstance(df['entry_time'].iloc[0], str):
                df['entry_time'] = pd.to_datetime(df['entry_time'])
            if 'exit_time' in df.columns and isinstance(df['exit_time'].iloc[0], str):
                df['exit_time'] = pd.to_datetime(df['exit_time'])
            
            self.logger.info(f"成功加载了 {len(df)} 条交易记录")
            return df
        except Exception as e:
            self.logger.error(f"加载交易数据失败: {e}")
            return pd.DataFrame()
    
    def _load_error_stats(self) -> Dict[str, Any]:
        """加载错误统计数据"""
        try:
            error_stats_file = self.data_dir / "collected_data/error_stats.json"
            if not error_stats_file.exists():
                self.logger.warning(f"错误统计文件不存在: {error_stats_file}")
                return {}
            
            with open(error_stats_file, 'r') as f:
                error_stats = json.load(f)
            
            self.logger.info(f"已加载错误统计数据: {error_stats_file}")
            return error_stats
        except Exception as e:
            self.logger.warning(f"加载错误统计数据失败: {e}")
            return {}
    
    def _load_model_metrics(self) -> Dict[str, Any]:
        """加载模型指标"""
        try:
            metrics_file = self.data_dir / "collected_data/model_metrics.json"
            if not metrics_file.exists():
                self.logger.warning(f"模型指标文件不存在: {metrics_file}")
                return {}
            
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            self.logger.info(f"已加载模型指标: {metrics_file}")
            return metrics
        except Exception as e:
            self.logger.warning(f"加载模型指标失败: {e}")
            return {}
    
    def _calculate_metrics(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """
        计算报告所需的指标
        
        参数:
        - trades_df: 交易数据DataFrame
        
        返回:
        - 指标字典
        """
        metrics = {}
        
        try:
            # 如果交易数据不为空
            if not trades_df.empty:
                # 基础交易统计
                metrics['total_trades'] = len(trades_df)
                
                if 'profit_pct' in trades_df.columns:
                    metrics['avg_profit'] = float(trades_df['profit_pct'].mean())
                    metrics['total_return'] = float(trades_df['profit_pct'].sum())
                
                if 'is_profitable' in trades_df.columns:
                    metrics['win_rate'] = float(trades_df['is_profitable'].mean())
                    metrics['profitable_trades'] = int(trades_df['is_profitable'].sum())
                    metrics['loss_trades'] = int(len(trades_df) - trades_df['is_profitable'].sum())
                
                # 计算平均持仓时间（小时）
                if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
                    try:
                        # 确保是datetime对象
                        if isinstance(trades_df['exit_time'].iloc[0], str):
                            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
                        if isinstance(trades_df['entry_time'].iloc[0], str):
                            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
                        
                        # 计算持仓时间
                        duration = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 3600
                        metrics['avg_holding_time'] = float(duration.mean())
                    except Exception as e:
                        self.logger.warning(f"计算平均持仓时间失败: {e}")
                
                # 计算市场环境
                if len(trades_df) >= 10:
                    # 简单规则确定市场环境
                    recent_trades = trades_df.sort_values('exit_time', ascending=False).head(10)
                    
                    # 计算胜率
                    if 'is_profitable' in recent_trades.columns:
                        recent_win_rate = recent_trades['is_profitable'].mean()
                    else:
                        recent_win_rate = 0.5
                    
                    # 计算波动率 - 使用收益的标准差
                    if 'profit_pct' in recent_trades.columns:
                        volatility = recent_trades['profit_pct'].std()
                        metrics['market_volatility'] = float(volatility)
                    
                    # 计算趋势强度 - 简单使用连续相同方向交易的比例
                    if 'profit_pct' in recent_trades.columns and len(recent_trades) > 1:
                        directions = np.sign(recent_trades['profit_pct']).values
                        same_directions = np.sum(directions[:-1] == directions[1:])
                        trend_strength = same_directions / (len(directions) - 1)
                        metrics['trend_strength'] = float(trend_strength)
                    
                    # 确定市场环境
                    if 'market_volatility' in metrics and 'trend_strength' in metrics:
                        volatility = metrics['market_volatility']
                        trend_strength = metrics['trend_strength']
                        
                        if volatility > 0.03:  # 高波动
                            if trend_strength > 0.6:
                                metrics['market_regime'] = "trending_volatile"
                            else:
                                metrics['market_regime'] = "volatile"
                        else:  # 低波动
                            if trend_strength > 0.7:
                                metrics['market_regime'] = "trending"
                            else:
                                metrics['market_regime'] = "ranging"
                    else:
                        metrics['market_regime'] = "neutral"
            
            # 添加模型指标
            model_metrics = self._load_model_metrics()
            metrics.update(model_metrics)
            
            # 添加错误统计
            error_stats = self._load_error_stats()
            if error_stats:
                total_predictions = error_stats.get('total_errors', 0) + error_stats.get('total_successes', 0)
                if total_predictions > 0:
                    metrics['error_rate'] = error_stats.get('total_errors', 0) / total_predictions
                else:
                    metrics['error_rate'] = 0
                
                metrics['error_stats'] = error_stats
            
            # 添加当前配置信息
            if 'trading' in self.config:
                metrics['risk_per_trade'] = self.config['trading'].get('risk_per_trade_pct', 0.02)
                metrics['max_leverage'] = self.config['trading'].get('max_leverage', 3.0)
            
            # 计算适应性指标
            metrics['compatibility_score'] = self._calculate_compatibility_score(metrics)
            
            self.logger.info(f"指标计算完成: compatibility_score={metrics.get('compatibility_score', 0):.2f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"计算指标时发生错误: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return metrics
    
    def _calculate_compatibility_score(self, metrics: Dict[str, Any]) -> float:
        """
        计算环境兼容性评分
        
        参数:
        - metrics: 计算出的指标字典
        
        返回:
        - 兼容性评分 (0-1)
        """
        score = 0.5  # 默认中等分数
        
        # 根据胜率调整分数
        win_rate = metrics.get('win_rate')
        if win_rate is not None:
            if win_rate >= 0.6:
                score += 0.3
            elif win_rate >= 0.5:
                score += 0.1
            elif win_rate < 0.4:
                score -= 0.2
        
        # 根据总收益调整分数
        total_return = metrics.get('total_return')
        if total_return is not None:
            if total_return > 0.1:  # 10%以上的收益
                score += 0.2
            elif total_return > 0:
                score += 0.1
            elif total_return < -0.05:  # 5%以上的亏损
                score -= 0.2
        
        # 根据错误率调整分数
        error_rate = metrics.get('error_rate')
        if error_rate is not None:
            if error_rate > 0.1:  # 10%以上的错误率
                score -= 0.1
            if error_rate > 0.2:  # 20%以上的错误率
                score -= 0.2
        
        # 确保分数在0-1范围内
        return max(0.0, min(1.0, score))
    
    def _generate_chart_base64(self, trades_df: pd.DataFrame) -> str:
        """
        生成交易性能图表并将其转换为base64编码的图像
        
        参数:
        - trades_df: 交易数据
        
        返回:
        - base64编码的PNG图像
        """
        try:
            if trades_df.empty:
                # 生成一个空图表
                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, "没有可用的交易数据", horizontalalignment='center', verticalalignment='center', fontsize=14)
                plt.axis('off')
            else:
                # 准备数据
                if 'entry_time' not in trades_df.columns or 'profit_pct' not in trades_df.columns:
                    plt.figure(figsize=(10, 6))
                    plt.text(0.5, 0.5, "交易数据缺少必要的列", horizontalalignment='center', verticalalignment='center', fontsize=14)
                    plt.axis('off')
                else:
                    # 确保日期列是datetime类型
                    if isinstance(trades_df['entry_time'].iloc[0], str):
                        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
                    
                    # 按日期排序
                    trades_df = trades_df.sort_values('entry_time')
                    
                    # 计算累积收益
                    trades_df['cumulative_return'] = (1 + trades_df['profit_pct']).cumprod() - 1
                    
                    # 创建图表
                    fig, ax1 = plt.subplots(figsize=(10, 6))
                    
                    # 绘制累积收益
                    ax1.plot(trades_df['entry_time'], trades_df['cumulative_return'] * 100, 'b-', label='累积收益 (%)')
                    ax1.set_xlabel('日期')
                    ax1.set_ylabel('累积收益 (%)', color='b')
                    ax1.tick_params('y', colors='b')
                    
                    # 添加胜率移动平均线
                    ax2 = ax1.twinx()
                    window = min(10, len(trades_df))
                    if window > 1:
                        # 计算滚动胜率
                        if 'is_profitable' in trades_df.columns:
                            trades_df['win_rate_ma'] = trades_df['is_profitable'].rolling(window=window).mean()
                            ax2.plot(trades_df['entry_time'], trades_df['win_rate_ma'] * 100, 'r-', label=f'胜率 (MA{window})')
                            ax2.set_ylabel('胜率 (%)', color='r')
                            ax2.tick_params('y', colors='r')
                    
                    fig.tight_layout()
                    
                    # 添加网格线
                    ax1.grid(True, alpha=0.3)
                    
                    # 添加图例
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels() if window > 1 else ([], [])
                    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
                    
                    # 添加标题
                    plt.title('交易性能跟踪')
            
            # 将图表转换为base64编码的PNG
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            return f"data:image/png;base64,{image_base64}"
        except Exception as e:
            self.logger.error(f"生成图表失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            # 返回一个简单的错误图表
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"生成图表时出错: {str(e)}", horizontalalignment='center', verticalalignment='center', fontsize=12, wrap=True)
            plt.axis('off')
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            return f"data:image/png;base64,{image_base64}"
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> str:
        """
        生成系统调整建议的HTML
        
        参数:
        - metrics: 指标字典
        
        返回:
        - 建议HTML
        """
        recommendations = []
        
        # 根据胜率生成建议
        win_rate = metrics.get('win_rate')
        if win_rate is not None:
            if win_rate < 0.4:
                recommendations.append({
                    'text': '胜率较低，建议降低风险参数并检查市场环境兼容性',
                    'priority': 'high'
                })
            elif win_rate < 0.45:
                recommendations.append({
                    'text': '胜率低于平均水平，考虑轻微降低风险参数',
                    'priority': 'medium'
                })
            elif win_rate > 0.6:
                recommendations.append({
                    'text': '胜率较高，可以考虑适度提高风险参数以增加回报',
                    'priority': 'low'
                })
        
        # 根据错误率生成建议
        error_rate = metrics.get('error_rate')
        if error_rate is not None:
            if error_rate > 0.2:
                recommendations.append({
                    'text': '预测错误率较高，需要检查数据预处理和模型输入',
                    'priority': 'high'
                })
            elif error_rate > 0.1:
                recommendations.append({
                    'text': '预测错误率有所上升，建议监控错误类型和频率',
                    'priority': 'medium'
                })
        
        # 根据市场环境生成建议
        market_regime = metrics.get('market_regime')
        volatility = metrics.get('market_volatility')
        trend_strength = metrics.get('trend_strength')
        
        if market_regime == 'volatile' and volatility and volatility > 0.04:
            recommendations.append({
                'text': '市场波动性很高，建议进一步降低风险并增加回退策略的保守性',
                'priority': 'high'
            })
        elif market_regime == 'trending' and trend_strength and trend_strength > 0.8:
            recommendations.append({
                'text': '市场趋势性强，建议增加持仓时间以更好地捕捉趋势',
                'priority': 'medium'
            })
        elif market_regime == 'ranging':
            recommendations.append({
                'text': '市场处于区间震荡状态，建议降低持仓时间并考虑区间策略',
                'priority': 'medium'
            })
        
        # 如果没有建议，添加默认建议
        if not recommendations:
            recommendations.append({
                'text': '系统参数运行正常，暂无调整建议',
                'priority': 'low'
            })
        
        # 构建HTML
        html_parts = []
        for rec in recommendations:
            priority_class = f"priority-{rec['priority']}"
            html_parts.append(f"<div class='recommendation-item'><span class='recommendation-priority {priority_class}'></span>{rec['text']}</div>")
        
        return "\n".join(html_parts)
    
    def _format_change(self, current: float, previous: float, higher_is_better: bool = True) -> Tuple[str, str]:
        """
        格式化变化值和类名
        
        参数:
        - current: 当前值
        - previous: 前一个值
        - higher_is_better: 值越高是否越好
        
        返回:
        - (变化文本, CSS类名)
        """
        if previous is None or current is None:
            return "无变化", "neutral-change"
        
        change = current - previous
        if abs(change) < 0.0001:
            return "无变化", "neutral-change"
        
        pct_change = change / abs(previous) if previous != 0 else 0
        
        if change > 0:
            text = f"↑ +{pct_change:.1%}"
            cls = "positive-change" if higher_is_better else "negative-change"
        else:
            text = f"↓ {pct_change:.1%}"
            cls = "negative-change" if higher_is_better else "positive-change"
        
        return text, cls
    
    def _get_status_class(self, value: float, thresholds: Dict[str, float], higher_is_better: bool = True) -> str:
        """
        获取状态CSS类
        
        参数:
        - value: 当前值
        - thresholds: 阈值字典 {'optimal': x, 'good': y, 'caution': z}
        - higher_is_better: 值越高是否越好
        
        返回:
        - CSS类名
        """
        if value is None:
            return "status-caution"
        
        if higher_is_better:
            if value >= thresholds.get('optimal', 0.8):
                return "status-optimal"
            elif value >= thresholds.get('good', 0.5):
                return "status-good"
            elif value >= thresholds.get('caution', 0.3):
                return "status-caution"
            else:
                return "status-warning"
        else:
            if value <= thresholds.get('optimal', 0.1):
                return "status-optimal"
            elif value <= thresholds.get('good', 0.2):
                return "status-good"
            elif value <= thresholds.get('caution', 0.3):
                return "status-caution"
            else:
                return "status-warning"
    
    def _get_status_text(self, cls: str) -> str:
        """根据CSS类获取状态文本"""
        status_texts = {
            "status-optimal": "最佳",
            "status-good": "良好",
            "status-caution": "注意",
            "status-warning": "警告"
        }
        return status_texts.get(cls, "未知")
    
    def _format_recent_trades(self, trades_df: pd.DataFrame) -> str:
        """
        格式化最近交易记录的HTML
        
        参数:
        - trades_df: 交易数据DataFrame
        
        返回:
        - 最近交易记录的HTML代码
        """
        if trades_df.empty:
            return "<tr><td colspan='6'>没有可用的交易记录</td></tr>"
        
        # 获取最近10笔交易
        recent = trades_df.sort_values('exit_time', ascending=False).head(10)
        
        rows = []
        for _, trade in recent.iterrows():
            # 格式化日期
            if 'exit_time' in trade and trade['exit_time']:
                if isinstance(trade['exit_time'], str):
                    time_str = trade['exit_time']
                else:
                    time_str = trade['exit_time'].strftime('%Y-%m-%d %H:%M')
            else:
                time_str = "未知"
            
            # 确定收益类和文本
            if 'profit_pct' in trade:
                profit_class = "positive-change" if trade['profit_pct'] > 0 else "negative-change"
                profit_text = f"{trade['profit_pct']:.2%}"
            else:
                profit_class = "neutral-change"
                profit_text = "未知"
            
            # 持仓时间
            if 'duration' in trade:
                duration_text = f"{trade['duration']:.1f}小时"
            else:
                duration_text = "未知"
                
            # 市场状态
            market_state = trade.get('market_type', '未知')
            
            # 生成表格行
            rows.append(f"""
            <tr>
                <td>{trade.get('trade_id', '未知')}</td>
                <td>{time_str}</td>
                <td>{trade.get('side', '未知')}</td>
                <td class="{profit_class}">{profit_text}</td>
                <td>{duration_text}</td>
                <td>{market_state}</td>
            </tr>
            """)
        
        return "\n".join(rows)
    
    def generate_report(self) -> str:
        """
        生成模型监控报告
        
        返回:
        - 报告文件路径
        """
        try:
            # 加载数据
            trades_df = self._load_trades_data()
            
            # 计算指标
            metrics = self._calculate_metrics(trades_df)
            self.current_data = metrics
            
            # 生成图表
            chart_base64 = self._generate_chart_base64(trades_df)
            
            # 获取上次报告中的指标
            prev_metrics = self.previous_data
            
            # 替换模板变量
            with open(self.template_path, 'r', encoding='utf-8') as f:
                template = f.read()
            
            # 格式化时间
            report_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            template = template.replace('{{REPORT_DATE}}', report_date)
            template = template.replace('{{CURRENT_YEAR}}', str(datetime.now().year))
            
            # 替换图表
            template = template.replace('{{CHART_URL}}', chart_base64)
            
            # 替换基础指标
            compatibility_score = metrics.get('compatibility_score', 0)
            template = template.replace('{{COMPATIBILITY_SCORE}}', f"{compatibility_score:.2f}/1.0")
            
            prev_compatibility = prev_metrics.get('compatibility_score')
            comp_change, comp_class = self._format_change(compatibility_score, prev_compatibility)
            template = template.replace('{{COMPATIBILITY_CHANGE}}', comp_change)
            template = template.replace('{{COMPATIBILITY_CHANGE_CLASS}}', comp_class)
            
            # 胜率
            win_rate = metrics.get('win_rate', 0)
            template = template.replace('{{WIN_RATE}}', f"{win_rate:.1%}")
            
            prev_win_rate = prev_metrics.get('win_rate')
            win_change, win_class = self._format_change(win_rate, prev_win_rate)
            template = template.replace('{{WIN_RATE_CHANGE}}', win_change)
            template = template.replace('{{WIN_RATE_CHANGE_CLASS}}', win_class)
            
            # 市场环境
            market_regime = metrics.get('market_regime', 'neutral')
            # 将下划线转换为空格并首字母大写
            display_regime = ' '.join(word.capitalize() for word in market_regime.split('_'))
            template = template.replace('{{MARKET_REGIME}}', display_regime)
            
            # 模型版本
            model_version = metrics.get('model_version', 'unknown')
            template = template.replace('{{MODEL_VERSION}}', model_version)
            
            prev_model = prev_metrics.get('model_version')
            if prev_model and prev_model != model_version:
                template = template.replace('{{MODEL_UPDATE_STATUS}}', "已更新")
            else:
                template = template.replace('{{MODEL_UPDATE_STATUS}}', "无变化")
            
            # 平均每笔收益
            avg_profit = metrics.get('avg_profit', 0)
            template = template.replace('{{AVG_PROFIT}}', f"{avg_profit:.2%}")
            
            prev_avg_profit = prev_metrics.get('avg_profit')
            profit_change, profit_class = self._format_change(avg_profit, prev_avg_profit)
            template = template.replace('{{PROFIT_CHANGE}}', profit_change)
            template = template.replace('{{PROFIT_CHANGE_CLASS}}', profit_class)
            
            # 风险设置
            risk_setting = metrics.get('risk_per_trade', 0.02)
            template = template.replace('{{RISK_SETTING}}', f"{risk_setting:.2%}")
            
            prev_risk = prev_metrics.get('risk_per_trade')
            risk_change, risk_class = self._format_change(risk_setting, prev_risk, False)
            template = template.replace('{{RISK_CHANGE}}', risk_change)
            template = template.replace('{{RISK_CHANGE_CLASS}}', risk_class)
            
            # 总收益率
            total_return = metrics.get('total_return', 0)
            template = template.replace('{{TOTAL_RETURN}}', f"{total_return:.2%}")
            
            prev_return = prev_metrics.get('total_return')
            return_change, return_class = self._format_change(total_return, prev_return)
            template = template.replace('{{RETURN_CHANGE}}', return_change)
            template = template.replace('{{RETURN_CHANGE_CLASS}}', return_class)
            
            # 预测准确度
            error_rate = metrics.get('error_rate', 0)
            prediction_accuracy = 1 - error_rate if error_rate is not None else 0
            template = template.replace('{{PREDICTION_ACCURACY}}', f"{prediction_accuracy:.1%}")
            
            prev_accuracy = 1 - prev_metrics.get('error_rate', 0) if prev_metrics.get('error_rate') is not None else None
            accuracy_change, accuracy_class = self._format_change(prediction_accuracy, prev_accuracy)
            template = template.replace('{{ACCURACY_CHANGE}}', accuracy_change)
            template = template.replace('{{ACCURACY_CHANGE_CLASS}}', accuracy_class)
            
            # 适应性指标表格
            # 波动率
            volatility = metrics.get('market_volatility', 0)
            template = template.replace('{{VOLATILITY_CURRENT}}', f"{volatility:.4f}")
            
            prev_volatility = prev_metrics.get('market_volatility')
            template = template.replace('{{VOLATILITY_PREVIOUS}}', f"{prev_volatility:.4f}" if prev_volatility is not None else "N/A")
            
            vol_change, vol_class = self._format_change(volatility, prev_volatility, False)
            template = template.replace('{{VOLATILITY_CHANGE}}', vol_change)
            template = template.replace('{{VOLATILITY_CHANGE_CLASS}}', vol_class)
            
            vol_status_class = self._get_status_class(volatility, {'optimal': 0.01, 'good': 0.02, 'caution': 0.03}, False)
            template = template.replace('{{VOLATILITY_STATUS_CLASS}}', vol_status_class)
            template = template.replace('{{VOLATILITY_STATUS}}', self._get_status_text(vol_status_class))
            
            # 趋势强度
            trend = metrics.get('trend_strength', 0)
            template = template.replace('{{TREND_CURRENT}}', f"{trend:.4f}")
            
            prev_trend = prev_metrics.get('trend_strength')
            template = template.replace('{{TREND_PREVIOUS}}', f"{prev_trend:.4f}" if prev_trend is not None else "N/A")
            
            trend_change, trend_class = self._format_change(trend, prev_trend)
            template = template.replace('{{TREND_CHANGE}}', trend_change)
            template = template.replace('{{TREND_CHANGE_CLASS}}', trend_class)
            
            trend_status_class = self._get_status_class(trend, {'optimal': 0.7, 'good': 0.5, 'caution': 0.3})
            template = template.replace('{{TREND_STATUS_CLASS}}', trend_status_class)
            template = template.replace('{{TREND_STATUS}}', self._get_status_text(trend_status_class))
            
            # 动作-奖励相关性 (假设为0.5，因为我们没有这个数据)
            corr = 0.5
            template = template.replace('{{CORRELATION_CURRENT}}', f"{corr:.4f}")
            template = template.replace('{{CORRELATION_PREVIOUS}}', "N/A")
            template = template.replace('{{CORRELATION_CHANGE}}', "无变化")
            template = template.replace('{{CORRELATION_CHANGE_CLASS}}', "neutral-change")
            
            corr_status_class = self._get_status_class(corr, {'optimal': 0.7, 'good': 0.5, 'caution': 0.3})
            template = template.replace('{{CORRELATION_STATUS_CLASS}}', corr_status_class)
            template = template.replace('{{CORRELATION_STATUS}}', self._get_status_text(corr_status_class))
            
            # 平均持仓时间
            hold_time = metrics.get('avg_holding_time', 0)
            template = template.replace('{{HOLDING_TIME_CURRENT}}', f"{hold_time:.1f}小时")
            
            prev_hold_time = prev_metrics.get('avg_holding_time')
            template = template.replace('{{HOLDING_TIME_PREVIOUS}}', f"{prev_hold_time:.1f}小时" if prev_hold_time is not None else "N/A")
            
            # 持仓时间没有明确的好坏，使用中性变化
            if prev_hold_time is not None:
                hold_change = f"{(hold_time-prev_hold_time)/prev_hold_time:.1%}" if prev_hold_time > 0 else "无变化"
                hold_class = "neutral-change"
            else:
                hold_change = "无变化"
                hold_class = "neutral-change"
            
            template = template.replace('{{HOLDING_TIME_CHANGE}}', hold_change)
            template = template.replace('{{HOLDING_TIME_CHANGE_CLASS}}', hold_class)
            
            hold_status_class = "status-good"  # 默认值，因为没有明确的指标
            template = template.replace('{{HOLDING_TIME_STATUS_CLASS}}', hold_status_class)
            template = template.replace('{{HOLDING_TIME_STATUS}}', self._get_status_text(hold_status_class))
            
            # 错误率
            template = template.replace('{{ERROR_RATE_CURRENT}}', f"{error_rate:.2%}")
            
            prev_error_rate = prev_metrics.get('error_rate')
            template = template.replace('{{ERROR_RATE_PREVIOUS}}', f"{prev_error_rate:.2%}" if prev_error_rate is not None else "N/A")
            
            error_change, error_class = self._format_change(error_rate, prev_error_rate, False)
            template = template.replace('{{ERROR_RATE_CHANGE}}', error_change)
            template = template.replace('{{ERROR_RATE_CHANGE_CLASS}}', error_class)
            
            error_status_class = self._get_status_class(error_rate, {'optimal': 0.05, 'good': 0.1, 'caution': 0.2}, False)
            template = template.replace('{{ERROR_RATE_STATUS_CLASS}}', error_status_class)
            template = template.replace('{{ERROR_RATE_STATUS}}', self._get_status_text(error_status_class))
            
            # 生成建议
            recommendations_html = self._generate_recommendations(metrics)
            template = template.replace('{{RECOMMENDATIONS_CONTENT}}', recommendations_html)
            
            # 生成最近交易记录
            recent_trades_html = self._format_recent_trades(trades_df)
            template = template.replace('{{RECENT_TRADES}}', recent_trades_html)
            
            # 保存报告
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = self.output_dir / f"model_monitor_report_{timestamp}.html"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(template)
            
            # 保存当前数据供未来比较
            data_path = self.output_dir / f"report_data_{timestamp}.json"
            with open(data_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            self.logger.info(f"报告生成成功: {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"生成报告失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return ""

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="模型监控报告生成工具")
    parser.add_argument("--config", type=str, default="/home/losesky/crypto-trading-rl/trading_system/config/test_config.json", help="配置文件路径")
    parser.add_argument("--data-dir", type=str, default="/home/losesky/crypto-trading-rl/trading_system/data", help="数据目录路径")
    parser.add_argument("--output-dir", type=str, default="/home/losesky/crypto-trading-rl/trading_system/reports", help="输出目录路径")
    args = parser.parse_args()
    
    generator = ModelMonitorReportGenerator(args.config, args.data_dir, args.output_dir)
    report_path = generator.generate_report()
    
    if report_path:
        print(f"报告已生成: {report_path}")
    else:
        print("报告生成失败，请检查日志")
