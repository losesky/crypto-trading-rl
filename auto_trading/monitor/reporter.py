import logging
import os
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
from pathlib import Path
import jinja2
import webbrowser
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，用于生成图像文件

class Reporter:
    """
    交易报告生成器类，用于生成日/周/月交易报告和绩效分析
    支持多种输出格式：HTML、PDF、JSON、CSV
    """
    
    def __init__(self, config_path=None, data_dir=None):
        """
        初始化报告生成器
        
        Args:
            config_path: 配置文件路径，默认为None（使用默认配置文件路径）
            data_dir: 数据存储目录，默认为None（使用配置中的路径）
        """
        self.logger = logging.getLogger(__name__)
        
        # 如果未提供配置文件路径，使用默认路径
        if config_path is None:
            config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
            config_path = os.path.join(config_dir, 'report_config.yaml')
        
        # 加载配置
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            self.logger.info("报告配置加载成功")
        except Exception as e:
            self.logger.error(f"加载报告配置失败: {e}")
            # 设置默认配置
            self.config = {
                'reports': {
                    'daily': {'enabled': True},
                    'weekly': {'enabled': True},
                    'monthly': {'enabled': True}
                },
                'formats': {
                    'html': {'enabled': True},
                    'pdf': {'enabled': False},
                    'json': {'enabled': True},
                    'csv': {'enabled': True}
                },
                'metrics': [
                    'total_pnl', 'win_rate', 'sharpe_ratio', 'max_drawdown',
                    'avg_profit_per_trade', 'avg_loss_per_trade', 
                    'profit_loss_ratio', 'total_trades'
                ],
                'output_dir': 'reports',
                'templates_dir': 'templates',
                'data_dir': 'data'
            }
        
        # 设置数据目录和输出目录
        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.data_dir = data_dir or os.path.join(base_dir, self.config.get('data_dir', 'data'))
        self.output_dir = os.path.join(base_dir, self.config.get('output_dir', 'reports'))
        self.templates_dir = os.path.join(base_dir, self.config.get('templates_dir', 'templates'))
        
        # 创建必要的目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'data'), exist_ok=True)
        
    def _load_trade_data(self, start_date, end_date):
        """
        加载指定日期范围内的交易数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame: 交易数据
        """
        try:
            # 获取指定日期范围内所有交易记录文件
            trade_files = []
            for date in pd.date_range(start=start_date, end=end_date):
                date_str = date.strftime('%Y%m%d')
                trade_file = os.path.join(self.data_dir, 'trades', f"trades_{date_str}.csv")
                if os.path.exists(trade_file):
                    trade_files.append(trade_file)
            
            # 合并所有交易记录
            if not trade_files:
                self.logger.warning(f"未找到{start_date}至{end_date}期间的交易数据")
                return pd.DataFrame()
                
            # 尝试合并所有文件
            dfs = [pd.read_csv(file) for file in trade_files]
            if not dfs:
                return pd.DataFrame()
                
            trade_data = pd.concat(dfs, ignore_index=True)
            return trade_data
        except Exception as e:
            self.logger.error(f"加载交易数据失败: {e}")
            return pd.DataFrame()
    
    def _load_balance_data(self, start_date=None, end_date=None):
        """
        加载余额历史数据
        
        Args:
            start_date: 开始日期 (可选)
            end_date: 结束日期 (可选)
            
        Returns:
            DataFrame: 余额历史数据
        """
        try:
            balance_file = os.path.join(self.data_dir, 'balance', 'daily_balance.csv')
            
            if not os.path.exists(balance_file):
                self.logger.warning(f"未找到余额记录文件: {balance_file}")
                return pd.DataFrame()
                
            balance_data = pd.read_csv(balance_file)
            
            # 日期过滤
            if start_date and end_date:
                balance_data = balance_data[(balance_data['date'] >= start_date) & 
                                           (balance_data['date'] <= end_date)]
            
            return balance_data
        except Exception as e:
            self.logger.error(f"加载余额数据失败: {e}")
            return pd.DataFrame()
    
    def _calculate_metrics(self, trades_df, balance_df):
        """
        计算交易绩效指标
        
        Args:
            trades_df: 交易数据DataFrame
            balance_df: 余额数据DataFrame
            
        Returns:
            dict: 指标结果字典
        """
        metrics = {}
        
        # 检查是否有数据可供计算
        if trades_df.empty:
            self.logger.warning("未提供交易数据，无法计算绩效指标")
            return {
                'total_pnl': 0,
                'win_rate': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'avg_profit_per_trade': 0,
                'avg_loss_per_trade': 0,
                'profit_loss_ratio': 0,
                'total_trades': 0,
                'profitable_trades': 0,
                'loss_trades': 0
            }
            
        try:
            # 计算基本指标
            metrics['total_trades'] = len(trades_df)
            
            # 计算盈利和亏损交易
            if 'pnl' in trades_df.columns:
                profitable_trades = trades_df[trades_df['pnl'] > 0]
                loss_trades = trades_df[trades_df['pnl'] < 0]
                
                metrics['profitable_trades'] = len(profitable_trades)
                metrics['loss_trades'] = len(loss_trades)
                
                # 胜率
                metrics['win_rate'] = metrics['profitable_trades'] / metrics['total_trades'] if metrics['total_trades'] > 0 else 0
                
                # 总盈亏
                metrics['total_pnl'] = trades_df['pnl'].sum()
                
                # 平均盈利和亏损
                metrics['avg_profit_per_trade'] = profitable_trades['pnl'].mean() if not profitable_trades.empty else 0
                metrics['avg_loss_per_trade'] = loss_trades['pnl'].mean() if not loss_trades.empty else 0
                
                # 盈亏比
                metrics['profit_loss_ratio'] = abs(metrics['avg_profit_per_trade'] / metrics['avg_loss_per_trade']) if metrics['avg_loss_per_trade'] != 0 else float('inf')
            
            # 计算夏普比率和最大回撤（需要余额数据）
            if not balance_df.empty and 'balance' in balance_df.columns:
                # 计算日收益率
                balance_df['return'] = balance_df['balance'].pct_change()
                
                # 计算夏普比率（假设无风险收益率为0）
                daily_returns = balance_df['return'].dropna()
                if len(daily_returns) > 1:
                    metrics['sharpe_ratio'] = np.sqrt(252) * (daily_returns.mean() / daily_returns.std()) if daily_returns.std() != 0 else 0
                else:
                    metrics['sharpe_ratio'] = 0
                
                # 计算最大回撤
                balance_df['cummax'] = balance_df['balance'].cummax()
                balance_df['drawdown'] = (balance_df['cummax'] - balance_df['balance']) / balance_df['cummax']
                metrics['max_drawdown'] = balance_df['drawdown'].max()
            
        except Exception as e:
            self.logger.error(f"计算绩效指标失败: {e}")
        
        return metrics
        
    def _generate_charts(self, trades_df, balance_df, period_type, start_date, end_date):
        """
        生成报告图表
        
        Args:
            trades_df: 交易数据
            balance_df: 余额数据
            period_type: 报告周期类型
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            dict: 图表文件路径的字典
        """
        charts = {}
        
        try:
            # 生成余额变化曲线图
            if not balance_df.empty and 'balance' in balance_df.columns and 'date' in balance_df.columns:
                plt.figure(figsize=(10, 6))
                plt.plot(balance_df['date'], balance_df['balance'], marker='o', linestyle='-')
                plt.title('账户余额变化')
                plt.xlabel('日期')
                plt.ylabel('余额 (USDT)')
                plt.grid(True)
                plt.tight_layout()
                
                balance_chart_path = os.path.join(self.output_dir, 'images', f'{period_type}_balance_{start_date}_{end_date}.png')
                plt.savefig(balance_chart_path)
                plt.close()
                
                charts['balance_chart'] = os.path.relpath(balance_chart_path, self.output_dir)
            
            # 生成每日收益率分布图
            if not balance_df.empty and 'return' in balance_df.columns:
                plt.figure(figsize=(10, 6))
                balance_df['return'].hist(bins=30)
                plt.title('日收益率分布')
                plt.xlabel('日收益率')
                plt.ylabel('频率')
                plt.grid(True)
                plt.tight_layout()
                
                returns_chart_path = os.path.join(self.output_dir, 'images', f'{period_type}_returns_{start_date}_{end_date}.png')
                plt.savefig(returns_chart_path)
                plt.close()
                
                charts['returns_chart'] = os.path.relpath(returns_chart_path, self.output_dir)
            
            # 生成交易盈亏分布图
            if not trades_df.empty and 'pnl' in trades_df.columns:
                plt.figure(figsize=(10, 6))
                trades_df['pnl'].hist(bins=30)
                plt.title('交易盈亏分布')
                plt.xlabel('盈亏 (USDT)')
                plt.ylabel('交易次数')
                plt.grid(True)
                plt.tight_layout()
                
                pnl_chart_path = os.path.join(self.output_dir, 'images', f'{period_type}_pnl_{start_date}_{end_date}.png')
                plt.savefig(pnl_chart_path)
                plt.close()
                
                charts['pnl_chart'] = os.path.relpath(pnl_chart_path, self.output_dir)
            
        except Exception as e:
            self.logger.error(f"生成图表失败: {e}")
        
        return charts
        
    def _generate_html_report(self, metrics, charts, trades_df, period_type, start_date, end_date):
        """
        生成HTML格式的报告
        
        Args:
            metrics: 绩效指标字典
            charts: 图表文件路径字典
            trades_df: 交易数据
            period_type: 报告周期类型
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            str: HTML报告文件路径
        """
        try:
            # 创建Jinja2环境
            env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(self.templates_dir)
            )
            
            # 尝试加载模板
            try:
                template = env.get_template('trading_report.html')
            except jinja2.exceptions.TemplateNotFound:
                # 如果模板不存在，创建一个基础模板
                os.makedirs(self.templates_dir, exist_ok=True)
                with open(os.path.join(self.templates_dir, 'trading_report.html'), 'w') as f:
                    f.write("""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .metrics-container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            width: calc(25% - 20px);
            min-width: 200px;
        }
        .metric-title {
            font-size: 14px;
            color: #6c757d;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }
        .chart-container {
            margin: 30px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .positive {
            color: #28a745;
        }
        .negative {
            color: #dc3545;
        }
        .summary-box {
            background-color: #e9ecef;
            border-left: 5px solid #007bff;
            padding: 15px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ title }}</h1>
        <div class="summary-box">
            <p>报告周期: {{ period_desc }}</p>
            <p>报告生成时间: {{ generation_time }}</p>
        </div>
        
        <h2>绩效指标概览</h2>
        <div class="metrics-container">
            {% for metric in metrics %}
            <div class="metric-card">
                <div class="metric-title">{{ metric.name }}</div>
                <div class="metric-value {% if metric.is_positive %}positive{% elif metric.is_negative %}negative{% endif %}">
                    {{ metric.value }}
                </div>
            </div>
            {% endfor %}
        </div>
        
        <h2>账户表现图表</h2>
        {% for chart_name, chart_path in charts.items() %}
        <div class="chart-container">
            <h3>{{ chart_name }}</h3>
            <img src="{{ chart_path }}" alt="{{ chart_name }}" width="100%">
        </div>
        {% endfor %}
        
        {% if trades|length > 0 %}
        <h2>交易详情</h2>
        <table>
            <thead>
                <tr>
                    {% for column in trade_columns %}
                    <th>{{ column }}</th>
                    {% endfor %}
                </tr>
            </thead>
            <tbody>
                {% for trade in trades %}
                <tr>
                    {% for value in trade %}
                    <td {% if loop.index == pnl_index %}class="{% if value > 0 %}positive{% elif value < 0 %}negative{% endif %}"{% endif %}>
                        {{ value }}
                    </td>
                    {% endfor %}
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endif %}
    </div>
</body>
</html>
                    """)
                template = env.get_template('trading_report.html')
            
            # 准备报告数据
            report_data = {
                'title': f"{period_type.capitalize()}交易报告 {start_date} - {end_date}",
                'period_desc': f"{start_date} 至 {end_date}",
                'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'metrics': [
                    {'name': '总盈亏 (USDT)', 'value': f"{metrics.get('total_pnl', 0):.2f}", 
                     'is_positive': metrics.get('total_pnl', 0) > 0, 'is_negative': metrics.get('total_pnl', 0) < 0},
                    {'name': '胜率', 'value': f"{metrics.get('win_rate', 0)*100:.2f}%", 
                     'is_positive': metrics.get('win_rate', 0) > 0.5, 'is_negative': metrics.get('win_rate', 0) < 0.5},
                    {'name': '夏普比率', 'value': f"{metrics.get('sharpe_ratio', 0):.2f}", 
                     'is_positive': metrics.get('sharpe_ratio', 0) > 1, 'is_negative': metrics.get('sharpe_ratio', 0) < 0},
                    {'name': '最大回撤', 'value': f"{metrics.get('max_drawdown', 0)*100:.2f}%", 
                     'is_negative': True},
                    {'name': '总交易次数', 'value': metrics.get('total_trades', 0)},
                    {'name': '盈利交易', 'value': metrics.get('profitable_trades', 0)},
                    {'name': '亏损交易', 'value': metrics.get('loss_trades', 0)},
                    {'name': '平均盈利 (USDT)', 'value': f"{metrics.get('avg_profit_per_trade', 0):.2f}", 
                     'is_positive': True},
                    {'name': '平均亏损 (USDT)', 'value': f"{metrics.get('avg_loss_per_trade', 0):.2f}", 
                     'is_negative': True},
                    {'name': '盈亏比', 'value': f"{metrics.get('profit_loss_ratio', 0):.2f}", 
                     'is_positive': metrics.get('profit_loss_ratio', 0) > 1, 'is_negative': metrics.get('profit_loss_ratio', 0) < 1}
                ],
                'charts': {
                    '余额变化趋势': charts.get('balance_chart', ''),
                    '日收益率分布': charts.get('returns_chart', ''),
                    '交易盈亏分布': charts.get('pnl_chart', '')
                } if charts else {}
            }
            
            # 添加交易记录（最多显示100条）
            if not trades_df.empty:
                # 按时间排序，最新的交易在前面
                if 'timestamp' in trades_df.columns:
                    trades_df = trades_df.sort_values('timestamp', ascending=False)
                
                # 限制显示记录数
                display_trades = trades_df.head(100)
                
                # 格式化为显示友好的形式
                display_trades_list = []
                for _, row in display_trades.iterrows():
                    display_trades_list.append(row.tolist())
                
                report_data['trades'] = display_trades_list
                report_data['trade_columns'] = trades_df.columns.tolist()
                
                # 找出PNL列的索引，用于设置颜色
                pnl_index = -1
                if 'pnl' in trades_df.columns:
                    pnl_index = trades_df.columns.get_loc('pnl')
                report_data['pnl_index'] = pnl_index
            else:
                report_data['trades'] = []
                report_data['trade_columns'] = []
            
            # 渲染模板
            html_content = template.render(**report_data)
            
            # 写入HTML文件
            report_file = os.path.join(self.output_dir, f"{period_type}_report_{start_date}_{end_date}.html")
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            return report_file
            
        except Exception as e:
            self.logger.error(f"生成HTML报告失败: {e}")
            return None
            
    def _generate_json_report(self, metrics, trades_df, period_type, start_date, end_date):
        """
        生成JSON格式的报告
        
        Args:
            metrics: 绩效指标字典
            trades_df: 交易数据
            period_type: 报告周期类型
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            str: JSON报告文件路径
        """
        try:
            # 准备报告数据
            report_data = {
                'report_info': {
                    'type': period_type,
                    'start_date': start_date,
                    'end_date': end_date,
                    'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                },
                'metrics': metrics
            }
            
            # 添加交易数据
            if not trades_df.empty:
                # 将DataFrame转换为字典列表
                trades_list = trades_df.to_dict('records')
                
                # JSON序列化时间
                for trade in trades_list:
                    for key, value in trade.items():
                        if isinstance(value, pd.Timestamp):
                            trade[key] = value.strftime('%Y-%m-%d %H:%M:%S')
                            
                report_data['trades'] = trades_list
                
            # 写入JSON文件
            json_file = os.path.join(self.output_dir, 'data', f"{period_type}_report_{start_date}_{end_date}.json")
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
                
            return json_file
            
        except Exception as e:
            self.logger.error(f"生成JSON报告失败: {e}")
            return None
            
    def _generate_csv_report(self, trades_df, period_type, start_date, end_date):
        """
        生成CSV格式的交易数据报告
        
        Args:
            trades_df: 交易数据
            period_type: 报告周期类型
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            str: CSV报告文件路径
        """
        try:
            if trades_df.empty:
                self.logger.warning("未提供交易数据，无法生成CSV报告")
                return None
                
            # 写入CSV文件
            csv_file = os.path.join(self.output_dir, 'data', f"{period_type}_trades_{start_date}_{end_date}.csv")
            trades_df.to_csv(csv_file, index=False)
            
            return csv_file
            
        except Exception as e:
            self.logger.error(f"生成CSV报告失败: {e}")
            return None
            
    def generate_report(self, period_type, start_date=None, end_date=None):
        """
        生成指定周期的交易报告
        
        Args:
            period_type: 报告周期类型（'daily', 'weekly', 'monthly'）
            start_date: 开始日期（可选，如果不提供会根据周期类型自动计算）
            end_date: 结束日期（可选，如果不提供会根据周期类型自动计算）
            
        Returns:
            dict: 包含生成的报告文件路径
        """
        # 检查报告类型是否启用
        if not self.config.get('reports', {}).get(period_type, {}).get('enabled', True):
            self.logger.info(f"{period_type}报告类型未启用")
            return {}
            
        # 如果未提供日期，根据报告类型自动计算日期范围
        if start_date is None or end_date is None:
            today = datetime.now().date()
            if period_type == 'daily':
                # 默认生成昨天的日报
                report_date = today - timedelta(days=1)
                start_date = report_date.strftime('%Y-%m-%d')
                end_date = report_date.strftime('%Y-%m-%d')
            elif period_type == 'weekly':
                # 默认生成上一周的周报（周一到周日）
                end_date = (today - timedelta(days=today.weekday() + 1)).strftime('%Y-%m-%d')
                start_date = (datetime.strptime(end_date, '%Y-%m-%d').date() - timedelta(days=6)).strftime('%Y-%m-%d')
            elif period_type == 'monthly':
                # 默认生成上个月的月报
                last_month = today.replace(day=1) - timedelta(days=1)
                start_date = last_month.replace(day=1).strftime('%Y-%m-%d')
                end_date = last_month.strftime('%Y-%m-%d')
            
        # 加载交易数据
        trades_df = self._load_trade_data(start_date, end_date)
        
        # 加载余额数据
        balance_df = self._load_balance_data(start_date, end_date)
        
        # 计算指标
        metrics = self._calculate_metrics(trades_df, balance_df)
        
        # 生成图表
        charts = self._generate_charts(trades_df, balance_df, period_type, start_date, end_date)
        
        # 生成各种格式的报告
        report_files = {}
        
        # HTML报告
        if self.config.get('formats', {}).get('html', {}).get('enabled', True):
            html_file = self._generate_html_report(metrics, charts, trades_df, period_type, start_date, end_date)
            if html_file:
                report_files['html'] = html_file
                
        # JSON报告
        if self.config.get('formats', {}).get('json', {}).get('enabled', True):
            json_file = self._generate_json_report(metrics, trades_df, period_type, start_date, end_date)
            if json_file:
                report_files['json'] = json_file
                
        # CSV报告
        if self.config.get('formats', {}).get('csv', {}).get('enabled', True):
            csv_file = self._generate_csv_report(trades_df, period_type, start_date, end_date)
            if csv_file:
                report_files['csv'] = csv_file
        
        # 打印生成报告信息
        self.logger.info(f"已生成{period_type}报告 ({start_date} - {end_date}): {', '.join(report_files.keys())}")
        
        return report_files
        
    def generate_daily_report(self, date):
        """
        生成指定日期的每日报告
        
        Args:
            date: 报告日期 (YYYY-MM-DD)
            
        Returns:
            List: 生成的报告文件路径列表
        """
        try:
            # 解析日期
            report_date = pd.to_datetime(date).strftime('%Y-%m-%d')
            
            # 加载该日的交易数据
            trade_data = self._load_trade_data(report_date, report_date)
            
            # 加载余额数据
            balance_data = self._load_balance_data()
            
            # 生成报告
            report_files = []
            
            # 计算性能指标
            metrics = self._calculate_metrics(trade_data)
            
            # 生成报告文件名
            report_base = f"daily_report_{report_date}_{report_date}"
            
            # 为每种启用的格式生成报告
            formats = self.config.get('formats', {})
            
            # HTML报告
            if formats.get('html', {}).get('enabled', True):
                html_file = os.path.join(self.output_dir, f"{report_base}.html")
                self._generate_html_report(html_file, report_date, report_date, trade_data, balance_data, metrics)
                report_files.append(html_file)
            
            # JSON报告
            if formats.get('json', {}).get('enabled', True):
                json_file = os.path.join(self.output_dir, 'data', f"{report_base}.json")
                self._save_json_report(json_file, report_date, report_date, metrics)
                report_files.append(json_file)
            
            # CSV报告
            if formats.get('csv', {}).get('enabled', True):
                csv_file = os.path.join(self.output_dir, 'data', f"{report_base}.csv")
                self._save_csv_report(csv_file, trade_data)
                report_files.append(csv_file)
            
            self.logger.info(f"已生成daily报告 ({report_date} - {report_date}): {', '.join([os.path.splitext(os.path.basename(f))[1][1:] for f in report_files])}")
            return report_files
            
        except Exception as e:
            self.logger.error(f"生成每日报告失败: {e}", exc_info=True)
            return []
    
    def generate_weekly_report(self, end_date=None):
        """
        生成周报告
        
        Args:
            end_date: 结束日期 (YYYY-MM-DD)，默认为昨天
            
        Returns:
            List: 生成的报告文件路径列表
        """
        try:
            # 确定日期范围
            if end_date is None:
                end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            end_dt = pd.to_datetime(end_date)
            start_dt = end_dt - timedelta(days=6)  # 一周的数据
            
            start_date = start_dt.strftime('%Y-%m-%d')
            end_date = end_dt.strftime('%Y-%m-%d')
            
            # 加载该周的交易数据
            trade_data = self._load_trade_data(start_date, end_date)
            
            # 加载余额数据
            balance_data = self._load_balance_data(start_date, end_date)
            
            # 生成报告
            report_files = []
            
            # 计算性能指标
            metrics = self._calculate_metrics(trade_data)
            
            # 生成报告文件名
            report_base = f"weekly_report_{start_date}_{end_date}"
            
            # 为每种启用的格式生成报告
            formats = self.config.get('formats', {})
            
            # HTML报告
            if formats.get('html', {}).get('enabled', True):
                html_file = os.path.join(self.output_dir, f"{report_base}.html")
                self._generate_html_report(html_file, start_date, end_date, trade_data, balance_data, metrics)
                report_files.append(html_file)
            
            # JSON报告
            if formats.get('json', {}).get('enabled', True):
                json_file = os.path.join(self.output_dir, 'data', f"{report_base}.json")
                self._save_json_report(json_file, start_date, end_date, metrics)
                report_files.append(json_file)
            
            # CSV报告
            if formats.get('csv', {}).get('enabled', True):
                csv_file = os.path.join(self.output_dir, 'data', f"{report_base}.csv")
                self._save_csv_report(csv_file, trade_data)
                report_files.append(csv_file)
            
            self.logger.info(f"已生成weekly报告 ({start_date} - {end_date}): {', '.join([os.path.splitext(os.path.basename(f))[1][1:] for f in report_files])}")
            return report_files
            
        except Exception as e:
            self.logger.error(f"生成周报告失败: {e}", exc_info=True)
            return []
    
    def generate_monthly_report(self, end_date=None):
        """
        生成月报告
        
        Args:
            end_date: 结束日期 (YYYY-MM-DD)，默认为昨天
            
        Returns:
            List: 生成的报告文件路径列表
        """
        try:
            # 确定日期范围
            if end_date is None:
                end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            end_dt = pd.to_datetime(end_date)
            
            # 计算一个月前的日期（简化处理，使用30天）
            start_dt = end_dt.replace(day=1)
            if start_dt.month == 1:
                prev_month = 12
                year = start_dt.year - 1
            else:
                prev_month = start_dt.month - 1
                year = start_dt.year
            
            start_dt = start_dt.replace(month=prev_month, year=year)
            
            start_date = start_dt.strftime('%Y-%m-%d')
            end_date = end_dt.strftime('%Y-%m-%d')
            
            # 加载该月的交易数据
            trade_data = self._load_trade_data(start_date, end_date)
            
            # 加载余额数据
            balance_data = self._load_balance_data(start_date, end_date)
            
            # 生成报告
            report_files = []
            
            # 计算性能指标
            metrics = self._calculate_metrics(trade_data)
            
            # 生成报告文件名
            report_base = f"monthly_report_{start_date}_{end_date}"
            
            # 为每种启用的格式生成报告
            formats = self.config.get('formats', {})
            
            # HTML报告
            if formats.get('html', {}).get('enabled', True):
                html_file = os.path.join(self.output_dir, f"{report_base}.html")
                self._generate_html_report(html_file, start_date, end_date, trade_data, balance_data, metrics)
                report_files.append(html_file)
            
            # JSON报告
            if formats.get('json', {}).get('enabled', True):
                json_file = os.path.join(self.output_dir, 'data', f"{report_base}.json")
                self._save_json_report(json_file, start_date, end_date, metrics)
                report_files.append(json_file)
            
            # CSV报告
            if formats.get('csv', {}).get('enabled', True):
                csv_file = os.path.join(self.output_dir, 'data', f"{report_base}.csv")
                self._save_csv_report(csv_file, trade_data)
                report_files.append(csv_file)
            
            self.logger.info(f"已生成monthly报告 ({start_date} - {end_date}): {', '.join([os.path.splitext(os.path.basename(f))[1][1:] for f in report_files])}")
            return report_files
            
        except Exception as e:
            self.logger.error(f"生成月报告失败: {e}", exc_info=True)
            return []
    
    def _calculate_metrics(self, trade_data):
        """
        计算交易性能指标
        
        Args:
            trade_data: 交易数据DataFrame
            
        Returns:
            Dict: 性能指标字典
        """
        try:
            if trade_data.empty:
                self.logger.warning("未提供交易数据，无法计算绩效指标")
                return {
                    'total_trades': 0,
                    'win_trades': 0,
                    'loss_trades': 0,
                    'win_rate': 0,
                    'total_pnl': 0,
                    'avg_profit_per_trade': 0,
                    'avg_loss_per_trade': 0,
                    'profit_loss_ratio': 0,
                    'max_drawdown': 0,
                    'sharpe_ratio': 0,
                }
            
            # 计算基本指标
            total_trades = len(trade_data)
            win_trades = len(trade_data[trade_data['pnl'] > 0])
            loss_trades = len(trade_data[trade_data['pnl'] < 0])
            
            # 胜率
            win_rate = win_trades / total_trades if total_trades > 0 else 0
            
            # 盈亏
            total_pnl = trade_data['pnl'].sum()
            
            # 平均盈利和亏损
            avg_profit = trade_data[trade_data['pnl'] > 0]['pnl'].mean() if win_trades > 0 else 0
            avg_loss = trade_data[trade_data['pnl'] < 0]['pnl'].mean() if loss_trades > 0 else 0
            
            # 盈亏比
            profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')
            
            # 计算累计余额曲线，用于计算最大回撤和夏普比率
            trade_data = trade_data.sort_values('timestamp')
            trade_data['cumulative_pnl'] = trade_data['pnl'].cumsum()
            
            # 最大回撤
            cumulative = trade_data['cumulative_pnl'].to_numpy()
            max_drawdown = 0
            peak = cumulative[0]
            
            for value in cumulative:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak if peak != 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
            
            # 夏普比率 (简化版，使用日收益率)
            daily_returns = []
            for date, group in trade_data.groupby(pd.to_datetime(trade_data['timestamp']).dt.date):
                daily_returns.append(group['pnl'].sum())
            
            if len(daily_returns) > 1:
                sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) if np.std(daily_returns) != 0 else 0
            else:
                sharpe_ratio = 0
            
            # 汇总指标
            metrics = {
                'total_trades': total_trades,
                'win_trades': win_trades,
                'loss_trades': loss_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_profit_per_trade': avg_profit,
                'avg_loss_per_trade': avg_loss,
                'profit_loss_ratio': profit_loss_ratio,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio
            }
            
            return metrics
        except Exception as e:
            self.logger.error(f"计算性能指标失败: {e}")
            return {}
            
    def _generate_html_report(self, output_file, start_date, end_date, trade_data, balance_data, metrics):
        """
        生成HTML格式报告
        
        Args:
            output_file: 输出文件路径
            start_date: 开始日期
            end_date: 结束日期
            trade_data: 交易数据
            balance_data: 余额数据
            metrics: 性能指标
        """
        try:
            # 加载模板
            template_file = os.path.join(self.templates_dir, 'trading_report.html')
            if not os.path.exists(template_file):
                self.logger.error(f"HTML模板文件不存在: {template_file}")
                return
            
            with open(template_file, 'r', encoding='utf-8') as f:
                template_content = f.read()
            
            # 渲染模板
            template = jinja2.Template(template_content)
            
            # 生成图表
            image_dir = os.path.join(self.output_dir, 'images')
            balance_chart = self._generate_balance_chart(balance_data, image_dir, f"balance_{start_date}_{end_date}")
            pnl_chart = self._generate_pnl_chart(trade_data, image_dir, f"pnl_{start_date}_{end_date}")
            
            # 准备数据
            # 为模板创建charts字典，避免'charts' is undefined错误
            charts = {}
            if balance_chart:
                charts['账户余额'] = os.path.basename(balance_chart)
            if pnl_chart:
                charts['盈亏变化'] = os.path.basename(pnl_chart)
                
            context = {
                'title': f"交易报告 ({start_date} 至 {end_date})",
                'start_date': start_date,
                'end_date': end_date,
                'period_desc': f"{start_date} 至 {end_date}",
                'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'metrics': metrics,
                'trades': trade_data.to_dict('records') if not trade_data.empty else [],
                'trade_columns': trade_data.columns.tolist() if not trade_data.empty else [],
                'pnl_index': trade_data.columns.get_loc('profit_loss') if not trade_data.empty and 'profit_loss' in trade_data.columns else None,
                'charts': charts
            }
            
            # 渲染HTML
            html_content = template.render(**context)
            
            # 保存文件
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
        except Exception as e:
            self.logger.error(f"生成HTML报告失败: {e}")
    
    def _save_json_report(self, output_file, start_date, end_date, metrics):
        """
        保存JSON格式报告
        
        Args:
            output_file: 输出文件路径
            start_date: 开始日期
            end_date: 结束日期
            metrics: 性能指标
        """
        try:
            # 准备数据
            report_data = {
                'start_date': start_date,
                'end_date': end_date,
                'generated_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'metrics': metrics
            }
            
            # 创建输出目录
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 保存文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"保存JSON报告失败: {e}")
    
    def _save_csv_report(self, output_file, trade_data):
        """
        保存CSV格式报告
        
        Args:
            output_file: 输出文件路径
            trade_data: 交易数据
        """
        try:
            if trade_data.empty:
                self.logger.warning("未提供交易数据，无法生成CSV报告")
                return
            
            # 创建输出目录
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 保存交易数据
            trade_data.to_csv(output_file, index=False)
            
        except Exception as e:
            self.logger.error(f"保存CSV报告失败: {e}")
    
    def _generate_balance_chart(self, balance_data, image_dir, filename):
        """
        生成资金曲线图
        
        Args:
            balance_data: 余额数据
            image_dir: 图片输出目录
            filename: 文件名前缀
            
        Returns:
            str: 图片文件路径
        """
        try:
            if balance_data.empty:
                return None
            
            # 创建图表
            plt.figure(figsize=(10, 6))
            plt.plot(balance_data['date'], balance_data['total_balance'], label='总资金')
            plt.plot(balance_data['date'], balance_data['available_balance'], label='可用资金')
            
            plt.title('账户资金曲线')
            plt.xlabel('日期')
            plt.ylabel('金额 (USDT)')
            plt.legend()
            plt.grid(True)
            
            # 保存图表
            image_path = os.path.join(image_dir, f"{filename}.png")
            plt.savefig(image_path)
            plt.close()
            
            return image_path
        except Exception as e:
            self.logger.error(f"生成资金曲线图失败: {e}")
            return None
    
    def _generate_pnl_chart(self, trade_data, image_dir, filename):
        """
        生成PnL曲线图
        
        Args:
            trade_data: 交易数据
            image_dir: 图片输出目录
            filename: 文件名前缀
            
        Returns:
            str: 图片文件路径
        """
        try:
            if trade_data.empty:
                return None
            
            # 计算累计PnL
            trade_data = trade_data.sort_values('timestamp')
            trade_data['cumulative_pnl'] = trade_data['pnl'].cumsum()
            
            # 创建图表
            plt.figure(figsize=(10, 6))
            plt.plot(pd.to_datetime(trade_data['timestamp']), trade_data['cumulative_pnl'])
            
            plt.title('累计盈亏曲线')
            plt.xlabel('时间')
            plt.ylabel('盈亏 (USDT)')
            plt.grid(True)
            
            # 保存图表
            image_path = os.path.join(image_dir, f"{filename}.png")
            plt.savefig(image_path)
            plt.close()
            
            return image_path
        except Exception as e:
            self.logger.error(f"生成盈亏曲线图失败: {e}")
            return None
