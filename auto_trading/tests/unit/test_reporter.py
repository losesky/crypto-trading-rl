import unittest
import sys
import os
from unittest.mock import patch, MagicMock
from datetime import datetime
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入测试目标
from monitor.reporter import Reporter

class TestReporter(unittest.TestCase):
    """报告生成器单元测试"""
    
    def setUp(self):
        """测试准备工作"""
        # 使用测试配置初始化报告生成器
        with patch('os.makedirs'):  # 避免实际创建目录
            self.reporter = Reporter()
        
        # 模拟配置
        self.reporter.config = {
            'reports': {
                'daily': {'enabled': True},
                'weekly': {'enabled': True},
                'monthly': {'enabled': True}
            },
            'formats': {
                'html': {'enabled': True},
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
        
        # 创建测试数据
        self.create_test_data()
        
    def create_test_data(self):
        """创建用于测试的样本数据"""
        # 创建交易数据
        self.trades_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=10, freq='D'),
            'symbol': ['BTCUSDT'] * 10,
            'side': ['BUY', 'SELL'] * 5,
            'price': [40000, 41000, 39000, 42000, 43000, 41500, 40500, 39500, 42500, 43500],
            'quantity': [0.1] * 10,
            'pnl': [100, -50, 200, 150, -100, 300, -200, 250, 180, -120]
        })
        
        # 创建余额数据
        self.balance_df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=30, freq='D'),
            'balance': np.linspace(10000, 12000, 30) + np.random.normal(0, 100, 30),
            'return': np.random.normal(0.01, 0.05, 30)
        })
        
    @patch('monitor.reporter.Reporter._load_trade_data')
    @patch('monitor.reporter.Reporter._load_daily_balance')
    @patch('monitor.reporter.Reporter._generate_charts')
    @patch('monitor.reporter.Reporter._generate_html_report')
    @patch('monitor.reporter.Reporter._generate_json_report')
    @patch('monitor.reporter.Reporter._generate_csv_report')
    def test_generate_daily_report(self, mock_csv, mock_json, mock_html, mock_charts, mock_balance, mock_trades):
        """测试生成日报"""
        # 设置模拟返回值
        mock_trades.return_value = self.trades_df
        mock_balance.return_value = self.balance_df
        mock_charts.return_value = {'balance_chart': 'path/to/chart.png'}
        mock_html.return_value = 'path/to/report.html'
        mock_json.return_value = 'path/to/report.json'
        mock_csv.return_value = 'path/to/report.csv'
        
        # 测试生成日报
        result = self.reporter.generate_daily_report('2023-01-10')
        
        # 验证调用
        mock_trades.assert_called_once()
        mock_balance.assert_called_once()
        mock_charts.assert_called_once()
        mock_html.assert_called_once()
        mock_json.assert_called_once()
        mock_csv.assert_called_once()
        
        # 验证结果
        self.assertEqual(len(result), 3)  # 应该生成3种格式的报告
        self.assertIn('html', result)
        self.assertIn('json', result)
        self.assertIn('csv', result)
        
    def test_calculate_metrics(self):
        """测试计算交易指标"""
        # 使用测试数据计算指标
        metrics = self.reporter._calculate_metrics(self.trades_df, self.balance_df)
        
        # 验证计算结果
        self.assertIn('total_pnl', metrics)
        self.assertIn('win_rate', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        
        # 验证具体值
        self.assertEqual(metrics['total_pnl'], 710)  # 710 = 100 - 50 + 200 + ... - 120
        self.assertAlmostEqual(metrics['win_rate'], 0.6, delta=0.01)  # 6次盈利，4次亏损
        
    @patch('monitor.reporter.plt.figure')
    @patch('monitor.reporter.plt.savefig')
    @patch('monitor.reporter.plt.close')
    def test_generate_charts(self, mock_close, mock_save, mock_figure):
        """测试生成图表"""
        # 测试图表生成
        charts = self.reporter._generate_charts(
            self.trades_df, 
            self.balance_df, 
            'daily', 
            '2023-01-01', 
            '2023-01-10'
        )
        
        # 验证调用
        self.assertTrue(mock_figure.called)
        self.assertTrue(mock_save.called)
        self.assertTrue(mock_close.called)
        
        # 验证结果
        self.assertIsInstance(charts, dict)
        
    def test_validate_input_dates(self):
        """测试日期格式验证和计算"""
        # 测试每日报告日期
        today = datetime.now().date()
        yesterday = (today - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        
        # 调用generate_report方法，但模拟内部方法避免实际执行
        with patch.object(self.reporter, '_load_trade_data', return_value=pd.DataFrame()):
            with patch.object(self.reporter, '_load_daily_balance', return_value=pd.DataFrame()):
                with patch.object(self.reporter, '_calculate_metrics', return_value={}):
                    with patch.object(self.reporter, '_generate_charts', return_value={}):
                        with patch.object(self.reporter, '_generate_html_report', return_value=None):
                            self.reporter.generate_report('daily')
        
        # 生成周报和月报也应该能够自动计算日期范围
        with patch.object(self.reporter, '_load_trade_data', return_value=pd.DataFrame()):
            with patch.object(self.reporter, '_load_daily_balance', return_value=pd.DataFrame()):
                with patch.object(self.reporter, '_calculate_metrics', return_value={}):
                    with patch.object(self.reporter, '_generate_charts', return_value={}):
                        with patch.object(self.reporter, '_generate_html_report', return_value=None):
                            self.reporter.generate_report('weekly')
                            self.reporter.generate_report('monthly')

if __name__ == '__main__':
    unittest.main()
