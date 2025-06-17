#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
集成测试 - 交易系统主要流程测试
该测试模拟了完整的交易流程，从数据获取、特征计算到模型预测和交易执行
"""

import os
import sys
import unittest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入被测试的模块
from main import TradingSystem


class TestTradingSystemIntegration(unittest.TestCase):
    """交易系统集成测试"""
    
    def setUp(self):
        """测试准备工作"""
        # 使用测试配置目录
        test_config_dir = os.path.join(os.path.dirname(__file__), 'test_config')
        
        # 确保测试配置目录存在
        os.makedirs(test_config_dir, exist_ok=True)
        
        # 初始化模拟
        self.setup_mocks()
        
        # 使用测试配置初始化交易系统
        with patch('main.BinanceAPI') as mock_api_class, \
             patch('main.DataProcessor') as mock_data_processor_class, \
             patch('main.FeatureEngineer') as mock_feature_engineer_class, \
             patch('main.ModelLoader') as mock_model_loader_class, \
             patch('main.ModelEnsemble') as mock_model_ensemble_class, \
             patch('main.PredictionProcessor') as mock_prediction_processor_class, \
             patch('main.OrderManager') as mock_order_manager_class, \
             patch('main.PositionManager') as mock_position_manager_class, \
             patch('main.Executor') as mock_executor_class, \
             patch('main.RiskChecker') as mock_risk_checker_class, \
             patch('main.MoneyManager') as mock_money_manager_class, \
             patch('main.CircuitBreaker') as mock_circuit_breaker_class, \
             patch('main.Dashboard') as mock_dashboard_class, \
             patch('main.Alerter') as mock_alerter_class, \
             patch('main.Reporter') as mock_reporter_class:
            
            # 设置模拟对象
            mock_api_class.return_value = self.mock_api
            mock_data_processor_class.return_value = self.mock_data_processor
            mock_feature_engineer_class.return_value = self.mock_feature_engineer
            mock_model_loader_class.return_value = self.mock_model_loader
            mock_model_ensemble_class.return_value = self.mock_model_ensemble
            mock_prediction_processor_class.return_value = self.mock_prediction_processor
            mock_order_manager_class.return_value = self.mock_order_manager
            mock_position_manager_class.return_value = self.mock_position_manager
            mock_executor_class.return_value = self.mock_executor
            mock_risk_checker_class.return_value = self.mock_risk_checker
            mock_money_manager_class.return_value = self.mock_money_manager
            mock_circuit_breaker_class.return_value = self.mock_circuit_breaker
            mock_dashboard_class.return_value = self.mock_dashboard
            mock_alerter_class.return_value = self.mock_alerter
            mock_reporter_class.return_value = self.mock_reporter
            
            # 初始化交易系统
            self.trading_system = TradingSystem(config_dir=test_config_dir, log_level='DEBUG')
    
    def setup_mocks(self):
        """设置各种模拟对象"""
        # API模拟
        self.mock_api = AsyncMock()
        self.mock_api.initialize = AsyncMock(return_value=True)
        self.mock_api.get_account_info = AsyncMock(return_value={
            'status': 'NORMAL',
            'balances': [
                {'asset': 'USDT', 'free': '10000', 'locked': '0'},
                {'asset': 'BTC', 'free': '0.1', 'locked': '0'}
            ]
        })
        self.mock_api.get_symbol_info = AsyncMock(return_value={
            'symbol': 'BTCUSDT',
            'status': 'TRADING',
            'baseAsset': 'BTC',
            'quoteAsset': 'USDT'
        })
        
        # 创建模拟K线数据
        kline_data = []
        base_time = int(datetime.now().timestamp() * 1000)
        base_price = 40000
        for i in range(100):
            open_time = base_time - (99-i) * 60 * 60 * 1000  # 每小时一根K线
            close_time = open_time + 60 * 60 * 1000 - 1
            open_price = base_price + np.random.normal(0, 200)
            high_price = open_price * (1 + abs(np.random.normal(0, 0.01)))
            low_price = open_price * (1 - abs(np.random.normal(0, 0.01)))
            close_price = open_price + np.random.normal(0, 200)
            volume = abs(np.random.normal(10, 5))
            
            kline = [
                open_time,                 # 开盘时间
                str(open_price),           # 开盘价
                str(high_price),           # 最高价
                str(low_price),            # 最低价
                str(close_price),          # 收盘价
                str(volume),               # 成交量
                close_time,                # 收盘时间
                str(volume * close_price), # 成交额
                100,                       # 成交笔数
                str(volume * 0.7),         # 主动买入成交量
                str(volume * 0.7 * close_price), # 主动买入成交额
                "0"                        # 忽略
            ]
            kline_data.append(kline)
            
        self.mock_api.get_klines = AsyncMock(return_value=kline_data)
        
        # 数据处理模拟
        self.mock_data_processor = MagicMock()
        self.mock_data_processor.process_klines = MagicMock(return_value=pd.DataFrame({
            'open_time': pd.date_range(start='2025-01-01', periods=100, freq='H'),
            'open': np.random.normal(40000, 200, 100),
            'high': np.random.normal(40200, 200, 100),
            'low': np.random.normal(39800, 200, 100),
            'close': np.random.normal(40000, 200, 100),
            'volume': np.abs(np.random.normal(10, 5, 100))
        }))
        
        # 特征工程模拟
        self.mock_feature_engineer = MagicMock()
        self.mock_feature_engineer.calculate_features = MagicMock(return_value=pd.DataFrame({
            'open_time': pd.date_range(start='2025-01-01', periods=100, freq='H'),
            'open': np.random.normal(40000, 200, 100),
            'high': np.random.normal(40200, 200, 100),
            'low': np.random.normal(39800, 200, 100),
            'close': np.random.normal(40000, 200, 100),
            'volume': np.abs(np.random.normal(10, 5, 100)),
            'rsi_14': np.random.uniform(30, 70, 100),
            'macd': np.random.normal(0, 50, 100),
            'macd_signal': np.random.normal(0, 40, 100),
            'macd_hist': np.random.normal(0, 20, 100),
            'ma_20': np.random.normal(40000, 100, 100),
            'ma_50': np.random.normal(39900, 100, 100)
        }))
        
        # 模型加载模拟
        self.mock_model_loader = MagicMock()
        self.mock_model_loader.load_models = MagicMock(return_value=[
            {'name': 'model1', 'model': MagicMock()},
            {'name': 'model2', 'model': MagicMock()}
        ])
        
        # 模型集成模拟
        self.mock_model_ensemble = MagicMock()
        self.mock_model_ensemble.predict = MagicMock(return_value={
            'model1': {'action': 1, 'confidence': 0.7},
            'model2': {'action': 1, 'confidence': 0.8},
            'ensemble': {'action': 1, 'confidence': 0.75}
        })
        
        # 预测处理器模拟
        self.mock_prediction_processor = MagicMock()
        self.mock_prediction_processor.process = MagicMock(return_value=(
            {'buy': 0.75, 'sell': 0.15, 'hold': 0.1},
            0.5,  # 建议仓位大小
            0.75  # 置信度
        ))
        
        # 订单管理器模拟
        self.mock_order_manager = MagicMock()
        self.mock_order_manager.create_order = AsyncMock(return_value={'orderId': '123456'})
        self.mock_order_manager.get_order_status = AsyncMock(return_value={'status': 'FILLED'})
        
        # 仓位管理器模拟
        self.mock_position_manager = MagicMock()
        self.mock_position_manager.update_positions = AsyncMock(return_value={
            'BTCUSDT': {'size': 0, 'entry_price': 0}
        })
        self.mock_position_manager.get_position = AsyncMock(return_value={'size': 0, 'entry_price': 0})
        self.mock_position_manager.get_all_positions = AsyncMock(return_value={
            'BTCUSDT': {'size': 0, 'entry_price': 0}
        })
        
        # 执行器模拟
        self.mock_executor = MagicMock()
        self.mock_executor.open_long = AsyncMock(return_value={
            'symbol': 'BTCUSDT',
            'orderId': '123456',
            'price': 40000,
            'quantity': 0.1,
            'status': 'FILLED'
        })
        self.mock_executor.close_long = AsyncMock(return_value={
            'symbol': 'BTCUSDT',
            'orderId': '123457',
            'price': 41000,
            'quantity': 0.1,
            'status': 'FILLED'
        })
        
        # 风控检查器模拟
        self.mock_risk_checker = MagicMock()
        self.mock_risk_checker.check_trade_risk = MagicMock(return_value={
            'allowed': True,
            'reason': ''
        })
        
        # 资金管理器模拟
        self.mock_money_manager = MagicMock()
        self.mock_money_manager.initialize_capital = MagicMock()
        self.mock_money_manager.calculate_position_size = MagicMock(return_value=0.1)
        
        # 熔断器模拟
        self.mock_circuit_breaker = MagicMock()
        self.mock_circuit_breaker.initialize = MagicMock()
        self.mock_circuit_breaker.check_status = MagicMock(return_value=True)
        self.mock_circuit_breaker.check_trade = MagicMock(return_value=True)
        self.mock_circuit_breaker.status = "NORMAL"
        
        # 仪表板模拟
        self.mock_dashboard = MagicMock()
        self.mock_dashboard.initialize = MagicMock()
        self.mock_dashboard.update = MagicMock()
        self.mock_dashboard.update_market_data = MagicMock()
        self.mock_dashboard.get_features_data = MagicMock(return_value=pd.DataFrame({
            'rsi_14': [50],
            'macd': [10],
            'macd_signal': [5],
            'macd_hist': [5]
        }))
        self.mock_dashboard.update_prediction = MagicMock()
        self.mock_dashboard.get_latest_prediction = MagicMock(return_value={
            'action_probas': {'buy': 0.75, 'sell': 0.15, 'hold': 0.1},
            'position_size': 0.5,
            'confidence': 0.75
        })
        self.mock_dashboard.update_trade_status = MagicMock()
        self.mock_dashboard.generate_summary = MagicMock()
        
        # 告警器模拟
        self.mock_alerter = MagicMock()
        self.mock_alerter.info = MagicMock()
        self.mock_alerter.warning = MagicMock()
        self.mock_alerter.alert = MagicMock()
        self.mock_alerter.critical = MagicMock()
        self.mock_alerter.emergency = MagicMock()
        
        # 报告生成器模拟
        self.mock_reporter = MagicMock()
        self.mock_reporter.generate_daily_report = MagicMock(return_value={
            'html': '/path/to/report.html'
        })
    
    def run_async_test(self, coroutine):
        """运行异步测试"""
        return asyncio.run(coroutine)
    
    def test_initialize(self):
        """测试系统初始化"""
        # 运行初始化
        result = self.run_async_test(self.trading_system.initialize())
        
        # 验证初始化结果
        self.assertTrue(result)
        
        # 验证API调用
        self.mock_api.initialize.assert_called_once()
        self.mock_api.get_account_info.assert_called_once()
        self.mock_api.get_symbol_info.assert_called()
        self.mock_api.get_klines.assert_called()
        
        # 验证其他模块初始化
        self.mock_money_manager.initialize_capital.assert_called_once()
        self.mock_circuit_breaker.initialize.assert_called_once()
        self.mock_dashboard.initialize.assert_called_once()
    
    def test_update_features(self):
        """测试特征更新"""
        # 运行特征更新
        result = self.run_async_test(
            self.trading_system._update_features('BTCUSDT', '1h')
        )
        
        # 验证结果
        self.assertTrue(result)
        
        # 验证调用
        self.mock_api.get_klines.assert_called()
        self.mock_data_processor.process_klines.assert_called_once()
        self.mock_feature_engineer.calculate_features.assert_called_once()
        self.mock_dashboard.update_market_data.assert_called_once()
    
    def test_make_predictions(self):
        """测试模型预测"""
        # 运行模型预测
        result = self.run_async_test(
            self.trading_system._make_predictions('BTCUSDT', '1h')
        )
        
        # 验证结果
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)
        
        # 验证调用
        self.mock_dashboard.get_features_data.assert_called_once()
        self.mock_model_ensemble.predict.assert_called_once()
        self.mock_prediction_processor.process.assert_called_once()
        self.mock_dashboard.update_prediction.assert_called_once()
    
    def test_execute_trades(self):
        """测试交易执行"""
        # 运行交易执行
        self.run_async_test(
            self.trading_system._execute_trades('BTCUSDT')
        )
        
        # 验证调用
        self.mock_dashboard.get_latest_prediction.assert_called_once()
        self.mock_position_manager.get_position.assert_called_once()
        self.mock_risk_checker.check_trade_risk.assert_called_once()
        self.mock_money_manager.calculate_position_size.assert_called_once()
        self.mock_executor.open_long.assert_called_once()
        self.mock_dashboard.update_trade_status.assert_called_once()
        self.mock_position_manager.update_positions.assert_called_once()
        self.mock_circuit_breaker.check_trade.assert_called_once()
    
    def test_close_all_positions(self):
        """测试关闭所有持仓"""
        # 模拟持有仓位
        self.mock_position_manager.get_all_positions = AsyncMock(return_value={
            'BTCUSDT': {'size': 0.1, 'entry_price': 40000}
        })
        
        # 运行关闭所有持仓
        self.run_async_test(
            self.trading_system._close_all_positions()
        )
        
        # 验证调用
        self.mock_position_manager.get_all_positions.assert_called_once()
        self.mock_executor.close_long.assert_called_once()
        self.mock_position_manager.update_positions.assert_called_once()
    
    def test_generate_reports(self):
        """测试生成报告"""
        # 运行报告生成
        self.run_async_test(
            self.trading_system._generate_reports()
        )
        
        # 验证调用
        self.mock_reporter.generate_daily_report.assert_called_once()
    
    def test_run_trading_cycle(self):
        """测试完整交易周期"""
        # 模拟运行状态
        self.trading_system.is_running = True
        
        # 创建停止函数，在一次循环后停止
        async def stop_after_one_cycle():
            await asyncio.sleep(0.1)
            self.trading_system.is_running = False
        
        # 运行交易周期和停止函数
        async def run_test():
            stop_task = asyncio.create_task(stop_after_one_cycle())
            await self.trading_system.run()
            await stop_task
        
        # 执行测试
        self.run_async_test(run_test())
        
        # 验证调用了所有关键函数
        self.mock_api.initialize.assert_called()
        self.mock_api.get_account_info.assert_called()
        self.mock_model_loader.load_models.assert_called()
        self.mock_money_manager.initialize_capital.assert_called()
        self.mock_dashboard.update.assert_called()
    
    def tearDown(self):
        """测试清理"""
        # 清理资源
        pass


if __name__ == '__main__':
    unittest.main()
