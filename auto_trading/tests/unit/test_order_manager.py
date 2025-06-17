#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch, MagicMock
import yaml
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到路径，确保可以导入所有模块
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.append(str(PROJECT_ROOT))

from trading.order_manager import OrderManager, OrderStatus, OrderType, OrderSide


class TestOrderManager(unittest.TestCase):
    """订单管理器测试类"""

    def setUp(self):
        """准备测试环境"""
        # 创建测试配置
        self.test_config = {
            'trading': {
                'order': {
                    'default_timeInForce': 'GTC',
                    'default_reduceOnly': False,
                    'max_retries': 3,
                    'retry_delay': 1.0,
                    'auto_adjust_price': True
                }
            }
        }
        
        # 创建临时配置文件
        self.config_path = "/tmp/test_trading_config.yaml"
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
        
        # 模拟BinanceAPI
        self.mock_binance_api = MagicMock()
        self.mock_binance_api.create_order.return_value = {
            'orderId': 12345,
            'clientOrderId': 'test-client-order-id',
            'status': 'NEW'
        }
        
        self.mock_binance_api.get_symbol_price_ticker.return_value = {
            'price': '36000'
        }
        
        # 创建订单管理器实例
        self.order_manager = OrderManager(
            config_path=self.config_path,
            binance_api=self.mock_binance_api
        )
        
    def tearDown(self):
        """清理测试环境"""
        # 删除临时配置文件
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.order_manager.default_tif, 'GTC')
        self.assertFalse(self.order_manager.default_reduce_only)
        self.assertEqual(self.order_manager.max_retries, 3)
        self.assertEqual(self.order_manager.retry_delay, 1.0)
        self.assertTrue(self.order_manager.auto_adjust_price)
    
    def test_create_order(self):
        """测试创建订单"""
        # 创建市价买单
        order = self.order_manager.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.01
        )
        
        # 验证API调用
        self.mock_binance_api.create_order.assert_called_once()
        
        # 验证返回的订单对象
        self.assertEqual(order.symbol, "BTCUSDT")
        self.assertEqual(order.side, OrderSide.BUY)
        self.assertEqual(order.order_type, OrderType.MARKET)
        self.assertEqual(order.quantity, 0.01)
        self.assertEqual(order.status, OrderStatus.NEW)
    
    def test_create_limit_order(self):
        """测试创建限价订单"""
        # 创建限价卖单
        order = self.order_manager.create_limit_order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            quantity=0.01,
            price=40000
        )
        
        # 验证API调用
        self.mock_binance_api.create_order.assert_called_once_with(
            symbol="BTCUSDT",
            side="SELL",
            order_type="LIMIT",
            quantity=0.01,
            price=40000,
            time_in_force='GTC'
        )
        
        # 验证返回的订单对象
        self.assertEqual(order.symbol, "BTCUSDT")
        self.assertEqual(order.side, OrderSide.SELL)
        self.assertEqual(order.order_type, OrderType.LIMIT)
        self.assertEqual(order.quantity, 0.01)
        self.assertEqual(order.price, 40000)
    
    def test_create_market_order(self):
        """测试创建市价订单"""
        # 创建市价买单
        order = self.order_manager.create_market_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=0.01
        )
        
        # 验证API调用
        self.mock_binance_api.create_order.assert_called_once_with(
            symbol="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=0.01
        )
        
        # 验证返回的订单对象
        self.assertEqual(order.symbol, "BTCUSDT")
        self.assertEqual(order.side, OrderSide.BUY)
        self.assertEqual(order.order_type, OrderType.MARKET)
        self.assertEqual(order.quantity, 0.01)
    
    def test_cancel_order(self):
        """测试取消订单"""
        # 创建一个订单
        order = self.order_manager.create_market_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=0.01
        )
        self.mock_binance_api.create_order.reset_mock()
        
        # 设置取消订单的返回值
        self.mock_binance_api.cancel_order.return_value = {
            'orderId': 12345,
            'status': 'CANCELED'
        }
        
        # 取消订单
        success = self.order_manager.cancel_order(order)
        
        # 验证API调用
        self.mock_binance_api.cancel_order.assert_called_once_with(
            symbol="BTCUSDT", 
            order_id=12345
        )
        
        # 验证返回值和订单状态
        self.assertTrue(success)
        self.assertEqual(order.status, OrderStatus.CANCELED)
    
    def test_get_open_orders(self):
        """测试获取未平仓订单"""
        # 设置API返回值
        self.mock_binance_api.get_open_orders.return_value = [
            {
                'symbol': 'BTCUSDT',
                'orderId': 12345,
                'clientOrderId': 'test-client-order-id',
                'price': '40000.00',
                'origQty': '0.01',
                'executedQty': '0.00',
                'type': 'LIMIT',
                'side': 'BUY',
                'timeInForce': 'GTC',
                'status': 'NEW'
            },
            {
                'symbol': 'ETHUSDT',
                'orderId': 67890,
                'clientOrderId': 'test-client-order-id-2',
                'price': '0.00',
                'origQty': '0.1',
                'executedQty': '0.00',
                'type': 'MARKET',
                'side': 'SELL',
                'status': 'NEW'
            }
        ]
        
        # 获取所有未平仓订单
        open_orders = self.order_manager.get_open_orders()
        
        # 验证API调用
        self.mock_binance_api.get_open_orders.assert_called_once_with()
        
        # 验证返回的订单列表
        self.assertEqual(len(open_orders), 2)
        self.assertEqual(open_orders[0].symbol, 'BTCUSDT')
        self.assertEqual(open_orders[0].order_id, 12345)
        self.assertEqual(open_orders[0].order_type, OrderType.LIMIT)
        self.assertEqual(open_orders[1].symbol, 'ETHUSDT')
        self.assertEqual(open_orders[1].order_type, OrderType.MARKET)
    
    def test_get_order_status(self):
        """测试获取订单状态"""
        # 创建一个订单
        order = self.order_manager.create_market_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=0.01
        )
        self.mock_binance_api.create_order.reset_mock()
        
        # 设置API返回值
        self.mock_binance_api._request.return_value = {
            'symbol': 'BTCUSDT',
            'orderId': 12345,
            'clientOrderId': 'test-client-order-id',
            'price': '0.00',
            'origQty': '0.01',
            'executedQty': '0.01',
            'type': 'MARKET',
            'side': 'BUY',
            'status': 'FILLED'
        }
        
        # 更新订单状态
        updated_order = self.order_manager.get_order_status(order)
        
        # 验证返回的订单状态
        self.assertEqual(updated_order.status, OrderStatus.FILLED)
        self.assertEqual(updated_order.executed_qty, 0.01)


if __name__ == '__main__':
    unittest.main()
