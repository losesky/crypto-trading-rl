#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch, MagicMock
import json
import yaml
import os
import sys
from pathlib import Path

# 添加项目根目录到路径，确保可以导入所有模块
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.append(str(PROJECT_ROOT))

from data.binance_api import BinanceAPI, BinanceAPIException, BinanceWebSocketException


class TestBinanceAPI(unittest.TestCase):
    """Binance API 测试类"""

    def setUp(self):
        """准备测试环境"""
        # 创建测试配置
        self.test_config = {
            'api': {
                'api_key': 'test_key',
                'api_secret': 'test_secret',
                'base_url': 'https://testnet.binancefuture.com',
                'websocket_url': 'wss://testnet.binancefuture.com',
                'timeout': 5000,
                'recv_window': 5000,
                'retry': {
                    'max_retries': 3,
                    'retry_delay': 1000
                }
            }
        }
        
        # 创建临时配置文件
        self.config_path = "/tmp/test_api_config.yaml"
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
            
        # 模拟响应
        self.mock_response = MagicMock()
        self.mock_response.status_code = 200
        self.mock_response.json.return_value = {"result": "success"}
        
        self.mock_error_response = MagicMock()
        self.mock_error_response.status_code = 400
        self.mock_error_response.json.return_value = {"code": -1000, "msg": "Error"}
        
        # 使用配置初始化API
        with patch('requests.Session'):
            self.api = BinanceAPI(config_path=self.config_path)
        
    def tearDown(self):
        """清理测试环境"""
        # 删除临时配置文件
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.api.API_KEY, 'test_key')
        self.assertEqual(self.api.API_SECRET, 'test_secret')
        self.assertEqual(self.api.BASE_URL, 'https://testnet.binancefuture.com')
    
    @patch('requests.Session.request')
    def test_request_success(self, mock_request):
        """测试请求成功"""
        mock_request.return_value = self.mock_response
        
        result = self.api._request('GET', '/test/endpoint')
        
        self.assertEqual(result, {"result": "success"})
        mock_request.assert_called_once()
    
    @patch('requests.Session.request')
    def test_request_error(self, mock_request):
        """测试请求错误"""
        mock_request.return_value = self.mock_error_response
        
        with self.assertRaises(BinanceAPIException):
            self.api._request('GET', '/test/endpoint')
    
    @patch('requests.Session.request')
    def test_get_server_time(self, mock_request):
        """测试获取服务器时间"""
        mock_request.return_value = self.mock_response
        
        self.api.get_server_time()
        
        mock_request.assert_called_once()
        args, kwargs = mock_request.call_args
        self.assertTrue('/fapi/v1/time' in kwargs['url'])
    
    @patch('requests.Session.request')
    def test_get_klines(self, mock_request):
        """测试获取K线数据"""
        mock_request.return_value = self.mock_response
        
        self.api.get_klines('BTCUSDT', '1h', limit=10)
        
        mock_request.assert_called_once()
        args, kwargs = mock_request.call_args
        self.assertTrue('/fapi/v1/klines' in kwargs['url'])
        self.assertTrue('symbol=BTCUSDT' in kwargs['url'])
        self.assertTrue('interval=1h' in kwargs['url'])
        self.assertTrue('limit=10' in kwargs['url'])
    
    @patch('requests.Session.request')
    def test_get_account_info(self, mock_request):
        """测试获取账户信息"""
        mock_request.return_value = self.mock_response
        
        self.api.get_account_info()
        
        mock_request.assert_called_once()
        args, kwargs = mock_request.call_args
        self.assertTrue('/fapi/v2/account' in kwargs['url'])
        self.assertTrue('signature=' in kwargs['url'])
    
    @patch('requests.Session.request')
    def test_create_order(self, mock_request):
        """测试创建订单"""
        mock_request.return_value = self.mock_response
        
        self.api.create_order(
            symbol='BTCUSDT',
            side='BUY',
            order_type='LIMIT',
            quantity=0.01,
            price=20000,
            time_in_force='GTC'
        )
        
        mock_request.assert_called_once()
        args, kwargs = mock_request.call_args
        self.assertTrue('/fapi/v1/order' in kwargs['url'])
        self.assertTrue('symbol=BTCUSDT' in kwargs['url'])
        self.assertTrue('side=BUY' in kwargs['url'])
        self.assertTrue('type=LIMIT' in kwargs['url'])
        self.assertTrue('price=20000' in kwargs['url'])
    
    @patch('websocket.WebSocketApp')
    def test_subscribe_kline(self, mock_ws):
        """测试订阅K线数据"""
        # 模拟回调函数
        mock_callback = MagicMock()
        
        # 禁用线程启动
        with patch('threading.Thread'):
            result = self.api.subscribe_kline('btcusdt', '1m', mock_callback)
        
        self.assertEqual(result, 'btcusdt@kline_1m')
        mock_ws.assert_called_once()

if __name__ == '__main__':
    unittest.main()
