#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交易执行代理类
提供与main.py兼容的接口，将调用转发到TradingExecutor
"""
import logging
import yaml
import os
from typing import Dict, List, Optional, Union, Any, Tuple
import asyncio

from data.binance_api import BinanceAPI
from trading.execution import TradingExecutor
from trading.order_manager import OrderManager
from trading.position_manager import PositionManager


class Executor:
    """
    交易执行代理类，提供main.py所需的API接口
    """
    
    def __init__(self, api: BinanceAPI, order_manager: OrderManager, position_manager: PositionManager, config_path: str = None):
        """
        初始化交易执行代理
        
        Args:
            api: API客户端
            order_manager: 订单管理器
            position_manager: 仓位管理器
            config_path: 配置文件路径
        """
        self.logger = logging.getLogger('Executor')
        self.api = api
        self.order_manager = order_manager
        self.position_manager = position_manager
        
        # 创建实际的TradingExecutor实例，并传递API和配置路径
        self.trading_executor = TradingExecutor(config_path=config_path, api=api)
        self.logger.info("交易执行器初始化完成")
    
    async def open_long(self, symbol: str, quantity: float, price: Optional[float] = None, is_usdt_amount: bool = True) -> Dict[str, Any]:
        """
        开多仓
        
        Args:
            symbol: 交易对
            quantity: 数量（如果is_usdt_amount=True，则为USDT金额）
            price: 价格(可选，默认为市价单)
            is_usdt_amount: 是否为USDT金额（默认为True，表示quantity为USDT金额而非资产数量）
            
        Returns:
            Dict[str, Any]: 开仓结果
        """
        if is_usdt_amount:
            self.logger.info(f"开多仓: {symbol}, USDT金额: {quantity}")
        else:
            self.logger.info(f"开多仓: {symbol}, 数量: {quantity}")
        
        # 调用position_manager的open_position方法 (异步)
        result = await self.position_manager.open_position(
            symbol=symbol,
            side='LONG',
            quantity=quantity,
            price=price,
            is_usdt_amount=is_usdt_amount
        )
        
        return result
    
    async def close_long(self, symbol: str, quantity: Optional[float] = None) -> Dict[str, Any]:
        """
        平多仓
        
        Args:
            symbol: 交易对
            quantity: 数量(可选，默认全部平仓)
            
        Returns:
            Dict[str, Any]: 平仓结果
        """
        self.logger.info(f"平多仓: {symbol}, 数量: {quantity if quantity else '全部'}")
        
        # 调用position_manager的close_position方法
        result = self.position_manager.close_position(
            symbol=symbol,
            quantity=quantity
        )
        
        return result
    
    async def open_short(self, symbol: str, quantity: float, price: Optional[float] = None, is_usdt_amount: bool = True) -> Dict[str, Any]:
        """
        开空仓
        
        Args:
            symbol: 交易对
            quantity: 数量（如果is_usdt_amount=True，则为USDT金额）
            price: 价格(可选，默认为市价单)
            is_usdt_amount: 是否为USDT金额（默认为True，表示quantity为USDT金额而非资产数量）
            
        Returns:
            Dict[str, Any]: 开仓结果
        """
        if is_usdt_amount:
            self.logger.info(f"开空仓: {symbol}, USDT金额: {quantity}")
        else:
            self.logger.info(f"开空仓: {symbol}, 数量: {quantity}")
        
        # 调用position_manager的open_position方法（异步）
        result = await self.position_manager.open_position(
            symbol=symbol,
            side='SHORT',
            quantity=quantity,
            price=price,
            is_usdt_amount=is_usdt_amount
        )
        
        return result
    
    async def close_short(self, symbol: str, quantity: Optional[float] = None) -> Dict[str, Any]:
        """
        平空仓
        
        Args:
            symbol: 交易对
            quantity: 数量(可选，默认全部平仓)
            
        Returns:
            Dict[str, Any]: 平仓结果
        """
        self.logger.info(f"平空仓: {symbol}, 数量: {quantity if quantity else '全部'}")
        
        # 调用position_manager的close_position方法
        result = self.position_manager.close_position(
            symbol=symbol,
            quantity=quantity
        )
        
        return result
