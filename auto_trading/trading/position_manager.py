"""
仓位管理模块
负责管理交易仓位、计算风险暴露和执行仓位调整策略
"""
import logging
import yaml
import json
import os
import time
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum

from data.binance_api import BinanceAPI
from trading.order_manager import OrderManager, Order, OrderSide, OrderType, TimeInForce

class PositionStatus(Enum):
    """仓位状态枚举"""
    NONE = "NONE"             # 无仓位
    LONG = "LONG"             # 多仓
    SHORT = "SHORT"           # 空仓
    CLOSING_LONG = "CLOSING_LONG"   # 正在平多
    CLOSING_SHORT = "CLOSING_SHORT" # 正在平空

class PositionManager:
    """
    仓位管理类
    负责管理交易仓位和执行仓位调整策略
    """
    
    def __init__(self, order_manager: OrderManager, config_path: str = None):
        """
        初始化仓位管理器
        
        Args:
            order_manager: 订单管理器
            config_path: 配置文件路径
        """
        self.logger = logging.getLogger('PositionManager')
        self.order_manager = order_manager
        self.binance_api = order_manager.binance_api
        
        # 导入os模块并使用别名避免混淆
        import os as os_module
        from pathlib import Path
        
        # 使用传入的配置路径或默认值
        if config_path is None:
            # 使用当前文件的相对路径找到配置文件
            current_dir = Path(__file__).parent.parent
            config_path = os_module.path.join(current_dir, 'config', 'risk_config.yaml')
            
        self._load_config(config_path)
        self.positions = {}  # symbol -> position_data
        
        # 待处理的止损止盈符号集合
        self._pending_stop_loss_symbols = set()
        
        # 创建仓位日志目录
        os_module.makedirs(os_module.path.dirname(self.position_log_path), exist_ok=True)
    
    def _load_config(self, config_path: str) -> None:
        """
        加载配置
        
        Args:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # 使用绝对路径而非相对路径
            from pathlib import Path
            import os as os_module
            current_dir = Path(__file__).parent.parent
            self.position_log_path = os_module.path.join(current_dir, "logs", "positions.json")
            
            risk_config = config.get('risk', {})
            capital_config = risk_config.get('capital', {})
            position_config = risk_config.get('position_management', {})
            
            # 资金配置
            self.total_capital_limit = capital_config.get('total_capital_limit', 0.5)
            self.initial_position_size = capital_config.get('initial_position_size', 0.02)
            self.max_position_size = capital_config.get('max_position_size', 0.1)
            
            # 仓位管理配置
            self.max_holding_time = position_config.get('max_holding_time', 72)  # 小时
            self.pyramid_scaling = position_config.get('pyramid_scaling', True)
            self.scale_in_steps = position_config.get('scale_in_steps', 3)
            self.scale_out_steps = position_config.get('scale_out_steps', 2)
            
            # 止盈止损配置
            stop_settings = risk_config.get('stop_settings', {})
            self.trailing_stop = stop_settings.get('trailing_stop', True)
            self.trailing_stop_distance = stop_settings.get('trailing_stop_distance', 0.02)
            self.fixed_stop_loss = stop_settings.get('fixed_stop_loss', 0.03)
            self.take_profit = stop_settings.get('take_profit', 0.05)
            
            self.logger.info("成功加载仓位管理配置")
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            raise
    
    async def update_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        更新所有仓位信息
        
        Returns:
            Dict[str, Dict[str, Any]]: 更新后的所有仓位信息
        """
        try:
            # 从交易所获取最新仓位信息
            position_risk = await self.binance_api.get_position_risk_async()
            
            # 更新本地仓位数据
            for position in position_risk:
                symbol = position['symbol']
                position_amt = float(position['positionAmt'])
                
                # 仓位方向
                if position_amt > 0:
                    status = PositionStatus.LONG
                elif position_amt < 0:
                    status = PositionStatus.SHORT
                else:
                    status = PositionStatus.NONE
                    
                # 更新或创建仓位记录
                if symbol not in self.positions:
                    self.positions[symbol] = {
                        'symbol': symbol,
                        'status': status,
                        'amount': abs(position_amt),
                        'entry_price': float(position['entryPrice']) if float(position['entryPrice']) > 0 else 0,
                        'mark_price': float(position['markPrice']),
                        'unrealized_pnl': float(position['unRealizedProfit']),
                        'leverage': float(position['leverage']),
                        'isolated': position['isolated'],
                        'entry_time': datetime.now().isoformat(),
                        'last_update': datetime.now().isoformat(),
                        'highest_price': float(position['markPrice']),
                        'lowest_price': float(position['markPrice']),
                        'trailing_stop_price': 0,
                        'stop_loss_price': 0,
                        'take_profit_price': 0,
                        'accumulated_realized_pnl': 0,
                        'average_cost': float(position['entryPrice']) if float(position['entryPrice']) > 0 else 0,
                        'scale_in_count': 0,
                        'scale_out_count': 0,
                        'stop_loss_order_id': None,
                        'take_profit_order_id': None
                    }
                else:
                    # 更新现有仓位
                    current_pos = self.positions[symbol]
                    current_pos['status'] = status
                    current_pos['amount'] = abs(position_amt)
                    
                    # 只在有仓位时更新其他字段
                    if position_amt != 0:
                        current_pos['entry_price'] = float(position['entryPrice'])
                        current_pos['mark_price'] = float(position['markPrice'])
                        current_pos['unrealized_pnl'] = float(position['unRealizedProfit'])
                        current_pos['leverage'] = float(position['leverage'])
                        current_pos['isolated'] = position['isolated']
                        current_pos['last_update'] = datetime.now().isoformat()
                        
                        # 更新最高/最低价格，用于跟踪止损
                        if status == PositionStatus.LONG:
                            if float(position['markPrice']) > current_pos['highest_price']:
                                current_pos['highest_price'] = float(position['markPrice'])
                                # 如果启用了追踪止损，更新止损价格
                                if self.trailing_stop:
                                    current_pos['trailing_stop_price'] = current_pos['highest_price'] * (1 - self.trailing_stop_distance)
                        elif status == PositionStatus.SHORT:
                            if float(position['markPrice']) < current_pos['lowest_price'] or current_pos['lowest_price'] == 0:
                                current_pos['lowest_price'] = float(position['markPrice'])
                                # 如果启用了追踪止损，更新止损价格
                                if self.trailing_stop:
                                    current_pos['trailing_stop_price'] = current_pos['lowest_price'] * (1 + self.trailing_stop_distance)
                    else:
                        # 如果仓位已平，重置部分字段
                        current_pos['highest_price'] = 0
                        current_pos['lowest_price'] = 0
                        current_pos['trailing_stop_price'] = 0
                        current_pos['stop_loss_price'] = 0
                        current_pos['take_profit_price'] = 0
                        current_pos['stop_loss_order_id'] = None
                        current_pos['take_profit_order_id'] = None
                        current_pos['scale_in_count'] = 0
                        current_pos['scale_out_count'] = 0
            
            # 处理待设置的止损止盈
            pending_symbols = list(self._pending_stop_loss_symbols)
            for symbol in pending_symbols:
                if symbol in self.positions:
                    position = self.positions[symbol]
                    if position['status'] != PositionStatus.NONE and position['entry_price'] > 0:
                        # 确定仓位方向
                        side = 'LONG' if position['status'] == PositionStatus.LONG else 'SHORT'
                        # 设置止损止盈
                        self._set_stop_loss_take_profit(symbol, side, position['entry_price'])
                        # 从待处理列表中移除
                        self._pending_stop_loss_symbols.remove(symbol)
                        self.logger.info(f"延迟设置止损止盈成功: {symbol}")
            
            # 记录仓位信息
            self.log_positions()
            
        except Exception as e:
            self.logger.error(f"更新仓位信息失败: {e}")
            raise
    
    async def open_position(self, symbol: str, side: str, quantity: float, 
                    price: Optional[float] = None, leverage: int = 1, is_usdt_amount: bool = True) -> Dict[str, Any]:
        """
        开仓
        
        Args:
            symbol: 交易对
            side: 仓位方向 ('LONG' or 'SHORT')
            quantity: 数量（如果is_usdt_amount=True，则为USDT金额）
            price: 价格，如果为None则使用市价
            leverage: 杠杆倍数
            is_usdt_amount: 是否为USDT金额（默认为True，表示quantity为USDT金额而非资产数量）
            
        Returns:
            Dict[str, Any]: 开仓结果
        """
        try:
            # 确保仓位方向有效
            if side not in ['LONG', 'SHORT']:
                raise ValueError(f"无效的仓位方向: {side}")
            
            # 如果quantity是USDT金额，将其转换为实际的交易数量
            if is_usdt_amount:
                self.logger.info(f"输入的是USDT金额: {quantity} USDT")
                quantity = self.binance_api.calculate_quantity_from_usdt(symbol, quantity)
                self.logger.info(f"转换后的数量: {quantity}")
            
            # 根据交易对格式化数量，确保符合精度要求
            quantity = self.binance_api.format_quantity(symbol, quantity)
            self.logger.info(f"格式化后的数量: {quantity}")
            
            if price is not None:
                price = self.binance_api.format_price(symbol, price)
                self.logger.info(f"格式化后的价格: {price}")

            # 直接从API获取最新持仓信息，确保数据最新
            positions = self.binance_api.get_position_risk()
            has_opposite_position = False
            
            # 检查是否有反向持仓
            for pos in positions:
                if pos['symbol'] == symbol:
                    position_amt = float(pos['positionAmt'])
                    
                    # 如果没有持仓，直接跳过
                    if position_amt == 0:
                        continue
                    
                    # 检查持仓方向是否与开仓方向相反
                    if (side == 'LONG' and position_amt < 0) or (side == 'SHORT' and position_amt > 0):
                        has_opposite_position = True
                        pos_direction = "空头" if position_amt < 0 else "多头"
                        self.logger.warning(f"尝试开{side}仓，但已有反向{pos_direction}仓位({abs(position_amt)}张)，先平掉现有仓位")
                        
                        # 先平掉现有仓位
                        close_result = self.close_position(symbol)
                        if isinstance(close_result, dict) and 'error' in close_result:
                            self.logger.error(f"平掉现有仓位失败: {close_result['error']}")
                            return {'error': f"无法处理反向仓位: {close_result['error']}"}
                        
                        # 确保订单执行和仓位更新（这里添加一个短暂延迟）
                        import asyncio
                        await asyncio.sleep(1)  # 等待平仓订单执行
                        await self.update_positions()  # 更新仓位信息
                        break
            
            # 设置杠杆
            leverage_result = self._set_leverage(symbol, leverage)
            
            # 转换为订单方向
            order_side = OrderSide.BUY if side == 'LONG' else OrderSide.SELL
            
            # 创建和发送订单
            if price:
                # 限价单
                order = self.order_manager.create_limit_order(
                    symbol=symbol,
                    side=order_side,
                    quantity=quantity,
                    price=price,
                    time_in_force=TimeInForce.GTC
                )
            else:
                # 市价单
                order = self.order_manager.create_market_order(
                    symbol=symbol,
                    side=order_side,
                    quantity=quantity
                )
            
            # 发送订单
            result = self.order_manager.send_order(order)
            
            # 获取实际成交价格（优先使用成交价格，如果没有则尝试获取订单价格或当前市价）
            entry_price = 0
            if hasattr(order, 'avg_price') and order.avg_price > 0:
                entry_price = order.avg_price
            elif price is not None and price > 0:
                entry_price = price
            else:
                # 如果没有价格信息，尝试从API获取当前市价
                try:
                    ticker = self.binance_api.get_symbol_price_ticker(symbol)
                    if ticker and 'price' in ticker:
                        entry_price = float(ticker['price'])
                        self.logger.info(f"使用当前市价作为入场价格: {entry_price}")
                except Exception as e:
                    self.logger.warning(f"获取市价失败，无法设置止损止盈: {e}")
            
            # 更新仓位信息
            self.update_positions()
            
            # 如果有有效的入场价格，设置止损和止盈
            if entry_price > 0:
                self._set_stop_loss_take_profit(symbol, side, entry_price, quantity)
            else:
                self.logger.warning(f"无有效入场价格，暂不设置止损止盈，将在价格更新后设置")
                # 记录需要设置止损止盈的仓位，后续在价格更新时处理
                self._pending_stop_loss_symbols.add(symbol)
            
            return {
                'order': order.to_dict(),
                'result': result,
                'quantity': quantity,
                'price': entry_price
            }
            
        except Exception as e:
            self.logger.error(f"开仓失败 {symbol} {side}: {e}")
            raise
    
    def close_position(self, symbol: str, percentage: float = 1.0) -> Dict[str, Any]:
        """
        平仓
        
        Args:
            symbol: 交易对
            percentage: 平仓比例 (0.0-1.0)
            
        Returns:
            Dict[str, Any]: 平仓结果
        """
        try:
            # 确保有仓位
            if symbol not in self.positions or self.positions[symbol]['status'] == PositionStatus.NONE:
                self.logger.warning(f"尝试平仓但没有仓位: {symbol}")
                return {'error': 'No position to close'}
                
            position = self.positions[symbol]
            current_amount = position['amount']
            close_amount = current_amount * percentage
            
            # 确定平仓方向
            if position['status'] == PositionStatus.LONG:
                order_side = OrderSide.SELL
                new_status = PositionStatus.CLOSING_LONG
            else:  # SHORT
                order_side = OrderSide.BUY
                new_status = PositionStatus.CLOSING_SHORT
                
            # 格式化数量以确保符合交易所精度要求
            formatted_close_amount = self.binance_api.format_quantity(symbol, close_amount)
            self.logger.info(f"格式化后的平仓数量: {formatted_close_amount} (原始: {close_amount})")
                
            # 创建市价平仓订单
            order = self.order_manager.create_market_order(
                symbol=symbol,
                side=order_side,
                quantity=formatted_close_amount,
                reduce_only=True
            )
            
            # 发送订单
            result = self.order_manager.send_order(order)
            
            # 更新仓位状态
            position['status'] = new_status
            
            # 取消现有止损止盈订单
            self._cancel_stop_orders(symbol)
            
            # 更新仓位信息
            self.update_positions()
            
            return {
                'order': order.to_dict(),
                'result': result
            }
            
        except Exception as e:
            self.logger.error(f"平仓失败 {symbol}: {e}")
            raise
    
    def scale_in_position(self, symbol: str, additional_amount: float, 
                        price: Optional[float] = None) -> Dict[str, Any]:
        """
        加仓
        
        Args:
            symbol: 交易对
            additional_amount: 增加的数量
            price: 价格，如果为None则使用市价
            
        Returns:
            Dict[str, Any]: 加仓结果
        """
        try:
            # 确保有仓位
            if symbol not in self.positions or self.positions[symbol]['status'] == PositionStatus.NONE:
                self.logger.warning(f"尝试加仓但没有仓位: {symbol}")
                return {'error': 'No position to scale in'}
                
            position = self.positions[symbol]
            
            # 检查加仓次数是否超过限制
            if position['scale_in_count'] >= self.scale_in_steps:
                self.logger.warning(f"已达到最大加仓次数: {symbol}")
                return {'error': 'Maximum scale-in steps reached'}
                
            # 确定加仓方向
            if position['status'] == PositionStatus.LONG:
                side = 'LONG'
            else:  # SHORT
                side = 'SHORT'
                
            # 执行加仓
            result = self.open_position(
                symbol=symbol,
                side=side,
                quantity=additional_amount,
                price=price,
                leverage=position['leverage']
            )
            
            # 更新加仓次数
            position['scale_in_count'] += 1
            
            # 更新止损和止盈
            self._update_stop_loss_take_profit(symbol)
            
            return result
            
        except Exception as e:
            self.logger.error(f"加仓失败 {symbol}: {e}")
            raise
    
    def scale_out_position(self, symbol: str, percentage: float = 0.5) -> Dict[str, Any]:
        """
        减仓
        
        Args:
            symbol: 交易对
            percentage: 减仓比例 (0.0-1.0)
            
        Returns:
            Dict[str, Any]: 减仓结果
        """
        try:
            # 确保有仓位
            if symbol not in self.positions or self.positions[symbol]['status'] == PositionStatus.NONE:
                self.logger.warning(f"尝试减仓但没有仓位: {symbol}")
                return {'error': 'No position to scale out'}
                
            position = self.positions[symbol]
            
            # 检查减仓次数是否超过限制
            if position['scale_out_count'] >= self.scale_out_steps:
                self.logger.warning(f"已达到最大减仓次数: {symbol}")
                return {'error': 'Maximum scale-out steps reached'}
                
            # 执行部分平仓
            result = self.close_position(symbol, percentage)
            
            # 更新减仓次数
            position['scale_out_count'] += 1
            
            # 更新止损和止盈
            self._update_stop_loss_take_profit(symbol)
            
            return result
            
        except Exception as e:
            self.logger.error(f"减仓失败 {symbol}: {e}")
            raise
    
    def _set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """
        设置杠杆倍数
        
        Args:
            symbol: 交易对
            leverage: 杠杆倍数
            
        Returns:
            Dict[str, Any]: API响应
        """
        try:
            # 先检查当前持仓的杠杆情况
            current_leverage = 1  # 默认杠杆
            has_position = False
            
            # 检查是否有任何持仓（多头或空头）
            position_risk = self.binance_api.get_position_risk()
            for pos in position_risk:
                if pos['symbol'] == symbol and abs(float(pos['positionAmt'])) > 0:
                    has_position = True
                    current_leverage = float(pos['leverage'])
                    break
                    
            # 如果已有持仓，就使用当前杠杆而不是尝试改变它
            if has_position:
                self.logger.info(f"有持仓时使用当前杠杆 {symbol} x{current_leverage}")
                return {"leverage": current_leverage}  # 返回当前杠杆，不做API调用
            
            # 正常设置杠杆
            response = self.binance_api._request(
                method='POST',
                endpoint='/fapi/v1/leverage',
                signed=True,
                symbol=symbol,
                leverage=leverage
            )
            
            self.logger.info(f"设置杠杆 {symbol} x{leverage}")
            return response
            
        except Exception as e:
            self.logger.error(f"设置杠杆失败 {symbol} x{leverage}: {e}")
            raise
    
    def _set_stop_loss_take_profit(self, symbol: str, side: str, entry_price: float, quantity: float = None) -> None:
        """
        设置止损和止盈订单
        
        Args:
            symbol: 交易对
            side: 仓位方向 ('LONG' or 'SHORT')
            entry_price: 入场价格
            quantity: 交易数量（可选，如果不提供则从仓位信息中获取）
        """
        try:
            if not entry_price:
                self.logger.warning(f"没有有效的入场价格，无法设置止损止盈")
                return
                
            position = self.positions.get(symbol)
            if not position:
                self.logger.warning(f"没有找到仓位: {symbol}")
                return
            
            # 确定使用的数量
            if quantity is None or quantity <= 0:
                # 从交易所获取最新的持仓数量
                try:
                    position_risk = self.binance_api.get_position_risk(symbol)
                    if position_risk:
                        for pos in position_risk:
                            if pos['symbol'] == symbol:
                                actual_amount = abs(float(pos['positionAmt']))
                                if actual_amount > 0:
                                    quantity = actual_amount
                                    break
                    
                    if quantity is None or quantity <= 0:
                        self.logger.warning(f"无法获取有效的持仓数量，取消设置止损止盈: {symbol}")
                        return
                        
                except Exception as e:
                    self.logger.error(f"获取持仓数量失败: {e}")
                    return
            
            # 格式化数量以确保符合交易所精度要求
            quantity = self.binance_api.format_quantity(symbol, quantity)
            self.logger.info(f"止损止盈订单数量: {quantity}")
            
            if quantity <= 0:
                self.logger.warning(f"格式化后的数量为0，无法设置止损止盈: {symbol}")
                return
                
            # 计算止损和止盈价格
            if side == 'LONG':
                stop_loss_price = entry_price * (1 - self.fixed_stop_loss)
                take_profit_price = entry_price * (1 + self.take_profit)
            else:  # SHORT
                stop_loss_price = entry_price * (1 + self.fixed_stop_loss)
                take_profit_price = entry_price * (1 - self.take_profit)
                
            # 格式化价格以确保符合交易所精度要求
            stop_loss_price = self.binance_api.format_price(symbol, stop_loss_price)
            take_profit_price = self.binance_api.format_price(symbol, take_profit_price)
            self.logger.info(f"格式化后的止损价格: {stop_loss_price}, 止盈价格: {take_profit_price}")
                
            # 更新仓位记录
            position['stop_loss_price'] = stop_loss_price
            position['take_profit_price'] = take_profit_price
            
            # 创建止损订单
            if side == 'LONG':
                stop_order = self.order_manager.create_stop_market_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=quantity,
                    stop_price=stop_loss_price,
                    reduce_only=True
                )
            else:  # SHORT
                stop_order = self.order_manager.create_stop_market_order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=quantity,
                    stop_price=stop_loss_price,
                    reduce_only=True
                )
                
            stop_result = self.order_manager.send_order(stop_order)
            position['stop_loss_order_id'] = stop_order.client_order_id
            
            # 创建止盈订单
            if side == 'LONG':
                tp_order = self.order_manager.create_take_profit_market_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=quantity,
                    stop_price=take_profit_price,
                    reduce_only=True
                )
            else:  # SHORT
                tp_order = self.order_manager.create_take_profit_market_order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=quantity,
                    stop_price=take_profit_price,
                    reduce_only=True
                )
                
            tp_result = self.order_manager.send_order(tp_order)
            position['take_profit_order_id'] = tp_order.client_order_id
            
            self.logger.info(f"设置止损{stop_loss_price}和止盈{take_profit_price}: {symbol}")
            
        except Exception as e:
            self.logger.error(f"设置止损止盈失败 {symbol}: {e}")
    
    def _update_stop_loss_take_profit(self, symbol: str) -> None:
        """
        更新止损和止盈订单
        
        Args:
            symbol: 交易对
        """
        try:
            position = self.positions.get(symbol)
            if not position or position['status'] == PositionStatus.NONE:
                return
                
            # 取消现有的止损止盈订单
            self._cancel_stop_orders(symbol)
            
            # 重新设置止损止盈
            side = 'LONG' if position['status'] == PositionStatus.LONG else 'SHORT'
            self._set_stop_loss_take_profit(symbol, side, position['entry_price'])
            
        except Exception as e:
            self.logger.error(f"更新止损止盈失败 {symbol}: {e}")
    
    def _cancel_stop_orders(self, symbol: str) -> None:
        """
        取消止损和止盈订单
        
        Args:
            symbol: 交易对
        """
        try:
            position = self.positions.get(symbol)
            if not position:
                return
                
            # 取消止损订单
            if position['stop_loss_order_id']:
                try:
                    self.order_manager.cancel_order(position['stop_loss_order_id'])
                except Exception as e:
                    self.logger.warning(f"取消止损订单失败 {position['stop_loss_order_id']}: {e}")
                position['stop_loss_order_id'] = None
            
            # 取消止盈订单
            if position['take_profit_order_id']:
                try:
                    self.order_manager.cancel_order(position['take_profit_order_id'])
                except Exception as e:
                    self.logger.warning(f"取消止盈订单失败 {position['take_profit_order_id']}: {e}")
                position['take_profit_order_id'] = None
                
        except Exception as e:
            self.logger.error(f"取消止损止盈订单失败 {symbol}: {e}")
    
    def check_trailing_stop(self, symbol: str) -> bool:
        """
        检查是否触发追踪止损
        
        Args:
            symbol: 交易对
            
        Returns:
            bool: 是否触发追踪止损
        """
        try:
            position = self.positions.get(symbol)
            if not position or position['status'] == PositionStatus.NONE:
                return False
                
            if not self.trailing_stop or position['trailing_stop_price'] == 0:
                return False
                
            current_price = position['mark_price']
            
            # 检查是否触发追踪止损
            if position['status'] == PositionStatus.LONG and current_price <= position['trailing_stop_price']:
                self.logger.info(f"触发追踪止损 {symbol} @ {position['trailing_stop_price']}")
                self.close_position(symbol)
                return True
            elif position['status'] == PositionStatus.SHORT and current_price >= position['trailing_stop_price']:
                self.logger.info(f"触发追踪止损 {symbol} @ {position['trailing_stop_price']}")
                self.close_position(symbol)
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"检查追踪止损失败 {symbol}: {e}")
            return False
    
    def check_holding_time(self, symbol: str) -> bool:
        """
        检查持仓时间是否超过最大持仓时间
        
        Args:
            symbol: 交易对
            
        Returns:
            bool: 是否超过最大持仓时间
        """
        try:
            position = self.positions.get(symbol)
            if not position or position['status'] == PositionStatus.NONE:
                return False
                
            # 计算持仓时间
            entry_time = datetime.fromisoformat(position['entry_time'])
            current_time = datetime.now()
            holding_hours = (current_time - entry_time).total_seconds() / 3600
            
            # 检查是否超过最大持仓时间
            if holding_hours >= self.max_holding_time:
                self.logger.info(f"超过最大持仓时间 {symbol} ({holding_hours:.1f}h > {self.max_holding_time}h)")
                self.close_position(symbol)
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"检查持仓时间失败 {symbol}: {e}")
            return False
    
    def calculate_position_size(self, symbol: str, account_balance: float, 
                             risk_percentage: float, price: float, 
                             stop_loss_percentage: Optional[float] = None) -> float:
        """
        计算仓位大小
        
        Args:
            symbol: 交易对
            account_balance: 账户余额
            risk_percentage: 风险百分比
            price: 当前价格
            stop_loss_percentage: 止损百分比，如果为None则使用默认止损比例
            
        Returns:
            float: 仓位大小
        """
        try:
            if stop_loss_percentage is None:
                stop_loss_percentage = self.fixed_stop_loss
                
            # 计算风险金额
            risk_amount = account_balance * risk_percentage
            
            # 计算仓位大小
            position_size = risk_amount / (price * stop_loss_percentage)
            
            # 确保不超过最大仓位大小
            max_position = account_balance * self.max_position_size
            position_size = min(position_size, max_position)
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"计算仓位大小失败: {e}")
            return 0.0
    
    async def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        获取指定交易对的仓位信息
        
        Args:
            symbol: 交易对
            
        Returns:
            Dict[str, Any]: 仓位信息
        """
        # 确保仓位信息是最新的
        await self.update_positions()
        return self.positions.get(symbol, {
            'symbol': symbol,
            'status': PositionStatus.NONE,
            'amount': 0,
            'entry_price': 0,
            'mark_price': 0,
            'unrealized_pnl': 0
        })
    
    async def get_all_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有仓位信息
        
        Returns:
            Dict[str, Dict[str, Any]]: 所有仓位信息
        """
        # 确保仓位信息是最新的
        await self.update_positions()
        return self.positions
    
    def log_positions(self) -> None:
        """记录仓位信息到日志文件"""
        try:
            # 导入os模块并使用别名避免混淆
            import os as os_module
            
            # 读取现有日志
            position_log = []
            if os_module.path.exists(self.position_log_path):
                try:
                    with open(self.position_log_path, 'r') as f:
                        position_log = json.load(f)
                except json.JSONDecodeError:
                    position_log = []
            
            # 创建JSON可序列化的仓位记录
            serializable_positions = {}
            for symbol, pos in self.positions.items():
                serializable_pos = pos.copy()
                # 将枚举转换为字符串
                if isinstance(serializable_pos.get('status'), PositionStatus):
                    serializable_pos['status'] = serializable_pos['status'].value
                serializable_positions[symbol] = serializable_pos
            
            # 添加新仓位记录
            position_log.append({
                'timestamp': datetime.now().isoformat(),
                'positions': serializable_positions
            })
            
            # 如果日志太长，移除最早的记录
            max_log_entries = 1000
            while len(position_log) > max_log_entries:
                position_log.pop(0)
            
            # 写入日志文件
            with open(self.position_log_path, 'w') as f:
                json.dump(position_log, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"记录仓位信息到日志文件失败: {e}")
    
    def get_position_summary(self) -> Dict[str, Any]:
        """
        获取仓位摘要信息
        
        Returns:
            Dict[str, Any]: 仓位摘要
        """
        # 确保仓位信息是最新的
        self.update_positions()
        
        # 计算总计
        total_long_value = 0
        total_short_value = 0
        total_unrealized_pnl = 0
        
        for symbol, position in self.positions.items():
            if position['status'] == PositionStatus.LONG:
                position_value = position['amount'] * position['mark_price']
                total_long_value += position_value
            elif position['status'] == PositionStatus.SHORT:
                position_value = position['amount'] * position['mark_price']
                total_short_value += position_value
                
            total_unrealized_pnl += position['unrealized_pnl']
        
        # 获取账户信息
        account_info = self.binance_api.get_account_info()
        available_balance = float(account_info['availableBalance'])
        total_margin_balance = float(account_info['totalMarginBalance'])
        
        return {
            'total_long_value': total_long_value,
            'total_short_value': total_short_value,
            'total_position_value': total_long_value + total_short_value,
            'total_unrealized_pnl': total_unrealized_pnl,
            'available_balance': available_balance,
            'total_margin_balance': total_margin_balance,
            'position_count': len([p for p in self.positions.values() if p['status'] != PositionStatus.NONE]),
            'long_position_count': len([p for p in self.positions.values() if p['status'] == PositionStatus.LONG]),
            'short_position_count': len([p for p in self.positions.values() if p['status'] == PositionStatus.SHORT]),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_position_size(self, symbol: str) -> float:
        """
        获取指定交易对的仓位大小
        
        Args:
            symbol: 交易对
            
        Returns:
            float: 仓位大小，如果没有仓位则返回0
        """
        position = self.positions.get(symbol, {})
        if position.get('status', PositionStatus.NONE) == PositionStatus.NONE:
            return 0.0
        return position.get('amount', 0.0)
        
    def get_open_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有活跃（非空）仓位信息
        
        Returns:
            Dict[str, Dict[str, Any]]: 所有活跃的仓位信息
        """
        open_positions = {}
        for symbol, pos in self.positions.items():
            # 只返回有实际持仓数量且状态不为NONE的仓位
            if pos.get('amount', 0) > 0 and pos.get('status') != PositionStatus.NONE:
                open_positions[symbol] = pos
                
        return open_positions
    
    def get_open_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有活跃（非空）仓位信息
        
        Returns:
            Dict[str, Dict[str, Any]]: 所有活跃的仓位信息
        """
        open_positions = {}
        for symbol, pos in self.positions.items():
            # 只返回有实际持仓数量且状态不为NONE的仓位
            if pos.get('amount', 0) > 0 and pos.get('status') != PositionStatus.NONE:
                open_positions[symbol] = pos
                
        return open_positions
