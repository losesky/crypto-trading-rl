"""
订单管理模块
负责创建、跟踪和管理交易订单
"""
import logging
import yaml
import uuid
import time
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
from enum import Enum
import json
import os

from data.binance_api import BinanceAPI

class OrderStatus(Enum):
    """订单状态枚举"""
    NEW = "NEW"                         # 新建订单
    PARTIALLY_FILLED = "PARTIALLY_FILLED" # 部分成交
    FILLED = "FILLED"                   # 全部成交
    CANCELED = "CANCELED"               # 已取消
    REJECTED = "REJECTED"               # 被拒绝
    EXPIRED = "EXPIRED"                 # 已过期
    PENDING = "PENDING"                 # 等待中（本地状态）
    FAILED = "FAILED"                   # 失败（本地状态）
    SYSTEM_ERROR = "SYSTEM_ERROR"       # 系统错误（本地状态）

class OrderType(Enum):
    """订单类型枚举"""
    LIMIT = "LIMIT"                     # 限价单
    MARKET = "MARKET"                   # 市价单
    STOP = "STOP"                       # 止损限价单
    STOP_MARKET = "STOP_MARKET"         # 止损市价单
    TAKE_PROFIT = "TAKE_PROFIT"         # 止盈限价单
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET" # 止盈市价单

class OrderSide(Enum):
    """订单方向枚举"""
    BUY = "BUY"                         # 买入
    SELL = "SELL"                       # 卖出

class PositionSide(Enum):
    """持仓方向枚举"""
    BOTH = "BOTH"                       # 双向持仓
    LONG = "LONG"                       # 多头
    SHORT = "SHORT"                     # 空头

class TimeInForce(Enum):
    """有效时间枚举"""
    GTC = "GTC"                         # 成交为止
    IOC = "IOC"                         # 无法立即成交的部分就撤销
    FOK = "FOK"                         # 无法全部立即成交就撤销
    GTX = "GTX"                         # 成为挂单为止

class Order:
    """订单对象"""
    
    def __init__(self, symbol: str, order_side: OrderSide, order_type: OrderType, 
               quantity: float, price: Optional[float] = None, 
               position_side: Optional[PositionSide] = None,
               time_in_force: Optional[TimeInForce] = None,
               stop_price: Optional[float] = None,
               client_order_id: Optional[str] = None,
               reduce_only: bool = False):
        """
        初始化订单对象
        
        Args:
            symbol: 交易对
            order_side: 订单方向
            order_type: 订单类型
            quantity: 订单数量
            price: 订单价格(限价单必需)
            position_side: 持仓方向
            time_in_force: 有效时间
            stop_price: 触发价格(止损/止盈单必需)
            client_order_id: 客户端订单ID
            reduce_only: 是否只减仓
        """
        self.symbol = symbol
        self.order_side = order_side
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.position_side = position_side or PositionSide.BOTH
        self.time_in_force = time_in_force
        self.stop_price = stop_price
        self.client_order_id = client_order_id or f"a_{uuid.uuid4().hex[:16]}"
        self.reduce_only = reduce_only
        
        self.order_id = None
        self.status = OrderStatus.PENDING
        self.create_time = datetime.now()
        self.update_time = datetime.now()
        self.filled_quantity = 0.0
        self.avg_price = 0.0
        self.commission = 0.0
        self.realized_pnl = 0.0
        
    def update(self, order_data: Dict[str, Any]) -> None:
        """
        使用交易所返回的数据更新订单
        
        Args:
            order_data: 交易所返回的订单数据
        """
        if 'orderId' in order_data:
            self.order_id = order_data['orderId']
        
        if 'status' in order_data:
            try:
                self.status = OrderStatus(order_data['status'])
            except (ValueError, KeyError):
                self.status = OrderStatus.SYSTEM_ERROR
        
        if 'executedQty' in order_data:
            self.filled_quantity = float(order_data['executedQty'])
        
        if 'avgPrice' in order_data:
            self.avg_price = float(order_data['avgPrice'])
        elif 'price' in order_data:
            self.avg_price = float(order_data['price'])
        
        if 'updateTime' in order_data:
            self.update_time = datetime.fromtimestamp(order_data['updateTime'] / 1000)
        else:
            self.update_time = datetime.now()
            
        if 'commission' in order_data:
            self.commission = float(order_data['commission'])
            
        if 'realizedPnl' in order_data:
            self.realized_pnl = float(order_data['realizedPnl'])
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将订单转换为字典
        
        Returns:
            Dict[str, Any]: 订单字典
        """
        return {
            'client_order_id': self.client_order_id,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.order_side.value,
            'type': self.order_type.value,
            'position_side': self.position_side.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'reduce_only': self.reduce_only,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'avg_price': self.avg_price,
            'commission': self.commission,
            'realized_pnl': self.realized_pnl,
            'create_time': self.create_time.isoformat(),
            'update_time': self.update_time.isoformat()
        }
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"Order({self.symbol}, {self.order_side.value}, {self.order_type.value}, {self.quantity}, {self.price}, {self.status.value})"

class OrderManager:
    """
    订单管理类
    负责创建、跟踪和管理订单
    """
    
    def __init__(self, config_path: str = None, api: Optional[BinanceAPI] = None):
        """
        初始化订单管理器
        
        Args:
            config_path: 配置文件路径
            api: 可选的BinanceAPI实例，如果不提供则创建新实例
        """
        from pathlib import Path
        import os as os_module  # 使用别名避免与全局os混淆
        
        self.logger = logging.getLogger('OrderManager')
        self.orders = {}  # client_order_id -> Order
        self.order_history = []  # 订单历史
        
        # 使用传入的配置路径或默认值
        if config_path is None:
            # 使用当前文件的相对路径找到配置文件
            current_dir = Path(__file__).parent.parent
            config_path = os_module.path.join(current_dir, 'config', 'api_config.yaml')
            
        self._load_config(config_path)
        self.binance_api = api if api is not None else BinanceAPI(config_path)
        
        # 创建订单日志目录
        os_module.makedirs(os_module.path.dirname(self.order_log_path), exist_ok=True)
    
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
            current_dir = Path(__file__).parent.parent
            # 确保使用全局的os模块
            import os as os_module
            self.order_log_path = os_module.path.join(current_dir, "logs", "orders.json")
            self.max_orders = 100  # 最大跟踪订单数量
            
            self.logger.info("成功加载订单管理配置")
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            raise
    
    def create_market_order(self, symbol: str, side: OrderSide, quantity: float, 
                          reduce_only: bool = False, client_order_id: Optional[str] = None) -> Order:
        """
        创建市价单
        
        Args:
            symbol: 交易对
            side: 订单方向
            quantity: 数量
            reduce_only: 是否只减仓
            client_order_id: 客户端订单ID
            
        Returns:
            Order: 创建的订单对象
        """
        order = Order(
            symbol=symbol,
            order_side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            reduce_only=reduce_only,
            client_order_id=client_order_id
        )
        
        self.orders[order.client_order_id] = order
        return order
    
    def create_limit_order(self, symbol: str, side: OrderSide, quantity: float, 
                         price: float, time_in_force: TimeInForce = TimeInForce.GTC,
                         reduce_only: bool = False, client_order_id: Optional[str] = None) -> Order:
        """
        创建限价单
        
        Args:
            symbol: 交易对
            side: 订单方向
            quantity: 数量
            price: 价格
            time_in_force: 有效时间
            reduce_only: 是否只减仓
            client_order_id: 客户端订单ID
            
        Returns:
            Order: 创建的订单对象
        """
        order = Order(
            symbol=symbol,
            order_side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            time_in_force=time_in_force,
            reduce_only=reduce_only,
            client_order_id=client_order_id
        )
        
        self.orders[order.client_order_id] = order
        return order
    
    def create_stop_market_order(self, symbol: str, side: OrderSide, quantity: float, 
                               stop_price: float, reduce_only: bool = False, 
                               client_order_id: Optional[str] = None) -> Order:
        """
        创建止损市价单
        
        Args:
            symbol: 交易对
            side: 订单方向
            quantity: 数量
            stop_price: 触发价格
            reduce_only: 是否只减仓
            client_order_id: 客户端订单ID
            
        Returns:
            Order: 创建的订单对象
        """
        order = Order(
            symbol=symbol,
            order_side=side,
            order_type=OrderType.STOP_MARKET,
            quantity=quantity,
            stop_price=stop_price,
            reduce_only=reduce_only,
            client_order_id=client_order_id
        )
        
        self.orders[order.client_order_id] = order
        return order
    
    def create_take_profit_market_order(self, symbol: str, side: OrderSide, quantity: float, 
                                     stop_price: float, reduce_only: bool = False, 
                                     client_order_id: Optional[str] = None) -> Order:
        """
        创建止盈市价单
        
        Args:
            symbol: 交易对
            side: 订单方向
            quantity: 数量
            stop_price: 触发价格
            reduce_only: 是否只减仓
            client_order_id: 客户端订单ID
            
        Returns:
            Order: 创建的订单对象
        """
        order = Order(
            symbol=symbol,
            order_side=side,
            order_type=OrderType.TAKE_PROFIT_MARKET,
            quantity=quantity,
            stop_price=stop_price,
            reduce_only=reduce_only,
            client_order_id=client_order_id
        )
        
        self.orders[order.client_order_id] = order
        return order
    
    def send_order(self, order: Order) -> Dict[str, Any]:
        """
        发送订单到交易所
        
        Args:
            order: 订单对象
            
        Returns:
            Dict[str, Any]: 交易所返回的订单数据
        """
        try:
            # 准备请求参数
            # 格式化数量和价格以确保符合交易所精度要求
            formatted_quantity = self.binance_api.format_quantity(order.symbol, order.quantity)
            
            params = {
                'symbol': order.symbol,
                'side': order.order_side.value,
                'order_type': order.order_type.value,  # 修复：使用正确的参数名
                'quantity': formatted_quantity,
            }
            
            self.logger.info(f"格式化后的订单数量: {formatted_quantity} (原始: {order.quantity})")
            
            # 只有在有client_order_id的情况下才添加
            if hasattr(order, 'client_order_id') and order.client_order_id:
                params['newClientOrderId'] = order.client_order_id
            
            # 根据订单类型添加特定参数
            if order.order_type in [OrderType.LIMIT, OrderType.STOP, OrderType.TAKE_PROFIT]:
                formatted_price = self.binance_api.format_price(order.symbol, order.price)
                params['price'] = formatted_price
                params['time_in_force'] = order.time_in_force.value if order.time_in_force else TimeInForce.GTC.value
                self.logger.info(f"格式化后的订单价格: {formatted_price} (原始: {order.price})")
            
            if order.order_type in [OrderType.STOP, OrderType.STOP_MARKET, OrderType.TAKE_PROFIT, OrderType.TAKE_PROFIT_MARKET]:
                formatted_stop_price = self.binance_api.format_price(order.symbol, order.stop_price)
                params['stop_price'] = formatted_stop_price
                self.logger.info(f"格式化后的止损/盈价格: {formatted_stop_price} (原始: {order.stop_price})")
            
            if order.reduce_only:
                params['reduce_only'] = True
            
            # 发送订单
            response = self.binance_api.create_order(**params)
            
            # 更新订单信息
            order.update(response)
            
            # 记录订单
            self.log_order(order)
            
            return response
            
        except Exception as e:
            # 更新订单状态为失败
            order.status = OrderStatus.FAILED
            self.log_order(order)
            self.logger.error(f"发送订单失败: {e}")
            raise
    
    def cancel_order(self, client_order_id: str) -> Dict[str, Any]:
        """
        取消订单
        
        Args:
            client_order_id: 客户端订单ID
            
        Returns:
            Dict[str, Any]: 交易所返回的订单数据
        """
        try:
            if client_order_id not in self.orders:
                self.logger.warning(f"尝试取消不存在的订单: {client_order_id}")
                return {}
                
            order = self.orders[client_order_id]
            
            # 发送取消请求
            response = self.binance_api.cancel_order(
                symbol=order.symbol,
                orig_client_order_id=client_order_id
            )
            
            # 更新订单信息
            order.update(response)
            
            # 记录订单
            self.log_order(order)
            
            return response
            
        except Exception as e:
            self.logger.error(f"取消订单失败: {e}")
            raise
    
    def get_order_status(self, client_order_id: str) -> Dict[str, Any]:
        """
        查询订单状态
        
        Args:
            client_order_id: 客户端订单ID
            
        Returns:
            Dict[str, Any]: 订单状态
        """
        try:
            if client_order_id not in self.orders:
                self.logger.warning(f"尝试查询不存在的订单: {client_order_id}")
                return {}
                
            order = self.orders[client_order_id]
            
            # 从交易所获取订单信息
            response = self.binance_api._request(
                method='GET',
                endpoint='/fapi/v1/order',
                signed=True,
                symbol=order.symbol,
                origClientOrderId=client_order_id
            )
            
            # 更新订单信息
            order.update(response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"查询订单状态失败: {e}")
            raise
    
    def update_all_orders(self) -> None:
        """更新所有活跃订单的状态"""
        try:
            # 获取所有开放订单
            open_orders = self.binance_api.get_open_orders()
            
            # 创建映射以快速查找
            open_orders_map = {order['clientOrderId']: order for order in open_orders}
            
            # 更新本地订单状态
            for client_order_id, order in list(self.orders.items()):
                # 如果订单已经是终态，则跳过
                if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                    continue
                    
                if client_order_id in open_orders_map:
                    # 订单仍然活跃，更新状态
                    order.update(open_orders_map[client_order_id])
                else:
                    # 订单不在活跃列表中，可能已经成交或取消，单独查询
                    try:
                        order_data = self.get_order_status(client_order_id)
                        order.update(order_data)
                    except Exception as e:
                        self.logger.warning(f"查询订单{client_order_id}失败: {e}")
                
                # 如果订单是终态，考虑移动到历史记录
                if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                    self.archive_order(order)
                    
        except Exception as e:
            self.logger.error(f"更新所有订单状态失败: {e}")
    
    def archive_order(self, order: Order) -> None:
        """
        将订单归档到历史记录
        
        Args:
            order: 订单对象
        """
        if order.client_order_id in self.orders:
            # 添加到历史记录
            self.order_history.append(order.to_dict())
            
            # 从活跃订单中删除
            del self.orders[order.client_order_id]
            
            # 如果历史记录太长，移除最早的记录
            while len(self.order_history) > self.max_orders:
                self.order_history.pop(0)
    
    def log_order(self, order: Order) -> None:
        """
        记录订单到日志文件
        
        Args:
            order: 订单对象
        """
        try:
            # 读取现有日志
            order_log = []
            import os as os_module
            if os_module.path.exists(self.order_log_path):
                try:
                    with open(self.order_log_path, 'r') as f:
                        order_log = json.load(f)
                except json.JSONDecodeError:
                    order_log = []
            
            # 添加新订单记录
            order_log.append({
                'timestamp': datetime.now().isoformat(),
                'order': order.to_dict()
            })
            
            # 如果日志太长，移除最早的记录
            while len(order_log) > self.max_orders * 2:
                order_log.pop(0)
            
            # 写入日志文件
            with open(self.order_log_path, 'w') as f:
                json.dump(order_log, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"记录订单到日志文件失败: {e}")
    
    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        获取活跃订单
        
        Args:
            symbol: 可选的交易对过滤
            
        Returns:
            List[Order]: 活跃订单列表
        """
        if symbol:
            return [order for order in self.orders.values() if order.symbol == symbol]
        else:
            return list(self.orders.values())
    
    def get_order_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取订单历史
        
        Args:
            symbol: 可选的交易对过滤
            limit: 返回记录数量限制
            
        Returns:
            List[Dict[str, Any]]: 订单历史记录
        """
        if symbol:
            filtered_history = [order for order in self.order_history if order['symbol'] == symbol]
            return filtered_history[-limit:]
        else:
            return self.order_history[-limit:]
    
    def get_order_by_id(self, client_order_id: str) -> Optional[Order]:
        """
        通过客户端订单ID获取订单
        
        Args:
            client_order_id: 客户端订单ID
            
        Returns:
            Optional[Order]: 找到的订单或None
        """
        return self.orders.get(client_order_id)
    
    def create_order_batch(self, orders: List[Order]) -> List[Dict[str, Any]]:
        """
        批量创建订单
        
        Args:
            orders: 订单列表
            
        Returns:
            List[Dict[str, Any]]: 交易所返回的订单数据列表
        """
        results = []
        for order in orders:
            try:
                result = self.send_order(order)
                results.append(result)
            except Exception as e:
                self.logger.error(f"批量创建订单失败: {e}")
                results.append({"error": str(e), "client_order_id": order.client_order_id})
                
        return results
