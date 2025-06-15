"""
订单管理器模块 - 负责处理和追踪订单状态
"""
import time
import logging
import json
import threading
import uuid
from datetime import datetime
from collections import deque

class OrderManager:
    """订单管理器，负责创建、追踪和管理订单"""
    
    def __init__(self, binance_client=None):
        """
        初始化订单管理器
        
        参数:
        - binance_client: 币安客户端实例
        """
        self.logger = logging.getLogger("OrderManager")
        self.binance_client = binance_client
        
        # 订单存储
        self.active_orders = {}  # 活跃订单 {order_id: order_data}
        self.completed_orders = deque(maxlen=500)  # 已完成订单，限制最大数量
        self.order_updates = {}  # 订单状态更新 {order_id: [updates]}
        
        # 回调函数
        self.on_order_created = None
        self.on_order_filled = None
        self.on_order_canceled = None
        self.on_order_rejected = None
        
        # 状态监控
        self.is_monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 5  # 秒
        
        # 统计数据
        self.stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'canceled_orders': 0,
            'rejected_orders': 0,
            'trades_today': 0
        }
    
    def set_client(self, client):
        """设置币安客户端"""
        self.binance_client = client
    
    def create_order(self, order_data):
        """
        创建订单
        
        参数:
        - order_data: 订单数据字典
        
        返回:
        - 创建的订单ID和状态
        """
        if not self.binance_client:
            self.logger.error("未设置币安客户端，无法创建订单")
            return None
        
        try:
            # 准备订单参数
            symbol = order_data.get('symbol')
            side = order_data.get('side')
            order_type = order_data.get('type', 'MARKET')
            quantity = order_data.get('quantity')
            price = order_data.get('price')
            stop_price = order_data.get('stop_price')
            reduce_only = order_data.get('reduce_only', False)
            time_in_force = order_data.get('time_in_force', 'GTC')
            close_position = order_data.get('close_position', False)
            
            # 创建客户端订单ID
            client_order_id = f"rl_trade_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
            
            # 记录创建请求
            timestamp = datetime.now().isoformat()
            order_data['timestamp'] = timestamp
            order_data['client_order_id'] = client_order_id
            order_data['status'] = 'NEW'
            
            # 发送订单请求
            result = self.binance_client.place_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                reduce_only=reduce_only,
                time_in_force=time_in_force,
                stop_price=stop_price,
                close_position=close_position
            )
            
            if result:
                # 获取订单ID，确保即使result的格式不符合预期，也能继续处理
                order_id = None
                # 尝试从不同的可能字段获取订单ID
                if isinstance(result, dict):
                    order_id = result.get('id') or result.get('orderId') or client_order_id
                
                # 如果没有有效的order_id，使用客户端生成的ID
                if not order_id:
                    order_id = client_order_id
                    self.logger.warning(f"无法从订单结果获取ID，使用客户端ID: {client_order_id}")
                
                # 更新订单数据
                order_data['order_id'] = order_id
                
                # 尝试获取时间戳，如果不可用则使用当前时间
                if isinstance(result, dict):
                    order_data['exchange_timestamp'] = result.get('datetime') or timestamp
                    order_data['status'] = result.get('status', 'NEW')
                else:
                    order_data['exchange_timestamp'] = timestamp
                    order_data['status'] = 'NEW'
                
                # 保存到活跃订单列表
                self.active_orders[order_id] = order_data
                
                # 初始化订单更新列表
                self.order_updates[order_id] = [{
                    'timestamp': timestamp,
                    'status': order_data['status'],
                    'message': '订单已创建'
                }]
                
                # 更新统计数据
                self.stats['total_orders'] += 1
                
                # 更新统计数据
                self.stats['total_orders'] += 1
                
                # 如果订单已经完成，移动到已完成订单列表
                if order_data['status'] in ['FILLED', 'CANCELED', 'REJECTED', 'EXPIRED']:
                    self._handle_completed_order(order_id, order_data)
                
                # 触发回调
                if self.on_order_created:
                    self.on_order_created(order_data)
                
                self.logger.info(f"订单已创建: {symbol} {side} {quantity} @ {price or 'MARKET'} [ID: {order_id}]")
                return order_id, order_data
            else:
                self.logger.error("创建订单失败")
                
                # 更新统计数据
                self.stats['rejected_orders'] += 1
                
                # 触发回调
                if self.on_order_rejected:
                    self.on_order_rejected(order_data, "订单创建失败")
                
                return None, None
        except Exception as e:
            self.logger.error(f"创建订单异常: {e}")
            return None, None
    
    def cancel_order(self, order_id=None, client_order_id=None, symbol=None):
        """
        取消订单
        
        参数:
        - order_id: 订单ID
        - client_order_id: 客户端订单ID
        - symbol: 交易对
        
        返回:
        - 取消结果
        """
        if not self.binance_client:
            self.logger.error("未设置币安客户端，无法取消订单")
            return False
            
        if not order_id and not client_order_id:
            self.logger.error("取消订单需要提供订单ID或客户端订单ID")
            return False
            
        if not symbol and order_id in self.active_orders:
            symbol = self.active_orders[order_id].get('symbol')
        
        try:
            result = self.binance_client.cancel_order(
                symbol=symbol,
                order_id=order_id,
                client_order_id=client_order_id
            )
            
            if result:
                # 获取订单ID和订单数据
                order_id = result.get('id', order_id)
                
                if order_id in self.active_orders:
                    order_data = self.active_orders[order_id]
                    
                    # 更新订单状态
                    order_data['status'] = 'CANCELED'
                    order_data['canceled_at'] = datetime.now().isoformat()
                    
                    # 添加订单更新记录
                    if order_id in self.order_updates:
                        self.order_updates[order_id].append({
                            'timestamp': datetime.now().isoformat(),
                            'status': 'CANCELED',
                            'message': '订单已取消'
                        })
                    
                    # 移动到已完成订单列表
                    self._handle_completed_order(order_id, order_data)
                    
                    # 更新统计数据
                    self.stats['canceled_orders'] += 1
                    
                    # 触发回调
                    if self.on_order_canceled:
                        self.on_order_canceled(order_data)
                    
                    self.logger.info(f"订单已取消: {order_id}")
                    return True
                else:
                    self.logger.warning(f"未找到要取消的订单: {order_id}")
            else:
                self.logger.error(f"取消订单失败: {order_id}")
            
            return False
        except Exception as e:
            self.logger.error(f"取消订单异常: {e}")
            return False
    
    def cancel_all_orders(self, symbol):
        """
        取消所有指定交易对的订单
        
        参数:
        - symbol: 交易对
        
        返回:
        - 取消结果
        """
        if not self.binance_client:
            self.logger.error("未设置币安客户端，无法取消所有订单")
            return False
            
        try:
            result = self.binance_client.cancel_all_orders(symbol)
            
            if result:
                # 查找并更新所有该交易对的活跃订单
                canceled_ids = []
                for order_id, order_data in list(self.active_orders.items()):
                    if order_data.get('symbol') == symbol:
                        # 更新订单状态
                        order_data['status'] = 'CANCELED'
                        order_data['canceled_at'] = datetime.now().isoformat()
                        
                        # 添加订单更新记录
                        if order_id in self.order_updates:
                            self.order_updates[order_id].append({
                                'timestamp': datetime.now().isoformat(),
                                'status': 'CANCELED',
                                'message': '订单已批量取消'
                            })
                        
                        # 移动到已完成订单列表
                        self._handle_completed_order(order_id, order_data)
                        canceled_ids.append(order_id)
                
                # 更新统计数据
                self.stats['canceled_orders'] += len(canceled_ids)
                
                self.logger.info(f"已取消所有{symbol}订单: {len(canceled_ids)}个")
                return True
            else:
                self.logger.error(f"取消所有{symbol}订单失败")
                return False
        except Exception as e:
            self.logger.error(f"取消所有订单异常: {e}")
            return False
    
    def update_order_status(self, order_id=None, client_order_id=None, symbol=None):
        """
        更新订单状态
        
        参数:
        - order_id: 订单ID
        - client_order_id: 客户端订单ID
        - symbol: 交易对
        
        返回:
        - 更新后的订单数据
        """
        if not self.binance_client:
            self.logger.error("未设置币安客户端，无法更新订单状态")
            return None
            
        if not order_id and not client_order_id:
            self.logger.error("更新订单状态需要提供订单ID或客户端订单ID")
            return None
            
        if not symbol and order_id in self.active_orders:
            symbol = self.active_orders[order_id].get('symbol')
            
        try:
            result = self.binance_client.get_order(
                symbol=symbol,
                order_id=order_id,
                client_order_id=client_order_id
            )
            
            if result:
                # 获取订单ID
                order_id = result.get('id', order_id)
                status = result.get('status')
                
                # 更新订单数据
                if order_id in self.active_orders:
                    order_data = self.active_orders[order_id]
                    old_status = order_data.get('status')
                    
                    # 更新订单状态
                    order_data['status'] = status
                    order_data['filled'] = result.get('filled')
                    order_data['remaining'] = result.get('remaining')
                    order_data['cost'] = result.get('cost')
                    order_data['fee'] = result.get('fee')
                    
                    # 添加订单更新记录
                    if order_id in self.order_updates:
                        self.order_updates[order_id].append({
                            'timestamp': datetime.now().isoformat(),
                            'status': status,
                            'message': f'订单状态更新: {old_status} -> {status}'
                        })
                    
                    # 如果订单已经完成，移动到已完成订单列表
                    if status in ['FILLED', 'CANCELED', 'REJECTED', 'EXPIRED']:
                        self._handle_completed_order(order_id, order_data)
                        
                        # 更新统计数据
                        if status == 'FILLED':
                            self.stats['filled_orders'] += 1
                            
                            # 触发回调
                            if self.on_order_filled:
                                self.on_order_filled(order_data)
                        elif status == 'CANCELED':
                            self.stats['canceled_orders'] += 1
                            
                            # 触发回调
                            if self.on_order_canceled:
                                self.on_order_canceled(order_data)
                        elif status == 'REJECTED':
                            self.stats['rejected_orders'] += 1
                            
                            # 触发回调
                            if self.on_order_rejected:
                                self.on_order_rejected(order_data, "订单被拒绝")
                    
                    self.logger.debug(f"订单状态已更新: {order_id} -> {status}")
                    return order_data
                else:
                    # 不在活跃订单列表中，可能是新订单或历史订单
                    self.logger.debug(f"订单不在活跃列表中: {order_id}")
                    
                    # 构建订单数据
                    order_data = {
                        'order_id': order_id,
                        'client_order_id': result.get('clientOrderId'),
                        'symbol': result.get('symbol'),
                        'side': result.get('side'),
                        'type': result.get('type'),
                        'status': status,
                        'price': result.get('price'),
                        'quantity': result.get('amount'),
                        'filled': result.get('filled'),
                        'remaining': result.get('remaining'),
                        'cost': result.get('cost'),
                        'fee': result.get('fee'),
                        'timestamp': result.get('datetime')
                    }
                    
                    # 如果是活跃状态，添加到活跃订单列表
                    if status not in ['FILLED', 'CANCELED', 'REJECTED', 'EXPIRED']:
                        self.active_orders[order_id] = order_data
                        
                        # 初始化订单更新列表
                        self.order_updates[order_id] = [{
                            'timestamp': datetime.now().isoformat(),
                            'status': status,
                            'message': '订单已添加到追踪列表'
                        }]
                    else:
                        # 直接添加到已完成订单列表
                        self.completed_orders.append(order_data)
                    
                    return order_data
            else:
                self.logger.warning(f"获取订单状态失败: {order_id or client_order_id}")
                return None
        except Exception as e:
            self.logger.error(f"更新订单状态异常: {e}")
            return None
    
    def start_monitoring(self, interval=5):
        """
        启动订单状态监控
        
        参数:
        - interval: 监控间隔，单位秒
        """
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitor_interval = interval
        self.logger.info(f"启动订单状态监控，间隔: {interval}秒")
        
        def monitor_loop():
            while self.is_monitoring:
                try:
                    self.update_all_active_orders()
                    time.sleep(interval)
                except Exception as e:
                    self.logger.error(f"订单监控错误: {e}")
                    time.sleep(5)
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止订单状态监控"""
        self.is_monitoring = False
        self.logger.info("订单状态监控已停止")
    
    def update_all_active_orders(self):
        """更新所有活跃订单的状态"""
        if not self.binance_client:
            self.logger.error("未设置币安客户端，无法更新订单状态")
            return
            
        try:
            # 复制活跃订单ID列表，避免在遍历过程中修改
            active_order_ids = list(self.active_orders.keys())
            
            for order_id in active_order_ids:
                # 获取订单数据
                order_data = self.active_orders.get(order_id)
                if order_data:
                    # 更新订单状态
                    self.update_order_status(
                        order_id=order_id,
                        symbol=order_data.get('symbol')
                    )
            
            self.logger.debug(f"已更新{len(active_order_ids)}个活跃订单状态")
            return True
        except Exception as e:
            self.logger.error(f"更新所有活跃订单状态异常: {e}")
            return False
    
    def get_order(self, order_id):
        """
        获取订单数据
        
        参数:
        - order_id: 订单ID
        
        返回:
        - 订单数据或None
        """
        # 先查找活跃订单
        if order_id in self.active_orders:
            return self.active_orders[order_id]
        
        # 再查找已完成订单
        for order in self.completed_orders:
            if order.get('order_id') == order_id:
                return order
        
        return None
    
    def get_active_orders(self, symbol=None):
        """
        获取活跃订单列表
        
        参数:
        - symbol: 交易对(可选)，如果提供则只返回该交易对的订单
        
        返回:
        - 活跃订单列表
        """
        if symbol:
            return [order for order in self.active_orders.values() 
                   if order.get('symbol') == symbol]
        else:
            return list(self.active_orders.values())
    
    def get_completed_orders(self, symbol=None, limit=100):
        """
        获取已完成订单列表
        
        参数:
        - symbol: 交易对(可选)，如果提供则只返回该交易对的订单
        - limit: 返回的最大订单数量
        
        返回:
        - 已完成订单列表
        """
        if symbol:
            filtered_orders = [order for order in self.completed_orders 
                               if order.get('symbol') == symbol]
            return filtered_orders[:limit]
        else:
            return list(self.completed_orders)[:limit]
    
    def get_order_history(self, order_id):
        """
        获取订单历史更新记录
        
        参数:
        - order_id: 订单ID
        
        返回:
        - 订单历史更新记录列表
        """
        return self.order_updates.get(order_id, [])
    
    def get_stats(self):
        """获取订单统计数据"""
        # 计算当天订单数量
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_timestamp = today_start.isoformat()
        
        today_orders = [order for order in self.completed_orders 
                        if order.get('timestamp', '') >= today_timestamp]
        
        self.stats['trades_today'] = len(today_orders)
        
        return self.stats
    
    def start_monitor(self):
        """
        启动订单监控线程，定期检查订单状态
        """
        def monitor_orders():
            while self._monitor_active:
                try:
                    self._check_active_orders()
                    time.sleep(5)  # 每5秒检查一次
                except Exception as e:
                    self.logger.error(f"订单监控发生错误: {e}")
                    time.sleep(10)  # 如果发生错误，等待更长时间
        
        self._monitor_active = True
        self._monitor_thread = threading.Thread(target=monitor_orders)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        self.logger.info("订单监控已启动")
    
    def stop_monitor(self):
        """
        停止订单监控线程
        """
        self._monitor_active = False
        if hasattr(self, '_monitor_thread') and self._monitor_thread.is_alive():
            # 等待线程结束
            self._monitor_thread.join(timeout=5)
        self.logger.info("订单监控已停止")
        
    def _check_active_orders(self):
        """
        检查所有活跃订单的状态
        """
        if not self.active_orders:
            return
        
        for order_id, order_data in list(self.active_orders.items()):
            try:
                # 检查订单状态
                updated_order = self.binance_client.get_order(
                    symbol=order_data['symbol'],
                    order_id=order_id
                )
                
                if not updated_order:
                    continue
                    
                # 添加订单更新记录
                if order_id in self.order_updates:
                    self.order_updates[order_id].append(updated_order)
                else:
                    self.order_updates[order_id] = [updated_order]
                
                # 如果订单已完成，移动到已完成订单列表
                if updated_order['status'] in ['FILLED', 'CANCELED', 'REJECTED', 'EXPIRED']:
                    self._handle_completed_order(order_id, updated_order)
                else:
                    # 更新活跃订单数据
                    self.active_orders[order_id] = {**order_data, **updated_order}
            
            except Exception as e:
                self.logger.error(f"检查订单 {order_id} 状态失败: {e}")
    
    def _handle_completed_order(self, order_id, order_data):
        """
        处理已完成的订单
        
        参数:
        - order_id: 订单ID
        - order_data: 订单数据
        """
        # 从活跃订单中移除
        if order_id in self.active_orders:
            del self.active_orders[order_id]
            
        # 添加到已完成订单列表
        self.completed_orders.append(order_data)
        
        # 如果订单已成交，触发回调
        if order_data['status'] == 'FILLED' and self.on_order_filled:
            try:
                self.on_order_filled(order_data)
            except Exception as e:
                self.logger.error(f"处理订单成交回调时发生错误: {e}")
                
        self.logger.info(f"订单 {order_id} 已完成，状态: {order_data['status']}")