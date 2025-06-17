"""
币安U本位合约API封装模块
提供与币安USDT保证金合约交易所交互的各种功能
"""
import time
import hmac
import hashlib
import requests
import json
import logging
import yaml
import websocket
import threading
import asyncio
from urllib.parse import urlencode
from typing import Dict, List, Optional, Union, Callable, Any
from concurrent.futures import ThreadPoolExecutor

class BinanceAPIException(Exception):
    """币安API异常类"""
    def __init__(self, response):
        self.code = 0
        try:
            json_res = response.json()
        except ValueError:
            self.message = f'无效的JSON错误响应: {response.text}'
        else:
            self.code = json_res.get('code', 0)
            self.message = json_res.get('msg', '')
        self.status_code = response.status_code
        self.response = response
        self.request = getattr(response, 'request', None)

    def __str__(self):
        return f'APIError(code={self.code}): {self.message}'

class BinanceWebSocketException(Exception):
    """币安WebSocket异常类"""
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return f'WebSocketError: {self.message}'

class BinanceAPI:
    """币安U本位合约API接口类"""
    
    def __init__(self, config_path: str = "../config/api_config.yaml"):
        """
        初始化币安API接口
        
        Args:
            config_path: API配置文件路径
        """
        self.logger = logging.getLogger('BinanceAPI')
        self._load_config(config_path)
        self.session = self._init_session()
        self._ws_connections = {}
        self._ws_callbacks = {}
        self._ws_running = False
        
        # 时间同步相关变量
        self._time_offset = 0  # 本地时间与服务器时间的偏差(毫秒)
        self._time_offset_initialized = False
        
        # 异步处理相关
        self._executor = ThreadPoolExecutor(max_workers=10)
        
    async def initialize(self):
        """
        初始化API连接
        
        验证API连接并确保可以访问所需的功能
        """
        try:
            # 先进行多次时间同步，确保精确
            self.logger.info("开始同步服务器时间...")
            # 多次同步以获取更准确的时间偏移
            for i in range(3):  # 尝试3次同步
                server_time = self.get_server_time()
                await asyncio.sleep(0.5)  # 短暂延迟
            
            # 验证API密钥有效性
            if server_time and 'serverTime' in server_time:
                server_time_str = time.strftime('%Y-%m-%d %H:%M:%S', 
                                   time.localtime(server_time['serverTime']/1000))
                self.logger.info(f"API连接成功，服务器时间：{server_time_str}")
                self.logger.info(f"时间同步完成，偏移量：{self._time_offset}毫秒")
                self.logger.info(f"接收窗口设置为 {self.RECV_WINDOW} 毫秒")
                
                # 获取交易所信息，使用异步方法
                exchange_info = await self._async_request('GET', '/fapi/v1/exchangeInfo')
                if exchange_info:
                    symbols = len(exchange_info.get('symbols', []))
                    self.logger.info(f"获取交易所信息成功，当前可交易合约数：{symbols}")
                return True
            else:
                self.logger.error("API连接验证失败")
                return False
        except Exception as e:
            self.logger.error(f"API初始化错误: {e}")
            raise
    
    def _load_config(self, config_path: str) -> None:
        """
        加载API配置
        
        Args:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            api_config = config.get('api', {})
            self.API_KEY = api_config.get('api_key')
            self.API_SECRET = api_config.get('api_secret')
            self.BASE_URL = api_config.get('base_url')
            self.WS_URL = api_config.get('websocket_url')
            self.TIMEOUT = api_config.get('timeout', 5000)
            self.RECV_WINDOW = api_config.get('recv_window', 5000)
            self.MAX_RETRIES = api_config.get('retry', {}).get('max_retries', 3)
            self.RETRY_DELAY = api_config.get('retry', {}).get('retry_delay', 1000) / 1000  # 转换为秒
            
            self.logger.info("成功加载API配置")
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            raise
    
    def _init_session(self) -> requests.Session:
        """
        初始化HTTP会话
        
        Returns:
            requests.Session: 配置好的会话对象
        """
        session = requests.Session()
        session.headers.update({
            'Content-Type': 'application/json',
            'X-MBX-APIKEY': self.API_KEY
        })
        return session
    
    def _generate_signature(self, query_string: str) -> str:
        """
        生成API请求签名
        
        Args:
            query_string: 请求参数字符串
            
        Returns:
            str: HMAC SHA256签名
        """
        return hmac.new(
            self.API_SECRET.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _handle_response(self, response: requests.Response) -> Dict:
        """
        处理API响应
        
        Args:
            response: 请求响应对象
            
        Returns:
            Dict: 响应数据
            
        Raises:
            BinanceAPIException: API请求错误
        """
        if not str(response.status_code).startswith('2'):
            raise BinanceAPIException(response)
        try:
            return response.json()
        except ValueError:
            raise BinanceAPIException(response)
    
    def _request(self, method: str, endpoint: str, signed: bool = False, 
                **kwargs) -> Dict:
        """
        发送API请求
        
        Args:
            method: HTTP方法 (GET, POST, DELETE, PUT)
            endpoint: API端点
            signed: 是否需要签名
            **kwargs: 请求参数
            
        Returns:
            Dict: API响应数据
            
        Raises:
            BinanceAPIException: API请求错误
        """
        url = f"{self.BASE_URL}{endpoint}"
        
        # 添加时间戳和接收窗口到签名请求
        if signed:
            # 在每次有签名的请求前强制同步时间，确保时间戳准确
            if endpoint != '/fapi/v1/time':  # 避免递归调用
                try:
                    self.get_server_time()
                except Exception as e:
                    self.logger.warning(f"预请求时间同步失败: {e}")
            
            kwargs['timestamp'] = self.get_timestamp()  # 使用调整后的时间戳
            # 使用更大的接收窗口，特别是在WSL环境下
            kwargs['recvWindow'] = max(self.RECV_WINDOW, 20000)  # 至少20秒的接收窗口
        
        # 构建请求参数
        if kwargs:
            query_string = urlencode(kwargs)
            # 对需要签名的请求添加签名
            if signed:
                query_string = f"{query_string}&signature={self._generate_signature(query_string)}"
            url = f"{url}?{query_string}"
        
        # 执行请求，支持重试
        for retry in range(self.MAX_RETRIES + 1):
            try:
                self.logger.debug(f"发送{method}请求到{endpoint}")
                # 增加超时时间，WSL下网络请求可能较慢
                response = self.session.request(
                    method=method,
                    url=url,
                    timeout=self.TIMEOUT / 500  # 转换为秒，并增加两倍超时时间
                )
                return self._handle_response(response)
            except (requests.exceptions.RequestException, BinanceAPIException) as e:
                # 检查是否为时间同步错误
                if isinstance(e, BinanceAPIException) and e.code == -1021:
                    self.logger.warning(f"时间同步错误，正在重新同步时间...")
                    # 强制重新同步时间
                    try:
                        old_offset = self._time_offset
                        server_time_resp = self.session.request(
                            method="GET", 
                            url=f"{self.BASE_URL}/fapi/v1/time",
                            timeout=self.TIMEOUT / 1000
                        ).json()
                        
                        if 'serverTime' in server_time_resp:
                            server_time_ms = server_time_resp['serverTime']
                            local_time_ms = int(time.time() * 1000)
                            self._time_offset = server_time_ms - local_time_ms
                            self._time_offset_initialized = True
                            self.logger.info(f"时间已重新同步：旧偏移={old_offset}ms, 新偏移={self._time_offset}ms")
                            
                            # 如果是签名请求，使用新的时间戳更新URL
                            if signed:
                                kwargs['timestamp'] = self.get_timestamp()
                                query_string = urlencode(kwargs)
                                query_string = f"{query_string}&signature={self._generate_signature(query_string)}"
                                url = f"{self.BASE_URL}{endpoint}?{query_string}"
                    except Exception as sync_error:
                        self.logger.error(f"重新同步时间失败: {sync_error}")
                
                if retry == self.MAX_RETRIES:
                    self.logger.error(f"请求失败后达到最大重试次数: {e}")
                    raise
                self.logger.warning(f"请求失败，尝试重试 ({retry+1}/{self.MAX_RETRIES}): {e}")
                time.sleep(self.RETRY_DELAY)
    
    def get_server_time(self) -> Dict:
        """
        获取服务器时间
        
        Returns:
            Dict: 包含服务器时间的字典
        """
        try:
            # 直接使用请求而不通过其他包装方法，以避免循环调用
            response = self.session.request(
                method="GET", 
                url=f"{self.BASE_URL}/fapi/v1/time",
                timeout=self.TIMEOUT / 1000 * 2  # 双倍超时以确保请求成功
            )
            server_time = response.json()
            
            if server_time and 'serverTime' in server_time:
                # 更新时间偏移
                server_time_ms = server_time['serverTime']
                local_time_ms = int(time.time() * 1000)
                self._time_offset = server_time_ms - local_time_ms
                self._time_offset_initialized = True
                
                # 提高日志级别以便于调试
                if abs(self._time_offset) > 1000:  # 如果偏移超过1秒，使用INFO级别
                    self.logger.info(f"时间同步: 服务器时间={server_time_ms}ms, 本地时间={local_time_ms}ms, 偏移量={self._time_offset}ms")
                else:
                    self.logger.debug(f"时间同步: 服务器时间={server_time_ms}ms, 本地时间={local_time_ms}ms, 偏移量={self._time_offset}ms")
            
            return server_time
        except Exception as e:
            self.logger.error(f"获取服务器时间失败: {e}")
            # 返回一个空字典而不是抛出异常，以便调用者可以继续工作
            return {}
    
    def get_timestamp(self) -> int:
        """
        获取经过时间偏移调整的当前时间戳
        
        Returns:
            int: 调整后的时间戳（毫秒）
        """
        if not self._time_offset_initialized:
            # 如果还没有初始化时间偏移，强制获取一次服务器时间
            try:
                self.get_server_time()
            except Exception as e:
                self.logger.warning(f"获取服务器时间失败，使用本地时间: {e}")
                
        return int(time.time() * 1000) + self._time_offset
        
    async def _async_request(self, method: str, endpoint: str, signed: bool = False, 
                           **kwargs) -> Dict:
        """
        异步发送API请求
        
        Args:
            method: HTTP方法 (GET, POST, DELETE, PUT)
            endpoint: API端点
            signed: 是否需要签名
            **kwargs: 请求参数
            
        Returns:
            Dict: API响应数据
        """
        # 使用线程池执行器运行同步请求
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, 
            lambda: self._request(method, endpoint, signed, **kwargs)
        )
    
    def get_exchange_info(self) -> Dict:
        """
        获取交易所规则和交易对信息
        
        Returns:
            Dict: 交易所交易规则和符号信息
        """
        return self._request('GET', '/fapi/v1/exchangeInfo')
    
    async def get_symbol_info(self, symbol: str) -> Dict:
        """
        获取特定交易对的详细信息
        
        Args:
            symbol: 交易对名称 (e.g. 'BTCUSDT')
            
        Returns:
            Dict: 交易对详细信息，如果未找到则返回None
        """
        try:
            exchange_info = await self._async_request('GET', '/fapi/v1/exchangeInfo')
            if not exchange_info or 'symbols' not in exchange_info:
                self.logger.error("获取交易所信息失败")
                return None
                
            for symbol_info in exchange_info['symbols']:
                if symbol_info.get('symbol') == symbol:
                    return symbol_info
                    
            self.logger.warning(f"交易所中未找到交易对: {symbol}")
            return None
        except Exception as e:
            self.logger.error(f"获取交易对信息时出错: {e}")
            return None
    
    def get_symbol_price_ticker(self, symbol: Optional[str] = None) -> Dict:
        """
        获取交易对最新价格
        
        Args:
            symbol: 交易对名称 (e.g. 'BTCUSDT')
            
        Returns:
            Dict: 价格数据
        """
        params = {'symbol': symbol} if symbol else {}
        return self._request('GET', '/fapi/v1/ticker/price', **params)
    
    def get_klines(self, symbol: str, interval: str, limit: int = 500, 
                  start_time: Optional[int] = None, end_time: Optional[int] = None) -> List:
        """
        获取K线数据
        
        Args:
            symbol: 交易对名称 (e.g. 'BTCUSDT')
            interval: K线间隔 ('1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M')
            limit: 返回的数据量, 默认 500, 最大 1500
            start_time: 开始时间 (毫秒时间戳)
            end_time: 结束时间 (毫秒时间戳)
        
        Returns:
            List: K线数据列表
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
            
        return self._request('GET', '/fapi/v1/klines', **params)
    
    async def get_klines_async(self, symbol: str, interval: str, limit: int = 500, 
                  start_time: Optional[int] = None, end_time: Optional[int] = None) -> List:
        """
        异步获取K线数据
        
        Args:
            symbol: 交易对名称 (e.g. 'BTCUSDT')
            interval: K线间隔 ('1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M')
            limit: 返回的数据量, 默认 500, 最大 1500
            start_time: 开始时间 (毫秒时间戳)
            end_time: 结束时间 (毫秒时间戳)
        
        Returns:
            List: K线数据列表
        """
        import time
        # 添加当前时间戳，确保获取最新数据
        current_time = int(time.time() * 1000)
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit,
            'endTime': current_time  # 使用当前时间作为结束时间，确保获取最新数据
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
            
        return await self._async_request('GET', '/fapi/v1/klines', **params)
    
    async def get_account_info(self) -> Dict:
        """
        获取账户信息(需要签名)
        
        Returns:
            Dict: 账户详情
        """
        return await self._async_request('GET', '/fapi/v2/account', signed=True)
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List:
        """
        获取当前未平仓订单(需要签名)
        
        Args:
            symbol: 可选的交易对过滤
            
        Returns:
            List: 活跃订单列表
        """
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        return self._request('GET', '/fapi/v1/openOrders', signed=True, **params)
    
    def create_order(self, symbol: str, side: str, order_type: str, 
                   quantity: Optional[float] = None,
                   price: Optional[float] = None,
                   time_in_force: Optional[str] = None,
                   reduce_only: Optional[bool] = None,
                   close_position: Optional[bool] = None,
                   stop_price: Optional[float] = None,
                   **kwargs) -> Dict:
        """
        创建新订单(需要签名)
        
        Args:
            symbol: 交易对 (e.g. 'BTCUSDT')
            side: 订单方向 'BUY' 或 'SELL'
            order_type: 订单类型 'LIMIT', 'MARKET', 'STOP', 'STOP_MARKET', 'TAKE_PROFIT', 'TAKE_PROFIT_MARKET'
            quantity: 订单数量
            price: 订单价格 (限价单必需)
            time_in_force: 有效方式 'GTC'=Good Till Cancel, 'IOC'=Immediate or Cancel, 'FOK'=Fill or Kill
            reduce_only: True或False, 仅减仓单
            close_position: True或False, 平仓单 (仅适用于STOP_MARKET或TAKE_PROFIT_MARKET)
            stop_price: 触发价格 (仅适用于STOP, STOP_MARKET, TAKE_PROFIT, TAKE_PROFIT_MARKET)
            **kwargs: 其他可选参数
            
        Returns:
            Dict: 订单响应
        """
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
        }
        
        # 根据订单类型添加必要参数
        if order_type == 'LIMIT':
            params['timeInForce'] = time_in_force or 'GTC'
            params['price'] = price
            params['quantity'] = quantity
        elif order_type == 'MARKET':
            params['quantity'] = quantity
        elif order_type in ['STOP', 'TAKE_PROFIT']:
            params['timeInForce'] = time_in_force or 'GTC'
            params['price'] = price
            params['quantity'] = quantity
            params['stopPrice'] = stop_price
        elif order_type in ['STOP_MARKET', 'TAKE_PROFIT_MARKET']:
            if close_position:
                params['closePosition'] = 'true'
            else:
                params['quantity'] = quantity
            params['stopPrice'] = stop_price
        
        # 添加可选参数
        if reduce_only is not None:
            params['reduceOnly'] = 'true' if reduce_only else 'false'
        
        # 添加其他参数
        params.update(kwargs)
        
        return self._request('POST', '/fapi/v1/order', signed=True, **params)
    
    def cancel_order(self, symbol: str, order_id: Optional[int] = None, 
                   orig_client_order_id: Optional[str] = None) -> Dict:
        """
        取消订单(需要签名)
        
        Args:
            symbol: 交易对名称
            order_id: 订单ID (order_id 和 orig_client_order_id 必须提供一个)
            orig_client_order_id: 原始客户端订单ID
            
        Returns:
            Dict: 取消订单的响应
        """
        params = {'symbol': symbol}
        
        if order_id:
            params['orderId'] = order_id
        elif orig_client_order_id:
            params['origClientOrderId'] = orig_client_order_id
        else:
            raise ValueError("order_id 或 orig_client_order_id 必须提供一个")
            
        return self._request('DELETE', '/fapi/v1/order', signed=True, **params)
    
    def cancel_all_orders(self, symbol: str) -> List:
        """
        取消交易对的所有订单(需要签名)
        
        Args:
            symbol: 交易对名称
            
        Returns:
            List: 取消订单的响应列表
        """
        params = {'symbol': symbol}
        return self._request('DELETE', '/fapi/v1/allOpenOrders', signed=True, **params)
    
    def get_position_risk(self, symbol: Optional[str] = None) -> List:
        """
        获取持仓风险(需要签名)
        
        Args:
            symbol: 可选的交易对过滤
            
        Returns:
            List: 持仓风险信息
        """
        params = {}
        if symbol:
            params['symbol'] = symbol
            
        return self._request('GET', '/fapi/v2/positionRisk', signed=True, **params)
    
    def get_account_info(self) -> Dict:
        """
        获取账户信息(需要签名)
        
        Returns:
            Dict: 账户信息，包括余额、权限、手续费等
        """
        return self._request('GET', '/fapi/v2/account', signed=True)
    
    async def get_account_info_async(self) -> Dict:
        """
        异步获取账户信息(需要签名)
        
        Returns:
            Dict: 账户信息，包括余额、权限、手续费等
        """
        return await self._async_request('GET', '/fapi/v2/account', signed=True)
    
    async def get_position_risk_async(self, symbol: Optional[str] = None) -> List:
        """
        异步获取持仓风险(需要签名)
        
        Args:
            symbol: 可选的交易对过滤
            
        Returns:
            List: 持仓风险信息
        """
        params = {}
        if symbol:
            params['symbol'] = symbol
            
        return await self._async_request('GET', '/fapi/v2/positionRisk', signed=True, **params)
    
    def get_symbol_precision(self, symbol: str) -> Dict[str, int]:
        """
        获取交易对的精度信息
        
        Args:
            symbol: 交易对名称 (e.g. 'BTCUSDT')
            
        Returns:
            Dict: 包含价格和数量精度的字典
                - price_precision: 价格精度
                - quantity_precision: 数量精度
        """
        try:
            exchange_info = self.get_exchange_info()
            
            for symbol_info in exchange_info.get('symbols', []):
                if symbol_info['symbol'] == symbol:
                    # 从 filters 中获取 LOT_SIZE 过滤器
                    quantity_precision = 3  # 默认精度
                    price_precision = 2     # 默认精度
                    
                    for filter_info in symbol_info.get('filters', []):
                        if filter_info['filterType'] == 'LOT_SIZE':
                            # 根据 stepSize 计算精度
                            step_size = float(filter_info['stepSize'])
                            if step_size == 1:
                                quantity_precision = 0
                            elif step_size == 0.1:
                                quantity_precision = 1
                            elif step_size == 0.01:
                                quantity_precision = 2
                            elif step_size == 0.001:
                                quantity_precision = 3
                            elif step_size == 0.0001:
                                quantity_precision = 4
                            else:
                                # 动态计算精度
                                quantity_precision = len(str(step_size).split('.')[-1]) if '.' in str(step_size) else 0
                        
                        elif filter_info['filterType'] == 'PRICE_FILTER':
                            # 根据 tickSize 计算价格精度
                            tick_size = float(filter_info['tickSize'])
                            if tick_size == 1:
                                price_precision = 0
                            elif tick_size == 0.1:
                                price_precision = 1
                            elif tick_size == 0.01:
                                price_precision = 2
                            elif tick_size == 0.001:
                                price_precision = 3
                            else:
                                # 动态计算精度
                                price_precision = len(str(tick_size).split('.')[-1]) if '.' in str(tick_size) else 0
                    
                    return {
                        'price_precision': price_precision,
                        'quantity_precision': quantity_precision
                    }
            
            # 如果没有找到交易对，返回默认精度
            self.logger.warning(f"未找到交易对 {symbol} 的精度信息，使用默认精度")
            return {
                'price_precision': 2,
                'quantity_precision': 3
            }
            
        except Exception as e:
            self.logger.error(f"获取交易对精度失败 {symbol}: {e}")
            return {
                'price_precision': 2,
                'quantity_precision': 3
            }

    def format_quantity(self, symbol: str, quantity: float) -> float:
        """
        根据交易对规则格式化数量
        
        Args:
            symbol: 交易对名称
            quantity: 原始数量
            
        Returns:
            float: 格式化后的数量
        """
        try:
            precision_info = self.get_symbol_precision(symbol)
            quantity_precision = precision_info['quantity_precision']
            
            # 使用 round 函数格式化到指定精度
            formatted_quantity = round(quantity, quantity_precision)
            
            self.logger.debug(f"数量格式化: {symbol} {quantity} -> {formatted_quantity} (精度: {quantity_precision})")
            
            return formatted_quantity
            
        except Exception as e:
            self.logger.error(f"格式化数量失败 {symbol}: {e}")
            return round(quantity, 3)  # 默认保留3位小数
            
    def calculate_quantity_from_usdt(self, symbol: str, usdt_amount: float) -> float:
        """
        根据USDT金额计算可以交易的数量
        
        Args:
            symbol: 交易对名称 (e.g. 'BTCUSDT')
            usdt_amount: USDT金额
            
        Returns:
            float: 可交易的基础资产数量 (如BTC数量)
        """
        try:
            if usdt_amount <= 0:
                self.logger.error(f"无效的USDT金额: {usdt_amount}")
                return 0
                
            # 获取交易对的最新价格（重试最多3次，避免网络波动导致失败）
            retry_count = 3
            current_price = 0
            
            for i in range(retry_count):
                try:
                    ticker = self.get_symbol_price_ticker(symbol)
                    current_price = float(ticker.get('price', 0))
                    
                    if current_price > 0:
                        break
                        
                    self.logger.warning(f"获取{symbol}价格失败，重试 ({i+1}/{retry_count})")
                    time.sleep(0.5)  # 短暂延迟后重试
                except Exception as e:
                    self.logger.warning(f"获取{symbol}价格异常，重试 ({i+1}/{retry_count}): {e}")
                    time.sleep(0.5)
            
            if current_price <= 0:
                self.logger.error(f"无法获取{symbol}的有效价格，交易取消")
                return 0
                
            # 计算可以购买的基础资产数量
            quantity = usdt_amount / current_price
            
            # 考虑市场波动，减少1%的数量，确保有足够的保证金余量
            safety_factor = 0.99
            adjusted_quantity = quantity * safety_factor
            
            # 格式化数量
            formatted_quantity = self.format_quantity(symbol, adjusted_quantity)
            
            base_asset = symbol[:len(symbol)-4] if symbol.endswith('USDT') else symbol.split('USDT')[0]
            self.logger.info(f"USDT金额转换: {usdt_amount} USDT -> {formatted_quantity} {base_asset} (价格: {current_price}, 安全系数: {safety_factor})")
            
            # 确保数量大于0
            if formatted_quantity <= 0:
                self.logger.error(f"计算出的交易数量为0，请增加交易金额")
                return 0
                
            return formatted_quantity
            
        except Exception as e:
            self.logger.error(f"USDT金额转换失败 {symbol}: {e}")
            return 0

    def format_price(self, symbol: str, price: float) -> float:
        """
        根据交易对规则格式化价格
        
        Args:
            symbol: 交易对名称
            price: 原始价格
            
        Returns:
            float: 格式化后的价格
        """
        try:
            precision_info = self.get_symbol_precision(symbol)
            price_precision = precision_info['price_precision']
            
            # 使用 round 函数格式化到指定精度
            formatted_price = round(price, price_precision)
            
            self.logger.debug(f"价格格式化: {symbol} {price} -> {formatted_price} (精度: {price_precision})")
            
            return formatted_price
            
        except Exception as e:
            self.logger.error(f"格式化价格失败 {symbol}: {e}")
            return round(price, 2)  # 默认保留2位小数

    # ============ WebSocket相关方法 ============
    
    def _start_ws_manager(self):
        """启动WebSocket管理线程"""
        if self._ws_running:
            return
            
        self._ws_running = True
        self._ws_thread = threading.Thread(target=self._ws_keeper)
        self._ws_thread.daemon = True
        self._ws_thread.start()
    
    def _ws_keeper(self):
        """WebSocket保持活跃的管理线程"""
        while self._ws_running:
            for stream_name, ws in list(self._ws_connections.items()):
                try:
                    # 心跳检查或其他维护操作
                    if not ws.sock or not ws.sock.connected:
                        self.logger.warning(f"WebSocket连接断开: {stream_name}，尝试重连")
                        # 重新连接
                        callbacks = self._ws_callbacks.get(stream_name, {})
                        self._start_websocket(stream_name, callbacks.get('on_message'),
                                             callbacks.get('on_error'),
                                             callbacks.get('on_close'),
                                             callbacks.get('on_open'))
                except Exception as e:
                    self.logger.error(f"WebSocket检查错误: {e}")
            
            # 休眠一段时间再检查
            time.sleep(30)
    
    def _on_ws_message(self, ws, message, stream_name):
        """
        WebSocket消息处理
        
        Args:
            ws: WebSocket连接
            message: 接收到的消息
            stream_name: 流名称
        """
        callback = self._ws_callbacks.get(stream_name, {}).get('on_message')
        if callback:
            try:
                data = json.loads(message)
                callback(data)
            except json.JSONDecodeError:
                self.logger.error(f"WebSocket无效的JSON: {message}")
            except Exception as e:
                self.logger.error(f"WebSocket消息处理错误: {e}")
    
    def _on_ws_error(self, ws, error, stream_name):
        """
        WebSocket错误处理
        
        Args:
            ws: WebSocket连接
            error: 错误信息
            stream_name: 流名称
        """
        self.logger.error(f"WebSocket错误 [{stream_name}]: {error}")
        callback = self._ws_callbacks.get(stream_name, {}).get('on_error')
        if callback:
            callback(error)
    
    def _on_ws_close(self, ws, close_status_code, close_msg, stream_name):
        """
        WebSocket关闭处理
        
        Args:
            ws: WebSocket连接
            close_status_code: 关闭状态码
            close_msg: 关闭消息
            stream_name: 流名称
        """
        self.logger.info(f"WebSocket连接关闭 [{stream_name}]: {close_status_code} {close_msg}")
        
        # 从连接字典中移除
        if stream_name in self._ws_connections:
            del self._ws_connections[stream_name]
            
        callback = self._ws_callbacks.get(stream_name, {}).get('on_close')
        if callback:
            callback(close_status_code, close_msg)
    
    def _on_ws_open(self, ws, stream_name):
        """
        WebSocket打开处理
        
        Args:
            ws: WebSocket连接
            stream_name: 流名称
        """
        self.logger.info(f"WebSocket连接已打开 [{stream_name}]")
        callback = self._ws_callbacks.get(stream_name, {}).get('on_open')
        if callback:
            callback()
    
    def _start_websocket(self, stream_name, on_message=None, on_error=None, 
                       on_close=None, on_open=None):
        """
        创建并启动WebSocket连接
        
        Args:
            stream_name: 流名称
            on_message: 消息回调
            on_error: 错误回调
            on_close: 关闭回调
            on_open: 打开回调
        """
        # 保存回调
        self._ws_callbacks[stream_name] = {
            'on_message': on_message,
            'on_error': on_error,
            'on_close': on_close,
            'on_open': on_open
        }
        
        ws_url = f"{self.WS_URL}/ws/{stream_name}"
        
        ws = websocket.WebSocketApp(
            ws_url,
            on_message=lambda ws, message: self._on_ws_message(ws, message, stream_name),
            on_error=lambda ws, error: self._on_ws_error(ws, error, stream_name),
            on_close=lambda ws, close_status_code, close_msg: self._on_ws_close(
                ws, close_status_code, close_msg, stream_name
            ),
            on_open=lambda ws: self._on_ws_open(ws, stream_name)
        )
        
        # 保存连接
        self._ws_connections[stream_name] = ws
        
        # 在新线程中启动WebSocket连接
        websocket_thread = threading.Thread(target=ws.run_forever)
        websocket_thread.daemon = True
        websocket_thread.start()
        
        return ws
    
    def subscribe_kline(self, symbol: str, interval: str, callback: Callable[[Dict], None]):
        """
        订阅K线数据
        
        Args:
            symbol: 交易对名称 (小写, 例如 'btcusdt')
            interval: K线间隔 ('1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M')
            callback: 数据回调函数
            
        Returns:
            str: 流名称
        """
        stream_name = f"{symbol.lower()}@kline_{interval}"
        self._start_websocket(stream_name, callback)
        
        # 确保WebSocket管理线程正在运行
        self._start_ws_manager()
        
        return stream_name
    
    def subscribe_ticker(self, symbol: str, callback: Callable[[Dict], None]):
        """
        订阅24小时价格变化统计
        
        Args:
            symbol: 交易对名称 (小写, 例如 'btcusdt')
            callback: 数据回调函数
            
        Returns:
            str: 流名称
        """
        stream_name = f"{symbol.lower()}@ticker"
        self._start_websocket(stream_name, callback)
        
        # 确保WebSocket管理线程正在运行
        self._start_ws_manager()
        
        return stream_name
    
    def subscribe_book_ticker(self, symbol: str, callback: Callable[[Dict], None]):
        """
        订阅最优挂单信息
        
        Args:
            symbol: 交易对名称 (小写, 例如 'btcusdt')
            callback: 数据回调函数
            
        Returns:
            str: 流名称
        """
        stream_name = f"{symbol.lower()}@bookTicker"
        self._start_websocket(stream_name, callback)
        
        # 确保WebSocket管理线程正在运行
        self._start_ws_manager()
        
        return stream_name
    
    def subscribe_mark_price(self, symbol: str, update_speed: str, callback: Callable[[Dict], None]):
        """
        订阅标记价格
        
        Args:
            symbol: 交易对名称 (小写, 例如 'btcusdt')
            update_speed: 更新速度 '1s' 或 '3s'
            callback: 数据回调函数
            
        Returns:
            str: 流名称
        """
        stream_name = f"{symbol.lower()}@markPrice@{update_speed}"
        self._start_websocket(stream_name, callback)
        
        # 确保WebSocket管理线程正在运行
        self._start_ws_manager()
        
        return stream_name
    
    def subscribe_depth(self, symbol: str, level: int, update_speed: str, callback: Callable[[Dict], None]):
        """
        订阅深度信息
        
        Args:
            symbol: 交易对名称 (小写, 例如 'btcusdt')
            level: 深度级别, 5, 10, 或 20
            update_speed: 更新速度 '100ms', '250ms', '500ms', 或 '1000ms'
            callback: 数据回调函数
            
        Returns:
            str: 流名称
        """
        stream_name = f"{symbol.lower()}@depth{level}@{update_speed}"
        self._start_websocket(stream_name, callback)
        
        # 确保WebSocket管理线程正在运行
        self._start_ws_manager()
        
        return stream_name
    
    def subscribe_user_data(self, listen_key: str, on_message=None, on_error=None, on_close=None):
        """
        订阅用户数据流 (需先获取listenKey)
        
        Args:
            listen_key: 监听密钥
            on_message: 消息回调
            on_error: 错误回调
            on_close: 关闭回调
            
        Returns:
            WebSocketApp: WebSocket连接对象
        """
        return self._start_websocket(listen_key, on_message, on_error, on_close)
    
    def get_listen_key(self) -> str:
        """
        获取用户数据流的listenKey
        
        Returns:
            str: listenKey
        """
        response = self._request('POST', '/fapi/v1/listenKey')
        return response['listenKey']
    
    def keep_alive_listen_key(self, listen_key: str) -> Dict:
        """
        延长listenKey有效期
        
        Args:
            listen_key: 监听密钥
            
        Returns:
            Dict: 响应
        """
        params = {'listenKey': listen_key}
        return self._request('PUT', '/fapi/v1/listenKey', **params)
    
    def close_listen_key(self, listen_key: str) -> Dict:
        """
        关闭listenKey
        
        Args:
            listen_key: 监听密钥
            
        Returns:
            Dict: 响应
        """
        params = {'listenKey': listen_key}
        return self._request('DELETE', '/fapi/v1/listenKey', **params)
    
    def close_all_connections(self):
        """关闭所有WebSocket连接"""
        self._ws_running = False
        for stream_name, ws in self._ws_connections.items():
            try:
                ws.close()
                self.logger.info(f"关闭WebSocket连接: {stream_name}")
            except Exception as e:
                self.logger.error(f"关闭WebSocket连接错误 [{stream_name}]: {e}")
        
        self._ws_connections.clear()
        self._ws_callbacks.clear()
        
    def __del__(self):
        """清理资源，关闭线程池"""
        if hasattr(self, '_executor') and self._executor:
            self._executor.shutdown(wait=False)
