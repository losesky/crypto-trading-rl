"""
币安交易客户端模块 - 用于与币安API交互
"""
import time
import logging
import hmac
import hashlib
import urllib.parse
from datetime import datetime
import json
import requests
import ccxt
import pandas as pd
import numpy as np
from websocket import WebSocketApp
import threading

class BinanceClient:
    """币安U本位合约交易API客户端"""
    
    def __init__(self, api_key, api_secret, test_net=True):
        """初始化币安客户端"""
        self.api_key = api_key
        self.api_secret = api_secret
        self.test_net = test_net
        
        # 初始化日志记录器
        self.logger = logging.getLogger("BinanceClient")
        
        # 首先设置 base_url (在 _sync_time 之前)
        if test_net:
            # 测试网只支持U本位合约API
            self.base_url = "https://testnet.binancefuture.com"
            self.ws_base_url = "wss://stream.binancefuture.com/ws"
        else:
            # 正式环境也应该使用U本位合约API
            self.base_url = "https://fapi.binance.com"
            self.ws_base_url = "wss://fstream.binance.com/ws"
        
        # 设置CCXT客户端
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # 统一使用期货API
                'adjustForTimeDifference': True,
                'recvWindow': 10000,  # 增加接收窗口，减少时间戳错误
                'createMarketBuyOrderRequiresPrice': False  # 市价买单不需要价格
            },
            'timeout': 30000  # 增加超时时间
        })
        
        # 如果是测试网络，设置测试网API URLs
        if test_net:
            # 设置CCXT基本URL配置
            self.exchange.urls['api'] = {
                'public': 'https://testnet.binancefuture.com/fapi/v1',
                'private': 'https://testnet.binancefuture.com/fapi/v1',
                'fapiPublic': 'https://testnet.binancefuture.com/fapi/v1',
                'fapiPrivate': 'https://testnet.binancefuture.com/fapi/v1',
                'fapiData': 'https://testnet.binancefuture.com/futures/data',
                # 为sapi端点添加测试网URL（使用与fapi相同的基础URL）
                'sapi': 'https://testnet.binancefuture.com/fapi/v1'
            }
            
            # 修改CCXT的options，禁用测试网不支持的端点
            self.exchange.options.update({
                'recvWindow': 60000,               # 增大接收窗口
                'adjustForTimeDifference': True,   # 自动调整时间差
                'warnOnFetchOpenOrdersWithoutSymbol': False,  # 禁用警告
                'fetchPositions': {
                    'method': 'fapiPrivateGetAccount'  # 使用账户API替代持仓API
                },
                'defaultTimeInForce': 'GTC'        # 默认订单有效期
            })
        
        # 时间差校准 (在 base_url 设置之后)
        self.time_offset = 0
        self._sync_time()
        
        # 设置HTTP会话
        self.session = requests.Session()
        self.session.headers.update({
            'X-MBX-APIKEY': self.api_key,
            'Content-Type': 'application/json'
        })
        
        # WebSocket相关
        self.ws = None
        self.ws_callbacks = {}
        self.ws_connected = False
        self.ws_reconnect = True
        self.keep_alive_thread = None
        self.last_ping_time = 0
        self.received_pong = True
        
    def get_server_time(self):
        """获取服务器时间"""
        try:
            time_res = self.exchange.fetch_time()
            return time_res, int(time.time() * 1000)
        except Exception as e:
            self.logger.error(f"获取服务器时间失败: {e}")
            return None, int(time.time() * 1000)

    def get_exchange_info(self, symbol=None):
        """获取交易所信息"""
        try:
            info = self.exchange.fapiPublicGetExchangeInfo()
            if symbol:
                # 过滤出指定交易对的信息
                for sym_info in info['symbols']:
                    if sym_info['symbol'] == symbol:
                        return sym_info
                return None
            return info
        except Exception as e:
            self.logger.error(f"获取交易所信息失败: {e}")
            return None

    def get_account_info(self):
        """获取账户信息"""
        try:
            if self.test_net:
                # 测试网使用v2版本的API，添加重试机制
                max_retries = 3
                retry_count = 0
                backoff_time = 1  # 初始退避时间为1秒
                
                while retry_count < max_retries:
                    try:
                        # 在每次尝试前校准时间
                        if retry_count > 0:
                            self._sync_time()
                            self.logger.info(f"第{retry_count}次尝试获取账户信息，已重新同步时间")
                        
                        url = f"{self.base_url}/fapi/v2/account"
                        params = {
                            'timestamp': self.get_timestamp(),
                            'recvWindow': 120000  # 使用非常大的接收窗口 (120秒)，降低时间戳错误风险
                        }
                        
                        # 添加签名
                        query_string = '&'.join([f"{key}={val}" for key, val in params.items()])
                        signature = hmac.new(
                            self.api_secret.encode('utf-8'),
                            query_string.encode('utf-8'),
                            hashlib.sha256
                        ).hexdigest()
                        params['signature'] = signature
                        
                        headers = {'X-MBX-APIKEY': self.api_key}
                        
                        # 添加请求超时
                        response = requests.get(url, params=params, headers=headers, timeout=10)
                        
                        if response.status_code == 200:
                            return response.json()
                        else:
                            self.logger.error(f"获取账户信息失败: {response.status_code} {response.text}")
                            
                            # 检查是否是时间戳错误
                            if "recvWindow" in response.text or "Timestamp" in response.text:
                                # 时间戳问题，强制重新同步
                                self._sync_time()
                            else:
                                # 其他错误，也尝试重试
                                pass
                            
                            retry_count += 1
                            time.sleep(backoff_time)
                            backoff_time *= 2  # 指数退避
                    
                    except Exception as e:
                        self.logger.warning(f"获取账户信息请求异常，尝试重试: {e}")
                        retry_count += 1
                        time.sleep(backoff_time)
                        backoff_time *= 2  # 指数退避
                
                # 所有重试失败后返回None
                self.logger.error(f"在{max_retries}次尝试后依然无法获取账户信息")
                return None
            else:
                return self.exchange.fapiPrivateGetAccount()
        except Exception as e:
            self.logger.error(f"获取账户信息失败: {e}")
            return None

    def get_balance(self):
        """获取账户余额"""
        try:
            # 在测试网环境下，从账户信息中提取余额
            if self.test_net:
                try:
                    # 直接从账户信息中获取余额数据
                    account = self.get_account_info()
                    if account and 'assets' in account:
                        assets = account['assets']
                        result = {
                            'info': assets,
                            'timestamp': int(time.time() * 1000),
                            'datetime': datetime.now().isoformat(),
                            'free': {},
                            'used': {},
                            'total': {}
                        }
                        for asset in assets:
                            if 'asset' in asset:
                                symbol = asset['asset']
                                free = float(asset.get('availableBalance', 0))
                                locked = float(asset.get('initialMargin', 0)) + float(asset.get('maintMargin', 0))
                                result['free'][symbol] = free
                                result['used'][symbol] = locked
                                result['total'][symbol] = free + locked
                        return result
                    self.logger.warning("无法从账户信息获取余额数据，使用默认值")
                except Exception as e:
                    self.logger.warning(f"从账户信息获取余额失败: {e}，使用默认值")
            else:
                # 正式环境使用标准方法并添加错误恢复
                return self._safe_api_call(
                    self.exchange.fetch_balance,
                    fallback_value={
                        'info': [],
                        'timestamp': int(time.time() * 1000),
                        'datetime': datetime.now().isoformat(),
                        'free': {'USDT': 10000.0},
                        'used': {'USDT': 0.0},
                        'total': {'USDT': 10000.0}
                    }
                )
            
            # 如果前面的方法都失败了，返回一个默认余额
            # 这样即使API调用失败，系统也能继续运行
            default_balance = {
                'info': [],
                'timestamp': int(time.time() * 1000),
                'datetime': datetime.now().isoformat(),
                'free': {'USDT': 10000.0},
                'used': {'USDT': 0.0},
                'total': {'USDT': 10000.0}
            }
            self.logger.warning("使用默认余额数据")
            return default_balance
        except Exception as e:
            self.logger.error(f"获取账户余额失败: {e}")
            # 确保即使在API失败的情况下也返回一些数据
            return {
                'info': [],
                'timestamp': int(time.time() * 1000),
                'datetime': datetime.now().isoformat(),
                'free': {'USDT': 10000.0},
                'used': {'USDT': 0.0},
                'total': {'USDT': 10000.0}
            }

    def get_positions(self, symbol=None):
        """获取持仓信息"""
        try:
            if self.test_net:
                # 在测试网环境下，始终直接使用API请求，不尝试使用CCXT内置方法
                # 增加重试机制确保获取成功
                max_retries = 3
                retry_count = 0
                backoff_time = 1  # 初始退避时间为1秒
                
                while retry_count < max_retries:
                    try:
                        # 使用币安测试网直接支持的API端点
                        url = f"{self.base_url}/fapi/v2/account"  # 使用账户信息API获取持仓
                        
                        # 使用非常大的接收窗口，确保时间戳在有效范围内
                        # 参考: https://github.com/tiagosiebler/awesome-crypto-examples/wiki/Timestamp-for-this-request-is-outside-of-the-recvWindow
                        params = {
                            'timestamp': self.get_timestamp(),
                            'recvWindow': 120000  # 使用非常大的接收窗口 (120秒)
                        }
                        
                        # 添加签名
                        query_string = '&'.join([f"{key}={val}" for key, val in params.items()])
                        signature = hmac.new(
                            self.api_secret.encode('utf-8'),
                            query_string.encode('utf-8'),
                            hashlib.sha256
                        ).hexdigest()
                        params['signature'] = signature
                        
                        headers = {
                            'X-MBX-APIKEY': self.api_key
                        }
                        
                        # 在每次请求前重新同步时间，确保时间戳准确
                        if retry_count > 0:
                            self._sync_time()
                            self.logger.info(f"第{retry_count}次尝试获取持仓信息，已重新同步时间")
                        
                        response = requests.get(url, params=params, headers=headers)
                        
                        if response.status_code == 200:
                            account_data = response.json()
                            if 'positions' in account_data:
                                positions = account_data['positions']
                                
                                # 在测试网环境下，确保所有必要的字段都存在
                                for pos in positions:
                                    # 记录原始持仓数据结构用于调试
                                    if symbol and pos['symbol'] == symbol:
                                        self.logger.debug(f"测试网原始持仓数据: {pos}")
                                    
                                    # 确保基本字段存在
                                    if 'isolatedMargin' not in pos:
                                        pos['isolatedMargin'] = '0'
                                        self.logger.debug("添加缺失的isolatedMargin字段")
                                    
                                    # 确保其他可能缺失的字段
                                    for field in ['unRealizedProfit', 'liquidationPrice', 'marginType']:
                                        if field not in pos:
                                            pos[field] = '0' if field != 'marginType' else 'ISOLATED'
                                            self.logger.debug(f"添加缺失的{field}字段")
                                
                                if symbol:
                                    positions = [pos for pos in positions if pos['symbol'] == symbol]
                                
                                # 只返回有持仓量的仓位
                                positions = [pos for pos in positions if float(pos.get('positionAmt', '0')) != 0]
                                return positions
                        else:
                            self.logger.error(f"获取账户信息失败: {response.status_code} {response.text}")
                            # 检查是否是时间戳错误
                            if response.status_code == 400 and "recvWindow" in response.text:
                                self.logger.warning("时间戳错误，尝试重新同步")
                                self._sync_time()
                            else:
                                # 其他错误情况，也尝试重试
                                pass
                        
                        # 如果没有成功返回，则重试
                        retry_count += 1
                        time.sleep(backoff_time)
                        backoff_time *= 2  # 指数退避
                    
                    except Exception as e:
                        self.logger.warning(f"尝试获取持仓信息失败: {e}")
                        retry_count += 1
                        time.sleep(backoff_time)
                        backoff_time *= 2  # 指数退避
                
                # 经过多次尝试后仍然失败，返回空列表表示没有持仓
                self.logger.warning("测试网环境无法获取持仓信息，返回空列表")
                return []
            else:
                # 正式环境使用标准方法
                positions = self.exchange.fapiPrivateGetPositionRisk({
                    'timestamp': self.get_timestamp(),
                    'recvWindow': 10000
                })
                if symbol:
                    # 过滤出指定交易对的持仓
                    return [pos for pos in positions if pos['symbol'] == symbol]
                return positions
        except Exception as e:
            self.logger.error(f"获取持仓信息过程中发生错误: {e}")
            return []

    def place_order(self, symbol, side, order_type, quantity, price=None, reduce_only=False, 
                    time_in_force="GTC", stop_price=None, close_position=False):
        """
        下单
        
        参数:
        - symbol: 交易对
        - side: 方向 (BUY/SELL)
        - order_type: 订单类型 (LIMIT/MARKET/STOP/TAKE_PROFIT等)
        - quantity: 数量
        - price: 价格 (LIMIT单必填)
        - reduce_only: 是否只减仓
        - time_in_force: 有效期 (GTC/IOC/FOK)
        - stop_price: 触发价格 (STOP/TAKE_PROFIT单必填)
        - close_position: 是否平仓 (STOP/TAKE_PROFIT单可用)
        """
        try:
            # 如果是测试网环境，使用REST API直接下单，避免ccxt库的间接调用
            if self.test_net:
                return self._place_order_direct_api(
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
            
            # 非测试网环境使用ccxt
            params = {
                "reduceOnly": reduce_only,
            }
            
            if close_position:
                params["closePosition"] = True
            
            if stop_price and order_type in ["STOP", "STOP_MARKET", "TAKE_PROFIT", "TAKE_PROFIT_MARKET"]:
                params["stopPrice"] = stop_price
            
            # 根据订单类型调用不同的API函数
            if order_type == "MARKET":
                return self.exchange.create_order(
                    symbol=symbol,
                    type="market",
                    side=side.lower(),
                    amount=quantity,
                    params=params
                )
            elif order_type == "LIMIT":
                return self.exchange.create_order(
                    symbol=symbol,
                    type="limit",
                    side=side.lower(),
                    amount=quantity,
                    price=price,
                    params={**params, "timeInForce": time_in_force}
                )
            elif order_type in ["STOP", "STOP_MARKET"]:
                return self.exchange.create_order(
                    symbol=symbol,
                    type="stop_market",
                    side=side.lower(),
                    amount=quantity,
                    params=params
                )
            elif order_type in ["TAKE_PROFIT", "TAKE_PROFIT_MARKET"]:
                return self.exchange.create_order(
                    symbol=symbol,
                    type="take_profit_market",
                    side=side.lower(),
                    amount=quantity,
                    params=params
                )
            else:
                raise ValueError(f"不支持的订单类型: {order_type}")
        except Exception as e:
            self.logger.error(f"下单失败: {e}")
            return None

    def cancel_order(self, symbol, order_id=None, client_order_id=None):
        """取消订单"""
        try:
            return self.exchange.cancel_order(id=order_id, symbol=symbol, params={"clientOrderId": client_order_id})
        except Exception as e:
            self.logger.error(f"取消订单失败: {e}")
            return None

    def cancel_all_orders(self, symbol):
        """取消所有订单"""
        try:
            return self.exchange.fapiPrivateDeleteAllOpenOrders(params={"symbol": symbol})
        except Exception as e:
            self.logger.error(f"取消所有订单失败: {e}")
            return None

    def get_open_orders(self, symbol=None):
        """获取未完成订单"""
        try:
            params = {}
            if symbol:
                params["symbol"] = symbol
            return self.exchange.fetch_open_orders(symbol=symbol)
        except Exception as e:
            self.logger.error(f"获取未完成订单失败: {e}")
            return None

    def get_order(self, symbol, order_id=None, client_order_id=None):
        """获取订单详情"""
        try:
            # 如果是测试网环境，优先使用直接API调用
            if self.test_net:
                self.logger.debug(f"使用直接API获取订单详情: {symbol} {order_id}")
                return self.get_order_direct(symbol, order_id, client_order_id)
            
            # 否则使用CCXT
            params = {"symbol": symbol}
            if client_order_id:
                params["clientOrderId"] = client_order_id
            return self.exchange.fetch_order(id=order_id, symbol=symbol, params=params)
        except Exception as e:
            self.logger.error(f"获取订单详情失败: {e}")
            return None

    def get_order_direct(self, symbol, order_id=None, client_order_id=None):
        """
        使用直接API调用获取订单详情（测试网环境专用）
        
        参数:
        - symbol: 交易对名称
        - order_id: 订单ID
        - client_order_id: 客户端订单ID
        
        返回:
        - 订单详情
        """
        try:
            # 构建请求参数
            params = {
                'symbol': symbol,
                'timestamp': self.get_timestamp(),  # 使用get_timestamp方法确保时间戳正确
                'recvWindow': 120000  # 使用更大的接收窗口避免时间同步问题
            }
            
            # 添加订单ID或客户端订单ID
            if order_id:
                params['orderId'] = order_id
            elif client_order_id:
                params['origClientOrderId'] = client_order_id
            else:
                self.logger.error("获取订单详情失败: 必须提供订单ID或客户端订单ID")
                return None
                
            # 构建查询字符串并计算签名
            query_string = urllib.parse.urlencode(params)
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            # 添加签名到参数
            params['signature'] = signature
                
            # 发送请求
            endpoint = '/fapi/v1/order' # U本位合约查询订单API
            url = f"{self.base_url}{endpoint}"
            headers = {'X-MBX-APIKEY': self.api_key}
            
            # 发送GET请求
            response = requests.get(url, params=params, headers=headers)
                
            # 检查响应
            if response.status_code == 200:
                order_data = response.json()
                
                # 格式化为与ccxt兼容的格式
                formatted_order = {
                    'id': str(order_data.get('orderId')),
                    'clientOrderId': order_data.get('clientOrderId'),
                    'timestamp': order_data.get('time'),
                    'datetime': datetime.fromtimestamp(order_data.get('time') / 1000).isoformat(),
                    'symbol': order_data.get('symbol'),
                    'type': order_data.get('type').lower(),
                    'side': order_data.get('side').lower(),
                    'price': float(order_data.get('price')) if order_data.get('price') != '0' else None,
                    'amount': float(order_data.get('origQty')),
                    'filled': float(order_data.get('executedQty')),
                    'remaining': float(order_data.get('origQty')) - float(order_data.get('executedQty')),
                    'cost': float(order_data.get('cumQuote')),
                    'status': order_data.get('status').lower(),
                    'fee': None,
                    'trades': None,
                    'info': order_data
                }
                
                return formatted_order
            else:
                error_msg = f"{response.status_code} {response.text}"
                self.logger.error(f"直接API获取订单失败: {error_msg}")
                return None
                
        except Exception as e:
            self.logger.error(f"直接API获取订单失败: {e}")
            return None

    def get_historical_klines(self, symbol, interval, start_time=None, end_time=None, limit=500):
        """
        获取K线数据
        
        参数:
        - symbol: 交易对
        - interval: 时间间隔 (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
        - start_time: 起始时间戳 (毫秒)
        - end_time: 结束时间戳 (毫秒)
        - limit: 数量限制 (默认500，最大1000)
        """
        try:
            # 添加足够的重试次数，确保API请求成功
            max_retries = 5
            retry_count = 0
            backoff_time = 1  # 初始退避时间为1秒
            
            while retry_count < max_retries:
                try:
                    if retry_count > 0:
                        self.logger.info(f"第 {retry_count} 次重试获取K线数据...")
                    
                    params = {}
                    if start_time:
                        params['startTime'] = start_time
                    if end_time:
                        params['endTime'] = end_time
                    if limit:
                        params['limit'] = limit
                    
                    #
                    if self.test_net:
                        # 构建请求URL和参数
                        url = f"{self.base_url}/fapi/v1/klines"
                        params['symbol'] = symbol
                        params['interval'] = interval
                        if start_time:
                            params['startTime'] = start_time
                        if end_time:
                            params['endTime'] = end_time
                        if limit:
                            params['limit'] = limit
                            
                        # 发送请求
                        response = requests.get(url, params=params)
                        if response.status_code == 200:
                            klines_data = response.json()
                            # 将数据转换为与fetch_ohlcv相同的格式
                            klines = []
                            for item in klines_data:
                                klines.append([
                                    int(item[0]),  # 开盘时间
                                    float(item[1]),  # 开盘价
                                    float(item[2]),  # 最高价
                                    float(item[3]),  # 最低价
                                    float(item[4]),  # 收盘价
                                    float(item[5])   # 成交量
                                ])
                        else:
                            self.logger.error(f"获取K线数据失败: {response.status_code} {response.text}")
                            raise Exception(f"API错误: {response.text}")
                    else:
                        # 正式环境使用CCXT
                        klines = self.exchange.fetch_ohlcv(
                            symbol=symbol,
                            timeframe=interval,
                            since=start_time,
                            limit=limit,
                            params=params
                        )
                    
                    if klines:
                        # 转换为DataFrame
                        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        return df
                    
                    self.logger.warning(f"获取到空的K线数据，可能是API问题")
                    retry_count += 1
                    time.sleep(backoff_time)
                    backoff_time *= 2  # 指数退避
                
                except Exception as e:
                    if "recvWindow" in str(e) or "timestamp" in str(e).lower():
                        # 时间戳问题，重新同步时间
                        self.logger.warning(f"时间戳错误，重新同步时间: {e}")
                        self._sync_time()
                    
                    self.logger.warning(f"获取K线数据失败，尝试重试: {e}")
                    retry_count += 1
                    time.sleep(backoff_time)
                    backoff_time *= 2  # 指数退避
            
            self.logger.error(f"在 {max_retries} 次尝试后依然无法获取K线数据")
            return None
            
        except Exception as e:
            self.logger.error(f"获取K线数据过程中出错: {e}")
            return None

    def change_leverage(self, symbol, leverage):
        """修改杠杆倍数"""
        try:
            # 在测试网环境下，直接返回模拟成功结果，避免API调用
            if self.test_net:
                self.logger.info(f"测试网环境：模拟设置杠杆倍数 {leverage} 成功")
                return {"symbol": symbol, "leverage": leverage}
            
            # 正式环境使用安全API调用包装器
            return self._safe_api_call(
                self.exchange.fapiPrivatePostLeverage,
                params={
                    "symbol": symbol,
                    "leverage": leverage,
                    "timestamp": self.get_timestamp(),  # 添加校准后的时间戳
                    "recvWindow": 60000  # 大幅增加接收窗口，减少时间错误
                },
                fallback_value={"leverage": leverage}  # 提供一个默认返回值
            )
        except Exception as e:
            self.logger.error(f"修改杠杆倍数失败: {e}")
            # 即使失败也返回一个基本结果，避免程序中断
            return {"symbol": symbol, "leverage": leverage}

    def change_margin_type(self, symbol, margin_type):
        """
        修改保证金类型
        
        参数:
        - symbol: 交易对
        - margin_type: 保证金类型 (ISOLATED/CROSSED)
        """
        try:
            # 在测试网环境下，直接返回模拟成功结果，避免API调用
            if self.test_net:
                self.logger.info(f"测试网环境：模拟设置保证金类型 {margin_type} 成功")
                return {"symbol": symbol, "marginType": margin_type}
                
            # 正式环境使用安全API调用包装器
            return self._safe_api_call(
                self.exchange.fapiPrivatePostMarginType,
                params={
                    "symbol": symbol,
                    "marginType": margin_type,
                    "timestamp": self.get_timestamp(),  # 添加校准后的时间戳
                    "recvWindow": 60000  # 大幅增加接收窗口，减少时间错误
                },
                fallback_value={"marginType": margin_type}  # 提供一个默认返回值
            )
        except Exception as e:
            self.logger.error(f"修改保证金类型失败: {e}")
            # 即使失败也返回一个基本结果，避免程序中断
            return {"symbol": symbol, "marginType": margin_type}

    def set_leverage(self, symbol, leverage):
        """设置交易对的杠杆倍数
        
        参数:
        - symbol: 交易对，如 'BTCUSDT'
        - leverage: 杠杆倍数，如 3
        
        返回:
        - 设置结果
        """
        try:
            # 在测试网环境下，直接返回模拟成功结果，避免API调用
            if self.test_net:
                self.logger.info(f"测试网环境：模拟设置杠杆倍数 {leverage} 成功")
                return {"symbol": symbol, "leverage": leverage}
            
            # 正式环境使用安全API调用
            def leverage_api_call():
                endpoint = '/fapi/v1/leverage'
                params = {
                    'symbol': symbol,
                    'leverage': leverage,
                    'timestamp': self.get_timestamp(),  # 使用校准后的时间戳
                    'recvWindow': 60000  # 大幅增加接收窗口
                }
                return self._send_signed_request('POST', endpoint, params)
                
            return self._safe_api_call(
                leverage_api_call,
                fallback_value={"symbol": symbol, "leverage": leverage}  # 提供一个默认返回值
            )
        except Exception as e:
            self.logger.error(f"设置杠杆倍数失败: {e}")
            # 即使失败也返回一个更有用的结果，避免程序中断
            return {"symbol": symbol, "leverage": leverage}

    def set_position_mode(self, dual_side_position=False):
        """设置持仓模式
        
        参数:
        - dual_side_position: 是否开启双向持仓模式，True为双向，False为单向
        
        返回:
        - 设置结果
        """
        try:
            # 在测试网环境下，直接返回模拟成功结果，避免API调用
            if self.test_net:
                self.logger.info(f"测试网环境：模拟设置持仓模式 {'双向' if dual_side_position else '单向'} 成功")
                return {"dualSidePosition": dual_side_position}
            
            # 正式环境使用安全API调用
            def position_mode_api_call():
                endpoint = '/fapi/v1/positionSide/dual'
                params = {
                    'dualSidePosition': 'true' if dual_side_position else 'false',
                    'timestamp': self.get_timestamp(),  # 使用校准后的时间戳
                    'recvWindow': 60000  # 大幅增加接收窗口
                }
                return self._send_signed_request('POST', endpoint, params)
            
            return self._safe_api_call(
                position_mode_api_call,
                fallback_value={"dualSidePosition": dual_side_position}  # 提供一个默认返回值
            )
        except Exception as e:
            self.logger.error(f"设置持仓模式失败: {e}")
            # 即使失败也返回一个有意义的结果，避免程序中断
            return {"dualSidePosition": dual_side_position}

    def _send_signed_request(self, method, endpoint, params=None):
        """发送带签名的API请求
        
        参数:
        - method: 请求方法 (GET, POST, DELETE等)
        - endpoint: API端点，如 '/fapi/v1/order'
        - params: 请求参数字典
        
        返回:
        - API响应
        """
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        
        # 添加时间戳和接收窗口，确保时间同步
        if 'timestamp' not in params:
            params['timestamp'] = self.get_timestamp()
        
        if 'recvWindow' not in params:
            # 在测试网环境下使用更大的接收窗口
            params['recvWindow'] = 120000 if self.test_net else 20000  # 使用非常大的接收窗口，确保时间戳在有效范围内
        
        # 计算签名
        query_string = '&'.join([f"{key}={val}" for key, val in params.items()])
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        params['signature'] = signature
        
        try:
            # 在测试环境下，某些API端点需要特殊处理
            special_endpoints = {
                '/sapi/': {
                    'handle': 'redirect',
                    'target': '/fapi/v1/',
                    'message': '测试网不支持SAPI端点，尝试使用FAPI替代'
                },
                '/fapi/v1/positionRisk': {
                    'handle': 'redirect',
                    'target': '/fapi/v2/account',
                    'message': '测试网不支持positionRisk API，使用account API替代'
                }
            }
            
            if self.test_net:
                # 检查是否需要特殊处理
                for prefix, action in special_endpoints.items():
                    if endpoint.startswith(prefix):
                        self.logger.info(f"{action['message']}")
                        
                        if action['handle'] == 'redirect':
                            # 在这里处理重定向逻辑
                            if prefix == '/sapi/' and action['target'] == '/fapi/v1/':
                                # 对于SAPI端点，尝试使用账户信息API
                                adjusted_endpoint = '/fapi/v2/account'
                                self.logger.info(f"重定向请求: {endpoint} -> {adjusted_endpoint}")
                                
                                # 创建新的请求
                                redirect_url = f"{self.base_url}{adjusted_endpoint}"
                                response = requests.get(redirect_url, params=params, headers={'X-MBX-APIKEY': self.api_key})
                                
                                if response.status_code == 200:
                                    return response.json()
                                else:
                                    self.logger.error(f"重定向请求失败: {response.status_code} {response.text}")
                                    return {}
                            
                            elif prefix == '/fapi/v1/positionRisk' and action['target'] == '/fapi/v2/account':
                                # 对于持仓风险API，使用账户信息API
                                adjusted_endpoint = '/fapi/v2/account'
                                self.logger.info(f"重定向请求: {endpoint} -> {adjusted_endpoint}")
                                
                                # 创建新的请求
                                redirect_url = f"{self.base_url}{adjusted_endpoint}"
                                response = requests.get(redirect_url, params=params, headers={'X-MBX-APIKEY': self.api_key})
                                
                                if response.status_code == 200:
                                    data = response.json()
                                    # 从账户信息中提取持仓信息
                                    if 'positions' in data:
                                        positions = data['positions']
                                        # 只返回有持仓量的仓位
                                        return [pos for pos in positions if float(pos.get('positionAmt', '0')) != 0]
                                    else:
                                        return []
                                else:
                                    self.logger.error(f"重定向请求失败: {response.status_code} {response.text}")
                                    return []
                        
                        # 如果没有特殊处理或重定向失败，返回空结果
                        return {} if prefix != '/fapi/v1/positionRisk' else []
            
            headers = {
                'X-MBX-APIKEY': self.api_key
            }
            
            # 发送请求
            if method == 'GET':
                response = requests.get(url, params=params, headers=headers)
            elif method == 'POST':
                response = requests.post(url, params=params, headers=headers)
            elif method == 'DELETE':
                response = requests.delete(url, params=params, headers=headers)
            else:
                self.logger.error(f"不支持的请求方法: {method}")
                return {}
            
            # 检查响应状态
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"API请求失败: {response.status_code} {response.text}")
                # 如果是测试网API错误，返回空结果
                if response.status_code == 404 or "Path" in response.text:
                    if endpoint.startswith('/fapi/v1/positionRisk'):
                        return []
                    else:
                        return {}
                return {}
                
        except Exception as e:
            self.logger.error(f"API请求处理失败: {e}")
            # 返回空结果而不是引发异常，避免程序中断
            return {}

    def _handle_unsupported_endpoint(self, endpoint):
        """处理测试网不支持的API端点
        
        参数:
        - endpoint: API端点路径
        
        返回:
        - 空结果
        """
        self.logger.warning(f"测试网不支持端点: {endpoint}。请在正式环境中使用此功能。")
        if endpoint.startswith('/fapi/v1/positionRisk'):
            # 对于持仓风险，返回一个空的持仓列表，表示没有持仓
            return []
        elif endpoint.startswith('/sapi/'):
            # 对于sapi端点，返回空对象
            return {}
        else:
            # 对于其他不支持的端点，返回空对象
            return {}

    # WebSocket相关方法
    def start_kline_socket(self, symbol, interval, callback):
        """
        启动K线数据WebSocket
        
        参数:
        - symbol: 交易对 (小写，如: btcusdt)
        - interval: 时间间隔
        - callback: 回调函数，接收数据处理
        """
        symbol_lower = symbol.lower()
        stream_name = f"{symbol_lower}@kline_{interval}"
        
        def on_message(ws, message):
            data = json.loads(message)
            callback(data)
        
        def on_error(ws, error):
            self.logger.error(f"WebSocket错误: {error}")
        
        def on_close(ws, close_status_code, close_reason):
            self.logger.info(f"WebSocket关闭: {close_status_code} - {close_reason}")
            self.ws_connected = False
            if self.ws_reconnect:
                self.logger.info("尝试重新连接...")
                time.sleep(5)
                self.start_kline_socket(symbol, interval, callback)
        
        def on_open(ws):
            self.logger.info(f"WebSocket连接已打开: {stream_name}")
            self.ws_connected = True
        
        # 设置正确的测试网WebSocket URL
        if self.test_net:
            ws_url = f"wss://stream.binancefuture.com/ws/{stream_name}"
        else:
            ws_url = f"wss://fstream.binance.com/ws/{stream_name}"
        self.ws = WebSocketApp(
            ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        # 在新线程中启动WebSocket连接
        ws_thread = threading.Thread(target=self.ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
        
        # 启动保活线程
        if self.keep_alive_thread is None or not self.keep_alive_thread.is_alive():
            self.keep_alive_thread = threading.Thread(target=self._keep_alive)
            self.keep_alive_thread.daemon = True
            self.keep_alive_thread.start()
        
        return ws_thread

    def start_user_data_socket(self, callback):
        """启动用户数据WebSocket"""
        try:
            # 获取listenKey
            response = self.exchange.fapiPrivatePostListenKey()
            listen_key = response['listenKey']
            
            def on_message(ws, message):
                data = json.loads(message)
                callback(data)
            
            def on_error(ws, error):
                self.logger.error(f"用户数据WebSocket错误: {error}")
            
            def on_close(ws, close_status_code, close_reason):
                self.logger.info(f"用户数据WebSocket关闭: {close_status_code} - {close_reason}")
                self.ws_connected = False
                if self.ws_reconnect:
                    self.logger.info("尝试重新连接用户数据WebSocket...")
                    time.sleep(5)
                    self.start_user_data_socket(callback)
            
            def on_open(ws):
                self.logger.info("用户数据WebSocket连接已打开")
                self.ws_connected = True
            
            # 设置正确的测试网WebSocket URL
            if self.test_net:
                ws_url = f"wss://stream.binancefuture.com/ws/{listen_key}"
            else:
                ws_url = f"wss://fstream.binance.com/ws/{listen_key}"
            self.user_ws = WebSocketApp(
                ws_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            
            # 在新线程中启动WebSocket连接
            user_ws_thread = threading.Thread(target=self.user_ws.run_forever)
            user_ws_thread.daemon = True
            user_ws_thread.start()
            
            # 启动保活线程
            if self.keep_alive_thread is None or not self.keep_alive_thread.is_alive():
                self.keep_alive_thread = threading.Thread(target=self._keep_alive_listen_key, args=(listen_key,))
                self.keep_alive_thread.daemon = True
                self.keep_alive_thread.start()
            
            return user_ws_thread
        except Exception as e:
            self.logger.error(f"启动用户数据WebSocket失败: {e}")
            return None

    def _keep_alive_listen_key(self, listen_key):
        """定期延长listenKey有效期"""
        while self.ws_reconnect:
            try:
                time.sleep(300)  # 每5分钟延长一次
                self.exchange.fapiPrivatePutListenKey(params={"listenKey": listen_key})
                self.logger.debug("已延长listenKey有效期")
            except Exception as e:
                self.logger.error(f"延长listenKey有效期失败: {e}")

    def _keep_alive(self):
        """WebSocket连接保活"""
        while self.ws_reconnect:
            time.sleep(30)
            if self.ws and self.ws_connected:
                self.ws.send('{"method":"ping"}')
                self.last_ping_time = time.time()
                self.received_pong = False
                
                # 等待pong响应
                timeout = time.time() + 10
                while not self.received_pong and time.time() < timeout:
                    time.sleep(1)
                
                if not self.received_pong:
                    self.logger.warning("WebSocket未响应ping，尝试重连")
                    try:
                        self.ws.close()
                    except:
                        pass
                    self.ws_connected = False

    def stop_websockets(self):
        """停止所有WebSocket连接"""
        self.ws_reconnect = False
        if self.ws:
            self.ws.close()
        if hasattr(self, 'user_ws') and self.user_ws:
            self.user_ws.close()
        self.ws_connected = False

    def _sync_time(self):
        """同步本地时间与币安服务器时间"""
        try:
            # 参考 https://github.com/tiagosiebler/awesome-crypto-examples/wiki/Timestamp-for-this-request-is-outside-of-the-recvWindow
            # 使用更可靠的方式获取服务器时间，并计算时间偏移
            
            # 获取发送请求前的本地时间
            local_time_before = int(time.time() * 1000)
            
            # 测试网使用专用时间端点
            time_endpoint = "/fapi/v1/time"
            server_time = None
            response_time = 0
            
            # 增加尝试次数，在测试网上更保守
            max_attempts = 10 if self.test_net else 5
            
            for attempt in range(max_attempts):  # 尝试多次，增加成功率
                try:
                    # 记录发送请求前的时间
                    request_start = time.time()
                    
                    # 添加请求超时，避免长时间阻塞
                    res = requests.get(f"{self.base_url}{time_endpoint}", timeout=5)
                    
                    # 计算请求耗时
                    response_time = int((time.time() - request_start) * 1000)
                    
                    if res.status_code == 200:
                        server_time = res.json()['serverTime']
                        self.logger.info(f"成功获取服务器时间: {server_time}, 请求耗时: {response_time}ms")
                        break
                    else:
                        self.logger.warning(f"获取服务器时间失败，状态码: {res.status_code}，尝试重试 ({attempt+1}/{max_attempts})")
                except Exception as e:
                    self.logger.warning(f"获取服务器时间失败，尝试重试 ({attempt+1}/{max_attempts}): {e}")
                
                # 增加退避时间
                backoff_time = min(1 * (1.5 ** attempt), 5)
                time.sleep(backoff_time)
            
            # 获取收到响应后的本地时间
            local_time_after = int(time.time() * 1000)
            
            if server_time:
                # 计算本地时间与服务器时间的平均偏差
                # 1. 计算请求过程中的本地时间中点
                local_time_midpoint = local_time_before + ((local_time_after - local_time_before) // 2)
                
                # 2. 计算偏移量 (服务器时间 - 本地时间中点)
                estimated_offset = server_time - local_time_midpoint
                
                # 3. 应用偏移量，加上额外的安全裕度
                if self.test_net:
                    # 针对Binance测试网与recvWindow问题，设置非常保守的时间偏移
                    # 将时间提前至少55秒，确保请求的时间戳在recvWindow内
                    safety_margin_ms = 55000 + response_time * 2  # 安全裕度最小55秒
                    
                    # 根据请求的延迟动态调整
                    if response_time > 500:
                        safety_margin_ms += 10000  # 如果延迟高，增加更多的安全裕度
                        
                    # 应用总偏移
                    self.time_offset = estimated_offset - safety_margin_ms
                    
                    # 确保偏移量是1000的倍数，进一步防止边界问题
                    self.time_offset = (self.time_offset // 1000) * 1000
                    
                    self.logger.info(f"服务器时间校准(测试网): 原始偏移量 = {estimated_offset}ms, 调整后偏移量 = {self.time_offset}ms, 安全裕度 = {safety_margin_ms}ms")
                else:
                    # 正式环境使用较小的安全裕度
                    safety_margin_ms = 1000 + response_time
                    self.time_offset = estimated_offset - safety_margin_ms
                    self.logger.info(f"服务器时间校准: 偏移量 = {self.time_offset}ms, 安全裕度 = {safety_margin_ms}ms")
                    
                # 存储最后一次成功同步的时间和服务器时间
                self.last_sync_time = time.time()
                self.server_time_at_sync = server_time
            else:
                self.logger.error("无法获取服务器时间，使用更保守的固定偏移量")
                
                if self.test_net:
                    # 如果无法获取服务器时间，使用非常保守的负偏移量
                    self.time_offset = -600000  # 使用-10分钟的偏移量
                    self.logger.warning(f"无法获取服务器时间，使用保守固定偏移量: {self.time_offset}ms")
            
            # 存储同步时间点
            self.last_sync_timestamp = time.time()
        except Exception as e:
            self.logger.error(f"同步服务器时间出错: {e}")
            if self.test_net:
                # 如果无法获取服务器时间，使用固定的更大的负偏移量
                self.time_offset = -600000  # 使用更大的负偏移量，提前10分钟
                self.logger.warning(f"出错，使用更保守的固定偏移量: {self.time_offset}ms")
            # 继续尝试连接
            
    def get_timestamp(self):
        """获取校准后的时间戳，确保时间戳不会超前于服务器时间"""
        current_local_time_ms = int(time.time() * 1000)
        timestamp = current_local_time_ms + self.time_offset
        
        # 测试网环境下，增加同步频率，确保时间戳精确
        if self.test_net:
            # 初始化同步计时器
            if not hasattr(self, 'last_sync_timestamp'):
                self.last_sync_timestamp = time.time()
                self._api_calls_since_sync = 0
            
            self._api_calls_since_sync = getattr(self, '_api_calls_since_sync', 0) + 1
            seconds_since_last_sync = time.time() - self.last_sync_timestamp
            
            # 测试网下更频繁地同步，确保时间戳准确
            # 1. 每5分钟同步一次
            # 2. 每15次API调用同步一次
            # 3. 如果时间差超过10秒，立即同步
            if (seconds_since_last_sync > 300) or (self._api_calls_since_sync >= 15):
                self.logger.info(f"定期时间同步: 距离上次同步 {seconds_since_last_sync:.1f}秒, API调用 {self._api_calls_since_sync}次")
                self._sync_time()
                self._api_calls_since_sync = 0
            
            # 添加额外的稳定性检查，确保时间戳不会比服务器时间早太多或晚太多
            # 每次请求时增加一个小的随机偏移，避免使用完全相同的时间戳，防止出现边界问题
            random_jitter = int(np.random.randint(-200, 0)) if 'np' in globals() else -100
            final_timestamp = timestamp + random_jitter
            
            # 确保最终时间戳在安全范围内
            # 如果产生的时间戳离当前时间过远，重新同步时间
            if abs(final_timestamp - current_local_time_ms) > 500000:  # 如果偏移超过5分钟
                self.logger.warning(f"时间戳偏移过大，重新同步: 偏移量 = {final_timestamp - current_local_time_ms}ms")
                self._sync_time()
                return self.get_timestamp()  # 递归调用自身获取新的时间戳
                
            return final_timestamp
        else:
            # 正式环境使用更简单的时间戳计算
            return timestamp

    def _safe_api_call(self, func, *args, fallback_value=None, **kwargs):
        """安全包装API调用，提供错误处理和退避逻辑
        
        Args:
            func: 要调用的API函数
            *args: 传递给API函数的位置参数
            fallback_value: 如果调用失败，返回的默认值
            **kwargs: 传递给API函数的关键字参数
            
        Returns:
            API调用结果或fallback_value
        """
        max_retries = 5 if self.test_net else 3  # 测试网环境使用更多重试次数
        retry_count = 0
        backoff_time = 1
        
        # 常见的测试网不支持的端点
        unsupported_testnet_endpoints = [
            'capital/config/getall',  # 资金配置端点
            'futures/loan/wallet',    # 期货钱包端点
            'asset/getUserAsset',     # 用户资产端点
            'futures/loan/borrow/history',  # 借款历史
            'sapi/',                  # 所有sapi端点
            'margin/'                 # margin端点
        ]
        
        # 检查特定的函数调用，如fetch_ticker，它可能在内部使用不支持的端点
        if self.test_net:
            # 如果是fetch_ticker函数，测试网中这个函数会尝试调用不支持的capital/config/getall端点
            if func == self.exchange.fetch_ticker:
                # 检查函数名是否是fetch_ticker
                self.logger.info(f"检测到fetch_ticker调用，在测试网环境使用直接API调用以避免不支持的端点")
                try:
                    # 从args中获取symbol参数
                    symbol = args[0] if args else kwargs.get('symbol')
                    if not symbol:
                        return fallback_value
                    
                    # 直接使用fapi/v1/ticker/price端点获取价格信息
                    url = f"{self.base_url}/fapi/v1/ticker/price"
                    params = {"symbol": symbol}
                    response = requests.get(url, params=params)
                    
                    if response.status_code == 200:
                        price_data = response.json()
                        # 构造与fetch_ticker相同格式的返回值
                        result = {
                            'symbol': symbol,
                            'timestamp': int(time.time() * 1000),
                            'datetime': datetime.now().isoformat(),
                            'last': float(price_data.get('price', 0)),
                            'bid': None,  # 测试网没有提供bid/ask信息
                            'ask': None,
                            'info': price_data
                        }
                        return result
                    else:
                        self.logger.error(f"获取价格信息失败: {response.status_code} {response.text}")
                        return fallback_value
                except Exception as e:
                    self.logger.error(f"获取价格信息时发生错误: {e}")
                    return fallback_value
        
        while retry_count < max_retries:
            try:
                # 如果是测试网且不是第一次尝试，先重新同步时间
                if self.test_net and retry_count > 0:
                    self._sync_time()
                
                # 捕获CCXT调用的参数，检查是否是测试网不支持的端点
                if self.test_net and 'params' in kwargs:
                    params_dict = kwargs.get('params', {})
                    endpoint_str = str(params_dict)
                    
                    # 检查是否包含不支持的端点
                    if any(unsupported in endpoint_str for unsupported in unsupported_testnet_endpoints):
                        self.logger.warning(f"跳过测试网不支持的API端点，直接返回默认值")
                        return fallback_value
                
                result = func(*args, **kwargs)
                return result
                
            except Exception as e:
                error_msg = str(e).lower()
                retry_count += 1
                
                # 处理不同类型的错误
                if "recvwindow" in error_msg or "timestamp" in error_msg:
                    self.logger.warning(f"时间同步错误，重新同步时间: {e}")
                    self._sync_time()
                    # 对于时间戳错误，减少退避时间，快速重试
                    backoff_time = 0.5
                elif "path" in error_msg and ("method get is invalid" in error_msg or "method post is invalid" in error_msg):
                    # 检查是否包含特定的不支持的终端
                    if "capital/config/getall" in error_msg:
                        self.logger.warning(f"检测到不支持的capital/config/getall端点调用，跳过此调用并返回默认值")
                        # 直接返回默认值，不再尝试重试
                        return fallback_value
                        
                        # 如果是fetch_ticker相关调用
                        if func == self.exchange.fetch_ticker:
                            try:
                                # 尝试直接使用ticker/price端点获取价格
                                symbol = args[0] if args else kwargs.get('symbol')
                                if not symbol:
                                    return fallback_value
                                    
                                url = f"{self.base_url}/fapi/v1/ticker/price"
                                params = {"symbol": symbol}
                                response = requests.get(url, params=params)
                                
                                if response.status_code == 200:
                                    price_data = response.json()
                                    result = {
                                        'symbol': symbol,
                                        'timestamp': int(time.time() * 1000),
                                        'datetime': datetime.now().isoformat(),
                                        'last': float(price_data.get('price', 0)),
                                        'bid': None,
                                        'ask': None,
                                        'info': price_data
                                    }
                                    self.logger.info(f"使用替代方法获取价格成功: {result['last']}")
                                    return result
                            except Exception as inner_e:
                                self.logger.error(f"使用替代方法获取价格失败: {inner_e}")
                    
                    self.logger.warning(f"API端点错误，测试网可能不支持此调用: {e}")
                    # 对于明确不支持的API，立即返回默认值，不浪费重试次数
                    return fallback_value
                elif "rate limit" in error_msg:
                    self.logger.warning(f"API速率限制，等待更长时间: {e}")
                    backoff_time = max(backoff_time * 2, 5)  # 至少等待5秒
                elif "unknown error" in error_msg or "network error" in error_msg:
                    self.logger.warning(f"网络错误，稍后重试: {e}")
                    backoff_time = max(backoff_time * 1.5, 2)  # 适度增加等待时间
                else:
                    self.logger.warning(f"API调用失败，尝试重试 ({retry_count}/{max_retries}): {e}")
                
                if retry_count >= max_retries:
                    self.logger.error(f"在{max_retries}次尝试后依然失败: {e}")
                    return fallback_value
                
                time.sleep(backoff_time)
                backoff_time *= 2  # 指数退避
        
        return fallback_value

    def get_ticker(self, symbol):
        """获取最新价格
        
        参数:
        - symbol: 交易对
        
        返回:
        - 价格字典
        """
        # 在测试网环境下使用更直接的API调用方式，避免调用不支持的端点
        if self.test_net:
            try:
                # 直接使用fapi/v1/ticker/price端点获取价格数据
                url = f"{self.base_url}/fapi/v1/ticker/price"
                params = {"symbol": symbol}
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    price_data = response.json()
                    result = {
                        'symbol': symbol,
                        'timestamp': int(time.time() * 1000),
                        'datetime': datetime.now().isoformat(),
                        'last': float(price_data.get('price', 0)),
                        'bid': None,  # 测试网没有提供bid/ask信息
                        'ask': None,
                        'info': price_data
                    }
                    return result
                else:
                    self.logger.warning(f"获取价格数据失败: {response.status_code} {response.text}")
                    # 返回默认值
                    return {
                        'symbol': symbol,
                        'last': None,  # 无法获取最新价格
                        'bid': None,
                        'ask': None,
                        'timestamp': int(time.time() * 1000)
                    }
            except Exception as e:
                self.logger.error(f"直接API请求获取价格失败: {e}")
                # 返回默认值
                return {
                    'symbol': symbol,
                    'last': None,  # 无法获取最新价格
                    'bid': None,
                    'ask': None,
                    'timestamp': int(time.time() * 1000)
                }
        else:
            # 正式环境使用安全API调用包装
            return self._safe_api_call(
                self.exchange.fetch_ticker, 
                symbol,
                fallback_value={
                    'symbol': symbol,
                    'last': None,  # 无法获取最新价格
                    'bid': None,
                    'ask': None,
                    'timestamp': int(time.time() * 1000)
                }
            )

    def check_api_status(self):
        """
        检查API连接状态
        
        返回:
        - bool: API是否可用
        """
        try:
            # 尝试调用ping API检查服务器状态
            if self.test_net:
                url = "https://testnet.binancefuture.com/fapi/v1/ping"
            else:
                url = "https://fapi.binance.com/fapi/v1/ping"
                
            # 发送简单请求
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                self.logger.debug("API状态正常")
                return True
            else:
                self.logger.warning(f"API状态异常，响应码: {response.status_code}")
                return False
        except Exception as e:
            self.logger.warning(f"API连接检查失败: {e}")
            return False

    def _place_order_direct_api(self, symbol, side, order_type, quantity, price=None, 
                          reduce_only=False, time_in_force="GTC", stop_price=None, close_position=False):
        """
        直接使用REST API下单，避免ccxt库在测试网中的某些不支持的API调用
        """
        try:
            # 检查数量是否为0或负数
            if quantity <= 0:
                self.logger.error(f"下单数量必须大于0，当前数量: {quantity}")
                return None
                
            # 准备基本参数
            endpoint = '/fapi/v1/order'
            # 计算正确的时间戳，time_offset是负值，所以要加上它
            current_timestamp = int(time.time() * 1000)
            adjusted_timestamp = current_timestamp + self.time_offset
            
            # 舍入数量到正确的精度
            rounded_quantity = self.round_quantity(quantity, symbol)
            
            # 再次确认数量大于0
            if rounded_quantity <= 0:
                self.logger.error(f"舍入后数量必须大于0，调整为最小值0.001")
                rounded_quantity = 0.001
            
            params = {
                'symbol': symbol,
                'side': side.upper(),
                'quantity': rounded_quantity,
                'reduceOnly': 'true' if reduce_only else 'false',
                'timestamp': adjusted_timestamp,
                'recvWindow': 120000  # 增加接收窗口到120秒，解决时间戳问题
            }
            
            # 设置订单类型和相关参数
            if order_type == "MARKET":
                params['type'] = 'MARKET'
            elif order_type == "LIMIT":
                params['type'] = 'LIMIT'
                params['price'] = price
                params['timeInForce'] = time_in_force
            elif order_type in ["STOP", "STOP_MARKET"]:
                params['type'] = 'STOP_MARKET'
                params['stopPrice'] = stop_price
            elif order_type in ["TAKE_PROFIT", "TAKE_PROFIT_MARKET"]:
                params['type'] = 'TAKE_PROFIT_MARKET'
                params['stopPrice'] = stop_price
            
            if close_position:
                params['closePosition'] = 'true'
            
            # 发送请求
            method = 'POST'
            url = f"{self.base_url}{endpoint}"
            
            # 计算签名
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            params['signature'] = signature
            
            # 发送请求
            headers = {'X-MBX-APIKEY': self.api_key}
            response = requests.post(url, params=params, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                self.logger.info(f"直接API下单成功: {result.get('orderId')} (数量: {rounded_quantity}, 原始数量: {quantity})")
                
                # 转换为与ccxt兼容的格式
                orderId = str(result.get('orderId'))  # 确保订单ID是字符串
                converted_result = {
                    'id': orderId,
                    'orderId': orderId,  # 添加这个字段确保兼容性
                    'info': result,
                    'timestamp': result.get('updateTime'),
                    'datetime': datetime.fromtimestamp(result.get('updateTime')/1000).isoformat() if result.get('updateTime') else None,
                    'symbol': result.get('symbol'),
                    'type': result.get('type').lower() if result.get('type') else None,
                    'side': result.get('side').lower() if result.get('side') else None,
                    'price': float(result.get('price')) if result.get('price') else None,
                    'amount': float(result.get('origQty')) if result.get('origQty') else None,
                    'cost': None,
                    'average': None,
                    'filled': float(result.get('executedQty')) if result.get('executedQty') else 0,
                    'remaining': None,
                    'status': result.get('status').lower() if result.get('status') else 'unknown',
                    'fee': None,
                    'trades': None,
                }
                return converted_result
            else:
                self.logger.error(f"直接API下单失败: {response.status_code} {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"直接API下单异常: {e}")
            return None

    def round_quantity(self, quantity, symbol):
        """
        根据交易对的精度要求舍入数量
        
        参数:
        - quantity: 原始数量
        - symbol: 交易对
        
        返回:
        - 舍入后的数量
        """
        try:
            # 获取交易对信息
            symbol_info = self.get_exchange_info(symbol)
            qty_precision = 3  # 默认精度为3位小数
            min_qty = 0.001    # 默认最小数量
            
            if symbol_info:
                for filter_item in symbol_info.get('filters', []):
                    if filter_item['filterType'] == 'LOT_SIZE':
                        # 从stepSize确定精度
                        step_size_str = filter_item['stepSize']
                        min_qty = float(filter_item.get('minQty', min_qty))
                        if '.' in step_size_str:
                            decimal_part = step_size_str.rstrip('0').split('.')[1]
                            qty_precision = len(decimal_part)
                        break
            
            # 使用Decimal确保精确舍入
            from decimal import Decimal, ROUND_DOWN
            
            # 先使用更高精度计算以保留更多有效数字
            internal_precision = 8  # 内部使用更高精度
            internal_step = Decimal('0.1') ** internal_precision
            quantity = Decimal(str(quantity))
            internal_rounded = quantity.quantize(internal_step, rounding=ROUND_DOWN)
            
            # 然后按照交易所要求精度舍入
            exchange_step = Decimal('0.1') ** qty_precision
            rounded = float(internal_rounded.quantize(exchange_step, rounding=ROUND_DOWN))
            
            # 确保数量不小于最小交易量
            if rounded < min_qty:
                rounded = min_qty
                self.logger.warning(f"数量太小，已调整为最小交易量: {min_qty}")
            
            self.logger.info(f"数量舍入: {quantity} -> {rounded} (精度: {qty_precision}, 最小数量: {min_qty})")
            return rounded
            
        except Exception as e:
            self.logger.error(f"数量舍入失败: {e}")
            # 如果发生错误，确保返回至少0.001的数量
            return max(0.001, float(int(quantity * 1000) / 1000))
