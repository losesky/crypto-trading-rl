#!/usr/bin/env python3
"""
增强版Binance客户端测试脚本 - 验证对测试网API错误的修复

此脚本测试所有API调用的稳定性，特别关注可能在测试网环境下失败的方法
"""

import os
import sys
import json
import time
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入币安客户端
from trading_system.src.binance_client import BinanceClient

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('binance_client_enhanced_test.log')
    ]
)

logger = logging.getLogger("BinanceEnhancedTest")

def load_config():
    """加载测试配置"""
    try:
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              'config', 'test_config.json'), 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"加载配置失败: {e}")
        return None

def test_time_sync(client):
    """测试时间同步"""
    logger.info("测试时间同步")
    try:
        server_time, local_time = client.get_server_time()
        time_diff = abs(server_time - local_time)
        logger.info(f"服务器时间: {server_time}, 本地时间: {local_time}, 差值: {time_diff}ms")
        return time_diff < 10000  # 差值小于10秒视为成功
    except Exception as e:
        logger.error(f"时间同步测试失败: {e}")
        return False

def test_get_account_info(client):
    """测试获取账户信息"""
    logger.info("测试获取账户信息")
    try:
        account_info = client.get_account_info()
        if account_info and isinstance(account_info, dict) and 'assets' in account_info:
            logger.info(f"获取账户信息成功, 资产数量: {len(account_info['assets'])}")
            return True
        logger.error(f"账户信息格式异常: {account_info}")
        return False
    except Exception as e:
        logger.error(f"获取账户信息失败: {e}")
        return False

def test_get_balance(client):
    """测试获取余额"""
    logger.info("测试获取余额")
    try:
        balance = client.get_balance()
        if balance and 'total' in balance:
            logger.info(f"获取余额成功, 可用货币数量: {len(balance['free'])}")
            return True
        logger.error(f"余额信息格式异常: {balance}")
        return False
    except Exception as e:
        logger.error(f"获取余额失败: {e}")
        return False

def test_get_positions(client):
    """测试获取持仓信息"""
    logger.info("测试获取持仓信息 - 连续调用5次确认稳定性")
    success_count = 0
    
    for i in range(5):
        try:
            logger.info(f"第 {i+1} 次调用获取持仓信息")
            positions = client.get_positions()
            if isinstance(positions, list):
                logger.info(f"获取持仓信息成功, 持仓数量: {len(positions)}")
                success_count += 1
            else:
                logger.error(f"持仓信息格式异常: {positions}")
            
            # 短暂等待避免API调用过于频繁
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"获取持仓信息失败: {e}")
    
    return success_count == 5  # 5次调用全部成功才返回True

def test_get_klines(client):
    """测试获取K线数据"""
    logger.info("测试获取K线数据")
    try:
        symbol = "BTCUSDT"
        interval = "1h"
        limit = 10
        
        klines = client.get_historical_klines(symbol, interval, limit=limit)
        
        if klines is not None and len(klines) > 0:
            logger.info(f"获取K线数据成功, 数据条数: {len(klines)}")
            return True
            
        logger.error(f"K线数据为空或格式异常: {klines}")
        return False
    except Exception as e:
        logger.error(f"获取K线数据失败: {e}")
        return False

def test_get_ticker(client):
    """测试获取最新价格"""
    logger.info("测试获取最新价格")
    try:
        symbol = "BTCUSDT"
        ticker = client.get_ticker(symbol)
        
        if ticker and 'symbol' in ticker:
            last_price = ticker.get('last')
            logger.info(f"获取{symbol}最新价格成功: {last_price}")
            return True
            
        logger.error(f"价格信息格式异常: {ticker}")
        return False
    except Exception as e:
        logger.error(f"获取最新价格失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("开始增强版Binance客户端测试")
    
    # 加载配置
    config = load_config()
    if not config:
        logger.error("无法加载配置，测试终止")
        return
    
    # 从配置中获取API密钥
    binance_config = config.get("binance", {})
    api_key = binance_config.get("api_key")
    api_secret = binance_config.get("api_secret")
    test_net = binance_config.get("test_net", True)
    
    if not api_key or not api_secret:
        logger.error("API密钥或密钥未配置，测试终止")
        return
    
    # 创建币安客户端实例
    logger.info(f"创建币安客户端 (测试网: {test_net})")
    client = BinanceClient(api_key, api_secret, test_net=test_net)
    
    # 运行各项测试
    tests = {
        "time_sync": test_time_sync(client),
        "account_info": test_get_account_info(client),
        "balance": test_get_balance(client),
        "positions": test_get_positions(client),
        "ticker": test_get_ticker(client),
        "klines": test_get_klines(client)
    }
    
    # 输出测试结果摘要
    logger.info("=" * 50)
    logger.info("测试结果摘要:")
    all_success = True
    
    for test_name, success in tests.items():
        status = "✅ 成功" if success else "❌ 失败"
        logger.info(f"{test_name}: {status}")
        if not success:
            all_success = False
    
    logger.info("=" * 50)
    if all_success:
        logger.info("🎉 所有测试通过，修复生效!")
    else:
        logger.error("❗ 部分测试失败，需要进一步修复")
    
if __name__ == "__main__":
    main()
