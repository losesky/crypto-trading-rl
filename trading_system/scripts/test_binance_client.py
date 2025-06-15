#!/usr/bin/env python3
"""
测试币安客户端连接和API调用

此脚本用于测试币安API连接问题的修复是否有效，包括测试网环境下的时间同步和API调用。

使用：
python test_binance_client.py
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
        logging.FileHandler('binance_client_test.log')
    ]
)

logger = logging.getLogger("BinanceClientTest")

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

def test_api_calls(client):
    """测试各种API调用"""
    results = {}
    
    # 测试1：获取服务器时间
    logger.info("测试1：获取服务器时间")
    try:
        server_time, local_time = client.get_server_time()
        results["server_time"] = {
            "success": server_time is not None,
            "data": {
                "server_time": server_time,
                "local_time": local_time,
                "difference_ms": server_time - local_time if server_time else None
            }
        }
    except Exception as e:
        logger.error(f"获取服务器时间失败: {e}")
        results["server_time"] = {"success": False, "error": str(e)}
    
    # 测试2：获取账户信息
    logger.info("测试2：获取账户信息")
    try:
        account_info = client.get_account_info()
        results["account_info"] = {
            "success": account_info is not None,
            "data_keys": list(account_info.keys()) if account_info else None
        }
    except Exception as e:
        logger.error(f"获取账户信息失败: {e}")
        results["account_info"] = {"success": False, "error": str(e)}
    
    # 测试3：获取余额
    logger.info("测试3：获取余额")
    try:
        balance = client.get_balance()
        results["balance"] = {
            "success": balance is not None,
            "data": {
                "free": balance.get("free", {}),
                "used": balance.get("used", {}),
                "total": balance.get("total", {})
            } if balance else None
        }
    except Exception as e:
        logger.error(f"获取余额失败: {e}")
        results["balance"] = {"success": False, "error": str(e)}
    
    # 测试4：获取持仓信息
    logger.info("测试4：获取持仓信息")
    try:
        positions = client.get_positions()
        results["positions"] = {
            "success": positions is not None,
            "data": {
                "count": len(positions) if positions else 0,
                "symbols": [pos.get("symbol") for pos in positions] if positions else []
            }
        }
    except Exception as e:
        logger.error(f"获取持仓信息失败: {e}")
        results["positions"] = {"success": False, "error": str(e)}
    
    # 测试5：获取最新价格
    logger.info("测试5：获取最新价格")
    try:
        ticker = client.get_ticker("BTCUSDT")
        results["ticker"] = {
            "success": ticker is not None,
            "data": {
                "last": ticker.get("last"),
                "bid": ticker.get("bid"),
                "ask": ticker.get("ask")
            } if ticker else None
        }
    except Exception as e:
        logger.error(f"获取最新价格失败: {e}")
        results["ticker"] = {"success": False, "error": str(e)}
    
    # 测试6：获取K线数据
    logger.info("测试6：获取K线数据")
    try:
        end_time = int(time.time() * 1000)
        start_time = end_time - 3600000  # 1小时前
        klines = client.get_historical_klines("BTCUSDT", "1m", start_time, end_time, 10)
        
        results["klines"] = {
            "success": klines is not None,
            "data": {
                "shape": klines.shape if klines is not None else None,
                "columns": list(klines.columns) if klines is not None else None
            }
        }
    except Exception as e:
        logger.error(f"获取K线数据失败: {e}")
        results["klines"] = {"success": False, "error": str(e)}
    
    return results

def main():
    """主函数"""
    logger.info("开始测试币安客户端")
    
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
    
    # 执行API调用测试
    results = test_api_calls(client)
    
    # 打印测试结果
    logger.info("测试结果摘要:")
    for test_name, result in results.items():
        status = "✅ 成功" if result.get("success") else "❌ 失败"
        logger.info(f"{test_name}: {status}")
    
    # 保存详细结果到文件
    with open('binance_api_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("测试完成，详细结果已保存到 binance_api_test_results.json")

if __name__ == "__main__":
    main()
