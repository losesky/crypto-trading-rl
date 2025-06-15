#!/usr/bin/env python3
"""
测试币安客户端在测试网环境下的问题修复

此脚本专门测试对"binance does not have a testnet/sandbox URL for sapi endpoints"错误的修复
"""

import os
import sys
import json
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
        logging.FileHandler('binance_client_fix_test.log')
    ]
)

logger = logging.getLogger("BinanceFixTest")

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

def test_get_positions(client):
    """测试获取持仓信息"""
    logger.info("测试获取持仓信息")
    try:
        positions = client.get_positions()
        logger.info(f"获取持仓信息成功: {positions}")
        return True
    except Exception as e:
        logger.error(f"获取持仓信息失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("开始测试币安客户端修复")
    
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
    
    # 测试获取持仓信息
    success = test_get_positions(client)
    
    if success:
        logger.info("✅ 测试通过: 持仓信息获取成功，修复有效")
    else:
        logger.error("❌ 测试失败: 持仓信息获取仍然存在问题")

if __name__ == "__main__":
    main()
