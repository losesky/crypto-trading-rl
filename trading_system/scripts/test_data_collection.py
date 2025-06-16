#!/usr/bin/env python3
"""
测试数据收集功能的正常工作
此脚本用于验证模型包装器的数据收集功能是否正常工作
"""

import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
script_dir = Path(__file__).parent
root_dir = script_dir.parent.parent
sys.path.append(str(root_dir))

# 导入模型包装器
from trading_system.src.model_wrapper import ModelWrapper

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("test_data_collection")

def load_config():
    """加载测试配置"""
    config_path = root_dir / "trading_system" / "config" / "test_config.json"
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return None

def generate_test_data():
    """生成测试用的市场和持仓数据"""
    current_price = 40000 + np.random.normal(0, 500)
    
    market_data = {
        'open': current_price * 0.99,
        'high': current_price * 1.02,
        'low': current_price * 0.98,
        'close': current_price,
        'volume': 100 + np.random.rand() * 900,
        'timestamp': int(datetime.now().timestamp() * 1000)
    }
    
    position_data = {
        'size': np.random.rand() * 0.5,
        'side': 'BUY' if np.random.rand() > 0.5 else 'SELL',
        'entry_price': current_price * 0.99,
        'unrealized_pnl': np.random.normal(0, 50),
        'margin': 1000,
        'leverage': 3
    }
    
    return market_data, position_data

def test_data_collection():
    """测试数据收集功能"""
    # 1. 加载配置
    config = load_config()
    if not config:
        return False
    
    # 2. 初始化模型包装器
    model_path = config.get('general', {}).get('model_path')
    if not model_path or not os.path.exists(model_path):
        logger.warning(f"模型文件不存在: {model_path}")
        model_path = str(root_dir / "btc_rl/models/sac_ep2.zip")
        logger.info(f"使用替代模型: {model_path}")
    
    logger.info("初始化模型包装器...")
    model_wrapper = ModelWrapper(model_path, config)
    
    # 3. 记录几条交易经验
    logger.info("测试记录交易经验...")
    for _ in range(5):
        market_data, position_data = generate_test_data()
        action_value = np.random.uniform(-1, 1)
        reward = np.random.normal(0, 10)
        
        model_wrapper.record_trading_experience(
            market_data=market_data,
            position_data=position_data,
            action_value=action_value,
            reward=reward
        )
    
    # 4. 记录一条交易结果
    logger.info("测试记录交易结果...")
    trade_result = {
        'trade_id': f"test_{int(datetime.now().timestamp())}",
        'entry_time': datetime.now().isoformat(),
        'exit_time': datetime.now().isoformat(),
        'side': 'BUY',
        'entry_price': 40000,
        'exit_price': 40500,
        'size': 0.1,
        'profit_pct': 0.0125,
        'absolute_profit': 50,
        'is_profitable': True
    }
    
    model_wrapper.record_trade_result(trade_result)
    
    # 5. 检查数据文件是否已创建
    logger.info("检查数据文件...")
    data_dir = model_wrapper.data_collection_dir
    experiences_dir = data_dir / "experiences"
    trades_dir = data_dir / "trades"
    metrics_file = data_dir / "model_metrics.json"
    
    files_created = []
    
    if experiences_dir.exists() and len(list(experiences_dir.glob("*.json"))) > 0:
        files_created.append(f"交易经验文件: {len(list(experiences_dir.glob('*.json')))}个")
    
    if trades_dir.exists() and len(list(trades_dir.glob("*.json"))) > 0:
        files_created.append(f"交易记录文件: {len(list(trades_dir.glob('*.json')))}个")
    
    if metrics_file.exists():
        files_created.append("模型指标文件")
    
    if files_created:
        logger.info(f"已成功创建数据文件: {', '.join(files_created)}")
        return True
    else:
        logger.error("未能创建任何数据文件，数据收集功能可能存在问题")
        return False

if __name__ == "__main__":
    logger.info("开始测试数据收集功能...")
    success = test_data_collection()
    
    if success:
        logger.info("数据收集功能测试成功！")
        logger.info("现在可以运行分析脚本检查结果:")
        logger.info("./trading_system/scripts/analyze_model_performance.sh")
        sys.exit(0)
    else:
        logger.error("数据收集功能测试失败")
        sys.exit(1)
