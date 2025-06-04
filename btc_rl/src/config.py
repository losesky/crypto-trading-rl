#!/usr/bin/env python3
"""
配置管理模块 - 加载和解析系统配置
"""

import configparser
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Config:
    """配置类，负责加载和管理系统配置"""
    
    # 默认配置
    DEFAULT_CONFIG = {
        'data': {
            'default_exchange': 'binance',
            'default_symbol': 'BTC/USDT',
            'default_timeframe': '1h',
            'default_train_start_date': '2018-01-01',
            'default_test_start_date': '2022-01-01',
            'api_max_retries': '5',
            'api_retry_delay': '10',
            'api_use_exponential_backoff': 'true'
        },
        'environment': {
            'initial_balance': 10000.0,
            'max_leverage': 3.0,
            'fee_rate': 0.0002,
            'maintenance_margin_rate': 0.05,
            'risk_fraction_per_trade': 0.05
        },
        'training': {
            'learning_rate': 0.0003,
            'batch_size': 256,
            'buffer_size': 1000000,
            'train_freq': 1,
            'gradient_steps': 1,
            'episodes': 1000
        },
        'visualization': {
            'http_port': 8080,
            'websocket_port': 8765,
            'chart_update_interval': 1000
        },
        'paths': {
            'data_dir': 'btc_rl/data',
            'models_dir': 'btc_rl/models',
            'logs_dir': 'btc_rl/logs',
            'episodes_dir': 'btc_rl/logs/episodes'
        }
    }
    
    def __init__(self, config_path=None):
        """初始化配置
        
        Args:
            config_path (str, optional): 配置文件路径. Defaults to None.
        """
        self.config = configparser.ConfigParser()
        
        # 设置默认配置
        for section, options in self.DEFAULT_CONFIG.items():
            if not self.config.has_section(section):
                self.config.add_section(section)
            for key, value in options.items():
                self.config.set(section, key, str(value))
        
        # 如果提供了配置文件路径，尝试加载
        if config_path:
            self.load(config_path)
        else:
            # 尝试从默认位置加载
            default_paths = [
                './config.ini',
                './btc_rl/config.ini',
                os.path.expanduser('~/.btc_rl/config.ini')
            ]
            
            for path in default_paths:
                if os.path.exists(path):
                    self.load(path)
                    break
    
    def load(self, config_path):
        """从文件加载配置
        
        Args:
            config_path (str): 配置文件路径
        
        Returns:
            bool: 加载是否成功
        """
        try:
            with open(config_path, 'r') as f:
                self.config.read_file(f)
            logger.info(f"配置已加载: {config_path}")
            return True
        except Exception as e:
            logger.warning(f"无法加载配置文件 {config_path}: {e}")
            return False
    
    def save(self, config_path):
        """保存配置到文件
        
        Args:
            config_path (str): 配置文件保存路径
        
        Returns:
            bool: 保存是否成功
        """
        try:
            directory = os.path.dirname(config_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            with open(config_path, 'w') as f:
                self.config.write(f)
            logger.info(f"配置已保存: {config_path}")
            return True
        except Exception as e:
            logger.error(f"无法保存配置文件 {config_path}: {e}")
            return False
    
    def get(self, section, option, fallback=None):
        """获取配置项
        
        Args:
            section (str): 配置节
            option (str): 配置项
            fallback: 默认值
        
        Returns:
            配置值
        """
        return self.config.get(section, option, fallback=fallback)
    
    def getint(self, section, option, fallback=None):
        """获取整数配置项
        
        Args:
            section (str): 配置节
            option (str): 配置项
            fallback: 默认值
        
        Returns:
            int: 配置值
        """
        return self.config.getint(section, option, fallback=fallback)
    
    def getfloat(self, section, option, fallback=None):
        """获取浮点数配置项
        
        Args:
            section (str): 配置节
            option (str): 配置项
            fallback: 默认值
        
        Returns:
            float: 配置值
        """
        return self.config.getfloat(section, option, fallback=fallback)
    
    def getboolean(self, section, option, fallback=None):
        """获取布尔值配置项
        
        Args:
            section (str): 配置节
            option (str): 配置项
            fallback: 默认值
        
        Returns:
            bool: 配置值
        """
        return self.config.getboolean(section, option, fallback=fallback)
    
    def set(self, section, option, value):
        """设置配置项
        
        Args:
            section (str): 配置节
            option (str): 配置项
            value: 配置值
        """
        if not self.config.has_section(section):
            self.config.add_section(section)
        self.config.set(section, option, str(value))
    
    def get_data_dir(self):
        """获取数据目录路径
        
        Returns:
            Path: 数据目录路径
        """
        # 移除可能存在的引号
        path_str = self.get('paths', 'data_dir').strip('"\'')
        return Path(path_str)
    
    def get_models_dir(self):
        """获取模型目录路径
        
        Returns:
            Path: 模型目录路径
        """
        # 移除可能存在的引号
        path_str = self.get('paths', 'models_dir').strip('"\'')
        return Path(path_str)
    
    def get_logs_dir(self):
        """获取日志目录路径
        
        Returns:
            Path: 日志目录路径
        """
        # 移除可能存在的引号
        path_str = self.get('paths', 'logs_dir').strip('"\'')
        return Path(path_str)
    
    def get_episodes_dir(self):
        """获取回合日志目录路径
        
        Returns:
            Path: 回合日志目录路径
        """
        # 移除可能存在的引号
        path_str = self.get('paths', 'episodes_dir').strip('"\'')
        return Path(path_str)
    
    def get_initial_balance(self):
        """获取初始余额
        
        Returns:
            float: 初始余额
        """
        return self.getfloat('environment', 'initial_balance')
    
    def get_max_leverage(self):
        """获取最大杠杆倍数
        
        Returns:
            float: 最大杠杆倍数
        """
        return self.getfloat('environment', 'max_leverage')
    
    def get_fee_rate(self):
        """获取交易费率
        
        Returns:
            float: 交易费率
        """
        return self.getfloat('environment', 'fee_rate')
    
    def get_default_exchange(self):
        """获取默认交易所
        
        Returns:
            str: 默认交易所
        """
        return self.get('data', 'default_exchange')
    
    def get_default_symbol(self):
        """获取默认交易对
        
        Returns:
            str: 默认交易对
        """
        return self.get('data', 'default_symbol')
    
    def get_default_timeframe(self):
        """获取默认时间周期
        
        Returns:
            str: 默认时间周期
        """
        return self.get('data', 'default_timeframe')
    
    def get_api_max_retries(self):
        """获取API最大重试次数
        
        Returns:
            int: API最大重试次数
        """
        return self.getint('data', 'api_max_retries')
    
    def get_api_retry_delay(self):
        """获取API重试延迟时间
        
        Returns:
            int: API重试延迟时间（秒）
        """
        return self.getint('data', 'api_retry_delay')
    
    def get_api_use_exponential_backoff(self):
        """获取是否使用指数退避算法
        
        Returns:
            bool: 是否使用指数退避算法
        """
        return self.getboolean('data', 'api_use_exponential_backoff')


# 创建全局配置实例
config = Config()

if __name__ == "__main__":
    # 测试配置模块
    logging.basicConfig(level=logging.INFO)
    
    cfg = Config("./config.ini")
    print(f"初始余额: {cfg.get_initial_balance()}")
    print(f"最大杠杆: {cfg.get_max_leverage()}")
    print(f"交易费率: {cfg.get_fee_rate()}")
    print(f"数据目录: {cfg.get_data_dir()}")
    print(f"默认交易所: {cfg.get_default_exchange()}")
    print(f"默认交易对: {cfg.get_default_symbol()}")
    print(f"默认时间周期: {cfg.get_default_timeframe()}")
    print(f"API最大重试次数: {cfg.get_api_max_retries()}")
    print(f"API重试延迟时间: {cfg.get_api_retry_delay()}")
    print(f"使用指数退避: {cfg.get_api_use_exponential_backoff()}")
