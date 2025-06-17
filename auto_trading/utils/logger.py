import os
import logging
import yaml
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


class Logger:
    """
    日志工具类，提供统一的日志记录方式
    支持多种日志级别、文件切割和控制台输出等功能
    """
    
    # 单例模式实现
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path=None, name='auto_trading', console_output=True):
        """
        初始化日志工具
        
        Args:
            config_path: 配置文件路径，默认为None（使用默认配置文件路径）
            name: 日志名称
            console_output: 是否输出到控制台
        """
        # 避免重复初始化
        if self._initialized:
            return
        
        self.name = name
        
        # 如果未提供配置文件路径，使用默认路径
        if config_path is None:
            base_dir = Path(__file__).parent.parent
            config_dir = base_dir / 'config'
            config_path = config_dir / 'log_config.yaml'
        
        # 加载配置
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            print(f"日志配置加载成功: {config_path}")
        except Exception as e:
            print(f"加载日志配置失败: {e}，使用默认配置")
            # 设置默认配置
            self.config = {
                'version': 1,
                'loggers': {
                    'base': {
                        'level': 'INFO',
                        'handlers': ['file', 'console'] if console_output else ['file'],
                        'propagate': False
                    },
                    'trading': {
                        'level': 'INFO',
                        'handlers': ['trading_file', 'console'] if console_output else ['trading_file'],
                        'propagate': False
                    },
                    'risk': {
                        'level': 'INFO',
                        'handlers': ['risk_file', 'console'] if console_output else ['risk_file'],
                        'propagate': False
                    },
                    'models': {
                        'level': 'INFO',
                        'handlers': ['model_file', 'console'] if console_output else ['model_file'],
                        'propagate': False
                    }
                },
                'handlers': {
                    'console': {
                        'class': 'logging.StreamHandler',
                        'level': 'INFO',
                        'formatter': 'standard',
                        'stream': 'ext://sys.stdout'
                    },
                    'file': {
                        'class': 'logging.handlers.RotatingFileHandler',
                        'level': 'INFO',
                        'formatter': 'detailed',
                        'filename': 'logs/app.log',
                        'maxBytes': 10485760,  # 10MB
                        'backupCount': 5
                    },
                    'trading_file': {
                        'class': 'logging.handlers.TimedRotatingFileHandler',
                        'level': 'INFO',
                        'formatter': 'detailed',
                        'filename': 'logs/trading.log',
                        'when': 'midnight',
                        'interval': 1,
                        'backupCount': 30
                    },
                    'risk_file': {
                        'class': 'logging.handlers.TimedRotatingFileHandler',
                        'level': 'INFO',
                        'formatter': 'detailed',
                        'filename': 'logs/risk.log',
                        'when': 'midnight',
                        'interval': 1,
                        'backupCount': 30
                    },
                    'model_file': {
                        'class': 'logging.handlers.TimedRotatingFileHandler',
                        'level': 'INFO',
                        'formatter': 'detailed',
                        'filename': 'logs/model.log',
                        'when': 'midnight',
                        'interval': 1,
                        'backupCount': 30
                    }
                },
                'formatters': {
                    'standard': {
                        'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
                    },
                    'detailed': {
                        'format': '%(asctime)s [%(levelname)s] %(name)s:%(filename)s:%(lineno)d - %(message)s'
                    }
                },
                'root': {
                    'level': 'INFO',
                    'handlers': ['console', 'file'] if console_output else ['file']
                }
            }
            
        # 确保日志目录存在
        base_dir = Path(__file__).parent.parent
        log_dir = base_dir / 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        # 更新日志文件路径为绝对路径
        for handler_name, handler_config in self.config.get('handlers', {}).items():
            if 'filename' in handler_config:
                # 将相对路径转换为绝对路径
                if not os.path.isabs(handler_config['filename']):
                    handler_config['filename'] = os.path.join(log_dir, os.path.basename(handler_config['filename']))
        
        # 初始化日志系统
        self.setup_logging()
        
        # 获取基础日志记录器
        self.logger = logging.getLogger(self.name)
        
        self._initialized = True
        
    def setup_logging(self):
        """设置日志系统"""
        try:
            # 重置现有的日志处理器
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            
            # 配置根日志记录器
            root_config = self.config.get('root', {})
            logging.root.setLevel(root_config.get('level', 'INFO'))
            
            # 创建格式化器
            formatters = {}
            for fmt_name, fmt_config in self.config.get('formatters', {}).items():
                formatters[fmt_name] = logging.Formatter(fmt_config.get('format'))
            
            # 创建处理器
            handlers = {}
            for handler_name, handler_config in self.config.get('handlers', {}).items():
                handler_class = handler_config.get('class')
                
                if handler_class == 'logging.StreamHandler':
                    handler = logging.StreamHandler(sys.stdout)
                elif handler_class == 'logging.handlers.RotatingFileHandler':
                    handler = RotatingFileHandler(
                        filename=handler_config.get('filename'),
                        maxBytes=handler_config.get('maxBytes', 10485760),  # 默认10MB
                        backupCount=handler_config.get('backupCount', 5)
                    )
                elif handler_class == 'logging.handlers.TimedRotatingFileHandler':
                    handler = TimedRotatingFileHandler(
                        filename=handler_config.get('filename'),
                        when=handler_config.get('when', 'midnight'),
                        interval=handler_config.get('interval', 1),
                        backupCount=handler_config.get('backupCount', 30)
                    )
                else:
                    # 默认使用RotatingFileHandler
                    handler = RotatingFileHandler(
                        filename=handler_config.get('filename', 'logs/app.log'),
                        maxBytes=handler_config.get('maxBytes', 10485760),
                        backupCount=handler_config.get('backupCount', 5)
                    )
                
                # 设置级别和格式化器
                handler.setLevel(handler_config.get('level', 'INFO'))
                formatter_name = handler_config.get('formatter', 'standard')
                if formatter_name in formatters:
                    handler.setFormatter(formatters[formatter_name])
                    
                handlers[handler_name] = handler
                
                # 添加到根记录器
                if handler_name in root_config.get('handlers', []):
                    logging.root.addHandler(handler)
            
            # 配置各个日志记录器
            for logger_name, logger_config in self.config.get('loggers', {}).items():
                logger = logging.getLogger(logger_name)
                logger.setLevel(logger_config.get('level', 'INFO'))
                logger.propagate = logger_config.get('propagate', False)
                
                # 清除现有处理器
                for handler in logger.handlers[:]:
                    logger.removeHandler(handler)
                    
                # 添加新的处理器
                for handler_name in logger_config.get('handlers', []):
                    if handler_name in handlers:
                        logger.addHandler(handlers[handler_name])
            
            print(f"日志系统初始化成功")
            
        except Exception as e:
            print(f"日志系统初始化失败: {e}")
            # 设置一个基本的控制台日志记录器，以确保至少有日志输出
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                handlers=[logging.StreamHandler()]
            )
    
    def get_logger(self, name=None):
        """
        获取指定名称的日志记录器
        
        Args:
            name: 日志记录器名称，默认为None（使用默认名称）
            
        Returns:
            Logger: 日志记录器对象
        """
        if name is None:
            return self.logger
        
        return logging.getLogger(name)
        
    def get_trading_logger(self):
        """获取交易模块的日志记录器"""
        return self.get_logger('trading')
        
    def get_risk_logger(self):
        """获取风控模块的日志记录器"""
        return self.get_logger('risk')
        
    def get_model_logger(self):
        """获取模型模块的日志记录器"""
        return self.get_logger('models')
        
    def get_data_logger(self):
        """获取数据模块的日志记录器"""
        return self.get_logger('data')
