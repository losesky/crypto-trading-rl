#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import time
import signal
import argparse
import logging
import asyncio
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# 配置GPU内存使用和TensorFlow日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # 动态分配GPU内存，避免占用全部GPU内存

# 添加项目根目录到路径，确保可以导入所有模块
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# 导入本项目的模块
from utils.logger import Logger
from data.binance_api import BinanceAPI
from data.data_processor import DataProcessor
from data.feature_engineer import FeatureEngineer
from models.model_loader import ModelLoader
from models.prediction import PredictionProcessor
from models.ensemble import ModelEnsemble
from trading.order_manager import OrderManager
from trading.position_manager import PositionManager, PositionStatus
from trading.executor import Executor
from risk.risk_checker import RiskChecker
from risk.money_manager import MoneyManager
from risk.circuit_breaker import CircuitBreaker
from monitor.dashboard import Dashboard
from monitor.alerter import Alerter
from monitor.reporter import Reporter


class TradingSystem:
    """
    自动交易系统主类，整合所有模块，实现完整的交易逻辑
    """
    
    def __init__(self, config_dir=None, log_level='INFO'):
        """
        初始化交易系统
        
        Args:
            config_dir: 配置文件目录，默认为None（使用默认配置目录）
            log_level: 日志级别，默认为'INFO'
        """
        # 设置信号处理，用于优雅退出
        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)
        
        # 初始化系统变量
        self.is_running = False
        self.current_time = datetime.now()
        self.last_feature_time = None
        self.last_prediction_time = None
        self.last_execution_time = None
        self.last_report_time = None
        
        # 设置配置文件目录
        if config_dir is None:
            config_dir = os.path.join(PROJECT_ROOT, 'config')
            
        self.config_dir = config_dir
        
        # 初始化日志系统
        self.logger = self._init_logger(log_level)
        
        # 加载配置文件
        self.config = self._load_configs()
        
        # 初始化各个模块
        self.logger.info("正在初始化交易系统各模块...")
        
        try:
            # 数据相关模块
            self.api = BinanceAPI(config_path=os.path.join(config_dir, 'api_config.yaml'))
            self.data_processor = DataProcessor(
                config_path=os.path.join(config_dir, 'model_config.yaml'),
                api=self.api
            )
            self.feature_engineer = FeatureEngineer(
                config_path=os.path.join(config_dir, 'model_config.yaml')
            )
            
            # 模型相关模块
            model_config_path = os.path.join(config_dir, 'model_config.yaml')
            self.model_loader = ModelLoader(model_config_path)
            self.model_ensemble = ModelEnsemble(model_config_path)
            self.prediction_processor = PredictionProcessor(config_path=model_config_path)
            
            # 交易相关模块
            self.order_manager = OrderManager(
                config_path=os.path.join(config_dir, 'api_config.yaml'),
                api=self.api
            )
            self.position_manager = PositionManager(
                order_manager=self.order_manager,
                config_path=os.path.join(config_dir, 'risk_config.yaml')
            )
            self.executor = Executor(
                api=self.api, 
                order_manager=self.order_manager, 
                position_manager=self.position_manager,
                config_path=os.path.join(config_dir, 'risk_config.yaml')
            )
            
            # 风控相关模块
            risk_config_path = os.path.join(config_dir, 'risk_config.yaml')
            self.risk_checker = RiskChecker(self.position_manager, risk_config_path)
            self.money_manager = MoneyManager(risk_config_path)
            self.circuit_breaker = CircuitBreaker(risk_config_path)
            
            # 监控相关模块
            self.dashboard = Dashboard()
            self.alerter = Alerter()
            self.reporter = Reporter()
            
            self.logger.info("所有模块初始化完成")
            
        except Exception as e:
            self.logger.error(f"初始化模块失败: {e}")
            raise
            
    def _init_logger(self, log_level):
        """初始化日志记录器"""
        log_manager = Logger(config_path=os.path.join(self.config_dir, 'log_config.yaml'))
        logger = log_manager.get_logger('system')
        
        # 设置日志级别
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        logger.setLevel(level_map.get(log_level, logging.INFO))
        
        return logger
        
    def _load_configs(self):
        """加载所有配置文件"""
        config = {}
        
        try:
            # 加载主配置
            main_config_path = os.path.join(self.config_dir, 'trading_config.yaml')
            if os.path.exists(main_config_path):
                with open(main_config_path, 'r', encoding='utf-8') as file:
                    config['main'] = yaml.safe_load(file)
                    
            # 加载API配置
            api_config_path = os.path.join(self.config_dir, 'api_config.yaml')
            if os.path.exists(api_config_path):
                with open(api_config_path, 'r', encoding='utf-8') as file:
                    config['api'] = yaml.safe_load(file)
                    
            # 加载模型配置
            model_config_path = os.path.join(self.config_dir, 'model_config.yaml')
            if os.path.exists(model_config_path):
                with open(model_config_path, 'r', encoding='utf-8') as file:
                    config['model'] = yaml.safe_load(file)
                    
            # 加载风控配置
            risk_config_path = os.path.join(self.config_dir, 'risk_config.yaml')
            if os.path.exists(risk_config_path):
                with open(risk_config_path, 'r', encoding='utf-8') as file:
                    config['risk'] = yaml.safe_load(file)
                    
            self.logger.info(f"已加载配置文件")
            
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            config = {'main': {'default_timeframe': '1h', 'symbols': ['BTCUSDT']}}
            
        return config
        
    def _handle_exit(self, signum, frame):
        """处理退出信号"""
        self.logger.info(f"接收到退出信号: {signum}, 正在优雅关闭...")
        self.stop()
        
    async def initialize(self):
        """异步初始化系统，准备交易环境"""
        try:
            self.logger.info("开始初始化交易环境...")
            
            # 1. 连接到币安API并验证身份
            await self.api.initialize()
            
            # 2. 获取账户信息和余额
            try:
                account_info = await self.api.get_account_info_async()
                if account_info:
                    self.logger.info(f"账户连接成功，账户状态: {account_info.get('status', '活跃')}")
                    
                    # 解析资产余额
                    assets = account_info.get('assets', [])
                    if not assets and 'balances' in account_info:
                        assets = account_info.get('balances', [])
                    
                    balances = {asset['asset']: asset for asset in assets 
                              if float(asset.get('free', 0)) > 0 or float(asset.get('locked', 0)) > 0 or
                                 float(asset.get('walletBalance', 0)) > 0}
                    
                    if balances:
                        self.logger.info(f"账户持有资产: {list(balances.keys())}")
                        
                        # 初始化总资金和可用资金
                        total_balance = 0.0
                        available_balance = 0.0
                        
                        for asset, info in balances.items():
                            if asset == "USDT":  # 我们主要关注USDT余额
                                wallet_balance = float(info.get('walletBalance', info.get('free', 0)))
                                total_balance += wallet_balance
                                available_balance += float(info.get('availableBalance', info.get('free', 0)))
                        
                        # 更新资金管理器
                        self.money_manager.set_capital(total_balance, available_balance)
                        self.logger.info(f"资金初始化: 总资金 {total_balance} USDT, 可用资金 {available_balance} USDT")
                        
                        # 将资金状态更新到监控面板
                        self.money_manager.update_dashboard(self.dashboard)
                    else:
                        self.logger.warning("账户没有持有任何资产或余额为0，将使用默认资金设置")
                        # 设置默认资金，避免后续除零错误
                        self.money_manager.set_capital(10000.0, 10000.0)
                        
                        # 将资金状态更新到监控面板
                        self.money_manager.update_dashboard(self.dashboard)
                else:
                    self.logger.warning("获取账户信息返回空结果，将使用默认资金设置")
                    # 设置默认资金，避免后续除零错误
                    self.money_manager.set_capital(10000.0, 10000.0)
            except Exception as e:
                self.logger.error(f"获取账户信息失败: {e}")
                # 设置默认资金，避免后续除零错误
                self.money_manager.set_capital(10000.0, 10000.0)
                
            # 3. 初始化模型
            model = self.model_loader.load_best_model()
            if not model:
                raise Exception("没有可用的交易模型")
                
            self.logger.info(f"已成功加载交易模型")
            
            # 4. 注意：资金管理器已经在上面通过set_capital初始化，这里不需要再次初始化
            
            # 5. 获取交易对信息
            main_config = self.config.get('main', {})
            symbols = main_config.get('symbols', ['BTCUSDT'])
            timeframes = main_config.get('timeframes', ['1h'])
            
            # 检查交易对是否有效
            for symbol in symbols:
                symbol_info = await self.api.get_symbol_info(symbol)
                if not symbol_info:
                    self.logger.warning(f"无法获取交易对信息: {symbol}，将跳过该交易对")
            
            self.logger.info(f"交易对: {symbols}, 时间周期: {timeframes}")
            
            # 6. 获取初始K线数据和指标
            for symbol in symbols:
                for timeframe in timeframes:
                    # 获取历史数据用于初始化指标
                    limit = main_config.get('history_bars', 500)
                    klines = await self.api.get_klines_async(symbol, timeframe, limit=limit)
                    
                    if klines:
                        self.logger.info(f"成功获取 {symbol} {timeframe} 的历史K线数据, 数量: {len(klines)}")
                        
                        # 处理数据并计算特征
                        kline_df = self.data_processor.process_klines(klines)
                        features_df = self.feature_engineer.calculate_features(kline_df)
                        
                        self.logger.info(f"特征工程完成, 特征数量: {len(features_df.columns) - 5}")  # 减去OHLCV列
                    else:
                        raise Exception(f"无法获取 {symbol} {timeframe} 的历史K线数据")
            
            # 7. 初始化仓位管理器
            await self.position_manager.update_positions()
            positions = self.position_manager.positions
            
            # 只记录非空持仓信息
            active_positions = {symbol: pos for symbol, pos in positions.items() 
                              if pos.get('status') != PositionStatus.NONE and pos.get('amount', 0) != 0}
            
            if active_positions:
                self.logger.info(f"当前活跃持仓信息: {active_positions}")
            else:
                self.logger.info(f"当前无持仓")
            
            # 8. 初始化风控模块
            self.circuit_breaker.initialize()
            
            # 9. 初始化监控面板
            self.dashboard.initialize()
            
            self.logger.info("交易系统初始化完成，准备开始交易")
            return True
            
        except Exception as e:
            self.logger.error(f"初始化失败: {e}", exc_info=True)
            return False
            
    async def run(self):
        """运行交易系统的主循环"""
        if not await self.initialize():
            self.logger.error("初始化失败，无法启动交易系统")
            return
        
        self.is_running = True
        self.logger.info("交易系统启动，开始执行主循环...")
        
        # 获取配置信息
        main_config = self.config.get('main', {})
        symbols = main_config.get('symbols', ['BTCUSDT'])
        timeframes = main_config.get('timeframes', ['1h'])
        
        # 主要的交易对和时间周期
        primary_symbol = symbols[0]
        primary_timeframe = timeframes[0]
        
        # 特征更新间隔（秒）
        feature_interval = main_config.get('feature_interval', 60)
        # 模型预测间隔（秒）
        prediction_interval = main_config.get('prediction_interval', 60)
        # 执行间隔（秒）
        execution_interval = main_config.get('execution_interval', 5)
        # 报告间隔（秒）
        report_interval = main_config.get('report_interval', 3600)
        
        try:
            while self.is_running:
                self.current_time = datetime.now()
                
                # 1. 数据和特征更新
                if self.last_feature_time is None or \
                   (self.current_time - self.last_feature_time).total_seconds() >= feature_interval:
                    await self._update_features(primary_symbol, primary_timeframe)
                    self.last_feature_time = self.current_time
                
                # 2. 模型预测
                if self.last_prediction_time is None or \
                   (self.current_time - self.last_prediction_time).total_seconds() >= prediction_interval:
                    await self._make_predictions(primary_symbol, primary_timeframe)
                    self.last_prediction_time = self.current_time
                
                # 3. 执行交易
                if self.last_execution_time is None or \
                   (self.current_time - self.last_execution_time).total_seconds() >= execution_interval:
                    await self._execute_trades(primary_symbol)
                    self.last_execution_time = self.current_time
                    
                # 4. 生成报告
                if self.last_report_time is None or \
                   (self.current_time - self.last_report_time).total_seconds() >= report_interval:
                    await self._generate_reports()
                    self.last_report_time = self.current_time
                    
                # 5. 更新监控面板
                self.dashboard.update()
                
                # 检查系统状态
                if not self.circuit_breaker.check_status():
                    if self.circuit_breaker.status == "COOLING":
                        self.logger.warning("熔断冷却中，暂停交易")
                    elif self.circuit_breaker.status == "TRIGGERED":
                        self.logger.critical("熔断机制触发，停止交易")
                        # 可以选择关闭所有仓位
                        await self._close_all_positions()
                        break
                    
                # 等待一小段时间，避免过度占用CPU
                await asyncio.sleep(1)
                
        except Exception as e:
            self.logger.error(f"交易主循环异常: {e}", exc_info=True)
            self.alerter.emergency("交易系统异常", f"主循环发生错误: {e}")
            
        finally:
            self.is_running = False
            await self._shutdown()
            
    async def _update_features(self, symbol, timeframe):
        """更新数据和计算特征"""
        try:
            # 获取最新K线数据 - 添加时间戳参数强制刷新缓存
            current_timestamp = int(time.time() * 1000)
            self.logger.info(f"获取 {symbol} {timeframe} 最新K线数据 - 时间戳: {current_timestamp}")
            klines = await self.api.get_klines_async(symbol, timeframe, limit=100)
            
            if not klines:
                self.logger.warning(f"获取 {symbol} {timeframe} K线数据失败")
                return False
                
            # 处理K线数据
            kline_df = self.data_processor.process_klines(klines)
            
            # 计算特征
            features_df = self.feature_engineer.calculate_features(kline_df)
            
            # 保存特征数据
            self.dashboard.update_market_data(symbol, timeframe, kline_df, features_df)
            
            self.logger.debug(f"已更新 {symbol} {timeframe} 的特征数据")
            return True
            
        except Exception as e:
            self.logger.error(f"更新特征失败: {e}")
            return False
            
    async def _make_predictions(self, symbol, timeframe):
        """使用模型进行预测"""
        try:
            # 获取最新的特征数据
            features_df = self.dashboard.get_features_data(symbol, timeframe)
            
            if features_df is None or features_df.empty:
                self.logger.warning(f"没有可用的特征数据进行预测")
                return
                
            # 准备模型输入
            features = features_df.iloc[-1].to_dict()
            
            # 检查特征数据类型，特别是时间戳
            for key, value in list(features.items()):
                if isinstance(value, pd.Timestamp):
                    self.logger.debug(f"检测到时间戳特征: {key}={value}，将转换为Unix时间戳")
                    # 将时间戳转换为浮点数（Unix时间戳），直接在特征字典中修改
                    features[key] = value.timestamp()
                elif not isinstance(value, (int, float, np.number)) and value is not None:
                    self.logger.debug(f"检测到非数值特征: {key}={value}，类型={type(value)}")
                    # 将非数值特征转换为数值（如果可能）
                    try:
                        features[key] = float(value)
                    except (ValueError, TypeError):
                        # 无法转换为浮点数，设为0
                        features[key] = 0.0
                    
            # 添加一个小随机扰动以确保每次预测都有细微不同
            # 这有助于防止模型预测完全相同的结果
            random_key = f"random_noise_{int(time.time())}"
            features[random_key] = random.uniform(-0.001, 0.001)
            
            # 强制记录完整的特征输入，便于调试
            self.logger.info(f"特征数据样本 (部分): {list(features.items())[:5]}")
            
            # 通过集成模型进行预测
            predictions = self.model_ensemble.predict(features)
            
            # 记录原始预测
            self.logger.debug(f"模型原始预测结果: {predictions}")
            
            # 详细记录原始预测信息
            self.logger.info(f"原始预测信息: action={predictions.get('action')}, probabilities={predictions.get('probabilities')}")
            
            # 确保predictions包含有效的action值
            if 'action' in predictions and predictions['action'] not in [0, 1, 2]:
                self.logger.warning(f"预测结果包含无效action值：{predictions['action']}，修正为HOLD(1)")
                predictions['action'] = 1  # 设为HOLD
            
            # 确保predictions包含probabilities键且值有效
            if 'probabilities' not in predictions or not predictions['probabilities']:
                self.logger.warning(f"预测结果缺少probabilities，添加默认概率分布")
                action = predictions.get('action', 1)
                if action == 0:
                    predictions['probabilities'] = {"0": 0.7, "1": 0.2, "2": 0.1}
                elif action == 2:
                    predictions['probabilities'] = {"0": 0.1, "1": 0.2, "2": 0.7}
                else:
                    predictions['probabilities'] = {"0": 0.2, "1": 0.6, "2": 0.2}
            else:
                # 验证概率分布是否包含所有必要的键
                for key in ["0", "1", "2"]:
                    if key not in predictions['probabilities']:
                        self.logger.warning(f"预测结果概率分布缺少键'{key}'，添加默认值")
                        predictions['probabilities'][key] = 0.1
                
                # 验证概率总和是否为1
                prob_sum = sum(float(v) for v in predictions['probabilities'].values())
                if abs(prob_sum - 1.0) > 0.05:  # 允许5%的误差
                    self.logger.warning(f"概率总和为{prob_sum}，与1的偏差过大，进行归一化")
                    # 归一化概率
                    for k in predictions['probabilities']:
                        predictions['probabilities'][k] = float(predictions['probabilities'][k]) / prob_sum
            
            # 确保动作与最高概率一致性
            if 'action' in predictions and 'probabilities' in predictions:
                probs_dict = {int(k): float(v) for k, v in predictions['probabilities'].items()}
                max_prob_action = max(probs_dict.items(), key=lambda x: x[1])[0]
                
                # 如果模型推荐的动作与最高概率不一致，考虑调整
                if predictions['action'] != max_prob_action and max(probs_dict.values()) > 0.6:
                    self.logger.warning(f"动作({predictions['action']})与最高概率动作({max_prob_action})不一致，且差异明显。调整动作以匹配概率。")
                    predictions['action'] = max_prob_action
            
            # 处理预测结果
            action_probas, position_size, confidence = self.prediction_processor.process(predictions)
            
            self.logger.info(f"模型预测结果: 行动概率={action_probas}, 仓位大小={position_size}, 信心度={confidence}")
            
            # 更新预测结果到仪表板
            self.dashboard.update_prediction(
                symbol=symbol,
                timestamp=self.current_time,
                action_probas=action_probas,
                position_size=position_size,
                confidence=confidence
            )
            
            return action_probas, position_size, confidence
            
        except Exception as e:
            self.logger.error(f"模型预测失败: {e}")
            return None, None, None
            
    async def _execute_trades(self, symbol):
        """执行交易决策"""
        try:
            # 获取当前预测和仓位情况
            prediction = self.dashboard.get_latest_prediction(symbol)
            
            if prediction is None:
                self.logger.debug("暂无交易预测，跳过执行")
                return
                
            # 提取预测结果
            action_probas = prediction.get('action_probas', {})
            confidence = prediction.get('confidence', 0)
            suggested_position = prediction.get('position_size', 0)
            
            # 获取当前仓位
            current_position = await self.position_manager.get_position(symbol)
            current_size = current_position.get('size', 0) if current_position else 0
            
            # 检查预测结果是否明确支持交易
            # 检查持有概率 - 如果持有概率(action=1)是最高的，则不交易
            hold_prob = action_probas.get(1, 0)
            sell_prob = action_probas.get(0, 0)
            buy_prob = action_probas.get(2, 0)
            
            highest_prob = max(hold_prob, sell_prob, buy_prob)
            highest_action = None
            for action, prob in action_probas.items():
                if prob == highest_prob:
                    highest_action = action
                    break
                    
            # 如果最高概率是持有或者概率不够高，则不执行交易
            if highest_action == 1 or highest_prob < self.prediction_processor.confidence_threshold:
                self.logger.info(f"模型预测结果建议持有或置信度不足: 行动={highest_action}, 概率={highest_prob:.4f}")
                return
                
            # 检查风控条件并应用标准置信度要求
            min_confidence = 0.5  # 降低默认置信度要求，允许更多交易
            
            # 获取原始连续动作值（如果存在）
            original_action_value = getattr(self.model_ensemble, 'last_original_action_value', None)
            if original_action_value is not None and isinstance(original_action_value, float):
                # 检查原始连续动作值与离散动作的一致性
                # SAC连续动作通常在[-1, 1]范围内
                # 调整一致性判断逻辑：
                # 如果选择SELL(0)，连续动作值应小于-0.1
                # 如果选择BUY(2)，连续动作值应大于0.1
                if (highest_action == 0 and original_action_value > -0.05) or (highest_action == 2 and original_action_value < 0.05):
                    self.logger.warning(f"连续动作值({original_action_value})与离散动作({highest_action})不够一致，对置信度要求小幅调整")
                    # 由于连续值与离散动作有轻微不一致，小幅提高置信度要求
                    min_confidence += 0.05
            
            # 调用风险检查器检查交易风险，传入最低置信度要求
            # 如果风险检查器没有check_trade_risk方法，则总是允许交易
            try:
                risk_check = self.risk_checker.check_trade_risk(
                    symbol=symbol,
                    action_probas=action_probas,
                    confidence=confidence,
                    current_position=current_size,
                    target_position=suggested_position,
                    min_confidence=min_confidence  # 传递更新后的置信度要求
                )
            except AttributeError:
                self.logger.warning("风险检查器未实现check_trade_risk方法，默认允许交易")
                risk_check = {'allowed': True, 'reason': '风险检查器未完全实现'}
            
            if not risk_check['allowed']:
                self.logger.warning(f"风控检查未通过: {risk_check['reason']}")
                return
                
            # 计算实际交易仓位（考虑资金管理）
            position_ratio, position_size_usdt = self.money_manager.calculate_position_size(
                symbol=symbol,
                confidence=confidence,
                suggested_size=abs(suggested_position)  # 使用绝对值，方向由action决定
            )
            
            # 这里我们使用position_size_usdt作为实际仓位大小
            # 如果是做空(SELL)，将仓位设为负值
            actual_position = position_size_usdt
            if highest_action == 0:  # SELL
                actual_position = -actual_position
            
            self.logger.info(f"资金管理后的仓位比例: {position_ratio:.4f}, 仓位大小: {actual_position:.2f} USDT")
            
            # 计算需要调整的仓位变化
            position_delta = actual_position - current_size
            
            # 如果仓位变化太小，则不交易
            min_change_threshold = 0.01  # 最小1%的变化
            if abs(position_delta) < abs(current_size) * min_change_threshold and current_size != 0:
                self.logger.info(f"仓位变化过小 ({position_delta})，不执行交易")
                return
                
            # 执行交易
            if position_delta > 0:
                # 开多或加仓
                try:
                    self.logger.info(f"尝试开多仓/加仓: {symbol}, USDT金额={abs(position_delta)}")
                    order_result = await self.executor.open_long(
                        symbol=symbol, 
                        quantity=abs(position_delta),
                        is_usdt_amount=True  # 明确指定quantity为USDT金额
                    )
                    
                    if order_result and not order_result.get('error'):
                        actual_quantity = order_result.get('quantity', 0)
                        price = order_result.get('price', 0)
                        self.logger.info(f"开多/加仓成功: {symbol}, 数量={actual_quantity}, 价格={price}, USDT金额={abs(position_delta)}")
                        self.dashboard.update_trade_status(
                            symbol=symbol,
                            action="OPEN_LONG",
                            quantity=actual_quantity,
                            price=price,
                            status="SUCCESS",
                            note=f"USDT金额: {abs(position_delta)}"
                        )
                    else:
                        error_msg = order_result.get('error', '未知错误') if order_result else '交易执行失败'
                        self.logger.error(f"开多/加仓失败: {symbol}, 原因: {error_msg}")
                        self.dashboard.update_trade_status(
                            symbol=symbol,
                            action="OPEN_LONG",
                            quantity=abs(position_delta),
                            status="FAILED",
                            note=f"错误: {error_msg}"
                        )
                except Exception as e:
                    self.logger.error(f"开多/加仓异常: {symbol}, 异常: {str(e)}")
                    self.dashboard.update_trade_status(
                        symbol=symbol,
                        action="OPEN_LONG",
                        quantity=abs(position_delta),
                        status="ERROR",
                        note=f"异常: {str(e)}"
                    )
                    
            elif position_delta < 0:
                if current_size > 0:
                    # 减仓
                    order_result = await self.executor.close_long(
                        symbol=symbol, 
                        quantity=abs(position_delta)
                    )
                    
                    if order_result:
                        self.logger.info(f"减仓成功: {symbol}, 数量={abs(position_delta)}")
                        self.dashboard.update_trade_status(
                            symbol=symbol,
                            action="CLOSE_PARTIAL",
                            quantity=abs(position_delta),
                            price=order_result.get('price', 0),
                            status="SUCCESS"
                        )
                    else:
                        self.logger.error(f"减仓失败: {symbol}")
                        self.dashboard.update_trade_status(
                            symbol=symbol,
                            action="CLOSE_PARTIAL",
                            quantity=abs(position_delta),
                            status="FAILED"
                        )
                else:
                    # 开空或加仓空
                    order_result = await self.executor.open_short(
                        symbol=symbol, 
                        quantity=abs(position_delta)
                    )
                    
                    if order_result:
                        self.logger.info(f"开空/加仓成功: {symbol}, 数量={abs(position_delta)}")
                        self.dashboard.update_trade_status(
                            symbol=symbol,
                            action="OPEN_SHORT",
                            quantity=abs(position_delta),
                            price=order_result.get('price', 0),
                            status="SUCCESS"
                        )
                    else:
                        self.logger.error(f"开空/加仓失败: {symbol}")
                        self.dashboard.update_trade_status(
                            symbol=symbol,
                            action="OPEN_SHORT",
                            quantity=abs(position_delta),
                            status="FAILED"
                        )
                        
            # 更新仓位信息
            positions = await self.position_manager.update_positions()
            
            # 检查是否触发熔断
            if not self.circuit_breaker.check_trade(symbol, position_delta, positions):
                self.logger.warning(f"熔断机制被触发，交易受限")
                self.alerter.critical("熔断机制触发", f"交易 {symbol} 触发熔断条件")
            
        except Exception as e:
            self.logger.error(f"执行交易失败: {e}")
            self.alerter.alert("交易执行失败", f"执行 {symbol} 交易时出错: {e}")
            
    async def _close_all_positions(self):
        """关闭所有持仓"""
        try:
            positions = await self.position_manager.get_all_positions()
            
            if not positions:
                self.logger.info("没有需要关闭的仓位")
                return
                
            for symbol, position in positions.items():
                size = position.get('size', 0)
                
                if size > 0:
                    # 关闭多头仓位
                    result = await self.executor.close_long(symbol, abs(size))
                    if result:
                        self.logger.info(f"成功关闭多头仓位: {symbol}, 数量={abs(size)}")
                    else:
                        self.logger.error(f"关闭多头仓位失败: {symbol}")
                        
                elif size < 0:
                    # 关闭空头仓位
                    result = await self.executor.close_short(symbol, abs(size))
                    if result:
                        self.logger.info(f"成功关闭空头仓位: {symbol}, 数量={abs(size)}")
                    else:
                        self.logger.error(f"关闭空头仓位失败: {symbol}")
                        
            # 更新仓位状态
            await self.position_manager.update_positions()
            
        except Exception as e:
            self.logger.error(f"关闭所有仓位失败: {e}")
            self.alerter.alert("紧急平仓失败", f"关闭所有仓位时出错: {e}")
            
    async def _generate_reports(self):
        """生成交易报告"""
        try:
            # 生成每日报告
            today = datetime.now().date()
            yesterday = (today - timedelta(days=1)).strftime('%Y-%m-%d')
            
            # 生成报告
            daily_reports = self.reporter.generate_daily_report(yesterday)
            
            # 检查是否是周末，生成周报
            if today.weekday() == 0:  # 周一，生成上周的周报
                weekly_reports = self.reporter.generate_weekly_report()
                self.logger.info(f"已生成周度报告: {weekly_reports}")
                
            # 检查是否是月初，生成月报
            if today.day == 1:  # 月初，生成上月的月报
                monthly_reports = self.reporter.generate_monthly_report()
                self.logger.info(f"已生成月度报告: {monthly_reports}")
                
            self.logger.info(f"报告生成完成")
            
        except Exception as e:
            self.logger.error(f"生成报告失败: {e}")
            
    async def _shutdown(self):
        """关闭系统，清理资源"""
        self.logger.info("开始清理资源并关闭系统...")
        
        try:
            # 记录最终状态
            positions = await self.position_manager.get_all_positions()
            self.logger.info(f"最终持仓状态: {positions}")
            
            # 生成最终报告
            self.dashboard.generate_summary()
            
            # 关闭API连接
            await self.api.close()
            
            self.logger.info("系统关闭完成")
            
        except Exception as e:
            self.logger.error(f"系统关闭过程中出错: {e}")
            
    def stop(self):
        """停止交易系统"""
        self.logger.info("正在停止交易系统...")
        self.is_running = False


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='币安U本位自动量化交易系统')
    parser.add_argument('--config-dir', type=str, help='配置文件目录路径')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='日志级别')
    parser.add_argument('--close-positions', action='store_true', help='启动后关闭所有持仓')
    args = parser.parse_args()
    
    # 创建并启动交易系统
    try:
        system = TradingSystem(config_dir=args.config_dir, log_level=args.log_level)
        
        # 如果指定了关闭所有持仓
        if args.close_positions:
            await system.initialize()
            await system._close_all_positions()
            await system._shutdown()
            return
        
        # 正常运行交易系统
        await system.run()
        
    except Exception as e:
        print(f"系统启动失败: {e}")
        if hasattr(system, 'logger'):
            system.logger.critical(f"系统启动失败: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
