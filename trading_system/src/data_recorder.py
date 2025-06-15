"""
数据记录器模块 - 负责记录和保存交易系统的数据
"""
import os
import json
import csv
import logging
import threading
import time
from datetime import datetime, timedelta
import pandas as pd

class DataRecorder:
    """数据记录器，负责记录和保存交易系统的数据"""
    
    def __init__(self, config, base_dir=None):
        """
        初始化数据记录器
        
        参数:
        - config: 配置字典
        - base_dir: 数据保存基础目录
        """
        self.logger = logging.getLogger("DataRecorder")
        self.config = config
        self.base_dir = base_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        
        # 确保数据目录存在
        self._ensure_directories()
        
        # 数据缓存
        self.market_data_cache = []
        self.trade_data_cache = []
        self.position_data_cache = []
        self.order_data_cache = []
        self.system_status_cache = []
        self.alert_cache = []  # 新增警报缓存
        
        # 缓存大小
        self.market_cache_max_size = 1000
        self.trade_cache_max_size = 100
        self.position_cache_max_size = 500
        self.order_cache_max_size = 200
        self.system_cache_max_size = 100
        self.alert_cache_max_size = 100  # 设置警报缓存大小
        
        # 保存间隔（秒）
        self.save_interval = 300  # 默认5分钟
        
        # 状态
        self.is_recording = False
        self.recorder_thread = None
        
        # 统计信息
        self.stats = {
            'market_data_points': 0,
            'trade_records': 0,
            'position_records': 0,
            'order_records': 0,
            'system_status_records': 0,
            'alert_records': 0,  # 新增警报统计
            'last_save_time': None,
            'saves_count': 0
        }
        
        # 加载配置
        self._load_config()
    
    def _ensure_directories(self):
        """确保所有必需的数据目录都存在"""
        try:
            # 创建基础目录
            os.makedirs(self.base_dir, exist_ok=True)
            
            # 创建子目录
            for subdir in ['market', 'trades', 'positions', 'orders', 'system', 'alerts', 'predictions']:
                os.makedirs(os.path.join(self.base_dir, subdir), exist_ok=True)
            
            self.logger.debug(f"数据目录已创建: {self.base_dir}")
            return True
        except Exception as e:
            self.logger.error(f"创建数据目录失败: {e}")
            return False
    
    def _load_config(self):
        """加载配置"""
        try:
            system_config = self.config.get('system', {})
            
            # 加载数据保存间隔
            if 'data_save_interval' in system_config:
                self.save_interval = system_config['data_save_interval']
            
            # 加载缓存大小限制
            if 'market_cache_size' in system_config:
                self.market_cache_max_size = system_config['market_cache_size']
            
            if 'trade_cache_size' in system_config:
                self.trade_cache_max_size = system_config['trade_cache_size']
            
            if 'position_cache_size' in system_config:
                self.position_cache_max_size = system_config['position_cache_size']
            
            if 'order_cache_size' in system_config:
                self.order_cache_max_size = system_config['order_cache_size']
            
            if 'system_cache_size' in system_config:
                self.system_cache_max_size = system_config['system_cache_size']
                
            if 'alert_cache_size' in system_config:
                self.alert_cache_max_size = system_config['alert_cache_size']
            
            self.logger.debug("数据记录器配置已加载")
        except Exception as e:
            self.logger.error(f"加载数据记录器配置失败: {e}")
    
    def start_recording(self):
        """启动数据记录线程"""
        if self.is_recording:
            return False
            
        self.is_recording = True
        self.logger.info(f"启动数据记录，保存间隔: {self.save_interval}秒")
        
        def recording_loop():
            while self.is_recording:
                try:
                    time.sleep(self.save_interval)
                    self.save_all_data()
                except Exception as e:
                    self.logger.error(f"数据记录错误: {e}")
                    time.sleep(5)
        
        # 启动记录线程
        self.recorder_thread = threading.Thread(target=recording_loop)
        self.recorder_thread.daemon = True
        self.recorder_thread.start()
        
        return True
    
    def stop_recording(self):
        """停止数据记录线程"""
        if not self.is_recording:
            return False
            
        self.is_recording = False
        self.logger.info("数据记录已停止")
        
        # 保存所有缓存的数据
        self.save_all_data()
        
        return True
    
    def record_market_data(self, data):
        """
        记录市场数据
        
        参数:
        - data: 市场数据字典
        """
        try:
            # 确保有时间戳
            if 'timestamp' not in data:
                data['timestamp'] = datetime.now().isoformat()
            
            # 添加到缓存
            self.market_data_cache.append(data)
            self.stats['market_data_points'] += 1
            
            # 如果缓存超过最大大小，保存并清空
            if len(self.market_data_cache) >= self.market_cache_max_size:
                self.save_market_data()
                
            return True
        except Exception as e:
            self.logger.error(f"记录市场数据失败: {e}")
            return False
    
    def record_trade(self, trade_data):
        """
        记录交易数据
        
        参数:
        - trade_data: 交易数据字典
        """
        try:
            # 确保有时间戳
            if 'timestamp' not in trade_data:
                trade_data['timestamp'] = datetime.now().isoformat()
            
            # 添加到缓存
            self.trade_data_cache.append(trade_data)
            self.stats['trade_records'] += 1
            
            # 如果缓存超过最大大小，保存并清空
            if len(self.trade_data_cache) >= self.trade_cache_max_size:
                self.save_trade_data()
                
            return True
        except Exception as e:
            self.logger.error(f"记录交易数据失败: {e}")
            return False
    
    def record_position(self, position_data):
        """
        记录仓位数据
        
        参数:
        - position_data: 仓位数据字典
        """
        try:
            # 确保有时间戳
            if 'timestamp' not in position_data:
                position_data['timestamp'] = datetime.now().isoformat()
            
            # 添加到缓存
            self.position_data_cache.append(position_data)
            self.stats['position_records'] += 1
            
            # 如果缓存超过最大大小，保存并清空
            if len(self.position_data_cache) >= self.position_cache_max_size:
                self.save_position_data()
                
            return True
        except Exception as e:
            self.logger.error(f"记录仓位数据失败: {e}")
            return False
    
    def record_order(self, order_data):
        """
        记录订单数据
        
        参数:
        - order_data: 订单数据字典
        """
        try:
            # 确保有时间戳
            if 'timestamp' not in order_data:
                order_data['timestamp'] = datetime.now().isoformat()
            
            # 添加到缓存
            self.order_data_cache.append(order_data)
            self.stats['order_records'] += 1
            
            # 如果缓存超过最大大小，保存并清空
            if len(self.order_data_cache) >= self.order_cache_max_size:
                self.save_order_data()
                
            return True
        except Exception as e:
            self.logger.error(f"记录订单数据失败: {e}")
            return False
    
    def record_system_status(self, status_data):
        """
        记录系统状态数据
        
        参数:
        - status_data: 系统状态数据字典
        """
        try:
            # 确保有时间戳
            if 'timestamp' not in status_data:
                status_data['timestamp'] = datetime.now().isoformat()
            
            # 添加到缓存
            self.system_status_cache.append(status_data)
            self.stats['system_status_records'] += 1
            
            # 如果缓存超过最大大小，保存并清空
            if len(self.system_status_cache) >= self.system_cache_max_size:
                self.save_system_status()
                
            return True
        except Exception as e:
            self.logger.error(f"记录系统状态数据失败: {e}")
            return False
    
    def record_alert(self, alert_data):
        """
        记录系统警报
        
        参数:
        - alert_data: 警报数据字典，应包含警报级别、信息等内容
        
        返回:
        - 布尔值，表示操作是否成功
        """
        try:
            # 确保有时间戳
            if 'timestamp' not in alert_data:
                alert_data['timestamp'] = datetime.now().isoformat()
                
            # 确保有警报级别
            if 'level' not in alert_data:
                alert_data['level'] = 'INFO'  # 默认级别
                
            # 确保有警报来源
            if 'source' not in alert_data:
                alert_data['source'] = 'SYSTEM'
                
            # 添加到缓存
            self.alert_cache.append(alert_data)
            self.stats['alert_records'] += 1
            
            # 根据警报级别记录到日志
            log_message = f"[{alert_data['level']}] {alert_data.get('message', 'No message')}"
            if alert_data['level'] == 'CRITICAL':
                self.logger.critical(log_message)
            elif alert_data['level'] == 'ERROR':
                self.logger.error(log_message)
            elif alert_data['level'] == 'WARNING':
                self.logger.warning(log_message)
            else:
                self.logger.info(log_message)
            
            # 如果缓存超过最大大小，保存并清空
            if len(self.alert_cache) >= self.alert_cache_max_size:
                self.save_alert_data()
                
            return True
        except Exception as e:
            self.logger.error(f"记录警报数据失败: {e}")
            return False
    
    def save_market_data(self):
        """保存市场数据"""
        if not self.market_data_cache:
            return True
            
        try:
            # 获取当前日期作为文件名一部分
            date_str = datetime.now().strftime("%Y%m%d")
            symbol = self.config['general']['symbol']
            filename = f"{symbol}_market_{date_str}.csv"
            filepath = os.path.join(self.base_dir, "market", filename)
            
            # 检查文件是否已存在
            file_exists = os.path.exists(filepath)
            
            # 转换为DataFrame并保存
            df = pd.DataFrame(self.market_data_cache)
            
            # 写入文件，如果文件已存在则追加，否则创建新文件
            mode = 'a' if file_exists else 'w'
            header = not file_exists
            df.to_csv(filepath, mode=mode, index=False, header=header)
            
            self.logger.debug(f"已保存{len(self.market_data_cache)}条市场数据到 {filepath}")
            
            # 清空缓存
            self.market_data_cache = []
            
            return True
        except Exception as e:
            self.logger.error(f"保存市场数据失败: {e}")
            return False
    
    def save_trade_data(self):
        """保存交易数据"""
        if not self.trade_data_cache:
            return True
            
        try:
            # 获取当前日期作为文件名一部分
            date_str = datetime.now().strftime("%Y%m%d")
            symbol = self.config['general']['symbol']
            filename = f"{symbol}_trades_{date_str}.json"
            filepath = os.path.join(self.base_dir, "trades", filename)
            
            # 检查文件是否已存在
            existing_data = []
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    existing_data = json.load(f)
            
            # 合并现有数据和新数据
            all_data = existing_data + self.trade_data_cache
            
            # 保存到文件
            with open(filepath, 'w') as f:
                json.dump(all_data, f, indent=2)
            
            self.logger.debug(f"已保存{len(self.trade_data_cache)}条交易记录到 {filepath}")
            
            # 清空缓存
            self.trade_data_cache = []
            
            return True
        except Exception as e:
            self.logger.error(f"保存交易数据失败: {e}")
            return False
    
    def save_position_data(self):
        """保存仓位数据"""
        if not self.position_data_cache:
            return True
            
        try:
            # 获取当前日期作为文件名一部分
            date_str = datetime.now().strftime("%Y%m%d")
            symbol = self.config['general']['symbol']
            filename = f"{symbol}_positions_{date_str}.csv"
            filepath = os.path.join(self.base_dir, "positions", filename)
            
            # 检查文件是否已存在
            file_exists = os.path.exists(filepath)
            
            # 转换为DataFrame并保存
            df = pd.DataFrame(self.position_data_cache)
            
            # 写入文件，如果文件已存在则追加，否则创建新文件
            mode = 'a' if file_exists else 'w'
            header = not file_exists
            df.to_csv(filepath, mode=mode, index=False, header=header)
            
            self.logger.debug(f"已保存{len(self.position_data_cache)}条仓位记录到 {filepath}")
            
            # 清空缓存
            self.position_data_cache = []
            
            return True
        except Exception as e:
            self.logger.error(f"保存仓位数据失败: {e}")
            return False
    
    def record_model_prediction(self, prediction):
        """记录模型预测结果"""
        try:
            if not hasattr(self, 'prediction_cache'):
                self.prediction_cache = []
                
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            prediction_record = {
                'timestamp': timestamp,
                'action': prediction.get('action', 'UNKNOWN'),
                'confidence': prediction.get('confidence', 0.0),
                'details': prediction
            }
            
            self.prediction_cache.append(prediction_record)
            
            # 如果缓存太大，保存
            if len(self.prediction_cache) >= 100:
                self._save_prediction_data()
                
            return True
        except Exception as e:
            self.logger.error(f"记录模型预测失败: {e}")
            return False
    
    def _save_prediction_data(self):
        """保存模型预测数据"""
        if not hasattr(self, 'prediction_cache') or not self.prediction_cache:
            return True
            
        try:
            # 获取当前日期作为文件名一部分
            date_str = datetime.now().strftime("%Y%m%d")
            filename = f"predictions_{date_str}.csv"
            
            # 构建保存路径
            predictions_dir = os.path.join(self.base_dir, "predictions")
            os.makedirs(predictions_dir, exist_ok=True)
            filepath = os.path.join(predictions_dir, filename)
            
            # 检查文件是否已存在
            file_exists = os.path.exists(filepath)
            
            # 转换为DataFrame并保存
            df = pd.DataFrame(self.prediction_cache)
            
            # 写入文件，如果文件已存在则追加，否则创建新文件
            mode = 'a' if file_exists else 'w'
            header = not file_exists
            df.to_csv(filepath, mode=mode, index=False, header=header)
            
            self.logger.debug(f"已保存{len(self.prediction_cache)}条预测记录到 {filepath}")
            
            # 清空缓存
            self.prediction_cache = []
            
            return True
        except Exception as e:
            self.logger.error(f"保存预测数据失败: {e}")
            return False
    
    def save_order_data(self):
        """保存订单数据"""
        if not self.order_data_cache:
            return True
            
        try:
            # 获取当前日期作为文件名一部分
            date_str = datetime.now().strftime("%Y%m%d")
            symbol = self.config['general']['symbol']
            filename = f"{symbol}_orders_{date_str}.json"
            filepath = os.path.join(self.base_dir, "orders", filename)
            
            # 检查文件是否已存在
            existing_data = []
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    existing_data = json.load(f)
            
            # 合并现有数据和新数据
            all_data = existing_data + self.order_data_cache
            
            # 保存到文件
            with open(filepath, 'w') as f:
                json.dump(all_data, f, indent=2)
            
            self.logger.debug(f"已保存{len(self.order_data_cache)}条订单记录到 {filepath}")
            
            # 清空缓存
            self.order_data_cache = []
            
            return True
        except Exception as e:
            self.logger.error(f"保存订单数据失败: {e}")
            return False
    
    def save_system_status(self):
        """保存系统状态数据"""
        if not self.system_status_cache:
            return True
            
        try:
            # 获取当前日期作为文件名一部分
            date_str = datetime.now().strftime("%Y%m%d")
            filename = f"system_status_{date_str}.csv"
            filepath = os.path.join(self.base_dir, "system", filename)
            
            # 检查文件是否已存在
            file_exists = os.path.exists(filepath)
            
            # 转换为DataFrame并保存
            df = pd.DataFrame(self.system_status_cache)
            
            # 写入文件，如果文件已存在则追加，否则创建新文件
            mode = 'a' if file_exists else 'w'
            header = not file_exists
            df.to_csv(filepath, mode=mode, index=False, header=header)
            
            self.logger.debug(f"已保存{len(self.system_status_cache)}条系统状态记录到 {filepath}")
            
            # 清空缓存
            self.system_status_cache = []
            
            return True
        except Exception as e:
            self.logger.error(f"保存系统状态数据失败: {e}")
            return False
    
    def save_alert_data(self):
        """保存警报数据"""
        if not self.alert_cache:
            return True
            
        try:
            # 获取当前日期作为文件名一部分
            date_str = datetime.now().strftime("%Y%m%d")
            filename = f"alerts_{date_str}.json"
            
            # 创建警报目录
            alerts_dir = os.path.join(self.base_dir, "alerts")
            os.makedirs(alerts_dir, exist_ok=True)
            filepath = os.path.join(alerts_dir, filename)
            
            # 检查文件是否已存在
            existing_data = []
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    existing_data = json.load(f)
            
            # 合并现有数据和新数据
            all_data = existing_data + self.alert_cache
            
            # 保存到文件
            with open(filepath, 'w') as f:
                json.dump(all_data, f, indent=2)
            
            self.logger.debug(f"已保存{len(self.alert_cache)}条警报记录到 {filepath}")
            
            # 清空缓存
            self.alert_cache = []
            
            return True
        except Exception as e:
            self.logger.error(f"保存警报数据失败: {e}")
            return False
    
    def save_all_data(self):
        """保存所有缓存数据"""
        try:
            self.save_market_data()
            self.save_trade_data()
            self.save_position_data()
            self.save_order_data()
            self.save_system_status()
            self.save_alert_data()  # 保存警报数据
            
            # 更新统计信息
            self.stats['last_save_time'] = datetime.now().isoformat()
            self.stats['saves_count'] += 1
            
            self.logger.info("所有数据已保存")
            return True
        except Exception as e:
            self.logger.error(f"保存所有数据失败: {e}")
            return False
    
    def load_market_data(self, days=1, symbol=None):
        """
        加载历史市场数据
        
        参数:
        - days: 加载最近几天的数据
        - symbol: 交易对符号
        
        返回:
        - DataFrame 格式的市场数据
        """
        try:
            symbol = symbol or self.config['general']['symbol']
            
            # 计算日期范围
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            date_range = []
            current_date = start_date
            while current_date <= end_date:
                date_range.append(current_date.strftime("%Y%m%d"))
                current_date += timedelta(days=1)
            
            # 加载所有日期的数据
            all_data = []
            for date_str in date_range:
                filename = f"{symbol}_market_{date_str}.csv"
                filepath = os.path.join(self.base_dir, "market", filename)
                
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath)
                    all_data.append(df)
            
            # 合并所有数据
            if all_data:
                result_df = pd.concat(all_data, ignore_index=True)
                
                # 如果有时间戳列，转换为datetime格式
                if 'timestamp' in result_df.columns:
                    result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
                    
                    # 对数据进行排序
                    result_df = result_df.sort_values('timestamp')
                
                self.logger.debug(f"已加载{len(result_df)}条历史市场数据")
                return result_df
            else:
                self.logger.warning(f"未找到{symbol}的历史市场数据")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"加载历史市场数据失败: {e}")
            return pd.DataFrame()
    
    def load_trade_data(self, days=7, symbol=None):
        """
        加载历史交易数据
        
        参数:
        - days: 加载最近几天的数据
        - symbol: 交易对符号
        
        返回:
        - 交易数据列表
        """
        try:
            symbol = symbol or self.config['general']['symbol']
            
            # 计算日期范围
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            date_range = []
            current_date = start_date
            while current_date <= end_date:
                date_range.append(current_date.strftime("%Y%m%d"))
                current_date += timedelta(days=1)
            
            # 加载所有日期的数据
            all_trades = []
            for date_str in date_range:
                filename = f"{symbol}_trades_{date_str}.json"
                filepath = os.path.join(self.base_dir, "trades", filename)
                
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        trades = json.load(f)
                        all_trades.extend(trades)
            
            self.logger.debug(f"已加载{len(all_trades)}条历史交易记录")
            return all_trades
                
        except Exception as e:
            self.logger.error(f"加载历史交易数据失败: {e}")
            return []
    
    def get_stats(self):
        """获取数据记录器统计信息"""
        return self.stats
    
    def export_data(self, data_type, start_date, end_date=None, format='csv'):
        """
        导出指定类型和日期范围的数据
        
        参数:
        - data_type: 数据类型 ('market', 'trades', 'positions', 'orders', 'system')
        - start_date: 开始日期 (YYYYMMDD)
        - end_date: 结束日期 (YYYYMMDD)，默认为当前日期
        - format: 导出格式 ('csv', 'json')
        
        返回:
        - 导出文件路径
        """
        try:
            # 设置默认结束日期为当前日期
            if not end_date:
                end_date = datetime.now().strftime("%Y%m%d")
            
            # 创建导出目录
            export_dir = os.path.join(self.base_dir, "exports")
            os.makedirs(export_dir, exist_ok=True)
            
            # 生成导出文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbol = self.config['general']['symbol']
            export_filename = f"{symbol}_{data_type}_{start_date}_{end_date}_{timestamp}.{format}"
            export_filepath = os.path.join(export_dir, export_filename)
            
            # 根据数据类型导出不同的数据
            if data_type == 'market':
                self._export_market_data(start_date, end_date, export_filepath, format)
            elif data_type == 'trades':
                self._export_trade_data(start_date, end_date, export_filepath, format)
            elif data_type == 'positions':
                self._export_position_data(start_date, end_date, export_filepath, format)
            elif data_type == 'orders':
                self._export_order_data(start_date, end_date, export_filepath, format)
            elif data_type == 'system':
                self._export_system_data(start_date, end_date, export_filepath, format)
            elif data_type == 'alerts':
                self._export_alert_data(start_date, end_date, export_filepath, format)
            else:
                self.logger.error(f"不支持的数据类型: {data_type}")
                return None
            
            self.logger.info(f"已导出{data_type}数据到 {export_filepath}")
            return export_filepath
            
        except Exception as e:
            self.logger.error(f"导出数据失败: {e}")
            return None
    
    def _export_market_data(self, start_date, end_date, export_filepath, format):
        """导出市场数据"""
        # 计算日期范围
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        
        date_range = []
        current_dt = start_dt
        while current_dt <= end_dt:
            date_range.append(current_dt.strftime("%Y%m%d"))
            current_dt += timedelta(days=1)
        
        # 加载数据
        all_data = []
        symbol = self.config['general']['symbol']
        
        for date_str in date_range:
            filename = f"{symbol}_market_{date_str}.csv"
            filepath = os.path.join(self.base_dir, "market", filename)
            
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                all_data.append(df)
        
        # 合并数据并导出
        if all_data:
            result_df = pd.concat(all_data, ignore_index=True)
            
            if format == 'csv':
                result_df.to_csv(export_filepath, index=False)
            elif format == 'json':
                result_df.to_json(export_filepath, orient='records', indent=2)
            
            return True
        else:
            self.logger.warning(f"未找到在{start_date}至{end_date}之间的市场数据")
            return False
    
    def _export_trade_data(self, start_date, end_date, export_filepath, format):
        """导出交易数据"""
        # 类似实现...省略代码
        pass
    
    def _export_position_data(self, start_date, end_date, export_filepath, format):
        """导出仓位数据"""
        # 类似实现...省略代码
        pass
    
    def _export_order_data(self, start_date, end_date, export_filepath, format):
        """导出订单数据"""
        # 类似实现...省略代码
        pass
    
    def _export_system_data(self, start_date, end_date, export_filepath, format):
        """导出系统数据"""
        # 类似实现...省略代码
        pass
    
    def _export_alert_data(self, start_date, end_date, export_filepath, format):
        """导出警报数据"""
        # 计算日期范围
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        
        date_range = []
        current_dt = start_dt
        while current_dt <= end_dt:
            date_range.append(current_dt.strftime("%Y%m%d"))
            current_dt += timedelta(days=1)
        
        # 加载数据
        all_alerts = []
        
        for date_str in date_range:
            filename = f"alerts_{date_str}.json"
            filepath = os.path.join(self.base_dir, "alerts", filename)
            
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    alerts = json.load(f)
                    all_alerts.extend(alerts)
        
        # 导出数据
        if all_alerts:
            if format == 'csv':
                # 转换为DataFrame
                df = pd.DataFrame(all_alerts)
                df.to_csv(export_filepath, index=False)
            elif format == 'json':
                with open(export_filepath, 'w') as f:
                    json.dump(all_alerts, f, indent=2)
            
            return True
        else:
            self.logger.warning(f"未找到在{start_date}至{end_date}之间的警报数据")
            return False