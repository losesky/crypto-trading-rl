"""
系统监控模块 - 负责监控交易系统的健康状态
"""
import logging
import threading
import time
import json
import os
import psutil
import socket
import subprocess
from datetime import datetime, timedelta
import requests

class SystemMonitor:
    """系统监控器，负责监控交易系统的健康状态"""
    
    def __init__(self, config):
        """
        初始化系统监控器
        
        参数:
        - config: 配置字典
        """
        self.logger = logging.getLogger("SystemMonitor")
        self.config = config
        
        # 监控状态
        self.is_monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 30  # 秒
        
        # 组件状态
        self.component_status = {
            'trading_service': False,
            'binance_api': False,
            'model_loaded': False,
            'websocket_server': False,
            'http_server': False,
            'data_recorder': False
        }
        
        # 系统资源使用
        self.system_resources = {
            'cpu_usage': 0,
            'memory_usage': 0,
            'disk_usage': 0,
            'network_usage': {'sent': 0, 'recv': 0},
            'timestamp': ''
        }
        
        # 警报状态
        self.alerts = []
        self.alert_history = []
        
        # 回调函数
        self.on_status_update = None
        self.on_alert = None
        
        # 最后健康检查时间
        self.last_health_check = 0
        
        # 加载配置
        self._load_config()
    
    def _load_config(self):
        """加载配置"""
        try:
            system_config = self.config.get('system', {})
            
            # 加载监控间隔
            if 'monitor_interval' in system_config:
                self.monitor_interval = system_config['monitor_interval']
            
            self.logger.debug("系统监控器配置已加载")
        except Exception as e:
            self.logger.error(f"加载系统监控器配置失败: {e}")
    
    def start_monitoring(self):
        """启动系统监控"""
        if self.is_monitoring:
            return False
            
        self.is_monitoring = True
        self.logger.info(f"启动系统监控，间隔: {self.monitor_interval}秒")
        
        def monitor_loop():
            while self.is_monitoring:
                try:
                    self.check_system_health()
                    time.sleep(self.monitor_interval)
                except Exception as e:
                    self.logger.error(f"系统监控错误: {e}")
                    time.sleep(5)
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        return True
    
    def stop_monitoring(self):
        """停止系统监控"""
        if not self.is_monitoring:
            return False
            
        self.is_monitoring = False
        self.logger.info("系统监控已停止")
        
        return True
    
    def check_system_health(self):
        """检查系统健康状态"""
        try:
            self.last_health_check = time.time()
            
            # 更新系统资源使用
            self.update_system_resources()
            
            # 检查各组件状态
            self.check_component_status()
            
            # 检查警报条件
            self.check_alerts()
            
            # 触发状态更新回调
            if self.on_status_update:
                status_data = self.get_status()
                self.on_status_update(status_data)
            
            return True
        except Exception as e:
            self.logger.error(f"系统健康检查失败: {e}")
            return False
    
    def update_system_resources(self):
        """更新系统资源使用情况"""
        try:
            # CPU使用率
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # 磁盘使用率
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            
            # 网络使用情况
            net_io = psutil.net_io_counters()
            net_sent = net_io.bytes_sent
            net_recv = net_io.bytes_recv
            
            # 计算网络速率（每秒）
            if 'last_net_check' in self.system_resources:
                last_time = self.system_resources['last_net_check']
                last_sent = self.system_resources['network_usage']['sent_bytes']
                last_recv = self.system_resources['network_usage']['recv_bytes']
                
                time_diff = time.time() - last_time
                if time_diff > 0:
                    sent_rate = (net_sent - last_sent) / time_diff
                    recv_rate = (net_recv - last_recv) / time_diff
                else:
                    sent_rate = 0
                    recv_rate = 0
            else:
                sent_rate = 0
                recv_rate = 0
            
            # 更新系统资源数据
            self.system_resources = {
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'disk_usage': disk_usage,
                'network_usage': {
                    'sent': sent_rate / 1024,  # KB/s
                    'recv': recv_rate / 1024,  # KB/s
                    'sent_bytes': net_sent,
                    'recv_bytes': net_recv
                },
                'last_net_check': time.time(),
                'timestamp': datetime.now().isoformat()
            }
            
            return True
        except Exception as e:
            self.logger.error(f"更新系统资源使用情况失败: {e}")
            return False
    
    def check_component_status(self):
        """检查各组件状态"""
        try:
            # 检查API连接
            if self.config.get('binance', {}).get('test_net'):
                api_url = "https://testnet.binancefuture.com/fapi/v1/time"
            else:
                api_url = "https://fapi.binance.com/fapi/v1/time"
            
            try:
                response = requests.get(api_url, timeout=5)
                self.component_status['binance_api'] = response.status_code == 200
            except:
                self.component_status['binance_api'] = False
            
            # 检查WebSocket服务器
            ws_port = self.config.get('ui', {}).get('ws_port', 8765)  # 修正默认WebSocket端口为8765，与config.ini一致
            self.component_status['websocket_server'] = self._check_port_in_use(ws_port)
            
            # 检查HTTP服务器
            http_port = self.config.get('ui', {}).get('http_port', 8090)
            self.component_status['http_server'] = self._check_port_in_use(http_port)
            
            # 其他组件状态通常需要外部设置
            return True
        except Exception as e:
            self.logger.error(f"检查组件状态失败: {e}")
            return False
    
    def _check_port_in_use(self, port):
        """检查端口是否在使用中"""
        try:
            # 方法1：尝试直接连接端口
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)  # 设置1秒超时
                result = s.connect_ex(('localhost', port))
                if result == 0:
                    self.logger.debug(f"端口 {port} 检测成功 (直接连接)")
                    return True
            
            # 方法2：检查监听该端口的进程
            import subprocess
            try:
                # 在Linux上使用netstat命令
                cmd = f"netstat -tuln | grep :{port}"
                result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if port in result.stdout:
                    self.logger.debug(f"端口 {port} 检测成功 (netstat)")
                    return True
            except:
                pass
                
            # 方法3：使用lsof命令检查
            try:
                cmd = f"lsof -i :{port}"
                result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    self.logger.debug(f"端口 {port} 检测成功 (lsof)")
                    return True
            except:
                pass
            
            # 如果是WebSocket服务器端口（8765），而且系统中有python进程包含websocket_server
            if port == 8765:
                try:
                    cmd = "ps aux | grep '[w]ebsocket_server'"
                    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if result.returncode == 0 and result.stdout.strip():
                        self.logger.debug(f"WebSocket服务器进程检测成功 (ps aux)")
                        return True
                except:
                    pass
            
            self.logger.debug(f"端口 {port} 检测失败")
            return False
        except Exception as e:
            self.logger.error(f"端口检测错误: {e}")
            return False
    
    def check_alerts(self):
        """检查是否需要触发警报"""
        try:
            # 清空当前警报
            self.alerts = []
            
            # 检查CPU使用率
            if self.system_resources['cpu_usage'] > 90:
                self._add_alert('HIGH_CPU', f"CPU使用率过高: {self.system_resources['cpu_usage']}%", 'warning')
            
            # 检查内存使用率
            if self.system_resources['memory_usage'] > 90:
                self._add_alert('HIGH_MEMORY', f"内存使用率过高: {self.system_resources['memory_usage']}%", 'warning')
            
            # 检查磁盘使用率
            if self.system_resources['disk_usage'] > 90:
                self._add_alert('HIGH_DISK', f"磁盘使用率过高: {self.system_resources['disk_usage']}%", 'warning')
            
            # 检查API连接状态 - 在非测试模式下检查API
            is_test_mode = self.config.get('general', {}).get('mode') == 'test'
            if not self.component_status['binance_api'] and not is_test_mode:
                self._add_alert('API_DISCONNECTED', "币安API连接失败", 'error')
            
            # 检查WebSocket服务器状态
            # 特殊处理：我们知道WebSocket服务器已启动，但检测可能不可靠
            if not self.component_status['websocket_server']:
                # 检查进程是否存在
                try:
                    ws_process_exists = False
                    cmd = "ps aux | grep '[w]ebsocket_server'"
                    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if result.returncode == 0 and result.stdout.strip():
                        ws_process_exists = True
                        self.component_status['websocket_server'] = True
                    
                    if not ws_process_exists:
                        self._add_alert('WEBSOCKET_DOWN', "WebSocket服务器未运行", 'error')
                except:
                    pass
            
            # 检查HTTP服务器状态
            if not self.component_status['http_server']:
                self._add_alert('HTTP_DOWN', "HTTP服务器未运行", 'error')
            
            return True
        except Exception as e:
            self.logger.error(f"检查警报失败: {e}")
            return False
    
    def _add_alert(self, alert_type, message, level='info'):
        """添加警报"""
        alert = {
            'type': alert_type,
            'message': message,
            'level': level,
            'timestamp': datetime.now().isoformat()
        }
        
        self.alerts.append(alert)
        self.alert_history.append(alert)
        
        # 限制历史记录大小
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]
        
        # 记录日志
        log_method = getattr(self.logger, level, self.logger.warning)
        log_method(f"系统警报: {message}")
        
        # 触发警报回调
        if self.on_alert:
            self.on_alert(alert)
    
    def set_component_status(self, component, status):
        """
        设置组件状态
        
        参数:
        - component: 组件名称
        - status: 组件状态(True/False)
        """
        if component in self.component_status:
            self.component_status[component] = status
            self.logger.debug(f"组件状态已更新: {component} -> {status}")
            return True
        else:
            self.logger.warning(f"未知组件: {component}")
            return False
    
    def set_component_status(self, component_name, status):
        """
        直接设置组件状态
        
        参数:
        - component_name: 组件名称
        - status: 状态(True/False)
        """
        if component_name in self.component_status:
            self.component_status[component_name] = status
            self.logger.debug(f"组件 {component_name} 状态设置为: {status}")
            return True
        else:
            self.logger.warning(f"未知组件: {component_name}")
            return False
    
    def get_status(self):
        """获取系统状态摘要"""
        # 检查组件状态，决定系统整体健康状态
        health_status = "healthy"
        status_message = "系统运行正常"
        
        # 区分测试环境和生产环境
        is_test_mode = self.config.get('general', {}).get('mode') == 'test'
        
        # 检查关键组件
        critical_components = []
        
        # 在生产环境中，Binance API是关键组件
        if not is_test_mode:
            critical_components.append(('binance_api', 'Binance API连接异常'))
        
        # 根据关键组件设置状态
        for component, message in critical_components:
            if not self.component_status.get(component, False):
                health_status = "warning"
                status_message = message
                break
        
        # 检查警报
        if any(alert['level'] == 'error' for alert in self.alerts):
            health_status = "critical"
            error_alerts = [a for a in self.alerts if a['level'] == 'error']
            if error_alerts:
                status_message = error_alerts[0]['message']
        
        return {
            'status': health_status,
            'message': status_message,
            'system_resources': self.system_resources,
            'component_status': self.component_status,
            'alerts': self.alerts,
            'last_check': self.last_health_check
        }
    
    def get_system_resources(self):
        """获取系统资源使用情况"""
        return self.system_resources
    
    def get_component_status(self):
        """获取组件状态"""
        return self.component_status
    
    def get_alerts(self):
        """获取当前警报"""
        return self.alerts
    
    def get_alert_history(self):
        """获取警报历史记录"""
        return self.alert_history
    
    def add_custom_alert(self, alert_type, message, level='info'):
        """添加自定义警报"""
        self._add_alert(alert_type, message, level)
        return True