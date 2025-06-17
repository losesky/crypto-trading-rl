import smtplib
import logging
import json
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import yaml
import os

class Alerter:
    """
    告警系统类，负责生成各类通知并通过配置的渠道发送
    支持邮件、Telegram、日志等多种告警通道
    """
    
    def __init__(self, config_path=None):
        """
        初始化告警系统
        
        Args:
            config_path: 配置文件路径，默认为None（使用默认配置文件路径）
        """
        self.logger = logging.getLogger(__name__)
        
        # 如果未提供配置文件路径，使用默认路径
        if config_path is None:
            config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
            config_path = os.path.join(config_dir, 'alert_config.yaml')
        
        # 加载配置
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            self.logger.info("告警配置加载成功")
        except Exception as e:
            self.logger.error(f"加载告警配置失败: {e}")
            # 设置默认配置
            self.config = {
                'enabled': True,
                'channels': {
                    'email': {
                        'enabled': False,
                    },
                    'telegram': {
                        'enabled': False,
                    },
                    'log': {
                        'enabled': True,
                        'level': 'WARNING'
                    }
                },
                'levels': ['INFO', 'WARNING', 'ALERT', 'CRITICAL', 'EMERGENCY'],
                'throttle': {
                    'max_alerts_per_hour': 10,
                    'cooldown_period_minutes': 10
                }
            }
            
        # 初始化告警计数器和最近告警时间
        self.alert_count = 0
        self.last_alert_time = datetime.now()
        
    def _should_throttle(self, level):
        """检查是否应该节流告警"""
        now = datetime.now()
        hour_diff = (now - self.last_alert_time).total_seconds() / 3600
        
        # 如果超过一小时，重置计数器
        if hour_diff >= 1:
            self.alert_count = 0
            self.last_alert_time = now
            return False
        
        # 如果在冷却期内，对于非紧急告警进行节流
        if level not in ['CRITICAL', 'EMERGENCY']:
            cooldown_minutes = self.config.get('throttle', {}).get('cooldown_period_minutes', 10)
            minutes_diff = (now - self.last_alert_time).total_seconds() / 60
            if minutes_diff < cooldown_minutes:
                return True
        
        # 检查每小时最大告警数
        max_alerts = self.config.get('throttle', {}).get('max_alerts_per_hour', 10)
        if self.alert_count >= max_alerts:
            # 记录被节流的告警
            self.logger.warning(f"告警被节流: 在过去一小时内已发送{self.alert_count}条告警")
            return True
            
        # 更新计数器
        self.alert_count += 1
        self.last_alert_time = now
        return False
    
    def _send_email(self, subject, message):
        """通过电子邮件发送告警"""
        email_config = self.config.get('channels', {}).get('email', {})
        if not email_config.get('enabled', False):
            return False
            
        try:
            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = email_config.get('sender')
            msg['To'] = ', '.join(email_config.get('recipients', []))
            msg['Subject'] = f"[Trading Alert] {subject}"
            
            # 添加正文
            msg.attach(MIMEText(message, 'plain'))
            
            # 发送邮件
            server = smtplib.SMTP(email_config.get('smtp_server'), email_config.get('smtp_port', 587))
            server.starttls()  # 启用TLS加密
            server.login(email_config.get('username'), email_config.get('password'))
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"邮件告警发送成功: {subject}")
            return True
        except Exception as e:
            self.logger.error(f"发送邮件告警失败: {e}")
            return False
            
    def _send_telegram(self, message):
        """通过Telegram发送告警"""
        telegram_config = self.config.get('channels', {}).get('telegram', {})
        if not telegram_config.get('enabled', False):
            return False
            
        try:
            bot_token = telegram_config.get('bot_token')
            chat_id = telegram_config.get('chat_id')
            
            # 发送消息
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            self.logger.info(f"Telegram告警发送成功")
            return True
        except Exception as e:
            self.logger.error(f"发送Telegram告警失败: {e}")
            return False
    
    def _log_alert(self, level, message):
        """记录告警到日志系统"""
        log_config = self.config.get('channels', {}).get('log', {})
        if not log_config.get('enabled', True):
            return False
            
        # 根据告警级别确定日志级别
        if level == 'INFO':
            self.logger.info(message)
        elif level == 'WARNING':
            self.logger.warning(message)
        elif level in ['ALERT', 'CRITICAL']:
            self.logger.error(message)
        elif level == 'EMERGENCY':
            self.logger.critical(message)
        else:
            self.logger.warning(message)
        
        return True
        
    def send_alert(self, level, title, message, details=None, notify_channels=None):
        """
        发送告警通知
        
        Args:
            level: 告警级别（INFO, WARNING, ALERT, CRITICAL, EMERGENCY）
            title: 告警标题
            message: 告警消息
            details: 附加详情（字典格式）
            notify_channels: 指定通知渠道列表，如果为None则使用配置中启用的所有渠道
            
        Returns:
            bool: 是否成功发送至少一个渠道
        """
        if not self.config.get('enabled', True):
            return False
            
        # 检查告警级别是否合法
        valid_levels = self.config.get('levels', ['INFO', 'WARNING', 'ALERT', 'CRITICAL', 'EMERGENCY'])
        if level not in valid_levels:
            level = 'INFO'  # 默认使用INFO级别
            
        # 检查是否应该节流告警
        if self._should_throttle(level):
            return False
            
        # 格式化告警时间
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 构建完整的告警消息
        full_message = f"[{level}] {title}\n时间: {timestamp}\n\n{message}"
        if details:
            try:
                full_message += f"\n\n详情:\n{json.dumps(details, indent=2, ensure_ascii=False)}"
            except:
                full_message += f"\n\n详情: {str(details)}"
                
        # 确定要发送的渠道
        channels = notify_channels or []
        if not channels:
            channels_config = self.config.get('channels', {})
            if channels_config.get('email', {}).get('enabled', False):
                channels.append('email')
            if channels_config.get('telegram', {}).get('enabled', False):
                channels.append('telegram')
            channels.append('log')  # 日志总是作为备选渠道
                
        # 发送告警到各个渠道
        success = False
        
        for channel in channels:
            if channel == 'email':
                if self._send_email(title, full_message):
                    success = True
            elif channel == 'telegram':
                if self._send_telegram(full_message):
                    success = True
            elif channel == 'log':
                if self._log_alert(level, full_message):
                    success = True
        
        return success
        
    def info(self, title, message, details=None, notify_channels=None):
        """发送信息级别告警"""
        return self.send_alert('INFO', title, message, details, notify_channels)
        
    def warning(self, title, message, details=None, notify_channels=None):
        """发送警告级别告警"""
        return self.send_alert('WARNING', title, message, details, notify_channels)
        
    def alert(self, title, message, details=None, notify_channels=None):
        """发送一般告警级别告警"""
        return self.send_alert('ALERT', title, message, details, notify_channels)
        
    def critical(self, title, message, details=None, notify_channels=None):
        """发送严重告警级别告警"""
        return self.send_alert('CRITICAL', title, message, details, notify_channels)
        
    def emergency(self, title, message, details=None, notify_channels=None):
        """发送紧急告警级别告警"""
        return self.send_alert('EMERGENCY', title, message, details, notify_channels)
