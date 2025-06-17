import unittest
import sys
import os
from unittest.mock import patch, MagicMock
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入测试目标
from monitor.alerter import Alerter

class TestAlerter(unittest.TestCase):
    """告警器单元测试"""
    
    def setUp(self):
        """测试准备工作"""
        # 使用测试配置初始化告警器
        self.alerter = Alerter()
        # 模拟配置
        self.alerter.config = {
            'enabled': True,
            'channels': {
                'email': {'enabled': False},
                'telegram': {'enabled': False},
                'log': {'enabled': True}
            },
            'levels': ['INFO', 'WARNING', 'ALERT', 'CRITICAL', 'EMERGENCY'],
            'throttle': {
                'max_alerts_per_hour': 10,
                'cooldown_period_minutes': 5
            }
        }
        
    def test_initialization(self):
        """测试初始化"""
        self.assertTrue(self.alerter._initialized)
        self.assertEqual(self.alerter.alert_count, 0)
        self.assertIsInstance(self.alerter.last_alert_time, datetime)
        
    @patch('monitor.alerter.Alerter._send_email')
    @patch('monitor.alerter.Alerter._send_telegram')
    @patch('monitor.alerter.Alerter._log_alert')
    def test_send_alert(self, mock_log, mock_telegram, mock_email):
        """测试发送告警"""
        # 设置模拟返回值
        mock_log.return_value = True
        mock_telegram.return_value = False
        mock_email.return_value = False
        
        # 测试发送普通告警
        result = self.alerter.send_alert(
            level='WARNING', 
            title='测试告警', 
            message='这是一条测试告警信息', 
            details={'test': 'data'}
        )
        
        # 验证调用
        mock_log.assert_called_once()
        mock_telegram.assert_not_called()
        mock_email.assert_not_called()
        self.assertTrue(result)
        self.assertEqual(self.alerter.alert_count, 1)
        
    def test_throttling(self):
        """测试告警节流"""
        # 模拟_log_alert方法
        self.alerter._log_alert = MagicMock(return_value=True)
        
        # 发送超过最大限制的告警
        for i in range(15):  # 超过配置的10条/小时
            result = self.alerter.send_alert('INFO', f'测试{i}', '内容')
            if i < 10:
                self.assertTrue(result)
            else:
                self.assertFalse(result)
                
        # 验证告警计数
        self.assertEqual(self.alerter.alert_count, 10)
        
    def test_alert_level_methods(self):
        """测试各个告警级别方法"""
        # 模拟send_alert方法
        self.alerter.send_alert = MagicMock(return_value=True)
        
        # 测试各个级别的便捷方法
        self.alerter.info('信息', '这是信息')
        self.alerter.send_alert.assert_called_with('INFO', '信息', '这是信息', None, None)
        
        self.alerter.warning('警告', '这是警告')
        self.alerter.send_alert.assert_called_with('WARNING', '警告', '这是警告', None, None)
        
        self.alerter.alert('告警', '这是告警')
        self.alerter.send_alert.assert_called_with('ALERT', '告警', '这是告警', None, None)
        
        self.alerter.critical('严重', '这是严重告警')
        self.alerter.send_alert.assert_called_with('CRITICAL', '严重', '这是严重告警', None, None)
        
        self.alerter.emergency('紧急', '这是紧急告警')
        self.alerter.send_alert.assert_called_with('EMERGENCY', '紧急', '这是紧急告警', None, None)

if __name__ == '__main__':
    unittest.main()
