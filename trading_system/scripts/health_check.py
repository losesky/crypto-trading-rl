#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
交易系统健康检查工具
用于诊断和解决系统连接问题
"""

import os
import sys
import json
import time
import logging
import argparse
import requests
import subprocess
import socket
from pathlib import Path
from datetime import datetime

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('HealthCheck')

# 获取项目根目录
SCRIPT_DIR = Path(__file__).parent.absolute()
TRADING_SYSTEM_DIR = SCRIPT_DIR.parent
ROOT_DIR = TRADING_SYSTEM_DIR.parent

class HealthChecker:
    """系统健康检查器"""
    
    def __init__(self, config_path=None, auto_fix=False):
        """初始化健康检查器"""
        self.auto_fix = auto_fix
        self.config = self._load_config(config_path)
        self.problems_found = []
        self.fixes_applied = []
        
        # 从配置中获取端口
        self.http_port = self.config['ui']['http_port']
        self.ws_port = self.config['ui']['ws_port']
        
    def _load_config(self, config_path):
        """加载配置文件"""
        if not config_path:
            config_path = TRADING_SYSTEM_DIR / "config" / "test_config.json"
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"已加载配置文件：{config_path}")
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败：{e}")
            # 使用默认配置
            return {
                "ui": {
                    "http_port": 8090,
                    "ws_port": 8095
                }
            }
            
    def check_port_availability(self, port):
        """检查端口是否可用"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            if result == 0:
                logger.info(f"端口 {port} 已被占用")
                return False
            else:
                logger.info(f"端口 {port} 可用")
                return True
        except Exception as e:
            logger.error(f"检查端口时出错: {e}")
            return False
            
    def check_process_running(self, process_name):
        """检查进程是否运行中"""
        try:
            cmd = f"ps aux | grep '{process_name}' | grep -v grep | wc -l"
            output = subprocess.check_output(cmd, shell=True).decode().strip()
            count = int(output)
            return count > 0
        except Exception as e:
            logger.error(f"检查进程时出错: {e}")
            return False
            
    def check_ui_server(self):
        """检查UI服务器是否运行"""
        try:
            response = requests.get(f"http://localhost:{self.http_port}", timeout=2)
            if response.status_code == 200:
                logger.info("UI服务器响应正常")
                return True
            else:
                logger.warning(f"UI服务器响应异常，状态码: {response.status_code}")
                self.problems_found.append(f"UI服务器响应异常，状态码: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            logger.warning("UI服务器未运行或无法访问")
            self.problems_found.append("UI服务器未运行或无法访问")
            return False
        except Exception as e:
            logger.error(f"检查UI服务器时出错: {e}")
            self.problems_found.append(f"检查UI服务器时出错: {e}")
            return False
            
    def check_websocket_proxy(self):
        """检查WebSocket代理是否运行"""
        try:
            # 使用HTTP端点检查WS代理的健康状态
            response = requests.get(f"http://localhost:{self.ws_port}/health", timeout=2)
            if response.status_code == 200:
                logger.info("WebSocket代理响应正常")
                return True
            else:
                logger.warning(f"WebSocket代理响应异常，状态码: {response.status_code}")
                self.problems_found.append(f"WebSocket代理响应异常，状态码: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            logger.warning("WebSocket代理未运行或无法访问")
            self.problems_found.append("WebSocket代理未运行或无法访问")
            return False
        except Exception as e:
            logger.error(f"检查WebSocket代理时出错: {e}")
            self.problems_found.append(f"检查WebSocket代理时出错: {e}")
            return False
    
    def fix_websocket_proxy(self):
        """修复WebSocket代理问题"""
        if self.check_port_availability(self.ws_port):
            logger.info(f"尝试启动WebSocket代理...")
            try:
                start_script = TRADING_SYSTEM_DIR / "scripts" / "start_optimized.sh"
                subprocess.Popen(f"bash {start_script} --ws-proxy", shell=True)
                time.sleep(2)  # 给点时间启动
                if self.check_websocket_proxy():
                    logger.info("WebSocket代理已成功启动")
                    self.fixes_applied.append("WebSocket代理已成功启动")
                    return True
                else:
                    logger.error("WebSocket代理启动失败")
                    return False
            except Exception as e:
                logger.error(f"启动WebSocket代理时出错: {e}")
                return False
        else:
            logger.warning(f"端口 {self.ws_port} 已被占用，无法启动WebSocket代理")
            return False
    
    def fix_ui_server(self):
        """修复UI服务器问题"""
        if self.check_port_availability(self.http_port):
            logger.info(f"尝试启动UI服务器...")
            try:
                start_script = TRADING_SYSTEM_DIR / "scripts" / "start_optimized.sh"
                subprocess.Popen(f"bash {start_script} --ui-only", shell=True)
                time.sleep(2)  # 给点时间启动
                if self.check_ui_server():
                    logger.info("UI服务器已成功启动")
                    self.fixes_applied.append("UI服务器已成功启动")
                    return True
                else:
                    logger.error("UI服务器启动失败")
                    return False
            except Exception as e:
                logger.error(f"启动UI服务器时出错: {e}")
                return False
        else:
            logger.warning(f"端口 {self.http_port} 已被占用，无法启动UI服务器")
            return False
    
    def check_system_health(self):
        """全面检查系统健康状态"""
        results = {}
        
        # 检查UI服务器
        ui_ok = self.check_ui_server()
        results['ui_server'] = ui_ok
        
        # 检查WebSocket代理
        ws_ok = self.check_websocket_proxy()
        results['websocket_proxy'] = ws_ok
        
        # 检查交易服务
        trading_running = self.check_process_running("python main.py")
        results['trading_service'] = trading_running
        
        # 检查网络连接
        try:
            response = requests.get("https://api.binance.com/api/v3/ping", timeout=5)
            network_ok = response.status_code == 200
        except:
            network_ok = False
        results['network_connection'] = network_ok
        
        # 自动修复
        if self.auto_fix:
            if not ws_ok:
                logger.info("正在尝试修复WebSocket代理...")
                self.fix_websocket_proxy()
                
            if not ui_ok:
                logger.info("正在尝试修复UI服务器...")
                self.fix_ui_server()
        
        return results
        
    def print_report(self):
        """打印健康检查报告"""
        print("\n========== 系统健康检查报告 ==========")
        print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n--- 发现的问题 ---")
        if self.problems_found:
            for i, problem in enumerate(self.problems_found, 1):
                print(f"{i}. {problem}")
        else:
            print("未发现问题，系统运行正常！")
            
        if self.auto_fix:
            print("\n--- 已应用的修复 ---")
            if self.fixes_applied:
                for i, fix in enumerate(self.fixes_applied, 1):
                    print(f"{i}. {fix}")
            else:
                print("未应用任何修复")
                
        print("\n--- 建议操作 ---")
        if self.problems_found and not self.fixes_applied:
            print("1. 使用 --auto-fix 选项再次运行此脚本尝试自动修复问题")
            print("2. 运行 'bash scripts/start_optimized.sh --new-ui' 重启整个系统")
        elif not self.problems_found:
            print("系统工作正常，无需操作")
        else:
            if len(self.problems_found) > len(self.fixes_applied):
                print("某些问题无法自动修复，请运行 'bash scripts/start_optimized.sh --new-ui' 重启整个系统")
                
        print("\n===========================================")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='交易系统健康检查工具')
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--auto-fix', action='store_true', help='尝试自动修复问题')
    args = parser.parse_args()
    
    checker = HealthChecker(args.config, args.auto_fix)
    checker.check_system_health()
    checker.print_report()

if __name__ == "__main__":
    main()
