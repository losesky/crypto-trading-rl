#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RL交易系统 - 数据修复工具
用于诊断和修复数据流问题
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
import requests

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DataDiagnostics')

# 获取项目根目录
SCRIPT_DIR = Path(__file__).parent.absolute()
TRADING_SYSTEM_DIR = SCRIPT_DIR.parent
ROOT_DIR = TRADING_SYSTEM_DIR.parent

# 默认配置
DEFAULT_CONFIG_PATH = TRADING_SYSTEM_DIR / "config" / "test_config.json"
DEFAULT_API_PORT = 8090
DEFAULT_WS_PORT = 8095

def load_config(config_path):
    """加载配置文件"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"已加载配置文件：{config_path}")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败：{e}")
        sys.exit(1)

def check_api_connectivity(host="localhost", port=8090):
    """检查API连接"""
    url = f"http://{host}:{port}/api/status"
    try:
        logger.info(f"检查API连接：{url}")
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            logger.info("API连接正常")
            try:
                data = response.json()
                logger.info(f"API返回状态：{data}")
                return True, data
            except Exception as e:
                logger.warning(f"解析API响应失败：{e}")
                return True, None
        else:
            logger.error(f"API连接失败，状态码：{response.status_code}")
            return False, None
    except requests.exceptions.RequestException as e:
        logger.error(f"API请求错误：{e}")
        return False, None

def check_websocket_connectivity(host="localhost", port=8095):
    """检查WebSocket连接"""
    # 尝试导入websocket库
    try:
        import websocket
    except ImportError:
        logger.error("未安装websocket-client库，无法检查WebSocket连接")
        logger.info("请运行 'pip install websocket-client' 安装")
        return False
    
    url = f"ws://{host}:{port}"
    logger.info(f"检查WebSocket连接：{url}")
    
    try:
        # 创建连接并等待
        ws = websocket.create_connection(url, timeout=5)
        logger.info("WebSocket连接成功")
        
        # 发送ping消息
        ws.send(json.dumps({"ping": int(time.time() * 1000)}))
        logger.info("已发送ping消息")
        
        # 等待响应
        try:
            response = ws.recv()
            logger.info(f"接收到WebSocket响应：{response}")
            ws.close()
            return True
        except websocket.WebSocketTimeoutException:
            logger.warning("WebSocket响应超时")
            ws.close()
            return False
            
    except Exception as e:
        logger.error(f"WebSocket连接失败：{e}")
        return False

def check_dependencies():
    """检查依赖项"""
    missing_packages = []
    
    # 检查必要的Python包
    required_packages = [
        ('flask', 'flask'),
        ('flask_cors', 'flask-cors'),
        ('flask_socketio', 'flask-socketio'),
        ('socketio', 'python-socketio'),
        ('websocket', 'websocket-client'),
        ('requests', 'requests'),
    ]
    
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            logger.info(f"✓ {package_name} 已安装")
        except ImportError:
            missing_packages.append(package_name)
            logger.warning(f"✗ {package_name} 未安装")
    
    # 返回缺失的包列表
    return missing_packages

def install_packages(packages):
    """安装缺失的包"""
    if not packages:
        return True
        
    logger.info(f"安装缺失的包：{', '.join(packages)}")
    try:
        import subprocess
        for package in packages:
            logger.info(f"安装 {package}...")
            process = subprocess.Popen(
                [sys.executable, "-m", "pip", "install", package],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"安装 {package} 失败：{stderr.decode()}")
                return False
            else:
                logger.info(f"已安装 {package}")
        return True
    except Exception as e:
        logger.error(f"安装包失败：{e}")
        return False

def repair_data_sender(create_backup=True):
    """修复数据发送器问题"""
    data_sender_path = TRADING_SYSTEM_DIR / "src" / "data_sender.py"
    backup_path = TRADING_SYSTEM_DIR / "src" / "data_sender.py.bak"
    
    # 检查文件是否存在
    if not data_sender_path.exists():
        logger.error(f"未找到数据发送器文件：{data_sender_path}")
        return False
    
    # 创建备份
    if create_backup and not backup_path.exists():
        logger.info("创建数据发送器备份文件")
        with open(data_sender_path, 'r') as source:
            with open(backup_path, 'w') as backup:
                backup.write(source.read())
    
    # 读取文件内容
    with open(data_sender_path, 'r') as f:
        content = f.read()
    
    # 检查是否需要修复
    needs_import_fix = 'from websocket_proxy import' not in content
    needs_type_fix = '"type": "market_update"' in content
    needs_send_fix = '_send_data(data_to_send)' in content and 'event_type' not in content

    if not (needs_import_fix or needs_type_fix or needs_send_fix):
        logger.info("数据发送器没有检测到需要修复的问题")
        return True
    
    # 应用修复
    logger.info("修复数据发送器问题")
    
    # 1. 导入WebSocket代理
    if needs_import_fix:
        import_code = '''
# 尝试导入WebSocket代理
try:
    from websocket_proxy import get_instance as get_websocket_proxy
    USE_WEBSOCKET_PROXY = True
except ImportError:
    USE_WEBSOCKET_PROXY = False
'''
        content = content.replace('import json\nimport logging\nimport threading', 'import json\nimport logging\nimport threading' + import_code)
        
        # 修复构造函数
        old_init = '''    def __init__(self, config):
        """
        初始化数据发送器
        
        参数:
        - config: 配置字典
        """
        self.logger = logging.getLogger("DataSender")
        self.config = config
        
        # 从配置中获取WebSocket服务器信息
        ws_port = self.config['ui'].get('ws_port', 8095)
        http_port = ws_port + 1  # HTTP端口是WebSocket端口+1
        
        # 数据发送URL
        self.server_url = f"http://localhost:{http_port}"
        
        # 状态变量
        self.is_sending = False
        self.send_thread = None
        self.send_interval = 1.0  # 默认发送间隔为1秒'''
        
        new_init = '''    def __init__(self, config):
        """
        初始化数据发送器
        
        参数:
        - config: 配置字典
        """
        self.logger = logging.getLogger("DataSender")
        self.config = config
        
        # 从配置中获取WebSocket服务器信息
        ws_port = self.config['ui'].get('ws_port', 8095)
        
        # WebSocket代理模式
        if USE_WEBSOCKET_PROXY:
            self.logger.info("使用WebSocket代理进行数据发送")
            self.ws_proxy = get_websocket_proxy(port=ws_port)
            self.ws_proxy.start()
        # HTTP模式
        else:
            self.logger.info("使用HTTP请求进行数据发送")
            http_port = ws_port + 1  # HTTP端口是WebSocket端口+1
            self.server_url = f"http://localhost:{http_port}"
        
        # 状态变量
        self.is_sending = False
        self.send_thread = None
        self.send_interval = self.config['ui'].get('update_interval', 1000) / 1000.0'''
        
        content = content.replace(old_init, new_init)
    
    # 2. 修复_send_data方法
    if needs_send_fix:
        old_send_data = '''    def _send_data(self, data):
        """发送数据到WebSocket服务器"""
        try:
            url = f"{self.server_url}"
            headers = {'Content-Type': 'application/json'}
            
            # 确保数据有时间戳
            if 'timestamp' not in data:
                data['timestamp'] = int(datetime.now().timestamp() * 1000)
            
            response = requests.post(url, json=data, headers=headers, timeout=2)
            
            if response.status_code != 200:
                self.logger.warning(f"发送数据失败，状态码: {response.status_code}")
                self.logger.debug(f"响应内容: {response.text}")
                return False
            
            return True
        except requests.exceptions.RequestException as e:
            self.logger.error(f"HTTP请求错误: {e}")
            return False'''
            
        new_send_data = '''    def _send_data(self, event_type, data):
        """发送数据到前端"""
        # 确保数据有时间戳
        if isinstance(data, dict) and 'timestamp' not in data:
            data['timestamp'] = int(datetime.now().timestamp() * 1000)
            
        # WebSocket代理模式
        if USE_WEBSOCKET_PROXY:
            return self.ws_proxy.send_data(event_type, data)
        
        # HTTP模式
        try:
            # 为HTTP请求增加事件类型
            data_to_send = data.copy()
            data_to_send['type'] = event_type
            
            url = f"{self.server_url}"
            headers = {'Content-Type': 'application/json'}
            
            response = requests.post(url, json=data_to_send, headers=headers, timeout=2)
            
            if response.status_code != 200:
                self.logger.warning(f"发送数据失败，状态码: {response.status_code}")
                self.logger.debug(f"响应内容: {response.text}")
                return False
            
            return True
        except requests.exceptions.RequestException as e:
            self.logger.error(f"HTTP请求错误: {e}")
            return False'''
            
        content = content.replace(old_send_data, new_send_data)
    
    # 3. 修复_send_loop方法
    if needs_send_fix:
        old_send_loop = '''    def _send_loop(self):
        """发送数据的主循环"""
        while self.is_sending:
            try:
                # 发送最新的市场数据
                if self.latest_market_data:
                    self._send_data(self.latest_market_data)
                
                # 发送最新的持仓数据
                if self.latest_position_data:
                    self._send_data(self.latest_position_data)
                
                # 发送最新的预测数据
                if self.latest_prediction_data:
                    self._send_data(self.latest_prediction_data)
                
                # 发送系统状态
                if self.system_status:
                    self._send_data(self.system_status)
                
            except Exception as e:
                self.logger.error(f"发送数据时出错: {e}")
            
            # 等待一段时间
            time.sleep(self.send_interval)'''
            
        new_send_loop = '''    def _send_loop(self):
        """发送数据的主循环"""
        heartbeat_counter = 0
        while self.is_sending:
            try:
                # 发送最新的市场数据
                if self.latest_market_data:
                    self._send_data("market_update", self.latest_market_data)
                
                # 发送最新的持仓数据
                if self.latest_position_data:
                    self._send_data("position_update", self.latest_position_data)
                
                # 发送最新的预测数据
                if self.latest_prediction_data:
                    self._send_data("prediction_update", self.latest_prediction_data)
                
                # 发送系统状态
                if self.system_status:
                    self._send_data("status_update", self.system_status)
                
                # 每10次循环发送一次心跳
                heartbeat_counter += 1
                if heartbeat_counter >= 10:
                    self._send_data("heartbeat", {"timestamp": int(datetime.now().timestamp() * 1000)})
                    heartbeat_counter = 0
                    
            except Exception as e:
                self.logger.error(f"发送数据时出错: {e}")
            
            # 等待一段时间
            time.sleep(self.send_interval)'''
            
        content = content.replace(old_send_loop, new_send_loop)
    
    # 4. 移除数据类型字段
    if needs_type_fix:
        # 修复update_market_data方法
        content = content.replace('"type": "market_update",', '')
        content = content.replace('"type": "position_update",', '')
        content = content.replace('"type": "prediction_update",', '')
        content = content.replace('"type": "order_update",', '')
        content = content.replace('"type": "alert",', '')
    
    # 5. 修复方法调用
    if needs_send_fix:
        content = content.replace('self._send_data(data_to_send)', 'self._send_data("market_update", data_to_send)')
        content = content.replace('self._send_data(self.latest_market_data)', 'self._send_data("market_update", self.latest_market_data)')
        content = content.replace('self._send_data(self.latest_position_data)', 'self._send_data("position_update", self.latest_position_data)')
        content = content.replace('self._send_data(self.latest_prediction_data)', 'self._send_data("prediction_update", self.latest_prediction_data)')
        content = content.replace('self._send_data(self.system_status)', 'self._send_data("status_update", self.system_status)')
        content = content.replace('self._send_data(status_data)', 'self._send_data("status_update", status_data)')
        content = content.replace('self._send_data(data_to_send)', 'self._send_data("order_update", data_to_send)')
    
    # 写回文件
    with open(data_sender_path, 'w') as f:
        f.write(content)
    
    logger.info("已成功修复数据发送器文件")
    return True

def setup_websocket_proxy():
    """设置WebSocket代理"""
    proxy_path = TRADING_SYSTEM_DIR / "src" / "websocket_proxy.py"
    
    # 检查文件是否存在
    if proxy_path.exists():
        logger.info("WebSocket代理文件已存在")
        return True
    
    # 创建WebSocket代理文件
    logger.info("创建WebSocket代理文件")
    
    proxy_content = """#!/usr/bin/env python
# -*- coding: utf-8 -*-
\"\"\"
WebSocket代理服务器 - 用于将系统数据实时推送到前端
\"\"\"
import json
import logging
import threading
import time
from datetime import datetime
from flask import Flask, json
from flask_socketio import SocketIO
from flask_cors import CORS

class WebSocketProxy:
    \"\"\"WebSocket代理服务，负责处理实时数据推送\"\"\"
    
    def __init__(self, port=8095):
        \"\"\"初始化WebSocket代理服务\"\"\"
        self.logger = logging.getLogger("WebSocketProxy")
        self.port = port
        
        # 创建Flask应用
        self.app = Flask(__name__)
        CORS(self.app)
        
        # 初始化SocketIO
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode="threading")
        
        # 状态变量
        self.is_running = False
        self.server_thread = None
        
        # 接收到的最新数据
        self.latest_data = {}
        self.connected_clients = 0
        
        # 设置事件处理
        @self.socketio.on('connect')
        def handle_connect():
            self.connected_clients += 1
            self.logger.info(f"客户端连接，当前连接数：{self.connected_clients}")
            # 连接后立即发送一些最新数据
            self._send_latest_data_to_client()
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.connected_clients -= 1
            self.logger.info(f"客户端断开连接，当前连接数：{self.connected_clients}")
        
        @self.socketio.on('ping')
        def handle_ping():
            self.socketio.emit('pong', {'timestamp': int(datetime.now().timestamp() * 1000)})
            
        @self.app.route('/health')
        def health_check():
            return json.dumps({"status": "ok", "clients": self.connected_clients})
    
    def _send_latest_data_to_client(self):
        \"\"\"向新连接的客户端发送最新数据\"\"\"
        for data_type, data in self.latest_data.items():
            try:
                self.socketio.emit(data_type, data)
            except Exception as e:
                self.logger.error(f"发送最新数据失败: {e}")
    
    def start(self):
        \"\"\"启动WebSocket代理服务\"\"\"
        if self.is_running:
            self.logger.warning("WebSocket代理服务已在运行")
            return
        
        self.is_running = True
        self.logger.info(f"启动WebSocket代理服务，端口：{self.port}")
        
        # 在单独的线程中启动服务器
        self.server_thread = threading.Thread(
            target=self.socketio.run,
            args=(self.app,),
            kwargs={'host': '0.0.0.0', 'port': self.port, 'debug': False, 'use_reloader': False}
        )
        self.server_thread.daemon = True
        self.server_thread.start()
    
    def stop(self):
        \"\"\"停止WebSocket代理服务\"\"\"
        if not self.is_running:
            return
        
        self.is_running = False
        self.logger.info("停止WebSocket代理服务")
        
        # 关闭所有连接
        self.socketio.stop()
    
    def send_data(self, event_type, data):
        \"\"\"发送数据到所有连接的客户端\"\"\"
        if not self.is_running:
            return
        
        try:
            # 确保数据有时间戳
            if isinstance(data, dict) and 'timestamp' not in data:
                data['timestamp'] = int(datetime.now().timestamp() * 1000)
            
            # 保存最新数据
            self.latest_data[event_type] = data
            
            # 发送数据
            self.socketio.emit(event_type, data)
            return True
        except Exception as e:
            self.logger.error(f"发送数据失败: {e}")
            return False

# 单例模式
_instance = None

def get_instance(port=8095):
    \"\"\"获取WebSocketProxy实例（单例模式）\"\"\"
    global _instance
    if _instance is None:
        _instance = WebSocketProxy(port)
    return _instance

# 命令行入口
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("WebSocketProxyMain")
    
    # 创建并启动代理服务
    proxy = get_instance()
    proxy.start()
    
    try:
        logger.info("WebSocket代理服务已启动，按Ctrl+C停止...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("接收到停止信号，关闭服务...")
    finally:
        proxy.stop()
"""
    
    # 写入文件
    with open(proxy_path, 'w') as f:
        f.write(proxy_content)
    
    logger.info("已成功创建WebSocket代理文件")
    return True

def run_diagnostics(config):
    """运行诊断并显示结果"""
    logger.info("正在运行系统诊断...")
    
    # 组装诊断结果
    results = {
        "依赖检查": {
            "状态": "通过",
            "详情": "所有必要依赖已安装"
        },
        "API连接": {
            "状态": "未检查",
            "详情": "未执行API连接测试"
        },
        "WebSocket连接": {
            "状态": "未检查",
            "详情": "未执行WebSocket连接测试"
        },
        "数据发送器": {
            "状态": "未检查",
            "详情": "未检查数据发送器配置"
        },
        "WebSocket代理": {
            "状态": "未检查",
            "详情": "未检查WebSocket代理配置"
        }
    }
    
    # 检查依赖
    missing_packages = check_dependencies()
    if missing_packages:
        results["依赖检查"]["状态"] = "失败"
        results["依赖检查"]["详情"] = f"缺少以下包：{', '.join(missing_packages)}"
        logger.warning(f"依赖检查失败：缺少 {len(missing_packages)} 个包")
    
    # 检查API连接
    http_port = config['ui'].get('http_port', DEFAULT_API_PORT)
    api_success, api_data = check_api_connectivity(port=http_port)
    results["API连接"]["状态"] = "通过" if api_success else "失败"
    results["API连接"]["详情"] = "API连接正常" if api_success else "无法连接到API服务"
    
    # 检查WebSocket连接
    ws_port = config['ui'].get('ws_port', DEFAULT_WS_PORT)
    ws_success = check_websocket_connectivity(port=ws_port)
    results["WebSocket连接"]["状态"] = "通过" if ws_success else "失败"
    results["WebSocket连接"]["详情"] = "WebSocket连接正常" if ws_success else "无法连接到WebSocket服务"
    
    # 检查数据发送器文件
    data_sender_path = TRADING_SYSTEM_DIR / "src" / "data_sender.py"
    if data_sender_path.exists():
        with open(data_sender_path, 'r') as f:
            content = f.read()
            has_ws_proxy = 'from websocket_proxy import' in content
            has_event_type = '_send_data(event_type, data)' in content
            
            if has_ws_proxy and has_event_type:
                results["数据发送器"]["状态"] = "通过"
                results["数据发送器"]["详情"] = "数据发送器配置正确"
            else:
                results["数据发送器"]["状态"] = "需要修复"
                problems = []
                if not has_ws_proxy:
                    problems.append("缺少WebSocket代理导入")
                if not has_event_type:
                    problems.append("send_data方法缺少事件类型参数")
                results["数据发送器"]["详情"] = f"发现问题：{', '.join(problems)}"
    else:
        results["数据发送器"]["状态"] = "失败"
        results["数据发送器"]["详情"] = "未找到数据发送器文件"
    
    # 检查WebSocket代理文件
    ws_proxy_path = TRADING_SYSTEM_DIR / "src" / "websocket_proxy.py"
    if ws_proxy_path.exists():
        results["WebSocket代理"]["状态"] = "通过"
        results["WebSocket代理"]["详情"] = "WebSocket代理文件存在"
    else:
        results["WebSocket代理"]["状态"] = "需要创建"
        results["WebSocket代理"]["详情"] = "需要创建WebSocket代理文件"
    
    # 打印诊断结果
    logger.info("\n==== 诊断结果 ====")
    for category, result in results.items():
        status_symbol = "✅" if result["状态"] == "通过" else "❌" if result["状态"] == "失败" else "⚠️"
        logger.info(f"{status_symbol} {category}: {result['状态']} - {result['详情']}")
    
    return results

def fix_issues(config, results):
    """修复发现的问题"""
    logger.info("\n==== 开始修复问题 ====")
    
    # 安装缺失的依赖
    if results["依赖检查"]["状态"] != "通过":
        missing_packages = check_dependencies()
        if missing_packages:
            logger.info("安装缺失的依赖...")
            if install_packages(missing_packages):
                logger.info("✅ 已安装所有缺失的依赖")
            else:
                logger.error("❌ 安装依赖失败")
                
    # 修复数据发送器
    if results["数据发送器"]["状态"] != "通过":
        logger.info("修复数据发送器...")
        if repair_data_sender():
            logger.info("✅ 数据发送器修复成功")
        else:
            logger.error("❌ 数据发送器修复失败")
    
    # 设置WebSocket代理
    if results["WebSocket代理"]["状态"] != "通过":
        logger.info("设置WebSocket代理...")
        if setup_websocket_proxy():
            logger.info("✅ WebSocket代理设置成功")
        else:
            logger.error("❌ WebSocket代理设置失败")
    
    logger.info("修复完成！")
    
    # 修复后重新诊断
    logger.info("\n==== 修复后重新诊断 ====")
    run_diagnostics(config)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RL交易系统数据诊断工具')
    parser.add_argument('--config', '-c', type=str, default=str(DEFAULT_CONFIG_PATH), help='配置文件路径')
    parser.add_argument('--fix', '-f', action='store_true', help='自动修复发现的问题')
    parser.add_argument('--test', '-t', action='store_true', help='运行测试后退出')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 运行诊断
    results = run_diagnostics(config)
    
    # 如果指定了修复，则尝试修复问题
    if args.fix:
        fix_issues(config, results)
    else:
        # 如果发现问题但没有指定修复，提示用户
        has_issues = any(result["状态"] != "通过" for result in results.values())
        if has_issues:
            logger.info("\n发现问题！使用 --fix 选项自动修复问题：")
            logger.info(f"python {sys.argv[0]} --fix")
    
    # 如果只是测试，直接退出
    if args.test:
        return
    
    # 提供进一步操作的建议
    logger.info("\n==== 后续步骤 ====")
    logger.info("1. 运行优化版启动脚本以启动交易系统:")
    logger.info(f"   {TRADING_SYSTEM_DIR}/scripts/start_optimized.sh --new-ui")
    logger.info("2. 如果WebSocket服务无法连接，可以单独启动WebSocket代理:")
    logger.info(f"   {TRADING_SYSTEM_DIR}/scripts/start_optimized.sh --ws-proxy")
    logger.info("3. 如果只想查看UI界面，可以运行:")
    logger.info(f"   {TRADING_SYSTEM_DIR}/scripts/start_optimized.sh --ui-only --new-ui")

if __name__ == '__main__':
    main()
