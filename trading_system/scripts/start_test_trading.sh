#!/bin/bash

# 启动测试环境交易脚本
echo "准备启动币安测试网交易系统..."

# 获取脚本所在目录的父目录（trading_system目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRADING_SYSTEM_DIR="$( dirname "$SCRIPT_DIR" )"
ROOT_DIR="$( dirname "$TRADING_SYSTEM_DIR" )"

# 检测是否存在虚拟环境并激活
if [ -d "$ROOT_DIR/venv" ]; then
    echo "激活虚拟环境..."
    source "$ROOT_DIR/venv/bin/activate"
else
    echo "错误：未找到虚拟环境。请先运行 './scripts/install_dependencies.sh'"
    exit 1
fi

# 检查API配置
CONFIG_FILE="$TRADING_SYSTEM_DIR/config/test_config.json"
API_KEY=$(grep -o '"api_key": "[^"]*"' "$CONFIG_FILE" | cut -d'"' -f4)

if [ -z "$API_KEY" ]; then
    echo "错误：未找到API密钥。请在配置文件中配置您的币安测试网API密钥和密码。"
    echo "编辑文件: $CONFIG_FILE"
    exit 1
fi

# 创建日志目录
mkdir -p "$TRADING_SYSTEM_DIR/logs"

# 安装缺少的依赖
pip install -q flask-cors

# 将必要的目录添加到PYTHONPATH，确保能够正确导入模块
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

# 启动交易系统
echo "启动交易系统..."
cd "$TRADING_SYSTEM_DIR/src"

# 创建备份日志文件（以防main.py中的日志配置有问题）
BACKUP_LOG="$TRADING_SYSTEM_DIR/logs/trading_$(date +"%Y%m%d_%H%M%S").log"
echo "日志文件: $BACKUP_LOG"

# 通过直接在src目录中运行main.py解决相对导入问题
# 同时将所有输出重定向到备份日志文件
python main.py --config "$CONFIG_FILE" --mode test 2>&1 | tee "$BACKUP_LOG"

echo "交易系统已启动！"
echo "您可以通过访问 http://localhost:8090 查看交易仪表盘。"
echo "按 Ctrl+C 停止交易服务..."

# 捕获Ctrl+C信号
trap "echo '收到停止信号，正在关闭...' && exit 0" SIGINT SIGTERM

# 等待用户Ctrl+C
wait

echo "交易系统已关闭。"