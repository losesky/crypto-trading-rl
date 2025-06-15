#!/bin/bash

# 生产环境币安U本位合约交易脚本
echo "准备启动生产环境交易系统..."

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
CONFIG_FILE="$TRADING_SYSTEM_DIR/config/prod_config.json"
API_KEY=$(grep -o '"api_key": "[^"]*"' "$CONFIG_FILE" | cut -d'"' -f4)

if [ -z "$API_KEY" ]; then
    echo "错误：未找到API密钥。请在配置文件中配置您的币安API密钥和密码。"
    echo "编辑文件: $CONFIG_FILE"
    exit 1
fi

# 安全确认
echo "⚠️ 警告：您正在启动生产环境交易系统，将使用真实资金进行交易！"
read -p "请输入 'CONFIRM' 确认您理解风险并希望继续: " confirmation

if [ "$confirmation" != "CONFIRM" ]; then
    echo "操作已取消。"
    exit 0
fi

# 创建日志目录
mkdir -p "$TRADING_SYSTEM_DIR/logs"
LOG_FILE="$TRADING_SYSTEM_DIR/logs/trading_$(date +%Y%m%d_%H%M%S).log"

# 安装缺少的依赖
pip install -q flask-cors

# 将当前目录添加到PYTHONPATH
export PYTHONPATH=$TRADING_SYSTEM_DIR/src:$PYTHONPATH

# 启动交易系统
echo "启动交易系统..."
cd "$TRADING_SYSTEM_DIR/src"
nohup python main.py --config "$CONFIG_FILE" --mode prod > "$LOG_FILE" 2>&1 &
TRADING_PID=$!

# 存储PID以便后续可能的清理
echo $TRADING_PID > "$TRADING_SYSTEM_DIR/logs/trading_service.pid"

echo "交易系统已成功在后台启动！"
echo "您可以通过访问 http://localhost:8091 查看交易仪表盘。"
echo "交易日志将保存在: $LOG_FILE"
echo ""
echo "要停止交易系统，请运行: ./scripts/stop_trading.sh"