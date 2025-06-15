#!/bin/bash

# 停止交易系统脚本
echo "正在停止交易系统..."

# 获取脚本所在目录的父目录（trading_system目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRADING_SYSTEM_DIR="$( dirname "$SCRIPT_DIR" )"

# 检查交易服务PID文件
TRADING_PID_FILE="$TRADING_SYSTEM_DIR/logs/trading_service.pid"
if [ -f "$TRADING_PID_FILE" ]; then
    PID=$(cat "$TRADING_PID_FILE")
    if kill -0 $PID 2>/dev/null; then
        echo "正在终止交易服务进程 (PID: $PID)..."
        kill $PID
        sleep 2
        if kill -0 $PID 2>/dev/null; then
            echo "进程未响应，强制终止..."
            kill -9 $PID
        fi
    else
        echo "交易进程 (PID: $PID) 已不存在"
    fi
    rm -f "$TRADING_PID_FILE"
else
    echo "未找到交易服务PID文件，尝试查找并终止所有相关进程..."
    # 尝试查找并终止所有相关Python进程
    ps aux | grep '[p]ython.*main.py' | awk '{print $2}' | xargs kill 2>/dev/null
fi

echo "交易系统已停止"
