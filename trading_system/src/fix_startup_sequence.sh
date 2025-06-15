#!/bin/bash
# 该脚本用于优化WebSocket代理和API代理的启动顺序

# 获取脚本参数
MODE=$1
UI_ONLY=$2
WS_PROXY_ONLY=$3

# 输出版本信息
echo "优化WebSocket连接启动脚本 v1.0"
echo "当前模式: $MODE"
echo "UI_ONLY: $UI_ONLY"
echo "WS_PROXY_ONLY: $WS_PROXY_ONLY"

# 获取主代理目录
TRADING_SYSTEM_DIR=`dirname $0`/../

# 检查native_websocket_proxy.py文件中是否需要更新引用
NATIVE_WS_PROXY="$TRADING_SYSTEM_DIR/src/native_websocket_proxy.py"
if [ -f "$NATIVE_WS_PROXY" ]; then
    echo "正在检查WebSocket代理文件..."
    # 更新实例获取函数
    sed -i 's/proxy = get_instance()/proxy = get_instance(port=8095, rest_port=8096)/' "$NATIVE_WS_PROXY" 2>/dev/null || true
fi

# 检查API代理是否正确启动
API_PROXY_LAUNCHER="$TRADING_SYSTEM_DIR/src/start_api_proxy.py" 
if [ -f "$API_PROXY_LAUNCHER" ]; then
    echo "正在检查API代理启动器..."
    # 确保API代理正确引用WebSocket端口
    if grep -q "ws_url=" "$API_PROXY_LAUNCHER"; then
        echo "API代理代码正常"
    else
        echo "API代理启动器可能需要更新，请检查"
    fi
fi

# 检查前端app.js
APP_JS="$TRADING_SYSTEM_DIR/ui/app.js"
if [ -f "$APP_JS" ]; then
    echo "正在检查前端app.js..."
    # 检查是否缺少updateSystemStatus函数
    if ! grep -q "function updateSystemStatus" "$APP_JS"; then
        echo "前端app.js中缺少updateSystemStatus函数，请添加"
    fi
fi

echo "优化脚本执行完毕"
