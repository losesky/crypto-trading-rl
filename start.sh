#!/bin/bash
# 启动BTC交易RL系统

# 定义清理函数，用于在脚本退出时杀死所有后台进程
cleanup() {
    echo -e "\n\n🛑 正在优雅地关闭所有服务..."
    
    # 关闭HTTP服务器
    if [ -n "$HTTP_PID" ] && ps -p $HTTP_PID &>/dev/null; then
        echo "关闭HTTP服务器 (PID: $HTTP_PID)..."
        kill -9 $HTTP_PID &>/dev/null || true
    fi
    
    # 静默终止所有相关Python进程
    echo "确保所有相关进程已终止..."
    pkill -9 -f "python.*btc_rl.src" &>/dev/null || true
    
    # 释放可能被占用的端口
    for PORT in 8080 8765; do
        fuser -k -9 $PORT/tcp &>/dev/null || true
    done
    
    echo "✅ 所有服务已关闭"
    exit 0
}

# 注册信号处理函数
trap cleanup SIGINT SIGTERM SIGHUP SIGQUIT

echo "🚀 启动BTC交易RL可视化系统"

# 激活虚拟环境
source venv/bin/activate

# 设置全局错误处理
exec 2>/dev/null  # 全局重定向错误输出

# 首先确保系统干净，关闭可能存在的任何相关进程
echo "🧹 清理环境，确保没有残留进程..."
pkill -f "python.*btc_rl.src" 2>/dev/null || true

# 检查并释放必要端口
for PORT in 8080 8765; do
    if netstat -tuln | grep ":$PORT" > /dev/null; then
        echo "⚠️ 发现端口 $PORT 被占用，尝试释放..."
        fuser -k $PORT/tcp &>/dev/null
        sleep 1
    fi
done

# 启动HTTP服务器（使用专用的错误抑制包装器）
echo "1️⃣ 启动HTTP服务器..."
python -m btc_rl.src.http_wrapper &
HTTP_PID=$!
sleep 1
echo "✅ HTTP服务器已启动 (PID: $HTTP_PID)"

# 打开浏览器（如果在有GUI的环境中）
echo "2️⃣ 正在打开浏览器..."
if command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:8080/index.html &>/dev/null &
elif command -v open &> /dev/null; then
    open http://localhost:8080/index.html &>/dev/null &
else
    echo "请手动在浏览器中打开: http://localhost:8080/index.html"
fi

# 提示训练即将开始
echo "3️⃣ 启动模型训练与WebSocket服务..."
echo -e "✅ 训练进程即将启动..."
echo -e "💡 按 Ctrl+C 可以随时优雅地停止训练和服务..."
# 临时恢复标准错误以便我们能看到训练进度
exec 2>/dev/tty

# 运行自定义的Python包装器，它会处理所有的错误信息
# 使用stdbuf确保输出及时刷新，错误重定向到自定义过滤器
stdbuf -oL -eL python -m btc_rl.src.run_wrapper 2> >(grep -v "Exception\|Error\|Traceback\|Broken" >&2)
# 训练进程结束后进行清理
echo "✅  训练已结束，正在关闭服务器..."
cleanup
