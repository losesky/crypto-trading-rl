#!/bin/bash
# 运行BTC交易模型比较服务

# 定义清理函数，用于在脚本退出时杀死所有后台进程
cleanup() {
    echo -e "\n\n🛑 正在优雅地关闭所有服务..."
    
    # 关闭HTTP服务器
    if [ -n "$HTTP_PID" ] && ps -p $HTTP_PID &>/dev/null; then
        echo "关闭HTTP服务器 (PID: $HTTP_PID)..."
        kill -9 $HTTP_PID &>/dev/null || true
    fi
    
    # 关闭WebSocket服务器
    if [ -n "$WS_PID" ] && ps -p $WS_PID &>/dev/null; then
        echo "关闭WebSocket服务器 (PID: $WS_PID)..."
        kill -9 $WS_PID &>/dev/null || true
    fi
    
    # 静默终止所有相关Python进程
    echo "确保所有相关进程已终止..."
    pkill -9 -f "python.*btc_rl.src.model_comparison" &>/dev/null || true
    
    # 释放可能被占用的端口
    for PORT in 8080 8765; do
        fuser -k -9 $PORT/tcp &>/dev/null || true
    done
    
    echo "✅ 所有服务已关闭"
    exit 0
}

# 注册信号处理函数
trap cleanup SIGINT SIGTERM SIGHUP SIGQUIT

# 定义颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # 无颜色

echo -e "${BLUE}🚀 启动BTC交易模型比较服务${NC}"

# 激活虚拟环境
source venv/bin/activate

# 首先确保系统干净，关闭可能存在的任何相关进程
echo -e "${YELLOW}🧹 清理环境，确保没有残留进程...${NC}"
pkill -f "python.*btc_rl.src" 2>/dev/null || true

# 等待片刻确保所有进程停止
sleep 2

# 检查并释放必要端口（更加严格地清理）
for PORT in 8080 8765; do
    if netstat -tuln | grep ":$PORT" > /dev/null; then
        echo -e "${YELLOW}⚠️ 发现端口 $PORT 被占用，尝试释放...${NC}"
        fuser -k -9 $PORT/tcp &>/dev/null
        sleep 2
        
        # 二次检查端口是否真的释放了
        if netstat -tuln | grep ":$PORT" > /dev/null; then
            echo -e "${RED}❌ 无法释放端口 $PORT，可能需要手动干预${NC}"
            echo -e "请尝试手动运行: sudo fuser -k -9 $PORT/tcp"
            exit 1
        fi
    fi
done

# 创建进度条函数
display_progress() {
    local percent=$1
    local width=50
    local num_filled=$((width * percent / 100))
    local num_empty=$((width - num_filled))
    local progress_bar=""
    
    for ((i=0; i<num_filled; i++)); do
        progress_bar+="█"
    done
    
    for ((i=0; i<num_empty; i++)); do
        progress_bar+="░"
    done
    
    # 清除当前行并显示进度条
    echo -ne "\r[${progress_bar}] ${percent}%"
}

# 启动模型比较WebSocket服务器并预加载模型
echo -e "${YELLOW}🔌 启动模型比较服务器并预加载模型...${NC}"
# 删除旧的进度文件（如果存在）
rm -f btc_rl/preload_progress.json
# 将错误输出保存到日志文件以便诊断
python -m btc_rl.src.model_comparison --preload > ws_output.log 2>&1 &
WS_PID=$!

# 打印进程信息以便调试
echo -e "${YELLOW}📊 WebSocket服务器进程ID: ${WS_PID}${NC}"

# 等待进度文件出现
echo -e "${YELLOW}⏳ 等待模型预加载开始...${NC}"
while [ ! -f btc_rl/preload_progress.json ]; do
    sleep 1
    if ! ps -p $WS_PID > /dev/null; then
        echo -e "${RED}❌ 模型比较服务器启动失败${NC}"
        echo -e "${YELLOW}查看日志文件 ws_output.log 获取详细错误信息${NC}"
        cat ws_output.log
        cleanup
        exit 1
    fi
done

# 显示预加载进度条
echo -e "${YELLOW}⏳ 模型预加载进度:${NC}"
PROGRESS=0
while [ $PROGRESS -lt 100 ]; do
    if [ -f btc_rl/preload_progress.json ]; then
        # 使用jq解析JSON，如果jq不可用，则使用Python作为备选
        if command -v jq > /dev/null; then
            PROGRESS=$(jq -r '.percent // 0' btc_rl/preload_progress.json 2>/dev/null || echo 0)
            FINISHED=$(jq -r '.finished // false' btc_rl/preload_progress.json 2>/dev/null || echo "false")
        else
            PROGRESS=$(python -c "import json; data=json.load(open('btc_rl/preload_progress.json')); print(data.get('percent', 0))" 2>/dev/null || echo 0)
            FINISHED=$(python -c "import json; data=json.load(open('btc_rl/preload_progress.json')); print(data.get('finished', False))" 2>/dev/null || echo "False")
        fi
        
        display_progress $PROGRESS
        
        # 如果标记为完成，跳出循环
        if [ "$FINISHED" = "true" ] || [ "$FINISHED" = "True" ]; then
            break
        fi
    fi
    
    # 检查WebSocket服务器是否仍在运行
    if ! ps -p $WS_PID > /dev/null; then
        echo -e "\n${RED}❌ 模型比较服务器意外终止${NC}"
        echo -e "${YELLOW}查看日志文件 ws_output.log 获取详细错误信息${NC}"
        cat ws_output.log
        cleanup
        exit 1
    fi
    
    sleep 1
done

# 完成预加载
echo -e "\n${GREEN}✅ 模型预加载完成${NC}"

# 检查WebSocket服务器是否成功启动
if ! ps -p $WS_PID > /dev/null; then
    echo -e "${RED}❌ 模型比较服务器启动失败${NC}"
    echo -e "${YELLOW}查看日志文件 ws_output.log 获取详细错误信息${NC}"
    cat ws_output.log
    cleanup
    exit 1
fi

# 启动HTTP服务器
echo -e "${YELLOW}🌐 启动HTTP服务器...${NC}"
python -m btc_rl.src.http_server > /dev/null 2>&1 &
HTTP_PID=$!

# 稍等片刻确保HTTP服务器启动
sleep 2

# 检查HTTP服务器是否成功启动
if ! ps -p $HTTP_PID > /dev/null; then
    echo -e "${RED}❌ HTTP服务器启动失败${NC}"
    cleanup
    exit 1
fi

# 告知用户可视化界面的URL
echo -e "${GREEN}✅ 服务已成功启动${NC}"
echo -e "💡 请在浏览器中访问: ${BLUE}http://localhost:8080/model_comparison.html${NC}"

# 临时恢复标准错误以便我们能看到服务日志
exec 2>/dev/tty

echo -e "${YELLOW}💡 模型比较服务器正在运行...${NC}"
echo -e "${YELLOW}💡 按 Ctrl+C 可以随时优雅地停止服务...${NC}"

# 等待用户终止
wait $WS_PID

# 如果WebSocket服务器结束，则清理资源
echo "✅ 服务已结束，正在关闭所有相关进程..."
cleanup
