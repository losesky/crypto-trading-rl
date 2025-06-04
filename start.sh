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
    
    # 关闭WebSocket服务器
    if [ -n "$WS_PID" ] && ps -p $WS_PID &>/dev/null; then
        echo "关闭WebSocket服务器 (PID: $WS_PID)..."
        kill -9 $WS_PID &>/dev/null || true
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

# 定义颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # 无颜色

echo -e "${BLUE}🚀 启动BTC交易RL系统${NC}"

# 激活虚拟环境
source venv/bin/activate

# 设置全局错误处理
exec 2>/dev/null  # 全局重定向错误输出

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

# 检查数据是否已准备好
if [ ! -f "btc_rl/data/train_data.npz" ] || [ ! -f "btc_rl/data/test_data.npz" ]; then
    echo -e "${YELLOW}⚠️ 未找到训练或测试数据。需要先获取和处理数据。${NC}"
    
    # 询问用户是否要获取数据
    read -p "是否要现在获取最新的比特币数据? (y/n): " fetch_data
    if [[ "$fetch_data" == "y" || "$fetch_data" == "Y" ]]; then
        # 默认获取最近3年的数据
        START_DATE=$(date -d "3 years ago" +%Y-%m-%d)
        END_DATE=$(date +%Y-%m-%d)
        
        echo -e "${YELLOW}将获取从 $START_DATE 到 $END_DATE 的比特币数据...${NC}"
        
        # 运行数据获取和预处理工作流
        python -m btc_rl.src.data_workflow \
            --exchange binance \
            --start_date "$START_DATE" \
            --end_date "$END_DATE"
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}❌ 数据获取失败，请检查网络连接或手动运行 run_workflow.sh${NC}"
            exit 1
        fi
    else
        echo -e "${RED}❌ 系统需要训练和测试数据才能运行。请先运行 run_workflow.sh 准备数据。${NC}"
        exit 1
    fi
fi

# 启动WebSocket服务器
echo -e "${YELLOW}🔌 启动WebSocket服务器...${NC}"
python -m btc_rl.src.websocket_server > /dev/null 2>&1 &
WS_PID=$!

# 稍等片刻确保WebSocket服务器启动
sleep 2

# 检查WebSocket服务器是否成功启动
if ! ps -p $WS_PID > /dev/null; then
    echo -e "${RED}❌ WebSocket服务器启动失败${NC}"
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
echo -e "💡 请在浏览器中访问: ${BLUE}http://localhost:8080/index.html${NC}"
    
# 临时恢复标准错误以便我们能看到训练进度
exec 2>/dev/tty

echo -e "${YELLOW}🧠 启动模型训练与WebSocket服务...${NC}"
echo -e "💡 按 Ctrl+C 可以随时优雅地停止训练和服务..."

# 使用stdbuf确保输出及时刷新，错误重定向到自定义过滤器
stdbuf -oL -eL python -m btc_rl.src.run_wrapper 2> >(grep -v "Exception\|Error\|Traceback\|Broken" >&2)

# 如果训练正常结束（没有被Ctrl+C中断）
echo "✅  训练已结束，正在关闭服务器..."
cleanup
