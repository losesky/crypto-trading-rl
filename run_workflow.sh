#!/bin/bash
# 运行BTC交易智能体的完整流程，从数据获取到模型训练

# 定义清理函数，用于在脚本退出时杀死所有后台进程
cleanup() {
    echo -e "\n\n🛑 正在优雅地关闭所有服务..."
    
    # 关闭HTTP服务器
    if [ -n "$HTTP_PID" ] && ps -p $HTTP_PID &>/dev/null; then
        echo "关闭HTTP服务器 (PID: $HTTP_PID)..."
        kill -9 $HTTP_PID &>/dev/null || true
    fi
    
    # 关闭WebSocket服务器
    if [ -n "$WEBSOCKET_PID" ] && ps -p $WEBSOCKET_PID &>/dev/null; then
        echo "关闭WebSocket服务器 (PID: $WEBSOCKET_PID)..."
        kill -9 $WEBSOCKET_PID &>/dev/null || true
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

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # 无颜色

# 设置工作目录
WORKSPACE="/home/losesky/crypto-trading-rl"
cd $WORKSPACE

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

# 恢复标准错误输出
exec 2>/dev/tty

# 读取配置文件
CONFIG_FILE="$WORKSPACE/config.ini"

# 从配置文件读取默认参数
get_config_value() {
    section=$1
    key=$2
    default=$3
    
    if [ -f "$CONFIG_FILE" ]; then
        value=$(grep -A 20 "^\[$section\]" "$CONFIG_FILE" | grep "^$key[ ]*=" | head -1 | cut -d '=' -f 2- | tr -d ' ' | tr -d '"' | tr -d "'")
        
        if [ -z "$value" ]; then
            echo $default
        else
            echo $value
        fi
    else
        echo $default
    fi
}

# 读取默认参数
DEFAULT_EXCHANGE=$(get_config_value "data" "default_exchange" "binance")
DEFAULT_SYMBOL=$(get_config_value "data" "default_symbol" "BTC/USDT")
DEFAULT_TIMEFRAME=$(get_config_value "data" "default_timeframe" "1h")
DEFAULT_START_DATE=$(get_config_value "data" "default_train_start_date" "2020-01-01")

# 设置默认参数
EXCHANGE="$DEFAULT_EXCHANGE"
START_DATE="$DEFAULT_START_DATE"
END_DATE=$(date +%Y-%m-%d)
TIMEFRAME="$DEFAULT_TIMEFRAME"
SYMBOL="$DEFAULT_SYMBOL"
SKIP_DATA_FETCH=false
FORCE_UPDATE=false
MAX_RETRIES=3

# 支持的时间精度
SUPPORTED_TIMEFRAMES=("1m" "5m" "15m" "30m" "1h" "4h" "1d")

# 显示帮助
show_help() {
    echo -e "${BLUE}BTC交易智能体自动化工作流${NC}"
    echo "此脚本执行从数据获取到模型训练的完整流程"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help                  显示帮助信息"
    echo "  -e, --exchange EXCHANGE     指定交易所 (默认: ${EXCHANGE})"
    echo "  -s, --start-date DATE       指定起始日期 (默认: ${START_DATE}, 格式: YYYY-MM-DD)"
    echo "  -u, --end-date DATE         指定结束日期 (默认: ${END_DATE}, 格式: YYYY-MM-DD)"
    echo "  -t, --timeframe TIMEFRAME   指定时间周期 (默认: ${TIMEFRAME}, 支持: ${SUPPORTED_TIMEFRAMES[*]})"
    echo "  -p, --pair SYMBOL           指定交易对 (默认: ${SYMBOL})"
    echo "  --skip-data-fetch           跳过数据获取步骤 (使用已有数据)"
    echo "  -f, --force                 强制更新数据，即使已有最新数据"
    echo "  -r, --retries NUMBER        API请求失败时的最大重试次数 (默认: ${MAX_RETRIES})"
    echo ""
    echo "示例:"
    echo "  $0 --exchange binance --start-date 2022-01-01 --end-date 2023-01-01 --timeframe 1h"
    echo "  $0 --timeframe 15m --start-date 2023-01-01 --force"
    echo ""
}

# 检查时间精度是否有效
is_valid_timeframe() {
    local tf="$1"
    for valid_tf in "${SUPPORTED_TIMEFRAMES[@]}"; do
        if [ "$tf" = "$valid_tf" ]; then
            return 0
        fi
    done
    return 1
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_help
            exit 0
            ;;
        -e|--exchange)
            EXCHANGE="$2"
            shift 2
            ;;
        -s|--start-date)
            START_DATE="$2"
            shift 2
            ;;
        -u|--end-date)
            END_DATE="$2"
            shift 2
            ;;
        -t|--timeframe)
            if is_valid_timeframe "$2"; then
                TIMEFRAME="$2"
            else
                echo -e "${RED}错误: 不支持的时间周期 '$2'${NC}"
                echo "支持的时间周期: ${SUPPORTED_TIMEFRAMES[*]}"
                exit 1
            fi
            shift 2
            ;;
        -p|--pair)
            SYMBOL="$2"
            shift 2
            ;;
        --skip-data-fetch)
            SKIP_DATA_FETCH=true
            shift
            ;;
        -f|--force)
            FORCE_UPDATE=true
            shift
            ;;
        -r|--retries)
            MAX_RETRIES="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 显示配置
echo -e "${BLUE}=== BTC交易智能体工作流配置 ===${NC}"
echo -e "交易所:     ${GREEN}${EXCHANGE}${NC}"
echo -e "交易对:     ${GREEN}${SYMBOL}${NC}"
echo -e "时间周期:   ${GREEN}${TIMEFRAME}${NC}"
echo -e "开始日期:   ${GREEN}${START_DATE}${NC}"
echo -e "结束日期:   ${GREEN}${END_DATE}${NC}"
echo -e "跳过数据获取: ${GREEN}${SKIP_DATA_FETCH}${NC}"
echo -e "强制更新:   ${GREEN}${FORCE_UPDATE}${NC}"
echo -e "最大重试次数: ${GREEN}${MAX_RETRIES}${NC}"
echo ""

# 确认执行
read -p "是否继续执行? (y/n): " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "已取消执行"
    exit 0
fi

echo -e "${BLUE}=== 开始执行工作流 ===${NC}"

# 检查依赖
echo -e "${YELLOW}正在检查依赖...${NC}"
pip install -q ccxt pandas numpy tqdm

# 步骤1: 数据获取和预处理
echo -e "${YELLOW}步骤 1: 数据获取和预处理${NC}"

RETRY_COUNT=0
SUCCESS=false

if [ "$SKIP_DATA_FETCH" = true ]; then
    echo "跳过数据获取步骤，使用已有数据"
    
    # 构建文件名，基于时间周期
    TIMEFRAME_STR=${TIMEFRAME//m/min}
    TIMEFRAME_STR=${TIMEFRAME_STR//h/hour}
    TIMEFRAME_STR=${TIMEFRAME_STR//d/day}
    DATA_FILE="$WORKSPACE/btc_rl/data/BTC_${TIMEFRAME_STR}.csv"
    
    if [ ! -f "$DATA_FILE" ] && [ "$TIMEFRAME" != "1h" ]; then
        echo -e "${YELLOW}警告: 找不到 $DATA_FILE${NC}"
        
        # 如果找不到特定时间精度的文件，但存在1h的文件，则使用1h文件
        if [ -f "$WORKSPACE/btc_rl/data/BTC_hourly.csv" ]; then
            echo -e "${YELLOW}将使用 BTC_hourly.csv 代替${NC}"
            python -m btc_rl.src.preprocessing --csv "$WORKSPACE/btc_rl/data/BTC_hourly.csv"
        else
            echo -e "${RED}错误: 找不到有效的数据文件${NC}"
            exit 1
        fi
    else
        # 如果是特定时间精度的文件，需要转换为小时级别
        if [ "$TIMEFRAME" != "1h" ] && [ -f "$DATA_FILE" ]; then
            echo -e "${YELLOW}使用 $DATA_FILE 并转换为小时级别数据${NC}"
            python -m btc_rl.src.data_workflow --skip-fetch --csv-path "$DATA_FILE" --timeframe "$TIMEFRAME"
        else
            python -m btc_rl.src.preprocessing --csv "$DATA_FILE"
        fi
    fi
else
    # 数据获取和预处理，带有重试机制
    while [ $RETRY_COUNT -lt $MAX_RETRIES ] && [ "$SUCCESS" = "false" ]; do
        echo -e "${YELLOW}正在获取和处理数据 (尝试 $((RETRY_COUNT+1))/$MAX_RETRIES)...${NC}"
        
        python -m btc_rl.src.data_workflow \
            --exchange "$EXCHANGE" \
            --symbol "$SYMBOL" \
            --timeframe "$TIMEFRAME" \
            --start_date "$START_DATE" \
            --end_date "$END_DATE"
            
        if [ $? -eq 0 ]; then
            SUCCESS=true
        else
            RETRY_COUNT=$((RETRY_COUNT+1))
            if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
                echo -e "${YELLOW}数据获取失败，将在30秒后重试...${NC}"
                sleep 30
            fi
        fi
    done
    
    if [ "$SUCCESS" = "false" ]; then
        echo -e "${RED}错误: 数据获取失败，达到最大重试次数${NC}"
        exit 1
    fi
fi

# 检查是否成功生成训练和测试数据
if [ ! -f "btc_rl/data/train_data.npz" ] || [ ! -f "btc_rl/data/test_data.npz" ]; then
    echo -e "${RED}错误: 未能成功生成训练和测试数据${NC}"
    exit 1
fi

echo -e "${GREEN}数据准备完成!${NC}"

# 步骤2: 准备可视化界面
echo -e "${YELLOW}步骤 2: 准备可视化界面${NC}"
echo "是否要启动可视化界面以实时监控训练过程?"
read -p "启动可视化? (y/n): " start_viz

# 变量，用于存储服务器进程ID
WEBSOCKET_PID=""
HTTP_PID=""

if [[ "$start_viz" == "y" || "$start_viz" == "Y" ]]; then
    # 设置全局错误处理，确保可以看到服务启动的错误
    exec 2>/dev/tty
    
    # 确保可视化目录存在
    mkdir -p "$WORKSPACE/btc_rl/visualizer"
    
    # 检查index.html是否存在
    if [ ! -f "$WORKSPACE/btc_rl/visualizer/index.html" ]; then
        echo -e "${YELLOW}警告: visualizer/index.html 不存在，创建一个简单的页面...${NC}"
        cat > "$WORKSPACE/btc_rl/visualizer/index.html" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>BTC交易智能体 - 可视化</title>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .chart { margin-bottom: 20px; height: 400px; border: 1px solid #ccc; }
        .status { padding: 10px; background-color: #f0f0f0; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>BTC交易智能体 - 实时监控</h1>
    <div class="status">
        <h2>连接状态</h2>
        <p id="connection">正在连接到服务器...</p>
    </div>
    <div class="chart">
        <h2>交易数据将显示在这里</h2>
        <p>请确保WebSocket服务器正在运行</p>
    </div>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // 尝试连接WebSocket
            const connectWebSocket = () => {
                const ws = new WebSocket('ws://localhost:8765');
                const connStatus = document.getElementById('connection');
                
                ws.onopen = () => {
                    connStatus.textContent = '已连接到服务器';
                    connStatus.style.color = 'green';
                };
                
                ws.onclose = () => {
                    connStatus.textContent = '与服务器断开连接，5秒后重试...';
                    connStatus.style.color = 'red';
                    setTimeout(connectWebSocket, 5000);
                };
                
                ws.onerror = (error) => {
                    console.error('WebSocket错误:', error);
                    connStatus.textContent = '连接出错，请检查服务器是否运行';
                    connStatus.style.color = 'red';
                };
                
                ws.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        console.log('收到数据:', data);
                        // 这里可以添加处理数据和更新图表的代码
                    } catch (e) {
                        console.error('解析消息出错:', e);
                    }
                };
            };
            
            // 启动连接
            connectWebSocket();
        });
    </script>
</body>
</html>
EOF
        echo -e "${GREEN}创建了简单的可视化页面${NC}"
    fi

    echo -e "${YELLOW}正在启动可视化服务器...${NC}"
    
    # 启动WebSocket服务器
    echo -e "${YELLOW}🔌 启动WebSocket服务器...${NC}"
    python -m btc_rl.src.websocket_server > /dev/null 2>&1 &
    WEBSOCKET_PID=$!

    # 稍等片刻确保WebSocket服务器启动
    sleep 2

    # 检查WebSocket服务器是否成功启动
    if ! ps -p $WEBSOCKET_PID > /dev/null; then
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
    
    # 验证服务器是否正常运行
    if ps -p $WEBSOCKET_PID > /dev/null && ps -p $HTTP_PID > /dev/null; then
        echo -e "${GREEN}两个服务器都已成功启动${NC}"
        echo -e "${GREEN}可视化界面已启动，请在浏览器中访问 http://localhost:8080/index.html${NC}"
        echo -e "${YELLOW}注意：训练开始后，可视化界面将实时更新${NC}"
    else
        echo -e "${RED}服务器启动失败，请检查日志${NC}"
        if ! ps -p $WEBSOCKET_PID > /dev/null; then
            echo -e "${RED}WebSocket服务器未正常运行${NC}"
        fi
        if ! ps -p $HTTP_PID > /dev/null; then
            echo -e "${RED}HTTP服务器未正常运行${NC}"
        fi
        # 尝试清理可能残留的进程
        kill $WEBSOCKET_PID $HTTP_PID 2>/dev/null || true
        WEBSOCKET_PID=""
        HTTP_PID=""
    fi
else
    echo "跳过可视化步骤"
fi

# 步骤3: 训练模型
echo -e "${YELLOW}步骤 3: 训练模型${NC}"
echo "是否要开始训练模型? 这可能需要相当长的时间。"
read -p "开始训练? (y/n): " start_training
if [[ "$start_training" == "y" || "$start_training" == "Y" ]]; then
    # 告知用户可视化界面的URL
    if [ ! -z "$WEBSOCKET_PID" ] && [ ! -z "$HTTP_PID" ]; then
        echo -e "${GREEN}✅ 服务已成功启动${NC}"
        echo -e "💡 请在浏览器中访问: ${BLUE}http://localhost:8080/index.html${NC}"
    fi
    
    # 临时恢复标准错误以便我们能看到训练进度
    exec 2>/dev/tty

    echo -e "${YELLOW}🧠 启动模型训练与WebSocket服务...${NC}"
    echo -e "💡 按 Ctrl+C 可以随时优雅地停止训练和服务..."
    
    # 使用与start.sh相同的方式启动训练过程
    stdbuf -oL -eL python -m btc_rl.src.run_wrapper 2> >(grep -v "Exception\|Error\|Traceback\|Broken" >&2)
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ 模型训练失败${NC}"
        cleanup
        exit 1
    fi
    
    echo -e "${GREEN}✅ 模型训练完成!${NC}"
else
    echo "跳过模型训练步骤"
fi

# 如果可视化服务在运行，询问是否要关闭
if [ ! -z "$WEBSOCKET_PID" ] || [ ! -z "$HTTP_PID" ]; then
    echo -e "${YELLOW}可视化服务仍在运行${NC}"
    read -p "是否要关闭可视化服务? (y/n): " stop_viz
    if [[ "$stop_viz" == "y" || "$stop_viz" == "Y" ]]; then
        echo "✅ 服务将被关闭..."
        cleanup
    else
        echo -e "${YELLOW}可视化服务将继续运行，您可以访问 http://localhost:8080/index.html${NC}"
        echo -e "${YELLOW}按 Ctrl+C 可在终端中终止服务${NC}"
        
        # 等待用户中断
        trap cleanup INT
        wait
    fi
fi

echo -e "${GREEN}✅ 工作流程执行完毕!${NC}"
