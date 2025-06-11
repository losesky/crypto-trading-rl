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

# 定义不会导致脚本退出的清理函数变体
cleanup_continue() {
    echo -e "\n🛑 正在关闭服务，但保持脚本运行..."
    
    # 关闭HTTP服务器
    if [ -n "$HTTP_PID" ] && ps -p $HTTP_PID &>/dev/null; then
        echo "关闭HTTP服务器 (PID: $HTTP_PID)..."
        kill -9 $HTTP_PID &>/dev/null || true
        wait $HTTP_PID 2>/dev/null || true
        HTTP_PID=""
    fi
    
    # 关闭WebSocket服务器
    if [ -n "$WEBSOCKET_PID" ] && ps -p $WEBSOCKET_PID &>/dev/null; then
        echo "关闭WebSocket服务器 (PID: $WEBSOCKET_PID)..."
        kill -9 $WEBSOCKET_PID &>/dev/null || true
        wait $WEBSOCKET_PID 2>/dev/null || true
        WEBSOCKET_PID=""
    fi
    
    # 确保没有训练相关的Python进程在后台
    echo "正在检查是否有残留的训练进程..."
    pkill -f "python.*btc_rl.src.run_wrapper" &>/dev/null || true
    
    echo "✅ 服务已关闭，继续执行脚本..."
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
    
    # 使用临时文件捕获输出，但避免复杂的管道处理
    TEMP_OUTPUT=$(mktemp)
    echo -e "${YELLOW}将训练输出保存在临时文件: $TEMP_OUTPUT${NC}"
    
    # 直接运行训练程序，将标准输出重定向到临时文件，同时保持错误输出到终端
    python -m btc_rl.src.run_wrapper 2>&1 | tee $TEMP_OUTPUT
    
    # 检查训练结果
    TRAIN_STATUS=${PIPESTATUS[0]}
    
    # 检查训练输出，增加对"所有模型训练和评估完成"这一消息的检测
    if [ $TRAIN_STATUS -ne 0 ]; then
        echo -e "${RED}❌ 模型训练失败 (退出状态: $TRAIN_STATUS)${NC}"
        rm -f $TEMP_OUTPUT
        if [ ! -z "$WEBSOCKET_PID" ] || [ ! -z "$HTTP_PID" ]; then
            cleanup_continue
        fi
        
        echo -e "${YELLOW}⚠️ 尽管训练退出状态码非零，我们将尝试继续执行后续步骤...${NC}"
        echo -e "${YELLOW}⚠️ 如果后续步骤失败，请检查训练日志${NC}"
    fi
    
    # 检查是否包含训练完成的消息
    echo -e "${YELLOW}正在检查训练是否完成...${NC}"
    TRAIN_COMPLETED=false
    
    if grep -q "训练已结束\|✅ 训练已结束" $TEMP_OUTPUT; then
        echo -e "${GREEN}✅ 检测到训练已正常完成${NC}"
        TRAIN_COMPLETED=true
    elif grep -q "模型训练完成\|所有模型训练和评估完成" $TEMP_OUTPUT; then
        echo -e "${GREEN}✅ 检测到训练已正常完成${NC}"
        TRAIN_COMPLETED=true
    else
        echo -e "${YELLOW}⚠️ 未检测到训练完成信息，但进程已结束${NC}"
        echo -e "${YELLOW}⚠️ 继续执行后续步骤，但请注意检查训练结果${NC}"
    fi
    
    # 删除临时文件并继续
    echo -e "${GREEN}✅ 模型训练过程已结束!${NC}"
    rm -f $TEMP_OUTPUT
    
    # 强制确保脚本继续执行
    echo -e "${GREEN}➡️ 正在准备进入下一步骤: 模型分析和评估${NC}"
    echo -e "${YELLOW}3 秒后继续执行...${NC}"
    
    # 强制刷新输出缓冲区并确保进程正常继续
    sleep 1
    echo -e "${YELLOW}2 秒后继续执行...${NC}"
    sleep 1
    echo -e "${YELLOW}1 秒后继续执行...${NC}"
    sleep 1
    echo -e "${GREEN}继续执行！${NC}"
else
    echo "跳过模型训练步骤"
fi

# 如果可视化服务在运行，询问是否要关闭
if [ ! -z "$WEBSOCKET_PID" ] || [ ! -z "$HTTP_PID" ]; then
    echo -e "${YELLOW}可视化服务仍在运行${NC}"
    read -p "是否要关闭可视化服务? (y/n): " stop_viz
    if [[ "$stop_viz" == "y" || "$stop_viz" == "Y" ]]; then
        echo "✅ 服务将被关闭但脚本继续执行..."
        cleanup_continue
    else
        echo -e "${YELLOW}可视化服务将继续运行，您可以在浏览器中访问 http://localhost:8080/index.html${NC}"
        echo -e "${YELLOW}脚本将继续执行后续步骤，服务会保持在后台运行${NC}"
        echo -e "${YELLOW}完成整个流程后，您可以选择是否关闭服务${NC}"
    fi
fi

# 添加检查点确认流程可以继续
echo -e "${GREEN}➡️ 准备进入模型评估阶段...${NC}"
sleep 1

# 步骤4: 分析和评估模型性能
echo -e "${YELLOW}步骤 4: 分析模型性能和选择最佳模型${NC}"
read -p "是否要分析模型性能并选择最佳模型? (y/n): " analyze_models
if [[ "$analyze_models" == "y" || "$analyze_models" == "Y" ]]; then
    echo -e "${BLUE}🔍 正在分析模型性能指标...${NC}"
    
    # 检查模型文件是否存在
    MODEL_COUNT=$(ls btc_rl/models/*.zip 2>/dev/null | wc -l)
    if [ "$MODEL_COUNT" -eq 0 ]; then
        echo -e "${RED}❌ 错误: 没有找到训练好的模型文件!${NC}"
        echo -e "${YELLOW}请确保训练步骤已经完成并正确生成了模型文件${NC}"
        exit 1
    else
        echo -e "${GREEN}✓ 检测到${MODEL_COUNT}个模型文件${NC}"
    fi
    
    # 临时恢复标准错误
    exec 2>/dev/tty
    
    # 设置分析选项
    ANALYZE_OPTIONS="--evaluate --full"
    
    # 根据用户配置提取回撤阈值
    MAX_DD_VALUE=$(get_config_value "model_selection" "maximum_drawdown" "0.35")
    # 转换为百分比格式
    MAX_DD_VALUE=$(echo "$MAX_DD_VALUE" | awk '{print $1+0}')
    
    # 获取最低夏普比率和索提诺比率
    MIN_SHARPE_VALUE=$(get_config_value "model_selection" "minimum_sharpe" "5.0")
    MIN_SORTINO_VALUE=$(echo "$MIN_SHARPE_VALUE * 4" | bc) # 通常索提诺比率选择夏普比率的4倍作为基准
    
    # 运行分析脚本
    echo -e "${BLUE}📊 运行模型分析...${NC}"
    ./analyze_metrics.sh $ANALYZE_OPTIONS --max-dd $MAX_DD_VALUE --min-sharpe $MIN_SHARPE_VALUE --min-sortino $MIN_SORTINO_VALUE
    
    echo -e "\n${BLUE}🏆 选择最佳模型（黄金法则评分）...${NC}"
    python select_best_model.py
    
    # 存储最佳模型信息
    BEST_MODEL_INFO=$(python -c "
import json
from btc_rl.src.model_comparison import get_best_model_by_golden_rule
model_info = get_best_model_by_golden_rule()
if model_info:
    print(model_info['model_name'])
    print(model_info['model_path'])
")
    
    BEST_MODEL_NAME=$(echo "$BEST_MODEL_INFO" | head -1)
    BEST_MODEL_PATH=$(echo "$BEST_MODEL_INFO" | tail -1)
    
    if [ -n "$BEST_MODEL_NAME" ] && [ -n "$BEST_MODEL_PATH" ] && [ -f "$BEST_MODEL_PATH" ]; then
        echo -e "${GREEN}✅ 最佳模型: $BEST_MODEL_NAME${NC}"
        echo -e "${GREEN}✅ 模型路径: $BEST_MODEL_PATH${NC}"
        
        # 询问是否要备份最佳模型
        read -p "是否要备份最佳模型? (y/n): " backup_model
        if [[ "$backup_model" == "y" || "$backup_model" == "Y" ]]; then
            BACKUP_DIR="$WORKSPACE/btc_rl/models/best_model"
            BACKUP_DATE=$(date +"%Y%m%d_%H%M%S")
            BACKUP_PATH="$BACKUP_DIR/${BEST_MODEL_NAME}_${BACKUP_DATE}.zip"
            
            # 确保备份目录存在
            mkdir -p "$BACKUP_DIR"
            
            # 复制模型文件
            cp "$BEST_MODEL_PATH" "$BACKUP_PATH"
            echo -e "${GREEN}✅ 最佳模型已备份至: $BACKUP_PATH${NC}"
        fi
        
        # 询问是否要回测最佳模型
        read -p "是否要回测最佳模型性能? (y/n): " backtest_model
        if [[ "$backtest_model" == "y" || "$backtest_model" == "Y" ]]; then
            echo -e "${YELLOW}🧪 开始回测最佳模型...${NC}"
            ./backtest_best_model.sh --model "$BEST_MODEL_NAME" --full --report
            echo -e "${GREEN}✅ 回测完成!${NC}"
        fi
    else
        echo -e "${RED}❌ 未能识别最佳模型${NC}"
    fi
else
    echo "跳过模型分析步骤"
fi

# 步骤5: 清理环境和进程
echo -e "${YELLOW}步骤 5: 清理环境${NC}"
read -p "是否要关闭所有服务并清理环境? (y/n): " cleanup_env
if [[ "$cleanup_env" == "y" || "$cleanup_env" == "Y" ]]; then
    echo -e "${YELLOW}🧹 清理环境并关闭所有进程...${NC}"
    # 这里使用cleanup函数，因为这是脚本最后阶段，可以安全退出
    cleanup
    exit 0  # 确保脚本在此处终止
else
    echo -e "${YELLOW}⚠️ 环境未完全清理，部分进程可能仍在运行${NC}"
    echo -e "${YELLOW}如需手动清理，请运行以下命令:${NC}"
    echo "pkill -f \"python.*btc_rl.src\""
    echo "fuser -k -9 8080/tcp 8765/tcp"
fi

echo -e "${GREEN}✅ 工作流程执行完毕!${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "${GREEN}BTC交易强化学习系统工作流程总结:${NC}"
echo -e "${BLUE}=================================================================${NC}"

# 打印流程总结
if [ -n "$BEST_MODEL_NAME" ] && [ -n "$BEST_MODEL_PATH" ] && [ -f "$BEST_MODEL_PATH" ]; then
    echo -e "📊 ${YELLOW}最佳交易模型:${NC} $BEST_MODEL_NAME"
    echo -e "📂 ${YELLOW}模型文件位置:${NC} $BEST_MODEL_PATH"
    
    # 调用Python脚本获取模型关键指标
    MODEL_METRICS=$(python -c "
import json
from btc_rl.src.model_comparison import get_best_model_by_golden_rule
model_info = get_best_model_by_golden_rule()
if model_info:
    metrics = {
        'score': model_info.get('golden_rule_score', 0),
        'return': model_info.get('total_return', 0),
        'drawdown': model_info.get('max_drawdown', 0),
        'sharpe': model_info.get('sharpe_ratio', 0),
        'winrate': model_info.get('win_rate', 0),
        'equity': model_info.get('final_equity', 0)
    }
    print(json.dumps(metrics))
")
    
    if [ -n "$MODEL_METRICS" ]; then
        # 解析模型指标
        MODEL_SCORE=$(echo "$MODEL_METRICS" | python -c "import json,sys; print(json.load(sys.stdin).get('score', 0))")
        MODEL_RETURN=$(echo "$MODEL_METRICS" | python -c "import json,sys; print(json.load(sys.stdin).get('return', 0))")
        MODEL_DRAWDOWN=$(echo "$MODEL_METRICS" | python -c "import json,sys; print(json.load(sys.stdin).get('drawdown', 0))")
        MODEL_SHARPE=$(echo "$MODEL_METRICS" | python -c "import json,sys; print(json.load(sys.stdin).get('sharpe', 0))")
        MODEL_WINRATE=$(echo "$MODEL_METRICS" | python -c "import json,sys; print(json.load(sys.stdin).get('winrate', 0))")
        MODEL_EQUITY=$(echo "$MODEL_METRICS" | python -c "import json,sys; print(json.load(sys.stdin).get('equity', 0))")
        
        # 格式化百分比和货币值
        MODEL_RETURN_PCT=$(printf "%.2f%%" $(echo "$MODEL_RETURN * 100" | bc))
        MODEL_DRAWDOWN_PCT=$(printf "%.2f%%" $(echo "$MODEL_DRAWDOWN * 100" | bc))
        MODEL_WINRATE_PCT=$(printf "%.2f%%" $(echo "$MODEL_WINRATE * 100" | bc))
        MODEL_EQUITY_FMT=$(printf "$%.2f" $MODEL_EQUITY)
        
        echo -e "📈 ${YELLOW}综合评分:${NC} ${GREEN}${MODEL_SCORE}${NC}"
        echo -e "💰 ${YELLOW}最终权益:${NC} ${GREEN}${MODEL_EQUITY_FMT}${NC}"
        echo -e "📈 ${YELLOW}总回报率:${NC} ${GREEN}${MODEL_RETURN_PCT}${NC}"
        echo -e "📉 ${YELLOW}最大回撤:${NC} ${RED}${MODEL_DRAWDOWN_PCT}${NC}"
        echo -e "📊 ${YELLOW}夏普比率:${NC} ${GREEN}${MODEL_SHARPE}${NC}"
        echo -e "🎯 ${YELLOW}胜率:${NC} ${GREEN}${MODEL_WINRATE_PCT}${NC}"
    else
        echo -e "${RED}⚠️ 未能加载模型详细指标${NC}"
    fi
else
    echo -e "${RED}⚠️ 未能确定最佳模型${NC}"
fi

echo -e "${BLUE}=================================================================${NC}"
echo -e "${GREEN}感谢使用BTC交易强化学习系统!${NC}"
echo -e "${BLUE}=================================================================${NC}"
