#!/bin/bash
# 定期更新比特币数据脚本
# 建议添加到crontab以实现自动更新，例如：
# 0 0 * * 1 /home/losesky/crypto-trading-rl/update_data.sh

# 定义颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # 无颜色

# 工作目录
WORKSPACE="/home/losesky/crypto-trading-rl"
cd $WORKSPACE

# 日志文件
LOG_DIR="$WORKSPACE/btc_rl/logs"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/data_updates.log"
touch $LOG_FILE

# 日志函数
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> $LOG_FILE
    echo -e "$1"
}

# 加载配置
CONFIG_FILE="$WORKSPACE/config.ini"
if [ ! -f "$CONFIG_FILE" ]; then
    log_message "${RED}错误: 找不到配置文件 $CONFIG_FILE${NC}"
    exit 1
fi

# 尝试从配置文件读取参数
get_config_value() {
    section=$1
    key=$2
    default=$3
    
    value=$(grep -A 20 "^\[$section\]" "$CONFIG_FILE" | grep "^$key[ ]*=" | head -1 | cut -d '=' -f 2- | tr -d ' ' | tr -d '"' | tr -d "'")
    
    if [ -z "$value" ]; then
        echo $default
    else
        echo $value
    fi
}

# 读取配置
EXCHANGE=$(get_config_value "data" "default_exchange" "binance")
SYMBOL=$(get_config_value "data" "default_symbol" "BTC/USDT")
TIMEFRAME=$(get_config_value "data" "default_timeframe" "1h")

# 支持的时间周期列表
SUPPORTED_TIMEFRAMES=("1m" "5m" "15m" "30m" "1h" "4h" "1d")

# 检查时间周期是否受支持
timeframe_supported=false
for tf in "${SUPPORTED_TIMEFRAMES[@]}"; do
    if [ "$TIMEFRAME" == "$tf" ]; then
        timeframe_supported=true
        break
    fi
done

if [ "$timeframe_supported" == "false" ]; then
    log_message "${YELLOW}警告: 配置的时间周期 '$TIMEFRAME' 不受支持，将使用默认值 '1h'${NC}"
    TIMEFRAME="1h"
fi

log_message "${BLUE}开始更新比特币数据...${NC}"
log_message "交易所: ${GREEN}$EXCHANGE${NC}"
log_message "交易对: ${GREEN}$SYMBOL${NC}"
log_message "时间周期: ${GREEN}$TIMEFRAME${NC}"

# 备份当前数据
BACKUP_DIR="$WORKSPACE/btc_rl/data/backups/$(date '+%Y%m%d')"
mkdir -p $BACKUP_DIR

# 构建文件名模式
TIMEFRAME_STR=${TIMEFRAME//m/min}
TIMEFRAME_STR=${TIMEFRAME_STR//h/hour}
TIMEFRAME_STR=${TIMEFRAME_STR//d/day}
DATA_FILE="$WORKSPACE/btc_rl/data/BTC_${TIMEFRAME_STR}.csv"

# 备份现有数据
if [ -f "$DATA_FILE" ]; then
    cp "$DATA_FILE" "$BACKUP_DIR/"
    log_message "${GREEN}已备份 $DATA_FILE 到 $BACKUP_DIR${NC}"
fi

# 同时备份1小时级别数据和训练/测试数据集
if [ -f "$WORKSPACE/btc_rl/data/BTC_hourly.csv" ]; then
    cp "$WORKSPACE/btc_rl/data/BTC_hourly.csv" "$BACKUP_DIR/" 
    log_message "${GREEN}已备份 BTC_hourly.csv 到 $BACKUP_DIR${NC}"
fi

if [ -f "$WORKSPACE/btc_rl/data/train_data.npz" ]; then
    cp "$WORKSPACE/btc_rl/data/train_data.npz" "$BACKUP_DIR/"
    cp "$WORKSPACE/btc_rl/data/test_data.npz" "$BACKUP_DIR/" 2>/dev/null || true
    log_message "${GREEN}已备份训练和测试数据到 $BACKUP_DIR${NC}"
fi

# 确定更新的日期范围
LAST_DATE=""

# 尝试从特定时间周期的文件中获取最后一个日期
if [ -f "$DATA_FILE" ]; then
    # 尝试提取最后一个日期（假设格式为ISO）
    LAST_DATE=$(tail -n 1 "$DATA_FILE" | cut -d',' -f1)
    
    # 如果提取成功，将其转换为YYYY-MM-DD格式
    if [ ! -z "$LAST_DATE" ]; then
        # 尝试处理ISO格式的时间戳
        LAST_DATE=$(date -d "$LAST_DATE" '+%Y-%m-%d' 2>/dev/null || echo "")
    fi
fi

# 如果从指定时间周期文件无法获取日期，尝试从小时级别文件获取
if [ -z "$LAST_DATE" ] && [ -f "$WORKSPACE/btc_rl/data/BTC_hourly.csv" ]; then
    LAST_DATE=$(tail -n 1 "$WORKSPACE/btc_rl/data/BTC_hourly.csv" | cut -d',' -f1)
    
    if [ ! -z "$LAST_DATE" ]; then
        LAST_DATE=$(date -d "$LAST_DATE" '+%Y-%m-%d' 2>/dev/null || echo "")
    fi
fi

# 如果无法提取日期或文件不存在，使用3个月前的日期作为起点
if [ -z "$LAST_DATE" ]; then
    LAST_DATE=$(date -d "3 months ago" '+%Y-%m-%d')
    log_message "${YELLOW}无法确定现有数据的最后日期，将使用 $LAST_DATE 作为起点${NC}"
else
    log_message "${BLUE}检测到现有数据最后日期: $LAST_DATE${NC}"
    # 修改为最后日期后一天，避免重复
    LAST_DATE=$(date -d "$LAST_DATE +1 day" '+%Y-%m-%d')
fi

TODAY=$(date '+%Y-%m-%d')

# 检查是否需要更新
if [ "$LAST_DATE" == "$TODAY" ]; then
    log_message "${GREEN}数据已是最新，无需更新${NC}"
    exit 0
fi

log_message "${BLUE}将获取从 $LAST_DATE 到 $TODAY 的比特币数据...${NC}"

# 设置最大重试次数
MAX_RETRIES=3
RETRY_COUNT=0
SUCCESS=false

while [ $RETRY_COUNT -lt $MAX_RETRIES ] && [ "$SUCCESS" == "false" ]; do
    # 运行数据获取和预处理工作流
    python -m btc_rl.src.data_workflow \
        --exchange "$EXCHANGE" \
        --symbol "$SYMBOL" \
        --timeframe "$TIMEFRAME" \
        --start_date "$LAST_DATE" \
        --end_date "$TODAY" > >(tee -a $LOG_FILE) 2>&1
    
    if [ $? -eq 0 ]; then
        SUCCESS=true
    else
        RETRY_COUNT=$((RETRY_COUNT+1))
        log_message "${YELLOW}数据更新失败 (尝试 $RETRY_COUNT/$MAX_RETRIES)，将在30秒后重试...${NC}"
        sleep 30
    fi
done

if [ "$SUCCESS" == "true" ]; then
    log_message "${GREEN}✅ 数据更新成功!${NC}"
    
    # 验证更新后的文件是否存在
    if [ ! -f "$WORKSPACE/btc_rl/data/train_data.npz" ] || [ ! -f "$WORKSPACE/btc_rl/data/test_data.npz" ]; then
        log_message "${YELLOW}警告: 未能找到生成的训练或测试数据文件${NC}"
    else
        log_message "${GREEN}已成功生成训练和测试数据文件${NC}"
    fi
    
    # 打印数据摘要
    if [ -f "$WORKSPACE/btc_rl/data/BTC_hourly.csv" ]; then
        TOTAL_ROWS=$(wc -l < "$WORKSPACE/btc_rl/data/BTC_hourly.csv")
        TOTAL_ROWS=$((TOTAL_ROWS-1))  # 减去标题行
        FIRST_DATE=$(head -2 "$WORKSPACE/btc_rl/data/BTC_hourly.csv" | tail -1 | cut -d',' -f1)
        LAST_DATE=$(tail -1 "$WORKSPACE/btc_rl/data/BTC_hourly.csv" | cut -d',' -f1)
        
        log_message "${BLUE}数据摘要:${NC}"
        log_message "总数据点: ${GREEN}$TOTAL_ROWS${NC}"
        log_message "数据范围: ${GREEN}$FIRST_DATE${NC} 至 ${GREEN}$LAST_DATE${NC}"
    fi
    
    # 可选：在这里添加自动训练新模型的命令
    # 可以根据需要取消注释下面的命令
    # log_message "${BLUE}开始使用新数据训练模型...${NC}"
    # python -m btc_rl.src.train_sac >> $LOG_FILE 2>&1
else
    log_message "${RED}❌ 达到最大重试次数 ($MAX_RETRIES)，数据更新失败${NC}"
    log_message "${RED}请检查日志文件 $LOG_FILE 获取更多信息${NC}"
    exit 1
fi

exit 0
