#!/bin/bash
# 对黄金法则选出的最佳BTC交易模型进行多环境回测验证

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR" && pwd )"

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # 无颜色

# 默认参数
MODEL_PATH=""
MODEL_NAME=""
START_DATE=""
END_DATE=""
EXCHANGE="binance"
SYMBOL="BTC/USDT"
TIMEFRAME="1h"
SKIP_DATA_FETCH=0
FULL_OUTPUT=0
GENERATE_REPORT=0
HELP=0
VERBOSE=0

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            if [[ $2 == *"/"* ]]; then
                # 如果包含路径分隔符，认为是模型路径
                MODEL_PATH=$2
                # 从路径中提取文件名
                MODEL_NAME=$(basename "$MODEL_PATH" .zip)
            else
                # 否则认为是模型名称
                MODEL_NAME=$2
                # 构建路径
                MODEL_PATH="$PROJECT_ROOT/btc_rl/models/${MODEL_NAME}.zip"
            fi
            shift 2
            ;;
        --start-date)
            START_DATE=$2
            shift 2
            ;;
        --end-date)
            END_DATE=$2
            shift 2
            ;;
        --exchange)
            EXCHANGE=$2
            shift 2
            ;;
        --symbol)
            SYMBOL=$2
            shift 2
            ;;
        --timeframe)
            TIMEFRAME=$2
            shift 2
            ;;
        --skip-data-fetch)
            SKIP_DATA_FETCH=1
            shift
            ;;
        --full)
            FULL_OUTPUT=1
            shift
            ;;
        --report)
            GENERATE_REPORT=1
            shift
            ;;
        --verbose|-v)
            VERBOSE=1
            shift
            ;;
        --help|-h)
            HELP=1
            shift
            ;;
        *)
            # 未知参数
            echo "未知参数: $1"
            HELP=1
            shift
            ;;
    esac
done

# 显示帮助信息
if [ $HELP -eq 1 ]; then
    echo "==================================================="
    echo "BTC交易模型多环境回测工具 v1.1"
    echo "==================================================="
    echo "该工具用于对模型进行多环境回测验证，包括上涨、下跌、震荡和反弹市场环境"
    echo "现在支持指定时间段的回测分析"
    echo
    echo "用法: $0 [选项]"
    echo
    echo "选项:"
    echo "  --model PATH/NAME   指定要回测的模型路径或模型名称，默认使用黄金法则选出的最佳模型"
    echo "  --start-date DATE   指定回测起始日期 (格式: YYYY-MM-DD)"
    echo "  --end-date DATE     指定回测结束日期 (格式: YYYY-MM-DD)"
    echo "  --exchange NAME     指定获取数据的交易所 (默认: binance)"
    echo "  --symbol PAIR       指定交易对 (默认: BTC/USDT)"
    echo "  --timeframe PERIOD  指定时间周期 (默认: 1h)"
    echo "  --skip-data-fetch   跳过数据获取，使用可用数据"
    echo "  --full              输出详细回测结果和性能指标"
    echo "  --report            生成完整回测报告并保存为HTML格式"
    echo "  --verbose           显示详细输出"
    echo "  --help              显示此帮助信息"
    echo
    echo "示例:"
    echo "  $0                                                # 使用黄金法则选出的最佳模型进行回测"
    echo "  $0 --model \"sac_ep12\"                             # 指定模型名称进行回测"
    echo "  $0 --model \"btc_rl/models/sac_ep12.zip\"           # 指定模型完整路径进行回测"
    echo "  $0 --start-date \"2023-01-01\" --end-date \"2023-12-31\" # 指定时间段回测"
    echo "  $0 --exchange \"binance\" --symbol \"ETH/USDT\" --timeframe \"4h\" # 指定其它交易对和时间周期" 
    echo "  $0 --start-date \"2024-01-01\" --end-date \"2024-06-01\" --full --report # 生成详细回测报告"
    echo "==================================================="
    exit 0
fi

# 打印欢迎信息
echo "==================================================="
echo "BTC交易模型多环境回测验证工具"
echo "==================================================="

# 确保激活了虚拟环境（如果存在）
if [ -f venv/bin/activate ]; then
    echo "激活虚拟环境..."
    source venv/bin/activate
fi

# 构建命令行参数
CMD_ARGS=""
if [ ! -z "$MODEL_PATH" ]; then
    CMD_ARGS="$CMD_ARGS --model \"$MODEL_PATH\""
    echo "使用指定模型: $MODEL_PATH"
else
    echo "使用黄金法则选出的最佳模型"
fi

# 添加日期范围参数
if [ ! -z "$START_DATE" ]; then
    CMD_ARGS="$CMD_ARGS --start-date \"$START_DATE\""
    echo "设置回测起始日期: $START_DATE"
fi

if [ ! -z "$END_DATE" ]; then
    CMD_ARGS="$CMD_ARGS --end-date \"$END_DATE\""
    echo "设置回测结束日期: $END_DATE"
fi

# 添加交易所和交易对参数
CMD_ARGS="$CMD_ARGS --exchange \"$EXCHANGE\""
echo "设置交易所: $EXCHANGE"

CMD_ARGS="$CMD_ARGS --symbol \"$SYMBOL\""
echo "设置交易对: $SYMBOL"

CMD_ARGS="$CMD_ARGS --timeframe \"$TIMEFRAME\""
echo "设置时间周期: $TIMEFRAME"

# 添加是否跳过数据获取的参数
if [ $SKIP_DATA_FETCH -eq 1 ]; then
    CMD_ARGS="$CMD_ARGS --skip-data-fetch"
    echo "跳过数据获取，使用可用数据"
fi

# 添加详细输出和报告选项
if [ $FULL_OUTPUT -eq 1 ]; then
    # 不传递--full给model_backtest.py，仅在脚本内处理
    echo -e "${BLUE}启用详细指标分析${NC}"
fi

if [ $GENERATE_REPORT -eq 1 ]; then
    # 不传递--report给model_backtest.py，仅在脚本内处理
    echo -e "${BLUE}将生成完整回测报告${NC}"
fi

# 运行Python模块
cd "$PROJECT_ROOT"
PYTHON_CMD="python -m btc_rl.src.model_backtest $CMD_ARGS"

if [ $VERBOSE -eq 1 ]; then
    echo -e "${YELLOW}执行命令: $PYTHON_CMD${NC}"
    eval $PYTHON_CMD
else
    echo -e "${GREEN}开始多环境回测分析...${NC}"
    eval $PYTHON_CMD
fi

EXIT_CODE=$?

# 处理回测报告
if [ $EXIT_CODE -eq 0 ] && [ $GENERATE_REPORT -eq 1 ]; then
    # 获取当前日期时间作为报告标识
    REPORT_DATE=$(date +"%Y%m%d_%H%M%S")
    REPORT_DIR="$PROJECT_ROOT/btc_rl/reports"
    
    # 确保报告目录存在
    mkdir -p "$REPORT_DIR"
    
    # 确定模型名称
    if [ -z "$MODEL_NAME" ]; then
        MODEL_NAME=$(python -c "
import json
from btc_rl.src.model_comparison import get_best_model_by_golden_rule
model_info = get_best_model_by_golden_rule()
if model_info:
    print(model_info['model_name'])
")
    fi
    
    # 生成报告名称
    REPORT_NAME="${MODEL_NAME}_${REPORT_DATE}"
    
    # 仅生成HTML报告
    echo -e "${BLUE}生成回测报告...${NC}"
    python -m btc_rl.src.report_generator \
        --model "$MODEL_NAME" \
        --output "$REPORT_DIR/${REPORT_NAME}.html" \
        --include-trades \
        --include-equity-curve \
        --include-drawdowns
    
    # 检查报告是否成功生成
    if [ -f "$REPORT_DIR/${REPORT_NAME}.html" ]; then
        echo -e "${GREEN}✅ 回测报告已生成:${NC}"
        echo -e "   - HTML: ${BLUE}$REPORT_DIR/${REPORT_NAME}.html${NC}"
        
        # 如果是在Linux图形环境中，尝试打开HTML报告
        if [ -n "$DISPLAY" ] && command -v xdg-open &> /dev/null; then
            read -p "是否打开HTML回测报告? (y/n): " open_report
            if [[ "$open_report" == "y" || "$open_report" == "Y" ]]; then
                xdg-open "$REPORT_DIR/${REPORT_NAME}.html" &> /dev/null &
            fi
        fi
    else
        echo -e "${RED}❌ 回测报告生成失败${NC}"
    fi
fi

echo -e "${BLUE}==================================================${NC}"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ 回测分析完成!${NC}"
else
    echo -e "${RED}❌ 回测分析失败，请检查错误信息${NC}"
fi
echo -e "${BLUE}==================================================${NC}"

exit $EXIT_CODE
