#!/bin/bash
# 模型指标分析工具脚本
# 用于显示和比较不同模型的指标，包括Sharpe比率、Sortino比率、交易数量和胜率等

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR" && pwd )"

# 默认参数
SHOW_FULL=0
PLOT=0
EVALUATE=0
FIX_WINRATE=0
HELP=0
VERBOSE=0

# 风控阈值参数，默认值
MAX_DD=0.05
MIN_SORTINO=25
MIN_SHARPE=12

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            SHOW_FULL=1
            shift
            ;;
        --plot)
            PLOT=1
            shift
            ;;
        --evaluate)
            EVALUATE=1
            shift
            ;;
        --fix-winrate)
            FIX_WINRATE=1
            shift
            ;;
        --max-dd)
            MAX_DD=$2
            shift 2
            ;;
        --min-sortino)
            MIN_SORTINO=$2
            shift 2
            ;;
        --min-sharpe)
            MIN_SHARPE=$2
            shift 2
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
    echo "模型指标分析工具 v1.0"
    echo "==================================================="
    echo "该工具用于分析和比较强化学习交易模型的性能指标"
    echo
    echo "用法: $0 [选项]"
    echo
    echo "选项:"
    echo "  --full            显示详细模型信息"
    echo "  --plot            生成并显示图表比较"
    echo "  --evaluate        重新评估所有模型指标"
    echo "  --fix-winrate     修复模型指标文件中的胜率数据"
    echo "  --max-dd VALUE    设置最大回撤阈值(默认: 0.05)"
    echo "  --min-sortino VALUE  设置索提诺比率最小阈值(默认: 25)"
    echo "  --min-sharpe VALUE   设置夏普比率最小阈值(默认: 12)"
    echo "  --verbose         显示详细输出"
    echo "  --help            显示此帮助信息"
    echo
    echo "示例:"
    echo "  $0 --evaluate --full      # 重新评估所有模型并显示完整信息"
    echo "  $0 --plot                 # 显示图表比较"
    echo "  $0 --max-dd 0.03 --min-sortino 20 --min-sharpe 10  # 自定义风控阈值"
    echo "==================================================="
    exit 0
fi

# 打印欢迎信息
echo "==================================================="
echo "BTC交易模型指标分析工具"
echo "==================================================="

# 确保激活了虚拟环境（如果存在）
if [ -f venv/bin/activate ]; then
    echo "激活虚拟环境..."
    source venv/bin/activate
fi

# 构建命令行参数
CMD_ARGS=""
if [ $SHOW_FULL -eq 1 ]; then
    CMD_ARGS="$CMD_ARGS --full"
    echo "模式: 显示详细信息"
else
    echo "模式: 显示摘要信息"
fi

if [ $PLOT -eq 1 ]; then
    CMD_ARGS="$CMD_ARGS --plot"
    echo "包含: 图表比较"
fi

if [ $EVALUATE -eq 1 ]; then
    CMD_ARGS="$CMD_ARGS --evaluate"
    echo "包含: 模型重新评估"
    
    # 检查Python虚拟环境是否已激活
    if [ -z "$VIRTUAL_ENV" ] && [ -f "venv/bin/activate" ]; then
        echo "警告: 需要激活Python虚拟环境以确保模型评估正常工作"
        source venv/bin/activate
        echo "已激活虚拟环境: $VIRTUAL_ENV"
    fi
fi

if [ $FIX_WINRATE -eq 1 ]; then
    CMD_ARGS="$CMD_ARGS --fix-winrate"
    echo "包含: 修复胜率数据"
    
    # 检查Python虚拟环境是否已激活
    if [ -z "$VIRTUAL_ENV" ] && [ -f "venv/bin/activate" ]; then
        echo "警告: 需要激活Python虚拟环境以确保胜率修复正常工作"
        source venv/bin/activate
        echo "已激活虚拟环境: $VIRTUAL_ENV"
    fi
fi

# 检查metrics目录是否存在
METRICS_DIR="$PROJECT_ROOT/btc_rl/metrics"
if [ ! -d "$METRICS_DIR" ]; then
    echo "创建指标目录: $METRICS_DIR"
    mkdir -p "$METRICS_DIR"
fi

echo "---------------------------------------------------"

# 添加风控阈值参数
CMD_ARGS="$CMD_ARGS --max-dd $MAX_DD --min-sortino $MIN_SORTINO --min-sharpe $MIN_SHARPE"

# 运行Python脚本
cd "$PROJECT_ROOT"
PYTHON_CMD="python btc_rl/src/show_model_metrics.py $CMD_ARGS"

if [ $VERBOSE -eq 1 ]; then
    echo "执行命令: $PYTHON_CMD"
    $PYTHON_CMD
else
    echo "正在分析模型指标..."
    echo "风控指标阈值: 最大回撤≤${MAX_DD}, 索提诺比率≥${MIN_SORTINO}, 夏普比率≥${MIN_SHARPE}"
    $PYTHON_CMD
fi

echo "---------------------------------------------------"
echo "分析完成!"
echo "==================================================="
