#!/bin/bash

# 交易数据分析脚本
echo "开始分析交易数据和模型性能..."

# 获取脚本所在目录的父目录（trading_system目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRADING_SYSTEM_DIR="$( dirname "$SCRIPT_DIR" )"
ROOT_DIR="$( dirname "$TRADING_SYSTEM_DIR" )"

# 初始化标志
INITIALIZE=false

# 处理命令行参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --initialize) INITIALIZE=true ;;
        --help|-h) 
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --initialize    初始化数据收集目录结构"
            echo "  --help, -h      显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0                     分析已收集的交易数据"
            echo "  $0 --initialize        初始化数据收集目录结构"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
    shift
done

# 检测是否存在虚拟环境并激活
if [ -d "$ROOT_DIR/venv" ]; then
    echo "激活虚拟环境..."
    source "$ROOT_DIR/venv/bin/activate"
else
    echo "错误：未找到虚拟环境。请先运行 './scripts/install_dependencies.sh'"
    exit 1
fi

# 安装所需依赖
pip install -q matplotlib pandas tabulate

# 将必要的目录添加到PYTHONPATH，确保能够正确导入模块
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

# 设置数据目录和报告目录
DATA_DIR="$TRADING_SYSTEM_DIR/data/collected_data"
REPORTS_DIR="$TRADING_SYSTEM_DIR/reports"

# 创建报告目录（如果不存在）
mkdir -p "$REPORTS_DIR"

# 运行分析脚本
if [ "$INITIALIZE" = true ]; then
    echo "初始化数据收集目录结构..."
    INIT_FLAG="--initialize"
    
    # 确保目录存在
    mkdir -p "$DATA_DIR"
    mkdir -p "$REPORTS_DIR"
else
    echo "运行数据分析脚本..."
    INIT_FLAG=""
    
    # 检查数据目录是否存在
    if [ ! -d "$DATA_DIR" ]; then
        echo "警告：数据收集目录不存在: $DATA_DIR"
        echo "请先使用 --initialize 选项初始化数据收集目录，然后运行交易系统收集数据"
        echo "用法："
        echo "  $0 --initialize       # 初始化数据收集目录"
        echo "  ./start_test_trading.sh  # 启动交易系统收集数据"
        echo "  $0                   # 分析收集的数据"
        exit 1
    fi
fi

python "$TRADING_SYSTEM_DIR/scripts/analyze_trading_data.py" \
    --data-dir "$DATA_DIR" \
    --reports-dir "$REPORTS_DIR" \
    --generate-plots \
    $INIT_FLAG

# 初始化时的特殊处理
if [ "$INITIALIZE" = true ]; then
    echo ""
    echo "数据收集目录已初始化完成！"
    echo ""
    echo "接下来的步骤："
    echo "1. 启动交易系统以收集真实交易数据："
    echo "   ./trading_system/scripts/start_test_trading.sh"
    echo ""
    echo "2. 让交易系统运行至少12-24小时，或至少完成10-20笔交易"
    echo ""
    echo "3. 再次运行分析工具（不带--initialize参数）："
    echo "   ./trading_system/scripts/analyze_model_performance.sh"
    echo ""
    exit 0
fi

# 查找最新的报告文件（可能是交易分析报告或设置指南）
LATEST_REPORT=$(ls -t "$REPORTS_DIR"/*.txt 2>/dev/null | head -n 1)

if [ -f "$LATEST_REPORT" ]; then
    echo ""
    echo "报告已生成: $LATEST_REPORT"
    echo ""
    echo "报告摘要:"
    echo "=========================="
    head -n 20 "$LATEST_REPORT"
    echo "..."
    echo "=========================="
    echo ""
    echo "使用文本编辑器查看完整报告，或在浏览器中查看图表（如果已生成）"
    
    # 检查是否是无数据报告
    if [[ "$LATEST_REPORT" == *"no_data_report"* ]] || [[ "$LATEST_REPORT" == *"setup_guide"* ]]; then
        echo ""
        echo "注意：未找到交易数据。请按照报告中的步骤操作以开始数据收集。"
        echo "如果您已经运行了交易系统，请确保它能正常工作并产生交易决策。"
        echo ""
    fi
else
    echo "未生成报告，请检查脚本是否正确执行"
fi

echo "分析完成。"
