#!/bin/bash

# 测试数据收集功能的脚本
echo "开始测试数据收集功能..."

# 获取脚本所在目录的父目录（trading_system目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRADING_SYSTEM_DIR="$( dirname "$SCRIPT_DIR" )"
ROOT_DIR="$( dirname "$TRADING_SYSTEM_DIR" )"

# 检测是否存在虚拟环境并激活
if [ -d "$ROOT_DIR/venv" ]; then
    echo "激活虚拟环境..."
    source "$ROOT_DIR/venv/bin/activate"
else
    echo "错误：未找到虚拟环境。请先运行 './scripts/install_dependencies.sh'"
    exit 1
fi

# 将必要的目录添加到PYTHONPATH，确保能够正确导入模块
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

# 运行数据收集测试脚本
echo "运行数据收集测试脚本..."
python "$TRADING_SYSTEM_DIR/scripts/test_data_collection.py"

# 检查脚本执行结果
TEST_RESULT=$?
if [ $TEST_RESULT -eq 0 ]; then
    echo ""
    echo "数据收集功能测试成功！"
    echo ""
    echo "您可以使用以下命令分析收集的测试数据："
    echo "  ./trading_system/scripts/analyze_model_performance.sh"
    echo ""
    echo "要开始实际的交易数据收集，请使用："
    echo "  ./trading_system/scripts/start_test_trading.sh"
else
    echo ""
    echo "数据收集功能测试失败！"
    echo ""
    echo "请检查日志以确定问题所在，并修复问题后再尝试运行交易系统。"
    echo ""
    echo "可能的问题："
    echo "1. 权限问题 - 确保脚本和目录有适当的权限"
    echo "2. 路径问题 - 确保配置文件中的路径设置正确"
    echo "3. 依赖问题 - 确保所有必要的Python包都已安装"
    echo ""
    echo "解决问题后，再次运行此测试或直接启动交易系统。"
fi

exit $TEST_RESULT
