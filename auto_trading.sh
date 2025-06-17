#!/bin/bash

# 自动交易系统启动脚本
# 该脚本用于启动、停止和管理自动交易系统

# 设置基础路径
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTO_TRADING_DIR="${BASE_DIR}/auto_trading"
CONFIG_DIR="${AUTO_TRADING_DIR}/config"
LOGS_DIR="${AUTO_TRADING_DIR}/logs"
PID_FILE="${AUTO_TRADING_DIR}/.trading_pid"

# 确保使用虚拟环境中的Python和pip
if [[ -n "$VIRTUAL_ENV" ]]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
    PIP="$VIRTUAL_ENV/bin/pip"
else
    PYTHON="python3"
    PIP="pip3"
fi

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 确保日志目录存在
mkdir -p ${LOGS_DIR}

# 显示帮助信息
show_help() {
    echo -e "${BLUE}币安U本位自动量化交易系统${NC}"
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  start         启动交易系统"
    echo "  stop          停止交易系统"
    echo "  restart       重启交易系统"
    echo "  status        查看交易系统运行状态"
    echo "  logs          查看最近日志"
    echo "  test          测试模式运行(不实际交易)"
    echo "  close-all     关闭所有持仓"
    echo "  help          显示此帮助信息"
    echo ""
    echo "附加参数:"
    echo "  --debug       以调试日志级别运行"
    echo "  --config-dir=PATH  指定配置目录路径"
    echo ""
    echo "例子:"
    echo "  $0 start --debug"
    echo "  $0 stop"
    echo "  $0 logs"
}

# 检查是否安装了必要的软件
check_requirements() {
    echo -e "${BLUE}检查系统要求...${NC}"
    
    # 检查Python 3.8+
    if ! command -v $PYTHON &> /dev/null; then
        echo -e "${RED}错误: 未找到Python 3，请安装Python 3.8或更高版本${NC}"
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if (( $(echo "$PYTHON_VERSION < 3.8" | bc -l) )); then
        echo -e "${RED}错误: 需要Python 3.8或更高版本，当前版本为${PYTHON_VERSION}${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Python版本: ${PYTHON_VERSION} ✓${NC}"
    
    # 检查pip
    if ! command -v $PIP &> /dev/null; then
        echo -e "${RED}错误: 未找到pip，请安装pip${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}pip已安装 ✓${NC}"
    
    # 检查依赖
    if [ ! -f "${BASE_DIR}/requirements.txt" ]; then
        echo -e "${YELLOW}警告: 未找到requirements.txt文件${NC}"
    else
        echo -e "${BLUE}检查Python依赖...${NC}"
        
        # 使用pip list检查已安装的包
        echo -e "${BLUE}检查已安装的包...${NC}"
        
        # 使用pip检查已安装的包，更可靠
        INSTALLED_PACKAGES=$(pip freeze | cut -d'=' -f1 | tr '[:upper:]' '[:lower:]' | tr '_' '-')
        MISSING_DEPS=0
        MISSING_PACKAGES=()
        
        while IFS= read -r package || [[ -n "$package" ]]; do
            if [[ $package != \#* && -n $package ]]; then
                # 提取包名，移除版本信息和额外选项
                package_name=$(echo $package | cut -d'>' -f1 | cut -d'<' -f1 | cut -d'=' -f1 | cut -d'[' -f1 | tr -d ' ')
                package_name_lower=$(echo $package_name | tr '[:upper:]' '[:lower:]')
                
                # 特殊处理某些基础包，使用pip show代替pip freeze检查
                if [[ "$package_name_lower" == "setuptools" ]] || [[ "$package_name_lower" == "pip" ]] || [[ "$package_name_lower" == "wheel" ]]; then
                    if ! $PIP show $package_name_lower &> /dev/null; then
                        echo -e "${YELLOW}未安装: $package_name${NC}"
                        MISSING_PACKAGES+=("$package")
                        MISSING_DEPS=1
                    fi
                elif ! echo "$INSTALLED_PACKAGES" | grep -q "^$package_name_lower"; then
                    echo -e "${YELLOW}未安装: $package_name${NC}"
                    MISSING_PACKAGES+=("$package")
                    MISSING_DEPS=1
                fi
            fi
        done < "${BASE_DIR}/requirements.txt"
        
        if [ $MISSING_DEPS -eq 1 ]; then
            echo -e "${YELLOW}请运行: pip install -r ${BASE_DIR}/requirements.txt${NC}"
            read -p "是否现在安装依赖? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                $PIP install -r "${BASE_DIR}/requirements.txt"
            fi
        else
            echo -e "${GREEN}所有依赖已安装 ✓${NC}"
        fi
    fi
}

# 启动交易系统
start_trading_system() {
    echo -e "${BLUE}启动自动交易系统...${NC}"
    
    # WSL环境检查和时间同步
    if grep -q Microsoft /proc/version || grep -q WSL /proc/version; then
        echo -e "${YELLOW}检测到WSL环境，尝试同步时间...${NC}"
        # 检查时间同步脚本是否存在
        if [ -f "${BASE_DIR}/sync_wsl_time.sh" ]; then
            echo -e "${BLUE}运行时间同步脚本...${NC}"
            bash "${BASE_DIR}/sync_wsl_time.sh" || echo -e "${YELLOW}时间同步可能需要sudo权限${NC}"
        else
            echo -e "${YELLOW}未找到时间同步脚本，跳过时间同步${NC}"
        fi
    fi
    
    # 检查是否已经在运行
    if [ -f "${PID_FILE}" ] && ps -p $(cat "${PID_FILE}") > /dev/null; then
        echo -e "${YELLOW}交易系统已经在运行，PID: $(cat ${PID_FILE})${NC}"
        return 1
    fi
    
    # 构建启动命令
    CMD="$PYTHON ${AUTO_TRADING_DIR}/main.py"
    
    # 添加额外参数
    for arg in "$@"; do
        case $arg in
            --debug)
                CMD="$CMD --log-level=DEBUG"
                ;;
            --config-dir=*)
                CONFIG_PATH="${arg#*=}"
                CMD="$CMD --config-dir=$CONFIG_PATH"
                ;;
            --test-mode)
                CMD="$CMD --test-mode"
                echo -e "${YELLOW}以测试模式启动，不会执行实际交易${NC}"
                ;;
            --close-positions)
                CMD="$CMD --close-positions"
                echo -e "${YELLOW}启动后将关闭所有持仓${NC}"
                ;;
        esac
    done
    
    # 启动并后台运行
    echo -e "${BLUE}执行: $CMD${NC}"
    nohup $CMD > "${LOGS_DIR}/trading_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
    
    # 保存PID
    echo $! > "${PID_FILE}"
    sleep 2
    
    # 检查是否成功启动
    if [ -f "${PID_FILE}" ] && ps -p $(cat "${PID_FILE}") > /dev/null; then
        echo -e "${GREEN}交易系统已成功启动，PID: $(cat ${PID_FILE})${NC}"
        return 0
    else
        echo -e "${RED}启动失败，请检查日志${NC}"
        return 1
    fi
}

# 停止交易系统
stop_trading_system() {
    echo -e "${BLUE}停止自动交易系统...${NC}"
    
    # 检查PID文件
    if [ ! -f "${PID_FILE}" ]; then
        echo -e "${YELLOW}未找到PID文件，交易系统可能未在运行${NC}"
        return 1
    fi
    
    PID=$(cat "${PID_FILE}")
    
    # 检查进程是否存在
    if ! ps -p $PID > /dev/null; then
        echo -e "${YELLOW}进程 $PID 不存在，删除过时的PID文件${NC}"
        rm "${PID_FILE}"
        return 1
    fi
    
    # 先尝试优雅关闭
    echo -e "${BLUE}发送SIGTERM信号到进程 $PID${NC}"
    kill -15 $PID
    
    # 等待进程退出
    for i in {1..30}; do
        if ! ps -p $PID > /dev/null; then
            echo -e "${GREEN}交易系统已停止${NC}"
            rm "${PID_FILE}"
            return 0
        fi
        sleep 1
    done
    
    # 如果进程未退出，强制终止
    echo -e "${YELLOW}进程未响应，强制终止${NC}"
    kill -9 $PID
    
    # 最后检查
    if ! ps -p $PID > /dev/null; then
        echo -e "${GREEN}交易系统已停止${NC}"
        rm "${PID_FILE}"
        return 0
    else
        echo -e "${RED}无法停止进程 $PID${NC}"
        return 1
    fi
}

# 查看系统状态
check_status() {
    echo -e "${BLUE}检查交易系统状态...${NC}"
    
    if [ ! -f "${PID_FILE}" ]; then
        echo -e "${YELLOW}交易系统未运行${NC}"
        return 1
    fi
    
    PID=$(cat "${PID_FILE}")
    
    if ps -p $PID > /dev/null; then
        echo -e "${GREEN}交易系统正在运行，PID: $PID${NC}"
        
        # 显示运行时间
        START_TIME=$(ps -o lstart= -p $PID)
        echo -e "${BLUE}启动时间: $START_TIME${NC}"
        
        # 显示资源使用情况
        CPU_USAGE=$(ps -p $PID -o %cpu | tail -n 1)
        MEM_USAGE=$(ps -p $PID -o %mem | tail -n 1)
        echo -e "${BLUE}CPU使用率: $CPU_USAGE%${NC}"
        echo -e "${BLUE}内存使用率: $MEM_USAGE%${NC}"
        
        return 0
    else
        echo -e "${YELLOW}进程 $PID 不存在，但PID文件仍然存在${NC}"
        echo -e "${YELLOW}删除过时的PID文件${NC}"
        rm "${PID_FILE}"
        return 1
    fi
}

# 查看日志
view_logs() {
    echo -e "${BLUE}查看最近日志...${NC}"
    
    # 找出最新的日志文件
    LATEST_LOG=$(find "${LOGS_DIR}" -name "trading_*.log" -type f -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [ -z "$LATEST_LOG" ]; then
        echo -e "${YELLOW}未找到日志文件${NC}"
        return 1
    fi
    
    echo -e "${BLUE}日志文件: $LATEST_LOG${NC}"
    
    # 使用less查看日志
    less "$LATEST_LOG"
}

# 测试模式运行
test_mode() {
    echo -e "${BLUE}以测试模式启动交易系统...${NC}"
    start_trading_system "--test-mode" "$@"
}

# 关闭所有持仓
close_all_positions() {
    echo -e "${BLUE}关闭所有持仓...${NC}"
    $PYTHON ${AUTO_TRADING_DIR}/main.py --close-positions
}

# 主逻辑
case "$1" in
    start)
        shift
        check_requirements
        start_trading_system "$@"
        ;;
    stop)
        stop_trading_system
        ;;
    restart)
        shift
        stop_trading_system
        sleep 2
        start_trading_system "$@"
        ;;
    status)
        check_status
        ;;
    logs)
        view_logs
        ;;
    test)
        shift
        check_requirements
        test_mode "$@"
        ;;
    close-all)
        check_requirements
        close_all_positions
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${YELLOW}未知选项: $1${NC}"
        show_help
        exit 1
        ;;
esac

exit 0
