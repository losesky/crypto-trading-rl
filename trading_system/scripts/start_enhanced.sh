#!/bin/bash
# 强化学习交易系统启动脚本 - 高级版
# 支持多种启动模式和错误恢复

# 获取脚本所在目录的父目录（trading_system目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRADING_SYSTEM_DIR="$( dirname "$SCRIPT_DIR" )"
ROOT_DIR="$( dirname "$TRADING_SYSTEM_DIR" )"

# 显示彩色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 显示帮助信息
show_help() {
    echo -e "${BLUE}RL交易系统启动脚本 - 高级版${NC}"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help            显示帮助信息"
    echo "  -t, --test            启动测试环境 (默认)"
    echo "  -p, --prod            启动生产环境"
    echo "  -u, --ui-only         只启动UI界面"
    echo "  -d, --diagnostics     运行系统诊断"
    echo "  --new-ui              使用新版UI界面"
    echo "  --ws-proxy            单独启动WebSocket代理"
    echo "  --force               强制重启所有进程"
    echo "  --health              运行健康检查和修复"
    echo "  --verbose             显示详细日志"
    echo ""
}

# 默认参数
MODE="test"
UI_ONLY=false
NEW_UI=true  # 默认使用新版UI
WS_PROXY_ONLY=false
FORCE_RESTART=false
RUN_HEALTH_CHECK=false
VERBOSE=false
RUN_DIAGNOSTICS=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -t|--test)
            MODE="test"
            shift
            ;;
        -p|--prod)
            MODE="prod"
            shift
            ;;
        -u|--ui-only)
            UI_ONLY=true
            shift
            ;;
        -d|--diagnostics)
            RUN_DIAGNOSTICS=true
            shift
            ;;
        --new-ui)
            NEW_UI=true
            shift
            ;;
        --ws-proxy)
            WS_PROXY_ONLY=true
            shift
            ;;
        --force)
            FORCE_RESTART=true
            shift
            ;;
        --health)
            RUN_HEALTH_CHECK=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 设置详细日志
if [ "$VERBOSE" = true ]; then
    set -x
fi

# 显示系统信息
echo -e "${BLUE}===============================================${NC}"
echo -e "${BLUE}         RL交易系统启动工具 - 高级版          ${NC}"
echo -e "${BLUE}===============================================${NC}"
echo -e "${YELLOW}系统信息:${NC}"
echo -e "  系统版本: $(uname -a)"
echo -e "  当前时间: $(date)"
echo -e "  运行模式: ${MODE}"
echo -e "  项目路径: ${TRADING_SYSTEM_DIR}"
echo -e "${BLUE}===============================================${NC}"

# 检测是否存在虚拟环境并激活
if [ -d "$ROOT_DIR/venv" ]; then
    echo -e "${GREEN}激活虚拟环境...${NC}"
    source "$ROOT_DIR/venv/bin/activate"
else
    echo -e "${YELLOW}警告: 未找到虚拟环境。建议先运行 './scripts/install_dependencies.sh' 建立环境。${NC}"
    echo -e "${YELLOW}尝试继续运行，但可能存在依赖问题...${NC}"
fi

# 设置配置文件路径
if [ "$MODE" = "test" ]; then
    CONFIG_FILE="$TRADING_SYSTEM_DIR/config/test_config.json"
    echo -e "${GREEN}使用测试配置: $CONFIG_FILE${NC}"
else
    CONFIG_FILE="$TRADING_SYSTEM_DIR/config/prod_config.json"
    echo -e "${GREEN}使用生产配置: $CONFIG_FILE${NC}"
fi

# 创建日志目录
mkdir -p "$TRADING_SYSTEM_DIR/logs"

# 将必要的目录添加到PYTHONPATH，确保能够正确导入模块
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

# 检查端口占用
check_port() {
    local port=$1
    netstat -tuln | grep -q ":$port " && return 0 || return 1
}

# 停止特定端口的进程
stop_process_on_port() {
    local port=$1
    if check_port $port; then
        echo -e "${YELLOW}端口 $port 已被占用，尝试释放...${NC}"
        pid=$(lsof -t -i:$port)
        if [ -n "$pid" ]; then
            echo -e "${YELLOW}正在停止进程 $pid...${NC}"
            kill $pid 2>/dev/null || kill -9 $pid 2>/dev/null
            sleep 1
        fi
    fi
}

# 如果需要强制重启或运行健康检查
if [ "$FORCE_RESTART" = true ] || [ "$RUN_HEALTH_CHECK" = true ]; then
    echo -e "${YELLOW}正在检查并停止可能冲突的进程...${NC}"
    
    # 获取配置中的端口号
    HTTP_PORT=$(grep -o '"http_port": [0-9]*' $CONFIG_FILE | awk '{print $2}')
    WS_PORT=$(grep -o '"ws_port": [0-9]*' $CONFIG_FILE | awk '{print $2}')
    
    # 如果没有找到端口，使用默认值
    HTTP_PORT=${HTTP_PORT:-8090}
    WS_PORT=${WS_PORT:-8095}
    
    # 停止相关进程
    stop_process_on_port $HTTP_PORT
    stop_process_on_port $WS_PORT
    
    # 停止可能的Python进程
    pkill -f "python.*main.py" 2>/dev/null || true
    pkill -f "python.*websocket_proxy" 2>/dev/null || true
    pkill -f "python.*http.server" 2>/dev/null || true
    
    echo -e "${GREEN}进程清理完成${NC}"
    sleep 1
fi

# 运行健康检查
if [ "$RUN_HEALTH_CHECK" = true ]; then
    echo -e "${GREEN}正在运行系统健康检查...${NC}"
    python "$TRADING_SYSTEM_DIR/scripts/health_check.py" --auto-fix
fi

# 运行系统诊断
if [ "$RUN_DIAGNOSTICS" = true ]; then
    echo -e "${GREEN}正在运行系统诊断...${NC}"
    python "$TRADING_SYSTEM_DIR/scripts/fix_data_issues.py" --full-check
    exit 0
fi

# 如果只是启动WebSocket代理
if [ "$WS_PROXY_ONLY" = true ]; then
    echo -e "${GREEN}启动WebSocket代理服务...${NC}"
    cd "$TRADING_SYSTEM_DIR/src"
    python -c "from websocket_proxy import get_instance; proxy = get_instance(); proxy.start(); import time; print('WebSocket代理已启动，按Ctrl+C停止...'); time.sleep(3600)" &
    echo -e "${GREEN}WebSocket代理已启动 (PID: $!)${NC}"
    exit 0
fi

# 如果是只启动UI，将主页替换为新版，并退出
if [ "$UI_ONLY" = true ]; then
    echo -e "${GREEN}准备启动UI界面...${NC}"
    
    # 根据参数决定使用哪个UI版本
    if [ "$NEW_UI" = true ]; then
        echo -e "${BLUE}使用新版UI界面...${NC}"
        # 复制新UI文件到默认位置（先备份原始文件）
        if [ ! -f "$TRADING_SYSTEM_DIR/ui/index.html.bak" ]; then
            cp "$TRADING_SYSTEM_DIR/ui/index.html" "$TRADING_SYSTEM_DIR/ui/index.html.bak" 2>/dev/null || true
            cp "$TRADING_SYSTEM_DIR/ui/app.js" "$TRADING_SYSTEM_DIR/ui/app.js.bak" 2>/dev/null || true
        fi
        cp "$TRADING_SYSTEM_DIR/ui/index-new.html" "$TRADING_SYSTEM_DIR/ui/index.html"
        cp "$TRADING_SYSTEM_DIR/ui/app-new.js" "$TRADING_SYSTEM_DIR/ui/app.js"
        cp "$TRADING_SYSTEM_DIR/ui/styles-new.css" "$TRADING_SYSTEM_DIR/ui/styles.css"
    else
        echo -e "${BLUE}使用标准UI界面...${NC}"
        # 恢复原始UI文件（如果备份存在）
        if [ -f "$TRADING_SYSTEM_DIR/ui/index.html.bak" ]; then
            cp "$TRADING_SYSTEM_DIR/ui/index.html.bak" "$TRADING_SYSTEM_DIR/ui/index.html"
        fi
        if [ -f "$TRADING_SYSTEM_DIR/ui/app.js.bak" ]; then
            cp "$TRADING_SYSTEM_DIR/ui/app.js.bak" "$TRADING_SYSTEM_DIR/ui/app.js"
        fi
    fi
    
    # 启动WebSocket代理（在后台）
    echo -e "${GREEN}启动WebSocket代理（UI模式）...${NC}"
    cd "$TRADING_SYSTEM_DIR/src"
    python -c "from websocket_proxy import get_instance; proxy = get_instance(); proxy.start(); import time; print('WebSocket代理已启动，按Ctrl+C停止...'); time.sleep(3600)" &
    WS_PID=$!
    echo -e "${GREEN}WebSocket代理已启动 (PID: $WS_PID)${NC}"
    sleep 1
    
    # 启动简单的HTTP服务器提供UI
    echo -e "${GREEN}启动本地Web服务器...${NC}"
    cd "$TRADING_SYSTEM_DIR/ui"
    python -m http.server 8090 &
    HTTP_PID=$!
    echo -e "${GREEN}Web服务器已启动 (PID: $HTTP_PID)${NC}"
    
    echo -e "${GREEN}您可以通过访问 http://localhost:8090 查看交易仪表盘${NC}"
    echo -e "${YELLOW}按 Ctrl+C 停止服务...${NC}"
    
    # 捕获Ctrl+C信号
    trap "echo -e '${YELLOW}收到停止信号，正在关闭...${NC}'; kill $WS_PID 2>/dev/null; kill $HTTP_PID 2>/dev/null; exit 0" SIGINT SIGTERM
    
    # 等待用户Ctrl+C
    wait
    
    exit 0
fi

# 安装缺少的依赖
echo -e "${GREEN}检查必要的依赖...${NC}"
pip install -q flask-cors flask-socketio python-socketio websocket-client

# 备份当前UI文件
if [ "$NEW_UI" = true ] && [ ! -f "$TRADING_SYSTEM_DIR/ui/index.html.bak" ]; then
    echo -e "${GREEN}备份原始UI文件...${NC}"
    cp "$TRADING_SYSTEM_DIR/ui/index.html" "$TRADING_SYSTEM_DIR/ui/index.html.bak" 2>/dev/null || true
    cp "$TRADING_SYSTEM_DIR/ui/app.js" "$TRADING_SYSTEM_DIR/ui/app.js.bak" 2>/dev/null || true
    
    # 使用新版UI
    echo -e "${GREEN}切换到新版UI界面...${NC}"
    cp "$TRADING_SYSTEM_DIR/ui/index-new.html" "$TRADING_SYSTEM_DIR/ui/index.html"
    cp "$TRADING_SYSTEM_DIR/ui/app-new.js" "$TRADING_SYSTEM_DIR/ui/app.js"
    cp "$TRADING_SYSTEM_DIR/ui/styles-new.css" "$TRADING_SYSTEM_DIR/ui/styles.css"
fi

# 启动交易系统
echo -e "${GREEN}启动RL交易系统...${NC}"

# 首先启动WebSocket代理（在后台）
echo -e "${BLUE}启动WebSocket代理...${NC}"
cd "$TRADING_SYSTEM_DIR/src"
python -c "from websocket_proxy import get_instance; proxy = get_instance(); proxy.start(); import time; print('WebSocket代理已启动'); time.sleep(1)" &
WS_PID=$!
sleep 2  # 给点时间启动

# 启动主系统
cd "$TRADING_SYSTEM_DIR/src"
echo -e "${BLUE}启动交易服务...${NC}"
python main.py --config "$CONFIG_FILE" --mode $MODE &
MAIN_PID=$!

echo -e "${GREEN}交易系统已启动！${NC}"
echo -e "${GREEN}您可以通过访问 http://localhost:8090 查看交易仪表盘。${NC}"
echo -e "${YELLOW}按 Ctrl+C 停止交易服务...${NC}"

# 捕获Ctrl+C信号
trap "echo -e '${YELLOW}收到停止信号，正在关闭...${NC}'; kill $WS_PID 2>/dev/null; kill $MAIN_PID 2>/dev/null; exit 0" SIGINT SIGTERM

# 等待主进程结束
wait $MAIN_PID
echo -e "${BLUE}主系统进程已结束${NC}"
# 杀掉WebSocket代理
kill $WS_PID 2>/dev/null || true

echo -e "${GREEN}交易系统已关闭。${NC}"
