#!/bin/bash
# 交易系统统一启动脚本
# 整合了所有功能的单一入口点

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRADING_SYSTEM_DIR="$( dirname "$SCRIPT_DIR" )"
ROOT_DIR="$( dirname "$TRADING_SYSTEM_DIR" )"

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 显示帮助信息
show_help() {
    echo -e "${BLUE}RL交易系统统一启动脚本${NC}"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help            显示帮助信息"
    echo "  -t, --test            启动测试环境 (默认)"
    echo "  -p, --prod            启动生产环境"
    echo "  -u, --ui-only         只启动UI界面"
    echo "  --ws-proxy            单独启动WebSocket代理"
    echo "  --force               强制重启所有进程"
    echo "  --health              运行健康检查和修复"
    echo "  --menu                显示交互式菜单 (默认无参数时)"
    echo ""
}

# 默认参数
MODE="test"
UI_ONLY=false
WS_PROXY_ONLY=false
FORCE_RESTART=false
RUN_HEALTH_CHECK=false
SHOW_MENU=false

# 如果没有参数，显示菜单
if [ $# -eq 0 ]; then
    SHOW_MENU=true
fi

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
        --menu)
            SHOW_MENU=true
            shift
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 设置配置文件路径
if [ "$MODE" = "test" ]; then
    CONFIG_FILE="$TRADING_SYSTEM_DIR/config/test_config.json"
    echo -e "${YELLOW}使用测试环境配置${NC}"
else
    CONFIG_FILE="$TRADING_SYSTEM_DIR/config/prod_config.json"
    echo -e "${RED}注意：使用生产环境配置，将使用实际资金！${NC}"
fi

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}错误: 找不到配置文件 $CONFIG_FILE${NC}"
    exit 1
fi

# 检测是否存在虚拟环境并激活
if [ -d "$ROOT_DIR/venv" ]; then
    echo -e "${GREEN}激活虚拟环境...${NC}"
    source "$ROOT_DIR/venv/bin/activate"
else
    echo -e "${YELLOW}提示: 未找到虚拟环境。尝试直接使用系统Python...${NC}"
fi

# 设置Python环境变量
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

# 停止可能运行中的进程
kill_existing_processes() {
    echo -e "${YELLOW}停止可能运行中的进程...${NC}"
    
    # 获取配置中的端口
    HTTP_PORT=8090
    WS_PORT=8095
    if [ -f "$CONFIG_FILE" ]; then
        HTTP_PORT=$(grep -o '"http_port": [0-9]*' $CONFIG_FILE | awk '{print $2}' || echo "8090")
        WS_PORT=$(grep -o '"ws_port": [0-9]*' $CONFIG_FILE | awk '{print $2}' || echo "8095")
    fi
    
    # 停止占用端口的进程
    for PORT in $HTTP_PORT $WS_PORT; do
        PID=$(lsof -t -i:$PORT 2>/dev/null)
        if [ -n "$PID" ]; then
            echo -e "${YELLOW}终止端口 $PORT 上的进程 $PID${NC}"
            kill -9 $PID 2>/dev/null || true
        fi
    done
    
    # 停止Python相关进程
    pkill -f "python.*main.py" 2>/dev/null || true
    pkill -f "python.*websocket_proxy" 2>/dev/null || true
    pkill -f "python.*ui_server" 2>/dev/null || true
    pkill -f "python.*http.server" 2>/dev/null || true
    
    echo -e "${GREEN}清理完成${NC}"
    sleep 1
}

# 启动UI界面
start_ui() {
    echo -e "${GREEN}正在启动UI服务器...${NC}"
    cd "$TRADING_SYSTEM_DIR/ui"
    
    # 使用新版UI
    if [ -f "index-new.html" ]; then
        echo -e "${GREEN}使用新版UI界面...${NC}"
        cp -f "index-new.html" "index.html"
        cp -f "app-new.js" "app.js"
        cp -f "styles-new.css" "styles.css"
    fi
    
    python -m http.server 8090 &
    echo -e "${GREEN}UI服务器已启动，可访问 http://localhost:8090${NC}"
}

# 启动WebSocket代理
start_websocket_proxy() {
    echo -e "${GREEN}正在启动WebSocket代理...${NC}"
    cd "$TRADING_SYSTEM_DIR/src"
    python -c "
try:
    from websocket_proxy import get_instance
    proxy = get_instance()
    proxy.start()
    print('WebSocket代理已启动')
except Exception as e:
    print(f'启动WebSocket代理时出错: {e}')
" &
    sleep 2
}

# 运行健康检查
run_health_check() {
    echo -e "${BLUE}运行健康检查...${NC}"
    # 检查健康检查脚本
    HEALTH_CHECK="$SCRIPT_DIR/health_check.py"
    if [ -f "$HEALTH_CHECK" ]; then
        python "$HEALTH_CHECK" --auto-fix
        echo -e "${GREEN}健康检查完成${NC}"
    else
        echo -e "${RED}错误: 找不到健康检查脚本${NC}"
    fi
}

# 显示交互式菜单
show_interactive_menu() {
    # 清屏
    clear
    
    # 显示欢迎信息
    echo -e "${BLUE}==================================================${NC}"
    echo -e "${BLUE}          RL加密货币交易系统启动工具           ${NC}"
    echo -e "${BLUE}==================================================${NC}"
    echo ""
    echo -e "${GREEN}项目目录: $ROOT_DIR${NC}"
    echo ""
    echo -e "${YELLOW}请选择操作:${NC}"
    echo ""
    echo -e "  ${GREEN}1)${NC} 启动完整交易系统（测试环境）"
    echo -e "  ${GREEN}2)${NC} 仅启动UI界面"
    echo -e "  ${GREEN}3)${NC} 启动生产环境系统"
    echo -e "  ${GREEN}4)${NC} 运行健康检查"
    echo -e "  ${GREEN}5)${NC} 强制重启（解决冲突）"
    echo -e "  ${GREEN}0)${NC} 退出"
    echo ""
    
    # 读取用户选择
    read -p "请输入选项 [0-5]: " choice
    
    case $choice in
        1)
            echo -e "${BLUE}启动完整交易系统...${NC}"
            MODE="test"
            CONFIG_FILE="$TRADING_SYSTEM_DIR/config/test_config.json"
            FORCE_RESTART=true
            ;;
            
        2)
            echo -e "${BLUE}仅启动UI界面...${NC}"
            UI_ONLY=true
            FORCE_RESTART=true
            ;;
            
        3)
            echo -e "${YELLOW}警告: 即将启动生产环境系统，将使用实际资金进行交易！${NC}"
            read -p "确认要继续吗？(y/N): " confirm
            if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
                echo -e "${BLUE}启动生产环境交易系统...${NC}"
                MODE="prod"
                CONFIG_FILE="$TRADING_SYSTEM_DIR/config/prod_config.json"
                FORCE_RESTART=true
            else
                echo -e "${YELLOW}操作已取消${NC}"
                exit 0
            fi
            ;;
            
        4)
            run_health_check
            exit 0
            ;;
            
        5)
            echo -e "${YELLOW}强制重启系统...${NC}"
            MODE="test"
            CONFIG_FILE="$TRADING_SYSTEM_DIR/config/test_config.json"
            FORCE_RESTART=true
            ;;
            
        0|"")
            echo -e "${BLUE}感谢使用RL交易系统，再见！${NC}"
            exit 0
            ;;
            
        *)
            echo -e "${RED}无效选项${NC}"
            exit 1
            ;;
    esac
}

# 如果要显示交互式菜单
if [ "$SHOW_MENU" = true ]; then
    show_interactive_menu
fi

# 如果需要强制重启
if [ "$FORCE_RESTART" = true ]; then
    kill_existing_processes
fi

# 如果需要运行健康检查
if [ "$RUN_HEALTH_CHECK" = true ]; then
    run_health_check
fi

# 根据选择执行相应操作
if [ "$WS_PROXY_ONLY" = true ]; then
    # 只启动WebSocket代理
    start_websocket_proxy
    echo -e "${YELLOW}按 Ctrl+C 停止服务...${NC}"
    wait
elif [ "$UI_ONLY" = true ]; then
    # 只启动UI界面和WebSocket代理
    start_websocket_proxy
    start_ui
    echo -e "${YELLOW}按 Ctrl+C 停止服务...${NC}"
    wait
else
    # 启动完整系统
    start_websocket_proxy
    echo -e "${GREEN}启动交易服务 (${MODE}模式)...${NC}"
    cd "$TRADING_SYSTEM_DIR/src"
    python main.py --config "$CONFIG_FILE" --mode "$MODE"
fi

echo "交易系统已关闭。"
