#!/bin/bash
# RL交易系统统一入口脚本
# 整合了所有功能的单一入口点，无需调用其他脚本

# 获取脚本所在目录（项目根目录）
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRADING_SYSTEM_DIR="$ROOT_DIR/trading_system"
SCRIPTS_DIR="$TRADING_SYSTEM_DIR/scripts"

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
    echo "  --auto-deps           自动安装缺失的依赖"
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
AUTO_INSTALL_DEPS=false

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
        --auto-deps|--install-deps)
            AUTO_INSTALL_DEPS=true
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

# 检查关键文件
check_key_files() {
    local missing_files=false
    
    # 检查WebSocket代理文件
    if [ ! -f "$TRADING_SYSTEM_DIR/src/websocket_proxy.py" ]; then
        echo -e "${RED}错误: 找不到WebSocket代理文件!${NC}"
        echo -e "${YELLOW}预期路径: $TRADING_SYSTEM_DIR/src/websocket_proxy.py${NC}"
        missing_files=true
    fi
    
    # 检查主程序文件
    if [ ! -f "$TRADING_SYSTEM_DIR/src/main.py" ]; then
        echo -e "${RED}错误: 找不到主程序文件!${NC}"
        echo -e "${YELLOW}预期路径: $TRADING_SYSTEM_DIR/src/main.py${NC}"
        missing_files=true
    fi
    
    # 检查UI文件
    if [ ! -d "$TRADING_SYSTEM_DIR/ui" ]; then
        echo -e "${RED}错误: 找不到UI目录!${NC}"
        echo -e "${YELLOW}预期路径: $TRADING_SYSTEM_DIR/ui${NC}"
        missing_files=true
    fi
    
    # 如果有缺失文件，提供错误修复选项
    if [ "$missing_files" = true ]; then
        echo -e "${RED}发现系统关键文件缺失！${NC}"
        echo -e "${YELLOW}您想要如何处理?${NC}"
        echo -e "  1) 尝试修复（如运行健康检查）"
        echo -e "  2) 继续尝试运行（可能会失败）"
        echo -e "  0) 退出"
        
        read -p "请选择 [0-2]: " fix_choice
        
        case $fix_choice in
            1)
                echo -e "${YELLOW}尝试修复系统...${NC}"
                run_health_check
                ;;
            2)
                echo -e "${YELLOW}继续尝试运行...${NC}"
                ;;
            *)
                echo -e "${RED}退出...${NC}"
                exit 1
                ;;
        esac
    fi
}

# 运行文件检查
check_key_files

# 设置Python环境变量
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

# 确保日志目录存在
ensure_log_dir() {
    if [ ! -d "$TRADING_SYSTEM_DIR/logs" ]; then
        echo -e "${YELLOW}创建日志目录...${NC}"
        mkdir -p "$TRADING_SYSTEM_DIR/logs"
    fi
}

# 检查并安装必要的依赖
check_dependencies() {
    echo -e "${YELLOW}检查必要的Python依赖...${NC}"
    
    # 检查是否已安装pip
    if ! command -v pip &> /dev/null; then
        echo -e "${RED}错误: 未找到pip，请先安装Python和pip${NC}"
        return 1
    fi
    
    # 要检查的依赖列表
    local required_packages=("Flask-SocketIO" "Flask-CORS" "eventlet" "gevent" "flask" "python-engineio" "websocket-client" "werkzeug")
    local missing_packages=()
    
    # 检查每个依赖
    for package in "${required_packages[@]}"; do
        if ! pip list | grep -i "$package" &> /dev/null; then
            echo -e "${YELLOW}缺少依赖: $package${NC}"
            missing_packages+=("$package")
        fi
    done
    
    # 如果有缺失依赖，询问是否安装
    if [ ${#missing_packages[@]} -gt 0 ]; then
        echo -e "${YELLOW}发现缺少的依赖项: ${missing_packages[*]}${NC}"
        echo -e "${YELLOW}WebSocket功能需要这些依赖才能正常工作${NC}"
        
        if [ "$AUTO_INSTALL_DEPS" = true ]; then
            install_choice="y"
        else
            read -p "是否要自动安装这些依赖? (y/N): " install_choice
        fi
        
        if [[ "$install_choice" == "y" || "$install_choice" == "Y" ]]; then
            echo -e "${GREEN}正在安装缺少的依赖...${NC}"
            for package in "${missing_packages[@]}"; do
                echo -e "${GREEN}安装 $package...${NC}"
                pip install "$package"
            done
            echo -e "${GREEN}依赖安装完成${NC}"
        else
            echo -e "${YELLOW}警告: 未安装缺少的依赖，系统可能无法正常运行${NC}"
        fi
    else
        echo -e "${GREEN}所有必要的依赖已安装${NC}"
    fi
}

# 调用确保日志目录存在
ensure_log_dir

# 检查依赖
check_dependencies

# 停止可能运行中的进程
kill_existing_processes() {
    echo -e "${YELLOW}停止可能运行中的进程...${NC}"
    
    # 获取配置中的端口
    HTTP_PORT=8090
    WS_PORT=8095
    if [ -f "$CONFIG_FILE" ]; then
        CFG_HTTP_PORT=$(grep -o '"http_port": [0-9]*' "$CONFIG_FILE" 2>/dev/null | awk '{print $2}')
        CFG_WS_PORT=$(grep -o '"ws_port": [0-9]*' "$CONFIG_FILE" 2>/dev/null | awk '{print $2}')
        
        if [ -n "$CFG_HTTP_PORT" ]; then
            HTTP_PORT=$CFG_HTTP_PORT
        fi
        
        if [ -n "$CFG_WS_PORT" ]; then
            WS_PORT=$CFG_WS_PORT
        fi
    fi
    
    echo -e "${YELLOW}正在检查端口 $HTTP_PORT (HTTP) 和 $WS_PORT (WebSocket)...${NC}"
    
    # 停止占用端口的进程
    for PORT in $HTTP_PORT $WS_PORT; do
        if command -v lsof &> /dev/null; then
            # 如果lsof可用
            PID=$(lsof -t -i:$PORT 2>/dev/null)
            if [ -n "$PID" ]; then
                echo -e "${YELLOW}终止端口 $PORT 上的进程 $PID${NC}"
                kill -9 $PID 2>/dev/null || true
            fi
        elif command -v netstat &> /dev/null; then
            # 如果netstat可用
            PID=$(netstat -tulnp 2>/dev/null | grep ":$PORT " | awk '{print $7}' | cut -d'/' -f1)
            if [ -n "$PID" ]; then
                echo -e "${YELLOW}终止端口 $PORT 上的进程 $PID${NC}"
                kill -9 $PID 2>/dev/null || true
            fi
        else
            echo -e "${RED}警告: 无法检查端口占用情况（缺少lsof和netstat工具）${NC}"
        fi
    done
    
    echo -e "${YELLOW}停止交易系统相关进程...${NC}"
    
    # 停止Python相关进程，先尝试正常终止，然后强制终止
    for PATTERN in "python.*main.py" "python.*websocket_proxy" "python.*ui_server" "python.*http.server"; do
        # 获取匹配进程的PID
        PIDS=$(ps aux | grep -E "$PATTERN" | grep -v grep | awk '{print $2}')
        
        if [ -n "$PIDS" ]; then
            # 先尝试正常终止
            for PID in $PIDS; do
                echo -e "${YELLOW}正常终止进程 $PID ($PATTERN)${NC}"
                kill $PID 2>/dev/null || true
            done
            
            # 等待一秒看是否终止
            sleep 1
            
            # 检查是否还在运行，如果是则强制终止
            for PID in $PIDS; do
                if ps -p $PID > /dev/null 2>&1; then
                    echo -e "${YELLOW}强制终止进程 $PID ($PATTERN)${NC}"
                    kill -9 $PID 2>/dev/null || true
                fi
            done
        fi
    done
    
    echo -e "${GREEN}清理完成${NC}"
    sleep 1
}

# 启动UI界面
start_ui() {
    echo -e "${GREEN}正在启动UI界面...${NC}"
    cd "$TRADING_SYSTEM_DIR/ui"
    
    # 使用新版UI
    if [ -f "index-new.html" ]; then
        echo -e "${GREEN}使用新版UI界面...${NC}"
        cp -f "index-new.html" "index.html"
        cp -f "app-new.js" "app.js"
        cp -f "styles-new.css" "styles.css"
    fi
    
    # 获取配置中的HTTP端口
    HTTP_PORT=8090
    if [ -f "$CONFIG_FILE" ]; then
        CFG_PORT=$(grep -o '"http_port": [0-9]*' "$CONFIG_FILE" | awk '{print $2}' 2>/dev/null)
        if [ -n "$CFG_PORT" ]; then
            HTTP_PORT=$CFG_PORT
        fi
    fi
    
    # UI现在由API代理服务器提供，不再需要单独的HTTP服务器
    echo -e "${GREEN}UI界面由API代理服务器提供，可访问 http://localhost:$HTTP_PORT${NC}"
    
    # 为了兼容性，我们仍然设置UI_SERVER_PID变量，但将其设为API_PROXY_PID
    UI_SERVER_PID=$API_PROXY_PID
    
    # 检查WebSocket代理是否在运行
    if [ -z "$WS_PROXY_PID" ] || ! ps -p $WS_PROXY_PID > /dev/null; then
        echo -e "${YELLOW}警告: WebSocket代理未运行，UI的WebSocket功能将无法正常工作${NC}"
        echo -e "${YELLOW}尝试启动WebSocket代理...${NC}"
        start_websocket_proxy
        # 等待WebSocket代理启动
        sleep 3
    fi
    
    # 检查API代理是否在运行
    if [ -z "$API_PROXY_PID" ] || ! ps -p $API_PROXY_PID > /dev/null; then
        echo -e "${YELLOW}警告: API代理服务器未运行，UI将无法正常工作${NC}"
        echo -e "${YELLOW}尝试启动API代理服务器...${NC}"
        start_api_proxy
    else
        echo -e "${GREEN}API代理服务器正在运行 (PID: $API_PROXY_PID)，UI可以通过 http://localhost:$HTTP_PORT 访问${NC}"
    fi
    
    echo -e "${BLUE}WebSocket连接URL: ws://localhost:8095${NC}"
    echo -e "${BLUE}API服务URL: http://localhost:$HTTP_PORT${NC}"
    echo -e "${BLUE}访问地址: http://localhost:$HTTP_PORT${NC}"
}

# 启动WebSocket代理
start_websocket_proxy() {
    echo -e "${GREEN}正在启动WebSocket代理...${NC}"
    
    # 创建日志目录（如果不存在）
    mkdir -p "$TRADING_SYSTEM_DIR/logs"
    
    # 检查原生WebSocket代理启动器脚本是否存在
    NATIVE_WS_LAUNCHER="$TRADING_SYSTEM_DIR/src/start_native_websocket.py"
    WS_COMPAT_LAUNCHER="$TRADING_SYSTEM_DIR/src/compat_websocket_proxy.py"
    WS_LAUNCHER="$TRADING_SYSTEM_DIR/src/start_websocket_proxy.py"
    
    if [ -f "$NATIVE_WS_LAUNCHER" ]; then
        # 优先使用原生WebSocket代理（最佳兼容）
        echo -e "${GREEN}使用原生WebSocket代理启动器...${NC}"
        cd "$TRADING_SYSTEM_DIR/src"
        python "$NATIVE_WS_LAUNCHER" > "$TRADING_SYSTEM_DIR/logs/websocket_proxy.log" 2>&1 &
    elif [ -f "$WS_COMPAT_LAUNCHER" ]; then
        # 其次使用兼容版启动器
        echo -e "${GREEN}使用兼容版WebSocket代理启动器...${NC}"
        cd "$TRADING_SYSTEM_DIR/src"
        python "$WS_COMPAT_LAUNCHER" > "$TRADING_SYSTEM_DIR/logs/websocket_proxy.log" 2>&1 &
    elif [ -f "$WS_LAUNCHER" ]; then
        # 使用标准启动器
        echo -e "${GREEN}使用标准WebSocket代理启动器...${NC}"
        cd "$TRADING_SYSTEM_DIR/src"
        python "$WS_LAUNCHER" > "$TRADING_SYSTEM_DIR/logs/websocket_proxy.log" 2>&1 &
    else
        # 直接内联启动
        echo -e "${YELLOW}找不到WebSocket代理启动器，将直接启动...${NC}"
        cd "$TRADING_SYSTEM_DIR/src"
        python -c "
import sys, traceback, os
try:
    from websocket_proxy import get_instance
    proxy = get_instance()
    proxy.start()
    print('WebSocket代理已启动')
    while True:
        import time
        time.sleep(1)
except Exception as e:
    print(f'启动WebSocket代理时出错: {e}')
    traceback.print_exc()
" > "$TRADING_SYSTEM_DIR/logs/websocket_proxy.log" 2>&1 &
    fi
    
    # 保存进程ID，便于以后清理
    WS_PROXY_PID=$!
    echo -e "${GREEN}WebSocket代理启动中 (PID: $WS_PROXY_PID)...${NC}"
    
    # 给代理一点时间启动
    sleep 3
    
    # 检查代理是否成功启动
    if ps -p $WS_PROXY_PID > /dev/null; then
        echo -e "${GREEN}WebSocket代理进程已启动 (PID: $WS_PROXY_PID)${NC}"
        
        # 检查日志中是否有错误
        if grep -i "error\|exception\|fail" "$TRADING_SYSTEM_DIR/logs/websocket_proxy.log" > /dev/null; then
            echo -e "${YELLOW}WebSocket代理启动有警告，查看日志末尾:${NC}"
            tail -n 5 "$TRADING_SYSTEM_DIR/logs/websocket_proxy.log"
            echo -e "${YELLOW}使用 'cat $TRADING_SYSTEM_DIR/logs/websocket_proxy.log' 查看完整日志${NC}"
        fi
        
        # 尝试连接到WebSocket REST API，确保服务真的启动了
        echo -e "${YELLOW}检查WebSocket REST API是否响应...${NC}"
        if command -v curl &> /dev/null; then
            HEALTH_CHECK=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8096/health 2>/dev/null)
            if [ "$HEALTH_CHECK" = "200" ]; then
                echo -e "${GREEN}WebSocket REST API健康检查成功${NC}"
            else
                echo -e "${YELLOW}WebSocket REST API健康检查失败，状态码: $HEALTH_CHECK${NC}"
                echo -e "${YELLOW}WebSocket服务可能未完全启动，可能需要等待几秒钟...${NC}"
                # 多等待几秒
                sleep 5
            fi
        else
            echo -e "${YELLOW}未找到curl命令，无法检查WebSocket REST API健康状态${NC}"
        fi
    else
        echo -e "${RED}WebSocket代理启动失败！检查日志:${NC}"
        cat "$TRADING_SYSTEM_DIR/logs/websocket_proxy.log"
    fi
}

# 启动API代理服务器
start_api_proxy() {
    echo -e "${GREEN}正在启动API代理服务器...${NC}"
    
    # 创建日志目录（如果不存在）
    mkdir -p "$TRADING_SYSTEM_DIR/logs"
    
    # 检查API代理服务器启动器脚本是否存在
    API_PROXY_LAUNCHER="$TRADING_SYSTEM_DIR/src/start_api_proxy.py"
    
    # 获取WebSocket代理URL
    WS_PORT=8095  # WebSocket代理端口
    WS_URL="ws://localhost:$WS_PORT"  # 使用ws协议，与前端代码匹配
    
    # 用于REST API的WebSocket URL (注意区分WebSocket连接和REST API)
    WS_REST_PORT=8096  # WebSocket代理REST API端口
    WS_REST_URL="http://localhost:$WS_REST_PORT"
    
    # 获取主交易系统URL
    TRADING_PORT=8091  # 主交易系统API端口
    TRADING_URL="http://localhost:$TRADING_PORT"
    
    if [ -f "$API_PROXY_LAUNCHER" ]; then
        echo -e "${GREEN}使用API代理服务器启动器...${NC}"
        cd "$TRADING_SYSTEM_DIR/src"
        # 显示配置信息
        echo -e "${BLUE}API代理服务器配置:${NC}"
        echo -e "${BLUE}- 监听端口: $HTTP_PORT${NC}"
        echo -e "${BLUE}- WebSocket REST URL: $WS_REST_URL${NC}"
        echo -e "${BLUE}- 交易系统URL: $TRADING_URL${NC}"
        
        # 设置环境变量以传递给启动脚本
        export WS_PROXY_URL="$WS_REST_URL"
        export TRADING_SYSTEM_URL="$TRADING_URL"
        
        python "$API_PROXY_LAUNCHER" --port "$HTTP_PORT" --ws-url "$WS_REST_URL" --trading-url "$TRADING_URL" > "$TRADING_SYSTEM_DIR/logs/api_proxy.log" 2>&1 &
    else
        # 直接内联启动
        echo -e "${YELLOW}找不到API代理启动器，将直接启动...${NC}"
        cd "$TRADING_SYSTEM_DIR/src"
        python -c "
import sys, traceback, os
try:
    from api_proxy_server import get_instance
    print('正在启动API代理服务器...')
    print(f'监听端口: {$HTTP_PORT}')
    print(f'WebSocket REST API: $WS_REST_URL')
    print(f'交易系统URL: $TRADING_URL')
    print(f'WebSocket连接URL: $WS_URL')
    
    # 确保api_proxy_server模块正常导入
    proxy = get_instance(port=$HTTP_PORT, 
                         trading_url='$TRADING_URL', 
                         ws_url='$WS_REST_URL')
    proxy.start()
    print('API代理服务器已成功启动')
    
    # 保持服务运行
    while True:
        import time
        time.sleep(1)
except Exception as e:
    print(f'启动API代理服务器时出错: {e}')
    traceback.print_exc()
" > "$TRADING_SYSTEM_DIR/logs/api_proxy.log" 2>&1 &
    fi
    
    # 保存进程ID，便于以后清理
    API_PROXY_PID=$!
    echo -e "${GREEN}API代理服务器启动中 (PID: $API_PROXY_PID)...${NC}"
    
    # 给代理一点时间启动
    sleep 3
    
    # 检查代理是否成功启动
    if ps -p $API_PROXY_PID > /dev/null; then
        echo -e "${GREEN}API代理服务器已成功启动${NC}"
        # 检查日志中是否有错误
        if grep -i "error\|exception\|fail" "$TRADING_SYSTEM_DIR/logs/api_proxy.log" > /dev/null; then
            echo -e "${YELLOW}API代理服务器启动有警告，查看日志末尾:${NC}"
            tail -n 5 "$TRADING_SYSTEM_DIR/logs/api_proxy.log"
            echo -e "${YELLOW}使用 'cat $TRADING_SYSTEM_DIR/logs/api_proxy.log' 查看完整日志${NC}"
        fi
    else
        echo -e "${RED}API代理服务器启动失败！检查日志:${NC}"
        cat "$TRADING_SYSTEM_DIR/logs/api_proxy.log"
    fi
}

# 运行健康检查
run_health_check() {
    echo -e "${BLUE}运行健康检查...${NC}"
    # 检查健康检查脚本
    HEALTH_CHECK="$SCRIPTS_DIR/health_check.py"
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
    echo -e "  ${GREEN}6)${NC} 检查/安装系统依赖"
    echo -e "  ${GREEN}7)${NC} 修复WebSocket连接问题"
    echo -e "  ${GREEN}8)${NC} 启动API代理服务器"
    echo -e "  ${GREEN}0)${NC} 退出"
    echo ""
    
    # 读取用户选择
    read -p "请输入选项 [0-8]: " choice
    
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
        
        6)
            echo -e "${BLUE}检查系统依赖...${NC}"
            AUTO_INSTALL_DEPS=true
            check_dependencies
            echo -e "${GREEN}按任意键继续...${NC}"
            read -n1
            show_interactive_menu  # 返回菜单
            ;;
            
        7)
            fix_websocket_proxy_issues  # 修复WebSocket问题
            ;;
            
        8)
            echo -e "${BLUE}启动API代理服务器...${NC}"
            kill_existing_processes  # 确保先清理掉可能运行的旧实例
            start_api_proxy
            echo -e "${YELLOW}服务已启动。${NC}"
            show_system_status
            echo -e "${YELLOW}按 Ctrl+C 停止服务...${NC}"
            wait
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

# 添加清理函数
cleanup() {
    echo -e "${YELLOW}正在清理资源...${NC}"
    # 记录要关闭的进程
    if [ -n "$WS_PROXY_PID" ] && ps -p $WS_PROXY_PID > /dev/null; then
        echo "关闭WebSocket代理 (PID: $WS_PROXY_PID)"
        kill $WS_PROXY_PID 2>/dev/null || true
    fi
    
    if [ -n "$UI_SERVER_PID" ] && ps -p $UI_SERVER_PID > /dev/null; then
        echo "关闭UI服务器 (PID: $UI_SERVER_PID)"
        kill $UI_SERVER_PID 2>/dev/null || true
    fi

    if [ -n "$API_PROXY_PID" ] && ps -p $API_PROXY_PID > /dev/null; then
        echo "关闭API代理服务器 (PID: $API_PROXY_PID)"
        kill $API_PROXY_PID 2>/dev/null || true
    fi

    # 删除临时文件
    rm -f /tmp/start_ws_proxy_$$.py 2>/dev/null || true
    
    echo -e "${GREEN}清理完成${NC}"
}

# 添加显示系统状态的函数
show_system_status() {
    echo -e "${BLUE}==================================================${NC}"
    echo -e "${BLUE}         系统状态信息           ${NC}"
    echo -e "${BLUE}==================================================${NC}"
    
    # 显示WebSocket代理状态
    if [ -n "$WS_PROXY_PID" ] && ps -p $WS_PROXY_PID > /dev/null; then
        echo -e "${GREEN}● WebSocket代理: 运行中 (PID: $WS_PROXY_PID)${NC}"
    else
        echo -e "${RED}○ WebSocket代理: 未运行${NC}"
    fi
    
    # 显示UI服务器状态
    if [ -n "$UI_SERVER_PID" ] && ps -p $UI_SERVER_PID > /dev/null; then
        echo -e "${GREEN}● UI服务器: 运行中 (PID: $UI_SERVER_PID)${NC}"
        echo -e "  访问地址: ${GREEN}http://localhost:${HTTP_PORT}${NC}"
    else
        echo -e "${RED}○ UI服务器: 未运行${NC}"
    fi
    
    # 显示API代理服务器状态
    if [ -n "$API_PROXY_PID" ] && ps -p $API_PROXY_PID > /dev/null; then
        echo -e "${GREEN}● API代理服务器: 运行中 (PID: $API_PROXY_PID)${NC}"
    else
        echo -e "${RED}○ API代理服务器: 未运行${NC}"
    fi
    
    # 显示交易系统状态
    TRADING_PID=$(pgrep -f "python.*main.py" | head -1)
    if [ -n "$TRADING_PID" ] && ps -p $TRADING_PID > /dev/null; then
        echo -e "${GREEN}● 交易服务: 运行中 (PID: $TRADING_PID)${NC}"
        echo -e "  运行模式: ${YELLOW}${MODE}${NC}"
    else
        echo -e "${RED}○ 交易服务: 未运行${NC}"
    fi
    
    echo -e "${BLUE}==================================================${NC}"
}

# 修复WebSocket代理问题
fix_websocket_proxy_issues() {
    echo -e "${YELLOW}正在尝试修复WebSocket代理问题...${NC}"
    echo -e "1) 安装/更新Socket.IO依赖(Flask-SocketIO、eventlet等)"
    echo -e "2) 安装/更新原生WebSocket依赖(websockets)"
    echo -e "3) 切换到原生WebSocket模式(推荐)"
    echo -e "4) 重启WebSocket代理"
    echo -e "5) 返回主菜单"
    read -p "请选择操作 [1-5]: " fix_choice
    
    case $fix_choice in
        1)
            echo -e "${GREEN}安装/更新Socket.IO依赖...${NC}"
            pip install --upgrade Flask-SocketIO flask-cors eventlet gevent python-engineio websocket-client
            echo -e "${GREEN}依赖更新完成${NC}"
            echo -e "按任意键继续..."
            read -n1
            fix_websocket_proxy_issues
            ;;
        2)
            echo -e "${GREEN}安装/更新原生WebSocket依赖...${NC}"
            pip install --upgrade websockets flask flask-cors
            echo -e "${GREEN}依赖更新完成${NC}"
            echo -e "按任意键继续..."
            read -n1
            fix_websocket_proxy_issues
            ;;
        3)
            echo -e "${GREEN}正在设置原生WebSocket模式...${NC}"
            # 检查原生WebSocket代理文件是否存在
            if [ ! -f "$TRADING_SYSTEM_DIR/src/native_websocket_proxy.py" ]; then
                echo -e "${RED}找不到原生WebSocket代理文件！${NC}"
                echo -e "${YELLOW}请确保文件存在: $TRADING_SYSTEM_DIR/src/native_websocket_proxy.py${NC}"
                echo -e "按任意键继续..."
                read -n1
                fix_websocket_proxy_issues
                return
            fi
            
            # 创建启动脚本(如果不存在)
            if [ ! -f "$TRADING_SYSTEM_DIR/src/start_native_websocket.py" ]; then
                echo -e "${YELLOW}正在创建原生WebSocket启动脚本...${NC}"
                cat > "$TRADING_SYSTEM_DIR/src/start_native_websocket.py" << 'EOF'
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
原生WebSocket代理启动器脚本
"""
import os
import sys
import logging
import traceback

# 确保正确设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))  # 项目根目录
sys.path.insert(0, parent_dir)

# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NativeWebSocketStarter")

try:
    logger.info("启动原生WebSocket代理服务器...")
    
    # 检查所需模块
    required_packages = ['websockets']
    missing_packages = []
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing_packages.append(pkg)
    
    if missing_packages:
        logger.error(f"缺少必要的包: {', '.join(missing_packages)}")
        logger.info("正在安装缺失的包...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            logger.info("依赖安装完成，继续启动...")
        except Exception as e:
            logger.error(f"安装依赖失败: {str(e)}")
            sys.exit(1)
    
    # 引入原生WebSocket代理
    from native_websocket_proxy import get_instance
    
    # 创建并启动代理
    proxy = get_instance()
    proxy.start()
    
    logger.info("WebSocket代理服务已启动，按Ctrl+C停止...")
    
    # 保持运行状态
    import time
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    logger.info("收到停止信号，正在关闭...")
    if 'proxy' in locals():
        proxy.stop()
except Exception as e:
    logger.error(f"启动WebSocket代理时出错: {str(e)}")
    traceback.print_exc()
    sys.exit(1)
EOF
                chmod +x "$TRADING_SYSTEM_DIR/src/start_native_websocket.py"
            fi
            
            # 安装依赖
            echo -e "${YELLOW}安装原生WebSocket所需的依赖...${NC}"
            pip install --upgrade websockets flask flask-cors
            
            # 重启代理
            echo -e "${YELLOW}重启WebSocket代理为原生模式...${NC}"
            # 先终止可能运行的进程
            if [ -n "$WS_PROXY_PID" ] && ps -p $WS_PROXY_PID > /dev/null; then
                kill $WS_PROXY_PID 2>/dev/null || true
                sleep 1
            fi
            # 杀死任何其他可能的WebSocket代理进程
            pkill -f "python.*websocket" 2>/dev/null || true
            sleep 1
            
            # 重新启动
            start_websocket_proxy
            echo -e "${GREEN}WebSocket代理已在原生模式下重启${NC}"
            echo -e "按任意键继续..."
            read -n1
            show_interactive_menu
            ;;
        4)
            echo -e "${YELLOW}重启WebSocket代理...${NC}"
            # 先终止可能运行的进程
            if [ -n "$WS_PROXY_PID" ] && ps -p $WS_PROXY_PID > /dev/null; then
                kill $WS_PROXY_PID 2>/dev/null || true
                sleep 1
            fi
            # 杀死任何其他可能的WebSocket代理进程
            pkill -f "python.*websocket" 2>/dev/null || true
            sleep 1
            
            # 重新启动
            start_websocket_proxy
            echo -e "${GREEN}WebSocket代理已重启${NC}"
            echo -e "按任意键继续..."
            read -n1
            show_interactive_menu
            ;;
        *)
            show_interactive_menu
            ;;
    esac
}

# 设置终止信号处理
trap cleanup EXIT INT TERM

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
    kill_existing_processes  # 确保先清理掉可能运行的旧实例
    start_websocket_proxy
    echo -e "${YELLOW}服务已启动。${NC}"
    show_system_status
    echo -e "${YELLOW}按 Ctrl+C 停止服务...${NC}"
    wait
elif [ "$UI_ONLY" = true ]; then
    # 只启动UI界面、WebSocket代理和API代理服务器
    kill_existing_processes  # 确保先清理掉可能运行的旧实例
    start_websocket_proxy
    start_api_proxy
    start_ui
    echo -e "${YELLOW}服务已启动。${NC}"
    show_system_status
    echo -e "${YELLOW}按 Ctrl+C 停止服务...${NC}"
    wait
else
    # 启动完整系统
    kill_existing_processes  # 确保先清理掉可能运行的旧实例
    start_websocket_proxy
    
    echo -e "${GREEN}启动交易服务 (${MODE}模式)...${NC}"
    cd "$TRADING_SYSTEM_DIR/src"
    # 在后台启动UI服务器，以便可以在同一终端显示交易服务日志
    start_ui
    echo -e "${YELLOW}所有服务已启动。${NC}"
    show_system_status
    echo -e "${GREEN}现在启动交易服务主程序 (${MODE}模式)...${NC}"
    
    # 启动主交易程序
    python main.py --config "$CONFIG_FILE" --mode "$MODE"
fi

echo "交易系统已关闭。"
