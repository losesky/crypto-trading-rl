#!/bin/bash
# 强化学习交易系统启动脚本 - 优化版
# 启动WebSocket代理和交易系统

# 获取脚本所在目录的父目录（trading_system目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRADING_SYSTEM_DIR="$( dirname "$SCRIPT_DIR" )"
ROOT_DIR="$( dirname "$TRADING_SYSTEM_DIR" )"

# 显示帮助信息
show_help() {
    echo "RL交易系统启动脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help            显示帮助信息"
    echo "  -t, --test            启动测试环境 (默认)"
    echo "  -p, --prod            启动生产环境"
    echo "  -u, --ui-only         只启动UI界面"
    echo "  --new-ui              使用新版UI界面"
    echo "  --ws-proxy            单独启动WebSocket代理"
    echo ""
}

# 默认参数
MODE="test"
UI_ONLY=false
NEW_UI=false
WS_PROXY_ONLY=false

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
        --new-ui)
            NEW_UI=true
            shift
            ;;
        --ws-proxy)
            WS_PROXY_ONLY=true
            shift
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检测是否存在虚拟环境并激活
if [ -d "$ROOT_DIR/venv" ]; then
    echo "激活虚拟环境..."
    source "$ROOT_DIR/venv/bin/activate"
else
    echo "警告: 未找到虚拟环境。建议先运行 './scripts/install_dependencies.sh' 建立环境。"
    echo "尝试继续运行，但可能存在依赖问题..."
fi

# 设置配置文件路径
if [ "$MODE" = "test" ]; then
    CONFIG_FILE="$TRADING_SYSTEM_DIR/config/test_config.json"
    echo "使用测试配置: $CONFIG_FILE"
else
    CONFIG_FILE="$TRADING_SYSTEM_DIR/config/prod_config.json"
    echo "使用生产配置: $CONFIG_FILE"
fi

# 创建日志目录
mkdir -p "$TRADING_SYSTEM_DIR/logs"

# 将必要的目录添加到PYTHONPATH，确保能够正确导入模块
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

# 如果只是启动WebSocket代理
if [ "$WS_PROXY_ONLY" = true ]; then
    echo "启动WebSocket代理服务..."
    cd "$TRADING_SYSTEM_DIR/src"
    python -c "from websocket_proxy import get_instance; proxy = get_instance(); proxy.start(); import time; print('WebSocket代理已启动，按Ctrl+C停止...'); time.sleep(3600)"
    exit 0
fi

# 如果是只启动UI，将主页替换为新版，并退出
if [ "$UI_ONLY" = true ]; then
    echo "准备启动UI界面..."
    
    # 根据参数决定使用哪个UI版本
    if [ "$NEW_UI" = true ]; then
        echo "使用新版UI界面..."
        # 复制新UI文件到默认位置
        cp "$TRADING_SYSTEM_DIR/ui/index-new.html" "$TRADING_SYSTEM_DIR/ui/index.html"
        cp "$TRADING_SYSTEM_DIR/ui/app-new.js" "$TRADING_SYSTEM_DIR/ui/app.js"
    else
        echo "使用标准UI界面..."
        # 恢复原始UI文件（如果备份存在）
        if [ -f "$TRADING_SYSTEM_DIR/ui/index.html.bak" ]; then
            cp "$TRADING_SYSTEM_DIR/ui/index.html.bak" "$TRADING_SYSTEM_DIR/ui/index.html"
        fi
        if [ -f "$TRADING_SYSTEM_DIR/ui/app.js.bak" ]; then
            cp "$TRADING_SYSTEM_DIR/ui/app.js.bak" "$TRADING_SYSTEM_DIR/ui/app.js"
        fi
    fi
    
    # 启动简单的HTTP服务器提供UI
    echo "启动本地Web服务器..."
    cd "$TRADING_SYSTEM_DIR/ui"
    python -m http.server 8090
    exit 0
fi

# 准备启动完整交易系统
echo "准备启动RL交易系统..."

# 安装缺少的依赖
pip install -q flask-cors flask-socketio python-socketio websocket-client

# 备份当前UI文件
if [ "$NEW_UI" = true ] && [ ! -f "$TRADING_SYSTEM_DIR/ui/index.html.bak" ]; then
    echo "备份原始UI文件..."
    cp "$TRADING_SYSTEM_DIR/ui/index.html" "$TRADING_SYSTEM_DIR/ui/index.html.bak"
    cp "$TRADING_SYSTEM_DIR/ui/app.js" "$TRADING_SYSTEM_DIR/ui/app.js.bak"
    
    # 使用新版UI
    echo "切换到新版UI界面..."
    cp "$TRADING_SYSTEM_DIR/ui/index-new.html" "$TRADING_SYSTEM_DIR/ui/index.html"
    cp "$TRADING_SYSTEM_DIR/ui/app-new.js" "$TRADING_SYSTEM_DIR/ui/app.js"
fi

# 启动交易系统
echo "启动交易系统..."
cd "$TRADING_SYSTEM_DIR/src"
# 通过直接在src目录中运行main.py解决相对导入问题
python main.py --config "$CONFIG_FILE" --mode $MODE

echo "交易系统已启动！"
echo "您可以通过访问 http://localhost:8090 查看交易仪表盘。"
echo "按 Ctrl+C 停止交易服务..."

# 捕获Ctrl+C信号
trap "echo '收到停止信号，正在关闭...' && exit 0" SIGINT SIGTERM

# 等待用户Ctrl+C
wait

echo "交易系统已关闭。"
