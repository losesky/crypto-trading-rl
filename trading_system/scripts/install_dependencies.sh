#!/bin/bash

# 安装交易系统依赖项脚本
echo "正在安装交易系统所需依赖项..."

# 检测是否存在虚拟环境
if [ -d "../venv" ]; then
    echo "检测到虚拟环境，激活中..."
    source ../venv/bin/activate
else
    echo "创建新的虚拟环境..."
    python -m venv ../venv
    source ../venv/bin/activate
fi

# 安装依赖项
echo "安装Python依赖..."
pip install -q ccxt pandas numpy websocket-client python-dotenv Flask Flask-SocketIO flask-cors requests plotly dash pyjwt

# 检查依赖项是否安装成功
echo "验证安装..."
python -c "import ccxt, pandas, numpy, websocket, dotenv, flask, flask_socketio, flask_cors, requests, plotly, dash, jwt" && echo "依赖项安装成功！" || echo "警告：部分依赖项可能未正确安装"

echo "安装完成！"
echo "您可以通过运行 './scripts/start_test_trading.sh' 来启动测试环境交易."