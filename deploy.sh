#!/bin/bash

echo "🚀 部署BTC交易强化学习项目"

# 检查Python版本
python_version=$(python3 --version 2>&1)
echo "Python版本: $python_version"

# 创建必要的目录
mkdir -p btc_rl/logs/episodes
mkdir -p btc_rl/logs/tb
mkdir -p btc_rl/data
mkdir -p btc_rl/models

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
echo "安装依赖包..."
pip install -r requirements.txt

# 检查数据文件
if [ ! -f "btc_rl/data/BTC_hourly.csv" ]; then
    echo "⚠️  警告: 缺少BTC价格数据文件"
    echo "请将BTC历史数据放置在 btc_rl/data/BTC_hourly.csv"
    echo "数据格式: timestamp,open,high,low,close,volume"
fi

# 预处理数据（如果存在原始数据）
if [ -f "btc_rl/data/BTC_hourly.csv" ]; then
    echo "预处理数据..."
    python -m btc_rl.src.preprocessing --csv btc_rl/data/BTC_hourly.csv
fi

echo "✅ 部署完成！"
echo ""
echo "启动说明:"
echo "方法1 (推荐): 一键启动所有服务"
echo "    ./start.sh"
echo ""
echo "方法2: 单独启动各个服务"
echo "1. 启动WebSocket服务器: python -m btc_rl.src.websocket_server"
echo "2. 启动HTTP服务器: python -m btc_rl.src.http_server" 
echo "3. 在浏览器中打开: http://localhost:8080/index.html"
echo "4. 启动模型训练: python -m btc_rl.src.train_sac"