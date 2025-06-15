#!/bin/bash

echo "正在检查WebSocket服务器状态..."

# 检查后台服务是否运行
ws_process=$(ps aux | grep "[w]ebsocket" | grep -v grep)
ui_process=$(ps aux | grep "ui_server" | grep -v grep)

if [ -z "$ws_process" ]; then
    echo "WebSocket服务器未运行，正在重新启动..."
    
    # 尝试启动WebSocket服务器
    cd /home/losesky/crypto-trading-rl
    python -m trading_system.src.websocket_server &
    
    echo "WebSocket服务器已在后台启动"
else
    echo "WebSocket服务器已在运行：$ws_process"
fi

# 确保UI服务器正在运行
if [ -z "$ui_process" ]; then
    echo "UI服务器未运行，正在重新启动..."
    
    cd /home/losesky/crypto-trading-rl
    python -m trading_system.src.ui_server &
    
    echo "UI服务器已在后台启动"
else
    echo "UI服务器已在运行：$ui_process"
fi

# 创建一个测试数据发送脚本
cat > /tmp/send_test_data.py << EOF
import json
import time
import random
import datetime
import websocket

def send_test_data():
    ws = websocket.WebSocket()
    try:
        print("尝试连接到WebSocket服务器...")
        ws.connect("ws://localhost:8765")
        print("已连接到WebSocket服务器")
        
        # 发送市场数据
        for i in range(10):
            timestamp = datetime.datetime.now()
            price = 60000 + random.uniform(-1000, 1000)
            volume = random.uniform(0.5, 10)
            
            market_data = {
                "type": "market_data",
                "data": {
                    "symbol": "BTCUSDT",
                    "price": price,
                    "timestamp": timestamp.timestamp() * 1000,
                    "volume": volume,
                    "high": price + random.uniform(100, 500),
                    "low": price - random.uniform(100, 500),
                    "open": price - random.uniform(-200, 200),
                    "close": price
                }
            }
            
            print(f"发送市场数据 #{i+1}: 价格 = {price}")
            ws.send(json.dumps(market_data))
            time.sleep(1)
            
        # 发送预测数据
        prediction = {
            "type": "prediction",
            "data": {
                "timestamp": datetime.datetime.now().timestamp() * 1000,
                "action": random.choice(["BUY", "SELL", "HOLD"]),
                "confidence": random.uniform(0.6, 0.95),
                "values": {
                    "buy": random.uniform(0.1, 0.9),
                    "sell": random.uniform(0.1, 0.9),
                    "hold": random.uniform(0.1, 0.9)
                }
            }
        }
        
        print(f"发送预测数据: 动作 = {prediction['data']['action']}")
        ws.send(json.dumps(prediction))
        
        # 发送账户数据
        account = {
            "type": "status_update",
            "data": {
                "is_running": True,
                "start_time": (datetime.datetime.now() - datetime.timedelta(minutes=30)).timestamp() * 1000,
                "account_info": {
                    "available_balance": 15000,
                    "margin_balance": 5000,
                    "daily_pnl": 120.5,
                    "total_pnl": 450.75
                },
                "trade_count": 5,
                "last_trade_time": (datetime.datetime.now() - datetime.timedelta(minutes=5)).timestamp() * 1000
            }
        }
        
        print("发送账户状态数据")
        ws.send(json.dumps(account))
        
        print("测试数据发送完成")
        
    except Exception as e:
        print(f"发送测试数据时出错: {e}")
    finally:
        ws.close()

if __name__ == "__main__":
    send_test_data()
EOF

echo "测试数据发送脚本已创建"
echo "正在安装websocket-client包..."
pip install websocket-client

echo "发送测试数据到WebSocket服务器..."
python /tmp/send_test_data.py

echo "网页可能需要刷新才能显示测试数据"
echo "完成"
