#!/bin/bash

# 设置交易系统作为系统服务的脚本
echo "准备将交易系统设置为系统服务..."

# 获取脚本所在目录的父目录（trading_system目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRADING_SYSTEM_DIR="$( dirname "$SCRIPT_DIR" )"
ROOT_DIR="$( dirname "$TRADING_SYSTEM_DIR" )"

# 创建服务文件
SERVICE_NAME="btc-trading-rl"
SERVICE_FILE="/tmp/${SERVICE_NAME}.service"

cat > $SERVICE_FILE << EOL
[Unit]
Description=BTC Trading RL - 自动化比特币期货交易服务
After=network.target

[Service]
User=$(whoami)
WorkingDirectory=${TRADING_SYSTEM_DIR}
ExecStart=/bin/bash ${TRADING_SYSTEM_DIR}/scripts/start_prod_trading.sh
Restart=on-failure
RestartSec=10
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=${SERVICE_NAME}

[Install]
WantedBy=multi-user.target
EOL

echo "服务文件已创建。"

# 设置文件权限
chmod +x "${TRADING_SYSTEM_DIR}/scripts/start_prod_trading.sh"
chmod +x "${TRADING_SYSTEM_DIR}/scripts/start_test_trading.sh"
chmod +x "${TRADING_SYSTEM_DIR}/scripts/install_dependencies.sh"

# 提示用户安装服务
echo "请运行以下命令安装服务（需要管理员权限）："
echo "  sudo cp $SERVICE_FILE /etc/systemd/system/"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl enable $SERVICE_NAME"
echo ""
echo "然后，您可以使用以下命令控制服务："
echo "  启动服务：sudo systemctl start $SERVICE_NAME"
echo "  停止服务：sudo systemctl stop $SERVICE_NAME"
echo "  查看状态：sudo systemctl status $SERVICE_NAME"
echo "  查看日志：sudo journalctl -u $SERVICE_NAME"