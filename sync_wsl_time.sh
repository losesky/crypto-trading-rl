#!/bin/bash
# 该脚本用于同步WSL时间与Windows主机时间
# 可以在启动交易系统前运行

echo "正在同步WSL时间与Windows主机..."

# 获取Windows系统时间并同步到WSL
HOST_TIME=$(powershell.exe -Command "(Get-Date).ToString('yyyy-MM-dd HH:mm:ss')")
if [ $? -eq 0 ]; then
    echo "Windows主机时间: $HOST_TIME"
    
    # 将Windows时间格式转换为Linux可用格式
    FORMATTED_TIME=$(echo "$HOST_TIME" | sed 's/\r//g')
    
    # 使用sudo设置WSL时间，需要输入密码
    echo "正在设置WSL系统时间..."
    sudo date -s "$FORMATTED_TIME"
    
    if [ $? -eq 0 ]; then
        echo "WSL时间同步成功!"
    else
        echo "WSL时间同步失败，请确保有sudo权限"
    fi
else
    echo "无法获取Windows系统时间"
fi

# 显示当前WSL时间
echo "当前WSL时间: $(date)"

# 运行NTP同步以确保更精确
echo "尝试使用NTP服务进行进一步同步..."
sudo ntpdate -u time.windows.com || echo "NTP同步失败，可能需要安装ntpdate: sudo apt install ntpdate"

echo "同步完成！"
