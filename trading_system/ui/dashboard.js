/**
 * RL交易系统 - 主控制面板脚本
 * 版本: 1.0
 * 日期: 2025-06-12
 */

// 全局变量
let priceChart = null;
let predictionChart = null;
let socket = null;
let isConnected = false;
let marketDataCache = [];
let positionDataCache = [];
let orderDataCache = [];
let predictionDataCache = [];
let alertDataCache = [];
let currentTimeframe = '1h';
let systemStatus = {};
let lastUpdateTime = new Date();
let reconnectAttempts = 0;
let maxReconnectAttempts = 10;
let reconnectInterval = 5000; // 毫秒

// 初始化函数
function initialize() {
    console.log('初始化交易系统控制面板...');
    
    // 初始化图表
    initializeCharts();
    
    // 初始化Socket.IO连接
    initializeSocketConnection();
    
    // 初始化UI控制器
    UIController.initialize();
    
    // 加载初始数据
    loadInitialData();
    
    // 设置定期更新
    setupPeriodicUpdates();
    
    console.log('控制面板初始化完成');
}

// 初始化图表
function initializeCharts() {
    // 初始化价格图表
    priceChart = ChartComponent.initPriceChart('price-chart');
    
    // 初始化预测图表
    predictionChart = ChartComponent.initPredictionChart('prediction-chart');
    
    // 设置时间框架变更处理器
    window.onTimeframeChange = function(timeframe) {
        currentTimeframe = timeframe;
        updatePriceChartTimeframe(timeframe);
    };
}

// 更新价格图表的时间框架
function updatePriceChartTimeframe(timeframe) {
    // 根据时间框架调整X轴
    switch (timeframe) {
        case '5m':
            priceChart.options.scales.x.time.unit = 'minute';
            priceChart.options.scales.x.time.displayFormats.minute = 'HH:mm';
            break;
        case '1h':
            priceChart.options.scales.x.time.unit = 'hour';
            priceChart.options.scales.x.time.displayFormats.hour = 'HH:mm';
            break;
        case '1d':
            priceChart.options.scales.x.time.unit = 'day';
            priceChart.options.scales.x.time.displayFormats.day = 'MM-dd';
            break;
    }
    
    // 重新加载相应时间范围的数据
    loadMarketDataForTimeframe(timeframe);
}

// 加载指定时间框架的市场数据
function loadMarketDataForTimeframe(timeframe) {
    // 请求特定时间框架的市场数据
    const endpoint = `/api/market_data?timeframe=${timeframe}`;
    
    fetch(endpoint)
        .then(response => response.json())
        .then(data => {
            if (data && Array.isArray(data) && data.length > 0) {
                // 重置图表数据
                priceChart.data.labels = [];
                priceChart.data.datasets[0].data = [];
                
                // 添加新数据
                ChartComponent.updatePriceChart(priceChart, data);
            }
        })
        .catch(error => {
            console.error('加载市场数据失败:', error);
        });
}

// 初始化WebSocket连接
function initializeSocketConnection() {
    // 创建WebSocket连接 (不使用Socket.IO)
    try {
        // 获取当前主机地址的WebSocket URL
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.hostname;
        const wsUrl = `${protocol}//${host}:8765`;
        
        console.log('尝试连接到WebSocket服务器:', wsUrl);
        
        // 关闭已存在的连接
        if (socket && socket.readyState !== WebSocket.CLOSED) {
            socket.close();
        }
        
        // 创建原生WebSocket连接
        socket = new WebSocket(wsUrl);
        
        // 连接事件
        socket.onopen = () => {
            console.log('WebSocket连接已建立');
            isConnected = true;
            reconnectAttempts = 0;
            UIStateManager.updateConnectionStatus(true);
            
            // 发送初始订阅消息
            socket.send(JSON.stringify({
                type: "subscribe",
                topics: ["market_data", "position", "order", "prediction", "alert"]
            }));
        };
        
        // 断开连接事件
        socket.onclose = () => {
            console.log('WebSocket连接已断开');
            isConnected = false;
            UIStateManager.updateConnectionStatus(false);
            attemptReconnect();
        };
        
        // 错误事件
        socket.onerror = (error) => {
            console.error('WebSocket错误:', error);
            isConnected = false;
            UIStateManager.updateConnectionStatus(false);
        };
        
        // 消息事件
        socket.onmessage = (event) => {
            lastUpdateTime = new Date();
            
            try {
                const message = JSON.parse(event.data);
                const messageType = message.type || '';
                
                // 根据消息类型处理不同数据
                switch(messageType) {
                    case 'market_update':
                        handleMarketUpdate(message);
                        break;
                    case 'position_update':
                        handlePositionUpdate(message);
                        break;
                    case 'order_update':
                        handleOrderUpdate(message);
                        break;
                    case 'prediction_update':
                        handlePredictionUpdate(message);
                        break;
                    case 'status_update':
                        handleStatusUpdate(message);
                        break;
                    case 'system_message':
                        console.log('系统消息:', message.message);
                        break;
                    default:
                        console.log('收到未知类型消息:', message);
                }
            } catch (error) {
                console.error('处理WebSocket消息时出错:', error);
            }
        };
    } catch (error) {
        console.error('初始化WebSocket连接时出错:', error);
    }
}

// 处理市场数据更新
function handleMarketUpdate(data) {
    // 添加到缓存
    marketDataCache.push(data);
        
    // 添加到缓存
    marketDataCache.push(data);
    
    // 限制缓存大小
    if (marketDataCache.length > 1000) {
        marketDataCache.shift();
    }
    
    // 更新图表
    ChartComponent.updatePriceChart(priceChart, [data]);
    
    // 更新仓位信息 (如果有持仓)
    if (positionDataCache.length > 0) {
        const lastPosition = positionDataCache[positionDataCache.length - 1];
        UIStateManager.updatePositionInfo(lastPosition, data);
    }
    
    // 仓位数据更新事件
    socket.on('position_update', (data) => {
        // 添加到缓存
        positionDataCache.push(data);
        
        // 限制缓存大小
        if (positionDataCache.length > 100) {
            positionDataCache.shift();
        }
        
        // 更新UI
        if (marketDataCache.length > 0) {
            const lastMarketData = marketDataCache[marketDataCache.length - 1];
            UIStateManager.updatePositionInfo(data, lastMarketData);
        } else {
            UIStateManager.updatePositionInfo(data, null);
        }
    });
    
    // 订单更新事件
    socket.on('order_update', (data) => {
        // 添加到缓存
        orderDataCache.push(data);
        
        // 限制缓存大小
        if (orderDataCache.length > 100) {
            orderDataCache.shift();
        }
        
        // 更新UI
        UIStateManager.updateOrdersList(orderDataCache);
    });
    
    // 预测更新事件
    socket.on('prediction_update', (data) => {
        // 添加到缓存
        predictionDataCache.push(data);
        
        // 限制缓存大小
        if (predictionDataCache.length > 200) {
            predictionDataCache.shift();
        }
        
        // 更新UI
        UIStateManager.updatePrediction(data);
        
        // 更新预测图表
        ChartComponent.updatePredictionChart(predictionChart, data);
    });
    
    // 警报事件
    socket.on('alert', (data) => {
        // 添加到缓存
        alertDataCache.push(data);
        
        // 限制缓存大小
        if (alertDataCache.length > 50) {
            alertDataCache.shift();
        }
        
        // 更新UI
        UIStateManager.updateAlertsList(alertDataCache);
        
        // 如果是重要警报，可以显示通知
        if (data.severity === 'critical') {
            showNotification('系统警报', data.message, 'error');
        }
    });
    
    // 状态更新事件
    socket.on('status_update', (data) => {
        systemStatus = data;
        
        // 更新系统状态UI
        UIStateManager.updateSystemStatus(data);
        
        // 更新账户信息
        if (data.account_info) {
            UIStateManager.updateAccountInfo(data.account_info);
        }
    });
}

// 尝试重新连接WebSocket
function attemptReconnect() {
    if (reconnectAttempts >= maxReconnectAttempts) {
        console.log('达到最大重连尝试次数，放弃重连');
        showNotification('连接失败', '无法连接到服务器，请检查网络连接或刷新页面', 'error');
        return;
    }
    
    reconnectAttempts++;
    
    // 计算退避时间
    const backoffTime = reconnectInterval * Math.pow(1.5, reconnectAttempts - 1);
    
    console.log(`尝试重新连接 (${reconnectAttempts}/${maxReconnectAttempts})，等待 ${backoffTime}ms...`);
    
    setTimeout(() => {
        if (!isConnected) {
            console.log('重新连接中...');
            socket.connect();
        }
    }, backoffTime);
}

// 加载初始数据
function loadInitialData() {
    // 加载系统状态
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            systemStatus = data;
            UIStateManager.updateSystemStatus(data);
            
            if (data.account_info) {
                UIStateManager.updateAccountInfo(data.account_info);
            }
            
            if (data.current_position) {
                UIStateManager.updatePositionInfo(data.current_position, null);
            }
        })
        .catch(error => {
            console.error('加载系统状态失败:', error);
        });
    
    // 加载市场数据
    fetch('/api/market_data')
        .then(response => response.json())
        .then(data => {
            if (data && Array.isArray(data)) {
                marketDataCache = data;
                ChartComponent.updatePriceChart(priceChart, data);
            }
        })
        .catch(error => {
            console.error('加载市场数据失败:', error);
        });
    
    // 加载订单数据
    fetch('/api/orders')
        .then(response => response.json())
        .then(data => {
            if (data && Array.isArray(data)) {
                orderDataCache = data;
                UIStateManager.updateOrdersList(data);
            }
        })
        .catch(error => {
            console.error('加载订单数据失败:', error);
        });
    
    // 加载预测数据
    fetch('/api/predictions')
        .then(response => response.json())
        .then(data => {
            if (data && Array.isArray(data) && data.length > 0) {
                predictionDataCache = data;
                
                // 更新最新预测
                const latestPrediction = data[data.length - 1];
                UIStateManager.updatePrediction(latestPrediction);
                
                // 更新预测图表
                data.forEach(prediction => {
                    ChartComponent.updatePredictionChart(predictionChart, prediction);
                });
            }
        })
        .catch(error => {
            console.error('加载预测数据失败:', error);
        });
    
    // 加载警报数据
    fetch('/api/alerts')
        .then(response => response.json())
        .then(data => {
            if (data && Array.isArray(data)) {
                alertDataCache = data;
                UIStateManager.updateAlertsList(data);
            }
        })
        .catch(error => {
            console.error('加载警报数据失败:', error);
        });
}

// 设置定期更新
function setupPeriodicUpdates() {
    // 每30秒检查连接状态
    setInterval(() => {
        // 检查上次更新时间，如果超过30秒没有更新，认为连接可能有问题
        const now = new Date();
        const timeSinceLastUpdate = now - lastUpdateTime;
        
        if (timeSinceLastUpdate > 30000 && isConnected) {
            console.warn('数据更新超时，可能连接已断开');
            socket.disconnect();
            setTimeout(() => {
                socket.connect();
            }, 1000);
        }
        
        // 如果没有连接，尝试重连
        if (!isConnected) {
            attemptReconnect();
        }
    }, 30000);
    
    // 更新运行时间显示
    setInterval(() => {
        if (systemStatus && systemStatus.start_time && systemStatus.is_running) {
            const startTime = new Date(systemStatus.start_time);
            const now = new Date();
            const uptime = Math.floor((now - startTime) / 1000);
            
            const uptimeElement = document.getElementById('uptime');
            if (uptimeElement) {
                uptimeElement.textContent = formatDuration(uptime);
            }
        }
    }, 1000);
}

// 显示通知
function showNotification(title, message, type = 'info') {
    // 检查浏览器通知支持
    if ('Notification' in window) {
        if (Notification.permission === 'granted') {
            new Notification(title, {
                body: message,
                icon: type === 'error' ? '/static/img/error-icon.png' : '/static/img/info-icon.png'
            });
        } else if (Notification.permission !== 'denied') {
            Notification.requestPermission().then(permission => {
                if (permission === 'granted') {
                    new Notification(title, {
                        body: message,
                        icon: type === 'error' ? '/static/img/error-icon.png' : '/static/img/info-icon.png'
                    });
                }
            });
        }
    }
    
    // 备用：在页面上显示警报
    const alertClass = type === 'error' ? 'alert-danger' : 
                     type === 'warning' ? 'alert-warning' :
                     'alert-info';
    
    const alertElement = document.createElement('div');
    alertElement.className = `alert ${alertClass} alert-dismissible fade show position-fixed`;
    alertElement.style.top = '20px';
    alertElement.style.right = '20px';
    alertElement.style.zIndex = '9999';
    
    alertElement.innerHTML = `
        <strong>${title}</strong>: ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    document.body.appendChild(alertElement);
    
    // 5秒后自动关闭
    setTimeout(() => {
        alertElement.classList.remove('show');
        setTimeout(() => {
            alertElement.remove();
        }, 300);
    }, 5000);
}

// 当文档加载完成时初始化
document.addEventListener('DOMContentLoaded', initialize);