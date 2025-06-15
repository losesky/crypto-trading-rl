/**
 * RL交易系统 - 前端应用主文件(重构版)
 * 版本: 3.0
 * 日期: 2025-06-14
 */

// 导入WebSocket管理器
// const WebSocketManager = require('./websocket-manager');

// =====================================
// 全局配置 
// =====================================
const CONFIG = {
    // WebSocket配置
    LOCAL_WS_URL: `ws://${window.location.hostname}:8095`,  // 本地WebSocket服务URL
    API_BASE_URL: `http://${window.location.hostname}:8090`, // 本地后端API URL
    BINANCE_WS_URL: `wss://fstream.binancefuture.com/ws`,    // 币安WebSocket URL
    
    // 图表配置
    CHART_MAX_POINTS: 200,
    CHART_DEFAULT_OPTIONS: {
        animation: false,
        tension: 0.3,
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            mode: 'index',
            intersect: false,
        },
        plugins: {
            tooltip: {
                enabled: true
            },
            legend: {
                display: true,
                position: 'top'
            }
        },
        scales: {
            x: {
                type: 'time',
                time: {
                    unit: 'minute',
                    tooltipFormat: 'yyyy-MM-dd HH:mm',
                    displayFormats: {
                        minute: 'HH:mm',
                        hour: 'MM-dd HH:mm'
                    }
                },
                title: {
                    display: true,
                    text: '时间'
                }
            },
            y: {
                type: 'linear',
                title: {
                    display: true,
                    text: '价格 (USDT)'
                }
            }
        }
    },
    
    // 更新间隔
    UPDATE_INTERVAL: 5000,           // 常规更新间隔(毫秒)
    CHART_UPDATE_INTERVAL: 1000,     // 图表更新间隔(毫秒)
    STATUS_CHECK_INTERVAL: 10000,    // 状态检查间隔(毫秒)
    
    // 数据限制
    MAX_MARKET_DATA_POINTS: 300,     // 市场数据最大点数
    MAX_PREDICTION_POINTS: 50,       // 预测数据最大点数
    MAX_ORDERS: 20,                  // 最大订单数
    MAX_ALERTS: 20,                  // 最大警报数
    
    // 模拟数据配置 (仅用于无连接时)
    USE_MOCK_DATA: false,            // 是否使用模拟数据
    MOCK_DATA_START_PRICE: 100000    // 模拟数据起始价格
};

// =====================================
// 应用程序状态
// =====================================
const app = {
    // 连接状态
    wsConnected: false,              // WebSocket连接状态
    apiConnected: false,             // API连接状态
    lastUpdateTime: 0,               // 最后更新时间
    connectionType: null,            // 连接类型："local", "remote", "mock"
    
    // WebSocket连接
    ws: null,                        // WebSocket管理器实例
    
    // 数据存储
    marketData: [],                  // 市场数据序列
    predictionData: [],              // 预测数据序列
    positionData: null,              // 当前仓位数据
    orderData: [],                   // 订单数据
    alertData: [],                   // 警报数据
    systemStatus: {                  // 系统状态
        is_running: false,
        is_paused: false,
        account_balance: 0,
        margin_balance: 0,
        unrealized_pnl: 0
    },
    
    // 图表实例
    priceChart: null,                // 价格图表实例
    predictionChart: null,           // 预测图表实例
    
    // 定时器
    updateTimers: {},                // 各种更新定时器
    
    // UI元素缓存
    elements: {},                    // 缓存的UI元素
    
    // 功能标志
    isLoaded: false,                 // 应用是否已加载
    isDarkMode: false,               // 是否为暗黑模式
    isInitialDataLoaded: false       // 是否已加载初始数据
};

// =====================================
// 初始化函数
// =====================================

/**
 * 应用初始化入口
 */
function initializeApp() {
    console.log('初始化RL交易系统前端 v3.0...');
    
    // 缓存常用UI元素
    cacheUIElements();
    
    // 初始化图表
    initializeCharts();
    
    // 初始化WebSocket连接
    initializeWebSocket();
    
    // 初始化REST API连接
    checkAPIConnection();
    
    // 绑定UI事件监听器
    setupEventListeners();
    
    // 开始定时更新
    startUpdateTimers();
    
    // 标记应用已加载
    app.isLoaded = true;
    console.log('应用初始化完成');
}

/**
 * 缓存常用UI元素
 */
function cacheUIElements() {
    // 状态指示器
    app.elements.wsStatus = document.getElementById('ws-status');
    app.elements.apiStatus = document.getElementById('api-status');
    app.elements.connectionText = document.getElementById('connection-text');
    app.elements.modeBadge = document.getElementById('mode-badge');
    
    // 状态字段
    app.elements.statusBadge = document.getElementById('status-badge');
    app.elements.runningTime = document.getElementById('running-time');
    app.elements.tradingVolume = document.getElementById('trading-volume');
    app.elements.lastTrade = document.getElementById('last-trade');
    
    // 表格元素
    app.elements.positionTableBody = document.getElementById('position-table-body');
    app.elements.orderTableBody = document.getElementById('order-table-body');
    app.elements.alertTableBody = document.getElementById('alert-table-body');
    
    // 账户信息元素
    app.elements.availableBalance = document.getElementById('available-balance');
    app.elements.marginBalance = document.getElementById('margin-balance');
    app.elements.todayProfit = document.getElementById('today-profit');
    app.elements.totalProfit = document.getElementById('total-profit');
    
    // 控制按钮
    app.elements.actionButtons = document.querySelectorAll('[data-action]');
    app.elements.timeframeButtons = document.querySelectorAll('[data-timeframe]');
    
    // 图表容器
    app.elements.priceChartCanvas = document.getElementById('price-chart');
    app.elements.predictionChartCanvas = document.getElementById('prediction-chart');
    
    // 仓位数据显示
    app.elements.positionSize = document.getElementById('position-size');
    app.elements.positionSide = document.getElementById('position-side');
    app.elements.entryPrice = document.getElementById('entry-price');
    app.elements.currentPrice = document.getElementById('current-price');
    app.elements.unrealizedPnl = document.getElementById('unrealized-pnl');
    app.elements.roe = document.getElementById('roe');
    app.elements.leverage = document.getElementById('leverage');
    
    // 检查是否所有必要元素都已找到
    const missingElements = Object.entries(app.elements)
        .filter(([key, element]) => !element)
        .map(([key]) => key);
    
    if (missingElements.length > 0) {
        console.warn(`未找到以下UI元素: ${missingElements.join(', ')}`);
    }
}

/**
 * 初始化图表
 */
function initializeCharts() {
    // 确保Chart.js已加载
    if (typeof Chart === 'undefined') {
        console.error('Chart.js 未加载！');
        showError('加载图表库失败，请刷新页面');
        return;
    }
    
    // 确保Luxon已加载
    if (typeof luxon === 'undefined') {
        console.error('Luxon 未加载！');
        showError('加载时间处理库失败，请刷新页面');
        return;
    }
    
    // 初始化价格图表
    if (app.elements.priceChartCanvas) {
        const ctx = app.elements.priceChartCanvas.getContext('2d');
        app.priceChart = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [{
                    label: '价格',
                    data: [],
                    borderColor: '#0d6efd',
                    backgroundColor: 'rgba(13, 110, 253, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: true
                }]
            },
            options: {
                ...CONFIG.CHART_DEFAULT_OPTIONS,
                plugins: {
                    ...CONFIG.CHART_DEFAULT_OPTIONS.plugins,
                    title: {
                        display: true,
                        text: 'BTC/USDT 价格'
                    }
                }
            }
        });
    }
    
    // 初始化预测图表
    if (app.elements.predictionChartCanvas) {
        const ctx = app.elements.predictionChartCanvas.getContext('2d');
        app.predictionChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: '买入',
                        data: [],
                        backgroundColor: 'rgba(40, 167, 69, 0.7)',
                        borderColor: '#28a745',
                        pointRadius: 6,
                        pointStyle: 'triangle'
                    },
                    {
                        label: '卖出',
                        data: [],
                        backgroundColor: 'rgba(220, 53, 69, 0.7)',
                        borderColor: '#dc3545',
                        pointRadius: 6,
                        pointStyle: 'triangle'
                    },
                    {
                        label: '持有',
                        data: [],
                        backgroundColor: 'rgba(255, 193, 7, 0.7)',
                        borderColor: '#ffc107',
                        pointRadius: 4,
                        pointStyle: 'circle'
                    }
                ]
            },
            options: {
                ...CONFIG.CHART_DEFAULT_OPTIONS,
                plugins: {
                    ...CONFIG.CHART_DEFAULT_OPTIONS.plugins,
                    title: {
                        display: true,
                        text: '模型预测'
                    }
                },
                scales: {
                    x: {
                        ...CONFIG.CHART_DEFAULT_OPTIONS.scales.x
                    },
                    y: {
                        ...CONFIG.CHART_DEFAULT_OPTIONS.scales.y,
                        min: 0,
                        max: 1,
                        title: {
                            display: true,
                            text: '置信度'
                        }
                    }
                }
            }
        });
    }
}

/**
 * 初始化WebSocket连接
 */
function initializeWebSocket() {
    // 检查WebSocketManager是否可用
    if (typeof WebSocketManager === 'undefined') {
        console.error('WebSocketManager 未加载！');
        fallbackToPolling();
        return;
    }
    
    // 关闭已有连接
    if (app.ws) {
        app.ws.disconnect();
        app.ws = null;
    }
    
    try {
        // 创建WebSocket管理器实例
        app.ws = new WebSocketManager(CONFIG.LOCAL_WS_URL, {
            reconnectInterval: 2000,
            reconnectAttempts: 10,
            debug: true,
            heartbeatInterval: 30000,
            heartbeatMessage: JSON.stringify({ping: Date.now()})
        });
        
        // 设置连接状态回调
        app.ws.on('open', handleWebSocketOpen);
        app.ws.on('close', handleWebSocketClose);
        app.ws.on('error', handleWebSocketError);
        app.ws.on('reconnect', handleWebSocketReconnect);
        
        // 设置数据处理回调
        app.ws.on('market_update', handleMarketUpdate);
        app.ws.on('position_update', handlePositionUpdate);
        app.ws.on('prediction_update', handlePredictionUpdate);
        app.ws.on('order_update', handleOrderUpdate);
        app.ws.on('alert', handleAlertUpdate);
        app.ws.on('status_update', handleStatusUpdate);
        app.ws.on('heartbeat', handleHeartbeat);
        
        app.connectionType = 'local';
        console.log('WebSocket初始化完成');
        
    } catch (error) {
        console.error('初始化WebSocket失败:', error);
        fallbackToPolling();
    }
}

/**
 * 检查API连接状态
 */
function checkAPIConnection() {
    fetch(`${CONFIG.API_BASE_URL}/api/status`)
        .then(response => {
            app.apiConnected = response.ok;
            updateConnectionStatus();
            return response.json();
        })
        .then(data => {
            if (data && !app.isInitialDataLoaded) {
                updateSystemStatus(data);
            }
        })
        .catch(error => {
            console.error('API连接检查失败:', error);
            app.apiConnected = false;
            updateConnectionStatus();
        });
}

/**
 * 设置事件监听器
 */
function setupEventListeners() {
    // 控制按钮点击事件
    app.elements.actionButtons.forEach(button => {
        button.addEventListener('click', (e) => {
            const action = e.currentTarget.dataset.action;
            sendCommand(action);
        });
    });
    
    // 时间框架按钮点击事件
    app.elements.timeframeButtons.forEach(button => {
        button.addEventListener('click', (e) => {
            const timeframe = e.currentTarget.dataset.timeframe;
            setTimeframe(timeframe);
        });
    });
    
    // 监听窗口大小变化，调整图表
    window.addEventListener('resize', () => {
        if (app.priceChart) app.priceChart.resize();
        if (app.predictionChart) app.predictionChart.resize();
    });
}

/**
 * 启动定时更新
 */
function startUpdateTimers() {
    // 检查连接状态
    app.updateTimers.connectionCheck = setInterval(() => {
        if (app.ws && !app.ws.isActive()) {
            app.wsConnected = false;
            updateConnectionStatus();
        }
        checkAPIConnection();
    }, CONFIG.STATUS_CHECK_INTERVAL);
    
    // 如果没有WebSocket连接，则启用轮询
    if (!app.ws || app.connectionType === 'polling') {
        startPolling();
    }
    
    // 更新运行时间
    app.updateTimers.runningTime = setInterval(() => {
        updateRunningTime();
    }, 1000);
}

// =====================================
// 数据处理函数
// =====================================

/**
 * 处理WebSocket打开事件
 */
function handleWebSocketOpen(event) {
    console.log('WebSocket连接已建立');
    app.wsConnected = true;
    app.connectionType = 'local';
    updateConnectionStatus();
    
    // 发送一个ping以确保连接正常
    if (app.ws) {
        setTimeout(() => {
            app.ws.send({ping: Date.now()});
        }, 1000);
    }
}

/**
 * 处理WebSocket关闭事件
 */
function handleWebSocketClose(event) {
    console.log('WebSocket连接已关闭', event);
    app.wsConnected = false;
    updateConnectionStatus();
    
    // 如果短时间内未重连成功，则切换到轮询模式
    setTimeout(() => {
        if (!app.wsConnected && app.connectionType !== 'polling') {
            fallbackToPolling();
        }
    }, 5000);
}

/**
 * 处理WebSocket错误事件
 */
function handleWebSocketError(error) {
    console.error('WebSocket错误:', error);
    app.wsConnected = false;
    updateConnectionStatus();
}

/**
 * 处理WebSocket重连事件
 */
function handleWebSocketReconnect() {
    console.log('WebSocket已重新连接');
    app.wsConnected = true;
    app.connectionType = 'local';
    updateConnectionStatus();
}

/**
 * 处理市场数据更新
 */
function handleMarketUpdate(data) {
    // 添加时间对象
    const timestamp = data.timestamp;
    const time = new Date(timestamp);
    
    // 添加到市场数据数组
    app.marketData.push({
        t: time,
        y: data.price,
        open: data.open || data.price,
        high: data.high || data.price,
        low: data.low || data.price,
        close: data.close || data.price,
        volume: data.volume || 0
    });
    
    // 限制数据点数量
    if (app.marketData.length > CONFIG.MAX_MARKET_DATA_POINTS) {
        app.marketData = app.marketData.slice(-CONFIG.MAX_MARKET_DATA_POINTS);
    }
    
    // 更新图表
    updatePriceChart();
    
    // 更新当前价格显示
    if (app.elements.currentPrice) {
        app.elements.currentPrice.textContent = formatPrice(data.price);
    }
}

/**
 * 处理仓位数据更新
 */
function handlePositionUpdate(data) {
    app.positionData = data;
    updatePositionDisplay();
}

/**
 * 处理预测数据更新
 */
function handlePredictionUpdate(data) {
    // 添加时间对象
    const timestamp = data.timestamp;
    const time = new Date(timestamp);
    
    // 获取预测值
    const action = data.action || 'HOLD';
    const confidence = data.confidence || 0.5;
    const values = data.values || { buy: 0.33, sell: 0.33, hold: 0.34 };
    
    // 添加到预测数据数组
    app.predictionData.push({
        time: time,
        action: action,
        confidence: confidence,
        values: values
    });
    
    // 限制数据点数量
    if (app.predictionData.length > CONFIG.MAX_PREDICTION_POINTS) {
        app.predictionData = app.predictionData.slice(-CONFIG.MAX_PREDICTION_POINTS);
    }
    
    // 更新预测图表
    updatePredictionChart();
}

/**
 * 处理订单数据更新
 */
function handleOrderUpdate(data) {
    // 检查是否已存在相同订单
    const existingOrderIndex = app.orderData.findIndex(order => order.order_id === data.order_id);
    
    if (existingOrderIndex >= 0) {
        // 更新现有订单
        app.orderData[existingOrderIndex] = data;
    } else {
        // 添加新订单并保持顺序
        app.orderData.unshift(data);
        
        // 限制最大订单数
        if (app.orderData.length > CONFIG.MAX_ORDERS) {
            app.orderData.pop();
        }
    }
    
    // 更新订单表显示
    updateOrderTable();
}

/**
 * 处理警报更新
 */
function handleAlertUpdate(data) {
    // 添加新警报
    app.alertData.unshift(data);
    
    // 限制最大警报数
    if (app.alertData.length > CONFIG.MAX_ALERTS) {
        app.alertData.pop();
    }
    
    // 更新警报表显示
    updateAlertTable();
    
    // 如果是重要警报，显示通知
    if (data.level === 'error' || data.level === 'warning') {
        showNotification(data.message, data.level);
    }
}

/**
 * 处理系统状态更新
 */
function handleStatusUpdate(data) {
    app.systemStatus = data;
    updateSystemStatusDisplay();
}

/**
 * 更新系统状态，与handleStatusUpdate类似，但可从其他地方直接调用
 * @param {object} data 系统状态数据
 */
function updateSystemStatus(data) {
    if (!data) return;
    app.systemStatus = data;
    updateSystemStatusDisplay();
}

/**
 * 处理心跳消息
 */
function handleHeartbeat(data) {
    app.lastUpdateTime = data.timestamp || Date.now();
    app.wsConnected = true;
    updateConnectionStatus();
}

// =====================================
// UI更新函数
// =====================================

/**
 * 更新连接状态指示器
 */
function updateConnectionStatus() {
    // 更新WebSocket状态指示器
    if (app.elements.wsStatus) {
        if (app.wsConnected) {
            app.elements.wsStatus.classList.remove('status-offline');
            app.elements.wsStatus.classList.add('status-online');
        } else {
            app.elements.wsStatus.classList.remove('status-online');
            app.elements.wsStatus.classList.add('status-offline');
        }
    }
    
    // 更新API状态指示器
    if (app.elements.apiStatus) {
        if (app.apiConnected) {
            app.elements.apiStatus.classList.remove('status-offline');
            app.elements.apiStatus.classList.add('status-online');
        } else {
            app.elements.apiStatus.classList.remove('status-online');
            app.elements.apiStatus.classList.add('status-offline');
        }
    }
    
    // 更新连接类型文本
    if (app.elements.connectionText) {
        let text = 'WebSocket';
        if (app.connectionType === 'polling') {
            text = '轮询模式';
        } else if (app.connectionType === 'mock') {
            text = '模拟数据';
        }
        app.elements.connectionText.textContent = text;
    }
}

/**
 * 更新价格图表
 */
function updatePriceChart() {
    if (!app.priceChart || app.marketData.length === 0) return;
    
    // 更新数据
    app.priceChart.data.datasets[0].data = app.marketData.map(point => ({
        x: point.t,
        y: point.y
    }));
    
    // 刷新图表
    app.priceChart.update('quiet');
}

/**
 * 更新预测图表
 */
function updatePredictionChart() {
    if (!app.predictionChart || app.predictionData.length === 0) return;
    
    // 分离不同动作的数据点
    const buyPoints = [];
    const sellPoints = [];
    const holdPoints = [];
    
    app.predictionData.forEach(point => {
        const action = point.action.toUpperCase();
        const data = {
            x: point.time,
            y: point.confidence
        };
        
        if (action === 'BUY') {
            buyPoints.push(data);
        } else if (action === 'SELL') {
            sellPoints.push(data);
        } else {
            holdPoints.push(data);
        }
    });
    
    // 更新数据集
    app.predictionChart.data.datasets[0].data = buyPoints;
    app.predictionChart.data.datasets[1].data = sellPoints;
    app.predictionChart.data.datasets[2].data = holdPoints;
    
    // 刷新图表
    app.predictionChart.update('quiet');
}

/**
 * 更新仓位显示
 */
function updatePositionDisplay() {
    if (!app.positionData) return;
    
    const position = app.positionData;
    
    // 更新仓位信息
    if (app.elements.positionSize) {
        app.elements.positionSize.textContent = position.size ? position.size.toFixed(3) : '0.000';
    }
    
    if (app.elements.positionSide) {
        let sideText = '无持仓';
        let sideClass = 'text-secondary';
        
        if (position.side === 'LONG') {
            sideText = '多头';
            sideClass = 'text-success';
        } else if (position.side === 'SHORT') {
            sideText = '空头';
            sideClass = 'text-danger';
        }
        
        app.elements.positionSide.textContent = sideText;
        app.elements.positionSide.className = sideClass;
    }
    
    if (app.elements.entryPrice) {
        app.elements.entryPrice.textContent = formatPrice(position.entry_price);
    }
    
    if (app.elements.currentPrice) {
        app.elements.currentPrice.textContent = formatPrice(position.current_price);
    }
    
    if (app.elements.unrealizedPnl) {
        const pnl = position.unrealized_pnl || 0;
        app.elements.unrealizedPnl.textContent = formatCurrency(pnl);
        app.elements.unrealizedPnl.className = pnl >= 0 ? 'text-success' : 'text-danger';
    }
    
    if (app.elements.roe) {
        const roe = position.roe || 0;
        app.elements.roe.textContent = formatPercent(roe);
        app.elements.roe.className = roe >= 0 ? 'text-success' : 'text-danger';
    }
    
    if (app.elements.leverage) {
        app.elements.leverage.textContent = `${position.leverage || 1}x`;
    }
    
    // 更新仓位表格，如果存在的话
    if (app.elements.positionTableBody) {
        let html = '';
        
        if (position.size && position.side !== 'NONE') {
            html = `
                <tr>
                    <td>${position.side === 'LONG' ? '多头' : '空头'}</td>
                    <td>${position.size.toFixed(3)}</td>
                    <td>${formatPrice(position.entry_price)}</td>
                    <td>${formatPrice(position.current_price)}</td>
                    <td class="${position.unrealized_pnl >= 0 ? 'text-success' : 'text-danger'}">${formatCurrency(position.unrealized_pnl)}</td>
                    <td class="${position.roe >= 0 ? 'text-success' : 'text-danger'}">${formatPercent(position.roe)}</td>
                    <td>${formatPrice(position.liquidation_price)}</td>
                </tr>
            `;
        } else {
            html = '<tr><td colspan="7" class="text-center">当前无持仓</td></tr>';
        }
        
        app.elements.positionTableBody.innerHTML = html;
    }
}

/**
 * 更新订单表格
 */
function updateOrderTable() {
    if (!app.elements.orderTableBody) return;
    
    let html = '';
    
    if (app.orderData.length > 0) {
        app.orderData.forEach(order => {
            const orderTime = new Date(order.timestamp).toLocaleString();
            const orderSide = order.side === 'BUY' ? '买入' : '卖出';
            const sideClass = order.side === 'BUY' ? 'text-success' : 'text-danger';
            
            html += `
                <tr>
                    <td>${orderTime}</td>
                    <td class="${sideClass}">${orderSide}</td>
                    <td>${order.size}</td>
                    <td>${formatPrice(order.price)}</td>
                    <td>${getOrderStatusText(order.status)}</td>
                </tr>
            `;
        });
    } else {
        html = '<tr><td colspan="5" class="text-center">无最近订单</td></tr>';
    }
    
    app.elements.orderTableBody.innerHTML = html;
}

/**
 * 更新警报表格
 */
function updateAlertTable() {
    if (!app.elements.alertTableBody) return;
    
    let html = '';
    
    if (app.alertData.length > 0) {
        app.alertData.forEach(alert => {
            const alertTime = new Date(alert.timestamp).toLocaleString();
            const levelClass = getLevelClass(alert.level);
            
            html += `
                <tr>
                    <td>${alertTime}</td>
                    <td><span class="badge ${levelClass}">${alert.level.toUpperCase()}</span></td>
                    <td>${alert.message}</td>
                </tr>
            `;
        });
    } else {
        html = '<tr><td colspan="3" class="text-center">无系统警报</td></tr>';
    }
    
    app.elements.alertTableBody.innerHTML = html;
}

/**
 * 更新系统状态显示
 */
function updateSystemStatusDisplay() {
    if (!app.systemStatus) return;
    
    // 更新状态标签
    if (app.elements.statusBadge) {
        let statusText = '已停止';
        let statusClass = 'bg-secondary';
        
        if (app.systemStatus.is_running) {
            if (app.systemStatus.is_paused) {
                statusText = '已暂停';
                statusClass = 'bg-warning';
            } else {
                statusText = '运行中';
                statusClass = 'bg-success';
            }
        }
        
        app.elements.statusBadge.textContent = statusText;
        app.elements.statusBadge.className = `badge rounded-pill ${statusClass}`;
    }
    
    // 更新账户余额
    const accountInfo = app.systemStatus.account_info || {};
    
    if (app.elements.availableBalance) {
        app.elements.availableBalance.textContent = formatCurrency(accountInfo.available_balance || 0);
    }
    
    if (app.elements.marginBalance) {
        app.elements.marginBalance.textContent = formatCurrency(accountInfo.margin_balance || 0);
    }
    
    if (app.elements.todayProfit) {
        const dailyPnl = accountInfo.daily_pnl || 0;
        app.elements.todayProfit.textContent = formatCurrency(dailyPnl);
        app.elements.todayProfit.className = dailyPnl >= 0 ? 'text-success' : 'text-danger';
    }
    
    if (app.elements.totalProfit) {
        const totalPnl = accountInfo.total_pnl || 0;
        app.elements.totalProfit.textContent = formatCurrency(totalPnl);
        app.elements.totalProfit.className = totalPnl >= 0 ? 'text-success' : 'text-danger';
    }
    
    // 更新交易次数
    if (app.elements.tradingVolume) {
        app.elements.tradingVolume.textContent = app.systemStatus.trade_count || 0;
    }
    
    // 更新最近交易
    if (app.elements.lastTrade && app.systemStatus.last_trade_time) {
        const lastTradeTime = new Date(app.systemStatus.last_trade_time).toLocaleString();
        app.elements.lastTrade.textContent = lastTradeTime;
    } else if (app.elements.lastTrade) {
        app.elements.lastTrade.textContent = '-';
    }
    
    // 更新运行时间
    updateRunningTime();
}

/**
 * 更新运行时间
 */
function updateRunningTime() {
    if (!app.elements.runningTime || !app.systemStatus.start_time) return;
    
    const startTime = app.systemStatus.start_time;
    const now = Date.now();
    const diff = now - startTime;
    
    // 如果系统未运行，显示停止状态
    if (!app.systemStatus.is_running) {
        app.elements.runningTime.textContent = "已停止";
        return;
    }
    
    // 计算时间差
    const seconds = Math.floor(diff / 1000) % 60;
    const minutes = Math.floor(diff / (1000 * 60)) % 60;
    const hours = Math.floor(diff / (1000 * 60 * 60)) % 24;
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    
    // 格式化运行时间
    let timeText = '';
    
    if (days > 0) {
        timeText += `${days}天 `;
    }
    
    timeText += `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    app.elements.runningTime.textContent = timeText;
}

// =====================================
// 工具函数
// =====================================

/**
 * 发送命令到服务器
 * @param {string} action 命令动作
 * @param {object} params 命令参数
 */
function sendCommand(action, params = {}) {
    // 构造命令数据
    const commandData = {
        command: action,
        params: params
    };
    
    // 显示加载指示器
    showLoadingIndicator(true);
    
    // 发送命令
    fetch(`${CONFIG.API_BASE_URL}/api/command`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(commandData)
    })
    .then(response => response.json())
    .then(data => {
        // 隐藏加载指示器
        showLoadingIndicator(false);
        
        // 检查响应
        if (data.error) {
            showError(`执行命令错误: ${data.error}`);
            console.error('命令执行错误:', data.error);
        } else {
            showSuccess(data.message || '命令执行成功');
            
            // 立即更新状态
            setTimeout(() => {
                checkAPIConnection();
            }, 500);
        }
    })
    .catch(error => {
        showLoadingIndicator(false);
        showError('执行命令失败，请检查服务器连接');
        console.error('发送命令出错:', error);
    });
}

/**
 * 设置时间框架
 * @param {string} timeframe 时间框架
 */
function setTimeframe(timeframe) {
    // 更新选中状态
    app.elements.timeframeButtons.forEach(button => {
        if (button.dataset.timeframe === timeframe) {
            button.classList.add('active');
        } else {
            button.classList.remove('active');
        }
    });
    
    // 更新图表配置
    if (app.priceChart) {
        const options = app.priceChart.options;
        let unit = 'minute';
        
        switch (timeframe) {
            case '1m':
                unit = 'minute';
                break;
            case '5m':
                unit = 'minute';
                break;
            case '15m':
                unit = 'minute';
                break;
            case '1h':
                unit = 'hour';
                break;
            case '4h':
                unit = 'hour';
                break;
            case '1d':
                unit = 'day';
                break;
        }
        
        options.scales.x.time.unit = unit;
        app.priceChart.update();
    }
    
    // TODO: 向服务器请求更新时间框架
    // sendCommand('change_timeframe', { timeframe });
}

/**
 * 回退到轮询模式
 */
function fallbackToPolling() {
    console.log('回退到轮询模式获取数据');
    app.connectionType = 'polling';
    startPolling();
    updateConnectionStatus();
}

/**
 * 启动轮询
 */
function startPolling() {
    // 清理现有的轮询定时器
    if (app.updateTimers.polling) {
        clearInterval(app.updateTimers.polling);
    }
    
    // 设置轮询间隔
    const pollingInterval = CONFIG.UPDATE_INTERVAL;
    
    // 启动轮询
    app.updateTimers.polling = setInterval(() => {
        pollServerData();
    }, pollingInterval);
    
    // 立即执行一次轮询
    pollServerData();
}

/**
 * 轮询服务器数据
 */
function pollServerData() {
    // 请求市场数据
    fetch(`${CONFIG.API_BASE_URL}/api/market_data`)
        .then(response => response.json())
        .then(data => {
            if (Array.isArray(data)) {
                data.forEach(item => {
                    handleMarketUpdate(item);
                });
            }
        })
        .catch(error => console.error('获取市场数据失败:', error));
    
    // 请求持仓数据
    fetch(`${CONFIG.API_BASE_URL}/api/position_data`)
        .then(response => response.json())
        .then(data => {
            if (data) {
                handlePositionUpdate(data);
            }
        })
        .catch(error => console.error('获取持仓数据失败:', error));
    
    // 请求预测数据
    fetch(`${CONFIG.API_BASE_URL}/api/predictions`)
        .then(response => response.json())
        .then(data => {
            if (Array.isArray(data)) {
                data.forEach(item => {
                    handlePredictionUpdate(item);
                });
            }
        })
        .catch(error => console.error('获取预测数据失败:', error));
    
    // 请求订单数据
    fetch(`${CONFIG.API_BASE_URL}/api/orders`)
        .then(response => response.json())
        .then(data => {
            if (Array.isArray(data)) {
                app.orderData = data;
                updateOrderTable();
            }
        })
        .catch(error => console.error('获取订单数据失败:', error));
    
    // 请求警报数据
    fetch(`${CONFIG.API_BASE_URL}/api/alerts`)
        .then(response => response.json())
        .then(data => {
            if (Array.isArray(data)) {
                app.alertData = data;
                updateAlertTable();
            }
        })
        .catch(error => console.error('获取警报数据失败:', error));
    
    // 请求系统状态
    fetch(`${CONFIG.API_BASE_URL}/api/status`)
        .then(response => response.json())
        .then(data => {
            if (data) {
                handleStatusUpdate(data);
            }
        })
        .catch(error => console.error('获取系统状态失败:', error));
}

/**
 * 显示加载指示器
 * @param {boolean} show 是否显示
 */
function showLoadingIndicator(show) {
    // TODO: 实现加载指示器
    // const loadingElement = document.getElementById('loading-indicator');
    // if (loadingElement) {
    //     loadingElement.style.display = show ? 'block' : 'none';
    // }
}

/**
 * 显示通知
 * @param {string} message 消息内容
 * @param {string} type 消息类型
 */
function showNotification(message, type = 'info') {
    // Bootstrap 5 Toast
    const toastContainer = document.getElementById('toast-container');
    
    if (!toastContainer) {
        console.error('未找到通知容器元素');
        return;
    }
    
    const toastId = 'toast-' + Date.now();
    const bgClass = getToastBgClass(type);
    const toastHTML = `
        <div id="${toastId}" class="toast align-items-center ${bgClass} border-0 text-white" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        </div>
    `;
    
    toastContainer.insertAdjacentHTML('beforeend', toastHTML);
    const toastElement = document.getElementById(toastId);
    
    // 初始化Toast
    const toast = new bootstrap.Toast(toastElement, {
        autohide: true,
        delay: 5000
    });
    
    // 显示Toast
    toast.show();
    
    // Toast隐藏后移除元素
    toastElement.addEventListener('hidden.bs.toast', () => {
        toastElement.remove();
    });
}

/**
 * 显示错误消息
 * @param {string} message 错误消息
 */
function showError(message) {
    showNotification(message, 'error');
}

/**
 * 显示成功消息
 * @param {string} message 成功消息
 */
function showSuccess(message) {
    showNotification(message, 'success');
}

/**
 * 格式化价格
 * @param {number} price 价格
 * @returns {string} 格式化后的价格
 */
function formatPrice(price) {
    if (price === undefined || price === null) return '0.00';
    return parseFloat(price).toLocaleString('en-US', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    });
}

/**
 * 格式化货币
 * @param {number} amount 金额
 * @returns {string} 格式化后的货币
 */
function formatCurrency(amount) {
    if (amount === undefined || amount === null) return '$0.00';
    
    const prefix = amount >= 0 ? '' : '-';
    return prefix + '$' + Math.abs(amount).toLocaleString('en-US', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    });
}

/**
 * 格式化百分比
 * @param {number} value 百分比值
 * @returns {string} 格式化后的百分比
 */
function formatPercent(value) {
    if (value === undefined || value === null) return '0.00%';
    
    return (value * 100).toLocaleString('en-US', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }) + '%';
}

/**
 * 获取订单状态文本
 * @param {string} status 订单状态
 * @returns {string} 状态文本
 */
function getOrderStatusText(status) {
    switch (status) {
        case 'NEW': return '新建';
        case 'FILLED': return '已成交';
        case 'PARTIALLY_FILLED': return '部分成交';
        case 'CANCELED': return '已取消';
        case 'REJECTED': return '已拒绝';
        case 'EXPIRED': return '已过期';
        default: return status;
    }
}

/**
 * 获取级别对应的样式类
 * @param {string} level 级别
 * @returns {string} 样式类
 */
function getLevelClass(level) {
    switch (level.toLowerCase()) {
        case 'info': return 'bg-info';
        case 'success': return 'bg-success';
        case 'warning': return 'bg-warning';
        case 'error': return 'bg-danger';
        default: return 'bg-secondary';
    }
}

/**
 * 获取Toast背景样式类
 * @param {string} type 消息类型
 * @returns {string} 样式类
 */
function getToastBgClass(type) {
    switch (type.toLowerCase()) {
        case 'info': return 'bg-primary';
        case 'success': return 'bg-success';
        case 'warning': return 'bg-warning';
        case 'error': return 'bg-danger';
        default: return 'bg-secondary';
    }
}

// =====================================
// 应用程序启动
// =====================================

// 当DOM加载完成后初始化应用
document.addEventListener('DOMContentLoaded', initializeApp);
