/**
 * RL交易系统 - UI组件库
 * 版本: 1.0
 * 日期: 2025-06-12
 */

/**
 * 图表组件 - 处理价格和预测图表的创建和更新
 */
class ChartComponent {
    /**
     * 初始化价格图表
     * @param {string} canvasId - 图表画布ID
     * @param {Object} options - 图表配置选项
     */
    static initPriceChart(canvasId, options = {}) {
        const ctx = document.getElementById(canvasId).getContext('2d');
        
        // 默认配置
        const defaultOptions = {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: '价格',
                        data: [],
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHitRadius: 10,
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'hour',
                            tooltipFormat: 'yyyy-MM-dd HH:mm',
                            displayFormats: {
                                hour: 'HH:mm',
                                day: 'MM-dd'
                            }
                        },
                        title: {
                            display: true,
                            text: '时间'
                        },
                        grid: {
                            display: false
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: '价格 (USDT)'
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                animation: {
                    duration: 0
                },
                hover: {
                    mode: 'nearest',
                    intersect: true
                }
            }
        };
        
        // 合并用户选项
        const chartOptions = deepMerge(defaultOptions, options);
        
        // 创建图表
        return new Chart(ctx, chartOptions);
    }
    
    /**
     * 初始化预测图表
     * @param {string} canvasId - 图表画布ID
     */
    static initPredictionChart(canvasId) {
        const ctx = document.getElementById(canvasId).getContext('2d');
        
        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: '预测值',
                        data: [],
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 2,
                        pointRadius: 2,
                        tension: 0.4
                    },
                    {
                        label: '置信度',
                        data: [],
                        backgroundColor: 'rgba(153, 102, 255, 0.2)',
                        borderColor: 'rgba(153, 102, 255, 1)',
                        borderWidth: 2,
                        pointRadius: 0,
                        borderDash: [5, 5],
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'minute',
                            tooltipFormat: 'HH:mm:ss',
                            displayFormats: {
                                minute: 'HH:mm'
                            }
                        },
                        title: {
                            display: true,
                            text: '时间'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: '预测值'
                        },
                        min: -1.2,
                        max: 1.2,
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    annotation: {
                        annotations: {
                            line1: {
                                type: 'line',
                                yMin: 0,
                                yMax: 0,
                                borderColor: 'rgba(0, 0, 0, 0.2)',
                                borderWidth: 1,
                                borderDash: [5, 5]
                            }
                        }
                    }
                }
            }
        });
    }

    /**
     * 更新价格图表数据
     * @param {Object} chart - Chart.js实例
     * @param {Array} data - 新的价格数据
     * @param {number} maxPoints - 最大数据点数量
     */
    static updatePriceChart(chart, data, maxPoints = 100) {
        // 确保数据是数组
        if (!Array.isArray(data)) return;
        
        // 添加新数据
        data.forEach(point => {
            if (point && point.timestamp && point.close) {
                const timestamp = new Date(point.timestamp * 1000);
                chart.data.labels.push(timestamp);
                chart.data.datasets[0].data.push(point.close);
                
                // 限制数据点数量
                if (chart.data.labels.length > maxPoints) {
                    chart.data.labels.shift();
                    chart.data.datasets[0].data.shift();
                }
            }
        });
        
        // 更新图表
        chart.update();
    }

    /**
     * 更新预测图表数据
     * @param {Object} chart - Chart.js实例
     * @param {Object} prediction - 预测数据
     * @param {number} maxPoints - 最大数据点数量
     */
    static updatePredictionChart(chart, prediction, maxPoints = 50) {
        if (!prediction || !prediction.timestamp) return;
        
        const timestamp = new Date(prediction.timestamp * 1000);
        const actionValue = prediction.action_value || 0;
        const confidence = prediction.confidence || 0;
        
        // 添加新数据
        chart.data.labels.push(timestamp);
        chart.data.datasets[0].data.push(actionValue);
        chart.data.datasets[1].data.push(confidence);
        
        // 限制数据点数量
        if (chart.data.labels.length > maxPoints) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
            chart.data.datasets[1].data.shift();
        }
        
        // 更新图表
        chart.update();
    }
}

/**
 * UI状态管理器 - 处理UI元素的显示和隐藏
 */
class UIStateManager {
    /**
     * 更新系统状态UI
     * @param {Object} status - 系统状态数据
     */
    static updateSystemStatus(status) {
        // 运行状态标记
        const statusBadge = document.getElementById('status-badge');
        if (statusBadge) {
            if (status.is_running) {
                if (status.is_trading_paused) {
                    statusBadge.className = 'badge bg-warning';
                    statusBadge.textContent = '已暂停';
                } else {
                    statusBadge.className = 'badge bg-success';
                    statusBadge.textContent = '运行中';
                }
            } else {
                statusBadge.className = 'badge bg-danger';
                statusBadge.textContent = '已停止';
            }
        }
        
        // 运行时间
        const uptimeElement = document.getElementById('uptime');
        if (uptimeElement && status.uptime) {
            uptimeElement.textContent = formatDuration(status.uptime);
        }
        
        // 交易计数
        const tradeCountElement = document.getElementById('trade-count');
        if (tradeCountElement) {
            tradeCountElement.textContent = status.trade_count || 0;
        }
        
        // 最近交易时间
        const lastTradeTimeElement = document.getElementById('last-trade-time');
        if (lastTradeTimeElement && status.last_action_time) {
            const lastActionTime = new Date(status.last_action_time);
            lastTradeTimeElement.textContent = formatDateTime(lastActionTime);
        }
        
        // 模式标签
        const modeBadge = document.getElementById('mode-badge');
        if (modeBadge) {
            if (status.mode === 'prod') {
                modeBadge.innerHTML = '<span class="badge bg-danger">生产模式</span>';
            } else {
                modeBadge.innerHTML = '<span class="badge bg-warning">测试模式</span>';
            }
        }
    }
    
    /**
     * 更新账户信息UI
     * @param {Object} accountInfo - 账户信息数据
     */
    static updateAccountInfo(accountInfo) {
        if (!accountInfo) return;
        
        // 可用余额
        const availableBalanceElement = document.getElementById('available-balance');
        if (availableBalanceElement && accountInfo.availableBalance) {
            availableBalanceElement.textContent = formatCurrency(accountInfo.availableBalance) + ' USDT';
        }
        
        // 保证金余额
        const marginBalanceElement = document.getElementById('margin-balance');
        if (marginBalanceElement && accountInfo.totalMarginBalance) {
            marginBalanceElement.textContent = formatCurrency(accountInfo.totalMarginBalance) + ' USDT';
        }
        
        // 总盈亏 (如果可用)
        const totalPnlElement = document.getElementById('total-pnl');
        if (totalPnlElement && accountInfo.totalUnrealizedProfit) {
            const pnl = parseFloat(accountInfo.totalUnrealizedProfit);
            totalPnlElement.textContent = formatCurrency(pnl) + ' USDT';
            totalPnlElement.className = pnl >= 0 ? 'profit' : 'loss';
        }
    }
    
    /**
     * 更新仓位信息UI
     * @param {Object} position - 仓位数据
     * @param {Object} marketData - 市场数据
     */
    static updatePositionInfo(position, marketData) {
        const positionInfoElement = document.getElementById('position-info');
        const noPositionMsgElement = document.getElementById('no-position-msg');
        
        // 检查是否有持仓
        if (!position || position.size <= 0) {
            // 显示无持仓消息，隐藏持仓信息
            positionInfoElement.classList.add('d-none');
            noPositionMsgElement.classList.remove('d-none');
            return;
        }
        
        // 显示持仓信息，隐藏无持仓消息
        positionInfoElement.classList.remove('d-none');
        noPositionMsgElement.classList.add('d-none');
        
        // 更新持仓方向
        const positionSideElement = document.getElementById('position-side');
        if (positionSideElement) {
            if (position.side === 'BUY') {
                positionSideElement.textContent = '多';
                positionSideElement.className = 'badge bg-success';
            } else {
                positionSideElement.textContent = '空';
                positionSideElement.className = 'badge bg-danger';
            }
        }
        
        // 更新持仓大小
        const positionSizeElement = document.getElementById('position-size');
        if (positionSizeElement) {
            positionSizeElement.textContent = position.size.toFixed(6);
        }
        
        // 更新杠杆
        const positionLeverageElement = document.getElementById('position-leverage');
        if (positionLeverageElement) {
            positionLeverageElement.textContent = `${position.leverage}x`;
        }
        
        // 更新入场价
        const entryPriceElement = document.getElementById('entry-price');
        if (entryPriceElement) {
            entryPriceElement.textContent = formatCurrency(position.entry_price);
        }
        
        // 更新当前价格
        const currentPriceElement = document.getElementById('current-price');
        if (currentPriceElement && marketData && marketData.close) {
            currentPriceElement.textContent = formatCurrency(marketData.close);
        }
        
        // 更新强平价
        const liquidationPriceElement = document.getElementById('liquidation-price');
        if (liquidationPriceElement) {
            liquidationPriceElement.textContent = formatCurrency(position.liquidation_price || 0);
        }
        
        // 更新未实现盈亏
        const positionPnlElement = document.getElementById('position-pnl');
        if (positionPnlElement) {
            const pnl = position.unrealized_pnl || 0;
            positionPnlElement.textContent = formatCurrency(pnl) + ' USDT';
            positionPnlElement.className = pnl >= 0 ? 'h4 profit' : 'h4 loss';
        }
        
        // 更新ROE
        const positionRoeElement = document.getElementById('position-roe');
        if (positionRoeElement && position.margin > 0) {
            const pnl = position.unrealized_pnl || 0;
            const roe = (pnl / position.margin) * 100;
            positionRoeElement.textContent = roe.toFixed(2) + '%';
            positionRoeElement.className = roe >= 0 ? 'h4 profit' : 'h4 loss';
        }
    }
    
    /**
     * 更新模型预测UI
     * @param {Object} prediction - 预测数据
     */
    static updatePrediction(prediction) {
        if (!prediction) return;
        
        const predictionActionElement = document.getElementById('prediction-action');
        const predictionConfidenceElement = document.getElementById('prediction-confidence');
        
        if (predictionActionElement) {
            let actionText, actionClass;
            
            switch (prediction.action_type) {
                case 'BUY':
                    actionText = '买入';
                    actionClass = 'prediction-buy';
                    break;
                case 'SELL':
                    actionText = '卖出';
                    actionClass = 'prediction-sell';
                    break;
                default:
                    actionText = '持有';
                    actionClass = 'prediction-hold';
            }
            
            predictionActionElement.textContent = actionText;
            predictionActionElement.className = 'h2 ' + actionClass;
        }
        
        if (predictionConfidenceElement) {
            const confidencePercent = Math.round((prediction.confidence || 0) * 100);
            predictionConfidenceElement.style.width = `${confidencePercent}%`;
            predictionConfidenceElement.textContent = `${confidencePercent}%`;
            
            // 根据置信度设置颜色
            if (confidencePercent > 80) {
                predictionConfidenceElement.className = 'progress-bar bg-success';
            } else if (confidencePercent > 50) {
                predictionConfidenceElement.className = 'progress-bar bg-info';
            } else if (confidencePercent > 30) {
                predictionConfidenceElement.className = 'progress-bar bg-warning';
            } else {
                predictionConfidenceElement.className = 'progress-bar bg-danger';
            }
        }
    }
    
    /**
     * 更新订单列表UI
     * @param {Array} orders - 订单数据数组
     * @param {number} maxItems - 最大显示数量
     */
    static updateOrdersList(orders, maxItems = 5) {
        const ordersListElement = document.getElementById('orders-list');
        if (!ordersListElement || !orders || !orders.length) return;
        
        // 清空列表
        ordersListElement.innerHTML = '';
        
        // 添加最近的订单
        const recentOrders = orders.slice(Math.max(0, orders.length - maxItems)).reverse();
        
        recentOrders.forEach(order => {
            const orderTime = new Date(order.time || order.timestamp * 1000);
            const orderSide = order.side || '';
            const orderStatus = order.status || '';
            const orderPrice = order.price || 0;
            const orderQty = order.origQty || order.quantity || 0;
            
            const orderItemElement = document.createElement('div');
            orderItemElement.className = `list-group-item order-item ${orderSide.toLowerCase() === 'buy' ? 'order-buy' : 'order-sell'}`;
            
            orderItemElement.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <span class="badge ${orderSide.toLowerCase() === 'buy' ? 'bg-success' : 'bg-danger'} me-2">${orderSide}</span>
                        <small>${formatCurrency(orderPrice)} × ${orderQty.toFixed(6)}</small>
                    </div>
                    <small class="text-muted">${formatTime(orderTime)}</small>
                </div>
                <div class="text-muted small">${orderStatus}</div>
            `;
            
            ordersListElement.appendChild(orderItemElement);
        });
    }
    
    /**
     * 更新警报列表UI
     * @param {Array} alerts - 警报数据数组
     * @param {number} maxItems - 最大显示数量
     */
    static updateAlertsList(alerts, maxItems = 5) {
        const alertsListElement = document.getElementById('alerts-list');
        if (!alertsListElement || !alerts || !alerts.length) return;
        
        // 清空列表
        alertsListElement.innerHTML = '';
        
        // 添加最近的警报
        const recentAlerts = alerts.slice(Math.max(0, alerts.length - maxItems)).reverse();
        
        recentAlerts.forEach(alert => {
            const alertTime = new Date(alert.timestamp * 1000);
            const alertType = alert.type || '';
            const alertMessage = alert.message || '';
            const alertSeverity = alert.severity || 'info';
            
            const alertItemElement = document.createElement('div');
            alertItemElement.className = `list-group-item alert-item alert-${alertSeverity}`;
            
            let iconClass = 'bi-info-circle';
            let badgeClass = 'bg-info';
            
            switch (alertSeverity) {
                case 'critical':
                    iconClass = 'bi-exclamation-triangle';
                    badgeClass = 'bg-danger';
                    break;
                case 'warning':
                    iconClass = 'bi-exclamation-circle';
                    badgeClass = 'bg-warning';
                    break;
                case 'info':
                default:
                    iconClass = 'bi-info-circle';
                    badgeClass = 'bg-info';
            }
            
            alertItemElement.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <i class="bi ${iconClass} me-2"></i>
                        <span class="badge ${badgeClass} me-2">${alertType}</span>
                    </div>
                    <small class="text-muted">${formatTime(alertTime)}</small>
                </div>
                <div class="mt-1">${alertMessage}</div>
            `;
            
            alertsListElement.appendChild(alertItemElement);
        });
    }
    
    /**
     * 更新连接状态UI
     * @param {boolean} connected - 连接状态
     */
    static updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connectivity-status');
        if (!statusElement) return;
        
        if (connected) {
            statusElement.innerHTML = '<i class="bi bi-circle-fill text-success"></i> 已连接';
        } else {
            statusElement.innerHTML = '<i class="bi bi-circle-fill text-danger"></i> 已断开';
        }
    }
}

/**
 * 控制器 - 处理用户交互和API调用
 */
class UIController {
    /**
     * 初始化控制器
     */
    static initialize() {
        // 绑定按钮事件
        this._bindButtonEvents();
    }
    
    /**
     * 绑定按钮事件
     * @private
     */
    static _bindButtonEvents() {
        // 启动交易按钮
        const startButton = document.getElementById('btn-start');
        if (startButton) {
            startButton.addEventListener('click', () => {
                this._sendCommand('start_trading');
            });
        }
        
        // 停止交易按钮
        const stopButton = document.getElementById('btn-stop');
        if (stopButton) {
            stopButton.addEventListener('click', () => {
                if (confirm('确定要停止交易系统吗？')) {
                    this._sendCommand('stop_trading');
                }
            });
        }
        
        // 暂停交易按钮
        const pauseButton = document.getElementById('btn-pause');
        if (pauseButton) {
            pauseButton.addEventListener('click', () => {
                this._sendCommand('pause_trading');
            });
        }
        
        // 恢复交易按钮
        const resumeButton = document.getElementById('btn-resume');
        if (resumeButton) {
            resumeButton.addEventListener('click', () => {
                this._sendCommand('resume_trading');
            });
        }
        
        // 平仓按钮
        const closePositionButton = document.getElementById('btn-close-position');
        if (closePositionButton) {
            closePositionButton.addEventListener('click', () => {
                if (confirm('确定要平仓当前所有持仓吗？')) {
                    this._sendCommand('close_position');
                }
            });
        }
        
        // 时间框架选择
        const timeframeButtons = document.querySelectorAll('[data-timeframe]');
        timeframeButtons.forEach(button => {
            button.addEventListener('click', () => {
                // 移除其他按钮的active类
                timeframeButtons.forEach(btn => btn.classList.remove('active'));
                // 添加当前按钮的active类
                button.classList.add('active');
                
                // 通知时间框架变更
                if (typeof window.onTimeframeChange === 'function') {
                    window.onTimeframeChange(button.dataset.timeframe);
                }
            });
        });
    }
    
    /**
     * 发送命令到API
     * @param {string} command - 命令名称
     * @param {Object} params - 命令参数
     * @private
     */
    static _sendCommand(command, params = {}) {
        fetch('/api/command', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                command: command,
                params: params
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('命令执行失败:', data.error);
                alert(`执行失败: ${data.error}`);
            } else {
                console.log('命令执行成功:', data);
                // 可以添加成功提示
            }
        })
        .catch(error => {
            console.error('API请求失败:', error);
            alert('通信错误，请检查网络连接');
        });
    }
}

/**
 * 实用工具函数
 */

/**
 * 格式化数字为货币格式
 * @param {number} value - 要格式化的数值
 * @param {number} decimals - 小数位数
 * @returns {string} 格式化后的字符串
 */
function formatCurrency(value, decimals = 2) {
    if (typeof value !== 'number') {
        value = parseFloat(value) || 0;
    }
    return value.toFixed(decimals).replace(/\d(?=(\d{3})+\.)/g, '$&,');
}

/**
 * 格式化日期时间
 * @param {Date} date - 日期对象
 * @returns {string} 格式化后的字符串
 */
function formatDateTime(date) {
    if (!(date instanceof Date)) return '-';
    
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    const seconds = String(date.getSeconds()).padStart(2, '0');
    
    return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
}

/**
 * 格式化时间（仅显示时:分:秒）
 * @param {Date} date - 日期对象
 * @returns {string} 格式化后的字符串
 */
function formatTime(date) {
    if (!(date instanceof Date)) return '-';
    
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    const seconds = String(date.getSeconds()).padStart(2, '0');
    
    return `${hours}:${minutes}:${seconds}`;
}

/**
 * 格式化持续时间（秒数转为时:分:秒）
 * @param {number} seconds - 秒数
 * @returns {string} 格式化后的字符串
 */
function formatDuration(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    return [
        String(hours).padStart(2, '0'),
        String(minutes).padStart(2, '0'),
        String(secs).padStart(2, '0')
    ].join(':');
}

/**
 * 深度合并两个对象
 * @param {Object} target - 目标对象
 * @param {Object} source - 源对象
 * @returns {Object} 合并后的对象
 */
function deepMerge(target, source) {
    const isObject = obj => obj && typeof obj === 'object';
    
    if (!isObject(target) || !isObject(source)) {
        return source;
    }
    
    Object.keys(source).forEach(key => {
        const targetValue = target[key];
        const sourceValue = source[key];
        
        if (Array.isArray(targetValue) && Array.isArray(sourceValue)) {
            target[key] = targetValue.concat(sourceValue);
        } else if (isObject(targetValue) && isObject(sourceValue)) {
            target[key] = deepMerge(targetValue, sourceValue);
        } else {
            target[key] = sourceValue;
        }
    });
    
    return target;
}