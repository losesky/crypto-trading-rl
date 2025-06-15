/**
 * RL交易系统 - 高级WebSocket通信模块
 * 用于处理WebSocket连接、重连和数据处理
 */
class WebSocketManager {
    /**
     * 初始化WebSocket管理器
     * @param {string} url - 建立连接的URL
     * @param {object} options - 配置选项
     */
    constructor(url, options = {}) {
        this.url = url;
        this.options = Object.assign({
            reconnectInterval: 2000,    // 重连间隔（毫秒）
            reconnectAttempts: 10,      // 重连尝试次数
            debug: false,               // 是否调试模式
            autoConnect: true,          // 是否自动连接
            protocols: null,            // WebSocket协议
            heartbeatInterval: 30000,   // 心跳间隔
            heartbeatMessage: JSON.stringify({ping: Date.now()}), // 心跳消息
        }, options);
        
        // 内部状态
        this._isConnected = false;
        this._reconnectCount = 0;
        this._lastHeartbeat = 0;
        this._heartbeatTimer = null;
        this._reconnectTimer = null;
        this.socket = null;
        
        // 回调函数映射
        this.eventHandlers = new Map();
        this.dataHandlers = {};
        this.metaHandlers = {
            open: [],
            close: [],
            error: [],
            reconnect: [],
        };
        
        // 存储最近收到的消息
        this.recentMessages = {};
        
        // 若需要自动连接，则立即初始化连接
        if (this.options.autoConnect) {
            this.connect();
        }
    }
    
    /**
     * 建立WebSocket连接
     * @returns {Promise<WebSocket>} 创建的WebSocket对象
     */
    connect() {
        if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) {
            this.log('WebSocket已连接或正在连接');
            return Promise.resolve(this.socket);
        }
        
        this.log(`尝试连接到 ${this.url}`);
        return new Promise((resolve, reject) => {
            try {
                this.socket = new WebSocket(this.url, this.options.protocols);
                
                // 设置回调
                this.socket.onopen = (event) => this._handleOpen(event, resolve);
                this.socket.onclose = (event) => this._handleClose(event);
                this.socket.onerror = (event) => this._handleError(event, reject);
                this.socket.onmessage = (event) => this._handleMessage(event);
                
            } catch (error) {
                this.log('创建WebSocket错误', error);
                this._scheduleReconnect();
                reject(error);
            }
        });
    }
    
    /**
     * 关闭WebSocket连接
     */
    disconnect() {
        this.log('关闭WebSocket连接');
        this._clearTimers();
        if (this.socket) {
            this.socket.close();
            this._isConnected = false;
        }
    }
    
    /**
     * 添加事件处理器
     * @param {string} eventType - 事件类型 
     * @param {Function} handler - 处理函数
     */
    on(eventType, handler) {
        if (!eventType || typeof handler !== 'function') return;
        
        // 元事件 (open, close, error, reconnect)
        if (this.metaHandlers.hasOwnProperty(eventType)) {
            this.metaHandlers[eventType].push(handler);
            return;
        }
        
        // 数据事件
        if (!this.dataHandlers[eventType]) {
            this.dataHandlers[eventType] = [];
        }
        this.dataHandlers[eventType].push(handler);
    }
    
    /**
     * 移除事件处理器
     * @param {string} eventType - 事件类型
     * @param {Function} handler - 处理函数
     */
    off(eventType, handler) {
        if (!eventType) return;
        
        // 元事件
        if (this.metaHandlers.hasOwnProperty(eventType) && handler) {
            this.metaHandlers[eventType] = this.metaHandlers[eventType].filter(h => h !== handler);
            return;
        }
        
        // 数据事件
        if (this.dataHandlers[eventType]) {
            if (handler) {
                this.dataHandlers[eventType] = this.dataHandlers[eventType].filter(h => h !== handler);
            } else {
                delete this.dataHandlers[eventType];
            }
        }
    }
    
    /**
     * 发送数据
     * @param {object|string} data - 要发送的数据
     * @returns {boolean} 是否发送成功
     */
    send(data) {
        if (!this._isConnected) {
            this.log('WebSocket未连接，无法发送数据');
            return false;
        }
        
        try {
            const message = typeof data === 'string' ? data : JSON.stringify(data);
            this.socket.send(message);
            return true;
        } catch (error) {
            this.log('发送数据错误', error);
            return false;
        }
    }
    
    /**
     * 获取最近的消息
     * @param {string} type - 消息类型
     * @returns {object|null} 最近的消息
     */
    getRecentMessage(type) {
        return this.recentMessages[type] || null;
    }
    
    /**
     * 手动触发心跳
     */
    sendHeartbeat() {
        if (this._isConnected) {
            this.send(this.options.heartbeatMessage);
            this._lastHeartbeat = Date.now();
        }
    }
    
    /**
     * 处理WebSocket打开事件
     * @private
     */
    _handleOpen(event, resolve) {
        this.log('WebSocket已连接');
        this._isConnected = true;
        this._reconnectCount = 0;
        
        // 启动心跳
        this._startHeartbeat();
        
        // 触发所有open事件处理器
        this.metaHandlers.open.forEach(handler => {
            try {
                handler(event);
            } catch (e) {
                this.log('处理open事件错误', e);
            }
        });
        
        resolve(this.socket);
    }
    
    /**
     * 处理WebSocket关闭事件
     * @private
     */
    _handleClose(event) {
        this.log(`WebSocket已关闭: ${event.code} ${event.reason}`);
        this._isConnected = false;
        this._clearTimers();
        
        // 触发所有close事件处理器
        this.metaHandlers.close.forEach(handler => {
            try {
                handler(event);
            } catch (e) {
                this.log('处理close事件错误', e);
            }
        });
        
        this._scheduleReconnect();
    }
    
    /**
     * 处理WebSocket错误事件
     * @private
     */
    _handleError(event, reject) {
        this.log('WebSocket错误', event);
        
        // 触发所有error事件处理器
        this.metaHandlers.error.forEach(handler => {
            try {
                handler(event);
            } catch (e) {
                this.log('处理error事件错误', e);
            }
        });
        
        if (reject) reject(event);
    }
    
    /**
     * 处理WebSocket消息事件
     * @private
     */
    _handleMessage(event) {
        let data = event.data;
        
        // 尝试解析JSON
        try {
            data = JSON.parse(data);
            this.log('收到消息:', data);
        } catch (e) {
            this.log('消息不是有效的JSON格式');
            return;
        }
        
        // 处理心跳响应
        if (data.pong || (data.event === 'pong')) {
            this._lastHeartbeat = Date.now();
            return;
        }
        
        // 处理币安订阅确认消息
        if (data.result === null && data.id) {
            this.log(`订阅确认: ID=${data.id}`);
            return;
        }
        
        // 处理原生WebSocket代理消息格式（event和data字段）
        if (data.event && data.data !== undefined) {
            const eventType = data.event;
            const eventData = data.data;
            
            // 存储最近消息
            this.recentMessages[eventType] = eventData;
            
            // 调试输出
            this.log(`接收到事件: ${eventType}`, eventData);
            
            // 触发该类型的所有处理器
            if (this.dataHandlers[eventType]) {
                this.dataHandlers[eventType].forEach(handler => {
                    try {
                        handler(eventData);
                    } catch (e) {
                        this.log(`处理 ${eventType} 类型消息错误`, e);
                        console.error(`处理 ${eventType} 消息错误:`, e);
                    }
                });
            } else {
                this.log(`没有处理器处理事件类型: ${eventType}`);
            }
            
            return;
        }
        
        // 处理普通数据消息（币安格式）
        const type = data.e || data.type;
        if (type && this.dataHandlers[type]) {
            // 存储最近消息
            this.recentMessages[type] = data;
            
            // 触发该类型的所有处理器
            this.dataHandlers[type].forEach(handler => {
                try {
                    handler(data);
                } catch (e) {
                    this.log(`处理 ${type} 类型消息错误`, e);
                }
            });
        } else {
            // 处理普通对象消息，尝试检查是否包含已注册的事件类型
            for (const eventType in this.dataHandlers) {
                if (eventType in data) {
                    // 存储最近消息
                    this.recentMessages[eventType] = data[eventType];
                    
                    // 触发该类型的所有处理器
                    this.dataHandlers[eventType].forEach(handler => {
                        try {
                            handler(data[eventType]);
                        } catch (e) {
                            this.log(`处理 ${eventType} 类型消息错误`, e);
                        }
                    });
                    return;
                }
            }
            
            // 如果没有找到匹配的处理器，记录未处理的消息
            this.log('收到未处理的消息:', data);
        }
    }
    
    /**
     * 启动心跳机制
     * @private
     */
    _startHeartbeat() {
        this._clearTimers();
        
        if (this.options.heartbeatInterval > 0) {
            this._heartbeatTimer = setInterval(() => {
                this.sendHeartbeat();
            }, this.options.heartbeatInterval);
        }
    }
    
    /**
     * 安排重连
     * @private
     */
    _scheduleReconnect() {
        // 如果已达到最大重连次数则停止
        if (this._reconnectCount >= this.options.reconnectAttempts) {
            this.log('达到最大重连尝试次数，停止重连');
            return;
        }
        
        this._clearTimers();
        const delay = Math.min(1000 * Math.pow(1.5, this._reconnectCount), 30000);
        this._reconnectCount++;
        
        this.log(`安排在 ${delay}ms 后重新连接 (${this._reconnectCount}/${this.options.reconnectAttempts})`);
        
        this._reconnectTimer = setTimeout(() => {
            this.connect().then(() => {
                // 触发所有reconnect事件处理器
                this.metaHandlers.reconnect.forEach(handler => {
                    try {
                        handler();
                    } catch (e) {
                        this.log('处理reconnect事件错误', e);
                    }
                });
            });
        }, delay);
    }
    
    /**
     * 清除所有计时器
     * @private
     */
    _clearTimers() {
        if (this._heartbeatTimer) {
            clearInterval(this._heartbeatTimer);
            this._heartbeatTimer = null;
        }
        
        if (this._reconnectTimer) {
            clearTimeout(this._reconnectTimer);
            this._reconnectTimer = null;
        }
    }
    
    /**
     * 记录日志
     * @private
     */
    log(...args) {
        if (this.options.debug) {
            console.log(`[WebSocketManager]`, ...args);
        }
    }
    
    /**
     * 检查连接是否活跃
     * @returns {boolean} 连接是否活跃
     */
    isActive() {
        // 检查内部状态标志
        if (!this._isConnected) {
            this.log('isActive: 内部状态为未连接');
            return false;
        }
        
        // 检查socket对象是否存在
        if (!this.socket) {
            this.log('isActive: socket对象不存在');
            return false;
        }
        
        // 检查socket的readyState
        if (this.socket.readyState !== WebSocket.OPEN) {
            this.log('isActive: socket状态不是OPEN', this.socket.readyState);
            return false;
        }
        
        // 如果设置了心跳间隔，检查最后心跳时间
        if (this.options.heartbeatInterval > 0 && this._lastHeartbeat > 0) {
            const elapsed = Date.now() - this._lastHeartbeat;
            // 如果上次心跳超过3倍心跳间隔，认为连接已断开
            if (elapsed > this.options.heartbeatInterval * 3) {
                this.log(`心跳超时: ${elapsed}ms 已过去，断开连接`);
                this._isConnected = false;
                return false;
            }
        }
        
        return true;
    }
}

// 导出模块
if (typeof module !== 'undefined' && module.exports) {
    module.exports = WebSocketManager;
}
