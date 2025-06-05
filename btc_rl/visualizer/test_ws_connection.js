// 测试WebSocket连接的脚本
console.log("开始测试WebSocket连接...");

const ws = new WebSocket('ws://localhost:8765');

ws.addEventListener('open', () => {
    console.log("WebSocket连接已建立");
    
    // 请求模型1的数据
    const message = JSON.stringify({
        type: 'request_model_data',
        model_id: '1'
    });
    
    console.log("发送请求:", message);
    ws.send(message);
});

ws.addEventListener('message', (event) => {
    console.log("收到服务器消息:", event.data);
    try {
        const data = JSON.parse(event.data);
        
        if (data.type === 'multi_model_data') {
            console.log("收到模型数据:", data.data);
            for (const [modelId, dataPoints] of Object.entries(data.data)) {
                console.log(`模型 ${modelId} 数据点数量: ${dataPoints.length}`);
                
                // 检查第一个数据点是否有错误标记
                if (dataPoints.length > 0) {
                    console.log("第一个数据点:", dataPoints[0]);
                    if (dataPoints[0].error) {
                        console.error(`模型 ${modelId} 报告错误: ${dataPoints[0].termination_reason}`);
                    }
                }
            }
        } else if (data.type === 'system_message') {
            console.log("系统消息:", data.content);
        }
    } catch (err) {
        console.error("解析消息失败:", err);
    }
});

ws.addEventListener('error', (error) => {
    console.error("WebSocket发生错误:", error);
});

ws.addEventListener('close', () => {
    console.log("WebSocket连接已关闭");
});
