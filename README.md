# BTC Trading RL - 基于强化学习的比特币交易智能体

这是一个使用强化学习算法（SAC - Soft Actor-Critic）来训练比特币交易智能体的项目。该项目实现了一个完整的交易环境、策略训练系统以及实时可视化界面。

## 📋 项目概述

该项目旨在通过强化学习训练一个能够自主进行比特币交易的智能体，支持做多做空、杠杆交易和风险管理。系统提供实时的交易可视化界面，能够监控智能体的交易决策和投资组合表现。

## 🚀 系统组件

项目由以下几个关键组件组成：

1. **强化学习训练系统** - 使用SAC算法训练的比特币交易智能体
2. **WebSocket服务器** - 用于实时数据通信
3. **可视化前端** - 提供交易策略和表现的实时可视化

## 💻 环境要求

- Python 3.8 或更高版本
- 支持WebSocket的现代浏览器
- 足够的计算资源（建议：4+ CPU核心, 8GB+ RAM）
- （可选）支持CUDA的GPU以加速训练

## 🗂️ 项目结构

```bash
btc_rl/
├── data/                          # 数据目录
│   ├── BTC_hourly.csv            # 比特币小时价格数据
│   ├── test_data.npz             # 测试数据集
│   └── train_data.npz            # 训练数据集
├── logs/                         # 日志目录
│   ├── episodes/                 # 训练回合日志
│   └── tb/                       # TensorBoard日志
├── models/                       # 模型保存目录
├── src/                          # 源代码
│   ├── env.py                    # 交易环境实现
│   ├── http_server.py            # 静态文件服务器
│   ├── policies.py               # 策略网络定义
│   ├── preprocessing.py          # 数据预处理
│   ├── train_sac.py              # SAC算法训练脚本
│   ├── websocket_client.py       # WebSocket客户端
│   └── websocket_server.py       # WebSocket服务器
└── visualizer/                   # 可视化界面
    └── index.html               # Web实时监控界面
```

## 🛠 部署与运行

### 1. 安装项目

运行部署脚本自动安装所需依赖并准备环境：

```bash
cd /home/losesky/crypto-trading-rl
./deploy.sh
```

### 2. 准备数据

确保比特币价格数据文件已放置在正确位置：

```bash
# 数据格式示例
# timestamp,open,high,low,close,volume
cp your_data_file.csv /home/losesky/crypto-trading-rl/btc_rl/data/BTC_hourly.csv
```

### 3. 运行系统

#### 方法1：一键启动（推荐）

使用 `start.sh` 脚本一键启动所有服务：

```bash
cd /home/losesky/crypto-trading-rl
./start.sh
```

这个脚本会自动启动WebSocket服务器、HTTP服务器、打开浏览器并启动训练进程。

#### 方法2：单独启动各服务

如果需要单独控制各个服务，可按照以下顺序手动启动：

```bash
# 1. 在一个终端中启动WebSocket服务器
cd /home/losesky/crypto-trading-rl
python -m btc_rl.src.websocket_server

# 2. 在另一个终端中启动HTTP服务器
cd /home/losesky/crypto-trading-rl
python -m btc_rl.src.http_server

# 3. 在浏览器中打开: http://localhost:8080/index.html

# 4. 在第三个终端中启动训练进程
cd /home/losesky/crypto-trading-rl
python -m btc_rl.src.train_sac
```

## 🔍 核心功能

### 1. 交易环境 (`env.py`)

- **杠杆交易支持**：支持多空双向交易
- **风险管理**：实现强制平仓机制
- **费用计算**：包含交易手续费
- **状态空间**：价格、技术指标、持仓信息等
- **奖励函数**：基于收益率和风险调整的奖励

### 2. 强化学习训练 (`train_sac.py`)

- **算法**：采用SAC（Soft Actor-Critic）算法
- **经验回放**：使用经验缓冲区提高学习效率
- **策略优化**：连续动作空间的策略梯度优化

### 3. 实时可视化系统 (`visualizer/index.html`)

#### 核心特性

- **WebSocket实时通信**：与训练进程实时数据交换
- **多维度图表展示**：
  - 比特币价格走势图
  - 保证金权益变化图
  - 买入持有策略对比
- **交易标注**：在价格图上标记买卖点和强制平仓点
- **统计面板**：实时显示关键指标

#### 界面布局

```html
<!-- 统计指标面板 -->
<section class="stats-wrapper card">
  <div class="stats-grid">
    <!-- 显示步数、动作、现金、未实现盈亏等 -->
  </div>
</section>

<!-- 图表区域 -->
<section class="chart-card card">
  <canvas id="btcChart"></canvas>        <!-- 价格图表 -->
</section>
<section class="chart-card card">
  <canvas id="marginEquityChart"></canvas> <!-- 权益图表 -->
</section>

<!-- 交易记录表 -->
<section class="trades-card card">
  <table id="tradesTable">
    <!-- 最近交易记录 -->
  </table>
</section>
```

#### 数据更新机制

```javascript
// WebSocket连接处理
const ws = new WebSocket('ws://localhost:8765');
ws.addEventListener('message', e => {
  const d = JSON.parse(e.data);
  updateStats(d);    // 更新统计数据
  updateCharts(d);   // 更新图表
  addTradeRow(d);    // 添加交易记录
});
```

## 📊 可视化功能详解

### 1. 实时统计面板

显示以下关键指标：

- **Step**: 当前训练步数
- **Action**: 当前动作值（正值买入，负值卖出）
- **Cash**: 现金余额
- **uPnL**: 未实现盈亏
- **Margin Equity**: 保证金权益
- **B&H Equity**: 买入持有策略权益
- **Reward**: 当前奖励值
- **Total Fees**: 累计手续费
- **Liquidated**: 是否被强制平仓

### 2. 图表系统

- **价格图表**：显示比特币价格走势，包含交易标注
- **权益图表**：对比智能体权益与买入持有策略
- **交易标注**：
  - 绿色三角形：买入操作
  - 红色倒三角形：卖出操作
  - 黑色方块：强制平仓

### 3. 交易记录表

实时更新最近100笔交易记录，包含：

- 执行步数
- 交易方向（BUY/SELL/LIQ）
- 动作值
- 执行价格
- 持仓数量
- 现金余额
- 保证金权益
- 获得奖励

## 🔧 技术实现

### 前端技术栈

- **Chart.js**: 图表渲染库
- **WebSocket**: 实时数据通信
- **原生JavaScript**: 无框架依赖
- **响应式CSS**: 自适应不同屏幕尺寸

### 图表配置

```javascript
function newLineChart(ctx, label, color) {
  return new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label, 
        borderColor: color, 
        borderWidth: 2, 
        tension: 0.4, 
        pointRadius: 1, 
        data: []
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      // 图表配置...
    }
  });
}
```

### 数据流处理

```javascript
function updateCharts(d) {
  // 添加新数据点
  btcChart.data.labels.push(String(d.step));
  btcChart.data.datasets[0].data.push(d.price);
  meChart.data.datasets[0].data.push(d.margin_equity);
  
  // 添加交易标注
  if (d.was_liquidated_this_step || Math.abs(d.action) > 0.01) {
    // 创建标注点...
  }
  
  // 保持数据量在合理范围（最多50个点）
  if (btcChart.data.labels.length > MAX) {
    // 删除旧数据...
  }
}
```

## ❓ 故障排除

如果遇到以下问题，尝试这些解决方案：

1. **无法连接到WebSocket服务器**
   - 确保WebSocket服务器正在运行
   - 检查防火墙是否允许8765端口

2. **可视化界面未显示数据**
   - 打开浏览器开发者工具查看WebSocket连接状态
   - 确保训练已开始并正在产生数据

3. **训练过程崩溃**
   - 检查GPU内存是否足够
   - 尝试降低批处理大小或网络复杂度

## 📈 项目评估

### 优势

1. **完整的端到端系统**：从数据预处理到模型训练再到实时监控
2. **直观的可视化界面**：实时显示交易决策和绩效指标
3. **专业的交易功能**：支持杠杆、做空、风险管理等专业交易功能
4. **实时监控能力**：通过WebSocket实现训练过程的实时监控
5. **对比分析**：与买入持有策略的直接对比

### 技术亮点

1. **响应式设计**：适配不同设备屏幕
2. **高效渲染**：通过跳帧机制优化图表更新性能
3. **内存管理**：限制数据点数量防止内存溢出
4. **错误处理**：WebSocket连接异常处理
5. **用户体验**：流畅的动画和直观的颜色编码

### 潜在改进空间

1. **历史数据回放**：添加历史训练数据的回放功能
2. **策略比较**：支持多个模型的同时比较
3. **风险指标**：增加更多风险评估指标（夏普比率、最大回撤等）
4. **参数调整**：实时调整训练参数的界面
5. **数据导出**：支持交易记录和图表数据的导出功能

## 🎯 项目意义

这个项目展示了强化学习在量化交易领域的应用潜力，通过实时可视化系统，研究人员和开发者可以：

1. **直观观察**智能体的学习过程和交易决策
2. **实时评估**模型性能和风险控制能力
3. **快速迭代**模型设计和参数调优
4. **理解行为**分析智能体的交易策略和决策逻辑

该项目为金融AI和量化交易提供了一个完整的开发和测试平台，具有很强的教育和研究价值。
