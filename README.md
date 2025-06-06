# BTC Trading RL - 基于强化学习的比特币交易智能体

这是一个使用强化学习算法（SAC - Soft Actor-Critic）来训练比特币交易智能体的项目。该项目实现了一个完整的交易环境、策略训练系统以及实时可视化界面。

## 📋 项目概述

该项目旨在通过强化学习训练一个能够自主进行比特币交易的智能体，支持做多做空、杠杆交易和风险管理。系统提供实时的交易可视化界面，能够监控智能体的交易决策和投资组合表现。

## 🚀 系统组件

项目由以下几个关键组件组成：

1. **强化学习训练系统** - 使用SAC算法训练的比特币交易智能体
2. **数据获取和处理** - 支持从多家交易所获取不同时间精度的比特币价格数据
3. **灵活的交易环境** - 可配置的交易参数（杠杆、风险管理等）
4. **WebSocket服务器** - 用于实时数据通信
5. **可视化前端** - 提供交易策略和表现的实时可视化
6. **模型比较系统** - 支持多个交易模型的并行比较和性能分析

## 💻 环境要求

- Python 3.8 或更高版本
- 支持WebSocket的现代浏览器
- 足够的计算资源（建议：4+ CPU核心, 8GB+ RAM）
- （可选）支持CUDA的GPU以加速训练
- 互联网连接（用于从交易所API获取数据）

## 🗂️ 项目结构

```bash
btc_rl/
├── data/                          # 数据目录
│   ├── BTC_hourly.csv            # 比特币小时价格数据
│   ├── BTC_1min.csv              # 比特币1分钟价格数据（可选）
│   ├── BTC_5min.csv              # 比特币5分钟价格数据（可选）
│   ├── test_data.npz             # 测试数据集
│   └── train_data.npz            # 训练数据集
├── logs/                         # 日志目录
│   ├── episodes/                 # 训练回合日志
│   └── tb/                       # TensorBoard日志
├── models/                       # 模型保存目录
├── src/                          # 源代码
│   ├── config.py                 # 配置管理
│   ├── data_fetcher.py           # 加密货币历史数据获取工具
│   ├── data_workflow.py          # 数据获取和处理工作流
│   ├── env.py                    # 交易环境实现
│   ├── http_server.py            # 静态文件服务器
│   ├── policies.py               # 策略网络定义
│   ├── preprocessing.py          # 数据预处理
│   ├── train_sac.py              # SAC算法训练脚本
│   ├── validate_data.py          # 数据质量验证工具
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

#### 方法1：使用现有数据文件

如果您已有数据文件，只需确保将其放在正确位置：

```bash
# 数据格式示例
# timestamp,open,high,low,close,volume
cp your_data_file.csv /home/losesky/crypto-trading-rl/btc_rl/data/BTC_hourly.csv
```

#### 方法2：获取真实交易所数据（推荐）

使用内置的数据获取工具从真实交易所获取BTC历史数据：

```bash
# 从Binance获取2020年至今的BTC/USDT小时级数据
python -m btc_rl.src.data_fetcher \
  --exchange binance \
  --symbol "BTC/USDT" \
  --timeframe 1h \
  --start_date 2020-01-01 \
  --end_date $(date +%Y-%m-%d)
  
# 处理获取的数据并生成训练/测试数据集
python -m btc_rl.src.preprocessing --csv btc_rl/data/BTC_hourly.csv
```

#### 方法3：使用自动化工作流（最简单）

我们提供了一个完整的工作流脚本，可以一键完成数据获取、预处理和模型训练：

```bash
# 运行自动化工作流
./run_workflow.sh --exchange binance --start-date 2020-01-01 --end-date 2023-01-01
```

支持参数:

- `--exchange`: 交易所名称 (支持binance, coinbase, kraken等)
- `--start-date`: 数据起始日期
- `--end-date`: 数据结束日期
- `--timeframe`: 时间周期 (支持1m, 5m, 15m, 30m, 1h, 4h, 1d)
- `--pair`: 交易对 (默认BTC/USDT)
- `--skip-data-fetch`: 跳过数据获取，使用现有数据
- `--force`: 强制更新数据，即使已有最新数据
- `--retries`: API请求失败时的最大重试次数

### 3. 多种时间精度的数据

系统现在支持多种时间精度的数据获取和处理：

```bash
# 获取1分钟级别数据
./run_workflow.sh --timeframe 1m --start-date 2023-01-01 --end-date 2023-06-30

# 获取5分钟级别数据
./run_workflow.sh --timeframe 5m --start-date 2023-01-01 --end-date 2023-06-30

# 获取15分钟级别数据
./run_workflow.sh --timeframe 15m --start-date 2022-01-01 --end-date 2023-01-01
```

系统会自动将不同精度的数据转换为小时级别数据用于训练，同时保留原始精度的数据备用。

### 4. 使用不同市场条件的数据训练模型

为了让交易智能体更贴近真实环境，可以使用不同时间段的数据进行训练和测试：

#### 使用特定市场阶段的数据

您可以针对不同市场条件训练专门的模型：

```bash
# 牛市数据 (如2020年底至2021年初)
./run_workflow.sh --exchange binance --start-date 2020-10-01 --end-date 2021-04-01

# 熊市数据 (如2022年)
./run_workflow.sh --exchange binance --start-date 2022-01-01 --end-date 2022-12-31

# 横盘整理市场 (如特定时期)
./run_workflow.sh --exchange binance --start-date 2019-01-01 --end-date 2019-06-30
```

### 5. 数据验证和质量检查

系统现在包含了数据质量验证工具，可以检查获取的数据是否符合训练要求：

```bash
# 验证小时级数据
python -m btc_rl.src.validate_data --csv btc_rl/data/BTC_hourly.csv --timeframe 1h

# 验证分钟级数据
python -m btc_rl.src.validate_data --csv btc_rl/data/BTC_1min.csv --timeframe 1m

# 使用严格模式验证，生成可视化报告
python -m btc_rl.src.validate_data --csv btc_rl/data/BTC_hourly.csv --timeframe 1h --strict --output-dir btc_rl/logs/data_quality
```

验证工具会检查：

- **数据完整性** - 检查数据点数量是否合理，是否覆盖整个时间范围
- **数据连续性** - 检查时间序列是否有大的缺口或异常间隔
- **数据异常值** - 检测价格、交易量等异常值
- **数据质量评分** - 生成总体数据质量评分和报告

### 6. 自动数据更新

系统提供了自动数据更新脚本，可以定期获取最新数据：

```bash
# 手动运行更新脚本
./update_data.sh

# 设置定时任务，每天凌晨2点更新数据
(crontab -l 2>/dev/null; echo "0 2 * * * /home/losesky/crypto-trading-rl/update_data.sh") | crontab -

# 查看当前定时任务
crontab -l
```

更新脚本会:

1. 检查现有数据的最后日期
2. 只获取从最后日期到当前日期的新数据
3. 合并新数据到现有数据集
4. 重新生成训练和测试数据集
5. 自动备份原始数据文件

### 7. 系统配置

可以通过编辑 `config.ini` 文件来自定义系统的各个方面：

```ini
; 数据配置
[data]
default_exchange = "binance"
default_symbol = "BTC/USDT"
default_timeframe = "1h"

; API请求设置
api_max_retries = 5
api_retry_delay = 10
api_use_exponential_backoff = true

; 交易环境配置
[environment]
initial_balance = 10000.0
max_leverage = 3.0
fee_rate = 0.0002
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

### 3. 模型比较分析系统 (`compare_models.sh` 与 `model_comparison.html`)

- **多模型并行比较**：同时对比多达10个不同模型的表现
- **性能指标分析**：
  - 最终权益和总收益率对比
  - 最大回撤对比
  - 夏普比率和索提诺比率计算
  - 总费用统计
  - 胜率统计
- **可视化对比**：
  - 权益曲线对比图表
  - 回撤对比图表
  - 完整性能指标对比表格
- **简易操作**：一键启动与切换不同模型组合
- **自动预加载**：启动时自动预加载所有模型数据，提供进度条显示
- **容错机制**：WebSocket断线自动重连，数据格式错误处理

### 4. 实时可视化系统 (`visualizer/index.html`)

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
- **Total Fees**: 累计交易手续费（总费用）
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
2. **高级分析工具**：添加更多的高级图表分析工具（如技术指标、相关性分析等）
3. **风险指标**：补充更复杂的风险评估指标（如卡玛比率、omega比率等）
4. **参数调整**：实时调整训练参数的界面
5. **数据导出**：支持交易记录和图表数据的导出功能

## 🎯 项目意义

这个项目展示了强化学习在量化交易领域的应用潜力，通过实时可视化系统，研究人员和开发者可以：

1. **直观观察**智能体的学习过程和交易决策
2. **实时评估**模型性能和风险控制能力
3. **快速迭代**模型设计和参数调优
4. **理解行为**分析智能体的交易策略和决策逻辑

该项目为金融AI和量化交易提供了一个完整的开发和测试平台，具有很强的教育和研究价值。

### 8. 模型指标分析工具

系统提供了专业的模型指标分析工具，用于评估和比较不同模型的性能：

```bash
# 显示基本模型指标摘要
./analyze_metrics.sh

# 显示详细模型信息
./analyze_metrics.sh --full

# 生成并显示图表比较
./analyze_metrics.sh --plot

# 重新评估所有模型指标
./analyze_metrics.sh --evaluate

# 修复模型胜率数据
./analyze_metrics.sh --fix-winrate

# 自定义风控阈值进行筛选
./analyze_metrics.sh --max-dd 0.03 --min-sortino 20 --min-sharpe 10
```

#### 主要功能

- **模型性能评估**：自动评估所有模型的关键指标
- **风控参数筛选**：基于自定义最大回撤、最小索提诺和夏普比率进行模型筛选
- **自动计算指标**：计算并显示夏普比率、索提诺比率、最大回撤等关键指标
- **数据修复功能**：自动修复模型指标文件中的数据问题
- **图表生成**：生成对比图表用于直观分析
- **虚拟环境自动激活**：智能检测并激活Python虚拟环境
- **格式化输出**：清晰的表格化输出，便于快速对比和分析

该工具是模型开发和迭代过程中进行指标评估的重要组件，提供了命令行友好的交互界面。

### 9. 模型比较分析

系统提供了可视化模型比较工具，可以直观地对比不同训练模型的性能：

```bash
# 启动模型比较服务
./compare_models.sh
```

启动后，在浏览器中访问 `http://localhost:8080/model_comparison.html` 即可使用模型比较功能：

#### 主要功能

- **模型选择**：同时选择多个模型进行对比（最多支持10个模型）
- **性能对比**：通过直观图表比较不同模型的权益曲线和回撤情况
- **统计分析**：查看并对比各模型的关键性能指标
  - 最终权益
  - 总收益率
  - 最大回撤
  - 夏普比率
  - 索提诺比率
  - 总费用（手续费）
  - 胜率
- **自动预加载**：启动时显示进度条，预加载所有模型数据，提升用户体验
- **错误处理**：优雅处理连接错误、数据解析错误和服务中断

#### 使用方法

1. 运行`./compare_models.sh`启动服务（自动预加载所有模型数据）
2. 在浏览器中访问`http://localhost:8080/model_comparison.html`
3. 点击模型按钮选择要比较的模型（支持多选）
4. 点击"比较所选模型"按钮加载数据
5. 查看生成的权益曲线对比图和回撤对比图
6. 分析底部表格中的详细性能指标，包括总费用等关键数据
7. 可以随时切换不同模型组合进行新的比较

这个工具对于选择最佳模型、理解不同训练策略的优缺点以及分析交易费用对模型性能的影响非常有帮助。
