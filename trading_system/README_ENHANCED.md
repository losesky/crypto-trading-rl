# RL交易系统增强版使用指南

本文档介绍了RL交易系统的最新增强功能和使用方法，重点关注数据流和界面优化。

## 系统架构

增强版交易系统使用了以下核心组件：

1. **WebSocket代理服务器** - 处理实时数据推送
2. **数据发送器** - 格式化并发送交易数据
3. **高级前端界面** - 提供实时数据可视化
4. **系统诊断工具** - 辅助排查和修复问题

## 启动方式

系统提供了统一的启动方式，简化了操作流程：

### 统一启动入口（推荐）

使用项目根目录下的统一启动脚本，可以选择不同的启动模式：

```bash
# 从项目根目录执行
./start_trading.sh

# 或者在任意位置执行
bash /path/to/crypto-trading-rl/start_trading.sh
```

该脚本会显示交互式菜单，您可以选择：

1. 启动完整交易系统（测试环境）
2. 仅启动UI界面
3. 启动生产环境系统
4. 运行健康检查
5. 强制重启（解决端口冲突）

您也可以使用命令行参数直接指定启动模式：

```bash
# 使用测试环境配置启动完整系统
./start_trading.sh --test

# 或仅启动UI界面
./start_trading.sh --ui-only

# 强制重启所有服务
./start_trading.sh --force
```

### 健康检查与修复

如果系统运行不正常，您可以使用健康检查功能：

```bash
# 使用统一脚本运行健康检查
./start_trading.sh --health

# 或通过交互菜单选择健康检查选项
./start_trading.sh  # 然后选择选项 4
```

### 高级诊断

对于复杂的数据流问题，可以使用健康检查和强制重启选项：

```bash
# 运行健康检查，解决常见问题
./start_trading.sh --health

# 强制重启所有服务，解决端口冲突等问题
./start_trading.sh --force
```

如果需要更细致的诊断，可以直接运行诊断脚本：

```bash
# 运行全面诊断
python trading_system/scripts/fix_data_issues.py --full-check
```

## 系统组件说明

### 1. WebSocket代理

WebSocket代理(`websocket_proxy.py`)是系统的核心组件，它负责：

- 管理与前端的WebSocket连接
- 转发实时市场和交易数据
- 处理连接中断和重连
- 提供健康检查API

该代理使用单例模式确保系统中只有一个实例运行。

### 2. 数据发送器

数据发送器(`data_sender.py`)负责处理来自交易系统的数据，主要功能包括：

- 格式化市场数据、持仓信息和模型预测
- 添加适当的时间戳和类型标记
- 通过WebSocket代理发送数据到前端
- 实现心跳机制确保连接稳定

### 3. 前端系统

新版前端(`app-new.js`和`index-new.html`)提供了更好的用户体验：

- 实现可靠的WebSocket连接管理（自动重连）
- 优化的数据处理和图表渲染
- 响应式设计，适应不同屏幕尺寸
- 改进的错误处理和通知系统

### 4. 数据流程

数据在系统中的流动路径如下：

```bash
交易服务(trading_service.py)
        ↓
数据发送器(data_sender.py)
        ↓
WebSocket代理(websocket_proxy.py)
        ↓
前端应用(app.js / app-new.js)
        ↓
用户界面(index.html / index-new.html)
```

## 常见问题与解决方案

### 前端数据不更新

如果前端不显示最新数据：

1. 检查WebSocket连接状态（前端界面上的连接指示灯）
2. 运行健康检查：`./start_trading.sh --health`
3. 尝试强制重启系统：`./start_trading.sh --force`

### WebSocket连接失败

如果WebSocket无法连接：

1. 确认WebSocket代理是否运行：`./start_trading.sh --ws-proxy`
2. 检查端口是否被占用：`netstat -tuln | grep 8095`
3. 使用强制重启选项自动处理冲突：`./start_trading.sh --force`

### 系统性能问题

如果系统运行缓慢：

1. 减少图表中显示的数据点数量（在配置文件中修改`max_data_points`）
2. 增加更新间隔时间（修改`update_interval`）
3. 检查系统资源使用情况：`top` 或 `htop`

## 配置说明

系统配置文件位于`trading_system/config/`目录，主要参数说明：

- `http_port`: UI服务器的端口
- `ws_port`: WebSocket服务器的端口
- `update_interval`: 数据更新间隔（毫秒）
- `max_data_points`: 图表中保留的最大数据点数量

## 开发者注意事项

如需扩展系统功能：

1. WebSocket代理扩展：在`websocket_proxy.py`中添加新的事件处理器
2. 前端功能扩展：修改`app-new.js`中的相应模块
3. 数据格式化：更新`data_sender.py`中的数据转换逻辑

请确保在修改后进行全面测试，特别是WebSocket连接和数据流部分。

## 日志位置

系统日志保存在以下位置：

- 主系统日志：`trading_system/logs/trading_*.log`
- UI服务器日志：通常输出到控制台
- WebSocket代理日志：通常输出到控制台

## 系统要求

- Python 3.8+
- Node.js 14+ (如需开发前端)
- 现代浏览器（支持WebSocket）

---

## 更新日志

- 添加WebSocket代理架构
- 重构前端数据处理逻辑
- 添加自动诊断和修复工具
- 优化UI界面和响应式设计
