# 模型指标一致性解决方案

## 问题说明

我们的BTC交易强化学习项目中有两个不同的评估系统：

1. `compare_models.sh` - 基于WebSocket的实时模型比较服务
2. `analyze_metrics.sh` - 基于预计算指标文件的静态分析

这两个系统在展示模型性能指标时出现了不一致的情况，特别是在交易次数和胜率方面。

## 原因分析

1. **数据源不同**：WebSocket服务使用实时评估生成数据，而分析工具使用预计算的指标文件
2. **计算方法不同**：两个系统使用不同方法来检测和计算交易
3. **数据采样频率不同**：WebSocket服务可能使用简化数据点提高响应速度
4. **无统一真相源**：缺乏一个共同认可的标准指标数据源

## 解决方案

我们已经实现了以下改进：

1. 创建了模型指标同步工具：`btc_rl/src/metrics_sync.py`
2. 修改了模型比较服务(`model_comparison.py`)，确保优先使用预计算指标
3. 更新了指标分析工具(`show_model_metrics.py`)，生成统一的汇总文件
4. 创建了同步脚本(`sync_metrics.sh`)，确保两个系统使用相同的数据源

## 使用方法

### 1. 同步模型指标

在进行模型比较前，先运行同步脚本确保指标一致性：

```bash
./sync_metrics.sh
```

如果需要重新评估所有模型指标：

```bash
./sync_metrics.sh --force
```

这将创建一个统一的`models_summary.json`文件，作为所有模型指标的真相源。

### 2. 查看模型指标

使用分析工具查看详细指标：

```bash
./analyze_metrics.sh --full
```

可视化比较多个模型：

```bash
./analyze_metrics.sh --plot
```

### 3. 实时比较模型

启动模型比较服务：

```bash
./compare_models.sh
```

在浏览器中访问 <http://localhost:8080/model_comparison.html>

## 技术实现

1. 统一了模型评估方法，确保`train_sac.py`中的`evaluate_model_with_metrics`函数成为标准评估方法
2. 创建了统一的指标汇总文件`models_summary.json`，提供一致的数据源
3. 修改了`model_comparison.py`，优先使用指标文件中的交易数据，同时保持实时权益曲线评估
4. 修改了`show_model_metrics.py`，确保生成完整的汇总数据文件

## 注意事项

1. 在修改模型或训练新模型后，应运行`sync_metrics.sh`更新指标数据
2. 如遇到数据不一致，请使用`--force`参数强制重新评估所有模型
3. 指标文件位于`btc_rl/metrics/`目录下，可直接查看或修改

## 文件位置

- 统一指标汇总文件：`btc_rl/metrics/models_summary.json`
- 单个模型指标文件：`btc_rl/metrics/<model_name>_metrics.json`
- 指标同步配置：`btc_rl/config/metrics_config.json`
