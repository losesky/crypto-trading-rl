#!/bin/bash
# 脚本用于同步模型指标数据，确保不同评估系统使用相同的数据源

# 定义颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # 无颜色

echo -e "${BLUE}🔄 开始同步模型指标数据${NC}"

# 激活虚拟环境
source venv/bin/activate

# 参数处理
FORCE=0
if [ "$1" == "--force" ]; then
    FORCE=1
    echo -e "${YELLOW}⚠️ 强制模式：将重新评估所有模型${NC}"
fi

# 运行指标同步工具
if [ $FORCE -eq 1 ]; then
    python btc_rl/src/metrics_sync.py --force
else
    python btc_rl/src/metrics_sync.py
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ 同步失败，请检查错误信息${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 指标同步完成${NC}"

# 创建配置文件以确保两种模式使用相同的数据源
CONFIG_DIR="btc_rl/config"
mkdir -p $CONFIG_DIR

cat > $CONFIG_DIR/metrics_config.json << EOF
{
    "use_synchronized_metrics": true,
    "prefer_metrics_file": true,
    "metrics_summary_file": "btc_rl/metrics/models_summary.json"
}
EOF

echo -e "${GREEN}✅ 配置已更新${NC}"
echo -e "${BLUE}ℹ️ 现在可以运行 compare_models.sh 或 analyze_metrics.sh 查看一致的模型指标${NC}"
