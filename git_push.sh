#!/bin/bash
# 自动提交项目到GitHub仓库的脚本

# 显示彩色输出
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
BLUE="\033[0;34m"
NC="\033[0m" # No Color

echo -e "${BLUE}===== 比特币交易强化学习项目 - GitHub提交脚本 =====${NC}"
echo

# 检查git是否已安装
if ! command -v git &> /dev/null; then
    echo -e "${RED}错误: 未找到git命令。请先安装git。${NC}"
    exit 1
fi

# 切换到项目根目录
cd "$(dirname "$0")"
echo -e "${YELLOW}当前工作目录: $(pwd)${NC}"

# 检查是否已经是git仓库
if [ ! -d ".git" ]; then
    echo -e "${YELLOW}初始化Git仓库...${NC}"
    git init
    echo -e "${GREEN}Git仓库初始化完成${NC}"
else
    echo -e "${GREEN}Git仓库已存在${NC}"
fi

# 创建或更新.gitignore文件
if [ ! -f ".gitignore" ]; then
    echo -e "${YELLOW}创建.gitignore文件...${NC}"
    cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# 虚拟环境
venv/
ENV/

# 日志和模型
btc_rl/logs/tb/
btc_rl/logs/episodes/*.json
btc_rl/models/*.zip

# 临时文件
.DS_Store
.vscode/
*.swp
*.swo
.idea/
*.log
EOF
    echo -e "${GREEN}.gitignore文件已创建${NC}"
else
    echo -e "${GREEN}.gitignore文件已存在${NC}"
fi

# 配置Git用户信息（如果未配置）
if [ -z "$(git config --get user.name)" ]; then
    echo -e "${YELLOW}请输入您的Git用户名:${NC}"
    read git_username
    git config user.name "$git_username"
fi

if [ -z "$(git config --get user.email)" ]; then
    echo -e "${YELLOW}请输入您的Git邮箱:${NC}"
    read git_email
    git config user.email "$git_email"
fi

# 设置远程仓库
echo -e "${YELLOW}设置远程仓库...${NC}"
git remote remove origin 2>/dev/null || true
git remote add origin https://github.com/losesky/crypto-trading-rl.git
echo -e "${GREEN}远程仓库已设置${NC}"

# 添加所有文件到Git
echo -e "${YELLOW}添加文件到Git...${NC}"
git add .
echo -e "${GREEN}文件已添加${NC}"

# 提交更改
echo -e "${YELLOW}提交更改...${NC}"
commit_message="更新比特币交易强化学习项目 - $(date '+%Y-%m-%d %H:%M:%S')"
echo -e "${YELLOW}提交信息: $commit_message${NC}"
echo -e "${YELLOW}是否自定义提交信息? (y/N)${NC}"
read -n 1 custom_msg
echo

if [ "$custom_msg" = "y" ] || [ "$custom_msg" = "Y" ]; then
    echo -e "${YELLOW}请输入自定义提交信息:${NC}"
    read custom_commit_message
    commit_message="$custom_commit_message"
    echo -e "${GREEN}使用自定义提交信息${NC}"
fi

git commit -m "$commit_message"
echo -e "${GREEN}更改已提交${NC}"

# 推送到远程仓库
echo -e "${YELLOW}推送到远程仓库...${NC}"
echo -e "${YELLOW}注意: 如果提示输入用户名和密码，请使用GitHub个人访问令牌作为密码${NC}"
echo -e "${YELLOW}准备推送...按任意键继续或Ctrl+C取消${NC}"
read -n 1

# 尝试推送
if git push -u origin master 2>/dev/null || git push -u origin main 2>/dev/null; then
    echo -e "${GREEN}成功推送到远程仓库${NC}"
else
    echo -e "${YELLOW}尝试创建main分支并推送...${NC}"
    git checkout -b main
    git push -u origin main
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}成功推送到remote/main分支${NC}"
    else
        echo -e "${RED}推送失败。请检查您的GitHub凭据和网络连接。${NC}"
        echo -e "${RED}您可能需要创建并使用GitHub个人访问令牌作为密码。${NC}"
        echo -e "${BLUE}请访问: https://github.com/settings/tokens 创建个人访问令牌${NC}"
        exit 1
    fi
fi

echo
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}✅ 项目已成功提交到GitHub仓库${NC}"
echo -e "${GREEN}   https://github.com/losesky/crypto-trading-rl${NC}"
echo -e "${GREEN}=========================================${NC}"
