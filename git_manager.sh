#!/bin/bash
# Git仓库管理脚本 - 支持提交(push)和获取(pull)操作

# 显示彩色输出
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
BLUE="\033[0;34m"
CYAN="\033[0;36m"
NC="\033[0m" # No Color

# 显示欢迎标题
clear
echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}    比特币交易强化学习项目 - Git管理脚本   ${NC}"
echo -e "${BLUE}==========================================${NC}"
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
if [ -z "$(git remote)" ]; then
    echo -e "${YELLOW}设置远程仓库...${NC}"
    git remote add origin https://github.com/losesky/crypto-trading-rl.git
    echo -e "${GREEN}远程仓库已设置${NC}"
elif [ "$(git remote get-url origin 2>/dev/null)" != "https://github.com/losesky/crypto-trading-rl.git" ]; then
    echo -e "${YELLOW}更新远程仓库URL...${NC}"
    git remote set-url origin https://github.com/losesky/crypto-trading-rl.git
    echo -e "${GREEN}远程仓库URL已更新${NC}"
else
    echo -e "${GREEN}远程仓库已正确设置${NC}"
fi

# 显示菜单
show_menu() {
    echo
    echo -e "${CYAN}请选择您要执行的操作:${NC}"
    echo -e "${CYAN}1. 提交代码到GitHub (Push)${NC}"
    echo -e "${CYAN}2. 从GitHub获取代码 (Pull)${NC}"
    echo -e "${CYAN}3. 查看状态和历史${NC}"
    echo -e "${CYAN}0. 退出${NC}"
    echo -e "${YELLOW}请输入选项 [0-3]: ${NC}"
    read -n 1 option
    echo
    return $option
}

# 提交代码到GitHub
push_to_github() {
    echo -e "${BLUE}===== 提交代码到GitHub =====${NC}"
    
    # 显示当前状态
    echo -e "${YELLOW}当前Git状态:${NC}"
    git status --short
    echo
    
    # 添加所有文件到Git
    echo -e "${YELLOW}添加文件到Git...${NC}"
    git add .
    echo -e "${GREEN}文件已添加${NC}"
    
    # 提交更改
    echo -e "${YELLOW}提交更改...${NC}"
    default_commit_message="更新比特币交易强化学习项目 - $(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "${YELLOW}默认提交信息: $default_commit_message${NC}"
    echo -e "${YELLOW}是否自定义提交信息? (y/N)${NC}"
    read -n 1 custom_msg
    echo
    
    commit_message="$default_commit_message"
    if [ "$custom_msg" = "y" ] || [ "$custom_msg" = "Y" ]; then
        echo -e "${YELLOW}请输入自定义提交信息:${NC}"
        read custom_commit_message
        commit_message="$custom_commit_message"
        echo -e "${GREEN}使用自定义提交信息${NC}"
    fi
    
    git commit -m "$commit_message"
    commit_result=$?
    
    if [ $commit_result -ne 0 ]; then
        echo -e "${YELLOW}没有新的更改需要提交或提交失败${NC}"
        return 1
    fi
    
    echo -e "${GREEN}更改已提交${NC}"
    
    # 推送到远程仓库
    echo -e "${YELLOW}推送到远程仓库...${NC}"
    echo -e "${YELLOW}注意: 如果提示输入用户名和密码，请使用GitHub个人访问令牌作为密码${NC}"
    echo -e "${YELLOW}准备推送...按任意键继续或Ctrl+C取消${NC}"
    read -n 1
    echo
    
    # 获取当前分支名
    current_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
    if [ -z "$current_branch" ]; then
        current_branch="main" # 默认为main分支
    fi
    
    # 尝试推送
    echo -e "${YELLOW}推送到 $current_branch 分支...${NC}"
    if git push -u origin $current_branch; then
        echo -e "${GREEN}成功推送到远程仓库${NC}"
        return 0
    else
        echo -e "${YELLOW}推送当前分支失败，尝试创建并推送main分支...${NC}"
        git checkout -b main 2>/dev/null || git checkout main 2>/dev/null
        if git push -u origin main; then
            echo -e "${GREEN}成功推送到remote/main分支${NC}"
            return 0
        else
            echo -e "${RED}推送失败。请检查您的GitHub凭据和网络连接。${NC}"
            echo -e "${RED}您可能需要创建并使用GitHub个人访问令牌作为密码。${NC}"
            echo -e "${BLUE}请访问: https://github.com/settings/tokens 创建个人访问令牌${NC}"
            return 1
        fi
    fi
}

# 从GitHub获取代码
pull_from_github() {
    echo -e "${BLUE}===== 从GitHub获取代码 =====${NC}"
    
    # 检查是否有未提交的更改
    if [ -n "$(git status --porcelain)" ]; then
        echo -e "${YELLOW}您有未提交的更改。获取前建议先处理这些更改。${NC}"
        echo -e "${YELLOW}选项:${NC}"
        echo -e "${YELLOW}1. 存储更改(stash)后获取${NC}"
        echo -e "${YELLOW}2. 丢弃更改并获取${NC}"
        echo -e "${YELLOW}3. 尝试合并(可能会有冲突)${NC}"
        echo -e "${YELLOW}0. 取消获取${NC}"
        echo -e "${YELLOW}请选择 [0-3]: ${NC}"
        read -n 1 stash_option
        echo
        
        case $stash_option in
            1)
                echo -e "${YELLOW}存储更改...${NC}"
                git stash
                echo -e "${GREEN}更改已存储${NC}"
                ;;
            2)
                echo -e "${YELLOW}丢弃更改...${NC}"
                git reset --hard
                echo -e "${GREEN}更改已丢弃${NC}"
                ;;
            3)
                echo -e "${YELLOW}继续获取并尝试合并...${NC}"
                ;;
            *)
                echo -e "${YELLOW}获取操作已取消${NC}"
                return 1
                ;;
        esac
    fi
    
    # 获取远程分支信息
    echo -e "${YELLOW}获取远程分支信息...${NC}"
    git fetch
    
    # 获取当前分支
    current_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
    if [ -z "$current_branch" ]; then
        current_branch="main" # 默认为main分支
    fi
    
    # 尝试拉取
    echo -e "${YELLOW}从远程仓库拉取 $current_branch 分支...${NC}"
    if git pull origin $current_branch; then
        echo -e "${GREEN}成功从远程仓库获取代码${NC}"
        
        # 如果之前进行了stash，尝试应用它
        if [ "$stash_option" = "1" ]; then
            echo -e "${YELLOW}应用之前存储的更改...${NC}"
            if git stash apply; then
                echo -e "${GREEN}存储的更改已成功应用${NC}"
            else
                echo -e "${RED}应用存储的更改时出现冲突。请手动解决冲突。${NC}"
                echo -e "${YELLOW}您可以使用 'git stash show' 查看存储的更改${NC}"
                echo -e "${YELLOW}使用 'git stash drop' 删除存储的更改${NC}"
            fi
        fi
        
        return 0
    else
        echo -e "${RED}获取失败。请检查您的网络连接和凭据。${NC}"
        return 1
    fi
}

# 查看状态和历史
view_status_history() {
    echo -e "${BLUE}===== Git状态和历史 =====${NC}"
    
    # 显示状态
    echo -e "${YELLOW}Git状态:${NC}"
    git status
    echo
    
    # 显示最近的提交
    echo -e "${YELLOW}最近提交历史(最近5条):${NC}"
    git log -5 --oneline --graph
    echo
    
    # 显示分支信息
    echo -e "${YELLOW}分支信息:${NC}"
    git branch -vv
    echo
    
    echo -e "${YELLOW}按任意键返回主菜单...${NC}"
    read -n 1
    return 0
}

# 主循环
while true; do
    show_menu
    option=$?
    
    case $option in
        1)
            push_to_github
            ;;
        2)
            pull_from_github
            ;;
        3)
            view_status_history
            ;;
        0)
            echo -e "${GREEN}感谢使用Git管理脚本，再见!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}无效选项，请重新选择${NC}"
            ;;
    esac
    
    echo
    echo -e "${YELLOW}按任意键继续...${NC}"
    read -n 1
    clear
done
