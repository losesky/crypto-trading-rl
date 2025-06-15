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
    echo -e "${CYAN}4. 分支管理${NC}"
    echo -e "${CYAN}0. 退出${NC}"
    echo -e "${YELLOW}请输入选项 [0-4]: ${NC}"
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

# 分支管理功能
branch_management() {
    while true; do
        clear
        echo -e "${BLUE}===== 分支管理 =====${NC}"
        
        # 显示当前分支
        current_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
        echo -e "${GREEN}当前分支: ${YELLOW}$current_branch${NC}"
        echo
        
        # 显示本地分支列表
        echo -e "${YELLOW}本地分支:${NC}"
        git branch -v
        echo
        
        # 显示远程分支列表
        echo -e "${YELLOW}远程分支:${NC}"
        git branch -r
        echo
        
        # 分支管理菜单
        echo -e "${CYAN}请选择操作:${NC}"
        echo -e "${CYAN}1. 创建新分支${NC}"
        echo -e "${CYAN}2. 切换分支${NC}"
        echo -e "${CYAN}3. 合并分支${NC}"
        echo -e "${CYAN}4. 删除分支${NC}"
        echo -e "${CYAN}5. 重命名分支${NC}"
        echo -e "${CYAN}6. 基于远程分支创建本地分支${NC}"
        echo -e "${CYAN}0. 返回主菜单${NC}"
        echo -e "${YELLOW}请输入选项 [0-6]: ${NC}"
        read -n 1 branch_option
        echo
        
        case $branch_option in
            1)  # 创建新分支
                echo -e "${YELLOW}请输入新分支名称:${NC}"
                read new_branch_name
                if [ -z "$new_branch_name" ]; then
                    echo -e "${RED}分支名不能为空${NC}"
                else
                    echo -e "${YELLOW}是否切换到新分支? (Y/n)${NC}"
                    read -n 1 switch_option
                    echo
                    if [ "$switch_option" = "n" ] || [ "$switch_option" = "N" ]; then
                        # 创建但不切换
                        if git branch $new_branch_name; then
                            echo -e "${GREEN}成功创建分支: $new_branch_name${NC}"
                        else
                            echo -e "${RED}创建分支失败，可能已存在同名分支${NC}"
                        fi
                    else
                        # 创建并切换
                        if git checkout -b $new_branch_name; then
                            echo -e "${GREEN}成功创建并切换到分支: $new_branch_name${NC}"
                        else
                            echo -e "${RED}创建或切换分支失败${NC}"
                        fi
                    fi
                fi
                ;;
                
            2)  # 切换分支
                echo -e "${YELLOW}请输入要切换到的分支名称:${NC}"
                read target_branch
                if [ -z "$target_branch" ]; then
                    echo -e "${RED}分支名不能为空${NC}"
                else
                    # 检查是否有未提交的更改
                    if [ -n "$(git status --porcelain)" ]; then
                        echo -e "${YELLOW}您有未提交的更改，切换分支前需要处理这些更改${NC}"
                        echo -e "${YELLOW}1. 提交更改后切换${NC}"
                        echo -e "${YELLOW}2. 存储更改(stash)后切换${NC}"
                        echo -e "${YELLOW}3. 强制切换(可能丢失更改)${NC}"
                        echo -e "${YELLOW}0. 取消切换${NC}"
                        echo -e "${YELLOW}请选择 [0-3]: ${NC}"
                        read -n 1 switch_handling
                        echo
                        
                        case $switch_handling in
                            1)  # 提交更改
                                echo -e "${YELLOW}输入提交信息:${NC}"
                                read commit_msg
                                commit_msg="${commit_msg:-自动提交更改 - 切换分支前}"
                                git add .
                                git commit -m "$commit_msg"
                                ;;
                            2)  # 存储更改
                                git stash save "切换到分支 $target_branch 前的自动存储"
                                echo -e "${GREEN}更改已存储${NC}"
                                ;;
                            3)  # 强制切换
                                git reset --hard
                                echo -e "${GREEN}未提交的更改已丢弃${NC}"
                                ;;
                            *)  # 取消
                                echo -e "${YELLOW}切换分支已取消${NC}"
                                break
                                ;;
                        esac
                    fi
                    
                    # 执行分支切换
                    if git checkout $target_branch; then
                        echo -e "${GREEN}成功切换到分支: $target_branch${NC}"
                        
                        # 如果之前进行了stash，询问是否要应用
                        if [ "$switch_handling" = "2" ]; then
                            echo -e "${YELLOW}是否应用之前存储的更改? (y/N)${NC}"
                            read -n 1 apply_stash
                            echo
                            if [ "$apply_stash" = "y" ] || [ "$apply_stash" = "Y" ]; then
                                if git stash apply; then
                                    echo -e "${GREEN}存储的更改已成功应用${NC}"
                                    echo -e "${YELLOW}是否删除存储记录? (y/N)${NC}"
                                    read -n 1 drop_stash
                                    echo
                                    if [ "$drop_stash" = "y" ] || [ "$drop_stash" = "Y" ]; then
                                        git stash drop
                                        echo -e "${GREEN}存储记录已删除${NC}"
                                    fi
                                else
                                    echo -e "${RED}应用存储的更改时出现冲突，请手动解决${NC}"
                                fi
                            fi
                        fi
                    else
                        echo -e "${RED}切换分支失败，请检查分支名是否正确${NC}"
                    fi
                fi
                ;;
                
            3)  # 合并分支
                echo -e "${YELLOW}请输入要合并到当前分支的源分支名称:${NC}"
                read source_branch
                if [ -z "$source_branch" ]; then
                    echo -e "${RED}分支名不能为空${NC}"
                else
                    echo -e "${YELLOW}正在将 $source_branch 合并到 $current_branch...${NC}"
                    if git merge $source_branch; then
                        echo -e "${GREEN}合并成功${NC}"
                    else
                        echo -e "${RED}合并时出现冲突，请手动解决冲突${NC}"
                        echo -e "${YELLOW}解决冲突后，使用以下命令:${NC}"
                        echo -e "${YELLOW}  git add .${NC}"
                        echo -e "${YELLOW}  git commit -m \"解决合并冲突\"${NC}"
                        echo -e "${RED}或者中止合并:${NC}"
                        echo -e "${YELLOW}  git merge --abort${NC}"
                    fi
                fi
                ;;
                
            4)  # 删除分支
                echo -e "${YELLOW}请输入要删除的分支名称:${NC}"
                read branch_to_delete
                if [ -z "$branch_to_delete" ]; then
                    echo -e "${RED}分支名不能为空${NC}"
                elif [ "$branch_to_delete" = "$current_branch" ]; then
                    echo -e "${RED}无法删除当前分支，请先切换到其他分支${NC}"
                else
                    echo -e "${YELLOW}是否删除远程分支? (y/N)${NC}"
                    read -n 1 delete_remote
                    echo
                    
                    # 删除本地分支
                    echo -e "${YELLOW}删除本地分支 $branch_to_delete...${NC}"
                    if git branch -d $branch_to_delete; then
                        echo -e "${GREEN}成功删除本地分支${NC}"
                    else
                        echo -e "${RED}删除本地分支失败，该分支可能包含未合并的更改${NC}"
                        echo -e "${YELLOW}是否强制删除? (y/N)${NC}"
                        read -n 1 force_delete
                        echo
                        if [ "$force_delete" = "y" ] || [ "$force_delete" = "Y" ]; then
                            if git branch -D $branch_to_delete; then
                                echo -e "${GREEN}成功强制删除本地分支${NC}"
                            else
                                echo -e "${RED}强制删除本地分支失败${NC}"
                            fi
                        fi
                    fi
                    
                    # 删除远程分支
                    if [ "$delete_remote" = "y" ] || [ "$delete_remote" = "Y" ]; then
                        echo -e "${YELLOW}删除远程分支 $branch_to_delete...${NC}"
                        if git push origin --delete $branch_to_delete; then
                            echo -e "${GREEN}成功删除远程分支${NC}"
                        else
                            echo -e "${RED}删除远程分支失败${NC}"
                        fi
                    fi
                fi
                ;;
                
            5)  # 重命名分支
                if [ "$current_branch" = "main" ] || [ "$current_branch" = "master" ]; then
                    echo -e "${RED}警告: 您正在尝试重命名主分支，这可能会导致问题${NC}"
                    echo -e "${YELLOW}是否继续? (y/N)${NC}"
                    read -n 1 continue_rename
                    echo
                    if [ "$continue_rename" != "y" ] && [ "$continue_rename" != "Y" ]; then
                        echo -e "${YELLOW}已取消重命名操作${NC}"
                        break
                    fi
                fi
                
                echo -e "${YELLOW}请输入新的分支名称:${NC}"
                read new_branch_name
                if [ -z "$new_branch_name" ]; then
                    echo -e "${RED}新分支名不能为空${NC}"
                else
                    # 重命名本地分支
                    if git branch -m $new_branch_name; then
                        echo -e "${GREEN}成功重命名本地分支为: $new_branch_name${NC}"
                        
                        # 询问是否处理远程分支
                        echo -e "${YELLOW}是否更新远程分支? (y/N)${NC}"
                        read -n 1 update_remote
                        echo
                        if [ "$update_remote" = "y" ] || [ "$update_remote" = "Y" ]; then
                            # 删除旧的远程分支，推送新的分支
                            echo -e "${YELLOW}删除旧的远程分支并推送新分支...${NC}"
                            if git push origin :$current_branch && git push -u origin $new_branch_name; then
                                echo -e "${GREEN}成功更新远程分支${NC}"
                            else
                                echo -e "${RED}更新远程分支失败${NC}"
                                echo -e "${YELLOW}您可能需要手动执行:${NC}"
                                echo -e "${YELLOW}  git push origin :$current_branch${NC}"
                                echo -e "${YELLOW}  git push -u origin $new_branch_name${NC}"
                            fi
                        fi
                    else
                        echo -e "${RED}重命名分支失败${NC}"
                    fi
                fi
                ;;
                
            6)  # 基于远程分支创建本地分支
                echo -e "${YELLOW}获取远程分支信息...${NC}"
                git fetch
                
                echo -e "${YELLOW}可用的远程分支:${NC}"
                git branch -r
                echo
                
                echo -e "${YELLOW}请输入远程分支名称(不含origin/):${NC}"
                read remote_branch
                if [ -z "$remote_branch" ]; then
                    echo -e "${RED}分支名不能为空${NC}"
                else
                    echo -e "${YELLOW}请输入本地分支名称 (默认: $remote_branch):${NC}"
                    read local_branch
                    local_branch="${local_branch:-$remote_branch}"
                    
                    # 创建并切换到新的本地分支
                    if git checkout -b $local_branch origin/$remote_branch; then
                        echo -e "${GREEN}成功创建并切换到本地分支: $local_branch${NC}"
                    else
                        echo -e "${RED}创建本地分支失败，请检查远程分支名称是否正确${NC}"
                    fi
                fi
                ;;
                
            0)  # 返回主菜单
                return 0
                ;;
                
            *)
                echo -e "${RED}无效选项，请重新选择${NC}"
                ;;
        esac
        
        echo
        echo -e "${YELLOW}按任意键继续...${NC}"
        read -n 1
    done
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
        4)
            branch_management
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
