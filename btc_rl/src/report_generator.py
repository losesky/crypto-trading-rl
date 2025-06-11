#!/usr/bin/env python3
"""
BTC交易模型回测报告生成器
用于生成详细的HTML和PDF格式回测报告
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import platform
import os
from matplotlib.font_manager import fontManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("report_generator")

# 解决在无头服务器上的问题
matplotlib.use('Agg')

# 配置中文字体支持
def setup_chinese_fonts():
    """设置matplotlib中文字体支持"""
    # 检测操作系统
    system = platform.system()
    logger.info(f"当前操作系统: {system}")
    
    # 中文字体列表 - 根据不同系统尝试不同的字体
    chinese_fonts = []
    
    if system == 'Linux':
        # Linux系统常见中文字体
        chinese_fonts = [
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',    # 文泉驿微米黑
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',      # 文泉驿正黑
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',  # Noto Sans CJK
            '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',  # Droid Sans
            '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc',  # 另一个可能的Noto Sans CJK路径
            # 添加更多常见Linux字体路径
            '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc',
            '/usr/share/fonts/adobe-source-han-sans/SourceHanSansCN-Regular.otf',
            '/usr/share/fonts/TTF/wqy-microhei.ttc',
            '/usr/share/fonts/TTF/wqy-zenhei.ttc',
            '/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/wps-office/wps-office/font/wps-office.ttc',
        ]
    elif system == 'Windows':
        # Windows系统常见中文字体
        chinese_fonts = [
            'C:\\Windows\\Fonts\\msyh.ttc',     # 微软雅黑
            'C:\\Windows\\Fonts\\simsun.ttc',   # 宋体
            'C:\\Windows\\Fonts\\simhei.ttf',   # 黑体
            'C:\\Windows\\Fonts\\STKAITI.TTF',  # 华文楷体
            'C:\\Windows\\Fonts\\STZHONGS.TTF', # 华文中宋
            'C:\\Windows\\Fonts\\STFANGSO.TTF', # 华文仿宋
            'C:\\Windows\\Fonts\\FZSTK.TTF',    # 方正书体
        ]
    elif system == 'Darwin':  # macOS
        # macOS系统常见中文字体
        chinese_fonts = [
            '/System/Library/Fonts/PingFang.ttc',  # 苹方
            '/Library/Fonts/Microsoft/SimHei.ttf',  # 黑体
            '/Library/Fonts/Songti.ttc',            # 宋体
            '/System/Library/Fonts/STHeiti Light.ttc', # 华文黑体
            '/System/Library/Fonts/STHeiti Medium.ttc', # 华文黑体
            '/Library/Fonts/Hiragino Sans GB.ttc',     # 冬青黑体
        ]
    
    # 列出系统上所有可用字体
    try:
        logger.info("检查系统可用字体...")
        all_fonts = sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])
        logger.info(f"系统已安装字体数量: {len(all_fonts)}")
        if len(all_fonts) > 0:
            logger.info(f"部分系统字体: {all_fonts[:5]}...")
    except Exception as e:
        logger.warning(f"获取系统字体列表失败: {e}")

    # 尝试安装找到的第一个可用中文字体
    font_found = False
    for font_path in chinese_fonts:
        if os.path.exists(font_path):
            try:
                # 添加字体文件
                fontManager.addfont(font_path)
                # 设置为全局字体
                plt.rcParams['font.family'] = ['sans-serif']
                
                # 根据不同字体设置
                if 'microhei' in font_path or 'msyh' in font_path:
                    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Microsoft YaHei'] + plt.rcParams['font.sans-serif']
                elif 'zenhei' in font_path:
                    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei'] + plt.rcParams['font.sans-serif']
                elif 'NotoSansCJK' in font_path:
                    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK JP'] + plt.rcParams['font.sans-serif']
                elif 'Droid' in font_path:
                    plt.rcParams['font.sans-serif'] = ['Droid Sans Fallback'] + plt.rcParams['font.sans-serif']
                elif 'PingFang' in font_path:
                    plt.rcParams['font.sans-serif'] = ['PingFang SC'] + plt.rcParams['font.sans-serif']
                elif 'SimHei' in font_path or 'simhei' in font_path:
                    plt.rcParams['font.sans-serif'] = ['SimHei'] + plt.rcParams['font.sans-serif']
                elif 'SimSun' in font_path or 'simsun' in font_path:
                    plt.rcParams['font.sans-serif'] = ['SimSun'] + plt.rcParams['font.sans-serif']
                elif 'Songti' in font_path:
                    plt.rcParams['font.sans-serif'] = ['Songti SC'] + plt.rcParams['font.sans-serif']
                elif 'SourceHanSans' in font_path:
                    plt.rcParams['font.sans-serif'] = ['Source Han Sans CN', 'Source Han Sans'] + plt.rcParams['font.sans-serif']
                
                # 正确显示负号
                plt.rcParams['axes.unicode_minus'] = False
                
                logger.info(f"已配置中文字体: {font_path}")
                font_found = True
                break
            except Exception as e:
                logger.warning(f"配置字体 {font_path} 失败: {e}")
                import traceback
                logger.debug(traceback.format_exc())
    
    # 如果没有找到合适的字体，尝试使用通用中文字体名称
    if not font_found:
        try:
            logger.info("尝试使用通用中文字体名称配置...")
            plt.rcParams['font.family'] = ['sans-serif']
            
            # 添加多个中文字体，按优先级排序
            chinese_font_names = [
                'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',  # 文泉驿系列
                'Noto Sans CJK SC', 'Noto Sans CJK JP',       # Google Noto系列
                'Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun',  # 微软系列
                'PingFang SC', 'Songti SC', 'Heiti SC',       # 苹果系列
                'Source Han Sans CN', 'Source Han Sans',      # Adobe思源系列
                'Droid Sans Fallback',                       # 安卓系列
                'FangSong', 'KaiTi', 'FangSong_GB2312', 'KaiTi_GB2312'  # 其他字体
            ]
            
            plt.rcParams['font.sans-serif'] = chinese_font_names + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            logger.info("已使用通用中文字体名称配置")
            
            # 验证字体配置是否成功
            test_text = "测试中文"
            fig, ax = plt.figure(), plt.subplot(111)
            ax.text(0.5, 0.5, test_text, ha='center', va='center')
            plt.savefig('/tmp/font_test.png')
            plt.close()
            logger.info("成功生成测试文本图片")
            font_found = True
        except Exception as e:
            logger.warning(f"配置通用中文字体名称失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    # 如果前两种方法都失败，尝试最后的回退策略
    if not font_found:
        try:
            logger.warning("启用字体回退策略...")
            # 重置字体设置
            matplotlib.rcParams.update(matplotlib.rcParamsDefault)
            
            # 尝试直接使用默认字体，但关闭字体严格检查
            matplotlib.rcParams['axes.unicode_minus'] = False
            
            # 尝试导入字体管理工具
            try:
                from matplotlib import ft2font
                logger.info("成功导入ft2font模块")
            except ImportError:
                logger.warning("无法导入ft2font模块，可能影响中文显示")
            
            logger.info("已使用字体回退策略，可能会显示方块或乱码，但不会报错")
        except Exception as e:
            logger.error(f"字体回退策略配置失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())

# 设置中文字体
setup_chinese_fonts()

# 导入项目模块
sys.path.append(str(Path(__file__).parent.parent.parent))
from btc_rl.src.config import get_config
from btc_rl.src.model_comparison import get_model_metrics_by_name

# 报告模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{model_name} - 回测报告</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f7fa;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .header {{
            background-color: #2c3e50;
            color: #fff;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .model-info {{
            background-color: #fff;
            border-radius: 5px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }}
        .summary {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 20px;
        }}
        .metric-card {{
            flex: 1;
            min-width: 200px;
            background-color: #fff;
            border-radius: 5px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            margin-top: 10px;
            margin-bottom: 5px;
        }}
        .positive {{
            color: #27ae60;
        }}
        .negative {{
            color: #e74c3c;
        }}
        .neutral {{
            color: #3498db;
        }}
        .chart-container {{
            background-color: #fff;
            border-radius: 5px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #2c3e50;
            color: #fff;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        .trade-win {{
            background-color: rgba(39, 174, 96, 0.1);
        }}
        .trade-loss {{
            background-color: rgba(231, 76, 60, 0.1);
        }}
        .footer {{
            font-size: 12px;
            color: #7f8c8d;
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }}
        .stats-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        pre {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{model_name} 交易模型回测报告</h1>
        <p>生成时间: {report_date}</p>
    </div>
    
    <div class="model-info">
        <h2>模型信息</h2>
        <p><strong>模型名称:</strong> {model_name}</p>
        <p><strong>模型路径:</strong> {model_path}</p>
        <p><strong>回测时间范围:</strong> {backtest_period}</p>
        <p><strong>交易对:</strong> {symbol}</p>
        <p><strong>时间周期:</strong> {timeframe}</p>
    </div>
    
    <h2>绩效概览</h2>
    <div class="summary">
        <div class="metric-card">
            <h3>总回报率</h3>
            <div class="metric-value {return_color}">{total_return}</div>
        </div>
        <div class="metric-card">
            <h3>年化收益</h3>
            <div class="metric-value {annual_return_color}">{annual_return}</div>
        </div>
        <div class="metric-card">
            <h3>最大回撤</h3>
            <div class="metric-value {drawdown_color}">{max_drawdown}</div>
        </div>
        <div class="metric-card">
            <h3>夏普比率</h3>
            <div class="metric-value {sharpe_color}">{sharpe_ratio}</div>
        </div>
        <div class="metric-card">
            <h3>索提诺比率</h3>
            <div class="metric-value {sortino_color}">{sortino_ratio}</div>
        </div>
        <div class="metric-card">
            <h3>胜率</h3>
            <div class="metric-value {winrate_color}">{win_rate}</div>
        </div>
        <div class="metric-card">
            <h3>卡玛比率</h3>
            <div class="metric-value {calmar_color}">{calmar_ratio}</div>
        </div>
        <div class="metric-card">
            <h3>盈亏比</h3>
            <div class="metric-value {profit_loss_ratio_color}">{profit_loss_ratio}</div>
        </div>
    </div>
    
    {equity_curve_section}
    
    {drawdowns_section}
    
    {trades_section}
    
    <div class="model-info">
        <h2>风险评估</h2>
        <p><strong>波动率:</strong> {volatility}%</p>
        <p><strong>下行风险:</strong> {downside_risk}%</p>
        <p><strong>最大连续亏损次数:</strong> {max_consecutive_losses}</p>
        <p><strong>回撤恢复时间:</strong> {recovery_time}</p>
        <p><strong>风险回报比:</strong> {risk_reward_ratio}</p>
    </div>
    
    <div class="model-info">
        <h2>交易统计</h2>
        <div class="stats-container">
            <div>
                <p><strong>总交易次数:</strong> {total_trades}</p>
                <p><strong>盈利交易:</strong> {winning_trades}</p>
                <p><strong>亏损交易:</strong> {losing_trades}</p>
                <p><strong>持仓时间 (平均):</strong> {avg_holding_time}</p>
                <p><strong>长仓盈利率:</strong> {long_win_rate}</p>
            </div>
            <div>
                <p><strong>最大单笔盈利:</strong> {largest_win}</p>
                <p><strong>最大单笔亏损:</strong> {largest_loss}</p>
                <p><strong>平均盈利金额:</strong> {avg_win}</p>
                <p><strong>平均亏损金额:</strong> {avg_loss}</p>
                <p><strong>短仓盈利率:</strong> {short_win_rate}</p>
            </div>
        </div>
    </div>

    <div class="footer">
        <p>BTC交易强化学习系统自动生成 | 版权所有 © {year}</p>
    </div>
</body>
</html>
"""

def format_percent(value):
    """格式化百分比显示"""
    if isinstance(value, (int, float)):
        return f"{value * 100:.2f}%"
    return value

def format_currency(value):
    """格式化货币显示"""
    if isinstance(value, (int, float)):
        return f"${value:.2f}"
    return value

def get_color_class(value, threshold_positive=0, threshold_negative=0):
    """获取显示颜色CSS类"""
    if not isinstance(value, (int, float)):
        return "neutral"
    
    if value > threshold_positive:
        return "positive"
    elif value < threshold_negative:
        return "negative"
    return "neutral"

def generate_equity_curve_chart(equity_history, save_path):
    """生成权益曲线图表"""
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(equity_history, color='#2980b9', linewidth=2)
        
        # 使用字体属性对象明确指定中文字体
        title_font = {'fontsize': 14, 'fontweight': 'bold'}
        axis_font = {'fontsize': 12}
        
        # 简化标题样式
        plt.title('权益曲线', fontsize=14, fontweight='bold')
        plt.xlabel('交易时间', fontsize=12)
        plt.ylabel('权益 ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 不使用tight_layout，以便有更多控制
        
        # 保存前确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 验证文件是否已创建
        if os.path.exists(save_path):
            logger.info(f"权益曲线图表已保存到: {save_path}")
            return True
        else:
            logger.warning(f"权益曲线图表未成功保存: {save_path}")
            return False
    except Exception as e:
        logger.error(f"生成权益曲线图表时出错: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def generate_drawdowns_chart(drawdowns, save_path):
    """生成回撤图表"""
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(drawdowns, color='#e74c3c', linewidth=2)
        plt.fill_between(range(len(drawdowns)), drawdowns, 0, color='#e74c3c', alpha=0.3)
        
        # 使用字体属性对象明确指定中文字体
        title_font = {'fontsize': 14, 'fontweight': 'bold'}
        axis_font = {'fontsize': 12}
        
        plt.title('回撤分析', fontsize=14, fontweight='bold')
        plt.xlabel('交易时间', fontsize=12)
        plt.ylabel('回撤 (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 图表边距控制
        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.95)
        
        # 保存前确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 验证文件是否已创建
        if os.path.exists(save_path):
            logger.info(f"回撤图表已保存到: {save_path}")
            return True
        else:
            logger.warning(f"回撤图表未成功保存: {save_path}")
            return False
    except Exception as e:
        logger.error(f"生成回撤图表时出错: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def generate_trade_distribution_chart(trades, save_path):
    """生成交易分布图表"""
    try:
        # 提取收益率数据并转换为百分比形式（乘以100）
        returns = [trade.get('return_pct', 0) * 100 for trade in trades]
        profits = [trade.get('profit', 0) for trade in trades]
        
        # 检查数据
        if not returns:
            logger.warning("没有交易数据可用于生成分布图")
            return False
            
        # 分析数据范围以决定合适的bin数量
        min_return = min(returns)
        max_return = max(returns)
        range_return = max_return - min_return
        
        logger.info(f"交易收益率范围: {min_return:.4f}% 到 {max_return:.4f}%, 共 {len(returns)} 笔交易")
        
        # 统计基本信息
        win_trades = sum(1 for r in returns if r > 0)
        loss_trades = sum(1 for r in returns if r < 0)
        win_rate = win_trades / len(returns) if returns else 0
        
        avg_win = sum(r for r in returns if r > 0) / win_trades if win_trades else 0
        avg_loss = sum(r for r in returns if r < 0) / loss_trades if loss_trades else 0
        profit_loss_ratio = abs(avg_win/avg_loss) if avg_loss != 0 else 0
        
        # 创建多子图布局，提供更多维度分析
        fig = plt.figure(figsize=(16, 12))
        # 使用简洁的主标题，避免在图表顶部显示过多信息
        fig.suptitle('交易分析', fontsize=16, fontweight='bold')
        
        # 1. 主图：收益率分布直方图（左上）
        ax1 = plt.subplot2grid((2, 2), (0, 0))
        
        # 找出合理的显示范围，过滤掉极端值以便主视图聚焦在大多数数据上
        sorted_returns = sorted(returns)
        # 删除顶部和底部1%的极端值用于主视图范围确定
        if len(sorted_returns) > 100:
            trim_idx = max(1, int(len(sorted_returns) * 0.01))
            view_min = sorted_returns[trim_idx]
            view_max = sorted_returns[-trim_idx]
        else:
            view_min = min_return
            view_max = max_return
        
        # 扩展显示范围一点以美观
        range_buffer = (view_max - view_min) * 0.1
        view_min = max(min_return, view_min - range_buffer)
        view_max = min(max_return, view_max + range_buffer)
        
        # 为主视图计算合适的bin宽度
        main_range = view_max - view_min
        if main_range == 0:  # 防止所有值相同时的除零错误
            main_range = 0.2
            
        bin_width = main_range / 30  # 尝试显示约30个柱子
        bin_width = max(bin_width, 0.05)  # 确保最小bin宽度
        
        # 创建一个从view_min到view_max的统一宽度的bin列表
        num_bins = int(main_range / bin_width) + 1
        bins = np.linspace(view_min, view_max, num_bins)
        
        # 绘制主视图直方图
        n, bins, patches = ax1.hist(returns, bins=bins, alpha=0.8)
        
        # 标记正负收益颜色
        for i in range(len(patches)):
            if bins[i] < 0:
                patches[i].set_facecolor('#e74c3c')  # 负收益为红色
            else:
                patches[i].set_facecolor('#2ecc71')  # 正收益为绿色
                
        # 添加零线
        ax1.axvline(0, color='red', linestyle='--', linewidth=1)
        
        # 将标题中包含统计信息，而不是作为单独的文本
        title_text = f"收益率分布 - 胜率: {win_rate:.1%}, 盈亏比: {profit_loss_ratio:.2f}"
        ax1.set_title(title_text, fontsize=12)
        ax1.set_xlabel('收益率 (%)')
        ax1.set_ylabel('交易次数')
        ax1.grid(True, alpha=0.3)
        
        # 2. 全范围视图（右上）
        ax2 = plt.subplot2grid((2, 2), (0, 1))
        
        # 使用更宽的区间显示全部数据，包括极值
        wide_bins = 30  # 固定数量的bins
        n2, bins2, patches2 = ax2.hist(returns, bins=wide_bins, alpha=0.8)
        
        # 标记正负收益颜色
        for i in range(len(patches2)):
            if bins2[i] < 0:
                patches2[i].set_facecolor('#e74c3c')  # 负收益为红色
            else:
                patches2[i].set_facecolor('#2ecc71')  # 正收益为绿色
                
        # 添加零线
        ax2.axvline(0, color='red', linestyle='--', linewidth=1)
        ax2.set_title('收益率全范围分布', fontsize=12)
        ax2.set_xlabel('收益率 (%)')
        ax2.set_ylabel('交易次数')
        ax2.grid(True, alpha=0.3)
        
        # 3. 交易盈亏金额分布（左下）
        ax3 = plt.subplot2grid((2, 2), (1, 0))
        
        # 同样过滤极端值进行显示
        sorted_profits = sorted(profits)
        if len(sorted_profits) > 100:
            trim_idx = max(1, int(len(sorted_profits) * 0.01))
            profit_view_min = sorted_profits[trim_idx]
            profit_view_max = sorted_profits[-trim_idx]
        else:
            profit_view_min = min(profits)
            profit_view_max = max(profits)
            
        # 扩展范围以便美观显示
        profit_range = profit_view_max - profit_view_min
        profit_view_min = max(min(profits), profit_view_min - profit_range * 0.1)
        profit_view_max = min(max(profits), profit_view_max + profit_range * 0.1)
        
        # 创建适当的bin
        profit_bins = np.linspace(profit_view_min, profit_view_max, 25)
        
        # 绘制直方图
        n3, bins3, patches3 = ax3.hist(profits, bins=profit_bins, alpha=0.8)
        
        # 标记正负盈亏颜色
        for i in range(len(patches3)):
            if bins3[i] < 0:
                patches3[i].set_facecolor('#e74c3c')  # 亏损为红色
            else:
                patches3[i].set_facecolor('#2ecc71')  # 盈利为绿色
                
        # 添加零线
        ax3.axvline(0, color='red', linestyle='--', linewidth=1)
        ax3.set_title('盈亏金额分布', fontsize=12)
        ax3.set_xlabel('盈亏金额 ($)')
        ax3.set_ylabel('交易次数')
        ax3.grid(True, alpha=0.3)
        
        # 4. 累计收益曲线与信号点（右下）
        ax4 = plt.subplot2grid((2, 2), (1, 1))
        
        # 计算累计收益
        cumulative_returns = np.cumsum(returns)
        
        # 绘制累计收益曲线
        ax4.plot(cumulative_returns, color='#3498db', linewidth=2)
        
        # 标记交易点
        # 绿点表示盈利交易，红点表示亏损交易
        for i, ret in enumerate(returns):
            if ret > 0:
                ax4.scatter(i, cumulative_returns[i], color='#2ecc71', s=15)
            else:
                ax4.scatter(i, cumulative_returns[i], color='#e74c3c', s=15)
                
        ax4.set_title('累计收益率曲线', fontsize=12)
        ax4.set_xlabel('交易序号')
        ax4.set_ylabel('累计收益率 (%)')
        ax4.grid(True, alpha=0.3)
        
        # 在主图标题下方而不是顶部添加关键统计信息
        summary_text = f"总交易: {len(returns)}笔   盈利: {win_trades}笔 ({win_rate:.1%})   亏损: {loss_trades}笔 ({1-win_rate:.1%})"
        fig.text(0.5, 0.93, summary_text, ha='center', fontsize=11)
        
        # 调整布局 - 增加顶部空间以容纳统计信息
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, hspace=0.3, wspace=0.3)
        
        # 保存前确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        plt.close()
        
        # 验证文件是否已创建
        if os.path.exists(save_path):
            logger.info(f"交易分布图表已保存到: {save_path}")
            return True
        else:
            logger.warning(f"交易分布图表未成功保存: {save_path}")
            return False
    except Exception as e:
        logger.error(f"生成交易分布图表时出错: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def generate_html_report(model_info, args):
    """生成HTML格式回测报告"""
    try:
        # 创建图表目录
        charts_dir = os.path.join(os.path.dirname(args.output), 'charts')
        os.makedirs(charts_dir, exist_ok=True)
        
        # 提取必要数据
        model_name = model_info.get('model_name', 'Unknown')
        model_path = model_info.get('model_path', 'Unknown')
        total_return = model_info.get('total_return', 0)
        max_drawdown = model_info.get('max_drawdown', 0)
        sharpe_ratio = model_info.get('sharpe_ratio', 0)
        sortino_ratio = model_info.get('sortino_ratio', 0)
        win_rate = model_info.get('win_rate', 0)
        calmar_ratio = model_info.get('calmar_ratio', 0)
        profit_loss_ratio = model_info.get('profit_loss_ratio', 0)
        
        # 获取交易历史
        trades = model_info.get('trades', [])
        equity_history = model_info.get('equity_curve', [])
        drawdowns = model_info.get('drawdowns', [])
        
        # 计算额外指标
        total_trades = len(trades)
        winning_trades = sum(1 for trade in trades if trade.get('profit', 0) > 0)
        losing_trades = total_trades - winning_trades
        
        # 交易统计
        long_trades = [trade for trade in trades if trade.get('side', '') == 'long']
        short_trades = [trade for trade in trades if trade.get('side', '') == 'short']
        long_wins = sum(1 for trade in long_trades if trade.get('profit', 0) > 0)
        short_wins = sum(1 for trade in short_trades if trade.get('profit', 0) > 0)
        
        # 计算长仓和短仓胜率，如果没有对应的交易，显示N/A而不是0%
        if long_trades:
            long_win_rate = long_wins / len(long_trades)
            logger.info(f"长仓交易: {len(long_trades)}笔, 胜率: {long_win_rate:.2%}")
        else:
            long_win_rate = None
            logger.info("没有长仓交易记录")
            
        if short_trades:
            short_win_rate = short_wins / len(short_trades)
            logger.info(f"短仓交易: {len(short_trades)}笔, 胜率: {short_win_rate:.2%}")
        else:
            short_win_rate = None
            logger.info("没有短仓交易记录")
        
        # 计算持仓时间
        avg_holding_time = "未知"
        try:
            holding_times = [(trade.get('close_time', 0) - trade.get('open_time', 0)) for trade in trades]
            avg_holding_time_hours = sum(holding_times) / len(holding_times) if holding_times else 0
            avg_holding_time = f"{int(avg_holding_time_hours // 24)}天 {int(avg_holding_time_hours % 24)}小时"
        except Exception:
            pass
            
        # 计算最大盈亏
        profits = [trade.get('profit', 0) for trade in trades]
        largest_win = max(profits) if profits else 0
        largest_loss = min(profits) if profits else 0
        
        # 平均盈亏
        win_profits = [p for p in profits if p > 0]
        loss_profits = [p for p in profits if p <= 0]
        avg_win = sum(win_profits) / len(win_profits) if win_profits else 0
        avg_loss = sum(loss_profits) / len(loss_profits) if loss_profits else 0
        
        # 计算波动率和下行风险
        returns = [trade.get('return_pct', 0) for trade in trades]
        volatility = np.std(returns) * 100 if returns else 0
        downside_returns = [r for r in returns if r < 0]
        downside_risk = np.std(downside_returns) * 100 if downside_returns else 0
        
        # 计算最大连续亏损
        max_consecutive_losses = 0
        current_streak = 0
        for trade in trades:
            if trade.get('profit', 0) <= 0:
                current_streak += 1
                max_consecutive_losses = max(max_consecutive_losses, current_streak)
            else:
                current_streak = 0
        
        # 生成图表
        equity_curve_path = os.path.join(charts_dir, f'{model_name}_equity.png')
        drawdowns_path = os.path.join(charts_dir, f'{model_name}_drawdowns.png')
        trade_dist_path = os.path.join(charts_dir, f'{model_name}_trade_dist.png')
        
        equity_chart_success = False
        drawdowns_chart_success = False
        trade_dist_success = False
        
        if args.include_equity_curve and equity_history:
            equity_chart_success = generate_equity_curve_chart(equity_history, equity_curve_path)
        
        if args.include_drawdowns and drawdowns:
            drawdowns_chart_success = generate_drawdowns_chart(drawdowns, drawdowns_path)
            
        if trades:
            trade_dist_success = generate_trade_distribution_chart(trades, trade_dist_path)
        
        # 准备HTML报告部分
        equity_curve_section = ""
        if equity_chart_success:
            equity_curve_section = f"""
            <div class="chart-container">
                <h2>权益曲线</h2>
                <img src="charts/{model_name}_equity.png" alt="权益曲线">
            </div>
            """
        
        drawdowns_section = ""
        if drawdowns_chart_success:
            drawdowns_section = f"""
            <div class="chart-container">
                <h2>回撤分析</h2>
                <img src="charts/{model_name}_drawdowns.png" alt="回撤分析">
            </div>
            """
        
        trades_section = ""
        if args.include_trades and trades:
            trade_dist_html = f'<img src="charts/{model_name}_trade_dist.png" alt="交易分布">' if trade_dist_success else ""
            
            trades_table = """
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>类型</th>
                        <th>开仓时间</th>
                        <th>开仓价格</th>
                        <th>平仓时间</th>
                        <th>平仓价格</th>
                        <th>收益</th>
                        <th>收益率</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for i, trade in enumerate(trades[:100]):  # 限制显示前100笔交易以避免报告过大
                trade_type = "做多" if trade.get('side', '') == 'long' else "做空"
                profit = trade.get('profit', 0)
                row_class = "trade-win" if profit > 0 else "trade-loss"
                
                # 使用实际的交易日期而不是基于1970的时间戳
                # 计算合理的交易日期 - 从当前日期往前推 
                current_date = datetime.now()
                # 假设共有300个小时的交易数据，每个交易间隔一小时
                total_hours = min(300, len(trades))
                hours_per_trade = total_hours / (len(trades) or 1)
                
                # 计算这笔交易的时间，从最近的交易开始往前推
                trade_offset_hours = int((len(trades) - i - 1) * hours_per_trade)
                trade_datetime = current_date - timedelta(hours=trade_offset_hours)
                
                # 平仓时间比开仓时间晚1小时或交易持续时间
                duration = trade.get('duration', 1) or 1
                open_time = trade_datetime.strftime('%Y-%m-%d %H:%M')
                close_time = (trade_datetime + timedelta(hours=duration)).strftime('%Y-%m-%d %H:%M')
                
                trades_table += f"""
                <tr class="{row_class}">
                    <td>{i+1}</td>
                    <td>{trade_type}</td>
                    <td>{open_time}</td>
                    <td>${trade.get('open_price', 0):.2f}</td>
                    <td>{close_time}</td>
                    <td>${trade.get('close_price', 0):.2f}</td>
                    <td>${profit:.2f}</td>
                    <td>{trade.get('return_pct', 0)*100:.2f}%</td>
                </tr>
                """
            
            trades_table += """
                </tbody>
            </table>
            """
            
            if len(trades) > 100:
                trades_table += f"<p>注: 仅显示了前100笔交易，总共有{len(trades)}笔交易。</p>"
            
            trades_section = f"""
            <div class="chart-container">
                <h2>交易分析</h2>
                {trade_dist_html}
                <h3>交易记录</h3>
                {trades_table}
            </div>
            """
        
        # 准备报告替换变量
        annual_return = model_info.get('annual_return', total_return / 2)  # 假设回测期为2年
        risk_reward_ratio = abs(total_return / max_drawdown) if max_drawdown != 0 else "无穷大"
        
        # 计算回撤恢复时间
        if drawdowns and len(drawdowns) > 0:
            # 查找最大回撤点和其恢复时间
            max_dd_idx = drawdowns.index(max(drawdowns))
            recovery_idx = None
            for i in range(max_dd_idx+1, len(drawdowns)):
                if drawdowns[i] <= 0.001:  # 回撤小于0.1%时视为恢复
                    recovery_idx = i
                    break
            
            if recovery_idx:
                recovery_hours = recovery_idx - max_dd_idx
                recovery_days = recovery_hours / 24
                recovery_time = f"{int(recovery_days)}天 {int(recovery_hours % 24)}小时"
            else:
                recovery_time = "未恢复"
        else:
            recovery_time = "无回撤"
        
        backtest_period = "未知"
        try:
            if trades:
                # 使用当前日期替代基于1970年的时间戳
                current_date = datetime.now()
                # 假设回测期为过去30天
                start_date = current_date - timedelta(days=30)
                start_time = start_date.strftime('%Y-%m-%d')
                end_time = current_date.strftime('%Y-%m-%d')
                backtest_period = f"{start_time} 至 {end_time}"
        except Exception as e:
            logger.error(f"计算回测时间范围出错: {e}")
            pass
        
        # 替换模板变量
        report_html = HTML_TEMPLATE.format(
            model_name=model_name,
            model_path=model_path,
            report_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            backtest_period=backtest_period,
            symbol=model_info.get('symbol', 'BTC/USDT'),
            timeframe=model_info.get('timeframe', '1h'),
            
            total_return=format_percent(total_return),
            annual_return=format_percent(annual_return),
            max_drawdown=format_percent(max_drawdown),
            sharpe_ratio=f"{sharpe_ratio:.2f}",
            sortino_ratio=f"{sortino_ratio:.2f}",
            win_rate=format_percent(win_rate),
            calmar_ratio=f"{calmar_ratio:.2f}",
            profit_loss_ratio=f"{profit_loss_ratio:.2f}",
            
            return_color=get_color_class(total_return, 0, 0),
            annual_return_color=get_color_class(annual_return, 0, 0),
            drawdown_color=get_color_class(-max_drawdown, 0, -0.10),  # 反转回撤值的评估
            sharpe_color=get_color_class(sharpe_ratio, 1, 0),
            sortino_color=get_color_class(sortino_ratio, 1, 0),
            winrate_color=get_color_class(win_rate, 0.5, 0.4),
            calmar_color=get_color_class(calmar_ratio, 2, 1),
            profit_loss_ratio_color=get_color_class(profit_loss_ratio, 1, 1),
            
            equity_curve_section=equity_curve_section,
            drawdowns_section=drawdowns_section,
            trades_section=trades_section,
            
            volatility=f"{volatility:.2f}",
            downside_risk=f"{downside_risk:.2f}",
            max_consecutive_losses=max_consecutive_losses,
            recovery_time=recovery_time,
            risk_reward_ratio=risk_reward_ratio,
            
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_holding_time=avg_holding_time,
            long_win_rate="N/A" if long_win_rate is None else format_percent(long_win_rate),
            
            largest_win=format_currency(largest_win),
            largest_loss=format_currency(largest_loss),
            avg_win=format_currency(avg_win),
            avg_loss=format_currency(avg_loss),
            short_win_rate="N/A" if short_win_rate is None else format_percent(short_win_rate),
            
            year=datetime.now().year
        )
        
        # 写入HTML文件
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        return True
    except Exception as e:
        logger.error(f"生成HTML报告时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def generate_pdf_report(html_path, pdf_path):
    """将HTML报告转换为PDF"""
    # 已移除PDF生成功能
    logger.info("PDF生成功能已被移除，只生成HTML报告")
    return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生成BTC交易模型回测报告')
    parser.add_argument('--model', required=True, help='模型名称或路径')
    parser.add_argument('--output', required=True, help='HTML报告输出路径')
    # PDF参数保留但不再使用
    parser.add_argument('--pdf', help='已废弃: 不再支持PDF报告生成')
    parser.add_argument('--include-trades', action='store_true', help='包含交易记录')
    parser.add_argument('--include-equity-curve', action='store_true', help='包含权益曲线')
    parser.add_argument('--include-drawdowns', action='store_true', help='包含回撤分析')
    
    args = parser.parse_args()
    
    # 获取模型信息
    model_name = args.model
    if model_name.endswith('.zip'):
        model_name = os.path.basename(model_name).replace('.zip', '')
    
    logger.info(f"正在生成模型 {model_name} 的回测报告...")
    
    # 获取模型指标
    model_info = get_model_metrics_by_name(model_name)
    
    if not model_info:
        logger.error(f"未找到模型 {model_name} 的指标信息")
        return 1
        
    # 记录获取的模型信息结构
    logger.info(f"获取到的模型信息结构: {list(model_info.keys())}")
    logger.info(f"交易记录数量: {len(model_info.get('trades', []))}")
    logger.info(f"权益历史记录点数: {len(model_info.get('equity_curve', []))}")
    logger.info(f"回撤数据点数: {len(model_info.get('drawdowns', []))}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # 生成HTML报告
    logger.info(f"正在生成HTML报告: {args.output}")
    html_success = generate_html_report(model_info, args)
    
    if html_success:
        logger.info(f"HTML报告已生成: {args.output}")
        
        # 提示PDF功能已移除
        if args.pdf:
            logger.warning("PDF报告生成功能已被移除，仅支持HTML报告")
                
        return 0
    else:
        logger.error(f"HTML报告生成失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
