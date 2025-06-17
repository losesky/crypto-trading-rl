import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional
import logging

logger = logging.getLogger(__name__)

class TradingMetrics:
    """
    交易性能指标计算工具类
    用于计算各种交易绩效指标，如夏普比率、最大回撤、胜率等
    """
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0, annualization_factor: int = 252) -> float:
        """
        计算夏普比率
        
        Args:
            returns: 日收益率序列
            risk_free_rate: 无风险收益率，默认为0
            annualization_factor: 年化系数，默认为252（交易日）
            
        Returns:
            float: 夏普比率
        """
        if len(returns) < 2:
            return 0.0
            
        excess_returns = returns - risk_free_rate
        
        # 计算平均收益率和标准差
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns, ddof=1)  # 使用样本标准差
        
        if std_return == 0:
            return 0.0
            
        # 计算并返回年化夏普比率
        return mean_return / std_return * np.sqrt(annualization_factor)
        
    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0, annualization_factor: int = 252) -> float:
        """
        计算索提诺比率（只考虑下行波动率的夏普比率）
        
        Args:
            returns: 日收益率序列
            risk_free_rate: 无风险收益率，默认为0
            annualization_factor: 年化系数，默认为252（交易日）
            
        Returns:
            float: 索提诺比率
        """
        if len(returns) < 2:
            return 0.0
            
        excess_returns = returns - risk_free_rate
        
        # 计算平均收益率
        mean_return = np.mean(excess_returns)
        
        # 计算下行标准差（仅考虑负收益）
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return float('inf')  # 没有负收益，返回无穷大
            
        downside_deviation = np.std(downside_returns, ddof=1)
        
        if downside_deviation == 0:
            return 0.0
            
        # 计算并返回年化索提诺比率
        return mean_return / downside_deviation * np.sqrt(annualization_factor)
        
    @staticmethod
    def calculate_max_drawdown(values: np.ndarray) -> Tuple[float, int, int]:
        """
        计算最大回撤及其开始和结束位置
        
        Args:
            values: 资产价值序列
            
        Returns:
            Tuple[float, int, int]: (最大回撤百分比, 开始位置, 结束位置)
        """
        if len(values) < 2:
            return 0.0, 0, 0
            
        # 计算历史峰值
        peak_values = np.maximum.accumulate(values)
        
        # 计算回撤序列
        drawdowns = (peak_values - values) / peak_values
        
        # 找出最大回撤及其位置
        max_dd = np.max(drawdowns)
        end_idx = np.argmax(drawdowns)
        
        # 找出最大回撤开始的位置（前一个峰值）
        peak_value = peak_values[end_idx]
        start_idx = np.where(values[:end_idx+1] == peak_value)[0][-1]
        
        return max_dd, start_idx, end_idx
        
    @staticmethod
    def calculate_calmar_ratio(returns: np.ndarray, max_drawdown: float, annualization_factor: int = 252) -> float:
        """
        计算卡玛比率（年化收益率除以最大回撤）
        
        Args:
            returns: 日收益率序列
            max_drawdown: 最大回撤（比例，非百分比）
            annualization_factor: 年化系数，默认为252（交易日）
            
        Returns:
            float: 卡玛比率
        """
        if max_drawdown == 0:
            return float('inf')  # 避免除以零
            
        # 计算年化收益率
        annualized_return = np.mean(returns) * annualization_factor
        
        return annualized_return / max_drawdown
        
    @staticmethod
    def calculate_win_rate(pnl_list: List[float]) -> Tuple[float, int, int]:
        """
        计算胜率
        
        Args:
            pnl_list: 每笔交易的盈亏列表
            
        Returns:
            Tuple[float, int, int]: (胜率, 盈利交易数, 亏损交易数)
        """
        if not pnl_list:
            return 0.0, 0, 0
            
        # 计算盈利和亏损交易数量
        profits = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p < 0]
        
        win_count = len(profits)
        loss_count = len(losses)
        total_trades = win_count + loss_count
        
        win_rate = win_count / total_trades if total_trades > 0 else 0.0
        
        return win_rate, win_count, loss_count
        
    @staticmethod
    def calculate_profit_factor(pnl_list: List[float]) -> float:
        """
        计算盈亏比（总盈利 / 总亏损的绝对值）
        
        Args:
            pnl_list: 每笔交易的盈亏列表
            
        Returns:
            float: 盈亏比
        """
        if not pnl_list:
            return 0.0
            
        # 分离盈利和亏损
        profits = [p for p in pnl_list if p > 0]
        losses = [abs(p) for p in pnl_list if p < 0]
        
        total_profit = sum(profits)
        total_loss = sum(losses)
        
        # 避免除以零
        if total_loss == 0:
            return float('inf') if total_profit > 0 else 0.0
            
        return total_profit / total_loss
        
    @staticmethod
    def calculate_average_profit_loss(pnl_list: List[float]) -> Tuple[float, float]:
        """
        计算平均盈利和平均亏损
        
        Args:
            pnl_list: 每笔交易的盈亏列表
            
        Returns:
            Tuple[float, float]: (平均盈利, 平均亏损)
        """
        if not pnl_list:
            return 0.0, 0.0
            
        # 分离盈利和亏损
        profits = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p < 0]
        
        avg_profit = np.mean(profits) if profits else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        
        return avg_profit, avg_loss
        
    @staticmethod
    def calculate_recovery_factor(total_return: float, max_drawdown: float) -> float:
        """
        计算恢复因子（总收益 / 最大回撤）
        
        Args:
            total_return: 总收益（比例，非百分比）
            max_drawdown: 最大回撤（比例，非百分比）
            
        Returns:
            float: 恢复因子
        """
        if max_drawdown == 0:
            return float('inf') if total_return > 0 else 0.0
            
        return total_return / max_drawdown
        
    @staticmethod
    def calculate_volatility(returns: np.ndarray, annualization_factor: int = 252) -> float:
        """
        计算波动率（收益率标准差）
        
        Args:
            returns: 日收益率序列
            annualization_factor: 年化系数，默认为252（交易日）
            
        Returns:
            float: 年化波动率
        """
        if len(returns) < 2:
            return 0.0
            
        # 计算日波动率
        daily_volatility = np.std(returns, ddof=1)
        
        # 年化波动率
        return daily_volatility * np.sqrt(annualization_factor)
        
    @staticmethod
    def calculate_expected_shortfall(returns: np.ndarray, alpha: float = 0.05) -> float:
        """
        计算期望损失（ES）/ 条件风险价值（CVaR）
        
        Args:
            returns: 日收益率序列
            alpha: 显著性水平，默认为0.05（5%）
            
        Returns:
            float: 期望损失
        """
        if len(returns) < 2:
            return 0.0
            
        # 计算VaR（风险价值）
        var = np.percentile(returns, alpha * 100)
        
        # 计算ES/CVaR（VaR以下收益的平均值）
        shortfall_returns = returns[returns <= var]
        
        if len(shortfall_returns) == 0:
            return var
            
        return np.mean(shortfall_returns)
        
    @staticmethod
    def calculate_information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray, annualization_factor: int = 252) -> float:
        """
        计算信息比率（超额收益 / 跟踪误差）
        
        Args:
            returns: 策略日收益率序列
            benchmark_returns: 基准日收益率序列
            annualization_factor: 年化系数，默认为252（交易日）
            
        Returns:
            float: 信息比率
        """
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 0.0
            
        # 计算超额收益
        excess_returns = returns - benchmark_returns
        
        # 计算平均超额收益和跟踪误差
        mean_excess_return = np.mean(excess_returns)
        tracking_error = np.std(excess_returns, ddof=1)
        
        if tracking_error == 0:
            return 0.0
            
        # 计算并返回年化信息比率
        return mean_excess_return / tracking_error * np.sqrt(annualization_factor)
        
    @staticmethod
    def calculate_r_squared(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """
        计算R平方（策略收益与基准收益的决定系数）
        
        Args:
            returns: 策略日收益率序列
            benchmark_returns: 基准日收益率序列
            
        Returns:
            float: R平方值
        """
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 0.0
            
        # 计算相关系数
        correlation = np.corrcoef(returns, benchmark_returns)[0, 1]
        
        # R平方是相关系数的平方
        return correlation ** 2
        
    @staticmethod
    def calculate_alpha_beta(returns: np.ndarray, benchmark_returns: np.ndarray, risk_free_rate: float = 0, annualization_factor: int = 252) -> Tuple[float, float]:
        """
        计算阿尔法和贝塔系数
        
        Args:
            returns: 策略日收益率序列
            benchmark_returns: 基准日收益率序列
            risk_free_rate: 无风险收益率，默认为0
            annualization_factor: 年化系数，默认为252（交易日）
            
        Returns:
            Tuple[float, float]: (阿尔法, 贝塔)
        """
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 0.0, 0.0
            
        # 计算超额收益
        excess_returns = returns - risk_free_rate
        excess_benchmark_returns = benchmark_returns - risk_free_rate
        
        # 计算贝塔系数（协方差 / 基准方差）
        covariance = np.cov(excess_returns, excess_benchmark_returns)[0, 1]
        benchmark_variance = np.var(excess_benchmark_returns, ddof=1)
        
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 0.0
        
        # 计算阿尔法（年化）
        mean_excess_return = np.mean(excess_returns)
        mean_excess_benchmark_return = np.mean(excess_benchmark_returns)
        
        alpha = mean_excess_return - beta * mean_excess_benchmark_return
        alpha_annualized = ((1 + alpha) ** annualization_factor) - 1
        
        return alpha_annualized, beta
        
    @staticmethod
    def calculate_drawdowns(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算所有回撤期间及其深度
        
        Args:
            values: 资产价值序列
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (回撤序列, 回撤开始位置, 回撤结束位置)
        """
        if len(values) < 2:
            return np.array([]), np.array([], dtype=int), np.array([], dtype=int)
            
        # 计算历史峰值
        peak_values = np.maximum.accumulate(values)
        
        # 计算回撤序列
        drawdowns = (peak_values - values) / peak_values
        
        # 找出回撤开始和结束位置
        # 回撤开始：价值达到新高点
        # 回撤结束：回撤变为0（回升到新高点）
        
        # 确定新高点位置
        peak_indices = np.where(values == peak_values)[0]
        
        # 对于每个高点，找出对应的回撤结束点
        starts = []
        ends = []
        
        for i in range(len(peak_indices) - 1):
            start_idx = peak_indices[i]
            next_peak_idx = peak_indices[i + 1]
            
            # 如果高点之间有回撤
            if np.any(drawdowns[start_idx:next_peak_idx] > 0):
                max_dd_idx = start_idx + np.argmax(drawdowns[start_idx:next_peak_idx])
                starts.append(start_idx)
                ends.append(max_dd_idx)
        
        # 处理最后一个高点到数据末尾的回撤
        if len(peak_indices) > 0 and peak_indices[-1] < len(values) - 1:
            start_idx = peak_indices[-1]
            if np.any(drawdowns[start_idx:] > 0):
                max_dd_idx = start_idx + np.argmax(drawdowns[start_idx:])
                starts.append(start_idx)
                ends.append(max_dd_idx)
        
        # 转换为numpy数组
        starts = np.array(starts, dtype=int)
        ends = np.array(ends, dtype=int)
        
        # 提取对应的回撤值
        drawdown_values = np.array([drawdowns[end] for end in ends])
        
        return drawdown_values, starts, ends
        
    @staticmethod
    def calculate_underwater_periods(values: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        计算水下期（连续未创新高的期间）
        
        Args:
            values: 资产价值序列
            
        Returns:
            List[Tuple[int, int, int]]: [(开始位置, 结束位置, 持续天数)]
        """
        if len(values) < 2:
            return []
            
        # 计算历史峰值
        peak_values = np.maximum.accumulate(values)
        
        # 找出创新高的位置
        new_highs = np.diff(peak_values) > 0
        new_high_indices = np.where(np.append(True, new_highs))[0]  # 添加起始位置
        
        # 计算水下期
        underwater_periods = []
        for i in range(len(new_high_indices) - 1):
            start_idx = new_high_indices[i]
            end_idx = new_high_indices[i + 1] - 1
            duration = end_idx - start_idx + 1
            
            if duration > 1:  # 忽略单日的水下期
                underwater_periods.append((start_idx, end_idx, duration))
        
        # 处理最后一段
        if len(new_high_indices) > 0 and new_high_indices[-1] < len(values) - 1:
            start_idx = new_high_indices[-1]
            end_idx = len(values) - 1
            duration = end_idx - start_idx + 1
            
            if duration > 1:
                underwater_periods.append((start_idx, end_idx, duration))
        
        return underwater_periods
        
    @staticmethod
    def calculate_trade_stats(trade_history: List[Dict]) -> Dict[str, Union[float, int]]:
        """
        计算交易统计信息
        
        Args:
            trade_history: 交易历史记录列表，每项包含pnl, entry_time, exit_time等信息
            
        Returns:
            Dict: 交易统计信息字典
        """
        if not trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'avg_holding_time': 0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0
            }
            
        # 提取交易盈亏
        pnl_list = [trade.get('pnl', 0) for trade in trade_history]
        
        # 计算基本统计信息
        win_rate, win_count, loss_count = TradingMetrics.calculate_win_rate(pnl_list)
        avg_profit, avg_loss = TradingMetrics.calculate_average_profit_loss(pnl_list)
        profit_factor = TradingMetrics.calculate_profit_factor(pnl_list)
        
        # 计算持仓时间
        holding_times = []
        for trade in trade_history:
            entry_time = trade.get('entry_time')
            exit_time = trade.get('exit_time')
            
            if entry_time and exit_time:
                try:
                    # 尝试转换为datetime对象（如果不是的话）
                    if isinstance(entry_time, str):
                        entry_time = pd.to_datetime(entry_time)
                    if isinstance(exit_time, str):
                        exit_time = pd.to_datetime(exit_time)
                        
                    # 计算持仓时间（小时）
                    holding_time = (exit_time - entry_time).total_seconds() / 3600
                    holding_times.append(holding_time)
                except Exception as e:
                    logger.warning(f"计算持仓时间失败: {e}")
        
        avg_holding_time = np.mean(holding_times) if holding_times else 0
        
        # 计算连续盈利和连续亏损
        result_streak = [1 if pnl > 0 else -1 if pnl < 0 else 0 for pnl in pnl_list]
        
        max_win_streak = 0
        max_loss_streak = 0
        current_streak = 0
        
        for result in result_streak:
            if result > 0:
                if current_streak > 0:
                    current_streak += 1
                else:
                    current_streak = 1
                max_win_streak = max(max_win_streak, current_streak)
            elif result < 0:
                if current_streak < 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                max_loss_streak = max(max_loss_streak, abs(current_streak))
            # Ignore zeros (break-even trades)
        
        return {
            'total_trades': len(pnl_list),
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'avg_holding_time': avg_holding_time,  # 小时
            'max_consecutive_wins': max_win_streak,
            'max_consecutive_losses': max_loss_streak
        }
        
    @staticmethod
    def calculate_all_metrics(returns: np.ndarray, values: np.ndarray, trade_history: List[Dict] = None, 
                            benchmark_returns: np.ndarray = None, risk_free_rate: float = 0) -> Dict:
        """
        计算所有交易绩效指标
        
        Args:
            returns: 日收益率序列
            values: 资产价值序列
            trade_history: 交易历史记录
            benchmark_returns: 基准收益率序列
            risk_free_rate: 无风险收益率
            
        Returns:
            Dict: 所有指标的字典
        """
        metrics = {}
        
        try:
            # 计算回报相关指标
            if len(returns) >= 2:
                metrics['annualized_return'] = np.mean(returns) * 252
                metrics['volatility'] = TradingMetrics.calculate_volatility(returns)
                metrics['sharpe_ratio'] = TradingMetrics.calculate_sharpe_ratio(returns, risk_free_rate)
                metrics['sortino_ratio'] = TradingMetrics.calculate_sortino_ratio(returns, risk_free_rate)
                
                # 计算VaR和ES
                metrics['var_95'] = np.percentile(returns, 5)  # 95% VaR
                metrics['expected_shortfall_95'] = TradingMetrics.calculate_expected_shortfall(returns)
            else:
                metrics['annualized_return'] = 0
                metrics['volatility'] = 0
                metrics['sharpe_ratio'] = 0
                metrics['sortino_ratio'] = 0
                metrics['var_95'] = 0
                metrics['expected_shortfall_95'] = 0
                
            # 计算回撤相关指标
            if len(values) >= 2:
                max_dd, max_dd_start, max_dd_end = TradingMetrics.calculate_max_drawdown(values)
                metrics['max_drawdown'] = max_dd
                metrics['max_drawdown_start'] = max_dd_start
                metrics['max_drawdown_end'] = max_dd_end
                
                if max_dd > 0:
                    # 计算从开始到最近的总收益
                    total_return = (values[-1] / values[0]) - 1
                    metrics['calmar_ratio'] = TradingMetrics.calculate_calmar_ratio(returns, max_dd)
                    metrics['recovery_factor'] = TradingMetrics.calculate_recovery_factor(total_return, max_dd)
                else:
                    metrics['calmar_ratio'] = float('inf')
                    metrics['recovery_factor'] = float('inf')
                    
                # 计算水下期
                underwater_periods = TradingMetrics.calculate_underwater_periods(values)
                if underwater_periods:
                    metrics['underwater_periods'] = len(underwater_periods)
                    metrics['max_underwater_duration'] = max(period[2] for period in underwater_periods)
                else:
                    metrics['underwater_periods'] = 0
                    metrics['max_underwater_duration'] = 0
            else:
                metrics['max_drawdown'] = 0
                metrics['max_drawdown_start'] = 0
                metrics['max_drawdown_end'] = 0
                metrics['calmar_ratio'] = 0
                metrics['recovery_factor'] = 0
                metrics['underwater_periods'] = 0
                metrics['max_underwater_duration'] = 0
                
            # 计算基准比较指标
            if benchmark_returns is not None and len(benchmark_returns) == len(returns) and len(returns) >= 2:
                metrics['information_ratio'] = TradingMetrics.calculate_information_ratio(returns, benchmark_returns)
                metrics['r_squared'] = TradingMetrics.calculate_r_squared(returns, benchmark_returns)
                alpha, beta = TradingMetrics.calculate_alpha_beta(returns, benchmark_returns, risk_free_rate)
                metrics['alpha'] = alpha
                metrics['beta'] = beta
            else:
                metrics['information_ratio'] = 0
                metrics['r_squared'] = 0
                metrics['alpha'] = 0
                metrics['beta'] = 0
                
            # 计算交易指标
            if trade_history:
                trade_stats = TradingMetrics.calculate_trade_stats(trade_history)
                metrics.update(trade_stats)
            else:
                # 添加默认的交易指标
                metrics.update({
                    'total_trades': 0,
                    'win_count': 0,
                    'loss_count': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'avg_profit': 0,
                    'avg_loss': 0
                })
                
        except Exception as e:
            logger.error(f"计算绩效指标失败: {e}")
            # 返回空指标字典
            metrics = {
                'annualized_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'calmar_ratio': 0,
                'win_rate': 0,
                'profit_factor': 0
            }
            
        return metrics


class ModelMetrics:
    """
    模型性能指标计算工具类
    用于计算各种机器学习模型的性能指标
    """
    
    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算回归模型的性能指标
        
        Args:
            y_true: 实际值
            y_pred: 预测值
            
        Returns:
            Dict: 性能指标字典
        """
        try:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            # 计算均方误差(MSE)
            mse = mean_squared_error(y_true, y_pred)
            
            # 计算均方根误差(RMSE)
            rmse = np.sqrt(mse)
            
            # 计算平均绝对误差(MAE)
            mae = mean_absolute_error(y_true, y_pred)
            
            # 计算决定系数(R²)
            r2 = r2_score(y_true, y_pred)
            
            # 计算平均绝对百分比误差(MAPE)
            mask = y_true != 0
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) if np.any(mask) else np.inf
            
            # 计算方向准确率(DA)
            if len(y_true) > 1 and len(y_pred) > 1:
                y_true_diff = np.diff(y_true)
                y_pred_diff = np.diff(y_pred)
                da = np.mean((y_true_diff > 0) == (y_pred_diff > 0))
            else:
                da = 0.5
                
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape,
                'direction_accuracy': da
            }
            
        except ImportError:
            logger.warning("scikit-learn未安装，无法计算回归指标")
            return {
                'mse': np.mean((y_true - y_pred) ** 2),
                'rmse': np.sqrt(np.mean((y_true - y_pred) ** 2)),
                'mae': np.mean(np.abs(y_true - y_pred)),
                'r2': 0,
                'mape': 0,
                'direction_accuracy': 0
            }
        except Exception as e:
            logger.error(f"计算回归指标失败: {e}")
            return {
                'mse': 0,
                'rmse': 0,
                'mae': 0,
                'r2': 0,
                'mape': 0,
                'direction_accuracy': 0
            }
            
    @staticmethod
    def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, float]:
        """
        计算分类模型的性能指标
        
        Args:
            y_true: 实际标签
            y_pred: 预测标签
            y_prob: 预测概率（可选）
            
        Returns:
            Dict: 性能指标字典
        """
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
            
            # 计算准确率
            accuracy = accuracy_score(y_true, y_pred)
            
            # 计算精确率、召回率和F1分数
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # 计算混淆矩阵
            cm = confusion_matrix(y_true, y_pred)
            
            # 计算ROC AUC（如果提供了概率）
            roc_auc = 0
            if y_prob is not None:
                try:
                    # 对于二分类
                    if y_prob.shape[1] == 2:
                        roc_auc = roc_auc_score(y_true, y_prob[:, 1])
                    # 对于多分类
                    else:
                        roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
                except (ValueError, IndexError):
                    # 如果y_prob是一维数组
                    try:
                        roc_auc = roc_auc_score(y_true, y_prob)
                    except:
                        roc_auc = 0
                        
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm.tolist(),
                'roc_auc': roc_auc
            }
            
        except ImportError:
            logger.warning("scikit-learn未安装，无法计算分类指标")
            return {
                'accuracy': np.mean(y_true == y_pred),
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'confusion_matrix': [],
                'roc_auc': 0
            }
        except Exception as e:
            logger.error(f"计算分类指标失败: {e}")
            return {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'confusion_matrix': [],
                'roc_auc': 0
            }
            
    @staticmethod
    def calculate_rl_metrics(rewards: np.ndarray, episode_lengths: np.ndarray = None) -> Dict[str, float]:
        """
        计算强化学习模型的性能指标
        
        Args:
            rewards: 每个episode的累积奖励
            episode_lengths: 每个episode的长度（步数）
            
        Returns:
            Dict: 性能指标字典
        """
        try:
            metrics = {}
            
            # 计算平均奖励
            metrics['mean_reward'] = np.mean(rewards)
            metrics['median_reward'] = np.median(rewards)
            metrics['std_reward'] = np.std(rewards)
            metrics['min_reward'] = np.min(rewards)
            metrics['max_reward'] = np.max(rewards)
            
            # 如果提供了episode长度，计算相关指标
            if episode_lengths is not None:
                metrics['mean_episode_length'] = np.mean(episode_lengths)
                metrics['median_episode_length'] = np.median(episode_lengths)
                metrics['std_episode_length'] = np.std(episode_lengths)
                
                # 计算每步平均奖励
                reward_per_step = rewards / episode_lengths
                metrics['mean_reward_per_step'] = np.mean(reward_per_step)
                
            # 计算最近n个episode的平均奖励（如果有足够多的数据）
            if len(rewards) >= 10:
                metrics['last_10_mean_reward'] = np.mean(rewards[-10:])
            if len(rewards) >= 100:
                metrics['last_100_mean_reward'] = np.mean(rewards[-100:])
                
            return metrics
            
        except Exception as e:
            logger.error(f"计算强化学习指标失败: {e}")
            return {
                'mean_reward': 0,
                'median_reward': 0,
                'std_reward': 0,
                'min_reward': 0,
                'max_reward': 0
            }
