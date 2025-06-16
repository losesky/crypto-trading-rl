#!/usr/bin/env python3
"""
自适应交易系统集成测试脚本

该脚本测试自适应风险控制器、错误处理器和模型适应性分析工具的功能，
提供一系列模拟测试场景，验证系统在不同情况下的表现。
"""
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional

# 添加项目根目录到路径中
root_path = str(Path(__file__).parent.parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

# 导入系统组件
from trading_system.src.adaptive_risk import AdaptiveRiskController
from trading_system.src.prediction_error_handler import PredictionErrorHandler
from trading_system.src.model_wrapper import ModelWrapper

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("trading_system/logs", f"system_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    ]
)
logger = logging.getLogger("system_test")

class AdaptiveSystemTester:
    """自适应交易系统测试器"""
    
    def __init__(self, config_path: str):
        """
        初始化测试器
        
        参数:
        - config_path: 配置文件路径
        """
        self.logger = logger
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # 初始化测试结果存储
        self.test_results = {
            "risk_control_tests": [],
            "error_handling_tests": [],
            "model_adaptation_tests": []
        }
        
        self.logger.info("自适应系统测试器初始化完成")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.logger.info(f"已加载配置文件: {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            return {}
    
    def test_risk_controller(self) -> None:
        """测试自适应风险控制器功能"""
        self.logger.info("开始测试自适应风险控制器...")
        
        # 创建风险控制器实例
        risk_controller = AdaptiveRiskController(self.config)
        
        # 测试不同市场状态下的风险参数调整
        test_cases = [
            {
                "name": "趋势市场测试",
                "market_data": self._generate_market_data(trend=0.8, volatility=0.02),
                "expected_regime": "trending"
            },
            {
                "name": "高波动市场测试",
                "market_data": self._generate_market_data(trend=0.3, volatility=0.05),
                "expected_regime": "volatile"
            },
            {
                "name": "区间震荡市场测试",
                "market_data": self._generate_market_data(trend=0.2, volatility=0.01),
                "expected_regime": "ranging"
            }
        ]
        
        for test_case in test_cases:
            self.logger.info(f"运行测试: {test_case['name']}")
            
            # 更新市场状态
            risk_controller.update_market_state(test_case["market_data"])
            
            # 获取调整后的风险参数
            risk_params = risk_controller.get_adjusted_risk_parameters()
            
            # 验证市场状态识别
            regime = risk_params.get("market_regime", "")
            expected_regime = test_case.get("expected_regime", "")
            
            success = regime == expected_regime
            
            # 记录测试结果
            result = {
                "test_name": test_case["name"],
                "success": success,
                "expected_regime": expected_regime,
                "actual_regime": regime,
                "risk_params": risk_params
            }
            
            self.test_results["risk_control_tests"].append(result)
            
            if success:
                self.logger.info(f"测试通过: 正确识别为{regime}市场")
            else:
                self.logger.warning(f"测试失败: 预期{expected_regime}，实际{regime}")
            
            self.logger.info(f"调整后的风险参数: risk_per_trade_pct={risk_params['risk_per_trade_pct']:.4f}, "
                          f"max_leverage={risk_params['max_leverage']:.1f}")
            
        self.logger.info(f"风险控制器测试完成，通过{sum(r['success'] for r in self.test_results['risk_control_tests'])}/{len(test_cases)}项测试")
    
    def test_error_handler(self) -> None:
        """测试预测错误处理器功能"""
        self.logger.info("开始测试预测错误处理器...")
        
        # 创建错误处理器实例
        error_handler = PredictionErrorHandler(self.config)
        
        # 测试NaN值处理
        nan_tests = [
            {
                "name": "少量NaN值测试",
                "state": np.array([1.0, 2.0, np.nan, 4.0, 5.0]),
                "expected_fixed": True
            },
            {
                "name": "大量NaN值测试",
                "state": np.array([np.nan, np.nan, 3.0, np.nan, np.nan, 6.0, np.nan, np.nan, np.nan, 10.0]),
                "expected_fixed": True,
                "expected_severe": True
            }
        ]
        
        for test in nan_tests:
            self.logger.info(f"运行测试: {test['name']}")
            
            # 处理NaN值
            fixed_state, info = error_handler.handle_nan_values(test["state"])
            
            # 验证结果
            success = info["fixed"] == test["expected_fixed"]
            if "expected_severe" in test:
                success = success and info.get("severe_issue", False) == test["expected_severe"]
            
            # 检查是否还有NaN值
            if success:
                success = not np.isnan(fixed_state).any()
            
            # 记录测试结果
            result = {
                "test_name": test["name"],
                "success": success,
                "info": info
            }
            
            self.test_results["error_handling_tests"].append(result)
            
            if success:
                self.logger.info(f"测试通过: 正确处理NaN值")
                self.logger.debug(f"原始状态: {test['state']}")
                self.logger.debug(f"修复后: {fixed_state}")
            else:
                self.logger.warning(f"测试失败: NaN处理不正确")
        
        # 测试形状错误处理
        shape_tests = [
            {
                "name": "形状不匹配测试",
                "state": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                "expected_shape": (2, 3),
                "expected_fixed": True
            },
            {
                "name": "维度不匹配测试",
                "state": np.array([[1.0, 2.0], [3.0, 4.0]]),
                "expected_shape": (1, 2, 2),
                "expected_fixed": True
            }
        ]
        
        for test in shape_tests:
            self.logger.info(f"运行测试: {test['name']}")
            
            # 处理形状错误
            fixed_state, info = error_handler.handle_shape_error(test["state"], test["expected_shape"])
            
            # 验证结果
            success = info["fixed"] == test["expected_fixed"] and fixed_state is not None
            if success:
                success = fixed_state.shape == test["expected_shape"]
            
            # 记录测试结果
            result = {
                "test_name": test["name"],
                "success": success,
                "info": info,
                "output_shape": fixed_state.shape if fixed_state is not None else None
            }
            
            self.test_results["error_handling_tests"].append(result)
            
            if success:
                self.logger.info(f"测试通过: 形状已从{test['state'].shape}调整为{fixed_state.shape}")
            else:
                self.logger.warning(f"测试失败: 形状调整不正确")
        
        # 测试回退策略
        fallback_tests = [
            {
                "name": "空持仓回退策略测试",
                "error_type": "预测失败",
                "market_data": {"close": 40000, "open": 39500},
                "position_data": {"size": 0, "side": ""},
                "expected_reasonable": True
            },
            {
                "name": "多头持仓回退策略测试",
                "error_type": "NaN值错误",
                "market_data": {"close": 39000, "open": 40000},
                "position_data": {"size": 0.5, "side": "BUY"},
                "expected_reasonable": True
            },
            {
                "name": "空头持仓回退策略测试",
                "error_type": "形状错误",
                "market_data": {"close": 41000, "open": 40000},
                "position_data": {"size": 0.5, "side": "SELL"},
                "expected_reasonable": True
            }
        ]
        
        for test in fallback_tests:
            self.logger.info(f"运行测试: {test['name']}")
            
            # 生成回退动作
            action, fallback_info = error_handler.generate_fallback_action(
                test["error_type"],
                test["market_data"],
                test["position_data"]
            )
            
            # 检查行为是否合理
            reasonable = self._is_reasonable_fallback(action, test)
            
            # 记录测试结果
            result = {
                "test_name": test["name"],
                "success": reasonable == test["expected_reasonable"],
                "action": float(action),
                "fallback_info": fallback_info
            }
            
            self.test_results["error_handling_tests"].append(result)
            
            if result["success"]:
                self.logger.info(f"测试通过: 回退动作 {action:.4f} 是合理的")
                self.logger.debug(f"回退策略: {fallback_info.get('strategy')}")
            else:
                self.logger.warning(f"测试失败: 回退动作 {action:.4f} 不合理")
        
        passed = sum(r["success"] for r in self.test_results["error_handling_tests"])
        total = len(nan_tests) + len(shape_tests) + len(fallback_tests)
        self.logger.info(f"错误处理器测试完成，通过{passed}/{total}项测试")
    
    def test_model_adaptation(self) -> None:
        """模拟测试模型适应机制"""
        self.logger.info("开始测试模型自适应机制...")
        
        # 模拟不同市场环境下的交易数据
        test_cases = [
            {
                "name": "趋势市场适应测试",
                "trades": self._generate_trades(20, win_rate=0.65, avg_profit=0.03, market_type="trending"),
                "expected_change": "positive"
            },
            {
                "name": "高波动市场适应测试",
                "trades": self._generate_trades(15, win_rate=0.4, avg_profit=0.05, market_type="volatile"),
                "expected_change": "negative"
            }
        ]
        
        for test_case in test_cases:
            self.logger.info(f"运行测试: {test_case['name']}")
            
            # 保存测试交易数据到临时文件
            trades_df = pd.DataFrame(test_case["trades"])
            temp_dir = Path("trading_system/data/test_data")
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            trades_file = temp_dir / "test_trades.json"
            with open(trades_file, 'w') as f:
                json.dump(test_case["trades"], f)
            
            # 运行适应性分析
            from trading_system.scripts.analyze_model_adaptation import analyze_environment_compatibility
            
            # 创建空的经验数据框
            experiences_df = pd.DataFrame()
            
            # 调用分析函数
            analysis_result = analyze_environment_compatibility(trades_df, experiences_df)
            
            # 验证结果
            has_adjustments = len(analysis_result.get("risk_adjustments", {})) > 0 or \
                              len(analysis_result.get("preprocessing_adjustments", {})) > 0
            
            change_direction = "unknown"
            if "risk_per_trade_pct" in analysis_result.get("risk_adjustments", {}):
                change_factor = analysis_result["risk_adjustments"]["risk_per_trade_pct"]
                if change_factor > 1.0:
                    change_direction = "positive"
                elif change_factor < 1.0:
                    change_direction = "negative"
                else:
                    change_direction = "neutral"
            
            success = has_adjustments and change_direction == test_case["expected_change"]
            
            # 记录测试结果
            result = {
                "test_name": test_case["name"],
                "success": success,
                "analysis_result": analysis_result,
                "expected_change": test_case["expected_change"],
                "actual_change": change_direction
            }
            
            self.test_results["model_adaptation_tests"].append(result)
            
            if success:
                self.logger.info(f"测试通过: 正确建议{change_direction}的风险调整")
                self.logger.debug(f"兼容性评分: {analysis_result.get('compatibility_score', 0):.2f}")
                self.logger.debug(f"市场环境: {analysis_result.get('environment_regime', 'unknown')}")
            else:
                self.logger.warning(f"测试失败: 建议调整不符合预期")
                self.logger.warning(f"预期{test_case['expected_change']}的调整，实际为{change_direction}")
        
        # 清理临时文件
        try:
            if 'trades_file' in locals():
                os.remove(trades_file)
            self.logger.debug("已清理临时测试文件")
        except:
            pass
        
        passed = sum(r["success"] for r in self.test_results["model_adaptation_tests"])
        total = len(test_cases)
        self.logger.info(f"模型适应机制测试完成，通过{passed}/{total}项测试")
    
    def test_integrated_system(self) -> None:
        """测试集成系统功能"""
        self.logger.info("开始测试集成系统功能...")
        
        # 创建系统组件
        risk_controller = AdaptiveRiskController(self.config)
        error_handler = PredictionErrorHandler(self.config)
        
        # 测试模型预测-风险控制-错误处理集成流程
        test_cases = [
            {
                "name": "正常预测流程测试",
                "market_data": self._generate_market_data(trend=0.6, volatility=0.02),
                "position_data": {"size": 0, "side": ""},
                "prediction": 0.7,  # 模拟预测结果
                "inject_error": False,
                "expected_action_type": "BUY"
            },
            {
                "name": "错误处理流程测试",
                "market_data": self._generate_market_data(trend=0.3, volatility=0.04),
                "position_data": {"size": 0.5, "side": "BUY"},
                "prediction": None,  # 模拟预测失败
                "inject_error": True,
                "expected_action_type": "HOLD"  # 预期错误处理后的回退动作为持有
            }
        ]
        
        for test_case in test_cases:
            self.logger.info(f"运行集成测试: {test_case['name']}")
            
            # 更新市场状态
            risk_controller.update_market_state(test_case["market_data"])
            
            # 获取风险参数
            risk_params = risk_controller.get_adjusted_risk_parameters()
            
            # 模拟预测过程
            if test_case["inject_error"]:
                # 模拟错误，生成回退动作
                action, fallback_info = error_handler.generate_fallback_action(
                    "模拟错误", 
                    test_case["market_data"],
                    test_case["position_data"]
                )
                used_fallback = True
            else:
                # 使用提供的预测值
                action = test_case["prediction"]
                fallback_info = {"used_fallback": False}
                used_fallback = False
            
            # 应用风险限制
            abs_action = abs(action) if action is not None else 0
            risk_limit = risk_params["risk_per_trade_pct"]
            if abs_action > risk_limit:
                adjusted_action = action * risk_limit / abs_action
            else:
                adjusted_action = action
            
            # 确定动作类型
            if adjusted_action is None:
                action_type = "ERROR"
            elif adjusted_action > 0.05:
                action_type = "BUY"
            elif adjusted_action < -0.05:
                action_type = "SELL"
            else:
                action_type = "HOLD"
            
            # 检查结果
            success = action_type == test_case["expected_action_type"]
            
            self.logger.info(f"市场状态: {risk_params['market_regime']}, 风险限制: {risk_limit:.4f}")
            self.logger.info(f"{'使用回退策略，' if used_fallback else ''}生成动作值: {adjusted_action if adjusted_action is not None else 'None'}")
            self.logger.info(f"动作类型: {action_type}, 预期: {test_case['expected_action_type']}")
            
            if success:
                self.logger.info("测试通过: 动作类型符合预期")
            else:
                self.logger.warning("测试失败: 动作类型不符合预期")
    
    def run_all_tests(self) -> None:
        """运行所有测试"""
        self.logger.info("开始运行自适应系统全面测试...")
        
        # 运行各项功能测试
        self.test_risk_controller()
        self.test_error_handler()
        self.test_model_adaptation()
        self.test_integrated_system()
        
        # 生成测试报告
        self._generate_test_report()
        
        self.logger.info("全面测试完成")
    
    def _generate_test_report(self) -> None:
        """生成测试报告"""
        try:
            # 统计测试结果
            risk_tests = self.test_results["risk_control_tests"]
            error_tests = self.test_results["error_handling_tests"]
            adapt_tests = self.test_results["model_adaptation_tests"]
            
            risk_passed = sum(test["success"] for test in risk_tests)
            error_passed = sum(test["success"] for test in error_tests)
            adapt_passed = sum(test["success"] for test in adapt_tests)
            
            total_tests = len(risk_tests) + len(error_tests) + len(adapt_tests)
            total_passed = risk_passed + error_passed + adapt_passed
            
            # 创建报告目录
            report_dir = Path("trading_system/reports")
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成报告文件名
            report_path = report_dir / f"system_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            
            # 写入报告
            with open(report_path, 'w') as f:
                f.write("# 自适应交易系统测试报告\n\n")
                f.write(f"**测试时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**配置文件:** {self.config_path}\n\n")
                
                f.write("## 测试摘要\n\n")
                f.write(f"**总测试数量:** {total_tests}\n")
                f.write(f"**通过测试:** {total_passed} ({total_passed/total_tests:.1%})\n")
                f.write(f"**失败测试:** {total_tests-total_passed} ({(total_tests-total_passed)/total_tests:.1%})\n\n")
                
                f.write("## 自适应风险控制测试\n\n")
                f.write(f"通过率: {risk_passed}/{len(risk_tests)} ({risk_passed/len(risk_tests):.1%})\n\n")
                for test in risk_tests:
                    f.write(f"### {test['test_name']}\n")
                    f.write(f"**结果:** {'通过' if test['success'] else '失败'}\n")
                    f.write(f"**预期市场状态:** {test['expected_regime']}\n")
                    f.write(f"**实际市场状态:** {test['actual_regime']}\n")
                    f.write(f"**风险参数:** {json.dumps(test['risk_params'], indent=2)}\n\n")
                
                f.write("## 错误处理测试\n\n")
                f.write(f"通过率: {error_passed}/{len(error_tests)} ({error_passed/len(error_tests):.1%})\n\n")
                for test in error_tests:
                    f.write(f"### {test['test_name']}\n")
                    f.write(f"**结果:** {'通过' if test['success'] else '失败'}\n")
                    if "action" in test:
                        f.write(f"**回退动作:** {test['action']}\n")
                        f.write(f"**回退策略:** {test.get('fallback_info', {}).get('strategy', 'unknown')}\n")
                    f.write("\n")
                
                f.write("## 模型适应性测试\n\n")
                f.write(f"通过率: {adapt_passed}/{len(adapt_tests)} ({adapt_passed/len(adapt_tests):.1%})\n\n")
                for test in adapt_tests:
                    f.write(f"### {test['test_name']}\n")
                    f.write(f"**结果:** {'通过' if test['success'] else '失败'}\n")
                    f.write(f"**预期调整方向:** {test['expected_change']}\n")
                    f.write(f"**实际调整方向:** {test['actual_change']}\n")
                    analysis = test.get("analysis_result", {})
                    f.write(f"**兼容性评分:** {analysis.get('compatibility_score', 0):.2f}\n")
                    f.write(f"**市场环境:** {analysis.get('environment_regime', 'unknown')}\n\n")
                    if "suggestions" in analysis:
                        f.write("**建议:**\n")
                        for suggestion in analysis["suggestions"]:
                            f.write(f"- {suggestion}\n")
                    f.write("\n")
            
            self.logger.info(f"测试报告已生成: {report_path}")
        except Exception as e:
            self.logger.error(f"生成测试报告失败: {e}")
    
    def _is_reasonable_fallback(self, action: float, test_case: Dict[str, Any]) -> bool:
        """判断回退动作是否合理"""
        # 空持仓
        if test_case["position_data"].get("size", 0) == 0:
            # 合理的回退应该是轻微交易或持有
            return abs(action) <= 0.3
        
        # 已有多头持仓，价格下跌
        elif (test_case["position_data"].get("side") == "BUY" and
              test_case["market_data"].get("close", 0) < test_case["market_data"].get("open", 0)):
            # 合理的回退应该是减仓或持有
            return action <= 0.1
        
        # 已有空头持仓，价格上涨
        elif (test_case["position_data"].get("side") == "SELL" and
              test_case["market_data"].get("close", 0) > test_case["market_data"].get("open", 0)):
            # 合理的回退应该是减仓或持有
            return action >= -0.1
        
        return True
    
    def _generate_market_data(self, trend: float = 0.5, volatility: float = 0.02) -> Dict[str, Any]:
        """
        生成模拟市场数据
        
        参数:
        - trend: 趋势强度 (0-1)
        - volatility: 波动率
        
        返回:
        - 市场数据字典
        """
        # 基础价格
        base_price = 40000
        
        # 根据趋势和波动率确定价格变动
        if trend > 0.5:  # 上升趋势
            direction = 1
        else:  # 下降趋势
            direction = -1
        
        # 计算OHLC
        change_pct = direction * trend * 0.01 * np.random.normal(1, volatility)
        open_price = base_price
        close_price = base_price * (1 + change_pct)
        high_price = max(open_price, close_price) * (1 + volatility * np.random.random())
        low_price = min(open_price, close_price) * (1 - volatility * np.random.random())
        
        # 交易量 (模拟)
        volume = base_price * 10 * (1 + np.random.random())
        
        return {
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
            "timestamp": int(datetime.now().timestamp() * 1000)
        }
    
    def _generate_trades(self, count: int, win_rate: float = 0.5, avg_profit: float = 0.02, market_type: str = "neutral") -> List[Dict[str, Any]]:
        """
        生成模拟交易数据
        
        参数:
        - count: 交易次数
        - win_rate: 胜率
        - avg_profit: 平均盈利
        - market_type: 市场类型
        
        返回:
        - 交易数据列表
        """
        trades = []
        
        base_time = datetime.now() - timedelta(days=7)
        for i in range(count):
            # 确定交易是否盈利
            is_profitable = np.random.random() < win_rate
            
            # 确定交易方向
            if market_type == "trending":
                # 趋势市场主要是同方向交易
                side = "BUY" if np.random.random() < 0.7 else "SELL"
            else:
                # 其他市场买卖均衡
                side = "BUY" if np.random.random() < 0.5 else "SELL"
            
            # 确定盈亏金额
            if is_profitable:
                profit_pct = avg_profit * np.random.normal(1, 0.3)
            else:
                profit_pct = -avg_profit * np.random.normal(1, 0.3)
            
            # 确定持仓时间
            if market_type == "volatile":
                # 高波动市场持仓时间相对较短
                hours = np.random.randint(1, 24)
            else:
                # 其他市场持仓时间较长
                hours = np.random.randint(6, 72)
            
            # 创建交易记录
            entry_time = base_time + timedelta(hours=i*8)
            exit_time = entry_time + timedelta(hours=hours)
            
            trade = {
                "trade_id": f"test_{i+1}",
                "entry_time": entry_time.isoformat(),
                "exit_time": exit_time.isoformat(),
                "side": side,
                "entry_price": 40000 * (1 + np.random.normal(0, 0.02)),
                "exit_price": 0,  # 后面计算
                "size": np.random.randint(1, 10) / 10,
                "profit_pct": profit_pct,
                "absolute_profit": 0,  # 后面计算
                "is_profitable": is_profitable,
                "market_type": market_type,
                "duration": hours
            }
            
            # 计算退出价格和绝对利润
            if side == "BUY":
                trade["exit_price"] = trade["entry_price"] * (1 + profit_pct)
            else:
                trade["exit_price"] = trade["entry_price"] * (1 - profit_pct)
            
            trade["absolute_profit"] = trade["size"] * trade["entry_price"] * profit_pct
            
            trades.append(trade)
        
        return trades

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="自适应交易系统测试工具")
    parser.add_argument("--config", type=str, default="/home/losesky/crypto-trading-rl/trading_system/config/test_config.json", help="配置文件路径")
    parser.add_argument("--test", type=str, choices=["all", "risk", "error", "adaptation"], default="all", help="要运行的测试类型")
    args = parser.parse_args()
    
    tester = AdaptiveSystemTester(args.config)
    
    if args.test == "all":
        tester.run_all_tests()
    elif args.test == "risk":
        tester.test_risk_controller()
    elif args.test == "error":
        tester.test_error_handler()
    elif args.test == "adaptation":
        tester.test_model_adaptation()
