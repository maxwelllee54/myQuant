#!/usr/bin/env python3
"""
风险评估与控制模块 (Risk Assessor)

负责：
1. 对投资建议进行风险审视
2. 对持仓组合进行风险评估
3. 提出风险控制措施
4. 生成风险报告
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from scipy import stats


@dataclass
class RiskMetrics:
    """风险指标"""
    volatility: float = 0.0  # 年化波动率
    max_drawdown: float = 0.0  # 最大回撤
    var_95: float = 0.0  # 95% VaR
    cvar_95: float = 0.0  # 95% CVaR
    sharpe_ratio: float = 0.0  # 夏普比率
    sortino_ratio: float = 0.0  # 索提诺比率
    beta: float = 0.0  # 贝塔系数
    correlation: float = 0.0  # 与市场相关性


@dataclass
class RiskAlert:
    """风险预警"""
    level: str  # 高/中/低
    category: str  # 市场风险/集中度风险/流动性风险等
    description: str
    suggestion: str


@dataclass
class PositionRisk:
    """个股持仓风险"""
    stock_code: str
    stock_name: str
    weight: float
    risk_metrics: RiskMetrics
    risk_level: str  # 高/中/低
    alerts: List[RiskAlert] = field(default_factory=list)


@dataclass
class PortfolioRiskReport:
    """组合风险报告"""
    # 整体风险指标
    portfolio_metrics: RiskMetrics
    risk_level: str  # 高/中/低
    
    # 个股风险
    position_risks: List[PositionRisk] = field(default_factory=list)
    
    # 风险预警
    alerts: List[RiskAlert] = field(default_factory=list)
    
    # 风险控制建议
    control_measures: List[str] = field(default_factory=list)
    
    # 压力测试结果
    stress_test_results: Dict = field(default_factory=dict)
    
    # 综合评估
    summary: str = ""


class RiskAssessor:
    """
    风险评估器
    
    整合V2.8的风险管理模块，提供全面的风险评估和控制服务。
    """
    
    def __init__(self, market_benchmark: pd.Series = None, 
                 risk_free_rate: float = 0.03, verbose: bool = True):
        """
        初始化风险评估器
        
        Args:
            market_benchmark: 市场基准收益率序列
            risk_free_rate: 无风险利率（年化）
            verbose: 是否打印详细信息
        """
        self.market_benchmark = market_benchmark
        self.risk_free_rate = risk_free_rate
        self.verbose = verbose
    
    def assess_investment_idea(self, stock_code: str, stock_name: str,
                               price_data: pd.DataFrame = None,
                               investment_rating: str = "",
                               qualitative_report: Dict = None) -> PositionRisk:
        """
        评估单个投资建议的风险
        
        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            price_data: 价格数据
            investment_rating: 投资评级
            qualitative_report: 定性分析报告
        
        Returns:
            个股风险评估结果
        """
        if self.verbose:
            print(f"   评估风险: {stock_code} ({stock_name})")
        
        # 计算风险指标
        metrics = self._calculate_risk_metrics(price_data)
        
        # 生成风险预警
        alerts = self._generate_alerts(metrics, investment_rating)
        
        # 确定风险等级
        risk_level = self._determine_risk_level(metrics, alerts)
        
        return PositionRisk(
            stock_code=stock_code,
            stock_name=stock_name,
            weight=0.0,  # 单股评估时权重为0
            risk_metrics=metrics,
            risk_level=risk_level,
            alerts=alerts
        )
    
    def assess_portfolio(self, holdings: List[Dict], 
                         stock_data: Dict = None) -> PortfolioRiskReport:
        """
        评估投资组合风险
        
        Args:
            holdings: 持仓列表 [{"code": "AAPL", "name": "苹果", "weight": 0.2, "data": ...}, ...]
            stock_data: 股票数据字典
        
        Returns:
            组合风险报告
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"⚠️ 风险评估与控制")
            print(f"   持仓数量: {len(holdings)}")
            print(f"{'='*60}\n")
        
        report = PortfolioRiskReport(
            portfolio_metrics=RiskMetrics(),
            risk_level="中"
        )
        
        # 1. 评估每个持仓的风险
        total_weight = sum(h.get("weight", 1/len(holdings)) for h in holdings)
        
        for holding in holdings:
            code = holding.get("code", "")
            name = holding.get("name", code)
            weight = holding.get("weight", 1/len(holdings)) / total_weight
            
            # 获取价格数据
            price_data = None
            if stock_data and code in stock_data:
                if hasattr(stock_data[code], 'price_data'):
                    price_data = stock_data[code].price_data
            
            position_risk = self.assess_investment_idea(code, name, price_data)
            position_risk.weight = weight
            report.position_risks.append(position_risk)
        
        # 2. 计算组合整体风险
        report.portfolio_metrics = self._calculate_portfolio_metrics(report.position_risks)
        
        # 3. 生成组合级别风险预警
        report.alerts = self._generate_portfolio_alerts(report)
        
        # 4. 进行压力测试
        report.stress_test_results = self._perform_stress_test(report.position_risks)
        
        # 5. 生成风险控制建议
        report.control_measures = self._generate_control_measures(report)
        
        # 6. 确定整体风险等级
        report.risk_level = self._determine_portfolio_risk_level(report)
        
        # 7. 生成综合评估
        report.summary = self._generate_summary(report)
        
        return report
    
    def _calculate_risk_metrics(self, price_data: pd.DataFrame = None) -> RiskMetrics:
        """计算风险指标"""
        metrics = RiskMetrics()
        
        if price_data is None or len(price_data) < 20:
            return metrics
        
        try:
            prices = price_data['Close']
            returns = prices.pct_change().dropna()
            
            if len(returns) < 10:
                return metrics
            
            # 年化波动率
            metrics.volatility = returns.std() * np.sqrt(252)
            
            # 最大回撤
            peak = prices.expanding().max()
            drawdown = (prices - peak) / peak
            metrics.max_drawdown = drawdown.min()
            
            # VaR和CVaR
            metrics.var_95 = np.percentile(returns, 5)
            metrics.cvar_95 = returns[returns <= metrics.var_95].mean()
            
            # 夏普比率
            excess_return = returns.mean() * 252 - self.risk_free_rate
            if metrics.volatility > 0:
                metrics.sharpe_ratio = excess_return / metrics.volatility
            
            # 索提诺比率
            neg_returns = returns[returns < 0]
            downside_vol = neg_returns.std() * np.sqrt(252) if len(neg_returns) > 0 else 0
            if downside_vol > 0:
                metrics.sortino_ratio = excess_return / downside_vol
            
            # Beta（如果有基准）
            if self.market_benchmark is not None and len(self.market_benchmark) > 0:
                common_idx = returns.index.intersection(self.market_benchmark.index)
                if len(common_idx) > 10:
                    stock_ret = returns[common_idx]
                    mkt_ret = self.market_benchmark[common_idx]
                    cov = np.cov(stock_ret, mkt_ret)[0, 1]
                    mkt_var = mkt_ret.var()
                    if mkt_var > 0:
                        metrics.beta = cov / mkt_var
                    metrics.correlation = stock_ret.corr(mkt_ret)
        
        except Exception as e:
            if self.verbose:
                print(f"      ⚠️ 风险指标计算异常: {e}")
        
        return metrics
    
    def _generate_alerts(self, metrics: RiskMetrics, 
                         investment_rating: str = "") -> List[RiskAlert]:
        """生成风险预警"""
        alerts = []
        
        # 高波动率预警
        if metrics.volatility > 0.4:
            alerts.append(RiskAlert(
                level="高",
                category="波动率风险",
                description=f"年化波动率达到{metrics.volatility:.1%}，显著高于市场平均水平",
                suggestion="考虑降低仓位或设置止损"
            ))
        elif metrics.volatility > 0.25:
            alerts.append(RiskAlert(
                level="中",
                category="波动率风险",
                description=f"年化波动率{metrics.volatility:.1%}，处于中等水平",
                suggestion="保持关注，做好波动准备"
            ))
        
        # 最大回撤预警
        if metrics.max_drawdown < -0.3:
            alerts.append(RiskAlert(
                level="高",
                category="回撤风险",
                description=f"历史最大回撤达到{metrics.max_drawdown:.1%}",
                suggestion="评估是否能承受类似回撤，设置止损点"
            ))
        
        # 夏普比率预警
        if metrics.sharpe_ratio < 0:
            alerts.append(RiskAlert(
                level="高",
                category="风险收益比",
                description=f"夏普比率为负({metrics.sharpe_ratio:.2f})，风险调整后收益不佳",
                suggestion="重新评估投资价值，考虑替代标的"
            ))
        elif metrics.sharpe_ratio < 0.5:
            alerts.append(RiskAlert(
                level="中",
                category="风险收益比",
                description=f"夏普比率较低({metrics.sharpe_ratio:.2f})",
                suggestion="关注风险收益比改善空间"
            ))
        
        # Beta预警
        if metrics.beta > 1.5:
            alerts.append(RiskAlert(
                level="中",
                category="系统性风险",
                description=f"Beta系数较高({metrics.beta:.2f})，对市场波动敏感",
                suggestion="在市场下跌时可能放大损失"
            ))
        
        return alerts
    
    def _determine_risk_level(self, metrics: RiskMetrics, 
                              alerts: List[RiskAlert]) -> str:
        """确定风险等级"""
        high_alerts = sum(1 for a in alerts if a.level == "高")
        
        if high_alerts >= 2:
            return "高"
        elif high_alerts == 1 or metrics.volatility > 0.3:
            return "中"
        else:
            return "低"
    
    def _calculate_portfolio_metrics(self, position_risks: List[PositionRisk]) -> RiskMetrics:
        """计算组合整体风险指标"""
        if not position_risks:
            return RiskMetrics()
        
        # 加权平均
        total_vol = 0
        total_dd = 0
        total_sharpe = 0
        total_weight = 0
        
        for pos in position_risks:
            w = pos.weight
            total_vol += w * pos.risk_metrics.volatility
            total_dd += w * abs(pos.risk_metrics.max_drawdown)
            total_sharpe += w * pos.risk_metrics.sharpe_ratio
            total_weight += w
        
        if total_weight > 0:
            return RiskMetrics(
                volatility=total_vol / total_weight,
                max_drawdown=-total_dd / total_weight,
                sharpe_ratio=total_sharpe / total_weight
            )
        
        return RiskMetrics()
    
    def _generate_portfolio_alerts(self, report: PortfolioRiskReport) -> List[RiskAlert]:
        """生成组合级别风险预警"""
        alerts = []
        
        # 集中度风险
        if report.position_risks:
            max_weight = max(p.weight for p in report.position_risks)
            if max_weight > 0.3:
                alerts.append(RiskAlert(
                    level="高",
                    category="集中度风险",
                    description=f"单一持仓权重达到{max_weight:.1%}，集中度过高",
                    suggestion="考虑分散投资，降低单一标的权重"
                ))
            
            # 高风险持仓占比
            high_risk_weight = sum(p.weight for p in report.position_risks if p.risk_level == "高")
            if high_risk_weight > 0.5:
                alerts.append(RiskAlert(
                    level="高",
                    category="组合风险",
                    description=f"高风险持仓占比{high_risk_weight:.1%}",
                    suggestion="降低高风险标的配置比例"
                ))
        
        # 整体波动率
        if report.portfolio_metrics.volatility > 0.25:
            alerts.append(RiskAlert(
                level="中",
                category="组合波动",
                description=f"组合整体波动率{report.portfolio_metrics.volatility:.1%}",
                suggestion="考虑增加低波动资产进行对冲"
            ))
        
        return alerts
    
    def _perform_stress_test(self, position_risks: List[PositionRisk]) -> Dict:
        """进行压力测试"""
        results = {
            "scenarios": [],
            "summary": ""
        }
        
        # 定义压力场景
        scenarios = [
            {"name": "市场下跌10%", "shock": -0.10},
            {"name": "市场下跌20%", "shock": -0.20},
            {"name": "市场下跌30%", "shock": -0.30},
            {"name": "波动率翻倍", "vol_multiplier": 2.0}
        ]
        
        for scenario in scenarios:
            impact = 0
            if "shock" in scenario:
                # 根据Beta估算影响
                for pos in position_risks:
                    beta = pos.risk_metrics.beta if pos.risk_metrics.beta != 0 else 1.0
                    impact += pos.weight * scenario["shock"] * beta
            
            results["scenarios"].append({
                "name": scenario["name"],
                "portfolio_impact": impact
            })
        
        # 生成摘要
        worst_case = min(s["portfolio_impact"] for s in results["scenarios"])
        results["summary"] = f"在最坏情景下，组合预计损失{abs(worst_case):.1%}"
        
        return results
    
    def _generate_control_measures(self, report: PortfolioRiskReport) -> List[str]:
        """生成风险控制建议"""
        measures = []
        
        # 基于风险预警生成建议
        for alert in report.alerts:
            if alert.level == "高":
                measures.append(f"【紧急】{alert.suggestion}")
        
        # 通用建议
        if report.portfolio_metrics.volatility > 0.2:
            measures.append("建议设置组合整体止损线，如最大回撤15%时减仓")
        
        if any(p.weight > 0.25 for p in report.position_risks):
            measures.append("建议单一持仓不超过组合的25%，降低集中度风险")
        
        measures.append("建议定期（每月）重新评估持仓风险，动态调整")
        measures.append("建议保留10-20%现金仓位，应对市场波动")
        
        return measures
    
    def _determine_portfolio_risk_level(self, report: PortfolioRiskReport) -> str:
        """确定组合整体风险等级"""
        high_alerts = sum(1 for a in report.alerts if a.level == "高")
        high_positions = sum(1 for p in report.position_risks if p.risk_level == "高")
        
        if high_alerts >= 2 or high_positions >= len(report.position_risks) / 2:
            return "高"
        elif high_alerts >= 1 or report.portfolio_metrics.volatility > 0.25:
            return "中"
        else:
            return "低"
    
    def _generate_summary(self, report: PortfolioRiskReport) -> str:
        """生成综合评估摘要"""
        lines = [
            f"## 风险评估总结",
            "",
            f"**整体风险等级**: {report.risk_level}",
            "",
            f"### 核心指标",
            f"- 组合波动率: {report.portfolio_metrics.volatility:.1%}",
            f"- 预期最大回撤: {report.portfolio_metrics.max_drawdown:.1%}",
            f"- 夏普比率: {report.portfolio_metrics.sharpe_ratio:.2f}",
            ""
        ]
        
        if report.alerts:
            lines.append("### 风险预警")
            for alert in report.alerts:
                lines.append(f"- [{alert.level}] {alert.category}: {alert.description}")
            lines.append("")
        
        if report.control_measures:
            lines.append("### 风控建议")
            for i, measure in enumerate(report.control_measures, 1):
                lines.append(f"{i}. {measure}")
        
        return "\n".join(lines)
    
    def generate_report_markdown(self, report: PortfolioRiskReport) -> str:
        """生成完整的Markdown风险报告"""
        lines = [
            "# 投资组合风险评估报告",
            "",
            f"**评估日期**: {pd.Timestamp.now().strftime('%Y-%m-%d')}",
            f"**整体风险等级**: **{report.risk_level}**",
            "",
            "---",
            "",
            "## 1. 组合概览",
            "",
            "| 持仓 | 权重 | 风险等级 | 波动率 | 最大回撤 |",
            "|:---|:---|:---|:---|:---|"
        ]
        
        for pos in report.position_risks:
            lines.append(
                f"| {pos.stock_code} ({pos.stock_name}) | {pos.weight:.1%} | {pos.risk_level} | "
                f"{pos.risk_metrics.volatility:.1%} | {pos.risk_metrics.max_drawdown:.1%} |"
            )
        
        lines.extend([
            "",
            "## 2. 风险指标",
            "",
            "| 指标 | 数值 | 说明 |",
            "|:---|:---|:---|",
            f"| 组合波动率 | {report.portfolio_metrics.volatility:.1%} | 年化标准差 |",
            f"| 预期最大回撤 | {report.portfolio_metrics.max_drawdown:.1%} | 历史最大回撤 |",
            f"| 夏普比率 | {report.portfolio_metrics.sharpe_ratio:.2f} | 风险调整后收益 |",
            ""
        ])
        
        if report.alerts:
            lines.extend([
                "## 3. 风险预警",
                ""
            ])
            for alert in report.alerts:
                emoji = "🔴" if alert.level == "高" else "🟡" if alert.level == "中" else "🟢"
                lines.append(f"### {emoji} {alert.category}")
                lines.append(f"**等级**: {alert.level}")
                lines.append(f"**描述**: {alert.description}")
                lines.append(f"**建议**: {alert.suggestion}")
                lines.append("")
        
        if report.stress_test_results.get("scenarios"):
            lines.extend([
                "## 4. 压力测试",
                "",
                "| 情景 | 预计影响 |",
                "|:---|:---|"
            ])
            for scenario in report.stress_test_results["scenarios"]:
                lines.append(f"| {scenario['name']} | {scenario['portfolio_impact']:.1%} |")
            lines.append("")
        
        if report.control_measures:
            lines.extend([
                "## 5. 风控建议",
                ""
            ])
            for i, measure in enumerate(report.control_measures, 1):
                lines.append(f"{i}. {measure}")
        
        return "\n".join(lines)


if __name__ == "__main__":
    # 测试
    print("=== 测试风险评估模块 ===\n")
    
    assessor = RiskAssessor(verbose=True)
    
    # 模拟持仓
    holdings = [
        {"code": "AAPL", "name": "苹果", "weight": 0.3},
        {"code": "MSFT", "name": "微软", "weight": 0.25},
        {"code": "GOOGL", "name": "谷歌", "weight": 0.25},
        {"code": "AMZN", "name": "亚马逊", "weight": 0.2}
    ]
    
    report = assessor.assess_portfolio(holdings)
    
    # 生成报告
    markdown = assessor.generate_report_markdown(report)
    print("\n" + markdown)
