#!/usr/bin/env python3
"""
Quant-Investor V4.0 统一主流水线 (Master Pipeline)

这是quant-investor技能的核心入口，整合V2.3-V3.6所有能力，
提供标准化的端到端投资分析流程。

流程：
1. 数据获取 (Data Acquisition)
2. 因子挖掘与选股 (Factor Mining & Stock Selection)
3. 定性分析与估值 (Qualitative Analysis & Valuation)
4. 风险评估与控制 (Risk Assessment & Control)
5. 生成投资建议 (Investment Recommendation)
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# 添加模块路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# 导入各模块
from data_provider import DataProvider, MarketConfig
from quant_selector import QuantSelector
from qualitative_analyzer import QualitativeAnalyzer, QualitativeReport
from risk_assessor import RiskAssessor, PortfolioRiskReport


@dataclass
class InvestmentRecommendation:
    """投资建议"""
    stock_code: str
    stock_name: str
    action: str  # 买入/持有/卖出
    target_weight: float  # 目标权重
    target_price: float  # 目标价格
    stop_loss: float  # 止损价格
    rationale: str  # 投资逻辑
    risk_level: str  # 风险等级
    confidence: str  # 置信度


@dataclass
class AnalysisResult:
    """完整分析结果"""
    # 元信息
    market: str
    analysis_date: str
    
    # 定量分析结果
    effective_factors: List[str] = field(default_factory=list)
    recommended_stocks: List[Dict] = field(default_factory=list)
    holding_analysis: List[Dict] = field(default_factory=list)
    
    # 定性分析结果
    qualitative_reports: List[QualitativeReport] = field(default_factory=list)
    
    # 风险评估结果
    risk_report: PortfolioRiskReport = None
    
    # 最终投资建议
    recommendations: List[InvestmentRecommendation] = field(default_factory=list)
    
    # 综合报告
    full_report: str = ""


class MasterPipeline:
    """
    Quant-Investor V4.0 统一主流水线
    
    整合所有版本能力，提供一站式投资分析服务。
    """
    
    def __init__(self, market: str = "US", llm_provider: str = "auto", verbose: bool = True):
        """
        初始化主流水线
        
        Args:
            market: 市场类型 (US/CN)
            llm_provider: LLM提供商 (auto/openai/gemini/deepseek/qwen/kimi)
            verbose: 是否打印详细信息
        """
        self.market = market.upper()
        self.llm_provider = llm_provider
        self.verbose = verbose
        
        # 加载API密钥
        self._load_credentials()
        
        # 初始化各模块
        self.data_provider = DataProvider(market=self.market, verbose=verbose)
        self.quant_selector = QuantSelector(verbose=verbose)
        self.qualitative_analyzer = QualitativeAnalyzer(llm_provider=llm_provider, verbose=verbose)
        self.risk_assessor = RiskAssessor(verbose=verbose)
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"🚀 Quant-Investor V4.0 统一主流水线")
            print(f"{'='*70}")
            print(f"   市场: {self.market}")
            print(f"   LLM: {llm_provider}")
            print(f"{'='*70}\n")
    
    def _load_credentials(self):
        """加载API密钥"""
        credentials_path = os.path.expanduser("~/.quant_investor/credentials.env")
        if os.path.exists(credentials_path):
            with open(credentials_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
    
    def run_full_analysis(self, holdings: List[Dict] = None,
                          num_recommendations: int = 5) -> AnalysisResult:
        """
        运行完整的投资分析流程
        
        Args:
            holdings: 当前持仓 [{"code": "AAPL", "name": "苹果", "weight": 0.2}, ...]
            num_recommendations: 推荐股票数量
        
        Returns:
            完整分析结果
        """
        result = AnalysisResult(
            market=self.market,
            analysis_date=datetime.now().strftime("%Y-%m-%d")
        )
        
        # ========== 第一阶段：数据获取 ==========
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"📊 第一阶段：数据获取")
            print(f"{'='*60}")
        
        # 获取市场数据
        market_data = self.data_provider.get_market_data()
        
        # 获取成分股数据
        stock_data = self.data_provider.get_constituent_stocks()
        
        # 获取宏观经济数据
        macro_data = self.data_provider.get_macro_data()
        
        # 获取行业数据
        industry_data = self.data_provider.get_industry_data()
        
        if self.verbose:
            print(f"\n   ✅ 数据获取完成")
            print(f"      - 股票数据: {len(stock_data)} 只")
            print(f"      - 宏观指标: {len(macro_data)} 个")
            print(f"      - 行业数据: {len(industry_data)} 个行业")
        
        # ========== 第二阶段：因子挖掘与选股 ==========
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🔬 第二阶段：因子挖掘与选股")
            print(f"{'='*60}")
        
        # 挖掘有效因子
        effective_factors = self.quant_selector.mine_factors(stock_data, macro_data)
        result.effective_factors = effective_factors
        
        # 基于因子选股
        recommended_stocks = self.quant_selector.select_stocks(
            stock_data, 
            effective_factors,
            top_n=num_recommendations
        )
        result.recommended_stocks = recommended_stocks
        
        # 分析持仓股票
        if holdings:
            holding_analysis = self.quant_selector.analyze_holdings(
                holdings, stock_data, effective_factors
            )
            result.holding_analysis = holding_analysis
        
        if self.verbose:
            print(f"\n   ✅ 因子挖掘与选股完成")
            print(f"      - 有效因子: {len(effective_factors)} 个")
            print(f"      - 推荐股票: {len(recommended_stocks)} 只")
        
        # ========== 第三阶段：定性分析与估值 ==========
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🎯 第三阶段：定性分析与估值")
            print(f"{'='*60}")
        
        # 合并待分析股票列表
        stocks_to_analyze = []
        
        # 添加推荐股票
        for stock in recommended_stocks:
            stocks_to_analyze.append({
                "code": stock.get("code", ""),
                "name": stock.get("name", ""),
                "data": stock_data.get(stock.get("code", ""))
            })
        
        # 添加持仓股票
        if holdings:
            for holding in holdings:
                code = holding.get("code", "")
                if code not in [s["code"] for s in stocks_to_analyze]:
                    stocks_to_analyze.append({
                        "code": code,
                        "name": holding.get("name", code),
                        "data": stock_data.get(code)
                    })
        
        # 进行定性分析
        qualitative_reports = self.qualitative_analyzer.analyze_multiple(stocks_to_analyze)
        result.qualitative_reports = qualitative_reports
        
        if self.verbose:
            print(f"\n   ✅ 定性分析完成")
            print(f"      - 分析股票: {len(qualitative_reports)} 只")
        
        # ========== 第四阶段：风险评估与控制 ==========
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"⚠️ 第四阶段：风险评估与控制")
            print(f"{'='*60}")
        
        # 构建待评估组合
        portfolio_for_risk = []
        
        # 添加持仓
        if holdings:
            for holding in holdings:
                portfolio_for_risk.append({
                    "code": holding.get("code", ""),
                    "name": holding.get("name", ""),
                    "weight": holding.get("weight", 0.1)
                })
        
        # 添加推荐股票（假设等权重）
        rec_weight = 0.1 / len(recommended_stocks) if recommended_stocks else 0
        for stock in recommended_stocks:
            code = stock.get("code", "")
            if code not in [p["code"] for p in portfolio_for_risk]:
                portfolio_for_risk.append({
                    "code": code,
                    "name": stock.get("name", ""),
                    "weight": rec_weight
                })
        
        # 进行风险评估
        risk_report = self.risk_assessor.assess_portfolio(portfolio_for_risk, stock_data)
        result.risk_report = risk_report
        
        if self.verbose:
            print(f"\n   ✅ 风险评估完成")
            print(f"      - 组合风险等级: {risk_report.risk_level}")
            print(f"      - 风险预警: {len(risk_report.alerts)} 个")
        
        # ========== 第五阶段：生成投资建议 ==========
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"💡 第五阶段：生成投资建议")
            print(f"{'='*60}")
        
        # 综合所有分析结果生成投资建议
        recommendations = self._generate_recommendations(
            recommended_stocks,
            holdings,
            qualitative_reports,
            risk_report
        )
        result.recommendations = recommendations
        
        # 生成完整报告
        result.full_report = self._generate_full_report(result)
        
        if self.verbose:
            print(f"\n   ✅ 投资建议生成完成")
            print(f"      - 建议数量: {len(recommendations)} 条")
        
        return result
    
    def _generate_recommendations(self, recommended_stocks: List[Dict],
                                   holdings: List[Dict],
                                   qualitative_reports: List[QualitativeReport],
                                   risk_report: PortfolioRiskReport) -> List[InvestmentRecommendation]:
        """生成投资建议"""
        recommendations = []
        
        # 创建定性报告索引
        qual_index = {r.stock_code: r for r in qualitative_reports}
        
        # 创建风险评估索引
        risk_index = {p.stock_code: p for p in risk_report.position_risks}
        
        # 为推荐股票生成建议
        for stock in recommended_stocks:
            code = stock.get("code", "")
            name = stock.get("name", code)
            
            qual = qual_index.get(code)
            risk = risk_index.get(code)
            
            # 确定行动
            action = "买入"
            if qual and qual.investment_rating in ["卖出"]:
                action = "观望"
            elif risk and risk.risk_level == "高":
                action = "谨慎买入"
            
            # 确定目标权重
            target_weight = 0.05  # 默认5%
            if risk and risk.risk_level == "低":
                target_weight = 0.08
            elif risk and risk.risk_level == "高":
                target_weight = 0.03
            
            recommendations.append(InvestmentRecommendation(
                stock_code=code,
                stock_name=name,
                action=action,
                target_weight=target_weight,
                target_price=stock.get("target_price", 0),
                stop_loss=stock.get("stop_loss", 0),
                rationale=qual.consensus[:200] if qual else stock.get("reason", ""),
                risk_level=risk.risk_level if risk else "中",
                confidence=qual.investment_rating if qual else "持有"
            ))
        
        # 为持仓股票生成建议
        if holdings:
            for holding in holdings:
                code = holding.get("code", "")
                if code in [r.stock_code for r in recommendations]:
                    continue
                
                name = holding.get("name", code)
                qual = qual_index.get(code)
                risk = risk_index.get(code)
                
                # 确定行动
                action = "持有"
                if qual and qual.investment_rating == "卖出":
                    action = "减仓"
                elif qual and qual.investment_rating in ["强烈买入", "买入"]:
                    action = "加仓"
                
                recommendations.append(InvestmentRecommendation(
                    stock_code=code,
                    stock_name=name,
                    action=action,
                    target_weight=holding.get("weight", 0.05),
                    target_price=0,
                    stop_loss=0,
                    rationale=qual.consensus[:200] if qual else "",
                    risk_level=risk.risk_level if risk else "中",
                    confidence=qual.investment_rating if qual else "持有"
                ))
        
        return recommendations
    
    def _generate_full_report(self, result: AnalysisResult) -> str:
        """生成完整的投资分析报告"""
        lines = [
            f"# Quant-Investor 投资分析报告",
            "",
            f"**市场**: {result.market}",
            f"**分析日期**: {result.analysis_date}",
            "",
            "---",
            "",
            "## 执行摘要",
            "",
            f"本报告基于Quant-Investor V4.0统一分析流程，对{result.market}市场进行了全面的定量和定性分析。",
            "",
            f"- **有效因子**: 发现 {len(result.effective_factors)} 个当前有效的量化因子",
            f"- **推荐股票**: 筛选出 {len(result.recommended_stocks)} 只具有投资价值的股票",
            f"- **风险等级**: 组合整体风险等级为 **{result.risk_report.risk_level if result.risk_report else '未评估'}**",
            "",
            "---",
            "",
            "## 一、定量分析",
            "",
            "### 1.1 有效因子",
            ""
        ]
        
        # 有效因子
        if result.effective_factors:
            lines.append("| 因子名称 | IC均值 | IR | 有效性 |")
            lines.append("|:---|:---|:---|:---|")
            for factor in result.effective_factors[:10]:
                if isinstance(factor, dict):
                    lines.append(f"| {factor.get('name', '')} | {factor.get('ic', 0):.3f} | {factor.get('ir', 0):.2f} | {factor.get('validity', '')} |")
                else:
                    lines.append(f"| {factor} | - | - | 有效 |")
            lines.append("")
        
        # 推荐股票
        lines.extend([
            "### 1.2 推荐股票",
            "",
            "| 股票代码 | 股票名称 | 因子得分 | 推荐理由 |",
            "|:---|:---|:---|:---|"
        ])
        
        for stock in result.recommended_stocks:
            lines.append(f"| {stock.get('code', '')} | {stock.get('name', '')} | {stock.get('score', 0):.2f} | {stock.get('reason', '')[:30]}... |")
        
        lines.extend([
            "",
            "---",
            "",
            "## 二、定性分析",
            ""
        ])
        
        # 定性分析摘要
        for report in result.qualitative_reports[:5]:
            lines.extend([
                f"### {report.stock_code} ({report.stock_name})",
                "",
                f"**投资评级**: {report.investment_rating}",
                "",
                f"**商业模式**: {report.business_model[:150]}...",
                "",
                f"**护城河**: {report.moat_analysis[:150]}...",
                "",
                f"**多方观点**: {report.bull_case[:100]}...",
                "",
                f"**空方观点**: {report.bear_case[:100]}...",
                "",
                "---",
                ""
            ])
        
        # 风险评估
        lines.extend([
            "## 三、风险评估",
            ""
        ])
        
        if result.risk_report:
            lines.append(result.risk_report.summary)
        
        # 投资建议
        lines.extend([
            "",
            "---",
            "",
            "## 四、投资建议",
            "",
            "| 股票 | 行动 | 目标权重 | 风险等级 | 核心逻辑 |",
            "|:---|:---|:---|:---|:---|"
        ])
        
        for rec in result.recommendations:
            lines.append(f"| {rec.stock_code} ({rec.stock_name}) | **{rec.action}** | {rec.target_weight:.1%} | {rec.risk_level} | {rec.rationale[:30]}... |")
        
        lines.extend([
            "",
            "---",
            "",
            "## 五、风控措施",
            ""
        ])
        
        if result.risk_report and result.risk_report.control_measures:
            for i, measure in enumerate(result.risk_report.control_measures, 1):
                lines.append(f"{i}. {measure}")
        
        lines.extend([
            "",
            "---",
            "",
            "*本报告由Quant-Investor V4.0自动生成，仅供参考，不构成投资建议。*"
        ])
        
        return "\n".join(lines)
    
    def save_report(self, result: AnalysisResult, output_dir: str = None) -> str:
        """保存分析报告"""
        if output_dir is None:
            output_dir = os.path.expanduser("~/.quant_investor/reports")
        
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"investment_report_{result.market}_{result.analysis_date}.md"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(result.full_report)
        
        if self.verbose:
            print(f"\n📄 报告已保存: {filepath}")
        
        return filepath


def run_analysis(market: str = "US", holdings: List[Dict] = None,
                 llm_provider: str = "auto", verbose: bool = True) -> AnalysisResult:
    """
    快速运行完整分析的便捷函数
    
    Args:
        market: 市场类型 (US/CN)
        holdings: 当前持仓
        llm_provider: LLM提供商
        verbose: 是否打印详细信息
    
    Returns:
        完整分析结果
    
    示例:
        # 分析美股市场
        result = run_analysis(market="US")
        
        # 分析A股市场，带持仓
        result = run_analysis(
            market="CN",
            holdings=[
                {"code": "600519", "name": "贵州茅台", "weight": 0.3},
                {"code": "000858", "name": "五粮液", "weight": 0.2}
            ]
        )
    """
    pipeline = MasterPipeline(market=market, llm_provider=llm_provider, verbose=verbose)
    return pipeline.run_full_analysis(holdings=holdings)


if __name__ == "__main__":
    # 测试
    print("=== Quant-Investor V4.0 统一主流水线测试 ===\n")
    
    # 模拟持仓
    holdings = [
        {"code": "AAPL", "name": "苹果", "weight": 0.25},
        {"code": "MSFT", "name": "微软", "weight": 0.25}
    ]
    
    # 运行分析
    result = run_analysis(market="US", holdings=holdings, verbose=True)
    
    # 打印报告摘要
    print("\n" + "="*70)
    print("📋 分析报告摘要")
    print("="*70)
    print(result.full_report[:3000] + "...")
