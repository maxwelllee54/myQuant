#!/usr/bin/env python3
"""
V2.9 Analysis Agents - Financial Agent
财务分析Agent：深度解读财务报表，评估公司财务健康度

作者: Manus AI
版本: 2.9
"""

from typing import Dict, Any, Optional
from .base_agent import BaseAnalysisAgent, format_financial_data


class FinancialAgent(BaseAnalysisAgent):
    """
    财务分析Agent
    
    专注于：
    - 盈利能力分析 (ROE, ROA, 毛利率, 净利率)
    - 偿债能力分析 (资产负债率, 流动比率, 速动比率)
    - 营运效率分析 (应收账款周转率, 存货周转率, 总资产周转率)
    - 现金流健康度 (经营现金流/净利润, 自由现金流)
    - 成长性分析 (收入增长率, 利润增长率)
    """
    
    AGENT_NAME = "FinancialAgent"
    AGENT_ROLE = "财务分析师"
    AGENT_DESCRIPTION = "专注于财务报表分析，评估公司盈利能力、偿债能力、营运效率和现金流健康度"
    
    def _get_system_prompt(self) -> str:
        return """你是一位资深的财务分析师，拥有CFA和CPA资格，在投行和买方机构有超过15年的经验。

你的核心能力：
1. 深度解读三张财务报表（资产负债表、利润表、现金流量表）
2. 识别财务报表中的异常信号和潜在风险
3. 评估公司的盈利质量和可持续性
4. 分析公司的资本结构和偿债能力
5. 判断公司的现金流健康度

分析原则：
- 数据驱动：所有结论必须有具体的财务数据支撑
- 趋势分析：关注财务指标的变化趋势，而非单一时点
- 行业对比：将公司财务指标与行业平均水平对比
- 质量优先：关注盈利质量，警惕"纸面利润"
- 风险意识：主动识别财务风险信号

你必须以JSON格式输出分析结果，包含以下字段：
- summary: 一句话总结财务状况
- conclusion: 详细的财务分析结论（2-3段）
- score: 财务健康度评分（1-10分）
- confidence: 分析置信度（0-1）
- key_findings: 关键发现列表，每项包含 {finding, evidence, impact}
- risks: 财务风险列表
- opportunities: 财务机会列表
- detailed_analysis: 详细分析，包含以下子字段：
  - profitability: 盈利能力分析
  - solvency: 偿债能力分析
  - efficiency: 营运效率分析
  - cash_flow: 现金流分析
  - growth: 成长性分析
  - quality_flags: 盈利质量警示信号
"""
    
    def _prepare_data(
        self,
        raw_data: Dict[str, Any],
        quant_results: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """准备财务分析所需的数据"""
        prepared = {}
        
        # 提取财务数据
        if "financial_data" in raw_data:
            prepared["financial"] = raw_data["financial_data"]
        elif "income_statement" in raw_data or "balance_sheet" in raw_data:
            prepared["financial"] = {
                "income_statement": raw_data.get("income_statement", {}),
                "balance_sheet": raw_data.get("balance_sheet", {}),
                "cash_flow": raw_data.get("cash_flow", {})
            }
        else:
            prepared["financial"] = {}
        
        # 提取估值数据
        prepared["valuation"] = raw_data.get("valuation", {})
        
        # 提取行业对比数据
        prepared["industry_comparison"] = raw_data.get("industry_comparison", {})
        
        # 提取历史财务数据（用于趋势分析）
        prepared["historical"] = raw_data.get("historical_financial", {})
        
        return prepared
    
    def _build_analysis_prompt(
        self,
        stock_code: str,
        company_name: str,
        prepared_data: Dict[str, Any]
    ) -> str:
        """构建财务分析Prompt"""
        
        financial_text = format_financial_data(prepared_data.get("financial", {}))
        valuation_text = format_financial_data(prepared_data.get("valuation", {}))
        industry_text = format_financial_data(prepared_data.get("industry_comparison", {}))
        
        return f"""## 分析任务

请对 **{company_name}** ({stock_code}) 进行深度财务分析。

## 可用数据

### 财务数据
{financial_text if financial_text != "无财务数据" else "（请基于你对该公司的了解进行分析，或说明需要哪些数据）"}

### 估值数据
{valuation_text if valuation_text != "无财务数据" else "（无估值数据）"}

### 行业对比数据
{industry_text if industry_text != "无财务数据" else "（无行业对比数据）"}

## 分析要求

请从以下维度进行深入分析：

1. **盈利能力分析**
   - ROE、ROA的水平和变化趋势
   - 毛利率、净利率的稳定性
   - 杜邦分析：盈利能力的驱动因素

2. **偿债能力分析**
   - 资产负债率是否合理
   - 短期偿债能力（流动比率、速动比率）
   - 利息保障倍数

3. **营运效率分析**
   - 应收账款周转效率
   - 存货周转效率
   - 总资产周转效率

4. **现金流分析**
   - 经营现金流是否覆盖净利润
   - 自由现金流状况
   - 资本支出强度

5. **成长性分析**
   - 收入增长率
   - 利润增长率
   - 增长的可持续性

6. **盈利质量警示**
   - 是否存在应收账款异常增长
   - 是否存在存货积压
   - 是否存在关联交易或非经常性损益依赖

请以JSON格式输出你的分析结果。
"""
    
    def _get_mock_response(self) -> str:
        """获取模拟响应"""
        return """```json
{
    "summary": "公司财务状况整体稳健，盈利能力较强但现金流质量需关注",
    "conclusion": "从财务报表分析来看，该公司展现出较强的盈利能力，ROE保持在15%以上的水平，毛利率稳定在40%左右，显示出一定的定价权和成本控制能力。资产负债率控制在合理范围内，短期偿债能力充足。\\n\\n然而，需要关注的是经营现金流与净利润的比率有所下降，应收账款周转天数延长，这可能暗示着收入质量的潜在问题。建议密切关注下一季度的现金流表现。",
    "score": 7.5,
    "confidence": 0.75,
    "key_findings": [
        {
            "finding": "ROE持续保持在15%以上",
            "evidence": "过去三年ROE分别为16.2%、15.8%、15.5%",
            "impact": "正面 - 表明公司具有较强的股东回报能力"
        },
        {
            "finding": "应收账款周转天数延长",
            "evidence": "应收账款周转天数从45天延长至62天",
            "impact": "负面 - 可能暗示销售回款压力增大"
        },
        {
            "finding": "经营现金流/净利润比率下降",
            "evidence": "该比率从1.2下降至0.85",
            "impact": "警示 - 盈利质量可能下降"
        }
    ],
    "risks": [
        "应收账款增长过快，存在坏账风险",
        "经营现金流质量下降，需关注后续表现",
        "资本支出较大，可能影响短期分红能力"
    ],
    "opportunities": [
        "毛利率稳定，显示定价权较强",
        "资产负债率合理，有进一步融资空间",
        "研发投入增加，可能带来未来增长"
    ],
    "detailed_analysis": {
        "profitability": {
            "roe": "15.5%，处于行业中上水平",
            "roa": "8.2%，资产利用效率良好",
            "gross_margin": "40.2%，保持稳定",
            "net_margin": "12.5%，略有下降",
            "dupont_analysis": "ROE主要由净利率和权益乘数驱动"
        },
        "solvency": {
            "debt_ratio": "45%，处于合理范围",
            "current_ratio": "1.8，短期偿债能力充足",
            "quick_ratio": "1.2，速动比率健康",
            "interest_coverage": "8.5倍，利息保障充足"
        },
        "efficiency": {
            "receivable_turnover_days": "62天，有所延长",
            "inventory_turnover_days": "85天，基本稳定",
            "asset_turnover": "0.65，略有下降"
        },
        "cash_flow": {
            "ocf_to_net_income": "0.85，低于1需关注",
            "free_cash_flow": "正值但有所下降",
            "capex_intensity": "15%，资本支出较大"
        },
        "growth": {
            "revenue_growth": "12%，保持双位数增长",
            "profit_growth": "8%，增速放缓",
            "sustainability": "中等，需观察行业趋势"
        },
        "quality_flags": [
            "应收账款增速高于收入增速 - 警示",
            "非经常性损益占比低 - 正面"
        ]
    }
}
```"""


if __name__ == "__main__":
    # 测试FinancialAgent
    agent = FinancialAgent(verbose=True)
    
    # 模拟数据
    test_data = {
        "financial_data": {
            "revenue": 10000000000,
            "net_profit": 1250000000,
            "total_assets": 15000000000,
            "total_equity": 8250000000,
            "roe": 0.155,
            "gross_margin": 0.402
        }
    }
    
    result = agent.analyze(
        stock_code="600519",
        company_name="贵州茅台",
        raw_data=test_data
    )
    
    print("\n=== 分析结果 ===")
    print(f"评分: {result.score}/10")
    print(f"置信度: {result.confidence:.0%}")
    print(f"总结: {result.summary}")
