#!/usr/bin/env python3
"""
V2.9 Analysis Agents - Valuation Agent
估值分析Agent：运用多种估值方法，评估公司价值和成长性

作者: Manus AI
版本: 2.9
"""

from typing import Dict, Any, Optional
from .base_agent import BaseAnalysisAgent, format_financial_data


class ValuationAgent(BaseAnalysisAgent):
    """
    估值分析Agent
    
    专注于：
    - 相对估值法 (PE, PB, PS, EV/EBITDA)
    - 绝对估值法 (DCF, DDM)
    - 成长性评估
    - 估值历史分位数分析
    - 安全边际判断
    """
    
    AGENT_NAME = "ValuationAgent"
    AGENT_ROLE = "估值分析师"
    AGENT_DESCRIPTION = "专注于公司估值分析，运用多种估值方法评估公司内在价值和成长性"
    
    def _get_system_prompt(self) -> str:
        return """你是一位资深的估值分析师，精通各种估值方法，在投行和资产管理公司有丰富的估值经验。

你的核心能力：
1. 熟练运用多种估值方法
2. 判断估值方法的适用性
3. 评估公司的成长性和成长质量
4. 分析估值的历史分位数
5. 判断安全边际

估值方法框架：

1. **相对估值法**
   - PE (市盈率)：适用于盈利稳定的公司
   - PB (市净率)：适用于重资产公司
   - PS (市销率)：适用于高增长但未盈利的公司
   - EV/EBITDA：适用于资本密集型行业
   - PEG：结合增长率的PE估值

2. **绝对估值法**
   - DCF (现金流折现)：理论上最正确的方法
   - DDM (股利折现)：适用于高分红公司
   - NAV (净资产价值)：适用于控股公司或房地产

3. **估值评估维度**
   - 当前估值水平（相对历史、相对行业、相对市场）
   - 估值的合理性（是否匹配基本面）
   - 安全边际（当前价格 vs 内在价值）

估值原则：
- 没有放之四海而皆准的估值方法，需根据公司特点选择
- 估值是艺术而非科学，需要综合判断
- 关注估值的驱动因素，而非单纯的数字
- 安全边际是价值投资的核心

你必须以JSON格式输出分析结果，包含以下字段：
- summary: 一句话总结估值状况
- conclusion: 详细的估值分析结论（2-3段）
- score: 估值吸引力评分（1-10分，10分表示极度低估）
- confidence: 分析置信度（0-1）
- key_findings: 关键发现列表
- risks: 估值风险（高估风险）
- opportunities: 估值机会（低估机会）
- detailed_analysis: 详细分析，包含：
  - relative_valuation: 相对估值分析
  - absolute_valuation: 绝对估值分析（如有数据）
  - growth_assessment: 成长性评估
  - historical_percentile: 历史分位数分析
  - peer_comparison: 同业对比
  - margin_of_safety: 安全边际判断
  - fair_value_range: 合理估值区间
"""
    
    def _prepare_data(
        self,
        raw_data: Dict[str, Any],
        quant_results: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """准备估值分析所需的数据"""
        prepared = {}
        
        # 提取估值数据
        prepared["valuation"] = raw_data.get("valuation", {})
        
        # 提取财务数据
        prepared["financial"] = raw_data.get("financial_data", {})
        
        # 提取历史估值数据
        prepared["historical_valuation"] = raw_data.get("historical_valuation", {})
        
        # 提取同业估值数据
        prepared["peer_valuation"] = raw_data.get("peer_valuation", {})
        
        # 提取价格数据
        prepared["price_data"] = raw_data.get("price_data", {})
        
        return prepared
    
    def _build_analysis_prompt(
        self,
        stock_code: str,
        company_name: str,
        prepared_data: Dict[str, Any]
    ) -> str:
        """构建估值分析Prompt"""
        
        valuation_text = format_financial_data(prepared_data.get("valuation", {}))
        financial_text = format_financial_data(prepared_data.get("financial", {}))
        
        return f"""## 分析任务

请对 **{company_name}** ({stock_code}) 进行深度估值分析。

## 可用数据

### 估值数据
{valuation_text if valuation_text != "无财务数据" else "（请基于你对该公司的了解进行分析）"}

### 财务数据
{financial_text if financial_text != "无财务数据" else "（无财务数据）"}

## 分析要求

请从以下维度进行深入分析：

1. **相对估值分析**
   - PE估值：当前PE水平，与历史、行业、市场对比
   - PB估值：当前PB水平，ROE是否支撑
   - PS估值：如适用
   - EV/EBITDA：如适用
   - PEG：结合增长率的估值

2. **绝对估值分析**（如有足够数据）
   - DCF估值的关键假设
   - 估算的内在价值区间

3. **成长性评估**
   - 收入增长预期
   - 利润增长预期
   - 增长的可持续性
   - 增长的质量（是否依赖外延并购）

4. **历史分位数分析**
   - 当前估值在历史上的分位数
   - 估值的周期性特征
   - 是否处于历史极端位置

5. **同业对比**
   - 与主要竞争对手的估值对比
   - 估值差异是否合理
   - 是否存在估值套利机会

6. **安全边际判断**
   - 当前价格相对内在价值的折扣/溢价
   - 安全边际是否足够
   - 下行风险评估

7. **合理估值区间**
   - 给出你认为的合理估值区间
   - 说明估值区间的假设前提

请以JSON格式输出你的分析结果。
"""
    
    def _get_mock_response(self) -> str:
        """获取模拟响应"""
        return """```json
{
    "summary": "当前估值处于历史中位数水平，考虑到成长性，估值合理但缺乏明显安全边际",
    "conclusion": "从相对估值角度看，公司当前PE为25倍，处于过去5年的50%分位数附近，既不便宜也不贵。考虑到公司未来3年预期15%的利润增长，PEG约为1.7，略高于1的合理水平，显示估值并不便宜。\\n\\n与同业对比，公司估值处于行业中上水平，这与其龙头地位和更强的盈利能力相匹配。从DCF角度粗略估算，假设10%的折现率和3%的永续增长率，公司内在价值约在当前价格的±15%范围内，安全边际有限。综合来看，当前估值合理，但对于价值投资者而言，可能需要等待更好的买入时机。",
    "score": 5.5,
    "confidence": 0.7,
    "key_findings": [
        {
            "finding": "PE处于历史中位数水平",
            "evidence": "当前PE 25倍，5年分位数50%",
            "impact": "中性 - 估值合理但不便宜"
        },
        {
            "finding": "PEG略高于合理水平",
            "evidence": "PEG约1.7，高于1的合理标准",
            "impact": "轻微负面 - 增长未完全反映在估值中"
        },
        {
            "finding": "估值溢价与龙头地位匹配",
            "evidence": "估值高于行业平均20%，但ROE也高于行业30%",
            "impact": "中性 - 溢价有基本面支撑"
        }
    ],
    "risks": [
        "增长不及预期将导致估值收缩",
        "市场风格切换可能压制高估值板块",
        "利率上行环境不利于高估值股票",
        "安全边际有限，下行空间存在"
    ],
    "opportunities": [
        "如增长超预期，估值有上行空间",
        "行业整合可能带来估值重估",
        "分红率提升可能吸引价值投资者",
        "市场回调时可能出现更好买点"
    ],
    "detailed_analysis": {
        "relative_valuation": {
            "pe": {
                "current": 25,
                "historical_avg": 24,
                "industry_avg": 20,
                "market_avg": 15,
                "assessment": "略高于历史和行业平均"
            },
            "pb": {
                "current": 6.5,
                "roe": "26%",
                "pb_roe_ratio": "合理，高ROE支撑高PB"
            },
            "peg": {
                "value": 1.7,
                "growth_rate": "15%",
                "assessment": "略高于合理水平"
            },
            "ev_ebitda": {
                "current": 18,
                "industry_avg": 15,
                "assessment": "高于行业平均"
            }
        },
        "absolute_valuation": {
            "dcf_assumptions": {
                "discount_rate": "10%",
                "terminal_growth": "3%",
                "fcf_growth_5y": "12%"
            },
            "intrinsic_value_range": {
                "low": "当前价格的-15%",
                "mid": "当前价格",
                "high": "当前价格的+15%"
            }
        },
        "growth_assessment": {
            "revenue_growth_3y": "12%",
            "profit_growth_3y": "15%",
            "sustainability": "较高，有机增长为主",
            "quality": "高质量，非依赖并购"
        },
        "historical_percentile": {
            "pe_percentile": "50%",
            "pb_percentile": "55%",
            "cyclicality": "估值周期性不明显",
            "extreme_position": false
        },
        "peer_comparison": {
            "vs_competitor_a": "估值高20%，但ROE也高25%",
            "vs_competitor_b": "估值高30%，增速也高20%",
            "premium_justified": true
        },
        "margin_of_safety": {
            "current_vs_intrinsic": "基本持平",
            "margin": "0-5%",
            "sufficient": false,
            "downside_risk": "15-20%"
        },
        "fair_value_range": {
            "low": "当前价格的85%",
            "mid": "当前价格",
            "high": "当前价格的115%",
            "assumptions": "基于15%利润增长和25倍PE"
        }
    }
}
```"""


if __name__ == "__main__":
    # 测试ValuationAgent
    agent = ValuationAgent(verbose=True)
    
    test_data = {
        "valuation": {
            "pe": 25,
            "pb": 6.5,
            "ps": 12
        },
        "financial_data": {
            "roe": 0.26,
            "net_profit_growth": 0.15
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
