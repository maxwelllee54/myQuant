#!/usr/bin/env python3
"""
V2.9 Analysis Agents - Risk Agent
定性风险分析Agent：识别和评估非量化风险因素

作者: Manus AI
版本: 2.9
"""

from typing import Dict, Any, Optional
from .base_agent import BaseAnalysisAgent, format_financial_data


class RiskAgent(BaseAnalysisAgent):
    """
    定性风险分析Agent
    
    专注于识别和评估非量化风险：
    - 管理层风险 (诚信、能力、激励)
    - 公司治理风险
    - 政策与监管风险
    - 技术颠覆风险
    - ESG风险
    - 地缘政治风险
    - 黑天鹅风险
    """
    
    AGENT_NAME = "RiskAgent"
    AGENT_ROLE = "风险控制官"
    AGENT_DESCRIPTION = "专注于识别和评估公司面临的非量化风险因素"
    
    def _get_system_prompt(self) -> str:
        return """你是一位资深的风险控制官，曾在大型资产管理公司和投行担任首席风险官，专注于识别和评估投资中的非量化风险。

你的核心能力：
1. 识别隐藏的风险因素
2. 评估风险的概率和影响
3. 分析风险之间的关联性
4. 提出风险缓解建议
5. 识别潜在的黑天鹅事件

风险分析框架：

1. **管理层风险**
   - 诚信度：是否有财务造假、关联交易等历史
   - 能力：战略眼光、执行力、行业经验
   - 激励：薪酬结构是否与股东利益一致
   - 稳定性：核心管理层是否稳定

2. **公司治理风险**
   - 股权结构：是否存在一股独大
   - 董事会独立性
   - 信息披露质量
   - 中小股东保护

3. **政策与监管风险**
   - 行业政策变化
   - 税收政策
   - 环保政策
   - 反垄断政策

4. **技术颠覆风险**
   - 替代技术的威胁
   - 公司的技术储备
   - 研发投入和创新能力

5. **ESG风险**
   - 环境风险（碳排放、污染）
   - 社会风险（劳工、社区关系）
   - 治理风险（腐败、合规）

6. **地缘政治风险**
   - 国际贸易摩擦
   - 供应链安全
   - 海外业务风险

7. **黑天鹅风险**
   - 低概率高影响事件
   - 公司的抗风险能力

风险评估原则：
- 风险 = 概率 × 影响
- 关注尾部风险，而非仅关注平均情况
- 识别风险之间的关联性
- 区分可控风险和不可控风险

你必须以JSON格式输出分析结果，包含以下字段：
- summary: 一句话总结风险状况
- conclusion: 详细的风险分析结论（2-3段）
- score: 风险水平评分（1-10分，10分表示风险极低）
- confidence: 分析置信度（0-1）
- key_findings: 关键风险发现列表
- risks: 主要风险列表（按重要性排序）
- opportunities: 风险中的机会（如风险被过度定价）
- detailed_analysis: 详细分析，包含各类风险的评估
"""
    
    def _prepare_data(
        self,
        raw_data: Dict[str, Any],
        quant_results: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """准备风险分析所需的数据"""
        prepared = {}
        
        # 提取公司信息
        prepared["company_info"] = raw_data.get("company_info", {})
        
        # 提取管理层信息
        prepared["management"] = raw_data.get("management", {})
        
        # 提取治理信息
        prepared["governance"] = raw_data.get("governance", {})
        
        # 提取新闻和事件
        prepared["news_events"] = raw_data.get("news_events", {})
        
        return prepared
    
    def _build_analysis_prompt(
        self,
        stock_code: str,
        company_name: str,
        prepared_data: Dict[str, Any]
    ) -> str:
        """构建风险分析Prompt"""
        
        company_info = prepared_data.get("company_info", {})
        
        return f"""## 分析任务

请对 **{company_name}** ({stock_code}) 进行全面的定性风险分析。

## 公司信息

{format_financial_data(company_info) if company_info else "（请基于你对该公司的了解进行分析）"}

## 分析要求

请从以下维度进行深入的风险分析：

1. **管理层风险**
   - 管理层的诚信记录如何？
   - 管理层的战略能力和执行力如何？
   - 管理层激励是否与股东利益一致？
   - 核心管理层是否稳定？

2. **公司治理风险**
   - 股权结构是否存在问题？
   - 董事会是否独立有效？
   - 信息披露是否透明？
   - 中小股东权益是否得到保护？

3. **政策与监管风险**
   - 公司面临哪些政策风险？
   - 监管趋势是收紧还是放松？
   - 政策变化的概率和影响？

4. **技术颠覆风险**
   - 是否存在颠覆性技术威胁？
   - 公司的技术储备如何？
   - 如果被颠覆，影响有多大？

5. **ESG风险**
   - 环境风险：碳排放、污染、资源消耗
   - 社会风险：劳工关系、社区关系、产品安全
   - 治理风险：腐败、合规、利益冲突

6. **地缘政治风险**
   - 国际业务面临的风险
   - 供应链安全
   - 贸易摩擦影响

7. **黑天鹅风险识别**
   - 有哪些低概率但高影响的潜在事件？
   - 公司的抗风险能力如何？
   - 是否有足够的安全垫？

8. **风险综合评估**
   - 最大的风险是什么？
   - 风险之间是否存在关联？
   - 整体风险水平如何？

请以JSON格式输出你的分析结果。
"""
    
    def _get_mock_response(self) -> str:
        """获取模拟响应"""
        return """```json
{
    "summary": "公司整体风险可控，主要关注政策风险和消费代际变化风险",
    "conclusion": "从定性风险角度看，该公司整体风险水平较低。管理层稳定且经验丰富，公司治理结构完善，作为国企有一定的政策支持。然而，需要关注的是行业政策风险，包括消费税调整的可能性以及对高端消费的政策态度变化。\\n\\n另一个中长期风险是消费习惯的代际变化，年轻一代的消费偏好可能与老一代有所不同，这可能在10-20年的时间尺度上影响公司的增长前景。ESG方面，公司在环境和社会责任方面表现良好，但需要持续关注。总体而言，公司的风险收益比较为有利。",
    "score": 7.5,
    "confidence": 0.75,
    "key_findings": [
        {
            "finding": "政策风险是主要关注点",
            "evidence": "消费税调整讨论、高端消费政策态度",
            "impact": "中等 - 可能影响盈利能力"
        },
        {
            "finding": "消费代际变化是长期风险",
            "evidence": "年轻消费者渗透率相对较低",
            "impact": "长期 - 可能影响增长前景"
        },
        {
            "finding": "公司治理结构完善",
            "evidence": "国企背景、规范的信息披露",
            "impact": "正面 - 降低治理风险"
        }
    ],
    "risks": [
        {
            "risk": "消费税调整风险",
            "probability": "中等",
            "impact": "高",
            "timeframe": "1-3年",
            "mitigation": "产品结构调整、成本控制"
        },
        {
            "risk": "消费代际变化",
            "probability": "中等",
            "impact": "中等",
            "timeframe": "10-20年",
            "mitigation": "年轻化营销、产品创新"
        },
        {
            "risk": "宏观经济下行",
            "probability": "中等",
            "impact": "低",
            "timeframe": "1-2年",
            "mitigation": "产品刚性需求提供缓冲"
        },
        {
            "risk": "食品安全事件",
            "probability": "低",
            "impact": "极高",
            "timeframe": "随时",
            "mitigation": "严格质量控制"
        }
    ],
    "opportunities": [
        "政策风险可能被市场过度定价",
        "短期波动可能创造买入机会",
        "ESG评级提升可能吸引更多资金"
    ],
    "detailed_analysis": {
        "management_risk": {
            "integrity": {
                "assessment": "良好",
                "evidence": "无重大诚信问题记录",
                "score": 8
            },
            "capability": {
                "assessment": "优秀",
                "evidence": "长期稳定的业绩表现",
                "score": 8
            },
            "incentive_alignment": {
                "assessment": "中等",
                "evidence": "国企薪酬体系，激励不如民企",
                "score": 6
            },
            "stability": {
                "assessment": "稳定",
                "evidence": "核心管理层任期较长",
                "score": 8
            }
        },
        "governance_risk": {
            "ownership_structure": {
                "assessment": "国有控股，结构清晰",
                "risk_level": "低"
            },
            "board_independence": {
                "assessment": "符合监管要求",
                "risk_level": "低"
            },
            "disclosure_quality": {
                "assessment": "信息披露规范透明",
                "risk_level": "低"
            },
            "minority_protection": {
                "assessment": "中等，国企通病",
                "risk_level": "中低"
            }
        },
        "policy_risk": {
            "consumption_tax": {
                "probability": "中等",
                "impact": "高",
                "trend": "不确定"
            },
            "luxury_consumption_policy": {
                "probability": "低",
                "impact": "中等",
                "trend": "稳定"
            },
            "environmental_policy": {
                "probability": "低",
                "impact": "低",
                "trend": "可控"
            }
        },
        "tech_disruption_risk": {
            "assessment": "极低",
            "rationale": "传统工艺是核心竞争力，不易被技术颠覆",
            "score": 9
        },
        "esg_risk": {
            "environmental": {
                "assessment": "低风险",
                "key_issues": ["水资源使用", "包装材料"],
                "score": 7
            },
            "social": {
                "assessment": "低风险",
                "key_issues": ["员工福利", "社区关系"],
                "score": 8
            },
            "governance": {
                "assessment": "低风险",
                "key_issues": ["关联交易监控"],
                "score": 7
            }
        },
        "geopolitical_risk": {
            "assessment": "低",
            "rationale": "主要业务在国内，海外业务占比低",
            "score": 8
        },
        "black_swan_risks": [
            {
                "event": "重大食品安全事件",
                "probability": "极低",
                "impact": "灾难性",
                "resilience": "品牌可能难以恢复"
            },
            {
                "event": "核心生产基地自然灾害",
                "probability": "极低",
                "impact": "高",
                "resilience": "有一定库存缓冲"
            }
        ]
    }
}
```"""


if __name__ == "__main__":
    # 测试RiskAgent
    agent = RiskAgent(verbose=True)
    
    test_data = {
        "company_info": {
            "name": "贵州茅台",
            "industry": "白酒",
            "ownership": "国有控股"
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
