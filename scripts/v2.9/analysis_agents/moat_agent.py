#!/usr/bin/env python3
"""
V2.9 Analysis Agents - Moat Agent
护城河与商业模式分析Agent：评估公司的竞争优势和商业模式可持续性

作者: Manus AI
版本: 2.9
"""

from typing import Dict, Any, Optional
from .base_agent import BaseAnalysisAgent, format_financial_data


class MoatAgent(BaseAnalysisAgent):
    """
    护城河与商业模式分析Agent
    
    专注于：
    - 护城河类型识别 (品牌、网络效应、成本优势、转换成本、无形资产)
    - 护城河宽度和持久性评估
    - 定价权分析
    - 商业模式解析
    - 竞争优势的可持续性
    """
    
    AGENT_NAME = "MoatAgent"
    AGENT_ROLE = "战略分析师"
    AGENT_DESCRIPTION = "专注于评估公司护城河、定价权和商业模式的可持续性"
    
    def _get_system_prompt(self) -> str:
        return """你是一位资深的战略分析师，深谙巴菲特和芒格的投资哲学，专注于识别和评估企业的竞争优势（护城河）。

你的核心能力：
1. 识别企业的护城河类型
2. 评估护城河的宽度和持久性
3. 分析企业的定价权
4. 解析商业模式的本质
5. 判断竞争优势的可持续性

护城河类型框架（晨星/巴菲特框架）：

1. **品牌护城河**
   - 消费者愿意为品牌支付溢价
   - 品牌代表着质量、信任或身份认同
   - 例：茅台、可口可乐、苹果

2. **网络效应护城河**
   - 用户越多，产品/服务价值越大
   - 形成正反馈循环
   - 例：微信、淘宝、Visa

3. **成本优势护城河**
   - 规模经济或独特的低成本结构
   - 竞争对手难以复制
   - 例：沃尔玛、格力

4. **转换成本护城河**
   - 客户更换供应商的成本很高
   - 包括金钱成本、时间成本、学习成本
   - 例：用友、SAP、医疗器械

5. **无形资产护城河**
   - 专利、许可证、特许经营权
   - 法律保护的独占权
   - 例：药企专利、矿业许可证

护城河评估标准：
- **宽度**：竞争对手跨越的难度
- **持久性**：护城河能维持多久
- **趋势**：护城河是在加宽还是收窄

你必须以JSON格式输出分析结果，包含以下字段：
- summary: 一句话总结护城河状况
- conclusion: 详细的护城河分析结论（2-3段）
- score: 护城河强度评分（1-10分）
- confidence: 分析置信度（0-1）
- key_findings: 关键发现列表
- risks: 护城河面临的威胁
- opportunities: 护城河加宽的机会
- detailed_analysis: 详细分析，包含：
  - moat_types: 护城河类型列表及强度
  - moat_width: 护城河宽度评估
  - moat_durability: 护城河持久性评估
  - moat_trend: 护城河趋势（加宽/稳定/收窄）
  - pricing_power: 定价权分析
  - business_model: 商业模式解析
  - sustainability: 可持续性评估
"""
    
    def _prepare_data(
        self,
        raw_data: Dict[str, Any],
        quant_results: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """准备护城河分析所需的数据"""
        prepared = {}
        
        # 提取公司基本信息
        prepared["company_info"] = raw_data.get("company_info", {})
        
        # 提取财务数据（用于验证护城河）
        prepared["financial"] = raw_data.get("financial_data", {})
        
        # 提取竞争对手数据
        prepared["competitors"] = raw_data.get("competitors", {})
        
        # 提取产品/服务信息
        prepared["products"] = raw_data.get("products", {})
        
        return prepared
    
    def _build_analysis_prompt(
        self,
        stock_code: str,
        company_name: str,
        prepared_data: Dict[str, Any]
    ) -> str:
        """构建护城河分析Prompt"""
        
        company_info = prepared_data.get("company_info", {})
        business_desc = company_info.get("business_description", "")
        
        financial_text = format_financial_data(prepared_data.get("financial", {}))
        
        return f"""## 分析任务

请对 **{company_name}** ({stock_code}) 的护城河和商业模式进行深度分析。

## 公司简介

{business_desc if business_desc else "（请基于你对该公司的了解进行分析）"}

## 财务数据参考

{financial_text if financial_text != "无财务数据" else "（无财务数据）"}

## 分析要求

请从以下维度进行深入分析：

1. **护城河类型识别**
   - 该公司拥有哪些类型的护城河？
   - 每种护城河的强度如何（强/中/弱）？
   - 主要护城河是什么？

2. **护城河宽度评估**
   - 竞争对手跨越这条护城河的难度有多大？
   - 需要多少时间和资源才能复制？
   - 是否有成功跨越的案例？

3. **护城河持久性评估**
   - 这条护城河能维持多久？
   - 有哪些因素可能侵蚀护城河？
   - 技术变革是否会威胁护城河？

4. **护城河趋势判断**
   - 护城河是在加宽、稳定还是收窄？
   - 公司是否在主动加固护城河？
   - 未来3-5年的趋势预判

5. **定价权分析**
   - 公司是否拥有定价权？
   - 定价权的来源是什么？
   - 历史上的提价能力如何？
   - 提价是否会导致客户流失？

6. **商业模式解析**
   - 公司如何创造价值？
   - 公司如何获取价值（盈利模式）？
   - 商业模式是否可持续？
   - 商业模式是否可扩展？

7. **可持续性综合评估**
   - 竞争优势能否持续10年以上？
   - 最大的威胁是什么？
   - 需要关注的预警信号

请以JSON格式输出你的分析结果。
"""
    
    def _get_mock_response(self) -> str:
        """获取模拟响应"""
        return """```json
{
    "summary": "公司拥有强大的品牌护城河和一定的定价权，护城河宽度较宽且趋势稳定",
    "conclusion": "该公司的核心护城河是其强大的品牌资产，这是经过数十年甚至上百年积累形成的，竞争对手几乎不可能在短期内复制。品牌不仅代表着产品质量，更承载着深厚的文化内涵和社会认同，这使得消费者愿意为其支付显著的溢价。\\n\\n从定价权角度看，公司过去10年多次成功提价，且每次提价后销量不降反升，这是极强定价权的有力证明。商业模式简单清晰，轻资产运营，现金流充沛，具有极高的可持续性。唯一需要关注的是消费习惯的代际变化，年轻一代的消费偏好可能与老一代有所不同。",
    "score": 9.0,
    "confidence": 0.85,
    "key_findings": [
        {
            "finding": "品牌护城河极其强大",
            "evidence": "品牌历史超过百年，品牌价值位居行业第一",
            "impact": "正面 - 支撑高毛利率和定价权"
        },
        {
            "finding": "定价权得到历史验证",
            "evidence": "过去10年累计提价超过200%，销量持续增长",
            "impact": "正面 - 证明需求刚性"
        },
        {
            "finding": "商业模式轻资产、高现金流",
            "evidence": "资本支出占收入比例低于5%，自由现金流充沛",
            "impact": "正面 - 高股东回报潜力"
        }
    ],
    "risks": [
        "消费习惯的代际变化可能影响长期需求",
        "替代品的出现可能分流部分消费者",
        "政策风险（如消费税调整）",
        "品牌声誉风险"
    ],
    "opportunities": [
        "产品系列扩展，覆盖更多价格带",
        "国际化拓展，开拓海外市场",
        "数字化营销，触达年轻消费者",
        "文化IP开发，强化品牌价值"
    ],
    "detailed_analysis": {
        "moat_types": [
            {
                "type": "品牌护城河",
                "strength": "极强",
                "description": "百年品牌积淀，文化符号地位"
            },
            {
                "type": "无形资产护城河",
                "strength": "强",
                "description": "独特的生产工艺和地理标志保护"
            },
            {
                "type": "转换成本护城河",
                "strength": "中等",
                "description": "消费习惯和社交场景形成一定粘性"
            }
        ],
        "moat_width": {
            "assessment": "宽",
            "replication_difficulty": "极高",
            "time_to_replicate": "几乎不可能复制",
            "historical_attempts": "多家企业尝试模仿，均未成功"
        },
        "moat_durability": {
            "assessment": "持久",
            "expected_duration": "10年以上",
            "erosion_factors": ["消费代际变化", "健康意识提升"],
            "tech_threat": "低，传统工艺反而是优势"
        },
        "moat_trend": {
            "direction": "稳定略有加宽",
            "company_actions": "持续品牌建设和文化营销",
            "outlook": "未来3-5年预计保持稳定"
        },
        "pricing_power": {
            "has_pricing_power": true,
            "source": "品牌溢价和需求刚性",
            "historical_evidence": "10年累计提价超200%",
            "customer_retention": "提价后客户流失率极低"
        },
        "business_model": {
            "value_creation": "生产稀缺的高端消费品",
            "value_capture": "高毛利率销售",
            "sustainability": "极高",
            "scalability": "中等，受产能限制"
        },
        "sustainability": {
            "10_year_outlook": "竞争优势大概率可持续",
            "biggest_threat": "消费代际变化",
            "warning_signs": ["年轻消费者渗透率下降", "替代品份额上升"]
        }
    }
}
```"""


if __name__ == "__main__":
    # 测试MoatAgent
    agent = MoatAgent(verbose=True)
    
    test_data = {
        "company_info": {
            "business_description": "中国高端白酒龙头企业"
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
