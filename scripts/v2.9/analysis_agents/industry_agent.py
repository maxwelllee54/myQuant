#!/usr/bin/env python3
"""
V2.9 Analysis Agents - Industry Agent
行业分析Agent：分析行业周期、竞争格局、技术趋势

作者: Manus AI
版本: 2.9
"""

from typing import Dict, Any, Optional
from .base_agent import BaseAnalysisAgent, format_financial_data


class IndustryAgent(BaseAnalysisAgent):
    """
    行业分析Agent
    
    专注于：
    - 行业周期定位 (导入期、成长期、成熟期、衰退期)
    - 经济周期敏感性 (周期性 vs 防御性)
    - 市场规模与增长前景
    - 竞争格局分析 (波特五力)
    - 技术趋势与颠覆风险
    - 政策环境与监管趋势
    """
    
    AGENT_NAME = "IndustryAgent"
    AGENT_ROLE = "行业分析师"
    AGENT_DESCRIPTION = "专注于行业研究，分析行业周期、竞争格局、技术趋势和政策环境"
    
    def _get_system_prompt(self) -> str:
        return """你是一位资深的行业研究分析师，在顶级券商研究所有超过10年的行业研究经验。

你的核心能力：
1. 准确判断行业所处的生命周期阶段
2. 分析行业的经济周期敏感性
3. 评估行业的竞争格局和进入壁垒
4. 追踪技术趋势和潜在的颠覆性变化
5. 解读政策环境对行业的影响

分析框架：

1. **行业生命周期**
   - 导入期：市场刚起步，增长快但规模小
   - 成长期：市场快速扩张，竞争加剧
   - 成熟期：增长放缓，格局稳定，现金流充沛
   - 衰退期：市场萎缩，需要转型

2. **波特五力分析**
   - 现有竞争者的竞争强度
   - 潜在进入者的威胁
   - 替代品的威胁
   - 供应商的议价能力
   - 买方的议价能力

3. **经济周期敏感性**
   - 强周期性：与GDP高度相关（如汽车、房地产）
   - 弱周期性：相对稳定（如消费品、医药）
   - 逆周期性：经济下行时反而受益（如折扣零售）

你必须以JSON格式输出分析结果，包含以下字段：
- summary: 一句话总结行业状况
- conclusion: 详细的行业分析结论（2-3段）
- score: 行业吸引力评分（1-10分）
- confidence: 分析置信度（0-1）
- key_findings: 关键发现列表
- risks: 行业风险列表
- opportunities: 行业机会列表
- detailed_analysis: 详细分析，包含：
  - lifecycle_stage: 生命周期阶段
  - economic_sensitivity: 经济周期敏感性
  - market_size: 市场规模分析
  - growth_prospects: 增长前景
  - competitive_landscape: 竞争格局（波特五力）
  - tech_trends: 技术趋势
  - policy_environment: 政策环境
"""
    
    def _prepare_data(
        self,
        raw_data: Dict[str, Any],
        quant_results: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """准备行业分析所需的数据"""
        prepared = {}
        
        # 提取行业信息
        prepared["industry_info"] = raw_data.get("industry_info", {})
        
        # 提取行业数据
        prepared["industry_data"] = raw_data.get("industry_data", {})
        
        # 提取宏观经济数据
        prepared["macro_data"] = raw_data.get("macro_data", {})
        
        # 提取公司在行业中的位置
        prepared["company_position"] = raw_data.get("company_position", {})
        
        return prepared
    
    def _build_analysis_prompt(
        self,
        stock_code: str,
        company_name: str,
        prepared_data: Dict[str, Any]
    ) -> str:
        """构建行业分析Prompt"""
        
        industry_info = prepared_data.get("industry_info", {})
        industry_name = industry_info.get("name", "未知行业")
        
        industry_text = format_financial_data(prepared_data.get("industry_data", {}))
        macro_text = format_financial_data(prepared_data.get("macro_data", {}))
        
        return f"""## 分析任务

请对 **{company_name}** ({stock_code}) 所处的行业进行深度分析。

## 公司所属行业

{industry_name}

## 可用数据

### 行业数据
{industry_text if industry_text != "无财务数据" else "（请基于你对该行业的了解进行分析）"}

### 宏观经济数据
{macro_text if macro_text != "无财务数据" else "（无宏观数据）"}

## 分析要求

请从以下维度进行深入分析：

1. **行业生命周期定位**
   - 当前处于哪个阶段（导入期/成长期/成熟期/衰退期）
   - 判断依据是什么
   - 预计还能持续多久

2. **经济周期敏感性**
   - 该行业是强周期、弱周期还是逆周期
   - 当前经济周期对行业的影响
   - 未来1-2年的周期展望

3. **市场规模与增长前景**
   - 当前市场规模
   - 预期增长率
   - 增长的主要驱动因素

4. **竞争格局分析（波特五力）**
   - 现有竞争者：竞争激烈程度
   - 潜在进入者：进入壁垒高低
   - 替代品威胁：是否存在替代技术/产品
   - 供应商议价能力
   - 买方议价能力

5. **技术趋势与颠覆风险**
   - 当前的技术发展趋势
   - 是否存在颠覆性技术威胁
   - 技术变革对行业格局的影响

6. **政策环境**
   - 当前的监管政策
   - 政策趋势（收紧/放松）
   - 政策对行业的影响

请以JSON格式输出你的分析结果。
"""
    
    def _get_mock_response(self) -> str:
        """获取模拟响应"""
        return """```json
{
    "summary": "行业处于成熟期后段，竞争格局稳定，但面临技术变革压力",
    "conclusion": "该行业已进入成熟期，市场增长放缓至个位数，但行业格局已经稳定，龙头企业享有较强的规模优势和品牌溢价。行业集中度持续提升，CR5已超过60%，中小企业生存空间被压缩。\\n\\n从周期角度看，该行业属于弱周期性行业，受经济波动影响相对较小，在经济下行期具有一定的防御属性。然而，需要关注的是新技术的冲击，数字化转型正在重塑行业竞争规则，传统企业如不能及时转型，可能面临被边缘化的风险。",
    "score": 6.5,
    "confidence": 0.8,
    "key_findings": [
        {
            "finding": "行业集中度持续提升",
            "evidence": "CR5从5年前的45%提升至当前的62%",
            "impact": "正面 - 龙头企业受益于行业整合"
        },
        {
            "finding": "数字化转型成为竞争焦点",
            "evidence": "头部企业数字化投入年增长超过30%",
            "impact": "中性 - 既是机会也是挑战"
        },
        {
            "finding": "政策环境趋于严格",
            "evidence": "近两年出台多项行业规范政策",
            "impact": "负面 - 合规成本上升"
        }
    ],
    "risks": [
        "行业增长放缓，天花板逐渐显现",
        "技术变革可能颠覆现有竞争格局",
        "政策监管趋严，合规成本上升",
        "人力成本持续上涨压缩利润空间"
    ],
    "opportunities": [
        "行业整合带来的市场份额提升机会",
        "数字化转型提升运营效率",
        "海外市场拓展空间",
        "产品升级带来的毛利率提升"
    ],
    "detailed_analysis": {
        "lifecycle_stage": {
            "current_stage": "成熟期后段",
            "evidence": "市场增速降至5%以下，格局稳定，并购活跃",
            "duration_estimate": "预计还将持续5-10年"
        },
        "economic_sensitivity": {
            "type": "弱周期性",
            "gdp_correlation": "0.3-0.4",
            "current_cycle_impact": "当前经济放缓对行业影响有限",
            "outlook": "经济复苏将带来温和增长"
        },
        "market_size": {
            "current_size": "约5000亿元",
            "cagr_5y": "5-7%",
            "drivers": ["消费升级", "渠道下沉", "产品创新"]
        },
        "growth_prospects": {
            "short_term": "稳定增长，增速5%左右",
            "medium_term": "增速可能进一步放缓",
            "long_term": "取决于技术变革和海外拓展"
        },
        "competitive_landscape": {
            "rivalry": "中等偏高，价格战时有发生",
            "entry_barriers": "较高，品牌和渠道是主要壁垒",
            "substitutes": "中等威胁，新技术可能带来替代品",
            "supplier_power": "较低，供应商分散",
            "buyer_power": "中等，大客户有一定议价能力"
        },
        "tech_trends": {
            "current_trends": ["数字化", "智能化", "绿色化"],
            "disruption_risk": "中等，需关注新技术发展",
            "impact": "可能重塑竞争格局"
        },
        "policy_environment": {
            "current_policy": "规范化监管加强",
            "trend": "趋严",
            "impact": "短期增加成本，长期利好规范经营企业"
        }
    }
}
```"""


if __name__ == "__main__":
    # 测试IndustryAgent
    agent = IndustryAgent(verbose=True)
    
    test_data = {
        "industry_info": {
            "name": "白酒行业",
            "classification": "消费品"
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
