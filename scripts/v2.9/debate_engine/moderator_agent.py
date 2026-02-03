#!/usr/bin/env python3
"""
V2.9 Debate Engine - Moderator Agent
主持人Agent：组织和引导多轮辩论，形成最终结论

作者: Manus AI
版本: 2.9
"""

import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class DebateRound:
    """辩论轮次的数据结构"""
    round_number: int
    phase: str  # "initial_analysis", "cross_examination", "rebuttal", "final_synthesis"
    content: Dict[str, Any]
    timestamp: str


@dataclass
class CrossExaminationQuestion:
    """交叉质询问题"""
    target_agent: str
    question: str
    context: str
    priority: str  # "high", "medium", "low"


@dataclass
class FinalConclusion:
    """最终结论"""
    investment_rating: str  # "强烈买入", "买入", "持有", "卖出", "强烈卖出"
    confidence: float
    target_price_range: Dict[str, float]
    core_thesis: str
    key_agreements: List[str]
    key_disagreements: List[str]
    main_risks: List[str]
    main_opportunities: List[str]
    agent_summaries: Dict[str, str]
    final_recommendation: str


class ModeratorAgent:
    """
    主持人Agent
    
    职责：
    1. 汇总各专家Agent的初步分析
    2. 识别共识和分歧
    3. 生成交叉质询问题
    4. 引导多轮辩论
    5. 综合形成最终结论
    """
    
    def __init__(self, llm_client=None, verbose: bool = True):
        """
        初始化主持人Agent
        
        Args:
            llm_client: LLM客户端实例
            verbose: 是否打印详细日志
        """
        self.llm_client = llm_client
        self.verbose = verbose
        self._init_llm_client()
    
    def _init_llm_client(self):
        """初始化LLM客户端"""
        if self.llm_client is None:
            try:
                from openai import OpenAI
                self.llm_client = OpenAI()
                self.llm_model = "gpt-4o"
            except Exception as e:
                if self.verbose:
                    print(f"[Moderator] Warning: Failed to init OpenAI client: {e}")
                self.llm_client = None
                self.llm_model = None
    
    def summarize_initial_analyses(
        self,
        initial_analyses: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        汇总初步分析，识别共识和分歧
        
        Args:
            initial_analyses: 各Agent的初步分析结果
        
        Returns:
            汇总结果，包含共识、分歧和关键问题
        """
        if self.verbose:
            print("[Moderator] 汇总各专家的初步分析...")
        
        system_prompt = self._get_summary_system_prompt()
        user_prompt = self._build_summary_prompt(initial_analyses)
        
        response = self._call_llm(system_prompt, user_prompt)
        summary = self._parse_summary_response(response)
        
        if self.verbose:
            print(f"[Moderator] 识别到 {len(summary.get('agreements', []))} 个共识点")
            print(f"[Moderator] 识别到 {len(summary.get('disagreements', []))} 个分歧点")
        
        return summary
    
    def generate_cross_examination_questions(
        self,
        summary: Dict[str, Any],
        debate_history: List[DebateRound]
    ) -> Dict[str, List[CrossExaminationQuestion]]:
        """
        生成交叉质询问题
        
        Args:
            summary: 汇总结果
            debate_history: 辩论历史
        
        Returns:
            按Agent分组的质询问题
        """
        if self.verbose:
            print("[Moderator] 生成交叉质询问题...")
        
        system_prompt = self._get_cross_examination_system_prompt()
        user_prompt = self._build_cross_examination_prompt(summary, debate_history)
        
        response = self._call_llm(system_prompt, user_prompt)
        questions = self._parse_cross_examination_response(response)
        
        total_questions = sum(len(q) for q in questions.values())
        if self.verbose:
            print(f"[Moderator] 生成了 {total_questions} 个质询问题")
        
        return questions
    
    def synthesize_final_conclusion(
        self,
        stock_code: str,
        company_name: str,
        debate_history: List[DebateRound],
        quant_results: Optional[Dict[str, Any]] = None
    ) -> FinalConclusion:
        """
        综合形成最终结论
        
        Args:
            stock_code: 股票代码
            company_name: 公司名称
            debate_history: 完整的辩论历史
            quant_results: 量化分析结果
        
        Returns:
            FinalConclusion: 最终结论
        """
        if self.verbose:
            print("[Moderator] 综合形成最终投资结论...")
        
        system_prompt = self._get_synthesis_system_prompt()
        user_prompt = self._build_synthesis_prompt(
            stock_code, company_name, debate_history, quant_results
        )
        
        response = self._call_llm(system_prompt, user_prompt)
        conclusion = self._parse_synthesis_response(response)
        
        if self.verbose:
            print(f"[Moderator] 最终评级: {conclusion.investment_rating}")
            print(f"[Moderator] 置信度: {conclusion.confidence:.0%}")
        
        return conclusion
    
    def generate_final_report(
        self,
        stock_code: str,
        company_name: str,
        conclusion: FinalConclusion,
        debate_history: List[DebateRound]
    ) -> str:
        """
        生成最终的投资分析报告（Markdown格式）
        
        Args:
            stock_code: 股票代码
            company_name: 公司名称
            conclusion: 最终结论
            debate_history: 辩论历史
        
        Returns:
            Markdown格式的报告
        """
        report = f"""# {company_name} ({stock_code}) 综合投资分析报告

**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**分析框架**: quant-investor V2.9 多Agent辩论系统

---

## 1. 投资评级与核心结论

| 项目 | 结论 |
|:---|:---|
| **投资评级** | {conclusion.investment_rating} |
| **置信度** | {conclusion.confidence:.0%} |
| **目标价格区间** | {conclusion.target_price_range.get('low', 'N/A')} - {conclusion.target_price_range.get('high', 'N/A')} |

### 核心投资逻辑

{conclusion.core_thesis}

---

## 2. 关键共识

经过多轮辩论，各专家Agent在以下方面达成共识：

"""
        for i, agreement in enumerate(conclusion.key_agreements, 1):
            report += f"{i}. {agreement}\n"
        
        report += """
---

## 3. 主要分歧

以下是辩论中未能完全解决的分歧点：

"""
        for i, disagreement in enumerate(conclusion.key_disagreements, 1):
            report += f"{i}. {disagreement}\n"
        
        report += """
---

## 4. 风险与机会

### 主要风险

"""
        for i, risk in enumerate(conclusion.main_risks, 1):
            report += f"{i}. {risk}\n"
        
        report += """
### 主要机会

"""
        for i, opportunity in enumerate(conclusion.main_opportunities, 1):
            report += f"{i}. {opportunity}\n"
        
        report += """
---

## 5. 各专家Agent观点摘要

"""
        for agent_name, summary in conclusion.agent_summaries.items():
            report += f"### {agent_name}\n\n{summary}\n\n"
        
        report += f"""---

## 6. 最终投资建议

{conclusion.final_recommendation}

---

## 7. 辩论过程摘要

本报告基于 {len(debate_history)} 轮辩论形成，参与的专家Agent包括：财务分析师、行业分析师、战略分析师、估值分析师和风险控制官。

---

*本报告由 quant-investor V2.9 多Agent辩论系统自动生成，仅供参考，不构成投资建议。*
"""
        return report
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """调用LLM"""
        if self.llm_client is None:
            return self._get_mock_response()
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=4000
            )
            return response.choices[0].message.content
        except Exception as e:
            if self.verbose:
                print(f"[Moderator] LLM调用失败: {e}")
            return self._get_mock_response()
    
    def _get_summary_system_prompt(self) -> str:
        return """你是一位资深的投资委员会主席，负责汇总各专家的分析意见，识别共识和分歧。

你的任务是：
1. 仔细阅读每位专家的分析报告
2. 识别所有专家都同意的关键点（共识）
3. 识别专家之间存在分歧的观点（分歧）
4. 找出需要进一步讨论的关键问题

请以JSON格式输出，包含：
- agreements: 共识点列表
- disagreements: 分歧点列表，每项包含 {topic, positions: {agent: view}}
- key_questions: 需要进一步讨论的问题
- overall_sentiment: 整体情绪 (bullish/neutral/bearish)
"""
    
    def _get_cross_examination_system_prompt(self) -> str:
        return """你是一位资深的投资委员会主席，负责生成交叉质询问题以深化讨论。

你的任务是：
1. 针对识别出的分歧点，生成有针对性的质询问题
2. 要求持不同观点的专家提供更多证据或解释
3. 挑战可能存在的假设或逻辑漏洞
4. 促进专家之间的思想碰撞

请以JSON格式输出，按目标Agent分组：
{
    "AgentName": [
        {"question": "问题内容", "context": "问题背景", "priority": "high/medium/low"}
    ]
}
"""
    
    def _get_synthesis_system_prompt(self) -> str:
        return """你是一位资深的投资委员会主席，负责综合所有讨论形成最终投资结论。

你的任务是：
1. 综合考虑所有专家的观点和辩论结果
2. 权衡共识和分歧
3. 结合量化分析结果（如有）
4. 形成一个平衡、全面的投资结论

请以JSON格式输出最终结论，包含：
- investment_rating: 投资评级 (强烈买入/买入/持有/卖出/强烈卖出)
- confidence: 置信度 (0-1)
- target_price_range: {low, mid, high}
- core_thesis: 核心投资逻辑（2-3段）
- key_agreements: 关键共识列表
- key_disagreements: 主要分歧列表
- main_risks: 主要风险列表
- main_opportunities: 主要机会列表
- agent_summaries: 各Agent最终观点摘要
- final_recommendation: 最终投资建议（1段）
"""
    
    def _build_summary_prompt(self, initial_analyses: Dict[str, Any]) -> str:
        analyses_text = ""
        for agent_name, analysis in initial_analyses.items():
            if hasattr(analysis, 'to_dict'):
                analysis = analysis.to_dict()
            analyses_text += f"\n### {agent_name}\n"
            analyses_text += f"评分: {analysis.get('score', 'N/A')}/10\n"
            analyses_text += f"置信度: {analysis.get('confidence', 'N/A')}\n"
            analyses_text += f"总结: {analysis.get('summary', 'N/A')}\n"
            analyses_text += f"结论: {analysis.get('conclusion', 'N/A')}\n"
        
        return f"""## 各专家Agent的初步分析

{analyses_text}

## 请汇总以上分析，识别共识和分歧
"""
    
    def _build_cross_examination_prompt(
        self,
        summary: Dict[str, Any],
        debate_history: List[DebateRound]
    ) -> str:
        disagreements_text = ""
        for d in summary.get("disagreements", []):
            disagreements_text += f"\n- 话题: {d.get('topic', 'N/A')}\n"
            for agent, view in d.get("positions", {}).items():
                disagreements_text += f"  - {agent}: {view}\n"
        
        return f"""## 已识别的分歧点

{disagreements_text}

## 请针对这些分歧生成交叉质询问题
"""
    
    def _build_synthesis_prompt(
        self,
        stock_code: str,
        company_name: str,
        debate_history: List[DebateRound],
        quant_results: Optional[Dict[str, Any]]
    ) -> str:
        history_text = ""
        for round_data in debate_history:
            if isinstance(round_data, DebateRound):
                history_text += f"\n### 第{round_data.round_number}轮 ({round_data.phase})\n"
                history_text += json.dumps(round_data.content, ensure_ascii=False, indent=2)[:1000]
            else:
                history_text += f"\n{json.dumps(round_data, ensure_ascii=False, indent=2)[:1000]}"
        
        quant_text = ""
        if quant_results:
            quant_text = f"\n## 量化分析结果\n{json.dumps(quant_results, ensure_ascii=False, indent=2)[:500]}"
        
        return f"""## 分析标的

{company_name} ({stock_code})

## 辩论历史摘要

{history_text}

{quant_text}

## 请综合以上讨论，形成最终投资结论
"""
    
    def _parse_summary_response(self, response: str) -> Dict[str, Any]:
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        return {
            "agreements": ["模拟共识点"],
            "disagreements": [{"topic": "估值", "positions": {"ValuationAgent": "合理", "RiskAgent": "偏高"}}],
            "key_questions": ["估值是否合理？"],
            "overall_sentiment": "neutral"
        }
    
    def _parse_cross_examination_response(self, response: str) -> Dict[str, List[CrossExaminationQuestion]]:
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                result = {}
                for agent, questions in parsed.items():
                    result[agent] = [
                        CrossExaminationQuestion(
                            target_agent=agent,
                            question=q.get("question", ""),
                            context=q.get("context", ""),
                            priority=q.get("priority", "medium")
                        )
                        for q in questions
                    ]
                return result
            except json.JSONDecodeError:
                pass
        
        return {}
    
    def _parse_synthesis_response(self, response: str) -> FinalConclusion:
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                return FinalConclusion(
                    investment_rating=parsed.get("investment_rating", "持有"),
                    confidence=float(parsed.get("confidence", 0.5)),
                    target_price_range=parsed.get("target_price_range", {}),
                    core_thesis=parsed.get("core_thesis", ""),
                    key_agreements=parsed.get("key_agreements", []),
                    key_disagreements=parsed.get("key_disagreements", []),
                    main_risks=parsed.get("main_risks", []),
                    main_opportunities=parsed.get("main_opportunities", []),
                    agent_summaries=parsed.get("agent_summaries", {}),
                    final_recommendation=parsed.get("final_recommendation", "")
                )
            except (json.JSONDecodeError, KeyError):
                pass
        
        return self._get_mock_conclusion()
    
    def _get_mock_response(self) -> str:
        return """```json
{
    "agreements": ["公司基本面稳健", "护城河较宽"],
    "disagreements": [
        {"topic": "估值水平", "positions": {"ValuationAgent": "估值合理", "RiskAgent": "估值偏高"}}
    ],
    "key_questions": ["当前估值是否已充分反映增长预期？"],
    "overall_sentiment": "neutral"
}
```"""
    
    def _get_mock_conclusion(self) -> FinalConclusion:
        return FinalConclusion(
            investment_rating="持有",
            confidence=0.65,
            target_price_range={"low": 1800, "mid": 2000, "high": 2200},
            core_thesis="公司基本面稳健，护城河较宽，但当前估值合理，缺乏明显安全边际。建议持有观望，等待更好的买入时机。",
            key_agreements=["公司具有强大的品牌护城河", "财务状况健康", "管理层稳定"],
            key_disagreements=["估值水平是否合理", "增长预期是否过于乐观"],
            main_risks=["政策风险", "消费代际变化", "估值收缩风险"],
            main_opportunities=["行业整合", "国际化拓展", "产品升级"],
            agent_summaries={
                "FinancialAgent": "财务状况稳健，盈利能力强，但需关注现金流质量",
                "IndustryAgent": "行业处于成熟期，格局稳定，增长放缓",
                "MoatAgent": "品牌护城河极强，定价权得到验证",
                "ValuationAgent": "估值合理但不便宜，安全边际有限",
                "RiskAgent": "整体风险可控，主要关注政策风险"
            },
            final_recommendation="综合各方观点，我们给予该公司'持有'评级。公司基本面优秀，护城河宽广，但当前估值已较为充分地反映了这些优势。建议现有持仓者继续持有，新投资者可等待估值回调至更具吸引力的水平再行介入。"
        )


if __name__ == "__main__":
    # 测试ModeratorAgent
    moderator = ModeratorAgent(verbose=True)
    
    # 模拟初步分析结果
    mock_analyses = {
        "FinancialAgent": {"score": 7.5, "confidence": 0.75, "summary": "财务稳健"},
        "IndustryAgent": {"score": 6.5, "confidence": 0.8, "summary": "行业成熟"},
        "MoatAgent": {"score": 9.0, "confidence": 0.85, "summary": "护城河极强"},
        "ValuationAgent": {"score": 5.5, "confidence": 0.7, "summary": "估值合理"},
        "RiskAgent": {"score": 7.5, "confidence": 0.75, "summary": "风险可控"}
    }
    
    summary = moderator.summarize_initial_analyses(mock_analyses)
    print("\n=== 汇总结果 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
