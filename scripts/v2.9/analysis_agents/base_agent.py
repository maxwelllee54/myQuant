#!/usr/bin/env python3
"""
V2.9 Analysis Agents - Base Agent
结构化多维分析Agent基类

作者: Manus AI
版本: 2.9
"""

import os
import json
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class AnalysisResult:
    """分析结果的标准化数据结构"""
    agent_name: str
    agent_role: str
    timestamp: str
    stock_code: str
    
    # 核心分析结论
    summary: str  # 一句话总结
    conclusion: str  # 详细结论 (1-2段)
    
    # 结构化评分 (1-10分)
    score: float
    confidence: float  # 置信度 (0-1)
    
    # 关键发现
    key_findings: List[Dict[str, Any]]  # [{finding, evidence, impact}]
    
    # 风险与机会
    risks: List[str]
    opportunities: List[str]
    
    # 原始分析数据 (Agent特有的详细分析)
    detailed_analysis: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class BaseAnalysisAgent(ABC):
    """
    分析Agent基类
    
    所有专家Agent都继承此类，实现特定领域的分析逻辑。
    """
    
    # 子类必须定义的属性
    AGENT_NAME: str = "BaseAgent"
    AGENT_ROLE: str = "基础分析师"
    AGENT_DESCRIPTION: str = "基础分析Agent"
    
    def __init__(self, llm_client=None, verbose: bool = True):
        """
        初始化Agent
        
        Args:
            llm_client: LLM客户端实例 (如果为None，将使用默认的OpenAI客户端)
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
                    print(f"[{self.AGENT_NAME}] Warning: Failed to init OpenAI client: {e}")
                self.llm_client = None
                self.llm_model = None
    
    def analyze(
        self,
        stock_code: str,
        company_name: str,
        raw_data: Dict[str, Any],
        quant_results: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """
        执行分析并返回结构化结果
        
        Args:
            stock_code: 股票代码
            company_name: 公司名称
            raw_data: 原始数据字典 (包含财务、行情、行业等数据)
            quant_results: 量化分析结果 (可选)
        
        Returns:
            AnalysisResult: 结构化的分析结果
        """
        if self.verbose:
            print(f"[{self.AGENT_NAME}] 开始分析 {stock_code} ({company_name})...")
        
        # 1. 准备数据
        prepared_data = self._prepare_data(raw_data, quant_results)
        
        # 2. 构建Prompt
        system_prompt = self._get_system_prompt()
        user_prompt = self._build_analysis_prompt(stock_code, company_name, prepared_data)
        
        # 3. 调用LLM
        llm_response = self._call_llm(system_prompt, user_prompt)
        
        # 4. 解析响应为结构化结果
        result = self._parse_response(stock_code, company_name, llm_response)
        
        if self.verbose:
            print(f"[{self.AGENT_NAME}] 分析完成. 评分: {result.score}/10, 置信度: {result.confidence:.0%}")
        
        return result
    
    def rebut(
        self,
        questions: List[str],
        debate_history: List[Dict[str, Any]],
        original_analysis: AnalysisResult
    ) -> Dict[str, Any]:
        """
        回应主持人的质询问题
        
        Args:
            questions: 主持人提出的问题列表
            debate_history: 辩论历史
            original_analysis: 本Agent之前的分析结果
        
        Returns:
            回应字典
        """
        if self.verbose:
            print(f"[{self.AGENT_NAME}] 回应 {len(questions)} 个质询问题...")
        
        system_prompt = self._get_rebuttal_system_prompt()
        user_prompt = self._build_rebuttal_prompt(questions, debate_history, original_analysis)
        
        llm_response = self._call_llm(system_prompt, user_prompt)
        
        # 解析回应
        rebuttal = self._parse_rebuttal_response(llm_response)
        
        return rebuttal
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """调用LLM获取响应"""
        if self.llm_client is None:
            # 如果没有LLM客户端，返回模拟响应
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
                print(f"[{self.AGENT_NAME}] LLM调用失败: {e}")
            return self._get_mock_response()
    
    def _parse_response(self, stock_code: str, company_name: str, response: str) -> AnalysisResult:
        """解析LLM响应为AnalysisResult"""
        # 尝试从响应中提取JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                parsed = self._extract_structured_data(response)
        else:
            parsed = self._extract_structured_data(response)
        
        return AnalysisResult(
            agent_name=self.AGENT_NAME,
            agent_role=self.AGENT_ROLE,
            timestamp=datetime.now().isoformat(),
            stock_code=stock_code,
            summary=parsed.get("summary", "分析完成"),
            conclusion=parsed.get("conclusion", response[:500]),
            score=float(parsed.get("score", 5.0)),
            confidence=float(parsed.get("confidence", 0.5)),
            key_findings=parsed.get("key_findings", []),
            risks=parsed.get("risks", []),
            opportunities=parsed.get("opportunities", []),
            detailed_analysis=parsed.get("detailed_analysis", {})
        )
    
    def _extract_structured_data(self, text: str) -> Dict[str, Any]:
        """从非结构化文本中提取关键信息"""
        # 简单的启发式提取
        result = {
            "summary": "",
            "conclusion": text[:500] if len(text) > 500 else text,
            "score": 5.0,
            "confidence": 0.5,
            "key_findings": [],
            "risks": [],
            "opportunities": [],
            "detailed_analysis": {"raw_text": text}
        }
        
        # 尝试提取评分
        score_match = re.search(r'评分[：:]\s*(\d+(?:\.\d+)?)', text)
        if score_match:
            result["score"] = float(score_match.group(1))
        
        return result
    
    def _parse_rebuttal_response(self, response: str) -> Dict[str, Any]:
        """解析回应响应"""
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        return {
            "responses": [response],
            "stance_changed": False,
            "new_score": None,
            "additional_evidence": []
        }
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """获取系统Prompt，由子类实现"""
        pass
    
    @abstractmethod
    def _build_analysis_prompt(
        self,
        stock_code: str,
        company_name: str,
        prepared_data: Dict[str, Any]
    ) -> str:
        """构建分析Prompt，由子类实现"""
        pass
    
    @abstractmethod
    def _prepare_data(
        self,
        raw_data: Dict[str, Any],
        quant_results: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """准备分析所需的数据，由子类实现"""
        pass
    
    def _get_rebuttal_system_prompt(self) -> str:
        """获取回应质询的系统Prompt"""
        return f"""你是一位资深的{self.AGENT_ROLE}。
在投资委员会的辩论中，主持人针对你之前的分析提出了一些质疑和问题。
你需要：
1. 认真回应每个问题，提供更多证据或澄清你的观点
2. 如果其他分析师的观点有道理，你可以适当调整自己的结论
3. 保持专业和客观，承认不确定性

请以JSON格式回复，包含以下字段：
- responses: 对每个问题的回应列表
- stance_changed: 是否调整了观点 (true/false)
- new_score: 如果调整了观点，新的评分 (1-10)
- additional_evidence: 补充的证据或数据
"""
    
    def _build_rebuttal_prompt(
        self,
        questions: List[str],
        debate_history: List[Dict[str, Any]],
        original_analysis: AnalysisResult
    ) -> str:
        """构建回应质询的Prompt"""
        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        
        return f"""## 你之前的分析结论

{original_analysis.conclusion}

评分: {original_analysis.score}/10
置信度: {original_analysis.confidence:.0%}

## 主持人的质询问题

{questions_text}

## 请回应上述问题
"""
    
    def _get_mock_response(self) -> str:
        """获取模拟响应（用于测试）"""
        return """```json
{
    "summary": "这是一个模拟分析结果",
    "conclusion": "由于未配置LLM客户端，返回模拟分析结果。在实际使用中，此处将包含详细的分析内容。",
    "score": 6.0,
    "confidence": 0.7,
    "key_findings": [
        {"finding": "模拟发现1", "evidence": "模拟证据", "impact": "中等"}
    ],
    "risks": ["模拟风险1"],
    "opportunities": ["模拟机会1"],
    "detailed_analysis": {}
}
```"""


# 辅助函数
def format_financial_data(data: Dict[str, Any]) -> str:
    """格式化财务数据为可读文本"""
    if not data:
        return "无财务数据"
    
    lines = []
    for key, value in data.items():
        if isinstance(value, (int, float)):
            if abs(value) >= 1e8:
                lines.append(f"- {key}: {value/1e8:.2f}亿")
            elif abs(value) >= 1e4:
                lines.append(f"- {key}: {value/1e4:.2f}万")
            else:
                lines.append(f"- {key}: {value:.2f}")
        else:
            lines.append(f"- {key}: {value}")
    
    return "\n".join(lines)


def format_price_data(data: Dict[str, Any]) -> str:
    """格式化行情数据为可读文本"""
    if not data:
        return "无行情数据"
    
    lines = []
    for key, value in data.items():
        lines.append(f"- {key}: {value}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # 测试基类
    print("BaseAnalysisAgent 模块加载成功")
    print(f"AnalysisResult 字段: {AnalysisResult.__dataclass_fields__.keys()}")
