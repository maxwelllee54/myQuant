#!/usr/bin/env python3
"""
V3.6 Enhanced Base Agent - 支持多LLM的增强版分析Agent基类

新增功能:
- 支持OpenAI、Gemini、DeepSeek、千问、Kimi等多种LLM
- 统一的LLM接口适配
- 自动选择可用的LLM提供商

作者: Manus AI
版本: 3.6
"""

import os
import json
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime

# 导入多LLM适配器
from .multi_llm_adapter import (
    create_llm_adapter,
    LLMProvider,
    LLMConfig,
    BaseLLMAdapter,
    get_available_providers,
    PROVIDER_CONFIGS
)


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
    
    # V3.6新增: LLM元信息
    llm_provider: str = ""
    llm_model: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class EnhancedBaseAgent(ABC):
    """
    增强版分析Agent基类 - 支持多LLM
    
    所有专家Agent都继承此类，实现特定领域的分析逻辑。
    支持OpenAI、Gemini、DeepSeek、千问、Kimi等多种LLM。
    """
    
    # 子类必须定义的属性
    AGENT_NAME: str = "EnhancedBaseAgent"
    AGENT_ROLE: str = "基础分析师"
    AGENT_DESCRIPTION: str = "增强版基础分析Agent"
    
    # LLM提供商优先级（按优先级顺序尝试）
    LLM_PRIORITY = [
        LLMProvider.OPENAI,
        LLMProvider.GEMINI,
        LLMProvider.DEEPSEEK,
        LLMProvider.QWEN,
        LLMProvider.KIMI
    ]
    
    def __init__(
        self,
        llm_provider: Optional[Union[str, LLMProvider]] = None,
        llm_model: Optional[str] = None,
        llm_adapter: Optional[BaseLLMAdapter] = None,
        api_key: Optional[str] = None,
        verbose: bool = True
    ):
        """
        初始化Agent
        
        Args:
            llm_provider: LLM提供商 (openai/gemini/deepseek/qwen/kimi)
            llm_model: 模型名称 (可选，使用默认模型)
            llm_adapter: 直接传入的LLM适配器实例 (可选)
            api_key: API密钥 (可选，从环境变量获取)
            verbose: 是否打印详细日志
        """
        self.verbose = verbose
        self.llm_adapter = llm_adapter
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.api_key = api_key
        
        # 初始化LLM
        if self.llm_adapter is None:
            self._init_llm()
    
    def _init_llm(self):
        """初始化LLM适配器"""
        # 如果指定了provider，直接使用
        if self.llm_provider:
            try:
                self.llm_adapter = create_llm_adapter(
                    provider=self.llm_provider,
                    model=self.llm_model,
                    api_key=self.api_key
                )
                if self.verbose:
                    print(f"[{self.AGENT_NAME}] 使用 {self.llm_adapter.provider.value} ({self.llm_adapter.model})")
                return
            except Exception as e:
                if self.verbose:
                    print(f"[{self.AGENT_NAME}] 初始化 {self.llm_provider} 失败: {e}")
        
        # 否则按优先级尝试
        for provider in self.LLM_PRIORITY:
            env_key = PROVIDER_CONFIGS[provider]["env_key"]
            if os.getenv(env_key):
                try:
                    self.llm_adapter = create_llm_adapter(
                        provider=provider,
                        model=self.llm_model,
                        api_key=self.api_key
                    )
                    if self.verbose:
                        print(f"[{self.AGENT_NAME}] 自动选择 {provider.value} ({self.llm_adapter.model})")
                    return
                except Exception as e:
                    if self.verbose:
                        print(f"[{self.AGENT_NAME}] 尝试 {provider.value} 失败: {e}")
                    continue
        
        # 没有可用的LLM
        if self.verbose:
            print(f"[{self.AGENT_NAME}] 警告: 未找到可用的LLM，将使用模拟响应")
        self.llm_adapter = None
    
    def set_llm(
        self,
        provider: Union[str, LLMProvider],
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        动态切换LLM提供商
        
        Args:
            provider: LLM提供商
            model: 模型名称
            api_key: API密钥
        """
        self.llm_adapter = create_llm_adapter(
            provider=provider,
            model=model,
            api_key=api_key
        )
        if self.verbose:
            print(f"[{self.AGENT_NAME}] 切换到 {self.llm_adapter.provider.value} ({self.llm_adapter.model})")
    
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
            llm_info = f"{self.llm_adapter.provider.value}/{self.llm_adapter.model}" if self.llm_adapter else "Mock"
            print(f"[{self.AGENT_NAME}] 开始分析 {stock_code} ({company_name}) [LLM: {llm_info}]...")
        
        # 1. 准备数据
        prepared_data = self._prepare_data(raw_data, quant_results)
        
        # 2. 构建Prompt
        system_prompt = self._get_system_prompt()
        user_prompt = self._build_analysis_prompt(stock_code, company_name, prepared_data)
        
        # 3. 调用LLM
        llm_response = self._call_llm(system_prompt, user_prompt)
        
        # 4. 解析响应为结构化结果
        result = self._parse_response(stock_code, company_name, llm_response)
        
        # 添加LLM元信息
        if self.llm_adapter:
            result.llm_provider = self.llm_adapter.provider.value
            result.llm_model = self.llm_adapter.model
        
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
        if self.llm_adapter is None:
            # 如果没有LLM适配器，返回模拟响应
            return self._get_mock_response()
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.llm_adapter.chat(
                messages=messages,
                temperature=0.3,
                max_tokens=4000
            )
            return response.content
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


# 便捷函数：创建使用特定LLM的Agent
def create_agent_with_llm(
    agent_class,
    provider: Union[str, LLMProvider],
    model: Optional[str] = None,
    **kwargs
):
    """
    创建使用特定LLM的Agent实例
    
    Args:
        agent_class: Agent类
        provider: LLM提供商
        model: 模型名称
        **kwargs: 其他Agent参数
    
    Returns:
        Agent实例
    """
    return agent_class(
        llm_provider=provider,
        llm_model=model,
        **kwargs
    )


if __name__ == "__main__":
    print("=== EnhancedBaseAgent 模块测试 ===")
    print(f"支持的LLM提供商: {get_available_providers()}")
    print(f"AnalysisResult 字段: {list(AnalysisResult.__dataclass_fields__.keys())}")
