#!/usr/bin/env python3
"""
V2.9 Analysis Agents Module
结构化多维分析Agent模块

作者: Manus AI
版本: 2.9
"""

from .base_agent import BaseAnalysisAgent, AnalysisResult, format_financial_data, format_price_data
from .financial_agent import FinancialAgent
from .industry_agent import IndustryAgent
from .moat_agent import MoatAgent
from .valuation_agent import ValuationAgent
from .risk_agent import RiskAgent

__all__ = [
    # 基类
    "BaseAnalysisAgent",
    "AnalysisResult",
    
    # 工具函数
    "format_financial_data",
    "format_price_data",
    
    # 专家Agents
    "FinancialAgent",
    "IndustryAgent",
    "MoatAgent",
    "ValuationAgent",
    "RiskAgent",
]

# Agent角色映射
AGENT_REGISTRY = {
    "financial": FinancialAgent,
    "industry": IndustryAgent,
    "moat": MoatAgent,
    "valuation": ValuationAgent,
    "risk": RiskAgent,
}


def get_agent(agent_type: str, **kwargs) -> BaseAnalysisAgent:
    """
    工厂函数：根据类型获取Agent实例
    
    Args:
        agent_type: Agent类型 (financial, industry, moat, valuation, risk)
        **kwargs: 传递给Agent构造函数的参数
    
    Returns:
        BaseAnalysisAgent: Agent实例
    """
    if agent_type not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(AGENT_REGISTRY.keys())}")
    
    return AGENT_REGISTRY[agent_type](**kwargs)


def get_all_agents(**kwargs) -> list:
    """
    获取所有专家Agent的实例列表
    
    Args:
        **kwargs: 传递给Agent构造函数的参数
    
    Returns:
        list: Agent实例列表
    """
    return [agent_class(**kwargs) for agent_class in AGENT_REGISTRY.values()]
