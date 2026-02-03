#!/usr/bin/env python3
"""
quant-investor V2.9 Module
结构化多维分析框架和多轮多Agent辩论机制

核心特性：
1. 5个专家分析Agent（财务、行业、护城河、估值、风险）
2. 多轮多Agent辩论引擎
3. 端到端投资分析流水线
4. 与V2.3-V2.8模块的深度整合

作者: Manus AI
版本: 2.9
"""

# 支持相对导入和直接导入
try:
    # 作为包导入时使用相对导入
    from .analysis_agents import (
        BaseAnalysisAgent,
        AnalysisResult,
        FinancialAgent,
        IndustryAgent,
        MoatAgent,
        ValuationAgent,
        RiskAgent,
        get_agent,
        get_all_agents,
    )
    from .debate_engine import (
        ModeratorAgent,
        DebateRound,
        CrossExaminationQuestion,
        FinalConclusion,
        DebateEngine,
        DebateConfig,
        create_debate_engine,
    )
    from .investment_pipeline import (
        InvestmentPipeline,
        PipelineConfig,
        quick_analyze,
    )
except ImportError:
    # 直接运行时使用绝对导入
    from analysis_agents import (
        BaseAnalysisAgent,
        AnalysisResult,
        FinancialAgent,
        IndustryAgent,
        MoatAgent,
        ValuationAgent,
        RiskAgent,
        get_agent,
        get_all_agents,
    )
    from debate_engine import (
        ModeratorAgent,
        DebateRound,
        CrossExaminationQuestion,
        FinalConclusion,
        DebateEngine,
        DebateConfig,
        create_debate_engine,
    )
    from investment_pipeline import (
        InvestmentPipeline,
        PipelineConfig,
        quick_analyze,
    )

__version__ = "2.9"

__all__ = [
    # 版本
    "__version__",
    
    # 分析Agents
    "BaseAnalysisAgent",
    "AnalysisResult",
    "FinancialAgent",
    "IndustryAgent",
    "MoatAgent",
    "ValuationAgent",
    "RiskAgent",
    "get_agent",
    "get_all_agents",
    
    # 辩论引擎
    "ModeratorAgent",
    "DebateRound",
    "CrossExaminationQuestion",
    "FinalConclusion",
    "DebateEngine",
    "DebateConfig",
    "create_debate_engine",
    
    # 投资流水线
    "InvestmentPipeline",
    "PipelineConfig",
    "quick_analyze",
]
