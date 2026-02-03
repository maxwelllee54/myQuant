#!/usr/bin/env python3
"""
V2.9 Debate Engine Module
多轮多Agent辩论引擎模块

作者: Manus AI
版本: 2.9
"""

from .moderator_agent import (
    ModeratorAgent,
    DebateRound,
    CrossExaminationQuestion,
    FinalConclusion
)
from .debate_engine import (
    DebateEngine,
    DebateConfig,
    create_debate_engine
)

__all__ = [
    # 主持人Agent
    "ModeratorAgent",
    
    # 数据结构
    "DebateRound",
    "CrossExaminationQuestion",
    "FinalConclusion",
    
    # 辩论引擎
    "DebateEngine",
    "DebateConfig",
    
    # 工厂函数
    "create_debate_engine",
]
