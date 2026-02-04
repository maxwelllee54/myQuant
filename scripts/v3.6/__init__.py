#!/usr/bin/env python3
"""
V3.6 多LLM支持模块

核心功能:
- 统一的多LLM适配器接口
- 支持OpenAI、Gemini、DeepSeek、千问、Kimi等多种LLM
- 增强版分析Agent基类，支持动态切换LLM

使用示例:
    from v3_6 import create_llm_adapter, LLMProvider
    
    # 创建DeepSeek适配器
    adapter = create_llm_adapter("deepseek")
    response = adapter.simple_chat("分析一下苹果公司的投资价值")
"""

from .llm_adapters import (
    # 核心类和枚举
    LLMProvider,
    LLMConfig,
    LLMResponse,
    BaseLLMAdapter,
    
    # 适配器
    OpenAIAdapter,
    GeminiAdapter,
    DeepSeekAdapter,
    QwenAdapter,
    KimiAdapter,
    
    # 工厂函数
    create_llm_adapter,
    get_available_providers,
    get_available_models,
    
    # 便捷函数
    chat_with_deepseek,
    chat_with_qwen,
    chat_with_kimi,
    
    # Agent相关
    AnalysisResult,
    EnhancedBaseAgent,
    create_agent_with_llm
)

__all__ = [
    "LLMProvider",
    "LLMConfig",
    "LLMResponse",
    "BaseLLMAdapter",
    "OpenAIAdapter",
    "GeminiAdapter",
    "DeepSeekAdapter",
    "QwenAdapter",
    "KimiAdapter",
    "create_llm_adapter",
    "get_available_providers",
    "get_available_models",
    "chat_with_deepseek",
    "chat_with_qwen",
    "chat_with_kimi",
    "AnalysisResult",
    "EnhancedBaseAgent",
    "create_agent_with_llm"
]

__version__ = "3.6"
