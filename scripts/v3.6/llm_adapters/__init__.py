#!/usr/bin/env python3
"""
V3.6 多LLM适配器模块

支持的LLM提供商:
- OpenAI (GPT-4, GPT-3.5等)
- Google Gemini
- DeepSeek
- 阿里千问 (Qwen/DashScope)
- Kimi (Moonshot)

使用示例:
    from v3_6.llm_adapters import create_llm_adapter, LLMProvider
    
    # 创建DeepSeek适配器
    adapter = create_llm_adapter("deepseek", model="deepseek-chat")
    response = adapter.simple_chat("你好")
    
    # 创建千问适配器
    adapter = create_llm_adapter(LLMProvider.QWEN, model="qwen-plus")
    response = adapter.simple_chat("你好")
"""

from .multi_llm_adapter import (
    # 枚举和配置
    LLMProvider,
    LLMConfig,
    LLMResponse,
    PROVIDER_CONFIGS,
    
    # 基类和适配器
    BaseLLMAdapter,
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
    chat_with_kimi
)

from .enhanced_base_agent import (
    AnalysisResult,
    EnhancedBaseAgent,
    create_agent_with_llm
)

__all__ = [
    # 枚举和配置
    "LLMProvider",
    "LLMConfig",
    "LLMResponse",
    "PROVIDER_CONFIGS",
    
    # 基类和适配器
    "BaseLLMAdapter",
    "OpenAIAdapter",
    "GeminiAdapter",
    "DeepSeekAdapter",
    "QwenAdapter",
    "KimiAdapter",
    
    # 工厂函数
    "create_llm_adapter",
    "get_available_providers",
    "get_available_models",
    
    # 便捷函数
    "chat_with_deepseek",
    "chat_with_qwen",
    "chat_with_kimi",
    
    # Agent相关
    "AnalysisResult",
    "EnhancedBaseAgent",
    "create_agent_with_llm"
]

__version__ = "3.6"
