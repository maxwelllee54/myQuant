#!/usr/bin/env python3
"""
V3.6 多LLM统一接口适配器

支持的LLM提供商:
- OpenAI (GPT-4, GPT-3.5等)
- Google Gemini
- DeepSeek
- 阿里千问 (Qwen/DashScope)
- Kimi (Moonshot)

设计参考: TradingAgents-CN项目
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum


class LLMProvider(str, Enum):
    """LLM提供商枚举"""
    OPENAI = "openai"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"  # 阿里千问/DashScope
    KIMI = "kimi"  # Moonshot


@dataclass
class LLMConfig:
    """LLM配置"""
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60


@dataclass
class LLMResponse:
    """LLM响应"""
    content: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    raw_response: Optional[Any] = None


# 各提供商的默认配置
PROVIDER_CONFIGS = {
    LLMProvider.OPENAI: {
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4",
        "models": ["gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "o1", "o1-mini"]
    },
    LLMProvider.GEMINI: {
        "base_url": None,  # Gemini使用专用SDK
        "env_key": "GEMINI_API_KEY",
        "default_model": "gemini-2.5-flash",
        "models": ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]
    },
    LLMProvider.DEEPSEEK: {
        "base_url": "https://api.deepseek.com",
        "env_key": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
        "models": ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"]
    },
    LLMProvider.QWEN: {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_key": "DASHSCOPE_API_KEY",
        "default_model": "qwen-plus",
        "models": ["qwen-turbo", "qwen-plus", "qwen-max", "qwen3-max", "qwen-long"]
    },
    LLMProvider.KIMI: {
        "base_url": "https://api.moonshot.cn/v1",
        "env_key": "MOONSHOT_API_KEY",
        "default_model": "moonshot-v1-8k",
        "models": ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"]
    }
}


class BaseLLMAdapter(ABC):
    """LLM适配器基类"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider = config.provider
        self.model = config.model
        
        # 获取API Key
        self.api_key = config.api_key or self._get_api_key_from_env()
        if not self.api_key:
            raise ValueError(f"API key not found for {self.provider.value}. "
                           f"Please set {PROVIDER_CONFIGS[self.provider]['env_key']} environment variable.")
    
    def _get_api_key_from_env(self) -> Optional[str]:
        """从环境变量获取API Key"""
        env_key = PROVIDER_CONFIGS[self.provider]["env_key"]
        api_key = os.getenv(env_key)
        
        # 验证API Key是否有效（排除占位符）
        if api_key and self._is_valid_api_key(api_key):
            return api_key
        return None
    
    def _is_valid_api_key(self, key: str) -> bool:
        """验证API Key是否有效"""
        if not key or len(key) <= 10:
            return False
        if key.startswith('your_') or key.startswith('your-'):
            return False
        if key.endswith('_here') or key.endswith('-here'):
            return False
        if '...' in key or 'xxx' in key.lower():
            return False
        return True
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """发送聊天请求"""
        pass
    
    def simple_chat(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """简单聊天接口"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.chat(messages, **kwargs)
        return response.content


class OpenAIAdapter(BaseLLMAdapter):
    """OpenAI适配器"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url or PROVIDER_CONFIGS[LLMProvider.OPENAI]["base_url"]
        
        # 初始化OpenAI客户端
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """发送聊天请求"""
        start_time = time.time()
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens)
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model,
            provider=self.provider.value,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
            latency_ms=latency_ms,
            raw_response=response
        )


class GeminiAdapter(BaseLLMAdapter):
    """Google Gemini适配器"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        
        # 初始化Gemini客户端
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
        except ImportError:
            raise ImportError("Please install google-genai: pip install google-genai")
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """发送聊天请求"""
        from google.genai import types
        
        start_time = time.time()
        
        # 转换消息格式
        contents = []
        system_instruction = None
        
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(text=msg["content"])]
                ))
            elif msg["role"] == "assistant":
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part(text=msg["content"])]
                ))
        
        # 配置生成参数
        config = types.GenerateContentConfig(
            temperature=kwargs.get("temperature", self.config.temperature),
            max_output_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            system_instruction=system_instruction
        )
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # 提取token使用量
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
            output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
        
        return LLMResponse(
            content=response.text,
            model=self.model,
            provider=self.provider.value,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency_ms=latency_ms,
            raw_response=response
        )


class DeepSeekAdapter(BaseLLMAdapter):
    """DeepSeek适配器（使用OpenAI兼容接口）"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url or PROVIDER_CONFIGS[LLMProvider.DEEPSEEK]["base_url"]
        
        # 使用OpenAI客户端（DeepSeek兼容OpenAI接口）
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """发送聊天请求"""
        start_time = time.time()
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens)
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model,
            provider=self.provider.value,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
            latency_ms=latency_ms,
            raw_response=response
        )


class QwenAdapter(BaseLLMAdapter):
    """阿里千问适配器（使用OpenAI兼容接口）"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url or PROVIDER_CONFIGS[LLMProvider.QWEN]["base_url"]
        
        # 使用OpenAI客户端（千问兼容OpenAI接口）
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """发送聊天请求"""
        start_time = time.time()
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens)
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model,
            provider=self.provider.value,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
            latency_ms=latency_ms,
            raw_response=response
        )


class KimiAdapter(BaseLLMAdapter):
    """Kimi/Moonshot适配器（使用OpenAI兼容接口）"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url or PROVIDER_CONFIGS[LLMProvider.KIMI]["base_url"]
        
        # 使用OpenAI客户端（Kimi兼容OpenAI接口）
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """发送聊天请求"""
        start_time = time.time()
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens)
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model,
            provider=self.provider.value,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
            latency_ms=latency_ms,
            raw_response=response
        )


# 适配器工厂
ADAPTER_CLASSES = {
    LLMProvider.OPENAI: OpenAIAdapter,
    LLMProvider.GEMINI: GeminiAdapter,
    LLMProvider.DEEPSEEK: DeepSeekAdapter,
    LLMProvider.QWEN: QwenAdapter,
    LLMProvider.KIMI: KimiAdapter
}


def create_llm_adapter(
    provider: Union[str, LLMProvider],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> BaseLLMAdapter:
    """
    创建LLM适配器的工厂函数
    
    Args:
        provider: LLM提供商名称或枚举
        model: 模型名称（可选，使用默认模型）
        api_key: API密钥（可选，从环境变量获取）
        **kwargs: 其他配置参数
    
    Returns:
        对应的LLM适配器实例
    
    Example:
        >>> adapter = create_llm_adapter("deepseek", model="deepseek-chat")
        >>> response = adapter.simple_chat("你好，请介绍一下你自己")
        >>> print(response)
    """
    # 转换provider为枚举
    if isinstance(provider, str):
        provider = LLMProvider(provider.lower())
    
    # 获取默认模型
    if model is None:
        model = PROVIDER_CONFIGS[provider]["default_model"]
    
    # 创建配置
    config = LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=kwargs.get("base_url"),
        temperature=kwargs.get("temperature", 0.7),
        max_tokens=kwargs.get("max_tokens", 4096),
        timeout=kwargs.get("timeout", 60)
    )
    
    # 创建适配器
    adapter_class = ADAPTER_CLASSES.get(provider)
    if adapter_class is None:
        raise ValueError(f"Unsupported provider: {provider}")
    
    return adapter_class(config)


def get_available_providers() -> List[str]:
    """获取所有可用的LLM提供商"""
    return [p.value for p in LLMProvider]


def get_available_models(provider: Union[str, LLMProvider]) -> List[str]:
    """获取指定提供商的可用模型列表"""
    if isinstance(provider, str):
        provider = LLMProvider(provider.lower())
    return PROVIDER_CONFIGS[provider]["models"]


# 便捷函数
def chat_with_deepseek(prompt: str, model: str = "deepseek-chat", **kwargs) -> str:
    """使用DeepSeek进行对话"""
    adapter = create_llm_adapter(LLMProvider.DEEPSEEK, model=model, **kwargs)
    return adapter.simple_chat(prompt, **kwargs)


def chat_with_qwen(prompt: str, model: str = "qwen-plus", **kwargs) -> str:
    """使用阿里千问进行对话"""
    adapter = create_llm_adapter(LLMProvider.QWEN, model=model, **kwargs)
    return adapter.simple_chat(prompt, **kwargs)


def chat_with_kimi(prompt: str, model: str = "moonshot-v1-8k", **kwargs) -> str:
    """使用Kimi进行对话"""
    adapter = create_llm_adapter(LLMProvider.KIMI, model=model, **kwargs)
    return adapter.simple_chat(prompt, **kwargs)


if __name__ == "__main__":
    # 测试代码
    print("=== 多LLM适配器测试 ===\n")
    
    print("可用的LLM提供商:")
    for provider in get_available_providers():
        models = get_available_models(provider)
        print(f"  - {provider}: {models}")
    
    print("\n测试适配器创建...")
    
    # 测试各适配器（需要设置对应的API Key）
    test_providers = [
        ("openai", "OPENAI_API_KEY"),
        ("gemini", "GEMINI_API_KEY"),
        ("deepseek", "DEEPSEEK_API_KEY"),
        ("qwen", "DASHSCOPE_API_KEY"),
        ("kimi", "MOONSHOT_API_KEY")
    ]
    
    for provider, env_key in test_providers:
        if os.getenv(env_key):
            print(f"\n测试 {provider}...")
            try:
                adapter = create_llm_adapter(provider)
                response = adapter.simple_chat("你好，请用一句话介绍你自己")
                print(f"  ✅ {provider} 响应: {response[:100]}...")
            except Exception as e:
                print(f"  ❌ {provider} 错误: {e}")
        else:
            print(f"\n跳过 {provider} (未设置 {env_key})")
