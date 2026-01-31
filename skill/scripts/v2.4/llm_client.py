#!/usr/bin/env python3
"""
LLM Client for quant-investor V2.4
统一封装对不同LLM API的调用，提供成本控制和可靠性保障
"""

import os
import json
import hashlib
import time
from typing import Dict, List, Optional, Any, Union
from pathlib import Path


class LLMClient:
    """
    统一的LLM客户端，支持多种模型和API
    """
    
    def __init__(
        self,
        default_model: str = "gpt-4",
        cache_dir: str = "/home/ubuntu/.cache/quant_investor/llm",
        enable_cache: bool = True,
        max_retries: int = 3
    ):
        """
        初始化LLM客户端
        
        Args:
            default_model: 默认使用的模型
            cache_dir: 缓存目录
            enable_cache: 是否启用缓存
            max_retries: 最大重试次数
        """
        self.default_model = default_model
        self.cache_dir = Path(cache_dir)
        self.enable_cache = enable_cache
        self.max_retries = max_retries
        
        # 创建缓存目录
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化API客户端
        self._init_clients()
    
    def _init_clients(self):
        """初始化各个LLM的API客户端"""
        # OpenAI (GPT-4, GPT-3.5)
        try:
            from openai import OpenAI
            base_url = os.getenv("OPENAI_API_BASE")
            if base_url:
                self.openai_client = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url=base_url
                )
            else:
                self.openai_client = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY")
                )
        except Exception as e:
            print(f"Warning: OpenAI client initialization failed: {e}")
            self.openai_client = None
        
        # Google Gemini
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.gemini_client = genai
        except Exception as e:
            print(f"Warning: Gemini client initialization failed: {e}")
            self.gemini_client = None
    
    def _get_cache_key(self, prompt: str, model: str, temperature: float) -> str:
        """生成缓存键"""
        content = f"{prompt}|{model}|{temperature}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[str]:
        """从缓存加载结果"""
        if not self.enable_cache:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('response')
            except Exception as e:
                print(f"Warning: Failed to load cache: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, response: str):
        """保存结果到缓存"""
        if not self.enable_cache:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'response': response,
                    'timestamp': time.time()
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        system_message: Optional[str] = None
    ) -> str:
        """
        生成LLM响应
        
        Args:
            prompt: 用户提示词
            model: 使用的模型（如果为None，使用default_model）
            temperature: 温度参数（0-1），越低越确定性
            max_tokens: 最大生成token数
            system_message: 系统消息（可选）
        
        Returns:
            LLM生成的响应文本
        """
        model = model or self.default_model
        
        # 检查缓存
        cache_key = self._get_cache_key(prompt, model, temperature)
        cached_response = self._load_from_cache(cache_key)
        if cached_response:
            print(f"[LLMClient] Using cached response for model={model}")
            return cached_response
        
        # 根据模型类型调用相应的API
        for attempt in range(self.max_retries):
            try:
                if model.startswith("gpt"):
                    response = self._call_openai(prompt, model, temperature, max_tokens, system_message)
                elif model.startswith("gemini"):
                    response = self._call_gemini(prompt, model, temperature, max_tokens, system_message)
                else:
                    raise ValueError(f"Unsupported model: {model}")
                
                # 保存到缓存
                self._save_to_cache(cache_key, response)
                
                return response
            
            except Exception as e:
                print(f"[LLMClient] Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # 指数退避
    
    def _call_openai(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        system_message: Optional[str]
    ) -> str:
        """调用OpenAI API"""
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized")
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def _call_gemini(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        system_message: Optional[str]
    ) -> str:
        """调用Google Gemini API"""
        if not self.gemini_client:
            raise RuntimeError("Gemini client not initialized")
        
        # 构建完整的prompt
        full_prompt = prompt
        if system_message:
            full_prompt = f"{system_message}\n\n{prompt}"
        
        # 创建模型实例
        gemini_model = self.gemini_client.GenerativeModel(model)
        
        # 生成响应
        response = gemini_model.generate_content(
            full_prompt,
            generation_config={
                'temperature': temperature,
                'max_output_tokens': max_tokens
            }
        )
        
        return response.text
    
    def generate_json(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        生成JSON格式的响应
        
        Args:
            prompt: 用户提示词
            model: 使用的模型
            temperature: 温度参数（默认较低以确保输出稳定）
            max_tokens: 最大生成token数
        
        Returns:
            解析后的JSON对象
        """
        # 在prompt中强调JSON输出
        json_prompt = f"{prompt}\n\n请以JSON格式返回结果，不要包含任何其他文本。"
        
        response = self.generate(
            prompt=json_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # 尝试解析JSON
        try:
            # 移除可能的markdown代码块标记
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"[LLMClient] Failed to parse JSON: {e}")
            print(f"[LLMClient] Raw response: {response}")
            raise
    
    def estimate_cost(self, prompt: str, model: str, max_tokens: int = 2000) -> float:
        """
        估算API调用成本（美元）
        
        Args:
            prompt: 提示词
            model: 模型名称
            max_tokens: 最大生成token数
        
        Returns:
            估算成本（美元）
        """
        # 简单估算：假设1个token约等于4个字符
        input_tokens = len(prompt) / 4
        output_tokens = max_tokens
        
        # 价格表（每1000 tokens的价格，单位：美元）
        pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "gemini-pro": {"input": 0.00025, "output": 0.0005},
            "gemini-2.0-flash": {"input": 0.0001, "output": 0.0002},
        }
        
        if model not in pricing:
            print(f"[LLMClient] Warning: Unknown model {model}, using gpt-4 pricing")
            model = "gpt-4"
        
        cost = (
            input_tokens / 1000 * pricing[model]["input"] +
            output_tokens / 1000 * pricing[model]["output"]
        )
        
        return cost
    
    def clear_cache(self):
        """清空缓存"""
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            print(f"[LLMClient] Cache cleared: {self.cache_dir}")


def demo():
    """演示LLM客户端的使用"""
    print("=== LLM Client Demo ===\n")
    
    # 初始化客户端
    client = LLMClient(default_model="gpt-3.5-turbo")
    
    # 测试1: 简单对话
    print("Test 1: Simple conversation")
    prompt = "请用一句话解释什么是量化投资。"
    response = client.generate(prompt, temperature=0.7)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}\n")
    
    # 测试2: JSON输出
    print("Test 2: JSON output")
    prompt = """
    请分析贵州茅台这只股票，并以JSON格式返回以下信息：
    {
        "stock_name": "股票名称",
        "industry": "所属行业",
        "investment_rating": "投资评级（买入/持有/卖出）",
        "key_strengths": ["优势1", "优势2"],
        "key_risks": ["风险1", "风险2"]
    }
    """
    try:
        response = client.generate_json(prompt)
        print(f"JSON Response: {json.dumps(response, ensure_ascii=False, indent=2)}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # 测试3: 成本估算
    print("Test 3: Cost estimation")
    cost = client.estimate_cost(prompt, "gpt-3.5-turbo")
    print(f"Estimated cost: ${cost:.6f}\n")
    
    # 测试4: 缓存机制
    print("Test 4: Cache mechanism")
    print("First call (no cache):")
    start = time.time()
    response1 = client.generate("1+1等于几？", temperature=0.0)
    time1 = time.time() - start
    print(f"Response: {response1}")
    print(f"Time: {time1:.2f}s\n")
    
    print("Second call (with cache):")
    start = time.time()
    response2 = client.generate("1+1等于几？", temperature=0.0)
    time2 = time.time() - start
    print(f"Response: {response2}")
    print(f"Time: {time2:.2f}s\n")
    
    print(f"Speedup: {time1/time2:.1f}x")


if __name__ == "__main__":
    demo()
