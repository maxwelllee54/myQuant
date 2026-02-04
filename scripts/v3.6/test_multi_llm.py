#!/usr/bin/env python3
"""
V3.6 多LLM适配器测试脚本

测试内容:
1. 适配器创建和基本功能
2. 各LLM提供商的连接测试
3. 增强版Agent基类测试
"""

import os
import sys

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v3_6.llm_adapters import (
    LLMProvider,
    create_llm_adapter,
    get_available_providers,
    get_available_models,
    EnhancedBaseAgent,
    AnalysisResult
)


def test_module_import():
    """测试模块导入"""
    print("=" * 60)
    print("测试1: 模块导入")
    print("=" * 60)
    
    print("✅ 模块导入成功")
    print(f"   可用的LLM提供商: {get_available_providers()}")
    
    for provider in LLMProvider:
        models = get_available_models(provider)
        print(f"   - {provider.value}: {models}")
    
    return True


def test_adapter_creation():
    """测试适配器创建"""
    print("\n" + "=" * 60)
    print("测试2: 适配器创建")
    print("=" * 60)
    
    # 测试各提供商的适配器创建
    test_cases = [
        ("openai", "OPENAI_API_KEY"),
        ("gemini", "GEMINI_API_KEY"),
        ("deepseek", "DEEPSEEK_API_KEY"),
        ("qwen", "DASHSCOPE_API_KEY"),
        ("kimi", "MOONSHOT_API_KEY")
    ]
    
    results = {}
    
    for provider, env_key in test_cases:
        if os.getenv(env_key):
            try:
                adapter = create_llm_adapter(provider)
                results[provider] = {
                    "status": "✅ 创建成功",
                    "model": adapter.model,
                    "provider": adapter.provider.value
                }
            except Exception as e:
                results[provider] = {
                    "status": f"❌ 创建失败: {e}",
                    "model": None,
                    "provider": provider
                }
        else:
            results[provider] = {
                "status": f"⏭️ 跳过 (未设置 {env_key})",
                "model": None,
                "provider": provider
            }
    
    for provider, result in results.items():
        print(f"   {provider}: {result['status']}")
        if result['model']:
            print(f"      模型: {result['model']}")
    
    return True


def test_llm_chat():
    """测试LLM对话功能"""
    print("\n" + "=" * 60)
    print("测试3: LLM对话功能")
    print("=" * 60)
    
    # 按优先级尝试可用的LLM
    priority = [
        ("openai", "OPENAI_API_KEY"),
        ("gemini", "GEMINI_API_KEY"),
        ("deepseek", "DEEPSEEK_API_KEY"),
        ("qwen", "DASHSCOPE_API_KEY"),
        ("kimi", "MOONSHOT_API_KEY")
    ]
    
    for provider, env_key in priority:
        if os.getenv(env_key):
            print(f"\n   测试 {provider} 对话...")
            try:
                adapter = create_llm_adapter(provider)
                
                # 简单对话测试
                response = adapter.simple_chat(
                    "请用一句话介绍你自己",
                    system_prompt="你是一个友好的AI助手"
                )
                
                print(f"   ✅ {provider} 对话成功")
                print(f"      响应: {response[:100]}...")
                print(f"      模型: {adapter.model}")
                
                return True
                
            except Exception as e:
                print(f"   ❌ {provider} 对话失败: {e}")
                continue
    
    print("   ⚠️ 没有可用的LLM进行对话测试")
    return False


def test_enhanced_agent():
    """测试增强版Agent基类"""
    print("\n" + "=" * 60)
    print("测试4: 增强版Agent基类")
    print("=" * 60)
    
    # 创建一个简单的测试Agent
    class TestAgent(EnhancedBaseAgent):
        AGENT_NAME = "TestAgent"
        AGENT_ROLE = "测试分析师"
        AGENT_DESCRIPTION = "用于测试的分析Agent"
        
        def _get_system_prompt(self):
            return "你是一个测试分析师，请简要分析给定的信息。"
        
        def _build_analysis_prompt(self, stock_code, company_name, prepared_data):
            return f"请简要分析 {stock_code} ({company_name}) 的投资价值。"
        
        def _prepare_data(self, raw_data, quant_results):
            return raw_data or {}
    
    # 测试Agent创建
    try:
        agent = TestAgent(verbose=True)
        print(f"   ✅ TestAgent 创建成功")
        
        if agent.llm_adapter:
            print(f"      LLM: {agent.llm_adapter.provider.value}/{agent.llm_adapter.model}")
        else:
            print(f"      LLM: Mock (无可用LLM)")
        
        # 测试分析功能
        result = agent.analyze(
            stock_code="AAPL",
            company_name="苹果公司",
            raw_data={"price": 180, "pe": 28}
        )
        
        print(f"   ✅ 分析完成")
        print(f"      评分: {result.score}/10")
        print(f"      置信度: {result.confidence:.0%}")
        print(f"      LLM提供商: {result.llm_provider or 'Mock'}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ TestAgent 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_llm_switching():
    """测试LLM动态切换"""
    print("\n" + "=" * 60)
    print("测试5: LLM动态切换")
    print("=" * 60)
    
    # 找到至少两个可用的LLM
    available = []
    providers = [
        ("openai", "OPENAI_API_KEY"),
        ("gemini", "GEMINI_API_KEY"),
        ("deepseek", "DEEPSEEK_API_KEY"),
        ("qwen", "DASHSCOPE_API_KEY"),
        ("kimi", "MOONSHOT_API_KEY")
    ]
    
    for provider, env_key in providers:
        if os.getenv(env_key):
            available.append(provider)
    
    if len(available) < 2:
        print(f"   ⏭️ 跳过 (需要至少2个可用LLM，当前: {len(available)})")
        return True
    
    # 创建Agent并测试切换
    class TestAgent(EnhancedBaseAgent):
        AGENT_NAME = "SwitchTestAgent"
        AGENT_ROLE = "切换测试分析师"
        
        def _get_system_prompt(self):
            return "你是一个测试分析师。"
        
        def _build_analysis_prompt(self, stock_code, company_name, prepared_data):
            return f"请用一句话评价 {stock_code}。"
        
        def _prepare_data(self, raw_data, quant_results):
            return {}
    
    try:
        # 使用第一个LLM创建Agent
        agent = TestAgent(llm_provider=available[0], verbose=True)
        print(f"   初始LLM: {agent.llm_adapter.provider.value}")
        
        # 切换到第二个LLM
        agent.set_llm(available[1])
        print(f"   切换后LLM: {agent.llm_adapter.provider.value}")
        
        print(f"   ✅ LLM切换成功")
        return True
        
    except Exception as e:
        print(f"   ❌ LLM切换失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("V3.6 多LLM适配器测试")
    print("=" * 60)
    
    tests = [
        ("模块导入", test_module_import),
        ("适配器创建", test_adapter_creation),
        ("LLM对话", test_llm_chat),
        ("增强版Agent", test_enhanced_agent),
        ("LLM切换", test_llm_switching)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"   ❌ {name} 测试异常: {e}")
            results.append((name, False))
    
    # 打印测试总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {name}: {status}")
    
    print(f"\n   总计: {passed}/{total} 通过")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
