#!/usr/bin/env python3
"""
V2.9 Module Test Script
测试多Agent辩论系统的核心功能

作者: Manus AI
版本: 2.9
"""

import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

def test_analysis_agents():
    """测试分析Agent模块"""
    print("\n" + "="*60)
    print("测试 1: 分析Agent模块")
    print("="*60)
    
    try:
        from analysis_agents import (
            FinancialAgent, IndustryAgent, MoatAgent, 
            ValuationAgent, RiskAgent, get_all_agents
        )
        
        # 测试Agent实例化
        agents = get_all_agents(verbose=False)
        print(f"✓ 成功创建 {len(agents)} 个专家Agent")
        
        for agent in agents:
            print(f"  - {agent.AGENT_NAME}: {agent.AGENT_ROLE}")
        
        # 测试单个Agent分析
        financial_agent = FinancialAgent(verbose=False)
        test_data = {
            "financial_data": {
                "revenue": 10000000000,
                "net_profit": 1250000000,
                "roe": 0.155
            }
        }
        
        result = financial_agent.analyze(
            stock_code="AAPL",
            company_name="Apple Inc.",
            raw_data=test_data
        )
        
        print(f"\n✓ FinancialAgent分析测试通过")
        print(f"  评分: {result.score}/10")
        print(f"  置信度: {result.confidence:.0%}")
        print(f"  总结: {result.summary[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ 分析Agent测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_debate_engine():
    """测试辩论引擎"""
    print("\n" + "="*60)
    print("测试 2: 辩论引擎")
    print("="*60)
    
    try:
        from debate_engine import (
            ModeratorAgent, DebateEngine, DebateConfig
        )
        from analysis_agents import get_all_agents
        
        # 测试主持人Agent
        moderator = ModeratorAgent(verbose=False)
        print("✓ ModeratorAgent创建成功")
        
        # 模拟初步分析结果
        mock_analyses = {
            "FinancialAgent": {
                "score": 7.5, 
                "confidence": 0.75, 
                "summary": "财务稳健，ROE保持在15%以上",
                "conclusion": "公司财务状况良好"
            },
            "IndustryAgent": {
                "score": 6.5, 
                "confidence": 0.8, 
                "summary": "行业处于成熟期",
                "conclusion": "行业增长放缓但格局稳定"
            },
            "MoatAgent": {
                "score": 9.0, 
                "confidence": 0.85, 
                "summary": "品牌护城河极强",
                "conclusion": "竞争优势可持续"
            },
            "ValuationAgent": {
                "score": 5.5, 
                "confidence": 0.7, 
                "summary": "估值合理但不便宜",
                "conclusion": "安全边际有限"
            },
            "RiskAgent": {
                "score": 7.5, 
                "confidence": 0.75, 
                "summary": "风险可控",
                "conclusion": "主要关注政策风险"
            }
        }
        
        # 测试汇总功能
        summary = moderator.summarize_initial_analyses(mock_analyses)
        print(f"✓ 初步分析汇总成功")
        print(f"  共识点: {len(summary.get('agreements', []))} 个")
        print(f"  分歧点: {len(summary.get('disagreements', []))} 个")
        
        # 测试辩论引擎配置
        config = DebateConfig(max_rounds=1, verbose=False)
        print(f"✓ DebateConfig创建成功 (max_rounds={config.max_rounds})")
        
        return True
        
    except Exception as e:
        print(f"✗ 辩论引擎测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_investment_pipeline():
    """测试投资流水线"""
    print("\n" + "="*60)
    print("测试 3: 投资流水线")
    print("="*60)
    
    try:
        from investment_pipeline import (
            InvestmentPipeline, PipelineConfig
        )
        
        # 测试配置
        config = PipelineConfig(
            verbose=False,
            save_report=False,
            max_debate_rounds=1,
            run_quant_analysis=False,
            run_risk_analysis=False
        )
        print("✓ PipelineConfig创建成功")
        
        # 测试流水线实例化
        pipeline = InvestmentPipeline(config=config)
        print("✓ InvestmentPipeline创建成功")
        
        # 注意：完整的analyze测试需要LLM，这里只测试实例化
        print("✓ 流水线模块加载测试通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 投资流水线测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_module_integration():
    """测试模块集成"""
    print("\n" + "="*60)
    print("测试 4: 模块集成")
    print("="*60)
    
    try:
        # 测试各模块的集成
        from analysis_agents import FinancialAgent, get_all_agents
        from debate_engine import DebateEngine, DebateConfig
        from investment_pipeline import InvestmentPipeline, PipelineConfig
        
        # 测试创建完整的辩论引擎
        agents = get_all_agents(verbose=False)
        config = DebateConfig(max_rounds=1, verbose=False)
        engine = DebateEngine(agents=agents, config=config)
        
        print(f"✓ 成功创建完整的辩论引擎")
        print(f"  - 专家Agent数量: {len(agents)}")
        print(f"  - 最大辩论轮数: {config.max_rounds}")
        
        # 测试流水线配置
        pipeline_config = PipelineConfig(verbose=False, save_report=False)
        pipeline = InvestmentPipeline(config=pipeline_config)
        
        print("✓ 所有核心模块集成测试通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 模块集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print("quant-investor V2.9 模块测试")
    print("="*70)
    
    results = {
        "分析Agent模块": test_analysis_agents(),
        "辩论引擎": test_debate_engine(),
        "投资流水线": test_investment_pipeline(),
        "模块集成": test_module_integration(),
    }
    
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {passed} 通过, {failed} 失败")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
