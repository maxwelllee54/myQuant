#!/usr/bin/env python3
"""
V2.9 Debate Engine - Core Engine
多轮多Agent辩论引擎

作者: Manus AI
版本: 2.9
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from .moderator_agent import ModeratorAgent, DebateRound, FinalConclusion


@dataclass
class DebateConfig:
    """辩论配置"""
    max_rounds: int = 2  # 最大辩论轮数（不含初始分析）
    min_confidence_threshold: float = 0.6  # 最低置信度阈值
    enable_quant_integration: bool = True  # 是否整合量化结果
    verbose: bool = True


class DebateEngine:
    """
    多轮多Agent辩论引擎
    
    核心流程：
    1. Round 0: 各专家Agent独立分析
    2. Round 1-N: 主持人组织交叉质询和回应
    3. Final: 主持人综合形成最终结论
    """
    
    def __init__(
        self,
        agents: List[Any],
        config: Optional[DebateConfig] = None,
        llm_client=None
    ):
        """
        初始化辩论引擎
        
        Args:
            agents: 专家Agent列表
            config: 辩论配置
            llm_client: LLM客户端
        """
        self.agents = agents
        self.config = config or DebateConfig()
        self.moderator = ModeratorAgent(llm_client=llm_client, verbose=self.config.verbose)
        self.debate_history: List[DebateRound] = []
        self.initial_analyses: Dict[str, Any] = {}
        
        if self.config.verbose:
            print(f"[DebateEngine] 初始化完成，共 {len(agents)} 个专家Agent")
            print(f"[DebateEngine] 最大辩论轮数: {self.config.max_rounds}")
    
    def run_debate(
        self,
        stock_code: str,
        company_name: str,
        raw_data: Dict[str, Any],
        quant_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        运行完整的辩论流程
        
        Args:
            stock_code: 股票代码
            company_name: 公司名称
            raw_data: 原始数据
            quant_results: 量化分析结果
        
        Returns:
            完整的辩论结果
        """
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"开始辩论: {company_name} ({stock_code})")
            print(f"{'='*60}\n")
        
        start_time = datetime.now()
        
        # ========== Round 0: 独立分析 ==========
        if self.config.verbose:
            print("\n" + "="*40)
            print("Round 0: 各专家独立分析")
            print("="*40)
        
        self.initial_analyses = self._run_initial_analyses(
            stock_code, company_name, raw_data, quant_results
        )
        
        self.debate_history.append(DebateRound(
            round_number=0,
            phase="initial_analysis",
            content={
                agent_name: analysis.to_dict() if hasattr(analysis, 'to_dict') else analysis
                for agent_name, analysis in self.initial_analyses.items()
            },
            timestamp=datetime.now().isoformat()
        ))
        
        # ========== 汇总初步分析 ==========
        if self.config.verbose:
            print("\n" + "="*40)
            print("主持人汇总初步分析")
            print("="*40)
        
        summary = self.moderator.summarize_initial_analyses(self.initial_analyses)
        
        # ========== Round 1-N: 交叉质询与回应 ==========
        for round_num in range(1, self.config.max_rounds + 1):
            if self.config.verbose:
                print("\n" + "="*40)
                print(f"Round {round_num}: 交叉质询与回应")
                print("="*40)
            
            # 生成质询问题
            questions = self.moderator.generate_cross_examination_questions(
                summary, self.debate_history
            )
            
            # 各Agent回应
            rebuttals = self._collect_rebuttals(questions)
            
            self.debate_history.append(DebateRound(
                round_number=round_num,
                phase="cross_examination",
                content={
                    "questions": {
                        agent: [asdict(q) for q in qs]
                        for agent, qs in questions.items()
                    },
                    "rebuttals": rebuttals
                },
                timestamp=datetime.now().isoformat()
            ))
            
            # 更新汇总
            summary = self._update_summary(summary, rebuttals)
        
        # ========== Final: 综合结论 ==========
        if self.config.verbose:
            print("\n" + "="*40)
            print("Final: 综合形成最终结论")
            print("="*40)
        
        final_conclusion = self.moderator.synthesize_final_conclusion(
            stock_code, company_name, self.debate_history, quant_results
        )
        
        # 生成报告
        final_report = self.moderator.generate_final_report(
            stock_code, company_name, final_conclusion, self.debate_history
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"辩论完成！耗时: {duration:.1f}秒")
            print(f"最终评级: {final_conclusion.investment_rating}")
            print(f"置信度: {final_conclusion.confidence:.0%}")
            print(f"{'='*60}\n")
        
        return {
            "stock_code": stock_code,
            "company_name": company_name,
            "debate_duration_seconds": duration,
            "total_rounds": len(self.debate_history),
            "initial_analyses": {
                name: analysis.to_dict() if hasattr(analysis, 'to_dict') else analysis
                for name, analysis in self.initial_analyses.items()
            },
            "debate_history": [
                asdict(round_data) if isinstance(round_data, DebateRound) else round_data
                for round_data in self.debate_history
            ],
            "final_conclusion": asdict(final_conclusion),
            "final_report": final_report
        }
    
    def _run_initial_analyses(
        self,
        stock_code: str,
        company_name: str,
        raw_data: Dict[str, Any],
        quant_results: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """运行各Agent的初步分析"""
        analyses = {}
        
        for agent in self.agents:
            agent_name = getattr(agent, 'AGENT_NAME', agent.__class__.__name__)
            
            if self.config.verbose:
                print(f"\n[{agent_name}] 开始分析...")
            
            try:
                result = agent.analyze(
                    stock_code=stock_code,
                    company_name=company_name,
                    raw_data=raw_data,
                    quant_results=quant_results
                )
                analyses[agent_name] = result
                
                if self.config.verbose:
                    score = result.score if hasattr(result, 'score') else result.get('score', 'N/A')
                    print(f"[{agent_name}] 分析完成，评分: {score}/10")
                    
            except Exception as e:
                if self.config.verbose:
                    print(f"[{agent_name}] 分析失败: {e}")
                analyses[agent_name] = {
                    "error": str(e),
                    "score": 5.0,
                    "confidence": 0.3,
                    "summary": f"分析失败: {e}"
                }
        
        return analyses
    
    def _collect_rebuttals(
        self,
        questions: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """收集各Agent对质询问题的回应"""
        rebuttals = {}
        
        for agent in self.agents:
            agent_name = getattr(agent, 'AGENT_NAME', agent.__class__.__name__)
            
            if agent_name not in questions or not questions[agent_name]:
                continue
            
            agent_questions = questions[agent_name]
            question_texts = [
                q.question if hasattr(q, 'question') else q.get('question', '')
                for q in agent_questions
            ]
            
            if self.config.verbose:
                print(f"\n[{agent_name}] 回应 {len(question_texts)} 个问题...")
            
            try:
                original_analysis = self.initial_analyses.get(agent_name)
                rebuttal = agent.rebut(
                    questions=question_texts,
                    debate_history=self.debate_history,
                    original_analysis=original_analysis
                )
                rebuttals[agent_name] = rebuttal
                
            except Exception as e:
                if self.config.verbose:
                    print(f"[{agent_name}] 回应失败: {e}")
                rebuttals[agent_name] = {
                    "error": str(e),
                    "responses": ["回应失败"]
                }
        
        return rebuttals
    
    def _update_summary(
        self,
        current_summary: Dict[str, Any],
        rebuttals: Dict[str, Any]
    ) -> Dict[str, Any]:
        """根据回应更新汇总"""
        # 简单实现：检查是否有Agent改变了立场
        updated_summary = current_summary.copy()
        
        for agent_name, rebuttal in rebuttals.items():
            if isinstance(rebuttal, dict) and rebuttal.get("stance_changed"):
                if self.config.verbose:
                    print(f"[{agent_name}] 调整了观点")
                # 可以在这里更新summary中的相关内容
        
        return updated_summary
    
    def get_debate_summary(self) -> Dict[str, Any]:
        """获取辩论摘要"""
        if not self.debate_history:
            return {"status": "no_debate_yet"}
        
        return {
            "total_rounds": len(self.debate_history),
            "agents": [
                getattr(agent, 'AGENT_NAME', agent.__class__.__name__)
                for agent in self.agents
            ],
            "phases": [round_data.phase for round_data in self.debate_history]
        }


def create_debate_engine(
    llm_client=None,
    max_rounds: int = 2,
    verbose: bool = True
) -> DebateEngine:
    """
    工厂函数：创建配置好的辩论引擎
    
    Args:
        llm_client: LLM客户端
        max_rounds: 最大辩论轮数
        verbose: 是否打印详细日志
    
    Returns:
        DebateEngine: 配置好的辩论引擎
    """
    # 导入所有Agent
    from analysis_agents import (
        FinancialAgent, IndustryAgent, MoatAgent, ValuationAgent, RiskAgent
    )
    
    # 创建Agent实例
    agents = [
        FinancialAgent(llm_client=llm_client, verbose=verbose),
        IndustryAgent(llm_client=llm_client, verbose=verbose),
        MoatAgent(llm_client=llm_client, verbose=verbose),
        ValuationAgent(llm_client=llm_client, verbose=verbose),
        RiskAgent(llm_client=llm_client, verbose=verbose),
    ]
    
    # 创建配置
    config = DebateConfig(
        max_rounds=max_rounds,
        verbose=verbose
    )
    
    # 创建引擎
    return DebateEngine(agents=agents, config=config, llm_client=llm_client)


if __name__ == "__main__":
    # 简单测试
    print("DebateEngine 模块加载成功")
    
    # 测试配置
    config = DebateConfig(max_rounds=2, verbose=True)
    print(f"默认配置: max_rounds={config.max_rounds}")
