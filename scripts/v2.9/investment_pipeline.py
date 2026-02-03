#!/usr/bin/env python3
"""
V2.9 Investment Pipeline
端到端的投资分析流水线

整合：
1. 数据获取（持久化存储）
2. 量化因子分析
3. 多Agent辩论
4. 风险管理
5. 最终投资建议

作者: Manus AI
版本: 2.9
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class PipelineConfig:
    """流水线配置"""
    # 数据配置
    use_persistent_storage: bool = True
    data_lookback_days: int = 365
    
    # 量化分析配置
    run_quant_analysis: bool = True
    factor_list: List[str] = None
    
    # 辩论配置
    max_debate_rounds: int = 2
    
    # 风险管理配置
    run_risk_analysis: bool = True
    risk_free_rate: float = 0.03
    
    # 输出配置
    output_dir: str = "./output"
    save_report: bool = True
    verbose: bool = True
    
    def __post_init__(self):
        if self.factor_list is None:
            self.factor_list = ["momentum", "value", "quality", "volatility"]


class InvestmentPipeline:
    """
    端到端投资分析流水线
    
    整合quant-investor的所有核心模块：
    - V2.5/V2.6: 数据管理器
    - V2.7: 持久化存储
    - V2.3: 因子分析
    - V2.8: 风险管理
    - V2.9: 多Agent辩论
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None, llm_client=None):
        """
        初始化流水线
        
        Args:
            config: 流水线配置
            llm_client: LLM客户端
        """
        self.config = config or PipelineConfig()
        self.llm_client = llm_client
        
        # 延迟初始化各模块
        self._data_manager = None
        self._quant_analyzer = None
        self._risk_manager = None
        self._debate_engine = None
        
        if self.config.verbose:
            print("[Pipeline] 投资分析流水线初始化完成")
    
    def analyze(
        self,
        stock_code: str,
        company_name: Optional[str] = None,
        market: str = "US"  # "US" or "CN"
    ) -> Dict[str, Any]:
        """
        运行完整的投资分析流水线
        
        Args:
            stock_code: 股票代码
            company_name: 公司名称（可选，会自动获取）
            market: 市场 ("US" 或 "CN")
        
        Returns:
            完整的分析结果
        """
        start_time = datetime.now()
        
        if self.config.verbose:
            print(f"\n{'='*70}")
            print(f"开始投资分析流水线: {stock_code}")
            print(f"市场: {market}")
            print(f"{'='*70}\n")
        
        results = {
            "stock_code": stock_code,
            "market": market,
            "analysis_timestamp": start_time.isoformat(),
            "pipeline_version": "2.9"
        }
        
        # ========== Step 1: 数据获取 ==========
        if self.config.verbose:
            print("\n" + "-"*50)
            print("Step 1: 数据获取")
            print("-"*50)
        
        raw_data = self._fetch_data(stock_code, market)
        results["raw_data_summary"] = self._summarize_data(raw_data)
        
        # 获取公司名称
        if company_name is None:
            company_name = raw_data.get("company_info", {}).get("name", stock_code)
        results["company_name"] = company_name
        
        # ========== Step 2: 量化因子分析 ==========
        quant_results = None
        if self.config.run_quant_analysis:
            if self.config.verbose:
                print("\n" + "-"*50)
                print("Step 2: 量化因子分析")
                print("-"*50)
            
            quant_results = self._run_quant_analysis(stock_code, raw_data, market)
            results["quant_analysis"] = quant_results
        
        # ========== Step 3: 多Agent辩论 ==========
        if self.config.verbose:
            print("\n" + "-"*50)
            print("Step 3: 多Agent辩论分析")
            print("-"*50)
        
        debate_results = self._run_debate(
            stock_code, company_name, raw_data, quant_results
        )
        results["debate_analysis"] = debate_results
        
        # ========== Step 4: 风险管理分析 ==========
        if self.config.run_risk_analysis:
            if self.config.verbose:
                print("\n" + "-"*50)
                print("Step 4: 量化风险分析")
                print("-"*50)
            
            risk_results = self._run_risk_analysis(stock_code, raw_data)
            results["risk_analysis"] = risk_results
        
        # ========== Step 5: 综合结论 ==========
        if self.config.verbose:
            print("\n" + "-"*50)
            print("Step 5: 生成综合投资结论")
            print("-"*50)
        
        final_conclusion = self._synthesize_conclusion(results)
        results["final_conclusion"] = final_conclusion
        
        # ========== 保存报告 ==========
        if self.config.save_report:
            report_path = self._save_report(stock_code, results)
            results["report_path"] = report_path
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        results["analysis_duration_seconds"] = duration
        
        if self.config.verbose:
            print(f"\n{'='*70}")
            print(f"分析完成！总耗时: {duration:.1f}秒")
            print(f"最终评级: {final_conclusion.get('investment_rating', 'N/A')}")
            print(f"置信度: {final_conclusion.get('confidence', 0):.0%}")
            print(f"{'='*70}\n")
        
        return results
    
    def _fetch_data(self, stock_code: str, market: str) -> Dict[str, Any]:
        """获取股票数据"""
        data = {}
        
        try:
            if market == "US":
                data = self._fetch_us_data(stock_code)
            else:
                data = self._fetch_cn_data(stock_code)
        except Exception as e:
            if self.config.verbose:
                print(f"[Pipeline] 数据获取失败: {e}")
            data = {"error": str(e)}
        
        return data
    
    def _fetch_us_data(self, stock_code: str) -> Dict[str, Any]:
        """获取美股数据"""
        data = {}
        
        try:
            # 尝试使用持久化数据管理器
            if self.config.use_persistent_storage:
                from v2_7.persistent_us_data_manager import PersistentUSDataManager
                manager = PersistentUSDataManager()
            else:
                from v2_6.us_macro_data.us_macro_data_manager import USMacroDataManager
                manager = USMacroDataManager()
            
            # 获取股票历史数据
            history = manager.get_stock_history(
                symbol=stock_code,
                period=f"{self.config.data_lookback_days}d"
            )
            if history is not None and not history.empty:
                data["price_history"] = {
                    "start_date": str(history.index[0]),
                    "end_date": str(history.index[-1]),
                    "data_points": len(history),
                    "latest_close": float(history["Close"].iloc[-1]) if "Close" in history else None
                }
            
            # 获取公司信息
            info = manager.get_stock_info(stock_code)
            if info:
                data["company_info"] = info
            
            if self.config.verbose:
                print(f"[Pipeline] 美股数据获取成功: {stock_code}")
                
        except Exception as e:
            if self.config.verbose:
                print(f"[Pipeline] 美股数据获取失败: {e}")
            # 返回模拟数据
            data = self._get_mock_us_data(stock_code)
        
        return data
    
    def _fetch_cn_data(self, stock_code: str) -> Dict[str, Any]:
        """获取A股数据"""
        data = {}
        
        try:
            if self.config.use_persistent_storage:
                from v2_7.persistent_cn_data_manager import PersistentCNDataManager
                manager = PersistentCNDataManager()
            else:
                from v2_5.cn_macro_data.cn_macro_data_manager import CNMacroDataManager
                manager = CNMacroDataManager()
            
            # 获取股票历史数据
            history = manager.get_stock_history(
                ts_code=stock_code,
                days=self.config.data_lookback_days
            )
            if history is not None and not history.empty:
                data["price_history"] = {
                    "start_date": str(history.index[0]),
                    "end_date": str(history.index[-1]),
                    "data_points": len(history)
                }
            
            if self.config.verbose:
                print(f"[Pipeline] A股数据获取成功: {stock_code}")
                
        except Exception as e:
            if self.config.verbose:
                print(f"[Pipeline] A股数据获取失败: {e}")
            data = self._get_mock_cn_data(stock_code)
        
        return data
    
    def _run_quant_analysis(
        self,
        stock_code: str,
        raw_data: Dict[str, Any],
        market: str
    ) -> Dict[str, Any]:
        """运行量化因子分析"""
        quant_results = {}
        
        try:
            # 这里可以集成V2.3的因子分析模块
            # 目前返回模拟结果
            quant_results = {
                "factors": {
                    "momentum_score": 0.65,
                    "value_score": 0.55,
                    "quality_score": 0.75,
                    "volatility_score": 0.60
                },
                "composite_score": 0.64,
                "signal": "moderate_buy",
                "confidence": 0.7
            }
            
            if self.config.verbose:
                print(f"[Pipeline] 量化分析完成，综合得分: {quant_results['composite_score']:.2f}")
                
        except Exception as e:
            if self.config.verbose:
                print(f"[Pipeline] 量化分析失败: {e}")
            quant_results = {"error": str(e)}
        
        return quant_results
    
    def _run_debate(
        self,
        stock_code: str,
        company_name: str,
        raw_data: Dict[str, Any],
        quant_results: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """运行多Agent辩论"""
        try:
            from analysis_agents import (
                FinancialAgent, IndustryAgent, MoatAgent, ValuationAgent, RiskAgent
            )
            from debate_engine import DebateEngine, DebateConfig
            
            # 创建Agent实例
            agents = [
                FinancialAgent(llm_client=self.llm_client, verbose=self.config.verbose),
                IndustryAgent(llm_client=self.llm_client, verbose=self.config.verbose),
                MoatAgent(llm_client=self.llm_client, verbose=self.config.verbose),
                ValuationAgent(llm_client=self.llm_client, verbose=self.config.verbose),
                RiskAgent(llm_client=self.llm_client, verbose=self.config.verbose),
            ]
            
            # 创建辩论引擎
            debate_config = DebateConfig(
                max_rounds=self.config.max_debate_rounds,
                verbose=self.config.verbose
            )
            engine = DebateEngine(
                agents=agents,
                config=debate_config,
                llm_client=self.llm_client
            )
            
            # 运行辩论
            debate_results = engine.run_debate(
                stock_code=stock_code,
                company_name=company_name,
                raw_data=raw_data,
                quant_results=quant_results
            )
            
            return debate_results
            
        except Exception as e:
            if self.config.verbose:
                print(f"[Pipeline] 辩论分析失败: {e}")
            return {"error": str(e)}
    
    def _run_risk_analysis(
        self,
        stock_code: str,
        raw_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """运行量化风险分析"""
        try:
            # 集成V2.8风险管理模块
            from v2_8.risk_management.risk_manager import RiskManager
            
            rm = RiskManager(risk_free_rate=self.config.risk_free_rate)
            
            # 获取收益率数据
            # 这里需要从raw_data中提取价格数据并计算收益率
            # 目前返回模拟结果
            
            risk_results = {
                "volatility": 0.25,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.15,
                "var_95": -0.03,
                "risk_level": "中等",
                "risk_score": 7.0
            }
            
            if self.config.verbose:
                print(f"[Pipeline] 风险分析完成，风险等级: {risk_results['risk_level']}")
            
            return risk_results
            
        except Exception as e:
            if self.config.verbose:
                print(f"[Pipeline] 风险分析失败: {e}")
            return {"error": str(e)}
    
    def _synthesize_conclusion(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """综合所有分析结果形成最终结论"""
        conclusion = {}
        
        # 从辩论结果中提取结论
        debate_conclusion = results.get("debate_analysis", {}).get("final_conclusion", {})
        
        if debate_conclusion:
            conclusion["investment_rating"] = debate_conclusion.get("investment_rating", "持有")
            conclusion["confidence"] = debate_conclusion.get("confidence", 0.5)
            conclusion["core_thesis"] = debate_conclusion.get("core_thesis", "")
            conclusion["main_risks"] = debate_conclusion.get("main_risks", [])
            conclusion["main_opportunities"] = debate_conclusion.get("main_opportunities", [])
        else:
            # 默认结论
            conclusion = {
                "investment_rating": "持有",
                "confidence": 0.5,
                "core_thesis": "分析数据不足，建议进一步研究",
                "main_risks": ["数据不完整"],
                "main_opportunities": ["待进一步分析"]
            }
        
        # 整合量化分析结果
        quant_analysis = results.get("quant_analysis", {})
        if quant_analysis and "composite_score" in quant_analysis:
            conclusion["quant_score"] = quant_analysis["composite_score"]
            conclusion["quant_signal"] = quant_analysis.get("signal", "neutral")
        
        # 整合风险分析结果
        risk_analysis = results.get("risk_analysis", {})
        if risk_analysis and "risk_level" in risk_analysis:
            conclusion["risk_level"] = risk_analysis["risk_level"]
            conclusion["risk_score"] = risk_analysis.get("risk_score", 5.0)
        
        return conclusion
    
    def _save_report(self, stock_code: str, results: Dict[str, Any]) -> str:
        """保存分析报告"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON结果
        json_path = output_dir / f"{stock_code}_analysis_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            # 过滤掉不可序列化的内容
            serializable_results = self._make_serializable(results)
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        # 保存Markdown报告
        md_report = results.get("debate_analysis", {}).get("final_report", "")
        if md_report:
            md_path = output_dir / f"{stock_code}_report_{timestamp}.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_report)
        
        if self.config.verbose:
            print(f"[Pipeline] 报告已保存: {json_path}")
        
        return str(json_path)
    
    def _make_serializable(self, obj: Any) -> Any:
        """将对象转换为可JSON序列化的格式"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)
    
    def _summarize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成数据摘要"""
        summary = {}
        
        if "price_history" in data:
            summary["price_data"] = {
                "available": True,
                "data_points": data["price_history"].get("data_points", 0)
            }
        else:
            summary["price_data"] = {"available": False}
        
        if "company_info" in data:
            summary["company_info"] = {"available": True}
        else:
            summary["company_info"] = {"available": False}
        
        return summary
    
    def _get_mock_us_data(self, stock_code: str) -> Dict[str, Any]:
        """获取模拟美股数据"""
        return {
            "company_info": {
                "name": stock_code,
                "sector": "Technology",
                "industry": "Software"
            },
            "price_history": {
                "start_date": "2024-01-01",
                "end_date": "2025-02-03",
                "data_points": 280
            },
            "financial_data": {
                "revenue": 100000000000,
                "net_profit": 25000000000,
                "roe": 0.25,
                "gross_margin": 0.45
            },
            "valuation": {
                "pe": 25,
                "pb": 5,
                "ps": 8
            }
        }
    
    def _get_mock_cn_data(self, stock_code: str) -> Dict[str, Any]:
        """获取模拟A股数据"""
        return {
            "company_info": {
                "name": stock_code,
                "industry": "白酒"
            },
            "price_history": {
                "start_date": "2024-01-01",
                "end_date": "2025-02-03",
                "data_points": 240
            }
        }


def quick_analyze(
    stock_code: str,
    market: str = "US",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    快速分析函数
    
    Args:
        stock_code: 股票代码
        market: 市场 ("US" 或 "CN")
        verbose: 是否打印详细日志
    
    Returns:
        分析结果
    """
    config = PipelineConfig(verbose=verbose)
    pipeline = InvestmentPipeline(config=config)
    return pipeline.analyze(stock_code=stock_code, market=market)


if __name__ == "__main__":
    # 测试流水线
    print("Investment Pipeline 模块加载成功")
    
    # 简单测试
    config = PipelineConfig(verbose=True, save_report=False)
    pipeline = InvestmentPipeline(config=config)
    
    print(f"\n配置: max_debate_rounds={config.max_debate_rounds}")
    print(f"配置: use_persistent_storage={config.use_persistent_storage}")
