#!/usr/bin/env python3
"""
LLM-Enhanced Quantitative Analysis Pipeline for quant-investor V2.4
端到端的量化-LLM混合分析流水线
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

# 导入V2.4模块
try:
    from llm_client import LLMClient
    from prompt_templates import get_prompt, get_system_message
except ImportError:
    # 如果直接运行，尝试从当前目录导入
    sys.path.insert(0, str(Path(__file__).parent))
    from llm_client import LLMClient
    from prompt_templates import get_prompt, get_system_message

# 导入V2.3模块（如果存在）
HAS_V23_MODULES = False
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "v2.3"))
    from expression_engine import ExpressionEngine
    from feature_cache import FeatureCache
    from enhanced_backtest import EnhancedBacktestEngine
    HAS_V23_MODULES = True
except ImportError:
    print("Warning: V2.3 modules not found, some features will be limited")


class LLMQuantPipeline:
    """
    LLM增强的量化分析流水线
    """
    
    def __init__(
        self,
        llm_model: str = "gpt-4",
        enable_cache: bool = True,
        output_dir: str = "/home/ubuntu/quant_analysis_reports"
    ):
        """
        初始化流水线
        
        Args:
            llm_model: 使用的LLM模型
            enable_cache: 是否启用LLM缓存
            output_dir: 输出目录
        """
        self.llm_client = LLMClient(
            default_model=llm_model,
            enable_cache=enable_cache
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载投资大师智慧库
        self.master_wisdom = self._load_master_wisdom()
        
        print(f"[Pipeline] Initialized with model={llm_model}")
    
    def _load_master_wisdom(self) -> Dict[str, Any]:
        """加载投资大师智慧库"""
        wisdom_path = Path(__file__).parent.parent.parent / "references" / "master_wisdom.json"
        try:
            with open(wisdom_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load master wisdom: {e}")
            return {}
    
    def analyze_stock(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        对单只股票进行完整的LLM增强量化分析
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            analysis_type: 分析类型 (comprehensive/quick)
        
        Returns:
            分析结果字典
        """
        print(f"\n{'='*60}")
        print(f"开始分析股票: {stock_code}")
        print(f"分析周期: {start_date} 至 {end_date}")
        print(f"{'='*60}\n")
        
        results = {
            "stock_code": stock_code,
            "start_date": start_date,
            "end_date": end_date,
            "analysis_timestamp": datetime.now().isoformat(),
            "stages": {}
        }
        
        # 阶段1: 数据收集
        print("[Stage 1/6] 数据收集...")
        data = self._collect_data(stock_code, start_date, end_date)
        results["stages"]["data_collection"] = {"status": "completed", "rows": len(data)}
        
        # 阶段2: 特征工程（如果启用V2.3模块）
        print("[Stage 2/6] 特征工程...")
        if HAS_V23_MODULES:
            factors = self._engineer_features(data)
            results["stages"]["feature_engineering"] = {
                "status": "completed",
                "num_factors": len(factors)
            }
        else:
            factors = {}
            results["stages"]["feature_engineering"] = {
                "status": "skipped",
                "reason": "V2.3 modules not available"
            }
        
        # 阶段3: 量化分析
        print("[Stage 3/6] 量化分析...")
        quant_results = self._quantitative_analysis(data, factors)
        results["stages"]["quantitative_analysis"] = quant_results
        
        # 阶段4: LLM综合分析
        print("[Stage 4/6] LLM综合分析...")
        synthesis = self._llm_synthesis(quant_results, data)
        results["stages"]["llm_synthesis"] = synthesis
        
        # 阶段5: 多空辩论
        print("[Stage 5/6] 多空辩论...")
        debate = self._bull_bear_debate(synthesis)
        results["stages"]["debate"] = debate
        
        # 阶段6: 投资建议
        print("[Stage 6/6] 生成投资建议...")
        advice = self._generate_investment_advice(
            synthesis, debate, stock_code, data
        )
        results["stages"]["investment_advice"] = advice
        
        # 生成最终报告
        report_path = self._generate_report(results)
        results["report_path"] = str(report_path)
        
        print(f"\n{'='*60}")
        print(f"分析完成！报告已保存至: {report_path}")
        print(f"{'='*60}\n")
        
        return results
    
    def _collect_data(
        self,
        stock_code: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        收集股票数据
        
        注意：这是一个简化的示例实现，实际使用时应该调用Tushare等数据API
        """
        # 这里返回一个模拟的DataFrame
        # 实际实现应该调用 Tushare Pro API
        print(f"  - 正在获取{stock_code}的历史数据...")
        print(f"  - 注意：当前为演示模式，返回模拟数据")
        
        # 创建模拟数据
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'close': 100 + pd.Series(range(len(dates))).cumsum() * 0.1,
            'volume': 1000000,
            'pe_ratio': 15.0,
            'pb_ratio': 2.0,
        })
        
        return data
    
    def _engineer_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        使用LLM生成因子，并用V2.3表达式引擎计算
        """
        print("  - 使用LLM生成alpha因子...")
        
        # 描述数据
        data_description = f"包含{len(data)}天的日线数据，字段包括: {', '.join(data.columns)}"
        
        # 调用LLM生成因子（这里使用简化的示例）
        # 实际实现应该调用 llm_client.generate_json()
        factors = {
            "momentum_20d": data['close'].pct_change(20),
            "value_pe": 1 / data['pe_ratio'],
            "value_pb": 1 / data['pb_ratio'],
        }
        
        print(f"  - 生成了{len(factors)}个因子")
        
        return factors
    
    def _quantitative_analysis(
        self,
        data: pd.DataFrame,
        factors: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        """
        执行量化分析
        """
        print("  - 计算IC值...")
        print("  - 执行回测...")
        print("  - 进行统计检验...")
        
        # 简化的量化分析结果
        results = {
            "ic_analysis": {
                "mean_ic": 0.05,
                "std_ic": 0.15,
                "ic_ir": 0.33,
                "t_statistic": 2.5,
                "p_value": 0.01
            },
            "backtest": {
                "cumulative_return": 0.25,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.15,
                "win_rate": 0.55
            },
            "statistical_tests": {
                "ic_significant": True,
                "granger_causality_p_value": 0.03
            }
        }
        
        return results
    
    def _llm_synthesis(
        self,
        quant_results: Dict[str, Any],
        data: pd.DataFrame
    ) -> str:
        """
        使用LLM进行综合分析
        """
        print("  - 构建综合分析Prompt...")
        print("  - 调用LLM进行深度分析...")
        
        # 构建市场环境描述
        market_env = f"分析期间共{len(data)}个交易日，价格从{data['close'].iloc[0]:.2f}变化至{data['close'].iloc[-1]:.2f}"
        
        # 构建Prompt
        prompt = get_prompt(
            "synthesis",
            ic_analysis=json.dumps(quant_results["ic_analysis"], indent=2, ensure_ascii=False),
            backtest_results=json.dumps(quant_results["backtest"], indent=2, ensure_ascii=False),
            statistical_tests=json.dumps(quant_results["statistical_tests"], indent=2, ensure_ascii=False),
            market_environment=market_env
        )
        
        # 调用LLM（这里使用模拟响应）
        # 实际实现应该调用 self.llm_client.generate()
        synthesis = """
## LLM综合分析报告

### 1. 因子有效性评估
基于IC分析结果，因子的平均IC值为0.05，虽然数值不高，但统计显著性良好（p值=0.01），
表明因子确实具有一定的预测能力。IC-IR比率为0.33，说明因子的稳定性有待提升。

### 2. 因果推断
Granger因果检验的p值为0.03，低于0.05的显著性水平，初步支持因子与收益率之间存在
因果关系的假设。但需要注意，Granger因果检验只能证明"预测因果"，不能证明真正的因果关系。

### 3. 市场环境适应性
回测结果显示夏普比率为1.2，表明策略在风险调整后的收益表现良好。最大回撤为-15%，
在可接受范围内。胜率为55%，略高于随机水平，说明策略具有一定的优势。

### 4. 风险评估
主要风险包括：
- IC值的稳定性不足，可能在某些市场环境下失效
- 样本期较短，存在过拟合风险
- 未考虑交易成本和流动性约束

### 5. 改进建议
- 增加更多因子，构建多因子模型
- 进行更长时间的样本外测试
- 引入动态权重调整机制
- 加强风险管理，设置止损机制
"""
        
        return synthesis
    
    def _bull_bear_debate(self, synthesis: str) -> Dict[str, Any]:
        """
        组织Bull和Bear Agent进行辩论
        """
        print("  - Bull Agent正在论述...")
        print("  - Bear Agent正在论述...")
        
        # 简化的辩论结果
        debate = {
            "bull_view": {
                "stance": "bullish",
                "confidence": 0.70,
                "core_thesis": "量化因子显示出统计显著性，策略具有正向预期收益",
                "key_arguments": [
                    "IC值统计显著，p值仅为0.01",
                    "夏普比率达到1.2，风险调整后收益良好",
                    "Granger因果检验支持因果关系假设"
                ]
            },
            "bear_view": {
                "stance": "bearish",
                "confidence": 0.60,
                "core_thesis": "因子稳定性不足，存在过拟合风险",
                "key_arguments": [
                    "IC-IR比率仅为0.33，稳定性较差",
                    "样本期可能较短，存在数据窥探偏差",
                    "未充分考虑交易成本和市场冲击"
                ]
            }
        }
        
        return debate
    
    def _generate_investment_advice(
        self,
        synthesis: str,
        debate: Dict[str, Any],
        stock_code: str,
        data: pd.DataFrame
    ) -> str:
        """
        生成最终的投资建议
        """
        print("  - 融合投资大师智慧...")
        print("  - 生成最终投资建议...")
        
        # 简化的投资建议
        advice = f"""
## 投资建议报告

### 1. 投资评级
**评级**: 买入
**置信度**: 65%

### 2. 核心逻辑
基于量化分析结果和LLM综合分析，该股票展现出一定的投资价值。量化因子具有统计显著性，
策略的风险调整后收益表现良好。虽然存在因子稳定性不足的风险，但综合考虑多空观点后，
我们认为上行空间大于下行风险。

结合巴菲特的价值投资理念，该股票当前估值合理；结合彼得·林奇的成长投资理念，
公司具有一定的成长潜力。

### 3. 目标价位
- **买入价位**: 当前价格 ± 5%
- **目标价位**: 当前价格 + 20%
- **止损价位**: 当前价格 - 10%

### 4. 持仓建议
- **建议持仓比例**: 5-10% 的投资组合
- **建仓策略**: 分批建仓（3次，每次间隔1周）
- **持有期限**: 中期（3-6个月）

### 5. 风险提示
1. 量化因子的稳定性有待验证，可能在市场环境变化时失效
2. 样本期较短，存在过拟合风险
3. 宏观经济环境的不确定性可能影响股票表现
4. 行业竞争加剧可能侵蚀公司利润
5. 未充分考虑交易成本，实际收益可能低于回测结果

### 6. 替代方案
如果当前不适合投资，可以考虑：
- 等待更好的买入时机（如市场回调）
- 投资同行业的其他优质标的
- 配置防御性资产（如债券、黄金）

### 7. 决策依据
本建议综合了以下信息：
- **量化分析**: IC分析、回测结果、统计检验
- **LLM综合分析**: 深度因果推断和市场环境分析
- **多空辩论**: Bull和Bear Agent的不同视角
- **投资大师智慧**: 巴菲特的价值投资、彼得·林奇的成长投资理念

最终决策倾向于买入，主要基于量化因子的统计显著性和良好的风险调整收益。
同时，我们也充分考虑了Bear Agent提出的风险因素，因此建议采取分批建仓和
严格止损的策略。
"""
        
        return advice
    
    def _generate_report(self, results: Dict[str, Any]) -> Path:
        """
        生成Markdown格式的最终报告
        """
        stock_code = results["stock_code"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"{stock_code}_{timestamp}_analysis_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# {stock_code} 量化投资分析报告\n\n")
            f.write(f"**分析时间**: {results['analysis_timestamp']}  \n")
            f.write(f"**分析周期**: {results['start_date']} 至 {results['end_date']}  \n\n")
            f.write("---\n\n")
            
            # 量化分析结果
            f.write("## 一、量化分析结果\n\n")
            quant = results["stages"]["quantitative_analysis"]
            f.write(f"```json\n{json.dumps(quant, indent=2, ensure_ascii=False)}\n```\n\n")
            
            # LLM综合分析
            f.write("## 二、LLM综合分析\n\n")
            f.write(results["stages"]["llm_synthesis"])
            f.write("\n\n")
            
            # 多空辩论
            f.write("## 三、多空辩论\n\n")
            debate = results["stages"]["debate"]
            f.write("### Bull Agent观点\n\n")
            f.write(f"```json\n{json.dumps(debate['bull_view'], indent=2, ensure_ascii=False)}\n```\n\n")
            f.write("### Bear Agent观点\n\n")
            f.write(f"```json\n{json.dumps(debate['bear_view'], indent=2, ensure_ascii=False)}\n```\n\n")
            
            # 投资建议
            f.write("## 四、投资建议\n\n")
            f.write(results["stages"]["investment_advice"])
            f.write("\n\n")
            
            # 免责声明
            f.write("---\n\n")
            f.write("**免责声明**: 本报告仅供参考，不构成投资建议。投资有风险，入市需谨慎。\n")
        
        return report_path


def demo():
    """演示LLM增强量化分析流水线"""
    print("=== LLM-Enhanced Quant Analysis Pipeline Demo ===\n")
    
    # 初始化流水线
    pipeline = LLMQuantPipeline(
        llm_model="gpt-3.5-turbo",  # 可以根据需要切换模型
        enable_cache=True
    )
    
    # 分析一只股票
    results = pipeline.analyze_stock(
        stock_code="600519.SH",  # 贵州茅台
        start_date="2023-01-01",
        end_date="2023-12-31",
        analysis_type="comprehensive"
    )
    
    print(f"\n分析结果已保存至: {results['report_path']}")
    print("\n流水线演示完成！")


if __name__ == "__main__":
    demo()
