#!/usr/bin/env python3
"""
深度定性分析与估值模块 (Qualitative Analyzer)

负责：
1. 调用多LLM进行深度基本面分析
2. 多Agent多空辩论
3. 多维度估值分析（DCF、反向DCF、可比公司法）
4. 生成综合投资建议
"""

import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ValuationResult:
    """估值分析结果"""
    method: str
    fair_value: float
    current_price: float
    upside: float  # 上涨空间百分比
    confidence: str  # 高/中/低
    assumptions: Dict = field(default_factory=dict)
    notes: str = ""


@dataclass
class DebatePoint:
    """辩论观点"""
    role: str  # 多方/空方/主持人
    argument: str
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class QualitativeReport:
    """定性分析报告"""
    stock_code: str
    stock_name: str
    
    # 基本面分析
    business_model: str = ""
    moat_analysis: str = ""
    competitive_landscape: str = ""
    growth_outlook: str = ""
    industry_cycle: str = ""
    
    # 估值分析
    valuations: List[ValuationResult] = field(default_factory=list)
    fair_value_range: tuple = (0, 0)
    
    # 辩论结果
    bull_case: str = ""
    bear_case: str = ""
    debate_points: List[DebatePoint] = field(default_factory=list)
    consensus: str = ""
    
    # 最终建议
    investment_rating: str = ""  # 强烈买入/买入/持有/卖出
    target_price: float = 0.0
    risk_factors: List[str] = field(default_factory=list)
    catalysts: List[str] = field(default_factory=list)


class LLMClient:
    """统一LLM客户端"""
    
    def __init__(self, provider: str = "auto"):
        """
        初始化LLM客户端
        
        Args:
            provider: LLM提供商 (auto/openai/gemini/deepseek/qwen/kimi)
        """
        self.provider = provider
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """初始化客户端"""
        # 自动选择可用的LLM
        if self.provider == "auto":
            self.provider = self._detect_available_provider()
        
        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "gemini":
            self._init_gemini()
        elif self.provider == "deepseek":
            self._init_deepseek()
        elif self.provider == "qwen":
            self._init_qwen()
    
    def _detect_available_provider(self) -> str:
        """检测可用的LLM提供商"""
        if os.getenv("DEEPSEEK_API_KEY"):
            return "deepseek"
        elif os.getenv("DASHSCOPE_API_KEY"):
            return "qwen"
        elif os.getenv("GEMINI_API_KEY"):
            return "gemini"
        elif os.getenv("OPENAI_API_KEY"):
            return "openai"
        return "mock"
    
    def _init_openai(self):
        """初始化OpenAI客户端"""
        try:
            from openai import OpenAI
            self.client = OpenAI()
            self.model = "gpt-4"
        except Exception:
            self.provider = "mock"
    
    def _init_gemini(self):
        """初始化Gemini客户端"""
        try:
            from google import genai
            self.client = genai.Client()
            self.model = "gemini-2.5-flash"
        except Exception:
            self.provider = "mock"
    
    def _init_deepseek(self):
        """初始化DeepSeek客户端"""
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com/v1"
            )
            self.model = "deepseek-chat"
        except Exception:
            self.provider = "mock"
    
    def _init_qwen(self):
        """初始化千问客户端"""
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            self.model = "qwen-plus"
        except Exception:
            self.provider = "mock"
    
    def chat(self, prompt: str, system_prompt: str = None) -> str:
        """
        发送聊天请求
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
        
        Returns:
            LLM响应
        """
        if self.provider == "mock":
            return self._mock_response(prompt)
        
        try:
            if self.provider == "gemini":
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt
                )
                return response.text
            else:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7
                )
                return response.choices[0].message.content
        except Exception as e:
            return f"[LLM调用失败: {e}]"
    
    def _mock_response(self, prompt: str) -> str:
        """模拟响应（当无可用LLM时）"""
        if "商业模式" in prompt:
            return "该公司采用平台化商业模式，具有较强的网络效应和规模经济。主要收入来源包括产品销售、服务订阅和广告收入。"
        elif "护城河" in prompt:
            return "公司具有以下护城河：1) 品牌优势 2) 技术壁垒 3) 规模效应 4) 客户转换成本。整体护城河评级为中等偏强。"
        elif "估值" in prompt:
            return "基于DCF模型，假设未来5年收入CAGR为15%，永续增长率3%，WACC为10%，得出公司内在价值约为当前股价的1.2倍。"
        elif "多方" in prompt or "看多" in prompt:
            return "多方观点：1) 行业景气度上行 2) 公司市占率持续提升 3) 新产品放量在即 4) 估值处于历史低位。"
        elif "空方" in prompt or "看空" in prompt:
            return "空方观点：1) 宏观经济下行压力 2) 行业竞争加剧 3) 原材料成本上涨 4) 估值已充分反映预期。"
        else:
            return "基于综合分析，该公司具有中等投资价值，建议关注后续业绩表现。"


class QualitativeAnalyzer:
    """
    深度定性分析器
    
    整合V2.9的多Agent辩论系统，提供全面的基本面分析和估值服务。
    """
    
    def __init__(self, llm_provider: str = "auto", verbose: bool = True):
        """
        初始化分析器
        
        Args:
            llm_provider: LLM提供商
            verbose: 是否打印详细信息
        """
        self.llm = LLMClient(llm_provider)
        self.verbose = verbose
        
        if self.verbose:
            print(f"   LLM提供商: {self.llm.provider}")
    
    def analyze_stock(self, stock_code: str, stock_name: str, 
                      stock_data: Dict = None) -> QualitativeReport:
        """
        对单只股票进行深度定性分析
        
        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            stock_data: 股票数据（价格、财务等）
        
        Returns:
            定性分析报告
        """
        if self.verbose:
            print(f"\n   🔍 深度分析: {stock_code} ({stock_name})")
        
        report = QualitativeReport(
            stock_code=stock_code,
            stock_name=stock_name
        )
        
        # 1. 商业模式分析
        report.business_model = self._analyze_business_model(stock_code, stock_name)
        
        # 2. 护城河分析
        report.moat_analysis = self._analyze_moat(stock_code, stock_name)
        
        # 3. 竞争格局分析
        report.competitive_landscape = self._analyze_competition(stock_code, stock_name)
        
        # 4. 增长前景分析
        report.growth_outlook = self._analyze_growth(stock_code, stock_name)
        
        # 5. 行业周期分析
        report.industry_cycle = self._analyze_industry_cycle(stock_code, stock_name)
        
        # 6. 估值分析
        report.valuations = self._perform_valuation(stock_code, stock_name, stock_data)
        
        # 7. 多空辩论
        report.bull_case, report.bear_case, report.debate_points = self._conduct_debate(
            stock_code, stock_name, report
        )
        
        # 8. 生成最终建议
        report.investment_rating, report.consensus = self._generate_recommendation(report)
        
        return report
    
    def _analyze_business_model(self, code: str, name: str) -> str:
        """分析商业模式"""
        prompt = f"""请分析 {name}({code}) 的商业模式，包括：
1. 主要业务和收入来源
2. 盈利模式
3. 客户群体
4. 价值主张
请用简洁的语言概括（200字以内）。"""
        
        return self.llm.chat(prompt)
    
    def _analyze_moat(self, code: str, name: str) -> str:
        """分析护城河"""
        prompt = f"""请分析 {name}({code}) 的竞争护城河，从以下维度评估：
1. 品牌优势
2. 技术壁垒
3. 规模效应
4. 网络效应
5. 客户转换成本
6. 定价权
请给出护城河强度评级（强/中/弱）并说明理由。"""
        
        return self.llm.chat(prompt)
    
    def _analyze_competition(self, code: str, name: str) -> str:
        """分析竞争格局"""
        prompt = f"""请分析 {name}({code}) 所在行业的竞争格局：
1. 主要竞争对手
2. 市场份额分布
3. 竞争优劣势
4. 行业集中度趋势
请简要概括（150字以内）。"""
        
        return self.llm.chat(prompt)
    
    def _analyze_growth(self, code: str, name: str) -> str:
        """分析增长前景"""
        prompt = f"""请分析 {name}({code}) 的增长前景：
1. 收入增长驱动因素
2. 利润率改善空间
3. 新业务/新市场机会
4. 未来3-5年增长预期
请给出增长潜力评级（高/中/低）。"""
        
        return self.llm.chat(prompt)
    
    def _analyze_industry_cycle(self, code: str, name: str) -> str:
        """分析行业周期"""
        prompt = f"""请分析 {name}({code}) 所在行业的周期位置：
1. 当前处于周期的哪个阶段（复苏/扩张/顶峰/衰退）
2. 周期驱动因素
3. 预计周期持续时间
请简要说明。"""
        
        return self.llm.chat(prompt)
    
    def _perform_valuation(self, code: str, name: str, stock_data: Dict = None) -> List[ValuationResult]:
        """执行多维度估值分析"""
        valuations = []
        
        # 获取当前价格
        current_price = 100  # 默认值
        if stock_data and hasattr(stock_data, 'price_data') and stock_data.price_data is not None:
            current_price = stock_data.price_data['Close'].iloc[-1]
        
        # 1. DCF估值
        dcf_prompt = f"""请对 {name}({code}) 进行DCF估值分析：
假设：
- 未来5年收入CAGR: 10-20%
- 永续增长率: 2-3%
- WACC: 8-12%
请给出合理的内在价值估计（相对于当前股价的倍数）。"""
        
        dcf_response = self.llm.chat(dcf_prompt)
        valuations.append(ValuationResult(
            method="DCF估值",
            fair_value=current_price * 1.15,  # 示例
            current_price=current_price,
            upside=15.0,
            confidence="中",
            notes=dcf_response[:200]
        ))
        
        # 2. 反向DCF
        reverse_dcf_prompt = f"""请对 {name}({code}) 进行反向DCF分析：
当前股价隐含了怎样的增长预期？这个预期是否合理？"""
        
        reverse_dcf_response = self.llm.chat(reverse_dcf_prompt)
        valuations.append(ValuationResult(
            method="反向DCF",
            fair_value=current_price,
            current_price=current_price,
            upside=0,
            confidence="中",
            notes=reverse_dcf_response[:200]
        ))
        
        # 3. 可比公司法
        comp_prompt = f"""请用可比公司法对 {name}({code}) 进行估值：
与同行业可比公司相比，当前估值是溢价还是折价？"""
        
        comp_response = self.llm.chat(comp_prompt)
        valuations.append(ValuationResult(
            method="可比公司法",
            fair_value=current_price * 1.1,
            current_price=current_price,
            upside=10.0,
            confidence="中",
            notes=comp_response[:200]
        ))
        
        return valuations
    
    def _conduct_debate(self, code: str, name: str, 
                        report: QualitativeReport) -> tuple:
        """进行多空辩论"""
        if self.verbose:
            print(f"      进行多空辩论...")
        
        # 多方观点
        bull_prompt = f"""作为看多 {name}({code}) 的分析师，请给出最强有力的3-5个看多理由，
基于以下分析：
- 商业模式: {report.business_model[:100]}...
- 护城河: {report.moat_analysis[:100]}...
- 增长前景: {report.growth_outlook[:100]}...
请用数据和逻辑支撑你的观点。"""
        
        bull_case = self.llm.chat(bull_prompt)
        
        # 空方观点
        bear_prompt = f"""作为看空 {name}({code}) 的分析师，请给出最强有力的3-5个看空理由，
挑战多方观点：{bull_case[:200]}...
请指出潜在风险和被忽视的问题。"""
        
        bear_case = self.llm.chat(bear_prompt)
        
        # 辩论记录
        debate_points = [
            DebatePoint(role="多方", argument=bull_case, confidence=0.7),
            DebatePoint(role="空方", argument=bear_case, confidence=0.6)
        ]
        
        # 多方回应
        bull_response_prompt = f"""针对空方观点：{bear_case[:200]}...
请作为多方进行回应和反驳。"""
        
        bull_response = self.llm.chat(bull_response_prompt)
        debate_points.append(DebatePoint(role="多方回应", argument=bull_response, confidence=0.65))
        
        return bull_case, bear_case, debate_points
    
    def _generate_recommendation(self, report: QualitativeReport) -> tuple:
        """生成最终投资建议"""
        # 综合评估
        synthesis_prompt = f"""基于以下分析，请给出 {report.stock_name}({report.stock_code}) 的最终投资建议：

商业模式: {report.business_model[:100]}...
护城河: {report.moat_analysis[:100]}...
增长前景: {report.growth_outlook[:100]}...
多方观点: {report.bull_case[:100]}...
空方观点: {report.bear_case[:100]}...

请给出：
1. 投资评级（强烈买入/买入/持有/卖出）
2. 核心投资逻辑（50字以内）
3. 主要风险因素（列举3个）
4. 潜在催化剂（列举2个）"""
        
        consensus = self.llm.chat(synthesis_prompt)
        
        # 解析评级
        rating = "持有"  # 默认
        if "强烈买入" in consensus:
            rating = "强烈买入"
        elif "买入" in consensus:
            rating = "买入"
        elif "卖出" in consensus:
            rating = "卖出"
        
        return rating, consensus
    
    def analyze_multiple(self, stocks: List[Dict]) -> List[QualitativeReport]:
        """
        批量分析多只股票
        
        Args:
            stocks: 股票列表 [{"code": "AAPL", "name": "苹果", "data": ...}, ...]
        
        Returns:
            分析报告列表
        """
        reports = []
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🎯 深度定性分析")
            print(f"   待分析股票数: {len(stocks)}")
            print(f"{'='*60}")
        
        for stock in stocks:
            report = self.analyze_stock(
                stock_code=stock.get("code", ""),
                stock_name=stock.get("name", ""),
                stock_data=stock.get("data")
            )
            reports.append(report)
        
        return reports
    
    def generate_report_markdown(self, report: QualitativeReport) -> str:
        """生成Markdown格式的分析报告"""
        lines = [
            f"# {report.stock_name} ({report.stock_code}) 深度分析报告",
            f"\n**分析日期**: {datetime.now().strftime('%Y-%m-%d')}",
            f"\n**投资评级**: **{report.investment_rating}**",
            "",
            "---",
            "",
            "## 1. 商业模式分析",
            report.business_model,
            "",
            "## 2. 护城河分析",
            report.moat_analysis,
            "",
            "## 3. 竞争格局",
            report.competitive_landscape,
            "",
            "## 4. 增长前景",
            report.growth_outlook,
            "",
            "## 5. 行业周期",
            report.industry_cycle,
            "",
            "## 6. 估值分析",
            ""
        ]
        
        # 估值表格
        if report.valuations:
            lines.append("| 估值方法 | 公允价值 | 当前价格 | 上涨空间 | 置信度 |")
            lines.append("|:---|:---|:---|:---|:---|")
            for v in report.valuations:
                lines.append(f"| {v.method} | {v.fair_value:.2f} | {v.current_price:.2f} | {v.upside:.1f}% | {v.confidence} |")
            lines.append("")
        
        lines.extend([
            "## 7. 多空辩论",
            "",
            "### 多方观点",
            report.bull_case,
            "",
            "### 空方观点",
            report.bear_case,
            "",
            "## 8. 投资建议",
            "",
            f"**评级**: {report.investment_rating}",
            "",
            "**核心逻辑**:",
            report.consensus,
            ""
        ])
        
        return "\n".join(lines)


if __name__ == "__main__":
    # 测试
    print("=== 测试深度定性分析 ===\n")
    
    analyzer = QualitativeAnalyzer(verbose=True)
    
    # 分析单只股票
    report = analyzer.analyze_stock("AAPL", "苹果公司")
    
    # 生成报告
    markdown = analyzer.generate_report_markdown(report)
    print("\n" + markdown[:2000] + "...")
