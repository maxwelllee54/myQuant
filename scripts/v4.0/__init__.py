"""
Quant-Investor V4.0 - 统一主流水线

整合V2.3-V3.6所有能力，提供标准化的端到端投资分析流程。

主要模块：
- master_pipeline: 统一主流水线入口
- data_provider: 数据获取模块
- quant_selector: 因子挖掘与选股模块
- qualitative_analyzer: 定性分析与估值模块
- risk_assessor: 风险评估与控制模块

使用示例：
    from v4_0 import run_analysis
    
    # 分析美股市场
    result = run_analysis(market="US")
    
    # 分析A股市场，带持仓
    result = run_analysis(
        market="CN",
        holdings=[
            {"code": "600519", "name": "贵州茅台", "weight": 0.3}
        ]
    )
    
    # 打印报告
    print(result.full_report)
"""

from .master_pipeline import (
    MasterPipeline,
    run_analysis,
    AnalysisResult,
    InvestmentRecommendation
)

from .data_provider import (
    DataProvider,
    MarketConfig
)

from .quant_selector import (
    QuantSelector
)

from .qualitative_analyzer import (
    QualitativeAnalyzer,
    QualitativeReport,
    ValuationResult
)

from .risk_assessor import (
    RiskAssessor,
    PortfolioRiskReport,
    RiskMetrics,
    RiskAlert
)

__all__ = [
    # 主流水线
    'MasterPipeline',
    'run_analysis',
    'AnalysisResult',
    'InvestmentRecommendation',
    
    # 数据模块
    'DataProvider',
    'MarketConfig',
    
    # 选股模块
    'QuantSelector',
    
    # 定性分析模块
    'QualitativeAnalyzer',
    'QualitativeReport',
    'ValuationResult',
    
    # 风险模块
    'RiskAssessor',
    'PortfolioRiskReport',
    'RiskMetrics',
    'RiskAlert'
]

__version__ = "4.0.0"
