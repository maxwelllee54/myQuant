"""
V4.1 基准对比与超额收益分析模块
==============================

以超越市场基准的长期稳定超额收益为核心目标，
对因子/策略进行全面的有效性验证。

核心组件:
- BenchmarkProvider: 市场基准数据获取
- AlphaAnalyzer: 超额收益分析
- EnhancedFactorValidator: 增强版因子验证器
"""

from .benchmark_provider import (
    BenchmarkProvider,
    get_benchmark_returns
)

from .alpha_analyzer import (
    AlphaAnalyzer,
    AlphaMetrics,
    StabilityAnalysis,
    evaluate_vs_benchmark
)

from .enhanced_factor_validator import (
    EnhancedFactorValidator,
    EnhancedFactorValidationResult
)

__all__ = [
    # 基准数据
    'BenchmarkProvider',
    'get_benchmark_returns',
    
    # 超额收益分析
    'AlphaAnalyzer',
    'AlphaMetrics',
    'StabilityAnalysis',
    'evaluate_vs_benchmark',
    
    # 因子验证
    'EnhancedFactorValidator',
    'EnhancedFactorValidationResult',
]

__version__ = "4.1"
