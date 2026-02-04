#!/usr/bin/env python3
"""
V3.3 工业级因子分析系统

核心模块:
- factor_tear_sheet: Alphalens风格的Tear Sheet分析报告
- feature_selector: 假设检验特征选择器
"""

from .factor_tear_sheet import (
    FactorTearSheet,
    TearSheetConfig,
    create_tear_sheet,
    ICAnalyzer,
    QuantileAnalyzer,
    TurnoverAnalyzer,
    DecayAnalyzer
)

from .feature_selector import (
    HypothesisTestSelector,
    SelectionConfig,
    FactorFilterPipeline,
    quick_select
)

__version__ = "3.3.0"
__all__ = [
    'FactorTearSheet',
    'TearSheetConfig', 
    'create_tear_sheet',
    'ICAnalyzer',
    'QuantileAnalyzer',
    'TurnoverAnalyzer',
    'DecayAnalyzer',
    'HypothesisTestSelector',
    'SelectionConfig',
    'FactorFilterPipeline',
    'quick_select'
]
