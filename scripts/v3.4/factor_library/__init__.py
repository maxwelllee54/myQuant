#!/usr/bin/env python3
"""
V3.4 海纳百川因子库

核心模块:
- alpha158: Qlib Alpha158因子库
- tsfresh_features: tsfresh时间序列特征
"""

from .alpha158 import (
    Alpha158Calculator,
    FactorDefinition,
    get_alpha158_factors,
    calculate_alpha158
)

from .tsfresh_features import (
    TSFreshCalculator,
    TSFeatureDefinition,
    calculate_tsfresh_features
)

__version__ = "3.4.0"
__all__ = [
    'Alpha158Calculator',
    'FactorDefinition',
    'get_alpha158_factors',
    'calculate_alpha158',
    'TSFreshCalculator',
    'TSFeatureDefinition',
    'calculate_tsfresh_features'
]
