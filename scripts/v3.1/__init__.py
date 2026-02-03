#!/usr/bin/env python3
"""
quant-investor V3.1 - 动态智能框架
Dynamic Intelligence Framework: 因子监控 + 组合优化

版本: 3.1
"""

from .intelligence_layer import (
    FactorDecayMonitor,
    FactorExposure,
    DecayAnalysis,
    RebalanceSignal,
    PortfolioOptimizer,
    OptimizationResult,
    PortfolioMetrics,
)

__version__ = "3.1"

__all__ = [
    "__version__",
    # 因子衰减监控
    "FactorDecayMonitor",
    "FactorExposure",
    "DecayAnalysis",
    "RebalanceSignal",
    # 组合优化
    "PortfolioOptimizer",
    "OptimizationResult",
    "PortfolioMetrics",
]
