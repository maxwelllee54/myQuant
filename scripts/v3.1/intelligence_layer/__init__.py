#!/usr/bin/env python3
"""
quant-investor V3.1 Intelligence Layer Module
动态智能框架：因子监控 + 组合优化

版本: 3.1
"""

from .factor_decay_monitor import (
    FactorDecayMonitor,
    FactorExposure,
    DecayAnalysis,
    RebalanceSignal,
)

from .portfolio_optimizer import (
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
