#!/usr/bin/env python3
"""
V3.5 深度特征合成引擎

核心模块:
- deep_feature_engine: 深度特征合成和增强版遗传规划
"""

from .deep_feature_engine import (
    DeepFeatureSynthesizer,
    EnhancedGeneticFactorGenerator,
    PrimitiveLibrary,
    Primitive,
    PrimitiveType,
    FeatureNode,
    deep_feature_synthesis
)

__version__ = "3.5.0"
__all__ = [
    'DeepFeatureSynthesizer',
    'EnhancedGeneticFactorGenerator',
    'PrimitiveLibrary',
    'Primitive',
    'PrimitiveType',
    'FeatureNode',
    'deep_feature_synthesis'
]
