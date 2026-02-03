#!/usr/bin/env python3
"""
quant-investor V3.0 - 全景数据层
Panoramic Data Layer: 期货期权 + 行业深度数据

版本: 3.0
"""

from .data_layer import (
    FuturesOptionsDataManager,
    OptionData,
    FutureData,
    IndustryDataManager,
    IndustryInfo,
    IndustryIndex,
)

__version__ = "3.0"

__all__ = [
    "__version__",
    "FuturesOptionsDataManager",
    "OptionData",
    "FutureData",
    "IndustryDataManager",
    "IndustryInfo",
    "IndustryIndex",
]
