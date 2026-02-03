#!/usr/bin/env python3
"""
quant-investor V3.0 Data Layer Module
全景数据层：期货期权 + 行业深度数据

作者: Manus AI
版本: 3.0
"""

from .futures_options_manager import (
    FuturesOptionsDataManager,
    OptionData,
    FutureData,
)

from .industry_data_manager import (
    IndustryDataManager,
    IndustryInfo,
    IndustryIndex,
)

__version__ = "3.0"

__all__ = [
    "__version__",
    # 期货期权
    "FuturesOptionsDataManager",
    "OptionData",
    "FutureData",
    # 行业数据
    "IndustryDataManager",
    "IndustryInfo",
    "IndustryIndex",
]
