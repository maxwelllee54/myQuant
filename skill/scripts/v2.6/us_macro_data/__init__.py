"""
quant-investor V2.6 美股宏观数据模块

提供美股市场的宏观经济数据获取能力，包括：
- GDP、CPI、PCE、失业率、非农就业等宏观经济指标
- 联邦基金利率、国债收益率等货币政策指标
- VIX、标普500等市场指标
- 美股行情和财务数据

数据源：
- FRED (Federal Reserve Economic Data) - 宏观经济数据主数据源
- yfinance (Yahoo Finance) - 美股行情数据主数据源
- Tushare Pro - 备用数据源
- AKShare - 备用数据源
"""

from .fred_client import FREDClient
from .yfinance_client import YFinanceClient
from .finnhub_client import FinnhubClient
from .us_macro_data_manager import USMacroDataManager

__all__ = [
    'FREDClient',
    'YFinanceClient',
    'FinnhubClient',
    'USMacroDataManager',
]

__version__ = '2.6.0'
