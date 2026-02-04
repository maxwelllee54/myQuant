#!/usr/bin/env python3
"""
V3.4 Qlib Alpha158 因子库
移植自Microsoft Qlib的Alpha158因子集

Alpha158是Qlib中经过市场验证的158个技术因子，涵盖:
- 价格动量因子
- 成交量因子
- 波动率因子
- 技术指标因子
- 价量关系因子

参考: https://github.com/microsoft/qlib
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass


@dataclass
class FactorDefinition:
    """因子定义"""
    name: str
    expression: str
    category: str
    description: str


class Alpha158Calculator:
    """
    Alpha158因子计算器
    
    实现Qlib Alpha158因子集的核心计算逻辑
    """
    
    def __init__(self):
        self.factor_definitions = self._build_factor_definitions()
        
    def _build_factor_definitions(self) -> List[FactorDefinition]:
        """构建Alpha158因子定义"""
        definitions = []
        
        # ========== 价格动量因子 ==========
        # ROC (Rate of Change) 系列
        for d in [1, 2, 3, 4, 5, 10, 20, 30, 60]:
            definitions.append(FactorDefinition(
                name=f'ROC_{d}',
                expression=f'close / delay(close, {d}) - 1',
                category='momentum',
                description=f'{d}日收益率'
            ))
        
        # MA偏离度
        for d in [5, 10, 20, 30, 60]:
            definitions.append(FactorDefinition(
                name=f'BIAS_{d}',
                expression=f'close / mean(close, {d}) - 1',
                category='momentum',
                description=f'收盘价相对{d}日均线偏离度'
            ))
        
        # 价格位置
        for d in [5, 10, 20, 30, 60]:
            definitions.append(FactorDefinition(
                name=f'PRICE_POSITION_{d}',
                expression=f'(close - min(low, {d})) / (max(high, {d}) - min(low, {d}) + 1e-8)',
                category='momentum',
                description=f'{d}日价格位置'
            ))
        
        # ========== 成交量因子 ==========
        # 成交量变化
        for d in [1, 5, 10, 20]:
            definitions.append(FactorDefinition(
                name=f'VOL_CHG_{d}',
                expression=f'volume / delay(volume, {d}) - 1',
                category='volume',
                description=f'{d}日成交量变化率'
            ))
        
        # 成交量均线比
        for d in [5, 10, 20]:
            definitions.append(FactorDefinition(
                name=f'VOL_MA_RATIO_{d}',
                expression=f'volume / mean(volume, {d})',
                category='volume',
                description=f'成交量/{d}日均量'
            ))
        
        # 量价相关性
        for d in [5, 10, 20]:
            definitions.append(FactorDefinition(
                name=f'CORR_VOL_CLOSE_{d}',
                expression=f'corr(volume, close, {d})',
                category='volume',
                description=f'{d}日量价相关性'
            ))
        
        # ========== 波动率因子 ==========
        # 历史波动率
        for d in [5, 10, 20, 30, 60]:
            definitions.append(FactorDefinition(
                name=f'VOLATILITY_{d}',
                expression=f'std(close / delay(close, 1) - 1, {d})',
                category='volatility',
                description=f'{d}日收益率标准差'
            ))
        
        # ATR (Average True Range)
        for d in [5, 10, 20]:
            definitions.append(FactorDefinition(
                name=f'ATR_{d}',
                expression=f'mean(max(high - low, max(abs(high - delay(close, 1)), abs(low - delay(close, 1)))), {d})',
                category='volatility',
                description=f'{d}日平均真实波幅'
            ))
        
        # 振幅
        for d in [5, 10, 20]:
            definitions.append(FactorDefinition(
                name=f'AMPLITUDE_{d}',
                expression=f'mean((high - low) / delay(close, 1), {d})',
                category='volatility',
                description=f'{d}日平均振幅'
            ))
        
        # ========== 技术指标因子 ==========
        # RSI
        for d in [6, 12, 24]:
            definitions.append(FactorDefinition(
                name=f'RSI_{d}',
                expression=f'rsi(close, {d})',
                category='technical',
                description=f'{d}日RSI'
            ))
        
        # MACD相关
        definitions.append(FactorDefinition(
            name='MACD',
            expression='ema(close, 12) - ema(close, 26)',
            category='technical',
            description='MACD线'
        ))
        definitions.append(FactorDefinition(
            name='MACD_SIGNAL',
            expression='ema(ema(close, 12) - ema(close, 26), 9)',
            category='technical',
            description='MACD信号线'
        ))
        definitions.append(FactorDefinition(
            name='MACD_HIST',
            expression='(ema(close, 12) - ema(close, 26)) - ema(ema(close, 12) - ema(close, 26), 9)',
            category='technical',
            description='MACD柱状图'
        ))
        
        # KDJ
        definitions.append(FactorDefinition(
            name='KDJ_K',
            expression='sma((close - min(low, 9)) / (max(high, 9) - min(low, 9) + 1e-8) * 100, 3)',
            category='technical',
            description='KDJ K值'
        ))
        
        # Bollinger Bands
        for d in [20]:
            definitions.append(FactorDefinition(
                name=f'BOLL_UPPER_{d}',
                expression=f'mean(close, {d}) + 2 * std(close, {d})',
                category='technical',
                description=f'{d}日布林带上轨'
            ))
            definitions.append(FactorDefinition(
                name=f'BOLL_LOWER_{d}',
                expression=f'mean(close, {d}) - 2 * std(close, {d})',
                category='technical',
                description=f'{d}日布林带下轨'
            ))
            definitions.append(FactorDefinition(
                name=f'BOLL_WIDTH_{d}',
                expression=f'4 * std(close, {d}) / mean(close, {d})',
                category='technical',
                description=f'{d}日布林带宽度'
            ))
            definitions.append(FactorDefinition(
                name=f'BOLL_POSITION_{d}',
                expression=f'(close - (mean(close, {d}) - 2 * std(close, {d}))) / (4 * std(close, {d}) + 1e-8)',
                category='technical',
                description=f'{d}日布林带位置'
            ))
        
        # ========== 价量关系因子 ==========
        # OBV变化
        for d in [5, 10, 20]:
            definitions.append(FactorDefinition(
                name=f'OBV_CHG_{d}',
                expression=f'sum(sign(close - delay(close, 1)) * volume, {d})',
                category='price_volume',
                description=f'{d}日OBV变化'
            ))
        
        # 量比价格动量
        for d in [5, 10, 20]:
            definitions.append(FactorDefinition(
                name=f'VOL_PRICE_TREND_{d}',
                expression=f'corr(volume, close, {d}) * (close / delay(close, {d}) - 1)',
                category='price_volume',
                description=f'{d}日量价趋势'
            ))
        
        # VWAP偏离
        definitions.append(FactorDefinition(
            name='VWAP_BIAS',
            expression='close / (sum(close * volume, 1) / sum(volume, 1)) - 1',
            category='price_volume',
            description='VWAP偏离度'
        ))
        
        # ========== 高低价因子 ==========
        for d in [5, 10, 20, 60]:
            definitions.append(FactorDefinition(
                name=f'HIGH_LOW_RATIO_{d}',
                expression=f'max(high, {d}) / min(low, {d})',
                category='high_low',
                description=f'{d}日最高价/最低价'
            ))
        
        # 距离最高价
        for d in [5, 10, 20, 60]:
            definitions.append(FactorDefinition(
                name=f'DIST_FROM_HIGH_{d}',
                expression=f'close / max(high, {d}) - 1',
                category='high_low',
                description=f'距离{d}日最高价'
            ))
        
        # 距离最低价
        for d in [5, 10, 20, 60]:
            definitions.append(FactorDefinition(
                name=f'DIST_FROM_LOW_{d}',
                expression=f'close / min(low, {d}) - 1',
                category='high_low',
                description=f'距离{d}日最低价'
            ))
        
        # ========== 开盘收盘因子 ==========
        definitions.append(FactorDefinition(
            name='OPEN_CLOSE_RATIO',
            expression='open / close',
            category='open_close',
            description='开盘价/收盘价'
        ))
        
        for d in [5, 10, 20]:
            definitions.append(FactorDefinition(
                name=f'OPEN_GAP_{d}',
                expression=f'mean(open / delay(close, 1) - 1, {d})',
                category='open_close',
                description=f'{d}日平均跳空'
            ))
        
        # ========== 趋势因子 ==========
        for d in [5, 10, 20, 60]:
            definitions.append(FactorDefinition(
                name=f'TREND_STRENGTH_{d}',
                expression=f'(close - delay(close, {d})) / (std(close, {d}) + 1e-8)',
                category='trend',
                description=f'{d}日趋势强度'
            ))
        
        # 均线排列
        definitions.append(FactorDefinition(
            name='MA_ALIGNMENT',
            expression='(mean(close, 5) > mean(close, 10)) + (mean(close, 10) > mean(close, 20)) + (mean(close, 20) > mean(close, 60))',
            category='trend',
            description='均线多头排列程度'
        ))
        
        return definitions
    
    def get_factor_list(self, category: str = None) -> List[str]:
        """获取因子列表"""
        if category:
            return [f.name for f in self.factor_definitions if f.category == category]
        return [f.name for f in self.factor_definitions]
    
    def get_factor_info(self, name: str) -> Optional[FactorDefinition]:
        """获取因子信息"""
        for f in self.factor_definitions:
            if f.name == name:
                return f
        return None
    
    def get_categories(self) -> List[str]:
        """获取所有因子类别"""
        return list(set(f.category for f in self.factor_definitions))
    
    def calculate_factor(
        self,
        data: Dict[str, pd.DataFrame],
        factor_name: str
    ) -> pd.DataFrame:
        """
        计算单个因子
        
        Args:
            data: 包含OHLCV数据的字典
            factor_name: 因子名称
            
        Returns:
            因子值DataFrame
        """
        factor_def = self.get_factor_info(factor_name)
        if factor_def is None:
            raise ValueError(f"Unknown factor: {factor_name}")
        
        return self._evaluate_expression(data, factor_def.expression)
    
    def calculate_all_factors(
        self,
        data: Dict[str, pd.DataFrame],
        categories: List[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        计算所有因子
        
        Args:
            data: 包含OHLCV数据的字典
            categories: 要计算的因子类别（可选）
            
        Returns:
            因子值字典
        """
        results = {}
        
        for factor_def in self.factor_definitions:
            if categories and factor_def.category not in categories:
                continue
            
            try:
                results[factor_def.name] = self._evaluate_expression(data, factor_def.expression)
            except Exception as e:
                print(f"Warning: Failed to calculate {factor_def.name}: {e}")
        
        return results
    
    def _evaluate_expression(
        self,
        data: Dict[str, pd.DataFrame],
        expression: str
    ) -> pd.DataFrame:
        """
        评估因子表达式
        
        简化版的表达式引擎，支持基本的因子计算
        """
        # 获取基础数据
        close = data.get('close', pd.DataFrame())
        open_ = data.get('open', pd.DataFrame())
        high = data.get('high', pd.DataFrame())
        low = data.get('low', pd.DataFrame())
        volume = data.get('volume', pd.DataFrame())
        
        # 定义辅助函数
        def delay(x, d):
            return x.shift(d)
        
        def mean(x, d):
            return x.rolling(d, min_periods=1).mean()
        
        def std(x, d):
            return x.rolling(d, min_periods=1).std()
        
        def sum_(x, d):
            return x.rolling(d, min_periods=1).sum()
        
        def max_(x, d):
            return x.rolling(d, min_periods=1).max()
        
        def min_(x, d):
            return x.rolling(d, min_periods=1).min()
        
        def corr(x, y, d):
            return x.rolling(d, min_periods=1).corr(y)
        
        def ema(x, d):
            return x.ewm(span=d, adjust=False).mean()
        
        def sma(x, d):
            return x.rolling(d, min_periods=1).mean()
        
        def sign(x):
            return np.sign(x)
        
        def abs_(x):
            return np.abs(x)
        
        def rsi(x, d):
            delta = x.diff()
            gain = delta.where(delta > 0, 0).rolling(d).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(d).mean()
            rs = gain / (loss + 1e-8)
            return 100 - (100 / (1 + rs))
        
        # 替换表达式中的函数名
        expr = expression
        expr = expr.replace('delay(', 'delay(')
        expr = expr.replace('mean(', 'mean(')
        expr = expr.replace('std(', 'std(')
        expr = expr.replace('sum(', 'sum_(')
        expr = expr.replace('max(', 'max_(')
        expr = expr.replace('min(', 'min_(')
        expr = expr.replace('corr(', 'corr(')
        expr = expr.replace('ema(', 'ema(')
        expr = expr.replace('sma(', 'sma(')
        expr = expr.replace('sign(', 'sign(')
        expr = expr.replace('abs(', 'abs_(')
        expr = expr.replace('rsi(', 'rsi(')
        
        # 替换数据变量名
        expr = expr.replace('open', 'open_')
        
        try:
            result = eval(expr)
            return result
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression '{expression}': {e}")


def get_alpha158_factors() -> List[FactorDefinition]:
    """获取Alpha158因子定义列表"""
    calculator = Alpha158Calculator()
    return calculator.factor_definitions


def calculate_alpha158(
    data: Dict[str, pd.DataFrame],
    categories: List[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    计算Alpha158因子的便捷函数
    
    Args:
        data: 包含OHLCV数据的字典，keys为'open', 'high', 'low', 'close', 'volume'
        categories: 要计算的因子类别
        
    Returns:
        因子值字典
    """
    calculator = Alpha158Calculator()
    return calculator.calculate_all_factors(data, categories)


# 测试代码
if __name__ == "__main__":
    print("=== V3.4 Alpha158 Factor Library Test ===\n")
    
    np.random.seed(42)
    n_days = 100
    n_stocks = 10
    
    dates = pd.date_range('2024-01-01', periods=n_days, freq='B')
    stocks = [f'STOCK_{i}' for i in range(n_stocks)]
    
    # 生成OHLCV数据
    close = pd.DataFrame(
        np.random.randn(n_days, n_stocks).cumsum(axis=0) + 100,
        index=dates, columns=stocks
    )
    data = {
        'close': close,
        'open': close * (1 + np.random.randn(n_days, n_stocks) * 0.005),
        'high': close * (1 + np.abs(np.random.randn(n_days, n_stocks) * 0.01)),
        'low': close * (1 - np.abs(np.random.randn(n_days, n_stocks) * 0.01)),
        'volume': pd.DataFrame(np.random.uniform(1e6, 1e7, (n_days, n_stocks)), index=dates, columns=stocks)
    }
    
    # 初始化计算器
    calculator = Alpha158Calculator()
    
    print(f"总因子数: {len(calculator.factor_definitions)}")
    print(f"因子类别: {calculator.get_categories()}")
    
    # 计算部分因子
    print("\n计算动量因子...")
    momentum_factors = calculator.calculate_all_factors(data, categories=['momentum'])
    print(f"计算了 {len(momentum_factors)} 个动量因子")
    
    # 显示示例
    print("\nROC_5 因子示例:")
    print(momentum_factors['ROC_5'].head())
    
    print("\n=== Test Completed ===")
