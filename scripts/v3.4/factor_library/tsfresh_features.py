#!/usr/bin/env python3
"""
V3.4 tsfresh时间序列特征集成
借鉴tsfresh的设计，实现100+种时间序列特征的计算

tsfresh特征类别:
- 统计特征（均值、方差、偏度、峰度等）
- 分布特征（分位数、熵等）
- 自相关特征
- 变化点特征
- 复杂度特征
- 能量特征

参考: https://tsfresh.readthedocs.io/en/latest/
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Union
from dataclasses import dataclass
from scipy import stats
from scipy.fft import fft
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TSFeatureDefinition:
    """时间序列特征定义"""
    name: str
    category: str
    description: str
    params: Dict = None


class TSFreshCalculator:
    """
    tsfresh风格的时间序列特征计算器
    
    实现100+种时间序列特征的计算
    """
    
    def __init__(self, window_sizes: List[int] = None):
        self.window_sizes = window_sizes or [5, 10, 20, 60]
        self.feature_definitions = self._build_feature_definitions()
    
    def _build_feature_definitions(self) -> List[TSFeatureDefinition]:
        """构建特征定义"""
        definitions = []
        
        # ========== 统计特征 ==========
        for w in self.window_sizes:
            definitions.extend([
                TSFeatureDefinition(f'mean_{w}', 'statistics', f'{w}日均值'),
                TSFeatureDefinition(f'std_{w}', 'statistics', f'{w}日标准差'),
                TSFeatureDefinition(f'var_{w}', 'statistics', f'{w}日方差'),
                TSFeatureDefinition(f'median_{w}', 'statistics', f'{w}日中位数'),
                TSFeatureDefinition(f'min_{w}', 'statistics', f'{w}日最小值'),
                TSFeatureDefinition(f'max_{w}', 'statistics', f'{w}日最大值'),
                TSFeatureDefinition(f'range_{w}', 'statistics', f'{w}日极差'),
                TSFeatureDefinition(f'skewness_{w}', 'statistics', f'{w}日偏度'),
                TSFeatureDefinition(f'kurtosis_{w}', 'statistics', f'{w}日峰度'),
                TSFeatureDefinition(f'sum_{w}', 'statistics', f'{w}日累计值'),
                TSFeatureDefinition(f'abs_sum_{w}', 'statistics', f'{w}日绝对值累计'),
            ])
        
        # ========== 分位数特征 ==========
        for w in self.window_sizes:
            for q in [0.1, 0.25, 0.75, 0.9]:
                definitions.append(
                    TSFeatureDefinition(f'quantile_{w}_{int(q*100)}', 'quantile', f'{w}日{int(q*100)}%分位数')
                )
        
        # ========== 变化特征 ==========
        for w in self.window_sizes:
            definitions.extend([
                TSFeatureDefinition(f'abs_change_{w}', 'change', f'{w}日绝对变化均值'),
                TSFeatureDefinition(f'mean_change_{w}', 'change', f'{w}日变化均值'),
                TSFeatureDefinition(f'mean_second_derivative_{w}', 'change', f'{w}日二阶导数均值'),
                TSFeatureDefinition(f'count_above_mean_{w}', 'change', f'{w}日高于均值次数'),
                TSFeatureDefinition(f'count_below_mean_{w}', 'change', f'{w}日低于均值次数'),
                TSFeatureDefinition(f'first_location_of_max_{w}', 'change', f'{w}日最大值首次位置'),
                TSFeatureDefinition(f'last_location_of_max_{w}', 'change', f'{w}日最大值末次位置'),
                TSFeatureDefinition(f'first_location_of_min_{w}', 'change', f'{w}日最小值首次位置'),
                TSFeatureDefinition(f'last_location_of_min_{w}', 'change', f'{w}日最小值末次位置'),
            ])
        
        # ========== 自相关特征 ==========
        for w in self.window_sizes:
            for lag in [1, 2, 3, 5]:
                if lag < w:
                    definitions.append(
                        TSFeatureDefinition(f'autocorr_{w}_lag{lag}', 'autocorrelation', f'{w}日滞后{lag}自相关')
                    )
        
        # ========== 复杂度特征 ==========
        for w in self.window_sizes:
            definitions.extend([
                TSFeatureDefinition(f'cid_ce_{w}', 'complexity', f'{w}日复杂度估计'),
                TSFeatureDefinition(f'binned_entropy_{w}', 'complexity', f'{w}日分箱熵'),
                TSFeatureDefinition(f'approximate_entropy_{w}', 'complexity', f'{w}日近似熵'),
            ])
        
        # ========== 趋势特征 ==========
        for w in self.window_sizes:
            definitions.extend([
                TSFeatureDefinition(f'linear_trend_slope_{w}', 'trend', f'{w}日线性趋势斜率'),
                TSFeatureDefinition(f'linear_trend_intercept_{w}', 'trend', f'{w}日线性趋势截距'),
                TSFeatureDefinition(f'linear_trend_rvalue_{w}', 'trend', f'{w}日线性趋势R值'),
            ])
        
        # ========== 能量特征 ==========
        for w in self.window_sizes:
            definitions.extend([
                TSFeatureDefinition(f'abs_energy_{w}', 'energy', f'{w}日绝对能量'),
                TSFeatureDefinition(f'mean_abs_change_{w}', 'energy', f'{w}日平均绝对变化'),
                TSFeatureDefinition(f'root_mean_square_{w}', 'energy', f'{w}日均方根'),
            ])
        
        # ========== 峰值特征 ==========
        for w in self.window_sizes:
            definitions.extend([
                TSFeatureDefinition(f'number_peaks_{w}', 'peaks', f'{w}日峰值数量'),
                TSFeatureDefinition(f'number_crossing_mean_{w}', 'peaks', f'{w}日穿越均值次数'),
            ])
        
        return definitions
    
    def get_feature_list(self, category: str = None) -> List[str]:
        """获取特征列表"""
        if category:
            return [f.name for f in self.feature_definitions if f.category == category]
        return [f.name for f in self.feature_definitions]
    
    def get_categories(self) -> List[str]:
        """获取所有特征类别"""
        return list(set(f.category for f in self.feature_definitions))
    
    def calculate_feature(
        self,
        series: pd.Series,
        feature_name: str
    ) -> pd.Series:
        """计算单个特征"""
        # 解析特征名
        parts = feature_name.split('_')
        
        # 提取窗口大小
        window = None
        for p in parts:
            if p.isdigit():
                window = int(p)
                break
        
        if window is None:
            raise ValueError(f"Cannot parse window size from feature name: {feature_name}")
        
        # 根据特征类型计算
        if feature_name.startswith('mean_'):
            return series.rolling(window, min_periods=1).mean()
        elif feature_name.startswith('std_'):
            return series.rolling(window, min_periods=1).std()
        elif feature_name.startswith('var_'):
            return series.rolling(window, min_periods=1).var()
        elif feature_name.startswith('median_'):
            return series.rolling(window, min_periods=1).median()
        elif feature_name.startswith('min_'):
            return series.rolling(window, min_periods=1).min()
        elif feature_name.startswith('max_'):
            return series.rolling(window, min_periods=1).max()
        elif feature_name.startswith('range_'):
            return series.rolling(window, min_periods=1).max() - series.rolling(window, min_periods=1).min()
        elif feature_name.startswith('skewness_'):
            return series.rolling(window, min_periods=1).skew()
        elif feature_name.startswith('kurtosis_'):
            return series.rolling(window, min_periods=1).kurt()
        elif feature_name.startswith('sum_'):
            return series.rolling(window, min_periods=1).sum()
        elif feature_name.startswith('abs_sum_'):
            return series.abs().rolling(window, min_periods=1).sum()
        elif feature_name.startswith('quantile_'):
            q = int(parts[-1]) / 100
            return series.rolling(window, min_periods=1).quantile(q)
        elif feature_name.startswith('abs_change_'):
            return series.diff().abs().rolling(window, min_periods=1).mean()
        elif feature_name.startswith('mean_change_'):
            return series.diff().rolling(window, min_periods=1).mean()
        elif feature_name.startswith('mean_second_derivative_'):
            return series.diff().diff().rolling(window, min_periods=1).mean()
        elif feature_name.startswith('count_above_mean_'):
            rolling_mean = series.rolling(window, min_periods=1).mean()
            return (series > rolling_mean).rolling(window, min_periods=1).sum()
        elif feature_name.startswith('count_below_mean_'):
            rolling_mean = series.rolling(window, min_periods=1).mean()
            return (series < rolling_mean).rolling(window, min_periods=1).sum()
        elif feature_name.startswith('autocorr_'):
            lag = int(parts[-1].replace('lag', ''))
            return series.rolling(window, min_periods=1).apply(
                lambda x: pd.Series(x).autocorr(lag) if len(x) > lag else np.nan, raw=False
            )
        elif feature_name.startswith('cid_ce_'):
            return series.rolling(window, min_periods=1).apply(self._cid_ce, raw=True)
        elif feature_name.startswith('binned_entropy_'):
            return series.rolling(window, min_periods=1).apply(self._binned_entropy, raw=True)
        elif feature_name.startswith('linear_trend_slope_'):
            return series.rolling(window, min_periods=1).apply(self._linear_trend_slope, raw=True)
        elif feature_name.startswith('linear_trend_intercept_'):
            return series.rolling(window, min_periods=1).apply(self._linear_trend_intercept, raw=True)
        elif feature_name.startswith('linear_trend_rvalue_'):
            return series.rolling(window, min_periods=1).apply(self._linear_trend_rvalue, raw=True)
        elif feature_name.startswith('abs_energy_'):
            return (series ** 2).rolling(window, min_periods=1).sum()
        elif feature_name.startswith('mean_abs_change_'):
            return series.diff().abs().rolling(window, min_periods=1).mean()
        elif feature_name.startswith('root_mean_square_'):
            return np.sqrt((series ** 2).rolling(window, min_periods=1).mean())
        elif feature_name.startswith('number_peaks_'):
            return series.rolling(window, min_periods=1).apply(self._count_peaks, raw=True)
        elif feature_name.startswith('number_crossing_mean_'):
            return series.rolling(window, min_periods=1).apply(self._count_crossing_mean, raw=True)
        elif feature_name.startswith('first_location_of_max_'):
            return series.rolling(window, min_periods=1).apply(self._first_location_of_max, raw=True)
        elif feature_name.startswith('last_location_of_max_'):
            return series.rolling(window, min_periods=1).apply(self._last_location_of_max, raw=True)
        elif feature_name.startswith('first_location_of_min_'):
            return series.rolling(window, min_periods=1).apply(self._first_location_of_min, raw=True)
        elif feature_name.startswith('last_location_of_min_'):
            return series.rolling(window, min_periods=1).apply(self._last_location_of_min, raw=True)
        else:
            raise ValueError(f"Unknown feature: {feature_name}")
    
    # ========== 辅助计算函数 ==========
    
    @staticmethod
    def _cid_ce(x):
        """复杂度估计"""
        if len(x) < 2:
            return 0
        return np.sqrt(np.sum(np.diff(x) ** 2))
    
    @staticmethod
    def _binned_entropy(x, bins=10):
        """分箱熵"""
        if len(x) < 2:
            return 0
        hist, _ = np.histogram(x, bins=bins)
        probs = hist / len(x)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))
    
    @staticmethod
    def _linear_trend_slope(x):
        """线性趋势斜率"""
        if len(x) < 2:
            return 0
        t = np.arange(len(x))
        slope, _, _, _, _ = stats.linregress(t, x)
        return slope
    
    @staticmethod
    def _linear_trend_intercept(x):
        """线性趋势截距"""
        if len(x) < 2:
            return x[0] if len(x) > 0 else 0
        t = np.arange(len(x))
        _, intercept, _, _, _ = stats.linregress(t, x)
        return intercept
    
    @staticmethod
    def _linear_trend_rvalue(x):
        """线性趋势R值"""
        if len(x) < 2:
            return 0
        t = np.arange(len(x))
        _, _, rvalue, _, _ = stats.linregress(t, x)
        return rvalue
    
    @staticmethod
    def _count_peaks(x, support=1):
        """计算峰值数量"""
        if len(x) < 3:
            return 0
        peaks = 0
        for i in range(support, len(x) - support):
            if all(x[i] > x[i-j] for j in range(1, support+1)) and \
               all(x[i] > x[i+j] for j in range(1, support+1)):
                peaks += 1
        return peaks
    
    @staticmethod
    def _count_crossing_mean(x):
        """计算穿越均值次数"""
        if len(x) < 2:
            return 0
        mean_val = np.mean(x)
        centered = x - mean_val
        return np.sum(np.diff(np.sign(centered)) != 0)
    
    @staticmethod
    def _first_location_of_max(x):
        """最大值首次出现位置（归一化）"""
        if len(x) == 0:
            return 0
        return np.argmax(x) / len(x)
    
    @staticmethod
    def _last_location_of_max(x):
        """最大值末次出现位置（归一化）"""
        if len(x) == 0:
            return 0
        return (len(x) - 1 - np.argmax(x[::-1])) / len(x)
    
    @staticmethod
    def _first_location_of_min(x):
        """最小值首次出现位置（归一化）"""
        if len(x) == 0:
            return 0
        return np.argmin(x) / len(x)
    
    @staticmethod
    def _last_location_of_min(x):
        """最小值末次出现位置（归一化）"""
        if len(x) == 0:
            return 0
        return (len(x) - 1 - np.argmin(x[::-1])) / len(x)
    
    def calculate_all_features(
        self,
        series: pd.Series,
        categories: List[str] = None
    ) -> pd.DataFrame:
        """
        计算所有特征
        
        Args:
            series: 输入时间序列
            categories: 要计算的特征类别
            
        Returns:
            特征DataFrame
        """
        results = {}
        
        for feature_def in self.feature_definitions:
            if categories and feature_def.category not in categories:
                continue
            
            try:
                results[feature_def.name] = self.calculate_feature(series, feature_def.name)
            except Exception as e:
                pass  # 静默跳过计算失败的特征
        
        return pd.DataFrame(results)
    
    def calculate_for_dataframe(
        self,
        df: pd.DataFrame,
        categories: List[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        为DataFrame的每一列计算所有特征
        
        Args:
            df: 输入DataFrame，每列是一个时间序列
            categories: 要计算的特征类别
            
        Returns:
            特征字典，key为特征名，value为DataFrame
        """
        all_features = {}
        
        for feature_def in self.feature_definitions:
            if categories and feature_def.category not in categories:
                continue
            
            feature_df = pd.DataFrame(index=df.index, columns=df.columns)
            
            for col in df.columns:
                try:
                    feature_df[col] = self.calculate_feature(df[col], feature_def.name)
                except:
                    pass
            
            all_features[feature_def.name] = feature_df
        
        return all_features


def calculate_tsfresh_features(
    series: pd.Series,
    window_sizes: List[int] = None,
    categories: List[str] = None
) -> pd.DataFrame:
    """
    计算tsfresh特征的便捷函数
    
    Args:
        series: 输入时间序列
        window_sizes: 窗口大小列表
        categories: 要计算的特征类别
        
    Returns:
        特征DataFrame
    """
    calculator = TSFreshCalculator(window_sizes)
    return calculator.calculate_all_features(series, categories)


# 测试代码
if __name__ == "__main__":
    print("=== V3.4 tsfresh Features Test ===\n")
    
    np.random.seed(42)
    n_days = 100
    
    # 生成测试数据
    dates = pd.date_range('2024-01-01', periods=n_days, freq='B')
    series = pd.Series(np.random.randn(n_days).cumsum() + 100, index=dates, name='price')
    
    # 初始化计算器
    calculator = TSFreshCalculator(window_sizes=[5, 10, 20])
    
    print(f"总特征数: {len(calculator.feature_definitions)}")
    print(f"特征类别: {calculator.get_categories()}")
    
    # 计算统计特征
    print("\n计算统计特征...")
    features = calculator.calculate_all_features(series, categories=['statistics'])
    print(f"计算了 {len(features.columns)} 个统计特征")
    print("\n特征示例:")
    print(features.tail())
    
    print("\n=== Test Completed ===")
