#!/usr/bin/env python3
"""
V3.3 假设检验特征选择器
借鉴tsfresh的设计，实现基于统计假设检验的特征/因子筛选

核心功能:
1. 基于相关性的特征筛选
2. 基于假设检验的显著性筛选
3. 多重检验校正（Benjamini-Hochberg）
4. 冗余特征剔除
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import stats
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SelectionConfig:
    """特征选择配置"""
    significance_level: float = 0.05  # 显著性水平
    min_ic: float = 0.02  # 最小IC阈值
    max_correlation: float = 0.7  # 最大因子间相关性（用于去冗余）
    use_fdr_correction: bool = True  # 是否使用FDR校正
    correlation_method: str = 'spearman'  # 相关性计算方法


class HypothesisTestSelector:
    """
    基于假设检验的特征选择器
    
    借鉴tsfresh的特征选择机制，通过统计检验筛选与目标相关的特征
    """
    
    def __init__(self, config: SelectionConfig = None):
        self.config = config or SelectionConfig()
        
    def calculate_feature_relevance(
        self,
        features: pd.DataFrame,
        target: pd.Series
    ) -> pd.DataFrame:
        """
        计算每个特征与目标的相关性和显著性
        
        Args:
            features: 特征DataFrame，index为样本，columns为特征名
            target: 目标变量Series
            
        Returns:
            包含相关性和p值的DataFrame
        """
        results = []
        
        # 对齐索引
        common_idx = features.index.intersection(target.index)
        features = features.loc[common_idx]
        target = target.loc[common_idx]
        
        for col in features.columns:
            feature_vals = features[col].values
            target_vals = target.values
            
            # 移除NaN
            valid_mask = ~(np.isnan(feature_vals) | np.isnan(target_vals))
            if valid_mask.sum() < 10:
                continue
            
            f_valid = feature_vals[valid_mask]
            t_valid = target_vals[valid_mask]
            
            # 计算相关性
            if self.config.correlation_method == 'spearman':
                corr, p_value = spearmanr(f_valid, t_valid)
            else:
                corr, p_value = pearsonr(f_valid, t_valid)
            
            results.append({
                'feature': col,
                'correlation': corr,
                'abs_correlation': abs(corr),
                'p_value': p_value,
                'n_samples': valid_mask.sum()
            })
        
        return pd.DataFrame(results)
    
    def benjamini_hochberg_correction(
        self,
        p_values: pd.Series,
        alpha: float = 0.05
    ) -> pd.Series:
        """
        Benjamini-Hochberg FDR校正
        
        控制错误发现率（False Discovery Rate）
        """
        n = len(p_values)
        if n == 0:
            return pd.Series()
        
        # 排序
        sorted_idx = p_values.argsort()
        sorted_p = p_values.iloc[sorted_idx]
        
        # 计算BH阈值
        ranks = np.arange(1, n + 1)
        bh_threshold = ranks / n * alpha
        
        # 找到最大的满足条件的rank
        significant = sorted_p.values <= bh_threshold
        
        # 创建结果
        adjusted_p = pd.Series(index=p_values.index, dtype=float)
        
        # 计算调整后的p值
        adjusted_values = sorted_p.values * n / ranks
        adjusted_values = np.minimum.accumulate(adjusted_values[::-1])[::-1]
        adjusted_values = np.minimum(adjusted_values, 1.0)
        
        for i, idx in enumerate(sorted_idx):
            adjusted_p.iloc[idx] = adjusted_values[i]
        
        return adjusted_p
    
    def select_by_significance(
        self,
        relevance_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        基于显著性筛选特征
        """
        if relevance_df.empty:
            return relevance_df
        
        df = relevance_df.copy()
        
        # FDR校正
        if self.config.use_fdr_correction:
            df['adjusted_p_value'] = self.benjamini_hochberg_correction(
                df['p_value'],
                self.config.significance_level
            )
            significant = df['adjusted_p_value'] < self.config.significance_level
        else:
            significant = df['p_value'] < self.config.significance_level
        
        # IC阈值筛选
        ic_filter = df['abs_correlation'] >= self.config.min_ic
        
        # 组合筛选
        selected = df[significant & ic_filter].copy()
        
        return selected.sort_values('abs_correlation', ascending=False)
    
    def remove_redundant_features(
        self,
        features: pd.DataFrame,
        selected_features: List[str]
    ) -> List[str]:
        """
        移除冗余特征（高度相关的特征只保留一个）
        """
        if len(selected_features) <= 1:
            return selected_features
        
        # 计算特征间相关性矩阵
        feature_subset = features[selected_features]
        corr_matrix = feature_subset.corr(method=self.config.correlation_method)
        
        # 贪心选择：按重要性顺序，移除与已选特征高度相关的
        final_features = []
        
        for feature in selected_features:
            if not final_features:
                final_features.append(feature)
                continue
            
            # 检查与已选特征的相关性
            max_corr = max(abs(corr_matrix.loc[feature, f]) for f in final_features)
            
            if max_corr < self.config.max_correlation:
                final_features.append(feature)
        
        return final_features
    
    def select_features(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        return_details: bool = False
    ) -> Union[List[str], Tuple[List[str], pd.DataFrame]]:
        """
        执行完整的特征选择流程
        
        Args:
            features: 特征DataFrame
            target: 目标变量
            return_details: 是否返回详细信息
            
        Returns:
            选中的特征列表，可选返回详细统计信息
        """
        # 1. 计算相关性和显著性
        relevance_df = self.calculate_feature_relevance(features, target)
        
        if relevance_df.empty:
            if return_details:
                return [], pd.DataFrame()
            return []
        
        # 2. 基于显著性筛选
        selected_df = self.select_by_significance(relevance_df)
        
        if selected_df.empty:
            if return_details:
                return [], relevance_df
            return []
        
        # 3. 移除冗余特征
        selected_features = selected_df['feature'].tolist()
        final_features = self.remove_redundant_features(features, selected_features)
        
        if return_details:
            # 标记最终选中的特征
            relevance_df['selected'] = relevance_df['feature'].isin(final_features)
            return final_features, relevance_df
        
        return final_features


class FactorFilterPipeline:
    """
    因子筛选流水线
    
    整合多种筛选方法，提供端到端的因子筛选服务
    """
    
    def __init__(self, config: SelectionConfig = None):
        self.config = config or SelectionConfig()
        self.selector = HypothesisTestSelector(config)
        
    def filter_factors(
        self,
        factor_values: Dict[str, pd.DataFrame],
        forward_returns: pd.DataFrame,
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        筛选有效因子
        
        Args:
            factor_values: 因子值字典，key为因子名，value为因子DataFrame
            forward_returns: 前向收益DataFrame
            verbose: 是否打印详细信息
            
        Returns:
            筛选结果字典
        """
        # 将因子转换为长格式
        all_factors = []
        
        for factor_name, factor_df in factor_values.items():
            # 展平为长格式
            factor_flat = factor_df.stack()
            factor_flat.name = factor_name
            all_factors.append(factor_flat)
        
        if not all_factors:
            return {'selected_factors': [], 'details': pd.DataFrame()}
        
        # 合并所有因子
        factors_df = pd.concat(all_factors, axis=1)
        
        # 展平收益
        returns_flat = forward_returns.stack()
        returns_flat.name = 'forward_return'
        
        # 对齐索引
        common_idx = factors_df.index.intersection(returns_flat.index)
        factors_df = factors_df.loc[common_idx]
        returns_flat = returns_flat.loc[common_idx]
        
        # 执行特征选择
        selected, details = self.selector.select_features(
            factors_df, returns_flat, return_details=True
        )
        
        if verbose:
            print(f"\n=== 因子筛选结果 ===")
            print(f"输入因子数: {len(factor_values)}")
            print(f"通过显著性检验: {(details['p_value'] < self.config.significance_level).sum() if not details.empty else 0}")
            print(f"最终选中: {len(selected)}")
            
            if selected:
                print(f"\n选中的因子:")
                for f in selected:
                    row = details[details['feature'] == f].iloc[0]
                    print(f"  - {f}: IC={row['correlation']:.4f}, p={row['p_value']:.4f}")
        
        return {
            'selected_factors': selected,
            'details': details,
            'config': self.config
        }


def quick_select(
    features: pd.DataFrame,
    target: pd.Series,
    significance_level: float = 0.05,
    min_ic: float = 0.02,
    max_correlation: float = 0.7
) -> List[str]:
    """
    快速特征选择的便捷函数
    
    Args:
        features: 特征DataFrame
        target: 目标变量
        significance_level: 显著性水平
        min_ic: 最小IC阈值
        max_correlation: 最大特征间相关性
        
    Returns:
        选中的特征列表
    """
    config = SelectionConfig(
        significance_level=significance_level,
        min_ic=min_ic,
        max_correlation=max_correlation
    )
    selector = HypothesisTestSelector(config)
    return selector.select_features(features, target)


# 测试代码
if __name__ == "__main__":
    print("=== V3.3 Feature Selector Test ===\n")
    
    np.random.seed(42)
    n_samples = 1000
    
    # 生成目标变量
    target = pd.Series(np.random.randn(n_samples), name='target')
    
    # 生成特征
    features = pd.DataFrame({
        # 与目标强相关的特征
        'strong_positive': target * 0.8 + np.random.randn(n_samples) * 0.2,
        'strong_negative': -target * 0.7 + np.random.randn(n_samples) * 0.3,
        
        # 与目标弱相关的特征
        'weak_positive': target * 0.2 + np.random.randn(n_samples) * 0.8,
        
        # 与目标无关的特征
        'noise_1': np.random.randn(n_samples),
        'noise_2': np.random.randn(n_samples),
        'noise_3': np.random.randn(n_samples),
        
        # 与strong_positive高度相关的冗余特征
        'redundant': target * 0.75 + np.random.randn(n_samples) * 0.25,
    })
    
    # 执行特征选择
    config = SelectionConfig(
        significance_level=0.05,
        min_ic=0.1,
        max_correlation=0.7
    )
    selector = HypothesisTestSelector(config)
    
    selected, details = selector.select_features(features, target, return_details=True)
    
    print("特征相关性分析:")
    print(details.to_string(index=False))
    
    print(f"\n选中的特征: {selected}")
    print(f"(注: redundant与strong_positive高度相关，被去冗余)")
    
    print("\n=== Test Completed ===")
