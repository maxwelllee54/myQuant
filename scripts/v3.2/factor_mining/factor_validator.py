#!/usr/bin/env python3
"""
因子有效性验证系统 (Factor Validator)
全面评估因子的预测能力和稳定性
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


@dataclass
class FactorMetrics:
    """因子评估指标"""
    # 基础IC指标
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ir: float = 0.0
    ic_positive_rate: float = 0.0
    
    # Rank IC指标
    rank_ic_mean: float = 0.0
    rank_ic_std: float = 0.0
    rank_ir: float = 0.0
    
    # 统计检验
    ic_t_stat: float = 0.0
    ic_p_value: float = 1.0
    
    # 分组收益
    group_returns: Dict[str, float] = field(default_factory=dict)
    long_short_return: float = 0.0
    long_short_sharpe: float = 0.0
    
    # 换手率
    turnover: float = 0.0
    
    # 衰减分析
    decay_half_life: float = 0.0
    
    # 综合评分
    overall_score: float = 0.0
    
    # 有效性判断
    is_valid: bool = False
    validity_reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'ic_mean': self.ic_mean,
            'ic_std': self.ic_std,
            'ir': self.ir,
            'ic_positive_rate': self.ic_positive_rate,
            'rank_ic_mean': self.rank_ic_mean,
            'rank_ic_std': self.rank_ic_std,
            'rank_ir': self.rank_ir,
            'ic_t_stat': self.ic_t_stat,
            'ic_p_value': self.ic_p_value,
            'group_returns': self.group_returns,
            'long_short_return': self.long_short_return,
            'long_short_sharpe': self.long_short_sharpe,
            'turnover': self.turnover,
            'decay_half_life': self.decay_half_life,
            'overall_score': self.overall_score,
            'is_valid': self.is_valid,
            'validity_reasons': self.validity_reasons
        }


@dataclass
class ValidationConfig:
    """验证配置"""
    # IC阈值
    min_ic: float = 0.02
    min_ir: float = 0.3
    min_ic_positive_rate: float = 0.52
    
    # 统计显著性
    max_p_value: float = 0.05
    
    # 分组设置
    n_groups: int = 5
    
    # 多空收益阈值
    min_long_short_return: float = 0.0
    
    # 换手率阈值
    max_turnover: float = 0.5
    
    # 衰减阈值
    min_half_life: int = 5  # 天


class FactorValidator:
    """因子验证器"""
    
    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
    
    def validate(
        self,
        factor_values: pd.DataFrame,
        target: pd.DataFrame,
        prices: pd.DataFrame = None
    ) -> FactorMetrics:
        """
        全面验证因子有效性
        
        Args:
            factor_values: 因子值 (日期 x 股票)
            target: 目标变量，如未来收益 (日期 x 股票)
            prices: 价格数据，用于计算换手率 (可选)
        
        Returns:
            FactorMetrics: 因子评估指标
        """
        metrics = FactorMetrics()
        
        # 对齐数据
        common_dates = factor_values.index.intersection(target.index)
        common_stocks = factor_values.columns.intersection(target.columns)
        
        if len(common_dates) < 30 or len(common_stocks) < 10:
            metrics.validity_reasons.append("数据量不足")
            return metrics
        
        factor_aligned = factor_values.loc[common_dates, common_stocks]
        target_aligned = target.loc[common_dates, common_stocks]
        
        # 1. 计算IC指标
        self._calculate_ic_metrics(factor_aligned, target_aligned, metrics)
        
        # 2. 统计检验
        self._perform_statistical_tests(metrics)
        
        # 3. 分组回测
        self._calculate_group_returns(factor_aligned, target_aligned, metrics)
        
        # 4. 换手率分析
        if prices is not None:
            self._calculate_turnover(factor_aligned, metrics)
        
        # 5. 衰减分析
        self._analyze_decay(factor_aligned, target_aligned, metrics)
        
        # 6. 综合评分
        self._calculate_overall_score(metrics)
        
        # 7. 有效性判断
        self._determine_validity(metrics)
        
        return metrics
    
    def _calculate_ic_metrics(
        self,
        factor: pd.DataFrame,
        target: pd.DataFrame,
        metrics: FactorMetrics
    ):
        """计算IC相关指标"""
        ic_series = []
        rank_ic_series = []
        
        for date in factor.index:
            f_vals = factor.loc[date].dropna()
            t_vals = target.loc[date].dropna()
            
            common_idx = f_vals.index.intersection(t_vals.index)
            if len(common_idx) < 10:
                continue
            
            f = f_vals.loc[common_idx]
            t = t_vals.loc[common_idx]
            
            # Pearson IC
            if f.std() > 1e-10 and t.std() > 1e-10:
                ic = f.corr(t)
                if not np.isnan(ic):
                    ic_series.append(ic)
                
                # Rank IC (Spearman)
                rank_ic = f.rank().corr(t.rank())
                if not np.isnan(rank_ic):
                    rank_ic_series.append(rank_ic)
        
        if len(ic_series) < 20:
            return
        
        # IC指标
        ic_array = np.array(ic_series)
        metrics.ic_mean = np.mean(ic_array)
        metrics.ic_std = np.std(ic_array)
        metrics.ir = metrics.ic_mean / metrics.ic_std if metrics.ic_std > 1e-10 else 0
        metrics.ic_positive_rate = np.mean(ic_array > 0)
        
        # Rank IC指标
        rank_ic_array = np.array(rank_ic_series)
        metrics.rank_ic_mean = np.mean(rank_ic_array)
        metrics.rank_ic_std = np.std(rank_ic_array)
        metrics.rank_ir = metrics.rank_ic_mean / metrics.rank_ic_std if metrics.rank_ic_std > 1e-10 else 0
    
    def _perform_statistical_tests(self, metrics: FactorMetrics):
        """执行统计检验"""
        if metrics.ic_std > 0:
            # t检验：IC均值是否显著不为0
            # t = IC_mean / (IC_std / sqrt(n))
            # 这里简化使用IR作为t统计量的近似
            n = 252  # 假设一年数据
            metrics.ic_t_stat = metrics.ir * np.sqrt(n)
            metrics.ic_p_value = 2 * (1 - stats.t.cdf(abs(metrics.ic_t_stat), n - 1))
    
    def _calculate_group_returns(
        self,
        factor: pd.DataFrame,
        target: pd.DataFrame,
        metrics: FactorMetrics
    ):
        """计算分组收益"""
        n_groups = self.config.n_groups
        group_returns = {f'G{i+1}': [] for i in range(n_groups)}
        
        for date in factor.index:
            f_vals = factor.loc[date].dropna()
            t_vals = target.loc[date].dropna()
            
            common_idx = f_vals.index.intersection(t_vals.index)
            if len(common_idx) < n_groups * 5:
                continue
            
            f = f_vals.loc[common_idx]
            t = t_vals.loc[common_idx]
            
            # 按因子值分组
            try:
                groups = pd.qcut(f, n_groups, labels=False, duplicates='drop')
            except:
                continue
            
            for g in range(n_groups):
                mask = groups == g
                if mask.sum() > 0:
                    group_returns[f'G{g+1}'].append(t[mask].mean())
        
        # 计算各组平均收益
        for g in range(n_groups):
            returns = group_returns[f'G{g+1}']
            if returns:
                metrics.group_returns[f'G{g+1}'] = np.mean(returns)
        
        # 多空收益
        if f'G1' in metrics.group_returns and f'G{n_groups}' in metrics.group_returns:
            long_returns = group_returns[f'G{n_groups}']  # 最高组做多
            short_returns = group_returns['G1']           # 最低组做空
            
            if long_returns and short_returns:
                long_short = [l - s for l, s in zip(long_returns, short_returns)]
                metrics.long_short_return = np.mean(long_short) * 252  # 年化
                
                if len(long_short) > 1:
                    metrics.long_short_sharpe = (
                        np.mean(long_short) / np.std(long_short) * np.sqrt(252)
                        if np.std(long_short) > 0 else 0
                    )
    
    def _calculate_turnover(self, factor: pd.DataFrame, metrics: FactorMetrics):
        """计算换手率"""
        # 每日选择因子值最高的20%股票
        top_pct = 0.2
        
        prev_holdings = None
        turnovers = []
        
        for date in factor.index:
            f_vals = factor.loc[date].dropna()
            if len(f_vals) < 10:
                continue
            
            # 选择top股票
            threshold = f_vals.quantile(1 - top_pct)
            current_holdings = set(f_vals[f_vals >= threshold].index)
            
            if prev_holdings is not None:
                # 计算换手率
                turnover = 1 - len(current_holdings & prev_holdings) / max(len(current_holdings), 1)
                turnovers.append(turnover)
            
            prev_holdings = current_holdings
        
        if turnovers:
            metrics.turnover = np.mean(turnovers)
    
    def _analyze_decay(
        self,
        factor: pd.DataFrame,
        target: pd.DataFrame,
        metrics: FactorMetrics
    ):
        """分析因子衰减"""
        # 计算不同持有期的IC
        max_holding = 20
        holding_ics = []
        
        for holding in range(1, max_holding + 1):
            # 计算holding天后的收益
            future_return = target.shift(-holding)
            
            ic_series = []
            for date in factor.index[:-holding]:
                f_vals = factor.loc[date].dropna()
                t_vals = future_return.loc[date].dropna()
                
                common_idx = f_vals.index.intersection(t_vals.index)
                if len(common_idx) < 10:
                    continue
                
                f = f_vals.loc[common_idx]
                t = t_vals.loc[common_idx]
                
                if f.std() > 1e-10 and t.std() > 1e-10:
                    ic = f.rank().corr(t.rank())
                    if not np.isnan(ic):
                        ic_series.append(ic)
            
            if ic_series:
                holding_ics.append(np.mean(ic_series))
            else:
                holding_ics.append(0)
        
        # 估算半衰期
        if len(holding_ics) > 5 and holding_ics[0] > 0:
            half_ic = holding_ics[0] / 2
            for i, ic in enumerate(holding_ics):
                if ic < half_ic:
                    metrics.decay_half_life = i + 1
                    break
            else:
                metrics.decay_half_life = max_holding
    
    def _calculate_overall_score(self, metrics: FactorMetrics):
        """计算综合评分"""
        # 权重
        weights = {
            'ir': 0.3,
            'rank_ir': 0.2,
            'ic_positive_rate': 0.15,
            'long_short_sharpe': 0.2,
            'half_life': 0.15
        }
        
        # 归一化各指标
        scores = {}
        
        # IR: 0-1映射，IR=1对应满分
        scores['ir'] = min(max(metrics.ir, 0), 1)
        scores['rank_ir'] = min(max(metrics.rank_ir, 0), 1)
        
        # IC胜率: 50%-70%映射到0-1
        scores['ic_positive_rate'] = min(max((metrics.ic_positive_rate - 0.5) / 0.2, 0), 1)
        
        # 多空夏普: 0-2映射到0-1
        scores['long_short_sharpe'] = min(max(metrics.long_short_sharpe / 2, 0), 1)
        
        # 半衰期: 5-20天映射到0-1
        scores['half_life'] = min(max((metrics.decay_half_life - 5) / 15, 0), 1)
        
        # 加权求和
        metrics.overall_score = sum(
            scores.get(k, 0) * v for k, v in weights.items()
        )
    
    def _determine_validity(self, metrics: FactorMetrics):
        """判断因子有效性"""
        reasons = []
        is_valid = True
        
        # IC检验
        if abs(metrics.ic_mean) < self.config.min_ic:
            is_valid = False
            reasons.append(f"IC均值({metrics.ic_mean:.4f})低于阈值({self.config.min_ic})")
        
        # IR检验
        if metrics.ir < self.config.min_ir:
            is_valid = False
            reasons.append(f"IR({metrics.ir:.4f})低于阈值({self.config.min_ir})")
        
        # IC胜率检验
        if metrics.ic_positive_rate < self.config.min_ic_positive_rate:
            is_valid = False
            reasons.append(f"IC胜率({metrics.ic_positive_rate:.2%})低于阈值({self.config.min_ic_positive_rate:.0%})")
        
        # 统计显著性检验
        if metrics.ic_p_value > self.config.max_p_value:
            is_valid = False
            reasons.append(f"p值({metrics.ic_p_value:.4f})高于阈值({self.config.max_p_value})")
        
        # 换手率检验
        if metrics.turnover > self.config.max_turnover:
            is_valid = False
            reasons.append(f"换手率({metrics.turnover:.2%})高于阈值({self.config.max_turnover:.0%})")
        
        # 半衰期检验
        if metrics.decay_half_life < self.config.min_half_life:
            is_valid = False
            reasons.append(f"半衰期({metrics.decay_half_life}天)低于阈值({self.config.min_half_life}天)")
        
        metrics.is_valid = is_valid
        metrics.validity_reasons = reasons if not is_valid else ["所有指标通过验证"]
    
    def generate_report(self, metrics: FactorMetrics, factor_name: str = "Factor") -> str:
        """生成因子验证报告"""
        report = []
        report.append(f"# {factor_name} 有效性验证报告\n")
        
        # 总体评估
        status = "✅ 有效" if metrics.is_valid else "❌ 无效"
        report.append(f"## 总体评估: {status}\n")
        report.append(f"**综合评分**: {metrics.overall_score:.2f}/1.00\n")
        
        # IC分析
        report.append("## IC分析\n")
        report.append("| 指标 | 数值 | 评价 |")
        report.append("|:---|:---:|:---|")
        report.append(f"| IC均值 | {metrics.ic_mean:.4f} | {'✅' if abs(metrics.ic_mean) >= self.config.min_ic else '❌'} |")
        report.append(f"| IC标准差 | {metrics.ic_std:.4f} | - |")
        report.append(f"| IR | {metrics.ir:.4f} | {'✅' if metrics.ir >= self.config.min_ir else '❌'} |")
        report.append(f"| IC胜率 | {metrics.ic_positive_rate:.2%} | {'✅' if metrics.ic_positive_rate >= self.config.min_ic_positive_rate else '❌'} |")
        report.append(f"| Rank IC均值 | {metrics.rank_ic_mean:.4f} | - |")
        report.append(f"| Rank IR | {metrics.rank_ir:.4f} | - |")
        report.append("")
        
        # 统计检验
        report.append("## 统计检验\n")
        report.append(f"- **t统计量**: {metrics.ic_t_stat:.4f}")
        report.append(f"- **p值**: {metrics.ic_p_value:.4f} {'✅' if metrics.ic_p_value <= self.config.max_p_value else '❌'}")
        report.append("")
        
        # 分组收益
        if metrics.group_returns:
            report.append("## 分组收益\n")
            report.append("| 分组 | 日均收益 |")
            report.append("|:---:|:---:|")
            for g, ret in sorted(metrics.group_returns.items()):
                report.append(f"| {g} | {ret:.4%} |")
            report.append("")
            report.append(f"- **多空年化收益**: {metrics.long_short_return:.2%}")
            report.append(f"- **多空夏普比率**: {metrics.long_short_sharpe:.2f}")
            report.append("")
        
        # 其他指标
        report.append("## 其他指标\n")
        report.append(f"- **日均换手率**: {metrics.turnover:.2%}")
        report.append(f"- **IC半衰期**: {metrics.decay_half_life}天")
        report.append("")
        
        # 有效性原因
        report.append("## 验证详情\n")
        for reason in metrics.validity_reasons:
            report.append(f"- {reason}")
        
        return "\n".join(report)


def validate_factor(
    factor_values: pd.DataFrame,
    target: pd.DataFrame,
    prices: pd.DataFrame = None,
    config: ValidationConfig = None
) -> FactorMetrics:
    """
    便捷函数：验证单个因子
    """
    validator = FactorValidator(config)
    return validator.validate(factor_values, target, prices)


if __name__ == "__main__":
    # 测试代码
    print("=== 因子有效性验证系统测试 ===\n")
    
    # 创建模拟数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    stocks = [f'STOCK_{i}' for i in range(100)]
    
    # 模拟因子值
    factor_values = pd.DataFrame(
        np.random.randn(len(dates), len(stocks)),
        index=dates,
        columns=stocks
    )
    
    # 模拟目标收益（与因子有一定相关性）
    noise = np.random.randn(len(dates), len(stocks)) * 0.05
    target = factor_values * 0.02 + noise  # 因子有2%的预测能力
    
    # 验证因子
    validator = FactorValidator()
    metrics = validator.validate(factor_values, target)
    
    # 生成报告
    report = validator.generate_report(metrics, "测试因子")
    print(report)
    
    print("\n=== 测试完成 ===")
