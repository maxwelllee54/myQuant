"""
V4.1 增强版因子验证器
====================

以超越市场基准的长期稳定超额收益为核心目标，
对因子进行全面的有效性验证。

核心改进：
1. 引入市场基准对比
2. 以信息比率(IR)为核心筛选标准
3. 增加稳定性分析
4. 统计显著性检验
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

# 导入本模块的其他组件
try:
    from .benchmark_provider import BenchmarkProvider, get_benchmark_returns
    from .alpha_analyzer import AlphaAnalyzer, AlphaMetrics, StabilityAnalysis
except ImportError:
    from benchmark_provider import BenchmarkProvider, get_benchmark_returns
    from alpha_analyzer import AlphaAnalyzer, AlphaMetrics, StabilityAnalysis


@dataclass
class EnhancedFactorValidationResult:
    """增强版因子验证结果"""
    # 基础信息
    factor_name: str
    market: str
    benchmark: str
    
    # 有效性判断
    is_effective: bool
    effectiveness_reasons: List[str]
    
    # 传统指标
    ic_mean: float                # IC均值
    ic_std: float                 # IC标准差
    icir: float                   # IC信息比率
    
    # 超额收益指标 (V4.1新增)
    alpha_metrics: AlphaMetrics   # 超额收益核心指标
    stability: StabilityAnalysis  # 稳定性分析
    
    # 综合评分
    overall_score: float          # 综合得分 (0-100)
    grade: str                    # 等级 (A/B/C/D/F)


class EnhancedFactorValidator:
    """增强版因子验证器"""
    
    def __init__(
        self,
        market: str = "US",
        benchmark: Optional[str] = None,
        risk_free_rate: float = 0.02
    ):
        """
        初始化验证器
        
        Args:
            market: 市场类型 ("US" 或 "CN")
            benchmark: 基准代码，默认美股SPY，A股沪深300
            risk_free_rate: 无风险利率
        """
        self.market = market.upper()
        self.benchmark = benchmark or ("SPY" if self.market == "US" else "HS300")
        self.risk_free_rate = risk_free_rate
        
        self.benchmark_provider = BenchmarkProvider()
        self.alpha_analyzer = AlphaAnalyzer(risk_free_rate=risk_free_rate)
        
        # 有效性阈值
        self.thresholds = {
            'min_ic': 0.02,           # 最低IC均值
            'min_icir': 0.5,          # 最低ICIR
            'min_ir': 0.5,            # 最低信息比率
            'min_win_rate': 0.52,     # 最低胜率
            'max_pvalue': 0.05,       # 最大p值（显著性）
            'min_stability': 'C'      # 最低稳定性等级
        }
    
    def validate_factor(
        self,
        factor_values: pd.DataFrame,
        forward_returns: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        factor_name: str = "Factor"
    ) -> EnhancedFactorValidationResult:
        """
        验证因子有效性（以超越基准为核心标准）
        
        Args:
            factor_values: 因子值DataFrame，索引为日期，列为股票代码
            forward_returns: 未来收益率DataFrame，格式同上
            start_date: 开始日期
            end_date: 结束日期
            factor_name: 因子名称
            
        Returns:
            EnhancedFactorValidationResult: 验证结果
        """
        # 1. 计算传统IC指标
        ic_series = self._calculate_ic_series(factor_values, forward_returns)
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        icir = ic_mean / ic_std if ic_std > 0 else 0
        
        # 2. 构建因子组合收益
        factor_returns = self._build_factor_portfolio_returns(
            factor_values, forward_returns
        )
        
        # 3. 获取基准收益
        if start_date is None:
            start_date = factor_returns.index.min().strftime('%Y-%m-%d')
        if end_date is None:
            end_date = factor_returns.index.max().strftime('%Y-%m-%d')
        
        try:
            benchmark_returns = self.benchmark_provider.get_benchmark_returns(
                start_date, end_date, self.market, self.benchmark
            )
        except Exception as e:
            print(f"获取基准数据失败: {e}，使用模拟基准")
            # 使用因子收益的市场平均作为替代基准
            benchmark_returns = forward_returns.mean(axis=1)
        
        # 对齐数据
        aligned = pd.DataFrame({
            'factor': factor_returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned) < 20:
            raise ValueError("对齐后数据点数不足")
        
        factor_returns_aligned = aligned['factor']
        benchmark_returns_aligned = aligned['benchmark']
        
        # 4. 计算超额收益指标
        alpha_metrics = self.alpha_analyzer.calculate_alpha_metrics(
            factor_returns_aligned, benchmark_returns_aligned
        )
        
        # 5. 稳定性分析
        stability = self.alpha_analyzer.analyze_stability(
            factor_returns_aligned, benchmark_returns_aligned
        )
        
        # 6. 有效性判断
        is_effective, reasons = self._evaluate_effectiveness(
            ic_mean, icir, alpha_metrics, stability
        )
        
        # 7. 综合评分
        overall_score, grade = self._calculate_overall_score(
            ic_mean, icir, alpha_metrics, stability
        )
        
        return EnhancedFactorValidationResult(
            factor_name=factor_name,
            market=self.market,
            benchmark=self.benchmark,
            is_effective=is_effective,
            effectiveness_reasons=reasons,
            ic_mean=ic_mean,
            ic_std=ic_std,
            icir=icir,
            alpha_metrics=alpha_metrics,
            stability=stability,
            overall_score=overall_score,
            grade=grade
        )
    
    def _calculate_ic_series(
        self,
        factor_values: pd.DataFrame,
        forward_returns: pd.DataFrame
    ) -> pd.Series:
        """计算IC序列"""
        ic_values = []
        ic_dates = []
        
        common_dates = factor_values.index.intersection(forward_returns.index)
        
        for date in common_dates:
            factor_row = factor_values.loc[date].dropna()
            return_row = forward_returns.loc[date].dropna()
            
            common_stocks = factor_row.index.intersection(return_row.index)
            
            if len(common_stocks) >= 10:
                ic = factor_row[common_stocks].corr(return_row[common_stocks])
                if not np.isnan(ic):
                    ic_values.append(ic)
                    ic_dates.append(date)
        
        return pd.Series(ic_values, index=ic_dates)
    
    def _build_factor_portfolio_returns(
        self,
        factor_values: pd.DataFrame,
        forward_returns: pd.DataFrame,
        n_groups: int = 5
    ) -> pd.Series:
        """
        构建因子多空组合收益
        
        策略：做多因子值最高的20%股票，做空因子值最低的20%股票
        """
        portfolio_returns = []
        dates = []
        
        common_dates = factor_values.index.intersection(forward_returns.index)
        
        for date in common_dates:
            factor_row = factor_values.loc[date].dropna()
            return_row = forward_returns.loc[date].dropna()
            
            common_stocks = factor_row.index.intersection(return_row.index)
            
            if len(common_stocks) >= n_groups * 2:
                # 分组
                factor_sorted = factor_row[common_stocks].sort_values()
                n_per_group = len(factor_sorted) // n_groups
                
                # 做多最高组，做空最低组
                long_stocks = factor_sorted.index[-n_per_group:]
                short_stocks = factor_sorted.index[:n_per_group]
                
                long_return = return_row[long_stocks].mean()
                short_return = return_row[short_stocks].mean()
                
                # 多空收益
                ls_return = long_return - short_return
                
                portfolio_returns.append(ls_return)
                dates.append(date)
        
        return pd.Series(portfolio_returns, index=dates)
    
    def _evaluate_effectiveness(
        self,
        ic_mean: float,
        icir: float,
        alpha_metrics: AlphaMetrics,
        stability: StabilityAnalysis
    ) -> Tuple[bool, List[str]]:
        """评估因子有效性"""
        is_effective = True
        reasons = []
        
        # 1. IC检验
        if abs(ic_mean) < self.thresholds['min_ic']:
            is_effective = False
            reasons.append(f"IC均值 ({ic_mean:.4f}) 低于阈值 ({self.thresholds['min_ic']})")
        
        # 2. ICIR检验
        if abs(icir) < self.thresholds['min_icir']:
            is_effective = False
            reasons.append(f"ICIR ({icir:.2f}) 低于阈值 ({self.thresholds['min_icir']})")
        
        # 3. 信息比率检验 (核心指标)
        if alpha_metrics.information_ratio < self.thresholds['min_ir']:
            is_effective = False
            reasons.append(f"信息比率 ({alpha_metrics.information_ratio:.2f}) 低于阈值 ({self.thresholds['min_ir']})")
        
        # 4. 胜率检验
        if alpha_metrics.win_rate < self.thresholds['min_win_rate']:
            is_effective = False
            reasons.append(f"胜率 ({alpha_metrics.win_rate:.1%}) 低于阈值 ({self.thresholds['min_win_rate']:.1%})")
        
        # 5. 统计显著性检验
        if alpha_metrics.alpha_pvalue > self.thresholds['max_pvalue']:
            is_effective = False
            reasons.append(f"Alpha不显著 (p={alpha_metrics.alpha_pvalue:.4f} > {self.thresholds['max_pvalue']})")
        
        # 6. 稳定性检验
        grade_order = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'F': 1}
        min_grade = self.thresholds['min_stability']
        if grade_order.get(stability.stability_grade, 0) < grade_order.get(min_grade, 0):
            is_effective = False
            reasons.append(f"稳定性等级 ({stability.stability_grade}) 低于要求 ({min_grade})")
        
        if is_effective:
            reasons.append("✅ 因子通过所有有效性检验")
        
        return is_effective, reasons
    
    def _calculate_overall_score(
        self,
        ic_mean: float,
        icir: float,
        alpha_metrics: AlphaMetrics,
        stability: StabilityAnalysis
    ) -> Tuple[float, str]:
        """计算综合评分"""
        # 各维度得分 (0-100)
        
        # IC得分 (IC > 0.05 满分)
        ic_score = min(100, abs(ic_mean) / 0.05 * 100)
        
        # ICIR得分 (ICIR > 1.5 满分)
        icir_score = min(100, abs(icir) / 1.5 * 100)
        
        # 信息比率得分 (IR > 1.5 满分)
        ir_score = min(100, alpha_metrics.information_ratio / 1.5 * 100)
        
        # 胜率得分 (胜率 > 60% 满分)
        win_rate_score = min(100, (alpha_metrics.win_rate - 0.5) / 0.1 * 100)
        win_rate_score = max(0, win_rate_score)
        
        # 稳定性得分
        stability_score = stability.consistency_score
        
        # 显著性得分
        significance_score = 100 if alpha_metrics.is_significant else 0
        
        # 加权综合得分
        overall_score = (
            ic_score * 0.10 +
            icir_score * 0.15 +
            ir_score * 0.30 +      # 信息比率权重最高
            win_rate_score * 0.15 +
            stability_score * 0.20 +
            significance_score * 0.10
        )
        
        # 等级
        if overall_score >= 80:
            grade = "A"
        elif overall_score >= 65:
            grade = "B"
        elif overall_score >= 50:
            grade = "C"
        elif overall_score >= 35:
            grade = "D"
        else:
            grade = "F"
        
        return overall_score, grade
    
    def generate_validation_report(
        self,
        result: EnhancedFactorValidationResult
    ) -> str:
        """生成因子验证报告"""
        report = f"""# 因子验证报告: {result.factor_name}

## 1. 基本信息

| 项目 | 内容 |
|:---|:---|
| **因子名称** | {result.factor_name} |
| **市场** | {result.market} |
| **基准** | {result.benchmark} |
| **综合评分** | {result.overall_score:.1f}/100 ({result.grade}) |
| **是否有效** | {'✅ 是' if result.is_effective else '❌ 否'} |

---

## 2. 传统因子指标

| 指标 | 数值 | 评价 |
|:---|---:|:---|
| **IC均值** | {result.ic_mean:.4f} | {'良好' if abs(result.ic_mean) > 0.03 else '一般'} |
| **IC标准差** | {result.ic_std:.4f} | - |
| **ICIR** | {result.icir:.2f} | {'优秀' if abs(result.icir) > 1 else '良好' if abs(result.icir) > 0.5 else '一般'} |

---

## 3. 超额收益指标 (相对{result.benchmark})

| 指标 | 数值 | 说明 |
|:---|---:|:---|
| **年化Alpha** | {result.alpha_metrics.annual_alpha:.2%} | {'✅ 显著' if result.alpha_metrics.is_significant else '⚠️ 不显著'} |
| **信息比率** | {result.alpha_metrics.information_ratio:.2f} | {'优秀' if result.alpha_metrics.information_ratio > 1 else '良好' if result.alpha_metrics.information_ratio > 0.5 else '一般'} |
| **Beta** | {result.alpha_metrics.beta:.2f} | - |
| **跟踪误差** | {result.alpha_metrics.tracking_error:.2%} | - |
| **胜率** | {result.alpha_metrics.win_rate:.1%} | {'良好' if result.alpha_metrics.win_rate > 0.55 else '一般'} |

---

## 4. 稳定性分析

| 指标 | 数值 |
|:---|---:|
| **稳定性等级** | {result.stability.stability_grade} |
| **一致性得分** | {result.stability.consistency_score:.1f}/100 |
| **牛市Alpha** | {result.stability.bull_market_alpha:.2%} |
| **熊市Alpha** | {result.stability.bear_market_alpha:.2%} |

### 分年度表现

{result.stability.yearly_performance.to_markdown(index=False) if not result.stability.yearly_performance.empty else "暂无年度数据"}

---

## 5. 有效性判断

"""
        for reason in result.effectiveness_reasons:
            report += f"- {reason}\n"
        
        report += f"""
---

## 6. 投资建议

"""
        if result.is_effective:
            if result.grade in ['A', 'B']:
                report += """
### ✅ 强烈推荐

该因子展现出**显著且稳定的超额收益**，建议：
1. 作为核心选股因子使用
2. 可以给予较高的权重
3. 持续监控信息比率的变化
"""
            else:
                report += """
### ⚠️ 谨慎推荐

该因子有效但**稳定性一般**，建议：
1. 与其他因子组合使用
2. 控制单因子权重
3. 密切监控表现
"""
        else:
            report += """
### ❌ 不推荐

该因子**未通过有效性检验**，建议：
1. 重新审视因子逻辑
2. 检查是否存在过拟合
3. 考虑其他替代因子
"""
        
        return report


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("V4.1 增强版因子验证器测试")
    print("=" * 60)
    
    # 生成模拟数据
    np.random.seed(42)
    n_days = 252
    n_stocks = 100
    
    dates = pd.date_range(start='2024-01-01', periods=n_days, freq='B')
    stocks = [f'STOCK_{i:03d}' for i in range(n_stocks)]
    
    # 模拟因子值
    factor_values = pd.DataFrame(
        np.random.randn(n_days, n_stocks),
        index=dates,
        columns=stocks
    )
    
    # 模拟未来收益 (与因子有一定相关性)
    noise = np.random.randn(n_days, n_stocks) * 0.02
    forward_returns = factor_values * 0.001 + noise
    forward_returns = pd.DataFrame(forward_returns, index=dates, columns=stocks)
    
    # 创建验证器
    validator = EnhancedFactorValidator(market="US", benchmark="SPY")
    
    # 验证因子
    print("\n正在验证因子...")
    try:
        result = validator.validate_factor(
            factor_values=factor_values,
            forward_returns=forward_returns,
            factor_name="测试因子"
        )
        
        print(f"\n验证结果:")
        print(f"  综合评分: {result.overall_score:.1f}/100 ({result.grade})")
        print(f"  是否有效: {'✅ 是' if result.is_effective else '❌ 否'}")
        print(f"  IC均值: {result.ic_mean:.4f}")
        print(f"  ICIR: {result.icir:.2f}")
        print(f"  信息比率: {result.alpha_metrics.information_ratio:.2f}")
        print(f"  年化Alpha: {result.alpha_metrics.annual_alpha:.2%}")
        
        print("\n有效性判断:")
        for reason in result.effectiveness_reasons:
            print(f"  - {reason}")
            
    except Exception as e:
        print(f"验证失败: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
