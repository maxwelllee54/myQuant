"""
V4.1 超额收益分析与稳定性验证模块
================================

以超越市场基准的长期稳定超额收益为核心目标，
对因子/策略进行全面的有效性验证。

核心指标：
- Alpha (年化): 相对基准的超额收益
- Beta: 相对基准的敏感性
- Information Ratio (IR): 超额收益的风险调整后表现
- Tracking Error: 跟踪误差
- 胜率: 超越基准的交易日比例

稳定性分析：
- 滚动窗口分析
- 分年度表现
- 牛熊市分离测试
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


@dataclass
class AlphaMetrics:
    """超额收益核心指标"""
    # 基础指标
    total_return: float           # 策略总收益
    benchmark_return: float       # 基准总收益
    excess_return: float          # 超额总收益
    
    # 年化指标
    annual_return: float          # 策略年化收益
    annual_benchmark: float       # 基准年化收益
    annual_alpha: float           # 年化Alpha
    
    # 风险指标
    volatility: float             # 策略年化波动率
    benchmark_volatility: float   # 基准年化波动率
    tracking_error: float         # 年化跟踪误差
    
    # 风险调整指标
    beta: float                   # Beta系数
    information_ratio: float      # 信息比率
    sharpe_ratio: float           # 夏普比率
    
    # 胜率指标
    win_rate: float               # 超越基准的交易日比例
    up_capture: float             # 上涨捕获率
    down_capture: float           # 下跌捕获率
    
    # 统计显著性
    alpha_tstat: float            # Alpha的t统计量
    alpha_pvalue: float           # Alpha的p值
    is_significant: bool          # Alpha是否显著 (p < 0.05)


@dataclass
class StabilityAnalysis:
    """稳定性分析结果"""
    # 滚动分析
    rolling_alpha: pd.Series      # 滚动Alpha
    rolling_ir: pd.Series         # 滚动信息比率
    rolling_beta: pd.Series       # 滚动Beta
    
    # 分年度表现
    yearly_performance: pd.DataFrame  # 年度表现对比
    
    # 牛熊市分析
    bull_market_alpha: float      # 牛市Alpha
    bear_market_alpha: float      # 熊市Alpha
    bull_periods: int             # 牛市天数
    bear_periods: int             # 熊市天数
    
    # 稳定性评分
    consistency_score: float      # 一致性得分 (0-100)
    stability_grade: str          # 稳定性等级 (A/B/C/D/F)


class AlphaAnalyzer:
    """超额收益分析器"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        初始化分析器
        
        Args:
            risk_free_rate: 无风险利率，默认2%
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = 252
    
    def calculate_alpha_metrics(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> AlphaMetrics:
        """
        计算超额收益核心指标
        
        Args:
            strategy_returns: 策略日收益率序列
            benchmark_returns: 基准日收益率序列
            
        Returns:
            AlphaMetrics: 超额收益指标
        """
        # 对齐数据
        aligned = pd.DataFrame({
            'strategy': strategy_returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned) < 20:
            raise ValueError("数据点数不足，至少需要20个交易日")
        
        strategy = aligned['strategy']
        benchmark = aligned['benchmark']
        excess = strategy - benchmark
        
        n_days = len(aligned)
        n_years = n_days / self.trading_days
        
        # 基础收益
        total_return = (1 + strategy).prod() - 1
        benchmark_return = (1 + benchmark).prod() - 1
        excess_return = total_return - benchmark_return
        
        # 年化收益
        annual_return = (1 + total_return) ** (1 / n_years) - 1
        annual_benchmark = (1 + benchmark_return) ** (1 / n_years) - 1
        
        # 波动率
        volatility = strategy.std() * np.sqrt(self.trading_days)
        benchmark_volatility = benchmark.std() * np.sqrt(self.trading_days)
        tracking_error = excess.std() * np.sqrt(self.trading_days)
        
        # Beta和Alpha (CAPM回归)
        cov_matrix = np.cov(strategy, benchmark)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 1.0
        
        # 年化Alpha (Jensen's Alpha)
        annual_alpha = annual_return - self.risk_free_rate - beta * (annual_benchmark - self.risk_free_rate)
        
        # 信息比率
        information_ratio = annual_alpha / tracking_error if tracking_error > 0 else 0
        
        # 夏普比率
        sharpe_ratio = (annual_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        # 胜率
        win_rate = (excess > 0).mean()
        
        # 上涨/下跌捕获率
        up_days = benchmark > 0
        down_days = benchmark < 0
        
        if up_days.sum() > 0:
            up_capture = strategy[up_days].mean() / benchmark[up_days].mean()
        else:
            up_capture = 1.0
            
        if down_days.sum() > 0:
            down_capture = strategy[down_days].mean() / benchmark[down_days].mean()
        else:
            down_capture = 1.0
        
        # Alpha显著性检验 (t检验)
        daily_alpha = excess.mean() * self.trading_days  # 年化日均超额
        alpha_std = excess.std() * np.sqrt(self.trading_days)
        alpha_tstat = daily_alpha / (alpha_std / np.sqrt(n_days)) if alpha_std > 0 else 0
        alpha_pvalue = 2 * (1 - stats.t.cdf(abs(alpha_tstat), df=n_days-1))
        is_significant = alpha_pvalue < 0.05
        
        return AlphaMetrics(
            total_return=total_return,
            benchmark_return=benchmark_return,
            excess_return=excess_return,
            annual_return=annual_return,
            annual_benchmark=annual_benchmark,
            annual_alpha=annual_alpha,
            volatility=volatility,
            benchmark_volatility=benchmark_volatility,
            tracking_error=tracking_error,
            beta=beta,
            information_ratio=information_ratio,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            up_capture=up_capture,
            down_capture=down_capture,
            alpha_tstat=alpha_tstat,
            alpha_pvalue=alpha_pvalue,
            is_significant=is_significant
        )
    
    def analyze_stability(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        rolling_window: int = 63  # 约3个月
    ) -> StabilityAnalysis:
        """
        分析策略的稳定性
        
        Args:
            strategy_returns: 策略日收益率序列
            benchmark_returns: 基准日收益率序列
            rolling_window: 滚动窗口大小（交易日）
            
        Returns:
            StabilityAnalysis: 稳定性分析结果
        """
        # 对齐数据
        aligned = pd.DataFrame({
            'strategy': strategy_returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        strategy = aligned['strategy']
        benchmark = aligned['benchmark']
        excess = strategy - benchmark
        
        # 1. 滚动分析
        rolling_alpha = excess.rolling(window=rolling_window).mean() * self.trading_days
        rolling_std = excess.rolling(window=rolling_window).std() * np.sqrt(self.trading_days)
        rolling_ir = rolling_alpha / rolling_std
        rolling_ir = rolling_ir.replace([np.inf, -np.inf], np.nan)
        
        # 滚动Beta
        def calc_rolling_beta(window_data):
            if len(window_data) < 20:
                return np.nan
            s = window_data['strategy']
            b = window_data['benchmark']
            cov = np.cov(s, b)
            return cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else np.nan
        
        rolling_beta = aligned.rolling(window=rolling_window).apply(
            lambda x: np.nan, raw=False
        )['strategy']  # placeholder
        
        # 简化的滚动beta计算
        rolling_beta_values = []
        for i in range(len(aligned)):
            if i < rolling_window:
                rolling_beta_values.append(np.nan)
            else:
                window = aligned.iloc[i-rolling_window:i]
                s = window['strategy']
                b = window['benchmark']
                cov = np.cov(s, b)
                beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else np.nan
                rolling_beta_values.append(beta)
        rolling_beta = pd.Series(rolling_beta_values, index=aligned.index)
        
        # 2. 分年度表现
        aligned['year'] = aligned.index.year
        yearly_data = []
        
        for year in aligned['year'].unique():
            year_mask = aligned['year'] == year
            year_strategy = aligned.loc[year_mask, 'strategy']
            year_benchmark = aligned.loc[year_mask, 'benchmark']
            
            strategy_ret = (1 + year_strategy).prod() - 1
            benchmark_ret = (1 + year_benchmark).prod() - 1
            excess_ret = strategy_ret - benchmark_ret
            
            yearly_data.append({
                'year': year,
                'strategy_return': strategy_ret,
                'benchmark_return': benchmark_ret,
                'excess_return': excess_ret,
                'outperform': excess_ret > 0
            })
        
        yearly_performance = pd.DataFrame(yearly_data)
        
        # 3. 牛熊市分析
        # 定义：基准累积收益上升为牛市，下降为熊市
        benchmark_cum = (1 + benchmark).cumprod()
        benchmark_peak = benchmark_cum.expanding().max()
        drawdown = (benchmark_cum - benchmark_peak) / benchmark_peak
        
        # 简化定义：回撤超过10%为熊市
        is_bear = drawdown < -0.10
        
        bull_excess = excess[~is_bear]
        bear_excess = excess[is_bear]
        
        bull_market_alpha = bull_excess.mean() * self.trading_days if len(bull_excess) > 0 else 0
        bear_market_alpha = bear_excess.mean() * self.trading_days if len(bear_excess) > 0 else 0
        
        # 4. 一致性评分
        # 基于：年度胜率、滚动IR稳定性、牛熊市表现
        yearly_win_rate = yearly_performance['outperform'].mean() if len(yearly_performance) > 0 else 0
        ir_stability = 1 - (rolling_ir.std() / abs(rolling_ir.mean())) if rolling_ir.mean() != 0 else 0
        ir_stability = max(0, min(1, ir_stability))
        
        # 牛熊市平衡性
        if bull_market_alpha != 0 or bear_market_alpha != 0:
            bull_bear_balance = min(bull_market_alpha, bear_market_alpha) / max(abs(bull_market_alpha), abs(bear_market_alpha))
            bull_bear_balance = max(0, bull_bear_balance)
        else:
            bull_bear_balance = 0
        
        consistency_score = (
            yearly_win_rate * 40 +
            ir_stability * 30 +
            bull_bear_balance * 30
        )
        
        # 稳定性等级
        if consistency_score >= 80:
            stability_grade = "A"
        elif consistency_score >= 60:
            stability_grade = "B"
        elif consistency_score >= 40:
            stability_grade = "C"
        elif consistency_score >= 20:
            stability_grade = "D"
        else:
            stability_grade = "F"
        
        return StabilityAnalysis(
            rolling_alpha=rolling_alpha,
            rolling_ir=rolling_ir,
            rolling_beta=rolling_beta,
            yearly_performance=yearly_performance,
            bull_market_alpha=bull_market_alpha,
            bear_market_alpha=bear_market_alpha,
            bull_periods=int((~is_bear).sum()),
            bear_periods=int(is_bear.sum()),
            consistency_score=consistency_score,
            stability_grade=stability_grade
        )
    
    def generate_alpha_report(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        strategy_name: str = "策略",
        benchmark_name: str = "基准"
    ) -> str:
        """
        生成超额收益分析报告
        
        Args:
            strategy_returns: 策略日收益率
            benchmark_returns: 基准日收益率
            strategy_name: 策略名称
            benchmark_name: 基准名称
            
        Returns:
            str: Markdown格式的分析报告
        """
        metrics = self.calculate_alpha_metrics(strategy_returns, benchmark_returns)
        stability = self.analyze_stability(strategy_returns, benchmark_returns)
        
        report = f"""# 超额收益分析报告

## 1. 执行摘要

| 指标 | {strategy_name} | {benchmark_name} | 差异 |
|:---|---:|---:|---:|
| **总收益** | {metrics.total_return:.2%} | {metrics.benchmark_return:.2%} | {metrics.excess_return:.2%} |
| **年化收益** | {metrics.annual_return:.2%} | {metrics.annual_benchmark:.2%} | {metrics.annual_alpha:.2%} |
| **年化波动率** | {metrics.volatility:.2%} | {metrics.benchmark_volatility:.2%} | - |
| **夏普比率** | {metrics.sharpe_ratio:.2f} | - | - |

### 核心结论

- **年化Alpha**: {metrics.annual_alpha:.2%} {'✅ 统计显著' if metrics.is_significant else '⚠️ 统计不显著'} (p={metrics.alpha_pvalue:.4f})
- **信息比率 (IR)**: {metrics.information_ratio:.2f} {'(优秀)' if metrics.information_ratio > 1 else '(良好)' if metrics.information_ratio > 0.5 else '(一般)'}
- **稳定性等级**: {stability.stability_grade} (得分: {stability.consistency_score:.1f}/100)

---

## 2. 超额收益核心指标

### 2.1 风险调整指标

| 指标 | 数值 | 说明 |
|:---|---:|:---|
| **Beta** | {metrics.beta:.2f} | 相对基准的敏感性 |
| **年化Alpha** | {metrics.annual_alpha:.2%} | Jensen's Alpha |
| **信息比率** | {metrics.information_ratio:.2f} | Alpha / 跟踪误差 |
| **跟踪误差** | {metrics.tracking_error:.2%} | 相对基准的偏离度 |

### 2.2 胜率指标

| 指标 | 数值 | 说明 |
|:---|---:|:---|
| **日胜率** | {metrics.win_rate:.1%} | 超越基准的交易日比例 |
| **上涨捕获率** | {metrics.up_capture:.2f} | 基准上涨时的参与度 |
| **下跌捕获率** | {metrics.down_capture:.2f} | 基准下跌时的参与度 |

> **理想状态**: 上涨捕获率 > 1, 下跌捕获率 < 1

---

## 3. 稳定性分析

### 3.1 分年度表现

{stability.yearly_performance.to_markdown(index=False) if not stability.yearly_performance.empty else "暂无年度数据"}

### 3.2 牛熊市表现

| 市场状态 | 交易日数 | 年化Alpha |
|:---|---:|---:|
| **牛市** | {stability.bull_periods} | {stability.bull_market_alpha:.2%} |
| **熊市** | {stability.bear_periods} | {stability.bear_market_alpha:.2%} |

### 3.3 稳定性评估

- **一致性得分**: {stability.consistency_score:.1f}/100
- **稳定性等级**: {stability.stability_grade}

---

## 4. 统计显著性检验

| 检验项 | 统计量 | p值 | 结论 |
|:---|---:|---:|:---|
| **Alpha显著性** | t={metrics.alpha_tstat:.2f} | {metrics.alpha_pvalue:.4f} | {'显著' if metrics.is_significant else '不显著'} |

> **注**: 在5%显著性水平下，p值 < 0.05 表示Alpha统计显著。

---

## 5. 投资建议

"""
        # 根据指标给出建议
        if metrics.is_significant and metrics.information_ratio > 0.5:
            if stability.stability_grade in ['A', 'B']:
                report += """
### ✅ 推荐采用

该策略/因子展现出**统计显著的超额收益**，且**稳定性良好**。建议：
1. 可以作为核心策略/因子使用
2. 持续监控信息比率的变化
3. 关注牛熊市表现的平衡性
"""
            else:
                report += """
### ⚠️ 谨慎采用

该策略/因子虽有显著Alpha，但**稳定性一般**。建议：
1. 降低配置权重
2. 与其他稳定策略组合使用
3. 密切监控表现波动
"""
        else:
            report += """
### ❌ 不推荐采用

该策略/因子的超额收益**不显著**或**不稳定**。建议：
1. 重新审视策略逻辑
2. 检查是否存在过拟合
3. 考虑其他替代方案
"""
        
        return report
    
    def evaluate_factor_effectiveness(
        self,
        factor_returns: pd.Series,
        benchmark_returns: pd.Series,
        min_ir: float = 0.5,
        min_win_rate: float = 0.52,
        require_significance: bool = True
    ) -> Tuple[bool, Dict]:
        """
        评估因子是否有效（以超越基准为标准）
        
        Args:
            factor_returns: 因子收益率序列
            benchmark_returns: 基准收益率序列
            min_ir: 最低信息比率要求
            min_win_rate: 最低胜率要求
            require_significance: 是否要求统计显著
            
        Returns:
            Tuple[bool, Dict]: (是否有效, 详细指标)
        """
        metrics = self.calculate_alpha_metrics(factor_returns, benchmark_returns)
        stability = self.analyze_stability(factor_returns, benchmark_returns)
        
        # 有效性判断
        is_effective = True
        reasons = []
        
        # 检查信息比率
        if metrics.information_ratio < min_ir:
            is_effective = False
            reasons.append(f"信息比率 ({metrics.information_ratio:.2f}) 低于阈值 ({min_ir})")
        
        # 检查胜率
        if metrics.win_rate < min_win_rate:
            is_effective = False
            reasons.append(f"胜率 ({metrics.win_rate:.1%}) 低于阈值 ({min_win_rate:.1%})")
        
        # 检查统计显著性
        if require_significance and not metrics.is_significant:
            is_effective = False
            reasons.append(f"Alpha不显著 (p={metrics.alpha_pvalue:.4f})")
        
        # 检查稳定性
        if stability.stability_grade in ['D', 'F']:
            is_effective = False
            reasons.append(f"稳定性等级过低 ({stability.stability_grade})")
        
        details = {
            'is_effective': is_effective,
            'reasons': reasons,
            'metrics': {
                'annual_alpha': metrics.annual_alpha,
                'information_ratio': metrics.information_ratio,
                'win_rate': metrics.win_rate,
                'is_significant': metrics.is_significant,
                'stability_grade': stability.stability_grade
            }
        }
        
        return is_effective, details


# 便捷函数
def evaluate_vs_benchmark(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series
) -> AlphaMetrics:
    """
    便捷函数：评估策略相对基准的表现
    """
    analyzer = AlphaAnalyzer()
    return analyzer.calculate_alpha_metrics(strategy_returns, benchmark_returns)


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("V4.1 超额收益分析模块测试")
    print("=" * 60)
    
    # 生成模拟数据
    np.random.seed(42)
    n_days = 504  # 约2年
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')
    
    # 模拟基准收益 (年化10%，波动率15%)
    benchmark_returns = pd.Series(
        np.random.normal(0.10/252, 0.15/np.sqrt(252), n_days),
        index=dates
    )
    
    # 模拟策略收益 (年化15%，波动率18%，有超额收益)
    strategy_returns = benchmark_returns + pd.Series(
        np.random.normal(0.05/252, 0.05/np.sqrt(252), n_days),
        index=dates
    )
    
    # 创建分析器
    analyzer = AlphaAnalyzer()
    
    # 1. 计算核心指标
    print("\n1. 超额收益核心指标:")
    metrics = analyzer.calculate_alpha_metrics(strategy_returns, benchmark_returns)
    print(f"   年化Alpha: {metrics.annual_alpha:.2%}")
    print(f"   信息比率: {metrics.information_ratio:.2f}")
    print(f"   Beta: {metrics.beta:.2f}")
    print(f"   胜率: {metrics.win_rate:.1%}")
    print(f"   Alpha显著性: {'显著' if metrics.is_significant else '不显著'} (p={metrics.alpha_pvalue:.4f})")
    
    # 2. 稳定性分析
    print("\n2. 稳定性分析:")
    stability = analyzer.analyze_stability(strategy_returns, benchmark_returns)
    print(f"   稳定性等级: {stability.stability_grade}")
    print(f"   一致性得分: {stability.consistency_score:.1f}/100")
    print(f"   牛市Alpha: {stability.bull_market_alpha:.2%}")
    print(f"   熊市Alpha: {stability.bear_market_alpha:.2%}")
    
    # 3. 有效性评估
    print("\n3. 因子有效性评估:")
    is_effective, details = analyzer.evaluate_factor_effectiveness(
        strategy_returns, benchmark_returns
    )
    print(f"   是否有效: {'✅ 是' if is_effective else '❌ 否'}")
    if not is_effective:
        print(f"   原因: {'; '.join(details['reasons'])}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
