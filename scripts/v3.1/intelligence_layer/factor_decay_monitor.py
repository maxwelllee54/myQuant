#!/usr/bin/env python3
"""
V3.1 Factor Decay Monitor
因子衰减监控模块

监控投资组合或策略在关键风格因子上的暴露度变化，
估算因子半衰期，生成再平衡预警信号。

理论基础：
- 因子暴露会随时间衰减，不同因子衰减速度不同
- 价值因子半衰期最长（>36个月），动量因子较短（~3个月）
- 通过监控因子暴露变化，可以判断策略是否需要再平衡

参考文献：
- Flint & Vermaak (2023). "Factor Information Decay: A Global Study." Journal of Portfolio Management.

作者: Manus AI
版本: 3.1
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from scipy import stats
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore')


@dataclass
class FactorExposure:
    """因子暴露数据结构"""
    factor_name: str
    beta: float
    t_stat: float
    p_value: float
    r_squared: float
    is_significant: bool


@dataclass
class DecayAnalysis:
    """衰减分析结果"""
    factor_name: str
    initial_exposure: float
    current_exposure: float
    half_life_days: Optional[float]
    decay_rate: float
    exposure_series: List[float]
    dates: List[str]
    rebalance_signal: bool
    signal_reason: Optional[str]


@dataclass
class RebalanceSignal:
    """再平衡信号"""
    timestamp: datetime
    signal_type: str  # "exposure_decay", "threshold_breach", "half_life_warning"
    factor_name: str
    current_exposure: float
    threshold: float
    urgency: str  # "low", "medium", "high"
    recommendation: str


class FactorDecayMonitor:
    """
    因子衰减监控器
    
    监控策略的因子暴露变化，估算半衰期，生成再平衡信号。
    """
    
    # 各因子的参考半衰期（月）和建议再平衡周期（月）
    # 基于 Flint & Vermaak (2023) 的研究结果
    FACTOR_REFERENCE = {
        "value": {"half_life_months": 36, "rebalance_months": 4, "threshold": 0.5},
        "momentum": {"half_life_months": 6, "rebalance_months": 3, "threshold": 0.5},
        "quality": {"half_life_months": 26, "rebalance_months": 5, "threshold": 0.5},
        "low_volatility": {"half_life_months": 12, "rebalance_months": 6, "threshold": 0.5},
        "size": {"half_life_months": 18, "rebalance_months": 4, "threshold": 0.5},
        "investment": {"half_life_months": 3, "rebalance_months": 1, "threshold": 0.5},
        "market": {"half_life_months": None, "rebalance_months": 12, "threshold": 0.3},
    }
    
    def __init__(
        self,
        exposure_threshold: float = 0.5,
        significance_level: float = 0.05,
        verbose: bool = True
    ):
        """
        初始化因子衰减监控器
        
        Args:
            exposure_threshold: 因子暴露阈值，低于此值触发再平衡信号
            significance_level: 统计显著性水平
            verbose: 是否打印详细日志
        """
        self.exposure_threshold = exposure_threshold
        self.significance_level = significance_level
        self.verbose = verbose
        
        # 存储历史暴露数据
        self._exposure_history: Dict[str, List[Tuple[datetime, float]]] = {}
        
        if self.verbose:
            print("[FactorDecayMonitor] 初始化完成")
    
    def calculate_factor_exposure(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        window: int = 60
    ) -> Dict[str, FactorExposure]:
        """
        计算组合对各因子的暴露度（Beta）
        
        使用滚动回归计算因子暴露：
        R_p = α + Σ(β_i * F_i) + ε
        
        Args:
            portfolio_returns: 组合收益率序列
            factor_returns: 因子收益率DataFrame，列为因子名
            window: 回归窗口（交易日）
        
        Returns:
            各因子的暴露度字典
        """
        exposures = {}
        
        # 对齐数据
        aligned = pd.concat([portfolio_returns, factor_returns], axis=1).dropna()
        if len(aligned) < window:
            if self.verbose:
                print(f"[FactorDecayMonitor] 数据不足，需要至少 {window} 个数据点")
            return exposures
        
        # 使用最近的数据进行回归
        recent_data = aligned.tail(window)
        y = recent_data.iloc[:, 0].values
        X = recent_data.iloc[:, 1:].values
        
        # 添加常数项
        X_with_const = np.column_stack([np.ones(len(X)), X])
        
        try:
            # OLS回归
            beta, residuals, rank, s = np.linalg.lstsq(X_with_const, y, rcond=None)
            
            # 计算统计量
            y_pred = X_with_const @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            # 计算标准误和t统计量
            n = len(y)
            k = X_with_const.shape[1]
            mse = ss_res / (n - k) if n > k else 0
            
            if mse > 0:
                var_beta = mse * np.linalg.inv(X_with_const.T @ X_with_const).diagonal()
                se_beta = np.sqrt(np.maximum(var_beta, 0))
            else:
                se_beta = np.ones(len(beta))
            
            # 构建结果
            factor_names = factor_returns.columns.tolist()
            for i, factor_name in enumerate(factor_names):
                beta_i = beta[i + 1]  # 跳过常数项
                se_i = se_beta[i + 1] if i + 1 < len(se_beta) else 1
                t_stat = beta_i / se_i if se_i > 0 else 0
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - k))
                
                exposures[factor_name] = FactorExposure(
                    factor_name=factor_name,
                    beta=beta_i,
                    t_stat=t_stat,
                    p_value=p_value,
                    r_squared=r_squared,
                    is_significant=p_value < self.significance_level
                )
                
                # 记录历史
                if factor_name not in self._exposure_history:
                    self._exposure_history[factor_name] = []
                self._exposure_history[factor_name].append((datetime.now(), beta_i))
            
            if self.verbose:
                print(f"[FactorDecayMonitor] 计算因子暴露完成")
                print(f"  R² = {r_squared:.4f}")
                for name, exp in exposures.items():
                    sig = "***" if exp.is_significant else ""
                    print(f"  {name}: β={exp.beta:.4f} (t={exp.t_stat:.2f}){sig}")
        
        except Exception as e:
            if self.verbose:
                print(f"[FactorDecayMonitor] 计算因子暴露失败: {e}")
        
        return exposures
    
    def calculate_rolling_exposure(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        window: int = 60,
        step: int = 20
    ) -> Dict[str, pd.Series]:
        """
        计算滚动因子暴露时间序列
        
        Args:
            portfolio_returns: 组合收益率序列
            factor_returns: 因子收益率DataFrame
            window: 回归窗口
            step: 滚动步长
        
        Returns:
            各因子的暴露时间序列
        """
        aligned = pd.concat([portfolio_returns, factor_returns], axis=1).dropna()
        
        if len(aligned) < window + step:
            if self.verbose:
                print("[FactorDecayMonitor] 数据不足以计算滚动暴露")
            return {}
        
        rolling_exposures = {col: [] for col in factor_returns.columns}
        dates = []
        
        for end in range(window, len(aligned), step):
            start = end - window
            sub_data = aligned.iloc[start:end]
            
            y = sub_data.iloc[:, 0].values
            X = sub_data.iloc[:, 1:].values
            X_with_const = np.column_stack([np.ones(len(X)), X])
            
            try:
                beta, _, _, _ = np.linalg.lstsq(X_with_const, y, rcond=None)
                
                for i, col in enumerate(factor_returns.columns):
                    rolling_exposures[col].append(beta[i + 1])
                
                dates.append(aligned.index[end - 1])
            except:
                continue
        
        # 转换为Series
        result = {}
        for col in factor_returns.columns:
            if rolling_exposures[col]:
                result[col] = pd.Series(rolling_exposures[col], index=dates[:len(rolling_exposures[col])])
        
        if self.verbose:
            print(f"[FactorDecayMonitor] 计算滚动暴露完成")
            print(f"  时间点数量: {len(dates)}")
        
        return result
    
    def estimate_half_life(
        self,
        exposure_series: pd.Series,
        method: str = "exponential"
    ) -> Optional[float]:
        """
        估算因子暴露的半衰期
        
        使用指数衰减模型拟合：
        E(t) = E_0 * exp(-λt)
        半衰期 = ln(2) / λ
        
        Args:
            exposure_series: 因子暴露时间序列
            method: 估算方法 ("exponential", "linear")
        
        Returns:
            半衰期（天数），如果无法估算则返回None
        """
        if len(exposure_series) < 5:
            return None
        
        # 归一化暴露值
        initial_exposure = abs(exposure_series.iloc[0])
        if initial_exposure < 1e-6:
            return None
        
        normalized = np.abs(exposure_series.values) / initial_exposure
        t = np.arange(len(normalized))
        
        try:
            if method == "exponential":
                # 指数衰减拟合
                def exp_decay(t, lambda_):
                    return np.exp(-lambda_ * t)
                
                popt, _ = curve_fit(exp_decay, t, normalized, p0=[0.01], bounds=(0, 1))
                lambda_ = popt[0]
                
                if lambda_ > 0:
                    half_life = np.log(2) / lambda_
                    return half_life
            
            elif method == "linear":
                # 线性回归估算衰减率
                slope, intercept, r_value, p_value, std_err = stats.linregress(t, normalized)
                
                if slope < 0:
                    # 估算到达0.5的时间
                    half_life = (0.5 - intercept) / slope
                    return max(0, half_life)
        
        except Exception as e:
            if self.verbose:
                print(f"[FactorDecayMonitor] 半衰期估算失败: {e}")
        
        return None
    
    def analyze_decay(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        window: int = 60,
        step: int = 10
    ) -> Dict[str, DecayAnalysis]:
        """
        分析各因子的衰减情况
        
        Args:
            portfolio_returns: 组合收益率序列
            factor_returns: 因子收益率DataFrame
            window: 回归窗口
            step: 滚动步长
        
        Returns:
            各因子的衰减分析结果
        """
        rolling_exposures = self.calculate_rolling_exposure(
            portfolio_returns, factor_returns, window, step
        )
        
        results = {}
        
        for factor_name, exposure_series in rolling_exposures.items():
            if len(exposure_series) < 3:
                continue
            
            initial_exposure = exposure_series.iloc[0]
            current_exposure = exposure_series.iloc[-1]
            
            # 计算衰减率
            if abs(initial_exposure) > 1e-6:
                decay_rate = (abs(initial_exposure) - abs(current_exposure)) / abs(initial_exposure)
            else:
                decay_rate = 0
            
            # 估算半衰期
            half_life = self.estimate_half_life(exposure_series)
            
            # 判断是否需要再平衡
            rebalance_signal = False
            signal_reason = None
            
            ref = self.FACTOR_REFERENCE.get(factor_name.lower(), {})
            threshold = ref.get("threshold", self.exposure_threshold)
            
            if abs(current_exposure) < threshold * abs(initial_exposure):
                rebalance_signal = True
                signal_reason = f"因子暴露已衰减至初始值的{abs(current_exposure/initial_exposure)*100:.1f}%"
            
            results[factor_name] = DecayAnalysis(
                factor_name=factor_name,
                initial_exposure=initial_exposure,
                current_exposure=current_exposure,
                half_life_days=half_life,
                decay_rate=decay_rate,
                exposure_series=exposure_series.tolist(),
                dates=[str(d) for d in exposure_series.index],
                rebalance_signal=rebalance_signal,
                signal_reason=signal_reason
            )
        
        if self.verbose:
            print(f"[FactorDecayMonitor] 衰减分析完成")
            for name, analysis in results.items():
                hl_str = f"{analysis.half_life_days:.1f}天" if analysis.half_life_days else "N/A"
                print(f"  {name}: 衰减率={analysis.decay_rate:.1%}, 半衰期={hl_str}")
        
        return results
    
    def check_rebalancing_signals(
        self,
        decay_analyses: Dict[str, DecayAnalysis]
    ) -> List[RebalanceSignal]:
        """
        检查再平衡信号
        
        Args:
            decay_analyses: 衰减分析结果
        
        Returns:
            再平衡信号列表
        """
        signals = []
        
        for factor_name, analysis in decay_analyses.items():
            if not analysis.rebalance_signal:
                continue
            
            # 确定紧急程度
            if analysis.decay_rate > 0.7:
                urgency = "high"
            elif analysis.decay_rate > 0.5:
                urgency = "medium"
            else:
                urgency = "low"
            
            # 生成建议
            ref = self.FACTOR_REFERENCE.get(factor_name.lower(), {})
            rebalance_months = ref.get("rebalance_months", 3)
            
            recommendation = (
                f"建议在未来{rebalance_months}个月内进行再平衡，"
                f"以恢复{factor_name}因子暴露"
            )
            
            signals.append(RebalanceSignal(
                timestamp=datetime.now(),
                signal_type="exposure_decay",
                factor_name=factor_name,
                current_exposure=analysis.current_exposure,
                threshold=self.exposure_threshold,
                urgency=urgency,
                recommendation=recommendation
            ))
        
        if self.verbose and signals:
            print(f"[FactorDecayMonitor] 生成 {len(signals)} 个再平衡信号")
        
        return signals
    
    def generate_monitoring_report(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        strategy_name: str = "策略"
    ) -> str:
        """
        生成因子监控报告
        
        Args:
            portfolio_returns: 组合收益率序列
            factor_returns: 因子收益率DataFrame
            strategy_name: 策略名称
        
        Returns:
            Markdown格式的监控报告
        """
        # 计算当前暴露
        current_exposures = self.calculate_factor_exposure(portfolio_returns, factor_returns)
        
        # 分析衰减
        decay_analyses = self.analyze_decay(portfolio_returns, factor_returns)
        
        # 检查信号
        signals = self.check_rebalancing_signals(decay_analyses)
        
        # 生成报告
        report = f"""# {strategy_name} 因子暴露监控报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. 当前因子暴露

| 因子 | Beta | t统计量 | 显著性 |
|:---|---:|---:|:---|
"""
        
        for name, exp in current_exposures.items():
            sig = "✓" if exp.is_significant else ""
            report += f"| {name} | {exp.beta:.4f} | {exp.t_stat:.2f} | {sig} |\n"
        
        report += f"""
**模型R²**: {list(current_exposures.values())[0].r_squared:.4f if current_exposures else 'N/A'}

---

## 2. 因子衰减分析

| 因子 | 初始暴露 | 当前暴露 | 衰减率 | 估算半衰期 |
|:---|---:|---:|---:|---:|
"""
        
        for name, analysis in decay_analyses.items():
            hl_str = f"{analysis.half_life_days:.0f}天" if analysis.half_life_days else "N/A"
            report += f"| {name} | {analysis.initial_exposure:.4f} | {analysis.current_exposure:.4f} | {analysis.decay_rate:.1%} | {hl_str} |\n"
        
        report += """
---

## 3. 再平衡信号

"""
        
        if signals:
            for signal in signals:
                urgency_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(signal.urgency, "")
                report += f"""### {urgency_emoji} {signal.factor_name}

- **信号类型**: {signal.signal_type}
- **当前暴露**: {signal.current_exposure:.4f}
- **紧急程度**: {signal.urgency}
- **建议**: {signal.recommendation}

"""
        else:
            report += "✅ 当前无需再平衡\n"
        
        report += """
---

## 4. 参考信息

| 因子 | 参考半衰期 | 建议再平衡周期 |
|:---|---:|---:|
"""
        
        for factor, ref in self.FACTOR_REFERENCE.items():
            hl = f"{ref['half_life_months']}个月" if ref['half_life_months'] else "N/A"
            report += f"| {factor} | {hl} | {ref['rebalance_months']}个月 |\n"
        
        report += """
*参考来源: Flint & Vermaak (2023). "Factor Information Decay: A Global Study." Journal of Portfolio Management.*
"""
        
        return report


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("FactorDecayMonitor 测试")
    print("=" * 60)
    
    # 生成模拟数据
    np.random.seed(42)
    n_days = 252
    dates = pd.date_range('2024-01-01', periods=n_days, freq='B')
    
    # 模拟因子收益率
    factor_returns = pd.DataFrame({
        'market': np.random.normal(0.0005, 0.01, n_days),
        'value': np.random.normal(0.0001, 0.005, n_days),
        'momentum': np.random.normal(0.0002, 0.008, n_days),
        'quality': np.random.normal(0.0001, 0.004, n_days),
    }, index=dates)
    
    # 模拟组合收益率（与因子有一定相关性，但暴露随时间衰减）
    market_beta = 1.0 - np.linspace(0, 0.3, n_days)  # 市场暴露从1.0衰减到0.7
    value_beta = 0.5 - np.linspace(0, 0.4, n_days)   # 价值暴露从0.5衰减到0.1
    
    portfolio_returns = (
        market_beta * factor_returns['market'] +
        value_beta * factor_returns['value'] +
        0.3 * factor_returns['momentum'] +
        0.2 * factor_returns['quality'] +
        np.random.normal(0.0002, 0.005, n_days)  # 特异性收益
    )
    portfolio_returns = pd.Series(portfolio_returns, index=dates)
    
    # 测试监控器
    monitor = FactorDecayMonitor(verbose=True)
    
    # 计算当前暴露
    print("\n--- 测试因子暴露计算 ---")
    exposures = monitor.calculate_factor_exposure(portfolio_returns, factor_returns)
    
    # 分析衰减
    print("\n--- 测试衰减分析 ---")
    decay_analyses = monitor.analyze_decay(portfolio_returns, factor_returns)
    
    # 检查信号
    print("\n--- 测试再平衡信号 ---")
    signals = monitor.check_rebalancing_signals(decay_analyses)
    
    print(f"\n生成的再平衡信号数量: {len(signals)}")
    for signal in signals:
        print(f"  - {signal.factor_name}: {signal.urgency} urgency")
    
    print("\n测试完成!")
