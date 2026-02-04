#!/usr/bin/env python3
"""
V3.3 因子Tear Sheet分析报告
借鉴alphalens的设计，实现标准化、可视化的因子分析报告

核心功能:
1. IC分析（时序图、分布图、统计检验）
2. 分组收益分析（分组收益图、累计收益图）
3. 换手率分析
4. 因子衰减分析
5. 一键生成完整Tear Sheet
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class TearSheetConfig:
    """Tear Sheet配置"""
    quantiles: int = 5  # 分组数量
    periods: List[int] = None  # 持有期列表
    benchmark_col: str = None  # 基准列名
    figsize: Tuple[int, int] = (14, 10)
    
    def __post_init__(self):
        if self.periods is None:
            self.periods = [1, 5, 10, 20]


class FactorDataProcessor:
    """因子数据预处理器"""
    
    @staticmethod
    def prepare_factor_data(
        factor: pd.DataFrame,
        prices: pd.DataFrame,
        periods: List[int] = [1, 5, 10, 20]
    ) -> pd.DataFrame:
        """
        准备因子分析所需的数据格式
        
        Args:
            factor: 因子值，index为日期，columns为股票代码
            prices: 价格数据，index为日期，columns为股票代码
            periods: 持有期列表
            
        Returns:
            处理后的因子数据，包含因子值和各期前向收益
        """
        # 计算各期前向收益
        forward_returns = {}
        for period in periods:
            fwd_ret = prices.pct_change(period).shift(-period)
            forward_returns[f'forward_{period}d'] = fwd_ret
        
        # 将因子和收益合并为长格式
        records = []
        for date in factor.index:
            if date not in prices.index:
                continue
            for stock in factor.columns:
                if stock not in prices.columns:
                    continue
                record = {
                    'date': date,
                    'asset': stock,
                    'factor': factor.loc[date, stock]
                }
                for period in periods:
                    col = f'forward_{period}d'
                    if date in forward_returns[col].index:
                        record[col] = forward_returns[col].loc[date, stock]
                records.append(record)
        
        df = pd.DataFrame(records)
        df = df.dropna(subset=['factor'])
        df = df.set_index(['date', 'asset'])
        
        return df
    
    @staticmethod
    def add_quantile_labels(
        factor_data: pd.DataFrame,
        quantiles: int = 5
    ) -> pd.DataFrame:
        """添加分位数标签"""
        factor_data = factor_data.copy()
        
        def quantile_label(x):
            try:
                return pd.qcut(x, quantiles, labels=False, duplicates='drop') + 1
            except:
                return pd.Series([np.nan] * len(x), index=x.index)
        
        # 按日期分组计算分位数
        factor_data['quantile'] = factor_data.groupby(level='date')['factor'].transform(quantile_label)
        
        return factor_data


class ICAnalyzer:
    """IC分析器"""
    
    @staticmethod
    def calculate_ic_series(
        factor_data: pd.DataFrame,
        period: int = 1
    ) -> pd.Series:
        """计算IC时间序列"""
        col = f'forward_{period}d'
        if col not in factor_data.columns:
            return pd.Series()
        
        def calc_ic(group):
            factor_vals = group['factor']
            return_vals = group[col]
            valid = ~(factor_vals.isna() | return_vals.isna())
            if valid.sum() < 10:
                return np.nan
            return stats.spearmanr(factor_vals[valid], return_vals[valid])[0]
        
        ic_series = factor_data.groupby(level='date').apply(calc_ic)
        return ic_series.dropna()
    
    @staticmethod
    def calculate_ic_statistics(ic_series: pd.Series) -> Dict[str, float]:
        """计算IC统计指标"""
        if len(ic_series) == 0:
            return {}
        
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        ir = ic_mean / ic_std if ic_std > 0 else 0
        
        # t检验
        t_stat, p_value = stats.ttest_1samp(ic_series, 0)
        
        # IC > 0 的比例
        ic_positive_ratio = (ic_series > 0).mean()
        
        return {
            'IC_Mean': ic_mean,
            'IC_Std': ic_std,
            'IR': ir,
            'IC_Positive_Ratio': ic_positive_ratio,
            't_stat': t_stat,
            'p_value': p_value,
            'IC_Skew': ic_series.skew(),
            'IC_Kurtosis': ic_series.kurtosis()
        }


class QuantileAnalyzer:
    """分组分析器"""
    
    @staticmethod
    def calculate_quantile_returns(
        factor_data: pd.DataFrame,
        period: int = 1
    ) -> pd.DataFrame:
        """计算各分组的平均收益"""
        col = f'forward_{period}d'
        if col not in factor_data.columns or 'quantile' not in factor_data.columns:
            return pd.DataFrame()
        
        # 按日期和分组计算平均收益
        quantile_returns = factor_data.groupby(['date', 'quantile'])[col].mean().unstack()
        
        return quantile_returns
    
    @staticmethod
    def calculate_cumulative_returns(quantile_returns: pd.DataFrame) -> pd.DataFrame:
        """计算累计收益"""
        return (1 + quantile_returns).cumprod() - 1
    
    @staticmethod
    def calculate_spread_returns(quantile_returns: pd.DataFrame) -> pd.Series:
        """计算多空组合收益（最高分位 - 最低分位）"""
        if quantile_returns.empty:
            return pd.Series()
        
        max_q = quantile_returns.columns.max()
        min_q = quantile_returns.columns.min()
        
        return quantile_returns[max_q] - quantile_returns[min_q]


class TurnoverAnalyzer:
    """换手率分析器"""
    
    @staticmethod
    def calculate_turnover(
        factor_data: pd.DataFrame,
        quantile: int = None
    ) -> pd.Series:
        """计算换手率"""
        if 'quantile' not in factor_data.columns:
            return pd.Series()
        
        dates = factor_data.index.get_level_values('date').unique().sort_values()
        turnover_list = []
        
        for i in range(1, len(dates)):
            prev_date = dates[i-1]
            curr_date = dates[i]
            
            try:
                prev_data = factor_data.xs(prev_date, level='date')
                curr_data = factor_data.xs(curr_date, level='date')
            except:
                continue
            
            if quantile is not None:
                prev_assets = set(prev_data[prev_data['quantile'] == quantile].index)
                curr_assets = set(curr_data[curr_data['quantile'] == quantile].index)
            else:
                prev_assets = set(prev_data.index)
                curr_assets = set(curr_data.index)
            
            if len(prev_assets) == 0 or len(curr_assets) == 0:
                continue
            
            # 换手率 = 1 - (交集 / 并集)
            intersection = len(prev_assets & curr_assets)
            union = len(prev_assets | curr_assets)
            turnover = 1 - (intersection / union) if union > 0 else 0
            
            turnover_list.append({'date': curr_date, 'turnover': turnover})
        
        if not turnover_list:
            return pd.Series()
        
        return pd.DataFrame(turnover_list).set_index('date')['turnover']


class DecayAnalyzer:
    """因子衰减分析器"""
    
    @staticmethod
    def calculate_ic_decay(
        factor_data: pd.DataFrame,
        max_period: int = 20
    ) -> pd.Series:
        """计算IC随持有期的衰减"""
        ic_by_period = {}
        
        for period in range(1, max_period + 1):
            col = f'forward_{period}d'
            if col in factor_data.columns:
                ic_series = ICAnalyzer.calculate_ic_series(factor_data, period)
                if len(ic_series) > 0:
                    ic_by_period[period] = ic_series.mean()
        
        return pd.Series(ic_by_period)


class FactorTearSheet:
    """
    因子Tear Sheet生成器
    
    借鉴alphalens的设计，提供标准化的因子分析报告
    """
    
    def __init__(self, config: TearSheetConfig = None):
        self.config = config or TearSheetConfig()
        self.ic_analyzer = ICAnalyzer()
        self.quantile_analyzer = QuantileAnalyzer()
        self.turnover_analyzer = TurnoverAnalyzer()
        self.decay_analyzer = DecayAnalyzer()
        
    def analyze(
        self,
        factor: pd.DataFrame,
        prices: pd.DataFrame,
        factor_name: str = "Factor"
    ) -> Dict[str, Any]:
        """
        执行完整的因子分析
        
        Args:
            factor: 因子值DataFrame，index为日期，columns为股票
            prices: 价格DataFrame，index为日期，columns为股票
            factor_name: 因子名称
            
        Returns:
            包含所有分析结果的字典
        """
        results = {'factor_name': factor_name}
        
        # 1. 准备数据
        factor_data = FactorDataProcessor.prepare_factor_data(
            factor, prices, self.config.periods
        )
        factor_data = FactorDataProcessor.add_quantile_labels(
            factor_data, self.config.quantiles
        )
        results['factor_data'] = factor_data
        
        # 2. IC分析
        ic_results = {}
        for period in self.config.periods:
            ic_series = self.ic_analyzer.calculate_ic_series(factor_data, period)
            ic_stats = self.ic_analyzer.calculate_ic_statistics(ic_series)
            ic_results[period] = {
                'ic_series': ic_series,
                'statistics': ic_stats
            }
        results['ic_analysis'] = ic_results
        
        # 3. 分组分析
        quantile_results = {}
        for period in self.config.periods:
            q_returns = self.quantile_analyzer.calculate_quantile_returns(factor_data, period)
            cum_returns = self.quantile_analyzer.calculate_cumulative_returns(q_returns)
            spread = self.quantile_analyzer.calculate_spread_returns(q_returns)
            quantile_results[period] = {
                'returns': q_returns,
                'cumulative': cum_returns,
                'spread': spread
            }
        results['quantile_analysis'] = quantile_results
        
        # 4. 换手率分析
        turnover_results = {}
        for q in range(1, self.config.quantiles + 1):
            turnover = self.turnover_analyzer.calculate_turnover(factor_data, q)
            turnover_results[q] = turnover
        results['turnover_analysis'] = turnover_results
        
        # 5. 衰减分析
        ic_decay = self.decay_analyzer.calculate_ic_decay(factor_data, max(self.config.periods))
        results['decay_analysis'] = ic_decay
        
        return results
    
    def generate_summary_table(self, results: Dict[str, Any]) -> pd.DataFrame:
        """生成汇总统计表"""
        summary_data = []
        
        for period, ic_data in results.get('ic_analysis', {}).items():
            stats = ic_data.get('statistics', {})
            quantile_data = results.get('quantile_analysis', {}).get(period, {})
            spread = quantile_data.get('spread', pd.Series())
            
            row = {
                'Period': f'{period}D',
                'IC Mean': stats.get('IC_Mean', np.nan),
                'IC Std': stats.get('IC_Std', np.nan),
                'IR': stats.get('IR', np.nan),
                'IC>0 Ratio': stats.get('IC_Positive_Ratio', np.nan),
                't-stat': stats.get('t_stat', np.nan),
                'p-value': stats.get('p_value', np.nan),
                'Spread Mean': spread.mean() if len(spread) > 0 else np.nan,
                'Spread Sharpe': (spread.mean() / spread.std() * np.sqrt(252)) if len(spread) > 0 and spread.std() > 0 else np.nan
            }
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def generate_report(
        self,
        results: Dict[str, Any],
        save_path: str = None
    ) -> str:
        """
        生成Markdown格式的分析报告
        
        Args:
            results: analyze()方法返回的结果
            save_path: 保存路径（可选）
            
        Returns:
            Markdown格式的报告字符串
        """
        factor_name = results.get('factor_name', 'Factor')
        
        report = f"""# 因子分析报告 (Tear Sheet)

## 因子: {factor_name}

---

## 1. 汇总统计

"""
        # 汇总表
        summary_df = self.generate_summary_table(results)
        if not summary_df.empty:
            report += summary_df.to_markdown(index=False, floatfmt='.4f')
            report += "\n\n"
        
        # IC分析详情
        report += """
---

## 2. IC分析

### 2.1 IC统计

"""
        for period, ic_data in results.get('ic_analysis', {}).items():
            stats = ic_data.get('statistics', {})
            ic_series = ic_data.get('ic_series', pd.Series())
            
            report += f"""
#### {period}日IC

| 指标 | 值 |
|:---|---:|
| IC均值 | {stats.get('IC_Mean', np.nan):.4f} |
| IC标准差 | {stats.get('IC_Std', np.nan):.4f} |
| IR (IC/Std) | {stats.get('IR', np.nan):.4f} |
| IC>0比例 | {stats.get('IC_Positive_Ratio', np.nan):.2%} |
| t统计量 | {stats.get('t_stat', np.nan):.4f} |
| p值 | {stats.get('p_value', np.nan):.4f} |

"""
        
        # 分组分析
        report += """
---

## 3. 分组收益分析

"""
        for period, q_data in results.get('quantile_analysis', {}).items():
            q_returns = q_data.get('returns', pd.DataFrame())
            if q_returns.empty:
                continue
            
            # 计算各组平均收益
            mean_returns = q_returns.mean()
            report += f"""
### {period}日持有期

| 分组 | 平均收益 |
|:---|---:|
"""
            for q in sorted(mean_returns.index):
                report += f"| Q{int(q)} | {mean_returns[q]:.4%} |\n"
            
            spread = q_data.get('spread', pd.Series())
            if len(spread) > 0:
                report += f"\n**多空收益 (Q{int(mean_returns.index.max())}-Q{int(mean_returns.index.min())})**: {spread.mean():.4%}\n\n"
        
        # 换手率分析
        report += """
---

## 4. 换手率分析

| 分组 | 平均换手率 |
|:---|---:|
"""
        for q, turnover in results.get('turnover_analysis', {}).items():
            if len(turnover) > 0:
                report += f"| Q{q} | {turnover.mean():.2%} |\n"
        
        # 衰减分析
        report += """
---

## 5. 因子衰减分析

| 持有期 | IC均值 |
|:---|---:|
"""
        ic_decay = results.get('decay_analysis', pd.Series())
        for period, ic in ic_decay.items():
            report += f"| {period}D | {ic:.4f} |\n"
        
        # 结论
        report += """
---

## 6. 分析结论

"""
        # 自动生成结论
        summary_df = self.generate_summary_table(results)
        if not summary_df.empty:
            best_period = summary_df.loc[summary_df['IR'].idxmax()]
            
            ic_mean = best_period['IC Mean']
            ir = best_period['IR']
            p_value = best_period['p-value']
            
            # 因子评级
            if ir > 0.5 and p_value < 0.05:
                rating = "**优秀**"
                conclusion = "该因子具有较强的预测能力，IC均值显著不为零，IR较高，建议纳入因子库。"
            elif ir > 0.3 and p_value < 0.1:
                rating = "**良好**"
                conclusion = "该因子具有一定的预测能力，可以考虑与其他因子组合使用。"
            elif ir > 0.1:
                rating = "**一般**"
                conclusion = "该因子预测能力较弱，建议进一步优化或与其他因子组合。"
            else:
                rating = "**较差**"
                conclusion = "该因子预测能力不足，不建议单独使用。"
            
            report += f"""
### 因子评级: {rating}

{conclusion}

**最佳持有期**: {best_period['Period']}
**最佳IR**: {ir:.4f}
"""
        
        report += """
---

*报告由 quant-investor V3.3 自动生成*
"""
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report
    
    def plot_tear_sheet(
        self,
        results: Dict[str, Any],
        save_path: str = None
    ) -> None:
        """
        生成可视化Tear Sheet图表
        
        Args:
            results: analyze()方法返回的结果
            save_path: 图片保存路径（可选）
        """
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not available, skipping visualization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=self.config.figsize)
        fig.suptitle(f"Factor Tear Sheet: {results.get('factor_name', 'Factor')}", fontsize=14)
        
        # 1. IC时序图
        ax1 = axes[0, 0]
        period = self.config.periods[0]
        ic_series = results.get('ic_analysis', {}).get(period, {}).get('ic_series', pd.Series())
        if len(ic_series) > 0:
            ax1.bar(ic_series.index, ic_series.values, alpha=0.7, width=1)
            ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax1.axhline(y=ic_series.mean(), color='red', linestyle='--', label=f'Mean: {ic_series.mean():.4f}')
            ax1.set_title(f'IC Time Series ({period}D)')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('IC')
            ax1.legend()
        
        # 2. 分组累计收益图
        ax2 = axes[0, 1]
        cum_returns = results.get('quantile_analysis', {}).get(period, {}).get('cumulative', pd.DataFrame())
        if not cum_returns.empty:
            for col in cum_returns.columns:
                ax2.plot(cum_returns.index, cum_returns[col], label=f'Q{int(col)}')
            ax2.set_title(f'Cumulative Returns by Quantile ({period}D)')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Cumulative Return')
            ax2.legend()
        
        # 3. IC衰减图
        ax3 = axes[1, 0]
        ic_decay = results.get('decay_analysis', pd.Series())
        if len(ic_decay) > 0:
            ax3.bar(ic_decay.index, ic_decay.values, alpha=0.7)
            ax3.set_title('IC Decay by Holding Period')
            ax3.set_xlabel('Holding Period (Days)')
            ax3.set_ylabel('IC Mean')
        
        # 4. 分组平均收益柱状图
        ax4 = axes[1, 1]
        q_returns = results.get('quantile_analysis', {}).get(period, {}).get('returns', pd.DataFrame())
        if not q_returns.empty:
            mean_returns = q_returns.mean()
            colors = ['red' if x < 0 else 'green' for x in mean_returns.values]
            ax4.bar([f'Q{int(q)}' for q in mean_returns.index], mean_returns.values, color=colors, alpha=0.7)
            ax4.set_title(f'Mean Returns by Quantile ({period}D)')
            ax4.set_xlabel('Quantile')
            ax4.set_ylabel('Mean Return')
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Tear Sheet saved to: {save_path}")
        
        plt.close()


def create_tear_sheet(
    factor: pd.DataFrame,
    prices: pd.DataFrame,
    factor_name: str = "Factor",
    quantiles: int = 5,
    periods: List[int] = [1, 5, 10, 20],
    report_path: str = None,
    plot_path: str = None
) -> Dict[str, Any]:
    """
    一键生成因子Tear Sheet的便捷函数
    
    Args:
        factor: 因子值DataFrame
        prices: 价格DataFrame
        factor_name: 因子名称
        quantiles: 分组数量
        periods: 持有期列表
        report_path: 报告保存路径
        plot_path: 图表保存路径
        
    Returns:
        分析结果字典
    """
    config = TearSheetConfig(quantiles=quantiles, periods=periods)
    tear_sheet = FactorTearSheet(config)
    
    results = tear_sheet.analyze(factor, prices, factor_name)
    
    if report_path:
        tear_sheet.generate_report(results, report_path)
    
    if plot_path:
        tear_sheet.plot_tear_sheet(results, plot_path)
    
    return results


# 测试代码
if __name__ == "__main__":
    print("=== V3.3 Factor Tear Sheet Test ===\n")
    
    # 生成测试数据
    np.random.seed(42)
    n_days = 252
    n_stocks = 50
    
    dates = pd.date_range('2024-01-01', periods=n_days, freq='B')
    stocks = [f'STOCK_{i}' for i in range(n_stocks)]
    
    # 生成价格数据
    prices = pd.DataFrame(
        np.random.randn(n_days, n_stocks).cumsum(axis=0) + 100,
        index=dates,
        columns=stocks
    )
    
    # 生成因子数据（与未来收益有一定相关性）
    future_returns = prices.pct_change(5).shift(-5)
    noise = pd.DataFrame(
        np.random.randn(n_days, n_stocks) * 0.5,
        index=dates,
        columns=stocks
    )
    factor = future_returns * 0.3 + noise  # 因子与未来收益有30%的相关性
    
    # 运行分析
    results = create_tear_sheet(
        factor=factor,
        prices=prices,
        factor_name="Test_Factor",
        quantiles=5,
        periods=[1, 5, 10, 20]
    )
    
    # 生成报告
    tear_sheet = FactorTearSheet()
    report = tear_sheet.generate_report(results)
    print(report)
    
    print("\n=== Test Completed ===")
