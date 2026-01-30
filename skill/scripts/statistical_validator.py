#!/usr/bin/env python3
"""
Statistical Validator - 统计显著性验证模块

本脚本提供了一套完整的统计检验工具，用于验证回测结果的可靠性。

核心功能:
1. Bootstrap检验 - 计算性能指标的置信区间
2. 多重检验校正 - 控制假阳性率
3. 蒙特卡洛模拟 - 评估策略的运气成分

参考文献:
- Efron & Tibshirani (1993), An Introduction to the Bootstrap
- Benjamini & Hochberg (1995), Controlling the False Discovery Rate
- Bailey & López de Prado (2014), The Deflated Sharpe Ratio
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class StatisticalValidator:
    """统计验证器"""
    
    def __init__(self, n_bootstrap=1000, n_monte_carlo=1000, confidence_level=0.95):
        """
        初始化
        
        Parameters:
        -----------
        n_bootstrap : int
            Bootstrap重采样次数
        n_monte_carlo : int
            蒙特卡洛模拟次数
        confidence_level : float
            置信水平 (default: 0.95)
        """
        self.n_bootstrap = n_bootstrap
        self.n_monte_carlo = n_monte_carlo
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def stationary_bootstrap(self, returns: np.ndarray, block_size: int = 20) -> np.ndarray:
        """
        平稳Bootstrap重采样
        
        与简单Bootstrap不同，平稳Bootstrap保留了时间序列的自相关结构
        
        Parameters:
        -----------
        returns : np.ndarray
            收益率序列
        block_size : int
            平均块大小
        
        Returns:
        --------
        np.ndarray
            重采样后的收益率序列
        """
        n = len(returns)
        resampled = []
        
        while len(resampled) < n:
            # 随机选择起始点
            start_idx = np.random.randint(0, n)
            # 随机决定块的长度（几何分布）
            block_length = np.random.geometric(1/block_size)
            # 提取块
            end_idx = min(start_idx + block_length, n)
            resampled.extend(returns[start_idx:end_idx])
        
        return np.array(resampled[:n])
    
    def bootstrap_sharpe_ratio(self, returns: np.ndarray) -> Dict:
        """
        Bootstrap夏普比率检验
        
        Returns:
        --------
        dict
            包含夏普比率的点估计、置信区间和p值
        """
        # 原始夏普比率
        sharpe_original = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        # Bootstrap重采样
        sharpe_bootstrap = []
        for _ in range(self.n_bootstrap):
            resampled_returns = self.stationary_bootstrap(returns)
            sharpe = np.mean(resampled_returns) / np.std(resampled_returns) * np.sqrt(252)
            sharpe_bootstrap.append(sharpe)
        
        sharpe_bootstrap = np.array(sharpe_bootstrap)
        
        # 计算置信区间
        ci_lower = np.percentile(sharpe_bootstrap, (self.alpha/2) * 100)
        ci_upper = np.percentile(sharpe_bootstrap, (1 - self.alpha/2) * 100)
        
        # 计算p值: 夏普比率小于等于0的概率
        p_value = np.mean(sharpe_bootstrap <= 0)
        
        return {
            'sharpe_ratio': sharpe_original,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'bootstrap_distribution': sharpe_bootstrap
        }
    
    def bootstrap_max_drawdown(self, returns: np.ndarray) -> Dict:
        """Bootstrap最大回撤检验"""
        def calculate_max_drawdown(rets):
            cumulative = np.cumprod(1 + rets)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return np.min(drawdown)
        
        # 原始最大回撤
        mdd_original = calculate_max_drawdown(returns)
        
        # Bootstrap重采样
        mdd_bootstrap = []
        for _ in range(self.n_bootstrap):
            resampled_returns = self.stationary_bootstrap(returns)
            mdd = calculate_max_drawdown(resampled_returns)
            mdd_bootstrap.append(mdd)
        
        mdd_bootstrap = np.array(mdd_bootstrap)
        
        # 计算置信区间
        ci_lower = np.percentile(mdd_bootstrap, (self.alpha/2) * 100)
        ci_upper = np.percentile(mdd_bootstrap, (1 - self.alpha/2) * 100)
        
        return {
            'max_drawdown': mdd_original,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'bootstrap_distribution': mdd_bootstrap
        }
    
    def monte_carlo_simulation(self, trades: List[float]) -> Dict:
        """
        蒙特卡洛模拟
        
        通过随机打乱交易顺序，评估策略的运气成分
        
        Parameters:
        -----------
        trades : List[float]
            每笔交易的收益率
        
        Returns:
        --------
        dict
            包含模拟的收益分布、最大回撤分布等
        """
        trades = np.array(trades)
        
        # 原始权益曲线
        original_equity = np.cumprod(1 + trades)
        original_return = original_equity[-1] - 1
        
        # 蒙特卡洛模拟
        simulated_returns = []
        simulated_max_drawdowns = []
        
        for _ in range(self.n_monte_carlo):
            # 随机打乱交易顺序
            shuffled_trades = np.random.permutation(trades)
            equity_curve = np.cumprod(1 + shuffled_trades)
            
            # 计算总收益
            total_return = equity_curve[-1] - 1
            simulated_returns.append(total_return)
            
            # 计算最大回撤
            running_max = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - running_max) / running_max
            max_dd = np.min(drawdown)
            simulated_max_drawdowns.append(max_dd)
        
        simulated_returns = np.array(simulated_returns)
        simulated_max_drawdowns = np.array(simulated_max_drawdowns)
        
        # 计算原始结果在模拟分布中的百分位
        return_percentile = stats.percentileofscore(simulated_returns, original_return)
        
        return {
            'original_return': original_return,
            'simulated_returns': simulated_returns,
            'return_percentile': return_percentile,
            'simulated_max_drawdowns': simulated_max_drawdowns,
            'mdd_95th_percentile': np.percentile(simulated_max_drawdowns, 95)
        }
    
    def multiple_testing_correction(self, p_values: List[float], method='fdr_bh') -> np.ndarray:
        """
        多重检验校正
        
        Parameters:
        -----------
        p_values : List[float]
            原始p值列表
        method : str
            校正方法: 'bonferroni' 或 'fdr_bh' (Benjamini-Hochberg)
        
        Returns:
        --------
        np.ndarray
            校正后的p值（q值）
        """
        p_values = np.array(p_values)
        n = len(p_values)
        
        if method == 'bonferroni':
            # Bonferroni校正
            return np.minimum(p_values * n, 1.0)
        
        elif method == 'fdr_bh':
            # Benjamini-Hochberg FDR校正
            sorted_indices = np.argsort(p_values)
            sorted_p_values = p_values[sorted_indices]
            
            # 计算q值
            q_values = np.zeros(n)
            for i in range(n):
                q_values[sorted_indices[i]] = min(sorted_p_values[i] * n / (i + 1), 1.0)
            
            # 确保单调性
            for i in range(n-2, -1, -1):
                if q_values[sorted_indices[i]] > q_values[sorted_indices[i+1]]:
                    q_values[sorted_indices[i]] = q_values[sorted_indices[i+1]]
            
            return q_values
        
        else:
            raise ValueError(f"未知的校正方法: {method}")
    
    def plot_bootstrap_results(self, bootstrap_results: Dict, title: str, output_path: str):
        """绘制Bootstrap结果"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. Bootstrap分布直方图
        distribution = bootstrap_results['bootstrap_distribution']
        axes[0].hist(distribution, bins=50, edgecolor='black', alpha=0.7, density=True)
        axes[0].axvline(bootstrap_results['sharpe_ratio'] if 'sharpe_ratio' in bootstrap_results else bootstrap_results['max_drawdown'], 
                       color='r', linestyle='--', linewidth=2, label='观测值')
        axes[0].axvline(bootstrap_results['ci_lower'], color='g', linestyle='--', linewidth=1.5, label='95% CI')
        axes[0].axvline(bootstrap_results['ci_upper'], color='g', linestyle='--', linewidth=1.5)
        axes[0].set_title(f'{title} Bootstrap分布', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('值')
        axes[0].set_ylabel('密度')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. QQ图
        stats.probplot(distribution, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q图 (正态性检验)', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_validation_report(self, results: Dict, output_path: str):
        """生成统计验证报告"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("统计显著性验证报告\n")
            f.write("=" * 80 + "\n\n")
            
            # Bootstrap夏普比率
            if 'sharpe_bootstrap' in results:
                sr = results['sharpe_bootstrap']
                f.write("【Bootstrap夏普比率检验】\n")
                f.write(f"  夏普比率:            {sr['sharpe_ratio']:.4f}\n")
                f.write(f"  95% 置信区间:        [{sr['ci_lower']:.4f}, {sr['ci_upper']:.4f}]\n")
                f.write(f"  P值 (SR <= 0):       {sr['p_value']:.6f}\n")
                if sr['ci_lower'] > 0:
                    f.write("  结论: ✓ 夏普比率显著大于0\n\n")
                else:
                    f.write("  结论: ✗ 夏普比率不显著\n\n")
            
            # Bootstrap最大回撤
            if 'mdd_bootstrap' in results:
                mdd = results['mdd_bootstrap']
                f.write("【Bootstrap最大回撤检验】\n")
                f.write(f"  最大回撤:            {mdd['max_drawdown']:.2%}\n")
                f.write(f"  95% 置信区间:        [{mdd['ci_lower']:.2%}, {mdd['ci_upper']:.2%}]\n\n")
            
            # 蒙特卡洛模拟
            if 'monte_carlo' in results:
                mc = results['monte_carlo']
                f.write("【蒙特卡洛模拟】\n")
                f.write(f"  原始总收益:          {mc['original_return']:.2%}\n")
                f.write(f"  收益百分位:          {mc['return_percentile']:.1f}%\n")
                f.write(f"  95%最大回撤:         {mc['mdd_95th_percentile']:.2%}\n")
                if mc['return_percentile'] > 90:
                    f.write("  结论: ⚠ 策略表现可能受运气影响较大\n\n")
                else:
                    f.write("  结论: ✓ 策略表现相对稳健\n\n")
            
            f.write("=" * 80 + "\n")
        
        print(f"统计验证报告已生成: {output_path}")

# 使用示例
if __name__ == '__main__':
    # 生成模拟数据进行测试
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, 500)  # 500天的日收益率
    
    validator = StatisticalValidator(n_bootstrap=1000, n_monte_carlo=1000)
    
    # Bootstrap夏普比率
    print("执行Bootstrap夏普比率检验...")
    sharpe_results = validator.bootstrap_sharpe_ratio(returns)
    print(f"夏普比率: {sharpe_results['sharpe_ratio']:.4f}")
    print(f"95% CI: [{sharpe_results['ci_lower']:.4f}, {sharpe_results['ci_upper']:.4f}]")
    print(f"P值: {sharpe_results['p_value']:.6f}\n")
    
    # Bootstrap最大回撤
    print("执行Bootstrap最大回撤检验...")
    mdd_results = validator.bootstrap_max_drawdown(returns)
    print(f"最大回撤: {mdd_results['max_drawdown']:.2%}")
    print(f"95% CI: [{mdd_results['ci_lower']:.2%}, {mdd_results['ci_upper']:.2%}]\n")
    
    # 蒙特卡洛模拟
    print("执行蒙特卡洛模拟...")
    trades = np.random.normal(0.001, 0.03, 100)  # 100笔交易
    mc_results = validator.monte_carlo_simulation(trades)
    print(f"原始总收益: {mc_results['original_return']:.2%}")
    print(f"收益百分位: {mc_results['return_percentile']:.1f}%")
    
    print("\n测试完成！")
