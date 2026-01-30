#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
投资组合优化脚本
基于现代投资组合理论，考虑相关性进行分散化配置
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy.optimize import minimize

# 候选股票池（扩展到不同行业）
CANDIDATE_STOCKS = {
    # AI/科技板块
    'NVDA': 'NVIDIA',
    'GOOGL': 'Alphabet',
    'MSFT': 'Microsoft',
    'META': 'Meta',
    
    # 半导体
    'TSM': 'TSMC',
    'AVGO': 'Broadcom',
    'MU': 'Micron',
    
    # 医疗健康
    'LLY': 'Eli Lilly',
    'JNJ': 'Johnson & Johnson',
    'ABBV': 'AbbVie',
    
    # 消费
    'COST': 'Costco',
    'WMT': 'Walmart',
    'HD': 'Home Depot',
    
    # 金融
    'JPM': 'JPMorgan',
    'V': 'Visa',
    'MA': 'Mastercard',
    
    # 能源
    'XOM': 'Exxon Mobil',
    'CVX': 'Chevron',
    
    # 工业
    'BA': 'Boeing',
    'CAT': 'Caterpillar',
    
    # 防御性/价值股
    'BRK-B': 'Berkshire Hathaway',
    'KO': 'Coca-Cola',
    'PG': 'Procter & Gamble'
}

def get_historical_data(tickers, period='1y'):
    """获取历史数据"""
    print(f"正在获取{len(tickers)}只股票的历史数据...")
    data = yf.download(list(tickers), period=period, progress=False, auto_adjust=True)['Close']
    return data

def calculate_returns(data):
    """计算收益率"""
    returns = data.pct_change().dropna()
    return returns

def calculate_correlation_matrix(returns):
    """计算相关性矩阵"""
    corr_matrix = returns.corr()
    return corr_matrix

def plot_correlation_heatmap(corr_matrix, save_path):
    """绘制相关性热力图"""
    plt.figure(figsize=(16, 14))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5)
    plt.title('Stock Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"相关性热力图已保存至: {save_path}")

def calculate_portfolio_metrics(weights, returns):
    """计算投资组合指标"""
    portfolio_return = np.sum(returns.mean() * weights) * 252  # 年化收益
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))  # 年化波动率
    sharpe_ratio = portfolio_return / portfolio_std if portfolio_std > 0 else 0
    return portfolio_return, portfolio_std, sharpe_ratio

def optimize_portfolio_max_sharpe(returns):
    """优化投资组合：最大化夏普比率"""
    num_assets = len(returns.columns)
    
    def neg_sharpe(weights):
        ret, std, sharpe = calculate_portfolio_metrics(weights, returns)
        return -sharpe
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # 权重和为1
    bounds = tuple((0.05, 0.25) for _ in range(num_assets))  # 每只股票5%-25%
    
    initial_guess = num_assets * [1. / num_assets]
    
    result = minimize(neg_sharpe, initial_guess, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    return result.x

def optimize_portfolio_min_correlation(returns, target_return=0.15):
    """优化投资组合：最小化平均相关性，同时满足目标收益"""
    num_assets = len(returns.columns)
    corr_matrix = returns.corr()
    
    def avg_correlation(weights):
        # 计算加权平均相关性
        weighted_corr = 0
        for i in range(num_assets):
            for j in range(i+1, num_assets):
                weighted_corr += weights[i] * weights[j] * corr_matrix.iloc[i, j]
        return weighted_corr
    
    def portfolio_return(weights):
        return np.sum(returns.mean() * weights) * 252
    
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 权重和为1
        {'type': 'ineq', 'fun': lambda x: portfolio_return(x) - target_return}  # 收益率>=目标
    )
    bounds = tuple((0.05, 0.25) for _ in range(num_assets))  # 每只股票5%-25%
    
    initial_guess = num_assets * [1. / num_assets]
    
    result = minimize(avg_correlation, initial_guess, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    return result.x

def select_low_correlation_stocks(corr_matrix, n_stocks=6, max_avg_corr=0.5):
    """选择低相关性的股票组合"""
    stocks = corr_matrix.columns.tolist()
    best_combo = None
    best_avg_corr = 1.0
    
    from itertools import combinations
    
    print(f"正在从{len(stocks)}只股票中选择{n_stocks}只低相关性组合...")
    
    # 遍历所有可能的组合
    for combo in combinations(stocks, n_stocks):
        subset_corr = corr_matrix.loc[list(combo), list(combo)]
        # 计算平均相关性（不包括对角线）
        mask = np.ones(subset_corr.shape, dtype=bool)
        np.fill_diagonal(mask, False)
        avg_corr = subset_corr.values[mask].mean()
        
        if avg_corr < best_avg_corr:
            best_avg_corr = avg_corr
            best_combo = combo
    
    print(f"找到最低平均相关性组合: {best_avg_corr:.3f}")
    return list(best_combo), best_avg_corr

def main():
    # 1. 获取数据
    tickers = list(CANDIDATE_STOCKS.keys())
    data = get_historical_data(tickers, period='1y')
    
    # 移除缺失数据过多的股票
    data = data.dropna(axis=1, thresh=len(data)*0.8)
    print(f"有效股票数量: {len(data.columns)}")
    
    # 2. 计算收益率
    returns = calculate_returns(data)
    
    # 3. 计算相关性矩阵
    corr_matrix = calculate_correlation_matrix(returns)
    
    # 4. 绘制相关性热力图
    plot_correlation_heatmap(corr_matrix, '/home/ubuntu/portfolio_analysis/correlation_heatmap.png')
    
    # 5. 选择低相关性股票组合
    low_corr_stocks, avg_corr = select_low_correlation_stocks(corr_matrix, n_stocks=6, max_avg_corr=0.5)
    
    print(f"\n推荐的低相关性股票组合:")
    for ticker in low_corr_stocks:
        print(f"  - {ticker}: {CANDIDATE_STOCKS.get(ticker, 'Unknown')}")
    
    # 6. 对选定的股票进行组合优化
    selected_returns = returns[low_corr_stocks]
    
    # 方法1: 最大化夏普比率
    weights_sharpe = optimize_portfolio_max_sharpe(selected_returns)
    ret_sharpe, std_sharpe, sharpe_sharpe = calculate_portfolio_metrics(weights_sharpe, selected_returns)
    
    # 方法2: 最小化相关性
    weights_min_corr = optimize_portfolio_min_correlation(selected_returns, target_return=0.12)
    ret_min_corr, std_min_corr, sharpe_min_corr = calculate_portfolio_metrics(weights_min_corr, selected_returns)
    
    # 7. 输出结果
    results = {
        'stocks': low_corr_stocks,
        'avg_correlation': avg_corr,
        'max_sharpe': {
            'weights': weights_sharpe,
            'return': ret_sharpe,
            'volatility': std_sharpe,
            'sharpe': sharpe_sharpe
        },
        'min_correlation': {
            'weights': weights_min_corr,
            'return': ret_min_corr,
            'volatility': std_min_corr,
            'sharpe': sharpe_min_corr
        }
    }
    
    # 保存结果
    with open('/home/ubuntu/portfolio_analysis/optimization_results.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("投资组合优化结果\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"选定的低相关性股票组合 (平均相关性: {avg_corr:.3f}):\n")
        for ticker in low_corr_stocks:
            f.write(f"  - {ticker}: {CANDIDATE_STOCKS.get(ticker, 'Unknown')}\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("方案1: 最大化夏普比率\n")
        f.write("-" * 80 + "\n")
        f.write(f"预期年化收益率: {ret_sharpe:.2%}\n")
        f.write(f"预期年化波动率: {std_sharpe:.2%}\n")
        f.write(f"夏普比率: {sharpe_sharpe:.2f}\n\n")
        f.write("权重分配:\n")
        for ticker, weight in zip(low_corr_stocks, weights_sharpe):
            f.write(f"  {ticker}: {weight:.2%}\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("方案2: 最小化相关性 (目标收益12%)\n")
        f.write("-" * 80 + "\n")
        f.write(f"预期年化收益率: {ret_min_corr:.2%}\n")
        f.write(f"预期年化波动率: {std_min_corr:.2%}\n")
        f.write(f"夏普比率: {sharpe_min_corr:.2f}\n\n")
        f.write("权重分配:\n")
        for ticker, weight in zip(low_corr_stocks, weights_min_corr):
            f.write(f"  {ticker}: {weight:.2%}\n")
    
    print("\n优化结果已保存至: /home/ubuntu/portfolio_analysis/optimization_results.txt")
    
    return results

if __name__ == '__main__':
    results = main()
