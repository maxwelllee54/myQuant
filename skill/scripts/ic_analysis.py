#!/usr/bin/env python3
"""
IC Analysis Script - Information Coefficient Analysis
信息系数分析脚本

本脚本用于在策略构建之前，验证因子（信号）的有效性和稳定性。
这是量化策略开发的第一道关卡，只有通过IC检验的因子才应进入回测阶段。

核心指标:
- IC (Information Coefficient): 因子预测排名与实际收益排名的相关系数
- IC_IR (Information Ratio): IC均值 / IC标准差，衡量稳定性
- 滚动IC: 观察IC在不同时间窗口的变化

参考: Grinold & Kahn (1999), Active Portfolio Management
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import configparser
import os
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ICAnalyzer:
    """信息系数分析器"""
    
    def __init__(self, config_path):
        """初始化"""
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        
        # 读取配置
        self.data_dir = self.config.get('Paths', 'data_dir')
        self.output_dir = self.config.get('Paths', 'output_dir')
        self.report_dir = os.path.join(self.output_dir, 'ic_reports')
        os.makedirs(self.report_dir, exist_ok=True)
        
        # IC分析参数
        self.ic_window = self.config.getint('IC', 'rolling_window', fallback=12)  # 滚动窗口（月）
        self.forward_periods = self.config.getint('IC', 'forward_periods', fallback=20)  # 前瞻期（天）
        
    def load_data(self, stock_code):
        """加载股票数据和特征"""
        feature_file = os.path.join(self.data_dir, f'{stock_code}_features.csv')
        if not os.path.exists(feature_file):
            raise FileNotFoundError(f"特征文件不存在: {feature_file}")
        
        df = pd.read_csv(feature_file, parse_dates=['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df
    
    def calculate_forward_returns(self, df):
        """计算前瞻收益率"""
        df['forward_return'] = df['close'].pct_change(self.forward_periods).shift(-self.forward_periods)
        return df
    
    def calculate_ic(self, df, factor_name):
        """计算单个因子的IC"""
        # 按月分组计算IC
        df['year_month'] = df['date'].dt.to_period('M')
        
        ic_list = []
        for period, group in df.groupby('year_month'):
            if len(group) < 10:  # 数据点太少则跳过
                continue
            
            # 去除NaN
            valid_data = group[[factor_name, 'forward_return']].dropna()
            if len(valid_data) < 10:
                continue
            
            # 计算Spearman相关系数（基于排名）
            ic, p_value = stats.spearmanr(valid_data[factor_name], valid_data['forward_return'])
            
            ic_list.append({
                'period': period.to_timestamp(),
                'ic': ic,
                'p_value': p_value,
                'n_samples': len(valid_data)
            })
        
        return pd.DataFrame(ic_list)
    
    def calculate_ic_metrics(self, ic_df):
        """计算IC的汇总指标"""
        metrics = {
            'ic_mean': ic_df['ic'].mean(),
            'ic_std': ic_df['ic'].std(),
            'ic_ir': ic_df['ic'].mean() / ic_df['ic'].std() if ic_df['ic'].std() > 0 else 0,
            'ic_positive_ratio': (ic_df['ic'] > 0).mean(),
            't_stat': None,
            'p_value': None
        }
        
        # T检验: 检验IC均值是否显著不为0
        t_stat, p_value = stats.ttest_1samp(ic_df['ic'], 0)
        metrics['t_stat'] = t_stat
        metrics['p_value'] = p_value
        
        return metrics
    
    def plot_ic_analysis(self, ic_df, factor_name, stock_code, metrics):
        """绘制IC分析图表"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # 1. IC时间序列
        axes[0].plot(ic_df['period'], ic_df['ic'], marker='o', linewidth=1.5, markersize=4)
        axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0].axhline(y=ic_df['ic'].mean(), color='g', linestyle='--', alpha=0.5, 
                       label=f'Mean IC = {ic_df["ic"].mean():.4f}')
        axes[0].set_title(f'{stock_code} - {factor_name} 月度IC时间序列', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('时间')
        axes[0].set_ylabel('IC')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. IC分布直方图
        axes[1].hist(ic_df['ic'], bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
        axes[1].axvline(x=ic_df['ic'].mean(), color='g', linestyle='--', alpha=0.5, 
                       label=f'Mean = {ic_df["ic"].mean():.4f}')
        axes[1].set_title(f'IC分布 (IC_IR = {metrics["ic_ir"]:.4f})', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('IC')
        axes[1].set_ylabel('频数')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. 滚动IC
        rolling_ic = ic_df.set_index('period')['ic'].rolling(window=self.ic_window).mean()
        axes[2].plot(rolling_ic.index, rolling_ic.values, linewidth=2, color='purple')
        axes[2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[2].set_title(f'{self.ic_window}个月滚动IC均值', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('时间')
        axes[2].set_ylabel('滚动IC')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        plot_file = os.path.join(self.report_dir, f'{stock_code}_{factor_name}_ic_analysis.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_file
    
    def generate_report(self, stock_code, factor_name, metrics, ic_df, plot_file):
        """生成IC分析报告"""
        report_file = os.path.join(self.report_dir, f'{stock_code}_{factor_name}_ic_report.txt')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"信息系数 (IC) 分析报告\n")
            f.write(f"股票代码: {stock_code}\n")
            f.write(f"因子名称: {factor_name}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("【核心指标】\n")
            f.write(f"  IC均值:              {metrics['ic_mean']:.6f}\n")
            f.write(f"  IC标准差:            {metrics['ic_std']:.6f}\n")
            f.write(f"  IC_IR (信息比率):    {metrics['ic_ir']:.6f}\n")
            f.write(f"  IC正值比例:          {metrics['ic_positive_ratio']:.2%}\n")
            f.write(f"  T统计量:             {metrics['t_stat']:.4f}\n")
            f.write(f"  P值:                 {metrics['p_value']:.6f}\n\n")
            
            f.write("【评估标准】\n")
            f.write("  - 月度IC > 0.05:     因子有价值\n")
            f.write("  - IC_IR > 0.5:       因子稳定\n")
            f.write("  - P值 < 0.05:        统计显著\n\n")
            
            f.write("【结论】\n")
            if metrics['ic_mean'] > 0.05 and metrics['ic_ir'] > 0.5 and metrics['p_value'] < 0.05:
                f.write("  ✓ 因子通过IC检验，建议进入策略构建阶段。\n")
            elif metrics['ic_mean'] > 0.03 and metrics['ic_ir'] > 0.3:
                f.write("  ⚠ 因子表现一般，建议谨慎使用或与其他因子组合。\n")
            else:
                f.write("  ✗ 因子未通过IC检验，不建议使用。\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"图表已保存至: {plot_file}\n")
        
        print(f"IC分析报告已生成: {report_file}")
        return report_file
    
    def analyze_factor(self, stock_code, factor_name):
        """对单个因子进行完整的IC分析"""
        print(f"\n开始分析: {stock_code} - {factor_name}")
        
        # 加载数据
        df = self.load_data(stock_code)
        
        # 计算前瞻收益
        df = self.calculate_forward_returns(df)
        
        # 计算IC
        ic_df = self.calculate_ic(df, factor_name)
        
        if len(ic_df) < 6:  # 至少需要6个月的数据
            print(f"警告: {stock_code} 的有效数据点不足，跳过")
            return None
        
        # 计算IC指标
        metrics = self.calculate_ic_metrics(ic_df)
        
        # 绘制图表
        plot_file = self.plot_ic_analysis(ic_df, factor_name, stock_code, metrics)
        
        # 生成报告
        report_file = self.generate_report(stock_code, factor_name, metrics, ic_df, plot_file)
        
        return {
            'stock_code': stock_code,
            'factor_name': factor_name,
            'metrics': metrics,
            'report_file': report_file,
            'plot_file': plot_file
        }

def main():
    parser = argparse.ArgumentParser(description='IC分析脚本')
    parser.add_argument('--config', required=True, help='配置文件路径')
    args = parser.parse_args()
    
    # 初始化分析器
    analyzer = ICAnalyzer(args.config)
    
    # 读取股票池
    config = configparser.ConfigParser()
    config.read(args.config)
    stock_codes = config.get('Strategy', 'stock_pool').split(',')
    stock_codes = [code.strip() for code in stock_codes]
    
    # 读取要分析的因子
    factors = config.get('IC', 'factors', fallback='MA_20,RSI_14,MACD').split(',')
    factors = [f.strip() for f in factors]
    
    print(f"\n将分析 {len(stock_codes)} 只股票的 {len(factors)} 个因子")
    print(f"股票池: {stock_codes}")
    print(f"因子列表: {factors}")
    
    # 执行分析
    results = []
    for stock_code in stock_codes:
        for factor in factors:
            try:
                result = analyzer.analyze_factor(stock_code, factor)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"错误: 分析 {stock_code} - {factor} 时出错: {e}")
                continue
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("IC分析完成汇总")
    print("=" * 80)
    for result in results:
        metrics = result['metrics']
        status = "✓" if (metrics['ic_mean'] > 0.05 and metrics['ic_ir'] > 0.5) else "✗"
        print(f"{status} {result['stock_code']} - {result['factor_name']}: "
              f"IC={metrics['ic_mean']:.4f}, IR={metrics['ic_ir']:.4f}, P={metrics['p_value']:.4f}")
    
    print(f"\n所有报告已保存至: {analyzer.report_dir}")

if __name__ == '__main__':
    main()
