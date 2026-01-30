#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import configparser
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime

def calculate_returns(prices):
    """计算收益率序列"""
    return prices.pct_change().dropna()

def calculate_max_drawdown(returns):
    """计算最大回撤"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def calculate_sharpe_ratio(returns, risk_free_rate=0.03):
    """计算夏普比率（年化）"""
    excess_returns = returns - risk_free_rate / 252
    if excess_returns.std() == 0:
        return 0
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_sortino_ratio(returns, risk_free_rate=0.03):
    """计算索提诺比率（年化）"""
    excess_returns = returns - risk_free_rate / 252
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0
    return np.sqrt(252) * excess_returns.mean() / downside_returns.std()

def calculate_calmar_ratio(returns):
    """计算卡玛比率"""
    max_dd = abs(calculate_max_drawdown(returns))
    if max_dd == 0:
        return 0
    annual_return = (1 + returns.mean()) ** 252 - 1
    return annual_return / max_dd

def calculate_var(returns, confidence=0.95):
    """计算VaR（Value at Risk）"""
    return np.percentile(returns, (1 - confidence) * 100)

def calculate_cvar(returns, confidence=0.95):
    """计算CVaR（Conditional Value at Risk）"""
    var = calculate_var(returns, confidence)
    return returns[returns <= var].mean()

def calculate_win_rate(returns):
    """计算胜率"""
    if len(returns) == 0:
        return 0
    return len(returns[returns > 0]) / len(returns)

def calculate_profit_loss_ratio(returns):
    """计算盈亏比"""
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    if len(wins) == 0 or len(losses) == 0 or losses.mean() == 0:
        return 0
    return abs(wins.mean() / losses.mean())

def run_backtest_advanced(df, model):
    """执行高级回测，返回详细交易记录和收益序列"""
    initial_capital = 100000.0
    capital = initial_capital
    shares = 0
    positions = []
    daily_values = []
    
    features = [f for f in ["ma5", "ma10", "rsi", "macd", "signal_line"] if f in df.columns]
    
    for i in range(len(df) - 1):
        current_date = df.iloc[i]['trade_date'] if 'trade_date' in df.columns else df.iloc[i].name
        current_price = df.iloc[i]['close']
        next_open = df.iloc[i+1]['open']
        
        # 记录每日资产价值
        if shares > 0:
            daily_value = shares * current_price
        else:
            daily_value = capital
        daily_values.append({'date': current_date, 'value': daily_value})
        
        if not df.iloc[i][features].isnull().any():
            prediction = model.predict(df.iloc[i][features].values.reshape(1, -1))[0]
            
            # 买入信号
            if prediction == 1 and shares == 0:
                shares = capital / next_open
                capital = 0
                positions.append(("BUY", df.iloc[i+1]['trade_date'] if 'trade_date' in df.iloc[i+1] else i+1, next_open))
            # 卖出信号
            elif prediction == 0 and shares > 0:
                capital = shares * next_open
                shares = 0
                positions.append(("SELL", df.iloc[i+1]['trade_date'] if 'trade_date' in df.iloc[i+1] else i+1, next_open))
    
    # 最后一天的资产价值
    final_price = df.iloc[-1]['close']
    final_date = df.iloc[-1]['trade_date'] if 'trade_date' in df.iloc[-1] else df.iloc[-1].name
    final_value = capital if shares == 0 else shares * final_price
    daily_values.append({'date': final_date, 'value': final_value})
    
    # 计算总收益
    total_return = (final_value - initial_capital) / initial_capital
    
    return total_return, final_value, positions, pd.DataFrame(daily_values)

def calculate_risk_metrics(daily_values_df):
    """计算风险指标"""
    daily_values_df['returns'] = daily_values_df['value'].pct_change().dropna()
    returns = daily_values_df['returns'].dropna()
    
    if len(returns) == 0:
        return {}
    
    metrics = {
        'max_drawdown': calculate_max_drawdown(returns),
        'sharpe_ratio': calculate_sharpe_ratio(returns),
        'sortino_ratio': calculate_sortino_ratio(returns),
        'calmar_ratio': calculate_calmar_ratio(returns),
        'var_95': calculate_var(returns, 0.95),
        'cvar_95': calculate_cvar(returns, 0.95),
        'win_rate': calculate_win_rate(returns),
        'profit_loss_ratio': calculate_profit_loss_ratio(returns),
        'volatility': returns.std() * np.sqrt(252)  # 年化波动率
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='高级回测脚本（含Barra风险分析）')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    feature_dir = config.get('features', 'feature_dir')
    model_dir = config.get('models', 'model_dir')
    report_dir = config.get('reports', 'report_dir')

    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    all_results = []

    for model_filename in os.listdir(model_dir):
        if model_filename.endswith('_model.joblib'):
            stock_code = model_filename.split('_model.joblib')[0]
            print(f"正在为 {stock_code} 进行高级回测...")

            # 加载模型
            model_path = os.path.join(model_dir, model_filename)
            model = joblib.load(model_path)

            # 加载特征数据
            feature_filename = f"{stock_code}_features.csv"
            feature_path = os.path.join(feature_dir, feature_filename)
            if not os.path.exists(feature_path):
                print(f"未找到 {stock_code} 的特征文件，跳过回测。")
                continue
            
            df = pd.read_csv(feature_path).dropna()
            if 'trade_date' not in df.columns and 'Date' in df.columns:
                df.rename(columns={'Date': 'trade_date'}, inplace=True)

            if df.empty:
                print(f"{stock_code} 的特征数据为空，跳过回测。")
                continue

            # 执行回测
            total_return, final_capital, positions, daily_values_df = run_backtest_advanced(df, model)
            
            # 计算风险指标
            risk_metrics = calculate_risk_metrics(daily_values_df)

            # 保存结果
            result = {
                'stock_code': stock_code,
                'total_return': total_return,
                'final_capital': final_capital,
                'num_trades': len(positions),
                **risk_metrics
            }
            all_results.append(result)

            # 生成详细报告
            report_path = os.path.join(report_dir, f"{stock_code}_advanced_report.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"高级回测报告: {stock_code}\n")
                f.write(f"{'='*50}\n\n")
                
                f.write(f"【收益指标】\n")
                f.write(f"总收益率: {total_return:.2%}\n")
                f.write(f"最终资产: {final_capital:.2f} 元\n")
                f.write(f"交易次数: {len(positions)} 次\n\n")
                
                f.write(f"【风险指标】\n")
                f.write(f"最大回撤: {risk_metrics.get('max_drawdown', 0):.2%}\n")
                f.write(f"夏普比率: {risk_metrics.get('sharpe_ratio', 0):.4f}\n")
                f.write(f"索提诺比率: {risk_metrics.get('sortino_ratio', 0):.4f}\n")
                f.write(f"卡玛比率: {risk_metrics.get('calmar_ratio', 0):.4f}\n")
                f.write(f"年化波动率: {risk_metrics.get('volatility', 0):.2%}\n")
                f.write(f"VaR(95%): {risk_metrics.get('var_95', 0):.2%}\n")
                f.write(f"CVaR(95%): {risk_metrics.get('cvar_95', 0):.2%}\n\n")
                
                f.write(f"【交易质量】\n")
                f.write(f"胜率: {risk_metrics.get('win_rate', 0):.2%}\n")
                f.write(f"盈亏比: {risk_metrics.get('profit_loss_ratio', 0):.2f}\n\n")
                
                f.write(f"【交易记录】\n")
                for pos in positions:
                    f.write(f"- {pos[0]} @ {pos[2]:.2f} on {pos[1]}\n")
            
            print(f"{stock_code} 的高级回测报告已生成: {report_path}")
    
    # 生成汇总报告
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = os.path.join(report_dir, "portfolio_summary.csv")
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n组合汇总报告已生成: {summary_path}")

if __name__ == '__main__':
    main()
