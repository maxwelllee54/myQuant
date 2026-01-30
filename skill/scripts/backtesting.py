#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import configparser
import pandas as pd
import os
import joblib

def run_backtest(df, model):
    """执行回测"""
    initial_capital = 100000.0
    capital = initial_capital
    shares = 0
    positions = []

    features = [f for f in ["ma5", "ma10", "rsi", "macd", "signal_line"] if f in df.columns]

    for i in range(len(df) - 1):
        if not df.iloc[i][features].isnull().any():
            prediction = model.predict(df.iloc[i][features].values.reshape(1, -1))[0]
            
            # 买入信号
            if prediction == 1 and shares == 0:
                shares = capital / df.iloc[i+1]['open']
                capital = 0
                positions.append(("BUY", df.iloc[i+1]['trade_date'], df.iloc[i+1]['open']))
            # 卖出信号
            elif prediction == 0 and shares > 0:
                capital = shares * df.iloc[i+1]['open']
                shares = 0
                positions.append(("SELL", df.iloc[i+1]['trade_date'], df.iloc[i+1]['open']))

    # 计算最终资产
    final_capital = capital if shares == 0 else shares * df.iloc[-1]['close']
    total_return = (final_capital - initial_capital) / initial_capital

    return total_return, final_capital, positions

def main():
    parser = argparse.ArgumentParser(description='回测脚本')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    feature_dir = config.get('features', 'feature_dir')
    model_dir = config.get('models', 'model_dir')
    report_dir = config.get('reports', 'report_dir')

    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    for model_filename in os.listdir(model_dir):
        if model_filename.endswith('_model.joblib'):
            stock_code = model_filename.split('_model.joblib')[0]
            print(f"正在为 {stock_code} 进行回测...")

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

            total_return, final_capital, positions = run_backtest(df, model)

            # 生成报告
            report_path = os.path.join(report_dir, f"{stock_code}_report.txt")
            with open(report_path, 'w') as f:
                f.write(f"回测报告: {stock_code}\n")
                f.write(f"='*20'\n")
                f.write(f"总收益率: {total_return:.2%}\n")
                f.write(f"最终资产: {final_capital:.2f}\n")
                f.write("\n交易记录:\n")
                for pos in positions:
                    f.write(f"- {pos[0]} @ {pos[2]:.2f} on {pos[1]}\n")
            
            print(f"{stock_code} 的回测报告已生成: {report_path}")

if __name__ == '__main__':
    main()
