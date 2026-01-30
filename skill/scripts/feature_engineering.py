#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import configparser
import pandas as pd
import os

def calculate_technical_indicators(df):
    """计算技术指标"""
    # 移动平均线
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()

    return df

def main():
    parser = argparse.ArgumentParser(description='特征工程脚本')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    input_dir = config.get('data', 'output_dir')
    feature_dir = config.get('features', 'feature_dir')

    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            stock_code = filename.split('.')[0]
            print(f"正在为 {stock_code} 生成特征...")
            
            input_path = os.path.join(input_dir, filename)
            df = pd.read_csv(input_path)

            # 为了兼容yfinance的数据列名，将其转换为Tushare的格式
            if 'Adj Close' in df.columns:
                df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'vol'}, inplace=True)

            df_features = calculate_technical_indicators(df)
            
            output_path = os.path.join(feature_dir, f"{stock_code}_features.csv")
            df_features.to_csv(output_path, index=False)
            print(f"{stock_code} 的特征已保存至 {output_path}")

if __name__ == '__main__':
    main()
