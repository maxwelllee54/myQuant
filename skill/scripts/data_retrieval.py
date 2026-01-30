#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import configparser
import tushare as ts
import yfinance as yf
import pandas as pd
import os

def get_data_from_tushare(pro, stock_code, start_date, end_date):
    """从Tushare获取前复权数据"""
    print(f"尝试从Tushare获取 {stock_code} 的前复权(qfq)数据...")
    df = ts.pro_bar(ts_code=stock_code, start_date=start_date, end_date=end_date, adj='qfq')
    if df is not None and not df.empty:
        # Tushare返回的列名是ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount
        # 重命名以匹配yfinance的格式
        df.rename(columns={
            'trade_date': 'date',
            'vol': 'volume'
        }, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        # 只保留需要的列
        df = df[['open', 'high', 'low', 'close', 'volume']]
    return df

def get_data_from_yfinance(stock_code, start_date, end_date):
    """从Yahoo Finance获取数据并计算前复权价格"""
    print(f"尝试从Yahoo Finance获取 {stock_code} 的数据...")
    # 1. 获取后复权数据 (auto_adjust=True)
    hist_adj = yf.download(stock_code, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if hist_adj.empty:
        return None

    # 2. 获取不复权数据 (auto_adjust=False)
    hist_unadj = yf.download(stock_code, start=start_date, end=end_date, auto_adjust=False, progress=False)
    if hist_unadj.empty:
        return None

    print(f"计算 {stock_code} 的前复权价格...")
    # 3. 计算前复权价格
    # 使用后复权价格的收益率来避免look-ahead bias
    returns = hist_adj['Close'].pct_change()
    
    # 以第一天的不复权价格为基准，累积收益率
    forward_adj_close = hist_unadj['Close'].iloc[0] * (1 + returns).cumprod()
    forward_adj_close.iloc[0] = hist_unadj['Close'].iloc[0] # 确保第一天价格完全相等

    # 4. 构建统一格式的DataFrame
    df = pd.DataFrame({
        'open': hist_adj['Open'], # Open, High, Low通常不需要严格复权，使用后复权数据近似即可
        'high': hist_adj['High'],
        'low': hist_adj['Low'],
        'close': forward_adj_close,  # 使用我们计算的前复权收盘价
        'volume': hist_adj['Volume']
    })
    
    return df

def main():
    parser = argparse.ArgumentParser(description='数据获取脚本 (统一使用前复权价格)')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    tushare_token = config.get('data', 'tushare_token', fallback=None)
    stock_codes = config.get('data', 'stock_codes').split(',')
    start_date = config.get('data', 'start_date')
    end_date = config.get('data', 'end_date')
    output_dir = config.get('data', 'output_dir')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if tushare_token:
        ts.set_token(tushare_token)
        pro = ts.pro_api()
    else:
        pro = None

    for stock_code in stock_codes:
        print(f"\n正在处理 {stock_code}...")
        df = None
        try:
            # 优先使用Tushare (如果提供了token)
            if pro:
                df = get_data_from_tushare(pro, stock_code, start_date, end_date)
                if df is None or df.empty:
                    print(f"从Tushare获取数据失败或无数据，尝试从yfinance获取...")
                    df = get_data_from_yfinance(stock_code, start_date, end_date)
            else:
                print("未提供Tushare token，直接从yfinance获取...")
                df = get_data_from_yfinance(stock_code, start_date, end_date)

        except Exception as e:
            print(f"获取数据时发生未知错误: {e}")
            print("尝试从yfinance获取作为备用...")
            try:
                df = get_data_from_yfinance(stock_code, start_date, end_date)
            except Exception as e_yf:
                print(f"从yfinance获取数据也失败: {e_yf}")

        if df is not None and not df.empty:
            # 验证数据
            if df['close'].isna().sum() > 0:
                print(f"警告: {stock_code} 的前复权价格包含缺失值，将进行前向填充")
                df['close'].fillna(method='ffill', inplace=True)
            
            output_path = os.path.join(output_dir, f"{stock_code}.csv")
            df.to_csv(output_path)
            print(f"{stock_code} 的前复权数据已保存至 {output_path}")
        else:
            print(f"最终无法获取 {stock_code} 的数据")

if __name__ == '__main__':
    main()
