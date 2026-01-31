#!/usr/bin/env python3
"""
Point-in-Time数据处理和动态复权模块

消除回测中的Look-Ahead Bias，确保在回测的每一天只使用该日之前的信息。

作者：quant-investor技能
版本：V2.2
日期：2026-01-31
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tushare as ts
import yfinance as yf


class PointInTimeDataManager:
    """
    Point-in-Time数据管理器
    
    核心功能：
    1. 获取真实价格（不复权）和分红数据
    2. 实现动态前复权
    3. 提供无Look-Ahead Bias的回测数据
    """
    
    def __init__(self, tushare_token=None):
        """
        初始化
        
        Args:
            tushare_token: Tushare API Token（可选）
        """
        self.tushare_token = tushare_token
        if tushare_token:
            ts.set_token(tushare_token)
            self.pro = ts.pro_api()
    
    def get_true_prices_tushare(self, symbol, start_date, end_date):
        """
        从Tushare获取真实价格（不复权）
        
        Args:
            symbol: 股票代码（如'600519.SH'）
            start_date: 开始日期（'YYYYMMDD'）
            end_date: 结束日期（'YYYYMMDD'）
        
        Returns:
            DataFrame: 真实价格数据
        """
        if not self.tushare_token:
            raise ValueError("需要提供Tushare Token")
        
        # 获取不复权价格
        df = self.pro.daily(
            ts_code=symbol,
            start_date=start_date,
            end_date=end_date,
            adj='none'  # 关键：不复权
        )
        
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date').set_index('trade_date')
        
        return df[['open', 'high', 'low', 'close', 'vol']]
    
    def get_dividends_tushare(self, symbol, start_date, end_date):
        """
        从Tushare获取分红数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            Series: 分红数据（索引为除权日，值为每股分红金额）
        """
        if not self.tushare_token:
            raise ValueError("需要提供Tushare Token")
        
        # 获取分红数据
        df = self.pro.dividend(
            ts_code=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        if df.empty:
            return pd.Series(dtype=float)
        
        df['ex_date'] = pd.to_datetime(df['ex_date'])
        df = df.set_index('ex_date')
        
        # 每股分红金额（元）
        return df['cash_div'] / 10  # Tushare返回的是每10股分红
    
    def get_true_prices_yfinance(self, symbol, start_date, end_date):
        """
        从yfinance获取真实价格（不复权）
        
        Args:
            symbol: 股票代码（如'AAPL'）
            start_date: 开始日期（'YYYY-MM-DD'）
            end_date: 结束日期（'YYYY-MM-DD'）
        
        Returns:
            DataFrame: 真实价格数据
        """
        ticker = yf.Ticker(symbol)
        
        # 获取不复权价格
        df = ticker.history(
            start=start_date,
            end=end_date,
            auto_adjust=False  # 关键：不自动复权
        )
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    def get_dividends_yfinance(self, symbol, start_date, end_date):
        """
        从yfinance获取分红数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            Series: 分红数据
        """
        ticker = yf.Ticker(symbol)
        dividends = ticker.dividends
        
        # 筛选日期范围
        mask = (dividends.index >= start_date) & (dividends.index <= end_date)
        return dividends[mask]
    
    def dynamic_forward_adjustment(self, prices, dividends, as_of_date):
        """
        动态前复权：在指定日期，只使用该日之前的分红信息
        
        Args:
            prices: DataFrame, 真实价格
            dividends: Series, 分红数据（索引为除权日）
            as_of_date: 回测日期（只使用该日期之前的信息）
        
        Returns:
            DataFrame: 动态前复权价格
        """
        adjusted_prices = prices.copy()
        
        # 只使用as_of_date之前的分红
        relevant_dividends = dividends[dividends.index <= as_of_date]
        
        if relevant_dividends.empty:
            return adjusted_prices
        
        # 对每个历史日期进行调整
        for date in adjusted_prices.index:
            # 计算该日期之后到as_of_date的所有分红
            future_divs = relevant_dividends[relevant_dividends.index > date]
            
            if not future_divs.empty:
                # 前复权：减去未来分红
                total_div_adjustment = future_divs.sum()
                
                # 调整所有价格列
                for col in ['Open', 'High', 'Low', 'Close']:
                    if col in adjusted_prices.columns:
                        adjusted_prices.loc[date, col] -= total_div_adjustment
        
        return adjusted_prices
    
    def calculate_total_return(self, prices, dividends, start_date, end_date):
        """
        使用真实价格计算总回报（包含分红）
        
        Args:
            prices: DataFrame, 真实价格
            dividends: Series, 分红数据
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            float: 总回报率
        """
        # 价格回报
        price_return = (
            prices.loc[end_date, 'Close'] / prices.loc[start_date, 'Close'] - 1
        )
        
        # 分红回报
        period_dividends = dividends[
            (dividends.index >= start_date) & (dividends.index <= end_date)
        ]
        
        dividend_return = period_dividends.sum() / prices.loc[start_date, 'Close']
        
        # 总回报
        total_return = price_return + dividend_return
        
        return total_return


class PointInTimeBacktester:
    """
    Point-in-Time回测引擎
    
    确保在回测的每一天只使用该日之前的信息，完全消除Look-Ahead Bias。
    """
    
    def __init__(self, data_manager, use_dynamic_adjustment=False):
        """
        初始化
        
        Args:
            data_manager: PointInTimeDataManager实例
            use_dynamic_adjustment: 是否使用动态前复权（False则使用真实价格模式）
        """
        self.data_manager = data_manager
        self.use_dynamic_adjustment = use_dynamic_adjustment
    
    def run(self, symbols, start_date, end_date, strategy_func, initial_capital=100000):
        """
        运行回测
        
        Args:
            symbols: list, 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            strategy_func: 策略函数，接收价格数据，返回交易信号
            initial_capital: 初始资金
        
        Returns:
            dict: 回测结果
        """
        # 获取数据
        prices_dict = {}
        dividends_dict = {}
        
        for symbol in symbols:
            prices_dict[symbol] = self.data_manager.get_true_prices_yfinance(
                symbol, start_date, end_date
            )
            dividends_dict[symbol] = self.data_manager.get_dividends_yfinance(
                symbol, start_date, end_date
            )
        
        # 初始化回测状态
        portfolio = {
            'cash': initial_capital,
            'positions': {},  # {symbol: shares}
            'value_history': [],
            'trades': []
        }
        
        # 获取所有交易日
        all_dates = prices_dict[symbols[0]].index
        
        # 逐日回测
        for current_date in all_dates:
            # 准备当日数据（关键：只使用current_date之前的数据）
            if self.use_dynamic_adjustment:
                # 方案一：动态前复权
                current_prices = {}
                for symbol in symbols:
                    hist_prices = prices_dict[symbol].loc[:current_date]
                    hist_dividends = dividends_dict[symbol][
                        dividends_dict[symbol].index <= current_date
                    ]
                    
                    current_prices[symbol] = self.data_manager.dynamic_forward_adjustment(
                        hist_prices, hist_dividends, current_date
                    )
            else:
                # 方案二：真实价格模式（推荐）
                current_prices = {
                    symbol: prices_dict[symbol].loc[:current_date]
                    for symbol in symbols
                }
            
            # 生成交易信号
            signals = strategy_func(current_prices, current_date)
            
            # 执行交易（使用真实价格）
            for symbol, signal in signals.items():
                current_price = prices_dict[symbol].loc[current_date, 'Close']
                
                if signal == 'BUY' and portfolio['cash'] > 0:
                    # 买入
                    shares = int(portfolio['cash'] * 0.95 // current_price)
                    if shares > 0:
                        cost = shares * current_price
                        portfolio['cash'] -= cost
                        portfolio['positions'][symbol] = portfolio['positions'].get(symbol, 0) + shares
                        
                        portfolio['trades'].append({
                            'date': current_date,
                            'symbol': symbol,
                            'action': 'BUY',
                            'shares': shares,
                            'price': current_price
                        })
                
                elif signal == 'SELL' and symbol in portfolio['positions']:
                    # 卖出
                    shares = portfolio['positions'][symbol]
                    proceeds = shares * current_price
                    portfolio['cash'] += proceeds
                    del portfolio['positions'][symbol]
                    
                    portfolio['trades'].append({
                        'date': current_date,
                        'symbol': symbol,
                        'action': 'SELL',
                        'shares': shares,
                        'price': current_price
                    })
            
            # 处理分红（关键！）
            for symbol in portfolio['positions']:
                if current_date in dividends_dict[symbol].index:
                    div_per_share = dividends_dict[symbol].loc[current_date]
                    total_dividend = portfolio['positions'][symbol] * div_per_share
                    portfolio['cash'] += total_dividend
                    
                    print(f"{current_date}: {symbol} 分红 {div_per_share:.2f}元/股, "
                          f"持有{portfolio['positions'][symbol]}股, "
                          f"获得分红{total_dividend:.2f}元")
            
            # 计算组合价值
            holdings_value = sum(
                portfolio['positions'][symbol] * prices_dict[symbol].loc[current_date, 'Close']
                for symbol in portfolio['positions']
            )
            total_value = portfolio['cash'] + holdings_value
            
            portfolio['value_history'].append({
                'date': current_date,
                'total_value': total_value,
                'cash': portfolio['cash'],
                'holdings_value': holdings_value
            })
        
        # 计算回测指标
        value_series = pd.DataFrame(portfolio['value_history']).set_index('date')['total_value']
        
        total_return = (value_series.iloc[-1] / initial_capital - 1) * 100
        
        returns = value_series.pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        max_drawdown = ((value_series / value_series.cummax()) - 1).min() * 100
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_value': value_series.iloc[-1],
            'trades': portfolio['trades'],
            'value_history': portfolio['value_history']
        }


def example_strategy(prices_dict, current_date):
    """
    示例策略：简单移动平均线交叉
    
    Args:
        prices_dict: dict, {symbol: DataFrame}
        current_date: 当前日期
    
    Returns:
        dict: {symbol: 'BUY'/'SELL'/None}
    """
    signals = {}
    
    for symbol, prices in prices_dict.items():
        if len(prices) < 50:
            continue
        
        # 计算MA
        ma5 = prices['Close'].rolling(5).mean().iloc[-1]
        ma20 = prices['Close'].rolling(20).mean().iloc[-1]
        
        # 金叉买入，死叉卖出
        if ma5 > ma20:
            signals[symbol] = 'BUY'
        elif ma5 < ma20:
            signals[symbol] = 'SELL'
    
    return signals


if __name__ == "__main__":
    """
    使用示例
    """
    print("=== Point-in-Time回测示例 ===\n")
    
    # 初始化数据管理器
    data_manager = PointInTimeDataManager()
    
    # 初始化回测引擎（使用真实价格模式）
    backtester = PointInTimeBacktester(
        data_manager, 
        use_dynamic_adjustment=False  # False=真实价格模式（推荐）
    )
    
    # 运行回测
    results = backtester.run(
        symbols=['AAPL', 'MSFT'],
        start_date='2023-01-01',
        end_date='2024-01-01',
        strategy_func=example_strategy,
        initial_capital=100000
    )
    
    # 输出结果
    print(f"总回报: {results['total_return']:.2f}%")
    print(f"夏普比率: {results['sharpe_ratio']:.2f}")
    print(f"最大回撤: {results['max_drawdown']:.2f}%")
    print(f"最终资产: ${results['final_value']:,.2f}")
    print(f"\n总交易次数: {len(results['trades'])}")
