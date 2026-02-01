"""
YFinanceClient - Yahoo Finance数据客户端

封装yfinance库，获取美股行情数据，包括：
- 股票日线、周线、月线行情
- 股票基本信息和财务数据
- 期权数据
- 市场指数数据

作为Tushare和AKShare的备用数据源。
"""

import os
import time
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise ImportError("请先安装yfinance: pip install yfinance")


class YFinanceClient:
    """Yahoo Finance数据客户端"""
    
    # 常用市场指数
    INDEX_MAP = {
        'sp500': '^GSPC',        # 标普500
        'nasdaq': '^IXIC',       # 纳斯达克综合指数
        'dow': '^DJI',           # 道琼斯工业平均指数
        'russell2000': '^RUT',   # 罗素2000
        'vix': '^VIX',           # VIX恐慌指数
        'dxy': 'DX-Y.NYB',       # 美元指数
        'gold': 'GC=F',          # 黄金期货
        'oil': 'CL=F',           # 原油期货
        'treasury_10y': '^TNX',  # 10年期国债收益率
        'treasury_2y': '^IRX',   # 2年期国债收益率
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        初始化YFinance客户端
        
        Args:
            cache_dir: 缓存目录，默认为~/.quant_investor/cache/yfinance
        """
        # 设置缓存目录
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / '.quant_investor' / 'cache' / 'yfinance'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # API调用统计
        self.api_calls = 0
        self.cache_hits = 0
        
        # 缓存有效期（秒）
        self.cache_ttl = {
            '1d': 3600,        # 日线数据缓存1小时
            '1wk': 86400,      # 周线数据缓存1天
            '1mo': 86400*7,    # 月线数据缓存7天
            'info': 86400,     # 基本信息缓存1天
        }
    
    def _get_cache_key(self, ticker: str, data_type: str, params: Dict) -> str:
        """生成缓存键"""
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(f"{ticker}_{data_type}_{param_str}".encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str, ext: str = 'parquet') -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.{ext}"
    
    def _is_cache_valid(self, cache_path: Path, interval: str) -> bool:
        """检查缓存是否有效"""
        if not cache_path.exists():
            return False
        
        mtime = cache_path.stat().st_mtime
        ttl = self.cache_ttl.get(interval, 3600)
        return (time.time() - mtime) < ttl
    
    def _load_from_cache(self, cache_path: Path, ext: str = 'parquet') -> Optional[Any]:
        """从缓存加载数据"""
        try:
            if ext == 'parquet':
                data = pd.read_parquet(cache_path)
            else:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
            self.cache_hits += 1
            return data
        except Exception:
            return None
    
    def _save_to_cache(self, data: Any, cache_path: Path, ext: str = 'parquet'):
        """保存数据到缓存"""
        try:
            if ext == 'parquet' and isinstance(data, pd.DataFrame):
                data.to_parquet(cache_path)
            else:
                with open(cache_path, 'w') as f:
                    json.dump(data, f)
        except Exception as e:
            print(f"警告: 缓存保存失败: {e}")
    
    def get_stock_history(
        self,
        ticker: str,
        period: str = '1y',
        interval: str = '1d',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        获取股票历史行情数据
        
        Args:
            ticker: 股票代码，如 'AAPL', 'MSFT'
            period: 数据周期，可选 '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
            interval: K线周期，可选 '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
            start_date: 开始日期，格式YYYY-MM-DD（与period二选一）
            end_date: 结束日期，格式YYYY-MM-DD
            use_cache: 是否使用缓存
            
        Returns:
            DataFrame，包含Open, High, Low, Close, Volume, Adj Close等列
        """
        # 构建缓存参数
        params = {'period': period, 'interval': interval}
        if start_date:
            params['start'] = start_date
        if end_date:
            params['end'] = end_date
        
        # 检查缓存
        cache_key = self._get_cache_key(ticker, 'history', params)
        cache_path = self._get_cache_path(cache_key)
        
        if use_cache and self._is_cache_valid(cache_path, interval):
            cached_df = self._load_from_cache(cache_path)
            if cached_df is not None:
                return cached_df
        
        # 获取数据
        try:
            stock = yf.Ticker(ticker)
            if start_date:
                df = stock.history(start=start_date, end=end_date, interval=interval)
            else:
                df = stock.history(period=period, interval=interval)
            
            self.api_calls += 1
            
            if df.empty:
                return pd.DataFrame()
            
            # 重置索引，将日期作为列
            df = df.reset_index()
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]
            
            # 保存到缓存
            if use_cache and not df.empty:
                self._save_to_cache(df, cache_path)
            
            return df
            
        except Exception as e:
            print(f"获取{ticker}历史数据失败: {e}")
            return pd.DataFrame()
    
    def get_stock_info(self, ticker: str, use_cache: bool = True) -> Dict:
        """
        获取股票基本信息
        
        Args:
            ticker: 股票代码
            use_cache: 是否使用缓存
            
        Returns:
            包含股票信息的字典
        """
        # 检查缓存
        cache_key = self._get_cache_key(ticker, 'info', {})
        cache_path = self._get_cache_path(cache_key, 'json')
        
        if use_cache and self._is_cache_valid(cache_path, 'info'):
            cached_data = self._load_from_cache(cache_path, 'json')
            if cached_data is not None:
                return cached_data
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            self.api_calls += 1
            
            # 提取关键信息
            result = {
                'symbol': ticker,
                'name': info.get('longName', info.get('shortName', '')),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'country': info.get('country', ''),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', ''),
                'market_cap': info.get('marketCap', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'trailing_pe': info.get('trailingPE', None),
                'forward_pe': info.get('forwardPE', None),
                'peg_ratio': info.get('pegRatio', None),
                'price_to_book': info.get('priceToBook', None),
                'dividend_yield': info.get('dividendYield', None),
                'beta': info.get('beta', None),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', None),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', None),
                'fifty_day_average': info.get('fiftyDayAverage', None),
                'two_hundred_day_average': info.get('twoHundredDayAverage', None),
                'revenue': info.get('totalRevenue', 0),
                'gross_profit': info.get('grossProfits', 0),
                'ebitda': info.get('ebitda', 0),
                'net_income': info.get('netIncomeToCommon', 0),
                'total_cash': info.get('totalCash', 0),
                'total_debt': info.get('totalDebt', 0),
                'free_cash_flow': info.get('freeCashflow', 0),
                'operating_cash_flow': info.get('operatingCashflow', 0),
                'employees': info.get('fullTimeEmployees', 0),
                'website': info.get('website', ''),
                'description': info.get('longBusinessSummary', ''),
            }
            
            # 保存到缓存
            if use_cache:
                self._save_to_cache(result, cache_path, 'json')
            
            return result
            
        except Exception as e:
            print(f"获取{ticker}基本信息失败: {e}")
            return {}
    
    def get_financials(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """
        获取股票财务报表
        
        Args:
            ticker: 股票代码
            
        Returns:
            包含income_statement, balance_sheet, cash_flow的字典
        """
        try:
            stock = yf.Ticker(ticker)
            self.api_calls += 1
            
            return {
                'income_statement': stock.income_stmt,
                'balance_sheet': stock.balance_sheet,
                'cash_flow': stock.cashflow,
            }
        except Exception as e:
            print(f"获取{ticker}财务报表失败: {e}")
            return {}
    
    def get_index_history(
        self,
        index_name: str,
        period: str = '1y',
        interval: str = '1d',
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        获取市场指数历史数据
        
        Args:
            index_name: 指数名称，可选 'sp500', 'nasdaq', 'dow', 'russell2000', 'vix', 'dxy', 'gold', 'oil'
            period: 数据周期
            interval: K线周期
            use_cache: 是否使用缓存
            
        Returns:
            DataFrame
        """
        ticker = self.INDEX_MAP.get(index_name.lower(), index_name)
        return self.get_stock_history(ticker, period=period, interval=interval, use_cache=use_cache)
    
    def get_multiple_stocks(
        self,
        tickers: List[str],
        period: str = '1y',
        interval: str = '1d'
    ) -> Dict[str, pd.DataFrame]:
        """
        批量获取多只股票的历史数据
        
        Args:
            tickers: 股票代码列表
            period: 数据周期
            interval: K线周期
            
        Returns:
            字典，键为股票代码，值为DataFrame
        """
        result = {}
        for ticker in tickers:
            df = self.get_stock_history(ticker, period=period, interval=interval)
            if not df.empty:
                result[ticker] = df
        return result
    
    def get_options_chain(self, ticker: str, expiration_date: Optional[str] = None) -> Dict:
        """
        获取期权链数据
        
        Args:
            ticker: 股票代码
            expiration_date: 到期日，格式YYYY-MM-DD，不指定则返回最近到期日
            
        Returns:
            包含calls和puts的字典
        """
        try:
            stock = yf.Ticker(ticker)
            self.api_calls += 1
            
            # 获取可用的到期日
            expirations = stock.options
            if not expirations:
                return {'calls': pd.DataFrame(), 'puts': pd.DataFrame(), 'expirations': []}
            
            # 选择到期日
            if expiration_date and expiration_date in expirations:
                exp = expiration_date
            else:
                exp = expirations[0]  # 使用最近的到期日
            
            # 获取期权链
            opt = stock.option_chain(exp)
            
            return {
                'calls': opt.calls,
                'puts': opt.puts,
                'expiration': exp,
                'expirations': list(expirations),
            }
            
        except Exception as e:
            print(f"获取{ticker}期权数据失败: {e}")
            return {'calls': pd.DataFrame(), 'puts': pd.DataFrame(), 'expirations': []}
    
    def get_stats(self) -> Dict:
        """获取API调用统计"""
        return {
            'api_calls': self.api_calls,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / (self.api_calls + self.cache_hits) if (self.api_calls + self.cache_hits) > 0 else 0
        }


# ========== 测试代码 ==========
if __name__ == '__main__':
    client = YFinanceClient()
    
    print("=" * 60)
    print("YFinance Client 测试")
    print("=" * 60)
    
    # 测试获取股票历史数据
    print("\n1. 获取AAPL历史数据:")
    aapl = client.get_stock_history('AAPL', period='3mo')
    print(f"   获取到 {len(aapl)} 条数据")
    if not aapl.empty:
        print(f"   最新价格: {aapl.iloc[-1]['date']} = ${aapl.iloc[-1]['close']:.2f}")
    
    # 测试获取股票基本信息
    print("\n2. 获取AAPL基本信息:")
    info = client.get_stock_info('AAPL')
    if info:
        print(f"   公司名称: {info.get('name', 'N/A')}")
        print(f"   市值: ${info.get('market_cap', 0)/1e9:.2f}B")
        print(f"   市盈率(TTM): {info.get('trailing_pe', 'N/A')}")
        print(f"   行业: {info.get('industry', 'N/A')}")
    
    # 测试获取市场指数
    print("\n3. 获取标普500指数:")
    sp500 = client.get_index_history('sp500', period='1mo')
    print(f"   获取到 {len(sp500)} 条数据")
    if not sp500.empty:
        print(f"   最新点位: {sp500.iloc[-1]['date']} = {sp500.iloc[-1]['close']:.2f}")
    
    # 测试获取VIX
    print("\n4. 获取VIX恐慌指数:")
    vix = client.get_index_history('vix', period='1mo')
    print(f"   获取到 {len(vix)} 条数据")
    if not vix.empty:
        print(f"   最新VIX: {vix.iloc[-1]['date']} = {vix.iloc[-1]['close']:.2f}")
    
    # 测试缓存（再次获取相同数据）
    print("\n5. 测试缓存机制:")
    aapl2 = client.get_stock_history('AAPL', period='3mo')
    print(f"   再次获取AAPL数据: {len(aapl2)} 条")
    
    # 统计信息
    print("\n" + "=" * 60)
    print("API调用统计:")
    stats = client.get_stats()
    print(f"   API调用次数: {stats['api_calls']}")
    print(f"   缓存命中次数: {stats['cache_hits']}")
    print(f"   缓存命中率: {stats['cache_hit_rate']:.1%}")
