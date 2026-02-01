"""
USMacroDataManager - 美股宏观数据管理器

智能调度多个数据源（FRED、yfinance、Finnhub、Tushare、AKShare），
提供统一的数据访问接口，实现自动容错和降级。

数据源优先级：
- 宏观经济数据: FRED (官方权威)
- 美股行情数据: yfinance > Finnhub > Tushare
- 实时报价/新闻: Finnhub > yfinance
- 市场指数数据: yfinance > FRED
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Union
import pandas as pd

# 导入各数据源客户端
from .fred_client import FREDClient
from .yfinance_client import YFinanceClient
from .finnhub_client import FinnhubClient


class USMacroDataManager:
    """美股宏观数据管理器"""
    
    def __init__(
        self,
        fred_api_key: Optional[str] = None,
        finnhub_api_key: Optional[str] = None,
        tushare_token: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        初始化数据管理器
        
        Args:
            fred_api_key: FRED API密钥
            finnhub_api_key: Finnhub API密钥
            tushare_token: Tushare Pro token
            cache_dir: 缓存目录
        """
        # 初始化各数据源客户端
        self.fred_client = None
        self.yfinance_client = None
        self.finnhub_client = None
        self.tushare_client = None
        self.akshare_client = None
        
        # 尝试初始化FRED客户端
        fred_key = fred_api_key or os.environ.get('FRED_API_KEY')
        if fred_key:
            try:
                self.fred_client = FREDClient(api_key=fred_key, cache_dir=cache_dir)
                print("✓ FRED客户端初始化成功")
            except Exception as e:
                print(f"✗ FRED客户端初始化失败: {e}")
        else:
            print("⚠ FRED API密钥未提供，宏观经济数据功能受限")
        
        # 初始化yfinance客户端（无需API密钥）
        try:
            self.yfinance_client = YFinanceClient(cache_dir=cache_dir)
            print("✓ YFinance客户端初始化成功")
        except Exception as e:
            print(f"✗ YFinance客户端初始化失败: {e}")
        
        # 尝试初始化Finnhub客户端
        fh_key = finnhub_api_key or os.environ.get('FINNHUB_API_KEY')
        if fh_key:
            try:
                self.finnhub_client = FinnhubClient(api_key=fh_key, cache_dir=cache_dir)
                print("✓ Finnhub客户端初始化成功")
            except Exception as e:
                print(f"⚠ Finnhub客户端初始化失败: {e}")
        
        # 尝试初始化Tushare客户端（如果有token）
        ts_token = tushare_token or os.environ.get('TUSHARE_TOKEN')
        if ts_token:
            try:
                # 动态导入，避免强制依赖
                import sys
                sys.path.append('/home/ubuntu/skills/quant-investor/scripts/v2.5/data_acquisition')
                from tushare_client import TushareClient
                self.tushare_client = TushareClient(token=ts_token)
                print("✓ Tushare客户端初始化成功")
            except Exception as e:
                print(f"⚠ Tushare客户端初始化失败: {e}")
        
        # 数据源调用统计
        self.stats = {
            'fred_calls': 0,
            'yfinance_calls': 0,
            'finnhub_calls': 0,
            'tushare_calls': 0,
            'akshare_calls': 0,
            'cache_hits': 0,
            'fallback_count': 0,
        }
    
    # ========== 宏观经济数据 (FRED为主) ==========
    
    def get_gdp(self, start_date: Optional[str] = None, real: bool = True) -> pd.DataFrame:
        """
        获取美国GDP数据
        
        Args:
            start_date: 开始日期
            real: 是否获取实际GDP（剔除通胀）
            
        Returns:
            DataFrame，包含date和value列
        """
        if not self.fred_client:
            raise ValueError("FRED客户端未初始化，无法获取GDP数据")
        
        self.stats['fred_calls'] += 1
        return self.fred_client.get_gdp(start_date=start_date, real=real)
    
    def get_cpi(self, start_date: Optional[str] = None, core: bool = False) -> pd.DataFrame:
        """
        获取美国CPI数据
        
        Args:
            start_date: 开始日期
            core: 是否获取核心CPI（剔除食品和能源）
        """
        if not self.fred_client:
            raise ValueError("FRED客户端未初始化，无法获取CPI数据")
        
        self.stats['fred_calls'] += 1
        return self.fred_client.get_cpi(start_date=start_date, core=core)
    
    def get_pce(self, start_date: Optional[str] = None, core: bool = True) -> pd.DataFrame:
        """
        获取美国PCE数据（美联储首选通胀指标）
        
        Args:
            start_date: 开始日期
            core: 是否获取核心PCE
        """
        if not self.fred_client:
            raise ValueError("FRED客户端未初始化，无法获取PCE数据")
        
        self.stats['fred_calls'] += 1
        return self.fred_client.get_pce(start_date=start_date, core=core)
    
    def get_unemployment(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """获取美国失业率数据"""
        if not self.fred_client:
            raise ValueError("FRED客户端未初始化，无法获取失业率数据")
        
        self.stats['fred_calls'] += 1
        return self.fred_client.get_unemployment(start_date=start_date)
    
    def get_nfp(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """获取美国非农就业人数数据"""
        if not self.fred_client:
            raise ValueError("FRED客户端未初始化，无法获取非农数据")
        
        self.stats['fred_calls'] += 1
        return self.fred_client.get_nfp(start_date=start_date)
    
    def get_fed_rate(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """获取联邦基金利率"""
        if not self.fred_client:
            raise ValueError("FRED客户端未初始化，无法获取联邦基金利率")
        
        self.stats['fred_calls'] += 1
        return self.fred_client.get_fed_rate(start_date=start_date)
    
    def get_treasury_yield(self, maturity: str = '10y', start_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取美国国债收益率
        
        Args:
            maturity: 期限，可选 '3m', '2y', '10y'
        """
        if not self.fred_client:
            raise ValueError("FRED客户端未初始化，无法获取国债收益率")
        
        self.stats['fred_calls'] += 1
        return self.fred_client.get_treasury_yield(maturity=maturity, start_date=start_date)
    
    def get_yield_curve(self, date: Optional[str] = None) -> Dict:
        """获取收益率曲线"""
        if not self.fred_client:
            raise ValueError("FRED客户端未初始化，无法获取收益率曲线")
        
        self.stats['fred_calls'] += 1
        return self.fred_client.get_yield_curve(date=date)
    
    def get_m2(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """获取M2货币供应量"""
        if not self.fred_client:
            raise ValueError("FRED客户端未初始化，无法获取M2数据")
        
        self.stats['fred_calls'] += 1
        return self.fred_client.get_m2(start_date=start_date)
    
    # ========== 市场指数数据 (yfinance为主，FRED为备) ==========
    
    def get_vix(self, start_date: Optional[str] = None, period: str = '1y') -> pd.DataFrame:
        """
        获取VIX恐慌指数
        
        优先使用yfinance，失败则使用FRED
        """
        # 尝试yfinance
        if self.yfinance_client:
            try:
                df = self.yfinance_client.get_index_history('vix', period=period)
                if not df.empty:
                    self.stats['yfinance_calls'] += 1
                    # 标准化列名
                    df = df[['date', 'close']].rename(columns={'close': 'value'})
                    if start_date:
                        df = df[df['date'] >= start_date]
                    return df
            except Exception as e:
                print(f"yfinance获取VIX失败: {e}")
                self.stats['fallback_count'] += 1
        
        # 降级到FRED
        if self.fred_client:
            self.stats['fred_calls'] += 1
            return self.fred_client.get_vix(start_date=start_date)
        
        return pd.DataFrame()
    
    def get_sp500(self, start_date: Optional[str] = None, period: str = '1y') -> pd.DataFrame:
        """
        获取标普500指数
        
        优先使用yfinance，失败则使用FRED
        """
        # 尝试yfinance
        if self.yfinance_client:
            try:
                df = self.yfinance_client.get_index_history('sp500', period=period)
                if not df.empty:
                    self.stats['yfinance_calls'] += 1
                    df = df[['date', 'close']].rename(columns={'close': 'value'})
                    if start_date:
                        df = df[df['date'] >= start_date]
                    return df
            except Exception as e:
                print(f"yfinance获取SP500失败: {e}")
                self.stats['fallback_count'] += 1
        
        # 降级到FRED
        if self.fred_client:
            self.stats['fred_calls'] += 1
            return self.fred_client.get_series('sp500', start_date=start_date)
        
        return pd.DataFrame()
    
    def get_nasdaq(self, period: str = '1y') -> pd.DataFrame:
        """获取纳斯达克指数"""
        if not self.yfinance_client:
            raise ValueError("YFinance客户端未初始化")
        
        self.stats['yfinance_calls'] += 1
        df = self.yfinance_client.get_index_history('nasdaq', period=period)
        if not df.empty:
            df = df[['date', 'close']].rename(columns={'close': 'value'})
        return df
    
    def get_dollar_index(self, start_date: Optional[str] = None, period: str = '1y') -> pd.DataFrame:
        """获取美元指数"""
        # 尝试yfinance
        if self.yfinance_client:
            try:
                df = self.yfinance_client.get_index_history('dxy', period=period)
                if not df.empty:
                    self.stats['yfinance_calls'] += 1
                    df = df[['date', 'close']].rename(columns={'close': 'value'})
                    if start_date:
                        df = df[df['date'] >= start_date]
                    return df
            except Exception as e:
                print(f"yfinance获取美元指数失败: {e}")
                self.stats['fallback_count'] += 1
        
        # 降级到FRED
        if self.fred_client:
            self.stats['fred_calls'] += 1
            return self.fred_client.get_series('dollar_index', start_date=start_date)
        
        return pd.DataFrame()
    
    # ========== 美股行情数据 (yfinance为主) ==========
    
    def get_stock_history(
        self,
        ticker: str,
        period: str = '1y',
        interval: str = '1d',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取美股历史行情数据
        
        Args:
            ticker: 股票代码，如 'AAPL', 'MSFT'
            period: 数据周期
            interval: K线周期
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame
        """
        # 优先使用yfinance
        if self.yfinance_client:
            try:
                df = self.yfinance_client.get_stock_history(
                    ticker, period=period, interval=interval,
                    start_date=start_date, end_date=end_date
                )
                if not df.empty:
                    self.stats['yfinance_calls'] += 1
                    return df
            except Exception as e:
                print(f"yfinance获取{ticker}失败: {e}")
                self.stats['fallback_count'] += 1
        
        # 降级到Tushare（如果可用）
        if self.tushare_client:
            try:
                # Tushare美股代码格式不同，需要转换
                df = self.tushare_client.get_us_daily(ticker, start_date=start_date, end_date=end_date)
                if df is not None and not df.empty:
                    self.stats['tushare_calls'] += 1
                    return df
            except Exception as e:
                print(f"Tushare获取{ticker}失败: {e}")
        
        return pd.DataFrame()
    
    def get_stock_info(self, ticker: str) -> Dict:
        """
        获取美股基本信息
        
        Args:
            ticker: 股票代码
            
        Returns:
            包含股票信息的字典
        """
        # 优先使用yfinance（信息更全面）
        if self.yfinance_client:
            try:
                info = self.yfinance_client.get_stock_info(ticker)
                if info:
                    self.stats['yfinance_calls'] += 1
                    return info
            except Exception as e:
                print(f"yfinance获取{ticker}信息失败: {e}")
                self.stats['fallback_count'] += 1
        
        # 降级到Finnhub
        if self.finnhub_client:
            try:
                profile = self.finnhub_client.get_company_profile(ticker)
                if profile:
                    self.stats['finnhub_calls'] += 1
                    return profile
            except Exception as e:
                print(f"Finnhub获取{ticker}信息失败: {e}")
        
        return {}
    
    def get_stock_financials(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """
        获取美股财务报表
        
        Args:
            ticker: 股票代码
            
        Returns:
            包含income_statement, balance_sheet, cash_flow的字典
        """
        if not self.yfinance_client:
            raise ValueError("YFinance客户端未初始化")
        
        self.stats['yfinance_calls'] += 1
        return self.yfinance_client.get_financials(ticker)
    
    # ========== 实时报价和新闻 (Finnhub为主) ==========
    
    def get_stock_quote(self, ticker: str) -> Dict:
        """
        获取股票实时报价
        
        Args:
            ticker: 股票代码
            
        Returns:
            包含报价信息的字典
        """
        if self.finnhub_client:
            try:
                quote = self.finnhub_client.get_quote(ticker)
                if quote and quote.get('c'):
                    self.stats['finnhub_calls'] += 1
                    return {
                        'symbol': ticker,
                        'price': quote.get('c'),
                        'open': quote.get('o'),
                        'high': quote.get('h'),
                        'low': quote.get('l'),
                        'prev_close': quote.get('pc'),
                        'change': quote.get('c', 0) - quote.get('pc', 0),
                        'change_pct': (quote.get('c', 0) - quote.get('pc', 0)) / quote.get('pc', 1) * 100 if quote.get('pc') else 0,
                        'timestamp': quote.get('t')
                    }
            except Exception as e:
                print(f"Finnhub获取{ticker}报价失败: {e}")
        
        return {}
    
    def get_company_news(self, ticker: str, days: int = 7) -> List[Dict]:
        """
        获取公司新闻
        
        Args:
            ticker: 股票代码
            days: 获取最近几天的新闻
            
        Returns:
            新闻列表
        """
        if self.finnhub_client:
            try:
                from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                to_date = datetime.now().strftime('%Y-%m-%d')
                news = self.finnhub_client.get_company_news(ticker, from_date=from_date, to_date=to_date)
                if news:
                    self.stats['finnhub_calls'] += 1
                    return news
            except Exception as e:
                print(f"Finnhub获取{ticker}新闻失败: {e}")
        
        return []
    
    def get_market_news(self, category: str = 'general') -> List[Dict]:
        """
        获取市场新闻
        
        Args:
            category: 新闻类别，可选 'general', 'forex', 'crypto', 'merger'
            
        Returns:
            新闻列表
        """
        if self.finnhub_client:
            try:
                news = self.finnhub_client.get_market_news(category=category)
                if news:
                    self.stats['finnhub_calls'] += 1
                    return news
            except Exception as e:
                print(f"Finnhub获取市场新闻失败: {e}")
        
        return []
    
    def get_economic_calendar(self) -> List[Dict]:
        """
        获取经济日历（即将发布的经济数据）
        
        Returns:
            经济事件列表
        """
        if self.finnhub_client:
            try:
                calendar = self.finnhub_client.get_economic_calendar()
                if calendar:
                    self.stats['finnhub_calls'] += 1
                    return calendar
            except Exception as e:
                print(f"Finnhub获取经济日历失败: {e}")
        
        return []
    
    def get_earnings_calendar(self, days: int = 30) -> List[Dict]:
        """
        获取财报日历
        
        Args:
            days: 获取未来几天的财报日历
            
        Returns:
            财报事件列表
        """
        if self.finnhub_client:
            try:
                from_date = datetime.now().strftime('%Y-%m-%d')
                to_date = (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d')
                calendar = self.finnhub_client.get_earnings_calendar(from_date=from_date, to_date=to_date)
                if calendar:
                    self.stats['finnhub_calls'] += 1
                    return calendar
            except Exception as e:
                print(f"Finnhub获取财报日历失败: {e}")
        
        return []
    
    def get_basic_financials(self, ticker: str) -> Dict:
        """
        获取股票财务指标
        
        Args:
            ticker: 股票代码
            
        Returns:
            包含财务指标的字典
        """
        if self.finnhub_client:
            try:
                financials = self.finnhub_client.get_basic_financials(ticker)
                if financials:
                    self.stats['finnhub_calls'] += 1
                    return financials
            except Exception as e:
                print(f"Finnhub获取{ticker}财务指标失败: {e}")
        
        return {}
    
    # ========== 综合分析数据 ==========
    
    def get_macro_snapshot(self) -> Dict:
        """
        获取宏观经济数据快照
        
        Returns:
            包含最新宏观经济指标的字典
        """
        snapshot = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data': {},
            'status': {}
        }
        
        # GDP
        try:
            gdp = self.get_gdp()
            if not gdp.empty:
                snapshot['data']['gdp'] = {
                    'date': gdp.iloc[-1]['date'].strftime('%Y-%m-%d'),
                    'value': gdp.iloc[-1]['value'],
                    'unit': 'Billion USD (Real)'
                }
                snapshot['status']['gdp'] = 'ok'
        except Exception as e:
            snapshot['status']['gdp'] = f'error: {e}'
        
        # CPI
        try:
            cpi = self.get_cpi(core=True)
            if not cpi.empty:
                snapshot['data']['cpi_core'] = {
                    'date': cpi.iloc[-1]['date'].strftime('%Y-%m-%d'),
                    'value': cpi.iloc[-1]['value'],
                    'unit': 'Index'
                }
                snapshot['status']['cpi'] = 'ok'
        except Exception as e:
            snapshot['status']['cpi'] = f'error: {e}'
        
        # 失业率
        try:
            unemployment = self.get_unemployment()
            if not unemployment.empty:
                snapshot['data']['unemployment'] = {
                    'date': unemployment.iloc[-1]['date'].strftime('%Y-%m-%d'),
                    'value': unemployment.iloc[-1]['value'],
                    'unit': '%'
                }
                snapshot['status']['unemployment'] = 'ok'
        except Exception as e:
            snapshot['status']['unemployment'] = f'error: {e}'
        
        # 联邦基金利率
        try:
            fed_rate = self.get_fed_rate()
            if not fed_rate.empty:
                snapshot['data']['fed_rate'] = {
                    'date': fed_rate.iloc[-1]['date'].strftime('%Y-%m-%d'),
                    'value': fed_rate.iloc[-1]['value'],
                    'unit': '%'
                }
                snapshot['status']['fed_rate'] = 'ok'
        except Exception as e:
            snapshot['status']['fed_rate'] = f'error: {e}'
        
        # 10年期国债收益率
        try:
            treasury = self.get_treasury_yield('10y')
            if not treasury.empty:
                snapshot['data']['treasury_10y'] = {
                    'date': treasury.iloc[-1]['date'].strftime('%Y-%m-%d'),
                    'value': treasury.iloc[-1]['value'],
                    'unit': '%'
                }
                snapshot['status']['treasury_10y'] = 'ok'
        except Exception as e:
            snapshot['status']['treasury_10y'] = f'error: {e}'
        
        # VIX
        try:
            vix = self.get_vix()
            if not vix.empty:
                snapshot['data']['vix'] = {
                    'date': vix.iloc[-1]['date'].strftime('%Y-%m-%d') if hasattr(vix.iloc[-1]['date'], 'strftime') else str(vix.iloc[-1]['date'])[:10],
                    'value': vix.iloc[-1]['value'],
                    'unit': 'Index'
                }
                snapshot['status']['vix'] = 'ok'
        except Exception as e:
            snapshot['status']['vix'] = f'error: {e}'
        
        return snapshot
    
    def get_stats(self) -> Dict:
        """获取数据源调用统计"""
        # 合并各客户端的统计
        if self.fred_client:
            fred_stats = self.fred_client.get_stats()
            self.stats['fred_cache_hits'] = fred_stats.get('cache_hits', 0)
        
        if self.yfinance_client:
            yf_stats = self.yfinance_client.get_stats()
            self.stats['yfinance_cache_hits'] = yf_stats.get('cache_hits', 0)
        
        return self.stats


# ========== 测试代码 ==========
if __name__ == '__main__':
    print("=" * 60)
    print("USMacroDataManager 测试")
    print("=" * 60)
    
    # 初始化管理器
    manager = USMacroDataManager()
    
    # 测试宏观经济数据
    print("\n1. 测试宏观经济数据快照:")
    try:
        snapshot = manager.get_macro_snapshot()
        print(f"   时间戳: {snapshot['timestamp']}")
        for key, data in snapshot['data'].items():
            print(f"   {key}: {data['value']} {data['unit']} ({data['date']})")
    except Exception as e:
        print(f"   获取快照失败: {e}")
    
    # 测试美股行情数据
    print("\n2. 测试美股行情数据:")
    try:
        aapl = manager.get_stock_history('AAPL', period='1mo')
        print(f"   AAPL获取到 {len(aapl)} 条数据")
        if not aapl.empty:
            print(f"   最新价格: ${aapl.iloc[-1]['close']:.2f}")
    except Exception as e:
        print(f"   获取AAPL失败: {e}")
    
    # 测试股票基本信息
    print("\n3. 测试股票基本信息:")
    try:
        info = manager.get_stock_info('AAPL')
        print(f"   公司名称: {info.get('name', 'N/A')}")
        print(f"   市值: ${info.get('market_cap', 0)/1e9:.2f}B")
    except Exception as e:
        print(f"   获取信息失败: {e}")
    
    # 统计信息
    print("\n" + "=" * 60)
    print("数据源调用统计:")
    stats = manager.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
