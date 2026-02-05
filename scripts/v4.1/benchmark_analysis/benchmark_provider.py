"""
V4.1 市场基准数据获取模块
=========================

提供统一的市场基准指数数据获取接口，支持美股和A股主要指数。

支持的基准：
- 美股: SPY (S&P 500), QQQ (NASDAQ 100), ^GSPC, ^IXIC
- A股: 沪深300 (000300), 中证1000 (000852), 中证500 (000905)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Union
import warnings

warnings.filterwarnings('ignore')


class BenchmarkProvider:
    """市场基准数据提供器"""
    
    # 美股基准定义
    US_BENCHMARKS = {
        'SPY': {'name': 'S&P 500 ETF', 'ticker': 'SPY', 'description': '追踪标普500指数'},
        'QQQ': {'name': 'NASDAQ 100 ETF', 'ticker': 'QQQ', 'description': '追踪纳斯达克100指数'},
        'SP500': {'name': 'S&P 500 Index', 'ticker': '^GSPC', 'description': '标普500指数'},
        'NASDAQ': {'name': 'NASDAQ Composite', 'ticker': '^IXIC', 'description': '纳斯达克综合指数'},
        'DJI': {'name': 'Dow Jones Industrial', 'ticker': '^DJI', 'description': '道琼斯工业指数'},
    }
    
    # A股基准定义
    CN_BENCHMARKS = {
        'HS300': {'name': '沪深300', 'code': '000300', 'description': '沪深300指数'},
        'ZZ1000': {'name': '中证1000', 'code': '000852', 'description': '中证1000指数'},
        'ZZ500': {'name': '中证500', 'code': '000905', 'description': '中证500指数'},
        'SZ50': {'name': '上证50', 'code': '000016', 'description': '上证50指数'},
        'CYB': {'name': '创业板指', 'code': '399006', 'description': '创业板指数'},
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        初始化基准数据提供器
        
        Args:
            cache_dir: 缓存目录路径，默认为 ~/.quant_investor/cache/benchmark/
        """
        import os
        self.cache_dir = cache_dir or os.path.expanduser("~/.quant_investor/cache/benchmark/")
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def get_benchmark_returns(
        self,
        start_date: str,
        end_date: str,
        market: str = "US",
        benchmark: Optional[str] = None
    ) -> pd.Series:
        """
        获取市场基准的日收益率序列
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            market: 市场类型 ("US" 或 "CN")
            benchmark: 基准代码，默认美股为SPY，A股为HS300
            
        Returns:
            pd.Series: 日收益率序列，索引为日期
        """
        # 确定默认基准
        if benchmark is None:
            benchmark = "SPY" if market.upper() == "US" else "HS300"
        
        # 获取价格数据
        prices = self.get_benchmark_prices(start_date, end_date, market, benchmark)
        
        if prices is None or prices.empty:
            raise ValueError(f"无法获取基准 {benchmark} 的数据")
        
        # 计算日收益率
        returns = prices.pct_change().dropna()
        returns.name = benchmark
        
        return returns
    
    def get_benchmark_prices(
        self,
        start_date: str,
        end_date: str,
        market: str = "US",
        benchmark: Optional[str] = None
    ) -> pd.Series:
        """
        获取市场基准的价格序列
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            market: 市场类型 ("US" 或 "CN")
            benchmark: 基准代码
            
        Returns:
            pd.Series: 价格序列，索引为日期
        """
        market = market.upper()
        
        if market == "US":
            return self._get_us_benchmark(start_date, end_date, benchmark or "SPY")
        elif market == "CN":
            return self._get_cn_benchmark(start_date, end_date, benchmark or "HS300")
        else:
            raise ValueError(f"不支持的市场类型: {market}")
    
    def _get_us_benchmark(
        self,
        start_date: str,
        end_date: str,
        benchmark: str
    ) -> pd.Series:
        """获取美股基准数据"""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("请安装 yfinance: pip install yfinance")
        
        # 获取ticker
        if benchmark in self.US_BENCHMARKS:
            ticker = self.US_BENCHMARKS[benchmark]['ticker']
        else:
            ticker = benchmark  # 直接使用输入作为ticker
        
        try:
            # 下载数据
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            if data.empty:
                return pd.Series()
            
            # 提取收盘价
            if 'Adj Close' in data.columns:
                prices = data['Adj Close']
            else:
                prices = data['Close']
            
            # 处理MultiIndex情况
            if isinstance(prices, pd.DataFrame):
                prices = prices.iloc[:, 0]
            
            prices.name = benchmark
            prices.index = pd.to_datetime(prices.index).date
            prices.index = pd.to_datetime(prices.index)
            
            return prices
            
        except Exception as e:
            print(f"获取美股基准 {benchmark} 数据失败: {e}")
            return pd.Series()
    
    def _get_cn_benchmark(
        self,
        start_date: str,
        end_date: str,
        benchmark: str
    ) -> pd.Series:
        """获取A股基准数据"""
        try:
            import akshare as ak
        except ImportError:
            raise ImportError("请安装 akshare: pip install akshare")
        
        # 获取指数代码
        if benchmark in self.CN_BENCHMARKS:
            code = self.CN_BENCHMARKS[benchmark]['code']
        else:
            code = benchmark
        
        try:
            # 使用akshare获取指数数据
            df = ak.index_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", "")
            )
            
            if df.empty:
                return pd.Series()
            
            # 提取收盘价
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.set_index('日期')
            prices = df['收盘']
            prices.name = benchmark
            
            return prices
            
        except Exception as e:
            print(f"获取A股基准 {benchmark} 数据失败: {e}")
            return pd.Series()
    
    def get_multiple_benchmarks(
        self,
        start_date: str,
        end_date: str,
        market: str = "US",
        benchmarks: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        获取多个基准的收益率数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            market: 市场类型
            benchmarks: 基准列表，默认获取所有主要基准
            
        Returns:
            pd.DataFrame: 多个基准的日收益率，列为基准代码
        """
        market = market.upper()
        
        if benchmarks is None:
            if market == "US":
                benchmarks = ["SPY", "QQQ"]
            else:
                benchmarks = ["HS300", "ZZ500"]
        
        returns_dict = {}
        for bm in benchmarks:
            try:
                returns = self.get_benchmark_returns(start_date, end_date, market, bm)
                if not returns.empty:
                    returns_dict[bm] = returns
            except Exception as e:
                print(f"获取基准 {bm} 失败: {e}")
        
        if not returns_dict:
            return pd.DataFrame()
        
        return pd.DataFrame(returns_dict)
    
    def get_benchmark_cumulative_returns(
        self,
        start_date: str,
        end_date: str,
        market: str = "US",
        benchmark: Optional[str] = None
    ) -> pd.Series:
        """
        获取基准的累积收益率
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            market: 市场类型
            benchmark: 基准代码
            
        Returns:
            pd.Series: 累积收益率序列
        """
        returns = self.get_benchmark_returns(start_date, end_date, market, benchmark)
        cumulative = (1 + returns).cumprod() - 1
        return cumulative
    
    def list_available_benchmarks(self, market: str = "US") -> Dict:
        """
        列出可用的基准
        
        Args:
            market: 市场类型
            
        Returns:
            Dict: 可用基准的详细信息
        """
        if market.upper() == "US":
            return self.US_BENCHMARKS.copy()
        else:
            return self.CN_BENCHMARKS.copy()
    
    def get_default_benchmark(self, market: str = "US") -> str:
        """
        获取默认基准
        
        Args:
            market: 市场类型
            
        Returns:
            str: 默认基准代码
        """
        return "SPY" if market.upper() == "US" else "HS300"


# 便捷函数
def get_benchmark_returns(
    start_date: str,
    end_date: str,
    market: str = "US",
    benchmark: Optional[str] = None
) -> pd.Series:
    """
    便捷函数：获取市场基准收益率
    
    Args:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        market: 市场类型 ("US" 或 "CN")
        benchmark: 基准代码
        
    Returns:
        pd.Series: 日收益率序列
    """
    provider = BenchmarkProvider()
    return provider.get_benchmark_returns(start_date, end_date, market, benchmark)


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("V4.1 市场基准数据获取模块测试")
    print("=" * 60)
    
    provider = BenchmarkProvider()
    
    # 测试美股基准
    print("\n1. 测试美股基准 (SPY)...")
    try:
        us_returns = provider.get_benchmark_returns(
            start_date="2024-01-01",
            end_date="2024-12-31",
            market="US",
            benchmark="SPY"
        )
        print(f"   获取数据点数: {len(us_returns)}")
        print(f"   年化收益率: {us_returns.mean() * 252:.2%}")
        print(f"   年化波动率: {us_returns.std() * np.sqrt(252):.2%}")
        print("   ✅ 美股基准测试通过")
    except Exception as e:
        print(f"   ❌ 美股基准测试失败: {e}")
    
    # 测试A股基准
    print("\n2. 测试A股基准 (沪深300)...")
    try:
        cn_returns = provider.get_benchmark_returns(
            start_date="2024-01-01",
            end_date="2024-12-31",
            market="CN",
            benchmark="HS300"
        )
        print(f"   获取数据点数: {len(cn_returns)}")
        print(f"   年化收益率: {cn_returns.mean() * 252:.2%}")
        print(f"   年化波动率: {cn_returns.std() * np.sqrt(252):.2%}")
        print("   ✅ A股基准测试通过")
    except Exception as e:
        print(f"   ❌ A股基准测试失败: {e}")
    
    # 列出可用基准
    print("\n3. 可用基准列表:")
    print("   美股:", list(provider.list_available_benchmarks("US").keys()))
    print("   A股:", list(provider.list_available_benchmarks("CN").keys()))
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
