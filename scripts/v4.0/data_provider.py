#!/usr/bin/env python3
"""
统一数据获取模块 (Data Provider)

负责根据市场类型自动获取核心指数成分股及其完整数据。
- A股: 沪深300 + 中证1000
- 美股: 纳斯达克100 + 标普500
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class MarketConfig:
    """市场配置"""
    name: str
    indices: List[str]
    index_codes: Dict[str, str]
    data_source: str
    currency: str


# 市场配置
MARKET_CONFIGS = {
    "CN": MarketConfig(
        name="A股市场",
        indices=["沪深300", "中证1000"],
        index_codes={"沪深300": "000300.SH", "中证1000": "000852.SH"},
        data_source="tushare",
        currency="CNY"
    ),
    "US": MarketConfig(
        name="美股市场",
        indices=["纳斯达克100", "标普500"],
        index_codes={"纳斯达克100": "NDX", "标普500": "SPX"},
        data_source="yfinance",
        currency="USD"
    )
}


@dataclass
class StockData:
    """单只股票的完整数据"""
    code: str
    name: str
    market: str
    price_data: pd.DataFrame = None  # OHLCV数据
    financial_data: Dict = field(default_factory=dict)  # 财务数据
    industry: str = ""
    sector: str = ""


@dataclass
class MarketData:
    """市场级别数据"""
    macro_data: Dict = field(default_factory=dict)  # 宏观经济数据
    industry_data: Dict = field(default_factory=dict)  # 行业数据
    sentiment_data: Dict = field(default_factory=dict)  # 市场情绪数据（如VIX）


class DataProvider:
    """
    统一数据获取器
    
    自动根据市场类型获取核心指数成分股及其完整数据。
    """
    
    def __init__(self, market: str = "CN", lookback_days: int = 365, verbose: bool = True):
        """
        初始化数据获取器
        
        Args:
            market: 市场类型 ("CN" 或 "US")
            lookback_days: 历史数据回溯天数
            verbose: 是否打印详细信息
        """
        self.market = market.upper()
        if self.market not in MARKET_CONFIGS:
            raise ValueError(f"不支持的市场类型: {market}. 支持: {list(MARKET_CONFIGS.keys())}")
        
        self.config = MARKET_CONFIGS[self.market]
        self.lookback_days = lookback_days
        self.verbose = verbose
        
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=lookback_days)
        
        # 初始化数据源客户端
        self._init_data_clients()
    
    def _init_data_clients(self):
        """初始化数据源客户端"""
        if self.market == "CN":
            self._init_tushare()
        else:
            self._init_yfinance()
    
    def _init_tushare(self):
        """初始化Tushare客户端"""
        try:
            import tushare as ts
            token = os.getenv("TUSHARE_TOKEN", "")
            if token:
                ts.set_token(token)
            self.ts_pro = ts.pro_api()
            if self.verbose:
                print("✅ Tushare客户端初始化成功")
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Tushare初始化失败: {e}")
            self.ts_pro = None
    
    def _init_yfinance(self):
        """初始化yfinance客户端"""
        try:
            import yfinance as yf
            self.yf = yf
            if self.verbose:
                print("✅ yfinance客户端初始化成功")
        except Exception as e:
            if self.verbose:
                print(f"⚠️ yfinance初始化失败: {e}")
            self.yf = None
    
    def fetch_all_data(self) -> Tuple[Dict[str, StockData], MarketData]:
        """
        获取所有数据
        
        Returns:
            stock_universe: 股票池数据字典 {code: StockData}
            market_data: 市场级别数据
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"📊 开始获取 {self.config.name} 数据")
            print(f"   时间范围: {self.start_date.strftime('%Y-%m-%d')} ~ {self.end_date.strftime('%Y-%m-%d')}")
            print(f"   核心指数: {', '.join(self.config.indices)}")
            print(f"{'='*60}\n")
        
        # 1. 获取指数成分股
        stock_universe = self._fetch_index_constituents()
        
        # 2. 获取股票价格数据
        stock_universe = self._fetch_price_data(stock_universe)
        
        # 3. 获取财务数据
        stock_universe = self._fetch_financial_data(stock_universe)
        
        # 4. 获取市场级别数据
        market_data = self._fetch_market_data()
        
        if self.verbose:
            print(f"\n✅ 数据获取完成!")
            print(f"   股票数量: {len(stock_universe)}")
            print(f"   有效价格数据: {sum(1 for s in stock_universe.values() if s.price_data is not None)}")
        
        return stock_universe, market_data
    
    def _fetch_index_constituents(self) -> Dict[str, StockData]:
        """获取指数成分股"""
        stock_universe = {}
        
        if self.market == "CN":
            stock_universe = self._fetch_cn_constituents()
        else:
            stock_universe = self._fetch_us_constituents()
        
        return stock_universe
    
    def _fetch_cn_constituents(self) -> Dict[str, StockData]:
        """获取A股指数成分股"""
        stock_universe = {}
        
        if self.ts_pro is None:
            if self.verbose:
                print("⚠️ Tushare不可用，使用模拟数据")
            return self._get_mock_cn_stocks()
        
        for index_name, index_code in self.config.index_codes.items():
            try:
                if self.verbose:
                    print(f"   获取 {index_name} 成分股...")
                
                # 获取成分股列表
                df = self.ts_pro.index_weight(index_code=index_code)
                if df is not None and len(df) > 0:
                    # 取最新日期的成分股
                    latest_date = df['trade_date'].max()
                    df = df[df['trade_date'] == latest_date]
                    
                    for _, row in df.iterrows():
                        code = row['con_code']
                        if code not in stock_universe:
                            stock_universe[code] = StockData(
                                code=code,
                                name="",
                                market="CN"
                            )
                    
                    if self.verbose:
                        print(f"      ✓ {index_name}: {len(df)} 只股票")
            except Exception as e:
                if self.verbose:
                    print(f"      ✗ {index_name} 获取失败: {e}")
        
        # 获取股票基本信息
        try:
            stock_basic = self.ts_pro.stock_basic(exchange='', list_status='L')
            if stock_basic is not None:
                for code in stock_universe:
                    info = stock_basic[stock_basic['ts_code'] == code]
                    if len(info) > 0:
                        stock_universe[code].name = info.iloc[0]['name']
                        stock_universe[code].industry = info.iloc[0].get('industry', '')
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️ 获取股票基本信息失败: {e}")
        
        return stock_universe
    
    def _fetch_us_constituents(self) -> Dict[str, StockData]:
        """获取美股指数成分股"""
        stock_universe = {}
        
        # 纳斯达克100成分股（硬编码部分核心股票）
        nasdaq100_core = [
            "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "AVGO", "COST", "NFLX",
            "AMD", "ADBE", "PEP", "CSCO", "TMUS", "INTC", "CMCSA", "TXN", "QCOM", "AMGN",
            "INTU", "AMAT", "ISRG", "HON", "BKNG", "VRTX", "SBUX", "GILD", "MDLZ", "ADI",
            "ADP", "REGN", "LRCX", "PANW", "KLAC", "SNPS", "CDNS", "MELI", "ASML", "PYPL"
        ]
        
        # 标普500核心股票（补充非科技股）
        sp500_core = [
            "JPM", "V", "JNJ", "UNH", "PG", "MA", "HD", "XOM", "CVX", "BAC",
            "MRK", "ABBV", "KO", "PFE", "LLY", "WMT", "DIS", "MCD", "VZ", "NKE",
            "CRM", "TMO", "ABT", "DHR", "ORCL", "ACN", "WFC", "PM", "RTX", "NEE"
        ]
        
        all_stocks = list(set(nasdaq100_core + sp500_core))
        
        if self.verbose:
            print(f"   获取美股核心股票: {len(all_stocks)} 只")
        
        for symbol in all_stocks:
            stock_universe[symbol] = StockData(
                code=symbol,
                name=symbol,
                market="US"
            )
        
        return stock_universe
    
    def _fetch_price_data(self, stock_universe: Dict[str, StockData]) -> Dict[str, StockData]:
        """获取股票价格数据"""
        if self.verbose:
            print(f"\n📈 获取价格数据...")
        
        if self.market == "CN":
            return self._fetch_cn_price_data(stock_universe)
        else:
            return self._fetch_us_price_data(stock_universe)
    
    def _fetch_cn_price_data(self, stock_universe: Dict[str, StockData]) -> Dict[str, StockData]:
        """获取A股价格数据"""
        if self.ts_pro is None:
            return stock_universe
        
        start_str = self.start_date.strftime('%Y%m%d')
        end_str = self.end_date.strftime('%Y%m%d')
        
        success_count = 0
        for code, stock in stock_universe.items():
            try:
                df = self.ts_pro.daily(ts_code=code, start_date=start_str, end_date=end_str)
                if df is not None and len(df) > 0:
                    df = df.sort_values('trade_date')
                    df['date'] = pd.to_datetime(df['trade_date'])
                    df = df.set_index('date')
                    df = df.rename(columns={
                        'open': 'Open', 'high': 'High', 'low': 'Low',
                        'close': 'Close', 'vol': 'Volume'
                    })
                    stock.price_data = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    success_count += 1
            except Exception as e:
                pass
        
        if self.verbose:
            print(f"   ✓ 成功获取 {success_count}/{len(stock_universe)} 只股票的价格数据")
        
        return stock_universe
    
    def _fetch_us_price_data(self, stock_universe: Dict[str, StockData]) -> Dict[str, StockData]:
        """获取美股价格数据"""
        if self.yf is None:
            return stock_universe
        
        symbols = list(stock_universe.keys())
        
        try:
            # 批量下载
            data = self.yf.download(
                symbols,
                start=self.start_date,
                end=self.end_date,
                progress=False
            )
            
            success_count = 0
            for symbol in symbols:
                try:
                    if len(symbols) > 1:
                        stock_df = data.xs(symbol, level=1, axis=1) if isinstance(data.columns, pd.MultiIndex) else data
                    else:
                        stock_df = data
                    
                    if stock_df is not None and len(stock_df) > 0:
                        stock_universe[symbol].price_data = stock_df[['Open', 'High', 'Low', 'Close', 'Volume']]
                        success_count += 1
                except Exception:
                    pass
            
            if self.verbose:
                print(f"   ✓ 成功获取 {success_count}/{len(stock_universe)} 只股票的价格数据")
        
        except Exception as e:
            if self.verbose:
                print(f"   ✗ 批量获取价格数据失败: {e}")
        
        return stock_universe
    
    def _fetch_financial_data(self, stock_universe: Dict[str, StockData]) -> Dict[str, StockData]:
        """获取财务数据"""
        if self.verbose:
            print(f"\n💰 获取财务数据...")
        
        # 简化实现：为每只股票添加基本财务指标
        for code, stock in stock_universe.items():
            if stock.price_data is not None and len(stock.price_data) > 0:
                # 计算基本统计指标
                returns = stock.price_data['Close'].pct_change().dropna()
                stock.financial_data = {
                    'avg_return': returns.mean() * 252,  # 年化收益
                    'volatility': returns.std() * np.sqrt(252),  # 年化波动率
                    'sharpe': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
                    'max_drawdown': self._calc_max_drawdown(stock.price_data['Close']),
                    'avg_volume': stock.price_data['Volume'].mean()
                }
        
        if self.verbose:
            print(f"   ✓ 财务数据计算完成")
        
        return stock_universe
    
    def _calc_max_drawdown(self, prices: pd.Series) -> float:
        """计算最大回撤"""
        peak = prices.expanding(min_periods=1).max()
        drawdown = (prices - peak) / peak
        return drawdown.min()
    
    def _fetch_market_data(self) -> MarketData:
        """获取市场级别数据"""
        if self.verbose:
            print(f"\n🌍 获取市场数据...")
        
        market_data = MarketData()
        
        # 获取VIX数据（美股市场情绪指标）
        if self.yf is not None:
            try:
                vix = self.yf.download("^VIX", start=self.start_date, end=self.end_date, progress=False)
                if vix is not None and len(vix) > 0:
                    market_data.sentiment_data['VIX'] = vix['Close']
                    if self.verbose:
                        print(f"   ✓ VIX数据: {len(vix)} 条记录")
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️ VIX获取失败: {e}")
        
        return market_data
    
    def _get_mock_cn_stocks(self) -> Dict[str, StockData]:
        """获取模拟A股数据（当Tushare不可用时）"""
        mock_stocks = {
            "600519.SH": StockData(code="600519.SH", name="贵州茅台", market="CN", industry="白酒"),
            "000858.SZ": StockData(code="000858.SZ", name="五粮液", market="CN", industry="白酒"),
            "601318.SH": StockData(code="601318.SH", name="中国平安", market="CN", industry="保险"),
            "600036.SH": StockData(code="600036.SH", name="招商银行", market="CN", industry="银行"),
            "000333.SZ": StockData(code="000333.SZ", name="美的集团", market="CN", industry="家电"),
        }
        return mock_stocks


# 便捷函数
def fetch_market_data(market: str = "CN", lookback_days: int = 365, verbose: bool = True):
    """
    便捷函数：获取指定市场的完整数据
    
    Args:
        market: 市场类型 ("CN" 或 "US")
        lookback_days: 历史数据回溯天数
        verbose: 是否打印详细信息
    
    Returns:
        stock_universe: 股票池数据
        market_data: 市场级别数据
    """
    provider = DataProvider(market=market, lookback_days=lookback_days, verbose=verbose)
    return provider.fetch_all_data()


if __name__ == "__main__":
    # 测试
    print("=== 测试美股数据获取 ===")
    us_stocks, us_market = fetch_market_data("US", lookback_days=90)
    print(f"\n美股股票数量: {len(us_stocks)}")
    
    # 显示部分股票信息
    for code, stock in list(us_stocks.items())[:5]:
        if stock.price_data is not None:
            print(f"  {code}: {len(stock.price_data)} 条价格记录")
