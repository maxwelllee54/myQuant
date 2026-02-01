"""
FinnhubClient - Finnhub金融数据API客户端

封装Finnhub API，获取美股实时行情、财务数据、新闻等，包括：
- 股票实时报价和历史K线
- 公司基本信息和财务报表
- 市场新闻和公司新闻
- 内部交易和机构持仓
- 经济日历和财报日历

Finnhub API文档: https://finnhub.io/docs/api
"""

import os
import time
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
import requests
import pandas as pd


class FinnhubClient:
    """Finnhub API客户端"""
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        初始化Finnhub客户端
        
        Args:
            api_key: Finnhub API密钥，如果不提供则从环境变量FINNHUB_API_KEY获取
            cache_dir: 缓存目录，默认为~/.quant_investor/cache/finnhub
        """
        self.api_key = api_key or os.environ.get('FINNHUB_API_KEY')
        if not self.api_key:
            raise ValueError("Finnhub API密钥未提供。请设置环境变量FINNHUB_API_KEY或在初始化时传入api_key参数。")
        
        self.base_url = "https://finnhub.io/api/v1"
        
        # 设置缓存目录
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / '.quant_investor' / 'cache' / 'finnhub'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # API调用统计
        self.api_calls = 0
        self.cache_hits = 0
        
        # 缓存有效期（秒）
        self.cache_ttl = {
            'quote': 60,           # 实时报价缓存1分钟
            'candle': 3600,        # K线数据缓存1小时
            'profile': 86400*7,    # 公司信息缓存7天
            'financials': 86400,   # 财务数据缓存1天
            'news': 1800,          # 新闻缓存30分钟
            'calendar': 3600,      # 日历缓存1小时
        }
        
        # 频率限制（免费版60次/分钟）
        self.rate_limit = 60
        self.last_call_time = 0
    
    def _get_cache_key(self, endpoint: str, params: Dict) -> str:
        """生成缓存键"""
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(f"{endpoint}_{param_str}".encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str, ext: str = 'json') -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.{ext}"
    
    def _is_cache_valid(self, cache_path: Path, data_type: str) -> bool:
        """检查缓存是否有效"""
        if not cache_path.exists():
            return False
        
        mtime = cache_path.stat().st_mtime
        ttl = self.cache_ttl.get(data_type, 3600)
        return (time.time() - mtime) < ttl
    
    def _load_from_cache(self, cache_path: Path) -> Optional[Any]:
        """从缓存加载数据"""
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            self.cache_hits += 1
            return data
        except Exception:
            return None
    
    def _save_to_cache(self, data: Any, cache_path: Path):
        """保存数据到缓存"""
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"警告: 缓存保存失败: {e}")
    
    def _rate_limit_wait(self):
        """等待以遵守频率限制"""
        elapsed = time.time() - self.last_call_time
        min_interval = 1.0 / self.rate_limit * 60  # 每次调用最小间隔
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """发送API请求"""
        self._rate_limit_wait()
        
        if params is None:
            params = {}
        params['token'] = self.api_key
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            self.api_calls += 1
            self.last_call_time = time.time()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Finnhub API请求失败: {e}")
    
    # ========== 股票报价 ==========
    
    def get_quote(self, symbol: str, use_cache: bool = True) -> Dict:
        """
        获取股票实时报价
        
        Args:
            symbol: 股票代码，如 'AAPL'
            use_cache: 是否使用缓存
            
        Returns:
            包含报价信息的字典：
            - c: 当前价格
            - h: 最高价
            - l: 最低价
            - o: 开盘价
            - pc: 前收盘价
            - t: 时间戳
        """
        params = {'symbol': symbol}
        cache_key = self._get_cache_key('quote', params)
        cache_path = self._get_cache_path(cache_key)
        
        if use_cache and self._is_cache_valid(cache_path, 'quote'):
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                return cached_data
        
        data = self._make_request('quote', params)
        
        if use_cache:
            self._save_to_cache(data, cache_path)
        
        return data
    
    def get_candles(
        self,
        symbol: str,
        resolution: str = 'D',
        from_timestamp: Optional[int] = None,
        to_timestamp: Optional[int] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        获取股票K线数据
        
        Args:
            symbol: 股票代码
            resolution: K线周期，可选 '1', '5', '15', '30', '60', 'D', 'W', 'M'
            from_timestamp: 开始时间戳
            to_timestamp: 结束时间戳
            use_cache: 是否使用缓存
            
        Returns:
            DataFrame，包含open, high, low, close, volume, timestamp列
        """
        # 默认获取最近1年数据
        if to_timestamp is None:
            to_timestamp = int(time.time())
        if from_timestamp is None:
            from_timestamp = to_timestamp - 365 * 24 * 60 * 60
        
        params = {
            'symbol': symbol,
            'resolution': resolution,
            'from': from_timestamp,
            'to': to_timestamp
        }
        
        cache_key = self._get_cache_key('candle', params)
        cache_path = self._get_cache_path(cache_key)
        
        if use_cache and self._is_cache_valid(cache_path, 'candle'):
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                return pd.DataFrame(cached_data)
        
        data = self._make_request('stock/candle', params)
        
        if data.get('s') != 'ok' or 'c' not in data:
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'timestamp': data['t'],
            'open': data['o'],
            'high': data['h'],
            'low': data['l'],
            'close': data['c'],
            'volume': data['v']
        })
        
        df['date'] = pd.to_datetime(df['timestamp'], unit='s')
        
        if use_cache:
            self._save_to_cache(df.to_dict('list'), cache_path)
        
        return df
    
    # ========== 公司信息 ==========
    
    def get_company_profile(self, symbol: str, use_cache: bool = True) -> Dict:
        """
        获取公司基本信息
        
        Args:
            symbol: 股票代码
            use_cache: 是否使用缓存
            
        Returns:
            包含公司信息的字典
        """
        params = {'symbol': symbol}
        cache_key = self._get_cache_key('profile', params)
        cache_path = self._get_cache_path(cache_key)
        
        if use_cache and self._is_cache_valid(cache_path, 'profile'):
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                return cached_data
        
        data = self._make_request('stock/profile2', params)
        
        if use_cache and data:
            self._save_to_cache(data, cache_path)
        
        return data
    
    def get_basic_financials(self, symbol: str, metric: str = 'all', use_cache: bool = True) -> Dict:
        """
        获取公司财务指标
        
        Args:
            symbol: 股票代码
            metric: 指标类型，'all' 获取所有指标
            use_cache: 是否使用缓存
            
        Returns:
            包含财务指标的字典
        """
        params = {'symbol': symbol, 'metric': metric}
        cache_key = self._get_cache_key('financials', params)
        cache_path = self._get_cache_path(cache_key)
        
        if use_cache and self._is_cache_valid(cache_path, 'financials'):
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                return cached_data
        
        data = self._make_request('stock/metric', params)
        
        if use_cache and data:
            self._save_to_cache(data, cache_path)
        
        return data
    
    # ========== 新闻 ==========
    
    def get_market_news(self, category: str = 'general', min_id: int = 0, use_cache: bool = True) -> List[Dict]:
        """
        获取市场新闻
        
        Args:
            category: 新闻类别，可选 'general', 'forex', 'crypto', 'merger'
            min_id: 最小新闻ID，用于分页
            use_cache: 是否使用缓存
            
        Returns:
            新闻列表
        """
        params = {'category': category, 'minId': min_id}
        cache_key = self._get_cache_key('market_news', params)
        cache_path = self._get_cache_path(cache_key)
        
        if use_cache and self._is_cache_valid(cache_path, 'news'):
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                return cached_data
        
        data = self._make_request('news', params)
        
        if use_cache and data:
            self._save_to_cache(data, cache_path)
        
        return data if isinstance(data, list) else []
    
    def get_company_news(
        self,
        symbol: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        use_cache: bool = True
    ) -> List[Dict]:
        """
        获取公司新闻
        
        Args:
            symbol: 股票代码
            from_date: 开始日期，格式YYYY-MM-DD
            to_date: 结束日期，格式YYYY-MM-DD
            use_cache: 是否使用缓存
            
        Returns:
            新闻列表
        """
        if to_date is None:
            to_date = datetime.now().strftime('%Y-%m-%d')
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        params = {'symbol': symbol, 'from': from_date, 'to': to_date}
        cache_key = self._get_cache_key('company_news', params)
        cache_path = self._get_cache_path(cache_key)
        
        if use_cache and self._is_cache_valid(cache_path, 'news'):
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                return cached_data
        
        data = self._make_request('company-news', params)
        
        if use_cache and data:
            self._save_to_cache(data, cache_path)
        
        return data if isinstance(data, list) else []
    
    # ========== 经济日历 ==========
    
    def get_economic_calendar(self, use_cache: bool = True) -> List[Dict]:
        """
        获取经济日历（即将发布的经济数据）
        
        Returns:
            经济事件列表
        """
        cache_key = self._get_cache_key('economic_calendar', {})
        cache_path = self._get_cache_path(cache_key)
        
        if use_cache and self._is_cache_valid(cache_path, 'calendar'):
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                return cached_data
        
        data = self._make_request('calendar/economic')
        
        result = data.get('economicCalendar', []) if isinstance(data, dict) else []
        
        if use_cache and result:
            self._save_to_cache(result, cache_path)
        
        return result
    
    def get_earnings_calendar(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        symbol: Optional[str] = None,
        use_cache: bool = True
    ) -> List[Dict]:
        """
        获取财报日历
        
        Args:
            from_date: 开始日期
            to_date: 结束日期
            symbol: 股票代码（可选，不指定则获取所有）
            use_cache: 是否使用缓存
            
        Returns:
            财报事件列表
        """
        if to_date is None:
            to_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
        if from_date is None:
            from_date = datetime.now().strftime('%Y-%m-%d')
        
        params = {'from': from_date, 'to': to_date}
        if symbol:
            params['symbol'] = symbol
        
        cache_key = self._get_cache_key('earnings_calendar', params)
        cache_path = self._get_cache_path(cache_key)
        
        if use_cache and self._is_cache_valid(cache_path, 'calendar'):
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                return cached_data
        
        data = self._make_request('calendar/earnings', params)
        
        result = data.get('earningsCalendar', []) if isinstance(data, dict) else []
        
        if use_cache and result:
            self._save_to_cache(result, cache_path)
        
        return result
    
    # ========== 内部交易和机构持仓 ==========
    
    def get_insider_transactions(self, symbol: str, use_cache: bool = True) -> List[Dict]:
        """
        获取内部交易数据
        
        Args:
            symbol: 股票代码
            use_cache: 是否使用缓存
            
        Returns:
            内部交易列表
        """
        params = {'symbol': symbol}
        cache_key = self._get_cache_key('insider', params)
        cache_path = self._get_cache_path(cache_key)
        
        if use_cache and self._is_cache_valid(cache_path, 'financials'):
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                return cached_data
        
        data = self._make_request('stock/insider-transactions', params)
        
        result = data.get('data', []) if isinstance(data, dict) else []
        
        if use_cache and result:
            self._save_to_cache(result, cache_path)
        
        return result
    
    def get_stats(self) -> Dict:
        """获取API调用统计"""
        return {
            'api_calls': self.api_calls,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / (self.api_calls + self.cache_hits) if (self.api_calls + self.cache_hits) > 0 else 0
        }


# ========== 测试代码 ==========
if __name__ == '__main__':
    api_key = os.environ.get('FINNHUB_API_KEY')
    if not api_key:
        print("请设置环境变量FINNHUB_API_KEY")
        exit(1)
    
    client = FinnhubClient(api_key=api_key)
    
    print("=" * 60)
    print("Finnhub Client 测试")
    print("=" * 60)
    
    # 测试获取实时报价
    print("\n1. 获取AAPL实时报价:")
    quote = client.get_quote('AAPL')
    print(f"   当前价格: ${quote.get('c', 'N/A')}")
    print(f"   开盘价: ${quote.get('o', 'N/A')}")
    print(f"   最高价: ${quote.get('h', 'N/A')}")
    print(f"   最低价: ${quote.get('l', 'N/A')}")
    print(f"   前收盘: ${quote.get('pc', 'N/A')}")
    
    # 测试获取K线数据
    print("\n2. 获取AAPL K线数据:")
    candles = client.get_candles('AAPL', resolution='D')
    print(f"   获取到 {len(candles)} 条数据")
    if not candles.empty:
        print(f"   最新: {candles.iloc[-1]['date']} = ${candles.iloc[-1]['close']:.2f}")
    
    # 测试获取公司信息
    print("\n3. 获取AAPL公司信息:")
    profile = client.get_company_profile('AAPL')
    print(f"   公司名称: {profile.get('name', 'N/A')}")
    print(f"   行业: {profile.get('finnhubIndustry', 'N/A')}")
    print(f"   市值: ${profile.get('marketCapitalization', 0):.2f}M")
    
    # 测试获取财务指标
    print("\n4. 获取AAPL财务指标:")
    financials = client.get_basic_financials('AAPL')
    metrics = financials.get('metric', {})
    print(f"   52周最高: ${metrics.get('52WeekHigh', 'N/A')}")
    print(f"   52周最低: ${metrics.get('52WeekLow', 'N/A')}")
    print(f"   市盈率(TTM): {metrics.get('peBasicExclExtraTTM', 'N/A')}")
    
    # 测试获取公司新闻
    print("\n5. 获取AAPL公司新闻:")
    news = client.get_company_news('AAPL')
    print(f"   获取到 {len(news)} 条新闻")
    if news:
        print(f"   最新: {news[0].get('headline', 'N/A')[:50]}...")
    
    # 测试获取经济日历
    print("\n6. 获取经济日历:")
    calendar = client.get_economic_calendar()
    print(f"   获取到 {len(calendar)} 条经济事件")
    if calendar:
        event = calendar[0]
        print(f"   最近事件: {event.get('event', 'N/A')} ({event.get('country', 'N/A')})")
    
    # 统计信息
    print("\n" + "=" * 60)
    print("API调用统计:")
    stats = client.get_stats()
    print(f"   API调用次数: {stats['api_calls']}")
    print(f"   缓存命中次数: {stats['cache_hits']}")
    print(f"   缓存命中率: {stats['cache_hit_rate']:.1%}")
