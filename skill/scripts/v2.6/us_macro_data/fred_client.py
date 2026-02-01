"""
FREDClient - 美联储经济数据库(FRED) API客户端

封装FRED API，获取美国核心宏观经济数据，包括：
- GDP、实际GDP
- CPI、核心CPI、PCE、核心PCE
- 联邦基金利率、10年期国债收益率
- 失业率、M2货币供应量
- 工业生产指数、零售销售等

FRED API文档: https://fred.stlouisfed.org/docs/api/fred/
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


class FREDClient:
    """FRED API客户端，获取美国宏观经济数据"""
    
    # 常用数据系列ID映射
    SERIES_MAP = {
        # GDP相关
        'gdp': 'GDP',                    # 名义GDP (季度, 十亿美元)
        'gdp_real': 'GDPC1',             # 实际GDP (季度, 2017年基准)
        'gdp_growth': 'A191RL1Q225SBEA', # GDP环比增速 (季度, 年化)
        
        # 通胀相关
        'cpi': 'CPIAUCSL',               # CPI城市消费者 (月度)
        'cpi_core': 'CPILFESL',          # 核心CPI (剔除食品能源, 月度)
        'pce': 'PCE',                    # 个人消费支出 (月度)
        'pce_core': 'PCEPILFE',          # 核心PCE (月度)
        'ppi': 'PPIACO',                 # 生产者价格指数 (月度)
        
        # 就业相关
        'unemployment': 'UNRATE',         # 失业率 (月度, %)
        'nfp': 'PAYEMS',                 # 非农就业人数 (月度, 千人)
        'initial_claims': 'ICSA',        # 首次申请失业金人数 (周度)
        
        # 利率相关
        'fed_rate': 'DFF',               # 联邦基金有效利率 (日度)
        'fed_rate_target': 'DFEDTARU',   # 联邦基金目标利率上限 (日度)
        'treasury_10y': 'DGS10',         # 10年期国债收益率 (日度)
        'treasury_2y': 'DGS2',           # 2年期国债收益率 (日度)
        'treasury_3m': 'DTB3',           # 3个月国债收益率 (日度)
        
        # 货币供应
        'm2': 'M2SL',                    # M2货币供应量 (月度)
        'fed_balance': 'WALCL',          # 美联储资产负债表 (周度)
        
        # 经济活动
        'industrial_prod': 'INDPRO',     # 工业生产指数 (月度)
        'retail_sales': 'RSXFS',         # 零售销售 (月度)
        'housing_starts': 'HOUST',       # 房屋开工数 (月度, 千套)
        'consumer_sentiment': 'UMCSENT', # 密歇根消费者信心指数 (月度)
        
        # 市场指标
        'vix': 'VIXCLS',                 # VIX恐慌指数 (日度)
        'sp500': 'SP500',                # 标普500指数 (日度)
        'dollar_index': 'DTWEXBGS',      # 美元指数 (日度)
    }
    
    # 数据系列的更新频率
    FREQUENCY_MAP = {
        'gdp': 'quarterly',
        'gdp_real': 'quarterly',
        'gdp_growth': 'quarterly',
        'cpi': 'monthly',
        'cpi_core': 'monthly',
        'pce': 'monthly',
        'pce_core': 'monthly',
        'ppi': 'monthly',
        'unemployment': 'monthly',
        'nfp': 'monthly',
        'initial_claims': 'weekly',
        'fed_rate': 'daily',
        'fed_rate_target': 'daily',
        'treasury_10y': 'daily',
        'treasury_2y': 'daily',
        'treasury_3m': 'daily',
        'm2': 'monthly',
        'fed_balance': 'weekly',
        'industrial_prod': 'monthly',
        'retail_sales': 'monthly',
        'housing_starts': 'monthly',
        'consumer_sentiment': 'monthly',
        'vix': 'daily',
        'sp500': 'daily',
        'dollar_index': 'daily',
    }
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        初始化FRED客户端
        
        Args:
            api_key: FRED API密钥，如果不提供则从环境变量FRED_API_KEY获取
            cache_dir: 缓存目录，默认为~/.quant_investor/cache/fred
        """
        self.api_key = api_key or os.environ.get('FRED_API_KEY')
        if not self.api_key:
            raise ValueError("FRED API密钥未提供。请设置环境变量FRED_API_KEY或在初始化时传入api_key参数。")
        
        self.base_url = "https://api.stlouisfed.org/fred"
        
        # 设置缓存目录
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / '.quant_investor' / 'cache' / 'fred'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # API调用统计
        self.api_calls = 0
        self.cache_hits = 0
        
        # 缓存有效期（秒）
        self.cache_ttl = {
            'daily': 3600,       # 日度数据缓存1小时
            'weekly': 86400,     # 周度数据缓存1天
            'monthly': 86400*7,  # 月度数据缓存7天
            'quarterly': 86400*30,  # 季度数据缓存30天
        }
    
    def _get_cache_key(self, series_id: str, params: Dict) -> str:
        """生成缓存键"""
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(f"{series_id}_{param_str}".encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.parquet"
    
    def _is_cache_valid(self, cache_path: Path, frequency: str) -> bool:
        """检查缓存是否有效"""
        if not cache_path.exists():
            return False
        
        # 检查缓存文件的修改时间
        mtime = cache_path.stat().st_mtime
        ttl = self.cache_ttl.get(frequency, 3600)
        return (time.time() - mtime) < ttl
    
    def _load_from_cache(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """从缓存加载数据"""
        try:
            df = pd.read_parquet(cache_path)
            self.cache_hits += 1
            return df
        except Exception:
            return None
    
    def _save_to_cache(self, df: pd.DataFrame, cache_path: Path):
        """保存数据到缓存"""
        try:
            df.to_parquet(cache_path)
        except Exception as e:
            print(f"警告: 缓存保存失败: {e}")
    
    def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """发送API请求"""
        params['api_key'] = self.api_key
        params['file_type'] = 'json'
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            self.api_calls += 1
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"FRED API请求失败: {e}")
    
    def get_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        获取指定数据系列
        
        Args:
            series_id: 数据系列ID，可以是简称(如'gdp')或FRED系列ID(如'GDP')
            start_date: 开始日期，格式YYYY-MM-DD
            end_date: 结束日期，格式YYYY-MM-DD
            use_cache: 是否使用缓存
            
        Returns:
            DataFrame，包含date和value两列
        """
        # 转换简称为FRED系列ID
        fred_series_id = self.SERIES_MAP.get(series_id.lower(), series_id)
        frequency = self.FREQUENCY_MAP.get(series_id.lower(), 'daily')
        
        # 构建请求参数
        params = {'series_id': fred_series_id}
        if start_date:
            params['observation_start'] = start_date
        if end_date:
            params['observation_end'] = end_date
        
        # 检查缓存
        cache_key = self._get_cache_key(fred_series_id, params)
        cache_path = self._get_cache_path(cache_key)
        
        if use_cache and self._is_cache_valid(cache_path, frequency):
            cached_df = self._load_from_cache(cache_path)
            if cached_df is not None:
                return cached_df
        
        # 发送API请求
        data = self._make_request('series/observations', params)
        
        # 解析数据
        observations = data.get('observations', [])
        if not observations:
            return pd.DataFrame(columns=['date', 'value'])
        
        df = pd.DataFrame(observations)
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df[['date', 'value']].dropna()
        
        # 保存到缓存
        if use_cache:
            self._save_to_cache(df, cache_path)
        
        return df
    
    def get_series_info(self, series_id: str) -> Dict:
        """
        获取数据系列的元信息
        
        Args:
            series_id: 数据系列ID
            
        Returns:
            包含系列信息的字典
        """
        fred_series_id = self.SERIES_MAP.get(series_id.lower(), series_id)
        data = self._make_request('series', {'series_id': fred_series_id})
        
        if 'seriess' in data and len(data['seriess']) > 0:
            return data['seriess'][0]
        return {}
    
    def get_latest_value(self, series_id: str) -> Dict:
        """
        获取数据系列的最新值
        
        Args:
            series_id: 数据系列ID
            
        Returns:
            包含最新日期和值的字典
        """
        df = self.get_series(series_id)
        if df.empty:
            return {'date': None, 'value': None}
        
        latest = df.iloc[-1]
        return {
            'date': latest['date'].strftime('%Y-%m-%d'),
            'value': latest['value']
        }
    
    # ========== 便捷方法 ==========
    
    def get_gdp(self, start_date: Optional[str] = None, real: bool = True) -> pd.DataFrame:
        """获取GDP数据"""
        series_id = 'gdp_real' if real else 'gdp'
        return self.get_series(series_id, start_date=start_date)
    
    def get_cpi(self, start_date: Optional[str] = None, core: bool = False) -> pd.DataFrame:
        """获取CPI数据"""
        series_id = 'cpi_core' if core else 'cpi'
        return self.get_series(series_id, start_date=start_date)
    
    def get_pce(self, start_date: Optional[str] = None, core: bool = True) -> pd.DataFrame:
        """获取PCE数据（美联储首选通胀指标）"""
        series_id = 'pce_core' if core else 'pce'
        return self.get_series(series_id, start_date=start_date)
    
    def get_unemployment(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """获取失业率数据"""
        return self.get_series('unemployment', start_date=start_date)
    
    def get_nfp(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """获取非农就业人数数据"""
        return self.get_series('nfp', start_date=start_date)
    
    def get_fed_rate(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """获取联邦基金利率"""
        return self.get_series('fed_rate', start_date=start_date)
    
    def get_treasury_yield(self, maturity: str = '10y', start_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取国债收益率
        
        Args:
            maturity: 期限，可选 '3m', '2y', '10y'
        """
        series_map = {'3m': 'treasury_3m', '2y': 'treasury_2y', '10y': 'treasury_10y'}
        series_id = series_map.get(maturity, 'treasury_10y')
        return self.get_series(series_id, start_date=start_date)
    
    def get_yield_curve(self, date: Optional[str] = None) -> Dict:
        """
        获取收益率曲线
        
        Args:
            date: 指定日期，默认为最新
            
        Returns:
            包含各期限收益率的字典
        """
        maturities = ['3m', '2y', '10y']
        result = {}
        
        for m in maturities:
            df = self.get_treasury_yield(m)
            if not df.empty:
                if date:
                    row = df[df['date'] == date]
                    if not row.empty:
                        result[m] = row.iloc[0]['value']
                else:
                    result[m] = df.iloc[-1]['value']
        
        # 计算利差
        if '10y' in result and '2y' in result:
            result['spread_10y_2y'] = result['10y'] - result['2y']
        
        return result
    
    def get_m2(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """获取M2货币供应量"""
        return self.get_series('m2', start_date=start_date)
    
    def get_vix(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """获取VIX恐慌指数"""
        return self.get_series('vix', start_date=start_date)
    
    def get_stats(self) -> Dict:
        """获取API调用统计"""
        return {
            'api_calls': self.api_calls,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / (self.api_calls + self.cache_hits) if (self.api_calls + self.cache_hits) > 0 else 0
        }


# ========== 测试代码 ==========
if __name__ == '__main__':
    # 测试需要FRED API密钥
    api_key = os.environ.get('FRED_API_KEY')
    if not api_key:
        print("请设置环境变量FRED_API_KEY")
        print("获取免费API密钥: https://fred.stlouisfed.org/docs/api/api_key.html")
        exit(1)
    
    client = FREDClient(api_key=api_key)
    
    print("=" * 60)
    print("FRED Client 测试")
    print("=" * 60)
    
    # 测试获取GDP
    print("\n1. 获取实际GDP数据:")
    gdp = client.get_gdp(start_date='2020-01-01')
    print(f"   获取到 {len(gdp)} 条数据")
    print(f"   最新GDP: {gdp.iloc[-1]['date'].strftime('%Y-%m-%d')} = ${gdp.iloc[-1]['value']:.1f}B")
    
    # 测试获取CPI
    print("\n2. 获取核心CPI数据:")
    cpi = client.get_cpi(start_date='2020-01-01', core=True)
    print(f"   获取到 {len(cpi)} 条数据")
    print(f"   最新核心CPI: {cpi.iloc[-1]['date'].strftime('%Y-%m-%d')} = {cpi.iloc[-1]['value']:.2f}")
    
    # 测试获取失业率
    print("\n3. 获取失业率数据:")
    unemployment = client.get_unemployment(start_date='2020-01-01')
    print(f"   获取到 {len(unemployment)} 条数据")
    print(f"   最新失业率: {unemployment.iloc[-1]['date'].strftime('%Y-%m-%d')} = {unemployment.iloc[-1]['value']:.1f}%")
    
    # 测试获取联邦基金利率
    print("\n4. 获取联邦基金利率:")
    fed_rate = client.get_fed_rate(start_date='2020-01-01')
    print(f"   获取到 {len(fed_rate)} 条数据")
    print(f"   最新利率: {fed_rate.iloc[-1]['date'].strftime('%Y-%m-%d')} = {fed_rate.iloc[-1]['value']:.2f}%")
    
    # 测试获取收益率曲线
    print("\n5. 获取收益率曲线:")
    yield_curve = client.get_yield_curve()
    for k, v in yield_curve.items():
        print(f"   {k}: {v:.2f}%")
    
    # 测试获取VIX
    print("\n6. 获取VIX恐慌指数:")
    vix = client.get_vix(start_date='2024-01-01')
    print(f"   获取到 {len(vix)} 条数据")
    print(f"   最新VIX: {vix.iloc[-1]['date'].strftime('%Y-%m-%d')} = {vix.iloc[-1]['value']:.2f}")
    
    # 统计信息
    print("\n" + "=" * 60)
    print("API调用统计:")
    stats = client.get_stats()
    print(f"   API调用次数: {stats['api_calls']}")
    print(f"   缓存命中次数: {stats['cache_hits']}")
    print(f"   缓存命中率: {stats['cache_hit_rate']:.1%}")
