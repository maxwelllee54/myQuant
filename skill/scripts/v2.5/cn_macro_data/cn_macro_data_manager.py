"""
CNMacroDataManager - 中国宏观数据管理器
对标美股V2.6的USMacroDataManager，为A股提供完整的宏观经济数据支持

数据覆盖:
- 宏观经济: GDP, CPI, PPI, PMI
- 货币政策: LPR, Shibor, MLF, 国债收益率
- 货币供应: M2, 社会融资规模
- 市场指数: 沪深300, 创业板, 中证500等
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path

import pandas as pd
import tushare as ts

# 尝试导入akshare
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False


class CNMacroDataManager:
    """中国宏观数据管理器"""
    
    # 主要市场指数代码
    INDEX_CODES = {
        'sse': '000001.SH',      # 上证指数
        'szse': '399001.SZ',     # 深证成指
        'hs300': '000300.SH',    # 沪深300
        'csi500': '000905.SH',   # 中证500
        'gem': '399006.SZ',      # 创业板指
        'star50': '000688.SH',   # 科创50
    }
    
    def __init__(self, tushare_token: Optional[str] = None, cache_dir: str = '/tmp/cn_macro_cache'):
        """
        初始化中国宏观数据管理器
        
        Args:
            tushare_token: Tushare Pro API token
            cache_dir: 缓存目录
        """
        self.token = tushare_token or os.environ.get('TUSHARE_TOKEN')
        if self.token:
            ts.set_token(self.token)
            self.pro = ts.pro_api()
        else:
            self.pro = None
            
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # API调用统计
        self.api_calls = {'tushare': 0, 'akshare': 0, 'cache_hits': 0}
    
    def _get_cache_key(self, func_name: str, **kwargs) -> str:
        """生成缓存键"""
        params_str = json.dumps(kwargs, sort_keys=True)
        return hashlib.md5(f"{func_name}:{params_str}".encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str, max_age_hours: int = 24) -> Optional[pd.DataFrame]:
        """从缓存获取数据"""
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        if cache_file.exists():
            mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - mtime < timedelta(hours=max_age_hours):
                self.api_calls['cache_hits'] += 1
                return pd.read_parquet(cache_file)
        return None
    
    def _save_to_cache(self, cache_key: str, df: pd.DataFrame):
        """保存数据到缓存"""
        if df is not None and not df.empty:
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            df.to_parquet(cache_file)
    
    # ==================== 宏观经济数据 ====================
    
    def get_gdp(self, start_quarter: str = None, end_quarter: str = None) -> pd.DataFrame:
        """
        获取中国GDP数据
        
        Args:
            start_quarter: 开始季度，格式如 '2020Q1'
            end_quarter: 结束季度
            
        Returns:
            DataFrame with GDP data
        """
        cache_key = self._get_cache_key('gdp', start=start_quarter, end=end_quarter)
        cached = self._get_from_cache(cache_key, max_age_hours=168)  # GDP数据缓存7天
        if cached is not None:
            return cached
        
        df = None
        
        # 优先使用Tushare
        if self.pro:
            try:
                df = self.pro.cn_gdp()
                self.api_calls['tushare'] += 1
            except Exception as e:
                print(f"Tushare获取GDP失败: {e}")
        
        # 备用AKShare
        if df is None or df.empty:
            if AKSHARE_AVAILABLE:
                try:
                    df = ak.macro_china_gdp()
                    self.api_calls['akshare'] += 1
                except Exception as e:
                    print(f"AKShare获取GDP失败: {e}")
        
        if df is not None and not df.empty:
            self._save_to_cache(cache_key, df)
        
        return df
    
    def get_cpi(self, start_month: str = None, end_month: str = None) -> pd.DataFrame:
        """
        获取中国CPI数据
        
        Args:
            start_month: 开始月份，格式如 '202001'
            end_month: 结束月份
            
        Returns:
            DataFrame with CPI data
        """
        cache_key = self._get_cache_key('cpi', start=start_month, end=end_month)
        cached = self._get_from_cache(cache_key, max_age_hours=168)
        if cached is not None:
            return cached
        
        df = None
        
        # 优先使用Tushare
        if self.pro:
            try:
                df = self.pro.cn_cpi(start_m=start_month, end_m=end_month)
                self.api_calls['tushare'] += 1
            except Exception as e:
                print(f"Tushare获取CPI失败: {e}")
        
        # 备用AKShare
        if df is None or df.empty:
            if AKSHARE_AVAILABLE:
                try:
                    df = ak.macro_china_cpi_monthly()
                    self.api_calls['akshare'] += 1
                except Exception as e:
                    print(f"AKShare获取CPI失败: {e}")
        
        if df is not None and not df.empty:
            self._save_to_cache(cache_key, df)
        
        return df
    
    def get_ppi(self, start_month: str = None, end_month: str = None) -> pd.DataFrame:
        """
        获取中国PPI数据
        
        Args:
            start_month: 开始月份
            end_month: 结束月份
            
        Returns:
            DataFrame with PPI data
        """
        cache_key = self._get_cache_key('ppi', start=start_month, end=end_month)
        cached = self._get_from_cache(cache_key, max_age_hours=168)
        if cached is not None:
            return cached
        
        df = None
        
        # 优先使用Tushare
        if self.pro:
            try:
                df = self.pro.cn_ppi(start_m=start_month, end_m=end_month)
                self.api_calls['tushare'] += 1
            except Exception as e:
                print(f"Tushare获取PPI失败: {e}")
        
        # 备用AKShare
        if df is None or df.empty:
            if AKSHARE_AVAILABLE:
                try:
                    df = ak.macro_china_ppi_yearly()
                    self.api_calls['akshare'] += 1
                except Exception as e:
                    print(f"AKShare获取PPI失败: {e}")
        
        if df is not None and not df.empty:
            self._save_to_cache(cache_key, df)
        
        return df
    
    def get_pmi(self, start_month: str = None, end_month: str = None) -> pd.DataFrame:
        """
        获取中国PMI数据（制造业采购经理指数）
        
        Args:
            start_month: 开始月份
            end_month: 结束月份
            
        Returns:
            DataFrame with PMI data
        """
        cache_key = self._get_cache_key('pmi', start=start_month, end=end_month)
        cached = self._get_from_cache(cache_key, max_age_hours=168)
        if cached is not None:
            return cached
        
        df = None
        
        # 优先使用Tushare
        if self.pro:
            try:
                df = self.pro.cn_pmi(start_m=start_month, end_m=end_month)
                self.api_calls['tushare'] += 1
            except Exception as e:
                print(f"Tushare获取PMI失败: {e}")
        
        # 备用AKShare
        if df is None or df.empty:
            if AKSHARE_AVAILABLE:
                try:
                    df = ak.macro_china_pmi()
                    self.api_calls['akshare'] += 1
                except Exception as e:
                    print(f"AKShare获取PMI失败: {e}")
        
        if df is not None and not df.empty:
            self._save_to_cache(cache_key, df)
        
        return df
    
    # ==================== 货币政策数据 ====================
    
    def get_lpr(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        获取LPR贷款基础利率
        
        Args:
            start_date: 开始日期，格式如 '20200101'
            end_date: 结束日期
            
        Returns:
            DataFrame with LPR data (1年期和5年期)
        """
        cache_key = self._get_cache_key('lpr', start=start_date, end=end_date)
        cached = self._get_from_cache(cache_key, max_age_hours=24)
        if cached is not None:
            return cached
        
        df = None
        
        # 优先使用Tushare
        if self.pro:
            try:
                df = self.pro.shibor_lpr(start_date=start_date, end_date=end_date)
                self.api_calls['tushare'] += 1
            except Exception as e:
                print(f"Tushare获取LPR失败: {e}")
        
        # 备用AKShare
        if df is None or df.empty:
            if AKSHARE_AVAILABLE:
                try:
                    df = ak.macro_china_lpr()
                    self.api_calls['akshare'] += 1
                except Exception as e:
                    print(f"AKShare获取LPR失败: {e}")
        
        if df is not None and not df.empty:
            self._save_to_cache(cache_key, df)
        
        return df
    
    def get_shibor(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        获取Shibor上海银行间同业拆放利率
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame with Shibor data (隔夜、1周、1月、3月、6月、1年)
        """
        cache_key = self._get_cache_key('shibor', start=start_date, end=end_date)
        cached = self._get_from_cache(cache_key, max_age_hours=24)
        if cached is not None:
            return cached
        
        df = None
        
        # 优先使用Tushare
        if self.pro:
            try:
                df = self.pro.shibor(start_date=start_date, end_date=end_date)
                self.api_calls['tushare'] += 1
            except Exception as e:
                print(f"Tushare获取Shibor失败: {e}")
        
        # 备用AKShare
        if df is None or df.empty:
            if AKSHARE_AVAILABLE:
                try:
                    df = ak.rate_interbank(market="上海银行间同业拆放利率", symbol="Shibor人民币", indicator="隔夜")
                    self.api_calls['akshare'] += 1
                except Exception as e:
                    print(f"AKShare获取Shibor失败: {e}")
        
        if df is not None and not df.empty:
            self._save_to_cache(cache_key, df)
        
        return df
    
    def get_bond_yield(self, maturity: str = '10y') -> pd.DataFrame:
        """
        获取中国国债收益率
        
        Args:
            maturity: 期限，如 '1y', '2y', '5y', '10y', '30y'
            
        Returns:
            DataFrame with bond yield data
        """
        cache_key = self._get_cache_key('bond_yield', maturity=maturity)
        cached = self._get_from_cache(cache_key, max_age_hours=24)
        if cached is not None:
            return cached
        
        df = None
        
        # 使用AKShare获取国债收益率
        if AKSHARE_AVAILABLE:
            try:
                # 中国国债收益率曲线
                df = ak.bond_china_yield(start_date="20200101")
                self.api_calls['akshare'] += 1
            except Exception as e:
                print(f"AKShare获取国债收益率失败: {e}")
        
        if df is not None and not df.empty:
            self._save_to_cache(cache_key, df)
        
        return df
    
    def get_yield_curve(self) -> Dict[str, float]:
        """
        获取中国国债收益率曲线（当前值）
        
        Returns:
            Dict with yield curve data: {'1y': 2.5, '2y': 2.6, '5y': 2.8, '10y': 3.0, ...}
        """
        df = self.get_bond_yield()
        if df is None or df.empty:
            return {}
        
        # 提取最新的收益率曲线
        try:
            latest = df.iloc[-1]
            yield_curve = {}
            for col in df.columns:
                if '年' in col or 'y' in col.lower():
                    yield_curve[col] = float(latest[col]) if pd.notna(latest[col]) else None
            return yield_curve
        except Exception as e:
            print(f"解析收益率曲线失败: {e}")
            return {}
    
    # ==================== 货币供应数据 ====================
    
    def get_money_supply(self, start_month: str = None, end_month: str = None) -> pd.DataFrame:
        """
        获取货币供应量数据（M0, M1, M2）
        
        Args:
            start_month: 开始月份
            end_month: 结束月份
            
        Returns:
            DataFrame with money supply data
        """
        cache_key = self._get_cache_key('money_supply', start=start_month, end=end_month)
        cached = self._get_from_cache(cache_key, max_age_hours=168)
        if cached is not None:
            return cached
        
        df = None
        
        # 优先使用Tushare
        if self.pro:
            try:
                df = self.pro.cn_m(start_m=start_month, end_m=end_month)
                self.api_calls['tushare'] += 1
            except Exception as e:
                print(f"Tushare获取货币供应量失败: {e}")
        
        # 备用AKShare
        if df is None or df.empty:
            if AKSHARE_AVAILABLE:
                try:
                    df = ak.macro_china_money_supply()
                    self.api_calls['akshare'] += 1
                except Exception as e:
                    print(f"AKShare获取货币供应量失败: {e}")
        
        if df is not None and not df.empty:
            self._save_to_cache(cache_key, df)
        
        return df
    
    def get_social_financing(self, start_month: str = None, end_month: str = None) -> pd.DataFrame:
        """
        获取社会融资规模数据
        
        Args:
            start_month: 开始月份
            end_month: 结束月份
            
        Returns:
            DataFrame with social financing data
        """
        cache_key = self._get_cache_key('social_financing', start=start_month, end=end_month)
        cached = self._get_from_cache(cache_key, max_age_hours=168)
        if cached is not None:
            return cached
        
        df = None
        
        # 优先使用Tushare
        if self.pro:
            try:
                df = self.pro.sf_month(start_m=start_month, end_m=end_month)
                self.api_calls['tushare'] += 1
            except Exception as e:
                print(f"Tushare获取社会融资规模失败: {e}")
        
        # 备用AKShare
        if df is None or df.empty:
            if AKSHARE_AVAILABLE:
                try:
                    df = ak.macro_china_shrzgm()
                    self.api_calls['akshare'] += 1
                except Exception as e:
                    print(f"AKShare获取社会融资规模失败: {e}")
        
        if df is not None and not df.empty:
            self._save_to_cache(cache_key, df)
        
        return df
    
    # ==================== 市场指数 ====================
    
    def get_index_daily(self, index_code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        获取指数日线行情
        
        Args:
            index_code: 指数代码，如 '000300.SH' 或 'hs300'
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame with index daily data
        """
        # 转换简称为代码
        if index_code.lower() in self.INDEX_CODES:
            index_code = self.INDEX_CODES[index_code.lower()]
        
        cache_key = self._get_cache_key('index_daily', code=index_code, start=start_date, end=end_date)
        cached = self._get_from_cache(cache_key, max_age_hours=12)
        if cached is not None:
            return cached
        
        df = None
        
        # 优先使用Tushare
        if self.pro:
            try:
                df = self.pro.index_daily(ts_code=index_code, start_date=start_date, end_date=end_date)
                self.api_calls['tushare'] += 1
            except Exception as e:
                print(f"Tushare获取指数行情失败: {e}")
        
        # 备用AKShare
        if df is None or df.empty:
            if AKSHARE_AVAILABLE:
                try:
                    # AKShare使用不同的代码格式
                    ak_code = index_code.split('.')[0]
                    df = ak.stock_zh_index_daily(symbol=f"sh{ak_code}" if 'SH' in index_code else f"sz{ak_code}")
                    self.api_calls['akshare'] += 1
                except Exception as e:
                    print(f"AKShare获取指数行情失败: {e}")
        
        if df is not None and not df.empty:
            self._save_to_cache(cache_key, df)
        
        return df
    
    def get_market_indices(self) -> Dict[str, Dict[str, Any]]:
        """
        获取主要市场指数的最新数据
        
        Returns:
            Dict with index data: {'hs300': {'close': 4000, 'change': 0.5, ...}, ...}
        """
        indices = {}
        
        for name, code in self.INDEX_CODES.items():
            try:
                df = self.get_index_daily(code)
                if df is not None and not df.empty:
                    latest = df.iloc[0] if 'trade_date' in df.columns else df.iloc[-1]
                    indices[name] = {
                        'code': code,
                        'close': float(latest.get('close', latest.get('收盘', 0))),
                        'change': float(latest.get('pct_chg', latest.get('涨跌幅', 0))),
                        'volume': float(latest.get('vol', latest.get('成交量', 0))),
                        'date': str(latest.get('trade_date', latest.get('日期', '')))
                    }
            except Exception as e:
                print(f"获取{name}指数失败: {e}")
        
        return indices
    
    # ==================== 宏观数据快照 ====================
    
    def get_macro_snapshot(self) -> Dict[str, Any]:
        """
        获取中国宏观经济数据快照
        
        Returns:
            Dict with macro snapshot data
        """
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'economy': {},
            'monetary_policy': {},
            'money_supply': {},
            'market_indices': {},
            'data_sources': self.api_calls.copy()
        }
        
        # 经济数据
        try:
            gdp_df = self.get_gdp()
            if gdp_df is not None and not gdp_df.empty:
                latest_gdp = gdp_df.iloc[0]
                snapshot['economy']['gdp'] = {
                    'value': float(latest_gdp.get('gdp', latest_gdp.get('国内生产总值', 0))),
                    'yoy': float(latest_gdp.get('gdp_yoy', latest_gdp.get('同比增长', 0))),
                    'quarter': str(latest_gdp.get('quarter', ''))
                }
        except Exception as e:
            print(f"获取GDP快照失败: {e}")
        
        try:
            cpi_df = self.get_cpi()
            if cpi_df is not None and not cpi_df.empty:
                latest_cpi = cpi_df.iloc[0]
                snapshot['economy']['cpi'] = {
                    'value': float(latest_cpi.get('nt_val', latest_cpi.get('全国', 0))),
                    'yoy': float(latest_cpi.get('nt_yoy', latest_cpi.get('同比增长', 0))),
                    'month': str(latest_cpi.get('month', ''))
                }
        except Exception as e:
            print(f"获取CPI快照失败: {e}")
        
        try:
            pmi_df = self.get_pmi()
            if pmi_df is not None and not pmi_df.empty:
                latest_pmi = pmi_df.iloc[0]
                snapshot['economy']['pmi'] = {
                    'manufacturing': float(latest_pmi.get('pmi', latest_pmi.get('制造业', 0))),
                    'month': str(latest_pmi.get('month', ''))
                }
        except Exception as e:
            print(f"获取PMI快照失败: {e}")
        
        # 货币政策
        try:
            lpr_df = self.get_lpr()
            if lpr_df is not None and not lpr_df.empty:
                latest_lpr = lpr_df.iloc[0]
                snapshot['monetary_policy']['lpr'] = {
                    '1y': float(latest_lpr.get('lpr1y', latest_lpr.get('1年', 0))),
                    '5y': float(latest_lpr.get('lpr5y', latest_lpr.get('5年', 0))),
                    'date': str(latest_lpr.get('date', ''))
                }
        except Exception as e:
            print(f"获取LPR快照失败: {e}")
        
        try:
            shibor_df = self.get_shibor()
            if shibor_df is not None and not shibor_df.empty:
                latest_shibor = shibor_df.iloc[0]
                snapshot['monetary_policy']['shibor'] = {
                    'overnight': float(latest_shibor.get('on', latest_shibor.get('隔夜', 0))),
                    '1w': float(latest_shibor.get('1w', latest_shibor.get('1周', 0))),
                    '1m': float(latest_shibor.get('1m', latest_shibor.get('1月', 0))),
                    'date': str(latest_shibor.get('date', ''))
                }
        except Exception as e:
            print(f"获取Shibor快照失败: {e}")
        
        # 货币供应
        try:
            money_df = self.get_money_supply()
            if money_df is not None and not money_df.empty:
                latest_money = money_df.iloc[0]
                snapshot['money_supply'] = {
                    'm2': float(latest_money.get('m2', latest_money.get('M2', 0))),
                    'm2_yoy': float(latest_money.get('m2_yoy', latest_money.get('M2同比', 0))),
                    'm1': float(latest_money.get('m1', latest_money.get('M1', 0))),
                    'm1_yoy': float(latest_money.get('m1_yoy', latest_money.get('M1同比', 0))),
                    'month': str(latest_money.get('month', ''))
                }
        except Exception as e:
            print(f"获取货币供应快照失败: {e}")
        
        # 市场指数
        try:
            snapshot['market_indices'] = self.get_market_indices()
        except Exception as e:
            print(f"获取市场指数快照失败: {e}")
        
        return snapshot
    
    def get_api_stats(self) -> Dict[str, int]:
        """获取API调用统计"""
        return self.api_calls.copy()


# 测试代码
if __name__ == '__main__':
    import os
    
    # 从环境变量获取token
    token = os.environ.get('TUSHARE_TOKEN')
    
    print("=" * 60)
    print("测试 CNMacroDataManager")
    print("=" * 60)
    
    manager = CNMacroDataManager(tushare_token=token)
    
    # 测试GDP
    print("\n1. 测试GDP数据...")
    gdp = manager.get_gdp()
    if gdp is not None and not gdp.empty:
        print(f"   获取GDP数据 {len(gdp)} 条")
        print(f"   最新: {gdp.iloc[0].to_dict()}")
    
    # 测试CPI
    print("\n2. 测试CPI数据...")
    cpi = manager.get_cpi()
    if cpi is not None and not cpi.empty:
        print(f"   获取CPI数据 {len(cpi)} 条")
    
    # 测试PMI
    print("\n3. 测试PMI数据...")
    pmi = manager.get_pmi()
    if pmi is not None and not pmi.empty:
        print(f"   获取PMI数据 {len(pmi)} 条")
    
    # 测试LPR
    print("\n4. 测试LPR数据...")
    lpr = manager.get_lpr()
    if lpr is not None and not lpr.empty:
        print(f"   获取LPR数据 {len(lpr)} 条")
        print(f"   最新: {lpr.iloc[0].to_dict()}")
    
    # 测试Shibor
    print("\n5. 测试Shibor数据...")
    shibor = manager.get_shibor()
    if shibor is not None and not shibor.empty:
        print(f"   获取Shibor数据 {len(shibor)} 条")
    
    # 测试货币供应
    print("\n6. 测试货币供应数据...")
    money = manager.get_money_supply()
    if money is not None and not money.empty:
        print(f"   获取货币供应数据 {len(money)} 条")
    
    # 测试沪深300指数
    print("\n7. 测试沪深300指数...")
    hs300 = manager.get_index_daily('hs300')
    if hs300 is not None and not hs300.empty:
        print(f"   获取沪深300数据 {len(hs300)} 条")
    
    # 测试宏观数据快照
    print("\n8. 测试宏观数据快照...")
    snapshot = manager.get_macro_snapshot()
    print(f"   快照时间: {snapshot['timestamp']}")
    print(f"   经济数据: {snapshot['economy']}")
    print(f"   货币政策: {snapshot['monetary_policy']}")
    
    # API调用统计
    print("\n" + "=" * 60)
    print("API调用统计:")
    print(f"   Tushare: {manager.api_calls['tushare']} 次")
    print(f"   AKShare: {manager.api_calls['akshare']} 次")
    print(f"   缓存命中: {manager.api_calls['cache_hits']} 次")
    print("=" * 60)
