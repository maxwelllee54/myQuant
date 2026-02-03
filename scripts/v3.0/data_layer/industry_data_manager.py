#!/usr/bin/env python3
"""
V3.0 Industry Data Manager
行业深度数据管理器

提供统一接口获取A股和美股的行业分类、成分股、行业指数等数据。

作者: Manus AI
版本: 3.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class IndustryInfo:
    """行业信息结构"""
    code: str
    name: str
    level: int
    parent_code: Optional[str]
    source: str  # "SW" (申万), "GICS", "CITIC" (中信)


@dataclass
class IndustryIndex:
    """行业指数数据结构"""
    code: str
    name: str
    history: pd.DataFrame
    valuation: Optional[Dict[str, float]]


class IndustryDataManager:
    """
    行业深度数据管理器
    
    提供统一接口获取A股和美股的行业分类、成分股、行业指数等数据，
    封装 akshare, tushare, yfinance 的相关接口。
    """
    
    # 申万一级行业代码映射
    SW_INDUSTRY_L1 = {
        "801010": "农林牧渔",
        "801020": "采掘",
        "801030": "化工",
        "801040": "钢铁",
        "801050": "有色金属",
        "801080": "电子",
        "801110": "家用电器",
        "801120": "食品饮料",
        "801130": "纺织服装",
        "801140": "轻工制造",
        "801150": "医药生物",
        "801160": "公用事业",
        "801170": "交通运输",
        "801180": "房地产",
        "801200": "商业贸易",
        "801210": "休闲服务",
        "801230": "综合",
        "801710": "建筑材料",
        "801720": "建筑装饰",
        "801730": "电气设备",
        "801740": "国防军工",
        "801750": "计算机",
        "801760": "传媒",
        "801770": "通信",
        "801780": "银行",
        "801790": "非银金融",
        "801880": "汽车",
        "801890": "机械设备",
    }
    
    # 美股GICS行业分类
    GICS_SECTORS = {
        "10": "Energy",
        "15": "Materials",
        "20": "Industrials",
        "25": "Consumer Discretionary",
        "30": "Consumer Staples",
        "35": "Health Care",
        "40": "Financials",
        "45": "Information Technology",
        "50": "Communication Services",
        "55": "Utilities",
        "60": "Real Estate",
    }
    
    def __init__(
        self,
        tushare_token: Optional[str] = None,
        cache_dir: str = "~/.quant_investor/data/industry",
        verbose: bool = True
    ):
        """
        初始化行业数据管理器
        
        Args:
            tushare_token: Tushare API token
            cache_dir: 缓存目录
            verbose: 是否打印详细日志
        """
        self.tushare_token = tushare_token
        self.cache_dir = cache_dir
        self.verbose = verbose
        
        # 延迟加载数据源
        self._ak = None
        self._ts_pro = None
        self._yf = None
        
        if self.verbose:
            print("[IndustryDataManager] 初始化完成")
    
    @property
    def ak(self):
        """延迟加载akshare"""
        if self._ak is None:
            try:
                import akshare as ak
                self._ak = ak
            except ImportError:
                raise ImportError("请安装akshare: pip install akshare")
        return self._ak
    
    @property
    def ts_pro(self):
        """延迟加载tushare"""
        if self._ts_pro is None:
            try:
                import tushare as ts
                if self.tushare_token:
                    ts.set_token(self.tushare_token)
                self._ts_pro = ts.pro_api()
            except ImportError:
                raise ImportError("请安装tushare: pip install tushare")
        return self._ts_pro
    
    @property
    def yf(self):
        """延迟加载yfinance"""
        if self._yf is None:
            try:
                import yfinance as yf
                self._yf = yf
            except ImportError:
                raise ImportError("请安装yfinance: pip install yfinance")
        return self._yf
    
    # ==================== A股行业分类 ====================
    
    def get_sw_industry_list(self, level: int = 1) -> List[IndustryInfo]:
        """
        获取申万行业分类列表
        
        Args:
            level: 行业级别 (1, 2, 3)
        
        Returns:
            IndustryInfo列表
        """
        industries = []
        
        if level == 1:
            for code, name in self.SW_INDUSTRY_L1.items():
                industries.append(IndustryInfo(
                    code=code,
                    name=name,
                    level=1,
                    parent_code=None,
                    source="SW"
                ))
        else:
            try:
                # 使用akshare获取更详细的行业分类
                df = self.ak.sw_index_spot()
                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        industries.append(IndustryInfo(
                            code=str(row.get('指数代码', '')),
                            name=str(row.get('指数名称', '')),
                            level=level,
                            parent_code=None,
                            source="SW"
                        ))
            except Exception as e:
                if self.verbose:
                    print(f"[IndustryDataManager] 获取申万行业分类失败: {e}")
        
        if self.verbose:
            print(f"[IndustryDataManager] 获取申万{level}级行业分类成功")
            print(f"  行业数量: {len(industries)}")
        
        return industries
    
    def get_industry_constituents(
        self,
        industry_code: str,
        source: str = "SW"
    ) -> Optional[pd.DataFrame]:
        """
        获取行业成分股
        
        Args:
            industry_code: 行业代码
            source: 分类来源 ("SW" 申万, "CITIC" 中信)
        
        Returns:
            DataFrame包含成分股信息
        """
        try:
            if source == "SW":
                # 使用akshare获取申万行业成分股
                df = self.ak.index_stock_cons_csindex(symbol=industry_code)
                
                if df is not None and not df.empty:
                    if self.verbose:
                        print(f"[IndustryDataManager] 获取 {industry_code} 成分股成功")
                        print(f"  成分股数量: {len(df)}")
                    return df
            
            return None
            
        except Exception as e:
            if self.verbose:
                print(f"[IndustryDataManager] 获取 {industry_code} 成分股失败: {e}")
            return None
    
    def get_sw_industry_history(
        self,
        industry_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[IndustryIndex]:
        """
        获取申万行业指数历史行情
        
        Args:
            industry_code: 行业代码 (如 "801010")
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
        
        Returns:
            IndustryIndex对象
        """
        try:
            # 使用akshare获取申万行业指数历史
            df = self.ak.sw_index_daily(symbol=industry_code)
            
            if df is None or df.empty:
                if self.verbose:
                    print(f"[IndustryDataManager] {industry_code} 没有历史数据")
                return None
            
            # 日期过滤
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                if start_date:
                    df = df[df['date'] >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df['date'] <= pd.to_datetime(end_date)]
            
            industry_name = self.SW_INDUSTRY_L1.get(industry_code, industry_code)
            
            index_data = IndustryIndex(
                code=industry_code,
                name=industry_name,
                history=df,
                valuation=None
            )
            
            if self.verbose:
                print(f"[IndustryDataManager] 获取 {industry_name} 历史数据成功")
                print(f"  数据点: {len(df)}")
            
            return index_data
            
        except Exception as e:
            if self.verbose:
                print(f"[IndustryDataManager] 获取 {industry_code} 历史数据失败: {e}")
            return None
    
    def get_industry_pe_pb(self) -> Optional[pd.DataFrame]:
        """
        获取各行业整体估值水平（PE、PB）
        
        Returns:
            DataFrame包含各行业的PE、PB等估值指标
        """
        try:
            # 使用akshare获取行业估值数据
            df = self.ak.sw_index_spot()
            
            if df is None or df.empty:
                if self.verbose:
                    print("[IndustryDataManager] 获取行业估值数据失败")
                return None
            
            if self.verbose:
                print("[IndustryDataManager] 获取行业估值数据成功")
                print(f"  行业数量: {len(df)}")
            
            return df
            
        except Exception as e:
            if self.verbose:
                print(f"[IndustryDataManager] 获取行业估值数据失败: {e}")
            return None
    
    # ==================== 美股行业分类 ====================
    
    def get_us_sector_etfs(self) -> Dict[str, str]:
        """
        获取美股行业ETF代码
        
        Returns:
            行业名称到ETF代码的映射
        """
        return {
            "Technology": "XLK",
            "Health Care": "XLV",
            "Financials": "XLF",
            "Consumer Discretionary": "XLY",
            "Consumer Staples": "XLP",
            "Energy": "XLE",
            "Industrials": "XLI",
            "Materials": "XLB",
            "Utilities": "XLU",
            "Real Estate": "XLRE",
            "Communication Services": "XLC",
        }
    
    def get_us_sector_history(
        self,
        sector: str,
        period: str = "1y"
    ) -> Optional[pd.DataFrame]:
        """
        获取美股行业ETF历史行情
        
        Args:
            sector: 行业名称 (如 "Technology")
            period: 时间周期 (如 "1y", "6mo")
        
        Returns:
            DataFrame包含行业ETF历史行情
        """
        sector_etfs = self.get_us_sector_etfs()
        
        if sector not in sector_etfs:
            if self.verbose:
                print(f"[IndustryDataManager] 未知行业: {sector}")
                print(f"  可用行业: {list(sector_etfs.keys())}")
            return None
        
        etf_symbol = sector_etfs[sector]
        
        try:
            ticker = self.yf.Ticker(etf_symbol)
            history = ticker.history(period=period)
            
            if history is None or history.empty:
                if self.verbose:
                    print(f"[IndustryDataManager] {sector} 没有历史数据")
                return None
            
            if self.verbose:
                print(f"[IndustryDataManager] 获取 {sector} ({etf_symbol}) 历史数据成功")
                print(f"  数据点: {len(history)}")
            
            return history
            
        except Exception as e:
            if self.verbose:
                print(f"[IndustryDataManager] 获取 {sector} 历史数据失败: {e}")
            return None
    
    def get_us_sector_performance(
        self,
        period: str = "1mo"
    ) -> Optional[pd.DataFrame]:
        """
        获取美股各行业表现对比
        
        Args:
            period: 时间周期 (如 "1mo", "3mo", "1y")
        
        Returns:
            DataFrame包含各行业的收益率对比
        """
        sector_etfs = self.get_us_sector_etfs()
        
        performance_data = []
        
        for sector, etf in sector_etfs.items():
            try:
                ticker = self.yf.Ticker(etf)
                history = ticker.history(period=period)
                
                if history is not None and len(history) > 1:
                    start_price = history['Close'].iloc[0]
                    end_price = history['Close'].iloc[-1]
                    returns = (end_price / start_price - 1) * 100
                    
                    performance_data.append({
                        "sector": sector,
                        "etf": etf,
                        "return_pct": returns,
                        "start_price": start_price,
                        "end_price": end_price
                    })
            except Exception as e:
                if self.verbose:
                    print(f"[IndustryDataManager] 获取 {sector} 表现失败: {e}")
        
        if not performance_data:
            return None
        
        df = pd.DataFrame(performance_data)
        df = df.sort_values('return_pct', ascending=False)
        
        if self.verbose:
            print(f"[IndustryDataManager] 获取美股行业表现成功")
            print(f"  行业数量: {len(df)}")
        
        return df
    
    # ==================== 行业轮动分析 ====================
    
    def analyze_industry_rotation(
        self,
        market: str = "CN",
        lookback_periods: List[str] = ["1mo", "3mo", "6mo"]
    ) -> Dict[str, Any]:
        """
        行业轮动分析
        
        通过分析不同时间周期的行业表现，识别行业轮动趋势。
        
        Args:
            market: 市场 ("CN" A股, "US" 美股)
            lookback_periods: 回看周期列表
        
        Returns:
            包含行业轮动分析结果的字典
        """
        if market == "US":
            return self._analyze_us_rotation(lookback_periods)
        else:
            return self._analyze_cn_rotation(lookback_periods)
    
    def _analyze_us_rotation(
        self,
        lookback_periods: List[str]
    ) -> Dict[str, Any]:
        """分析美股行业轮动"""
        results = {
            "market": "US",
            "timestamp": datetime.now().isoformat(),
            "periods": {}
        }
        
        for period in lookback_periods:
            perf = self.get_us_sector_performance(period)
            if perf is not None:
                results["periods"][period] = {
                    "top_3": perf.head(3)[['sector', 'return_pct']].to_dict('records'),
                    "bottom_3": perf.tail(3)[['sector', 'return_pct']].to_dict('records'),
                    "spread": perf['return_pct'].max() - perf['return_pct'].min()
                }
        
        # 识别持续强势和弱势行业
        if len(results["periods"]) >= 2:
            all_tops = []
            all_bottoms = []
            for period_data in results["periods"].values():
                all_tops.extend([x['sector'] for x in period_data['top_3']])
                all_bottoms.extend([x['sector'] for x in period_data['bottom_3']])
            
            from collections import Counter
            top_counter = Counter(all_tops)
            bottom_counter = Counter(all_bottoms)
            
            results["consistently_strong"] = [s for s, c in top_counter.items() if c >= 2]
            results["consistently_weak"] = [s for s, c in bottom_counter.items() if c >= 2]
        
        return results
    
    def _analyze_cn_rotation(
        self,
        lookback_periods: List[str]
    ) -> Dict[str, Any]:
        """分析A股行业轮动"""
        # A股行业轮动分析需要更多数据支持
        # 这里返回基础结构
        return {
            "market": "CN",
            "timestamp": datetime.now().isoformat(),
            "note": "A股行业轮动分析需要Tushare Pro权限"
        }


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("IndustryDataManager 测试")
    print("=" * 60)
    
    manager = IndustryDataManager(verbose=True)
    
    # 测试申万行业分类
    print("\n--- 测试申万行业分类 ---")
    industries = manager.get_sw_industry_list(level=1)
    print(f"申万一级行业数量: {len(industries)}")
    
    # 测试美股行业表现
    print("\n--- 测试美股行业表现 ---")
    us_perf = manager.get_us_sector_performance(period="1mo")
    if us_perf is not None:
        print("\n近1月美股行业表现:")
        print(us_perf[['sector', 'return_pct']].to_string(index=False))
    
    # 测试行业轮动分析
    print("\n--- 测试行业轮动分析 ---")
    rotation = manager.analyze_industry_rotation(market="US")
    if "consistently_strong" in rotation:
        print(f"持续强势行业: {rotation['consistently_strong']}")
    if "consistently_weak" in rotation:
        print(f"持续弱势行业: {rotation['consistently_weak']}")
    
    print("\n测试完成!")
