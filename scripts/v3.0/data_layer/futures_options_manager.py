#!/usr/bin/env python3
"""
V3.0 Futures and Options Data Manager
期货期权数据管理器

提供统一接口获取A股和美股的期货及期权数据。

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
class OptionData:
    """期权数据结构"""
    symbol: str
    expiration: str
    calls: pd.DataFrame
    puts: pd.DataFrame
    underlying_price: float
    timestamp: datetime


@dataclass
class FutureData:
    """期货数据结构"""
    symbol: str
    exchange: str
    history: pd.DataFrame
    contract_info: Dict[str, Any]


class FuturesOptionsDataManager:
    """
    期货期权数据管理器
    
    提供统一接口获取A股和美股的期货及期权数据，
    封装 yfinance, akshare, tushare 的相关接口。
    """
    
    def __init__(
        self,
        tushare_token: Optional[str] = None,
        cache_dir: str = "~/.quant_investor/data/derivatives",
        verbose: bool = True
    ):
        """
        初始化期货期权数据管理器
        
        Args:
            tushare_token: Tushare API token
            cache_dir: 缓存目录
            verbose: 是否打印详细日志
        """
        self.tushare_token = tushare_token
        self.cache_dir = cache_dir
        self.verbose = verbose
        
        # 延迟加载数据源
        self._yf = None
        self._ak = None
        self._ts_pro = None
        
        if self.verbose:
            print("[FuturesOptionsDataManager] 初始化完成")
    
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
    
    # ==================== 美股期权 ====================
    
    def get_us_option_chain(
        self,
        symbol: str,
        expiration: Optional[str] = None
    ) -> Optional[OptionData]:
        """
        获取美股期权链
        
        Args:
            symbol: 股票代码 (如 "AAPL")
            expiration: 到期日 (如 "2024-03-15")，None则返回最近到期日
        
        Returns:
            OptionData对象，包含看涨和看跌期权数据
        """
        try:
            ticker = self.yf.Ticker(symbol)
            
            # 获取所有到期日
            expirations = ticker.options
            if not expirations:
                if self.verbose:
                    print(f"[FuturesOptionsDataManager] {symbol} 没有期权数据")
                return None
            
            # 选择到期日
            if expiration is None:
                expiration = expirations[0]  # 最近到期日
            elif expiration not in expirations:
                if self.verbose:
                    print(f"[FuturesOptionsDataManager] 到期日 {expiration} 不可用，使用最近到期日")
                expiration = expirations[0]
            
            # 获取期权链
            opt_chain = ticker.option_chain(expiration)
            
            # 获取标的价格
            info = ticker.info
            underlying_price = info.get('regularMarketPrice', info.get('previousClose', 0))
            
            option_data = OptionData(
                symbol=symbol,
                expiration=expiration,
                calls=opt_chain.calls,
                puts=opt_chain.puts,
                underlying_price=underlying_price,
                timestamp=datetime.now()
            )
            
            if self.verbose:
                print(f"[FuturesOptionsDataManager] 获取 {symbol} 期权链成功")
                print(f"  到期日: {expiration}")
                print(f"  看涨期权: {len(opt_chain.calls)} 个")
                print(f"  看跌期权: {len(opt_chain.puts)} 个")
            
            return option_data
            
        except Exception as e:
            if self.verbose:
                print(f"[FuturesOptionsDataManager] 获取 {symbol} 期权链失败: {e}")
            return None
    
    def get_us_option_expirations(self, symbol: str) -> List[str]:
        """获取美股期权所有到期日"""
        try:
            ticker = self.yf.Ticker(symbol)
            return list(ticker.options)
        except Exception as e:
            if self.verbose:
                print(f"[FuturesOptionsDataManager] 获取 {symbol} 到期日失败: {e}")
            return []
    
    def calculate_put_call_ratio(self, option_data: OptionData) -> Dict[str, float]:
        """
        计算Put-Call Ratio (PCR)
        
        PCR是衡量市场情绪的重要指标：
        - PCR > 1: 看跌期权多于看涨，可能预示市场悲观
        - PCR < 1: 看涨期权多于看跌，可能预示市场乐观
        
        Args:
            option_data: 期权数据
        
        Returns:
            包含多种PCR指标的字典
        """
        calls = option_data.calls
        puts = option_data.puts
        
        # 成交量PCR
        call_volume = calls['volume'].sum() if 'volume' in calls else 0
        put_volume = puts['volume'].sum() if 'volume' in puts else 0
        volume_pcr = put_volume / call_volume if call_volume > 0 else np.nan
        
        # 持仓量PCR
        call_oi = calls['openInterest'].sum() if 'openInterest' in calls else 0
        put_oi = puts['openInterest'].sum() if 'openInterest' in puts else 0
        oi_pcr = put_oi / call_oi if call_oi > 0 else np.nan
        
        return {
            "volume_pcr": volume_pcr,
            "open_interest_pcr": oi_pcr,
            "call_volume": call_volume,
            "put_volume": put_volume,
            "call_open_interest": call_oi,
            "put_open_interest": put_oi,
            "sentiment": "bearish" if volume_pcr > 1 else "bullish" if volume_pcr < 0.7 else "neutral"
        }
    
    def get_implied_volatility_surface(
        self,
        symbol: str,
        num_expirations: int = 5
    ) -> pd.DataFrame:
        """
        获取隐含波动率曲面
        
        Args:
            symbol: 股票代码
            num_expirations: 获取的到期日数量
        
        Returns:
            DataFrame，行为行权价，列为到期日，值为隐含波动率
        """
        expirations = self.get_us_option_expirations(symbol)[:num_expirations]
        
        iv_data = []
        
        for exp in expirations:
            opt_data = self.get_us_option_chain(symbol, exp)
            if opt_data is None:
                continue
            
            # 合并看涨和看跌期权的IV
            for _, row in opt_data.calls.iterrows():
                if 'impliedVolatility' in row and pd.notna(row['impliedVolatility']):
                    iv_data.append({
                        'expiration': exp,
                        'strike': row['strike'],
                        'iv': row['impliedVolatility'],
                        'type': 'call'
                    })
            
            for _, row in opt_data.puts.iterrows():
                if 'impliedVolatility' in row and pd.notna(row['impliedVolatility']):
                    iv_data.append({
                        'expiration': exp,
                        'strike': row['strike'],
                        'iv': row['impliedVolatility'],
                        'type': 'put'
                    })
        
        if not iv_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(iv_data)
        
        # 创建透视表
        pivot = df.pivot_table(
            values='iv',
            index='strike',
            columns='expiration',
            aggfunc='mean'
        )
        
        return pivot
    
    # ==================== A股期权 ====================
    
    def get_cn_etf_option_chain(
        self,
        underlying: str = "510050"
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """
        获取A股ETF期权链（如50ETF期权、300ETF期权）
        
        Args:
            underlying: 标的代码 (如 "510050" 50ETF, "510300" 300ETF)
        
        Returns:
            包含期权数据的字典
        """
        try:
            # 使用akshare获取期权数据
            # 获取期权合约列表
            option_sse = self.ak.option_sse_list_sina(symbol=f"op_{underlying}", exchange="null")
            
            if option_sse is None or option_sse.empty:
                if self.verbose:
                    print(f"[FuturesOptionsDataManager] {underlying} 没有期权数据")
                return None
            
            if self.verbose:
                print(f"[FuturesOptionsDataManager] 获取 {underlying} ETF期权成功")
                print(f"  合约数量: {len(option_sse)}")
            
            return {"contracts": option_sse}
            
        except Exception as e:
            if self.verbose:
                print(f"[FuturesOptionsDataManager] 获取 {underlying} ETF期权失败: {e}")
            return None
    
    # ==================== 期货数据 ====================
    
    def get_cn_future_history(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[FutureData]:
        """
        获取A股期货历史行情
        
        Args:
            symbol: 期货代码 (如 "IF2403" 股指期货, "CU2403" 铜期货)
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
        
        Returns:
            FutureData对象
        """
        try:
            # 使用akshare获取期货数据
            # 获取主力合约历史数据
            df = self.ak.futures_main_sina(symbol=symbol[:2].upper())
            
            if df is None or df.empty:
                if self.verbose:
                    print(f"[FuturesOptionsDataManager] {symbol} 没有期货数据")
                return None
            
            # 日期过滤
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                if start_date:
                    df = df[df['date'] >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df['date'] <= pd.to_datetime(end_date)]
            
            future_data = FutureData(
                symbol=symbol,
                exchange="SHFE/DCE/CZCE",
                history=df,
                contract_info={"symbol": symbol}
            )
            
            if self.verbose:
                print(f"[FuturesOptionsDataManager] 获取 {symbol} 期货数据成功")
                print(f"  数据点: {len(df)}")
            
            return future_data
            
        except Exception as e:
            if self.verbose:
                print(f"[FuturesOptionsDataManager] 获取 {symbol} 期货数据失败: {e}")
            return None
    
    def get_stock_index_future(
        self,
        index_type: str = "IF"
    ) -> Optional[pd.DataFrame]:
        """
        获取股指期货数据
        
        Args:
            index_type: 股指期货类型
                - "IF": 沪深300股指期货
                - "IH": 上证50股指期货
                - "IC": 中证500股指期货
                - "IM": 中证1000股指期货
        
        Returns:
            DataFrame包含股指期货行情
        """
        try:
            df = self.ak.futures_main_sina(symbol=index_type)
            
            if self.verbose:
                print(f"[FuturesOptionsDataManager] 获取 {index_type} 股指期货成功")
            
            return df
            
        except Exception as e:
            if self.verbose:
                print(f"[FuturesOptionsDataManager] 获取 {index_type} 股指期货失败: {e}")
            return None
    
    # ==================== VIX恐慌指数 ====================
    
    def get_vix_history(
        self,
        period: str = "1y"
    ) -> Optional[pd.DataFrame]:
        """
        获取VIX恐慌指数历史数据
        
        VIX是衡量市场恐慌情绪的重要指标：
        - VIX < 15: 市场平静
        - 15 <= VIX < 25: 市场正常波动
        - 25 <= VIX < 35: 市场担忧
        - VIX >= 35: 市场恐慌
        
        Args:
            period: 时间周期 (如 "1y", "6mo", "1mo")
        
        Returns:
            DataFrame包含VIX历史数据
        """
        try:
            vix = self.yf.Ticker("^VIX")
            history = vix.history(period=period)
            
            if history is None or history.empty:
                if self.verbose:
                    print("[FuturesOptionsDataManager] 获取VIX数据失败")
                return None
            
            if self.verbose:
                print(f"[FuturesOptionsDataManager] 获取VIX数据成功")
                print(f"  数据点: {len(history)}")
                print(f"  最新VIX: {history['Close'].iloc[-1]:.2f}")
            
            return history
            
        except Exception as e:
            if self.verbose:
                print(f"[FuturesOptionsDataManager] 获取VIX数据失败: {e}")
            return None
    
    def get_vix_analysis(self) -> Dict[str, Any]:
        """
        获取VIX分析结果
        
        Returns:
            包含VIX分析的字典
        """
        history = self.get_vix_history(period="3mo")
        
        if history is None or history.empty:
            return {"error": "无法获取VIX数据"}
        
        current_vix = history['Close'].iloc[-1]
        avg_vix = history['Close'].mean()
        max_vix = history['Close'].max()
        min_vix = history['Close'].min()
        
        # 判断市场情绪
        if current_vix < 15:
            sentiment = "极度平静"
            risk_level = "低"
        elif current_vix < 20:
            sentiment = "平静"
            risk_level = "较低"
        elif current_vix < 25:
            sentiment = "正常"
            risk_level = "中等"
        elif current_vix < 30:
            sentiment = "担忧"
            risk_level = "较高"
        elif current_vix < 40:
            sentiment = "恐慌"
            risk_level = "高"
        else:
            sentiment = "极度恐慌"
            risk_level = "极高"
        
        return {
            "current_vix": current_vix,
            "avg_vix_3m": avg_vix,
            "max_vix_3m": max_vix,
            "min_vix_3m": min_vix,
            "percentile": (history['Close'] < current_vix).mean() * 100,
            "sentiment": sentiment,
            "risk_level": risk_level,
            "timestamp": datetime.now().isoformat()
        }


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("FuturesOptionsDataManager 测试")
    print("=" * 60)
    
    manager = FuturesOptionsDataManager(verbose=True)
    
    # 测试VIX数据
    print("\n--- 测试VIX数据 ---")
    vix_analysis = manager.get_vix_analysis()
    print(f"当前VIX: {vix_analysis.get('current_vix', 'N/A')}")
    print(f"市场情绪: {vix_analysis.get('sentiment', 'N/A')}")
    
    print("\n测试完成!")
