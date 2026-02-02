"""
统一数据API
为上层分析应用提供统一、简洁、语义化的数据访问接口

V2.5.1更新：集成CNMacroDataManager，对标美股V2.6的数据结构
"""

import sys
sys.path.append('/home/ubuntu/skills/quant-investor/scripts/v2.5/data_acquisition')
sys.path.append('/home/ubuntu/skills/quant-investor/scripts/v2.5/cn_macro_data')

import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from data_source_manager import DataSourceManager
from cn_macro_data_manager import CNMacroDataManager


class DataAPI:
    """统一数据API"""
    
    def __init__(self, tushare_token: Optional[str] = None):
        """
        初始化数据API
        
        Args:
            tushare_token: Tushare Pro的token
        """
        self.manager = DataSourceManager(tushare_token=tushare_token)
        self.macro_manager = CNMacroDataManager(tushare_token=tushare_token)
    
    # ==================== 股票数据 ====================
    
    def get_stock_list(self, market: str = 'all') -> pd.DataFrame:
        """
        获取股票列表
        
        Args:
            market: 市场 'all' 全部 'sh' 上交所 'sz' 深交所
            
        Returns:
            DataFrame: 股票列表
        """
        if market == 'all':
            return self.manager.get_stock_list()
        elif market == 'sh':
            return self.manager.get_stock_list(exchange='SSE')
        elif market == 'sz':
            return self.manager.get_stock_list(exchange='SZSE')
        else:
            raise ValueError(f"不支持的市场类型: {market}")
    
    def get_stock_price(self,
                       stock_code: str,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       adjust: str = 'qfq',
                       days: int = 250) -> pd.DataFrame:
        """
        获取股票价格数据
        
        Args:
            stock_code: 股票代码（如 600519.SH）
            start_date: 开始日期 YYYYMMDD，如果不提供则自动计算
            end_date: 结束日期 YYYYMMDD，如果不提供则使用今天
            adjust: 复权类型 'qfq' 前复权 'hfq' 后复权 '' 不复权
            days: 如果不提供start_date，则获取最近N天的数据
            
        Returns:
            DataFrame: 价格数据
        """
        if not end_date:
            end_date = datetime.now().strftime('%Y%m%d')
        
        if not start_date:
            start_dt = datetime.now() - timedelta(days=days)
            start_date = start_dt.strftime('%Y%m%d')
        
        return self.manager.get_daily_price(stock_code, start_date, end_date, adjust)
    
    def get_stock_financial(self,
                           stock_code: str,
                           period: Optional[str] = None) -> pd.DataFrame:
        """
        获取股票财务指标
        
        Args:
            stock_code: 股票代码（如 600519.SH）
            period: 报告期 YYYYMMDD，如果不提供则获取最新的
            
        Returns:
            DataFrame: 财务指标数据
        """
        if not period:
            # 获取最近的年报期（去年12月31日）
            last_year = datetime.now().year - 1
            period = f"{last_year}1231"
        
        return self.manager.get_financial_indicator(stock_code, period)
    
    def get_stock_fund_flow(self,
                           stock_code: str,
                           days: int = 30) -> pd.DataFrame:
        """
        获取股票资金流向
        
        Args:
            stock_code: 股票代码（如 600519.SH）
            days: 获取最近N天的数据
            
        Returns:
            DataFrame: 资金流向数据
        """
        end_date = datetime.now().strftime('%Y%m%d')
        start_dt = datetime.now() - timedelta(days=days)
        start_date = start_dt.strftime('%Y%m%d')
        
        return self.manager.get_fund_flow(stock_code, start_date, end_date)
    
    # ==================== 宏观经济数据 (V2.5.1新增) ====================
    
    def get_gdp(self) -> pd.DataFrame:
        """
        获取GDP数据
        
        Returns:
            DataFrame: GDP数据
        """
        return self.macro_manager.get_gdp()
    
    def get_cpi(self) -> pd.DataFrame:
        """
        获取CPI数据
        
        Returns:
            DataFrame: CPI数据
        """
        return self.macro_manager.get_cpi()
    
    def get_ppi(self) -> pd.DataFrame:
        """
        获取PPI数据
        
        Returns:
            DataFrame: PPI数据
        """
        return self.macro_manager.get_ppi()
    
    def get_pmi(self) -> pd.DataFrame:
        """
        获取PMI数据（制造业采购经理指数）
        
        Returns:
            DataFrame: PMI数据
        """
        return self.macro_manager.get_pmi()
    
    # ==================== 货币政策数据 (V2.5.1新增) ====================
    
    def get_lpr(self) -> pd.DataFrame:
        """
        获取LPR贷款基础利率
        
        Returns:
            DataFrame: LPR数据（1年期和5年期）
        """
        return self.macro_manager.get_lpr()
    
    def get_shibor(self) -> pd.DataFrame:
        """
        获取Shibor上海银行间同业拆放利率
        
        Returns:
            DataFrame: Shibor数据
        """
        return self.macro_manager.get_shibor()
    
    def get_bond_yield(self, maturity: str = '10y') -> pd.DataFrame:
        """
        获取中国国债收益率
        
        Args:
            maturity: 期限，如 '1y', '2y', '5y', '10y', '30y'
            
        Returns:
            DataFrame: 国债收益率数据
        """
        return self.macro_manager.get_bond_yield(maturity)
    
    def get_yield_curve(self) -> Dict[str, float]:
        """
        获取中国国债收益率曲线（当前值）
        
        Returns:
            Dict: 收益率曲线数据
        """
        return self.macro_manager.get_yield_curve()
    
    # ==================== 货币供应数据 (V2.5.1新增) ====================
    
    def get_money_supply(self) -> pd.DataFrame:
        """
        获取货币供应量数据（M0, M1, M2）
        
        Returns:
            DataFrame: 货币供应量数据
        """
        return self.macro_manager.get_money_supply()
    
    def get_social_financing(self) -> pd.DataFrame:
        """
        获取社会融资规模数据
        
        Returns:
            DataFrame: 社会融资规模数据
        """
        return self.macro_manager.get_social_financing()
    
    # ==================== 市场指数数据 (V2.5.1新增) ====================
    
    def get_index_daily(self, index_code: str, days: int = 250) -> pd.DataFrame:
        """
        获取指数日线行情
        
        Args:
            index_code: 指数代码，如 '000300.SH' 或 'hs300'
            days: 获取最近N天的数据
            
        Returns:
            DataFrame: 指数日线数据
        """
        end_date = datetime.now().strftime('%Y%m%d')
        start_dt = datetime.now() - timedelta(days=days)
        start_date = start_dt.strftime('%Y%m%d')
        
        return self.macro_manager.get_index_daily(index_code, start_date, end_date)
    
    def get_market_indices(self) -> Dict[str, Dict[str, Any]]:
        """
        获取主要市场指数的最新数据
        
        Returns:
            Dict: 市场指数数据
        """
        return self.macro_manager.get_market_indices()
    
    # ==================== 宏观数据快照 (V2.5.1新增) ====================
    
    def get_macro_snapshot(self) -> Dict[str, Any]:
        """
        获取中国宏观经济数据快照
        
        Returns:
            Dict: 宏观数据快照，包含经济数据、货币政策、货币供应、市场指数
        """
        return self.macro_manager.get_macro_snapshot()
    
    # ==================== 分析辅助方法 ====================
    
    def get_latest_trading_day(self) -> str:
        """
        获取最新交易日
        
        Returns:
            str: 最新交易日 YYYYMMDD
        """
        # 简单实现：返回今天或昨天（如果今天是周末）
        today = datetime.now()
        if today.weekday() == 5:  # 周六
            today = today - timedelta(days=1)
        elif today.weekday() == 6:  # 周日
            today = today - timedelta(days=2)
        
        return today.strftime('%Y%m%d')
    
    def normalize_stock_code(self, code: str) -> str:
        """
        标准化股票代码为Tushare格式
        
        Args:
            code: 股票代码（可能是 600519 或 600519.SH）
            
        Returns:
            str: 标准化后的代码（600519.SH）
        """
        if '.' in code:
            return code
        
        # 根据代码前缀判断市场
        if code.startswith('6'):
            return f"{code}.SH"
        elif code.startswith(('0', '3')):
            return f"{code}.SZ"
        elif code.startswith('8') or code.startswith('4'):
            return f"{code}.BJ"
        else:
            raise ValueError(f"无法识别的股票代码: {code}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取数据API统计信息"""
        stock_stats = self.manager.get_stats()
        macro_stats = self.macro_manager.get_api_stats()
        
        return {
            'stock_data': stock_stats,
            'macro_data': macro_stats
        }


if __name__ == '__main__':
    # 测试代码
    import os
    
    token = os.getenv('TUSHARE_TOKEN')
    api = DataAPI(tushare_token=token)
    
    print("=" * 60)
    print("统一数据API测试 (V2.5.1)")
    print("=" * 60)
    
    # 测试获取股票列表
    print("\n1. 测试获取上交所股票列表...")
    stock_list = api.get_stock_list(market='sh')
    print(f"✅ 获取 {len(stock_list)} 只股票")
    
    # 测试获取股票价格
    print("\n2. 测试获取股票价格（最近30天）...")
    price_data = api.get_stock_price('600519.SH', days=30, adjust='qfq')
    print(f"✅ 获取 {len(price_data)} 条价格数据")
    
    # 测试获取GDP数据
    print("\n3. 测试获取GDP数据...")
    gdp_data = api.get_gdp()
    print(f"✅ 获取 {len(gdp_data)} 条GDP数据")
    
    # 测试获取CPI数据
    print("\n4. 测试获取CPI数据...")
    cpi_data = api.get_cpi()
    print(f"✅ 获取 {len(cpi_data)} 条CPI数据")
    
    # 测试获取PMI数据
    print("\n5. 测试获取PMI数据...")
    pmi_data = api.get_pmi()
    print(f"✅ 获取 {len(pmi_data)} 条PMI数据")
    
    # 测试获取LPR数据
    print("\n6. 测试获取LPR数据...")
    lpr_data = api.get_lpr()
    print(f"✅ 获取 {len(lpr_data)} 条LPR数据")
    
    # 测试获取Shibor数据
    print("\n7. 测试获取Shibor数据...")
    shibor_data = api.get_shibor()
    print(f"✅ 获取 {len(shibor_data)} 条Shibor数据")
    
    # 测试获取货币供应数据
    print("\n8. 测试获取货币供应数据...")
    money_data = api.get_money_supply()
    print(f"✅ 获取 {len(money_data)} 条货币供应数据")
    
    # 测试获取沪深300指数
    print("\n9. 测试获取沪深300指数...")
    hs300_data = api.get_index_daily('hs300', days=30)
    print(f"✅ 获取 {len(hs300_data)} 条沪深300数据")
    
    # 测试获取宏观数据快照
    print("\n10. 测试获取宏观数据快照...")
    snapshot = api.get_macro_snapshot()
    print(f"✅ 快照时间: {snapshot['timestamp']}")
    print(f"   经济数据: {snapshot['economy']}")
    print(f"   货币政策: {snapshot['monetary_policy']}")
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("数据API统计信息:")
    print("=" * 60)
    stats = api.get_stats()
    print(f"股票数据统计: {stats['stock_data']}")
    print(f"宏观数据统计: {stats['macro_data']}")
