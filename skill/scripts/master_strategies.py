"""
投资大师策略实现模块

整合六位投资大师的智慧，提供多维度的股票筛选和评分系统。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class MasterStrategies:
    """投资大师策略集成类"""
    
    def __init__(self):
        self.cycle_weights = {
            'LATE_BULL': {'graham': 0.30, 'buffett': 0.20, 'lynch': 0.10, 
                         'dalio': 0.10, 'burry': 0.10, 'marks': 0.20},
            'BEAR_CRISIS': {'graham': 0.20, 'buffett': 0.15, 'lynch': 0.05,
                           'dalio': 0.20, 'burry': 0.30, 'marks': 0.10},
            'EARLY_RECOVERY': {'graham': 0.15, 'buffett': 0.25, 'lynch': 0.30,
                              'dalio': 0.15, 'burry': 0.10, 'marks': 0.05},
            'STABLE_GROWTH': {'graham': 0.20, 'buffett': 0.30, 'lynch': 0.25,
                             'dalio': 0.15, 'burry': 0.05, 'marks': 0.05}
        }
    
    def graham_screen(self, stock_data: pd.Series) -> Tuple[bool, float, Dict]:
        """
        本杰明·格雷厄姆的价值投资筛选
        
        Returns:
            (是否通过, 得分0-100, 详细信息)
        """
        criteria = {
            'pe_ratio': stock_data.get('PE', 999) < 15,
            'pb_ratio': stock_data.get('PB', 999) < 1.5,
            'debt_equity': stock_data.get('DE_Ratio', 999) < 1.0,
            'current_ratio': stock_data.get('Current_Ratio', 0) > 1.5,
            'dividend_history': stock_data.get('Dividend_Years', 0) >= 10,
            'margin_of_safety': stock_data.get('Price', 0) / max(stock_data.get('Intrinsic_Value', 1), 1) < 0.67
        }
        
        passed_count = sum(criteria.values())
        score = (passed_count / len(criteria)) * 100
        passed = passed_count >= 4
        
        return passed, score, criteria
    
    def buffett_screen(self, stock_data: pd.Series) -> Tuple[bool, float, Dict]:
        """
        沃伦·巴菲特的优质企业筛选
        
        Returns:
            (是否通过, 得分0-100, 详细信息)
        """
        criteria = {
            'roe_consistency': stock_data.get('ROE_5Y_Avg', 0) > 15 and stock_data.get('ROE_Std', 999) < 5,
            'low_debt': stock_data.get('DE_Ratio', 999) < 0.5,
            'profit_margin_growth': stock_data.get('Profit_Margin_Trend', 0) > 0,
            'fcf_positive': stock_data.get('FCF', 0) > 0 and stock_data.get('FCF_Growth', 0) > 0,
            'moat_score': stock_data.get('Economic_Moat', 0) >= 3
        }
        
        passed_count = sum(criteria.values())
        score = (passed_count / len(criteria)) * 100
        passed = all(criteria.values())
        
        return passed, score, criteria
    
    def lynch_screen(self, stock_data: pd.Series) -> Tuple[bool, float, Dict]:
        """
        彼得·林奇的成长股筛选（PEG策略）
        
        Returns:
            (是否通过, 得分0-100, 详细信息)
        """
        pe = stock_data.get('PE', 999)
        growth = stock_data.get('EPS_Growth', 0)
        peg = pe / max(growth, 0.1)
        
        criteria = {
            'peg_ratio': peg < 1.0,
            'peg_excellent': peg < 0.5,
            'earnings_growth': growth > 20,
            'low_debt': stock_data.get('DE_Ratio', 999) < 0.3,
            'fcf_positive': stock_data.get('FCF', 0) > 0,
            'institutional_ownership': stock_data.get('Inst_Own', 100) < 30
        }
        
        # PEG < 0.5 直接通过
        if peg < 0.5:
            score = 100
            passed = True
        else:
            passed_count = sum(criteria.values())
            score = (passed_count / len(criteria)) * 100
            passed = passed_count >= 4
        
        criteria['peg_value'] = peg
        
        return passed, score, criteria
    
    def burry_contrarian_screen(self, stock_data: pd.Series, market_sentiment: Dict) -> Tuple[bool, float, Dict]:
        """
        Michael Burry的逆向投资筛选
        
        Returns:
            (是否通过, 得分0-100, 详细信息)
        """
        criteria = {
            'deep_value': stock_data.get('PB', 999) < 1.0 and stock_data.get('PE', 999) < 10,
            'market_pessimism': market_sentiment.get('Sector_Sentiment', 0) < -2,
            'strong_fundamentals': stock_data.get('ROE', 0) > 15 and stock_data.get('DE_Ratio', 999) < 0.5,
            'analyst_downgrades': stock_data.get('Analyst_Downgrades_3M', 0) > 3,
            'price_drop': stock_data.get('Price_Change_6M', 0) < -30
        }
        
        # 逆向机会：基本面好但市场极度悲观
        passed = criteria['strong_fundamentals'] and criteria['market_pessimism']
        
        passed_count = sum(criteria.values())
        score = (passed_count / len(criteria)) * 100
        
        return passed, score, criteria
    
    def marks_cycle_position(self, market_data: Dict) -> Tuple[str, float, Dict]:
        """
        霍华德·马克斯的市场周期定位
        
        Returns:
            (周期阶段, 周期得分0-100, 详细指标)
        """
        indicators = {
            'valuation': market_data.get('CAPE', 20) / market_data.get('CAPE_Median', 20),
            'vix': market_data.get('VIX', 15),
            'credit_spread': market_data.get('Credit_Spread', 2),
            'sentiment': market_data.get('Investor_Sentiment', 50)
        }
        
        # 计算周期得分（0-100，50为中性）
        valuation_score = min(indicators['valuation'] * 25, 50)
        vix_score = max(50 - indicators['vix'], 0)
        spread_score = min(indicators['credit_spread'] * 10, 50)
        sentiment_score = indicators['sentiment']
        
        cycle_score = (valuation_score + vix_score + spread_score + sentiment_score) / 4
        
        if cycle_score > 75:
            phase = "LATE_BULL"
        elif cycle_score < 25:
            phase = "BEAR_CRISIS"
        elif cycle_score < 50:
            phase = "EARLY_RECOVERY"
        else:
            phase = "STABLE_GROWTH"
        
        return phase, cycle_score, indicators
    
    def comprehensive_score(self, stock_data: pd.Series, market_data: Dict, 
                          market_sentiment: Dict) -> Dict:
        """
        综合六位大师的智慧，对股票进行多维度评分
        
        Returns:
            包含总分、各维度得分和推荐的字典
        """
        # 获取市场周期
        cycle_phase, cycle_score, cycle_indicators = self.marks_cycle_position(market_data)
        
        # 获取各策略得分
        graham_passed, graham_score, graham_details = self.graham_screen(stock_data)
        buffett_passed, buffett_score, buffett_details = self.buffett_screen(stock_data)
        lynch_passed, lynch_score, lynch_details = self.lynch_screen(stock_data)
        burry_passed, burry_score, burry_details = self.burry_contrarian_screen(stock_data, market_sentiment)
        
        # 根据市场周期调整权重
        weights = self.cycle_weights[cycle_phase]
        
        # 计算加权总分
        total_score = (
            graham_score * weights['graham'] +
            buffett_score * weights['buffett'] +
            lynch_score * weights['lynch'] +
            burry_score * weights['burry'] +
            cycle_score * weights['marks']
        )
        
        # 生成推荐
        if total_score >= 80:
            recommendation = "STRONG_BUY"
        elif total_score >= 60:
            recommendation = "BUY"
        elif total_score >= 40:
            recommendation = "HOLD"
        else:
            recommendation = "AVOID"
        
        return {
            'total_score': total_score,
            'recommendation': recommendation,
            'cycle_phase': cycle_phase,
            'cycle_score': cycle_score,
            'scores': {
                'graham': graham_score,
                'buffett': buffett_score,
                'lynch': lynch_score,
                'burry': burry_score,
                'marks': cycle_score
            },
            'passed': {
                'graham': graham_passed,
                'buffett': buffett_passed,
                'lynch': lynch_passed,
                'burry': burry_passed
            },
            'details': {
                'graham': graham_details,
                'buffett': buffett_details,
                'lynch': lynch_details,
                'burry': burry_details,
                'cycle': cycle_indicators
            },
            'weights': weights
        }
    
    def batch_screen(self, stocks_df: pd.DataFrame, market_data: Dict,
                    market_sentiment: Dict) -> pd.DataFrame:
        """
        批量筛选股票
        
        Args:
            stocks_df: 包含股票数据的DataFrame
            market_data: 市场数据字典
            market_sentiment: 市场情绪数据字典
            
        Returns:
            包含评分和推荐的DataFrame
        """
        results = []
        
        for idx, row in stocks_df.iterrows():
            score_result = self.comprehensive_score(row, market_data, market_sentiment)
            results.append({
                'symbol': row.get('Symbol', idx),
                'total_score': score_result['total_score'],
                'recommendation': score_result['recommendation'],
                'graham_score': score_result['scores']['graham'],
                'buffett_score': score_result['scores']['buffett'],
                'lynch_score': score_result['scores']['lynch'],
                'burry_score': score_result['scores']['burry'],
                'cycle_phase': score_result['cycle_phase']
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('total_score', ascending=False)
        
        return results_df


if __name__ == "__main__":
    # 示例用法
    print("投资大师策略模块")
    print("=" * 50)
    
    # 创建策略实例
    strategies = MasterStrategies()
    
    # 示例股票数据
    stock_example = pd.Series({
        'Symbol': 'AAPL',
        'PE': 25,
        'PB': 30,
        'DE_Ratio': 1.5,
        'Current_Ratio': 1.1,
        'Dividend_Years': 12,
        'Price': 150,
        'Intrinsic_Value': 200,
        'ROE_5Y_Avg': 45,
        'ROE_Std': 3,
        'Profit_Margin_Trend': 0.02,
        'FCF': 90000000000,
        'FCF_Growth': 0.08,
        'Economic_Moat': 4,
        'EPS_Growth': 12,
        'Inst_Own': 60,
        'Analyst_Downgrades_3M': 1,
        'Price_Change_6M': 5
    })
    
    # 示例市场数据
    market_example = {
        'CAPE': 30,
        'CAPE_Median': 20,
        'VIX': 18,
        'Credit_Spread': 2.5,
        'Investor_Sentiment': 65
    }
    
    sentiment_example = {
        'Sector_Sentiment': 0.5
    }
    
    # 综合评分
    result = strategies.comprehensive_score(stock_example, market_example, sentiment_example)
    
    print(f"\n股票: {stock_example['Symbol']}")
    print(f"总分: {result['total_score']:.2f}")
    print(f"推荐: {result['recommendation']}")
    print(f"市场周期: {result['cycle_phase']}")
    print(f"\n各维度得分:")
    for master, score in result['scores'].items():
        print(f"  {master}: {score:.2f}")
