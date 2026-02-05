#!/usr/bin/env python3
"""
因子挖掘与定量选股模块 (Quant Selector)

负责：
1. 基于股票池数据计算多维度因子
2. 验证因子有效性并筛选最佳因子
3. 使用有效因子进行定量选股
4. 分析用户持仓的因子暴露
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy import stats


@dataclass
class FactorResult:
    """因子计算结果"""
    name: str
    category: str  # 动量、价值、质量、波动率等
    values: pd.Series  # 因子值
    ic: float = 0.0  # 信息系数
    ir: float = 0.0  # 信息比率
    is_effective: bool = False
    description: str = ""


@dataclass
class StockScore:
    """股票综合评分"""
    code: str
    name: str
    total_score: float
    factor_scores: Dict[str, float] = field(default_factory=dict)
    rank: int = 0
    recommendation: str = ""  # 强烈推荐/推荐/中性/不推荐


@dataclass
class SelectionResult:
    """选股结果"""
    recommended_stocks: List[StockScore]
    holdings_analysis: List[StockScore]
    effective_factors: List[FactorResult]
    factor_summary: str


class QuantSelector:
    """
    因子挖掘与定量选股器
    
    整合V3.2-V3.5的因子挖掘能力，提供端到端的定量选股服务。
    """
    
    def __init__(self, stock_universe: Dict, market_data: Dict = None, verbose: bool = True):
        """
        初始化选股器
        
        Args:
            stock_universe: 股票池数据 {code: StockData}
            market_data: 市场级别数据
            verbose: 是否打印详细信息
        """
        self.stock_universe = stock_universe
        self.market_data = market_data or {}
        self.verbose = verbose
        
        # 因子计算结果
        self.factors: List[FactorResult] = []
        self.effective_factors: List[FactorResult] = []
        
        # 准备数据
        self._prepare_data()
    
    def _prepare_data(self):
        """准备因子计算所需的数据"""
        # 构建价格矩阵
        price_dict = {}
        volume_dict = {}
        
        for code, stock in self.stock_universe.items():
            if stock.price_data is not None and len(stock.price_data) > 20:
                price_dict[code] = stock.price_data['Close']
                volume_dict[code] = stock.price_data['Volume']
        
        if price_dict:
            self.price_df = pd.DataFrame(price_dict)
            self.volume_df = pd.DataFrame(volume_dict)
            self.returns_df = self.price_df.pct_change()
        else:
            self.price_df = pd.DataFrame()
            self.volume_df = pd.DataFrame()
            self.returns_df = pd.DataFrame()
        
        if self.verbose:
            print(f"   数据准备完成: {len(self.price_df.columns)} 只股票, {len(self.price_df)} 个交易日")
    
    def run_factor_mining(self) -> List[FactorResult]:
        """
        运行因子挖掘流程
        
        Returns:
            有效因子列表
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🔬 开始因子挖掘")
            print(f"{'='*60}\n")
        
        # 1. 计算各类因子
        self._calculate_momentum_factors()
        self._calculate_value_factors()
        self._calculate_quality_factors()
        self._calculate_volatility_factors()
        self._calculate_volume_factors()
        
        # 2. 验证因子有效性
        self._validate_factors()
        
        # 3. 筛选有效因子
        self.effective_factors = [f for f in self.factors if f.is_effective]
        
        if self.verbose:
            print(f"\n✅ 因子挖掘完成!")
            print(f"   计算因子数: {len(self.factors)}")
            print(f"   有效因子数: {len(self.effective_factors)}")
            if self.effective_factors:
                print(f"\n   有效因子列表:")
                for f in self.effective_factors:
                    print(f"      - {f.name} ({f.category}): IC={f.ic:.4f}, IR={f.ir:.4f}")
        
        return self.effective_factors
    
    def _calculate_momentum_factors(self):
        """计算动量类因子"""
        if len(self.returns_df) < 20:
            return
        
        if self.verbose:
            print("   计算动量因子...")
        
        # 1. 短期动量 (20日)
        mom_20 = self.returns_df.rolling(20).sum().iloc[-1]
        self.factors.append(FactorResult(
            name="Momentum_20D",
            category="动量",
            values=mom_20,
            description="20日累计收益率"
        ))
        
        # 2. 中期动量 (60日)
        if len(self.returns_df) >= 60:
            mom_60 = self.returns_df.rolling(60).sum().iloc[-1]
            self.factors.append(FactorResult(
                name="Momentum_60D",
                category="动量",
                values=mom_60,
                description="60日累计收益率"
            ))
        
        # 3. 动量反转 (5日)
        mom_5 = self.returns_df.rolling(5).sum().iloc[-1]
        self.factors.append(FactorResult(
            name="Reversal_5D",
            category="动量",
            values=-mom_5,  # 反转因子取负
            description="5日短期反转"
        ))
    
    def _calculate_value_factors(self):
        """计算价值类因子"""
        if self.verbose:
            print("   计算价值因子...")
        
        # 使用财务数据中的指标
        sharpe_values = {}
        for code, stock in self.stock_universe.items():
            if stock.financial_data:
                sharpe_values[code] = stock.financial_data.get('sharpe', 0)
        
        if sharpe_values:
            self.factors.append(FactorResult(
                name="Sharpe_Ratio",
                category="价值",
                values=pd.Series(sharpe_values),
                description="夏普比率"
            ))
    
    def _calculate_quality_factors(self):
        """计算质量类因子"""
        if len(self.returns_df) < 20:
            return
        
        if self.verbose:
            print("   计算质量因子...")
        
        # 1. 收益稳定性
        stability = 1 / (self.returns_df.rolling(20).std().iloc[-1] + 0.001)
        self.factors.append(FactorResult(
            name="Return_Stability",
            category="质量",
            values=stability,
            description="收益稳定性（波动率倒数）"
        ))
        
        # 2. 最大回撤
        max_dd = {}
        for code in self.price_df.columns:
            prices = self.price_df[code].dropna()
            if len(prices) > 0:
                peak = prices.expanding().max()
                dd = (prices - peak) / peak
                max_dd[code] = -dd.min()  # 取负使得回撤小的得分高
        
        if max_dd:
            self.factors.append(FactorResult(
                name="Max_Drawdown",
                category="质量",
                values=pd.Series(max_dd),
                description="最大回撤（越小越好）"
            ))
    
    def _calculate_volatility_factors(self):
        """计算波动率类因子"""
        if len(self.returns_df) < 20:
            return
        
        if self.verbose:
            print("   计算波动率因子...")
        
        # 1. 历史波动率
        vol_20 = self.returns_df.rolling(20).std().iloc[-1] * np.sqrt(252)
        self.factors.append(FactorResult(
            name="Volatility_20D",
            category="波动率",
            values=-vol_20,  # 低波动率优先
            description="20日年化波动率（越低越好）"
        ))
        
        # 2. 下行波动率
        neg_returns = self.returns_df.copy()
        neg_returns[neg_returns > 0] = 0
        downside_vol = neg_returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        self.factors.append(FactorResult(
            name="Downside_Vol",
            category="波动率",
            values=-downside_vol,
            description="下行波动率（越低越好）"
        ))
    
    def _calculate_volume_factors(self):
        """计算成交量类因子"""
        if len(self.volume_df) < 20:
            return
        
        if self.verbose:
            print("   计算成交量因子...")
        
        # 1. 成交量变化率
        vol_change = self.volume_df.rolling(5).mean().iloc[-1] / self.volume_df.rolling(20).mean().iloc[-1]
        self.factors.append(FactorResult(
            name="Volume_Change",
            category="成交量",
            values=vol_change,
            description="短期成交量/长期成交量"
        ))
    
    def _validate_factors(self):
        """验证因子有效性"""
        if self.verbose:
            print("\n   验证因子有效性...")
        
        # 计算未来收益（用于IC计算）
        if len(self.returns_df) < 5:
            return
        
        future_returns = self.returns_df.rolling(5).sum().shift(-5).iloc[-6]
        
        for factor in self.factors:
            try:
                # 对齐数据
                common_idx = factor.values.index.intersection(future_returns.index)
                if len(common_idx) < 10:
                    continue
                
                factor_vals = factor.values[common_idx].dropna()
                ret_vals = future_returns[common_idx].dropna()
                
                common_idx2 = factor_vals.index.intersection(ret_vals.index)
                if len(common_idx2) < 10:
                    continue
                
                # 计算IC（秩相关系数）
                ic, _ = stats.spearmanr(factor_vals[common_idx2], ret_vals[common_idx2])
                factor.ic = ic if not np.isnan(ic) else 0
                
                # 简化的IR计算
                factor.ir = abs(factor.ic) * np.sqrt(20)  # 假设20个观测期
                
                # 判断有效性：|IC| > 0.02 且 IR > 0.3
                factor.is_effective = abs(factor.ic) > 0.02 and factor.ir > 0.3
                
            except Exception as e:
                factor.ic = 0
                factor.ir = 0
                factor.is_effective = False
    
    def select_top_stocks(self, top_n: int = 5) -> List[StockScore]:
        """
        使用有效因子选择Top N股票
        
        Args:
            top_n: 选择的股票数量
        
        Returns:
            推荐股票列表
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"📊 定量选股")
            print(f"{'='*60}\n")
        
        # 如果没有有效因子，使用所有因子
        factors_to_use = self.effective_factors if self.effective_factors else self.factors
        
        if not factors_to_use:
            if self.verbose:
                print("   ⚠️ 没有可用因子，无法进行选股")
            return []
        
        # 计算综合得分
        scores = {}
        for code in self.stock_universe.keys():
            factor_scores = {}
            valid_factors = 0
            total_score = 0
            
            for factor in factors_to_use:
                if code in factor.values.index:
                    # 标准化因子值
                    val = factor.values[code]
                    mean = factor.values.mean()
                    std = factor.values.std()
                    if std > 0:
                        z_score = (val - mean) / std
                        # 加权（有效因子权重更高）
                        weight = 2.0 if factor.is_effective else 1.0
                        factor_scores[factor.name] = z_score
                        total_score += z_score * weight
                        valid_factors += weight
            
            if valid_factors > 0:
                scores[code] = StockScore(
                    code=code,
                    name=self.stock_universe[code].name,
                    total_score=total_score / valid_factors,
                    factor_scores=factor_scores
                )
        
        # 排序并选择Top N
        sorted_scores = sorted(scores.values(), key=lambda x: x.total_score, reverse=True)
        
        # 添加排名和推荐等级
        for i, score in enumerate(sorted_scores):
            score.rank = i + 1
            if i < top_n // 2:
                score.recommendation = "强烈推荐"
            elif i < top_n:
                score.recommendation = "推荐"
            elif i < len(sorted_scores) // 2:
                score.recommendation = "中性"
            else:
                score.recommendation = "不推荐"
        
        recommended = sorted_scores[:top_n]
        
        if self.verbose:
            print(f"   使用 {len(factors_to_use)} 个因子进行选股")
            print(f"\n   Top {top_n} 推荐股票:")
            for stock in recommended:
                print(f"      {stock.rank}. {stock.code} ({stock.name})")
                print(f"         综合得分: {stock.total_score:.4f}")
                print(f"         推荐等级: {stock.recommendation}")
        
        return recommended
    
    def analyze_holdings(self, holdings: List[str] = None) -> List[StockScore]:
        """
        分析用户持仓的因子暴露
        
        Args:
            holdings: 用户持仓股票代码列表
        
        Returns:
            持仓分析结果
        """
        if not holdings:
            return []
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"📋 持仓因子分析")
            print(f"{'='*60}\n")
        
        factors_to_use = self.effective_factors if self.effective_factors else self.factors
        
        holdings_analysis = []
        for code in holdings:
            # 尝试匹配股票代码
            matched_code = None
            for universe_code in self.stock_universe.keys():
                if code.upper() in universe_code.upper() or universe_code.upper() in code.upper():
                    matched_code = universe_code
                    break
            
            if matched_code is None:
                if self.verbose:
                    print(f"   ⚠️ 未找到股票: {code}")
                continue
            
            # 计算因子得分
            factor_scores = {}
            total_score = 0
            valid_factors = 0
            
            for factor in factors_to_use:
                if matched_code in factor.values.index:
                    val = factor.values[matched_code]
                    mean = factor.values.mean()
                    std = factor.values.std()
                    if std > 0:
                        z_score = (val - mean) / std
                        factor_scores[factor.name] = z_score
                        total_score += z_score
                        valid_factors += 1
            
            if valid_factors > 0:
                stock_score = StockScore(
                    code=matched_code,
                    name=self.stock_universe[matched_code].name,
                    total_score=total_score / valid_factors,
                    factor_scores=factor_scores
                )
                
                # 计算在全市场的排名
                all_scores = []
                for c in self.stock_universe.keys():
                    c_score = 0
                    c_valid = 0
                    for factor in factors_to_use:
                        if c in factor.values.index:
                            val = factor.values[c]
                            mean = factor.values.mean()
                            std = factor.values.std()
                            if std > 0:
                                c_score += (val - mean) / std
                                c_valid += 1
                    if c_valid > 0:
                        all_scores.append(c_score / c_valid)
                
                all_scores.sort(reverse=True)
                stock_score.rank = all_scores.index(stock_score.total_score) + 1 if stock_score.total_score in all_scores else len(all_scores)
                
                # 推荐等级
                percentile = stock_score.rank / len(all_scores)
                if percentile <= 0.1:
                    stock_score.recommendation = "强烈推荐持有"
                elif percentile <= 0.3:
                    stock_score.recommendation = "推荐持有"
                elif percentile <= 0.7:
                    stock_score.recommendation = "中性持有"
                else:
                    stock_score.recommendation = "建议减持"
                
                holdings_analysis.append(stock_score)
                
                if self.verbose:
                    print(f"   {matched_code} ({self.stock_universe[matched_code].name})")
                    print(f"      综合得分: {stock_score.total_score:.4f}")
                    print(f"      市场排名: {stock_score.rank}/{len(all_scores)}")
                    print(f"      建议: {stock_score.recommendation}")
                    print(f"      因子暴露:")
                    for fname, fscore in factor_scores.items():
                        direction = "↑" if fscore > 0 else "↓"
                        print(f"         - {fname}: {fscore:.2f} {direction}")
                    print()
        
        return holdings_analysis
    
    def get_selection_result(self, holdings: List[str] = None, top_n: int = 5) -> SelectionResult:
        """
        获取完整的选股结果
        
        Args:
            holdings: 用户持仓
            top_n: 推荐股票数量
        
        Returns:
            完整选股结果
        """
        # 运行因子挖掘
        self.run_factor_mining()
        
        # 选股
        recommended = self.select_top_stocks(top_n)
        
        # 分析持仓
        holdings_analysis = self.analyze_holdings(holdings)
        
        # 生成因子摘要
        factor_summary = self._generate_factor_summary()
        
        return SelectionResult(
            recommended_stocks=recommended,
            holdings_analysis=holdings_analysis,
            effective_factors=self.effective_factors,
            factor_summary=factor_summary
        )
    
    def _generate_factor_summary(self) -> str:
        """生成因子分析摘要"""
        lines = ["## 因子分析摘要\n"]
        
        lines.append(f"本次分析共计算了 **{len(self.factors)}** 个因子，其中 **{len(self.effective_factors)}** 个因子通过有效性验证。\n")
        
        if self.effective_factors:
            lines.append("### 有效因子列表\n")
            lines.append("| 因子名称 | 类别 | IC | IR | 说明 |")
            lines.append("|:---|:---|:---|:---|:---|")
            for f in self.effective_factors:
                lines.append(f"| {f.name} | {f.category} | {f.ic:.4f} | {f.ir:.4f} | {f.description} |")
            lines.append("")
        
        # 按类别统计
        categories = {}
        for f in self.factors:
            if f.category not in categories:
                categories[f.category] = {"total": 0, "effective": 0}
            categories[f.category]["total"] += 1
            if f.is_effective:
                categories[f.category]["effective"] += 1
        
        lines.append("### 因子类别统计\n")
        lines.append("| 类别 | 总数 | 有效数 | 有效率 |")
        lines.append("|:---|:---|:---|:---|")
        for cat, stats in categories.items():
            rate = stats["effective"] / stats["total"] * 100 if stats["total"] > 0 else 0
            lines.append(f"| {cat} | {stats['total']} | {stats['effective']} | {rate:.1f}% |")
        
        return "\n".join(lines)


if __name__ == "__main__":
    # 测试
    from data_provider import fetch_market_data
    
    print("=== 测试因子挖掘与选股 ===\n")
    
    # 获取数据
    stocks, market = fetch_market_data("US", lookback_days=90)
    
    # 运行选股
    selector = QuantSelector(stocks, market)
    result = selector.get_selection_result(
        holdings=["AAPL", "MSFT", "GOOGL"],
        top_n=5
    )
    
    print("\n" + result.factor_summary)
