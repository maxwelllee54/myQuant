#!/usr/bin/env python3
"""
V3.1 Portfolio Optimizer
高级组合优化模块

实现多种现代投资组合优化算法，包括：
- 均值-方差优化 (Mean-Variance Optimization)
- 风险平价 (Risk Parity)
- 最小方差 (Minimum Variance)
- 最大分散化 (Maximum Diversification)
- Black-Litterman模型

版本: 3.1
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from scipy.optimize import minimize, LinearConstraint
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


@dataclass
class OptimizationResult:
    """优化结果数据结构"""
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    method: str
    success: bool
    message: str
    metrics: Dict[str, float]


@dataclass
class PortfolioMetrics:
    """组合指标"""
    expected_return: float
    volatility: float
    sharpe_ratio: float
    max_weight: float
    min_weight: float
    effective_n: float  # 有效资产数量
    herfindahl_index: float  # 集中度指数


class PortfolioOptimizer:
    """
    高级组合优化器
    
    提供多种优化方法，支持各种约束条件。
    """
    
    METHODS = [
        "mean_variance",
        "min_variance",
        "risk_parity",
        "max_diversification",
        "max_sharpe",
        "equal_weight",
        "black_litterman"
    ]
    
    def __init__(
        self,
        risk_free_rate: float = 0.02,
        verbose: bool = True
    ):
        """
        初始化组合优化器
        
        Args:
            risk_free_rate: 无风险利率（年化）
            verbose: 是否打印详细日志
        """
        self.risk_free_rate = risk_free_rate
        self.verbose = verbose
        
        if self.verbose:
            print("[PortfolioOptimizer] 初始化完成")
            print(f"  无风险利率: {risk_free_rate:.2%}")
    
    def _calculate_portfolio_metrics(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray
    ) -> PortfolioMetrics:
        """计算组合指标"""
        port_return = np.dot(weights, expected_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0
        
        # 集中度指标
        herfindahl = np.sum(weights ** 2)
        effective_n = 1 / herfindahl if herfindahl > 0 else len(weights)
        
        return PortfolioMetrics(
            expected_return=port_return,
            volatility=port_vol,
            sharpe_ratio=sharpe,
            max_weight=np.max(weights),
            min_weight=np.min(weights),
            effective_n=effective_n,
            herfindahl_index=herfindahl
        )
    
    def optimize(
        self,
        expected_returns: Union[pd.Series, np.ndarray],
        cov_matrix: Union[pd.DataFrame, np.ndarray],
        method: str = "max_sharpe",
        constraints: Optional[Dict[str, Any]] = None,
        target_return: Optional[float] = None,
        target_volatility: Optional[float] = None
    ) -> OptimizationResult:
        """
        执行组合优化
        
        Args:
            expected_returns: 预期收益率向量
            cov_matrix: 协方差矩阵
            method: 优化方法
            constraints: 约束条件字典
                - min_weight: 最小权重
                - max_weight: 最大权重
                - sector_constraints: 行业约束
            target_return: 目标收益率（用于均值-方差优化）
            target_volatility: 目标波动率
        
        Returns:
            OptimizationResult对象
        """
        # 转换为numpy数组
        if isinstance(expected_returns, pd.Series):
            asset_names = expected_returns.index.tolist()
            expected_returns = expected_returns.values
        else:
            asset_names = [f"Asset_{i}" for i in range(len(expected_returns))]
        
        if isinstance(cov_matrix, pd.DataFrame):
            cov_matrix = cov_matrix.values
        
        n_assets = len(expected_returns)
        
        # 默认约束
        if constraints is None:
            constraints = {}
        
        min_weight = constraints.get("min_weight", 0.0)
        max_weight = constraints.get("max_weight", 1.0)
        
        # 选择优化方法
        if method == "equal_weight":
            result = self._equal_weight(n_assets, asset_names, expected_returns, cov_matrix)
        elif method == "min_variance":
            result = self._min_variance(n_assets, asset_names, expected_returns, cov_matrix, min_weight, max_weight)
        elif method == "max_sharpe":
            result = self._max_sharpe(n_assets, asset_names, expected_returns, cov_matrix, min_weight, max_weight)
        elif method == "risk_parity":
            result = self._risk_parity(n_assets, asset_names, expected_returns, cov_matrix)
        elif method == "max_diversification":
            result = self._max_diversification(n_assets, asset_names, expected_returns, cov_matrix, min_weight, max_weight)
        elif method == "mean_variance":
            result = self._mean_variance(n_assets, asset_names, expected_returns, cov_matrix, min_weight, max_weight, target_return)
        else:
            result = OptimizationResult(
                weights={},
                expected_return=0,
                expected_volatility=0,
                sharpe_ratio=0,
                method=method,
                success=False,
                message=f"未知的优化方法: {method}",
                metrics={}
            )
        
        if self.verbose and result.success:
            print(f"[PortfolioOptimizer] {method} 优化完成")
            print(f"  预期收益: {result.expected_return:.2%}")
            print(f"  预期波动: {result.expected_volatility:.2%}")
            print(f"  夏普比率: {result.sharpe_ratio:.2f}")
        
        return result
    
    def _equal_weight(
        self,
        n_assets: int,
        asset_names: List[str],
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray
    ) -> OptimizationResult:
        """等权重配置"""
        weights = np.ones(n_assets) / n_assets
        metrics = self._calculate_portfolio_metrics(weights, expected_returns, cov_matrix)
        
        return OptimizationResult(
            weights=dict(zip(asset_names, weights)),
            expected_return=metrics.expected_return,
            expected_volatility=metrics.volatility,
            sharpe_ratio=metrics.sharpe_ratio,
            method="equal_weight",
            success=True,
            message="等权重配置",
            metrics=asdict(metrics)
        )
    
    def _min_variance(
        self,
        n_assets: int,
        asset_names: List[str],
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        min_weight: float,
        max_weight: float
    ) -> OptimizationResult:
        """最小方差组合"""
        
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # 约束条件
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # 权重和为1
        ]
        
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        initial_weights = np.ones(n_assets) / n_assets
        
        result = minimize(
            portfolio_volatility,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if result.success:
            weights = result.x
            metrics = self._calculate_portfolio_metrics(weights, expected_returns, cov_matrix)
            
            return OptimizationResult(
                weights=dict(zip(asset_names, weights)),
                expected_return=metrics.expected_return,
                expected_volatility=metrics.volatility,
                sharpe_ratio=metrics.sharpe_ratio,
                method="min_variance",
                success=True,
                message="最小方差优化成功",
                metrics=asdict(metrics)
            )
        else:
            return OptimizationResult(
                weights={},
                expected_return=0,
                expected_volatility=0,
                sharpe_ratio=0,
                method="min_variance",
                success=False,
                message=f"优化失败: {result.message}",
                metrics={}
            )
    
    def _max_sharpe(
        self,
        n_assets: int,
        asset_names: List[str],
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        min_weight: float,
        max_weight: float
    ) -> OptimizationResult:
        """最大夏普比率组合"""
        
        def neg_sharpe(weights):
            port_return = np.dot(weights, expected_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0
        
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        initial_weights = np.ones(n_assets) / n_assets
        
        result = minimize(
            neg_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if result.success:
            weights = result.x
            metrics = self._calculate_portfolio_metrics(weights, expected_returns, cov_matrix)
            
            return OptimizationResult(
                weights=dict(zip(asset_names, weights)),
                expected_return=metrics.expected_return,
                expected_volatility=metrics.volatility,
                sharpe_ratio=metrics.sharpe_ratio,
                method="max_sharpe",
                success=True,
                message="最大夏普比率优化成功",
                metrics=asdict(metrics)
            )
        else:
            return OptimizationResult(
                weights={},
                expected_return=0,
                expected_volatility=0,
                sharpe_ratio=0,
                method="max_sharpe",
                success=False,
                message=f"优化失败: {result.message}",
                metrics={}
            )
    
    def _risk_parity(
        self,
        n_assets: int,
        asset_names: List[str],
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray
    ) -> OptimizationResult:
        """
        风险平价组合
        
        目标：使每个资产对组合风险的贡献相等
        """
        
        def risk_contribution(weights):
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / port_vol
            risk_contrib = weights * marginal_contrib
            return risk_contrib
        
        def risk_parity_objective(weights):
            rc = risk_contribution(weights)
            target_rc = np.sum(rc) / n_assets
            return np.sum((rc - target_rc) ** 2)
        
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        bounds = tuple((0.01, 1.0) for _ in range(n_assets))
        initial_weights = np.ones(n_assets) / n_assets
        
        result = minimize(
            risk_parity_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if result.success:
            weights = result.x
            weights = weights / np.sum(weights)  # 归一化
            metrics = self._calculate_portfolio_metrics(weights, expected_returns, cov_matrix)
            
            # 计算风险贡献
            rc = risk_contribution(weights)
            
            return OptimizationResult(
                weights=dict(zip(asset_names, weights)),
                expected_return=metrics.expected_return,
                expected_volatility=metrics.volatility,
                sharpe_ratio=metrics.sharpe_ratio,
                method="risk_parity",
                success=True,
                message="风险平价优化成功",
                metrics={**asdict(metrics), "risk_contributions": dict(zip(asset_names, rc))}
            )
        else:
            return OptimizationResult(
                weights={},
                expected_return=0,
                expected_volatility=0,
                sharpe_ratio=0,
                method="risk_parity",
                success=False,
                message=f"优化失败: {result.message}",
                metrics={}
            )
    
    def _max_diversification(
        self,
        n_assets: int,
        asset_names: List[str],
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        min_weight: float,
        max_weight: float
    ) -> OptimizationResult:
        """
        最大分散化组合
        
        目标：最大化分散化比率 = 加权平均波动率 / 组合波动率
        """
        asset_vols = np.sqrt(np.diag(cov_matrix))
        
        def neg_diversification_ratio(weights):
            weighted_avg_vol = np.dot(weights, asset_vols)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -weighted_avg_vol / port_vol if port_vol > 0 else 0
        
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        initial_weights = np.ones(n_assets) / n_assets
        
        result = minimize(
            neg_diversification_ratio,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if result.success:
            weights = result.x
            metrics = self._calculate_portfolio_metrics(weights, expected_returns, cov_matrix)
            
            # 计算分散化比率
            weighted_avg_vol = np.dot(weights, asset_vols)
            div_ratio = weighted_avg_vol / metrics.volatility if metrics.volatility > 0 else 1
            
            return OptimizationResult(
                weights=dict(zip(asset_names, weights)),
                expected_return=metrics.expected_return,
                expected_volatility=metrics.volatility,
                sharpe_ratio=metrics.sharpe_ratio,
                method="max_diversification",
                success=True,
                message="最大分散化优化成功",
                metrics={**asdict(metrics), "diversification_ratio": div_ratio}
            )
        else:
            return OptimizationResult(
                weights={},
                expected_return=0,
                expected_volatility=0,
                sharpe_ratio=0,
                method="max_diversification",
                success=False,
                message=f"优化失败: {result.message}",
                metrics={}
            )
    
    def _mean_variance(
        self,
        n_assets: int,
        asset_names: List[str],
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        min_weight: float,
        max_weight: float,
        target_return: Optional[float]
    ) -> OptimizationResult:
        """
        均值-方差优化
        
        如果指定目标收益率，则在该收益率下最小化方差
        否则，最大化效用函数 U = E[R] - λ/2 * Var[R]
        """
        
        if target_return is not None:
            # 在目标收益率下最小化方差
            def portfolio_volatility(weights):
                return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            constraints_list = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.dot(x, expected_returns) - target_return}
            ]
        else:
            # 最大化效用函数（风险厌恶系数λ=2）
            lambda_ = 2
            
            def neg_utility(weights):
                port_return = np.dot(weights, expected_returns)
                port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
                return -(port_return - lambda_ / 2 * port_var)
            
            portfolio_volatility = neg_utility
            constraints_list = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            ]
        
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        initial_weights = np.ones(n_assets) / n_assets
        
        result = minimize(
            portfolio_volatility,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if result.success:
            weights = result.x
            metrics = self._calculate_portfolio_metrics(weights, expected_returns, cov_matrix)
            
            return OptimizationResult(
                weights=dict(zip(asset_names, weights)),
                expected_return=metrics.expected_return,
                expected_volatility=metrics.volatility,
                sharpe_ratio=metrics.sharpe_ratio,
                method="mean_variance",
                success=True,
                message="均值-方差优化成功",
                metrics=asdict(metrics)
            )
        else:
            return OptimizationResult(
                weights={},
                expected_return=0,
                expected_volatility=0,
                sharpe_ratio=0,
                method="mean_variance",
                success=False,
                message=f"优化失败: {result.message}",
                metrics={}
            )
    
    def black_litterman(
        self,
        market_caps: pd.Series,
        cov_matrix: pd.DataFrame,
        views: Dict[str, float],
        view_confidences: Optional[Dict[str, float]] = None,
        tau: float = 0.05,
        risk_aversion: float = 2.5
    ) -> OptimizationResult:
        """
        Black-Litterman模型
        
        结合市场均衡预期和主观观点进行优化。
        
        Args:
            market_caps: 市值权重
            cov_matrix: 协方差矩阵
            views: 主观观点 {资产名: 预期超额收益}
            view_confidences: 观点置信度 {资产名: 置信度0-1}
            tau: 不确定性参数
            risk_aversion: 风险厌恶系数
        
        Returns:
            OptimizationResult对象
        """
        asset_names = market_caps.index.tolist()
        n_assets = len(asset_names)
        
        # 市场权重
        market_weights = market_caps / market_caps.sum()
        
        # 协方差矩阵
        sigma = cov_matrix.values
        
        # 隐含均衡收益率
        pi = risk_aversion * np.dot(sigma, market_weights.values)
        
        # 构建观点矩阵
        n_views = len(views)
        if n_views == 0:
            # 无观点，返回市场均衡组合
            metrics = self._calculate_portfolio_metrics(market_weights.values, pi, sigma)
            return OptimizationResult(
                weights=dict(zip(asset_names, market_weights.values)),
                expected_return=metrics.expected_return,
                expected_volatility=metrics.volatility,
                sharpe_ratio=metrics.sharpe_ratio,
                method="black_litterman",
                success=True,
                message="Black-Litterman优化成功（无观点，返回市场均衡）",
                metrics=asdict(metrics)
            )
        
        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        
        for i, (asset, view) in enumerate(views.items()):
            if asset in asset_names:
                P[i, asset_names.index(asset)] = 1
                Q[i] = view
        
        # 观点不确定性矩阵
        if view_confidences is None:
            view_confidences = {k: 0.5 for k in views.keys()}
        
        omega_diag = []
        for asset in views.keys():
            conf = view_confidences.get(asset, 0.5)
            # 置信度越高，omega越小
            omega_diag.append(tau * (1 - conf) / conf if conf > 0 else 1e6)
        
        omega = np.diag(omega_diag)
        
        # Black-Litterman公式
        tau_sigma = tau * sigma
        
        try:
            # 后验收益率
            M1 = np.linalg.inv(np.linalg.inv(tau_sigma) + P.T @ np.linalg.inv(omega) @ P)
            M2 = np.linalg.inv(tau_sigma) @ pi + P.T @ np.linalg.inv(omega) @ Q
            bl_returns = M1 @ M2
            
            # 后验协方差
            bl_cov = sigma + M1
            
            # 使用后验收益率进行优化
            result = self._max_sharpe(
                n_assets, asset_names, bl_returns, bl_cov, 0.0, 1.0
            )
            
            result.method = "black_litterman"
            result.message = "Black-Litterman优化成功"
            result.metrics["implied_returns"] = dict(zip(asset_names, pi))
            result.metrics["bl_returns"] = dict(zip(asset_names, bl_returns))
            
            return result
            
        except Exception as e:
            return OptimizationResult(
                weights={},
                expected_return=0,
                expected_volatility=0,
                sharpe_ratio=0,
                method="black_litterman",
                success=False,
                message=f"Black-Litterman优化失败: {e}",
                metrics={}
            )
    
    def efficient_frontier(
        self,
        expected_returns: Union[pd.Series, np.ndarray],
        cov_matrix: Union[pd.DataFrame, np.ndarray],
        n_points: int = 50,
        min_weight: float = 0.0,
        max_weight: float = 1.0
    ) -> pd.DataFrame:
        """
        计算有效前沿
        
        Args:
            expected_returns: 预期收益率
            cov_matrix: 协方差矩阵
            n_points: 前沿上的点数
            min_weight: 最小权重
            max_weight: 最大权重
        
        Returns:
            DataFrame包含有效前沿上各点的收益率和波动率
        """
        if isinstance(expected_returns, pd.Series):
            expected_returns = expected_returns.values
        if isinstance(cov_matrix, pd.DataFrame):
            cov_matrix = cov_matrix.values
        
        # 计算最小方差组合和最大收益组合
        min_var_result = self._min_variance(
            len(expected_returns), 
            [f"A{i}" for i in range(len(expected_returns))],
            expected_returns, cov_matrix, min_weight, max_weight
        )
        
        min_return = min_var_result.expected_return
        max_return = np.max(expected_returns)
        
        # 生成目标收益率序列
        target_returns = np.linspace(min_return, max_return, n_points)
        
        frontier_data = []
        
        for target in target_returns:
            result = self._mean_variance(
                len(expected_returns),
                [f"A{i}" for i in range(len(expected_returns))],
                expected_returns, cov_matrix, min_weight, max_weight, target
            )
            
            if result.success:
                frontier_data.append({
                    "return": result.expected_return,
                    "volatility": result.expected_volatility,
                    "sharpe": result.sharpe_ratio
                })
        
        return pd.DataFrame(frontier_data)
    
    def compare_methods(
        self,
        expected_returns: Union[pd.Series, np.ndarray],
        cov_matrix: Union[pd.DataFrame, np.ndarray],
        methods: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        比较不同优化方法的结果
        
        Args:
            expected_returns: 预期收益率
            cov_matrix: 协方差矩阵
            methods: 要比较的方法列表
        
        Returns:
            DataFrame包含各方法的结果对比
        """
        if methods is None:
            methods = ["equal_weight", "min_variance", "max_sharpe", "risk_parity", "max_diversification"]
        
        results = []
        
        for method in methods:
            result = self.optimize(expected_returns, cov_matrix, method=method)
            
            if result.success:
                results.append({
                    "method": method,
                    "expected_return": result.expected_return,
                    "volatility": result.expected_volatility,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_weight": result.metrics.get("max_weight", 0),
                    "effective_n": result.metrics.get("effective_n", 0)
                })
        
        return pd.DataFrame(results)


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("PortfolioOptimizer 测试")
    print("=" * 60)
    
    # 生成模拟数据
    np.random.seed(42)
    
    assets = ["股票A", "股票B", "股票C", "股票D", "股票E"]
    n_assets = len(assets)
    
    # 预期收益率（年化）
    expected_returns = pd.Series([0.12, 0.10, 0.08, 0.15, 0.09], index=assets)
    
    # 生成协方差矩阵
    vols = np.array([0.20, 0.18, 0.15, 0.25, 0.16])
    corr = np.array([
        [1.0, 0.3, 0.2, 0.4, 0.1],
        [0.3, 1.0, 0.5, 0.3, 0.2],
        [0.2, 0.5, 1.0, 0.2, 0.3],
        [0.4, 0.3, 0.2, 1.0, 0.1],
        [0.1, 0.2, 0.3, 0.1, 1.0]
    ])
    cov_matrix = pd.DataFrame(
        np.outer(vols, vols) * corr,
        index=assets,
        columns=assets
    )
    
    # 测试优化器
    optimizer = PortfolioOptimizer(risk_free_rate=0.02, verbose=True)
    
    # 测试各种方法
    print("\n--- 测试各优化方法 ---")
    comparison = optimizer.compare_methods(expected_returns, cov_matrix)
    print("\n优化方法对比:")
    print(comparison.to_string(index=False))
    
    # 测试最大夏普比率
    print("\n--- 最大夏普比率组合详情 ---")
    result = optimizer.optimize(expected_returns, cov_matrix, method="max_sharpe")
    print("权重分配:")
    for asset, weight in result.weights.items():
        print(f"  {asset}: {weight:.2%}")
    
    print("\n测试完成!")
