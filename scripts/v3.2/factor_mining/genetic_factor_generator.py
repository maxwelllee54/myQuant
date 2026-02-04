#!/usr/bin/env python3
"""
遗传规划因子生成器 (Genetic Programming Factor Generator)
使用遗传算法自动挖掘有效因子
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import time
import json
import os

from expression_engine import (
    FactorExpression, ExpressionGenerator, OperatorLibrary,
    create_operator_library, FeatureNode
)


@dataclass
class FactorCandidate:
    """因子候选"""
    expression: FactorExpression
    fitness: float = 0.0
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ir: float = 0.0
    ic_positive_rate: float = 0.0
    generation: int = 0
    
    def to_dict(self) -> dict:
        return {
            'expression': self.expression.to_string(),
            'fitness': self.fitness,
            'ic_mean': self.ic_mean,
            'ic_std': self.ic_std,
            'ir': self.ir,
            'ic_positive_rate': self.ic_positive_rate,
            'generation': self.generation
        }


@dataclass
class GPConfig:
    """遗传规划配置"""
    population_size: int = 100          # 种群大小
    generations: int = 50               # 迭代代数
    tournament_size: int = 5            # 锦标赛选择大小
    crossover_rate: float = 0.7         # 交叉概率
    mutation_rate: float = 0.2          # 变异概率
    elite_size: int = 5                 # 精英保留数量
    max_depth: int = 6                  # 最大表达式深度
    min_ic: float = 0.02                # 最小IC阈值
    correlation_threshold: float = 0.7  # 相关性阈值
    n_jobs: int = 1                     # 并行任务数
    random_seed: Optional[int] = None   # 随机种子


class GeneticFactorGenerator:
    """遗传规划因子生成器"""
    
    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        target: pd.DataFrame,
        config: GPConfig = None,
        op_lib: OperatorLibrary = None,
        features: List[str] = None
    ):
        """
        初始化遗传规划因子生成器
        
        Args:
            data: 原始数据字典，key为特征名，value为DataFrame (日期x股票)
            target: 目标变量（如未来收益），DataFrame (日期x股票)
            config: 遗传规划配置
            op_lib: 算子库
            features: 可用特征列表
        """
        self.data = data
        self.target = target
        self.config = config or GPConfig()
        self.op_lib = op_lib or create_operator_library()
        self.features = features or list(data.keys())
        
        # 表达式生成器
        self.expr_generator = ExpressionGenerator(
            self.op_lib,
            features=self.features,
            max_depth=self.config.max_depth
        )
        
        # 种群
        self.population: List[FactorCandidate] = []
        
        # 历史最优因子
        self.best_factors: List[FactorCandidate] = []
        
        # 进化历史
        self.history: List[dict] = []
        
        # 设置随机种子
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
    
    def _calculate_ic(self, factor_values: pd.DataFrame) -> Tuple[float, float, float, float]:
        """
        计算因子的IC指标
        
        Returns:
            (ic_mean, ic_std, ir, ic_positive_rate)
        """
        # 确保对齐
        common_dates = factor_values.index.intersection(self.target.index)
        common_stocks = factor_values.columns.intersection(self.target.columns)
        
        if len(common_dates) < 20 or len(common_stocks) < 5:
            return 0.0, 1.0, 0.0, 0.0
        
        factor_aligned = factor_values.loc[common_dates, common_stocks]
        target_aligned = self.target.loc[common_dates, common_stocks]
        
        # 计算每日IC
        ic_series = []
        for date in common_dates:
            f_vals = factor_aligned.loc[date].dropna()
            t_vals = target_aligned.loc[date].dropna()
            
            common_idx = f_vals.index.intersection(t_vals.index)
            if len(common_idx) < 5:
                continue
            
            f = f_vals.loc[common_idx]
            t = t_vals.loc[common_idx]
            
            # Rank IC (Spearman相关系数)
            if f.std() > 1e-10 and t.std() > 1e-10:
                ic = f.rank().corr(t.rank())
                if not np.isnan(ic):
                    ic_series.append(ic)
        
        if len(ic_series) < 10:
            return 0.0, 1.0, 0.0, 0.0
        
        ic_array = np.array(ic_series)
        ic_mean = np.mean(ic_array)
        ic_std = np.std(ic_array)
        ir = ic_mean / ic_std if ic_std > 1e-10 else 0.0
        ic_positive_rate = np.mean(ic_array > 0)
        
        return ic_mean, ic_std, ir, ic_positive_rate
    
    def _evaluate_fitness(self, candidate: FactorCandidate) -> FactorCandidate:
        """评估因子适应度"""
        try:
            # 计算因子值
            factor_values = candidate.expression.evaluate(self.data, self.op_lib)
            
            # 计算IC指标
            ic_mean, ic_std, ir, ic_positive_rate = self._calculate_ic(factor_values)
            
            # 更新候选因子
            candidate.ic_mean = ic_mean
            candidate.ic_std = ic_std
            candidate.ir = ir
            candidate.ic_positive_rate = ic_positive_rate
            
            # 综合适应度：IR为主，IC胜率为辅
            candidate.fitness = ir * 0.7 + ic_positive_rate * 0.3
            
        except Exception as e:
            candidate.fitness = -1.0
        
        return candidate
    
    def _initialize_population(self):
        """初始化种群"""
        self.population = []
        for _ in range(self.config.population_size):
            expr = self.expr_generator.generate_random()
            candidate = FactorCandidate(expression=expr, generation=0)
            self.population.append(candidate)
    
    def _evaluate_population(self):
        """评估整个种群的适应度"""
        for i, candidate in enumerate(self.population):
            self._evaluate_fitness(candidate)
    
    def _tournament_selection(self) -> FactorCandidate:
        """锦标赛选择"""
        tournament = random.sample(self.population, self.config.tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def _create_offspring(self, generation: int) -> List[FactorCandidate]:
        """创建下一代"""
        offspring = []
        
        # 精英保留
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        for i in range(self.config.elite_size):
            elite = FactorCandidate(
                expression=sorted_pop[i].expression.copy(),
                generation=generation
            )
            offspring.append(elite)
        
        # 生成剩余个体
        while len(offspring) < self.config.population_size:
            if random.random() < self.config.crossover_rate:
                # 交叉
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                child_expr = self.expr_generator.crossover(
                    parent1.expression, parent2.expression
                )
            else:
                # 复制
                parent = self._tournament_selection()
                child_expr = parent.expression.copy()
            
            # 变异
            if random.random() < self.config.mutation_rate:
                child_expr = self.expr_generator.mutate(child_expr)
            
            child = FactorCandidate(expression=child_expr, generation=generation)
            offspring.append(child)
        
        return offspring
    
    def _check_correlation(self, candidate: FactorCandidate) -> bool:
        """检查与已有最优因子的相关性"""
        if not self.best_factors:
            return True
        
        try:
            new_values = candidate.expression.evaluate(self.data, self.op_lib)
            
            for best in self.best_factors:
                best_values = best.expression.evaluate(self.data, self.op_lib)
                
                # 计算相关性
                common_dates = new_values.index.intersection(best_values.index)
                if len(common_dates) < 20:
                    continue
                
                new_flat = new_values.loc[common_dates].values.flatten()
                best_flat = best_values.loc[common_dates].values.flatten()
                
                # 去除NaN
                mask = ~(np.isnan(new_flat) | np.isnan(best_flat))
                if mask.sum() < 100:
                    continue
                
                corr = np.corrcoef(new_flat[mask], best_flat[mask])[0, 1]
                
                if abs(corr) > self.config.correlation_threshold:
                    return False
            
            return True
        except:
            return False
    
    def evolve(self, verbose: bool = True) -> List[FactorCandidate]:
        """
        执行遗传进化
        
        Args:
            verbose: 是否打印进度
        
        Returns:
            最优因子列表
        """
        if verbose:
            print("=== 遗传规划因子挖掘 ===")
            print(f"种群大小: {self.config.population_size}")
            print(f"迭代代数: {self.config.generations}")
            print(f"特征数量: {len(self.features)}")
            print()
        
        # 初始化种群
        self._initialize_population()
        
        start_time = time.time()
        
        for gen in range(self.config.generations):
            gen_start = time.time()
            
            # 评估适应度
            self._evaluate_population()
            
            # 统计
            fitness_values = [c.fitness for c in self.population if c.fitness > -0.5]
            if fitness_values:
                best_fitness = max(fitness_values)
                avg_fitness = np.mean(fitness_values)
                best_candidate = max(self.population, key=lambda x: x.fitness)
            else:
                best_fitness = 0
                avg_fitness = 0
                best_candidate = None
            
            # 记录历史
            self.history.append({
                'generation': gen,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'best_expression': best_candidate.expression.to_string() if best_candidate else "",
                'best_ic': best_candidate.ic_mean if best_candidate else 0,
                'best_ir': best_candidate.ir if best_candidate else 0
            })
            
            # 保存最优因子
            if best_candidate and best_candidate.ic_mean > self.config.min_ic:
                if self._check_correlation(best_candidate):
                    self.best_factors.append(FactorCandidate(
                        expression=best_candidate.expression.copy(),
                        fitness=best_candidate.fitness,
                        ic_mean=best_candidate.ic_mean,
                        ic_std=best_candidate.ic_std,
                        ir=best_candidate.ir,
                        ic_positive_rate=best_candidate.ic_positive_rate,
                        generation=gen
                    ))
            
            if verbose:
                gen_time = time.time() - gen_start
                print(f"代 {gen+1:3d}/{self.config.generations} | "
                      f"最优适应度: {best_fitness:.4f} | "
                      f"平均适应度: {avg_fitness:.4f} | "
                      f"最优IC: {best_candidate.ic_mean:.4f} | "
                      f"最优IR: {best_candidate.ir:.4f} | "
                      f"耗时: {gen_time:.2f}s")
            
            # 创建下一代
            if gen < self.config.generations - 1:
                self.population = self._create_offspring(gen + 1)
        
        total_time = time.time() - start_time
        
        if verbose:
            print()
            print(f"=== 进化完成 ===")
            print(f"总耗时: {total_time:.2f}s")
            print(f"发现有效因子: {len(self.best_factors)}个")
        
        # 按IR排序
        self.best_factors.sort(key=lambda x: x.ir, reverse=True)
        
        return self.best_factors
    
    def get_top_factors(self, n: int = 10) -> List[FactorCandidate]:
        """获取前N个最优因子"""
        return self.best_factors[:n]
    
    def save_results(self, filepath: str):
        """保存结果到文件"""
        results = {
            'config': {
                'population_size': self.config.population_size,
                'generations': self.config.generations,
                'max_depth': self.config.max_depth,
                'min_ic': self.config.min_ic,
                'correlation_threshold': self.config.correlation_threshold
            },
            'best_factors': [f.to_dict() for f in self.best_factors],
            'history': self.history
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def load_results(self, filepath: str):
        """从文件加载结果"""
        with open(filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        self.history = results.get('history', [])
        # 注意：因子表达式需要重新解析，这里只加载元数据


def quick_mine_factors(
    data: Dict[str, pd.DataFrame],
    target: pd.DataFrame,
    n_factors: int = 10,
    generations: int = 30,
    population_size: int = 50,
    verbose: bool = True
) -> List[FactorCandidate]:
    """
    快速挖掘因子的便捷函数
    
    Args:
        data: 原始数据字典
        target: 目标变量
        n_factors: 目标因子数量
        generations: 迭代代数
        population_size: 种群大小
        verbose: 是否打印进度
    
    Returns:
        最优因子列表
    """
    config = GPConfig(
        population_size=population_size,
        generations=generations,
        min_ic=0.02
    )
    
    generator = GeneticFactorGenerator(data, target, config)
    generator.evolve(verbose=verbose)
    
    return generator.get_top_factors(n_factors)


if __name__ == "__main__":
    # 测试代码
    print("=== 遗传规划因子生成器测试 ===\n")
    
    # 创建模拟数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')  # 一年交易日
    stocks = [f'STOCK_{i}' for i in range(50)]  # 50只股票
    
    # 模拟价格数据
    data = {}
    base_price = np.random.uniform(50, 200, len(stocks))
    
    prices = np.zeros((len(dates), len(stocks)))
    prices[0] = base_price
    for i in range(1, len(dates)):
        returns = np.random.randn(len(stocks)) * 0.02
        prices[i] = prices[i-1] * (1 + returns)
    
    data['close'] = pd.DataFrame(prices, index=dates, columns=stocks)
    data['open'] = data['close'] * (1 + np.random.randn(*prices.shape) * 0.005)
    data['high'] = data['close'] * (1 + np.abs(np.random.randn(*prices.shape) * 0.01))
    data['low'] = data['close'] * (1 - np.abs(np.random.randn(*prices.shape) * 0.01))
    data['volume'] = pd.DataFrame(
        np.random.uniform(1e6, 1e7, prices.shape),
        index=dates, columns=stocks
    )
    data['returns'] = data['close'].pct_change()
    
    # 目标：未来5日收益
    target = data['close'].pct_change(5).shift(-5)
    
    # 运行因子挖掘
    print("开始因子挖掘...")
    factors = quick_mine_factors(
        data=data,
        target=target,
        n_factors=5,
        generations=10,
        population_size=30,
        verbose=True
    )
    
    print("\n=== 挖掘结果 ===")
    for i, factor in enumerate(factors):
        print(f"\n因子 {i+1}:")
        print(f"  表达式: {factor.expression.to_string()}")
        print(f"  IC均值: {factor.ic_mean:.4f}")
        print(f"  IR: {factor.ir:.4f}")
        print(f"  IC胜率: {factor.ic_positive_rate:.2%}")
        print(f"  发现代数: {factor.generation}")
    
    print("\n=== 测试完成 ===")
