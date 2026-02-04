#!/usr/bin/env python3
"""
V3.5 深度特征合成引擎
借鉴featuretools的DFS思想，实现递归特征合成

核心功能:
1. 深度特征合成 (Deep Feature Synthesis)
2. 多层递归特征生成
3. 特征复杂度控制
4. 与遗传规划的深度集成

参考: https://github.com/alteryx/featuretools
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import random
import warnings
warnings.filterwarnings('ignore')


class PrimitiveType(Enum):
    """算子类型"""
    TRANSFORM = "transform"  # 单变量变换
    AGGREGATION = "aggregation"  # 聚合操作
    BINARY = "binary"  # 二元操作


@dataclass
class Primitive:
    """算子定义"""
    name: str
    type: PrimitiveType
    func: Callable
    input_types: List[str]  # 输入类型
    output_type: str  # 输出类型
    description: str = ""
    complexity: int = 1  # 复杂度权重


@dataclass
class FeatureNode:
    """特征节点（表达式树）"""
    primitive: Optional[Primitive] = None
    children: List['FeatureNode'] = field(default_factory=list)
    base_feature: str = None  # 基础特征名
    depth: int = 0
    
    def __str__(self):
        if self.base_feature:
            return self.base_feature
        if self.primitive:
            if self.primitive.type == PrimitiveType.BINARY:
                return f"({self.children[0]} {self.primitive.name} {self.children[1]})"
            else:
                child_str = ', '.join(str(c) for c in self.children)
                return f"{self.primitive.name}({child_str})"
        return "?"
    
    def get_complexity(self) -> int:
        """计算特征复杂度"""
        if self.base_feature:
            return 1
        complexity = self.primitive.complexity if self.primitive else 0
        for child in self.children:
            complexity += child.get_complexity()
        return complexity


class PrimitiveLibrary:
    """算子库"""
    
    def __init__(self):
        self.primitives: Dict[str, Primitive] = {}
        self._build_default_primitives()
    
    def _build_default_primitives(self):
        """构建默认算子库"""
        
        # ========== 单变量变换算子 ==========
        self.add_primitive(Primitive(
            name='lag_1', type=PrimitiveType.TRANSFORM,
            func=lambda x: x.shift(1),
            input_types=['numeric'], output_type='numeric',
            description='滞后1期', complexity=1
        ))
        
        self.add_primitive(Primitive(
            name='lag_5', type=PrimitiveType.TRANSFORM,
            func=lambda x: x.shift(5),
            input_types=['numeric'], output_type='numeric',
            description='滞后5期', complexity=1
        ))
        
        self.add_primitive(Primitive(
            name='diff', type=PrimitiveType.TRANSFORM,
            func=lambda x: x.diff(),
            input_types=['numeric'], output_type='numeric',
            description='一阶差分', complexity=1
        ))
        
        self.add_primitive(Primitive(
            name='pct_change', type=PrimitiveType.TRANSFORM,
            func=lambda x: x.pct_change(),
            input_types=['numeric'], output_type='numeric',
            description='百分比变化', complexity=1
        ))
        
        self.add_primitive(Primitive(
            name='log', type=PrimitiveType.TRANSFORM,
            func=lambda x: np.log(np.abs(x) + 1e-8),
            input_types=['numeric'], output_type='numeric',
            description='对数变换', complexity=1
        ))
        
        self.add_primitive(Primitive(
            name='sqrt', type=PrimitiveType.TRANSFORM,
            func=lambda x: np.sqrt(np.abs(x)),
            input_types=['numeric'], output_type='numeric',
            description='平方根', complexity=1
        ))
        
        self.add_primitive(Primitive(
            name='square', type=PrimitiveType.TRANSFORM,
            func=lambda x: x ** 2,
            input_types=['numeric'], output_type='numeric',
            description='平方', complexity=1
        ))
        
        self.add_primitive(Primitive(
            name='abs', type=PrimitiveType.TRANSFORM,
            func=lambda x: np.abs(x),
            input_types=['numeric'], output_type='numeric',
            description='绝对值', complexity=1
        ))
        
        self.add_primitive(Primitive(
            name='sign', type=PrimitiveType.TRANSFORM,
            func=lambda x: np.sign(x),
            input_types=['numeric'], output_type='numeric',
            description='符号函数', complexity=1
        ))
        
        self.add_primitive(Primitive(
            name='rank', type=PrimitiveType.TRANSFORM,
            func=lambda x: x.rank(pct=True),
            input_types=['numeric'], output_type='numeric',
            description='百分位排名', complexity=2
        ))
        
        # ========== 聚合算子 ==========
        for window in [5, 10, 20]:
            self.add_primitive(Primitive(
                name=f'mean_{window}', type=PrimitiveType.AGGREGATION,
                func=lambda x, w=window: x.rolling(w, min_periods=1).mean(),
                input_types=['numeric'], output_type='numeric',
                description=f'{window}日均值', complexity=2
            ))
            
            self.add_primitive(Primitive(
                name=f'std_{window}', type=PrimitiveType.AGGREGATION,
                func=lambda x, w=window: x.rolling(w, min_periods=1).std(),
                input_types=['numeric'], output_type='numeric',
                description=f'{window}日标准差', complexity=2
            ))
            
            self.add_primitive(Primitive(
                name=f'max_{window}', type=PrimitiveType.AGGREGATION,
                func=lambda x, w=window: x.rolling(w, min_periods=1).max(),
                input_types=['numeric'], output_type='numeric',
                description=f'{window}日最大值', complexity=2
            ))
            
            self.add_primitive(Primitive(
                name=f'min_{window}', type=PrimitiveType.AGGREGATION,
                func=lambda x, w=window: x.rolling(w, min_periods=1).min(),
                input_types=['numeric'], output_type='numeric',
                description=f'{window}日最小值', complexity=2
            ))
            
            self.add_primitive(Primitive(
                name=f'sum_{window}', type=PrimitiveType.AGGREGATION,
                func=lambda x, w=window: x.rolling(w, min_periods=1).sum(),
                input_types=['numeric'], output_type='numeric',
                description=f'{window}日累计', complexity=2
            ))
        
        # ========== 二元算子 ==========
        self.add_primitive(Primitive(
            name='+', type=PrimitiveType.BINARY,
            func=lambda x, y: x + y,
            input_types=['numeric', 'numeric'], output_type='numeric',
            description='加法', complexity=1
        ))
        
        self.add_primitive(Primitive(
            name='-', type=PrimitiveType.BINARY,
            func=lambda x, y: x - y,
            input_types=['numeric', 'numeric'], output_type='numeric',
            description='减法', complexity=1
        ))
        
        self.add_primitive(Primitive(
            name='*', type=PrimitiveType.BINARY,
            func=lambda x, y: x * y,
            input_types=['numeric', 'numeric'], output_type='numeric',
            description='乘法', complexity=1
        ))
        
        self.add_primitive(Primitive(
            name='/', type=PrimitiveType.BINARY,
            func=lambda x, y: x / (y + 1e-8),
            input_types=['numeric', 'numeric'], output_type='numeric',
            description='除法', complexity=1
        ))
        
        self.add_primitive(Primitive(
            name='corr_10', type=PrimitiveType.BINARY,
            func=lambda x, y: x.rolling(10, min_periods=1).corr(y),
            input_types=['numeric', 'numeric'], output_type='numeric',
            description='10日相关性', complexity=3
        ))
    
    def add_primitive(self, primitive: Primitive):
        """添加算子"""
        self.primitives[primitive.name] = primitive
    
    def get_primitives_by_type(self, ptype: PrimitiveType) -> List[Primitive]:
        """按类型获取算子"""
        return [p for p in self.primitives.values() if p.type == ptype]


class DeepFeatureSynthesizer:
    """
    深度特征合成器
    
    实现递归特征合成，生成多层复合特征
    """
    
    def __init__(
        self,
        max_depth: int = 3,
        max_complexity: int = 10,
        primitive_library: PrimitiveLibrary = None
    ):
        self.max_depth = max_depth
        self.max_complexity = max_complexity
        self.primitive_library = primitive_library or PrimitiveLibrary()
        self.generated_features: Dict[str, FeatureNode] = {}
    
    def synthesize(
        self,
        base_features: List[str],
        data: pd.DataFrame,
        n_features: int = 100,
        seed: int = None
    ) -> Dict[str, pd.Series]:
        """
        执行深度特征合成
        
        Args:
            base_features: 基础特征列表
            data: 数据DataFrame
            n_features: 要生成的特征数量
            seed: 随机种子
            
        Returns:
            生成的特征字典
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        generated = {}
        attempts = 0
        max_attempts = n_features * 10
        
        while len(generated) < n_features and attempts < max_attempts:
            attempts += 1
            
            # 随机生成特征树
            feature_node = self._generate_feature_tree(base_features, depth=0)
            
            # 检查复杂度
            if feature_node.get_complexity() > self.max_complexity:
                continue
            
            # 生成特征名
            feature_name = str(feature_node)
            
            # 避免重复
            if feature_name in generated:
                continue
            
            # 计算特征值
            try:
                feature_values = self._evaluate_feature(feature_node, data)
                
                # 检查有效性
                if feature_values.isna().all() or feature_values.std() < 1e-8:
                    continue
                
                generated[feature_name] = feature_values
                self.generated_features[feature_name] = feature_node
                
            except Exception:
                continue
        
        return generated
    
    def _generate_feature_tree(
        self,
        base_features: List[str],
        depth: int
    ) -> FeatureNode:
        """递归生成特征树"""
        
        # 达到最大深度或随机终止
        if depth >= self.max_depth or (depth > 0 and random.random() < 0.3):
            # 返回基础特征
            return FeatureNode(base_feature=random.choice(base_features), depth=depth)
        
        # 随机选择算子类型
        ptype = random.choice([
            PrimitiveType.TRANSFORM,
            PrimitiveType.AGGREGATION,
            PrimitiveType.BINARY
        ])
        
        primitives = self.primitive_library.get_primitives_by_type(ptype)
        if not primitives:
            return FeatureNode(base_feature=random.choice(base_features), depth=depth)
        
        primitive = random.choice(primitives)
        
        # 生成子节点
        if primitive.type == PrimitiveType.BINARY:
            children = [
                self._generate_feature_tree(base_features, depth + 1),
                self._generate_feature_tree(base_features, depth + 1)
            ]
        else:
            children = [self._generate_feature_tree(base_features, depth + 1)]
        
        return FeatureNode(primitive=primitive, children=children, depth=depth)
    
    def _evaluate_feature(
        self,
        node: FeatureNode,
        data: pd.DataFrame
    ) -> pd.Series:
        """评估特征节点"""
        
        if node.base_feature:
            return data[node.base_feature]
        
        # 递归评估子节点
        child_values = [self._evaluate_feature(c, data) for c in node.children]
        
        # 应用算子
        if node.primitive.type == PrimitiveType.BINARY:
            return node.primitive.func(child_values[0], child_values[1])
        else:
            return node.primitive.func(child_values[0])
    
    def get_feature_info(self, feature_name: str) -> Optional[Dict]:
        """获取特征信息"""
        node = self.generated_features.get(feature_name)
        if node is None:
            return None
        
        return {
            'name': feature_name,
            'expression': str(node),
            'depth': self._get_max_depth(node),
            'complexity': node.get_complexity()
        }
    
    def _get_max_depth(self, node: FeatureNode) -> int:
        """获取特征树最大深度"""
        if not node.children:
            return 0
        return 1 + max(self._get_max_depth(c) for c in node.children)


class EnhancedGeneticFactorGenerator:
    """
    增强版遗传规划因子生成器
    
    集成深度特征合成，使用高质量因子作为初始种群
    """
    
    def __init__(
        self,
        population_size: int = 50,
        generations: int = 20,
        max_depth: int = 4,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.2,
        elite_ratio: float = 0.1
    ):
        self.population_size = population_size
        self.generations = generations
        self.max_depth = max_depth
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio
        
        self.primitive_library = PrimitiveLibrary()
        self.dfs = DeepFeatureSynthesizer(
            max_depth=max_depth,
            primitive_library=self.primitive_library
        )
    
    def generate(
        self,
        base_features: List[str],
        data: pd.DataFrame,
        target: pd.Series,
        seed_factors: Dict[str, pd.Series] = None,
        verbose: bool = True
    ) -> List[Dict]:
        """
        生成因子
        
        Args:
            base_features: 基础特征列表
            data: 数据DataFrame
            target: 目标变量
            seed_factors: 种子因子（高质量初始因子）
            verbose: 是否打印详细信息
            
        Returns:
            生成的因子列表
        """
        # 1. 初始化种群
        if verbose:
            print("初始化种群...")
        
        population = []
        
        # 使用种子因子
        if seed_factors:
            for name, values in seed_factors.items():
                fitness = self._evaluate_fitness(values, target)
                population.append({
                    'name': name,
                    'values': values,
                    'fitness': fitness,
                    'generation': 0
                })
        
        # 使用DFS生成初始因子
        dfs_features = self.dfs.synthesize(
            base_features, data,
            n_features=self.population_size - len(population)
        )
        
        for name, values in dfs_features.items():
            fitness = self._evaluate_fitness(values, target)
            population.append({
                'name': name,
                'values': values,
                'fitness': fitness,
                'generation': 0
            })
        
        # 2. 进化循环
        best_fitness_history = []
        
        for gen in range(self.generations):
            # 排序
            population.sort(key=lambda x: x['fitness'], reverse=True)
            best_fitness = population[0]['fitness']
            best_fitness_history.append(best_fitness)
            
            if verbose and gen % 5 == 0:
                print(f"Generation {gen}: Best IC = {best_fitness:.4f}")
            
            # 精英保留
            n_elite = max(1, int(self.population_size * self.elite_ratio))
            new_population = population[:n_elite]
            
            # 生成新个体
            while len(new_population) < self.population_size:
                if random.random() < self.crossover_rate:
                    # 交叉
                    parent1 = self._tournament_select(population)
                    parent2 = self._tournament_select(population)
                    child = self._crossover(parent1, parent2, data, target, gen)
                else:
                    # 变异
                    parent = self._tournament_select(population)
                    child = self._mutate(parent, base_features, data, target, gen)
                
                if child:
                    new_population.append(child)
            
            population = new_population
        
        # 3. 返回最优因子
        population.sort(key=lambda x: x['fitness'], reverse=True)
        
        return population[:self.population_size // 2]
    
    def _evaluate_fitness(self, factor: pd.Series, target: pd.Series) -> float:
        """评估因子适应度（IC）"""
        try:
            common_idx = factor.index.intersection(target.index)
            f = factor.loc[common_idx]
            t = target.loc[common_idx]
            valid = ~(f.isna() | t.isna())
            if valid.sum() < 10:
                return 0
            from scipy.stats import spearmanr
            ic, _ = spearmanr(f[valid], t[valid])
            return abs(ic) if not np.isnan(ic) else 0
        except:
            return 0
    
    def _tournament_select(self, population: List[Dict], k: int = 3) -> Dict:
        """锦标赛选择"""
        candidates = random.sample(population, min(k, len(population)))
        return max(candidates, key=lambda x: x['fitness'])
    
    def _crossover(
        self,
        parent1: Dict,
        parent2: Dict,
        data: pd.DataFrame,
        target: pd.Series,
        generation: int
    ) -> Optional[Dict]:
        """交叉操作"""
        try:
            # 简单的值混合
            alpha = random.random()
            child_values = alpha * parent1['values'] + (1 - alpha) * parent2['values']
            
            if child_values.isna().all() or child_values.std() < 1e-8:
                return None
            
            fitness = self._evaluate_fitness(child_values, target)
            
            return {
                'name': f"cross_{generation}_{random.randint(0, 9999)}",
                'values': child_values,
                'fitness': fitness,
                'generation': generation
            }
        except:
            return None
    
    def _mutate(
        self,
        parent: Dict,
        base_features: List[str],
        data: pd.DataFrame,
        target: pd.Series,
        generation: int
    ) -> Optional[Dict]:
        """变异操作"""
        try:
            # 随机选择变换
            transforms = self.primitive_library.get_primitives_by_type(PrimitiveType.TRANSFORM)
            transform = random.choice(transforms)
            
            child_values = transform.func(parent['values'])
            
            if child_values.isna().all() or child_values.std() < 1e-8:
                return None
            
            fitness = self._evaluate_fitness(child_values, target)
            
            return {
                'name': f"{transform.name}({parent['name']})",
                'values': child_values,
                'fitness': fitness,
                'generation': generation
            }
        except:
            return None


def deep_feature_synthesis(
    data: pd.DataFrame,
    base_features: List[str] = None,
    n_features: int = 100,
    max_depth: int = 3
) -> Dict[str, pd.Series]:
    """
    深度特征合成的便捷函数
    
    Args:
        data: 数据DataFrame
        base_features: 基础特征列表
        n_features: 要生成的特征数量
        max_depth: 最大深度
        
    Returns:
        生成的特征字典
    """
    if base_features is None:
        base_features = list(data.columns)
    
    synthesizer = DeepFeatureSynthesizer(max_depth=max_depth)
    return synthesizer.synthesize(base_features, data, n_features)


# 测试代码
if __name__ == "__main__":
    print("=== V3.5 Deep Feature Synthesis Test ===\n")
    
    np.random.seed(42)
    n_days = 200
    
    # 生成测试数据
    dates = pd.date_range('2024-01-01', periods=n_days, freq='B')
    data = pd.DataFrame({
        'close': np.random.randn(n_days).cumsum() + 100,
        'volume': np.random.uniform(1e6, 1e7, n_days),
        'high': np.random.randn(n_days).cumsum() + 101,
        'low': np.random.randn(n_days).cumsum() + 99,
    }, index=dates)
    
    # 生成目标
    target = data['close'].pct_change(5).shift(-5)
    
    # 测试深度特征合成
    print("测试深度特征合成...")
    synthesizer = DeepFeatureSynthesizer(max_depth=3)
    features = synthesizer.synthesize(
        base_features=['close', 'volume', 'high', 'low'],
        data=data,
        n_features=20,
        seed=42
    )
    
    print(f"生成了 {len(features)} 个特征")
    print("\n特征示例:")
    for i, (name, values) in enumerate(list(features.items())[:5]):
        info = synthesizer.get_feature_info(name)
        print(f"  {i+1}. {name[:50]}... (depth={info['depth']}, complexity={info['complexity']})")
    
    # 测试增强版遗传规划
    print("\n测试增强版遗传规划...")
    generator = EnhancedGeneticFactorGenerator(
        population_size=20,
        generations=5,
        max_depth=3
    )
    
    best_factors = generator.generate(
        base_features=['close', 'volume', 'high', 'low'],
        data=data,
        target=target,
        verbose=True
    )
    
    print(f"\n最优因子:")
    for i, factor in enumerate(best_factors[:3]):
        print(f"  {i+1}. {factor['name'][:50]}... IC={factor['fitness']:.4f}")
    
    print("\n=== Test Completed ===")
