#!/usr/bin/env python3
"""
因子表达式引擎 (Expression Engine)
实现因子表达式的解析、计算和生成
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Callable, Any
from dataclasses import dataclass
from enum import Enum
import random
import copy


class OperatorType(Enum):
    """算子类型"""
    ARITHMETIC = "arithmetic"      # 算术运算
    TIMESERIES = "timeseries"      # 时序运算
    CROSSSECTION = "crosssection"  # 横截面运算
    COMPARISON = "comparison"      # 比较运算


@dataclass
class Operator:
    """算子定义"""
    name: str
    arity: int  # 参数数量
    op_type: OperatorType
    func: Callable
    description: str


class OperatorLibrary:
    """算子库：包含所有可用的因子计算算子"""
    
    def __init__(self):
        self.operators: Dict[str, Operator] = {}
        self._register_default_operators()
    
    def _register_default_operators(self):
        """注册默认算子"""
        
        # ========== 算术运算 ==========
        self.register(Operator(
            name="add", arity=2, op_type=OperatorType.ARITHMETIC,
            func=lambda x, y: x + y,
            description="加法: x + y"
        ))
        
        self.register(Operator(
            name="sub", arity=2, op_type=OperatorType.ARITHMETIC,
            func=lambda x, y: x - y,
            description="减法: x - y"
        ))
        
        self.register(Operator(
            name="mul", arity=2, op_type=OperatorType.ARITHMETIC,
            func=lambda x, y: x * y,
            description="乘法: x * y"
        ))
        
        self.register(Operator(
            name="div", arity=2, op_type=OperatorType.ARITHMETIC,
            func=lambda x, y: np.where(y != 0, x / y, 0),
            description="除法: x / y (除零保护)"
        ))
        
        self.register(Operator(
            name="abs", arity=1, op_type=OperatorType.ARITHMETIC,
            func=lambda x: np.abs(x),
            description="绝对值: |x|"
        ))
        
        self.register(Operator(
            name="neg", arity=1, op_type=OperatorType.ARITHMETIC,
            func=lambda x: -x,
            description="取负: -x"
        ))
        
        self.register(Operator(
            name="log", arity=1, op_type=OperatorType.ARITHMETIC,
            func=lambda x: np.log(np.maximum(x, 1e-10)),
            description="对数: log(x)"
        ))
        
        self.register(Operator(
            name="sign", arity=1, op_type=OperatorType.ARITHMETIC,
            func=lambda x: np.sign(x),
            description="符号函数: sign(x)"
        ))
        
        self.register(Operator(
            name="power", arity=2, op_type=OperatorType.ARITHMETIC,
            func=lambda x, y: np.power(np.abs(x) + 1e-10, y),
            description="幂运算: x^y"
        ))
        
        # ========== 时序运算 ==========
        self.register(Operator(
            name="delay", arity=2, op_type=OperatorType.TIMESERIES,
            func=self._ts_delay,
            description="延迟: delay(x, d) 返回d天前的值"
        ))
        
        self.register(Operator(
            name="delta", arity=2, op_type=OperatorType.TIMESERIES,
            func=self._ts_delta,
            description="差分: delta(x, d) = x - delay(x, d)"
        ))
        
        self.register(Operator(
            name="ts_mean", arity=2, op_type=OperatorType.TIMESERIES,
            func=self._ts_mean,
            description="时序均值: ts_mean(x, d) 过去d天均值"
        ))
        
        self.register(Operator(
            name="ts_std", arity=2, op_type=OperatorType.TIMESERIES,
            func=self._ts_std,
            description="时序标准差: ts_std(x, d) 过去d天标准差"
        ))
        
        self.register(Operator(
            name="ts_max", arity=2, op_type=OperatorType.TIMESERIES,
            func=self._ts_max,
            description="时序最大值: ts_max(x, d) 过去d天最大值"
        ))
        
        self.register(Operator(
            name="ts_min", arity=2, op_type=OperatorType.TIMESERIES,
            func=self._ts_min,
            description="时序最小值: ts_min(x, d) 过去d天最小值"
        ))
        
        self.register(Operator(
            name="ts_rank", arity=2, op_type=OperatorType.TIMESERIES,
            func=self._ts_rank,
            description="时序排名: ts_rank(x, d) 当前值在过去d天的排名"
        ))
        
        self.register(Operator(
            name="ts_sum", arity=2, op_type=OperatorType.TIMESERIES,
            func=self._ts_sum,
            description="时序求和: ts_sum(x, d) 过去d天求和"
        ))
        
        self.register(Operator(
            name="ts_corr", arity=3, op_type=OperatorType.TIMESERIES,
            func=self._ts_corr,
            description="时序相关: ts_corr(x, y, d) 过去d天相关系数"
        ))
        
        # ========== 横截面运算 ==========
        self.register(Operator(
            name="rank", arity=1, op_type=OperatorType.CROSSSECTION,
            func=self._cs_rank,
            description="横截面排名: rank(x) 当日所有股票的排名"
        ))
        
        self.register(Operator(
            name="scale", arity=1, op_type=OperatorType.CROSSSECTION,
            func=self._cs_scale,
            description="横截面标准化: scale(x) 均值0标准差1"
        ))
        
        self.register(Operator(
            name="zscore", arity=1, op_type=OperatorType.CROSSSECTION,
            func=self._cs_zscore,
            description="横截面Z分数: zscore(x)"
        ))
        
        # ========== 比较运算 ==========
        self.register(Operator(
            name="greater", arity=2, op_type=OperatorType.COMPARISON,
            func=lambda x, y: (x > y).astype(float),
            description="大于: greater(x, y) = 1 if x > y else 0"
        ))
        
        self.register(Operator(
            name="less", arity=2, op_type=OperatorType.COMPARISON,
            func=lambda x, y: (x < y).astype(float),
            description="小于: less(x, y) = 1 if x < y else 0"
        ))
        
        self.register(Operator(
            name="if_else", arity=3, op_type=OperatorType.COMPARISON,
            func=lambda cond, x, y: np.where(cond > 0, x, y),
            description="条件: if_else(cond, x, y) = x if cond > 0 else y"
        ))
    
    # ========== 时序运算实现 ==========
    
    @staticmethod
    def _ts_delay(x: pd.DataFrame, d: int) -> pd.DataFrame:
        """延迟d天"""
        d = max(1, int(abs(d) % 20) + 1)  # 限制在1-20天
        return x.shift(d)
    
    @staticmethod
    def _ts_delta(x: pd.DataFrame, d: int) -> pd.DataFrame:
        """差分"""
        d = max(1, int(abs(d) % 20) + 1)
        return x - x.shift(d)
    
    @staticmethod
    def _ts_mean(x: pd.DataFrame, d: int) -> pd.DataFrame:
        """时序均值"""
        d = max(2, int(abs(d) % 60) + 2)  # 限制在2-60天
        return x.rolling(window=d, min_periods=1).mean()
    
    @staticmethod
    def _ts_std(x: pd.DataFrame, d: int) -> pd.DataFrame:
        """时序标准差"""
        d = max(2, int(abs(d) % 60) + 2)
        return x.rolling(window=d, min_periods=2).std()
    
    @staticmethod
    def _ts_max(x: pd.DataFrame, d: int) -> pd.DataFrame:
        """时序最大值"""
        d = max(2, int(abs(d) % 60) + 2)
        return x.rolling(window=d, min_periods=1).max()
    
    @staticmethod
    def _ts_min(x: pd.DataFrame, d: int) -> pd.DataFrame:
        """时序最小值"""
        d = max(2, int(abs(d) % 60) + 2)
        return x.rolling(window=d, min_periods=1).min()
    
    @staticmethod
    def _ts_rank(x: pd.DataFrame, d: int) -> pd.DataFrame:
        """时序排名"""
        d = max(2, int(abs(d) % 60) + 2)
        def rank_pct(arr):
            if len(arr) < 2:
                return 0.5
            return (arr.argsort().argsort()[-1] + 1) / len(arr)
        return x.rolling(window=d, min_periods=2).apply(rank_pct, raw=True)
    
    @staticmethod
    def _ts_sum(x: pd.DataFrame, d: int) -> pd.DataFrame:
        """时序求和"""
        d = max(2, int(abs(d) % 60) + 2)
        return x.rolling(window=d, min_periods=1).sum()
    
    @staticmethod
    def _ts_corr(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
        """时序相关系数"""
        d = max(5, int(abs(d) % 60) + 5)
        return x.rolling(window=d, min_periods=5).corr(y)
    
    # ========== 横截面运算实现 ==========
    
    @staticmethod
    def _cs_rank(x: pd.DataFrame) -> pd.DataFrame:
        """横截面排名（百分位）"""
        return x.rank(axis=1, pct=True)
    
    @staticmethod
    def _cs_scale(x: pd.DataFrame) -> pd.DataFrame:
        """横截面标准化到[-1, 1]"""
        x_min = x.min(axis=1)
        x_max = x.max(axis=1)
        range_val = x_max - x_min
        range_val = range_val.replace(0, 1)
        return ((x.T - x_min) / range_val * 2 - 1).T
    
    @staticmethod
    def _cs_zscore(x: pd.DataFrame) -> pd.DataFrame:
        """横截面Z分数"""
        mean = x.mean(axis=1)
        std = x.std(axis=1).replace(0, 1)
        return ((x.T - mean) / std).T
    
    def register(self, operator: Operator):
        """注册新算子"""
        self.operators[operator.name] = operator
    
    def get(self, name: str) -> Optional[Operator]:
        """获取算子"""
        return self.operators.get(name)
    
    def list_operators(self, op_type: Optional[OperatorType] = None) -> List[str]:
        """列出所有算子"""
        if op_type is None:
            return list(self.operators.keys())
        return [name for name, op in self.operators.items() if op.op_type == op_type]


class ExpressionNode(ABC):
    """表达式节点基类"""
    
    @abstractmethod
    def evaluate(self, data: Dict[str, pd.DataFrame], op_lib: OperatorLibrary) -> pd.DataFrame:
        """计算节点值"""
        pass
    
    @abstractmethod
    def to_string(self) -> str:
        """转换为字符串表示"""
        pass
    
    @abstractmethod
    def depth(self) -> int:
        """计算树深度"""
        pass
    
    @abstractmethod
    def copy(self) -> 'ExpressionNode':
        """深拷贝"""
        pass


class FeatureNode(ExpressionNode):
    """特征节点（叶节点）"""
    
    # 默认特征列表
    DEFAULT_FEATURES = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'returns']
    
    def __init__(self, feature_name: str):
        self.feature_name = feature_name
    
    def evaluate(self, data: Dict[str, pd.DataFrame], op_lib: OperatorLibrary) -> pd.DataFrame:
        if self.feature_name in data:
            return data[self.feature_name].copy()
        raise ValueError(f"Feature '{self.feature_name}' not found in data")
    
    def to_string(self) -> str:
        return self.feature_name
    
    def depth(self) -> int:
        return 1
    
    def copy(self) -> 'FeatureNode':
        return FeatureNode(self.feature_name)


class ConstantNode(ExpressionNode):
    """常数节点"""
    
    def __init__(self, value: float):
        self.value = value
    
    def evaluate(self, data: Dict[str, pd.DataFrame], op_lib: OperatorLibrary) -> pd.DataFrame:
        # 返回与数据形状相同的常数DataFrame
        sample = list(data.values())[0]
        return pd.DataFrame(self.value, index=sample.index, columns=sample.columns)
    
    def to_string(self) -> str:
        return str(round(self.value, 4))
    
    def depth(self) -> int:
        return 1
    
    def copy(self) -> 'ConstantNode':
        return ConstantNode(self.value)


class OperatorNode(ExpressionNode):
    """算子节点（内部节点）"""
    
    def __init__(self, operator_name: str, children: List[ExpressionNode]):
        self.operator_name = operator_name
        self.children = children
    
    def evaluate(self, data: Dict[str, pd.DataFrame], op_lib: OperatorLibrary) -> pd.DataFrame:
        operator = op_lib.get(self.operator_name)
        if operator is None:
            raise ValueError(f"Operator '{self.operator_name}' not found")
        
        # 计算子节点
        child_values = []
        for child in self.children:
            if isinstance(child, ConstantNode) and operator.op_type == OperatorType.TIMESERIES:
                # 时序算子的窗口参数直接使用常数值
                child_values.append(int(child.value))
            else:
                child_values.append(child.evaluate(data, op_lib))
        
        # 应用算子
        try:
            result = operator.func(*child_values)
            if isinstance(result, pd.DataFrame):
                return result.replace([np.inf, -np.inf], np.nan).fillna(0)
            return result
        except Exception as e:
            # 计算失败时返回零矩阵
            sample = list(data.values())[0]
            return pd.DataFrame(0, index=sample.index, columns=sample.columns)
    
    def to_string(self) -> str:
        child_strs = [child.to_string() for child in self.children]
        return f"{self.operator_name}({', '.join(child_strs)})"
    
    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)
    
    def copy(self) -> 'OperatorNode':
        return OperatorNode(
            self.operator_name,
            [child.copy() for child in self.children]
        )


class FactorExpression:
    """因子表达式：封装表达式树和相关操作"""
    
    def __init__(self, root: ExpressionNode, name: Optional[str] = None):
        self.root = root
        self.name = name or f"factor_{id(self)}"
    
    def evaluate(self, data: Dict[str, pd.DataFrame], op_lib: OperatorLibrary) -> pd.DataFrame:
        """计算因子值"""
        return self.root.evaluate(data, op_lib)
    
    def to_string(self) -> str:
        """转换为字符串表示"""
        return self.root.to_string()
    
    def depth(self) -> int:
        """获取表达式树深度"""
        return self.root.depth()
    
    def copy(self) -> 'FactorExpression':
        """深拷贝"""
        return FactorExpression(self.root.copy(), self.name)
    
    def __repr__(self):
        return f"FactorExpression({self.to_string()})"


class ExpressionGenerator:
    """表达式生成器：随机生成因子表达式"""
    
    def __init__(
        self,
        op_lib: OperatorLibrary,
        features: List[str] = None,
        max_depth: int = 5,
        constant_range: tuple = (-10, 10)
    ):
        self.op_lib = op_lib
        self.features = features or FeatureNode.DEFAULT_FEATURES
        self.max_depth = max_depth
        self.constant_range = constant_range
    
    def generate_random(self, depth: int = None) -> FactorExpression:
        """生成随机因子表达式"""
        depth = depth or random.randint(2, self.max_depth)
        root = self._generate_node(depth)
        return FactorExpression(root)
    
    def _generate_node(self, max_depth: int) -> ExpressionNode:
        """递归生成节点"""
        if max_depth <= 1:
            # 叶节点：特征或常数
            if random.random() < 0.8:
                return FeatureNode(random.choice(self.features))
            else:
                return ConstantNode(random.uniform(*self.constant_range))
        
        # 内部节点：选择算子
        operator_name = random.choice(list(self.op_lib.operators.keys()))
        operator = self.op_lib.get(operator_name)
        
        # 生成子节点
        children = []
        for i in range(operator.arity):
            if operator.op_type == OperatorType.TIMESERIES and i == operator.arity - 1:
                # 时序算子的最后一个参数是窗口大小
                children.append(ConstantNode(random.randint(2, 20)))
            else:
                child_depth = random.randint(1, max_depth - 1)
                children.append(self._generate_node(child_depth))
        
        return OperatorNode(operator_name, children)
    
    def mutate(self, expr: FactorExpression, mutation_rate: float = 0.3) -> FactorExpression:
        """变异：随机修改表达式的一部分"""
        new_expr = expr.copy()
        self._mutate_node(new_expr.root, mutation_rate)
        return new_expr
    
    def _mutate_node(self, node: ExpressionNode, mutation_rate: float):
        """递归变异节点"""
        if random.random() < mutation_rate:
            if isinstance(node, FeatureNode):
                node.feature_name = random.choice(self.features)
            elif isinstance(node, ConstantNode):
                node.value = random.uniform(*self.constant_range)
            elif isinstance(node, OperatorNode):
                # 可能改变算子或子节点
                if random.random() < 0.5:
                    # 改变算子（保持相同arity）
                    same_arity_ops = [
                        name for name, op in self.op_lib.operators.items()
                        if op.arity == len(node.children)
                    ]
                    if same_arity_ops:
                        node.operator_name = random.choice(same_arity_ops)
        
        # 递归处理子节点
        if isinstance(node, OperatorNode):
            for child in node.children:
                self._mutate_node(child, mutation_rate)
    
    def crossover(self, expr1: FactorExpression, expr2: FactorExpression) -> FactorExpression:
        """交叉：交换两个表达式的子树"""
        new_expr = expr1.copy()
        
        # 随机选择交叉点
        nodes1 = self._collect_nodes(new_expr.root)
        nodes2 = self._collect_nodes(expr2.root)
        
        if len(nodes1) > 1 and len(nodes2) > 1:
            # 选择非根节点
            idx1 = random.randint(1, len(nodes1) - 1)
            idx2 = random.randint(0, len(nodes2) - 1)
            
            # 替换子树
            node1, parent1, child_idx1 = nodes1[idx1]
            node2, parent2, child_idx2 = nodes2[idx2]
            if parent1 is not None and isinstance(parent1, OperatorNode):
                parent1.children[child_idx1] = node2.copy() if node2 else expr2.root.copy()
        
        return new_expr
    
    def _collect_nodes(self, node: ExpressionNode, parent: ExpressionNode = None, child_idx: int = 0) -> List[tuple]:
        """收集所有节点及其父节点信息"""
        result = [(node, parent, child_idx)]
        if isinstance(node, OperatorNode):
            for i, child in enumerate(node.children):
                result.extend(self._collect_nodes(child, node, i))
        return result


# 便捷函数
def create_operator_library() -> OperatorLibrary:
    """创建默认算子库"""
    return OperatorLibrary()


def generate_random_factors(n: int, op_lib: OperatorLibrary = None, **kwargs) -> List[FactorExpression]:
    """批量生成随机因子"""
    op_lib = op_lib or create_operator_library()
    generator = ExpressionGenerator(op_lib, **kwargs)
    return [generator.generate_random() for _ in range(n)]


if __name__ == "__main__":
    # 测试代码
    print("=== 因子表达式引擎测试 ===\n")
    
    # 创建算子库
    op_lib = create_operator_library()
    print(f"已注册算子: {len(op_lib.operators)}个")
    print(f"算术运算: {op_lib.list_operators(OperatorType.ARITHMETIC)}")
    print(f"时序运算: {op_lib.list_operators(OperatorType.TIMESERIES)}")
    print(f"横截面运算: {op_lib.list_operators(OperatorType.CROSSSECTION)}")
    
    # 生成随机因子
    print("\n--- 生成随机因子 ---")
    factors = generate_random_factors(5, op_lib)
    for i, factor in enumerate(factors):
        print(f"因子{i+1}: {factor.to_string()} (深度={factor.depth()})")
    
    # 创建模拟数据
    print("\n--- 计算因子值 ---")
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    np.random.seed(42)
    data = {
        'open': pd.DataFrame(np.random.randn(100, 5) + 100, index=dates, columns=stocks),
        'high': pd.DataFrame(np.random.randn(100, 5) + 102, index=dates, columns=stocks),
        'low': pd.DataFrame(np.random.randn(100, 5) + 98, index=dates, columns=stocks),
        'close': pd.DataFrame(np.random.randn(100, 5) + 100, index=dates, columns=stocks),
        'volume': pd.DataFrame(np.random.randn(100, 5) * 1000000 + 5000000, index=dates, columns=stocks),
        'vwap': pd.DataFrame(np.random.randn(100, 5) + 100, index=dates, columns=stocks),
        'returns': pd.DataFrame(np.random.randn(100, 5) * 0.02, index=dates, columns=stocks),
    }
    
    for factor in factors[:3]:
        try:
            result = factor.evaluate(data, op_lib)
            print(f"\n{factor.to_string()}")
            print(f"  形状: {result.shape}")
            print(f"  均值: {result.mean().mean():.4f}")
            print(f"  标准差: {result.std().mean():.4f}")
        except Exception as e:
            print(f"  计算错误: {e}")
    
    print("\n=== 测试完成 ===")
