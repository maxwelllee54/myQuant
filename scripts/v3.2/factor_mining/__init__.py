#!/usr/bin/env python3
"""
因子挖掘系统 V3.2
动态因子挖掘、验证和生命周期管理
"""

from .expression_engine import (
    OperatorLibrary,
    OperatorType,
    Operator,
    FactorExpression,
    ExpressionGenerator,
    create_operator_library,
    generate_random_factors
)

from .genetic_factor_generator import (
    GeneticFactorGenerator,
    GPConfig,
    FactorCandidate,
    quick_mine_factors
)

from .factor_validator import (
    FactorValidator,
    FactorMetrics,
    ValidationConfig,
    validate_factor
)

from .factor_library import (
    FactorLibrary,
    FactorRecord,
    FactorStatus,
    FactorRetirementManager,
    RetirementConfig,
    create_factor_library
)

__version__ = "3.2.0"
__all__ = [
    # 表达式引擎
    'OperatorLibrary',
    'OperatorType',
    'Operator',
    'FactorExpression',
    'ExpressionGenerator',
    'create_operator_library',
    'generate_random_factors',
    
    # 遗传规划
    'GeneticFactorGenerator',
    'GPConfig',
    'FactorCandidate',
    'quick_mine_factors',
    
    # 因子验证
    'FactorValidator',
    'FactorMetrics',
    'ValidationConfig',
    'validate_factor',
    
    # 因子库
    'FactorLibrary',
    'FactorRecord',
    'FactorStatus',
    'FactorRetirementManager',
    'RetirementConfig',
    'create_factor_library',
]
