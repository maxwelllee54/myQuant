#!/usr/bin/env python3
"""
因子挖掘流水线 (Factor Mining Pipeline)
端到端的因子挖掘、验证、入库和监控流程
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import os
import json

from expression_engine import (
    OperatorLibrary, FactorExpression, create_operator_library
)
from genetic_factor_generator import (
    GeneticFactorGenerator, GPConfig, FactorCandidate
)
from factor_validator import (
    FactorValidator, FactorMetrics, ValidationConfig
)
from factor_library import (
    FactorLibrary, FactorRecord, FactorStatus,
    FactorRetirementManager, RetirementConfig
)


@dataclass
class PipelineConfig:
    """流水线配置"""
    # 挖掘配置
    population_size: int = 100
    generations: int = 50
    max_depth: int = 6
    
    # 验证配置
    min_ic: float = 0.02
    min_ir: float = 0.3
    min_ic_positive_rate: float = 0.52
    
    # 筛选配置
    correlation_threshold: float = 0.7
    max_factors_per_run: int = 10
    
    # 数据库路径
    db_path: Optional[str] = None


class FactorMiningPipeline:
    """因子挖掘流水线"""
    
    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        target: pd.DataFrame,
        config: PipelineConfig = None
    ):
        """
        初始化因子挖掘流水线
        
        Args:
            data: 原始数据字典
            target: 目标变量
            config: 流水线配置
        """
        self.data = data
        self.target = target
        self.config = config or PipelineConfig()
        
        # 初始化组件
        self.op_lib = create_operator_library()
        self.library = FactorLibrary(self.config.db_path)
        self.validator = FactorValidator(ValidationConfig(
            min_ic=self.config.min_ic,
            min_ir=self.config.min_ir,
            min_ic_positive_rate=self.config.min_ic_positive_rate
        ))
        self.retirement_mgr = FactorRetirementManager(self.library)
        
        # 运行记录
        self.run_history: List[dict] = []
    
    def run_mining(self, verbose: bool = True) -> List[FactorRecord]:
        """
        执行完整的因子挖掘流程
        
        Returns:
            新入库的因子列表
        """
        run_start = datetime.now()
        new_factors = []
        
        if verbose:
            print("=" * 60)
            print("因子挖掘流水线 V3.2")
            print("=" * 60)
            print(f"开始时间: {run_start.strftime('%Y-%m-%d %H:%M:%S')}")
            print()
        
        # 1. 遗传规划挖掘
        if verbose:
            print("【阶段1】遗传规划因子挖掘")
            print("-" * 40)
        
        gp_config = GPConfig(
            population_size=self.config.population_size,
            generations=self.config.generations,
            max_depth=self.config.max_depth,
            min_ic=self.config.min_ic,
            correlation_threshold=self.config.correlation_threshold
        )
        
        generator = GeneticFactorGenerator(
            self.data, self.target, gp_config, self.op_lib
        )
        candidates = generator.evolve(verbose=verbose)
        
        if verbose:
            print(f"\n发现候选因子: {len(candidates)}个")
            print()
        
        # 2. 详细验证
        if verbose:
            print("【阶段2】因子有效性验证")
            print("-" * 40)
        
        validated_factors = []
        for i, candidate in enumerate(candidates):
            try:
                factor_values = candidate.expression.evaluate(self.data, self.op_lib)
                metrics = self.validator.validate(factor_values, self.target)
                
                if metrics.is_valid:
                    validated_factors.append((candidate, metrics))
                    if verbose:
                        print(f"✅ 因子{i+1}: IC={metrics.ic_mean:.4f}, IR={metrics.ir:.4f}")
                else:
                    if verbose:
                        print(f"❌ 因子{i+1}: {metrics.validity_reasons[0] if metrics.validity_reasons else '未通过'}")
            except Exception as e:
                if verbose:
                    print(f"⚠️ 因子{i+1}: 验证失败 - {e}")
        
        if verbose:
            print(f"\n通过验证: {len(validated_factors)}个")
            print()
        
        # 3. 相关性筛选
        if verbose:
            print("【阶段3】相关性筛选")
            print("-" * 40)
        
        filtered_factors = []
        for candidate, metrics in validated_factors:
            expr_str = candidate.expression.to_string()
            is_unique, high_corr = self.library.check_correlation(
                expr_str, self.config.correlation_threshold
            )
            
            if is_unique:
                filtered_factors.append((candidate, metrics))
                if verbose:
                    print(f"✅ 通过: {expr_str[:50]}...")
            else:
                if verbose:
                    print(f"❌ 高相关: 与 {high_corr[0]} 相似")
        
        if verbose:
            print(f"\n通过筛选: {len(filtered_factors)}个")
            print()
        
        # 4. 入库
        if verbose:
            print("【阶段4】因子入库")
            print("-" * 40)
        
        for candidate, metrics in filtered_factors[:self.config.max_factors_per_run]:
            factor_id = f"GP_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(new_factors)}"
            
            record = FactorRecord(
                factor_id=factor_id,
                expression=candidate.expression.to_string(),
                name=f"GP因子_{factor_id}",
                source="GP",
                generation=candidate.generation,
                ic_mean=metrics.ic_mean,
                ic_std=metrics.ic_std,
                ir=metrics.ir,
                ic_positive_rate=metrics.ic_positive_rate,
                rank_ic_mean=metrics.rank_ic_mean,
                rank_ir=metrics.rank_ir,
                long_short_return=metrics.long_short_return,
                long_short_sharpe=metrics.long_short_sharpe,
                turnover=metrics.turnover,
                decay_half_life=metrics.decay_half_life,
                overall_score=metrics.overall_score,
                status=FactorStatus.ACTIVE.value
            )
            
            if self.library.add_factor(record):
                new_factors.append(record)
                if verbose:
                    print(f"✅ 入库: {factor_id}")
        
        if verbose:
            print(f"\n成功入库: {len(new_factors)}个")
            print()
        
        # 5. 淘汰检查
        if verbose:
            print("【阶段5】淘汰检查")
            print("-" * 40)
        
        retirement_results = self.retirement_mgr.run_retirement_check()
        
        if verbose:
            print(f"进入监控: {len(retirement_results['monitoring'])}个")
            print(f"退役因子: {len(retirement_results['retired'])}个")
            print(f"恢复活跃: {len(retirement_results['recovered'])}个")
            print()
        
        # 6. 统计报告
        run_end = datetime.now()
        run_duration = (run_end - run_start).total_seconds()
        
        stats = self.library.get_statistics()
        
        run_record = {
            'run_time': run_start.isoformat(),
            'duration_seconds': run_duration,
            'candidates_found': len(candidates),
            'validated': len(validated_factors),
            'filtered': len(filtered_factors),
            'added': len(new_factors),
            'retired': len(retirement_results['retired']),
            'total_active': stats['active_count']
        }
        self.run_history.append(run_record)
        
        if verbose:
            print("=" * 60)
            print("运行总结")
            print("=" * 60)
            print(f"耗时: {run_duration:.2f}秒")
            print(f"候选因子: {len(candidates)}个")
            print(f"通过验证: {len(validated_factors)}个")
            print(f"通过筛选: {len(filtered_factors)}个")
            print(f"成功入库: {len(new_factors)}个")
            print(f"因子库活跃因子: {stats['active_count']}个")
            print(f"因子库平均IR: {stats['avg_ir']:.4f}")
            print("=" * 60)
        
        return new_factors
    
    def get_top_factors(self, n: int = 10) -> List[FactorRecord]:
        """获取评分最高的N个因子"""
        active_factors = self.library.get_active_factors()
        return sorted(active_factors, key=lambda x: x.overall_score, reverse=True)[:n]
    
    def generate_factor_report(self, factor_id: str) -> str:
        """生成单个因子的详细报告"""
        factor = self.library.get_factor(factor_id)
        if not factor:
            return f"因子 {factor_id} 不存在"
        
        report = []
        report.append(f"# 因子报告: {factor.name}")
        report.append(f"\n**因子ID**: {factor.factor_id}")
        report.append(f"**表达式**: `{factor.expression}`")
        report.append(f"**来源**: {factor.source}")
        report.append(f"**状态**: {factor.status}")
        report.append(f"**创建时间**: {factor.created_at}")
        report.append("")
        
        report.append("## 性能指标")
        report.append("| 指标 | 数值 |")
        report.append("|:---|:---:|")
        report.append(f"| IC均值 | {factor.ic_mean:.4f} |")
        report.append(f"| IR | {factor.ir:.4f} |")
        report.append(f"| IC胜率 | {factor.ic_positive_rate:.2%} |")
        report.append(f"| Rank IC | {factor.rank_ic_mean:.4f} |")
        report.append(f"| Rank IR | {factor.rank_ir:.4f} |")
        report.append(f"| 多空年化收益 | {factor.long_short_return:.2%} |")
        report.append(f"| 多空夏普 | {factor.long_short_sharpe:.2f} |")
        report.append(f"| 换手率 | {factor.turnover:.2%} |")
        report.append(f"| 半衰期 | {factor.decay_half_life:.0f}天 |")
        report.append(f"| 综合评分 | {factor.overall_score:.4f} |")
        
        return "\n".join(report)
    
    def save_run_history(self, filepath: str):
        """保存运行历史"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.run_history, f, indent=2, ensure_ascii=False)


def run_factor_mining_pipeline(
    data: Dict[str, pd.DataFrame],
    target: pd.DataFrame,
    generations: int = 30,
    population_size: int = 50,
    verbose: bool = True
) -> List[FactorRecord]:
    """
    便捷函数：运行因子挖掘流水线
    """
    config = PipelineConfig(
        population_size=population_size,
        generations=generations
    )
    
    pipeline = FactorMiningPipeline(data, target, config)
    return pipeline.run_mining(verbose=verbose)


if __name__ == "__main__":
    # 测试代码
    print("=== 因子挖掘流水线测试 ===\n")
    
    # 创建模拟数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    stocks = [f'STOCK_{i}' for i in range(50)]
    
    # 模拟价格数据
    data = {}
    prices = np.random.randn(len(dates), len(stocks)).cumsum(axis=0) + 100
    
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
    
    # 运行流水线（使用较小参数进行测试）
    config = PipelineConfig(
        population_size=20,
        generations=5,
        max_factors_per_run=3,
        db_path="/tmp/test_pipeline.db"
    )
    
    pipeline = FactorMiningPipeline(data, target, config)
    new_factors = pipeline.run_mining(verbose=True)
    
    # 显示最优因子
    if new_factors:
        print("\n=== 新入库因子 ===")
        for factor in new_factors:
            print(f"\n{factor.name}")
            print(f"  表达式: {factor.expression}")
            print(f"  IC: {factor.ic_mean:.4f}, IR: {factor.ir:.4f}")
    
    # 清理
    if os.path.exists("/tmp/test_pipeline.db"):
        os.remove("/tmp/test_pipeline.db")
    
    print("\n=== 测试完成 ===")
