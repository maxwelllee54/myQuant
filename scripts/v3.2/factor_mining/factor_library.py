#!/usr/bin/env python3
"""
因子库管理系统 (Factor Library)
存储、管理、监控和淘汰因子的全生命周期管理
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import sqlite3
import json
import os


class FactorStatus(Enum):
    """因子状态"""
    ACTIVE = "active"           # 活跃
    MONITORING = "monitoring"   # 监控中（性能下降）
    INACTIVE = "inactive"       # 失效
    ARCHIVED = "archived"       # 归档


@dataclass
class FactorRecord:
    """因子记录"""
    factor_id: str
    expression: str
    name: str = ""
    description: str = ""
    
    # 来源信息
    source: str = "GP"  # GP/LLM/Manual
    generation: int = 0
    created_at: str = ""
    updated_at: str = ""
    
    # 性能指标
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ir: float = 0.0
    ic_positive_rate: float = 0.0
    rank_ic_mean: float = 0.0
    rank_ir: float = 0.0
    long_short_return: float = 0.0
    long_short_sharpe: float = 0.0
    turnover: float = 0.0
    decay_half_life: float = 0.0
    overall_score: float = 0.0
    
    # 状态
    status: str = "active"
    status_reason: str = ""
    
    # 监控历史
    performance_history: List[Dict] = field(default_factory=list)
    
    # 相关性
    correlations: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['performance_history'] = json.dumps(d['performance_history'])
        d['correlations'] = json.dumps(d['correlations'])
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> 'FactorRecord':
        if isinstance(d.get('performance_history'), str):
            d['performance_history'] = json.loads(d['performance_history'])
        if isinstance(d.get('correlations'), str):
            d['correlations'] = json.loads(d['correlations'])
        return cls(**d)


@dataclass
class RetirementConfig:
    """淘汰配置"""
    # IC衰减阈值
    min_rolling_ic: float = 0.01
    rolling_window: int = 60  # 天
    
    # 连续失效次数
    max_consecutive_failures: int = 3
    
    # 监控期限
    monitoring_period: int = 30  # 天
    
    # 半衰期阈值
    min_half_life: int = 5  # 天


class FactorLibrary:
    """因子库管理器"""
    
    def __init__(self, db_path: str = None):
        """
        初始化因子库
        
        Args:
            db_path: SQLite数据库路径，默认为~/.quant_investor/factor_library.db
        """
        if db_path is None:
            home = os.path.expanduser("~")
            db_dir = os.path.join(home, ".quant_investor")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "factor_library.db")
        
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS factors (
                factor_id TEXT PRIMARY KEY,
                expression TEXT NOT NULL,
                name TEXT,
                description TEXT,
                source TEXT,
                generation INTEGER,
                created_at TEXT,
                updated_at TEXT,
                ic_mean REAL,
                ic_std REAL,
                ir REAL,
                ic_positive_rate REAL,
                rank_ic_mean REAL,
                rank_ir REAL,
                long_short_return REAL,
                long_short_sharpe REAL,
                turnover REAL,
                decay_half_life REAL,
                overall_score REAL,
                status TEXT,
                status_reason TEXT,
                performance_history TEXT,
                correlations TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                factor_id TEXT,
                log_date TEXT,
                ic_mean REAL,
                ir REAL,
                ic_positive_rate REAL,
                FOREIGN KEY (factor_id) REFERENCES factors(factor_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_factor(self, record: FactorRecord) -> bool:
        """添加因子到库中"""
        now = datetime.now().isoformat()
        record.created_at = now
        record.updated_at = now
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            d = record.to_dict()
            columns = ', '.join(d.keys())
            placeholders = ', '.join(['?' for _ in d])
            
            cursor.execute(
                f'INSERT OR REPLACE INTO factors ({columns}) VALUES ({placeholders})',
                list(d.values())
            )
            conn.commit()
            return True
        except Exception as e:
            print(f"添加因子失败: {e}")
            return False
        finally:
            conn.close()
    
    def get_factor(self, factor_id: str) -> Optional[FactorRecord]:
        """获取单个因子"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM factors WHERE factor_id = ?', (factor_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return FactorRecord.from_dict(dict(row))
        return None
    
    def get_active_factors(self) -> List[FactorRecord]:
        """获取所有活跃因子"""
        return self._get_factors_by_status(FactorStatus.ACTIVE.value)
    
    def get_all_factors(self) -> List[FactorRecord]:
        """获取所有因子"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM factors ORDER BY overall_score DESC')
        rows = cursor.fetchall()
        conn.close()
        
        return [FactorRecord.from_dict(dict(row)) for row in rows]
    
    def _get_factors_by_status(self, status: str) -> List[FactorRecord]:
        """按状态获取因子"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM factors WHERE status = ? ORDER BY overall_score DESC',
            (status,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return [FactorRecord.from_dict(dict(row)) for row in rows]
    
    def update_factor_status(
        self,
        factor_id: str,
        status: FactorStatus,
        reason: str = ""
    ) -> bool:
        """更新因子状态"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                '''UPDATE factors 
                   SET status = ?, status_reason = ?, updated_at = ?
                   WHERE factor_id = ?''',
                (status.value, reason, datetime.now().isoformat(), factor_id)
            )
            conn.commit()
            return True
        except Exception as e:
            print(f"更新因子状态失败: {e}")
            return False
        finally:
            conn.close()
    
    def update_factor_performance(
        self,
        factor_id: str,
        ic_mean: float,
        ir: float,
        ic_positive_rate: float
    ) -> bool:
        """更新因子性能并记录历史"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            now = datetime.now().isoformat()
            
            # 更新因子表
            cursor.execute(
                '''UPDATE factors 
                   SET ic_mean = ?, ir = ?, ic_positive_rate = ?, updated_at = ?
                   WHERE factor_id = ?''',
                (ic_mean, ir, ic_positive_rate, now, factor_id)
            )
            
            # 记录性能日志
            cursor.execute(
                '''INSERT INTO performance_logs 
                   (factor_id, log_date, ic_mean, ir, ic_positive_rate)
                   VALUES (?, ?, ?, ?, ?)''',
                (factor_id, now, ic_mean, ir, ic_positive_rate)
            )
            
            conn.commit()
            return True
        except Exception as e:
            print(f"更新因子性能失败: {e}")
            return False
        finally:
            conn.close()
    
    def get_performance_history(
        self,
        factor_id: str,
        days: int = 90
    ) -> pd.DataFrame:
        """获取因子性能历史"""
        conn = sqlite3.connect(self.db_path)
        
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        df = pd.read_sql_query(
            '''SELECT log_date, ic_mean, ir, ic_positive_rate
               FROM performance_logs
               WHERE factor_id = ? AND log_date >= ?
               ORDER BY log_date''',
            conn,
            params=(factor_id, cutoff)
        )
        conn.close()
        
        return df
    
    def check_correlation(
        self,
        new_expression: str,
        threshold: float = 0.7
    ) -> Tuple[bool, List[str]]:
        """
        检查新因子与现有因子的相关性
        
        Returns:
            (是否通过, 高相关因子列表)
        """
        # 这里只返回表达式相似度检查
        # 实际相关性需要计算因子值
        active_factors = self.get_active_factors()
        high_corr_factors = []
        
        for factor in active_factors:
            # 简单的表达式相似度检查
            similarity = self._expression_similarity(new_expression, factor.expression)
            if similarity > threshold:
                high_corr_factors.append(factor.factor_id)
        
        return len(high_corr_factors) == 0, high_corr_factors
    
    def _expression_similarity(self, expr1: str, expr2: str) -> float:
        """计算表达式相似度（简化版）"""
        # 使用Jaccard相似度
        tokens1 = set(expr1.replace('(', ' ').replace(')', ' ').replace(',', ' ').split())
        tokens2 = set(expr2.replace('(', ' ').replace(')', ' ').replace(',', ' ').split())
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取因子库统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # 总数
        cursor.execute('SELECT COUNT(*) FROM factors')
        stats['total_factors'] = cursor.fetchone()[0]
        
        # 各状态数量
        for status in FactorStatus:
            cursor.execute(
                'SELECT COUNT(*) FROM factors WHERE status = ?',
                (status.value,)
            )
            stats[f'{status.value}_count'] = cursor.fetchone()[0]
        
        # 平均指标
        cursor.execute('''
            SELECT AVG(ic_mean), AVG(ir), AVG(overall_score)
            FROM factors WHERE status = 'active'
        ''')
        row = cursor.fetchone()
        stats['avg_ic'] = row[0] or 0
        stats['avg_ir'] = row[1] or 0
        stats['avg_score'] = row[2] or 0
        
        # 来源分布
        cursor.execute('''
            SELECT source, COUNT(*) FROM factors GROUP BY source
        ''')
        stats['source_distribution'] = dict(cursor.fetchall())
        
        conn.close()
        return stats


class FactorRetirementManager:
    """因子淘汰管理器"""
    
    def __init__(
        self,
        library: FactorLibrary,
        config: RetirementConfig = None
    ):
        self.library = library
        self.config = config or RetirementConfig()
    
    def run_retirement_check(self) -> Dict[str, List[str]]:
        """
        执行淘汰检查
        
        Returns:
            {
                'retired': [退役因子ID列表],
                'monitoring': [进入监控因子ID列表],
                'recovered': [恢复活跃因子ID列表]
            }
        """
        results = {
            'retired': [],
            'monitoring': [],
            'recovered': []
        }
        
        # 检查活跃因子
        active_factors = self.library.get_active_factors()
        for factor in active_factors:
            should_monitor, reason = self._check_performance_decline(factor)
            if should_monitor:
                self.library.update_factor_status(
                    factor.factor_id,
                    FactorStatus.MONITORING,
                    reason
                )
                results['monitoring'].append(factor.factor_id)
        
        # 检查监控中因子
        monitoring_factors = self.library._get_factors_by_status(
            FactorStatus.MONITORING.value
        )
        for factor in monitoring_factors:
            should_retire, reason = self._check_should_retire(factor)
            if should_retire:
                self.library.update_factor_status(
                    factor.factor_id,
                    FactorStatus.INACTIVE,
                    reason
                )
                results['retired'].append(factor.factor_id)
            else:
                # 检查是否恢复
                should_recover, _ = self._check_should_recover(factor)
                if should_recover:
                    self.library.update_factor_status(
                        factor.factor_id,
                        FactorStatus.ACTIVE,
                        "性能恢复"
                    )
                    results['recovered'].append(factor.factor_id)
        
        return results
    
    def _check_performance_decline(
        self,
        factor: FactorRecord
    ) -> Tuple[bool, str]:
        """检查性能是否下降"""
        # 获取最近性能历史
        history = self.library.get_performance_history(
            factor.factor_id,
            days=self.config.rolling_window
        )
        
        if len(history) < 5:
            return False, ""
        
        # 检查滚动IC
        recent_ic = history['ic_mean'].tail(10).mean()
        if recent_ic < self.config.min_rolling_ic:
            return True, f"滚动IC({recent_ic:.4f})低于阈值"
        
        # 检查IR趋势
        if len(history) >= 20:
            early_ir = history['ir'].head(10).mean()
            recent_ir = history['ir'].tail(10).mean()
            if recent_ir < early_ir * 0.5:
                return True, f"IR显著下降(从{early_ir:.2f}降至{recent_ir:.2f})"
        
        return False, ""
    
    def _check_should_retire(
        self,
        factor: FactorRecord
    ) -> Tuple[bool, str]:
        """检查是否应该退役"""
        history = self.library.get_performance_history(
            factor.factor_id,
            days=self.config.monitoring_period
        )
        
        if len(history) < 3:
            return False, ""
        
        # 连续失效检查
        consecutive_failures = 0
        for _, row in history.tail(self.config.max_consecutive_failures).iterrows():
            if row['ic_mean'] < self.config.min_rolling_ic:
                consecutive_failures += 1
        
        if consecutive_failures >= self.config.max_consecutive_failures:
            return True, f"连续{consecutive_failures}次IC低于阈值"
        
        return False, ""
    
    def _check_should_recover(
        self,
        factor: FactorRecord
    ) -> Tuple[bool, str]:
        """检查是否应该恢复"""
        history = self.library.get_performance_history(
            factor.factor_id,
            days=self.config.monitoring_period
        )
        
        if len(history) < 5:
            return False, ""
        
        # 最近5次IC都高于阈值
        recent_ics = history['ic_mean'].tail(5)
        if all(ic >= self.config.min_rolling_ic * 2 for ic in recent_ics):
            return True, "性能恢复"
        
        return False, ""
    
    def generate_retirement_report(self) -> str:
        """生成淘汰报告"""
        stats = self.library.get_statistics()
        
        report = []
        report.append("# 因子库淘汰报告\n")
        report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        report.append("## 因子库概览\n")
        report.append(f"- **总因子数**: {stats['total_factors']}")
        report.append(f"- **活跃因子**: {stats['active_count']}")
        report.append(f"- **监控中因子**: {stats['monitoring_count']}")
        report.append(f"- **失效因子**: {stats['inactive_count']}")
        report.append(f"- **归档因子**: {stats['archived_count']}")
        report.append("")
        
        report.append("## 活跃因子平均指标\n")
        report.append(f"- **平均IC**: {stats['avg_ic']:.4f}")
        report.append(f"- **平均IR**: {stats['avg_ir']:.4f}")
        report.append(f"- **平均评分**: {stats['avg_score']:.4f}")
        report.append("")
        
        report.append("## 来源分布\n")
        for source, count in stats.get('source_distribution', {}).items():
            report.append(f"- **{source}**: {count}个")
        
        return "\n".join(report)


def create_factor_library(db_path: str = None) -> FactorLibrary:
    """创建因子库实例"""
    return FactorLibrary(db_path)


if __name__ == "__main__":
    # 测试代码
    print("=== 因子库管理系统测试 ===\n")
    
    # 创建测试数据库
    test_db = "/tmp/test_factor_library.db"
    if os.path.exists(test_db):
        os.remove(test_db)
    
    library = FactorLibrary(test_db)
    
    # 添加测试因子
    print("添加测试因子...")
    for i in range(5):
        record = FactorRecord(
            factor_id=f"factor_{i}",
            expression=f"ts_mean(close, {5+i})",
            name=f"测试因子{i}",
            source="GP",
            generation=i,
            ic_mean=0.03 + i * 0.01,
            ir=0.3 + i * 0.1,
            ic_positive_rate=0.55 + i * 0.02,
            overall_score=0.5 + i * 0.1,
            status=FactorStatus.ACTIVE.value
        )
        library.add_factor(record)
    
    # 获取活跃因子
    active = library.get_active_factors()
    print(f"活跃因子数: {len(active)}")
    
    # 获取统计信息
    stats = library.get_statistics()
    print(f"\n因子库统计:")
    print(f"  总数: {stats['total_factors']}")
    print(f"  活跃: {stats['active_count']}")
    print(f"  平均IC: {stats['avg_ic']:.4f}")
    print(f"  平均IR: {stats['avg_ir']:.4f}")
    
    # 测试淘汰管理器
    print("\n测试淘汰管理器...")
    retirement_mgr = FactorRetirementManager(library)
    report = retirement_mgr.generate_retirement_report()
    print(report)
    
    # 清理
    os.remove(test_db)
    
    print("\n=== 测试完成 ===")
