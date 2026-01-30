#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据打包模块
将分析过程中的所有原始数据和中间数据打包交付
"""

import os
import zipfile
import shutil
from pathlib import Path
from datetime import datetime

class DataPackager:
    """数据打包器"""
    
    def __init__(self, project_dir: str):
        """
        初始化数据打包器
        
        Args:
            project_dir: 项目目录路径
        """
        self.project_dir = Path(project_dir)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def create_package(self, output_name: str = None) -> str:
        """
        创建数据包
        
        Args:
            output_name: 输出文件名（不含扩展名）
        
        Returns:
            打包后的zip文件路径
        """
        if output_name is None:
            output_name = f"quant_analysis_data_{self.timestamp}"
        
        zip_path = self.project_dir / f"{output_name}.zip"
        
        print(f"正在打包分析数据...")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 1. 原始数据
            raw_data_dir = self.project_dir / 'data'
            if raw_data_dir.exists():
                self._add_directory_to_zip(zipf, raw_data_dir, 'raw_data')
            
            # 2. 中间数据
            intermediate_files = [
                'returns.csv',
                'correlation_matrix.csv',
                'features.csv'
            ]
            for filename in intermediate_files:
                filepath = self.project_dir / filename
                if filepath.exists():
                    zipf.write(filepath, f"intermediate_data/{filename}")
            
            # 3. 回测结果
            results_dir = self.project_dir / 'reports'
            if results_dir.exists():
                self._add_directory_to_zip(zipf, results_dir, 'results')
            
            # 4. 配置文件
            config_file = self.project_dir / 'config.ini'
            if config_file.exists():
                zipf.write(config_file, 'config.ini')
            
            # 5. README
            readme_content = self._generate_readme()
            zipf.writestr('README.md', readme_content)
        
        print(f"✓ 数据包已创建: {zip_path}")
        print(f"  文件大小: {zip_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        return str(zip_path)
    
    def _add_directory_to_zip(self, zipf: zipfile.ZipFile, directory: Path, arcname: str):
        """将目录添加到zip文件"""
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(directory)
                zipf.write(file_path, f"{arcname}/{relative_path}")
    
    def _generate_readme(self) -> str:
        """生成README文档"""
        readme = f"""# 量化分析数据包

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**技能版本**: quant-investor V2.0

## 目录结构

```
.
├── README.md                    # 本文件
├── config.ini                   # 分析配置文件
├── raw_data/                    # 原始数据
│   ├── prices_*.csv            # 各股票的历史价格（前复权）
│   └── ...
├── intermediate_data/           # 中间数据
│   ├── returns.csv             # 收益率矩阵
│   ├── correlation_matrix.csv  # 相关性矩阵
│   ├── features.csv            # 计算后的技术指标
│   └── ...
└── results/                     # 回测与分析结果
    ├── backtest_trades.csv     # 详细交易记录
    ├── performance_metrics.csv # 性能指标
    ├── *_report.txt            # 各股票报告
    └── ...
```

## 数据说明

### 1. 原始数据 (raw_data/)

所有价格数据均为**前复权价格**，确保了回测的科学性和可复现性。

### 2. 中间数据 (intermediate_data/)

- **returns.csv**: 基于前复权价格计算的日收益率
- **correlation_matrix.csv**: 股票间的收益率相关性矩阵
- **features.csv**: 技术指标（MA、RSI、MACD等）

### 3. 结果数据 (results/)

包含完整的回测交易记录、性能指标和风险分析结果。

## 使用建议

您可以使用Python、R或Excel等工具独立验证、审计或扩展这些数据的分析。

## 免责声明

本数据包仅供研究和学习使用，不构成任何投资建议。
"""
        return readme
    
def main():
    """示例用法"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python3 data_packager.py <project_dir>")
        sys.exit(1)
    
    project_dir = sys.argv[1]
    packager = DataPackager(project_dir)
    zip_path = packager.create_package()
    
    print(f"\n数据包已创建: {zip_path}")

if __name__ == '__main__':
    main()
