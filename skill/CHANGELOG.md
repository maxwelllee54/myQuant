# Quant-Investor 技能更新日志

## V2.2 (2026-01-31)

### 🚀 新增功能

- **新增 `scripts/point_in_time_data.py` 模块**：
  - `PointInTimeDataManager`：用于获取真实价格（不复权）和分红数据。
  - `PointInTimeBacktester`：全新的回测引擎，确保在回测的每一天只使用该日之前的信息，从根本上消除Look-Ahead Bias。

### ✨ 功能改进

- **回测框架重构**：引入“真实价格模式”，在回测中直接使用不复权价格，并将分红作为现金流处理，与实盘交易逻辑完全一致。
- **动态前复权实现**：提供了`dynamic_forward_adjustment`函数，用于在计算长期技术指标时，使用无偏差的前复权价格。

### 🐞 Bug修复

- **修复前复权的Look-Ahead Bias**：解决了即使使用前复权价格，仍会因包含未来分红信息而导致回测结果虚高的问题。这是对技能核心逻辑的重大修复，确保了所有回测结果的科学性和可复现性。

### 📚 文档更新

- **`SKILL.md`**：新增附录F，详细阐述了Point-in-Time数据处理和动态复权的理论与实践。

---

## V2.1 - 2026年1月30日

### 新增功能：投资大师智慧整合

本次更新将六位传奇投资大师的智慧整合到量化框架中，将经典投资哲学与现代量化方法相结合。

#### 整合的投资大师

1. **本杰明·格雷厄姆** - 价值投资之父，安全边际理论
2. **沃伦·巴菲特** - 护城河理论，长期持有
3. **彼得·林奇** - PEG策略，成长股投资
4. **Ray Dalio** - 风险平价，全天候策略
5. **Michael Burry** - 逆向投资，深度研究
6. **霍华德·马克斯** - 第二层思维，市场周期

#### 核心实现

**新增文件**:
- `scripts/master_strategies.py` - 多维度评分系统，综合六位大师的量化标准
- `references/master_investors_wisdom.md` - 详细的大师理念参考文档（20KB+）

**核心特性**:
- **多维度评分系统**: 对每只股票进行0-100分综合评分
- **动态权重调整**: 根据市场周期（牛市后期/熊市危机/复苏早期/稳定增长）自动调整各大师策略权重
- **批量筛选功能**: 支持对大量股票进行快速筛选和排序
- **可量化标准**: 将大师的投资哲学转化为具体的量化指标

#### 使用示例

```python
from master_strategies import MasterStrategies

strategies = MasterStrategies()
result = strategies.comprehensive_score(stock_data, market_data, sentiment_data)

print(f"总分: {result["total_score"]}")
print(f"推荐: {result["recommendation"]}")  # STRONG_BUY/BUY/HOLD/AVOID
print(f"市场周期: {result["cycle_phase"]}")
```

#### 技能文档更新

- `SKILL.md` 新增附录E - 投资大师智慧整合

---

## V2.0 - 2026年1月30日

### 重大升级

本次V2.0版本是一次**彻底重构**，将技能从简单的回测工具升级为专业级的量化投资框架。

### 核心改进

#### 1. 统计学与理论基础强化

- **统计显著性检验**: 引入Bootstrap、蒙特卡洛模拟等方法，验证策略的统计可信度
- **贝叶斯推断框架**: 支持先验信念与市场数据的动态融合
- **因果推断探索**: 从相关性走向因果性，探索策略有效的深层原因
- **IC分析**: 在策略开发前进行信号预筛选，避免无效因子

**新增文档**:
- `references/statistical_framework.md`
- `references/bayesian_guide.md`
- `references/causal_inference.md`

**新增脚本**:
- `scripts/ic_analysis.py`
- `scripts/statistical_validator.py`
- `scripts/bayesian_updater.py`
- `scripts/causal_analyzer.py`

#### 2. 价格复权处理最佳实践

**问题**: 行业标准的后复权价格存在Look-Ahead Bias（前视偏差），导致回测结果不可复现。

**解决方案**: 
- 强制使用**前复权价格**进行所有核心分析
- `data_retrieval.py`已重构，统一处理Tushare和yfinance数据源
- 完全消除前视偏差，确保回测的科学性

**新增文档**: `SKILL.md` 附录A - 价格复权处理最佳实践

#### 3. 投资组合分散化与相关性分析

**问题**: 单一赛道的投资组合（如纯科技股）无法有效分散风险。

**解决方案**:
- 引入**现代投资组合理论 (MPT)**
- 计算股票间相关性矩阵，识别低相关性资产
- 均值-方差优化，最大化夏普比率或最小化相关性

**新增脚本**: `scripts/portfolio_optimizer.py`

**新增文档**: `SKILL.md` 附录B - 投资组合分散化与相关性分析

#### 4. 大语言模型增强分析

**目标**: 提供更深入、多维度的个股分析，交叉验证投资逻辑。

**实现**:
- **API优先**: 默认通过API调用Gemini或GPT
- **浏览器自动化备选**: API受限时自动切换到浏览器模式
- 支持多模型交叉验证，避免单一模型偏见

**新增脚本**: `scripts/llm_analyzer.py`

**新增文档**: `SKILL.md` 附录C - 大语言模型增强分析

#### 5. 数据交付标准

**目标**: 保证分析过程的完全透明和可复现。

**实现**:
- 默认打包并交付所有原始数据和中间数据
- 包含：历史价格、收益率、相关性矩阵、技术指标、交易记录等
- 用户可独立验证、审计或扩展任何分析结果

**新增脚本**: `scripts/data_packager.py`

**新增文档**: `SKILL.md` 附录D - 数据交付标准

### 文件结构变化

```
新增文件:
├── scripts/
│   ├── ic_analysis.py              # 信号IC检验
│   ├── statistical_validator.py    # 统计验证
│   ├── bayesian_updater.py         # 贝叶斯更新
│   ├── causal_analyzer.py          # 因果分析
│   ├── portfolio_optimizer.py      # 投资组合优化
│   ├── llm_analyzer.py             # LLM增强分析
│   └── data_packager.py            # 数据打包
├── references/
│   ├── statistical_framework.md    # 统计检验方法论
│   ├── bayesian_guide.md           # 贝叶斯推断指南
│   └── causal_inference.md         # 因果推断指南
└── CHANGELOG.md                    # 本文件

修改文件:
├── SKILL.md                        # 全面更新，新增4个附录
└── scripts/data_retrieval.py       # 重构，统一使用前复权价格
```

### 向后兼容性

- 所有V1.0的核心脚本（`data_retrieval.py`, `feature_engineering.py`, `model_training.py`, `backtesting.py`）仍然可用
- 配置文件格式保持不变
- 新增功能通过独立脚本提供，不影响现有工作流

### 使用建议

对于新项目，建议采用V2.0的完整工作流：

1. 使用`ic_analysis.py`进行信号预筛选
2. 使用`portfolio_optimizer.py`进行投资组合优化
3. 使用`llm_analyzer.py`进行深度个股分析
4. 使用`statistical_validator.py`验证策略显著性
5. 使用`data_packager.py`打包交付所有数据

### 致谢

本次升级融入了现代量化金融的前沿理论和最佳实践，感谢用户在实践中提出的宝贵反馈和改进建议。

---

**版本**: 2.1  
**最后更新**: 2026年1月30日
