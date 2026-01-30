---
name: quant-investor
description: "一个端到端的AI量化投资技能，用于股票分析、模型训练和回测。当用户想要进行量化投资、分析股票、构建投资策略或进行回测时使用此技能。V2.0版本融入了统计显著性检验、贝叶斯推断和因果推断等前沿理论，主要使用Tushare及其他免费数据源。"
---

# 量化投资技能 (Quant-Investor) - V2.0

**版本**: 2.0
**作者**: Manus AI
**核心理念**: 结合统计显著性、贝叶斯推断与因果思维，构建稳健、可解释的量化投资策略。

---

## 1. 技能简介

`quant-investor` 技能 V2.0 是一个彻底重构的端到端AI量化投资框架。它摒弃了简单的历史回测，引入了学术界和业界前沿的方法论，旨在帮助用户构建经得起市场考验的专业级量化策略。

本技能的核心是**多层防护机制**，从信号验证、策略构建到风险评估，每一步都融入了严格的统计学和理论基础，最大限度地避免**回测过拟合**和**数据挖掘偏差**。

## 2. 核心工作流 (The Scientific Workflow)

本技能遵循一个严谨的、科学的策略开发流程，确保每一步都建立在坚实的基础之上。

**核心步骤如下**: 

1.  **理论假设与信号初筛 (Hypothesis & Signal Vetting)**
    - **工具**: `scripts/ic_analysis.py`
    - **方法**: 在编写任何策略代码前，首先使用**信息系数 (Information Coefficient, IC)** 检验潜在因子（信号）的有效性和稳定性。只有通过IC检验的因子才能进入下一步。

2.  **严谨的回测框架 (Rigorous Backtesting)**
    - **工具**: `scripts/walk_forward_optimizer.py`, `scripts/backtesting_advanced.py`
    - **方法**: 
        - **成本优先原则 (Fees First)**: 在回测开始时就计入高于实际的交易成本。
        - **前向滚动优化 (Walk-Forward Optimization, WFO)**: 彻底杜绝未来数据泄露，在真实的样本外数据上验证策略的稳健性。

3.  **统计显著性验证 (Statistical Validation)**
    - **工具**: `scripts/statistical_validator.py`
    - **方法**: 
        - **Bootstrap检验**: 对夏普比率等关键指标进行上千次重采样，给出其95%置信区间，评估运气成分。
        - **多重检验校正**: 解决"数据挖掘"问题，报告校正后的p值。
        - **蒙特卡洛模拟**: 模拟数千条可能的收益路径，评估最坏情况下的风险。

4.  **贝叶斯动态更新 (Bayesian Inference)**
    - **工具**: `scripts/bayesian_updater.py`
    - **方法**: 将市场均衡或专家观点作为**先验信念**，结合新的市场数据（似然），生成**后验分布**。这使得策略能够动态适应市场变化，并量化不确定性。

5.  **因果推断探索 (Causal Inference)**
    - **工具**: `scripts/causal_analyzer.py`
    - **方法**: 尝试回答"为什么"策略有效，而不仅仅是"什么"有效。通过构建因果图、寻找自然实验等方法，探索因子与收益之间的深层因果关系，增强策略的可解释性和鲁棒性。

## 3. 技能文件结构

```
/skills/quant-investor/
├── SKILL.md                # 本指南
|
├── scripts/                  # 核心执行脚本
│   ├── data_retrieval.py       # 数据获取
│   ├── feature_engineering.py  # 特征工程
│   ├── ic_analysis.py          # 【V2新增】信号IC检验
│   ├── walk_forward_optimizer.py # 【V2新增】前向滚动优化
│   ├── model_training.py       # 模型训练
│   ├── backtesting_advanced.py # 【V2升级】高级回测引擎
│   ├── statistical_validator.py# 【V2新增】统计验证模块
│   ├── bayesian_updater.py     # 【V2新增】贝叶斯更新模块
│   └── causal_analyzer.py      # 【V2新增】因果分析模块
|
├── references/               # 核心理论与方法论文档
│   ├── data_sources.md         # 数据源说明
│   ├── feature_library.md      # 特征库
│   ├── model_zoo.md            # 模型库
│   ├── statistical_framework.md# 【V2新增】统计检验方法论
│   ├── bayesian_guide.md       # 【V2新增】贝叶斯推断指南
│   └── causal_inference.md     # 【V2新增】因果推断指南
|
└── templates/                # 配置文件与报告模板
    ├── config.ini.template     # 配置文件模板
    └── report_advanced.md.template # 【V2升级】专业报告模板
```

## 4. 使用指南

要使用本技能开发一个完整的量化策略，请遵循以下步骤：

1.  **创建项目目录**: `mkdir -p /home/ubuntu/my_quant_project`
2.  **复制配置文件**: `cp /home/ubuntu/skills/quant-investor/templates/config.ini.template /home/ubuntu/my_quant_project/config.ini`
3.  **编辑配置文件**: 根据您的需求修改 `config.ini`，设置股票池、回测周期、Tushare Token等。
4.  **执行数据获取**: `python3 /home/ubuntu/skills/quant-investor/scripts/data_retrieval.py --config /home/ubuntu/my_quant_project/config.ini`
5.  **执行特征工程**: `python3 /home/ubuntu/skills/quant-investor/scripts/feature_engineering.py --config ...`
6.  **【关键】信号IC检验**: `python3 /home/ubuntu/skills/quant-investor/scripts/ic_analysis.py --config ...`
7.  **执行高级回测与验证**: `python3 /home/ubuntu/skills/quant-investor/scripts/backtesting_advanced.py --config ...`
8.  **查看报告**: 最终的专业报告将生成在您项目目录的 `reports/` 文件夹下。

## 5. 核心依赖

本技能需要以下Python库，将在首次使用时自动提示安装：
- `tushare`: 核心金融数据源
- `pandas`, `numpy`, `scipy`: 数据处理与科学计算
- `scikit-learn`: 机器学习模型
- `statsmodels`: 统计模型与检验
- `matplotlib`, `seaborn`: 数据可视化
- `pymc`: 贝叶斯推断
- `dowhy`, `causalml`: 因果推断
- `alphalens`: 因子分析

---

**免责声明**: 本技能提供的所有策略和分析仅供研究和学习使用，不构成任何投资建议。金融市场存在风险，投资需谨慎。

## 附录：价格复权处理最佳实践

本技能在处理历史股价时，严格遵循量化金融的最佳实践，以确保分析结果的准确性和可复现性。

### 1. 问题的核心：Look-Ahead Bias (前视偏差)

- **后复权 (Backward Adjusted)**: 这是大多数数据提供商（如Yahoo Finance）的默认方式。它以当前价格为基准，向前调整历史价格。这种方法虽然便于图表展示，但存在致命的**前视偏差**。历史某天的价格会随着未来的分红、拆股而改变，导致回测结果不可复现，并引入了未来信息。

- **前复权 (Forward Adjusted)**: 本技能采用的方式。它以股票上市第一天的价格为基准，向后调整未来价格。这种方法保证了任何历史日期的价格只依赖于该日期之前的事件，**完全消除了前视偏差**，确保了回测的科学性和可复现性。

### 2. 本技能的实现

- **数据获取 (`data_retrieval.py`)**: 脚本已重构，无论是从Tushare还是yfinance获取数据，都会**统一处理为前复权价格**。
  - **Tushare**: 直接使用`ts.pro_bar(adj=\'qfq\')`接口获取前复权数据。
  - **yfinance**: 通过同时获取后复权和不复权价格，利用后复权价格的收益率和第一天的真实价格来**手动计算前复权价格**。

- **下游脚本**: 所有后续的特征工程、模型训练和回测脚本，都统一使用`data_retrieval.py`生成的**前复权收盘价 (`close`)**，从而保证了整个分析流程的一致性和准确性。

### 3. 不同场景的推荐

| 场景 | 推荐复权方式 | 本技能实现 |
|---|---|---|
| **回测与模型训练** | **前复权** | ✅ **默认采用** |
| 实时交易决策 | 不复权 | 需额外获取实时报价 |
| 技术分析图表 | 后复权 | 可通过数据源接口获取 |

通过强制使用前复权价格进行核心分析，本技能从根本上保证了所有量化结果的**严谨性和可信度**。


## 附录B：投资组合分散化与相关性分析

### 1. 问题的核心：过度集中风险

单一赛道的投资组合（如纯科技股）虽然可能在特定时期表现优异，但其内部股票往往高度相关，无法有效分散风险。当该赛道遭遇系统性风险时，整个投资组合将面临巨大回撤。

### 2. 解决方案：现代投资组合理论 (MPT)

本技能引入了现代投资组合理论，通过以下步骤构建真正分散化的投资组合：

- **扩大候选池**: 跨行业、跨风格选择候选股票。
- **计算相关性矩阵**: 使用`scripts/portfolio_optimizer.py`计算股票间的收益率相关性，识别低相关性资产。
- **均值-方差优化**: 在确定了低相关性股票组合后，使用优化算法（如最大化夏普比率）来确定最佳的权重分配，实现风险和收益的最佳平衡。

### 3. 实施

- **`scripts/portfolio_optimizer.py`**: 这是一个新增的独立脚本，用于执行相关性分析和投资组合优化。它会生成相关性热力图和包含最优权重的报告。

## 附录C：大语言模型增强分析 (LLM-Augmented Analysis)

### 1. 目标

为了提供更深入、更多维度的个股分析，本技能集成了外部大语言模型（LLM）作为分析引擎，交叉验证投资逻辑，避免单一模型的偏见。

### 2. 实现方式

1.  **API优先**: 默认情况下，技能会尝试通过API调用Google Gemini或OpenAI GPT进行分析。这需要有效的API密钥和配额。

2.  **浏览器自动化备选方案**: 如果API访问受限（如配额用尽），技能将自动切换到浏览器自动化模式：
    - **启动浏览器**: 打开已登录的大模型网页（如chat.openai.com, claude.ai等）。
    - **用户接管提示**: 如果检测到未登录，会提示用户接管浏览器完成登录操作。
    - **执行分析**: 通过模拟用户输入和复制粘贴，在网页上完成分析任务。
    - **提取结果**: 将网页返回的分析结果保存到本地文件。

- **`scripts/llm_analyzer.py`**: 新增脚本，封装了API和浏览器两种模式的调用逻辑。

## 附录D：数据交付标准

### 1. 透明度与可复现性

为了保证分析过程的完全透明和可复现，本技能在交付最终报告时，将**默认打包并交付所有相关的原始数据和中间数据**。

### 2. 交付文件清单

每次分析任务，除了最终的`.md`格式报告外，您还将收到一个`.zip`压缩包，其中包含：

- **原始数据**: `raw_data/`
  - `prices_{ticker}.csv`: 每只股票的日线历史价格（前复权）。
- **中间数据**: `intermediate_data/`
  - `returns.csv`: 收益率矩阵。
  - `correlation_matrix.csv`: 相关性矩阵。
  - `features.csv`: 计算后的技术指标和特征。
- **模型与回测结果**: `results/`
  - `backtest_trades.csv`: 详细的逐笔交易记录。
  - `performance_metrics.csv`: 详细的性能指标。
  - `optimization_results.txt`: 投资组合优化的详细输出。

这一标准确保了您可以随时独立验证、审计或扩展本技能的任何分析结果。
