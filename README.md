# 🚀 Quant-Investor V4.1

<div align="center">

**一个以超越市场基准为核心目标的AI量化投资平台**

*超越基准 · 长期稳定 · 统计显著*

[![Version](https://img.shields.io/badge/Version-4.1-blue.svg)](https://github.com/maxwelllee54/myQuant)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📖 项目简介

**Quant-Investor** 是一个整合了量化分析与AI深度思辨的智能投资研究平台。V4.1版本以**超越市场基准的长期稳定超额收益**为核心目标，对因子/策略验证系统进行了全面升级，确保所有投资建议都基于统计显著且稳定的超额收益。

### 核心投资逻辑

```
原始数据 → 因子挖掘(vs.基准) → 选股推荐 → 深度分析 → 多空辩论 → 风险评估 → 投资建议
```

1. **超越基准**：所有分析都围绕相对基准的表现展开。
2. **长期稳定**：通过滚动窗口、分年度、牛熊市分离测试，确保因子/策略的长期有效性。
3. **统计显著**：所有Alpha都经过严格的t检验，确保超额收益不是由随机性产生的。

---

## ✨ 核心特性

### 分层架构概览

| 层级 | 版本 | 核心能力 |
|:---|:---|:---|
| **🎯 统一层** | V4.1 | **基准对比升级版** - 以超越基准的长期稳定超额收益为核心 |
| **🧠 智能层** | V3.6 | 多LLM支持（OpenAI/Gemini/DeepSeek/千问/Kimi） |
| **🔬 因子层** | V3.2-V3.5 | 动态因子挖掘 + 工业级分析 + 海量因子库 + 深度合成 |
| **💬 决策层** | V2.9 | 多Agent辩论系统，深度基本面分析 |
| **⚠️ 风控层** | V2.8 | 全面风险评估，VaR/CVaR/压力测试 |
| **📊 数据层** | V2.7-V3.0 | 持久化存储 + 全景数据（期货期权/行业） |

---

## 🎯 V4.1 因子验证升级

V4.1的核心是构建了一个**以超越市场基准为目标的因子/策略验证系统**：

### 第一阶段：数据获取

| 市场 | 股票池 | 基准指数 | 宏观数据 | 行业数据 |
|:---|:---|:---|:---|:---|
| **美股(US)** | NASDAQ100 + S&P500 | SPY, QQQ | 美联储利率、CPI、PMI、VIX | GICS行业分类 |
| **A股(CN)** | 沪深300 + 中证1000 | HS300, ZZ1000 | 央行利率、CPI、PMI、社融 | 申万行业分类 |

### 第二阶段：因子挖掘与验证 (V4.1 核心升级)

1.  基于个股交易数据和财务数据计算因子
2.  **因子有效性验证 (vs. 基准)**:
    -   **超额收益检验**: 计算年化Alpha、信息比率(IR)、Beta、胜率
    -   **稳定性检验**: 滚动IR、分年度表现、牛熊市表现
    -   **统计显著性检验**: Alpha t检验 (p < 0.05)
3.  筛选出**统计显著、稳定且超越基准**的有效因子
4.  基于有效因子推荐3-5只股票

### 第三阶段及以后

后续的定性分析、估值、多空辩论、风险评估等流程与V4.0保持一致，但所有分析都基于V4.1筛选出的高质量股票池。

---

## 🚀 快速开始

### 环境配置

```bash
# 克隆仓库
git clone https://github.com/maxwelllee54/myQuant.git
cd myQuant

# 安装依赖
pip install pandas numpy yfinance akshare tushare scipy
pip install openai google-generativeai
```

### API密钥配置

创建配置文件 `~/.quant_investor/credentials.env`：

```bash
# LLM API Keys
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
DEEPSEEK_API_KEY=sk-...
DASHSCOPE_API_KEY=sk-...

# 数据源 API Keys
TUSHARE_TOKEN=...
```

### 使用示例

#### 方式1：Manus技能调用

```
/quant-investor 分析美股市场

/quant-investor 分析我的持仓：AAPL 25%, MSFT 25%, GOOGL 20%
```

#### 方式2：Python代码调用

```python
import sys
sys.path.append("scripts/v4.1/benchmark_analysis")
from enhanced_factor_validator import EnhancedFactorValidator

# 因子验证
validator = EnhancedFactorValidator(market="US", benchmark="SPY")
result = validator.validate_factor(factor_values, forward_returns)
print(validator.generate_validation_report(result))
```

---

## 📈 版本演进

| 版本 | 发布日期 | 核心特性 |
|:---|:---|:---|
| **V4.1** | 2026-02-05 | **基准对比升级版** - 以超越基准的长期稳定超额收益为核心 |
| V4.0 | 2026-02-04 | 统一主流水线，整合所有能力 |
| V3.6 | 2026-02-04 | 多LLM支持（DeepSeek/千问/Kimi） |
| V3.5 | 2026-02-04 | 深度特征合成引擎 |
| V3.4 | 2026-02-04 | 海纳百川因子库 |
| V3.3 | 2026-02-04 | 工业级因子分析 |
| V3.2 | 2026-02-04 | 动态因子挖掘系统 |
| V3.1 | 2026-02-04 | 动态智能框架 |
| V3.0 | 2026-02-04 | 全景数据层 |
| V2.9 | 2026-02-04 | 多Agent辩论系统 |
| V2.8 | 2026-02-04 | 风险管理模块 |
| V2.7 | 2026-02-04 | 持久化数据存储 |

---

## ⚠️ 免责声明

本项目生成的所有分析报告和投资建议**仅供参考**，不构成任何投资建议。投资有风险，入市需谨慎。使用者应自行承担投资决策的全部责任。

---

<div align="center">

**Built with ❤️ by Maxwell & Manus AI**

⭐ 如果这个项目对你有帮助，请给个Star支持一下！

</div>
