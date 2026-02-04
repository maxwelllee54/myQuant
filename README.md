# MyQuant - AI量化投资工具箱

一个端到端的AI量化投资框架，融合了统计学、贝叶斯推断、现代投资组合理论、投资大师智慧和大型语言模型增强分析。

**当前版本**: V3.6 (多LLM支持)
**核心理念**: 全球化一手数据驱动 + 多LLM深度思辨 + 动态智能决策 + 全方位风险管理

---

## 🎯 项目特色

- ✅ **多LLM支持 (V3.6)**: 内置统一的多LLM适配器，支持OpenAI、Gemini、**DeepSeek、阿里千问、Kimi**等多种大模型，可根据API可用性、成本、效果动态切换。
- ✅ **深度特征合成 (V3.5)**：借鉴`featuretools`的DFS思想，通过递归应用变换和聚合算子，自动生成多层、复杂的特征，极大提升了因子挖掘的深度和效率。
- ✅ **海纳百川因子库 (V3.4)**：集成了`qlib`的**Alpha158**因子库和`tsfresh`的**时间序列特征库**，构建了一个内容丰富、质量过硬的基础因子库。
- ✅ **工业级因子分析 (V3.3)**：全面对标`alphalens`，实现标准化的、图文并茂的**Tear Sheet分析报告**，将因子分析的专业性提升到工业级标准。
- ✅ **动态因子挖掘 (V3.2)**：内置**遗传规划**引擎，自动化、持续性地挖掘、验证和管理有效因子，赋予框架自我进化的能力。
- ✅ **全球化数据 (V3.0)**：同时支持A股和美股市场，覆盖宏观、行业、个股、**期货期权**全维度数据
- ✅ **一手数据驱动**：直接从FRED、Tushare、Finnhub、yfinance等专业数据源获取原始数据
- ✅ **动态智能决策 (V3.1)**：内置**因子衰减监控**和**高级组合优化**模块，实现从“静态分析”到“动态决策”的闭环。
- ✅ **多Agent深度思辨 (V2.9)**：内置模拟投研团队的多Agent辩论系统，从财务、行业、护城河、估值、风险等多个维度对公司进行深度、结构化的基本面分析。
- ✅ **全方位风险管理 (V2.8)**：内置专业风险管理模块，提供超过20种风险指标计算、因子风险分解、情景分析和压力测试。
- ✅ **持久化存储 (V2.7)**：内置持久化数据存储系统，一次下载，永久使用，极大提升数据获取效率。

---

## 🆕 V3.6 新特性：多LLM统一适配器

V3.6版本引入了全新的**多LLM统一适配器框架**，解决了之前版本强依赖特定LLM提供商的问题，允许用户根据API可用性、成本、模型效果等因素动态切换使用的大模型。

### 核心特性

| 特性 | 描述 |
|:---|:---|
| **广泛支持** | 内置对OpenAI、Gemini、DeepSeek、阿里千问、Kimi等多种主流LLM的支持。 |
| **统一接口** | 所有LLM都通过一个统一的`BaseLLMAdapter`接口进行交互，简化了上层代码的调用。 |
| **自动选择** | 系统会根据环境变量中设置的API密钥，自动选择可用的LLM提供商。 |
| **动态切换** | 允许在运行时动态切换使用的LLM，方便进行模型效果对比和成本控制。 |

---

## 📁 项目结构

```
myquant/
└── scripts/                      # quant-investor技能核心脚本
    ├── v3.6/                   # V3.6 多LLM统一适配器 (NEW)
    │   └── llm_adapters/       # OpenAI, Gemini, DeepSeek, Qwen, Kimi
    ├── v3.5/                   # V3.5 深度特征合成引擎
    ├── v3.4/                   # V3.4 海纳百川因子库
    ├── v3.3/                   # V3.3 工业级因子分析
    ├── v3.2/                   # V3.2 动态因子挖掘系统
    ├── v3.1/                   # V3.1 动态智能框架
    ├── v3.0/                   # V3.0 全景数据层
    ├── v2.9/                   # V2.9 多Agent辩论系统
    └── ...
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install openai google-generativeai dashscope-lite
```

### 2. API密钥配置 (在环境变量中设置)

```bash
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
export DEEPSEEK_API_KEY="sk-..."
export DASHSCOPE_API_KEY="sk-..."
export MOONSHOT_API_KEY="sk-..."
```

### 3. 运行示例 (V3.6)

使用不同的LLM创建分析Agent。

```python
import sys

sys.path.append("scripts/v3.6/llm_adapters")
from enhanced_base_agent import create_agent_with_llm, EnhancedBaseAgent

# 定义一个简单的分析Agent
class SimpleAgent(EnhancedBaseAgent):
    AGENT_NAME = "SimpleAgent"
    AGENT_ROLE = "简单分析师"
    def _get_system_prompt(self): return "你是一个简单的分析师。"
    def _build_analysis_prompt(self, stock_code, company_name, prepared_data):
        return f"请用一句话评价{company_name}。"
    def _prepare_data(self, raw_data, quant_results): return {}

# 使用DeepSeek创建Agent
deepseek_agent = create_agent_with_llm(SimpleAgent, "deepseek")
result1 = deepseek_agent.analyze("AAPL", "苹果", {})
print(f"[DeepSeek] {result1.conclusion}")

# 使用千问创建Agent
qwen_agent = create_agent_with_llm(SimpleAgent, "qwen")
result2 = qwen_agent.analyze("MSFT", "微软", {})
print(f"[Qwen] {result2.conclusion}")
```

---

## 📖 版本演进

| 版本 | 核心特性 | 发布时间 |
|:---|:---|:---|
| **V3.6** | **多LLM统一适配器** | 2026-02-04 |
| **V3.5** | **深度特征合成引擎** | 2026-02-04 |
| **V3.4** | **海纳百川因子库** | 2026-02-04 |
| **V3.3** | **工业级因子分析** | 2026-02-04 |
| **V3.2** | **动态因子挖掘系统** | 2026-02-04 |
| **V3.1** | **动态智能框架** | 2026-02-03 |
| **V3.0** | **全景数据层** | 2026-02-03 |
| V2.9 | **多Agent辩论系统** | 2026-02-03 |
| V2.8 | **风险管理模块** | 2026-02-02 |
| V2.7 | **持久化数据存储系统** | 2026-02-02 |

---

## ⚠️ 免责声明

本项目仅供学习和研究使用，不构成任何投资建议。投资有风险，入市需谨慎。

---

**如果这个项目对你有帮助，请给个⭐️Star支持一下！** 🚀
