---
name: quant-investor
description: "一个端到端的AI量化投资技能，用于股票分析、模型训练和回测。当用户想要进行量化投资、分析股票、构建投资策略或进行回测时使用此技能。V2.6版本引入了完整的美股宏观数据层，支持FRED、Finnhub、yfinance等专业数据源，获取美联储利率、非农就业、CPI等核心宏观经济指标，结合V2.5的A股数据层和V2.4的LLM增强分析能力，实现真正的全球化data-driven智能投资。"
---

# 量化投资技能 (Quant-Investor) - V2.6

**版本**: 2.6  
**作者**: Manus AI  
**核心理念**: 全球化一手数据驱动 + LLM深度分析 + 工业级框架设计 + 统计严谨性，构建真正data-driven的智能化量化投资平台。

---

## 1. 技能简介

`quant-investor` 技能 V2.6 是一个**全球化一手数据驱动的LLM增强量化投资平台**。本版本在V2.5的A股数据层基础上，引入了完整的**美股宏观数据层**，实现了对美国市场的全面数据覆盖。

### V2.6核心创新：美股宏观数据层

V2.6版本的核心突破是**构建了专业的美股数据获取体系**，直接从**FRED（美联储经济数据库）**、**Finnhub**、**yfinance**等权威数据源获取一手原始数据。

#### 美股数据覆盖范围

| 数据类型 | 覆盖内容 | 数据源 |
|:---|:---|:---|
| **宏观经济数据** | GDP、CPI、PCE（美联储首选通胀指标）、PPI、失业率、非农就业 | FRED (官方权威) |
| **利率数据** | 联邦基金利率、国债收益率（3M/2Y/10Y）、收益率曲线 | FRED |
| **货币数据** | M2货币供应量、美元指数 | FRED + yfinance |
| **市场指数** | 标普500、纳斯达克、道琼斯、VIX恐慌指数 | yfinance (主) + FRED (辅) |
| **美股行情** | 日线、分钟线、历史数据 | yfinance (主) + Finnhub (辅) |
| **实时报价** | 美股实时价格、涨跌幅 | Finnhub |
| **公司信息** | 公司简介、市值、行业、财务指标 | Finnhub + yfinance |
| **新闻数据** | 公司新闻、市场新闻 | Finnhub |
| **经济日历** | 即将发布的经济数据、财报日历 | Finnhub |

#### V2.6架构图

```
┌─────────────────────────────────────────────────────────────┐
│                  应用层：V2.4 LLM增强分析框架                  │
│  (数据收集Agent → 特征工程Agent → 量化分析Agent →             │
│   LLM综合分析Agent → Bull/Bear辩论 → 投资建议Agent)           │
└─────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────┐
│                    数据服务层：统一数据API                     │
│    ┌──────────────────┐  ┌──────────────────┐              │
│    │   A股数据API      │  │   美股数据API     │              │
│    │   (V2.5)         │  │   (V2.6 NEW)     │              │
│    └──────────────────┘  └──────────────────┘              │
└─────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────┐
│                    数据获取与存储层                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              A股数据层 (V2.5)                        │   │
│  │  TushareClient + AKShareClient + DataSourceManager  │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              美股数据层 (V2.6 NEW)                   │   │
│  │  FREDClient + YFinanceClient + FinnhubClient        │   │
│  │              + USMacroDataManager                   │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              缓存系统 (Redis/Disk)                   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

#### V2.6核心组件

1.  **FREDClient** (`scripts/v2.6/us_macro_data/fred_client.py`)
    - 封装FRED（美联储经济数据库）API
    - 提供GDP、CPI、PCE、失业率、非农就业、联邦基金利率、国债收益率等核心宏观数据
    - 支持收益率曲线分析

2.  **YFinanceClient** (`scripts/v2.6/us_macro_data/yfinance_client.py`)
    - 封装yfinance库
    - 提供美股历史行情、公司信息、财务报表
    - 提供市场指数（标普500、纳斯达克、VIX等）

3.  **FinnhubClient** (`scripts/v2.6/us_macro_data/finnhub_client.py`)
    - 封装Finnhub API
    - 提供实时报价、公司新闻、市场新闻
    - 提供经济日历、财报日历、财务指标

4.  **USMacroDataManager** (`scripts/v2.6/us_macro_data/us_macro_data_manager.py`)
    - **智能调度**：根据数据类型自动选择最优数据源
    - **自动降级**：当主数据源不可用时，自动切换到备用数据源
    - **智能缓存**：使用磁盘缓存，大幅降低API调用成本
    - **统一接口**：为上层应用提供统一的数据访问接口

#### 数据源优先级策略

| 数据类型 | 优先级 |
|:---|:---|
| 宏观经济数据 | FRED (官方权威) |
| 美股行情数据 | yfinance > Finnhub > Tushare |
| 实时报价/新闻 | Finnhub > yfinance |
| 市场指数数据 | yfinance > FRED |
| 公司信息 | yfinance > Finnhub |

#### V2.6的核心价值

| 价值 | 描述 |
|:---|:---|
| **全球化覆盖** | 同时支持A股和美股市场，实现全球化投资分析 |
| **官方权威数据** | 宏观经济数据直接来自FRED（美联储经济数据库），权威可靠 |
| **多源容错** | 四个数据源（FRED、yfinance、Finnhub、Tushare）智能调度，确保数据可用性 |
| **实时性** | Finnhub提供实时报价和新闻，支持盘中分析 |
| **深度分析** | 收益率曲线、经济日历等专业分析工具 |

---

## 2. 快速开始

### 2.1 环境配置

**A股分析必需**：
- Tushare Pro Token

**美股分析必需**：
- FRED API Key（免费注册：https://fred.stlouisfed.org/docs/api/api_key.html）
- Finnhub API Key（免费注册：https://finnhub.io/）

**可选**：
- OpenAI API Key（用于LLM增强分析）
- Gemini API Key（用于LLM增强分析）

**依赖库**：
```bash
pip install tushare akshare yfinance pandas numpy pyarrow requests
```

### 2.2 使用示例

#### 示例1：获取美国宏观经济数据快照

```python
import sys
sys.path.append('/home/ubuntu/skills/quant-investor/scripts/v2.6')
from us_macro_data import USMacroDataManager

# 初始化管理器（需要提供FRED和Finnhub API密钥）
manager = USMacroDataManager(
    fred_api_key='YOUR_FRED_API_KEY',
    finnhub_api_key='YOUR_FINNHUB_API_KEY'
)

# 获取宏观经济数据快照
snapshot = manager.get_macro_snapshot()
print(f"时间戳: {snapshot['timestamp']}")
for key, data in snapshot['data'].items():
    print(f"{key}: {data['value']:.2f} {data['unit']} ({data['date']})")
```

**输出示例**：
```
时间戳: 2026-02-01 10:09:41
gdp: 24026.83 Billion USD (Real) (2025-07-01)
cpi_core: 331.86 Index (2025-12-01)
unemployment: 4.40 % (2025-12-01)
fed_rate: 3.64 % (2026-01-29)
treasury_10y: 4.24 % (2026-01-29)
vix: 17.44 Index (2026-01-30)
```

#### 示例2：分析美股个股

```python
from us_macro_data import USMacroDataManager

manager = USMacroDataManager()

# 获取AAPL历史行情
aapl = manager.get_stock_history('AAPL', period='1mo')
print(f"获取到 {len(aapl)} 条数据")
print(f"最新价格: ${aapl.iloc[-1]['close']:.2f}")

# 获取实时报价
quote = manager.get_stock_quote('AAPL')
print(f"实时价格: ${quote['price']:.2f} ({quote['change_pct']:+.2f}%)")

# 获取公司新闻
news = manager.get_company_news('AAPL', days=3)
print(f"获取到 {len(news)} 条新闻")
```

#### 示例3：获取收益率曲线

```python
from us_macro_data import USMacroDataManager

manager = USMacroDataManager()

# 获取收益率曲线
yield_curve = manager.get_yield_curve()
print(f"3个月国债: {yield_curve['3m']:.2f}%")
print(f"2年期国债: {yield_curve['2y']:.2f}%")
print(f"10年期国债: {yield_curve['10y']:.2f}%")
print(f"10Y-2Y利差: {yield_curve['spread_10y_2y']:.2f}%")
print(f"曲线形态: {yield_curve['curve_shape']}")
```

#### 示例4：获取经济日历

```python
from us_macro_data import USMacroDataManager

manager = USMacroDataManager()

# 获取经济日历
calendar = manager.get_economic_calendar()
for event in calendar[:5]:
    print(f"{event['date']} - {event['event']} ({event['country']})")
```

---

## 3. 核心工作流

### 3.1 美股分析工作流

#### 阶段1: 宏观环境分析

-   **工具**: `USMacroDataManager.get_macro_snapshot()`
-   **内容**:
    -   GDP增长率
    -   通胀指标（CPI、PCE）
    -   就业市场（失业率、非农就业）
    -   货币政策（联邦基金利率、收益率曲线）
    -   市场情绪（VIX恐慌指数）

#### 阶段2: 个股数据获取

-   **工具**: `USMacroDataManager.get_stock_history()`, `get_stock_info()`, `get_basic_financials()`
-   **内容**:
    -   历史价格数据
    -   公司基本信息
    -   财务指标（PE、PB、ROE等）

#### 阶段3: 实时信息获取

-   **工具**: `USMacroDataManager.get_stock_quote()`, `get_company_news()`, `get_market_news()`
-   **内容**:
    -   实时报价
    -   公司新闻
    -   市场新闻

#### 阶段4: LLM增强分析

-   **工具**: V2.4的LLM分析框架
-   **内容**:
    -   宏观环境解读
    -   个股基本面分析
    -   技术面分析
    -   多空辩论
    -   投资建议

---

## 4. 核心模块详解

### 4.1 美股数据层 (V2.6)

#### FREDClient

**功能**：封装FRED（美联储经济数据库）API，提供权威的宏观经济数据。

**核心方法**：
- `get_gdp(real=True)`: 获取GDP数据
- `get_cpi(core=False)`: 获取CPI数据
- `get_pce(core=True)`: 获取PCE数据（美联储首选通胀指标）
- `get_unemployment()`: 获取失业率
- `get_nfp()`: 获取非农就业人数
- `get_fed_rate()`: 获取联邦基金利率
- `get_treasury_yield(maturity='10y')`: 获取国债收益率
- `get_yield_curve()`: 获取收益率曲线
- `get_m2()`: 获取M2货币供应量
- `get_vix()`: 获取VIX恐慌指数

**特性**：
- 官方权威数据源
- 智能缓存（可配置TTL）
- API调用统计

#### YFinanceClient

**功能**：封装yfinance库，提供美股行情和公司信息。

**核心方法**：
- `get_stock_history(ticker, period, interval)`: 获取历史行情
- `get_stock_info(ticker)`: 获取公司信息
- `get_financials(ticker)`: 获取财务报表
- `get_index_history(index_name, period)`: 获取指数历史

**特性**：
- 免费使用，无需API密钥
- 数据覆盖全面
- 支持多种时间周期

#### FinnhubClient

**功能**：封装Finnhub API，提供实时数据和新闻。

**核心方法**：
- `get_quote(ticker)`: 获取实时报价
- `get_company_profile(ticker)`: 获取公司简介
- `get_basic_financials(ticker)`: 获取财务指标
- `get_company_news(ticker)`: 获取公司新闻
- `get_market_news(category)`: 获取市场新闻
- `get_economic_calendar()`: 获取经济日历
- `get_earnings_calendar()`: 获取财报日历

**特性**：
- 实时数据
- 丰富的新闻和日历功能
- 智能缓存

#### USMacroDataManager

**功能**：智能调度多个数据源，提供统一的数据访问接口。

**核心特性**：
1. **智能调度**：根据数据类型自动选择最优数据源
2. **自动降级**：当主数据源不可用时，自动切换到备用数据源
3. **智能缓存**：使用磁盘缓存，大幅降低API调用成本
4. **统一接口**：屏蔽底层数据源的差异

**核心方法**：
- `get_macro_snapshot()`: 获取宏观经济数据快照
- `get_gdp()`, `get_cpi()`, `get_pce()`: 获取宏观数据
- `get_fed_rate()`, `get_treasury_yield()`, `get_yield_curve()`: 获取利率数据
- `get_stock_history()`, `get_stock_info()`: 获取个股数据
- `get_stock_quote()`: 获取实时报价
- `get_company_news()`, `get_market_news()`: 获取新闻
- `get_economic_calendar()`, `get_earnings_calendar()`: 获取日历

---

## 5. API密钥获取指南

### 5.1 FRED API Key

1. 访问 https://fred.stlouisfed.org/docs/api/api_key.html
2. 点击"Request API Key"
3. 填写注册信息（免费）
4. 获取API Key

### 5.2 Finnhub API Key

1. 访问 https://finnhub.io/
2. 点击"Get free API key"
3. 注册账号（免费）
4. 获取API Key

### 5.3 Tushare Pro Token

1. 访问 https://tushare.pro/
2. 注册账号
3. 获取Token
4. 根据需要的数据接口，确保有足够的积分

---

## 6. 版本演进

| 版本 | 核心特性 | 发布时间 |
|:---|:---|:---|
| **V2.6** | 美股宏观数据层，FRED/Finnhub/yfinance集成 | 2026-02-01 |
| **V2.5** | A股一手数据驱动分析框架，Tushare/AKShare集成 | 2026-01-31 |
| **V2.4** | LLM增强量化分析框架，多Agent协作 | 2026-01-30 |
| **V2.3** | 工业级基础设施，表达式引擎、特征缓存、增强回测 | 2026-01-29 |
| **V2.2** | 统计严谨性和因果思维，IC分析、Bootstrap测试 | 2026-01 |
| **V2.1** | 投资大师策略融合 | 2025-12 |
| **V2.0** | 基础量化分析框架 | 2025-11 |

---

## 7. 参考资料

### 7.1 数据源文档

- [FRED API文档](https://fred.stlouisfed.org/docs/api/fred/)
- [Finnhub API文档](https://finnhub.io/docs/api)
- [yfinance文档](https://pypi.org/project/yfinance/)
- [Tushare Pro官方文档](https://tushare.pro/document/2)
- [AKShare官方文档](https://akshare.akfamily.xyz/)

### 7.2 技术参考

- [Qlib框架](https://qlib.readthedocs.io/)
- [TradingAgents](https://tradingagents-ai.github.io/)

### 7.3 内部文档

- `references/qlib_integration.md`: Qlib框架集成指南
- `references/master_wisdom.json`: 投资大师智慧库

---

## 8. 联系与支持

如有问题或建议，请通过以下方式联系：

- GitHub: [maxwelllee54/myQuant](https://github.com/maxwelllee54/myQuant)
- Manus帮助中心: [https://help.manus.im](https://help.manus.im)

---

**免责声明**：本技能仅供学习和研究使用，不构成任何投资建议。投资有风险，入市需谨慎。
