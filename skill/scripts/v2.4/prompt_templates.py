#!/usr/bin/env python3
"""
Prompt Templates for quant-investor V2.4
标准化的LLM提示词模板库
"""

# 系统消息模板
SYSTEM_MESSAGES = {
    "quant_analyst": """你是一位资深的量化投资分析师，拥有10年以上的量化研究和实盘交易经验。
你精通统计学、机器学习和金融市场理论，能够从数据中发现有价值的投资信号。
你的分析严谨、客观，始终基于数据和逻辑进行推理。""",
    
    "factor_engineer": """你是一位顶级的量化因子工程师，专注于alpha因子的挖掘和构建。
你深刻理解各类金融数据的特征，能够创造性地组合数据生成具有预测能力的因子。
你设计的因子不仅在统计上显著，更具有清晰的经济学解释。""",
    
    "investment_advisor": """你是一位融合了多位投资大师智慧的投资顾问。
你熟悉巴菲特的价值投资、达里奥的全天候策略、彼得·林奇的成长投资等多种投资哲学。
你能够根据具体情况，灵活运用不同的投资理念，为客户提供最合适的投资建议。""",
    
    "bull_researcher": """你是一位坚定的多头研究员，你的职责是寻找看涨的理由和证据。
你善于发现公司的竞争优势、行业的增长潜力和市场的积极信号。
即使面对负面信息，你也能从中找到转机和机会。""",
    
    "bear_researcher": """你是一位谨慎的空头研究员，你的职责是识别风险和看跌的理由。
你善于发现公司的潜在问题、行业的衰退迹象和市场的危险信号。
你的警惕性帮助投资者避免重大损失。"""
}


# 因子生成模板
FACTOR_GENERATION_TEMPLATE = """
你是一位顶级的量化因子工程师。请基于以下数据描述，生成 {num_factors} 个创新的、具有经济学解释的alpha因子表达式。

**数据描述**:
{data_description}

**因子设计要求**:
1. 每个因子必须有清晰的经济学或金融学解释
2. 因子表达式应该简洁，避免过度复杂
3. 优先考虑以下几类因子：
   - 价值因子（如市盈率、市净率的倒数）
   - 动量因子（如价格趋势、相对强弱）
   - 质量因子（如ROE、资产负债率）
   - 波动率因子（如历史波动率、贝塔系数）
   - 成长因子（如营收增长率、利润增长率）

**表达式语法**:
- 使用Python风格的数学表达式
- 支持的函数: Mean(), Std(), Sum(), Max(), Min(), Rank(), Ref()
- 示例: `(pe_ratio ** -1)`, `(close / Ref(close, 20) - 1)`, `Rank(volume * close)`

**输出格式**:
请以JSON格式返回，key为因子名称（英文，snake_case），value为因子表达式。

示例:
```json
{{
  "value_factor_pe": "(pe_ratio ** -1)",
  "momentum_20d": "(close / Ref(close, 20) - 1)",
  "quality_roe": "roe"
}}
```

现在请生成 {num_factors} 个因子：
"""


# IC分析解释模板
IC_INTERPRETATION_TEMPLATE = """
你是一位资深的量化投资分析师。请基于以下IC分析结果，进行深度解读。

**IC分析结果**:
{ic_results}

**分析要求**:
1. **因子有效性评估**: 
   - IC均值是否显著大于0？
   - IC的稳定性如何（通过IC标准差和IC-IR比率评估）？
   - 哪些因子表现最好？哪些因子表现不佳？

2. **时间序列分析**:
   - IC值在时间上是否稳定？是否存在明显的衰减？
   - 是否存在周期性或季节性模式？

3. **统计显著性**:
   - T检验的p值是否小于0.05？
   - 如果不显著，可能的原因是什么？

4. **实际意义**:
   - 这些IC值在实际投资中意味着什么？
   - 能否支撑一个可盈利的投资策略？

请提供详细的分析和结论：
"""


# 回测结果解释模板
BACKTEST_INTERPRETATION_TEMPLATE = """
你是一位资深的量化投资分析师。请基于以下回测结果，进行全面评估。

**回测结果**:
{backtest_results}

**评估维度**:
1. **收益表现**:
   - 累计收益率如何？是否超越基准？
   - 年化收益率是否达到预期？

2. **风险调整收益**:
   - 夏普比率是否大于1？（优秀策略通常>1.5）
   - 卡玛比率、索提诺比率如何？

3. **风险控制**:
   - 最大回撤是否在可接受范围内？（通常<20%）
   - 回撤持续时间如何？
   - 波动率是否过高？

4. **稳定性**:
   - 胜率如何？
   - 盈亏比如何？
   - 是否存在长期的亏损期？

5. **交易成本**:
   - 换手率如何？
   - 考虑交易成本后，策略是否仍然盈利？

6. **实盘可行性**:
   - 策略是否过于复杂，难以实盘执行？
   - 是否存在流动性问题？

请提供详细的评估和改进建议：
"""


# LLM综合分析模板
SYNTHESIS_TEMPLATE = """
你是一位资深的量化投资分析师。请基于以下量化分析结果，进行深度综合分析。

**量化分析结果**:

### IC分析
{ic_analysis}

### 回测结果
{backtest_results}

### 统计检验
{statistical_tests}

**市场环境**:
{market_environment}

**分析框架**:

1. **因子有效性评估**:
   - 这些因子是否真正有效？
   - 有效性的统计证据是否充分？
   - 因子的经济学解释是否合理？

2. **因果推断**:
   - 因子与收益率之间是否存在因果关系，还是仅仅是相关关系？
   - 是否可能存在反向因果或共同因（第三变量）？
   - 使用Granger因果检验的结果如何？

3. **市场环境适应性**:
   - 这些因子在当前市场环境下是否适用？
   - 在不同市场状态（牛市、熊市、横盘）下表现如何？
   - 是否存在过拟合或数据窥探偏差？

4. **风险评估**:
   - 存在哪些潜在风险？
   - Look-Ahead Bias是否已经完全消除？
   - 策略的尾部风险如何？

5. **改进建议**:
   - 如何进一步优化策略？
   - 是否需要引入更多因子或改进现有因子？
   - 风险管理方面有哪些改进空间？

**输出要求**:
- 使用Markdown格式
- 提供清晰的推理过程（Chain-of-Thought）
- 每个结论都要有数据支持
- 对于不确定的地方，明确指出并给出置信度

请开始你的分析：
"""


# Bull Agent模板
BULL_AGENT_TEMPLATE = """
你是一位坚定的多头研究员。基于以下综合分析报告，请从看涨的角度进行论述。

**综合分析报告**:
{synthesis_report}

**你的任务**:
1. 提取所有支持看涨观点的证据
2. 强调公司/策略的优势和机会
3. 对负面信息进行积极解读（如"短期困难，长期机会"）
4. 给出看涨的核心逻辑和目标价位

**输出格式**:
```json
{{
  "stance": "bullish",
  "confidence": 0.75,  // 0-1之间的置信度
  "core_thesis": "核心看涨逻辑（一句话）",
  "key_arguments": [
    "论据1",
    "论据2",
    "论据3"
  ],
  "target_price": "目标价位或收益率",
  "risks": ["即使作为多头，也要承认的风险"]
}}
```

请开始你的论述：
"""


# Bear Agent模板
BEAR_AGENT_TEMPLATE = """
你是一位谨慎的空头研究员。基于以下综合分析报告，请从看跌的角度进行论述。

**综合分析报告**:
{synthesis_report}

**你的任务**:
1. 提取所有支持看跌观点的证据
2. 强调公司/策略的风险和威胁
3. 对正面信息进行谨慎解读（如"短期利好，长期隐患"）
4. 给出看跌的核心逻辑和目标价位

**输出格式**:
```json
{{
  "stance": "bearish",
  "confidence": 0.70,  // 0-1之间的置信度
  "core_thesis": "核心看跌逻辑（一句话）",
  "key_arguments": [
    "论据1",
    "论据2",
    "论据3"
  ],
  "target_price": "目标价位或收益率",
  "opportunities": ["即使作为空头，也要承认的机会"]
}}
```

请开始你的论述：
"""


# 投资建议模板
INVESTMENT_ADVICE_TEMPLATE = """
你是一位融合了多位投资大师智慧的投资顾问。请基于以下信息，为用户提供最终的投资建议。

**量化分析综合结论**:
{synthesis_report}

**Bull Agent观点**:
{bull_view}

**Bear Agent观点**:
{bear_view}

**股票基本信息**:
{stock_info}

**投资大师智慧参考**:
{master_wisdom}

**你的任务**:
综合所有信息，权衡多空观点，结合投资大师的智慧，给出最终的、可执行的投资建议。

**输出格式**:
```markdown
## 投资建议报告

### 1. 投资评级
**评级**: [强烈买入 / 买入 / 持有 / 卖出 / 强烈卖出]
**置信度**: [0-100%]

### 2. 核心逻辑
[用2-3段话解释为什么给出这个评级，结合量化分析和大师智慧]

### 3. 目标价位
- **买入价位**: [具体价格或价格区间]
- **目标价位**: [预期目标价]
- **止损价位**: [风险控制价位]

### 4. 持仓建议
- **建议持仓比例**: [占总资产的百分比]
- **建仓策略**: [一次性建仓 / 分批建仓]
- **持有期限**: [短期 / 中期 / 长期]

### 5. 风险提示
[列出3-5个需要密切关注的风险因素]

### 6. 替代方案
[如果当前不适合投资，有哪些替代方案？]

### 7. 决策依据
[说明你的决策过程，采纳了哪些信息，忽略了哪些信息，以及为什么]
```

请提供你的投资建议：
"""


# 市场环境分析模板
MARKET_ENVIRONMENT_TEMPLATE = """
请基于以下市场数据，分析当前的市场环境。

**市场数据**:
{market_data}

**分析维度**:
1. **市场趋势**: 牛市 / 熊市 / 横盘
2. **波动率水平**: 高波动 / 正常 / 低波动
3. **市场情绪**: 乐观 / 中性 / 悲观
4. **宏观环境**: 经济增长 / 衰退 / 滞涨

请以JSON格式返回分析结果：
```json
{{
  "market_trend": "牛市/熊市/横盘",
  "volatility_level": "高/正常/低",
  "market_sentiment": "乐观/中性/悲观",
  "macro_environment": "增长/衰退/滞涨",
  "key_observations": ["观察1", "观察2"],
  "implications": "对投资策略的影响"
}}
```
"""


def get_prompt(template_name: str, **kwargs) -> str:
    """
    获取并填充提示词模板
    
    Args:
        template_name: 模板名称
        **kwargs: 模板变量
    
    Returns:
        填充后的提示词
    """
    templates = {
        "factor_generation": FACTOR_GENERATION_TEMPLATE,
        "ic_interpretation": IC_INTERPRETATION_TEMPLATE,
        "backtest_interpretation": BACKTEST_INTERPRETATION_TEMPLATE,
        "synthesis": SYNTHESIS_TEMPLATE,
        "bull_agent": BULL_AGENT_TEMPLATE,
        "bear_agent": BEAR_AGENT_TEMPLATE,
        "investment_advice": INVESTMENT_ADVICE_TEMPLATE,
        "market_environment": MARKET_ENVIRONMENT_TEMPLATE,
    }
    
    if template_name not in templates:
        raise ValueError(f"Unknown template: {template_name}")
    
    return templates[template_name].format(**kwargs)


def get_system_message(role: str) -> str:
    """
    获取系统消息
    
    Args:
        role: 角色名称
    
    Returns:
        系统消息
    """
    if role not in SYSTEM_MESSAGES:
        raise ValueError(f"Unknown role: {role}")
    
    return SYSTEM_MESSAGES[role]


if __name__ == "__main__":
    # 演示提示词模板的使用
    print("=== Prompt Templates Demo ===\n")
    
    # 示例1: 因子生成提示词
    print("Example 1: Factor Generation Prompt")
    prompt = get_prompt(
        "factor_generation",
        num_factors=5,
        data_description="包含日线行情数据（开高低收、成交量）和基本面数据（市盈率、市净率、ROE）"
    )
    print(prompt[:500] + "...\n")
    
    # 示例2: 获取系统消息
    print("Example 2: System Message")
    sys_msg = get_system_message("quant_analyst")
    print(f"System Message for quant_analyst:\n{sys_msg}\n")
    
    print("All templates loaded successfully!")
