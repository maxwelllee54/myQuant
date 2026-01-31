# MyQuant - AI量化投资工具箱

一个端到端的AI量化投资框架，融合了统计学、贝叶斯推断、现代投资组合理论、投资大师智慧和大型语言模型增强分析。

## 🎯 项目特色

- ✅ **科学严谨**：基于统计显著性检验，避免过拟合和数据挖掘偏差
- ✅ **理论完备**：整合现代投资组合理论(MPT)、贝叶斯推断、因果推断
- ✅ **大师智慧**：融入巴菲特、格雷厄姆、林奇等6位投资大师的可量化策略
- ✅ **AI增强**：支持ChatGPT、Gemini等大模型深度分析
- ✅ **开箱即用**：完整的数据获取、特征工程、回测、优化工作流
- ✅ **可复现**：引入Point-in-Time回测框架，彻底消除Look-Ahead Bias

## 📁 项目结构

```
myquant/
├── skill/              # quant-investor技能完整代码
│   ├── scripts/        # 核心脚本（数据获取、特征工程、回测等）
│   ├── references/     # 理论文档（统计学、贝叶斯、因果推断、大师智慧）
│   └── templates/      # 配置模板
├── examples/           # 示例代码和数据
└── docs/               # 详细文档
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install tushare yfinance pandas numpy scikit-learn scipy matplotlib seaborn
```

### 2. 配置数据源

复制配置模板并填入你的API Token：

```bash
cp skill/templates/config.ini.template config.ini
# 编辑config.ini，填入Tushare Token
```

### 3. 运行示例

```bash
# 获取数据
python skill/scripts/data_retrieval.py

# 特征工程
python skill/scripts/feature_engineering.py

# 模型训练
python skill/scripts/model_training.py

# 回测分析
python skill/scripts/point_in_time_data.py
```

## 📚 核心模块

### 数据层
- **Point-in-Time回测框架**：彻底消除Look-Ahead Bias
- **多源数据整合**：Tushare、yfinance、AKShare

### 特征工程
- 技术指标：MA、RSI、MACD、布林带等
- 基本面因子：PE、PB、ROE等
- 另类数据：情绪指标、资金流向

### 模型训练
- 传统机器学习：随机森林、XGBoost
- 深度学习：LSTM、Transformer
- 强化学习：DQN、PPO

### 统计验证
- **Bootstrap检验**：评估策略指标的置信区间
- **蒙特卡洛模拟**：压力测试和风险评估
- **IC分析**：因子有效性验证

### 投资组合优化
- **相关性分析**：构建真正分散化的组合
- **均值-方差优化**：寻找有效前沿
- **Black-Litterman模型**：融合主观观点与市场均衡

### 大师智慧
- **多维度评分系统**：综合6位大师策略对股票进行0-100分评分
- **动态权重调整**：根据市场周期自动调整各大师策略的权重

### LLM增强
- API调用：Gemini、GPT等
- 浏览器自动化：绕过配额限制
- 多角度分析："魔鬼代言人"辩论模式

## 📖 理论文档

- [统计检验方法论](skill/references/statistical_framework.md)
- [贝叶斯推断指南](skill/references/bayesian_guide.md)
- [因果推断指南](skill/references/causal_inference.md)
- [投资大师智慧](skill/references/master_investors_wisdom.md)
- [数据源参考](skill/references/data_sources.md)
- [特征库说明](skill/references/feature_library.md)
- [模型库参考](skill/references/model_zoo.md)

## ⚠️ 免责声明

本项目仅供学习和研究使用，不构成任何投资建议。

量化投资存在风险，历史回测结果不代表未来收益。请根据自身风险承受能力谨慎决策。

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

---

**如果这个项目对你有帮助，请给个⭐️Star支持一下！** 🚀
