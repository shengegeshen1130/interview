# 通用答题框架 (Universal Frameworks for DS System Design)

> **适用场景**：本文档为 senior DS / Applied Scientist / MLE 面试中两类高频题的通用答题框架。后续每道题都套用以下框架，请先精读本章。

---

## 一、面试题分类 (Question Taxonomy)

大厂 senior DS 面试中的"系统设计"通常分两类：

| 类型 | 典型题目 | 考察重点 | 时长 |
|------|---------|---------|------|
| **A. ML System Design** | "设计 YouTube 推荐系统" / "设计 Ad CTR 预估" | end-to-end ML 系统能力：metrics、data、model、serving、monitoring | 45-60 min |
| **B. Product / Case Study** | "DAU 下降 5%，怎么排查" / "怎么验证新功能上线效果" | 商业 sense + 统计严谨度 + causal 思维 | 30-45 min |

> **Senior 信号**：面试官期待你**主动识别题目类型**，并在前 2 分钟说出框架。不要直接进入"我会用 XGBoost"。

---

## 二、ML System Design 框架（7-step）

```
1. Problem Clarification  →  2. Metrics  →  3. Data  →  4. Features
                                  ↓
8. Monitoring & Iteration  ←  7. Evaluation  ←  6. Serving  ←  5. Model
```

### Step 1: Problem Clarification（问清楚！）
**目标**：把模糊的 prompt 转化为 well-defined ML 问题。Senior 候选人 80% 的差距来自于这一步。

必问的 clarifying questions：
- **Business goal**: 真正要优化的是什么？(engagement? revenue? retention? 用户满意度?)
- **Scope**: 哪个产品/场景/用户群？(全球 vs 单一 market、新用户 vs 老用户)
- **Scale**: DAU、QPS、item pool size、latency budget (e.g., 推荐 < 200ms)
- **Constraints**: 隐私、公平、可解释性、监管、cost budget
- **Success criteria**: launch criteria 是什么？(primary metric +X%, guardrail 不掉)

⚠️ **典型陷阱**：直接把 "engagement" 当目标。要追问"哪种 engagement？短期 click 还是长期 retention？"

### Step 2: Metrics（offline / online / guardrail 三层）
- **Offline metrics**：用 historical data 评估 model（AUC、NDCG@K、Recall@K、log loss、calibration）
- **Online metrics**：A/B test 中观察的（CTR、watch time、retention、revenue）
- **Guardrail metrics**：不能恶化的副作用指标（latency、diversity、creator-side metrics、misinformation rate、user satisfaction survey）

> **Senior 关键点**：discuss **metric mismatch**——offline AUC 提升 ≠ online CTR 提升的常见原因（distribution shift、feedback loop、selection bias、position bias）。

### Step 3: Data
- **Sources**: logs（client-side / server-side）、CRM、third-party
- **Labels**: 
  - Implicit (click, watch, dwell time) — 廉价但 noisy
  - Explicit (rating, survey) — 高质但稀疏
  - **Discuss label leakage & delayed labels** (e.g., 转化要 7 天才到)
- **Sampling**: down-sampling negatives + 校正、stratified sampling、recency weighting
- **Quality**: instrumentation 准确性、bot filtering、duplicate handling

### Step 4: Features
分四类列举（避免遗漏）：

| 类别 | 例子 |
|------|------|
| User | demographic, history embedding, recent activity, lifetime value |
| Item | content embedding, category, freshness, popularity |
| Context | time of day, device, location, network condition |
| Cross / Interaction | user-item past interaction, similarity |

**Senior 加分点**：feature store（Feast、Tecton），online/offline feature consistency（training-serving skew 是 ML 系统最常见的 bug 之一）。

### Step 5: Model
- **Baseline first**：popular item、logistic regression、collaborative filtering
- **Architecture choice 取决于问题**：
  - 召回（candidate generation）: two-tower / matrix factorization / ANN (FAISS, ScaNN)
  - 排序（ranking）: GBDT (LightGBM) / DNN / DLRM / Transformer
  - 多目标：MMoE、PLE、shared-bottom
- **Training**：loss function（pointwise / pairwise / listwise）、negative sampling、class imbalance
- **Hyperparameter / regularization** 简单提一下，别陷入细节

### Step 6: Serving
- **架构**：candidate generation → ranking → re-ranking (diversity, business rules)
- **Latency budget 拆解**：feature fetch / model inference / post-processing
- **Caching**：user embedding 离线算、ANN index 预构建
- **Cold start**：content-based、contextual bandit、热门 fallback
- **A/B testing infrastructure**：traffic split、shadow mode、ramp-up

### Step 7: Evaluation
- **Offline**: 离线指标 + segment 分析（new vs power users, geo, device）
- **Online A/B**: 详见 Q4
- **Counterfactual / off-policy evaluation**: IPS、doubly robust（在 A/B 之前判断是否值得做）

### Step 8: Monitoring & Iteration
- **Data drift**: PSI、KL divergence on feature distribution
- **Model drift**: prediction distribution、calibration plot
- **Business metric**: dashboard + alert
- **Feedback loop & bias**: position bias、popularity bias、exposure bias 怎么 debias（IPW、causal embedding、随机 exploration slot）
- **Retrain cadence**: daily / weekly / online learning，trade-off 是 freshness vs stability

---

## 三、Product / Case Study 框架（CHIME-D）

我用 **CHIME-D** 帮助记忆：

| 步骤 | 内容 |
|------|------|
| **C** - Clarify | 业务背景、metric 定义、时间窗口、scope |
| **H** - Hypothesize | 列 3-5 个候选假设，按可能性排序 |
| **I** - Investigate | 用数据验证假设：segment、漏斗、对比 |
| **M** - Measure | 选指标（primary / secondary / guardrail），定义可量化 |
| **E** - Experiment | A/B、quasi-experiment、causal inference |
| **D** - Decide & Communicate | 决策框架、权衡、stakeholder 沟通 |

### Clarify 阶段必问：
- 这个指标是**怎么定义的**？（DAU = unique users / day? 用 server log 还是 client log? 去 bot 了吗?）
- **时间窗口**？（Day-over-day? Week-over-week? YoY? holiday-adjusted?）
- **scope**？（全球 vs 某 region? 某 segment? 某 platform?）
- **business context**？（最近有 launch 吗? infra change 吗? 竞品动作?）

### Hypothesize 阶段（结构化）：
按"内部 vs 外部 × 短期 vs 长期"两维分类：

|              | 内部 (我们能控制) | 外部 (我们控制不了) |
|--------------|------------------|----------------------|
| **短期**     | bug、A/B ramp、infra outage、营销活动结束 | 节假日、新闻事件、竞品 launch |
| **长期**     | 产品质量下滑、价格调整、target market shift | 行业趋势、宏观经济、监管 |

### Investigate 阶段（segment 分析模板）：
- **维度**：geo、platform、app version、user cohort（new / dormant / active / power）、feature usage、acquisition channel
- **方法**：找出"哪个 segment 跌得最狠"——80% 的问题靠 segment 就能定位
- **配套指标**：DAU 跌的同时，session 数、页面浏览、注册数怎么样？（联动指标可以排除 instrumentation 问题）

### 常见陷阱（Senior 一定要 call out）：
- **Simpson's paradox**：总体趋势和 segment 趋势相反 → segment level reasoning
- **Sample Ratio Mismatch (SRM)**：A/B 流量分配不均 → instrumentation bug
- **Twyman's law**："Anything surprising is probably wrong" → effect 大得离谱时先怀疑数据
- **Survivorship bias**: 只看留存用户得出错误结论
- **Novelty / Primacy**: 短期 effect ≠ 长期 effect

---

## 四、时间分配建议（45 分钟 ML SD）

| 阶段 | 分钟 | 注意 |
|------|------|------|
| Clarification + scoping | 5-7 | 不要省，senior 信号 |
| Metrics 定义 | 3-5 | offline + online + guardrail 三层 |
| Data + features | 5-7 | 包含 label 噪音、leakage |
| Model architecture | 8-10 | baseline → improvement |
| Serving + scaling | 5-7 | latency budget、cold start |
| Evaluation + A/B | 5-7 | 要谈 long-term metric |
| Monitoring + iteration | 3-5 | drift、bias、retrain |
| Trade-offs + Q&A | 余下 | 主动 surface trade-off |

---

## 五、面试官的"Senior 信号"清单

面试官心里 checklist（你要主动 hit 这些 signal）：

- [ ] 主动识别 ambiguity 并问 clarifying questions
- [ ] 提出至少 2 种解法，并 discuss trade-off
- [ ] 主动谈 failure mode（bias、drift、cold start、latency）
- [ ] 区分 short-term vs long-term metric
- [ ] 提到 monitoring & iteration（不止 launch 一次就完）
- [ ] 谈 cross-functional concern（product、legal、infra）
- [ ] 用具体数字举例（"假设 DAU 1B、QPS 50K"）
- [ ] 主动 surface trade-off：accuracy vs latency / freshness vs stability / personalization vs diversity

---

## 六、常见 Anti-pattern（不要犯）

1. **Jump to model**：还没问 business goal 就开始 "I would use Transformer"
2. **Over-engineer**：一上来就 LLM、causal forest——baseline 都没建
3. **Ignore data quality**：不谈 label noise、bot、bias
4. **Single metric**：只谈 CTR 不谈 long-term retention 和 guardrail
5. **No A/B plan**：模型训好就完了，不谈怎么 ship
6. **Ignore monitoring**：launch 完就 declare victory
7. **Memorized solution**：照搬某博客架构而不解释为什么

---

> 后续 6 道题都遵循以上框架。每题会标注用的是 ML SD 框架还是 CHIME-D 框架。
