# Q4: A/B Test 设计精讲 (A/B Testing Deep Dive)

> **类型**：Product / Case Study (套用 CHIME-D framework，但 A/B test 是其中 E 步骤的展开)
> **常见 follow-up 公司**：Meta、Airbnb、Netflix、Google、Uber、Pinterest
> **难度**：⭐⭐⭐⭐⭐ (Product DS / Senior DS 必考，差距来自细节深度)
> **典型题面**：
> - "How would you design an A/B test to evaluate a new feature?"
> - "PM 给你看 A/B 结果说 metric +5%，你怎么判断要不要 ship？"
> - "我们的 A/B 流量已经满了，怎么做更多 test？"

---

## 一、核心 mental model

A/B test 的本质是**因果推断的 RCT 形式**——random assignment ⇒ treatment 组和 control 组 in expectation 唯一区别是 treatment。但实际工业场景里有 5 大类问题会破坏这个保证：

| 问题 | 后果 | 解法 |
|------|------|------|
| Sample size 不够 / 周期不够 | 假阴性、novelty 误判 | Power analysis、长期 holdout |
| Multiple comparison | 假阳性 | Bonferroni / BH-FDR / 预注册 primary metric |
| Peeking | 假阳性 inflated | 预定 sample size 或 sequential test |
| Network effect / interference | bias 估计 | Cluster / switchback / ego-graph 随机 |
| Selection / SRM | bias | Instrumentation 监控、SRM check |

Senior DS 的差距 = 你能不能在 30 分钟里把这 5 类问题主动 surface 出来。

---

## 二、A/B Test 设计完整流程（10 步）

### Step 1: 明确 Hypothesis 和 Decision

不是 "我们试试这个 feature"，而是：
> "我假设 feature X 会让 user 在 7 天内 session count 增加 ≥ 2%，如果数据支持这个 effect，就 ship；否则 kill。"

**Senior signal**: 主动定义 **launch criteria**（不只是 p < 0.05，还要 effect size、guardrail）。

### Step 2: 选 Primary Metric

**好 metric 的标准 (OEC — Overall Evaluation Criterion)**：
1. **Sensitive**：能在合理样本量下检出实际 effect
2. **Causal proximity**: feature 改动和 metric 之间逻辑链短
3. **Long-term aligned**: 短期 metric 和长期目标方向一致
4. **Interpretable** to stakeholders

**常见误区**：
- 选了 noisy metric (e.g., revenue per user) → 检测不出来 → 换 proxy (conversion rate)
- 选了易优化 metric (e.g., page views) → 短期赢长期输

**经典 trap**：用 ratio metric (e.g., CTR = clicks / impressions) 时分子分母都是随机变量，不能直接 t-test，要用 **delta method** 或 **bootstrap** 算 variance。

### Step 3: Secondary + Guardrail Metrics

| 类型 | 用途 | 例子 |
|------|------|------|
| Primary | 决策 | 7-day session count |
| Secondary | 理解 mechanism | clicks, time on page, signup |
| Guardrail | 防副作用 | latency, error rate, ad revenue, complaint rate |
| North Star | 长期对齐 | 28-day retention, NPS |

### Step 4: Power Analysis (算 sample size)

公式（双样本均值检验）：
$$n = \frac{2 \sigma^2 (z_{1-\alpha/2} + z_{1-\beta})^2}{\Delta^2}$$

其中：
- $\sigma$ = baseline metric standard deviation
- $\Delta$ = MDE (Minimum Detectable Effect)
- $\alpha$ = 0.05, $\beta$ = 0.2 (power 80%) → $(z_{0.975} + z_{0.8})^2 ≈ 7.85$

**实际操作**：
1. 拉历史数据估 $\sigma$（per-user metric 的 std）
2. 和 PM 商量 MDE：能 detect 的最小 business meaningful effect (e.g., 1% retention lift)
3. 计算 $n$，再除以可用 traffic 算需要几天
4. **加 buffer**：novelty effect、weekday/weekend mix、节假日 → 至少 2 周

**实际坑**：
- $\sigma$ 不是常数，long-tail metric (revenue) 极不正态 → 用 bootstrap 估 variance
- Ratio metric 用 delta method
- Cluster randomization 时 $n$ 是 cluster 数，不是 user 数

### Step 5: Variance Reduction (省 sample size)

#### CUPED (Controlled-experiment Using Pre-Experiment Data)
原理：用 pre-experiment 数据做协变量调整，剔除 pre-existing variance：
$$Y'_i = Y_i - \theta (X_i - \bar{X})$$
其中 $X_i$ 是用户 pre-period 的相同 metric，$\theta = \text{Cov}(Y, X) / \text{Var}(X)$。

效果：variance reduction 30-50%，sample size 等比例下降。Microsoft、Netflix、Booking 都在用。

#### Stratification
按 user segment（geo、platform、tenure）分层，每层独立 randomize → 减少 imbalance 带来的 variance。

#### Switchback (for marketplace)
时间块交替 treatment / control，适合 ride-hailing、food delivery。

### Step 6: Randomization Unit（关键！）

| Unit | 适用 | 缺点 |
|------|------|------|
| User | 默认 | 简单 |
| Session | 用户内 carryover 小 | 同一用户不同 session 体验不同 |
| Cookie / device | 未登录 | logged-in 时 conflict |
| Cluster (e.g., city, group) | 有 network effect 的场景 | sample size 大幅减小 |
| Time-block (switchback) | marketplace | 时序 confounding |

> **Senior signal**：能识别题目要不要 cluster randomization。
>
> **判断**：treatment 是否会 spillover 到 control？
> - 推荐系统改 ranker → 不会 spillover (no)
> - 推荐"你朋友也喜欢"→ 会 spillover (yes, 朋友间相互影响)
> - 改 marketplace pricing → 会 spillover (供需联动)

### Step 7: Sample Ratio Mismatch (SRM) Check

**做法**：上线前几小时检查 control / treatment 流量比是否符合预期。

**统计方法**：chi-square test
$$\chi^2 = \sum \frac{(\text{observed} - \text{expected})^2}{\text{expected}}$$
p < 0.001 → SRM。

**如果 SRM 触发**：**绝对**不要继续看 metric，先排查：
- 流量分配 bug
- 不同组用户被 filter 概率不同（e.g., 新版本 app 才能进 treatment → biased sample）
- Bot / spam 在两组分布不同

**真实案例**：Microsoft 报告 ~6% 的 A/B test 有 SRM，多数被忽视导致错误结论。

### Step 8: Run experiment + 监控

- 每天监控 (但不能用来决策)：流量、SRM、guardrail metric
- 看 dashboard 不算 peeking（peeking 是基于 primary metric 决策提前停止）
- 如果 critical bug，可以提前 kill；不能因为"看上去赢了"就提前 ship

### Step 9: Analysis

#### t-test (continuous metric, normal-ish)
$$t = \frac{\bar{Y}_T - \bar{Y}_C}{\sqrt{s_T^2 / n_T + s_C^2 / n_C}}$$

#### Bootstrap (for ratio metric, skewed distribution)
- Resample with replacement, 计算 metric
- 1000+ bootstrap samples 得 empirical 95% CI

#### Delta method (for ratio metric)
$$\text{Var}\left(\frac{X}{Y}\right) ≈ \frac{1}{\bar{Y}^2} \text{Var}(X) + \frac{\bar{X}^2}{\bar{Y}^4} \text{Var}(Y) - 2 \frac{\bar{X}}{\bar{Y}^3} \text{Cov}(X, Y)$$

### Step 10: Decision

不要只看 p-value。Senior DS 的决策矩阵：

| 情况 | 决定 |
|------|------|
| Primary 显著 + Guardrail OK + Effect ≥ MDE | ✅ Ship |
| Primary 显著但 effect < MDE | ❓ Cost-benefit 分析，可能不值得 ship |
| Primary 显著 + Guardrail 严重恶化 | ❌ 不 ship 或迭代 |
| Primary 不显著 + 趋势对 + 流量不足 | ⏳ 加 sample 或长期 follow-up |
| Primary 不显著 + 流量充足 | ❌ Kill |
| Primary 显著但太好 (Twyman's law) | 🚨 排查 instrumentation / SRM |

---

## 三、5 大经典 Pitfall（Senior 必谈）

### Pitfall 1: Multiple Comparison

**问题**：同时看 10 个 metric，每个 α=0.05，至少一个假阳性概率 = 1 - 0.95^10 ≈ 40%。

**解法**：
- **Bonferroni**：α / k (保守)
- **Benjamini-Hochberg**: 控制 False Discovery Rate (FDR)
- **预注册 single primary metric**：只有 primary 能决定 ship

### Pitfall 2: Peeking

**问题**：每天看 metric，一旦 p < 0.05 就 stop → false positive rate 远超 5%。

**解法**：
- 预定 sample size，到了再 analyze
- **Sequential testing** (Always-Valid CI): mSPRT、Bayesian sequential、alpha spending function (O'Brien-Fleming)

### Pitfall 3: Novelty & Primacy Effect

| Effect | 现象 | 时间尺度 |
|--------|------|----------|
| **Novelty** | 新 feature 短期吸引点击，长期回归 | 1-4 周衰减 |
| **Primacy** | 老用户不适应，短期 negative，长期接受 | 1-4 周回升 |

**解法**：
- 至少 2 周 (覆盖 weekly cycle 和初期 novelty)
- 看 daily effect curve 是否 stable
- 长期 holdout cell (1% 永远 control，6 个月对比)
- Segment by user tenure（新 vs 老用户）

### Pitfall 4: Network Effect / Interference

**问题**：treatment 的 user 影响 control 的 user，违反 SUTVA 假设。

**例子**：
- Facebook 给 A 推 better feed，A 多发 post → B (control) 也看到更多内容 → control 也被 treat 了
- Uber 给 driver A 涨 incentive → city-wide supply ↑ → control driver wait time ↓
- Marketplace pricing change

**解法**：
- **Cluster randomization**: 按 city、country、social cluster 随机
  - 代价：cluster 数小，power 低
- **Ego-cluster** (Facebook): 一个 user 和他的所有朋友是同一个 cluster
- **Switchback**: 时间块交替 (Doordash、Uber 用)
- **Two-sided experiment**: 同时 randomize 两边
- **Synthetic control / DiD**: 当 cluster 数太少 (详见 Q6)

### Pitfall 5: Heterogeneous Treatment Effect (HTE)

**问题**：average effect = +1%，但 70% 用户 -2%，30% 用户 +9% → ship 后伤多数。

**解法**：
- Pre-specified segment analysis (geo, tenure, platform)
- Causal forest / X-learner / uplift modeling 找 CATE (Conditional Average Treatment Effect)
- 决策：给 30% 受益用户 ship；其他保持 control（personalized treatment）

---

## 四、A/B test 不可行 / 不够时

1. **流量不够**：用 CUPED + stratification 减 variance；调高 MDE
2. **网络效应**：cluster A/B 或 switchback
3. **Marketplace-wide change** (e.g., 全 city 调价)：geo-experiment + synthetic control
4. **Long-term effect**：1% holdout cell 长期不切回
5. **伦理 / 法规**：observational study + causal inference (Q6)
6. **Test 队列堵了**：interleaving (推荐系统专用) — 一次 query 同时展示 model A 和 B 的结果

---

## 五、常见面试 Follow-up

### Q: "PM 说 metric +5% p=0.03，要不要 ship？"

**Senior answer**：
1. Sample size 是 pre-specified 的吗 (vs peeking)？
2. SRM 检查过了吗？
3. Effect 在不同 segment 一致吗 (HTE)?
4. Primary 之外 secondary / guardrail 怎么样？
5. 持续了多久 — 有 novelty 嫌疑吗？看 daily curve
6. Multiple comparison adjusted 了吗？
7. Effect 量级和 historical lift 比是不是 too good (Twyman's law)？

### Q: "怎么测 long-term effect？"

- Long-term holdout: 1% 永久 control vs 1% 永久 treatment，6 个月看 retention
- Cohort analysis: 看 same-tenure user 在 treatment 下后续 retention
- Surrogate index: 用短期 metric 组合预测长期 retention (Athey et al.)

### Q: "新功能上线后我看了 50 个 metric，有 3 个显著，可以 ship 吗？"

- 50 个 metric × 5% α = 期望 2.5 个假阳性，3 个不奇怪
- 必须区分 primary（pre-registered）vs exploratory
- Primary 是哪个？显著吗？
- Exploratory 用 BH-FDR 修正

### Q: "ABA test 是什么，用处？"

- 把流量分成 A1, A2, B 三组，A1 和 A2 都是 control
- 如果 A1 vs A2 也显著差异 → 系统有 bug 或 metric noise 太大
- Sanity check 工具

### Q: "如果你只有 1000 用户怎么 A/B？"

- Power analysis: 1000 user 能 detect 的 MDE 多大？
- 如果 MDE 太大 → 不能用 A/B
- 替代：observational + causal inference (Q6)、structured user research

---

## 六、Senior Trade-off 总结

| 决策 | Option A | Option B | 怎么选 |
|------|---------|---------|--------|
| 长期 vs 短期 metric | session count (短) | 28d retention (长) | 都要，retention 是 north star |
| Variance reduction | CUPED | stratification | CUPED 简单好用，组合更好 |
| Randomization unit | user | cluster | 看 spillover 风险 |
| Test 持续时间 | 1 周 | 4 周 | 至少 2 周覆盖 weekly cycle 和 novelty |
| Stop rule | fixed sample | sequential | sequential 更高效但要专门工具 |

---

## 一句话答案 (Elevator Pitch)

> "好的 A/B test 设计有 10 个步骤：定 hypothesis 和 launch criteria、选 OEC primary metric、guardrail metrics、power analysis 算 sample size、用 CUPED 或 stratification 降 variance、决定 randomization unit (是否要 cluster)、SRM check 防 instrumentation bug、run + 监控但不 peek、用 t-test/bootstrap/delta method 做合适的统计推断、综合 effect size + guardrail 做决策。Senior 必须主动 surface 5 类经典 pitfall：multiple comparison、peeking、novelty/primacy、network effect、heterogeneous treatment effect。当 A/B 不可行时（流量不够、有 spillover、市场级 intervention），用 cluster randomization、switchback、或转向 quasi-experiment / causal inference。"
