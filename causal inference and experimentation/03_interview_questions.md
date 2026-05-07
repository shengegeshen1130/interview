# 因果推断 & 实验设计 — 高频面试题 (Interview Questions)

> 整理自 FAANG、Marketplace、量化、咨询等公司的真实面试题。每题给出 *简答* + *follow-up 提示*。

---

## 目录

- [Part A：基础概念 (Conceptual)](#part-a基础概念-conceptual)
- [Part B：A/B 测试设计 (Experiment Design)](#part-bab-测试设计-experiment-design)
- [Part C：统计计算 (Statistics & Math)](#part-c统计计算-statistics--math)
- [Part D：因果推断方法 (Causal Methods)](#part-d因果推断方法-causal-methods)
- [Part E：诊断与排错 (Debug & Postmortem)](#part-e诊断与排错-debug--postmortem)
- [Part F：开放性产品题 (Product Sense)](#part-f开放性产品题-product-sense)
- [Part G：编码题 (Coding)](#part-g编码题-coding)

---

## Part A：基础概念 (Conceptual)

### Q1. 什么是 correlation 和 causation 的区别？为什么"correlation ≠ causation"？

**回答：**
1. Correlation 描述两变量统计关联，causation 描述一个变量 *引起* 另一个变化。
2. 三种 correlation 但 *非* causation 的常见情况：
   - **Confounding** (`X ← Z → Y`)：第三变量同时影响 X 和 Y。
   - **Reverse causation**：实际是 Y → X。
   - **Selection bias / collider**：条件化在 collider 上引入虚假关联。

> **Follow-up 提示：**
> - 给一个具体例子（冰淇淋销量 vs 溺水率：confounder = 气温）。
> - 如何用 RCT 解决？

---

### Q2. 解释 Potential Outcomes 框架。

**回答：**
1. 每个个体有两个潜在结果 `Y(0)`、`Y(1)`，对应不接受 / 接受 treatment。
2. **Fundamental Problem**：永远只能观测到一个 → 因果推断本质是 *missing data problem*。
3. **ATE** = `E[Y(1) − Y(0)]` 是关心的量。

> **Follow-up 提示：**
> - 为什么 RCT 能识别 ATE？因为随机化使 `(Y(0), Y(1)) ⊥ T`。
> - 写出 ATE / ATT / CATE 的差异。

---

### Q3. 解释 SUTVA。什么时候它会失效？

**回答：**
SUTVA 包含两条假设：
1. **No interference**：个体 i 的 outcome 只依赖 *自己的* treatment，不受他人影响。
2. **Consistency**：treatment 没有 hidden versions。

**失效场景：**
- 社交网络功能（朋友推荐）→ 干扰传染。
- Marketplace（Uber/Airbnb）：treatment 改变价格/匹配，影响 control 用户。
- 流行病（vaccination）：群体免疫降低未接种者风险。

> **Follow-up 提示：**
> - 如何应对？答 cluster randomization、switchback、ego-network。

---

### Q4. 什么是 confounder？为什么必须控制？

**回答：**
Confounder = 同时影响 treatment 和 outcome 的变量（fork 结构 `T ← Z → Y`）。

**不控制 confounder 会：**
- 造成 *偏差* 估计（biased ATE）。
- 经典例：吸烟 → 肺癌；confounder = 基因。

> **Follow-up 提示：**
> - DAG 上画出 confounder 与 mediator 的区别。
> - mediator 不能控制（会 block 因果路径）。

---

### Q5. 什么是 Simpson's Paradox？举一个例子。

**回答：**
当各 *子组* 的趋势与 *整体* 趋势相反时出现。

**经典例：** UC Berkeley 1973 录取性别歧视诉讼 —— 整体看女性录取率低，但每个 department 看女性录取率都 *较高或相当*。
**原因：** 女性更多申请竞争激烈的院系。

> **Follow-up 提示：**
> - 关键是 *什么* confounder（这里是 department）造成了反转。
> - 如何避免？分层 (stratify) + DAG 思考。

---

## Part B：A/B 测试设计 (Experiment Design)

### Q6. 你怎么设计一个 A/B test 来评估"新搜索算法"？

**回答模板：**

1. **目标**：明确 OEC（如 search satisfaction = (clicks + 0.5·dwell_time) / queries）。
2. **Hypothesis**：H₀ = 新算法不优于旧；H₁ = 新算法的 OEC 更高。
3. **Metrics**：
   - Primary: OEC
   - Secondary: CTR、average dwell time
   - Guardrail: latency p95、error rate
   - Counter: query volume（avoid gaming）
4. **Randomization**：user-level（确保同一用户体验一致）。
5. **Sample size**：计算 MDE → 如想检测 1% 相对提升、baseline 3 queries/session、std=2，→ N≈140k per group。
6. **Duration**：考虑 weekly seasonality → 至少 2 weeks。
7. **Pre-experiment checks**：A/A test、SRM。
8. **Analysis**：双侧 t-test、CUPED 减方差、segment 分析（new/old user）。
9. **Decision**：综合 primary + guardrail，做决策。

> **Follow-up 提示：**
> - 如果只能 1 周怎么办？答：增加流量比例 / 用 CUPED。
> - 怎么处理 novelty effect？

---

### Q7. 怎么决定样本量？

**回答：**
公式：`N ≈ 16 · σ² / Δ²`（α=0.05, power=0.8, two-sided）

**输入：**
1. **基线方差 σ²**：从历史数据估计。
2. **MDE Δ**：业务可接受的最小有意义效应。
3. **α** = 0.05（默认）、**power** = 0.8（默认）。

**容易忽略的点：**
- 比例指标用 `p(1-p)` 代替 σ²。
- 比率指标（CTR = clicks / impressions）要用 *delta method*。
- Cluster randomization → 加 design effect = `1 + (m̄−1)·ρ`。

> **Follow-up 提示：**
> - 如果 MDE 给得太小？样本量会膨胀（成本高）。
> - 怎么从 product 角度选 MDE？看 baseline lift 历史 + 业务影响。

---

### Q8. CUPED 是什么？为什么有效？

**回答：**
**CUPED** (Controlled-experiment Using Pre-Experiment Data) 用实验前的协变量来减小 outcome 的方差：

```
Y_adj = Y − θ·(X − E[X])    其中 θ = Cov(X,Y)/Var(X)
```

**有效原因：**
- 不影响 ATE 的无偏性（因为 X 实验前就固定）。
- 方差降低为 `Var(Y) · (1 − ρ²)`，其中 `ρ = Corr(X, Y)`。

**实务：** 用同一指标的 pre-period 值做 X，通常 `ρ` 0.5-0.8，→ 样本量节省 25-64%。

> **Follow-up 提示：**
> - 为什么 X 必须 *实验前* 测量？答：避免被 treatment 影响 → 否则会改变 ATE。
> - 二分类指标怎么 CUPED？答：可用 logistic CUPED 或 stratified estimator。

---

### Q9. 什么是 Sample Ratio Mismatch (SRM)？怎么诊断？

**回答：**
**SRM** = 实际分配比例与设计比例显著偏离（如设计 50/50，实际 49.2/50.8）。即使是 *轻微* 偏离也能让结果不可信，因为它说明随机化失败了。

**诊断：**
```python
from scipy.stats import chisquare
chi2, p = chisquare([n_T, n_C], [(n_T+n_C)/2, (n_T+n_C)/2])
# p < 0.001 → 严重 SRM
```

**常见原因：**
1. Bot/crawler 过滤不一致。
2. Bucketing bug。
3. Redirects 导致用户掉队。
4. 不同平台兼容性差异。

> **Follow-up 提示：**
> - SRM 时的临时建议：**不要相信** treatment effect。先修因。
> - Kohavi 比喻 SRM 像"安全带"。

---

### Q10. Novelty effect 与 primacy effect 区别？

**回答：**

| | Novelty | Primacy |
|---|---|---|
| 早期反应 | 高（新鲜感） | 低（不适应） |
| 长期 | 衰减回 baseline | 上升至稳态 |
| 例 | 新 UI、新功能 | 改变熟悉的工作流 |

**应对：**
1. 跑 4+ 周看趋势是否稳定。
2. 分新/老用户分析。
3. **2-stage analysis**: pre-period & post-period 拆分 holdout。

> **Follow-up 提示：**
> - 怎么 *统计上* 检测？答：分时段拟合趋势，看 treatment effect 随时间斜率。

---

## Part C：统计计算 (Statistics & Math)

### Q11. p-value 的精确含义是什么？

**回答：**
> "p-value 是 *假设 H₀ 真* 时，看到当前 (或更极端) 检验统计量的概率。"

**常见错误：**
- ❌ p-value 是 H₀ 真的概率。
- ❌ p-value 是 H₁ 真的概率。
- ❌ 1−p 是 effect 真实存在的概率。

> **Follow-up 提示：**
> - Bayesian 视角：p-value vs posterior probability of H₁。
> - 为什么"p < 0.05" 不等于 effect 大？

---

### Q12. Type I vs Type II error；power？

**回答：**

**核心定义：**
- **Type I error (α)**：假阳性 (false positive) —— H₀ 真但被拒绝。"误判 treatment 有效"。
- **Type II error (β)**：假阴性 (false negative) —— H₁ 真但没拒绝。"漏检了真实效应"。
- **Power = 1 − β**：H₁ 真时检测到的概率。

**4 种结果矩阵：**

|  | H₀ 真 | H₁ 真 |
|---|---|---|
| 拒绝 H₀ | ❌ Type I (α) | ✅ Power (1−β) |
| 不拒绝 H₀ | ✅ 1−α | ❌ Type II (β) |

**关键关系：**

1. **α ≠ 1 − β**！它们在不同分布下计算（α 在 H₀，β 在 H₁）。
2. **固定 N 时 α 和 β 是 trade-off**：缩小拒绝域 → α↓ β↑；扩大 → α↑ β↓。
3. **增大 N 同时降低 α 和 β**（standard error ↓ → 两分布重叠减少）。
4. **Effect size Δ 越大 → β 越小** → power 越高。

**Power 增加因素：**
```
N ↑、|Δ| ↑、σ ↓、α ↑ (放宽接受假阳)
```

**业务上的代价权衡：**

| 场景 | Type I 代价 | Type II 代价 | 推荐 |
|---|---|---|---|
| 支付/登录关键功能 | 高 | 中 | α=0.01, β=0.2 |
| 一般产品功能 | 中 | 中 | α=0.05, β=0.2（默认）|
| 增长实验（多轮） | 低 | 高 | α=0.1, β=0.1 |
| 医疗/安全 | 极高 | 高 | α=0.001, β=0.1 |

**直观类比（推荐用法庭）：**
> H₀ = "被告无罪"。Type I = 冤枉好人；Type II = 放走凶手。
> 法律选择小 α（"宁可放过，不可冤枉"）；癌症筛查选择小 β（"宁可误诊，不可漏诊"）。

**常见误区：**
- ❌ "α + β = 1"（错，不互补）
- ❌ "p < 0.05 就一定有真效应"（不是 H₁ 真的概率）
- ❌ "p > 0.05 就是没效应"（可能只是 power 不够）

> **Follow-up 提示：**
> - "Peeking 怎么影响 Type I？" → 偷看 10 次，名义 α=0.05 实际可达 19%。
> - "Multiple testing？" → m 个测试 FWER ≈ `1 − (1−α)^m`，要 Bonferroni / BH 校正。
> - "为什么 power 默认 0.8？" → Cohen 1988 经验值，平衡 N 与漏检风险；高风险场景应该用 0.9 或 0.95。
> - "为什么 α 默认 0.05？" → Fisher 1925 任意选的；现代实验平台开始用更严的 α=0.01 + sequential testing。

---

### Q13. 解释 Central Limit Theorem (CLT) 在 A/B test 中的作用。

**回答：**
样本均值 `X̄` 在 N 大时近似服从正态：`X̄ ~ N(μ, σ²/N)`。

→ 即使 outcome 本身不是正态（如 binary），样本均值仍可用 z/t-test。

**实务注意：**
- N 太小或 outcome 极度偏态（如 收入有 long tail）→ CLT 收敛慢。
- 解决：log transform、bootstrap、winsorize。

> **Follow-up 提示：**
> - 二项分布 CLT 收敛大概需要 `N·p > 5` 且 `N·(1-p) > 5`。

---

### Q14. 给定 conversion rate 5%，要检测 5%→5.5% 的提升，需要多少样本？

**计算：**
- `Δ = 0.005`
- `σ² ≈ p(1-p) ≈ 0.05 · 0.95 ≈ 0.0475`
- `N = 16 · σ² / Δ² = 16 · 0.0475 / 0.000025 = 30,400 per group`

```python
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
es = proportion_effectsize(0.05, 0.055)
n = NormalIndPower().solve_power(es, alpha=0.05, power=0.8, alternative='two-sided')
# n ≈ 30,000+
```

> **Follow-up 提示：**
> - 检测相对 1% lift 比绝对 1pp 难得多。
> - 业务上 baseline 越低 → MDE 难达到 → 需更多流量。

---

### Q15. 什么是 multiple testing？怎么校正？

**回答：**
测试 m 个假设，至少一个误报概率 ≈ `1 − (1−α)^m`。当 `m=20, α=0.05` → 64% 至少一个假阳！

**校正方法：**
| 方法 | 控制 | α 调整 | 性质 |
|---|---|---|---|
| **Bonferroni** | FWER | `α/m` | 保守 |
| **Holm** | FWER | step-down | 比 Bonferroni 强 |
| **BH (Benjamini-Hochberg)** | FDR | `k·α/m` | 业界常用 |

> **Follow-up 提示：**
> - OEC 不需要校正（只一个）；但 secondary metrics 多则要 FDR 校正。

---

## Part D：因果推断方法 (Causal Methods)

### Q16. RCT 不能做时，怎么估计因果效应？

**回答：**
按照能识别的假设强度递减：

1. **DiD**：有 panel data + 平行趋势假设。
2. **RD**：有明确 cutoff。
3. **IV**：有合法工具 (relevance + exclusion + independence)。
4. **PSM / IPW / Doubly Robust**：假设可观测 confounder 完整。
5. **Synthetic Control**：单一 treated 单元 + 多 control。
6. **Sensitivity Analysis**：评估 unobserved confounder 的影响幅度。

> **Follow-up 提示：**
> - 给出 *具体场景* 选 *具体方法*。
> - 强调每个方法的核心假设。

---

### Q17. 解释 Propensity Score Matching。

**回答：**
1. **Propensity score**：`e(X) = P(T=1 | X)`，给定协变量下接受 treatment 的概率。
2. **关键定理 (Rosenbaum-Rubin 1983)**：在给定 X 的 ignorability 下，给定 `e(X)` 的 ignorability 也成立 → 可用 1D 标量代替高维匹配。
3. **流程**：
   - 估计 PS（logistic regression / GBM）。
   - 1:1 nearest neighbor / kernel / stratification 匹配。
   - 检查 covariate balance（standardized mean diff < 0.1）。
   - 估计 ATT。

**局限：**
- 不能解决 *未观测* confounder。
- Common support 不足时不能比较。

> **Follow-up 提示：**
> - PSM 估计的是 ATT 还是 ATE？取决于匹配方向（treated → control 是 ATT）。
> - 怎么检查 balance？SMD、QQ plot。

---

### Q18. Difference-in-Differences 的关键假设？

**回答：**
**平行趋势 (Parallel Trends Assumption)**：在 *没有 treatment* 时，treatment 组和 control 组的 outcome 会 *平行* 演化。

**ATT 公式：**
```
ATT = [E(Y_post|T=1) − E(Y_pre|T=1)] − [E(Y_post|T=0) − E(Y_pre|T=0)]
```

**检验平行趋势：** 用 *pre-period* 的多期数据，画两组趋势线 → 应平行。

**陷阱：**
- 不要拿 *post-period* 的趋势来支持平行假设（这是被 treatment 污染的）。
- TWFE 在 staggered treatment 时失效（Goodman-Bacon 2021）。

> **Follow-up 提示：**
> - PSM + DiD 联合方法？
> - 现代替代：Callaway-Sant'Anna estimator。

---

### Q19. 什么是 Instrumental Variable？给三个条件。

**回答：**
**IV** = 满足以下三条的变量 Z：
1. **Relevance**: `Cov(Z, T) ≠ 0`（影响 T）。
2. **Exclusion**: Z 只通过 T 影响 Y（无直接路径）。
3. **Independence**: Z 与 unobserved 误差独立。

**估计方法：** 2SLS（two-stage least squares）。

**LATE：** IV 估计的是 *complier subset* 上的效应（不是 ATE）。

**经典例子：**
- 距大学远近 → 学历（Card）。
- 抽签 → 投保（Oregon Health Experiment）。

> **Follow-up 提示：**
> - "Weak instrument" 是什么？答：first-stage F-statistic < 10 → 偏差大。
> - 怎么检验 exclusion？无法直接检验，只能讲故事。

---

### Q20. Heterogeneous Treatment Effects (HTE) 怎么估计？

**回答：**
- **目标**：估计 `τ(x) = E[Y(1) − Y(0) | X = x]` 即 CATE。
- **Meta-learners**：
  - S-learner: 单模型 + treatment 作为 feature。
  - T-learner: 两模型，差为 CATE。
  - X-learner: 不平衡数据更鲁棒。
  - R-learner: residualization。
- **Causal Forest** (Athey & Wager)：honest 树 → 每叶节点估计 CATE。
- **Uplift Modeling**：营销中 CATE 的别名 —— 找 persuadable 用户。

```python
from econml.dml import CausalForestDML
est = CausalForestDML(model_y=..., model_t=...).fit(Y, T, X=X, W=W)
cate = est.effect(X_test)
```

> **Follow-up 提示：**
> - 怎么 *验证* CATE？答：HTE eval 困难（无 ground truth），用 uplift curve、Qini coefficient。

---

## Part E：诊断与排错 (Debug & Postmortem)

### Q21. 你的 A/B test 显示巨大正向效应（+30%），你会怎么验证？

**回答：**
1. **SRM 检验**：T:C 比例是否 50/50？
2. **A/A test**：同一 variant 之间是否也"显著"？
3. **检查 metric 计算**：分子分母对吗？bot 过滤一致？
4. **Pre-period analysis**：检查 pre-period treatment 组是否本来就高。
5. **Segment 分析**：是否某 segment（如新用户）异常贡献？
6. **持续时长**：是否短期 novelty？跑长一点看是否稳定。
7. **业务直觉**：30% 是否合理？历史 lift 多大？
8. **Replicate**：在不同时间或地区再做。

> **Follow-up 提示：**
> - 如果 30% 是真的为什么不直接发布？答：ROI 高但需 *验证*；可能是埋点/bug。

---

### Q22. 第一周 lift 显著，第二周消失了。可能原因？

**回答：**
1. **Novelty effect** 衰减。
2. 用户构成变化（第一周新用户多 → 后续老用户进入）。
3. 季节性（节假日 vs 平日）。
4. Bug：第一周 instrumentation 差异。
5. Regression to the mean（如果第一周是 outlier）。
6. 第三方因素（竞品、新闻、节日）。

> **Follow-up 提示：**
> - 怎么决定？答：通常以更长期数据为准，或用 segment 拆解。

---

### Q23. Treatment 组比 Control 组多了 1% 的样本。问题大吗？

**回答：**
**很大。** 这就是 SRM。即使 1% 也能让结果完全错误，因为它意味着随机化失败 → 两组 *不可比*。

**如果是 50/50 设计 + N=1M：** chi-squared p-value 应该 ≈ 1e-100，绝不会自然发生。

> **Follow-up 提示：**
> - 1% 样本量差异 vs 1% effect size：完全不同概念！
> - 修因后重启实验。

---

## Part F：开放性产品题 (Product Sense)

### Q24. 你怎么衡量"YouTube 推荐算法变好了"？

**回答框架：**
1. **明确目标**：用户满意度 / 长期价值（不是 CTR）。
2. **Primary metric**：watch time per active day（时长直接代表满意度）。
3. **Secondary**：CTR、completion rate、return rate 7-day。
4. **Guardrails**：satisfaction survey、search rate（高 → 推荐不准）、unsubscribe 率、ads revenue。
5. **Counter-metrics**：clickbait 风险（CTR ↑ 但 dwell ↓ → 内容质量差）。
6. **Long-term proxy**：30-day retention。

> **Follow-up 提示：**
> - 注意 Goodhart's law（过度优化 watch time → 推 clickbait）。
> - 怎么处理 *负面* externalities（如 misinfo）？

---

### Q25. Uber 想测试新的派单算法，怎么做？

**回答：**
**关键问题：网络效应 (interference)。** 司机看到 treatment 价格 → 接单行为变化 → 影响 control 用户。

**方案：Switchback Experiment**
1. 单元 = (city × time window)，例如 30-min windows。
2. 每个 (city, window) 随机分配 T 或 C。
3. 整城同时运行同一 variant → 司机间无 cross-contamination。
4. 统计：OLS + city/time fixed effects + cluster-robust SE。

**指标：** 完成订单率、ETA、司机收入、乘客等待时间。

**陷阱：** 时间窗口选择（太短 → 切换成本高；太长 → 数据点少）。

> **Follow-up 提示：**
> - 替代：geo-experiment（城市级 cluster-randomized）。

---

### Q26. 设计一个实验来测试"新课程"对学生成绩的影响。

**回答：**
1. **理想：** 学校随机化（cluster RCT），治疗组实施新课程。
2. **若不能随机化：**
   - **DiD**：找 treatment 和 control 学校，对比 pre/post 成绩。
   - **RD**：如果有"分数 ≥ X 才能上新课"的 cutoff。
   - **IV**：找影响选课但不直接影响成绩的工具（如：随机抽奖名额）。
3. **指标：** 短期（期末考）、长期（升学率、就业率）。
4. **Confounders：** 家庭背景、入学成绩、教师质量。

> **Follow-up 提示：**
> - 怎么处理学生间 spillover（同学影响）？答：cluster on classroom/school。
> - 关注 long-term outcome（不只期末）。

---

## Part G：编码题 (Coding)

### Q27. 实现一个 t-test 函数（不用 scipy）。

```python
import numpy as np

def t_test_two_sample(x, y):
    """Welch's t-test (unequal variance)."""
    n1, n2 = len(x), len(y)
    m1, m2 = np.mean(x), np.mean(y)
    v1, v2 = np.var(x, ddof=1), np.var(y, ddof=1)
    se = np.sqrt(v1/n1 + v2/n2)
    t = (m1 - m2) / se
    # Welch–Satterthwaite df
    df = (v1/n1 + v2/n2)**2 / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))
    # p-value via t distribution
    from scipy.stats import t as t_dist
    p = 2 * (1 - t_dist.cdf(abs(t), df))
    return t, p, df
```

---

### Q28. 实现 CUPED。

```python
import numpy as np

def cuped(Y_pre, Y_post, T):
    """
    Y_pre: pre-experiment metric (covariate, must NOT be affected by T)
    Y_post: post-experiment outcome
    T: 0/1 treatment
    Returns: ATE estimate, adjusted Y
    """
    theta = np.cov(Y_pre, Y_post, ddof=1)[0, 1] / np.var(Y_pre, ddof=1)
    Y_adj = Y_post - theta * (Y_pre - Y_pre.mean())
    ate = Y_adj[T == 1].mean() - Y_adj[T == 0].mean()
    return ate, Y_adj
```

---

### Q29. 实现 Propensity Score Matching。

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

def psm_att(df, treatment, outcome, covariates):
    # 1) Estimate propensity
    ps = LogisticRegression(max_iter=1000).fit(df[covariates], df[treatment])
    df = df.copy()
    df['ps'] = ps.predict_proba(df[covariates])[:, 1]
    # 2) Nearest neighbor match (1:1)
    treated = df[df[treatment] == 1].reset_index(drop=True)
    control = df[df[treatment] == 0].reset_index(drop=True)
    nbrs = NearestNeighbors(n_neighbors=1).fit(control[['ps']])
    _, idx = nbrs.kneighbors(treated[['ps']])
    matched_ctrl = control.iloc[idx.flatten()].reset_index(drop=True)
    # 3) ATT
    return (treated[outcome] - matched_ctrl[outcome]).mean()
```

---

### Q30. 实现 SRM 检验。

```python
from scipy.stats import chisquare

def srm_check(n_treatment, n_control, expected_ratio=0.5):
    n_total = n_treatment + n_control
    expected = [n_total * expected_ratio, n_total * (1 - expected_ratio)]
    observed = [n_treatment, n_control]
    chi2, p = chisquare(observed, expected)
    is_srm = p < 0.001    # Kohavi's recommended threshold
    return {'chi2': chi2, 'p_value': p, 'is_srm': is_srm}
```

---

### Q31. 用 numpy 实现 sample size calculator。

```python
import numpy as np
from scipy.stats import norm

def sample_size_continuous(delta, sigma, alpha=0.05, power=0.8, alternative='two-sided'):
    if alternative == 'two-sided':
        z_alpha = norm.ppf(1 - alpha/2)
    else:
        z_alpha = norm.ppf(1 - alpha)
    z_beta = norm.ppf(power)
    n = 2 * (z_alpha + z_beta)**2 * sigma**2 / delta**2
    return int(np.ceil(n))

def sample_size_proportion(p_baseline, p_new, alpha=0.05, power=0.8):
    delta = abs(p_new - p_baseline)
    sigma2 = (p_baseline*(1-p_baseline) + p_new*(1-p_new))   # pooled approximate
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    n = (z_alpha + z_beta)**2 * sigma2 / delta**2
    return int(np.ceil(n))

# Example
print(sample_size_continuous(delta=0.02, sigma=0.5))   # ~9800
print(sample_size_proportion(0.10, 0.11))              # ~14750
```

---

## 面试小技巧 (Interview Tips)

### 必须掌握的口头公式

> "16 σ² over Δ² 是样本量的 rule of thumb（α=0.05, power=0.8）。"
> "CUPED 把方差从 Var(Y) 缩到 Var(Y)·(1−ρ²)。"
> "p-value 是 *假设 H₀ 真* 时看到这种或更极端结果的概率。"
> "ATE = E[Y(1) − Y(0)]，但只能观测一个 → 因果推断核心问题。"
> "RCT 通过随机化让 (Y(0), Y(1)) ⊥ T，所以 E[Y|T=1] − E[Y|T=0] = ATE。"

### 答题节奏

1. **Restate** 题意（确认理解）。
2. **Define** 变量与目标 metric。
3. **Identify** 假设（独立性？SUTVA？平行趋势？）。
4. **Choose** 方法 + 解释为什么。
5. **Compute** 或描述算法。
6. **Discuss** 局限与 follow-up。

### 高频陷阱

- ⚠️ p-value ≠ effect 大小。
- ⚠️ correlation ≠ causation（除非随机化）。
- ⚠️ peeking 让 Type I error 失控。
- ⚠️ SRM = 红色警报。
- ⚠️ TWFE 在 staggered 时失效。
- ⚠️ IV 估计 LATE 不是 ATE。
- ⚠️ PSM 不能解决 unobserved confounder。
