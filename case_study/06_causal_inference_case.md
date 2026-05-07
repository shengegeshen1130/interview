# Q6: A/B Test 不可行时的因果推断 (Causal Inference When A/B Test Isn't Feasible)

> **类型**：Product / Case Study (CHIME-D framework，重点在 E 步骤)
> **常见 follow-up 公司**：Airbnb、Uber、Lyft、Netflix、Meta (marketplace 团队)、LinkedIn
> **难度**：⭐⭐⭐⭐⭐ (高级 senior 题，能区分懂统计 vs 真懂 causal inference)
> **典型题面**：
> - "We rolled out a new pricing algorithm city-by-city — measure its impact."
> - "Estimate the effect of LinkedIn's new feed redesign on engagement, but you can't run A/B test."
> - "How do you measure the impact of a Super Bowl ad?"

---

## 一、什么时候 A/B 不可行？

| 场景 | 为什么不能 A/B |
|------|---------------|
| **Network / spillover effect** | treatment 影响 control（朋友圈、marketplace、电网） |
| **市场级 intervention** | 改 city-wide pricing、改税收政策、改司机 incentive 全城上线 |
| **One-off event** | Super Bowl 投放、品牌 campaign |
| **伦理 / 法律** | 不能给一半用户故意降级体验 |
| **历史数据问题** | 想分析过去某事件的 impact，没法重来 |
| **Long-term / 缓慢 effect** | A/B 跑 2 年成本太高 |
| **小样本 / 唯一单元** | 只有 1 个 treated unit (e.g., 一个新 launch 的国家) |

> **Senior 必备直觉**：A/B 是 gold standard，但工业里 30-40% 的因果问题都没法做 A/B。掌握 quasi-experiment 工具是 senior 必修。

---

## 二、Causal Inference 工具箱（5 大方法）

### 0. 核心 mental model：Potential Outcomes Framework

每个单元有两个潜在结果 $Y(1)$ 和 $Y(0)$，individual treatment effect 是 $Y_i(1) - Y_i(0)$。但**只能观察到一个**——这是 fundamental problem of causal inference。

**Average Treatment Effect**: $ATE = E[Y(1) - Y(0)]$

A/B 通过 randomization 让 treated 和 control 在 expectation 上 exchangeable，所以 $E[Y | T=1] - E[Y | T=0] = ATE$。

不能 randomize 时，要靠**identification assumption**让 observational 数据等价于 RCT。

---

### 1. Difference-in-Differences (DiD) — 最常用

**何时用**：
- 有 treatment group 和 control group
- 有 pre-treatment 和 post-treatment 数据
- 平行趋势假设成立

**公式**：
$$DiD = (\bar{Y}_{T, \text{post}} - \bar{Y}_{T, \text{pre}}) - (\bar{Y}_{C, \text{post}} - \bar{Y}_{C, \text{pre}})$$

**Identification 假设**：**Parallel trends**——如果没 treatment，treated 和 control 的 trend 会保持平行（差值恒定）。

**做法**：
1. 看 pre-period 多个时点 trend 是否平行（visual + placebo test）
2. 回归形式：
   $$Y_{it} = \alpha + \beta_1 T_i + \beta_2 \text{Post}_t + \beta_3 (T_i \times \text{Post}_t) + \epsilon_{it}$$
   $\beta_3$ 即 DiD 估计
3. Standard error 要 cluster (e.g., by city)

**例题**：
> Uber 在 SF 涨 base rate 5%，没在 NYC 涨。 estimate 涨价对 driver acceptance rate 的 impact。
>
> - Treated: SF；Control: NYC（或多个未涨城市）
> - Pre/post 各 4 周
> - DiD = (SF_post - SF_pre) - (NYC_post - NYC_pre)

**Pitfall**：
- 平行趋势 violation → check pre-trend、用 multiple periods 做 event study
- Heterogeneous treatment timing (城市分批 launch) → use Callaway-Sant'Anna or Sun-Abraham 估计器（2021 后修正传统 two-way FE bias）

---

### 2. Synthetic Control — 单 treated unit 时的 DiD

**何时用**：
- Treated unit 数量很少（甚至只有 1 个 city / country / firm）
- DiD 找不到合适的 single control

**核心 idea**：用多个 control units 加权组合成一个 "synthetic control"，使其 pre-period 完美 mimic treated unit。Treatment effect = 真实 treated 的 post outcome − synthetic 的 post outcome。

$$\hat{\tau}_t = Y_{1, t} - \sum_{j=2}^{J+1} w_j Y_{j, t}$$

权重 $w$ 通过最小化 pre-period 差异学到（Abadie, Diamond, Hainmueller 2010）。

**例题**：
> "Estimate the impact of California's tobacco tax on cigarette consumption."
>
> 没法把"加州"randomize。Synthetic control 用其他州按权重组合成"合成加州"，pre-period 匹配，post-period 看差距。

**工业用法**：
- Geo experiment 后 measure：少数城市 launch 新功能，用 synthetic control 估计
- Long-term campaign / brand impact

**Pitfall**：
- 需要长 pre-period（≥ 10 时点）保证权重稳定
- Donor pool 必须没受 treatment spillover 影响
- Inference 用 placebo test：把每个 control unit 假装当 treated，看 effect 分布做 permutation test

---

### 3. Regression Discontinuity Design (RDD)

**何时用**：
- Treatment 由某个**连续 score 的 threshold** 决定
- 在 threshold 附近，treated 和 control 几乎一样

**例子**：
- "credit score ≥ 700 给 loan，< 700 不给"——比较 700 附近 ±10 区间的人
- "GPA ≥ 3.5 给奖学金"——比较 3.5 附近的学生
- "video 时长 ≥ 1 min 才插 mid-roll ad"

**Identification**：在 cutoff 附近，treated 和 control 的 confounder 是 continuous 的，差距全部归因于 treatment。

**做法**：
1. Visualize：x 轴 running variable，y 轴 outcome，看 cutoff 处是否 jump
2. 局部 polynomial regression on each side of cutoff
3. Bandwidth selection (Imbens-Kalyanaraman)

**Pitfall**：
- **Manipulation**：用户能 manipulate score 接近 cutoff（McCrary density test 检查）
- 估计的是 cutoff 附近的 LATE (Local ATE)，不是全局 ATE

---

### 4. Instrumental Variables (IV)

**何时用**：
- 有 unmeasured confounder
- 找到一个 **instrument** Z 满足：
  1. **Relevance**: Z 影响 treatment T (Cov(Z, T) ≠ 0)
  2. **Exclusion**: Z 只通过 T 影响 Y (no direct effect)
  3. **Independence**: Z 和 confounder 独立

**经典例子**：
- Angrist 用 lottery draft number（越南战争兵役随机抽签）作 instrument 估计当兵对收入 impact
- Random encouragement design：随机给 50% 用户 push notification 推广某 feature，feature 使用率不是 random，但 push 是

**两阶段最小二乘 (2SLS)**：
1. Stage 1: $T = \pi Z + \nu$，得 $\hat{T}$
2. Stage 2: $Y = \beta \hat{T} + \epsilon$，$\hat{\beta}$ 是 LATE

**工业用法 (Encouragement design)**：
- 不能强制让用户用新 feature → 随机 push 通知是 instrument
- 估计 LATE = 那些被 push 推动去用 feature 的人的 effect (compliers)

**Pitfall**：
- Weak instrument → biased
- Exclusion 假设无法 statistical 验证，只能逻辑论证
- LATE ≠ ATE，要明确 estimand 是什么

---

### 5. Propensity Score Methods (Observational)

**何时用**：纯观察数据，没 quasi-experiment 设置。
- Selection on observables 假设：control 了所有 confounder
- 用 propensity score $e(x) = P(T=1 | X=x)$ 平衡 treated 和 control

**方法**：
- **Matching**: 给每个 treated 找 propensity score 最近的 control
- **IPW (Inverse Propensity Weighting)**: weight = T/e + (1-T)/(1-e)
- **Doubly Robust**: combine outcome model + propensity model（对 misspecification 更 robust）

**Pitfall**：
- **Unobserved confounders** 不能解决 → 要做 sensitivity analysis (e-value)
- Propensity overlap 必须充分（common support）

---

## 三、工业常见 case 详解

### Case A: 网络效应——LinkedIn 推 "you may know" 怎么衡量？

**问题**：A/B test 给一半用户更好的"你可能认识的人"推荐，他们 connect 后另一半 control user 也看到了 → control 也间接受 treat → underestimate effect。

**解法**：
1. **Ego-cluster randomization**: 把每个 user + 他的二度网络当 cluster，cluster 内同 treatment
2. **Cluster A/B**: 按 industry / region 分 cluster
3. **Sample size cost**: cluster 数 << user 数，power 大幅下降——必须 long duration

### Case B: Marketplace—Uber 涨 base fare 5%

**问题**：city-wide 改 base fare → driver 和 rider 都 affected → 不能 user-level random。

**解法**：
1. **Geo-experiment** (Uber 标准做法): 选若干 city pair（demographics 类似），一半涨一半不涨
2. **Synthetic control**: 单一 city 涨价时，其他 cities 加权拟合 synthetic comparison
3. **Switchback** (Doordash 用): 同 city 时间块交替——但要小心 carryover (用户上周经历影响这周决策)
4. **Difference-in-Differences**: 多 city 不同时间 launch，event study with staggered adoption

### Case C: 一次性事件—Super Bowl ad 的 lift

**问题**：单次事件、全国曝光，无 control。

**解法**：
1. **Synthetic control**：用未在 Super Bowl 投放的相似国家做 donor pool
2. **Pre/post comparison + 季节调整**：危险——可能 confound holiday、weather
3. **Geo-targeted ads + DiD**: 如果 ad 是 geo-targeted（部分 region 看不到），DiD on geo

### Case D: Long-term effect 又不能跑 2 年

**问题**：想知道某 feature 对 5-year retention 的 impact，但产品要 ship。

**解法**：
1. **Surrogate index** (Athey, Chetty et al.): 找一组 short-term metric 组合，pre-trained 预测 long-term outcome
2. **Long-term holdout**: 1% 用户长期保留 control，定期 measure
3. **Cohort analysis** + survival model

### Case E: 分阶段 rollout 的 city-by-city pricing

**问题**：定价算法在 50 个城市分 6 个月 rollout，每月 launch 一批。

**解法**：
1. **Staggered DiD / Event study**：考虑 treatment timing 异质，用 Callaway-Sant'Anna 或 De Chaisemartin-D'Haultfœuille (2020+) 估计器
2. **Synthetic Difference-in-Differences** (Arkhangelsky et al. 2021): combines DiD 和 synthetic control
3. ⚠️ **不要直接用 two-way FE**——staggered adoption 时有 contamination bias

---

## 四、Senior 答题的 6 个关键点

### 1. 显式声明 identification assumption

每个 method 都有一个不可验证的假设：
- DiD: parallel trends
- Synthetic Control: 权重稳定 + 无 anticipation
- RDD: no manipulation, continuity at cutoff
- IV: exclusion restriction
- Matching: no unobserved confounders

**Senior signal**：主动说 "我用 DiD，关键 assumption 是 parallel trends，我会 (1) 看 pre-period trend 平行性、(2) 做 placebo test 在虚假 treatment 时间点"。

### 2. 做 robustness check

- **Placebo test**: 在 fake treatment 时点 / unit 上跑同样 method，应该 0 effect
- **Sensitivity analysis**: 假设 unobserved confounder 多大才能 explain away effect (Rosenbaum bound, e-value)
- **Multiple methods**: DiD 和 synthetic control 同时跑，结果一致才可信

### 3. 区分 ATE / ATT / LATE / CATE

- ATE: average over whole population
- ATT: average over treated
- LATE: IV / RDD 估的是 compliers 的 effect
- CATE: heterogeneous effect by subgroup

不同 method 估的 estimand 不同，明确说"我估的是 X"。

### 4. Heterogeneous effect

**Causal forest** (Wager-Athey)、**X-learner** 估计 CATE：
$$\tau(x) = E[Y(1) - Y(0) | X = x]$$

工业用法：
- 找哪些 subgroup 受益最多 → personalized rollout
- 看 effect 是否在某 segment 反向 → potential harm

### 5. Inference 严谨

- DiD: cluster SE by treatment unit
- Synthetic control: permutation / placebo for inference
- IV: weak instrument F > 10 rule
- Matching: bootstrap or sandwich SE

### 6. 谈 "if we could A/B"

总要承认 A/B 仍是 gold standard，并 propose：
- 未来类似改动是否能设计 A/B（即使是 pilot 1-2 城市）
- Quasi-experiment 是 second-best，要在数据上谨慎

---

## 五、常见 Follow-up 问题

### Q: "DiD 和 synthetic control 怎么选？"

| 维度 | DiD | Synthetic Control |
|------|-----|-------------------|
| Treated 单元数 | 多个 | 1 个或少数几个 |
| Control 单元数 | 多个 | 多个 |
| Pre-period | 短也行 (≥ 1) | 必须长 (≥ 10) |
| 平行趋势 | 必须 | 不需要严格平行，但要 pre-period 拟合 |
| Inference | 标准回归 | placebo / permutation |

### Q: "Synthetic control 怎么验证 weights 合理？"

- Pre-period RMSPE 小（synthetic 紧密拟合 treated）
- Permutation: 把每个 control 当 treated 做同样 fit，看 treated unit 的 effect 是否在分布尾部
- 看 weights 是否稀疏且 economically meaningful（e.g., 加州合成 = 30% 内华达 + 25% 亚利桑那 + ...）

### Q: "Parallel trends 不成立怎么办？"

- 加 covariates: $Y_{it} = \alpha + \beta_1 T_i + \beta_2 \text{Post}_t + \beta_3 (T_i \times \text{Post}_t) + X_{it}\gamma + \epsilon$
- 用 synthetic control 替代
- Triple difference (DDD)

### Q: "怎么处理 spillover？"

- Cluster randomize
- 估计 spillover effect 大小：把 control 分"近 treated"和"远 treated"，看 outcome 差异
- Bound treatment effect (Manski-style partial identification)

### Q: "你怎么 communicate 结果给非技术 stakeholder？"

- "我估计这个 feature 让 retention 提升 2%，95% CI [0.5%, 3.5%]"
- 主动 disclose assumption: "这依赖于 X 假设，如果 X 不成立结果会偏 Y"
- 给 actionable interpretation，不要堆 method name

### Q: "面试官说我们已经全量 launch 了，没有 control，怎么办？"

- 时序角度：interrupted time series + structural break
- 跨 unit 角度：synthetic control 用其他 region / product
- Surrogate：用 leading indicator 预测 counterfactual
- 如果都不行，诚实说"我们对这个 launch 的 causal impact 估计精度有限，only correlation"

---

## 六、答题模板（Senior 标配）

> "因为有 X (network effect / marketplace / 一次性事件)，标准 user-level A/B 不可行。我会按以下顺序考虑：
>
> 1. **如果还有可能 randomize**：cluster A/B (geo / ego-cluster) 或 switchback。算 cluster 数下的 power，确认 detectable。
> 2. **不能 randomize 时，找 quasi-experiment 设置**：
>    - 有 pre/post + control unit → **DiD**，验 parallel trends + event study
>    - 单 treated unit + 长 pre-period → **Synthetic control**，permutation inference
>    - Threshold-based assignment → **RDD**
>    - 自然随机的 instrument → **IV / encouragement design**
> 3. **纯观察数据**：propensity score + doubly robust + e-value sensitivity
> 4. **多种方法 triangulate**：跑两种独立 method，结果一致才有信心
> 5. **Robustness check**: placebo test、pre-trend 检查、sensitivity bound
> 6. **明确 estimand**：是 ATE / ATT / LATE / CATE
>
> 沟通时要主动 disclose identification assumption 和 confidence band，不要让 stakeholder 把 quasi-experiment 当 RCT 看。"

---

## 七、Senior Trade-off 总结

| 决策 | Option A | Option B | 怎么选 |
|------|---------|---------|--------|
| Method | DiD | Synthetic Control | 多 treated 用 DiD，少 treated 用 SC |
| Inference | parametric SE | placebo / bootstrap | quasi-experiment 优先 placebo |
| Estimand | ATE | LATE / CATE | 看 stakeholder 要回答的问题 |
| Identification rigor | strong assumption (RDD) | weak (matching) | 选最容易 defend 的 |
| Single best method | one approach | triangulation | senior 必须 triangulate |

---

## 一句话答案 (Elevator Pitch)

> "A/B 不可行的核心原因是 network effect、marketplace、一次性事件、伦理或 long-term。Senior 要掌握 5 类工具：DiD（多 treated unit + parallel trends 假设）、synthetic control（少 treated unit + long pre-period）、RDD（threshold assignment）、IV（找到 exogenous instrument，常用 random encouragement design）、propensity score / doubly robust（纯观察数据）。每个 method 必须显式声明 identification assumption，做 placebo test + sensitivity analysis 验证，至少用两种方法 triangulate。明确说清估的是 ATE、ATT 还是 LATE，沟通 confidence band 和假设依赖。最 senior 的信号是承认 A/B 仍是 gold standard，但当现实约束让 RCT 不可能时，能熟练用 quasi-experiment 工具且严谨 communicate 局限性。"
