# 因果推断 (Causal Inference) — 学习手册

> 因果推断核心问题：从 *观察数据* 中估计干预 (intervention) 效应。`Correlation ≠ Causation`，但用对方法可以从相关性中识别 (identify) 因果效应。

---

## 目录

1. [基础框架：Potential Outcomes (Rubin)](#1-基础框架potential-outcomes-rubin)
2. [核心假设：SUTVA、ignorability、positivity](#2-核心假设sutvaignorabilitypositivity)
3. [因果图 (DAG) 与 d-separation](#3-因果图-dag-与-d-separation)
4. [Backdoor / Frontdoor 调整 & do-calculus](#4-backdoor--frontdoor-调整--do-calculus)
5. [Propensity Score Matching (PSM) & IPW](#5-propensity-score-matching-psm--ipw)
6. [Difference-in-Differences (DiD)](#6-difference-in-differences-did)
7. [Instrumental Variables (IV)](#7-instrumental-variables-iv)
8. [Regression Discontinuity (RD)](#8-regression-discontinuity-rd)
9. [Heterogeneous Treatment Effects (HTE) & Uplift Modeling](#9-heterogeneous-treatment-effects-hte--uplift-modeling)
10. [因果识别决策树（什么时候用什么方法）](#10-因果识别决策树)
11. [参考资料](#11-参考资料)

---

## 1. 基础框架：Potential Outcomes (Rubin)

### 关键定义

- **Treatment** `D ∈ {0, 1}`：是否接受干预（如：用了新功能 / 没用）。
- **Potential outcomes** `Y(0), Y(1)`：每个个体在两种情况下的"可能"结果。
- **Observed outcome** `Y = D·Y(1) + (1−D)·Y(0)` —— **只能观测到一个**。
- **Fundamental Problem of Causal Inference (FPCI)**：每个个体的反事实 (counterfactual) 永远观测不到。

### 因果效应 (Treatment Effects)

| 量 | 定义 | 含义 |
|---|---|---|
| **ITE** (Individual Treatment Effect) | `Y_i(1) − Y_i(0)` | 个体效应（不可识别） |
| **ATE** (Average Treatment Effect) | `E[Y(1) − Y(0)]` | 总体平均效应 |
| **ATT** (Average Treatment effect on Treated) | `E[Y(1) − Y(0) \| D=1]` | 在干预组上的平均效应 |
| **ATC** (Average Treatment effect on Control) | `E[Y(1) − Y(0) \| D=0]` | 在对照组上的平均效应 |
| **CATE** (Conditional ATE) | `E[Y(1) − Y(0) \| X=x]` | 给定特征 x 下的效应（HTE 的对象） |

> 🎯 **面试金句：** *"Causal inference 是估计 ATE 或 CATE，本质是处理 Y(1) 和 Y(0) 中那个观测不到的反事实。"*

---

## 2. 核心假设：SUTVA、ignorability、positivity

### SUTVA (Stable Unit Treatment Value Assumption)

1. **No interference**：个体 i 的结果不受其他个体 treatment 影响。
2. **Consistency / no hidden versions**：treatment 只有一种形式。

> ⚠️ **网络效应 (network effects) 违反 SUTVA**。例如：朋友用了新功能，你也会被影响。Marketplace（Uber、Airbnb）通常违反 SUTVA → 用 switchback。

### Ignorability / Unconfoundedness（也叫 conditional independence）

`(Y(0), Y(1)) ⊥ D | X`

给定可观测协变量 `X`，treatment 与 potential outcomes 独立。这是 **观察性研究 (observational study)** 的核心假设。RCT 通过随机化 *设计上* 满足此条件。

### Positivity / Overlap

`0 < P(D=1 | X=x) < 1` 对所有 `x`。

每个 X 值下都有 treatment 和 control 样本；否则无法比较。

> 🎯 **面试常考：** "为什么 RCT 是 gold standard？"
> 答："因为随机化使 `(Y(0), Y(1)) ⊥ D` *无条件* 成立 —— 不需要观测所有 confounders，因此 ATE 等于 `E[Y|D=1] − E[Y|D=0]`。"

---

## 3. 因果图 (DAG) 与 d-separation

### DAG 三种基本结构

| 结构 | 图 | 含义 | 控制 X 的影响 |
|---|---|---|---|
| **Chain** (mediator) | `T → X → Y` | X 是 T → Y 的中介 | **不要控制** X（会 block 因果路径） |
| **Fork** (confounder) | `T ← X → Y` | X 是 confounder | **必须控制** X（block 后门路径） |
| **Collider** | `T → X ← Y` | X 是 collider | **不要控制** X（控制反而会引入 spurious 关联） |

```
T = Treatment, Y = Outcome, X = 第三变量
```

### d-Separation 规则

判断两节点是否 *条件独立* 的图论规则：

- **Chain / Fork**：默认连通；条件化中间变量 → 阻断 (blocked)。
- **Collider**：默认阻断；条件化 collider 或其后代 → 打开 (opened)。

```python
# Python 工具：DoWhy / pgmpy / networkx
from dowhy import CausalModel
import pandas as pd

# 构造 DAG (graph in DOT format)
graph = """digraph {
    X -> T;
    X -> Y;
    T -> Y;
}"""

model = CausalModel(data=df, treatment='T', outcome='Y', graph=graph)
identified = model.identify_effect()
print(identified)   # 自动给出 backdoor / frontdoor 调整公式
```

> 🎯 **面试题：** "如果 mediator 也被控制了会怎样？"
> 答："会 *under-estimate* 总效应（total effect），只剩下未经过 mediator 的 direct effect。"

---

## 4. Backdoor / Frontdoor 调整 & do-calculus

### Backdoor Criterion（后门准则）

集合 `Z` 满足后门准则当且仅当：
1. `Z` 中没有 `T` 的后代。
2. `Z` 阻断所有从 `T` 指向 `Y` 的"后门路径"（即指向 `T` 的入向路径）。

**Backdoor 调整公式：**

```
P(Y | do(T=t)) = Σ_z P(Y | T=t, Z=z) · P(Z=z)
```

### Frontdoor Criterion（前门准则）

当无法找到满足 backdoor 的 `Z` 时（例如有未观测 confounder），如果存在 `M` 使得：
1. `M` 完全 mediate `T → Y` 的所有路径。
2. 没有 `T → M` 的后门路径。
3. 所有 `M → Y` 的后门路径被 `T` 阻断。

**Frontdoor 调整公式：**

```
P(Y | do(T)) = Σ_m P(M=m | T) · Σ_t' P(Y | M=m, T=t') · P(T=t')
```

### Do-calculus 三规则（Pearl）

| Rule | 当时...时 | 可以重写 |
|---|---|---|
| **Rule 1** | Z 与 Y 独立给定 (W, do(X)) | `P(Y\|do(X), Z, W) = P(Y\|do(X), W)` |
| **Rule 2** | 干预 = 观测时 | `P(Y\|do(X), do(Z), W) = P(Y\|do(X), Z, W)` |
| **Rule 3** | 干预无效 | `P(Y\|do(X), do(Z), W) = P(Y\|do(X), W)` |

> Pearl 证明了 do-calculus 是 *完备* 的：任何可识别的因果效应都能通过这三条规则推导出来。

```python
# Backdoor 调整代码示例（Sklearn 实现）
from sklearn.linear_model import LinearRegression
import numpy as np

# Y = 0.5*T + 1.2*X + ε  (X 是 confounder, X 同时影响 T)
def backdoor_adjustment(df, treatment='T', outcome='Y', confounders=['X']):
    """E[Y | do(T)] via regression adjustment"""
    X = df[[treatment] + confounders].values
    y = df[outcome].values
    model = LinearRegression().fit(X, y)
    # ATE = treatment 系数（线性模型下 backdoor 调整等价于多元回归）
    return model.coef_[0]
```

---

## 5. Propensity Score Matching (PSM) & IPW

### Propensity Score

`e(X) = P(T=1 | X)`：给定协变量 X 时，接受 treatment 的概率。

**关键定理 (Rosenbaum & Rubin, 1983)：** 如果 ignorability 给定 X 成立，则 ignorability 给定 `e(X)` 也成立。换言之，可以用 `e(X)` 这个 *一维* 标量代替 X 的高维匹配。

### PSM 实现

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

def propensity_score_matching(df, treatment='T', outcome='Y', covariates=None):
    # 1) 估计 propensity score
    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(df[covariates], df[treatment])
    df['ps'] = ps_model.predict_proba(df[covariates])[:, 1]

    # 2) 对每个 treated 找最近的 control（1:1 nearest neighbor）
    treated = df[df[treatment] == 1].reset_index(drop=True)
    control = df[df[treatment] == 0].reset_index(drop=True)
    nbrs = NearestNeighbors(n_neighbors=1).fit(control[['ps']])
    distances, indices = nbrs.kneighbors(treated[['ps']])

    matched_control = control.iloc[indices.flatten()].reset_index(drop=True)

    # 3) ATT estimate
    att = (treated[outcome] - matched_control[outcome]).mean()
    return att, treated, matched_control
```

### Inverse Propensity Weighting (IPW)

```
ATE_IPW = (1/N) · Σ [ T_i·Y_i / e(X_i) − (1−T_i)·Y_i / (1−e(X_i)) ]
```

```python
def ipw_ate(df, treatment='T', outcome='Y', covariates=None):
    ps_model = LogisticRegression(max_iter=1000).fit(df[covariates], df[treatment])
    e = ps_model.predict_proba(df[covariates])[:, 1]
    e = np.clip(e, 0.01, 0.99)   # 避免极端权重
    T, Y = df[treatment].values, df[outcome].values
    ate = (T * Y / e - (1 - T) * Y / (1 - e)).mean()
    return ate
```

### Doubly Robust (DR) Estimator

结合 outcome model 和 propensity model —— 任一正确即可一致估计 ATE：

```python
from sklearn.linear_model import LinearRegression

def doubly_robust(df, T='T', Y='Y', X=None):
    ps = LogisticRegression(max_iter=1000).fit(df[X], df[T]).predict_proba(df[X])[:, 1]
    ps = np.clip(ps, 0.01, 0.99)
    mu1 = LinearRegression().fit(df[df[T]==1][X], df[df[T]==1][Y]).predict(df[X])
    mu0 = LinearRegression().fit(df[df[T]==0][X], df[df[T]==0][Y]).predict(df[X])
    t, y = df[T].values, df[Y].values
    ate = (mu1 - mu0).mean() + (t * (y - mu1) / ps).mean() - ((1 - t) * (y - mu0) / (1 - ps)).mean()
    return ate
```

> 🎯 **面试要点：** PSM 不能解决 *未观测* confounder（unobserved confounding）；只能平衡可观测的协变量。

---

## 6. Difference-in-Differences (DiD)

### 适用场景

有 panel data（同一些个体被多次观测），且：
- 一部分个体在某时间点 *被* treatment（policy change 等）。
- 另一部分作为对照。

### 核心思想：用"control 组随时间的变化"作为 treatment 组"反事实"的估计。

### 平行趋势假设 (Parallel Trends Assumption)

> 在没有 treatment 的情况下，treatment 组与 control 组的 outcome 会 *平行* 变化。

**ATT (DiD estimator):**

```
ATT = [E(Y_post | T=1) − E(Y_pre | T=1)] − [E(Y_post | T=0) − E(Y_pre | T=0)]
```

```python
import statsmodels.formula.api as smf

# DiD via interaction term
# Y = α + β·post + γ·treated + δ·(post × treated) + ε
# δ 就是 ATT
result = smf.ols('Y ~ post + treated + post:treated', data=df).fit()
print(result.params['post:treated'])    # ATT
print(result.summary())
```

### 双向固定效应 (Two-way Fixed Effects, TWFE)

```python
# Y_it = α_i + λ_t + δ·D_it + ε_it
result = smf.ols('Y ~ C(unit) + C(time) + D', data=df).fit()
```

> ⚠️ **2021+ 警告：** TWFE 在 *多期、staggered* 处理时可能给出错误的 ATT（Goodman-Bacon 2021）。新方法：Callaway-Sant'Anna、de Chaisemartin-D'Haultfœuille。

> 🎯 **面试题：** "PSM 和 DiD 怎么结合？"
> 答："先用 PSM 平衡 pre-treatment 协变量，再在匹配样本上做 DiD。这样 *联合* 解决可观测 + 时间不变的不可观测 confounder。"

---

## 7. Instrumental Variables (IV)

### IV 三个条件

工具 (instrument) `Z` 必须：
1. **Relevance**: `Cov(Z, T) ≠ 0` —— Z 影响 T。
2. **Exclusion**: Z 只通过 T 影响 Y（无直接路径）。
3. **Independence**: Z 与误差项独立（无 confounder 影响 Z）。

### 经典例子

| 问题 | 干预 T | 工具 Z | 论文 |
|---|---|---|---|
| 上学回报 | 学历 | 距大学远近 | Card (1995) |
| 警察对犯罪 | 警察人数 | 选举周期 | Levitt |
| 健康保险效应 | 投保 | 抽签结果 | Oregon Health Experiment |

### 2SLS (Two-Stage Least Squares)

```python
from linearmodels.iv import IV2SLS

# Stage 1: T = π·Z + γ·X + v   →  T_hat
# Stage 2: Y = β·T_hat + γ·X + u
result = IV2SLS.from_formula(
    'Y ~ 1 + X + [T ~ Z]',
    data=df
).fit()
print(result.params['T'])    # IV estimate
```

### LATE vs ATE

IV 估计的是 **LATE (Local Average Treatment Effect)** —— "complier" 子总体上的效应（即被工具影响的人）。这与 ATE *不同*。

> 🎯 **面试金句：** *"IV 处理 unobserved confounding，但只能识别 LATE，不是 ATE。除非假设效应同质 (homogeneity)。"*

---

## 8. Regression Discontinuity (RD)

### 适用场景

存在一个 *cutoff*：当某个 running variable `R` 越过阈值 `c` 时，treatment 状态发生跳跃。

例：考试成绩 ≥ 60 → 录取奖学金；GPA ≥ 3.5 → 毕业荣誉。

### Sharp RD vs Fuzzy RD

| 类型 | 性质 |
|---|---|
| **Sharp RD** | `R ≥ c` 完全决定 `T`（确定性） |
| **Fuzzy RD** | `R ≥ c` 提高 `T` 的 *概率*（用 IV 思想处理） |

### 核心公式（Sharp RD）

```
ATE_RD = lim_{r↓c} E[Y | R=r] − lim_{r↑c} E[Y | R=r]
```

只识别 `R = c` 附近的效应（local effect）。

```python
import statsmodels.formula.api as smf

# Local linear regression on bandwidth around cutoff
def sharp_rd(df, R='running', Y='outcome', cutoff=0, bandwidth=1.0):
    df = df[(df[R] >= cutoff - bandwidth) & (df[R] <= cutoff + bandwidth)].copy()
    df['T'] = (df[R] >= cutoff).astype(int)
    df['R_centered'] = df[R] - cutoff
    # Y = α + τ·T + β·R + γ·T·R + ε
    result = smf.ols(f'{Y} ~ T + R_centered + T:R_centered', data=df).fit()
    return result.params['T']    # τ = LATE at cutoff
```

> 🎯 **面试要点：** RD 的假设比 PSM 弱（不需要无未观测 confounder），但只能识别 *cutoff 附近* 的效应。

---

## 9. Heterogeneous Treatment Effects (HTE) & Uplift Modeling

### 为什么关心 HTE？

ATE 只告诉你 *平均* 效应，但实际中：
- 哪些用户对优惠券最敏感？→ targeting
- 哪些病人对药物效果最好？→ personalized medicine
- 哪些 SKU 对营销活动反应最强？→ ROI 优化

`τ(x) = E[Y(1) − Y(0) | X=x]` 即 **CATE**。

### Meta-Learners

| 方法 | 思路 | 何时用 |
|---|---|---|
| **S-learner** | 单一模型 `f(X, T)`，预测时设置 `T=1` 或 `T=0`，取差 | T 信号弱时性能差 |
| **T-learner** | 训两个模型 `f_1(X)`, `f_0(X)`，差为 CATE | 数据量足够大 |
| **X-learner** | 在 T-learner 基础上交叉估计 | unbalanced treatment / control |
| **R-learner** | residualization，减少模型偏差 | 鲁棒性好 |

```python
# T-learner 示例
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def t_learner(X_train, T_train, Y_train, X_test):
    m1 = RandomForestRegressor().fit(X_train[T_train==1], Y_train[T_train==1])
    m0 = RandomForestRegressor().fit(X_train[T_train==0], Y_train[T_train==0])
    cate = m1.predict(X_test) - m0.predict(X_test)
    return cate
```

### Causal Forests (Athey & Wager)

基于 honest random forest，每棵树用一半数据切分、另一半数据估计叶子节点的 CATE。

```python
# econml 库示例
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor

est = CausalForestDML(
    model_y=GradientBoostingRegressor(),
    model_t=GradientBoostingRegressor(),
)
est.fit(Y=Y, T=T, X=X, W=W)    # W: confounders, X: features for HTE
cate = est.effect(X_test)       # CATE estimate
ci = est.effect_interval(X_test)  # 95% CI
```

### Uplift Modeling

营销中常见，目标：**最大化 incremental treatment 反应**，而不是绝对反应。

四类用户（可分类）：

| | T=1 响应 | T=0 不响应 |
|---|---|---|
| **Persuadable** ✅ | 1 | 0 |
| Sure thing | 1 | 1 |
| Lost cause | 0 | 0 |
| Do-not-disturb ❌ | 0 | 1 |

只该对 Persuadable 投放干预 → uplift modeling 找他们。

> 🎯 **面试题：** "Uplift modeling vs 一般分类模型有什么区别？"
> 答："分类预测 `P(Y=1 | X, T=1)`；uplift 预测 `P(Y=1 | X, T=1) − P(Y=1 | X, T=0)` —— 是 *个体层面 ATE*，是 CATE 估计的另一名字。"

---

## 10. 因果识别决策树

```
是否能做随机化实验？
├── 是 → RCT / A/B test (gold standard)
└── 否 → 观察性研究
        │
        ├── 有 panel data？
        │   ├── 是 + 平行趋势假设 → DiD
        │   └── 否 ↓
        │
        ├── 存在 cutoff？
        │   └── 是 → RD (sharp / fuzzy)
        │
        ├── 找得到合法 instrument？
        │   └── 是 → IV / 2SLS
        │
        └── 假设 ignorability ⊥ X？
            ├── 是 → PSM / IPW / DR / Causal Forest
            └── 否 → frontdoor or 灵敏度分析
```

### 方法对比表

| 方法 | 假设 | 识别量 | 优点 | 缺点 |
|---|---|---|---|---|
| RCT | SUTVA | ATE | gold standard | 贵、伦理、不总能做 |
| PSM / IPW | ignorability + positivity | ATE / ATT | 易实现 | 需观测所有 confounder |
| DiD | parallel trends | ATT | 处理时间不变 unobserved | 需 panel data |
| IV | relevance + exclusion + indep. | LATE | 处理 unobserved confounding | 找好 IV 难，仅 complier |
| RD | continuity at cutoff | LATE at cutoff | 假设最弱 | 只识别 cutoff 附近 |
| Causal Forest | ignorability + 大样本 | CATE | 非线性、HTE | 需要大样本、解释性差 |

---

## 11. 参考资料

### 入门书 / 课程

- [*Causal Inference: The Mixtape* — Scott Cunningham](https://mixtape.scunning.com/) (免费 PDF + Python/R 代码)
- [*Causal Inference: What If* — Hernán & Robins](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/) (免费 PDF)
- [*Causal Inference for the Brave and True* — Matheus Facure](https://matheusfacure.github.io/python-causality-handbook/) (Python 在线书)
- [*The Effect* — Nick Huntington-Klein](https://theeffectbook.net/) (在线书)

### Python 库

- [DoWhy](https://github.com/py-why/dowhy) (Microsoft，自动 identification)
- [EconML](https://github.com/py-why/EconML) (CATE / HTE 工具)
- [CausalML](https://github.com/uber/causalml) (Uber，uplift modeling)
- [grf (R)](https://grf-labs.github.io/grf/) (Athey 团队的 causal forest)
- [linearmodels](https://bashtage.github.io/linearmodels/) (IV / 2SLS / panel)

### 重要论文

- Pearl, J. (2009). *Causality* (book)
- Athey, S., & Imbens, G. (2017). The State of Applied Econometrics: Causality and Policy Evaluation. *JEP*.
- Wager, S., & Athey, S. (2018). Estimation and Inference of Heterogeneous Treatment Effects using Random Forests. *JASA*.
- Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. *J. Econometrics*.
- Künzel et al. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. *PNAS*.

### 在线学习

- [Top 10 Causal Inference Interview Questions — GrabNGoInfo (Medium)](https://medium.com/grabngoinfo/top-10-causal-inference-interview-questions-and-answers-7c2c2a3e3f84)
- [Interview Preparation: Causal Inference — TDS](https://towardsdatascience.com/interview-preparation-causal-inference-44fbb8b0a5c6/)
- [Causal Inference cheat sheet — NC233](https://nc233.com/2020/04/causal-inference-cheat-sheet-for-data-scientists/)
- [Do-calculus adventures — Andrew Heiss](https://www.andrewheiss.com/blog/2021/09/07/do-calculus-backdoors/)
- [Causal Inference Is Eating Machine Learning — TDS](https://towardsdatascience.com/causal-inference-is-eating-machine-learning/)
- [Causal Effect Estimation — Kenneth Styppa, Medium](https://medium.com/causality-in-data-science/hands-on-causal-effect-estimation-with-python-aac40ca2cae0)
- [Causal Forests practical introduction — Mark H. White II](https://www.markhw.com/blog/causalforestintro)
