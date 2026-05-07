# 实验设计 (A/B Testing & Experimentation) — 学习手册

> A/B test 是工业界因果推断的"gold standard"。本文涵盖 FAANG / Marketplace 公司面试中所有高频考点。

---

## 目录

1. [假设检验基础 (Hypothesis Testing)](#1-假设检验基础-hypothesis-testing)
   - [1.1 基本概念](#11-基本概念)
   - [1.2 t-test / z-test 实现](#12-t-test--z-test-实现)
   - [1.3 ⭐ Type I 错误 与 Type II 错误 — 深入解析](#13--type-i-错误-与-type-ii-错误--深入解析)
2. [Power 分析与样本量](#2-power-分析与样本量)
3. [Metric 设计与 Multiple Testing](#3-metric-设计与-multiple-testing)
4. [方差缩减 (Variance Reduction): CUPED](#4-方差缩减-variance-reduction-cuped)
5. [随机化单位 (Randomization Unit) 与 Stratification](#5-随机化单位-randomization-unit-与-stratification)
6. [常见陷阱 (Pitfalls)](#6-常见陷阱-pitfalls)
   - [Sample Ratio Mismatch (SRM)](#sample-ratio-mismatch-srm)
   - [Novelty / Primacy Effect](#novelty--primacy-effect)
   - [Peeking](#peeking)
   - [Simpson's Paradox](#simpsons-paradox)
7. [高级实验设计](#7-高级实验设计)
   - [Switchback Experiments](#switchback-experiments)
   - [Cluster / Network Experiments](#cluster--network-experiments)
   - [Quasi-Experiments](#quasi-experiments)
   - [Sequential Testing](#sequential-testing)
   - [Multi-armed Bandits](#multi-armed-bandits)
8. [面试常考公式速查](#8-面试常考公式速查)
9. [参考资料](#9-参考资料)

---

## 1. 假设检验基础 (Hypothesis Testing)

### 1.1 基本概念

| 概念 | 定义 |
|---|---|
| **H₀** (null) | 通常 = "treatment 无效" |
| **H₁** (alternative) | "treatment 有效" |
| **Type I error (α)** | 假阳性 (false positive)：H₀ 真但拒绝（默认 0.05） |
| **Type II error (β)** | 假阴性 (false negative)：H₁ 真但没拒绝（默认 0.2 → power=0.8） |
| **Power (1−β)** | 真有效时检测到的概率 |
| **p-value** | 假设 H₀ 真时，看到当前或更极端结果的概率 |
| **Effect size** | treatment 与 control 的差距（如 Δ = μ_T − μ_C） |
| **MDE (Min. Detectable Effect)** | 给定 α、β、N，能检测的最小效应 |

### 1.2 t-test / z-test 实现

#### 双样本 t-test (连续指标)

```python
import numpy as np
from scipy import stats

def two_sample_t_test(treatment, control):
    t_stat, p_val = stats.ttest_ind(treatment, control, equal_var=False)
    diff = treatment.mean() - control.mean()
    se = np.sqrt(treatment.var(ddof=1)/len(treatment) + control.var(ddof=1)/len(control))
    ci_low, ci_high = diff - 1.96*se, diff + 1.96*se
    return diff, ci_low, ci_high, p_val
```

#### 比例 z-test (转化率指标)

```python
def proportion_z_test(x_T, n_T, x_C, n_C):
    p_T, p_C = x_T/n_T, x_C/n_C
    p_pool = (x_T + x_C) / (n_T + n_C)
    se = np.sqrt(p_pool*(1-p_pool)*(1/n_T + 1/n_C))
    z = (p_T - p_C) / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    return p_T - p_C, z, p_val
```

#### 单侧 vs 双侧

- **单侧**：明确假设方向（"treatment 比 control 好"）。p-value 减半，但被认为是 *cherry-picking*。
- **双侧** (默认)：treatment ≠ control，未指定方向。

> 🎯 **面试金句：** *"业界默认双侧 t-test，因为单侧仅在事先有强先验时合理。"*

---

### 1.3 ⭐ Type I 错误 与 Type II 错误 — 深入解析

> 这是面试中 *最* 高频的概念题（FAANG / 量化 / 咨询都问）。要做到 30 秒内能讲清 4 种情境、3 个相关公式、和它们如何随 α、β、N、Δ 变化。

#### 🎲 4 种决策结果（核心 2×2 表格）

|  | **真实状态：H₀ 真**（treatment 无效）| **真实状态：H₁ 真**（treatment 有效） |
|---|---|---|
| **决策：拒绝 H₀** | ❌ **Type I 错误**（α = 假阳）<br>"误判 treatment 有效" | ✅ **正确决策**（power = 1−β）<br>"成功检测到效应" |
| **决策：不拒绝 H₀** | ✅ **正确决策**（1−α）<br>"正确识别无效" | ❌ **Type II 错误**（β = 假阴）<br>"漏检了真实效应" |

> **记忆技巧：** "Type **I** 数字小、字母靠前" → 通常 α=0.05 *也很小*；"Type **II**" → β=0.2 *较大*。 业界习惯 α 严控（0.05），β 宽松（0.2）。

---

#### 🏛️ 直观类比

##### 类比 1：法庭（最经典）

> "无罪推定"原则：H₀ = "被告无罪"

| | 实际无罪 | 实际有罪 |
|---|---|---|
| **判定有罪** | Type I：冤枉好人 ❌ | 正确 ✅ |
| **判定无罪** | 正确 ✅ | Type II：放走凶手 ❌ |

法律系统选择 *小 α*（"宁可放过，不可冤枉"）→ 类似科学上选 α=0.05 是为了避免假阳。

##### 类比 2：医学诊断

> H₀ = "病人健康"

| | 实际健康 | 实际有病 |
|---|---|---|
| **诊断有病** | Type I：误诊（虚惊一场，但可能浪费治疗钱）| 正确 |
| **诊断健康** | 正确 | Type II：漏诊（病人错过治疗）|

癌症筛查通常容忍 *较高* Type I（α 大），换 *低* Type II（β 小） —— 因为漏诊代价 >> 误诊。

##### 类比 3：火灾报警

> H₀ = "没火"

- **Type I**（假警报）：吓人、但可控。
- **Type II**（火来了不响）：致命。

→ 火警系统设计成 *高* power（低 β），代价是有较多误报（高 α）。

##### 类比 4：A/B 测试场景（业务上的代价）

> H₀ = "新功能不优于旧功能"

| | 实际无提升 | 实际有提升 |
|---|---|---|
| **决定上线** | Type I：发布无用功能（浪费工程师 + 可能伤害用户体验）| 正确 |
| **决定不上线** | 正确（保留 baseline） | Type II：错过好功能（机会成本 = $$$）|

→ **不同业务下 α 和 β 的权衡不同：**

| 场景 | α 重要 | β 重要 | 建议 |
|---|---|---|---|
| 高风险改动（支付、登录） | 高 | 中 | α=0.01, β=0.2 |
| 一般产品功能 | 中 | 中 | α=0.05, β=0.2 |
| 增长实验（多轮迭代） | 低 | 高 | α=0.1, β=0.1 |
| 医疗 / 安全 | 极高 | 高 | α=0.001, β=0.1 |

> 🎯 **面试金句：** *"α 和 β 的最优权衡取决于 Type I 与 Type II 错误的 *相对成本* — 业务决策，不是统计决策。"*

---

#### 📐 数学关系（必须秒答）

##### 关系 1：α 和 β **不是互补的**

❌ **常见错误**：以为 `α + β = 1`。

实际上 α 和 β 在 *不同分布下* 计算（α 在 H₀ 分布，β 在 H₁ 分布）：

```
α = P(reject H₀ | H₀ 真)        ← 在 H₀ 分布下计算
β = P(fail to reject H₀ | H₁ 真) ← 在 H₁ 分布下计算
```

##### 关系 2：α 和 β 的 trade-off（给定固定 N）

固定样本量 N 时，缩小 α（更难拒绝）→ 必然增大 β（漏检更多）。

```
拒绝域阈值 → ⬆ (更严格)
α            → ⬇ (假阳少)
β            → ⬆ (漏检多)
power = 1−β  → ⬇
```

##### 关系 3：增加 N 同时降低 α 和 β

```
N ↑    →    α ↓ 且 β ↓ （两者同时改善）
       →    standard error ↓
       →    分布更"窄" → 重叠区域更小
```

##### 关系 4：Effect size Δ 的影响

```
|Δ| ↑    →    H₀ 和 H₁ 分布"分得更开"   →    β ↓ (power ↑)
|Δ| ↓    →    两分布几乎重叠           →    β → 1−α (近乎随机)
```

##### 关系 5：Power 的精确公式（双样本 z-test）

对单侧检验：
```
power = Φ( |Δ|/SE − z_α )
其中 SE = σ·√(2/n)
```

对双侧检验：
```
power ≈ Φ( |Δ|/SE − z_{α/2} ) + Φ( −|Δ|/SE − z_{α/2} )
       ≈ Φ( |Δ|/SE − z_{α/2} )    (当 Δ > 0 时第二项可忽略)
```

```python
from scipy.stats import norm
import numpy as np

def calc_power(delta, sigma, n_per_group, alpha=0.05, two_sided=True):
    """计算给定参数下的 power = 1 − β"""
    se = sigma * np.sqrt(2 / n_per_group)
    z_alpha = norm.ppf(1 - alpha/2) if two_sided else norm.ppf(1 - alpha)
    return 1 - norm.cdf(z_alpha - abs(delta)/se)

# 例：Δ=0.05, σ=1, n=1000 per group
print(calc_power(0.05, 1, 1000))   # ≈ 0.61
print(calc_power(0.05, 1, 3000))   # ≈ 0.97
print(calc_power(0.10, 1, 1000))   # ≈ 0.99
```

---

#### 📊 可视化（脑中要画出的图）

```
                  H₀ 分布            H₁ 分布
                    ▲                  ▲
                   ╱ ╲                ╱ ╲
                  ╱   ╲              ╱   ╲
                 ╱     ╲            ╱     ╲
              ──┘       └──────────┘       └──
                          │  ← 拒绝阈值 (z_{α/2})
                          │
                ─────────────────────────────► 检验统计量
                          
            ┌── α/2 ──┐              ┌── β ──┐
            (右尾在 H₀ 下面积)        (H₁ 分布在阈值左边的面积)
```

- **α** = H₀ 分布右尾（超阈值的面积）
- **β** = H₁ 分布左侧（不超阈值的面积）
- **Power** = H₁ 分布右侧（成功超阈值的面积）

调整阈值 → α 和 β 此消彼长。

---

#### 🔢 具体数字感受

##### 例 1：典型 A/B 测试

```
H₀: μ_T = μ_C
σ = 1.0, n = 1000 per group, α = 0.05 (双侧)

不同真实 Δ 下的 power 和 β：
Δ = 0.00 → power = 0.05  (= α，纯随机水平), β = 0.95
Δ = 0.05 → power = 0.61, β = 0.39  ← 不够灵敏
Δ = 0.10 → power = 0.99, β = 0.01
Δ = 0.20 → power ≈ 1.00, β ≈ 0
```

→ **结论：** 同样的实验，对小 effect 漏检率 (β) 高达 39%！

##### 例 2：peeking 对 α 的影响

每天偷看 p-value，一旦 < 0.05 就停 → **实际 Type I error 远高于 0.05**。

| 偷看次数 | 实际 α |
|---|---|
| 1 (固定样本量) | 0.05 |
| 2 | 0.083 |
| 5 | 0.142 |
| 10 | 0.193 |
| 100 | 0.405 |

→ **结论：** 偷看 10 次，假阳率从 5% 飙到 19%！

```python
import numpy as np
from scipy.stats import ttest_ind

def peeking_alpha_simulation(n_peeks=10, total_n=10000, n_sims=10000):
    """模拟无效应情境下，多次偷看的实际 Type I error"""
    false_positives = 0
    check_points = np.linspace(100, total_n, n_peeks).astype(int)
    for _ in range(n_sims):
        # H₀ 真：两组无差异
        a = np.random.normal(0, 1, total_n)
        b = np.random.normal(0, 1, total_n)
        for n in check_points:
            _, p = ttest_ind(a[:n], b[:n])
            if p < 0.05:
                false_positives += 1
                break
    return false_positives / n_sims

# print(peeking_alpha_simulation(n_peeks=10))  # 约 0.19
```

##### 例 3：Multiple testing 对 α 的累积影响

```
家族水平 α (FWER) = 1 − (1 − α)^m    其中 m = 测试数

m=1   → 0.05
m=5   → 0.226
m=10  → 0.401
m=20  → 0.642
m=100 → 0.994
```

→ **结论：** 测 20 个 metrics，至少一个假阳概率 > 64%！必须校正。

---

#### ⚠️ 7 个常见误区

| # | 误区 | 真相 |
|---|---|---|
| 1 | "α + β = 1" | ❌ 二者在不同分布下计算，无此关系 |
| 2 | "p < 0.05 就一定有真效应" | ❌ p-value 不是 H₁ 真的概率 |
| 3 | "p > 0.05 就是没效应" | ❌ 可能只是 power 不够（β 大） |
| 4 | "Type I 比 Type II 重要" | ❌ 取决于业务代价，不是绝对的 |
| 5 | "α=0.05 是标准答案" | ❌ Fisher 任意选的，不同领域不同 |
| 6 | "增大样本只为降 β" | ❌ 同时降低 α 和 β 的标准误 |
| 7 | "可以多次偷看 p-value" | ❌ 每次偷看都膨胀 α，必须用 sequential testing |

---

#### 🛠️ 实务工具：sample size 同时控制 α 和 β

```python
from scipy.stats import norm
import math

def required_sample_size(delta, sigma, alpha=0.05, beta=0.2, two_sided=True):
    """同时满足 α 和 β 要求的每组样本量"""
    z_alpha = norm.ppf(1 - alpha/2) if two_sided else norm.ppf(1 - alpha)
    z_beta = norm.ppf(1 - beta)
    n = 2 * (z_alpha + z_beta)**2 * sigma**2 / delta**2
    return math.ceil(n)

# 例：检测 Δ=0.05, σ=1
print(required_sample_size(0.05, 1, alpha=0.05, beta=0.2))   # ~6280
print(required_sample_size(0.05, 1, alpha=0.05, beta=0.1))   # ~8407 (more strict β)
print(required_sample_size(0.05, 1, alpha=0.01, beta=0.2))   # ~9437 (more strict α)
print(required_sample_size(0.05, 1, alpha=0.01, beta=0.1))   # ~12,049
```

---

#### 🎯 5 句面试必背

1. **"Type I 是 H₀ 真但被拒绝（false positive，α）；Type II 是 H₁ 真但没拒绝（false negative，β）。"**
2. **"Power = 1−β = 真有效时能检测到的概率，业界默认 0.8。"**
3. **"α 和 β 不互补 — 它们在不同的分布下定义。"**
4. **"固定 N 时 α 和 β 是 trade-off；增大 N 才能同时降两者。"**
5. **"Peeking、multiple testing 都会让实际 α 远超名义 α，必须校正（sequential testing / BH）。"**

---

## 2. Power 分析与样本量

### 样本量公式（双样本 t-test）

```
n_per_group = 2 · (z_{α/2} + z_β)² · σ² / Δ²
```

- `z_{0.025} ≈ 1.96`, `z_{0.2} ≈ 0.84`
- 简化：`n ≈ 16 · σ² / Δ²` (α=0.05, power=0.8)

```python
from statsmodels.stats.power import tt_ind_solve_power, NormalIndPower

# 给定 effect size 求 n
n = tt_ind_solve_power(
    effect_size=0.1,    # Cohen's d = Δ/σ
    alpha=0.05,
    power=0.8,
    alternative='two-sided'
)

# 二项分布版（转化率）
from statsmodels.stats.proportion import proportions_chisquare_effectsize
es = proportions_chisquare_effectsize([0.10, 0.11])    # 10% vs 11%
n = NormalIndPower().solve_power(effect_size=es, alpha=0.05, power=0.8)
```

### 经验法则

| 情境 | 样本量 |
|---|---|
| 转化率从 10% 到 11%（绝对 1pp） | 每组 ~14,750 |
| 转化率从 5% 到 5.5% | 每组 ~28,000 |
| 检测 1% 的相对提升（baseline 10%） | 每组 ~140,000 |

### Power 决定因素

```
Power ↑ 当：
  - N ↑
  - Effect size ↑
  - σ ↓ (variance 越小)
  - α ↑ (但通常固定 0.05)
```

> 🎯 **面试题：** "Power 为什么重要？"
> 答："低 power → 即使 treatment 真有效，也常 *漏检*，浪费实验资源；同时 *significant 的结果* 也更不可信（winner's curse）。"

---

## 3. Metric 设计与 Multiple Testing

### Metric 分类

| 类型 | 示例 | 用途 |
|---|---|---|
| **Primary / OEC** (Overall Evaluation Criterion) | 转化率、收入、留存 | 决策 |
| **Secondary** | 点击率、停留时长 | 辅助 |
| **Guardrail** | 加载延迟、crash 率、SRM | 防止破坏 |
| **Counter-metric** | 人工干预指标 | 平衡正负影响 |
| **Diagnostic / Debug** | 错误日志、bot 比例 | 排查 |

### Goodhart's Law

> "When a measure becomes a target, it ceases to be a good measure."

`Click-through rate` 当 metric → 用 clickbait 做封面 → 用户体验崩。

### Multiple Testing Problem

测 100 个 metric，每个 α=0.05，**至少一个误报**的概率 ≈ `1 − 0.95^100 ≈ 99.4%`！

### 校正方法

| 方法 | 控制 | 公式 | 性质 |
|---|---|---|---|
| **Bonferroni** | FWER (Family-Wise Error Rate) | `α' = α / m` | 保守 |
| **Holm** | FWER | step-down，按 p-value 排序 | 比 Bonferroni 强 |
| **Benjamini-Hochberg (BH)** | FDR (False Discovery Rate) | `α'_k = k·α/m` | 较宽松，业界常用 |

```python
from statsmodels.stats.multitest import multipletests

p_values = [0.001, 0.04, 0.05, 0.07, 0.1]
reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
```

> 🎯 **面试金句：** *"OEC 不要校正（只一个），但 secondary metrics 多则要 FDR 校正，否则随便能挑出'显著'的 metric。"*

---

## 4. 方差缩减 (Variance Reduction): CUPED

### 直觉

样本量公式中 `n ∝ σ²`。如果能 *降低* σ，样本量需求线性下降。

### CUPED (Deng, Xu, Kohavi, Walker, 2013, Microsoft)

用 *实验前* 的协变量 `X`（与 outcome `Y` 相关）来 *减去* 个体的"基线波动"。

**调整公式：**
```
Y_cuped = Y − θ · (X − E[X])
其中 θ = Cov(X, Y) / Var(X)
```

- `Y_cuped` 与 `Y` 有相同的 ATE，但方差更小：`Var(Y_cuped) = Var(Y) · (1 − ρ²)`，其中 `ρ = Corr(X, Y)`。
- 如果 `ρ = 0.7`，方差降 51% → 样本量需求约减半。

```python
import numpy as np

def cuped_adjust(Y_pre, Y_post, T):
    """
    Y_pre: pre-experiment metric (covariate)
    Y_post: outcome
    T: 0/1 treatment indicator
    """
    theta = np.cov(Y_pre, Y_post)[0, 1] / np.var(Y_pre)
    Y_adj = Y_post - theta * (Y_pre - Y_pre.mean())
    ate = Y_adj[T == 1].mean() - Y_adj[T == 0].mean()
    return ate, Y_adj

# Example: pre-exp page views very correlated with post-exp page views
# ρ ≈ 0.8 → variance drop 64% → need 1/0.36 ≈ 2.78x fewer users!
```

### CUPED 注意事项

1. `X` 必须 *实验前* 测量（否则会被 treatment 污染）。
2. `X` 与 `Y` 相关性越高越好（`ρ²` 越大缩减越多）。
3. 对二分类指标（点击率），用 logistic regression 形式。

### 其他方差缩减方法

| 方法 | 适用 |
|---|---|
| **Stratification** | 分层（如新/老用户、设备）后分别估计 |
| **Post-stratification** | 实验后按特征分层加权 |
| **Variance reduction via ML** | 用 ML 模型预测 Y 作为 covariate |
| **CUPAC** (DoorDash) | CUPED + ML 模型预测的 covariate |

> 🎯 **面试金句：** *"Netflix、Microsoft、Booking 都报告 CUPED 至少节省 30-60% 样本量；目前是业界标配。"*

---

## 5. 随机化单位 (Randomization Unit) 与 Stratification

### 选择随机化单位

| 单位 | 何时用 | 缺点 |
|---|---|---|
| **User** | 用户体验改动（最常用） | 网络效应可能违反 SUTVA |
| **Session** | UI 改动且 user-level 学习不重要 | 同用户不同 session 不一致 |
| **Page view** | 实验粒度细 | 用户体验不一致 |
| **Time** (switchback) | Marketplace、网络效应大 | 时间相关性 |
| **Cluster** (city, region) | 网络效应、市场效应 | 样本量小 |

### Stratified Randomization

提前按重要协变量分层（如：新老用户 × 设备），每层内做随机化。

```python
import pandas as pd

def stratified_assign(df, strata=['new_user', 'device']):
    df['variant'] = 'C'  # default control
    for keys, grp in df.groupby(strata):
        n = len(grp)
        treated = grp.sample(n//2, random_state=42).index
        df.loc[treated, 'variant'] = 'T'
    return df
```

**好处：**
- 保证每层 treatment / control 比例一致。
- 减少层间 imbalance 引起的方差。

---

## 6. 常见陷阱 (Pitfalls)

### Sample Ratio Mismatch (SRM)

实验设定 50/50 分流，但实际收到的 T:C 显著偏离（如 49.2:50.8）→ 系统性偏差 (bias)。

**SRM 检验：** chi-squared test

```python
from scipy.stats import chisquare

def srm_test(n_T, n_C, expected_ratio=0.5):
    n_total = n_T + n_C
    expected_T = n_total * expected_ratio
    chi2, p = chisquare([n_T, n_C], [expected_T, n_total - expected_T])
    return p

# p < 0.001 → SRM detected, 不要相信实验结果！
```

**常见原因：**
1. Bot / crawler 在某一组被过度过滤。
2. Bucketing bug。
3. 重定向 / 加载错误使某一组用户掉出。
4. 不同 SDK / 平台不兼容。

> ⚠️ **Kohavi**："SRM 像安全带 —— 没装也能开车，但出事就是大事。"

### Novelty / Primacy Effect

| 效应 | 含义 |
|---|---|
| **Novelty** | 新功能"新鲜感"早期高，长期回归 |
| **Primacy** | 老用户对改动初期反感，长期适应 |

**检测：**
- Treatment effect 随时间下降 → 可能 novelty。
- Treatment effect 随时间上升 → 可能 primacy。

**应对：**
- 跑长一点（2-4 周）看是否稳定。
- 分新/老用户子样本。

### Peeking

每天看 p-value，一旦 < 0.05 就停 → 大幅膨胀 Type I error（实际可能 > 30%）。

**应对：**
- 提前固定样本量再查看。
- 用 sequential testing（如 Always Valid Inference, mSPRT）。

```python
# Always Valid p-value (Optimizely / Statsig)
# 允许任意时刻 peek 而不增加 Type I error
# 实现复杂；建议用现成库
```

### Simpson's Paradox

整体趋势与各子组趋势相反。

**例：** 整体 treatment > control，但每个 country 都是 treatment < control。

**原因：** treatment 在某 country（基线高的）被过度抽样。

**应对：**
- 检查 SRM 在每个 segment。
- 用 stratification。

---

## 7. 高级实验设计

### Switchback Experiments

Marketplace（Uber、Lyft、DoorDash、Airbnb）常用：在 *时间* 上交替 treatment / control，所有用户同时收到当前 variant。

**为什么用？** 解决 *interference / network effect* —— 司机看到 treatment 价格 → 行为变化 → 影响 control 用户。

**设计：**
- 把时间切成窗口（如 30 分钟）。
- 每个窗口随机分配 T 或 C（或交替）。
- 单元 = (city × time window)。

```python
def switchback_design(cities, time_windows, p_treatment=0.5):
    """每个 (city, window) 独立随机分配"""
    import random
    return {
        (city, t): 'T' if random.random() < p_treatment else 'C'
        for city in cities
        for t in time_windows
    }

# Analyze with regression including city + time fixed effects
```

**统计分析：** 用 OLS + city/time FE，cluster-robust SE。

> 🎯 **DoorDash 用 30-min windows；Lyft 在 city level 切换。**

### Cluster / Network Experiments

社交网络上 treatment 会通过 *边* 传染 → SUTVA 违反。

**解决：**
- **Cluster-randomized**：把网络切成"densely connected"的 cluster（如：用 community detection），整 cluster 同 treatment。
- **Ego-network experiment**：treatment 个体 + 其朋友圈。

### Quasi-Experiments

无法做随机化时退而求其次。

| 方法 | 何时用 |
|---|---|
| **DiD** | 有 pre/post 数据 + control 组 |
| **RD** | 有明确 cutoff |
| **IV** | 有合法工具 |
| **Synthetic Control** | 单个 treated 单元，多个 control，构造加权 control |
| **Interrupted Time Series** | 单组 + 干预时间已知 |

### Sequential Testing

允许 *提前停止*，但保持 Type I error 控制。

| 方法 | 思路 |
|---|---|
| **Group Sequential** | 预设 K 个 look，调整 α 在每个 look |
| **mSPRT** (Always Valid Inference) | 任意时刻 peek，对 likelihood ratio 做检验 |
| **Bayesian** | 用 posterior 做决策（无固定 frequentist 控制） |

### Multi-armed Bandits

把流量更多 *动态* 分给好的 variant —— 减少 regret，但 *估计 ATE 偏差大*。

```python
# Thompson Sampling 经典实现
import numpy as np

class ThompsonSampling:
    def __init__(self, n_arms):
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)
    def select_arm(self):
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)
    def update(self, arm, reward):
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
```

**A/B test vs Bandit：**
- A/B：目标是 *学习* 最优 → 估计精确，但 regret 高。
- Bandit：目标是 *最大化奖励* → regret 低，但估计有偏。

---

## 8. 面试常考公式速查

### 样本量

```
N_per_group ≈ 16 · σ² / Δ²              (continuous, α=0.05, power=0.8)
N_per_group ≈ 16 · p(1-p) / (p_T - p_C)²  (proportion)
```

### 标准误 (SE)

```
SE(diff in means)    = √(σ_T²/n_T + σ_C²/n_C)
SE(diff in props)    = √(p̂(1-p̂)·(1/n_T + 1/n_C))    where p̂ = pooled
SE(ratio metric)     = use delta method
```

### 置信区间

```
95% CI = [Δ̂ − 1.96·SE, Δ̂ + 1.96·SE]
```

### Power

```
Power(Δ) = P(reject H₀ | true effect = Δ)
        = Φ(z_α/2 − Δ/SE)   for one-sided
```

### CUPED 缩减率

```
Var(Y_cuped) / Var(Y) = 1 − ρ²
样本量缩减比例 = 1 / (1 − ρ²)
```

| ρ | 方差缩减 | 等效样本量倍数 |
|---|---|---|
| 0.3 | 9% | 1.10× |
| 0.5 | 25% | 1.33× |
| 0.7 | 49% | 1.96× |
| 0.9 | 81% | 5.26× |

### Delta Method（比率指标）

`R = X / Y` 的方差近似：
```
Var(R) ≈ (1/μ_Y²) · Var(X) + (μ_X²/μ_Y⁴) · Var(Y) − 2·(μ_X/μ_Y³) · Cov(X,Y)
```

---

## 9. 参考资料

### 必读书 / 论文

- **Kohavi, Tang, Xu (2020).** *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*（业界圣经）
- [Deng et al. (2013). *Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data* (CUPED)](https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Bojinov & Simchi-Levi (2020). *Design and Analysis of Switchback Experiments*](https://arxiv.org/pdf/2009.00148)
- [Fabijan et al. (2019). *Diagnosing Sample Ratio Mismatch* (KDD)](https://exp-platform.com/Documents/2019_KDDFabijanGupchupFuptaOmhoverVermeerDmitriev.pdf)
- [Statistical Challenges in Online Controlled Experiments (T&F, 2023)](https://www.tandfonline.com/doi/full/10.1080/00031305.2023.2257237)

### 工程博客

- [Microsoft ExP Platform](https://exp-platform.com/)
- [Netflix TechBlog — experimentation](https://netflixtechblog.com/tagged/experimentation)
- [Airbnb Engineering — experimentation](https://medium.com/airbnb-engineering/tagged/data)
- [DoorDash Engineering — switchback](https://doordash.engineering/category/data-science/)
- [Booking.com data science](https://booking.ai/)

### 教程

- [50 A/B Testing Interview Questions & Answers — DataLemur](https://datalemur.com/blog/ab-testing-interview-questions-and-answers)
- [Top 30 A/B Testing Interview Questions (2026) — DataInterview](https://www.datainterview.com/blog/ab-testing-interview-questions)
- [Top 60 Statistics & A/B Testing Interview Questions — InterviewQuery](https://www.interviewquery.com/p/statistics-ab-testing-interview-questions)
- [80 A/B Testing Interview Questions — MentorCruise](https://mentorcruise.com/questions/abtesting/)
- [A/B Testing Data Science Interview Guide — StrataScratch](https://www.stratascratch.com/blog/ab-testing-data-science-interview-questions-guide)
- [How to Double A/B Testing Speed with CUPED — TDS](https://towardsdatascience.com/how-to-double-a-b-testing-speed-with-cuped-f80460825a90/)
- [Understanding CUPED — Matteo Courthoud](https://matteocourthoud.github.io/post/cuped/)
- [CUPED Explained — Statsig](https://www.statsig.com/blog/cuped)
- [Sample Ratio Mismatch — TDS](https://towardsdatascience.com/sample-ratio-mismatch-so-many-questions-how-to-answer-them-a86a1893e35/)
- [Switchback experiments — TDS optimization guide](https://towardsdatascience.com/how-to-optimize-your-switchback-a-b-test-configuration-791a28bee678/)
- [Marketplace experimentation — Cornell ORIE 5355 lecture](https://orie5355.github.io/Fall_2022/static_files/lectures/Lecture14_Experimentation_marketplaces.pdf)

### Python / R 工具

- [`statsmodels`](https://www.statsmodels.org/) — t-test, regression, multiple testing
- [`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html) — basic tests
- [`growthbook`](https://www.growthbook.io/) (open source A/B platform with CUPED)
- [`spotify/confidence`](https://github.com/spotify/confidence) (Spotify A/B test library)
