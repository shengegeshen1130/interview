# 期望值/方差 & 多数投票分类器 (Expected Value/Variance & Majority-Vote Classifier) — 面试备战手册

针对两个概率类面试主题精心整理的学习指南，类似题目来源于真实面试题库（Jane Street、FAANG、量化基金）和 ML 参考资料。每道题都包含核心思路 (key insight)；计算非平凡的题目附有可运行的 Python 解答。

---

## 目录 (Table of Contents)

- [主题 1：期望值 & 方差](#topic-1--expected-value--variance)
  - [1.0 原题 (Original problem)](#10--original-problem)
  - [1.1 公平 6 面骰子](#11--fair-6-sided-die)
  - [1.2 两颗骰子之和](#12--sum-of-two-dice)
  - [1.3 两颗骰子的最大/最小值 (Jane Street)](#13--max-and-min-of-two-dice-jane-street)
  - [1.4 Bernoulli / Binomial — 推导 E 和 Var](#14--bernoulli--binomial--derive-e-and-var)
  - [1.5 几何分布 — 等待时间](#15--geometric-distribution--waiting-time)
  - [1.6 偏置硬币：贝叶斯翻转 (Facebook)](#16--biased-coin-bayesian-flip-facebook)
  - [1.7 复合和 — Law of Total Variance](#17--compound-sum--law-of-total-variance)
  - [1.8 混合分布 (Mixture distribution)](#18--mixture-distribution)
  - [1.9 ⭐ 优惠券收集者问题 — E 和 Var](#19--coupon-collector--e-and-var)
  - [1.10 ⭐ 随机游走击中时间 / 赌徒破产](#110--random-walk-hitting-time--gamblers-ruin)
  - [1.11 ⭐ 相关性三角不等式 (Jane Street)](#111--correlation-triangle-inequality-jane-street)
  - [1.12 ⭐ 圣彼得堡悖论](#112--st-petersburg-paradox)
  - [1.13 ⭐ Wald 恒等式 & 停止时间下的期望和](#113--walds-identity--expected-sum-at-stopping-time)
- [主题 2：多数投票分类器](#topic-2--majority-vote-classifier)
  - [2.0 原题 (Original problem)](#20--original-problem)
  - [2.1 N 个分类器 @ p 的多数投票通用公式](#21--general-n-classifiers--p-majority-vote)
  - [2.2 5 个 70% 准确率的分类器](#22--5-classifiers-at-70-accuracy)
  - [2.3 异质分类器（不同准确率）](#23--heterogeneous-classifiers-different-accuracies)
  - [2.4 加权多数投票 (Weighted majority vote)](#24--weighted-majority-vote)
  - [2.5 Condorcet 陪审团定理（渐近）](#25--condorcets-jury-theorem-asymptotic)
  - [2.6 相关分类器 — 独立性失效](#26--correlated-classifiers--independence-breaks)
  - [2.7 硬投票 vs 软投票 (Hard vote vs soft vote)](#27--hard-vote-vs-soft-vote)
  - [2.8 偶数票数（平局打破）](#28--even-number-of-voters-tie-breaking)
  - [2.9 准确率 < 50% 的分类器（集成何时反受其害）](#29--below-50-classifiers-when-ensembling-hurts)
  - [2.10 ⭐ AdaBoost 训练误差界](#210--adaboost-training-error-bound)
  - [2.11 ⭐ Bagging 的偏差-方差分解](#211--biasvariance-decomposition-for-bagging)
  - [2.12 ⭐ 多类多数投票 (Borda vs plurality)](#212--multi-class-plurality-voting-borda-vs-plurality)
  - [2.13 ⭐ Stacking 配 logistic-regression meta-learner](#213--stacking-with-logistic-regression-meta-learner)
  - [2.14 ⭐ 集成误差的 Hoeffding 边界](#214--hoeffding-bound-on-ensemble-error)
- [速查表 (Cheat Sheet)](#cheat-sheet)
- [参考资料 (Sources)](#sources)

---

# 主题 1 — 期望值 & 方差 (Topic 1 — Expected Value & Variance)

> **离散 X 的通用模板**
>
> 1. `E[X] = Σ x · P(X=x)`
> 2. `E[X²] = Σ x² · P(X=x)`
> 3. `Var(X) = E[X²] − (E[X])²`  *（手算时永远优先于定义式 `E[(X−μ)²]`）*
>
> 对于和与函数：`Var(aX + b) = a²·Var(X)`，`Var(X + Y) = Var(X) + Var(Y)` *当且仅当* X ⊥ Y。

---

## 1.0 — 原题 (Original problem)

> X 是离散随机变量，`P(X=0)=0.5`、`P(X=1)=0.4`、`P(X=6)=0.1`。求 E[X] 和 Var(X)。

```python
xs   = [0, 1, 6]
ps   = [0.5, 0.4, 0.1]

EX   = sum(x * p for x, p in zip(xs, ps))           # 0 + 0.4 + 0.6 = 1.0
EX2  = sum(x * x * p for x, p in zip(xs, ps))       # 0 + 0.4 + 3.6 = 4.0
varX = EX2 - EX ** 2                                # 4.0 - 1.0 = 3.0

# E[X] = 1.0,  Var(X) = 3.0,  SD = sqrt(3) ≈ 1.732
```

> 🎯 **面试提示：** 口头表达：*"期望值就是概率加权和。方差我会用 `E[X²] − (E[X])²`，因为少一次减法。"* 然后口算并念出来。

---

## 1.1 — 公平 6 面骰子 (Fair 6-sided die)

> 计算公平 6 面骰子结果 X 的 E[X] 和 Var(X)。

```python
EX   = sum(k for k in range(1, 7)) / 6              # 21/6 = 3.5
EX2  = sum(k * k for k in range(1, 7)) / 6          # 91/6
varX = EX2 - EX ** 2                                # 91/6 - 49/4 = 35/12 ≈ 2.917
```

| 统计量 | 值 | 闭式解 |
|---|---|---|
| E[X] | 3.5 | `(n+1)/2`（对 `1..n`） |
| Var(X) | 35/12 ≈ 2.917 | `(n²−1)/12` |

---

## 1.2 — 两颗骰子之和 (Sum of two dice)

> S = X₁ + X₂，两颗公平骰子。求 E[S] 和 Var(S)。

由线性性 (linearity) 和独立性：
- `E[S] = 3.5 + 3.5 = 7`
- `Var(S) = 35/12 + 35/12 = 35/6 ≈ 5.833`

```python
from collections import Counter
pmf = Counter()
for a in range(1, 7):
    for b in range(1, 7):
        pmf[a + b] += 1 / 36

ES   = sum(s * p for s, p in pmf.items())                     # 7
ES2  = sum(s * s * p for s, p in pmf.items())                 # 329/6
varS = ES2 - ES ** 2                                          # 35/6
```

---

## 1.3 — 两颗骰子的最大/最小值 (Jane Street)

> M = max(X₁, X₂)、m = min(X₁, X₂)，X₁、X₂ 独立同分布的公平骰子。

**技巧：** 用 `P(M ≤ k) = (k/6)²` 得 `P(M=k) = ((k)² − (k−1)²)/36 = (2k−1)/36`。最小值对称：`P(m ≥ k) = ((7−k)/6)²`。

```python
EM = sum(k * (2 * k - 1) / 36 for k in range(1, 7))           # 161/36 ≈ 4.472
Em = sum(k * (2 * (7 - k) - 1) / 36 for k in range(1, 7))     # 91/36  ≈ 2.528

# Sanity check: M + m = X1 + X2 always, so E[M] + E[m] = 7. ✓
assert abs(EM + Em - 7) < 1e-9
```

`Var(M) = E[M²] − E[M]² = 791/36 − (161/36)² ≈ 1.97`。

> 🎯 **Follow-up：** *为什么 `E[M] + E[m] = E[X₁] + E[X₂]`？* 因为 `max + min = a + b` 是逐点等式。期望的线性性自然成立。

---

## 1.4 — Bernoulli / Binomial — 推导 E 和 Var

> X ~ Bernoulli(p)。则 Y = X₁ + ... + Xₙ ~ Binomial(n, p)。求两者的 E 和 Var。

**Bernoulli：** `E[X] = p`、`E[X²] = p`（因为 X² = X），所以 `Var(X) = p − p² = p(1−p)`。

**Binomial：** 由线性性和独立性：
- `E[Y] = np`
- `Var(Y) = np(1−p)`

```python
def binom_moments(n, p):
    return n * p, n * p * (1 - p)

# Most asked at FAANG: "X is # heads in 10 flips of a fair coin. E? Var?"
EY, vY = binom_moments(10, 0.5)                                # 5.0, 2.5
```

---

## 1.5 — 几何分布 — 等待时间 (Geometric distribution — waiting time)

> "正面概率为 `p` 的硬币，平均要翻多少次才能首次出现正面？"

X ~ Geometric(p)（计算包括成功在内的总次数）：
- `E[X] = 1/p`
- `Var(X) = (1−p)/p²`

```python
def geometric_moments(p):
    return 1 / p, (1 - p) / p ** 2

# Fair coin -> E[X]=2, Var(X)=2
```

**E 的巧妙推导：** 对第一次翻转条件化：
`E[X] = p·1 + (1−p)·(1 + E[X])  ⇒  E[X] = 1/p`。

---

## 1.6 — 偏置硬币：贝叶斯翻转 (Biased coin: Bayesian flip, Facebook)

> "有两枚硬币：公平（50/50）和偏置（始终反面）。等概率随机选一枚，翻 5 次 —— 全是反面。你选到偏置硬币的概率是多少？"

贝叶斯：

`P(biased | 5T) = P(5T | biased)·0.5 / [P(5T | biased)·0.5 + P(5T | fair)·0.5] = 1 / (1 + (1/2)⁵) = 32/33 ≈ 0.970`。

```python
p_biased_prior = 0.5
p_fair_prior   = 0.5
p_5T_biased    = 1.0
p_5T_fair      = 0.5 ** 5

posterior = p_5T_biased * p_biased_prior / (p_5T_biased * p_biased_prior +
                                             p_5T_fair * p_fair_prior)
# 0.96970 ≈ 32/33
```

> 🎯 **推广：** 连续 `k` 次反面，后验 = `2ᵏ / (2ᵏ + 1)`。

---

## 1.7 — 复合和 — Law of Total Variance

> 一家店每天有 `N ~ Poisson(λ)` 个顾客。每个顾客花费 `Xᵢ`（独立同分布，均值 `μ`，方差 `σ²`）。求 `S = Σᵢ Xᵢ` 的 E[S] 和 Var(S)。

**Eve 定律：** `Var(S) = E[Var(S|N)] + Var(E[S|N])`。

- `E[S|N] = Nμ` ⇒ `E[S] = E[N]·μ = λμ`
- `Var(S|N) = Nσ²` ⇒ `E[Var(S|N)] = λσ²`
- `Var(E[S|N]) = μ²·Var(N) = μ²·λ`
- **`Var(S) = λ(σ² + μ²) = λ·E[X²]`**

```python
def compound_poisson(lam, mu, sigma2):
    ES   = lam * mu
    varS = lam * (sigma2 + mu ** 2)
    return ES, varS

# Example: λ=100 customers, each spends with μ=$20, σ²=$30 -> E[S]=$2000, Var(S)=43000
```

这是经典的 "tower / Eve 定律" 面试题。

---

## 1.8 — 混合分布 (Mixture distribution)

> 一枚硬币正面概率为 `p`（`p` 来自 `Beta(α,β)`）。求单次翻转的无条件 E 和 Var。

`E[X] = E[E[X|p]] = E[p] = α/(α+β)`。方差用混合方差公式。

```python
def beta_bernoulli_moments(a, b):
    p_mean = a / (a + b)
    p_var  = (a * b) / ((a + b) ** 2 * (a + b + 1))
    EX     = p_mean
    # E[X²] = E[E[X²|p]] = E[p]
    EX2    = p_mean
    varX   = EX2 - EX ** 2                       # = p_mean*(1-p_mean), same as Bernoulli with p_mean
    return EX, varX

# Note Var(X) only depends on E[p], not Var(p), because a single Bernoulli has variance p(1-p).
```

> 易被忽略的技巧：即便 `p` 本身随机，单次二元 X 仍然方差是 `μ(1−μ)`。`p` 的方差只在你观察 **同一硬币的多次翻转** 时才发挥作用。

---

# ⭐ 主题 1 — 进阶变体 (Harder Variants)

> 从一次性 E/Var 计算升级到 **依赖随机变量之和**（优惠券收集者）、**随机过程**（随机游走、Wald 恒等式）、**相关矩阵约束**（Jane Street 经典题）和 **探究效用理论的悖论**。

---

## 1.9 — 优惠券收集者问题 — E 和 Var

> "有 `n` 张不同的优惠券。每次随机抽一张（有放回）。问收齐所有的次数的 E 和 Var。"

**分解** 为各阶段等待时间。当已有 `i−1` 张不同时，下次抽到新的概率是 `(n − i + 1)/n`，所以 `Tᵢ ~ Geometric(p_i)`，`p_i = (n−i+1)/n`。Tᵢ 相互独立。

- `E[T] = Σᵢ 1/pᵢ = n · Hₙ ≈ n·(ln n + γ)`（γ ≈ 0.5772 — Euler-Mascheroni 常数）
- `Var[T] = Σᵢ (1−pᵢ)/pᵢ² = n²·Σᵢ 1/i² − n·Hₙ ≈ (π²/6)·n² − n·ln n`

```python
import math

def coupon_collector_moments(n: int):
    H_n      = sum(1 / k for k in range(1, n + 1))
    sum_inv2 = sum(1 / k ** 2 for k in range(1, n + 1))
    EX  = n * H_n
    VarX = n * n * sum_inv2 - n * H_n
    return EX, VarX

# n=10  -> E ≈ 29.29, SD ≈ 11.21
# n=52  -> E ≈ 235.97 (full deck of cards), SD ≈ 61.5
# n=100 -> E ≈ 518.74
```

> 🎯 **Quant follow-up：** *"超过 `2·n·ln n` 次的概率有多大？"* 用 Markov：`P(T > 2·E[T]) ≤ 1/2`。Chebyshev 给出更紧界：`P(|T−E[T]| > c·SD) ≤ 1/c²`。

---

## 1.10 — 随机游走击中时间 / 赌徒破产 (Random walk hitting time / Gambler's Ruin)

> "在 `Z` 上的对称简单随机游走，从位置 `i` 出发，吸收边界为 `0` 和 `N`（`0 < i < N`）。期望被吸收前的步数？"

**递推 (Recurrence).** 设 `h(i) = E[T | start at i]`。`h(0) = h(N) = 0`，对 `0 < i < N`：

`h(i) = 1 + 0.5 · h(i−1) + 0.5 · h(i+1)`

**闭式解：** `h(i) = i · (N − i)`。

```python
def gamblers_ruin_expected_time(i: int, N: int) -> int:
    return i * (N - i)

def gamblers_ruin_win_prob(i: int, N: int, p: float = 0.5):
    """Asymmetric walk: prob of hitting N before 0, biased coin with prob p of +1."""
    if abs(p - 0.5) < 1e-12:
        return i / N
    q = 1 - p
    r = q / p
    return (1 - r ** i) / (1 - r ** N)

# i=10, N=20  -> E[T] = 100 steps until ruin or fortune
# Asymmetric: i=10, N=20, p=0.49 -> P(reach 20) ≈ 0.401  (drift kills you)
```

> 🎯 **难点：** 这是经典的"列出递推、解出闭式"题。方差更难：对称情形 `Var[T] = (1/3) · i · (N−i) · (N² + i(N−i) − 2)`。

---

## 1.11 — 相关性三角不等式 (Jane Street 真题)

> "如果 `Corr(X, Y) = 0.9`、`Corr(Y, Z) = 0.8`，`Corr(X, Z)` 的 **最小** 和 **最大** 可能值是多少？"

**思路：** 3×3 相关矩阵必须半正定 (positive semi-definite, PSD)。设 `r_xy=0.9`、`r_yz=0.8`、`r_xz=ρ`。PSD 要求 `det ≥ 0`：

`det = 1 − r_xy² − r_yz² − ρ² + 2·r_xy·r_yz·ρ ≥ 0`

解关于 ρ 的二次：

`ρ ∈ [r_xy·r_yz − √((1−r_xy²)(1−r_yz²)), r_xy·r_yz + √((1−r_xy²)(1−r_yz²))]`

```python
import math

def correlation_bounds(r_xy: float, r_yz: float):
    discriminant = (1 - r_xy ** 2) * (1 - r_yz ** 2)
    half_width = math.sqrt(discriminant)
    center     = r_xy * r_yz
    return center - half_width, center + half_width

correlation_bounds(0.9, 0.8)
# (0.4582, 0.9818)  -- so Corr(X,Z) is in [~0.46, ~0.98]
```

> 🎯 **几何直觉：** 把单位方差的随机变量看作单位向量；相关性 = 向量夹角的余弦。三个角必须满足三角不等式 `|θ_xy − θ_yz| ≤ θ_xz ≤ θ_xy + θ_yz`。这恰好就是上面的公式所表达的。

---

## 1.12 — 圣彼得堡悖论 (St. Petersburg paradox)

> "公平硬币翻到第一次正面为止。若第一次正面在第 `k` 次出现，你赢得 `2^k` 美元。期望收益是多少？你愿意付多少入场费？"

`E[X] = Σ_{k=1}^{∞} (1/2)^k · 2^k = Σ 1 = ∞`。

```python
def st_petersburg_truncated_expectation(max_k: int) -> float:
    return float(max_k)            # diverges; truncating to max_k flips gives EV = max_k

# After max_k=30 (≈ casino's bankroll), EV is just $30. So nobody pays $1M.
```

**用效用理论解决：** 用 log 效用 `u(x) = log x`，期望效用 = `Σ 2^{-k}·log(2^k) = log 2 · Σ k·2^{-k} = 2·log 2`。所以 log 效用下的确定性等价值 (certainty equivalent) 是 `e^{2·log 2} = 4` —— 也就是说你大约愿意付 $4。

> 🎯 **意义：** 量化基金的行为金融 / 决策理论问题。知道解决方法（效用理论、有界支付、Cramér 变换）能体现深度。

---

## 1.13 — Wald 恒等式 & 停止时间下的期望和

> "独立同分布随机变量 `X₁, X₂, …`，均值 `μ`。`N` 是停止时间，`E[N] < ∞`。证明 `E[Σ_{i=1}^{N} Xᵢ] = μ · E[N]`。"

这就是 **Wald 恒等式 (Wald's identity)**。关键微妙之处：`N` 可以依赖 X，但作为 *停止时间* —— 第 `n` 步的决策只依赖 `X₁..Xₙ`，不依赖未来。

**应用：** "公平硬币一直翻到看到连续 3 次正面为止。期望翻几次？"

状态图：「连续 0 次」→「连续 1 次」→「连续 2 次」→「连续 3 次（吸收态）」。

```python
def expected_flips_to_pattern(pattern: str, p_head: float = 0.5):
    """Expected coin flips until we see consecutive 'pattern' (e.g. 'HHH', 'HTH')."""
    # KMP-style failure-function trick: E[T] = sum 1/p_i where p_i = 2^{-i}
    # for autocorrelation. For 'HHH': 2 + 4 + 8 = 14 flips.
    # For 'HTH': overlapping pattern -> 2 + 8 = 10 flips.
    # General: E[T] = sum over autocorrelation matches of 1/(P(prefix))
    n = len(pattern)
    EX = 0
    for k in range(n):
        if pattern[:k+1] == pattern[n-k-1:]:
            EX += 1 / (p_head ** sum(c == 'H' for c in pattern[:k+1]) *
                       (1 - p_head) ** sum(c == 'T' for c in pattern[:k+1]))
    return EX

# Fair coin:
# expected_flips_to_pattern('HHH') -> 14
# expected_flips_to_pattern('HTH') -> 10
# expected_flips_to_pattern('HTT') -> 8     (no self-overlap, lowest)
```

反直觉的洞察：尽管两者概率都是 `1/8`，但 `'HHH'` 的期望等待时间 **比 `'HTH'` 长** —— 因为 `'HHH'` 与自身高度重叠。

> 🎯 **面试金句：** *"概率相同的两个模式，期望等待时间居然不同！"* 如果你能用自相关 (autocorrelation) 推导出来，就完美。

---

# 主题 2 — 多数投票分类器 (Topic 2 — Majority-Vote Classifier)

> **通用公式** —— `n` 个独立的二元投票者，每人正确概率为 `p`。多数投票（n 为奇数）正确当且仅当至少 `⌈n/2⌉` 人正确。
>
> `P(majority correct) = Σ_{k=⌈n/2⌉}^{n} C(n,k) · pᵏ · (1−p)^(n−k)`

---

## 2.0 — 原题 (Original problem)

> 三个独立的二元分类器各 80% 准确率。多数投票的准确率？

```python
from math import comb
p, n = 0.8, 3
acc  = sum(comb(n, k) * p**k * (1 - p)**(n - k) for k in range(2, n + 1))
# 3 * 0.64 * 0.2 + 1 * 0.512 = 0.384 + 0.512 = 0.896
```

**结果：89.6%。** 集成把准确率提升了约 10 个百分点。

> 🎯 **面试提示：** 明确陈述假设：*"这是 `p=0.8` 时 ≥2 人正面的二项概率，**假设独立**。如果分类器相关，集成的提升会缩水。"*

---

## 2.1 — N 个分类器 @ p 的多数投票通用公式

```python
from math import comb

def majority_vote_accuracy(n: int, p: float) -> float:
    """n independent classifiers each correct with prob p. n is odd."""
    if n % 2 == 0:
        raise ValueError("Use n odd to avoid ties; see Section 2.8 for even-n.")
    threshold = n // 2 + 1
    return sum(comb(n, k) * p**k * (1 - p)**(n - k) for k in range(threshold, n + 1))

# A few useful data points
for n in [1, 3, 5, 7, 11, 21, 51, 101]:
    print(n, round(majority_vote_accuracy(n, 0.8), 4))
# 1 -> 0.8000   3 -> 0.8960   5 -> 0.9421   7 -> 0.9667
# 11 -> 0.9883  21 -> 0.9988  51 -> 0.99999  101 -> ~1
```

`P` 随 `n → ∞` 单调趋近 1。（这就是 Condorcet 陪审团定理 —— 见 2.5。）

---

## 2.2 — 5 个 70% 准确率的分类器

> "如果集成 3 个 70% 准确率的分类器，多数投票准确率？5 个呢？"

```python
# n=3, p=0.7
print(majority_vote_accuracy(3, 0.7))   # 0.784
# n=5, p=0.7
print(majority_vote_accuracy(5, 0.7))   # 0.83692
```

3 → 5 提升约 5 pp；再翻倍到 11 得到约 92.2%。**收益递减** —— 边际增益约以 `1/√n` 衰减。

---

## 2.3 — 异质分类器（不同准确率）

> 3 个独立分类器，准确率 `p₁ = 0.9`、`p₂ = 0.7`、`p₃ = 0.6`。多数投票的准确率？

枚举 `2³ = 8` 种正确性模式：

```python
from itertools import product

def heterogeneous_majority(ps):
    n = len(ps)
    threshold = n // 2 + 1
    total = 0.0
    for outcome in product([0, 1], repeat=n):                 # 1 = correct, 0 = wrong
        prob = 1.0
        for o, p in zip(outcome, ps):
            prob *= p if o else (1 - p)
        if sum(outcome) >= threshold:
            total += prob
    return total

heterogeneous_majority([0.9, 0.7, 0.6])
# = P(all 3 correct) + sum of P(exactly 2 correct)
# = 0.378 + (0.9*0.7*0.4 + 0.9*0.3*0.6 + 0.1*0.7*0.6)
# = 0.378 + 0.252 + 0.162 + 0.042 = 0.834
```

**关键教训：** 当其他模型很弱时，异质集成可能比单一最佳分类器更差 —— 试 `[0.95, 0.5, 0.5]` 得 0.7250，*不如* 单独的 0.95 模型。

---

## 2.4 — 加权多数投票 (Weighted majority vote)

> 三个分类器，准确率 `p = [0.9, 0.7, 0.6]`、权重 `w = [3, 1, 1]`。如果对二元标签做加权投票（加权计数大者获胜，平局任选），准确率多少？

```python
from itertools import product

def weighted_vote_accuracy(ps, ws, tie_break_correct=False):
    n = len(ps)
    total_w = sum(ws)
    total = 0.0
    for outcome in product([0, 1], repeat=n):                 # 1 = correct, 0 = wrong
        prob = 1.0
        for o, p in zip(outcome, ps):
            prob *= p if o else (1 - p)
        correct_weight = sum(w for o, w in zip(outcome, ws) if o)
        if correct_weight > total_w / 2:
            total += prob
        elif correct_weight == total_w / 2:                   # tie
            total += prob * (1.0 if tie_break_correct else 0.5)
    return total

weighted_vote_accuracy([0.9, 0.7, 0.6], [3, 1, 1])  # ≈ 0.900
```

最强模型权重很大时，集成准确率约等于该分类器；最优 log-likelihood-ratio 权重（Berend & Sapir）是 `wᵢ = log(pᵢ/(1−pᵢ))`。

```python
import math
def lr_weights(ps):
    return [math.log(p / (1 - p)) for p in ps]
# [2.197, 0.847, 0.405]   -- the strong model gets ~5.4× the weight of the weak one
```

---

## 2.5 — Condorcet 陪审团定理（渐近）

> "如果 `p > 0.5` 且分类器独立，多数投票准确率随 `n → ∞` 会怎样？"

**定理 (Condorcet, 1785).** `n` 为奇数、独立同分布且正确概率 `p > 0.5`：

`P(majority correct) → 1`，当 `n → ∞`。
若 `p < 0.5`，极限为 0。若 `p = 0.5`，恒为 0.5。

这是集成 (ensembling) 和 bagging 的理论基础。基于 CLT 的快速近似：

`P(correct) ≈ Φ((p − 0.5)·√n / √(p(1−p)))`

```python
from math import sqrt
from statistics import NormalDist

def normal_majority_approx(n, p):
    z = (p - 0.5) * sqrt(n) / sqrt(p * (1 - p))
    return NormalDist().cdf(z)

# n=51, p=0.6 -> exact 0.9282 vs normal-approx 0.9242
```

---

## 2.6 — 相关分类器 — 独立性失效

> "如果三个 80% 分类器完全相关（总是给出相同答案），多数投票准确率？"

如果 correlation = 1，集成 *就是* 一个分类器：**80%**。独立性假设撑起了所有的提升。实际中相关性介于 0 和 1 之间，集成准确率被独立情形上界。

边际准确率不变但预测共享一个公共信号时的快速模拟：

```python
import random

def correlated_majority(p, n, rho, trials=100_000):
    """
    Latent-variable model: each classifier i is correct with prob p, but
    its decision is shifted by a shared latent component with corr rho.
    """
    correct = 0
    for _ in range(trials):
        shared = random.gauss(0, 1)
        votes = 0
        for _ in range(n):
            indep = random.gauss(0, 1)
            score = (rho ** 0.5) * shared + ((1 - rho) ** 0.5) * indep
            # threshold so that P(score < tau) = p
            from statistics import NormalDist
            tau = NormalDist().inv_cdf(p)
            if score < tau:
                votes += 1
        if votes >= (n + 1) // 2:
            correct += 1
    return correct / trials

# rho=0   -> approaches 0.896 (independent case)
# rho=0.5 -> ~0.86
# rho=1.0 -> 0.80
```

> 🎯 **要点：** 让集成多样化（不同架构、不同特征子集、不同 bootstrap 样本的 bagging）才能让数学奏效。

---

## 2.7 — 硬投票 vs 软投票 (Hard vote vs soft vote)

> "如果每个分类器输出 *概率* 而不是标签，你还会按标签投票吗？什么时候软投票胜过硬投票？"

- **硬投票：** 多数类预测胜出。丢失置信度信息。
- **软投票：** 平均每类概率，取 argmax。当分类器校准良好时更优。

```python
def hard_vote(probs):       # probs: list of P(y=1) per classifier
    votes = sum(p > 0.5 for p in probs)
    return 1 if votes > len(probs) / 2 else 0

def soft_vote(probs):
    return 1 if sum(probs) / len(probs) > 0.5 else 0

# Example: probs = [0.49, 0.49, 0.95]
# hard -> 1 vote for 1, 2 votes for 0 -> predicts 0
# soft -> mean 0.643 -> predicts 1 (the one confident classifier overrides)
```

软投票要求概率被校准 (calibrated)；否则一个过度自信的弱模型会主导结果。

---

## 2.8 — 偶数票数（平局打破）

> "如果用 4 个 80% 准确率的分类器呢？"

```python
def even_majority_accuracy(n, p, tie_correct=0.5):
    """Strict majority needs >n/2 correct. On a tie, half-credit is conventional."""
    from math import comb
    total = 0
    for k in range(0, n + 1):
        prob = comb(n, k) * p**k * (1 - p)**(n - k)
        if k > n / 2:
            total += prob
        elif k == n / 2:
            total += prob * tie_correct
    return total

even_majority_accuracy(4, 0.8)
# strict-majority correct: 4 right + 3 right = 0.4096 + 0.4096 = 0.8192
# half tie credit: + 0.5 * (6 * 0.64 * 0.04) = + 0.0768  -> 0.896
```

惊人的结果：**4 个分类器（带平局打破）= 3 个分类器**，对 `p=0.8` 都是 0.896。所以面试惯例：永远用 *奇数*。

---

## 2.9 — 准确率 < 50% 的分类器（集成何时反受其害）

> "三个 30% 准确率的二元任务'分类器'。多数投票准确率？"

```python
majority_vote_accuracy(3, 0.3)  # 0.216
```

低于 50% 时，**集成反而更差**（Condorcet 诅咒）。但你可以翻转每个分类器的输出，得到三个 70% 的 —— 它们的多数投票得 0.784。两者对偶：`acc(majority of n @ p) + acc(majority of n @ 1−p) = 1`。

---

# ⭐ 主题 2 — 进阶变体 (Harder Variants)

> 从"二项概率"升级到 **自适应权重**（AdaBoost）、集成的 **偏差-方差分解**（bagging）、**多类** 投票、**stacking** 加 meta-learner，以及 **PAC 风格的边界**（Hoeffding）。

---

## 2.10 — AdaBoost 训练误差界

> "每轮 AdaBoost 用加权误差 `εₜ < 0.5` 训练弱学习器，设 `αₜ = ½ ln((1−εₜ)/εₜ)`，然后重新加权样本。证明 T 轮后训练误差不超过 `Π_t 2·√(εₜ(1−εₜ))`。"

**关键事实 (Schapire & Freund).** 定义 `γₜ = 0.5 − εₜ`（第 `t` 轮的"优势"）。则：

`training_error ≤ exp(−2 · Σ_t γₜ²)`

所以只要每个弱学习器比随机略好（`γ ≥ γ₀ > 0`），训练误差就会随 T **指数** 衰减。

```python
import math, numpy as np

def adaboost_simple(X, y, n_rounds, weak_learner_fn):
    """y in {-1, +1}. weak_learner_fn(X, y, w) returns (predictions, eps)."""
    n = len(y)
    w = np.ones(n) / n
    learners, alphas = [], []
    for t in range(n_rounds):
        h, eps = weak_learner_fn(X, y, w)
        eps = max(eps, 1e-10)                                    # avoid div-zero
        alpha = 0.5 * math.log((1 - eps) / eps)
        # Re-weight: up-weight misclassified
        w = w * np.exp(-alpha * y * h)
        w = w / w.sum()                                          # re-normalize
        learners.append(h); alphas.append(alpha)
    return learners, alphas

def adaboost_predict(learners, alphas):
    """Final prediction = sign(Σ αₜ hₜ(x))."""
    H = sum(a * h for a, h in zip(alphas, learners))
    return np.sign(H)

# Training error after T rounds is bounded by:  product over t of 2*sqrt(eps_t * (1-eps_t))
# If every eps_t = 0.4 (edge γ=0.1), bound = (2·sqrt(0.24))^T = 0.98^T  ->  fast decay
```

> 🎯 **面试金句：** *"AdaBoost 是指数损失上的坐标下降。α 公式是从一维最小化推出来的。"*

---

## 2.11 — Bagging 的偏差-方差分解

> "为什么 bagging 减少方差但不减少偏差？把数学讲清楚。"

**设定：** 学习器从训练集 `D` 产生估计 `f̂(x; D)`。Bagging 平均 `M` 个 bootstrap 训练的估计：`f̄(x) = (1/M) Σₘ f̂(x; Dₘ)`。

对 **平方误差** 情形分解：

`E[(f̄(x) − y)²] = bias²(f̄) + Var(f̄) + σ²`

- **偏差：** `E[f̄(x)] ≈ E[f̂(x; D)]`（bootstrap 副本大致同分布）→ **偏差不变**。
- **方差：** 设 `Var(f̂) = σ_f²`，bagged 学习器之间成对相关性为 `ρ`：

  `Var(f̄) = ρ·σ_f² + (1−ρ)/M · σ_f² → ρ·σ_f²`，当 `M → ∞`。

所以 bagging 消除了方差中 **独立的** 部分，但留下相关残差 `ρ·σ_f²`。**Random Forests** 通过随机特征子采样进一步去相关（更小的 ρ）。

```python
import numpy as np

def bagging_variance(sigma_f2, rho, M):
    """Variance of average of M correlated estimators."""
    return rho * sigma_f2 + (1 - rho) / M * sigma_f2

# sigma_f2=1, rho=0.5
print([bagging_variance(1, 0.5, M) for M in [1, 5, 20, 100, 1000]])
# [1.0, 0.6, 0.525, 0.505, 0.5005] -- floored at rho=0.5

# Random forest reduces rho (e.g. to 0.1) -> floor drops to 0.1
print([bagging_variance(1, 0.1, M) for M in [1, 5, 20, 100, 1000]])
# [1.0, 0.28, 0.145, 0.109, 0.1009]
```

> 🎯 **面试加分：** 写出 `Var(f̄)` 的公式，并解释为什么 **去相关 (decorrelation)**（RF 的特征 bagging、模型多样性）才是关键，而不仅是"更多树"。

---

## 2.12 — 多类多数投票 (Borda vs plurality)

> "K 个分类器，每个预测 `C` 类中的一个。当 `C > 2` 时，哪种投票规则最大化准确率？"

`C = 2` 时 plurality = majority。`C > 2` 时，**plurality**（"票数最多类胜出"）在没有任何类获得多数时会失败。**Borda count** 用排序预测：每个分类器给类排名 1..C，类 `c` 累计 `C − rank_i(c)` 分。总分最高者胜。

```python
from collections import Counter

def plurality(votes):
    """votes: list of class labels (one per classifier)."""
    return Counter(votes).most_common(1)[0][0]

def borda_count(rankings):
    """rankings: list of lists, where rankings[i] is classifier i's ranking
       (rankings[i][0] = top choice, rankings[i][-1] = worst)."""
    scores = Counter()
    C = len(rankings[0])
    for ranking in rankings:
        for r, cls in enumerate(ranking):
            scores[cls] += C - 1 - r
    return scores.most_common(1)[0][0]

# Example: 3 classifiers ranking 4 classes
rankings = [['A', 'B', 'C', 'D'],
            ['B', 'A', 'C', 'D'],
            ['C', 'A', 'B', 'D']]
plurality([r[0] for r in rankings])    # 'A' (only one wins; others tie)
borda_count(rankings)                  # 'A' (8 pts vs B=6, C=5, D=0)
```

> 🎯 **意义：** 软投票（平均预测概率）本质上是连续的 Borda。对校准良好的分类器，它严格优于 plurality。

---

## 2.13 — Stacking 配 logistic-regression meta-learner

> "不投票，而是在基分类器输出上训练一个 *meta-learner*。怎么做？为什么更好？"

**Stacking：**
1. 用交叉验证训练基分类器 `h₁, ..., hₖ`，生成 **out-of-fold 预测** `ẑᵢⱼ = hⱼ(xᵢ)`（避免数据泄露）。
2. 在 `(ẑᵢ, yᵢ)` 对上训练 meta-learner `g`（通常是 logistic regression）。
3. 推理时：每个 `hⱼ` 预测，再传给 `g`。

```python
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def stacking(X, y, base_learners, meta_learner=None, n_folds=5):
    n, K = len(y), len(base_learners)
    oof_preds = np.zeros((n, K))                                # out-of-fold predictions
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    for train_idx, val_idx in kf.split(X):
        for k, learner in enumerate(base_learners):
            learner_clone = learner.__class__(**learner.get_params())
            learner_clone.fit(X[train_idx], y[train_idx])
            oof_preds[val_idx, k] = learner_clone.predict_proba(X[val_idx])[:, 1]
    # Train meta-learner on OOF predictions
    meta_learner = meta_learner or LogisticRegression()
    meta_learner.fit(oof_preds, y)
    # Re-fit base learners on full data
    for learner in base_learners:
        learner.fit(X, y)
    return base_learners, meta_learner

# Usage:
# base = [DecisionTreeClassifier(), RandomForestClassifier(), LogisticRegression()]
# bases, meta = stacking(X_train, y_train, base)
# preds = meta.predict_proba(np.column_stack([b.predict_proba(X_test)[:, 1] for b in bases]))
```

**为什么比投票好：** meta-learner 学习在 *什么条件下* 信任 *哪个* 基分类器，而不是固定权重。Logistic-regression meta-learner 给出可解释的权重，与最优 log-likelihood-ratio 形式一致（见 2.4 节）。

---

## 2.14 — 集成误差的 Hoeffding 边界

> "我有 `n` 个独立分类器各 `p > 0.5` 准确率。需要多少个才能让多数投票误差以 ≥ 1−δ 概率低于 ε？"

**Hoeffding 不等式**（样本均值集中）。对独立同分布 {0,1} 指示变量 `Yᵢ`（分类器 i 的正确性，均值 `p`）：

`P(Ȳₙ ≤ p − γ) ≤ exp(−2 n γ²)`

多数错误当且仅当 `Ȳₙ ≤ 0.5`，即偏差 `γ = p − 0.5`。所以：

`P(majority wrong) ≤ exp(−2n(p − 0.5)²)`

解 `n`：`n ≥ ln(1/δ) / (2·(p − 0.5)²)`。

```python
import math

def hoeffding_n_for_error(p: float, epsilon: float) -> int:
    """How many independent classifiers needed so majority error <= epsilon."""
    if p <= 0.5: raise ValueError("Need p > 0.5 for the bound to be useful.")
    return math.ceil(math.log(1 / epsilon) / (2 * (p - 0.5) ** 2))

# p = 0.55, target majority error 0.01:
hoeffding_n_for_error(0.55, 0.01)   # 921 classifiers
# p = 0.7, target 0.01:
hoeffding_n_for_error(0.7, 0.01)    # 116
# p = 0.9, target 0.01:
hoeffding_n_for_error(0.9, 0.01)    # 29
```

> 🎯 **面试提示：** *"Hoeffding 给出的是非渐近的、distribution-free 的边界。CLT（2.5 节）给出更紧的近似但只在极限情况下成立。需要有限 `n` 保证时，Hoeffding 是首选。"*

---

# 速查表 (Cheat Sheet)

## 期望值 & 方差 — 必备公式

| 分布 | E[X] | Var(X) |
|---|---|---|
| Bernoulli(p) | p | p(1−p) |
| Binomial(n, p) | np | np(1−p) |
| Geometric(p)（包括成功的次数） | 1/p | (1−p)/p² |
| Uniform on {1..n} | (n+1)/2 | (n²−1)/12 |
| Poisson(λ) | λ | λ |
| Uniform[a, b] | (a+b)/2 | (b−a)²/12 |
| Exponential(λ) | 1/λ | 1/λ² |
| Normal(μ, σ²) | μ | σ² |

| 等式 | 适用场景 |
|---|---|
| `Var(X) = E[X²] − (E[X])²` | 永远比定义式快 |
| `Var(aX + b) = a²Var(X)` | 线性变换 |
| `Var(X+Y) = Var(X) + Var(Y) + 2Cov(X,Y)` | 联合矩 |
| `E[E[Y\|X]] = E[Y]` | Tower / 全期望 |
| `Var(Y) = E[Var(Y\|X)] + Var(E[Y\|X])` | Eve 定律 / 全方差 |
| `E[T] = n · H_n`、`Var(T) ≈ π²n²/6` | 优惠券收集者 |
| `E[T_i] = i(N − i)`，对称游走 | 赌徒破产击中时间 |
| `\|ρ_xz − ρ_xy ρ_yz\| ≤ √((1−ρ_xy²)(1−ρ_yz²))` | 相关性三角 |
| `E[Σ Xᵢ] = μ · E[N]` | 停止时间下的 Wald 恒等式 |

## 多数投票分类器 — 模式识别

| 场景 | 公式 / 思路 |
|---|---|
| n 奇数、独立同分布 p | `Σ_{k=⌈n/2⌉..n} C(n,k) p^k (1-p)^(n-k)` |
| 异质准确率 | 枚举 `2^n` 种正确性模式 |
| 加权投票 | 求正确投票者权重和；与 total/2 比较 |
| 最优权重 | `w_i = log(p_i / (1 − p_i))`（对数似然比） |
| 渐近，p > 0.5、独立 | 准确率 → 1（Condorcet） |
| 相关投票者 | 准确率介于 p 和独立情形之间 |
| 软投票 | 平均概率取 argmax — 需校准 |
| n 偶数 | 严格多数较低；惯例：平局给一半积分 |
| p < 0.5 | 集成 **降低** 准确率 → 0 |

> **面试中始终明示假设：** 独立性、校准、等权、`n` 为奇数。一半的 follow-up 都在探究某个假设失效会怎样。

---

# 参考资料 (Sources)

## 主题 1 — 期望值 & 方差

- [40 个 FAANG 与华尔街问的概率统计数据科学面试题 — NickSingh](https://www.nicksingh.com/posts/40-probability-statistics-data-science-interview-questions-asked-by-fang-wall-street)
- [概率（数据科学）面试题 — InterviewBit](https://www.interviewbit.com/probability-interview-questions/)
- [25 道概率统计题搞定数据科学面试 — Analytics Vidhya](https://www.analyticsvidhya.com/blog/2021/04/25-probability-and-statistics-questions-to-ace-your-data-science-interviews/)
- [30 道数据科学家概率统计面试题 — StrataScratch](https://www.stratascratch.com/blog/30-probability-and-statistics-interview-questions-for-data-scientists)
- [Top 25 Random Variables 面试题 — InterviewPrep](https://interviewprep.org/random-variables-interview-questions/)
- [离散随机变量的期望 — LibreTexts](https://stats.libretexts.org/Courses/Saint_Mary's_College_Notre_Dame/DSCI_500B_Essential_Probability_Theory_for_Data_Science_(Kuter)/03:_Discrete_Random_Variables/3.04:_Expected_Value_of_Discrete_Random_Variables)
- [离散随机变量的方差 — LibreTexts (Grinstead & Snell)](https://stats.libretexts.org/Bookshelves/Probability_Theory/Introductory_Probability_(Grinstead_and_Snell)/06:_Expected_Value_and_Variance/6.02:_Variance_of_Discrete_Random_Variables)
- [离散 PDF 的期望与方差 — LibreTexts (Geraghty)](https://stats.libretexts.org/Bookshelves/Introductory_Statistics/Inferential_Statistics_and_Probability_-_A_Holistic_Approach_(Geraghty)/06:_Discrete_Random_Variables/6.04:_Expected_Value_and_Variance_of_a_Discrete_Probability_Distribution_Function)
- [Jane Street 面试题 — 两骰子最大值期望 (Glassdoor)](https://www.glassdoor.com/Interview/What-is-the-expected-value-of-the-max-of-two-dice-QTN_133823.htm)
- [Jane Street 面试题 — 两骰子差值期望 (Glassdoor)](https://www.glassdoor.com/Interview/Here-s-an-easy-one-You-are-given-a-six-sided-die-What-is-the-expected-value-of-the-difference-between-the-two-dice-rolls-QTN_834923.htm)
- [ML Interview Q Series — 两骰子期望最大值 (Rohan Paul)](https://www.rohan-paul.com/p/ml-interview-q-series-calculating-80e)
- [一份概率题清单 Week 3 — Jerry Qin](https://jerryqin.com/posts/a-working-list-of-probability-problems-week-three/)
- [掷骰子期望 — 数据科学面试 (bugfree.ai)](https://bugfree.ai/data-question/expected-value-dice-roll)
- [概率面试题（硬币游戏）— Yashwanth Reddy, Medium](https://medium.com/@reddyyashu20/q1-you-and-your-friend-are-playing-a-game-with-a-fair-coin-7560914f7121)
- [偏置硬币面试题 — Henry George, Medium](https://medium.com/@hjegeorge/interview-question-1-a-biased-coin-toss-9dc2af96321)
- [全方差定律 — Wikipedia](https://en.wikipedia.org/wiki/Law_of_total_variance)
- [全期望定律 — Wikipedia](https://en.wikipedia.org/wiki/Law_of_total_expectation)
- [全方差定律 — The Book of Statistical Proofs](https://statproofbook.github.io/P/var-tot.html)
- [用全期望、方差、协方差解条件概率问题 — Saurabh Maheshwari, TDS Archive](https://medium.com/data-science/solving-conditional-probability-problems-with-the-laws-of-total-expectation-variance-and-c38c07cfebfa)
- [STAT 24400 Lecture 10 — 期望与方差 (UChicago)](https://www.stat.uchicago.edu/~yibi/teaching/stat244/L10.pdf)
- [条件方差 / 迭代期望 — probabilitycourse.com](https://www.probabilitycourse.com/chapter5/5_1_5_conditional_expectation.php)

## ⭐ 进阶变体 — 主题 1

- [优惠券收集者问题 — Brilliant Math & Science Wiki](https://brilliant.org/wiki/coupon-collector-problem/)
- [优惠券收集者问题 — Wikipedia](https://en.wikipedia.org/wiki/Coupon_collector's_problem)
- [优惠券收集者问题 — Tufts CS](https://www.cs.tufts.edu/comp/250P/classpages/coupon.html)
- [Coupon Collector's Problem: A Probability Masterpiece — TDS Archive](https://towardsdatascience.com/coupon-collectors-problem-a-probability-masterpiece-1d5aed4af439/)
- [Coupon Collector — 方差推导 (Quora)](https://www.quora.com/Whats-the-variance-of-the-number-of-coupons-a-coupon-collector-needs-to-collect-before-seeing-each-type)
- [随机算法 Lecture 6 — 优惠券收集者 (Patras)](https://www.ceid.upatras.gr/webpages/courses/randalgs/slides/lesson6.pdf)
- [击中时间 — Wikipedia](https://en.wikipedia.org/wiki/Hitting_time)
- [随机游走 — Wikipedia](https://en.wikipedia.org/wiki/Random_walk)
- [随机游走讲义 (Leiden, PDF)](https://prob.math.leidenuniv.nl/lecturenotes/RandomWalks.pdf)
- [击中时间 — Markov 过程笔记 (MATH2750)](https://mpaldridge.github.io/math2750/S08-hitting-times.html)
- [停止时间 — Karl Sigman, Columbia (PDF)](http://www.columbia.edu/~ks20/stochastic-I/stochastic-I-ST.pdf)
- [Jane Street 面试 — 相关性三角 (Glassdoor)](https://www.glassdoor.com/Interview/If-X-Y-and-Z-are-three-random-variables-such-that-X-and-Y-have-a-correlation-of-0-9-and-Y-and-Z-have-correlation-of-0-8-QTN_467199.htm)
- [协方差、相关性与联合概率 — AnalystPrep CFA](https://analystprep.com/cfa-level-1-exam/quantitative-methods/covariance-correlation-and-joint-probability/)
- [11 个最常见的相关性问题 — Analytics Vidhya](https://www.analyticsvidhya.com/blog/2015/06/correlation-common-questions/)
- [圣彼得堡悖论 & 效用理论（决策理论参考）](https://en.wikipedia.org/wiki/St._Petersburg_paradox)
- [Wald 恒等式 — Wikipedia](https://en.wikipedia.org/wiki/Wald%27s_equation)
- [布朗运动笔记 — Sigman, Columbia (PDF)](http://www.columbia.edu/~ww2040/4701Sum07/4701-06-Notes-BM.pdf)

## 主题 2 — 多数投票分类器

- [Voting Classifier — GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/voting-classifier/)
- [Voting Classifier 用 Sklearn — GeeksforGeeks](https://www.geeksforgeeks.org/ml-voting-classifier-using-sklearn/)
- [VotingClassifier — scikit-learn 文档](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)
- [EnsembleVoteClassifier — mlxtend](https://rasbt.github.io/mlxtend/user_guide/classifier/EnsembleVoteClassifier/)
- [用 Python 开发投票集成 — Machine Learning Mastery](https://machinelearningmastery.com/voting-ensembles-with-python/)
- [投票集成：硬投票 vs 软投票详解 — MCP Analytics](https://mcpanalytics.ai/articles/voting-ensemble-practical-guide-for-data-driven-decisions)
- [实现加权多数投票集成分类器 — Sebastian Raschka](https://sebastianraschka.com/Articles/2014_ensemble_classifier.html)
- [理解 ML 中的投票分类器 — Lomash Bhuva, Medium](https://medium.com/@lomashbhuva/understanding-voting-classifiers-in-machine-learning-a-comprehensive-guide-6589b5f17e0f)
- [40 个数据科学家集成建模面试题 — Analytics Vidhya](https://www.analyticsvidhya.com/blog/2017/02/40-questions-to-ask-a-data-scientist-on-ensemble-modeling-techniques-skilltest-solution/)
- [Condorcet 陪审团定理 — Wikipedia](https://en.wikipedia.org/wiki/Condorcet's_jury_theorem)
- [多数投票与 Condorcet 陪审团定理 (arXiv 2002.03153)](https://arxiv.org/abs/2002.03153)
- [集成情感分析中独立性的探究 — arXiv](https://arxiv.org/html/2409.0094)
- [多数投票分类器何时有益？— arXiv 1307.6522](https://arxiv.org/abs/1307.6522)
- [多类多数投票准确率新边界 — arXiv 2309.09564](https://arxiv.org/pdf/2309.09564)
- [独立分类器的多数投票可提升准确率 — Iowa State](https://dr.lib.iastate.edu/bitstreams/8fa1d6f7-9779-4bdd-9cb0-8987ee9f416b/download)
- [多数投票综述 — ScienceDirect Topics](https://www.sciencedirect.com/topics/computer-science/majority-voting)
- [通过多数投票组合分类器 — Draconian Fleet library](http://library.draconianfleet.com/epubfs.php?data=7626&comp=ch07s02.html)

## ⭐ 进阶变体 — 主题 2

- [AdaBoost 算法问答 — Sanfoundry](https://www.sanfoundry.com/machine-learning-questions-answers-adaboost-algorithm/)
- [AdaBoost 算法面试题 — Analytics Vidhya](https://www.analyticsvidhya.com/blog/2022/11/interview-questions-on-adaboost-algorithm-in-data-science/)
- [Boosting (Aarti Singh, CMU 10-701/15-781 PDF)](https://www.cs.cmu.edu/~aarti/Class/10701/slides/Lecture10.pdf)
- [The AdaBoost Algorithm (Sontag, MIT CSAIL)](https://people.csail.mit.edu/dsontag/courses/ml12/slides/lecture13.pdf)
- [深度解读 AdaBoost 算法 — ProjectPro](https://www.projectpro.io/article/adaboost-algorithm/972)
- [集成学习面试题 — Devinterview-io](https://github.com/Devinterview-io/ensemble-learning-interview-questions)
- [偏差-方差权衡 — Wikipedia](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)
- [单估计器 vs bagging：偏差-方差分解 — scikit-learn](https://scikit-learn.org/stable/auto_examples/ensemble/plot_bias_variance.html)
- [偏差/方差权衡与集成方法 (Vibhav Gogate, UTD PDF)](https://personal.utdallas.edu/~vibhav.gogate/ml/2020f/lectures/EnsembleMethods.pdf)
- [统一的偏差-方差分解 (Pedro Domingos, UW PDF)](https://homes.cs.washington.edu/~pedrod/papers/mlc00a.pdf)
- [偏差-方差分解、集成方法 (CSC2515 Toronto, PDF)](https://csc2515.github.io/csc2515-fall2024/lectures/lec04.pdf)
- [Bagging 与 Boosting 分类性能 — Springer](https://link.springer.com/article/10.1007/s00354-011-0303-0)
- [关于特征选择、偏差-方差与 Bagging (Cornell)](https://www.cs.cornell.edu/~mmunson/publications/docs/fs-bagging.pdf)
- [Hoeffding 不等式 — 样本均值集中 (Wikipedia)](https://en.wikipedia.org/wiki/Hoeffding%27s_inequality)
- [Borda Count — Wikipedia（多类投票）](https://en.wikipedia.org/wiki/Borda_count)
- [Stacked Generalization — Wolpert 1992 (原始论文)](https://www.machine-learning.martinsewell.com/ensembles/stacking/Wolpert1992.pdf)
- [动态加权多数投票的概率预测 — ResearchGate](https://www.researchgate.net/publication/260379631_Probabilistic_Predictions_of_Ensemble_of_Classifiers_Combined_with_Dynamically_Weighted_Majority_Vote)
- [基于交叉验证的概率分类器集成加权方案 — Springer DMKD](https://link.springer.com/article/10.1007/s10618-019-00638-y)
