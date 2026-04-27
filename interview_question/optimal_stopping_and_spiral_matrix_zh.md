# 最优停止 & 矩阵螺旋 (Optimal Stopping & Matrix Spiral) — 面试备战手册

针对两个即将到来的面试主题精心整理的学习指南，类似题目来源于真实面试题库（Jane Street、Meta、Google、Goldman Sachs、BlackRock）和 LeetCode。每道题都包含核心思路 (key insight) 和可直接运行的 Python 解答。

---

## 目录 (Table of Contents)

- [主题 1：最优停止（骰子）](#topic-1--optimal-stopping-dice)
  - [1.0 原题 (Original problem)](#10--original-problem)
  - [1.1 两次/三次掷骰变体 (Jane Street, Meta)](#11--two-roll--three-roll-variant-jane-street-meta)
  - [1.2 取最大值变体 (Meta)](#12--take-the-max-variant-meta)
  - [1.3 100 面骰带成本 (Jane Street)](#13--100-sided-die-with-cost-jane-street)
  - [1.4 连续 Uniform[0,1] 变体](#14--continuous-uniform01-variant)
  - [1.5 不放回抽牌 (Google, Goldman Sachs)](#15--card-draw-without-replacement-google-goldman-sachs)
  - [1.6 红黑赌徒问题 (Goldman Sachs)](#16--blackred-gambler-goldman-sachs)
  - [1.7 Pig — 累加爆牌骰子游戏](#17--pig--sum-with-bust-dice-game)
  - [1.8 秘书问题 (1/e 法则)](#18--secretary-problem-1e-rule)
  - [1.9 ⭐ 先知不等式 (Prophet inequality, 1/2 近似)](#19--prophet-inequality-12-approximation)
  - [1.10 ⭐ 卖房问题带折扣因子 (无限时域)](#110--house-selling-with-discount-factor-infinite-horizon)
  - [1.11 ⭐ 多臂老虎机 & Gittins 指数](#111--multi-armed-bandit--the-gittins-index)
  - [1.12 ⭐ 样本驱动的秘书问题 (分布未知)](#112--sample-driven-secretary-unknown-distribution)
  - [1.13 ⭐ Robbins 问题 (最小化期望排名)](#113--robbins-problem-minimize-expected-rank)
- [主题 2：矩阵螺旋](#topic-2--matrix-spiral)
  - [2.0 原题 (从中心开始的螺旋)](#20--original-problem-spiral-from-middle)
  - [2.1 螺旋矩阵 (LeetCode 54)](#21--spiral-matrix-leetcode-54)
  - [2.2 螺旋矩阵 II (LeetCode 59)](#22--spiral-matrix-ii-leetcode-59)
  - [2.3 螺旋矩阵 III (LeetCode 885)](#23--spiral-matrix-iii-leetcode-885)
  - [2.4 螺旋矩阵 IV (LeetCode 2326)](#24--spiral-matrix-iv-leetcode-2326)
  - [2.5 对角线遍历 (LeetCode 498)](#25--diagonal-traverse-leetcode-498)
  - [2.6 任意起点的螺旋 (GeeksforGeeks)](#26--spiral-from-arbitrary-point-geeksforgeeks)
  - [2.7 ⭐ 原地旋转图像 (LeetCode 48)](#27--rotate-image-in-place-leetcode-48)
  - [2.8 ⭐ 对角线遍历 II — 不规则二维数组 (LeetCode 1424)](#28--diagonal-traverse-ii--jagged-2d-array-leetcode-1424)
  - [2.9 ⭐ 带障碍物的螺旋 (BFS-on-spiral)](#29--spiral-with-obstacles-bfs-on-spiral)
  - [2.10 ⭐ 三维立方体螺旋 (逐层剥)](#210--3d-cube-spiral-layer-by-layer)
  - [2.11 ⭐ 逆螺旋 — 从螺旋输出反推矩阵](#211--inverse-spiral--reconstruct-from-spiral-output)
- [速查表 (Cheat Sheet)](#cheat-sheet)
- [参考资料 (Sources)](#sources)

---

# 主题 1 — 最优停止（骰子）(Topic 1 — Optimal Stopping (Dice))

> **通用模板** — 下面所有变体都使用同一个反向归纳 (backward induction) 递推：
>
> 1. 定义 `V[k]` = 还剩 `k` 次决策/掷骰时的最优期望值。
> 2. 计算 **continuation value** （继续游戏的期望值）。
> 3. 在每次随机结果出现后，当 `immediate_reward ≥ continuation_value` 时停止。
> 4. 对随机结果取期望。

---

## 1.0 — 原题 (Original problem)

> 投掷一颗公平的 6 面骰子最多 `n` 次。每次掷完后，可以选择停止保留当前值，或者再掷一次。最大化期望得分。

**递推 (Recurrence).** `E[1] = 3.5`；`E[k] = (1/6) · Σ_{v=1..6} max(v, E[k-1])`。

```python
def expected_optimal_dice(n: int) -> float:
    E = 3.5
    for _ in range(n - 1):
        E = sum(max(v, E) for v in range(1, 7)) / 6
    return E

# n=1 -> 3.500    n=2 -> 4.250    n=3 -> 4.667
# n=4 -> 4.944    n=5 -> 5.130    n→∞ -> 6
```

---

## 1.1 — 两次/三次掷骰变体 (Jane Street, Meta)

> **Jane Street：** "投掷一颗骰子。如果不满意，可以再投一次，但必须保留第二次的结果。这个游戏的公平价值是多少？" 然后：最多允许 2 次重投的版本。
>
> **Meta：** "最多投掷 3 次。你得到 `$x`，其中 `x` 是你最终停止时的值（如果继续，则视为最后一次的值）。"

这是原题 `n=2` 和 `n=3` 的特例。关键技巧是 **记住阈值 (threshold)**：

| 剩余次数 | 继续值 (Continuation value) | 阈值（≥ 时停止）|
|---|---|---|
| 1 (最后一次) | n/a | n/a — 必须保留 |
| 2 | 3.5 | 4 |
| 3 | 4.25 | 5 |
| 4 | 4.667 | 5 |

```python
def two_roll_value():
    # Roll once, may reroll once. Threshold = E[1] = 3.5, so stop on >=4.
    return sum(max(v, 3.5) for v in range(1, 7)) / 6   # 4.25

def three_roll_value():
    # Threshold for first roll is E[2] = 4.25, so stop on >=5.
    return sum(max(v, 4.25) for v in range(1, 7)) / 6  # 4.667

assert round(three_roll_value(), 3) == 4.667
```

> 🎯 **面试提示：** 面试官几乎总是希望你 *口头复述阈值规则*：*"3 次投掷情况下：第一次投掷 5 或 6 时停止；第二次投掷 4-6 时停止；第三次保留任何结果。"*

---

## 1.2 — 取最大值变体 (Take-the-max variant, Meta)

> "你最多投掷 3 次。你得到 `$x`，其中 `x` 是你掷出的 **最大** 值。可以随时停止。"

现在状态是 `(rolls_left, max_so_far)`，不仅仅是 `rolls_left` —— 因为一旦掷出 6，就没必要继续。所以阈值变成了 "如果 `max_so_far ≥ E[k-1, max_so_far]` 则停止" —— 由于最大值不会减小，递推得以简化。

```python
from functools import lru_cache

def expected_max_dice(n: int) -> float:
    @lru_cache(maxsize=None)
    def V(rolls_left, current_max):
        if rolls_left == 0:
            return current_max
        # Continuation: roll once, take new max, then optimal play with one fewer roll
        cont = sum(V(rolls_left - 1, max(current_max, v)) for v in range(1, 7)) / 6
        return max(current_max, cont)
    # Initial state: must take first roll, then decide
    return sum(V(n - 1, v) for v in range(1, 7)) / 6

# n=1 -> 3.500    n=2 -> 4.667    n=3 -> 4.958 (pre-roll value)
```

3 次投掷的答案约为 **4.958**，比 4.667 略高，因为保留最大值相当于免费获得"安全网"。

---

## 1.3 — 100 面骰带成本 (Jane Street)

> "100 面骰子。投掷后，要么取走那么多美元，要么 **支付 $1** 再投一次。最优策略和期望值 (EV) 是多少？"

**思路 (Insight)：** 继续值减少 1（成本）。阈值：当 `roll ≥ E_continue + 1` 时停止。

```python
def hundred_sided_with_cost(faces: int = 100, cost: float = 1.0,
                             max_iter: int = 10_000, tol: float = 1e-9) -> float:
    # Infinite-horizon: V satisfies V = (1/faces) * Σ max(v, V - cost).
    # Iterate to fixed point.
    V = faces / 2
    for _ in range(max_iter):
        new_V = sum(max(v, V - cost) for v in range(1, faces + 1)) / faces
        if abs(new_V - V) < tol:
            return new_V
        V = new_V
    return V

ev = hundred_sided_with_cost()
# ev ≈ 87.357. Threshold: stop on rolls >= 88.
```

最优阈值是 `ceil(ev + 1)`：当看到 ≥ 88 时停止。

---

## 1.4 — 连续 Uniform[0,1] 变体 (Continuous Uniform[0,1] variant)

> 同样的游戏，但每次抽取都是独立同分布的 (i.i.d.) `Uniform[0,1]`。最多 `n` 次抽取。

**闭式递推 (Closed-form recurrence).** 如果阈值是 `t = E[k-1]`：

`E[k] = t · P(X<t) + E[X | X≥t] · P(X≥t) = t·t + (1+t)/2 · (1-t) = (1 + t²) / 2`。

```python
def expected_uniform_stop(n: int) -> float:
    E = 0.5
    for _ in range(n - 1):
        E = (1 + E * E) / 2
    return E

# n=1 -> 0.500   n=2 -> 0.625   n=3 -> 0.695   n=10 -> 0.861   n→∞ -> 1
```

quant 面试官最爱的 follow-up，因为它展示了你能把骰子求和换成积分。

---

## 1.5 — 不放回抽牌 (Google, Goldman Sachs)

> "一副牌。每次不放回地抽一张。得分 = 牌面值。可随时停止。最大化 EV？"

状态是 **剩余牌的 multiset**，但由对称性 (symmetry) 只有 order statistics 重要。

```python
from functools import lru_cache

def expected_card_draw_distinct(m: int) -> float:
    """Cards 1..m, each appearing once."""
    @lru_cache(maxsize=None)
    def V(remaining: tuple) -> float:
        if len(remaining) == 1:
            return remaining[0]
        total = 0.0
        for i, v in enumerate(remaining):
            cont = V(remaining[:i] + remaining[i + 1 :])
            total += max(v, cont)
        return total / len(remaining)
    return V(tuple(range(1, m + 1)))

# m=6 (like dice without replacement) -> 4.667
```

---

## 1.6 — 红黑赌徒问题 (Black/Red gambler, Goldman Sachs, Google)

> "26 张红 + 26 张黑。每次抽一张。红 = +$1，黑 = −$1。可随时停止。最优 EV？"

**状态：** `(red_left, black_left)`。**思路：** 如果到目前为止抽到的黑牌多于红牌，且牌堆"足够平衡"，就 *不应* 继续 —— 而且你总是可以等到最后（牌堆净值为 0）。"继续"的最优值就是等待的期望值；只有当当前盈利超过它时才兑现。

```python
from functools import lru_cache

def red_black_game(R: int = 26, B: int = 26) -> float:
    @lru_cache(maxsize=None)
    def V(r, b, profit):
        # If no cards left, profit is forced.
        if r == 0 and b == 0:
            return profit
        # Continue value: average over next card
        total = 0
        if r > 0:
            total += r * V(r - 1, b, profit + 1)
        if b > 0:
            total += b * V(r, b - 1, profit - 1)
        cont = total / (r + b)
        # You may stop and lock in 'profit' (only meaningful when profit > cont).
        return max(profit, cont)
    return V(R, B, 0)

print(red_black_game(26, 26))   # ≈ 2.624
print(red_black_game(4, 4))     # ≈ 0.714
```

简单边界：负盈利时绝不停止（等到牌堆抽空时至少能拿到 0）。最优阈值：只在净正且继续 EV 低于当前盈利时提前停止。

---

## 1.7 — Pig — 累加爆牌骰子游戏 (Sum-with-bust dice game)

> "每回合：投掷骰子，把点数累加到回合总分。掷出 1 → 整个回合总分清零，回合结束。否则可随时停止保留回合总分。最优持有阈值是多少？"

**单回合分析**（贪心版本）：当回合总分 `s` 使得继续投掷的边际期望值变为负时，就停手。

`expected_change_per_roll = (1/6)(−s) + (1/6)(2 + 3 + 4 + 5 + 6) = (20 − s)/6`。

`s < 20` 时继续投。所以单独看 **持有 20** 是贪心最优。（实际两人玩到 100 分的 Pig 中，最优阈值会因得分差异下移到 ~21–25。）

```python
def pig_greedy_threshold():
    # Continue while expected delta is positive: (20 - s)/6 > 0  =>  s < 20.
    return 20

def pig_simulate(threshold: int, trials: int = 1_000_000) -> float:
    import random
    total = 0
    for _ in range(trials):
        s = 0
        while s < threshold:
            r = random.randint(1, 6)
            if r == 1:
                s = 0; break
            s += r
        total += s
    return total / trials

# pig_simulate(20) ≈ 8.21 expected points per turn
```

---

## 1.8 — 秘书问题 (Secretary problem, 1/e 法则)

> "随机顺序面试 `n` 个候选人，逐一决定接受还是拒绝。最大化选中 **唯一最佳** 候选人的概率。"

**策略：** 拒绝前 `n/e` 个候选人，然后接受第一个比已见过的所有人都好的候选人。`P(success) → 1/e ≈ 0.368`。

```python
import math, random

def secretary_optimal_cutoff(n: int) -> int:
    return max(1, int(n / math.e))

def secretary_simulation(n: int, trials: int = 200_000) -> float:
    cutoff = secretary_optimal_cutoff(n)
    wins = 0
    for _ in range(trials):
        perm = list(range(1, n + 1))
        random.shuffle(perm)
        best_in_sample = max(perm[:cutoff]) if cutoff else 0
        picked = perm[-1]   # forced last if no candidate beats sample
        for v in perm[cutoff:]:
            if v > best_in_sample:
                picked = v; break
        wins += (picked == n)
    return wins / trials

# secretary_simulation(100) ≈ 0.371
```

---

# ⭐ 主题 1 — 进阶变体 (Harder Variants)

> 从"计算阈值"升级到 **竞争分析 (competitive analysis)**（先知不等式）、**无限时域折扣 MDP**（卖房问题）、**多项目分配**（Gittins）和 **distribution-free** 停止规则。

---

## 1.9 — 先知不等式 (Prophet inequality, 1/2 近似)

> 独立随机变量序列 `X₁, ..., Xₙ` 来自 **已知但不同的分布**，逐个观察。每次观察后接受或拒绝。一个能看到所有抽样的"先知"得到 `E[max Xᵢ]`。找一个 online 规则，对任何分布序列都至少能取到先知值的 **一半**。

**Krengel–Sucheston 定理 (1977)：** 一个简单的 **单阈值规则** 即可：选阈值 `τ` 满足 `Σ_i E[(Xᵢ − τ)⁺] = τ`（或等价地 `P(max Xᵢ ≥ τ) = 1/2`）。在第一个 `Xᵢ ≥ τ` 时停止。

`E[ALG] ≥ 0.5 · E[max Xᵢ]`，常数 `1/2` 是 **紧的 (tight)**。

```python
def prophet_threshold(samples_per_dist, n_trials=200_000):
    """
    samples_per_dist: list of length n; samples_per_dist[i] is a function
    that returns a fresh sample from the i-th distribution.
    Numerically estimates the median of max X_i and uses it as threshold.
    """
    import statistics, random
    n = len(samples_per_dist)
    maxes = [max(s() for s in samples_per_dist) for _ in range(20_000)]
    tau = statistics.median(maxes)                  # P(max >= tau) = 1/2

    alg, prophet = 0.0, 0.0
    for _ in range(n_trials):
        draws = [s() for s in samples_per_dist]
        prophet += max(draws)
        for x in draws:
            if x >= tau:
                alg += x; break
        else:
            alg += draws[-1]                        # forced
    return alg / n_trials, prophet / n_trials, tau

# Example with mix of exponential and uniform distributions:
# import random
# samplers = [lambda: random.expovariate(1)] * 5 + [lambda: random.uniform(0, 5)] * 5
# alg, prophet, tau = prophet_threshold(samplers)
# alg/prophet should be >= 0.5
```

> 🎯 **重要性：** 这是经典的 online vs offline 竞争比 (competitive ratio) 结果；它是算法机制设计 (algorithmic mechanism design) 中 posted-price 机制的基础（Hajiaghayi, Kleinberg）。

---

## 1.10 — 卖房问题带折扣因子 (House-selling with discount factor, 无限时域)

> "每个时期到来一个独立同分布的 offer `Xᵢ ~ F`。接受 offer 拿到 `Xᵢ`，或拒绝等下一期。未来奖励折扣因子为 `β ∈ (0,1)`。最优停止规则和 `V` 是什么？"

**Bellman 方程 (Bellman equation).** `V = E[max(X, β·V)]`。最优规则是 **常数阈值** `c = β·V`：当 `X ≥ c` 时接受。

`V = β V · F(c) + ∫_{c}^{∞} x dF(x)`，结合 `c = βV` 得 `c = β·E[max(X, c)]`。

```python
def house_selling_threshold(samples, beta, max_iter=10_000, tol=1e-9):
    """Numerically solve fixed point V = E[max(X, beta*V)] from samples of F."""
    V = sum(samples) / len(samples)                 # warm start
    for _ in range(max_iter):
        new_V = sum(max(x, beta * V) for x in samples) / len(samples)
        if abs(new_V - V) < tol: break
        V = new_V
    threshold = beta * V
    return V, threshold

# Uniform[0,1] offers, β=0.9
import random
samples = [random.random() for _ in range(100_000)]
V, c = house_selling_threshold(samples, beta=0.9)
# V ≈ 0.755, threshold c ≈ 0.679. Accept first offer >= 0.679.
```

**Uniform[0,1] 的闭式解：** `c` 满足 `c = β·(1 + c²)/2 ⇒ c = (1 − √(1−β²))/β`。

---

## 1.11 — 多臂老虎机 & Gittins 指数 (Multi-armed bandit & Gittins index)

> "K 个独立的'项目'。每步选一个推进，它根据其（Markov）状态返回随机奖励。折扣因子 β。最大化期望折扣总奖励。"

**Gittins (1979)：** 最优策略是 **指数策略 (index policy)**：给每个项目分配一个数 `g(state)`（"Gittins 指数" —— 单独玩该项目时折扣奖励率在所有停止规则 `τ` 上的最大值），始终选择当前指数最大的项目。

`g(state) = max_τ  E[Σ_{t=0}^{τ-1} βᵗ Rₜ] / E[Σ_{t=0}^{τ-1} βᵗ]`

对于 **Bernoulli bandit + Beta(α,β) 先验**，可以通过对 (α,β) 做 value iteration 计算指数：

```python
from functools import lru_cache

def gittins_bernoulli(alpha, beta, gamma=0.9, horizon=200):
    """
    Value iteration to compute Gittins index for Beta(alpha, beta) Bernoulli arm.
    Solves: g = sup_τ  E[discounted reward from arm] / E[discounted plays]
    via the 'restart' formulation.
    """
    # Index = smallest g such that the value of the calibration MDP
    # (compete arm against constant-reward g arm) prefers g.
    # We'll binary-search on g.
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        @lru_cache(maxsize=None)
        def V(a, b, t):
            if t == horizon: return 0.0
            p = a / (a + b)
            play_arm = p * (1 + gamma * V(a + 1, b, t + 1)) + \
                       (1 - p) * (0 + gamma * V(a, b + 1, t + 1))
            stop = mid / (1 - gamma)                 # value of taking constant g forever
            return max(play_arm, stop)
        V.cache_clear()
        if V(alpha, beta, 0) > mid / (1 - gamma) + 1e-9:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

# gittins_bernoulli(1, 1, gamma=0.9) ≈ 0.78
# Beta(1,1) is uniform prior; Gittins > 0.5 because of the value of exploration.
```

> 🎯 **重要性：** Restless / contextual bandit 变体（如能量感知调度的 Whittle 指数）是 NP-完全的，但 index policy 是强启发式，Gittins 给出标准 MAB 的精确答案。

---

## 1.12 — 样本驱动的秘书问题 (Sample-driven secretary, 分布未知)

> 同秘书问题，但在 online 阶段开始 **之前**，会先收到来自同一分布的 `s` 个样本。当 `n = ∞` 时，存在一个漂亮的 `1−1/e` 边界。

**结果 (Correa et al., 2020)：** 给定 `O(n)` 个样本，可达到最优 i.i.d. prophet inequality 边界 `1 − 1/e ≈ 0.632`。

实用启发式：使用样本的 `(1−1/e)`-quantile 作为阈值。

```python
def sample_driven_secretary(samples, online_stream):
    """samples: pre-arrival batch (list of values).
       online_stream: generator of online values."""
    samples = sorted(samples)
    k = max(1, int(len(samples) * (1 - 1 / 2.71828)))
    threshold = samples[k - 1]
    for x in online_stream:
        if x >= threshold:
            return x
    return None                                       # walk away

# Empirically achieves close to (1 - 1/e) · OPT for many distributions.
```

> 🎯 **与经典 1.8 的区别：** 经典秘书问题 *没有任何分布信息*，得到 1/e。增加历史样本后提升到 1−1/e。

---

## 1.13 — Robbins 问题 (最小化期望排名)

> "Online：`n` 个 i.i.d. 均匀值。必须选一个。最小化所选值的 **期望排名** (rank 1 = 最佳，rank n = 最差)。"

**未解决问题** (Robbins 自 1990 年起)：极限期望排名 `V_∞ = lim_{n→∞} E[rank]` 已知在 `[1.908, 2.327]` 范围内，但精确值 *未知*。这是最优停止理论中最著名的开放问题。

一个合理的启发式根据剩余时间和当前排名设置阈值：

```python
def robbins_heuristic(values):
    """Greedy rank-aware heuristic: take if current value is best-so-far AND
    enough remaining trials are unlikely to beat it."""
    n = len(values)
    best, best_idx = values[0], 0
    for i, v in enumerate(values):
        if v <= best:
            best, best_idx = v, i
        # Heuristic: stop if we are "deep" enough into the stream and v is the running min
        if i >= n // 2 and v == best:
            return v
    return best                                       # forced last min
```

> 🎯 **面试加分：** 引用此问题表明你知道最优停止理论存在开放问题；Jane Street / Citadel 等量化团队的招聘者 *很欣赏* 未解决领域的话题。

---

# 主题 2 — 矩阵螺旋 (Topic 2 — Matrix Spiral)

> **两个通用模式 (universal patterns)** 覆盖几乎所有螺旋题目：
>
> - **模式 A — 收缩边界 (Shrinking boundaries)：** 维护 `top, bottom, left, right`；走完一边后收缩。
> - **模式 B — 扩张步长 `1,1,2,2,3,3,…`：** 沿 `R, D, L, U` 方向循环；每个步长用 *两次*。跳过越界格子。
>
> 由外向内的螺旋 → 模式 A。由内向外 / 任意起点的螺旋 → 模式 B。

---

## 2.0 — 原题 (从中心开始的螺旋)

> 给定 `n × n` 矩阵，从 **中心** 开始按螺旋顺序输出元素。

模式 B：从中心出发，步长模式为 `1, 1, 2, 2, 3, 3, …`，方向循环 `R → D → L → U`。

```python
def spiral_from_middle(matrix):
    n = len(matrix)
    if n == 0: return []
    cx, cy = n // 2, n // 2          # center (works cleanly for odd n)
    result = [matrix[cx][cy]]
    dx = [0, 1, 0, -1]               # R, D, L, U
    dy = [1, 0, -1, 0]
    x, y, step, d = cx, cy, 1, 0
    total = n * n
    while len(result) < total:
        for _ in range(2):           # each step length used twice
            for _ in range(step):
                x += dx[d]; y += dy[d]
                if 0 <= x < n and 0 <= y < n:
                    result.append(matrix[x][y])
                    if len(result) == total:
                        return result
            d = (d + 1) % 4
        step += 1
    return result

# 3x3 of [[1,2,3],[4,5,6],[7,8,9]]
# -> [5, 6, 9, 8, 7, 4, 1, 2, 3]
```

> ⚠️ **偶数 `n` 的 follow-up：** "中心"是有歧义的；常见约定：选 `(n//2 - 1, n//2 - 1)`（中央 2×2 的左上角）或换不同的初始方向。最好和面试官明确。

---

## 2.1 — 螺旋矩阵 (Spiral Matrix, LeetCode 54)

> 给定 `m × n` 矩阵，返回 **从左上角** 开始的顺时针螺旋顺序的所有元素。

模式 A。

```python
def spiral_order(matrix):
    if not matrix: return []
    res = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    while top <= bottom and left <= right:
        for j in range(left, right + 1):  res.append(matrix[top][j])
        top += 1
        for i in range(top, bottom + 1):  res.append(matrix[i][right])
        right -= 1
        if top <= bottom:
            for j in range(right, left - 1, -1): res.append(matrix[bottom][j])
            bottom -= 1
        if left <= right:
            for i in range(bottom, top - 1, -1): res.append(matrix[i][left])
            left += 1
    return res
```

**与 2.0 的对偶检查：** `spiral_from_middle`（3×3 中心起点）的 *逆序* 是 `[3, 2, 1, 4, 7, 8, 9, 6, 5]`，等价于 3×3 在适当方向下从角落起点的螺旋。它们是对偶的。

---

## 2.2 — 螺旋矩阵 II (LeetCode 59)

> 给定 `n`，生成 `n × n` 矩阵，按螺旋顺序填入 `1..n²`。

模式 A，但是写而非读。

```python
def generate_spiral(n: int):
    M = [[0] * n for _ in range(n)]
    top, bottom, left, right = 0, n - 1, 0, n - 1
    v = 1
    while top <= bottom and left <= right:
        for j in range(left, right + 1):           M[top][j]    = v; v += 1
        top += 1
        for i in range(top, bottom + 1):           M[i][right]  = v; v += 1
        right -= 1
        if top <= bottom:
            for j in range(right, left - 1, -1):   M[bottom][j] = v; v += 1
            bottom -= 1
        if left <= right:
            for i in range(bottom, top - 1, -1):   M[i][left]   = v; v += 1
            left += 1
    return M

# n=3 -> [[1,2,3],
#         [8,9,4],
#         [7,6,5]]
```

---

## 2.3 — 螺旋矩阵 III (LeetCode 885)

> 在 `R × C` 网格中从 `(rStart, cStart)` 顺时针游走；按访问顺序返回单元格坐标。

这是 **原题最近的亲戚** —— 只是泛化到矩形网格和任意起点。

```python
def spiral_matrix_iii(R: int, C: int, rStart: int, cStart: int):
    res = [[rStart, cStart]]
    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]
    x, y, step, d = rStart, cStart, 1, 0
    total = R * C
    while len(res) < total:
        for _ in range(2):
            for _ in range(step):
                x += dx[d]; y += dy[d]
                if 0 <= x < R and 0 <= y < C:
                    res.append([x, y])
                    if len(res) == total:
                        return res
            d = (d + 1) % 4
        step += 1
    return res
```

如果你能解 LC 885，原题就是一行：调用 `spiral_matrix_iii(n, n, n//2, n//2)` 然后查每个 `(r, c)` 对应的 `matrix[r][c]`。

---

## 2.4 — 螺旋矩阵 IV (LeetCode 2326)

> 给定 `m`、`n` 和链表头 `head`，按螺旋顺序填入 `m × n` 矩阵。剩余格子用 `−1` 填充。

模式 A，但从流式输入写入。

```python
class ListNode:
    def __init__(self, val=0, nxt=None): self.val, self.next = val, nxt

def spiral_matrix_iv(m, n, head):
    M = [[-1] * n for _ in range(m)]
    top, bottom, left, right = 0, m - 1, 0, n - 1
    while head and top <= bottom and left <= right:
        for j in range(left, right + 1):
            if not head: return M
            M[top][j] = head.val; head = head.next
        top += 1
        for i in range(top, bottom + 1):
            if not head: return M
            M[i][right] = head.val; head = head.next
        right -= 1
        if top <= bottom:
            for j in range(right, left - 1, -1):
                if not head: return M
                M[bottom][j] = head.val; head = head.next
            bottom -= 1
        if left <= right:
            for i in range(bottom, top - 1, -1):
                if not head: return M
                M[i][left] = head.val; head = head.next
            left += 1
    return M
```

---

## 2.5 — 对角线遍历 (Diagonal Traverse, LeetCode 498)

> 按 zig-zag 对角线顺序（右上 ↔ 左下交替）遍历 `m × n` 矩阵。

不同的遍历模式但相同的"矩阵游走 + 方向"心智模型。

```python
def find_diagonal_order(matrix):
    if not matrix or not matrix[0]: return []
    m, n = len(matrix), len(matrix[0])
    res = []
    for d in range(m + n - 1):
        i_start, i_end = max(0, d - n + 1), min(m - 1, d)
        diag = [matrix[i][d - i] for i in range(i_start, i_end + 1)]
        res.extend(reversed(diag) if d % 2 == 0 else diag)
    return res

# [[1,2,3],[4,5,6],[7,8,9]] -> [1,2,4,7,5,3,6,8,9]
```

---

## 2.6 — 任意起点的螺旋 (GeeksforGeeks)

> 给定 `m × n` 矩阵和一个 *矩阵内部* 的点 `P(c, r)`，从 `P` 开始按顺时针螺旋打印矩阵。

类似 LC 885，但起点限定在内部，只访问网格内单元。复用 `spiral_matrix_iii`：

```python
def print_spiral_from_point(matrix, r, c):
    R, C = len(matrix), len(matrix[0])
    return [matrix[i][j] for i, j in spiral_matrix_iii(R, C, r, c)]
```

---

# ⭐ 主题 2 — 进阶变体 (Harder Variants)

> 从"游走并追加"升级到 **原地变换 (in-place transformations)**（LC 48）、**不规则结构**（LC 1424）、**螺旋内的搜索**（BFS）、**高维遍历**（3D）和 **逆问题**（从螺旋反推）。

---

## 2.7 — 原地旋转图像 (Rotate Image in place, LeetCode 48)

> 把 `n × n` 矩阵 **顺时针旋转 90°**，要求 **原地** (in place)（不分配额外矩阵）。`O(1)` 额外空间。

**两步技巧 (Two-step trick)：** 先转置 (transpose)，再翻转每一行。或者层层 4 元交换：

```python
def rotate(matrix):
    n = len(matrix)
    # 1) transpose
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    # 2) reverse each row
    for row in matrix:
        row.reverse()

# Alternative: layer-by-layer 4-cycle swap (true in-place spiral rotation)
def rotate_layered(matrix):
    n = len(matrix)
    for layer in range(n // 2):
        first, last = layer, n - 1 - layer
        for i in range(first, last):
            offset = i - first
            top                = matrix[first][i]
            matrix[first][i]   = matrix[last - offset][first]
            matrix[last - offset][first] = matrix[last][last - offset]
            matrix[last][last - offset]  = matrix[i][last]
            matrix[i][last]    = top
```

> 🎯 **难点：** 4 元交换是经典的 "我正确遍历了所有四层" 练习。约 80% 候选人栽在 off-by-one 错误上。

**实战中的变体：**
- 180° / 270°：旋转两次/三次，或转置 + 翻转列。
- 逆时针：先翻转行，再转置。

---

## 2.8 — 对角线遍历 II — 不规则二维数组 (LeetCode 1424)

> 类似 LC 498 但输入是一个 list of lists，**每行长度不同**。返回所有元素的反对角线顺序，每条对角线从右上到左下。

朴素的 `for d in range(m+n-1)` 不再适用，因为每行 `n` 不同。用 **按对角线索引分桶** 的方法：

```python
from collections import defaultdict

def find_diagonal_order_ii(nums):
    diag = defaultdict(list)
    for i, row in enumerate(nums):
        for j, v in enumerate(row):
            diag[i + j].append(v)              # all (i,j) with same i+j on one diag
    res = []
    for d in sorted(diag):
        res.extend(reversed(diag[d]))           # bottom-up reversal gives the right order
    return res

# [[1,2,3],[4,5,6],[7,8,9]]      -> [1,4,2,7,5,3,8,6,9]
# [[1,2,3,4,5],[6,7],[8],[9,10,11],[12,13,14,15,16]]
#                                -> [1,6,2,8,7,3,9,12,10,4,13,11,5,14,15,16]
```

**复杂度：** `O(N)`（`N = 总元素数`）—— 分桶每元素 O(1)，按 `d` 键排序为 `O(diag · log diag)`。

不错的替代方案：从 `(0,0)` 做 **BFS** 探索 `(i+1, j)` 和 `(i, j+1)` —— 但要小心去重。

---

## 2.9 — 带障碍物的螺旋 (Spiral with obstacles, BFS-on-spiral)

> 给定 `m × n` 网格，部分单元被阻塞 (`'#'`)，从左上角开始按顺时针螺旋顺序输出值，**跳过** 被阻塞的单元。

两种合理解读：
1. **跳过但保持螺旋路径：** 走标准螺旋；遇到阻塞单元就不输出。LC 54 的简单修改。
2. **绕过障碍物重路由：** 把螺旋视为 *方向优先级*（按螺旋顺序尝试 R, D, L, U）；遇阻或越界时转向。更难，但更有意思。

```python
def spiral_with_obstacles(grid):
    """Interpretation #2: walk in spiral direction, but reroute around blocked cells."""
    if not grid or not grid[0]: return []
    m, n = len(grid), len(grid[0])
    seen = [[False] * n for _ in range(m)]
    dx = [0, 1, 0, -1]                          # R, D, L, U
    dy = [1, 0, -1, 0]
    res = []
    i, j, d = 0, 0, 0
    if grid[0][0] != '#':
        res.append(grid[0][0])
    seen[0][0] = True
    for _ in range(m * n - 1):
        for turn in range(4):
            ni, nj = i + dx[(d + turn) % 4], j + dy[(d + turn) % 4]
            if 0 <= ni < m and 0 <= nj < n and not seen[ni][nj]:
                i, j = ni, nj
                d = (d + turn) % 4
                seen[i][j] = True
                if grid[i][j] != '#':
                    res.append(grid[i][j])
                break
        else:
            break                               # all neighbours seen
    return res
```

> 🎯 **现实场景：** 此模式出现在 **机器人路径规划**（带障碍物的扫地机覆盖）和 **视觉**（有效像素的 zigzag 扫描）中。

---

## 2.10 — 三维立方体螺旋 (3D cube spiral, 逐层剥)

> 给定 `n × n × n` 整数立方体，按"洋葱皮 (onion-peel)"螺旋顺序输出元素：先整个外壳（按某种规范的逐面顺序），再下一内壳，依次类推。

**思路：** 对每层 `s = 0, ..., n//2`，遍历 6 个面。每个面跑 2D 螺旋。总元素数 = `n³`。

```python
def shell_count_3d(n, s):
    """Number of cells on the s-th onion-peel shell of an n^3 cube."""
    if 2 * s + 1 == n: return 1                 # innermost single cell when n is odd
    inner = n - 2 * s
    return inner ** 3 - (inner - 2) ** 3 if inner > 2 else inner ** 3

def cube_spiral(cube):
    n = len(cube)
    res = []
    for s in range(n // 2 + 1):
        lo, hi = s, n - 1 - s
        if lo > hi: break
        if lo == hi:
            res.append(cube[lo][lo][lo]); break
        # Six faces of the shell, traversed in fixed order; we de-dup edges via a 'seen' set.
        seen = set()
        def emit(i, j, k):
            if (i, j, k) not in seen:
                seen.add((i, j, k))
                res.append(cube[i][j][k])

        # Top face (k = lo), in spiral
        for j in range(lo, hi + 1):    emit(lo, j, lo)
        for i in range(lo + 1, hi + 1): emit(i, hi, lo)
        for j in range(hi - 1, lo - 1, -1): emit(hi, j, lo)
        for i in range(hi - 1, lo, -1): emit(i, lo, lo)
        # Sides (lo < k < hi)
        for k in range(lo + 1, hi):
            for j in range(lo, hi + 1):    emit(lo, j, k)
            for i in range(lo + 1, hi + 1): emit(i, hi, k)
            for j in range(hi - 1, lo - 1, -1): emit(hi, j, k)
            for i in range(hi - 1, lo, -1): emit(i, lo, k)
        # Bottom face (k = hi)
        for j in range(lo, hi + 1):    emit(lo, j, hi)
        for i in range(lo + 1, hi + 1): emit(i, hi, hi)
        for j in range(hi - 1, lo - 1, -1): emit(hi, j, hi)
        for i in range(hi - 1, lo, -1): emit(i, lo, hi)
    return res
```

> 🎯 **面试价值：** 测试你能否把 2D 心智模型扩展到 3D *而不被索引淹没*。去重 set 是保持理智的关键技巧。

---

## 2.11 — 逆螺旋 — 从螺旋输出反推矩阵

> 给定长度为 `m·n` 的一维数组 `arr` 和维度 `m, n`，**反推** 矩阵，使按顺时针螺旋（LC 54）顺序读取它得到 `arr`。

这是 LC 54 的逆问题 → 准确说就是 LC 59 的泛化：不是写入 `1..n²`，而是按螺旋顺序写入 `arr` 中的值。

```python
def inverse_spiral(arr, m, n):
    assert len(arr) == m * n
    M = [[0] * n for _ in range(m)]
    top, bottom, left, right = 0, m - 1, 0, n - 1
    idx = 0
    while top <= bottom and left <= right:
        for j in range(left, right + 1):           M[top][j]    = arr[idx]; idx += 1
        top += 1
        for i in range(top, bottom + 1):           M[i][right]  = arr[idx]; idx += 1
        right -= 1
        if top <= bottom:
            for j in range(right, left - 1, -1):   M[bottom][j] = arr[idx]; idx += 1
            bottom -= 1
        if left <= right:
            for i in range(bottom, top - 1, -1):   M[i][left]   = arr[idx]; idx += 1
            left += 1
    return M

# inverse_spiral([1,2,3,4,5,6,7,8,9], 3, 3)
# -> [[1,2,3],[8,9,4],[7,6,5]]    (same as LC 59 generate_spiral(3))
```

**变形（Bloomberg 真题）：** 给定 `arr = sorted([...])`，结果矩阵长什么样？最小元素在 **左上角**，最大元素在 **几何中心**（蛇形洋葱模式），适合可视化。

---

# 速查表 (Cheat Sheet)

## 最优停止 — 模式识别

| 看到... | 用... |
|---|---|
| 随机奖励 + 有限轮次 + 不可逆停止 | 对剩余轮次做反向归纳 |
| 每轮成本 | 阈值 = `E_continue − cost` |
| 状态依赖于历史（max, sum, deck） | 在 `(round, state)` 上 memoize |
| 无限轮次 + 固定成本 | Bellman 方程的 fixed-point 迭代 |
| "选中最佳的概率"（不是 EV） | 秘书问题 / 1/e 法则 |
| 爆牌风险 (Pig) | 比较期望增量 `(non_bust_avg − bust_prob × current)` 与 0 |
| Online vs offline 竞争比 | 先知不等式（单阈值 1/2） |
| 折扣因子 + i.i.d. offers | 卖房问题：`c = β·E[max(X, c)]` |
| 多个项目，每次只能推进一个 | Gittins 指数策略 |
| 没有分布信息但有样本 | 样本驱动秘书问题，阈值 = `(1−1/e)`-quantile |

## 矩阵螺旋 — 模式识别

| 螺旋类型 | 模式 | 模板 |
|---|---|---|
| 由外向内 (LC 54, 59, 2326) | A — 收缩边界 | 4 嵌套循环，递减 `top/right/bottom/left` |
| 由内向外 / 任意起点 (LC 885, 原题) | B — 扩张步长 `1,1,2,2,3,3,…` | 方向数组 + `for _ in range(2)` |
| 对角线 (LC 498) | 按 `i+j` 分组，交替反转 | `for d in range(m+n-1)` |
| 不规则对角线 (LC 1424) | 按 `i+j` 分桶到 dict | `defaultdict(list)`，再排序键 |
| 原地旋转 (LC 48) | 转置 + 行翻转，**或** 逐层 4 元交换 | 两次遍历 vs `n//2` 层 |
| 带障碍物螺旋 | 方向优先级，遇阻转向 | "按螺旋循环顺序尝试 R, D, L, U" |
| 3D 螺旋 | Onion-peel；每壳 = 6 面 | 用 `seen` set 去重壳棱 |

> **写代码前总要澄清：** 矩形 vs 方阵、偶 vs 奇维度、起点位置、越界处理、返回值（值 vs 坐标）。

---

# 参考资料 (Sources)

## 主题 1 — 最优停止

- [Jane Street 面试题 (Glassdoor) — 重投骰子问题](https://www.glassdoor.com/Interview/a-expected-value-of-a-die-b-suppose-you-play-a-game-where-you-get-a-dollar-amount-equivalent-to-the-number-of-dots-that-QTN_30411.htm)
- [Jane Street 面试题 (Glassdoor) — 100 面骰带 $1 重投成本](https://www.glassdoor.com/Interview/You-are-given-a-die-with-100-sides-One-side-has-1-dot-one-has-2-dots-and-so-on-up-until-100-You-are-given-a-chance-to-ro-QTN_688189.htm)
- [Meta 面试题 (Glassdoor) — 最多 3 次掷骰取最高](https://www.glassdoor.com/Interview/You-can-roll-a-dice-3-times-You-will-be-given-x-where-x-is-the-highest-roll-you-get-You-can-choose-to-stop-rolling-at-an-QTN_802648.htm)
- [Goldman Sachs 面试题 (Glassdoor) — 4 张牌 2 黑 2 红](https://www.glassdoor.com/Interview/You-have-4-cards-2-black-and-2-red-You-play-a-game-where-during-each-round-you-draw-a-card-If-it-s-black-you-lose-a-poi-QTN_257709.htm)
- [Google 面试题 (Glassdoor) — 52 张牌红黑最优停止](https://www.glassdoor.sg/Interview/You-have-52-playing-cards-26-red-26-black-You-draw-cards-one-by-one-A-red-card-pays-you-a-dollar-A-black-one-fines-yo-QTN_3421153.htm)
- [Jane Street 面试题 (Glassdoor) — 26 红 26 黑猜颜色](https://www.glassdoor.com/Interview/3-Poker-26-red-26-black-Take-one-every-time-you-can-choose-to-guess-whether-it-s-red-You-have-only-one-chance-If-you-QTN_155340.htm)
- [BlackRock 电话面试 — 掷骰子 (QuantNet)](https://quantnet.com/threads/blackrock-phone-interview-dice-roll.13712/)
- [ML Interview Q Series — 骰子游戏策略：最优停止](https://www.rohan-paul.com/p/ml-interview-q-series-dice-game-strategy)
- [Quant 面试题答案 — 反向归纳 walk-through](https://quantinvestor.wordpress.com/2009/10/20/solution-for-quant-interview-question/)
- [骰子游戏最优停止策略 (Wendy Hu, Medium)](https://medium.com/@whystudying/dice-game-optimized-stopping-strategy-59faa0862d8e)
- [Rolling the Dice：基于概率的最大化策略 (Gaurav Kandel, Medium)](https://medium.com/@dswithgk/rolling-the-dice-a-probability-based-strategy-for-maximum-gain-a9aa80ed86d0)
- [掷骰子游戏的最优值 (Pascal Bercker, Medium)](https://medium.com/@pbercker/the-optimal-value-for-a-game-of-dice-or-knowing-when-to-quit-29c69ac01a0e)
- [掷骰子游戏的期望收益 — Predictive Hacks](https://predictivehacks.com/the-expected-payoff-of-a-dice-game/)
- [d20 停止难题 (DataGenetics)](http://datagenetics.com/blog/february32016/index.html)
- [红/黑赌博游戏 (DataGenetics)](http://datagenetics.com/blog/october42014/index.html)
- [一道最优停止 quant 谜题 — Emir's blog](https://emiruz.com/post/2023-07-30-optimal-stopping/)
- [骰子问题集 (PDF)](https://www.madandmoonly.com/doctormatt/mathematics/dice1older.pdf)
- ["Pig (Pig-out)" 分析 — Durango Bill's](http://www.durangobill.com/Pig.html)
- [Pig 骰子游戏的最优策略 (Neller & Presser, Gettysburg)](https://cs.gettysburg.edu/~tneller/papers/pig.pdf)
- [求解 Pig 骰子游戏 — DP 入门](https://cs.gettysburg.edu/~tneller/nsf/pig/pig.pdf)
- [秘书问题（最优停止）— GeeksforGeeks](https://www.geeksforgeeks.org/dsa/secretary-problem-optimal-stopping-problem/)
- [秘书问题 — Wikipedia](https://en.wikipedia.org/wiki/Secretary_problem)
- [最优停止规则 — Subhash Suri (UCSB, PDF)](https://sites.cs.ucsb.edu/~suri/ccs130a/OptStopping.pdf)
- [用 Python 解秘书问题 — Imran Khan](http://www.imrankhan.dev/pages/Solving%20the%20secretary%20problem%20with%20Python.html)
- [不放回抽牌 — Heath Henley](https://heathhenley.dev/posts/drawing-without-replacement/)

## 主题 2 — 矩阵螺旋

- [螺旋矩阵 — LeetCode 54](https://leetcode.com/problems/spiral-matrix/)
- [螺旋矩阵 II — LeetCode 59](https://leetcode.com/problems/spiral-matrix-ii/)
- [螺旋矩阵 III — LeetCode 885](https://leetcode.com/problems/spiral-matrix-iii/)
- [螺旋矩阵 IV — LeetCode 2326](https://leetcode.com/problems/spiral-matrix-iv/)
- [对角线遍历 — LeetCode 498](https://leetcode.com/problems/diagonal-traverse/)
- [54. 螺旋矩阵 — 深度解析 (algo.monster)](https://algo.monster/liteproblems/54)
- [59. 螺旋矩阵 II — 深度解析 (algo.monster)](https://algo.monster/liteproblems/59)
- [885. 螺旋矩阵 III — 深度解析 (algo.monster)](https://algo.monster/liteproblems/885)
- [2326. 螺旋矩阵 IV — 深度解析 (algo.monster)](https://algo.monster/liteproblems/2326)
- [498. 对角线遍历 — 深度解析 (algo.monster)](https://algo.monster/liteproblems/498)
- [从某点开始的螺旋打印矩阵 — GeeksforGeeks](https://www.geeksforgeeks.org/dsa/print-matrix-spiral-form-starting-point/)
- [螺旋形式打印矩阵 — GeeksforGeeks](https://www.geeksforgeeks.org/dsa/print-a-given-matrix-in-spiral-form/)
- [矩阵螺旋遍历 — takeUforward](https://takeuforward.org/data-structure/spiral-traversal-of-matrix)
- [掌握 LeetCode 螺旋矩阵问题 (Neelam Yadav, Medium)](https://medium.com/@yaduvanshineelam09/mastering-the-spiral-matrix-problem-on-leetcode-3be6bd897f27)
- [LeetCode 885 螺旋矩阵 III walk-through (walkccc.me)](https://walkccc.me/LeetCode/problems/885/)
- [用递归解螺旋矩阵 III (Sai Krupa, Medium)](https://medium.com/@saikrupar82/solving-spiral-matrix-iii-with-recursion-885-spiral-matrix-iii-b495fd6fec2a)
- [解 LeetCode 2326 螺旋矩阵 IV (Sai Krupa, Medium)](https://medium.com/@saikrupar82/solving-leetcode-problem-2326-spiral-matrix-iv-66276120b3ec)

## ⭐ 进阶变体 — 主题 1

- [先知不等式 — Wikipedia](https://en.wikipedia.org/wiki/Prophet_inequality)
- [最优停止理论中的先知不等式综述 (Hill & Kertz, Wharton PDF)](http://www-stat.wharton.upenn.edu/~steele/Courses/900/Library/Prophet82Survey.pdf)
- [先知不等式 — Matt Weinberg, Princeton (Simons tutorial PDF)](https://simons.berkeley.edu/sites/default/files/docs/5302/simonstutorial-prophetinequalities.pdf)
- [Prophets and Secretaries — IPCO talk (NYU)](https://cs.nyu.edu/~anupamg/talks/ipco17/ipco-talk3.pdf)
- [Prophet Inequality — Brown CSCI 1440/2440 lecture](https://cs.brown.edu/courses/csci1440/lectures/fall-2025/prophet_inequality.pdf)
- [样本驱动最优停止 (arXiv 2011.06516)](https://arxiv.org/abs/2011.06516)
- [最优停止 — Wikipedia (卖房、卖资产)](https://en.wikipedia.org/wiki/Optimal_stopping)
- [有限时域停止规则 — UCLA Ferguson (PDF)](https://www.math.ucla.edu/~tom/Stopping/sr2.pdf)
- [无限时域折扣成本问题 — Polytechnique Montréal (PDF)](https://www.professeurs.polymtl.ca/jerome.le-ny/teaching/DP_fall09/notes/lec9_discounted.pdf)
- [Markov 链的最优停止 — Brown DAM (PDF)](https://www.dam.brown.edu/people/huiwang/classes/am226/Archive/stop.pdf)
- [Gittins 指数 — Wikipedia](https://en.wikipedia.org/wiki/Gittins_index)
- [多臂老虎机和 Gittins 指数定理 — Richard Weber (Cambridge PDF)](https://www.statslab.cam.ac.uk/~rrw1/oc/ocgittins.pdf)
- [多臂老虎机、Gittins 指数及其计算 — Chakravorty (McGill PDF)](https://www.ece.mcgill.ca/~amahaj1/projects/bandits/book/2013-bandit-computations.pdf)
- [乐观 Gittins 指数 — Gutin (MIT)](http://web.mit.edu/~vivekf/www/papers/OptGittins.pdf)
- [退休、停止时间和老虎机：Gittins 指数 — ML without tears](https://mlwithouttears.com/2023/11/24/retirement-stopping-times-and-bandits-the-gittins-index/)

## ⭐ 进阶变体 — 主题 2

- [旋转图像 — LeetCode 48](https://leetcode.com/problems/rotate-image/)
- [48. 旋转图像 — 深度解析 (algo.monster)](https://algo.monster/liteproblems/48)
- [对角线遍历 II — LeetCode 1424](https://leetcode.com/problems/diagonal-traverse-ii/)
- [1424. 对角线遍历 II — 深度解析 (algo.monster)](https://algo.monster/liteproblems/1424)
- [1424. 对角线遍历 II — walkccc 解答](https://walkccc.me/LeetCode/problems/1424/)
- [Toeplitz 矩阵 — LeetCode 766 (相关对角线模式)](https://leetcode.com/problems/toeplitz-matrix/)
- [对角线 & 反对角线遍历模式 — LeetCode discussion](https://leetcode.com/problems/toeplitz-matrix/solutions/1520613/diagonal-and-antidiagonal-traversal-something-that-will-help-with-all-matrix-problems/)
- [反对角线遍历 — GeeksforGeeks 练习](https://www.geeksforgeeks.org/problems/print-diagonally1623/1)
- [Facebook onsite — 矩阵反对角线遍历 (LeetCode discuss)](https://leetcode.com/discuss/interview-question/346342/facebook-onsite-matrix-antidiagonal-traverse/)
- [掌握 Java 矩阵遍历 (Yodgorbek Komilov, Medium)](https://medium.com/@YodgorbekKomilo/mastering-matrix-traversal-in-java-from-basics-to-spiral-leetcode-practice-3a68f66d1e82)
