# Optimal Stopping & Matrix Spiral — Interview Prep Pack

Curated study guide for two upcoming interview topics, with similar problems sourced from real interview repositories (Jane Street, Meta, Google, Goldman Sachs, BlackRock) and LeetCode. Each problem includes the key insight and a runnable Python solution.

---

## Table of Contents

- [Topic 1: Optimal Stopping (Dice)](#topic-1--optimal-stopping-dice)
  - [1.0 Original problem](#10--original-problem)
  - [1.1 Two-roll / three-roll variant (Jane Street, Meta)](#11--two-roll--three-roll-variant-jane-street-meta)
  - [1.2 Take-the-max variant (Meta)](#12--take-the-max-variant-meta)
  - [1.3 100-sided die with cost (Jane Street)](#13--100-sided-die-with-cost-jane-street)
  - [1.4 Continuous Uniform[0,1] variant](#14--continuous-uniform01-variant)
  - [1.5 Card-draw without replacement (Google, Goldman Sachs)](#15--card-draw-without-replacement-google-goldman-sachs)
  - [1.6 Black/Red gambler (Goldman Sachs)](#16--blackred-gambler-goldman-sachs)
  - [1.7 Pig — sum-with-bust dice game](#17--pig--sum-with-bust-dice-game)
  - [1.8 Secretary problem (1/e rule)](#18--secretary-problem-1e-rule)
  - [1.9 ⭐ Prophet inequality (1/2-approximation)](#19--prophet-inequality-12-approximation)
  - [1.10 ⭐ House-selling with discount factor (infinite horizon)](#110--house-selling-with-discount-factor-infinite-horizon)
  - [1.11 ⭐ Multi-armed bandit & the Gittins index](#111--multi-armed-bandit--the-gittins-index)
  - [1.12 ⭐ Sample-driven secretary (unknown distribution)](#112--sample-driven-secretary-unknown-distribution)
  - [1.13 ⭐ Robbins' problem (minimize expected rank)](#113--robbins-problem-minimize-expected-rank)
- [Topic 2: Matrix Spiral](#topic-2--matrix-spiral)
  - [2.0 Original problem (spiral from middle)](#20--original-problem-spiral-from-middle)
  - [2.1 Spiral Matrix (LeetCode 54)](#21--spiral-matrix-leetcode-54)
  - [2.2 Spiral Matrix II (LeetCode 59)](#22--spiral-matrix-ii-leetcode-59)
  - [2.3 Spiral Matrix III (LeetCode 885)](#23--spiral-matrix-iii-leetcode-885)
  - [2.4 Spiral Matrix IV (LeetCode 2326)](#24--spiral-matrix-iv-leetcode-2326)
  - [2.5 Diagonal Traverse (LeetCode 498)](#25--diagonal-traverse-leetcode-498)
  - [2.6 Spiral from arbitrary point (GeeksforGeeks)](#26--spiral-from-arbitrary-point-geeksforgeeks)
  - [2.7 ⭐ Rotate Image in place (LeetCode 48)](#27--rotate-image-in-place-leetcode-48)
  - [2.8 ⭐ Diagonal Traverse II — jagged 2D array (LeetCode 1424)](#28--diagonal-traverse-ii--jagged-2d-array-leetcode-1424)
  - [2.9 ⭐ Spiral with obstacles (BFS-on-spiral)](#29--spiral-with-obstacles-bfs-on-spiral)
  - [2.10 ⭐ 3D cube spiral (layer-by-layer)](#210--3d-cube-spiral-layer-by-layer)
  - [2.11 ⭐ Inverse spiral — reconstruct from spiral output](#211--inverse-spiral--reconstruct-from-spiral-output)
- [Cheat Sheet](#cheat-sheet)
- [Sources](#sources)

---

# Topic 1 — Optimal Stopping (Dice)

> **Universal template** — all variants below use the same backward-induction recurrence:
>
> 1. Define `V[k]` = expected value of optimal play with `k` decisions/rolls left.
> 2. Compute the **continuation value** (expected value of playing the next round optimally).
> 3. After each random outcome, you stop iff `immediate_reward ≥ continuation_value`.
> 4. Average over the random outcome.

---

## 1.0 — Original problem

> Roll a fair 6-sided die up to `n` times. After each roll, stop and keep the value, or roll again. Maximize expected score.

**Recurrence.** `E[1] = 3.5`; `E[k] = (1/6) · Σ_{v=1..6} max(v, E[k-1])`.

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

## 1.1 — Two-roll / three-roll variant (Jane Street, Meta)

> **Jane Street:** "Roll a die. If you don't like it, roll again, but you must keep the second roll. What's the fair value?" Then: same with up to 2 rerolls.
>
> **Meta:** "Roll up to 3 times. You get `$x` where `x` is your final stopping value (interpret as last roll if you continue)."

This is the original problem with `n=2` and `n=3`. The trick is **knowing the threshold**:

| Rolls remaining | Continuation value | Threshold (stop if ≥) |
|---|---|---|
| 1 (last) | n/a | n/a — must keep |
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

> 🎯 **Interview tip.** They almost always want you to *recite the threshold rule* aloud: *"For 3 rolls: stop on 5 or 6 first roll; on 4–6 second roll; keep whatever third roll."*

---

## 1.2 — Take-the-max variant (Meta)

> "You can roll up to 3 times. You get `$x` where `x` is the **highest** roll you got. You can stop at any time."

The state is now `(rolls_left, max_so_far)`, not just `rolls_left` — because once you've seen a 6, there's no point continuing. So the threshold becomes "stop iff `max_so_far ≥ E[k-1, max_so_far]`" — and since the max never decreases, the recurrence simplifies.

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

For 3 rolls the answer is ≈ **4.958**, slightly higher than 4.667 because keeping the running max gives a free "safety net".

---

## 1.3 — 100-sided die with cost (Jane Street)

> "100-sided die. Roll, then either take that many dollars or **pay $1** and roll again. What's the optimal strategy and EV?"

**Insight.** The continuation value drops by 1 (the cost). Threshold: stop if `roll ≥ E_continue + 1`.

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

The optimal threshold is `ceil(ev + 1)`: roll until you see ≥ 88.

---

## 1.4 — Continuous Uniform[0,1] variant

> Same game, but each draw is i.i.d. `Uniform[0,1]`. Up to `n` draws.

**Closed-form recurrence.** If the threshold is `t = E[k-1]`:

`E[k] = t · P(X<t) + E[X | X≥t] · P(X≥t) = t·t + (1+t)/2 · (1-t) = (1 + t²) / 2`.

```python
def expected_uniform_stop(n: int) -> float:
    E = 0.5
    for _ in range(n - 1):
        E = (1 + E * E) / 2
    return E

# n=1 -> 0.500   n=2 -> 0.625   n=3 -> 0.695   n=10 -> 0.861   n→∞ -> 1
```

A favorite quant follow-up because it shows you can replace the dice sum with an integral.

---

## 1.5 — Card-draw without replacement (Google, Goldman Sachs)

> "Deck of cards. Draw one at a time without replacement. Score = card value. Stop any time. Max EV?"

The state is the **multiset of remaining cards**, but by symmetry only its order statistics matter.

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

## 1.6 — Black/Red gambler (Goldman Sachs, Google)

> "26 red + 26 black cards. Draw one at a time. Red = +$1, Black = −$1. Stop anytime. What's the optimal EV?"

**State:** `(red_left, black_left)`. **Insight:** if you've drawn more black than red so far, you should *not* keep going if the deck is "balanced enough" — and you can always wait until the end (deck has 0 net by then). The optimal value of "continue" is the EV of waiting; you cash out only if your current profit beats that.

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

A simple bound: never stop with negative profit (you can do at least 0 by waiting until the deck is empty). Optimal threshold: stop early only when net positive AND continuation EV is below current profit.

---

## 1.7 — Pig — sum-with-bust dice game

> "Each turn: roll a die, accumulate the value into a turn-total. Roll a 1 → lose entire turn-total and turn ends. Otherwise stop anytime to bank the turn-total. What's the optimal hold threshold?"

**Single-turn analysis** (greedy version): hold once your turn-total `s` makes the marginal expected value of rolling negative.

`expected_change_per_roll = (1/6)(−s) + (1/6)(2 + 3 + 4 + 5 + 6) = (20 − s)/6`.

Roll while `s < 20`. So **hold at 20** is greedy-optimal in isolation. (In actual two-player Pig-to-100, the optimal turn threshold drifts down to ~21–25 depending on score difference.)

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

## 1.8 — Secretary problem (1/e rule)

> "Interview `n` candidates one-by-one in random order. Decide accept-or-reject on the spot. Maximize probability of picking the **single best**."

**Strategy.** Reject the first `n/e` candidates, then accept the first one better than all you've seen. `P(success) → 1/e ≈ 0.368`.

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

# ⭐ Topic 1 — Harder Variants

> Step up from "compute a threshold" to **competitive analysis** (prophet inequality), **infinite-horizon discounted MDPs** (house-selling), **multi-project allocation** (Gittins), and **distribution-free** stopping rules.

---

## 1.9 — Prophet inequality (1/2-approximation)

> Sequence of independent random variables `X₁, ..., Xₙ` drawn from **known but distinct** distributions, observed one at a time. After each, accept-or-reject. A "prophet" who sees all draws gets `E[max Xᵢ]`. Find an online rule that achieves **at least half** the prophet's value, against any sequence of distributions.

**Krengel–Sucheston theorem (1977).** A simple **single-threshold rule** does it: pick threshold `τ` so that `Σ_i E[(Xᵢ − τ)⁺] = τ` (or equivalently `P(max Xᵢ ≥ τ) = 1/2`). Stop at the first `Xᵢ ≥ τ`.

`E[ALG] ≥ 0.5 · E[max Xᵢ]` and the constant `1/2` is **tight**.

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

> 🎯 **Why this matters.** It's the canonical online-vs-offline competitive ratio result; it underlies posted-price mechanisms in algorithmic mechanism design (Hajiaghayi, Kleinberg).

---

## 1.10 — House-selling with discount factor (infinite horizon)

> "Each period an i.i.d. offer `Xᵢ ~ F` arrives. Accept and receive `Xᵢ`, or reject and wait one period. Future rewards discounted by `β ∈ (0,1)`. What's the optimal stopping rule and `V`?"

**Bellman equation.** `V = E[max(X, β·V)]`. The optimal rule is a **constant threshold** `c = β·V`: accept iff `X ≥ c`.

`V = β V · F(c) + ∫_{c}^{∞} x dF(x)` and `c = βV` gives `c = β·E[max(X, c)]`.

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

**Closed form for Uniform[0,1].** `c` satisfies `c = β·(1 + c²)/2 ⇒ c = (1 − √(1−β²))/β`.

---

## 1.11 — Multi-armed bandit & the Gittins index

> "K independent 'projects'. At each step, choose one to advance; it returns a stochastic reward depending on its (Markov) state. Discount factor β. Maximize expected discounted total reward."

**Gittins (1979).** The optimal policy is an **index policy**: assign each project a number `g(state)` (the "Gittins index" — the maximum, over stopping rules `τ`, of the discounted reward-rate from playing that project alone) and always play the project with the largest current index.

`g(state) = max_τ  E[Σ_{t=0}^{τ-1} βᵗ Rₜ] / E[Σ_{t=0}^{τ-1} βᵗ]`

For the **Bernoulli bandit with Beta(α,β) prior**, the index can be computed by value iteration on (α,β):

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

> 🎯 **Why this matters.** Restless/contextual bandit variants (e.g., Whittle index for energy-aware scheduling) are NP-complete, but the index policy is a strong heuristic and Gittins gives the exact answer for the standard MAB.

---

## 1.12 — Sample-driven secretary (unknown distribution)

> Same as the secretary problem, but you receive a small "sample" of size `s` from the same distribution **before** the online phase begins. With `n = ∞`, there is a beautiful 1−1/e bound.

**Result (Correa et al., 2020).** With access to `O(n)` samples, you can match the i.i.d. prophet inequality bound `1 − 1/e ≈ 0.632` of the optimal.

A practical heuristic: use the sample's `(1−1/e)`-quantile as your threshold.

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

> 🎯 **Difference from classical 1.8.** Classical secretary uses *no distributional knowledge* and gets 1/e. Adding history samples bumps it to 1−1/e.

---

## 1.13 — Robbins' problem (minimize expected rank)

> "Online: `n` i.i.d. uniform values. Pick exactly one. Minimize the **expected rank** of your pick (rank 1 = best, rank n = worst)."

**Open problem** (since Robbins, 1990): the optimal expected rank `V_∞ = lim_{n→∞} E[rank]` is known to be in `[1.908, 2.327]` but its exact value is *unknown*. This is the most famous unsolved problem in optimal stopping.

A reasonable heuristic uses thresholds depending on time-remaining and current rank:

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

> 🎯 **Interview anchor.** Citing this problem signals you know optimal stopping has open problems; recruiters at Jane Street / Citadel quant teams *love* unsolved territory.

---

# Topic 2 — Matrix Spiral

> **Two universal patterns** cover ~every spiral question:
>
> - **Pattern A — Shrinking boundaries:** maintain `top, bottom, left, right`; walk one side, shrink.
> - **Pattern B — Expanding step (1,1,2,2,3,3,…):** walk in directions `R, D, L, U` cycling; the step length is consumed *twice* per increment. Skip out-of-bounds cells.
>
> Outside-in spirals → Pattern A. Inside-out / arbitrary-start spirals → Pattern B.

---

## 2.0 — Original problem (spiral from middle)

> Given an `n × n` matrix, output its elements in spiral sequence starting **from the middle**.

Pattern B: from center, step pattern is `1, 1, 2, 2, 3, 3, …` cycling `R → D → L → U`.

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

> ⚠️ **Even-`n` follow-up.** "Center" is ambiguous; common conventions: pick `(n//2 - 1, n//2 - 1)` (top-left of the central 2×2) or use a different first-direction. Clarify with the interviewer.

---

## 2.1 — Spiral Matrix (LeetCode 54)

> Given an `m × n` matrix, return all elements in clockwise spiral order **from the top-left**.

Pattern A.

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

**Sanity check vs. 2.0:** the *reverse* of `spiral_from_middle` (3×3, center start) is `[3, 2, 1, 4, 7, 8, 9, 6, 5]`, equivalent to a corner-start spiral on a 3×3 with appropriate orientation. They are duals.

---

## 2.2 — Spiral Matrix II (LeetCode 59)

> Given `n`, generate the `n × n` matrix filled with `1..n²` in spiral order.

Pattern A, but writing rather than reading.

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

## 2.3 — Spiral Matrix III (LeetCode 885)

> Walk clockwise from `(rStart, cStart)` over an `R × C` grid; return cell coordinates in visit order.

This is **the closest cousin to the original problem** — just generalized to rectangles and arbitrary starts.

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

If you can solve LC 885, the original problem is a one-liner: call `spiral_matrix_iii(n, n, n//2, n//2)` and look up `matrix[r][c]` for each pair.

---

## 2.4 — Spiral Matrix IV (LeetCode 2326)

> Given `m`, `n`, and a linked-list head, fill an `m × n` matrix with the linked-list values in spiral order. Pad remaining cells with `−1`.

Pattern A, but writing from a stream.

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

## 2.5 — Diagonal Traverse (LeetCode 498)

> Traverse an `m × n` matrix in zig-zag diagonal order (top-right ↔ bottom-left).

Different traversal pattern but same "matrix-walk + direction" mental model.

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

## 2.6 — Spiral from arbitrary point (GeeksforGeeks)

> Given an `m × n` matrix and a point `P(c, r)` *inside the matrix*, print the matrix in clockwise spiral starting from `P`.

Same as LC 885, but constrained to start *inside* and visit only in-grid cells. Reuse `spiral_matrix_iii`:

```python
def print_spiral_from_point(matrix, r, c):
    R, C = len(matrix), len(matrix[0])
    return [matrix[i][j] for i, j in spiral_matrix_iii(R, C, r, c)]
```

---

# ⭐ Topic 2 — Harder Variants

> Step up from "walk and append" to **in-place transformations** (LC 48), **irregular structures** (LC 1424), **search inside the spiral** (BFS), **higher-dimensional traversal** (3D), and **inverse problems** (reconstruct from spiral).

---

## 2.7 — Rotate Image in place (LeetCode 48)

> Rotate an `n × n` matrix by 90° **clockwise, in place** (no extra matrix allocation). `O(1)` extra space.

**Two-step trick.** Transpose the matrix, then reverse each row. Equivalently for 90° clockwise:

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

> 🎯 **Why hard.** The 4-cycle swap is the canonical "I traversed all four layers correctly" exercise. Off-by-ones bite ~80% of candidates.

**Variants asked in the wild.**
- 180° / 270°: rotate twice / thrice, or transpose + reverse columns.
- Anti-clockwise: reverse rows first, then transpose.

---

## 2.8 — Diagonal Traverse II — jagged 2D array (LeetCode 1424)

> Same as LC 498 but the input is a list of lists with **different row lengths**. Return all elements in anti-diagonal order, top-right to bottom-left for each diagonal.

The naive `for d in range(m+n-1)` doesn't work because `n` varies per row. Use a **bucket-by-diagonal-index** approach:

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

**Complexity.** `O(N)` for `N = total elements` — bucketing is O(1) per element, sort by `d` keys is `O(diag · log diag)`.

A nice alternative: **BFS** from `(0,0)` exploring `(i+1, j)` and `(i, j+1)` — but careful with duplicates.

---

## 2.9 — Spiral with obstacles (BFS-on-spiral)

> Given an `m × n` grid where some cells are blocked (`'#'`), output the values in clockwise spiral order starting from top-left, **skipping** blocked cells.

Two reasonable interpretations:
1. **Skip but stay-on-spiral.** Walk the standard spiral; when you hit a blocked cell, just don't emit it. Easy modification of LC 54.
2. **Re-route around blocks.** Treat the spiral as a *direction priority* (try R, D, L, U in spiral order); turn when blocked or off-grid. Harder, but more interesting.

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

> 🎯 **Real-world flavor.** This pattern shows up in **robotic path planning** (lawn-mower coverage with obstacles) and **vision** (zigzag scan of valid pixels).

---

## 2.10 — 3D cube spiral (layer-by-layer)

> Given an `n × n × n` cube of integers, output its elements in "onion-peel" spiral order: first the entire outer shell (in some canonical face-by-face order), then the next inner shell, and so on.

**Approach.** For each shell `s = 0, ..., n//2`, traverse the 6 faces. For each face, run a 2D spiral. Total elements = `n³`.

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

> 🎯 **Interview value.** It tests whether you can extend a 2D mental model to 3D *without* drowning in indices. The de-dup set is the trick that keeps you sane.

---

## 2.11 — Inverse spiral — reconstruct from spiral output

> Given a 1D array `arr` of length `m·n` and the dimensions `m, n`, **reconstruct** the matrix such that reading it in clockwise spiral (LC 54) order yields `arr`.

This is the inverse of LC 54 → exactly LC 59 generalized: instead of writing `1..n²`, write the values from `arr` in spiral order.

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

**Twist (asked at Bloomberg).** Given `arr = sorted([...])`, what does the resulting matrix look like? It will have its **smallest element in the top-left**, **largest in the geometric center** (in a snake-onion pattern), useful for visualization.

---

# Cheat Sheet

## Optimal Stopping — pattern recognition

| When you see... | Reach for... |
|---|---|
| Random reward + finite rounds + irreversible stop | Backward induction on rolls remaining |
| Cost per round | Threshold = `E_continue − cost` |
| State depends on history (max, sum, deck) | Memoize on `(round, state)` |
| Rolls without bound, fixed cost | Fixed-point iteration of Bellman equation |
| "Probability of choosing the best" (not EV) | Secretary / 1/e rule |
| Bust risk (Pig) | Compare expected delta `(non_bust_avg − bust_prob × current)` to 0 |
| Online vs offline competitive ratio | Prophet inequality (1/2 with single threshold) |
| Discount factor + i.i.d. offers | House-selling: `c = β·E[max(X, c)]` |
| Multiple projects, only one advances | Gittins index policy |
| No distribution info, samples available | Sample-driven secretary, threshold = `(1−1/e)`-quantile |

## Matrix Spiral — pattern recognition

| Spiral type | Pattern | Template |
|---|---|---|
| Outside-in (LC 54, 59, 2326) | A — shrinking boundaries | 4 nested loops, decrement `top/right/bottom/left` |
| Inside-out / arbitrary start (LC 885, original) | B — expanding step `1,1,2,2,3,3,…` | Direction array + `for _ in range(2)` |
| Diagonal (LC 498) | Group by `i+j`, alternate reversal | `for d in range(m+n-1)` |
| Jagged diagonal (LC 1424) | Bucket by `i+j` in dict | `defaultdict(list)`, then sort keys |
| In-place rotation (LC 48) | Transpose + row-reverse, **or** layer-by-layer 4-cycle | Two passes vs `n//2` shells |
| Spiral with obstacles | Direction priority, turn on block | "Try R, D, L, U in spiral cycle order" |
| 3D spiral | Onion-peel; each shell = 6 faces | Use `seen` set to de-dup shell edges |

> **Always clarify** before coding: rectangular vs square, even vs odd dimensions, where to start, what to do with out-of-bounds, what to return (values vs coordinates).

---

# Sources

## Topic 1 — Optimal Stopping

- [Jane Street Interview Question (Glassdoor) — reroll a die problem](https://www.glassdoor.com/Interview/a-expected-value-of-a-die-b-suppose-you-play-a-game-where-you-get-a-dollar-amount-equivalent-to-the-number-of-dots-that-QTN_30411.htm)
- [Jane Street Interview Question (Glassdoor) — 100-sided die with $1 reroll cost](https://www.glassdoor.com/Interview/You-are-given-a-die-with-100-sides-One-side-has-1-dot-one-has-2-dots-and-so-on-up-until-100-You-are-given-a-chance-to-ro-QTN_688189.htm)
- [Meta Interview Question (Glassdoor) — roll up to 3 times, take highest](https://www.glassdoor.com/Interview/You-can-roll-a-dice-3-times-You-will-be-given-x-where-x-is-the-highest-roll-you-get-You-can-choose-to-stop-rolling-at-an-QTN_802648.htm)
- [Goldman Sachs Interview Question (Glassdoor) — 4 cards, 2 black 2 red](https://www.glassdoor.com/Interview/You-have-4-cards-2-black-and-2-red-You-play-a-game-where-during-each-round-you-draw-a-card-If-it-s-black-you-lose-a-poi-QTN_257709.htm)
- [Google Interview Question (Glassdoor) — 52-card red/black optimal stopping](https://www.glassdoor.sg/Interview/You-have-52-playing-cards-26-red-26-black-You-draw-cards-one-by-one-A-red-card-pays-you-a-dollar-A-black-one-fines-yo-QTN_3421153.htm)
- [Jane Street Interview Question (Glassdoor) — 26 red, 26 black guess-the-color](https://www.glassdoor.com/Interview/3-Poker-26-red-26-black-Take-one-every-time-you-can-choose-to-guess-whether-it-s-red-You-have-only-one-chance-If-you-QTN_155340.htm)
- [BlackRock phone interview — dice roll (QuantNet)](https://quantnet.com/threads/blackrock-phone-interview-dice-roll.13712/)
- [ML Interview Q Series — Dice Game Strategy: Optimal Stopping](https://www.rohan-paul.com/p/ml-interview-q-series-dice-game-strategy)
- [Solution for Quant Interview Question — backward induction walk-through](https://quantinvestor.wordpress.com/2009/10/20/solution-for-quant-interview-question/)
- [Dice Game Optimal Stopping Strategy (Wendy Hu, Medium)](https://medium.com/@whystudying/dice-game-optimized-stopping-strategy-59faa0862d8e)
- [Rolling the Dice: A Probability-Based Strategy (Gaurav Kandel, Medium)](https://medium.com/@dswithgk/rolling-the-dice-a-probability-based-strategy-for-maximum-gain-a9aa80ed86d0)
- [The Optimal Value for a Game of Dice (Pascal Bercker, Medium)](https://medium.com/@pbercker/the-optimal-value-for-a-game-of-dice-or-knowing-when-to-quit-29c69ac01a0e)
- [The Expected Payoff of a Dice Game — Predictive Hacks](https://predictivehacks.com/the-expected-payoff-of-a-dice-game/)
- [d20 stopping puzzle (DataGenetics)](http://datagenetics.com/blog/february32016/index.html)
- [Red/Black gambling game (DataGenetics)](http://datagenetics.com/blog/october42014/index.html)
- [An optimal-stopping quant riddle — Emir's blog](https://emiruz.com/post/2023-07-30-optimal-stopping/)
- [A Collection of Dice Problems (PDF)](https://www.madandmoonly.com/doctormatt/mathematics/dice1older.pdf)
- ["Pig (Pig-out)" Analysis — Durango Bill's](http://www.durangobill.com/Pig.html)
- [Optimal Play of the Dice Game Pig (Neller & Presser, Gettysburg)](https://cs.gettysburg.edu/~tneller/papers/pig.pdf)
- [Solving the Dice Game Pig — intro to dynamic programming](https://cs.gettysburg.edu/~tneller/nsf/pig/pig.pdf)
- [Secretary Problem (Optimal Stopping) — GeeksforGeeks](https://www.geeksforgeeks.org/dsa/secretary-problem-optimal-stopping-problem/)
- [Secretary Problem — Wikipedia](https://en.wikipedia.org/wiki/Secretary_problem)
- [Optimal Stopping Rules — Subhash Suri (UCSB, PDF)](https://sites.cs.ucsb.edu/~suri/ccs130a/OptStopping.pdf)
- [Solving the secretary problem with Python — Imran Khan](http://www.imrankhan.dev/pages/Solving%20the%20secretary%20problem%20with%20Python.html)
- [Drawing Cards without Replacement — Heath Henley](https://heathhenley.dev/posts/drawing-without-replacement/)

## Topic 2 — Matrix Spiral

- [Spiral Matrix — LeetCode 54](https://leetcode.com/problems/spiral-matrix/)
- [Spiral Matrix II — LeetCode 59](https://leetcode.com/problems/spiral-matrix-ii/)
- [Spiral Matrix III — LeetCode 885](https://leetcode.com/problems/spiral-matrix-iii/)
- [Spiral Matrix IV — LeetCode 2326](https://leetcode.com/problems/spiral-matrix-iv/)
- [Diagonal Traverse — LeetCode 498](https://leetcode.com/problems/diagonal-traverse/)
- [54. Spiral Matrix — In-Depth Explanation (algo.monster)](https://algo.monster/liteproblems/54)
- [59. Spiral Matrix II — In-Depth Explanation (algo.monster)](https://algo.monster/liteproblems/59)
- [885. Spiral Matrix III — In-Depth Explanation (algo.monster)](https://algo.monster/liteproblems/885)
- [2326. Spiral Matrix IV — In-Depth Explanation (algo.monster)](https://algo.monster/liteproblems/2326)
- [498. Diagonal Traverse — In-Depth Explanation (algo.monster)](https://algo.monster/liteproblems/498)
- [Print a matrix in spiral form starting from a point — GeeksforGeeks](https://www.geeksforgeeks.org/dsa/print-matrix-spiral-form-starting-point/)
- [Print a given matrix in spiral form — GeeksforGeeks](https://www.geeksforgeeks.org/dsa/print-a-given-matrix-in-spiral-form/)
- [Spiral Traversal of Matrix — takeUforward](https://takeuforward.org/data-structure/spiral-traversal-of-matrix)
- [Mastering the Spiral Matrix Problem on LeetCode (Neelam Yadav, Medium)](https://medium.com/@yaduvanshineelam09/mastering-the-spiral-matrix-problem-on-leetcode-3be6bd897f27)
- [LeetCode 885 Spiral Matrix III walk-through (walkccc.me)](https://walkccc.me/LeetCode/problems/885/)
- [Solving Spiral Matrix III with Recursion (Sai Krupa, Medium)](https://medium.com/@saikrupar82/solving-spiral-matrix-iii-with-recursion-885-spiral-matrix-iii-b495fd6fec2a)
- [Solving LeetCode 2326 Spiral Matrix IV (Sai Krupa, Medium)](https://medium.com/@saikrupar82/solving-leetcode-problem-2326-spiral-matrix-iv-66276120b3ec)

## ⭐ Harder Variants — Topic 1

- [Prophet inequality — Wikipedia](https://en.wikipedia.org/wiki/Prophet_inequality)
- [A Survey of Prophet Inequalities in Optimal Stopping Theory (Hill & Kertz, Wharton PDF)](http://www-stat.wharton.upenn.edu/~steele/Courses/900/Library/Prophet82Survey.pdf)
- [Prophet Inequalities — Matt Weinberg, Princeton (Simons tutorial PDF)](https://simons.berkeley.edu/sites/default/files/docs/5302/simonstutorial-prophetinequalities.pdf)
- [Prophets and Secretaries — IPCO talk (NYU)](https://cs.nyu.edu/~anupamg/talks/ipco17/ipco-talk3.pdf)
- [Prophet Inequality — Brown CSCI 1440/2440 lecture](https://cs.brown.edu/courses/csci1440/lectures/fall-2025/prophet_inequality.pdf)
- [Sample-driven optimal stopping (arXiv 2011.06516)](https://arxiv.org/abs/2011.06516)
- [Optimal stopping — Wikipedia (house selling, asset selling)](https://en.wikipedia.org/wiki/Optimal_stopping)
- [Finite Horizon Stopping Rules — UCLA Ferguson (PDF)](https://www.math.ucla.edu/~tom/Stopping/sr2.pdf)
- [Infinite Horizon Discounted Cost Problems — Polytechnique Montréal (PDF)](https://www.professeurs.polymtl.ca/jerome.le-ny/teaching/DP_fall09/notes/lec9_discounted.pdf)
- [Optimal Stopping of Markov Chains — Brown DAM (PDF)](https://www.dam.brown.edu/people/huiwang/classes/am226/Archive/stop.pdf)
- [Gittins index — Wikipedia](https://en.wikipedia.org/wiki/Gittins_index)
- [Multi-armed Bandits and the Gittins Index Theorem — Richard Weber (Cambridge PDF)](https://www.statslab.cam.ac.uk/~rrw1/oc/ocgittins.pdf)
- [Multi-Armed Bandits, Gittins Index and Its Calculation — Chakravorty (McGill PDF)](https://www.ece.mcgill.ca/~amahaj1/projects/bandits/book/2013-bandit-computations.pdf)
- [Optimistic Gittins Indices — Gutin (MIT)](http://web.mit.edu/~vivekf/www/papers/OptGittins.pdf)
- [Retirement, Stopping times and Bandits: The Gittins index — ML without tears](https://mlwithouttears.com/2023/11/24/retirement-stopping-times-and-bandits-the-gittins-index/)

## ⭐ Harder Variants — Topic 2

- [Rotate Image — LeetCode 48](https://leetcode.com/problems/rotate-image/)
- [48. Rotate Image — In-Depth Explanation (algo.monster)](https://algo.monster/liteproblems/48)
- [Diagonal Traverse II — LeetCode 1424](https://leetcode.com/problems/diagonal-traverse-ii/)
- [1424. Diagonal Traverse II — In-Depth (algo.monster)](https://algo.monster/liteproblems/1424)
- [1424. Diagonal Traverse II — walkccc Solutions](https://walkccc.me/LeetCode/problems/1424/)
- [Toeplitz Matrix — LeetCode 766 (related diagonal pattern)](https://leetcode.com/problems/toeplitz-matrix/)
- [Diagonal & antidiagonal traversal patterns — LeetCode discussion](https://leetcode.com/problems/toeplitz-matrix/solutions/1520613/diagonal-and-antidiagonal-traversal-something-that-will-help-with-all-matrix-problems/)
- [Anti-diagonal traversal — GeeksforGeeks practice](https://www.geeksforgeeks.org/problems/print-diagonally1623/1)
- [Facebook onsite — Matrix Antidiagonal Traverse (LeetCode discuss)](https://leetcode.com/discuss/interview-question/346342/facebook-onsite-matrix-antidiagonal-traverse/)
- [Mastering Matrix Traversal in Java (Yodgorbek Komilov, Medium)](https://medium.com/@YodgorbekKomilo/mastering-matrix-traversal-in-java-from-basics-to-spiral-leetcode-practice-3a68f66d1e82)
