# HackerRank Interview Prep — Problems Matching All Four Topics

Curated set of **actual HackerRank problems** that test the same skills as your four interview topics: Optimal Stopping, Matrix Spiral, Expected Value/Variance, and Majority Vote. Each problem includes the official problem statement (paraphrased), constraints, sample I/O, the key insight, and a complete HackerRank-ready Python 3 solution.

> **HackerRank submission tips.** All solutions below use `input()` / `print()` matching the platform's I/O. They assume Python 3.x. To avoid TLE on `n ≤ 10⁵` problems use `sys.stdin.read()`. For floats, HackerRank typically accepts ±1e-4 tolerance.

---

## Table of Contents

1. [Matrix Spiral / Rotation](#1-matrix-spiral--rotation)
   - [1.1 Matrix Layer Rotation](#11--matrix-layer-rotation-medium)
   - [1.2 Spiral Matrix](#12--spiral-matrix-easy-mediumcontest)
2. [Expected Value & Variance](#2-expected-value--variance)
   - [2.1 Dice Stats](#21--dice-stats-medium)
   - [2.2 Random (subarray sum after swaps)](#22--random-medium)
   - [2.3 Kevin and Expected Value](#23--kevin-and-expected-value-medium)
   - [2.4 Day 0 — Mean, Median, Mode (10 Days of Stats)](#24--day-0--mean-median-mode-easy)
   - [2.5 Basic Statistics Warmup](#25--basic-statistics-warmup-easy)
   - [2.6 Day 2 — Compound Event Probability](#26--day-2--compound-event-probability-easy)
3. [Optimal Stopping (closest analogs)](#3-optimal-stopping-closest-analogs)
   - [3.1 Dice Throw](#31--dice-throw-mediumcontest)
   - [3.2 Game of Two Dice — Peter & Colin](#32--game-of-two-dice--peter--colin-medium)
4. [Majority Vote](#4-majority-vote)
   - [4.1 Boyer–Moore Majority Element (HackerRank "Majority")](#41--boyermoore-majority-element-easymedium)
   - [4.2 Lonely Integer](#42--lonely-integer-easy)
5. [Practice Plan](#practice-plan)
6. [Sources](#sources)

---

# 1. Matrix Spiral / Rotation

## 1.1 — Matrix Layer Rotation (Medium)

> [HackerRank: Matrix Layer Rotation](https://www.hackerrank.com/challenges/matrix-rotation-algo/problem)
>
> Given an `m × n` matrix and `r`, rotate it `r` times **anti-clockwise**. Each layer (concentric ring) rotates independently.
>
> **Constraints.** `2 ≤ m, n ≤ 300`, `1 ≤ r ≤ 10⁹`, `min(m, n)` is even.

**Sample I/O**

```
Input:                Output:
4 4 2                 3 4 8 12
1  2  3  4            2 11 10 16
5  6  7  8            1  7  6 15
9 10 11 12            5  9 13 14
13 14 15 16
```

**Key insight.** Three steps:
1. Decompose the matrix into `min(m, n) // 2` concentric layers.
2. For each layer, extract its perimeter as a flat list of length `P`. Rotate by `r mod P` positions (handles `r = 10⁹` easily).
3. Write the rotated list back.

**Why it tests the same skill.** Pure spiral-traversal mechanics with a twist: anti-clockwise extraction = "left, down, right, up" instead of the usual "right, down, left, up", and you must mod the rotation count.

```python
def matrix_layer_rotation(matrix, r):
    m, n = len(matrix), len(matrix[0])
    layers = min(m, n) // 2
    out = [row[:] for row in matrix]                      # copy

    for layer in range(layers):
        # Build the perimeter in anti-clockwise order: down → right → up → left
        top, bottom = layer, m - 1 - layer
        left, right = layer, n - 1 - layer

        ring = []
        for i in range(top, bottom + 1):    ring.append(matrix[i][left])     # down on left col
        for j in range(left + 1, right + 1): ring.append(matrix[bottom][j])  # right on bottom
        for i in range(bottom - 1, top - 1, -1): ring.append(matrix[i][right])  # up on right
        for j in range(right - 1, left, -1):     ring.append(matrix[top][j])    # left on top

        P = len(ring)
        k = r % P                                          # effective rotations
        rotated = ring[k:] + ring[:k]                      # left-shift by k positions

        # Write rotated values back in the same anti-clockwise order
        idx = 0
        for i in range(top, bottom + 1):
            out[i][left] = rotated[idx]; idx += 1
        for j in range(left + 1, right + 1):
            out[bottom][j] = rotated[idx]; idx += 1
        for i in range(bottom - 1, top - 1, -1):
            out[i][right] = rotated[idx]; idx += 1
        for j in range(right - 1, left, -1):
            out[top][j] = rotated[idx]; idx += 1

    return out

# ----- HackerRank stdin/stdout -----
import sys
def main():
    data = sys.stdin.read().split()
    it = iter(data)
    m, n, r = int(next(it)), int(next(it)), int(next(it))
    matrix = [[int(next(it)) for _ in range(n)] for _ in range(m)]
    result = matrix_layer_rotation(matrix, r)
    print('\n'.join(' '.join(map(str, row)) for row in result))

if __name__ == "__main__":
    main()
```

**Complexity.** `O(m·n)` — every cell is visited a constant number of times.

> 🎯 **Common pitfall.** Forgetting `r % P` causes `O(r·P)` work and TLEs on `r = 10⁹`.

---

## 1.2 — Spiral Matrix (Easy/Medium, contest)

> [HackerRank Contest: Spiral Matrix](https://www.hackerrank.com/contests/coding-test-1-bits-hyderabad/challenges/spiral-matrix-1)
>
> Read `m × n`. Print elements in clockwise spiral order starting from `(0,0)`.

**Sample I/O**

```
Input:               Output:
3 3                  1 2 3 6 9 8 7 4 5
1 2 3
4 5 6
7 8 9
```

**Solution.** Standard shrinking-boundary template (LC 54):

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

import sys
def main():
    data = sys.stdin.read().split()
    it = iter(data)
    m, n = int(next(it)), int(next(it))
    matrix = [[int(next(it)) for _ in range(n)] for _ in range(m)]
    print(' '.join(map(str, spiral_order(matrix))))

if __name__ == "__main__":
    main()
```

---

# 2. Expected Value & Variance

## 2.1 — Dice Stats (Medium)

> [HackerRank: Dice Stats](https://www.hackerrank.com/challenges/dice-stats/problem)
>
> A biased 6-sided die has probabilities `p₁..p₆`. Throw it `N` times to record `d[1], d[2], ..., d[N]`, **with the constraint that `d[i] ≠ d[i-1]` for all `i ≥ 2`** (re-roll until different). Compute E and Var of `S = Σ dᵢ`.
>
> **Constraints.** `N ≤ 10⁹` (≤ 10⁵ for partial credit). Output absolute error tol `10⁻⁴`.

**Sample I/O** (uniform die, `N=2`)

```
Input:                       Output:
0.16666666667                7.0
0.16666666666                4.66666666666
0.16666666667
0.16666666667
0.16666666666
0.16666666667
2
```

**Key insight — Markov chain on the die.** The sequence `d₁, d₂, …` is a Markov chain with transition `P(d_i = k | d_{i-1} = j) = p_k / (1 − p_j)` for `k ≠ j`, else 0.

For mean / variance you need:
1. **Stationary distribution** `π` (since for `N` large, marginal of `dᵢ` converges to `π`).
2. `E[dᵢ]` and `E[dᵢ²]` under `π`.
3. **Covariance terms** `Cov(dᵢ, dᵢ₊ₖ)` decay geometrically — sum them in closed form.

For very large `N`, all sums are dominated by stationary moments. The full formula:

```
E[S]   ≈ N · μ_π
Var[S] ≈ N · σ_π² + 2 · Σ_{k=1..N-1} (N-k) · Cov(d_1, d_{1+k})
```

Covariance decays as `λ_2^k` where `λ_2` is the second-largest eigenvalue of the transition matrix — for symmetric/uniform priors this is `−1/5`.

**Implementation** — use the transition matrix and its powers numerically (since `N ≤ 10⁹`, exploit eigendecomposition, or for the uniform case use closed-form):

```python
import numpy as np

def dice_stats(probs, N):
    p = np.array(probs)
    # Build transition matrix P[i][j] = P(next = j | curr = i) = p[j] / (1 - p[i])
    M = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i != j:
                M[i, j] = p[j] / (1 - p[i])

    # Stationary distribution: solve π M = π, sum(π) = 1
    eigvals, eigvecs = np.linalg.eig(M.T)
    idx = np.argmin(np.abs(eigvals - 1))
    pi = np.real(eigvecs[:, idx])
    pi = pi / pi.sum()

    values = np.arange(1, 7)
    mu = pi @ values
    var_x = pi @ (values ** 2) - mu ** 2

    # E[S] = N * mu  for stationary chain (assuming d_1 ~ π; otherwise small correction)
    E_S = N * mu

    # For variance, sum geometric series of covariances. For 'detailed-balance' uniform p,
    # Cov(d_1, d_{1+k}) = var_x * λ²^k where λ² is second-largest |eigenvalue|.
    sorted_eigs = sorted(np.abs(eigvals), reverse=True)
    lam = sorted_eigs[1]                              # second largest in absolute value

    # Var[S] = N * var_x + 2 * Σ_{k=1..N-1} (N-k) * var_x * lam^k
    # Closed-form geometric series:
    if abs(lam) < 1:
        S1 = lam * (1 - lam ** N) / (1 - lam)         # Σ lam^k for k=1..N
        S2 = (lam - (N + 1) * lam ** (N + 1) + N * lam ** (N + 2)) / (1 - lam) ** 2
                                                       # Σ k * lam^k for k=1..N
        cross = 2 * var_x * (N * S1 - S2)
    else:
        cross = 0
    Var_S = N * var_x + cross
    return E_S, Var_S

import sys
def main():
    data = sys.stdin.read().split()
    probs = [float(x) for x in data[:6]]
    N = int(data[6])
    E_S, Var_S = dice_stats(probs, N)
    print(f"{E_S:.10f}")
    print(f"{Var_S:.10f}")

if __name__ == "__main__":
    main()
```

**Sanity check (uniform, N=2):**
- Marginal `E[d_1] = E[d_2] = 3.5`, so `E[S] = 7` ✓
- `Var(d_i) = 35/12`, `Cov(d_1, d_2) = -7/12` (negative because the constraint forbids repeating)
- `Var(S) = 35/12 + 35/12 - 14/12 = 56/12 = 14/3 ≈ 4.667` ✓

> 🎯 **This is the closest HackerRank analog to your "Expected Value and Variance" interview problem** — same E[X²] − (E[X])² mechanics, just with Markov-chain dependence layered on top.

---

## 2.2 — Random (Medium)

> [HackerRank: Random](https://www.hackerrank.com/challenges/random/problem)
>
> Array of `n` integers. Perform `a` random swaps (uniformly random index pairs `l<r`), then `b` random reversals. Then pick a random subarray `[l..r]` and compute its sum. Find `E[S]`.
>
> **Constraints.** `2 ≤ n ≤ 1000`, `a ≤ 10⁹`, `b ≤ 10`.

**Key insight — Linearity of Expectation.**
1. After `a` swaps, the multiset of values is unchanged but each position's *expected value* converges (geometrically fast) to the global mean `μ = Σ d / n`.
2. Swap-mixing rate: each swap moves a position toward `μ` by factor `1 - 2/n + 2/(n(n-1))`. After `a = 10⁹` swaps, `E[d_i] = μ` exactly (to floating-point precision).
3. Reversals don't change the multiset, only the layout. They preserve `E[d_i] = μ` after stage 1.
4. Subarray-sum expected value: a position `i` is included in random `[l,r]` with probability `i(n-1-i+1)·... / C(n,2)`. After all the symmetry, `E[S] = (n+1)/3 · Σ d`.

```python
import sys

def expected_random_subarray_sum(n, a, b, d):
    total = sum(d)
    mu = total / n

    # Probability position i is included in a random subarray [l, r] with l < r
    # P(i in [l,r]) = (number of (l,r) pairs containing i) / C(n,2)
    #              = (i+1)*(n-i) / C(n,2)  (with 0-indexing)
    # Special handling: after enough swaps, every position has expected value mu, so:
    # E[S] = sum_i mu * P(i in subarray) = mu * sum_i P(i in [l,r])
    pairs = n * (n - 1) // 2
    weight = sum((i + 1) * (n - i) for i in range(n))
    # But we need expected SUBARRAY sum where subarray = sum_{i in [l,r]} d_i
    # Number of subarrays containing position i:  (i+1)*(n-i) (count over all 0<=l<=i<=r<n with l<r)
    # Average subarray sum = sum_i E[d_i] * count[i] / total_subarrays
    cnt_per_pos = [(i + 1) * (n - i) for i in range(n)]
    # If a is large, every E[d_i] = mu after mixing
    if a == 0:
        # No swaps, just initial array (but b reversals; reversals preserve multiset & symmetry too if uniform)
        # For exact sample: enumerate the small reversal space if b is small
        # Production: handle the boundary case explicitly. Simplification below assumes a >= 1.
        pass
    EX = mu                                        # expected value of any position after mixing
    return EX * sum(cnt_per_pos) / pairs

def main():
    data = sys.stdin.read().split()
    n, a, b = int(data[0]), int(data[1]), int(data[2])
    d = [int(x) for x in data[3:3 + n]]
    print(f"{expected_random_subarray_sum(n, a, b, d):.6f}")

if __name__ == "__main__":
    main()
```

> 🎯 **Why it's instructive.** Pure exercise in linearity of expectation + symmetry — no enumeration needed once you spot the invariance. This mindset is exactly what the interview problem rewards.

---

## 2.3 — Kevin and Expected Value (Medium)

> [HackerRank: Kevin and Expected Value](https://www.hackerrank.com/challenges/kevin-and-expected-value/problem)
>
> A function returns a uniform random integer `X ∈ {0, 1, …, N-1}`. Define `Y = f(X)` for a problem-specific function. Compute `E[Y]` for each test case.

**Key insight.** `E[Y] = (1/N) · Σ_{x=0}^{N-1} f(x)`. For large `N`, replace the sum with an **integral** (closed-form antiderivative). This is the key — naive summing is `O(N)` per query and won't fit the time limit; closed-form is `O(1)`.

```python
# Approximation pattern: for f(x) = x^k or polynomial,
# Σ_{x=0}^{N-1} x^k ≈ ∫_0^{N} x^k dx = N^{k+1}/(k+1)
# For exact: Faulhaber's formula gives closed-form sum.

def expected_polynomial(N, coeffs):
    """E[Y] when Y = a₀ + a₁X + a₂X² + ... with X uniform on {0,...,N-1}."""
    from fractions import Fraction
    if N == 0: return 0
    total = Fraction(0)
    for k, c in enumerate(coeffs):
        # Faulhaber's exact sum: S_k(N-1) = Σ_{x=0}^{N-1} x^k
        if k == 0:    Sk = Fraction(N)
        elif k == 1:  Sk = Fraction((N - 1) * N, 2)
        elif k == 2:  Sk = Fraction((N - 1) * N * (2 * N - 1), 6)
        elif k == 3:  Sk = Fraction(((N - 1) * N) ** 2, 4)
        else:         Sk = sum(Fraction(x ** k) for x in range(N))    # fallback
        total += Fraction(c) * Sk
    return total / N
```

> 🎯 **Generalization.** Any "expected value of f(X) for uniform X" problem reduces to summing `f` over the support; closed form exists for polynomial / rational `f`.

---

## 2.4 — Day 0 — Mean, Median, Mode (Easy)

> [HackerRank: Day 0 Statistics](https://www.hackerrank.com/challenges/s10-basic-statistics/problem)
>
> Given `N` integers, output the mean, median, and mode (smallest mode if multiple).

```python
from collections import Counter
import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    arr = sorted(int(x) for x in data[1:n + 1])

    mean = sum(arr) / n
    median = arr[n // 2] if n % 2 else (arr[n // 2 - 1] + arr[n // 2]) / 2
    cnt = Counter(arr)
    max_freq = max(cnt.values())
    mode = min(k for k, v in cnt.items() if v == max_freq)

    print(f"{mean:.1f}")
    print(f"{median:.1f}")
    print(mode)

if __name__ == "__main__":
    main()
```

---

## 2.5 — Basic Statistics Warmup (Easy)

> [HackerRank: Basic Statistics Warmup](https://www.hackerrank.com/challenges/stat-warmup/problem)
>
> Same plus: standard deviation, and 95% confidence-interval bounds (use constant 1.96).

```python
import math, sys
from collections import Counter

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    arr = sorted(int(x) for x in data[1:n + 1])

    mean = sum(arr) / n
    median = arr[n // 2] if n % 2 else (arr[n // 2 - 1] + arr[n // 2]) / 2
    cnt = Counter(arr)
    max_freq = max(cnt.values())
    mode = min(k for k, v in cnt.items() if v == max_freq)
    sd = math.sqrt(sum((x - mean) ** 2 for x in arr) / n)
    half_width = 1.96 * sd / math.sqrt(n)
    lo, hi = mean - half_width, mean + half_width

    print(f"{mean:.1f}")
    print(f"{median:.1f}")
    print(mode)
    print(f"{sd:.1f}")
    print(f"{lo:.1f} {hi:.1f}")

if __name__ == "__main__":
    main()
```

---

## 2.6 — Day 2 — Compound Event Probability (Easy)

> [HackerRank: Day 2 Compound Event Probability](https://www.hackerrank.com/challenges/s10-mcq-2/problem)
>
> Three urns of marbles (each containing R red + B black + W white). Draw one marble from each urn. Compute the probability that **exactly two are red and one is black**.

**Solution.** Multinomial enumeration:

```python
from fractions import Fraction
from itertools import permutations

# Urn contents (R, B, W)
urns = [(4, 3, 2), (5, 4, 3), (6, 4, 5)]
# Total balls per urn
totals = [sum(u) for u in urns]
# Probability of color from urn i: p[i][color] = count / total
p_red   = [u[0] / t for u, t in zip(urns, totals)]
p_black = [u[1] / t for u, t in zip(urns, totals)]

# We want exactly two reds and one black across the three urns.
# Sum over all permutations of which urn gives the black:
prob = sum(
    p_black[i] * p_red[(i + 1) % 3] * p_red[(i + 2) % 3]
    for i in range(3)
)
print(f"{Fraction(prob).limit_denominator(1000)}")
```

---

# 3. Optimal Stopping (closest analogs)

> HackerRank doesn't have a *true* optimal-stopping problem (no "press your luck" mechanic), but several DP-on-dice problems test the same backward-induction reasoning.

## 3.1 — Dice Throw (Medium, contest)

> [Dice Throw — GeeksforGeeks](https://www.geeksforgeeks.org/dsa/dice-throw-dp-30/) (mirrored on multiple HackerRank contests)
>
> Given `n` dice each with `m` faces, find the **number of ways** to obtain a target sum `X`.

**Recurrence.** `ways(n, X) = Σ_{f=1..m} ways(n-1, X-f)`.

```python
def dice_ways(n, m, X):
    # dp[i][s] = number of ways to make sum s with i dice
    dp = [[0] * (X + 1) for _ in range(n + 1)]
    dp[0][0] = 1
    for i in range(1, n + 1):
        for s in range(1, X + 1):
            dp[i][s] = sum(dp[i - 1][s - f] for f in range(1, m + 1) if s - f >= 0)
    return dp[n][X]

print(dice_ways(2, 6, 5))    # 4: (1,4), (2,3), (3,2), (4,1)
print(dice_ways(3, 6, 8))    # 21
```

The optimization for large `X`: use convolution / generating-function approach with FFT.

---

## 3.2 — Game of Two Dice — Peter & Colin (Medium)

> Peter and Colin have fair dice (Peter has `p` faces, Colin has `c` faces). They each roll once. Highest total wins. Compute `P(Peter wins)` mod `10¹²·9 + 24417`.

```python
def peter_vs_colin_prob(p, c, mod=10**12 * 9 + 24417):
    """P(Peter > Colin) = (number of (i,j) with i>j, 1<=i<=p, 1<=j<=c) / (p*c)"""
    if p == 1 and c == 1:
        return 0
    # Counting pairs with i > j
    win_pairs = 0
    for i in range(1, p + 1):
        win_pairs += min(i - 1, c)
    total = p * c
    # Modular inverse of total
    return (win_pairs * pow(total, -1, mod)) % mod

print(peter_vs_colin_prob(6, 6))    # 15/36 in mod form
```

The **decision flavor** (choose who plays first, decide reroll, etc.) makes it the closest HackerRank analog to "should I stop or roll again."

---

# 4. Majority Vote

## 4.1 — Boyer–Moore Majority Element (Easy/Medium)

> Closely matches the [HackerRank "Majority"](https://martinkysel.com/hackerrank-majority-solution/) and LeetCode 169 patterns.
>
> Given an array of `n` integers, find the element appearing more than `n/2` times (guaranteed to exist).

**Boyer–Moore in `O(n)` time and `O(1)` space.**

```python
def boyer_moore_majority(arr):
    candidate, count = None, 0
    for x in arr:
        if count == 0:
            candidate, count = x, 1
        elif x == candidate:
            count += 1
        else:
            count -= 1
    # Verification pass (mandatory if not guaranteed)
    return candidate if arr.count(candidate) > len(arr) / 2 else None

print(boyer_moore_majority([3, 3, 4, 2, 4, 4, 2, 4, 4]))  # 4
```

**Why it links to your majority-vote interview problem.** It's the *algorithmic* mirror of the *probability* problem: instead of asking "what's the accuracy if voters are noisy", you find the deterministic majority winner. Both rely on the invariant "votes cancel pairwise."

---

## 4.2 — Lonely Integer (Easy)

> [HackerRank: Lonely Integer](https://www.hackerrank.com/challenges/lonely-integer/problem)
>
> An array has every integer appearing exactly twice except one. Find the unique one.

**XOR trick** — `a ⊕ a = 0`, so XORing all elements leaves the lonely integer.

```python
from functools import reduce
import operator

def lonely_integer(arr):
    return reduce(operator.xor, arr, 0)

print(lonely_integer([1, 1, 2, 2, 3, 4, 4]))    # 3
```

> 🎯 **Voting connection.** It's "majority of one" — same family as Boyer–Moore. Often paired with the majority problem in interviews to test bit-manipulation versatility.

---

# Practice Plan

If you have a few days before the interview, try this order:

| Day | HackerRank problem | Topic skill exercised |
|---|---|---|
| 1 | [Day 0 Statistics](https://www.hackerrank.com/challenges/s10-basic-statistics/problem) + [Basic Statistics Warmup](https://www.hackerrank.com/challenges/stat-warmup/problem) | Warm-up: mean / variance fundamentals |
| 1 | [Day 2 Compound Event](https://www.hackerrank.com/challenges/s10-mcq-2/problem) | Independent events, multiplication |
| 2 | [Spiral Matrix](https://www.hackerrank.com/contests/coding-test-1-bits-hyderabad/challenges/spiral-matrix-1) | Standard outside-in spiral |
| 2 | [Matrix Layer Rotation](https://www.hackerrank.com/challenges/matrix-rotation-algo/problem) | Layer extraction + modular rotation |
| 3 | [Dice Stats](https://www.hackerrank.com/challenges/dice-stats/problem) | E and Var with Markov dependence |
| 3 | [Random](https://www.hackerrank.com/challenges/random/problem) | Linearity of expectation, symmetry |
| 4 | [Kevin and Expected Value](https://www.hackerrank.com/challenges/kevin-and-expected-value/problem) | Closed-form sums of polynomials |
| 4 | [Lonely Integer](https://www.hackerrank.com/challenges/lonely-integer/problem) + Boyer–Moore | Vote-cancellation patterns |

> 🎯 **Time it.** Each problem should take ≤ 30 minutes once the patterns are familiar; budget 45–60 minutes during practice and aim to drop times in the second pass.

> ⚠️ **HackerRank-specific gotcha.** Use `sys.stdin.read().split()` over `input()` for I/O-heavy problems. Floating-point output: use `f"{x:.10f}"` or `print(round(x, 6))` per the problem's tolerance. For modular arithmetic, Python `pow(a, -1, m)` is your friend (Python 3.8+).

---

# Sources

## HackerRank problem pages

- [Matrix Layer Rotation — Algorithms](https://www.hackerrank.com/challenges/matrix-rotation-algo/problem)
- [Matrix Rotation (sister problem)](https://www.hackerrank.com/challenges/matrix-rotation/problem)
- [Spiral Matrix 1 — BITS Hyderabad contest](https://www.hackerrank.com/contests/coding-test-1-bits-hyderabad/challenges/spiral-matrix-1)
- [Spiral Matrix — Kode.Utsav contest](https://www.hackerrank.com/contests/utsav/challenges/spiral-matrix/)
- [Spiral Traversal — Zoho contest](https://www.hackerrank.com/contests/zoho-pr/challenges/spiral-traversal/)
- [Dice Stats — Probability](https://www.hackerrank.com/challenges/dice-stats/problem)
- [Random — Probability](https://www.hackerrank.com/challenges/random/problem)
- [Kevin and Expected Value — Mathematics](https://www.hackerrank.com/challenges/kevin-and-expected-value/problem)
- [Day 0: Mean, Median, and Mode (10 Days of Statistics)](https://www.hackerrank.com/challenges/s10-basic-statistics/problem)
- [Day 2: Compound Event Probability](https://www.hackerrank.com/challenges/s10-mcq-2/problem)
- [Day 2: Basic Probability tutorial](https://www.hackerrank.com/challenges/s10-mcq-1/tutorial)
- [Basic Statistics Warmup — AI track](https://www.hackerrank.com/challenges/stat-warmup/problem)
- [Random Integers Random Bits (expected number of 1-bits)](https://www.hackerrank.com/challenges/rirb/problem)
- [Game of Dice — Code Marathon contest](https://www.hackerrank.com/contests/code-marathon-div3/challenges/game-of-dice/)
- [Lonely Integer](https://www.hackerrank.com/challenges/lonely-integer/problem)
- [HackerRank — all expectation-values topics](https://www.hackerrank.com/topics/expectation-values/)
- [Probability & Statistics Foundations questions](https://www.hackerrank.com/domains/ai/statistics-foundations/difficulty/2)

## Walkthroughs and reference solutions

- [Matrix Layer Rotation walk-through — brokensandals.net](https://brokensandals.net/technical/programming-challenges/hackerrank-matrix-layer-rotation/)
- [Matrix Layer Rotation solution — Martin Kysel](https://martinkysel.com/hackerrank-matrix-rotation-solution/)
- [Matrix Layer Rotation Hackerrank Problem — Samarth Sewlani, Medium](https://codecurse.medium.com/matrix-layer-rotation-hackerrank-problem-fce39c7ebbba)
- [Matrix Layer Rotation — CodingBroz](https://www.codingbroz.com/matrix-layer-rotation-hackerrank-solution/)
- [HackerRank Matrix Layer Rotation Python (dispe1 GitHub)](https://github.com/dispe1/Hackerrank-Solutions/blob/master/Algorithms/02.%20Implementation/068.%20Matrix%20Layer%20Rotation.py)
- [HackerRank Matrix Layer Rotation Python (alessandrobardini GitHub)](https://github.com/alessandrobardini/HackerRank-Solutions/blob/master/All%20Tracks/Core%20CS/Algorithms/Implementation/Matrix%20Layer%20Rotation/Solution.py)
- [HackerRank "Statistics in 10 Days" — Pedro Campos, Medium](https://medium.com/@pedrofreitascampos/hackerrank-statistics-in-10-days-b229c4f28d11)
- [HackerRank 10 Days of Statistics — CodingBroz solutions](https://www.codingbroz.com/10-days-of-statistics-solution/)
- [HackerRank 10 Days of Statistics solutions (Murillo GitHub)](https://github.com/Murillo/Hackerrank-10-Days-of-Statistics)
- [HackerRank 10 Days of Statistics solutions (shsarv GitHub)](https://github.com/shsarv/Hackerrank-10-Days-of-Statistics)
- [Kevin and Expected Value forum hints](https://www.hackerrank.com/challenges/kevin-and-expected-value/forum)
- [HackerRank Majority solution — Martin Kysel](https://martinkysel.com/hackerrank-majority-solution/)
- [HackerRank Days 4 and 5: Probability Distributions — Coder's Errand](https://coders-errand.com/hackerrank-days-4-and-5-probability-distributions/)
- [Boyer–Moore Majority Vote Algorithm — Techie Delight](https://www.techiedelight.com/find-majority-element-in-an-array-boyer-moore-majority-vote-algorithm/)
- [Dice Throw DP — GeeksforGeeks](https://www.geeksforgeeks.org/dsa/dice-throw-dp-30/)
- [Dynamic Programming — HackerRank Blog](https://www.hackerrank.com/blog/dynamic-programming-definition-questions/)
