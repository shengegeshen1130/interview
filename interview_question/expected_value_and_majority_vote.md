# Expected Value/Variance & Majority-Vote Classifier — Interview Prep Pack

Curated study guide for two probability-flavored interview topics, with similar problems sourced from real interview repositories (Jane Street, FAANG, quant funds) and ML references. Each problem includes the key insight and a runnable Python solution where computation is non-trivial.

---

## Table of Contents

- [Topic 1: Expected Value & Variance](#topic-1--expected-value--variance)
  - [1.0 Original problem](#10--original-problem)
  - [1.1 Fair 6-sided die](#11--fair-6-sided-die)
  - [1.2 Sum of two dice](#12--sum-of-two-dice)
  - [1.3 Max and min of two dice (Jane Street)](#13--max-and-min-of-two-dice-jane-street)
  - [1.4 Bernoulli / Binomial — derive E and Var](#14--bernoulli--binomial--derive-e-and-var)
  - [1.5 Geometric distribution — waiting time](#15--geometric-distribution--waiting-time)
  - [1.6 Biased coin: Bayesian flip (Facebook)](#16--biased-coin-bayesian-flip-facebook)
  - [1.7 Compound sum — Law of Total Variance](#17--compound-sum--law-of-total-variance)
  - [1.8 Mixture distribution](#18--mixture-distribution)
  - [1.9 ⭐ Coupon collector — E and Var](#19--coupon-collector--e-and-var)
  - [1.10 ⭐ Random walk hitting time / Gambler's Ruin](#110--random-walk-hitting-time--gamblers-ruin)
  - [1.11 ⭐ Correlation triangle inequality (Jane Street)](#111--correlation-triangle-inequality-jane-street)
  - [1.12 ⭐ St. Petersburg paradox](#112--st-petersburg-paradox)
  - [1.13 ⭐ Wald's identity & expected sum at stopping time](#113--walds-identity--expected-sum-at-stopping-time)
- [Topic 2: Majority-Vote Classifier](#topic-2--majority-vote-classifier)
  - [2.0 Original problem](#20--original-problem)
  - [2.1 General N classifiers @ p, majority vote](#21--general-n-classifiers--p-majority-vote)
  - [2.2 5 classifiers at 70% accuracy](#22--5-classifiers-at-70-accuracy)
  - [2.3 Heterogeneous classifiers (different accuracies)](#23--heterogeneous-classifiers-different-accuracies)
  - [2.4 Weighted majority vote](#24--weighted-majority-vote)
  - [2.5 Condorcet's Jury Theorem (asymptotic)](#25--condorcets-jury-theorem-asymptotic)
  - [2.6 Correlated classifiers — independence breaks](#26--correlated-classifiers--independence-breaks)
  - [2.7 Hard vote vs soft vote](#27--hard-vote-vs-soft-vote)
  - [2.8 Even number of voters (tie-breaking)](#28--even-number-of-voters-tie-breaking)
  - [2.9 Below-50% classifiers (when ensembling hurts)](#29--below-50-classifiers-when-ensembling-hurts)
  - [2.10 ⭐ AdaBoost training-error bound](#210--adaboost-training-error-bound)
  - [2.11 ⭐ Bias–variance decomposition for bagging](#211--biasvariance-decomposition-for-bagging)
  - [2.12 ⭐ Multi-class plurality voting (Borda vs plurality)](#212--multi-class-plurality-voting-borda-vs-plurality)
  - [2.13 ⭐ Stacking with logistic-regression meta-learner](#213--stacking-with-logistic-regression-meta-learner)
  - [2.14 ⭐ Hoeffding bound on ensemble error](#214--hoeffding-bound-on-ensemble-error)
- [Cheat Sheet](#cheat-sheet)
- [Sources](#sources)

---

# Topic 1 — Expected Value & Variance

> **Universal template for discrete X**
>
> 1. `E[X] = Σ x · P(X=x)`
> 2. `E[X²] = Σ x² · P(X=x)`
> 3. `Var(X) = E[X²] − (E[X])²`  *(always preferred over the definition `E[(X−μ)²]` for hand calculation)*
>
> For sums and functions: `Var(aX + b) = a²·Var(X)`, `Var(X + Y) = Var(X) + Var(Y)` *iff* X ⊥ Y.

---

## 1.0 — Original problem

> X is discrete with `P(X=0)=0.5`, `P(X=1)=0.4`, `P(X=6)=0.1`. Find E[X] and Var(X).

```python
xs   = [0, 1, 6]
ps   = [0.5, 0.4, 0.1]

EX   = sum(x * p for x, p in zip(xs, ps))           # 0 + 0.4 + 0.6 = 1.0
EX2  = sum(x * x * p for x, p in zip(xs, ps))       # 0 + 0.4 + 3.6 = 4.0
varX = EX2 - EX ** 2                                # 4.0 - 1.0 = 3.0

# E[X] = 1.0,  Var(X) = 3.0,  SD = sqrt(3) ≈ 1.732
```

> 🎯 **Interview tip.** Verbally: *"Expected value is just the probability-weighted sum. For variance I'll use `E[X²] − (E[X])²` because it's one less subtraction."* Then narrate the arithmetic.

---

## 1.1 — Fair 6-sided die

> Compute E[X] and Var(X) for X = result of rolling a fair 6-sided die.

```python
EX   = sum(k for k in range(1, 7)) / 6              # 21/6 = 3.5
EX2  = sum(k * k for k in range(1, 7)) / 6          # 91/6
varX = EX2 - EX ** 2                                # 91/6 - 49/4 = 35/12 ≈ 2.917
```

| Statistic | Value | Closed form |
|---|---|---|
| E[X] | 3.5 | `(n+1)/2` for `1..n` |
| Var(X) | 35/12 ≈ 2.917 | `(n²−1)/12` |

---

## 1.2 — Sum of two dice

> S = X₁ + X₂, both fair dice. Find E[S] and Var(S).

By linearity and independence:
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

## 1.3 — Max and min of two dice (Jane Street)

> M = max(X₁, X₂), m = min(X₁, X₂) where X₁, X₂ are i.i.d. fair dice.

**Trick.** Use `P(M ≤ k) = (k/6)²` so `P(M=k) = ((k)² − (k−1)²)/36 = (2k−1)/36`. Symmetric for min: `P(m ≥ k) = ((7−k)/6)²`.

```python
EM = sum(k * (2 * k - 1) / 36 for k in range(1, 7))           # 161/36 ≈ 4.472
Em = sum(k * (2 * (7 - k) - 1) / 36 for k in range(1, 7))     # 91/36  ≈ 2.528

# Sanity check: M + m = X1 + X2 always, so E[M] + E[m] = 7. ✓
assert abs(EM + Em - 7) < 1e-9
```

`Var(M) = E[M²] − E[M]² = 791/36 − (161/36)² ≈ 1.97`.

> 🎯 **Follow-up.** *Why is `E[M] + E[m] = E[X₁] + E[X₂]`?* Because `max + min = a + b` pointwise. Linearity of expectation does the rest.

---

## 1.4 — Bernoulli / Binomial — derive E and Var

> X ~ Bernoulli(p). Then Y = X₁ + ... + Xₙ ~ Binomial(n, p). Find E and Var of each.

**Bernoulli:** `E[X] = p`, `E[X²] = p` (since X² = X), so `Var(X) = p − p² = p(1−p)`.

**Binomial:** by linearity and independence:
- `E[Y] = np`
- `Var(Y) = np(1−p)`

```python
def binom_moments(n, p):
    return n * p, n * p * (1 - p)

# Most asked at FAANG: "X is # heads in 10 flips of a fair coin. E? Var?"
EY, vY = binom_moments(10, 0.5)                                # 5.0, 2.5
```

---

## 1.5 — Geometric distribution — waiting time

> "How many flips on average until the first head with a coin that shows heads with probability `p`?"

X ~ Geometric(p) (counting trials including the success):
- `E[X] = 1/p`
- `Var(X) = (1−p)/p²`

```python
def geometric_moments(p):
    return 1 / p, (1 - p) / p ** 2

# Fair coin -> E[X]=2, Var(X)=2
```

**Slick derivation for E.** Condition on the first flip:
`E[X] = p·1 + (1−p)·(1 + E[X])  ⇒  E[X] = 1/p`.

---

## 1.6 — Biased coin: Bayesian flip (Facebook)

> "There are two coins: fair (50/50) and biased (always tails). You pick one uniformly at random and flip 5 times — all tails. What's the probability you picked the biased coin?"

Bayes:

`P(biased | 5T) = P(5T | biased)·0.5 / [P(5T | biased)·0.5 + P(5T | fair)·0.5] = 1 / (1 + (1/2)⁵) = 32/33 ≈ 0.970`.

```python
p_biased_prior = 0.5
p_fair_prior   = 0.5
p_5T_biased    = 1.0
p_5T_fair      = 0.5 ** 5

posterior = p_5T_biased * p_biased_prior / (p_5T_biased * p_biased_prior +
                                             p_5T_fair * p_fair_prior)
# 0.96970 ≈ 32/33
```

> 🎯 **Generalize:** with `k` tails in a row, posterior = `2ᵏ / (2ᵏ + 1)`.

---

## 1.7 — Compound sum — Law of Total Variance

> A store has `N ~ Poisson(λ)` customers per day. Each customer spends `Xᵢ` i.i.d. with mean `μ`, variance `σ²`. Find E[S] and Var(S) where `S = Σᵢ Xᵢ`.

**Eve's law:** `Var(S) = E[Var(S|N)] + Var(E[S|N])`.

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

This is the canonical "tower / Eve's law" interview problem.

---

## 1.8 — Mixture distribution

> A coin shows heads w.p. `p` (drawn from `Beta(α,β)`). Find unconditional E and Var of a single flip.

`E[X] = E[E[X|p]] = E[p] = α/(α+β)`. For variance, use the mixture variance formula.

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

> Trick this often hides: even though `p` is itself random, a single binary X still has variance `μ(1−μ)`. The variance of `p` only kicks in when you observe **multiple** trials with the same coin.

---

# ⭐ Topic 1 — Harder Variants

> Step up from one-shot E/Var calculations to **sums of dependent random variables** (coupon collector), **stochastic processes** (random walks, Wald's identity), **constraints on the correlation matrix** (Jane Street's classic), and **paradoxes that probe utility theory**.

---

## 1.9 — Coupon collector — E and Var

> "There are `n` distinct coupons. Each draw, you get a uniformly random one (with replacement). What's E and Var of the number of draws to collect them all?"

**Decompose** into stage waiting times. Once you have `i−1` distinct coupons, the prob of a new one is `(n − i + 1)/n`, so `Tᵢ ~ Geometric(p_i)` with `p_i = (n−i+1)/n`. The Tᵢ are independent.

- `E[T] = Σᵢ 1/pᵢ = n · Hₙ ≈ n·(ln n + γ)` (γ ≈ 0.5772 — Euler-Mascheroni)
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

> 🎯 **Quant follow-up.** *"How likely is it to take more than `2·n·ln n` draws?"* Use Markov: `P(T > 2·E[T]) ≤ 1/2`. Tighter bound via Chebyshev: `P(|T−E[T]| > c·SD) ≤ 1/c²`.

---

## 1.10 — Random walk hitting time / Gambler's Ruin

> "Symmetric simple random walk on `Z`, starting at position `i`, with absorbing barriers at `0` and `N` (`0 < i < N`). What's the expected number of steps until absorption?"

**Recurrence.** Let `h(i) = E[T | start at i]`. `h(0) = h(N) = 0`, and for `0 < i < N`:

`h(i) = 1 + 0.5 · h(i−1) + 0.5 · h(i+1)`

**Closed form:** `h(i) = i · (N − i)`.

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

> 🎯 **Why it's hard.** It's the canonical "set up the recurrence, solve it" question. Variance is harder: `Var[T] = (1/3) · i · (N−i) · (N² + i(N−i) − 2)` for the symmetric case.

---

## 1.11 — Correlation triangle inequality (Jane Street)

> "If `Corr(X, Y) = 0.9` and `Corr(Y, Z) = 0.8`, what are the **min** and **max** possible values of `Corr(X, Z)`?"

**Insight.** The 3×3 correlation matrix must be positive semi-definite. Let `r_xy=0.9`, `r_yz=0.8`, `r_xz=ρ`. PSD requires `det ≥ 0`:

`det = 1 − r_xy² − r_yz² − ρ² + 2·r_xy·r_yz·ρ ≥ 0`

Solving the quadratic in ρ:

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

> 🎯 **Geometric intuition.** Treat unit-variance random variables as unit vectors; correlation = cosine of angle between them. The triangle of angles must satisfy the triangle inequality `|θ_xy − θ_yz| ≤ θ_xz ≤ θ_xy + θ_yz`. That's exactly what the formula above expresses.

---

## 1.12 — St. Petersburg paradox

> "A fair coin is flipped until the first head. If the first head appears on flip `k`, you win `2^k` dollars. What's the expected payout? How much would you pay to play?"

`E[X] = Σ_{k=1}^{∞} (1/2)^k · 2^k = Σ 1 = ∞`.

```python
def st_petersburg_truncated_expectation(max_k: int) -> float:
    return float(max_k)            # diverges; truncating to max_k flips gives EV = max_k

# After max_k=30 (≈ casino's bankroll), EV is just $30. So nobody pays $1M.
```

**Resolution via utility.** With log utility `u(x) = log x`, expected utility = `Σ 2^{-k}·log(2^k) = log 2 · Σ k·2^{-k} = 2·log 2`. So the certainty equivalent under log utility is `e^{2·log 2} = 4` — *i.e.* you'd pay around $4.

> 🎯 **Why it shows up.** Behavioral-finance / decision-theory questions at quant funds. Knowing the resolution (utility theory, bounded payouts, Cramér transform) signals depth.

---

## 1.13 — Wald's identity & expected sum at stopping time

> "i.i.d. random variables `X₁, X₂, …` with mean `μ`. `N` is a stopping time with `E[N] < ∞`. Prove `E[Σ_{i=1}^{N} Xᵢ] = μ · E[N]`."

This is **Wald's identity**. The key subtlety: `N` may depend on the X's, but as a *stopping time* — its decision at step `n` depends only on `X₁..Xₙ`, not on the future.

**Application.** "I keep flipping a fair coin until I see 3 heads in a row. What's the expected number of flips?"

State diagram: "0 in a row" → "1 in a row" → "2 in a row" → "3 in a row (absorbing)".

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

The non-intuitive insight: pattern `'HHH'` takes **longer** than `'HTH'` despite both having probability `1/8` — because `'HHH'` overlaps with itself heavily.

> 🎯 **Interview gold.** *"Two patterns with the same probability have different expected waiting times!"* That's the soundbite — if you can derive it from autocorrelation, you've nailed the problem.

---

# Topic 2 — Majority-Vote Classifier

> **Universal formula** — `n` independent binary voters, each correct with probability `p`. Majority vote (n odd) is correct iff at least `⌈n/2⌉` of them are.
>
> `P(majority correct) = Σ_{k=⌈n/2⌉}^{n} C(n,k) · pᵏ · (1−p)^(n−k)`

---

## 2.0 — Original problem

> Three independent binary classifiers each at 80% accuracy. Accuracy of majority vote?

```python
from math import comb
p, n = 0.8, 3
acc  = sum(comb(n, k) * p**k * (1 - p)**(n - k) for k in range(2, n + 1))
# 3 * 0.64 * 0.2 + 1 * 0.512 = 0.384 + 0.512 = 0.896
```

**Result: 89.6%.** Ensemble lifted accuracy by ~10 percentage points over each base learner.

> 🎯 **Interview tip.** State the assumption explicitly: *"This is the binomial probability of getting ≥2 heads with `p=0.8`, **assuming independence**. With correlated classifiers the ensemble gains shrink."*

---

## 2.1 — General N classifiers @ p, majority vote

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

`P` increases monotonically toward 1 as `n → ∞`. (This is Condorcet's Jury Theorem — see 2.5.)

---

## 2.2 — 5 classifiers at 70% accuracy

> "If you ensemble 3 classifiers each at 70% accuracy, what's the majority accuracy? What about 5?"

```python
# n=3, p=0.7
print(majority_vote_accuracy(3, 0.7))   # 0.784
# n=5, p=0.7
print(majority_vote_accuracy(5, 0.7))   # 0.83692
```

Going from 3 → 5 adds ~5 pp; doubling members again to 11 yields ~92.2%. **Diminishing returns** — the marginal gain shrinks like `1/√n`.

---

## 2.3 — Heterogeneous classifiers (different accuracies)

> 3 independent classifiers with accuracies `p₁ = 0.9`, `p₂ = 0.7`, `p₃ = 0.6`. Accuracy of majority vote?

Enumerate the `2³ = 8` correctness patterns:

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

**Key lesson.** Heterogeneous ensembles can do worse than the best single classifier if the others are weak — try `[0.95, 0.5, 0.5]` and you get 0.7250, *worse than* the 0.95 model alone.

---

## 2.4 — Weighted majority vote

> Three classifiers with accuracies `p = [0.9, 0.7, 0.6]` and weights `w = [3, 1, 1]`. Accuracy if we use weighted vote on the binary label (the class with the larger weighted-count wins; tie → arbitrary)?

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

With a heavy weight on the strongest model the ensemble accuracy ~ tracks that classifier; the optimal log-likelihood-ratio weights (Berend & Sapir) are `wᵢ = log(pᵢ/(1−pᵢ))`.

```python
import math
def lr_weights(ps):
    return [math.log(p / (1 - p)) for p in ps]
# [2.197, 0.847, 0.405]   -- the strong model gets ~5.4× the weight of the weak one
```

---

## 2.5 — Condorcet's Jury Theorem (asymptotic)

> "If `p > 0.5` and classifiers are independent, what happens to the majority-vote accuracy as `n → ∞`?"

**Theorem (Condorcet, 1785).** With `n` odd and i.i.d. correct-prob `p > 0.5`:

`P(majority correct) → 1` as `n → ∞`.
If `p < 0.5`, the limit is 0. If `p = 0.5`, it's 0.5.

This is the theoretical justification for ensembling and bagging. Quick CLT-based approximation:

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

## 2.6 — Correlated classifiers — independence breaks

> "Now suppose all three 80% classifiers are perfectly correlated (they always agree). Majority accuracy?"

If correlation = 1, the ensemble *is* one classifier: **80%**. The independence assumption was doing all the work. In practice corr is between 0 and 1, and the ensemble accuracy is bounded above by the independent case.

A quick simulation when accuracies marginal but predictions share a common signal:

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

> 🎯 **Punchline.** Diversifying ensembles (different architectures, different feature subsets, bagging with different bootstrap samples) is what makes the math work.

---

## 2.7 — Hard vote vs soft vote

> "If each classifier outputs a *probability* rather than a label, would you still vote on labels? When would soft voting beat hard voting?"

- **Hard:** majority of class predictions. Loses confidence info.
- **Soft:** average the per-class probabilities, take argmax. Better when classifiers are well-calibrated.

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

Soft voting requires calibrated probabilities; otherwise an over-confident weak model dominates.

---

## 2.8 — Even number of voters (tie-breaking)

> "What if we use 4 classifiers at 80% accuracy?"

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

Surprising result: **4 classifiers with tie-break = 3 classifiers**, both at 0.896 for `p=0.8`. Hence interview convention: always use *odd* counts.

---

## 2.9 — Below-50% classifiers (when ensembling hurts)

> "Three 'classifiers' each at 30% accuracy on a binary task. Majority-vote accuracy?"

```python
majority_vote_accuracy(3, 0.3)  # 0.216
```

Below 50%, **ensembling makes things worse** (Condorcet's curse). However, you can flip every classifier's output and now have three 70% classifiers — majority of those gives 0.784. The two are dual: `acc(majority of n @ p) + acc(majority of n @ 1−p) = 1`.

---

# ⭐ Topic 2 — Harder Variants

> Step up from "binomial probability of being correct" to **adaptive weighting** (AdaBoost), **bias–variance accounting** for ensembles (bagging), **multi-class** voting, **stacking** with a meta-learner, and **PAC-style bounds** (Hoeffding).

---

## 2.10 — AdaBoost training-error bound

> "Each round, AdaBoost trains a weak learner with weighted error `εₜ < 0.5`, sets `αₜ = ½ ln((1−εₜ)/εₜ)`, and re-weights samples. Show that after T rounds, the training error is at most `Π_t 2·√(εₜ(1−εₜ))`."

**Key fact (Schapire & Freund).** Define `γₜ = 0.5 − εₜ` (the "edge" of round `t`). Then:

`training_error ≤ exp(−2 · Σ_t γₜ²)`

So if every weak learner is even slightly better than chance (`γ ≥ γ₀ > 0`), training error decays **exponentially** in T.

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

> 🎯 **Interview soundbite.** *"AdaBoost is a coordinate-descent on exponential loss. The α formula falls out of one-dimensional minimization."*

---

## 2.11 — Bias–variance decomposition for bagging

> "Why does bagging reduce variance but not bias? Make the math precise."

**Setup.** A learner produces estimator `f̂(x; D)` from training set `D`. Bagging averages `M` bootstrap-trained estimators: `f̄(x) = (1/M) Σₘ f̂(x; Dₘ)`.

For the **squared-error case**, decompose:

`E[(f̄(x) − y)²] = bias²(f̄) + Var(f̄) + σ²`

- **Bias.** `E[f̄(x)] ≈ E[f̂(x; D)]` since bootstrap replicas are roughly i.d. → **bias unchanged**.
- **Variance.** If `Var(f̂) = σ_f²` and pairwise correlation `ρ` between bagged learners:

  `Var(f̄) = ρ·σ_f² + (1−ρ)/M · σ_f² → ρ·σ_f²` as `M → ∞`.

So bagging eliminates the **independent** part of variance but leaves the correlated residual `ρ·σ_f²`. **Random Forests** decorrelate further (smaller ρ) by random feature subsetting.

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

> 🎯 **Interview gold.** State the formula for `Var(f̄)` and explain why **decorrelation** (RF feature bagging, model diversity) is the lever, not just "more trees".

---

## 2.12 — Multi-class plurality voting (Borda vs plurality)

> "K classifiers, each predicts one of `C` classes. With `C > 2`, what voting rule maximizes accuracy?"

For `C = 2`, plurality = majority. For `C > 2`, **plurality** ("class with most votes wins") can fail when no class wins a majority. **Borda count** uses ranked predictions: each classifier ranks classes 1..C, and class `c` accumulates `C − rank_i(c)` points across classifiers. Class with highest total wins.

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

> 🎯 **Where it matters.** Soft voting (averaging predicted probabilities) is essentially a continuous-Borda. For calibrated classifiers it strictly dominates plurality.

---

## 2.13 — Stacking with logistic-regression meta-learner

> "Instead of voting, train a *meta-learner* on the base classifiers' outputs. How does this work and why is it better?"

**Stacking.**
1. Train base classifiers `h₁, ..., hₖ` on data using cross-validation to generate **out-of-fold predictions** `ẑᵢⱼ = hⱼ(xᵢ)` (avoids leakage).
2. Train a meta-learner `g` (often logistic regression) on `(ẑᵢ, yᵢ)` pairs.
3. At inference: predict with each `hⱼ`, then pass to `g`.

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

**Why it's better than voting.** The meta-learner learns *which* base classifier to trust under *which* conditions, instead of fixed weights. Logistic-regression meta-learner gives interpretable weights that satisfy the optimal log-likelihood-ratio form (Section 2.4).

---

## 2.14 — Hoeffding bound on ensemble error

> "I have `n` independent classifiers each at accuracy `p > 0.5`. How many do I need so that majority-vote error is below ε with probability ≥ 1−δ?"

**Hoeffding's inequality** (sample-mean concentration). For i.i.d. {0,1} indicators `Yᵢ` (correctness of classifier i, mean `p`):

`P(Ȳₙ ≤ p − γ) ≤ exp(−2 n γ²)`

Majority is wrong iff `Ȳₙ ≤ 0.5`, i.e. with deviation `γ = p − 0.5`. So:

`P(majority wrong) ≤ exp(−2n(p − 0.5)²)`

Solve for `n`: `n ≥ ln(1/δ) / (2·(p − 0.5)²)`.

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

> 🎯 **Interview tip.** *"Hoeffding gives you a non-asymptotic, distribution-free bound. The CLT (Section 2.5) gives a tighter approximation but only in the limit. For finite-`n` guarantees, Hoeffding is the tool."*

---

# Cheat Sheet

## Expected Value & Variance — go-to formulas

| Distribution | E[X] | Var(X) |
|---|---|---|
| Bernoulli(p) | p | p(1−p) |
| Binomial(n, p) | np | np(1−p) |
| Geometric(p) (trials incl. success) | 1/p | (1−p)/p² |
| Uniform on {1..n} | (n+1)/2 | (n²−1)/12 |
| Poisson(λ) | λ | λ |
| Uniform[a, b] | (a+b)/2 | (b−a)²/12 |
| Exponential(λ) | 1/λ | 1/λ² |
| Normal(μ, σ²) | μ | σ² |

| Identity | Use when |
|---|---|
| `Var(X) = E[X²] − (E[X])²` | Always faster than the definition |
| `Var(aX + b) = a²Var(X)` | Linear transforms |
| `Var(X+Y) = Var(X) + Var(Y) + 2Cov(X,Y)` | Joint moments |
| `E[E[Y\|X]] = E[Y]` | Tower / total expectation |
| `Var(Y) = E[Var(Y\|X)] + Var(E[Y\|X])` | Eve's law / total variance |
| `E[T] = n · H_n`, `Var(T) ≈ π²n²/6` | Coupon collector |
| `E[T_i] = i(N − i)`, symmetric walk | Gambler's ruin hitting time |
| `\|ρ_xz − ρ_xy ρ_yz\| ≤ √((1−ρ_xy²)(1−ρ_yz²))` | Correlation triangle |
| `E[Σ Xᵢ] = μ · E[N]` | Wald's identity for stopping times |

## Majority-Vote Classifier — pattern recognition

| Setting | Formula / Insight |
|---|---|
| n odd, i.i.d. p | `Σ_{k=⌈n/2⌉..n} C(n,k) p^k (1-p)^(n-k)` |
| Heterogeneous accuracies | enumerate `2^n` correctness patterns |
| Weighted vote | sum weights of correct voters; compare to total/2 |
| Optimal weights | `w_i = log(p_i / (1 − p_i))` (log-likelihood ratio) |
| Asymptotic, p > 0.5, indep | accuracy → 1 (Condorcet) |
| Correlated voters | accuracy bounded between p and the indep. case |
| Soft voting | average probabilities, argmax — needs calibration |
| Even n | strict majority lower; convention: half-credit on ties |
| p < 0.5 | ensembling **decreases** accuracy → 0 |

> **Always state your assumptions** in the interview: independence, calibration, equal weights, odd `n`. Half the follow-ups probe what happens when one assumption breaks.

---

# Sources

## Topic 1 — Expected Value & Variance

- [40 Probability & Statistics Data Science Interview Questions Asked By FAANG & Wall Street — NickSingh](https://www.nicksingh.com/posts/40-probability-statistics-data-science-interview-questions-asked-by-fang-wall-street)
- [Probability (Data Science) Interview Questions — InterviewBit](https://www.interviewbit.com/probability-interview-questions/)
- [25 Probability and Statistics questions to ace data-science interviews — Analytics Vidhya](https://www.analyticsvidhya.com/blog/2021/04/25-probability-and-statistics-questions-to-ace-your-data-science-interviews/)
- [30 Probability and Statistics Interview Questions for Data Scientists — StrataScratch](https://www.stratascratch.com/blog/30-probability-and-statistics-interview-questions-for-data-scientists)
- [Top 25 Random Variables Interview Questions — InterviewPrep](https://interviewprep.org/random-variables-interview-questions/)
- [Expected Value of Discrete Random Variables — LibreTexts](https://stats.libretexts.org/Courses/Saint_Mary's_College_Notre_Dame/DSCI_500B_Essential_Probability_Theory_for_Data_Science_(Kuter)/03:_Discrete_Random_Variables/3.04:_Expected_Value_of_Discrete_Random_Variables)
- [Variance of Discrete Random Variables — LibreTexts (Grinstead & Snell)](https://stats.libretexts.org/Bookshelves/Probability_Theory/Introductory_Probability_(Grinstead_and_Snell)/06:_Expected_Value_and_Variance/6.02:_Variance_of_Discrete_Random_Variables)
- [Expected Value & Variance of a Discrete PDF — LibreTexts (Geraghty)](https://stats.libretexts.org/Bookshelves/Introductory_Statistics/Inferential_Statistics_and_Probability_-_A_Holistic_Approach_(Geraghty)/06:_Discrete_Random_Variables/6.04:_Expected_Value_and_Variance_of_a_Discrete_Probability_Distribution_Function)
- [Jane Street Interview — expected value of max of two dice (Glassdoor)](https://www.glassdoor.com/Interview/What-is-the-expected-value-of-the-max-of-two-dice-QTN_133823.htm)
- [Jane Street Interview — expected difference between two dice rolls (Glassdoor)](https://www.glassdoor.com/Interview/Here-s-an-easy-one-You-are-given-a-six-sided-die-What-is-the-expected-value-of-the-difference-between-the-two-dice-rolls-QTN_834923.htm)
- [ML Interview Q Series — Expected Maximum Value of Two Dice (Rohan Paul)](https://www.rohan-paul.com/p/ml-interview-q-series-calculating-80e)
- [A Working List of Probability Questions Week 3 — Jerry Qin](https://jerryqin.com/posts/a-working-list-of-probability-problems-week-three/)
- [Expected Value Dice Roll — Data Science Interview (bugfree.ai)](https://bugfree.ai/data-question/expected-value-dice-roll)
- [Probability Interview Questions (coin game) — Yashwanth Reddy, Medium](https://medium.com/@reddyyashu20/q1-you-and-your-friend-are-playing-a-game-with-a-fair-coin-7560914f7121)
- [A Biased Coin Toss Interview Question — Henry George, Medium](https://medium.com/@hjegeorge/interview-question-1-a-biased-coin-toss-9dc2af96321)
- [Law of Total Variance — Wikipedia](https://en.wikipedia.org/wiki/Law_of_total_variance)
- [Law of Total Expectation — Wikipedia](https://en.wikipedia.org/wiki/Law_of_total_expectation)
- [Law of Total Variance — The Book of Statistical Proofs](https://statproofbook.github.io/P/var-tot.html)
- [Solving Conditional Probability Problems with Total Expectation, Variance, Covariance — Saurabh Maheshwari, TDS Archive](https://medium.com/data-science/solving-conditional-probability-problems-with-the-laws-of-total-expectation-variance-and-c38c07cfebfa)
- [STAT 24400 Lecture 10 — Expectation & Variance (UChicago)](https://www.stat.uchicago.edu/~yibi/teaching/stat244/L10.pdf)
- [Conditional Variance / Iterated Expectations — probabilitycourse.com](https://www.probabilitycourse.com/chapter5/5_1_5_conditional_expectation.php)

## ⭐ Harder Variants — Topic 1

- [Coupon Collector Problem — Brilliant Math & Science Wiki](https://brilliant.org/wiki/coupon-collector-problem/)
- [Coupon Collector's Problem — Wikipedia](https://en.wikipedia.org/wiki/Coupon_collector's_problem)
- [Coupon Collector's Problem — Tufts CS](https://www.cs.tufts.edu/comp/250P/classpages/coupon.html)
- [Coupon Collector's Problem: A Probability Masterpiece — TDS Archive](https://towardsdatascience.com/coupon-collectors-problem-a-probability-masterpiece-1d5aed4af439/)
- [Coupon Collector — Variance derivation (Quora)](https://www.quora.com/Whats-the-variance-of-the-number-of-coupons-a-coupon-collector-needs-to-collect-before-seeing-each-type)
- [Randomized Algorithms Lecture 6 — Coupon Collector (Patras)](https://www.ceid.upatras.gr/webpages/courses/randalgs/slides/lesson6.pdf)
- [Hitting Time — Wikipedia](https://en.wikipedia.org/wiki/Hitting_time)
- [Random Walk — Wikipedia](https://en.wikipedia.org/wiki/Random_walk)
- [Random Walks lecture notes (Leiden, PDF)](https://prob.math.leidenuniv.nl/lecturenotes/RandomWalks.pdf)
- [Hitting Times — Markov Processes notes (MATH2750)](https://mpaldridge.github.io/math2750/S08-hitting-times.html)
- [Stopping Times — Karl Sigman, Columbia (PDF)](http://www.columbia.edu/~ks20/stochastic-I/stochastic-I-ST.pdf)
- [Jane Street Interview — correlation triangle (Glassdoor)](https://www.glassdoor.com/Interview/If-X-Y-and-Z-are-three-random-variables-such-that-X-and-Y-have-a-correlation-of-0-9-and-Y-and-Z-have-correlation-of-0-8-QTN_467199.htm)
- [Covariance, Correlation, and Joint Probability — AnalystPrep CFA](https://analystprep.com/cfa-level-1-exam/quantitative-methods/covariance-correlation-and-joint-probability/)
- [11 Most Commonly Asked Questions on Correlation — Analytics Vidhya](https://www.analyticsvidhya.com/blog/2015/06/correlation-common-questions/)
- [St. Petersburg Paradox & Utility Theory (decision-theory references)](https://en.wikipedia.org/wiki/St._Petersburg_paradox)
- [Wald's Identity — Wikipedia](https://en.wikipedia.org/wiki/Wald%27s_equation)
- [Brownian Motion notes — Sigman, Columbia (PDF)](http://www.columbia.edu/~ww2040/4701Sum07/4701-06-Notes-BM.pdf)

## Topic 2 — Majority-Vote Classifier

- [Voting Classifier — GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/voting-classifier/)
- [Voting Classifier using Sklearn — GeeksforGeeks](https://www.geeksforgeeks.org/ml-voting-classifier-using-sklearn/)
- [VotingClassifier — scikit-learn docs](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)
- [EnsembleVoteClassifier — mlxtend](https://rasbt.github.io/mlxtend/user_guide/classifier/EnsembleVoteClassifier/)
- [How to Develop Voting Ensembles With Python — Machine Learning Mastery](https://machinelearningmastery.com/voting-ensembles-with-python/)
- [Voting Ensemble: Hard vs Soft Voting Explained — MCP Analytics](https://mcpanalytics.ai/articles/voting-ensemble-practical-guide-for-data-driven-decisions)
- [Implementing a Weighted Majority Rule Ensemble Classifier — Sebastian Raschka](https://sebastianraschka.com/Articles/2014_ensemble_classifier.html)
- [Understanding Voting Classifiers in ML — Lomash Bhuva, Medium](https://medium.com/@lomashbhuva/understanding-voting-classifiers-in-machine-learning-a-comprehensive-guide-6589b5f17e0f)
- [40 Questions to ask a Data Scientist on Ensemble Modeling — Analytics Vidhya](https://www.analyticsvidhya.com/blog/2017/02/40-questions-to-ask-a-data-scientist-on-ensemble-modeling-techniques-skilltest-solution/)
- [Condorcet's Jury Theorem — Wikipedia](https://en.wikipedia.org/wiki/Condorcet's_jury_theorem)
- [Majority Voting and the Condorcet's Jury Theorem (arXiv 2002.03153)](https://arxiv.org/abs/2002.03153)
- [Examining Independence in Ensemble Sentiment Analysis using the Condorcet Jury Theorem — arXiv](https://arxiv.org/html/2409.0094)
- [When is the majority-vote classifier beneficial? — arXiv 1307.6522](https://arxiv.org/abs/1307.6522)
- [New Bounds on the Accuracy of Majority Voting for Multi-Class — arXiv 2309.09564](https://arxiv.org/pdf/2309.09564)
- [Majority Voting by Independent Classifiers Can Increase Accuracy — Iowa State](https://dr.lib.iastate.edu/bitstreams/8fa1d6f7-9779-4bdd-9cb0-8987ee9f416b/download)
- [Majority Voting overview — ScienceDirect Topics](https://www.sciencedirect.com/topics/computer-science/majority-voting)
- [Combining classifiers via majority vote — Draconian Fleet library](http://library.draconianfleet.com/epubfs.php?data=7626&comp=ch07s02.html)

## ⭐ Harder Variants — Topic 2

- [AdaBoost Algorithm Questions and Answers — Sanfoundry](https://www.sanfoundry.com/machine-learning-questions-answers-adaboost-algorithm/)
- [Interview Questions on AdaBoost Algorithm — Analytics Vidhya](https://www.analyticsvidhya.com/blog/2022/11/interview-questions-on-adaboost-algorithm-in-data-science/)
- [Boosting (Aarti Singh, CMU 10-701/15-781 PDF)](https://www.cs.cmu.edu/~aarti/Class/10701/slides/Lecture10.pdf)
- [The AdaBoost Algorithm (Sontag, MIT CSAIL)](https://people.csail.mit.edu/dsontag/courses/ml12/slides/lecture13.pdf)
- [AdaBoost Algorithm Explained in Depth — ProjectPro](https://www.projectpro.io/article/adaboost-algorithm/972)
- [Ensemble Learning Interview Questions — Devinterview-io](https://github.com/Devinterview-io/ensemble-learning-interview-questions)
- [Bias–Variance Tradeoff — Wikipedia](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)
- [Single estimator vs bagging: bias–variance decomposition — scikit-learn](https://scikit-learn.org/stable/auto_examples/ensemble/plot_bias_variance.html)
- [Bias/Variance Tradeoff and Ensemble Methods (Vibhav Gogate, UTD PDF)](https://personal.utdallas.edu/~vibhav.gogate/ml/2020f/lectures/EnsembleMethods.pdf)
- [Unified Bias–Variance Decomposition (Pedro Domingos, UW PDF)](https://homes.cs.washington.edu/~pedrod/papers/mlc00a.pdf)
- [Bias–Variance Decomposition, Ensemble Methods (CSC2515 Toronto, PDF)](https://csc2515.github.io/csc2515-fall2024/lectures/lec04.pdf)
- [Classification Performance of Bagging and Boosting — Springer](https://link.springer.com/article/10.1007/s00354-011-0303-0)
- [On Feature Selection, Bias–Variance, and Bagging (Cornell)](https://www.cs.cornell.edu/~mmunson/publications/docs/fs-bagging.pdf)
- [Hoeffding's Inequality — concentration of sample means (Wikipedia)](https://en.wikipedia.org/wiki/Hoeffding%27s_inequality)
- [Borda Count — Wikipedia (multi-class voting)](https://en.wikipedia.org/wiki/Borda_count)
- [Stacked Generalization — Wolpert 1992 (original paper)](https://www.machine-learning.martinsewell.com/ensembles/stacking/Wolpert1992.pdf)
- [Probabilistic Predictions with Dynamically Weighted Majority Vote — ResearchGate](https://www.researchgate.net/publication/260379631_Probabilistic_Predictions_of_Ensemble_of_Classifiers_Combined_with_Dynamically_Weighted_Majority_Vote)
- [A probabilistic classifier ensemble weighting scheme — Springer DMKD](https://link.springer.com/article/10.1007/s10618-019-00638-y)
