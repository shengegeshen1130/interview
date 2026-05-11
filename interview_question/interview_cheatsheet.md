# ML Engineering Interview — Master Cheatsheet

One-stop reference. Formulas, code, and one-line answers. **Skim before the interview; recall during it.**

> **Universal tip:** (1) Ask clarifying questions. (2) State assumptions. (3) Verbalize the approach before coding. (4) Test a tiny example. (5) State complexity.

---

## Table of Contents

### PART I — MATH, PROBABILITY & STATISTICS *(non-coding)*

1. [Probability — Core Identities](#1-probability--core-identities)
2. [Distributions Reference Table](#2-distributions-reference-table)
3. [Expected Value, Variance, Covariance, Correlation — Complete Properties](#3-expected-value-variance-covariance-correlation--complete-properties)
4. [Gaussian / Normal Distribution — All Angles](#4-gaussian--normal-distribution--all-angles)
5. [Statistics Fundamentals — Estimation & CLT](#5-statistics-fundamentals--estimation--clt)
6. [Hypothesis Testing & A/B Testing](#6-hypothesis-testing--ab-testing)
7. [Expected Value — Problem-Solving Tricks](#7-expected-value--problem-solving-tricks)
8. [Optimal Stopping — Decision Templates](#8-optimal-stopping--decision-templates)
9. [Quick Reference — Must-Know Answers](#9-quick-reference--must-know-answers)

### PART II — ML & SYSTEM CONCEPTS *(non-coding)*

10. [Transformer Architecture — Formulas & Variants](#10-transformer-architecture--formulas--variants)
11. [Deep Learning Cookbook](#11-deep-learning-cookbook)
12. [LLM-specific Concepts](#12-llm-specific-concepts)
13. [ML Systems & Production](#13-ml-systems--production)
14. [Majority Vote / Ensemble — Theory](#14-majority-vote--ensemble--theory)

### PART III — CODING

15. [Matrix Algorithms — Spiral, Rotation, Diagonal](#15-matrix-algorithms--spiral-rotation-diagonal)
16. [Coding Patterns (DSA)](#16-coding-patterns-dsa)
17. [PyTorch Essentials](#17-pytorch-essentials)
18. [Transformer — Code Implementations](#18-transformer--code-implementations)
19. [Probability Programming Patterns](#19-probability-programming-patterns)

### PART IV — BEHAVIORAL

20. [Interview Strategy & Communication](#20-interview-strategy--communication)
21. [Appendix — Must-Know Numbers](#21-appendix--must-know-numbers)

---

# PART I — MATH, PROBABILITY & STATISTICS

---

# 1. Probability — Core Identities

```
Bayes theorem:        P(A|B) = P(B|A)·P(A) / P(B)
Total probability:    P(B)   = Σᵢ P(B|Aᵢ)·P(Aᵢ)
Complement:           P(Aᶜ)  = 1 − P(A)
Inclusion-exclusion:  P(A∪B) = P(A) + P(B) − P(A∩B)
                      P(A∪B∪C) = ΣP(Aᵢ) − ΣP(Aᵢ∩Aⱼ) + P(A∩B∩C)
Independence:         P(A∩B) = P(A)·P(B)
Conditional:          P(A|B) = P(A∩B) / P(B)
```

**Conditional independence vs marginal independence:**
- X ⊥ Y | Z  does NOT imply  X ⊥ Y  (and vice versa)
- Counter-example: coin flip C determines whether X₁,X₂ are correlated

**Odds form of Bayes (log-likelihood ratio):**
```
Posterior odds = Prior odds × Likelihood ratio
P(H|E)/P(Hᶜ|E) = P(H)/P(Hᶜ) × P(E|H)/P(E|Hᶜ)
```

---

# 2. Distributions Reference Table

| Distribution | PMF / PDF | E[X] | Var(X) | Key property |
|---|---|---|---|---|
| **Bernoulli(p)** | P(X=1)=p | p | p(1−p) | Single trial |
| **Binomial(n,p)** | C(n,k)pᵏ(1−p)ⁿ⁻ᵏ | np | np(1−p) | n Bernoulli trials |
| **Geometric(p)** | (1−p)ᵏ⁻¹p | 1/p | (1−p)/p² | Memoryless (discrete) |
| **Negative Binomial(r,p)** | — | r/p | r(1−p)/p² | r-th success wait |
| **Hypergeometric(N,K,n)** | C(K,k)C(N−K,n−k)/C(N,n) | nK/N | n·K/N·(N−K)/N·(N−n)/(N−1) | No replacement |
| **Poisson(λ)** | e⁻λλᵏ/k! | λ | λ | E=Var=λ |
| **Uniform{1..n}** | 1/n | (n+1)/2 | (n²−1)/12 | — |
| **Uniform[a,b]** | 1/(b−a) | (a+b)/2 | (b−a)²/12 | — |
| **Exponential(λ)** | λe⁻λˣ | 1/λ | 1/λ² | Memoryless (continuous) |
| **Normal(μ,σ²)** | see §4 | μ | σ² | CLT limit |
| **Beta(α,β)** | xᵅ⁻¹(1−x)^{β−1}/B(α,β) | α/(α+β) | αβ/((α+β)²(α+β+1)) | Conjugate to Bernoulli |
| **Gamma(α,β)** | xᵅ⁻¹e⁻ˣ/ᵝ/Γ(α)βᵅ | αβ | αβ² | Sum of exponentials |
| **Chi-squared(k)** | Gamma(k/2, 2) | k | 2k | Sum of k squared normals |

**Poisson ≈ Binomial** when n large, p small, np = λ moderate.

**Normal ≈ Binomial** when n large (CLT), μ=np, σ²=np(1-p).

---

# 3. Expected Value, Variance, Covariance, Correlation — Complete Properties

## 3.1 Expected Value E[X]

```
Discrete:    E[X] = Σₓ x · P(X = x)
Continuous:  E[X] = ∫₋∞^∞ x · f(x) dx

LOTUS (Law of the Unconscious Statistician):
  E[g(X)] = Σₓ g(x) P(X=x)       (discrete)
  E[g(X)] = ∫ g(x) f(x) dx       (continuous)
```

**Properties of E:**

| Rule | Formula | Note |
|---|---|---|
| Linearity | E[aX + bY + c] = aE[X] + bE[Y] + c | **Always** — no independence needed |
| Constant | E[c] = c | — |
| Product (indep) | E[XY] = E[X]·E[Y] | **Only if X ⊥ Y** |
| Tower property | E[X] = E[E[X\|Y]] | Useful for multi-stage problems |
| E[X²] | E[X²] = Var(X) + (E[X])² | Rearrangement of Var formula |

## 3.2 Variance Var(X)

```
Definition:     Var(X) = E[(X − μ)²]           μ = E[X]
Shortcut:       Var(X) = E[X²] − (E[X])²        ← always use this for calculations
Std deviation:  SD(X) = σ = √Var(X)
```

**Properties of Var:**

| Rule | Formula | Note |
|---|---|---|
| Scale | Var(aX) = a²·Var(X) | squaring the constant |
| Shift | Var(X + c) = Var(X) | constant doesn't affect spread |
| Linear combo | Var(aX + bY) = a²Var(X) + b²Var(Y) + 2ab·Cov(X,Y) | general |
| If X ⊥ Y | Var(X + Y) = Var(X) + Var(Y) | Cov = 0 |
| Sample variance | s² = Σ(xᵢ − x̄)²/(n−1) | unbiased; divide by n−1 |
| Eve's law (LTP) | Var(X) = E[Var(X\|Y)] + Var(E[X\|Y]) | "total variance" |

## 3.3 Covariance Cov(X, Y)

```
Definition:  Cov(X,Y) = E[(X−μX)(Y−μY)]
Shortcut:    Cov(X,Y) = E[XY] − E[X]·E[Y]      ← use this
```

**Properties of Cov:**

| Rule | Formula |
|---|---|
| Symmetric | Cov(X,Y) = Cov(Y,X) |
| Self | Cov(X,X) = Var(X) |
| Bilinear | Cov(aX+b, cY+d) = ac·Cov(X,Y) |
| Distributive | Cov(X+Y, Z) = Cov(X,Z) + Cov(Y,Z) |
| Independent | If X ⊥ Y → Cov(X,Y) = 0 (converse false!) |
| Var of sum | Var(X+Y) = Var(X) + Var(Y) + 2Cov(X,Y) |
| Var of sum n | Var(ΣXᵢ) = Σ Var(Xᵢ) + 2Σᵢ<ⱼ Cov(Xᵢ,Xⱼ) |

**Important**: Cov(X,Y) = 0 does NOT imply independence. Counter-example: X ~ U(-1,1), Y = X².

## 3.4 Correlation Coefficient ρ (Pearson)

```
ρ(X,Y) = Cov(X,Y) / (SD(X)·SD(Y))     range: [−1, 1]
```

**Interpretation:**

| ρ | Meaning |
|---|---|
| ρ = +1 | Perfect positive linear relationship: Y = aX + b, a > 0 |
| ρ = −1 | Perfect negative linear relationship: Y = aX + b, a < 0 |
| ρ = 0 | Uncorrelated — but NOT necessarily independent |
| 0 < ρ < 1 | Positive linear association |
| −1 < ρ < 0 | Negative linear association |

**Pearson vs Spearman:**

| | Pearson | Spearman |
|---|---|---|
| Measures | Linear relationship | Monotone relationship |
| Data | Raw values | Ranks of values |
| Robust to outliers | No | Yes |
| Use when | Normal, linear | Skewed, ordinal, non-linear monotone |

**Sample correlation (estimate from data):**
```
r = Σ(xᵢ−x̄)(yᵢ−ȳ) / √[Σ(xᵢ−x̄)² · Σ(yᵢ−ȳ)²]
```

## 3.5 Moment Generating Function (MGF)

```
M_X(t) = E[e^{tX}]
E[Xⁿ] = M_X^{(n)}(0)      (n-th derivative at 0)
```

| Distribution | MGF |
|---|---|
| Bernoulli(p) | 1 − p + pe^t |
| Binomial(n,p) | (1 − p + pe^t)ⁿ |
| Poisson(λ) | exp(λ(eᵗ − 1)) |
| Normal(μ,σ²) | exp(μt + σ²t²/2) |
| Exponential(λ) | λ/(λ−t), t < λ |

---

# 4. Gaussian / Normal Distribution — All Angles

## 4.1 Core Formulas

```
PDF:   f(x; μ, σ²) = (1 / √(2πσ²)) · exp(−(x−μ)² / (2σ²))

Standard (μ=0, σ=1):
       φ(x) = (1/√(2π)) · exp(−x²/2)
       Φ(x) = P(Z ≤ x) = ∫₋∞ˣ φ(t) dt      (no closed form)

Standardize:  Z = (X − μ) / σ  ~  N(0, 1)
```

## 4.2 Key Quantiles (Memorize)

| Scenario | z-value | Meaning |
|---|---|---|
| 90% CI → z₀.₀₅ | **1.645** | P(Z > 1.645) = 5% |
| 95% CI → z₀.₀₂₅ | **1.960** | P(Z > 1.96) = 2.5% |
| 99% CI → z₀.₀₀₅ | **2.576** | P(Z > 2.576) = 0.5% |
| 1σ rule | ±1.0 | P(-1 ≤ Z ≤ 1) ≈ **68.3%** |
| 2σ rule | ±2.0 | P(-2 ≤ Z ≤ 2) ≈ **95.4%** |
| 3σ rule | ±3.0 | P(-3 ≤ Z ≤ 3) ≈ **99.7%** |

## 4.3 Properties

```
Linear transform:     X ~ N(μ,σ²)  →  aX+b ~ N(aμ+b, a²σ²)
Sum of independents:  X+Y ~ N(μ₁+μ₂, σ₁²+σ₂²)
Scaling:              X ~ N(0,1)  →  μ + σX ~ N(μ, σ²)
Symmetry:             φ(−x) = φ(x),  Φ(−x) = 1 − Φ(x)
MGF:                  M_X(t) = exp(μt + σ²t²/2)
```

## 4.4 Normal Distribution From Multiple Angles

| Angle | Statement |
|---|---|
| **Probability / CLT** | Sum of n i.i.d. r.v.s → N(nμ, nσ²); rescaled → N(0,1) |
| **Information theory** | Max-entropy distribution with fixed mean & variance (among all distributions on ℝ) |
| **Bayesian** | Conjugate prior for the mean (with known variance); N·N → N |
| **Geometry (high-d)** | In ℝᵈ, N(0,I): almost all mass on thin shell at radius √d; components almost orthogonal |
| **Physics / Diffusion** | Solution to heat equation: Brownian motion density at time t is N(0, t) |
| **Log-space** | Log-normal: if log X ~ N(μ,σ²), then X = eˣ is right-skewed |
| **Sampling** | Box-Muller: Z₁=√(−2ln U₁)·cos(2πU₂), Z₂=√(−2ln U₁)·sin(2πU₂) are i.i.d. N(0,1) |

## 4.5 Gaussian Function Properties (FWHM)

```
y = exp(−(x−μ)²/(2σ²))
Maximum at x = μ:  y_max = 1

Half-maximum at:   x = μ ± σ√(2 ln 2)
FWHM = 2σ√(2 ln 2) ≈ 2.355σ
```

| Function | FWHM |
|---|---|
| Gaussian  exp(−x²/2σ²) | 2√(2ln2)·σ ≈ 2.355σ |
| Cauchy/Lorentz  1/(1+(x/γ)²) | 2γ |
| Laplace  exp(−\|x\|/b) | 2b·ln2 ≈ 1.386b |

---

# 5. Statistics Fundamentals — Estimation & CLT

## 5.1 Sample Statistics

```
Sample mean:      x̄  = (1/n) Σ xᵢ                  (unbiased: E[x̄] = μ)
Sample variance:  s²  = Σ(xᵢ − x̄)² / (n−1)          (unbiased; divide by n−1 not n)
Standard error:   SE  = σ/√n  ≈ s/√n                 (uncertainty of x̄)
```

**Why n-1?** Bessel's correction. One degree of freedom is "used" to estimate μ, leaving n-1 free.

## 5.2 Central Limit Theorem (CLT)

```
X₁, X₂, ..., Xₙ i.i.d.  with mean μ, variance σ²

CLT:   √n · (X̄ − μ) / σ  →  N(0, 1)   as  n → ∞
i.e.:  X̄  ~  N(μ,  σ²/n)             for large n (typically n ≥ 30)
```

**CLT vs Law of Large Numbers (LLN):**

| | LLN | CLT |
|---|---|---|
| What it says | X̄ **converges to** μ | **Distribution of (X̄−μ)** converges to N(0,σ²/n) |
| Type | Convergence of value | Convergence of distribution shape |
| Analogy | "The average will be right" | "The average's error is bell-curved" |

## 5.3 Maximum Likelihood Estimation (MLE)

```
Likelihood:       L(θ) = P(data | θ) = Πᵢ f(xᵢ; θ)
Log-likelihood:   ℓ(θ) = log L(θ) = Σᵢ log f(xᵢ; θ)
MLE:              θ̂ = argmax ℓ(θ)  →  set dℓ/dθ = 0
```

**Common MLEs:**

| Model | MLE |
|---|---|
| Bernoulli(p) | p̂ = k/n (sample proportion) |
| Normal(μ,σ²) | μ̂ = x̄,  σ̂² = Σ(xᵢ−x̄)²/n (biased!) |
| Exponential(λ) | λ̂ = 1/x̄ |
| Poisson(λ) | λ̂ = x̄ |

**MLE vs MAP:**

| | MLE | MAP |
|---|---|---|
| Objective | max P(data\|θ) | max P(θ\|data) = P(data\|θ)·P(θ) |
| Prior | ignored | included |
| Small-n stability | unstable (can overfit) | stable (regularized by prior) |
| Bernoulli example (7H/10) | p̂=0.7 | p̂=(7+α)/(10+α+β) with Beta(α,β) prior |

## 5.4 Confidence Intervals

```
95% CI for mean (σ known):      x̄  ±  1.96 · σ/√n
95% CI for mean (σ unknown):    x̄  ±  t_{0.025, n-1} · s/√n
95% CI for proportion:          p̂  ±  1.96 · √(p̂(1−p̂)/n)
```

**Correct interpretation:** "If we repeated this procedure many times, ~95% of resulting intervals would contain the true parameter." The true parameter is fixed; the interval is random.

**Common wrong interpretations:**
- ❌ "There's a 95% chance μ is in (a, b)" — μ is fixed; probability doesn't apply
- ❌ "95% of data points are in the CI" — CI is about the mean, not data spread

**Width determinants:**
- Wider CI ← smaller n, higher σ, higher confidence level, p closer to 0.5

## 5.5 Bias, Variance, and MSE

```
Bias(θ̂)   = E[θ̂] − θ
MSE(θ̂)    = Var(θ̂) + Bias(θ̂)²
Unbiased:   E[θ̂] = θ   (Bias = 0)
```

Bias-variance tradeoff in ML: regularization increases bias but reduces variance; may lower MSE overall.

---

# 6. Hypothesis Testing & A/B Testing

## 6.1 Framework

```
H₀: null hypothesis (default, what we're trying to disprove)
H₁: alternative hypothesis (what we're trying to establish)

α   = P(Type I error)  = P(reject H₀ | H₀ true)    ← significance level (set by us, usually 0.05)
β   = P(Type II error) = P(fail to reject H₀ | H₁ true)
Power = 1 − β                                         ← sensitivity of the test
```

**Decision matrix:**

|  | H₀ true | H₀ false |
|---|---|---|
| **Reject H₀** | Type I Error (α) | Correct (Power = 1−β) |
| **Fail to reject H₀** | Correct (1−α) | Type II Error (β) |

## 6.2 p-value

```
p-value = P(data as extreme or more extreme | H₀ true)

Reject H₀ iff  p < α
```

**p-value is NOT:**
- Probability H₀ is true
- Probability your discovery is false
- A measure of effect size or practical significance

## 6.3 Common Test Statistics

| Test | Statistic | Distribution under H₀ | Use when |
|---|---|---|---|
| 1-sample z | z = (x̄−μ₀)/(σ/√n) | N(0,1) | σ known |
| 1-sample t | t = (x̄−μ₀)/(s/√n) | t_{n-1} | σ unknown |
| 2-sample z (proportions) | z = (p̂₁−p̂₂)/SE | N(0,1) | large n |
| 2-sample t | t = (x̄₁−x̄₂)/SE | t_{n₁+n₂-2} | small n, σ unknown |
| Chi-squared | χ² = Σ(O−E)²/E | χ²_{k-1} | categorical; goodness of fit |
| ANOVA (F-test) | F = MSB/MSW | F_{k-1, N-k} | compare ≥3 group means |

## 6.4 A/B Testing — Design Checklist

**Step 1: Define hypothesis**
```
H₀: p₁ = p₂  (no difference in conversion rates)
H₁: p₁ ≠ p₂  (two-tailed; use one-tailed only if direction certain)
```

**Step 2: Sample size calculation**
```
For two proportions (most common in practice):

n = (z_{α/2} + z_β)² · [p₁(1−p₁) + p₂(1−p₂)] / (p₁ − p₂)²

Standard params:  α=0.05  →  z_{α/2}=1.96
                  Power=80% →  z_β=0.84    → (1.96+0.84)²≈7.84
                  Power=90% →  z_β=1.28    → (1.96+1.28)²≈10.5
```

**Example:** p₁=10%, p₂=12%, α=0.05, power=80%
```
n = 7.84 × (0.1×0.9 + 0.12×0.88) / (0.02)² = 7.84 × 0.1956 / 0.0004 ≈ 3,832 per group
```

**Step 3: Run the experiment**

| Requirement | Why |
|---|---|
| Pre-register n and α | Prevent optional stopping (p-hacking) |
| Randomize assignment | Eliminate confounds |
| One variable at a time | Attribution clarity |
| Run full duration | Weekly seasonality effects |
| Check sample ratio mismatch (SRM) | Detect experiment bugs |

**Step 4: Pitfalls to watch**

| Pitfall | Problem | Fix |
|---|---|---|
| Early stopping (peeking) | Inflates Type I error | Sequential testing (mSPRT) or pre-commit to n |
| Multiple metrics | FWER inflates | Pre-specify primary metric; FDR for secondary |
| Network effects (spillover) | Control/treatment bleed | Cluster randomization |
| Simpson's Paradox | Confound reverses direction | Stratify by segment; check each subgroup |
| Novelty effect | Short-term spike, then decay | Run longer; measure sustained behavior |

## 6.5 Multiple Testing

```
FWER (Family-Wise Error Rate) with m independent tests:
P(≥1 false positive) = 1 − (1−α)^m

m=20, α=0.05:  FWER = 1 − 0.95²⁰ ≈ 64.2% !
```

| Method | Controls | Formula | Conservative? |
|---|---|---|---|
| **Bonferroni** | FWER | α' = α/m | Very (use for confirmatory) |
| **Holm-Bonferroni** | FWER | Sequential Bonferroni | Less than plain Bonferroni |
| **Benjamini-Hochberg (BH)** | FDR | Sort p-values; reject p_{(i)} ≤ (i/m)·α | Least (use for exploration) |

**FDR (False Discovery Rate)** = expected fraction of rejected H₀ that are false positives. Preferred in genomics, multi-metric dashboards.

## 6.6 Common Interview: Power Analysis

```
"What increases statistical power?"
1. Larger n         → SE shrinks → easier to detect effect
2. Larger effect size (Δ = |μ₁−μ₂|) → easier to detect
3. Smaller σ        → less noise
4. Higher α         → (but at cost of more false positives)
5. One-tailed test  → (but requires directional prior)
```

---

# 7. Expected Value — Problem-Solving Tricks

## Trick 1 — Linearity (no independence needed)

```
E[aX + bY + c] = aE[X] + bE[Y] + c   ← always true

Classic: E[fixed points in random permutation of n] = 1
(sum of n indicator variables, each with E = 1/n)
```

## Trick 2 — Condition on first step

```
"Expected flips until first head (p = prob of H)":
  E[X] = 1·p + (1 + E[X])·(1−p)  →  E[X] = 1/p

Template: E[X] = (immediate payoff) + (prob of continue) × E[X]
```

## Trick 3 — Indicator variables

```
"Expected number of distinct coupons after n draws (k types)":
  Iⱼ = 1 if coupon j appears at least once
  E[Iⱼ] = 1 − ((k−1)/k)^n
  E[distinct] = k · (1 − ((k−1)/k)^n)
```

## Trick 4 — Wald's Identity

```
Xᵢ i.i.d. with mean μ; N = stopping time with E[N] < ∞:
  E[X₁ + X₂ + ... + X_N] = μ · E[N]
```

## Trick 5 — Eve's Law (Total Variance)

```
Var(X) = E[Var(X|Y)] + Var(E[X|Y])
       = (expected within-group variance) + (variance of group means)

Compound Poisson: N~Pois(λ), each Xᵢ has mean μ and var σ²:
  E[S] = λμ
  Var[S] = λ(μ² + σ²) = λ E[X²]
```

## Trick 6 — Order Statistics

```
X₁,...,Xₙ ~ U(0,1) i.i.d., X_(k) = k-th smallest:
  E[X_(k)] = k/(n+1)
  E[max] = n/(n+1)      E[min] = 1/(n+1)

General: X_(k) ~ Beta(k, n−k+1)
```

## Trick 7 — Geometric Series / Telescoping

```
Σ_{k=0}^∞ kpqᵏ⁻¹ = 1/p     (expected value of Geometric(p))

For E[X] when X counts until first success in rounds:
  Often easier to write E[X] = Σ_{k=0}^∞ P(X > k)   (CDF form)
```

---

# 8. Optimal Stopping — Decision Templates

## Universal recurrence

```
V[k] = expected value with k decisions remaining
V[1] = base case (forced last action)
V[k] = E[ max(take_now, V[k−1]) ]   or   E[ max(reward, threshold) ]
```

## Secretary problem (1/e rule)

```
Reject first ⌊n/e⌋ ≈ 0.368n candidates.
Pick first one strictly better than all seen.
P(success) → 1/e ≈ 0.368.
```

## Prophet inequality

```
Single threshold τ: set P(max Xᵢ ≥ τ) = 1/2.
Guarantees ≥ 0.5 · E[max Xᵢ] in expectation.
```

## Dice game (n rolls, take highest)

```
Threshold rule: V[k] = threshold for stopping
V[1] = 3.5
V[k] = (1/6) Σᵥ max(v, V[k−1])

n=2 → V=4.25, stop if roll > 4.25 (i.e., roll 5 or 6)
n=3 → V=4.67, stop first roll if ≥5; stop second if ≥4; keep third
```

---

# 9. Quick Reference — Must-Know Answers

| Question | Answer |
|---|---|
| E[fair die] | 3.5 |
| Var(fair die) | 35/12 ≈ 2.917 |
| E[max(X₁,X₂)] two fair dice | 161/36 ≈ 4.47 |
| E[min(X₁,X₂)] two fair dice | 91/36 ≈ 2.53 (sum to 7 ✓) |
| Expected flips for 1st head | 2 |
| Expected flips for HHH | 14 |
| Expected flips for HTH | 10 |
| Expected flips for HH vs TH — which is longer? | HH (14), TH (8) — TH is shorter |
| Coupon collector (n types): E[draws] | n·Hₙ ≈ n ln n + 0.577n |
| Coupon collector: Var | π²n²/6 |
| Gambler's ruin P(win) from position i, range [0,N] | i/N (symmetric) |
| Gambler's ruin E[time] | i(N−i) |
| Secretary problem cutoff | n/e ≈ 0.368n |
| 3 classifiers @ 80%, majority | 0.896 |
| Mismatch: P(HH first) vs P(TH first) starting fresh | P(HH first) = 1/4, P(TH first) = 1/4 **BUT** in a sequence HH requires 2 consecutive H, while TH is easier to "catch"—TH appears first with prob 3/4 |
| Two-envelope paradox | Switching and staying have same expected value = 3m/2 |
| Simpson's Paradox exists? | Yes — can reverse direction when stratified |

---

# PART II — ML & SYSTEM CONCEPTS *(non-coding)*

---

# 10. Transformer Architecture — Formulas & Variants

## Core formulas

```
Scaled dot-product attention:
  Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V

Multi-head:
  MultiHead(Q,K,V) = Concat(head₁,...,headₕ) · Wᴼ
  headᵢ = Attention(Q·Wᵢᵠ, K·Wᵢᴷ, V·Wᵢᵛ)

Positional encoding (sinusoidal):
  PE(pos,2i)   = sin(pos / 10000^{2i/d})
  PE(pos,2i+1) = cos(pos / 10000^{2i/d})
```

## Attention head variants

| Variant | Q heads | K/V heads | Use case |
|---|---|---|---|
| MHA (vanilla) | H | H | Small models |
| MQA | H | 1 | Inference speed (PaLM, Falcon) |
| GQA | H | G (G<H) | Llama 2/3, balance |
| Cross-attention | from decoder | from encoder | Encoder-decoder |

## Positional encodings

| Type | Idea | Pros / Cons |
|---|---|---|
| Sinusoidal | Fixed sin/cos | Extrapolates poorly |
| Learned absolute | nn.Embedding | Simple; bounded length |
| RoPE | Rotate Q,K by angle θ=pos·10000^{-2i/d} | Llama, GPT-NeoX; relative-position aware |
| ALiBi | Add linear bias −m·dist to attn scores | Train short, infer long |

## Normalization variants

| Variant | Formula | Notes |
|---|---|---|
| Post-LN (original) | LN(x + Sublayer(x)) | Hard to train deep |
| Pre-LN | x + Sublayer(LN(x)) | Default in modern LLMs |
| RMSNorm | x / RMS(x) · γ (no mean centering) | Llama; faster, no bias |

## Complexity

| Component | Time | Memory |
|---|---|---|
| Attention QKᵀ | O(T²d) | O(T²) |
| FFN (4× expand) | O(Td²) | O(Td) |
| FlashAttention | O(T²d) time | **O(T) memory** (tiled, IO-aware) |

## Key numbers

```
d_model = H × d_k         (split, NOT duplicate)
GPT-2 small:  d=768,  H=12, 12 layers
GPT-3 175B:   d=12288, H=96, 96 layers
Llama-2 7B:   d=4096,  H=32, 32 layers
```

---

# 11. Deep Learning Cookbook

## Activations

| Name | Formula | Use |
|---|---|---|
| ReLU | max(0, x) | Default; can "die" |
| GELU | x·Φ(x) | BERT, GPT-2 |
| SiLU/Swish | x·σ(x) | Llama, modern |
| Sigmoid | 1/(1+e⁻ˣ) | Binary output; **d/dx = σ(x)(1−σ(x))** |
| Tanh | (eˣ−e⁻ˣ)/(eˣ+e⁻ˣ) | RNNs |
| Softmax | eˣⁱ / Σeˣʲ | Output probabilities |

**Sigmoid derivative:** σ'(x) = σ(x)(1−σ(x)); max at x=0, value = 1/4.

## Initialization

| Method | Variance | Use |
|---|---|---|
| Xavier/Glorot | 2/(fan_in + fan_out) | Sigmoid/tanh |
| He/Kaiming | 2/fan_in | ReLU/SiLU |
| Transformer default | N(0, 0.02²) | GPT-style |

## Regularization

| Method | How it helps |
|---|---|
| Dropout | Random zero-out → ensemble effect; p=0.1 for transformers |
| Weight decay (L2) | Penalizes large weights → smaller model space |
| Label smoothing α=0.1 | Prevents overconfident predictions |
| Gradient clipping ‖g‖≤1 | Prevents exploding gradients |
| Early stopping | Stops at minimum val loss |

## Backprop key gradients

```
Y = XW:      dL/dX = dL/dY · Wᵀ
             dL/dW = Xᵀ · dL/dY
Chain rule:  dL/dx = dL/dy · dy/dx
```

## Bias–Variance in ML

| Symptom | Cause | Fix |
|---|---|---|
| High train loss | High bias (underfitting) | Bigger model, more features |
| High val loss, low train loss | High variance (overfitting) | More data, dropout, regularization |

---

# 12. LLM-specific Concepts

## Tokenization methods

| Method | Used by | Idea |
|---|---|---|
| BPE | GPT-2/3, Llama | Greedy merge frequent pairs |
| WordPiece | BERT | Merge by likelihood |
| SentencePiece | T5, Llama, Gemma | Raw bytes; language-agnostic |

## Generation strategies

| Strategy | Hyperparameter | Notes |
|---|---|---|
| Greedy | none | Fast; degenerates (repetition) |
| Beam search | num_beams | Good for translation; less diverse |
| Top-k | k=50 | Sample from k highest prob tokens |
| Top-p (nucleus) | p=0.9 | Sample from smallest set summing to p |
| Temperature | T: <1 sharpen, >1 flatten | Applied before softmax |

## Fine-tuning techniques

| Method | Trainable % | Use |
|---|---|---|
| Full FT | 100% | Best quality if compute available |
| LoRA | ~0.5% | Low-rank adapters; most popular |
| QLoRA | ~0.5% on 4-bit base | Single-GPU 70B fine-tuning |
| Prefix tuning | learned KV prefix | Task tokens |
| RLHF / DPO | reward-aligned | Alignment |

## LLM one-liners

| Term | Definition |
|---|---|
| KV cache | Cache K,V across decode steps → O(T) per step vs O(T²) |
| MoE | Only top-k of N FFNs activate per token |
| Speculative decoding | Draft model proposes tokens; big model verifies |
| In-context learning | Few-shot examples in prompt; no weight update |
| RAG | Retrieve relevant docs → concat to prompt → generate |
| Hallucination | Confident, wrong factual output |

---

# 13. ML Systems & Production

## Training pipeline

```
Data → Features → Model → Eval → Deploy → Monitor → Retrain
```

## Serving optimizations

| Technique | Gain | Notes |
|---|---|---|
| Batching (continuous) | throughput | vLLM, TGI |
| Quantization INT8/INT4 | 2–4× memory | GPTQ, AWQ |
| Distillation | smaller model | Student learns from teacher |
| KV cache | O(T) decode | Cache across generation steps |
| Speculative decoding | latency | Draft + verify |

## Distributed training strategies

| Strategy | Splits | Use when |
|---|---|---|
| Data parallel (DDP) | batch | Model fits 1 GPU |
| Tensor parallel | weight matrices | Layer too big |
| Pipeline parallel | layers | Memory-bound |
| ZeRO / FSDP | optim states+grads+params | Large models, limited GPU |

## Eval metrics

| Task | Primary metrics |
|---|---|
| Binary classification | Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC |
| Multi-class | Macro-F1, Confusion matrix |
| Regression | MAE, RMSE, R² |
| Ranking / IR | nDCG, MAP, MRR, Hit@k |
| Generation | BLEU, ROUGE, BERTScore, Perplexity |
| LLM | MMLU, HumanEval, MT-Bench, win-rate |

---

# 14. Majority Vote / Ensemble — Theory

## Master formula (n independent voters @ accuracy p, n odd)

```
P(majority correct) = Σ_{k=⌈n/2⌉}^{n} C(n,k) pᵏ (1−p)^{n−k}
```

## Quick table

| n | p=0.6 | p=0.7 | p=0.8 | p=0.9 |
|---|---|---|---|---|
| 1 | 0.600 | 0.700 | 0.800 | 0.900 |
| 3 | 0.648 | 0.784 | 0.896 | 0.972 |
| 5 | 0.683 | 0.837 | 0.942 | 0.991 |
| 11 | 0.753 | 0.922 | 0.988 | 0.9999 |

**Condorcet theorem:** p > 0.5 and independent → accuracy → 1 as n → ∞.

## Bias-variance for bagging

```
Var(f̄) = ρ·σ_f² + (1−ρ)/M · σ_f²    →    ρ·σ_f²  as M → ∞
```
Bagging reduces independent variance. Random Forest reduces ρ further via feature subsetting.

## AdaBoost training error bound

```
training_error ≤ exp(−2 · Σ_t γ_t²)     where γ_t = 0.5 − ε_t
```

---

# PART III — CODING

---

# 15. Matrix Algorithms — Spiral, Rotation, Diagonal

## Pattern A — Shrinking boundaries (outside-in spiral)

```python
def spiral_order(matrix):
    if not matrix: return []
    res = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    while top <= bottom and left <= right:
        for j in range(left, right + 1):   res.append(matrix[top][j]);  top += 1
        for i in range(top, bottom + 1):   res.append(matrix[i][right]); right -= 1
        if top <= bottom:
            for j in range(right, left-1, -1): res.append(matrix[bottom][j]); bottom -= 1
        if left <= right:
            for i in range(bottom, top-1, -1): res.append(matrix[i][left]);   left += 1
    return res
```

## Pattern B — Expanding step 1,1,2,2,3,3,... (inside-out / arbitrary start)

```python
def spiral_from(R, C, rStart, cStart):
    res = [[rStart, cStart]]
    dx, dy = [0,1,0,-1], [1,0,-1,0]   # R, D, L, U
    x, y, step, d = rStart, cStart, 1, 0
    while len(res) < R*C:
        for _ in range(2):
            for _ in range(step):
                x += dx[d]; y += dy[d]
                if 0 <= x < R and 0 <= y < C:
                    res.append([x, y])
                    if len(res) == R*C: return res
            d = (d + 1) % 4
        step += 1
    return res
```

## Rotate Image 90° clockwise (in-place)

```python
def rotate(M):
    n = len(M)
    for i in range(n):
        for j in range(i+1, n):
            M[i][j], M[j][i] = M[j][i], M[i][j]   # transpose
    for row in M:
        row.reverse()                                # reverse each row
```

## Diagonal traverse

```python
from collections import defaultdict
def diag(matrix):
    d = defaultdict(list)
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            d[i+j].append(v)
    out = []
    for k in sorted(d):
        out += reversed(d[k]) if k % 2 == 0 else d[k]
    return out
```

## Decision tree

| Spiral type | Pattern |
|---|---|
| Outside-in, corner start | Pattern A |
| Inside-out / arbitrary start | Pattern B |
| Diagonal | defaultdict keyed by i±j |
| In-place 90° rotation | transpose + row reverse |

---

# 16. Coding Patterns (DSA)

## Two pointers

```python
l, r = 0, len(arr) - 1
while l < r:
    s = arr[l] + arr[r]
    if s == target: return [l, r]
    elif s < target: l += 1
    else: r -= 1
```

## Sliding window

```python
from collections import Counter
def longest_k_distinct(s, k):
    cnt, l, best = Counter(), 0, 0
    for r, c in enumerate(s):
        cnt[c] += 1
        while len(cnt) > k:
            cnt[s[l]] -= 1
            if cnt[s[l]] == 0: del cnt[s[l]]
            l += 1
        best = max(best, r - l + 1)
    return best
```

## Binary search (leftmost template)

```python
def bisect_left(arr, target):
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < target: lo = mid + 1
        else: hi = mid
    return lo
```

## DP pattern recognition

| Pattern | Signature |
|---|---|
| 1D linear | dp[i] from dp[i-1], dp[i-2] |
| 2D grid paths | dp[i][j] from neighbors |
| Knapsack 0/1 | dp[i][w] = max(skip, take) |
| LIS | O(n log n) via patience sort |
| Edit distance | dp[i][j] from 3 transitions |
| Interval DP | dp[i][j] = best over all splits |
| Backward induction | V[k] = expected value with k steps left |

## BFS template

```python
from collections import deque
def bfs(start, neighbors):
    seen = {start}
    q = deque([start])
    while q:
        node = q.popleft()
        for nb in neighbors(node):
            if nb not in seen:
                seen.add(nb); q.append(nb)
```

## Heap (priority queue)

```python
import heapq
heap = []
heapq.heappush(heap, (priority, item))
priority, item = heapq.heappop(heap)
# Top-k: maintain min-heap of size k; push and pop when len > k
```

## Union-Find (Disjoint Set Union)

```python
class DSU:
    def __init__(self, n): self.p = list(range(n)); self.rank = [0]*n
    def find(self, x): 
        if self.p[x] != x: self.p[x] = self.find(self.p[x])  # path compress
        return self.p[x]
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py: return False
        if self.rank[px] < self.rank[py]: px, py = py, px
        self.p[py] = px
        if self.rank[px] == self.rank[py]: self.rank[px] += 1
        return True
```

## Monotone stack

```python
# Next greater element
def next_greater(nums):
    res, stack = [-1]*len(nums), []
    for i, v in enumerate(nums):
        while stack and nums[stack[-1]] < v:
            res[stack.pop()] = v
        stack.append(i)
    return res
```

---

# 17. PyTorch Essentials

## Training loop template

```python
model.train()
for x, y in loader:
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()                               # ← never forget
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

model.eval()
with torch.no_grad():                                   # ← never forget
    for x, y in val_loader:
        pred = model(x.to(device))
```

## 5 common bugs

| Bug | Symptom | Fix |
|---|---|---|
| Forget `zero_grad()` | Loss explodes, gradients accumulate | Add before backward |
| Forget `model.eval()` | Dropout/BN wrong at val | Set before inference |
| Forget `no_grad()` | OOM during inference | Wrap inference in `with torch.no_grad()` |
| `print(loss)` | Graph held in memory | Use `loss.item()` |
| Device mismatch | RuntimeError | Move all tensors to same device |

## Shapes cheatsheet

| Layer | Input | Output |
|---|---|---|
| `nn.Linear(in, out)` | (*, in) | (*, out) |
| `nn.Conv2d(C, F, k)` | (B, C, H, W) | (B, F, H', W') |
| `nn.MultiheadAttention(d, h)` | (T, B, d) | (T, B, d) |
| `nn.LayerNorm(d)` | (*, d) | (*, d) |
| `nn.BatchNorm1d(C)` | (B, C) | (B, C) |

## Losses

| Task | Loss | Note |
|---|---|---|
| Binary | `BCEWithLogitsLoss` | Input logits, not probs |
| Multi-class | `CrossEntropyLoss` | Logits + integer labels |
| Regression | `MSELoss / HuberLoss` | Huber for outliers |

```python
# Cross-entropy on (B, C) logits + (B,) integer labels
loss = F.cross_entropy(logits, labels)   # = log_softmax + NLL
```

## Optimizers

| Optimizer | lr default | Use |
|---|---|---|
| SGD | 0.01 | Vision; fine-tuning |
| Adam | 3e-4 | General default |
| AdamW | 1e-4 | Transformers (decoupled weight decay) |

## Mixed precision

```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast():
    loss = model(x, y)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## LR warmup + cosine decay

```python
def lr_lambda(step):
    if step < warmup_steps: return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))
```

---

# 18. Transformer — Code Implementations

## Scaled dot-product attention

```python
import torch, torch.nn.functional as F

def scaled_dot_product(q, k, v, mask=None):
    # q: (B, H, T, d_k)
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / d_k**0.5    # (B, H, T, T)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    return F.softmax(scores, dim=-1) @ v
```

## Causal mask

```python
causal = torch.triu(torch.ones(T, T), diagonal=1).bool()
scores.masked_fill_(causal, float('-inf'))
```

## RMSNorm

```python
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
```

## RoPE (Rotary Position Embedding)

```python
def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
```

## SwiGLU FFN (Llama-style)

```python
class SwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.val  = nn.Linear(d, d_ff, bias=False)
        self.proj = nn.Linear(d_ff, d, bias=False)
    def forward(self, x):
        return self.proj(F.silu(self.gate(x)) * self.val(x))
```

## KV cache (inference)

```python
K_cache = torch.cat([K_cache, k_new], dim=-2)    # grow along T dim
V_cache = torch.cat([V_cache, v_new], dim=-2)
attn = F.softmax(q_new @ K_cache.T / d_k**0.5, dim=-1) @ V_cache
# O(T) per step; memory grows linearly with T
```

## Decoder block (Pre-LN)

```python
class DecoderBlock(nn.Module):
    def __init__(self, d, h, d_ff):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
        self.attn = MultiHeadAttention(d, h)
        self.ffn  = SwiGLU(d, d_ff)
    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.ffn(self.norm2(x))
        return x
```

---

# 19. Probability Programming Patterns

## Optimal stopping (backward induction)

```python
def expected_optimal_dice(n: int) -> float:
    E = 3.5
    for _ in range(n - 1):
        E = sum(max(v, E) for v in range(1, 7)) / 6
    return E
# n=2 → 4.25, n=3 → 4.667, n→∞ → 6
```

## Majority vote probability

```python
from math import comb
def majority_prob(n: int, p: float) -> float:
    return sum(comb(n, k) * p**k * (1-p)**(n-k) for k in range(n//2+1, n+1))
```

## Top-p nucleus sampling

```python
def sample(logits, temperature=1.0, top_p=0.9):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    sorted_p, sorted_i = probs.sort(descending=True)
    cum = sorted_p.cumsum(-1)
    mask = cum > top_p
    mask[..., 1:] = mask[..., :-1].clone(); mask[..., 0] = False
    sorted_p[mask] = 0
    sorted_p /= sorted_p.sum(-1, keepdim=True)
    idx = torch.multinomial(sorted_p, 1)
    return sorted_i.gather(-1, idx)
```

## Reservoir sampling (stream, maintain k samples)

```python
import random
def reservoir_sample(stream, k):
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir
```

## Rand7 from Rand5 (rejection sampling)

```python
def rand7():
    while True:
        x = (rand5() - 1) * 5 + (rand5() - 1)   # uniform [0, 24]
        if x < 21:                                  # reject 21-24
            return x % 7 + 1
```

## Box-Muller (Uniform → Normal)

```python
import math, random
def box_muller():
    u1, u2 = random.random(), random.random()
    z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    z1 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
    return z0, z1  # both ~ N(0, 1)
```

---

# PART IV — BEHAVIORAL

---

# 20. Interview Strategy & Communication

## Coding problem rhythm

1. **Restate** the problem in your own words
2. **Examples** — give one yourself; ask for edge cases
3. **Brute force** first; state complexity
4. **Optimize** — narrate the insight
5. **Code** — clean variable names, small functions
6. **Trace** — walk through one example step by step
7. **Discuss** complexity and alternatives

## Probability problem rhythm

1. **Define your random variable**: "Let X = ..."
2. **State independence / distribution assumptions**
3. **Pick the tool**: linearity / Bayes / conditioning / indicator method
4. **Compute**, then **sanity-check** (e.g., E[max] + E[min] = E[sum])
5. **For stopping problems**: verbalize the threshold rule

## Stats/A-B testing rhythm

1. **Define H₀ and H₁** explicitly
2. **State α and desired power**
3. **Calculate sample size** before running
4. **Identify potential confounds** (Simpson's paradox, selection bias)
5. **Report effect size**, not just p-value

## STAR format (behavioral)

> **S**ituation → **T**ask → **A**ction → **R**esult (with a metric)

## "Explain X" template

1. **One-sentence definition**
2. **Why it matters / when used**
3. **Concrete example**
4. **Trade-offs / failure modes**

## Power phrases

- "Let me clarify the input format..."
- "I'll start with brute force O(n²), then optimize."
- "By linearity of expectation, even though variables are dependent..."
- "This is backward induction — let me define V[k]..."
- "Two spiral patterns: shrinking boundaries vs expanding step."
- "I'm assuming independence here — worth noting if that breaks."
- "p-value is the probability of data this extreme **given H₀**, not the probability H₀ is true."

## Questions to ask interviewers

- "What does a typical on-call / production-ML burden look like?"
- "What's the team's ratio of research to applied work?"
- "How do you measure success in this role at 6 months?"
- "What's the hardest engineering problem the team solved this year?"

---

# 21. Appendix — Must-Know Numbers

## Mathematical constants

| Constant | Value | Use |
|---|---|---|
| e | 2.71828 | Secretary cutoff: n/e |
| 1/e | 0.368 | Secretary P(success) |
| 1 − 1/e | 0.632 | Fraction seen in bootstrap |
| ln 2 | 0.693 | Exponential half-life |
| π²/6 | 1.645 | Coupon collector variance coefficient |
| Euler γ | 0.577 | Harmonic number offset |
| √(2ln2) | 1.177 | FWHM/σ half-factor |
| 2√(2ln2) | 2.355 | FWHM = 2.355σ for Gaussian |

## Standard normal critical values

| α (two-tailed) | z_{α/2} | Use for CI |
|---|---|---|
| 0.10 | 1.645 | 90% CI |
| 0.05 | **1.960** | **95% CI** ← memorize |
| 0.01 | 2.576 | 99% CI |

## Sample size rules of thumb

| Test | Minimum n (each group) |
|---|---|
| CTR 10%→12%, α=0.05, power=80% | ~3,800 |
| CTR 10%→11%, α=0.05, power=80% | ~14,700 |
| Rough rule: (2.8/Δ)² · p(1-p) | — |
| CLT kicks in | n ≥ 30 |

## Model architecture landmarks

| Model | Params | d_model | Layers | Heads |
|---|---|---|---|---|
| BERT-base | 110M | 768 | 12 | 12 |
| GPT-2 small | 117M | 768 | 12 | 12 |
| GPT-3 | 175B | 12288 | 96 | 96 |
| Llama-2 7B | 7B | 4096 | 32 | 32 |
| Llama-2 70B | 70B | 8192 | 80 | 64 |

## Optimizer defaults

| Optimizer | lr | β₁ | β₂ | ε |
|---|---|---|---|---|
| Adam | 3e-4 | 0.9 | 0.999 | 1e-8 |
| AdamW | 1e-4 | 0.9 | 0.999 | 1e-8 |
| SGD w/ momentum | 0.01 | 0.9 | — | — |

---

> **Final reminder:** When stuck, **say what you're thinking out loud**. Interviewers grade reasoning as much as the final answer. Name your variables, state your invariants, sanity-check small cases, and acknowledge edge conditions before moving on.
>
> **Good luck!**
