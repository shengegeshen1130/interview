# ML 工程师面试 — 速查总表 (Master Cheatsheet)

一站式速查：公式、代码、一句话答案。**面试前扫一遍，面试中随时回忆。**

> **万能提示 Universal Tip：** (1) 先问清题意 (2) 说明假设 (3) 写代码前先口述思路 (4) 用小例子验证 (5) 报时间/空间复杂度

---

## 目录 Table of Contents

### 第一部分 — 数学、概率与统计 *(非编程 Non-Coding)*

1. [概率核心公式 Probability — Core Identities](#1-概率核心公式-probability--core-identities)
2. [分布速查表 Distributions Reference Table](#2-分布速查表-distributions-reference-table)
3. [期望、方差、协方差、相关系数 E / Var / Cov / Corr](#3-期望方差协方差相关系数-e--var--cov--corr)
4. [正态分布全角度 Gaussian / Normal — All Angles](#4-正态分布全角度-gaussian--normal--all-angles)
5. [统计推断基础 Statistics — Estimation & CLT](#5-统计推断基础-statistics--estimation--clt)
6. [假设检验与 A/B 测试 Hypothesis Testing & A/B Testing](#6-假设检验与-ab-测试-hypothesis-testing--ab-testing)
7. [期望值解题技巧 Expected Value — Problem-Solving Tricks](#7-期望值解题技巧-expected-value--problem-solving-tricks)
8. [最优停止决策模板 Optimal Stopping — Decision Templates](#8-最优停止决策模板-optimal-stopping--decision-templates)
9. [必背答案速查 Quick Reference — Must-Know Answers](#9-必背答案速查-quick-reference--must-know-answers)

### 第二部分 — ML 与系统概念 *(非编程 Non-Coding)*

10. [Transformer 架构公式与变体](#10-transformer-架构公式与变体-formulas--variants)
11. [深度学习速查 Deep Learning Cookbook](#11-深度学习速查-deep-learning-cookbook)
12. [LLM 专项概念 LLM-specific Concepts](#12-llm-专项概念-llm-specific-concepts)
13. [ML 系统与生产 ML Systems & Production](#13-ml-系统与生产-ml-systems--production)
14. [多数投票与集成理论 Majority Vote / Ensemble — Theory](#14-多数投票与集成理论-majority-vote--ensemble--theory)

### 第三部分 — 编程 Coding

15. [矩阵算法 Matrix Algorithms — Spiral, Rotation, Diagonal](#15-矩阵算法-matrix-algorithms--spiral-rotation-diagonal)
16. [编程模式 DSA Coding Patterns](#16-编程模式-dsa-coding-patterns)
17. [PyTorch 要点 PyTorch Essentials](#17-pytorch-要点-pytorch-essentials)
18. [Transformer 代码实现 Code Implementations](#18-transformer-代码实现-code-implementations)
19. [概率编程模板 Probability Programming Patterns](#19-概率编程模板-probability-programming-patterns)

### 第四部分 — 行为面试 Behavioral

20. [面试策略与沟通 Interview Strategy & Communication](#20-面试策略与沟通-interview-strategy--communication)
21. [附录：必背数字 Appendix — Must-Know Numbers](#21-附录必背数字-appendix--must-know-numbers)

---

# 第一部分 — 数学、概率与统计

---

# 1. 概率核心公式 Probability — Core Identities

```
贝叶斯定理 Bayes:        P(A|B) = P(B|A)·P(A) / P(B)
全概率公式 Total prob:   P(B)   = Σᵢ P(B|Aᵢ)·P(Aᵢ)
补集 Complement:         P(Aᶜ)  = 1 − P(A)
容斥原理 Incl-Excl:      P(A∪B) = P(A) + P(B) − P(A∩B)
                         P(A∪B∪C) = ΣP(Aᵢ) − ΣP(Aᵢ∩Aⱼ) + P(A∩B∩C)
独立性 Independence:     P(A∩B) = P(A)·P(B)
条件概率 Conditional:    P(A|B) = P(A∩B) / P(B)
```

**条件独立 ≠ 边际独立 Conditional independence vs marginal independence：**
- X ⊥ Y | Z **不能推出** X ⊥ Y（反之亦然）
- 反例：硬币 C 决定 X₁, X₂ 是否相关

**贝叶斯比数形式 Odds form of Bayes（对数似然比）：**
```
后验比 Posterior odds = 先验比 Prior odds × 似然比 Likelihood ratio
P(H|E)/P(Hᶜ|E) = P(H)/P(Hᶜ) × P(E|H)/P(E|Hᶜ)
```

---

# 2. 分布速查表 Distributions Reference Table

| 分布 Distribution | PMF / PDF | 期望 E[X] | 方差 Var(X) | 关键性质 Key Property |
|---|---|---|---|---|
| **Bernoulli(p)** | P(X=1)=p | p | p(1−p) | 单次试验 Single trial |
| **Binomial(n,p)** | C(n,k)pᵏ(1−p)ⁿ⁻ᵏ | np | np(1−p) | n 次伯努利 |
| **Geometric(p)** | (1−p)ᵏ⁻¹p | 1/p | (1−p)/p² | 无记忆性（离散）|
| **Negative Binomial(r,p)** | — | r/p | r(1−p)/p² | 等第 r 次成功 |
| **Hypergeometric(N,K,n)** | C(K,k)C(N−K,n−k)/C(N,n) | nK/N | — | 不放回抽样 |
| **Poisson(λ)** | e⁻λλᵏ/k! | λ | λ | 期望=方差=λ |
| **Uniform{1..n}** | 1/n | (n+1)/2 | (n²−1)/12 | 离散均匀 |
| **Uniform[a,b]** | 1/(b−a) | (a+b)/2 | (b−a)²/12 | 连续均匀 |
| **Exponential(λ)** | λe⁻λˣ | 1/λ | 1/λ² | 无记忆性（连续）|
| **Normal(μ,σ²)** | 见 §4 | μ | σ² | CLT 的极限 |
| **Beta(α,β)** | xᵅ⁻¹(1−x)^{β−1}/B(α,β) | α/(α+β) | αβ/((α+β)²(α+β+1)) | Bernoulli 共轭先验 |
| **Gamma(α,β)** | xᵅ⁻¹e⁻ˣ/ᵝ/Γ(α)βᵅ | αβ | αβ² | 指数分布之和 |
| **Chi-squared(k)** | Gamma(k/2, 2) | k | 2k | k 个标准正态的平方和 |

**近似关系 Approximations：**
- **Poisson ≈ Binomial**：n 大、p 小、np = λ 适中时
- **Normal ≈ Binomial**：n 大（CLT），μ=np，σ²=np(1-p)

---

# 3. 期望、方差、协方差、相关系数 E / Var / Cov / Corr

## 3.1 期望 Expected Value E[X]

```
离散 Discrete:    E[X] = Σₓ x · P(X = x)
连续 Continuous:  E[X] = ∫₋∞^∞ x · f(x) dx

LOTUS（无意识统计学家定律）:
  E[g(X)] = Σₓ g(x)·P(X=x)      （离散）
  E[g(X)] = ∫ g(x)·f(x) dx      （连续）
```

**期望的性质 Properties of E：**

| 性质 Rule | 公式 Formula | 备注 Note |
|---|---|---|
| 线性 Linearity | E[aX + bY + c] = aE[X] + bE[Y] + c | **永远成立，不需要独立 Always — no independence needed** |
| 常数 Constant | E[c] = c | — |
| 乘积（独立时）Product (indep) | E[XY] = E[X]·E[Y] | **仅当 X ⊥ Y 时成立 Only if independent** |
| 塔性质 Tower property | E[X] = E[E[X\|Y]] | 多阶段问题利器 |
| E[X²] 关系 | E[X²] = Var(X) + (E[X])² | 由方差公式变形 |

## 3.2 方差 Variance Var(X)

```
定义 Definition:   Var(X) = E[(X − μ)²]           μ = E[X]
计算捷径 Shortcut: Var(X) = E[X²] − (E[X])²        ← 手算时永远用这个
标准差 Std dev:    SD(X) = σ = √Var(X)
```

**方差的性质 Properties of Var：**

| 性质 Rule | 公式 Formula | 备注 Note |
|---|---|---|
| 缩放 Scale | Var(aX) = a²·Var(X) | 常数要平方 |
| 平移 Shift | Var(X + c) = Var(X) | 常数不改变离散程度 |
| 线性组合 | Var(aX + bY) = a²Var(X) + b²Var(Y) + 2ab·Cov(X,Y) | 通用公式 |
| 独立时 If X ⊥ Y | Var(X + Y) = Var(X) + Var(Y) | Cov = 0 |
| 样本方差 Sample var | s² = Σ(xᵢ − x̄)²/(n−1) | 无偏；分母用 n−1 |
| Eve 定律（全方差）| Var(X) = E[Var(X\|Y)] + Var(E[X\|Y]) | 组内方差期望 + 组间均值方差 |

## 3.3 协方差 Covariance Cov(X, Y)

```
定义 Definition: Cov(X,Y) = E[(X−μX)(Y−μY)]
计算捷径 Shortcut: Cov(X,Y) = E[XY] − E[X]·E[Y]   ← 用这个
```

**协方差的性质 Properties of Cov：**

| 性质 Rule | 公式 Formula |
|---|---|
| 对称 Symmetric | Cov(X,Y) = Cov(Y,X) |
| 自协方差 Self | Cov(X,X) = Var(X) |
| 双线性 Bilinear | Cov(aX+b, cY+d) = ac·Cov(X,Y) |
| 分配律 Distributive | Cov(X+Y, Z) = Cov(X,Z) + Cov(Y,Z) |
| 独立 → Cov=0 | X ⊥ Y → Cov(X,Y) = 0 **（反之不成立！）** |
| 方差展开 | Var(X+Y) = Var(X) + Var(Y) + 2Cov(X,Y) |
| n 个变量之和方差 | Var(ΣXᵢ) = ΣVar(Xᵢ) + 2Σᵢ<ⱼ Cov(Xᵢ,Xⱼ) |

**重要陷阱：** Cov(X,Y) = 0 **不代表独立**。反例：X ~ U(-1,1)，Y = X²，Cov=0 但完全非线性相关。

## 3.4 相关系数 Correlation Coefficient ρ (Pearson)

```
ρ(X,Y) = Cov(X,Y) / (SD(X)·SD(Y))     取值范围 Range: [−1, 1]
```

**解读 Interpretation：**

| ρ 值 | 含义 Meaning |
|---|---|
| ρ = +1 | 完全正线性关系：Y = aX + b，a > 0 |
| ρ = −1 | 完全负线性关系：Y = aX + b，a < 0 |
| ρ = 0 | 不相关 Uncorrelated —— **但不一定独立！** |
| 0 < ρ < 1 | 正线性相关 Positive linear association |
| −1 < ρ < 0 | 负线性相关 Negative linear association |

**Pearson vs Spearman：**

| | Pearson | Spearman |
|---|---|---|
| 衡量 Measures | 线性关系 Linear | 单调关系 Monotone |
| 数据 Data | 原始值 Raw values | 秩 Ranks |
| 对异常值鲁棒 Outlier robust | 否 No | 是 Yes |
| 适用场景 | 正态、线性 Normal, linear | 偏态、序数、非线性单调 |

**样本相关系数 Sample correlation：**
```
r = Σ(xᵢ−x̄)(yᵢ−ȳ) / √[Σ(xᵢ−x̄)² · Σ(yᵢ−ȳ)²]
```

## 3.5 矩母函数 Moment Generating Function (MGF)

```
M_X(t) = E[e^{tX}]
E[Xⁿ] = M_X^{(n)}(0)      （对 t 求 n 阶导，令 t=0）
```

| 分布 | MGF |
|---|---|
| Bernoulli(p) | 1 − p + pe^t |
| Binomial(n,p) | (1 − p + pe^t)ⁿ |
| Poisson(λ) | exp(λ(eᵗ − 1)) |
| Normal(μ,σ²) | exp(μt + σ²t²/2) |
| Exponential(λ) | λ/(λ−t)，t < λ |

---

# 4. 正态分布全角度 Gaussian / Normal — All Angles

## 4.1 核心公式 Core Formulas

```
PDF（概率密度函数）:
  f(x; μ, σ²) = (1 / √(2πσ²)) · exp(−(x−μ)² / (2σ²))

标准正态 Standard N(0,1):
  φ(x) = (1/√(2π)) · exp(−x²/2)
  Φ(x) = P(Z ≤ x) = ∫₋∞ˣ φ(t) dt      （无闭合形式 no closed form）

标准化 Standardize:  Z = (X − μ) / σ  ~  N(0, 1)
```

## 4.2 关键分位数（必背）Key Quantiles — Memorize

| 使用场景 Scenario | z 值 | 含义 Meaning |
|---|---|---|
| 90% 置信区间 → z₀.₀₅ | **1.645** | P(Z > 1.645) = 5% |
| 95% 置信区间 → z₀.₀₂₅ | **1.960** | P(Z > 1.96) = 2.5% |
| 99% 置信区间 → z₀.₀₀₅ | **2.576** | P(Z > 2.576) = 0.5% |
| 1σ 规则 | ±1.0 | P(-1 ≤ Z ≤ 1) ≈ **68.3%** |
| 2σ 规则 | ±2.0 | P(-2 ≤ Z ≤ 2) ≈ **95.4%** |
| 3σ 规则 | ±3.0 | P(-3 ≤ Z ≤ 3) ≈ **99.7%** |

## 4.3 性质 Properties

```
线性变换 Linear transform:  X ~ N(μ,σ²)  →  aX+b ~ N(aμ+b, a²σ²)
独立之和 Sum of indep:       X+Y ~ N(μ₁+μ₂, σ₁²+σ₂²)
缩放 Scaling:                X ~ N(0,1)  →  μ + σX ~ N(μ, σ²)
对称性 Symmetry:             φ(−x) = φ(x)，Φ(−x) = 1 − Φ(x)
矩母函数 MGF:                M_X(t) = exp(μt + σ²t²/2)
```

## 4.4 正态分布的多角度解读 Multiple Angles

| 角度 Angle | 核心论述 Statement |
|---|---|
| **概率论 / CLT** | n 个 i.i.d. 随机变量之和 → N(nμ, nσ²)；标准化后 → N(0,1) |
| **信息论 Information theory** | 在固定均值和方差的所有分布中，正态分布具有**最大熵 maximum entropy** |
| **贝叶斯 Bayesian** | 均值的**共轭先验 conjugate prior**（方差已知时）；N × N → N |
| **高维几何 Geometry (high-d)** | ℝᵈ 中 N(0,I)：几乎所有质量集中在半径 √d 的薄壳上；分量近似正交 |
| **物理 / 扩散 Physics / Diffusion** | 热方程的解：布朗运动在时刻 t 的密度为 N(0, t) |
| **对数空间 Log-space** | Log-normal：若 log X ~ N(μ,σ²)，则 X = eˣ 是右偏分布 |
| **抽样 Sampling** | Box-Muller 变换：Z₁=√(−2ln U₁)·cos(2πU₂)，Z₂=√(−2ln U₁)·sin(2πU₂) 均为 N(0,1) |

## 4.5 FWHM（半高全宽）

```
y = exp(−(x−μ)²/(2σ²))
最大值在 x = μ 处：y_max = 1

半最大值对应位置：x = μ ± σ√(2 ln 2)
FWHM = 2σ√(2 ln 2) ≈ 2.355σ
```

| 函数 Function | FWHM |
|---|---|
| 高斯 Gaussian exp(−x²/2σ²) | 2√(2ln2)·σ ≈ **2.355σ** |
| 柯西 Cauchy/Lorentz 1/(1+(x/γ)²) | **2γ** |
| 拉普拉斯 Laplace exp(−\|x\|/b) | 2b·ln2 ≈ **1.386b** |

## 4.6 二元正态与角度：arcsin 公式 Bivariate Normal & Angles

设 X, Y 为**标准二元正态**，相关系数 ρ，几何夹角 θ = arccos(ρ)：

```
核心公式 Core formula:
  P(X > 0, Y > 0) = 1/4 + arcsin(ρ) / (2π)

等价角度形式 Angle form（θ = arccos(ρ)）:
  P(X > 0, Y > 0) = (π − θ) / (2π)
  P(同号 same sign)  = (π − θ) / π  =  1 − θ/π
  P(异号 diff sign)  =      θ / π   =  arccos(ρ) / π
```

**三个边界验证（必查 sanity check）：**

| ρ（θ）| P(同号) | 直觉 |
|---|---|---|
| ρ=1（θ=0）| 1 | 完全正相关，永远同号 |
| ρ=0（θ=π/2）| 1/2 | 独立，同号概率 50% |
| ρ=-1（θ=π）| 0 | 完全负相关，永远异号 |

**几何直觉：** 将 (X, Y) 写成 2D 各向同性高斯点在夹角为 θ 的两个方向上的投影，随机点落在"两轴同侧扇形"的概率 = 扇形弧长 (π−θ) / 总周长 2π。

**应用：随机超平面哈希 LSH (SimHash)**

```
哈希函数：h(x) = sign(w · x)，  w ~ N(0, I)

P(h(x) ≠ h(y)) = θ / π = arccos(cos θ) / π

其中 θ = 向量 x 和 y 的夹角
→ 角度越小（越相似），哈希碰撞概率越大 ✓
```

这是 **局部敏感哈希（Locality-Sensitive Hashing）** 中 SimHash / 随机投影的理论基础。

---

# 5. 统计推断基础 Statistics — Estimation & CLT

## 5.1 样本统计量 Sample Statistics

```
样本均值 Sample mean:      x̄  = (1/n) Σ xᵢ                  （无偏：E[x̄] = μ）
样本方差 Sample variance:  s²  = Σ(xᵢ − x̄)² / (n−1)          （无偏；除以 n−1）
标准误差 Standard error:   SE  = σ/√n  ≈ s/√n                 （x̄ 的不确定性）
```

**为什么除以 n-1？（Bessel 修正）** 用了 1 个自由度估计 μ，只剩 n-1 个自由度，除以 n 会低估方差。

## 5.2 中心极限定理 Central Limit Theorem (CLT)

```
X₁, X₂, ..., Xₙ i.i.d.，均值 μ，方差 σ²

CLT:  √n · (X̄ − μ) / σ  →  N(0, 1)   当 n → ∞
即:   X̄  ~  N(μ,  σ²/n)               大样本时（通常 n ≥ 30）
```

**CLT vs 大数定律 Law of Large Numbers (LLN)：**

| | LLN 大数定律 | CLT 中心极限定理 |
|---|---|---|
| 结论 Says | X̄ **收敛到** μ（值收敛）| (X̄−μ) 的**分布**收敛到正态 |
| 类型 Type | 值的收敛 | 分布形状的收敛 |
| 比喻 | "平均值会趋向真值" | "平均值的误差是钟形曲线" |

## 5.3 最大似然估计 Maximum Likelihood Estimation (MLE)

```
似然函数 Likelihood:     L(θ) = P(data | θ) = Πᵢ f(xᵢ; θ)
对数似然 Log-likelihood: ℓ(θ) = log L(θ) = Σᵢ log f(xᵢ; θ)
MLE:                     θ̂ = argmax ℓ(θ)  →  令 dℓ/dθ = 0
```

**常见 MLE 结果 Common MLEs：**

| 模型 Model | MLE 结果 |
|---|---|
| Bernoulli(p) | p̂ = k/n（样本频率）|
| Normal(μ,σ²) | μ̂ = x̄，σ̂² = Σ(xᵢ−x̄)²/n（**有偏！biased**）|
| Exponential(λ) | λ̂ = 1/x̄ |
| Poisson(λ) | λ̂ = x̄ |

**MLE vs MAP（最大后验估计）：**

| | MLE | MAP |
|---|---|---|
| 目标函数 Objective | max P(data\|θ) | max P(θ\|data) = P(data\|θ)·P(θ) |
| 先验 Prior | 忽略 ignored | 纳入 included |
| 小样本稳定性 Small-n | 不稳定，易过拟合 | 稳定（先验起正则化作用）|
| 伯努利例子（7正/10次）| p̂=0.7 | p̂=(7+α)/(10+α+β)，Beta(α,β) 先验 |

## 5.4 置信区间 Confidence Intervals

```
均值（σ 已知）95% CI:     x̄  ±  1.96 · σ/√n
均值（σ 未知）95% CI:     x̄  ±  t_{0.025, n-1} · s/√n
比例 95% CI:              p̂  ±  1.96 · √(p̂(1−p̂)/n)
```

**正确解读（非常重要！）：** "如果用同样的方法反复抽样并建立区间，约 95% 的区间会包含真实参数。" 真实参数是固定的，区间是随机的。

**常见错误解读：**
- ❌ "有 95% 的概率 μ 在 (a, b) 内" —— μ 是固定值，不是随机变量
- ❌ "95% 的数据点在置信区间内" —— CI 是关于均值的，不是数据分布

**影响区间宽度的因素：**
- 更宽的 CI ← 更小的 n、更大的 σ、更高的置信水平、p 更接近 0.5

## 5.5 偏差、方差与 MSE Bias, Variance & MSE

```
偏差 Bias(θ̂)   = E[θ̂] − θ
MSE(θ̂)         = Var(θ̂) + Bias(θ̂)²
无偏 Unbiased:  E[θ̂] = θ   （Bias = 0）
```

ML 中的偏差-方差权衡：正则化增大偏差但降低方差，总体 MSE 可能降低。

---

# 6. 假设检验与 A/B 测试 Hypothesis Testing & A/B Testing

## 6.1 检验框架 Framework

```
H₀：原假设 null hypothesis（默认，我们试图推翻它）
H₁：备择假设 alternative hypothesis（我们试图证明的）

α   = P(第一类错误 Type I error)  = P(拒绝 H₀ | H₀ 为真)    ← 显著性水平，通常设 0.05
β   = P(第二类错误 Type II error) = P(不拒绝 H₀ | H₁ 为真)
Power（检验效能）= 1 − β                                      ← 检验的灵敏度
```

**决策矩阵 Decision Matrix：**

| | H₀ 为真 H₀ true | H₀ 为假 H₀ false |
|---|---|---|
| **拒绝 H₀ Reject** | 第一类错误 Type I Error (α)，假阳性 | 正确 Correct (Power = 1−β) |
| **不拒绝 Fail to reject** | 正确 Correct (1−α) | 第二类错误 Type II Error (β)，假阴性 |

## 6.2 p 值 p-value

```
p-value = P（在 H₀ 为真的前提下，出现当前或更极端结果的概率）

当 p < α 时拒绝 H₀
```

**p 值不是（常见误解）：**
- ❌ H₀ 为真的概率
- ❌ 你的发现是假的概率
- ❌ 效应大小或实际显著性的衡量

## 6.3 常用检验统计量 Common Test Statistics

| 检验 Test | 统计量 Statistic | H₀ 下分布 | 使用场景 |
|---|---|---|---|
| 单样本 z 检验 | z = (x̄−μ₀)/(σ/√n) | N(0,1) | σ 已知 |
| 单样本 t 检验 | t = (x̄−μ₀)/(s/√n) | t_{n-1} | σ 未知 |
| 双比例 z 检验 | z = (p̂₁−p̂₂)/SE | N(0,1) | n 较大 |
| 双样本 t 检验 | t = (x̄₁−x̄₂)/SE | t_{n₁+n₂-2} | n 小，σ 未知 |
| 卡方检验 Chi-squared | χ² = Σ(O−E)²/E | χ²_{k-1} | 类别数据，拟合优度 |
| 方差分析 ANOVA (F-test) | F = MSB/MSW | F_{k-1, N-k} | 比较 ≥3 组均值 |

## 6.4 A/B 测试设计清单 Design Checklist

**第一步：定义假设 Define hypothesis**
```
H₀: p₁ = p₂  （转化率无差异）
H₁: p₁ ≠ p₂  （双尾；只有方向明确时才用单尾）
```

**第二步：样本量计算 Sample size calculation**
```
双比例检验（最常用）:

n = (z_{α/2} + z_β)² · [p₁(1−p₁) + p₂(1−p₂)] / (p₁ − p₂)²

常用参数:  α=0.05  →  z_{α/2}=1.96
           Power=80% →  z_β=0.84   → (1.96+0.84)²≈7.84
           Power=90% →  z_β=1.28   → (1.96+1.28)²≈10.5
```

**示例 Example：** p₁=10%，p₂=12%，α=0.05，power=80%
```
n = 7.84 × (0.1×0.9 + 0.12×0.88) / (0.02)² = 7.84 × 0.1956 / 0.0004 ≈ 3,832 per group
```

**第三步：实验要求 Run the experiment**

| 要求 Requirement | 原因 Why |
|---|---|
| 预先注册 n 和 α | 防止偷窥（p-hacking / optional stopping）|
| 随机分配 Randomize | 消除混淆变量 |
| 一次只改一个变量 | 因果归因需要单一变量 |
| 跑完设计时长 Full duration | 消除周期性效应（如工作日/周末）|
| 检查样本比例失配 SRM | 发现实验 bug |

**第四步：常见陷阱 Pitfalls**

| 陷阱 Pitfall | 问题 Problem | 修复 Fix |
|---|---|---|
| 早停 Early stopping / Peeking | 膨胀第一类错误 | Sequential testing (mSPRT) 或严格执行预设 n |
| 多指标 Multiple metrics | FWER 膨胀 | 预设主指标；次要指标用 FDR |
| 网络效应 Spillover | 控制组/实验组渗透 | Cluster randomization |
| 辛普森悖论 Simpson's Paradox | 分层后方向反转 | 按分群分层检验 |
| 新奇效应 Novelty effect | 短期激增后回落 | 延长实验周期；衡量持续行为 |

## 6.5 多重检验 Multiple Testing

```
m 个独立检验，FWER（族错误率）:
P(至少1个假阳性) = 1 − (1−α)^m

m=20，α=0.05:  FWER = 1 − 0.95²⁰ ≈ 64.2% ！！
```

| 方法 Method | 控制目标 Controls | 公式 Formula | 保守程度 Conservative |
|---|---|---|---|
| **Bonferroni** | FWER | α' = α/m | 最保守 Very（用于验证性研究）|
| **Holm-Bonferroni** | FWER | 逐步 Bonferroni | 稍弱 |
| **Benjamini-Hochberg (BH)** | FDR | 排序后 p_{(i)} ≤ (i/m)·α | 最宽松（用于探索性研究）|

**FDR（False Discovery Rate）** = 被拒绝的 H₀ 中实际为假阳性的期望比例。适用于基因组学、多指标 dashboard。

## 6.6 提升统计效能 What Increases Power

```
增大效能的方法：
1. 更大的 n              → SE 缩小 → 更容易检测效应
2. 更大的效应量 Δ = |μ₁−μ₂| → 信号更强
3. 更小的 σ（降低噪声）  → 信噪比提升
4. 更大的 α              → 代价是更多假阳性
5. 单尾检验 One-tailed   → 代价是需要先验方向知识
```

---

# 7. 期望值解题技巧 Expected Value — Problem-Solving Tricks

## 技巧1 — 线性期望 Linearity（不需要独立）

```
E[aX + bY + c] = aE[X] + bE[Y] + c   ← 永远成立 always true

经典题：n 个人的帽子随机排列，平均拿到自己帽子的期望人数 = 1
（n 个示性变量，每个期望值 = 1/n，加总 = 1）
```

## 技巧2 — 条件第一步 Condition on First Step

```
"期望正面次数（硬币正面概率 p）":
  E[X] = 1·p + (1 + E[X])·(1−p)  →  E[X] = 1/p

模板: E[X] = (即时收益) + (继续的概率) × E[X]
```

## 技巧3 — 示性变量 Indicator Variables

```
"n 次抽取（k 种）得到的不同优惠券期望种数":
  Iⱼ = 1 当且仅当 coupon j 至少出现一次
  E[Iⱼ] = 1 − ((k−1)/k)^n
  E[不同种数] = k · (1 − ((k−1)/k)^n)
```

## 技巧4 — Wald 恒等式 Wald's Identity

```
Xᵢ i.i.d.，均值 μ；N 为停止时间，E[N] < ∞:
  E[X₁ + X₂ + ... + X_N] = μ · E[N]
```

## 技巧5 — Eve 定律（全方差公式）Eve's Law

```
Var(X) = E[Var(X|Y)] + Var(E[X|Y])
       = 组内方差的期望 + 组间均值的方差

复合泊松 Compound Poisson：N~Pois(λ)，每个 Xᵢ 均值 μ，方差 σ²:
  E[S] = λμ
  Var[S] = λ(μ² + σ²) = λ E[X²]
```

## 技巧6 — 顺序统计量 Order Statistics

```
X₁,...,Xₙ ~ U(0,1) i.i.d.，X_(k) = 第 k 小:
  E[X_(k)] = k/(n+1)
  E[最大值 max] = n/(n+1)
  E[最小值 min] = 1/(n+1)

一般形式: X_(k) ~ Beta(k, n−k+1)
```

## 技巧7 — 尾概率求期望 E[X] via CDF

```
非负整数随机变量:  E[X] = Σ_{k=0}^∞ P(X > k)
非负连续随机变量:  E[X] = ∫₀^∞ P(X > t) dt

适用于：等待时间、停止时间等，常使几何级数更容易计算
```

---

# 8. 最优停止决策模板 Optimal Stopping — Decision Templates

## 通用递推 Universal Recurrence

```
V[k] = 剩余 k 次决策机会时的期望最优值
V[1] = 基础情况（被迫执行最后一次）
V[k] = E[ max(立即收益, V[k−1]) ]
```

## 秘书问题 Secretary Problem（1/e 规则）

```
拒绝前 ⌊n/e⌋ ≈ 0.368n 个候选人
之后选第一个优于所有已见过的
P(成功选到最佳) → 1/e ≈ 0.368
```

## 先知不等式 Prophet Inequality

```
单阈值 τ：设 P(max Xᵢ ≥ τ) = 1/2
可保证期望收益 ≥ 0.5 · E[max Xᵢ]
```

## 骰子游戏 Dice Game（n 次投掷，保留最高值）

```
V[1] = 3.5
V[k] = (1/6) Σᵥ max(v, V[k−1])

n=2 → V=4.25，第一次 ≥5 则停（即 5 或 6）
n=3 → V=4.67，第一次 ≥5 停；第二次 ≥4 停；否则用第三次
```

---

# 9. 必背答案速查 Quick Reference — Must-Know Answers

| 问题 Question | 答案 Answer |
|---|---|
| 公平骰子 E[X] | 3.5 |
| 公平骰子 Var(X) | 35/12 ≈ 2.917 |
| E[两骰子最大值 max] | 161/36 ≈ 4.47 |
| E[两骰子最小值 min] | 91/36 ≈ 2.53（两者之和 = 7 ✓）|
| 期望抛几次出现第一个正面 | 2 |
| 期望抛几次出现 HHH | 14 |
| 期望抛几次出现 HTH | 10 |
| HH vs TH，哪个序列先出现？| HH 平均 14 次，TH 平均 8 次；TH 先出现概率 3/4 |
| 优惠券收集（n 种）期望抽数 | n·Hₙ ≈ n ln n + 0.577n |
| 优惠券收集方差 | π²n²/6 |
| 赌徒破产胜率（位置 i，范围 [0,N]）| i/N（对称时）|
| 赌徒破产期望步数 | i(N−i) |
| 秘书问题最优截止点 | n/e ≈ 0.368n |
| 3 个独立分类器各 80%，多数投票准确率 | 0.896 |
| 两信封悖论结论 | 换与不换期望相同，均为 3m/2 |
| 辛普森悖论 Simpson's Paradox | 分层后趋势可能与汇总完全相反（存在混淆变量时）|

---

# 第二部分 — ML 与系统概念 *(非编程)*

---

# 10. Transformer 架构公式与变体 Formulas & Variants

## 核心公式 Core Formulas

```
缩放点积注意力 Scaled dot-product attention:
  Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V

多头注意力 Multi-head:
  MultiHead(Q,K,V) = Concat(head₁,...,headₕ) · Wᴼ
  headᵢ = Attention(Q·Wᵢᵠ, K·Wᵢᴷ, V·Wᵢᵛ)

正弦位置编码 Sinusoidal PE:
  PE(pos,2i)   = sin(pos / 10000^{2i/d})
  PE(pos,2i+1) = cos(pos / 10000^{2i/d})
```

## 注意力头变体 Attention Head Variants

| 变体 Variant | Q 头数 | K/V 头数 | 使用场景 |
|---|---|---|---|
| MHA（标准）| H | H | 小模型 |
| MQA | H | 1 | 推理速度优先（PaLM, Falcon）|
| GQA | H | G（G<H）| 均衡方案（Llama 2/3）|
| Cross-attention | 来自 decoder | 来自 encoder | Encoder-decoder 架构 |

## 位置编码 Positional Encodings

| 类型 Type | 原理 Idea | 优劣 Pros/Cons |
|---|---|---|
| 正弦 Sinusoidal | 固定 sin/cos | 外推差 |
| 可学习绝对 Learned | nn.Embedding | 简单；长度有上限 |
| RoPE | 按 θ=pos·10000^{-2i/d} 旋转 Q,K | Llama, GPT-NeoX；相对位置感知强 |
| ALiBi | 在注意力分数上加线性偏置 −m·dist | 短序列训练，长序列推理 |

## 归一化变体 Normalization Variants

| 变体 | 公式 | 备注 |
|---|---|---|
| Post-LN（原始）| LN(x + Sublayer(x)) | 深层训练困难 |
| Pre-LN | x + Sublayer(LN(x)) | 现代 LLM 默认 |
| RMSNorm | x / RMS(x) · γ（无均值中心化）| Llama；更快，无偏置项 |

## 复杂度 Complexity

| 组件 Component | 时间 Time | 内存 Memory |
|---|---|---|
| 注意力 QKᵀ | O(T²d) | O(T²) |
| FFN（4× 扩展）| O(Td²) | O(Td) |
| FlashAttention | O(T²d) 时间 | **O(T) 内存**（分块 IO 感知）|

## 关键参数 Key Numbers

```
d_model = H × d_k         （头是分割的，不是复制的 split, NOT duplicate）
GPT-2 small:  d=768,  H=12，12 层
GPT-3 175B:   d=12288，H=96，96 层
Llama-2 7B:   d=4096，H=32，32 层
```

---

# 11. 深度学习速查 Deep Learning Cookbook

## 激活函数 Activations

| 名称 | 公式 | 使用场景 |
|---|---|---|
| ReLU | max(0, x) | 默认；可能"死亡" |
| GELU | x·Φ(x) | BERT, GPT-2 |
| SiLU/Swish | x·σ(x) | Llama，现代 |
| Sigmoid | 1/(1+e⁻ˣ) | 二分类输出；**导数 = σ(x)(1−σ(x))，最大值在 x=0 时为 1/4** |
| Tanh | (eˣ−e⁻ˣ)/(eˣ+e⁻ˣ) | RNN |
| Softmax | eˣⁱ / Σeˣʲ | 概率输出 |

## 初始化 Initialization

| 方法 Method | 方差 Variance | 适用 Use |
|---|---|---|
| Xavier/Glorot | 2/(fan_in + fan_out) | Sigmoid/tanh |
| He/Kaiming | 2/fan_in | ReLU/SiLU |
| Transformer 默认 | N(0, 0.02²) | GPT 风格 |

## 正则化 Regularization

| 方法 | 作用 |
|---|---|
| Dropout | 随机置零 → 集成效果；Transformer 用 p=0.1 |
| Weight decay (L2) | 惩罚大权重 → 缩小模型空间 |
| Label smoothing α=0.1 | 防止过度自信预测，改善校准 |
| Gradient clipping ‖g‖≤1 | 防止梯度爆炸 |
| Early stopping | 在验证损失最低点停止 |

## 反向传播关键梯度 Backprop Key Gradients

```
Y = XW:      dL/dX = dL/dY · Wᵀ
             dL/dW = Xᵀ · dL/dY
链式法则:    dL/dx = dL/dy · dy/dx
```

## 偏差-方差 Bias–Variance in ML

| 现象 Symptom | 原因 Cause | 修复 Fix |
|---|---|---|
| 训练损失高 High train loss | 高偏差（欠拟合）| 更大模型，更多特征 |
| 训练低验证高 Val >> Train | 高方差（过拟合）| 更多数据，dropout，正则化 |

---

# 12. LLM 专项概念 LLM-specific Concepts

## Tokenization

| 方法 | 使用者 | 原理 |
|---|---|---|
| BPE | GPT-2/3, Llama | 贪心合并高频 pair |
| WordPiece | BERT | 按似然合并 |
| SentencePiece | T5, Llama, Gemma | 原始字节；语言无关 |

## 生成策略 Generation Strategies

| 策略 Strategy | 超参 Hyperparameter | 说明 |
|---|---|---|
| Greedy | 无 | 快速；容易重复退化 |
| Beam search | num_beams | 翻译好；多样性差 |
| Top-k | k=50 | 从概率最高的 k 个采样 |
| Top-p (nucleus) | p=0.9 | 从累积概率 ≥ p 的最小集合采样 |
| Temperature | T<1 变尖，T>1 变平 | 在 softmax 前除以 T |

## 微调方法 Fine-tuning Methods

| 方法 | 可训练参数 | 使用场景 |
|---|---|---|
| Full FT | 100% | 效果最好，算力充足时 |
| LoRA | ~0.5%（低秩适配器）| 最流行 |
| QLoRA | ~0.5% 在 4-bit 基础上 | 单卡微调 70B |
| Prefix tuning | 可学习 KV 前缀 | 任务特定 token |
| RLHF / DPO | 奖励对齐 | 价值对齐 Alignment |

## LLM 一句话概念 One-liners

| 术语 Term | 定义 |
|---|---|
| KV cache | 缓存 K,V → 每步解码 O(T) vs O(T²) |
| MoE（混合专家）| 每个 token 只激活 top-k 个 FFN |
| Speculative decoding | 小草稿模型提议 token，大模型验证 |
| In-context learning | 示例在 prompt 中；**无参数更新** |
| RAG | 检索相关文档 → 拼接到 prompt → 生成 |
| Hallucination（幻觉）| 自信但事实错误的输出 |

---

# 13. ML 系统与生产 ML Systems & Production

## 训练流水线 Training Pipeline

```
数据 Data → 特征 Features → 模型 Model → 评估 Eval → 部署 Deploy → 监控 Monitor → 重训 Retrain
```

## 推理优化 Serving Optimizations

| 技术 Technique | 收益 Gain | 工具 |
|---|---|---|
| 连续批处理 Continuous batching | 吞吐量 | vLLM, TGI |
| 量化 Quantization INT8/INT4 | 内存 2–4× | GPTQ, AWQ |
| 蒸馏 Distillation | 更小模型 | 学生从教师学习 |
| KV cache | 解码 O(T) | 跨生成步缓存 |
| Speculative decoding | 延迟 | 草稿+验证 |

## 分布式训练策略 Distributed Training

| 策略 Strategy | 分割对象 Splits | 适用场景 |
|---|---|---|
| 数据并行 Data parallel (DDP) | batch | 模型能放入单卡 |
| 张量并行 Tensor parallel | 权重矩阵 | 单层太大 |
| 流水线并行 Pipeline parallel | 层 layers | 内存受限 |
| ZeRO / FSDP | 优化器状态+梯度+参数 | 大模型+有限 GPU |

## 评估指标 Eval Metrics

| 任务 Task | 主要指标 Primary Metrics |
|---|---|
| 二分类 Binary | Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC |
| 多分类 Multi-class | Macro-F1，混淆矩阵 Confusion Matrix |
| 回归 Regression | MAE, RMSE, R² |
| 排序/信息检索 Ranking | nDCG, MAP, MRR, Hit@k |
| 文本生成 Generation | BLEU, ROUGE, BERTScore, Perplexity |
| LLM | MMLU, HumanEval, MT-Bench, win-rate |

---

# 14. 多数投票与集成理论 Majority Vote / Ensemble — Theory

## 主公式 Master Formula（n 个独立投票者，准确率 p，n 为奇数）

```
P(多数正确) = Σ_{k=⌈n/2⌉}^{n} C(n,k) pᵏ (1−p)^{n−k}
```

## 快速对照表 Quick Table

| n | p=0.6 | p=0.7 | p=0.8 | p=0.9 |
|---|---|---|---|---|
| 1 | 0.600 | 0.700 | 0.800 | 0.900 |
| 3 | 0.648 | 0.784 | 0.896 | 0.972 |
| 5 | 0.683 | 0.837 | 0.942 | 0.991 |
| 11 | 0.753 | 0.922 | 0.988 | 0.9999 |

**Condorcet 定理：** p > 0.5 且独立 → n → ∞ 时准确率 → 1。

## Bagging 的偏差-方差 Bias–Variance for Bagging

```
Var(f̄) = ρ·σ_f² + (1−ρ)/M · σ_f²    →    ρ·σ_f²  当 M → ∞
```
Bagging 降低独立方差；Random Forest 通过特征子集进一步降低 ρ。

## AdaBoost 训练误差上界

```
training_error ≤ exp(−2 · Σ_t γ_t²)     其中 γ_t = 0.5 − ε_t
```

---

# 第三部分 — 编程 Coding

---

# 15. 矩阵算法 Matrix Algorithms — Spiral, Rotation, Diagonal

## 模式 A — 缩小边界（由外向内螺旋）Outside-in Spiral

```python
def spiral_order(matrix):
    if not matrix: return []
    res = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    while top <= bottom and left <= right:
        for j in range(left, right + 1):    res.append(matrix[top][j]);   top += 1
        for i in range(top, bottom + 1):    res.append(matrix[i][right]); right -= 1
        if top <= bottom:
            for j in range(right, left-1, -1): res.append(matrix[bottom][j]); bottom -= 1
        if left <= right:
            for i in range(bottom, top-1, -1): res.append(matrix[i][left]);   left += 1
    return res
```

## 模式 B — 扩展步长 1,1,2,2,3,3,...（由内向外 / 任意起点）

```python
def spiral_from(R, C, rStart, cStart):
    res = [[rStart, cStart]]
    dx, dy = [0,1,0,-1], [1,0,-1,0]   # 右下左上 R, D, L, U
    x, y, step, d = rStart, cStart, 1, 0
    while len(res) < R*C:
        for _ in range(2):              # 每个步长用两次
            for _ in range(step):
                x += dx[d]; y += dy[d]
                if 0 <= x < R and 0 <= y < C:
                    res.append([x, y])
                    if len(res) == R*C: return res
            d = (d + 1) % 4
        step += 1
    return res
```

## 原地顺时针旋转 90° In-place Rotate

```python
def rotate(M):
    n = len(M)
    # 第一步：转置 Transpose
    for i in range(n):
        for j in range(i+1, n):
            M[i][j], M[j][i] = M[j][i], M[i][j]
    # 第二步：每行翻转 Reverse each row
    for row in M:
        row.reverse()
```

## 对角线遍历 Diagonal Traverse

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

## 决策树 Decision Tree

| 螺旋类型 | 用法 |
|---|---|
| 由外向内，固定角落起点 | 模式 A Pattern A |
| 由内向外 / 任意起点 | 模式 B Pattern B |
| 对角线 / 反对角线 | defaultdict 按 i±j |
| 原地旋转 90° | 转置 + 行翻转 |

---

# 16. 编程模式 DSA Coding Patterns

## 双指针 Two Pointers

```python
l, r = 0, len(arr) - 1
while l < r:
    s = arr[l] + arr[r]
    if s == target: return [l, r]
    elif s < target: l += 1
    else: r -= 1
```

## 滑动窗口 Sliding Window

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

## 二分查找通用模板 Binary Search

```python
def bisect_left(arr, target):
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < target: lo = mid + 1
        else: hi = mid
    return lo
```

## DP 模式识别 DP Pattern Recognition

| 模式 Pattern | 特征 Signature |
|---|---|
| 一维线性 1D | dp[i] 依赖 dp[i-1], dp[i-2] |
| 二维网格 2D grid | dp[i][j] 依赖邻居 |
| 0-1 背包 Knapsack | dp[i][w] = max(跳过, 选取) |
| 最长递增子序列 LIS | O(n log n)，patience sort |
| 编辑距离 Edit distance | dp[i][j] 有 3 种转移 |
| 区间 DP Interval | dp[i][j] = 枚举所有分割点 |
| 逆向归纳 Backward induction | V[k] = 剩余 k 步的期望最优值 |

## BFS 模板

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

## 堆 Heap（优先队列）

```python
import heapq
heap = []
heapq.heappush(heap, (priority, item))
priority, item = heapq.heappop(heap)
# Top-k：维护大小为 k 的最小堆；超过 k 时 pop
```

## 并查集 Union-Find (DSU)

```python
class DSU:
    def __init__(self, n): self.p = list(range(n)); self.rank = [0]*n
    def find(self, x):
        if self.p[x] != x: self.p[x] = self.find(self.p[x])  # 路径压缩
        return self.p[x]
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py: return False
        if self.rank[px] < self.rank[py]: px, py = py, px
        self.p[py] = px
        if self.rank[px] == self.rank[py]: self.rank[px] += 1
        return True
```

## 单调栈 Monotone Stack

```python
def next_greater(nums):
    res, stack = [-1]*len(nums), []
    for i, v in enumerate(nums):
        while stack and nums[stack[-1]] < v:
            res[stack.pop()] = v
        stack.append(i)
    return res
```

---

# 17. PyTorch 要点 PyTorch Essentials

## 训练循环模板 Training Loop

```python
model.train()
for x, y in loader:
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()                               # ← 千万别忘
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

model.eval()
with torch.no_grad():                                   # ← 千万别忘
    for x, y in val_loader:
        pred = model(x.to(device))
```

## 5 大常见 Bug

| Bug | 现象 Symptom | 修复 Fix |
|---|---|---|
| 忘记 `zero_grad()` | 梯度累积，loss 爆炸 | 每次 backward 前调用 |
| 忘记 `model.eval()` | Dropout/BN 在验证集行为错误 | 推理前设置 |
| 忘记 `no_grad()` | 推理时 OOM | 用 `with torch.no_grad()` 包裹 |
| `print(loss)` | 计算图驻留内存 | 用 `loss.item()` |
| 设备不一致 Device mismatch | RuntimeError | 所有 tensor 移到同一设备 |

## 形状速查 Shapes

| 层 Layer | 输入 Input | 输出 Output |
|---|---|---|
| `nn.Linear(in, out)` | (*, in) | (*, out) |
| `nn.Conv2d(C, F, k)` | (B, C, H, W) | (B, F, H', W') |
| `nn.MultiheadAttention(d, h)` | (T, B, d) | (T, B, d) |
| `nn.LayerNorm(d)` | (*, d) | (*, d) |
| `nn.BatchNorm1d(C)` | (B, C) | (B, C) |

## 损失函数 Losses

| 任务 Task | 损失 Loss | 备注 Note |
|---|---|---|
| 二分类 Binary | `BCEWithLogitsLoss` | 输入 logits，不是概率 |
| 多分类 Multi-class | `CrossEntropyLoss` | logits + 整数标签 |
| 回归 Regression | `MSELoss / HuberLoss` | Huber 对异常值更鲁棒 |

```python
# 多分类：(B, C) logits + (B,) 整数标签
loss = F.cross_entropy(logits, labels)   # = log_softmax + NLL
```

## 优化器 Optimizers

| 优化器 | 默认 lr | 使用场景 |
|---|---|---|
| SGD | 0.01 | 视觉；微调 |
| Adam | 3e-4 | 通用默认 |
| AdamW | 1e-4 | Transformer（解耦 weight decay）|

## 混合精度 Mixed Precision

```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast():
    loss = model(x, y)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Warmup + 余弦衰减 LR Schedule

```python
def lr_lambda(step):
    if step < warmup_steps: return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))
```

---

# 18. Transformer 代码实现 Code Implementations

## 缩放点积注意力 Scaled Dot-Product Attention

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

## 因果掩码 Causal Mask

```python
# 上三角设为 -inf，防止未来 token 信息泄露
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
        # 无均值中心化，只做 RMS 归一化
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
```

## RoPE（旋转位置编码）

```python
def apply_rope(x, cos, sin):
    # x: (..., T, d)，按奇偶对分割
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
```

## SwiGLU FFN（Llama 风格）

```python
class SwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=False)  # 门控
        self.val  = nn.Linear(d, d_ff, bias=False)  # 值
        self.proj = nn.Linear(d_ff, d, bias=False)  # 下投影
    def forward(self, x):
        return self.proj(F.silu(self.gate(x)) * self.val(x))
```

## KV Cache（推理加速）

```python
# 每步只计算新 token 的 Q，K/V 从缓存中读取
K_cache = torch.cat([K_cache, k_new], dim=-2)    # 沿 T 维度扩展
V_cache = torch.cat([V_cache, v_new], dim=-2)
attn = F.softmax(q_new @ K_cache.T / d_k**0.5, dim=-1) @ V_cache
# 每步 O(T)，内存线性增长
```

## Decoder Block（Pre-LN）

```python
class DecoderBlock(nn.Module):
    def __init__(self, d, h, d_ff):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
        self.attn = MultiHeadAttention(d, h)
        self.ffn  = SwiGLU(d, d_ff)
    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask=mask)   # 残差连接
        x = x + self.ffn(self.norm2(x))
        return x
```

---

# 19. 概率编程模板 Probability Programming Patterns

## 最优停止（逆向归纳）Optimal Stopping

```python
def expected_optimal_dice(n: int) -> float:
    E = 3.5
    for _ in range(n - 1):
        E = sum(max(v, E) for v in range(1, 7)) / 6
    return E
# n=2 → 4.25，n=3 → 4.667，n→∞ → 6
```

## 多数投票概率 Majority Vote Probability

```python
from math import comb
def majority_prob(n: int, p: float) -> float:
    return sum(comb(n, k) * p**k * (1-p)**(n-k) for k in range(n//2+1, n+1))
```

## Top-p 核采样 Nucleus Sampling

```python
def sample(logits, temperature=1.0, top_p=0.9):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    sorted_p, sorted_i = probs.sort(descending=True)
    cum = sorted_p.cumsum(-1)
    # 找到累积概率超过 top_p 的位置，屏蔽后面的 token
    mask = cum > top_p
    mask[..., 1:] = mask[..., :-1].clone(); mask[..., 0] = False
    sorted_p[mask] = 0
    sorted_p /= sorted_p.sum(-1, keepdim=True)
    idx = torch.multinomial(sorted_p, 1)
    return sorted_i.gather(-1, idx)
```

## 水库采样 Reservoir Sampling（数据流，保留 k 个）

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
                reservoir[j] = item    # 等概率替换
    return reservoir
```

## 用 Rand5 构造 Rand7（拒绝采样）

```python
def rand7():
    while True:
        x = (rand5() - 1) * 5 + (rand5() - 1)   # 均匀分布在 [0, 24]
        if x < 21:                                  # 拒绝 21-24
            return x % 7 + 1
```

## Box-Muller 变换（均匀 → 正态）

```python
import math, random
def box_muller():
    u1, u2 = random.random(), random.random()
    z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    z1 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
    return z0, z1  # 两个均为 N(0, 1)
```

---

# 第四部分 — 行为面试 Behavioral

---

# 20. 面试策略与沟通 Interview Strategy & Communication

## 编程题解题节奏 Coding Problem Rhythm

1. **复述题意** Restate：用自己的话说一遍
2. **举例** Examples：自己给一个例子；问边界情况
3. **暴力解** Brute force first：先说暴力，报复杂度
4. **优化** Optimize：口述优化思路
5. **写代码** Code：变量名清晰，小函数
6. **手动追踪** Trace：走一遍具体例子
7. **讨论复杂度** Complexity：时间+空间，有无改进空间

## 概率题解题节奏 Probability Problem Rhythm

1. **定义随机变量**："设 X = ..."
2. **说明独立性/分布假设**
3. **选择工具**：线性期望 / 贝叶斯 / 条件期望 / 示性变量
4. **计算，然后验证**（如 E[max] + E[min] = E[sum]）
5. **停止问题** → 口述阈值规则

## 统计/A/B 测试节奏 Stats Rhythm

1. **明确 H₀ 和 H₁**
2. **说明 α 和目标 power**
3. **先计算样本量**，再开始实验
4. **识别潜在混淆变量**（辛普森悖论、选择偏差）
5. **报告效应大小**，不只是 p 值

## STAR 格式（行为面试）

> **S**ituation 情境 → **T**ask 任务 → **A**ction 行动 → **R**esult 结果（**带数字指标！**）

## 解释 X 的模板 "Explain X" Template

1. **一句话定义** One-sentence definition
2. **为什么重要 / 何时使用**
3. **具体例子** Concrete example
4. **权衡与失败模式** Trade-offs / failure modes

## 有力的表达 Power Phrases

- "先让我确认一下输入格式..." Let me clarify the input format...
- "先给暴力解 O(n²)，再优化。" Start with brute force, then optimize.
- "由期望的线性性，即使变量不独立..." By linearity of expectation, even if variables are dependent...
- "这是逆向归纳——让我定义 V[k]..." This is backward induction — let me define V[k]...
- "p 值是在 H₀ 为真的前提下看到当前数据的概率，不是 H₀ 为真的概率。" p-value is P(data this extreme | H₀), NOT P(H₀ is true).
- "我在这里假设独立性——如果不成立需要注意。" I'm assuming independence here — worth flagging if that breaks.

## 问面试官的问题 Questions to Ask

- "这个团队日常 on-call / 生产维护的负担大概是怎样的？"
- "团队研究与应用工作的比例大概是多少？"
- "入职 6 个月时，如何衡量这个岗位的成功？"
- "今年团队解决的最难的工程问题是什么？"

---

# 21. 附录：必背数字 Appendix — Must-Know Numbers

## 数学常数 Mathematical Constants

| 常数 Constant | 值 Value | 用途 Use |
|---|---|---|
| e | 2.71828 | 秘书问题截止点 n/e |
| 1/e | 0.368 | 秘书问题成功概率 |
| 1 − 1/e | 0.632 | Bootstrap 中被选到的比例 |
| ln 2 | 0.693 | 指数分布半衰期 |
| π²/6 | 1.645 | 优惠券收集方差系数 |
| 欧拉常数 γ | 0.577 | 调和级数偏移 Harmonic offset |
| 2√(2ln2) | 2.355 | 高斯 FWHM = 2.355σ |

## 标准正态分位数（必背）Standard Normal Critical Values

| α（双尾）| z_{α/2} | 置信区间 |
|---|---|---|
| 0.10 | 1.645 | 90% CI |
| 0.05 | **1.960** | **95% CI ← 背这个** |
| 0.01 | 2.576 | 99% CI |

## 样本量经验值 Sample Size Rules of Thumb

| 场景 Test | 每组最小 n |
|---|---|
| CTR 10%→12%，α=0.05，power=80% | ~3,800 |
| CTR 10%→11%，α=0.05，power=80% | ~14,700 |
| CLT 开始生效 | n ≥ 30 |
| 粗略估计 Rough rule | n ≈ (2.8/Δ)² · p(1-p) |

## 模型架构参数 Model Architecture Landmarks

| 模型 | 参数量 Params | d_model | 层数 Layers | 注意力头 Heads |
|---|---|---|---|---|
| BERT-base | 110M | 768 | 12 | 12 |
| GPT-2 small | 117M | 768 | 12 | 12 |
| GPT-3 | 175B | 12288 | 96 | 96 |
| Llama-2 7B | 7B | 4096 | 32 | 32 |
| Llama-2 70B | 70B | 8192 | 80 | 64 |

## 优化器默认参数 Optimizer Defaults

| 优化器 | lr | β₁ | β₂ | ε |
|---|---|---|---|---|
| Adam | 3e-4 | 0.9 | 0.999 | 1e-8 |
| AdamW | 1e-4 | 0.9 | 0.999 | 1e-8 |
| SGD w/ momentum | 0.01 | 0.9 | — | — |

---

> **最后提醒 Final Reminder：** 卡住时，**大声说出你的思路**。面试官评分的是推理过程，不只是最终答案。命名变量、说明不变量、用小例子验证、承认边界情况，然后继续推进。
>
> **加油！Good luck!**
