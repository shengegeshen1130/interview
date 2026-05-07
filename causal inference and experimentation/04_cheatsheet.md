# 因果推断 & 实验设计 — 速查表 (Cheatsheet)

> 面试前最后 30 分钟扫一遍。所有公式、决策树、代码骨架。

---

## 1. 因果效应定义

| 量 | 公式 | 含义 |
|---|---|---|
| ITE | `Y_i(1) − Y_i(0)` | 个体（不可识别） |
| **ATE** | `E[Y(1) − Y(0)]` | 总体平均 |
| ATT | `E[Y(1) − Y(0) \| T=1]` | 治疗组上的平均 |
| CATE / HTE | `E[Y(1) − Y(0) \| X=x]` | 给定特征下 |
| LATE | complier 子集上的 ATE | IV 识别的量 |

## 2. 核心假设

| 假设 | 公式 | 用途 |
|---|---|---|
| **SUTVA** | no interference + consistency | 所有方法的基础 |
| **Ignorability** | `(Y(0), Y(1)) ⊥ T \| X` | 观察性研究 |
| **Positivity** | `0 < P(T=1\|X) < 1` | overlap |
| **Parallel Trends** | trend 平行 (no T) | DiD |
| **Relevance + Exclusion + Indep.** | IV 三条件 | IV / 2SLS |
| **Continuity at cutoff** | E[Y\|R] 在 R=c 处连续 | RD |

## 3. DAG 三种结构

```
Chain:    T → X → Y      → 不要控制 X (mediator)
Fork:     T ← X → Y      → 必须控制 X (confounder)
Collider: T → X ← Y      → 不要控制 X (会引入虚假关联)
```

**Backdoor**：阻断所有 `→ T` 的入向路径。
**Frontdoor**：当存在完全 mediator M 时使用。

## 4. 因果方法决策树

```
能 randomize？
├── 是 → RCT / A/B test
└── 否 ↓
    
有 panel data？
├── 是 + 平行趋势 → DiD
└── 否 ↓

有 cutoff？
├── 是 → RD
└── 否 ↓

有合法 IV？
├── 是 → 2SLS
└── 否 ↓

ignorability 假设可信？
├── 是 → PSM / IPW / DR / Causal Forest
└── 否 → frontdoor / sensitivity analysis
```

## 5. 方法对比

| 方法 | 假设 | 量 | 优 | 缺 |
|---|---|---|---|---|
| RCT | SUTVA | ATE | gold | 贵 |
| PSM/IPW | ignorability | ATE/ATT | 直观 | 不解决 unobserved |
| DiD | 平行趋势 | ATT | 处理 time-invariant unobs | 需 panel |
| IV | 三条件 | LATE | unobserved 也可 | LATE ≠ ATE |
| RD | 连续性 | LATE@c | 假设最弱 | 仅 cutoff 附近 |
| Causal Forest | ignorability + 大样本 | CATE | 非线性 HTE | 难解释 |

## 6. 核心公式

### Type I / II 错误（必备 2×2）

|  | H₀ 真 | H₁ 真 |
|---|---|---|
| **拒绝 H₀** | Type I (α) ❌ | Power (1−β) ✅ |
| **不拒绝 H₀** | 1−α ✅ | Type II (β) ❌ |

```
α = P(reject H₀ | H₀ 真)        ← 在 H₀ 分布下
β = P(fail to reject | H₁ 真)    ← 在 H₁ 分布下
α + β ≠ 1   (常见误区！)

Trade-off (固定 N):  α↓ ⇒ β↑
增大 N:               α 和 β 同时↓
增大 |Δ|:             β↓ (power↑)，α 不变
```

**类比记忆：** 法庭 H₀=无罪 → Type I=冤好人，Type II=放凶手。
**业务代价决定 α/β 选择：** 高风险用 α=0.01；增长实验可用 α=0.1。

**常见膨胀情境：**
- Peeking 10 次 → 实际 α ≈ 19%（名义 5%）
- 测 20 metrics 不校正 → FWER ≈ 64%
- 修正：sequential testing / Bonferroni / BH

### 样本量
```
N_per_group ≈ 16 · σ² / Δ²              (continuous, α=0.05, power=0.8)
N_per_group ≈ 16 · p(1−p) / (p_T − p_C)²  (proportion)

更精确:  N = 2(z_{α/2} + z_β)² σ² / Δ²
```

### CUPED 缩减
```
Var(Y_cuped) = Var(Y) · (1 − ρ²)
```

| ρ | 缩减 | 等效 N 倍数 |
|---|---|---|
| 0.3 | 9% | 1.10× |
| 0.5 | 25% | 1.33× |
| 0.7 | 49% | 1.96× |
| 0.9 | 81% | 5.26× |

### Backdoor 调整
```
P(Y | do(T=t)) = Σ_z P(Y | T=t, Z=z) · P(Z=z)
```

### IPW
```
ATE = (1/N) Σ [ T·Y/e(X) − (1−T)·Y/(1−e(X)) ]
```

### DiD ATT
```
ATT = (Y_post,T=1 − Y_pre,T=1) − (Y_post,T=0 − Y_pre,T=0)
```

### 2SLS
```
Stage 1: T̂ = π·Z + γ·X
Stage 2: Y = β·T̂ + γ·X + u    →  β̂ 是 LATE
```

## 7. 代码骨架

### t-test
```python
from scipy.stats import ttest_ind
t, p = ttest_ind(treatment, control, equal_var=False)
```

### CUPED
```python
theta = np.cov(Y_pre, Y_post)[0,1] / np.var(Y_pre)
Y_adj = Y_post - theta * (Y_pre - Y_pre.mean())
ate = Y_adj[T==1].mean() - Y_adj[T==0].mean()
```

### SRM
```python
from scipy.stats import chisquare
chi2, p = chisquare([n_T, n_C], [(n_T+n_C)/2]*2)
# p < 0.001 → SRM!
```

### DiD (regression form)
```python
import statsmodels.formula.api as smf
result = smf.ols('Y ~ post + treated + post:treated', data=df).fit()
att = result.params['post:treated']
```

### IV (2SLS)
```python
from linearmodels.iv import IV2SLS
result = IV2SLS.from_formula('Y ~ 1 + X + [T ~ Z]', data=df).fit()
```

### PSM
```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
ps = LogisticRegression().fit(df[X], df[T]).predict_proba(df[X])[:,1]
nbrs = NearestNeighbors(n_neighbors=1).fit(ps[T==0].reshape(-1,1))
_, idx = nbrs.kneighbors(ps[T==1].reshape(-1,1))
# match treated[i] to control[idx[i]]
```

### Causal Forest
```python
from econml.dml import CausalForestDML
est = CausalForestDML(model_y=GBR(), model_t=GBR())
est.fit(Y=Y, T=T, X=X, W=W)
cate = est.effect(X_test)
```

### Sample size calculator
```python
from scipy.stats import norm
z_a = norm.ppf(0.975)         # alpha=0.05 two-sided
z_b = norm.ppf(0.8)           # power=0.8
n = 2 * (z_a + z_b)**2 * sigma**2 / delta**2
```

### Multiple testing (BH)
```python
from statsmodels.stats.multitest import multipletests
reject, p_corr, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
```

## 8. 速记口诀

| 概念 | 一句话 |
|---|---|
| Correlation ≠ Causation | "可能是 confounder、reverse、selection" |
| RCT 优 | "随机化让 `(Y(0),Y(1)) ⊥ T` 无条件成立" |
| SUTVA 失效 | "网络效应、Marketplace、群体免疫" |
| Confounder vs Mediator | "Confounder 控；mediator 不控（会 block）" |
| Collider | "条件化 collider 引入虚假关联" |
| Backdoor | "阻断所有 `→ T` 的后门路径" |
| PSM 局限 | "不解决 unobserved confounder" |
| DiD 关键 | "平行趋势假设；用 pre-period 检验" |
| IV 局限 | "估计 LATE 不是 ATE" |
| RD 局限 | "只 cutoff 附近的 LATE" |
| CUPED | "用 pre-experiment X 减方差，前提 X ⊥ T" |
| SRM | "T:C 偏离设计 → 别信结果" |
| Novelty | "新鲜感衰减；跑长一点" |
| Peeking | "随时看 p-value → Type I 失控" |
| Power = 1−β | "真有效时检测到的概率" |
| Multiple testing | "测 m 个假设要 BH 校正" |
| Switchback | "时间窗口切换；解决 marketplace 干扰" |
| Uplift | "找 persuadable 用户 = CATE 的别名" |

## 9. 常见数字记忆

| 概念 | 数字 |
|---|---|
| Default α | 0.05 |
| Default power | 0.8 |
| z_{0.025} | 1.96 |
| z_{0.2} | 0.84 |
| (z_α + z_β)² for α=0.05, β=0.2 | ≈ 7.85 |
| 经验 N 公式系数 | 16 (= 2·7.85≈15.7) |
| SRM threshold | p < 0.001 |
| Common balance threshold | SMD < 0.1 |
| Weak instrument threshold | F < 10 |
| 1/e (secretary) | 0.368 |

---

## 10. 一页背诵清单

### 必须秒答

1. **ATE = E[Y(1) − Y(0)]**
2. **RCT works because: random T ⇒ (Y(0),Y(1)) ⊥ T**
3. **N ≈ 16 σ² / Δ²**
4. **Var(Y_cuped) = Var(Y)(1 − ρ²)**
5. **CUPED 要求 X 是 *实验前* 测量**
6. **SRM = T:C 比例显著偏离 → 别信结果**
7. **Bonferroni: α/m. BH: k·α/m. FDR ≤ α.**
8. **DiD 假设: parallel trends.**
9. **IV 三条件: relevance + exclusion + independence.**
10. **IV → LATE, RD → LATE@cutoff, PSM → ATE/ATT.**
11. **Confounder 控；mediator 不控；collider 不控.**
12. **p-value 是 *H₀ 真* 时看到这种或更极端结果的概率.**
13. **SUTVA 失效 → switchback / cluster-randomized.**
14. **CATE / HTE → meta-learners (S/T/X/R) or causal forest.**
15. **Type I (α) = 假阳；Type II (β) = 假阴；Power = 1−β.**
16. **α 和 β 不互补（在不同分布下定义）；trade-off 仅在固定 N 下.**
17. **Peeking 把名义 α 膨胀；FWER = 1−(1−α)^m.**
18. **业务代价不对称 → α、β 不必都用默认值.**

### 答题套话模板

> "我会先用 RCT 因为 *gold standard*；如果不行考虑 quasi-experimental：DiD 如果有 panel + 平行趋势，IV 如果有合法工具，RD 如果有 cutoff，否则 PSM/DR + sensitivity analysis。"

> "样本量我用 `N ≈ 16σ²/Δ²`，σ 从历史数据估，Δ 是业务上的 MDE。然后用 CUPED 减 ρ² 的方差，看 baseline 相关性。"

> "实验前 sanity check：A/A test、SRM、guardrail metric。运行后看 primary OEC，secondary metrics 用 BH 校正。"
