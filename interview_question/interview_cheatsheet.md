# ML Engineering Interview — Master Cheatsheet

One-stop quick reference for your upcoming interview. Densely packed: formulas, code snippets, and one-line answers. Skim before the interview; recall during it.

> **Universal tip.** Always (1) ask clarifying questions, (2) state assumptions, (3) verbalize the recurrence/threshold/architecture before coding, (4) test with a tiny example, (5) note complexity.

---

## Table of Contents

1. [Probability — Core Formulas](#1-probability--core-formulas)
2. [Expected Value & Variance — Tricks](#2-expected-value--variance--tricks)
3. [Optimal Stopping — Decision Templates](#3-optimal-stopping--decision-templates)
4. [Matrix Algorithms — Spiral, Rotation, Diagonal](#4-matrix-algorithms--spiral-rotation-diagonal)
5. [Majority Vote / Ensemble](#5-majority-vote--ensemble)
6. [Transformer Architecture](#6-transformer-architecture)
7. [PyTorch Essentials](#7-pytorch-essentials)
8. [Deep Learning Cookbook](#8-deep-learning-cookbook)
9. [LLM-specific Concepts](#9-llm-specific-concepts)
10. [Coding Patterns (DSA)](#10-coding-patterns-dsa)
11. [ML Systems / Production](#11-ml-systems--production)
12. [Behavioral & Communication](#12-behavioral--communication)

---

# 1. Probability — Core Formulas

## Common distributions

| Distribution | PMF/PDF | E[X] | Var(X) |
|---|---|---|---|
| **Bernoulli(p)** | `p`, `1−p` | `p` | `p(1−p)` |
| **Binomial(n, p)** | `C(n,k) p^k (1−p)^{n−k}` | `np` | `np(1−p)` |
| **Geometric(p)** (count incl. success) | `(1−p)^{k−1} p` | `1/p` | `(1−p)/p²` |
| **Negative Binomial(r, p)** | wait for `r` successes | `r/p` | `r(1−p)/p²` |
| **Poisson(λ)** | `e^{−λ} λ^k / k!` | `λ` | `λ` |
| **Uniform{1..n}** | `1/n` | `(n+1)/2` | `(n²−1)/12` |
| **Uniform[a,b]** | `1/(b−a)` | `(a+b)/2` | `(b−a)²/12` |
| **Exponential(λ)** | `λ e^{−λx}` | `1/λ` | `1/λ²` |
| **Normal(μ, σ²)** | std bell | `μ` | `σ²` |
| **Beta(α, β)** | conjugate to Bernoulli | `α/(α+β)` | `αβ / ((α+β)²(α+β+1))` |

## Key identities

```
Bayes:           P(A|B) = P(B|A)·P(A) / P(B)
Total prob:      P(B)   = Σ P(B|Aᵢ) P(Aᵢ)
Linearity:       E[X+Y] = E[X] + E[Y]    (always, even if dependent!)
Var sum:         Var(X+Y) = Var(X) + Var(Y) + 2 Cov(X,Y)
Cov:             Cov(X,Y) = E[XY] − E[X]E[Y]
Var formula:     Var(X) = E[X²] − (E[X])²        ← always preferred for hand calc
Tower / TPE:     E[X]   = E[E[X|Y]]
Eve's law (TPV): Var(X) = E[Var(X|Y)] + Var(E[X|Y])
```

## Quick reference — answers you should know cold

| Question | Answer |
|---|---|
| E[X] of fair die | `3.5` |
| Var(X) of fair die | `35/12 ≈ 2.917` |
| E[max(X₁,X₂)] of two fair dice | `161/36 ≈ 4.472` |
| E[min(X₁,X₂)] of two fair dice | `91/36 ≈ 2.528` (sums to 7 ✓) |
| Expected flips for first head, fair coin | `2` |
| Expected flips for `HHH`, fair coin | `14` |
| Expected flips for `HTH`, fair coin | `10` |
| Coupon collector, n distinct | `E ≈ n ln n + γn`, `Var ≈ π²n²/6` |
| Gambler's ruin hitting time (sym, start `i`, range `[0,N]`) | `i(N−i)` |
| Gambler's ruin win prob (sym) | `i/N` |
| 3 indep classifiers @ 80%, majority | `0.896` |
| Secretary problem optimal cutoff | `n/e ≈ 0.368·n` |

---

# 2. Expected Value & Variance — Tricks

## Trick 1 — Linearity (no independence needed)

```python
# E[sum of indicators] = sum of probabilities
# Example: expected number of fixed points in random permutation = 1
# (for any n, by linearity over n indicator variables, each with prob 1/n)
```

## Trick 2 — Condition on first step

```
"Expected flips until first head" — let X = answer.
E[X] = 1·p + (1+E[X])·(1−p)  ⇒  E[X] = 1/p
```

## Trick 3 — Indicator method

```
"Expected number of distinct coupons after n draws (with replacement, k types):"
Let Iⱼ = 1 if coupon j ever appears.
E[Iⱼ] = 1 − ((k−1)/k)^n
E[distinct] = k · (1 − ((k−1)/k)^n)
```

## Trick 4 — Wald's identity

If `N` is a stopping time with `E[N] < ∞` and `Xᵢ` i.i.d. with mean `μ`:

```
E[X₁ + X₂ + ... + X_N] = μ · E[N]
```

## Trick 5 — Eve's law for compound sums

Customers `N ~ Poisson(λ)`, each spends `Xᵢ` (mean `μ`, var `σ²`):
```
E[S]   = λμ
Var(S) = λ(σ² + μ²) = λ·E[X²]
```

## Trick 6 — Variance via E[X²] − (E[X])²

Always faster than `Σ (xᵢ − μ)² P(X=xᵢ)`. Memorize this; never use the definition unless asked.

---

# 3. Optimal Stopping — Decision Templates

## Universal recurrence

```
V[k] = expected value with k decisions/rolls remaining
V[1] = base case (forced last action)
V[k] = E[ max(immediate_reward, continuation_value) ]
```

## Dice game (the original)

```python
def expected_optimal_dice(n: int) -> float:
    E = 3.5
    for _ in range(n - 1):
        E = sum(max(v, E) for v in range(1, 7)) / 6
    return E
# n=2 → 4.25, n=3 → 4.667, n→∞ → 6
```

**Threshold rule (3 rolls):** Stop on 5 or 6 first roll; on 4–6 second; keep third.

## Dice with cost `c` per re-roll

```python
E = 3.5
for _ in range(n-1):
    E = sum(max(v, E - c) for v in range(1, 7)) / 6  # threshold drops by c
```

## Continuous Uniform[0,1]

`E[k] = (1 + E[k-1]²) / 2`, `E[1]=0.5`, `E[2]=0.625`, `E[3]≈0.695`, `E[∞]=1`.

## Secretary problem (1/e rule)

Reject first `n/e` candidates; pick first one better than all seen. `P(success) → 1/e ≈ 0.368`.

## Prophet inequality

Single threshold `τ` with `P(max Xᵢ ≥ τ) = 1/2` achieves `≥ 0.5 · E[max Xᵢ]`.

## Hint

If problem says "decide whether to keep this offer" → optimal stopping.
If problem says "find the best one online" → secretary.
If problem says "minimize expected rank" → Robbins (open problem!).

---

# 4. Matrix Algorithms — Spiral, Rotation, Diagonal

## Pattern A — Shrinking boundaries (outside-in spiral)

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

## Pattern B — Expanding step `1,1,2,2,3,3,...` (inside-out / arbitrary start)

```python
def spiral_from(R, C, rStart, cStart):
    res = [[rStart, cStart]]
    dx, dy = [0,1,0,-1], [1,0,-1,0]      # R, D, L, U
    x, y, step, d = rStart, cStart, 1, 0
    while len(res) < R*C:
        for _ in range(2):                # each step length used twice
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
        row.reverse()                              # reverse each row
```

## Diagonal traverse

```python
# group by i+j, alternate reversal
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

## Matrix Layer Rotation (HackerRank)

Rotate `r mod perimeter` per layer; otherwise TLE on `r = 10⁹`.

## Decision tree

| Spiral type | Use |
|---|---|
| Outside-in, fixed corner start | Pattern A |
| Inside-out / arbitrary start / over-sized rectangle | Pattern B |
| Diagonal / anti-diagonal | `defaultdict(list)` keyed by `i±j` |
| In-place 90° rotation | transpose + row reverse |

---

# 5. Majority Vote / Ensemble

## Master formula (n indep voters @ p, n odd)

```
P(majority correct) = Σ_{k=⌈n/2⌉}^{n} C(n,k) p^k (1-p)^{n-k}
```

## Quick computational table

| n | p=0.6 | p=0.7 | p=0.8 | p=0.9 |
|---|---|---|---|---|
| 1 | 0.600 | 0.700 | 0.800 | 0.900 |
| 3 | 0.648 | 0.784 | 0.896 | 0.972 |
| 5 | 0.683 | 0.837 | 0.942 | 0.991 |
| 11 | 0.753 | 0.922 | 0.988 | 0.9999 |
| 51 | 0.926 | 0.999 | ~1 | ~1 |

**Condorcet:** if `p > 0.5` and indep, accuracy → 1 as `n → ∞`.

## Heterogeneous ensemble

Enumerate `2^n` correctness patterns. Optimal log-likelihood-ratio weights: `wᵢ = log(pᵢ / (1−pᵢ))`.

## Bias–variance for bagging

```
Var(f̄) = ρ·σ_f² + (1−ρ)/M · σ_f²    →    ρ·σ_f²  as M → ∞
```
Bagging reduces independent variance only. Random Forest reduces `ρ` further via feature subsetting.

## AdaBoost training-error bound

```
training_error ≤ exp(−2 · Σ_t γₜ²)        where γₜ = 0.5 − εₜ
```

## Hoeffding-based sample complexity

```
n ≥ ln(1/δ) / (2 · (p − 0.5)²)          for majority error ≤ δ
```
Example: `p=0.55, δ=0.01` → 921 voters.

---

# 6. Transformer Architecture

## Scaled dot-product attention

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
```

```python
import torch, torch.nn.functional as F

def scaled_dot_product(q, k, v, mask=None):
    # q: (B, H, T, d_k), k: (B, H, T, d_k), v: (B, H, T, d_v)
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / d_k**0.5     # (B, H, T, T)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    return attn @ v
```

## Multi-head split

```
d_model = H * d_k        # split, not duplicate
Wq, Wk, Wv: (d_model, d_model)  →  reshape to (B, H, T, d_k)
Wo: (d_model, d_model)            ← concat heads then project
```

## Variants — when to use each

| Variant | Q heads | K heads | V heads | Use case |
|---|---|---|---|---|
| MHA (vanilla) | H | H | H | Small models |
| MQA | H | 1 | 1 | Inference speed (PaLM, Falcon) |
| GQA | H | G | G (G < H) | Llama 2/3, balance |
| Cross-attn | from decoder | from encoder | from encoder | Encoder-decoder |

## Positional encodings

| Type | Formula / Idea | Pros / Cons |
|---|---|---|
| **Sinusoidal** | `PE(pos, 2i) = sin(pos / 10000^{2i/d})` | Fixed; extrapolates poorly |
| **Learned absolute** | `nn.Embedding(max_len, d)` | Simple; bounded length |
| **RoPE** | Rotate Q,K by angle `θ_i = pos · 10000^{-2i/d}` | Used in Llama, GPT-NeoX; better extrapolation |
| **ALiBi** | Add linear bias `−m·dist` to attn scores | Train short, infer long |

```python
# RoPE essentials (per head dim, pairs of features)
def apply_rope(x, cos, sin):
    # x: (..., T, d), split into pairs
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
```

## Normalization placement

| Variant | Order | Notes |
|---|---|---|
| Post-LN (orig) | `LN(x + Sublayer(x))` | Hard to train deep |
| Pre-LN | `x + Sublayer(LN(x))` | Default in modern LLMs |
| RMSNorm | `x / RMS(x) · γ` (no mean centering) | Llama; faster |

```python
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
```

## Masking

| Mask | When | Shape |
|---|---|---|
| **Causal** (decoder) | Autoregressive | upper-triangular `−inf` |
| **Padding** | Variable-length batches | `(B, 1, 1, T)` zero-out pad tokens |
| **Cross-attn** | Encoder-decoder | None unless padded |

```python
# Causal mask
causal = torch.triu(torch.ones(T, T), diagonal=1).bool()
scores.masked_fill_(causal, float('-inf'))
```

## Complexity

| Operation | Time | Memory |
|---|---|---|
| Attention scores `QKᵀ` | `O(T² d)` | `O(T²)` |
| Softmax × V | `O(T² d)` | `O(T²)` |
| FFN (4× expand) | `O(T d²)` | `O(T d)` |
| **FlashAttention** | `O(T² d)` time, **`O(T)` memory** | tiled, IO-aware |

## Decoder block (Pre-LN, modern)

```python
class DecoderBlock(nn.Module):
    def __init__(self, d, h, d_ff):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
        self.attn = MultiHeadAttention(d, h)
        self.ffn  = nn.Sequential(nn.Linear(d, d_ff), nn.SiLU(),
                                  nn.Linear(d_ff, d))
    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.ffn(self.norm2(x))
        return x
```

## KV cache (inference)

```python
# Append new token's K,V to cached tensors; only compute attention for the new token's Q
K_cache = torch.cat([K_cache, k_new], dim=-2)   # (B, H, T_total, d_k)
V_cache = torch.cat([V_cache, v_new], dim=-2)
attn = softmax(q_new @ K_cache.transpose(-2,-1) / sqrt(d_k)) @ V_cache
```
Reduces decode cost from `O(T²)` per step to `O(T)`. Memory grows linearly with `T`.

## FFN (SwiGLU — modern Llama-style)

```python
class SwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d, d_ff, bias=False)   # gate
        self.w2 = nn.Linear(d, d_ff, bias=False)   # value
        self.w3 = nn.Linear(d_ff, d, bias=False)   # down
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
```

---

# 7. PyTorch Essentials

## Training loop template (memorize)

```python
model.train()
for x, y in loader:
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()                  # don't forget!
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()                        # autograd
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # optional
    optimizer.step()

# Validation
model.eval()
with torch.no_grad():
    for x, y in val_loader:
        pred = model(x.to(device))
        ...
```

## The 5 commandments (common bugs)

| Mistake | Fix |
|---|---|
| Forgetting `optimizer.zero_grad()` | Grads accumulate; loss explodes |
| Forgetting `model.eval()` | Dropout / BN behave wrong at val |
| Forgetting `with torch.no_grad():` | OOM during inference |
| `print(loss)` instead of `loss.item()` | Holds graph in memory |
| `tensor.cuda()` mixed with `tensor.to('cpu')` | Devices mismatch |

## Autograd intuition

- **`requires_grad=True`** on inputs makes them part of the graph.
- **`.backward()`** computes gradients for all leaves.
- **`.detach()`** breaks the graph (use for "stop gradient").
- **`with torch.no_grad():`** disables graph building entirely.

## Common shapes / layers

| Layer | Input shape | Output shape |
|---|---|---|
| `nn.Linear(in, out)` | `(*, in)` | `(*, out)` |
| `nn.Conv2d(C, F, k)` | `(B, C, H, W)` | `(B, F, H', W')` |
| `nn.LSTM(in, hid)` | `(T, B, in)` or `(B, T, in)` w/ `batch_first` | `(T, B, hid)` + `(h, c)` |
| `nn.MultiheadAttention(d, h)` | `(T, B, d)` (or `batch_first`) | `(T, B, d)` |
| `nn.LayerNorm(d)` | `(*, d)` | `(*, d)` |
| `nn.BatchNorm1d(C)` | `(B, C)` or `(B, C, T)` | same |
| `nn.Dropout(p)` | any | same |

## Loss functions

| Task | Loss | Notes |
|---|---|---|
| Binary classif | `BCEWithLogitsLoss` | Pass logits, not probs |
| Multi-class | `CrossEntropyLoss` | Pass logits + class indices, **not** one-hot |
| Regression | `MSELoss` / `L1Loss` / `HuberLoss` | Huber for outliers |
| Embedding | `CosineEmbeddingLoss` / `TripletMarginLoss` | |

```python
# CE on (B, C) logits + (B,) integer labels
loss = F.cross_entropy(logits, labels)        # combines log_softmax + NLL
```

## Optimizers

| Name | Use case | Hyperparameters |
|---|---|---|
| **SGD** | Vision; fine-tuning | `lr`, `momentum=0.9`, `weight_decay` |
| **Adam** | General default | `lr=3e-4`, `betas=(0.9, 0.999)` |
| **AdamW** | Transformers (decoupled weight decay) | `lr=1e-4`, `weight_decay=0.01` |
| **Lion** | Memory-efficient alt | `lr ≈ 1/3 of Adam's` |

## LR schedules

```python
# Cosine warmup + decay (modern LLM standard)
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
# Or manually:
warmup_steps = 1000
def lr_at(step):
    if step < warmup_steps: return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))
```

## DataLoader gotchas

- `num_workers=0` for debugging; `4–8` in production
- `pin_memory=True` if GPU
- `shuffle=True` for training only
- `drop_last=True` if BN with small final batch

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

---

# 8. Deep Learning Cookbook

## Activations

| Name | Formula | Use |
|---|---|---|
| ReLU | `max(0, x)` | Default; dies at 0 |
| GELU | `x · Φ(x)` | Transformers (BERT, GPT-2) |
| SiLU/Swish | `x · σ(x)` | Llama, modern |
| Sigmoid | `1 / (1 + e^{−x})` | Binary output |
| Tanh | `(e^x − e^{−x})/(e^x + e^{−x})` | RNNs |
| Softmax | `e^{x_i} / Σ e^{x_j}` | Output probabilities |

## Initialization

| Method | Formula | Use |
|---|---|---|
| **Xavier/Glorot** | `Var = 2 / (fan_in + fan_out)` | Sigmoid/tanh |
| **He/Kaiming** | `Var = 2 / fan_in` | ReLU/SiLU |
| **Truncated normal σ=0.02** | std init for transformers | GPT |

## Regularization

- **Dropout** (`p=0.1` for transformers, `0.5` for FCs)
- **Weight decay** (L2 in optimizer)
- **Data augmentation** (vision/audio)
- **Label smoothing** `α=0.1` (helps calibration)
- **Early stopping** on validation loss
- **Gradient clipping** `max_norm=1.0`

## Backprop checklist

- Forward computes loss; backward computes `dL/dθ` for every leaf.
- Chain rule: `dL/dx = dL/dy · dy/dx`.
- For matrix mul `Y = XW`: `dL/dX = dL/dY · Wᵀ`, `dL/dW = Xᵀ · dL/dY`.

## Bias-variance

- High **bias**: train loss high → bigger model, more features.
- High **variance**: gap between train and val → more data, regularization, dropout.

---

# 9. LLM-specific Concepts

## Tokenization

| Method | Used by | Idea |
|---|---|---|
| **BPE** | GPT-2/3, Llama | Greedy merge frequent pairs |
| **WordPiece** | BERT | Merge by likelihood |
| **SentencePiece** | T5, Llama, Gemma | Treats text as raw bytes; language-agnostic |
| **Unigram LM** | T5 alt | Probabilistic vocab pruning |

## Generation strategies

| Strategy | Output | Hyperparam |
|---|---|---|
| **Greedy** | argmax each step | none |
| **Beam search** | top-k beams | `num_beams` |
| **Top-k** sampling | sample from top k | `k=50` |
| **Top-p** (nucleus) | sample from smallest set with `Σp ≥ p` | `p=0.9` |
| **Temperature** | divide logits | `T<1` sharpens, `T>1` flattens |

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

## Fine-tuning techniques

| Method | Trainable params | Use |
|---|---|---|
| Full FT | all | Best quality if you have compute |
| **LoRA** | `B·A` low-rank adapters, ~0.5% | Most popular |
| **QLoRA** | LoRA on 4-bit quantized base | Single GPU FT of 70B |
| Prefix tuning | learned KV prefix | Tasks-specific tokens |
| Prompt tuning | learned soft prompt only | Tiny |
| RLHF / DPO | reward-aligned | Alignment |

## Quantization

| Bits | Method | Quality drop |
|---|---|---|
| FP16 / BF16 | half precision | none |
| INT8 | row/col scaling | minor |
| INT4 (GPTQ, AWQ) | weight-only | small |
| FP8 (Hopper) | training in fp8 | none on H100 |

## RAG (Retrieval-Augmented Generation)

```
1. Embed corpus offline   → vector DB (FAISS, Pinecone, Qdrant)
2. Embed query            → top-k similarity search
3. Concat retrieved docs into prompt
4. Generate
```
**Common pitfalls:** stale index, chunk-size mismatch, no reranker, no citation grounding.

## Common LLM concepts you should be able to define in 1 sentence

| Term | One-liner |
|---|---|
| **Self-attention** | Each token weights all tokens via Q·Kᵀ to update its representation. |
| **Causal mask** | Prevents position `t` from attending to positions `> t`. |
| **KV cache** | Cache K and V across decoding steps to avoid O(T²) recomputation. |
| **MoE** | Mixture-of-Experts; only top-`k` of `N` FFNs activate per token. |
| **Speculative decoding** | Use a small draft model to propose tokens; verify with the big one. |
| **In-context learning** | Few-shot examples in prompt; no weight updates. |
| **CoT (chain-of-thought)** | Prompt model to "think step by step" before answering. |
| **Hallucination** | Confidently wrong output, often factual. Mitigate with RAG, grounding, eval. |

---

# 10. Coding Patterns (DSA)

## Two pointers

```python
# E.g., 2-sum on sorted array
l, r = 0, len(arr) - 1
while l < r:
    s = arr[l] + arr[r]
    if s == target: return [l, r]
    elif s < target: l += 1
    else: r -= 1
```

## Sliding window

```python
# Longest substring with at most k distinct chars
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

## Binary search (general template)

```python
def bisect_left(arr, target):
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < target: lo = mid + 1
        else: hi = mid
    return lo
```

## DP — recognize the type

| Pattern | Signature |
|---|---|
| **1D** (Fibonacci, climb stairs) | `dp[i]` depends on `dp[i-1], dp[i-2]` |
| **2D grid** | `dp[i][j]` from neighbors |
| **Knapsack 0/1** | `dp[i][w] = max(dp[i-1][w], dp[i-1][w-wᵢ] + vᵢ)` |
| **LIS** | `O(n log n)` via patience sort |
| **Edit distance** | `dp[i][j]` from 3 transitions |
| **Stopping problems** | backward induction (this interview!) |

## BFS / DFS template

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

## Heap

```python
import heapq
heap = []
heapq.heappush(heap, (priority, item))
priority, item = heapq.heappop(heap)
# top-k = heap of size k; push and pop if len > k
```

---

# 11. ML Systems / Production

## Training-serving mental model

```
Data → Features → Model → Eval → Deploy → Monitor
                                    ↑          ↓
                                    └─ Retrain ┘
```

## Serving optimizations

- **Batching:** dynamic / continuous batching (vLLM, TGI) — group concurrent requests
- **Quantization:** INT8 / INT4 weight-only for 2-4× speedup
- **Distillation:** train small "student" model from large "teacher"
- **Caching:** prompt cache, KV cache, embedding cache
- **Speculative decoding:** small draft model + verification

## Distributed training

| Strategy | Splits | Use when |
|---|---|---|
| **Data parallel (DDP)** | batch | Model fits on one GPU |
| **Tensor parallel** | weight matrices | Layer too big |
| **Pipeline parallel** | layers across GPUs | Memory > compute bound |
| **ZeRO / FSDP** | optim states + grads + params | Big models, modest hardware |

## Common monitoring

- Train loss, val loss, gradient norms
- Throughput (tokens/sec, samples/sec)
- GPU util, memory, network IO
- For prod: latency p50/p95/p99, error rate, drift detection

## Eval metrics quick reference

| Task | Metric |
|---|---|
| Binary classif | Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC |
| Multi-class | Macro-F1, Confusion matrix |
| Regression | MAE, RMSE, R² |
| Ranking / IR | nDCG, MAP, MRR, Hit@k |
| Generation | BLEU, ROUGE, BERTScore, perplexity |
| LLM | MMLU, HumanEval, MT-Bench, win-rate, hallucination rate |

---

# 12. Behavioral & Communication

## STAR format for behavioral

> **S**ituation → **T**ask → **A**ction → **R**esult (with metric).

## Answer template for "explain X"

1. **One-sentence definition.**
2. **Why it matters / when used.**
3. **Concrete example.**
4. **Trade-offs / failure modes.**

> Example: "What is attention?"
> 1. Attention computes a token's representation as a weighted sum of all tokens, weights coming from `softmax(QKᵀ/√d)`.
> 2. Replaces the sequential bottleneck of RNNs; unlocks parallel training.
> 3. In a transformer block, every word in the sentence attends to every other word.
> 4. Trade-off: `O(T²)` memory and time. Mitigations: FlashAttention, sparse attention, linear attention, sliding window.

## Coding interview rhythm

1. **Restate problem** in your own words.
2. **Examples** — give one yourself, ask for edge cases.
3. **Brute force first**, state complexity.
4. **Optimize** — narrate the insight.
5. **Code** — clean variables, small functions, comment trade-offs.
6. **Trace / test** — walk through one example.
7. **Discuss complexity & alternatives.**

## Probability problem rhythm

1. **Define your random variable** explicitly: `Let X = ...`
2. **State independence/distribution assumptions.**
3. **Pick the right tool**: linearity / Bayes / conditioning / generating function.
4. **Compute, then sanity-check** (e.g. `E[max] + E[min] = E[sum]`).
5. **Verbalize the threshold rule** for stopping problems.

## Power phrases for live interviews

- "Let me clarify the input format..."
- "I'll start with a brute-force `O(n²)`, then optimize."
- "By linearity of expectation, even though the variables are dependent..."
- "This is a backward induction; let me define `V[k]`..."
- "Two patterns cover spirals: shrinking boundaries vs expanding step."
- "For majority vote, I'll use the binomial sum, **assuming independence** — let me note that."
- "I'll defer this edge case for the moment and revisit."

## Questions to ask your interviewer

- "What's a typical day on this team?"
- "What does the on-call / production-ML burden look like?"
- "What's the team's ratio of research to applied work?"
- "How do you measure success in this role at 6 months?"
- "What's the most challenging engineering problem the team has solved this year?"

---

# Appendix — Last-minute "must-know" numbers

| Concept | Value |
|---|---|
| `e ≈ 2.718`, `1/e ≈ 0.368`, `1 − 1/e ≈ 0.632` | Secretary, sample-driven secretary |
| `π² / 6 ≈ 1.645` | Coupon collector variance |
| `H_n ≈ ln n + 0.577` | Coupon collector mean |
| Llama-2 7B: 32 layers, d=4096, 32 heads | Standard config |
| GPT-3: 175B, 96 layers, d=12288 | Standard |
| Attention complexity: `O(T² d)` time, `O(T²)` memory | (FlashAttention: `O(T)` memory) |
| Adam default `β₁=0.9, β₂=0.999, ε=1e-8` | Memorize |
| Standard transformer initialization: `N(0, 0.02²)` | GPT-style |
| Word2Vec / BERT / GPT-2 hidden sizes: 300 / 768 / 1600 | Reference |

---

> **Final reminder:** When stuck, **say what you're thinking out loud**. Interviewers grade reasoning as much as the answer. State your invariants, name your variables, sanity-check small cases, and acknowledge edge conditions before moving on.

> **Good luck!**
