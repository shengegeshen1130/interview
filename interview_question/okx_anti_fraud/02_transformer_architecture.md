# Transformer 架构深度解析（Transformer Architecture Deep Dive）

> **文档定位：** OKX Anti-Fraud AI Engineer 面试准备系列 File 2 of 4。
>
> **目标读者：** 有 transformer 工作经验（用过 BERT/GPT 做下游任务）、但需要在面试里讲清楚 attention 数学细节和训练 trick 的工程师。本文不重复 "transformer 是什么"，而是把每个组件的 **为什么这样设计、为什么这个数值、和欺诈检测场景的关联** 讲透。
>
> **配套阅读：** `01_*.md`（blockchain data primer）、`03_*.md`（graph ML on blockchain）、`04_*.md`（sequence modeling & fraud detection systems）。

---

## 1. 为什么需要 Transformer（Why Transformers）

---

### 1.1 RNN/LSTM 的局限性

在 2017 年 *Attention Is All You Need* 之前，sequence modeling 主流是 RNN / LSTM / GRU。它们在 NLP 上取得了显著成功，但有几个根本性局限：

| 局限 | 具体表现 | 对欺诈检测的影响 |
|------|---------|------------------|
| **顺序计算瓶颈 (Sequential Bottleneck)** | RNN 必须按时间步串行展开：$h_{t} = f(h_{t-1}, x_{t})$，时间步 $t$ 必须等 $t-1$ 算完。无法在序列维度上并行，训练慢、GPU 利用率低 | 链上一个活跃 address 的 tx history 长度可达 $10^{4}$ 级别，RNN 训练时间不可接受 |
| **长距离依赖退化 (Long-Range Dependency Decay)** | 梯度在时间维度上反向传播时被反复乘以 Jacobian，要么 vanishing 要么 exploding；即便 LSTM 用 gating 缓解，超过 100 步后信息仍严重衰减 | Peel chain、layering 这类 fraud pattern 的 signal 可能跨越数百笔 tx，RNN 学不到 |
| **信息瓶颈 (Information Bottleneck)** | 整段历史被压缩进固定维度的 hidden state $h_{t}$，越往后信息覆盖越严重，早期细节几乎丢失 | 30 天前的一笔可疑入金可能是当前 fraud 的关键 evidence，RNN 的 hidden state 早已遗忘 |
| **LSTM 是缓解不是根本解决** | LSTM 通过 cell state + gating 把"有效记忆距离"从 RNN 的 ~10 步拉长到 ~200 步，但仍然是 $O(\text{length})$ 串行计算 + 单一 bottleneck state | 在 $10^{3}$+ tx 的 sequence 上仍然力不从心 |

### 1.2 Attention 的核心思想

Attention 的革命性在于：**任意两个位置之间建立 direct connection，不再依赖中间步骤传递信息**。

- RNN 里位置 $t$ 想看到位置 $1$ 的信息，必须经过 $t-1$ 次 hidden state transition
- Attention 里位置 $t$ 直接计算 $t$ 和 $1$ 的相似度并 weighted sum，**路径长度恒为 1**

这一点对 fraud detection 极其关键。举个具体例子：
- 30 天前一个 address 收到了来自已知 mixer 的 dust transfer
- 今天该 address 发起一笔大额提现到 OKX
- 这两个事件中间隔了几百笔 noise tx
- **RNN 会把 30 天前的 signal 在 hidden state 里稀释掉；transformer 通过 attention 可以直接把今天的 tx 和 30 天前那笔 dust 关联起来，得出"该 address 已被 mixer 间接污染"的结论**

> **面试 sound bite：** "Transformer 把 sequence modeling 从 'compressed memory + sequential propagation' 范式转成了 'random access + parallel attention' 范式。这对链上行为分析尤其重要，因为关键 signal 经常跨越极长的时间窗口。"

---

## 2. Scaled Dot-Product Attention

---

### 2.1 完整数学定义

Scaled dot-product attention 是 transformer 的核心算子：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^{T}}{\sqrt{d_{k}}}\right)V$$

其中各 tensor 的 shape：
- $Q \in \mathbb{R}^{n \times d_{k}}$：query 矩阵，每行是一个长度 $d_{k}$ 的 query vector，共 $n$ 个 query（query sequence 长度）
- $K \in \mathbb{R}^{m \times d_{k}}$：key 矩阵，每行是一个 key vector，共 $m$ 个 key（key sequence 长度）
- $V \in \mathbb{R}^{m \times d_{v}}$：value 矩阵，每行是一个 value vector，共 $m$ 个 value
- 输出 $\in \mathbb{R}^{n \times d_{v}}$

在 self-attention 中 $n = m$，且 $Q, K, V$ 都是同一序列 $X \in \mathbb{R}^{n \times d_{\text{model}}}$ 通过三个独立的线性投影得到：

$$Q = XW^{Q}, \quad K = XW^{K}, \quad V = XW^{V}$$

### 2.2 语义直觉：Soft Retrieval

把 attention 想象成一个 **可微的、连续版本的 key-value 数据库查询**：

- $Q$：你的"搜索 query"——想找什么
- $K$：每条记录的"索引 label"——能被搜到吗
- $V$：每条记录的"实际内容"——找到后返回什么
- $QK^{T}$：query 和每个 key 的匹配分（dot product 衡量相似度）
- $\text{softmax}(\cdot)$：把匹配分归一化成 attention weight（一个概率分布）
- 最后乘以 $V$：按 attention weight 加权聚合所有 value

和传统 hard retrieval 的区别：传统数据库返回 top-1 精确匹配，attention 返回 **所有记录的 soft 加权和**，这让它可微、可端到端训练。

**对欺诈检测的直观映射：**
- Query = "当前这笔 tx 我想理解它的 fraud risk"
- Keys = "历史上每一笔 tx 的特征"
- Values = "历史上每一笔 tx 带的实际信息（金额、对手方风险、method 类型）"
- Attention 输出 = "在所有历史 tx 中，与当前 tx 'fraud 相关'的部分被高 weight，聚合后形成对当前 tx 的 contextual representation"

### 2.3 为什么除以 $\sqrt{d_{k}}$

这个 $\sqrt{d_{k}}$ 是 transformer 设计里最容易被忽视、但面试官最爱问的细节。

**问题来源：** 假设 $Q$ 和 $K$ 的每个 entry 都是 i.i.d. 服从 $\mathcal{N}(0, 1)$，那么 $Q_{i}$ 和 $K_{j}$ 的 dot product：

$$q \cdot k = \sum_{l=1}^{d_{k}} q_{l} k_{l}$$

每个 $q_{l} k_{l}$ 的均值是 0、方差是 1，$d_{k}$ 个独立项相加后总方差是 $d_{k}$，标准差是 $\sqrt{d_{k}}$。

**这导致的问题：** 当 $d_{k}$ 较大（比如 64、128）时，$QK^{T}$ 的 entry 数值会很大（量级 $\sqrt{d_{k}}$），进入 softmax 后会 **饱和**——某个最大值的 softmax 概率接近 1，其他接近 0。softmax 饱和区域的梯度接近 0，导致：
- 训练初期梯度消失，模型学不动
- attention 退化成 hard one-hot 选择，丧失"soft 加权"的好处

**解决方案：** 除以 $\sqrt{d_{k}}$ 把方差 normalize 回 1，让 softmax input 保持在合理动态范围内：

$$\text{Var}\left(\frac{q \cdot k}{\sqrt{d_{k}}}\right) = \frac{d_{k}}{d_{k}} = 1$$

> **面试细节：** 注意是除以 $\sqrt{d_{k}}$ 而不是 $d_{k}$。除以 $d_{k}$ 会让 dot product 的标准差变成 $1/\sqrt{d_{k}}$，依然不是 unit variance。"variance normalization → 除以 std → 除以 $\sqrt{d_{k}}$" 是这个设计的完整理由链。

### 2.4 计算复杂度

Self-attention 的复杂度（设 sequence 长度 $n$，hidden dim $d$）：
- **时间复杂度：** $QK^{T}$ 是 $n \times d$ 乘 $d \times n$，得 $n \times n$ 矩阵，需要 $O(n^{2} \cdot d)$ FLOPs；softmax 和最后乘以 $V$ 是 $O(n^{2} \cdot d)$；总计 $O(n^{2} \cdot d)$
- **空间复杂度：** $QK^{T}$ 这个 $n \times n$ attention matrix 是显式存储的，$O(n^{2})$ memory

**对长序列的瓶颈：**
- $n = 1024$：attention matrix 1M entry，可接受
- $n = 8192$：attention matrix 64M entry，单层就吃几百 MB
- $n = 32768$：attention matrix 1G entry，单层 GB 级 memory，单卡放不下

**这就是为什么链上长序列建模一定要用 efficient attention：**
- **Sparse attention（Longformer、BigBird）：** 限制每个 token 只 attend 到邻域 + 少量 global token，$O(n \cdot w)$
- **Linear attention（Performer、Linformer）：** kernel approximation 把 $\text{softmax}(QK^{T})V$ 改写成 $\phi(Q)(\phi(K)^{T}V)$，$O(n \cdot d^{2})$
- **FlashAttention：** 不改变数学定义，只改 GPU memory access pattern，把 IO complexity 从 $O(n^{2})$ 降到 $O(n^{2} d / M)$（其中 $M$ 为 SRAM 大小，$d$ 为 head dim），实际 wall-clock 提速 2-4×

---

## 3. Multi-Head Attention（多头注意力）

---

### 3.1 数学定义

Single-head attention 只能在一个 representation subspace 里做聚合。Multi-head attention 把 $d_{\text{model}}$ 拆成 $h$ 个 head，每个 head 在低维子空间里独立做 attention，最后 concat 起来：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_{1}, \ldots, \text{head}_{h}) W^{O}$$

$$\text{head}_{i} = \text{Attention}(QW^{Q}_{i}, KW^{K}_{i}, VW^{V}_{i})$$

其中：
- $W^{Q}_{i}, W^{K}_{i} \in \mathbb{R}^{d_{\text{model}} \times d_{k}}$
- $W^{V}_{i} \in \mathbb{R}^{d_{\text{model}} \times d_{v}}$
- $W^{O} \in \mathbb{R}^{h \cdot d_{v} \times d_{\text{model}}}$
- 一般 $d_{k} = d_{v} = d_{\text{model}} / h$，所以 multi-head 的总参数量 = single-head ($d_{\text{model}} \times d_{\text{model}}$) 大致相当

### 3.2 为什么要多头：subspace decomposition

Single attention head 学到的 attention pattern 是 **一种** 加权方式。但真实世界的 token 关联是 **多重的**：一个 token 可能同时和上下文有 syntactic 关系、semantic 关系、coreference 关系等。Single head 强行把这些关系压在一个 attention map 里，会互相干扰。

Multi-head 的设计：
- 每个 head 投影到一个独立的低维 subspace（$d_{k} = d_{\text{model}}/h$）
- 每个 head 在自己 subspace 里学一种 attention pattern
- 不同 head 可以学到 **不同类型** 的关联，互不干扰
- 最后 concat + linear 让模型学会如何 fuse 这些不同视角

### 3.3 不同 head 学到什么（empirical findings）

NLP 上的 probing 研究（Clark et al. 2019, "What Does BERT Look At?"）发现 BERT 不同 head 学到的 attention pattern 大致可以分为：

| Head 类型 | Attention 模式 | 例子 |
|----------|---------------|------|
| **局部 head** | Attend 到 $\pm 1$ 邻居 | 学到 n-gram-like local context |
| **句法 head** | Attend 到 syntactic head（被动词的主语、被介词的宾语） | 学到 dependency parse 结构 |
| **长距离 head** | Attend 到远距离 token，常对应 coreference | "He" attend 回到几句之前的 "John" |
| **语义 head** | Attend 到语义相关 token（不一定句法相关） | "bank" attend 到 "river" 或 "money" |
| **[SEP]/位置 head** | Attend 到特殊 token 或固定位置 | 部分 head 退化成"广播 [SEP]"模式 |

### 3.4 欺诈检测场景的类比

把 multi-head 概念迁移到 transaction sequence model 上，可以期待不同 head 学到：

| Head 类型 | 学到的关联 | 对应 fraud pattern |
|----------|-----------|-------------------|
| **Amount-pattern head** | Attend 到金额相近的历史 tx | 检测 layering 中"快进快出相同金额" |
| **Timing head** | Attend 到时间窗口内（如同一小时）的 tx | 检测 burst activity、bot-like regular interval |
| **Counterparty head** | Attend 到相同对手方/相同 cluster 的历史 tx | 检测与已知 mixer/sanctioned cluster 的重复交互 |
| **Method head** | Attend 到相同 method_id 的 tx | 检测高频 approve/swap pattern |
| **Cross-field head** | 学到 field 之间的 interaction（如 high gas + new contract = 抢跑） | 复合特征学习 |

实际生产模型里，**强烈建议在训练后做 head visualization**：把不同 fraud 样本的 attention map 画出来，能定性验证模型确实学到了"看哪些历史 tx 来判断当前 tx 风险"的语义。这也是 explainability 工具的基础。

---

## 4. 位置编码（Positional Encoding）

---

Transformer 本身是 permutation-equivariant 的——你打乱 input token 顺序，输出（除了对应位置）不会变。但 sequence 任务里位置信息是核心 signal（"先发生的 tx 不能受后发生的影响"），所以必须显式注入位置信息。

下面 4 种 position encoding 是面试高频问点，每一种都要能说清楚 **优势、劣势、用在哪个 model**。

### 4.1 正弦位置编码（Sinusoidal Positional Encoding，原始 Transformer）

原始 *Attention Is All You Need* 提出，对位置 $\text{pos}$ 和维度 $i$（$0 \leq i < d_{\text{model}}/2$）：

$$\text{PE}(\text{pos}, 2i) = \sin\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

$$\text{PE}(\text{pos}, 2i+1) = \cos\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

**关键性质：**
- **无可学习参数：** 直接由公式定义，省参数、省训练数据
- **可外推 (extrapolation)：** 因为是 closed-form 函数，理论上可以算任意长度的 PE；但实际效果不如 RoPE
- **相对位置可被学到：** 由三角恒等式 $\sin(a+b)$ 可分解为 $\sin a$ 和 $\cos b$ 的线性组合，模型可以通过线性变换学到 relative position 信息
- **使用方式：** $X' = X + \text{PE}$（element-wise add 到 token embedding）

### 4.2 可学习位置编码（Learned Positional Embedding，BERT/GPT）

BERT 和 GPT-1/2 用的方案：直接为每个位置学一个 embedding vector：

$$\text{PE}_{\text{pos}} \in \mathbb{R}^{d_{\text{model}}}, \quad \text{pos} \in \{0, 1, \ldots, L_{\max} - 1\}$$

**关键性质：**
- **简单灵活：** 给模型最大自由度去学位置 representation
- **不能外推：** 训练时 $L_{\max} = 512$，推理时输入 1024 长度的序列会直接报 index out of range，必须做插值或重新训练
- **位置语义被吃在 embedding 里：** 但每个位置独立学，没有 inductive bias 让相近位置 embedding 也相近，需要靠数据驱动

**为什么 BERT 选 learned：** 训练数据量大，learned 比 sinusoidal 略好；且 BERT 只针对固定长度训练，不需要外推能力。

### 4.3 旋转位置编码 RoPE（Rotary Position Embedding，LLaMA / Qwen / DeepSeek）

RoPE 是 Su et al. 2021 提出，现在几乎所有 SOTA LLM 都用它。核心思想：**把位置编码作为 query 和 key 的旋转，而不是加法**。

**数学定义（2D 情形）：** 对于位置 $m$ 和维度对 $(2i, 2i+1)$，把 $q$ 或 $k$ 的这两个分量看成复数 $q_{2i} + i q_{2i+1}$，乘以 $e^{i m \theta_{i}}$，其中 $\theta_{i} = 10000^{-2i/d}$。

等价于对每对相邻维度做 2D 旋转矩阵：

$$\begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos m\theta_{i} & -\sin m\theta_{i} \\ \sin m\theta_{i} & \cos m\theta_{i} \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$

**关键性质：**
- **相对位置自然嵌入 attention score：** 因为 $\langle R_{m} q, R_{n} k \rangle = \langle q, R_{n-m} k \rangle$，旋转后的 dot product 只依赖于 $n - m$（相对位置），不依赖绝对位置。Attention 计算时自动包含 relative position
- **不加在 token embedding 上、加在 Q 和 K 上：** 这是和 sinusoidal/learned 的最大区别。Token embedding 保持纯净，位置信号只在 attention 阶段起作用
- **强外推能力：** 配合 NTK-aware scaling 或 YaRN 等方法，训练在 4K context 的模型可以外推到 32K-128K
- **欺诈检测优势：** 链上 sequence 里 **相对时间间隔** 远比绝对位置重要（"这笔 tx 距离上一笔 5 秒还是 5 天" 决定行为含义），RoPE 的相对位置 bias 天然契合这一点

### 4.4 ALiBi（Attention with Linear Biases，BLOOM / MPT）

Press et al. 2021 提出的极简方案：**不做 position embedding，直接在 attention score 上加一个跟距离成正比的 penalty**。

$$\text{attention\_score}(i, j) = \frac{q_{i} \cdot k_{j}}{\sqrt{d_{k}}} - m \cdot |i - j|$$

其中 $m$ 是 head-specific 的固定 slope（不同 head 用不同 slope，例如 $m_{h} = 2^{-8h/H}$）。

**关键性质：**
- **零额外参数：** 比 sinusoidal 还简洁
- **极强外推：** ALiBi 训练在 1K context 可以外推到 16K 并保持效果，是 RoPE 之前外推能力最强的方案之一
- **强 inductive bias：** "距离越远 attention 越弱" 是 hardcoded 的，对于绝大多数 sequence task 是合理的；但对需要长距离强 attention 的 task（如 retrieval-aug）会损害效果
- **缺点：** 表达能力不如 RoPE，无法表示"远距离但仍然强相关"的关系（比如 30 天前的关键 tx 现在仍然重要）

### 4.5 选型对比表

| 方案 | 参数量 | 外推能力 | 相对位置 | 主流采用 |
|------|-------|---------|---------|----------|
| **Sinusoidal** | 0 | 弱 | 隐式可学 | 原始 Transformer |
| **Learned** | $L_{\max} \times d$ | 无 | 无 | BERT、GPT-1/2/3 |
| **RoPE** | 0 | 强 | 显式 | LLaMA、Qwen、DeepSeek、Mistral |
| **ALiBi** | 0（仅 $h$ 个 slope） | 极强 | 隐式（距离 penalty） | BLOOM、MPT |

> **面试 sound bite：** "做交易序列建模我会选 RoPE，因为 ① 链上 sequence 里相对时间差比绝对位置更有语义；② 活跃 address 的 tx 序列长度可能远超训练时见过的，需要外推能力；③ RoPE 把位置注入 Q/K 而不污染 value，对 multi-field token embedding 干扰最小。"

---

## 5. Transformer Block 组件详解

---

一个标准 transformer block 包含：multi-head attention → residual + layer norm → FFN → residual + layer norm。下面逐个拆解。

### 5.1 Layer Normalization

**公式：** 对一个 token 的 hidden vector $x \in \mathbb{R}^{d}$：

$$\mu = \frac{1}{d} \sum_{i=1}^{d} x_{i}, \quad \sigma^{2} = \frac{1}{d} \sum_{i=1}^{d} (x_{i} - \mu)^{2}$$

$$\text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^{2} + \epsilon}} + \beta$$

其中 $\gamma, \beta \in \mathbb{R}^{d}$ 是可学习的 scale 和 shift。

**为什么是 LayerNorm 而不是 BatchNorm：**

| 对比维度 | BatchNorm | LayerNorm |
|---------|-----------|-----------|
| **归一化维度** | 跨 batch 同一个 channel/feature | 单个 sample 内的所有 hidden dim |
| **依赖 batch size** | 是（小 batch 时统计不稳） | 否 |
| **依赖 sequence 长度** | 是（不同长度 padding 后统计被污染） | 否 |
| **训练 vs 推理一致性** | 不一致（用 running mean/var） | 一致 |
| **NLP/sequence 适用性** | 差（变长序列 + 小 batch + 同位置不同 sample 语义差异巨大） | 好 |

LayerNorm 对每个 token 独立计算，不跨 sample / 不跨 position，所以对 variable-length sequence 完全无副作用，是 transformer 的标配。

**Pre-Norm vs Post-Norm：** 这是面试高频细节。

| 形式 | 公式 | 优点 | 缺点 | 代表模型 |
|------|------|------|------|---------|
| **Post-Norm**（原始） | $x_{\text{out}} = \text{LN}(x + \text{Sublayer}(x))$ | 表达能力略强 | 训练不稳定，需 warmup + 小 lr；深层容易 NaN | 原始 Transformer、BERT |
| **Pre-Norm** | $x_{\text{out}} = x + \text{Sublayer}(\text{LN}(x))$ | 训练稳定，深层可堆 100+ 层，可用更大 lr，需要的 warmup 大幅缩短（几百 step 即可） | 表达能力略弱（可通过加宽/加深补偿） | GPT-2/3/4、LLaMA、几乎所有现代 LLM |

**为什么 pre-norm 更稳定：** Pre-norm 让残差连接的"identity path"始终是 untouched 的 $x$，梯度可以无衰减地传到底层；post-norm 里残差和 sublayer 输出加起来后再 norm，深层梯度会被反复 norm 放缩，容易爆。

> **2026 年现状：** 主流 LLM 几乎 100% 用 pre-norm，且很多模型（LLaMA 系）用 **RMSNorm** 代替 LayerNorm（去掉 mean 计算只保留 scale），计算更快且效果几乎一致。

### 5.2 Feed-Forward Network（FFN）

每个 transformer block 在 attention 之后接一个 position-wise FFN：

$$\text{FFN}(x) = \max(0, xW_{1} + b_{1}) W_{2} + b_{2}$$

其中：
- $W_{1} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$（升维）
- $W_{2} \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$（降维回原维）
- 通常 $d_{\text{ff}} = 4 \cdot d_{\text{model}}$（如 $d_{\text{model}} = 768 \Rightarrow d_{\text{ff}} = 3072$）

**为什么需要 FFN（在已经有 attention 的情况下）：**
- **Attention 是线性的 (modulo softmax)：** Attention 本质是 weighted sum，对单个 token 来说没有非线性变换能力。FFN 引入非线性（ReLU / GELU），让模型能学复杂函数
- **位置独立（position-wise）：** FFN 对每个 token 独立施加（同样的 $W_{1}, W_{2}$），不在序列维度做交互；attention 负责"跨位置"信息流，FFN 负责"位置内"特征加工——分工明确

**为什么中间维度是 $4 \times d_{\text{model}}$：**
- **经验最优：** 原 transformer 设计选了 4×，后续大量实验（包括 scaling law 研究）发现 ratio 在 2.5-4 之间都接近最优，4× 是 sweet spot
- **参数容量：** FFN 占 transformer block 总参数量的 $\frac{2 \times 4d^{2}}{4d^{2} + 8d^{2}} = \frac{2}{3}$，是模型主要的 knowledge 存储位置（attention 是 routing 机制，FFN 是 knowledge memory）
- **太小（ratio<2）：** 容量不够，loss 下不去
- **太大（ratio>8）：** 边际收益递减，且 memory/compute 浪费

**GLU 家族（现代变体）：** LLaMA / PaLM 等用 **SwiGLU** 替代标准 FFN：

$$\text{SwiGLU}(x) = (\text{Swish}(xW_{1}) \odot xW_{3}) W_{2}$$

引入了 gating 机制（$xW_{3}$ 作为 gate），表达能力更强；为了保持参数量不变，通常把 $d_{\text{ff}}$ 从 $4d$ 缩到 $\frac{8}{3}d \approx 2.67d$，因为多了一个 $W_{3}$ 矩阵。SwiGLU 在等参数预算下持续优于 ReLU FFN，目前是 LLM 默认选择。

### 5.3 残差连接（Residual Connection）

每个 sublayer 都包在 residual 里：

$$x_{\text{out}} = x_{\text{in}} + \text{Sublayer}(x_{\text{in}})$$

**作用：**
- **梯度高速公路：** $\frac{\partial x_{\text{out}}}{\partial x_{\text{in}}} = I + \frac{\partial \text{Sublayer}}{\partial x_{\text{in}}}$，identity term 保证梯度可以无衰减地传到底层，让 100+ 层的 transformer 可训
- **保持信息：** 即便 sublayer 学得不好（output 接近 0），$x_{\text{out}} \approx x_{\text{in}}$，至少不会丢信息
- **优化平面 smoothing：** 实验上残差连接显著降低 loss surface 的 sharpness，让 SGD/Adam 更容易收敛

---

## 6. Encoder vs Decoder vs Encoder-Decoder

---

Transformer 家族按架构可分为三类，**每类的 self-attention 是否带 mask 是关键区别**。

| 架构 | 代表模型 | Self-Attention 类型 | 典型任务 | 欺诈检测适用场景 |
|------|---------|---------------------|---------|------------------|
| **Encoder-only** | BERT、RoBERTa、DeBERTa、ELECTRA | Bidirectional（全可见） | 分类、NER、句子表示 | 离线 batch 评分一段交易序列（已知完整 history） |
| **Decoder-only** | GPT 系列、LLaMA、Qwen、Claude、Mistral | Causal（只看左侧） | 生成、自回归预测、LLM | Real-time streaming detection（边收 tx 边评分） |
| **Encoder-Decoder** | 原始 Transformer、T5、BART、mT5 | Encoder bi-dir + Decoder causal + cross-attention | 翻译、摘要、seq2seq | 跨 sequence 任务（如把可疑 tx 序列翻译成 fraud explanation） |

### 6.1 Encoder-only：双向 self-attention

每个位置可以 attend 到所有其他位置（左右都看）：

- 优点：完整 context、对分类/表示学习友好
- 缺点：不能直接做生成（不能自回归 predict next token）
- 训练目标：MLM（masked language modeling）—— 随机 mask 15% token 让模型 predict

**对欺诈检测：** Encoder-only 是 **离线 fraud classification** 的最佳选择。给定 address 完整 tx history（或最近 K 笔），输出 risk score 或 fraud type 分类。BERT-style 的 `[CLS]` token 设计完美适配序列分类。

### 6.2 Decoder-only：causal masked self-attention

位置 $i$ 只能 attend 到位置 $\leq i$，通过 **upper triangular mask** 实现：

$$\text{attention\_score}(i, j) = \begin{cases} \frac{q_{i} \cdot k_{j}}{\sqrt{d_{k}}} & j \leq i \\ -\infty & j > i \end{cases}$$

实现上：构造一个 $n \times n$ 的 mask matrix $M$，上三角部分（$j > i$）填 $-\infty$，下三角和对角填 0；attention score $+ M$ 后过 softmax，被 mask 位置的 softmax 输出严格为 0。

- 优点：天然支持自回归生成；可以做 next-event prediction
- 缺点：每个位置看不到右侧，表示能力略弱于 bi-dir（但 scaling 起来后差距消失）

**对欺诈检测：** Decoder-only 是 **real-time streaming detection** 的最佳选择。每来一笔新 tx，只用左侧（历史）信息预测当前 tx 的 risk，不用 right-context（也确实没有），符合生产 latency 要求。GPT-style next-event prediction 还能直接 pretrain。

### 6.3 Encoder-Decoder：两阶段架构

Encoder 处理 source sequence，decoder 自回归生成 target sequence。Decoder 里除了 causal self-attention，还有 **cross-attention**：$Q$ 来自 decoder hidden state，$K, V$ 来自 encoder output。

- 优点：明确分离"理解"和"生成"，对 seq2seq task 自然
- 缺点：架构复杂、参数量大；纯生成任务上被 decoder-only 超越

**对欺诈检测：** 通常 overkill，除非有明确 seq2seq 需求（如给定可疑 tx sequence，生成 human-readable 解释 / SAR 报告草稿）。

### 6.4 Masked Self-Attention 实现细节

```python
# 伪代码：causal mask
n = sequence_length
mask = torch.triu(torch.ones(n, n), diagonal=1).bool()  # 上三角为 True
scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)      # (n, n)
scores = scores.masked_fill(mask, float('-inf'))
attn = F.softmax(scores, dim=-1)
out = attn @ V
```

实际生产里 FlashAttention 把 mask 融合进 kernel，不显式 materialize $n \times n$ matrix，省显存。

---

## 7. 训练细节（Training Details）

---

### 7.1 预训练目标对比

| 目标 | 全称 | 代表模型 | 机制 | 数据效率 |
|------|------|---------|------|---------|
| **MLM** | Masked Language Modeling | BERT、RoBERTa | 随机 mask 15% token，用 bi-dir context 预测被 mask 的 token | 中等（每个样本只学 15% 位置） |
| **CLM** | Causal Language Modeling | GPT 系列、LLaMA | 自回归预测下一个 token，loss 在每个位置都算 | 高（每个 token 都贡献 loss） |
| **Span Corruption** | Span Masking | T5、UL2 | 随机选 span（连续若干 token）替换成 sentinel，让 decoder 重建原 span | 适合 seq2seq |
| **RTD** | Replaced Token Detection | ELECTRA | 用一个小 generator 替换部分 token，让 discriminator 判断每个 token 是否被替换 | 极高（每个 token 都有 label） |

**MLM 细节（用于欺诈场景类比）：**
- BERT 原始 MLM：随机选 15% token，其中 80% 替换为 `[MASK]`、10% 替换为随机 token、10% 保持不变，model 预测原 token
- 为什么 15%：太低 → 学习信号稀疏；太高 → 上下文信息丢失太多无法预测
- 为什么 80/10/10：避免 train-test mismatch（test 时输入不会有 `[MASK]`）

### 7.2 欺诈检测的预训练：MLM on Transaction Sequences

把 MLM 思路迁移到 tx sequence：
- **Vocabulary：** discrete field 的取值（method_id、counterparty_cluster、token_id 等），每个 field 各有 vocabulary
- **Masking：** 随机 mask 15% tx，预测其 discrete field（method、counterparty 等）；continuous field 可以用 regression head 预测
- **优势：** 链上 unlabeled tx 数据巨大（Ethereum 单链就有 $> 10^{9}$ tx），有 label 的 fraud 极稀缺。MLM 在海量 unlabeled data 上 pretrain，然后用少量 labeled fraud 做 fine-tune，是典型的 label-scarce 场景最优范式
- **变体：** 可以 mask 整个 tx token，也可以只 mask 某个 field（比如保持 counterparty 可见，预测 method）——后者学到的 representation 更聚焦于行为模式

### 7.3 Fine-tuning 实践细节

- **Learning rate：** Pre-training lr 通常 $1\times 10^{-4}$ 量级，fine-tuning 必须降到 $2\times 10^{-5}$ ~ $5\times 10^{-5}$。太大会破坏 pretrain learned representation
- **Warmup：** 几百 step 的 linear warmup，避免初期 update 把 pretrain weight 打乱
- **Catastrophic forgetting：** 长 fine-tune 后模型可能丢失 pretrain 阶段的 general knowledge。缓解方法：① 加 KL divergence regularization 限制偏离 pretrain logits；② Adapter / LoRA 只 fine-tune 少量参数；③ Multi-task fine-tune 同时保留 MLM loss
- **Layer-wise lr decay：** 越底层 lr 越小（如 each layer × 0.95），底层学的是 generic feature 不该被破坏，顶层学 task-specific
- **数据量门槛：** Fine-tune 需要的标注量取决于 task 难度和 pretrain 质量，fraud classification 一般 $10^{3} \sim 10^{4}$ 标注样本可达到生产可用水平

### 7.4 长序列的 Computational Implications

链上一个活跃 address 可能有 $10^{4}$ 笔 tx。标准 transformer 的 $O(n^{2})$ memory 在 $n = 10^{4}$ 时是 $10^{8}$ entry——单层 attention matrix 就要 400 MB（fp32）或 200 MB（fp16），单卡放不下整模型。

**工程对策：**
- **Truncation + sliding window：** 只取最近 K 笔，K = 512 或 1024，配 RoPE 做长度外推
- **Hierarchical encoding：** 先把 long sequence 切 chunk，每 chunk 内做 attention 得到 chunk embedding，再在 chunk embedding 上做更高层 attention
- **Efficient attention：** Longformer / BigBird 的 sparse attention 把 $O(n^{2})$ 降到 $O(n \cdot w)$
- **FlashAttention：** 不改变数学定义，IO-aware 实现，实测 2-4× 提速 + 显存大幅下降
- **State Space Models (Mamba)：** 完全替代 attention，$O(n \cdot d)$ 线性复杂度，在极长 sequence 上极有竞争力，2024-2025 年是热门方向

---

## Interview Q&A

---

### Q1: 详细描述 Scaled Dot-Product Attention 的完整计算过程，以及为什么要除以 $\sqrt{d_{k}}$。

**回答：**

Scaled dot-product attention 是 transformer 的核心算子，整个计算可以拆成 5 步，关键是搞清楚每步的 shape 和数值稳定性考虑。

1. **三个输入的来源**：在 self-attention 里，输入是一个 $X \in \mathbb{R}^{n \times d_{\text{model}}}$（$n$ 是 sequence 长度），通过三个独立的线性投影 $W^{Q}, W^{K}, W^{V}$ 得到 $Q, K, V$。$Q \in \mathbb{R}^{n \times d_{k}}$、$K \in \mathbb{R}^{n \times d_{k}}$、$V \in \mathbb{R}^{n \times d_{v}}$，通常 $d_{k} = d_{v} = d_{\text{model}}/h$（$h$ 是 head 数）。

2. **相似度矩阵 $QK^{T}$**：$Q$ 和 $K$ 矩阵相乘得到 $n \times n$ 的 score matrix $S = QK^{T}$，$S_{ij}$ 是 query $i$ 和 key $j$ 的 dot product 相似度。这一步是 $O(n^{2} d_{k})$ FLOPs，也是 attention 的复杂度瓶颈。

3. **缩放 $\frac{S}{\sqrt{d_{k}}}$**：这是关键步骤，下面详细解释。假设 $q$ 和 $k$ 的每个 entry 都是 i.i.d. 零均值单位方差，那么 dot product $q \cdot k = \sum_{l=1}^{d_{k}} q_{l} k_{l}$ 的方差是 $d_{k}$，标准差是 $\sqrt{d_{k}}$。当 $d_{k}$ 较大（64、128）时，未缩放的 $S$ entry 数值就会很大，**进入 softmax 时会饱和**——softmax 对大数值非常敏感，最大值的输出会接近 1、其他接近 0，等价于 hard one-hot 选择；更严重的是 softmax 饱和区域梯度接近 0，**反向传播时整个 attention 学不到东西**。除以 $\sqrt{d_{k}}$ 把方差 normalize 回 1，让 softmax 输入在合理动态范围内，避免饱和。注意是除以 std（$\sqrt{d_{k}}$）而不是除以 $d_{k}$ 本身——除以 $d_{k}$ 会让方差变成 $1/d_{k}$，依然不是 unit。

4. **Masking（如果需要）**：Decoder 里 causal mask 把上三角填 $-\infty$，被 mask 的 softmax 输出严格为 0；padding mask 同理。

5. **Softmax + 加权聚合**：对每行做 softmax 得到 attention weight matrix $A \in \mathbb{R}^{n \times n}$，每行是一个 probability distribution；最后 $A V$ 得到 $n \times d_{v}$ 的输出，第 $i$ 行是用 attention weight 加权聚合所有 $V$ 行的结果，可以理解为 "query $i$ 看完所有 key 后从 value 库里 soft-retrieve 出的相关内容"。

> **Follow-up 提示：** 面试官可能追问"如果不除以 $\sqrt{d_{k}}$ 会有什么具体后果？"——答：实测在 $d_{k} = 64$ 时，未缩放的 transformer 训练初期 loss 几乎不下降，gradient norm 比 scaled 版本小 1-2 个数量级，可以现场画 softmax 输入分布说明饱和现象。还可能问"为什么用 dot product 不用其他相似度？"——答：dot product 可矩阵化（$QK^{T}$ 一次完成所有 pair），相比 additive attention（$v^{T} \tanh(W_{1}q + W_{2}k)$）在 GPU 上效率高 3-5×，且 scaling 后效果几乎相当。

---

### Q2: Multi-Head Attention 中，多个 head 分别学到了什么？为什么不直接用一个大 head？

**回答：**

Multi-head 的设计哲学是 "subspace decomposition + parallel pattern learning"，下面从动机、empirical 发现、和欺诈场景三个角度展开。

1. **为什么不用单大 head**：一个 single head 在维度 $d_{\text{model}}$ 上做 attention，理论上参数量和 $h$ 个 $d_{\text{model}}/h$ 的 head concat 起来相当。但 **single head 强制把所有类型的关联压在一个 attention map 里**——syntactic 关联、coreference 关联、local context 关联会互相干扰，model 只能学到一个 compromised 的 attention pattern。Multi-head 让每个 head 在独立 subspace 里学一种关联，**用 architectural inductive bias 实现 attention 的多任务分解**。

2. **每个 head 实际学到什么（empirical findings）**：BERT 上的 probing 研究发现不同 head 学到 distinctive pattern：① 局部 head：attend 到 $\pm 1$ 邻居，学 n-gram；② 句法 head：attend 到 syntactic head（如动词的主语、介词的宾语），相当于学到 dependency parse；③ 长距离 head：attend 到几句之前，学 coreference；④ 语义 head：attend 到语义相关但句法无关的 token；⑤ 特殊功能 head：attend 到 `[SEP]` 或固定位置，作为 "no-op" 或 broadcast 通道。研究还发现不同 head 的重要性差异很大——可以 prune 掉 30-50% head 几乎不影响下游 task（Voita et al. 2019），说明 multi-head 提供了 redundancy 和 specialization 的灵活性。

3. **理论视角：rank 与表达能力**：Single $d \times d$ attention 在 softmax 之后的 rank 受限；multi-head concat 后等价于 $h$ 个低秩 attention 的混合，可以 represent 更高效的 attention 结构。这一点在 transformer 表达能力的理论分析里也有支持（multi-head 不只是工程 trick）。

4. **欺诈检测场景的类比**：把 multi-head 概念迁移到 tx sequence model，可以期待不同 head 学到 ① Amount head：attend 到金额相近的历史 tx，捕捉 layering 中"相同金额快进快出"；② Timing head：attend 到时间窗口内的 tx，识别 burst pattern；③ Counterparty head：attend 到相同对手方/cluster 的 tx，发现 mixer 重复交互；④ Method head：attend 到相同 method_id 的 tx，识别高频 approve/swap 行为；⑤ Cross-field head：学到 field 之间的 interaction。生产模型训练后做 head visualization 是 explainability 的基础工具。

5. **工程实践注意点**：① head 数选择：通常 $h \in \{8, 12, 16, 32\}$，$d_{\text{model}}/h$ 不宜过小（< 32 时表达能力受限）；② 不同 head 的训练动态可能差异巨大，部分 head 早期就退化为 attend-to-[SEP] 模式，这本身不是 bug 而是模型自适应；③ Multi-Query Attention (MQA) 和 Grouped-Query Attention (GQA) 是现代变体，把 K/V 在多 head 间共享或分组共享，省 KV cache 显存且效果几乎不损，是 LLaMA-2/3 等模型的 standard。

> **Follow-up 提示：** 面试官可能问"既然有 head 可以 prune，是不是说明 multi-head 有冗余、设计不优？"——答：冗余有两面性，pretraining 阶段冗余提供了 ensemble 效应和 robustness，让训练更稳定；部署阶段确实可以 prune 来节省 inference cost（用 head importance score 排序后剪枝），但训练阶段不能去掉。还可能问"MQA / GQA 和 MHA 的区别？"——答：MHA 是每个 head 独立的 Q/K/V；MQA 是所有 head 共享 K/V，只有 Q 是 head-specific；GQA 是 head 分组、组内共享 K/V，是 MHA 和 MQA 的折中。MQA 在 LLM inference 阶段省 KV cache 极有效，但训练 quality 略差，GQA 是当前最优 trade-off。

---

### Q3: Pre-Norm 和 Post-Norm 有什么区别？现在为什么主流用 Pre-Norm？

**回答：**

Pre-Norm 和 Post-Norm 的差别看起来只是 LayerNorm 位置不同，但对训练稳定性影响极大，是 transformer 演化中一个非常 educational 的设计选择。

1. **数学定义**：设 Sublayer 是 attention 或 FFN，残差连接的两种 norm placement 公式：
   - **Post-Norm（原始 Transformer / BERT）：** $x_{\text{out}} = \text{LN}(x + \text{Sublayer}(x))$
   - **Pre-Norm（GPT-2+ / LLaMA / 几乎所有现代 LLM）：** $x_{\text{out}} = x + \text{Sublayer}(\text{LN}(x))$
   关键区别：post-norm 把 residual 加完后整体 norm；pre-norm 只 norm sublayer 输入，残差路径 untouched。

2. **训练稳定性差异**：Pre-norm 的 **identity path 是 untouched 的 $x$**，从顶层到底层有一条无衰减的梯度通路；post-norm 里每经过一个 LN，梯度会被 LN 的 Jacobian 放缩，多层叠加后梯度 norm 在深层网络里会指数级变化，要么爆炸要么消失。实验上 24 层 post-norm transformer 在不做 careful warmup 的情况下会直接 NaN，而 100+ 层 pre-norm transformer 用标准 warmup 就能稳定训练。这也是为什么 GPT-3（96 层）、LLaMA-65B（80 层）、GPT-4（更多）都用 pre-norm。

3. **学习率与 warmup 要求**：
   - **Post-norm：** 必须 warmup（数千 step linear），lr 上限较低（约 $1\times 10^{-4}$ 量级）；warmup 不到位直接发散
   - **Pre-norm：** Warmup 几百 step 就够，lr 上限可以高 2-5×（$3\times 10^{-4}$ 量级），训练更省时间

4. **表达能力 trade-off**：理论分析（Xiong et al. 2020）指出 post-norm 的表达能力略强——因为它的 sublayer 输出被 norm 后混入 residual，参数更新有更大幅度的影响；pre-norm 的 sublayer 输出加在 untouched residual 上，单步影响幅度被限制，深层有 "implicit smaller step size" 效果。但这个表达能力差距可以通过加深/加宽模型很容易补偿，**实践中 pre-norm 的稳定性收益远超表达力损失**。

5. **现代变体**：
   - **RMSNorm（LLaMA 系）**：去掉 LayerNorm 里的 mean 计算，只保留 RMS scale + learnable gain：$\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\sqrt{\frac{1}{d}\sum x_{i}^{2} + \epsilon}}$。计算更快（少一次 reduction），效果与 LayerNorm 几乎一致，是 LLaMA 默认选择。
   - **DeepNorm（Microsoft）**：post-norm 的改进版，通过 careful 的初始化和 scaling 让 1000 层 post-norm transformer 也能稳定训，但工程复杂度高，未被主流采用。

6. **欺诈检测训练的 pragmatic 选择**：长 tx sequence 上叠 12-24 层 transformer 是常见配置。直接用 pre-norm + RMSNorm + RoPE + SwiGLU 这套 "LLaMA recipe" 是最 robust 的起点；如果遇到训练发散，第一时间检查的不是模型本身而是 lr / warmup / data quality。

> **Follow-up 提示：** 面试官可能问"既然 pre-norm 这么好，为什么 BERT 当年选了 post-norm？"——答：BERT 时代 ① 层数浅（12-24 层）post-norm 的不稳定问题没那么尖锐；② 配套 warmup schedule 已经 robust；③ post-norm 的轻微表达力优势在那个 scale 上仍有意义。直到 GPT-2 把层数推到 48+ 才让 post-norm 暴露出致命问题，自此 pre-norm 成为标准。还可能问"有没有 pre-norm 解决不了的问题？"——答：极深网络（100+ 层）的 pre-norm 在顶层会出现 "representation collapse"（token representations 趋向相似，即 rank collapse / representation degeneration），需要配合 sandwich layer norm 或 DeepNorm-style scaling，但这是 frontier 问题，绝大多数生产模型不会触及。

---

### Q4: Transformer 的时间/空间复杂度是多少？为什么对长序列是瓶颈？

**回答：**

Transformer 复杂度是面试必考题，关键是要分清 attention 和 FFN 各自的复杂度，以及 memory 和 compute 的不同瓶颈。

1. **Self-attention 复杂度**：设 sequence 长度 $n$，hidden dim $d$（这里 $d = d_{\text{model}}$，假设 head dim $d_{k} \approx d/h$）：
   - **Time：** $QK^{T}$ 是 $(n \times d) \times (d \times n)$，得 $n \times n$ 矩阵，FLOPs 是 $O(n^{2} d)$；softmax 是 $O(n^{2})$；$\text{attn} \times V$ 是 $(n \times n) \times (n \times d) = O(n^{2} d)$。**总 time：$O(n^{2} d)$**
   - **Memory：** Attention matrix $n \times n$ 显式存储是 $O(n^{2})$；加上 Q/K/V 的 $O(nd)$，主导项是 $O(n^{2})$
   - **Per-layer 参数量：** $W^{Q}, W^{K}, W^{V}, W^{O}$ 各 $d^{2}$，共 $4d^{2}$，与 $n$ 无关

2. **FFN 复杂度**：FFN 有两个矩阵 $W_{1} \in \mathbb{R}^{d \times 4d}$、$W_{2} \in \mathbb{R}^{4d \times d}$：
   - **Time：** $O(n \cdot d \cdot 4d) = O(8nd^{2})$（两个矩阵乘加起来）
   - **Memory：** Hidden activation $O(n \cdot 4d) = O(nd)$
   - **Per-layer 参数量：** $8d^{2}$，FFN 占 transformer 总参数的 ~2/3

3. **对比看瓶颈**：
   - **短序列（$n < d$）：** FFN 主导，复杂度 $O(nd^{2})$
   - **长序列（$n \gg d$）：** Attention 主导，复杂度 $O(n^{2} d)$
   - **Cross-over point：** 当 $n \approx d$ 时两者相当；$d = 768$ 时 $n > 768$ 后 attention 成为瓶颈
   - **Memory 上 attention 更早成为瓶颈：** Activation memory $O(n^{2})$ 在 $n = 2048$ 时就是 4M entry × per-layer × per-head × backward storage，单层就消耗 GB 级

4. **长序列具体痛点（用实际数字）**：
   - $n = 1024$：attention matrix 1M entry，fp16 占 2MB/head，48 head × 24 layer ≈ 2.3GB，可接受
   - $n = 8192$：attention matrix 64M entry，单卡 80GB 也吃力
   - $n = 32768$：attention matrix 1G entry，不可行
   - **训练 memory 比 inference 更紧张**：backward 需要保存所有 forward activation，再乘 2-3×

5. **为什么链上 fraud 场景受 $n^{2}$ 影响特别严重**：
   - 一个活跃 address 一年 tx 数轻松破万
   - Mixer / bridge / 高频套利 address 单月 $10^{4}$+ tx
   - 如果想看全 history，标准 transformer 几乎不可行
   - 即便用 sliding window，window 跨度不能太短（否则 lose context），实际生产 $n$ 通常取 256-2048

6. **缓解方案对照表**：
   | 方法 | Time | Memory | 牺牲 |
   |------|------|--------|------|
   | **Sparse attention（Longformer/BigBird）** | $O(nw)$ | $O(nw)$ | 限制 attention 模式，部分 long-range 信息丢失 |
   | **Linear attention（Performer/Linformer）** | $O(nd^{2})$ | $O(nd)$ | Kernel approximation 引入误差，质量略降 |
   | **FlashAttention** | $O(n^{2}d)$ | **$O(n)$**（不显式存 attention matrix） | 无（数学上 exact），仅工程实现优化 |
   | **State Space (Mamba)** | $O(nd)$ | $O(nd)$ | 完全替换 attention，inductive bias 不同 |
   | **Hierarchical（chunked）** | $O((n/c)^{2}d + n/c \cdot c^{2} d)$ | 同 time | 跨 chunk attention 受限 |

> **Follow-up 提示：** 面试官可能追问"FlashAttention 怎么做到既保持 $O(n^{2})$ time 又把 memory 降到 $O(n)$？"——答：核心是 tiling + recomputation。Forward 时把 Q/K/V 分块加载到 GPU SRAM，在 SRAM 内算 partial attention block，online softmax 维护 running max 和 normalizer，**永远不 materialize 完整 $n \times n$ matrix**。Backward 时不存中间 attention，直接 recompute（reconstruct block 比存到 HBM 再读快得多）。这是典型的"IO-aware algorithm"——FLOPs 数学上不变甚至略多（recompute），但因为 HBM-SRAM bandwidth 是瓶颈，实测 2-4× 加速 + 内存大幅下降。

---

### Q5: 什么是 Masked Language Modeling（MLM）？如何把它迁移到交易序列的预训练？

**回答：**

MLM 是 BERT 引爆 NLP 的核心训练目标，迁移到链上 tx sequence 是 fraud detection 一个非常 promising 的方向。下面分 NLP 原始定义 → 设计细节 → 链上迁移 → 工程注意点四层讲。

1. **MLM 在 NLP 的定义**：给定一句话，随机 mask 一部分 token，让模型用剩下的 bi-directional context 预测被 mask 的 token。Loss 只在 mask 位置算 cross-entropy。具体细节：
   - **Mask 比例 15%：** 太低 → 信号稀疏，每个 batch 学到的内容少；太高 → context 残缺，模型学不到有效 representation
   - **80/10/10 策略：** 被选中的 15% token 里，80% 替换成 `[MASK]`、10% 替换成随机 token、10% 保持不变。原因是 fine-tune / inference 时输入不会有 `[MASK]`，如果训练时只见 `[MASK]`，模型对真实 token 的预测能力会退化（train-test mismatch）

2. **MLM 为什么有效**：
   - 强制模型从 **bi-directional context** 推断被 mask 信息，等价于学到 deep contextual representation
   - 对比 left-to-right LM（GPT-style），MLM 利用了未来 context，对 understanding task（分类、NER）的表示更强
   - 计算高效：一次 forward 同时学多个位置（虽然只在 mask 位置算 loss）

3. **迁移到 transaction sequence 的设计**：把"sentence"换成"address 的 tx history"，把"word token"换成"tx token"。
   - **Token 表示：** 每笔 tx 是一组 field 的 embedding 融合：discrete fields（method_id、counterparty_cluster、token_id、direction）+ continuous fields（log(value)、log(gas_price)、log(time_delta)）
   - **Masking 策略选择：**
     a. **Whole-tx masking：** Mask 整笔 tx 的所有 field，预测所有 field 的取值；学到"给定上下文 tx，下一笔最可能是什么 tx" 的整体 prior
     b. **Field-level masking：** Mask 单个 field（如保持 counterparty 可见，mask method_id），预测被 mask field；学到 field 之间的 conditional dependency，对 anomaly detection 更直接（异常 tx 的 field combination 在 conditional distribution 下概率极低）
     c. **Joint：** 两种 mask 混合，更 robust
   - **Mask 比例：** 链上 tx 信息密度比 NLP 词更高（每笔 tx 多个 field），15% 是合理起点；如果用 field-level mask，可以提到 20-30%
   - **80/10/10：** 同样保留（随机替换可以用 vocab 里同 field 的另一个值；保持不变就是原 tx）

4. **Loss 函数设计**：
   - Discrete field：standard cross-entropy on field-specific vocab
   - Continuous field：MSE 或 huber loss；或者把 continuous field 离散化成 bucket 后也用 cross-entropy（实践上 bucket 化往往更稳定）
   - 多 field 时 loss 加权求和，可以按 field difficulty / business importance 调权

5. **预训练后的下游使用**：
   - **Fraud classification fine-tune：** Sequence 末尾加 `[CLS]` token，pretrain 后用少量 labeled fraud 数据 fine-tune 二分类头
   - **Address embedding 提取：** 取 `[CLS]` 或 mean-pool 全 sequence 作为 address representation，feed 给下游 GNN 或 tabular model 当 dense feature
   - **Anomaly detection（无监督）：** 用 reconstruction loss 或 next-token perplexity 作为 anomaly score，不依赖 label——对 unknown new fraud type 特别有用

6. **工程注意点**：
   - **Vocab 长尾：** Counterparty cluster 和 contract address 极其长尾，必须 top-K + OOV bucket；否则 embedding table 爆炸且尾部学不动
   - **Length 不均匀：** 活跃 address 几万笔 tx、僵尸 address 几笔，要 truncation + bucketed batching；建议用 sliding window 切多个长度 $\leq L_{\max}$ 的 sub-sequence，每个 sub-sequence 是一个训练样本
   - **Look-ahead leakage：** 训练 sequence 末尾不能包含 label 之后的 tx
   - **Adversarial drift：** 攻击者行为模式持续演化，pretrain corpus 要定期 refresh（如月度更新），生产环境监控 sequence distribution shift 触发再训练

> **Follow-up 提示：** 面试官可能问"MLM 和 next-event prediction（GPT-style）在 tx sequence 上哪个更好？"——答：取决于下游 task。① 离线 fraud classification：MLM 略优，因为 bi-dir context 让表示更丰富；② Real-time streaming detection：必须用 causal LM（GPT-style），因为生产时只有 left context；③ Pretrain 一份模型同时服务两种 task：可以用 UL2 / GLM 这类混合目标（同时含 causal 和 prefix masking），是 2024-2025 年趋势。还可能问"15% mask 比例在 tx 上是否需要调整？"——答：tx 信息密度高、field 间冗余少，可以略提到 20%；如果做 field-level mask 而非 whole-tx，可以到 25-30%。建议做 mask ratio 的 sweep。

---

### Q6: Encoder-only 和 Decoder-only 架构在欺诈分类任务上哪个更合适？

**回答：**

这个问题没有绝对答案，**取决于 deployment scenario（offline batch vs real-time streaming）和 label availability**。下面分维度对比。

1. **架构核心区别**：
   - **Encoder-only（BERT 系）：** Bi-directional self-attention，每个位置可看左右两侧；典型训练目标 MLM；天然适配分类、表示学习
   - **Decoder-only（GPT 系）：** Causal self-attention，每个位置只可看左侧；典型训练目标 next-token prediction；天然适配生成、自回归预测

2. **在 fraud classification 上的具体对比**：

| 维度 | Encoder-only (BERT-style) | Decoder-only (GPT-style) |
|------|--------------------------|-------------------------|
| **看到 right context** | 是 | 否 |
| **离线 batch 评分（已知完整 history）** | 表示更强，是首选 | 也可以，但 bi-dir 优势不可用 |
| **Real-time streaming（边收 tx 边评分）** | **不适合**——right context 是未来 tx，real-time 时还不存在 | **必选**——天然 causal |
| **训练 token 利用率** | 低（每样本只在 15% mask 位置算 loss） | 高（每个 token 都算 loss） |
| **训练所需数据量** | 中等 | 更大（但更高效） |
| **Pretrain → fine-tune 范式成熟度** | 极成熟（BERT-style） | 极成熟（GPT-style） |
| **下游 fraud classification 微调** | 加 `[CLS]` head 二分类，标准做法 | 末位 token 接 classification head 或 prompt-based |

3. **结合 OKX anti-fraud 实际场景的推荐**：
   - **场景 A：离线 batch fraud screening（每天对全量 user 跑一遍）：** Encoder-only 更合适。给定 address 完整 tx history（或最近 K 笔），bi-dir context 让 `[CLS]` representation 同时利用过去和"该窗口内的未来"信息（注意是 sequence 内的未来，不是真实未来），表示能力更强；离线允许 latency，可以用大模型
   - **场景 B：Real-time streaming（用户提现时 5ms 内决策）：** Decoder-only。每来一笔新 tx 只用左侧 history 算 risk score；causal mask 天然契合"只看过去"的生产约束；inference 时 KV cache 可以增量更新（每次新 tx 只算 1 步），延迟极低
   - **场景 C：Hybrid system：** 两种都训。离线 encoder model 输出 address-level long-term risk embedding，缓存到 feature store；real-time decoder model 消费这个 embedding + recent tx 做 online scoring。这是大多数成熟 fraud platform 的实际架构

4. **为什么 LLM 时代 decoder-only 占主流**：在通用 LLM 领域，decoder-only 几乎吃掉了 encoder-only 的所有 use case，是因为 ① decoder 可以做生成而 encoder 不能，能力上是 superset；② Token 利用率高，scaling 起来训练效率优；③ Prompt-based 方法让 decoder 也能做分类（in-context learning）。**但在 fraud classification 这种 well-defined 分类任务上，bi-dir encoder 仍然有质量优势，不能盲目跟风用 decoder-only**。

5. **混合方案：UL2 / prefix LM**：T5、UL2、GLM 等用 prefix LM 目标——prefix 部分 bi-dir，suffix 部分 causal——结合了两种架构优势。理论上更适合既要 offline 表示又要 online 生成的统一模型，但工程复杂度高，未在 fraud 领域普及。

> **Follow-up 提示：** 面试官可能问"如果只有 1 万条 labeled fraud，怎么选？"——答：标注稀缺时倾向用 encoder-only + 大量 MLM pretrain，因为 ① pretrain 利用 unlabeled 数据更高效；② fine-tune 数据少时 bi-dir 比 causal 信息更充足，泛化更好。还可能问"能不能用 encoder-only 做 real-time？"——答：技术上可以但 expensive。每来一笔 tx 要把"history + 新 tx" 整个序列重新跑一遍 forward，复杂度 $O(n^{2})$，无法做增量更新；而 decoder 用 KV cache 每次只算新 token 是 $O(n)$。对 sub-100ms latency 要求，几乎只能选 decoder。

---

### Q7: 为什么 RoPE（旋转位置编码）优于原始正弦位置编码？

**回答：**

RoPE 是当前 LLM 几乎统一采用的方案，相比原始 sinusoidal PE 有几个不可替代的优势。下面从机制差异、相对位置、外推能力、欺诈场景四个角度展开。

1. **机制差异：位置信息注入方式**：
   - **Sinusoidal PE：** 把位置 vector $\text{PE}_{\text{pos}}$ 直接 element-wise add 到 token embedding 上：$x' = x + \text{PE}$，位置信号污染了 token embedding 本身，attention 计算时无法分离 token 内容和位置
   - **RoPE：** 不动 token embedding，只在 attention 计算时对 Q 和 K 做位置相关的 **旋转**：$q'_{m} = R_{m} q$、$k'_{n} = R_{n} k$，其中 $R_{m}$ 是和位置 $m$ 相关的旋转矩阵。Token 内容保持纯净，位置信号只在 attention score 阶段起作用

2. **相对位置自然嵌入 attention score**：这是 RoPE 最 elegant 的性质。由旋转矩阵的群性质：

$$\langle R_{m} q, R_{n} k \rangle = q^{T} R_{m}^{T} R_{n} k = q^{T} R_{n-m} k$$

旋转后的 dot product **只依赖于相对位置 $n - m$**，不依赖绝对位置 $m, n$ 本身。这意味着：
   - 位置 $(10, 15)$ 的 attention score 和位置 $(100, 105)$ 的 attention score 在数学上同等对待（如果 q, k 相同）
   - 模型学到的是 "距离 5 的两个 token 之间的相关性"，而不是 "位置 10 和位置 15 之间的相关性"
   - 这种相对位置归纳偏置在大多数 sequence task 上是合理的（包括 NLP 和链上 tx sequence）

   相比之下，sinusoidal PE 通过 $\sin(a+b)$ 三角恒等式 **隐式** 允许模型学到 relative position，但不是 hardcoded 的，需要数据驱动学到；且 PE 加在 embedding 上后，relative position 信息和 token semantic 纠缠在一起，attention 分离这两者更困难。

3. **外推能力（length extrapolation）**：
   - **Sinusoidal：** 理论上可外推（公式 closed-form），但实际效果差——模型在训练时只见过 $L_{\max}$ 内的 position，超出后 attention 行为退化
   - **Learned PE：** 完全无法外推（超出 index 范围）
   - **RoPE：** 默认外推也有限，但配合 NTK-aware scaling、YaRN、LongRoPE 等技术，4K 训练的模型可以外推到 32K-128K 且效果几乎不损。这是 long-context LLM（Claude 200K、Gemini 1M）的基础

4. **欺诈检测场景的关键优势**：
   - **相对时间差比绝对位置更有语义：** 链上 tx 序列里，"两笔 tx 距离 5 笔（或 5 秒）"的含义远比"这笔 tx 是 sequence 第几个位置"重要。Bot 通常有 regular interval、layering 有 quick pass-through、Ponzi 有特定 payout cadence——这些都是相对时间/相对位置模式。RoPE 的相对位置 inductive bias 天然契合
   - **活跃 address 的 sequence 长度极不均匀且经常超出训练长度：** 训练时用最近 512 笔，但 inference 时活跃 address 可能 attend 到几千笔（如果配合长 context 推理）。RoPE 的外推能力让生产部署更灵活
   - **多 field token 的 embedding 干扰最小：** Sinusoidal 把位置 add 到 embedding 上会污染原本就由 discrete + continuous field 复合 fusion 出的 token vector；RoPE 只在 attention 计算时旋转 Q/K，不动 token 本体，对 multi-field embedding 干扰最小

5. **工程实现 trade-off**：
   - **计算开销：** RoPE 每次 attention 都要做旋转，比 sinusoidal "一次性加 PE" 略多计算；但旋转是 element-wise + 简单 sin/cos，可融合进 attention kernel，实际开销可忽略
   - **参数量：** 两者都是 0 参数（除了 RoPE 的 $\theta_{i}$ 是 hardcoded 公式）
   - **混合使用：** 一些模型同时用 RoPE（位置）+ ALiBi（远距离 penalty），结合两者优势

6. **现状（2026 年）**：LLaMA、Qwen、DeepSeek、Mistral、Gemma、Claude 内部架构（推测）几乎全部用 RoPE 或 RoPE-derived 方案。Sinusoidal 只在 academic baseline 和早期模型里出现，learned PE 仅在 BERT 等旧模型里保留。Fraud detection model 从零搭新架构时，**RoPE 是几乎无脑的默认选择**。

> **Follow-up 提示：** 面试官可能问"RoPE 的 $\theta_{i} = 10000^{-2i/d}$ 这个 base 10000 怎么来的？能调吗？"——答：base 决定了不同维度上 rotation frequency 的分布——低维高频（短距离敏感）、高维低频（长距离敏感）。10000 是从 sinusoidal PE 沿用过来的经验值，对 2K-8K context 表现好；扩展到 long context 时（如 32K+）需要把 base 增大（NTK scaling）或对 $\theta$ 做 frequency-specific scaling（YaRN），否则高频维度在长距离上 alias，attention 行为退化。还可能问"RoPE 在 multi-head 间怎么共享？"——答：通常所有 head 共享相同 $\theta$ schedule，但因为每个 head 在不同 subspace，实际 attention pattern 仍然 diverse。也有研究尝试 head-specific $\theta$，效果提升有限。

---

### Q8: FFN 层的作用是什么？为什么中间维度是 $4 \times d_{\text{model}}$？

**回答：**

FFN 是 transformer block 的"另一半"（attention 是另一半），在很多分析里被忽视，但实际上 FFN 承载了模型的主要 knowledge capacity。

1. **FFN 的角色与必要性**：
   - **数学形式：** $\text{FFN}(x) = \max(0, xW_{1} + b_{1}) W_{2} + b_{2}$（标准 ReLU 版本），现代变体常用 GELU / SwiGLU
   - **Position-wise：** 对每个 token 独立施加相同的 $W_{1}, W_{2}$，不在 sequence 维度交互——和 attention 形成清晰分工：**attention 负责跨 token 信息流，FFN 负责单 token 内特征加工**
   - **引入非线性：** Attention 本质是 weighted sum（softmax 是逐元素非线性但不引入跨维度交互），对单个 token 来说 attention output 是 $V$ 行的线性组合。如果没有 FFN 的非线性，整个 transformer 在每个 token 上就退化为线性变换，**无法学习复杂函数**

2. **FFN 是模型的 knowledge memory**：
   - 研究（Geva et al. 2021, "Transformer Feed-Forward Layers Are Key-Value Memories"）发现 FFN 可以解释为一个 key-value memory：$W_{1}$ 的每一行是一个 "key"，$W_{2}$ 的每一列是一个 "value"，中间的 activation 是 key-matching score
   - 具体的事实性知识（"巴黎是法国首都"）很大程度存储在 FFN 的 weight 里；attention 提供 routing，FFN 提供 lookup
   - **参数量占比：** Transformer block 总参数 $\sim 12 d^{2}$，其中 attention 是 $4 d^{2}$（$Q, K, V, O$ 四个矩阵），FFN 是 $8 d^{2}$（$W_{1}, W_{2}$ 两个矩阵每个 $4 d^{2}$），**FFN 占 2/3**

3. **为什么中间维度 $4 \times d_{\text{model}}$**：
   - **经验最优：** 原 transformer 选 4×，后续大量实验（包括 Chinchilla scaling law 研究）发现 ratio 在 2.5-4 之间都接近最优，4× 是 sweet spot
   - **容量与计算的 trade-off：**
     | Ratio | 影响 |
     |-------|------|
     | < 2 | 容量不够，模型 underfit |
     | 2-4 | 最优区间，容量与 compute 平衡 |
     | 4-8 | 边际收益递减，每多 1× ratio 带来的 loss 下降越来越小 |
     | > 8 | 显著浪费 compute，且容易过拟合（在小数据上） |
   - **理论视角：** FFN 中间维度可以视为"模型的 working memory size"。过小限制能 represent 的复杂函数；过大让训练样本不足以填满这些参数，泛化变差
   - **历史细节：** 原始 transformer ($d_{\text{model}} = 512$, $d_{\text{ff}} = 2048$) 用 4×；BERT-base ($d = 768$, $d_{\text{ff}} = 3072$) 4×；GPT-3 ($d = 12288$, $d_{\text{ff}} = 49152$) 4×。**几乎是行业标准**

4. **GLU 家族的改造**：LLaMA / PaLM 使用 SwiGLU 替代 ReLU FFN：

$$\text{SwiGLU}(x) = (\text{Swish}(xW_{1}) \odot xW_{3}) W_{2}$$

引入 gating（$xW_{3}$ 作为 gate）让 FFN 表达能力更强。但因为多了一个 $W_{3}$ 矩阵，**为了保持总参数预算不变，$d_{\text{ff}}$ 通常从 $4d$ 缩到 $\frac{8}{3}d \approx 2.67d$**，所以 LLaMA 里你会看到 $d_{\text{ff}} = 2.67 d$ 而不是 $4d$——这不是"ratio 改了"，而是"加 GLU 后等参数预算下的 ratio"。SwiGLU 在等参数下持续优于 ReLU，是当前 LLM 默认。

5. **在欺诈检测 transformer 上的考虑**：
   - **数据量决定 ratio：** 小规模 fraud-specific transformer（参数量 $10^{7}$-$10^{8}$）训练数据有限，4× 是安全选择；如果数据极少可以降到 2-3× 防过拟合
   - **Knowledge nature：** 链上 fraud detection 的 "knowledge" 更多是 behavioral pattern 而非 factual knowledge，FFN 容量需求可能比 NLP 模型略低，但 4× 仍是 robust default
   - **Multi-field token 与 FFN：** 不同 field 的信息在 attention 后混合到 single hidden vector，FFN 负责进一步加工这个混合 representation。可以考虑用 mixture-of-experts（MoE）FFN——不同 expert 专精不同 fraud pattern（mixer-related expert、wash-trade expert 等），训练时按 token type route，是 frontier 方向

6. **现代变体一览**：
   - **SwiGLU**：LLaMA 默认，效果最好
   - **GeGLU**：用 GELU 代替 Swish 的 GLU 变体，效果类似
   - **MoE FFN**：把 FFN 替换成 mixture of experts（如 Mixtral 用 8 expert，每 token 激活 top-2），等效参数量大幅提升但每 token compute 不变
   - **Parallel FFN + Attention**（PaLM）：把 FFN 和 attention 并联而非串联：$y = x + \text{Attn}(x) + \text{FFN}(x)$，省 latency 且效果几乎不损

> **Follow-up 提示：** 面试官可能问"为什么不直接把 FFN 维度做到 8× 或 16× 让模型更强？"——答：① 等参数预算下增大 FFN 必然减少 layer 数或减小 attention 维度，trade-off 不划算（layer 深度对 representation 重要）；② 8×+ 后边际 loss 改善小到不值；③ Memory 和 compute 翻倍。Scaling law 实验（Chinchilla / GPT-3）都验证了 4× 是接近 optimal frontier 的选择。还可能问"FFN 能不能去掉用 attention 替代？"——答：理论上可以——纯 attention transformer（only-attention）确实可以训，但等参数预算下质量显著低于 attention + FFN，因为 ① attention 缺非线性表达力；② FFN 的 key-value memory 角色没有等价替代；③ 工程上 FFN 的 layout 更利于 GPU 优化。FFN 是 transformer 的 essential component，不是可选组件。

---
