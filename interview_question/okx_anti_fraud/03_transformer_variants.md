# Transformer 变种全景与欺诈检测选型（Transformer Variants Survey for Fraud Detection）

> **文档定位：** OKX Anti-Fraud AI Engineer 面试准备系列 File 3 of 4。
>
> **目标读者：** 已经熟悉 vanilla Transformer / BERT / GPT 工作流程，需要在面试里讲清楚 "BERT 系列演进、Efficient Transformer、表格 / 时间序列 / 异常检测 / Graph Transformer 各家如何改造 attention、以及在欺诈检测里如何选型" 的工程师。本文不重复 transformer 基础数学，重点是 **每个变种解决什么本质问题、设计 trade-off、在 fraud 场景的适用性**。
>
> **配套阅读：** `01_blockchain_data_primer.md`、`02_transformer_architecture.md`、`04_*.md`（fraud system end-to-end）。

---

## 1. BERT 家族（BERT Family）

---

### 1.1 BERT (2018, Google)

**论文：** Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (NAACL 2019)。

BERT 是第一个把 **pre-train + fine-tune 范式** 在 NLP 上系统跑通的模型，奠定了之后 5 年的 NLP 主流路线。

**核心设计：**
- **Encoder-only：** 只保留 Transformer encoder 部分，bi-directional self-attention，每个位置可以看到左右两侧
- **双任务预训练：**
  - **MLM (Masked Language Model)：** 随机 mask 15% token，其中 80% 替换为 `[MASK]`、10% 替换为随机 token、10% 保持不变。模型预测被 mask 的原 token
  - **NSP (Next Sentence Prediction)：** 给两句话 A 和 B，预测 B 是不是 A 在原文中的下一句。50% 是真 next sentence，50% 是随机抽的负样本
- **输入格式：** `[CLS] sentence_A [SEP] sentence_B [SEP]`，`[CLS]` 的最终 hidden state 用于句对分类，`[SEP]` 分隔两个 segment

**参数规格：**

| 模型 | Layers | Hidden size | Attention heads | 参数量 |
|------|--------|-------------|-----------------|--------|
| **BERT-base** | 12 | 768 | 12 | 110M |
| **BERT-large** | 24 | 1024 | 16 | 340M |

**事后被发现的缺陷：**
1. **NSP 任务本身没用：** RoBERTa 实验证明去掉 NSP 反而效果更好。NSP 太简单（随机负样本和正样本主题完全不同，trivially 区分），模型学的是 "topic prediction" 而非 "sentence coherence"
2. **静态 masking：** 数据预处理时一次性决定哪些 token 被 mask，整个训练过程中相同 epoch 见到相同 mask pattern，限制了 generalization
3. **`[MASK]` token 在下游 fine-tune 时不存在：** 造成 pretrain / fine-tune mismatch（mitigated by 10% 不替换的设计，但仍是问题）

**对 fraud detection 的启发：** Encoder-only + MLM 范式对 "把 tx sequence 编码成 address-level representation" 这类离线表示学习极其有效。但 BERT 原始版本不能直接用，需要换 tokenizer、调整 NSP（或干脆去掉），并把 vocab 从 wordpiece 换成 tx-field-based encoding。

### 1.2 RoBERTa (2019, Facebook)

**论文：** Liu et al., *RoBERTa: A Robustly Optimized BERT Pretraining Approach*。

RoBERTa **不改架构，只优化训练**——结果在相同参数量下显著超越 BERT，证明 BERT 远未训练充分。

**关键改动：**

| 改动 | 细节 | 效果 |
|------|------|------|
| **去掉 NSP** | 只保留 MLM，每个 sample 直接拼接来自同一文档的连续句子直到长度 512 | NSP 无用且占训练资源 |
| **动态 masking** | 每个 epoch 重新决定哪些 token 被 mask（vs. BERT 一次性预处理） | 模型见到更多样的 mask pattern，泛化更好 |
| **更大 batch size** | 从 BERT 的 256 增大到 8K | 训练更稳定，learning rate 可以调大 |
| **更长训练** | 训练 token 数从 BERT 的 ~130B 增大到 ~2.2T（RoBERTa-large） | 更长训练直接换更好效果 |
| **更大 vocab** | 用 50K BPE（Byte-level BPE）替代 BERT 的 30K WordPiece | 减少 `<unk>` 出现，handle 多语言 / 罕见词更好 |
| **去掉 sentence pair 限制** | 直接拼接长文档而非两个独立句子 | 让模型见到更长 context |

**对 fraud detection 的启发：** "训练 trick 比架构改动更重要" 是 RoBERTa 的核心 lesson。在搭 fraud transformer 时，BERT-base 架构 + RoBERTa-style 训练（去 NSP、动态 mask、足够大 batch 和长训练）通常比换更复杂的架构性价比更高。

> **面试 sound bite：** "RoBERTa 告诉我们 BERT 不是 architecture limited 而是 training limited。同样的预算下，先把训练 recipe 调对，再考虑换架构。"

### 1.3 DeBERTa (2020, Microsoft)

**论文：** He et al., *DeBERTa: Decoding-enhanced BERT with Disentangled Attention* (ICLR 2021)。DeBERTa 是 2020-2021 年 NLU 榜单的 SOTA，**首次在 SuperGLUE 上超越人类 baseline**。

**核心创新 1：Disentangled Attention（解耦注意力）**

vanilla Transformer 把位置信息直接加到 token embedding 上：$x'_{i} = E_{w_{i}} + P_{i}$，attention score 是混合了 content 和 position 的：

$$
A_{ij} = (E_{w_{i}} + P_{i})(E_{w_{j}} + P_{j})^{T}
$$

展开后有 4 项：content-content、content-position、position-content、position-position。

DeBERTa 把 content 和 position 拆开维护两套 representation，attention score 显式分解：

$$
A_{ij} = H_{i} \cdot H_{j}^{T} + H_{i} \cdot P_{i|j}^{T} + P_{j|i} \cdot H_{j}^{T}
$$

其中：
- $H_{i} \cdot H_{j}^{T}$：content-to-content（语义相关性）
- $H_{i} \cdot P_{i|j}^{T}$：content-to-position（"我这个 content 关心多远的位置"）
- $P_{j|i} \cdot H_{j}^{T}$：position-to-content（"这个相对位置上的内容对我是否重要"）
- 故意省略了 position-to-position 项，因为相对位置对相对位置的 attention 没有语义意义

$P_{i|j}$ 是相对位置 encoding（不再是绝对位置），意为"$i$ 相对于 $j$ 的位置"。

**核心创新 2：Enhanced Mask Decoder (EMD)**

DeBERTa 注意到 MLM 解码时除了 relative position，**absolute position** 在某些情况下也是必要的（比如句首的 token 有特殊语义）。但如果把 absolute position 早早加入 input，会污染所有层的 attention。

EMD 的做法：在最后几层 transformer 之前才注入 absolute position embedding，让大部分层只用相对位置，最后才考虑绝对位置。这种"延迟注入"让 representation 学习和位置 grounding 各司其职。

**结果：**
- DeBERTa-xxlarge（1.5B）在 SuperGLUE 上得分 89.9，首次超过人类 baseline 89.8
- DeBERTa-v3（用 ELECTRA-style replaced token detection 替代 MLM）进一步提升

**对 fraud detection 的启发：** 链上 tx sequence 的 "position" 含义比 NLP 更复杂——既有 sequence index，又有 timestamp delta、block height delta。DeBERTa 把 content 和 position 显式分离的思想对处理 tx sequence 特别合适，可以把 timestamp delta 当作单独的 position channel 而不污染 tx feature embedding。

### 1.4 ALBERT (2019, Google)

**论文：** Lan et al., *ALBERT: A Lite BERT for Self-supervised Learning of Language Representations*。

ALBERT 的目标是 **减少 BERT 参数量** 以应对显存压力，用了两个 orthogonal 技术。

**技术 1：Factorized Embedding Parameterization**

BERT 的 token embedding 矩阵是 $V \times H$（$V$=30K vocab，$H$=768 hidden），占 23M 参数（接近 BERT-base 总参数 110M 的 20%）。

ALBERT 把 embedding 矩阵分解为两步：
$$
V \times E \to E \times H
$$

其中 $E \ll H$（比如 $E = 128$）。第一个矩阵 $V \times E$ 把 token id 映射到低维 embedding，第二个 $E \times H$ 把低维 embedding 投影到 hidden size。

**节省的参数量：** 从 $V \cdot H = 30000 \times 768 = 23M$ 降到 $V \cdot E + E \cdot H = 30000 \times 128 + 128 \times 768 = 3.94M$，节省 80%。

**直觉：** token embedding 是 sparse 的、context-independent 的，不需要和 hidden state 同样大的维度；hidden state 是 contextualized 的、要承载语义复杂性，应该保持高维。

**技术 2：Cross-Layer Parameter Sharing**

BERT 每一层的 $W^{Q}, W^{K}, W^{V}, W^{O}, W^{1}, W^{2}$ 都是独立参数。ALBERT 把所有 12 层（或 24 层）的参数 **完全共享**——只有一组 transformer block 参数，循环用 12 次。

**节省的参数量：** 从 12 层 × $12 d^{2}$ 缩到 1 层 × $12 d^{2}$，减少 11/12。

**技术 3：SOP 替换 NSP**

ALBERT 也质疑 NSP 没用。它提出 **Sentence Order Prediction (SOP)**：给两个连续句子 A、B，预测它们的顺序是 A→B 还是 B→A。这迫使模型学 sentence coherence 而非 topic prediction。

**最终参数量：**

| 模型 | BERT 对应版本 | 参数量 |
|------|---------------|--------|
| ALBERT-base | BERT-base (110M) | 12M |
| ALBERT-large | BERT-large (340M) | 18M |
| ALBERT-xxlarge | -- | 235M（仍比 BERT-large 小） |

**关键 trade-off：**
- **参数量大幅减少**：内存占用降低，单机可以训更大模型
- **inference 时间不变甚至更慢**：因为 forward pass 仍然是 12 层 × $O(n^{2})$ 计算量。参数共享不减 FLOPs，只减 memory
- **下游任务效果有损失**：参数共享降低了 model capacity，ALBERT-base 比 BERT-base 略弱

**对 fraud detection 的启发：** ALBERT 的参数共享对 fraud 模型不太适用——fraud detection 通常 latency 敏感，参数共享省了内存但 inference 不变快。但 factorized embedding 思想可以借鉴：tx 里 categorical field（如 method id）的 vocab 可能很大（几百万 contract），用低维 embedding factorize 是直接收益。

---

## 2. GPT 家族（GPT Family）

---

### 2.1 GPT-1/2/3/4 演进

GPT 系列是 **decoder-only + causal LM** 的代表，每一代都是 "scale up" 故事。

| 模型 | 年份 | 参数量 | 核心特点 |
|------|------|--------|---------|
| **GPT-1** | 2018 | 117M | 首个 decoder-only pretrain + fine-tune 范式；12 层；用 BookCorpus 训练 |
| **GPT-2** | 2019 | 117M / 345M / 762M / 1.5B | 强调 zero-shot；WebText 数据；首次展示 in-context generation 能力；OpenAI 当时拒绝 release 1.5B 版本（"too dangerous"） |
| **GPT-3** | 2020 | 175B（96 层、$d=12288$、96 heads） | 首次 in-context learning：few-shot prompt 不微调即可解多种任务；scaling law 验证 |
| **GPT-4** | 2023 | 未公开（估计 1T+ MoE） | 多模态（image+text）；context 32K → 128K；RLHF + instruction following；tool use；闭源 |

**演进趋势：**
- **不改架构：** GPT-1/2/3 几乎是同一架构（decoder-only + causal mask + learned PE），区别是 scale
- **去 fine-tune 化：** GPT-3 之后主流不再 task-specific fine-tune，而是 prompt engineering + in-context learning
- **后训练越来越重：** GPT-4 大量精力在 RLHF、instruction following、tool use，pretrain 之后的工作占比剧增

### 2.2 GPT vs BERT 在 fraud detection 的选型

| 维度 | BERT (encoder-only) | GPT (decoder-only) |
|------|---------------------|---------------------|
| **架构** | Bi-directional attention | Causal (left-to-right only) |
| **训练目标** | MLM（mask 部分 token 预测） | Next-token prediction（每个位置都算 loss） |
| **训练 token 利用率** | 低（只对 15% mask 位置算 loss） | 高（每个 token 都算 loss） |
| **下游 fraud 分类** | **首选**：`[CLS]` head 接二分类，bi-dir context 表示更强 | 可以但次优：取末位 token 接 head |
| **生成 fraud 解释 / report** | 不擅长 | **首选**：自回归生成自然语言解释 |
| **离线 batch screening** | **首选**：每天对全量 address 评分，bi-dir 优势可用 | 次优 |
| **Real-time streaming（边收 tx 边评分）** | 不适合：right context 是未来 tx，online 时不存在 | **首选**：causal 天然 fit；KV cache 增量更新 latency 低 |
| **欺诈 case study 自动生成** | 不行 | 用大 LM（GPT-4 / Claude）做事件总结、风险解释、客服话术 |

**面试 sound bite：** "Fraud detection 的 representation 学习选 BERT 系，fraud 解释 / 报告 / 对话选 GPT 系。在生产系统里两类模型并存，BERT 做 risk score，GPT 做 explanation。"

---

## 3. Efficient Transformers（解决 O(n²) 问题）

---

vanilla self-attention 的 $O(n^{2})$ 复杂度在长 sequence 上不可承受。一个活跃链上 address 可能有 $10^{4}$ 笔 tx 历史，直接套 BERT 算不动。Efficient Transformer 的核心思路是 **稀疏化或近似 attention 矩阵**。

### 3.1 Longformer (2020, AI2)

**论文：** Beltagy et al., *Longformer: The Long-Document Transformer*。

**核心机制：稀疏 + 全局混合 attention**

- **Sliding Window Attention：** 每个 token 只 attend 到自己前后 $w/2$ 个 token，复杂度 $O(n \cdot w)$
- **Dilated Sliding Window：** 在高层用空洞滑动窗口，间隔变大但仍保持局部性，扩展感受野
- **Global Attention：** 少量特殊 token（`[CLS]` 或任务特定的 token）和所有其他 token 双向 attend，复杂度 $O(n)$（这些 global token 数量少）

**总复杂度：** $O(n \cdot w + n \cdot g) = O(n)$，其中 $g$ 是 global token 数量。

**架构示意：**
- 普通 token 只看局部窗口（典型 $w = 512$）
- `[CLS]` 等 global token 看全局，所有 token 也都看 `[CLS]`
- 信息流：local pattern 在 sliding window 内传播 + 跨距离 signal 经过 global token 中转

**Fraud application：长 tx history 的 address-level risk scoring**
- 一个活跃 address 有 $10^{4}$ 笔 tx 历史，vanilla BERT 直接 OOM
- 用 Longformer：每笔 tx token 只 attend 前后 256 笔（local pattern：bot 节奏、连续 layering）
- `[CLS]` 作为 global token：每笔 tx 都 attend 到 `[CLS]`，`[CLS]` 也 attend 到所有 tx，形成 address-level summary
- 关键 fraud signal（30 天前的 mixer dust transfer）可以通过 `[CLS]` 中转到当前 tx

### 3.2 BigBird (2020, Google)

**论文：** Zaheer et al., *Big Bird: Transformers for Longer Sequences* (NeurIPS 2020)。

**核心机制：三种 attention 混合**

- **Random Attention：** 每个 token 随机 attend 到 $r$ 个其他 token（用于保证图连通性，理论上让任意两个 token 经 $O(1)$ 跳达到）
- **Window Attention：** 类似 Longformer 的 sliding window
- **Global Attention：** 类似 Longformer 的 `[CLS]` 全局 token

**复杂度：** $O(n)$。

**理论保证（区别于 Longformer 的关键贡献）：** BigBird 证明这种稀疏 attention pattern **保留了 full attention 的 Turing-completeness 和 universal approximation 性质**。这是数学上的 nice property，但实际工程效果和 Longformer 接近。

**Fraud application：** 同 Longformer。BigBird 的 random attention 在 fraud sequence 上有个额外好处：random 连接让远距离 fraud pattern（如 6 个月前的污染 → 今天的提现）有机会被直接 attend，不必经过 global token 中转。

### 3.3 Performer (2020, Google)

**论文：** Choromanski et al., *Rethinking Attention with Performers* (ICLR 2021)。

**核心机制：用 Random Feature Map 近似 softmax attention**

Softmax attention 的复杂度瓶颈是 $\text{softmax}(QK^{T})$ 必须先算 $n \times n$ 矩阵。Performer 的 insight：

$$
\text{softmax}(QK^{T}/\sqrt{d}) \approx \phi(Q) \phi(K)^{T}
$$

其中 $\phi(\cdot)$ 是一个 $d \to r$ 的非线性映射（基于 random feature 理论，称作 FAVOR+ 方法），可以让 softmax 被分解为两个 feature map 的内积。

**算法变化：**

vanilla attention 顺序：
1. 算 $QK^{T} \in \mathbb{R}^{n \times n}$（$O(n^{2} d)$）
2. softmax 归一化
3. 乘 $V$ 得 $O \in \mathbb{R}^{n \times d}$（$O(n^{2} d)$）

Performer 利用结合律变成：
1. 先算 $\phi(K)^{T} V \in \mathbb{R}^{r \times d}$（$O(n r d)$）
2. 再算 $\phi(Q) \cdot (\phi(K)^{T} V) \in \mathbb{R}^{n \times d}$（$O(n r d)$）

**复杂度：** $O(n^{2} d) \to O(n r d)$，当 $r \ll n$ 时显著加速。典型 $r = 256$。

**Trade-off：** 是 **近似 attention**（unbiased estimator，方差随 $r$ 减小）。在长序列上效果接近 full attention，但有时收敛比 sparse attention 慢；不可解释（softmax weight 显式存在但 Performer 隐式拆开了）。

**Fraud application：** 真正需要全局 attention（每对 tx 都可能有 fraud signal 关联，不能只看局部）的极长序列。在 fraud 实践中相对 niche——大多数 fraud pattern 局部或通过 global token 中转就够，sparse attention 通常够用。Performer 更适合 attention 模式确实是 dense 的场景。

### 3.4 三者对比

| 模型 | 核心机制 | 复杂度 | 适用场景 |
|------|---------|--------|----------|
| **Longformer** | Sliding window + Global token | $O(n \cdot w)$ | 文档分类、长 tx history scoring |
| **BigBird** | Window + Random + Global | $O(n)$ | 同 Longformer，有理论保证 |
| **Performer** | Random feature map | $O(n r d)$ | 真正 dense attention 需求，序列极长 |

> **面试 sound bite：** "选 efficient transformer 的关键是先问'attention pattern 应该是 sparse 还是 dense'。fraud 序列里局部 pattern + 少量长距离 signal 占主导，sparse attention（Longformer / BigBird）够用；如果业务证明 attention 真的 dense，再考虑 Performer 这类 kernel approximation。"

---

## 4. 表格数据 Transformers（Tabular Transformers）

---

链上 fraud detection 的 feature 大量是 tabular（聚合统计特征：tx count、总金额、地址 age、interaction with mixer 等），用 Transformer 处理 tabular 数据是 2020 年后兴起的方向。

### 4.1 TabTransformer (2020, Amazon)

**论文：** Huang et al., *TabTransformer: Tabular Data Modeling Using Contextual Embeddings*。

**核心设计：只对 categorical features 做 transformer，numerical features 跳过**

输入流程：
1. **Categorical features：** 每个 categorical column 各有一个 embedding table，把每个 categorical value 转成 $d$-dim embedding
2. **Categorical embeddings 进 Transformer：** 把所有 categorical feature 的 embedding 当作 sequence input，过 $L$ 层 transformer encoder，每个 column 之间通过 attention 学交互
3. **Numerical features：** 直接做 batch / layer normalization，不进 transformer
4. **拼接：** Transformer 输出（contextualized categorical embedding）和 numerical features 直接 concatenate
5. **MLP head：** 拼接结果过 MLP 输出预测

**为什么 numerical 不进 transformer：** 原作者认为 numerical 是 continuous scalar，"tokenize" 后语义不清晰；categorical 是 discrete id，天然适合 embedding+attention。

**Fraud application：mixed transaction features**
- Categorical: tx method (transfer / swap / mint), counterparty type (EOA / contract), chain id
- Numerical: amount, gas used, address age in days, total tx count
- TabTransformer 让 categorical features 之间通过 attention 学交互（比如 "swap method × counterparty=mixer-contract" 是高风险组合），numerical 直接拼接

**局限：** Numerical 不进 transformer 意味着 model 学不到 categorical × numerical 的交互（除了最后 MLP 浅层学的简单交互）。这在 fraud 上是问题：金额 amount × counterparty type 是关键 cross feature。

### 4.2 FT-Transformer (2021, Yandex)

**论文：** Gorishniy et al., *Revisiting Deep Learning Models for Tabular Data* (NeurIPS 2021)。

FT 全称 **Feature Tokenizer Transformer**。修正了 TabTransformer 不把 numerical 进 transformer 的问题。

**核心创新：所有 feature（包括 numerical）都 tokenize**

对每个 feature $j$：
- **Numerical feature** $x_{j}$：投影成 $d$-dim token，$\text{token}_{j} = x_{j} \cdot w_{j} + b_{j}$（每个 numerical column 独立学一个 $w_{j} \in \mathbb{R}^{d}$ 和 $b_{j} \in \mathbb{R}^{d}$）
- **Categorical feature** $x_{j}$（取值 $c$）：用 embedding table，$\text{token}_{j} = E_{j}[c]$，等价于 lookup

所有 token 拼成 sequence，前面加 `[CLS]` token，过 $L$ 层 transformer。`[CLS]` 的最终 hidden state 接 MLP head 做预测。

**关键 trick：**
- Numerical 的 tokenize 不是简单 broadcast，是学习 per-column 的 $w_{j}, b_{j}$，把不同 column 的 scale 和语义 disentangle
- `[CLS]` token 充当全局聚合器，attention 决定哪些 feature 重要

**实验结果：**
- 在多个 tabular benchmark 上，FT-Transformer 显著超过 TabTransformer
- 在复杂 feature 交互的任务上有时甚至超过 XGBoost
- 在简单任务（小数据、feature 数少）上仍然不如 XGBoost

### 4.3 FT-Transformer vs XGBoost 选型

| 维度 | FT-Transformer 更好 | XGBoost 更好 |
|------|---------------------|--------------|
| **数据规模** | 大（$10^{6}$+ rows） | 中小（$10^{3}$-$10^{5}$ rows） |
| **Feature 数** | 多（$10^{2}$+） | 少（$10$-$10^{2}$） |
| **Feature 交互复杂度** | 复杂（高阶交互、上下文相关） | 简单（树的浅层交互） |
| **数据中混合 modality** | 是（categorical + numerical + 后续接 text/graph embedding） | 否（纯 tabular） |
| **训练 budget** | 充足（GPU 训练） | 紧（CPU 几分钟搞定） |
| **可解释性需求** | 低（接受黑盒） | 高（gain importance / SHAP 成熟） |
| **Online learning / 增量更新** | 难（要全量重训） | 容易（GBDT 增量训练成熟） |
| **缺失值处理** | 需要 mask token + 特殊 embedding | 树原生支持 |

**Fraud detection 实践：**
- **第一版基线：** XGBoost / LightGBM，因为 feature engineering 加 GBDT 是最快出 baseline 的路径
- **第二版升级：** 如果有大量 unlabeled fraud-relevant data 可以 pretrain，且有 GPU 预算，可以试 FT-Transformer，特别是当 features 包含 categorical id（contract address、token id）和 numerical 的混合时
- **生产组合：** 很多团队最终是 FT-Transformer 输出 embedding + XGBoost 二级，融合两者优势

> **面试 sound bite：** "FT-Transformer 在 high-cardinality categorical + 大数据 + 复杂交互上能赢 XGBoost；但日常 fraud feature engineering 出来的 100 维 tabular 上，XGBoost 仍然是 default。先用 GBDT 出 baseline，证明有提升空间再上 transformer。"

---

## 5. 时间序列 Transformers（Time Series Transformers）

---

链上 fraud 经常表现为 **行为时间模式异常**——金额时序、tx 频率时序、price impact 时序。处理这类数据用通用 transformer 不一定 fit，专门的 time series transformer 是更好选择。

### 5.1 TFT - Temporal Fusion Transformer (2019, Google)

**论文：** Lim et al., *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting* (IJoF 2021)。

**核心设计：LSTM + multi-head attention + Gating + Variable Selection**

TFT 是一个 **multi-horizon forecasting** 模型（同时预测未来 $1, 2, \dots, H$ 步），主要组件：

1. **Variable Selection Network (VSN)：** 在每个时间步，学习一个 weight 决定哪些 feature 重要（feature-level attention）。VSN 输出 sparse weight，自动做 feature selection
2. **LSTM Encoder-Decoder：** 处理短期时序依赖。Encoder LSTM 处理历史，Decoder LSTM 自回归预测未来
3. **Multi-Head Attention 层：** 在 LSTM 输出之上加 attention，捕捉长距离依赖（LSTM 本身长依赖弱）
4. **Gated Residual Networks (GRN)：** 类似 transformer block 的 FFN，但加 gating 控制信息流，对每个 sub-component 都 wrap
5. **Quantile loss：** 不预测 point estimate，预测多个 quantile（如 0.1 / 0.5 / 0.9），输出 prediction interval

**预测形式：**

不是预测 $\hat{y}_{t+h}$ 单个点，而是预测 $\hat{y}_{t+h}^{(q)}$ 对应不同 quantile $q \in \{0.1, 0.5, 0.9\}$，loss 用 quantile loss：

$$
L_{q}(y, \hat{y}^{(q)}) = \max(q(y - \hat{y}^{(q)}), (q-1)(y - \hat{y}^{(q)}))
$$

**Fraud application：预测未来 tx 行为分布，outlier 即 anomaly**

- 输入：address 过去 30 天的每日 tx count、amount、active counterparty 数量
- TFT 预测：未来 7 天每日 tx count 的 10% / 50% / 90% quantile
- 实际 tx count 落在 [10% quantile, 90% quantile] 之外 → 行为异常 → 触发 fraud review

**优势：**
- Quantile prediction 比 point prediction 更适合 anomaly detection（明确给出"正常区间"）
- VSN 自动 feature selection，可解释性强
- 支持多种 input：static covariates（address 类型）、known future（已知日历事件）、observed past（实际 tx 行为）

### 5.2 PatchTST (2023, MIT + IBM)

**论文：** Nie et al., *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers* (ICLR 2023)。

**核心创新：Patch + Channel-independent**

传统时序 transformer 用每个 timestep 作为 token，长度 $T = 1000$ 序列就有 1000 个 token，$O(T^{2}) = 10^{6}$ 计算量。

PatchTST 借鉴 Vision Transformer 的 patch 思想：
- 把时序划分为 patch（如 $p = 16$，patch 数 $n = T / p$）
- 每个 patch 内的 $p$ 个 timestep 一起线性投影成一个 $d$-dim token
- Transformer 在 $n = T / p$ 长度上做 attention，复杂度 $O((T/p)^{2})$，降低 $p^{2}$ 倍

**Channel-independent：**
- 多变量时序（$M$ 个 channel），每个 channel 独立过 transformer，**不在 channel 间做 attention**
- 共享 transformer 参数（所有 channel 用同一组 weight）
- 多变量 forecasting 时分别预测每个 channel 的未来

**为什么 channel-independent 反直觉但有效：**
- 多变量 channel attention 在小数据上容易过拟合
- 共享参数 + per-channel forward 等于 batch 维度扩大，训练更稳
- 不同 channel 的 cross effect 由后续 MLP head 隐式建模

**Fraud application：**
- Address 的多个时序 channel（daily tx count、daily amount、daily unique counterparty）
- Patch 大小设为 24（一天 24 小时聚合为一个 patch）或 7（一周一个 patch）
- 每个 channel 独立 forecast，预测精度高，channel-specific anomaly 可定位

### 5.3 iTransformer (2024, THUML)

**论文：** Liu et al., *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting* (ICLR 2024)。

**核心创新：反转 attention 维度**

传统时序 transformer：每个 timestep 是 token，attention 跨 timestep，FFN 跨 feature。

iTransformer **完全反过来**：
- 把每个 variable（channel）的整个时序当作一个 token——即一个 $T$ 维向量 → 投影到 $d$ 维 → 得到一个 token
- 多变量时序就是 $M$ 个 token（$M$ 个 variable）
- Attention 跨 variable（不同 channel 之间）
- FFN 在每个 token 内（即在每个 variable 的时序 pattern 上）施加非线性

**对比维度：**

| 维度 | 传统时序 Transformer | iTransformer |
|------|---------------------|--------------|
| Token 是什么 | 一个 timestep（含 $M$ 个 channel value） | 一个 variable 的整段时序（$T$ 维） |
| Token 数量 | $T$（很长） | $M$（很短） |
| Attention 跨什么 | 跨 timestep（时间维交互） | 跨 variable（channel 维交互） |
| FFN 跨什么 | 跨 channel（每 timestep 内） | 跨 timestep（每 variable 内） |
| 适用 | 短序列 + 多 channel 强交互 | 长序列 + variable 间有 cross effect |

**何时 iTransformer 优于 PatchTST：**
- 多变量 channel 之间 **有强 cross effect**（PatchTST 的 channel-independent 假设被打破）
- 序列长度 $T$ 很大但 channel 数 $M$ 中等（如 $T=1000, M=20$）
- 例子：fraud 场景里 daily tx count、daily amount、daily unique address 之间显然有强相关（活跃度提升时三者同时涨）

**何时 PatchTST 优于 iTransformer：**
- Channel 间几乎独立（如某些 IoT sensor，每个 sensor 是独立物理量）
- 单 channel 内有强时间局部 pattern（cycle、trend），patch 化能捕捉

**Fraud application：**
- iTransformer：address 多 channel 时序（金额、计数、对手方多样性），channel 之间强交互，iTransformer 自然 fit
- PatchTST：单变量时序 forecast（如未来 24 小时每小时 tx count），patch 捕捉日内 pattern

---

## 6. 异常检测 Transformers（Anomaly Detection Transformers）

---

Fraud detection 经常是 **极不平衡 + 标签稀缺** 问题，无监督异常检测在工业实践中非常常用。两个代表性方案：

### 6.1 Anomaly Transformer (2022, THUML)

**论文：** Xu et al., *Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy* (ICLR 2022 Spotlight)。

**核心 insight：正常点 vs 异常点的 attention pattern 不同**

观察：在时序数据中，
- **正常点** 的 attention 通常 **集中在邻居**——和它时间上靠近的 timestep 高度相关（temporal consistency）
- **异常点** 的 attention 通常 **分散**——因为它和邻居"不一致"，模型找不到强 local association，attention spread 到远处

把这种"attention 是否集中在邻居"量化，就是 **Association Discrepancy**。

**两个分支：**

1. **Prior-association（先验关联）：** 给每个 timestep $i$，固定假设它的 attention 服从以 $i$ 为中心的 Gaussian：

$$
P_{i,j} = \frac{1}{\sqrt{2\pi}\sigma_{i}} \exp\left(-\frac{(j-i)^{2}}{2\sigma_{i}^{2}}\right)
$$

$\sigma_{i}$ 是可学习的（控制 prior Gaussian 的宽度）。Prior 表示"如果这是正常点，它的 attention 应该是这样的局部 Gaussian"。

2. **Series-association（序列关联）：** 学习出来的真实 attention 矩阵 $S_{i,j}$（标准 self-attention softmax 输出）。

**Anomaly score：两个分布的 KL divergence**

对每个 timestep $i$：
$$
\text{AssDis}(i) = \frac{1}{2}\left[\text{KL}(P_{i,:} \| S_{i,:}) + \text{KL}(S_{i,:} \| P_{i,:})\right]
$$

- 正常点：$S$ 和 $P$ 接近（attention 确实集中在邻居），KL 小
- 异常点：$S$ 偏离 Gaussian（attention 分散），KL 大

**Minimax 训练策略：**

为了让 prior 和 series 在 normal data 上趋同（让正常点 KL 小），但不让 series 学坏（即 series 不能直接退化为 Gaussian），采用 minimax：
- **Minimize 阶段：** 固定 $S$，更新 Gaussian 宽度 $\sigma$ 使 KL 最小（让 prior 拟合 series）
- **Maximize 阶段：** 固定 $\sigma$，更新模型参数让 series **远离 prior**（被迫学到 series 内更复杂的 long-range association）

结果：在正常点上 series 仍然集中在邻居（因为数据本身如此），在异常点上 series 被训练得"故意远离 prior"。Inference 时 KL 大 = 异常。

**为什么 unsupervised：** 只用 reconstruction loss + association discrepancy loss 训练，不需要 anomaly label。

**Fraud application：**
- Address 的 tx amount 时序，无监督训练 Anomaly Transformer
- Inference 时打 anomaly score，高分 tx 进入 manual review queue
- 不需要标注 fraud label，适合冷启动

### 6.2 TranAD (2022, Microsoft Research)

**论文：** Tuli et al., *TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data* (VLDB 2022)。

**核心设计：两个 transformer + adversarial training + self-conditioning**

**架构：**
- **Transformer T1（reconstructor）：** 输入序列 $W$，输出重构 $\hat{W}_{1}$
- **Transformer T2（refiner）：** 输入 $W$ 和 T1 的输出 $\hat{W}_{1}$（self-conditioning），输出二次重构 $\hat{W}_{2}$
- **Anomaly score：** $\|W - \hat{W}_{2}\|^{2}$

**Self-conditioning：** T2 的输入包含 T1 的重构结果，等于 "在 T1 的猜测基础上 refine"。这种 self-conditioning 让模型在 boundary anomaly（即不太明显的异常）上更敏感。

**Adversarial training：**
- T2 使用对抗目标训练——对 T1 重建误差高的窗口放大关注（adversarial focus），而不是使用独立的 discriminator 网络。这个自聚焦机制（self-focusing mechanism）使模型对边界异常更敏感。

**Anomaly score = reconstruction error：** 经过对抗训练后，normal data 重构误差极低；anomalous data 因为不在分布内，T1 重建误差大，T2 的对抗聚焦进一步放大这一差距，最终重构误差显著升高。

**对比 Anomaly Transformer：**

| 维度 | Anomaly Transformer | TranAD |
|------|---------------------|--------|
| **Anomaly score** | Association discrepancy (KL) | Reconstruction error |
| **训练方式** | Minimax on Gaussian prior | Adversarial training |
| **核心 insight** | 异常点 attention 分散 | 异常点重构难 |
| **是否需要 label** | 否 | 否 |
| **对 boundary anomaly 敏感度** | 中等 | 高（self-conditioning） |
| **对突发 spike anomaly** | 高（attention 立即分散） | 中等（取决于重构质量） |

**Fraud application：**
- 多变量 fraud 时序数据（金额、计数、对手方），TranAD 重构每个 channel，重构误差 = 异常分
- 适合"行为缓慢偏移"型 fraud（boundary anomaly），如 mule account 缓慢被注入异常流量

---

## 7. Graph Transformers（图 + Transformer 混合）

---

链上数据天然是 graph 结构（address-to-address transfer graph），传统 GNN 有 over-smoothing 和长距离传播弱的问题。Graph Transformer 用 attention 替代 message passing，结合两者优势。

### 7.1 Graphormer (2021, Microsoft)

**论文：** Ying et al., *Do Transformers Really Perform Badly for Graph Representation?* (NeurIPS 2021)。Graphormer **赢得了 OGB-LSC 2021 graph-level prediction track 冠军**，是 graph transformer 早期标杆。

**核心 insight：** vanilla Transformer 用于 graph 时缺少图结构信息（attention 是 token-pair 全连接，不知道图拓扑）。Graphormer 通过 **三种编码** 把图结构注入 attention：

**编码 1：Centrality Encoding（中心度编码）**

对每个 node，把它的 degree 编码成 embedding 加到 input token 上：
$$
x_{i}^{(0)} = h_{i} + z^{-}_{\deg^{-}(v_{i})} + z^{+}_{\deg^{+}(v_{i})}
$$

其中 $z^{-}, z^{+}$ 分别是 in-degree 和 out-degree 的 embedding（可学习）。

**直觉：** Node 的 degree 是它在图中重要性的简单 proxy。高 degree node（hub）应该有不同的 representation 倾向。在 fraud 图上，hub address（exchange / mixer 入口）和 leaf address（普通用户）行为差异巨大，centrality encoding 把这个信息显式注入。

**编码 2：Spatial Encoding（空间编码）—— 把图距离作为 attention bias**

修改 attention score：
$$
A_{ij} = \frac{(W^{Q} h_{i})(W^{K} h_{j})^{T}}{\sqrt{d}} + b_{\phi(v_{i}, v_{j})}
$$

其中 $\phi(v_{i}, v_{j})$ 是 $v_{i}$ 到 $v_{j}$ 的 **最短路径距离**，$b_{\phi}$ 是该距离对应的可学习 scalar bias。

**直觉：**
- 在 NLP 里相对位置 $j - i$ 重要，在 graph 里"相对位置"就是图距离 $\phi(v_{i}, v_{j})$
- 距离近的 node attention 大、距离远的 attention 小，但 attention 仍然是 dense（保留长距离可能性）
- 如果 $v_{i}$ 和 $v_{j}$ 不连通，$\phi$ 设为某个特殊值（如 $\infty$ 或 -1），$b$ 学到大负数把这种 pair 的 attention 压下去

**编码 3：Edge Encoding（边特征编码）**

shortest path 上的 edge 也带特征（如 transfer amount、tx timestamp）。Graphormer 把 shortest path 上 edge feature 的平均值作为额外 bias：
$$
A_{ij} \mathrel{+}= \frac{1}{N} \sum_{n=1}^{N} (e_{n})(w_{n})
$$

其中 $e_{n}$ 是 path 上第 $n$ 条边的 feature，$w_{n}$ 是位置相关的学习权重。

**复杂度：** $O(|V|^{2})$，因为是全连接 attention。对小图（hundreds of nodes）OK，对大图（millions of address）需要 subgraph sampling（GraphSAINT、ego-network 提取）。

**Fraud application：**
- Address 局部 ego-network 提取（中心 address + 1-hop / 2-hop 邻居）作为 subgraph
- Graphormer 对 subgraph 做 graph-level prediction（是不是 fraud cluster）
- Centrality + Spatial encoding 让模型显式知道"哪些是 hub、各 node 之间的距离"——这两个 signal 在 wash trading、layering 上极强（wash trading 的 cluster 通常呈环形 / 完全图结构）

### 7.2 Graph Transformer (general framework)

**论文：** Dwivedi & Bresson, *A Generalization of Transformer Networks to Graphs* (AAAI 2021 Workshop)；以及后续大量 GT 变种。

这不是某个具体模型，而是一类通用框架——**把 vanilla GNN 的 message passing 替换为 attention，但 attention 限制在邻居范围**。

**核心做法：**

1. **Attention 限定到邻居：** 对每个 node $i$，attention 只在它的 1-hop 邻居 $N(i)$ 内做 softmax（vanilla Transformer 是全 sequence）：
$$
\text{attn}(i, j) = \frac{\exp(s_{ij})}{\sum_{k \in N(i)} \exp(s_{ik})}, \quad j \in N(i)
$$

2. **保留 Transformer 的多头 + residual + LayerNorm + FFN：** 让信息聚合不只是简单 average / sum（vanilla GNN），而是 attention-weighted

3. **支持 edge features：** 把 edge feature 拼接到 $j$ 的 representation 上再算 attention

**vs GNN 优势：**
- **多头 attention** 让聚合权重 adaptive，比固定 weight（GCN）或简单 attention（GAT）更强
- **Residual + LayerNorm** 缓解 over-smoothing：堆 deeper layer 不会让所有 node representation 趋同
- **Edge feature 自然融入**：vanilla GNN 处理 edge feature 笨拙，GT 一开始就 design 进 attention 计算

**vs Graphormer 区别：**
- Graphormer 是 **全连接** attention + 用距离 bias 控制；GT 是 **邻居受限** attention，保持稀疏
- Graphormer 适合 small graph 的 graph-level task；GT 适合 large graph 的 node-level task

**Fraud application：**
- Large-scale address graph 上的 node-level fraud classification
- 每个 address node 的特征是 transformer 编码过的 tx history embedding（来自 BERT-style）
- GT 在 address graph 上做 attention-based message passing，输出 fraud probability per address

---

## 8. 变种对比总结（Comparison Summary）

---

| 模型 | 解决的核心问题 | 欺诈检测适用场景 | 输入类型 |
|------|--------------|----------------|---------|
| **BERT** | Bi-dir 表示学习（pretrain + fine-tune） | tx sequence → address embedding（离线 batch） | Token sequence |
| **RoBERTa** | BERT 训练不充分 | 同 BERT，但用更优训练 recipe | Token sequence |
| **DeBERTa** | Content 和 position 耦合在 attention 里 | tx sequence + 复杂时间/位置信号（timestamp delta + sequence position 双 channel） | Token sequence |
| **Longformer** | vanilla attention 的 $O(n^{2})$ 在长序列上不可承受 | 长 tx history（$10^{4}$ 笔）的 address-level scoring | Long token sequence |
| **TabTransformer** | Tabular categorical feature 的 attention 表示 | 离线 fraud classification on aggregate tabular features | Tabular (cat + num) |
| **FT-Transformer** | TabTransformer 不处理 numerical features 的局限 | 大规模 tabular + categorical / numerical 混合 + 强交互场景 | Tabular (all tokenized) |
| **TFT** | 多变量时序的 multi-horizon + 可解释 forecasting | 预测 address 未来行为分布，outlier 判定 anomaly | Multivariate time series |
| **PatchTST** | 长时序的 $O(T^{2})$ 复杂度 + 多变量过拟合 | 长时序单 channel forecast（如未来 24 小时每小时 tx count） | Long time series |
| **Anomaly Transformer** | 无监督异常检测 + label 稀缺 | Tx amount / count 时序的无监督异常打分 | Time series |
| **TranAD** | Boundary anomaly 检测灵敏度 | 缓慢偏移型 fraud（mule account） | Multivariate time series |
| **Graphormer** | Vanilla Transformer 用于 graph 缺图结构 | Address ego-network 上的 graph-level fraud cluster 分类 | Small graph |

---

## 面试 Q&A（Interview Q&A）

---

### Q1: BERT、RoBERTa、DeBERTa 三者的主要区别？做欺诈分类选哪个？

**回答：**

三者都是 encoder-only + MLM 范式，但每一代都解决了上一代的痛点。

1. **BERT (2018)**：开创了 encoder-only + pretrain + fine-tune 范式，但训练 trick 不够好且架构有可改进空间：
   - NSP 任务被后续证明无用（topic 而非 coherence）
   - 静态 masking 限制泛化
   - 位置编码（learned absolute PE）和 content 直接 add，attention 时无法解耦
   - BERT-base: 12L / 768H / 12A / 110M；BERT-large: 24L / 1024H / 16A / 340M

2. **RoBERTa (2019)**：**不改架构、只优化训练**——证明 BERT 远未训练充分：
   - 去掉 NSP，只保留 MLM
   - 动态 masking（每 epoch 重新决定 mask 位置）
   - Batch size 256 → 8K，训练 token 130B → 2.2T
   - Vocab 30K WordPiece → 50K Byte-level BPE
   - 在相同参数量下显著超过 BERT

3. **DeBERTa (2020)**：**架构创新**——首次 SuperGLUE 超过人类：
   - **Disentangled Attention：** content embedding 和 position embedding 分开维护，attention score 显式拆为 content-content + content-position + position-content 三项；位置用 relative position
   - **Enhanced Mask Decoder (EMD)：** 最后几层才注入 absolute position，让 representation 学习和位置 grounding 各司其职
   - 在 NLU 榜单上是当时（2020-2021）SOTA

4. **欺诈分类选型推荐**：
   - **首选 DeBERTa：** Disentangled attention 让 content 和 position 解耦——对 tx sequence 特别重要，因为"金额"（content）和"timestamp 间隔"（position）是两个独立信号，DeBERTa 的设计天然 fit
   - **次选 RoBERTa：** 工程成熟、训练稳定、HuggingFace 实现完善；如果团队没有充足 GPU 预算 pretrain，直接用 RoBERTa pretrained checkpoint fine-tune 是最实用的方案
   - **不推荐裸 BERT：** 除非有历史负担（旧 pipeline 已基于 BERT），否则没有理由用 BERT 而不用 RoBERTa

5. **实际部署考量**：
   - DeBERTa 的 disentangled attention 引入额外 forward 计算（约 1.5-2× BERT 的 FLOPs），latency 敏感场景需要测算
   - DeBERTa-v3 用 RTD (Replaced Token Detection) 替代 MLM 进一步提升，是 2024 年仍然有竞争力的 encoder model

> **Follow-up 提示：** 面试官可能问"DeBERTa 的 disentangled attention 比 RoPE 好吗？"——答：两者解决不同维度的问题。RoPE 是 position encoding 方式的优化（rotation 而非 add），是 decoder 友好的；DeBERTa 是 attention score 计算结构的优化，显式拆解 content 和 position 的交互项。两者可以同时用（一些后续工作尝试 RoPE + disentangled）。还可能问"如果 fraud 标签极少（<1万条），还选 DeBERTa 吗？"——答：标签少时倾向选 fine-tune 数据效率高的——RoBERTa pretrained checkpoint 数据多、覆盖通用语言知识，fine-tune 1 万条 fraud 比从头 pretrain DeBERTa 更稳。

---

### Q2: FT-Transformer 和 XGBoost 在 fraud detection 上各在什么场景下更好？

**回答：**

这是一个典型的"deep learning vs tree-based"权衡问题，要分维度讨论。

1. **FT-Transformer 更好的场景**：
   - **大数据量**（$10^{6}$+ rows）：Transformer 在大数据上 capacity 优势体现出来，能 fit 复杂分布
   - **高 cardinality categorical features**：链上场景常见——contract address vocab 上百万，FT 学 embedding 比 XGBoost 的 target encoding / one-hot 更高效
   - **复杂 feature 交互**：fraud detection 经常有 cross feature（如 method=swap × counterparty=mixer × time=night → 高风险），FT 的 multi-layer attention 能学高阶交互，XGBoost 树深限制只能学到树深以内的交互
   - **多 modality 融合**：FT 输出的 `[CLS]` embedding 可以和 text embedding（KYC 文本）、graph embedding（address graph）拼接进一步用 transformer 融合；XGBoost 是 dead-end，输出 scalar 难以做下游融合
   - **Pretrain 收益可用**：如果有大量 unlabeled tabular data（如全量历史 tx），可以先用 MLM-style 在 tabular 上 pretrain FT，再 fine-tune 到 fraud label 上——XGBoost 没有 pretrain 概念

2. **XGBoost 更好的场景**：
   - **中小数据**（$10^{3}$-$10^{5}$ rows）：树模型 sample efficient，少量数据就能学；transformer 在小数据上严重过拟合
   - **Tabular features 少且 well-engineered**：100 个手工 feature 的场景，GBDT 几乎打不过
   - **强 interpretability 需求**：监管 / 合规要解释为什么标了 fraud，XGBoost 有 gain importance、SHAP value，工业成熟；Transformer 的 attention 可视化能给 hint 但远不如 SHAP 可靠
   - **快速迭代**：CPU 几分钟训练完，比 transformer GPU 训练快几个数量级
   - **Online learning / 增量更新**：GBDT 有成熟增量训练方案；Transformer 增量训练困难，几乎只能全量重训
   - **缺失值原生处理**：树原生处理缺失值，transformer 需要 mask token + 特殊 embedding，工程复杂

3. **实际 fraud 系统的标准做法**：
   - **L0（baseline）：** XGBoost on engineered tabular features，快速出 baseline、验证 problem
   - **L1（升级）：** 如果有足够数据 + GPU，加 FT-Transformer 作为另一个模型
   - **L2（融合）：** FT-Transformer 输出的 representation embedding（不是预测）+ XGBoost 二级——FT 学复杂交互、XGBoost 做最终决策，两者互补
   - **L3（端到端）：** 完整 deep stack（tx sequence → BERT-style → address embedding → GT/Graphormer → fraud score），适合大公司 + 充足资源

4. **OKX-style 场景具体推荐**：
   - **新 chain 冷启动 fraud 模型：** XGBoost，因为数据少、需要快迭代
   - **成熟 chain（ETH / BSC）的 fraud 模型：** FT-Transformer + XGBoost stacking，数据充足、能容忍训练 overhead
   - **对抗性场景（fraudster 会试图绕过模型）：** XGBoost 优势——决策边界是 axis-aligned，attacker 更难 craft adversarial example；transformer 的 smooth 决策边界容易被 gradient-based attack

> **Follow-up 提示：** 面试官可能问"既然 stacking 经常更好，为什么不一开始就 stack？"——答：① stacking 模型部署复杂度高，需维护两套训练 pipeline + 推理 latency 翻倍；② 在 baseline 还没稳之前 stack 容易 overfit 到验证集；③ 工程上"先 simple 再 complex"是迭代的 best practice。还可能问"FT-Transformer 在 5K 数据上能 work 吗？"——答：基本不行——直接训会严重过拟合。Workaround：① 先在更大 unlabeled tabular dataset 上 MLM-pretrain FT，再 fine-tune 5K 标签；② 用很强的 regularization（dropout、weight decay、early stop）；③ 数据量真的少（<10K）就别 deep learning，老老实实 XGBoost。

---

### Q3: Anomaly Transformer 的 Association Discrepancy 是什么？为什么有效？

**回答：**

Association Discrepancy 是 Anomaly Transformer 的核心 contribution，用 attention pattern 本身作为 anomaly signal。

1. **直觉起点：normal vs anomaly 的 attention 模式差异**：
   - **Normal point：** 时序有局部 consistency，正常点和它的邻居（前后几步）模式类似，self-attention 计算时会发现"邻居很相关"，attention 概率集中在邻居（窄分布）
   - **Anomaly point：** 异常点和邻居"不一致"——比如突然 spike 的金额、突然中断的 tx 频率——attention 找不到强 local pattern，被迫 spread 到远处（宽分布，甚至接近 uniform）

2. **数学定义**：
   - **Prior-association** $P_{i,:}$：人工指定的"理想正常点 attention 分布"，用以 $i$ 为中心的 Gaussian：
$$
P_{i,j} = \frac{1}{\sqrt{2\pi}\sigma_{i}} \exp\left(-\frac{(j-i)^{2}}{2\sigma_{i}^{2}}\right)
$$
   $\sigma_{i}$ 可学习（控制 Gaussian 宽度，让 prior 拟合数据 local scale）
   - **Series-association** $S_{i,:}$：标准 self-attention softmax 输出（learned）
   - **Association Discrepancy**：两者的对称 KL：
$$
\text{AssDis}(i) = \frac{1}{2}[\text{KL}(P_{i,:} \| S_{i,:}) + \text{KL}(S_{i,:} \| P_{i,:})]
$$

3. **关键 trick：Minimax 训练**：
   - 单纯 minimize KL 会让 $S$ 退化为 Gaussian（series 直接抄 prior），失去学习意义
   - Anomaly Transformer 用 minimax 两阶段交替：
     - **Minimize 阶段：** 固定 $S$，只更新 $\sigma$，让 prior 拟合 series（这一步让 $\sigma$ 反映数据 local scale）
     - **Maximize 阶段：** 固定 $\sigma$，更新模型参数，让 series **远离 prior**（被迫学到更复杂的 long-range association）
   - 这种对抗式优化让正常点的 series 仍然集中在邻居（数据本身如此，model 即使被推开也会被 reconstruction loss 拉回），但异常点的 series 被训练得显著偏离 Gaussian

4. **Inference 时的 anomaly score**：
   - $\text{AnomalyScore}(i) = \text{ReconError}(i) \cdot \text{AssDisDirection}(i)$
   - 结合重构误差和 association discrepancy：reconstruction error 衡量"数值上不对"，association discrepancy 衡量"模式上不对"，乘起来对两种异常都敏感

5. **为什么有效（vs 传统方法）**：
   - **比 Autoencoder 强：** 单纯 reconstruction 在异常点上也可能 reconstruct 得不错（如果 anomaly 是 distribution 内但少见的 pattern）；association discrepancy 抓的是"局部 inconsistency"，互补
   - **完全无监督：** 不需要 fraud label，只需要 normal data（或假设大多数 data 是 normal）
   - **可解释：** 异常分高的位置直接对应"它和周围 timestep 关联不一致"，业务上容易解释

6. **Fraud application**：
   - Address 的 tx amount / count / fee 时序作为输入
   - 无监督训练 Anomaly Transformer（用大量正常 address 历史，假设其中 fraud 比例低）
   - Inference 时每笔 tx 出一个 anomaly score，超阈值的进 review queue
   - 适合冷启动（label 极少时）和 long-tail fraud（未见过的新型 fraud pattern，因为没有 label 可学，无监督才能 catch）

> **Follow-up 提示：** 面试官可能问"如果 fraud 在数据里占比高（如 20%），还能用吗？"——答：困难，因为 Anomaly Transformer 假设大多数训练数据是 normal。Workaround：① 用 outlier exposure 等技术筛掉明显 anomaly 后训练；② 半监督——把已知 fraud 标记的 sample 排除出训练集；③ 切换到有监督方法。还可能问"和 Isolation Forest 比优势在哪？"——答：① IF 处理 multivariate time series 弱（没有时间结构），Anomaly Transformer 显式建模时间；② IF 没有 representation learning，Anomaly Transformer 学到的 hidden state 可以作为 representation 给下游用；③ IF 的 anomaly score 是基于"被孤立的难易度"——抽象、难解释；Anomaly Transformer 的 score 有"attention pattern 偏离 Gaussian"的物理解释。

---

### Q4: 为什么 Graphormer 在图级别预测（graph-level prediction）上优于 GNN？

**回答：**

Graphormer 在 OGB-LSC 2021 graph-level track 上夺冠，相对 GNN 有几个本质优势。

1. **GNN 的根本局限：locality bias + over-smoothing**：
   - GNN（GCN / GAT / GraphSAGE）通过 message passing 聚合邻居信息，$k$ 层 GNN 只能看到 $k$-hop 邻居
   - **Over-smoothing：** 堆深 GNN 后所有 node representation 趋同（每层都做 neighbor average），$k > 5$ 后效果严重下降
   - **长距离信息丢失：** 图中相距 10 hop 的两个 node 在 4 层 GNN 里互相看不到，但它们可能在 graph-level task 上很相关

2. **Graphormer 的解决方案：全连接 attention + 图结构编码**：
   - **全连接 attention：** 每个 node 和所有其他 node 都做 attention，没有 locality 限制；长距离 node 也能直接交互
   - **Centrality encoding：** 把 degree 作为额外 embedding 注入——hub node 和 leaf node 在 representation 上立刻区分
   - **Spatial encoding：** 把 shortest path distance $\phi(v_{i}, v_{j})$ 作为 attention bias $b_{\phi}$——保留"距离近的 node 更相关"的归纳偏置，但不强制 locality
   - **Edge encoding：** path 上 edge feature 也注入 attention bias

3. **核心优势之一：long-range interaction 无损**：
   - GNN 跨 10 层把 signal 传过去会严重稀释（每层 ReLU + average 都损失信息）
   - Graphormer 一层 attention 就能让相距 10 hop 的 node 直接计算相关性
   - 在分子性质预测里，例如分子的 HOMO-LUMO gap 取决于全分子电子结构，long-range 极重要——Graphormer 显著超过 GNN

4. **核心优势之二：可控的 inductive bias**：
   - Vanilla Transformer 在图上的问题是"忘了图结构"——Graphormer 通过三种编码 explicit 注入，但仍保持 attention 的 flexibility（不强制 hard locality）
   - $b_{\phi}$ 是可学习的——模型自己决定"距离 1 的 bias 多大、距离 5 的 bias 多大"，比 GNN 的 hard locality 更灵活

5. **核心优势之三：graph-level readout 自然**：
   - GNN 要做 graph-level prediction 必须设计 pooling（mean / max / sum / attention pool），都是 ad-hoc
   - Graphormer 加 `[VNode]` virtual node，连接所有 node、距离 1，attention 自然聚合 graph-level information；`[VNode]` 的最终 hidden state 直接做 graph-level prediction，干净

6. **trade-off 和局限**：
   - **复杂度 $O(|V|^{2})$：** 对小图（$|V| < 500$）友好，对大图（million-node address graph）必须 subgraph sampling
   - **需要 explicit 图结构：** Spatial encoding 要 precompute shortest path distance，每个 batch 都要算，工程上要 cache 或在 dataloader 里 parallelize

7. **Fraud detection 应用**：
   - **Graph-level fraud cluster detection：** 提取 address 1-2 hop ego-network（小图），Graphormer 判断"这个 cluster 整体是 fraud cluster 吗"（如 wash trade ring、layering chain）
   - **Centrality encoding 的具体含义：** 链上 wash trade ring 通常是 cycle / complete graph，每个 node degree 类似；正常 user graph 有 hub-and-spoke 结构。Centrality encoding 让模型一眼区分这两类拓扑

> **Follow-up 提示：** 面试官可能问"Graphormer 在 node-level prediction 上也好吗？"——答：node-level task 上 Graphormer 不如纯 GT 或 GNN，因为 ① 大图 $O(|V|^{2})$ 不可承受；② node-level 任务 local pattern 占主导，全连接 attention 多余。Graphormer 的甜区是 graph-level + small graph。还可能问"Spatial encoding 的 shortest path 怎么算？大图怎么办？"——答：BFS 算 shortest path，对 small graph 几秒搞定。大图需要 sampling subgraph 后再算（如 GraphSAINT 采 10K node subgraph）；也有 work 用 truncated distance（限制最大 $\phi$ 为 5，距离 > 5 视为同一类）减少 unique distance 数量。

---

### Q5: 对于一个新地址（交易历史很短，比如只有 5 笔），如何处理 sequence 太短的问题？

**回答：**

新地址 (new wallet) 是 fraud detection 的典型 cold-start 场景，多个角度可以缓解。

1. **Padding + 模型适配**：
   - 把短序列 pad 到固定长度（如 64 或 128），用 mask 告诉模型哪些位置是 pad
   - Transformer 训练时本来就有 padding mask 处理（attention 不 attend pad 位置），所以技术上 trivial
   - 但纯 padding 不解决"信息不足"问题——5 笔 tx 的 representation 信息量本来就少

2. **多 source 特征融合（关键策略）**：
   - **Address-level static features：** address age（首笔 tx 距今天数）、是否来自 known exchange、initial funding source 风险等级
   - **Counterparty inheritance：** 新 address 的对手方（哪个 address 给它发了第一笔 tx）的风险特征"继承"过来——如果第一笔来自 mixer 或已 label fraud 的 address，新 address 立刻高分
   - **Graph context：** 这 5 笔 tx 形成的 1-hop / 2-hop subgraph 喂给 Graph Transformer，graph 结构本身也是 signal（如果这 5 笔形成 cycle，就是 wash trade 嫌疑）
   - **协议元数据：** 5 笔 tx 涉及哪些 contract、method id 分布、token 类型——even short sequence 的 metadata 仍有 signal

3. **Few-shot learning / meta-learning 思路**：
   - 把 fraud detection 重新 formulate 为 few-shot classification：训练时 sample 出"5 笔 tx 的子序列"作为 episode，meta-train 模型让它在短序列上也能 generalize
   - **Prototypical Networks：** 学每个 fraud type 的 prototype embedding，新 address 的短序列 embedding 和各 prototype 算 distance 决定分类
   - **Reptile / MAML：** Meta-learning 让模型快速适应 small support set

4. **半监督 / self-supervised 预训练**：
   - 用大量长 sequence（active address）做 MLM-style pretrain，学到"什么是 normal tx pattern"
   - 新 address 的短序列直接用 pretrained encoder 编码——pretrained representation 本身有先验，短序列也能得到合理 embedding
   - 这是 OKX-style 工业最常用方案

5. **模型层面的 architecture choice**：
   - **避免纯 sequence model（如 PatchTST），它要求至少 1 个 patch（16-64 笔）；5 笔不够**
   - **TabTransformer / FT-Transformer + sequence aggregation：** 把 5 笔 tx 聚合成 tabular feature（mean amount、unique counterparty count、max gas），用 tabular transformer 处理
   - **Hybrid：** 当 sequence 长度 $> K$ 时用 sequence model，$\leq K$ 时切到 tabular model；两个模型共享 pretrain

6. **业务规则 fallback**：
   - 极短序列（< 3 笔）几乎无法 ML 建模，直接用规则：first-funding source 是 mixer / sanctioned address → 高分；金额 > 10万 USD → 高分
   - 等 address 积累更多 tx 后再切到 ML 模型——分层 system

7. **Cold-start period 的 risk policy**：
   - 不一定要"建模出准确分数"，可以"用 conservative policy 替代"——新 address 默认放在更严格的 review queue，等积累足够 history 再降低 friction
   - 这是 product side 的策略，但 modeling team 要明白：cold-start 的最优解经常是 product，不是 pure model

> **Follow-up 提示：** 面试官可能问"5 笔 tx 的特殊情况——其中 4 笔来自同一 known mixer，你的模型怎么处理？"——答：这是 high-signal 情况，counterparty inheritance 立即起作用。把 counterparty 的 risk embedding 当作 feature 拼接到 tx token 上，attention 计算时直接看到"4/5 对手方是高风险"——分类 head 应该输出高分。可能问"如果 train 时只有长序列（pretrain corpus），inference 时遇到极短序列会出现 distribution shift 吗？"——答：会。Mitigation：① 训练时随机 truncate sequence 到不同长度（5、10、50、100），让模型见过短序列；② 用 length-aware position encoding（如 ALiBi），不依赖固定 max length。

---

### Q6: Longformer 的 sliding window attention 如何同时保留全局信息？

**回答：**

Longformer 的关键 design 是 **sparse local attention + sparse global attention 的组合**——既保留 long sequence 处理能力，又不丢全局 signal。

1. **Sliding window attention（局部）**：
   - 每个 token 只 attend 到自己前后 $w/2$ 个 token（典型 $w = 512$）
   - 复杂度 $O(n \cdot w)$，对 $n = 8K$ 来说是 $4M$ FLOPs 量级（vanilla self-attention $O(n^{2}) = 64M$）
   - 单层只看 $w$ 范围，但 stack $L$ 层后**感受野扩展到 $L \cdot w$**（类似 CNN 的 receptive field 累积）

2. **Dilated sliding window（扩展感受野）**：
   - 在高层用空洞 attention：每隔 $d$ 个 token 取一个，windowsize 不变但跨度变大
   - 感受野从 $L \cdot w$ 扩到 $L \cdot w \cdot d$，让信息传播更远
   - 类似 Dilated Convolution 思路

3. **Global attention（全局）**：
   - 少量特殊 token 设为 "global"：它们 attend 到所有 token，同时所有 token 也 attend 到它们
   - 双向：global token 看到所有人 + 所有人看到 global token
   - Global token 的选择：
     - 分类任务的 `[CLS]` token
     - QA 任务的 question token（让 question 看到全文，全文也看 question）
     - 任务相关的特定 token（如 NER 任务里所有 entity 候选位置）
   - 复杂度：$g$ 个 global token，额外 $O(n \cdot g)$ —— $g$ 通常 $\ll n$（个位数到几十）

4. **为什么这样设计能保留全局信息**：
   - 单纯 sliding window：信息传播需要 $n / w$ 层才能跨 sequence，深度不够会丢长距离 signal
   - 加 global token：任意两个普通 token 可以通过 global token **中转**——两步即可（普通 token → global → 普通 token），相当于全局信息 hub
   - 类比：sliding window 是高速公路上的近距离行驶，global token 是机场——任何两地都可以通过机场快速连接

5. **Global token 在 fraud detection 上的应用**：
   - **`[CLS]` 作为 global token：** 用于 address-level fraud classification。`[CLS]` 看到所有 tx，所有 tx 看到 `[CLS]`，最终 `[CLS]` hidden state 聚合 address-level signal
   - **关键 tx 作为 global token：** 已 label 为可疑的 tx 设为 global，让它和所有其他 tx 强交互，help 判断"这笔可疑 tx 是否真异常 vs 是否是 fraud cluster 的一部分"
   - **Recent tx 作为 global token：** 最近的 K 笔 tx 设 global，让它们和长 history 任意 tx 交互，模拟"当前 risk 同时看过去远期 context"

6. **复杂度对比**：

   | 模型 | 单层 attention 复杂度 | $n = 8K$ 时 |
   |------|---------------------|------------|
   | vanilla self-attention | $O(n^{2})$ | $64M$ ops |
   | Longformer sliding window $w=512$ | $O(n \cdot w)$ | $4M$ ops |
   | Longformer + global tokens $g=10$ | $O(n \cdot w + n \cdot g)$ | $4.08M$ ops |

7. **工程实现：custom CUDA kernel**：
   - Sliding window 的 attention 是 banded matrix（带状），不能用通用 dense matmul kernel
   - Longformer 提供 custom CUDA kernel（基于 TVM）实现 banded attention，能在 GPU 上接近 dense attention 的速度
   - 没有自定义 kernel 的情况下，Longformer 反而比 dense attention 慢（因为 GPU 优化的是 dense matmul）

> **Follow-up 提示：** 面试官可能问"global token 设置太多会怎样？"——答：复杂度从 $O(nw)$ 变成 $O(ng)$，如果 $g$ 接近 $n$ 就退化为 vanilla attention 失去意义。Longformer 设计假设 global token 数量是 task 相关的几个到几十个，通常远小于 $n$。还可能问"为什么不直接用 Performer / Linear Transformer？"——答：选择取决于"attention 是 sparse 还是 dense"。fraud 序列的关键信号经常局部 + 少量长距离（mixer 污染、关键 entry tx），sparse + global pattern 完美 fit；Performer 是 dense approximation，在 sparse pattern 上反而 over-engineering。

---

### Q7: 在真实欺诈检测系统中，你会选择哪种 Transformer 变种？说明理由。

**回答：**

实际生产系统不会是单一 transformer，而是多模型组合。我会按"输入数据类型 + 业务延迟约束"分层选型。

1. **整体架构设计：多模型组合**：
   ```
   原始数据
     ├── tx sequence (per address)        → BERT-style transformer  → address embedding
     ├── tabular aggregate features        → FT-Transformer / XGBoost → tabular score
     ├── time series (daily metrics)       → PatchTST / Anomaly Trans → behavior score
     ├── address graph (ego-network)       → Graph Transformer        → graph embedding
     └── time series + tx hist + graph fused → final stacker (XGBoost / MLP) → fraud probability
   ```

2. **每个 layer 选型理由**：
   - **Address embedding（tx sequence encoder）：** **RoBERTa-style encoder + Longformer attention**
     - 选 encoder-only：fraud 是 classification，bi-dir 优势可用
     - 选 RoBERTa-style 训练 recipe：MLM + 动态 masking + 大 batch，pretrain 充分
     - 选 Longformer attention：active address 有 $10^{4}$ 笔 tx 历史，vanilla attention 算不动
     - Pretrain 策略：MLM on 全量历史 tx（unlabeled），fine-tune on labeled fraud cases
   - **Tabular features：** **XGBoost baseline + FT-Transformer stacking**
     - XGBoost 出 quick baseline + 提供 interpretable feature importance（合规需求）
     - FT-Transformer 在 high-cardinality categorical（contract address、token id）+ 大数据上提升
     - 两者 stack：XGBoost 决策 + FT-Transformer 输出 embedding 作为额外 feature
   - **Behavior time series：** **PatchTST forecast + Anomaly Transformer**
     - PatchTST 预测每日 tx count / amount，预测值 vs 实际值 deviation = anomaly signal
     - Anomaly Transformer 做 unsupervised anomaly score（label 之外的兜底）
   - **Graph：** **Graph Transformer + Centrality encoding（轻量版 Graphormer）**
     - Subgraph 提取 address 2-hop ego-network
     - Graph Transformer 学 message passing 替代纯 GNN，过 smoothing 缓解
     - Centrality encoding 让 hub address（exchange / mixer）和 leaf 区分

3. **延迟分级部署**：

   | 模式 | 延迟约束 | 用什么模型 |
   |------|---------|----------|
   | **Offline batch（每天全量评分）** | 数小时 | 全部模型，完整 stack；可用大 transformer + Graph Transformer |
   | **Near-real-time（5 秒）** | < 5s | Tabular model (XGBoost) + cached address embedding（pretrained 模型每天 refresh 一次） |
   | **Hard real-time（用户提现 < 100ms）** | < 100ms | Decoder-only model + KV cache 增量更新；最近的 address embedding 从 feature store 读 |

4. **关键 trade-off 决策**：
   - **不会选 GPT-style decoder 做 fraud classification：** Encoder-only 的 bi-dir 优势在 offline scoring 上不可放弃
   - **不会一开始就上 Graphormer：** 复杂度 $O(|V|^{2})$ + subgraph sampling 工程负担重，先用 GNN 做 baseline，确认有提升空间后再升级
   - **不会盲目用 Anomaly Transformer 替代 XGBoost：** Unsupervised anomaly 是补充不是替代，主路径仍然 supervised
   - **会重视 Pretrain：** Fraud labels 稀缺（< 1% 数据有 label），unsupervised pretrain on tx 是最 ROI 的工程投资

5. **对抗性的考量（fraudster 主动绕过模型）**：
   - 单一 deep model 容易被 gradient-based attack 绕过——多模型 ensemble 提升 robustness
   - XGBoost 的 axis-aligned 决策边界 attacker 更难 craft adversarial example，作为 ensemble 一员能稳住
   - Graph signal 难以伪造：fraudster 可以改自己的 tx 行为，但很难改"我的对手方都是什么 address"——graph feature 是 robust signal

6. **冷启动 + 长尾的 fallback**：
   - 新链 / 新 address：tabular + rule-based first，等数据积累再切 deep
   - Long-tail / 新型 fraud：Anomaly Transformer + 人工审核，label 后回流到 supervised 训练

7. **简洁回答：**
   "**我会选 RoBERTa 思想 + Longformer attention 的 encoder 做 tx sequence 编码 + FT-Transformer 处理 tabular + Graph Transformer 处理 graph，最后用 XGBoost stacker 融合。XGBoost 在生产 inference 链路上保留兜底作用。**"

> **Follow-up 提示：** 面试官可能问"为什么不直接训一个 multi-modal transformer 端到端处理所有信号？"——答：理论上可行，但工程上 ① 不同 modality 数据 schema / 采集频率不同，统一 tokenize 困难；② 每个 modality 独立训能并行迭代，多模态端到端要全栈一起调；③ 多模型组合各自 SLA 独立，单模型 fail 不影响整体；④ Stack approach 让 XGBoost feature importance 仍可用于 explain fraud decision，满足合规需求。还可能问"延迟敏感场景怎么办？"——答：① 离线 pretrain → 缓存 address embedding 到 feature store；② Real-time 链路只查 cache + 跑 small XGBoost，sub-100ms 可达；③ 长尾不在 cache 的 address 走"打回人工审核"或 conservative default。

---

### Q8: PatchTST 和 iTransformer 在时间序列上的本质区别是什么？

**回答：**

两者代表对"时序 transformer 应该如何切 token"的两种相反 philosophy。

1. **PatchTST：时间维度上切 patch**：
   - 把单变量时序 $\{x_{1}, x_{2}, \dots, x_{T}\}$ 切成 patch：每 $p$ 个 timestep 是一个 patch，共 $n = T / p$ 个 patch
   - 每个 patch 内的 $p$ 个 value 一起线性投影成一个 $d$-dim token
   - Token 数 $n = T/p$，attention 在 $n$ 个 patch 之间进行
   - **Channel-independent：** 多变量时不做 channel 间交互，每个 channel 独立 forward（共享参数）

2. **iTransformer：变量维度上切 token**：
   - 多变量时序 $X \in \mathbb{R}^{T \times M}$（$T$ 时间步、$M$ 变量）
   - 把**每个变量的完整时序**（$T$ 维向量）投影成一个 $d$-dim token
   - Token 数 $= M$（变量数），attention 在 $M$ 个变量之间进行
   - **FFN 在 token 内**：每个 token 是 $d$-dim，FFN 处理这个 $d$-dim 向量（即处理"该变量的整段时序 pattern"）

3. **核心区别对比表**：

   | 维度 | PatchTST | iTransformer |
   |------|----------|--------------|
   | Token 是什么 | 时序的一个 patch（$p$ 个 timestep 的 value） | 一个变量的完整时序（$T$ 维向量） |
   | Token 数量 | $T / p$（很多，与时序长度相关） | $M$（少，与变量数相关） |
   | Attention 跨什么 | 跨时间 patch（时间维交互） | 跨变量（channel 维交互） |
   | FFN 处理什么 | 每个 patch 内的 representation | 每个变量的整段时序 pattern |
   | 序列长度敏感 | 是（$T$ 大时复杂度敏感） | 否（attention 复杂度只依赖 $M$） |
   | 变量数敏感 | 否（变量独立） | 是（变量多时 attention 大） |

4. **设计哲学的差异**：
   - **PatchTST 假设：** 单变量内有强时间局部 pattern（cycle、trend），变量间近似独立或交互弱
   - **iTransformer 假设：** 变量间有强 cross effect（多变量协同变化），变量内的时序 pattern 用 MLP 学就够

5. **何时 PatchTST 更好**：
   - 变量数少（$M \leq 10$）或变量间几乎独立（如不同物理 sensor）
   - 单变量内有清晰 cycle / trend / seasonality（如电力消耗的日周月 pattern）
   - 时序长但变量少：$T = 5000, M = 5$，PatchTST 切 patch 后处理 patch 间 attention
   - **Fraud 场景：** 单 address 的 daily tx count forecast（forecast 未来 24 小时），单变量、长时序、强日内 pattern—— PatchTST 合适

6. **何时 iTransformer 更好**：
   - 变量数中等且变量间有 cross effect（$M = 10-100$）
   - 业务上明确变量耦合（如 fraud 场景：daily tx count、daily amount、daily unique counterparty 三者同涨同跌）
   - 时序长度极大但变量少：iTransformer attention $O(M^{2})$ 与 $T$ 无关，对长序列友好
   - **Fraud 场景：** Address 多 channel 时序（amount + count + counterparty diversity + gas pattern），变量间强相关，iTransformer 显式建模 cross-variable attention

7. **共同 insight：channel-independent / 反转 attention 的有效性**：
   - 两者都反对"传统时序 transformer 每个 timestep 一个 token + 多 channel concat"的设计
   - PatchTST 通过 channel-independent 避免 channel 噪声干扰
   - iTransformer 通过反转把"channel attention"和"time pattern"分到不同模块
   - 共同 lesson：**多变量时序里 channel 和 time 应该分开建模，不要混在 attention 里**

8. **实践选择简单 rule**：
   - $T \gg M$ 且单变量 pattern 强：PatchTST
   - $M$ 中等且变量耦合强：iTransformer
   - 不确定：两个都跑一遍，benchmark 数据上选好的；不同业务数据集结论可能反

> **Follow-up 提示：** 面试官可能问"既然两者各有优势，能不能结合？"——答：可以——已有工作（如 2024 年的 TimeXer、UniMTS）尝试"variable-level token + time-level token 同时存在 + cross-attention 在两类 token 间交互"。架构复杂，效果在多个 benchmark 上是 SOTA，但工程实现不成熟。还可能问"channel-independent 看起来反直觉，为什么有效？"——答：① 多 channel concat 时容易 overfitting 噪声 channel；② channel-independent 等于 batch 维度扩大，训练更稳；③ 不同 channel 的 cross effect 由后续 MLP head 隐式学，attention 模块专注时间维更高效。是"deep learning 里 prior 比 capacity 重要"的又一例证。

---
