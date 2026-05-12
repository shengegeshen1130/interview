# OKX Anti-Fraud Interview Prep — Transformer Fraud Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create 4 comprehensive interview study files in `interview_question/okx_anti_fraud/` covering blockchain data fundamentals, transformer architecture, transformer variants, and transformer-based fraud detection systems.

**Architecture:** Each file is self-contained and builds on the previous. File 1 establishes blockchain data vocabulary. File 2 solidifies transformer math. File 3 covers the variant landscape. File 4 synthesizes everything into an end-to-end fraud detection system design.

**Tech Stack:** Markdown, LaTeX math (`_{...}` subscript notation), Chinese/English bilingual format matching existing repo conventions (`### Q{N}:` → `**回答：**` → `> **Follow-up 提示：**`)

---

## File Map

| File | Path | Responsibility |
|------|------|----------------|
| Blockchain Primer | `interview_question/okx_anti_fraud/01_blockchain_data_primer.md` | Blockchain data structures, fraud patterns, on-chain feature vocabulary |
| Transformer Architecture | `interview_question/okx_anti_fraud/02_transformer_architecture.md` | Full attention math, positional encoding, block components, training details |
| Transformer Variants | `interview_question/okx_anti_fraud/03_transformer_variants.md` | BERT/GPT families, tabular/time-series/anomaly/graph variants, comparison table |
| Fraud Detection Application | `interview_question/okx_anti_fraud/04_fraud_detection_with_transformers.md` | End-to-end system design, 4 modeling approaches, production concerns, OKX-specific patterns |

---

## Task 1: Create Folder and Write Blockchain Data Primer

**Files:**
- Create: `interview_question/okx_anti_fraud/01_blockchain_data_primer.md`

- [ ] **Step 1: Create the output folder**

```bash
mkdir -p interview_question/okx_anti_fraud
```

- [ ] **Step 2: Research blockchain transaction data structures**

Search for:
- "blockchain UTXO vs account model transaction structure"
- "ethereum transaction data fields gas fee nonce"
- "on-chain fraud patterns money laundering blockchain"
- "blockchain address clustering features machine learning"

Focus on: what fields exist in a raw transaction, how UTXO differs from account model, what derived features ML engineers extract for fraud.

- [ ] **Step 3: Research blockchain fraud taxonomy**

Search for:
- "DeFi fraud taxonomy flash loan attack rug pull wash trading"
- "blockchain money laundering mixing layering techniques"
- "crypto phishing wallet drainer on-chain patterns"
- "Ponzi scheme blockchain on-chain detection features"

Focus on: what each fraud type looks like in raw transaction data (amounts, timing, address patterns).

- [ ] **Step 4: Write `01_blockchain_data_primer.md`**

Write the file with this exact structure:

```markdown
# 区块链数据基础 (Blockchain Data Primer)

> OKX Anti-Fraud 面试准备 · File 1 of 4

---

## 1. 什么是区块链 (What Is a Blockchain)

[Explain: distributed ledger, blocks linked by hash, transactions inside blocks, consensus mechanisms (PoW/PoS) at a high level sufficient for ML context. NOT a crypto course — just enough to understand what the data is.]

---

## 2. 交易数据结构 (Transaction Data Anatomy)

### 2.1 UTXO 模型 (Bitcoin-style)

[Explain UTXO: unspent transaction outputs, inputs reference previous UTXOs, change outputs, why there's no concept of "account balance" in raw data]

| 字段 | 说明 | ML意义 |
|------|------|--------|
| txid | 交易哈希 | 唯一标识 |
| inputs | 引用的 UTXO | 资金来源 |
| outputs | 新 UTXO (地址 + 金额) | 资金去向 |
| block_height | 所在区块高度 | 时间特征 |
| fee | 矿工费 | 异常特征 |

### 2.2 账户模型 (Ethereum-style)

[Explain account model: from/to addresses, value, gas price, gas limit, nonce, data field for smart contract calls]

| 字段 | 说明 | ML意义 |
|------|------|--------|
| from | 发送地址 | 节点特征 |
| to | 接收地址/合约 | 节点特征 |
| value | ETH金额 (Wei) | 金额特征 |
| gas_price | Gas单价 | 紧急程度特征 |
| gas_used | 实际消耗Gas | 合约复杂度 |
| nonce | 发送方交易序号 | 序列特征 |
| input | 合约调用数据 | 行为特征 |
| block_timestamp | 区块时间戳 | 时间特征 |

---

## 3. 链上实体类型 (On-Chain Entity Types)

[Cover: EOA (externally owned accounts) vs. smart contracts, token contracts (ERC-20 fungible, ERC-721 NFT), DEX (Uniswap/Sushiswap) swap transactions, bridge contracts]

---

## 4. 区块链数据作为ML数据集 (Blockchain Data as an ML Dataset)

### 4.1 表格视角 (Tabular View)

[Each transaction = one row. Address-level aggregated features:]

| 特征类别 | 示例特征 |
|---------|---------|
| 交易量特征 | tx_count_7d, total_volume_30d, avg_tx_value |
| 时间特征 | account_age_days, active_hours_distribution, tx_frequency |
| 对手方特征 | unique_counterparties, new_counterparty_ratio |
| 合约交互 | defi_protocol_count, contract_call_ratio |
| 金额模式 | round_number_ratio, amount_entropy, max_single_tx |

### 4.2 图视角 (Graph View)

[Transaction graph: nodes = addresses, edges = transfers (weighted by amount, labeled by token). Graph features: in-degree, out-degree, PageRank, clustering coefficient, connected component size.]

### 4.3 序列视角 (Sequence View)

[An address's transactions sorted by timestamp = a sequence. Each transaction = a token/event. This enables sequence modeling (transformer).]

---

## 5. 区块链欺诈模式 (Fraud Patterns in Blockchain)

### 5.1 洗钱 (Money Laundering)

[Layering: funds pass through many intermediate wallets to obscure origin. Mixing services (Tornado Cash). On-chain signals: many small equal transfers, short-lived wallets, fan-out fan-in patterns.]

### 5.2 庞氏骗局 (Ponzi Schemes)

[Early investors paid from new investor funds. On-chain signal: address receives many small deposits, makes large payouts to early depositors, eventually drains.]

### 5.3 钓鱼/地址污染 (Phishing / Address Poisoning)

[Attacker creates addresses visually similar to victim's. Sends tiny tx to pollute victim's tx history. Victim copies wrong address. On-chain signal: addresses with no outgoing txs, dust amounts.]

### 5.4 闪电贷攻击 (Flash Loan Attacks)

[Borrow massive uncollateralized funds in single tx, manipulate DeFi protocol, repay in same tx. On-chain signal: single tx with massive loan + repay + profit in same block, unusual gas usage.]

### 5.5 抹布拉盘 (Rug Pulls)

[Project team drains liquidity pool after raising funds. On-chain signal: liquidity removal tx by contract owner, token price → 0.]

### 5.6 虚假刷量 (Wash Trading)

[Self-trading to inflate volume on DEX or NFT marketplace. On-chain signal: circular fund flows between related addresses, abnormal trade frequency.]

---

## Interview Q&A

### Q1: 在区块链欺诈检测中，UTXO 模型和账户模型的数据有什么主要区别？对特征工程有什么影响？

**回答：**

1. **UTXO 模型（比特币）**：没有账户概念，每笔交易消耗之前的 UTXO 并创建新的 UTXO。"余额"需要通过聚合该地址所有未花费 UTXO 计算得出。地址可以单次使用（HD wallet），导致同一用户可能有数百个地址，**地址归因（address clustering）** 是关键挑战。
2. **账户模型（以太坊）**：有明确的 from/to/value 字段，更接近传统金融交易。但多了 smart contract 交互（input data），需要解析 ABI 才能理解调用了什么函数。
3. **特征工程影响**：UTXO 需要先做地址聚类再提取用户级特征；以太坊可以直接按地址聚合，但需要区分 EOA 行为和合约行为，token transfer 需要从 event log 解析而非直接读取 value 字段。

> **Follow-up 提示：** 如何在 UTXO 链上做地址归因？（Common Input Ownership Heuristic: 同一笔交易的所有 input 地址通常归属同一实体）

### Q2: 你会如何把区块链地址的交易历史构造成 Transformer 的输入序列？

**回答：**

1. **序列定义**：以地址为主体，按时间戳排序该地址的所有交易，每笔交易视为一个 token/event。
2. **每个 token 的特征**：`[amount, direction(in/out), counterparty_type, hour_of_day, days_since_last_tx, gas_fee, is_contract_call]` — 这些连续/类别特征需要先 embedding 或归一化后 concatenate，作为 token 的 input representation。
3. **序列截断**：活跃地址交易数量差异很大（几笔到数万笔），需要固定窗口（取最近 N 笔）或分层采样。
4. **特殊 token**：加入 `[CLS]` token（BERT 风格）用于分类任务，或 `[SEP]` 分隔不同时间窗口。

> **Follow-up 提示：** 金额是连续值，如何输入 Transformer？（log normalization + linear projection 到 d_model 维度；或者分 bucket 后做 embedding）

### Q3: Flash loan attack 在链上有哪些可检测的特征？

**回答：**

1. **原子性**：整个攻击在 **单笔交易** 内完成（借款 + 操作 + 还款），因此 gas_used 极高（单笔可达 200万+ gas）。
2. **合约调用链**：Internal transaction 深度很深，一笔 tx 会 call 多个 DeFi 协议（Aave → Uniswap → 目标协议 → Uniswap → Aave）。
3. **金额异常**：借款金额远超地址历史均值，通常是数百万美元量级。
4. **价格异常**：同一区块内目标代币价格剧烈波动（DEX 价格 oracle 被操纵）。
5. **检测方法**：在交易级别提取以上特征，用异常检测或分类模型；也可以在区块级别检测同区块内相关协议的价格偏差。

> **Follow-up 提示：** Flash loan 本身不是欺诈（合法套利也用），如何降低误报率？（结合 profit address 分析 — 获利地址是否是已知合法套利机器人）

### Q4: 为什么图模型对区块链欺诈检测特别重要？

**回答：**

1. **多跳洗钱**：洗钱通常经过 5-20 个中间地址，每个单点看起来都正常，只有把 **整个转账链** 作为图才能检测到异常结构（如 fan-out → fan-in 的"混币"模式）。
2. **团伙特征**：同一个诈骗团伙会控制多个地址，这些地址之间有密集交易，图的 community detection 可以发现这些集群。
3. **图特征丰富度**：PageRank（地址重要性）、clustering coefficient（是否处于密集小团体）、shortest path to known fraud（到已知欺诈地址的距离）这些特征无法从单笔交易中提取。
4. **实践**：OKX JD 中特别提到 Neo4j / Amazon Neptune，说明公司用图数据库存储交易图，用 Cypher/Gremlin 做图查询，结合 GNN 做图级别的欺诈检测。

> **Follow-up 提示：** 图数据库 vs. 关系型数据库做图查询的区别？（图数据库的 index-free adjacency 使多跳查询 O(1) per hop，关系型数据库 JOIN 随跳数指数级增长）

### Q5: Wash trading 的链上检测思路是什么？

**回答：**

1. **环形资金流**：A → B → C → A 或 A → B → A，可以用图算法检测 cycle（特别是短 cycle，2-4 hop）。
2. **地址关联性**：参与 wash trading 的地址通常由同一实体控制，可通过：同一 gas 钱包充值（共同资金来源）、交易时间高度相关、IP/设备指纹（链下数据）来关联。
3. **成交价格分布**：真实交易价格应该符合市场价格分布；wash trading 成交价格可能偏离市价或高度集中。
4. **频率异常**：同一地址对之间交易频率远超市场均值。

> **Follow-up 提示：** 没有链下数据（IP/设备）时，纯链上能检测 wash trading 吗？（可以，基于资金流环路 + 地址行为相似度）
```

- [ ] **Step 5: Verify all spec sections are covered**

Check against spec: blockchain basics ✓, UTXO vs account model ✓, on-chain entities ✓, dataset views (tabular/graph/sequence) ✓, fraud patterns (6 types) ✓, 5 Q&A ✓

- [ ] **Step 6: Commit**

```bash
git add interview_question/okx_anti_fraud/01_blockchain_data_primer.md
git commit -m "feat: add blockchain data primer for OKX anti-fraud interview"
```

---

## Task 2: Write Transformer Architecture Deep Dive

**Files:**
- Create: `interview_question/okx_anti_fraud/02_transformer_architecture.md`

- [ ] **Step 1: Research attention mechanism details**

Search for:
- "transformer attention mechanism Q K V intuition derivation"
- "multi-head attention what different heads learn"
- "pre-norm vs post-norm layer normalization transformer"
- "rotary positional encoding RoPE vs sinusoidal ALiBi"

Focus on: the math behind scaled dot-product attention, why we divide by √d_k, what Q/K/V represent semantically, what different attention heads capture empirically.

- [ ] **Step 2: Research training and complexity**

Search for:
- "transformer masked language model pre-training fine-tuning"
- "transformer O(n²) complexity sequence length bottleneck"
- "transformer vs LSTM vanishing gradient sequential bottleneck"

- [ ] **Step 3: Write `02_transformer_architecture.md`**

```markdown
# Transformer 架构详解 (Transformer Architecture Deep Dive)

> OKX Anti-Fraud 面试准备 · File 2 of 4

---

## 1. 为什么需要 Transformer (Why Transformers)

### 1.1 RNN/LSTM 的局限性

| 问题 | 具体表现 |
|------|---------|
| 顺序计算瓶颈 | 必须逐步处理序列，无法并行化，训练慢 |
| 长距离依赖退化 | 梯度信号经过多步后消失/爆炸，捕捉 100+ 步外的依赖困难 |
| 信息瓶颈 | 所有历史信息压缩进固定大小的 hidden state |
| LSTM 的 cell gate 只是缓解 | 并非根本解决，长序列（>512）效果仍下降 |

### 1.2 Attention 的核心思想

**直接建立序列中任意两个位置的依赖关系，无论距离多远。** 对于交易序列，这意味着模型可以直接关注 30 天前的一笔异常转账，而不需要通过 30 个中间步骤传递信号。

---

## 2. Scaled Dot-Product Attention

### 2.1 数学公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^{T}}{\sqrt{d_{k}}}\right)V$$

其中：
- $Q \in \mathbb{R}^{n \times d_{k}}$：Query 矩阵（当前位置想"查找什么"）
- $K \in \mathbb{R}^{m \times d_{k}}$：Key 矩阵（每个位置"提供什么标签"）
- $V \in \mathbb{R}^{m \times d_{v}}$：Value 矩阵（每个位置"实际包含什么信息"）
- $n$：Query 序列长度；$m$：Key/Value 序列长度

### 2.2 语义直觉

把 attention 理解为一个软检索（soft retrieval）：
- **Query**：当前 token 的"搜索关键词"
- **Key**：其他 token 的"索引标签"
- $QK^{T}$：相似度矩阵，第 $(i,j)$ 个元素 = token $i$ 对 token $j$ 的"关注程度"
- $\text{softmax}$：归一化为概率分布（attention weights），所有权重之和 = 1
- **Value**：加权求和的"内容"，权重越高 = 从该位置获取的信息越多

### 2.3 为什么除以 $\sqrt{d_{k}}$

当 $d_{k}$ 较大时，点积 $QK^{T}$ 的方差 $\approx d_{k}$（假设 Q/K 各维度独立均值0方差1）。方差大 → 某些点积值远大于其他 → softmax 进入饱和区（梯度 ≈ 0）。

除以 $\sqrt{d_{k}}$ 将方差归一化为 1，保持梯度流动。

### 2.4 计算复杂度

- 时间复杂度：$O(n^{2} \cdot d)$（每对 token 都要计算相似度）
- 空间复杂度：$O(n^{2})$（attention 矩阵）
- **瓶颈**：序列长度 $n$ 较大时（>2048）内存爆炸。这是 Longformer/BigBird 等 efficient transformer 要解决的核心问题。

---

## 3. Multi-Head Attention (多头注意力)

### 3.1 公式

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_{1}, ..., \text{head}_{h})W^{O}$$

$$\text{head}_{i} = \text{Attention}(QW^{Q}_{i}, KW^{K}_{i}, VW^{V}_{i})$$

每个 head 有独立的投影矩阵 $W^{Q}_{i}, W^{K}_{i} \in \mathbb{R}^{d_{model} \times d_{k}}$，其中 $d_{k} = d_{model}/h$。

### 3.2 为什么多头

单头 attention 在每个位置只学到一种"关注方式"。多头允许模型在 **不同子空间** 中同时学习多种依赖关系：

| Head 类型（经验观察） | 学到的模式 |
|---------------------|-----------|
| 局部 head | 关注相邻位置（类似 CNN 的局部感受野） |
| 语法 head | 关注依存关系（主谓宾） |
| 共指 head | 关注代词与先行词 |
| 长距离 head | 关注远距离的语义相关 token |

对于欺诈检测：不同 head 可能分别捕捉"金额模式"、"时间间隔"、"对手方类型"等不同维度的关联。

---

## 4. 位置编码 (Positional Encoding)

Attention 本身是 **置换不变的**（permutation-invariant）——打乱输入顺序，输出不变。位置编码注入序列顺序信息。

### 4.1 正弦位置编码（原始论文）

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

- 不需要学习参数
- 可以外推到训练时未见过的序列长度
- 不同频率的正弦波编码不同粒度的位置信息

### 4.2 可学习位置编码（BERT/GPT 使用）

每个位置有一个可训练的 embedding 向量。优点：更灵活；缺点：无法外推到训练时的最大长度之外。

### 4.3 旋转位置编码 RoPE（LLaMA/GPT-NeoX 使用）

将位置信息编码为 Q 和 K 的旋转变换，使得 $Q_{m} \cdot K_{n}$ 自然包含相对位置 $m-n$ 的信息。

**优势**：天然支持相对位置、外推能力强、与 attention 计算自然结合。目前主流 LLM 的首选。

### 4.4 ALiBi（位置偏置，BLOOM 使用）

不修改 embedding，直接在 attention score 上减去与距离成正比的 penalty：$\text{score}_{ij} = Q_{i} \cdot K_{j} - m \cdot |i-j|$。训练推理长度泛化性好。

---

## 5. Transformer Block 组件详解

### 5.1 Layer Normalization

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sigma + \epsilon} \cdot \gamma + \beta$$

对每个 token 的 $d_{model}$ 维向量做归一化（而非 BatchNorm 的跨 batch 归一化），不受 batch size 影响。

**Pre-norm vs. Post-norm：**

| | Post-Norm（原始论文） | Pre-Norm（现代主流） |
|---|---|---|
| 位置 | Attention/FFN **之后** | Attention/FFN **之前** |
| 训练稳定性 | 需要 warmup，深层模型易不稳定 | 更稳定，可用更大学习率 |
| 性能 | 理论峰值更高 | 工程首选（GPT-2+, LLaMA） |

### 5.2 Feed-Forward Network (FFN)

$$\text{FFN}(x) = \max(0, xW_{1} + b_{1})W_{2} + b_{2}$$

- 两层线性变换 + ReLU（或 GeLU/SwiGLU）
- 中间维度 = $4 \times d_{model}$（原始论文经验值，被广泛沿用）
- FFN 对每个位置独立计算（position-wise），不做跨位置交互
- 作用：增加非线性，让模型可以表达复杂的特征变换

### 5.3 残差连接 (Residual Connections)

$$x' = x + \text{Sublayer}(x)$$

防止梯度消失，使深层网络可训练。对于 Pre-Norm：$x' = x + \text{Sublayer}(\text{LayerNorm}(x))$

---

## 6. Encoder vs. Decoder vs. Encoder-Decoder

| 架构 | 代表模型 | Self-Attention 类型 | 典型任务 |
|------|---------|-------------------|---------|
| Encoder-only | BERT, RoBERTa | 双向（全序列可见） | 分类、NER、语义匹配 |
| Decoder-only | GPT 系列 | 因果（只看左侧） | 生成、补全 |
| Encoder-Decoder | T5, BART | Encoder双向 + Decoder因果 + Cross-Attention | 翻译、摘要、Seq2Seq |

**欺诈检测中的选择：**
- **地址分类（是否欺诈）** → Encoder-only（BERT 风格），使用 `[CLS]` token 的 representation 做分类
- **生成式异常描述** → Encoder-Decoder 或 Decoder-only

### 6.1 Masked Self-Attention（因果 Attention）

Decoder 在生成位置 $t$ 时，只能看到位置 $1, 2, ..., t-1$ 的 token（避免"作弊"）。实现方式：在 attention score 矩阵上加一个上三角 $-\infty$ mask，softmax 后上三角变为 0。

---

## 7. 训练细节

### 7.1 Pre-training 目标

| 目标 | 说明 | 代表模型 |
|------|------|---------|
| MLM（Masked Language Model） | 随机 mask 15% token，预测被 mask 的词 | BERT |
| CLM（Causal Language Model） | 预测下一个 token | GPT |
| Span Corruption | mask 连续片段，生成被 mask 的片段 | T5 |
| RTD（Replaced Token Detection） | 判断每个 token 是否被替换 | ELECTRA |

**欺诈检测的 pre-training 类比：**
- 在大量未标注交易序列上做 MLM（mask 某些交易，让模型预测）→ 学习正常行为分布
- Fine-tune 阶段：少量标注欺诈样本做二分类

### 7.2 Fine-tuning

在 pre-trained 模型基础上加 task-specific head（如线性分类层），用标注数据继续训练。通常使用较小学习率（1e-5 ~ 5e-5），避免灾难性遗忘。

### 7.3 计算复杂度含义

序列长度 $n$ 翻倍 → 计算量 $4\times$，显存 $4\times$。实际约束：
- BERT-base：最大 512 tokens（后来的模型扩展到 4096/8192）
- 对于活跃地址（数千笔交易），需要滑动窗口或分层采样

---

## Interview Q&A

### Q1: 请解释 Scaled Dot-Product Attention 的完整计算过程，以及为什么要除以 $\sqrt{d_{k}}$。

**回答：**

1. **输入**：输入序列 $X \in \mathbb{R}^{n \times d_{model}}$，通过三个线性变换得到 $Q = XW^{Q}$，$K = XW^{K}$，$V = XW^{V}$
2. **相似度计算**：$S = QK^{T} \in \mathbb{R}^{n \times n}$，第 $(i,j)$ 个元素表示 token $i$ 对 token $j$ 的原始关注度
3. **缩放**：$S' = S / \sqrt{d_{k}}$，防止高维向量点积过大导致 softmax 饱和
4. **归一化**：$A = \text{softmax}(S') \in \mathbb{R}^{n \times n}$，每行求和为 1
5. **加权求和**：$\text{Output} = AV \in \mathbb{R}^{n \times d_{v}}$
6. **为什么除以 $\sqrt{d_{k}}$**：若 $Q, K$ 各元素独立 $\sim \mathcal{N}(0,1)$，则 $q \cdot k = \sum_{i=1}^{d_{k}} q_{i}k_{i}$ 的方差为 $d_{k}$。除以 $\sqrt{d_{k}}$ 后方差变为 1，softmax 输入在合理范围，梯度不会消失。

> **Follow-up 提示：** Self-attention 和 Cross-attention 的区别？（Self: Q/K/V 来自同一序列；Cross: Q 来自 decoder，K/V 来自 encoder output）

### Q2: Multi-Head Attention 中，多个 head 分别学到了什么？

**回答：**

1. **理论上**：每个 head 在不同的 $d_{k}$ 维子空间中独立学习 attention 模式，可以同时捕捉多种依赖关系。
2. **实验观察**（来自 Vig & Belinkov 等研究）：
   - 部分 head 专注于**局部依赖**（关注相邻 1-3 个 token）
   - 部分 head 专注于**句法结构**（主谓宾依存关系）
   - 部分 head 专注于**长距离语义关联**
3. **在交易序列中的类比**：不同 head 可能捕捉"金额规律"、"时间间隔模式"、"对手方类型切换"等不同维度的关联。
4. **Head pruning 实验**表明：相当一部分 head 是冗余的，可以剪枝而不影响性能（说明有效 head 数 < h）。

> **Follow-up 提示：** 为什么不用 h 个独立的 single-head attention 而要用 multi-head？（参数共享 + 最后的 $W^{O}$ 投影允许跨 head 的信息整合）

### Q3: Pre-Norm 和 Post-Norm 有什么区别？现在为什么主流用 Pre-Norm？

**回答：**

1. **Post-Norm（原始 Transformer）**：$x' = \text{LayerNorm}(x + \text{Sublayer}(x))$。归一化在残差之后，初始化时梯度通过残差路径 + sublayer 两条路径传递，深层容易不稳定，需要精心设计 warmup schedule。
2. **Pre-Norm（现代 LLM）**：$x' = x + \text{Sublayer}(\text{LayerNorm}(x))$。归一化在 sublayer 之前，梯度主要通过残差"高速公路"传递，sublayer 的梯度贡献更稳定。
3. **Pre-Norm 优势**：无需 warmup 也能稳定训练，可以使用更大学习率，扩展到更深的网络（100+ 层）不崩溃。
4. **Pre-Norm 代价**：理论表达能力略逊（实验上深度较浅时 Post-Norm 可能性能更高），但工程稳定性优势使其成为 GPT-2+、LLaMA 等的标准选择。

> **Follow-up 提示：** RMSNorm vs LayerNorm？（RMSNorm 省略了 mean-centering，只做 scale，计算更快，LLaMA 采用）

### Q4: Transformer 的时间/空间复杂度是多少？为什么对长序列是瓶颈？

**回答：**

1. **时间复杂度**：$O(n^{2} \cdot d)$，其中 $n$ 为序列长度，$d$ 为 $d_{model}$。来源：$QK^{T}$ 矩阵乘法需要 $O(n^{2} \cdot d_{k})$，$AV$ 需要 $O(n^{2} \cdot d_{v})$。
2. **空间复杂度**：$O(n^{2})$，attention matrix 需要存储 $n \times n$ 矩阵（所有 head）。
3. **实际瓶颈**：$n=512$ 时显存友好，$n=4096$ 时 attention matrix 大 64 倍，$n=32768$（某些长文本场景）时接近不可行。
4. **欺诈检测中的含义**：若将地址的完整交易历史（可能 10000+ 笔）作为序列，标准 Transformer 不可行，需要 Longformer（滑动窗口 attention）或 PatchTST（分 patch 建模）等变体。

> **Follow-up 提示：** Flash Attention 如何解决显存问题？（分块（tiling）计算 attention，避免实例化完整 $n \times n$ 矩阵，将显存从 $O(n^{2})$ 降为 $O(n)$）

### Q5: 什么是 Masked Language Modeling（MLM）？如何把它迁移到交易序列的预训练？

**回答：**

1. **标准 MLM（BERT）**：随机 mask 15% 的 token（80% 用 `[MASK]` 替换，10% 用随机 token，10% 保持不变），让模型预测原始 token。迫使模型学习双向上下文信息。
2. **交易序列迁移**：
   - 将每笔交易视为一个 token（由多个特征组成的 embedding）
   - 随机 mask 某些交易（例如 mask 掉金额或 mask 掉整笔交易）
   - 让模型预测被 mask 交易的关键属性（金额范围、是否合约调用、对手方类型）
3. **预训练优势**：用大量未标注正常交易学习"正常行为分布"，fine-tune 时只需少量标注欺诈样本，解决欺诈标签稀少的问题。
4. **实践注意**：交易 token 是连续/类别混合特征，不能直接用词汇表 softmax 预测，需要针对不同特征类型设计不同的预测 head。

> **Follow-up 提示：** CLM（自回归）预训练在欺诈检测中有用吗？（有，可以做异常检测——对历史序列的 next-transaction 预测困难度 = perplexity，超高 perplexity 的交易可能是异常）

### Q6: Encoder-only 和 Decoder-only 架构在欺诈分类任务上哪个更合适？

**回答：**

1. **Encoder-only（BERT 风格）更适合欺诈分类**：
   - 双向 attention 可以同时利用历史和未来交易的信息（全序列上下文）
   - `[CLS]` token 的 representation 汇聚整个序列信息，直接接分类头
   - 预训练目标（MLM）鼓励学习双向语义，更适合理解型任务
2. **Decoder-only 不太适合直接分类**：因果 attention 只看过去，无法利用"未来"交易信息来判断当前时刻是否可疑。但适合**实时检测场景**（只有历史信息）。
3. **实时 vs. 批量**：批量离线分析用 Encoder-only；实时流式评分用 Decoder-only 或 Encoder-only（截断到当前时刻）。

> **Follow-up 提示：** 什么是 Prefix LM？（Encoder 部分双向 attention，Decoder 部分因果 attention，兼顾两者，T5 的一种变体）

### Q7: 为什么 RoPE（旋转位置编码）优于原始正弦位置编码？

**回答：**

1. **原始正弦编码**：绝对位置编码，$PE_{pos}$ 与 token embedding 相加。两个位置的 attention score $Q_{m} \cdot K_{n}$ 无法直接表达相对距离 $m-n$。
2. **RoPE**：将位置信息编码为 Q 和 K 的旋转矩阵：$Q_{m} = R_{m}W^{Q}x_{m}$，使得 $Q_{m} \cdot K_{n} = f(x_{m}, x_{n}, m-n)$，即 attention score **自然包含相对位置信息**。
3. **优势**：
   - 只编码相对距离，泛化性更强
   - 可以通过 RoPE scaling 外推到训练时未见的更长序列
   - 与 attention 计算紧密结合，不占用 embedding 维度
4. **欺诈检测应用**：交易序列中，两笔交易的时间间隔（相对位置）比绝对时间戳更重要，RoPE 的相对位置建模更自然。

> **Follow-up 提示：** 实际上在交易序列中，位置应该用 token index 还是实际时间戳？（实际时间戳更好，可以将时间差 $\Delta t$ 直接编码为连续的位置信息）

### Q8: 请用一句话解释 FFN 层的作用，以及为什么中间维度是 $4 \times d_{model}$？

**回答：**

1. **FFN 的作用**：在每个位置独立地做非线性特征变换，Attention 负责跨位置信息聚合，FFN 负责对聚合后的信息做深度变换（可以理解为"记忆存储"）。
2. **为什么 $4\times$**：原始论文经验值。研究表明 FFN 权重存储了大量"事实知识"（factual knowledge），$4\times$ 扩展给了足够的容量。后续研究（如 Cramming）表明这个比例在 compute-optimal 角度不一定最优，但被沿用。
3. **GLU 变体**（SwiGLU，LLaMA 使用）：$\text{FFN}(x) = (\text{Swish}(xW_{1}) \odot xW_{3}) W_{2}$，引入门控机制，性能更好，中间维度通常用 $\frac{8}{3} d_{model}$。

> **Follow-up 提示：** Attention 和 FFN 各占 Transformer 参数量的多少？（各约 1/3，embedding 层占剩余 1/3；FFN 参数量 = $2 \times d_{model} \times 4d_{model}$ = $8d_{model}^{2}$，Attention = $4d_{model}^{2}$）
```

- [ ] **Step 4: Verify completeness**

Check against spec: RNN limitations ✓, attention math ✓, multi-head ✓, positional encoding (4 variants) ✓, full block components ✓, encoder/decoder/encoder-decoder ✓, training details ✓, 8 Q&A ✓

- [ ] **Step 5: Commit**

```bash
git add interview_question/okx_anti_fraud/02_transformer_architecture.md
git commit -m "feat: add transformer architecture deep dive for OKX anti-fraud interview"
```

---

## Task 3: Write Transformer Variants

**Files:**
- Create: `interview_question/okx_anti_fraud/03_transformer_variants.md`

- [ ] **Step 1: Research tabular and anomaly transformers**

Search for:
- "TabTransformer categorical feature embedding attention"
- "FTTransformer feature tokenizer transformer tabular"
- "Anomaly Transformer association discrepancy time series"
- "TranAD adversarial training transformer anomaly detection"
- "Graphormer graph transformer self-supervised"

Focus on: architecture differences from standard transformer, what problem each solves, fraud detection relevance.

- [ ] **Step 2: Research efficient and time-series transformers**

Search for:
- "Longformer sliding window attention long document"
- "Temporal Fusion Transformer TFT time series forecasting"
- "PatchTST patch time series transformer"
- "iTransformer inverted transformer time series"

- [ ] **Step 3: Write `03_transformer_variants.md`**

```markdown
# Transformer 变种全景 (Transformer Variants)

> OKX Anti-Fraud 面试准备 · File 3 of 4

---

## 1. BERT 家族

### 1.1 BERT（2018, Google）

**核心创新：** Bidirectional Encoder Representations from Transformers。MLM + NSP 双目标预训练，第一个证明预训练 Transformer + fine-tuning 在 NLP 各任务全面 SOTA。

| 配置 | BERT-base | BERT-large |
|------|-----------|------------|
| Layers | 12 | 24 |
| Hidden size | 768 | 1024 |
| Heads | 12 | 16 |
| Parameters | 110M | 340M |

**缺陷：** NSP 目标被后续研究证明没什么用；static masking（每个 epoch 相同 mask 模式）限制了学习效率。

### 1.2 RoBERTa（2019, Facebook）

**改进：**
- 去掉 NSP（只用 MLM）
- Dynamic masking（每次 forward 重新 mask）
- 更大 batch size、更长训练时间
- 更大词汇表（50K BPE vs BERT 的 30K）

**结果：** 同等参数量下显著优于 BERT，成为 NLP 分类任务的常用基线。

### 1.3 DeBERTa（2020, Microsoft）

**核心创新：** Disentangled Attention — 将内容和位置信息分离编码：

$$A_{i,j} = H_{i} \cdot H_{j}^{T} + H_{i} \cdot P_{i|j}^{T} + P_{j|i} \cdot H_{j}^{T}$$

- $H_{i}$：内容 embedding；$P_{i|j}$：从位置 $i$ 看位置 $j$ 的相对位置 embedding
- 使得 attention 同时考虑内容-内容、内容-位置、位置-内容三种交互

**额外创新：** EMD（Enhanced Mask Decoder）：在 MLM fine-tune 时恢复绝对位置信息。

**结果：** SuperGLUE 排行榜曾超越人类基线。

### 1.4 ALBERT（2019, Google）

**为了减少参数量：**
1. **Factorized embedding parameterization**：词汇表 embedding 维度（128）与 hidden size（768）解耦，词汇表矩阵 $V \times H$ 拆分为 $V \times E + E \times H$
2. **Cross-layer parameter sharing**：所有层共享参数（默认），参数量降低 ~18x（ALBERT-large vs BERT-large）
3. **SOP（Sentence Order Prediction）** 替换 NSP：判断两句是否顺序对调

**代价：** 参数共享导致推理时间不减少（层数不变），模型容量受限。

---

## 2. GPT 家族

| 模型 | 年份 | 参数量 | 关键特性 |
|------|------|--------|---------|
| GPT-1 | 2018 | 117M | 第一个证明生成式预训练 + 判别式 fine-tune |
| GPT-2 | 2019 | 1.5B | Zero-shot 生成能力，引发 AI safety 讨论 |
| GPT-3 | 2020 | 175B | In-context learning，few-shot 无需 fine-tune |
| GPT-4 | 2023 | ~1T（估计） | 多模态，RLHF 对齐 |

**GPT vs. BERT 使用场景：**

| 场景 | BERT 更好 | GPT 更好 |
|------|---------|---------|
| 文本分类 | ✓ | |
| NER/信息抽取 | ✓ | |
| 语义相似度 | ✓ | |
| 文本生成 | | ✓ |
| 指令遵循 | | ✓ |
| 欺诈分类（离线） | ✓ | |
| 欺诈描述生成/解释 | | ✓ |

---

## 3. Efficient Transformers（解决 $O(n^{2})$ 问题）

### 3.1 Longformer（2020, AI2）

**核心：** 将 global dense attention 替换为 sliding window attention + global attention：
- **Sliding window**：每个 token 只 attend 到 $\pm w$ 个邻近 token，复杂度 $O(n \cdot w)$
- **Global attention**：特定 token（如 `[CLS]`、问题中的关键词）attend 全局
- 复杂度：$O(n \cdot w)$，支持 4096+ tokens

**欺诈应用：** 长地址历史序列（数百笔交易）建模，局部 attention 捕捉短期模式，global `[CLS]` 聚合全局异常信号。

### 3.2 BigBird（2020, Google）

**三种 attention 混合：**
1. Random attention（随机关注若干 token）
2. Window attention（局部滑动窗口）
3. Global attention（特殊 token）

理论上可以近似任意 attention 矩阵，同时将复杂度降为 $O(n)$。

### 3.3 Performer（2020, Google）

**核心思路：** 用随机特征（Random Feature Map）近似 softmax attention：

$$\text{softmax}(QK^{T}/\sqrt{d}) \approx \phi(Q)\phi(K)^{T}$$

其中 $\phi$ 是一个随机特征函数，使得计算可以改写为先算 $\phi(K)^{T}V$，再乘 $\phi(Q)$，复杂度从 $O(n^{2}d)$ 降为 $O(nd^{2})$（当 $n \gg d$ 时大幅节省）。

---

## 4. 表格数据 Transformers（Tabular Transformers）

### 4.1 TabTransformer（2020, Amazon）

**问题：** 表格数据有类别特征（categorical）和连续特征（numerical），标准 Transformer 只处理序列 token。

**方案：**
- 类别特征 → 独立 embedding → 输入 Transformer encoder（捕捉特征间交互）
- 连续特征 → 直接 concatenate 到最终 representation（不过 Transformer）
- 最终：类别 Transformer output + 连续特征 → MLP 分类头

**欺诈应用：** 表格型特征（交易金额、时间、地区、设备类型等混合特征）的欺诈分类。

### 4.2 FTTransformer（2021, Yandex）

**改进 TabTransformer：**
- **连续特征也做 tokenization**：$x_{j} \to x_{j} \cdot w_{j} + b_{j}$，每个连续特征乘以可学习的向量并加偏置，投影到 $d$ 维空间
- 所有特征（类别 + 连续）都过 Transformer
- 加入 `[CLS]` token，用其 representation 做分类

**结果：** 在多个 tabular benchmark 上优于 TabTransformer 和经典 GBDT（但不总是）。

**关键结论：** FTTransformer 在特征间存在复杂交互时（如欺诈检测）优于 XGBoost；XGBoost 在特征间交互简单、数据量小时更优。

---

## 5. 时间序列 Transformers（Time Series Transformers）

### 5.1 Temporal Fusion Transformer — TFT（2019, Google）

**专为多变量时间序列预测设计：**
- **LSTM** 编码局部时序依赖
- **Multi-head Attention** 捕捉长距离依赖
- **Variable Selection Network**：自动学习每个特征的重要性（稀疏门控）
- **Gated Residual Networks**：自适应跳过不必要的非线性

**输出**：分位数预测（而非点估计），提供不确定性估计。

**欺诈应用：** 预测地址的未来交易行为分布，超出预测分位数范围的交易 = 潜在异常。

### 5.2 PatchTST（2023, MIT + IBM）

**核心思想：** 把时间序列分成固定大小的 patch（类似 ViT 对图像分 patch），每个 patch 作为一个 token。

**优势：**
- 将序列长度从 $n$ 降到 $n/p$（$p$ = patch size），大幅降低计算量
- 每个 patch 内的局部模式得到保留
- Channel-independent：每个特征维度独立建模，减少变量间干扰

**欺诈应用：** 将地址的小时级交易量序列分 patch，检测交易模式的阶段性变化。

### 5.3 iTransformer（2024, THUML）

**反直觉设计：** 把时间步（timestep）和变量（variable）的角色对调：
- 每个 **变量** 的完整时间序列 = 一个 token
- Attention 在**变量间**（而非时间步间）做
- FFN 在每个变量内部做时序建模

**结果：** 在多变量预测任务上超越 PatchTST，特别适合变量数量多、变量间关联重要的场景。

---

## 6. 异常检测 Transformers（Anomaly Detection Transformers）

### 6.1 Anomaly Transformer（2022, THUML）

**核心创新：** Association Discrepancy（关联差异）

**正常 vs. 异常的 attention 模式不同：**
- 正常 time point：attention 集中在**相邻时间步**（局部一致性）
- 异常 time point：因为异常难以与邻近点建立语义关联，attention 分散（高熵）

**两路 attention：**
1. **Prior-association**：Gaussian kernel 生成以对角线为中心的先验（强制局部聚焦）
2. **Series-association**：标准 self-attention（学习实际分布）

**异常分数** = KL 散度 $(P_{prior} || P_{series}) + KL(P_{series} || P_{prior})$：差异越大 = 越不符合正常模式 = 越可能是异常。

**欺诈应用：** 交易时间序列异常检测，不需要标注数据（无监督）。

### 6.2 TranAD（2022, Microsoft Research）

**结合 Transformer + 对抗训练（GAN 思想）：**
- 两个 Transformer：**Transformer 1** 做粗粒度重建，**Transformer 2** 做精细重建（带对抗 loss）
- **Self-conditioning**：Transformer 2 的输入包含 Transformer 1 的输出，实现二阶段细化
- 异常分数 = 重建误差（正常序列重建误差小，异常序列重建误差大）

**优势：** 对抗训练使模型对边界异常更敏感，训练更稳定（相比纯 autoencoder）。

---

## 7. Graph Transformers（图 + Transformer 混合）

### 7.1 Graphormer（2021, Microsoft）

**把图结构信息编码进标准 Transformer：**
- **Node feature** → token embedding
- **Centrality Encoding**：节点的 in/out degree 作为额外位置编码（$z_{i} = z_{deg^{-}(v_{i})} + z_{deg^{+}(v_{i})}$）
- **Spatial Encoding**：节点对之间的最短路径长度 $\phi(v_{i}, v_{j})$ 作为 attention bias（$A_{ij} = b_{\phi(v_{i},v_{j})}$）
- **Edge Encoding**：沿最短路径的 edge feature 均值作为额外 attention bias

**结果：** 在图级别预测任务（OGB-LSC）上显著优于纯 GNN。

**欺诈应用：** 将整个子图（如地址的 2-hop 邻域）输入 Graphormer，判断该子图是否是欺诈团伙。

### 7.2 Graph Transformer（GT）通用框架

**基本范式：**
$$h_{v}^{(l+1)} = \text{Transformer}\left(\left\{h_{u}^{(l)} : u \in \mathcal{N}(v) \cup \{v\}\right\}\right)$$

用 Transformer 的 attention 替代 GNN 的消息传递（message passing），节点只 attend 其邻居（图拓扑限制 attention 范围）。

**优势 vs. 纯 GNN：**
- 天然支持可变大小邻域的全局聚合
- 避免 GNN 的过平滑（over-smoothing）问题
- 可以加入 edge features

---

## 8. 变种对比总结

| 模型 | 解决的核心问题 | 欺诈检测适用场景 | 输入类型 |
|------|-------------|--------------|---------|
| BERT | 双向理解 | 地址行为序列分类 | Token 序列 |
| RoBERTa | 更稳健的 BERT | 同上，更稳定 | Token 序列 |
| DeBERTa | 内容/位置解耦 | 高精度分类 | Token 序列 |
| Longformer | 长序列 | 长交易历史（>512笔） | 长序列 |
| TabTransformer | 类别特征交互 | 表格型交易特征 | 表格 |
| FTTransformer | 所有特征统一 tokenize | 混合类型欺诈特征 | 表格 |
| TFT | 时序预测 + 不确定性 | 行为模式预测异常 | 多变量时序 |
| PatchTST | 高效时序建模 | 交易量时序 | 时间序列 |
| Anomaly Transformer | 无监督时序异常 | 无标签欺诈检测 | 时间序列 |
| TranAD | 对抗增强异常检测 | 边界异常检测 | 时间序列 |
| Graphormer | 图结构 + Transformer | 欺诈团伙图分析 | 图 |

---

## Interview Q&A

### Q1: BERT、RoBERTa、DeBERTa 三者的主要区别是什么？做欺诈分类你会选哪个？

**回答：**

1. **BERT**：MLM + NSP 预训练，static masking，奠基之作但存在设计缺陷。
2. **RoBERTa**：去掉 NSP，动态 masking，更多训练数据/步骤，同参数量下优于 BERT。
3. **DeBERTa**：Disentangled Attention 分离内容和相对位置编码，在 GLUE/SuperGLUE 上超越 RoBERTa。
4. **欺诈分类选择**：生产环境首选 **RoBERTa**（性能好、社区成熟、Hugging Face 支持完善）；追求 SOTA 准确率选 **DeBERTa-v3**；资源受限选 **BERT-base**（110M 参数，推理快）。

> **Follow-up 提示：** ALBERT 为什么参数少但推理不一定快？（Cross-layer 参数共享 → 参数量减少，但层数不变 → FLOPs 不减）

### Q2: FTTransformer 和 XGBoost 各在什么场景下更好？

**回答：**

1. **FTTransformer 更好**：特征间存在复杂高阶交互、数据量大（>50K 样本）、有许多类别特征（高基数）时。
2. **XGBoost 更好**：数据量较小（<10K）、特征间交互相对简单、需要快速迭代（训练调参时间短）、需要强可解释性（SHAP 值成熟）时。
3. **欺诈检测实践**：通常先用 XGBoost 建立基线，如果特征工程已经充分但瓶颈明显，再尝试 FTTransformer。在有大量历史数据的成熟欺诈系统中，FTTransformer 有潜力超越 XGBoost。
4. **混合方案**：GNN 提取图特征 → 与表格特征 concatenate → FTTransformer（目前较前沿的做法）。

> **Follow-up 提示：** FTTransformer 如何处理缺失值？（缺失值对应的 feature token 可以用特殊 embedding 或 mask，Transformer 通过 attention 从其他特征推断）

### Q3: Anomaly Transformer 的 Association Discrepancy 是什么？为什么有效？

**回答：**

1. **核心假设**：正常时间点与其**相邻时间点**天然有较强关联（时间序列的局部一致性）；异常时间点因为"异常"，无法与相邻点建立正常的关联，其 attention 会分散到整个序列。
2. **两路 attention 设计**：
   - Prior-association：用 Gaussian kernel 生成对角线集中的先验（强制局部）
   - Series-association：学习到的实际 attention 分布
3. **Association Discrepancy**：正常点的 prior ≈ series（都局部集中），差异小；异常点的 series 分散（无法局部集中），与局部 prior 差异大。
4. **无监督**：用 minimax 训练——最大化正常点的 discrepancy（让模型尽量分开），最小化重建损失，不需要标签。

> **Follow-up 提示：** 与传统 autoencoder 异常检测相比优势是什么？（传统 AE 只用重建误差，容易把简单异常重建好；Association Discrepancy 从 attention 模式角度检测，更适合细微但持续的异常）

### Q4: 为什么 Graphormer 在图级别预测上优于 GNN？

**回答：**

1. **GNN 的局限**：
   - 消息传递聚合容易导致**过平滑**（多跳后所有节点 embedding 趋同）
   - 1-WL 测试表明 GNN 区分图结构的能力有理论上限
   - 全局结构信息（如中心性、长距离依赖）难以在有限层数内聚合
2. **Graphormer 的优势**：
   - Centrality Encoding 直接编码每个节点的全局重要性（degree）
   - Spatial Encoding 通过最短路径长度，让任意两节点都能在一层内建立关联（突破 GNN 的层数限制）
   - Attention 本质上是全局的，没有过平滑问题
3. **代价**：计算复杂度 $O(|V|^{2})$（所有节点对），大图不可扩展。实际应用需要对子图采样。

> **Follow-up 提示：** 如何把 Graphormer 用于欺诈团伙检测？（1. 提取可疑地址的 k-hop 子图；2. 整个子图输入 Graphormer；3. 图级别 representation 做二分类——是否是欺诈团伙子图）

### Q5: 对于一个新地址（交易历史很短），如何处理序列太短的问题？

**回答：**

1. **Padding + Masking**：将序列 pad 到固定长度，加 padding mask 告诉模型忽略 pad 位置。
2. **冷启动特征增强**：对短序列地址，更多依赖图特征（邻居行为）和创建时特征（地址生成时间、初始资金来源）。
3. **元学习（Few-Shot）**：用 Prototypical Network 或 MAML 在少量样本上快速适应。
4. **预训练对齐**：预训练阶段包含各种长度的序列（包括短序列），让模型学会"只有 3 笔交易时如何判断"。
5. **风险分层**：新地址默认进入"观察期"规则引擎（传统特征 + 规则）而非 Transformer 模型，等积累足够历史后切换。

> **Follow-up 提示：** 能用 Transformer 做 zero-shot 欺诈检测吗？（可以，通过 in-context learning：给 GPT-style 模型几个欺诈 examples + 待判断地址的交易历史，让模型判断；局限是 context window 有限）

### Q6: Longformer 的 sliding window attention 如何同时保留全局信息？

**回答：**

1. **Sliding window**：每个 token 只 attend $\pm w$ 范围内的邻居，时间复杂度 $O(n \cdot w)$。局部模式（短期交易规律）被很好地捕捉。
2. **Global attention token**：指定特定 token（如 `[CLS]`）做 **全局 attention**——这些 token 可以 attend 序列中所有 token，其他所有 token 也可以 attend 这些全局 token。
3. **信息流**：局部信息通过多层 sliding window 逐渐传播（每层感受野扩展 $2w$），$L$ 层后感受野 = $2wL$。全局 token 在第一层就能聚合全局信息。
4. **欺诈实践**：`[CLS]` 做全局 token，加入一些"标志性时间点"（如异常大额交易）也做全局 token，保证 Transformer 能直接关注关键事件。

> **Follow-up 提示：** BigBird 和 Longformer 的主要区别？（BigBird 多了 random attention，理论上完备性更强；Longformer 实现更简单，实践中效果相当）

### Q7: 在真实欺诈检测系统中，你会选择哪种 Transformer 变种？说明理由。

**回答：**

针对 OKX Anti-Fraud 的具体场景，我会根据子任务选择不同变种：

1. **链上地址行为分类（离线批量）**：**FTTransformer** 处理表格型特征 + **RoBERTa-style encoder** 处理交易序列，两路 embedding 融合后分类。表格特征处理成熟，序列特征捕捉行为模式。
2. **欺诈团伙识别**：**Graphormer** 或 **Graph Transformer**，将地址关系图作为输入，利用图结构识别洗钱环路、团伙集群。
3. **实时流式检测（毫秒级）**：**轻量 Transformer**（BERT-tiny 或蒸馏版）+ 规则引擎 ensemble，保证推理延迟 <10ms。
4. **无监督/半监督（标签稀少）**：**Anomaly Transformer** 或 **TranAD**，在大量无标注正常交易上预训练，无需标注即可给出异常分数。

> **Follow-up 提示：** 如何将序列模型和图模型的结果融合？（早期融合：GNN embedding concatenate 到 Transformer 输入；晚期融合：分别得到分数后 ensemble；中间融合：用 Cross-attention 让序列 query 图结构 key/value）

### Q8: PatchTST 和 iTransformer 的本质区别是什么？

**回答：**

1. **PatchTST**：把时间维度分组（patch），每个 patch 是一个 token，Attention 在 **时间 patches 之间** 做。每个变量（特征维度）独立建模（channel-independent）。解决了 token 数量过多的问题，保留了局部时序模式。
2. **iTransformer**：倒置设计——每个**变量的完整时序**是一个 token，Attention 在 **变量之间** 做（学变量间的关联），FFN 在每个变量内部学时序模式（position-wise，对时间步做非线性变换）。
3. **关键区别**：
   - PatchTST：建模**时间维度的长距离依赖**，变量独立
   - iTransformer：建模**变量间的相关性**，时序由 FFN 建模
4. **选择依据**：欺诈检测中如果不同特征维度（如转账金额、gas费、频率）之间的联动关系更重要，iTransformer 更合适；如果关心单一特征的时序演变，PatchTST 更合适。

> **Follow-up 提示：** 为什么 channel-independent（PatchTST）有时优于 channel-mixing？（多变量时序中，变量间的虚假相关（spurious correlation）会引入噪声，独立建模反而更鲁棒）
```

- [ ] **Step 4: Verify completeness**

Check against spec: BERT family ✓, GPT family ✓, efficient transformers ✓, tabular (TabTransformer, FTTransformer) ✓, time-series (TFT, PatchTST, iTransformer) ✓, anomaly (Anomaly Transformer, TranAD) ✓, graph+transformer (Graphormer, GT) ✓, comparison table ✓, 8 Q&A ✓

- [ ] **Step 5: Commit**

```bash
git add interview_question/okx_anti_fraud/03_transformer_variants.md
git commit -m "feat: add transformer variants overview for OKX anti-fraud interview"
```

---

## Task 4: Write Fraud Detection with Transformers

**Files:**
- Create: `interview_question/okx_anti_fraud/04_fraud_detection_with_transformers.md`

- [ ] **Step 1: Research fraud detection system design**

Search for:
- "transformer fraud detection blockchain address classification"
- "GNN transformer hybrid fraud detection cryptocurrency"
- "concept drift fraud detection model retraining"
- "attention interpretability fraud explanation"
- "DeFi fraud detection flash loan rug pull on-chain ML"

Focus on: production challenges, real papers/systems, OKX-specific DeFi fraud patterns.

- [ ] **Step 2: Research class imbalance and label scarcity solutions**

Search for:
- "imbalanced learning fraud detection SMOTE focal loss"
- "semi-supervised anomaly detection transformer few-shot fraud"

- [ ] **Step 3: Write `04_fraud_detection_with_transformers.md`**

```markdown
# Transformer + 区块链欺诈检测系统设计 (Transformer-Based Fraud Detection)

> OKX Anti-Fraud 面试准备 · File 4 of 4

---

## 1. 问题定义 (Problem Framing)

### 1.1 欺诈检测 = 异常检测 + 分类的混合问题

| 维度 | 挑战 | 影响 |
|------|------|------|
| 标签稀缺 | 欺诈地址占比 0.1%-1%，标注成本高 | 监督学习效果受限，需要半监督/无监督 |
| 类别极度不平衡 | 正负样本比可达 1:1000 | 模型倾向预测全部为正常；需要 focal loss / 重采样 |
| 对抗性漂移 | 欺诈者主动规避检测，攻击模式持续进化 | 模型性能随时间衰减，需要持续更新 |
| 标签噪声 | 已知欺诈地址不代表所有欺诈地址；存在"未被发现的欺诈" | 训练数据不完整，召回率天花板有限 |
| 实时约束 | 部分场景需要毫秒级判断（交易拦截） | 模型大小 ↔ 推理延迟 trade-off |

### 1.2 任务分层

```
Level 1: 交易级别（Transaction-level）
  → 单笔交易是否异常？
  → 特征：单笔金额、gas、合约调用类型
  → 模型：规则引擎 + 轻量 XGBoost/DNN
  
Level 2: 地址级别（Address-level）
  → 该地址是否是欺诈地址？
  → 特征：历史交易序列、统计聚合特征
  → 模型：Transformer 序列模型 + FTTransformer
  
Level 3: 团伙级别（Cluster-level）
  → 一组地址是否构成欺诈团伙？
  → 特征：交易图结构、资金流向
  → 模型：GNN + Graph Transformer
```

---

## 2. 特征工程 (Feature Engineering from Blockchain Data)

### 2.1 地址级聚合特征（Address-Level Features）

| 特征类别 | 特征 | 欺诈信号 |
|---------|------|---------|
| 交易量 | `tx_count_1d/7d/30d`，`volume_in/out_7d` | 突发高频交易 |
| 时间 | `account_age_days`，`active_hour_entropy`，`avg_tx_interval` | 新账户 + 夜间活跃 |
| 金额分布 | `amount_mean/std`，`round_number_ratio`，`max_single_tx_ratio` | 大量整数金额（洗钱）|
| 对手方 | `unique_counterparty_count`，`new_counterparty_ratio`，`known_fraud_neighbor_count` | 接触已知欺诈地址 |
| 合约交互 | `defi_protocol_diversity`，`contract_call_ratio`，`flash_loan_count` | 高频合约操作 |
| 图特征 | `in_degree`，`out_degree`，`pagerank`，`clustering_coef`，`2hop_fraud_ratio` | 处于密集欺诈子图中 |

### 2.2 序列特征（Per-Transaction Token Features）

每笔交易作为序列中的一个 token，包含：

```python
transaction_token = {
    "log_amount": log(amount + 1),           # 连续，归一化
    "direction": 0 or 1,                      # 类别 (in/out)
    "counterparty_type": embedding,           # 类别 (EOA/contract/exchange/...)
    "hour_of_day": sin/cos encoding,          # 周期性时间特征
    "days_since_account_creation": float,     # 连续
    "log_gas_fee": log(gas_fee + 1),         # 连续
    "is_contract_call": bool,                 # 类别
    "token_type": embedding,                  # 类别 (ETH/ERC20/ERC721)
    "delta_t_since_last_tx": float,           # 时间间隔
}
```

所有连续特征：log 归一化 → linear projection 到 $d_{model}$ 维
所有类别特征：embedding lookup → $d_{model}$ 维
最终 token embedding = 所有特征 embedding 求和或 concatenate 后投影

---

## 3. 建模方案一：序列 Transformer（地址行为建模）

### 3.1 架构

```
[CLS] tx_1 tx_2 ... tx_N
  ↓
BERT-style Transformer Encoder (L=6, H=256, A=8)
  ↓
[CLS] output → Linear(256 → 2) → Sigmoid
  ↓
P(fraud)
```

### 3.2 预训练阶段（用无标注数据）

**目标：** MLM（Masked Transaction Modeling）

- 随机 mask 15% 的交易 token
- 预测被 mask 交易的关键属性（金额分桶、是否合约调用）
- 用 **数百万个正常地址** 的交易序列训练
- 模型学习"正常地址的行为分布"

**为什么重要：** 欺诈标签极少，但无标注数据大量存在；预训练大幅降低下游 fine-tune 所需标注量。

### 3.3 Fine-tuning 阶段

- 加载预训练权重
- 在 `[CLS]` token 输出上接分类头
- 用标注欺诈/正常地址数据训练
- Loss = Focal Loss（处理类别不平衡）：$FL(p_{t}) = -\alpha_{t}(1-p_{t})^{\gamma}\log(p_{t})$，$\gamma=2$ 降低简单样本的权重

### 3.4 关键实现细节

| 问题 | 解决方案 |
|------|---------|
| 序列太长（>512笔交易） | 取最近 512 笔，或用 Longformer |
| 序列太短（新地址） | Padding + mask，或回退到规则引擎 |
| 时间信息 | 用交易时间差 $\Delta t$ 作为 token 间距，而非 token index |
| 实时推理 | 量化（INT8）+ ONNX 导出，延迟 <20ms |

---

## 4. 建模方案二：表格 Transformer（混合特征分类）

### 4.1 适用场景

当地址的交易历史序列特征和聚合统计特征都重要时，FTTransformer 统一建模所有特征。

### 4.2 架构

```
[address_age, tx_count_7d, volume_30d, ...]   # 连续特征
[country_code, device_type, wallet_type, ...]  # 类别特征
        ↓ Feature Tokenizer
[t_1] [t_2] [t_3] ... [t_F] [CLS]             # F个特征 token + CLS
        ↓ Transformer Encoder
   [CLS] representation → Linear → P(fraud)
```

### 4.3 与 XGBoost 的组合策略

实际生产中的分层策略：
1. **Level 1 快速过滤**：XGBoost（<1ms）过滤明显正常地址（置信度高的 negative）
2. **Level 2 精细判断**：FTTransformer（~5ms）对可疑地址做精细评分
3. **Level 3 专家审查**：高风险地址进入人工审查队列

---

## 5. 建模方案三：Graph + Transformer（团伙检测）

### 5.1 动机

单地址级别的模型无法发现：
- 洗钱链路（A→B→C→D→E，每个节点单独看都正常）
- 欺诈团伙（20个地址互相转账，图密度异常高）
- 混币器（一个中心节点接收多人资金再拆分）

### 5.2 架构：GNN + Transformer 双塔

```
Stage 1: GNN（邻域特征提取）
  - 输入：以目标地址为中心的 2-hop 子图
  - 模型：GraphSAGE 或 GAT（2-3层）
  - 输出：目标地址的图上下文 embedding h_graph

Stage 2: Transformer（序列特征提取）
  - 输入：目标地址的交易序列
  - 模型：BERT-style encoder（6层）
  - 输出：[CLS] token embedding h_seq

Stage 3: Fusion（融合）
  - h_fusion = MLP([h_graph; h_seq])
  - 或 Cross-Attention: h_seq 作为 Query，h_graph 作为 Key/Value
  - 输出：P(fraud)
```

### 5.3 标签传播（Label Propagation）

利用图结构扩展标签：
- 已确认欺诈地址的一阶邻居的欺诈概率升高（先验）
- 用 GNN 从已知欺诈节点向外传播风险分数
- 传播结果作为额外特征输入 Transformer

---

## 6. 建模方案四：无监督异常检测

### 6.1 适用场景

- 新型欺诈手法，没有历史标注（zero-day fraud）
- 标注成本极高、标注不及时
- 需要覆盖未知未知（unknown unknowns）

### 6.2 Anomaly Transformer 流程

```
大量无标注正常交易序列
        ↓ 无监督训练
Anomaly Transformer（学习正常 attention 模式）
        ↓
对每笔交易/每段序列计算 Association Discrepancy Score
        ↓
高分 → 异常候选 → 规则过滤 → 人工复核 → 更新标注数据
```

### 6.3 Autoencoder 基线对比

| 方法 | 异常分数来源 | 优势 | 劣势 |
|------|-----------|------|------|
| Transformer Autoencoder | 重建误差 | 简单直观 | 对"容易重建的异常"失效 |
| Anomaly Transformer | Association Discrepancy | 从 attention 模式角度，更鲁棒 | 超参数调整较复杂 |
| TranAD | 对抗重建误差 | 对边界异常敏感 | 训练不稳定性 |
| Isolation Forest | 路径长度 | 无需时序，计算快 | 忽略时序结构 |

---

## 7. 生产部署挑战 (Production Concerns)

### 7.1 概念漂移（Concept Drift）

欺诈者主动适应检测系统，模型性能随时间下降。

**检测漂移：**
- 监控线上样本的特征分布（PSI / KL 散度）
- 监控模型输出分布（score distribution shift）
- 监控误报/漏报率（需要人工复核结果）

**应对漂移：**
- **定期全量重训练**：每周/月全量更新（成本高）
- **在线增量学习**：每天用新确认标签做 fine-tune（BERT fine-tune 成本可接受）
- **对抗训练**：在训练时加入 adversarial examples（对已知规避手法的数据增强）

### 7.2 可解释性（Explainability）

监管合规和运营团队需要理解为什么一个地址被判断为欺诈。

**Transformer 的解释工具：**
- **Attention weights**：可视化哪些历史交易对最终判断影响最大（注意：attention ≠ importance，需要谨慎解读）
- **Integrated Gradients**：计算每个 token/特征对输出的梯度贡献，更可靠的 attribution
- **SHAP（KernelSHAP）**：模型无关，可用于任何 Transformer；计算慢
- **注意力探针**：训练一个 probing classifier 在特定 attention head 的输出上，解释该 head 学到了什么

**实践：** 对运营人员展示"最高 attention 的 Top-5 历史交易 + 对应异常原因"，结合规则生成自然语言解释（LangGraph agent）。

### 7.3 延迟约束（Latency）

| 场景 | 延迟要求 | 推荐方案 |
|------|---------|---------|
| 实时交易拦截 | <10ms | 规则引擎 + 蒸馏小 Transformer（2层） |
| 充值/提现风控 | <100ms | INT8 量化 BERT-tiny + ONNX |
| 每日批量扫描 | 无严格要求 | 完整 Transformer + GNN ensemble |

**模型压缩：**
- **知识蒸馏**：用大 BERT（teacher）蒸馏小 BERT-4L/2L（student），保留 90%+ 性能
- **量化（INT8）**：Hugging Face Optimum + ONNX Runtime，推理速度 ~3x
- **剪枝（Attention Head Pruning）**：移除冗余 head（通常 30-50% head 可以剪掉而不影响性能）

### 7.4 类别不平衡

| 方法 | 适用场景 | 注意事项 |
|------|---------|---------|
| Focal Loss | 训练时动态降低简单样本权重 | $\gamma$ 超参敏感 |
| SMOTE | 过采样少数类（合成样本） | 对高维特征效果差 |
| 类别权重 | `class_weight='balanced'` | 简单有效，首选 |
| 阈值调整 | 调整分类阈值（默认0.5可能不最优） | 用 PR curve 选最优阈值 |
| 无监督预训练 | 用大量正常样本预训练，再少量欺诈 fine-tune | 本质上解决标签稀缺问题 |

---

## 8. OKX 特定场景 (OKX-Specific Patterns)

### 8.1 DeFi 欺诈的特殊性

与传统金融欺诈相比，DeFi 欺诈有以下特点：
- **原子性**：一次攻击（flash loan）在单笔交易内完成，传统序列模型可能只看到一个异常点
- **智能合约逻辑**：需要理解合约 ABI、事件 log，才能理解资金流向
- **跨链**：资金通过 bridge 跨链后，链上数据断层（需要跨链数据关联）
- **MEV（矿工可提取价值）**：sandwich attack、front-running 在技术上利用区块链机制，监管灰色地带

### 8.2 OKX 交易所场景的额外数据

作为中心化交易所，OKX 拥有纯链上系统没有的数据：
- **KYC 信息**：实名认证地址（提高图数据关联精度）
- **充提记录**：链上地址与交易所账户的映射（关键！）
- **IP/设备指纹**：跨账户关联
- **交易行为**（现货/合约/期权）：与链上行为结合判断

**核心优势：** 可以把链上地址映射到实名账户，再把账户的链下行为（登录时间、设备、操作序列）与链上行为融合，大幅提升检测精度。

### 8.3 LangGraph 在欺诈检测中的应用

JD 特别提到 LangGraph agent。可能的应用场景：

```
触发条件：高风险地址被 Transformer 模型标记
    ↓
LangGraph Agent 启动
    ├── Sub-agent 1: 查询链上图数据（Neo4j Cypher）→ 资金来源链路
    ├── Sub-agent 2: 查询 OSINT 数据库 → 地址是否在黑名单
    ├── Sub-agent 3: 分析交易序列 → 生成行为摘要
    └── Sub-agent 4: 查询合规规则库 → 适用的监管框架
    ↓
Synthesizer: 综合所有信息，生成风险报告
    ↓
Human Review: 运营人员确认/驳回
```

这与你在 AML Investigation Mate 项目中的经验高度重合，可以作为面试中的项目案例。

---

## Interview Q&A

### Q1: 请设计一个基于 Transformer 的端到端区块链欺诈检测系统。

**回答：**

**系统分三层：**

**Layer 1 — 实时规则引擎（<5ms）：**
- 基于已知欺诈地址黑名单、简单阈值规则（单笔金额 > X）、IP 封禁
- 拦截明显欺诈，降低下层系统压力

**Layer 2 — ML 评分引擎（<50ms）：**
- **序列模型**：BERT-style Transformer 处理地址最近 256 笔交易序列，输出欺诈概率
- **表格模型**：FTTransformer 处理地址聚合特征（过去 7/30/90 天统计量）
- **Ensemble**：两个模型分数加权融合，超过阈值进入 Layer 3

**Layer 3 — 深度分析引擎（异步，秒级）：**
- **GNN + Transformer**：构建 2-hop 子图，检测欺诈团伙
- **Anomaly Transformer**：对高风险地址做无监督异常分析
- **LangGraph Agent**：自动化调查 + 生成风险报告

**训练策略：**
- 大量无标注正常交易 → Transformer MLM 预训练
- 少量标注欺诈数据 → Fine-tune（Focal Loss）
- 每周用新确认标注增量更新

> **Follow-up 提示：** 如何评估这个系统的效果？（用历史标注数据：Precision/Recall/F1；但更重要的是 business metrics：人工复核确认率（Precision@K）和欺诈金额拦截率（Recall × avg_fraud_amount）；PR curve 选最优阈值）

### Q2: 如何处理欺诈检测中极度不平衡的标签（正负比 1:1000）？

**回答：**

1. **Focal Loss**（首选）：$FL(p_{t}) = -(1-p_{t})^{\gamma} \log p_{t}$，$\gamma=2$ 时简单样本（高置信度正常）的 loss 被压缩 ~99%，模型集中学习难分样本（欺诈边界案例）。
2. **类别权重**：在 cross-entropy loss 中对欺诈样本赋权 1000，正常样本赋权 1。简单有效，作为基线。
3. **不用 SMOTE**：高维 embedding 空间中合成样本质量差，实验通常不如上面两种。
4. **阈值调整**：不使用默认 0.5 阈值，在 validation set 的 PR curve 上选择 F1 最优或业务指标最优的阈值（通常 0.1-0.3）。
5. **无监督预训练**（根本解法）：标签不平衡是因为有标注欺诈样本少，不是因为欺诈行为少。大规模预训练让模型先学好"正常是什么样"，再用少量标注样本 fine-tune，本质上绕开了不平衡问题。

> **Follow-up 提示：** Precision 和 Recall 在欺诈检测中哪个更重要？（取决于业务：实时拦截（会冻结用户资金）要求高 Precision（少冻结无辜用户）；事后审计要求高 Recall（不漏过欺诈）；通常先定 Precision 下限，再最大化 Recall）

### Q3: 如何应对欺诈者的对抗性规避（Adversarial Evasion）？

**回答：**

1. **欺诈者的规避策略**：
   - 拆分大额为小额（structuring，反过来符合正常行为特征）
   - 插入正常交易"洗白"历史
   - 使用全新地址（无历史）逃避地址级别检测
   - 模拟正常地址的行为模式（频率、金额分布）
2. **应对方案**：
   - **对抗训练**：将已知规避手法生成对抗样本加入训练集，增强模型鲁棒性
   - **图特征补充**：即使行为特征被伪造，图结构（资金来源、圈子）很难同时伪造
   - **跨层特征融合**：链上行为 + 链下行为（KYC/IP）多维度融合，提高规避成本
   - **概念漂移监控**：检测到新的规避模式后，快速生成标注数据并增量更新模型
   - **集成多个模型**：规避一个模型但很难同时规避多个不同架构的模型

> **Follow-up 提示：** 什么是 Adversarial Examples 在表格数据中的含义？（修改连续特征的小幅扰动，使模型预测翻转；用 PGD/FGSM 对表格特征生成对抗样本）

### Q4: 如何在欺诈检测中使用 Transformer 的 Attention 做可解释性分析？

**回答：**

1. **Attention Visualization**：
   - 提取特定 attention head 的权重矩阵，可视化每笔交易对 `[CLS]` 最终判断的贡献
   - 例：模型判断一个地址为欺诈，attention 权重最高的 3 笔交易 = 模型认为最可疑的交易
   - **局限**：attention weight ≠ feature importance（Jain & Wallace 2019 证明 attention 的解释性有限）

2. **Integrated Gradients（更可靠）**：
   - 计算输出对每个输入 token 每个维度的梯度，沿直线路径积分
   - $\text{IG}_{i}(x) = (x_{i} - x'_{i}) \times \int_{0}^{1} \frac{\partial F(x' + \alpha(x-x'))}{\partial x_{i}} d\alpha$
   - 结果：每个 token 的重要性分数，可以进一步分解到哪个特征（金额/时间/类型）

3. **实际系统中的应用**：
   - 生成自然语言解释："该地址在过去 7 天内有 3 笔可疑的整数金额转账（分别于 D-3/D-5/D-7），且接收方为已知混币器地址，综合风险评分 0.92"
   - 结合 LangGraph agent，Transformer 的 attribution 作为 agent 的分析输入

> **Follow-up 提示：** 如果运营团队说"attention 权重最高的那笔交易根本不可疑，但你们判断欺诈了"，说明什么问题？（attention ≠ causal contribution；需要用 causal probing 或 counterfactual 分析，移除那笔交易看预测是否改变）

### Q5: 你会如何在有限标注数据下构建欺诈检测模型？（Few-shot 场景）

**回答：**

**分阶段策略：**

**阶段 1（标注极少，<100 个欺诈样本）：**
- **无监督**：Anomaly Transformer / Isolation Forest 给所有地址打分
- **规则引擎**：基于领域知识的确定性规则（已知黑名单地址、极端金额阈值）
- 目标：尽量 high recall，接受 low precision，靠人工复核提高精度

**阶段 2（标注适中，100-10K 欺诈样本）：**
- **预训练 + 少量 fine-tune**：在大量无标注正常交易上做 MLM 预训练，再用有限标注 fine-tune
- **标签传播**：从已知欺诈节点向图中传播标签（得到软标签）
- **主动学习**：模型选择最不确定（entropy 最高）的样本送人工标注，最大化标注效率

**阶段 3（标注充足，>10K 欺诈样本）：**
- 全监督 Transformer + GNN ensemble
- 重点转向对抗鲁棒性和概念漂移

> **Follow-up 提示：** 什么是 Curriculum Learning？在欺诈检测中如何用？（先用简单样本（明显欺诈/明显正常）训练，逐渐引入难样本（边界案例），让模型稳定收敛后再处理困难样本）

### Q6: 请比较 GNN 和 Graph Transformer 在欺诈团伙检测中的优劣。

**回答：**

| 维度 | GNN (GCN/GAT/GraphSAGE) | Graph Transformer (Graphormer) |
|------|------------------------|-------------------------------|
| 信息传播 | 局部消息传递（k跳邻域） | 全局 attention（所有节点对） |
| 长距离关系 | 需要多层叠加，有过平滑风险 | 一层内即可建立远距离依赖 |
| 计算复杂度 | $O(|E| \cdot d)$，稀疏图高效 | $O(|V|^{2} \cdot d)$，大图受限 |
| 可扩展性 | 亿级节点可扩展（GraphSAGE 采样） | 需要子图采样，通常 <1K 节点 |
| 欺诈团伙（大图） | ✓（GraphSAGE 全图推理） | ✗（需要子图，信息不完整）|
| 欺诈团伙（小图分析） | 一般 | ✓（捕捉全局图结构特征）|
| 实际方案 | GNN 全图打分 → 高风险子图 → Graph Transformer 精细分析 | 通常作为 GNN 后的第二阶段 |

**推荐架构**：GNN（如 GraphSAGE）做全图地址级别风险评分，对高风险地址构建局部子图（2-hop），用 Graphormer 做图级别团伙判断。两阶段结合效率和精度。

> **Follow-up 提示：** 图上的过平滑（over-smoothing）是什么？如何缓解？（多层 GNN 后所有节点 embedding 趋于相似，无法区分；缓解：residual connections, PairNorm, DropEdge, 限制层数 2-3层）

### Q7: OKX 的 JD 提到 LangGraph——你会如何把它整合进欺诈检测 pipeline？

**回答：**

LangGraph 最适合**欺诈调查自动化**（而非欺诈检测本身）：

**触发条件：** Transformer 模型输出高风险分数（如 >0.85）的地址

**LangGraph 工作流：**

```python
# 伪代码描述 agent 结构
graph = StateGraph(FraudInvestigationState)

graph.add_node("fetch_onchain_data", fetch_tx_history_and_graph)
graph.add_node("query_blacklist", check_known_fraud_databases)
graph.add_node("analyze_fund_flow", run_cypher_on_neo4j)
graph.add_node("assess_defi_patterns", check_flash_loan_rug_pull)
graph.add_node("generate_report", synthesize_and_write_report)
graph.add_node("human_review", wait_for_analyst_decision)

graph.add_edge("fetch_onchain_data", "query_blacklist")
graph.add_edge("fetch_onchain_data", "analyze_fund_flow")
graph.add_edge("fetch_onchain_data", "assess_defi_patterns")
graph.add_edge(["query_blacklist", "analyze_fund_flow", 
                "assess_defi_patterns"], "generate_report")
graph.add_edge("generate_report", "human_review")
```

**关键价值：**
- 分析人员从"手动查 10 个数据源"变为"review agent 的分析报告"
- 与你的 AML Investigation Mate 项目架构高度一致，可以直接对标讲

> **Follow-up 提示：** LangGraph 中如何处理 tool call 失败（Neo4j 超时）？（retry with exponential backoff + fallback node：如果图查询失败，降级到只用表格特征的判断；state machine 设计保证 partial failure 不导致整个 workflow 崩溃）

### Q8: 如何评估和监控线上欺诈检测模型的健康状态？

**回答：**

**离线指标（评估）：**
- **AUC-ROC**：对阈值不敏感，衡量整体排序能力
- **AUC-PR（Precision-Recall AUC）**：类别不平衡时比 AUC-ROC 更有意义
- **F1@K**：在 Top-K 高风险地址中的 F1（更贴近业务实际操作量）

**线上监控（监控）：**
- **模型输出分布**：每日高风险比例是否突变（突增可能是欺诈潮，突降可能是模型退化）
- **特征分布漂移**：PSI（Population Stability Index）监控每个输入特征的分布变化，PSI > 0.2 触发告警
- **人工复核结果**：分析师确认/驳回率作为 Precision 的近似估计（如果驳回率从 10% 升到 40%，说明模型 Precision 下降）
- **新型欺诈覆盖**：对已知新型欺诈手法，检查模型是否能识别（结合 threat intelligence 定期测试）

**模型更新触发条件：**
- PSI > 0.2 for any critical feature
- 人工复核 Precision < 阈值（如 60%）
- 新型重大欺诈事件发生后（如新型 DeFi 攻击手法）

> **Follow-up 提示：** 如何在不停机的情况下更新线上模型？（Shadow mode: 新模型与旧模型同时运行，仅记录新模型输出但不用于决策，对比一段时间后确认无问题再切流量；Blue-Green deployment 或 Canary release 逐步切换流量）

### Q9: 区块链欺诈检测和传统金融欺诈检测（信用卡欺诈）有哪些关键区别？

**回答：**

| 维度 | 传统金融（信用卡） | 区块链（DeFi）|
|------|----------------|------------|
| 数据可访问性 | 银行内部闭源数据 | 链上数据完全公开透明 |
| 实体身份 | 与实名账户直接绑定 | 假名（pseudonymous），需要地址归因 |
| 攻击原子性 | 攻击通常跨多个时间点 | Flash loan: 单笔交易完成整个攻击 |
| 不可撤销性 | 可以 chargeback | 区块链交易不可逆，需要预防而非事后 |
| 攻击者技术能力 | 通常低，买 stolen cards | 通常高，需要写智能合约 |
| 监管要求 | 明确 AML/KYC 要求 | 新兴监管，部分灰色地带 |
| 数据丰富度 | 交易数据 + KYC | 链上全图 + 跨协议行为 |
| 图结构重要性 | 中等（卡号关联） | 极高（洗钱链路必须图分析）|

**关键启示：** 区块链数据的公开透明性是双刃剑——欺诈行为留下永久痕迹（有利于检测），但欺诈者也可以分析其他地址的行为来规避检测。

> **Follow-up 提示：** 中心化交易所（OKX）相比纯链上分析的额外优势是什么？（链上地址 ↔ 实名账户映射、登录行为、设备指纹、充提记录；这些链下信号大幅提升检测精度）

### Q10: 用一段话描述你会如何向面试官 pitch 你的完整技术方案。

**回答（示例 pitch）：**

"我会设计一个三层级联系统。第一层是毫秒级规则引擎，处理黑名单地址和极端异常，拦截明显欺诈。第二层是 ML 评分引擎：用 BERT-style Transformer 对地址的最近 256 笔交易序列建模，捕捉行为模式；同时用 FTTransformer 处理地址聚合特征；两个模型 ensemble 输出风险分数。第三层是深度分析：对高风险地址构建 2-hop 交易图，用 GNN + Graphormer 检测洗钱团伙；同时用 Anomaly Transformer 做无监督异常分析覆盖未知攻击手法。训练策略上，先用数百万无标注正常地址的交易序列做 MLM 预训练，解决标签稀缺问题，再用有标注欺诈数据 fine-tune，用 Focal Loss 处理类别不平衡。生产上，每周增量更新模型，用 PSI 监控特征漂移。对于高风险地址，LangGraph agent 自动化调查 pipeline，结合 Neo4j 图查询和链下 KYC 数据，生成风险报告供分析师复核。"

> **Follow-up 提示：** 如果面试官说"系统太复杂了，如果你只能做一件事，做哪个"？（Transformer 序列模型 fine-tuning，因为：1. 直接建模地址行为，可解释；2. 预训练迁移学习解决标签稀缺；3. 端到端可以快速迭代；图模型和 agent 是增量改进）
```

- [ ] **Step 4: Verify completeness**

Check against spec: problem framing ✓, feature engineering ✓, sequence modeling ✓, tabular transformer ✓, graph+transformer ✓, anomaly detection ✓, production concerns (drift, explainability, latency, imbalance) ✓, OKX-specific angle (DeFi fraud, LangGraph, exchange advantages) ✓, 10 Q&A ✓

- [ ] **Step 5: Commit**

```bash
git add interview_question/okx_anti_fraud/04_fraud_detection_with_transformers.md
git commit -m "feat: add fraud detection with transformers system design for OKX interview"
```

---

## Self-Review Checklist

- [ ] **Spec coverage:** All 4 files cover all spec sections — no gaps
- [ ] **No placeholders:** No TBD, TODO, or vague "add appropriate X"
- [ ] **Type consistency:** All function/method references are self-contained within each task
- [ ] **Format:** All Q&A follows `### Q{N}:` → `**回答：**` → `> **Follow-up 提示：**` convention
- [ ] **Math notation:** All LaTeX uses `_{...}` subscript notation per CLAUDE.md
- [ ] **Bilingual headers:** All section titles are Chinese + English in parentheses

---

*Plan created: 2026-05-12*
