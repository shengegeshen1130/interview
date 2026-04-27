# Proactive Trend Detection with Account Linking & Clustering 面试准备

---

## Part 1: Project Story（项目背景与故事线）

---

### 1.1 项目背景

经过一系列reorg之后，我们组专门负责为PayPal backoffice的交易做solution。Front office model是交易发生当下可以拦截的model，拦不住的交易都会落到backoffice的范畴。所以我们专注在offline batch solution——因为不需要满足严格的SLA要求，也不需要real-time scoring，可以用更复杂的方法通过batch的方式来做。

### 1.2 Use Case

我主要负责的是linking相关的fraud detection项目，我一直是这个项目的PoC，从0到1不断迭代，把项目做到每年有 **100M net loss saving**。

**项目目标：** 在account lifecycle尽量早的时候，主动通过各种linking手段，把fraud trend里的账号聚类起来，然后做account level和group level的model，对高分段的账号和组直接take action或者送给investigator做组级别的review。

**核心假设：** 绝大部分fraudster不会只操作一个账号——为了效率和成功率，他们一定会用多个账号操作。这些账号之间存在某种linking关系，如果我们能成功捕捉到这些关系，就可以将他们一网打尽。

### 1.3 Impact 量化

**100M net loss saving 的计算方式：**
- **Net loss saving = 被我们拦截的 fraud 交易金额 − false positive 造成的客户损失/赔偿**
- 具体来说：我们把 model 输出的高分账号/组 take action（限制、关闭账号等），统计这些账号在被 action 之后本来会产生的 loss（通过对照组/历史行为外推），减去 false positive 导致的 customer harm 和 operational cost
- 每个 quarter 都会做 performance review，tracking action 后的 realized loss reduction
- 这个数字是年化的、跨所有 use case（buyer + seller）的总和

---

## Part 2: Technical Pipeline（技术流程详解）

---

### Step 1: Graph Construction（建图）

**边的类型：**
- **Strong linking：** credit card number, bank account number, device ID —— 这些是高置信度的共享信号
- **Fuzzy linking：** IP address, post code, device name —— 置信度较低，但覆盖面更广
- **Transaction edges：** 买卖双方之间的交易关系

**Asset Riskiness Aggregation：**
- 对每一个 asset（如一张信用卡、一个设备）统计其关联的 transaction 数量、account 数量、loss 金额等
- 这些统计量用来衡量一个 asset 的 riskiness，后续作为 edge feature 或 filtering 依据

**Common Node Removal：**
- 有些 asset 被大量正常账号共享（如公共 Wi-Fi 的 IP、大型商户的 device），会把不相关的账号连在一起，产生噪声
- **阈值策略：** 基于 asset 关联的 account 数量设阈值。比如一个 device ID 如果关联了超过 N 个账号（N 根据 asset type 不同而不同，通常几百到几千），就认为是 common node 予以移除
- 同时移除完全 safe 的 asset（无任何 loss、无任何 flag 的 asset）

**最终图的规模：**
- 建成 account-to-account 的同构图
- Account 数量约 **100M**，边的数量 **200-300M**（取决于 account 种类，seller account 的图边会更多）

---

### Step 2: Clustering（建组）

我们迭代了很多个版本，就像推荐系统里的**多路召回**——对不同的 use case 和 population 用不同的方法提升 recall，后面交给模型来决定哪些组是真正 risky 的。

**迭代历程：**

| 版本 | 方法 | 适用场景 |
|------|------|----------|
| V1 | **Seed-based** — 多个账号共享一个 asset，直接归为一组 | 最简单直接，适合 strong linking 明确的场景 |
| V2 | **Gremlin 2-3跳查询** — 在图上做多跳遍历扩展组 | 适合已知 seed，需要向外扩展找关联账号 |
| V3 | **Community Detection** — 先用 Spark Label Propagation，后来和 PD team 一起将 Louvain 搬上 Scala | 无监督发现社区结构，适合大规模未知 pattern |
| V4 | **Embedding Similarity** — 用 account embedding 算 cosine similarity | 找不到显式 linking 但行为高度相似的账号 |

> **V3 说明：** Community detection 算法的 implementation 是 PD team 做的，我们负责提 request、定义 evaluation 标准、做 A/B 对比。

---

### Step 3: Modeling（建模）

**Account Embedding 生成：**
1. 对内部已有的不同 MO（modus operandi）flag model 做**模型蒸馏**（knowledge distillation）
2. 把蒸馏模型输出的倒数第二层 embedding 拼接起来，作为 account 的初始特征表示
3. 通过 **AutoEncoder-Decoder** 对拼接后的高维 embedding 做降维，得到最终的 account embedding

**模型迭代：**

| 版本 | 方法 | 说明 |
|------|------|------|
| 早期 | **Group-level 聚合特征 + 传统 ML** | 对 account level 变量做 max/min/mean/entropy 等聚合到 group level，直接训练分类模型 |
| 后期 | **GNN（Graph Attention Network）** | 把每个生成的组作为一个小图，跑 graph-level 和 node-level 的 GNN，同时得到 account-level 和 group-level score |

**GAT 魔改细节：**
- 基于 DGL 的 Graph Attention Layer
- **核心改动：将 edge feature concat 到 attention layer 的输出（即 node representation）上**
- 原因：标准 GAT 只用 node feature 算 attention weight，忽略了边上的信息（如 linking type、asset riskiness）。我们的场景中 edge information 非常关键（比如两个账号是通过 credit card 还是 IP 连接的，riskiness 完全不同），所以把 edge feature concat 到 attention 输出的 node representation 上，让后续 layer 能利用这些信息

**两种融合 edge feature 的方法对比（我们试过两种，最终选了方法 B）：**

**方法 A — Edge-aware Attention（在 attention score 里融入 edge feature）：**
- 标准 GAT：$e_{ij} = \text{LeakyReLU}(a^T [h_i' \| h_j'])$
- Edge-aware：$e_{ij} = \text{LeakyReLU}(a^T [h_i' \| h_j' \| f_{ij}])$，把 edge feature $f_{ij}$ 塞进 attention score 计算
- 问题：edge info 影响的是 attention weight（"这个邻居有多重要"），经过 softmax 变成标量 $\alpha_{ij}$，**edge 信息被压缩成一个权重就没了，信息损失大**
- 流程：`h_i, h_j, f_ij → attention score → softmax → α_ij(标量) → h_i_new = Σ α_ij · h_j'`

**方法 B — Concat 到 attention 输出（我们的做法）：**
- Attention score 保持标准算法，只看 node feature
- 聚合完成后，把 edge feature **concat 到输出的 node representation** 上：$h_i^{new} = [\sum_{j} \alpha_{ij} \cdot h_j' \; \| \; \text{Agg}(f_{ij}, j \in N(i))]$
- 对邻居的 edge feature 做 mean/max pooling 聚合，**edge info 以完整向量形式保留**在 node representation 里
- 流程：`h_i, h_j → standard attention → Σ α_ij · h_j' = node_repr → [node_repr || Agg(f_ij)] = h_i_new`

| 维度 | 方法 A: Edge-aware Attention | 方法 B: Concat 到输出（我们的选择） |
|------|---------------------------|----------------------------------|
| Edge info 的角色 | 影响"听谁的"（attention weight） | 作为独立 feature 保留在 representation 里 |
| 信息保留 | 被压缩成标量权重，信息损失大 | 完整向量保留，后续层可充分利用 |
| 适合场景 | Edge 信息主要用于区分邻居重要性 | Edge 信息本身就是有价值的 feature |

**选择方法 B 的原因：** 我们的 edge feature（linking type、asset riskiness）不仅仅是"这个邻居重不重要"，而是本身就携带关键信号——两个账号通过 credit card 连接 vs 通过 IP 连接，riskiness 完全不同。这些信息需要完整地传递到下游做判断，而不是被压缩成一个 attention 权重

**多层 GAT 的意义：**
- **扩大感受野（receptive field）：** 1 层 GAT 只能看到 1-hop 邻居，K 层能感知 K-hop 范围内的图结构
- **我们的场景：** fraud ring 里的账号不一定直接相连（如 A→asset→B→asset→C），多层 GAT 让 A 和 C 即使没有直接边也能通过 message passing 互相影响，捕捉更远的 linking pattern
- **层数选择：** 不是越多越好——层数太多会导致 **over-smoothing**（所有节点的 representation 趋于相同，失去区分性）。我们的组通常几十到几百个节点，**2-3 层就足够覆盖整个小图的直径**

---

### Step 4: Dedup & Enqueue（去重与排队）

**ILP（Integer Linear Programming）去重：**

一个账号可能出现在多个组里（因为多路召回），需要做 dedup 来决定每个账号最终归属哪个组。

- **目标函数：** 最大化总体 risk score（或预期 loss saving），即选择的组合使得被覆盖的高风险账号总价值最大
- **约束条件：**
  - 每个账号只能被分配到一个组（避免重复 action）
  - 每个组的大小在合理范围内（太小不值得 review，太大 reviewer 处理不过来）
  - 总 enqueue 数量不超过 investigator 的 capacity

**Group Merge：**
- 对相邻/重叠的小组做合并，生成尽量大的组，提升 review 效率
- Investigator 做组级别的 review 比逐个账号 review 效率高很多

---

## Part 3: 核心技术问题

---

### Q1: 为什么用 GAT？其他 GNN 算法对比？

**回答：**

1. **GAT（Graph Attention Network）的优势：**
   - 通过 attention mechanism 自动学习不同邻居的重要性权重，不需要手动定义聚合方式
   - 在我们的场景中，一个账号的不同邻居重要性差异很大（比如通过 credit card 连接的邻居 vs 通过 IP 连接的邻居），GAT 能自动区分

2. **与其他 GNN 的对比：**
   - **GCN（Graph Convolutional Network）：** 对所有邻居等权聚合（基于度的归一化），无法区分邻居重要性。在我们的异质 linking 场景中表现不够好
   - **GraphSAGE：** 支持采样+聚合，scalability 更好，但聚合函数（mean/max/LSTM）是固定的，不如 attention 灵活
   - **GIN（Graph Isomorphism Network）：** 理论表达能力最强（等价于 WL test），但在我们的实际数据上并没有比 GAT 好，且可解释性更差
   - **RGCN（Relational GCN）：** 专门处理多种边类型，但参数量随边类型线性增长，在我们边类型较多时 overhead 较大
   - **Graphormer：** 见下方 Q1.1 详细对比

3. **最终选择 GAT 的原因：**
   - Attention 机制天然适合我们的场景（不同 linking type 重要性不同）
   - 可以方便地魔改来融合 edge information
   - 在实际实验中 performance 最好，且 attention weight 有一定可解释性

> **Follow-up 提示：** 面试官可能追问 GAT 的 multi-head attention 怎么工作，或者 attention weight 如何用于解释模型。

---

### Q1.1: Graphormer 是什么？和 GAT 有什么区别？

**回答：**

1. **Graphormer 简介：**
   - 微软研究院提出的 Graph Transformer 模型（2021, NeurIPS），将 Transformer 架构应用到图结构数据上
   - 核心思想：不再像 GNN 那样只聚合邻居信息（local message passing），而是让**每个节点都能 attend to 图中所有其他节点**（global attention），同时通过特殊的编码方式把图的结构信息注入到 Transformer 中
   - 在 OGB 等多个 benchmark 上刷新了 SOTA

2. **三大核心创新（结构信息编码）：**

   **a) Centrality Encoding（中心性编码）：**
   - 用节点的入度和出度作为 bias 加到 input embedding 上
   - $h_i^{(0)} = x_i + z_{deg^-(i)}^- + z_{deg^+(i)}^+$
   - 其中 $z^-$ 和 $z^+$ 是可学习的 embedding，按度数索引
   - **意义：** 让模型感知节点在图中的"重要程度"——高度数节点（hub）和低度数节点（peripheral）有不同的 bias

   **b) Spatial Encoding（空间编码）：**
   - 计算任意两个节点之间的最短路径距离（SPD），作为 attention bias
   - $A_{ij} = \frac{(h_i W_Q)(h_j W_K)^T}{\sqrt{d}} + b_{\phi(v_i, v_j)}$
   - 其中 $\phi(v_i, v_j)$ 是节点 i 和 j 之间的 SPD，$b$ 是按距离索引的可学习标量
   - **意义：** 让模型知道两个节点在图中有多"远"——距离近的节点 attention 更强

   **c) Edge Encoding（边编码）：**
   - 对两个节点之间最短路径上的所有边的 feature 做加权平均，作为额外的 attention bias
   - $c_{ij} = \frac{1}{N} \sum_{n=1}^{N} x_{e_n} \cdot w_n^E$
   - 其中 $e_1, e_2, ..., e_N$ 是 i 到 j 最短路径上的边，$w_n^E$ 是可学习的权重
   - **意义：** 边的信息不仅仅是"有没有连接"，还编码了路径上边的具体属性

3. **完整的 Attention 计算：**
   $$A_{ij} = \frac{(h_i W_Q)(h_j W_K)^T}{\sqrt{d}} + b_{\phi(v_i, v_j)} + c_{ij}$$
   - 第一项：标准 Transformer attention（内容相关性）
   - 第二项：空间编码 bias（图上距离）
   - 第三项：边编码 bias（路径上的边属性）

4. **GAT vs Graphormer 核心区别：**

   | 维度 | GAT | Graphormer |
   |------|-----|------------|
   | **Attention 范围** | 只看 1-hop 邻居（local） | 看所有节点（global） |
   | **感受野扩展** | 靠堆叠多层（K 层 = K-hop） | 一层就能看到全图 |
   | **结构信息** | 隐式通过 message passing 传递 | 显式编码（degree、SPD、edge） |
   | **Edge feature 融合** | 需要额外魔改（如我们的 concat 方案） | 原生支持（Edge Encoding） |
   | **计算复杂度** | O(E)，只看边 | O(N²)，全节点两两计算 |
   | **Over-smoothing** | 层数多时严重 | 不存在（不靠堆层扩大感受野） |
   | **位置信息** | 无显式位置编码 | Centrality + Spatial Encoding |

5. **为什么我们没有用 Graphormer：**
   - **计算复杂度 O(N²)：** 虽然我们的小图只有几十到几百个节点，Graphormer 理论上可行，但整体工程复杂度高于 GAT
   - **SPD 预计算开销：** 需要对每个小图预计算所有节点对的最短路径，增加 preprocessing 负担
   - **GAT 魔改已经够用：** 我们通过 concat edge feature 到 attention 输出已经解决了 edge information 融合的问题，在实际效果上已经很好
   - **可解释性：** GAT 的 attention weight 直接对应"哪个邻居重要"，更直观；Graphormer 的 attention 混合了内容、距离、边信息，解释起来更复杂
   - **但值得关注：** 如果后续组的规模变大或需要更强的结构感知能力，Graphormer 是一个升级方向

> **Follow-up 提示：** 可能问 Graphormer 的 positional encoding 和 Transformer 原版的 sinusoidal encoding 有什么区别、O(N²) 在你们的小图上是否真的是瓶颈。

---

### Q2: 解释一下 GAT 的原理

**回答：**

1. **核心思想：** 在聚合邻居信息时，通过 attention mechanism 学习每个邻居的重要性权重

2. **计算步骤：**
   - **Step 1 — Linear Transformation：** 对每个节点的特征做线性变换：$h_i' = W \cdot h_i$
   - **Step 2 — Attention Score：** 对每对相邻节点 (i, j)，计算 attention score：$e_{ij} = \text{LeakyReLU}(a^T [h_i' \| h_j'])$，其中 $a$ 是可学习的 attention vector，$\|$ 表示 concat
   - **Step 3 — Softmax 归一化：** 对节点 i 的所有邻居做 softmax：$\alpha_{ij} = \text{softmax}_j(e_{ij})$
   - **Step 4 — 加权聚合：** $h_i^{new} = \sigma(\sum_{j \in N(i)} \alpha_{ij} \cdot h_j')$

3. **Multi-head Attention：** 类似 Transformer，用 K 个独立的 attention head 并行计算，最后 concat 或 average：
   - 中间层：concat → $h_i^{new} = \|_{k=1}^{K} \sigma(\sum_j \alpha_{ij}^k h_j'^k)$
   - 输出层：average → $h_i^{new} = \sigma(\frac{1}{K}\sum_k \sum_j \alpha_{ij}^k h_j'^k)$

4. **我们的魔改：** 在 Step 4 得到 $h_i^{new}$ 之后，将 edge feature concat 到这个 node representation 上，再送入下一层。这样后续层在计算 attention 和聚合时能感知到边的信息。

> **Follow-up 提示：** 可能问 attention 的时间复杂度、与 Transformer attention 的区别。

---

### Q3: LPA vs Louvain vs Leiden 对比

**回答：**

| 维度 | LPA (Label Propagation) | Louvain | Leiden |
|------|------------------------|---------|--------|
| **原理** | 每个节点初始一个 unique label，每轮迭代中每个节点采纳邻居中最多的 label | 两阶段迭代：(1) 局部贪心优化 modularity (2) 将社区缩为超级节点，重复 | 改进 Louvain，增加了 refinement 阶段，确保社区内部连通 |
| **时间复杂度** | 近线性 O(m)，非常快 | 近线性 O(m)，实际比 LPA 慢 | 与 Louvain 相当，refinement 有额外开销 |
| **结果稳定性** | 不稳定，每次运行结果可能不同（随机选择顺序） | 较稳定，但可能产生内部不连通的社区 | 最稳定，保证社区内部连通 |
| **社区质量** | 较粗糙，容易产生超大社区 | 好，modularity 优化有理论保证 | 最好，修复了 Louvain 的 disconnected community 问题 |
| **可扩展性** | 最好，适合超大规模图 | 好，Spark 上有成熟实现 | 中等，大规模实现相对少 |
| **我们的使用** | 最早的版本，Spark 原生支持，快速上线 | 主力版本，和 PD team 合作搬上 Scala，效果最好 | 评估过但当时没有成熟的分布式实现 |

**选型总结：**
- 快速原型 / 超大图 → LPA
- 生产系统 / 需要高质量社区 → Louvain
- 对社区连通性有严格要求 → Leiden

**Modularity（模块度）详解：**

Modularity 是衡量社区划分质量的核心指标，Louvain 算法的优化目标就是最大化 modularity。

- **直觉：** 一个好的社区划分，社区内部的边应该比"随机情况下预期的边数"多得多
- **公式：**
  $$Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$
  - $A_{ij}$：节点 i 和 j 之间的边权（有边=1，无边=0）
  - $k_i, k_j$：节点 i 和 j 的度数
  - $m$：图中总边数
  - $c_i, c_j$：节点 i 和 j 所属的社区
  - $\delta(c_i, c_j)$：如果 i 和 j 在同一社区则为 1，否则为 0
  - $\frac{k_i k_j}{2m}$：**null model**——在随机图中 i 和 j 之间的期望边数（保持度数分布不变的前提下）

- **取值范围：** Q ∈ [-0.5, 1]，通常有意义的社区结构 Q > 0.3
- **含义：** Q 越大 → 社区内部边密度远超随机预期 → 社区划分越好

**Louvain 算法的两阶段迭代：**

1. **Phase 1 — 局部贪心优化：**
   - 每个节点初始为独立社区
   - 遍历每个节点，尝试将其移入邻居所在的社区，计算 modularity gain $\Delta Q$
   - $\Delta Q = \left[ \frac{\sum_{in} + 2k_{i,in}}{2m} - \left(\frac{\sum_{tot} + k_i}{2m}\right)^2 \right] - \left[ \frac{\sum_{in}}{2m} - \left(\frac{\sum_{tot}}{2m}\right)^2 - \left(\frac{k_i}{2m}\right)^2 \right]$
     - $\sum_{in}$：目标社区内部边权总和
     - $\sum_{tot}$：目标社区所有节点的度数总和
     - $k_{i,in}$：节点 i 与目标社区内部节点的边权总和
     - $k_i$：节点 i 的度数
   - 选择 $\Delta Q$ 最大的社区移入，如果所有 $\Delta Q \leq 0$ 则不移动
   - 反复迭代直到没有节点再移动（局部最优）

2. **Phase 2 — 社区聚合：**
   - 把 Phase 1 得到的每个社区缩成一个**超级节点**
   - 社区内部的边变成超级节点的自环，社区之间的边合并为超级节点之间的边
   - 在这个缩小的图上重新跑 Phase 1
   - 重复直到 modularity 不再增加

**Resolution Limit 问题：**
- Modularity 优化有一个已知的缺陷：**倾向于合并小社区**，即使它们之间连接很少
- 当社区的边数少于 $\sqrt{2m}$ 时，modularity 优化可能无法正确识别它们
- **对我们的影响：** 非常小的 fraud group（3-5 个账号）可能被合并到更大的社区里
- **解决方案：** 调整 resolution parameter（Louvain 的变体支持）+ 后处理拆分过大的社区

> **Follow-up 提示：** 可能问 null model 为什么用 $\frac{k_i k_j}{2m}$（configuration model）、resolution parameter 具体怎么调。

---

### Q4: Embedding Similarity 方法是怎么做的？

**回答：**

1. **动机：** 有些 fraud ring 非常隐蔽，账号之间没有任何显式的 linking（不共享任何 asset），但行为模式高度相似。传统图方法完全无法发现这类 pattern。

2. **方法：**
   - 利用 Step 3 中生成的 account embedding（AutoEncoder 输出）
   - 对所有账号的 embedding 计算 pairwise cosine similarity
   - 设定 similarity threshold（通过 precision-recall 分析确定），超过阈值的账号对视为潜在同组
   - 对这些 "虚拟边" 再做一轮 clustering

3. **挑战与优化：**
   - **计算量：** 100M 账号的 pairwise 计算不可行，所以先用 LSH（Locality Sensitive Hashing）或 ANN（Approximate Nearest Neighbor）做候选集缩减
   - **Precision 控制：** 这个方法的 recall 高但 precision 低，所以阈值设得比较保守，更多作为其他方法的补充

4. **效果：** 作为多路召回的一路，额外发现了约 10-15% 的 fraud group 是其他方法找不到的

> **Follow-up 提示：** 可能问 LSH 的原理、embedding 的质量如何评估。

---

### Q4.1: LSH / ANN 候选集缩减具体怎么做？

**回答：**

1. **问题：** 100M 账号算 pairwise cosine similarity = O(n²) ≈ 10^16 次计算，完全不可行。

2. **我们的方案：Spark MLlib 原生 LSH**
   - Spark MLlib 提供两种 LSH 实现：
     - **BucketedRandomProjectionLSH** — 基于 Euclidean distance
     - **MinHashLSH** — 基于 Jaccard similarity
   - **我们用的是 BucketedRandomProjectionLSH：** 先对 account embedding 做 L2 normalize，这样 Euclidean distance 和 cosine similarity 单调相关（$\|a-b\|^2 = 2 - 2\cos\theta$），可以直接用 Euclidean LSH 等价地做 cosine similarity 搜索
   - **两个核心 API，对应不同场景：**
     - `approxSimilarityJoin(dfA, dfB, threshold)` — 输出所有距离小于 threshold 的相似对，适合**无监督全量发现**
     - `approxNearestNeighbors(dataset, key, K)` — 为单个查询向量找 K 个近邻，适合**seed-based 扩展**

3. **LSH 原理：**
   - **核心思想：** 把相似的向量大概率 hash 到同一个 bucket，不相似的大概率 hash 到不同 bucket
   - **做法：** 随机生成 k 个超平面（random projection vectors），每个 embedding 根据在每个超平面的哪一侧得到一个 k-bit 的 hash signature（如 `01101...`），hash 值相同的向量落在同一个 bucket 里，**只在同一个 bucket 内部做 pairwise similarity 计算**
   - **数学直觉：** $P(\text{same hash bit}) = 1 - \frac{\theta}{\pi}$，其中 θ 是两向量夹角，越相似概率越大
   - **精度控制：** 用多组独立的 hash function（band technique）取并集降低漏掉相似对的概率。k 越大 → precision 越高但 recall 越低；band 数越多 → recall 越高

4. **Spark 上的关键参数：**
   - `bucketLength`：bucket 宽度，越小 → precision 越高但计算量越大
   - `numHashTables`：hash table 数量（对应 band 数），越多 → recall 越高但计算量越大
   - 实际调参通过在 sample 数据上做 precision-recall 分析来确定

5. **两种使用场景与代码示例：**

   **场景 A：无监督全量发现（approxSimilarityJoin）**
   - 在所有账号之间找相似对，发现未知的 fraud group
   ```python
   from pyspark.ml.feature import BucketedRandomProjectionLSH

   brp = BucketedRandomProjectionLSH(
       inputCol="normalized_embedding",
       outputCol="hashes",
       bucketLength=2.0,
       numHashTables=5
   )
   model = brp.fit(df)
   # 一次性输出所有距离 < threshold 的相似对
   similar_pairs = model.approxSimilarityJoin(df, df, threshold=1.0, distCol="distance")
   ```

   **场景 B：Seed-based 扩展（approxNearestNeighbors）**
   - 已知一批 fraud seed 账号，快速找到与每个 seed 最相似的 K 个账号来扩展组
   ```python
   # 对每个 seed account，找 top-K 相似账号
   for seed_vector in seed_accounts:
       neighbors = model.approxNearestNeighbors(df, seed_vector, numNearestNeighbors=100)
   ```
   - 适用于 Step 2 中 V1/V2 的 seed-based clustering：已有明确 seed 时，用 embedding 相似度向外扩展，替代或补充 Gremlin 多跳查询

   **两者的关系：** approxSimilarityJoin 是多路召回中的无监督全量发现；approxNearestNeighbors 是 seed-based 定向扩展。两者互补，覆盖不同场景。

6. **为什么用 Spark LSH 而不是 FAISS 等：**
   - Pipeline 全程跑在 Spark 上，用 MLlib LSH 无需额外的基础设施
   - FAISS 需要单机/GPU，100M 向量可能放不进单机内存，还需要额外的 driver 收集分发逻辑
   - Spark LSH 天然分布式，和上下游 Spark job 无缝衔接

7. **核心 tradeoff：** 用近似换速度——牺牲极少量 recall（可能漏掉几个真正相似的对），换来几个数量级的加速。在 fraud detection 场景中完全可接受，因为 embedding similarity 本身只是多路召回的一路。

> **Follow-up 提示：** 可能问 bucketLength 怎么调、L2 normalize 后 Euclidean 和 cosine 的数学关系、为什么不用 MinHashLSH。

---

## Part 4: 面试官视角深度问题

---

### 系统设计类

---

#### Q5: 图的规模是多少？怎么做到 scalable？

**回答：**

1. **规模：** 100M 节点、200-300M 条边，整个 pipeline 是 daily/weekly batch job

2. **Scalability 策略：**
   - **图构建阶段：** 全部在 Spark + BigQuery 上完成，天然分布式。Common node removal 和 safe asset filtering 能把图规模缩减 30-50%
   - **Community detection：** LPA/Louvain 都是在 Spark 上跑的分布式版本
   - **GNN 训练：** 不是在整张大图上跑 GNN，而是把 clustering 输出的每个组作为独立的小图（通常几十到几百个节点），分别跑 GNN。这样天然可以并行，也避免了大图 GNN 的 memory 和 message passing 问题
   - **Embedding similarity：** 用 LSH/ANN 做近似搜索，避免 O(n²) 的全量计算

3. **关键设计决策：** 先 clustering 再 modeling，而不是直接在大图上跑 GNN。这样既降低了计算复杂度，又让每个小图的 signal-to-noise ratio 更高

> **Follow-up 提示：** 可能追问为什么不直接在大图上跑 GNN、mini-batch training 的考量。

---

#### Q6: Batch pipeline 是怎么设计的？频率如何？

**回答：**

1. **Pipeline 架构：** 典型的 batch ETL pipeline
   - **数据源：** BigQuery（交易数据、账号数据）+ 内部 feature store
   - **图构建 & Clustering：** Spark job，跑在公司的 Hadoop 集群上
   - **GNN 训练 & 推理：** PyTorch + DGL，GPU 集群上跑
   - **Dedup & Enqueue：** Python + OR-tools（ILP solver）

2. **频率：**
   - 图构建和 clustering：**weekly**（图的结构变化不会太快）
   - Model scoring：**daily**（新交易会带来新的 feature 变化）
   - Model retraining：**monthly** 或按需（fraud pattern shift 时 retrain）

3. **与 front office 的关系：**
   - Front office model 在交易发生时做 real-time scoring，能拦截的直接拦截
   - 我们是 backoffice，处理的是 front office 漏掉的交易。两者是互补关系，不是替代关系
   - 我们会用 front office model 的 score 作为 feature 之一

> **Follow-up 提示：** 可能问如何监控 pipeline 健康度、data freshness 的 tradeoff。

---

### 建模类

---

#### Q7: 为什么用 AutoEncoder 做 embedding？其他方法对比？

**回答：**

1. **为什么用 AutoEncoder：**
   - 输入是多个蒸馏模型倒数第二层 embedding 的 concat，维度非常高（几千维）
   - AutoEncoder 做非线性降维，比 PCA 能保留更多非线性结构信息
   - 训练是无监督的，不需要额外标签，利用 reconstruction loss 就能学到有意义的 representation

2. **为什么不用其他方法：**
   - **PCA：** 线性降维，会丢失非线性关系。实验中 PCA 降维后的 embedding 在下游任务上效果差 10-15%
   - **t-SNE / UMAP：** 适合可视化但不适合做 downstream task 的 feature，且不能对新数据做 transform
   - **直接用原始高维 embedding：** 维度过高会导致 GNN 训练困难、过拟合、计算开销大
   - **VAE：** 试过，效果和 AutoEncoder 差不多，但训练更不稳定（KL loss 的权重需要调）

3. **蒸馏的意义：** 每个 flag model 从不同 MO（如 ATO、stolen finance）的角度理解账号，蒸馏后的 embedding 捕捉了多维度的风险信号，比单独用任何一个 model 的 feature 都更全面

> **Follow-up 提示：** 可能问 AutoEncoder 的 architecture（层数、bottleneck 维度）、如何选择 bottleneck 维度。

---

#### Q8: GNN 相比传统 ML 模型的优势是什么？

**回答：**

1. **传统方法的局限：**
   - 把 account-level 变量做 max/min/entropy 聚合到 group level，本质是人工定义的 aggregation，信息损失大
   - 无法利用图的拓扑结构信息（谁和谁连接、连接的模式）
   - Group 内部的结构差异被抹平了（比如一个 star topology 和一个 chain topology 的 group 看起来一样）

2. **GNN 的优势：**
   - **结构感知：** 通过 message passing 自动学习图的拓扑特征
   - **端到端学习：** 不需要手动设计聚合函数，模型自己学怎么聚合最优
   - **多尺度输出：** 同时得到 node-level 和 graph-level 的 representation，一个模型解决两个问题（account risk 和 group risk）
   - **边信息利用：** 通过魔改 GAT，能把 edge feature 融入模型，传统方法很难做到

3. **实际提升：** 从传统方法切换到 GNN 后，group-level 的 precision@top1000 提升了约 15-20%

4. **实验设计与 Ablation 数据（补充）：**
   - **实验方法：** 在同一批 clustering 输出上，固定 train/val/test split（temporal split），分别用传统方法和 GNN 训练，对比 precision@K
   - **Ablation 实验——各组件的 marginal contribution：**

   | 模型配置 | Precision@1000 | 相对 baseline 提升 |
   |---------|---------------|-------------------|
   | Baseline: Group aggregate features + LightGBM | 0.62 | — |
   | + GNN (标准 GAT，无 edge feature) | 0.69 | +11.3% |
   | + Edge feature (方法 A: edge-aware attention) | 0.71 | +14.5% |
   | + Edge feature (方法 B: concat 到输出) | 0.74 | +19.4% |
   | + Multi-head attention (K=4) | 0.75 | +21.0% |
   | + Multi-task (node + graph level) | 0.76 | +22.6% |

   - **结论：** 最大的单步提升来自 GNN 本身（+11.3%），其次是 edge feature concat 方案（+5.0%），multi-task 的 marginal gain 较小但一个模型同时输出两个 level 的 score，工程上更简洁

> **Follow-up 提示：** 可能问 node-level 和 graph-level prediction 是怎么同时训练的（multi-task loss）。

---

#### Q9: 模型蒸馏的具体策略是什么？

**回答：**

1. **Teacher Models：** 内部已有的多个 MO-specific flag model（如 ATO model、stolen finance model、collusion model 等），每个都是已经在线上跑的成熟模型

2. **蒸馏过程：**
   - 用 teacher model 对所有账号做推理，得到 soft label（probability output）
   - 训练一个小的 student DNN，目标是拟合 teacher 的 soft label（而不是原始 hard label）
   - 取 student model 倒数第二层的 hidden representation 作为 embedding

3. **为什么蒸馏而不是直接用 teacher model 的 feature：**
   - Teacher model 的 feature 很多是 raw feature，维度高且噪声大
   - 蒸馏后的 embedding 是压缩过的、有 discriminative power 的 representation
   - 不同 teacher model 的 feature space 不同，蒸馏可以统一到同一个 representation space

4. **多个蒸馏 embedding 的 concat：** 每个蒸馏模型捕捉不同 MO 的 signal，concat 起来形成多视角的账号画像

> **Follow-up 提示：** 可能问 soft label vs hard label 蒸馏的区别、temperature 的作用。

---

### 特征工程类

---

#### Q10: Asset riskiness aggregation 具体怎么做？

**回答：**

1. **Asset 粒度的统计量：**
   - 关联的 account 数量（distinct count）
   - 关联的 transaction 数量和金额
   - 关联账号中已被标记为 fraud 的比例
   - 历史 loss 总额
   - 关联账号的 age 分布（新账号比例高意味着 riskier）

2. **Aggregation 方式：**
   - 对每个 asset type（credit card、device、IP 等）分别计算
   - 归一化为 percentile rank（避免不同 asset type 量级差异）
   - 最终生成 asset-level risk score

3. **用途：**
   - **作为 edge feature：** 在 GAT 中，两个账号之间的边的 feature 就包含了连接它们的 asset 的 riskiness
   - **作为 filtering 条件：** 高 riskiness 的 asset 优先保留，低 riskiness 的 asset 可以在图简化时移除
   - **作为 common node 判断依据：** 高度数但低 riskiness 的 asset 大概率是 common node

> **Follow-up 提示：** 可能问如何处理 asset riskiness 的时效性（同一个 device 可能先正常后被 compromise）。

---

#### Q11: Fuzzy linking 怎么处理？不会引入太多噪声吗？

**回答：**

1. **Fuzzy linking 的特点：**
   - IP address、post code、device name 这些信号不像 credit card number 那样有唯一性
   - 同一个 IP 可能被大量无关用户共享（如企业网络、VPN）
   - 但对于某些 fraud pattern（如同一地区批量注册），fuzzy linking 是唯一的线索

2. **噪声控制策略：**
   - **Common node removal：** 上面提到的阈值过滤，高度数的 fuzzy asset 直接移除
   - **Edge weight 设计：** 给 fuzzy linking 边更低的权重，在 community detection 和 GNN 中影响更小
   - **时间窗口限制：** 只考虑一定时间窗口内（如 30 天）共享同一 fuzzy asset 的账号对
   - **组合信号：** 单独一个 fuzzy link 不足以成组，但多个 fuzzy link 叠加（如同时共享 IP + post code）则可信度提高

3. **实际效果：** 加入 fuzzy linking 后 recall 提升了约 20%，precision 下降不到 5%（因为后面有 model 兜底）

> **Follow-up 提示：** 可能问 edge weight 的具体设计、不同 asset type 的权重怎么确定。

---

### 评估类

---

#### Q12: Precision 和 Recall 的 tradeoff 怎么做？

**回答：**

1. **两层 tradeoff：**
   - **Clustering 层（召回）：** 这一层偏重 recall，因为后面有 model 做 precision 把控。多路召回就是为了尽量不漏掉任何 fraud group
   - **Modeling 层（精排）：** 这一层偏重 precision，因为 investigator capacity 有限，enqueue 的组必须高质量

2. **评估指标：**
   - **Precision@K：** 在 top K 个组中，真正是 fraud 的比例。K 由 investigator capacity 决定
   - **Recall：** 在所有已知 fraud group 中，我们发现了多少比例
   - **Net loss saving：** 最终的 business metric，综合了 precision 和 recall 的效果

3. **调整方式：**
   - Clustering 阈值（如 similarity threshold）影响 recall
   - Model score threshold 影响 precision
   - ILP 中的 capacity constraint 影响最终 enqueue 数量
   - 通过调整这些 knob 来达到业务目标（通常目标是在 investigator capacity 固定的前提下最大化 net loss saving）

> **Follow-up 提示：** 可能问如何获取 ground truth label 来计算这些指标。

---

#### Q13: 有做 A/B test 吗？怎么设计的？

**回答：**

1. **挑战：** 传统 A/B test 在 fraud detection 中不太适用——你不能对一半的 fraud 不采取行动来做对照组

2. **我们的评估方式：**
   - **Backtest：** 用历史数据验证——如果我们在 T 时间点发现了这些组，到 T+30/60/90 天这些账号实际产生了多少 loss
   - **Holdout test：** 随机 hold out 一部分低风险组不做 action，观察它们后续的 loss 发展，与被 action 的组做对比
   - **Champion-Challenger：** 新版本模型和旧版本模型同时跑一段时间，比较 precision、recall、net loss saving
   - **Investigator feedback：** Investigator review 后会标注 group 是否确实是 fraud，这个作为最终的 ground truth

3. **关键指标对比：**
   - Action 后的 realized loss reduction
   - Investigator 的 confirm rate（组被 review 后确认为 fraud 的比例）
   - False positive rate 和 customer harm

4. **Holdout Test 的样本量计算与 Statistical Power（补充）：**
   - **问题建模：** 想检测新模型 vs 旧模型的 precision 差异。假设旧模型 precision=0.62，希望检测到 10% relative improvement（即 new precision ≥ 0.68）
   - **Power Analysis：**
     - H₀: p_new = p_old = 0.62，H₁: p_new = 0.68
     - 使用 two-proportion z-test，α=0.05（单侧），power=0.80
     - 所需样本量：每组约 550-600 个 group（用公式 $n = \frac{(Z_\alpha \sqrt{2\bar{p}(1-\bar{p})} + Z_\beta \sqrt{p_1(1-p_1)+p_2(1-p_2)})^2}{(p_1-p_2)^2}$）
   - **实际限制：** 每周 enqueue 的 group 数量有限（几百到几千），所以积累足够的样本量需要几周到一个月的时间
   - **Holdout 比例选择：** 通常 holdout 10-20% 的低风险组。比例太高会增加 fraud loss 风险，太低则积累样本时间过长
   - **Statistical Test：** 用 McNemar test（paired）更合适，因为 champion 和 challenger 面对的是相同的 group population，配对检验 power 更高

> **Follow-up 提示：** 可能问 holdout test 的伦理问题、如何平衡实验和用户保护。

---

### 业务类

---

#### Q14: Fraud pattern 是怎么演变的？你们怎么应对？

**回答：**

1. **常见的演变方式：**
   - **Linking 规避：** Fraudster 学会避免共享 device 或 credit card，改用每次不同的设备
   - **行为模拟：** 注册后先做正常交易，养号一段时间再开始 fraud
   - **分散化：** 从大 ring 拆分成多个小 ring，降低被一网打尽的风险
   - **新 MO 出现：** 出现全新的 fraud 手法，已有模型完全没见过

2. **应对策略：**
   - **多路召回：** 不依赖单一方法，新加的 embedding similarity 就是为了应对 linking 规避
   - **持续迭代：** 定期分析 miss 掉的 fraud case，识别新的 pattern，补充新的召回方法
   - **Feature refresh：** 不断加入新的 signal（如新的 device fingerprint、behavioral feature）
   - **Model retrain：** 当 performance drift 被监控到时，触发 retrain
   - **与 front office 联动：** 他们发现的新 pattern 会 feed 给我们，反之亦然

> **Follow-up 提示：** 可能问具体的 pattern shift 案例、adversarial ML 的考量。

---

#### Q15: 这个系统和 front office 是什么关系？

**回答：**

1. **定位区分：**
   - **Front office：** 交易发生时的 real-time scoring，延迟要求 <100ms，只能用简单模型和有限 feature
   - **Backoffice（我们）：** 交易完成后的 batch analysis，无延迟限制，可以用复杂方法（graph、GNN、ILP）

2. **互补关系：**
   - Front office 拦不住的 → 落到我们这里
   - 我们发现的 fraud trend → feed 回 front office 作为新 signal
   - 我们的 model score 可以作为 front office model 的 feature 之一

3. **信息流：**
   - Front office model score 是我们的输入 feature 之一
   - 我们的 group risk score 可以被 front office 用来对 group 内的新交易加强审核

---

### 工程挑战类

---

#### Q16: 图太大怎么办？有没有 memory 问题？

**回答：**

1. **问题：** 100M 节点 + 300M 边的图放不进单机内存

2. **解决方案：**
   - **图构建：** 在 Spark 上做，天然分布式，不存在单机 memory 问题
   - **Community detection：** Spark 上的分布式 LPA/Louvain，每个 partition 只处理图的一部分
   - **GNN：** 关键设计——先 clustering 把大图切成小图，每个小图（组）通常只有几十到几百个节点，单个小图可以轻松放进 GPU memory。batch 处理多个小图时用 DGL 的 batched graph
   - **Embedding similarity：** 用 ANN index（如 FAISS）代替暴力搜索

3. **如果不先 clustering：**
   - 直接在大图上跑 GNN 需要 mini-batch training（如 GraphSAGE 的 neighbor sampling），但 100M 节点的图即使做 sampling 也非常吃资源
   - 所以我们的 "先 clustering 后 modeling" 架构从工程角度也是更合理的

> **Follow-up 提示：** 可能问 DGL batched graph 的原理、FAISS 的使用细节。

---

#### Q17: Cold start 问题怎么解决？

**回答：**

1. **问题：** 新注册的账号几乎没有交易历史和 feature，传统模型很难对其做出判断

2. **我们的优势：**
   - **这正是我们项目的核心价值** —— 在 account lifecycle 早期，通过 linking 而非交易历史来识别风险
   - 新账号虽然没有交易 feature，但注册时就可能暴露 linking 信号（如 device ID、IP）
   - 如果一个新账号和一批已知 fraud 账号共享 device，即使它还没做任何交易，也能被我们发现

3. **具体策略：**
   - **注册阶段 linking：** 新账号注册时立即检查与现有图的 linking
   - **蒸馏 embedding fallback：** 对于没有任何 linking 的新账号，用蒸馏模型的 embedding（基于有限的注册信息）做初始画像
   - **动态更新：** 随着新账号开始产生交易，逐步更新其 feature 和 embedding，在下一个 batch 周期重新评估

> **Follow-up 提示：** 可能问注册阶段的 linking 是 real-time 还是 batch 的。

---

#### Q18: 迭代过程中有哪些失败的经验？

**回答：**

1. **直接在大图上跑 GNN（失败）：**
   - 最开始尝试过在整张图上做 GraphSAGE，training 极其缓慢，而且因为图太大、噪声太多，模型效果反而不如传统方法
   - **教训：** 先 clustering 再 modeling 是更好的架构

2. **纯规则的 clustering 方法（效果有限）：**
   - V1/V2 的 seed-based 和 Gremlin 方法 precision 高但 recall 低，只能发现最明显的 fraud ring
   - **教训：** 需要无监督方法来发现未知 pattern

3. **Edge feature 直接加到 attention score 计算中（不如 concat 到输出）：**
   - 尝试过在 attention score 计算时融入 edge feature（类似 edge-aware attention），但效果不如简单地 concat 到输出
   - **教训：** 不是所有理论上更优雅的方法都效果更好，有时候简单的方法更 robust

4. **Modularity-based method 的 resolution limit：**
   - Louvain 有 resolution limit 问题，对于非常小的 fraud group（3-5 个账号）可能被合并到更大的社区里
   - **解决：** 通过调整 resolution parameter + 后处理拆分过大的社区

5. **过于激进的 common node removal：**
   - 一开始阈值设太低，把一些实际上有 signal 的 asset 也移除了，导致一些 fraud ring 被断开
   - **解决：** 通过 ablation study 调整阈值，同时结合 asset riskiness 而不仅仅看度数

> **Follow-up 提示：** 这类问题面试官很喜欢问，展示迭代思维和从失败中学习的能力。

---

### 统计严谨性与实验设计类 (Statistical Rigor & Experiment Design)

---

#### Q19: GNN 相比传统方法的 15-20% precision 提升——怎么验证是统计显著的？

**回答：**

1. **为什么需要统计检验：**
   - 15-20% 的 relative improvement 听起来很大，但如果 test set 太小或 variance 太大，可能只是噪声
   - 面试官会 challenge："你怎么知道这不是偶然的？"

2. **McNemar Test（配对检验，推荐）：**
   - 适用于同一 test set 上两个模型的对比——比 independent two-sample test 更 powerful
   - 构造 contingency table：

   |  | Model B Correct | Model B Wrong |
   |--|----------------|--------------|
   | **Model A Correct** | a | b |
   | **Model A Wrong** | c | d |

   - McNemar statistic: $\chi^2 = \frac{(b-c)^2}{b+c}$，自由度 df=1
   - 只看 **disagreement cells**（b 和 c）——两个模型都对或都错的样本不提供 differential information
   - 如果 p-value < 0.05，差异是统计显著的

3. **Bootstrap Confidence Interval：**
   - 从 test set 中有放回地抽样 B 次（B=1000-10000），每次计算 precision_GNN - precision_baseline
   - 取 2.5% 和 97.5% 分位数作为 95% CI
   - 如果 CI 不包含 0，说明 GNN 显著优于 baseline
   - 优点：不需要分布假设，适合小样本

4. **Effect Size (Cohen's d)：**
   - $d = \frac{\bar{x}_1 - \bar{x}_2}{s_{pooled}}$
   - d > 0.2（小效应）、d > 0.5（中效应）、d > 0.8（大效应）
   - 仅有 p-value 不够——p-value 只说明 "差异是否存在"，effect size 说明 "差异有多大"

5. **Fraud Detection 中的特殊挑战：**
   - **Label delay：** Test set 的 label 可能不完全 matured，影响评估
   - **Non-stationarity：** Fraud pattern 在变化，不同时期的 test set 结果可能不同
   - **Class imbalance：** 正样本（fraud）很少，标准 statistical test 的 power 可能不够
   - **解决：** 在多个 temporal split 上重复实验，报告结果的均值和方差，而不是单次实验的结果

> **Follow-up 提示：** 面试官可能追问 "如果 p-value 是 0.06 你怎么报告？"（答案：不要仅看 0.05 cutoff，报告 CI 和 effect size，让 stakeholder 综合判断）

---

#### Q20: Ablation Study——每个组件的贡献是多少？

**回答：**

1. **为什么 Ablation Study 重要：**
   - 展示你理解每个设计决策的 marginal value，而不是 "碰巧加了很多东西然后效果好了"
   - 帮助做资源分配决策——如果某个组件的 marginal gain 很小但工程成本高，可能可以去掉

2. **我们的 Ablation 实验设计：**

   **Pipeline 组件 Ablation：**

   | 去掉的组件 | Precision@1000 变化 | 说明 |
   |-----------|-------------------|------|
   | Full pipeline (baseline) | 0.76 | — |
   | − Multi-task loss (只用 graph-level) | 0.75 (−1.3%) | Multi-task 的 marginal gain 小 |
   | − Multi-head attention (K=1) | 0.74 (−2.6%) | Multi-head 有帮助但不关键 |
   | − Edge feature concat | 0.69 (−9.2%) | **Edge feature 是关键组件** |
   | − GNN (回到传统 ML) | 0.62 (−18.4%) | **GNN 是最大的提升来源** |

   **Clustering 方法 Ablation：**

   | Clustering 方法组合 | Group-level Recall | Group-level Precision |
   |-------------------|-------------------|----------------------|
   | 仅 Seed-based | 0.35 | 0.78 |
   | + Gremlin 多跳 | 0.48 | 0.72 |
   | + Community Detection | 0.65 | 0.68 |
   | + Embedding Similarity | 0.72 | 0.65 |
   | 全部方法 (多路召回) | 0.72 | 由后续 model 把控 |

3. **AutoEncoder Embedding Ablation：**

   | Embedding 方法 | 下游 GNN 的 Precision@1000 |
   |--------------|--------------------------|
   | Raw feature concat (无 AE) | 0.68 |
   | PCA 降维 | 0.70 |
   | AutoEncoder 降维 | 0.76 |
   | VAE | 0.75 |

4. **实验方法论要点：**
   - **固定随机种子：** 确保每次实验的 train/val/test split 和模型初始化一致
   - **多次重复：** 每个配置跑 3-5 次，报告均值 ± 标准差
   - **控制变量：** 每次只改一个组件，其他保持不变
   - **Temporal split：** 所有实验用相同的 temporal split，避免 data leakage

> **Follow-up 提示：** 面试官可能追问 "Edge feature 为什么这么重要？"（答案：在我们的场景中，linking type 和 asset riskiness 直接反映了 fraud 的性质）

---

#### Q21: GNN 中的 Over-smoothing 怎么检测和缓解？

**回答：**

1. **什么是 Over-smoothing：**
   - 随着 GNN 层数增加，所有节点的 representation 趋于相同（收敛到同一个 vector），失去区分性
   - 直觉：每一层 message passing 相当于一次 "平均化"，层数越多，信息扩散到整个图，所有节点都看到了全图的信息，变得无差异
   - 在我们的场景中，fraud 节点和 non-fraud 节点的 representation 变得相似，分类 performance 下降

2. **检测方法——MAD (Mean Average Distance)：**
   - 计算所有节点 representation 的两两距离的平均值
   - $MAD = \frac{2}{n(n-1)} \sum_{i<j} \|h_i - h_j\|_2$
   - 监控 MAD 随层数的变化：如果 MAD 持续下降，说明 over-smoothing 正在发生
   - 实验中我们发现：1 层 GAT 的 MAD = 0.85，2 层 = 0.72，3 层 = 0.65，5 层 = 0.31（已严重 over-smoothing）

3. **为什么 2-3 层是最优的：**
   - 我们的小图（几十到几百个节点）直径通常 3-5
   - 2-3 层 GAT 的感受野已经能覆盖整个小图的大部分节点
   - 更多层带来的 over-smoothing 损失 > 感受野扩大的收益

4. **缓解方法（如果需要更深的 GNN）：**
   - **Residual Connections：** $h_i^{(l+1)} = h_i^{(l)} + \text{GAT}^{(l)}(h_i^{(l)}, \{h_j^{(l)}\})$，让每层的原始信息 skip 到下一层
   - **DropEdge：** 训练时随机 drop 一定比例的边（如 10-20%），减缓信息扩散速度，类似 graph 上的 dropout
   - **PairNorm：** 对 node representation 做归一化，保持不同节点之间的 pairwise distance
   - **JKNet (Jumping Knowledge)：** 把每一层的 output concat 或 attention-aggregate，让最终 representation 包含不同感受野的信息

5. **实际选择：** 在我们的场景中，2-3 层 + residual connection 就够了，不需要更复杂的 anti-over-smoothing 技术。因为先 clustering 后 modeling 的架构意味着每个小图规模有限，over-smoothing 的风险本身就可控

> **Follow-up 提示：** 面试官可能追问 "如果图很大（不先做 clustering），over-smoothing 怎么办？"、"怎么选最优层数？"（答案：在 validation set 上比较不同层数的 performance + MAD）

---

#### Q22: 模型的 adversarial robustness——fraudster 能否逆向工程你的模型？

**回答：**

1. **Threat Model（威胁模型）：**
   - Fraudster 的目标：让自己的账号/组不被模型检测到
   - Fraudster 能做的：改变自己的行为（使用新设备、新 IP、模拟正常交易 pattern），但不能直接修改模型或数据
   - 这是一种 **evasion attack**——在 inference 阶段修改输入来逃避检测

2. **Feature Sensitivity Analysis：**
   - 分析哪些 feature 对模型 score 影响最大（SHAP/permutation importance）
   - **风险：** 如果关键 feature 是 fraudster 容易操纵的（如 IP 地址），模型就容易被 evade
   - **对策：** 确保 feature set 中包含 fraudster 难以操纵的 signal（如 network topology、historical loss）

3. **Graph-level 的 Adversarial 考量：**
   - **结构攻击：** Fraudster 可能故意添加 noise edges（和正常账号建立虚假 linking）来干扰 community detection
   - **特征攻击：** 模拟正常的交易行为来让 node feature 看起来正常
   - **检测方法：** 对 edge 的 "可信度" 做评估——如果一条 edge 两端的 node 特征差异很大，这条 edge 可能是 adversarial 的

4. **Robustness Testing 方法：**
   - **Feature Perturbation Test：** 对 top feature 做小幅扰动（±10-20%），看 model score 的变化。稳定的模型应该对小幅扰动不敏感
   - **Structure Perturbation Test：** 随机添加/删除 5-10% 的边，看 GNN 的 precision 变化。我们的实验显示添加 5% noise edges 后 precision 下降约 3%，说明模型有一定 robustness
   - **Temporal Robustness：** 在不同时间段的数据上测试，确保模型不依赖 temporal artifact

5. **实际的 Defense 策略：**
   - **多路召回的鲁棒性：** 即使 fraudster 规避了一种 linking 方式（如不共享 device），其他方式（embedding similarity、transaction pattern）仍可能 catch
   - **Feature diversity：** 不要过度依赖少数 feature，保持 feature set 的多样性
   - **定期 retrain：** Adversarial drift 可以通过 retrain 来部分缓解
   - **Red team exercise：** 定期让安全团队模拟 adversarial 行为，测试模型的弱点

> **Follow-up 提示：** 面试官可能追问 "有没有用过 adversarial training 来增强 robustness？"、"GNN 比传统 ML 更容易被攻击吗？"

---

#### Q23: Multi-task Loss（node-level + graph-level）怎么设计？权重怎么调？

**回答：**

1. **Multi-task 设计：**
   - **Node-level task：** 预测每个 account 是否是 fraud（binary classification）
   - **Graph-level task：** 预测整个 group 是否是 fraud group（binary classification）
   - 一个 GNN 同时输出两个预测，两个 loss 加权求和

2. **Loss 设计：**
   ```
   L_total = α × L_node + β × L_graph

   L_node = Binary Cross-Entropy (每个 node 的预测 vs label)
   L_graph = Binary Cross-Entropy (graph-level readout vs group label)
   ```
   - Graph-level readout 用 mean pooling + MLP 得到：$h_{graph} = \text{MLP}(\text{MeanPool}(h_i^{(L)}, \forall i \in G))$

3. **权重调整方法：**

   **方法 A — 手动调参（我们主要用的）：**
   - 从 α=1, β=1 开始，在 validation set 上 grid search
   - 发现 α=0.3, β=1.0 效果最好——graph-level task 是 primary objective，node-level 作为辅助 regularization
   - 直觉：group-level 的 label 更可靠（经过 investigator 确认），node-level label 有更多 noise

   **方法 B — Uncertainty Weighting (Kendall et al., 2018)：**
   - 核心思想：用 homoscedastic uncertainty 自动学习每个 task 的权重
   - 公式：$L = \frac{1}{2\sigma_1^2} L_{node} + \frac{1}{2\sigma_2^2} L_{graph} + \log\sigma_1 + \log\sigma_2$
   - $\sigma_1, \sigma_2$ 是可学习的参数，代表每个 task 的 noise level
   - Noise 大的 task 自动降低权重，noise 小的 task 权重更高
   - 避免了手动调 α, β 的麻烦，但增加了训练不稳定性

   **方法 C — GradNorm：**
   - 通过均衡不同 task 梯度的 norm 来调整权重
   - 如果某个 task 的梯度 norm 比平均值大很多，降低它的权重
   - 理论上最优雅但实现复杂，在我们的 small graph 上 overhead 不值得

4. **两个 task 之间的 Trade-off：**
   - Node-level prediction 帮助 GNN 学到更好的 node representation（每个 node 都有监督信号）
   - Graph-level prediction 是最终的 business objective
   - Node-level 作为 auxiliary task 起到了 regularization 的作用——防止 GNN 只学 graph-level 的粗粒度 pattern

5. **实际效果：**
   - 单独 graph-level loss 的 precision@1000 = 0.74
   - 加上 node-level auxiliary loss 后 = 0.76（+2.7%）
   - 提升主要来自 node representation 质量的提高，间接改善了 graph-level readout

> **Follow-up 提示：** 面试官可能追问 "如果两个 task 的 gradient 冲突怎么办？"（答案：冲突不严重，因为 node-level 和 graph-level fraud 高度相关）

---

#### Q24: Graph Embedding 的质量怎么评估（不仅仅看下游任务）？

**回答：**

1. **为什么需要 Intrinsic Evaluation：**
   - 如果只看下游任务（fraud detection precision），无法区分 "embedding 好" 还是 "下游 classifier 碰巧适配"
   - Intrinsic evaluation 帮助理解 embedding 本身的质量，指导 embedding 方法的选择和调优

2. **Visualization（定性评估）：**
   - **t-SNE / UMAP 降维可视化：** 把高维 embedding 投影到 2D，观察 fraud vs non-fraud 节点是否有清晰的分离
   - **按 MO type 染色：** 不同的 fraud type（ATO、stolen finance、collusion）在 embedding 空间中是否形成独立 cluster
   - **局限：** t-SNE 会扭曲距离，只能看 local structure，不能做 quantitative 结论

3. **Silhouette Score（聚类质量）：**
   - $S(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$，其中 a(i) = 样本 i 到同 cluster 其他样本的平均距离，b(i) = 样本 i 到最近 cluster 的平均距离
   - S ∈ [-1, 1]，越高越好。S > 0.5 通常表示较好的 cluster separation
   - 在我们的场景中，对 fraud vs non-fraud 两类做 silhouette 分析

4. **Cluster Purity：**
   - 对 embedding 做 KMeans 聚类，检查每个 cluster 中 fraud/non-fraud 的纯度
   - 高纯度说明 embedding 能自然地把 fraud 和 non-fraud 分开
   - Purity = $\frac{1}{n}\sum_{k} \max_c |C_k \cap L_c|$

5. **Link Prediction as Proxy Task：**
   - 随机隐藏一部分已知的边，用 embedding 的 cosine similarity 预测这些边是否存在
   - 如果 AUC 高，说明 embedding 保留了图的结构信息
   - 在我们的场景中：hidden edge prediction AUC = 0.87，说明 embedding 较好地保留了 account 之间的 linking 关系

6. **Intrinsic vs Extrinsic Evaluation 的关系：**
   - Intrinsic evaluation 是快速反馈——不需要训练下游 classifier，可以快速迭代 embedding 方法
   - Extrinsic evaluation（下游 fraud detection performance）是最终标准——intrinsic 好不一定 extrinsic 好
   - 最佳实践：用 intrinsic metrics 做 fast screening，shortlist 后用 extrinsic evaluation 做最终选择

> **Follow-up 提示：** 面试官可能追问 "t-SNE 和 UMAP 的区别是什么？"、"embedding 的维度怎么选？"

---

#### Q25: 时序数据的交叉验证怎么做？有没有 data leakage 风险？

**回答：**

1. **为什么 Random CV 在 fraud detection 中有问题：**
   - Fraud pattern 随时间演变（concept drift），random split 的 train/test 时间交叉，模型可能学到了未来的 pattern 来预测过去
   - 某些 feature 本身就有时序性（如 '近 30 天交易次数'），random split 会导致 feature 计算中包含了 test 时间段的数据

2. **我们的做法——Temporal Split + Walk-Forward Validation：**
   ```
   Graph 1: 基于 Week 1-4 的数据建图 → 用 Week 5-6 的 label 评估
   Graph 2: 基于 Week 1-6 的数据建图 → 用 Week 7-8 的 label 评估
   Graph 3: 基于 Week 1-8 的数据建图 → 用 Week 9-10 的 label 评估
   ...
   ```
   - 每个 fold 的训练数据严格在评估数据之前
   - 图是重新建的（因为 linking 关系随时间变化）
   - 模型 retrain 或直接 apply 取决于 fold 的目的（evaluation vs production simulation）

3. **Feature Engineering 中的 Data Leakage 风险：**
   - **Asset riskiness aggregation：** 计算 asset 的 riskiness 时，必须只用 observation date 之前的 loss 和 flag 数据。如果用了未来的 chargeback 数据，会 leak
   - **Community detection：** 图的构建本身用了边的信息，如果某些边只在未来才会出现（如未来的交易），需要确保图只用 observation date 之前的数据
   - **Account embedding (蒸馏)：** 蒸馏的 teacher model 是在历史数据上训练的，但需要确保 teacher model 没有用到 test 时间段的数据

4. **Label Delay 对 CV 的影响：**
   - Fraud label 有 30-90 天的 delay（chargeback）
   - 在 walk-forward 中，评估窗口需要预留 maturation gap：
   ```
   Train: Week 1-8 | Maturation Gap: Week 9-12 | Evaluate: Week 13-14 (用 matured label)
   ```
   - 如果不预留 maturation gap，近期数据的 label 不完整，会低估模型的 recall
   - 这个 gap 同时也是 production 中的 latency——模型上线后 90 天才能看到真实 performance

5. **交叉验证结果的汇报方式：**
   - 不要只报平均值——报告每个时间窗口的 performance 和方差
   - Performance 的 trend 本身也是有价值的信息：持续下降可能暗示 concept drift
   - 用 error bar 或 CI 展示结果的不确定性

> **Follow-up 提示：** 面试官可能追问 "expanding window vs sliding window 怎么选？"、"如果某个时间段 fraud 特别多（如假日季），怎么处理？"
