# STA-GT Paper Reading: Transaction Fraud Detection via Spatial-Temporal-Aware Graph Transformer

Paper: [Transaction Fraud Detection via Spatial-Temporal-Aware Graph Transformer](https://arxiv.org/pdf/2307.05121)  
Authors: Yue Tian, Guanjun Liu  
Topic: transaction fraud detection, heterogeneous graph neural network, temporal encoding, graph transformer

---

## 1. One-Sentence Summary / 一句话总结

English:

STA-GT models transactions as a heterogeneous temporal graph, uses graph neural networks to capture local relation-based fraud patterns, and then uses Transformer self-attention to capture global long-range dependencies between transactions.

中文：

STA-GT 把交易建模成一个带时间信息的异构图，先用图神经网络捕捉局部关系型欺诈模式，再用 Transformer 自注意力捕捉交易之间的全局远距离依赖。

---

## 2. Problem Background / 问题背景

### English

The paper studies transaction fraud detection. Each transaction is represented as a node. If two transactions share a meaningful attribute, such as the same IP address, MAC address, or device identifier, an edge is created between them.

The task is binary node classification:

```text
y_i = 0: legitimate transaction
y_i = 1: fraudulent transaction
```

Fraud detection is naturally relational. Fraudsters often reuse infrastructure, devices, accounts, IPs, payment channels, or behavioral patterns. A single transaction may look normal in isolation, but it may become suspicious when connected to other transactions.

For example:

```text
Transaction A: normal-looking amount, normal-looking user profile
Transaction B: same device as A, already labeled fraudulent
Transaction C: same IP as A, occurred within seconds

Even if A looks normal alone, its graph neighborhood makes it suspicious.
```

### 中文

这篇论文研究的是交易欺诈检测。每一笔交易被看作图上的一个节点。如果两笔交易共享某种重要属性，例如相同 IP、相同 MAC 地址、相同设备 ID，就在它们之间建立一条边。

任务是二分类节点预测：

```text
y_i = 0：正常交易
y_i = 1：欺诈交易
```

欺诈检测天然适合用图建模。欺诈者经常复用基础设施、设备、账户、IP、支付通道或行为模式。单笔交易单独看可能很正常，但放到关系网络里可能变得非常可疑。

例如：

```text
交易 A：金额正常，用户画像正常
交易 B：和 A 使用同一设备，且 B 已经被标记为欺诈
交易 C：和 A 使用同一 IP，并且几秒内发生

即使 A 单独看不明显异常，它的图邻居也会提供强欺诈信号。
```

---

## 3. Why Existing Methods Are Not Enough / 为什么已有方法不够

### English

The paper argues that existing methods have two main limitations.

First, many models do not represent time precisely enough. Fraud often happens in bursts. Two transactions sharing the same device are much more suspicious if they happen within seconds than if they happen months apart.

Some previous methods split transactions into time windows, but this has drawbacks:

```text
1. Fine-grained time differences inside one window are lost.
2. Transactions in different windows may not interact well.
3. The model depends heavily on how the window size is chosen.
```

Second, ordinary GNNs are mostly local. A GNN layer aggregates one-hop neighbors. To reach far-away useful nodes, the model needs many GNN layers. But deep GNNs can suffer from over-smoothing, where node embeddings become too similar and lose discriminative power.

The paper's solution is:

```text
Use GNNs for local graph structure.
Use temporal encoding for transaction time.
Use Transformer attention for global transaction interaction.
```

### 中文

论文认为已有方法主要有两个不足。

第一，很多模型对时间信息建模不够精细。欺诈行为常常在短时间内集中爆发。两笔交易如果共享同一个设备，并且相隔几秒发生，会比相隔几个月发生更可疑。

一些已有方法会把交易切成时间窗口，但这种做法有问题：

```text
1. 同一个时间窗口内部的细粒度时间差会丢失。
2. 不同时间窗口之间的交易交互不充分。
3. 模型效果会强依赖时间窗口大小的选择。
```

第二，普通 GNN 主要是局部聚合。一层 GNN 聚合一跳邻居。如果一个有用节点距离目标节点很远，就需要堆很多层 GNN。但 GNN 太深容易出现 over-smoothing，即节点表示变得过于相似，区分能力下降。

论文的解决思路是：

```text
用 GNN 学习局部图结构；
用时间编码表示交易发生时间；
用 Transformer attention 学习全局交易交互。
```

---

## 4. Graph Formulation / 图建模方式

### English

The paper defines a multi-relation transaction graph:

```text
G = {V, {E_r}_{r=1}^R, X, Y}
```

Where:

```text
V: transaction nodes
E_r: edges of relation type r
R: number of relation types
X: transaction feature matrix
Y: transaction labels
```

Examples of relation types:

```text
Relation 1: same IP
Relation 2: same MAC address
Relation 3: same device identifier
Relation 4: same browser/device fingerprint
```

This is a heterogeneous or multi-relation graph because different edges represent different semantics. Sharing a MAC address may mean something different from sharing an IP address.

### 中文

论文定义了一个多关系交易图：

```text
G = {V, {E_r}_{r=1}^R, X, Y}
```

其中：

```text
V：交易节点
E_r：第 r 种关系类型下的边
R：关系类型数量
X：交易特征矩阵
Y：交易标签
```

关系类型的例子：

```text
关系 1：相同 IP
关系 2：相同 MAC 地址
关系 3：相同设备 ID
关系 4：相同浏览器或设备指纹
```

这是一个异构图或多关系图，因为不同边代表不同语义。共享 MAC 地址和共享 IP 地址的含义并不一样。

---

## 5. Model Architecture / 模型结构

STA-GT has five major parts:

```text
1. Attribute-driven embedding
2. Temporal encoding
3. Spatial dependency learning with heterogeneous GNN
4. Relation-level attention and inter-layer fusion
5. Transformer-based global information learning
```

STA-GT 主要包含五个部分：

```text
1. 属性驱动嵌入
2. 时间编码
3. 异构 GNN 学习空间依赖
4. 关系级注意力与层间融合
5. 基于 Transformer 的全局信息学习
```

---

## 6. Attribute-Driven Embedding / 属性驱动嵌入

### English

The model first transforms raw transaction features into an initial hidden representation:

```text
h_i^0 = sigma(x_i W_1)
```

Here:

```text
x_i: raw feature vector of transaction i
W_1: learnable projection matrix
sigma: nonlinear activation function
h_i^0: initial node embedding
```

This step is important because graph structure alone is not enough. A transaction's own attributes, such as amount, merchant category, user behavior, device information, and historical statistics, still contain useful fraud signals.

### 中文

模型首先把原始交易特征映射成初始隐表示：

```text
h_i^0 = sigma(x_i W_1)
```

其中：

```text
x_i：第 i 笔交易的原始特征
W_1：可学习的投影矩阵
sigma：非线性激活函数
h_i^0：初始节点表示
```

这一步很重要，因为单靠图结构是不够的。交易自身属性，例如金额、商户类别、用户行为、设备信息、历史统计特征，本身也包含欺诈信号。

---

## 7. Temporal Encoding / 时间编码

### English

The paper adds a time representation to each transaction. It uses sinusoidal-style encoding, similar to positional encoding in Transformers:

```text
Base(t, 2i)     = sin(t / 10000^(2i / d))
Base(t, 2i + 1) = cos(t / 10000^((2i + 1) / d))
TE(t) = Linear(Base(t))
```

Then the temporal embedding is added to the node representation:

```text
h_i^{0,t} = h_i^0 + TE(t_i)
```

Intuition:

```text
Same device + 5 seconds apart  -> highly suspicious
Same device + 5 months apart   -> weaker signal
```

The edge relation tells the model that two transactions are connected. The temporal encoding tells the model when the transactions happened.

### 中文

论文给每个交易节点加入时间表示。它使用类似 Transformer positional encoding 的正弦和余弦时间编码：

```text
Base(t, 2i)     = sin(t / 10000^(2i / d))
Base(t, 2i + 1) = cos(t / 10000^((2i + 1) / d))
TE(t) = Linear(Base(t))
```

然后把时间编码加到节点表示上：

```text
h_i^{0,t} = h_i^0 + TE(t_i)
```

直观理解：

```text
相同设备 + 相隔 5 秒发生  -> 非常可疑
相同设备 + 相隔 5 个月发生 -> 信号较弱
```

边关系告诉模型两笔交易是否有关联。时间编码告诉模型这些交易什么时候发生。

---

## 8. Spatial Dependency with Heterogeneous GNN / 用异构 GNN 学习空间依赖

### English

For each relation type, the model aggregates information from neighboring transactions. Unlike a simple GCN, STA-GT considers both:

```text
1. Neighbor information
2. Difference between the target node and neighbor nodes
```

This is useful because fraud can be detected through similarity and contrast.

Similarity example:

```text
Target transaction shares many features with known fraudulent transactions.
```

Difference example:

```text
Target transaction is connected to normal-looking nodes but has abnormal amount or timing.
```

The relation-specific representation can be understood as:

```text
h_{i,r}^l = relation-specific aggregation for node i under relation r at layer l
```

### 中文

对每一种关系类型，模型都会聚合邻居交易的信息。和简单 GCN 不同，STA-GT 同时考虑：

```text
1. 邻居节点信息
2. 目标节点和邻居节点之间的差异
```

这对欺诈检测有意义，因为欺诈信号既可能来自相似性，也可能来自对比差异。

相似性例子：

```text
目标交易和已知欺诈交易共享大量特征。
```

差异性例子：

```text
目标交易连接到一些看似正常的节点，但金额或时间模式明显异常。
```

可以把关系特定表示理解为：

```text
h_{i,r}^l = 第 l 层中，节点 i 在关系 r 下聚合得到的表示
```

---

## 9. Relation-Level Attention / 关系级注意力

### English

Different relation types have different importance. In fraud detection, sharing a strong device fingerprint may be more informative than sharing a broad IP address.

STA-GT learns attention weights over relation types:

```text
h_i^l = sum_r alpha_r^l * h_{i,r}^l
```

Where:

```text
h_{i,r}^l: node i representation under relation r
alpha_r^l: learned importance of relation r at layer l
h_i^l: fused node representation at layer l
```

This lets the model learn which relation types are more useful instead of treating all edge types equally.

### 中文

不同关系类型的重要性不同。在欺诈检测中，共享强设备指纹可能比共享一个公共 IP 更有信息量。

STA-GT 对不同关系类型学习注意力权重：

```text
h_i^l = sum_r alpha_r^l * h_{i,r}^l
```

其中：

```text
h_{i,r}^l：节点 i 在关系 r 下的表示
alpha_r^l：第 l 层中关系 r 的重要性权重
h_i^l：第 l 层融合后的节点表示
```

这样模型可以自动学习哪些关系更重要，而不是把所有边类型等同对待。

---

## 10. Inter-Layer Fusion / 层间融合

### English

Instead of only using the last GNN layer, STA-GT combines representations from multiple layers:

```text
h_i^t = COMBINE(h_i^{1,t}, h_i^{2,t}, ..., h_i^{L,t})
```

This matters because different GNN layers capture different neighborhood ranges:

```text
Layer 1: immediate neighbors
Layer 2: two-hop neighbors
Layer L: broader graph context
```

Using multiple layers helps preserve both local and broader information. It can also reduce the risk of relying only on an over-smoothed final layer.

### 中文

STA-GT 不只使用最后一层 GNN 表示，而是融合多层表示：

```text
h_i^t = COMBINE(h_i^{1,t}, h_i^{2,t}, ..., h_i^{L,t})
```

原因是不同 GNN 层捕捉的邻域范围不同：

```text
第 1 层：直接邻居
第 2 层：两跳邻居
第 L 层：更大范围的图上下文
```

融合多层表示可以同时保留局部信息和更广邻域信息，也能降低只依赖最后一层导致 over-smoothing 的风险。

---

## 11. Transformer for Global Information / 用 Transformer 学习全局信息

### English

After the GNN module, STA-GT applies Transformer self-attention to the transaction embeddings.

For hidden matrix `H`:

```text
Q = H W_Q
K = H W_K
V = H W_V

Attention(H) = softmax(QK^T / sqrt(d_k)) V
```

Why this helps:

```text
GNNs are strong at local graph aggregation.
Transformers are strong at global pairwise interaction.
```

A suspicious transaction may be related to another transaction that is many hops away in the graph. A normal GNN would need many layers to connect them. A Transformer can directly compute attention between them after the GNN embeddings are produced.

So the final intuition is:

```text
GNN module: learns local spatial-temporal fraud patterns
Transformer module: learns global long-range transaction dependencies
```

### 中文

在 GNN 模块之后，STA-GT 对交易表示使用 Transformer 自注意力。

对于隐藏表示矩阵 `H`：

```text
Q = H W_Q
K = H W_K
V = H W_V

Attention(H) = softmax(QK^T / sqrt(d_k)) V
```

这样做的原因是：

```text
GNN 擅长局部图聚合；
Transformer 擅长全局两两交互。
```

某个可疑交易可能和图中距离很远的另一个交易有关。普通 GNN 需要堆很多层才能让它们交互，而 Transformer 可以在 GNN 生成节点表示后，直接让不同交易之间通过 attention 建立联系。

最终直觉是：

```text
GNN 模块：学习局部时空欺诈模式
Transformer 模块：学习全局远距离交易依赖
```

---

## 12. Prediction and Loss / 预测与损失函数

### English

The final node representation is passed into an MLP classifier:

```text
p_i = sigmoid(MLP(z_i))
```

The model is trained with binary cross-entropy loss:

```text
L = - sum_i [y_i log(p_i) + (1 - y_i) log(1 - p_i)]
```

The output `p_i` is the predicted fraud probability for transaction `i`.

Note: the paper's recall formula appears to contain a typo. Standard recall is:

```text
Recall = TP / (TP + FN)
```

### 中文

最终节点表示会进入 MLP 分类器：

```text
p_i = sigmoid(MLP(z_i))
```

模型使用二分类交叉熵损失：

```text
L = - sum_i [y_i log(p_i) + (1 - y_i) log(1 - p_i)]
```

输出 `p_i` 表示第 `i` 笔交易是欺诈交易的预测概率。

注意：论文中的 Recall 公式疑似有笔误。标准 Recall 应该是：

```text
Recall = TP / (TP + FN)
```

---

## 13. Experiments / 实验

### English

The paper evaluates STA-GT on one private dataset and one public dataset.

Private dataset:

```text
Scale: about 5.2 million transactions
Time: 2016 and 2017
Labels: manually labeled by bank investigators
Relations: same IP, same MAC
```

Public TC dataset:

```text
Transactions: 160,764
Fraudulent transactions: 44,982
Legitimate transactions: 115,782
Relations: IP, MAC, device1, device2
```

Baselines include:

```text
GCN
GraphSAGE
GAT
CARE-GNN
SSA
RGCN
HAN
FRAUDRE
```

Metrics:

```text
Recall
F1
AUC
```

### 中文

论文在一个私有数据集和一个公开数据集上评估 STA-GT。

私有数据集：

```text
规模：约 520 万笔交易
时间：2016 年和 2017 年
标签：由银行调查员人工标注
关系：相同 IP、相同 MAC
```

公开 TC 数据集：

```text
交易数量：160,764
欺诈交易：44,982
正常交易：115,782
关系：IP、MAC、device1、device2
```

对比方法包括：

```text
GCN
GraphSAGE
GAT
CARE-GNN
SSA
RGCN
HAN
FRAUDRE
```

评价指标：

```text
Recall
F1
AUC
```

---

## 14. Results Interpretation / 结果解读

### English

STA-GT generally performs best or near-best across reported settings.

On the private dataset, STA-GT improves recall in several prediction settings. For example:

```text
PR1 Recall: STA-GT 86.4, FRAUDRE 82.6
PR2 Recall: STA-GT 87.3, CARE-GNN 86.2
PR4 Recall: STA-GT 93.4, FRAUDRE 88.3
PR5 Recall: STA-GT 90.8, GAT 86.1
```

On the public TC dataset, the gains are also clear:

```text
TC12 Recall: STA-GT 72.7, CARE-GNN 62.8
TC23 Recall: STA-GT 81.9, FRAUDRE 77.6
TC34 Recall: STA-GT 81.5, CARE-GNN 66.5
```

The result supports the paper's main hypothesis:

```text
Fraud detection benefits from combining temporal signals,
heterogeneous graph relations, and global attention.
```

### 中文

STA-GT 在大多数实验设置下取得最好或接近最好的结果。

在私有数据集上，STA-GT 在多个预测设置中提升了 Recall。例如：

```text
PR1 Recall: STA-GT 86.4, FRAUDRE 82.6
PR2 Recall: STA-GT 87.3, CARE-GNN 86.2
PR4 Recall: STA-GT 93.4, FRAUDRE 88.3
PR5 Recall: STA-GT 90.8, GAT 86.1
```

在公开 TC 数据集上，提升也比较明显：

```text
TC12 Recall: STA-GT 72.7, CARE-GNN 62.8
TC23 Recall: STA-GT 81.9, FRAUDRE 77.6
TC34 Recall: STA-GT 81.5, CARE-GNN 66.5
```

这些结果支持论文的核心假设：

```text
欺诈检测需要同时利用时间信号、异构图关系和全局 attention。
```

---

## 15. Key Contributions / 核心贡献

### English

The paper's contribution is not a single isolated trick. It is the combination of several components that match the nature of fraud detection:

```text
1. Heterogeneous transaction graph
2. Temporal encoding
3. Relation-aware graph aggregation
4. Relation-level attention
5. Inter-layer GNN fusion
6. Transformer global self-attention
```

Each component addresses a practical fraud detection issue:

```text
Heterogeneous graph: transactions are connected by different relation types.
Temporal encoding: suspiciousness depends on timing.
Relation attention: not all relations are equally reliable.
Layer fusion: local and broader neighborhoods both matter.
Transformer: useful signals may be far away in the graph.
```

### 中文

这篇论文的贡献不是某一个孤立技巧，而是把多个符合欺诈检测特点的模块组合起来：

```text
1. 异构交易图
2. 时间编码
3. 关系感知图聚合
4. 关系级注意力
5. 多层 GNN 融合
6. Transformer 全局自注意力
```

每个组件都对应一个实际欺诈检测问题：

```text
异构图：交易之间存在不同类型关系。
时间编码：可疑程度和发生时间强相关。
关系注意力：不同关系的可靠性不同。
层间融合：局部邻域和更广邻域都重要。
Transformer：有用信号可能在图中距离很远。
```

---

## 16. Critical Thinking / 批判性理解

### English

The model idea is reasonable, but several issues should be considered.

First, scalability is a concern. Full Transformer self-attention has quadratic complexity with respect to the number of nodes in the attention set. For millions of transactions, a naive global Transformer would be expensive. A production system would likely need sampling, batching, subgraph attention, approximate attention, or time-windowed attention.

Second, the private dataset is not reproducible. The large-scale evidence is useful but cannot be independently verified.

Third, temporal leakage is a major risk in fraud detection. The paper uses earlier periods for training and later periods for testing, which is directionally correct. However, feature engineering and graph construction must also avoid using future information.

Fourth, the ablation evidence in the available version is limited. The architecture has several components, so stronger ablations would help answer:

```text
How much does temporal encoding help?
How much does the Transformer help over GNN alone?
How much does relation-level attention help?
Is inter-layer fusion necessary?
```

Fifth, online deployment is not fully discussed. In real fraud systems, the model must handle new transactions continuously, update graph connections quickly, and make low-latency predictions.

### 中文

这个模型思路是合理的，但也有一些需要注意的问题。

第一，可扩展性是一个问题。完整 Transformer self-attention 对节点数量是二次复杂度。如果面对百万级交易，朴素全局 Transformer 会非常昂贵。真实生产系统很可能需要采样、batching、子图 attention、近似 attention 或基于时间窗口的 attention。

第二，私有数据集不可复现。大规模实验有参考价值，但外部研究者无法独立验证。

第三，时间泄漏是欺诈检测里的重大风险。论文使用较早时间段训练、较晚时间段测试，这个方向是正确的。但特征工程和图构造也必须避免使用未来信息。

第四，当前版本中的消融实验信息有限。模型包含多个组件，因此更充分的 ablation 可以回答：

```text
时间编码到底贡献多大？
Transformer 相比只用 GNN 提升多少？
关系级注意力贡献多少？
层间融合是否必要？
```

第五，论文没有充分讨论在线部署。真实欺诈系统需要持续处理新交易、快速更新图连接，并在低延迟要求下完成预测。

---

## 17. Interview-Level Explanation / 面试讲法

### English

If asked to explain this paper in an interview, a strong answer could be:

```text
This paper treats fraud detection as node classification on a heterogeneous transaction graph.
Each transaction is a node, and edges represent shared entities like IP, MAC, or device.
The key problem is that fraud is both relational and temporal.

STA-GT first embeds transaction attributes, adds temporal encoding,
then applies relation-aware GNN aggregation to capture local neighborhood fraud patterns.
Because GNNs are limited in long-range dependency modeling and can suffer from over-smoothing,
the model further uses Transformer self-attention to capture global dependencies among transactions.

The main benefit is that the model combines local graph structure, time information,
relation importance, and global attention in one architecture.
```

### 中文

如果面试中被问到这篇论文，可以这样回答：

```text
这篇论文把欺诈检测建模成异构交易图上的节点分类问题。
每笔交易是一个节点，边表示共享 IP、MAC、设备等实体关系。
核心问题是欺诈行为既有关系性，也有时间性。

STA-GT 先对交易属性做 embedding，再加入时间编码，
然后用关系感知 GNN 聚合局部邻居中的欺诈模式。
由于 GNN 对远距离依赖建模能力有限，而且深层 GNN 容易 over-smoothing，
模型进一步使用 Transformer self-attention 捕捉交易之间的全局依赖。

它的主要优势是把局部图结构、时间信息、关系重要性和全局 attention 统一到一个模型里。
```

---

## 18. How This Relates to Anti-Fraud System Design / 和反欺诈系统设计的关系

### English

For an anti-fraud interview or system design discussion, this paper gives a useful modeling pattern:

```text
Raw transaction features are not enough.
Graph relations are important.
Time ordering is important.
Different relation types need different weights.
Long-range suspicious patterns should be modeled.
```

In a real system, this could map to:

```text
Online features: amount, user profile, merchant, velocity statistics
Graph features: shared device, IP, account, card, address, payment instrument
Temporal features: transaction time, recent frequency, burst behavior
Model: GNN or graph-enhanced model
Serving: precomputed embeddings + real-time feature updates
```

### 中文

对于反欺诈面试或系统设计讨论，这篇论文提供了一个很有用的建模思路：

```text
原始交易特征不够；
图关系很重要；
时间顺序很重要；
不同关系类型需要不同权重；
远距离可疑模式也应该被建模。
```

在真实系统中，可以对应到：

```text
在线特征：金额、用户画像、商户、频率统计
图特征：共享设备、IP、账户、银行卡、地址、支付工具
时间特征：交易时间、近期频率、爆发行为
模型：GNN 或图增强模型
服务：预计算 embedding + 实时特征更新
```

---

## 19. Practical Production Considerations / 生产落地注意点

### English

If implementing a similar model in production, key design questions include:

```text
1. How to construct graph edges without future leakage?
2. How often should graph embeddings be refreshed?
3. How to serve predictions for brand-new transactions?
4. How to handle high-degree nodes such as public IPs?
5. How to keep latency low enough for transaction authorization?
6. How to explain model decisions to risk analysts?
```

High-degree nodes are especially important. A public IP or shared network can connect many unrelated users, creating noisy edges. Production systems often need edge filtering, degree caps, relation weights, or risk-aware sampling.

### 中文

如果要把类似模型落地到生产系统，需要考虑几个关键问题：

```text
1. 如何构图才能避免未来信息泄漏？
2. 图 embedding 多久刷新一次？
3. 新交易如何实时预测？
4. 如何处理公共 IP 这类高出度节点？
5. 如何满足交易授权场景的低延迟要求？
6. 如何向风控分析师解释模型判断？
```

高出度节点尤其重要。公共 IP 或共享网络可能连接大量无关用户，产生很多噪声边。生产系统通常需要边过滤、degree cap、关系权重或风险感知采样。

---

## 20. Final Takeaway / 最终理解

English:

STA-GT is best understood as a hybrid fraud detection architecture:

```text
Transaction attributes provide local evidence.
Graph relations provide relational evidence.
Temporal encoding provides time-sensitive evidence.
GNN aggregation provides neighborhood evidence.
Transformer attention provides global evidence.
```

中文：

STA-GT 可以理解为一个混合式欺诈检测架构：

```text
交易属性提供局部证据；
图关系提供关系证据；
时间编码提供时间敏感证据；
GNN 聚合提供邻域证据；
Transformer attention 提供全局证据。
```

The main lesson is:

```text
Fraud detection should not only ask "what does this transaction look like?"
It should also ask "who and what is this transaction connected to, and when did those connections happen?"
```

核心启发是：

```text
欺诈检测不应该只问“这笔交易长什么样？”
还应该问“这笔交易和谁有关联？这些关联发生在什么时候？”
```
