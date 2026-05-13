# 基于 Transformer 的区块链欺诈检测系统设计（Fraud Detection with Transformers）

> **文档定位：** OKX Anti-Fraud AI Engineer 面试准备系列 File 4 of 4。
>
> **目标读者：** 有 ML/NLP 工作经验、正在准备 OKX Senior Staff AI Engineer Anti-Fraud 岗位面试的工程师。本文是系列中最 OKX-specific 的一篇，聚焦"如何用 Transformer 及其变体构建一个从数据到部署的端到端区块链欺诈检测系统"，同时覆盖生产部署挑战、可解释性、和 LangGraph 集成。
>
> **配套阅读：** `01_*.md`（blockchain data primer）、`02_*.md`（transformer 架构深度解析）、`03_*.md`（graph ML on blockchain）。

---

## 1. 问题定义（Problem Framing）

---

### 1.1 欺诈检测的核心挑战

在区块链欺诈检测场景下，ML 工程师面对的挑战远比标准 text classification 复杂。以下五类挑战贯穿系统设计的每一个决策：

| 挑战 | 具体表现 | 对模型设计的影响 |
|------|---------|----------------|
| **标签稀缺** | 已知欺诈地址占全链地址不足 0.1%，且大量欺诈行为无法被及时确认和标注 | 必须依赖无监督预训练、半监督学习、主动学习来应对标签不足 |
| **类别不平衡（1:1000）** | 正常交易远多于欺诈交易，朴素训练会导致模型退化为"全部预测 normal" | 需要 Focal Loss / class weights / 阈值调整等专门技术 |
| **对抗性漂移** | 欺诈者会持续调整行为模式以规避检测，导致训练集和线上数据分布不断偏离 | 需要定期 retrain、概念漂移监控、对抗训练 |
| **标签噪声** | 部分"欺诈"标签是基于规则误判，"正常"标签包含尚未暴露的欺诈 | 需要 label smoothing、PU learning、人工 review 闭环 |
| **实时约束** | CEX 充提拦截要求在 <100ms 内给出结论，全 GNN 推理往往需要秒级 | 需要分层决策、模型蒸馏、量化、缓存策略 |

### 1.2 三级任务分解

针对上述挑战，一个成熟的系统不会用单一模型 "all-in-one" 解决所有问题，而是按 **粒度** 和 **延迟要求** 分层：

**Level 1：交易级别（Transaction-level）**
- 粒度：单笔 tx
- 模型：规则引擎 + 轻量级 XGBoost / 简单特征
- 目标：<5ms 内过滤明显恶意 tx（与 sanctioned address 交互、已知 mixer、OFAC 命中）
- 召回为王，宁可误伤也不漏过

**Level 2：地址级别（Address-level）**
- 粒度：单个 address 的行为序列（最近 N 笔 tx）
- 模型：Transformer 序列模型 + 表格特征 Transformer
- 目标：<100ms 内对地址给出风险评分
- 捕捉 temporal behavior pattern（layering、structuring、velocity abuse）

**Level 3：团伙级别（Cluster-level）**
- 粒度：address cluster / 交易关系子图
- 模型：GNN + Graph Transformer
- 目标：异步批量运行（分钟级），识别协同欺诈、洗钱网络、刷量 cluster
- 输出作为 Level 2 的上游特征（邻居风险分）和 Level 2 无法检测到的团伙告警

---

## 2. 特征工程（Feature Engineering）

---

### 2.1 地址级聚合特征

以 **address** 为粒度，在给定时间窗口内（通常 as-of-timestamp 快照）聚合出 flat feature vector 喂给树模型或表格 Transformer：

| 特征类别 | 示例特征 | 欺诈信号 |
|---------|---------|---------|
| **交易量** | `tx_count_7d`、`tx_count_24h`、`tx_count_1h`、`unique_active_days_30d` | 短时间内 `tx_count_1h` 激增（bot / 批量转账脚本）；`unique_active_days_30d` 为 1（新建一次性地址） |
| **时间** | `account_age_days`、`days_since_last_tx`、`tx_burstiness`（Fano factor）、`night_ratio`（23-5 点交易占比） | `account_age_days < 7` + 大额活动；`night_ratio > 0.8`（时区异常 bot）；`tx_burstiness` 极低（clock-like bot） |
| **金额分布** | `median_tx_value_usd`、`max_tx_value_usd`、`round_amount_ratio`（整数金额占比）、`value_concentration`（top-1 tx / total）、`total_out_usd_7d` | `round_amount_ratio > 0.9`（脚本生成）；`value_concentration > 0.95`（单笔大额清仓）；`total_out_usd_7d / balance_usd > 0.99`（快速 pass-through） |
| **对手方** | `unique_counterparty_count`、`repeat_counterparty_ratio`、`known_fraud_neighbor_count`、`sanctioned_1hop_count` | `known_fraud_neighbor_count > 0` 直接触发规则；`unique_counterparty_count` 极高但 `repeat_counterparty_ratio` 极低 = mixer fan-out |
| **合约交互** | `dex_swap_count_7d`、`bridge_use_count_30d`、`mixer_interaction_flag`、`token_approve_count_to_unknown`、`unique_contract_count` | `mixer_interaction_flag = 1` 是高优先级 signal；`bridge_use_count_30d > 3` + 高金额 = 跨链洗钱嫌疑；`token_approve_count_to_unknown > 5` = 钓鱼受害高风险 |
| **图特征** | `pagerank`、`in_degree`、`out_degree`、`clustering_coefficient`、`shortest_path_to_known_fraud`、`community_fraud_ratio` | `pagerank` 高但 `shortest_path_to_known_fraud <= 2` = 洗钱关键中转节点；`clustering_coefficient > 0.5` + 少量对手方 = wash trade cluster；`community_fraud_ratio > 0.3` = 高风险社区成员 |

**工程注意事项：**
- 所有 window-based feature 必须在 **as-of-timestamp** 时刻计算（point-in-time correctness），避免 label leakage
- `pagerank` 等图特征通常在离线批量计算后存储到 feature store，线上查询而非实时计算
- `known_fraud_neighbor_count` 依赖实时更新的黑名单库，需要增量维护

---

### 2.2 序列特征（Per-Transaction Token）

在序列模型中，每笔 tx 被编码为一个 **multi-field token**，包含以下 9 个字段：

```python
tx_token = {
    # 连续特征（continuous fields）
    "log_amount":             log10(tx_value_usd + 1),    # 金额对数，压缩量纲
    "hour_of_day_sin":        sin(2π × hour / 24),         # 时间周期性编码
    "hour_of_day_cos":        cos(2π × hour / 24),         # （sin/cos 避免边界跳变）
    "days_since_account_creation": (tx_timestamp - account_first_seen) / 86400,
    "log_gas_fee":            log10(gas_fee_usd + 1),      # gas 行为信号
    "delta_t_since_last_tx":  log10(seconds_since_prev_tx + 1),  # 时间间隔

    # 离散特征（categorical fields）
    "direction":              0 if tx_is_outgoing else 1,  # 资金方向（以 target address 视角）
    "counterparty_type":      one of {EOA, DEX, Bridge, Mixer, CEX, NFT, Unknown},  # 对手方类型
    "is_contract_call":       0 or 1,                      # 是否调用合约（含函数 selector 类别）
    "token_type":             one of {ETH, USDT, USDC, ERC20_Other, ERC721, Native_Other},
}
```

**编码与融合方式：**

- **连续特征**：归一化（z-score 或 min-max）后通过一个小型 MLP（2 层，hidden=64）映射到 $d$ 维向量
- **离散特征**：每个字段独立学一个 embedding table，查表得到 $d$ 维向量
- **Token 融合**：把所有字段的 embedding 相加（如 BERT 的 segment embedding 加法）或 concat 后线性投影到 $d$ 维，得到最终 token embedding $x_{t} \in \mathbb{R}^{d}$

**为什么用 sin/cos 编码小时而不是 one-hot：**

`hour_of_day` 是周期性变量，`23` 和 `0` 在语义上相邻但 integer encoding 距离最远。sin/cos 双通道编码保证 `23` 和 `0` 的向量距离趋近于 0，和 NLP 里 positional encoding 的思想一致。

**为什么用 `delta_t` 而不是绝对时间戳：**

绝对时间戳难以泛化（不同年份的 address 时间范围不同），而相对时间间隔 `delta_t` 是行为节律的直接表达。欺诈 bot 的 `delta_t` 分布极其规律（方差接近 0），正常用户的 `delta_t` 方差大。

---

## 3. 建模方案一：序列 Transformer（Sequence Modeling）

---

### 3.1 架构总览

```
输入: 地址最近 N 笔交易
  [CLS]  tx_1  tx_2  tx_3  ...  tx_N
    ↓      ↓     ↓     ↓           ↓
  [Token Embeddings (multi-field fusion)]
    +
  [Positional Encoding (index + Δt)]
    ↓
  BERT Encoder
    L=6 层  H=256 维  A=8 头
    （每层: Multi-Head Self-Attention → Add&Norm → FFN → Add&Norm）
    ↓
  [CLS] 位置的最终 hidden state
    ↓
  Linear(256 → 1) + Sigmoid
    ↓
  P(fraud)
```

参数量约 ~6M，可在单 GPU 完成 batch inference，满足 <100ms 的充提风控要求。

---

### 3.2 预训练（MLM Pre-training）

**为什么预训练是核心，不是可选项：**

链上 labeled fraud 数据极为稀缺（全链亿级地址中已确认欺诈地址可能不足 10 万）。直接在少量有标签数据上 fine-tune 一个 6 层 transformer 极易过拟合。预训练通过在大量 **无标签** 交易序列上学习行为分布，解决了标签稀缺问题。

**预训练任务：Masked Transaction Modeling（MTM）**

类比 BERT 的 MLM，随机 mask 掉 15% 的 tx token，要求模型从上下文预测被 mask 掉的 tx 的 `counterparty_type`、`token_type`、`is_contract_call` 等离散字段（`direction`、`log_amount` 等连续字段用 MSE loss 重建）。

**预训练数据：** 可从公链上无差别采样最近 1 年内活跃地址的 tx 序列，不需要任何标注。亿级 tx 的数据量足以使 transformer 学到通用的区块链行为表征。

---

### 3.3 Fine-tuning：Focal Loss

针对 1:1000 的类别不平衡，使用 **Focal Loss** 替代标准 Binary Cross-Entropy：

$$FL(p_{t}) = -\alpha_{t}(1-p_{t})^{\gamma}\log(p_{t})$$

其中：
- $p_{t}$ 是模型对正确类别的预测概率（欺诈样本用预测为欺诈的概率，正常样本用预测为正常的概率）
- $\alpha_{t}$ 是类别权重（通常对 minority class 设置 $\alpha > 0.5$，如 $\alpha = 0.75$）
- $\gamma = 2$ 是 focusing parameter：当 $p_{t} \to 1$（模型已确信的样本），$(1-p_{t})^{2} \to 0$，loss 被大幅压低；当 $p_{t} \to 0$（模型不确定的困难样本），loss 权重接近标准 CE

**$\gamma=2$ 的直觉：** 1:1000 的不平衡会导致大量 easy negative（明显的正常交易）主导梯度更新，模型倾向于"无脑预测 normal"。Focal Loss 的 $(1-p_{t})^{\gamma}$ 项把 easy sample 的梯度贡献几乎压为 0，迫使模型把注意力集中在难以分类的边界样本上——而这些边界样本中，欺诈行为恰恰是最需要被捕捉的。

---

### 3.4 工程应对策略

| 工程问题 | 具体解法 |
|---------|---------|
| **序列太长**（活跃地址 tx 数 > 1000） | 用 Longformer 的 sliding window attention（$O(n)$ 复杂度代替标准 $O(n^{2})$）；或截取最近 K 笔 + 全局摘要 token 覆盖远期行为 |
| **序列太短**（新地址只有 1-3 笔 tx） | Padding 到最小长度 + attention mask 屏蔽 padding 位置；同时添加 `account_age_days` 作为全局 meta-feature 输入 `[CLS]` |
| **时间信息失真**（不同 address 的时间跨度差异巨大） | 使用 $\Delta t$（相对时间间隔）而非绝对时间戳作为位置信号；$\Delta t$ 用对数压缩消除量纲差异 |
| **实时推理延迟**（要求 <50ms） | INT8 量化 + ONNX Runtime 导出；或蒸馏到更小的 2-3 层 student model；序列 embedding 可预计算并缓存（增量更新只在有新 tx 时触发） |

---

## 4. 建模方案二：表格 Transformer（Tabular Transformer）

---

### 4.1 适用场景

当输入是 **聚合统计特征**（Section 2.1 中的地址级特征）而非原始序列时，传统 XGBoost 对 feature interaction 的捕捉有限。**FT-Transformer（Feature Tokenizer + Transformer）** 把每个 tabular feature 单独 tokenize，通过 self-attention 学到特征之间的 cross-feature interaction，在复杂混合特征（连续 + 离散 + 图特征混合）场景下效果优于 XGBoost。

**架构：**

```
输入: 地址聚合特征向量（50-200 维）
  每个 feature 独立 tokenize → feature token
    ↓
  [CLS]  f_1  f_2  f_3  ...  f_n   ← feature tokens（不是 tx tokens）
    ↓
  Transformer Encoder（L=3, H=128, A=4）
    ↓
  [CLS] → Linear → P(fraud)
```

**Feature Tokenization：** 对连续特征 $x_{i}$，用 `Linear(1 → d)` 投影；对离散特征，用 embedding table 查询。每个 feature 都变成同维度的 token。

---

### 4.2 分层决策策略

三层 cascade 对应不同的精度/延迟 tradeoff：

| 层级 | 模型 | 典型延迟 | 触发条件 | 作用 |
|------|------|---------|---------|------|
| **L1：规则 + XGBoost** | 50 棵树，20 个 hand-crafted feature | <1ms | 所有请求 | 低成本过滤明显欺诈（sanctioned、mixer 1-hop、velocity rule 触发）和明显正常 |
| **L2：FT-Transformer** | 3 层，128 维 | ~5ms | L1 score 在 (0.2, 0.8) 之间的"模糊区域" | 细粒度特征交叉，给出更准确的中间层风险分 |
| **L3：人工审核** | Human-in-the-loop | 分钟级 | L2 score > 0.85 且金额 > 阈值 | 高价值高风险案例，合规和业务联合决策 |

**工程价值：** 约 80% 的请求在 L1 即可解决，只有约 15-20% 需要进入 L2，<1% 需要 L3 人工介入，整体系统平均延迟远低于 L2 单独运行的平均值。

---

## 5. 建模方案三：Graph + Transformer

---

### 5.1 为什么需要 Graph 视角

单地址模型（无论是序列 Transformer 还是表格模型）只能看到 **目标 address 自己的行为**，而许多欺诈模式的 signal 在 **关系层面**：

- **多跳洗钱**：资金通过 5-10 个中间地址在 24h 内转移，每个中间地址单独看行为正常
- **欺诈团伙**：一组 address 受同一实体控制，coordinated action 才是 signal
- **Mixing pattern**：单看 downstream address，无法看到它从 mixer 流出的事实（需要 2-hop 路径信息）

Graph + Transformer 的两阶段设计既保留了序列行为建模能力，又纳入了图结构信息。

---

### 5.2 两阶段融合架构

**Stage 1：图神经网络（GNN）→ 图上下文表征**

使用 **GraphSAGE** 或 **GAT** 在 address transfer graph 上做 K 跳邻域聚合：

```
h_graph = GNN(address_node_features, adjacency)
         ↑ GraphSAGE-3hop 或 GAT-2hop
         ↑ 邻居特征聚合包含: neighbor_type, edge_weight (transfer_volume), edge_direction
```

输出 `h_graph ∈ ℝ^{d_g}` 是该 address 的图上下文表征，编码了"它周围是什么人"。

**Stage 2：序列 Transformer → 行为序列表征**

按 Section 3 描述的序列 Transformer，输出 `h_seq ∈ ℝ^{d_s}`，编码了"它自己做了什么"。

**Stage 3：融合策略**

| 融合方式 | 方法 | 适用场景 |
|---------|------|---------|
| **Concatenation + MLP** | `MLP([h_graph; h_seq])` → P(fraud) | 两路信息相对独立，快速工程化实现 |
| **Cross-Attention** | `h_seq` 作为 query，`h_graph` 的邻居 token 序列作为 key/value，允许序列模型 attend to 图邻居信息 | 图和序列信息高度关联时效果更好，但实现更复杂 |
| **简单加权平均** | $\alpha \cdot \text{score\_graph} + (1-\alpha) \cdot \text{score\_seq}$，$\alpha$ 可学习 | 作为 ensemble baseline，快速验证两路信息的互补性 |

---

### 5.3 标签传播（Label Propagation）

对于图中 **标注稀缺** 的问题，可以从已知欺诈节点出发做 label propagation：

- **初始化：** 已确认欺诈地址的"欺诈分"设为 1.0，未知地址设为 0
- **传播规则：** 每次迭代，每个节点的分数 = `β × 自身初始分 + (1-β) × 邻居分的加权平均`，$\beta \in (0.5, 0.9)$
- **迭代终止：** 分数收敛后，高分未标注地址作为 **soft label 正样本** 参与训练
- **工程实现：** 在 Neo4j 中用 GDS（Graph Data Science Library）的 Label Propagation 算法可以在百亿边规模图上高效执行

**注意：** 标签传播结果是 soft label，训练时用 `label_smoothing` 或为传播标签设置更低的 loss weight，避免把传播噪声当作 ground truth。

---

## 6. 建模方案四：无监督异常检测（Unsupervised Anomaly Detection）

---

### 6.1 适用场景

当面对 **zero-day 欺诈**（全新攻击手法，历史数据中完全没有对应标签）或 **冷启动阶段**（标签数量不足以 fine-tune 监督模型）时，无监督方法是唯一选择。核心思路：让模型学习"正常行为是什么"，偏离正常分布的行为被标记为异常。

### 6.2 Anomaly Transformer Pipeline

```
无标签交易序列
    ↓
训练 Anomaly Transformer
（学习 series-association 与 prior-association 的差异作为异常分数）
    ↓
对全量地址计算 discrepancy score（异常分）
    ↓
高分候选（top 1% 或超过阈值）
    ↓
人工 review → 确认 labels
    ↓
有监督模型的训练数据（闭环）
```

**Anomaly Transformer 核心思想（Xu et al., 2022）：** 正常序列的 attention map 呈现 "association discrepancy" 较小的特征（prior distribution 和 series distribution 接近），而异常点的局部邻域统计与全局统计差异显著。用两者的 KL 散度作为异常分，能在无标签条件下检测序列异常点。

---

### 6.3 四种无监督方法对比

| 方法 | 异常分数来源 | 优势 | 劣势 |
|------|------------|------|------|
| **Transformer AutoEncoder** | 序列重建误差（MSE） | 实现简单，训练稳定；对分布外 pattern 敏感 | 对"难以重建但并非异常"的罕见合法行为 FP 高；重建误差不等于欺诈风险 |
| **Anomaly Transformer** | Prior-series association discrepancy（KL 散度） | 有理论支撑，不依赖重建误差；时间序列上 SOTA 效果 | 对超参数敏感；在极短序列（<10 tx）上效果退化 |
| **TranAD** | Two-phase attention magnification anomaly score | 对多变量时序异常检测效果好；收敛快 | 实现复杂度高；原论文针对工业 IoT，blockchain 序列需要较多 adaptation |
| **Isolation Forest** | 随机特征分割次数（少次 = 异常） | 速度极快（毫秒级）；对全局稀疏异常效果好；无需 sequence 建模 | 无法捕捉序列依赖和时序 pattern；对局部密集异常（如 structuring）效果差 |

**工程推荐：** 生产环境用 Isolation Forest 做第一层（速度快、容易解释），Anomaly Transformer 做第二层（捕捉序列 pattern），两层分数融合后决定是否送 human review。

---

## 7. 生产部署挑战（Production Concerns）

---

### 7.1 概念漂移（Concept Drift）

欺诈者会不断调整策略，导致训练时的特征分布与线上数据产生偏移，是生产中最常见的"模型性能悄悄下降"的原因。

**检测方法：**

- **特征分布监控（PSI）：** 对关键特征（如 `log_amount`、`account_age_days`、`tx_count_7d`）计算 **Population Stability Index**，PSI = $\sum_{i}(A_{i} - E_{i})\ln(A_{i}/E_{i})$。PSI < 0.1 稳定；0.1~0.2 轻微漂移；**PSI > 0.2 触发告警**，需要调查是数据问题还是分布漂移
- **Score distribution shift：** 监控模型输出 P(fraud) 的分布（均值、分位数、方差）是否发生系统性偏移
- **FP/FN 监控：** 结合人工 review 结果，追踪线上 precision 和 recall 的变化趋势（人工 review 确认的"确实欺诈/确实正常"是最可信的 proxy label）

**应对方法：**

- **周期性 retrain：** 每周用最新 2-4 周数据重新训练，保持模型与当前欺诈分布对齐
- **增量 fine-tune：** 对有新标签的数据做 small learning rate fine-tune（<10% 原始 lr），保留预训练知识同时吸收新 pattern
- **对抗训练：** 在训练时加入合成的"规避行为"样本（模拟欺诈者的常见规避策略，如 structuring、延迟、插入噪声 tx），提高模型对 evasion 的鲁棒性

---

### 7.2 可解释性（Explainability）

OKX 作为持牌 CEX，对风控决策有合规层面的可解释性要求（SAR 报告、冻结账户申诉等场景）。

**方法一：Attention Weights 可视化**

最直观的方法：可视化序列中每个 tx 的 attention weight，把 attention 最高的几笔 tx 展示给调查员。

**局限性（重要）：** Jain & Wallace（2019）的研究表明 attention weight 与特征归因之间 **并没有可靠的因果关系**——注意力高的 token 不一定是影响预测最大的 token。因此 attention 只能作为"看哪里"的粗略 hint，不能作为正式的解释依据。

**方法二：Integrated Gradients（推荐）**

Integrated Gradients（Sundararajan et al., 2017）通过沿从 baseline（全零序列）到实际输入的路径积分梯度，给出每个输入 feature 的精确归因：

$$\text{IG}_{i}(x) = (x_{i} - x'_{i}) \times \int_{0}^{1} \frac{\partial F(x' + \alpha(x-x'))}{\partial x_{i}} d\alpha$$

其中 $x'$ 是 baseline 输入，$\alpha \in [0,1]$ 是插值系数，$F(\cdot)$ 是模型输出。满足 **完整性公理**（所有 feature 归因之和等于模型输出与 baseline 输出之差），理论上比 attention 更可靠。

**方法三：SHAP（工程推荐）**

对于表格模型（XGBoost / FT-Transformer），用 **TreeSHAP** / **DeepSHAP** 给出每个 feature 对当前预测的贡献值。SHAP 值的优势：统一框架、满足 Shapley 公理、可做全局 feature importance 分析。

**实际应用：自然语言解释**

将可解释性工具与 LangGraph agent 结合（详见 Section 8）：

1. 提取 attention top-5 最高的 tx（作为"关键证据 tx"）
2. 运行 Integrated Gradients，找到 top-3 最重要的 feature
3. 将上述信息输入 LangGraph agent 的 `generate_report` 节点
4. Agent 生成自然语言解释："该地址在 2024-03-15 向已知混币器地址转账 $50,000（异常点 1），此后在 24h 内向 8 个新建地址分散转账（layering 模式，异常点 2），其 `pagerank` 值为全链前 0.1%（高风险网络中心，异常点 3），综合评分为 0.92（高风险）。"
5. 若调查员反馈"那笔 tx 其实是正常的"，可运行 **counterfactual analysis**（将该 tx 从序列中移除后重新推理，观察分数变化），量化该 tx 对结论的实际影响

---

### 7.3 延迟约束（Latency Constraints）

| 业务场景 | 延迟要求 | 推荐方案 |
|---------|---------|---------|
| **实时拦截**（充值到达即扫描、可疑提现即拦截） | <10ms | 2-3 层蒸馏后的 Distilled Transformer（teacher: 6层256维 → student: 2层64维）+ 规则引擎并行；提前缓存近期活跃地址的序列 embedding |
| **充值/提现 enhanced check** | <100ms | INT8 量化的 BERT-tiny（4层128维）导出为 ONNX，用 ONNX Runtime GPU 推理；表格模型 XGBoost 并行打分 |
| **批量风险评估**（日终全量扫描、案件调查触发） | 无实时要求（分钟级） | 全尺寸 Transformer（6层256维）+ GNN 联合推理；支持使用 graph-based 标签传播扩散结果 |

**延迟优化技术：**

- **知识蒸馏（Knowledge Distillation）：** 用大模型（teacher）输出的 soft probability 训练小模型（student），student 保留 teacher 约 80-90% 的精度，但推理速度快 5-10 倍
- **INT8 量化：** 将 float32 权重量化为 int8，内存占用减半，CPU/GPU 推理速度提升约 2-4 倍，精度损失通常 <1% AUC
- **Attention Head Pruning：** 移除 attention 权重接近均匀分布的"无效"head（通常 40-60% 的 head 可安全移除），减少计算量
- **ONNX Runtime：** 将 PyTorch 模型导出为 ONNX 格式，利用 ONNX Runtime 的算子融合和硬件特定优化（CUDA / CoreML / TensorRT），相比原生 PyTorch 通常有 2-5 倍加速

---

### 7.4 类别不平衡（Class Imbalance）

| 方法 | 原理 | 推荐场景 | 注意事项 |
|------|------|---------|---------|
| **Focal Loss** | 通过 $(1-p_{t})^{\gamma}$ 动态降低 easy sample 的 loss 权重，迫使模型关注困难样本 | **首选**，适用于所有深度学习模型 | $\gamma=2$ 是经验默认值，可在 validation 上调参（范围 1-5） |
| **Class Weights** | 在 loss 中对少数类样本乘以权重因子（如正负比 1:1000 → 正样本权重 1000）等效于过采样 | 作为 Focal Loss 的补充；XGBoost 的 `scale_pos_weight` 参数 | 权重过大会导致训练不稳定，配合 gradient clipping 使用 |
| **SMOTE** | 在特征空间插值生成合成少数类样本 | **不推荐用于链上高维特征**：链上特征分布复杂（长尾、稀疏图特征），SMOTE 插值出的合成样本往往不合理 | 在低维（<50 维）统计特征上可以用，但高维 embedding 上效果差 |
| **阈值调整** | 不改变模型训练，在 inference 时调低判定阈值（如从 0.5 降到 0.2）来提高 recall | 快速调整 precision-recall 平衡；部署后的业务需求变化应对 | 阈值选择应基于 PR 曲线而非 ROC 曲线（更适合不平衡场景） |
| **无监督预训练** | 在无标签数据上预训练，让模型先学到正常行为分布，再 fine-tune 时更容易识别偏差 | **从根本上解决标签稀缺问题**；是上述所有方法的基础设施 | 预训练语料必须干净（避免大量欺诈行为污染"正常行为"学习） |

---

## 8. OKX 特定场景（OKX-Specific Patterns）

---

### 8.1 DeFi 欺诈的特殊性

DeFi 欺诈与传统金融欺诈在几个维度上有根本性不同，必须在系统设计中特别处理：

**原子性（Atomicity）：Flash Loan 的单 tx 完整攻击**

Flash loan 攻击的整个"借-攻击-还"发生在 **一笔 transaction** 内。这意味着：
- 序列模型（基于 address 历史 tx 序列）无法在 tx 之间看到"准备动作"，只能 post-hoc 分析
- 正确的检测角度是 **tx 内 trace 分析**（单 tx 的 internal call sequence），而非地址行为序列
- 特征应包括：`gas_used > 500K`、`internal_call_count > 10`、`flashLoan_event_present`、`price_deviation_in_block > 20%`

**Smart Contract ABI 解码**

很多 DeFi 欺诈行为隐藏在 contract call 的 `input data` 字段里（如 approve 恶意合约、调用 exploit 函数）。Feature pipeline 必须：
1. 维护常见 contract 的 ABI 库（Uniswap、Aave、Compound、Curve 等）
2. 用 4-byte function selector 匹配 + ABI decode 解析出 `function_name` 和 `parameters`
3. 关键信号：`approve(address, uint256)` 中 `amount = 2^256-1`（无限授权给未知地址）；`transfer` 目标是 known drainer

**跨链数据缺口（Cross-chain Data Gaps）**

资金一旦过 bridge 就在当前链上"消失"，目标链地址是全新的——链上 graph 在 bridge 处断裂。应对：
- 维护主要 bridge（Wormhole、LayerZero、Across、Stargate、OKX 自有 X-Chain）的 lock/unlock 事件对，通过 `(timestamp, amount, initiator)` fuzzy match 重建跨链路径
- 将 cross-chain bridge 接触本身作为 risk factor（`bridge_use_count > 3 AND amount > $10K` 触发增强审查）

**MEV 灰色地带**

Sandwich attack、arbitrage bot、liquidation bot 在链上 pattern 与某些欺诈行为非常相似（高 gas、快速资金移动、新地址），但这些行为本身在法律和合规层面的定性取决于管辖权和具体行为。系统设计需要：
- 维护已知 MEV bot 地址白名单（Flashbots、known arbitrage bots），避免误伤
- 在可解释性层面明确标注"疑似 MEV bot 行为"，让人工 review 做最终判断

---

### 8.2 CEX 专属数据优势

OKX 作为 CEX 拥有纯链上分析系统没有的独特数据资产，这是 OKX 欺诈检测的核心竞争力：

| 数据类型 | 在链上不可见 | CEX 内部可见 | 欺诈检测价值 |
|---------|------------|------------|------------|
| **KYC 信息** | 不可见 | 姓名、身份证、手机、邮箱、IP 注册地 | 多账户识别；KYC 信息与风险地址关联 |
| **充提记录（on-chain address ↔ OKX account 映射）** | 链上只能看到地址，看不到交易所账户 | 精确的"哪个链上地址属于哪个 OKX 用户" | **核心优势**：结合链上欺诈 signal 和交易所账户身份，实现 real-person attribution；地址风险分可直接转化为账户风险分 |
| **IP / 设备指纹** | 完全不可见 | 登录 IP 历史、设备 ID、浏览器指纹 | 多账户同 IP/设备聚合；登录地与习惯不符的告警 |
| **OKX 内部交易行为** | CEX 内部撮合不上链 | 现货/合约交易历史、持仓变化、杠杆使用 | 结合链上充提和链下交易，构建更完整的用户行为 profile |

**充提记录的战略价值（最关键）：** 当一个高风险链上地址向 OKX 充值时，OKX 是极少数能知道"这个地址背后是哪个真实账户"的机构之一。这使得：
- 链上 fraud detection 的结论可以立即 **归因到具体账户**（而不是停在"地址 0x..."）
- 同一人的多链行为可以 **跨链聚合**（虽然链上 graph 被 bridge 割裂，但在 OKX 侧通过账户 ID 可以关联）
- 可以实现 **双向追溯**：从账户异常反查其关联的链上地址；从链上欺诈地址反查是否有关联账户在 OKX

---

### 8.3 LangGraph 集成：自动化调查 Agent

对于高风险地址，纯模型评分已经不够——合规团队需要的是可以用于 SAR 报告的完整调查报告，而不是一个风险分数。这就是 LangGraph agent 的价值所在：**在 ML scoring 之后，自动完成一个完整的调查链条**。

**系统定位：** LangGraph agent 不是欺诈检测本身的一部分，而是 **investigation automation layer**，位于 ML 评分之后、人工审核之前。

**工作流设计（Python 伪代码）：**

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class InvestigationState(TypedDict):
    address: str
    risk_score: float
    onchain_data: dict
    blacklist_result: dict
    fund_flow_analysis: dict
    defi_patterns: dict
    report: str
    human_review_required: bool

# 节点 1：获取链上数据
def fetch_onchain_data(state: InvestigationState) -> InvestigationState:
    """从链上数据服务拉取地址的交易历史、图特征、余额等"""
    address = state["address"]
    state["onchain_data"] = {
        "tx_history": get_recent_txs(address, limit=100),
        "balance": get_balance(address),
        "graph_metrics": get_graph_features(address),  # pagerank, degree, etc.
        "account_age_days": get_account_age(address),
    }
    return state

# 节点 2：黑名单与合规检查
def query_blacklist(state: InvestigationState) -> InvestigationState:
    """查询 OFAC、内部黑名单、Chainalysis、TRM Labs"""
    address = state["address"]
    state["blacklist_result"] = {
        "ofac_match": check_ofac_sdn(address),
        "internal_blacklist": check_internal_blacklist(address),
        "chainalysis_risk": chainalysis_api.get_risk(address),
        "direct_exposure": get_direct_exposure_to_sanctioned(address),
    }
    return state

# 节点 3：资金流分析（Neo4j Cypher 查询）
def analyze_fund_flow(state: InvestigationState) -> InvestigationState:
    """在 Neo4j 图数据库中追踪资金来源和去向"""
    address = state["address"]
    # 用 Cypher 查询 K 跳内的资金路径
    cypher_query = """
    MATCH p = (source:Address {id: $address})-[t:TRANSFER*1..5]->(dest:Address)
    WHERE dest.is_known_fraud = true OR dest.is_mixer = true
    RETURN p, [rel in relationships(p) | rel.amount_usd] AS amounts
    ORDER BY reduce(total = 0, a IN amounts | total + a) DESC
    LIMIT 10
    """
    paths = neo4j_session.run(cypher_query, address=address)
    state["fund_flow_analysis"] = {
        "paths_to_fraud": list(paths),
        "total_exposure_usd": sum_exposure(paths),
        "mixer_1hop": check_mixer_direct(address),
        "layering_pattern": detect_layering(address),
    }
    return state

# 节点 4：DeFi 模式分析
def assess_defi_patterns(state: InvestigationState) -> InvestigationState:
    """分析 flash loan、bridge 跨链、DEX 异常等 DeFi 特有欺诈模式"""
    address = state["address"]
    state["defi_patterns"] = {
        "flash_loan_involvement": detect_flash_loan(address),
        "cross_chain_activity": get_bridge_history(address),
        "token_approve_anomalies": check_suspicious_approvals(address),
        "rug_pull_involvement": check_rug_pull_history(address),
    }
    return state

# 节点 5：生成调查报告
def generate_report(state: InvestigationState) -> InvestigationState:
    """用 LLM 将结构化调查结果合成为自然语言报告"""
    prompt = f"""
    你是一名区块链反欺诈调查员。请根据以下结构化数据，生成一份完整的地址风险评估报告，
    包含：风险等级（高/中/低）、主要风险因素（3-5条）、证据链、建议行动。
    
    地址：{state['address']}
    风险评分：{state['risk_score']}
    链上数据摘要：{summarize(state['onchain_data'])}
    黑名单检查结果：{state['blacklist_result']}
    资金流分析：{state['fund_flow_analysis']}
    DeFi 模式分析：{state['defi_patterns']}
    """
    state["report"] = llm.invoke(prompt)
    state["human_review_required"] = (
        state["risk_score"] > 0.85
        or state["blacklist_result"]["ofac_match"]
        or state["fund_flow_analysis"]["total_exposure_usd"] > 100000
    )
    return state

# 构建 StateGraph
workflow = StateGraph(InvestigationState)

workflow.add_node("fetch_onchain_data", fetch_onchain_data)
workflow.add_node("query_blacklist", query_blacklist)
workflow.add_node("analyze_fund_flow", analyze_fund_flow)
workflow.add_node("assess_defi_patterns", assess_defi_patterns)
workflow.add_node("generate_report", generate_report)

# 定义边（前3个节点可并行，然后汇入 generate_report）
workflow.set_entry_point("fetch_onchain_data")
workflow.add_edge("fetch_onchain_data", "query_blacklist")
workflow.add_edge("fetch_onchain_data", "analyze_fund_flow")
workflow.add_edge("fetch_onchain_data", "assess_defi_patterns")
workflow.add_edge("query_blacklist", "generate_report")
workflow.add_edge("analyze_fund_flow", "generate_report")
workflow.add_edge("assess_defi_patterns", "generate_report")
workflow.add_conditional_edges(
    "generate_report",
    lambda s: "human_review" if s["human_review_required"] else END
)

app = workflow.compile()
```

**与 AML Investigation Mate 项目的关联：**

这套 LangGraph multi-agent 架构与我之前在 **AML Investigation Mate** 项目中构建的系统高度一致——同样使用 LangGraph StateGraph 协调多个 sub-agent，同样以 **Neo4j 作为图数据库**存储交易关系网络，同样用 Cypher 查询追踪资金流，同样最终输出供人工审核的调查报告。

核心区别在于应用场景：AML Investigation Mate 面向传统金融的 AML 案件调查，输入是可疑交易报告（STR）；OKX 场景面向链上地址的实时风险评估，输入是高风险地址触发的自动告警。两者的 agent 框架和 Neo4j 集成模式完全可复用，是直接可以迁移的工程经验。

---

## Interview Q&A

---

### Q1: 请设计一个基于 Transformer 的端到端区块链欺诈检测系统

**回答：**

这是一道综合性系统设计题，核心是展示对"多层次、多模态、可落地"系统架构的掌握。下面是我会在面试中给出的完整答案框架。

1. **总体架构：三层级联（3-layer Cascade）**

   系统的最高设计原则是：**不同风险层次用不同粒度的模型，在延迟预算内做最精准的判断**。

   - **Layer 1（规则引擎，<5ms）：** 处理所有请求。规则包括：OFAC SDN List 命中、与已知 mixer 1-hop 接触、速率限制触发（如 1h 内向 50 个新地址转账）、已知 attacker 地址黑名单匹配。这一层 recall 为 100% 设计，允许高 FP，快速排除明显正常和捕捉明显恶意。

   - **Layer 2（ML 评分引擎，<50ms）：** 处理 Layer 1 通过的"模糊地带"请求。包含两个并行子模型：
     - **序列 Transformer：** 对目标地址的最近 256 笔 tx 序列进行行为建模（Section 3 描述的 BERT 架构）
     - **表格 FT-Transformer：** 对聚合统计特征进行评分（Section 4）
     - 两路分数做 weighted ensemble（权重可学习，也可通过 validation 集手动调优）

   - **Layer 3（深度分析，异步）：** 对 Layer 2 评分 > 0.7 的地址触发异步深度分析：GNN（GraphSAGE 3-hop 聚合，看图邻居上下文）+ LangGraph 调查 agent（自动拉链上数据、查黑名单、分析资金流、生成调查报告）。结果在 30s-5min 内推送给合规团队。

2. **训练策略**

   - **Step 1（预训练，解决标签稀缺）：** 在全量无标签 tx 序列上做 **Masked Transaction Modeling（MTM）**。数据来源：BTC、ETH、TRON、BSC 等主链近 1 年活跃地址的 tx 序列，总计约 5-50 亿笔 tx。训练 6 层 256 维 BERT，约需 2-4 周 A100 集群时间。
   - **Step 2（Fine-tune，监督学习）：** 用已确认的欺诈/正常地址（正负比约 1:1000）进行 fine-tune，使用 **Focal Loss（γ=2, α=0.75）**。每周用新增标签数据做增量 fine-tune（learning rate = 1e-5，原始 fine-tune 的 10%）。
   - **Step 3（持续学习闭环）：** LangGraph agent 的 human review 结果（确认欺诈/确认正常）自动入库，每周 batch 重新 fine-tune；高置信度的无监督异常检测候选（Section 6）作为 soft label 补充训练数据。

3. **特征工程**

   - 序列特征：9-field per-tx token（Section 2.2）
   - 聚合特征：50+ 地址级统计特征（Section 2.1），包含图特征（pagerank、fraud neighbor count）
   - 实时查询：特征 store（Redis + Kafka streaming feature pipeline），保证 as-of-timestamp 正确性

4. **监控与维护**

   - 每日：score distribution 监控（均值/分位数漂移），PSI 计算（PSI > 0.2 触发告警）
   - 每周：incremental fine-tune + 模型性能对比（新模型 vs 当前生产模型在 hold-out 集上的 PR-AUC）
   - 事件驱动：新大型欺诈事件发生后（如 major protocol exploit），48h 内完成专项模型更新

> **Follow-up 提示：** 面试官可能追问"Layer 2 的序列 Transformer 和表格模型分别擅长什么？" —— 答：序列模型擅长捕捉 temporal behavior pattern（velocity、layering、structuring 等随时间展开的模式）；表格模型擅长处理 static profile 特征（account age、graph metrics、historical volume）。两者互补：欺诈者可能能改变近期行为序列，但难以同时改变所有聚合统计特征；也可能聚合特征正常但序列行为异常。Ensemble 使规避成本显著提高。

---

### Q2: 如何处理欺诈检测中极度不平衡的标签（正负比 1:1000）

**回答：**

标签不平衡是区块链欺诈检测的常态，处理思路要从训练损失、采样策略、评估指标、和业务目标四个维度同时考虑。

1. **Focal Loss（首选）**：$(1-p_{t})^{\gamma}$ 项动态降低 easy negative 的梯度权重，让模型聚焦困难边界样本。$\gamma=2$ 是默认值，在更极端不平衡时可尝试 $\gamma=3-5$。同时设置 $\alpha=0.75$ 给正样本更高的 loss 权重。

2. **Class Weights（辅助）**：对于 XGBoost，设置 `scale_pos_weight = 1000`（即负:正比例）。对于深度模型，可在 loss 中对正样本乘以 weight factor（与 Focal Loss 的 $\alpha_{t}$ 作用类似，可以结合使用）。

3. **为什么不用 SMOTE**：SMOTE 在低维特征（<20 维）上有效，但链上特征向量是 50-200 维的混合特征（含 log 变换、图特征、时序统计），SMOTE 通过 KNN 插值生成的合成样本往往是特征空间中不存在的"幻象数据点"，引入噪声而非有用信息。高维离散特征（如 `counterparty_type`）的插值本身就没有语义。

4. **阈值调整（独立于训练）**：模型 training 完成后，在 validation 集上画 **PR 曲线**（Precision-Recall，而非 ROC，因为不平衡场景下 ROC 会过于乐观），根据业务需求选择操作点：
   - 实时拦截：优化 **Precision**（宁可漏报，不能误伤正常用户体验）
   - 批量稽核：优化 **Recall**（宁可多报，不能漏掉欺诈）

5. **无监督预训练是根本解法**：上述四种方法都是在有限标签上"做文章"。真正从根本上解决的是 **MLM 预训练**——让模型在亿级无标签序列上先学到"什么是正常行为分布"，fine-tune 阶段只需要少量标签指示"哪里偏离了正常"。预训练质量直接决定了下游不平衡 fine-tune 的效果上限。

> **Follow-up 提示：** 面试官可能问"PR-AUC 和 F1@K 哪个更适合这个场景？" —— 答：PR-AUC 评估阈值无关的整体性能，适合比较不同模型；F1@K（在 top-K 风险地址上的 F1）更贴近业务实际——合规团队每天只能 review N 个 case，我们需要 top-N 里有多高比例是真实欺诈。两者结合使用：PR-AUC 做模型选型，F1@K 做业务指标监控。

---

### Q3: 如何应对欺诈者的对抗性规避（Adversarial Evasion）

**回答：**

欺诈者会持续调整行为模式以规避检测系统，这是所有生产欺诈检测系统面临的长期博弈。

1. **常见规避手法分析：**
   - **Structuring（分拆）：** 把一笔大额转账拆成多笔小额，规避金额阈值规则
   - **插入噪声 tx：** 在欺诈行为前后插入几笔正常转账（如给已知 CEX 充值小额），稀释行为序列中的异常 signal
   - **使用全新地址：** 每次欺诈换一个新建地址，规避基于 account_age 和历史行为的检测
   - **模拟正常行为：** 先让地址"养号"数周（做正常 DeFi 操作），再发动欺诈，混淆历史行为特征

2. **应对措施：**
   - **对抗训练：** 在训练集中加入合成的规避样本（模拟 structuring、插入噪声 tx 等），让模型对这些扰动具有鲁棒性
   - **图特征（难以伪造）：** 无论如何换地址，欺诈团伙的资金来源、最终归宿、和中间节点之间的关系结构难以完全隐藏。`known_fraud_neighbor_count`、`shortest_path_to_known_fraud`、`community_fraud_ratio` 这类图特征对新地址也有效（因为新地址仍然会与已知欺诈 cluster 有资金往来）
   - **多信号融合（链上 + 链下）：** 仅凭链上数据可能被规避，但 OKX 的 KYC、IP/设备指纹、内部交易行为与链上 signal 同时满足的概率极低。攻击者难以同时规避所有维度的检测
   - **概念漂移监控：** PSI 监控特征分布变化，一旦某类地址的行为分布开始系统性偏移（说明欺诈者在整体调整策略），立即触发模型 review 和更新
   - **模型 Ensemble（多样性攻击成本高）：** 单一模型容易被针对性规避，但同时规避序列 Transformer、表格 FT-Transformer、GNN 三个基于不同特征视角的模型，需要的规避成本极高

> **Follow-up 提示：** 面试官可能追问"如果欺诈者开始用 AI 生成的'正常行为模式'怎么办？" —— 答：这是 GAN-adversarial attack 的变体。应对：① 用 statistical test 检测 AI 生成的过于规律的行为（真实人类行为有 self-similar 的 burstiness，AI 生成的序列往往过于"平滑"）；② 引入对比学习，让模型学习"人类行为流形"，检测不在流形上的合成行为；③ 最终防线还是多信号融合——AI 能模拟链上序列，但难以同时伪造 IP、设备、KYC 这些链下信号。

---

### Q4: 如何在欺诈检测中使用 Transformer 的 Attention 做可解释性分析

**回答：**

可解释性在欺诈检测中有硬需求：合规人员需要 SAR 报告依据，被拦截用户需要申诉路径，监管要求决策可审计。

1. **Attention Weights 可视化（直觉工具，非正式解释）：** 提取 [CLS] token 对每笔 tx 的 attention weight，高亮 top-5 最高 attention 的 tx 展示给调查员。

   **重要局限：** Jain & Wallace（2019）在 *Attention Is Not Explanation* 中证明：attention weight 与梯度归因方法给出的 feature importance 之间相关性很低，高 attention 的 token 不一定是 causally 影响预测的 token。因此 attention 只能作为"快速看哪里"的 hint，**不能用作正式的合规解释依据**。

2. **Integrated Gradients（推荐，理论可靠）：** 沿从 baseline $x'$（全零或均值序列）到实际输入 $x$ 的路径积分梯度，精确量化每个 feature 对预测的贡献：

   $$\text{IG}_{i}(x) = (x_{i} - x'_{i}) \times \int_{0}^{1} \frac{\partial F(x' + \alpha(x-x'))}{\partial x_{i}} d\alpha$$

   满足 **完整性公理**（所有 feature 的 IG 之和 = $F(x) - F(x')$），理论上比 attention 更可靠。实现上用 Captum 库（Facebook Research），对一个 256-tx 序列的 IG 计算约需 500ms（可以接受，因为这是在 Layer 3 异步分析中使用）。

3. **SHAP（表格模型推荐）：** 对 XGBoost / FT-Transformer 用 TreeSHAP / DeepSHAP 给出每个聚合 feature 的 Shapley value，直接告诉调查员"是 `known_fraud_neighbor_count=2` 贡献了 +0.3 的风险分"。

4. **自然语言解释（最实用）：** 把上述分析结果输入 LangGraph 的 `generate_report` 节点，由 LLM 生成人类可读的解释：

   > "该地址 `0xabc...` 在 2024-03-15 向已知 Tornado Cash 出口地址转账 \$47,200（`known_fraud_neighbor_count=1` 触发高风险）；随后在 6h 内分 12 笔向新建地址分散转账（structuring pattern，序列模型 attention 最高的 5 笔均在此时间段）；其 pagerank 值为全链 top 0.05%，处于高风险资金网络的核心位置。综合风险分：0.94（高风险）。建议：冻结充提并上报合规。"

5. **反事实分析（Counterfactual Analysis）：** 当调查员认为某笔 tx 不可疑时，可以运行反事实：将该 tx 从序列中移除，重新推理，观察风险分是否显著下降。如果 $P(\text{fraud} | \text{seq without tx}) \ll P(\text{fraud} | \text{seq})$，则该 tx 确实是关键证据；如果变化很小，则调查员的直觉可能是正确的。

> **Follow-up 提示：** 面试官可能追问"如何在保证解释质量的同时满足实时延迟要求？" —— 答：解释性分析（IG、SHAP、LangGraph）都在 Layer 3 异步运行，不影响实时决策的 <100ms 延迟。Layer 1/2 给出评分和决策，Layer 3 在后台完成解释报告，推送到调查员工作台。这是"先决策、后解释"的实用工程设计。

---

### Q5: 你会如何在有限标注数据下构建欺诈检测模型？

**回答：**

标注数据量不同，对应的最优策略完全不同。下面按三个阶段分别说明：

1. **Phase 1：标注极少（<100 个欺诈样本）**

   这个阶段有监督模型毫无意义，应该：
   - **无监督异常检测：** Isolation Forest + Anomaly Transformer，学习正常行为分布，偏离者作为候选
   - **规则工程：** 与业务专家共同整理 hand-crafted 规则（OFAC match、mixer 1-hop、velocity rule），高 recall 低 precision
   - **主动学习启动：** 把无监督高分候选提交人工 review，快速积累第一批高质量标签
   - 目标：在 2-4 周内把标注量从 <100 扩展到 1000+

2. **Phase 2：少量标注（100 - 10K 欺诈样本）**

   - **MLM 预训练：** 用全量无标签序列预训练 BERT 模型（Section 3.2），确立行为表征基础
   - **Few-shot Fine-tune：** 在 100-10K 样本上用 Focal Loss fine-tune，learning rate 极小（1e-5），early stopping 严格（防止在少量样本上过拟合）
   - **标签传播：** 从已确认欺诈地址出发，在图上做 label propagation，扩充软标签（Section 5.3）
   - **主动学习（Active Learning）：** 用 uncertainty sampling（预测概率接近 0.5 的样本最有价值）+ diversity sampling，引导下一批人工标注最大化信息增益

3. **Phase 3：规模化标注（>10K 欺诈样本）**

   - **全监督训练：** 三路模型（序列 Transformer + 表格 FT-Transformer + GNN）联合训练 + ensemble
   - **数据增强：** 规则合成 structuring、peel chain 等规避模式的变体，扩充训练分布
   - **对抗鲁棒性：** 加入对抗样本训练，提高模型对规避行为的抵抗力
   - **持续学习：** 建立 human-in-the-loop 闭环，confirmed label 自动入库，每周增量 fine-tune

> **Follow-up 提示：** 面试官可能追问"如何衡量主动学习的效果？" —— 答：对比"随机采样 N 个样本标注 + fine-tune" vs "主动学习选 N 个样本标注 + fine-tune"，在 hold-out 集上的 PR-AUC。通常主动学习用同样的标注预算可以达到随机采样 2-3 倍标注量的效果。

---

### Q6: 请比较 GNN 和 Graph Transformer 在欺诈团伙检测中的优劣

**回答：**

| 维度 | GNN（GraphSAGE / GAT） | Graph Transformer（Graphormer / Graph-BERT） |
|------|----------------------|---------------------------------------------|
| **信息传播机制** | 局部 message passing，信息逐层从邻居传播 | 全局 attention，任意两节点可直接交互 |
| **长距离依赖** | K 层 GNN 只能看 K hop 邻居，长距离需要堆叠多层（过平滑风险） | 理论上可以直接建模任意距离的节点关系，适合捕捉多跳洗钱路径 |
| **计算复杂度** | $O(N \cdot d^{2})$（neighborhood sampling 后），适合大规模图 | $O(N^{2} \cdot d)$ 全 attention，在大图上不可行（需要 subgraph 采样）|
| **可扩展性** | 优秀：GraphSAGE + neighbor sampling 可扩展到亿级节点 | 受限：Graphormer 原始版本只适合小图（<10K 节点）；需要 subgraph 采样才能用于大图 |
| **大图（>1M 节点）** | 推荐：GraphSAGE / GAT + cluster sampling | 不推荐直接用；需要用 subgraph 提取 + Graphormer 分析局部子图 |
| **小图精细分析（欺诈子图，<1K 节点）** | 效果良好，但难以捕捉超长距离关系 | 优秀：全图 attention 能同时看到所有节点间的关系，适合欺诈团伙内部结构分析 |

**推荐架构（结合两者优势）：**

1. **全图扫描（GNN）：** 用 GraphSAGE 3-hop 在全链 address graph 上批量计算所有节点的 embedding，给出初步风险分。识别出高风险 cluster（risk score > 0.6 的地址构成的连通子图）。

2. **子图精细分析（Graph Transformer）：** 对高风险 cluster 提取子图（通常 <500 节点），用 Graphormer 对子图做完整 attention 分析。Graphormer 的全局 attention 能看到"欺诈团伙内部谁是 hub、谁是 peripheral、资金流向是否有闭环"等细粒度 pattern。

3. **输出融合：** GNN embedding（全局上下文）+ Graphormer embedding（局部精细）通过 MLP 融合，输入下游分类头。

> **Follow-up 提示：** 面试官可能追问"过平滑（over-smoothing）问题在欺诈检测中有多严重？" —— 答：在欺诈检测中，过平滑的表现是"欺诈地址和其正常邻居的 embedding 越来越接近"，导致 fraud cluster 内部的边界地址（轻度参与洗钱的中间地址）与正常用户难以区分。应对：① 用 Jumping Knowledge（JK-Net）连接各层 embedding；② 使用 GraphSAGE 的 concat aggregation（保留中心节点自身信息）而非 mean aggregation；③ 限制 GNN 层数（通常 2-3 层足够，不要堆 6+ 层）。

---

### Q7: OKX 的 JD 提到 LangGraph——你会如何把它整合进欺诈检测 pipeline

**回答：**

先明确定位：**LangGraph 解决的是 investigation automation 问题，而不是检测本身**。ML 模型（Transformer + GNN）负责快速给出风险分，LangGraph agent 负责在高风险案件上自动完成调查员本需要手动做的事情：查数据、对比黑名单、追踪资金流、生成报告。

1. **为什么需要 Agent 而不是单一 LLM 调用：**

   高风险地址的完整调查需要：① 链上数据查询（多次 RPC / Dune API 调用）；② 图数据库 Cypher 查询（Neo4j）；③ 合规黑名单 API（Chainalysis、OFAC）；④ 内部系统查询（OKX 账户关联）；⑤ 最终报告生成。这是一个有状态的、多步骤的、工具调用依赖的流程，正好是 LangGraph StateGraph 的 use case——每个步骤是一个 node，工具调用是 edge，状态在节点间传递。

2. **LangGraph Workflow 设计（完整 Python 伪代码见 Section 8.3）：**

   关键节点：
   - `fetch_onchain_data`：拉取 address tx history、graph metrics、余额
   - `query_blacklist`：OFAC + 内部黑名单 + Chainalysis risk score
   - `analyze_fund_flow`：**Neo4j Cypher 查询** K-hop 路径到已知欺诈/mixer 地址
   - `assess_defi_patterns`：Flash loan 检测、bridge 跨链记录、suspicious approve
   - `generate_report`：LLM 综合以上信息生成自然语言调查报告
   - **Conditional Edge：** 高风险（score > 0.85）或 OFAC 命中则进入 `human_review` 队列，否则自动存档

3. **与 AML Investigation Mate 的关联（直接可迁移的经验）：**

   这套架构与我在 **AML Investigation Mate** 项目中构建的 multi-agent AML 调查系统高度一致：
   - 同样用 **LangGraph StateGraph** 协调多个 sub-agent 的执行顺序和状态传递
   - 同样用 **Neo4j** 作为知识图谱存储交易关系，用 Cypher 查询追踪资金路径
   - 同样最终输出供合规人员 review 的结构化调查报告

   主要差异：AML Investigation Mate 处理的是传统金融 STR（可疑交易报告）驱动的案件，数据源是银行交易记录；OKX 场景处理的是链上地址告警，数据源是区块链数据。框架和工具链完全相同，数据接口层需要调整。这是我面试中可以直接 refer 的 production experience。

4. **工程注意事项：**
   - Agent 运行在 Layer 3（异步），不影响 Layer 1/2 的实时决策延迟
   - Neo4j 的 Cypher 查询对于 5-hop 路径分析在百亿边规模的图上可能超时，需要设置 query timeout + fallback（退化为 2-hop 查询）
   - LLM 调用（report generation）是成本中心，对明显低风险地址（score 0.5-0.7 且无 OFAC match）可跳过 LLM 调用，只输出结构化 JSON 报告

> **Follow-up 提示：** 面试官可能追问"如何保证 LangGraph agent 的输出质量和一致性？" —— 答：① 对每个 node 的输出做 schema validation（Pydantic model）；② `generate_report` 节点用 few-shot prompt engineering（提供 3-5 个人工撰写的高质量报告作为示例）；③ 建立 report quality scoring（由 second LLM 评估报告的完整性和一致性）；④ 对 LLM 输出的风险等级判断做 sanity check（如 LLM 说"低风险"但 ML score 是 0.9，触发告警人工 review）。

---

### Q8: 如何评估和监控线上欺诈检测模型的健康状态

**回答：**

模型上线不是终点，持续监控是保证系统有效运行的关键。

1. **离线评估指标（模型选型和版本对比用）：**
   - **AUC-ROC：** 模型整体排序能力，但在极度不平衡时可能虚高（因为 TN 很多）
   - **AUC-PR（推荐）：** Precision-Recall 曲线下面积，对不平衡数据更敏感，更准确反映模型对欺诈样本的区分能力
   - **F1@K：** 在 top-K 风险地址上的 F1，直接对应"每天 review N 个 case，准确率多少"的业务问题

2. **线上监控指标：**
   - **Score distribution shift：** 监控 P(fraud) 的分布（mean、p50、p95、p99）是否系统性漂移。骤增可能是新欺诈模式涌现；骤降可能是模型过度偏向 normal
   - **PSI（Population Stability Index）：** 对关键输入特征（`log_amount`、`tx_count_7d`、`account_age_days`）和模型输出计算 PSI：
     - PSI < 0.1：稳定，无需干预
     - PSI 0.1 ~ 0.2：轻微漂移，加强监控
     - **PSI > 0.2：触发告警，启动人工 review 和模型更新评估**
   - **Human Review Confirmation Rate：** 进入人工 review 队列的案件中，调查员确认"确实欺诈"的比例。这是线上 Precision 的 proxy（无法直接计算线上 Precision，因为未被拦截的欺诈没有标签）

3. **模型更新触发条件：**
   - PSI > 0.2（特征分布漂移）
   - Human review Precision 连续 3 天下降超过 5%
   - 发生新型重大欺诈事件（链上大型 exploit、新 mixer 上线、新攻击手法被发现）

4. **更新策略：**
   - **轻微漂移：** 增量 fine-tune（使用最近 2 周新增标签，lr = 1e-5，5 epoch）
   - **中度漂移：** 全量 retrain（使用滑动窗口数据，保留最近 3 个月，lr = 3e-5）
   - **重大新型欺诈：** 专项模型更新（针对新 pattern 收集 case，加入训练集，1-2 天内完成更新）

> **Follow-up 提示：** 面试官可能追问"线上没有 ground truth label，如何做实时 AUC 监控？" —— 答：这是"delayed label"问题的典型解法：① 用 human review 结果作为 proxy（但有 selection bias：只有高 score 的 case 才被 review）；② 用 label propagation 把 confirmed fraud cluster 扩展到相关地址作为 delayed labels，3-7 天后做 retrospective 评估；③ 用"后续是否被交易所 ban / 被 OFAC 加入制裁列表"作为延迟 label，90 天后回测。

---

### Q9: 区块链欺诈检测和传统金融欺诈检测（信用卡欺诈）有哪些关键区别

**回答：**

| 维度 | 传统金融欺诈（信用卡欺诈） | 区块链欺诈检测 |
|------|--------------------------|--------------|
| **数据可获取性** | 交易数据在内部系统，外部不可见；需要 data partnership 或内部权限 | 公链数据完全公开，任何人可下载全量历史 tx；是罕见的"数据完全透明"场景 |
| **身份绑定** | 每笔交易绑定到真实账户（姓名、卡号、银行账户），身份 attribution 明确 | 链上地址是假名（pseudonymous），地址不直接绑定真实身份；需要 KYC + off-chain linkage 才能 de-anonymize |
| **攻击原子性** | 欺诈通常是序列行为（多笔 tx 构成攻击 pattern） | Flash loan 等 DeFi 攻击可在 **单笔 tx** 内完成完整的"攻击-获利"，传统 ML 序列模型无法检测 |
| **不可逆性** | 银行可以 chargeback（事后撤销）；争议解决机制完善 | 区块链 tx 一旦确认不可撤销；欺诈损失几乎无法追回；检测必须在 tx 发生前完成（或 block confirmation 前） |
| **攻击者复杂度** | 多为人工欺诈 + 初级脚本；高级攻击者相对少 | 大量专业黑客、DeFi exploit 团队，技术能力强；会针对检测模型做 adversarial evasion |
| **监管清晰度** | 明确的监管框架（PCI-DSS、GDPR、各国金融监管）；chargeback 有法律保障 | 监管框架仍在演进（各国对 DeFi 的监管立场不同）；部分欺诈行为（如 MEV）处于法律灰色地带 |
| **数据丰富度** | 仅限交易记录 + KYC；历史数据较少（数年） | 链上 graph + sequence + smart contract code + event log + off-chain（CEX 数据）；历史可追溯到 2009 年（BTC） |
| **图数据重要性** | 图结构次要（账户关系相对简单，主要是 account → merchant） | 图数据 first-class（on-chain 原始数据就是 edge list，fund flow pattern 是关键欺诈 signal） |

**关键洞察：**

最本质的区别是 **"永久证据链 vs 实时不可追回"的矛盾**：区块链提供了比传统金融更完整的交易历史（所有历史 tx 都在链上，可以随时回溯分析），但同时也意味着一旦欺诈完成，损失几乎不可挽回。这使得欺诈检测的重心必须放在 **事前/事中** 而非事后。同时，公链数据的完全透明性也给了攻击者"研究 detection system"的机会——他们可以模拟检测行为，针对性地设计规避策略。

> **Follow-up 提示：** 面试官可能追问"针对这些区别，你会如何调整 model architecture？" —— 答：① 为应对 single-tx atomicity，增加 tx 内部 trace 分析（基于 gas usage、internal call count 的规则，不依赖序列历史）；② 为应对不可逆性，把检测延迟目标从"事后分析"压到"充提触发的 pre-confirmation 检查"（<100ms）；③ 为应对对抗性复杂度，重点投资 adversarial training 和多信号融合；④ 为利用图数据优势，把 graph feature（pagerank、fraud neighbor）作为 first-class feature 而非 optional add-on。

---

### Q10: 用一段话描述你会如何向面试官 pitch 你的完整技术方案

**回答：**

以下是完整的 Staff Engineer pitch（3 句话 + 配套展开）：

---

**核心 Pitch（3 句话）：**

> "我的方案是一个三层级联系统：Layer 1 用规则引擎在 5ms 内处理 OFAC / 黑名单命中，Layer 2 用 Focal Loss fine-tune 的 BERT-style 序列 Transformer 和 FT-Transformer 表格模型在 50ms 内对地址行为打分，Layer 3 异步运行 GraphSAGE + LangGraph 调查 agent 完成图上的团伙识别和自然语言报告生成。训练策略上先在亿级无标签 tx 序列上做 MLM 预训练解决标签稀缺，再用 Focal Loss fine-tune 对抗 1:1000 的类别不平衡，每周增量更新应对概念漂移。部署上用知识蒸馏 + INT8 量化满足实时延迟要求，PSI 监控特征分布偏移，human review confirmation rate 作为线上 precision 的 proxy，确保系统在与欺诈者的长期博弈中保持有效性。"

---

**每个组件为什么是最优选择：**

1. **为什么是三层级联而不是单一模型：** 不同层次的欺诈有不同的延迟要求和精度需求；级联设计让 80% 的请求在 Layer 1 快速处理，计算资源集中在真正"难以判断"的 20%，同时支持不同模型并行迭代升级而不相互干扰。

2. **为什么 MLM 预训练是核心投资：** 链上标签数据是永远稀缺的（新型欺诈天然没有标签）。预训练一次，所有下游任务（fraud detection、anomaly detection、address clustering）都受益，是边际收益最高的技术投资。

3. **为什么 LangGraph 而不是单纯规则报告：** 高风险案件的调查需要"动态决策"——根据初步发现决定下一步查什么，这不是 DAG 规则能表达的；LangGraph 的 conditional edge 和 tool-use 能力让 agent 自适应地完成调查路径，同时我在 AML Investigation Mate 项目中已有这套架构的生产验证，迁移成本极低。

> **Follow-up 提示：** 面试官可能追问"如果只有 3 个月时间，你会优先实现哪部分？" —— 答：优先级：① Layer 1 规则引擎（1 周，保证底线安全合规）→ ② 地址聚合特征 + XGBoost Layer 2 基础版（2 周，快速拿到可用基线）→ ③ MLM 预训练（持续进行，6-8 周）→ ④ 序列 Transformer fine-tune（2 周，基于预训练）→ ⑤ LangGraph agent MVP（2 周，接入调查工作台）。GNN 和 Graph Transformer 放在 3 个月之后做，因为前几步已经能捕捉大部分欺诈 signal，GNN 的边际提升需要更多标注数据来验证。

---
