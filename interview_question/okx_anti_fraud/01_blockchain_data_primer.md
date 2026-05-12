# 区块链数据入门（Blockchain Data Primer for ML Engineers）

> **文档定位：** OKX Anti-Fraud AI Engineer 面试准备系列 File 1 of 4。
>
> **目标读者：** 有 ML 工作经验、但对 blockchain data 结构不熟悉的工程师。本文不是 crypto 课程，而是把链上数据当作 ML dataset 来讲解，重点回答"这些数据长什么样、怎么转成 feature/graph/sequence、有哪些 fraud pattern 值得建模"。
>
> **配套阅读：** `02_*.md`（on-chain feature engineering）、`03_*.md`（graph ML on blockchain）、`04_*.md`（sequence modeling & fraud detection systems）。

---

## 1. 什么是区块链（What Is a Blockchain）

---

### 1.1 一句话定义

区块链是一个 **distributed append-only ledger**：所有 transaction 被打包成 block，block 之间通过 **cryptographic hash pointer** 串成 chain，由分布式 network 的多个 node 共同维护，没有中央 trusted party。

对 ML 工程师来说，最重要的不是密码学细节，而是：
- **数据完全公开（public chain 场景下）：** 每一笔历史交易都可以下载下来做分析。这是和传统金融最大的区别——你不需要内部数据访问权限就能拿到 raw transaction log。
- **数据不可篡改（immutable）：** 历史不会被改写，所以 label 和 feature 之间的时间关系是可信的。
- **数据是 graph + sequence 双重结构：** 既是 address-to-address transfer graph，又是按 block height / timestamp 排好序的 event stream。

### 1.2 Block 的链式结构

一个 block 包含 header + body：
- **Header：** previous block hash、timestamp、block height、Merkle root、nonce/validator info
- **Body：** 一批 transaction（取决于链：Bitcoin 一个 block 约 2k-3k tx，Ethereum 约 100-400 tx）

每个 block header 里嵌入了 **previous block hash**，所以任何人篡改历史 block 会导致后续所有 block 的 hash 校验失败。这种结构对 ML 的实际意义：
- **天然的时间戳：** `block_height` 和 `block_timestamp` 是可信的时间字段，可以直接作为 sequence 排序键。
- **天然的 immutable label：** 一旦 transaction 被打包进 block 并经过足够 confirmation，就不会再变。这避免了传统欺诈数据集里 label 被反复 revise 的问题。

### 1.3 Consensus Mechanism（够用即可）

Consensus mechanism 决定谁有权把新 block 打包到 chain 上。ML 工程师只需要知道两类：

| 类型 | 代表 | 关键点 | ML 视角的意义 |
|------|------|--------|--------------|
| **Proof of Work (PoW)** | Bitcoin | Miner 通过算 hash 谜题竞争打包权 | 出块时间慢且有 variance（Bitcoin ~10min/block），影响 real-time detection latency |
| **Proof of Stake (PoS)** | Ethereum（合并后）、Solana 等 | Validator 按质押的 token 比例被选中打包 | 出块更稳定（Ethereum 12s/block），适合做接近 real-time 的 streaming feature |

> **面试要点：** 面试官不会让你解释 PoW 的 hash 难度调整，但可能问"为什么 on-chain real-time fraud detection 比传统 card fraud 更难？"——答案中"block confirmation latency + reorg risk"是 consensus mechanism 带来的。

### 1.4 Public Chain vs Permissioned Chain

OKX 是 centralized exchange (CEX)，但它服务的用户在 public chain（BTC、ETH、BSC、TRX、Polygon 等）上充值、提现、与 DeFi 交互。所以这里讨论的"链上数据"主要指 **public chain data**：任何人都可以通过 full node 或第三方 indexer（Etherscan、Blockchair、Dune、Allium）拿到 raw transaction。

---

## 2. 交易数据结构（Transaction Data Anatomy）

链上的 transaction 数据结构有两种主流范式：UTXO 模型（Bitcoin 系）和 Account 模型（Ethereum 系）。两者底层数据 schema 完全不同，**对应的 feature engineering 思路也完全不同**——这是面试高频问点。

---

### 2.1 UTXO 模型（Bitcoin-style）

**UTXO = Unspent Transaction Output**。Bitcoin 没有"账户余额"这个概念，每个 address 的余额是它持有的所有 **unspent outputs** 的总和。

#### 直观理解

你可以把 UTXO 想成"现金钞票"：
- 你给别人付钱时，不是从余额里扣，而是把手里的若干张钞票（UTXO）整体交出去
- 如果钞票面额比要付的金额大，对方会找你零钱（**change output**，通常返回到你自己控制的另一个 address）
- 所以一笔 transaction 通常包含 **多个 input UTXO** 和 **多个 output UTXO**

举例：你想给 Alice 转 0.7 BTC，你有两个 UTXO：一个 0.5、一个 0.4。这笔 tx 会：
- Inputs：消耗 0.5 + 0.4 = 0.9 BTC（两个 UTXO）
- Outputs：0.7 给 Alice，0.19 找零给你自己（剩下 0.01 是 miner fee）

#### 为什么没有"账户余额"字段

因为 Bitcoin 协议里根本没有"account"这个 first-class entity。一个 address 的余额是 derived field，需要 scan 全链所有 UTXO 才能算出来。**这对 feature engineering 的含义：**
- 不能简单 `SELECT balance FROM accounts`；必须维护 UTXO set 或者用 indexer (Blockchair / BigQuery `bitcoin-public-data`) 提供的 derived view
- 一个用户通常控制多个 address（HD wallet），所以 address 不等于 entity，需要做 **address clustering / heuristic linking** 才能还原 user-level 视图

#### UTXO 模型 Transaction 字段表

| 字段 | 说明 | ML 意义 |
|------|------|---------|
| **txid** | Transaction hash，全链唯一标识 | 主键，可作为 join key |
| **inputs** | List of referenced previous UTXOs `(prev_txid, prev_vout_index, signature)` | 反向追溯资金来源；**common-input heuristic** 假设同一 tx 的所有 input 由同一 entity 控制，是地址聚类的核心 signal |
| **outputs** | List of `(amount, recipient_script/address)` | Fan-out structure；零钱 output 通常 amount 较小且 round 不到整数 |
| **block_height** | Tx 所在 block 的高度 | 排序键，也用于计算 confirmation depth |
| **fee** | inputs total − outputs total | 高 fee 往往意味着 urgency（time-sensitive arbitrage / panic exit），是 anomaly feature |
| **lock_time** | 该 tx 可被打包的最早 block | 大多数 fraud tx 不用此字段，但非零值本身就是 outlier signal |
| **witness / segwit data** | SegWit 之后的见证数据 | 一般不直接用做 feature，但可以判断 wallet type（legacy vs segwit vs taproot） |

#### UTXO 模型的 fraud detection 特殊考量
- **No-balance feature：** 必须 reconstruct UTXO set 时序快照，才能算"address 在 tx 发生时持有多少 BTC"
- **Address ≠ Entity：** 一个 user 用 100 个一次性 address 是常态，feature 必须在 cluster level 聚合
- **Mixer / CoinJoin 友好：** UTXO 允许多 input + 多 output 的 transaction batch，天然适合做 mixing（多个用户的 input 凑在一起，output 分散到新 address），追踪难度高

---

### 2.2 账户模型（Ethereum-style）

Ethereum 用 **account-based model**：每个 address 有显式的 `balance` 字段，转账就是 `from.balance -= value; to.balance += value`。这和传统数据库的账户表非常像，对 ML 工程师更直观。

#### 两类 Account
- **EOA (Externally Owned Account)：** 由私钥控制的"用户账户"，可以发起 transaction
- **Contract Account：** 由 smart contract code 控制，不能主动发 tx，只能被调用

#### 账户模型 Transaction 字段表

| 字段 | 说明 | ML 意义 |
|------|------|---------|
| **from** | 发起方 address（必须是 EOA） | Sender entity，最主要的 graph edge 起点 |
| **to** | 接收方 address（可以是 EOA 或 contract） | 如果 `to` 是 contract address，说明这是一笔 contract call 而不是普通转账 |
| **value** | 转账金额（单位 wei，$1 ETH = 10^{18}$ wei） | 金额特征：分布、round number 检测、与历史均值的偏离 |
| **gas_price** | 发起方愿意为每单位 gas 支付的价格（Gwei） | 异常高的 gas_price 往往是 MEV bot / 抢跑 / panic 行为 |
| **gas_limit** | 该 tx 允许消耗的最大 gas | 标准转账固定 21000；高 gas_limit 意味着复杂 contract interaction |
| **gas_used** | 实际消耗的 gas | 复杂度指标；`gas_used == gas_limit` 可能是 out-of-gas revert 的信号（但 revert 也可在 gas 未耗尽时发生，例如通过 REVERT opcode） |
| **nonce** | 该 from address 已发送过的 tx 数量（从 0 递增） | 单调递增，可以 detect address activity intensity；也是反 replay attack 的设计 |
| **input** / **data** | 调用 contract 时的 calldata，前 4 字节是 function selector | **核心特征源**：解码后可以知道用户调了哪个 contract method（swap / transfer / approve / bridge） |
| **block_timestamp** | Tx 被打包的时间 | Sequence 排序键、time-of-day / day-of-week feature |
| **status** | 1 = success, 0 = reverted | Reverted tx 很多时候是 sandwich attack / front-run failure 的信号 |
| **logs / events** | Contract 执行过程中 emit 的 event log | ERC-20 transfer、DEX swap 等"内部转账"只在 log 里出现，不在 top-level tx 里 |

#### 账户模型的 fraud detection 特殊考量
- **Balance 直接可读：** 不需要 reconstruct UTXO set
- **Internal transactions：** Smart contract 在执行中产生的子调用（internal tx）不在 main tx 表里，需要 trace（debug_traceTransaction）才能拿到。**很多 DeFi 欺诈的钱流只能从 internal tx 看到**
- **ERC-20 transfer 是 event log，不是 tx：** ETH 转账有 `value` 字段，但 USDT/USDC 这类 token 转账要从 `Transfer(address,address,uint256)` event 里解出来。意味着 fraud feature pipeline 必须同时处理 native transfer + token transfer 两类事件

---

## 3. 链上实体类型（On-Chain Entity Types）

链上不是只有"用户转账给用户"。在做 fraud detection 之前必须搞清楚 transaction 的 counterparty 到底是什么类型的 entity。

### 3.1 EOA vs Smart Contract

| 类型 | 怎么识别 | 行为特征 |
|------|---------|---------|
| **EOA** | `eth_getCode(address)` 返回 `0x`（空 bytecode） | 由私钥控制，受人操作，活跃度有 daily pattern |
| **Smart Contract** | `eth_getCode(address)` 返回非空 bytecode | 被动响应调用，活跃度由调用者决定，行为可预测（同一 method call 应该得到相同效果） |

**Fraud detection 含义：** 检测 EOA 的 anomaly 和检测 contract 的 anomaly 是两个完全不同的问题。EOA 异常关注"用户行为偏离"，contract 异常关注"被滥用/被 exploit 的模式"。

### 3.2 Token Contracts

| Standard | 类型 | 特点 |
|----------|------|------|
| **ERC-20** | Fungible token | 同质化、可分割（USDT、USDC、UNI 等）；通过 `Transfer(from, to, value)` event 转账 |
| **ERC-721** | Non-fungible token (NFT) | 每个 tokenId 唯一；wash trading / pump-and-dump 高发地 |
| **ERC-1155** | Multi-token | 一个 contract 管多个 token，常见于游戏资产 |

**关键点：** token transfer 的 from/to 是在 token contract 的 event log 里，**top-level tx 的 from 是 caller、to 是 token contract address**。新手最常踩的坑：用 top-level tx 来构建 transfer graph，结果把所有 USDT 转账的 to 都画成"USDT 合约地址"——这是错的，必须从 ERC-20 Transfer event 里解析。

### 3.3 DEX（Decentralized Exchange）

代表：Uniswap、Sushiswap、Curve、PancakeSwap。用户通过调用 router contract 来 swap token。

- **典型 tx 结构：** `user → Router.swapExactTokensForTokens() → 多个 Pair contract → user`
- **链上可见 signal：** Router method selector、`Swap(sender, amount0In, amount1In, amount0Out, amount1Out, to)` event、路径上的 pair address
- **欺诈关联：** rug pull 之前的大量 liquidity removal、wash trade（用户左手 swap 右手）、sandwich attack（MEV bot 夹击大单）

### 3.4 Bridge Contracts

跨链桥：把资产从 chain A 锁定、在 chain B 上 mint 等价 token（或反向 burn-unlock）。代表：Wormhole、LayerZero、Across、Stargate。

- **Fraud 视角：** Bridge 是 fund flow 跨链追踪的"断点"——资金一旦经过 bridge，链上 graph 就被割裂。检测 cross-chain laundering 的核心难点就是把 bridge in / bridge out 的事件按 timestamp + amount + recipient 配对起来。
- **历史教训：** Ronin、Wormhole、Nomad 都被攻击过，单次损失 $100M+ 级别。这类事件在链上有 clear signature：短时间内 bridge contract 被异常 drain。

### 3.5 其他重要 entity（识别它们能极大提高 feature 质量）
- **CEX hot/cold wallet：** Binance、OKX、Coinbase 等交易所的已知地址。资金流入/流出 CEX 是 KYT (Know Your Transaction) 的关键 boundary
- **Mixer：** Tornado Cash、Wasabi CoinJoin、Samourai Whirlpool（2024年被 DOJ 取缔） —— 资金进入 mixer 后下游交易难以追踪，是 high-risk signal
- **Sanctioned address：** OFAC SDN List 上的链上地址，合规上必须实时拦截

---

## 4. 区块链数据作为 ML 数据集（Blockchain Data as an ML Dataset）

理解完 raw schema 后，关键是怎么把它转成 ML model 能消费的格式。同一份链上数据可以同时支持三种 view：tabular、graph、sequence。**一个成熟的 fraud detection 系统通常三种 view 都用**。

---

### 4.1 表格视角（Tabular View）

最直接的视角：以 **address** 或 **transaction** 为粒度，把每个 entity 展开成一行 feature vector，喂给 XGBoost / LightGBM / DNN。

**Address-level features 的分类（这是 fraud detection 的工程主战场）：**

| Feature 类别 | 举例 | 典型 fraud signal |
|-------------|------|------------------|
| **交易量特征 (Volume)** | total_tx_count、total_in_value、total_out_value、avg_tx_value、max_tx_value、tx_count_24h/7d/30d | 短期内交易量暴涨、in/out 严重不平衡 |
| **时间特征 (Temporal)** | address_age（从首次活跃到现在）、tx_hour_distribution（按小时统计 tx 数）、tx_burstiness、inactive_period_count、time_to_first_outflow | 新地址 + 快速大额流入流出 = layering 嫌疑；超出 human 24h pattern（如全天均匀 = bot） |
| **对手方特征 (Counterparty)** | unique_counterparty_count、in_counterparty_count、out_counterparty_count、counterparty_repeat_rate、counterparty_risk_score_max | 大量一次性对手方 = mixer 用户；与已知 sanctioned address 有 1-hop 接触 |
| **合约交互 (Contract Interaction)** | contract_call_count、unique_contract_count、dex_swap_count、bridge_use_count、token_approve_count、interacted_with_mixer | 与 mixer/bridge 接触是 high-risk；高频 token approve 给 unknown contract = 钓鱼 victim 风险 |
| **金额模式 (Amount Pattern)** | round_amount_ratio（整数金额占比）、amount_concentration（top-1 金额 / total）、value_to_gas_ratio、small_dust_tx_count | 大量 round number = 人工脚本/Ponzi 派彩；大量 dust = address poisoning 攻击者或受害者 |
| **Gas/Fee 特征** | avg_gas_price、max_gas_price、gas_price_zscore_vs_block_avg | 高 gas 抢跑、MEV bot 行为 |
| **Token 持仓特征** | token_diversity、stablecoin_ratio、nft_holding_count | Drainer 受害者通常 token diversity 突降 |

**关键工程点：** 这些 feature 都需要在 **as-of-timestamp** 状态下计算（point-in-time correctness），否则会有 label leakage。

---

### 4.2 图视角（Graph View）

链上数据天然是一张 graph：
- **Nodes：** address（EOA + contract）
- **Edges：** transfer（带 amount、timestamp、token type、direction）
- **Edge weight：** 通常用累计 transfer amount（按 USD 计价），或交易次数

#### 经典 graph feature（无需 GNN 也能用）

| Feature | 含义 | Fraud signal |
|---------|------|--------------|
| **in-degree / out-degree** | 接收/发送的对手方数 | Mixer-like：极高 in 极高 out；Ponzi receiver：高 in 极低 out |
| **weighted in/out value** | 累计 in/out 金额 | 配合 degree 看是否资金集中或分散 |
| **PageRank / 风险传播 score** | 节点重要性 / 离已知坏地址的 proximity | 高 PageRank 且与 sanctioned cluster 相连 = 重要洗钱节点 |
| **clustering coefficient** | 邻居之间互相连接的比例 | 高 clustering = wash trade / 自循环；正常用户应该接近 0 |
| **connected component size** | 所在弱连通分量的大小 | 孤立的小 component 出现大额转账 = 可疑 |
| **community / cluster id** | Louvain、LPA 等社区发现的输出 | 同一 community 内同步异常 = coordinated attack |
| **shortest path to known bad** | 到已知 fraud address 的最短跳数 | 1-2 hop 接触 sanctioned address 是 KYT 红线 |

#### GNN 视角

把上面的 graph 喂给 GAT / GraphSAGE / GIN 类的 GNN，可以学到 **address embedding**，下游既可以做 node classification（fraud / clean）又可以做 link prediction（潜在的 entity 关联）。在 OKX 场景下，常见任务包括：
- Address clustering / entity resolution
- 已知 fraud cluster 的相似性扩散
- 资金流追踪与异常 path scoring

---

### 4.3 序列视角（Sequence View）

把同一个 address 的所有交易按 timestamp 排序，就得到一个 event sequence。这个视角的力量来自：**user behavior over time 是 fraud detection 最强的 signal 之一**。

#### 把 transaction 当作 token

每笔 tx 可以编码为一个"token"：
- 离散字段：method_id / contract_type / counterparty_cluster_id / token_id
- 连续字段：log(value)、log(gas_price)、time_delta（距离上一笔的时间）

把这些字段拼成 token embedding 输入 transformer，就可以做：
- **Sequence classification：** 这个 address 的近 N 笔行为是不是 fraud
- **Next event prediction：** Pre-training 任务，学到通用 address representation
- **Masked behavior modeling：** BERT-style，遮住若干笔 tx 预测，学习行为依赖

#### 与 NLP 的对应

| NLP | On-chain sequence |
|-----|-------------------|
| Sentence | An address's tx history |
| Token | A single transaction (encoded into discrete + continuous fields) |
| Vocabulary | Discrete fields (method_id, counterparty_cluster, token_id) |
| Positional embedding | Block timestamp 或 sequence index + log(time_delta) |
| Masked LM | Masked transaction modeling，预训练 address encoder |

> **面试 sound bite：** "我把 address 的 transaction history 当作 sentence，每笔 tx 当作一个 multi-field token，用 transformer 学到 address embedding，下游可以做 fraud classification、entity linking、和 anomaly detection。" 这是 Q2 的核心答法。

---

## 5. 区块链欺诈模式（Fraud Patterns in Blockchain）

了解了数据视角后，必须把它对应到 **具体的 fraud type**，否则 ML 是空的。下面 6 类是 OKX 这种 CEX 在链上最关心的 fraud pattern（合规层面 + 用户保护层面）。

---

### 5.1 洗钱（Money Laundering）

**链上 typical pipeline：** placement → **layering** → integration。链上 layering 比传统银行更复杂，主要手法：

- **Mixing service（Tornado Cash 类）：** 多个用户把 ETH/token deposit 到同一个 mixer pool，过一段时间从 mixer 提到新 address。链上看到的是"X 个 EOA 在不同时间存入同一 contract，又有 X 个 EOA 在更晚时间从同一 contract 提走相同金额"。
- **Chain hopping：** 用 cross-chain bridge 把资金从 Ethereum 转到 BSC / Tron / Solana，断开 graph 连续性。
- **Fan-out / fan-in pattern：** 把一笔大额拆成几十上百笔小额分散到中间 address，再汇集到 destination address。中间层的 address 通常生命周期极短、单一用途。
- **Peel chain：** 像剥洋葱——每笔 tx 拿一小部分去 cash-out（CEX deposit），剩下大部分进入新 address 继续 peel。

**On-chain detection signals：**
- 与已知 mixer 的 1-hop 接触
- Address 的 in/out flow 在短时间窗口内几乎相等且金额接近（quick pass-through）
- Address lifecycle 极短（首次入金后 24h 内全部出金，再无活动）
- 与 bridge contract 接触后下游 chain 出现"对称"的 unlock 事件

---

### 5.2 庞氏骗局（Ponzi Schemes）

链上的 Ponzi 通常以 smart contract 形式存在（如 Forsage、PlusToken 部分模块），承诺高 yield。

**核心机制：** 新进入的投资者的资金被直接转给早期投资者，没有真实收益来源。

**On-chain detection signals：**
- **Payout traceability：** 用 trace 工具能 explicitly 看到"new deposit tx 的 value 在同一 block / 几个 block 内被 forward 到旧 address"
- **Contract 状态特征：** TVL（total value locked）随时间下降，但 deposit 仍在持续 → 不是正常 yield 协议
- **投资者增长曲线：** unique depositor 数 ~ 指数级增长后突然 plateau 或断崖
- **Payout 集中度：** 早期 N 个 address 收到了绝大部分 payout，但他们的 deposit 极少
- **Contract owner 行为：** owner address 频繁 sweep 资金到外部 EOA，且无任何"yield generation"相关的对外调用

---

### 5.3 钓鱼 / 地址污染（Phishing / Address Poisoning）

利用人类视觉惯性的攻击：用户从历史 tx 复制粘贴地址时，可能复制到攻击者预先 "poisoning" 进 history 的伪造地址。

**Address poisoning 机制：**
- 攻击者预先生成一批"前几位 + 后几位与受害者常用地址相同"的 vanity address（用 GPU 或 profanity 类工具）
- 给受害者发送 **dust** 数量（如 $0.01 USDT）的 tx，让这些伪造地址出现在受害者钱包的 tx history 顶部
- 受害者下次转账时如果只核对地址首尾几位，就会复制到攻击者地址

**链上 signal：**
- 极小金额的 token transfer，from 是从未见过的新 address
- 该新 address 与受害者真实常用对手方地址在前 4-6 位和后 4-6 位完全相同
- 攻击者 address 在短时间内向成百上千个受害者发起 dust 转账（fan-out spam pattern）

**钓鱼合约：** 另一种是 phishing dApp 诱导用户 `approve` token 给 malicious contract，然后 drain。signal 包括"用户对一个新 contract 发起 `approve(spender, max_uint256)`，紧接着该 contract 在 30 分钟内把 user 的 token 全部 transferFrom 走"。

---

### 5.4 闪电贷攻击（Flash Loan Attacks）

DeFi 独有的攻击形式。Flash loan 允许 borrower 在 **同一笔 transaction** 内借出大额资金并归还（不还就 revert）。攻击者用这点临时获得巨额资金来操纵 DeFi 协议。

**典型攻击 pattern（all in 1 atomic tx）：**
1. 从 Aave / dYdX flash-borrow 1000 万美元等值的 token
2. 用借来的资金砸盘某个低流动性 token，制造价格异常
3. 在被操纵的价格上 mint / redeem / liquidate，套利
4. 归还 flash loan + 手续费
5. 落袋差额

**On-chain signal（非常 distinctive）：**
- **极高 gas_used：** 单笔 tx 涉及数十次 contract call，gas_used 通常 > 1M（正常转账 21k）
- **极高 value flow：** 单 tx 的内部 transfer 总额 > $1M
- **Flash loan event marker：** 同一 tx 内有 `FlashLoan` 或 `flashLoan` event emit
- **Atomic 套利结构：** tx 起点和终点是同一 EOA，中间经过多个 protocol contract，end-of-tx 时 attacker 净余额增加
- **价格异常：** 同一 block 内某 pair 的价格出现 spike → revert 模式

---

### 5.5 抹布拉盘 / Rug Pull

代币项目方在拉高币价、吸引散户买入后，**移除流动性 / 卖出团队持仓**，导致币价归零。

**类型分类：**
- **Hard rug：** Contract 有恶意 backdoor（如 mint unlimited token、disable transfer），项目方直接 drain
- **Soft rug：** 没有 backdoor，但 team 偷偷抛售或撤池子

**On-chain signal：**
- **流动性突降：** DEX pair 的 reserve 在短时间内被单一 address（通常是 deployer）抽走
- **Deployer 大额抛售：** Token contract 的 deployer / owner address 在短时间内向多个 CEX deposit address 转入大量该 token
- **持仓集中度：** Token top-10 holder 占比 > 80%，且大量在 deployer 关联 cluster 内
- **合约权限：** owner 可调用 `mint()` 或 `pause()` 等危险方法且 ownership 未 renounce
- **Honey pot 行为：** Tx history 显示 "买入成功率 100%，卖出几乎 100% revert"

---

### 5.6 虚假刷量（Wash Trading）

用户用自己控制的多个 address 互相交易，制造虚假交易量（在 NFT 市场尤其严重，常见于 LooksRare、Blur 早期挖矿激励期）。

**Pattern：**
- **直接互相买卖：** A → B → A，循环 N 次
- **多 hop 闭环：** A → B → C → ... → A，资金最终回流
- **同步行为 cluster：** 一组 address 在相同时间窗口内做相同方向的小额 swap

**On-chain detection signals：**
- **闭环资金流：** Address 之间的净 value flow 接近 0（gross flow 巨大但 net flow 小）
- **High self-clustering：** 几个 address 之间的双向 edge 形成 dense subgraph
- **Funding 同源：** 这些 address 都从同一个 funder address 收到 initial gas
- **时间同步性：** 行动时间 cluster 在相同的几分钟窗口
- **NFT 场景特有：** 同一 NFT 在短时间内多次易主，每次价格变化巨大，但买卖 address 都属于同一 cluster

---

## Interview Q&A

---

### Q1: UTXO 模型和账户模型的区别是什么？这种区别对 feature engineering 有哪些具体影响？

**回答：**

UTXO 和 account model 是两种根本不同的链上数据范式，差异远不止"Bitcoin 用 UTXO、Ethereum 用 account"这么简单，它直接决定了我们如何构建 address-level feature。

1. **数据 schema 的本质差异**：UTXO 模型里没有 "balance" 这个 first-class 字段——一个 address 的余额必须通过 scan 全链 unspent outputs 累加得出，每笔 tx 是 *消耗一批 UTXO → 产出一批新 UTXO*；account 模型里 balance 是显式存储的状态字段，每笔 tx 直接修改 `from.balance` 和 `to.balance`。对 feature engineering 的第一个影响：在 UTXO 上算"at-time-of-tx, address 持有多少 BTC" 必须 reconstruct 时序 UTXO set 快照，而 account model 上这个特征几乎 free。

2. **Identity granularity 不同**：UTXO 模型鼓励用户每次收款都用新 address（best practice 是 one-time address），所以 address ≠ user。Bitcoin 上一个真实用户可能控制几十上百个 address。我们必须先用 **common-input heuristic**（同一 tx 的所有 input 假设由同一 entity 控制）做 address clustering，再在 cluster 粒度上算 feature；account model 上 EOA 倾向于长期复用，address 更接近 entity，feature 可以直接在 address 上算。

3. **Feature 含义变化**：在 UTXO 上 "transaction count" 这个 feature 在 address 粒度毫无意义（用户故意打散），必须在 cluster 粒度算；而 Ethereum 上 EOA 的 `nonce` 直接给你一个 monotonic transaction counter，免费拿到 activity intensity。"Counterparty count" 在 UTXO 上要去重 cluster，在 account 上去重 address 即可。

4. **资金追踪难度**：UTXO 的 input 显式指向上游 tx 的具体 output，给定一个 UTXO 你可以 100% 回溯它的 provenance（taint tracking 在 UTXO 上是 well-defined operation，FIFO / haircut 这些 taint model 都有形式化定义）；account model 上由于 balance 是 fungible 的，taint 实际上是估计问题——一个 address 收到 10 ETH dirty money 和 10 ETH clean money 后，再转出 5 ETH，"这 5 ETH 多脏" 取决于你选哪种 taint propagation rule。

5. **Internal transaction 与 token 处理**：account model 因为有 smart contract，引入了 **internal tx**（contract-to-contract sub-call）和 **token transfer event**（ERC-20 不出现在 main tx 表）这两层数据，feature pipeline 必须同时消费 top-level tx、internal tx trace、event log 三套数据源；UTXO 没有 smart contract（除了 Bitcoin script 极简 case），数据源单一。

> **Follow-up 提示：** 面试官可能追问"那 graph feature 在两个模型上如何统一？"——回答：在两个模型上都先 normalize 到 **entity-level transfer graph**。UTXO 通过 address clustering 折叠到 entity，再构 entity → entity edge；account model 先把 top-level tx + ERC-20 event 合并成统一的 transfer event，再构 EOA → EOA edge（contract address 通常作为特殊节点或通过 method-aware decoding 展开）。统一之后，PageRank / community detection / GNN 这些算法的代码路径就完全一致了。

---

### Q2: 如何把一个区块链地址的交易历史构造成 Transformer 模型的输入序列？

**回答：**

把 address tx history 输入 transformer 的核心思路是 "**把每笔 transaction 当作一个 multi-field token**"，下面是工程上的完整 pipeline。

1. **Sequence 定义与排序**：对每个 target address，拉取它作为 sender 或 receiver 的所有 tx（在 account model 上再加上以它为参与方的 ERC-20 Transfer event），按 `block_timestamp` + `block_index`（同 block 内的次序）严格排序，得到长度为 N 的 event list。为了控制 sequence 长度，通常取最近 K 笔（如 K=256）或最近 T 天的 tx，再做 sliding window。

2. **Token encoding（每笔 tx → embedding）**：每笔 tx 不是单个 categorical token，而是一组 field 的组合：
   - **Discrete fields：** direction (in/out)、counterparty_cluster_id（hashed or top-N + OOV bucket）、method_id（取 input 前 4 字节解码后的 function name，常见的 transfer/approve/swap/bridge 等单独 token，长尾合并）、token_id（ETH / USDT / USDC / 其他）、interaction_type（EOA-to-EOA / EOA-to-DEX / EOA-to-bridge / EOA-to-mixer 等粗粒度 contract tag）
   - **Continuous fields：** $\log_{10}(value\_usd + 1)$、$\log_{10}(gas\_price)$、$\log_{10}(\Delta t + 1)$（距离上一笔的秒数）、$\text{value\_to\_balance\_ratio}$
   - **Token embedding：** 对每个 discrete field 学一个 embedding table，对 continuous field 做 normalize 后过一个小 MLP 或 sinusoidal bucket embedding，然后把所有 field 的 vector concat / sum / cross-attention 融合成最终的 token vector $x_{t}$

3. **Position encoding 设计**：不要简单用 index-based positional embedding。我会用两层位置信号：
   - **Index position：** 标准 sinusoidal 或 learnable，表示在 sequence 中的相对位置
   - **Time-aware position：** 把 $\log_{10}(\Delta t)$ 作为额外的"时间位置"信号 concat 进去，因为链上事件的语义高度依赖时间间隔——"两笔 tx 间隔 5 秒"和"间隔 5 天"代表完全不同的行为

4. **预训练任务（self-supervised）**：链上 labeled fraud 数据极稀缺，所以一般先 pretrain 再 fine-tune：
   - **Masked transaction modeling (BERT-style)：** 随机 mask 15% 的 tx，预测其 method_id / counterparty_cluster 等 discrete field
   - **Next event prediction (GPT-style)：** 自回归预测下一笔 tx 的 field
   - **Contrastive learning：** 同一 address 的不同时间窗口为正样本对，不同 address 为负样本，得到 address-level embedding

5. **Downstream task 适配**：
   - **Fraud classification：** 在 sequence 末尾加 `[CLS]` token，取其 final hidden state 接 MLP 二分类
   - **Address embedding 提取：** 直接用 `[CLS]` 或 mean-pool 全 sequence 作为 address representation，feed 给 graph model 当 node feature
   - **Anomaly detection：** 用 reconstruction loss / next-token perplexity 当 anomaly score，不依赖 label

6. **工程上的坑**：
   - **Vocabulary 长尾：** counterparty cluster 和 contract 极其长尾，必须做 top-K + OOV bucket，否则 embedding table 爆炸且尾部学不到
   - **Length skew：** 活跃 address 可能有几万笔 tx，僵尸 address 只有 1-2 笔，要做 truncation + bucketed batching 否则训练效率极低
   - **Look-ahead leakage：** 训练时 sequence 末尾不能包含 label 后的 tx，否则未来信息泄漏到 representation
   - **Adversarial drift：** 攻击者会调整行为模式，预训练 corpus 要定期 refresh，且生产环境要监控 sequence distribution shift

> **Follow-up 提示：** 面试官可能追问"为什么不直接用 LSTM / GRU？" —— 答：transformer 的优势在 ① 可并行训练（链上数据量巨大）；② 长距离依赖更好（peel chain、layering 这类 fraud 的 signal 可能跨数百笔 tx）；③ multi-field token 可以用 cross-field attention 学到 field 之间的 interaction；④ 预训练范式成熟，更容易做 label-scarce setting。LSTM 仍可作为 lightweight baseline，但 transformer 是 SOTA 方向。

---

### Q3: Flash loan attack 在链上有哪些可检测的特征？怎么设计一个 detector？

**回答：**

Flash loan attack 是 DeFi 独有的攻击模式，幸运的是它在链上的 signal 非常 distinctive，detector 设计相对清晰。

1. **核心 atomicity signature**：Flash loan 的本质是"借-用-还" 必须发生在同一笔 transaction 内（否则 revert）。这导致整个攻击在链上呈现为 **单笔 tx 内包含极其复杂的 internal call sequence**。具体可观测特征：
   - 单笔 tx 的 `gas_used` 异常高，通常 > 1M（普通 ERC-20 transfer ~50k，标准 DEX swap ~100-250k，flash loan attack 常见 2M-10M）
   - 单 tx 内 internal call count 通常 > 20，跨越多个不同 protocol contract
   - Tx trace 中出现 `flashLoan` / `flashBorrow` / `executeOperation` 等 Aave/dYdX/Balancer 的 flash loan event marker

2. **资金流 atomic 闭环**：攻击者必然把借来的钱在同一 tx 内还回去，所以 trace 中能看到 "attacker EOA → flash loan provider → 借出 → 一系列 protocol interaction → 归还 flash loan + premium → attacker EOA 净余额 > 0"。net flow 的"借→还" 在 1 个 block 内完成是关键区别于普通 arbitrage 的特征。

3. **价格操纵 signature**：大多数 flash loan attack 的目的是操纵某个低流动性 pool 的价格，进而 exploit 依赖该价格 oracle 的 protocol。链上可见：
   - 同 block 内某个 DEX pair 的 reserve 发生剧烈变化（>20% 偏离 30-block 均值），紧接着同 tx 内有 `borrow` / `liquidate` / `redeem` 等依赖价格的操作
   - 该 pool 在 attack tx 结束后 reserve 回到接近攻击前的水平（因为攻击者也要 unwind 抛压）
   - Oracle event log 与 swap event log 的 timestamp 差距非常小

4. **Detector 工程设计（real-time + post-hoc 两层）**：
   - **Layer 1 (cheap pre-filter)：** 在 streaming pipeline 上对每笔 tx 做规则 filter：`gas_used > 1M` AND (`internal_call_count > 20` OR `flashLoan event present`)。这是低 precision 但保证 100% recall 的初筛
   - **Layer 2 (feature model)：** 对通过 Layer 1 的 tx 提取详细 feature 喂给 GBM / DNN：
     - Internal call sequence 上调用的 unique protocol 数
     - 涉及的 pool 是否出现 >X% 价格偏离
     - Attacker EOA 是否为新地址（< 24h、< 5 historical tx）
     - 涉及的 token 是否包含已知 vulnerable target（基于历史 incident 库）
     - Tx end-state 下 attacker 净资产增量 / borrowed amount 比例
   - **Layer 3 (graph + sequence context)：** 用 GNN 看 attacker EOA 在过去 N 天的 graph neighborhood 是否与已知 attacker cluster 相连；用 sequence model 看 attacker 是否有 reconnaissance 模式（小额测试 tx → 大额攻击 tx）

5. **挑战与对抗**：
   - **正常 MEV arbitrage 也用 flash loan：** 单纯靠"用了 flash loan" 会产生大量 false positive。区分点：合法 MEV 的 victim 是 inefficient pricing，没有 protocol 被 exploit；attack 的 victim 是 specific protocol（其 TVL 短期下降可证实）
   - **Multi-tx attack：** 高级攻击者把单 atomic tx 拆成 reconnaissance tx + main tx，用 commit-reveal 等手法绕过单 tx atomic signature。这时需要在 sequence 模型里 catch 短时间内同一 attacker cluster 的协同行为
   - **Latency：** Real-time block ~12s（Ethereum），detector 必须在亚秒级完成评分，否则资金已经过 bridge 跨链消失

> **Follow-up 提示：** 面试官可能问"如果你是 OKX，能在 CEX 层面做什么 mitigation？"——答：CEX 没法阻止链上 attack 本身，但可以在 deposit / withdrawal 边界做 control：① 已知 attacker cluster 的 address 来源资金触发 enhanced due diligence、freeze、报告；② 监控 attacker 把资金提现到 OKX 的 deposit address，结合 attack 发生时间 + 资金路径在分钟级内冻结；③ 与 Chainalysis / TRM Labs 等数据供应商联动获取 attack alert。

---

### Q4: 为什么图模型（graph model）对区块链欺诈检测特别重要？相比于纯 tabular feature 有什么不可替代的优势？

**回答：**

图模型在链上 fraud detection 是 first-class citizen，不是"锦上添花"。原因可以从数据本质、欺诈模式、和工程实践三个层面回答。

1. **数据本质：链上数据 first-class 是 graph，tabular 是 derived view**：传统金融的 transaction record 是以 account 为中心的 ledger，关系数据是次生的；而链上原始数据格式就是 `(from, to, value, ...)` 这种 edge list，**图结构是 raw data 的天然 schema**。把它拍平成 tabular 一定会丢信息——比如 address A 通过 5 hop 到达 sanctioned address 这个事实，在 tabular feature 里要么不存在，要么要预先 hardcode 一个 "K-hop reachable to bad" feature；而 graph model 端到端就能学到这种结构。

2. **欺诈模式本质是"关系模式"**：链上欺诈 rarely 体现为"单个 address 自己看起来异常"。绝大多数 fraud signal 是 **关系层面** 的：
   - **Mixer 检测：** 不是某个 address 自己异常，而是它和已知 mixer 的距离、它 1-hop 内的 mixer 邻居数等关系特征
   - **Layering：** 关键不是单个 hop 的金额，而是"短时间内 N 跳之后资金回流到同一 cluster"这种 graph topology
   - **Wash trading：** 本质是 dense subgraph + 短闭环，tabular 完全无法表达"这几个 address 之间有强密度互动"
   - **Coordinated attack：** 多个 attacker EOA 同时启动且有共同 funder，只有在 graph 上看才有 signal
   GNN 通过 message passing 能在 K hop 邻域内 aggregate 这些 relational signal，得到的 node embedding 天然包含 "我是谁 + 我周围是谁" 的复合信息。

3. **Label efficiency：在 graph 上做半监督学习**：链上 labeled fraud 极少（Chainalysis、TRM 等的标签库总量也就几十万 address，相对全链十亿级 address 来说极稀疏）。GNN 的 message passing 本身就是一种 label propagation 机制——给定少量 seed bad address，邻居 embedding 会受到污染从而被 detector 识别。这是纯 tabular model 做不到的"标签外溢"。

4. **Entity resolution / address clustering 是图原生任务**：把分散的 address 聚合成 user-level entity 本身就是 graph clustering 问题（common-input heuristic、change address detection、相同 funder 同源 detection 等）。这一步是后续所有 fraud detection 的基础设施，本质上不能用 tabular 方法替代。

5. **可解释性优势**：对 compliance / SAR filing，detector 必须给出可被 investigator review 的 evidence。Graph model 的输出天然支持 **subgraph extraction** —— "这个 address 之所以被判为 high risk，是因为它和 OFAC sanctioned address X 之间有 2-hop path，途经地址 Y、Z；其中 Y 在过去 30 天和 5 个已知 mixer 有交互"。这种"证据子图" 是 tabular black-box model 给不出来的。

6. **工程实践：tabular 和 graph 互补，不是二选一**：实际生产中是 hybrid 架构：
   - Tabular feature 捕捉 address 自身行为（volume / temporal / amount pattern）
   - Graph feature（degree、PageRank、cluster id）作为额外 column 喂给 GBM，是 1.0 baseline
   - GNN embedding 作为 dense feature 进入下游 model 是 2.0 升级
   - 最高级的方案是 **end-to-end 联合训练**：把 tabular + graph + sequence 三种 view 在一个统一 model 里 fuse
   面试时强调"图模型重要"但也别贬低 tabular —— 二者结合才是 production reality。

> **Follow-up 提示：** 面试官可能问"GNN 在 billion-scale address graph 上怎么训练？" —— 答：① 用 neighborhood sampling (GraphSAGE / PinSAGE) 把 full-graph training 转成 mini-batch；② 用 cluster-GCN / GraphSAINT 等 subgraph sampling 方法；③ 工程上用 DGL / PyG + 分布式（DGL-distributed、PyG with NeighborLoader on GPU）；④ 实时 inference 用预计算 embedding + 增量更新（只对最近活跃 address 重算邻居 aggregation）。这部分细节会在 File 3 graph ML 文档详细展开。

---

### Q5: Wash trading 在链上要怎么检测？给一个 end-to-end 的思路。

**回答：**

Wash trading 的本质是 "同一个 economic entity 用不同 address 互相交易制造虚假 volume"。在 NFT 市场（LooksRare/Blur 早期）和某些 DEX token 上尤其严重。下面是一个 end-to-end 的检测思路。

1. **问题定义与 label 来源**：Wash trading 的 ground truth 难拿，常用替代：① 已知 token 项目方公布的 wash address 黑名单；② Chainalysis / Nansen 等数据提供商的 label；③ 历史上被 marketplace 官方判定为 wash 的 trade（如 LooksRare 后期取消的 reward eligibility）；④ 自标注：基于强 rule 标一批 high-confidence positive 作为 weak label。建模时通常采用 **PU learning** 或 anomaly detection，因为 negative（真实 organic trade）数据天然占多数但不是显式标注。

2. **特征层一：闭环检测（graph topology）**：
   - **直接闭环：** 检测 `A → B → A` 这种 2-cycle 在窗口期内的交易，且 net flow 接近 0（gross 巨大、net 极小）
   - **多 hop 闭环：** 找长度 3-5 的 directed cycle，且 cycle 上的 net flow / gross flow 比例极小
   - **Dense subgraph：** 找 small subgraph（3-10 node），内部 edge density 远高于全图平均，且 internal/external transaction ratio 异常高
   - **Algorithm：** 用 Johnson's algorithm 找 simple cycle，或者用 community detection 找 tight community + 内部 trade 占比作为 feature

3. **特征层二：资金同源（funding analysis）**：Wash trader 控制的所有 address 通常需要从同一个 funder 获取 gas 和初始资产，所以：
   - 追溯每个 address 的 **first funding tx**（首次收到 ETH/native token 的来源）
   - 把 funding 同源的 address 聚成 cluster
   - 在 cluster 粒度计算 internal trade 占总 trade 的比例 —— 高比例 + 高 trade frequency 是强 signal

4. **特征层三：时间同步性（behavioral sync）**：
   - 嫌疑 cluster 内的 address 在时间序列上是否高度同步（在相同 minute / hour bucket 内同时活跃）
   - 单个 NFT 在短时间内 N 次易主，且每次买卖双方都在同一 cluster
   - Trade interval distribution 异常 regular（人工 bot 写的脚本通常间隔近似常数或 follow 简单规律）

5. **特征层四：经济非理性（economic irrationality）**：
   - 同一 NFT 在短期内价格变化巨大且非市场原因（无显著 news / floor price shift）
   - 卖家 collected 的 ETH 立即返回给买家 funder（净亏损 = marketplace fee + gas，但赚到 token reward）
   - **Reward farming pattern：** 在有 trade-to-earn 激励的 marketplace 上，wash trade 的目标函数是 reward >> fee + gas，回归到 reward issuance schedule 就能识别

6. **模型架构**：
   - **Graph-side：** GNN 在 address graph 上学 node embedding，目标是把 wash cluster 的 address embedding 推近、与 organic trader 推远（contrastive loss with weak supervision）
   - **Tabular-side：** 把上面四层 feature 喂给 LightGBM，作为 strong baseline
   - **Sequence-side：** 用 transformer 对每个嫌疑 address 的 tx history 做行为分类（bot-like vs human-like）
   - **Ensemble：** 三路 score 加权后过阈值，配合 cluster-level 而不是 address-level decision

7. **评估与上线**：
   - **Offline 评估：** 在已知 positive label 上看 recall，在 organic trader（如有声誉 collector address）上看 precision；用 PR-AUC 而不是 ROC-AUC（label 极不平衡）
   - **Sanity check：** 检查 model 是否过度依赖某个 marketplace 的特殊 schema，应该跨 marketplace 验证
   - **生产部署：** Cluster-level 决策；对边缘 case 走 human-in-the-loop review；feedback loop 把 confirmed wash cluster 回流到 label set 持续训练

8. **对抗（adversarial concern）**：Wash trader 会演化：
   - 增加 cluster 内 address 数（从 2-3 扩到几十），稀释 cycle signal → 应对：cluster 粒度的统计 feature 比 pairwise cycle 更稳健
   - 引入 noise tx（夹杂少量真正向外部的 trade）扰动 net flow → 应对：仍然看 gross/net ratio 而非绝对 net flow
   - 用 mixer 模糊 funding 同源 → 应对：funding 不是唯一 signal，时间同步 + 行为 sync 仍然有效

> **Follow-up 提示：** 面试官可能问"如果发现 wash trader 已经在 OKX 充提怎么处理？"——答：① 触发账户级 review，结合 OKX 内部 KYC 信息看是否多账户实际同 user；② 如果是 trade-to-earn reward farming，可以追回未释放奖励 + 拉黑后续 eligibility；③ 严重 case 上报 compliance team 评估是否触发 market manipulation 范畴的合规义务；④ 把识别出的 cluster 反馈给链上 model 作为 high-confidence label，闭环。
