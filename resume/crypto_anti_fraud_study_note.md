# Crypto Anti-Fraud 学习笔记（从传统 Fintech ML 视角出发）

---

## Section 1: Blockchain 基础（Blockchain Fundamentals）

---

### 1.1 什么是 Blockchain

Blockchain 是一个分布式、不可篡改的账本（distributed immutable ledger），所有交易记录按照时间顺序打包成 block，每个 block 通过 cryptographic hash 链接到前一个 block，形成 chain。

**核心特性：**
- **Decentralization：** 没有中心化的控制方，由网络中的 node 共同维护
- **Immutability：** 一旦写入，数据极难被篡改（需要控制 51% 算力/质押量）
- **Transparency：** 所有交易公开可查（public chain）
- **Pseudonymity：** 地址是匿名的，但交易记录是公开的

### 1.2 UTXO 模型 vs Account 模型

**UTXO (Unspent Transaction Output) — Bitcoin 的数据模型：**
- 每笔交易消耗一些 UTXO（input），生成新的 UTXO（output）
- 余额不存在于某个"账户"中，而是分散在各个 UTXO 中
- 类比：现金交易——用100元钞票买50元东西，收到50元找零
- 每个 UTXO 有一个 locking script（通常需要 private key 签名才能消耗）

```
Transaction Example:
  Inputs:  UTXO_A (0.5 BTC from Alice)
           UTXO_B (0.3 BTC from Alice)
  Outputs: UTXO_C (0.7 BTC to Bob)      ← payment
           UTXO_D (0.09 BTC to Alice)    ← change (找零)
           (0.01 BTC implicit fee)       ← miner fee
```

**Account 模型 — Ethereum 的数据模型：**
- 每个地址有一个状态（balance、nonce、code、storage）
- 交易直接修改发送方和接收方的余额
- 类比：银行转账——直接修改账户余额
- 两种账户类型：
  - **EOA (Externally Owned Account)：** 由 private key 控制，只有 balance 和 nonce
  - **Contract Account：** 由 smart contract code 控制，有 code + storage + balance

**对 Fraud Detection 的关键影响：**

| 维度 | UTXO (Bitcoin) | Account (Ethereum) |
|------|---------------|-------------------|
| **地址复用** | 鼓励每次交易用新地址（privacy） | 地址通常重复使用 |
| **Address clustering** | **Common-input heuristic** 非常有效 | 需要行为分析、无直接 heuristic |
| **Fund tracing** | 精确追踪每个 UTXO | 余额混合，tracing 更难 |
| **Transaction graph** | DAG（有向无环图） | General directed graph |
| **Smart contract fraud** | 有限 | 主要战场 |
| **数据规模** | Bitcoin ~1B UTXO | Ethereum ~250M 地址 |

**与 PayPal 的对比：**
- PayPal 更类似 Account 模型（单一余额、直接修改）
- PayPal 有完整的 KYC 信息，crypto 地址是 pseudonymous
- PayPal 的 "shared asset linking"（credit card, device）在 UTXO 中有天然对应（common input），但在 Ethereum Account 模型中需要替代方案

### 1.3 EVM (Ethereum Virtual Machine)

**EVM 是什么：**
- Ethereum 的运行时环境，所有 smart contract 都在 EVM 中执行
- Stack-based 虚拟机，有 256-bit word size
- 图灵完备（理论上可以执行任何计算），但通过 gas 机制限制执行时间

**关键概念：**
- **Bytecode：** Smart contract 编译后的二进制代码，存储在 contract account 中
- **Opcode：** EVM 指令集（如 ADD, MUL, SSTORE, CALL 等），每个 opcode 有固定 gas cost
- **State：** EVM 维护全局状态（所有账户的 balance, code, storage）
- **Transaction execution：** 每个 transaction 调用 EVM 执行，修改 state

**对 fraud detection 的意义：**
- 可以分析 bytecode/opcode 来检测恶意合约（见 Section 6）
- Gas consumption pattern 是 address fingerprinting 的 signal
- Internal transactions (trace)：一个 transaction 可能触发多个内部调用，需要用 debug_traceTransaction 获取完整执行路径

### 1.4 Gas 机制

**Gas 基础：**
- 每个 EVM operation 消耗 gas（如 ADD = 3 gas, SSTORE = 20000 gas）
- Transaction 发起者设定 gas limit 和 gas price
- 实际费用 = gas used × gas price
- 如果 gas 不够 → transaction revert（但 gas 不退还）

**EIP-1559 改革（2021 年 London 升级）：**
- 引入 base fee（根据网络拥堵自动调整）+ priority fee（给 validator 的小费）
- 实际费用 = gas used × (base fee + priority fee)
- Base fee 被 burn（销毁），不给 validator
- 用户设定 maxFeePerGas 和 maxPriorityFeePerGas

**Gas 作为 Fraud Signal：**

| Signal | 含义 | 关联的 Fraud Pattern |
|--------|------|---------------------|
| 异常高 priority fee | 愿意付高溢价优先打包 | MEV bot, front-running |
| 异常高 gas used | 执行了大量 opcode | Complex malicious contract, flash loan attack |
| Gas limit >> gas used | 预留了远超需要的 gas | 探测/测试行为 |
| 大量 failed transactions | 交易频繁失败 | Honeypot token (sell function revert) |
| 固定 gas strategy | 每笔交易 gas 设置一致 | Bot 行为特征（自动化程序） |

### 1.5 共识机制概览

| 机制 | 代表链 | 原理 | 特点 |
|------|-------|------|------|
| **PoW (Proof of Work)** | Bitcoin, (旧 Ethereum) | 消耗算力竞争出块权 | 安全但能耗高、TPS 低 |
| **PoS (Proof of Stake)** | Ethereum 2.0 | 质押 token 竞争出块权 | 节能、TPS 较高 |
| **PoSA (Proof of Staked Authority)** | BSC | PoS + 有限验证者集合 | 快速出块、中心化风险 |
| **PoH (Proof of History)** | Solana | Verifiable delay function 做时间戳 | 极高 TPS（~3000+） |

**对 fraud detection 的影响：**
- PoW/PoS 的 finality 时间影响 real-time fraud detection 的延迟
- 不同共识机制下 MEV 的表现不同（PoS 的 MEV 主要通过 validator 排序）
- 出块速度影响 data pipeline 的处理吞吐量

---

## Section 2: On-Chain Data 数据源与查询（Data Sources & Querying）

---

### 2.1 数据源全景

| 数据源 | 类型 | 说明 | 适用场景 | 费用 |
|--------|------|------|---------|------|
| **Full Node (Geth/Erigon)** | 原始数据 | 运行完整以太坊节点，获取所有 block/transaction/state 数据 | 需要完整数据、高频查询 | 硬件成本（几 TB SSD） |
| **Archive Node** | 历史数据 | 保存所有历史 state，支持任意 block number 的 state 查询 | 历史分析、回溯审计 | 存储成本极高（> 10 TB） |
| **Etherscan API** | Indexed API | 第三方 block explorer，提供 REST API | 快速查询、原型开发 | 免费 tier + 付费（rate limit） |
| **Dune Analytics** | SQL 分析 | 将 blockchain data 导入 SQL 数据库，支持 SQL 查询 | 探索性分析、dashboard | 免费 + Pro |
| **The Graph** | Decentralized indexing | 定义 subgraph，index 特定 contract event | DeFi protocol-specific 数据 | Per-query payment (GRT token) |
| **Alchemy / Infura / QuickNode** | Node-as-a-Service | 提供 RPC endpoint，免去运行自己节点 | 开发和 production 查询 | Per-request pricing |
| **Chainalysis / Elliptic** | 商业 intelligence | 地址归因、risk score、entity resolution | Compliance、investigation | 企业级 licensing |
| **Nansen** | 标注数据 | 大量地址已标注（exchange, whale, MEV bot 等） | 分析、labeling | Enterprise pricing |

### 2.2 核心数据表结构

**Ethereum 的关键数据表（以 BigQuery Public Dataset 为参考）：**

**blocks 表：**
```sql
SELECT number, timestamp, miner, gas_used, gas_limit, base_fee_per_gas, transaction_count
FROM `bigquery-public-data.crypto_ethereum.blocks`
```

**transactions 表：**
```sql
SELECT hash, block_number, from_address, to_address, value, gas, gas_price,
       input, nonce, transaction_index, receipt_status
FROM `bigquery-public-data.crypto_ethereum.transactions`
```
- `input`: 调用 smart contract 的 calldata（前 4 字节是 function selector / method ID）
- `receipt_status`: 1 = success, 0 = failed

**logs 表（event logs）：**
```sql
SELECT log_index, transaction_hash, address, topics, data
FROM `bigquery-public-data.crypto_ethereum.logs`
```
- `address`: 发出 event 的 contract address
- `topics[0]`: event signature hash（如 Transfer event = `0xddf252ad...`）
- `topics[1], topics[2]`: indexed parameters
- `data`: non-indexed parameters

**traces 表（internal transactions）：**
```sql
SELECT transaction_hash, trace_address, from_address, to_address, value,
       call_type, gas, gas_used, input, output, error
FROM `bigquery-public-data.crypto_ethereum.traces`
```
- Internal transactions 不是独立的 transaction，而是 EVM 执行过程中的 CALL/DELEGATECALL/CREATE
- 对 fund tracing 至关重要：一笔 transaction 可能通过 internal call 转移资金到多个地址

**token_transfers 表（ERC-20 Transfer events）：**
```sql
SELECT token_address, from_address, to_address, value, transaction_hash, block_number
FROM `bigquery-public-data.crypto_ethereum.token_transfers`
```

### 2.3 Data Pipeline 架构

```
┌─────────────────────────────────────────────────────┐
│ Layer 1: Data Ingestion                             │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │Full Node │  │WebSocket │  │Third-party APIs  │  │
│  │(Erigon)  │  │listener  │  │(Etherscan,       │  │
│  │          │  │(new blocks│  │ Chainalysis)     │  │
│  │          │  │ & txs)   │  │                  │  │
│  └────┬─────┘  └────┬─────┘  └────────┬─────────┘  │
│       └──────────────┼────────────────┘             │
│                      ↓                              │
│              Kafka (message bus)                     │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│ Layer 2: ETL & Processing                           │
│                                                     │
│  Spark / Flink                                      │
│  ├── Parse block + transactions + logs + traces     │
│  ├── ABI decode (calldata → function + params)      │
│  ├── Token transfer extraction                      │
│  ├── Address profile aggregation                    │
│  ├── Feature computation (windowed aggregations)    │
│  └── Graph edge generation                          │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│ Layer 3: Storage                                    │
│                                                     │
│  ┌──────────────┐  ┌───────────┐  ┌──────────────┐ │
│  │Data Warehouse│  │Graph DB   │  │Feature Store │ │
│  │(BigQuery /   │  │(Neo4j /   │  │(Redis /      │ │
│  │ Snowflake)   │  │ Neptune)  │  │ Feast)       │ │
│  │              │  │           │  │              │ │
│  │Analytics &   │  │Address    │  │Real-time     │ │
│  │ad-hoc query  │  │graph &    │  │ML feature    │ │
│  │              │  │compliance │  │lookup        │ │
│  │              │  │knowledge  │  │              │ │
│  └──────────────┘  └───────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────┘
```

### 2.4 Erigon vs Geth

| 维度 | Geth | Erigon |
|------|------|--------|
| **存储效率** | ~1 TB (full sync) | ~2 TB (archive mode) vs Geth archive ~12+ TB |
| **同步速度** | 较慢 | 快得多（flat data model） |
| **Archive mode** | 可选但存储极大 | 默认 archive mode，存储效率高 |
| **API 兼容性** | 标准 JSON-RPC | 兼容 Geth API |
| **适用场景** | 通用 | 需要历史数据分析的场景 |
| **推荐** | 开发/测试 | **Production anti-fraud data pipeline** |

**为什么 anti-fraud 需要 Erigon：**
- Fund tracing 需要查询任意历史时刻的 state → archive mode 必须
- Erigon 的 archive mode 比 Geth 节省 6-10x 存储

### 2.5 与 PayPal Data Pipeline 的对比

| 维度 | PayPal | Crypto |
|------|--------|--------|
| **数据源** | 内部 DB (Teradata/BigQuery) | Blockchain nodes + APIs |
| **数据可靠性** | 内部系统，高可靠 | Blockchain 数据 immutable 且 consistent（比内部系统更可靠） |
| **数据完整性** | 内部系统有完整的用户 profile | On-chain 数据只有交易记录，缺少 off-chain context |
| **ETL** | Spark + BigQuery | 同，但需要额外的 ABI decode 和 trace parsing |
| **Graph DB** | AWS Neptune (Gremlin) | Neo4j / Neptune（可复用经验） |
| **Feature Store** | 内部平台 | Redis / Feast |

**核心迁移价值：**
- Spark ETL pipeline 的设计和优化经验直接适用
- Graph DB 的 schema 设计和查询优化经验直接适用
- Feature store 的架构模式直接适用
- 主要学习成本在 data source 层（blockchain 数据的获取和解析）

---

## Section 3: 地址聚类技术（Address Clustering）

---

### 3.1 为什么地址聚类如此重要

Blockchain 地址是 pseudonymous 的——一个 entity（个人、组织、交易所）可能控制数百到数千个地址。地址聚类的目标是将属于同一 entity 的地址归为一组，是所有 blockchain analytics 的基础。

**类比 PayPal：**
- PayPal 的 account linking（通过 shared asset 将属于同一 entity 的多个账号关联）
- 目标完全一致：entity resolution
- 但 PayPal 有 KYC 信息可以辅助验证，crypto 只能依靠链上行为

### 3.2 Common-Input Heuristic (Bitcoin UTXO)

**原理：** 如果一笔 Bitcoin 交易有多个 input，这些 input 对应的地址大概率属于同一 entity——因为发起这笔交易需要同时持有这些地址的 private key。

```
Transaction:
  Inputs:
    - UTXO from Address_A (requires private_key_A)
    - UTXO from Address_B (requires private_key_B)
  Outputs:
    - UTXO to Address_C
    - UTXO to Address_D (change)

→ Conclusion: Address_A and Address_B likely belong to the same entity
```

**与 PayPal 的映射：**
- Common-input ≈ 多个账号共享同一张信用卡的 strong linking
- 都是基于 "需要同时控制多个凭证" 的假设

**局限：**
- **CoinJoin：** 多个用户合并交易以增强隐私，产生 false positive（不同 entity 的地址出现在同一 input）
- **Detection：** CoinJoin 交易通常有特征——多个输入、多个相同金额的输出、使用特定 CoinJoin 协议（如 Wasabi Wallet 的 Whirlpool）

### 3.3 Change Address Detection (Bitcoin)

**原理：** Bitcoin 交易通常有一个 change output（找零地址），这个地址属于发送方。

**识别 heuristic：**
- 如果一个 output 是新地址（从未出现在 blockchain 上），更可能是 change address
- 如果交易只有两个 output，金额较小且是 "整数外的零头" 的更可能是 change
- 如果 output 使用了和 input 相同的 address type（如 P2PKH/P2SH），更可能是 change

**与 common-input heuristic 配合：** change address 被归入发送方的 entity cluster，扩大 cluster 范围。

### 3.4 Ethereum 的行为聚类

Ethereum 没有 UTXO 的 common-input heuristic，需要依赖行为分析。

**方法 1 — Deposit Address Pattern：**
- 交易所为每个用户生成唯一的 deposit address
- 这些 deposit address 收到资金后会统一转入 hot wallet
- 通过 "多个地址 → 单一 hot wallet" 的 pattern 可以识别交易所的 deposit addresses

**方法 2 — Behavioral Feature Clustering：**
- 计算每个地址的行为特征：
  - Transaction timing pattern（活跃时段、交易间隔分布）
  - Gas price strategy（设置 gas 的习惯）
  - Contract interaction fingerprint（常用哪些 DeFi protocol）
  - Value distribution（交易金额分布特征）
- 用 embedding + cosine similarity 做聚类

**方法 3 — Transaction Graph Clustering：**
- 同 PayPal 的 community detection：在 transaction graph 上跑 Louvain / LPA
- 经常互相交易的地址更可能属于同一 entity

**与 PayPal 的直接映射：**

| PayPal 方法 | Crypto 对应 |
|------------|------------|
| Seed-based clustering (shared credit card) | Common-input heuristic (Bitcoin) |
| Gremlin 多跳查询 | Transaction graph BFS/DFS |
| Community detection (LPA/Louvain) | 同，直接迁移 |
| Embedding similarity (AutoEncoder + LSH) | Address embedding + LSH（方法论完全一致） |

### 3.5 Cross-Chain Address Linking

**挑战：** 同一 entity 在不同链上使用不同地址，如何关联？

**方法：**
- **Bridge transaction 关联：** 在 Chain A 存入 bridge → 在 Chain B 从 bridge 取出，时间和金额匹配
- **Behavioral fingerprint：** 相似的交易时间、金额 pattern、gas strategy
- **Off-chain intel：** 同一 exchange account 在不同链上的 deposit/withdrawal address
- **ENS / domain 关联：** 同一 entity 可能在 ENS 上注册了关联域名

---

## Section 4: Crypto Fraud 完整分类（Complete Fraud Taxonomy）

---

### 4.1 分类总览

```
Crypto Fraud
├── Token/Project-level Fraud
│   ├── Rug Pull (hard / soft)
│   ├── Ponzi / Pyramid
│   ├── Honeypot Token
│   └── Pump & Dump
├── Transaction-level Fraud
│   ├── Phishing (approval phishing, fake website)
│   ├── Address Poisoning
│   ├── Dusting Attack
│   └── Fake Airdrop
├── Market Manipulation
│   ├── Wash Trading
│   ├── MEV Exploitation (sandwich, front-running)
│   └── Oracle Manipulation
├── Protocol Exploit
│   ├── Flash Loan Attack
│   ├── Reentrancy Attack
│   ├── Bridge Exploit
│   └── Logic Bug Exploit
├── Money Laundering
│   ├── Mixing / Tumbling (Tornado Cash, etc.)
│   ├── Chain Hopping (cross-chain laundering)
│   ├── DeFi Layering (multiple protocol hops)
│   └── Peel Chain (gradually peeling off funds)
└── Social Engineering
    ├── Romance / Pig Butchering Scam
    ├── Impersonation
    └── Fake Investment Platform
```

### 4.2 详细分类与传统 Fintech 映射

| Crypto Pattern | 说明 | 传统 Fintech 对应 | 检测难度 |
|---------------|------|------------------|---------|
| **Rug Pull** | 项目方抽走流动性跑路 | 商户卷款跑路 | 中（有链上 signal） |
| **Ponzi** | 新钱养旧钱 | 庞氏骗局 | 中（transaction flow 分析） |
| **Honeypot Token** | 能买不能卖的 token | — | 低（contract analysis） |
| **Phishing** | 诱骗用户签恶意交易 | 钓鱼邮件/网站 | 高（难以在链上检测，发生在链下） |
| **Address Poisoning** | 发小额交易污染交易历史 | — | 低（pattern matching） |
| **Wash Trading** | 自买自卖刷量 | 交易刷量 | 中（graph cycle detection） |
| **MEV** | 利用 transaction ordering 获利 | 内幕交易 | 高（需要 mempool 数据） |
| **Flash Loan Attack** | 单笔交易内操纵市场 | — (crypto-native) | 高（需理解 DeFi protocol） |
| **Mixing** | 隐藏资金来源 | AML layering | 中（mixer interaction 可检测） |
| **Pig Butchering** | 长期培养信任后骗投资 | Romance scam | 高（链下社交工程） |

### 4.3 重点 Pattern 详解

#### Rug Pull

**Hard Rug Pull：** 合约中有后门，项目方直接调用后门函数抽走资金
- 常见后门：hidden `mint()`、`setFee(100%)`、`removeLiquidity()` without timelock
- 检测：smart contract code analysis（见 Section 6）

**Soft Rug Pull：** 没有代码后门，但项目方逐渐抛售 token / 停止开发
- 检测更难，需要监控 token holder 集中度、developer activity、social media 活跃度

**完整 lifecycle：**
```
Deploy token → Add liquidity on DEX → Marketing hype
→ Price rises → Insiders sell / remove liquidity → Token collapses
```

**检测 signal：**
1. Token holder 集中度 > 80%（前 10 地址）
2. Liquidity 没有 time-lock
3. Contract 有 owner-only critical functions
4. Creator address 有历史 rug pull 记录
5. Buy/sell ratio 异常（只有 buy，没有 sell → 可能是 honeypot）

#### Flash Loan Attack

**Flash Loan 原理：**
- DeFi protocol（如 Aave）提供无抵押贷款，条件是在同一笔 transaction 内还款
- 如果还不上 → 整笔 transaction revert（原子性保证）

**Attack 流程：**
```
Single Transaction:
  1. Flash loan 1M USDC from Aave
  2. Swap 1M USDC → Token_X on DEX_A (pushes price up)
  3. Use Token_X as collateral to borrow on Lending_Protocol
  4. Or: Trigger oracle price update based on manipulated DEX_A price
  5. Profit from price discrepancy
  6. Repay flash loan + fee

  All in one transaction, no capital required!
```

**检测方法：**
- Single-transaction anomaly：异常高的 gas used、涉及多个 DeFi protocol 的复杂调用链
- Oracle price deviation：在 flash loan 交易前后，oracle price 出现异常偏移
- Transaction trace analysis：分析 internal transactions 的调用链

#### Mixing / Money Laundering

**Peel Chain（逐层剥离）：**
```
Stolen funds (100 BTC)
  ├── Send 99 BTC to Address_B
  │   ├── Send 98 BTC to Address_C
  │   │   ├── Send 97 BTC to Address_D
  │   │   │   └── ... (continues)
  │   │   └── 1 BTC → exchange (convert to fiat)
  │   └── 1 BTC → exchange
  └── 1 BTC → exchange
```
- 每次 "剥" 一小笔去交易所变现，主体资金继续向前
- 检测：识别线性 chain 结构 + 渐减金额 pattern

**DeFi Layering：**
```
Stolen ETH → Swap to USDC on Uniswap → Deposit to Aave
→ Borrow DAI → Bridge to Polygon → Swap to WMATIC → ...
```
- 通过多次 DeFi 操作增加 tracing 难度
- 检测：跟踪 token swap + bridge + lending 的完整 path

---

## Section 5: DeFi 概念与 Fraud（DeFi Concepts & Fraud Vectors）

---

### 5.1 DeFi 核心协议类型

| 协议类型 | 代表项目 | 核心机制 | Fraud 风险 |
|---------|---------|---------|-----------|
| **DEX (Decentralized Exchange)** | Uniswap, SushiSwap | AMM (Automated Market Maker) | Wash trading, front-running, rug pull tokens |
| **Lending** | Aave, Compound | Over-collateralized lending | Flash loan attack, oracle manipulation |
| **Stablecoin** | Tether (USDT), USDC, DAI | Pegged to USD (各自机制不同) | De-peg event, reserve fraud |
| **Yield Farming** | Yearn, Convex | 自动化收益策略 | Smart contract exploit, economic attack |
| **Bridge** | Wormhole, Ronin | Cross-chain asset transfer | Bridge exploit ($600M+ losses) |
| **Derivatives** | dYdX, GMX | On-chain futures/options | Oracle manipulation, liquidation cascade |

### 5.2 AMM (Automated Market Maker) 原理

**核心概念：**
- DEX 不用 order book，而用 liquidity pool + 数学公式自动定价
- **Constant Product Formula：** $x \times y = k$
  - x = token A 的数量，y = token B 的数量，k = 常数
  - 用户 swap 时改变 x 和 y 的比例，价格自动调整

**Liquidity Provider (LP)：**
- 用户将两种 token 按比例存入 pool（提供流动性）
- 获得 LP token 作为凭证
- 赚取 swap 手续费
- 面临 **Impermanent Loss** 风险

**与 Fraud 的关系：**
- **Rug Pull：** 项目方抽走 LP → 用户无法 swap → token 归零
- **Sandwich Attack：** MEV bot 在用户 swap 前后夹单，利用 price slippage 获利
- **Liquidity Manipulation：** 通过大额 swap 操纵 pool 价格

### 5.3 Flash Loan 详解

**工作原理：**
```solidity
// Simplified flash loan
function flashLoan(uint amount) external {
    uint balanceBefore = token.balanceOf(address(this));
    token.transfer(msg.sender, amount);  // Lend

    // Borrower executes arbitrary logic here
    IFlashLoanReceiver(msg.sender).executeOperation(amount);

    // Check repayment
    require(token.balanceOf(address(this)) >= balanceBefore + fee);
}
```

**为什么 flash loan 可以被 exploit：**
- 无需资本 → 攻击者可以借到巨额资金
- 原子性 → 如果攻击失败，只损失 gas fee
- DeFi composability → 一笔交易内可以调用多个协议

**主要攻击向量：**
1. **Price oracle manipulation：** 通过大额 swap 操纵 DEX 价格 → lending protocol 的 oracle 读取到错误价格 → 以有利价格借贷
2. **Governance attack：** 借大量 governance token → 发起并通过恶意提案 → 归还 token
3. **Re-entrancy + flash loan：** 结合 reentrancy 漏洞放大攻击

### 5.4 Oracle 机制与操纵

**Oracle 是什么：**
- Smart contract 无法直接获取链外数据（如 token 价格）
- Oracle 将链外数据"喂"给链上 contract

**Oracle 类型：**
- **On-chain oracle (TWAP)：** 基于 DEX 的 Time-Weighted Average Price
- **Off-chain oracle (Chainlink)：** 多个 node 从链外获取数据，聚合后上链
- **Hybrid：** 结合 on-chain 和 off-chain

**Oracle 操纵 = 操纵 DeFi protocol 的价格输入，导致错误的抵押/清算/swap**

### 5.5 Bridges 与跨链风险

**Bridge 工作原理：**
```
Chain A: User locks 100 ETH in Bridge Contract_A
    ↓ (validator / relay network confirms)
Chain B: Bridge Contract_B mints 100 wETH to User
```

**主要风险：**
- **Smart contract vulnerability：** Bridge contract 的代码漏洞（Ronin: $625M, Wormhole: $320M, Nomad: $190M）
- **Validator compromise：** Bridge 的 validator 被攻击（Ronin: validator private keys stolen）
- **Money laundering vector：** Cross-chain transfer 增加 tracing 难度

---

## Section 6: Smart Contract 基础与恶意合约检测（Smart Contract Analysis）

---

### 6.1 Solidity 基础

**Solidity 是以太坊的主流 smart contract 语言。**

**最小 ERC-20 Token Contract 示例：**
```solidity
contract SimpleToken {
    mapping(address => uint256) public balances;
    mapping(address => mapping(address => uint256)) public allowance;
    uint256 public totalSupply;
    address public owner;

    function transfer(address to, uint256 amount) public returns (bool) {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        balances[to] += amount;
        return true;
    }

    function approve(address spender, uint256 amount) public returns (bool) {
        allowance[msg.sender][spender] = amount;
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) public returns (bool) {
        require(allowance[from][msg.sender] >= amount, "Insufficient allowance");
        require(balances[from] >= amount, "Insufficient balance");
        allowance[from][msg.sender] -= amount;
        balances[from] -= amount;
        balances[to] += amount;
        return true;
    }
}
```

**关键安全概念：**
- `msg.sender`：调用者地址
- `require()`：条件不满足时 revert
- `onlyOwner` modifier：限制只有 owner 可以调用某函数
- `approve / transferFrom`：允许第三方从你的账户转账——phishing attack 的关键

### 6.2 恶意合约常见模式

| 模式 | 原理 | 代码特征 |
|------|------|---------|
| **Honeypot** | 用户可以 buy 但无法 sell | `transfer()` 中有隐藏条件阻止某些地址卖出；或 `approve()` 有白名单机制 |
| **Hidden Mint** | Owner 可以无限铸造 token | 有 `mint()` 函数且没有 supply cap，或 `mint()` 隐藏在 fallback 函数中 |
| **Fee Manipulation** | Owner 可以将交易税设为 100% | 有 `setFee()` / `setTax()` 函数，且无上限检查 |
| **Proxy Upgrade** | 合约可以被升级为恶意版本 | 使用 delegatecall + proxy pattern，owner 可以指向新 implementation |
| **Blacklist** | Owner 可以冻结任何地址 | 有 `blacklist(address)` 函数，影响 `transfer()` |
| **Self-destruct** | 合约可以自毁，锁住所有资金 | 有 `selfdestruct()` 调用 |

### 6.3 Smart Contract 分析工具

| 工具 | 类型 | 说明 |
|------|------|------|
| **Slither** | Static analysis | 自动检测 common vulnerability patterns |
| **Mythril** | Symbolic execution | 深度漏洞检测，支持 formal verification |
| **Etherscan Verified Source** | Source code | 查看已验证的合约源代码 |
| **4byte.directory** | Function selector | 将 bytecode 的前 4 字节映射到函数签名 |
| **Tenderly** | Transaction trace | 可视化 transaction 的完整执行过程 |
| **DeFi Safety** | Audit score | DeFi 项目安全评分 |

### 6.4 Transaction Trace 分析

**为什么 trace 对 fraud detection 重要：**
- 一笔 external transaction 可能触发多个 internal transactions
- Fund flow 需要看 trace 才能完整追踪
- Flash loan attack 的完整攻击链只有在 trace 中可见

**获取 trace：**
```
// Geth/Erigon RPC
eth_call / debug_traceTransaction(txHash, {tracer: "callTracer"})
```

**Trace 结构：**
```json
{
  "type": "CALL",
  "from": "0xAttacker",
  "to": "0xFlashLoanProvider",
  "value": "0x0",
  "input": "0x...",
  "calls": [
    {
      "type": "CALL",
      "from": "0xFlashLoanProvider",
      "to": "0xAttacker",
      "value": "1000000000000000000000",  // 1000 ETH
      "calls": [
        // Attacker's exploit logic
        { "type": "CALL", "to": "0xDEX", ... },
        { "type": "CALL", "to": "0xLending", ... }
      ]
    }
  ]
}
```

---

## Section 7: Graph-Based Blockchain Analysis（图分析方法）

---

### 7.1 行业实践

**Chainalysis：**
- 全球最大的 blockchain analytics 公司
- 核心产品：KYT (Know Your Transaction) — 实时 transaction risk scoring
- 技术核心：massive address clustering + entity attribution + risk propagation
- 数据库：标注了数百万个地址（exchange, darknet, mixer, scam, etc.）

**Elliptic：**
- 公开了 Elliptic Dataset（Bitcoin transaction graph with fraud labels）
- 是学术研究的重要 benchmark
- 约 200K transactions，标注了 illicit / licit / unknown

**TRM Labs, Crystal, Scorechain：** 类似的 blockchain analytics 提供商

### 7.2 Address Graph 构建

**Bitcoin Transaction Graph：**
```
Address_A --[tx_hash, amount, timestamp]--> Address_B
```
- 有向图（from → to）
- Edge 上有 transaction metadata

**Ethereum Address Graph：**
```
Address_A --[tx_hash, value, gas, method_id]--> Address_B (EOA)
Address_A --[tx_hash, value, gas, method_id]--> Contract_C
Contract_C --[internal_tx]--> Address_D
```
- 需要包含 internal transactions（从 trace 提取）
- Contract interaction 是重要的边类型

**Graph Construction Pipeline（与 PayPal 对比）：**

| 步骤 | PayPal | Crypto |
|------|--------|--------|
| 1. Raw data | Internal DB transactions | Block + transaction + trace + token_transfer |
| 2. Node definition | Account | Address (可能需要 entity resolution 后作为 super-node) |
| 3. Edge definition | Shared asset (credit card, device, IP) | Direct transaction, token transfer, contract interaction |
| 4. Edge feature | Linking type, asset riskiness | Amount, gas, token type, method_id, timestamp |
| 5. Noise filtering | Remove common nodes (shared WiFi IP) | Remove exchange hot wallet, popular contract |
| 6. Graph DB | AWS Neptune (Gremlin) | Neo4j / Neptune |

### 7.3 图算法应用

| 算法 | 应用 | PayPal 经验 |
|------|------|-----------|
| **Community Detection (Louvain)** | 将相关地址聚类为 entity | 直接迁移（LPA → Louvain 的经验） |
| **PageRank** | 计算地址的"重要性"，识别 hub 地址 | 可迁移（未在 PayPal 直接用，但理论理解） |
| **Shortest Path** | Fund tracing (A→...→B 的最短资金路径) | 类似 Gremlin 多跳查询 |
| **Connected Components** | 基础聚类——识别孤立的子图 | 直接迁移 |
| **Cycle Detection** | 识别 wash trading（A→B→C→A） | 直接迁移 |
| **Centrality Analysis** | 识别关键节点（mixer, bridge, exchange） | 可迁移 |

### 7.4 Risk Propagation

**在 crypto address graph 上做 risk propagation 的方法：**

**方法 1 — Rule-based Propagation：**
```
risk(B) = max(direct_risk(B), α × risk(A) × edge_weight(A→B))

其中 α = decay factor (0 < α < 1)
edge_weight = f(amount, recency, tx_count)
```
- 简单、可解释
- 适合 1-2 hop 的直接传播

**方法 2 — GNN-based Propagation：**
- 用 GAT/GCN 学习 propagation function
- Attention weight 自动学习不同边的 propagation strength
- 多层 GNN = 多跳 propagation
- 与 PayPal 的 GAT 经验直接对接

**方法 3 — Risk-weighted PageRank：**
```
PR(v) = (1-d) × initial_risk(v) + d × Σ_{u→v} PR(u) / out_degree(u)
```
- 将已知 fraud address 的 initial_risk 设为高值
- 通过迭代将 risk 传播到网络中

**PayPal 经验的价值：**
- PayPal 的 GAT model 已经隐式实现了 risk propagation（GNN message passing = risk propagation）
- 2-3 层 GAT = 2-3 hop risk propagation
- Edge feature (linking type, riskiness) → crypto 的 edge feature (amount, token type, gas)
- 方法论完全一致，只是 domain context 不同

### 7.5 大规模 Graph 处理

| 技术 | 用途 | PayPal 经验 |
|------|------|-----------|
| **Spark GraphX / GraphFrames** | 分布式图计算（PageRank, connected components） | Spark LPA 使用经验 |
| **DGL (distributed)** | 分布式 GNN training | DGL 使用经验（非分布式） |
| **Neo4j** | Graph DB for compliance knowledge | Neo4j 使用经验 (Graph-RAG) |
| **AWS Neptune** | Managed graph DB for large-scale graph | Neptune + Gremlin 使用经验 |
| **LSH (Spark MLlib)** | 大规模 embedding similarity search | 直接迁移（BucketedRandomProjectionLSH） |

---

## Section 8: Transformer/GNN 研究前沿（Research Frontiers）

---

### 8.1 Elliptic Dataset

**数据集概览：**
- Bitcoin transaction graph, 约 200K transactions, ~234K edges
- 标签：illicit (2%), licit (21%), unknown (77%)
- 166 node features + 72 temporal features
- 49 time steps (biweekly)
- 公开可用，是 blockchain fraud detection 的标准 benchmark

**经典 baseline 结果：**

| 方法 | Illicit F1 | 说明 |
|------|-----------|------|
| Random Forest | 0.58 | Tabular features |
| GCN | 0.63 | 1-layer |
| GAT | 0.67 | 2-layer |
| EvolveGCN | 0.70 | Temporal |
| Graph Transformer | 0.72+ | SOTA |

### 8.2 Temporal GNN

**为什么需要 temporal：**
- Blockchain graph 是动态的——新 transaction 不断产生
- Fraud pattern 有时间特征（如 "first fast deposit, then gradual withdrawal"）
- Static GNN 忽略了时间维度

**关键模型：**

**EvolveGCN (2019, AAAI)：**
- 核心思想：用 RNN (GRU/LSTM) 更新 GCN 的权重矩阵
- 在每个 time step t，GCN 的权重 $W_t$ 由 RNN 基于前一步 $W_{t-1}$ 生成
- 两种变体：
  - **EvolveGCN-H：** 用 GRU 更新权重，$W_t = \text{GRU}(H_{t-1}, W_{t-1})$
  - **EvolveGCN-O：** 用 LSTM 更新权重，$W_t = \text{LSTM}(W_{t-1})$
- 在 Elliptic dataset 上效果好于 static GCN/GAT

**TGN (Temporal Graph Network, 2020)：**
- 核心思想：为每个节点维护一个 memory state，随新事件（边/交易）动态更新
- 组件：
  - **Memory Module：** 每个节点有一个 hidden state，记录该节点的"历史"
  - **Message Function：** 新边出现时，生成 message 更新相关节点的 memory
  - **Memory Updater：** GRU/LSTM 用 message 更新 memory
  - **Embedding Module：** 基于 memory + graph structure 生成 node embedding
- 优势：支持 event-level (transaction-level) 的实时更新，不需要固定 time step

**TGAT (Temporal Graph Attention Network)：**
- 在 GAT 的 attention 中加入 time encoding
- Time encoding: $\Phi(t) = \cos(\omega_1 t + \phi_1), \sin(\omega_2 t + \phi_2), ...$
- Attention 计算时考虑边的时间信息

### 8.3 Graph Transformer

**Graphormer (Microsoft, 2021 NeurIPS)：**
- 将 Transformer 应用到图上，三大创新编码图结构：
  1. **Centrality Encoding：** 用 node 的入度/出度作为 bias 加到 input embedding
  2. **Spatial Encoding：** 任意两节点的最短路径距离作为 attention bias
  3. **Edge Encoding：** 最短路径上所有边的 feature 加权平均作为 attention bias
- 在 OGB benchmarks 上刷新 SOTA

**GPS (General, Powerful, Scalable Graph Transformer, 2022)：**
- 结合 local message passing (MPNN) 和 global attention (Transformer)
- 公式：$h_i = \text{MPNN}(h_i, \{h_j : j \in N(i)\}) + \text{Transformer}(h_i, \{h_j : j \in V\})$
- 比纯 Graphormer 更 scalable（O(N²) 的 global attention 有优化版本）

### 8.4 Self-Supervised Learning for Graph

| 方法 | 原理 | 适用场景 |
|------|------|---------|
| **GraphCL (Graph Contrastive Learning)** | 对图做 augmentation（node drop, edge perturbation, subgraph）生成 view，对比学习 | 图级别表示学习 |
| **GCA (Graph Contrastive Learning with Adaptive Augmentation)** | 自适应选择 augmentation 策略 | 节点级别 |
| **DGI (Deep Graph Infomax)** | 最大化 node embedding 和 graph-level summary 的互信息 | 节点级别 |
| **GraphMAE** | Masked autoencoding on graph（mask node features, reconstruct） | 通用 pre-training |

**在 crypto 场景的应用：**
- 用 self-supervised learning 在大量未标注的 blockchain data 上 pre-train address embedding
- Fine-tune on labeled fraud data（Elliptic, Chainalysis labels）
- 类比 PayPal 的 AutoEncoder embedding（无监督 → 下游任务）

### 8.5 关键论文列表

| 论文 | 年份 | 关键贡献 |
|------|------|---------|
| Elliptic Dataset (Weber et al.) | 2019 | 首个 large-scale Bitcoin fraud detection dataset |
| EvolveGCN (Pareja et al.) | 2019 | Temporal GNN, RNN 更新 GCN 权重 |
| Anti-Money Laundering in Bitcoin (Lorenz et al.) | 2020 | 综述 blockchain AML 方法 |
| TGN (Rossi et al.) | 2020 | Event-level temporal graph network |
| Graphormer (Ying et al.) | 2021 | Graph Transformer with structural encoding |
| GPS (Rampasek et al.) | 2022 | MPNN + Transformer 的 scalable 融合 |
| Ethereum Phishing Detection with GNN (Wu et al.) | 2022 | GNN 用于以太坊钓鱼检测 |
| Bitcoin Entity Classification (Lin et al.) | 2022 | GCN + temporal features for entity classification |
| Blockchain Address Clustering Survey (Harrigan & Fretter) | 2016/updated | 地址聚类方法综述 |

---

## Section 9: Crypto 合规与监管（Compliance & Regulation）

---

### 9.1 核心监管框架

**FATF (Financial Action Task Force)：**
- 国际反洗钱/反恐融资标准制定机构
- 2019 年发布 "Virtual Asset" 指南，将 VASP (Virtual Asset Service Provider) 纳入 AML/CFT 监管
- 核心要求：VASP 必须进行 KYC、transaction monitoring、suspicious activity reporting

**OFAC (Office of Foreign Assets Control, US)：**
- 美国财政部下属，管理经济制裁
- **SDN List (Specially Designated Nationals)：** 制裁名单
- 2022 年首次制裁 smart contract（Tornado Cash）
- 影响：任何与 sanctioned address 交互的地址都可能面临 secondary sanctions

**MAS (Monetary Authority of Singapore)：**
- 新加坡金融管理局
- **Payment Services Act (PSA)：** 要求 crypto exchange（如 OKX Singapore）持有 license
- 对 Digital Payment Token (DPT) service provider 的要求：
  - KYC / CDD (Customer Due Diligence)
  - Ongoing transaction monitoring
  - STR filing (Suspicious Transaction Report)
  - Travel Rule compliance

### 9.2 Travel Rule

**什么是 Travel Rule：**
- FATF Recommendation 16：VASP 在转账时必须传递 originator 和 beneficiary 的信息
- 类似银行间的 SWIFT message——转账时需要告诉对方 "谁给谁转的钱"

**Crypto Travel Rule 的技术挑战：**
- 没有统一的 messaging 标准（传统金融有 SWIFT）
- 需要 VASP 之间建立通信通道
- 非托管钱包（non-custodial wallet）不受 Travel Rule 约束
- **实现方案：** TRISA, OpenVASP, Sygna Bridge 等协议

**与 PayPal 的对比：**
- PayPal 作为银行间转账的参与者，早已遵守 Travel Rule
- Crypto 的 Travel Rule 合规体系还在建设中，这是 OKX 的重点工作之一

### 9.3 KYC / KYT

**KYC (Know Your Customer)：**
- 用户注册时的身份验证
- Crypto exchange 的标准流程：ID verification + address proof + selfie
- 不同 tier 不同要求（如 withdrawal limit 与 KYC level 挂钩）

**KYT (Know Your Transaction)：**
- 对每笔交易进行实时风险评估
- Crypto-specific：不仅看用户行为，还要看 **on-chain history of the address**
- 技术实现：address risk scoring → transaction risk scoring → alert generation
- 常用第三方服务：Chainalysis KYT, Elliptic Lens

**KYC + KYT 配合：**
```
User deposit:
  1. KYC check (user identity) ✓
  2. KYT check (deposit address on-chain history):
     - Has the address interacted with sanctioned addresses?
     - Has the address received funds from known fraud sources?
     - What is the address risk score?
  3. If KYT flagged → enhanced investigation
```

### 9.4 SAR / STR Filing

**与 PayPal 的对比：**

| 维度 | PayPal (SAR) | OKX/Crypto (STR) |
|------|-------------|-----------------|
| **全称** | Suspicious Activity Report | Suspicious Transaction Report |
| **监管方** | FinCEN (US) | MAS (Singapore), local regulators |
| **Filing threshold** | $5,000 (mandatory for certain types) | Varies by jurisdiction |
| **Content** | 用户信息 + 可疑交易描述 + 风险分析 | 同，但需要包含 on-chain transaction details |
| **Timeline** | 30 days from detection | Typically 15 business days (MAS) |
| **自动化** | PayPal AML agent 可自动生成 SAR draft | 同样的方法论可迁移 |

**迁移价值：**
- PayPal 的 AML agent 已经可以自动生成 SAR report
- Graph-RAG knowledge base 只需要将 BSA/FinCEN 的法规替换为 MAS/FATF 的法规
- Report generation 的 LLM workflow 完全复用

### 9.5 Stablecoin 监管趋势

**主要 stablecoin 类型：**

| Stablecoin | 机制 | 监管关注点 |
|-----------|------|-----------|
| **USDT (Tether)** | 声称 1:1 USD 储备 | Reserve transparency, de-peg risk |
| **USDC (Circle)** | 1:1 USD 储备，受监管 | 相对合规，但仍有 freeze 权限 |
| **DAI (MakerDAO)** | Over-collateralized by crypto assets | Decentralized, but smart contract risk |
| **BUSD** | Binance 发行，Paxos 托管 | 已被 SEC 叫停 |

**监管趋势：**
- 各国正在推出 stablecoin 专项法规
- MiCA (EU) 要求 stablecoin 发行方持有 license
- Singapore 的 MAS 对 stablecoin 有明确的 reserve 和 redemption 要求
- 与 fraud detection 的关系：洗钱资金大量通过 stablecoin 转移（因为价值稳定），monitoring stablecoin flow 是 KYT 的重点

---

## Section 10: 从 Fintech 到 Crypto 的桥接（Bridging Fintech to Crypto）

---

### 10.1 技术能力映射表

| 能力维度 | PayPal 经验（具体项目） | Crypto 应用 | 迁移难度 |
|---------|----------------------|------------|---------|
| **Graph Construction** | Account linking graph (100M nodes, 200-300M edges) | Address graph construction | 低——方法论一致，数据源不同 |
| **Community Detection** | LPA → Louvain on Spark/Scala | Address clustering (Bitcoin + Ethereum) | 低——算法直接复用 |
| **GNN** | Edge-aware GAT (DGL) | Transaction graph GNN for fraud scoring | 低——模型架构直接复用 |
| **Embedding Similarity** | AutoEncoder + LSH on Spark | Address embedding similarity search | 低——完全一致 |
| **LLM Agent** | AML Investigation Mate (LangChain Deep Agent) | Crypto fraud investigation agent (LangGraph) | 低——架构直接复用 |
| **Graph-RAG** | Neo4j + compliance knowledge base | Crypto compliance knowledge base | 低——换法规即可 |
| **Text-to-SQL** | LangGraph + BigQuery + RAG | On-chain data query tool | 低——换数据源即可 |
| **PU Learning** | Buyer AUP violation (label expansion) | Crypto fraud label expansion | 低——方法论一致 |
| **Anomaly Detection** | LSTM AutoEncoder (behavior sequence) | Transaction sequence anomaly detection | 低——升级为 Transformer |
| **Feature Engineering** | IV/PSI/WOE + domain-driven design | Crypto feature design | 中——需要 domain knowledge |
| **Blockchain Data** | — | On-chain data ingestion, EVM, gas 分析 | **高——主要 gap** |
| **DeFi Knowledge** | — | DeFi protocol 理解, flash loan, AMM | **高——主要 gap** |
| **Smart Contract** | — | Solidity 分析, 恶意合约检测 | **高——主要 gap** |
| **Crypto Compliance** | BSA/FinCEN AML compliance | FATF/OFAC/MAS crypto compliance | 中——框架类似，条文不同 |

### 10.2 Fraud Pattern 对应

| PayPal Fraud Pattern | Detection Method | Crypto 对应 | 迁移策略 |
|---------------------|-----------------|------------|---------|
| Stolen credit card fraud | Transaction anomaly + device fingerprint | Stolen private key / wallet drain | Transaction anomaly 可迁移，fingerprint 换为 gas pattern |
| Account takeover | Login behavior anomaly | Wallet compromise | 行为异常检测可迁移 |
| Collusion (buyer-seller) | Behavior similarity + cycle detection | Wash trading | 完全对应 |
| Money mule | Graph-based detection (linking + clustering) | Mixing / layering | Graph 方法可迁移 |
| Unauthorized account creation | Velocity + pattern detection | Sybil attack / mass address creation | 方法论可迁移 |
| Merchant website fraud | HTML analysis + content analysis | Honeypot token / fake dApp | Unstructured data analysis 可迁移 |
| Structuring | Transaction pattern detection | Peel chain | Pattern detection 可迁移 |

### 10.3 90-Day Ramp-Up Plan

**Month 1: Foundation（区块链基础 + 数据）**

- **Week 1-2：On-chain data 深度实践**
  - 搭建 local Erigon node（或用 Alchemy free tier）
  - 用 web3.py 查询 transaction、trace、log
  - 在 Dune Analytics 上做 5+ 个探索性分析（如 top Tornado Cash depositors、DEX volume trend）
  - 学习 ABI decode，理解 function selector

- **Week 3-4：复现经典分析**
  - 在 Elliptic dataset 上跑 GCN/GAT baseline（用 DGL，复用 PayPal 经验）
  - 实现 Bitcoin common-input heuristic address clustering
  - 学习 Etherscan API，构建简单的 address profiling pipeline

**Month 2: Application（应用 PayPal 经验到 crypto）**

- **Week 5-6：Address Clustering Pipeline**
  - 将 PayPal 的多路召回架构迁移：heuristic + community detection + embedding similarity
  - 在 Ethereum data 上实现 address behavior clustering
  - 搭建 feature engineering pipeline（crypto-specific features）

- **Week 7-8：Fraud Detection Prototype**
  - 用 PayPal GAT 的代码框架，适配 crypto transaction graph
  - 实现 rug pull detection 的基础 pipeline
  - 学习 DeFi protocol 基础：手动做几次 Uniswap swap、Aave deposit

**Month 3: Integration（系统集成 + 合规）**

- **Week 9-10：Investigation Agent**
  - 将 AML agent 改造为 crypto investigation agent
  - 替换 data source（内部 DB → on-chain API）
  - 构建 crypto compliance knowledge base（OFAC/FATF/MAS）

- **Week 11-12：Production Readiness**
  - Data pipeline 优化（incremental processing, real-time streaming）
  - Model deployment 方案确定（embedding cache + lightweight scoring）
  - 学习 smart contract analysis 基础

### 10.4 面试中如何 Frame Gap

**原则：诚实 + 有计划 + 展示底层能力迁移**

**示例回答：**

> "I'll be transparent — I don't have production experience with blockchain data. But here's why I'm confident I can ramp up quickly:
>
> First, **the ML methodology transfers directly.** My graph construction, community detection, GNN, and agent architecture are exactly what's needed for crypto anti-fraud. The difference is the data source, not the approach.
>
> Second, **I've done this before.** When I started the AML project at PayPal, I had zero AML compliance knowledge. Within 3 months, I'd built a system that matched senior investigators' accuracy. I did this by reverse-engineering from data — analyzing historical cases to understand the domain. I'll do the same with blockchain data.
>
> Third, **I've already started learning.** [Reference your study note / specific analyses you've done]
>
> The blockchain-specific knowledge — UTXO model, DeFi protocols, smart contract analysis — these are domain knowledge that can be learned. What takes years to build — graph ML expertise, production agent systems, fraud detection intuition — I already have."

### 10.5 面试中的 "Crypto Bridge" 句式

面对每个技术问题，用以下结构回答：

1. **"在 PayPal，我做了 X..."**（展示直接经验）
2. **"在 crypto 场景，对应的是 Y..."**（展示已经理解了映射关系）
3. **"关键差异是 Z..."**（展示诚实和深度思考）
4. **"我的迁移方案是..."**（展示行动力和具体计划）

**示例：**
> "At PayPal, I built a graph-based fraud detection pipeline using GAT on account linking graphs. In crypto, the equivalent is running GNN on address transaction graphs. The key difference is that crypto graphs are public, larger, and more dynamic — addresses can be created for free, so the noise ratio is higher. My approach would be to start with the same GAT architecture but add temporal encoding for transaction timestamps and adapt the edge features to include gas patterns and token types."
