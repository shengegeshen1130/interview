# Blockchain 基础入门（面向传统软件工程师的区块链指南）

> **文档定位：** 纯概念基础入门，面向有传统软件工程背景但区块链零基础的工程师。本文档**不涉及 fraud detection**，专注讲解区块链核心概念与原理。
>
> **配套阅读：** 建立基础概念后，可阅读 `crypto_anti_fraud_study_note.md` 了解区块链在反欺诈领域的应用。

---

## Section 1: 什么是区块链（What is Blockchain）

---

### 1.1 核心定义

区块链（Blockchain）是一种**分布式、不可篡改的账本技术**（distributed immutable ledger）。它的核心思想是：

- 所有交易记录按时间顺序打包进一个个 `block`
- 每个 block 通过 cryptographic hash 链接到前一个 block，形成一条 `chain`
- 这条 chain 由网络中的多个节点（`node`）共同维护，没有中央控制方

**类比 git：** 你可以把区块链想象成一个只能 append、不能 rewrite history 的 git 仓库。每个 block 就像一个 commit，包含了一批变更（交易），并且引用了上一个 commit 的 hash。任何人试图篡改历史记录，都会导致后续所有 commit 的 hash 失效——整个网络会拒绝这条被篡改的链。

### 1.2 区块结构

一个典型的 block 包含 **header** 和 **body** 两部分：

```
┌─────────────────────────────────────────┐
│              Block Header               │
│  ┌─────────────────────────────────┐    │
│  │  Previous Block Hash            │────┼──→ 指向上一个 block
│  │  Timestamp                      │    │
│  │  Merkle Root (交易摘要)          │    │
│  │  Nonce / Validator Info         │    │
│  │  Block Number (Height)          │    │
│  └─────────────────────────────────┘    │
│              Block Body                 │
│  ┌─────────────────────────────────┐    │
│  │  Transaction 1                  │    │
│  │  Transaction 2                  │    │
│  │  Transaction 3                  │    │
│  │  ...                            │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
           │
           │  Previous Block Hash
           ▼
┌─────────────────────────────────────────┐
│           Previous Block                │
│           ...                           │
└─────────────────────────────────────────┘
```

**Merkle Root** 是所有交易的 hash 树的根节点，可以高效验证某笔交易是否包含在 block 中，而不需要下载整个 block 的所有交易数据。

### 1.3 交易如何工作

一笔区块链交易的完整生命周期：

```
创建交易 → 数字签名 → 广播到网络 → 进入 Mempool → 被矿工/验证者打包 → 写入区块 → 确认
```

1. **创建交易：** 用户通过钱包软件构造一笔交易（发送方、接收方、金额、手续费等）
2. **数字签名：** 使用发送方的 `private key` 对交易进行签名，证明发送方有权使用这笔资金
3. **广播到网络：** 签名后的交易通过 P2P 网络广播给附近的节点
4. **进入 Mempool：** 节点验证交易格式和签名有效后，将其放入 `mempool`（待处理交易池）
5. **打包：** 矿工 / 验证者从 mempool 中选取交易（通常优先选手续费高的），打包成新的 block
6. **写入区块：** 新 block 被添加到链上
7. **确认：** 后续每多一个 block 被添加，这笔交易就多了一个 `confirmation`。通常 6 个确认（Bitcoin）或 12-32 个确认（Ethereum）被认为是"最终确认"

### 1.4 分布式账本

区块链的"分布式"体现在：

- **全节点（Full Node）：** 存储完整的区块链数据，独立验证所有交易和区块
- **轻节点（Light Node / SPV）：** 只存储 block header，需要向全节点请求交易数据
- **没有单点故障：** 即使部分节点下线，网络仍然运行
- **数据一致性：** 通过共识机制（Section 9）保证所有诚实节点最终达成一致

### 1.5 Public Chain vs Private Chain vs Consortium Chain

| 维度 | Public Chain | Private Chain | Consortium Chain |
|------|-------------|---------------|-----------------|
| **准入** | 任何人可加入 | 单一组织控制准入 | 多个组织共同控制准入 |
| **代表** | Bitcoin, Ethereum | Hyperledger Fabric | R3 Corda, Quorum |
| **去中心化程度** | 高 | 低 | 中 |
| **交易速度** | 较慢（全球共识） | 快（少量节点） | 中等 |
| **透明度** | 完全公开 | 仅内部可见 | 成员间可见 |
| **典型用例** | 加密货币、DeFi | 企业内部系统 | 跨机构协作（供应链、金融） |

---

## Section 2: Bitcoin 基础（Bitcoin Basics）

---

### 2.1 概述

Bitcoin 是第一个成功运行的区块链系统：

- **2008 年**：中本聪（Satoshi Nakamoto）发布白皮书 *"Bitcoin: A Peer-to-Peer Electronic Cash System"*
- **2009 年 1 月**：创世区块（Genesis Block）被挖出
- **总量上限：** 2100 万枚（`21,000,000 BTC`），永不增发
- **出块时间：** 约 10 分钟
- **最小单位：** 1 Satoshi = 0.00000001 BTC（1 亿分之一）
- **共识机制：** Proof of Work（PoW）

Bitcoin 的核心创新在于解决了**双花问题（double-spending problem）**——在没有中心化机构的情况下，如何防止同一笔数字资金被花费两次。

### 2.2 UTXO 模型

Bitcoin 使用 **UTXO（Unspent Transaction Output）** 模型来跟踪资金，类比现金交易：

- 你的"余额"不是一个数字，而是你拥有的所有未花费的"零钱"的总和
- 每笔交易消耗一些 UTXO（inputs），并生成新的 UTXO（outputs）

```
Alice 有两个 UTXO: 0.5 BTC 和 0.3 BTC

Alice 要给 Bob 转 0.7 BTC:

  Inputs:                         Outputs:
  ┌──────────────────┐           ┌──────────────────┐
  │ UTXO: 0.5 BTC   │──────────→│ 0.7 BTC → Bob    │  (支付)
  │ (from Alice)     │           ├──────────────────┤
  ├──────────────────┤           │ 0.09 BTC → Alice │  (找零)
  │ UTXO: 0.3 BTC   │──────────→├──────────────────┤
  │ (from Alice)     │           │ 0.01 BTC         │  (矿工手续费,
  └──────────────────┘           └──────────────────┘   隐含,不出现在 output)
  Total: 0.8 BTC                 Total: 0.79 BTC + 0.01 fee = 0.8 BTC
```

**关键特性：**
- 每个 UTXO 只能被完整消耗，不能部分花费（类似你不能撕碎一张100元纸币）
- "找零"会创建新的 UTXO 返回给发送方
- Input 总额 - Output 总额 = 矿工手续费（隐含的 fee）

### 2.3 挖矿

Bitcoin 的挖矿过程本质上是一个 **hash puzzle**：

1. 矿工收集 mempool 中的交易，构造一个候选 block
2. 不断尝试不同的 `nonce` 值，对 block header 计算 `SHA-256` hash
3. 要求 hash 值小于目标值（即 hash 以若干个 0 开头）
4. 第一个找到合法 nonce 的矿工获得出块权和奖励

**难度调整（Difficulty Adjustment）：**
- 每 2016 个 block（约两周）调整一次难度
- 目标是保持平均出块时间在 ~10 分钟
- 全网算力增加 → 难度提高；算力减少 → 难度降低

**减半（Halving）：**
- 每 210,000 个 block（约四年），block reward 减半
- 2009: 50 BTC → 2012: 25 BTC → 2016: 12.5 BTC → 2020: 6.25 BTC → 2024: 3.125 BTC
- 预计到 2140 年左右所有 Bitcoin 将被挖完

### 2.4 Bitcoin 的局限性

| 局限 | 说明 |
|------|------|
| **交易速度慢** | ~7 TPS（每秒交易数），对比 Visa ~65,000 TPS |
| **不支持智能合约** | Script 语言功能有限，不是图灵完备的 |
| **能耗高** | PoW 需要大量算力 |
| **可编程性差** | 无法构建复杂的链上应用 |

这些局限直接推动了 Ethereum 的诞生——一个可编程的区块链平台。

---

## Section 3: 以太坊（Ethereum）

---

### 3.1 概述

Ethereum 由 Vitalik Buterin 于 2013 年提出、2015 年上线，核心理念是打造一个"世界计算机"（World Computer）：

- 不仅仅是数字货币，而是一个**可编程的区块链平台**
- 开发者可以在上面部署 `smart contract`（智能合约），构建去中心化应用（`DApp`）
- 原生代币为 `ETH`（Ether），用于支付交易手续费（`gas`）和参与生态

### 3.2 Account 模型 vs UTXO

Ethereum 使用 **Account 模型**，与 Bitcoin 的 UTXO 模型有本质区别：

| 维度 | UTXO (Bitcoin) | Account (Ethereum) |
|------|---------------|-------------------|
| **余额表示** | 多个未花费输出之和 | 单一账户余额数值 |
| **交易方式** | 消耗旧 UTXO，产生新 UTXO | 直接修改账户余额 |
| **类比** | 用现金付款（凑零钱 + 找零） | 银行转账（直接改余额） |
| **并行性** | UTXO 间天然并行 | 需要 nonce 保证顺序 |
| **隐私性** | 每次可用新地址（较好） | 地址重复使用（较差） |

**Ethereum 的两种账户类型：**

- **EOA（Externally Owned Account）：** 由 private key 控制的普通用户账户
  - 有 `balance`（ETH 余额）和 `nonce`（交易计数器，防止重放攻击）
  - 可以主动发起交易

- **Contract Account：** 由 smart contract 代码控制
  - 有 `balance`、`nonce`、`code`（合约字节码）、`storage`（持久化存储）
  - 不能主动发起交易，只能被 EOA 或其他合约调用后执行

### 3.3 EVM（Ethereum Virtual Machine）

`EVM` 是 Ethereum 的执行引擎，负责运行 smart contract：

- **类比 JVM：** 就像 Java 代码编译成 bytecode 在 JVM 上运行，Solidity 代码编译成 EVM bytecode 在 EVM 上运行
- **Stack-based VM：** 使用栈结构执行操作，每个操作对应一个 `opcode`
- **确定性：** 给定相同输入，所有节点执行结果完全一致
- **沙盒环境：** 合约无法访问网络、文件系统或其他外部资源
- **Gas 计量：** 每个 opcode 消耗一定 gas（Section 6），防止无限循环

```
Solidity 源代码 (.sol)
        │
        ▼  编译 (solc)
EVM Bytecode + ABI
        │
        ▼  部署到链上
Contract Account (有唯一地址)
        │
        ▼  调用
EVM 执行 bytecode，修改链上状态
```

### 3.4 以太坊的状态

Ethereum 维护一个**全局状态（global state）**，本质上是从地址到账户状态的映射：

```
Global State = { address → account_state }

account_state = {
    nonce:        交易计数 / 合约创建计数
    balance:      ETH 余额 (以 wei 为单位, 1 ETH = 10^18 wei)
    storageRoot:  合约存储的 Merkle Patricia Trie 根
    codeHash:     合约代码的 hash
}
```

- 状态存储在 **Merkle Patricia Trie** 数据结构中
- 每个 block header 包含一个 `stateRoot`，表示执行完该 block 内所有交易后的全局状态快照
- 这使得轻节点可以高效验证任意账户的状态

### 3.5 发展历程

| 时间 | 里程碑 | 说明 |
|------|--------|------|
| 2015.07 | Frontier | Ethereum 主网上线 |
| 2016.03 | Homestead | 第一个稳定版本 |
| 2016.06 | The DAO Hack | DAO 合约被攻击，导致 Ethereum 硬分叉（ETH / ETC） |
| 2017.10 | Byzantium | 引入 `REVERT` opcode，改善合约安全 |
| 2019.12 | Istanbul | 降低部分 opcode 的 gas 成本 |
| 2021.08 | London (EIP-1559) | 新的 gas 费用机制：`base fee` 燃烧 + `priority fee` |
| 2022.09 | **The Merge** | 从 PoW 切换到 PoS，能耗降低 ~99.95% |
| 2023.04 | Shanghai/Capella | 允许提取质押的 ETH |
| 2024.03 | Dencun (EIP-4844) | 引入 `blob` 交易，大幅降低 L2 数据成本 |

---

## Section 4: 密码学基础（Cryptographic Foundations）

---

### 4.1 哈希函数

哈希函数是区块链的基石，将任意长度的输入映射为固定长度的输出：

**类比指纹：** 就像每个人有唯一的指纹，每份数据有唯一的 hash。数据哪怕改动一个比特，hash 值也会完全不同。

**核心特性：**
- **确定性：** 相同输入 → 相同输出
- **抗碰撞性：** 极难找到两个不同输入产生相同 hash
- **雪崩效应：** 输入微小变化 → 输出剧烈变化
- **不可逆：** 无法从 hash 反推原始数据

**常见算法：**

| 算法 | 输出长度 | 用于 |
|------|---------|------|
| `SHA-256` | 256 bit | Bitcoin 的挖矿和区块链接 |
| `Keccak-256` | 256 bit | Ethereum 的地址生成和状态存储 |
| `RIPEMD-160` | 160 bit | Bitcoin 地址生成（与 SHA-256 组合） |

```
SHA-256("Hello") = 185f8db32271fe25f561a6fc938b2e264306ec304eda518007d1764826381969
SHA-256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
                   ↑ 仅一个字母大小写不同，hash 完全不同
```

### 4.2 公钥密码学

区块链使用**椭圆曲线密码学（Elliptic Curve Cryptography, ECC）**，具体使用 `secp256k1` 曲线：

```
Private Key (私钥)
    │  256-bit 随机数，必须保密
    │
    ▼  椭圆曲线乘法 (单向函数，不可逆)
Public Key (公钥)
    │  可以公开
    │
    ▼  Hash 处理 (Keccak-256 / SHA-256 + RIPEMD-160)
Address (地址)
    │  公开的收款地址
    ▼
```

**关键特性：**
- Private key → Public key：**单向**，数学上不可逆
- 知道 private key = 拥有资产的控制权
- Private key 丢失 = 资产永久丢失（没有"找回密码"）

### 4.3 数字签名

数字签名用于证明交易确实由 private key 持有者发起：

**签名过程：**
1. 对交易数据计算 hash
2. 使用 private key 对 hash 进行签名，生成 signature（包含 `r`, `s`, `v` 三个值）
3. 将交易数据 + signature 广播到网络

**验证过程：**
1. 从 signature + 交易 hash 中恢复出 public key
2. 将 public key 转换为 address
3. 验证 address 与交易声明的发送方是否一致

```
签名：  sign(private_key, hash(tx_data)) → signature (r, s, v)
验证：  recover(signature, hash(tx_data)) → public_key → address == sender?
```

### 4.4 钱包

钱包（Wallet）是管理 private key 和与区块链交互的工具：

**按连网方式分类：**

| 类型 | 说明 | 代表 | 安全性 |
|------|------|------|--------|
| **Hot Wallet** | 联网，使用方便 | MetaMask, Trust Wallet | 较低（易被攻击） |
| **Cold Wallet** | 离线，物理隔离 | Ledger, Trezor | 高 |

**按托管方式分类：**

| 类型 | 说明 | 代表 |
|------|------|------|
| **Custodial** | 第三方管理你的 private key | 交易所（Binance, OKX） |
| **Non-custodial** | 用户自己掌管 private key | MetaMask, Ledger |

**Seed Phrase（助记词）：**
- 12 或 24 个英文单词，由 BIP-39 标准生成
- 本质上是 private key 的人类可读编码
- 拥有 seed phrase = 拥有所有关联地址的控制权

**HD Wallet（Hierarchical Deterministic Wallet）：**
- 基于 BIP-32/BIP-44 标准
- 从一个 seed 派生出无限多个 private key / address
- 结构：`m / purpose' / coin_type' / account' / change / index`
- 例如：`m/44'/60'/0'/0/0` 是 Ethereum 的第一个地址

### 4.5 地址格式

| 链 | 格式 | 示例 | 说明 |
|------|------|------|------|
| Bitcoin | Base58Check | `1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa` | 以 1 或 3 开头 |
| Bitcoin (SegWit) | Bech32 | `bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4` | 以 bc1 开头 |
| Ethereum | Hex (0x 前缀) | `0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045` | 40 个十六进制字符 |
| Tron | Base58 | `T9yD14Nj9j7xAB4dbGeiX9h8unkKHxuWwb` | 以 T 开头 |

---

## Section 5: 智能合约（Smart Contracts）

---

### 5.1 定义

`Smart Contract`（智能合约）是部署在区块链上的程序，满足特定条件时自动执行预设逻辑。

**类比自动售货机：**
- 投入硬币（发送交易 + ETH）
- 选择商品（调用合约函数）
- 机器自动出货（合约自动执行）
- 没有人工介入，规则由代码决定
- 一旦部署，规则对所有人透明且无法随意更改

**核心特性：**
- **确定性执行：** 相同输入 → 相同输出，所有节点结果一致
- **不可篡改：** 部署后代码不可修改（除非使用 proxy 模式）
- **透明公开：** 任何人可以查看合约代码和状态
- **无需信任：** 不依赖任何中心化机构执行

### 5.2 Solidity 基础

`Solidity` 是 Ethereum 智能合约的主流编程语言（语法类似 JavaScript/C++）：

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract SimpleToken {
    // State Variables (存储在链上，永久保存)
    string public name;
    mapping(address => uint256) public balances;
    address public owner;

    // Event (日志，供链下监听)
    event Transfer(address indexed from, address indexed to, uint256 amount);

    // Modifier (函数修饰器，复用权限检查逻辑)
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;  // 继续执行被修饰的函数体
    }

    // Constructor (仅在部署时执行一次)
    constructor(string memory _name, uint256 _initialSupply) {
        name = _name;
        owner = msg.sender;
        balances[msg.sender] = _initialSupply;
    }

    // Function (可被外部调用)
    function transfer(address to, uint256 amount) external {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        balances[to] += amount;
        emit Transfer(msg.sender, to, amount);
    }

    // View Function (只读，不修改状态，不消耗 gas)
    function getBalance(address account) external view returns (uint256) {
        return balances[account];
    }
}
```

**关键概念：**
- `state variable`：存储在链上的持久化数据（修改需要 gas）
- `function`：合约的可调用方法（`external` / `public` / `internal` / `private`）
- `event`：合约发出的日志，不存储在 state 中，但链下可以监听
- `modifier`：函数的前置 / 后置检查，常用于权限控制
- `msg.sender`：调用者地址；`msg.value`：发送的 ETH 金额

### 5.3 合约生命周期

```
  编写 (.sol)          编译 (solc)         部署 (交易)          交互 (调用)
┌───────────┐     ┌───────────┐     ┌───────────────┐     ┌───────────┐
│ Solidity  │────→│ Bytecode  │────→│ 发送部署交易    │────→│ 用户/合约  │
│ 源代码     │     │ + ABI     │     │ to=0x0        │     │ 调用函数   │
└───────────┘     └───────────┘     │ data=bytecode │     └───────────┘
                                    └───────────────┘
                                          │
                                          ▼
                                    合约获得唯一地址
                                    代码永久存储在链上
```

- **编写：** 用 Solidity 编写合约源代码
- **编译：** `solc` 编译器生成 `bytecode`（给 EVM 执行）和 `ABI`（给外部调用者使用的接口描述）
- **部署：** 发送一笔特殊交易（`to` 字段为空），`data` 字段包含 bytecode
- **交互：** 通过合约地址发送交易来调用合约函数

### 5.4 合约交互

**ABI（Application Binary Interface）：**
- 合约的接口描述（JSON 格式），定义了可调用的函数、参数类型、返回值
- 类比 REST API 的 OpenAPI/Swagger 文档

**Function Selector：**
- 函数签名的 Keccak-256 hash 的前 4 个字节
- 例如：`transfer(address,uint256)` → `0xa9059cbb`
- EVM 通过 function selector 确定要执行哪个函数

**Calldata：**
- 交易的 `data` 字段 = function selector + 编码后的参数
- 例如调用 `transfer(0xBob..., 100)` 的 calldata：`0xa9059cbb` + `000...Bob地址` + `000...64`（100 的十六进制）

### 5.5 常见安全漏洞

| 漏洞 | 说明 | 经典案例 |
|------|------|---------|
| **Reentrancy** | 合约在更新状态前调用外部合约，攻击者通过回调重复提取资金 | 2016 The DAO Hack（360 万 ETH） |
| **Integer Overflow/Underflow** | 整数溢出导致余额异常（Solidity 0.8+ 默认检查） | BEC Token 事件 |
| **Access Control** | 关键函数缺乏权限检查，任何人可调用 | Parity 多签钱包漏洞 |
| **Front-running** | 攻击者观察 mempool，在目标交易前插入自己的交易获利 | DEX 交易三明治攻击 |
| **Oracle Manipulation** | 操纵链上价格预言机，获取错误价格后套利 | 多个 DeFi 协议被攻击 |
| **Delegatecall** | 被代理调用的合约修改了调用者的 storage | Parity 钱包被"销毁" |

---

## Section 6: Gas 机制（Gas Mechanism）

---

### 6.1 什么是 Gas

`Gas` 是 Ethereum 网络中衡量计算工作量的单位。每笔交易、每个操作都需要消耗 gas。

**类比云计算计费：** 就像 AWS 按 CPU 时间 + 内存 + 网络流量计费，Ethereum 按操作复杂度（gas 消耗量）计费。Gas 机制的核心目的是：

- **防止滥用：** 无限循环、垃圾交易需要付出经济代价
- **资源分配：** 通过市场化定价机制（gas price），在网络拥堵时优先处理愿意付更多费用的交易
- **激励验证者：** Gas 费作为收入激励验证者处理交易

### 6.2 组成部分

一笔交易的总费用计算：

```
交易费 (Transaction Fee) = Gas Units × Gas Price

其中：
- Gas Units：交易实际消耗的计算量（由操作类型决定）
- Gas Price：每单位 gas 的价格（以 Gwei 为单位，1 Gwei = 10^-9 ETH）
```

**Gas Limit：**
- 用户设置的该笔交易最大 gas 消耗上限
- 如果实际消耗 < gas limit，剩余 gas 会退还
- 如果执行中 gas 耗尽，交易失败（`out of gas`），但已消耗的 gas **不退还**

### 6.3 EIP-1559（London 升级）

2021 年 8 月的 London 升级引入了 `EIP-1559`，彻底改变了 gas 费用机制：

**旧模型（First-price Auction）：**
```
用户出价 Gas Price → 矿工选高价交易 → 费用全给矿工
问题：用户难以预估合理价格，常常多付
```

**新模型（EIP-1559）：**
```
交易费 = (Base Fee + Priority Fee) × Gas Units

- Base Fee：由协议根据网络拥堵程度自动调整
  - 上一个 block 使用量 > 50% → base fee 上升（最多 +12.5%）
  - 上一个 block 使用量 < 50% → base fee 下降（最多 -12.5%）
  - Base fee 被燃烧 (burn)，不给验证者

- Priority Fee (Tip)：用户自愿给验证者的小费
  - 激励验证者优先打包你的交易
  - 用户设置 Max Priority Fee Per Gas

- Max Fee Per Gas：用户愿意支付的最高价格
  - 实际支付 = Base Fee + Priority Fee
  - 退还 = Max Fee - 实际支付
```

**EIP-1559 的影响：**
- Gas 费更可预测（base fee 透明且可计算）
- ETH 部分通缩（base fee 被燃烧，减少 ETH 供给量）
- 用户体验改善（不需要自己猜合理价格）

### 6.4 常见操作 Gas 成本

| 操作 | 大约 Gas Units | 说明 |
|------|---------------|------|
| ETH 转账 | 21,000 | 最基础的操作 |
| ERC-20 `transfer` | ~65,000 | Token 转账 |
| ERC-20 `approve` | ~46,000 | 授权 |
| Uniswap V2 Swap | ~150,000 | DEX 代币兑换 |
| Uniswap V3 Swap | ~130,000 | 优化后的兑换 |
| NFT Mint (ERC-721) | ~150,000 | 铸造 NFT |
| 合约部署（简单） | ~200,000 - 500,000 | 取决于合约复杂度 |
| 合约部署（复杂） | ~1,000,000 - 5,000,000+ | 如 Uniswap 合约 |

**费用示例（假设 base fee = 30 Gwei, priority fee = 2 Gwei）：**
```
ETH 转账费 = 21,000 × 32 Gwei = 672,000 Gwei = 0.000672 ETH ≈ $2
Uniswap 兑换费 = 150,000 × 32 Gwei = 4,800,000 Gwei = 0.0048 ETH ≈ $14
```

### 6.5 Gas 与用户体验

高 gas 费是 Ethereum 主网的主要痛点：

- 网络拥堵时（如 NFT 热潮），gas fee 可能飙升到数十甚至上百美元
- 小额交易不经济（转 $10 可能要付 $5 手续费）
- 复杂 DeFi 操作（多步交互）的总费用更高

**解决方案 → Layer 2（Section 10）：**
- 将交易在 L2 上执行，大幅降低 gas 费用
- 例如 Arbitrum / Optimism 上的交易费用通常只有 L1 的 1/10 到 1/100

---

## Section 7: Token 标准（Token Standards）

---

### 7.1 Coin vs Token

| | Coin | Token |
|---|------|-------|
| **定义** | 区块链的原生资产 | 基于智能合约创建的资产 |
| **示例** | BTC (Bitcoin), ETH (Ethereum), SOL (Solana) | USDT, UNI, LINK, BAYC |
| **运行层** | 协议层（Layer 1） | 应用层（智能合约） |
| **创建方式** | 内置在区块链协议中 | 任何人可以通过部署合约创建 |
| **用途** | 支付 gas fee、质押 | 治理、访问权、代表资产等 |

### 7.2 ERC-20（Fungible Token）

`ERC-20` 是 Ethereum 上最广泛使用的 token 标准，定义了 fungible token（同质化代币）的标准接口：

**核心接口函数：**

```solidity
interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}
```

**`approve` + `transferFrom` 模式：**
- 用户不能直接让合约"拿走"自己的 token（合约无法替你调用 `transfer`）
- 需要两步操作：
  1. 用户调用 `approve(spenderContract, amount)` 授权合约最多可转走 `amount` 个 token
  2. 合约调用 `transferFrom(user, recipient, amount)` 实际转移 token

**代表性 ERC-20 Token：**
- `USDT` / `USDC`：稳定币（锚定美元）
- `UNI`：Uniswap 治理代币
- `LINK`：Chainlink 预言机 token
- `WETH`：Wrapped ETH（ETH 的 ERC-20 封装版）

### 7.3 ERC-721（Non-Fungible Token / NFT）

`ERC-721` 定义了 non-fungible token（非同质化代币）标准，每个 token 都是唯一的：

**核心概念：**
- 每个 token 有唯一的 `tokenId`
- `tokenURI(tokenId)` 返回 metadata（通常指向 IPFS 上的 JSON，包含名称、描述、图片链接）
- 不可分割：你不能拥有半个 NFT

**典型用例：**
- 数字艺术品（Bored Ape Yacht Club, CryptoPunks）
- 游戏道具
- 域名（ENS: Ethereum Name Service）
- 门票 / 会员凭证

### 7.4 ERC-1155（Multi-Token Standard）

`ERC-1155` 是一个多代币标准，一个合约可以同时管理 fungible 和 non-fungible token：

- **批量操作：** `safeBatchTransferFrom` 一次性转移多种 token，节省 gas
- **灵活性：** 同一合约中，tokenId=1 可以是 fungible（有 1000 个），tokenId=2 可以是 unique（只有 1 个）
- **典型用例：** 游戏资产（一个合约管理金币 + 装备 + 稀有道具）

### 7.5 Token 创建原理

在 Ethereum 上创建 token 非常简单——**任何人**都可以部署一个 ERC-20 合约：

- 部署合约 ≈ 发送一笔交易，gas 费几美元到几十美元
- Token 的"价值"完全取决于市场共识和流动性，不取决于技术门槛
- 这就是为什么存在大量"垃圾 token"和骗局（rug pull）

### 7.6 Token 分类

| 类别 | 说明 | 示例 |
|------|------|------|
| **Utility Token** | 提供平台内特定服务的访问权 | LINK（支付 oracle 服务费） |
| **Governance Token** | 持有者可参与协议治理投票 | UNI, AAVE, COMP |
| **Stablecoin** | 锚定法币或其他资产的稳定价值 token | USDT, USDC, DAI |
| **Wrapped Token** | 将一条链的资产封装为另一条链的 token | WBTC（Bitcoin 在 Ethereum 上的封装） |
| **LP Token** | 代表流动性提供者在 DEX 中的份额 | Uniswap LP tokens |
| **Meme Token** | 基于社区文化和 meme 的投机性 token | DOGE, SHIB, PEPE |
| **Security Token** | 代表现实世界资产（股权、房产）的合规 token | （受监管） |

---

## Section 8: DeFi 生态（DeFi Ecosystem）

---

### 8.1 什么是 DeFi

**DeFi（Decentralized Finance）** 是构建在区块链上的去中心化金融服务生态：

- 用 `smart contract` 替代传统金融中介（银行、交易所、保险公司）
- **Composability（可组合性）：** DeFi 协议像乐高积木一样可以相互组合——又称 **"Money Legos"**
  - 例如：在 Aave 借出 USDC → 存入 Curve 赚取利息 → 用 LP token 在 Convex 再次质押
- **Permissionless：** 任何人都可以使用，无需 KYC、无需银行账户
- **透明：** 所有交易和协议规则公开可查

**DeFi 总锁仓量（TVL - Total Value Locked）** 是衡量生态规模的关键指标，峰值曾超过 $180B（2021 年 11 月）。

### 8.2 DEX 与 AMM

**DEX（Decentralized Exchange）** 是去中心化的代币交易平台，不需要中心化机构托管资金。

**核心机制——AMM（Automated Market Maker）：**

传统交易所使用 order book（买卖挂单），DEX 主流模式是 AMM：

- 流动性提供者（`LP`）将代币对存入流动性池（Liquidity Pool）
- 交易者与池子进行交易，价格由数学公式自动决定
- 最经典的公式是 **Constant Product Formula**：

```
x × y = k

x = 池中 Token A 的数量
y = 池中 Token B 的数量
k = 常数（在无新增/减少流动性时不变）
```

**示例：**
```
初始池子: 100 ETH × 200,000 USDC = k (20,000,000)

Alice 想用 USDC 买 10 ETH:
  新的 ETH 数量 = 100 - 10 = 90
  新的 USDC 数量 = k / 90 = 222,222
  Alice 需支付 = 222,222 - 200,000 = 22,222 USDC
  实际价格 = 22,222 / 10 = 2,222.2 USDC/ETH

注意：比初始价格 (200,000/100 = 2,000) 高了！
这就是 Price Impact —— 交易量越大，滑点越高
```

**Impermanent Loss（无常损失）：**
- LP 提供流动性后，如果代币价格发生变化，LP 的资产价值可能低于直接持有
- 称为"无常"是因为如果价格回到初始水平，损失消失
- 价格波动越大，无常损失越大

**代表项目：** Uniswap (V2/V3)、SushiSwap、Curve（专注稳定币对）、PancakeSwap (BSC)

### 8.3 借贷协议

去中心化借贷协议允许用户无需中间人即可借贷加密资产：

**类比当铺：** 你把值 10 万的手表押在当铺，借出 7 万现金。如果到期不还或手表贬值太多，当铺有权处置你的手表。

**核心机制：**

1. **超额抵押（Over-collateralization）：**
   - 借款人需要存入价值高于借款金额的抵押品
   - 例如：抵押价值 $10,000 的 ETH，最多借出 $7,500 的 USDC（LTV = 75%）

2. **清算（Liquidation）：**
   - 当抵押品价值下降到清算线以下时，任何人都可以发起清算
   - 清算人偿还部分债务，获得打折的抵押品作为奖励
   - 这保证了协议的偿付能力

3. **利率模型：**
   - 借贷利率由供需关系动态决定
   - 利用率（Utilization Rate）= 借出总量 / 存入总量
   - 利用率越高 → 借款利率越高（激励存入更多资金）

**代表项目：** Aave, Compound, MakerDAO

### 8.4 稳定币

稳定币（Stablecoin）是价格锚定某个稳定资产（通常是美元）的加密货币：

| 类型 | 机制 | 代表 | 优点 | 风险 |
|------|------|------|------|------|
| **法币抵押型** | 每发行 1 token 对应银行中 $1 储备 | USDT, USDC | 简单直觉，稳定性好 | 中心化，需信任发行方 |
| **加密资产抵押型** | 用加密资产超额抵押生成 | DAI (MakerDAO) | 去中心化，透明 | 抵押品波动可能导致清算 |
| **算法型** | 通过算法调节供给维持锚定 | UST (已崩盘) | 无需抵押，资本效率高 | 可能脱锚（death spiral） |
| **混合型** | 结合多种机制 | FRAX | 平衡效率和稳定性 | 复杂度高 |

**USDT vs USDC：**
- `USDT`（Tether）：市值最大的稳定币，但储备透明度受质疑
- `USDC`（Circle）：合规性更好，储备更透明（定期审计）
- 两者都在多条链上发行（Ethereum, Tron, BSC 等）

### 8.5 Yield Farming

**流动性挖矿（Yield Farming / Liquidity Mining）** 是通过提供流动性或参与 DeFi 协议来赚取收益的策略：

**工作流程：**
1. 用户将资产存入 DeFi 协议（如提供 Uniswap 流动性）
2. 获得 LP token 作为凭证
3. 将 LP token 质押到 farming 合约中
4. 获得协议治理代币作为奖励

**APY（Annual Percentage Yield）：**
- 年化收益率，考虑了复利效应
- DeFi 中的 APY 可以非常高（几百甚至几千 %），但通常伴随着高风险
- 高 APY 的来源：代币增发（通胀）、交易手续费分成、协议补贴

**风险：**
- 无常损失
- 智能合约漏洞
- 奖励代币贬值（挖提卖导致价格下跌）
- Rug pull（项目方跑路）

### 8.6 Flash Loan

`Flash Loan`（闪电贷）是 DeFi 的独特创新——**无需抵押**即可借出巨额资金，但必须在**同一笔交易内**归还：

```
同一笔交易内：
  1. 从 Aave 借出 1,000,000 USDC    ← 无需抵押
  2. 在 DEX A 买入 ETH (价格较低)
  3. 在 DEX B 卖出 ETH (价格较高)
  4. 归还 1,000,000 USDC + 手续费   ← 必须在交易结束前归还
  5. 获利留给自己

如果步骤 4 无法完成 → 整笔交易 revert → 就像什么都没发生
```

**原子性保证：** 区块链交易的原子性确保了"要么全部执行，要么全部回滚"。如果借款人无法在同一笔交易中归还，整笔交易回滚，资金安全。

**合法用途：**
- 套利（跨 DEX 价格差异）
- 自清算（用闪电贷替换自己的抵押品）
- 债务置换（从一个借贷协议转移到另一个）

### 8.7 Oracle

`Oracle`（预言机）是连接区块链和真实世界数据的桥梁：

**为什么需要 Oracle：**
- Smart contract 只能访问链上数据，无法直接获取链下信息
- 但很多 DeFi 协议需要外部数据（如 ETH/USD 价格）来执行逻辑
- 例如：借贷协议需要知道抵押品的美元价格才能判断是否需要清算

**Chainlink：**
- 最大的去中心化 Oracle 网络
- 多个独立的 data feed 节点从不同数据源获取价格
- 聚合后发布到链上的 price feed 合约
- DeFi 协议读取 Chainlink price feed 获取价格

**TWAP（Time-Weighted Average Price）：**
- 基于 DEX 交易对的时间加权平均价格
- Uniswap V2/V3 内置了 TWAP oracle
- 优点：去中心化，不依赖第三方
- 缺点：可能被闪电贷操纵（低流动性池子）

### 8.8 跨链桥

`Cross-chain Bridge`（跨链桥）用于在不同区块链之间转移资产：

**Lock-and-Mint 模式：**

```
Chain A (源链)                        Chain B (目标链)
┌──────────────┐                    ┌──────────────┐
│ 用户锁定      │                    │ 铸造等量      │
│ 100 ETH      │ ──── 桥协议 ────→  │ 100 wETH     │
│ 到桥合约中    │     (验证+中继)     │ (wrapped版)   │
└──────────────┘                    └──────────────┘

返回时：
Chain B: 销毁 100 wETH  ──→  Chain A: 解锁 100 ETH
```

**风险：**
- 跨链桥是高价值攻击目标（合约中锁定了大量资产）
- 历史上多起重大安全事件：
  - Ronin Bridge: $624M（2022）
  - Wormhole: $320M（2022）
  - Nomad Bridge: $190M（2022）
- 核心问题：桥的安全性取决于验证机制（多签、中继链、零知识证明等）

---

## Section 9: 共识机制（Consensus Mechanisms）

---

### 9.1 为什么需要共识

在去中心化网络中，没有中央权威来决定"真相"。共识机制解决的核心问题是：

**拜占庭将军问题（Byzantine Fault Tolerance, BFT）：**
- 分布式系统中，部分节点可能故障或作恶
- 如何让诚实节点在存在恶意节点的情况下达成一致？
- 区块链的共识机制 = 在不信任的环境中建立信任

### 9.2 PoW（Proof of Work）

**工作量证明**——Bitcoin 采用的共识机制：

**工作原理：**
1. 矿工竞争解决一个计算密集的 hash puzzle
2. 找到合法 nonce 使 `hash(block_header) < target` 的矿工获得出块权
3. 其他节点验证 hash 是否有效（验证很快，计算很慢）
4. 获胜矿工获得 block reward + 交易手续费

**安全性——51% 攻击：**
- 如果某个实体控制超过 50% 的全网算力，理论上可以：
  - 进行双花攻击（double spending）
  - 审查/阻止特定交易
  - 但**无法**偷取他人资金或修改历史交易
- 对 Bitcoin 网络发起 51% 攻击的成本极高（需要天文数字的算力和电力）

**缺点：**
- 能耗巨大（Bitcoin 全网年耗电量曾超过一些国家）
- 矿工中心化趋势（大型矿场经济效益更好）
- 交易确认速度慢

### 9.3 PoS（Proof of Stake）

**权益证明**——Ethereum（The Merge 后）采用的共识机制：

**工作原理：**
1. 验证者（Validator）质押（stake）一定数量的代币（Ethereum 要求 32 ETH）
2. 协议随机选择验证者来提议（propose）新 block
3. 其他验证者对 block 进行投票（attest）
4. 获得足够投票的 block 被添加到链上

**惩罚机制（Slashing）：**
- 如果验证者作恶（如试图双签、提议冲突 block），其质押的 ETH 将被部分没收
- 这使得攻击的经济代价非常高

**优势对比 PoW：**
- 能耗降低 ~99.95%
- 不需要专用挖矿硬件
- 更好的去中心化潜力（降低参与门槛）

### 9.4 其他共识机制

| 机制 | 全称 | 说明 | 代表链 |
|------|------|------|--------|
| **DPoS** | Delegated PoS | 持币者投票选出少量代表验证交易 | EOS, Tron |
| **PoSA** | Proof of Staked Authority | PoS + PoA 混合，少量授权验证者 | BSC (BNB Chain) |
| **PoH** | Proof of History | 用 VDF 创建可验证的时间戳序列，提高吞吐量 | Solana |
| **PoA** | Proof of Authority | 已知身份的验证者，适合联盟链 | VeChain, 私有链 |
| **PBFT** | Practical BFT | 经典 BFT 算法，适合小规模节点集 | Hyperledger Fabric |

### 9.5 比较总结

| 维度 | PoW | PoS | DPoS |
|------|-----|-----|------|
| **安全性** | 非常高（攻击成本极高） | 高（经济惩罚） | 中（少量验证者） |
| **去中心化** | 中（矿池集中化） | 高 | 低（少量代表） |
| **能耗** | 极高 | 低 | 低 |
| **吞吐量** | 低 (~7 TPS for BTC) | 中 (~30 TPS for ETH) | 高 (~数千 TPS) |
| **最终性** | 概率性（6 confirmations） | 较快确定性 | 快速确定性 |

**Blockchain Trilemma（区块链不可能三角）：**

```
        Decentralization
             /\
            /  \
           /    \
          /  只能 \
         /  同时优化 \
        /   其中两个  \
       /________________\
Security          Scalability
```

- **Decentralization（去中心化）：** 大量节点参与
- **Security（安全性）：** 抗攻击能力强
- **Scalability（可扩展性）：** 高吞吐量、低延迟

大多数区块链在这三者之间做权衡。例如：
- Bitcoin / Ethereum L1：牺牲 Scalability，保证 Decentralization + Security
- BSC / Solana：牺牲 Decentralization，提高 Scalability + Security
- Layer 2 方案：尝试借助 L1 的安全性同时提高 Scalability

---

## Section 10: Layer 2 与扩容（Layer 2 & Scaling）

---

### 10.1 为什么需要扩容

Ethereum L1 的瓶颈：
- 约 15-30 TPS（每秒交易数）
- 网络拥堵时 gas 费飙升
- 无法满足大规模应用的需求

**扩容思路：**
- **Layer 1 扩容：** 增大区块大小、优化共识机制（有去中心化权衡）
- **Layer 2 扩容：** 将交易执行移到链下，只将关键数据/证明提交到 L1

### 10.2 Rollups

`Rollup` 是目前最主流的 L2 扩容方案，核心思想是：

- 在 L2 上执行交易（便宜、快速）
- 将交易数据/状态压缩后提交到 L1（继承 L1 安全性）

**Optimistic Rollup：**
```
┌─────────── L2 (Optimistic Rollup) ───────────┐
│  执行交易，计算新状态                           │
│  将交易数据 + 状态根 提交到 L1                  │
│  默认假设所有交易有效 ("optimistic")             │
└──────────────────────────────────────────────┘
         │
         │  提交到 L1
         ▼
┌─────────── L1 (Ethereum) ────────────────────┐
│  如果有人发现问题：                              │
│  → 在 7 天挑战期内提交 "Fraud Proof"            │
│  → L1 验证，如果确实有问题则回滚                 │
│  如果无人挑战：状态确认                          │
└──────────────────────────────────────────────┘

代表: Arbitrum, Optimism, Base
```

**ZK Rollup：**
```
┌─────────── L2 (ZK Rollup) ───────────────────┐
│  执行交易，计算新状态                           │
│  生成 Zero-Knowledge Proof (ZKP)              │
│  将 ZKP + 压缩数据 提交到 L1                   │
└──────────────────────────────────────────────┘
         │
         │  提交到 L1
         ▼
┌─────────── L1 (Ethereum) ────────────────────┐
│  验证 ZKP（数学上保证计算正确）                  │
│  → 验证通过即刻确认，无需等待期                  │
└──────────────────────────────────────────────┘

代表: zkSync Era, StarkNet, Polygon zkEVM, Scroll
```

**Optimistic vs ZK Rollup：**

| 维度 | Optimistic Rollup | ZK Rollup |
|------|-------------------|-----------|
| **验证方式** | Fraud Proof（有人挑战才验证） | Validity Proof（数学证明） |
| **提款延迟** | ~7 天（挑战期） | 分钟级（证明验证即确认） |
| **计算成本** | L2 计算成本低 | 生成 ZKP 计算成本高 |
| **EVM 兼容性** | 较好（接近完全兼容） | 较难（zkEVM 仍在发展） |
| **成熟度** | 更成熟（Arbitrum/Optimism 已广泛使用） | 快速发展中 |
| **适合场景** | 通用 DeFi / DApp | 支付、转账等高频低复杂度操作 |

### 10.3 Sidechains

`Sidechain`（侧链）是与主链平行运行的独立区块链，通过桥与主链通信：

- 有自己的共识机制和验证者集合
- 安全性**不继承主链**（独立的安全模型）
- 代表：Polygon PoS（从侧链逐步向 ZK Rollup 转型）

### 10.4 State Channels

`State Channel`（状态通道）适用于双方高频交互的场景：

**类比酒吧 Tab：**
- 开 tab（打开 channel）：双方在链上锁定资金
- 消费（链下交易）：双方在链下反复更新状态，不上链
- 结账（关闭 channel）：将最终状态提交到链上结算

- 代表：Bitcoin Lightning Network、Ethereum Raiden Network
- 适合场景：高频小额支付

### 10.5 比较总结

| 方案 | 安全性来源 | 吞吐量 | 最终性 | 适合场景 |
|------|-----------|--------|--------|---------|
| **Optimistic Rollup** | 继承 L1（Fraud Proof） | ~2,000-4,000 TPS | 7 天（挑战期） | 通用 DeFi, DApp |
| **ZK Rollup** | 继承 L1（Validity Proof） | ~2,000-10,000+ TPS | 分钟级 | 支付、转账、高频交易 |
| **Sidechain** | 独立共识 | 取决于自身设计 | 秒级 | 游戏、低价值交易 |
| **State Channel** | 链上仲裁 | 理论上无限 | 即时（链下） | 高频双方交互 |

---

## Section 11: 多链生态（Multi-Chain Ecosystem）

---

### 11.1 为什么有这么多链

- **Blockchain Trilemma：** 不同链在去中心化、安全性、扩展性之间做不同权衡
- **不同应用场景：** 游戏需要高吞吐、DeFi 需要安全性、企业需要隐私
- **竞争和创新：** 不同团队尝试不同的技术方案和生态策略
- **经济激励：** 新链发行自己的 token，吸引开发者和用户

### 11.2 主要公链对比

| 链 | 原生代币 | 共识 | TPS | 出块时间 | 特点 |
|----|---------|------|-----|---------|------|
| **Ethereum** | ETH | PoS | ~30 | ~12s | 最大生态，安全性最高 |
| **BSC (BNB Chain)** | BNB | PoSA | ~300 | ~3s | 低手续费，Binance 生态 |
| **Solana** | SOL | PoH + PoS | ~4,000 | ~0.4s | 极高吞吐量，但偶尔宕机 |
| **Polygon PoS** | MATIC/POL | PoS | ~65 | ~2s | Ethereum 侧链/L2 |
| **Arbitrum** | ARB | Optimistic Rollup | ~4,000 | ~0.26s | 最大的 Ethereum L2 |
| **Optimism** | OP | Optimistic Rollup | ~4,000 | ~2s | OP Stack 生态（Base 基于它） |
| **Avalanche** | AVAX | Avalanche Consensus | ~4,500 | ~2s | 子网（Subnet）架构 |
| **Base** | ETH | Optimistic Rollup | ~4,000 | ~2s | Coinbase 的 L2，基于 OP Stack |

### 11.3 EVM 兼容 vs 非 EVM

**EVM 兼容链：**
- 可以运行 Solidity 编写的智能合约
- 开发者可以无缝迁移 Ethereum 上的 DApp
- 代表：BSC, Polygon, Arbitrum, Optimism, Avalanche C-Chain, Base

**非 EVM 链：**
- 使用自己的虚拟机和编程语言
- 需要用不同的语言/工具重新开发

| 链 | 虚拟机 | 智能合约语言 |
|----|--------|-------------|
| Solana | SVM (Sealevel) | Rust |
| Aptos / Sui | Move VM | Move |
| Cosmos 系 | CosmWasm | Rust (CosmWasm) |
| Polkadot | Wasm | Rust (ink!) |
| TON | TVM | FunC / Tact |

### 11.4 TVL（Total Value Locked）

**TVL** 是衡量区块链/DeFi 生态规模的核心指标：

- 定义：锁定在协议智能合约中的加密资产总价值
- 数据来源：[DeFiLlama](https://defillama.com) 是最权威的跨链 TVL 追踪工具
- TVL 的局限：不能反映用户数、交易量、生态健康度的全貌

**TVL 排名趋势（概况）：**
- Ethereum 长期占据 TVL 第一（~60%+）
- L2（Arbitrum、Base、Optimism）的 TVL 快速增长
- BSC、Solana、Tron 也有显著的 TVL 份额

---

## Section 12: 常见术语速查（Key Terminology Glossary）

---

| 英文术语 | 中文 | 简要解释 |
|---------|------|---------|
| `ABI` | 应用二进制接口 | 智能合约的接口描述（JSON），定义可调用的函数和参数 |
| `Address` | 地址 | 区块链上的账户标识符（如 Ethereum 的 `0x...` 格式） |
| `Airdrop` | 空投 | 免费分发 token 给特定地址（通常作为营销或奖励） |
| `AMM` | 自动做市商 | 使用算法（如 `x*y=k`）自动定价的去中心化交易机制 |
| `APY` | 年化收益率 | 计入复利的年化回报率 |
| `Base Fee` | 基础费用 | EIP-1559 中由协议自动调整的 gas 费用（会被燃烧） |
| `Block` | 区块 | 包含一批交易记录的数据单元 |
| `Block Explorer` | 区块浏览器 | 查看链上数据的工具（如 Etherscan、BscScan） |
| `Bridge` | 跨链桥 | 在不同区块链之间转移资产的协议 |
| `Bytecode` | 字节码 | 智能合约编译后的 EVM 可执行代码 |
| `Calldata` | 调用数据 | 交易中传递给合约的输入数据 |
| `Cold Wallet` | 冷钱包 | 离线存储私钥的硬件设备 |
| `Composability` | 可组合性 | DeFi 协议可像积木一样相互组合的特性 |
| `Confirmation` | 确认 | 交易被打包后，后续新增的区块数量 |
| `Consensus` | 共识 | 网络节点就账本状态达成一致的机制 |
| `DAO` | 去中心化自治组织 | 由 smart contract 和 token 投票治理的组织 |
| `DApp` | 去中心化应用 | 运行在区块链上的应用程序 |
| `DeFi` | 去中心化金融 | 构建在区块链上的金融服务生态 |
| `DEX` | 去中心化交易所 | 无需中心化托管的代币交易平台 |
| `Double Spending` | 双花 | 同一笔数字资金被花费两次的攻击 |
| `EOA` | 外部拥有账户 | 由 private key 控制的 Ethereum 用户账户 |
| `ERC-20` | — | Ethereum 上同质化代币的标准接口 |
| `ERC-721` | — | Ethereum 上非同质化代币（NFT）的标准接口 |
| `EVM` | 以太坊虚拟机 | 执行智能合约字节码的运行环境 |
| `Flash Loan` | 闪电贷 | 无需抵押、必须在同一笔交易内归还的贷款 |
| `Fork` | 分叉 | 区块链协议的版本分裂（Hard Fork / Soft Fork） |
| `Gas` | — | Ethereum 中衡量计算工作量的单位 |
| `Gas Limit` | — | 交易中设定的最大 gas 消耗量 |
| `Genesis Block` | 创世区块 | 区块链的第一个区块 |
| `Governance` | 治理 | 通过 token 投票决定协议参数和升级方向 |
| `Halving` | 减半 | Bitcoin 出块奖励每 ~4 年减半的机制 |
| `Hash` | 哈希 | 将任意数据映射为固定长度摘要的函数 |
| `Hot Wallet` | 热钱包 | 联网的钱包软件，使用方便但安全性较低 |
| `Impermanent Loss` | 无常损失 | LP 因代币价格变动导致资产价值低于单纯持有的损失 |
| `Layer 1 (L1)` | — | 基础区块链（如 Ethereum、Bitcoin） |
| `Layer 2 (L2)` | — | 建立在 L1 之上的扩容方案（如 Arbitrum、Optimism） |
| `Liquidity Pool` | 流动性池 | AMM 中存放代币对的智能合约 |
| `Mempool` | 交易池 | 待打包的已广播但未确认的交易集合 |
| `Merkle Tree` | 默克尔树 | 用于高效验证数据完整性的 hash 树结构 |
| `Mint` | 铸造 | 创建新的 token（如铸造 NFT 或增发稳定币） |
| `NFT` | 非同质化代币 | 每个 token 唯一且不可互换的数字资产 |
| `Node` | 节点 | 运行区块链软件、维护账本的计算机 |
| `Nonce` | — | 交易计数器（防重放）/ PoW 中的随机数 |
| `Oracle` | 预言机 | 将链下数据（如价格）提供给链上智能合约的服务 |
| `Private Key` | 私钥 | 控制区块链地址资产的 256-bit 密钥 |
| `Public Key` | 公钥 | 由私钥派生的公开密钥，用于验证签名 |
| `Rollup` | — | 将交易执行移到 L2、数据/证明提交到 L1 的扩容方案 |
| `Rug Pull` | 跑路骗局 | 项目方卷走流动性资金跑路 |
| `Seed Phrase` | 助记词 | 12/24 个单词编码的私钥备份 |
| `Slashing` | 罚没 | PoS 中对作恶验证者的经济惩罚 |
| `Slippage` | 滑点 | 交易执行价格与预期价格的偏差 |
| `Smart Contract` | 智能合约 | 部署在区块链上的自动执行的程序 |
| `Stablecoin` | 稳定币 | 价格锚定法币或其他稳定资产的加密货币 |
| `Staking` | 质押 | 锁定代币参与 PoS 共识并获取奖励 |
| `Token` | 代币 | 基于智能合约创建的加密资产 |
| `TVL` | 总锁仓量 | 锁定在 DeFi 协议中的加密资产总价值 |
| `UTXO` | 未花费交易输出 | Bitcoin 的资金跟踪模型 |
| `Validator` | 验证者 | PoS 系统中负责验证交易和出块的节点 |
| `Wallet` | 钱包 | 管理私钥和与区块链交互的工具 |
| `Wei` | — | ETH 的最小单位（1 ETH = 10^18 Wei） |
| `Yield Farming` | 流动性挖矿 | 通过提供流动性参与 DeFi 协议赚取收益 |
| `Zero-Knowledge Proof` | 零知识证明 | 在不泄露信息的情况下证明某命题为真的密码学技术 |

---

> **文档版本：** v1.0 | **最后更新：** 2026-02-27
