# Q7: 设计广告 Bidding 系统 (Ad Bidding System Design)

> **类型**：ML System Design (套用 7-step framework，但需结合 mechanism design / auction theory)
> **常见 follow-up 公司**：Meta、Google、Amazon、Trade Desk、TikTok、字节
> **难度**：⭐⭐⭐⭐⭐⭐ (比 Q3 CTR 预估更难——多了 mechanism design + multi-agent dynamics + budget control 三层复杂度)
> **与 Q3 区别**：Q3 是**平台估 pCTR**（预测层），Q7 是**怎么用 pCTR 跑 auction + 帮 advertiser 出价 + 控预算**（决策层）。pCTR 是 Q7 的输入。

---

## 题目 (Prompt)

> "Design an ad bidding system for Meta / Google / TikTok. Cover (1) auction mechanism, (2) advertiser bid optimization (auto-bidding), (3) budget pacing, (4) marketplace health."

变体：
- "Design Meta's oCPM optimization system."
- "Design Google's Target CPA auto-bidding."
- "How would you build a budget pacing system that maximizes conversions?"

---

## 一、整个系统的 mental model

广告 bidding 系统是一个**双边 ML 系统**：

```
┌─── PLATFORM SIDE ─────────────────┐    ┌── ADVERTISER SIDE ──┐
│  pCTR / pCVR 模型 (Q3)            │    │  Auto-bidder         │
│       ↓                            │    │  (target CPA / ROAS) │
│  Auction (GSP / VCG / 1st)        │ ←→ │  Budget pacer        │
│       ↓                            │    │  Bid landscape model │
│  Reserve price + Quality          │    │                      │
│       ↓                            │    └──────────────────────┘
│  Pricing (charge advertiser)      │
└────────────────────────────────────┘
                    ↑
               Marketplace dynamics
              (bid landscape, churn,
               long-tail advertiser fairness)
```

**Senior 关键直觉**：
- 平台 side 优化 **revenue**，advertiser side 优化 **ROAS**，user side 关心 **relevance**
- 三方利益**并不天然对齐**——好的 bidding 系统是激励机制设计 (mechanism design) 而不只是 ML
- "Auto-bidding" 本质是**两层 ML 的串联**：advertiser 用 ML 给 platform 设 bid，platform 用 ML 算 pCTR

---

## 二、Step 1: Problem Clarification

### 必问 clarifying questions

1. **Bidding 类型范围**：要 cover 哪些？
   - **Manual bidding**：advertiser 直接出 CPC bid
   - **Auto-bidding (smart bidding)**：advertiser 给 target CPA / target ROAS，平台帮自动出价
   - **oCPM**：advertiser 出 CPM bid，平台优化把 ad 投给最容易转化的人
   - **Max conversions**：花完预算最大化转化数
   - **Manual + ML 混合**：Lowest cost / Cost cap

2. **Auction 机制**：GSP / VCG / first-price？(2018+ 行业向 first-price + bid shading 转)

3. **Single slot vs multi-slot**：search ad 通常 multi-slot (position auction)，feed ad 一次 1-3 个

4. **Advertiser objective**：clicks? installs? conversions? revenue? ROAS? brand awareness?

5. **Scale**: QPS、advertiser 数量、ad 数、daily auctions（10B+ daily auctions for Meta/Google）

6. **Latency budget**: 单次 auction p99 < 50ms（含 pCTR 推理）

### 简化假设
- Cover Meta-style 系统：feed ad、oCPM + auto-bidder
- 10M advertisers, 100M ads, 10B daily auctions
- Auction: hybrid (cost cap 用 GSP-like, max-conversions 用 truthful)
- Latency: end-to-end p99 < 100ms

> **Senior signal**：主动指出 "bidding 系统设计涉及三方 (platform/advertiser/user)，要在 (a) 收入、(b) advertiser ROI、(c) user satisfaction 三个目标间 balance；这是一个 multi-agent reinforcement learning + mechanism design 问题，不是单纯 ML"。

---

## 三、Step 2: Metrics

### Platform-side metrics
| 类型 | Metric |
|------|--------|
| Primary | Total revenue (per day / per quarter) |
| Per-auction | eCPM (effective cost per mille) |
| Auction efficiency | Allocation efficiency = realized total value / max possible value |
| Marketplace health | Bid landscape stability, advertiser churn rate, long-tail advertiser reach |

### Advertiser-side metrics
| Metric | 含义 |
|--------|------|
| **CPA** (Cost per Acquisition) | $/conversion |
| **ROAS** (Return on Ad Spend) | revenue / ad spend |
| **CPC / CPM** | 单 click / 千次曝光成本 |
| **Conversion volume** | 转化数 |
| **Budget utilization** | 实际花 / 预算 |
| **Pacing curve** | 一天内花费曲线是否 smooth |

### User-side metrics
- Ad relevance (survey / hide rate)
- Feed quality (organic engagement 不能 因 ad load 上升而降)
- Diverse ad exposure（防"羊毛党"广告刷屏）

### Guardrail
- Latency p99
- Auction stability (% auctions with > 1 eligible ad)
- Long-tail advertiser ROAS (防大客户垄断)
- Reserve price hit rate（不能太高也不能太低）

> **Senior 难点**：advertiser ROAS 优化 vs platform revenue 优化是**短期 zero-sum**（advertiser 多赚 = 平台少赚），但**长期 positive-sum**（高 ROAS → advertiser 留下来 → 平台长期收入↑）。设计目标必须含长期 advertiser retention。

---

## 四、Step 3: Auction Mechanism (核心！)

### 4.1 GSP (Generalized Second-Price) — 最经典

**规则**：按 rank score 排序，winner 付 next-place 的 minimum-to-beat 价。

**Rank score** (Google search ad style):
$$\text{rank}_i = \text{bid}_i \times \text{pCTR}_i \times \text{quality}_i$$

**Pricing** (per click):
$$\text{price}_i = \frac{\text{rank}_{i+1}}{\text{pCTR}_i \times \text{quality}_i} + \epsilon$$

**优点**：
- 简单、实现成本低
- 行业 default，advertiser 熟
- 在均衡下（symmetric bidder）效果接近 VCG

**缺点**：
- **Not truthful**：advertiser 有动机 shade bid（虚报 bid）
- 多 slot 时 incentive 复杂
- Advertiser 互相侦测对方 bid → bidding war

### 4.2 VCG (Vickrey-Clarke-Groves) — 教科书最优

**规则**：每个 winner 付他对 others 造成的 externality。

$$\text{payment}_i = \sum_{j \neq i} (\text{value of } j \text{ without } i) - \sum_{j \neq i} (\text{value of } j \text{ with } i)$$

**优点**：
- **Truthful** (dominant strategy = bid 真实 value)
- Allocation efficient (welfare-maximizing)

**缺点**：
- 计算复杂（要算 counterfactual allocation）
- 对 pCTR 估计噪声敏感（噪声 → 错误 allocation → revenue 下降）
- Advertiser 不熟、不直观
- 工业界 Facebook 早期用过，后来部分场景退回 GSP-like

### 4.3 First-Price Auction (FPA) — 2018 后主流

**规则**：winner 付自己的 bid 价（不打折）。

**为什么 industry 转 FPA**：
- Display ads 转 header bidding 后，FPA 成为 SSP 间默认
- Programmatic 透明化要求（GSP 在 SSP 间不透明）
- Google Ad Manager 2019 全面转 first-price

**Advertiser side 必须配 bid shader**:
$$\text{shaded\_bid} = \text{value} \times f(\text{bid landscape})$$
$f$ 是基于 win rate 模型学的（防出价过高）

### 4.4 Position Auction (multi-slot)

Search ad 一页 4 slot，每个 slot CTR 衰减:
- Slot 1: CTR multiplier 1.0
- Slot 2: CTR multiplier 0.7
- Slot 3: CTR multiplier 0.5

**Allocation**：按 rank score 降序填充。
**Pricing**：GSP 或 VCG 在 position 上的扩展（每个 slot 单独 pricing）。

### 4.5 大厂选型对比

| 平台 | 机制 | 备注 |
|------|------|------|
| Google Search | GSP-like with quality | 长期 GSP，2019 转 first-price for display |
| Meta | Total-Value 排序 + second-price-ish | 不是 textbook GSP；多目标 quality 因子 |
| Display Ad Exchange | First-price (2019+) | header bidding 推动 |
| TikTok | first-price + auto-bid + budget pacing 联合优化 | 算法 black-box |

---

## 五、Step 4: Bid Optimization (Advertiser-side ML, 即 Auto-bidding)

### 5.1 Bidding 类型分类

| Strategy | 输入 | 优化目标 |
|----------|------|---------|
| Manual CPC | bid | win 多 click |
| oCPM (Meta) | CPM bid + optimization event (e.g., conversion) | 找最易转化用户 |
| Target CPA (Google) | target $/conversion | conversions 最大且 CPA ≈ target |
| Target ROAS | target revenue/spend | revenue 最大且 ROAS ≥ target |
| Max Conversions | budget cap | 在预算内 max conversions |
| Lowest Cost | budget cap | 最低 CPA |

### 5.2 Auto-bidder 核心问题

给定 advertiser 目标（e.g., target CPA = $10），**对每个 auction 出多少 bid** 让总体满足约束？

#### Lagrangian 视角
最大化：
$$\sum_t \text{value}_t \cdot \mathbb{1}[\text{win}_t]$$
约束：
$$\sum_t \text{cost}_t \cdot \mathbb{1}[\text{win}_t] \leq B$$
$$\frac{\sum \text{cost}}{\sum \text{conv}} \leq \text{target CPA}$$

引入 Lagrangian：
$$\max \sum_t (\text{value}_t - \lambda \cdot \text{cost}_t) \cdot \mathbb{1}[\text{win}_t]$$

**Optimal bid**:
$$\text{bid}_t^* = \frac{\text{pConv}_t \cdot \text{target CPA}}{1 + \lambda}$$

$\lambda$ 的物理意义：**budget shadow price**，反映边际机会成本。

#### Online learning $\lambda$
- $\lambda$ 太小 → spend 超 budget → 加大 $\lambda$
- $\lambda$ 太大 → spend 不够 → 减小 $\lambda$
- PID controller / dual gradient descent: $\lambda_{t+1} = \lambda_t + \eta \cdot (\text{spend rate} - \text{target rate})$

### 5.3 Bid Landscape Model

预测"出 X 价的 win 概率"，用于 bid shader（first-price）和 ROI 估计：
$$P(\text{win} | \text{bid}, \text{auction context}) = \sigma(f(\text{bid}, \text{features}))$$

数据源：**自己历史 bid + win/lose label**（只对参与过的 auction 有 label）→ selection bias，要 IPW debias 或 minimum-bid 探索。

### 5.4 RL-based bidding (advanced)

- State: budget remaining, time remaining, recent win rate
- Action: bid value
- Reward: conversion delta - cost penalty
- 算法：DDPG, PPO, off-policy critic

工业界 (Alibaba 2018 paper *Real-Time Bidding by Reinforcement Learning*, Trade Desk) 用得多，但难训稳。

---

## 六、Step 5: Budget Pacing

### 6.1 为什么需要 pacing

不 pacing：早 8 点 budget 花光 → 错过下午 prime time 高质量流量 → ROAS 差。

### 6.2 经典算法

#### Throttling (probabilistic pacing)
按比例随机参与 auction：
$$P(\text{participate}) = \frac{\text{target rate}_t}{\text{actual rate}_t}$$
低耗：实现简单
缺点：丢的 auction 可能恰好是高价值的

#### Bid modulation
不丢 auction，但调低 bid:
$$\text{bid}_t = \text{base bid} \cdot p(t)$$
$p(t)$ 由 PID 或 Lagrangian 控制
优点：保留高价值 auction 机会

#### Dual / Lagrangian 方法
和 5.2 同框架，$\lambda$ 自动平衡 spend 和 value

### 6.3 Pacing curve 选择

| 曲线 | 适用 |
|------|------|
| Even (匀速) | branding，全天均匀曝光 |
| Front-loaded (ASAP) | 时效性强（限时折扣） |
| Performance-aware (动态) | conversion-focused，按预测的 hourly opportunity 分配 |

**Senior insight**：performance-aware pacing 用 hourly conversion forecast 作为 weight：
$$\text{budget}_h = B \cdot \frac{\hat{c}_h}{\sum_h \hat{c}_h}$$

### 6.4 Cross-day / Cross-campaign budget
- Daily budget vs campaign lifetime budget
- Multi-campaign sharing：advertiser 多 campaign 共享 budget pool
- Carryover：今日没花完明天能不能用？(Google Ads 允许 daily ±2x，但月度封顶)

---

## 七、Step 6: System Architecture (端到端)

### 整体 pipeline
```
User request (impression opportunity)
    ↓
[Targeting filter: 召回符合 advertiser targeting 的候选 ad ~10K → 1K]
    ↓
[Per-ad pCTR / pCVR 推理 (Q3 model) ~30ms]
    ↓
[Per-ad bid 计算]
    ├── Manual bidder: 直接读 advertiser 设的 bid
    └── Auto-bidder:
            ├── Lagrangian 给 base bid
            ├── Pacing controller 给 modulation factor
            ├── Bid shader (first-price) 给 shading
            └── 输出 final bid
    ↓
[Rank score 计算: rank_i = bid_i × pCTR_i × quality_i]
    ↓
[Reserve price filter (rank_i < reserve → drop)]
    ↓
[Auction (GSP / FPA): 选 winner + 算 price]
    ↓
[Quality / policy re-ranking]
    ↓
[Charge advertiser, log event]
    ↓
[Realtime spend tracker → pacing controller feedback loop]
```

### 数据流
- 实时 spend 必须秒级更新到 pacing controller (防 over-spend)
- Conversion 是 delayed signal (秒~天) → delayed feedback model
- pCTR / pCVR 模型 daily / hourly retrain

### Cold start
- 新 advertiser：无 historical data → 用 industry-level prior
- 新 ad creative：multi-armed bandit 探索
- 新 audience：transfer learning from similar audiences

---

## 八、Step 7: Evaluation

### Offline
- **Auction simulator**: replay historical auctions with new logic
  - 关键：要 simulate counterfactual win/lose（其他 advertiser bid 不变假设）
  - **IPS / counterfactual evaluator** 校正 selection bias
- Per-segment: large vs long-tail advertiser, 高/低 budget campaign
- Robustness: 扰动 pCTR 噪声，看 allocation 稳定性

### Online A/B
- **Two-sided dilemma**：randomize on user vs randomize on advertiser
  - User-level: advertiser bids are confounded across treatments
  - Advertiser-level: 给一半 advertiser 新 bidder → marketplace level effect 不可比
  - Solution: cluster by advertiser-vertical 或时间块 switchback
- Long-term holdout: 1% advertiser 永远在 control，看 retention / churn

### Counterfactual / off-policy
- IPS estimator: $\hat{V} = \frac{1}{N} \sum \frac{\pi_{\text{new}}(a|x)}{\pi_{\text{old}}(a|x)} r$
- Doubly robust: combine IPS + outcome model

---

## 九、Step 8: Monitoring & Iteration

### 关键 dashboard
| 维度 | 指标 |
|------|------|
| Platform revenue | Real-time, hourly, daily |
| Auction stats | Avg bid, win rate, eCPM distribution |
| Bid landscape | Per-vertical bid distribution stability |
| Advertiser side | CPA / ROAS distribution，target hit rate |
| Pacing | Spend curve vs target curve per campaign |
| Marketplace health | Top-K advertiser concentration (Gini)，long-tail churn |
| Latency | p50/p99/p999 per stage |
| Calibration | pCTR / pCVR 实测 vs 预测（影响所有 downstream） |

### Marketplace stability monitoring
- 任何 model / mechanism 改动可能引发 advertiser bid landscape shift
- 关键 alert：bid distribution 偏移 > X%（防 cascade failure）
- A/B 必须看 advertiser-side **长期 ROAS distribution**，不只是平台 revenue

---

## 十、Senior 必谈的深度话题

### 10.1 Truthfulness vs Revenue trade-off

| 机制 | Truthful? | 长期 advertiser 信任 | Platform 短期 revenue |
|------|-----------|----------------------|------------------------|
| VCG | ✅ | 高 | 中 |
| GSP | ❌ | 中 | 高 |
| First-price | ❌（要 shade） | 中（透明化加分） | 中-高 |

**Senior 观点**：truthful 机制 → advertiser 不需要 game system → 把精力放在 creative & product → 长期生态健康。

### 10.2 Multi-objective auction

实际系统中 rank score 不是 simple bid × pCTR：
$$\text{rank} = \text{bid} \times \text{pCTR}^{\alpha} \times \text{quality}^{\beta} \times \text{relevance}^{\gamma} - \text{user disutility}$$

各 exponent 是 product 决策 + A/B 调。$\alpha < 1$ 偏向 user 体验，$\alpha = 1$ 标准 GSP。

### 10.3 Reserve price 设计

- 太低：低质广告进入，user 体验差
- 太高：auction 没人 → revenue 0
- **Personalized reserve**：基于 user/context (Myerson optimal)
- A/B 测：调 reserve 看 short-term revenue + long-term advertiser retention

### 10.4 Information asymmetry & cheating

- Advertiser 可以 bid shading game（特别 first-price）
- Click fraud / install fraud（bot 假转化）→ fraud detection 模型
- "Bid stuffing"（advertiser 大量假 bid 探查 landscape）

### 10.5 Long-term effects

- 短期 revenue 优化 → 大客户垄断 → 长尾流失 → marketplace 生态退化
- **Fairness constraint**：reach floor for new advertiser
- **Auction simulation in counterfactual marketplace**：估计长期均衡

### 10.6 Auto-bidder vs platform 的 incentive 冲突

- Auto-bidder 优化 advertiser CPA → 把 advertiser 真实 willingness-to-pay 隐藏
- 平台 GSP 收入是 second-price，advertiser 用 auto-bid → effective bid 接近 truthful → 平台 revenue ↓
- 解法：转 first-price + 让 auto-bidder 直接出 second-price (truthful)，或 reserve price uplift

### 10.7 Multi-agent equilibrium

- 多个 advertiser 同时用 auto-bidder，会形成 bid landscape 均衡
- 单个 advertiser 改 strategy → 其他被迫 react → cascade
- 大改 mechanism (e.g., GSP → FPA) 需要 transition period 让 advertiser learn 新 equilibrium

---

## 十一、常见 Follow-up 问题

### Q: "Why did Google move from GSP to first-price?"
- 透明化（programmatic 要求）
- Header bidding 时代 SSP 间互相 first-price 竞价，平台内部 GSP 不一致
- First-price + bid shading 在实际 advertiser 自动化场景下 revenue 接近 GSP 甚至更高
- 简化 unified auction across platforms

### Q: "Target CPA 系统怎么决定每个 auction 的 bid？"
- Step 1: 基于 user/context 估 pConv
- Step 2: bid = pConv × target_CPA / (1 + λ)
- Step 3: λ 由 dual ascent 控制 spend rate
- Step 4: pacing 调 modulation factor
- Step 5: 提交到 auction

### Q: "如果 advertiser bid 太低，导致 budget 花不完？"
- λ 自动减小，bid 增大
- 也可能 target CPA 太低 → suggest 调整
- Cold start 期 explore：raise bid floor 学习

### Q: "Auction 中如何处理 ad fatigue？"
- 加 frequency feature 进 pCTR / pConv 模型
- Bid 自动衰减：见过该 ad N 次后 bid × decay
- Creative rotation：multi-armed bandit 选 creative

### Q: "Budget 还剩 5 分钟才花完，怎么处理？"
- Pacing controller 突增 bid 强投
- 或 ASAP 模式 (advertiser 选)
- 风险：尾部 hour 流量质量低，CPA 上升

### Q: "Bid shader 的 input feature？"
- Auction context: user, slot, time, vertical
- Recent win rate at given bid
- Bid landscape distribution
- Output: shading factor f ∈ (0, 1)

### Q: "如何防止 auto-bidder 把所有 budget 烧给同一个高 pConv 用户？"
- Frequency cap (per-user impression limit)
- Reach floor (要求 unique reach ≥ X)
- Diversity reward in objective

### Q: "Marketplace level 效果怎么 A/B？"
- Geo / vertical cluster A/B
- Switchback (时间块)
- Synthetic control (Q6 method)
- 长期 holdout cell

### Q: "如果 pCTR 模型偏差 2x 高估，会怎样？"
- bid × pCTR 会让 ad winning rate 上升 → eCPM 错估
- Calibration drift → advertiser 实际 CPA 偏离 target
- Auction efficiency 下降，VCG 尤其敏感
- 解决：强制 calibration check + 自动 fallback 到 last-known-good model

---

## 十二、Senior Trade-off 总结

| 决策 | Option A | Option B | 怎么选 |
|------|---------|---------|--------|
| Auction 机制 | GSP | VCG | GSP 简单常用，VCG 长期 truthful 但实现复杂 |
| Pricing | second-price | first-price | first-price 透明，但需 advertiser bid shader |
| Pacing | throttling | bid modulation | 后者保留高价值 auction，更优 |
| Auto-bid optimization | greedy heuristic | Lagrangian / RL | Lagrangian 工程稳定，RL 上限高但难训 |
| Reserve price | flat | personalized (Myerson) | personalized 提升 revenue，cold start 难 |
| Quality multiplier α | 0 (纯 revenue) | 1+ (重 user) | 0.5-1，A/B 调 |
| Long-tail protection | none | fairness floor | 长期生态考虑必加 |

---

## 十三、答题节奏（45 min）

| 时段 | 内容 |
|------|------|
| 0-5 min | Clarify (auction 类型、bid type、scale、目标) |
| 5-10 min | Three-sided framework (platform/advertiser/user) + metrics |
| 10-15 min | Auction mechanism (GSP vs VCG vs FPA) + 选型理由 |
| 15-25 min | Auto-bidder + Lagrangian + pacing |
| 25-30 min | System architecture + latency budget |
| 30-35 min | Evaluation (auction simulator + A/B 难点) + monitoring |
| 35-40 min | 选 1-2 个深度话题 (truthfulness, marketplace stability, 长期 effect) |
| 40-45 min | Trade-offs + Q&A |

---

## 一句话答案 (Elevator Pitch)

> "广告 bidding 系统是一个三方 (platform / advertiser / user) 的双边 ML 系统：平台用 ML 估 pCTR/pCVR + 跑 auction，advertiser 用 ML auto-bid 来优化 ROAS。架构上是 (1) 选 auction 机制 (GSP / VCG / FPA，2018 后业界向 FPA 转)，(2) advertiser auto-bidder 用 Lagrangian 把 target CPA / ROAS 转成 per-auction bid，(3) budget pacer 用 PID/dual gradient 控制 spend curve，(4) bid shader 在 FPA 下学 win-rate 模型避免 over-bid。关键 senior 点是：(a) auction mechanism 设计是 mechanism design 问题不只是 ML，truthfulness vs revenue trade-off；(b) 平台 short-term revenue 和 long-term advertiser retention 长期 zero-sum 但 long-term positive-sum；(c) marketplace dynamics 是 multi-agent 均衡，任何 model 改动需要 monitor bid landscape stability 防 cascade；(d) Counterfactual auction simulator + cluster-level A/B 是评估 mechanism 改动的标准方法。"

---

## 与 Q3 (CTR 预估) 的对比

| 维度 | Q3 (CTR Prediction) | Q7 (Bidding System) |
|------|---------------------|----------------------|
| Stack 层级 | Prediction layer | Decision + mechanism layer |
| Output | pCTR 概率 | Auction allocation + price |
| 主要技术 | DLRM, DIN, Calibration | Auction theory, Lagrangian, RL |
| 主要挑战 | Calibration, delayed conv, position bias | Truthfulness, pacing, marketplace dynamics |
| Stakeholder | 主要 platform | Platform + advertiser + user |
| ML 类型 | Supervised | Constrained optimization + multi-agent RL |

**面试组合建议**：如果 Q3 已经被问，Q7 的回答里 pCTR 部分可以一带而过，把火力放在 auction + auto-bidding。
