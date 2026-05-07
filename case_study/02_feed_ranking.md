# Q2: 设计 Feed Ranking 系统 (Facebook / LinkedIn / Twitter)

> **类型**：ML System Design (套用 7-step framework)
> **常见 follow-up 公司**：Meta、LinkedIn、Twitter/X、Pinterest
> **难度**：⭐⭐⭐⭐⭐ (比纯 recsys 更难——多元内容类型、严格 latency、整合广告、长期 wellbeing)
> **与 Q1 的区别**：feed 内容**异质**（post / photo / video / ad / 系统通知混合），多目标更强，creator-side 公平性是一等公民

---

## 题目 (Prompt)

> "Design Facebook News Feed ranking" 或 "Design LinkedIn home feed."

---

## Step 1: Problem Clarification

### 必问的关键 clarifying questions

1. **Surface**：home feed 还是 group feed / topic feed？(home feed 最难)
2. **Goal**：
   - Meta 的演进：CTR → time spent → "meaningful social interactions" (MSI)
   - LinkedIn：professional engagement、job-seeking signal、creator participation
   - 一定要追问"long-term goal"——是不是 retention / DAU / professional outcomes
3. **Content types**：post (text/photo/video) + ad + group recommendation + people you may know + 系统通知
4. **Stakeholders**：viewer、poster (creator)、advertiser、platform integrity
5. **Constraints**：integrity (misinfo)、minor protection、regulatory (DSA in EU)、latency p99 < 250ms

> **Senior signal**：明确 feed 是**多边市场** (multi-sided)，不能只优化 viewer。要谈 creator-side metric (post reach, comment received) 和 advertiser 的 ROI。

### 简化假设
- Surface: Facebook home feed (logged-in)
- Primary goal: Meaningful Social Interactions (MSI) + 长期 retention
- DAU 2B、avg feed length 50 posts、QPS 100K

---

## Step 2: Metrics

### Offline
- 多任务 multi-head model 的每个 head 的 AUC、log loss、calibration
- NDCG@K（K=10，feed 顶部最重要）
- 每个 task 单独的 AUC（click、like、comment、share、hide、report）

### Online (A/B test 中)
| 层级 | 指标 |
|------|------|
| Primary | **Meaningful Social Interactions (MSI)** = 加权 like+comment+share+message |
| Secondary | time spent, session count, daily sessions per DAU |
| Long-term | 28-day retention, DAU/MAU |
| Creator-side | % creators with ≥ 1 like in 7d, comment received, post reach Gini |
| Integrity | misinfo prevalence, hate speech rate, hide rate |
| Wellbeing | passive scrolling ratio, late-night usage |

> **关键 senior insight**：Meta 在 2018 改 algorithm 主推 MSI，是因为发现 time spent 上去了但 wellbeing 下来了——这是面试官想听的"product context"。

### Guardrail
- Latency p99 < 250ms
- Ad load 不能超 X%
- Diversity index、creator distribution
- Misinfo / borderline content rate

---

## Step 3: Data

### Labels（每种交互独立 label）
- Click, dwell time, like, comment (写了多少字 → coarse 信号), share, save, hide, "see less", report, follow, message
- Negative：scroll-past、hide、see-less、report
- 不同 label 的 **noise level** 和 **稀疏度** 完全不同
  - Click：高频高噪
  - Comment with text：低频高质，是 MSI 的核心
  - Hide / report：极稀疏但 strong negative

### Label 处理
- **Delayed label**：comment 可能 1 小时后才发，retention 是 28 天后才知道 → online learning 难做
- **Label noise**：misclick (移动端尤其严重) → bot detection、dwell-time threshold filter
- **Class imbalance**：comment rate << click rate << impression → loss reweighting

### Bias
- **Position bias**：feed 顶部天然 CTR 高 → position feature in training, mask at serving
- **Selection bias**：当前 ranker 决定曝光 → exploration slot
- **Conformity bias**：用户 like 是因为别人 like 了

---

## Step 4: Features

| 类别 | 例子 |
|------|------|
| Viewer | demographic、language、device、historical interaction sequence (last 200 posts as Transformer input)、social graph features |
| Author | followers count、post-history quality、past engagement rate、content type expertise |
| Content | text embedding (BERT-derived)、image/video embedding、length、language、topic、freshness |
| Viewer-Author | are they friends? closeness score、past interaction count、shared groups |
| Context | time of day、device、network、session position |
| Cross-network | did viewer's friends like this? early-engagement signal |

**Senior 注意**：**Edge features** in social graph (viewer-author 关系) 是 Feed 比纯 recsys 多出来的关键维度，不能漏。

---

## Step 5: Model

### 5.1 召回 (Inventory Candidate Generation)
**和纯 recsys 不同**：feed 召回的 candidate pool 主要来自**社交图谱**：
- 朋友 + 关注的人 + 加入的 group 的 recent posts (e.g., last 7 days)
- 通常已经天然 narrow 到几百-几千 候选 → 不需要 ANN，直接走 ranking
- 加入"可能感兴趣但未关注"的探索内容（5-10%）

### 5.2 排序 — Multi-task Learning（核心）

**Why MTL?**
- 多个 head 共享 representation → 数据效率
- 不同 label 互相 regularize
- 出 **value model** 时直接用每个 head 的预测

**架构演进**：
1. **Shared bottom**：底层 shared，top 多个 task tower
2. **MMoE** (Multi-gate Mixture of Experts)：每个 task 有自己的 gate 选 expert combination
3. **PLE** (Progressive Layered Extraction)：解决 MMoE 中 task interference
4. **Sequence transformer**：处理 long user history

```
[features] → [shared bottom DNN] → [expert pool 1..K]
                                        ↓ (MMoE gates)
              ┌───────────┬───────────┬───────────┐
        Tower-Click  Tower-Like  Tower-Comment  Tower-Share  Tower-Hide
              ↓           ↓           ↓             ↓           ↓
           P(click)   P(like)    P(comment)     P(share)    P(hide)
```

### 5.3 Value Model（融合公式）
$$V = \sum_i w_i \cdot \hat{P}(\text{task}_i) - \sum_j w_j \cdot \hat{P}(\text{negative}_j) + \alpha \cdot \text{long-term term}$$

例：
$$V = 1 \cdot P(\text{click}) + 5 \cdot P(\text{like}) + 30 \cdot P(\text{comment}) + 50 \cdot P(\text{share}) - 100 \cdot P(\text{hide}) - 1000 \cdot P(\text{report})$$

权重 $w_i$ 是 **product decision + A/B tuning**，反映商业价值。

> **Senior 问题**：怎么设权重？答案：(1) 跟 product 一起定 prior（"comment 比 like 价值高 6×"），(2) 用 multi-objective optimization 找 Pareto frontier，(3) A/B test 在几个权重组合上对比 long-term metric。

### 5.4 长期价值 (Long-term value)
- **Reinforcement Learning**: predict not just immediate click but future state (off-policy RL hard at scale)
- **Cumulative value model**: 训练 model 直接预测"接下来 7 天该用户的 MSI"
- **Two-stage value**: short-term head + long-term head 加权

### 5.5 集成广告（Hybrid feed ranking）
- Ad 和 organic 在同一个 ranker 里出价：
  $$\text{rank score} = \max(V_{\text{organic}}, \text{eCPM} \cdot \alpha)$$
- 关键 trade-off：ad load 高 → 短期 revenue ↑，长期 retention ↓
- A/B test ad load 是经典 senior 题

---

## Step 6: Serving

```
User opens feed
    ↓
[拉 inventory: 朋友 / follow / group 的 last-N posts (~5ms)]
    ↓ ~2000 candidates
[Light ranking — small DNN per candidate (~30ms)]
    ↓ top 500
[Heavy ranking — MMoE / PLE (~100ms)]
    ↓ top 100
[Mix in ads (auction with eCPM) (~10ms)]
    ↓
[Re-ranking — diversity + integrity filter (~20ms)]
    ↓
返回 top 20
```

### Cold start
- New user: onboarding (follow suggestions)、popular-in-region、demographic-based
- New post: 用 author 历史 + content embedding 给 prior
- New author: ramp-up exploration

### Real-time signal
- 用户刚 like 了一个 post → 立刻影响接下来的 feed (online feature update)
- 朋友刚发了帖子 → freshness boost

---

## Step 7: Evaluation

### Offline
- Per-task offline metric (AUC、log loss、calibration)
- Per-segment：new vs existing user、active vs casual、各 country
- Calibration plot：每个 head 预测的 P 和实际 rate 是否匹配（影响 value model 融合）

### Online A/B test 设计
- **Primary metric**: MSI per DAU (proportion test or t-test on cohort-level mean)
- **Secondary**: 上面提到的所有
- **Long-term metric**: 28-day retention, monthly active sessions
- **Holdout group**: 1% 永远在 control，用来测 long-term cumulative effect
- **Power analysis**：MSI 的 variance 大 → 需要 ~10M+ DAU/group, 至少 14 天
- **Network effect**: A/B 在 user level 随机，但 viewer 影响 creator → ego-cluster randomization 或 cluster A/B

### Counterfactual evaluation
- 在 ramp 之前用 IPS / Doubly Robust 估计新模型 treatment effect
- 决定要不要进 A/B 队列

---

## Step 8: Monitoring & Iteration

| 维度 | 监控 |
|------|------|
| Feature drift | PSI per feature, alert > 0.2 |
| Calibration drift | per task head 预测 vs 实际 rate |
| Top-K composition | top 20 中 ad / organic / friend / publisher 比例 |
| Creator distribution | Gini coefficient of post reach |
| Integrity | misinfo prevalence, hate speech rate |
| Latency | per-stage breakdown |
| Per-segment performance | new user vs power user, mobile vs desktop |

**Retrain**：
- Heavy ranker: daily incremental + weekly full
- Light ranker: daily
- User embedding: hourly batch (or streaming for very active users)

---

## 常见 Follow-up 问题

1. **"用户抱怨 feed 越来越无聊（filter bubble），怎么办？"**
   - Diversity 加 hard constraint：每个 feed 至少 N 个不同 topic / creator
   - Exploration slot：5% 给 collaborative filtering 之外的内容
   - DPP / MMR re-ranker
   - 长期：multi-armed bandit 测 user 的 diversity 偏好（每个用户 personalize diversity）

2. **"如何衡量 wellbeing trade-off？"**
   - 主观：survey ("did this feed bring you joy/value?")
   - 客观：sleep-time usage、对 social connection 的 perceived effect (longitudinal study)
   - 引入 wellbeing head 加 negative weight

3. **"Hide / report 是 strong signal，但太稀疏怎么办？"**
   - Multi-task 用 shared representation 增强稀疏 head
   - Loss reweighting (focal loss)
   - Active learning：找最不确定样本去 collect explicit label

4. **"广告插入位置怎么决定？"**
   - 一种是固定位置 (e.g., position 2, 8, 16)，一种是 dynamic auction（更复杂）
   - 关键 trade-off：固定位置 inventory 利用率低，dynamic 容易 ad-bias-feed

5. **"Comment 中文字很长 vs 一个 emoji，怎么区别质量？"**
   - 加 comment-length feature 作 quality proxy
   - 训练 comment-quality classifier，把 high-quality comment 作为 stronger label
   - Down-weight bot-like / spam comments

6. **"Friend / family 内容 vs creator 内容怎么平衡？"**
   - 这是 product priority 决定的（Meta 2018 push "friends and family first" 就是 product decision）
   - 模型可以训练，但权重靠 product
   - 不同 segment 不同：年轻 user 偏 creator，年长 user 偏 family

---

## Senior Trade-off 总结

| 决策 | 取舍 | 怎么 frame |
|------|------|-----------|
| 优化短期 click vs 长期 MSI | 长期更难 measure | 用 holdout cell + 长期 cohort 看 retention |
| Heavy model accuracy vs latency | quality vs UX | distillation、quantization、early exit |
| Personalization vs filter bubble | engagement vs wellbeing | exploration slot、diversity rule、user-level personalize diversity |
| Creator fairness vs viewer engagement | Gini constraint vs CTR | reach floor for new creator、fairness regularization |
| Ad load vs retention | revenue vs LT | A/B 测 ad load curve，找 retention-revenue Pareto |

---

## 一句话答案 (Elevator Pitch)

> "Feed ranking 与纯推荐的本质区别是**多边市场**——viewer / creator / advertiser 三方利益冲突，且**多元内容类型**（post / video / ad / 系统通知）混排。架构上是 inventory（社交图谱拉取）→ multi-task ranker (MMoE/PLE 同时预测 click/like/comment/share/hide) → value model 加权融合 → diversity & ad 混排。关键 senior 点：(1) primary metric 要从 time spent 进化到 MSI 防止 reward clickbait；(2) value model 权重是 product decision + A/B tuning，不是 model 学的；(3) 必须有 creator-side 公平性 guardrail (Gini)、wellbeing guardrail (late-night usage)、integrity guardrail (misinfo)；(4) long-term effect 靠 1% holdout cell 测 28-day retention。"
