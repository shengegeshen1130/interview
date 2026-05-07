# Q1: 设计 YouTube / Netflix 推荐系统 (Recommendation System Design)

> **类型**：ML System Design (套用 7-step framework)
> **常见 follow-up 公司**：Meta、Google、Netflix、TikTok、Airbnb
> **难度**：⭐⭐⭐⭐ (推荐系统是 ML SD 的"必考题"，要答到能区分 senior vs junior 的深度)

---

## 题目 (Prompt)

> "Design YouTube's homepage recommendation system."

变体：Netflix 首页推荐 / TikTok For You / Spotify Discover Weekly / Amazon "you might also like"。

---

## Step 1: Problem Clarification

**必问 clarifying questions（前 5 分钟）**：

1. **Surface 是哪里**？(homepage feed? watch-next sidebar? notification?) —— 不同 surface 的 candidate pool、latency budget、目标都不同
2. **Business goal**：DAU? watch time? long-term retention? creator-side metrics?
   - YouTube 历史上从 CTR 切到 watch time 切到 "valued watch time"（含满意度调研）
3. **Scale**：DAU 2B、video pool 10B+、QPS 100K+、latency < 200ms
4. **User type**：logged-in vs anonymous? cold-start 比例？
5. **Constraints**：misinformation policy? minor-safety? creator fairness?

> **Senior 加分**：主动谈 "我们在为谁优化？"——viewer、creator、advertiser 是三方市场，纯 viewer 优化会伤 creator 生态。

**简化假设**（接下来按这个推进）：
- Surface: YouTube 首页 logged-in user
- Goal: long-term watch satisfaction（用 watch time + survey 拟合）
- Scale: 2B DAU, 10B videos, latency p99 < 300ms

---

## Step 2: Metrics

### Offline metrics
| Metric | 用途 | 局限 |
|--------|------|------|
| AUC | ranking 整体能力 | 不反应 top-K，不 calibrate |
| **NDCG@K** (K=10) | top-K ranking quality | 假设有 graded relevance label |
| Recall@K | candidate generation 的覆盖能力 | 不考虑顺序 |
| log loss / calibration | 预测概率绝对值 | 对召回阶段不重要 |

### Online metrics（A/B 中观察）
- **Primary**: long-term watch time (e.g., 14-day rolling)
- **Secondary**: CTR、completion rate、likes、shares、subscribes
- **Long-term**: 28-day retention, DAU/MAU
- **Survey-based**: "did you find this video valuable" (sampled survey)

### Guardrail metrics（不能恶化）
- Latency p99
- **Diversity**: unique creators / topics per session（防 filter bubble）
- **Creator-side**: Gini coefficient of creator views（防头部垄断）
- **Integrity**: misinformation / borderline content rate
- **Wellbeing**: late-night watch time（防成瘾）

> ⚠️ **Senior 关键 trade-off**：watch time 优化容易 reward clickbait → 引入 survey-based "valued watch time" 做 reward shaping，这是 YouTube 论文里的经典做法。

---

## Step 3: Data

### Labels（implicit feedback 为主）
- **Click**: 廉价丰富但 noisy（标题党）
- **Watch time**: 比 click 信号强，但有 length bias（长视频天然 watch time 高 → 用 normalized watch time 或 quantile）
- **Completion rate**: dwell ratio
- **Likes / shares / subscribes**: 稀疏但高质
- **Negative**: skip-after-3s、dislike、"not interested"、survey "low quality"

### Negative sampling 策略
- In-batch negatives（candidate gen 阶段）
- Hard negatives：用户看了同一 channel 但没点的视频
- Random negatives：from popular pool（防止 popularity bias）
- ⚠️ 不要把"曝光未点击"当 hard negative（position bias 严重）

### Bias 处理
- **Position bias**：训练时加 position feature，serving 时置 0 → "model debiased"
- **Popularity bias**：normalize label by item popularity，或用 IPW
- **Selection bias**：当前推荐系统决定了哪些 video 被曝光 → exploration slot（5% 流量随机推荐用于训练数据多样性）

### Data pipeline
- 实时日志（client + server）→ Kafka → 离线（Hive / BigQuery）+ 在线（feature store）
- Label window: watch time 至少 7 天聚合
- Training data 量级: 100B+ row/day, 用 sampling

---

## Step 4: Features

| 类别 | 例子 | 工程注意 |
|------|------|---------|
| **User** | demographic、language、device、historical watch sequence (last 50 videos as embedding seq)、subscription list、search history | sequence features → Transformer encoder |
| **Item** | title/description/tags 的 embedding、creator id、category、duration、upload time、historical CTR | content embedding 离线预计算 |
| **Context** | time of day、day of week、device、network speed、previous video in session | session-level state |
| **Interaction** | user-creator past interaction count、user-category affinity、similarity (user-item dot product) | 高基数 cross feature |

**Feature store 设计要点**：
- 离线训练特征 / 在线服务特征**同源**（防 training-serving skew）
- Point-in-time correctness：训练 label time T，feature 必须只用 T 之前的
- Feature freshness 分级：user embedding daily，real-time interaction features secondly

---

## Step 5: Model — Two-stage architecture

### 5.1 召回 (Candidate Generation)
**目标**：从 10B 视频里捞出 top 1000 候选，要快、要多样。

**主流方案：Two-tower model**
```
User tower:  user features  →  256-d embedding
Item tower:  item features  →  256-d embedding
Score    :  dot product 或 cosine
Loss     :  in-batch sampled softmax
```

- **训练**：batch 内其他 item 当 negative，loss = -log( exp(sim(u,i+)) / sum exp(sim(u,i')) )
- **Serving**：item embedding 离线灌进 ANN index（FAISS / ScaNN），user embedding 在线算后做 top-K ANN search → millisecond level
- **多路召回**：除了 two-tower，还有 collaborative filter、heuristic（recent uploads from subscribed creators）、graph-based (PinSage)、freshness 通道

> **Senior 加分**：谈 sampled softmax bias correction（high-freq item over-represented），用 logQ correction（Google 论文 *Sampling-Bias-Corrected Neural Modeling*）。

### 5.2 排序 (Ranking)
**目标**：1000 候选精排到 top 20-50，要准、要多目标。

**架构演进**：
1. **基础**: GBDT (LightGBM / XGBoost) — 强 baseline，特征工程为主
2. **DNN**: Wide & Deep / DCN-V2 / DeepFM
3. **Multi-task DNN**: shared bottom + task-specific tower
4. **MMoE / PLE**: 缓解 task conflict（YouTube 论文 *Recommending What Video to Watch Next*）

**多目标** 同时预测：
- P(click)、P(watch ≥ 50%)、E(watch time)、P(like)、P(share)、P(skip)

**Value model（融合公式）**：
$$V = w_1 \cdot \hat{P}(click) + w_2 \cdot \hat{E}(WT) + w_3 \cdot \hat{P}(like) + ... - w_k \cdot \hat{P}(skip)$$

权重 w_i 用 A/B test 调（不是模型学出来的）。

### 5.3 Re-ranking
- **Diversity**: MMR (Maximal Marginal Relevance)、DPP (Determinantal Point Process)
- **Business rules**: 不在同一个 channel 连续推 3 个、新视频 boost
- **Integrity filter**: 移除 borderline content

---

## Step 6: Serving

### 架构图
```
User request
    ↓
[Feature Store 拉特征 (~30ms)]
    ↓
[Candidate Gen — 多路召回 (~50ms)]
    ↓ ~10000 candidates
[Light ranking — small DNN (~50ms)]
    ↓ top 1000
[Heavy ranking — multi-task DNN (~80ms)]
    ↓ top 100
[Re-ranking — diversity + business rules (~20ms)]
    ↓ top 20
返回客户端
```

### 关键 trade-off
- **Latency vs accuracy**: heavy ranking 模型不能太大 → distillation、quantization、ONNX
- **Freshness vs stability**: 模型 daily retrain？或 online learning？大厂常用 daily fine-tune + weekly full retrain
- **Personalization vs cold start**: 新用户走 demographic-based、popular pool

### Cold start
- **New user**: onboarding survey、demographic-based、popular-by-region
- **New item**: content embedding（title/thumbnail/transcript）、creator history、exploration slot
- **Contextual bandit**：LinUCB / Thompson sampling 在 exploration slot 上

---

## Step 7: Evaluation

### 离线
- 时间切分（不能 random split — 会 leak）：train T0-T7, val T8, test T9
- Segment 分析：new vs existing user, mobile vs desktop, top creator vs long-tail
- **Counterfactual eval (off-policy)**：IPS、SNIPS、Doubly Robust → 在 A/B 之前判断是否值得 ramp

### 在线 A/B
- Sample size: 由 baseline variance + MDE 决定（详见 Q4）
- Ramp: 1% → 5% → 50% → 100%，每步看 guardrail
- Holdout cell: 1% 长期不暴露新模型，用来测 long-term effect
- **Long-term effect** measurement: 1% 用户 6 个月不切回，看 retention / DAU

---

## Step 8: Monitoring & Iteration

| 监控对象 | 指标 | 告警 |
|---------|------|------|
| Feature drift | PSI、KL on each feature | PSI > 0.2 alert |
| Prediction drift | output distribution histogram | 显著偏移 alert |
| Calibration | 预测 P(click) vs 实际 CTR | reliability diagram |
| Business | CTR、watch time、retention | 实时 dashboard |
| Latency | p50/p99/p999 | SLO 违规 alert |
| Model performance per segment | 按 cohort 分 | 长尾 drop alert |

**Feedback loop 处理**：
- 推荐系统 → 用户 → 行为 → 训练数据 → 推荐系统（自我强化）
- 缓解：exploration、debias 训练（IPW）、causal embedding

**Retrain**：
- Light ranking: daily incremental
- Heavy ranking: weekly full retrain
- Two-tower: weekly retrain + 实时更新 user embedding

---

## 常见 Follow-up 问题

1. **"How would you handle cold start for new creator?"**
   - Content-based embedding（thumbnail + title + transcript）
   - Boost in exploration slot
   - 用 creator metadata（topic 分类 + 同 niche 老 creator 的 prior）

2. **"How to measure long-term effect?"**
   - Long holdout cell (1% never see new model)，对比 6 个月 retention
   - Causal forest 找 heterogeneous effect

3. **"How to debias training data from current recommender?"**
   - IPW: weight = 1 / P(item shown to user)
   - Exploration slot 喂随机数据
   - Counterfactual evaluation before launch

4. **"What if engagement up but user satisfaction down?"**
   - Indicates clickbait reward → 加 survey-based label 做 multi-task
   - Add "skip after 3s" / "dislike" 作为 negative
   - Wellbeing guardrail metric

5. **"How would you add LLM into this?"**
   - Generative retrieval（用 LLM 直接生成 candidate item id）
   - Semantic embedding（替换 content embedding）
   - Conversational recsys（user 说"我想看搞笑的"）
   - Re-ranker LLM (cost 太高，可能只在头部 K 候选用)

---

## Trade-off 总结（Senior 必谈）

| 决策 | Option A | Option B | 怎么选 |
|------|---------|---------|--------|
| 召回 | two-tower (双塔) | item-CF | 双塔 cold start 好，CF 历史丰富用户准 → 多路并存 |
| 排序 loss | pointwise | pairwise / listwise | pairwise 对 ranking 直接，但训练复杂 → 折中: pointwise 多目标 + value model 融合 |
| 优化目标 | watch time | satisfied watch time | 后者更长期，但 label 稀疏 → multi-task 混合 |
| Diversity | hard rule | DPP / MMR | DPP 平衡可调 |
| Cold start | popular fallback | exploration | 新用户 popular，老用户加 5% exploration |

---

## 一句话答案 (Elevator Pitch)

> "我会用一个两阶段系统：召回用 two-tower 模型 + ANN 从 10B 视频中召回 1000 候选；排序用 multi-task DNN（MMoE）联合预测 watch time、satisfaction、各类交互信号，再用可调权重的 value model 融合，最后做 diversity re-ranking。关键 senior 信号是 metric 上不能只看 watch time，要叠加 survey-based satisfaction 防 clickbait；data 上要用 exploration slot 和 IPW 处理 feedback loop 和 position bias；evaluation 上要靠 long-term holdout cell 测 retention，不能只看短期 A/B。"
