# Q3: 设计 Ad CTR Prediction 系统

> **类型**：ML System Design (套用 7-step framework)
> **常见 follow-up 公司**：Meta、Google、Amazon、字节、TikTok
> **难度**：⭐⭐⭐⭐⭐ (比 recsys 多了 **calibration** 和 **auction** 两层难度，是大厂广告业务的核心)
> **与 Q1/Q2 区别**：CTR 是手段不是目的——最终目标是 **eCPM = bid × pCTR**，所以 **绝对概率值**（calibration）和**长尾事件 (低 CTR)** 的精度比 ranking quality 更重要

---

## 题目 (Prompt)

> "Design a click-through rate prediction system for Facebook ads / Google search ads."

变体：conversion rate (CVR) prediction、eCPM ranking、ad bidding system。

---

## Step 1: Problem Clarification

### 必问的 clarifying questions

1. **哪个广告 surface**？
   - Search ads (query 强 intent) — Google search style
   - Display / feed ads (低 intent) — Meta / TikTok feed
   - Video ads (pre-roll / mid-roll)
   - 不同 surface 数据分布、CTR 量级、bidding 机制都不同
2. **Auction 机制**：GSP (Generalized Second-Price)? VCG? First-price？
3. **Bidding model**：CPC (cost per click)? CPM? CPA (cost per action)? oCPM (optimized CPM)?
4. **Goal**: 平台 revenue? advertiser ROI? user satisfaction (relevance)? — 三者要 balance
5. **Conversion definition**：click 还是 download / purchase / subscription？
6. **Scale**: 10B+ daily impressions, 1M+ ads, latency p99 < 100ms（比 recsys 严）

### 简化假设
- Surface: Facebook Feed ads
- Bidding: oCPM（系统优化转化）
- Primary metric: eCPM、advertiser ROI、user relevance
- Latency: p99 < 50ms (per ad scoring), 一次 request 要给几十个候选广告打分

> **Senior signal**：主动指出 "CTR 预估只是中间产物，最终是 eCPM 排序服务于 auction，所以 calibration 比 AUC 重要"。这是面试官想听的。

---

## Step 2: Metrics

### Offline
| Metric | 重要性 | 解释 |
|--------|--------|------|
| **AUC** | ⭐⭐⭐ | ranking 能力 |
| **PR-AUC / log loss** | ⭐⭐⭐⭐ | CTR 极不平衡 (1-3%) AUC 容易虚高 |
| **Calibration (ECE, reliability diagram)** | ⭐⭐⭐⭐⭐ | bidding 用绝对概率，必看 |
| **Calibrated NE (Normalized Entropy)** | ⭐⭐⭐⭐ | Meta 论文用的 metric |
| **GAUC (group AUC)** | ⭐⭐⭐⭐ | 每个 user 内 ranking 准确性 |

### Online (A/B)
- **Revenue**: total revenue, eCPM
- **Advertiser-side**: ROAS (Return on Ad Spend), conversion rate, impression share
- **User-side**: ad CTR, "hide ad" rate, organic engagement (relevance proxy)
- **Marketplace health**: bid landscape, advertiser diversity

### Guardrail
- **User satisfaction**: ad-hide rate、organic feed metric (新模型不能伤组织 engagement)
- **Latency** p99
- **Long-tail advertiser**: small advertiser 的 reach (防垄断)
- **Sensitive content**: 政治广告、医疗广告 compliance

> **关键 senior 概念**：CTR 模型有 calibration → bid landscape → advertiser bidding behavior → 模型训练数据的 **反馈环**。任何模型变化都要监控 marketplace 是否 stable。

---

## Step 3: Data

### Labels
- **Click** (binary, immediate, 1-3% positive rate)
- **Conversion** (binary, **delayed** —— 可能 7-30 天才发生)
  - **Delayed feedback problem**：训练时 "no conversion yet" 的样本到底是 negative 还是未到？
  - 解决：delayed feedback model（Chapelle & Manavoglu 2014）—— 用两个 model 联合预测 P(convert) 和 P(time-to-convert)

### Negative sampling
- 真 negative：曝光未点击 (但有 position bias)
- Sub-sampling：CTR=2%，可以 keep 全部 positive + sample 10% negative，然后 calibrate 回去：
  $$p_{\text{cal}} = \frac{p_{\text{model}}}{p_{\text{model}} + (1 - p_{\text{model}}) / w}$$
  其中 $w$ = neg sampling rate

### Class imbalance
- 99% negative → 默认模型输出全 0 也 99% accuracy
- 用 log loss 和 calibration metric，不要用 accuracy
- Focal loss、class weight、re-balanced sampling

### Bias 处理
- **Position bias**：广告位置极大影响 CTR
  - **PAL (Position-Aware Learning)**：训练时把 position 作为 feature，serving 时设为 default position
  - **Inverse Propensity Weighting (IPW)**：weight = 1 / P(seen at position k)
- **Selection bias**：当前 ranker 决定哪些广告曝光 → exploration slot + counterfactual eval

---

## Step 4: Features

### 高基数 categorical features
- ad id (10M+)、advertiser id、user id (1B+)、creative id、ad slot id
- 用 **embedding** + hashing trick（哈希到固定维度）
- **Frequency cap embedding**: low-freq id 的 embedding 不稳定 → 频次过滤、shared embedding

### 数值 features
- Historical CTR (per ad, per advertiser, per user)
- Bid amount
- Time since ad created, ad fatigue (impressions to this user)
- User's recent ad interactions

### Cross features (经典的 CTR feature engineering)
- (user_id, ad_category) 的 historical CTR
- (user_demographic, advertiser_category) cross
- 由 model 自动学（DCN、DeepFM、xDeepFM）

### Sequence features
- 用户最近 100 个 ad 交互序列 → Transformer encoder
- DIN (Deep Interest Network)：用 attention 在 user history 上聚焦与目标 ad 相关的部分

---

## Step 5: Model

### 5.1 架构演进

| 时代 | 架构 | 特点 |
|------|------|------|
| Pre-DL | LR + 大量人工 cross feature | calibration 好 |
| 2014 | GBDT + LR (Facebook 2014 paper) | 树做 feature transform |
| 2016 | Wide & Deep (Google) | memorization + generalization |
| 2017+ | DeepFM, DCN, xDeepFM | 自动 cross |
| 2018+ | DIN, DIEN (Alibaba) | attention on user history |
| 2019+ | DLRM (Meta) | 工业级 dense + sparse |
| 2022+ | Transformer-based, MoE | scale to billions of params |

### 5.2 推荐架构：DLRM + DIN-style attention

```
[user history seq]  →  [Transformer / DIN attention against target ad]  →  user-interest emb
[user features]     →  [embedding + dense]  ─┐
[ad features]       →  [embedding + dense]  ─┼→ [feature interaction (DCN / FM / dot)]  →  [DNN]  →  pCTR
[context features]  →  [embedding + dense]  ─┘
```

### 5.3 Multi-task: CTR + CVR
- Click → Conversion 是漏斗：$P(\text{conv}) = P(\text{click}) \cdot P(\text{conv} | \text{click})$
- **ESMM** (Entire Space Multi-task Model, Alibaba)：同时学 pCTR 和 pCVR，在整个曝光空间上训练（避免 sample selection bias —— CVR 只能在 clicked 样本上学）

### 5.4 Calibration（关键）
模型 raw 输出 ≠ 准确的 P(click)，必须 post-calibrate：

| 方法 | 优势 | 劣势 |
|------|------|------|
| Platt scaling | 简单 | 假设 sigmoid form |
| Isotonic regression | 无参 | 需要足够数据 |
| Beta calibration | 灵活 | 较新 |
| Temperature scaling | DNN 适用 | 单 scalar |

实践：每个 segment 单独 calibrate（new user 和 power user 的 base rate 不同）。

### 5.5 处理 delayed conversion
- **Delayed Feedback Model**: 联合建模 P(conv|click) 和 P(time-to-conv)
- **Importance sampling**：负样本随时间 re-weight
- **Online learning**：streaming update with delayed labels

---

## Step 6: Serving — 端到端 Auction

### 整个 ad serving pipeline
```
Ad request (user × surface × context)
    ↓
[Ad retrieval：根据 targeting (geo, demo, interest) 召回 候选 ads — 可能 10K candidates]
    ↓
[Light ranking model — 粗排打分到 top 200]
    ↓
[Heavy ranking model — 精排打 pCTR / pCVR (~30ms)]
    ↓
[Bidding：advertiser 设的 bid × pCTR (or pCVR) = eCPM]
    ↓
[Auction (GSP / VCG)，pick winner，second-price 计费]
    ↓
[Re-ranking：relevance threshold、frequency cap、policy filter]
    ↓
返回 winning ad
```

### eCPM 公式
- CPC bidding: $\text{eCPM} = \text{bid} \times \text{pCTR} \times 1000$
- CPM bidding: $\text{eCPM} = \text{bid}$
- oCPM: advertiser 出 conversion 出价，$\text{eCPM} = \text{bid} \times \text{pCTR} \times \text{pCVR} \times 1000$

### Reserve price
- 平台设 floor，eCPM < floor 不展示
- 防 advertiser 用极低 bid 赢得 auction

### Auction quality vs revenue trade-off
- 加 quality multiplier: $\text{rank score} = \text{eCPM} \times \text{quality}^\alpha$
- $\alpha$ 越大越偏 user 体验，越小越偏 revenue → A/B 调

---

## Step 7: Evaluation

### Offline
- Time-based train/val/test split (T0-T13 train, T14 val, T15 test)
- Per-segment: new advertiser、small advertiser、long-tail ad
- **Counterfactual evaluation** before A/B：用 IPS estimator 估计新模型的预期 revenue / CTR
  $$\hat{V}_{\text{IPS}} = \frac{1}{N} \sum_i \frac{\pi_{\text{new}}(a_i | x_i)}{\pi_{\text{old}}(a_i | x_i)} r_i$$

### Online A/B
- **Primary**: revenue per request, eCPM
- **Secondary**: CTR、CVR、advertiser ROAS distribution
- **Guardrail**: ad-hide rate、organic engagement、长尾 advertiser reach
- **Long-term**: advertiser retention, marketplace health
- ⚠️ **A/B 注意**：广告 A/B 有**两侧 randomization** 问题——你 randomize user 还是 ad？ user-level randomization 对 advertiser 有 confounding

### Special: Holdout for marketplace
- 1% user 永远看 baseline 模型
- 衡量长期 advertiser bid landscape 变化

---

## Step 8: Monitoring & Iteration

### 关键监控
| 维度 | 指标 | 重要性 |
|------|------|--------|
| **Calibration drift** | reliability diagram, ECE per segment | ⭐⭐⭐⭐⭐ — bid 直接受影响 |
| Feature drift | PSI per feature | ⭐⭐⭐⭐ |
| Marketplace stability | bid landscape, eCPM distribution | ⭐⭐⭐⭐⭐ |
| Long-tail advertiser reach | small advertiser impression share | ⭐⭐⭐⭐ |
| Latency | p99 < SLO | ⭐⭐⭐⭐⭐ |
| Ad-hide rate | per surface, per segment | ⭐⭐⭐⭐ |

### Retrain
- **Online learning**: streaming update (Meta、Alibaba 都做)
  - 5-10 分钟级 delay
  - 处理 distribution shift 极快
- **Daily / hourly batch retrain**: incremental
- **Weekly full retrain**: 防 catastrophic drift

### 边缘 case
- **Campaign 突然爆量 ad**：pCTR 短期被低估 → 加 ad freshness boost
- **新 ad cold start**：advertiser-level prior、creative content embedding、exploration budget
- **Holiday / event**：seasonal model, hour-of-day re-calibration

---

## 常见 Follow-up 问题

1. **"Why calibration matters more than AUC for ads?"**
   - 因为 bid = bid_advertiser × pCTR，pCTR 偏差直接传到 auction
   - AUC 只关心 ranking，不管绝对值
   - 例：AUC 相同，但 pCTR 整体高估 2× → 平台超收 advertiser 的钱 → advertiser 流失

2. **"Delayed conversion 怎么处理？"**
   - Delayed feedback model: 联合 P(convert) 和 time-to-convert
   - Streaming label update：先把样本标 negative，convert 来了再 patch
   - Importance weighting

3. **"Position bias 怎么 debias?"**
   - **PAL (Position-Aware Learning)**：训练时位置 feature，serving 设默认值
   - 数学上：assume CTR(u, ad, pos) = CTR(u, ad) × P(seen at pos)
   - 如果 ad 重排不知道最终位置，可以预测 expected position

4. **"How to balance revenue vs user experience?"**
   - quality multiplier: eCPM × quality_score^α
   - Diversity / hide-rate guardrail
   - Survey-based ad-relevance label
   - Long-term holdout cell 测 retention

5. **"Cold-start advertiser 怎么办？"**
   - Advertiser-level prior（同类 advertiser 的 historical CTR）
   - Creative content embedding
   - Exploration budget：新 ad 强制曝光 K 次收 signal
   - Multi-armed bandit (LinUCB)

6. **"如何知道改进 model 不是恶化 marketplace？"**
   - 监控 advertiser bid distribution / churn / ROAS distribution
   - 长期 holdout cell
   - 监控 long-tail advertiser 的 reach（防垄断）

7. **"Auction 类型怎么选？(GSP vs VCG vs First-price)"**
   - **GSP**: 简单、稳定、industry standard but not truthful
   - **VCG**: truthful (advertiser 没动机虚报 bid) but 复杂
   - **First-price**: simple, 但 advertiser 要 shading
   - 大多数广告平台用 GSP 或 hybrid

---

## Senior Trade-off 总结

| 决策 | 取舍 | 怎么做 |
|------|------|--------|
| AUC vs Calibration | ranking quality vs absolute prob | 都要看，calibration 优先 |
| Online learning vs batch | freshness vs stability | hybrid: 5-min streaming + daily full retrain |
| Negative sampling rate | training speed vs calibration | sample 后做 calibration correction |
| Heavy model vs latency | accuracy vs serving cost | distillation, ONNX, quantization |
| eCPM 优化 vs user satisfaction | revenue vs LT retention | quality multiplier, hide-rate guardrail |
| Delayed conversion attribution window | coverage vs label correctness | 7d for click-through, 28d for view-through |

---

## 一句话答案 (Elevator Pitch)

> "广告 CTR 预估的核心区别是它服务于 auction (eCPM = bid × pCTR)，所以 **calibration** 比 ranking AUC 更重要。架构上是 ad retrieval → light ranking → heavy DLRM/DIN 模型预测 pCTR/pCVR → calibrate → 进 GSP auction → second-price 计费。data 上要处理 (1) class imbalance (CTR 1-3%)、(2) position bias 用 PAL、(3) delayed conversion 用 delayed feedback model；model 上多任务联合 ESMM 学 CTR/CVR；serving 必须 streaming online learning + per-segment calibration；monitoring 重点是 calibration drift 和 marketplace health (bid landscape、long-tail advertiser reach)。最 senior 的点是要把 ML 决策和 auction theory + marketplace dynamics 联起来——一个看似无害的 model change 可能 destabilize advertiser bidding behavior。"
