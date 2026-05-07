# Q5: DAU 下降 5% 的根因分析 (Metric Drop Root Cause Analysis)

> **类型**：Product / Case Study (套用 CHIME-D framework)
> **常见 follow-up 公司**：Meta、Google、Airbnb、Uber、Stripe、TikTok
> **难度**：⭐⭐⭐⭐ (考察 product sense + 数据 debug 能力 + 沟通)
> **典型题面**：
> - "Yesterday DAU dropped 5%. Walk me through how you'd investigate."
> - "Conversion rate has been flat for a month — find out why."
> - "App store rating fell from 4.6 to 4.2 — what's going on?"

---

## 一、为什么这道题难（Senior 看什么）

面试官**不期待你猜出答案**——他期待看到：

1. **结构化 thinking**：不要一上来就 "I'd check feature X"。要先有 framework
2. **统计严谨**：5% 下降是 noise 还是 signal？statistical significance？
3. **优先级判断**：先排查最高概率原因，不要 brute force
4. **沟通能力**：你在和 CEO 汇报，不是在写 SQL
5. **Action orientation**：找出原因后怎么修、怎么防再犯

> **典型 junior fail mode**：
> - 立刻跳到"是不是某个 feature 的 bug" → 没 framework
> - 列 50 个 hypothesis 但不分优先级 → 没判断力
> - 不问 metric 怎么定义的 → 不严谨

---

## 二、答题 framework: CHIME-D 应用

### C - Clarify（前 2 分钟必做）

**关于 metric 本身**：
- DAU 怎么定义？unique active user / day? "active" 是什么 (open app? send 1+ event?)
- 数据来源：server-side log 还是 client-side？后者有 bot / fraud
- Time zone? 天的边界？
- 去 bot 了吗？(20-30% raw traffic 可能是 bot)

**关于变化**：
- "下降" 怎么算的？vs 昨天？vs 上周同日？vs YoY？
- 是单一一天还是 trend？
- **一定要看图**——单点 vs 趋势 vs 反复 spike 含义完全不同
- 是不是 statistically significant？(5% 有可能是 normal day-over-day variance)

**关于 scope**：
- 全球 vs 某 region？
- 哪个 platform (iOS / Android / Web)？
- 用户群 (new / returning / power)？
- 哪个 surface / feature？

> **Senior 加分**：在 clarify 阶段就主动说 "在 jump to investigation 之前我想确认这个 5% 是真信号"——展示对数据的怀疑态度。

---

### H - Hypothesize（结构化列假设）

按"内部 vs 外部 × 短期 vs 长期"两维 + "data quality"维度：

#### Bucket 1: 数据质量问题（最高优先级排查！）
- ⚠️ **Instrumentation 故障**：log pipeline 坏了、event schema 改了、SDK 版本 bug、时区错误
- **定义变更**：DAU 定义被人改了、bot filter 规则变了
- **管道 backfill**：某天的数据还没到位
- **采样率变了**：从 100% 改 1%

#### Bucket 2: 内部因素

| 短期 | 长期 |
|------|------|
| Bug (新版本 release 引入 crash) | 产品质量持续下滑 |
| A/B test ramp 50%→100% (新模型差) | UX 改版用户不适应 |
| Infra outage (CDN、API 挂了) | 价格 / 货币化策略变化 |
| Marketing campaign 结束 | Target market 改变 |
| Feature flag 误开 | 算法长期 reward hacking |

#### Bucket 3: 外部因素

| 短期 | 长期 |
|------|------|
| 节假日 (春节、圣诞) | 行业趋势 (TikTok 抢用户) |
| 重大新闻事件 (天灾、政治) | 监管变化 (Apple ATT) |
| 竞品 launch / promo | 用户口味变化 |
| 平台政策 (App Store rule) | 经济衰退 |
| 天气 / 体育赛事 | 市场饱和 |

> **Senior signal**：把这个 framework 显式画出来或口述。让面试官知道你不会漏维度。

---

### I - Investigate（数据驱动验证假设）

#### Step 1: 验证数据质量（5 分钟）

**优先做这一步**——80% 的"DAU 异常"最后是 instrumentation bug，不是真的用户行为变化。

- 看 raw event count、login count、API call count 是否同比例下降
- 看 instrumentation pipeline 监控（Kafka lag、ETL 失败率）
- 比较 client log 和 server log 的差异
- 看不同数据 mart 的 DAU 是否一致

> **真实案例**：Twitter 历史上多次 DAU "下降"被定位为 SDK 升级导致 event 命名变化、bot filter 升级、deduplication 算法 bug。

#### Step 2: 时间维度分析

- **What time exactly**：精确到小时——突变还是 gradual？
- **突变** → 大概率是 deploy / outage / config change
- **Gradual** → 大概率是 product 趋势或 A/B test ramp
- **同日同时段历史对比**：周一通常和周一比，不和周日比 (day-of-week effect)

#### Step 3: Segment 分析（核心！）

把 DAU 按各维度切开，找"哪里跌得最狠"：

| 维度 | 切法 | 暗示 |
|------|------|------|
| **Platform** | iOS / Android / Web / Desktop App | 单 platform 跌 → 该端 release 问题或 platform-side bug |
| **Geo** | Country / Region | 单一 region 跌 → 当地事件 / 监管 / 节假日 |
| **App version** | 各 version DAU | 新版本 DAU 低 → release bug |
| **User cohort** | 新 / 老 / 流失召回 | 新用户跌 → acquisition 问题；老用户跌 → 体验问题 |
| **Acquisition channel** | organic / paid / referral | paid 跌 → ad spend 问题 |
| **Funnel stage** | open → engage → return | 找出漏斗哪一步崩 |

**Simpson's paradox** warning：总体跌 5%，可能只是 user mix 变化（low-engagement segment 增长），各 segment 都没跌。

#### Step 4: 关联指标 (Companion metrics)

DAU 跌的同时其他指标怎么样？

- **Sign-up rate**: 跌 → acquisition 问题
- **Retention (D7, D30)**: 跌 → 体验问题
- **Crash rate**: 升 → bug
- **Latency p99**: 升 → infra 问题
- **Push notification CTR**: 跌 → 通知策略变化
- **External traffic** (search referral, deep link): 跌 → SEO / partner 问题

如果 DAU 跌但其他 metric 全 OK → 强信号是 **instrumentation 问题**。

#### Step 5: 关联事件时间线

把可疑变化和 DAU 时间线对齐：
- Release notes (mobile app, web, backend service)
- Infra changes (CDN cutover, DB migration)
- A/B test ramp 时间表
- ML model push 时间表
- Marketing campaign 起止
- 外部事件 (节假日 calendar, competitor launches, news)

> **真实手段**：bisect-style 二分查找——如果是 5/3 跌，看 5/2 deploy 列表，按 likely impact 排序逐个回滚验证。

---

### M - Measure (量化 confirm)

找到 candidate 原因后要量化：
- 这个原因能解释 5% 中的多少 percentage point？
- 加一起够不够 5%？(常常是多因素叠加)
- Statistical significance：5% 在历史 day-over-day variance 里属于什么 percentile？

**典型 attribution**：
> "5% 下降中：3% 来自 Android 新版本 (v2.5) 的 push notification bug；1.5% 来自欧洲圣诞假期；0.5% noise。"

---

### E - Experiment / Validate

如果定位到 candidate cause，怎么 validate？

| 方法 | 用法 |
|------|------|
| **回滚 + 观察** | 把可疑 release 回滚 1 小时，DAU 恢复? |
| **Canary release** | 1% 用户上 fix，看是否反弹 |
| **Holdout** | 2% 永远在新版本 vs 2% 永远在老版本，长期对比 |
| **DiD (差异中的差异)** | iOS 跌 (treated)、Android 没动 (control)，前后对比 |
| **External signal** | App store review、Reddit、Twitter 提到 bug 吗 |

---

### D - Decide & Communicate

#### 决策树
- 真 bug → 即刻 fix + postmortem
- 外部因素 → 不 actionable，但要监控持续期
- A/B 副作用 → 调整或 kill experiment
- 长期趋势 → escalate to product / leadership

#### Stakeholder 沟通模板
> "**TL;DR**：DAU 下降 5%，主要来自 Android v2.5 的 push notification bug（贡献 3pp）+ EU 圣诞假期效应（贡献 1.5pp），剩 0.5pp 在 noise band 内。
>
> **行动**：(1) Mobile team 已 hotfix push 流程，预计 6h 后恢复；(2) 假期效应历年类似，预计 1 月初回归；(3) 我加了 push notification 的 leading indicator dashboard 防止再发生。
>
> **要 leadership 决策的事**：暂时不需要。"

**Senior signal**：
- 不浪费 leadership 时间猜测 → 给具体 attribution 和 action
- 区分 actionable vs not
- 提防御性措施（dashboard、alert）

---

## 三、Senior 必谈的 6 个深度点

### 1. Survivorship bias
"老用户 retention 没变"——但你只看了今天还在的老用户，流失的不在样本里。

### 2. Simpson's paradox
总体 -5%，每个 segment 都 +/- 0%。原因可能是 user mix 变化（high-DAU segment 比例下降）。要做 weighted decomposition。

### 3. Mix shift decomposition
$$\Delta \text{DAU} = \sum_s \Delta(\text{share}_s) \cdot \text{DAU}_s + \sum_s \text{share}_s \cdot \Delta(\text{DAU}_s)$$

把"组成变化"和"组内变化"分开，是诊断 mix shift 的标准工具。

### 4. Twyman's Law
"任何 surprising 的数据，第一反应是怀疑数据本身。" 5% drop 听起来"合理"，正因为合理才更要查 instrumentation。

### 5. Leading vs lagging indicator
DAU 是 lagging (用户已经流失了)。长期防御要找 leading：crash rate、latency、push CTR、early-session abandonment。

### 6. Postmortem culture
- Blameless postmortem
- 要 propose 防御机制：alert on metric 跌 > X% in 1 hour
- 要 propose root-cause-class fix（不止 fix this bug，fix bug class）

---

## 四、常见 Follow-up 问题

### Q: "How would you build alerting for this?"
- Anomaly detection on DAU + companion metrics
- Statistical: control chart (Shewhart, CUSUM)、Holt-Winters seasonal
- ML-based: Prophet、isolation forest 检测 multivariate anomaly
- Alert 规则：drop > 3 sigma in 1h，分 segment alert（避免被 mix shift 掩盖）
- 防 alert fatigue: severity 分级、duty rotation

### Q: "Conversion 一年才跌 1%，太小怎么办？"
- 这是慢性病，不是急性 → 长期 cohort 分析
- Cohort 看 retention curve 是否退化
- A/B test 之前不会发现的累积 effect → 必须 longitudinal study
- Causal inference：DiD 对比同期未受影响 product

### Q: "你怀疑是 bug，但回滚也没恢复 DAU，怎么办？"
- 说明假设错了，回 step H 重新生成 hypothesis
- 可能是多因素叠加，回滚只解决了一部分
- 看 segment-level 是否部分 segment 已恢复
- 考虑外部因素（竞品、监管）

### Q: "如果 DAU 没动但 revenue 跌了？"
- DAU 是 user count，revenue 是 user × ARPU
- ARPU 跌 → 价格策略、付费率、付费用户混合变化
- 同样套 CHIME-D，但 metric tree 不同：revenue = users × paid_rate × ARPPU
- Decompose by 各 component 找哪个 driver 跌

### Q: "面试官说 'just one more day of data'，你怎么响应？"
- 说明：我已经有 hypothesis，多一天数据能 validate；但同时我已经在做 (1) instrumentation check、(2) recent change list、(3) segment 切片，等 data 回来直接 plug
- 不要被动等——并行推进

---

## 五、答题节奏（30 分钟）

| 时段 | 内容 |
|------|------|
| 0-3 min | Clarify metric 定义 + 时间窗口 + scope |
| 3-5 min | 列 framework (instrumentation / 内部 / 外部 × 短/长) |
| 5-10 min | Investigate: data quality first → 时间分析 |
| 10-20 min | Segment 分析 + 关联指标 + 关联事件 |
| 20-25 min | Quantify attribution + validate |
| 25-30 min | Decision + 沟通 + 防御机制 |

---

## 六、答题模板（背下来）

> "在 jump to investigation 前我想先 clarify 几件事：DAU 是怎么定义的（去 bot 了吗、server vs client log）、时间窗口（vs 昨天还是同期）、scope（全球还是 region）。同时我想强调，5% 在 normal day-over-day variance 里可能是 noise，要先 quantify statistical significance。
>
> 接下来我会按这个 framework 排查：(1) **data quality 先**——instrumentation、定义变更、pipeline lag；(2) **内部因素**——recent release、A/B ramp、infra；(3) **外部因素**——节假日、竞品、新闻。
>
> Investigate 三层：**时间维度** (突变 vs gradual)、**segment 维度** (geo / platform / version / cohort)、**关联指标** (sign-up / retention / crash / latency)。如果能把 5% 拆解 attribution（比如 3% Android bug + 1.5% holiday + 0.5% noise），我就有 actionable 答案。
>
> 然后 quantify + validate：可疑 cause 回滚 1 小时看是否恢复、用 DiD 对比未受影响 segment、看 external signal (App store reviews)。
>
> 最后 communicate：TL;DR + attribution + action + leading indicator dashboard 防御。
>
> Senior 必须避免的 traps：Simpson's paradox、survivorship bias、Twyman's law、混淆 leading vs lagging indicator。"

---

## 一句话答案 (Elevator Pitch)

> "DAU 下降 5% 的诊断核心是**结构化 + 数据质量优先**。先 clarify metric 定义和 statistical significance，然后按 'instrumentation / 内部 / 外部' 分桶生成 hypothesis，按 '时间 → segment → 关联指标 → 关联事件' 顺序 investigate。优先排查 instrumentation（80% 异常是数据问题），再看 release timeline 和 segment 切片定位 cause，用 DiD 或回滚 validate，最后量化 attribution 给 stakeholder。Senior 信号是主动 surface Simpson's paradox / survivorship bias / Twyman's law，并提防御措施（leading indicator dashboard + alert）。"
