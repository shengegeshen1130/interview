# DS / ML System Design 学习资料汇总 (Curated Resources)

> **更新日期**: 2026-05
> **目标读者**: senior DS / Applied Scientist / MLE 备战大厂
> **使用方式**: 不要试图都读完——按"核心 → 进阶 → 参考"顺序，根据自己短板选 2-3 个深读

---

## 一、核心书籍 (按优先级排序)

### 🥇 Tier 1：必读

#### 1. *Machine Learning System Design Interview* (Ali Aminian, Alex Xu, 2023)
- **为什么必读**: 大厂 MLE/DS 系统设计面试事实标准教材，10 道经典题（visual search、video recommendation、event recommendation、ad click prediction 等）每题给完整 step-by-step 解法
- **匹配本 case_study**: Q1/Q2/Q3 直接对应
- **怎么用**: 不要照背架构图——学他的"clarification → metrics → data → model → serving"框架
- 链接: [Amazon](https://www.amazon.com/Machine-Learning-System-Design-Interview/dp/1736049127)

#### 2. *Designing Machine Learning Systems* (Chip Huyen, O'Reilly 2022)
- **为什么必读**: 不是面试书，是真做 production ML 的工程视角——training-serving skew、feature store、monitoring、test in production——senior 面试这些都会问
- **匹配本 case_study**: Q1-Q3 的 serving + monitoring 章节
- **怎么用**: 第 6-10 章重点（feature engineering、model dev、deployment、monitoring）

#### 3. *Trustworthy Online Controlled Experiments* (Kohavi, Tang, Xu, 2020)
- **为什么必读**: A/B test 圣经。Microsoft / Airbnb / Bing 三位作者数十年实战
- **匹配本 case_study**: Q4 (A/B Test Deep Dive) 完全基于本书
- **怎么用**: 重点章节：Ch 3 (Twyman's law), Ch 17 (variance reduction CUPED), Ch 22 (network effects)

### 🥈 Tier 2：进阶选读

#### 4. *Introduction to Machine Learning Interviews* (Chip Huyen, free book online)
- **链接**: [https://huyenchip.com/ml-interviews-book/](https://huyenchip.com/ml-interviews-book/)
- **特点**: 免费、200+ 知识题 + 30 open-ended，覆盖广
- **用法**: 当 review checklist 用，不要顺着读

#### 5. *Mostly Harmless Econometrics* (Angrist & Pischke, 2009)
- **匹配**: Q6 因果推断
- **特点**: DiD / IV / RDD 的经典，工业界 senior DS 常拿来 reference
- **用法**: 第 5-6 章 (DiD, RDD)

#### 6. *Causal Inference: The Mixtape* (Scott Cunningham, 2021, free online)
- **链接**: [https://mixtape.scunning.com/](https://mixtape.scunning.com/)
- **特点**: 比 *Mostly Harmless* 更易读，有代码示例
- **用法**: 替代 Mostly Harmless 的 friendly 版本

### 🥉 Tier 3：领域参考

#### 7. *Auction Theory* (Vijay Krishna, 2009)
- **匹配**: Q7 广告 bidding
- **用法**: 第 1-3 章 first-price / second-price / VCG 基础

#### 8. *Recommender Systems: The Textbook* (Charu Aggarwal, 2016)
- **匹配**: Q1 推荐系统
- **用法**: Ch 2-3 collaborative filtering + Ch 12 cold start

---

## 二、必读论文 (按主题)

### 推荐系统 (Q1, Q2)
- **YouTube 推荐**: Covington et al., *Deep Neural Networks for YouTube Recommendations* (RecSys 2016)
- **YouTube 多目标**: Zhao et al., *Recommending What Video to Watch Next* (RecSys 2019)（介绍 MMoE 在 ranking 中的应用）
- **Two-tower**: Yi et al., *Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations* (RecSys 2019, Google)
- **Pinterest PinSage**: Ying et al., *Graph Convolutional Neural Networks for Web-Scale Recommender Systems* (KDD 2018)
- **Meta DLRM**: Naumov et al., *Deep Learning Recommendation Model for Personalization and Recommendation Systems* (2019)
- **Alibaba DIN**: Zhou et al., *Deep Interest Network for Click-Through Rate Prediction* (KDD 2018)

### CTR / Ad (Q3, Q7)
- **Wide & Deep**: Cheng et al., *Wide & Deep Learning for Recommender Systems* (DLRS 2016, Google)
- **Facebook GBDT-LR**: He et al., *Practical Lessons from Predicting Clicks on Ads at Facebook* (ADKDD 2014) — 经典必读
- **Delayed Feedback**: Chapelle, *Modeling Delayed Feedback in Display Advertising* (KDD 2014)
- **ESMM (CTR + CVR)**: Ma et al., *Entire Space Multi-Task Model* (SIGIR 2018, Alibaba)
- **Position Bias (PAL)**: Guo et al., *PAL: a position-bias aware learning framework for CTR prediction* (RecSys 2019, Huawei)
- **Auto-bidding**: Aggarwal et al., *Auto-bidding and Auctions in Online Advertising: A Survey* (2024) — [arXiv](https://arxiv.org/pdf/2408.07685)
- **GSP 经典**: Edelman, Ostrovsky, Schwarz, *Internet Advertising and the Generalized Second-Price Auction* (AER 2007)

### A/B Test (Q4)
- **CUPED**: Deng, Xu, Kohavi, Walker, *Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data* (WSDM 2013, Microsoft)
- **Network Effects**: Saint-Jacques et al., *Using Ego-Clusters to Measure Network Effects at LinkedIn* (2019)
- **Sequential Testing**: Johari et al., *Always Valid Inference: Bringing Sequential Analysis to A/B Testing* (2017)

### 因果推断 (Q6)
- **Synthetic Control**: Abadie, Diamond, Hainmueller, *Synthetic Control Methods for Comparative Case Studies* (JASA 2010)
- **Synthetic DiD**: Arkhangelsky et al., *Synthetic Difference-in-Differences* (AER 2021)
- **Causal Forest**: Wager & Athey, *Estimation and Inference of Heterogeneous Treatment Effects using Random Forests* (JASA 2018)
- **Surrogate Index**: Athey et al., *The Surrogate Index: Combining Short-Term Proxies to Estimate Long-Term Treatment Effects More Rapidly and Precisely* (NBER 2019)
- **Staggered DiD**: Callaway & Sant'Anna (2021), de Chaisemartin & D'Haultfœuille (2020)

---

## 三、博客 / 工程文章 (大厂 engineering blog)

### Meta Engineering
- [https://engineering.fb.com/](https://engineering.fb.com/)
- 关键词搜：news feed ranking, ads ranking, instagram reels recommendation, value model, MSI

### Netflix Tech Blog
- [https://netflixtechblog.com/](https://netflixtechblog.com/)
- 关键词：personalization、recommendation、A/B testing、causal inference、quasi-experiment

### Airbnb Tech Blog (Engineering)
- [https://medium.com/airbnb-engineering](https://medium.com/airbnb-engineering)
- 关键词：search ranking、experimentation、causal inference、two-sided marketplace

### Uber Engineering
- [https://www.uber.com/blog/engineering/](https://www.uber.com/blog/engineering/)
- 关键词：marketplace、surge pricing、experimentation platform、Michelangelo

### LinkedIn Engineering
- [https://engineering.linkedin.com/blog](https://engineering.linkedin.com/blog)
- 关键词：feed、network effect、ego-cluster、experimentation

### Pinterest Engineering
- [https://medium.com/@Pinterest_Engineering](https://medium.com/@Pinterest_Engineering)
- 关键词：PinSage、homefeed、related pins

### Doordash / Instacart
- 关键词：marketplace experimentation、switchback、surge pricing

### Stripe / Spotify Engineering
- 关键词：ML platform、feature store、experimentation

---

## 四、面试准备平台 (付费/部分免费)

### 综合
| 平台 | 特点 | 适用 |
|------|------|------|
| **Exponent** ([tryexponent.com](https://www.tryexponent.com/)) | 大厂 MLSD 整体框架 + mock | 系统训练 |
| **IGotAnOffer** ([igotanoffer.com](https://igotanoffer.com/en/advice/machine-learning-system-design-interview)) | 文章扎实，FAANG 真题分类 | 题目浏览 |
| **InterviewQuery** ([interviewquery.com](https://www.interviewquery.com/)) | 题库大、按公司分 | 刷题 |
| **DataInterview** ([datainterview.com](https://www.datainterview.com/)) | DS-specific，product sense 多 | Product DS 面 |
| **Hello Interview** ([hellointerview.com](https://www.hellointerview.com/)) | ML system design + breakdowns | 系统题 |
| **bytebytego** ([bytebytego.com](https://bytebytego.com/courses/machine-learning-system-design-interview)) | 配 Alex Xu 书 | 书的视频化 |
| **InterviewKickstart** | 偏长期班课 | 系统提升 |

### Mock Interview
- **interviewing.io**: peer mock，有大厂面试官
- **Pramp**: 免费 peer mock
- **ADPList**: 找 mentor 1-on-1

---

## 五、GitHub 资源

### 综合 repo
| Repo | 内容 |
|------|------|
| [alirezadir/Machine-Learning-Interviews](https://github.com/alirezadir/Machine-Learning-Interviews) | MLSD 模板 + 例题 |
| [chiphuyen/machine-learning-systems-design](https://github.com/chiphuyen/machine-learning-systems-design) | Chip Huyen 早期 free book |
| [khangich/machine-learning-interview](https://github.com/khangich/machine-learning-interview) | 大厂 ML 面经 |
| [Devinterview-io/recommendation-systems-interview-questions](https://github.com/Devinterview-io/recommendation-systems-interview-questions) | 推荐系统题库 |
| [eugeneyan/applied-ml](https://github.com/eugeneyan/applied-ml) | 大厂应用 ML 论文/blog 索引 |

---

## 六、YouTube / 视频

- **Stanford CS229M / CS246**: 系统课
- **Data Interview Pro**: DS interview 频道
- **Mahesh Shrivastav**: ML system design 教学
- **Ryan Schachte**: 系统设计 mock
- **Exponent YouTube**: 各种 mock interview 录像

---

## 七、专题 deep dive 推荐

### 想深入 A/B test？
1. 读 Kohavi 的书（必读）
2. 读 Microsoft Experimentation Platform blog
3. 读 LinkedIn ego-cluster paper
4. 读 *Always Valid Inference* paper

### 想深入推荐系统？
1. 读 YouTube 2016 + 2019 paper
2. 读 DLRM paper + 代码
3. 读 Pinterest PinSage paper
4. 读 Eugene Yan's *applied-ml* repo

### 想深入因果推断？
1. 读 *Causal Inference: The Mixtape* (free)
2. 看 Susan Athey 的 lectures
3. 读 Synthetic Control paper
4. 读 Athey-Wager *Causal Forest* paper

### 想深入广告 / bidding？
1. 读 *Auto-bidding and Auctions: A Survey* (2024)
2. 读 GSP 经典 paper (Edelman 2007)
3. 读 Facebook 2014 GBDT-LR paper
4. 读 Alibaba RTB 系列 paper

---

## 八、备战节奏建议（针对 senior DS @ FAANG）

### 4 周计划

**Week 1: 框架 & 推荐系统**
- 读 `00_framework.md` + Q1 (本 case_study)
- 读 *MLSD Interview* book Ch 1-4
- 读 YouTube 2016 paper
- Mock 1 次（推荐系统）

**Week 2: A/B Test & Product Sense**
- 读 `04_ab_test_deep_dive.md` + `05_metric_diagnosis.md`
- 读 Kohavi *Trustworthy Online Controlled Experiments* Ch 1-5, 17, 22
- 刷 InterviewQuery 上 Meta / Airbnb product sense 题 20+
- Mock 1 次（A/B + case study）

**Week 3: Ranking & Ads**
- 读 Q2, Q3, Q7（本 case_study）
- 读 DIN paper + ESMM paper
- 读 *Auto-bidding survey*
- Mock 1 次（ad CTR / bidding）

**Week 4: 因果 & 收尾**
- 读 `06_causal_inference_case.md`
- 读 *Causal Inference: Mixtape* Ch 1, 8, 9, 10
- Review 所有 7 题的"一句话 elevator pitch"——能背
- 终极 mock 2-3 次

---

## 九、面试当天 checklist

**前一天**：
- [ ] Review 所有题的 elevator pitch
- [ ] 准备 3 个自己 own 过的项目，能讲 metric / trade-off
- [ ] 列 5 个 clarifying question 模板（先问业务、再问 scale、再问 constraint）

**面试当天**：
- [ ] 前 5 分钟一定 clarify，不直接进 model
- [ ] 主动 surface trade-off，至少 2 个
- [ ] 提到 monitoring & long-term metric
- [ ] 不知道时大方说"I'd hypothesize X but want to verify"
- [ ] 最后留 5 分钟问面试官好问题（团队、tech stack、roadmap）

---

## Sources（本汇总主要参考来源）

- [Machine Learning System Design Interview (2026 Guide) - Exponent](https://www.tryexponent.com/blog/machine-learning-system-design-interview-guide)
- [Machine Learning System Design Interview Guide (2026) - Interview Kickstart](https://interviewkickstart.com/blogs/articles/machine-learning-system-design-interview-guide)
- [ML System Design Interview - IGotAnOffer](https://igotanoffer.com/en/advice/machine-learning-system-design-interview)
- [Introduction to Machine Learning Interviews (Chip Huyen)](https://huyenchip.com/ml-interviews-book/)
- [Machine Learning Interviews repo - alirezadir](https://github.com/alirezadir/machine-learning-interviews/blob/main/src/MLSD/ml-system-design.md)
- [Top 5 Machine Learning System Design Books and Courses](https://medium.com/javarevisited/top-5-machine-learning-system-design-interview-courses-and-books-for-ml-engineers-in-2025-8476d04960a8)
- [Meta ML System Design Interview Questions and Guide (2026)](https://davidfosterhq.medium.com/meta-ml-system-design-interview-questions-and-guide-2026-39a79bbc2c0b)
- [Video Recommendation System Design - Hello Interview](https://www.hellointerview.com/learn/ml-system-design/problem-breakdowns/video-recommendations)
- [Data Science Case Interview Complete Guide (2026)](https://www.hackingthecaseinterview.com/pages/data-science-case-interview)
- [Airbnb Data Scientist Guide (2026) - Data Interview](https://www.datainterview.com/blog/airbnb-data-scientist-interview)
- [Meta Data Scientist Guide (2026) - Data Interview](https://www.datainterview.com/blog/meta-data-scientist-interview)
- [Product Sense Interview Guide for Data Scientists (2026)](https://medium.com/data-science-collective/demystifying-product-sense-in-data-scientist-interviews-ba25b3bc0cd0)
- [Top 29 Product Sense Interview Questions (2026)](https://www.datainterview.com/blog/product-sense-interview-questions)
- [Auto-bidding and Auctions in Online Advertising: A Survey (arXiv)](https://arxiv.org/pdf/2408.07685)
- [Internet Advertising and the Generalized Second-Price Auction (Edelman)](https://www.benedelman.org/publications/gsp-060801.pdf)
- [Understanding Digital Advertising Auctions: GSP & VCG Variants - Umbrex](https://umbrex.com/resources/economics-concepts/microeconomic-theory/digital-advertising-auctions-gsp-vcg-variants/)
- [The VCG Auction in Theory and Practice - Hal Varian](https://courses.cs.washington.edu/courses/cse490z/20sp/slides/Varian.pdf)
- [About Target CPA bidding - Google Ads Help](https://support.google.com/google-ads/answer/6268632?hl=en)
- [PPC Garage: Under The Hood of Google's Target CPA Bidding Algorithm](https://www.adventureppc.com/blog/ppc-garage-under-the-hood-of-googles-target-cpa-bidding-algorithm)
- [Target CPA Bidding in Google Ads (2026) - Store Growers](https://www.storegrowers.com/target-cpa/)
- [Budget Pacing Guide 2026 - Improvado](https://improvado.io/blog/budget-pacing)
- [Facebook Ad Auctions Explained - Meta for Business](https://www.facebook.com/business/ads/ad-auction)
- [How Machine Learning Facebook Ads Work: 2025 Algorithm Guide - Madgicx](https://madgicx.com/blog/machine-learning-facebook-ads)
- [Recommendation Systems Interview Questions - Devinterview-io](https://github.com/Devinterview-io/recommendation-systems-interview-questions)
