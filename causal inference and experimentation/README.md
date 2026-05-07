# 因果推断与实验设计 (Causal Inference & Experimentation)

ML / Data Science 面试中越来越重要的两个领域，特别是 senior 岗位、产品数据科学家和量化角色。

## 文件结构

| 文件 | 内容 |
|---|---|
| [`01_causal_inference.md`](./01_causal_inference.md) | 因果推断基础：potential outcomes、DAG、do-calculus、PSM、DiD、IV、RD、HTE/uplift modeling |
| [`02_experimentation.md`](./02_experimentation.md) | A/B 实验：假设检验、样本量、CUPED、SRM、novelty effect、switchback、双边市场 |
| [`03_interview_questions.md`](./03_interview_questions.md) | 高频面试题 + 简答（FAANG / 量化） |
| [`04_cheatsheet.md`](./04_cheatsheet.md) | 速查表：公式、代码片段、决策树 |

## 学习路径

**1 周冲刺路径（每天 1-2 小时）：**

| Day | 主题 | 文件 |
|---|---|---|
| 1 | Potential outcomes、ATE、SUTVA | `01` 第 1-2 节 |
| 2 | DAG、do-calculus、backdoor/frontdoor | `01` 第 3-4 节 |
| 3 | PSM、IPW、DiD | `01` 第 5-6 节 |
| 4 | IV、RD、HTE | `01` 第 7-8 节 |
| 5 | A/B 测试基础（hypothesis testing、power、sample size） | `02` 第 1-3 节 |
| 6 | CUPED、SRM、novelty effect、stratification | `02` 第 4-6 节 |
| 7 | Switchback、network effects、面试题模拟 | `02` 第 7 节 + `03` |

> 💡 **建议：** 看 `01`/`02` 时同时把代码片段在 Jupyter 里跑一下；面试前 1 天扫 `04` 速查表。

## 重要参考书与论文

**因果推断：**
- *Causal Inference: The Mixtape* (Scott Cunningham) — 实操向，配 Python/R 代码
- *Causal Inference: What If* (Hernán & Robins) — 流行病学视角，免费 PDF
- *The Book of Why* (Judea Pearl) — DAG 与 do-calculus 入门
- *Causal Inference for the Brave and True* (Matheus Facure) — 在线 Python 教程

**实验设计：**
- *Trustworthy Online Controlled Experiments* (Kohavi, Tang, Xu, 2020) — 业界圣经
- Microsoft Research 实验平台博客：https://exp-platform.com
- Netflix Tech Blog、Booking、Airbnb 工程博客

## 谁在面试时会问？

| 公司类型 | 常考方向 |
|---|---|
| FAANG（产品 DS / DSA） | A/B testing、metric design、debugging weird results |
| 量化基金（Citadel, Two Sigma, Jane Street） | Causal identification、IV、selection bias |
| Marketplace（Uber, Airbnb, DoorDash, Lyft） | Switchback、network effects、interference |
| 增长 / 营销（Meta, TikTok） | Uplift modeling、HTE、incrementality testing |
| 医疗 / 政策（FDA, CDC contractors） | RCT、observational studies、PSM |
