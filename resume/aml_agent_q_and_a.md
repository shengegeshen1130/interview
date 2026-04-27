# AML Investigation Mate 面试准备

---

## Part 1: Project Story（项目背景与故事线）

---

### 1.1 项目背景

Anti-Money Laundering（AML）是金融机构的合规硬性要求。当一个 case 被 alert system 触发后，investigator 需要手动完成一整套调查流程：

- **耗时长：** 一个 case 从开始 review 到最终提交，平均需要 3-4 小时
- **流程复杂：** 需要综合分析账户 profiling、transaction pattern、外部数据（LexisNexis、LinkedIn 等），再对照内部 SOP 和外部法规做判断
- **写 report 痛苦：** 每个需要上报的 case 都要写一篇符合格式要求的 SAR（Suspicious Activity Report），向 regulator 说明账号如何参与 AML 行为，这是法律要求
- **经验依赖重：** 很多决策逻辑存在 investigator 个人经验和 team 内部文档中，新人上手慢，不同人判断标准不一致

### 1.2 Use Case

做一个可以 **端到端** 自动化 AML case 调查的 multi-agent 系统：

1. **Evidence Retrieval：** 从不同数据源搜集、处理 case 相关数据
2. **Rule Reasoning：** 基于 SOP 和法规，对数据做规则推理和风险判断
3. **Decision Recommendation：** 给出是否需要 file SAR 的建议
4. **SAR Writing：** 自动生成符合格式要求的 SAR report
5. **Interactive Q&A：** investigator 可以和 agent 交互，追问 case 细节或修改分析

**系统定位：** 辅助而非替代——investigator 始终拥有最终决策权，agent 提供分析、建议和 report 草稿，investigator review 后提交。

### 1.3 核心难点

1. **DS 不了解 SOP 和法规：** 作为 data scientist 和 engineer，我们不掌握内部 SOP 和外部 AML 法律条文的细节。这些知识大部分存在 investigator 的个人经验、team 内部文档、以及分散的 regulatory document 中
2. **不了解 investigator 的决策逻辑：** 我们不知道 investigator 在看 case 的时候关注什么方向、用什么计算逻辑去判断，这些逻辑没有显式文档化

**解决方案：**

1. **Graph Knowledge Base：** 搜集内部 SOP 和外部 AML 法律条文，建了一个 graph knowledge base（Neo4j），连接 SOP → condition → action → legal document。agent 通过 Graph-RAG 获取 guidance，investigator 有流程疑问时也可以直接 query
2. **历史 Report 反向推理：** 分析历史上 filed 过的 SAR report 和对应的 case 信息，反向推理出 investigator 在 review case 时的关注方向和计算逻辑。这些推理结果决定了 sub-agent 和 analytical tool 的具体实现

### 1.4 Impact 量化

**80% Decision Accuracy 的计算方式：**
- 从历史 closed case 中选取 100 个 case 作为 test set，这些 case 有 investigator 的最终 decision（file / no file）和 senior reviewer 的确认
- 让 agent 对每个 case 独立做分析并给出 decision recommendation
- **Accuracy 定义：** agent 的推荐 decision 与 investigator + senior reviewer 的最终 decision 一致的比例
- 80% = 80/100 个 case 的 decision 一致
- 剩余 20% 主要集中在边界 case（evidence 不充分、需要主观判断的情况）和涉及复杂 multi-entity 关系的 case
- "on par with senior investigators" 的含义：我们同时让 junior investigator 做同样的 test，他们的 accuracy 约 70-75%

**85% Time Reduction 的衡量方式：**
- 选取 15 个 investigator（混合 senior 和 junior），对比使用 agent 前后完成相同类型 case 的时间
- 使用前：平均 3-4 小时（包括数据收集、分析、写 report）
- 使用后：平均 25-35 分钟（review agent 的分析 + 修改 report + 最终确认）
- 85% 是取中位数的 reduction ratio：从 ~3.5h 降到 ~30min

---

## Part 2: Technical Design（技术设计详解）

---

### 2.1 整体架构

系统采用 **main agent + sub-agents** 的 multi-agent 架构：

- **Main Agent：** 负责理解用户 request、根据 case 状态和任务难度做 planning、调度 sub-agent、汇总结果
- **Sub-Agents：** 按职责分为三类（详见 2.3），每个 sub-agent 有独立的 context，main agent 只负责提供 input 和处理 output
- **Knowledge Layer：** Graph-RAG 提供 SOP 和法规的 grounded reasoning
- **File System：** 所有 sub-agent 的分析结果存储在 file system 中，通过路径暴露给需要的 agent

**架构设计的核心考量：**
- Context 隔离：sub-agent 的 context 不会污染 main agent，保证 main agent 的 context window 可以支持多轮对话
- 可扩展：新增分析能力只需要加新的 sub-agent，不需要改 main agent 的逻辑
- 并行执行：不同 sub-agent 可以并行处理独立的数据采集和分析任务

### 2.2 知识层：Graph-RAG

**为什么用 Graph 而不是普通 Vector RAG：**
- SOP 和法规文档之间有明确的层级和引用关系（法规 → SOP 条款 → 具体 condition → 对应 action），这种关系用 graph 天然建模
- 一个 condition 可能关联多个 action，一个 action 可能引用多个法律条文，graph 可以精确遍历这些关系
- 普通 RAG 只能做语义检索，无法做 "从这个 condition 出发，找到所有相关的 action 和对应的 legal requirement" 这样的结构化推理

**Neo4j Schema 设计：**

| Node Type | 说明 | 示例 |
|-----------|------|------|
| **SOP_Document** | 内部 SOP 文档 | "AML Investigation Playbook v3.2" |
| **Legal_Document** | 外部法规 | "BSA/AML Manual - FinCEN" |
| **Section** | 文档中的章节 | "Chapter 4: Transaction Monitoring" |
| **Condition** | 触发条件 | "Transaction amount > $10K in 24h" |
| **Action** | 对应操作 | "File SAR within 30 days" |
| **Risk_Indicator** | 风险指标 | "Structuring pattern detected" |

| Relationship | 说明 |
|-------------|------|
| `CONTAINS` | Document → Section |
| `DEFINES` | Section → Condition / Action |
| `TRIGGERS` | Condition → Action |
| `REFERENCES` | SOP_Document → Legal_Document |
| `INDICATES` | Risk_Indicator → Condition |

**检索策略：**
1. **Query Routing：** 根据用户问题的意图，决定走 graph traversal 还是 vector search
   - 结构化问题（"这个 condition 应该怎么处理"）→ graph traversal，从 Condition 节点出发遍历关联的 Action 和 Legal_Document
   - 语义问题（"有没有关于 crypto transaction 的规定"）→ vector search，对 Section 节点的文本做 embedding 检索
2. **Hybrid Search：** 对于复杂问题，先用 vector search 定位相关 Section，再用 graph traversal 展开关联的 Condition → Action 链路
3. **Context Enrichment：** 检索到目标节点后，自动携带上下文（parent document、关联的 legal reference），让 LLM 有足够的信息做推理

### 2.3 Sub-Agent 分类与职责

**第一类：数据采集类 Agent**
- 从不同数据源搜集、处理 case 相关数据
- 包括内部数据库查询（账户信息、交易记录）、外部数据获取（LexisNexis、LinkedIn、社交媒体）
- 输出标准化的 data profile 文件

**第二类：分析类 Agent**
- 负责不同 domain 的分析，每个 agent 配有一系列负责逻辑计算的 tool：
  - **Transaction Velocity Agent：** 分析交易频率、金额变化趋势、时间序列异常
  - **Transaction Density Agent：** 分析交易集中度、network density
  - **Crypto Agent：** 分析加密货币相关交易模式
  - **Pattern Recognition Agent：** 识别 structuring、layering 等常见 AML pattern
- 另外一部分 agent 处理无法通过纯计算得到结论的内容：
  - Transaction memo 中是否提到 risky 的关键词
  - Email address 是否是随机生成的（乱码检测）
  - P2P 交易方之间的对话是否有可疑内容

**第三类：Report / Knowledge Agent**
- **Report Writer Agent：** 根据分析结果，按照 SOP 格式要求生成 SAR report
- **Decision Agent：** 综合所有分析结果，对照 SOP 和法规，给出 file / no-file 的建议
- **Knowledge Retrieval Agent：** 负责 Graph-RAG 检索，回答 investigator 关于 SOP 和法规的问题

### 2.4 Implementation

**LangChain Deep Agent 框架：**
- 采用 LangChain 1.0 发布的 **Deep Agent** 框架
- 这是一个以 file system 命令为主要工具、支持 sub-agent 并行的框架，built on LangGraph
- 选择原因：
  - 原生支持 sub-agent 并行执行，适合我们多个数据采集和分析 agent 同时工作的场景
  - File system 作为通信媒介，天然解决了 sub-agent 之间的 context 隔离问题
  - Built on LangGraph，继承了 state management、checkpoint、human-in-the-loop 等能力

**File System 通信机制：**
- 所有 sub-agent 的分析结果（analytical report）存储到 file system 的指定路径
- Main agent 通过路径读取 sub-agent 的输出，进行汇总和 planning
- 优势：解耦了 sub-agent 之间的依赖、支持异步执行、中间结果可追溯

**Planning 机制：**
- Main agent 根据 request 的难易程度和 case 状态进行 planning
- Planning 产出一个 step-by-step 的 to-do list
- 每完成一个 step 后，main agent 会 track 进度并决定下一步

### 2.5 Context Management

在 multi-agent + 多轮对话场景下，context management 是关键挑战：

1. **Summarization：** 每累积约 1M token 的对话内容，对之前的对话做一个总结，append 到最近的 message 里，保留最近 20 条 message 的完整内容
2. **Message Pruning：** 定期删除 context 中占空间大但信息密度低的部分，主要是 edit file 的 tool call（包含完整文件内容但对后续对话没有价值）
3. **Sub-Agent Context 隔离：** 每个 sub-agent 有独立的 context window，不与 main agent 共享。Main agent 只接收 sub-agent 的最终 output，不接收过程 context

---

## Part 3: 优化与演进

---

### 3.1 Prompt Optimization

**背景：** Agent 的输出质量高度依赖 prompt 的质量，手工调 prompt 效率低且难以系统化。我们在单个 sub-agent 上做了 auto prompt optimization 的尝试，效果非常好。

**使用的框架和方法：**

| 方法 | 核心思想 | 适用场景 | 我们的使用 |
|------|---------|---------|-----------|
| **DSPy** | 把 prompt 当作可优化的 "程序"，定义 signature（input → output），用 optimizer 自动搜索最优 prompt | 有明确 input/output 对的任务 | 作为基础框架，定义每个 sub-agent 的 signature 和 metric |
| **GEPA** | Genetic Evolution of Prompts and Architectures，用遗传算法搜索 prompt 变体 | 需要探索大的 prompt 空间 | 初始 prompt 优化阶段，快速搜索好的 prompt 结构 |
| **MIPROv2** | DSPy 内置的 optimizer，用 Bayesian 方法优化 few-shot example 选择和 instruction | 有 few-shot examples 的 RAG 场景 | 优化 Knowledge Retrieval Agent 的检索 prompt |
| **TextGrad** | 把 LLM 输出的 "文本" 当作可微分变量，用 LLM 生成 "梯度"（反馈），迭代优化 prompt | 需要细粒度优化 prompt 措辞 | 精调 Report Writer Agent 的 prompt，提升 report 质量 |

**优化流程：**
1. 先用 GEPA 做粗搜索，找到若干 promising 的 prompt template
2. 用 MIPROv2 优化 few-shot example 的选择
3. 用 TextGrad 对最终 prompt 做细粒度的措辞优化
4. 所有优化都基于 DSPy 框架管理 signature 和 evaluation

### 3.2 正在进行的研究

1. **Latency 优化：** Deep Agent 在调用 tool 和 middleware 时存在延时，正在排查瓶颈（可能是 serialization、network round trip、或 LangGraph 的 state checkpoint overhead）

2. **Skill 支持：** LangChain 最近更新了对 skill 的支持，正在评估如何将现有的 workflow 步骤封装为 skill，使 agent 能更灵活地组合

3. **动态 Tool 生成：** 目前所有 analytical tool 是写死的，main agent 只能调整参数。正在开发一个功能：根据用户输入，参考已有的 analytical tool，现场生成新的 tool 然后执行分析
   - 技术挑战：sandbox 环境执行、代码正确性验证、安全性保障
   - 这是下一阶段的重点方向

4. **Multi-Agent Optimization：** 整体系统级别的优化，难度很大。三个方向：
   - **Auto Prompt Optimization：** 扩展到整个 agent system（已在单个 sub-agent 上验证有效）
   - **Agentic Context Engineering：** 用 LLM-as-judge 对每次完整的 multi-agent trace 做 reflection，将 reflection 结果存入外置 workbook，agent 运行时导入。这种 "物理外挂" 的方式对原有代码改动最小
   - **AgentLightening（微软）：** 号称不改代码，直接对 prompt 或 LLM model 做 tuning。核心是把 GRPO apply 到 multi-step agent 上，通过封装层获取 trace 数据，结合 reward function 做优化。瓶颈：缺少足够的 trace 数据

### 3.3 当前限制与反思

1. **数据不足：** 用于优化的 case trace 数据量有限，制约了 AgentLightening 等 data-driven 方法的应用
2. **Latency：** 端到端的响应时间还不够快，investigator 对交互式场景的等待容忍度有限
3. **边界 Case 处理：** 涉及主观判断的边界 case，agent 的准确率还有提升空间
4. **Tool 灵活性：** 固定 tool 集合限制了 agent 处理非标准 case 的能力，动态 tool 生成是解决方向但尚未成熟
5. **Multi-Agent Debugging 难度大：** 当输出质量有问题时，定位是哪个 sub-agent 的问题需要逐层 trace，工具链还不够完善

---

## Part 4: 核心技术问题

---

### Q1: Graph-RAG 和普通 RAG 有什么区别？为什么用 Graph-RAG？

**回答：**

1. **核心区别：**

   | 维度 | 普通 Vector RAG | Graph-RAG |
   |------|----------------|-----------|
   | **数据建模** | 把文档 chunk 成片段，存为 embedding vector | 把知识编码为 graph（节点 + 关系） |
   | **检索方式** | 语义相似度匹配（cosine similarity） | 语义检索 + graph traversal（关系遍历） |
   | **上下文** | 返回最相似的 chunk，chunk 之间无关联 | 返回节点 + 关联节点，自带结构化上下文 |
   | **推理能力** | 只能做 "找到相关内容"，无法做链式推理 | 可以做 "从 A 出发，经过 B，到达 C" 的多跳推理 |
   | **适用场景** | 知识碎片化、文档之间无明显关联 | 知识有层级/引用关系、需要关系推理 |

2. **为什么我们的场景适合 Graph-RAG：**
   - SOP 和法规之间有明确的层级关系：法规 → SOP 条款 → condition → action
   - 一个查询经常需要多跳推理："这个 transaction pattern 触发了什么 condition → 对应什么 action → 依据什么法规"
   - Investigator 的问题经常是结构化的："这个 condition 对应的 action 是什么？" 而不是模糊的语义搜索
   - 法规更新时，graph 只需要更新受影响的节点和关系，而不是重新 embed 所有文档

3. **实际使用中的 hybrid 策略：**
   - 不是纯 graph，结合了 vector search：先用语义检索找到相关的 Section 节点，再用 graph traversal 展开关系
   - 对于没有明确结构化关系的内容（如 guidance notes、best practices），仍然用 vector RAG

> **Follow-up 提示：** 面试官可能追问 "Graph-RAG 的 graph 怎么构建的？是手动标注还是自动抽取？"、"graph 数据多久更新一次？SOP 变了怎么办？"、"有没有遇到 graph traversal 返回太多结果的情况？"

---

### Q2: Short-term / Long-term / Task Memory 怎么实现的？

**回答：**

1. **Short-term Memory（对话级）：**
   - 就是当前对话的 message history
   - 通过 summarization + message pruning 管理（每 ~1M token 做一次总结，保留最近 20 条完整 message）
   - 目的：让 agent 记住当前对话中已经讨论过的内容，避免重复分析

2. **Long-term Memory（跨对话）：**
   - 实现方式：外置 workbook + Graph KB
   - **外置 workbook：** 存储 agent 在历史对话中学到的 patterns、常见问题的处理方式、investigator 的偏好。通过 LLM-as-judge 对每次完整 trace 做 reflection 后写入
   - **Graph KB：** SOP 和法规知识，本身就是 long-term knowledge，通过 Graph-RAG 检索
   - Agent 每次启动时，加载 workbook 的相关内容到 system prompt

3. **Task Memory（任务级）：**
   - 针对当前正在处理的 case 的所有中间结果
   - 实现方式：file system 中按 case ID 组织的目录结构
   - 包括：每个 sub-agent 的分析报告、数据采集结果、planning 的 to-do list 和进度
   - 当 main agent 需要回顾之前的分析时，直接读取 file 而不是从 context 中搜索
   - 优势：不占用 context window、可持久化、支持跨 session 继续处理同一个 case

4. **三者的关系：**
   - Short-term memory 管理 "当前对话记住了什么"
   - Task memory 管理 "当前任务产出了什么"
   - Long-term memory 管理 "从历史中学到了什么"
   - Main agent 在每次决策时，综合参考三者的信息

> **Follow-up 提示：** 面试官可能追问 "long-term memory 的 workbook 具体存什么？格式是什么？"、"task memory 在 file system 中的目录结构是怎样的？"、"如果 long-term memory 存了错误的 pattern 怎么纠正？"

---

### Q3: Prompt Optimization 具体怎么做的？DSPy 怎么用的？

**回答：**

1. **DSPy 的角色：**
   - 用 DSPy 定义每个 sub-agent 的 **signature**：明确 input 是什么、output 是什么、evaluation metric 是什么
   - 例如 Decision Agent 的 signature：`CaseData, AnalysisReports → Decision(file/no_file), Confidence, Reasoning`
   - DSPy 提供了统一的 optimization 框架，让我们可以在同一个体系下使用不同的 optimizer

2. **优化流程（以 Decision Agent 为例）：**
   - **Step 1：定义 metric** — 用历史 case 的 ground truth decision 做评估，metric = decision accuracy
   - **Step 2：GEPA 粗搜索** — 生成多个 prompt 变体（改变 instruction 措辞、reasoning 步骤的顺序、evidence 组织方式），用遗传算法筛选
   - **Step 3：MIPROv2 优化** — 对筛选出的 top prompt，优化 few-shot example 的选择（从历史 case 中选最有代表性的示例）
   - **Step 4：TextGrad 精调** — 对最终 prompt 的关键措辞做微调，基于 LLM 生成的 "gradient"（反馈）迭代

3. **效果：**
   - 单个 sub-agent 上 prompt optimization 效果显著，accuracy 提升了 10-15 个百分点
   - 但 multi-agent 级别的 optimization 还在探索中——优化一个 sub-agent 的 prompt 可能影响其他 sub-agent 的表现

4. **挑战：**
   - 评估数据有限（只有 100 个 test case），优化容易过拟合
   - Multi-agent 的优化空间太大，单独优化各 sub-agent 不等于全局最优
   - Prompt optimization 的 cost 高（需要大量 LLM 调用做 evaluation）

> **Follow-up 提示：** 面试官可能追问 "GEPA 的遗传算法具体怎么设计的？crossover 和 mutation 操作是什么？"、"TextGrad 的 gradient 具体是什么形式？"、"怎么防止 prompt optimization 过拟合？"

---

### Q4: 怎么评估 agent 的 decision accuracy（80% 怎么来的）？

**回答：**

1. **评估数据集：**
   - 从历史 closed case 中选取 100 个 case，覆盖 file 和 no-file 两种 decision，以及不同复杂度和 case type
   - 每个 case 有 investigator 的原始 decision + senior reviewer 的确认，作为 ground truth

2. **评估流程：**
   - 让 agent 对每个 case 做完整的端到端分析（和 production 流程一致）
   - Agent 输出：decision（file / no-file）、confidence score、reasoning
   - 对比 agent decision 与 ground truth

3. **Accuracy 计算：**
   - 80% = 80/100 case 的 decision 一致
   - 按 case type 拆分：简单 case（clear evidence）accuracy ~95%，复杂 case（边界 case）accuracy ~60%

4. **补充评估维度：**
   - **Decision Accuracy：** 80%（上述）
   - **Report Quality：** 由 investigator 打分（1-5 分），平均 3.8 分——"可以作为草稿直接修改，不需要重写"
   - **Evidence Coverage：** agent 找到的 evidence 是否涵盖了 investigator 在 ground truth report 中提到的所有关键点，coverage ~85%
   - **False Positive / False Negative：** 在 AML 场景下，false negative（漏报）的后果远大于 false positive（误报），所以我们特别关注 recall

5. **与 junior investigator 的对比：**
   - 同样的 100 个 case，junior investigator 的 accuracy 约 70-75%
   - Agent 的 80% "on par with senior investigators" 指的是 senior 的 accuracy 约 85-90%，agent 已接近这个水平

6. **Stratified Evaluation（按 case 复杂度分层，补充）：**
   - 按 case complexity 分层后的 accuracy：
     - **Simple case（clear evidence）：** ~95%（agent 和 senior investigator 表现接近）
     - **Medium case（需要综合多个信号）：** ~80%
     - **Complex case（边界 case、多实体关系、需要主观判断）：** ~60%
   - **按 case type 分层：** structuring pattern 的 accuracy 最高（~90%），因为规则明确；layering 和 crypto-related 的 accuracy 较低（~65%），因为 pattern 更复杂
   - Stratified view 比 overall accuracy 更有 actionable insight——可以针对性地优化低 accuracy 的 segment

7. **Inter-rater Reliability（Investigator 一致性评估，补充）：**
   - 问题：不同 investigator 对同一个 case 的判断也不完全一致，那 80% accuracy 是和谁的判断比？
   - **Cohen's Kappa 量化一致性：**
     - 计算 investigator vs investigator 的 Kappa（通常 0.65-0.75）
     - 计算 agent vs investigator 的 Kappa（约 0.70）
     - Agent 的一致性已经在 investigator 之间一致性的范围内
   - $\kappa = \frac{p_o - p_e}{1 - p_e}$，其中 $p_o$ = observed agreement，$p_e$ = expected agreement by chance
   - Kappa > 0.6 = substantial agreement，> 0.8 = almost perfect
   - **意义：** 这比 raw accuracy 更有说服力——说明 agent 的判断一致性和人类专家相当

> **Follow-up 提示：** 面试官可能追问 "100 个 test case 够不够？有没有做 bootstrap confidence interval？"、"false negative rate 具体是多少？"、"investigator 对 agent 的 report 改动量大吗？"

---

### Q5: 为什么选 LangChain Deep Agent？和 LangGraph / AutoGen 的区别？

**回答：**

1. **LangChain Deep Agent 是什么：**
   - LangChain 1.0 推出的 multi-agent 框架，核心特点是以 file system 为主要通信工具，built on LangGraph
   - Agent 可以创建文件、读取文件、在文件中记录分析结果，sub-agent 之间通过 file system 共享信息
   - 支持 sub-agent 的并行执行

2. **和其他框架的对比：**

   | 维度 | LangChain Deep Agent | LangGraph（直接用） | AutoGen |
   |------|---------------------|-------------------|---------|
   | **通信方式** | File system | State dict / Message passing | Agent 之间的对话（chat） |
   | **Context 隔离** | 天然隔离——每个 sub-agent 有独立 context，通过 file 交换结果 | 需要手动管理 state 和 subgraph 的 context | Agent 之间对话共享，context 容易膨胀 |
   | **并行支持** | 原生支持 sub-agent 并行 | 支持 parallel node | 支持但需要额外配置 |
   | **适用场景** | 需要 sub-agent 独立工作、产出大量中间结果 | 流程明确的 workflow 编排 | 多角色对话协作 |
   | **生态** | LangChain 生态（prompt template、tool、LLM wrapper） | 同上 | 微软生态 |

3. **为什么选 Deep Agent：**
   - **Context 隔离需求：** 我们的 sub-agent 各自处理大量数据（transaction 分析、profile 数据），如果共享 context 会迅速撑爆 context window
   - **File system 天然适合：** 每个 sub-agent 的输出是一篇分析报告，用 file 存储很自然，也方便 investigator 直接查看中间结果
   - **并行执行：** 数据采集和部分分析任务是独立的，可以并行执行提升速度
   - **Built on LangGraph：** 继承了 state management、checkpoint、human-in-the-loop 等能力

4. **不选 AutoGen 的原因：**
   - AutoGen 更适合 "多个 agent 互相讨论" 的场景（如 coder + reviewer），我们的场景是 "main agent 调度 sub-agent 执行具体任务"，不需要 agent 之间对话
   - AutoGen 的 chat-based 通信方式会导致 context 膨胀

> **Follow-up 提示：** 面试官可能追问 "Deep Agent 有什么 limitation？"、"如果 LangChain 不维护 Deep Agent 了怎么办？"、"file system 通信的性能瓶颈是什么？"

---

### Q6: Sub-agent 之间怎么通信？为什么用 file system？

**回答：**

1. **通信机制：**
   - Sub-agent 不直接通信——所有通信经过 main agent 和 file system
   - 工作流程：main agent 给 sub-agent 发 task → sub-agent 执行并将结果写入 file → main agent 读取 file 获取结果 → 根据需要将结果作为 input 传给下一个 sub-agent
   - File 按 case ID 和 agent type 组织：`/cases/{case_id}/{agent_type}/report.md`

2. **为什么用 file system 而不是 message passing：**
   - **中间结果量大：** 每个 sub-agent 的分析报告可能有数千 token，如果用 message passing 全部放进 main agent 的 context，几轮就会撑爆 context window
   - **持久化：** file 自动持久化，session 断了可以恢复，也方便 debug 和 audit
   - **选择性读取：** main agent 不需要读取每个 sub-agent 的完整报告，可以只读取 summary 部分或特定 section
   - **Investigator 可查看：** file system 中的中间结果可以直接暴露给 investigator，增加透明度

3. **潜在问题和应对：**
   - **IO 延迟：** 文件读写比 in-memory message passing 慢，但在我们的场景下 IO 不是瓶颈（LLM inference 才是）
   - **文件格式一致性：** 需要约定统一的文件格式（markdown with structured sections），否则 main agent 解析 sub-agent 的输出会出错
   - **并发写入：** 多个 sub-agent 并行时不会写同一个文件，通过目录结构隔离

> **Follow-up 提示：** 面试官可能追问 "如果 sub-agent 的输出格式不一致怎么办？"、"有没有考虑用 database 替代 file system？"、"文件清理策略是什么？"

---

## Part 5: 面试官视角深度问题

---

### 系统设计类

---

#### Q7: Multi-agent 架构选型——为什么不用单个大 agent？

**回答：**

1. **单个 agent 的问题：**
   - **Context window 限制：** 一个 AML case 的完整分析涉及大量数据——账户信息、交易记录、外部数据、SOP 引用、法规条文，单个 agent 的 context 很快就满
   - **任务过于复杂：** 要求一个 agent 同时做数据采集、分析、推理、写 report，prompt 会变得极其复杂，LLM 容易丢失关注点
   - **不可并行：** 单 agent 只能串行处理，无法并行采集数据和做分析

2. **Multi-agent 的优势：**
   - **分治：** 每个 sub-agent 只关注一个 domain，prompt 简洁、任务明确
   - **Context 隔离：** sub-agent 各自管理 context，main agent 只需要处理高层信息
   - **可扩展：** 新增分析能力只需加 sub-agent，不需要改动 main agent
   - **并行加速：** 独立的数据采集和分析任务可以并行

3. **Trade-off：**
   - Multi-agent 引入了额外的复杂性：agent 之间的通信、error propagation、debugging 难度增加
   - Latency 增加：多次 LLM 调用 + file IO
   - 但在我们的场景下，这些 trade-off 是值得的——case 分析的质量和完整性更重要

> **Follow-up 提示：** 面试官可能追问 "sub-agent 的数量是怎么决定的？"、"有没有遇到 sub-agent 之间结果矛盾的情况？"、"main agent 的 planning 逻辑是写死的还是 LLM 决定的？"

---

#### Q8: Latency 优化——端到端的响应时间是多少？怎么改善？

**回答：**

1. **当前 latency：**
   - 完整 case 分析（first run）：5-10 分钟（包含多个 sub-agent 的分析 + report 生成）
   - 交互式追问：15-30 秒（已有分析结果时，只需要 1-2 次 LLM 调用）

2. **Latency 来源分析：**
   - **LLM Inference：** 占比最大，每个 sub-agent 可能有多轮 LLM 调用
   - **Deep Agent Middleware：** tool 调用和 state management 的 overhead，目前正在排查具体瓶颈
   - **数据采集：** 外部 API 调用（LexisNexis 等）有 network latency
   - **File IO：** 占比较小

3. **优化方向：**
   - **Sub-agent 并行：** 已实现——不相互依赖的 sub-agent 并行执行
   - **Streaming：** 向 investigator stream 中间状态（"正在分析 transaction pattern..."），改善感知体验
   - **缓存：** 对相同 case 的重复查询缓存分析结果
   - **Middleware 优化：** 正在排查 Deep Agent 的 middleware overhead
   - **模型选择：** 对简单任务用更快的模型（如 Claude Haiku），复杂推理用更强的模型

> **Follow-up 提示：** 面试官可能追问 "investigator 对 latency 的容忍度是多少？"、"有没有做过 latency budget 分解？"

---

### LLM 工程类

---

#### Q9: Hallucination 在 compliance 场景的风险怎么处理？

**回答：**

1. **风险特殊性：** AML compliance 场景下，hallucination 的后果很严重——如果 agent 编造了一个不存在的 SOP 条款或法规引用，investigator 据此做 decision 可能导致合规违规

2. **防御措施：**
   - **Graph-RAG Grounding：** agent 的 reasoning 必须基于 Graph KB 中的实际节点，不能自由发挥。每个 decision 和 reasoning step 都要引用具体的 SOP section 或法规条文
   - **Source Attribution：** agent 的输出必须附带引用来源（具体是哪个 SOP 的哪个 section），investigator 可以点击验证
   - **Confidence Score：** agent 对不确定的判断给出低 confidence，并明确告知 investigator "这个判断不够确定，建议人工核实"
   - **Analytical Tool 兜底：** 涉及数据计算的部分（如 transaction velocity），由确定性的 tool 计算而非让 LLM 做数学，避免数值 hallucination
   - **Human-in-the-loop：** investigator 始终 review agent 的输出，特别是 decision 和 report，不是自动提交

3. **检测方式：**
   - 检查 agent 引用的 SOP section / 法规条文是否在 Graph KB 中存在
   - 对比 agent 的 reasoning 和 Graph-RAG 返回的原始内容，检查是否有偏离

> **Follow-up 提示：** 面试官可能追问 "有没有实际发生过 hallucination 导致错误 decision 的案例？"、"怎么量化 hallucination rate？"

---

#### Q10: Context Window 管理的具体策略？

**回答：**

1. **问题：** Multi-agent 多轮对话场景下，context 增长很快——每轮对话可能有 tool call（包含完整文件内容）、sub-agent 的分析结果、用户追问等

2. **三层策略：**
   - **Layer 1 — Sub-agent 隔离：** 最有效的一层。每个 sub-agent 的完整 context 不进入 main agent，main agent 只接收 sub-agent 的最终 summary output
   - **Layer 2 — Summarization：** 每累积 ~1M token，用 LLM 对历史对话生成 structured summary，append 到最新的 context 中
   - **Layer 3 — Message Pruning：** 删除信息密度低的历史 message（如 edit file tool call 的完整内容、重复的数据查询结果）

3. **保留策略：**
   - 始终保留：system prompt、最近 20 条 message、summarized history、当前 case 的 key findings
   - 可删除：file edit 的完整内容、data retrieval 的 raw output（已被 sub-agent 分析过的）

> **Follow-up 提示：** 面试官可能追问 "summarization 会不会丢失关键信息？"、"有没有做过 context window 使用率的监控？"

---

### 知识管理类

---

#### Q11: SOP 变更了怎么处理？Graph KB 怎么更新？

**回答：**

1. **更新机制：**
   - SOP 和法规有版本管理，每次变更会触发 Graph KB 的增量更新
   - 更新流程：识别变更的 section → 更新对应的 graph 节点和关系 → 重新 embed 受影响的节点 → 验证更新后的检索质量

2. **增量 vs 全量：**
   - 日常变更（minor revision）：增量更新，只修改受影响的节点
   - 重大变更（regulatory change）：全量重建 graph，因为关系可能大范围变动

3. **验证：**
   - 更新后，用一组 test query 验证检索结果是否正确反映新的 SOP 内容
   - 对比更新前后同一 query 的检索结果，确认变更被正确反映

> **Follow-up 提示：** 面试官可能追问 "SOP 更新的频率是什么？"、"有没有自动检测 SOP 变更的机制？"

---

### 评估类

---

#### Q12: 除了 decision accuracy，还有哪些评估维度？

**回答：**

1. **Report Quality（report 质量）：**
   - Investigator 对 agent 生成的 SAR report 打分（1-5 分，5 = 可直接提交，1 = 需要重写）
   - 评估维度：格式是否正确、evidence 引用是否完整、narrative 是否逻辑清晰、是否符合 SOP 要求
   - 当前平均分 3.8 分——"可以作为草稿直接修改，不需要从头写"

2. **Evidence Coverage（证据覆盖率）：**
   - Agent 找到的 evidence 是否涵盖了 investigator 在 ground truth report 中提到的所有关键点
   - 当前 coverage ~85%

3. **Time Reduction（时间减少）：**
   - 使用 agent 前后完成同类型 case 的时间对比
   - 当前 ~85% reduction

4. **Investigator Satisfaction（用户满意度）：**
   - 定期收集 investigator 的定性反馈
   - 重点关注：agent 的建议是否有帮助、交互体验是否顺畅、是否信任 agent 的分析

5. **Error Analysis（错误分析）：**
   - 对 agent 判断错误的 case 做 root cause 分析
   - 分类：evidence 遗漏、reasoning 错误、SOP 理解偏差、hallucination

6. **Quantitative Agreement Metrics（量化一致性指标，补充）：**
   - **Cohen's Kappa（双评估者一致性）：** 用于 agent vs single investigator 的一致性量化
     - $\kappa = \frac{p_o - p_e}{1 - p_e}$
     - Agent vs senior investigator 的 Kappa ≈ 0.70（substantial agreement）
   - **Fleiss' Kappa（多评估者一致性）：** 当有多个 investigator + agent 对同一组 case 做判断时
     - 衡量 "超出随机一致性" 的部分
     - 用于回答 "agent 的判断和 investigator 群体的一致性如何？"
   - **Why Kappa > Raw Agreement：** Raw agreement（百分比一致）没有扣除 "碰巧一致" 的部分。如果 90% 的 case 都是 no-file，两个随机评估者也有 81% 的 raw agreement。Kappa 修正了这个 base rate 效应
   - **实际应用：** 定期随机抽取 case，让多个 investigator 和 agent 独立判断，计算 Kappa 趋势。如果 agent 的 Kappa 随时间下降，说明可能有 knowledge drift 或 model degradation

> **Follow-up 提示：** 面试官可能追问 "这些 metric 的优先级怎么排？"、"有没有做 A/B testing？"

---

### 业务类

---

#### Q13: Investigator 实际使用这个系统的模式是什么？

**回答：**

1. **典型使用流程：**
   - Investigator 打开一个新 case → 启动 agent → agent 自动做数据采集和初步分析 → investigator review 分析结果 → 和 agent 交互追问细节 → agent 生成 SAR report 草稿 → investigator 修改后提交

2. **最受欢迎的功能：**
   - SAR report 自动生成（节省最多时间的环节）
   - Transaction pattern 分析（agent 能快速识别 investigator 可能遗漏的 pattern）
   - SOP 查询（新 investigator 特别依赖这个功能）

3. **Adoption 挑战：**
   - **信任建立：** Senior investigator 初期不信任 agent 的 decision，需要 "并行期"——agent 和 investigator 同时做，对比结果
   - **使用习惯：** 部分 investigator 习惯手动流程，需要培训和渐进式推广
   - **Edge Case 预期管理：** 让 investigator 理解 agent 在边界 case 上的局限性，避免过度依赖

> **Follow-up 提示：** 面试官可能追问 "investigator 反馈最多的 complaint 是什么？"、"adoption rate 目前是多少？"

---

### 工程挑战类

---

#### Q14: Multi-agent 系统怎么 debug？

**回答：**

1. **挑战：** 输出质量有问题时，需要定位是哪个 sub-agent 的问题——是数据采集不完整、分析逻辑错误、还是 report 生成时 hallucinate 了

2. **Debug 工具链：**
   - **Trace Logging：** 记录每个 sub-agent 的完整 input/output，包括中间的 tool call 和 LLM response
   - **File System 作为 snapshot：** sub-agent 的所有中间结果都在 file system 中，可以逐步检查
   - **LLM-as-Judge：** 用 LLM 对 sub-agent 的输出做自动化质量评估，标记异常

3. **常见问题模式：**
   - **Cascading Error：** 上游 sub-agent 的错误 propagate 到下游——如数据采集遗漏导致分析结论错误
   - **Planning Error：** Main agent 的 planning 漏掉了某个必要的分析步骤
   - **Context Lost：** Summarization 过程中丢失了关键信息

> **Follow-up 提示：** 面试官可能追问 "有没有做 automated regression testing？"、"发现 bug 后怎么 reproduce？"

---

#### Q15: 动态 Tool 生成的安全性怎么保障？

**回答：**

1. **场景：** Agent 根据用户需求现场生成新的 analytical tool（Python 代码），执行分析后返回结果

2. **安全风险：**
   - 生成的代码可能有逻辑错误（错误的计算、遗漏 edge case）
   - 恶意代码注入（虽然 input 来自 investigator 而非外部用户，风险较低）
   - 资源滥用（无限循环、内存泄漏）

3. **保障措施：**
   - **Sandbox 执行：** 生成的代码在隔离的 sandbox 环境中运行，限制 file system access、network access、execution time
   - **代码 Review 步骤：** 生成代码后先展示给 investigator 确认，不自动执行
   - **参考已有 Tool：** 生成新 tool 时，以现有的 verified analytical tool 为模板和参考，降低出错概率
   - **输出验证：** 对生成 tool 的输出做 sanity check（数据类型、数值范围、是否有 null）

4. **当前状态：** 这个功能还在研发中，安全性是最大的 concern

> **Follow-up 提示：** 面试官可能追问 "sandbox 具体用什么技术实现？"、"怎么验证生成的代码逻辑是正确的？"

---

#### Q16: 数据隐私怎么处理？AML 数据很敏感

**回答：**

1. **数据敏感性：** AML case 数据包含客户个人信息、交易记录、调查结论，属于高敏感数据

2. **隐私保障：**
   - **LLM 部署：** 使用企业级 LLM 部署（不是公开 API），数据不离开公司网络
   - **Data Residency：** 所有 case 数据和分析结果存储在公司内部基础设施
   - **Access Control：** 基于角色的访问控制，investigator 只能访问自己负责的 case
   - **Audit Trail：** 完整的操作日志——谁看了什么数据、agent 做了什么分析、最终 decision 是什么

3. **合规考量：**
   - Agent 的输出（包括 SAR report）需要经过 investigator review 才能提交，确保 human oversight
   - 定期做 compliance review，确保 agent 的行为符合内部 data governance 政策

> **Follow-up 提示：** 面试官可能追问 "有没有做 PII masking？"、"LLM 的 training data 会不会包含客户数据？"

---

## Part 6: 如何设计一个 LLM Agent 应用（How to Design an LLM Agent Application）

---

本章总结了从 AML 项目中提炼出的 **通用设计方法论**。先讲设计原则和实践框架，再以面试 Q&A 的形式覆盖高频问题。所有原则都用 AML 项目作为 running example，展示的是 battle-tested 的经验而非泛泛而谈的理论。

---

### 6.1 设计方法论（Design Methodology）

---

#### 6.1.1 什么时候该用 Agent（Problem Framing）

**Agent Decision Ladder — 5 个层级：**

在决定"要不要用 agent"之前，先判断你的任务在下面这个阶梯的哪一层。**核心原则：永远从最低的足够层级开始，不要跳级。**

| 层级 | 方案 | 特征 | 适用场景 |
|------|------|------|---------|
| Level 1 | 确定性代码（Deterministic Code） | 输入输出固定，逻辑完全可枚举 | ETL pipeline、规则引擎、数据转换 |
| Level 2 | 单次 LLM 调用（Single LLM Call） | 需要语言理解/生成，但只需一步 | 文本分类、摘要、翻译、简单问答 |
| Level 3 | LLM + Tools | 需要 LLM 调用外部工具，但步骤固定 | RAG 检索 + 回答、结构化数据提取 |
| Level 4 | Agent with Planning | 步骤不固定，需要根据中间结果动态决策 | 复杂调查、代码生成与调试、多步骤分析 |
| Level 5 | Multi-Agent System | 任务可分解为多个独立子任务，单 agent 的 context 或能力不够 | 端到端 AML 调查、复杂软件工程、多角色协作 |

**可行性清单（Feasibility Checklist）：**

在决定用 agent 之前，回答以下 4 个问题：

1. **可分解性：** 任务是否可以分解为多个子步骤？
2. **变化性：** 不同 input 是否需要不同的处理路径和步骤顺序？
3. **知识可获取性：** 是否有足够的领域知识（SOP、文档、专家）来指导 agent？
4. **可评估性：** 是否能定义明确的评估指标来衡量 agent 的表现？

**"What Varies" Test：**

- 如果每个 case 都走 **完全相同的固定步骤** → 用 pipeline（Level 3）
- 如果 **步骤、工具、顺序随 input 变化** → 需要 agent（Level 4+）

**AML 为什么需要 Level 5：**

- Case type 多样（structuring、layering、crypto-related），每种类型需要不同的分析路径
- Investigator 的追问不可预测，agent 需要动态规划回答策略
- 数据源的选择取决于初步分析结果——不可能预先固定 pipeline
- 单个 agent 的 context window 无法容纳完整 case 的所有分析数据

**常见陷阱：**

- **Agent as a Hammer：** 不是所有问题都需要 agent。如果固定 pipeline 能解决，就别用 agent——更简单、更可靠、更便宜
- **低估评估难度：** Agent 不像传统 ML 有标准 benchmark，评估体系需要从零建立，这个工作量经常被低估
- **忽视 Human-in-the-loop：** 在高风险场景（合规、医疗、金融），纯自动化是不可接受的，必须从一开始就设计人工介入点

---

#### 6.1.2 领域知识工程（Domain Knowledge Engineering）

**知识获取问题（The Knowledge Acquisition Problem）：**

做 agent 的 DS/engineer 通常不具备领域专业知识。这是 agent 项目最容易被低估的挑战——不是技术问题，而是 **知识鸿沟** 问题。三种策略：

1. **SME Interview + Structured Extraction：**
   - 和领域专家做结构化访谈，不是泛泛聊，而是用具体 case 走读的方式提取决策逻辑
   - 输出物：决策树、条件-行动映射、关键判断标准
   - 技巧：让 SME 对着真实 case 边做边讲，比抽象描述有效得多

2. **Historical Data Reverse Engineering：**
   - 从历史决策记录中反向推理专家的思考过程
   - **AML 实践：** 分析历史 SAR report 和对应 case 数据，推理出 investigator 在 review 时关注什么方向、用什么计算逻辑做判断
   - 这些推理结果直接决定了 `sub-agent` 和 analytical tool 的具体实现

3. **Document Mining：**
   - 系统性地挖掘 SOP、法规、内部文档、培训材料
   - 关键是 **建立文档之间的关系**，而不是简单地 chunk 和 embed

**知识表示选择（Knowledge Representation）：**

| 表示方式 | 优势 | 劣势 | 适用场景 |
|---------|------|------|---------|
| **Vector DB** | 语义搜索强，建设成本低 | 无法表达关系，缺少结构化推理 | 知识碎片化、无明显层级关系 |
| **Knowledge Graph** | 关系明确，支持多跳推理 | 建设和维护成本高，需要 schema 设计 | 知识有层级/引用关系（法规、SOP） |
| **Structured Prompts** | 简单直接，无额外基础设施 | 受 context window 限制，不可扩展 | 知识量小且相对固定的场景 |
| **Code / Rules** | 确定性强，可测试 | 不灵活，难以处理模糊逻辑 | 可枚举的业务规则 |
| **Hybrid** | 综合优势 | 系统复杂度高 | 大多数实际项目 |

**知识完备性设计：**

知识不可能 100% 覆盖所有场景，必须设计 **graceful degradation**：

- 置信度评分（confidence score）：agent 对自身判断的确定程度
- 明确的"不确定"输出：agent 应该能说 "我不确定，建议人工核实"
- 人工升级路径（human escalation）：当 agent 超出能力边界时，自动触发人工介入

**AML 实践：**

- 用 `Graph-RAG` 表示 SOP 和法规的层级引用关系（法规 → SOP 条款 → condition → action）
- 通过 SAR report reverse engineering 推理 investigator 的决策逻辑，这些逻辑被编码为 `sub-agent` 的 prompt 和 tool 设计
- Hybrid 策略：结构化关系走 graph traversal，语义检索走 vector search

**常见陷阱：**

- **先建 RAG 再理流程：** 很多团队上来就 chunk 文档建 vector DB，但还没搞清楚决策流程是什么。应该先 map 完决策过程，再决定用什么知识表示
- **过度依赖 vector search 处理结构化知识：** 如果知识之间有明确的层级和引用关系，纯 vector search 会丢失这些结构信息
- **不和 SME 验证：** 从文档中提取的知识未必反映实际操作，必须让 SME review

---

#### 6.1.3 架构设计（Architecture Design）

**Single vs Multi-Agent 决策框架：**

不要默认使用 multi-agent。只有在以下三个驱动力之一明确存在时才拆分：

1. **Context Window 压力：** 单个 agent 的 context 无法容纳所有必要信息
2. **并行需求：** 多个子任务可以独立并行执行，串行太慢
3. **独立优化需求：** 不同子任务需要不同的 prompt 策略、模型选择、或评估标准

**通信模式对比：**

| 通信模式 | 优势 | 劣势 | 适用场景 |
|---------|------|------|---------|
| **Message Passing** | 低延迟，实时交互 | Context 膨胀快，agent 之间耦合 | Agent 之间需要频繁、短小的交互 |
| **File System** | Context 隔离，结果可追溯，支持大量中间输出 | IO 开销，需要约定文件格式 | 中间结果量大、需要持久化和审计 |
| **Chat-based** | 自然语言交互，灵活 | Context 增长最快，难以控制信息流 | 多角色讨论、头脑风暴式协作 |

**编排模式（Orchestration Patterns）：**

- **Hierarchical（层级式）：** Main agent 指挥 sub-agents，统一 planning 和决策。适合有明确中心控制需求的场景
- **Pipeline（流水线式）：** Agent 按固定顺序传递结果。适合步骤明确且依赖线性的场景
- **Blackboard（黑板式）：** 所有 agent 共享一个中心化的 state，各自独立工作并更新。适合松耦合的协作场景

**Context Management 作为架构决策：**

Context management 不是事后优化，而是 **架构设计的核心组成部分**。三层策略是一个可复用的 pattern：

1. **Sub-agent 隔离：** 每个 sub-agent 有独立 context，main agent 只接收最终 output
2. **Summarization：** 定期对历史对话做结构化总结
3. **Message Pruning：** 删除信息密度低的历史内容（如 file edit 的完整内容）

**AML 实践：**

- 选择 Hierarchical + File System 的组合
- **原因 1：** sub-agent 的分析报告可能有数千 token（transaction pattern 分析、profile 数据），通过 file system 传递避免 context 膨胀
- **原因 2：** Investigator 需要看到中间结果（transparency），file system 天然支持
- **原因 3：** 合规要求完整的 audit trail，file system 自带持久化

**常见陷阱：**

- **Premature Multi-Agent Split：** 没有遇到 context 或并行问题就拆 agent，增加了不必要的复杂性。先用单 agent 做 MVP，遇到瓶颈再拆
- **忽视 Context Management 直到危机：** 很多团队在 context window 被撑爆后才开始考虑 context management，这时候改架构的成本很高。应该从一开始就设计
- **Unstructured Agent Communication：** Agent 之间的通信没有明确的格式约定，导致解析错误和信息丢失

---

#### 6.1.4 工具与能力设计（Tool & Capability Design）

**工具设计光谱（Tool Design Spectrum）：**

| 工具类型 | 特征 | 适用场景 | 示例 |
|---------|------|---------|------|
| **确定性工具（Deterministic）** | 输入输出确定，逻辑可验证 | 数据查询、数学计算、格式转换 | Transaction velocity 计算 |
| **LLM 辅助工具（LLM-assisted）** | 需要 LLM 做理解/判断 | 文本分析、模式识别、自然语言处理 | Memo 关键词分析、邮箱乱码检测 |
| **混合工具（Hybrid）** | 确定性逻辑 + LLM 判断 | 先计算再解读、先提取再分析 | 先算 transaction stats，再让 LLM 判断是否异常 |

**核心原则：尽可能把逻辑推到确定性工具中。** LLM 擅长理解和推理，不擅长精确计算和数据处理。

**工具接口设计：**

1. **清晰的类型化契约：** 每个 tool 的 input/output 要有明确的类型定义，不要用模糊的 "data" 或 "result"
2. **Tool Description 就是给 LLM 看的文档：** LLM 根据 tool description 决定何时、如何调用工具。Description 写得不好 = agent 不会正确使用工具
3. **粒度平衡：** 太粗（一个 tool 做太多事）→ LLM 不知道什么时候该用；太细（tool 太多）→ LLM 选择困难。经验法则：一个 tool 做一件事，tool 总数控制在 10-20 个

**工具发现策略：**

- **Static Set（静态集合）：** 所有 tool 预定义好，agent 只能从中选择。最简单、最可控
- **Dynamic Loading（动态加载）：** 根据任务类型动态加载相关 tool 子集。减轻 LLM 的选择负担
- **Dynamic Generation（动态生成）：** Agent 根据需求现场生成新 tool。最灵活但安全风险最大

**AML 实践：**

- Transaction velocity、density 等 **数值计算全部用确定性 tool**，绝不让 LLM 做数学
- Memo keyword 分析、email 乱码检测等 **需要语言理解的任务用 LLM-assisted tool**
- 正在研发 **dynamic tool generation**：根据用户需求，参考已有 tool 模板，现场生成新的 analytical tool
  - 最大挑战：sandbox 执行环境、代码正确性验证、安全性保障

**常见陷阱：**

- **让 LLM 做数学：** LLM 做加减乘除都可能出错。任何涉及精确计算的逻辑，都应该用 deterministic tool
- **Tool Description 写得模糊：** "分析交易数据" 不如 "计算指定时间窗口内的交易频率、平均金额和金额标准差"
- **不独立测试 Tool：** Tool 本身有 bug，agent 怎么调用都不会对。每个 tool 应该有独立的 unit test

---

#### 6.1.5 评估策略（Evaluation Strategy）

**评估层级（Evaluation Hierarchy）：**

| 层级 | 评估什么 | 方法 | 示例 |
|------|---------|------|------|
| **Component-level** | 单个 sub-agent 或 tool 的质量 | Unit test、ground truth 对比 | Decision Agent 的 accuracy |
| **Task-level** | 端到端完成一个任务的质量 | Test case suite、多维度 metric | 一个完整 case 的分析质量 |
| **User-level** | 用户实际使用中的效果 | 时间减少、满意度、adoption rate | Investigator 使用前后的效率对比 |

**核心原则：先 component-level，再 task-level，最后 user-level。** 如果 component 有问题，task-level 的结果没有意义。

**Ground Truth 构建方法：**

1. **历史决策（Historical Decisions）：** 用过去的专家决策作为 ground truth。最直接但可能包含错误
2. **专家标注（Expert Annotation）：** 请 SME 对 agent 的输出做标注。高质量但成本高、速度慢
3. **LLM-as-Judge：** 用另一个 LLM 对 agent 的输出做评分。成本低、可扩展，但需要验证 judge 本身的准确性
4. **自动化指标（Automated Metrics）：** 可量化的硬指标（如覆盖率、格式正确率）。客观但无法覆盖所有质量维度

**多维度评估模板：**

单一指标在不对称成本的领域是不够的。评估模板应包括：

- **Decision Accuracy：** 核心决策的正确率
- **Output Quality：** 输出内容的质量（格式、完整性、逻辑性）
- **Evidence Coverage：** 是否找到了所有关键证据
- **Time Reduction：** 相比人工的效率提升
- **Domain-specific Metrics：** 根据领域特殊性定义，如 AML 中的 false negative rate（漏报率比误报率的后果严重得多）

**迭代循环（Iteration Loop）：**

```
Evaluate → Error Analysis → Categorize Failures → Targeted Fixes → Re-evaluate
```

关键是 **error analysis 和 failure categorization**——不是看到 accuracy 不够就去调 prompt，而是先分析错在哪类 case、错的原因是什么，然后针对性修复。

**AML 实践：**

- 100-case 评估集，覆盖不同 case type 和复杂度
- 多维度 dashboard：decision accuracy（80%）、report quality（3.8/5）、evidence coverage（85%）、time reduction（85%）
- 和 junior investigator 做对比（junior 70-75% vs agent 80%），证明 agent 已接近 senior 水平
- **特别关注 false negative rate：** 在合规场景，漏报一个可疑 case 的后果远大于误报

**常见陷阱：**

- **只评估 Happy Path：** 只用 "好做的 case" 测试，忽略边界 case 和异常场景。评估集必须包含 hard case
- **对称成本场景用单一 Metric：** 在 AML 中，false negative 和 false positive 的成本完全不同。只看 accuracy 会掩盖关键问题
- **不做 Error Analysis：** 只看总体 accuracy 数字，不分析错误的分布和原因，无法做针对性优化

---

#### 6.1.6 上线考量（Production Considerations）

**Latency Budget 分解：**

把端到端延迟分解到每个组件，为每个组件设定延迟预算：

| 组件 | 延迟来源 | 优化方向 |
|------|---------|---------|
| LLM Inference | 模型推理时间，通常是最大瓶颈 | 模型分级（简单任务用小模型）、并行调用 |
| Tool Execution | 外部 API 调用、数据库查询 | 缓存、并行、超时控制 |
| Middleware | 框架 overhead（state management、checkpoint） | Profiling、精简不必要的中间件 |
| File IO | 中间结果读写 | 通常不是瓶颈，除非文件很大 |

**成本管理：**

- **Per-task 成本追踪：** 记录每个 task 消耗的 token 数和对应的费用
- **模型分级（Model Tiering）：** 简单任务用小/快模型（如 `Claude Haiku`），复杂推理用强模型（如 `Claude Sonnet`/`Opus`）
- **缓存策略：** 对相同 case 的重复查询缓存分析结果，避免重复 LLM 调用

**可靠性设计：**

- **重试策略：** API 失败时的重试逻辑（exponential backoff），区分可重试错误和不可重试错误
- **Fallback 行为：** 当某个 sub-agent 失败时的降级策略——是跳过、用默认值、还是报错给用户
- **级联失败预防：** 上游 sub-agent 的错误不应导致整个系统崩溃。设置超时、熔断、错误隔离

**安全与合规：**

- **Human-in-the-loop 设计：** 明确哪些决策必须人工确认，哪些可以自动执行
- **置信度阈值：** Agent 输出的 confidence 低于阈值时，自动触发人工 review
- **Audit Trail：** 完整记录 agent 的每一步决策、使用的数据、引用的知识——合规场景的硬性要求
- **数据隐私：** 企业级 LLM 部署、data residency、基于角色的访问控制（RBAC）

**可观测性（Observability）：**

- **监控指标：** 延迟、错误率、token 使用量、决策分布、用户反馈
- **Trace 可视化：** 对每次 agent 执行生成完整 trace，方便 debug 和性能分析
- **异常检测：** 监控决策分布的变化——如果 agent 突然开始大量给出某个 decision，可能有问题

**AML 实践：**

- 完整 case 分析 5-10 分钟（首次），交互式追问 15-30 秒
- 使用企业级 LLM 部署，数据不离开公司网络
- 所有 decision 必须经过 investigator review 后才能提交——合规场景的 mandatory human-in-the-loop
- 完整 audit trail：谁看了什么数据、agent 做了什么分析、最终 decision 是什么

---

#### 6.1.7 优化与演进（Optimization & Evolution）

**Prompt Optimization 作为系统工程：**

Prompt optimization 不是 "试几个措辞看哪个好"，而是一个 **可系统化的工程流程**。三阶段 pipeline：

1. **GEPA 粗搜索：** 用遗传算法探索 prompt 的结构空间（instruction 措辞、reasoning 步骤顺序、evidence 组织方式），快速筛选 promising 的 prompt template
2. **MIPROv2 Few-shot 优化：** 在 DSPy 框架下，用 Bayesian 方法优化 few-shot example 的选择——从历史数据中选最有代表性的示例
3. **TextGrad 精调：** 把 LLM 的文本输出当作 "可微分变量"，用 LLM 生成 "梯度"（反馈），迭代优化 prompt 的关键措辞

所有步骤在 DSPy 框架下统一管理 signature 和 evaluation。

**Multi-Agent 优化挑战：**

优化单个 sub-agent 的 prompt 不等于优化整个系统——Local Optimum ≠ Global Optimum。一个 sub-agent 的 prompt 改变可能影响下游 sub-agent 的表现。

三个探索方向：

1. **Auto Prompt Optimization：** 从单个 sub-agent 扩展到整个 agent system，挑战在于搜索空间爆炸
2. **Agentic Context Engineering：** 用 `LLM-as-judge` 对每次完整的 multi-agent trace 做 reflection，将结果存入外置 workbook，agent 运行时导入。这种 "物理外挂" 方式对原有代码改动最小
3. **AgentLightening（微软）：** 把 GRPO 应用到 multi-step agent，通过封装层获取 trace 数据，结合 reward function 做优化。瓶颈：需要大量 trace 数据

**持续改进循环：**

- **用户反馈整合：** 收集 investigator 的使用反馈，转化为具体的优化 action item
- **Error-driven Evolution：** 每次发现新的错误模式，分析 root cause，更新对应的 prompt、tool 或知识层
- **知识层更新：** SOP 和法规变更时，及时更新 Graph KB 并验证检索质量

**知道什么时候该停：**

- **Diminishing Returns：** Prompt optimization 到一定程度后收益递减——如果 accuracy 从 78% 到 80% 花了和从 60% 到 78% 一样的时间，该换方向了
- **"Last 20%" 问题：** 最后的 20% 性能提升通常不是靠调 prompt 能解决的，需要不同的方法——更好的知识、更好的 tool、或者接受 human-in-the-loop 处理边界 case

---

### 6.2 面试问题（Interview Q&A）

---

#### Q17: 如果让你为一个新领域设计 agent 系统，你会怎么做？

**回答：**

1. **Step 1 — Problem Framing（判断是否需要 agent）：**
   - 用 Agent Decision Ladder 评估：任务的变化性是否足以需要 agent？固定 pipeline 能否解决？
   - 用 "What Varies" Test：如果步骤、工具、顺序随 input 变化，才需要 agent
   - 跑通 Feasibility Checklist：可分解性、变化性、知识可获取性、可评估性

2. **Step 2 — Domain Knowledge Acquisition（获取领域知识）：**
   - 和 SME 做结构化访谈，用具体 case 走读的方式提取决策逻辑
   - 分析历史数据和决策记录，反向推理专家思维过程
   - 挖掘 SOP、法规、内部文档，建立知识之间的关系
   - 输出物：决策流程图、条件-行动映射、关键判断标准

3. **Step 3 — Architecture Design（架构设计）：**
   - 先用 single agent + static tools 做 MVP，验证核心流程可行
   - 根据 context pressure、并行需求、独立优化需求决定是否拆 multi-agent
   - 选择通信模式和编排模式
   - 从一开始就设计 context management 策略

4. **Step 4 — Tool & Capability Design（工具设计）：**
   - 把尽可能多的逻辑推入确定性工具
   - 为每个 tool 写清晰的 description 和类型化接口
   - 独立测试每个 tool

5. **Step 5 — Evaluation Infrastructure（评估基础设施）：**
   - 在写 agent 代码之前，先建评估体系
   - 构建 test case suite，定义多维度 metric
   - 建立迭代循环：evaluate → error analysis → targeted fixes → re-evaluate

6. **Step 6 — Iterate & Optimize（迭代优化）：**
   - 从 component-level 到 task-level 到 user-level 逐层优化
   - 用 prompt optimization pipeline 系统化提升质量
   - 关注 diminishing returns，知道什么时候该换方向

> **Follow-up 提示：** 面试官可能追问 "每个 step 大概花多长时间？"、"哪个 step 最容易被低估？"（答案：Step 2 的知识获取和 Step 5 的评估基础设施）

---

#### Q18: 怎么从非技术 SME 那里获取领域知识？

**回答：**

1. **核心挑战：** SME（Subject Matter Expert）和 DS/engineer 之间存在知识鸿沟。SME 知道 "怎么做" 但说不清楚具体逻辑，engineer 不了解领域但需要把逻辑编码成 prompt 和 tool

2. **Case 走读法（Case Walkthrough）：**
   - 不要抽象地问 "你是怎么判断的"，而是拿一个具体 case 让 SME 边做边讲
   - 记录每一步：看了什么数据、做了什么判断、判断依据是什么
   - 多走几个不同类型的 case，识别共性和变化性
   - 特别关注 SME 说 "这个要看情况" 的地方——那就是 agent 需要 planning 能力的地方

3. **Historical Data Reverse Engineering：**
   - 当 SME 的时间有限（通常如此），用历史数据做补充
   - **AML 实践：** 分析历史 SAR report（输出）和对应的 case 数据（输入），推理出从输入到输出的决策路径
   - 优势：不需要 SME 的时间，可以大规模分析
   - 劣势：只能看到 "做了什么"，看不到 "为什么这样做"，需要和 SME 做验证

4. **Structured Output：**
   - 将获取的知识整理为结构化形式：决策树、条件-行动表、检查清单
   - 每个条目标注来源（哪个 SME、哪个文档、哪个历史案例）
   - 必须让 SME 对最终输出做 review，确认逻辑正确

5. **迭代验证：**
   - 知识获取不是一次性的。Agent 开发过程中持续和 SME 验证——让 SME 看 agent 的输出，指出哪里不对
   - 每次验证都是新的知识获取机会

> **Follow-up 提示：** 面试官可能追问 "SME 不配合怎么办？"、"怎么处理不同 SME 之间意见不一致？"、"知识获取的文档用什么格式？"

---

#### Q19: 什么时候选 Knowledge Graph，什么时候选 Vector DB？

**回答：**

1. **关系结构测试（Relational Structure Test）：**
   - 问自己一个问题："知识之间是否有明确的、重要的关系？"
   - 如果答案是 yes → Knowledge Graph 是强候选
   - 如果知识是扁平的、碎片化的 → Vector DB 足够

2. **Knowledge Graph 适用场景：**
   - 知识有 **层级关系**（法规 → SOP → condition → action）
   - 查询需要 **多跳推理**（"从这个 condition 出发，关联的 action 和法律依据是什么？"）
   - 知识之间有 **引用关系**，丢失引用会导致推理不完整
   - 更新时需要 **精确定位** 受影响的节点和关系

3. **Vector DB 适用场景：**
   - 知识是 **语义相关但结构松散** 的文本（如 FAQ、文章、对话记录）
   - 查询是 **模糊的语义搜索**（"有没有关于 X 的内容？"）
   - 建设和维护成本需要控制在较低水平
   - 知识更新频繁但结构不变

4. **Hybrid Approach（大多数实际场景的答案）：**
   - 结构化关系用 Knowledge Graph 表示和检索
   - 非结构化内容用 Vector DB 做语义搜索
   - **AML 实践：** Graph 处理 SOP/法规的层级结构，Vector 处理 guidance notes 和 best practices 的语义检索。先用 vector search 定位相关 Section 节点，再用 graph traversal 展开关联的 Condition → Action 链路

5. **实际考量：**
   - Knowledge Graph 的 **建设成本高**：需要 schema 设计、数据清洗、关系抽取
   - Knowledge Graph 的 **维护成本高**：schema 变更影响面大
   - 如果不确定，先从 Vector DB 开始，发现关系推理需求明确后再引入 Graph

> **Follow-up 提示：** 面试官可能追问 "Graph 的 schema 怎么设计的？"、"Graph 数据怎么维护和更新？"、"有没有用过 LLM 自动抽取关系？"

---

#### Q20: 怎么决定 sub-agent 的粒度？多少个算太多？

**回答：**

1. **核心原则：Start with One, Split on Concrete Problems**
   - 不要一开始就设计 10 个 sub-agent。先用 1 个 agent 做 MVP
   - 当遇到具体问题时才拆分：context window 不够了、某类任务需要独立优化、需要并行加速

2. **拆分信号：**
   - **Context Window 溢出：** 单个 agent 的 context 被中间结果撑满 → 拆出数据密集型的子任务
   - **Prompt 过于复杂：** 一个 prompt 要指导 agent 做 5 种不同类型的分析 → 拆出每种分析为独立 sub-agent
   - **评估需要独立化：** 不同子任务的评估标准完全不同 → 拆出来可以独立优化和评估
   - **并行加速：** 多个子任务可以独立并行执行 → 拆出来并行跑

3. **每个 Sub-agent 的责任应该是可测试的：**
   - 如果你无法为一个 sub-agent 单独定义 input/output 和评估标准，它的粒度可能不对
   - 好的粒度：Transaction Velocity Agent（输入：交易记录 → 输出：频率和金额的统计分析）
   - 不好的粒度："分析 Agent"（输入：case data → 输出："分析结果"）——太模糊

4. **多少算太多？**
   - **经验法则：** 如果 main agent 需要在 10+ 个 sub-agent 中做路由选择，LLM 的选择准确率会下降
   - 解决方案：按类别分组，main agent 先选类别，再选类别内的 sub-agent
   - **AML 实践：** 我们有三类 sub-agent（数据采集、分析、报告/知识），每类 3-5 个，总共约 10 个。Main agent 先判断需要哪类能力，再调度具体的 sub-agent

5. **常见反模式：**
   - 一个 sub-agent 只做一次 LLM 调用就结束 → 粒度太细，overhead 大于收益
   - 一个 sub-agent 负责太多不相关的任务 → 粒度太粗，和单 agent 没区别

> **Follow-up 提示：** 面试官可能追问 "sub-agent 之间有依赖怎么处理？"、"有没有遇到拆多了又合回去的情况？"、"新增一个 sub-agent 的流程是什么？"

---

#### Q21: 确定性工具 vs LLM 工具——怎么决定？

**回答：**

1. **决策框架 — Tool Design Spectrum：**
   - 问两个问题：
     - **输入输出是否确定？** 如果 yes → 确定性工具
     - **是否需要语言理解或模糊推理？** 如果 yes → LLM 工具
   - 如果两者都需要 → Hybrid 工具（先确定性处理，再 LLM 判断）

2. **确定性工具的场景：**
   - 所有数学计算（金额汇总、频率统计、标准差计算）
   - 数据库查询和数据提取
   - 格式转换和数据清洗
   - 规则匹配（正则表达式、阈值判断）
   - **黄金法则：永远不要让 LLM 做数学。** LLM 连简单的加减乘除都可能出错

3. **LLM 工具的场景：**
   - 自然语言理解（判断一段 memo 是否包含可疑内容）
   - 模式识别需要 "常识"（判断一个 email 地址是否像随机生成的）
   - 需要跨多个信息源做综合判断
   - 输出是非结构化的文本（report 生成、reasoning 解释）

4. **Hybrid 工具的设计模式：**
   - **先算后判：** 确定性工具算出统计指标 → LLM 根据指标和上下文判断是否异常
   - **先提后分：** 确定性工具提取结构化数据 → LLM 分析数据含义
   - **AML 示例：** 先用 deterministic tool 计算 transaction velocity（30 天内交易次数、平均金额、峰值），再让 LLM 结合 case context 判断这个 velocity 是否异常

5. **测试策略不同：**
   - 确定性工具：标准 unit test，input/output 验证
   - LLM 工具：需要 evaluation set + 多维度 metric，本质上是在测 LLM 的判断质量
   - 两种工具都应该 **独立于 agent 系统单独测试**

> **Follow-up 提示：** 面试官可能追问 "有没有一开始用 LLM 后来换成确定性工具的例子？"、"LLM 工具的 latency 怎么控制？"、"hybrid 工具中 LLM 判断出错怎么 fallback？"

---

#### Q22: 没有明确 ground truth 的时候，怎么评估 agent？

**回答：**

1. **构建 Proxy Ground Truth：**
   - **历史决策：** 用过去专家做过的决策作为 ground truth。不完美（专家也可能犯错），但通常是最实际的起点
   - **专家标注：** 请 SME 对 agent 的输出做标注。可以是完整标注（file/no-file）或相对比较（"A 的分析比 B 好"）
   - **AML 实践：** 100 个历史 closed case，每个 case 有 investigator 决策 + senior reviewer 确认，作为双重验证的 ground truth

2. **多维度评估（避免单一 Metric 盲区）：**
   - Decision accuracy 只是一个维度。完整评估应包括：
     - **Output Quality：** 输出内容的格式、完整性、逻辑性（可以用 LLM-as-Judge 或人工评分）
     - **Evidence Coverage：** 是否找到了所有关键证据（可以和 ground truth report 对比）
     - **Domain-specific Metrics：** 如 AML 中特别关注 false negative rate
   - 不同维度的权重取决于业务需求——合规场景中 recall 可能比 precision 重要

3. **LLM-as-Judge：**
   - 用另一个 LLM 对 agent 的输出做质量评估
   - 优势：成本低、可扩展、可以评估 "难以量化" 的质量维度（如 reasoning 是否逻辑自洽）
   - 劣势：judge 本身可能不准确。需要用一部分人工标注数据验证 judge 的一致性
   - 最佳实践：用 LLM-as-Judge 做初筛，对边界 case 做人工复核

4. **对比评估（Comparative Evaluation）：**
   - 如果绝对评估困难，可以做 **相对比较**：
     - Agent vs 人类专家（AML 中和 junior investigator 做对比）
     - Agent v1 vs Agent v2（迭代优化的效果）
     - 不同 prompt 策略的 A/B 对比
   - 相对比较比绝对评估更容易、更有说服力

5. **人工评估的规模化：**
   - 人工评估不可避免，但可以用 **分层抽样** 控制规模：
     - Agent 高 confidence 的 case → 自动 metric + 少量抽查
     - Agent 低 confidence 的 case → 全量人工 review
     - 新类型的 case → 全量人工 review

> **Follow-up 提示：** 面试官可能追问 "LLM-as-Judge 的 prompt 怎么设计？"、"人工评估的一致性怎么保证（inter-annotator agreement）？"、"评估集多大才够？"

---

#### Q23: 维护生产环境的 agent 系统，最大的技术挑战是什么？

**回答：**

1. **Knowledge Drift（知识漂移）：**
   - 领域知识不是静态的——SOP 会更新、法规会变化、业务规则会调整
   - 如果 agent 的知识层没有及时更新，输出会变得过时甚至错误
   - **应对：** 建立知识更新的 pipeline——变更检测 → 增量更新 Graph KB → 回归测试 → 验证检索质量

2. **Prompt Fragility（Prompt 脆弱性）：**
   - Prompt 对措辞高度敏感——一个小改动可能导致输出质量大幅变化
   - LLM 模型升级时，之前调好的 prompt 可能不再适用
   - **应对：** 把 prompt optimization 当作工程流程而非一次性任务；建立 regression test suite，模型升级后自动跑回归

3. **Cascading Failures（级联故障）：**
   - Multi-agent 系统中，上游 sub-agent 的错误会 propagate 到下游
   - 例如数据采集 agent 遗漏了关键交易数据 → 分析 agent 得出错误结论 → report 中包含错误信息
   - **应对：** 每个 sub-agent 的输出做 sanity check；关键数据做交叉验证；设置 error boundary 防止级联

4. **Debugging 难度：**
   - 当 agent 的最终输出有问题时，定位 root cause 需要逐层 trace——是数据采集的问题？分析逻辑的问题？还是 report 生成时 hallucinate 了？
   - Multi-agent 的调试比单一程序复杂一个数量级
   - **应对：** 完整的 trace logging；file system 作为 snapshot；LLM-as-Judge 做自动化异常检测

5. **不可复现性：**
   - LLM 的输出有随机性（即使 temperature=0），同一个 input 可能产生不同 output
   - 这导致 bug 难以复现，debug 难度增加
   - **应对：** 记录完整的 input + output + 中间状态；用 seed 控制随机性（在支持的模型中）；关注 pattern 而非单次结果

> **Follow-up 提示：** 面试官可能追问 "遇到过最棘手的生产 bug 是什么？"、"怎么做 rollback？"、"有没有做 canary deployment？"

---

#### Q24: Agent 自主性和人工控制之间怎么权衡？

**回答：**

1. **Risk-based Intervention Points（基于风险的介入点设计）：**
   - 不是所有决策都需要人工确认。按照风险等级设定介入点：
     - **低风险操作：** 数据采集、格式转换 → 全自动，不需要人工确认
     - **中风险操作：** 分析结论、pattern 识别 → 自动执行，但结果需要人工 review
     - **高风险操作：** 最终 decision（file/no-file）、SAR report 提交 → 必须人工确认后才能执行

2. **Confidence Thresholds（置信度阈值）：**
   - Agent 对自己的判断给出 confidence score
   - High confidence → 展示结果，供 investigator 快速确认
   - Low confidence → 明确标注不确定性，建议 investigator 重点 review
   - Very low confidence → 自动触发人工介入，agent 不做推荐

3. **Progressive Trust（渐进式信任）：**
   - 初期：Agent 和 investigator 并行工作，对比结果，建立信任基线
   - 中期：Investigator 开始依赖 agent 的分析，但仍然全量 review
   - 后期：Investigator 只 review agent 标记为低 confidence 的 case
   - **关键：** 信任是逐步建立的，不能一步到位。让用户在使用中验证 agent 的能力

4. **AML 场景的特殊考量：**
   - 合规要求（regulatory requirement）决定了 human-in-the-loop 是 **mandatory** 而非 optional
   - 即使 agent 的 accuracy 达到 99%，最终的 SAR filing decision 仍然必须由人做
   - 但这不意味着 agent 没有价值——agent 把 investigator 的角色从 "做分析" 变成了 "review 分析"，效率提升巨大

5. **Design Principle：**
   - Agent 的定位是 **augmentation（增强），not replacement（替代）**
   - 系统设计时就要明确：哪些决策 agent 可以做，哪些必须人做，哪些是 agent 建议 + 人确认
   - 这个边界不是技术决定的，而是 **业务和合规需求** 决定的

> **Follow-up 提示：** 面试官可能追问 "人工 review 会不会变成橡皮图章（rubber stamp）？"、"怎么防止 investigator 过度依赖 agent？"、"不同行业的自主性边界有什么区别？"

---

#### Q25: 如果重新开始这个项目，你会做什么不同？

**回答：**

1. **评估基础设施先行（Evaluation Infrastructure Upfront）：**
   - 我们最初把大部分时间花在了 agent 开发上，评估体系是后来才建的
   - 如果重来，会 **先建评估体系再写 agent 代码**。没有评估就无法衡量改进，优化就是盲目的
   - 包括：test case suite、自动化评估 pipeline、多维度 metric dashboard

2. **Component Evaluation 先于 System Evaluation：**
   - 我们一开始直接评估端到端的 system performance，发现问题后很难定位是哪个 component 的锅
   - 如果重来，会先对每个 sub-agent 和 tool 做独立评估，确保每个 component 的质量，再做 system-level 评估
   - 这也让 debugging 变得更容易

3. **更简单的初始架构：**
   - 我们一开始就设计了完整的 multi-agent 架构。虽然最终证明这个架构是合理的，但中间有一段时间在和不必要的复杂性做斗争
   - 如果重来，会 **先用单 agent + 少量 tool 做 MVP**，验证核心流程可行，再根据具体瓶颈逐步拆分为 multi-agent

4. **更多时间在 Domain Knowledge Extraction 上：**
   - 知识获取比我们预想的 **难得多、花的时间多得多**
   - 如果重来，会在项目初期就安排更多的 SME 访谈时间，而不是等到开发中发现知识不够才去补
   - 也会更早开始 SAR report reverse engineering——这个方法非常有效但我们开始得太晚

5. **Prompt Optimization 更系统化：**
   - 初期我们手动调 prompt，效率很低。后来引入 DSPy + GEPA + TextGrad 的 pipeline 后效率大幅提升
   - 如果重来，从一开始就采用系统化的 prompt optimization 流程

6. **更好的 Observability：**
   - 初期的 logging 和 tracing 不够完善，production 出问题时 debugging 很痛苦
   - 如果重来，从 day 1 就建完整的 trace logging、metric monitoring、异常检测

> **Follow-up 提示：** 面试官可能追问 "这些教训中最重要的一条是哪个？"（通常是评估基础设施先行）、"有没有过度工程的例子？"、"团队组成会做什么调整？"

---

### 6.3 统计严谨性与商业决策类 (Statistical Rigor & Business Decision)

---

#### Q26: 80% accuracy 的统计置信度——100 个 test case 够不够？

**回答：**

1. **问题本质：** 如果真实 accuracy 是 80%，在 n=100 的 test set 上观测到 80% 的 confidence interval 有多宽？

2. **Binomial Proportion Confidence Interval：**
   - 用 Wilson interval（比 Wald interval 在小样本或极端概率下更准确）：
   - 95% CI ≈ [71.1%, 87.3%]
   - 这意味着：我们有 95% 的信心认为真实 accuracy 在 71%-87% 之间
   - **区间含义：** 最差情况下可能只有 71%，最好情况可能达到 87%——这个不确定性在合规场景下需要被 acknowledge

3. **Bootstrap Confidence Interval（推荐，更灵活）：**
   ```python
   # Bootstrap CI 计算方法
   import numpy as np
   results = [1]*80 + [0]*20  # 80/100 correct
   bootstrap_accuracies = []
   for _ in range(10000):
       sample = np.random.choice(results, size=100, replace=True)
       bootstrap_accuracies.append(np.mean(sample))
   CI_lower = np.percentile(bootstrap_accuracies, 2.5)  # ≈ 0.72
   CI_upper = np.percentile(bootstrap_accuracies, 97.5)  # ≈ 0.88
   ```
   - Bootstrap 不需要 normality 假设，对 class-imbalanced metric 也适用

4. **Power Analysis——需要多大的 test set？**
   - 如果目标是 95% CI 宽度 ≤ ±5%：需要 $n \geq \frac{Z^2 \times p(1-p)}{E^2} = \frac{1.96^2 \times 0.8 \times 0.2}{0.05^2} \approx 246$ 个 case
   - 如果目标是 ±3%：需要约 683 个 case
   - 目前 100 个 case 的 CI ≈ ±8%，确实有一些宽

5. **改善方法：**
   - **扩充 test set：** 最直接的方法。但 AML case 的标注成本高（需要 senior investigator + reviewer 双重确认）
   - **Stratified sampling：** 确保 test set 覆盖不同 case type 和复杂度的分布，与 production 一致
   - **持续积累：** 随着 production 使用，持续收集 investigator 对 agent decision 的 agree/disagree 反馈，逐步扩大评估样本
   - **分层报告 CI：** 对 simple/medium/complex case 分别报告 CI，虽然每层样本更少，但更有 actionable insight

6. **如何向管理层沟通：**
   - 不要只说 "accuracy 80%"，而是说 "在 100 个 test case 上，accuracy 80%，95% CI [71%, 87%]"
   - 强调 trend："随着使用量增加，我们在持续积累评估数据，CI 会逐步收窄"
   - 对比 baseline："junior investigator 在同样 test set 上的 accuracy 70-75%，agent 的 CI 下限 71% 仍然 at least comparable"

> **Follow-up 提示：** 面试官可能追问 "如果 accuracy 的 CI 包含了 70%，你还能说 agent on par with senior 吗？"、"怎么用 sequential testing 来持续评估？"

---

#### Q27: Agent 系统的 ROI 怎么算？如何说服管理层投入？

**回答：**

1. **ROI 计算框架：**

   ```
   ROI = (Annual Benefit − Annual Cost) / Annual Cost × 100%

   Annual Benefit:
     = Time Saving × Hourly Cost × Volume
     = (3.5h − 0.5h) × $50/h × 5000 cases/year
     = $750,000

   Annual Cost:
     = LLM API Cost + Infrastructure + Maintenance + Development
     = $50K (API) + $30K (infra) + $80K (0.5 FTE maintenance) + $100K (amortized dev cost over 3 years)
     = $260,000

   ROI = ($750K − $260K) / $260K = 188%
   Payback Period ≈ 4-5 months
   ```

2. **Tangible Benefits（可量化收益）：**
   - **Time saving：** 每个 case 从 3.5h 降到 30min（85% reduction）
   - **Capacity increase：** 同样的 investigator 团队可以处理更多 case（等效于增加 5-6x 产能）
   - **Quality consistency：** 减少人为差异，标准化输出质量
   - **Faster turnaround：** 监管对 SAR filing 有时间要求（30 天内），agent 减少了超期风险

3. **Intangible Benefits（不易量化但重要）：**
   - **Audit trail：** Agent 的每一步分析都有完整记录，满足合规审计要求
   - **Knowledge preservation：** 将 senior investigator 的经验编码到系统中，减少 brain drain 风险
   - **Training tool：** Junior investigator 可以通过 agent 的输出学习，加速 onboarding
   - **Scalability：** 当 case volume 增长时，不需要线性增加人员

4. **Opportunity Cost（不投入的代价）：**
   - 如果不做 agent，需要招更多 investigator 来应对 case volume 增长
   - 每个 investigator 的 fully loaded cost ~$120K/year，招 5 个 = $600K/year
   - 而 agent 系统的年维护成本远低于此

5. **Risk-adjusted ROI：**
   - Agent accuracy 80% 意味着 20% 的 case 需要 investigator 做更多修改
   - 考虑 false negative 的 regulatory risk：如果 agent 漏报了 case，罚款可能很高
   - 但 agent 是辅助而非替代——investigator 做最终决策，regulatory risk 可控

6. **向管理层 Pitch 的要点：**
   - Lead with impact numbers（time saving、capacity increase）
   - Acknowledge limitations（accuracy CI、edge case 处理）
   - Show trajectory（accuracy 在提升、CI 在收窄、scope 在扩大）
   - Compare alternatives（招人 vs 建系统的 total cost of ownership）

> **Follow-up 提示：** 面试官可能追问 "如果 LLM API 成本翻倍怎么办？"、"开发成本怎么估算的？"、"怎么衡量 intangible benefits？"

---

#### Q28: Agent 的决策是否存在 bias？不同 case type 的表现差异？

**回答：**

1. **为什么 Bias 在 AML 中是关键问题：**
   - AML 调查涉及对账户所有者的审查，如果 agent 对某些 demographic group 系统性地做出不同判断，可能导致：
     - **Regulatory risk：** 监管机构要求 fair lending / fair investigation 实践
     - **Reputation risk：** 被外界质疑歧视性执法
     - **Legal risk：** 违反 anti-discrimination 法律

2. **按维度拆分 Accuracy 分析：**

   | 维度 | 拆分方式 | 关注的 metric |
   |------|---------|-------------|
   | **Case type** | Structuring / Layering / Crypto / Wire transfer | Decision accuracy by type |
   | **Customer demographics** | 国籍、年龄段、账户类型（个人 vs 商业） | FPR by demographic group |
   | **Geographic region** | 客户所在国家 / 交易对手国家 | Detection rate by region |
   | **Transaction volume** | 高交易量 vs 低交易量账户 | Precision by volume tier |

3. **Disparate Impact Analysis：**
   - **Four-fifths Rule (80% Rule)：** 如果对某个 protected group 的 adverse action rate < 80% × 最高 group 的 rate，可能存在 disparate impact
   - 例如：如果 agent 对 Region A 客户的 "file SAR" rate = 25%，对 Region B = 15%，ratio = 60% < 80%，需要进一步调查
   - **注意：** Rate 差异不一定是 bias——可能 Region A 确实有更多 AML 活动。需要结合 base rate 做 calibration

4. **Bias 来源分析：**
   - **Training data bias：** 如果历史 case 中某些 group 被过度调查，agent 学到了这个偏见
   - **LLM inherent bias：** LLM 本身可能对某些地区、名字、文化背景有 stereotype
   - **Feature proxy：** 某些 feature（如 transaction 的国家）可能是 demographic 的 proxy
   - **Knowledge base bias：** 如果 SOP 本身对某些 pattern 有更详细的 guidance，agent 在这些 pattern 上表现更好

5. **Mitigation 策略：**
   - **Blind evaluation：** 在评估 agent 时隐去 demographic 信息，看是否影响 accuracy
   - **Prompt debiasing：** 在 system prompt 中明确指示 agent 不考虑客户的 demographic 特征做决策，只基于行为和交易 pattern
   - **Post-hoc analysis：** 定期做 bias audit，按维度拆分 accuracy 和 action rate
   - **Diverse test set：** 确保 test case 覆盖不同 demographic group
   - **Counterfactual testing：** 改变 case 中的 demographic 信息（名字、国家），看 agent 的 decision 是否变化——如果变了，说明 agent 在使用 demographic 信号

> **Follow-up 提示：** 面试官可能追问 "有没有实际发现过 bias 的案例？"、"如何在 accuracy 和 fairness 之间做 tradeoff？"

---

#### Q29: 如果模型底座从 GPT-4 切换到 Claude，怎么做迁移和评估？

**回答：**

1. **为什么可能需要切换：**
   - 成本优化（不同模型 pricing 差异大）
   - 性能提升（新模型在特定任务上可能更强）
   - Vendor risk 分散（不依赖单一供应商）
   - 合规要求（data residency、model governance）

2. **Regression Testing 框架：**
   - 用现有的 100-case test set 作为 regression baseline
   - 在新模型上跑同样的完整 pipeline，对比所有维度的 metric：
     - Decision accuracy（primary）
     - Report quality score
     - Evidence coverage
     - Hallucination rate
   - **Pass criteria：** 所有 metric 不显著低于旧模型（允许 ±2% 波动）

3. **Prompt Compatibility 分析：**
   - **问题：** 不同 LLM 对同一 prompt 的响应方式不同（格式、verbosity、reasoning style）
   - **方法：** 先直接用原 prompt 在新模型上测试，记录 failure mode
   - 常见问题：
     - 输出格式不一致（JSON 结构变化、markdown 格式差异）
     - Reasoning 深度不同（有些模型更 verbose，有些更简洁）
     - Tool calling behavior 差异（调用 tool 的时机和频率不同）
   - **解决：** 针对新模型调整 prompt——但尽量保持 prompt 的核心逻辑不变，只调整格式约束和风格指令

4. **Behavioral Difference Analysis：**
   - 逐 case 对比两个模型的输出，找出 systematic differences：
     - 有没有某类 case 新模型系统性地做错了？
     - 新模型是否更容易 hallucinate 或更保守？
     - Reasoning chain 的质量和 depth 有没有变化？
   - 用 LLM-as-Judge 自动化这个分析：让第三个 LLM 评估两个模型的输出质量

5. **模型无关的架构设计原则：**
   - **Abstraction layer：** 通过 LangChain 的 LLM wrapper 抽象，切换模型只需要改配置
   - **Prompt 模块化：** 将 prompt 的核心逻辑（如 CoT steps、tool description）和模型特定的格式指令分离
   - **Output parsing 鲁棒性：** Parser 要能处理不同模型的输出格式变体（不要 hard code 特定的 JSON 结构）
   - **Model-specific config：** 维护每个模型的最优 temperature、max tokens、system prompt 格式等

6. **迁移步骤（推荐顺序）：**
   1. Shadow mode：新模型并行运行，不影响 production
   2. Regression test：100-case test set 全量评估
   3. Prompt adjustment：根据 failure mode 调整 prompt
   4. A/B evaluation：选 20-30 个 case 让 investigator 盲评两个模型的输出
   5. Gradual rollout：先对低风险 case 切换，再逐步扩大

> **Follow-up 提示：** 面试官可能追问 "不同模型之间 prompt 的可移植性如何？"、"切换过程中怎么保证 production 不受影响？"

---

#### Q30: Multi-agent 系统的 end-to-end latency budget 怎么分解？

**回答：**

1. **当前 End-to-end Latency（完整 case 分析）：5-10 分钟**

2. **Latency Budget 分解：**

   | 组件 | 占比 | 典型延迟 | 说明 |
   |------|------|---------|------|
   | **LLM Inference** | ~60% | 3-6 min | Main agent + 多个 sub-agent 的 LLM 调用。每个 sub-agent 可能有 2-5 轮 LLM 调用（tool use loop） |
   | **Tool Execution** | ~20% | 1-2 min | 数据库查询（BigQuery）、外部 API（LexisNexis）、File IO |
   | **Middleware Overhead** | ~15% | 45-90s | LangGraph state checkpoint、Deep Agent 的 task dispatch、sub-agent 启动和 context initialization |
   | **Network Round-trip** | ~5% | 15-30s | LLM API network latency、inter-service communication |

3. **Profiling Methodology：**
   - **Trace-level profiling：** 在每个 node/edge/tool call 前后打 timestamp，生成完整的 timeline view
   - **LLM call profiling：** 记录每次 LLM 调用的 input token count、output token count、TTFT（Time to First Token）、total latency
   - **Tool call profiling：** 记录每个 tool 的调用时间、等待时间、返回数据量
   - **Aggregation：** 按 component type 汇总，得到上面的 budget 分解

4. **Critical Path Analysis：**
   ```
   Main Agent Planning (LLM call, 5s)
     ├── Data Collection Sub-agents (并行, 30-60s)
     │     ├── Internal DB query (15s)
     │     ├── External API calls (30-60s) ← bottleneck
     │     └── Profile data fetch (10s)
     ├── Analysis Sub-agents (部分并行, 2-3min) ← main bottleneck
     │     ├── Transaction Velocity Agent (LLM × 3 rounds, 45s)
     │     ├── Pattern Recognition Agent (LLM × 4 rounds, 60s)
     │     └── Crypto Agent (if applicable, LLM × 2 rounds, 30s)
     ├── Decision Agent (LLM × 2 rounds, 30s)
     └── Report Writer Agent (LLM × 3 rounds, 45s)
   ```
   - **Critical path = Data Collection (串行等待外部 API) + Analysis (最慢的 sub-agent)**
   - 优化应聚焦 critical path 上的组件

5. **Bottleneck Identification & 优化策略：**

   **Bottleneck 1 — LLM Inference（最大）：**
   - 每个 sub-agent 多轮 tool use loop 导致累计 LLM 调用次数多
   - 优化：(1) 对简单任务用更快的模型（Haiku < 1s vs Sonnet ~3s per call）(2) Batch 不需要交互的 analysis steps (3) 减少 tool call rounds（optimize prompt 让 agent 一次性提供更多 output）

   **Bottleneck 2 — Middleware Overhead（unexpected 的大头）：**
   - LangGraph 的 state checkpoint 在每个 node 后都做序列化，对大 state 对象开销显著
   - Deep Agent 的 sub-agent 启动有 context initialization 开销
   - 优化：(1) 精简 state 对象（只存必要信息）(2) 异步 checkpoint (3) 预热 sub-agent context

   **Bottleneck 3 — External API：**
   - LexisNexis 等外部服务的响应时间不可控（15-60s）
   - 优化：(1) 缓存常用查询结果 (2) 设置 timeout + fallback (3) 并行调用多个外部 API

6. **用户感知优化：**
   - Streaming 中间状态给 investigator："正在收集数据..."、"正在分析 transaction pattern..."
   - 分阶段返回结果：数据收集完就先展示 profile，不等分析完
   - Progress bar 或 estimated time remaining

> **Follow-up 提示：** 面试官可能追问 "profiling 结果中最 surprising 的发现是什么？"（答案：middleware overhead 比预期高）、"有没有做 latency SLA？"
