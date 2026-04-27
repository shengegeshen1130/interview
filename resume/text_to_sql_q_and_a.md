# Text-to-SQL for Risk Case Investigation 面试准备

---

## Part 1: Project Story（项目背景与故事线）

---

### 1.1 项目背景

Investigator 在 review case 的时候，如果发现可疑 pattern，需要手写 BigQuery SQL 去查询数据库，找到其他符合该 pattern 的账号。这个过程非常痛苦：

- **时间长：** 一个复杂查询可能需要 30-60 分钟手写和调试 SQL
- **准确率低：** Investigator 不是 SQL 专家，经常写错 syntax 或逻辑，导致查询结果不准确
- **重复劳动：** 类似的查询模式反复出现，但每次都要从头写
- **门槛高：** 新入职的 investigator 需要很长时间才能熟悉表结构和 SQL 写法

### 1.2 Use Case

做一个 text-to-SQL 的工具套件，让 investigator 输入自然语言，自动生成 BigQuery SQL，跑查询，处理报错，支持 human-in-the-loop 的交互式修改，并提供可视化和进一步分析能力。

**工具定位：** 不是一个黑盒——investigator 始终可以看到、修改、确认生成的 SQL，工具是辅助而非替代。

### 1.3 Impact 量化

**94% SQL Correctness 的计算方式：**
- 基于 golden dataset 中的 200 条 query 做评估
- **Correctness 定义：** 生成的 SQL 能通过 dry run（语法正确）且返回的结果集与 ground truth 一致（逻辑正确）
- 具体来说，对每条 query 检查：(1) SQL 是否能成功执行不报错，(2) 返回的 column 和 row 是否与预期匹配（用 set comparison，不要求顺序完全一致）
- 94% = 188/200 条 query 在首次生成或经过 dry run error feedback 后达到 correctness
- 剩余 6% 主要是极复杂的多表 join 或涉及 nested subquery 的情况

**80% Reduction in Manual Coding Time 的衡量方式：**
- 选取 20 个 investigator，对比使用工具前后完成相同类型查询任务的时间
- 使用前平均 30-45 分钟（手写 SQL + 调试），使用后平均 5-8 分钟（输入自然语言 + 少量修改）
- 80% 是取中位数的 reduction ratio

---

## Part 2: Technical Pipeline（技术流程详解）

---

### Step 1: Golden Dataset 准备

1. **收集真实样本：** 从 investigator 日常工作中人工收集 50 条真实的 query input，分析他们实际用到的 table 和 column
2. **对齐偏好：** 和 investigator 团队 align 他们输入时候的语法和内容偏好（比如习惯用缩写、特定术语）
3. **LLM 扩充：** 通过 LLM 基于真实 sample 生成新的 input-SQL pair，扩充到 200 条作为初始 dataset
4. **质量把控：** 每条扩充的数据都经过 dry run 验证 SQL 可执行，并由 investigator 抽检确认语义正确

### Step 2: RAG Pipeline

这是核心的 SQL 生成流程，是一个 multi-step RAG：

1. **Rephrase：** 对用户的 input query 先做一次 rephrase，目的是：
   - 将用户随意的自然语言转化为 vector DB 中 query 的规范语法
   - 统一术语（比如用户说"可疑账户"，rephrase 成"flagged accounts with risk score > threshold"）
   - 去除口语化表达，让 embedding 匹配更精准
2. **Embedding & Vector Search：** 用 **BGE-M3** 模型对 rephrased query 做 embedding，在 vector DB 中检索最相似的 query，取出对应的 BigQuery SQL 代码
3. **Prompt 拼接 & SQL 生成：** 将检索到的 few-shot examples（query + SQL pairs）concat 到 prompt 中，结合 table schema 信息，让 LLM 生成对应的 BigQuery SQL
4. **Chain-of-Thought：** 在 prompt 中要求 LLM 先分析 query 意图、列出需要的表和字段、规划 join 逻辑，再生成 SQL（详见 Part 4 Q3）

### Step 3: Error Handling & Dry Run

1. **Dry Run：** 生成的 SQL 先做 BigQuery dry run（不实际执行，只验证语法和权限）
2. **Error Feedback Loop：** 如果 dry run 报错，把报错信息（error message + 出错位置）反馈给 LLM，让它修正 SQL
3. **重试机制：** 最多重试 3 次，每次把上一轮的错误信息拼接到 context 中，让 LLM 逐步修正
4. **成功执行：** dry run 通过后，实际执行 SQL，把结果整理后返回给 investigator

### Step 4: Human-in-the-Loop

1. **结果反馈：** 如果查询没有返回结果，告知 investigator 并给出建议选项（比如放松某个条件、扩大时间范围）
2. **交互式修改：** Investigator 可以和 agent 交互，用自然语言描述想要的修改，agent 改写 SQL 后重新执行
3. **SQL 修改用 Subgraph：** 改写 SQL 的交互在 LangGraph 的 subgraph 中进行，不和 main agent 共享 context（详见 Part 3）
4. **循环直到满意：** Investigator 可以反复修改，直到获得满意的 dataframe 结果

### Step 5: 分析与可视化

1. **进一步分析：** 用户可以要求对返回的 dataframe 做更深入的分析
2. **PandasAI：** 用 PandasAI 做自然语言驱动的数据分析（如 "帮我统计这些账号的交易金额分布"、"按地区分组看风险等级"）
3. **可视化：** 生成图表帮助 investigator 直观理解数据 pattern
4. **PandasAI 同样用 Subgraph：** 和 SQL 修改一样，在独立的 subgraph 中运行

### Step 6: Feedback Loop

1. **收集反馈：** 用户对最终结果给出满意/不满意的反馈
2. **扩充 Vector DB：** 对于满意的 case，检查该 query-SQL pair 与现有 vector DB 的相似程度
3. **去重入库：** 如果相似度低于阈值（说明是新的 pattern），将该 pair 加入 vector DB，持续扩充 few-shot 资源
4. **Evaluation 数据：** 同时记录 query 的成功与否、用户满意度，作为系统持续评估的数据源

---

## Part 3: 架构设计

---

### 3.1 LangGraph Workflow 设计

**整体架构：** 用 LangGraph 构建 stateful multi-agent workflow，核心组件：

**Nodes（节点）：**
- **Intent Router Node：** 接收用户输入，判断意图类型（新查询 / 修改 SQL / 分析请求 / 反馈提交）
- **Rephrase Node：** 对用户 query 做 rephrase 和规范化
- **RAG Node：** 执行 embedding → vector search → few-shot retrieval
- **SQL Generator Node：** 拼接 prompt + few-shot examples + schema，调用 LLM 生成 SQL
- **Dry Run Node：** 执行 BigQuery dry run，收集报错信息
- **Error Fix Node：** 接收 error message，调用 LLM 修正 SQL
- **Execution Node：** 执行验证通过的 SQL，整理返回结果
- **SQL Modification Subgraph：** 处理用户的 SQL 修改请求
- **PandasAI Subgraph：** 处理进一步数据分析请求
- **Feedback Node：** 收集用户反馈，更新 vector DB

**Edges（边 / 路由逻辑）：**
- Intent Router → 根据意图分发到不同 node
- SQL Generator → Dry Run → 成功则 → Execution；失败则 → Error Fix → SQL Generator（最多 3 次循环）
- Execution → 有结果则返回用户；无结果则提供建议选项
- 用户选择修改 → SQL Modification Subgraph
- 用户选择分析 → PandasAI Subgraph

**State（状态管理）：**
- `user_query`: 用户原始输入
- `rephrased_query`: rephrase 后的 query
- `retrieved_examples`: RAG 检索到的 few-shot examples
- `generated_sql`: 当前生成的 SQL
- `dry_run_result`: dry run 结果（成功/报错信息）
- `retry_count`: 重试计数
- `execution_result`: SQL 执行结果（dataframe）
- `conversation_history`: 对话历史（用于 human-in-the-loop）

**Intent Routing 逻辑：**
- 使用 LLM 做 intent classification，输入是用户的当前消息 + 少量 context
- 分为 4 类 intent：`new_query`、`modify_sql`、`analyze_data`、`give_feedback`
- 根据 intent 将用户路由到对应的处理流程

### 3.2 Subgraph 设计

**为什么 SQL 修改和 PandasAI 要用 Subgraph：**

1. **Context 隔离：** Main agent 的 context 包含 table schema、RAG 检索结果、SQL 生成历史等大量信息。如果 SQL 修改和 PandasAI 的多轮对话也塞进 main context，很快就会超出 context window（尤其是 GPT-4o-mini 的 128K window 限制）
2. **职责清晰：** SQL 修改是基于已有 SQL 做增量修改，需要的 context 和从头生成 SQL 不同；PandasAI 分析是基于 dataframe 做操作，和 SQL 生成完全无关。用 subgraph 让每个组件只关注自己需要的信息
3. **State 不污染：** Subgraph 有自己的 state，不会和 main graph 的 state 互相干扰。比如 PandasAI 的多轮对话历史不会占用 main graph 的 context budget
4. **可独立迭代：** 每个 subgraph 可以独立优化 prompt、调整逻辑，不影响其他部分

**Subgraph 与 Main Graph 的交互：**
- Main graph 传入 subgraph 的是精简的 input（如 current SQL + 用户修改请求，或 dataframe + 分析请求）
- Subgraph 返回的是处理结果（修改后的 SQL 或分析结果）
- 不共享 conversation history

### 3.3 当时的限制与反思

**限制 1 — GPT-4o-mini 的 Context Window：**
- 当时由于种种原因只能用 GPT-4o-mini，context window 相对有限
- 虽然用了 subgraph 做 context 隔离，但如果对话轮次多的话还是容易丢信息
- 如果有更大 context window 的模型，可以减少 subgraph 拆分的必要性，让更多交互在同一 context 中完成

**限制 2 — 没有 Skill 的概念：**
- 当时还没有 skill（tool use）这种模式
- 现在回想，用 skill 对这个项目很有帮助：
  - 可以把一些常用流程固定为 tool（如 "查某个时间段的高风险账号"）
  - 把 investigator 的经验和 feedback 写成 skill 的 description，让 LLM take 更多 control
  - 比我们用 LangGraph 写死的逻辑要灵活，也更 scalable
  - LLM 可以根据用户意图自主决定调用哪些 tool，而不是走固定的 routing 逻辑

---

## Part 4: 核心技术问题

---

### Q1: 为什么用 RAG 而不是 Fine-tuning？

**回答：**

1. **数据量不够：** 只有 200 条 golden dataset，fine-tuning 一个 text-to-SQL 模型需要的数据量远远不够，容易过拟合
2. **Schema 经常变：** BigQuery 的 table schema 会随业务变化更新（新增 column、改表名），fine-tuning 的模型无法动态适应这些变化，而 RAG 只需要更新 vector DB 中的 examples 和 prompt 中的 schema 信息
3. **可解释性和可控性：** RAG 可以看到检索到了哪些 few-shot examples，方便 debug 和理解模型为什么生成某个 SQL。Fine-tuning 是黑盒，出了问题很难定位
4. **成本和速度：** Fine-tuning 需要 GPU 资源和训练时间，每次 schema 变化都要重新 fine-tune。RAG 只需要更新 embedding 和 vector DB，几分钟就能上线
5. **Few-shot 已经够用：** 对于我们的 use case，大部分查询模式是相似的（几种常见的 table join pattern），2-3 个相关的 few-shot examples 就能让 LLM 生成正确的 SQL
6. **持续改进更容易：** 通过 feedback loop 不断扩充 vector DB，RAG 的效果会随使用量自然提升

> **Follow-up 提示：** 面试官可能追问 "如果数据量够大会考虑 fine-tuning 吗？"、"RAG 和 fine-tuning 能不能结合？"

---

### Q2: BGE-M3 选型理由

**回答：**

1. **多语言支持：** BGE-M3 支持 100+ 种语言，我们的 investigator 团队有多地区成员，query 中可能混合英文和其他语言的术语
2. **多粒度检索（Multi-Granularity）：** BGE-M3 支持 dense retrieval、sparse retrieval（类 BM25）、multi-vector retrieval 三种模式，可以灵活组合，提升检索召回率
3. **长文本能力：** 支持最长 8192 tokens 的输入，足以处理我们的 query（有些 investigator 会写很长的描述性查询）
4. **性能均衡：** 在 MTEB benchmark 上排名靠前，且推理速度适中，满足我们接近实时的响应需求
5. **与我们场景的适配：** 我们的 query 是技术性文本（包含 SQL 关键词、表名、字段名），BGE-M3 对这类 technical text 的 embedding 质量高于通用模型（如 OpenAI text-embedding-ada）
6. **开源可控：** 开源模型，可以本地部署，不需要调用外部 API，符合公司的数据安全要求

> **Follow-up 提示：** 可能追问 "做过 embedding model 的对比实验吗？"、"sparse 和 dense retrieval 怎么结合的？"

---

### Q3: Chain-of-Thought 怎么用的？

**回答：**

1. **使用环节：** 主要在 SQL Generator Node 中，让 LLM 在生成 SQL 之前先做结构化思考

2. **Prompt 设计：** 要求 LLM 按以下步骤输出思考过程：
   - **Step 1 — Intent Analysis：** 分析用户想要查询什么（"用户想找过去 30 天内交易金额超过 10K 的可疑账号"）
   - **Step 2 — Table & Column Identification：** 列出需要用到的表和字段（"需要 transactions 表的 amount、date 字段，accounts 表的 risk_score 字段"）
   - **Step 3 — Join Logic：** 规划表之间的 join 关系（"transactions JOIN accounts ON account_id"）
   - **Step 4 — Filter Conditions：** 列出所有 WHERE 条件
   - **Step 5 — SQL Generation：** 基于以上分析生成最终 SQL

3. **效果：** 加入 CoT 后 SQL correctness 从约 85% 提升到 94%，主要改善了多表 join 和复杂条件的场景

4. **为什么有效：**
   - 强迫 LLM 先理解意图再生成 SQL，减少了 "直接生成" 导致的语义错误
   - 中间步骤的输出可以作为 debug 信息，当 SQL 有问题时可以看是哪一步理解错了
   - 分步思考让 LLM 更好地利用 few-shot examples 中的 pattern

> **Follow-up 提示：** 可能追问 "CoT 的 token 开销大不大？"、"有没有用 self-consistency（多次生成取多数）？"

---

### Q4: 如何评估 SQL Correctness（94% 怎么来的）？

**回答：**

1. **评估数据集：** 200 条 golden dataset，每条包含 natural language query + ground truth SQL + expected result set

2. **Correctness 的多层定义：**
   - **Level 1 — Syntax Correct：** SQL 能通过 BigQuery dry run，无语法错误（这个达到了 ~98%）
   - **Level 2 — Semantically Correct：** SQL 的逻辑正确，返回结果集与 ground truth 匹配（这个是 94%）
   - **结果匹配方式：** 用 set comparison——检查返回的 row 集合是否一致（不要求顺序），允许 column alias 不同

3. **评估流程：**
   - 对每条 query，运行完整 pipeline（rephrase → RAG → CoT → SQL generation → dry run → error fix）
   - 最终生成的 SQL 实际执行，结果与 ground truth 做对比
   - 包含了 dry run error feedback 的重试（最多 3 次），所以 94% 是经过 error correction 后的最终数字

4. **失败案例分析（6%）：**
   - 极复杂的多表 join（4+ 张表）
   - 涉及 nested subquery 或 window function 的高级 SQL
   - 用户 query 表述模糊，存在多种合理的 SQL 解读

5. **Execution Accuracy vs Exact Match（补充）：**
   - **Exact Match (EM)：** 生成的 SQL 和 ground truth SQL 完全一致（token-level）。这个标准太严格——同一个查询有多种等价写法（如 `WHERE a > 5 AND b < 10` 和 `WHERE b < 10 AND a > 5`），EM 会把等价 SQL 判为错误
   - **Execution Accuracy (EX)：** 我们主要用的标准——生成的 SQL 和 ground truth SQL 执行后的 **结果集一致**。不要求 SQL 写法一样，只要结果对就行
   - **Partial Credit Evaluation：**
     - 有些 SQL 结果 "部分正确"（如正确的 table join 但 WHERE 条件缺了一个），完全算错太浪费信息
     - 我们设计了 partial credit：(1) Table coverage — 是否用了正确的 table (2) Column coverage — SELECT 的 column 是否包含 ground truth 的所有 column (3) Row filter accuracy — WHERE 条件的覆盖度
     - Partial credit 帮助定位 "错在哪一步"，指导 prompt 优化方向
   - **实际指标体系：**

     | Metric | 定义 | 我们的结果 |
     |--------|------|----------|
     | Syntax Correctness | 通过 dry run，无语法错误 | ~98% |
     | Execution Accuracy | 结果集与 ground truth 一致 | 94% |
     | Exact Match | SQL token 完全一致 | ~65%（过于严格） |
     | Partial Credit (avg) | 按 table/column/filter 加权 | ~92% |

> **Follow-up 提示：** 可能追问 "如果不做 error feedback，correctness 是多少？"、"ground truth 是谁标注的？"

---

### Q5: Rephrase 这一步具体做什么？

**回答：**

1. **核心目的：** 弥合用户自然语言和 vector DB 中标准 query 之间的语义 gap

2. **具体操作：**
   - **术语标准化：** 把用户随意的说法转换为 vector DB 中使用的标准术语（如 "可疑交易" → "transactions flagged as suspicious"）
   - **结构化表达：** 把口语化的长句拆解为结构化的查询描述（如 "帮我找过去一个月那些转了很多钱给同一个人的账号" → "Find accounts with total outgoing transfers > threshold to a single recipient in last 30 days"）
   - **去噪：** 去除无关信息（如 "我刚才忘了说" 这类上下文无关的表述）
   - **补全：** 如果用户 query 缺少关键信息（如没说时间范围），保留缺失部分以便后续追问

3. **实现方式：** 使用 LLM + 专门的 rephrase prompt，few-shot examples 展示 "原始输入 → rephrased 输出" 的对照

4. **效果：** 加入 rephrase 后，vector search 的 retrieval recall@3 从 72% 提升到 89%（更容易找到相关的 few-shot examples）

> **Follow-up 提示：** 可能追问 "rephrase 会不会引入误差？"、"rephrase 的 LLM 和 SQL 生成的 LLM 是同一个吗？"

---

### Q6: Vector DB 的选型与管理

**回答：**

1. **选型：** 使用 ChromaDB 作为 vector store
   - 轻量、易部署，适合我们 200-500 条数据的规模
   - Python 原生支持，和 LangChain/LangGraph 集成好
   - 支持 metadata filtering（可以按 table name、query type 等做过滤）

2. **数据结构：** 每条记录包含：
   - `query_text`: 标准化的自然语言 query
   - `sql_code`: 对应的 BigQuery SQL
   - `embedding`: BGE-M3 生成的向量
   - `metadata`: table names、query type、创建时间等

3. **管理策略：**
   - **初始加载：** 200 条 golden dataset 作为 seed
   - **增量更新：** 通过 feedback loop，用户满意的新 case 自动入库
   - **去重检查：** 新 case 入库前检查与现有数据的 cosine similarity，高于阈值的不重复入库
   - **版本管理：** 定期 snapshot，支持回滚

4. **为什么不用更重量级的方案（如 Pinecone、Milvus）：**
   - 数据量小（几百条），不需要分布式向量数据库
   - 延迟要求不极端，ChromaDB 单机即可满足
   - 简化运维，减少外部依赖

> **Follow-up 提示：** 可能追问 "数据量增长到几千条怎么办？"、"embedding 模型升级时怎么处理？"

---

## Part 5: 面试官视角深度问题

---

### 系统设计类

---

#### Q7: Workflow 和 Agent 有什么区别？为什么这个项目选择 LangGraph 做 Workflow？

**回答：**

1. **Workflow vs Agent 的核心区别：**

   | 维度 | Workflow（编排式） | Agent（自主式） |
   |------|-------------------|----------------|
   | **控制权** | 开发者定义流程，LLM 在每个节点内执行具体任务 | LLM 自主决定下一步做什么、调用什么 tool |
   | **执行路径** | 确定性的——走预定义的 nodes 和 edges，条件分支也是开发者写的 | 不确定性的——LLM 根据上下文动态决策，每次可能走不同路径 |
   | **典型模式** | DAG / State Machine：rephrase → RAG → generate → dry run | ReAct loop：Thought → Action → Observation → 循环直到完成 |
   | **可靠性** | 高——流程可预测、可复现、易 debug | 较低——LLM 可能做出意外决策、陷入循环、跳过关键步骤 |
   | **灵活性** | 低——新需求需要改代码加节点/边 | 高——给 agent 新的 tool 就能处理新场景 |
   | **适用场景** | 流程明确、步骤固定、对可靠性要求高 | 开放式任务、难以预定义所有路径、需要 LLM 创造性决策 |
   | **代表框架** | LangGraph、Prefect、Airflow | LangChain ReAct Agent、AutoGen、CrewAI |

2. **为什么这个项目适合用 Workflow 而不是 Agent：**
   - **流程是固定的：** text-to-SQL 的核心链路非常明确——rephrase → vector search → SQL generation → dry run → execute。不需要 LLM 自主决定 "接下来该做什么"，每一步该做什么是确定的
   - **对可靠性要求高：** 生成的 SQL 要执行在生产数据库上，不能让 LLM 随意跳过 dry run 或跳过安全检查。Workflow 保证每个关键步骤一定会执行
   - **Error handling 有固定策略：** dry run 失败 → 重试最多 3 次 → 超过则兜底。这种确定性的 error handling 用 workflow 的 conditional edge 实现更可靠，agent 可能选择不同策略
   - **可审计性：** 作为 internal tool，每次查询的流程需要完整记录（走了哪些步骤、每步的输入输出）。Workflow 的执行路径可预测，审计和 debug 都很方便

3. **为什么选择 LangGraph 而不是其他 workflow 框架：**
   - **Stateful 多轮对话：** LangGraph 的 state machine 天然支持在多轮交互中维护状态（如 conversation history、current SQL、retry count）
   - **Subgraph 支持：** 可以把 SQL 修改和 PandasAI 分析拆成独立的 subgraph，做 context 隔离
   - **Human-in-the-loop 原生支持：** LangGraph 支持 interrupt 和 resume，investigator 可以在任意节点介入修改
   - **与 LangChain 生态集成：** 直接复用 LangChain 的 LLM wrapper、prompt template、output parser 等组件
   - **Checkpoint & 持久化：** 内置 state checkpoint，支持断点续跑和历史回溯

4. **与其他方案的对比：**
   - **LangChain ReAct Agent：** 让 LLM 自主决定用什么 tool，适合开放式任务。但我们的流程不需要 LLM 自主决策，用 ReAct 反而引入不稳定性——LLM 可能跳过 rephrase、选错 tool、或在 error handling 时做出意外行为
   - **AutoGen / CrewAI：** 更适合多个 agent 角色之间的对话和协作（如 "coder" 和 "reviewer" 互相讨论），我们的场景是单一流程内的步骤编排，不需要多 agent 协商
   - **Prefect / Airflow：** 传统 data pipeline 编排工具，不支持 LLM 交互和 human-in-the-loop
   - **纯代码编排：** 也可以，但 LangGraph 提供了 state 持久化、可视化、checkpoint 等开箱即用的功能，减少了开发量

5. **Latency 考量：**
   - 整个 workflow 涉及多次 LLM 调用（rephrase、SQL generation、可能的 error fix），总 latency 约 10-15 秒
   - 可接受范围——对比 investigator 手写 SQL 的 30-60 分钟，这个等待完全值得
   - 通过 streaming 让用户看到中间过程（如 "正在检索相似查询..."、"正在生成 SQL..."），改善感知体验

6. **反思——什么时候会切换到 Agent 模式：**
   - 如果未来 investigator 的需求变得更开放（不仅是 text-to-SQL，还要自主搜索知识库、调用多个内部 API、做跨系统分析），那 agent 模式会更合适
   - 上面提到的 "skill" 的想法其实就是向 agent 模式演进——把固定的 workflow 步骤变成 tool，让 LLM 根据意图自主组合调用

> **Follow-up 提示：** 可能追问 "如果现在重新做，会选 workflow 还是 agent？"、"LangGraph 的 state 是怎么持久化的？"、"有没有考虑 hybrid 方案（大流程用 workflow，局部用 agent）？"

---

#### Q8: 如果要做 Production 部署，你会怎么设计？

**回答：**

1. **当前状态：** 项目是以 internal tool 的形式部署，用户通过 web UI 交互

2. **Production 化的考量：**
   - **服务化：** 用 FastAPI 包装 LangGraph workflow，提供 REST API
   - **异步处理：** SQL 执行可能较慢，用 async + WebSocket 推送结果和中间状态
   - **Auth & RBAC：** 不同 investigator 可能有不同的 table 访问权限，需要在 SQL 执行前做权限检查
   - **Logging & Monitoring：** 记录每次 query 的完整 trace（input → rephrase → retrieved examples → generated SQL → result），用于 debug 和审计
   - **Rate Limiting：** 控制 LLM API 调用频率和 BigQuery 查询量

3. **安全性：**
   - **SQL Injection 防护：** 虽然 SQL 是 LLM 生成的，但仍需 validate 生成的 SQL 不包含 destructive 操作（DELETE、DROP、UPDATE）
   - **只允许 SELECT：** 通过 SQL parser 强制只允许 SELECT 语句
   - **Row-level Security：** 利用 BigQuery 的 row-level security 确保 investigator 只能看到自己权限范围内的数据

> **Follow-up 提示：** 可能追问 "怎么处理并发？"、"BigQuery 的 cost 怎么控制？"

---

### LLM 工程类

---

#### Q9: Prompt Engineering 策略是什么？Hallucination 怎么处理？

**回答：**

1. **Prompt 结构：**
   - **System prompt：** 定义角色（"你是一个 BigQuery SQL 专家"）+ 严格约束（"只生成 SELECT 语句"、"只使用以下 table 和 column"）
   - **Schema 信息：** 把相关 table 的 schema（table name、column name、data type、description）放在 prompt 中
   - **Few-shot examples：** RAG 检索到的 2-3 个最相似的 query-SQL pair
   - **CoT instruction：** 要求分步思考
   - **User query：** rephrased 后的用户输入

2. **Hallucination 防护：**
   - **Schema 限制：** Prompt 中明确列出所有可用的 table 和 column，要求 LLM 只使用这些。如果 LLM 生成了不存在的 table/column，dry run 会报错并触发 error fix
   - **Dry Run 是最强防护：** 语法错误和引用不存在的 table/column 都会在 dry run 阶段被捕获
   - **Few-shot grounding：** 通过 RAG 提供的 few-shot examples 让 LLM "照着写"，大幅降低幻觉概率
   - **Output Parsing：** 用 regex 提取 SQL block，忽略 LLM 可能输出的无关文本

3. **Token 管理：**
   - Schema 信息只包含与当前 query 相关的 table（通过 CoT 的 table identification 步骤或关键词匹配筛选）
   - Few-shot examples 限制为 2-3 条，避免 context 过长
   - Subgraph 隔离避免历史对话占用 main context

> **Follow-up 提示：** 可能追问 "schema 信息怎么动态筛选？"、"如果 LLM 总是对某类 query hallucinate 怎么办？"

---

#### Q10: Token 开销大吗？怎么优化成本？

**回答：**

1. **每次 query 的 token 消耗：**
   - Rephrase: ~500 tokens（input + output）
   - SQL Generation: ~2000-3000 tokens（schema + few-shot + CoT + output）
   - Error Fix（如果需要）: ~1000 tokens per retry
   - 总计：约 3000-5000 tokens per query

2. **成本优化策略：**
   - **GPT-4o-mini：** 选用成本更低的模型（当时的限制反而成了优势）
   - **Schema 动态裁剪：** 不把全部 schema 都放进 prompt，只放相关的 table
   - **Cache：** 对于完全相同的 query，cache 住结果避免重复调用 LLM
   - **Subgraph 隔离：** 避免无用的 context 占用 token

3. **实际成本：** 每条 query 约 $0.003-0.005（GPT-4o-mini 价格），对于 internal tool 完全可接受

> **Follow-up 提示：** 可能追问 "如果切换到更贵的模型（如 GPT-4o），ROI 怎么算？"

---

### RAG 类

---

#### Q11: Retrieval 质量怎么评估？Few-shot 样本数量怎么确定的？

**回答：**

1. **Retrieval 质量评估：**
   - **Recall@K：** 在 K 条检索结果中，包含正确 SQL pattern 的比例
   - 我们主要看 **Recall@3**：检索 3 条最相似的 example，其中是否至少有一条与目标 SQL 的 pattern 匹配
   - Rephrase 前 Recall@3 约 72%，加入 rephrase 后提升到 89%

2. **Few-shot 数量选择：**
   - 实验了 1、2、3、5 条 few-shot examples
   - **3 条效果最好：** 提供足够的 pattern 覆盖，又不会占用太多 context
   - 1 条容易受噪声影响（如果唯一的 example 不太相关）
   - 5 条 context 太长，GPT-4o-mini 的生成质量反而下降（context window 压力）

3. **Few-shot 的排列：**
   - 相似度最高的放最后（closest to the generation point），利用 LLM 的 recency bias

> **Follow-up 提示：** 可能追问 "Recall@3 的 ground truth 怎么定义的？"、"有没有用 reranker？"

---

#### Q11.1: Reranker Model 是什么？有什么优点？

**回答：**

1. **什么是 Reranker：**
   - Reranker 是 RAG pipeline 中 retrieval 之后、送入 LLM 之前的一个**精排模型**
   - 典型的 RAG 检索分两阶段：**Stage 1（召回）** 用 embedding cosine similarity 快速从 vector DB 中捞出 top-K 候选；**Stage 2（精排）** 用 reranker 对这 K 个候选重新打分排序，选出最终的 top-N 送进 prompt
   - 本质上是一个 **cross-encoder**：把 query 和每个候选 document 拼接在一起作为输入，输出一个 relevance score

2. **Reranker vs Embedding Retrieval（Bi-encoder）的核心区别：**

   | 维度 | Embedding Retrieval（Bi-encoder） | Reranker（Cross-encoder） |
   |------|----------------------------------|--------------------------|
   | **编码方式** | Query 和 document 分别独立编码为向量，算 cosine similarity | Query 和 document 拼接后一起送入模型，做 full cross-attention |
   | **交互深度** | 无交互——两个向量独立生成，只在最后算相似度 | 深度交互——query 和 document 的每个 token 之间都做 attention |
   | **精度** | 较低——独立编码丢失了 query-document 之间的细粒度交互信息 | 更高——cross-attention 能捕捉精确的语义匹配和细微差异 |
   | **速度** | 极快——document embedding 可预计算，在线只需算 query embedding + ANN 搜索 | 慢——每个 query-document pair 都要过一次模型，无法预计算 |
   | **适用阶段** | 召回阶段（从百万级候选中快速捞出 top-K） | 精排阶段（对 K 个候选重新排序，K 通常 10-50） |

3. **Reranker 的优点：**
   - **精度显著提升：** Cross-attention 让模型能理解 query 和 document 之间的细粒度关系。比如 query 是 "找共享 device 的账号"，embedding 检索可能把 "找共享 IP 的账号" 排得很高（语义接近），但 reranker 能区分 "device" 和 "IP" 的差异，把真正匹配的排到前面
   - **解决语义漂移：** Bi-encoder 的 embedding 是通用的语义表示，可能无法精确区分领域内细微的意图差异。Reranker 对 query-document pair 做联合理解，能更好地处理这类 case
   - **即插即用：** 不需要修改 vector DB 或 embedding 模型，只需在 retrieval 后面加一层 reranker，就能提升最终送入 LLM 的 few-shot 质量
   - **容错性：** 即使 Stage 1 的 embedding 检索不够精准（比如有些噪声结果混入 top-K），reranker 可以把这些噪声过滤掉

4. **常见的 Reranker 模型：**
   - **BGE-Reranker（BAAI）：** 和我们用的 BGE-M3 embedding 同系列，有 base/large/v2 等多个版本
   - **Cohere Rerank：** API 服务，效果好但需要调用外部 API
   - **cross-encoder/ms-marco-MiniLM：** 基于 MS MARCO 训练的轻量级 cross-encoder

5. **为什么我们当时没有用 Reranker：**
   - Vector DB 只有 200-500 条数据，候选池非常小，embedding retrieval 的 top-3 已经足够精准
   - 加 reranker 意味着多一次模型调用，增加 latency（约 100-200ms），对我们小数据量场景的 marginal gain 不大
   - 但如果 vector DB 增长到几千条以上，retrieval 噪声增大，reranker 就变得有价值了——这是一个明确的优化方向

> **Follow-up 提示：** 可能追问 "cross-encoder 和 bi-encoder 能不能结合训练？"、"reranker 的 fine-tuning 需要什么数据？"、"如果加了 reranker，retrieval 的 top-K 应该设多大？"

---

#### Q12: Embedding 更新策略是什么？Vector DB 怎么保持质量？

**回答：**

1. **Embedding 更新：**
   - 当 BGE-M3 模型版本升级时，需要对整个 vector DB 重新 encode（因为不同版本的 embedding space 不兼容）
   - 数据量小（几百条），full re-encode 只需几分钟，不是瓶颈

2. **数据质量维护：**
   - **入库审核：** 自动入库的新 case 会标记为 "auto-added"，定期由 team 成员抽检确认质量
   - **淘汰机制：** 如果某个 example 长期没有被检索命中（说明不常用），或者 schema 变更导致 SQL 失效，标记为 inactive
   - **Schema 同步：** 当 BigQuery table 发生 schema change 时，自动检查 vector DB 中受影响的 SQL，标记需要更新的 entries

3. **防止 drift：**
   - 每月跑一次 golden dataset 的 regression test，确保整体 correctness 没有下降
   - 如果新加入的 examples 导致效果下降（retrieval 混淆），回滚到上一个 snapshot

> **Follow-up 提示：** 可能追问 "如果 vector DB 增长到几千条，性能会有影响吗？"

---

### 评估类

---

#### Q13: 用户满意度怎么衡量？

**回答：**

1. **直接指标：**
   - **首次成功率：** 用户第一轮生成的 SQL 就满意（不需要修改）的比例，约 70%
   - **总成功率：** 经过修改交互后最终满意的比例，约 90%
   - **Feedback 评分：** 每次 session 结束后用户给的满意度评分（1-5 分）

2. **间接指标：**
   - **使用频率：** Investigator 每周使用工具的次数，反映 adoption 程度
   - **手写 SQL 减少量：** 对比使用前后 investigator 手写 SQL 的频率
   - **Session 时长：** 从输入 query 到获得满意结果的总时间

3. **定性反馈：** 定期和 investigator 做 user interview，收集 feature request 和痛点

4. **标准化 UX 指标（补充）：**
   - **NPS (Net Promoter Score)：** 问 investigator "你有多大可能向同事推荐这个工具？"（0-10 分）
     - Promoter (9-10) / Passive (7-8) / Detractor (0-6)
     - NPS = % Promoter − % Detractor
     - 我们的 NPS 约 45（在内部工具中算不错——50+ 为优秀）
   - **Task Completion Rate：** Investigator 使用工具成功完成查询任务的比例（不放弃、不回退到手写 SQL）
     - 当前约 88%——12% 的 session 最终放弃工具转为手写
   - **SUS (System Usability Scale)：** 标准化的 10 题问卷（Brooke, 1996），涵盖易用性、学习曲线、功能集成度等
     - 每题 1-5 分，最终分数 0-100
     - SUS > 68 算 above average，我们的 SUS 约 72
   - **Time-on-Task Distribution：** 不仅看平均时间，还看时间分布——如果有 bimodal 分布（多数很快、少数很慢），说明有特定类型的 query 是 pain point
   - **Error Recovery Rate：** 当首次生成的 SQL 不正确时，通过交互修改最终成功的比例——反映 human-in-the-loop 的有效性

> **Follow-up 提示：** 可能追问 "不满意的 10% 主要是什么问题？"、"怎么提升 NPS？"

---

### 业务类

---

#### Q14: Investigator 的使用模式是什么？Adoption 有什么挑战？

**回答：**

1. **使用模式：**
   - **典型流程：** Investigator 在 review case 时发现可疑 pattern → 打开工具 → 用自然语言描述想查询的条件 → 获得结果 → 可能做几轮修改 → 满意后导出结果
   - **高频场景：** "找过去 N 天与账号 X 共享 device/IP 的其他账号"、"查金额超过 Y 的可疑交易"
   - **Session 频率：** 平均每个 investigator 每天使用 3-5 次

2. **Adoption 挑战：**
   - **信任问题：** 一开始 investigator 不信任 AI 生成的 SQL，会逐条检查。需要通过展示中间过程（CoT 思考步骤、检索到的 examples）来建立信任
   - **习惯改变：** 资深 investigator 已经习惯手写 SQL，觉得工具不如自己写的精准。通过 human-in-the-loop 的设计（始终可以修改 SQL）降低抵触
   - **Query 表述差异：** 不同 investigator 的描述方式差异很大，rephrase 步骤就是为了解决这个问题
   - **Cold start（初期）：** Vector DB 只有 200 条，coverage 不够全面。通过 feedback loop 随使用量增长逐步改善

> **Follow-up 提示：** 可能追问 "怎么做 onboarding？"、"有没有 power user 和 casual user 的区别？"

---

### 工程挑战类

---

#### Q15: Complex SQL（多表 join、嵌套查询）怎么处理？

**回答：**

1. **多表 Join：**
   - CoT 中专门有一步 "Join Logic Planning"，让 LLM 先规划好 join 关系
   - Few-shot examples 中刻意包含多表 join 的样例，覆盖常见的 join pattern
   - Schema 信息中标注了 table 之间的 foreign key 关系，帮助 LLM 理解表的关联

2. **嵌套查询：**
   - 对于复杂的 nested subquery，CoT 会引导 LLM 先写内层 query，再包装外层
   - 这是 94% correctness 之外那 6% 失败的主要来源

3. **兜底策略：**
   - 如果 3 次 error fix 后仍然失败，展示最接近正确的 SQL 给 investigator，让他们手动修改
   - 同时记录这个失败 case，后续人工编写正确的 SQL 加入 golden dataset

> **Follow-up 提示：** 可能追问 "有没有做 SQL 拆分——把复杂 query 拆成多个简单 query？"

---

#### Q16: 安全性怎么保障？

**回答：**

1. **SQL 层面：**
   - **只允许 SELECT：** 生成的 SQL 经过 SQL parser 检查，只放行 SELECT 语句。任何包含 DELETE、DROP、UPDATE、INSERT 的 SQL 直接拒绝
   - **Table 白名单：** 只允许查询预定义的 table 列表，防止 LLM 生成访问敏感表的 SQL
   - **Dry Run 前置：** 所有 SQL 先 dry run，不会直接执行未验证的语句

2. **数据层面：**
   - **BigQuery IAM：** 利用 BigQuery 原生的 IAM 权限控制，service account 只有特定 dataset 的 read 权限
   - **结果脱敏：** 返回结果中的敏感字段（如完整的 credit card number）做 masking
   - **审计日志：** 每次查询都记录 who/when/what，支持合规审计

3. **LLM 层面：**
   - **Prompt injection 防护：** 用户输入在拼接到 prompt 前做 sanitization，防止通过 query 注入恶意指令
   - **Output validation：** LLM 输出通过 regex 提取 SQL block，忽略任何非 SQL 内容

> **Follow-up 提示：** 可能追问 "有没有遇到过 prompt injection 的案例？"、"BigQuery 的 cost 怎么防止 expensive query？"

---

### 统计严谨性与实验设计类 (Statistical Rigor & Experiment Design)

---

#### Q17: 200 条 golden dataset 的样本量是否足够？怎么论证？

**回答：**

1. **Confidence Interval 分析：**
   - 94% correctness 在 n=200 时的 95% Wilson CI ≈ [90.2%, 96.7%]
   - 区间含义：我们有 95% 信心认为真实 correctness 在 90%-97% 之间
   - 区间宽度 ≈ ±3.3%，对于 internal tool 来说精度可接受

2. **Coverage Analysis（比样本量更重要）：**
   - 样本量够不够，不仅看数量，还要看 **coverage**——是否覆盖了所有常见的 query pattern
   - 我们做了 query pattern 分类：

   | Query Pattern | 数量 | 占比 | Correctness |
   |--------------|------|------|-------------|
   | 单表查询 | 60 | 30% | 98% |
   | 双表 JOIN | 75 | 37.5% | 96% |
   | 3 表 JOIN | 35 | 17.5% | 91% |
   | 4+ 表 JOIN | 15 | 7.5% | 73% |
   | Window Function | 10 | 5% | 80% |
   | Nested Subquery | 5 | 2.5% | 60% |

   - **发现：** 复杂 query（4+ 表 JOIN、nested subquery）的样本量不够——15 条的 CI 非常宽
   - 这说明 overall 94% 可能掩盖了复杂 query 上的不足

3. **Bootstrap Analysis：**
   ```python
   # Bootstrap CI for overall correctness
   results = [1]*188 + [0]*12  # 188/200 correct
   bootstrap_scores = [np.mean(np.random.choice(results, 200, replace=True)) for _ in range(10000)]
   CI = (np.percentile(bootstrap_scores, 2.5), np.percentile(bootstrap_scores, 97.5))
   # CI ≈ (0.905, 0.970)
   ```
   - Bootstrap 还可以对 stratified metrics 做 CI——如 "复杂 query 的 correctness CI"

4. **Power Analysis——如果要检测 improvement：**
   - 假设优化后 correctness 从 94% 提升到 97%，需要多大 test set 才能检测到？
   - McNemar test: 需要约 500 条 query 才有 80% power 在 α=0.05 下检测到 3% 的 absolute improvement
   - 当前 200 条对于检测大的改进（>5%）够用，但对于检测小的改进不够

5. **改善策略：**
   - **优先扩充低覆盖 pattern：** 重点增加 4+ 表 JOIN、nested subquery 的样本
   - **Production 数据回流：** 通过 feedback loop 持续积累新的 query-SQL pair
   - **Synthetic augmentation 的局限：** LLM 生成的样本可能集中在 "容易的" pattern，对难 pattern 的覆盖不足——需要和 investigator 合作收集真实的复杂 query

> **Follow-up 提示：** 面试官可能追问 "golden dataset 的质量怎么保证？"、"有没有 test set contamination 的风险？"（即 test set 中的 pattern 可能已经在 vector DB 中）

---

#### Q18: 80% time reduction 的 user study 怎么设计的？统计上怎么验证？

**回答：**

1. **实验设计：**
   - **参与者：** 20 个 investigator（混合 senior/junior、不同 team）
   - **设计类型：** Within-subject（每人都做 with-tool 和 without-tool 两组任务）
   - **任务分配：** 每人完成 10 个 query 任务——5 个手写 SQL，5 个用工具。任务难度 matched（按 pattern 类型配对）

2. **Confounding Factor 控制：**
   - **Learning Effect（学习效应）：** 先做手写 SQL 还是先用工具会影响结果。解决：随机化顺序——一半人先手写再用工具，一半人反过来（counterbalanced design）
   - **Task Difficulty Matching：** 两组任务的难度需要 matched。解决：预先按 query pattern 分类，每对任务来自同一 pattern 类别
   - **Familiarity Bias：** 对工具的熟悉程度不同。解决：实验前给所有参与者 15 分钟的工具使用培训
   - **Hawthorne Effect：** 被观察可能导致表现异常。解决：告知参与者 "评估工具而非评估个人"

3. **统计验证——Paired t-test：**
   - 因为是 within-subject 设计，用 paired t-test（或非参数的 Wilcoxon signed-rank test）
   - H₀: μ_tool = μ_manual（使用工具和手写 SQL 没有时间差异）
   - 结果：p < 0.001，差异高度显著
   - **Effect Size (Cohen's d)：** d ≈ 3.2（极大效应），说明不仅是统计显著，而且实际差异非常大
   - 95% CI for mean time reduction: [75%, 85%]

4. **为什么选 Within-subject 而非 Between-subject：**
   - **Within-subject 优势：** 每个参与者作为自己的控制组，消除了个体差异（有些人本来就写 SQL 快），power 更高
   - **Between-subject 劣势：** 需要更多参与者来控制个体差异，我们只有 20 个 investigator 可用
   - **Within-subject 风险：** 学习效应和 fatigue 效应——通过 counterbalanced design 缓解

5. **结果报告：**
   ```
   手写 SQL: M = 37.5 min (SD = 12.3)
   使用工具: M = 6.8 min (SD = 3.1)
   Time Reduction: M = 81.3% (SD = 8.7%)
   Paired t-test: t(19) = 11.2, p < 0.001
   Cohen's d = 3.2
   ```

6. **局限性：**
   - 样本量 20 人偏小（但 within-subject + large effect size 使结果仍然可靠）
   - 任务是预设的，可能不完全反映日常工作的 query 分布
   - 工具使用培训后立即测试，可能高估了 "新手" 使用工具的效率

> **Follow-up 提示：** 面试官可能追问 "如果做 between-subject design 需要多少人？"、"long-term 使用后 time reduction 是否会变化？"

---

#### Q19: Rephrase 这一步会不会引入 semantic drift？怎么控制？

**回答：**

1. **什么是 Semantic Drift：**
   - Rephrase 的目的是标准化用户 query 以提升 retrieval recall，但如果 rephrase 改变了用户的原始意图，就会导致生成错误的 SQL
   - 例如：用户说 "找最近一周转账多的人" → rephrase 成 "Find accounts with high transaction volume in the last 7 days" → 但 "转账" 特指 P2P transfer，"交易" 是所有 transaction，意思已经变了

2. **Failure Mode Analysis：**
   - 我们分析了所有 rephrase 导致的错误（约 3% 的 query），主要分为：
     - **术语泛化（最常见）：** 用户用的是 specific term，rephrase 变成了 generic term（如 "credit card transaction" → "transaction"）
     - **条件丢失：** 用户 query 中的隐含条件在 rephrase 时被 drop（如 "最近" 被忽略）
     - **歧义消解错误：** 用户 query 有歧义，rephrase 选了错误的解读

3. **Semantic Similarity 监控：**
   - 计算 original query 和 rephrased query 的 cosine similarity（用 BGE-M3 embedding）
   - 设定阈值：如果 similarity < 0.75，触发 warning，可能意味着 rephrase 偏离太多
   - 实践中约 5% 的 rephrase 触发 warning，其中约一半确实有 semantic drift

4. **控制策略：**
   - **Fallback to Original：** 如果 rephrase 的 similarity score 低于阈值，同时用 original query 和 rephrased query 做 retrieval，取并集
   - **Rephrase Prompt 约束：** 在 rephrase prompt 中明确要求 "保留所有 specific conditions 和 domain terms，不要泛化"
   - **Few-shot Examples 引导：** 在 rephrase prompt 中提供 "好的 rephrase" 和 "坏的 rephrase" 的对照示例
   - **User Confirmation（可选）：** 对低 confidence 的 rephrase，展示给用户确认（"您是想查询 P2P 转账还是所有交易？"）。但这会增加交互步骤

5. **Ablation 分析：**
   - 有 rephrase vs 无 rephrase：
     - Retrieval Recall@3: 72% → 89%（+17%）
     - End-to-end SQL Correctness: 88% → 94%（+6%）
     - Semantic drift 导致的错误: 0% → 3%（−3%）
   - 净效果是正面的（+6% net），但 3% 的 semantic drift 是需要持续优化的

> **Follow-up 提示：** 面试官可能追问 "有没有考虑用 query expansion 代替 rephrase？"、"rephrase 和原始 query 都检索后怎么 merge？"

---

#### Q20: 如果 vector DB 从 200 条增长到 5000 条，系统设计怎么变？

**回答：**

1. **ChromaDB 的性能评估：**
   - **当前（200-500 条）：** ChromaDB 完全够用，single machine，retrieval latency < 50ms
   - **5000 条时：** ChromaDB 仍然可以 handle（它在百万级数据上仍能工作），但需要关注：
     - Retrieval 精度下降：更多 candidate 意味着更多 noise，top-K 中可能混入不相关的 example
     - Memory 使用增加（但 5000 条 embedding 仍然很小）
     - 索引构建时间增加（但仍然是秒级）

2. **什么时候需要切换 Vector DB：**
   - **5000 条：不需要切换**——ChromaDB 完全能应对
   - **50K+ 条 + 低延迟要求：** 考虑 Milvus（分布式、GPU 加速）或 Pinecone（managed service）
   - **切换触发条件：** retrieval latency > 200ms 或 recall@K 明显下降

3. **更重要的问题——Data Curation Strategy：**
   - **不是越多越好**：5000 条如果质量参差不齐，retrieval 会被 noise 污染
   - **质量 > 数量** 的策略：
     - **去重：** 语义相似度高的 example 只保留最好的那个
     - **Coverage-driven curation：** 按 query pattern 分类，确保每个类别有足够但不过多的 example
     - **Performance-based pruning：** 如果某个 example 从未被 retrieval 命中（说明不常用）或总是导致错误 SQL，标记为 inactive
     - **Staleness check：** schema 变更后受影响的 SQL 必须更新或移除

4. **Retrieval 精度的应对：**
   - **加入 Reranker：** 当 candidate 池变大，embedding retrieval 的 top-K 中噪声增多，此时 reranker 的价值就体现了
     - Retrieval top-20 → Reranker 精排 → 取 top-3 作为 few-shot
     - 推荐 BGE-Reranker（和我们的 BGE-M3 同系列）
   - **Metadata Filtering：** 利用 ChromaDB 的 metadata filter，先按 table name / query type 过滤，再做 semantic search，缩小 candidate 池
   - **Cluster-based Retrieval：** 对 5000 条做 clustering，retrieval 时先定位最相关的 cluster，再在 cluster 内做 fine-grained search

5. **Embedding Model 的 Scalability：**
   - BGE-M3 的 embedding 维度 1024，5000 条 = 5000 × 1024 × 4 bytes ≈ 20MB，完全放得进内存
   - 如果未来升级到更大的 embedding model，re-encode 5000 条也只需要几分钟
   - **版本管理：** embedding model 升级时必须全量 re-encode（不同版本的 embedding space 不兼容）

6. **系统架构的渐进式演进：**
   ```
   200 条:  ChromaDB (单机) + 直接 retrieval
   2000 条: ChromaDB + metadata filter + 去重策略
   5000 条: ChromaDB + metadata filter + Reranker + curation pipeline
   50K+ 条: Milvus/Pinecone + Reranker + cluster-based retrieval
   ```

> **Follow-up 提示：** 面试官可能追问 "5000 条的 curation 谁来做？自动还是人工？"、"metadata filter 的设计具体是什么？"

---

#### Q21: 和直接 fine-tune 一个 code LLM（如 CodeLlama）做 text-to-SQL 相比，你的方案优势在哪？

**回答：**

1. **系统性对比：**

   | 维度 | Few-shot RAG (我们的方案) | Fine-tuning Code LLM |
   |------|------------------------|---------------------|
   | **Data Efficiency** | 200 条即可工作 | 需要数千-上万条高质量训练数据 |
   | **Schema Adaptability** | Schema 变更只需更新 prompt 中的 schema 信息 + vector DB 中受影响的 example | 需要重新 fine-tune（新表名/列名不在训练数据中） |
   | **Interpretability** | 可以看到检索了哪些 example，CoT 展示推理过程 | 黑盒，难以解释为什么生成某个 SQL |
   | **Maintenance Cost** | 更新 vector DB + 调 prompt，几分钟上线 | 重新训练 + 部署 + 验证，days to weeks |
   | **上线速度** | 从 0 到可用只需 2-3 周 | 数据收集 + 标注 + 训练 + 调参，months |
   | **Generalization** | 利用 base LLM 的通用 SQL 知识 + 特定 few-shot 引导 | 可能过拟合到训练数据中的 pattern |
   | **Cost** | API 调用成本（~$0.005/query） | 训练 GPU 成本 + 推理 GPU 成本 + 运维 |

2. **Fine-tuning 的优势场景：**
   - 数据量足够大（5000+）且 pattern 相对固定
   - 需要极低延迟（fine-tuned 小模型 < 100ms vs RAG pipeline 5-15s）
   - 有 GPU 资源和 MLOps 能力做持续训练
   - 场景高度 specialized（如只生成特定 dialect 的 SQL）

3. **我们的场景为什么 RAG 更合适：**
   - **数据量只有 200 条**：fine-tuning 的最大瓶颈。即使用 LoRA 等 efficient fine-tuning，200 条仍然容易过拟合
   - **Schema 经常变**：BigQuery table 会新增列、改表名，fine-tuning 的模型无法动态适应——每次 schema 变更都要重新训练
   - **需要可解释性**：Investigator 需要看到 "为什么生成这个 SQL"（CoT + retrieved examples），fine-tuned 模型无法提供
   - **快速上线需求**：项目从立项到上线只有几周时间，fine-tuning pipeline 的搭建成本太高

4. **Hybrid 方案（如果未来数据量增长）：**
   - **RAG + Fine-tuned Embedding：** 不 fine-tune LLM 本身，而是 fine-tune embedding model（如在 domain-specific query pair 上 fine-tune BGE-M3），提升 retrieval 精度
   - **RAG + Fine-tuned Small Model：** 用 fine-tuned 小模型做简单 query（单表、常见 pattern），复杂 query 仍走 RAG pipeline
   - 这样既利用了 fine-tuning 在常见 pattern 上的高效率，又保留了 RAG 对新 pattern 的适应性

5. **最新进展的考量：**
   - 开源 text-to-SQL 模型（如 SQLCoder、NSQL）在 Spider benchmark 上效果很好，但在 **custom schema + domain-specific terms** 上仍然需要大量适配
   - 这些模型的训练数据是 public SQL 数据集，和我们的 BigQuery schema 差距很大
   - **Bottom line：** 在我们 200 条数据 + 频繁 schema 变更的场景下，RAG 仍然是最务实的选择

> **Follow-up 提示：** 面试官可能追问 "如果给你 5000 条数据，你会怎么选择？"（答案：先 RAG，然后 fine-tune embedding model 提升 retrieval，最后考虑 fine-tune LLM 作为 base + RAG 作为 augmentation 的 hybrid 方案）
