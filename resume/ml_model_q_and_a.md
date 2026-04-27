# ML模型开发面试准备 (Feb 2019 - May 2023)

---

## Part 1: Use Case详解

---

### 1.1 Buyer AUP Violation (买家违规)

**业务背景**
AUP = Acceptable Use Policy，即PayPal规定用户能做什么、不能做什么。这个项目的目标是检测buyer有没有用PayPal去购买违规商品，比如武器、毒品、色情服务等。

**方法论 (5 Steps)**

**Step 1: 样本挖掘**
从已知的违规用户出发，找到曾经和这些违规用户做交易的商家，然后去找和这些商家做交易的其他买家，对这些买家建模。Assumption是有相同违规行为的买家可能会去类似的商家购买商品（比如买毒品的人可能还会去买针头）。

**Step 2: Feature Engineering**
基础的买家/卖家profiling变量、交易变量、limitation、activity、industry、transaction note通过BERT做embedding等。

**Step 3: Word2Vec Buyer Embedding**
把每一个商家的交易对象当成一个sentence，每一个buyer就是一个word，然后用Word2Vec建模得到每一个buyer的embedding。用KMeans做聚类，把每一个类别的信息作为buyer的一部分变量。这些变量最终被证明对模型效果有正面作用。

**Step 4: PU Learning扩充标签**
由于违规buyer的tagging非常不完整（需要人工打标），我们用Positive-Unlabeled Learning这种半监督学习方法去扩充tagging。通过这个方法我们得到一部分model score较高的sample，拿给人工去确认——在某个threshold上有95%都被人工证明为正样本。

**Step 5: 建模**
直接用了LightGBM，表现很好，不需要处理categorical变量。对于feature数量不多且需要快速上线的项目，LightGBM是第一选择。

**亮点/难点**
- PU Learning解决标签不完整问题，详见 Part 2.2
- Word2Vec embedding捕捉buyer行为网络关系
- 业务驱动的样本挖掘策略

---

### 1.2 Stolen Financial (盗刷检测)

**业务背景**
用偷来的信用卡/银行卡做交易。如果卡的原主人提了chargeback，loss就要PayPal自己承担。Stolen financial在支付公司是永恒的议题。

**方法论**

**Tagging分析**
- 和业务团队合作，他们会给我们一些当前模型没有办法捕捉到的case
- 从分析tagging开始：先看这些case能不能被当前的tagging逻辑打成正样本
- 如果不行，研究如何改进tagging逻辑尽可能把新case包进去
- Tagging分析完成后得出结论：这些case没被抓到是tagging的问题还是模型的问题

**Feature Engineering**
一般会对tagging进行调整，然后用已有变量先训练出一个baseline版本。有了baseline后开始迭代，设计新变量来更好地抓到坏人。对stolen CC来说，一般从以下方面考虑：
- 卡BIN风险评分
- 设备指纹异常
- IP地理异常
- Fraudster的行为序列模型（有没有异常的行为）
- 历史dispute率
- Graph fraud detection（资金流图）

很多时候stolen CC会是新账户来操作的，所以新账户的email pattern、IP、或者其他profile的mismatch也会被考虑进去。

**实战案例**
之前遇到的一个较大的trend：一批非洲国家的账号，用Gmail邮箱注册，然后先做几笔正常交易去测试我们的rule，一部分没有被封的账号就开始添加黑市里买的CC卡做交易。对collusion来说也是一样的，测试完账号之后可以作为中间账号出钱。

**建模**
- Feature selection: IV、PSI、correlation，以及通过变量对模型影响力做feature selection（原理：用一个简单的NN或树模型，看如果把某个feature设为空，会对模型结果有多大影响，通过多次循环去掉影响力低的feature）
- 一般用LightGBM作为baseline，然后不断调整NN model去看能不能beat baseline
- NN调优时看model layer层面有没有可调整的（深度、activation function等），以及weight上能不能做文章（比如用loss金额作为sample weight，让模型更在意高loss的样本）

**亮点/难点**
- 与业务团队的tagging协作流程
- Iterative model development（tagging → baseline → feature iteration → model tuning）
- Loss-weighted training让模型关注高损失样本

---

### 1.3 Merchant Website Risk (商户网站合规)

**业务背景**
不是传统的建模，而是做小的component，通过处理unstructured数据，对一个单独的use case做一套solution。一些不怀好意的商家会在页面上做手脚。

**风险类型**
- Underprice（价格过低）
- 虚假广告
- 奇怪的HTML结构（模板化，批量创建）
- 假的payment链接：用户点击PayPal但实际跳转到假的付款页面

**数据来源**
- 内部工具爬回来的网络结构+信息
- 外部流量数据（如Alexa）

**建模方法**
基于爬取的网站结构和内容信息，结合外部流量数据，构建变量做模型。重点在于从非结构化的网页数据中提取有意义的signal，比如HTML结构的模板化程度、页面内容与实际商品的匹配度、流量模式的异常等。

**亮点/难点**
- 非结构化数据处理（HTML parsing、网页内容分析）
- 多数据源融合（内部爬取 + 外部流量）
- 需要对商户网站的各种欺诈模式有业务理解

---

### 1.4 User Behavior Similarity & Anomaly Detection

**业务背景**
针对bot操作以及buyer-seller action similarity（collusion检测）。不是传统的建模，而是做component级别的solution。

**方法论: LSTM Encoder-Decoder**
对账号的历史行为（click、page view、页面停留时长）做sequence，然后通过LSTM + Encoder-Decoder的方式得到：
1. Sequence的embedding（用于相似性分析）
2. 重构误差（用于异常检测）

**LSTM (Long Short-Term Memory) 原理**

标准RNN的问题：在长序列上梯度会vanish或explode，无法学习长距离依赖。LSTM通过引入cell state和三个gate来解决。

核心结构——每个时间步有三个gate和一个cell state：

```
Forget Gate:  f_t = σ(W_f · [h_{t-1}, x_t] + b_f)       决定丢弃多少旧信息
Input Gate:   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)       决定写入多少新信息
Candidate:    C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)    候选新信息
Cell Update:  C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t           更新cell state
Output Gate:  o_t = σ(W_o · [h_{t-1}, x_t] + b_o)       决定输出多少信息
Hidden State: h_t = o_t ⊙ tanh(C_t)                      当前时间步的输出
```
其中 σ = sigmoid，⊙ = element-wise乘法

**直觉理解**：
- **Cell state (C_t)**：像一条传送带，信息可以沿着它无损地向前传递（这就是解决梯度消失的关键——梯度可以通过cell state直接回传）
- **Forget gate (f_t)**：看当前输入和上一步的hidden state，决定cell state中哪些旧信息要丢掉（输出0-1之间的值，0=完全丢弃，1=完全保留）
- **Input gate (i_t)**：决定哪些新信息要写入cell state
- **Output gate (o_t)**：决定cell state中哪些信息要输出到hidden state

**为什么LSTM能解决长距离依赖**：
- 标准RNN中信息必须通过连续的矩阵乘法传递，导致梯度指数衰减
- LSTM的cell state通过加法更新（C_t = f_t⊙C_{t-1} + ...），梯度可以沿cell state直接回传而不会vanish
- Forget gate学会在需要时保持f_t接近1，让信息长期保留

**LSTM变体**：
- **GRU (Gated Recurrent Unit)**：合并了forget和input gate为一个update gate，参数更少，训练更快，效果通常与LSTM相当
- **Bidirectional LSTM**：同时从前向后和从后向前处理序列，捕捉双向上下文
- **Stacked LSTM**：多层LSTM叠加，增加模型容量

**在本项目中的应用 (Encoder-Decoder)**：
- **Encoder**：LSTM逐步读入用户行为sequence，最终的hidden state就是整个sequence的compressed embedding
- **Decoder**：另一个LSTM，从embedding出发尝试重构原始sequence
- 训练目标：最小化重构误差（input和output的差异）
- Embedding用于similarity → 相似行为的用户embedding接近
- 重构误差用于anomaly detection → 异常行为难以被正常模式训练的模型重构，误差大

**Word2Vec Action Embedding**
先用Word2Vec给每一个action做embedding，然后再输入LSTM。这样做的好处是训练过程更稳定——对于无监督学习来说，Word2Vec作为起始embedding能提供更多的信息量，相比随机初始化的embedding。

**额外维度**
除了页面和点击数据，sequence里还加入了一些profiling信息去提供额外的分析维度。

**应用**
- **Similarity**: Embedding拿来做clustering，找到有类似行为的账号群体（用于collusion检测）
- **Anomaly Detection**: 重构误差用来做anomaly detection，误差大的说明行为模式偏离正常（可能是bot或异常操作）

**亮点/难点**
- 无监督学习方案，不依赖标签
- Word2Vec pre-training stabilize LSTM训练
- 同一个模型同时产出两种signal（similarity + anomaly）

---

## Part 2: 横向技术专题

---

### 2.1 Feature Engineering: IV, PSI, WOE

**IV (Information Value)**
- 用途：衡量一个变量对目标变量（好/坏）的预测能力
- 公式：IV = Σ (Good% - Bad%) × WOE
- 判断标准：
  - IV < 0.02：无预测能力，可以剔除
  - 0.02 ≤ IV < 0.1：弱预测能力
  - 0.1 ≤ IV < 0.3：中等预测能力
  - IV > 0.3：强预测能力（但要注意overfitting风险）
- 实践：作为feature selection的第一步筛选，IV过低的变量直接剔除

**WOE (Weight of Evidence)**
- 公式：WOE = ln(Good% / Bad%)
- 作用：把categorical变量转成与log-odds线性相关的连续值
- 优点：
  - 处理categorical变量，避免one-hot导致维度爆炸
  - 转换后变量与log-odds线性相关，对logistic regression特别友好
  - 自动处理缺失值（缺失值作为单独一个bin）
- WOE > 0 说明该bin中好样本占比更高，WOE < 0 说明坏样本占比更高

**PSI (Population Stability Index)**
- 用途：监控变量分布随时间的偏移（feature drift / population drift）
- 公式：PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
- 判断标准：
  - PSI < 0.1：分布稳定
  - 0.1 ≤ PSI < 0.25：需要关注，可能有shift
  - PSI > 0.25：显著偏移，需要调查或重新训练
- 实践：
  - 定期计算每个feature的PSI，作为模型监控的一部分
  - PSI高的feature可能导致模型performance degradation
  - 也可以对model score本身计算PSI来监控模型输出的稳定性

---

### 2.2 PU Learning详解

**场景**
PU Learning是一种弱监督学习场景：
- 只有正样本P（label=1）
- 以及一堆未标注样本U（可能是正也可能是负）
- 没有显式的负样本

在fraud detection中非常常见：你知道一些confirmed fraud case，但大部分fraud case其实没有被发现/标注。

**Spy流程（经典方法）**
1. 从P中抽一部分样本当作"spy"
2. 把spy放入U中
3. 训练P vs U的分类器
4. 根据spy在U中的score分布，确定负样本阈值（spy应该和真正的正样本有相似的score，如果spy的score低于某个阈值，那些score更低的U样本可以被认为是可靠的负样本）
5. 挑出可靠负样本（Reliable Negatives）
6. 用P和Reliable Negatives重新训练分类器

**与标准半监督学习的区别**
- 标准semi-supervised：有少量labeled（包含正负样本）+ 大量unlabeled，利用unlabeled数据的分布信息来提升模型
- PU Learning：完全没有负样本标签，核心问题是如何从U中识别可靠的负样本
- 使用场景不同：PU Learning适用于"正样本可以确认但负样本不确定"的场景（如fraud detection），semi-supervised适用于"正负样本都有但数量少"的场景

---

### 2.3 Explainable AI / SHAP单样本解释

**业务需求**
模型做出决策后，需要给analyst/business stakeholder解释"为什么这个case被标记为高风险"。

**做法**
1. **维护可解释变量集**：维护一批业务含义清晰的可解释变量（比如"近30天交易次数""IP国家与账户注册国家是否一致"等），而不是直接用production model的所有变量（可能包含embedding等不可解释的feature）
2. **训练Surrogate Model**：用这批可解释变量训练一个surrogate model（通常是树模型），去模拟production model的输出（不是模拟真实label，而是模拟production model的score）
3. **计算Shapley值**：对surrogate model用TreeExplainer（树模型）或KernelExplainer（通用）计算单样本的Shapley值
4. **展示结果**：将top contributing features及其SHAP值展示给analyst，比如"该用户被标记为高风险主要因为：IP国家与注册国家不一致（+0.15）、近7天交易金额异常高（+0.12）..."

**为什么用Surrogate Model而不是直接解释Production Model**
- Production model可能包含不可解释的feature（embedding、交叉特征等）
- Surrogate model用业务可理解的变量，analyst能直接理解每个feature的含义
- 同时保持了production model的复杂度和performance

---

### 2.4 样本不平衡处理 & Focal Loss

**样本不平衡应对方法**

**1. Class Weight调整**
在loss function中给少数类更高的权重。例如正负样本比1:100，可以给正样本设置weight=100。
```
weighted_loss = -[w_pos × y × log(p) + w_neg × (1-y) × log(1-p)]
```

**2. Sample Weight（按金额加权）**
不仅考虑类别不平衡，还按loss金额加权，让模型更关注高损失样本。比如一个$10,000的fraud case比$10的fraud case重要得多。

**3. Focal Loss**
公式：
```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
```
其中 p_t = p if y=1, else (1-p)

参数解释：
- **α (alpha)**：类别权重，控制正负样本的balance，和class weight类似
- **γ (gamma)**：focusing parameter，常用γ=2。作用是降低easy sample（模型已经很confident的样本）的权重，让模型更关注hard sample
- 当γ=0时退化为标准cross-entropy
- 直觉：一个easy negative（p_t=0.9），其loss会被 (1-0.9)^2 = 0.01 大幅缩小；而一个hard sample（p_t=0.5），loss只被 (1-0.5)^2 = 0.25 缩小，相对保留了更多梯度

**4. PU Learning**
针对正样本不完整的场景，不是传统意义上的"不平衡"，而是标签本身就不完整。详见 Part 2.2。

**5. SMOTE (Synthetic Minority Over-sampling Technique)**
通过在少数类样本之间插值生成合成样本，而不是简单复制。
- 算法：对每个少数类样本，找到它的K个最近邻（通常K=5），随机选一个近邻，在两者之间的连线上随机取一个点作为新样本
- 公式：
```
x_new = x_i + λ × (x_nn - x_i)，其中 λ ∈ [0, 1] 随机
```
- 变体：
  - **Borderline-SMOTE**：只对处于决策边界附近的少数类样本做插值（这些样本最容易被误分），而不对被多数类包围的noise点或远离边界的safe点做插值
  - **SMOTE-ENN**：SMOTE过采样后，用Edited Nearest Neighbors清理掉噪声样本（如果一个样本的多数近邻和它类别不同，就删掉）
  - **ADASYN (Adaptive Synthetic Sampling)**：对越难学习的少数类样本（周围多数类越多的）生成越多的合成样本
- 优点：比简单复制（random oversampling）多样性更好，不容易overfitting
- 缺点：高维空间中插值可能生成不合理的样本；对noise敏感（如果原始少数类样本本身是noise，插值会放大noise）
- 实践注意：**只对training set做SMOTE，validation/test set保持原始分布**

**6. Random Undersampling & Tomek Links**
减少多数类样本来平衡数据。

- **Random Undersampling**：随机删除多数类样本直到达到目标比例。简单但会丢失信息。
- **Tomek Links**：找到互为最近邻但类别不同的样本对（Tomek link），删除其中的多数类样本。效果是清理决策边界，让boundary更清晰。
```
Tomek Link: (x_i, x_j) 满足 d(x_i, x_j) < d(x_i, x_k) 且 d(x_i, x_j) < d(x_j, x_l)
对所有 k ≠ j 且 l ≠ i，且 y_i ≠ y_j
即 x_i 和 x_j 互为最近邻但属于不同类别
```
- **NearMiss**：有策略地选择多数类样本保留——比如NearMiss-1保留离少数类样本最近的多数类样本（保留最有信息量的boundary samples）

**7. Ensemble方法: EasyEnsemble & BalanceCascade**

- **EasyEnsemble**：从多数类中多次随机采样（每次采样数量等于少数类），每次和全部少数类组合训练一个base learner，最终ensemble所有base learner。这样每个learner看到balanced数据，同时多次采样减少了信息丢失。
- **BalanceCascade**：迭代式训练——每轮训练一个classifier，然后把被正确分类的多数类样本去掉，用剩余的多数类样本进入下一轮。逐步减少多数类中的easy sample。

**8. Cost-Sensitive Learning（代价敏感学习）**
不改变数据分布，而是在算法层面赋予不同类别不同的误分类代价。
```
Total Cost = Σ C(predicted, actual) × count
其中 C(predicted=neg, actual=pos) >> C(predicted=pos, actual=neg)
```
- 和class weight的区别：class weight是在loss function层面加权，cost-sensitive learning是在决策层面定义代价矩阵，可以更灵活地定义不同错误类型的代价
- 在fraud detection中特别适用：miss一笔$10K的fraud的代价远高于误封一个正常用户
- 可以结合业务的actual dollar loss来定义cost matrix

**9. Threshold Moving（阈值调整）**
不改变模型训练，而是在inference阶段调整decision threshold。
- 默认threshold=0.5对不平衡数据不合适
- 方法：在validation set上根据业务目标（如target precision、target recall、最小化expected cost）选择最优threshold
- 公式（基于代价的最优threshold）：
```
threshold* = C_FP / (C_FP + C_FN)
其中 C_FP = false positive的代价，C_FN = false negative的代价
```
- 优点：简单，不需要重新训练模型；可以根据业务需求灵活调整
- 实践中常用：设置多个threshold对应不同action（自动block / 人工review / 通过）

**10. Class-Balanced Loss Based on Effective Number of Samples (CB Loss)**
来自CVPR 2019论文。核心思想：随着样本数量增加，新样本带来的信息增益递减（因为新样本可能和已有样本重叠）。用"有效样本数"而非原始样本数来计算类别权重。

- **Effective Number公式**：
```
E_n = (1 - β^n) / (1 - β)，其中 β = (N - 1) / N，n = 该类别样本数，N 为总样本数
```
- β ∈ [0, 1) 是一个hyperparameter：
  - β → 0 时，E_n → 1，所有类别权重相同（不做reweighting）
  - β → 1 时，E_n → n，退化为按原始样本数的逆频率加权（即传统inverse class frequency）
  - 实践中常用 β = 0.9, 0.99, 0.999 等

- **Class-Balanced Loss公式**：
```
CB Loss = (1 / E_n_y) × L(y, p)

展开: CB Loss = [(1 - β) / (1 - β^n_y)] × L(y, p)

其中 n_y 是样本所属类别的样本数，L 可以是任意loss（CE, Focal Loss等）
```

- **直觉理解**：
  - 假设每个样本覆盖feature space中的一个小区域，新样本和已有样本的区域有重叠的概率为 β
  - 少数类：每个新样本的信息增益高（重叠少），effective number接近实际样本数
  - 多数类：大量样本之间高度重叠，effective number远小于实际样本数
  - 所以多数类的权重会被压低，但不像简单inverse frequency那样激进

- **与其他方法的结合**：
  - CB + Softmax Cross-Entropy：最基础的组合
  - CB + Focal Loss：同时处理类别不平衡（CB权重）和难易样本不平衡（Focal的γ），论文中效果最好
  ```
  CB Focal Loss = [(1 - β) / (1 - β^n_y)] × [-α_t × (1 - p_t)^γ × log(p_t)]
  ```

- **优点**：
  - 理论上比简单的inverse class frequency更合理——考虑了数据重叠/冗余
  - β提供了一个smooth的控制旋钮，从"不加权"到"完全逆频率"之间平滑过渡
  - 可以和任意loss function组合

- **实践注意**：
  - β需要调参，一般从0.9999开始往小调
  - 在极度不平衡时（如fraud detection中1:10000），CB Loss比简单class weight效果更稳定，因为它不会给少数类过于激进的权重

**各方法适用场景总结**
| 方法 | 适用场景 | 注意事项 |
|---|---|---|
| Class Weight | 通用，第一选择 | 简单有效 |
| Sample Weight | 不同样本重要性不同（如按金额） | 需要业务定义weight |
| Focal Loss | 极度不平衡 + 大量easy sample | 多一个γ需要调 |
| SMOTE | 少数类样本少但质量好 | 只对train set做；高维慎用 |
| Undersampling | 多数类数据量极大时 | 用ensemble减少信息丢失 |
| Cost-Sensitive | 不同错误代价差异大 | 需要业务定义cost matrix |
| Threshold Moving | 模型已训练好，调整决策点 | 最简单，不改模型 |
| CB Loss | 类别数多且不平衡严重 | β需调参；可与Focal Loss组合 |
| PU Learning | 正样本标签不完整 | 详见 Part 2.2 |

**Focal Loss vs Class-Weighted Cross-Entropy的区别**
- Class weight：对所有正样本（或负样本）统一加权，不区分难易
- Focal Loss：除了类别权重外，还根据样本难度动态调权——easy sample权重低，hard sample权重高
- Focal Loss更适合类别极度不平衡且存在大量easy negative的场景（如fraud detection中大部分正常交易模型已经能轻松分对）

---

### 2.5 DNN调参方法论

**Architecture**
- 层数/宽度：从小到大尝试，先2-3层，再逐步增加
- 过深的网络在tabular数据上往往不如浅网络
- 经验法则：tabular数据一般3-5层就足够

**Activation Function**
- ReLU：default选择，简单高效
- LeakyReLU：解决dying ReLU问题
- GELU：Transformer系列中常用，smoother than ReLU

**Regularization**
- Dropout：随机丢弃一定比例的neuron（常用0.1-0.5）
- L2 weight decay：防止权重过大
- BatchNorm：稳定训练过程，加速收敛

**Learning Rate & Scheduler**
- 一般从1e-3开始，根据training curve调整
- Scheduler：warmup + decay（先逐步增大lr，再逐步减小）
- Step decay：每隔N个epoch将lr乘以衰减因子（如0.1）
- Cosine annealing：lr按余弦曲线从初始值逐步降到接近0，周期性重启可以跳出局部最优

**Optimizer详解**

**SGD (Stochastic Gradient Descent)**
```
θ = θ - η × ∇L(θ)
```
- 最基础的优化器，每次用一个mini-batch的梯度更新参数
- 缺点：所有参数用同一个learning rate；在saddle point和ravine（狭长山谷）附近收敛慢、震荡大
- 加Momentum后改善震荡问题：`v = βv + η∇L(θ)`, `θ = θ - v`（β常用0.9）

**Adagrad (Adaptive Gradient)**
```
G_t = G_{t-1} + (∇L)²            （累积历史梯度平方和）
θ = θ - η / (√G_t + ε) × ∇L     （ε ≈ 1e-8 防除零）
```
- 核心思想：对每个参数自适应调整learning rate——更新频繁的参数lr变小，更新稀疏的参数lr保持大
- 优点：适合sparse data（如NLP中的embedding层，大部分词很少出现）
- 缺点：G_t单调递增，lr会持续衰减，训练后期lr趋近于0导致无法继续学习

**RMSProp (Root Mean Square Propagation)**
```
G_t = β × G_{t-1} + (1-β) × (∇L)²     （指数移动平均，β常用0.9）
θ = θ - η / (√G_t + ε) × ∇L
```
- 解决Adagrad的lr持续衰减问题：用指数移动平均代替累积和，只关注近期梯度
- Hinton提出，未正式发表但被广泛使用

**Adam (Adaptive Moment Estimation)**
```
m_t = β₁ × m_{t-1} + (1-β₁) × ∇L        （一阶矩：梯度的均值，β₁常用0.9）
v_t = β₂ × v_{t-1} + (1-β₂) × (∇L)²     （二阶矩：梯度平方的均值，β₂常用0.999）
m̂_t = m_t / (1 - β₁^t)                    （bias correction，解决初始化偏差）
v̂_t = v_t / (1 - β₂^t)
θ = θ - η × m̂_t / (√v̂_t + ε)
```
- 结合了Momentum（一阶矩）和RMSProp（二阶矩）的优点
- 自适应lr + 动量加速，是目前最常用的默认optimizer
- Bias correction解决了训练初期m和v被零初始化导致的估计偏低问题

**AdamW (Adam with Decoupled Weight Decay)**
```
θ = θ - η × (m̂_t / (√v̂_t + ε) + λ × θ)
```
- Adam中的L2 regularization实际上和weight decay不等价（因为Adam对梯度做了自适应缩放，L2的梯度也被缩放了）
- AdamW将weight decay从梯度更新中解耦出来，直接对参数做衰减
- 效果：regularization更一致，泛化性能通常优于Adam + L2
- 是目前训练Transformer等大模型的标准选择

**Optimizer选择建议**
| 场景 | 推荐 | 原因 |
|---|---|---|
| 通用default | Adam | 几乎不需要调参就能work |
| 需要更好泛化 | AdamW | weight decay更合理 |
| Sparse data / NLP embedding | Adagrad / Adam | 自适应lr适合稀疏梯度 |
| 大模型 / Transformer | AdamW | 标准选择 |
| CV经典模型 / 需要极致性能 | SGD + Momentum + Scheduler | 调好后泛化往往最好，但需要仔细调lr |

**Loss**
- 自定义加权：按金额加权，让模型关注高损失样本
- Focal Loss：关注hard sample

**Early Stopping**
- 在validation set上监控metric，当metric不再改善时停止训练
- Patience一般设3-5 epochs

**实践经验**
- 先用LightGBM做baseline，NN尝试beat
- Tabular数据上NN不一定能beat tree model，但在有非结构化数据（text embedding、sequence等）的场景下NN有优势
- 训练时间和可解释性是NN相比tree model的劣势

---

## Part 3: 模拟面试Q&A（面试官视角）

---

### 3.1 项目叙述类 (Behavioral)

**Q1: "Walk me through your most impactful model project end-to-end."**

参考答案：选Stolen Financial项目。

"我做的最有影响力的项目是stolen credit card detection。这个项目的impact直接体现在减少PayPal的chargeback loss上。

整个流程是这样的：首先和业务团队合作，他们会给我们一些当前模型漏掉的case。我们从tagging分析开始——先看这些case能不能被当前tagging逻辑打上标签，如果不行就改进tagging。

有了新tagging后，用已有变量训练baseline。然后iteratively加新变量：卡BIN风险、设备指纹、IP地理异常、行为序列特征、历史dispute率等。

Feature selection用IV筛选、PSI监控稳定性、加上feature importance的permutation方法。最终用LightGBM做baseline，再用NN尝试beat，NN中加入了loss-weighted training让模型更关注高loss的case。"

**Q2: "Tell me about a time you dealt with incomplete labels."**

参考答案：选Buyer AUP Violation项目中的PU Learning。

"在做buyer AUP violation detection时，我们面临的核心问题是正样本标签非常不完整——大量的violating buyer并没有被人工标注。

我们用了PU Learning的spy方法：从confirmed positive中抽一部分放进unlabeled数据中当spy，训练P vs U分类器，然后根据spy的score分布确定阈值，找出reliable negatives。

效果验证：我们把model score较高的unlabeled sample拿给人工review，在某个threshold上有95%被确认为正样本。这说明PU Learning成功帮我们找到了大量之前漏掉的违规用户。"

**Q3: "How did you decide between LightGBM and DNN?"**

参考答案：

"我们的标准做法是先用LightGBM做baseline。LightGBM的优势是：不需要处理categorical变量、训练快、可解释性强、在tabular数据上通常表现很好。

然后尝试用DNN去beat这个baseline。DNN的优势场景是：
- 有非结构化数据需要整合（text embedding、behavior sequence等）
- 数据量足够大
- 需要end-to-end learning

实践中我们发现，纯tabular数据上DNN不一定能beat LightGBM，但当我们把text embedding、action sequence等整合进来时，DNN有明显优势。

最终选择还要考虑：训练时间、inference latency、可解释性需求、团队维护成本。"

**Q4: "Describe how you collaborated with business stakeholders on tagging."**

参考答案：

"以stolen financial为例。业务团队是我们最重要的合作伙伴——他们是domain expert，能告诉我们当前模型漏掉了什么样的case。

协作流程是：
1. 业务团队给我们漏掉的case样本
2. 我们分析这些case的特征，判断是tagging问题还是模型问题
3. 如果是tagging问题，我们一起讨论怎么调整tagging逻辑
4. 新tagging出来后，先review一批sample确保质量
5. 模型上线后，定期和业务团队review false positive和false negative

这个合作模式的关键是让业务团队理解模型的能力边界，同时让我们理解业务的真实需求。"

---

### 3.2 技术深挖类 (Technical Deep-Dive)

**Q5: "Explain PU Learning. When would you use it vs standard semi-supervised?"**

参考答案：详见 Part 2.2。核心区别：
- PU Learning：没有负样本标签，需要从unlabeled中识别reliable negatives
- Semi-supervised：有少量正负样本标签，利用unlabeled数据的分布信息
- PU Learning适用于"能确认正样本但无法确认负样本"的场景，如fraud detection

**Q6: "How does your LSTM encoder-decoder anomaly detection work? How do you set the threshold?"**

参考答案：

"我们用LSTM encoder-decoder来处理用户行为序列。输入是用户的click、page view、停留时长等action组成的sequence。

首先用Word2Vec给每个action做embedding（这比随机初始化更稳定），然后输入LSTM encoder得到sequence embedding，再通过decoder重构原始sequence。

Anomaly detection基于重构误差：正常行为模式的重构误差低，异常行为（bot、collusion）的重构误差高。

Threshold设定：
- 在正常用户群体上计算重构误差的分布
- 根据业务需求选择percentile作为threshold（比如99th percentile）
- 也可以结合业务review来calibrate：选一个threshold，review一批case，根据precision调整
- 实践中会设多个threshold对应不同的action（自动block vs 人工review）"

**Q7: "Why Word2Vec for buyer embeddings instead of directly using transaction features?"**

参考答案：

"直接用transaction features是point-in-time的信息——某个buyer在某个时间点的交易金额、商品类别等。但Word2Vec embedding捕捉的是buyer之间的关系网络。

我们的做法是把商家的所有交易对象当作一个sentence，每个buyer是一个word。这样Word2Vec学到的是：经常出现在同一个商家客群中的buyer会有相似的embedding。

好处是：
- 捕捉了buyer的network behavior，而不仅仅是individual behavior
- 隐式地学习了buyer的风险关联——和已知bad buyer去同类商家的buyer会被拉近
- 这个signal和traditional features是互补的

效果：KMeans聚类后的cluster特征作为额外变量，被证明对模型有正面贡献。"

**Q8: "How would you explain a model's decision to a non-technical stakeholder?"**

参考答案：详见 Part 2.3。

"我们维护了一套surrogate model方案。核心思路是：

1. 选一批业务含义清晰的变量（比如'近30天交易次数'而不是'embedding dimension 47'）
2. 训练一个树模型去模拟production model的output
3. 对具体case用SHAP计算每个变量的贡献

展示给stakeholder的是：'这个用户被标记为高风险，主要原因是：IP国家与注册国家不一致（贡献最大）、近7天交易金额异常高（第二大贡献）...'

这种方式让stakeholder能理解决策逻辑，也能帮我们发现模型的问题——如果top reason不make sense，可能模型学到了错误的pattern。"

**Q9: "How do you handle concept drift in fraud models?"**

参考答案：

"Fraud领域concept drift非常严重——fraudster会不断改变策略来绕过模型。我们的应对方式：

**监控层面：**
- 定期计算feature PSI，监控变量分布偏移
- 监控model score的分布变化
- 跟踪precision/recall随时间的变化

**应对方式：**
- 定期retrain：用最近的数据重新训练模型
- Tagging更新：和业务团队合作识别新的fraud pattern，更新tagging逻辑
- Feature迭代：针对新的fraud trend设计新变量
- 双模型策略：保持一个稳定的base model + 一个频繁更新的incremental model

**量化 Drift Detection 方法：**
- **ADWIN (Adaptive Windowing)**：自动检测数据流中统计特性的变化。维护一个可变长度的窗口，当窗口内两个子窗口的均值差异超过统计显著性阈值时，判定发生 drift 并缩短窗口。优点是不需要预设窗口大小，能自适应 drift 的速度
- **DDM (Drift Detection Method)**：监控模型 error rate 的变化。维护 error rate 的均值 $p$ 和标准差 $s = \sqrt{p(1-p)/n}$。当 $p + s$ 超过历史最小值 + 2s 时发出 warning，超过 + 3s 时确认 drift。适合监控 online learning 场景
- **Page-Hinkley Test**：基于累积和（CUSUM）的检测方法，监控均值的持续偏移。计算累积偏差 $m_T = \sum_{t=1}^{T}(x_t - \bar{x}_T - \delta)$，当 $M_T - m_T > \lambda$ 时判定 drift（$\delta$ 是容忍的最小偏移量，$\lambda$ 是检测阈值）。对 gradual drift 比 DDM 更敏感
- **实践建议：** 在我们的场景中，PSI 用来监控 feature distribution drift（定期 batch 计算），Page-Hinkley 或 ADWIN 用来监控 model performance drift（error rate 或 score distribution 的持续偏移）。两者结合可以区分是 "数据分布变了" 还是 "模型性能退化了"

最重要的其实是和业务团队的紧密合作——他们能最快发现新的fraud trend。"

**Q10: "Walk me through your feature selection pipeline."**

参考答案：

"我们的feature selection是多层次的：

1. **IV筛选**：先计算每个变量的Information Value，IV < 0.02的直接剔除
2. **PSI检查**：PSI > 0.25的变量要格外注意，可能不稳定
3. **Correlation分析**：高相关性的变量选一个（通常选IV更高的）
4. **Permutation Importance**：用一个简单的NN或树模型，逐个把feature设为空值，看对模型输出的影响。通过多轮循环去掉影响力低的feature
5. **Business Review**：最终入模的变量需要和业务团队review，确保make sense

这个pipeline的好处是既有统计筛选（IV、PSI、correlation）也有模型驱动的筛选（permutation importance），还有人工review来防止data leakage。"

**Q11: "Compare Class-Weighted CE, Focal Loss, and Class-Balanced Loss."**

参考答案：详见 Part 2.4。

"三者都是通过修改loss function来处理类别不平衡，但关注的维度不同：

- **Class-weighted CE**：对所有正样本统一加权（比如weight=100），不区分样本难易程度。权重通常直接用inverse class frequency（1/n_y）。简单直接，是第一选择。

- **Focal Loss**：在class weight基础上，还根据样本confidence动态调权——easy sample（模型已经很confident的）loss被大幅缩小，hard sample保留更多梯度。关注的是**难易维度**。

- **Class-Balanced (CB) Loss**：用effective number of samples `E_n = (1-β^n)/(1-β)` 替代原始样本数来计算权重。核心insight是多数类样本之间有大量重叠/冗余，实际的有效信息量没有样本数那么多。关注的是**数据冗余维度**。

三者可以组合——CB + Focal Loss同时处理类别不平衡（CB权重）和难易不平衡（Focal的γ），论文中效果最好。

| 维度 | Class-Weighted CE | Focal Loss | CB Loss |
|---|---|---|---|
| 类别权重 | inverse frequency | α参数 | effective number |
| 难易区分 | 不区分 | γ参数动态调权 | 不区分 |
| Hyperparameter | 无额外参数 | α, γ | β |
| 适用场景 | 通用baseline | 大量easy sample | 多数类高度冗余 |

选择建议：
- 先试class weight，够用就不加复杂度
- Easy negative很多（如fraud detection中99%正常交易模型轻松分对） → 加Focal Loss
- 类别多且不平衡严重、多数类数据冗余度高 → 用CB Loss
- 两个问题都有 → CB + Focal Loss

注意：Focal Loss如果数据中hard sample主要是noise/mislabel，会让模型过度关注noise。CB Loss的β需要调参，一般从0.9999往小调。"

**Q12: "How do you evaluate a fraud model beyond AUC?"**

参考答案：

"AUC是一个overall metric，但在fraud detection中不够：

- **Precision@K / Recall@K**：在实际运营中，团队只能review固定数量的case，所以precision at specific threshold比overall AUC更重要
- **Dollar-weighted metrics**：按金额加权的precision/recall，因为抓一个$10K的fraud比抓十个$10的更重要
- **False Positive Rate**：误伤好用户的代价很高（用户体验、客户流失），需要控制FPR
- **Capture Rate by Segment**：按不同风险场景（新用户 vs 老用户、不同国家等）拆分看capture rate
- **Time-to-detection**：模型能多快发现fraud，越早发现loss越少
- **Stability over time**：模型performance随时间的变化，用PSI和drift analysis监控
- **Calibration Metrics（概率校准指标）**：
  - **ECE (Expected Calibration Error)**：将预测概率分成 M 个 bin，计算每个 bin 内预测概率均值与实际正样本比例的加权差异：$ECE = \sum_{m=1}^{M} \frac{|B_m|}{n} |acc(B_m) - conf(B_m)|$。ECE 越低说明概率校准越好
  - **Brier Score**：$BS = \frac{1}{n}\sum_{i=1}^{n}(p_i - y_i)^2$，同时衡量 discrimination 和 calibration，分数越低越好。可以分解为 calibration term + resolution term + uncertainty term
  - **为什么重要：** 在 fraud detection 中，model score 不仅用于排序，还用于 threshold 决策。如果模型说 "这个交易有 90% 概率是 fraud"，我们希望在 score=0.9 的交易中确实约有 90% 是 fraud——这就是 calibration 的含义。Uncalibrated 的模型会导致 threshold 选择不准确"

---

### 3.3 系统设计/Trade-off类

**Q13: "Design a stolen credit card detection system from scratch."**

参考答案：

"我会分几个层来设计：

**数据层：**
- 交易数据（金额、商品、时间）
- 用户profile（注册时间、历史行为、设备信息）
- 卡信息（BIN、发卡国家、历史dispute率）
- 外部信号（IP geolocation、device fingerprint）

**Feature Engineering：**
- Real-time features：当前交易的基础特征
- Aggregated features：近N天/N笔交易的统计特征（金额均值、交易频率变化等）
- Network features：和其他高风险账户的关联
- Behavior sequence features：交易行为序列的embedding

**模型层：**
- 第一层：Rule-based filter（高confidence的已知pattern直接block）
- 第二层：ML model（LightGBM + NN ensemble），给出risk score
- 第三层：根据score做routing——低风险通过、中风险人工review、高风险自动block

**反馈循环：**
- Chargeback数据作为delayed label回流
- 人工review结果实时更新tagging
- 定期retrain应对concept drift

**监控：**
- Model score分布监控（PSI）
- Feature drift监控
- 业务metric dashboard（loss rate、FPR、review volume）"

**Q14: "Real-time vs batch scoring — how do you choose?"**

参考答案：

"取决于业务场景：

**Real-time scoring适用于：**
- 需要即时决策的场景（交易时决定是否block）
- Latency要求高（< 100ms）
- Feature主要基于当前transaction + pre-computed aggregates

**Batch scoring适用于：**
- 可以延迟决策的场景（比如定期扫描存量用户）
- 需要复杂feature（graph features、跨用户的aggregate等）
- 模型本身计算量大（deep learning model）

**实践中往往是混合方案：**
- Real-time：简单模型做first-pass筛选
- Near-real-time（几分钟延迟）：更复杂的模型做second-pass
- Batch（每天/每周）：全量扫描，发现慢性风险

Feature层面也是混合的：一些aggregate features预计算好存在feature store中，real-time时直接查；一些需要实时计算的feature在线计算。"

**Q15: "High precision but low recall — business wants both. What do you do?"**

参考答案：

"首先要理解为什么recall低：
1. **标签不完整**：很多positive case没被标注 → PU Learning扩充标签
2. **Feature不够**：模型没有足够的signal来区分这些miss的case → 新增feature，特别是针对miss的pattern
3. **Threshold太高**：降低threshold可以提高recall但precision会下降

实际解决方案：
- **分层策略**：高confidence的自动action，中间层人工review。把business的review capacity利用起来
- **多模型策略**：主模型保持high precision，额外加一个recall-oriented模型专门catch主模型漏掉的case
- **Segment-specific models**：不同segment可能需要不同的precision-recall trade-off
- **和业务量化trade-off**：一个false positive的成本 vs 一个false negative的成本（miss一个$10K的fraud vs 误封一个好用户），用expected cost来选最优threshold"

**Q16: "How do you handle the feedback loop in fraud detection?"**

参考答案：

"Fraud detection中有一个经典的feedback loop问题：模型只能学习被模型flag的case的outcome，没有被flag的case我们不知道是不是fraud。

这会导致：
- 模型越来越只能抓到和历史fraud类似的pattern
- 新型fraud pattern可能被持续漏掉
- 训练数据有selection bias

应对方式：
- **Random holdout / Exploration**：对一小部分交易不做模型干预，观察自然outcome。这样可以得到unbiased的label（但有一定的financial cost）
- **Delayed labels**：利用chargeback等delayed signal来补充label——有些被模型放过的transaction最终也会通过chargeback暴露出来
- **Business escalation**：和业务团队合作，他们通过其他渠道（客户投诉、manual review等）发现的fraud case也要回流到训练数据
- **Counterfactual reasoning**：训练时考虑模型的干预效果，用propensity score等方法calibrate
- **定期retrain + 多数据源**：不仅依赖模型flag的数据，还要整合多种数据源来丰富训练集"

---

### 3.4 统计严谨性与实验设计类 (Statistical Rigor & Experiment Design)

---

**Q17: "模型的概率输出校准如何？Calibration curve 怎么看？"**

参考答案：

"模型校准（Calibration）回答的是这个问题：当模型输出 score=0.8 时，实际上有多大比例是 fraud？

**为什么 calibration 在 fraud detection 中特别重要：**
- 我们用 model score 做 threshold 决策——score > 0.9 自动 block、0.7-0.9 人工 review、< 0.7 放行。这些 threshold 的设定隐含了一个假设：score 反映真实概率
- 如果模型 uncalibrated（比如所有 fraud 的 score 都挤在 0.6-0.7 之间），threshold 选择就会出问题
- 业务做 expected loss 计算时也依赖概率的准确性：expected loss = P(fraud) × transaction amount

**Reliability Diagram（校准曲线）：**
- X 轴是预测概率（分 bin），Y 轴是实际正样本比例
- 完美校准 = 对角线。曲线在对角线上方 = under-confident，下方 = over-confident
- 看的时候不仅看偏离程度，还要看每个 bin 的样本量——样本少的 bin 偏离大可能只是 noise

**ECE (Expected Calibration Error)：**
```
ECE = Σ (|B_m| / n) × |acc(B_m) - conf(B_m)|
```
- 将 predicted probability 分成 M 个 bin（通常 M=10 或 15）
- 每个 bin 内计算：实际正样本比例 (accuracy) vs 平均预测概率 (confidence) 的差异
- 加权平均得到 ECE。一般 ECE < 0.05 认为校准较好

**校准方法：**
- **Platt Scaling：** 在 validation set 上拟合一个 logistic regression：$p_{calibrated} = \sigma(a \cdot f(x) + b)$，学习两个参数 a 和 b。适用于 sigmoid 形状的偏差（树模型常见）
- **Isotonic Regression：** 非参数方法，用分段常数函数拟合 score → probability 的映射。更灵活，但需要更多数据，容易过拟合
- **Temperature Scaling：** 在 logit 上除以一个 temperature T：$p_{calibrated} = \sigma(f(x) / T)$，只学一个参数，对 NN 模型效果好

**实践建议：**
- 树模型（LightGBM）输出的概率通常 poorly calibrated——叶子节点的比例不等于概率。建议用 Platt Scaling 校准
- NN 模型在 overfit 时会 over-confident，Focal Loss 训练的模型也会 miscalibrated（因为改变了 loss landscape）
- **校准必须在 held-out calibration set 上做**，不能用 training set，否则会过拟合
- 监控中应定期检查 ECE，calibration 也会 drift"

> **Follow-up 提示：** 面试官可能追问 "Platt Scaling 和 Temperature Scaling 有什么区别？"、"校准后 AUC 会变吗？"（答案：不会，calibration 是单调变换，不改变排序）

---

**Q18: "时序交叉验证——怎么避免 temporal leakage？"**

参考答案：

"在 fraud detection 中，temporal leakage 是一个容易被忽视但影响很大的问题。

**什么是 Temporal Leakage：**
- 用未来的信息去预测过去。比如用 12 月的数据训练模型去预测 10 月的交易——模型可能学到了 '12 月才暴露的 fraud pattern' 来回判 10 月的交易，这在 production 中不可能发生
- Random split 的 CV 会导致这个问题——训练集和测试集在时间上交叉，模型 performance 被高估

**正确的做法——Temporal Split：**
```
Train: Month 1-6  →  Validate: Month 7  →  Test: Month 8
```
- 训练集在前，测试集在后，严格按时间分割
- 模拟 production 场景：模型只能看到过去的数据

**Walk-Forward Validation：**
```
Fold 1: Train [M1-M3]  →  Test [M4]
Fold 2: Train [M1-M4]  →  Test [M5]
Fold 3: Train [M1-M5]  →  Test [M6]
...
```
- 每次测试窗口向前滑动，训练集逐步扩大（expanding window）
- 也可以用 sliding window（固定大小训练窗口），适合 concept drift 严重的场景
- 最终 performance = 所有 fold 的平均值

**Feature Engineering 中的 Look-ahead Bias：**
- 计算 aggregate feature（如 '近 30 天平均交易金额'）时，必须确保用的是 observation date 之前的数据
- 容易出错的地方：用了全局统计量（如 feature 的 mean/std）做归一化，但这个统计量包含了未来数据
- **解决：** 在 pipeline 中严格按 observation date 截断数据，所有 aggregation 只能 look backward

**Fraud Label Delay 的影响：**
- Chargeback 通常 30-90 天后才确认，这意味着最近的数据标签不完整
- **Maturation Window：** 只用至少 90 天前的数据做训练，确保 label 已 matured
- **实践中的 tradeoff：** maturation window 太长会导致训练数据不够新鲜，太短会引入 label noise
```
数据时间线：
|----训练数据----|--maturation gap--|--observation point
  确认的 label    label 未成熟        预测时间点
```

**Random Split 导致的 Performance 高估：**
- 我们做过对比：random 5-fold CV 的 AUC 比 temporal split 的 AUC 高 3-5 个百分点
- 这说明 random split 确实泄露了未来信息，导致对 production performance 的过度乐观估计"

> **Follow-up 提示：** 面试官可能追问 "expanding window vs sliding window 怎么选？"、"label delay 时 validation set 怎么处理？"

---

**Q19: "特征交互效应怎么捕捉？Tree model vs NN 在这方面的差异？"**

参考答案：

"Feature interaction 是指两个或多个特征组合起来产生的 signal，单独看每个特征时看不到。

**Tree Model 的自动交互捕捉：**
- 决策树通过分裂（split）天然捕捉交互：先按 feature A 分裂，再按 feature B 分裂，就等价于学到了 A × B 的交互
- GBDT（如 LightGBM）通过 boosting 逐步学习残差，能捕捉高阶交互
- **优势：** 不需要显式定义交互特征，模型自动发现
- **局限：** 每棵树的深度有限（通常 6-8 层），所以能捕捉的交互阶数也有限（每条路径最多涉及 depth 个特征）

**NN 的隐式交互：**
- 全连接层通过矩阵乘法 + 非线性激活，理论上能学习任意阶交互
- 但在 tabular 数据上，NN 经常不如 tree model——因为 NN 对所有交互一视同仁地去学，没有 tree 那种 greedy 的特征选择能力
- NN 需要更多数据来学到 tree 能用少量数据发现的交互 pattern

**显式交叉网络 (DCN - Deep & Cross Network)：**
- Cross Network 层显式地做特征交叉：$x_{l+1} = x_0 \cdot x_l^T \cdot w_l + b_l + x_l$
- 每一层增加一阶交互（1 层 = 2 阶交互，2 层 = 3 阶交互）
- 和 DNN 并行组合：Cross Network 捕捉显式交互，DNN 捕捉隐式交互
- 在推荐系统中效果很好，但在 fraud detection 中提升有限——因为 tabular 数据的交互不如 user-item 交互那么结构化

**Tabular 数据上 Tree Model 经常赢的原因：**
- Tree model 天然适合处理不规则的特征空间（不同 feature 量级不同、分布不同）
- NN 对 feature scaling 敏感，需要仔细的预处理
- Tree model 的 greedy split 是一种隐式的 feature selection，自动忽略 noise feature
- Tree model 对 categorical feature 处理更自然（LightGBM 的 optimal split）
- 最近的研究（Grinsztajn et al., 2022）做了大规模 benchmark，结论是 tree model 在 medium-sized tabular dataset 上仍然 SOTA

**实践建议：**
- Tabular 数据首选 LightGBM/XGBoost
- NN 的优势场景：(1) 数据量很大（>百万级）(2) 有 sequential 或 unstructured 数据需要整合 (3) 需要 end-to-end 学习
- 混合方案：用 tree model 的 leaf index 作为 NN 的 input feature（Facebook 的经典做法），兼得两者优势"

> **Follow-up 提示：** 面试官可能追问 "TabNet 等 attention-based tabular model 怎么看？"、"有没有用 feature interaction 来做 feature engineering？"

---

**Q20: "模型上线后怎么做 monitoring？什么信号触发 retrain？"**

参考答案：

"模型上线后的 monitoring 分为三个层面：输入数据监控、模型输出监控、业务指标监控。

**1. 输入数据监控（Feature Drift）：**
- **PSI (Population Stability Index)**：每个 feature 定期计算 PSI，PSI > 0.25 触发 alert
- **Missing Rate 变化**：某个 feature 的缺失率突然增加，可能是上游数据源出了问题
- **Data Volume 变化**：日交易量的突然增减可能预示着 population 变化

**2. 模型输出监控（Score Distribution）：**
- **Score PSI**：监控 model score 分布的变化
- **Score 均值/方差的时间趋势**：score 均值持续上升或下降都需要关注
- **各 decision bucket 的 volume**（如高风险/中风险/低风险的占比变化）

**3. 业务指标监控（Performance Metrics）：**
- **Precision@K 随时间的变化**：precision 下降意味着 false positive 增多
- **Recall 的代理指标**：由于 fraud label 有 delay，用 "matured label 上的 recall" 做延迟监控
- **Dollar-weighted capture rate**：按金额加权的捕获率
- **False positive rate by segment**：不同 segment 的误伤率

**Dashboard 设计：**
```
┌─────────────────────────────────────────────────┐
│  Feature Drift Panel  │  Score Distribution Panel│
│  - Top 5 PSI 变化      │  - Score histogram 对比   │
│  - Missing rate trend  │  - Score PSI              │
├─────────────────────────────────────────────────┤
│  Performance Panel    │  Business Impact Panel    │
│  - Precision@K trend  │  - Net loss saving        │
│  - Recall (matured)   │  - Review volume          │
│  - AUC trend          │  - Customer complaint     │
└─────────────────────────────────────────────────┘
```

**Retrain 触发条件（三级机制）：**
- **定期 retrain（routine）：** 固定周期（如每月）用最新数据 retrain，保持模型的 freshness
- **Alert-driven retrain：** 当 PSI > 0.25 或 precision@K 下降超过 5% 时触发
- **Event-driven retrain：** 当业务团队报告新的 fraud trend 或 tagging 发生重大变更时

**Build vs Detect 的 tradeoff：**
- 频繁 retrain（build）可以保持模型新鲜，但有工程成本和 regression 风险
- 只在检测到 drift 时才 retrain（detect）更高效，但检测本身有延迟
- 实践中取折中：定期 retrain（如月度）作为 baseline，加上 alert-driven 的紧急 retrain 机制"

> **Follow-up 提示：** 面试官可能追问 "retrain 后怎么验证新模型比旧模型好？"、"有没有做 shadow mode 部署？"

---

**Q21: "在 fraud detection 中如何处理 delayed label 问题？"**

参考答案：

"Delayed label 是 fraud detection 中最重要但最容易被低估的挑战之一。

**问题本质：**
- 交易发生时，我们不知道它是否是 fraud
- Chargeback（持卡人争议）是最可靠的 fraud label，但通常在交易后 30-90 天才发生
- 有些 fraud 甚至永远不会被发现（持卡人没有注意到）

**Maturation Window（标签成熟窗口）：**
```
交易时间: Day 0
Chargeback 窗口: Day 1 - Day 90
Maturation point: Day 90（此后该交易的 label 基本确定）
```
- 只有过了 maturation window 的数据，label 才被认为是 reliable 的
- 训练集应该只包含 matured 数据——如果用未成熟的数据训练，很多 fraud 被错误标记为 non-fraud

**Provisional Labels（临时标签）的使用：**
- 在 maturation window 内，可以用 rule-based 的 provisional label 辅助：
  - 已有 chargeback → 确认 fraud
  - 被其他 model/rule 拦截并确认 → fraud
  - 账号被 investigation team 关闭 → likely fraud
  - 其他 → 暂时标为 non-fraud
- **风险：** Provisional label 有 noise，需要在 label correction 时修正

**Label Correction（标签修正）：**
- 当 matured label 到来时，回头修正之前用 provisional label 训练的模型
- 周期性 retrain 时用 matured label 替换 provisional label
- 对于 never-discovered fraud（永远没有 chargeback），只能接受这个 noise

**Training Window 选择：**
- 太近（如只用最近 1 个月的 matured 数据）：数据量少，不够代表
- 太远（如用过去 2 年的数据）：old pattern 可能已经过时，concept drift 严重
- 经验法则：6-12 个月的 matured 数据，加上 sliding window 逐步更新
- 还要考虑季节性：至少覆盖一个完整的 business cycle

**Evaluation Window 选择：**
- 评估模型效果时，也只能用 matured 数据
- 这意味着 evaluation 有 90 天延迟——模型上线后 90 天才能看到真实 performance
- 中间用 provisional metric 做过渡，matured metric 到来后修正"

> **Follow-up 提示：** 面试官可能追问 "如果 maturation window 缩短到 30 天会怎样？"、"有没有用 early signals 来预测最终 label？"

---

**Q22: "如果让你设计一个 fraud detection 的 A/B test，你会怎么做？"**

参考答案：

"Fraud detection 的 A/B test 和传统的 product A/B test 有本质区别——我们不能让一半用户不受保护。

**核心挑战：**
- **伦理问题：** 不能对 control group 完全不做 fraud detection（用户会受到实际损失）
- **Interference / Spillover：** Fraudster 可能跨 treatment 和 control group 操作（一个 fraud ring 的交易可能分散在两个 group）
- **Label delay：** 效果需要等 90 天才能 mature

**设计方案——Champion-Challenger with Shadow Mode：**
1. **Champion model**（current）对所有交易做决策——这保证了用户保护
2. **Challenger model**（new）对同样的交易做 shadow scoring——只记录结果，不实际 action
3. 比较两个模型的 score 和 decision：
   - Challenger 能 catch 到哪些 Champion 漏掉的 fraud？
   - Challenger 会 miss 哪些 Champion catch 到的 fraud？
   - Challenger 的 false positive rate 如何？

**如果要做 live A/B test（需要管理层批准）：**
- **Randomization unit 选择：** 用 user 而非 transaction 作为随机化单位。因为 fraud 是 user-level 行为，如果按 transaction 随机，同一个 fraudster 的不同交易可能被不同模型处理，干扰评估
- **Risk-bounded design：**
  - 只在 **低风险段** 做 randomization——对于 Champion model score > 0.9 的交易，两组都 block（不拿高风险交易做实验）
  - 在 0.3 < score < 0.7 的 "灰色地带" 做 A/B test——这部分 challenger 可能表现更好
  - score < 0.3 的两组都放行
- **样本量计算：** 由于 fraud 的 base rate 很低（<1%），需要很大的样本量才有 statistical power
  - 用 McNemar test（paired）而非 independent two-sample test，因为两组面对的是相同的 fraud population
  - 如果 base rate = 0.5%，想检测 10% 的 relative improvement，power = 0.8，α = 0.05，大约需要 100K+ 交易

**评估指标：**
- **Primary：** Net loss saving = blocked fraud $ − false positive $ harm
- **Secondary：** Precision@K、Recall、FPR、Average detection speed
- **Guardrail：** Customer complaint rate、false positive rate 不能显著上升

**Spillover 的处理：**
- 如果用 user-level 随机化，同一个 fraud ring 的 accomplice 可能在不同组——但这其实是现实反映
- 可以做 network-aware randomization（把有 linking 关系的 user 分到同一组），但实施复杂

**实践建议：** 在大多数情况下，shadow mode + backtest 足以评估新模型，live A/B test 只在 shadow mode 结果 inconclusive 时才做。live A/B test 的 ethical 和 operational overhead 很大。"

> **Follow-up 提示：** 面试官可能追问 "shadow mode 有什么局限？"（答案：无法评估 'challenger block 后 fraudster 的反应'——因为实际上没有被 block）

---

**Q23: "Ensemble Methods——有没有尝试 stacking 或 blending？"**

参考答案：

"有，我们尝试过 LightGBM + NN 的 stacking，也探索过 blending。

**Stacking（堆叠）架构：**
```
Layer 0 (Base Models):
  - LightGBM (tabular features)
  - NN (tabular + embedding features)
  - 可选: Logistic Regression (linear baseline)

Layer 1 (Meta-learner):
  - 用 Layer 0 各 model 的 out-of-fold predictions 作为 input
  - 训练一个简单的 meta-learner (通常 Logistic Regression 或轻量级 GBDT)
  - Meta-learner 学习 '什么时候该信 LightGBM，什么时候该信 NN'
```

**实现细节：**
- **Out-of-fold prediction：** 为了避免 information leakage，Layer 0 的 predictions 必须用 K-fold 交叉验证生成——每条样本的预测值来自未见过该样本的模型
- **时序场景的调整：** 在我们的场景中用 walk-forward split 替代 K-fold，避免 temporal leakage
- **Meta-learner 复杂度：** 越简单越好。如果 meta-learner 太复杂，容易过拟合 base model 的 bias

**Blending vs Stacking：**
- **Blending：** 把数据分成 train + blending set，Layer 0 在 train 上训练，在 blending set 上生成 prediction，meta-learner 在 blending set 上训练。比 stacking 简单但浪费数据
- **简单加权平均：** score = α × LightGBM_score + (1-α) × NN_score，在 validation set 上 grid search α。最简单但不如 stacking 灵活

**实际效果与 Trade-off：**
- Stacking 比单一最佳模型（LightGBM）AUC 提升约 0.5-1 个百分点
- 但在 **production 中的 trade-off** 很重要：
  - **Complexity：** 维护多个模型的 pipeline（训练、部署、监控都翻倍）
  - **Latency：** 多个模型串行 inference 增加延迟
  - **Debugging：** 出问题时需要查看每个 base model 和 meta-learner
  - **Marginal gain vs overhead：** 0.5-1% AUC 的提升是否值得 2x 的工程复杂度？

**什么时候 ensemble 值得：**
- Base models 之间 **diversity 高** 的时候——LightGBM 和 NN 看到不同的 pattern，ensemble 才有价值
- 可以用 error correlation 量化 diversity：如果两个模型犯同样的错误，ensemble 帮助不大
- 竞赛中必用 ensemble，production 中需要权衡

**实践建议：**
- 如果 LightGBM 和 NN 的 performance 差距大（如一个 AUC=0.90，另一个 0.85），直接用好的那个
- 如果差距小且有 complementary errors，用 stacking
- 对 latency 敏感的 real-time 场景，用简单的加权平均代替 stacking"

> **Follow-up 提示：** 面试官可能追问 "base model 的 diversity 怎么量化？"、"online ensemble 怎么做？"
