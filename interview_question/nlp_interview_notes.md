# NLP Interview Notes

**Comprehensive Natural Language Processing Interview Preparation Guide**

*Aggregated from 2 markdown files covering NLP text representation and deep learning models.*

---

## Table of Contents

### 1. Text Representation
- [1.1 Text Representation Methods](#11-text-representation-methods)

### 2. Deep Learning Models for NLP
- [2.1 Deep Learning Models](#21-deep-learning-models)

---


# 1. Text Representation

## 1.1 Text Representation Methods


## 1.1 模型概述

Word2Vec是Google在2013年提出的一个NLP工具，它通过一个浅层的双层神经网络，高效率、高质量地将海量单词向量化。训练得到的词向量满足：

- 相似单词的词向量彼此接近。例如$\text{dis}(\vec V(\text{man}),\vec V(\text{woman})) \ll \text{dis}(\vec V(\text{man}),\vec V(\text{computer}))$
- 保留单词间的线性规则性。例如$\vec V(\text{king})-\vec V(\text{man})+\vec V(\text{woman})\approx \vec V(\text{queen})$

Word2Vec模型的灵感来源于Bengio在2003年提出的NNLM模型（Nerual Network Language Model），该模型使用一个三层前馈神经网络$f(w_k,w_{k-1},w_{k-2},...,w_{k-n+1};\theta)$来拟合一个词序列的条件概率$P(w_k|w_{k-1},w_{k-2},...,w_1)$。第一层是映射层，通过一个共享矩阵，将One-Hot向量转化为词向量，第二层是一个激活函数为tanh的隐含层，第三层是Softmax输出层，将向量映射到$[0,1]$概率空间中。根据条件概率公式与大数定律，使用词频$\frac{\text{Count}(w_k,w_{k-1},w_{k-2},...,w_{k-n+1})}{\text{Count}(w_{k-1},w_{k-2},...,w_{k-n+1})}$来近似地估计真实的条件概率。

<img src="/images/NNLM.png" alt="NNLM" style="zoom: 50%;" />

Bengio发现，我们可以使用映射层的权值作为词向量表征。但是，由于参数空间非常庞大，NNLM模型的训练速度非常慢，在百万级的数据集上需要耗时数周才能得到相对不错的结果，而在千万级甚至更大的数据集上，几乎无法得到结果。

Mikolov发现，NNLM模型可以被拆分成两个步骤：

- 用一个简单的模型训练出一个连续的词向量（映射层）
- 基于词向量表征，训练出一个N-Gram神经网络模型（隐含层+输出层）

而模型的计算瓶颈主要在第二步，特别是输出层的Sigmoid归一化部分。如果我们只是想得到词向量，可以对第二步的神经网络模型进行简化，从而提高模型的训练效率。因此，Mikolov对NNLM模型进行了以下几个部分的修改：

- 舍弃了隐含层。
- NNLM在利用上文词预测目标词时，对上文词的词向量进行了拼接，Word2Vec模型对其直接进行了求和，从而降低了隐含元的维度。
- NNLM在进行Sigmoid归一化时需要遍历整个词汇表，Word2Vec模型提出了Hierarchical Softmax与Negative Sampling两种策略进行优化。
- 依据分布式假设（上下文环境相似的两个词有着相近的语义），将下文单词也纳入训练环境，并提出了两种训练策略，一种是用上下文预测中心词，称为CBOW，另一种是用中心词预测上下文，称为Skip-Gram。

<img src="/images/Word2Vec.png" alt="Word2Vec" style="zoom:40%;" />

## 1.2 CBOW模型

假设我们的语料是**"NLP is so interesting and challenging"**。循环使用每个词作为中心词，来其上下文词来预测中心词。我们通常使用一个指定长度的窗口，根据马尔可夫性质，忽略窗口以外的单词。

|   中心词    |            上下文            |
| :---------: | :--------------------------: |
|     NLP     |            is, so            |
|     is      |     NLP, so, interesting     |
|     so      |  NLP, is, interesting, and   |
| interesting |   is, so, and, challenging   |
|     and     | so, interesting, challenging |
| challenging |       interesting, and       |

我们的目标是通过上下文来预测中心词，也就是给定上下文词，出现该中心词的概率最大。这和完形填空颇有点异曲同工之妙。也即$\max P(\text{NLP|is, so})*P(\text{is|NLP, so, interesting})*\dots$

用公式表示如下：
$$
\begin{align}
\max\limits_{\theta} L(\theta)&=\prod\limits_{w\in D}p(w|C(w)) \\
&=\sum\limits_{w \in D}\log p(w|C(w))
\end{align}
$$

其中$w$指中心词，$C(w)$指上下文词集，$D$指语料库，也即所有中心词的词集。

问题的核心变成了如何构造$\log p(w|C(w))$。我们知道，NNLM模型的瓶颈在Sigmoid归一化上，Mikolov提出了两种改进思路来绕过Sigmoid归一化这一操作。一种思想是将输出改为一个霍夫曼树，每一个单词的概率用其路径上的权重乘积来表示，从而减少高频词的搜索时间；另一种思想是将预测每一个单词的概率，概率最高的单词是中心词改为预测该单词是不是正样本，通过负采样减少负样本数量，从而减少训练时间。

### 1.2.1 Hierarchical Softmax

### 1.2.2 Negative Sampling

基于Hierachical Softmax的模型使用Huffman树代替了传统的线性神经网络，可以提高模型训练的效率。但是，如果训练样本的中心词是一个很生僻的词，那么在Huffman树中仍旧需要进行很复杂的搜索。负采样方法的核心思想是：设计一个分类器， 对于我们需要预测的样本，设为正样本；而对于不是我们需要的样本，设置成负样本。在CBOW模型中，我们需要预测中心词$w$，因此正样本只有$w$，也即$\text{Pos}(w)=\{w\}$，而负样本为除了$w$之外的所有词。对负样本进行**随机采样**，得到$\text{Neg}(w)$，大大简化了模型的计算。

我们首先将$C(w)$输入映射层并求和得到隐含表征$h_w=\sum\limits_{u \in C(w)}\vec v(u)$

从而，
$$
\begin{align}
p(u|C(w))&=
\begin{cases}
\sigma(h_w^T\theta_u), &\mathcal{D}(w,u)=1 \\
1-\sigma(h_w^T\theta_u), &\mathcal{D}(w,u)=0 \\
\end{cases}\\
&=[\sigma(h_w^T\theta_u)]^{\mathcal{D}(w,u)} \cdot [1-\sigma(h_w^T\theta_u)]^{1-\mathcal{D}(w,u)}
\end{align}
$$

从而，
$$
\begin{align}
\max\limits_{\theta} L(\theta)&=\sum\limits_{w \in D}\log p(w|C(w))\\
&=\sum\limits_{w \in D}\log \prod\limits_{u \in D}p(u|C(w)) \\
&\approx\sum\limits_{w \in D}\log \prod\limits_{u \in \text{Pos(w)}\cup \text{Neg(w)} }p(u|C(w))\\
&=\sum\limits_{w \in D}\log\prod\limits_{u \in \text{Pos(w)}\cup \text{Neg(w)}}[\sigma(h_w^T\theta_u)]^{\mathcal{D}(w,u)} \cdot [1-\sigma(h_w^T\theta_u)]^{1-\mathcal{D}(w,u)} \\
&=\sum\limits_{w \in D}\sum\limits_{u \in \text{Pos}(w)\cup \text{Neg}(w)}\mathcal{D}(w,u)\cdot\log \sigma(h_w^T\theta_u)+[1-\mathcal{D}(w,u)]\cdot \log [1-\sigma(h_w^T\theta_u)]\\
&=\sum\limits_{w \in D}\left\{\sum\limits_{u \in \text{Pos}(w)}\log \sigma(h_w^T\theta_u)+\sum\limits_{u \in \text{Neg}(w)}\log [1-\sigma(h_w^T\theta_u)]\right\}
\end{align}
$$

由于上式是一个最大化问题，因此使用随机梯度上升法对问题进行求解。

令$L(w,u,\theta)=\mathcal{D}(w,u)\cdot\log \sigma(h_w^T\theta_u)+[1-\mathcal{D}(w,u)]\cdot \log [1-\sigma(h_w^T\theta_u)]$

则$\frac{\partial L}{\partial\theta_u}=\mathcal{D}(w,u)\cdot[1-\sigma(h_w^T\theta_u)]h_w+[1-\mathcal{D}(w,u)]\cdot \sigma(h_w^T\theta_u)h_w=[\mathcal{D}(w,u)-\sigma(h_w^T\theta_u)]h_w$

因此$\theta_u$的更新公式为：$\theta_u:=\theta_u+\eta[\mathcal{D}(w,u)-\sigma(h_w^T\theta_u)]h_w$

同样地，$\frac{\partial L}{\partial h_w}=[\mathcal{D}(w,u)-\sigma(h_w^T\theta_u)]\theta_u$

上下文词的更新公式为：$v(\tilde{w}):=v(\tilde{w})+\eta\sum\limits_{u \in \text{Pos}(w)\cup \text{Neg}(w)}[\mathcal{D}(w,u)-\sigma(h_w^T\theta_u)]\theta_u$

## 1.3 Skip-Gram模型

仍旧使用上文的语料库**"NLP is so interesting and challenging"**，这次，我们的目标是通过中心词来预测上下文，也就是给定中心词，出现这些上下文词的概率最大。也即$\max P(is|NLP)*P(so|NLP)*P(NLP|is)*P(so|is)*P(interesting|is)*\dots$

用公式表示如下：
$$
\begin{align}
\max\limits_{\theta} L(\theta)&=\prod\limits_{w\in D}\prod\limits_{c \in C(w)}p(c|w) \\
&=\sum\limits_{w \in D}\sum\limits_{c \in C(w)}\log p(c|w)
\end{align}
$$

### 1.3.1 Hierarchical Softmax

### 1.3.2 Negative Sampling

# 2 常见面试问题

**Q1：介绍一下Word2Vec模型。**

> A：两个模型：CBOW/Skip-Gram
>
> 两种加速方案：Hierarchical Softmax/Negative Sampling

**Q2：Word2Vec模型为什么要定义两套词向量？**

>  A：因为每个单词承担了两个角色：中心词和上下文词。通过定义两套词向量，可以将两种角色分开。cs224n中提到是为了更方便地求梯度。参考见：https://www.zhihu.com/answer/706466139

**Q3：Hierarchial Softmax 和 Negative Sampling对比**

> A：基于Huffman树的Hierarchial Softmax 虽然在一定程度上能够提升模型运算效率，但是，如果中心词是生僻词，那么在Huffman树中仍旧需要进行很复杂的搜索$(O(\log N))$。而Negative Sampling通过随机负采样来提升运算效率，其复杂度和设定的负样本数$K$线性相关$(O(K))$，当$K$取较小的常数时，负采样在每⼀步的梯度计算开销都较小。

**Q4：HS为什么用霍夫曼树而不用其他二叉树？**

> 这是因为Huffman树对于高频词会赋予更短的编码，使得高频词离根节点距离更近，从而使得训练速度加快。

**Q5：Word2Vec模型为什么要进行负采样？**

>  A：因为负样本的数量很庞大，是$O(|V^2|)$。

**Q6：负采样为什么要用词频来做采样概率？**

> 为这样可以让频率高的词先学习，然后带动其他词的学习。

**Q7：One-hot模型与Word2Vec模型比较？**

>  A：One-hot模型的缺点
>
> - 稀疏 Sparsity
> - 只能表示维度数量的单词 Capacity
> - 无法表示单词的语义 Meaning

**Q8：Word2Vec模型在NNLM模型上做了哪些改进？**

> A：相同点：其本质都可以看作是语言模型；
>
> 不同点：词向量只不过 NNLM 一个产物，Word2vec 虽然其本质也是语言模型，但是其专注于词向量本身，因此做了许多优化来提高计算效率：
>
> - 与 NNLM 相比，词向量直接 sum，不再拼接，并舍弃隐层；
>
> - 考虑到 sofmax 归一化需要遍历整个词汇表，采用 hierarchical softmax 和 negative sampling 进行优化，hierarchical softmax 实质上生成一颗带权路径最小的哈夫曼树，让高频词搜索路劲变小；negative sampling 更为直接，实质上对每一个样本中每一个词都进行负例采样；

**Q9：Word2Vec与LSA对比？**

> A：LSA是基于共现矩阵构建词向量，本质上是基于全局语料进行SVD矩阵分解，计算效率低；
>
> 而Word2Vec是基于上下文局部语料计算共现概率，计算效率高。

**Q10：Word2Vec的缺点？**

> 忽略了词语的语序；
>
> 没有考虑一词多义现象

**Q11：怎么从语言模型理解词向量？怎么理解分布式假设？**

> 词向量是语言模型的一个副产物，可以理解为，在语言模型训练的过程中，势必在一定程度上理解了每个单词的含义。而这在计算机的表示下就是词向量。
>
> 分布式假设指的是相同上下文语境的词有似含义。

**参考资料**

word2vec 中的数学原理详解 https://blog.csdn.net/itplus/article/details/37969519

Word2Vec原理介绍 https://www.cnblogs.com/pinard/p/7160330.html

词向量介绍 https://www.cnblogs.com/sandwichnlp/p/11596848.html

一些关于词向量的问题 https://zhuanlan.zhihu.com/p/56382372

一个在线尝试Word2Vec的小demo https://ronxin.github.io/wevi/

---


# 2. Deep Learning Models for NLP

## 2.1 Deep Learning Models



## 知识体系

主要包括深度学习相关的特征抽取模型，包括卷积网络、循环网络、注意力机制、预训练模型等。

### CNN

TextCNN 是 CNN 的 NLP 版本，来自 Kim 的 [[1408.5882] Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

结构如下：

![](http://qnimg.lovevivian.cn/paper-textcnn-1.jpg)

大致原理是使用多个不同大小的 filter（也叫 kernel） 对文本进行特征提取，如上图所示：

- 首先通过 Embedding 将输入的句子映射为一个 `n_seq * embed_size` 大小的张量（实际中一般还会有 batch_size）
- 使用 `(filter_size, embed_size)` 大小的 filter 在输入句子序列上平滑移动，这里使用不同的 padding 策略，会得到不同 size 的输出
- 由于有 `num_filters` 个输出通道，所以上面的输出会有 `num_filters` 个
- 使用 `Max Pooling` 或 `Average Pooling`，沿着序列方向得到结果，最终每个 filter 的输出 size 为 `num_filters`
- 将不同 filter 的输出拼接后展开，作为句子的表征

### RNN

RNN 的历史比 CNN 要悠久的多，常见的类型包括：

- 一对一（单个 Cell）：给定单个 Token 输出单个结果
- 一对多：给定单个字符，在时间步向前时同时输出结果序列
- 多对一：给定文本序列，在时间步向前执行完后输出单个结果
- 多对多1：给定文本序列，在时间步向前时同时输出结果序列
- 多对多2：给定文本序列，在时间步向前执行完后才开始输出结果序列

由于 RNN 在长文本上有梯度消失和梯度爆炸的问题，它的两个变种在实际中使用的更多。当然，它们本身也是有一些变种的，这里我们只介绍主要的模型。

- LSTM：全称 Long Short-Term Memory，一篇 Sepp Hochreiter 等早在 1997 年的论文[《LONG SHORT-TERM MEMORY》](https://www.bioinf.jku.at/publications/older/2604.pdf)中被提出。主要通过对原始的 RNN 添加三个门（遗忘门、更新门、输出门）和一个记忆层使其在长文本上表现更佳。

    ![](https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/LSTM_Cell.svg/1280px-LSTM_Cell.svg.png)

- GRU：全称 Gated Recurrent Units，由 Kyunghyun Cho 等人 2014 年在论文[《Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation》](https://arxiv.org/pdf/1406.1078v3.pdf) 中首次被提出。主要将 LSTM 的三个门调整为两个门（更新门和重置门），同时将记忆状态和输出状态合二为一，在效果没有明显下降的同时，极大地提升了计算效率。

    ![](https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Gated_Recurrent_Unit%2C_base_type.svg/1920px-Gated_Recurrent_Unit%2C_base_type.svg.png)

## Questions

###  CNN相关

#### CNN 有什么好处？

- 稀疏（局部）连接：卷积核尺寸远小于输入特征尺寸，输出层的每个节点都只与部分输入层连接
- 参数共享：卷积核的滑动窗在不同位置的权值是一样的
- 等价表示（输入/输出数据的结构化）：输入和输出在结构上保持对应关系（长文本处理容易）

#### CNN 有什么不足？

- 只有局部语义，无法从整体获取句子语义
- 没有位置信息，丢失了前后顺序信息

#### 卷积层输出 size？

给定 n×n 输入，f×f 卷积核，padding p，stride s，输出的尺寸为：

$$
\lfloor \frac{n+2p-f}{s} + 1 \rfloor \times \lfloor \frac{n+2p-f}{s} + 1 \rfloor
$$

### RNN

#### LSTM 网络结构？

LSTM 即长短时记忆网络，包括三个门：更新门（输入门）、遗忘门和输出门。公式如下：

$$
\hat{c}^{<t>} = \tanh (W_c [a^{<t-1}>, x^{<t>}] + b_c) \\
\Gamma_u = \sigma(W_u [a^{<t-1}>, x^{<t>}] + b_u) \\
\Gamma_f = \sigma(W_f [a^{<t-1}>, x^{<t>}] + b_f) \\
\Gamma_o = \sigma(W_o [a^{<t-1}>, x^{<t>}] + b_o) \\
c^{<t>} = \Gamma_u * \hat{c}^{<t>} + \Gamma_f*c^{<t-1>} \\
a^{<t>} = \Gamma_o * c^{<t>}
$$

#### 如何解决 RNN 中的梯度消失或梯度爆炸问题？

- 梯度截断
- ReLU、LeakReLU、Elu 等激活函数
- Batch Normalization
- 残差连接
- LSTM、GRU 等架构

#### 假设输入维度为 m，输出为 n，求 GRU 参数？

输入  W：3nm，隐层 W：3nn，隐层 b：3n，合计共：`3*(nn+nm+n)`。当然，也有的实现会把前一时刻的隐层和当前时刻的输入分开，使用两个 bias，此时需要再增加 3n 个参数。

#### LSTM 和 GRU 的区别？

- GRU 将 LSTM 的更新门、遗忘门和输出门替换为更新门和重置门
- GRU 将记忆状态和输出状态合并为一个状态
- GRU 参数更少，更容易收敛，但数据量大时，LSTM 效果更好

### Attention

#### Attention 机制

Attention 核心是从输入中有选择地聚焦到特定重要信息上的一种机制。有三种不同用法：

- 在 encoder-decoder attention 层，query 来自上一个 decoder layer，memory keys 和 values 来自 encoder 的 output
- encoder 包含 self-attention，key value 和 query 来自相同的位置，即前一层的输出。encoder 的每个位置都可以注意到前一层的所有位置
- decoder 与 encoder 类似，通过将所有不合法连接 mask 以防止信息溢出

#### 自注意力中为何要缩放？

维度较大时，向量内积容易使得 SoftMax 将概率全部分配给最大值对应的 Label，其他 Label 的概率几乎为 0，反向传播时这些梯度会变得很小甚至为 0，导致无法更新参数。因此，一般会对其进行缩放，缩放值一般使用维度 dk 开根号，是因为点积的方差是 dk，缩放后点积的方差为常数 1，这样就可以避免梯度消失问题。

另外，Hinton 等人的研究发现，在知识蒸馏过程中，学生网络以一种略微不同的方式从教师模型中抽取知识，它使用大模型在现有标记数据上生成软标签，而不是硬的二分类。直觉是软标签捕获了不同类之间的关系，这是大模型所没有的。这里的软标签就是缩放的 SoftMax。

至于为啥最后一层为啥一般不需要缩放，因为最后输出的一般是分类结果，参数更新不需要继续传播，自然也就不会有梯度消失的问题。

### Transformer

#### Transformer 中为什么用 Add 而不是 Concat？

在 Embedding 中，Add 等价于 Concat，三个 Embedding 相加与分别 One-Hot Concat 效果相同。

### ELMO

#### 简单介绍下ELMO

使用双向语言模型建模，两层 LSTM 分别学习语法和语义特征。首次使用两阶段训练方法，训练后可以在下游任务微调。

Feature-Based 微调，预训练模型作为纯粹的表征抽取器，表征依赖微调任务网络结构适配（任务缩放因子 γ）。

### ELMO的缺点

ELMO 的缺点主要包括：不完全的双向预训练（Bi 是分开的，仅在 Loss 合并）；需要进行任务相关的网络设计（每种下游任务都要特定的设计）；仅有词向量无句向量（没有句向量任务）。


### GPT

#### 简单介绍下GPT

使用 Transformer 的 Decoder 替换 LSTM 作为特征提取器。

Model-Based 微调，预训练模型作为任务网络的一部分参与任务学习，简化了下游任务架构设计。

#### GPT的缺点

GPT 的缺点包括：单项预训练模型；仅有词向量无句向量（仅学习语言模型）。

### BERT

#### 简单介绍下BERT

使用 Transformer Encoder 作为特征提取器，交互式双向语言建模（MLM），Token 级别+句子级别任务（MLM+NSP），两阶段预训练。

Feature-Based 和 Model-Based，实际一般使用 Model-Based。

#### BERT缺点

BERT 的缺点是：字粒度难以学到词、短语、实体的完整语义。

### ERNIE

#### ERNIE对BERT进行了哪些优化？

对 BERT 的缺点进行了优化，Mask 从字粒度的 Token 修改为完整的词或实体。ERNIE2.0 引入更多的预训练任务以捕捉更丰富的语义知识。





---

**End of NLP Interview Notes**

*This comprehensive guide contains the complete content from 2 markdown files:*
- *Text Representation Methods*
- *Deep Learning Models for NLP*

*Total: 2 comprehensive NLP interview preparation documents aggregated into one file.*

