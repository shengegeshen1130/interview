# Machine Learning Complete

## 目录
- [1. Liner Regression (线性回归)](#1-liner-regression-线性回归)
- [2. Logistics Regression (逻辑回归)](#2-logistics-regression-逻辑回归)
- [3. Decision Tree (决策树)](#3-decision-tree-决策树)
- [3.1 Random Forest (随机森林)](#31-random-forest-随机森林)
- [3.2 GBDT](#32-gbdt)
- [3.3 XGBoost](#33-xgboost)
- [3.4 LightGBM](#34-lightgbm)
- [4. SVM (支持向量机)](#4-svm-支持向量机)
- [5.1 Bayes Network (贝叶斯网络)](#51-bayes-network-贝叶斯网络)
- [5.2 Markov (马尔可夫)](#52-markov-马尔可夫)
- [5.3 Topic Model (主题模型)](#53-topic-model-主题模型)
- [6. EM算法](#6-em算法)
- [7. Clustering (聚类)](#7-clustering-聚类)
- [8. ML特征工程和优化方法](#8-ml特征工程和优化方法)
- [9. KNN](#9-knn)

---

# 1. Liner Regression (线性回归)

## 目录
- [1.什么是线性回归](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/Liner%20Regression#1什么是线性回归)
- [2. 能够解决什么样的问题](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/Liner%20Regression#2-能够解决什么样的问题)
- [3. 一般表达式是什么](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/Liner%20Regression#3-一般表达式是什么)
- [4. 如何计算](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/Liner%20Regression#4-如何计算)
- [5. 过拟合、欠拟合如何解决](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/Liner%20Regression#5-过拟合欠拟合如何解决)
  - [5.1 什么是L2正则化(岭回归)](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/Liner%20Regression#51-什么是l2正则化岭回归)
  - [5.2 什么场景下用L2正则化](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/Liner%20Regression#52-什么场景下用l2正则化)
  - [5.3 什么是L1正则化(Lasso回归)](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/Liner%20Regression#53-什么是l1正则化lasso回归)
  - [5.4 什么场景下使用L1正则化](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/Liner%20Regression#54-什么场景下使用l1正则化)
  - [5.5 什么是ElasticNet回归](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/Liner%20Regression#55-什么是elasticnet回归)
  - [5.6 ElasticNet回归的使用场景](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/Liner%20Regression#56--elasticnet回归的使用场景)
- [6. 线性回归要求因变量服从正态分布？](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/Liner%20Regression#6-线性回归要求因变量服从正态分布)
- [7. 代码实现](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/Liner%20Regression/demo)

## 1.什么是线性回归

- 线性：两个变量之间的关系**是**一次函数关系的——图象**是直线**，叫做线性。
- 非线性：两个变量之间的关系**不是**一次函数关系的——图象**不是直线**，叫做非线性。
- 回归：人们在测量事物的时候因为客观条件所限，求得的都是测量值，而不是事物真实的值，为了能够得到真实值，无限次的进行测量，最后通过这些测量数据计算**回归到真实值**，这就是回归的由来。

## 2. 能够解决什么样的问题

对大量的观测数据进行处理，从而得到比较符合事物内部规律的数学表达式。也就是说寻找到数据与数据之间的规律所在，从而就可以模拟出结果，也就是对结果进行预测。解决的就是通过已知的数据得到未知的结果。例如：对房价的预测、判断信用评价、电影票房预估等。

## 3. 一般表达式是什么

![](https://latex.codecogs.com/gif.latex?Y=wx+b)

w叫做x的系数，b叫做偏置项。

## 4. 如何计算

### 4.1 Loss Function--MSE

![](https://latex.codecogs.com/gif.latex?J=\frac{1}{2m}\sum^{i=1}_{m}(y^{'}-y)^2)

利用**梯度下降法**找到最小值点，也就是最小误差，最后把 w 和 b 给求出来。

## 5. 过拟合、欠拟合如何解决

使用正则化项，也就是给loss function加上一个参数项，正则化项有**L1正则化、L2正则化、ElasticNet**。加入这个正则化项好处：

- 控制参数幅度，不让模型"无法无天"。
- 限制参数搜索空间
- 解决欠拟合与过拟合的问题。

### 5.1 什么是L2正则化(岭回归)

方程：

![](https://latex.codecogs.com/gif.latex?J=J_0+\lambda\sum_{w}w^2)

![](https://latex.codecogs.com/gif.latex?J_0)表示上面的 loss function ，在loss function的基础上加入w参数的平方和乘以 ![](https://latex.codecogs.com/gif.latex?\lambda) ，假设：

![](https://latex.codecogs.com/gif.latex?L=\lambda({w_1}^2+{w_2}^2))

回忆以前学过的单位元的方程：

![](https://latex.codecogs.com/gif.latex?x^2+y^2=1)

正和L2正则化项一样，此时我们的任务变成在L约束下求出J取最小值的解。求解J0的过程可以画出等值线。同时L2正则化的函数L也可以在w1w2的二维平面上画出来。如下图：

![image](https://wx4.sinaimg.cn/large/00630Defgy1g4ns9qha1nj308u089aav.jpg)

L表示为图中的黑色圆形，随着梯度下降法的不断逼近，与圆第一次产生交点，而这个交点很难出现在坐标轴上。这就说明了L2正则化不容易得到稀疏矩阵，同时为了求出损失函数的最小值，使得w1和w2无限接近于0，达到防止过拟合的问题。

### 5.2 什么场景下用L2正则化

只要数据线性相关，用LinearRegression拟合的不是很好，**需要正则化**，可以考虑使用岭回归(L2), 如何输入特征的维度很高,而且是稀疏线性关系的话， 岭回归就不太合适,考虑使用Lasso回归。

### 5.3 什么是L1正则化(Lasso回归)

L1正则化与L2正则化的区别在于惩罚项的不同：

![](https://latex.codecogs.com/gif.latex?J=J_0+\lambda(|w_1|+|w_2|))

求解J0的过程可以画出等值线。同时L1正则化的函数也可以在w1w2的二维平面上画出来。如下图：

![image](https://ws2.sinaimg.cn/large/00630Defgy1g4nse7rf9xj308u089gme.jpg)

惩罚项表示为图中的黑色棱形，随着梯度下降法的不断逼近，与棱形第一次产生交点，而这个交点很容易出现在坐标轴上。**这就说明了L1正则化容易得到稀疏矩阵。**

### 5.4 什么场景下使用L1正则化

**L1正则化(Lasso回归)可以使得一些特征的系数变小,甚至还使一些绝对值较小的系数直接变为0**，从而增强模型的泛化能力 。对于高的特征数据,尤其是线性关系是稀疏的，就采用L1正则化(Lasso回归),或者是要在一堆特征里面找出主要的特征，那么L1正则化(Lasso回归)更是首选了。

### 5.5 什么是ElasticNet回归

**ElasticNet综合了L1正则化项和L2正则化项**，以下是它的公式：

![](https://latex.codecogs.com/gif.latex?min(\frac{1}{2m}[\sum_{i=1}^{m}({y_i}^{'}-y_i)^2+\lambda\sum_{j=1}^{n}\theta_j^2]+\lambda\sum_{j=1}^{n}|\theta|))

### 5.6  ElasticNet回归的使用场景

ElasticNet在我们发现用Lasso回归太过(太多特征被稀疏为0),而岭回归也正则化的不够(回归系数衰减太慢)的时候，可以考虑使用ElasticNet回归来综合，得到比较好的结果。

## 6. 线性回归要求因变量服从正态分布？

我们假设线性回归的噪声服从均值为0的正态分布。 当噪声符合正态分布N(0,delta^2)时，因变量则符合正态分布N(ax(i)+b,delta^2)，其中预测函数y=ax(i)+b。这个结论可以由正态分布的概率密度函数得到。也就是说当噪声符合正态分布时，其因变量必然也符合正态分布。

在用线性回归模型拟合数据之前，首先要求数据应符合或近似符合正态分布，否则得到的拟合函数不正确。

## 7. [代码实现](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/Liner%20Regression/demo)

------

> 作者：[@mantchs](https://github.com/mantchs)
>
> 欢迎大家加入讨论！共同完善此项目！<a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=863f915b9178560bd32ca07cd090a7d9e6f5f90fcff5667489697b1621cecdb3"><img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="NLP面试学习群" title="NLP面试学习群"></a>

---

### 线性回归 Demo: 房价预测

## 目录
- [1.题目](#1题目)
- [2.步骤](#2步骤)
- [3.模型选择](#3模型选择)
- [4.环境配置](#4环境配置)
- [5.csv数据处理](#5csv数据处理)
- [6.数据处理](#6数据处理)
- [7.模型训练](#7模型训练)
- [8.完整代码](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/Liner%20Regression/demo/housing_price.py)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;这篇介绍的是我在做房价预测模型时的python代码，房价预测在机器学习入门中已经是个经典的题目了，但我发现目前网上还没有能够很好地做一个demo出来，使得入门者不能很快的找到"入口"在哪，所以在此介绍我是如何做的预测房价模型的题目，仅供参考。
## 1.题目：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;从给定的房屋基本信息以及房屋销售信息等，建立一个回归模型预测房屋的销售价格。
数据下载请点击：[下载](https://pan.baidu.com/share/init?surl=kVdwI3d)，密码：mfqy。
- **数据说明**：
数据主要包括2014年5月至2015年5月美国King County的房屋销售价格以及房屋的基本信息。
数据分为训练数据和测试数据，分别保存在kc_train.csv和kc_test.csv两个文件中。
其中训练数据主要包括10000条记录，14个字段，主要字段说明如下：
第一列"销售日期"：2014年5月到2015年5月房屋出售时的日期
第二列"销售价格"：房屋交易价格，单位为美元，是目标预测值
第三列"卧室数"：房屋中的卧室数目
第四列"浴室数"：房屋中的浴室数目
第五列"房屋面积"：房屋里的生活面积
第六列"停车面积"：停车坪的面积
第七列"楼层数"：房屋的楼层数
第八列"房屋评分"：King County房屋评分系统对房屋的总体评分
第九列"建筑面积"：除了地下室之外的房屋建筑面积
第十列"地下室面积"：地下室的面积
第十一列"建筑年份"：房屋建成的年份
第十二列"修复年份"：房屋上次修复的年份
第十三列"纬度"：房屋所在纬度
第十四列"经度"：房屋所在经度

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;测试数据主要包括3000条记录，13个字段，跟训练数据的不同是测试数据并不包括房屋销售价格，学员需要通过由训练数据所建立的模型以及所给的测试数据，得出测试数据相应的房屋销售价格预测值。

## 2.步骤
![](http://www.wailian.work/images/2018/12/10/12400f554.png)

- 1.选择合适的模型，对模型的好坏进行评估和选择。
- 2.对缺失的值进行补齐操作，可以使用均值的方式补齐数据，使得准确度更高。
- 3.数据的取值一般跟属性有关系，但世界万物的属性是很多的，有些值小，但不代表不重要，所有为了提高预测的准确度，统一数据维度进行计算，方法有特征缩放和归一法等。
- 4.数据处理好之后就可以进行调用模型库进行训练了。
- 5.使用测试数据进行目标函数预测输出，观察结果是否符合预期。或者通过画出对比函数进行结果线条对比。

## 3.模型选择
这里我们选择多元线性回归模型。公式如下：选择多元线性回归模型。
![](http://www.wailian.work/images/2018/12/10/12409d868.png)

y表示我们要求的销售价格，x表示特征值。需要调用sklearn库来进行训练。


## 4.环境配置
- python3.5
- numpy库
- pandas库
- matplotlib库进行画图
- seaborn库
- sklearn库

## 5.csv数据处理
下载的是两个数据文件，一个是真实数据，一个是测试数据，打开*kc_train.csv*，能够看到第二列是销售价格，而我们要预测的就是销售价格，所以在训练过程中是不需要销售价格的，把第二列删除掉，新建一个csv文件存放销售价格这一列，作为后面的结果对比。

## 6.数据处理
首先先读取数据，查看数据是否存在缺失值，然后进行特征缩放统一数据维度。代码如下：(注：最后会给出完整代码)
```python
#读取数据
housing = pd.read_csv('kc_train.csv')
target=pd.read_csv('kc_train2.csv')  #销售价格
t=pd.read_csv('kc_test.csv')         #测试数据

#数据预处理
housing.info()    #查看是否有缺失值

#特征缩放
from sklearn.preprocessing import MinMaxScaler
minmax_scaler=MinMaxScaler()
minmax_scaler.fit(housing)   #进行内部拟合，内部参数会发生变化
scaler_housing=minmax_scaler.transform(housing)
scaler_housing=pd.DataFrame(scaler_housing,columns=housing.columns)
```

## 7.模型训练
使用sklearn库的线性回归函数进行调用训练。梯度下降法获得误差最小值。最后使用均方误差法来评价模型的好坏程度，并画图进行比较。
```python
#选择基于梯度下降的线性回归模型
from sklearn.linear_model import LinearRegression
LR_reg=LinearRegression()
#进行拟合
LR_reg.fit(scaler_housing,target)


#使用均方误差用于评价模型好坏
from sklearn.metrics import mean_squared_error
preds=LR_reg.predict(scaler_housing)   #输入数据进行预测得到结果
mse=mean_squared_error(preds,target)   #使用均方误差来评价模型好坏，可以输出mse进行查看评价值

#绘图进行比较
plot.figure(figsize=(10,7))       #画布大小
num=100
x=np.arange(1,num+1)              #取100个点进行比较
plot.plot(x,target[:num],label='target')      #目标取值
plot.plot(x,preds[:num],label='preds')        #预测取值
plot.legend(loc='upper right')  #线条显示位置
plot.show()
```
最后输出的图是这样的：
![](http://www.wailian.work/images/2018/12/10/124094e96.png)
从这张结果对比图中就可以看出模型是否得到精确的目标函数，是否能够精确预测房价。
- 如果想要预测test文件里的数据，那就把test文件里的数据进行读取，并且进行特征缩放，调用：
**LR_reg.predict(test)**
就可以得到预测结果，并进行输出操作。
- 到这里可以看到机器学习也不是不能够学会，只要深入研究和总结，就能够找到学习的方法，重要的是总结，最后就是调用一些机器学习的方法库就行了，当然这只是入门级的，我觉得入门级的写到这已经足够了，很多人都能够看得懂，代码量不多。但要理解线性回归的概念性东西还是要多看资料。

## [8.完整代码](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/Liner%20Regression/demo/housing_price.py)

------

> 作者：[@mantchs](https://github.com/mantchs)
>
> 欢迎大家加入讨论！共同完善此项目！<a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=863f915b9178560bd32ca07cd090a7d9e6f5f90fcff5667489697b1621cecdb3"><img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="NLP面试学习群" title="NLP面试学习群"></a>

---


# 2. Logistics Regression (逻辑回归)

## 目录
- [1. 什么是逻辑回归](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/2.Logistics%20Regression#1-什么是逻辑回归)
- [2. 什么是Sigmoid函数](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/2.Logistics%20Regression#2-什么是sigmoid函数)
- [3. 损失函数是什么](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/2.Logistics%20Regression#3-损失函数是什么)
- [4.可以进行多分类吗？](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/2.Logistics%20Regression#4可以进行多分类吗)
- [5.逻辑回归有什么优点](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/2.Logistics%20Regression#5逻辑回归有什么优点)
- [6. 逻辑回归有哪些应用](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/2.Logistics%20Regression#6-逻辑回归有哪些应用)
- [7. 逻辑回归常用的优化方法有哪些](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/2.Logistics%20Regression#7-逻辑回归常用的优化方法有哪些)
  - [7.1 一阶方法](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/2.Logistics%20Regression#71-一阶方法)
  - [7.2 二阶方法：牛顿法、拟牛顿法：](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/2.Logistics%20Regression#72-二阶方法牛顿法拟牛顿法)
- [8. 逻辑斯特回归为什么要对特征进行离散化。](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/2.Logistics%20Regression#8-逻辑斯特回归为什么要对特征进行离散化)
- [9. 逻辑回归的目标函数中增大L1正则化会是什么结果。](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/2.Logistics%20Regression#9-逻辑回归的目标函数中增大l1正则化会是什么结果)
- [10. 代码实现](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/2.Logistics%20Regression/demo/CreditScoring.ipynb)

## 1. 什么是逻辑回归

逻辑回归是用来做分类算法的，大家都熟悉线性回归，一般形式是Y=aX+b，y的取值范围是[-∞, +∞]，有这么多取值，怎么进行分类呢？不用担心，伟大的数学家已经为我们找到了一个方法。

也就是把Y的结果带入一个非线性变换的**Sigmoid函数**中，即可得到[0,1]之间取值范围的数S，S可以把它看成是一个概率值，如果我们设置概率阈值为0.5，那么S大于0.5可以看成是正样本，小于0.5看成是负样本，就可以进行分类了。

## 2. 什么是Sigmoid函数

函数公式如下：

![image](https://wx4.sinaimg.cn/large/00630Defly1g4pvk2ctatj30cw0b63yq.jpg)

函数中t无论取什么值，其结果都在[0,-1]的区间内，回想一下，一个分类问题就有两种答案，一种是“是”，一种是“否”，那0对应着“否”，1对应着“是”，那又有人问了，你这不是[0,1]的区间吗，怎么会只有0和1呢？这个问题问得好，我们假设分类的**阈值**是0.5，那么超过0.5的归为1分类，低于0.5的归为0分类，阈值是可以自己设定的。

好了，接下来我们把aX+b带入t中就得到了我们的逻辑回归的一般模型方程：

![](https://latex.codecogs.com/gif.latex?H(a,b)=\frac{1}{1+e^{(aX+b)}})

结果P也可以理解为概率，换句话说概率大于0.5的属于1分类，概率小于0.5的属于0分类，这就达到了分类的目的。

## 3. 损失函数是什么

逻辑回归的损失函数是 **log loss**，也就是**对数似然函数**，函数公式如下：

![image](https://wx1.sinaimg.cn/large/00630Defly1g4pvtz3tw9j30et04v0sw.jpg)

公式中的 y=1 表示的是真实值为1时用第一个公式，真实 y=0 用第二个公式计算损失。为什么要加上log函数呢？可以试想一下，当真实样本为1是，但h=0概率，那么log0=∞，这就对模型最大的惩罚力度；当h=1时，那么log1=0，相当于没有惩罚，也就是没有损失，达到最优结果。所以数学家就想出了用log函数来表示损失函数。

最后按照梯度下降法一样，求解极小值点，得到想要的模型效果。

## 4.可以进行多分类吗？

可以的，其实我们可以从二分类问题过度到多分类问题(one vs rest)，思路步骤如下：

1.将类型class1看作正样本，其他类型全部看作负样本，然后我们就可以得到样本标记类型为该类型的概率p1。

2.然后再将另外类型class2看作正样本，其他类型全部看作负样本，同理得到p2。

3.以此循环，我们可以得到该待预测样本的标记类型分别为类型class i时的概率pi，最后我们取pi中最大的那个概率对应的样本标记类型作为我们的待预测样本类型。

![image](https://wx2.sinaimg.cn/large/00630Defly1g4pw11fo1tj30cv0c50tj.jpg)

总之还是以二分类来依次划分，并求出最大概率结果。

## 5.逻辑回归有什么优点

- LR能以概率的形式输出结果，而非只是0,1判定。
- LR的可解释性强，可控度高(你要给老板讲的嘛…)。
- 训练快，feature engineering之后效果赞。
- 因为结果是概率，可以做ranking model。

## 6. 逻辑回归有哪些应用

- CTR预估/推荐系统的learning to rank/各种分类场景。
- 某搜索引擎厂的广告CTR预估基线版是LR。
- 某电商搜索排序/广告CTR预估基线版是LR。
- 某电商的购物搭配推荐用了大量LR。
- 某现在一天广告赚1000w+的新闻app排序基线是LR。

## 7. 逻辑回归常用的优化方法有哪些

### 7.1 一阶方法

梯度下降、随机梯度下降、mini 随机梯度下降降法。随机梯度下降不但速度上比原始梯度下降要快，局部最优化问题时可以一定程度上抑制局部最优解的发生。 

### 7.2 二阶方法：牛顿法、拟牛顿法： 

这里详细说一下牛顿法的基本原理和牛顿法的应用方式。牛顿法其实就是通过切线与x轴的交点不断更新切线的位置，直到达到曲线与x轴的交点得到方程解。在实际应用中我们因为常常要求解凸优化问题，也就是要求解函数一阶导数为0的位置，而牛顿法恰好可以给这种问题提供解决方法。实际应用中牛顿法首先选择一个点作为起始点，并进行一次二阶泰勒展开得到导数为0的点进行一个更新，直到达到要求，这时牛顿法也就成了二阶求解问题，比一阶方法更快。我们常常看到的x通常为一个多维向量，这也就引出了Hessian矩阵的概念（就是x的二阶导数矩阵）。

缺点：牛顿法是定长迭代，没有步长因子，所以不能保证函数值稳定的下降，严重时甚至会失败。还有就是牛顿法要求函数一定是二阶可导的。而且计算Hessian矩阵的逆复杂度很大。

拟牛顿法： 不用二阶偏导而是构造出Hessian矩阵的近似正定对称矩阵的方法称为拟牛顿法。拟牛顿法的思路就是用一个特别的表达形式来模拟Hessian矩阵或者是他的逆使得表达式满足拟牛顿条件。主要有DFP法（逼近Hession的逆）、BFGS（直接逼近Hession矩阵）、 L-BFGS（可以减少BFGS所需的存储空间）。

## 8. 逻辑斯特回归为什么要对特征进行离散化。

1. 非线性！非线性！非线性！逻辑回归属于广义线性模型，表达能力受限；单变量离散化为N个后，每个变量有单独的权重，相当于为模型引入了非线性，能够提升模型表达能力，加大拟合； 离散特征的增加和减少都很容易，易于模型的快速迭代； 
2. 速度快！速度快！速度快！稀疏向量内积乘法运算速度快，计算结果方便存储，容易扩展； 
3. 鲁棒性！鲁棒性！鲁棒性！离散化后的特征对异常数据有很强的鲁棒性：比如一个特征是年龄>30是1，否则0。如果特征没有离散化，一个异常数据“年龄300岁”会给模型造成很大的干扰； 
4. 方便交叉与特征组合：离散化后可以进行特征交叉，由M+N个变量变为M*N个变量，进一步引入非线性，提升表达能力； 
5. 稳定性：特征离散化后，模型会更稳定，比如如果对用户年龄离散化，20-30作为一个区间，不会因为一个用户年龄长了一岁就变成一个完全不同的人。当然处于区间相邻处的样本会刚好相反，所以怎么划分区间是门学问； 
6. 简化模型：特征离散化以后，起到了简化了逻辑回归模型的作用，降低了模型过拟合的风险。

## 9. 逻辑回归的目标函数中增大L1正则化会是什么结果。

所有的参数w都会变成0。

## 10. 代码实现

GitHub：[https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/2.Logistics%20Regression/demo/CreditScoring.ipynb](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/2.Logistics%20Regression/demo/CreditScoring.ipynb)

------



> 作者：[@mantchs](https://github.com/NLP-LOVE/ML-NLP)
>
> GitHub：[https://github.com/NLP-LOVE/ML-NLP](https://github.com/NLP-LOVE/ML-NLP)
>
> 欢迎大家加入讨论！共同完善此项目！群号：【541954936】<a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=863f915b9178560bd32ca07cd090a7d9e6f5f90fcff5667489697b1621cecdb3"><img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="NLP面试学习群" title="NLP面试学习群"></a>


---

### demo

## Kaggle上简单的金融信用分类，代码都有注释，此处不再赘述！
[CreditScoring.ipynb](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/2.Logistics%20Regression/demo/CreditScoring.ipynb)

---

# 3. Decision Tree (决策树)

## 目录
- [1. 什么是决策树](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.Desition%20Tree#1-什么是决策树)
  - [1.1 决策树的基本思想](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.Desition%20Tree#11-决策树的基本思想)
  - [1.2 “树”的成长过程](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.Desition%20Tree#12-树的成长过程)
  - [1.3 "树"怎么长](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.Desition%20Tree#13-树怎么长)
  - [1.3.1 ID3算法](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.Desition%20Tree#131-id3算法)
  - [1.3.2 C4.5](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.Desition%20Tree#132-c45)
  - [1.3.3 CART算法](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.Desition%20Tree#133-cart算法)
  - [1.3.4 三种不同的决策树](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.Desition%20Tree#134-三种不同的决策树)
- [2. 树形结构为什么不需要归一化?](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.Desition%20Tree#2-树形结构为什么不需要归一化)
- [3. 分类决策树和回归决策树的区别](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.Desition%20Tree#3-分类决策树和回归决策树的区别)
- [4. 决策树如何剪枝](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.Desition%20Tree#4-决策树如何剪枝)
- [5. 代码实现](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/3.Desition%20Tree/DecisionTree.ipynb)

## 1. 什么是决策树

### 1.1 决策树的基本思想

其实用一下图片能更好的理解LR模型和决策树模型算法的根本区别，我们可以思考一下一个决策问题：是否去相亲，一个女孩的母亲要给这个女海介绍对象。

![image](https://wx2.sinaimg.cn/large/00630Defly1g4q286viibj30pk0pfk09.jpg)

大家都看得很明白了吧！LR模型是一股脑儿的把所有特征塞入学习，而决策树更像是编程语言中的if-else一样，去做条件判断，这就是根本性的区别。

### 1.2 “树”的成长过程

决策树基于“树”结构进行决策的，这时我们就要面临两个问题 ：

- “树”怎么长。
- 这颗“树”长到什么时候停。

弄懂了这两个问题，那么这个模型就已经建立起来了，决策树的总体流程是“分而治之”的思想，一是自根至叶的递归过程，一是在每个中间节点寻找一个“划分”属性，相当于就是一个特征属性了。接下来我们来逐个解决以上两个问题。

#### 这颗“树”长到什么时候停

- 当前结点包含的样本全属于同一类别，无需划分；例如：样本当中都是决定去相亲的，属于同一类别，就是不管特征如何改变都不会影响结果，这种就不需要划分了。
- 当前属性集为空，或是所有样本在所有属性上取值相同，无法划分；例如：所有的样本特征都是一样的，就造成无法划分了，训练集太单一。
- 当前结点包含的样本集合为空，不能划分。

### 1.3 "树"怎么长

在生活当中，我们都会碰到很多需要做出决策的地方，例如：吃饭地点、数码产品购买、旅游地区等，你会发现在这些选择当中都是依赖于大部分人做出的选择，也就是跟随大众的选择。其实在决策树当中也是一样的，当大部分的样本都是同一类的时候，那么就已经做出了决策。

我们可以把大众的选择抽象化，这就引入了一个概念就是纯度，想想也是如此，大众选择就意味着纯度越高。好，在深入一点，就涉及到一句话：**信息熵越低，纯度越高**。我相信大家或多或少都听说过“熵”这个概念，信息熵通俗来说就是用来度量包含的“信息量”，如果样本的属性都是一样的，就会让人觉得这包含的信息很单一，没有差异化，相反样本的属性都不一样，那么包含的信息量就很多了。

一到这里就头疼了，因为马上要引入信息熵的公式，其实也很简单：

![](https://latex.codecogs.com/gif.latex?Ent(D)=-\sum_{k=1}^{|y|}p_klog_2p_k)

Pk表示的是：当前样本集合D中第k类样本所占的比例为Pk。

**信息增益**

废话不多说直接上公式：

![image](https://wx3.sinaimg.cn/large/00630Defly1g4q5h6oby7j30he08tdh5.jpg)

看不懂的先不管，简单一句话就是：划分前的信息熵--划分后的信息熵。表示的是向纯度方向迈出的“步长”。

好了，有了前面的知识，我们就可以开始“树”的生长了。

#### 1.3.1 ID3算法

解释：在根节点处计算信息熵，然后根据属性依次划分并计算其节点的信息熵，用根节点信息熵--属性节点的信息熵=信息增益，根据信息增益进行降序排列，排在前面的就是第一个划分属性，其后依次类推，这就得到了决策树的形状，也就是怎么“长”了。

如果不理解的，可以查看我分享的图片示例，结合我说的，包你看懂：

1. [第一张图.jpg](https://www.wailian.work/images/2018/12/11/image39e7b.png)
2. [第二张图.jpg](https://www.wailian.work/images/2018/12/11/image61cdc.png)
3. [第三张图.jpg](https://www.wailian.work/images/2018/12/11/image9e194.png)
4. [第四张图.jpg](https://www.wailian.work/images/2018/12/11/image09288.png)

不过，信息增益有一个问题：对可取值数目较多的属性有所偏好，例如：考虑将“编号”作为一个属性。为了解决这个问题，引出了另一个 算法C4.5。

#### 1.3.2 C4.5

为了解决信息增益的问题，引入一个信息增益率：

![](https://latex.codecogs.com/gif.latex?Gain\_ratio(D,a)=\frac{Gain(D,a)}{IV(a)})

其中：

![](https://latex.codecogs.com/gif.latex?IV(a)=-\sum_{v=1}^{V}\frac{|D^v|}{|D|}log_2\frac{|D^v|}{|D|})

属性a的可能取值数目越多(即V越大)，则IV(a)的值通常就越大。**信息增益比本质： 是在信息增益的基础之上乘上一个惩罚参数。特征个数较多时，惩罚参数较小；特征个数较少时，惩罚参数较大。**不过有一个缺点：

- 缺点：信息增益率偏向取值较少的特征。

使用信息增益率：基于以上缺点，并不是直接选择信息增益率最大的特征，而是现在候选特征中找出信息增益高于平均水平的特征，然后在这些特征中再选择信息增益率最高的特征。

#### 1.3.3 CART算法

数学家真实聪明，想到了另外一个表示纯度的方法，叫做基尼指数(讨厌的公式)：

![image](https://wx1.sinaimg.cn/large/00630Defly1g4q5dmvyykj30eb01edfs.jpg)

表示在样本集合中一个随机选中的样本被分错的概率。举例来说，现在一个袋子里有3种颜色的球若干个，伸手进去掏出2个球，颜色不一样的概率，这下明白了吧。**Gini(D)越小，数据集D的纯度越高。**

##### 举个例子

假设现在有特征 “学历”，此特征有三个特征取值： “本科”，“硕士”， “博士”，

当使用“学历”这个特征对样本集合D进行划分时，划分值分别有三个，因而有三种划分的可能集合，划分后的子集如下：

1.划分点： “本科”，划分后的子集合 ： {本科}，{硕士，博士}

2.划分点： “硕士”，划分后的子集合 ： {硕士}，{本科，博士}

3.划分点： “硕士”，划分后的子集合 ： {博士}，{本科，硕士}}

对于上述的每一种划分，都可以计算出基于 **划分特征= 某个特征值** 将样本集合D划分为两个子集的纯度：

![](https://latex.codecogs.com/gif.latex?Gini(D,A)=\frac{|D_1|}{|D|}Gini(D_1)+\frac{|D_2|}{|D|}Gini(D_2))

因而**对于一个具有多个取值（超过2个）的特征，需要计算以每一个取值作为划分点，对样本D划分之后子集的纯度Gini(D,Ai)，(其中Ai 表示特征A的可能取值)**

然后从所有的可能划分的Gini(D,Ai)中找出Gini指数最小的划分，这个划分的划分点，便是使用特征A对样本集合D进行划分的最佳划分点。到此就可以长成一棵“大树”了。

#### 1.3.4 三种不同的决策树

- **ID3**：取值多的属性，更容易使数据更纯，其信息增益更大。

  训练得到的是一棵庞大且深度浅的树：不合理。

- **C4.5**：采用信息增益率替代信息增益。

- **CART**：以基尼系数替代熵，最小化不纯度，而不是最大化信息增益。

## 2. 树形结构为什么不需要归一化?

因为数值缩放不影响分裂点位置，对树模型的结构不造成影响。
按照特征值进行排序的，排序的顺序不变，那么所属的分支以及分裂点就不会有不同。而且，树模型是不能进行梯度下降的，因为构建树模型（回归树）寻找最优点时是通过寻找最优分裂点完成的，因此树模型是阶跃的，阶跃点是不可导的，并且求导没意义，也就不需要归一化。

既然树形结构（如决策树、RF）不需要归一化，那为何非树形结构比如Adaboost、SVM、LR、Knn、KMeans之类则需要归一化。

对于线性模型，特征值差别很大时，运用梯度下降的时候，损失等高线是椭圆形，需要进行多次迭代才能到达最优点。
但是如果进行了归一化，那么等高线就是圆形的，促使SGD往原点迭代，从而导致需要的迭代次数较少。

## 3. 分类决策树和回归决策树的区别

Classification And Regression Tree(CART)是决策树的一种，CART算法既可以用于创建分类树（Classification Tree），也可以用于创建回归树（Regression Tree），两者在建树的过程稍有差异。

**回归树**：

CART回归树是假设树为二叉树，通过不断将特征进行分裂。比如当前树结点是基于第j个特征值进行分裂的，设该特征值小于s的样本划分为左子树，大于s的样本划分为右子树。 

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase6415343854853617715.png)

而CART回归树实质上就是在该特征维度对样本空间进行划分，而这种空间划分的优化是一种NP难问题，因此，在决策树模型中是使用启发式方法解决。典型CART回归树产生的目标函数为：

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64153438551488112806.png)

因此，当我们为了求解最优的切分特征j和最优的切分点s，就转化为求解这么一个目标函数：

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase6415343855213970444.png)

所以我们只要遍历所有特征的的所有切分点，就能找到最优的切分特征和切分点。最终得到一棵回归树。

参考文章：[经典算法详解--CART分类决策树、回归树和模型树](https://blog.csdn.net/jiede1/article/details/76034328)

## 4. 决策树如何剪枝

决策树的剪枝基本策略有 预剪枝 (Pre-Pruning) 和 后剪枝 (Post-Pruning)。

- **预剪枝**：其中的核心思想就是，在每一次实际对结点进行进一步划分之前，先采用验证集的数据来验证如果划分是否能提高划分的准确性。如果不能，就把结点标记为叶结点并退出进一步划分；如果可以就继续递归生成节点。
- **后剪枝**：后剪枝则是先从训练集生成一颗完整的决策树，然后自底向上地对非叶结点进行考察，若将该结点对应的子树替换为叶结点能带来泛化性能提升，则将该子树替换为叶结点。

参考文章：[决策树及决策树生成与剪枝](https://blog.csdn.net/am290333566/article/details/81187562)

## 5. 代码实现

GitHub：[https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/3.Desition%20Tree/DecisionTree.ipynb](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/3.Desition%20Tree/DecisionTree.ipynb)

------



> 作者：[@mantchs](https://github.com/NLP-LOVE/ML-NLP)
>
> GitHub：[https://github.com/NLP-LOVE/ML-NLP](https://github.com/NLP-LOVE/ML-NLP)
>
> 欢迎大家加入讨论！共同完善此项目！群号：【541954936】<a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=863f915b9178560bd32ca07cd090a7d9e6f5f90fcff5667489697b1621cecdb3"><img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="NLP面试学习群" title="NLP面试学习群"></a>

---

# 3.1 Random Forest (随机森林)

## 目录
- [1.什么是随机森林](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.1%20Random%20Forest#1什么是随机森林)
  - [1.1 Bagging思想](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.1%20Random%20Forest#11-bagging思想)
  - [1.2 随机森林](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.1%20Random%20Forest#12-随机森林)
- [2. 随机森林分类效果的影响因素](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.1%20Random%20Forest#2-随机森林分类效果的影响因素)
- [3. 随机森林有什么优缺点](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.1%20Random%20Forest#3-随机森林有什么优缺点)
- [4. 随机森林如何处理缺失值？](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.1%20Random%20Forest#4-随机森林如何处理缺失值)
- [5. 什么是OOB？随机森林中OOB是如何计算的，它有什么优缺点？](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.1%20Random%20Forest#5-什么是oob随机森林中oob是如何计算的它有什么优缺点)
- [6. 随机森林的过拟合问题](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.1%20Random%20Forest#6-随机森林的过拟合问题)
- [7. 代码实现](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/3.1%20Random%20Forest/RandomForestRegression.ipynb)

## 1.什么是随机森林

### 1.1 Bagging思想

Bagging是bootstrap aggregating。思想就是从总体样本当中随机取一部分样本进行训练，通过多次这样的结果，进行投票获取平均值作为结果输出，这就极大可能的避免了不好的样本数据，从而提高准确度。因为有些是不好的样本，相当于噪声，模型学入噪声后会使准确度不高。

**举个例子**：

假设有1000个样本，如果按照以前的思维，是直接把这1000个样本拿来训练，但现在不一样，先抽取800个样本来进行训练，假如噪声点是这800个样本以外的样本点，就很有效的避开了。重复以上操作，提高模型输出的平均值。

### 1.2 随机森林

Random Forest(随机森林)是一种基于树模型的Bagging的优化版本，一棵树的生成肯定还是不如多棵树，因此就有了随机森林，解决决策树泛化能力弱的特点。(可以理解成三个臭皮匠顶过诸葛亮)

而同一批数据，用同样的算法只能产生一棵树，这时Bagging策略可以帮助我们产生不同的数据集。**Bagging**策略来源于bootstrap aggregation：从样本集（假设样本集N个数据点）中重采样选出Nb个样本（有放回的采样，样本数据点个数仍然不变为N），在所有样本上，对这n个样本建立分类器（ID3\C4.5\CART\SVM\LOGISTIC），重复以上两步m次，获得m个分类器，最后根据这m个分类器的投票结果，决定数据属于哪一类。

**每棵树的按照如下规则生成：**

1. 如果训练集大小为N，对于每棵树而言，**随机**且有放回地从训练集中的抽取N个训练样本，作为该树的训练集；
2. 如果每个样本的特征维度为M，指定一个常数m<<M，**随机**地从M个特征中选取m个特征子集，每次树进行分裂时，从这m个特征中选择最优的；
3. 每棵树都尽最大程度的生长，并且没有剪枝过程。

一开始我们提到的随机森林中的“随机”就是指的这里的两个随机性。两个随机性的引入对随机森林的分类性能至关重要。由于它们的引入，使得随机森林不容易陷入过拟合，并且具有很好得抗噪能力（比如：对缺省值不敏感）。

总的来说就是随机选择样本数，随机选取特征，随机选择分类器，建立多颗这样的决策树，然后通过这几课决策树来投票，决定数据属于哪一类(**投票机制有一票否决制、少数服从多数、加权多数**)

## 2. 随机森林分类效果的影响因素

- 森林中任意两棵树的相关性：相关性越大，错误率越大；
- 森林中每棵树的分类能力：每棵树的分类能力越强，整个森林的错误率越低。

减小特征选择个数m，树的相关性和分类能力也会相应的降低；增大m，两者也会随之增大。所以关键问题是如何选择最优的m（或者是范围），这也是随机森林唯一的一个参数。

## 3. 随机森林有什么优缺点

**优点：**

- 在当前的很多数据集上，相对其他算法有着很大的优势，表现良好。
- 它能够处理很高维度（feature很多）的数据，并且不用做特征选择(因为特征子集是随机选择的)。
- 在训练完后，它能够给出哪些feature比较重要。
- 训练速度快，容易做成并行化方法(训练时树与树之间是相互独立的)。
- 在训练过程中，能够检测到feature间的互相影响。
- 对于不平衡的数据集来说，它可以平衡误差。
- 如果有很大一部分的特征遗失，仍可以维持准确度。

**缺点：**

- 随机森林已经被证明在某些**噪音较大**的分类或回归问题上会过拟合。
- 对于有不同取值的属性的数据，取值划分较多的属性会对随机森林产生更大的影响，所以随机森林在这种数据上产出的属性权值是不可信的。

## 4. 随机森林如何处理缺失值？

根据随机森林创建和训练的特点，随机森林对缺失值的处理还是比较特殊的。

- 首先，给缺失值预设一些估计值，比如数值型特征，选择其余数据的中位数或众数作为当前的估计值
- 然后，根据估计的数值，建立随机森林，把所有的数据放进随机森林里面跑一遍。记录每一组数据在决策树中一步一步分类的路径.
- 判断哪组数据和缺失数据路径最相似，引入一个相似度矩阵，来记录数据之间的相似度，比如有N组数据，相似度矩阵大小就是N*N
- 如果缺失值是类别变量，通过权重投票得到新估计值，如果是数值型变量，通过加权平均得到新的估计值，如此迭代，直到得到稳定的估计值。

其实，该缺失值填补过程类似于推荐系统中采用协同过滤进行评分预测，先计算缺失特征与其他特征的相似度，再加权得到缺失值的估计，而随机森林中计算相似度的方法（数据在决策树中一步一步分类的路径）乃其独特之处。

## 5. 什么是OOB？随机森林中OOB是如何计算的，它有什么优缺点？

**OOB**：

上面我们提到，构建随机森林的关键问题就是如何选择最优的m，要解决这个问题主要依据计算袋外错误率oob error（out-of-bag error）。

bagging方法中Bootstrap每次约有1/3的样本不会出现在Bootstrap所采集的样本集合中，当然也就没有参加决策树的建立，把这1/3的数据称为**袋外数据oob（out of bag）**,它可以用于取代测试集误差估计方法。

**袋外数据(oob)误差的计算方法如下：**

- 对于已经生成的随机森林,用袋外数据测试其性能,假设袋外数据总数为O,用这O个袋外数据作为输入,带进之前已经生成的随机森林分类器,分类器会给出O个数据相应的分类
- 因为这O条数据的类型是已知的,则用正确的分类与随机森林分类器的结果进行比较,统计随机森林分类器分类错误的数目,设为X,则袋外数据误差大小=X/O

**优缺点**：

这已经经过证明是无偏估计的,所以在随机森林算法中不需要再进行交叉验证或者单独的测试集来获取测试集误差的无偏估计。 

## 6. 随机森林的过拟合问题

1. 你已经建了一个有10000棵树的随机森林模型。在得到0.00的训练误差后，你非常高兴。但是，验证错误是34.23。到底是怎么回事？你还没有训练好你的模型吗？

   答：该模型过度拟合，因此，为了避免这些情况，我们要用交叉验证来调整树的数量。

## 7. 代码实现

GitHub：[https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/3.1%20Random%20Forest/RandomForestRegression.ipynb](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/3.1%20Random%20Forest/RandomForestRegression.ipynb)

> 作者：[@mantchs](https://github.com/NLP-LOVE/ML-NLP)
>
> GitHub：[https://github.com/NLP-LOVE/ML-NLP](https://github.com/NLP-LOVE/ML-NLP)
>
> 欢迎大家加入讨论！共同完善此项目！qq群号：【541954936】<a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=863f915b9178560bd32ca07cd090a7d9e6f5f90fcff5667489697b1621cecdb3"><img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="NLP面试学习群" title="NLP面试学习群"></a>

---

# 3.2 GBDT

## 目录
- [1. 解释一下GBDT算法的过程](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.2%20GBDT#1-解释一下gbdt算法的过程)
  - [1.1 Boosting思想](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.2%20GBDT#11-boosting思想)
  - [1.2 GBDT原来是这么回事](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.2%20GBDT#12-gbdt原来是这么回事)
- [2. 梯度提升和梯度下降的区别和联系是什么？](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.2%20GBDT#2-梯度提升和梯度下降的区别和联系是什么)
- [3. GBDT的优点和局限性有哪些？](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.2%20GBDT#3-gbdt的优点和局限性有哪些)
  - [3.1 优点](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.2%20GBDT#31-优点)
  - [3.2 局限性](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.2%20GBDT#32-局限性)
- [4. RF(随机森林)与GBDT之间的区别与联系](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.2%20GBDT#4-rf随机森林与gbdt之间的区别与联系)
- [5. 代码实现](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/3.2%20GBDT/GBDT_demo.ipynb)

## 1. 解释一下GBDT算法的过程

GBDT(Gradient Boosting Decision Tree)，全名叫梯度提升决策树，使用的是**Boosting**的思想。

### 1.1 Boosting思想

Boosting方法训练基分类器时采用串行的方式，各个基分类器之间有依赖。它的基本思路是将基分类器层层叠加，每一层在训练的时候，对前一层基分类器分错的样本，给予更高的权重。测试时，根据各层分类器的结果的加权得到最终结果。 

Bagging与Boosting的串行训练方式不同，Bagging方法在训练过程中，各基分类器之间无强依赖，可以进行并行训练。

### 1.2 GBDT原来是这么回事

GBDT的原理很简单，就是所有弱分类器的结果相加等于预测值，然后下一个弱分类器去拟合误差函数对预测值的残差(这个残差就是预测值与真实值之间的误差)。当然了，它里面的弱分类器的表现形式就是各棵树。

举一个非常简单的例子，比如我今年30岁了，但计算机或者模型GBDT并不知道我今年多少岁，那GBDT咋办呢？

- 它会在第一个弱分类器（或第一棵树中）随便用一个年龄比如20岁来拟合，然后发现误差有10岁；
- 接下来在第二棵树中，用6岁去拟合剩下的损失，发现差距还有4岁；
- 接着在第三棵树中用3岁拟合剩下的差距，发现差距只有1岁了；
- 最后在第四课树中用1岁拟合剩下的残差，完美。
- 最终，四棵树的结论加起来，就是真实年龄30岁（实际工程中，gbdt是计算负梯度，用负梯度近似残差）。

**为何gbdt可以用用负梯度近似残差呢？**

回归任务下，GBDT 在每一轮的迭代时对每个样本都会有一个预测值，此时的损失函数为均方差损失函数，

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155214962034944638.gif)

那此时的负梯度是这样计算的

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155214962416670973.gif)

所以，当损失函数选用均方损失函数是时，每一次拟合的值就是（真实值 - 当前模型预测的值），即残差。此时的变量是![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155214963633267938.gif)，即“当前预测模型的值”，也就是对它求负梯度。

**训练过程**

简单起见，假定训练集只有4个人：A,B,C,D，他们的年龄分别是14,16,24,26。其中A、B分别是高一和高三学生；C,D分别是应届毕业生和工作两年的员工。如果是用一棵传统的回归决策树来训练，会得到如下图所示结果：

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64153438568191303958.png)

现在我们使用GBDT来做这件事，由于数据太少，我们限定叶子节点做多有两个，即每棵树都只有一个分枝，并且限定只学两棵树。我们会得到如下图所示结果：

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64153438570529256895.png)

在第一棵树分枝和图1一样，由于A,B年龄较为相近，C,D年龄较为相近，他们被分为左右两拨，每拨用平均年龄作为预测值。

- 此时计算残差（残差的意思就是：A的实际值 - A的预测值 = A的残差），所以A的残差就是实际值14 - 预测值15 = 残差值-1。
- 注意，A的预测值是指前面所有树累加的和，这里前面只有一棵树所以直接是15，如果还有树则需要都累加起来作为A的预测值。

然后拿它们的残差-1、1、-1、1代替A B C D的原值，到第二棵树去学习，第二棵树只有两个值1和-1，直接分成两个节点，即A和C分在左边，B和D分在右边，经过计算（比如A，实际值-1 - 预测值-1 = 残差0，比如C，实际值-1 - 预测值-1 = 0），此时所有人的残差都是0。残差值都为0，相当于第二棵树的预测值和它们的实际值相等，则只需把第二棵树的结论累加到第一棵树上就能得到真实年龄了，即每个人都得到了真实的预测值。

换句话说，现在A,B,C,D的预测值都和真实年龄一致了。Perfect！

- A: 14岁高一学生，购物较少，经常问学长问题，预测年龄A = 15 – 1 = 14
- B: 16岁高三学生，购物较少，经常被学弟问问题，预测年龄B = 15 + 1 = 16
- C: 24岁应届毕业生，购物较多，经常问师兄问题，预测年龄C = 25 – 1 = 24
- D: 26岁工作两年员工，购物较多，经常被师弟问问题，预测年龄D = 25 + 1 = 26

所以，GBDT需要将多棵树的得分累加得到最终的预测得分，且每一次迭代，都在现有树的基础上，增加一棵树去拟合前面树的预测结果与真实值之间的残差。

## 2. 梯度提升和梯度下降的区别和联系是什么？ 

下表是梯度提升算法和梯度下降算法的对比情况。可以发现，两者都是在每 一轮迭代中，利用损失函数相对于模型的负梯度方向的信息来对当前模型进行更 新，只不过在梯度下降中，模型是以参数化形式表示，从而模型的更新等价于参 数的更新。而在梯度提升中，模型并不需要进行参数化表示，而是直接定义在函 数空间中，从而大大扩展了可以使用的模型种类。

![](http://wx3.sinaimg.cn/mw690/00630Defgy1g4tdwhqzsdj30rp0afdho.jpg)

## 3. **GBDT**的优点和局限性有哪些？ 

### 3.1 优点

1. 预测阶段的计算速度快，树与树之间可并行化计算。
2. 在分布稠密的数据集上，泛化能力和表达能力都很好，这使得GBDT在Kaggle的众多竞赛中，经常名列榜首。 
3. 采用决策树作为弱分类器使得GBDT模型具有较好的解释性和鲁棒性，能够自动发现特征间的高阶关系。

### 3.2 局限性

1. GBDT在高维稀疏的数据集上，表现不如支持向量机或者神经网络。
2. GBDT在处理文本分类特征问题上，相对其他模型的优势不如它在处理数值特征时明显。 
3. 训练过程需要串行训练，只能在决策树内部采用一些局部并行的手段提高训练速度。 

## 4. RF(随机森林)与GBDT之间的区别与联系

**相同点**：

- 都是由多棵树组成，最终的结果都是由多棵树一起决定。
- RF和GBDT在使用CART树时，可以是分类树或者回归树。

**不同点**：

- 组成随机森林的树可以并行生成，而GBDT是串行生成
- 随机森林的结果是多数表决表决的，而GBDT则是多棵树累加之和
- 随机森林对异常值不敏感，而GBDT对异常值比较敏感
- 随机森林是减少模型的方差，而GBDT是减少模型的偏差
- 随机森林不需要进行特征归一化。而GBDT则需要进行特征归一化

## 5. 代码实现

GitHub：[https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/3.2%20GBDT/GBDT_demo.ipynb](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/3.2%20GBDT/GBDT_demo.ipynb)

> 作者：[@mantchs](https://github.com/NLP-LOVE/ML-NLP)
>
> GitHub：[https://github.com/NLP-LOVE/ML-NLP](https://github.com/NLP-LOVE/ML-NLP)
>
> 欢迎大家加入讨论！共同完善此项目！qq群号：【541954936】<a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=863f915b9178560bd32ca07cd090a7d9e6f5f90fcff5667489697b1621cecdb3"><img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="NLP面试学习群" title="NLP面试学习群"></a>




---

# 3.3 XGBoost

## 目录
- [1. 什么是XGBoost](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.3%20XGBoost#1-什么是xgboost)
  - [1.1 XGBoost树的定义](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.3%20XGBoost#11-xgboost树的定义)
  - [1.2 正则项：树的复杂度](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.3%20XGBoost#12-正则项树的复杂度)
  - [1.3 树该怎么长](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.3%20XGBoost#13-树该怎么长)
  - [1.4 如何停止树的循环生成](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.3%20XGBoost#14-如何停止树的循环生成)
- [2. XGBoost与GBDT有什么不同](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.3%20XGBoost#2-xgboost与gbdt有什么不同)
- [3. 为什么XGBoost要用泰勒展开，优势在哪里？](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.3%20XGBoost#3-为什么xgboost要用泰勒展开优势在哪里)
- [4. 代码实现](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/3.3%20XGBoost/3.3%20XGBoost.ipynb)
- [5. 参考文献](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.3%20XGBoost#5-参考文献)

## 1. 什么是XGBoost

XGBoost是陈天奇等人开发的一个开源机器学习项目，高效地实现了GBDT算法并进行了算法和工程上的许多改进，被广泛应用在Kaggle竞赛及其他许多机器学习竞赛中并取得了不错的成绩。

说到XGBoost，不得不提GBDT(Gradient Boosting Decision Tree)。因为XGBoost本质上还是一个GBDT，但是力争把速度和效率发挥到极致，所以叫X (Extreme) GBoosted。包括前面说过，两者都是boosting方法。

关于GBDT，这里不再提，可以查看我前一篇的介绍，[点此跳转](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/3.2%20GBDT/3.2%20GBDT.md)。

### 1.1 XGBoost树的定义

先来举个**例子**，我们要预测一家人对电子游戏的喜好程度，考虑到年轻和年老相比，年轻更可能喜欢电子游戏，以及男性和女性相比，男性更喜欢电子游戏，故先根据年龄大小区分小孩和大人，然后再通过性别区分开是男是女，逐一给各人在电子游戏喜好程度上打分，如下图所示。

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64153438577232516800.png)

就这样，训练出了2棵树tree1和tree2，类似之前gbdt的原理，两棵树的结论累加起来便是最终的结论，所以小孩的预测分数就是两棵树中小孩所落到的结点的分数相加：2 + 0.9 = 2.9。爷爷的预测分数同理：-1 + （-0.9）= -1.9。具体如下图所示：

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64153438578739198433.png)

恩，你可能要拍案而起了，惊呼，这不是跟上文介绍的GBDT乃异曲同工么？

事实上，如果不考虑工程实现、解决问题上的一些差异，XGBoost与GBDT比较大的不同就是目标函数的定义。XGBoost的目标函数如下图所示：

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64153438580139159593.png)

其中：

- 红色箭头所指向的L 即为损失函数（比如平方损失函数：![](https://latex.codecogs.com/gif.latex?l(y_i,y^i)=(y_i-y^i)^2))
- 红色方框所框起来的是正则项（包括L1正则、L2正则）
- 红色圆圈所圈起来的为常数项
- 对于f(x)，XGBoost利用泰勒展开三项，做一个近似。**f(x)表示的是其中一颗回归树。**

看到这里可能有些读者会头晕了，这么多公式，**我在这里只做一个简要式的讲解，具体的算法细节和公式求解请查看这篇博文，讲得很仔细**：[通俗理解kaggle比赛大杀器xgboost](https://blog.csdn.net/v_JULY_v/article/details/81410574)

XGBoost的**核心算法思想**不难，基本就是：

1. 不断地添加树，不断地进行特征分裂来生长一棵树，每次添加一个树，其实是学习一个新函数**f(x)**，去拟合上次预测的残差。
2. 当我们训练完成得到k棵树，我们要预测一个样本的分数，其实就是根据这个样本的特征，在每棵树中会落到对应的一个叶子节点，每个叶子节点就对应一个分数
3. 最后只需要将每棵树对应的分数加起来就是该样本的预测值。

显然，我们的目标是要使得树群的预测值![](https://latex.codecogs.com/gif.latex?y_i^{'})尽量接近真实值![](https://latex.codecogs.com/gif.latex?y_i)，而且有尽量大的泛化能力。类似之前GBDT的套路，XGBoost也是需要将多棵树的得分累加得到最终的预测得分（每一次迭代，都在现有树的基础上，增加一棵树去拟合前面树的预测结果与真实值之间的残差）。

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64153438657261833493.png)

那接下来，我们如何选择每一轮加入什么 f 呢？答案是非常直接的，选取一个 f 来使得我们的目标函数尽量最大地降低。这里 f 可以使用泰勒展开公式近似。

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase6415343865867530120.png)

实质是把样本分配到叶子结点会对应一个obj，优化过程就是obj优化。也就是分裂节点到叶子不同的组合，不同的组合对应不同obj，所有的优化围绕这个思想展开。到目前为止我们讨论了目标函数中的第一个部分：训练误差。接下来我们讨论目标函数的第二个部分：正则项，即如何定义树的复杂度。

### 1.2 正则项：树的复杂度

XGBoost对树的复杂度包含了两个部分：

- 一个是树里面叶子节点的个数T
- 一个是树上叶子节点的得分w的L2模平方（对w进行L2正则化，相当于针对每个叶结点的得分增加L2平滑，目的是为了避免过拟合）

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64153438674199471483.png)

我们再来看一下XGBoost的目标函数（损失函数揭示训练误差 + 正则化定义复杂度）：

![](https://latex.codecogs.com/gif.latex?L(\phi)=\sum_{i}l(y_i^{'}-y_i)+\sum_k\Omega(f_t))

正则化公式也就是目标函数的后半部分，对于上式而言，![](https://latex.codecogs.com/gif.latex?y_i^{'})是整个累加模型的输出，正则化项∑kΩ(ft)是则表示树的复杂度的函数，值越小复杂度越低，泛化能力越强。

### 1.3 树该怎么长

很有意思的一个事是，我们从头到尾了解了xgboost如何优化、如何计算，但树到底长啥样，我们却一直没看到。很显然，一棵树的生成是由一个节点一分为二，然后不断分裂最终形成为整棵树。那么树怎么分裂的就成为了接下来我们要探讨的关键。对于一个叶子节点如何进行分裂，XGBoost作者在其原始论文中给出了一种分裂节点的方法：**枚举所有不同树结构的贪心法**

不断地枚举不同树的结构，然后利用打分函数来寻找出一个最优结构的树，接着加入到模型中，不断重复这样的操作。这个寻找的过程使用的就是**贪心算法**。选择一个feature分裂，计算loss function最小值，然后再选一个feature分裂，又得到一个loss function最小值，你枚举完，找一个效果最好的，把树给分裂，就得到了小树苗。

总而言之，XGBoost使用了和CART回归树一样的想法，利用贪婪算法，遍历所有特征的所有特征划分点，不同的是使用的目标函数不一样。具体做法就是分裂后的目标函数值比单子叶子节点的目标函数的增益，同时为了限制树生长过深，还加了个阈值，只有当增益大于该阈值才进行分裂。从而继续分裂，形成一棵树，再形成一棵树，**每次在上一次的预测基础上取最优进一步分裂/建树。**

### 1.4 如何停止树的循环生成

凡是这种循环迭代的方式必定有停止条件，什么时候停止呢？简言之，设置树的最大深度、当样本权重和小于设定阈值时停止生长以防止过拟合。具体而言，则

1. 当引入的分裂带来的增益小于设定阀值的时候，我们可以忽略掉这个分裂，所以并不是每一次分裂loss function整体都会增加的，有点预剪枝的意思，阈值参数为（即正则项里叶子节点数T的系数）；
2. 当树达到最大深度时则停止建立决策树，设置一个超参数max_depth，避免树太深导致学习局部样本，从而过拟合；
3. 样本权重和小于设定阈值时则停止建树。什么意思呢，即涉及到一个超参数-最小的样本权重和min_child_weight，和GBM的 min_child_leaf 参数类似，但不完全一样。大意就是一个叶子节点样本太少了，也终止同样是防止过拟合；

## 2. XGBoost与GBDT有什么不同

除了算法上与传统的GBDT有一些不同外，XGBoost还在工程实现上做了大量的优化。总的来说，两者之间的区别和联系可以总结成以下几个方面。 

1. GBDT是机器学习算法，XGBoost是该算法的工程实现。
2. 在使用CART作为基分类器时，XGBoost显式地加入了正则项来控制模 型的复杂度，有利于防止过拟合，从而提高模型的泛化能力。
3. GBDT在模型训练时只使用了代价函数的一阶导数信息，XGBoost对代 价函数进行二阶泰勒展开，可以同时使用一阶和二阶导数。
4. 传统的GBDT采用CART作为基分类器，XGBoost支持多种类型的基分类 器，比如线性分类器。
5. 传统的GBDT在每轮迭代时使用全部的数据，XGBoost则采用了与随机 森林相似的策略，支持对数据进行采样。
6. 传统的GBDT没有设计对缺失值进行处理，XGBoost能够自动学习出缺 失值的处理策略。

## 3. 为什么XGBoost要用泰勒展开，优势在哪里？

XGBoost使用了一阶和二阶偏导, 二阶导数有利于梯度下降的更快更准. 使用泰勒展开取得函数做自变量的二阶导数形式, 可以在不选定损失函数具体形式的情况下, 仅仅依靠输入数据的值就可以进行叶子分裂优化计算, 本质上也就把损失函数的选取和模型算法优化/参数选择分开了. 这种去耦合增加了XGBoost的适用性, 使得它按需选取损失函数, 可以用于分类, 也可以用于回归。

## 4. 代码实现

GitHub：[点击进入](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/3.3%20XGBoost/3.3%20XGBoost.ipynb)

## 5. 参考文献

[通俗理解kaggle比赛大杀器xgboost](https://blog.csdn.net/v_JULY_v/article/details/81410574)

> 作者：[@mantchs](https://github.com/NLP-LOVE/ML-NLP)
>
> GitHub：[https://github.com/NLP-LOVE/ML-NLP](https://github.com/NLP-LOVE/ML-NLP)
>
> 欢迎大家加入讨论！共同完善此项目！群号：【541954936】<a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=863f915b9178560bd32ca07cd090a7d9e6f5f90fcff5667489697b1621cecdb3"><img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="NLP面试学习群" title="NLP面试学习群"></a>

---

# 3.4 LightGBM

## 目录
- [1. LightGBM是什么东东](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.4%20LightGBM#1-lightgbm是什么东东)
  - [1.1 LightGBM在哪些地方进行了优化 (区别XGBoost)？](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.4%20LightGBM#11-lightgbm在哪些地方进行了优化----区别xgboost)
  - [1.2 Histogram算法](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.4%20LightGBM#12-histogram算法)
  - [1.3 带深度限制的Leaf-wise的叶子生长策略](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.4%20LightGBM#13-带深度限制的leaf-wise的叶子生长策略)
  - [1.4 直方图差加速](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.4%20LightGBM#14-直方图差加速)
  - [1.5 直接支持类别特征](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.4%20LightGBM#15-直接支持类别特征)
- [2. LightGBM优点](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/3.4%20LightGBM#2-lightgbm优点)
- [3. 代码实现](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/3.4%20LightGBM/3.4%20LightGBM.ipynb)

## 1. LightGBM是什么东东

不久前微软DMTK(分布式机器学习工具包)团队在GitHub上开源了性能超越其他boosting工具的LightGBM，在三天之内GitHub上被star了1000次，fork了200次。知乎上有近千人关注“如何看待微软开源的LightGBM？”问题，被评价为“速度惊人”，“非常有启发”，“支持分布式”，“代码清晰易懂”，“占用内存小”等。

LightGBM （Light Gradient Boosting Machine）(请点击[https://github.com/Microsoft/LightGBM](https://github.com/Microsoft/LightGBM))是一个实现GBDT算法的框架，支持高效率的并行训练。

LightGBM在Higgs数据集上LightGBM比XGBoost快将近10倍，内存占用率大约为XGBoost的1/6，并且准确率也有提升。GBDT在每一次迭代的时候，都需要遍历整个训练数据多次。如果把整个训练数据装进内存则会限制训练数据的大小；如果不装进内存，反复地读写训练数据又会消耗非常大的时间。尤其面对工业级海量的数据，普通的GBDT算法是不能满足其需求的。

LightGBM提出的主要原因就是为了解决GBDT在海量数据遇到的问题，让GBDT可以更好更快地用于工业实践。

### 1.1 LightGBM在哪些地方进行了优化    (区别XGBoost)？

- 基于Histogram的决策树算法
- 带深度限制的Leaf-wise的叶子生长策略
- 直方图做差加速直接
- 支持类别特征(Categorical Feature)
- Cache命中率优化
- 基于直方图的稀疏特征优化多线程优化。

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155197431597512984.jpg)

### 1.2 Histogram算法

直方图算法的基本思想是先把连续的浮点特征值离散化成k个整数（其实又是分桶的思想，而这些桶称为bin，比如[0,0.1)→0, [0.1,0.3)→1），同时构造一个宽度为k的直方图。

在遍历数据的时候，根据离散化后的值作为索引在直方图中累积统计量，当遍历一次数据后，直方图累积了需要的统计量，然后根据直方图的离散值，遍历寻找最优的分割点。

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155197418746568601.jpg)

使用直方图算法有很多优点。首先，最明显就是内存消耗的降低，直方图算法不仅不需要额外存储预排序的结果，而且可以只保存特征离散化后的值，而这个值一般用8位整型存储就足够了，内存消耗可以降低为原来的1/8。然后在计算上的代价也大幅降低，预排序算法每遍历一个特征值就需要计算一次分裂的增益，而直方图算法只需要计算k次（k可以认为是常数），时间复杂度从O(#data*#feature)优化到O(k*#features)。

### 1.3 带深度限制的Leaf-wise的叶子生长策略

在XGBoost中，树是按层生长的，称为Level-wise tree growth，同一层的所有节点都做分裂，最后剪枝，如下图所示：

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155197509149646916.png)

Level-wise过一次数据可以同时分裂同一层的叶子，容易进行多线程优化，也好控制模型复杂度，不容易过拟合。但实际上Level-wise是一种低效的算法，因为它不加区分的对待同一层的叶子，带来了很多没必要的开销，因为实际上很多叶子的分裂增益较低，没必要进行搜索和分裂。

在Histogram算法之上，LightGBM进行进一步的优化。首先它抛弃了大多数GBDT工具使用的按层生长 (level-wise)
的决策树生长策略，而使用了带有深度限制的按叶子生长 (leaf-wise)算法。

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155197520844369289.png)

Leaf-wise则是一种更为高效的策略，每次从当前所有叶子中，找到分裂增益最大的一个叶子，然后分裂，如此循环。因此同Level-wise相比，在分裂次数相同的情况下，Leaf-wise可以降低更多的误差，得到更好的精度。Leaf-wise的缺点是可能会长出比较深的决策树，产生过拟合。因此LightGBM在Leaf-wise之上增加了一个最大深度的限制，在保证高效率的同时防止过拟合。

### 1.4 直方图差加速

LightGBM另一个优化是Histogram（直方图）做差加速。一个容易观察到的现象：一个叶子的直方图可以由它的父亲节点的直方图与它兄弟的直方图做差得到。通常构造直方图，需要遍历该叶子上的所有数据，但直方图做差仅需遍历直方图的k个桶。

利用这个方法，LightGBM可以在构造一个叶子的直方图后，可以用非常微小的代价得到它兄弟叶子的直方图，在速度上可以提升一倍。

### 1.5 直接支持类别特征

实际上大多数机器学习工具都无法直接支持类别特征，一般需要把类别特征，转化到多维的0/1特征，降低了空间和时间的效率。而类别特征的使用是在实践中很常用的。基于这个考虑，LightGBM优化了对类别特征的支持，可以直接输入类别特征，不需要额外的0/1展开。并在决策树算法上增加了类别特征的决策规则。在Expo数据集上的实验，相比0/1展开的方法，训练速度可以加速8倍，并且精度一致。据我们所知，LightGBM是第一个直接支持类别特征的GBDT工具。

## 2. LightGBM优点

LightGBM （Light Gradient Boosting Machine）(请点击[https://github.com/Microsoft/LightGBM](https://github.com/Microsoft/LightGBM))是一个实现GBDT算法的框架，支持高效率的并行训练，并且具有以下优点：

- 更快的训练速度
- 更低的内存消耗
- 更好的准确率
- 分布式支持，可以快速处理海量数据

## 3. 代码实现

为了演示LightGBM在Python中的用法，本代码以sklearn包中自带的鸢尾花数据集为例，用lightgbm算法实现鸢尾花种类的分类任务。

GitHub：[点击进入](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/3.4%20LightGBM/3.4%20LightGBM.ipynb)

> 作者：[@mantchs](https://github.com/NLP-LOVE/ML-NLP)
>
> GitHub：[https://github.com/NLP-LOVE/ML-NLP](https://github.com/NLP-LOVE/ML-NLP)
>
> 欢迎大家加入讨论！共同完善此项目！群号：【541954936】<a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=863f915b9178560bd32ca07cd090a7d9e6f5f90fcff5667489697b1621cecdb3"><img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="NLP面试学习群" title="NLP面试学习群"></a>

---

# 4. SVM (支持向量机)

## 目录
- [1. 讲讲SVM](#1-讲讲svm)
  - [1.1 一个关于SVM的童话故事](#11-一个关于svm的童话故事)
  - [1.2 理解SVM：第一层](#12-理解svm第一层)
  - [1.3 深入SVM：第二层](#13-深入svm第二层)
  - [1.4 SVM的应用](#14-svm的应用)
- [2. SVM的一些问题](#2-svm的一些问题)
- [3. LR和SVM的联系与区别](#3-lr和svm的联系与区别)
- [4. 线性分类器与非线性分类器的区别以及优劣](#4-线性分类器与非线性分类器的区别以及优劣)
- [5. 代码实现](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/4.%20SVM/news%20classification)
- [6. 参考文献](#6-参考文献)

## 1. 讲讲SVM

### 1.1 一个关于SVM的童话故事

支持向量机（Support Vector Machine，SVM）是众多监督学习方法中十分出色的一种，几乎所有讲述经典机器学习方法的教材都会介绍。关于SVM，流传着一个关于天使与魔鬼的故事。

传说魔鬼和天使玩了一个游戏，魔鬼在桌上放了两种颜色的球。魔鬼让天使用一根木棍将它们分开。这对天使来说，似乎太容易了。天使不假思索地一摆，便完成了任务。魔鬼又加入了更多的球。随着球的增多，似乎有的球不能再被原来的木棍正确分开，如下图所示。 

![](http://wx4.sinaimg.cn/mw690/00630Defgy1g4vn6ow8vtj30i40ddgpt.jpg)

SVM实际上是在为天使找到木棒的最佳放置位置，使得两边的球都离分隔它们的木棒足够远。依照SVM为天使选择的木棒位置，魔鬼即使按刚才的方式继续加入新球，木棒也能很好地将两类不同的球分开。

看到天使已经很好地解决了用木棒线性分球的问题，魔鬼又给了天使一个新的挑战，如下图所示。

![](http://wx4.sinaimg.cn/mw690/00630Defgy1g4vn9xl3pcj30iy0dswgr.jpg)



按照这种球的摆法，世界上貌似没有一根木棒可以将它们 完美分开。但天使毕竟有法力，他一拍桌子，便让这些球飞到了空中，然后凭借 念力抓起一张纸片，插在了两类球的中间。从魔鬼的角度看这些 球，则像是被一条曲线完美的切开了。

![](http://wx2.sinaimg.cn/mw690/00630Defgy1g4vnbaltf7j30mo0ar77n.jpg)

后来，“无聊”的科学家们把这些球称为“数据”，把木棍称为“分类面”，找到最 大间隔的木棒位置的过程称为“优化”，拍桌子让球飞到空中的念力叫“核映射”，在 空中分隔球的纸片称为“分类超平面”。这便是SVM的童话故事。

### 1.2 理解SVM：第一层

支持向量机，因其英文名为support vector machine，故一般简称SVM，通俗来讲，它是一种二类分类模型，其基本模型定义为特征空间上的间隔最大的线性分类器，其学习策略便是间隔最大化，最终可转化为一个凸二次规划问题的求解。

**线性分类器：**给定一些数据点，它们分别属于两个不同的类，现在要找到一个线性分类器把这些数据分成两类。如果用x表示数据点，用y表示类别（y可以取1或者0，分别代表两个不同的类），一个线性分类器的学习目标便是要在n维的数据空间中找到一个超平面（hyper plane），这个超平面的方程可以表示为（ wT中的T代表转置）：

![](https://latex.codecogs.com/gif.latex?w^Tx+b=0)

这里可以查看我之前的逻辑回归章节回顾：[点击打开](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/2.Logistics%20Regression/2.Logistics%20Regression.md)

这个超平面可以用分类函数  ![](https://latex.codecogs.com/gif.latex?f(x)=w^Tx+b)表示，当f(x) 等于0的时候，x便是位于超平面上的点，而f(x)大于0的点对应 y=1 的数据点，f(x)小于0的点对应y=-1的点，如下图所示：

![](http://wx4.sinaimg.cn/mw690/00630Defly1g4w7oendezj30gz0aojrq.jpg)

#### 1.2.1 函数间隔与几何间隔

在超平面w*x+b=0确定的情况下，|w*x+b|能够表示点x到距离超平面的远近，而通过观察w*x+b的符号与类标记y的符号是否一致可判断分类是否正确，所以，可以用(y*(w*x+b))的正负性来判定或表示分类的正确性。于此，我们便引出了**函数间隔**（functional margin）的概念。

函数间隔公式： ![](https://latex.codecogs.com/gif.latex?\gamma=y(w^Tx+b)=yf(x))

而超平面(w，b)关于数据集T中所有样本点(xi，yi)的函数间隔最小值（其中，x是特征，y是结果标签，i表示第i个样本），便为超平面(w, b)关于训练数据集T的函数间隔：

![](https://latex.codecogs.com/gif.latex?\gamma=min\gamma_{}i(i=1,...n))

但这样定义的函数间隔有问题，即如果成比例的改变w和b（如将它们改成2w和2b），则函数间隔的值f(x)却变成了原来的2倍（虽然此时超平面没有改变），所以只有函数间隔还远远不够。

**几何间隔**

事实上，我们可以对法向量w加些约束条件，从而引出真正定义点到超平面的距离--几何间隔（geometrical margin）的概念。假定对于一个点 x ，令其垂直投影到超平面上的对应点为 x0 ，w 是垂直于超平面的一个向量，$\gamma$为样本x到超平面的距离，如下图所示：

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64153146493788072777.png)

这里我直接给出几何间隔的公式，详细推到请查看博文：[点击进入](https://blog.csdn.net/v_july_v/article/details/7624837)

几何间隔： ![](https://latex.codecogs.com/gif.latex?\gamma^{'}=\frac{\gamma}{||w||})

从上述函数间隔和几何间隔的定义可以看出：几何间隔就是**函数间隔除以||w||**，而且函数间隔y*(wx+b) = y*f(x)实际上就是|f(x)|，只是人为定义的一个间隔度量，而几何间隔|f(x)|/||w||才是直观上的点到超平面的距离。

#### 1.2.2 最大间隔分类器的定义

对一个数据点进行分类，当超平面离数据点的“间隔”越大，分类的确信度（confidence）也越大。所以，为了使得分类的确信度尽量高，需要让所选择的超平面能够最大化这个“间隔”值。这个间隔就是下图中的Gap的一半。

![](http://wx4.sinaimg.cn/mw690/00630Defly1g4w7q9cnymj30dt09mglu.jpg)

通过由前面的分析可知：函数间隔不适合用来最大化间隔值，因为在超平面固定以后，可以等比例地缩放w的长度和b的值，这样可以使得  ![](https://latex.codecogs.com/gif.latex?f(x)=w^Tx+b)的值任意大，亦即函数间隔可以在超平面保持不变的情况下被取得任意大。但几何间隔因为除上了，使得在缩放w和b的时候几何间隔的值是不会改变的，它只随着超平面的变动而变动，因此，这是更加合适的一个间隔。换言之，这里要找的最大间隔分类超平面中的**“间隔”指的是几何间隔。**

如下图所示，中间的实线便是寻找到的最优超平面（Optimal Hyper Plane），其到两条虚线边界的距离相等，这个距离便是几何间隔，两条虚线间隔边界之间的距离等于2倍几何间隔，而虚线间隔边界上的点则是支持向量。由于这些支持向量刚好在虚线间隔边界上，所以它们满足 ![](https://latex.codecogs.com/gif.latex?y(w_Tx+b)=1)，对于所有不是支持向量的点，则显然有 ![](https://latex.codecogs.com/gif.latex?y(w_Tx+b)>1)。

![](http://wx2.sinaimg.cn/mw690/00630Defly1g4w7r9zvunj30o60hdaar.jpg)

OK，到此为止，算是了解到了SVM的第一层，对于那些只关心怎么用SVM的朋友便已足够，不必再更进一层深究其更深的原理。

#### 1.2.3 最大间隔损失函数Hinge loss

SVM 求解使通过建立二次规划原始问题，引入拉格朗日乘子法，然后转换成对偶的形式去求解，这是一种理论非常充实的解法。这里换一种角度来思考，在机器学习领域，一般的做法是经验风险最小化 （empirical risk minimization,ERM），即构建假设函数（Hypothesis）为输入输出间的映射，然后采用损失函数来衡量模型的优劣。求得使损失最小化的模型即为最优的假设函数，采用不同的损失函数也会得到不同的机器学习算法。SVM采用的就是Hinge Loss，用于“最大间隔(max-margin)”分类。

![](https://latex.codecogs.com/gif.latex?L_i=\sum_{j\neq_{}t_i}max(0,f(x_i,W)_j-(f(x_i,W)_{y_i}-\bigtriangleup)))

- 对于训练集中的第i个数据xi
- 在W下会有一个得分结果向量f(xi,W)
- 第j类的得分为我们记作f(xi,W)j 

要理解这个公式，首先先看下面这张图片：

![](http://wx1.sinaimg.cn/mw690/00630Defly1g4w5ezjr64j30se03pmy6.jpg)

1. 在生活中我们都会认为没有威胁的才是最好的，比如拿成绩来说，自己考了第一名99分，而第二名紧随其后98分，那么就会有不安全的感觉，就会认为那家伙随时都有可能超过我。如果第二名是85分，那就会感觉安全多了，第二名需要花费很大的力气才能赶上自己。拿这个例子套到上面这幅图也是一样的。
2. 上面这幅图delta左边的红点是一个**安全警戒线**，什么意思呢？也就是说**预测错误得分**超过这个安全警戒线就会得到一个惩罚权重，让这个预测错误值退回到安全警戒线以外，这样才能够保证预测正确的结果具有唯一性。
3. 对应到公式中， ![](https://latex.codecogs.com/gif.latex?f(x_i,W)_j)就是错误分类的得分。后面一项就是 **正确得分 - delta = 安全警戒线值**，两项的差代表的就是惩罚权重，越接近正确得分，权重越大。当错误得分在警戒线以外时，两项相减得到负数，那么损失函数的最大值是0，也就是没有损失。
4. 一直往复循环训练数据，直到最小化损失函数为止，也就找到了分类超平面。

### 1.3 深入SVM：第二层

#### 1.3.1 从线性可分到线性不可分

接着考虑之前得到的目标函数(令函数间隔=1)：

![](https://latex.codecogs.com/gif.latex?max\frac{1}{||w||}s.t.,y_i(w^Tx_i+b)\ge1,i=1,...,n)

**转换为对偶问题**，解释一下什么是对偶问题，对偶问题是实质相同但从不同角度提出不同提法的一对问题。

由于求  ![](https://latex.codecogs.com/gif.latex?\frac{1}{||w||})的最大值相当于求  ![](https://latex.codecogs.com/gif.latex?\frac{1}{2}||w||^2)的最小值，所以上述目标函数等价于（w由分母变成分子，从而也有原来的max问题变为min问题，很明显，两者问题等价）：

![](https://latex.codecogs.com/gif.latex?min\frac{1}{2}||w||^2s.t.,y_i(w^Tx_i+b)\ge1,i=1,...,n)

因为现在的目标函数是二次的，约束条件是线性的，所以它是一个凸二次规划问题。这个问题可以用现成的QP (Quadratic Programming) 优化包进行求解。一言以蔽之：在一定的约束条件下，目标最优，损失最小。

此外，由于这个问题的特殊结构，还可以通过拉格朗日对偶性（Lagrange Duality）变换到对偶变量 (dual variable) 的优化问题，即通过求解与原问题等价的对偶问题（dual problem）得到原始问题的最优解，这就是线性可分条件下支持向量机的对偶算法，这样做的优点在于：**一者对偶问题往往更容易求解；二者可以自然的引入核函数，进而推广到非线性分类问题。**

详细过程请参考文章末尾给出的参考链接。

#### 1.3.2 核函数Kernel

事实上，大部分时候数据并不是线性可分的，这个时候满足这样条件的超平面就根本不存在。在上文中，我们已经了解到了SVM处理线性可分的情况，那对于非线性的数据SVM咋处理呢？对于非线性的情况，SVM 的处理方法是选择一个核函数 κ(⋅,⋅) ，通过将数据映射到高维空间，来解决在原始空间中线性不可分的问题。

具体来说，**在线性不可分的情况下，支持向量机首先在低维空间中完成计算，然后通过核函数将输入空间映射到高维特征空间，最终在高维特征空间中构造出最优分离超平面，从而把平面上本身不好分的非线性数据分开。**如图所示，一堆数据在二维空间无法划分，从而映射到三维空间里划分：

![](http://wx3.sinaimg.cn/mw690/00630Defly1g4w7t050xfj30nl0bajrv.jpg)

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase6415311358438441728.gif)

通常人们会从一些常用的核函数中选择（根据问题和数据的不同，选择不同的参数，实际上就是得到了不同的核函数），例如：**多项式核、高斯核、线性核。**

读者可能还是没明白核函数到底是个什么东西？我再简要概括下，即以下三点：

1. 实际中，我们会经常遇到线性不可分的样例，此时，我们的常用做法是把样例特征映射到高维空间中去(映射到高维空间后，相关特征便被分开了，也就达到了分类的目的)；
2. 但进一步，如果凡是遇到线性不可分的样例，一律映射到高维空间，那么这个维度大小是会高到可怕的。那咋办呢？
3. 此时，核函数就隆重登场了，核函数的价值在于它虽然也是将特征进行从低维到高维的转换，但核函数绝就绝在它事先在低维上进行计算，而将实质上的分类效果表现在了高维上，避免了直接在高维空间中的复杂计算。

**如果数据中出现了离群点outliers，那么就可以使用松弛变量来解决。**

#### 1.3.3 总结

不准确的说，SVM它本质上即是一个分类方法，用 w^T+b 定义分类函数，于是求w、b，为寻最大间隔，引出1/2||w||^2，继而引入拉格朗日因子，化为对拉格朗日乘子a的求解（求解过程中会涉及到一系列最优化或凸二次规划等问题），如此，求w.b与求a等价，而a的求解可以用一种快速学习算法SMO，至于核函数，是为处理非线性情况，若直接映射到高维计算恐维度爆炸，故在低维计算，等效高维表现。

OK，理解到这第二层，已经能满足绝大部分人一窥SVM原理的好奇心，针对于面试来说已经足够了。

### 1.4 SVM的应用

SVM在很多诸如**文本分类，图像分类，生物序列分析和生物数据挖掘，手写字符识别等领域有很多的应用**，但或许你并没强烈的意识到，SVM可以成功应用的领域远远超出现在已经在开发应用了的领域。

## 2. SVM的一些问题

1. 是否存在一组参数使SVM训练误差为0？

   答：存在

2. 训练误差为0的SVM分类器一定存在吗？

   答：一定存在

3. 加入松弛变量的SVM的训练误差可以为0吗？

   答：使用SMO算法训练的线性分类器并不一定能得到训练误差为0的模型。这是由 于我们的优化目标改变了，并不再是使训练误差最小。

4. **带核的SVM为什么能分类非线性问题?**

   答：核函数的本质是两个函数的內积，通过核函数将其隐射到高维空间，在高维空间非线性问题转化为线性问题, SVM得到超平面是高维空间的线性分类平面。其分类结果也视为低维空间的非线性分类结果, 因而带核的SVM就能分类非线性问题。

5. **如何选择核函数？**

   - 如果特征的数量大到和样本数量差不多，则选用LR或者线性核的SVM；
   - 如果特征的数量小，样本的数量正常，则选用SVM+高斯核函数；
   - 如果特征的数量小，而样本的数量很大，则需要手工添加一些特征从而变成第一种情况。

## 3. LR和SVM的联系与区别

### 3.1 相同点

- 都是线性分类器。本质上都是求一个最佳分类超平面。
- 都是监督学习算法。
- 都是判别模型。判别模型不关心数据是怎么生成的，它只关心信号之间的差别，然后用差别来简单对给定的一个信号进行分类。常见的判别模型有：KNN、SVM、LR，常见的生成模型有：朴素贝叶斯，隐马尔可夫模型。

### 3.2 不同点

- LR是参数模型，svm是非参数模型，linear和rbf则是针对数据线性可分和不可分的区别；
- 从目标函数来看，区别在于逻辑回归采用的是logistical loss，SVM采用的是hinge loss，这两个损失函数的目的都是增加对分类影响较大的数据点的权重，减少与分类关系较小的数据点的权重。 
- SVM的处理方法是只考虑support vectors，也就是和分类最相关的少数点，去学习分类器。而逻辑回归通过非线性映射，大大减小了离分类平面较远的点的权重，相对提升了与分类最相关的数据点的权重。 
- 逻辑回归相对来说模型更简单，好理解，特别是大规模线性分类时比较方便。而SVM的理解和优化相对来说复杂一些，SVM转化为对偶问题后,分类只需要计算与少数几个支持向量的距离,这个在进行复杂核函数计算时优势很明显,能够大大简化模型和计算。 
- logic 能做的 svm能做，但可能在准确率上有问题，svm能做的logic有的做不了。

## 4. 线性分类器与非线性分类器的区别以及优劣

线性和非线性是针对模型参数和输入特征来讲的；比如输入x，模型y=ax+ax^2 那么就是非线性模型，如果输入是x和X^2则模型是线性的。

- 线性分类器可解释性好，计算复杂度较低，不足之处是模型的拟合效果相对弱些。

  LR,贝叶斯分类，单层感知机、线性回归

- 非线性分类器效果拟合能力较强，不足之处是数据量不足容易过拟合、计算复杂度高、可解释性不好。

  决策树、RF、GBDT、多层感知机

**SVM两种都有（看线性核还是高斯核）**

## 5. 代码实现

新闻分类   GitHub：[点击进入](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/4.%20SVM/news%20classification)

## 6. 参考文献

[支持向量机通俗导论（理解SVM的三层境界）](https://blog.csdn.net/v_july_v/article/details/7624837)

> 作者：[@mantchs](https://github.com/NLP-LOVE/ML-NLP)
>
> GitHub：[https://github.com/NLP-LOVE/ML-NLP](https://github.com/NLP-LOVE/ML-NLP)
>
> 欢迎大家加入讨论！共同完善此项目！群号：【541954936】<a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=863f915b9178560bd32ca07cd090a7d9e6f5f90fcff5667489697b1621cecdb3"><img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="NLP面试学习群" title="NLP面试学习群"></a>


---

### news classification

## 目录
- [1.新闻分类案例](#1新闻分类案例)
  - [1.1介绍](#11介绍)
  - [1.2数据集下载](#12数据集下载)
  - [1.3libsvm库安装](#13libsvm库安装)
  - [1.4实现步骤](#14实现步骤)
  - [1.5代码实现](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/4.%20SVM/news%20classification/svm_classification.ipynb)

## 1.新闻分类案例

### 1.1介绍

这是一个预测新闻分类的案例，通过给定的数据集来预测测试集的新闻分类，该案例用到的是libsvm库，实现步骤我已经写到代码里面了，每一步都有注释，相信很多人都能够看得明白。

### 1.2数据集下载

因为数据集比较大，不适合放到github里，所以单独下载吧，放到与代码同级目录即可。

有三个文件，一个是训练数据，一个是测试数据，一个是分类。

训练数据：https://pan.baidu.com/s/1ZkxGIvvGml3vig-9_s1pRw

百度网盘加速下载地址：https://www.baiduwp.com/?m=index

### 1.3libsvm库安装

LIBSVM是台湾大学林智仁(Lin Chih-Jen)教授等开发设计的一个简单、易于使用和快速有效的SVM[模式识别](https://baike.baidu.com/item/%E6%A8%A1%E5%BC%8F%E8%AF%86%E5%88%AB/295301)与回归的软件包。其它的svm库也有，这里以libsvm为例。

libsvm下载地址：[libsvm-3.23.zip](http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/libsvm.cgi?+http://www.csie.ntu.edu.tw/~cjlin/libsvm+zip)

#### MAC系统

1.下载libsvm后解压，进入目录有个文件：**libsvm.so.2**，把这个文件复制到python安装目录**site-packages/**下。

2.在**site-packages/**下新建libsvm文件夹，并进入**libsvm**目录新建init.py的空文件。

3.进入libsvm解压路径：**libsvm-3.23/python/**，把里面的三个文件：**svm.py、svmutil.py、commonutil.py**，复制到新建的：**site-packages/libsvm/**目录下。之后就可以使用libsvm了。

#### Windows系统

安装教程：https://www.cnblogs.com/bbn0111/p/8318629.html
### 1.4实现步骤

1.先对数据集进行分词，本案例用的是**jieba**分词。

2.对分词的结果进行词频统计，分配词ID。

3.根据词ID生成词向量，这就是最终的训练数据。

4.调用libsvm训练器进行训练。

### 1.5代码实现
GitHub：[点击进入](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/4.%20SVM/news%20classification/svm_classification.ipynb)

> 作者：[@mantchs](https://github.com/NLP-LOVE/ML-NLP)
>
> GitHub：[https://github.com/NLP-LOVE/ML-NLP](https://github.com/NLP-LOVE/ML-NLP)
>
> 欢迎大家加入讨论！共同完善此项目！群号：【541954936】<a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=863f915b9178560bd32ca07cd090a7d9e6f5f90fcff5667489697b1621cecdb3"><img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="NLP面试学习群" title="NLP面试学习群"></a>

---

# 5.1 Bayes Network (贝叶斯网络)

## 目录
- [1. 对概率图模型的理解](#1-对概率图模型的理解)
- [2. 细数贝叶斯网络](#2-细数贝叶斯网络)
  - [2.1 频率派观点](#21-频率派观点)
  - [2.2 贝叶斯学派](#22-贝叶斯学派)
  - [2.3 贝叶斯定理](#23-贝叶斯定理)
  - [2.4 贝叶斯网络](#24-贝叶斯网络)
  - [2.5 朴素贝叶斯](#25-朴素贝叶斯)
- [3. 基于贝叶斯的一些问题](#3-基于贝叶斯的一些问题)
- [4. 生成式模型和判别式模型的区别](#4-生成式模型和判别式模型的区别)
- [5. 代码实现](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/5.1%20Bayes%20Network/Naive%20Bayes%20Classifier.ipynb)
- [6. 参考文献](#6-参考文献)

## 1. 对概率图模型的理解

概率图模型是用图来表示变量概率依赖关系的理论，结合概率论与图论的知识，利用图来表示与模型有关的变量的联合概率分布。由图灵奖获得者Pearl开发出来。

如果用一个词来形容概率图模型（Probabilistic Graphical Model）的话，那就是“优雅”。对于一个实际问题，我们希望能够挖掘隐含在数据中的知识。概率图模型构建了这样一幅图，用观测结点表示观测到的数据，用隐含结点表示潜在的知识，用边来描述知识与数据的相互关系，**最后基于这样的关系图获得一个概率分布**，非常“优雅”地解决了问题。

概率图中的节点分为隐含节点和观测节点，边分为有向边和无向边。从概率论的角度，节点对应于随机变量，边对应于随机变量的依赖或相关关系，其中**有向边表示单向的依赖，无向边表示相互依赖关系**。 

概率图模型分为**贝叶斯网络（Bayesian Network）和马尔可夫网络（Markov Network）**两大类。贝叶斯网络可以用一个有向图结构表示，马尔可夫网络可以表 示成一个无向图的网络结构。更详细地说，概率图模型包括了朴素贝叶斯模型、最大熵模型、隐马尔可夫模型、条件随机场、主题模型等，在机器学习的诸多场景中都有着广泛的应用。 

## 2. 细数贝叶斯网络

### 2.1 频率派观点

长久以来，人们对一件事情发生或不发生的概率，只有固定的0和1，即要么发生，要么不发生，从来不会去考虑某件事情发生的概率有多大，不发生的概率又是多大。而且概率虽然未知，但最起码是一个确定的值。比如如果问那时的人们一个问题：“有一个袋子，里面装着若干个白球和黑球，请问从袋子中取得白球的概率是多少？”他们会想都不用想，会立马告诉你，取出白球的概率就是1/2，要么取到白球，要么取不到白球，即θ只能有一个值，而且不论你取了多少次，取得白球的**概率θ始终都是1/2**，即不随观察结果X 的变化而变化。

这种**频率派**的观点长期统治着人们的观念，直到后来一个名叫Thomas Bayes的人物出现。

### 2.2 贝叶斯学派

托马斯·贝叶斯Thomas Bayes（1702-1763）在世时，并不为当时的人们所熟知，很少发表论文或出版著作，与当时学术界的人沟通交流也很少，用现在的话来说，贝叶斯就是活生生一民间学术“屌丝”，可这个“屌丝”最终发表了一篇名为“An essay towards solving a problem in the doctrine of chances”，翻译过来则是：机遇理论中一个问题的解。你可能觉得我要说：这篇论文的发表随机产生轰动效应，从而奠定贝叶斯在学术史上的地位。

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155385172674171677.jpg)

这篇论文可以用上面的例子来说明，“有一个袋子，里面装着若干个白球和黑球，请问从袋子中取得白球的概率θ是多少？”贝叶斯认为取得白球的概率是个不确定的值，因为其中含有机遇的成分。比如，一个朋友创业，你明明知道创业的结果就两种，即要么成功要么失败，但你依然会忍不住去估计他创业成功的几率有多大？你如果对他为人比较了解，而且有方法、思路清晰、有毅力、且能团结周围的人，你会不由自主的估计他创业成功的几率可能在80%以上。这种不同于最开始的“非黑即白、非0即1”的思考方式，便是**贝叶斯式的思考方式。**

**先简单总结下频率派与贝叶斯派各自不同的思考方式：**

- 频率派把需要推断的参数θ看做是固定的未知常数，即概率虽然是未知的，但最起码是确定的一个值，同时，样本X 是随机的，所以频率派重点研究样本空间，大部分的概率计算都是针对样本X 的分布；
- 而贝叶斯派的观点则截然相反，他们认为参数是随机变量，而样本X 是固定的，由于样本是固定的，所以他们重点研究的是参数的分布。

贝叶斯派既然把看做是一个随机变量，所以要计算的分布，便得事先知道的无条件分布，即在有样本之前（或观察到X之前），有着怎样的分布呢？

比如往台球桌上扔一个球，这个球落会落在何处呢？如果是不偏不倚的把球抛出去，那么此球落在台球桌上的任一位置都有着相同的机会，即球落在台球桌上某一位置的概率服从均匀分布。这种在实验之前定下的属于基本前提性质的分布称为**先验分布，或着无条件分布**。

其中，先验信息一般来源于经验跟历史资料。比如林丹跟某选手对决，解说一般会根据林丹历次比赛的成绩对此次比赛的胜负做个大致的判断。再比如，某工厂每天都要对产品进行质检，以评估产品的不合格率θ，经过一段时间后便会积累大量的历史资料，这些历史资料便是先验知识，有了这些先验知识，便在决定对一个产品是否需要每天质检时便有了依据，如果以往的历史资料显示，某产品的不合格率只有0.01%，便可视为信得过产品或免检产品，只每月抽检一两次，从而省去大量的人力物力。

而**后验分布**π（θ|X）一般也认为是在给定样本X的情况下的θ条件分布，而使π（θ|X）达到最大的值θMD称为**最大后验估计**，类似于经典统计学中的**极大似然估计**。

综合起来看，则好比是人类刚开始时对大自然只有少得可怜的先验知识，但随着不断观察、实验获得更多的样本、结果，使得人们对自然界的规律摸得越来越透彻。所以，贝叶斯方法既符合人们日常生活的思考方式，也符合人们认识自然的规律，经过不断的发展，最终占据统计学领域的半壁江山，与经典统计学分庭抗礼。

### 2.3 贝叶斯定理

**条件概率**（又称后验概率）就是事件A在另外一个事件B已经发生条件下的发生概率。条件概率表示为P(A|B)，读作“在B条件下A的概率”。

![](https://gss3.bdstatic.com/7Po3dSag_xI4khGkpoWK1HF6hhy/baike/c0%3Dbaike80%2C5%2C5%2C80%2C26/sign=28ccd0787e310a55d029d6a6d62c28cc/5243fbf2b21193136f84b53b62380cd790238d72.jpg)

比如上图，在同一个样本空间Ω中的事件或者子集A与B，如果随机从Ω中选出的一个元素属于B，那么这个随机选择的元素还属于A的概率就定义为在B的前提下A的条件概率：

![](https://latex.codecogs.com/gif.latex?P(A|B)=\frac{P(A\cap_{.}B)}{P(B)})

联合概率： ![](https://latex.codecogs.com/gif.latex?P(A\cap_{.}B)或者P(A,B))

边缘概率(先验概率)：P(A)或者P(B)

### 2.4 贝叶斯网络

贝叶斯网络(Bayesian network)，又称信念网络(Belief Network)，或有向无环图模型(directed acyclic graphical model)，是一种概率图模型，于1985年由Judea Pearl首先提出。它是一种模拟人类推理过程中因果关系的不确定性处理模型，其网络拓朴结构是一个有向无环图(DAG)。 

![](http://wx1.sinaimg.cn/mw690/00630Defgy1g4wsx3zf05j306o06xglm.jpg)

贝叶斯网络的有向无环图中的节点表示随机变量 ![](https://latex.codecogs.com/gif.latex?\{X_1,X_2,...,X_n\})

它们可以是可观察到的变量，或隐变量、未知参数等。认为有因果关系（或非条件独立）的变量或命题则用箭头来连接。若两个节点间以一个单箭头连接在一起，表示其中一个节点是“因(parents)”，另一个是“果(children)”，两节点就会产生一个条件概率值。

例如，假设节点E直接影响到节点H，即E→H，则用从E指向H的箭头建立结点E到结点H的有向弧(E,H)，权值(即连接强度)用条件概率P(H|E)来表示，如下图所示：

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155385260415416726.png)

简言之，把某个研究系统中涉及的随机变量，根据是否条件独立绘制在一个有向图中，就形成了贝叶斯网络。其主要用来描述随机变量之间的条件依赖，用圈表示随机变量(random variables)，用箭头表示条件依赖(conditional dependencies)。

此外，对于任意的随机变量，其联合概率可由各自的局部条件概率分布相乘而得出：

![](https://latex.codecogs.com/gif.latex?P(x_1,...,x_k)=P(x_k|x_1,...,x_{k-1})...P(x_2|x_1)P(x_1))

#### 2.4.1 贝叶斯网络的结构形式

**1. head-to-head**

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155385304281092472.png)

依上图，所以有：P(a,b,c) = P(a)*P(b)*P(c|a,b)成立，即在c未知的条件下，a、b被阻断(blocked)，是独立的，称之为head-to-head条件独立。

**2. tail-to-tail**

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155385308065053106.png)

考虑c未知，跟c已知这两种情况：

1. 在c未知的时候，有：P(a,b,c)=P(c)*P(a|c)*P(b|c)，此时，没法得出P(a,b) = P(a)P(b)，即c未知时，a、b不独立。
2. 在c已知的时候，有：P(a,b|c)=P(a,b,c)/P(c)，然后将P(a,b,c)=P(c)*P(a|c)*P(b|c)带入式子中，得到：P(a,b|c)=P(a,b,c)/P(c) = P(c)*P(a|c)*P(b|c) / P(c) = P(a|c)*P(b|c)，即c已知时，a、b独立。

**3. head-to-tail**

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155385312510915900.png)

还是分c未知跟c已知这两种情况：

1. c未知时，有：P(a,b,c)=P(a)*P(c|a)*P(b|c)，但无法推出P(a,b) = P(a)P(b)，即c未知时，a、b不独立。

2. c已知时，有：P(a,b|c)=P(a,b,c)/P(c)，且根据P(a,c) = P(a)*P(c|a) = P(c)*P(a|c)，可化简得到：

   ![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155385314416808319.jpg)

   所以，在c给定的条件下，a，b被阻断(blocked)，是独立的，称之为head-to-tail条件独立。

   这个head-to-tail其实就是一个链式网络，如下图所示：

   ![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155385317268812285.png)

   根据之前对head-to-tail的讲解，我们已经知道，在xi给定的条件下，xi+1的分布和x1,x2…xi-1条件独立。意味着啥呢？意味着：xi+1的分布状态只和xi有关，和其他变量条件独立。通俗点说，当前状态只跟上一状态有关，跟上上或上上之前的状态无关。这种顺次演变的随机过程，就叫做**马尔科夫链**（Markov chain）。对于马尔科夫链我们下一节再细讲。

#### 2.4.2 因子图

wikipedia上是这样定义因子图的：将一个具有多变量的全局函数因子分解，得到几个局部函数的乘积，以此为基础得到的一个双向图叫做因子图（Factor Graph）。

通俗来讲，所谓因子图就是对函数进行因子分解得到的**一种概率图**。一般内含两种节点：变量节点和函数节点。我们知道，一个全局函数通过因式分解能够分解为多个局部函数的乘积，这些局部函数和对应的变量关系就体现在因子图上。

举个例子，现在有一个全局函数，其因式分解方程为：

![](https://latex.codecogs.com/gif.latex?g(x_1,x_2,x_3,x_4,x_5)=f_A(x_1)f_B(x_2)f_C(x1,x2,x3)f_D(x_3,x_4)f_E(x_3,x_5))

其中fA,fB,fC,fD,fE为各函数，表示变量之间的关系，可以是条件概率也可以是其他关系。其对应的因子图为：

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155385445056239438.jpg)

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155385446168745485.png)

在概率图中，求某个变量的边缘分布是常见的问题。这问题有很多求解方法，其中之一就是把贝叶斯网络或马尔科夫随机场转换成因子图，然后用sum-product算法求解。换言之，基于因子图可以用**sum-product 算法**高效的求各个变量的边缘分布。

详细的sum-product算法过程，请查看博文：[从贝叶斯方法谈到贝叶斯网络](https://blog.csdn.net/v_july_v/article/details/40984699)

### 2.5 朴素贝叶斯

朴素贝叶斯(Naive Bayesian)是经典的机器学习算法之一，也是为数不多的基于概率论的分类算法。朴素贝叶斯原理简单，也很容易实现，多用于文本分类，比如垃圾邮件过滤。**朴素贝叶斯可以看做是贝叶斯网络的特殊情况：即该网络中无边，各个节点都是独立的。 **

朴素贝叶斯朴素在哪里呢？ —— **两个假设**：

- 一个特征出现的概率与其他特征（条件）独立； 
- 每个特征同等重要。

贝叶斯公式如下：

![](http://wx3.sinaimg.cn/mw690/00630Defly1g4y19ya865j30hf01qjrd.jpg)

下面以一个例子来解释朴素贝叶斯，给定数据如下：

![](http://wx2.sinaimg.cn/mw690/00630Defly1g4xzs3aydjj30nb0c4gmq.jpg)

现在给我们的问题是，如果一对男女朋友，男生想女生求婚，男生的四个特点分别是不帅，性格不好，身高矮，不上进，请你判断一下女生是嫁还是不嫁？

这是一个典型的分类问题，转为数学问题就是比较p(嫁|(不帅、性格不好、身高矮、不上进))与p(不嫁|(不帅、性格不好、身高矮、不上进))的概率，谁的概率大，我就能给出嫁或者不嫁的答案！这里我们联系到朴素贝叶斯公式：

![](http://wx2.sinaimg.cn/mw690/00630Defly1g4y1ayvvndj30io01qgll.jpg)

我们需要求p(嫁|(不帅、性格不好、身高矮、不上进),这是我们不知道的，但是通过朴素贝叶斯公式可以转化为好求的三个量，这三个变量都能通过统计的方法求得。

等等，为什么这个成立呢？学过概率论的同学可能有感觉了，这个等式成立的条件需要特征之间相互独立吧！对的！这也就是为什么朴素贝叶斯分类有朴素一词的来源，朴素贝叶斯算法是假设各个特征之间相互独立，那么这个等式就成立了！

**但是为什么需要假设特征之间相互独立呢？**

1. 我们这么想，假如没有这个假设，那么我们对右边这些概率的估计其实是不可做的，这么说，我们这个例子有4个特征，其中帅包括{帅，不帅}，性格包括{不好，好，爆好}，身高包括{高，矮，中}，上进包括{不上进，上进}，那么四个特征的联合概率分布总共是4维空间，总个数为2*3*3*2=36个。

   36个，计算机扫描统计还可以，但是现实生活中，往往有非常多的特征，每一个特征的取值也是非常之多，那么通过统计来估计后面概率的值，变得几乎不可做，这也是为什么需要假设特征之间独立的原因。

2. 假如我们没有假设特征之间相互独立，那么我们统计的时候，就需要在整个特征空间中去找，比如统计p(不帅、性格不好、身高矮、不上进|嫁),我们就需要在嫁的条件下，去找四种特征全满足分别是不帅，性格不好，身高矮，不上进的人的个数，这样的话，由于数据的稀疏性，很容易统计到0的情况。 这样是不合适的。

根据上面俩个原因，朴素贝叶斯法对条件概率分布做了条件独立性的假设，由于这是一个较强的假设，朴素贝叶斯也由此得名！这一假设使得朴素贝叶斯法变得简单，但有时会牺牲一定的分类准确率。

**朴素贝叶斯优点**：

- 算法逻辑简单,易于实现（算法思路很简单，只要使用贝叶斯公式转化即可！）
- 分类过程中时空开销小（假设特征相互独立，只会涉及到二维存储）

**朴素贝叶斯缺点**：

理论上，朴素贝叶斯模型与其他分类方法相比具有最小的误差率。但是实际上并非总是如此，这是因为朴素贝叶斯模型假设属性之间相互独立，这个假设在实际应用中往往是不成立的，在属性个数比较多或者属性之间相关性较大时，分类效果不好。

朴素贝叶斯模型(Naive Bayesian Model)的**朴素(Naive)的含义是"很简单很天真"**地假设样本特征彼此独立. 这个假设现实中基本上不存在, 但特征相关性很小的实际情况还是很多的, 所以这个模型仍然能够工作得很好。

## 3. 基于贝叶斯的一些问题

1. 解释朴素贝叶斯算法里面的先验概率、似然估计和边际似然估计？
   - **先验概率：**就是因变量（二分法）在数据集中的比例。这是在你没有任何进一步的信息的时候，是对分类能做出的最接近的猜测。
   - **似然估计：**似然估计是在其他一些变量的给定的情况下，一个观测值被分类为1的概率。例如，“FREE”这个词在以前的垃圾邮件使用的概率就是似然估计。
   - **边际似然估计：**边际似然估计就是，“FREE”这个词在任何消息中使用的概率。

## 4. 生成式模型和判别式模型的区别

- **判别模型**(discriminative model)通过求解条件概率分布P(y|x)或者直接计算y的值来预测y。

  线性回归（Linear Regression）,逻辑回归（Logistic Regression）,支持向量机（SVM）, 传统神经网络（Traditional Neural Networks）,线性判别分析（Linear Discriminative Analysis），条件随机场（Conditional Random Field）

- **生成模型**（generative model）通过对观测值和标注数据计算联合概率分布P(x,y)来达到判定估算y的目的。

  朴素贝叶斯（Naive Bayes）, 隐马尔科夫模型（HMM）,贝叶斯网络（Bayesian Networks）和隐含狄利克雷分布（Latent Dirichlet Allocation）、混合高斯模型

## 5. 代码实现

新闻分类 GitHub：[点击进入](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/5.1%20Bayes%20Network/Naive%20Bayes%20Classifier.ipynb)

## 6. 参考文献

[从贝叶斯方法谈到贝叶斯网络](https://blog.csdn.net/v_july_v/article/details/40984699)

> 作者：[@mantchs](https://github.com/NLP-LOVE/ML-NLP)
>
> GitHub：[https://github.com/NLP-LOVE/ML-NLP](https://github.com/NLP-LOVE/ML-NLP)
>
> 欢迎大家加入讨论！共同完善此项目！群号：【541954936】<a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=863f915b9178560bd32ca07cd090a7d9e6f5f90fcff5667489697b1621cecdb3"><img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="NLP面试学习群" title="NLP面试学习群"></a>

---

# 5.2 Markov (马尔可夫)

## 目录

- [1. 马尔可夫网络、马尔可夫模型、马尔可夫过程、贝叶斯网络的区别](#1-马尔可夫网络马尔可夫模型马尔可夫过程贝叶斯网络的区别)
- [2. 马尔可夫模型](#2-马尔可夫模型)
  - [2.1 马尔可夫过程](#21-马尔可夫过程)
- [3. 隐马尔可夫模型(HMM)](#3-隐马尔可夫模型hmm)
  - [3.1 隐马尔可夫三大问题](#31-隐马尔可夫三大问题)
- [4. 马尔可夫网络](#4-马尔可夫网络)
  - [4.1 因子图](#41-因子图)
  - [4.2 马尔可夫网络](#42-马尔可夫网络)
- [5. 条件随机场(CRF)](#5-条件随机场crf)
- [6. EM算法、HMM、CRF的比较](#6-em算法hmmcrf的比较)
- [7. 参考文献](#7-参考文献)
- [8. 词性标注代码实现](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/5.2%20Markov/5.2%20HMM.ipynb)

## 1. 马尔可夫网络、马尔可夫模型、马尔可夫过程、贝叶斯网络的区别

相信大家都看过上一节我讲得贝叶斯网络，都明白了概率图模型是怎样构造的，如果现在还没明白，请看我上一节的总结：[贝叶斯网络](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/5.1%20Bayes%20Network/5.1%20Bayes%20Network.md)

这一节我们重点来讲一下马尔可夫，正如题目所示，看了会一脸蒙蔽，好在我们会一点一点的来解释上面的概念，请大家按照顺序往下看就会完全弄明白了，这里我给一个通俗易懂的定义，后面我们再来一个个详解。

以下共分六点说明这些概念，分成条目只是方便边阅读边思考，这6点是依次递进的，不要跳跃着看。

1. 将随机变量作为结点，若两个随机变量相关或者不独立，则将二者连接一条边；若给定若干随机变量，则形成一个有向图，即构成一个**网络**。
2. 如果该网络是有向无环图，则这个网络称为**贝叶斯网络。**
3. 如果这个图退化成线性链的方式，则得到**马尔可夫模型**；因为每个结点都是随机变量，将其看成各个时刻(或空间)的相关变化，以随机过程的视角，则可以看成是**马尔可夫过程**。
4. 若上述网络是无向的，则是无向图模型，又称**马尔可夫随机场或者马尔可夫网络**。
5. 如果在给定某些条件的前提下，研究这个马尔可夫随机场，则得到**条件随机场**。
6. 如果使用条件随机场解决标注问题，并且进一步将条件随机场中的网络拓扑变成线性的，则得到**线性链条件随机场**。

## 2. 马尔可夫模型

### 2.1 马尔可夫过程

马尔可夫过程（Markov process）是一类随机过程。它的原始模型马尔可夫链，由俄国数学家A.A.马尔可夫于1907年提出。该过程具有如下特性：在已知目前状态（现在）的条件下，它未来的演变（将来）不依赖于它以往的演变 (过去 )。例如森林中动物头数的变化构成——马尔可夫过程。在现实世界中，有很多过程都是马尔可夫过程，如液体中微粒所作的布朗运动、传染病受感染的人数、车站的候车人数等，都可视为马尔可夫过程。

每个状态的转移只依赖于之前的n个状态，这个过程被称为1个n阶的模型，其中n是影响转移状态的数目。最简单的马尔可夫过程就是一阶过程，**每一个状态的转移只依赖于其之前的那一个状态**，这个也叫作**马尔可夫性质**。用数学表达式表示就是下面的样子：

假设这个模型的每个状态都只依赖于之前的状态，这个假设被称为**马尔科夫假设**，这个假设可以大大的简化这个问题。显然，这个假设可能是一个非常糟糕的假设，导致很多重要的信息都丢失了。

![](https://latex.codecogs.com/gif.latex?P(X_{n+1}|X_1=x_1,X_2=x_2,...,X_n=x_n)=P(X_{n+1}=x|X_n=x_n))

假设天气服从**马尔可夫链**：

![](http://wx4.sinaimg.cn/mw690/00630Defly1g4y8tb8xekj30b602z74n.jpg)

从上面这幅图可以看出：

- 假如今天是晴天，明天变成阴天的概率是0.1
- 假如今天是晴天，明天任然是晴天的概率是0.9，和上一条概率之和为1，这也符合真实生活的情况。

|        | 晴   | 阴   |
| ------ | ---- | ---- |
| **晴** | 0.9  | 0,1  |
| **阴** | 0.5  | 0.5  |

由上表我们可以得到马尔可夫链的**状态转移矩阵**：

![](http://wx3.sinaimg.cn/mw690/00630Defly1g4y92k2basj306i0220su.jpg)

因此，一阶马尔可夫过程定义了以下三个部分：

- **状态**：晴天和阴天
- **初始向量**：定义系统在时间为0的时候的状态的概率
- **状态转移矩阵**：每种天气转换的概率

马尔可夫模型（Markov Model）是一种统计模型，广泛应用在语音识别，词性自动标注，音字转换，概率文法等各个自然语言处理等应用领域。经过长期发展，尤其是在语音识别中的成功应用，使它成为一种通用的统计工具。到目前为止，它一直被认为是实现快速精确的语音识别系统的最成功的方法。

## 3. 隐马尔可夫模型(HMM)

在某些情况下马尔科夫过程不足以描述我们希望发现的模式。回到之前那个天气的例子，一个隐居的人可能不能直观的观察到天气的情况，但是有一些海藻。民间的传说告诉我们海藻的状态在某种概率上是和天气的情况相关的。在这种情况下我们有两个状态集合，一个可以观察到的状态集合（海藻的状态）和一个隐藏的状态（天气的状况）。我们希望能找到一个算法可以根据海藻的状况和马尔科夫假设来预测天气的状况。

而这个算法就叫做**隐马尔可夫模型(HMM)**。

![](http://wx3.sinaimg.cn/mw690/00630Defly1g4yaq7ndgij30nd0c9mzy.jpg)

隐马尔可夫模型 (Hidden Markov Model) 是一种统计模型，用来描述一个含有隐含未知参数的马尔可夫过程。**它是结构最简单的动态贝叶斯网，这是一种著名的有向图模型**，主要用于时序数据建模，在语音识别、自然语言处理等领域有广泛应用。

### 3.1 隐马尔可夫三大问题

1. 给定模型，如何有效计算产生观测序列的概率？换言之，如何评估模型与观测序列之间的匹配程度？
2. 给定模型和观测序列，如何找到与此观测序列最匹配的状态序列？换言之，如何根据观测序列推断出隐藏的模型状态？
3. 给定观测序列，如何调整模型参数使得该序列出现的概率最大？换言之，如何训练模型使其能最好地描述观测数据？

前两个问题是模式识别的问题：1) 根据隐马尔科夫模型得到一个可观察状态序列的概率(**评价**)；2) 找到一个隐藏状态的序列使得这个序列产生一个可观察状态序列的概率最大(**解码**)。第三个问题就是根据一个可以观察到的状态序列集产生一个隐马尔科夫模型（**学习**）。

对应的三大问题解法：

1. 向前算法(Forward Algorithm)、向后算法(Backward Algorithm)
2. 维特比算法(Viterbi Algorithm)
3. 鲍姆-韦尔奇算法(Baum-Welch Algorithm)   (约等于EM算法)

下面我们以一个场景来说明这些问题的解法到底是什么？

小明现在有三天的假期，他为了打发时间，可以在每一天中选择三件事情来做，这三件事情分别是散步、购物、打扫卫生(**对应着可观测序列**)，可是在生活中我们所做的决定一般都受到天气的影响，可能晴天的时候想要去购物或者散步，可能下雨天的时候不想出门，留在家里打扫卫生。而天气(晴天、下雨天)就属于隐藏状态，用一幅概率图来表示这一马尔可夫过程：

![](http://wx1.sinaimg.cn/mw690/00630Defly1g4ycb8jrazj30k90ftdi0.jpg)

那么，我们提出三个问题，分别对应马尔可夫的三大问题：

1. 已知整个模型，我观测到连续三天做的事情是：散步，购物，收拾。那么，根据模型，计算产生这些行为的概率是多少。
2. 同样知晓这个模型，同样是这三件事，我想猜，这三天的天气是怎么样的。
3. 最复杂的，我只知道这三天做了这三件事儿，而其他什么信息都没有。我得建立一个模型，晴雨转换概率，第一天天气情况的概率分布，根据天气情况选择做某事的概率分布。

下面我们就依据这个场景来一一解答这些问题。

#### 3.1.1 第一个问题解法

**遍历算法**：

这个是最简单的算法了，假设第一天(T=1 时刻)是晴天，想要购物，那么就把图上的对应概率相乘就能够得到了。

第二天(T=2 时刻)要做的事情，在第一天的概率基础上乘上第二天的概率，依次类推，最终得到这三天(T=3 时刻)所要做的事情的概率值，这就是遍历算法，简单而又粗暴。但问题是用遍历算法的复杂度会随着观测序列和隐藏状态的增加而成指数级增长。

复杂度为： ![](https://latex.codecogs.com/gif.latex?2TN^T)

于是就有了第二种算法

**前向算法**：

1. 假设第一天要购物，那么就计算出第一天购物的概率(包括晴天和雨天)；假设第一天要散步，那么也计算出来，依次枚举。
2. 假设前两天是购物和散步，也同样计算出这一种的概率；假设前两天是散步和打扫卫生，同样计算，枚举出前两天行为的概率。
3. 第三步就是计算出前三天行为的概率。

细心的读者已经发现了，第二步中要求的概率可以在第一步的基础上进行，同样的，第三步也会依赖于第二步的计算结果。那么这样做就能够**节省很多计算环节，类似于动态规划**。

这种算法的复杂度为： ![](https://latex.codecogs.com/gif.latex?N^2T)

**后向算法**

跟前向算法相反，我们知道总的概率肯定是1，那么B_t=1，也就是最后一个时刻的概率合为1，先计算前三天的各种可能的概率，在计算前两天、前一天的数据，跟前向算法相反的计算路径。

#### 3.1.2 第二个问题解法

**维特比算法(Viterbi)**

> 说起安德鲁·维特比（Andrew Viterbi），通信行业之外的人可能知道他的并不多，不过通信行业的从业者大多知道以他的名字命名的维特比算法（ViterbiAlgorithm）。维特比算法是现代数字通信中最常用的算法，同时也是很多自然语言处理采用的解码算法。可以毫不夸张地讲，维特比是对我们今天的生活影响力最大的科学家之一，因为基于CDMA的3G移动通信标准主要就是他和厄文·雅各布（Irwin Mark Jacobs）创办的高通公司（Qualcomm）制定的，并且高通公司在4G时代依然引领移动通信的发展。

维特比算法是一个特殊但应用最广的**动态规划算法**。利用动态规划，可以解决任何一个图中的最短路径问题。而维特比算法是针对一个特殊的图—篱笆网络（Lattice）的有向图最短路径问题而提出的。它之所以重要，是因为凡是使用隐含马尔可夫模型描述的问题都可以用它来解码，包括今天的数字通信、语音识别、机器翻译、拼音转汉字、分词等。

维特比算法一般用于模式识别，通过观测数据来反推出隐藏状态，下面一步步讲解这个算法。

![](http://wx4.sinaimg.cn/mw690/00630Defgy1g50m54c1xmj30nk0gpqel.jpg)

因为是要根据观测数据来反推，所以这里要进行一个假设，**假设这三天所做的行为分别是：散步、购物、打扫卫生，**那么我们要求的是这三天的天气(路径)分别是什么。

1. 初始计算第一天下雨和第一天晴天去散步的概率值：

    ![](https://latex.codecogs.com/gif.latex?\bigtriangleup_1(R))表示第一天下雨的概率

    ![](https://latex.codecogs.com/gif.latex?\pi_R)表示中间的状态(下雨)s概率

    ![](https://latex.codecogs.com/gif.latex?b_R(O_1=w))表示下雨并且散步的概率

    ![](https://latex.codecogs.com/gif.latex?a_{R-R})表示下雨天到下雨天的概率

![](https://latex.codecogs.com/gif.latex?\bigtriangleup_1(R)=\pi_R*b_R(O_1=w)=0.6*0.1=0.06)

![](https://latex.codecogs.com/gif.latex?\bigtriangleup_1(S)=\pi_S*b_S(O_1=w)=0.4*0.6=0.24)

   初始路径为：

​    ![](https://latex.codecogs.com/gif.latex?\phi_1(R)=Rainy)

​    ![](https://latex.codecogs.com/gif.latex?\phi_1(S)=Sunny)

2. 计算第二天下雨和第二天晴天去购物的概率值：

   ![](http://wx4.sinaimg.cn/mw690/00630Defly1g50tj6zxhaj330c0lkkjo.jpg)
   
   对应路径为：
   
   ![](http://wx2.sinaimg.cn/mw690/00630Defly1g50tj9ihr6j330s0sonph.jpg)
   
3. 计算第三天下雨和第三天晴天去打扫卫生的概率值：

   ![](http://wx4.sinaimg.cn/mw690/00630Defly1g50tvha741j331c0jg7wk.jpg)

   对应路径为：

   ![](http://wx2.sinaimg.cn/mw690/00630Defly1g50tvjsu61j32nk0kwnpf.jpg)

4. 比较每一步中 $\bigtriangleup$ 的概率大小，选取**最大值**并找到对应的路径，依次类推就能找到最有可能的**隐藏状态路径**。 

   第一天的概率最大值为  ![](https://latex.codecogs.com/gif.latex?\bigtriangleup_1S)，对应路径为Sunny，

   第二天的概率最大值为  ![](https://latex.codecogs.com/gif.latex?\bigtriangleup_2S)，对应路径为Sunny，

   第三天的概率最大值为  ![](https://latex.codecogs.com/gif.latex?\bigtriangleup_3R)，对应路径为Rainy。

5. 合起来的路径就是Sunny->Sunny->Rainy，这就是我们所求。

以上是比较通俗易懂的维特比算法，如果需要严谨表述，可以查看《数学之美》这本书的第26章，讲的就是维特比算法，很详细。附：《数学之美》下载地址，[点击下载](https://www.lanzous.com/i3ousch)

#### 3.1.3 第三个问题解法

鲍姆-韦尔奇算法(Baum-Welch Algorithm)   (约等于EM算法)，详细讲解请见：[监督学习方法与Baum-Welch算法](https://blog.csdn.net/qq_37334135/article/details/86302735)

## 4. 马尔可夫网络

### 4.1 因子图

wikipedia上是这样定义因子图的：将一个具有多变量的全局函数因子分解，得到几个局部函数的乘积，以此为基础得到的一个双向图叫做因子图（Factor Graph）。

通俗来讲，所谓因子图就是对函数进行因子分解得到的**一种概率图**。一般内含两种节点：变量节点和函数节点。我们知道，一个全局函数通过因式分解能够分解为多个局部函数的乘积，这些局部函数和对应的变量关系就体现在因子图上。

举个例子，现在有一个全局函数，其因式分解方程为：

![](https://latex.codecogs.com/gif.latex?g(x_1,x_2,x_3,x_4,x_5)=f_A(x_1)f_B(x_2)f_C(x1,x2,x3)f_D(x_3,x_4)f_E(x_3,x_5))

其中fA,fB,fC,fD,fE为各函数，表示变量之间的关系，可以是条件概率也可以是其他关系。其对应的因子图为：

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155385445056239438.jpg)

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155385446168745485.png)

### 4.2 马尔可夫网络

我们已经知道，有向图模型，又称作贝叶斯网络，但在有些情况下，强制对某些结点之间的边增加方向是不合适的。**使用没有方向的无向边，形成了无向图模型**（Undirected Graphical Model,UGM）, 又被称为**马尔可夫随机场或者马尔可夫网络**（Markov Random Field,  MRF or Markov network）。

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase6415538545181179588.png)

设X=(X1,X2…Xn)和Y=(Y1,Y2…Ym)都是联合随机变量，若随机变量Y构成一个无向图 G=(V,E)表示的马尔可夫随机场（MRF），则条件概率分布P(Y|X)称为**条件随机场**（Conditional Random Field, 简称CRF，后续新的博客中可能会阐述CRF）。如下图所示，便是一个线性链条件随机场的无向图模型：

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155385453918339567.jpg)

在概率图中，求某个变量的边缘分布是常见的问题。这问题有很多求解方法，其中之一就是把贝叶斯网络或马尔可夫随机场转换成因子图，然后用sum-product算法求解。换言之，基于因子图可以用**sum-product 算法**高效的求各个变量的边缘分布。

详细的sum-product算法过程，请查看博文：[从贝叶斯方法谈到贝叶斯网络](https://blog.csdn.net/v_july_v/article/details/40984699)

## 5. 条件随机场(CRF)

**一个通俗的例子**

假设你有许多小明同学一天内不同时段的照片，从小明提裤子起床到脱裤子睡觉各个时间段都有（小明是照片控！）。现在的任务是对这些照片进行分类。比如有的照片是吃饭，那就给它打上吃饭的标签；有的照片是跑步时拍的，那就打上跑步的标签；有的照片是开会时拍的，那就打上开会的标签。问题来了，你准备怎么干？

一个简单直观的办法就是，不管这些照片之间的时间顺序，想办法训练出一个多元分类器。就是用一些打好标签的照片作为训练数据，训练出一个模型，直接根据照片的特征来分类。例如，如果照片是早上6:00拍的，且画面是黑暗的，那就给它打上睡觉的标签;如果照片上有车，那就给它打上开车的标签。

乍一看可以！但实际上，由于我们忽略了这些照片之间的时间顺序这一重要信息，我们的分类器会有缺陷的。举个例子，假如有一张小明闭着嘴的照片，怎么分类？显然难以直接判断，需要参考闭嘴之前的照片，如果之前的照片显示小明在吃饭，那这个闭嘴的照片很可能是小明在咀嚼食物准备下咽，可以给它打上吃饭的标签；如果之前的照片显示小明在唱歌，那这个闭嘴的照片很可能是小明唱歌瞬间的抓拍，可以给它打上唱歌的标签。

所以，为了让我们的分类器能够有更好的表现，在为一张照片分类时，我们必须将与它**相邻的照片的标签信息考虑进来。这——就是条件随机场(CRF)大显身手的地方！**这就有点类似于词性标注了，只不过把照片换成了句子而已，本质上是一样的。

如同马尔可夫随机场，条件随机场为具有无向的[图模型](https://baike.baidu.com/item/图模型)，图中的顶点代表随机变量，顶点间的连线代表随机变量间的相依关系，在条件随机场中，[随机变量](https://baike.baidu.com/item/随机变量)Y 的分布为条件机率，给定的观察值则为随机变量 X。下图就是一个线性连条件随机场。

![](http://wx4.sinaimg.cn/mw690/00630Defly1g5336jslurj30u00ey414.jpg)

条件概率分布P(Y|X)称为**条件随机场**。

## 6. EM算法、HMM、CRF的比较

1. **EM算法**是用于含有隐变量模型的极大似然估计或者极大后验估计，有两步组成：E步，求期望（expectation）；M步，求极大（maxmization）。本质上EM算法还是一个迭代算法，通过不断用上一代参数对隐变量的估计来对当前变量进行计算，直到收敛。注意：EM算法是对初值敏感的，而且EM是不断求解下界的极大化逼近求解对数似然函数的极大化的算法，也就是说**EM算法不能保证找到全局最优值**。对于EM的导出方法也应该掌握。 

2. **隐马尔可夫模型**是用于标注问题的生成模型。有几个参数（π，A，B）：初始状态概率向量π，状态转移矩阵A，观测概率矩阵B。称为马尔科夫模型的三要素。马尔科夫三个基本问题：

   概率计算问题：给定模型和观测序列，计算模型下观测序列输出的概率。–》前向后向算法

   学习问题：已知观测序列，估计模型参数，即用极大似然估计来估计参数。–》Baum-Welch(也就是EM算法)和极大似然估计。

   预测问题：已知模型和观测序列，求解对应的状态序列。–》近似算法（贪心算法）和维比特算法（动态规划求最优路径）

3. **条件随机场CRF**，给定一组输入随机变量的条件下另一组输出随机变量的条件概率分布密度。条件随机场假设输出变量构成马尔科夫随机场，而我们平时看到的大多是线性链条随机场，也就是由输入对输出进行预测的判别模型。求解方法为极大似然估计或正则化的极大似然估计。 

4. 之所以总把HMM和CRF进行比较，主要是因为CRF和HMM都利用了图的知识，但是CRF利用的是马尔科夫随机场（无向图），而HMM的基础是贝叶斯网络（有向图）。而且CRF也有：概率计算问题、学习问题和预测问题。大致计算方法和HMM类似，只不过不需要EM算法进行学习问题。

5. **HMM和CRF对比：**其根本还是在于基本的理念不同，一个是生成模型，一个是判别模型，这也就导致了求解方式的不同。

## 7. 参考文献

1. [条件随机场的简单理解](https://blog.csdn.net/weixin_41911765/article/details/82465697)
2. [如何轻松愉快地理解条件随机场（CRF）](https://blog.csdn.net/dcx_abc/article/details/78319246)
3. [《数学之美》](https://www.lanzous.com/i3ousch)
4. [监督学习方法与Baum-Welch算法](https://blog.csdn.net/qq_37334135/article/details/86302735)
5. [从贝叶斯方法谈到贝叶斯网络](https://blog.csdn.net/v_july_v/article/details/40984699)

## 8. 词性标注代码实现

HMM词性标注，GitHub：[点击进入](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/5.2%20Markov/5.2%20HMM.ipynb)

> 作者：[@mantchs](https://github.com/NLP-LOVE/ML-NLP)
>
> GitHub：[https://github.com/NLP-LOVE/ML-NLP](https://github.com/NLP-LOVE/ML-NLP)
>
> 欢迎大家加入讨论！共同完善此项目！群号：【541954936】<a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=863f915b9178560bd32ca07cd090a7d9e6f5f90fcff5667489697b1621cecdb3"><img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="NLP面试学习群" title="NLP面试学习群"></a>


---

# 5.3 Topic Model (主题模型)

## 目录
- [1. LDA模型是什么](#1-lda模型是什么)
  - [1.1 5个分布的理解](#11-5个分布的理解)
  - [1.2 3个基础模型的理解](#12-3个基础模型的理解)
  - [1.3 LDA模型](#13-lda模型)
- [2. 怎么确定LDA的topic个数？](#2-怎么确定lda的topic个数)
- [3. 如何用主题模型解决推荐系统中的冷启动问题？](#3-如何用主题模型解决推荐系统中的冷启动问题)
- [4. 参考文献](#4-参考文献)
- [5. 代码实现](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/5.3%20Topic%20Model/HillaryEmail.ipynb)

## 1. LDA模型是什么

LDA可以分为以下5个步骤：

- 一个函数：gamma函数。
- 四个分布：二项分布、多项分布、beta分布、Dirichlet分布。
- 一个概念和一个理念：共轭先验和贝叶斯框架。
- 两个模型：pLSA、LDA。
- 一个采样：Gibbs采样

关于LDA有两种含义，一种是线性判别分析（Linear Discriminant Analysis），一种是概率主题模型：**隐含狄利克雷分布（Latent Dirichlet Allocation，简称LDA）**，本文讲后者。

按照wiki上的介绍，LDA由Blei, David M.、Ng, Andrew Y.、Jordan于2003年提出，是一种主题模型，它可以将文档集 中每篇文档的主题以概率分布的形式给出，从而通过分析一些文档抽取出它们的主题（分布）出来后，便可以根据主题（分布）进行主题聚类或文本分类。同时，它是一种典型的词袋模型，即一篇文档是由一组词构成，词与词之间没有先后顺序的关系。此外，一篇文档可以包含多个主题，文档中每一个词都由其中的一个主题生成。

人类是怎么生成文档的呢？首先先列出几个主题，然后以一定的概率选择主题，以一定的概率选择这个主题包含的词汇，最终组合成一篇文章。如下图所示(其中不同颜色的词语分别对应上图中不同主题下的词)。

![](http://wx4.sinaimg.cn/mw690/00630Defgy1g5f9yubudij30gh05rq4o.jpg)

那么LDA就是跟这个反过来：**根据给定的一篇文档，反推其主题分布。**

在LDA模型中，一篇文档生成的方式如下：

- 从狄利克雷分布 ![](https://latex.codecogs.com/gif.latex?\alpha)中取样生成文档 i 的主题分布 ![](https://latex.codecogs.com/gif.latex?\theta_i)。
- 从主题的多项式分布![](https://latex.codecogs.com/gif.latex?\theta_i) 中取样生成文档i第 j 个词的主题![](https://latex.codecogs.com/gif.latex?z_{i,j})。
- 从狄利克雷分布 ![](https://latex.codecogs.com/gif.latex?\beta)中取样生成主题 ![](https://latex.codecogs.com/gif.latex?z_{i,j}) 对应的词语分布![](https://latex.codecogs.com/gif.latex?\phi_{z_{i,j}})。
- 从词语的多项式分布 ![](https://latex.codecogs.com/gif.latex?\phi_{z_{i,j}})中采样最终生成词语 ![](https://latex.codecogs.com/gif.latex?w_{i,j})。

其中，类似Beta分布是二项式分布的共轭先验概率分布，而狄利克雷分布（Dirichlet分布）是多项式分布的共轭先验概率分布。此外，LDA的图模型结构如下图所示（类似贝叶斯网络结构）：

![](http://wx2.sinaimg.cn/mw690/00630Defgy1g5facicxbej30bi0d30t5.jpg)

### 1.1 5个分布的理解

先解释一下以上出现的概念。

1. **二项分布（Binomial distribution）**

   二项分布是从伯努利分布推进的。伯努利分布，又称两点分布或0-1分布，是一个离散型的随机分布，其中的随机变量只有两类取值，非正即负{+，-}。而二项分布即重复n次的伯努利试验，记为 ![](https://latex.codecogs.com/gif.latex?X\sim_{}b(n,p))。简言之，只做一次实验，是伯努利分布，重复做了n次，是二项分布。

2. **多项分布**

   是二项分布扩展到多维的情况。多项分布是指单次试验中的随机变量的取值不再是0-1的，而是有多种离散值可能（1,2,3...,k）。比如投掷6个面的骰子实验，N次实验结果服从K=6的多项分布。其中：

   ![](https://latex.codecogs.com/gif.latex?\sum_{i=1}^{k}p_i=1,p_i>0)

3. **共轭先验分布**

   在[贝叶斯统计](https://baike.baidu.com/item/贝叶斯统计/3431194)中，如果[后验分布](https://baike.baidu.com/item/后验分布/2022914)与[先验分布](https://baike.baidu.com/item/先验分布/7513047)属于同类，则先验分布与后验分布被称为**共轭分布**，而先验分布被称为似然函数的**共轭先验**。

4. **Beta分布**

   二项分布的共轭先验分布。给定参数 ![](https://latex.codecogs.com/gif.latex?\alpha>0) 和 ![](https://latex.codecogs.com/gif.latex?\beta>0)，取值范围为[0,1]的随机变量 x 的概率密度函数：

   ![](https://latex.codecogs.com/gif.latex?f(x;\alpha,\beta)=\frac{1}{B(\alpha,\beta)}x^{\alpha-1}(1-x)^{\beta-1})

   其中：

   ![](https://latex.codecogs.com/gif.latex?\frac{1}{B(\alpha,\beta)}=\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)})

   ![](https://latex.codecogs.com/gif.latex?\Gamma(z)=\int_{0}^{\infty}t^{z-1}e^{-t}dt)

   **注：这便是所谓的gamma函数，下文会具体阐述。**

5. **狄利克雷分布**

   是beta分布在高维度上的推广。Dirichlet分布的的密度函数形式跟beta分布的密度函数如出一辙：

   ![](https://latex.codecogs.com/gif.latex?f(x_1,x_2,...,x_k;\alpha_1,\alpha_2,...,\alpha_k)=\frac{1}{B(\alpha)}\prod_{i=1}^{k}x_i^{\alpha^i-1})

   其中

   ![](http://wx4.sinaimg.cn/mw690/00630Defgy1g5fiqijle0j30bj02h0so.jpg)

 至此，我们可以看到二项分布和多项分布很相似，Beta分布和Dirichlet 分布很相似。

如果想要深究其原理可以参考：[通俗理解LDA主题模型](https://blog.csdn.net/v_july_v/article/details/41209515)，也可以先往下走，最后在回过头来看详细的公式，就更能明白了。

总之，**可以得到以下几点信息。**

- beta分布是二项式分布的共轭先验概率分布：对于非负实数 ![](https://latex.codecogs.com/gif.latex?\alpha)和 ![](https://latex.codecogs.com/gif.latex?\beta)，我们有如下关系：

  ![](https://latex.codecogs.com/gif.latex?Beta(p|\alpha,\beta)+Count(m_1,m_2)=Beta(p|\alpha+m_1,\beta+m_2))

  其中 ![](https://latex.codecogs.com/gif.latex?(m_1,m_2))对应的是二项分布 ![](https://latex.codecogs.com/gif.latex?B(m_1+m_2,p))的记数。针对于这种观测到的数据符合二项分布，参数的先验分布和后验分布都是Beta分布的情况，就是Beta-Binomial 共轭。”

- 狄利克雷分布（Dirichlet分布）是多项式分布的共轭先验概率分布，一般表达式如下：

  ![](https://latex.codecogs.com/gif.latex?Dir(\vec{p}|\vec\alpha)+MultCount(\vec{m})=Dir(p|\vec{\alpha}+\vec{m}))

  针对于这种观测到的数据符合多项分布，参数的先验分布和后验分布都是Dirichlet 分布的情况，就是 Dirichlet-Multinomial 共轭。 ”

- 贝叶斯派思考问题的固定模式：

  先验分布 ![](https://latex.codecogs.com/gif.latex?\pi(\theta))+ 样本信息![](https://latex.codecogs.com/gif.latex?X) = 后验分布 ![](https://latex.codecogs.com/gif.latex?\pi(\theta|x))。

### 1.2 3个基础模型的理解

在讲LDA模型之前，再循序渐进理解基础模型：Unigram model、mixture of unigrams model，以及跟LDA最为接近的pLSA模型。为了方便描述，首先定义一些变量：

- ![](https://latex.codecogs.com/gif.latex?w)表示词，![](https://latex.codecogs.com/gif.latex?V)示所有单词的个数（固定值）。
- ![](https://latex.codecogs.com/gif.latex?z) 表示主题，![](https://latex.codecogs.com/gif.latex?k)主题的个数（预先给定，固定值）。
- ![](https://latex.codecogs.com/gif.latex?D=(W_1,...,W_M)) 表示语料库，其中的M是语料库中的文档数（固定值）。
- ![](https://latex.codecogs.com/gif.latex?W=(w_1,w_2,...,w_N))表示文档，其中的N表示一个文档中的词数（随机变量）。

1. **Unigram model**

   对于文档 ![](https://latex.codecogs.com/gif.latex?W=(w_1,w_2,...,w_N))，用 ![](https://latex.codecogs.com/gif.latex?p(w_n))表示词 ![](https://latex.codecogs.com/gif.latex?w_n)的先验概率，生成文档w的概率为：

   ![](https://latex.codecogs.com/gif.latex?p(W)=\prod_{n=1}^{N}p(w_n))

2. **Mixture of unigrams model**

   该模型的生成过程是：给某个文档先选择一个主题z,再根据该主题生成文档，该文档中的所有词都来自一个主题。假设主题有 ![](https://latex.codecogs.com/gif.latex?z_1,...,z_n)，生成文档w的概率为：

   ![](https://latex.codecogs.com/gif.latex?p(W)=p(z_1)\prod_{n=1}^{N}p(w_n|z_1)+...+p(z_k)\prod_{n=1}^{N}p(w_n|z_k)=\sum_{z}p(z)\prod_{n=1}^{N}p(w_n|z))

3. **PLSA模型**

   理解了pLSA模型后，到LDA模型也就一步之遥——给pLSA加上贝叶斯框架，便是LDA。

   在上面的Mixture of unigrams model中，我们假定一篇文档只有一个主题生成，可实际中，一篇文章往往有多个主题，只是这多个主题各自在文档中出现的概率大小不一样。比如介绍一个国家的文档中，往往会分别从教育、经济、交通等多个主题进行介绍。那么在pLSA中，文档是怎样被生成的呢？

   假定你一共有K个可选的主题，有V个可选的词，**咱们来玩一个扔骰子的游戏。**

   **一、**假设你每写一篇文档会制作一颗K面的“文档-主题”骰子（扔此骰子能得到K个主题中的任意一个），和K个V面的“主题-词项” 骰子（每个骰子对应一个主题，K个骰子对应之前的K个主题，且骰子的每一面对应要选择的词项，V个面对应着V个可选的词）。

   比如可令K=3，即制作1个含有3个主题的“文档-主题”骰子，这3个主题可以是：教育、经济、交通。然后令V = 3，制作3个有着3面的“主题-词项”骰子，其中，教育主题骰子的3个面上的词可以是：大学、老师、课程，经济主题骰子的3个面上的词可以是：市场、企业、金融，交通主题骰子的3个面上的词可以是：高铁、汽车、飞机。

   ![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155351982937840324.png)

   **二、**每写一个词，先扔该“文档-主题”骰子选择主题，得到主题的结果后，使用和主题结果对应的那颗“主题-词项”骰子，扔该骰子选择要写的词。

   先扔“文档-主题”的骰子，假设（以一定的概率）得到的主题是教育，所以下一步便是扔教育主题筛子，（以一定的概率）得到教育主题筛子对应的某个词：大学。

   上面这个投骰子产生词的过程简化下便是：**“先以一定的概率选取主题，再以一定的概率选取词”。**

   **三、**最后，你不停的重复扔“文档-主题”骰子和”主题-词项“骰子，重复N次（产生N个词），完成一篇文档，重复这产生一篇文档的方法M次，则完成M篇文档。

   **上述过程抽象出来即是PLSA的文档生成模型。在这个过程中，我们并未关注词和词之间的出现顺序，所以pLSA是一种词袋方法。生成文档的整个过程便是选定文档生成主题，确定主题生成词。**

   

   反过来，既然文档已经产生，那么如何根据已经产生好的文档反推其主题呢？这个利用看到的文档推断其隐藏的主题（分布）的过程（其实也就是产生文档的逆过程），便是**主题建模的目的：自动地发现文档集中的主题（分布）。**

   

   文档d和词w是我们得到的样本，可观测得到，所以对于任意一篇文档，其 ![](https://latex.codecogs.com/gif.latex?P(w_j|d_i))是已知的。从而可以根据大量已知的文档-词项信息 ![](https://latex.codecogs.com/gif.latex?P(w_j|d_i))，训练出文档-主题 ![](https://latex.codecogs.com/gif.latex?P(z_k|d_i))和主题-词项 ![](https://latex.codecogs.com/gif.latex?P(w_j|z_k))，如下公式所示：

   ![](https://latex.codecogs.com/gif.latex?P(w_j|d_i)=\sum_{k=1}^{K}P(w_j|z_k)P(z_k|d_i))

   故得到文档中每个词的生成概率为：

   ![](https://latex.codecogs.com/gif.latex?P(d_i,w_j)=P(d_i)P(w_j|d_i)=P(d_i)\sum_{k=1}^{K}P(w_j|z_k)P(z_k|d_i))

   由于 ![](https://latex.codecogs.com/gif.latex?P(d_i))可事先计算求出，而 ![](https://latex.codecogs.com/gif.latex?P(w_j|z_k)^{})和 ![](https://latex.codecogs.com/gif.latex?P(z_k|d_i))未知，所以 ![](https://latex.codecogs.com/gif.latex?\theta=(P(w_j|z_k),P(z_k|d_i)))就是我们要估计的参数（值），通俗点说，就是要最大化这个θ。

   用什么方法进行估计呢，常用的参数估计方法有极大似然估计MLE、最大后验证估计MAP、贝叶斯估计等等。因为该待估计的参数中含有隐变量z，所以我们可以考虑EM算法。详细的EM算法可以参考之前写过的 [EM算法](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/6.%20EM) 章节。

### 1.3 LDA模型

事实上，理解了pLSA模型，也就差不多快理解了LDA模型，因为LDA就是在pLSA的基础上加层贝叶斯框架，即LDA就是pLSA的贝叶斯版本（正因为LDA被贝叶斯化了，所以才需要考虑历史先验知识，才加的两个先验参数）。

下面，咱们对比下本文开头所述的LDA模型中一篇文档生成的方式是怎样的：

- 按照先验概率 ![](https://latex.codecogs.com/gif.latex?P(d_i))选择一篇文档 ![](https://latex.codecogs.com/gif.latex?d_i)。
- 从狄利克雷分布（即Dirichlet分布） ![](https://latex.codecogs.com/gif.latex?\alpha)中取样生成文档 ![](https://latex.codecogs.com/gif.latex?d_i)的主题分布 ![](https://latex.codecogs.com/gif.latex?\theta_i)，换言之，主题分布 ![](https://latex.codecogs.com/gif.latex?\theta_i)由超参数为 ![](https://latex.codecogs.com/gif.latex?\alpha)的Dirichlet分布生成。
- 从主题的多项式分布 ![](https://latex.codecogs.com/gif.latex?\theta_i)中取样生成文档 ![](https://latex.codecogs.com/gif.latex?d_i)第 j 个词的主题 ![](https://latex.codecogs.com/gif.latex?z_{i,j})。
- 从狄利克雷分布（即Dirichlet分布） ![](https://latex.codecogs.com/gif.latex?\beta)中取样生成主题 ![](https://latex.codecogs.com/gif.latex?z_{i,j})对应的词语分布 ![](https://latex.codecogs.com/gif.latex?\phi_{z_{i,j}})，换言之，词语分布 ![](https://latex.codecogs.com/gif.latex?\phi_{z_{i,j}}) 由参数为 ![](https://latex.codecogs.com/gif.latex?\beta)的Dirichlet分布生成。
- 从词语的多项式分布 ![](https://latex.codecogs.com/gif.latex?\phi_{z_{i,j}})中采样最终生成词语 ![](https://latex.codecogs.com/gif.latex?w_{i,j})。

LDA中，选主题和选词依然都是两个随机的过程，依然可能是先从主题分布{教育：0.5，经济：0.3，交通：0.2}中抽取出主题：教育，然后再从该主题对应的词分布{大学：0.5，老师：0.3，课程：0.2}中抽取出词：大学。

那PLSA跟LDA的区别在于什么地方呢？区别就在于：

PLSA中，主题分布和词分布是唯一确定的，能明确的指出主题分布可能就是{教育：0.5，经济：0.3，交通：0.2}，词分布可能就是{大学：0.5，老师：0.3，课程：0.2}。
但在LDA中，主题分布和词分布不再唯一确定不变，即无法确切给出。例如主题分布可能是{教育：0.5，经济：0.3，交通：0.2}，也可能是{教育：0.6，经济：0.2，交通：0.2}，到底是哪个我们不再确定（即不知道），因为它是随机的可变化的。但再怎么变化，也依然服从一定的分布，**即主题分布跟词分布由Dirichlet先验随机确定。正因为LDA是PLSA的贝叶斯版本，所以主题分布跟词分布本身由先验知识随机给定。**

换言之，LDA在pLSA的基础上给这两参数 ![](https://latex.codecogs.com/gif.latex?(P(z_k|d_i)、P(w_j|z_k)))加了两个先验分布的参数（贝叶斯化）：一个主题分布的先验分布Dirichlet分布 ![](https://latex.codecogs.com/gif.latex?\alpha)，和一个词语分布的先验分布Dirichlet分布 ![](https://latex.codecogs.com/gif.latex?\beta)。

综上，LDA真的只是pLSA的贝叶斯版本，文档生成后，两者都要根据文档去推断其主题分布和词语分布（即两者本质都是为了估计给定文档生成主题，给定主题生成词语的概率），只是用的参数推断方法不同，在pLSA中用极大似然估计的思想去推断两未知的固定参数，而LDA则把这两参数弄成随机变量，且加入dirichlet先验。

所以，pLSA跟LDA的本质区别就在于它们去估计未知参数所采用的思想不同，前者用的是频率派思想，后者用的是贝叶斯派思想。

LDA参数估计：**Gibbs采样**，详见文末的参考文献。

## 2. 怎么确定LDA的topic个数？

1. 基于经验 主观判断、不断调试、操作性强、最为常用。
2. 基于困惑度（主要是比较两个模型之间的好坏）。
3. 使用Log-边际似然函数的方法，这种方法也挺常用的。
4. 非参数方法：Teh提出的基于狄利克雷过程的HDP法。
5. 基于主题之间的相似度：计算主题向量之间的余弦距离，KL距离等。

## 3. 如何用主题模型解决推荐系统中的冷启动问题？ 

推荐系统中的冷启动问题是指在没有大量用户数据的情况下如何给用户进行个性化推荐，目的是最优化点击率、转化率或用户 体验（用户停留时间、留存率等）。冷启动问题一般分为用户冷启动、物品冷启动和系统冷启动三大类。

- 用户冷启动是指对一个之前没有行为或行为极少的新用户进行推荐；
- 物品冷启动是指为一个新上市的商品或电影（这时没有与之相关的 评分或用户行为数据）寻找到具有潜在兴趣的用户；
- 系统冷启动是指如何为一个 新开发的网站设计个性化推荐系统。

解决冷启动问题的方法一般是基于内容的推荐。以Hulu的场景为例，对于用 户冷启动来说，我们希望根据用户的注册信息（如：年龄、性别、爱好等）、搜 索关键词或者合法站外得到的其他信息（例如用户使用Facebook账号登录，并得 到授权，可以得到Facebook中的朋友关系和评论内容）来推测用户的兴趣主题。 得到用户的兴趣主题之后，我们就可以找到与该用户兴趣主题相同的其他用户， 通过他们的历史行为来预测用户感兴趣的电影是什么。

同样地，对于物品冷启动问题，我们也可以根据电影的导演、演员、类别、关键词等信息推测该电影所属于的主题，然后基于主题向量找到相似的电影，并将新电影推荐给以往喜欢看这 些相似电影的用户。**可以使用主题模型（pLSA、LDA等）得到用户和电影的主题。**

以用户为例，我们将每个用户看作主题模型中的一篇文档，用户对应的特征 作为文档中的单词，这样每个用户可以表示成一袋子特征的形式。通过主题模型 学习之后，经常共同出现的特征将会对应同一个主题，同时每个用户也会相应地 得到一个主题分布。每个电影的主题分布也可以用类似的方法得到。

**那么如何解决系统冷启动问题呢？**首先可以得到每个用户和电影对应的主题向量，除此之外，还需要知道用户主题和电影主题之间的偏好程度，也就是哪些主题的用户可能喜欢哪些主题的电影。当系统中没有任何数据时，我们需要一些先验知识来指定，并且由于主题的数目通常比较小，随着系统的上线，收集到少量的数据之后我们就可以对主题之间的偏好程度得到一个比较准确的估计。

## 4. 参考文献

[通俗理解LDA主题模型](https://blog.csdn.net/v_july_v/article/details/41209515)

## 5. 代码实现

[LDA模型应用：一眼看穿希拉里的邮件](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/5.3%20Topic%20Model/HillaryEmail.ipynb)





-----

> 作者：[@mantchs](https://github.com/NLP-LOVE/ML-NLP)
>
> GitHub：[https://github.com/NLP-LOVE/ML-NLP](https://github.com/NLP-LOVE/ML-NLP)
>
> 欢迎大家加入讨论！共同完善此项目！群号：【541954936】<a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=863f915b9178560bd32ca07cd090a7d9e6f5f90fcff5667489697b1621cecdb3"><img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="NLP面试学习群" title="NLP面试学习群"></a>

---

# 6. EM算法

## 目录
- [1. 什么是EM算法](#1-什么是em算法)
  - [1.1 似然函数](#11-似然函数)
  - [1.3 极大似然函数的求解步骤](#13-极大似然函数的求解步骤)
  - [1.4 EM算法](#14-em算法)
- [2. 采用 EM 算法求解的模型有哪些？](#2-采用-em-算法求解的模型有哪些)
- [3.代码实现](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/6.%20EM/gmm_em)
- [4. 参考文献](#4-参考文献)

## 1. 什么是EM算法

最大期望算法（Expectation-maximization algorithm，又译为期望最大化算法），是在概率模型中寻找参数最大似然估计或者最大后验估计的算法，其中概率模型依赖于无法观测的隐性变量。

最大期望算法经过两个步骤交替进行计算，

**第一步**是计算期望（E），利用对隐藏变量的现有估计值，计算其最大似然估计值；
**第二步**是最大化（M），最大化在E步上求得的最大似然值来计算参数的值。M步上找到的参数估计值被用于下一个E步计算中，这个过程不断交替进行。

极大似然估计用一句话概括就是：知道结果，反推条件θ。

### 1.1 似然函数

在数理统计学中，**似然函数**是一种关于统计模型中的参数的函数，表示模型参数中的似然性。“似然性”与“或然性”或“概率”意思相近，都是指某种事件发生的可能性。**而极大似然就相当于最大可能的意思。**

比如你一位同学和一位猎人一起外出打猎，一只野兔从前方窜过。只听一声枪响，野兔应声到下，如果要你推测，这一发命中的子弹是谁打的？你就会想，只发一枪便打中，由于猎人命中的概率一般大于你那位同学命中的概率，从而推断出这一枪应该是猎人射中的。

这个例子所作的推断就体现了最大似然法的基本思想。

**多数情况下我们是根据已知条件来推算结果，而最大似然估计是已经知道了结果，然后寻求使该结果出现的可能性最大的条件，以此作为估计值。**

### 1.3 极大似然函数的求解步骤

假定我们要从10万个人当中抽取100个人来做身高统计，那么抽到这100个人的概率就是(概率连乘)：

![](https://latex.codecogs.com/gif.latex?L(\theta)=L(x_1,...,x_n|\theta)=\prod_{i=1}^{n}p(x_i|\theta),\theta\in\ominus)

现在要求的就是这个 ![](https://latex.codecogs.com/gif.latex?\theta)值，也就是使得 ![](https://latex.codecogs.com/gif.latex?L(\theta))的概率最大化，那么这时的参数![](https://latex.codecogs.com/gif.latex?\theta) 就是所求。

为了便于分析，我们可以定义对数似然函数，将其变成连加的形式：

![](https://latex.codecogs.com/gif.latex?H(\theta)=lnL(\theta)=ln\prod_{i=1}^{n}p(x_i|\theta)=\sum_{i=1}^{n}lnp(x_i|\theta))

对于求一个函数的极值，通过我们在本科所学的微积分知识，最直接的设想是求导，然后让导数为0，那么解这个方程得到的θ就是了（当然，前提是函数L(θ)连续可微）。但，如果θ是包含多个参数的向量那怎么处理呢？当然是求L(θ)对所有参数的偏导数，也就是梯度了，从而n个未知的参数，就有n个方程，方程组的解就是似然函数的极值点了，最终得到这n个参数的值。

求极大似然函数估计值的一般步骤：

1. 写出似然函数；
2. 对似然函数取对数，并整理；
3. 求导数，令导数为0，得到似然方程；
4. 解似然方程，得到的参数即为所求；

### 1.4 EM算法

两枚硬币A和B，假定随机抛掷后正面朝上概率分别为PA，PB。为了估计这两个硬币朝上的概率，咱们轮流抛硬币A和B，每一轮都连续抛5次，总共5轮：

| 硬币 | 结果       | 统计    |
| ---- | ---------- | ------- |
| A    | 正正反正反 | 3正-2反 |
| B    | 反反正正反 | 2正-3反 |
| A    | 正反反反反 | 1正-4反 |
| B    | 正反反正正 | 3正-2反 |
| A    | 反正正反反 | 2正-3反 |

硬币A被抛了15次，在第一轮、第三轮、第五轮分别出现了3次正、1次正、2次正，所以很容易估计出PA，类似的，PB也很容易计算出来(**真实值**)，如下：

PA = （3+1+2）/ 15 = 0.4
PB= （2+3）/10 = 0.5

问题来了，如果我们不知道抛的硬币是A还是B呢（即硬币种类是隐变量），然后再轮流抛五轮，得到如下结果：

| 硬币    | 结果       | 统计    |
| ------- | ---------- | ------- |
| Unknown | 正正反正反 | 3正-2反 |
| Unknown | 反反正正反 | 2正-3反 |
| Unknown | 正反反反反 | 1正-4反 |
| Unknown | 正反反正正 | 3正-2反 |
| Unknown | 反正正反反 | 2正-3反 |

OK，问题变得有意思了。现在我们的目标没变，还是估计PA和PB，需要怎么做呢？

显然，此时我们多了一个硬币种类的隐变量，设为z，可以把它认为是一个5维的向量（z1,z2,z3,z4,z5)，代表每次投掷时所使用的硬币，比如z1，就代表第一轮投掷时使用的硬币是A还是B。

- 但是，这个变量z不知道，就无法去估计PA和PB，所以，我们必须先估计出z，然后才能进一步估计PA和PB。
- 可要估计z，我们又得知道PA和PB，这样我们才能用极大似然概率法则去估计z，这不是鸡生蛋和蛋生鸡的问题吗，如何破？

答案就是先随机初始化一个PA和PB，用它来估计z，然后基于z，还是按照最大似然概率法则去估计新的PA和PB，然后依次循环，如果新估计出来的PA和PB和我们真实值差别很大，直到**PA和PB收敛到真实值为止。**

我们不妨这样，先随便给PA和PB赋一个值，比如：
硬币A正面朝上的概率PA = 0.2
硬币B正面朝上的概率PB = 0.7

然后，我们看看第一轮抛掷最可能是哪个硬币。
如果是硬币A，得出3正2反的概率为 0.2*0.2*0.2*0.8*0.8 = 0.00512
如果是硬币B，得出3正2反的概率为0.7*0.7*0.7*0.3*0.3=0.03087
然后依次求出其他4轮中的相应概率。做成表格如下：

| 轮数 | 若是硬币A                               | 若是硬币B        |
| ---- | --------------------------------------- | ---------------- |
| 1    | 0.00512，即0.2 0.2 0.2 0.8 0.8，3正-2反 | 0.03087，3正-2反 |
| 2    | 0.02048，即0.2 0.2 0.8 0.8 0.8，2正-3反 | 0.01323，2正-3反 |
| 3    | 0.08192，即0.2 0.8 0.8 0.8 0.8，1正-4反 | 0.00567，1正-4反 |
| 4    | 0.00512，即0.2 0.2 0.2 0.8 0.8，3正-2反 | 0.03087，3正-2反 |
| 5    | 0.02048，即0.2 0.2 0.8 0.8 0.8，2正-3反 | 0.01323，2正-3反 |

按照最大似然法则：
第1轮中最有可能的是硬币B
第2轮中最有可能的是硬币A
第3轮中最有可能的是硬币A
第4轮中最有可能的是硬币B
第5轮中最有可能的是硬币A

我们就把概率更大，即更可能是A的，即第2轮、第3轮、第5轮出现正的次数2、1、2相加，除以A被抛的总次数15（A抛了三轮，每轮5次），作为z的估计值，B的计算方法类似。然后我们便可以按照最大似然概率法则来估计新的PA和PB。

PA = （2+1+2）/15 = 0.33
PB =（3+3）/10 = 0.6

就这样，不断迭代 不断接近真实值，这就是EM算法的奇妙之处。

可以期待，我们继续按照上面的思路，用估计出的PA和PB再来估计z，再用z来估计新的PA和PB，反复迭代下去，就可以最终得到PA = 0.4，PB=0.5，此时无论怎样迭代，PA和PB的值都会保持0.4和0.5不变，于是乎，我们就找到了PA和PB的最大似然估计。

**总结一下计算步骤：**

1. 随机初始化分布参数θ

2. E步，求Q函数，对于每一个i，计算根据上一次迭代的模型参数来计算出隐性变量的后验概率（其实就是隐性变量的期望），来作为隐藏变量的现估计值：

   ![](http://wx1.sinaimg.cn/mw690/00630Defly1g57brmij1wj307l01cq2s.jpg)

3. M步，求使Q函数获得极大时的参数取值）将似然函数最大化以获得新的参数值

   ![](http://wx1.sinaimg.cn/mw690/00630Defly1g57bswusk1j30de01s3yh.jpg)

4. 然后循环重复2、3步直到收敛。

详细的推导过程请参考文末的参考文献。

## 2. 采用 EM 算法求解的模型有哪些？

用EM算法求解的模型一般有GMM或者协同过滤，k-means其实也属于EM。EM算法一定会收敛，但是可能收敛到局部最优。由于求和的项数将随着隐变量的数目指数上升，会给梯度计算带来麻烦。

## 3.代码实现

[高斯混合模型 EM 算法](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/6.%20EM/gmm_em)

## 4. 参考文献

[如何通俗理解EM算法](https://blog.csdn.net/v_july_v/article/details/81708386)

> 作者：[@mantchs](https://github.com/NLP-LOVE/ML-NLP)
>
> GitHub：[https://github.com/NLP-LOVE/ML-NLP](https://github.com/NLP-LOVE/ML-NLP)
>
> 欢迎大家加入讨论！共同完善此项目！群号：【541954936】<a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=863f915b9178560bd32ca07cd090a7d9e6f5f90fcff5667489697b1621cecdb3"><img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="NLP面试学习群" title="NLP面试学习群"></a>

---

### gmm_em

# gmm-em-clustering
高斯混合模型（GMM 聚类）的 EM 算法实现。

在给出代码前，先作一些说明。

- 在对样本应用高斯混合模型的 EM 算法前，需要先进行数据预处理，即把所有样本值都缩放到 0 和 1 之间。
- 初始化模型参数时，要确保任意两个模型之间参数没有完全相同，否则迭代到最后，两个模型的参数也将完全相同，相当于一个模型。
- 模型的个数必须大于 1。当 K 等于 1 时相当于将样本聚成一类，没有任何意义。

**代码在本目录下的main.py和gmm.py**

# 相关文章
[高斯混合模型 EM 算法的 Python 实现](http://www.codebelief.com/article/2017/11/gmm-em-algorithm-implementation-by-python/)

---

# 7. Clustering (聚类)

## 目录
- [1. 聚类算法都是无监督学习吗?](#1-聚类算法都是无监督学习吗)
- [2. k-means(k均值)算法](#2-k-meansk均值算法)
  - [2.1 算法过程](#21-算法过程)
  - [2.2 损失函数](#22-损失函数)
  - [2.3 k值的选择](#23-k值的选择)
  - [2.4 KNN与K-means区别？](#24-knn与k-means区别)
  - [2.5 K-Means优缺点及改进](#25-k-means优缺点及改进)
- [3. 高斯混合模型(GMM)](#3-高斯混合模型gmm)
  - [3.1 GMM的思想](#31-gmm的思想)
  - [3.2 GMM与K-Means相比](#32-gmm与k-means相比)
- [4. DBSCAN聚类算法](#4-dbscan聚类算法)
  - [4.1 DBSCAN算法简介](#41-dbscan算法简介)
  - [4.2 核心概念](#42-核心概念)
  - [4.3 算法过程](#43-算法过程)
  - [4.4 参数选择](#44-参数选择)
  - [4.5 DBSCAN的优缺点](#45-dbscan的优缺点)
  - [4.6 DBSCAN与K-Means比较](#46-dbscan与k-means比较)
  - [4.7 DBSCAN的改进算法](#47-dbscan的改进算法)
  - [4.8 DBSCAN应用场景](#48-dbscan应用场景)
- [5. 聚类算法如何评估](#5-聚类算法如何评估)
- [6. 代码实现](#6-代码实现)

## 1. 聚类算法都是无监督学习吗?

什么是聚类算法？聚类是一种机器学习技术，它涉及到数据点的分组。给定一组数据点，我们可以使用聚类算法将每个数据点划分为一个特定的组。理论上，同一组中的数据点应该具有相似的属性和/或特征，而不同组中的数据点应该具有高度不同的属性和/或特征。**聚类是一种无监督学习的方法**，是许多领域中常用的统计数据分析技术。

常用的算法包括**K-MEANS、高斯混合模型（Gaussian Mixed Model，GMM）、DBSCAN（Density-Based Spatial Clustering of Applications with Noise）、自组织映射神经网络（Self-Organizing Map，SOM）**

## 2. k-means(k均值)算法

### 2.1 算法过程

K-均值是最普及的聚类算法，算法接受一个未标记的数据集，然后将数据聚类成不同的组。

K-均值是一个迭代算法，假设我们想要将数据聚类成 n 个组，其方法为:

- 首先选择𝐾个随机的点，称为聚类中心（cluster centroids）；
- 对于数据集中的每一个数据，按照距离𝐾个中心点的距离，将其与距离最近的中心点关联起来，与同一个中心点关联的所有点聚成一类。
- 计算每一个组的平均值，将该组所关联的中心点移动到平均值的位置。
- 重复步骤，直至中心点不再变化。

用  ![](https://latex.codecogs.com/gif.latex?u^1,u^2,...,u^k)来表示聚类中心，用𝑐(1),𝑐(2),...,𝑐(𝑚)来存储与第𝑖个实例数据最近的聚类中心的索引，K-均值算法的伪代码如下：

```
Repeat {
    for i = 1 to m
    c(i) := index (form 1 to K) of cluster centroid closest to x(i)
    for k = 1 to K
    μk := average (mean) of points assigned to cluster k
}
```

算法分为两个步骤，第一个 for 循环是赋值步骤，即：对于每一个样例𝑖，计算其应该属于的类。第二个 for 循环是聚类中心的移动，即：对于每一个类𝐾，重新计算该类的质心。

K-均值算法也可以很便利地用于将数据分为许多不同组，即使在没有非常明显区分的组群的情况下也可以。下图所示的数据集包含身高和体重两项特征构成的，利用 K-均值算法将数据分为三类，用于帮助确定将要生产的 T-恤衫的三种尺寸。

![](http://wx3.sinaimg.cn/mw690/00630Defly1g5b746zwjgj30fh0c7gnj.jpg)

### 2.2 损失函数

K-均值最小化问题，是要最小化所有的数据点与其所关联的聚类中心点之间的距离之和，因此 K-均值的代价函数（又称畸变函数 Distortion function）为：

![](http://wx2.sinaimg.cn/mw690/00630Defgy1g5bu9wfwotj30jp01tq2v.jpg)

其中 ![](https://latex.codecogs.com/gif.latex?u_{c^{(i)}})代表与 ![](https://latex.codecogs.com/gif.latex?x^{(i)})最近的聚类中心点。 我们的的优化目标便是找出使得代价函数最小的 ![](https://latex.codecogs.com/gif.latex?c^{(1)},c^{(2)},...,c^{(m)})和 ![](https://latex.codecogs.com/gif.latex?u_1,u_2,...,u_k)。

### 2.3 k值的选择

在运行 K-均值算法的之前，我们首先要随机初始化所有的聚类中心点，下面介绍怎样做：
1. 我们应该选择𝐾 < 𝑚，即聚类中心点的个数要小于所有训练集实例的数量。
2. 随机选择𝐾个训练实例，然后令𝐾个聚类中心分别与这𝐾个训练实例相等K-均值的一个问题在于，它有可能会停留在一个局部最小值处，而这取决于初始化的情况。

为了解决这个问题，我们通常需要多次运行 K-均值算法，每一次都重新进行随机初始化，最后再比较多次运行 K-均值的结果，选择代价函数最小的结果。这种方法在𝐾较小的时候（2--10）还是可行的，**但是如果𝐾较大，这么做也可能不会有明显地改善。**

没有所谓最好的选择聚类数的方法，通常是需要根据不同的问题，人工进行选择的。选择的时候思考我们运用 K-均值算法聚类的动机是什么。有一个可能会谈及的方法叫作**“肘部法则”**。关 于“肘部法则”，我们所需要做的是改变𝐾值，也就是聚类类别数目的总数。我们用一个聚类来运行 K 均值聚类方法。这就意味着，所有的数据都会分到一个聚类里，然后计算成本函数或者计算畸变函数𝐽。𝐾代表聚类数字。

![](http://wx2.sinaimg.cn/mw690/00630Defly1g5b7pbaeavj30qo0cwtc9.jpg)

我们可能会得到一条类似于这样的曲线。像一个人的肘部。这就是“肘部法则”所做的，让我们来看这样一个图，看起来就好像有一个很清楚的肘在那儿。你会发现这种模式，它的畸变值会迅速下降，从 1 到 2，从 2 到 3 之后，你会在 3 的时候达到一个肘点。在此之后，畸变值就下降的非常慢，看起来就像使用 3 个聚类来进行聚类是正确的，**这是因为那个点是曲线的肘点，畸变值下降得很快，𝐾 = 3之后就下降得很慢，那么我们就选𝐾 = 3。**当你应用“肘部法则”的时候，如果你得到了一个像上面这样的图，那么这将是一种用来选择聚类个数的合理方法。

### 2.4 KNN与K-means区别？

K最近邻(k-Nearest Neighbor，KNN)分类算法，是一个理论上比较成熟的方法，也是最简单的机器学习算法之一。

| KNN                                                          | K-Means                                                      |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| 1.KNN是分类算法<br/>2.属于监督学习<br/>3.训练数据集是带label的数据 | 1.K-Means是聚类算法<br/>2.属于非监督学习<br/>3.训练数据集是无label的数据，是杂乱无章的，经过聚类后变得有序，先无序，后有序。 |
| 没有明显的前期训练过程，属于memory based learning            | 有明显的前期训练过程                                         |
| K的含义：一个样本x，对它进行分类，就从训练数据集中，在x附近找离它最近的K个数据点，这K个数据点，类别c占的个数最多，就把x的label设为c。 | K的含义：K是人工固定好的数字，假设数据集合可以分为K个蔟，那么就利用训练数据来训练出这K个分类。 |

**相似点**

都包含这样的过程，给定一个点，在数据集中找离它最近的点。即二者都用到了NN(Nears Neighbor)算法思想。

### 2.5 K-Means优缺点及改进

k-means：在大数据的条件下，会耗费大量的时间和内存。 优化k-means的建议： 

1. 减少聚类的数目K。因为，每个样本都要跟类中心计算距离。 

2. 减少样本的特征维度。比如说，通过PCA等进行降维。 

3. 考察其他的聚类算法，通过选取toy数据，去测试不同聚类算法的性能。 

4. hadoop集群，K-means算法是很容易进行并行计算的。 

5. 算法可能找到局部最优的聚类，而不是全局最优的聚类。使用改进的二分k-means算法。

   二分k-means算法：首先将整个数据集看成一个簇，然后进行一次k-means（k=2）算法将该簇一分为二，并计算每个簇的误差平方和，选择平方和最大的簇迭代上述过程再次一分为二，直至簇数达到用户指定的k为止，此时可以达到的全局最优。

## 3. 高斯混合模型(GMM)

### 3.1 GMM的思想

高斯混合模型（Gaussian Mixed Model，GMM）也是一种常见的聚类算法，与K均值算法类似，同样使用了EM算法进行迭代计算。高斯混合模型假设每个簇的数据都是符合高斯分布（又叫正态分布）的，当前**数据呈现的分布就是各个簇的高斯分布叠加在一起的结果。**

第一张图是一个数据分布的样例，如果只用一个高斯分布来拟合图中的数据，图 中所示的椭圆即为高斯分布的二倍标准差所对应的椭圆。直观来说，图中的数据 明显分为两簇，因此只用一个高斯分布来拟和是不太合理的，需要推广到用多个 高斯分布的叠加来对数据进行拟合。第二张图是用两个高斯分布的叠加来拟合得到的结果。**这就引出了高斯混合模型，即用多个高斯分布函数的线形组合来对数据分布进行拟合。**理论上，高斯混合模型可以拟合出任意类型的分布。

![](http://wx4.sinaimg.cn/mw690/00630Defly1g5b8mu9mjmj30iu0apjuh.jpg)

![](http://wx3.sinaimg.cn/mw690/00630Defly1g5b8ms101aj30id0aswhh.jpg)

高斯混合模型的核心思想是，假设数据可以看作从多个高斯分布中生成出来 的。在该假设下，每个单独的分模型都是标准高斯模型，其均值 $u_i$ 和方差 $\sum_i$ 是待估计的参数。此外，每个分模型都还有一个参数 $\pi_i$，可以理解为权重或生成数据的概 率。高斯混合模型的公式为：

![](https://latex.codecogs.com/gif.latex?p(x)=\sum_{i=1}^{k}\pi_iN(x|u_i,\sum_i))

通常我们并不能直接得到高斯混合模型的参数，而是观察到了一系列 数据点，给出一个类别的数量K后，希望求得最佳的K个高斯分模型。因此，高斯 混合模型的计算，便成了最佳的均值μ，方差Σ、权重π的寻找，这类问题通常通过 最大似然估计来求解。遗憾的是，此问题中直接使用最大似然估计，得到的是一 个复杂的非凸函数，目标函数是和的对数，难以展开和对其求偏导。

**在这种情况下，可以用EM算法。 **EM算法是在最大化目标函数时，先固定一个变量使整体函数变为凸优化函数，求导得到最值，然后利用最优参数更新被固定的变量，进入下一个循环。具体到高 斯混合模型的求解，EM算法的迭代过程如下。

首先，初始随机选择各参数的值。然后，重复下述两步，直到收敛。 

- E步骤。根据当前的参数，计算每个点由某个分模型生成的概率。 
- M步骤。使用E步骤估计出的概率，来改进每个分模型的均值，方差和权重。

> 高斯混合模型是一个生成式模型。可以这样理解数据的生成过程，假设一个最简单的情况，即只有两个一维标准高斯分布的分模型*N*(0,1)和*N*(5,1)，其权重分别为0.7和0.3。那么，在生成第一个数据点时，先按照权重的比例，随机选择一个分布，比如选择第一个高斯分布，接着从*N*(0,1)中生成一个点，如−0.5，便是第一个数据点。在生成第二个数据点时，随机选择到第二个高斯分布*N*(5,1)，生成了第二个点4.7。如此循环执行，便生成出了所有的数据点。 

也就是说，我们并不知道最佳的K个高斯分布的各自3个参数，也不知道每个 数据点究竟是哪个高斯分布生成的。所以每次循环时，先固定当前的高斯分布不 变，获得每个数据点由各个高斯分布生成的概率。然后固定该生成概率不变，根据数据点和生成概率，获得一个组更佳的高斯分布。循环往复，直到参数的不再变化，或者变化非常小时，便得到了比较合理的一组高斯分布。

### 3.2 GMM与K-Means相比

高斯混合模型与K均值算法的相同点是：

- 它们都是可用于聚类的算法；
- 都需要 指定K值；
- 都是使用EM算法来求解；
- 都往往只能收敛于局部最优。

而它相比于K 均值算法的优点是，可以给出一个样本属于某类的概率是多少；不仅仅可以用于聚类，还可以用于概率密度的估计；并且可以用于生成新的样本点。

## 4. DBSCAN聚类算法

### 4.1 DBSCAN算法简介

DBSCAN(Density-Based Spatial Clustering of Applications with Noise，具有噪声的基于密度的聚类方法)是一种基于密度的空间聚类算法。该算法将具有足够密度的区域划分为簇，并在具有噪声的空间数据库中发现任意形状的簇，它将簇定义为密度相连的点的最大集合。

与K-means、GMM等算法不同，DBSCAN不需要预先指定聚类的数量，能够发现任意形状的聚类，并且能够识别出噪声点。这使得DBSCAN在处理实际应用中的复杂数据分布时具有明显优势。

### 4.2 核心概念

DBSCAN算法中有几个重要的概念：

**1. Eps(ε-邻域半径)**
给定对象半径为Eps内的区域称为该对象的Eps邻域。

**2. MinPts(最小包含点数)**
如果给定对象Eps邻域内的样本点数大于等于MinPts，则称该对象为核心对象。

**3. 核心点(Core Point)**
如果一个点的ε-邻域内至少包含MinPts个点(包括该点自身)，则该点为核心点。

**4. 边界点(Border Point)**
边界点不是核心点，但落在某个核心点的邻域内。

**5. 噪声点(Noise Point)**
既不是核心点也不是边界点的点称为噪声点或离群点。

**6. 密度直达**
如果点p在核心点q的ε-邻域内，则称p由q密度直达。

**7. 密度可达**
如果存在一个点的序列p1, p2, ..., pn，其中p1=q, pn=p，且pi+1由pi密度直达，则称p由q密度可达。

**8. 密度相连**
如果存在一个核心点o，使得点p和点q都由o密度可达，则称p和q密度相连。

### 4.3 算法过程

DBSCAN的聚类过程如下：

1. **随机选择一个未访问过的点**

2. **检查该点的ε-邻域**
   - 如果邻域内点的数量 ≥ MinPts，则该点为核心点，创建一个新簇
   - 如果邻域内点的数量 < MinPts，则标记为噪声点(暂时)

3. **对于核心点，递归处理其邻域内的点**
   - 将所有密度直达的点加入到当前簇
   - 对新加入的核心点，重复查找其邻域内的点
   - 继续扩展，直到没有新的点可以加入

4. **重复步骤1-3**，直到所有点都被访问

5. **最终未被分配到任何簇的点被标记为噪声点**

伪代码如下：

```
DBSCAN(D, eps, MinPts) {
   C = 0  // 簇计数器
   for each point P in dataset D {
      if P is visited
         continue next point
      mark P as visited
      NeighborPts = regionQuery(P, eps)
      if sizeof(NeighborPts) < MinPts
         mark P as NOISE
      else {
         C = next cluster
         expandCluster(P, NeighborPts, C, eps, MinPts)
      }
   }
}

expandCluster(P, NeighborPts, C, eps, MinPts) {
   add P to cluster C
   for each point P' in NeighborPts {
      if P' is not visited {
         mark P' as visited
         NeighborPts' = regionQuery(P', eps)
         if sizeof(NeighborPts') >= MinPts
            NeighborPts = NeighborPts joined with NeighborPts'
      }
      if P' is not yet member of any cluster
         add P' to cluster C
   }
}

regionQuery(P, eps)
   return all points within P's eps-neighborhood (including P)
```

### 4.4 参数选择

DBSCAN算法的效果很大程度上取决于参数ε和MinPts的选择：

**MinPts的选择**：
- 一般建议MinPts ≥ 维度数 + 1
- 对于二维数据，MinPts通常设置为4
- 对于噪声较多的数据集，可以适当增大MinPts
- 作为经验法则，MinPts = 2×dim通常是一个较好的起点

**ε的选择**：
- 可以使用K距离图(K-distance Graph)来确定
- 计算每个点到其第K个最近邻的距离(K=MinPts)
- 将这些距离按降序排列并绘制曲线
- 曲线中出现"拐点"的位置对应的距离值可作为ε的参考值

实际操作步骤：
1. 计算每个点到第MinPts个邻居的距离
2. 按降序排列这些距离
3. 绘制排序后的K-距离图
4. 在曲线的"肘部"(最大曲率点)处选择ε值

### 4.5 DBSCAN的优缺点

**优点**：

1. **不需要预先指定簇的数量** - 算法能够自动确定簇的数目
2. **可以发现任意形状的簇** - 不像K-means那样假设簇是球形的
3. **能够识别噪声点** - 对离群点不敏感，能有效处理噪声数据
4. **对数据输入顺序不敏感** - 除了边界点可能分配到不同的簇外
5. **簇的定义符合直觉** - 基于密度的定义更接近人类对"簇"的理解
6. **参数相对容易设定** - 只需要两个参数，且有较好的选择方法

**缺点**：

1. **对参数敏感** - ε和MinPts的选择对结果影响很大，需要一定的领域知识
2. **难以处理密度不均的数据** - 当数据密度变化较大时，同一组参数可能无法得到好的结果
3. **高维数据性能下降** - 在高维空间中，距离度量的意义减弱("维度灾难")
4. **计算复杂度较高** - 时间复杂度为O(n²)，虽然可以用空间索引优化到O(nlogn)
5. **无法给出簇的代表点** - 不像K-means有聚类中心的概念

### 4.6 DBSCAN与K-Means比较

| 特性 | DBSCAN | K-Means |
|:---|:---|:---|
| **聚类数量** | 自动确定 | 需要预先指定K值 |
| **簇的形状** | 可以是任意形状 | 假设簇是凸的，通常是球形 |
| **噪声处理** | 能够识别并标记噪声点 | 所有点都会被分配到某个簇 |
| **参数** | ε和MinPts | K值和初始中心点 |
| **算法类型** | 基于密度 | 基于距离(划分方法) |
| **时间复杂度** | O(n²)或O(nlogn)(使用索引) | O(nkt)，k为簇数，t为迭代次数 |
| **对初始化敏感** | 否(除边界点外) | 是(对初始中心点敏感) |
| **适用场景** | 任意形状的簇，有噪声 | 球形簇，数据较"干净" |
| **簇的密度** | 假设簇内密度相似 | 无此假设 |
| **可解释性** | 无聚类中心，较难解释 | 有聚类中心，容易解释 |

### 4.7 DBSCAN的改进算法

针对DBSCAN的缺点，研究者提出了多种改进算法：

**1. OPTICS (Ordering Points To Identify the Clustering Structure)**
- 不需要显式指定ε参数
- 生成簇的可达性图，可以从中提取不同密度的簇
- 适合密度不均匀的数据

**2. HDBSCAN (Hierarchical DBSCAN)**
- 结合了DBSCAN和层次聚类的优点
- 能够处理密度变化的数据
- 可以自动选择最优的ε值

**3. ST-DBSCAN (Spatial-Temporal DBSCAN)**
- 扩展到时空数据
- 考虑空间和时间两个维度的密度

**4. DBSCAN++**
- 使用更高效的数据结构(如KD-tree, R-tree)
- 降低时间复杂度

### 4.8 DBSCAN应用场景

DBSCAN在以下场景中特别有效：

1. **地理空间数据分析** - 如城市规划中识别热点区域
2. **异常检测** - 利用噪声点识别功能检测异常值
3. **图像分割** - 根据像素密度进行区域分割
4. **交通流分析** - 识别交通拥堵区域
5. **社交网络分析** - 发现社区结构
6. **天文学** - 星系和星团的识别
7. **市场细分** - 客户群体识别
8. **生物信息学** - 基因表达数据聚类

## 5. 聚类算法如何评估

由于数据以及需求的多样性，没有一种算法能够适用于所有的数据类型、数 据簇或应用场景，似乎每种情况都可能需要一种不同的评估方法或度量标准。例 如，K均值聚类可以用误差平方和来评估，但是基于密度的数据簇可能不是球形， 误差平方和则会失效。在许多情况下，判断聚类算法结果的好坏强烈依赖于主观 解释。尽管如此，聚类算法的评估还是必需的，它是聚类分析中十分重要的部分之一。

聚类评估的任务是估计在数据集上进行聚类的可行性，以及聚类方法产生结 果的质量。这一过程又分为三个子任务。

1. **估计聚类趋势。**

   这一步骤是检测数据分布中是否存在非随机的簇结构。如果数据是基本随机 的，那么聚类的结果也是毫无意义的。我们可以观察聚类误差是否随聚类类别数 量的增加而单调变化，如果数据是基本随机的，即不存在非随机簇结构，那么聚 类误差随聚类类别数量增加而变化的幅度应该较不显著，并且也找不到一个合适 的K对应数据的真实簇数。

2. **判定数据簇数。**

   确定聚类趋势之后，我们需要找到与真实数据分布最为吻合的簇数，据此判定聚类结果的质量。数据簇数的判定方法有很多，例如手肘法和Gap Statistic方 法。需要说明的是，用于评估的最佳数据簇数可能与程序输出的簇数是不同的。 例如，有些聚类算法可以自动地确定数据的簇数，但可能与我们通过其他方法确 定的最优数据簇数有所差别。

3. **测定聚类质量。**

   在无监督的情况下，我们可以通过考察簇的分离情况和簇的紧 凑情况来评估聚类的效果。定义评估指标可以展现面试者实际解决和分析问题的 能力。事实上测量指标可以有很多种，以下列出了几种常用的度量指标，更多的 指标可以阅读相关文献。

   轮廓系数、均方根标准偏差、R方（R-Square）、改进的HubertΓ统计。

## 6. 代码实现

[高斯混合模型代码](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/7.%20Clustering/GMM.ipynb)

[K-Means代码](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/7.%20Clustering/K-Means.ipynb)

> 作者：[@mantchs](https://github.com/NLP-LOVE/ML-NLP)
>
> GitHub：[https://github.com/NLP-LOVE/ML-NLP](https://github.com/NLP-LOVE/ML-NLP)
>
> 欢迎大家加入讨论！共同完善此项目！群号：【541954936】<a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=863f915b9178560bd32ca07cd090a7d9e6f5f90fcff5667489697b1621cecdb3"><img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="NLP面试学习群" title="NLP面试学习群"></a>



---

# 8. ML特征工程和优化方法

## 目录
- [1. 特征工程有哪些？](#1-特征工程有哪些)
  - [1.1 特征归一化](#11-特征归一化)
  - [1.2 类别型特征](#12-类别型特征)
  - [1.3 高维组合特征的处理](#13-高维组合特征的处理)
  - [1.4 文本表示模型](#14-文本表示模型)
  - [1.5 其它特征工程](#15-其它特征工程)
  - [1.6 特征工程脑图](#16-特征工程脑图)
- [2. 机器学习优化方法](#2-机器学习优化方法)
  - [2.1 机器学习常用损失函数](#21-机器学习常用损失函数)
  - [2.2 什么是凸优化](#22-什么是凸优化)
  - [2.3 正则化项](#23-正则化项)
  - [2.4 常见的几种最优化方法](#24-常见的几种最优化方法)
  - [2.5 降维方法](#25-降维方法)
- [3. 机器学习评估方法](#3-机器学习评估方法)
  - [3.1 准确率(Accuracy)](#31-准确率accuracy)
  - [3.2 精确率（Precision）](#32-精确率precision)
  - [3.3 召回率(Recall)](#33-召回率recall)
  - [3.4 F1值(H-mean值)](#34-f1值h-mean值)
  - [3.4 ROC曲线](#34-roc曲线)
  - [3.5 余弦距离和欧式距离](#35-余弦距离和欧式距离)
  - [3.6 A/B测试](#36-ab测试)
  - [3.7 模型评估方法](#37-模型评估方法)
  - [3.8 超参数调优](#38-超参数调优)
  - [3.9 过拟合和欠拟合](#39-过拟合和欠拟合)
- [4. 检验方法](#4-检验方法)
  - [4.1 KS检验](#41-ks检验)
  - [4.2 T检验](#42-t检验)
  - [4.3 F检验](#43-f检验)
  - [4.4 Grubbs检验](#44-grubbs检验)
  - [4.5 卡方检验](#45-卡方检验)
- [5. 参考文献](#4-参考文献)

## 1. 特征工程有哪些？

特征工程，顾名思义，是对原始数据进行一系列工程处理，将其提炼为特征，作为输入供算法和模型使用。从本质上来讲，特征工程是一个表示和展现数 据的过程。在实际工作中，**特征工程旨在去除原始数据中的杂质和冗余**，设计更高效的特征以刻画求解的问题与预测模型之间的关系。

主要讨论以下两种常用的数据类型。

1. 结构化数据。结构化数据类型可以看作关系型数据库的一张表，每列都 有清晰的定义，包含了数值型、类别型两种基本类型；每一行数据表示一个样本 的信息。
2. 非结构化数据。非结构化数据主要包括文本、图像、音频、视频数据， 其包含的信息无法用一个简单的数值表示，也没有清晰的类别定义，并且每条数 据的大小各不相同。

### 1.1 特征归一化

为了消除数据特征之间的量纲影响，我们需要对特征进行归一化处理，使得 不同指标之间具有可比性。例如，分析一个人的身高和体重对健康的影响，如果 使用米（m）和千克（kg）作为单位，那么身高特征会在1.6～1.8m的数值范围 内，体重特征会在50～100kg的范围内，分析出来的结果显然会倾向于数值差别比 较大的体重特征。想要得到更为准确的结果，就需要进行特征归一化 （Normalization）处理，使各指标处于同一数值量级，以便进行分析。

对数值类型的特征做归一化可以将所有的特征都统一到一个大致相同的数值 区间内。最常用的方法主要有以下两种。

1. **线性函数归一化**（Min-Max Scaling）。它对原始数据进行线性变换，使 结果映射到[0, 1]的范围，实现对原始数据的等比缩放。归一化公式如下，其中*X*为原始数据， ![](https://latex.codecogs.com/gif.latex?X_{max}、X_{min})分别为数据最大值和最小值。 

   ![](https://latex.codecogs.com/gif.latex?X_{norm}=\frac{X-X_{min}}{X_{max}-X_{min}})

2. **零均值归一化**（Z-Score Normalization）。它会将原始数据映射到均值为 0、标准差为1的分布上。具体来说，假设原始特征的均值为μ、标准差为σ，那么 归一化公式定义为

   ![](https://latex.codecogs.com/gif.latex?z=\frac{x-u}{\sigma})

优点：**训练数据归一化后，容易更快地通过梯度下降找 到最优解。**

![](http://wx4.sinaimg.cn/mw690/00630Defly1g5cdl44ubjj30gz08i40j.jpg)

当然，数据归一化并不是万能的。在实际应用中，通过梯度下降法求解的模 型通常是需要归一化的，包括线性回归、逻辑回归、支持向量机、神经网络等模 型。但对于决策树模型则并不适用。

### 1.2 类别型特征

类别型特征（Categorical Feature）主要是指性别（男、女）、血型（A、B、 AB、O）等只在有限选项内取值的特征。类别型特征原始输入通常是字符串形 式，除了决策树等少数模型能直接处理字符串形式的输入，对于逻辑回归、支持 向量机等模型来说，类别型特征必须经过处理转换成数值型特征才能正确工作。

1. **序号编码**

   序号编码通常用于处理类别间具有大小关系的数据。例如成绩，可以分为 低、中、高三档，并且存在“高>中>低”的排序关系。序号编码会按照大小关系对 类别型特征赋予一个数值ID，例如高表示为3、中表示为2、低表示为1，转换后依 然保留了大小关系。

2. **独热编码(one-hot)**

   独热编码通常用于处理类别间不具有大小关系的特征。例如血型，一共有4个 取值（A型血、B型血、AB型血、O型血），独热编码会把血型变成一个4维稀疏 向量，A型血表示为（1, 0, 0, 0），B型血表示为（0, 1, 0, 0），AB型表示为（0, 0, 1, 0），O型血表示为（0, 0, 0, 1）。对于类别取值较多的情况下使用独热编码。

3. **二进制编码 **

   二进制编码主要分为两步，先用序号编码给每个类别赋予一个类别ID，然后 将类别ID对应的二进制编码作为结果。以A、B、AB、O血型为例，下图是二进制编码的过程。A型血的ID为1，二进制表示为001；B型血的ID为2，二进制表示为 010；以此类推可以得到AB型血和O型血的二进制表示。

   ![](http://wx1.sinaimg.cn/mw690/00630Defly1g5cdqz4zruj30lf07d74g.jpg)

### 1.3 高维组合特征的处理 

为了提高复杂关系的拟合能力，在特征工程中经常会把一阶离散特征两两组 合，构成高阶组合特征。以广告点击预估问题为例，原始数据有语言和类型两种 离散特征，第一张图是语言和类型对点击的影响。为了提高拟合能力，语言和类型可 以组成二阶特征，第二张图是语言和类型的组合特征对点击的影响。

![](http://wx3.sinaimg.cn/mw690/00630Defly1g5cdvbua1aj30n30kf752.jpg)

### 1.4 文本表示模型 

文本是一类非常重要的非结构化数据，如何表示文本数据一直是机器学习领 域的一个重要研究方向。

1. **词袋模型和N-gram模型**

   最基础的文本表示模型是词袋模型。顾名思义，就是将每篇文章看成一袋子 词，并忽略每个词出现的顺序。具体地说，就是将整段文本以词为单位切分开， 然后每篇文章可以表示成一个长向量，向量中的每一维代表一个单词，而该维对 应的权重则反映了这个词在原文章中的重要程度。常用TF-IDF来计算权重。

2. **主题模型**

   主题模型用于从文本库中发现有代表性的主题（得到每个主题上面词的分布 特性），并且能够计算出每篇文章的主题分布。

3. **词嵌入与深度学习模型**

   词嵌入是一类将词向量化的模型的统称，核心思想是将每个词都映射成低维 空间（通常K=50～300维）上的一个稠密向量（Dense Vector）。K维空间的每一 维也可以看作一个隐含的主题，只不过不像主题模型中的主题那样直观。

### 1.5 其它特征工程

1. 如果某个特征当中有**缺失值**，缺失比较少的话，可以使用该特征的平均值或者其它比较靠谱的数据进行填充；缺失比较多的话可以考虑删除该特征。
2. 可以分析特征与结果的相关性，把相关性小的特征去掉。

### 1.6 特征工程脑图

![](https://julyedu-img-public.oss-cn-beijing.aliyuncs.com/Public/Image/Question/1512980743_407.png)

## 2. 机器学习优化方法

优化是应用数学的一个分支，也是机器学习的核心组成部分。实际上，机器 学习算法 = 模型表征 + 模型评估 + 优化算法。其中，优化算法所做的事情就是在 模型表征空间中找到模型评估指标最好的模型。不同的优化算法对应的模型表征 和评估指标不尽相同。

### 2.1 机器学习常用损失函数

损失函数（loss function）是用来估量你模型的预测值f(x)与真实值Y的不一致程度，它是一个非负实值函数,通常使用L(Y, f(x))来表示，损失函数越小，模型的鲁棒性就越好。常见的损失函数如下：

1. **平方损失函数**

   ![](http://wx1.sinaimg.cn/mw690/00630Defly1g5e5tzcgisj30aa01hwed.jpg)

   Y-f(X)表示的是残差，整个式子表示的是残差的平方和，而我们的目的就是最小化这个目标函数值（注：该式子未加入正则项），也就是最小化残差的平方和。而在实际应用中，通常会使用均方差（MSE）作为一项衡量指标，公式如下：

   ![](https://latex.codecogs.com/gif.latex?MSE=\frac{1}{n}\sum_{i=1}^{n}(Y_i^{'}-Y_i)^2)

   该损失函数一般使用在线性回归当中。

2. **log损失函数**

   ![](https://wx1.sinaimg.cn/large/00630Defly1g4pvtz3tw9j30et04v0sw.jpg)
   
   公式中的 y=1 表示的是真实值为1时用第一个公式，真实 y=0 用第二个公式计算损失。为什么要加上log函数呢？可以试想一下，当真实样本为1是，但h=0概率，那么log0=∞，这就对模型最大的惩罚力度；当h=1时，那么log1=0，相当于没有惩罚，也就是没有损失，达到最优结果。所以数学家就想出了用log函数来表示损失函数。
   
   最后按照梯度下降法一样，求解极小值点，得到想要的模型效果。该损失函数一般使用在逻辑回归中。
   
   ![](http://wx2.sinaimg.cn/mw690/00630Defly1g5cf7z1k1rj30b40b4wej.jpg)
   
3. **Hinge损失函数**

   ![](https://latex.codecogs.com/gif.latex?L_i=\sum_{j\neq t_i}max(0,f(x_i,W)_j-(f(x_i,W)_{y_i}-\bigtriangleup)))

   SVM采用的就是Hinge Loss，用于“最大间隔(max-margin)”分类。

   ![](http://wx1.sinaimg.cn/mw690/00630Defly1g4w5ezjr64j30se03pmy6.jpg)

   详细见之前[SVM的文章1.2.3](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/4.%20SVM)

### 2.2 什么是凸优化

**凸函数**的严格定义为，函数L(·) 是凸函数当且仅当对定义域中的任意两点x，y和任意实数λ∈[0,1]总有：

![](http://wx4.sinaimg.cn/mw690/00630Defly1g5e5wisdtuj30d401ijra.jpg)

该不等式的一个直观解释是，凸函数曲面上任意两点连接而成的线段，其上的任 意一点都不会处于该函数曲面的下方，如下图所示所示。

![](http://wx4.sinaimg.cn/mw690/00630Defly1g5cfpms6woj30e1049wez.jpg)

凸优化问题的例子包括支持向量机、线性回归等 线性模型，非凸优化问题的例子包括低秩模型（如矩阵分解）、深度神经网络模型等。

### 2.3 正则化项

使用正则化项，也就是给loss function加上一个参数项，正则化项有**L1正则化、L2正则化、ElasticNet**。加入这个正则化项好处：

- 控制参数幅度，不让模型“无法无天”。
- 限制参数搜索空间
- 解决欠拟合与过拟合的问题。

详细请参考之前的文章：[线性回归--第5点](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/Liner%20Regression)

### 2.4 常见的几种最优化方法

1. **梯度下降法**

   梯度下降法是最早最简单，也是最为常用的最优化方法。梯度下降法实现简单，当目标函数是凸函数时，梯度下降法的解是全局解。一般情况下，其解不保证是全局最优解，梯度下降法的速度也未必是最快的。梯度下降法的优化思想是用当前位置负梯度方向作为搜索方向，因为该方向为当前位置的最快下降方向，所以也被称为是”最速下降法“。最速下降法越接近目标值，步长越小，前进越慢。梯度下降法的搜索迭代示意图如下图所示：

   ![](https://images2017.cnblogs.com/blog/1022856/201709/1022856-20170916201932735-243646199.png)

   缺点：靠近极小值时收敛速度减慢；直线搜索时可能会产生一些问题；可能会“之字形”地下降。

2. **牛顿法**

   牛顿法是一种在实数域和复数域上近似求解方程的方法。方法使用函数f (x)的泰勒级数的前面几项来寻找方程f (x) = 0的根。牛顿法最大的特点就在于它的收敛速度很快。具体步骤：

   - 首先，选择一个接近函数 f (x)零点的 x0，计算相应的 f (x0) 和切线斜率f  ' (x0)（这里f ' 表示函数 f  的导数）。

   - 然后我们计算穿过点(x0,  f  (x0)) 并且斜率为f '(x0)的直线和 x 轴的交点的x坐标，也就是求如下方程的解：

     ![](https://latex.codecogs.com/gif.latex?x*f^{'}(x_0)+f(x_0)-x_0*f^{'}(x_0)=0)

   - 我们将新求得的点的 x 坐标命名为x1，通常x1会比x0更接近方程f  (x) = 0的解。因此我们现在可以利用x1开始下一轮迭代。

   由于牛顿法是基于当前位置的切线来确定下一次的位置，所以牛顿法又被很形象地称为是"切线法"。牛顿法搜索动态示例图：

   ![](https://images2017.cnblogs.com/blog/1022856/201709/1022856-20170916202719078-1588446775.gif)

   从本质上去看，牛顿法是二阶收敛，梯度下降是一阶收敛，所以牛顿法就更快。**缺点：**

   - 牛顿法是一种迭代算法，每一步都需要求解目标函数的Hessian矩阵的逆矩阵，计算比较复杂。
   - 在高维情况下这个矩阵非常大，计算和存储都是问题。
   - 在小批量的情况下，牛顿法对于二阶导数的估计噪声太大。
   - 目标函数非凸的时候，牛顿法容易受到鞍点或者最大值点的吸引。

3. **拟牛顿法**

   拟牛顿法是求解非线性优化问题最有效的方法之一，**本质思想是改善牛顿法每次需要求解复杂的Hessian矩阵的逆矩阵的缺陷，它使用正定矩阵来近似Hessian矩阵的逆，从而简化了运算的复杂度。**拟牛顿法和梯度下降法一样只要求每一步迭代时知道目标函数的梯度。通过测量梯度的变化，构造一个目标函数的模型使之足以产生超线性收敛性。这类方法大大优于梯度下降法，尤其对于困难的问题。另外，因为拟牛顿法不需要二阶导数的信息，所以有时比牛顿法更为有效。如今，优化软件中包含了大量的拟牛顿算法用来解决无约束，约束，和大规模的优化问题。

4. **共轭梯度法**

   共轭梯度法是介于梯度下降法与牛顿法之间的一个方法，它仅需利用一阶导数信息，但克服了梯度下降法收敛慢的缺点，又避免了牛顿法需要存储和计算Hesse矩阵并求逆的缺点，共轭梯度法不仅是解决大型线性方程组最有用的方法之一，也是解大型非线性最优化最有效的算法之一。 在各种优化算法中，共轭梯度法是非常重要的一种。其优点是所需存储量小，具有步收敛性，稳定性高，而且不需要任何外来参数。

   具体的实现步骤请参加wiki百科[共轭梯度法](https://en.wikipedia.org/wiki/Conjugate_gradient_method#Example_code_in_MATLAB)。下图为共轭梯度法和梯度下降法搜索最优解的路径对比示意图：

   ![](http://wx2.sinaimg.cn/mw690/00630Defly1g5ch0r2p48j308z0almy4.jpg)

### 2.5 降维方法

#### 2.5.1 线性判别分析（LDA）

线性判别分析（Linear Discriminant Analysis，LDA）是一种经典的降维方法。和主成分分析PCA不考虑样本类别输出的无监督降维技术不同，LDA是一种监督学习的降维技术，数据集的每个样本有类别输出。  

LDA分类思想简单总结如下：  

1. 多维空间中，数据处理分类问题较为复杂，LDA算法将多维空间中的数据投影到一条直线上，将d维数据转化成1维数据进行处理。  
2. 对于训练数据，设法将多维数据投影到一条直线上，同类数据的投影点尽可能接近，异类数据点尽可能远离。  
3. 对数据进行分类时，将其投影到同样的这条直线上，再根据投影点的位置来确定样本的类别。  

如果用一句话概括LDA思想，**即“投影后类内方差最小，类间方差最大”。**

假设有红、蓝两类数据，这些数据特征均为二维，如下图所示。我们的目标是将这些数据投影到一维，让每一类相近的数据的投影点尽可能接近，不同类别数据尽可能远，即图中红色和蓝色数据中心之间的距离尽可能大。

![](http://wx4.sinaimg.cn/mw690/00630Defgy1g5ma79ujplj30qr0ac3z8.jpg)

左图和右图是两种不同的投影方式。

​	左图思路：让不同类别的平均点距离最远的投影方式。

​	右图思路：让同类别的数据挨得最近的投影方式。

​	从上图直观看出，右图红色数据和蓝色数据在各自的区域来说相对集中，根据数据分布直方图也可看出，所以右图的投影效果好于左图，左图中间直方图部分有明显交集。

​	以上例子是基于数据是二维的，分类后的投影是一条直线。如果原始数据是多维的，则投影后的分类面是一低维的超平面。

**优缺点**

| 优缺点 | 简要说明                                                     |
| :----: | :----------------------------------------------------------- |
|  优点  | 1. 可以使用类别的先验知识；<br />2. 以标签、类别衡量差异性的有监督降维方式，相对于PCA的模糊性，其目的更明确，更能反映样本间的差异； |
|  缺点  | 1. LDA不适合对非高斯分布样本进行降维；<br />2. LDA降维最多降到分类数k-1维；<br />3. LDA在样本分类信息依赖方差而不是均值时，降维效果不好；<br />4. LDA可能过度拟合数据。 |

#### 2.5.2 主成分分析（PCA）

1. PCA就是将高维的数据通过线性变换投影到低维空间上去。
2. 投影思想：找出最能够代表原始数据的投影方法。被PCA降掉的那些维度只能是那些噪声或是冗余的数据。
3. 去冗余：去除可以被其他向量代表的线性相关向量，这部分信息量是多余的。
4. 去噪声，去除较小特征值对应的特征向量，特征值的大小反映了变换后在特征向量方向上变换的幅度，幅度越大，说明这个方向上的元素差异也越大，要保留。
5. 对角化矩阵，寻找极大线性无关组，保留较大的特征值，去除较小特征值，组成一个投影矩阵，对原始样本矩阵进行投影，得到降维后的新样本矩阵。
6. 完成PCA的关键是——协方差矩阵。协方差矩阵，能同时表现不同维度间的相关性以及各个维度上的方差。协方差矩阵度量的是维度与维度之间的关系，而非样本与样本之间。
7. 之所以对角化，因为对角化之后非对角上的元素都是0，达到去噪声的目的。对角化后的协方差矩阵，对角线上较小的新方差对应的就是那些该去掉的维度。所以我们只取那些含有较大能量(特征值)的维度，其余的就舍掉，即去冗余。

**图解PCA**

PCA可解决训练数据中存在数据特征过多或特征累赘的问题。核心思想是将m维特征映射到n维（n < m），这n维形成主元，是重构出来最能代表原始数据的正交特征。

​	假设数据集是m个n维，$(\boldsymbol x^{(1)}, \boldsymbol x^{(2)}, \cdots, \boldsymbol x^{(m)})$。如果$n=2$，需要降维到$n'=1$，现在想找到某一维度方向代表这两个维度的数据。下图有$u_1, u_2$两个向量方向，但是哪个向量才是我们所想要的，可以更好代表原始数据集的呢？

![](F:/jianguo_syc/GitHub/DeepLearning-500-questions-master/ch02_%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80/img/ch2/2.34/1.png)

从图可看出，$u_1$比$u_2$好，为什么呢？有以下两个主要评价指标：

1. 样本点到这个直线的距离足够近。
2. 样本点在这个直线上的投影能尽可能的分开。

如果我们需要降维的目标维数是其他任意维，则：

1. 样本点到这个超平面的距离足够近。
2. 样本点在这个超平面上的投影能尽可能的分开。

**优缺点**

| 优缺点 | 简要说明                                                     |
| :----: | :----------------------------------------------------------- |
|  优点  | 1. 仅仅需要以方差衡量信息量，不受数据集以外的因素影响。　2.各主成分之间正交，可消除原始数据成分间的相互影响的因素。3. 计算方法简单，主要运算是特征值分解，易于实现。 |
|  缺点  | 1.主成分各个特征维度的含义具有一定的模糊性，不如原始样本特征的解释性强。2. 方差小的非主成分也可能含有对样本差异的重要信息，因降维丢弃可能对后续数据处理有影响。 |

#### 2.5.3 比较这两种方法

**降维的必要性**：

1. 多重共线性和预测变量之间相互关联。多重共线性会导致解空间的不稳定，从而可能导致结果的不连贯。
2. 高维空间本身具有稀疏性。一维正态分布有68%的值落于正负标准差之间，而在十维空间上只有2%。
3. 过多的变量，对查找规律造成冗余麻烦。
4. 仅在变量层面上分析可能会忽略变量之间的潜在联系。例如几个预测变量可能落入仅反映数据某一方面特征的一个组内。

**降维的目的**：

1. 减少预测变量的个数。
2. 确保这些变量是相互独立的。
3. 提供一个框架来解释结果。相关特征，特别是重要特征更能在数据中明确的显示出来；如果只有两维或者三维的话，更便于可视化展示。
4. 数据在低维下更容易处理、更容易使用。
5. 去除数据噪声。
6. 降低算法运算开销。

**LDA和PCA区别**

| 异同点 | LDA                                                          | PCA                                |
| :----: | :----------------------------------------------------------- | :--------------------------------- |
| 相同点 | 1. 两者均可以对数据进行降维；<br />2. 两者在降维时均使用了矩阵特征分解的思想；<br />3. 两者都假设数据符合高斯分布； |                                    |
| 不同点 | 有监督的降维方法；                                           | 无监督的降维方法；                 |
|        | 降维最多降到k-1维；                                          | 降维多少没有限制；                 |
|        | 可以用于降维，还可以用于分类；                               | 只用于降维；                       |
|        | 选择分类性能最好的投影方向；                                 | 选择样本点投影具有最大方差的方向； |
|        | 更明确，更能反映样本间差异；                                 | 目的较为模糊；                     |   

## 3. 机器学习评估方法

混淆矩阵也称误差矩阵，是表示精度评价的一种标准格式，用n行n列的矩阵形式来表示。具体评价指标有总体精度、制图精度、用户精度等，这些精度指标从不同的侧面反映了图像分类的精度。下图为混淆矩阵

|          | 正类                | 负类                |
| -------- | ------------------- | ------------------- |
| 预测正确 | TP(True Positives)  | FP(False Positives) |
| 预测错误 | FN(False Negatives) | TN(True Negatives)  |

### 3.1 准确率(Accuracy)

**准确率（Accuracy）。**顾名思义，就是所有的预测正确（正类负类）的占总的比重。

![](https://latex.codecogs.com/gif.latex?Accuracy=\frac{TP+TN}{TP+TN+FP+FN})

准确率是分类问题中最简单也是最直观的评价指标，但存在明显的缺陷。比 如，当负样本占99%时，分类器把所有样本都预测为负样本也可以获得99%的准确 率。所以，当不同类别的样本比例非常不均衡时，占比大的类别往往成为影响准 确率的最主要因素。

### 3.2 精确率（Precision）

**精确率（Precision）**，查准率。即正确预测为正的占全部预测为正的比例。个人理解：真正正确的占所有预测为正的比例。

![](https://latex.codecogs.com/gif.latex?Precision=\frac{TP}{TP+FP})

### 3.3 召回率(Recall)

**召回率（Recall）**，查全率。即正确预测为正的占全部实际为正的比例。个人理解：真正正确的占所有实际为正的比例。

![](https://latex.codecogs.com/gif.latex?Recall=\frac{TP}{TP+FN})

为了综合评估一个排序模型的好坏，不仅要看模型在不同 Top N下的Precision@N和Recall@N，而且最好绘制出模型的P-R（Precision- Recall）曲线。这里简单介绍一下P-R曲线的绘制方法。

P-R曲线的横轴是召回率，纵轴是精确率。对于一个排序模型来说，其P-R曲 线上的一个点代表着，在某一阈值下，模型将大于该阈值的结果判定为正样本， 小于该阈值的结果判定为负样本，此时返回结果对应的召回率和精确率。整条P-R 曲线是通过将阈值从高到低移动而生成的。下图是P-R曲线样例图，其中实线代表 模型A的P-R曲线，虚线代表模型B的P-R曲线。原点附近代表当阈值最大时模型的 精确率和召回率。

![](http://wx2.sinaimg.cn/mw690/00630Defly1g5e4ocvl8aj30ep0d275m.jpg)

由图可见，当召回率接近于0时，模型A的精确率为0.9，模型B的精确率是1， 这说明模型B得分前几位的样本全部是真正的正样本，而模型A即使得分最高的几 个样本也存在预测错误的情况。并且，随着召回率的增加，精确率整体呈下降趋 势。但是，当召回率为1时，模型A的精确率反而超过了模型B。**这充分说明，只用某个点对应的精确率和召回率是不能全面地衡量模型的性能，只有通过P-R曲线的 整体表现，才能够对模型进行更为全面的评估。**

### 3.4 F1值(H-mean值)

F1值（H-mean值）。F1值为算数平均数除以几何平均数，且越大越好，将Precision和Recall的上述公式带入会发现，当F1值小时，True Positive相对增加，而false相对减少，即Precision和Recall都相对增加，即F1对Precision和Recall都进行了加权。

![](https://latex.codecogs.com/gif.latex?\frac{2}{F_1}=\frac{1}{Precision}+\frac{1}{Recall})

![](https://latex.codecogs.com/gif.latex?F_1=\frac{2PR}{P+R}=\frac{2TP}{2TP+FP+FN})

### 3.4 ROC曲线

ROC曲线。接收者操作特征曲线（receiver operating characteristic curve），是反映敏感性和特异性连续变量的综合指标，ROC曲线上每个点反映着对同一信号刺激的感受性。下图是ROC曲线例子。

![](http://wx2.sinaimg.cn/mw690/00630Defly1g5e4fnjx1dj308w08waby.jpg)

横坐标：1-Specificity，伪正类率(False positive rate，FPR，FPR=FP/(FP+TN))，预测为正但实际为负的样本占所有负例样本的比例；

纵坐标：Sensitivity，真正类率(True positive rate，TPR，TPR=TP/(TP+FN))，预测为正且实际为正的样本占所有正例样本的比例。

**真正的理想情况**，TPR应接近1，FPR接近0，即图中的（0,1）点。**ROC曲线越靠拢（0,1）点，越偏离45度对角线越好**。

**AUC值**

AUC (Area Under Curve) 被定义为ROC曲线下的面积，显然这个面积的数值不会大于1。又由于ROC曲线一般都处于y=x这条直线的上方，所以AUC的取值范围一般在0.5和1之间。使用AUC值作为评价标准是因为很多时候ROC曲线并不能清晰的说明哪个分类器的效果更好，而作为一个数值，对应AUC更大的分类器效果更好。

从AUC判断分类器（预测模型）优劣的标准：

- AUC = 1，是完美分类器，采用这个预测模型时，存在至少一个阈值能得出完美预测。绝大多数预测的场合，不存在完美分类器。
- 0.5 < AUC < 1，优于随机猜测。这个分类器（模型）妥善设定阈值的话，能有预测价值。
- AUC = 0.5，跟随机猜测一样（例：丢铜板），模型没有预测价值。
- AUC < 0.5，比随机猜测还差；但只要总是反预测而行，就优于随机猜测。

**一句话来说，AUC值越大的分类器，正确率越高。**

### 3.5 余弦距离和欧式距离

**余弦距离：** ![](https://latex.codecogs.com/gif.latex?cos(A,B)=\frac{A*B}{||A||_2||B||_2})

**欧式距离：**在数学中，欧几里得距离或欧几里得度量是欧几里得空间中两点间“普通”（即直线）距离。

对于两个向量A和B，余弦距离关注的是向量之间的角度关系，并不关心它们的绝对大小，其取值 范围是[−1,1]。当一对文本相似度的长度差距很大、但内容相近时，如果使用词频 或词向量作为特征，它们在特征空间中的的欧氏距离通常很大；而如果使用余弦 相似度的话，它们之间的夹角可能很小，因而相似度高。此外，在文本、图像、 视频等领域，研究的对象的特征维度往往很高，余弦相似度在高维情况下依然保 持“相同时为1，正交时为0，相反时为−1”的性质，而欧氏距离的数值则受维度的 影响，范围不固定，并且含义也比较模糊。

### 3.6 A/B测试

AB测试是为Web或App界面或流程制作两个（A/B）或多个（A/B/n）版本，在同一时间维度，分别让组成成分相同（相似）的访客群组（目标人群）随机的访问这些版本，收集各群组的用户体验数据和业务数据，最后分析、评估出最好版本，正式采用。

### 3.7 模型评估方法

1. **Holdout检验**

   Holdout 检验是最简单也是最直接的验证方法，它将原始的样本集合随机划分 成训练集和验证集两部分。比方说，对于一个点击率预测模型，我们把样本按照 70%～30% 的比例分成两部分，70% 的样本用于模型训练；30% 的样本用于模型 验证，包括绘制ROC曲线、计算精确率和召回率等指标来评估模型性能。

   Holdout 检验的缺点很明显，即在验证集上计算出来的最后评估指标与原始分 组有很大关系。为了消除随机性，研究者们引入了“交叉检验”的思想。

2. **交叉检验**

   k-fold交叉验证：首先将全部样本划分成k个大小相等的样本子集；依次遍历 这k个子集，每次把当前子集作为验证集，其余所有子集作为训练集，进行模型的 训练和评估；最后把k次评估指标的平均值作为最终的评估指标。在实际实验 中，k经常取10。

3. **自助法**

   不管是Holdout检验还是交叉检验，都是基于划分训练集和测试集的方法进行 模型评估的。然而，当样本规模比较小时，将样本集进行划分会让训练集进一步 减小，这可能会影响模型训练效果。有没有能维持训练集样本规模的验证方法 呢？自助法可以比较好地解决这个问题。 

   自助法是基于自助采样法的检验方法。对于总数为n的样本集合，进行n次有 放回的随机抽样，得到大小为n的训练集。n次采样过程中，有的样本会被重复采 样，有的样本没有被抽出过，将这些没有被抽出的样本作为验证集，进行模型验 证，这就是自助法的验证过程。

### 3.8 超参数调优

为了进行超参数调优，我们一般会采用网格搜索、随机搜索、贝叶斯优化等 算法。在具体介绍算法之前，需要明确超参数搜索算法一般包括哪几个要素。一 是目标函数，即算法需要最大化/最小化的目标；二是搜索范围，一般通过上限和 下限来确定；三是算法的其他参数，如搜索步长。

- **网格搜索**，可能是最简单、应用最广泛的超参数搜索算法，它通过查找搜索范 围内的所有的点来确定最优值。如果采用较大的搜索范围以及较小的步长，网格 搜索有很大概率找到全局最优值。然而，**这种搜索方案十分消耗计算资源和时间**，特别是需要调优的超参数比较多的时候。因此，在实际应用中，网格搜索法一般会先使用较广的搜索范围和较大的步长，来寻找全局最优值可能的位置；然 后会逐渐缩小搜索范围和步长，来寻找更精确的最优值。这种操作方案可以降低 所需的时间和计算量，但由于目标函数一般是非凸的，所以很可能会错过全局最 优值。
- **随机搜索**，随机搜索的思想与网格搜索比较相似，只是不再测试上界和下界之间的所有 值，而是在搜索范围中随机选取样本点。它的理论依据是，如果样本点集足够 大，那么通过随机采样也能大概率地找到全局最优值，或其近似值。随机搜索一 般会比网格搜索要快一些，但是和网格搜索的快速版一样，它的结果也是没法保证的。
- **贝叶斯优化算法**，贝叶斯优化算法在寻找最优最值参数时，采用了与网格搜索、随机搜索完全 不同的方法。网格搜索和随机搜索在测试一个新点时，会忽略前一个点的信息； 而贝叶斯优化算法则充分利用了之前的信息。贝叶斯优化算法通过对目标函数形 状进行学习，找到使目标函数向全局最优值提升的参数。

### 3.9 过拟合和欠拟合

过拟合是指模型对于训练数据拟合呈过当的情况，反映到评估指标上，就是 模型在训练集上的表现很好，但在测试集和新数据上的表现较差。欠拟合指的是 模型在训练和预测时表现都不好的情况。下图形象地描述了过拟合和欠拟合的区别。

![](http://wx1.sinaimg.cn/mw690/00630Defly1g5e5937nuej30hw05i3zl.jpg)

1. **防止过拟合：**
   - 从数据入手，获得更多的训练数据。
   - 降低模型复杂度。
   - 正则化方法，给模型的参数加上一定的正则约束。
   - 集成学习方法，集成学习是把多个模型集成在一起。
2. **防止欠拟合：**
   - 添加新特征。
   - 增加模型复杂度。
   - 减小正则化系数。

## 4. 检验方法

### 4.1 KS检验

Kolmogorov-Smirnov检验是基于累计分布函数的，用于检验一个分布是否符合某种理论分布或比较两个经验分布是否有显著差异。

- 单样本K-S检验是用来检验一个数据的观测经验分布是否符合已知的理论分布。
- 两样本K-S检验由于对两样本的经验分布函数的位置和形状参数的差异都敏感，所以成为比较两样本的最有用且最常用的非参数方法之一。

检验统计量为：![](https://latex.codecogs.com/gif.latex?D_r=max_x|F_n(x)-F(x)|)

其中  ![](https://latex.codecogs.com/gif.latex?F_n(x))为观察序列值，![](https://latex.codecogs.com/gif.latex?F(x))为理论序列值或另一观察序列值。

### 4.2 T检验

T检验，也称student t检验，主要用户样本含量较小，总体标准差未知的正态分布。

t检验是用t分布理论来推论差异发生的概率，从而比较两个平均数的差异是否显著。

t检验分为单总体检验和双总体检验。

### 4.3 F检验

T检验和F检验的由来：为了确定从样本中的统计结果推论到总体时所犯错的概率。F检验又叫做联合假设检验，也称方差比率检验、方差齐性检验。是由英国统计学家Fisher提出。通过比较两组数据的方差，以确定他们的精密度是否有显著性差异。

### 4.4 Grubbs检验

一组测量数据中，如果个别数据偏离平均值很远，那么称这个数据为“可疑值”。用格拉布斯法判断，能将“可疑值”从测量数据中剔除。

### 4.5 卡方检验

卡方检验就是统计样本的实际观测值与理论推断值之间的偏离程度，实际观测值与理论推断值之间的偏离程度就决定卡方值的大小，卡方值越大，越不符合；卡方值越小，偏差越小，越趋于符合，若两个值完全相等时，卡方值就为0，表明理论值完全符合。

1. 提出原假设H0：总体X的分布函数F(x)；

2. 将总体x的取值范围分成k个互不相交的小区间A1-Ak；

3. 把落入第i个区间Ai的样本的个数记做fi，成为组频数，f1+f2+f3+...+fk = n；

4. 当H0为真时，根据假设的总体理论分布，可算出总体X的值落入第i个小区间Ai的概率pi，于是n*pi就是落入第i个小区间Ai的样本值的理论频数；

5. 当H0为真时，n次试验中样本落入第i个小区间Ai的频率fi/n与概率pi应该很接近。基于这种思想，皮尔逊引入检测统计量：

   ![](https://latex.codecogs.com/gif.latex?x^2=\sum_{i=1}^{k}\frac{(f_i-np_i)^2}{np_i})

   在H0假设成立的情况下服从自由度为k-1的卡方分布。

**KS检验与卡方检验**

**相同点：**都采用实际频数和期望频数只差进行检验

**不同点：**

- 卡方检验主要用于类别数据，而KS检验主要用于有计量单位的连续和定量数据。
- 卡方检验也可以用于定量数据，但必须先将数据分组才能获得实际的观测频数，而KS检验能直接对原始数据进行检验，所以它对数据的利用比较完整。

## 5. 参考文献

[百面机器学习](https://www.lanzous.com/i56i24f)

> 作者：[@mantchs](https://github.com/NLP-LOVE/ML-NLP)
>
> GitHub：[https://github.com/NLP-LOVE/ML-NLP](https://github.com/NLP-LOVE/ML-NLP)
>
> 欢迎大家加入讨论！共同完善此项目！群号：【541954936】<a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=863f915b9178560bd32ca07cd090a7d9e6f5f90fcff5667489697b1621cecdb3"><img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="NLP面试学习群" title="NLP面试学习群"></a>

---

# 9. KNN

## 目录
- [1. 什么是KNN](#1-什么是knn)
  - [1.1 KNN的通俗解释](#11-knn的通俗解释)
  - [1.2 近邻的距离度量](#12-近邻的距离度量)
  - [1.3 K值选择](#13-k值选择)
  - [1.4 KNN最近邻分类算法的过程](#14-knn最近邻分类算法的过程)
- [2. KDD的实现：KD树](#2-kdd的实现kd树)
  - [2.1 构建KD树](#21-构建kd树)
  - [2.2 KD树的插入](#22-kd树的插入)
  - [2.3 KD树的删除](#23-kd树的删除)
  - [2.4 KD树的最近邻搜索算法](#24-kd树的最近邻搜索算法)
  - [2.5 kd树近邻搜索算法的改进：BBF算法](#25-kd树近邻搜索算法的改进bbf算法)
  - [2.6 KD树的应用](#26-kd树的应用)
- [3. 关于KNN的一些问题](#3-关于knn的一些问题)
- [4. 参考文献](#4-参考文献)
- [5. 手写数字识别案例](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/9.%20KNN/handwritingClass)

## 1. 什么是KNN

### 1.1 KNN的通俗解释

何谓K近邻算法，即K-Nearest Neighbor algorithm，简称KNN算法，单从名字来猜想，可以简单粗暴的认为是：K个最近的邻居，当K=1时，算法便成了最近邻算法，即寻找最近的那个邻居。

用官方的话来说，所谓K近邻算法，即是给定一个训练数据集，对新的输入实例，**在训练数据集中找到与该实例最邻近的K个实例（也就是上面所说的K个邻居），这K个实例的多数属于某个类，就把该输入实例分类到这个类中。**

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155283963472895660.jpg)

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155283966654403646.png)

如上图所示，有两类不同的样本数据，分别用蓝色的小正方形和红色的小三角形表示，而图正中间的那个绿色的圆所标示的数据则是待分类的数据。也就是说，现在，我们不知道中间那个绿色的数据是从属于哪一类（蓝色小正方形or红色小三角形），KNN就是解决这个问题的。

如果K=3，绿色圆点的最近的3个邻居是2个红色小三角形和1个蓝色小正方形，少数从属于多数，基于统计的方法，判定绿色的这个待分类点属于红色的三角形一类。

如果K=5，绿色圆点的最近的5个邻居是2个红色三角形和3个蓝色的正方形，还是少数从属于多数，基于统计的方法，判定绿色的这个待分类点属于蓝色的正方形一类。

**于此我们看到，当无法判定当前待分类点是从属于已知分类中的哪一类时，我们可以依据统计学的理论看它所处的位置特征，衡量它周围邻居的权重，而把它归为(或分配)到权重更大的那一类。这就是K近邻算法的核心思想。**

### 1.2 近邻的距离度量

我们看到，K近邻算法的核心在于找到实例点的邻居，这个时候，问题就接踵而至了，如何找到邻居，邻居的判定标准是什么，用什么来度量。这一系列问题便是下面要讲的距离度量表示法。

**有哪些距离度量的表示法**(普及知识点，可以跳过)：

1. **欧氏距离**，最常见的两点之间或多点之间的距离表示法，又称之为欧几里得度量，它定义于欧几里得空间中，如点 x = (x1,...,xn) 和 y = (y1,...,yn) 之间的距离为：

   ![](https://latex.codecogs.com/gif.latex?d(x,y)=\sqrt{(x_1-y_1)^2+(x_2-y_2)^2+...+(x_n-y_n)^2}=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2})

   - 二维平面上两点a(x1,y1)与b(x2,y2)间的欧氏距离：

     ![](https://latex.codecogs.com/gif.latex?d_{12}=\sqrt{(x_1-x_2)^2+(y_1-y_2)^2})

   - 三维空间两点a(x1,y1,z1)与b(x2,y2,z2)间的欧氏距离：

     ![](https://latex.codecogs.com/gif.latex?d_{12}=\sqrt{(x_1-x_2)^2+(y_1-y_2)^2+(z_1-z_2)^2})

   - 两个n维向量a(x11,x12,…,x1n)与 b(x21,x22,…,x2n)间的欧氏距离：

     ![](https://latex.codecogs.com/gif.latex?d_{12}=\sqrt{\sum_{k=1}^{n}(x_{1k}-x_{2k})^2})

     也可以用表示成向量运算的形式：

     ![](https://latex.codecogs.com/gif.latex?d_{12}=\sqrt{(a-b)(a-b)^T})

2. **曼哈顿距离**，我们可以定义曼哈顿距离的正式意义为L1-距离或城市区块距离，也就是在欧几里得空间的固定直角坐标系上两点所形成的线段对轴产生的投影的距离总和。例如在平面上，坐标（x1, y1）的点P1与坐标（x2, y2）的点P2的曼哈顿距离为： ![](https://latex.codecogs.com/gif.latex?|x_1-x_2|+|y_1-y_2|)，要注意的是，曼哈顿距离依赖座标系统的转度，而非系统在座标轴上的平移或映射。 

   通俗来讲，想象你在曼哈顿要从一个十字路口开车到另外一个十字路口，驾驶距离是两点间的直线距离吗？显然不是，除非你能穿越大楼。而实际驾驶距离就是这个“曼哈顿距离”，此即曼哈顿距离名称的来源， 同时，曼哈顿距离也称为城市街区距离(City Block distance)。

   - 二维平面两点a(x1,y1)与b(x2,y2)间的曼哈顿距离 

     ![](https://latex.codecogs.com/gif.latex?d_{12}=|x_1-x_2|+|y_1-y_2|)

   - 两个n维向量a(x11,x12,…,x1n)与 b(x21,x22,…,x2n)间的曼哈顿距离

     ![](https://latex.codecogs.com/gif.latex?d_{12}=\sum_{k=1}^{n}|x_{1k}-x_{2k}|)

3. **切比雪夫距离**，若二个向量或二个点p 、and q，其座标分别为Pi及qi，则两者之间的切比雪夫距离定义如下：

   ![](https://latex.codecogs.com/gif.latex?D_{Chebyshev}(p,q)=max_i(|p_i-q_i|))

   这也等于以下Lp度量的极值： ![](https://gitee.com/kkweishe/images/raw/master/ML/2019-9-24_22-19-41.png)，因此切比雪夫距离也称为L∞度量。

   以数学的观点来看，切比雪夫距离是由一致范数（uniform norm）（或称为上确界范数）所衍生的度量，也是超凸度量（injective metric space）的一种。

   在平面几何中，若二点p及q的直角坐标系坐标为(x1,y1)及(x2,y2)，则切比雪夫距离为：

   ![](https://latex.codecogs.com/gif.latex?D_{Chess}=max(|x_2-x_1|,|y_2-y_1|))

   **玩过国际象棋的朋友或许知道，国王走一步能够移动到相邻的8个方格中的任意一个。那么国王从格子(x1,y1)走到格子(x2,y2)最少需要多少步？。你会发现最少步数总是max( | x2-x1 | , | y2-y1 | ) 步 。有一种类似的一种距离度量方法叫切比雪夫距离。**

   - 二维平面两点a(x1,y1)与b(x2,y2)间的切比雪夫距离 ：

     ![](https://latex.codecogs.com/gif.latex?d_{12}=max(|x_2-x_1|,|y_2-y_1|))

   - 两个n维向量a(x11,x12,…,x1n)与 b(x21,x22,…,x2n)间的切比雪夫距离：

     ![](https://latex.codecogs.com/gif.latex?d_{12}=max_i(|x_{1i}-x_{2i}|))

     这个公式的另一种等价形式是

     ![](https://latex.codecogs.com/gif.latex?d_{12}=lim_{k\to\infin}(\sum_{i=1}^{n}|x_{1i}-x_{2i}|^k)^{1/k})

4. **闵可夫斯基距离**(Minkowski Distance)，闵氏距离不是一种距离，而是一组距离的定义。

   两个n维变量a(x11,x12,…,x1n)与 b(x21,x22,…,x2n)间的闵可夫斯基距离定义为： 

   ![](https://latex.codecogs.com/gif.latex?d_{12}=\sqrt[p]{\sum_{k=1}^{n}|x_{1k}-x_{2k}|^p})

   其中p是一个变参数。
   当p=1时，就是曼哈顿距离
   当p=2时，就是欧氏距离
   当p→∞时，就是切比雪夫距离       
   根据变参数的不同，闵氏距离可以表示一类的距离。 

5. **标准化欧氏距离**，标准化欧氏距离是针对简单欧氏距离的缺点而作的一种改进方案。标准欧氏距离的思路：既然数据各维分量的分布不一样，那先将各个分量都“标准化”到均值、方差相等。至于均值和方差标准化到多少，先复习点统计学知识。

   假设样本集X的数学期望或均值(mean)为m，标准差(standard deviation，方差开根)为s，那么X的“标准化变量”X*表示为：(X-m）/s，而且标准化变量的数学期望为0，方差为1。
   即，样本集的标准化过程(standardization)用公式描述就是：

   ![](https://latex.codecogs.com/gif.latex?X^*=\frac{X-m}{s})

   标准化后的值 =  ( 标准化前的值  － 分量的均值 ) /分量的标准差　　
   经过简单的推导就可以得到两个n维向量a(x11,x12,…,x1n)与 b(x21,x22,…,x2n)间的标准化欧氏距离的公式：

   ![](https://latex.codecogs.com/gif.latex?d_{12}=\sqrt{\sum_{k=1}^{n}(\frac{x_{1k}-x_{2k}}{s_k})^2})

6. **马氏距离**

   有M个样本向量X1~Xm，协方差矩阵记为S，均值记为向量μ，则其中样本向量X到u的马氏距离表示为：

   ![](https://latex.codecogs.com/gif.latex?D(X)=\sqrt{(X-u)^TS^{-1}(X_i-X_j)})

   - 若协方差矩阵是单位矩阵（各个样本向量之间独立同分布）,则公式就成了,也就是欧氏距离了：

     ![](https://latex.codecogs.com/gif.latex?D(X_i,X_j)=\sqrt{(X_i-X_j)^T(X_i-X_j)})

   - 若协方差矩阵是对角矩阵，公式变成了标准化欧氏距离。

   马氏距离的优缺点：量纲无关，排除变量之间的相关性的干扰。

7. **巴氏距离**

   在统计中，巴氏距离距离测量两个离散或连续概率分布的相似性。它与衡量两个统计样品或种群之间的重叠量的巴氏距离系数密切相关。巴氏距离距离和巴氏距离系数以20世纪30年代曾在印度统计研究所工作的一个统计学家A. Bhattacharya命名。同时，Bhattacharyya系数可以被用来确定两个样本被认为相对接近的，它是用来测量中的类分类的可分离性。

   对于离散概率分布 p和q在同一域 X，它被定义为：

   ![](https://latex.codecogs.com/gif.latex?D_B(p,q)=-ln(BC(p,q)))

   其中：

   ![](https://latex.codecogs.com/gif.latex?BC(p,q)=\sum_{x\in_{}X}\sqrt{p(x)q(x)})

   是Bhattacharyya系数。

8. **汉明距离**

   两个等长字符串s1与s2之间的汉明距离定义为将其中一个变为另外一个所需要作的最小替换次数。例如字符串“1111”与“1001”之间的汉明距离为2。应用：信息编码（为了增强容错性，应使得编码间的最小汉明距离尽可能大）。

9. **夹角余弦**

   几何中夹角余弦可用来衡量两个向量方向的差异，机器学习中借用这一概念来衡量样本向量之间的差异。

   - 在二维空间中向量A(x1,y1)与向量B(x2,y2)的夹角余弦公式：

     ![](https://latex.codecogs.com/gif.latex?cos\theta=\frac{x_1x_2+y_1y_2}{\sqrt{x_1^2+y_1^2}\sqrt{x_2^2+y_2^2}})

   - 两个n维样本点a(x11,x12,…,x1n)和b(x21,x22,…,x2n)的夹角余弦：

     ![](https://latex.codecogs.com/gif.latex?cos\theta=\frac{a*b}{|a||b|})

   夹角余弦取值范围为[-1,1]。夹角余弦越大表示两个向量的夹角越小，夹角余弦越小表示两向量的夹角越大。当两个向量的方向重合时夹角余弦取最大值1，当两个向量的方向完全相反夹角余弦取最小值-1。 

10. **杰卡德相似系数**

    两个集合A和B的交集元素在A，B的并集中所占的比例，称为两个集合的杰卡德相似系数，用符号J(A,B)表示。杰卡德相似系数是衡量两个集合的相似度一种指标。

    ![](https://latex.codecogs.com/gif.latex?J(A,B)=\frac{|A\cap_{}B|}{|A\cup_{}B|})

    与杰卡德相似系数相反的概念是杰卡德距离：

    ![](https://latex.codecogs.com/gif.latex?J_{\delta}(A,B)=1-J(A,B)=\frac{|A\cup_{}B|-|A\cap_{}B|}{|A\cup_{}B|})

11. **皮尔逊系数**

    在统计学中，皮尔逊积矩相关系数用于度量两个变量X和Y之间的相关（线性相关），其值介于-1与1之间。通常情况下通过以下取值范围判断变量的相关强度：

    0.8-1.0     极强相关
    0.6-0.8     强相关
    0.4-0.6     中等程度相关
    0.2-0.4     弱相关
    0.0-0.2     极弱相关或无相关

简单说来，各种“距离”的应用场景简单概括为，

- 空间：欧氏距离，
- 路径：曼哈顿距离，国际象棋国王：切比雪夫距离，
- 以上三种的统一形式:闵可夫斯基距离，
- 加权：标准化欧氏距离，
- 排除量纲和依存：马氏距离，
- 向量差距：夹角余弦，
- 编码差别：汉明距离，
- 集合近似度：杰卡德类似系数与距离，
- 相关：相关系数与相关距离。

### 1.3 K值选择

1. 如果选择较小的K值，就相当于用较小的领域中的训练实例进行预测，“学习”近似误差会减小，只有与输入实例较近或相似的训练实例才会对预测结果起作用，与此同时带来的问题是“学习”的估计误差会增大，换句话说，**K值的减小就意味着整体模型变得复杂，容易发生过拟合；**
2. 如果选择较大的K值，就相当于用较大领域中的训练实例进行预测，其优点是可以减少学习的估计误差，但缺点是学习的近似误差会增大。这时候，与输入实例较远（不相似的）训练实例也会对预测器作用，使预测发生错误，且**K值的增大就意味着整体的模型变得简单。**
3. K=N，则完全不足取，因为此时无论输入实例是什么，都只是简单的预测它属于在训练实例中最多的累，模型过于简单，忽略了训练实例中大量有用信息。

在实际应用中，K值一般取一个比较小的数值，**例如采用交叉验证法（简单来说，就是一部分样本做训练集，一部分做测试集）来选择最优的K值。**

### 1.4 KNN最近邻分类算法的过程

1. 计算测试样本和训练样本中每个样本点的距离（常见的距离度量有欧式距离，马氏距离等）；
2. 对上面所有的距离值进行排序；
3. 选前 k 个最小距离的样本；
4. 根据这 k 个样本的标签进行投票，得到最后的分类类别；



## 2. KDD的实现：KD树

Kd-树是K-dimension tree的缩写，是对数据点在k维空间（如二维(x，y)，三维(x，y，z)，k维(x1，y，z..)）中划分的一种数据结构，主要应用于多维空间关键数据的搜索（如：范围搜索和最近邻搜索）。本质上说，Kd-树就是一种平衡二叉树。

首先必须搞清楚的是，k-d树是一种空间划分树，说白了，就是把整个空间划分为特定的几个部分，然后在特定空间的部分内进行相关搜索操作。想像一个三维(多维有点为难你的想象力了)空间，kd树按照一定的划分规则把这个三维空间划分了多个空间，如下图所示：

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155344275027763857.png)

### 2.1 构建KD树

kd树构建的伪代码如下图所示：

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155344266926262911.jpg)

再举一个简单直观的实例来介绍k-d树构建算法。假设有6个二维数据点{(2,3)，(5,4)，(9,6)，(4,7)，(8,1)，(7,2)}，数据点位于二维空间内，如下图所示。为了能有效的找到最近邻，k-d树采用分而治之的思想，即将整个空间划分为几个小部分，首先，粗黑线将空间一分为二，然后在两个子空间中，细黑直线又将整个空间划分为四部分，最后虚黑直线将这四部分进一步划分。

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155368320742672975.png)

6个二维数据点{(2,3)，(5,4)，(9,6)，(4,7)，(8,1)，(7,2)}构建kd树的具体步骤为：

1. 确定：split域=x。具体是：6个数据点在x，y维度上的数据方差分别为39，28.63，所以在x轴上方差更大，故split域值为x；
2. 确定：Node-data = （7,2）。具体是：根据x维上的值将数据排序，6个数据的中值(所谓中值，即中间大小的值)为7，所以Node-data域位数据点（7,2）。这样，该节点的分割超平面就是通过（7,2）并垂直于：split=x轴的直线x=7；
3. 确定：左子空间和右子空间。具体是：分割超平面x=7将整个空间分为两部分：x<=7的部分为左子空间，包含3个节点={(2,3),(5,4),(4,7)}；另一部分为右子空间，包含2个节点={(9,6)，(8,1)}；
4. 如上算法所述，kd树的构建是一个递归过程，我们对左子空间和右子空间内的数据重复根节点的过程就可以得到一级子节点（5,4）和（9,6），同时将空间和数据集进一步细分，如此往复直到空间中只包含一个数据点。

与此同时，经过对上面所示的空间划分之后，我们可以看出，点(7,2)可以为根结点，从根结点出发的两条红粗斜线指向的(5,4)和(9,6)则为根结点的左右子结点，而(2,3)，(4,7)则为(5,4)的左右孩子(通过两条细红斜线相连)，最后，(8,1)为(9,6)的左孩子(通过细红斜线相连)。如此，便形成了下面这样一棵k-d树：

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155368335593855374.png)

**对于n个实例的k维数据来说，建立kd-tree的时间复杂度为O(k*n*logn)。**

k-d树算法可以分为两大部分，除了上部分有关k-d树本身这种数据结构建立的算法，另一部分是在建立的k-d树上各种诸如插入，删除，查找(最邻近查找)等操作涉及的算法。下面，咱们依次来看kd树的插入、删除、查找操作。

### 2.2 KD树的插入

元素插入到一个K-D树的方法和二叉检索树类似。本质上，在偶数层比较x坐标值，而在奇数层比较y坐标值。当我们到达了树的底部，（也就是当一个空指针出现），我们也就找到了结点将要插入的位置。生成的K-D树的形状依赖于结点插入时的顺序。给定N个点，其中一个结点插入和检索的平均代价是O(log2N)。

插入的过程如下:

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155377895462114526.png)

应该清楚，这里描述的插入过程中，每个结点将其所在的平面分割成两部分。因比，Chicago 将平面上所有结点分成两部分，一部分所有的结点x坐标值小于35，另一部分结点的x坐标值大于或等于35。同样Mobile将所有x坐标值大于35的结点以分成两部分，一部分结点的Y坐标值是小于10，另一部分结点的Y坐标值大于或等于10。后面的Toronto、Buffalo也按照一分为二的规则继续划分。

### 2.3 KD树的删除

KD树的删除可以用递归程序来实现。我们假设希望从K-D树中删除结点（a,b）。如果（a,b）的两个子树都为空，则用空树来代替（a,b）。否则，在（a,b）的子树中寻找一个合适的结点来代替它，譬如(c,d)，则递归地从K-D树中删除（c,d）。一旦(c,d)已经被删除，则用（c,d）代替（a,b）。假设(a,b)是一个X识别器，那么，它得替代节点要么是（a,b）左子树中的X坐标最大值的结点，要么是（a,b）右子树中x坐标最小值的结点。

下面来举一个实际的例子(来源：中国地质大学电子课件，原课件错误已经在下文中订正)，如下图所示，原始图像及对应的kd树，现在要删除图中的A结点，请看一系列删除步骤：

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155377904784496247.jpg)

要删除上图中结点A，选择结点A的右子树中X坐标值最小的结点，这里是C，C成为根，如下图：

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155377912448672600.jpg)

从C的右子树中找出一个结点代替先前C的位置，

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155377914973991477.jpg)

这里是D，并将D的左子树转为它的右子树，D代替先前C的位置，如下图：

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155377917753460670.jpg)

在D的新右子树中，找X坐标最小的结点，这里为H，H代替D的位置，

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155377920028856740.jpg)

在D的右子树中找到一个Y坐标最小的值，这里是I，将I代替原先H的位置，从而A结点从图中顺利删除，如下图所示：

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase6415537792238505282.jpg)

从K-D树中删除一个结点是代价很高的，很清楚删除子树的根受到子树中结点个数的限制。用TPL（T）表示树T总的路径长度。可看出树中子树大小的总和为TPL（T）+N。 以随机方式插入N个点形成树的TPL是O(N*log2N),这就意味着从一个随机形成的K-D树中删除一个随机选取的结点平均代价的上界是O(log2N) 。

### 2.4 KD树的最近邻搜索算法

k-d树查询算法的伪代码如下所示：

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155377943526545714.png)

我写了一个递归版本的二维kd tree的搜索函数你对比的看看：

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase6415537794775411079.png)

**举例**

星号表示要查询的点查询点（2，4.5）。通过二叉搜索，顺着搜索路径很快就能找到最邻近的近似点。而找到的叶子节点并不一定就是最邻近的，最邻近肯定距离查询点更近，应该位于以查询点为圆心且通过叶子节点的圆域内。为了找到真正的最近邻，还需要进行相关的‘回溯'操作。也就是说，算法首先沿搜索路径反向查找是否有距离查询点更近的数据点。

1. **二叉树搜索**：先从（7,2）查找到（5,4）节点，在进行查找时是由y = 4为分割超平面的，由于查找点为y值为4.5，因此进入右子空间查找到（4,7），形成搜索路径<(7,2)，(5,4)，(4,7)>，但（4,7）与目标查找点的距离为3.202，而（5,4）与查找点之间的距离为3.041，所以（5,4）为查询点的最近点；
2. **回溯查找**：以（2，4.5）为圆心，以3.041为半径作圆，如下图所示。可见该圆和y = 4超平面交割，所以需要进入（5,4）左子空间进行查找，也就是将（2,3）节点加入搜索路径中得<(7,2)，(2,3)>；于是接着搜索至（2,3）叶子节点，（2,3）距离（2,4.5）比（5,4）要近，所以最近邻点更新为（2，3），最近距离更新为1.5；
3. 回溯查找至（5,4），直到最后回溯到根结点（7,2）的时候，以（2,4.5）为圆心1.5为半径作圆，并不和x = 7分割超平面交割，如下图所示。至此，搜索路径回溯完，返回最近邻点（2,3），最近距离1.5。

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155377998951412539.jpg)

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155377966743482553.png)

### 2.5 kd树近邻搜索算法的改进：BBF算法

实例点是随机分布的，那么kd树搜索的平均计算复杂度是O（logN），这里的N是训练实例树。所以说，kd树更适用于训练实例数远大于空间维数时的k近邻搜索，当空间维数接近训练实例数时，**它的效率会迅速下降，一降降到“解放前”：线性扫描的速度。**

也正因为上述k最近邻搜索算法的第4个步骤中的所述：“回退到根结点时，搜索结束”，每个最近邻点的查询比较完成过程最终都要回退到根结点而结束，而导致了许多不必要回溯访问和比较到的结点，这些多余的损耗在高维度数据查找的时候，搜索效率将变得相当之地下，那有什么办法可以改进这个原始的kd树最近邻搜索算法呢？

从上述标准的kd树查询过程可以看出其搜索过程中的“回溯”是由“查询路径”决定的，并没有考虑查询路径上一些数据点本身的一些性质。一个简单的改进思路就是将“查询路径”上的结点进行排序，如按各自分割超平面（也称bin）与查询点的距离排序，**也就是说，回溯检查总是从优先级最高（Best Bin）的树结点开始。**

还是以上面的查询（2,4.5）为例，搜索的算法流程为：

1. 将（7,2）压人优先队列中；
2. 提取优先队列中的（7,2），由于（2,4.5）位于（7,2）分割超平面的左侧，所以检索其左子结点（5,4）。
3. 同时，根据BBF机制”搜索左/右子树，就把对应这一层的兄弟结点即右/左结点存进队列”，将其（5,4）对应的兄弟结点即右子结点（9,6）压人优先队列中
4. 此时优先队列为{（9,6）}，最佳点为（7,2）；然后一直检索到叶子结点（4,7），此时优先队列为{（2,3），（9,6）}，“最佳点”则为（5,4）；
5. 提取优先级最高的结点（2,3），重复步骤2，直到优先队列为空。

![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155378000183167311.jpg)

### 2.6 KD树的应用

SIFT+KD_BBF搜索算法，详细参考文末的参考文献。



## 3. 关于KNN的一些问题

1. 在k-means或kNN，我们是用欧氏距离来计算最近的邻居之间的距离。为什么不用曼哈顿距离？

   **答：**我们不用曼哈顿距离，因为它只计算水平或垂直距离，有维度的限制。另一方面，欧式距离可用于任何空间的距离计算问题。因为，数据点可以存在于任何空间，欧氏距离是更可行的选择。例如：想象一下国际象棋棋盘，象或车所做的移动是由曼哈顿距离计算的，因为它们是在各自的水平和垂直方向的运动。

2. KD-Tree相比KNN来进行快速图像特征比对的好处在哪里?

   答：极大的节约了时间成本．点线距离如果 >　最小点，无需回溯上一层，如果<,则再上一层寻找。



## 4. 参考文献

[从K近邻算法、距离度量谈到KD树、SIFT+BBF算法](http://blog.csdn.net/v_july_v/article/details/8203674)

## 5. 手写数字识别案例

[KNN手写数字识别系统](https://github.com/NLP-LOVE/ML-NLP/tree/master/Machine%20Learning/9.%20KNN/handwritingClass)

----

> 作者：[@mantchs](https://github.com/NLP-LOVE/ML-NLP)
>
> GitHub：[https://github.com/NLP-LOVE/ML-NLP](https://github.com/NLP-LOVE/ML-NLP)
>
> 欢迎大家加入讨论！共同完善此项目！群号：【541954936】<a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=863f915b9178560bd32ca07cd090a7d9e6f5f90fcff5667489697b1621cecdb3"><img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="NLP面试学习群" title="NLP面试学习群"></a>

---

### handwritingClass

### 项目案例: 手写数字识别系统

[完整代码地址](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/9.%20KNN/handwritingClass/handwritingClass.py)

#### 项目概述

构造一个能识别数字 0 到 9 的基于 KNN 分类器的手写数字识别系统。

需要识别的数字是存储在文本文件中的具有相同的色彩和大小：宽高是 32 像素 * 32 像素的黑白图像。

#### 开发流程

```
收集数据：提供文本文件。
准备数据：编写函数 img2vector(), 将图像格式转换为分类器使用的向量格式
分析数据：在 Python 命令提示符中检查数据，确保它符合要求
训练算法：此步骤不适用于 KNN
测试算法：编写函数使用提供的部分数据集作为测试样本，测试样本与非测试样本的
         区别在于测试样本是已经完成分类的数据，如果预测分类与实际类别不同，
         则标记为一个错误
使用算法：本例没有完成此步骤，若你感兴趣可以构建完整的应用程序，从图像中提取
         数字，并完成数字识别，美国的邮件分拣系统就是一个实际运行的类似系统
```

> 收集数据: 提供文本文件

压缩包 [trainingDigits.zip](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/9.%20KNN/handwritingClass/trainingDigits.zip) 中包含了大约 2000 个例子，每个例子内容如下图所示，每个数字大约有 200 个样本；目录 [testDigits](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/9.%20KNN/handwritingClass/testDigits.zip) 中包含了大约 900 个测试数据。**下载后解压**。

![手写数字数据集的例子](http://wx4.sinaimg.cn/mw690/00630Defgy1g5l35mlzy7j30jb0aj7gx.jpg)

> 准备数据: 编写函数 img2vector(), 将图像文本数据转换为分类器使用的向量

将图像文本数据转换为向量

```python
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect
```

> 分析数据：在 Python 命令提示符中检查数据，确保它符合要求

在 Python 命令行中输入下列命令测试 img2vector 函数，然后与文本编辑器打开的文件进行比较: 

```python
>>> testVector = kNN.img2vector('testDigits/0_13.txt')
>>> testVector[0,0:32]
array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
>>> testVector[0,32:64]
array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
```

> 训练算法：此步骤不适用于 KNN

因为测试数据每一次都要与全量的训练数据进行比较，所以这个过程是没有必要的。

> 测试算法：编写函数使用提供的部分数据集作为测试样本，如果预测分类与实际类别不同，则标记为一个错误

```python
def handwritingClassTest():
    # 1. 导入训练数据
    hwLabels = []
    trainingFileList = listdir('data/2.KNN/trainingDigits')  # load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    # hwLabels存储0～9对应的index位置， trainingMat存放的每个位置对应的图片向量
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        # 将 32*32的矩阵->1*1024的矩阵
        trainingMat[i, :] = img2vector('data/2.KNN/trainingDigits/%s' % fileNameStr)

    # 2. 导入测试数据
    testFileList = listdir('data/2.KNN/testDigits')  # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('data/2.KNN/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount / float(mTest))
```

> 使用算法：本例没有完成此步骤，若你感兴趣可以构建完整的应用程序，从图像中提取数字，并完成数字识别，美国的邮件分拣系统就是一个实际运行的类似系统。

> 本项目转载自：[apachecn/AiLearning](https://github.com/apachecn/AiLearning/blob/master/docs/ml/2.k-近邻算法.md)

---
