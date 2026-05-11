# Agoda Data Scientist / Senior / Staff Data Scientist 面试题整理

> 来源：Glassdoor、Interview Query、Prepfully、HackerRank 岗位页等公开信息。  
> 标注为“已出现”的题目来自公开面经题面；标注为“题型线索”的题目只有方向描述，下面给出可准备的标准版本。

---

## 1. 五格菱形平台随机游走概率（已出现，Senior/Data Scientist）

### 题目

平台由 5 个格子组成，坐标为：

```text
(-1,0), (0,-1), (0,0), (0,1), (1,0)
```

从起点 `(xs, ys)` 出发，每一步等概率向上下左右移动。若走出平台则失败。求在失败前到达终点 `(xe, ye)` 的概率。

示例：

```text
input:  -1 0 0 0
output: 0.25
```

### 思路

这是一个吸收马尔可夫链 / 递归概率题。设 `P(pos)` 表示从当前位置最终先到达终点的概率：

- 若 `pos == end`，`P(pos)=1`
- 若下一步出界，贡献 `0`
- 否则 `P(pos)=平均(P(next_pos))`

因为状态只有 5 个，可以直接建立线性方程；也可以利用结构手算。

### 手算结论

记中心为 `C=(0,0)`，四个边缘点为 arms。

1. 从边缘点到中心：只有一步走向中心成功，概率 `1/4`
2. 从中心到某个边缘终点：
   - 直接走到目标：`1/4`
   - 走到其他 3 个边缘点：概率 `3/4`，再以 `1/4` 回到中心继续

设 `x = P(C -> target_arm)`：

```text
x = 1/4 + (3/4) * (1/4) * x
x = 4/13
```

3. 从一个边缘点到另一个边缘点：

```text
P = (1/4) * P(C -> target_arm) = 1/13
```

### Python 解法

面试中最稳妥的写法是把上面的三种情况直接编码：

```python
def diamond_probability(xs, ys, xe, ye):
    start = (xs, ys)
    end = (xe, ye)
    center = (0, 0)
    valid = {(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)}

    if start not in valid or end not in valid:
        raise ValueError("invalid coordinates")
    if start == end:
        return 1.0

    if start != center and end == center:
        return 1 / 4
    if start == center and end != center:
        return 4 / 13
    if start != center and end != center:
        return 1 / 13

    raise RuntimeError("unreachable case")
```

### 面试要点

- 先说清楚这是 hitting probability，不是固定步数概率。
- 说明出界是吸收失败态，终点是吸收成功态。
- 如果状态更多，用线性方程或 value iteration；此题可手推。

---

## 2. Majority Voting Ensemble 准确率（已出现线索）

### 题目

有 `n` 个独立分类器，每个分类器预测正确的概率为 `p`，用多数投票作为最终预测。求 ensemble 的准确率。

### 答案

若 `n` 为奇数，多数投票正确等价于至少 `ceil(n/2)` 个分类器正确：

```text
P(correct) = sum_{k=ceil(n/2)}^n C(n,k) p^k (1-p)^(n-k)
```

Python：

```python
from math import comb

def majority_vote_accuracy(n: int, p: float) -> float:
    need = n // 2 + 1
    return sum(comb(n, k) * p**k * (1 - p)**(n - k)
               for k in range(need, n + 1))

print(majority_vote_accuracy(3, 0.7))  # 0.784
print(majority_vote_accuracy(5, 0.7))  # 0.83692
```

### 延伸

- 若 `p > 0.5` 且独立，增加模型数通常提升准确率。
- 若 `p < 0.5`，多数投票会更差；反转每个分类器反而有用。
- 若分类器错误高度相关，独立二项分布公式不成立，这是面试常见追问。

---

## 3. 不用 numpy 生成随机数 / Random Seed Function（已出现线索）

### 题目

不用 `numpy` 实现一个随机数生成器，并支持 seed。

### 思路

可实现线性同余生成器（Linear Congruential Generator, LCG）：

```text
X_{t+1} = (a * X_t + c) mod m
U_t = X_t / m
```

### Python 实现

```python
class LCG:
    def __init__(self, seed: int = 1):
        self.m = 2**31
        self.a = 1103515245
        self.c = 12345
        self.state = seed % self.m

    def random(self) -> float:
        """Return a pseudo-random float in [0, 1)."""
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m

    def randint(self, low: int, high: int) -> int:
        """Return integer in [low, high]."""
        return low + int(self.random() * (high - low + 1))

rng = LCG(seed=42)
print([rng.random() for _ in range(3)])
print([rng.randint(1, 6) for _ in range(5)])
```

### 面试要点

- seed 控制初始状态，因此同 seed 可复现。
- `m, a, c` 的选择会影响周期和质量。
- LCG 适合解释伪随机机制，不适合密码学。
- 若追问“均匀分布生成正态分布”，可用 Box-Muller：

```python
import math

def normal_pair(rng):
    u1 = max(rng.random(), 1e-12)
    u2 = rng.random()
    r = math.sqrt(-2 * math.log(u1))
    theta = 2 * math.pi * u2
    return r * math.cos(theta), r * math.sin(theta)
```

---

## 4. 字符串相邻整数所有组合求和（Interview Query 题库）

### 题目

给定数字字符串 `int_str`，求所有相邻子串对应整数之和。

示例：

```text
"12"  -> 1 + 2 + 12 = 15
"123" -> 1 + 2 + 3 + 12 + 23 + 123 = 164
```

### 直接解法

枚举所有连续子串：

```python
def adjacent_integer_sum(s: str) -> int:
    total = 0
    for i in range(len(s)):
        value = 0
        for j in range(i, len(s)):
            value = value * 10 + int(s[j])
            total += value
    return total

print(adjacent_integer_sum("12"))   # 15
print(adjacent_integer_sum("123"))  # 164
```

复杂度 `O(n^2)`，面试中通常足够；注意不要把非相邻组合如 `"13"` 算进去。

---

## 5. Matrix Sum 函数题（Interview Query 题库）

### 题目

写函数返回矩阵中所有元素的和，元素可正可负。

```python
def matrix_sum(matrix):
    return sum(sum(row) for row in matrix)

print(matrix_sum([[1, 2, 3], [4, 5, 6]]))      # 21
print(matrix_sum([[-1, -2], [3, 4]]))          # 4
```

### 追问

- 空矩阵返回什么？建议返回 `0`
- 行长度不一致怎么办？若题目未禁止，以上写法仍可处理 ragged list
- 若输入可能不是数字，应在生产代码中做校验；面试中先确认假设

---

## 6. Nested List 递归处理（已出现线索）

### 题目

给定可能嵌套的 list，递归求所有数字之和。

```python
def nested_sum(x):
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, list):
        return sum(nested_sum(item) for item in x)
    return 0

print(nested_sum([1, [2, 3], [[4], 5]]))  # 15
```

### 面试要点

- 递归终止条件：数字直接返回。
- 递归展开条件：list 中每个元素继续处理。
- 可追问最大递归深度，极深嵌套时可改用 stack。

---

## 7. 组合概率题（已出现线索）

### 标准练习版本

从 `n` 个物品中随机选 `k` 个，其中有 `m` 个是目标物品。求至少选中一个目标物品的概率。

### 答案

用反面事件更简单：

```text
P(at least one) = 1 - C(n-m, k) / C(n, k)
```

```python
from math import comb

def prob_at_least_one(n, m, k):
    if k > n:
        return 0.0
    if k > n - m:
        return 1.0
    return 1 - comb(n - m, k) / comb(n, k)

print(prob_at_least_one(n=100, m=5, k=10))
```

### 可能追问

- 恰好选中 `r` 个目标：超几何分布

```text
P(X=r) = C(m,r) C(n-m,k-r) / C(n,k)
```

---

## 8. p-value 给非技术同事解释（Interview Query 题库）

### 回答模板

p-value 是：如果“新功能其实没有效果”这个假设成立，我们仍然观察到当前这么极端结果的概率。

不能说：

```text
p-value = 新功能无效的概率
```

应该说：

```text
在无效假设为真的前提下，数据看起来至少这么极端的概率。
```

业务解释：

> 如果 p-value = 0.04，意思是：假设按钮颜色改变完全没有真实影响，那么我们仍然看到当前或更极端提升的概率约为 4%。这提供了反对“无影响”的证据，但不等于有 96% 概率新按钮有效。

---

## 9. A/B Test 显著性与有效性（Interview Query 题库）

### 题目

PM 看到实验 p-value = 0.04，就认为新版本一定更好。你如何评估？

### 答案框架

不能只看 p-value，需要检查：

1. 实验是否预先设定假设和主指标
2. 是否有多重检验问题
3. 是否提前 peeking，多次查看结果后停止
4. 样本是否随机分流，是否有 SRM（sample ratio mismatch）
5. 效应大小是否有业务意义
6. 置信区间是否稳定
7. 是否影响 guardrail metrics，如取消率、退款率、页面延迟

简洁回答：

> `p=0.04` 只是统计证据的一部分。我会先确认实验设计是否有效，再看 effect size、confidence interval、multiple testing、peeking 和 guardrail metrics。若这些都通过，才建议逐步 rollout。

---

## 10. Agoda Staff / Lead DS 准备重点

公开岗位页和面经显示，Staff/Lead DS 重点包括：

- Python / SQL / algorithms
- statistics, probability, experiment design
- ML/DL 基础与推荐、排序、定价、个性化
- 大规模数据处理，如 PySpark / Scala / Hadoop
- 能把模型结果和业务目标连接起来

建议准备顺序：

1. 概率 DP：随机游走、赌徒破产、吸收概率
2. 组合概率：超几何、二项分布、多数投票
3. Python 函数题：递归、字符串、矩阵、随机数
4. A/B test：p-value、CI、power、SRM、peeking
5. SQL：moving average、cohort retention、窗口函数

---

## Sources

- Glassdoor Agoda Data Scientist / Senior Data Scientist interview questions: https://www.glassdoor.co.uk/Interview/Agoda-Data-Scientist-Interview-Questions-EI_IE461386.0,5_KO6,20.htm
- Glassdoor diamond platform probability question: https://www.glassdoor.co.uk/Interview/Problem-Description-You-are-standing-on-a-unique-diamond-shaped-platform-composed-of-5-tiles-These-tiles-are-positioned-a-QTN_6695329.htm
- Interview Query Agoda Data Scientist interview guide: https://www.interviewquery.com/interview-guides/agoda-data-scientist
- Prepfully Agoda Data Scientist question bank: https://prepfully.com/interview-questions/agoda/data-scientist
- HackerRank Agoda Lead/Staff Data Scientist posting: https://www.hackerrank.com/apply/473881
