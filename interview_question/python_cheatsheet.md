# Python Cheatsheet（不常用 Python 的人版）

> 收录最容易忘记的函数签名、语法和常用模式。语言特性 > 算法细节。

---

## 1. 数学函数 `math`

```python
import math

math.comb(n, r)        # C(n,r) 组合数，不放回，无顺序  comb(5,2)=10
math.perm(n, r)        # P(n,r) 排列数，有顺序            perm(5,2)=20
math.factorial(n)      # n!
math.gcd(a, b)         # 最大公约数
math.lcm(a, b)         # 最小公倍数 (Python 3.9+)

math.sqrt(x)           # √x （返回 float）
math.isqrt(x)          # √x 取整 （返回 int，等价 int(x**0.5)）
math.exp(x)            # e^x
math.log(x)            # ln x（自然对数）
math.log(x, base)      # log_base(x)
math.log2(x)           # log₂x
math.log10(x)          # log₁₀x
math.ceil(x)           # 向上取整
math.floor(x)          # 向下取整

math.pi                # 3.14159...
math.e                 # 2.71828...
math.inf               # 正无穷
float('inf')           # 同上（不用 import）
float('-inf')          # 负无穷
```

---

## 2. 组合 / 迭代工具 `itertools`

```python
from itertools import combinations, permutations, product, combinations_with_replacement

# 组合（无顺序，不重复）
list(combinations([1,2,3,4], 2))
# [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]

# 排列（有顺序）
list(permutations([1,2,3], 2))
# [(1,2),(1,3),(2,1),(2,3),(3,1),(3,2)]

# 笛卡尔积（多个可迭代对象的所有组合）
list(product([0,1], repeat=3))
# [(0,0,0),(0,0,1),...,(1,1,1)]  → 枚举所有二进制串

list(product('AB', [1,2]))
# [('A',1),('A',2),('B',1),('B',2)]

# 有放回组合
list(combinations_with_replacement([1,2,3], 2))
# [(1,1),(1,2),(1,3),(2,2),(2,3),(3,3)]

# 其他实用
from itertools import accumulate, chain, islice
list(accumulate([1,2,3,4]))        # [1,3,6,10] 前缀和
list(chain([1,2],[3,4],[5]))       # [1,2,3,4,5] 拼接多个迭代器
list(islice(range(100), 5, 10))    # [5,6,7,8,9] 切片迭代器
```

---

## 3. NumPy 基础

```python
import numpy as np

# ── 创建数组 ────────────────────────────────────────────────
np.array([1, 2, 3])                 # 从列表创建
np.zeros(5)                         # [0. 0. 0. 0. 0.]
np.ones((2, 3))                     # 2×3 全1矩阵
np.full(4, 7)                       # [7 7 7 7]
np.eye(3)                           # 3×3 单位矩阵

np.arange(0, 10, 2)                 # [0 2 4 6 8]  start, stop, step（不含stop）
np.linspace(0, 1, 5)                # [0. 0.25 0.5 0.75 1.]  start, stop, 点数（含两端）

# ── 形状操作 ────────────────────────────────────────────────
a = np.array([[1,2,3],[4,5,6]])
a.shape                             # (2, 3)
a.reshape(3, 2)                     # 变形，元素数不变
a.flatten()                         # 展平为一维
a.T                                 # 转置

# ── 数学运算（均逐元素）────────────────────────────────────
a + b;  a * b;  a ** 2;  np.sqrt(a)
np.exp(a);  np.log(a);  np.abs(a)

# ── 统计 ────────────────────────────────────────────────────
a.sum();  a.sum(axis=0)             # 全部求和 / 按列求和
a.mean(); a.std(); a.var()
a.min(); a.max()
a.argmin(); a.argmax()              # 最小/最大值的索引
np.cumsum(a)                        # 累积和
np.sort(a); np.argsort(a)           # 排序 / 返回排序索引
np.percentile(a, 75)                # 第75百分位数
np.median(a)                        # 中位数

# ── 随机数 ────────────────────────────────────────────────
np.random.seed(42)
np.random.rand(3)                   # [0,1) 均匀分布，3个
np.random.randn(3)                  # 标准正态，3个
np.random.normal(mu, sigma, size)   # N(mu, sigma^2)，size个
np.random.uniform(low, high, size)  # 均匀分布
np.random.randint(0, 10, size=5)    # 随机整数
np.random.choice(arr, size, replace=True)  # 有放回抽样

# ── 线性代数 ────────────────────────────────────────────────
np.dot(a, b)                        # 矩阵乘法（也可用 a @ b）
np.linalg.solve(A, b)               # 解线性方程组 Ax = b
np.linalg.inv(A)                    # 矩阵求逆
np.linalg.eig(A)                    # 特征值和特征向量
np.linalg.norm(v)                   # 向量范数（默认 L2）

# ── 布尔索引 ────────────────────────────────────────────────
a = np.array([1, 5, 2, 8, 3])
a[a > 3]                            # array([5, 8])
a[(a > 2) & (a < 7)]               # array([5, 3])
np.where(a > 3, 1, 0)              # 条件替换：[0,1,0,1,0]
```

---

## 4. 列表 / 字典 / 集合操作

```python
# ── 列表 ────────────────────────────────────────────────────
lst = [3, 1, 4, 1, 5, 9]

lst.append(x)           # 末尾加一个
lst.extend([7, 8])      # 末尾加多个（extend vs append 的区别）
lst.insert(i, x)        # 在位置 i 插入 x
lst.pop()               # 删除并返回最后一个
lst.pop(i)              # 删除并返回位置 i 的元素
lst.remove(x)           # 删除第一个值为 x 的元素
lst.index(x)            # 返回 x 第一次出现的索引
lst.count(x)            # x 出现次数
lst.reverse()           # 原地反转
lst[::-1]               # 反转（返回新列表）

lst.sort()              # 原地升序
lst.sort(reverse=True)  # 原地降序
sorted(lst, key=abs)    # 返回新列表，按绝对值排序
sorted(lst, key=lambda x: -x)   # 降序（用负号）

# 列表推导式
squares = [x**2 for x in range(10)]
evens   = [x for x in range(20) if x % 2 == 0]
flat    = [x for row in matrix for x in row]   # 二维展平

# ── 字典 ────────────────────────────────────────────────────
d = {'a': 1, 'b': 2}

d.get('c', 0)           # 安全取值，不存在返回默认值 0
d.setdefault('c', 0)    # 若不存在则设为 0，存在则不变
d.update({'c': 3})      # 合并/更新
d.pop('a')              # 删除键 'a' 并返回值
'a' in d                # 判断键是否存在

d.keys(); d.values(); d.items()
sorted(d.items(), key=lambda kv: kv[1])   # 按值排序

# 字典推导式
inv = {v: k for k, v in d.items()}   # 键值互换
freq = {x: lst.count(x) for x in lst}

# ── 集合 ────────────────────────────────────────────────────
s = {1, 2, 3}
s.add(4); s.remove(3); s.discard(99)   # discard 不存在不报错
s1 | s2   # 并集
s1 & s2   # 交集
s1 - s2   # 差集
s1 ^ s2   # 对称差（只在一个集合中的元素）
```

---

## 5. 字符串操作

```python
s = "  Hello, World!  "

s.strip()              # "Hello, World!"  去除两端空白
s.lower(); s.upper()   # 大小写
s.replace('World', 'Python')
s.split(', ')          # ['Hello', 'World!']
', '.join(['a', 'b', 'c'])   # "a, b, c"
s.startswith('H'); s.endswith('!')
s.find('llo')          # 返回索引，找不到返回 -1
s.count('l')           # 出现次数
'42'.zfill(5)          # '00042'  左补零

# f-string（最常用的格式化方式）
name, val = "pi", 3.14159
f"{name} = {val:.2f}"     # "pi = 3.14"
f"{1000000:,}"             # "1,000,000"  千位分隔符
f"{0.05:.1%}"              # "5.0%"  百分比

# 字符与 ASCII
ord('A')   # 65
chr(65)    # 'A'
```

---

## 6. 内置函数速查

```python
# 数值
abs(-3)              # 3
round(3.14159, 2)    # 3.14
divmod(17, 5)        # (3, 2)  商和余数
pow(2, 10)           # 1024 （等价 2**10）
pow(2, -1, MOD)      # 模逆元（Python 3.8+）

# 迭代
max([3,1,4,1,5])                        # 5
min([3,1,4,1,5])                        # 1
max('ab', 'cd', key=len)                # 按长度取最大
sum([1,2,3,4])                          # 10
sum(x**2 for x in range(5))            # 生成器求和

enumerate(['a','b','c'])                # (0,'a'),(1,'b'),(2,'c')
zip([1,2,3], ['a','b','c'])            # (1,'a'),(2,'b'),(3,'c')
zip(*matrix)                            # 矩阵转置（解包）
map(str, [1,2,3])                       # 迭代器，转成 list: ['1','2','3']
filter(lambda x: x>2, [1,2,3,4])       # 迭代器，转成 list: [3,4]
any([False, True, False])               # True
all([True, True, False])                # False

# 类型转换
int('42'); int('ff', 16)               # 字符串→整数；十六进制字符串→十进制
float('3.14')
list(range(5))                          # [0,1,2,3,4]
tuple([1,2,3])                          # (1,2,3)
set([1,1,2,3])                          # {1,2,3} 去重
''.join(map(str, [1,2,3]))             # '123'  整数列表→字符串
```

---

## 7. `collections` 模块

```python
from collections import Counter, defaultdict, deque, OrderedDict

# Counter — 计数器
c = Counter("abracadabra")   # Counter({'a':5,'b':2,'r':2,'c':1,'d':1})
c.most_common(2)             # [('a',5),('b',2)]
c['a']                       # 5；不存在的键返回 0（不报 KeyError）
c1 + c2; c1 - c2             # 计数器加减

# defaultdict — 有默认值的字典（不用手动初始化）
d = defaultdict(list)
d['key'].append(1)           # 直接追加，无需先 d['key'] = []
d = defaultdict(int)
d['count'] += 1              # 无需先初始化为 0

# deque — 双端队列（比 list 在头部操作快）
q = deque([1,2,3])
q.appendleft(0)   # [0,1,2,3]
q.popleft()       # 返回0，O(1)
q.append(4)       # [1,2,3,4]
q.pop()           # 返回4
deque(maxlen=3)   # 固定大小，超出自动丢弃最旧的
```

---

## 8. Scipy 统计函数

```python
from scipy import stats
from scipy.stats import norm, binom, poisson, expon

# 正态分布 N(mu, sigma)
norm.cdf(1.96)                    # P(Z ≤ 1.96) ≈ 0.975
norm.cdf(x, loc=mu, scale=sigma)  # P(X ≤ x)
norm.ppf(0.975)                   # 分位数（逆CDF）：z使P(Z≤z)=0.975 → 1.96
norm.pdf(x, loc=mu, scale=sigma)  # 概率密度
norm.rvs(size=1000)               # 随机采样

# 二项分布 Bin(n, p)
binom.pmf(k, n, p)               # P(X = k)
binom.cdf(k, n, p)               # P(X ≤ k)

# 泊松分布 Pois(lambda)
poisson.pmf(k, mu=lam)           # P(X = k)
poisson.cdf(k, mu=lam)           # P(X ≤ k)

# 指数分布 Exp(lambda)：scipy 用 scale=1/lambda
expon.cdf(x, scale=1/lam)        # P(X ≤ x) = 1 - e^{-lambda*x}

# 假设检验
t_stat, p_val = stats.ttest_ind(group_a, group_b)      # 双样本 t 检验
t_stat, p_val = stats.ttest_1samp(sample, popmean)     # 单样本 t 检验
chi2, p_val, dof, expected = stats.chi2_contingency(table)   # 卡方检验
corr, p_val = stats.pearsonr(x, y)                     # 皮尔逊相关
corr, p_val = stats.spearmanr(x, y)                    # 斯皮尔曼相关
```

---

## 9. 常用代码模式

```python
# ── 频率统计 ────────────────────────────────────────────────
from collections import Counter
Counter([1,2,2,3,3,3]).most_common()   # 频率排序

# ── 分组（group by）────────────────────────────────────────
from itertools import groupby
data = sorted(data, key=lambda x: x['type'])   # 必须先排序
for key, group in groupby(data, key=lambda x: x['type']):
    print(key, list(group))

# ── 堆（优先队列）─────────────────────────────────────────
import heapq
heap = []
heapq.heappush(heap, val)     # 默认最小堆
heapq.heappop(heap)           # 弹出最小值
heapq.nlargest(k, lst)        # 前k大（等价 sorted(lst)[-k:]，但更快）
heapq.nsmallest(k, lst)
# 最大堆：存 -val，取出时再取负
heapq.heappush(heap, -val); top = -heapq.heappop(heap)

# ── 二分查找 ───────────────────────────────────────────────
import bisect
bisect.bisect_left(sorted_lst, x)   # 插入位置（左侧，等价lower_bound）
bisect.bisect_right(sorted_lst, x)  # 插入位置（右侧，等价upper_bound）
bisect.insort(sorted_lst, x)        # 插入并保持有序

# ── 记忆化 / 缓存 ──────────────────────────────────────────
from functools import lru_cache
@lru_cache(maxsize=None)
def fib(n):
    if n <= 1: return n
    return fib(n-1) + fib(n-2)

# ── Monte Carlo 模拟（通用模板）────────────────────────────
import random
def monte_carlo(n=100_000):
    hits = sum(1 for _ in range(n) if <condition>)
    return hits / n

# ── 种子固定（复现结果）────────────────────────────────────
import random, numpy as np
random.seed(42)
np.random.seed(42)

# ── 排序技巧 ───────────────────────────────────────────────
# 按多个字段排序
sorted(lst, key=lambda x: (x[0], -x[1]))   # 先按第0列升序，再按第1列降序
# 字典按值排序
sorted(d.items(), key=lambda kv: kv[1], reverse=True)

# ── 展平嵌套列表 ───────────────────────────────────────────
flat = [x for row in nested for x in row]   # 二层
import itertools
flat = list(itertools.chain.from_iterable(nested))  # 通用

# ── 矩阵转置 ───────────────────────────────────────────────
transposed = list(zip(*matrix))             # matrix 是列表的列表

# ── 滑动窗口 ───────────────────────────────────────────────
from collections import deque
window = deque(maxlen=k)
for x in arr:
    window.append(x)   # 超出 k 个自动丢弃最旧
    if len(window) == k:
        process(window)
```

---

## 10. 常用算法框架（面试）

```python
# ── 回溯（Backtracking）────────────────────────────────────
result = []
def backtrack(path, choices):
    if is_complete(path):
        result.append(path[:])
        return
    for c in choices:
        path.append(c)
        backtrack(path, next_choices(choices, c))
        path.pop()

# ── BFS（最短路径 / 层序遍历）──────────────────────────────
from collections import deque
q = deque([(start, 0)]); visited = {start}
while q:
    node, dist = q.popleft()
    for nb in neighbors(node):
        if nb not in visited:
            visited.add(nb)
            q.append((nb, dist + 1))

# ── DFS（连通区域 / 拓扑）─────────────────────────────────
def dfs(node, visited=None):
    if visited is None: visited = set()
    if node in visited: return
    visited.add(node)
    for nb in neighbors(node): dfs(nb, visited)

# ── 二分搜索 ───────────────────────────────────────────────
lo, hi = 0, len(arr) - 1
while lo <= hi:
    mid = (lo + hi) // 2
    if arr[mid] == target: return mid
    elif arr[mid] < target: lo = mid + 1
    else: hi = mid - 1

# ── DP 模板 ────────────────────────────────────────────────
dp = [0] * (n + 1)       # 1D
dp = [[0]*m for _ in range(n)]   # 2D（注意不能用 [[0]*m]*n，共享引用）

@lru_cache(None)          # 递归+记忆化（更直观）
def dp(i, j): ...
```

---

## 11. 常见陷阱

```python
# 1. 可变默认参数（经典 bug）
def bad(lst=[]):    lst.append(1); return lst   # 每次调用共享同一个 []
def good(lst=None): 
    if lst is None: lst = []
    lst.append(1); return lst

# 2. 二维列表初始化
bad  = [[0]*3]*3         # 三行指向同一个列表！
good = [[0]*3 for _ in range(3)]

# 3. 整数除法 vs 浮点除法
7 // 2    # 3  （地板除）
7 / 2     # 3.5
-7 // 2   # -4  （向下取整，不是截断）

# 4. is vs ==
a = [1,2,3]; b = [1,2,3]
a == b    # True（值相等）
a is b    # False（不同对象）
# 小整数 (-5~256) 会被缓存，is 可能为 True，但不要依赖这个

# 5. 深拷贝 vs 浅拷贝
import copy
shallow = lst[:]           # 浅拷贝，嵌套对象仍共享
deep    = copy.deepcopy(lst)   # 完全独立

# 6. 字典遍历时修改会报错
for k in list(d.keys()):   # 先转成 list 再遍历
    if condition: del d[k]
```

---

## 12. 复杂度速查

| 操作 | list | dict / set | heapq | bisect |
|---|---|---|---|---|
| 访问 `a[i]` | O(1) | O(1) | — | O(log n) |
| 末尾插入/删除 | O(1) | O(1) | O(log n) | — |
| 头部插入/删除 | O(n) | O(1) | — | — |
| 搜索 | O(n) | O(1) | — | O(log n) |
| 排序 | O(n log n) | — | — | — |
| 前k大/小 | O(n log k) via heapq | — | O(n + k log n) | — |
