# Python DS Interview Cheatsheet

---

## Essential Imports

```python
# Analytical / Stats
import random, math
import numpy as np
from collections import Counter, defaultdict
from functools import lru_cache
from scipy import stats                        # optional: ttest_ind, norm, binom

# Algorithmic
from collections import deque, Counter, defaultdict
import heapq, bisect, itertools, math
from functools import lru_cache
from typing import List, Optional
```

---

## Analytical Patterns (Probability / Stats)

### Monte Carlo — estimate any probability by simulation
```python
def monte_carlo(n=100_000):
    hits = sum(1 for _ in range(n) if <condition>)
    return hits / n
# E.g., P(circle): hits if random.random()**2 + random.random()**2 < 1
```

### Markov Chain / DP — expected value / probability
```python
# Backward induction: dp[state] = value of being in this state
@lru_cache(None)
def dp(state):
    if is_terminal(state): return base_case
    return some_combination(dp(next_state) for next_state in transitions(state))
# For linear systems: numpy.linalg.solve(A, b)
```

### Reservoir Sampling — uniformly sample k from stream of unknown size n
```python
reservoir = []
for i, item in enumerate(stream):
    if i < k:
        reservoir.append(item)
    else:
        j = random.randint(0, i)           # P(keep new) = k/(i+1)
        if j < k:
            reservoir[j] = item
```

### Rejection Sampling — rand_a → rand_b
```python
# rand5 → rand7: generate uniform in [0, 20], reject [21,24]
def rand7():
    while True:
        val = 5 * (rand5() - 1) + (rand5() - 1)   # uniform [0,24]
        if val < 21: return val % 7 + 1
```

### Biased → Fair Coin (Von Neumann)
```python
def fair_coin(biased):           # P(H)=p, P(T)=1-p, P(HT)=P(TH)=p(1-p)
    while True:
        a, b = biased(), biased()
        if a != b: return a      # HT→1, TH→0
```

### Uniform Disk Sampling (rejection vs polar)
```python
# Polar (correct): r = sqrt(U) avoids center clustering
r = math.sqrt(random.random())
theta = random.uniform(0, 2 * math.pi)
x, y = r * math.cos(theta), r * math.sin(theta)
```

### Optimal Stopping (Secretary Problem) — skip first n/e, then take first better
```python
k = int(n / math.e)              # explore phase length
best_in_phase = max(candidates[:k])
chosen = next((c for c in candidates[k:] if c > best_in_phase), candidates[-1])
```

### A/B Testing / Hypothesis Testing
```python
from scipy import stats
t_stat, p_val = stats.ttest_ind(group_a, group_b)   # two-sample t-test
# Proportions: statsmodels.stats.proportion.proportions_ztest([s_a,s_b],[n_a,n_b])
print("significant" if p_val < 0.05 else "not significant")
```

### Key Formulas
| Problem | Formula |
|---|---|
| Birthday paradox (50% at n) | `n ≈ 1.177 * sqrt(N)` where N = days |
| Coupon collector (k coupons) | `E = k * H_k` where `H_k = sum(1/i, i=1..k)` |
| Gambler's ruin P(reach N from k) | `P = k/N` (fair), `P = (1-(q/p)^k)/(1-(q/p)^N)` (biased) |
| Geometric distribution | `E[trials until first success] = 1/p` |

---

## Algorithm Patterns (LeetCode)

### Backtracking — choose / explore / unchoose
```python
def backtrack(path, choices):
    if is_complete(path):
        result.append(path[:])
        return
    for c in choices:
        if not valid(c, path): continue
        path.append(c)                     # choose
        backtrack(path, remaining(choices, c))  # explore
        path.pop()                         # unchoose

# Covers: Permutations, Subsets, Combination Sum,
#         Generate Parentheses, Word Search (DFS on grid)
```

### Dynamic Programming
```python
# 1D (Coin Change / Knapsack)
dp = [float('inf')] * (amount + 1); dp[0] = 0
for coin in coins:
    for i in range(coin, amount + 1):
        dp[i] = min(dp[i], dp[i - coin] + 1)

# 2D Grid (Min Path Sum)
for i in range(m):
    for j in range(n):
        dp[i][j] = grid[i][j] + min(dp[i-1][j] if i else inf,
                                     dp[i][j-1] if j else inf)

# Memoization
@lru_cache(None)
def solve(i, j): ...
```

### DFS / BFS
```python
# DFS recursive (tree/graph)
def dfs(node, visited=set()):
    if node in visited: return
    visited.add(node)
    for nb in graph[node]: dfs(nb, visited)

# BFS (shortest path, level order)
from collections import deque
q = deque([start]); visited = {start}
while q:
    node = q.popleft()
    for nb in graph[node]:
        if nb not in visited:
            visited.add(nb); q.append(nb)
```

### Two Pointers / Sliding Window
```python
# Opposite ends (Two Sum sorted, container with water)
lo, hi = 0, len(arr) - 1
while lo < hi:
    if condition: lo += 1
    else: hi -= 1

# Same direction (max subarray with constraint)
l = 0
for r in range(len(arr)):
    window.add(arr[r])
    while not valid(window): window.remove(arr[l]); l += 1
    ans = max(ans, r - l + 1)
```

### Binary Search
```python
lo, hi = 0, len(arr) - 1
while lo <= hi:
    mid = (lo + hi) // 2
    if arr[mid] == target: return mid
    elif arr[mid] < target: lo = mid + 1
    else: hi = mid - 1

# "Search on answer": bisect.bisect_left(range(lo, hi+1), target, key=feasible)
```

### Heap (Priority Queue)
```python
import heapq
heap = []
heapq.heappush(heap, val)        # min-heap by default
heapq.heappop(heap)
heapq.nlargest(k, iterable)      # top-k
heapq.nsmallest(k, iterable)
# Max-heap: push/pop negated values (-val)
```

### Collections Quickref
```python
Counter("abracadabra").most_common(3)    # [('a',5),('b',2),('r',2)]
d = defaultdict(list); d[key].append(v) # no KeyError
q = deque(maxlen=k)                      # fixed-size sliding window
q.appendleft(x); q.popleft(); q.pop()
```

---

## Complexity Quick Reference

| Operation | List | Dict/Set | heapq | bisect (sorted list) |
|---|---|---|---|---|
| Access | O(1) | O(1) | — | O(log n) |
| Insert/Delete end | O(1) | O(1) | O(log n) push/pop | O(log n) + O(n) shift |
| Search | O(n) | O(1) | — | O(log n) |
| Sort | O(n log n) | — | — | — |
| k-th largest | O(n) | — | O(n + k log n) | — |
