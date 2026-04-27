# Tensor Shapes and Manipulation in PyTorch

## Understanding Dimensions (Axes)

A **dimension** (also called an **axis**) is one independent direction along which a tensor stores data. The number of dimensions is the tensor's **rank** (or `ndim`).

```python
import torch

scalar = torch.tensor(42)
# rank 0 — no dimensions, just a value

vector = torch.tensor([10, 20, 30])
# rank 1 — one dimension (axis 0) of length 3
#   axis 0 ──►  [10, 20, 30]

matrix = torch.tensor([[1, 2, 3],
                        [4, 5, 6]])
# rank 2 — two dimensions
#   axis 0 (rows) ↓     axis 1 (columns) →
#       [[1, 2, 3],
#        [4, 5, 6]]

batch = torch.randn(8, 3, 32, 32)
# rank 4 — four dimensions
#   axis 0: batch    (8 samples)
#   axis 1: channel  (3 color channels)
#   axis 2: height   (32 pixels)
#   axis 3: width    (32 pixels)
```

### How `dim` Works in PyTorch Functions

Many PyTorch functions accept a `dim` argument that tells the function **which axis to operate along**. The selected dimension is consumed (collapsed) in the output.

```python
x = torch.tensor([[1, 2, 3],
                   [4, 5, 6]])   # shape: (2, 3)

x.sum(dim=0)    # tensor([5, 7, 9])   — sum down rows,    result shape: (3,)
x.sum(dim=1)    # tensor([6, 15])     — sum across cols,   result shape: (2,)
x.sum(dim=-1)   # tensor([6, 15])     — last dim = dim 1

x.mean(dim=0)   # tensor([2.5, 3.5, 4.5])
x.max(dim=1)    # values: tensor([3, 6]), indices: tensor([2, 2])

x.softmax(dim=1)  # softmax across columns (each row sums to 1)
x.argmax(dim=0)   # index of max in each column
```

**Mental model**: `dim=d` means "walk along axis `d` and collapse it." The result has all the original dimensions except `d`.

### Keeping Dimensions with `keepdim`

By default, reduction operations remove the reduced dimension. Use `keepdim=True` to preserve it as size 1 — this is critical for correct broadcasting.

```python
x = torch.randn(4, 5)

x.sum(dim=1)                 # shape: (4,)
x.sum(dim=1, keepdim=True)   # shape: (4, 1) — can broadcast back against x

# Common pattern: normalize each row
row_sums = x.sum(dim=1, keepdim=True)   # (4, 1)
normalized = x / row_sums               # (4, 5) / (4, 1) → broadcasts to (4, 5)
```

### Negative Indexing

Dimensions can be referenced from the end using negative indices. This is useful when writing dimension-agnostic code.

```python
x = torch.randn(2, 3, 4, 5)

x.size(-1)     # 5  — last dimension
x.size(-2)     # 4  — second to last

x.sum(dim=-1)  # sums along last dim → shape: (2, 3, 4)
```

### Named Dimensions (Named Tensors)

PyTorch supports optional named dimensions to reduce indexing errors:

```python
images = torch.randn(8, 3, 32, 32, names=('B', 'C', 'H', 'W'))

images.sum('C')              # sum over channel dim by name
images.align_to('B', 'H', 'W', 'C')   # reorder by name instead of memorizing indices
```

---

## Dimension-Specific Manipulation Functions

### `torch.narrow` — select a slice along a dimension

```python
x = torch.arange(12).reshape(3, 4)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

x.narrow(dim=1, start=1, length=2)
# tensor([[ 1,  2],
#         [ 5,  6],
#         [ 9, 10]])
```

### `torch.select` — index a single position along a dimension (removes that dim)

```python
x = torch.randn(3, 4, 5)

x.select(dim=1, index=2)    # shape: (3, 5) — picks index 2 along dim 1
# equivalent to x[:, 2, :]
```

### `torch.index_select` — gather specific indices along a dimension

```python
x = torch.randn(4, 5)
indices = torch.tensor([0, 3])

torch.index_select(x, dim=0, index=indices)  # shape: (2, 5) — rows 0 and 3
torch.index_select(x, dim=1, index=indices)  # shape: (4, 2) — columns 0 and 3
```

### `torch.gather` — collect values along a dimension using index tensor

```python
x = torch.tensor([[10, 20, 30],
                   [40, 50, 60]])

# Pick one element per row: row 0 → col 2, row 1 → col 0
idx = torch.tensor([[2],
                     [0]])

torch.gather(x, dim=1, index=idx)
# tensor([[30],
#         [40]])
```

### `torch.scatter` — inverse of gather, writes values into positions

```python
target = torch.zeros(2, 4, dtype=torch.float)
idx = torch.tensor([[1, 3],
                     [0, 2]])
src = torch.tensor([[10., 20.],
                     [30., 40.]])

target.scatter_(dim=1, index=idx, src=src)
# tensor([[ 0, 10,  0, 20],
#         [30,  0, 40,  0]])
```

### `torch.flip` — reverse elements along given dimensions

```python
x = torch.tensor([[1, 2, 3],
                   [4, 5, 6]])

torch.flip(x, dims=[0])      # flip rows:    [[4,5,6], [1,2,3]]
torch.flip(x, dims=[1])      # flip columns: [[3,2,1], [6,5,4]]
torch.flip(x, dims=[0, 1])   # flip both
```

### `torch.roll` — shift elements along a dimension (wraps around)

```python
x = torch.tensor([1, 2, 3, 4, 5])

torch.roll(x, shifts=2)       # tensor([4, 5, 1, 2, 3])
torch.roll(x, shifts=-1)      # tensor([2, 3, 4, 5, 1])

# Works on specific dims in higher-rank tensors
m = torch.arange(6).reshape(2, 3)
torch.roll(m, shifts=1, dims=1)   # rolls each row right by 1
```

### `torch.unbind` — remove a dimension, returning a tuple of slices

```python
x = torch.randn(3, 4)

torch.unbind(x, dim=0)   # tuple of 3 tensors, each shape (4,)
torch.unbind(x, dim=1)   # tuple of 4 tensors, each shape (3,)
```

### `torch.movedim` — move dimensions to new positions

```python
x = torch.randn(2, 3, 4, 5)

torch.movedim(x, source=1, destination=3)   # shape: (2, 4, 5, 3)
torch.movedim(x, source=(0, 1), destination=(2, 3))  # shape: (4, 5, 2, 3)
```

---

## Broadcasting and Dimension Alignment

When operating on tensors with different shapes, PyTorch **broadcasts** by automatically expanding size-1 dimensions. Dimensions are aligned from the right.

```python
a = torch.randn(4, 3)     # (4, 3)
b = torch.randn(   3)     # (   3)  — treated as (1, 3) then expanded to (4, 3)
result = a + b             # (4, 3)

a = torch.randn(4, 1)     # (4, 1)
b = torch.randn(1, 3)     # (1, 3)
result = a + b             # (4, 3) — both dimensions broadcast

# This FAILS — dimensions don't match and neither is 1
a = torch.randn(4, 3)
b = torch.randn(4, 2)
# a + b → RuntimeError
```

**Broadcasting rules** (checked right-to-left):
1. If a tensor has fewer dims, pad with 1s on the left
2. For each dimension pair, sizes must either match or one must be 1
3. Size-1 dimensions are stretched to match the other tensor

---

## Tensor Shape Basics

A tensor's **shape** (or **size**) describes the number of elements along each dimension. It is the single most important concept to internalize when working with PyTorch — the majority of runtime errors in deep learning code come down to shape mismatches.

```python
import torch

scalar = torch.tensor(42)          # shape: ()        — 0-D, a single number
vector = torch.tensor([1, 2, 3])   # shape: (3,)      — 1-D
matrix = torch.tensor([[1, 2],
                        [3, 4]])   # shape: (2, 2)    — 2-D
cube   = torch.randn(2, 3, 4)     # shape: (2, 3, 4) — 3-D
```

Inspect a tensor's shape with any of:

```python
cube.shape      # torch.Size([2, 3, 4])
cube.size()     # torch.Size([2, 3, 4])  — equivalent method call
cube.ndim       # 3                      — number of dimensions (rank)
cube.numel()    # 24                     — total number of elements (2*3*4)
```

### Common Shape Conventions

| Domain | Typical Shape | Meaning |
|---|---|---|
| Batch of vectors | `(B, D)` | B samples, D features |
| Batch of images (CHW) | `(B, C, H, W)` | batch, channels, height, width |
| Sequence data | `(B, T, D)` | batch, time steps, features |
| Single image (PIL/torchvision) | `(C, H, W)` | channels first |

---

## Reshaping Tensors

### `reshape` and `view`

Both change a tensor's shape without changing its data. The total number of elements must stay the same.

```python
x = torch.arange(12)        # shape: (12,)

x.reshape(3, 4)              # shape: (3, 4)
x.reshape(2, 2, 3)           # shape: (2, 2, 3)
x.view(4, 3)                 # shape: (4, 3)
```

**Key difference**: `view` requires the tensor to be contiguous in memory and always returns a view (shared memory). `reshape` works on any tensor — it returns a view when possible, otherwise copies the data.

```python
# Use -1 to let PyTorch infer one dimension
x.reshape(3, -1)   # shape: (3, 4) — PyTorch computes 12/3 = 4
x.reshape(-1, 6)   # shape: (2, 6)
```

### `flatten` and `unflatten`

```python
images = torch.randn(8, 3, 32, 32)

# Flatten all spatial dims into one — common before a linear layer
images.flatten(start_dim=1)              # shape: (8, 3072)   i.e. 3*32*32

# Flatten specific range of dims
images.flatten(start_dim=2)              # shape: (8, 3, 1024) i.e. 32*32

# Unflatten restores structure
flat = images.flatten(start_dim=1)       # (8, 3072)
flat.unflatten(1, (3, 32, 32))           # (8, 3, 32, 32)
```

---

## Adding and Removing Dimensions

### `unsqueeze` — add a dimension of size 1

```python
x = torch.tensor([1, 2, 3])   # shape: (3,)

x.unsqueeze(0)                 # shape: (1, 3) — new batch dim
x.unsqueeze(1)                 # shape: (3, 1) — column vector
x.unsqueeze(-1)                # shape: (3, 1) — same as above
```

### `squeeze` — remove dimensions of size 1

```python
x = torch.randn(1, 3, 1, 5)

x.squeeze()       # shape: (3, 5)   — removes ALL size-1 dims
x.squeeze(0)      # shape: (3, 1, 5) — removes only dim 0
x.squeeze(2)      # shape: (1, 3, 5) — removes only dim 2
```

### Indexing with `None` (equivalent to `unsqueeze`)

```python
x = torch.randn(3, 4)

x[None, :, :]     # shape: (1, 3, 4) — same as x.unsqueeze(0)
x[:, None, :]     # shape: (3, 1, 4) — same as x.unsqueeze(1)
x[:, :, None]     # shape: (3, 4, 1) — same as x.unsqueeze(2)
```

---

## Transposing and Permuting

### `transpose` — swap exactly two dimensions

```python
x = torch.randn(2, 3, 4)

x.transpose(0, 1)    # shape: (3, 2, 4)
x.transpose(1, 2)    # shape: (2, 4, 3)
```

### `.T` — shorthand for 2-D transpose

```python
m = torch.randn(3, 5)
m.T                  # shape: (5, 3)
```

### `permute` — reorder any number of dimensions

```python
# Convert from (B, H, W, C) to (B, C, H, W)
x = torch.randn(8, 32, 32, 3)      # BHWC
x.permute(0, 3, 1, 2)               # BCHW — shape: (8, 3, 32, 32)
```

---

## Concatenation and Stacking

### `cat` — join tensors along an existing dimension

```python
a = torch.randn(2, 3)
b = torch.randn(4, 3)

torch.cat([a, b], dim=0)    # shape: (6, 3) — stack vertically
```

All tensors must match in every dimension except the one being concatenated.

### `stack` — join tensors along a new dimension

```python
a = torch.randn(3, 4)
b = torch.randn(3, 4)

torch.stack([a, b], dim=0)   # shape: (2, 3, 4) — new dim 0
torch.stack([a, b], dim=1)   # shape: (3, 2, 4) — new dim 1
```

All tensors must have the same shape.

### `chunk` and `split` — inverse of cat

```python
x = torch.randn(6, 4)

torch.chunk(x, 3, dim=0)         # tuple of 3 tensors, each (2, 4)
torch.split(x, [1, 2, 3], dim=0) # tensors of shapes (1,4), (2,4), (3,4)
```

---

## Expanding and Repeating

### `expand` — broadcast a size-1 dim without copying data

```python
x = torch.tensor([[1], [2], [3]])  # shape: (3, 1)
x.expand(3, 4)                     # shape: (3, 4) — no memory copy
x.expand(-1, 4)                    # shape: (3, 4) — -1 means keep existing size
```

### `repeat` — tile the tensor (copies data)

```python
x = torch.tensor([1, 2, 3])       # shape: (3,)
x.repeat(2)                        # tensor([1, 2, 3, 1, 2, 3]) — shape: (6,)
x.repeat(2, 3)                     # shape: (2, 9) — 2 rows, each row is x tiled 3 times
```

Prefer `expand` over `repeat` when possible — it avoids unnecessary memory allocation.

---

## Contiguity

Some operations (`view`, certain in-place ops) require a tensor to be **contiguous** — meaning its elements are laid out sequentially in memory with no gaps or stride tricks.

```python
x = torch.randn(3, 4)
y = x.transpose(0, 1)    # y is NOT contiguous

y.is_contiguous()          # False
y.contiguous()             # returns a contiguous copy
y.reshape(12)              # works (reshape handles non-contiguous)
y.view(12)                 # ERROR — view requires contiguous input
```

Rule of thumb: use `reshape` unless you specifically want to guarantee a zero-copy view (then use `view` and handle the contiguity yourself).

---

## `einops`-Style Rearrange (Bonus)

The `einops` library provides a readable alternative for complex reshaping:

```python
from einops import rearrange

x = torch.randn(8, 3, 32, 32)

# Flatten spatial dims
rearrange(x, 'b c h w -> b (c h w)')       # (8, 3072)

# Patch embedding: split image into 4x4 patches
rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=4, p2=4)
```

`einops` is not part of PyTorch but is widely used in research code and makes reshape intentions explicit.

---

## Quick Reference

| Operation | Method | Example Result Shape |
|---|---|---|
| Reshape | `x.reshape(a, b)` | `(a, b)` |
| View (zero-copy reshape) | `x.view(a, b)` | `(a, b)` |
| Flatten | `x.flatten(start_dim)` | collapses dims |
| Add dim | `x.unsqueeze(d)` | inserts size-1 at `d` |
| Remove dim | `x.squeeze(d)` | removes size-1 at `d` |
| Swap 2 dims | `x.transpose(d1, d2)` | swaps `d1` and `d2` |
| Reorder all dims | `x.permute(d0, d1, ...)` | arbitrary order |
| Concatenate | `torch.cat([a, b], dim)` | joins along existing dim |
| Stack | `torch.stack([a, b], dim)` | joins along new dim |
| Broadcast expand | `x.expand(sizes)` | no copy |
| Tile repeat | `x.repeat(times)` | copies data |
| Slice along dim | `x.narrow(dim, start, len)` | sub-range of one dim |
| Pick index on dim | `x.select(dim, index)` | removes that dim |
| Gather indices | `torch.index_select(x, dim, idx)` | subset along dim |
| Gather by index tensor | `torch.gather(x, dim, idx)` | element-wise lookup |
| Scatter values | `x.scatter_(dim, idx, src)` | inverse of gather |
| Reverse along dims | `torch.flip(x, dims)` | reverses elements |
| Circular shift | `torch.roll(x, shifts, dims)` | wraps around |
| Remove dim to tuple | `torch.unbind(x, dim)` | tuple of slices |
| Move dims | `torch.movedim(x, src, dst)` | repositions dims |
