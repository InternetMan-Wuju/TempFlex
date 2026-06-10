---
name: flex_attn_perf_gaps
description: NPU flex_attention 已知性能漏洞 — 现象、根因、代码定位、修复伪代码
metadata:
  skill: true
  updated: 2026-06-10
---

# NPU Flex Attention 已知性能漏洞

## 漏洞 1：非因果稀疏模式随序列长度性能急剧恶化

### 现象

| Pattern | S=128 | S=512 | S=1024 | S=2048 |
|---------|-------|-------|--------|--------|
| `sliding_window_64` | 1.78x | 2.70x | 6.15x | 7.54x |
| `sliding_window_128` | 1.58x | 2.64x | 4.90x | 6.91x |
| `prefix_lm` | 1.96x | 1.53x | 2.80x | 5.23x |
| `band_global_32` | 2.13x | 6.99x | 13.4x | 16.0x |

- flex/manual 比率随 S 增长而**非线性恶化**
- `band_global_32` 在 S=2048 时慢 **16 倍**

### 根因：通用 block-sparse 模板无法跳过 masked-out blocks

当前 NPU flex_attention 内核无论 block 稀疏度如何，都对**每个 block 执行完整 attention 计算**（QK dot + softmax + PV dot）。CPU/CUDA 上的 block-sparse 模板可以通过 `kv_indices` 跳过完全 masked 的 blocks，但 NPU Triton adapter 生成的代码做不到这一点——每个 block 都变成一个小 matmul kernel launch。

```
典型 causal 场景（BLOCKS_ARE_CONTIGUOUS=True）：
  内核只需 BLOCK_N 步进，简洁高效

非因果稀疏场景（BLOCKS_ARE_CONTIGUOUS=False）：
  内核遍历所有 S/BLOCK_N 个 block，每个 block 独立 launch → kernel launch 数量爆炸
  block 越多 → launch 越多 → 性能急剧下降
```

### 关键代码位置

**`forward_inner` 循环** (`flex_attention.py:~381`):
```python
for start_n in range(block_n_start, block_n_end):
    # 每个 block 都做完整 attention
    acc, l_i, m_i = forward_block_mn(...)

    offset = get_offset_for_next_block(start_n, kv_indices, ...)
    V_block_ptr = tl.advance(V_block_ptr, (offset, 0))
    K_block_ptr = tl.advance(K_block_ptr, (0, offset))
    offs_n = offs_n + offset
```

**问题**: `forward_block_mn` 对 `BLOCK_M × BLOCK_N` 做完整 QK dot + softmax + PV dot，即使该 block 内 90% 的元素被 mask。没有 early skip 机制。

### 修复方向：Block Reorder + 稀疏感知调度

```
伪代码 — block reorder 优化：

1. 分析阶段（CPU，offline 或编译期）:
   for each query block qb:
       统计有效的 kv blocks 数量 effective_kv_blocks[qb]
       按 effective_kv_blocks 降序排列 → 重排后的 query block 顺序

2. 内核阶段（NPU）:
   for each query block qb in reordered_blocks:
       kv_blocks = kv_indices[qb, 0:kv_num_blocks[qb]]
       for each kv_block in kv_blocks:
           # 只对有效的 kv block 做 attention
           if is_fully_masked(kv_block):
               continue  # ← 核心优化：跳过全 mask block
           compute_attention_block(qb, kv_block)
```

### 预期收益

- `band_global_32` @ S=2048: 期望从 16x 降到 3-4x（减少大量无效 block 计算）
- `sliding_window` 系列: 期望从 5-7x 降到 1.5-2x
- `prefix_lm`: 期望降到接近 1.5x

---

## 漏洞 2：bishengir 编译器 UB overflow（复杂 mask_mod）

### 现象

`global_local` 模式编译失败：
```
ub overflow, requires 1724416 bits while 1572864 bits available!
```

### 根因

mask_mod subgraph 超过 10 个操作时，生成的 bishengir IR 需要超过 NPU UB 空间限制（~1.5MB）。每个 `.int()` 调用（bool→int 转换）在 subgraph 中分配临时 buffer，3 个 `.int()` 调用就会超出预算。

### 关键代码位置

`sparse_masks.py` 中的复杂 mask_mod 函数。

### 修复方向

- 将 mask_mod 的 OR 逻辑从 3 条件压缩到 2 条件（如 `torch.minimum(q,k) < G` 替代 `q<G OR k<G`）
- 或：将 mask_mod 的计算移到 Python 预处理阶段，在 `create_block_mask` 时就完成
- 或：bishengir 层面优化 UB 分配策略

---

## 漏洞 3：bishengir `bool_to_bool_rintmode`（整数除法 //）

### 现象

8 个模式编译失败：
```
'hivm.hir.vcast' op currently don't support cast bool_to_bool_rintmode
```

### 根因

`token_q // BLOCK_SIZE` 在 Triton→MLIR lowering 中生成 `arith.divsi`，后续在 bishengir 中触发了不支持的 `bool_to_bool` cast。这是 bishengir 编译器对整数除法的支持缺陷。

### 关键代码位置

任何使用 `//` 的 mask_mod 函数，如：
```python
# sparse_masks.py
q_block = token_q // block_size    # ← 触发 bool_to_bool_rintmode
kv_block = token_kv // block_size
```

### 修复方向

- bishengir 层面添加对 `arith.divsi` → `bool_to_bool` cast 路径的支持
- 或：预计算 block 索引作为额外输入传入 kernel，避免在 subgraph 中做除法

---

## 漏洞 4：`aten.index.Tensor` 不支持（动态 tensor 索引）

### 现象

`random_block_sparse` 和 `alibi_causal` 编译失败：
```
Buffers cannot be created while lowering a pointwise subgraph.
While executing %index: aten.index.Tensor
```

### 根因

NPU inductor 的 pointwise subgraph lowering 不支持 `aten.index.Tensor`（用 tensor 下标索引另一个 tensor）。这在 `random_block_sparse`（`block_mask[q_block, kv_block]`）和 `alibi_causal`（`slopes[head]`）中都会触发。

### 修复方向

- inductor 层面添加对 `aten.index.Tensor` 的 pointwise lowering 支持
- 或：对于 `alibi_causal`，将 slopes 作为 kernel 参数传入而非在 subgraph 中索引
- 或：对于 `random_block_sparse`，将 block_mask 展平并传入为额外 buffer

---

## 当前整体状态

```
                编译通过    性能可接受
                 ├─ ✅ ──── ├─ ✅ sliding_window_64/128 (S≤512)
                 │          └─ ⚠️ sliding_window_64/128 (S>512, 5-8x)
                 │
                 │          └─ ⚠️ prefix_lm (1.5-5x)
                 │          └─ ❌ band_global_32 (2-16x)
                 │
非因果模式 ──────┤
                 ├─ ❌ UB overflow ──── global_local
                 ├─ ❌ bool_to_bool ─── nested, strided, checkerboard,
                 │                      dilated, hybrid, block_diagonal,
                 │                      uniform_doc, multiscale
                 └─ ❌ aten.index ───── random_block_sparse, alibi_causal
```

### 优先级建议

1. **P0**: Block Reorder（解决漏洞 1）— 这是最大的性能收益来源
2. **P1**: bishengir `bool_to_bool` 修复（解决漏洞 3）— 解锁 8 个模式
3. **P2**: bishengir UB overflow 修复（解决漏洞 2）— 解锁 `global_local`
4. **P3**: inductor `aten.index` 支持（解决漏洞 4）— 解锁 2 个模式
