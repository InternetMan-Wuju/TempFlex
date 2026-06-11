---
name: Flex_attn_reorder_experience
description: Block Reorder 实现经验总结 — 踩过的坑、BlockMask 格式、NPU 编译器限制、正确做法
metadata:
  skill: true
  updated: 2026-06-11 (新增 2.4 外部重排性能实验 + 教训 2 和 6)
---

# Block Reorder 实现经验总结

记录在 NPU flex_attention 上实现 block reorder 过程中遇到的所有问题、根因分析、修复方法和关键教训。

## 1. BlockMask 内部格式

### 问题：`rebuild_block_mask` 重建的 dense mask 不正确

**现象**：用 `kv_indices` 重建的 causal mask 是 `[[1,0],[0,1]]`（对角），但正确 causal mask 应该是 `[[1,0],[1,1]]`（下三角）。

**根因**：PyTorch BlockMask 用两个结构存储稀疏块信息：

| 字段 | 含义 | 存储方式 |
|------|------|----------|
| `kv_indices` | PARTIALLY valid KV blocks | 单个 block 索引，前 `kv_num_blocks` 个有效 |
| `full_kv_indices` | FULLY valid KV blocks（块内所有 token 都满足 mask） | 单个 block 索引，**不是** [start,end) 对！ |
| `kv_num_blocks` | Partial block 数量 | — |
| `full_kv_num_blocks` | Full block 数量 | — |

**关键发现**：`full_kv_indices` 存储的是**单个 block 索引**，不是范围对。例如 Q block 1 的 causal mask 有 2 个 full blocks [0, 1]，则：
- `full_kv_num_blocks = 2`
- `full_kv_indices = [0, 1, pad, pad]`

**修复**：`rebuild_block_mask` 和 `compute_block_hit_rate` 必须同时处理 partial + full blocks：

```python
# 1) Partial blocks
n_partial = int(kv_num_blocks[b, h, row].item())
for i in range(n_partial):
    idx = int(kv_indices[b, h, row, i].item())
    mask[b, row, idx] = 1.0

# 2) Full blocks (individual indices, NOT ranges!)
n_full = int(full_kv_num_blocks[b, h, row].item())
for i in range(n_full):
    idx = int(full_kv_indices[b, h, row, i].item())
    mask[b, row, idx] = 1.0
```

### 如何验证 mask 是否正确

用 `create_block_mask` 直接构建，然后对比 `kv_indices` 和 `from_kv_blocks` 往返结果：

```bash
python3 -c "
import torch_npu._inductor
from torch.nn.attention.flex_attention import create_block_mask, BlockMask
bm = create_block_mask(lambda b,h,q,k: q>=k, 1, 1, 256, 256, device='cpu')
print('BLOCK_SIZE:', bm.BLOCK_SIZE)
print('kv_num_blocks:', bm.kv_num_blocks)
print('kv_indices:', bm.kv_indices)
print('full_kv_num_blocks:', bm.full_kv_num_blocks)
print('full_kv_indices:', bm.full_kv_indices)
"
```

## 2. 外部 Q Reorder 为什么行不通（正确性 + 性能双重失败）

### 2.1 正确性问题：非 identity permutation 导致输出错误

**现象**：sliding_window_64 @ S=512 产生 perm = [0, 3, 1, 2]（非 identity），reorder 后输出与 manual 差异巨大（max_diff=3.6，fail_ratio=69%）。

**根因**：重排 Q 后，`score_mod` 中的 `token_q` 是重排后的物理位置，不再是原始 Q token 位置。例如：
- 重排后位置 128 的 token 实际是原始位置 384 的 token
- `score_mod(score, b, h, 128, kv_idx)` 计算 `128 - kv_idx <= 63`
- 正确的应该是 `384 - kv_idx <= 63`

`score_mod` 使用物理 token 位置，不会自动感知 Q 重排。

### 问题：transposed score_mod 导致 bishengir segfault

**尝试方案**：用 `torch.where` 链编写 transposed score_mod：

```python
def transposed_score_mod(score, b, h, q_idx, kv_idx):
    in_block_i = (q_idx >= i*B) & (q_idx < (i+1)*B)
    orig_q = torch.where(in_block_i, inv[i]*B + (q_idx - i*B), orig_q)
    # ...
```

**结果**：bishengir 编译器 segfault。

**根因**：`torch.where` 链产生过大的 MLIR 子图，bishengir 编译器无法处理。

### 问题：用 `//` 和 `%` 编码映射也不行

**尝试方案**：
```python
block_idx = q_idx // block_size
offset = q_idx % block_size
orig_q = inv[block_idx] * block_size + offset  # 需要 tensor 索引
```

**根因**：
- `//` 在 mask_mod 子图中可能触发 `bool_to_bool_rintmode`（已知 bishengir bug）
- `%` 同理，生成 `aten.remainder.Scalar`，bishengir 无法 lowering
- `inv[block_idx]` 生成 `aten.index.Tensor`，NPU 不支持 tensor 下标索引

### 2.4 性能问题：外部重排即使跳过正确性也无法改变 kernel 耗时

**实验设计**：用 causal mask @ S=2048（确保能编译），对比两个版本：
- Identity order：Q 和 mask 保持原始顺序
- Reverse order：Q 和 mask 完全反序（perm = [n-1, n-2, ..., 0]）

两个版本的 Q 和 mask 内容完全相同，只是排列顺序不同。输出都是错的（score_mod 不感知重排），但只测 kernel 耗时。

**实验结果**（2026-06-11）：

```
Identity order (原始):  9.047 ms
Reverse order  (反序):  9.100 ms
差异: +0.054 ms (+0.6%) — 噪声级别
```

**为什么外部重排改变不了 kernel 耗时？**

NPU flex_attention kernel 的架构：

```
program[0] → load Q block 0 → load KV blocks of block 0 → compute → write output[0]
program[1] → load Q block 1 → load KV blocks of block 1 → compute → write output[1]
...
program[N] → load Q block N → load KV blocks of block N → compute → write output[N]
```

每个 program 是**完全独立**的：
- 各自从 global memory 加载自己需要的 KV blocks（通过 `kv_indices` 确定）
- 没有跨 program 的 L2 cache 复用机制
- 重排 program 启动顺序 ≠ 改变 KV block 加载顺序 ≠ 改变 cache 行为

**与 GPU vllm-omni 的关键区别**：

| | GPU vllm-omni | NPU flex_attention（外部重排） |
|---|---|---|
| 重排位置 | **Kernel 内部**，一个 thread block 内重排 Q block 迭代 | **Kernel 外部**，重排 Q tensor 和 mask |
| Cache 效果 | 连续 Q blocks 处理相邻 KV blocks → L2 cache 命中 | 每个 program 独立加载 K/V → 无跨 program cache |
| 正确性 | score_mod 看到正确 token_q（Q 不重排） | score_mod 看到错误 token_q（Q 已重排） |

**结论**：

> 外部 Q+mask 重排**既做不了正确性**（2.1-2.3），**也做不了性能**（2.4）。NPU 上 block reorder 的唯一正确路径是 **kernel 内部实现**。

## 3. Kernel 内部 Reorder 的正确方案（✅ 已实现 2026-06-11）

### 实现方式

通过 **side-channel** 将运行时 PERM tensor 传递给 inductor lowering，避免修改 PyTorch flex_attention 接口。

```
eager code (测试脚本)
  │
  ├─ compute_and_set_pending_perm(kv_indices, ...)  ← 计算 mask-level perm
  │   └─ set_pending_perm(perm)                      ← 存入模块级全局变量
  │
  └─ flex_attention(q, k, v, block_mask=bm)  ← torch.compile 追踪
      │
      └─ lowering 函数
          ├─ _get_and_clear_perm()                   ← 从全局变量取出 perm
          ├─ V.graph.add_tensor_constant(perm)       ← 转为 inductor 常量 buffer
          └─ inputs_for_autotuning = [..., perm]     ← 传给 kernel
```

### Kernel 修改

在 `flex_attention.py` 的 Triton 模板中：

```python
# 加载 mask-level perm，展开为 triton-level
if ENABLE_REORDER:
    src_mask_block = tl.load(PERM + (q_start // SPARSE_Q_MULTIPLE))
    src_block = src_mask_block * SPARSE_Q_MULTIPLE + (q_start % SPARSE_Q_MULTIPLE)
else:
    src_block = q_start

# src_block 用于：Q 加载、mask 索引、输出写入
offs_m = src_block * BLOCK_M + tl.arange(0, BLOCK_M)
sparse_kv_num_blks_offset = ... + src_block // SPARSE_Q_MULTIPLE
Q_block_ptr offsets=(src_block * BLOCK_M, 0)
```

### 正确性验证结果

| 测试 | perm | flex_reorder vs manual |
|------|------|----------------------|
| causal @ (1,4,512,64) | identity → 跳过 | ✅ allclose |
| sliding_window_64 @ (1,4,512,64) | [0, 3, 1, 2] | ✅ allclose |
| sliding_window_128 @ (2,8,1024,64) | [0,1,2,3,4,7,5,6] | ✅ allclose |
| causal @ (4,8,2048,128) | [15,14,...,0] (完全反序) | ✅ allclose |

### 为什么正确
1. Q tensor **不重排** → score_mod 看到正确的 token_q
2. Q 从 `src_block * BLOCK_M` 加载 → kernel 按 perm 顺序处理
3. 输出写入 `src_block * BLOCK_M` → 自动在原位，无需 unpermute
4. Mask 从原始 kv_indices 加载（kv_indices 不重排）→ 每个 Q block 获得自己的 mask

## 3b. Kernel 内部 Reorder 的旧方案（仅作记录）

### 需要修改的位置

文件：`flex_attention.py` 的 Triton kernel 模板。

核心思路：修改 Q block 迭代顺序，不修改 Q tensor 本身。

```
当前：
  for start_m in range(0, Q_LEN, BLOCK_M):     # 顺序 0, BLOCK_M, 2*BLOCK_M, ...
      for start_n in block_range(start_m):
          forward_inner(...)

改为：
  for perm_idx in range(n_blocks_q):
      start_m = perm[perm_idx] * BLOCK_M           # 按 perm 顺序
      for start_n in block_range(start_m):
          forward_inner(...)
```

### 关键点

1. **Q block 按 perm 顺序处理**，Q tensor 本身不重排
2. **kv_indices 行按 perm 重排**（`reordered_kv_indices[row] = original_kv_indices[perm[row]]`）
3. **输出按 inv_perm 写回**：`output[inv[row] * BLOCK_M : (inv[row]+1) * BLOCK_M] = computed[row]`
4. **score_mod 不需要修改**：因为 Q tensor 没动，`token_q` 依然是原始位置

### 参考实现

`sparse-attn-source/new-vllm-omni-for-sparse-attn-main/vllm_omni/diffusion/attention/backends/`
- `greedy_reorder_cuda.py` — CUDA reorder 算法（global_greedy, banded_greedy, optimal_reorder_cuda）
- `reorder_rows_graph.py` — CPU numba 算法（greedy nearest-neighbor）

vllm-omni 的 reorder 是在 GPU kernel 内部实现的，不是外部重排。

## 4. 已完成的 Reorder 模块

`flex_attention_reorder.py` 目前提供的功能：

| 函数 | 状态 | 说明 |
|------|------|------|
| `wave_overlap_reorder` | ✅ | 谱重排算法（power iteration），CPU 运行 |
| `compute_block_hit_rate` | ✅ | 正确计算 cache 命中率（含 full_kv） |
| `rebuild_block_mask` | ✅ | 正确重建 dense mask（含 full_kv） |
| `reorder_flex_forward` | ⚠️ | 外部重排流程，仅 identity perm 正确 |
| `unpermute_output` | ✅ | 输出逆重排 |
| `make_reordered_score_mod` | ❌ | score_mod 转置，bishengir segfault，不可用 |

## 5. NPU 编译器限制速查

| 操作 | 状态 | 说明 |
|------|------|------|
| `>=`, `<=`, `==`, `!=`, `>`, `<` | ✅ | 基本比较 |
| `+`, `-`, `*` | ✅ | 基本算术 |
| `&`, `\|` | ✅ | 布尔运算（`\|` 需 `bool+bool→int` 替代） |
| `torch.where` | ⚠️ | 简单使用 OK，链式使用导致 segfault |
| `//` (floor divide) | ⚠️ | 部分情况 OK，可能触发 `bool_to_bool_rintmode` |
| `%` (modulo) | ❌ | 生成 `aten.remainder.Scalar`，无法 lowering |
| `aten.index.Tensor` | ❌ | Tensor 下标索引，NPU 不支持 |
| `.abs()` | ⚠️ | 需替代 `(diff<=n)&(-diff<=n)` |

## 6. 调试技巧

### 验证 BlockMask 往返一致性

```python
bm_original = create_block_mask(mask_mod, 1, 1, S, S, device="cpu")
bm_rt = BlockMask.from_kv_blocks(
    kv_num_blocks=bm_original.kv_num_blocks,
    kv_indices=bm_original.kv_indices,
    full_kv_num_blocks=bm_original.full_kv_num_blocks,
    full_kv_indices=bm_original.full_kv_indices,
    BLOCK_SIZE=bm_original.BLOCK_SIZE,
    mask_mod=bm_original.mask_mod,
)
# bm_rt 的 flex_attention 输出应与 bm_original 完全一致
```

### 检查 permutation 是否为 identity

```python
is_identity = torch.equal(perm[0], torch.arange(perm.shape[1]))
```

### 验证 reorder 正确性

```bash
# identity permutation 的情况（如 causal）
python3 flex_attention_run_script.py --shape 1,4,512,64 --enable-block-reorder

# 应该看到：
# ✅ 测试通过（allclose=True）
# ✅ reorder 测试通过（allclose=True）
```

### 排查 segfault

```bash
# 清除缓存后重试，判断是 cache 问题还是代码问题
rm -rf /tmp/torchinductor_root /root/.triton/cache
# 再运行测试
```

## 7. 关键教训

1. **BlockMask 格式**：`full_kv_indices` 存的是单个 block 索引，不是 [start,end) 对。这是踩坑最多的点。

2. **外部 reorder 双重重失败**（正确性 + 性能）：
   - **正确性**：NPU 的 mask_mod/score_mod 子图无法处理 `//`、`%`、`aten.index.Tensor` 和 torch.where 链。外部重排后修正 score_mod 的方案注定失败。
   - **性能**：即使跳过正确性只测时间，外部重排对 kernel 耗时几乎无影响（<1%）。每个 Q block 作为独立 program 处理，没有跨 program 的 KV cache 复用。外部重排 program 顺序 ≠ 改变 cache 行为。

3. **Kernel 内部实现是唯一正道**：参考 sparse-attn-source 的 vllm-omni，reorder 应该在 kernel 的 Q block 迭代循环中实现，Q tensor 保持原位。参见 [[Flex_attn_opt]] 的核心要求章节。

4. **先验证 identity perm**：如果 perm 不是 identity 且无法修正 score_mod，reorder 一定错误。identity perm 时 reorder = no-op，可以作为正确性基线。

5. **bishengir 编译器很脆弱**：任何稍复杂的操作（`%`、`//`、`aten.index.Tensor`、torch.where 链）都可能触发编译器 crash 或 segfault。简单的 `torch.where` 可用，但链式使用会导致 UB overflow 或 segfault。设计算法时必须避开这些操作。

6. **区分 GPU 和 NPU 的架构差异**：GPU 上 vllm-omni 的 kernel 内部 reorder 有效（同一 thread block 内连续 Q blocks 共享 L2 cache）。NPU 的 program 模型不同，外部重排没有对应的 cache 收益。不要直接移植，必须理解 NPU 的 program 调度和 cache 层次。
