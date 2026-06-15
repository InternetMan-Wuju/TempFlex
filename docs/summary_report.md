# Flex Attention NPU 优化报告
> 日期: 2026-06-15 08:53 | 测试: B vs C reorder ablation | warmup=10 repeat=10 (MAD outlier rejection)

## 1. Reorder 贡献分解

加速来源分为两部分：
- **(a) PURE_BLOCK_SPARSE 模板**：跳过无效 KV blocks，通过 FULL_KV 元数据路径避免 kernel subgraph
- **(b) wave_overlap reorder**：重排 Q block 处理顺序，提升 KV cache 命中率

测试方法：
- **Path B** = PURE_BLOCK_SPARSE (无 reorder) → 隔离贡献 (a)
- **Path C** = PURE_BLOCK_SPARSE + wave_overlap reorder → 隔离贡献 (a)+(b)
- **B/C** = reorder 单独加速比（贡献 (b)）

## 2. 多序列长度 B/C Reorder 消融实验

### 2.1 S=32768 (B=2 H=4, 256x256 blocks)

| Pattern | t_B (ms) | t_C (ms) | B/C | Δ% | Notes |
|---------|:--------:|:--------:|:---:|:---:|-------|
| causal | 179.44 | 179.17 | 1.0015x | +0.15% | identity (no reorder) |
| sliding_window | 3.84 | 3.86 | 0.9946x | -0.54% |  |
| block_diagonal | 2.47 | 2.45 | 1.0100x | +1.00% |  |
| nested | 48.82 | 48.88 | 0.9988x | -0.12% |  |
| strided | 90.50 | 90.83 | 0.9964x | -0.36% |  |
| dilated_window | 5.27 | 5.24 | 1.0048x | +0.48% |  |
| checkerboard | 178.45 | 178.21 | 1.0013x | +0.13% |  |
| prefix_lm | 179.32 | 179.22 | 1.0006x | +0.06% | identity (no reorder) |
| hybrid_sparse | 65.79 | 65.68 | 1.0016x | +0.16% |  |
| random_block_sparse | 55.48 | 55.47 | 1.0001x | +0.01% |  |

**分析**: reorder 增益范围 0.9946x–1.0100x，均值 1.0010x。

### 2.2 S=49152 (B=2 H=4, 384x384 blocks)

| Pattern | t_B (ms) | t_C (ms) | B/C | Δ% | Notes |
|---------|:--------:|:--------:|:---:|:---:|-------|
| causal | 401.45 | 401.23 | 1.0005x | +0.05% | identity (no reorder) |
| sliding_window | 5.64 | 5.63 | 1.0009x | +0.09% |  |
| block_diagonal | 3.57 | 3.56 | 1.0050x | +0.50% |  |
| nested | 106.33 | 106.45 | 0.9989x | -0.11% |  |
| strided | 201.96 | 202.50 | 0.9973x | -0.27% |  |
| dilated_window | 7.71 | 7.71 | 0.9992x | -0.08% |  |
| checkerboard | 401.35 | 400.64 | 1.0018x | +0.18% |  |
| prefix_lm | 401.36 | 401.45 | 0.9998x | -0.02% | identity (no reorder) |
| hybrid_sparse | 144.03 | 144.09 | 0.9996x | -0.04% |  |
| random_block_sparse | 122.93 | 122.93 | 1.0000x | +0.00% |  |

**分析**: reorder 增益范围 0.9973x–1.0050x，均值 1.0003x。

### 2.3 S=65536 (B=2 H=4, 512x512 blocks)

| Pattern | t_B (ms) | t_C (ms) | B/C | Δ% | Notes |
|---------|:--------:|:--------:|:---:|:---:|-------|
| causal | 711.06 | 711.10 | 0.9999x | -0.01% | identity (no reorder) |
| sliding_window | 7.45 | 7.43 | 1.0020x | +0.20% |  |
| block_diagonal | 4.66 | 4.65 | 1.0005x | +0.05% |  |
| nested | 185.73 | 186.51 | 0.9958x | -0.42% |  |
| strided | 357.49 | 358.17 | 0.9981x | -0.19% |  |
| dilated_window | 10.21 | 10.23 | 0.9985x | -0.15% |  |
| checkerboard | 709.18 | 712.76 | 0.9950x | -0.50% |  |
| prefix_lm | 711.21 | 710.72 | 1.0007x | +0.07% | identity (no reorder) |
| hybrid_sparse | 252.54 | 252.67 | 0.9995x | -0.05% |  |
| random_block_sparse | 218.06 | 218.26 | 0.9991x | -0.09% |  |

**分析**: reorder 增益范围 0.9950x–1.0020x，均值 0.9986x。

## 3. 代码与参考实现对比

当前 NPU `wave_overlap_reorder` (flex_attention_reorder.py:177) 与参考实现 (greedy_reorder_cuda.py:2441) 的差异：

| 步骤 | 参考 (vllm-omni) | 修复前 NPU | 修复后 NPU |
|------|:---:|:---:|:---:|
| Power iteration (10 iter) | ✅ | ✅ | ✅ |
| Spectral sort (descending) | ✅ | ❌ ascending | ✅ descending |
| Wave partition (132) | ✅ | ❌ | ✅ |
| Intra-wave NNZ sort | ✅ | ❌ | ✅ |
| Inter-wave NNZ scheduling | ✅ | ❌ | ✅ |
| Padding/strip | ✅ | ❌ | ✅ |

**修复后 B/C 结果（S=32K, warmup=10, repeat=15, MAD）**:

| Pattern | B/C | Δ% |
|---------|:---:|:---:|
| nested | 1.0037x | +0.37% |
| strided | 1.0030x | +0.30% |
| checkerboard | 1.0013x | +0.13% |
| hybrid_sparse | 1.0017x | +0.17% |
| sliding_window | 1.0065x | +0.65% |
| block_diagonal | 0.9994x | -0.06% |

修复后补全了 wave scheduling 管线，B/C 仍然在 ±1% 噪音内。

**原因**：参考中的 wave scheduling（分区 → 波内 NNZ 排序 → 波间调度）设计目的是 GPU SM 级别的 warp divergence 均衡和 latency hiding。NPU 的硬件线程调度机制不同，这些步骤不产生实际加速。Power iteration + spectral sort 本身（修复前后相同的部分）对规则稀疏 pattern 的 KV cache 局部性提升极其有限——因为规则 pattern 本身的 cache 命中率已经很高。

## 4. 结论

1. **Reorder 贡献在 NPU 上可忽略**：所有实测 pattern 在所有序列长度上 B/C 均在 0.99x–1.02x 范围内（测量噪音级别）
2. **加速完全来自 PURE_BLOCK_SPARSE 模板**：跳过无效 KV blocks 是唯一有效优化
3. **与 GPU 结论一致**：GPU 上 reorder 到 80K+ 才有 ~1% 增益；NPU 上在 65K 以下无意义
4. **prefix_lm 恒为 identity**：几乎满的 causal 三角 mask，wave_overlap 无法找到非平凡重排

## 5. 如何让 bishengir 支持 12 种稀疏模式：Subgraph → Metadata

### 问题背景

原始实现中，稀疏 pattern 通过 `mask_mod(b, h, q_idx, kv_idx)` 函数定义，该函数会在 Triton kernel 内部被编译为 MLIR subgraph，由 bishengir 编译器处理。但 bishengir 对以下操作存在严重限制：

| 操作 | bishengir 行为 | 受影响的模式 |
|------|:---:|------|
| `//` (floor divide) | `bool_to_bool_rintmode` crash | nested, strided, checkerboard, dilated, hybrid, block_diagonal, uniform_doc, multiscale (共 8 个) |
| `%` (modulo) | `aten.remainder.Scalar` 无法 lowering | 同上 |
| `torch.where` 链 | segfault 或 UB overflow | global_local, band_global |
| `aten.index.Tensor` | NPU 不支持 tensor 下标索引 | random_block_sparse, alibi_causal |

结果：18 种模式中只有 4 种能编译通过（causal, sliding_window_64/128, prefix_lm, band_global_32），其余全部挂掉。

### 解决方案：三步绕开 bishengir

核心思路：**把 subgraph 问题变成 metadata 问题**。bishengir 编译不了的复杂逻辑，全部在 host 侧 Python 算好，kernel 只需要机械地按照预先算好的 block 列表加载 KV。

#### Step 1: `pattern_to_block_mask.py` — Host 侧构建 block mask

12 种 pattern 的 mask 构建逻辑全部用纯 Python 实现，直接生成 `[1, 1, MQ, NK]` 的 bool tensor：

```python
# 以 sliding_window 为例（pattern_to_block_mask.py）
def _build_sliding_window(MQ, NK, window_blocks, causal, device):
    i = torch.arange(MQ, device=device).unsqueeze(1)  # [MQ, 1]
    j = torch.arange(NK, device=device).unsqueeze(0)   # [1, NK]
    mask = (i - j).abs() < window_blocks
    if causal:
        mask = mask & (i >= j)
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, MQ, NK]
```

完全不经过 Triton → MLIR → bishengir 编译链路，不触发任何编译器 bug。

#### Step 2: `block_mask_to_full_kv()` — 转为 FULL_KV 元数据

```python
full_kv_num_blks = mask.sum(dim=-1).to(torch.int32)         # 每行有效 block 数
full_kv_idx = torch.argsort(~mask, dim=-1, stable=True)      # 有效 block 索引（排序）
# partial metadata 全空 — 所有 block 走 FULL_KV 路径
kv_num_blks = zeros(...)
kv_idx = zeros(...)
```

两种 metadata 的区别：
- **Partial blocks** (`kv_indices`): 块内部分 token 有效，kernel 需要执行 `mask_mod` subgraph 做 per-token 过滤 → 触发 bishengir bug
- **Full blocks** (`full_kv_indices`): 块内所有 token 都有效，kernel 直接做完整 attention，无需 mask_mod → 完全安全

关键 trick：**所有 block 都走 FULL_KV 路径，partial metadata 全置零**。这样 kernel 根本不会调用 mask_mod subgraph。

#### Step 3: `PURE_BLOCK_SPARSE` 模板 — 零 subgraph kernel

```python
# 标准模板中，这两个 subgraph 是 bishengir crash 的根源：
#   score_mod(score, b, h, m, n) → post_mod_scores
#   mask_mod(b, h, m, n)         → mask_mod_output

# PURE_BLOCK_SPARSE 模板直接删除它们：
post_mod_scores = qk  # identity score, 无 subgraph
# 无 mask_mod，attention pattern 完全由 kv_indices/full_kv_indices 决定
```

kernel 内部的循环变成纯粹的 block 迭代：
```
for each Q block:
    for each valid KV block (from full_kv_indices):
        load K_block, V_block
        compute QK^T + softmax + PV
        write output
```

零 subgraph，零 bishengir 风险。

### 效果

| 指标 | 修复前 | 修复后 |
|------|:---:|:---:|
| 可编译模式 | 4/18 | 12/12 (FULL_KV 路径) |
| 正确性 (S=1024/8192) | — | 0% fail rate |
| 编译 crash | bool_to_bool, UB overflow, segfault, aten.index | 无 |
| block_diagonal @ S=8192 | ❌ 编译失败 | **23x 加速** vs dense |
| sliding_window @ S=8192 | ❌ 编译失败 | **14.7x 加速** vs dense |

### 具体例子：sliding_window (window=2 blocks) @ S=1024

下面用一个具体例子串联三步，同时解释为什么 **FULL_KV ≠ Dense**。

设定：S=1024, block_size=128 → 8 Q blocks × 8 KV blocks。sliding_window, window_blocks=2, causal=True。

**Step 1: pattern_to_block_mask.py 输出的 block mask**

```
         kv block:   0    1    2    3    4    5    6    7
   q block 0:      [F]  [F]  [ ]  [ ]  [ ]  [ ]  [ ]  [ ]    ← 2 个有效 block
   q block 1:      [F]  [F]  [F]  [ ]  [ ]  [ ]  [ ]  [ ]    ← 3 个有效 block
   q block 2:      [ ]  [F]  [F]  [F]  [ ]  [ ]  [ ]  [ ]
   q block 3:      [ ]  [ ]  [F]  [F]  [F]  [ ]  [ ]  [ ]
   q block 4:      [ ]  [ ]  [ ]  [F]  [F]  [F]  [ ]  [ ]
   q block 5:      [ ]  [ ]  [ ]  [ ]  [F]  [F]  [F]  [ ]
   q block 6:      [ ]  [ ]  [ ]  [ ]  [ ]  [F]  [F]  [F]
   q block 7:      [ ]  [ ]  [ ]  [ ]  [ ]  [ ]  [F]  [F]
```

F = 整个 block 有效（128×128 内所有 token pair 都满足 pattern），[ ] = 完全无效。

这是 **block 级别的** pattern，关键性质：**每个 block 要么全有效，要么全无效，没有"部分有效"的 block**。这是因为 pattern 是在 block 粒度上定义的（`|i-j| < 2` 是 block 索引比较），不是 token 粒度。

**Step 2: block_mask_to_full_kv() 的转换结果**

以 Q block 1 为例（mask 行 = `[F, F, F,  ,  ,  ,  , ]`）：

```
full_kv_num_blocks[0,0,1] = 3            ← 这一行 3 个有效 KV block
full_kv_idx[0,0,1]       = [0, 1, 2, 填充]  ← 它们的索引

# Partial metadata 全部置零 → kernel 只走 FULL_KV 循环
kv_num_blocks[0,0,1]     = 0
kv_idx[0,0,1]            = [0, 0, 0, 0]
```

**Step 3: kernel 实际执行**

kernel 里两个循环：

```python
# ── Partial blocks 循环 ── (kv_num_blocks=0, 跳过，不执行)
for ...  # 0 次迭代

# ── FULL_KV blocks 循环 ── (full_kv_num_blocks=3)
for kv_block in [0, 1, 2]:   # 只遍历 3 个 block
    load K block kv_block, V block kv_block
    qk = Q @ K^T
    if causal: qk = tl.where(offs_m >= offs_n, qk, -inf)  # 内置因果检查
    softmax + PV
```

Q block 1 只加载 3 个 KV blocks，跳过 5 个。对 S=8192（64 blocks），每行也是 3 个有效 → 3/64 ≈ 4.7% 的计算量 → ~20x 加速。

**为什么 FULL_KV ≠ Dense**

| | Dense | FULL_KV (PURE_BLOCK_SPARSE) |
|---|---|---|
| Q block 1 处理的 KV blocks | 8/8 (100%) | 3/8 (37.5%) |
| S=8192 时 KV blocks | 64/64 (100%) | 3/64 (~5%) |
| mask_mod subgraph | `q_idx >= kv_idx` 作为 MLIR 子图 | `offs_m >= offs_n` 寄存器比较 |
| bishengir 参与 | ✅ 需要 | ❌ 不需要 |

FULL_KV 的意思是"所有有效 block 都通过 FULL_KV 元数据路径传递"，不是说"把所有 block 都标记为 full"。kernel 遍历的 block 数量由每行的 `full_kv_num_blocks` 决定，稀疏 pattern 下这个数字远小于总 block 数。

**causal check 的特殊处理**

PURE_BLOCK_SPARSE 模板不调用 `mask_mod` subgraph，causal 的 `q_idx >= kv_idx` 检查用内核内置的 `offs_m >= offs_n` 替代：

```python
# ❌ 标准模板（触发 bishengir，即使 causal 也会生成 subgraph）
mask_mod_output = mask_mod(b, h, offs_m, offs_n)  # 函数调用 → MLIR subgraph

# ✅ PURE_BLOCK_SPARSE（Triton 基本比较，直接生成 MLIR compare 指令）
if PURE_BLOCK_SPARSE_CAUSAL:
    mask_mod_output = offs_m >= offs_n  # 寄存器级比较，无 subgraph
```

`offs_m >= offs_n` 在 MLIR 层面就是一条 `arith.cmpi` 指令，bishengir 处理它毫无压力。这就是为什么把 `//`、`%`、`torch.where` 链换成 block-level metadata 之后，所有 12 种 pattern 都能编译通过的深层原因。

### 取舍

- **粒度损失**：从 token-level mask 降级为 block-level mask（block_size=128）。对于 block-level 定义的 attention pattern 这完全等价，但对于 token-level 的 pattern（如精确的 token-level sliding_window）会引入最多 1 个 block 的边界误差。
- **灵活性损失**：无法在 kernel 内部动态计算 mask（如 data-dependent sparsity），但对于静态稀疏 pattern 完全够用。

## 6. 测试配置

- 设备: NPU Ascend (60GB)
- 精度: bfloat16
- 编译: torch.compile(backend='inductor', dynamic=False)
- 计时: warmup=10, repeat=10, MAD outlier rejection (threshold=3.0)
- 隔离: 每次测试独立 subprocess 避免 compile cache 污染
- Block size: SPARSE_Q_BLOCK_SIZE=128, SPARSE_KV_BLOCK_SIZE=128
