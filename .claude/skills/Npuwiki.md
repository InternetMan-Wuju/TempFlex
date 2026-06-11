---
name: Npuwiki
description: NPU 知识库 — Ascend 硬件、bishengir 编译器、Triton ascend backend、torch_npu 的架构知识和调试技巧
metadata:
  skill: true
  updated: 2026-06-11
---

# NPU 知识库

持续积累的 NPU（Ascend）相关知识，随着项目推进不断补充。

## 1. bishengir 编译器限制与 Workaround

### 1.1 bool_to_bool_rintmode（8 个模式受影响）

**现象**：
```
'hivm.hir.vcast' op currently don't support cast bool_to_bool_rintmode
```

**根因**：整数 `//`（floor divide）在 Triton→MLIR lowering 中生成 `arith.floordivsi`，bishengir 对该操作生成的中间 cast（`hivm.hir.vcast bool_to_bool_rintmode`）不支持。

**已尝试的 workaround**（全部失败）：
| 尝试 | 代码 | 结果 |
|------|------|------|
| `torch.div` 替代 | `torch.div(a, 128, rounding_mode='floor')` | ❌ 同样 bool_to_bool_rintmode |
| true_divide + floor | `(a.float()/128.0).floor().int()` | ❌ 同样 |
| 右移替代 | `a >> 7`（= a // 128） | ❌ 同样（inductor 不支持 pointwise 中的位移） |
| 乘倒数 | `(a * 0.0078125).floor().int()` | ❌ 同样 |

**结论**：❌ Python/Triton 层面无 workaround。这是 CANN 8.5.0 bishengir 编译器 bug，`arith.floordivsi` 的 lowering 路径中存在 unsupported cast。需升级 CANN 或向华为提交 bug。

**替代方案（绕过 `//`）**：
- 用 CPU pre-build kv_indices + `BlockMask.from_kv_blocks` 可以绕过 `//`（适用于纯 block-level 的 mask，如 `random_block_sparse`）
- 对于 token-level 需要 `//` 的模式（如 `block_diagonal`），无法绕过

**受影响模式**：nested, dilated_window, strided, checkerboard_64, block_diagonal_64/128, uniform_doc_256, hybrid_sparse, multiscale_dilated

### 1.2 UB overflow（1 个模式受影响）

**现象**：
```
ub overflow, requires 1724416 bits while 1572864 bits available
```

**根因**：mask_mod 子图中间值超出 NPU Unified Buffer 容量（约 192KB = 1572864 bits）。

**Workaround**：✅ **减小 BLOCK_M/BLOCK_N**。实测 `BLOCK_M=32, BLOCK_N=32` 可让 `global_local` 通过编译。

```python
flex_attention(q, k, v, block_mask=bm, kernel_options={"BLOCK_M": 32, "BLOCK_N": 32})
```

**验证**：
```
global_local @ S=512: BLOCK_M=64,BLOCK_N=64 → UB overflow ❌
global_local @ S=512: BLOCK_M=32,BLOCK_N=32 → 编译通过 ✅
```

**受影响模式**：global_local

### 1.3 aten.index.Tensor（2 个模式受影响）

**现象**：`aten.index.Tensor` 无法在 pointwise subgraph 中 lowering。

**根因**：NPU inductor 不支持用 tensor 下标索引另一个 tensor。

**Workaround**：✅ **预构建 kv_indices + from_kv_blocks**。

对于 `random_block_sparse`：在 CPU 上预计算随机 block mask → 手动构建 `kv_indices`/`kv_num_blocks` → 用 `BlockMask.from_kv_blocks` 传入，配合简单的 `mask_mod`（如 causal）。

```python
# 1. 在 CPU 上构建 kv_indices
kv_indices_t = ...  # [1, 1, n_blocks, max_blocks] int32
kv_num_blocks_t = ...  # [1, 1, n_blocks] int32

# 2. 用 from_kv_blocks 创建 BlockMask，mask_mod 用简单版本
bm = BlockMask.from_kv_blocks(
    kv_num_blocks=kv_num_blocks_t,
    kv_indices=kv_indices_t,
    BLOCK_SIZE=(128, 128),
    mask_mod=simple_mask,  # 不能包含 tensor 索引
).to(device)

# 3. 正常运行 flex_attention
flex_attention(q, k, v, block_mask=bm)
```

对于 `alibi_causal`：用算数运算（`2.0**(-8.0*(head+1)/num_heads)`）代替 `slopes[head]` 下标索引。但当前 Triton ascend backend 不支持 score_mod 中的浮点指数运算（`**`）。**暂时无 workaround。**

**受影响模式**：random_block_sparse（✅ 已解决），alibi_causal（❌ 仍需 compiler 支持 FP 运算）

## 2. Ascend 910B NPU 规格

- CANN 版本：8.5.0
- UB (Unified Buffer)：约 1572864 bits ≈ 192KB
- bishengir 路径：`/usr/local/Ascend/cann-8.5.0/tools/bishengir/bin/bishengir-compile`
- 编译选项：`--enable-auto-multi-buffer=True --enable-auto-bind-sub-block=True --enable-hfusion-compile=true --enable-hivm-compile=true`

## 3. torch_npu 架构

- `torch_npu._inductor.kernel.flex_attention.py` — NPU flex_attention kernel
- 基于 PyTorch Inductor 的 TritonTemplate 机制
- 使用 bishengir 作为 Triton backend
- 编译缓存：`/tmp/torchinductor_root/` 和 `/root/.triton/cache/`

## 4. 调试技巧

### 清除编译缓存
```bash
rm -rf /tmp/torchinductor_root /root/.triton/cache
```

### 快速判断错误类型
```python
try:
    compiled_fn(q, k, v)
except Exception as e:
    msg = str(e)
    if 'bool_to_bool_rintmode' in msg:       # // 操作
    elif 'ub overflow' in msg:                # UB 不足
    elif 'aten.index.Tensor' in msg:          # tensor 索引
    elif 'hivm.hir.vcast' in msg:             # bishengir cast 问题
```

### 减小 BLOCK_M/BLOCK_N 排查 UB overflow
```python
kernel_options = {"BLOCK_M": 32, "BLOCK_N": 32}  # 默认是 64/64
```

### 用 from_kv_blocks 绕过 mask_mod 限制
对于纯 block-level 的模式（如 random_block_sparse），可以在 CPU 上预计算 kv_indices，配合简单 mask_mod。

## 5. Reorder 性能效果总结

### 实测数据

| Pattern | S | hit_rate | baseline | reorder | diff |
|---------|---|----------|----------|---------|------|
| causal | 512-8192 | 1.0 | — | identity perm (跳过) | — |
| sliding_window_64 | 512 | 1.0 | 1.072ms | 1.060ms | -1.1% |
| global_local | 512 | 0.875 | 3.212ms | 3.286ms | +2.3% |
| global_local | 4096 | 0.875 | 30.888ms | 30.910ms | +0.07% |
| global_local | 8192 | 0.875 | 61.802ms | 61.811ms | +0.01% |
| random_block_sparse | 512 | 1.0 | 0.454ms | identity perm (跳过) | — |
| scattered (自定义) | 2048 | 0.0 | — | hit_rate still 0 | 0% |

### 结论：当前 NPU 架构下 reorder 反而更慢（2.3x 实测）

**实验**：random_block_sparse @ S=2048, hit_rate=0.43
- Baseline（无 reorder）：1.301 ms
- Reorder（kernel-internal PERM）：3.023 ms → **慢 2.3x**

**根因**：每个 Q block = 一个独立的 program。Program 按顺序加载 Q 内存时有 coalesced access。Reorder 破坏了这种连续性：

```
原始:  program[i] → Q[i*64 : i*64+64]  ← 连续 DRAM 访问
重排:  program[i] → Q[perm[i]*64 : ...]  ← 散乱 DRAM 访问
```

Q 加载从连续变为散乱，DRAM 带宽效率大幅下降，其开销远超 KV cache 改善。

```
GPU vllm-omni:
  wave[0..7] 在同一个 thread block 内
  → 8个 Q blocks 共享 L2 cache
  → reorder 有效

NPU flex_attention:
  program[0], program[1], ... 独立 launch
  → 每个 program 各自加载 KV blocks
  → reorder 无效
```

### 正确方案：Wave-based kernel

修改 kernel 模板，让单个 program 处理多个连续的 Q blocks：

```python
# 当前：每个 program 处理 1 个 Q block
program_id → 处理 Q block[pid]

# Wave 方案：每个 program 处理 WAVE_SIZE 个 Q blocks
wave_id → for w in range(WAVE_SIZE):
             处理 Q block[wave_id * WAVE_SIZE + w]
```

这样：
1. 同一 wave 内的 Q blocks 可以共享已加载的 KV blocks
2. Reorder 让相邻 Q blocks 访问相似的 KV blocks → 减少重复加载
3. Score_mod 不需要修改（Q tensor 不变）

## 6. Wave-based Kernel 实现计划

### 背景

实验证明：每个 Q block 一个 program 的模型下，reorder 破坏 Q DRAM 连续性，反而慢 2.3x。要让 reorder 有效，必须实现 wave-based 处理。

### 核心思路

```
当前: program[i] → 处理 Q block[i] → load KV → compute → write
Wave: program[w] → 处理 Q blocks[w*K .. w*K+K-1] → load KV once → compute all → write all
```

### 实现方法（非 ad-hoc）

不直接编辑 370 行的模板字符串，而是用 `TritonTemplate` 的多 variant 机制：

**Step 1: 添加 WAVE_SIZE 参数到 kernel_options**
```python
kernel_options.setdefault("WAVE_SIZE", 1)
```

**Step 2: 修改 grid 函数**
```python
def _flex_attention_grid_with_wave(*args, **kwargs):
    grid = list(flex_attention_grid(*args, **kwargs))
    wave_size = kwargs.get('WAVE_SIZE', 1)
    grid[0] = max(1, (grid[0] + wave_size - 1) // wave_size)
    return tuple(grid)
```

**Step 3: 生成 wave-variant kernel 模板**（关键）
不是手动编辑字符串，而是用 Python 函数生成：
```python
def _generate_wave_kernel_source(base_source, wave_size):
    """在 base_source 的 Q block 处理循环外包裹 for wave_w 循环"""
    # 找到 Q block 处理段（m_i 初始化 → store_output）
    # 添加 4 空格缩进
    # 在开头插入 for wave_w in range(WAVE_SIZE):
    # 在结尾插入 # end wave
    ...
```

**Step 4: 两个 variant 都注册到 choices**
```python
# WAVE_SIZE=1: 原始版本
error = flex_attention_template.maybe_append_choice(choices=choices, ...)

# WAVE_SIZE>1: wave 版本（如果 kernel_options 指定）
if kernel_options.get('WAVE_SIZE', 1) > 1:
    wave_template = TritonTemplate(
        name="flex_attention_wave",
        grid=_flex_attention_grid_with_wave,
        source=generate_wave_source(compute_flex_attention, wave_size),
    )
    wave_template.maybe_append_choice(choices=choices, ...)
```

### 预计效果

- **Program launch 减少**: WAVE_SIZE=2 → grid 减半 → launch overhead 减半
- **KV cache 潜力**: 同 wave 内 Q blocks 共享 KV（需后续实现 KV buffer）
- **立即验证**: 对比 WAVE_SIZE=1 vs 2/4/8 的 kernel 时间

### 风险

- 每个 program 的 SRAM 使用量增加（多个 Q blocks 的 acc 需要更多空间）
- Triton 模板生成需要精确的字符串处理（验证用 git diff 确保只改了目标行）
- bishengir 编译器可能对大 kernel 有限制

## 7. 待补充

- [ ] Ascend 910B L2 cache / HBM 带宽 / AI Core 数量
- [ ] alibi_causal 的 FP score_mod workaround
- [ ] CANN 版本升级路径（8.5.0 → ?）
- [ ] Wave-based kernel 实现
- [ ] bishengir bug 向华为提交的渠道
