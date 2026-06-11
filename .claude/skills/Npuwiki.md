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

**根因**：整数 `//`（floor divide）在 Triton→MLIR lowering 中生成 `arith.floordivsi`，bishengir 对该操作生成的中间 cast 不支持。所有除法变体（`torch.div`、`>>`右移、`true_divide`、乘倒数）均触发同样错误。

**Workaround**：❌ Python/Triton 层面无 workaround。需 bishengir/Triton ascend backend 层面修复 `arith.floordivsi` 的 lowering。

**受影响模式**：nested, dilated_window, strided, checkerboard_64, block_diagonal_64, uniform_doc_256, hybrid_sparse, multiscale_dilated

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

## 5. 待补充

- [ ] Ascend 910B L2 cache / HBM 带宽 / AI Core 数量
- [ ] 更多 bishengir 编译选项
- [ ] alibi_causal 的 FP score_mod workaround
- [ ] CANN 版本升级路径
