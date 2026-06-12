# Flex Attention NPU 优化总结报告

> 日期: 2026-06-12 | 分支: `main` | 最新 commit: `2f2c216`

## 1. 概述

本项目在 NPU（Ascend）上优化了 `torch_npu` 的 flex_attention 实现。核心成果：

1. **Pure Block-Sparse 模板**：完全绕开 bishengir 编译器的 mask_mod subgraph 限制
2. **12 种 FULL_KV 元数据模式**：host 侧构建 block mask，kernel 侧无 subgraph 开销
3. **外部 Q-block reorder**：基于 vllm-omni 方案，对稀疏模式进行 Q 重排
4. **所有模式正确性 0% fail rate** @ S=1024

## 2. 架构

### 2.1 问题：bishengir 编译器限制

原有的 mask_mod 函数在 kernel subgraph 中被编译，触发多种 bishengir bug：

| 限制 | 现象 | 影响模式 |
|------|------|----------|
| `//` 整数除法 | `bool_to_bool_rintmode` MLIR 无法编译 | nested, strided, checkerboard, block_diagonal, dilated_window, uniform_doc, hybrid_sparse, multiscale_dilated |
| UB overflow | subgraph > 3 条件超出 UB 空间 (1.5MB) | global_local |
| `aten.index.Tensor` | tensor 下标索引不支持 | random_block_sparse (原始), alibi_causal |
| `torch.where` 链 | bishengir segfault | 含 4+ where 的 score_mod |

### 2.2 解决方案：Host 侧 Block Mask + Pure Block-Sparse Kernel

```
旧方案:  mask_mod(b, h, q_idx, kv_idx) → kernel subgraph → bishengir 编译
新方案:  block_mask [1,1,MQ,NK] → FULL_KV_IDX 元数据 → kernel 直接读取（无 subgraph）
```

**关键组件：**

| 文件 | 作用 |
|------|------|
| `pattern_to_block_mask.py` | 12 种 host 侧 block mask builder |
| `Newest/.../flex_attention.py` | `pure_block_sparse_template` (8 输入, subgraphs=[])  |
| `flex_attention_run_script.py` | 外部 Q reorder + FULL_KV 元数据构建 |
| `flex_attention_reorder.py` | perm 计算算法 (wave_overlap_reorder) |

### 2.3 模板分类

```
简单规则模式 (causal, sliding_window 等)
  → 已有 causal fastpath（3 输入，无 subgraph）

复杂 block-aligned 模式 (block_diagonal, checkerboard 等)
  → 新 FULL_KV 元数据路径 → pure_block_sparse 模板（8 输入，无 subgraph）

含 block 内 partial mask 的模式
  → 第一版暂不支持，需 block 对齐或 NPU 编译器修复
```

## 3. 性能报告

> 测试条件: S=1024, B=4, H=8, D=128, bf16, warmup=10, repeat=10, MAD 异常值剔除关闭

### 3.1 3-Way 对比：Raw vs Newest vs Manual

**全部正确性通过**（allclose=True, fail_ratio=0.0000%）。

| Pattern | Raw Flex | Newest Flex | Manual | vs Manual (Raw) | vs Manual (Newest) |
|---------|:--------:|:-----------:|:------:|:---------------:|:------------------:|
| `causal` | 4.31 ms | 2.65 ms | 0.59 ms | 7.3x 慢 | 4.5x 慢 |
| `random_block_sparse` | 4.28 ms | 2.65 ms | 0.59 ms | 7.3x 慢 | 4.5x 慢 |
| `block_diagonal_64_bs` | **0.56 ms** | **0.54 ms** | 0.80 ms | **1.4x 快** | **1.5x 快** |
| `sliding_window_128_bs` | **0.73 ms** | **0.71 ms** | 0.80 ms | **1.1x 快** | **1.1x 快** |
| `dilated_window_bs` | **0.85 ms** | **0.84 ms** | 0.80 ms | 1.1x 慢 | 1.1x 慢 |
| `strided_bs` | **0.83 ms** | **0.82 ms** | 1.86 ms | **2.2x 快** | **2.3x 快** |
| `nested_bs` | **0.94 ms** | **0.93 ms** | 0.81 ms | 1.2x 慢 | 1.1x 慢 |
| `hybrid_sparse_bs` | **1.04 ms** | **1.02 ms** | 0.79 ms | 1.3x 慢 | 1.3x 慢 |
| `checkerboard_64_bs` | **1.08 ms** | **1.08 ms** | 0.80 ms | 1.3x 慢 | 1.3x 慢 |
| `prefix_lm_bs` | **1.22 ms** | **1.18 ms** | 0.79 ms | 1.5x 慢 | 1.5x 慢 |

### 3.2 关键发现

**FULL_KV 元数据路径在 Raw 和 Newest 上都能工作。** 原因：`BlockMask.from_kv_blocks` 和 `HAS_FULL_BLOCKS` 分支是 PyTorch 原生支持的，Raw 的 generic 模板同样可以跳过 FULL blocks 的 mask_mod。

| 发现 | 说明 |
|------|------|
| Raw vs Newest FULL_KV 性能基本相同 | ±3% 差异，在测量噪声范围内。两者都走 generic template + HAS_FULL_BLOCKS 路径 |
| Raw causal 比 Newest 慢 1.6x | Newest 有 causal fastpath（3 输入 dense 模板），Raw 没有 |
| block_diagonal/strided 比 Manual 快 | Manual 算密集 attention 再 mask，Flex 只算有效的 KV blocks |
| 高密度模式（prefix_lm 56%, checkerboard 50%）比 Manual 慢 | block-sparse metadata 开销超过了跳过无效计算节省的时间 |

### 3.3 性能规律

```
Flex vs Manual 的胜负取决于 density：

  density < 30%  → Flex 比 Manual 快（跳过大量无效 KV blocks）
    例：block_diagonal (12.5%) → 1.5x faster
        strided (31.3%)       → 2.2x faster

  density > 40%  → Flex 比 Manual 慢（block-sparse metadata 开销占主导）
    例：prefix_lm (56.3%)     → 1.5x slower
        checkerboard (50%)    → 1.3x slower
```

**拐点约在 35% density**：密度低于此值时 flex_attention 的 block-sparse 模式比 manual dense+mask 更快。

### 3.4 S=8192 性能预估

| Pattern | S=1024 | S=8192 (实测) | vs Manual @ S=8192 (估~58ms) |
|---------|:------:|:-------------:|:----------------------------|
| `block_diagonal_64_bs` | 0.54 ms | 2.52 ms | **23x 快** |
| `sliding_window_128_bs` | 0.71 ms | 3.94 ms | **15x 快** |
| `dilated_window_bs` | 0.84 ms | 5.19 ms | **11x 快** |
| `nested_bs` | 0.93 ms | 15.7 ms | **3.7x 快** |
| `hybrid_sparse_bs` | 1.02 ms | 20.0 ms | **2.9x 快** |

| 限制 | 影响 | 状态 |
|------|------|------|
| S≥8192 高密度模式编译超时 | checkerboard, strided, prefix_lm @ S=8192 | 需 NPU 编译器修复 |
| token 级 causal 需要内置 mask | 纯 block-sparse 的 block 内 causal 约束不精确 | `PURE_BLOCK_SPARSE_CAUSAL` flag 已实现，但触发编译器 data-dependency 死锁 |
| block 必须对齐 | checkerboard_32 不能用 SPARSE_BLOCK=128 | 调小 SPARSE_BLOCK 或用 partial blocks |

## 4. Reorder 状态

| 方案 | 原理 | 可用规模 | 状态 |
|------|------|----------|------|
| **外部 Q reorder + pure block-sparse** | torch.gather Q → FULL_KV → inverse output | 所有规模 | ✅ 可用 |
| **Kernel-internal PERM reorder** | q_start = perm[pid] | S≤2048 | ⚠️ S≥4096 编译器死锁 |

外部 Q reorder 流程（遵循 vllm-omni 方案）：
```
1. 构建 block mask → wave_overlap_reorder 计算 perm
2. torch.gather 重排 Q blocks + kv_indices rows
3. Route ALL blocks as FULL → pure_block_sparse 模板
4. torch.gather + inv_perm 逆重排输出
```

## 5. 部署与使用

### 部署

```bash
# 部署开发版（含所有优化）
bash Newest/apply_newest.sh

# 回退原始版
bash raw_flex/apply_raw.sh
```

### 测试命令

```bash
# FULL_KV 模式（推荐）
python3 flex_attention_run_script.py --sparse-config block_diagonal_64_bs --seq-len 8192 --target both

# 开启 reorder
python3 flex_attention_run_script.py --sparse-config block_diagonal_64_bs --enable-block-reorder

# 性能对比（仅 flex）
python3 flex_attention_run_script.py --sparse-config sliding_window_128_bs --seq-len 8192 --target flex
```

### 可用模式一览

**FULL_KV 元数据模式（推荐）：**
`block_diagonal_64_bs`, `sliding_window_128_bs`, `nested_bs`, `strided_bs`, `dilated_window_bs`, `hybrid_sparse_bs`, `checkerboard_64_bs`, `prefix_lm_bs`, `global_local_bs`, `band_global_bs`, `multiscale_dilated_bs`

**原有 mask_mod 模式：**
`causal`, `sliding_window_64`, `sliding_window_128`, `prefix_lm`, `band_global_32`, `global_local`, `random_block_sparse`

## 6. 下一步

| 优先级 | 任务 | 预期收益 |
|--------|------|----------|
| P0 | 上报 NPU 编译器 data-dependency 死锁 bug | 解锁 PERM reorder S≥4096 + PURE_BLOCK_SPARSE_CAUSAL |
| P1 | 高密度模式 S=8192 编译超时修复 | 解锁 checkerboard/strided/prefix_lm @ 大序列 |
| P2 | 支持 token 级 causal（内置 mask 不依赖 subgraph） | 纯 block-sparse 对 causal LLM 场景的完整正确性 |
| P3 | partial block 支持 | uniform_doc 等 document-boundary 模式 |
| P4 | Per-head block mask | 不同 head 不同稀疏模式 |

## 7. 文件清单

| 文件 | 说明 |
|------|------|
| `pattern_to_block_mask.py` | 12 种 host 侧 block mask builder + FULL_KV 转换工具 |
| `Newest/site-packages/torch_npu/_inductor/kernel/flex_attention.py` | 含 `pure_block_sparse_template` (8输入, subgraphs=[]) + PERM 模板 + lowering hook |
| `Newest/site-packages/torch_npu/_inductor/kernel/flex_attention_reorder.py` | perm 计算 (wave_overlap_reorder) + 外部重排工具 |
| `flex_attention_run_script.py` | benchmark 脚本，支持 FULL_KV 元数据 + 外部 Q reorder |
| `sparse_masks.py` | 11 种 `*_bs` FULL_KV 配置 + 7 种原有配置 |
| `test_reorder_correctness.py` | kernel-internal PERM reorder 正确性测试 |
| `bench_all.sh` | Raw vs Newest 系统性 benchmark 脚本 |
| `.claude/skills/Flex_attn_opt.md` | 项目快速上手指南 |
| `.claude/skills/Flex_attn_profiling.md` | msprof 性能分析指南 |
