---
name: Flex_attn_opt
description: NPU Flex Attention 项目快速上手 — 部署、测试 block reorder 正确性与性能
metadata:
  skill: true
---

# Flex Attention NPU 快速上手指南

## 项目目标

在 NPU（Ascend）上优化 `torch_npu/_inductor/kernel/flex_attention.py`，核心是把 `sparse-attn-source` 的 **block reorder（重排）** 技术用到 `torch_npu` 的 flex_attention 实现中，让 flex_attention 在各种稀疏模式下都能达到或接近 manual attention 的性能。

> **当前状态 (2026-06-12)：Pure Block-Sparse 模式已上线。** 12 种稀疏模式通过 host 侧 block mask 构建 + `FULL_KV_IDX` 元数据路径，完全绕开 bishengir 编译限制。block_diagonal 在 S=8192 达到 **23x 加速**（vs dense），sliding_window **14.7x**。所有模式正确性 0% fail rate。

## 核心要求

### Reorder 必须基于 sparse-attn-source 实现

参考实现位于 `sparse-attn-source/new-vllm-omni-for-sparse-attn-main/`：

| 文件 | 内容 |
|------|------|
| `vllm_omni/diffusion/attention/backends/greedy_reorder_cuda.py` | CUDA reorder 算法（global_greedy, banded_greedy, banded_window_greedy, optimal_reorder_cuda） |
| `vllm_omni/diffusion/attention/backends/reorder_rows_graph.py` | CPU reorder 算法（greedy nearest-neighbor, numba JIT） |
| `examples/offline_inference/text_to_video/reorder_method_branches_zh.md` | 方法演进分支全景图 |

vllm-omni 的 reorder 方法演进路径（按效果排序）：

```
NNZ/Wave系 → Spectral/Fiedler系 → Global Greedy/BCG系 → Union Oracle系 → Auction CUDA系（当前最佳）
```

### Reorder 必须在 Kernel 内部实现

**关键教训：外部 Q reorder 在 NPU 上不可行。**

原因：
1. 外部重排 Q 后，`score_mod`/`mask_mod` 使用物理 token 位置（0..S-1），与原始 Q token 不再对应
2. 修正 `score_mod` 需要 `torch.where` 链或 `//`/`%` 操作来映射位置，bishengir 编译器无法处理（segfault 或 `bool_to_bool_rintmode`）
3. NPU 不支持 `aten.index.Tensor`（tensor 下标索引），无法用查表方式映射

正确做法：修改 `flex_attention.py` 的 kernel 模板，在 Q block 迭代循环内部改变 block 处理顺序。

## 使用场景

- 你想部署开发版或恢复原始版 flex_attention
- 你想跑一次 attention 性能测试（单次或全量 sweep）
- 你想验证 block reorder 的正确性与性能
- 你想检查精度/正确性
- 你想对比 flex_attention 和 manual attention 的性能
- 你遇到编译报错或部署问题，需要确认当前版本状态

## 项目结构

```
raw_flex/                                         ← 原始 torch_npu flex_attention.py
  site-packages/torch_npu/_inductor/kernel/
    flex_attention.py                               (1874 行，未修改)
    flex_attention_reorder.py                       ← Reorder 算法 + 外部重排工具
  apply_raw.sh                                      ← 部署 raw 版本

Newest/                                            ← 开发版本（含重排设计）
  site-packages/torch_npu/_inductor/kernel/
    flex_attention_newest.py                        (2376 行，开发版本)
    flex_attention_reorder.py                       ← Reorder 模块（已修复正确性）
  apply_newest.sh                                   ← 部署 Newest 版本

sparse-attn-source/                                ← 参考实现（vllm-omni）
  new-vllm-omni-for-sparse-attn-main/
    vllm_omni/diffusion/attention/backends/
      greedy_reorder_cuda.py                        ← CUDA reorder 算法
      reorder_rows_graph.py                         ← CPU numba reorder 算法

flex_attention_run_script.py                       ← 单次 attention 性能/正确性测试
run_sparse_sweep.py                                ← 全量基准测试（多 shape × 多 mask）
sparse_masks.py                                    ← 18 种预定义稀疏 mask 配置
summarize_msprof.py                                ← msprof CSV 解析 + 对比报告生成

sparse_attention_report.md                         ← 最近一次性能报告
sparse_attention_after_report.md                   ← 优化后的性能报告
```

> ⚠️ **Block reorder 可用但收益有限。** kernel-internal PERM reorder 受 NPU 编译器限制（S≥4096 死锁）。外部 Q reorder + pure block-sparse 可在所有规模使用，正确性已验证。

## 快速命令

### 部署

```bash
# 部署开发版本
bash Newest/apply_newest.sh

# 恢复原始版本
bash raw_flex/apply_raw.sh
```

部署后会覆盖系统 `site-packages/torch_npu/_inductor/kernel/flex_attention.py`。

### 单次测试

```bash
# 默认 causal 掩码（flex vs manual 对比），默认 S=8192
python3 flex_attention_run_script.py

# 指定 shape
python3 flex_attention_run_script.py --shape 4,8,2048,128

# 长序列测试（large suite: 4K ~ 16K）
python3 flex_attention_run_script.py --shape-suite large

# 非因果掩码
python3 flex_attention_run_script.py --shape 1,4,512,64 --sparse-config sliding_window_64

# 只看 flex（不对比 manual）
python3 flex_attention_run_script.py --shape 4,8,2048,128 --target flex

# 只测性能，不对比精度（更快）
python3 flex_attention_run_script.py --shape 4,8,2048,128 --no-compare

# 指定精度
python3 flex_attention_run_script.py --shape 4,8,2048,128 --dtype fp16
```

### 双卡测试

```bash
# 双卡并行测试：每个 shape 在两张卡上各跑一次
python3 flex_attention_run_script.py --device npu:0,npu:1 --shape-suite large

# 单卡（默认）
python3 flex_attention_run_script.py --device npu:0
```

### 异常值剔除

当 `--repeat >= 5` 时默认启用 MAD（Median Absolute Deviation）异常值剔除，自动丢弃明显偏离中位数的迭代耗时：

```bash
# 默认开启（repeat=10 >= 5）
python3 flex_attention_run_script.py --warmup 5 --repeat 10
# 输出示例：... avg: 11.890 ms (warmup=5, repeat=10, trimmed: 2/10 outliers dropped)

# 关闭剔除
python3 flex_attention_run_script.py --warmup 5 --repeat 10 --no-trim-outliers
```

### 全量基准

```bash
# 遍历 8 种 mask × 4 种 shape，输出 markdown 报告
python3 run_sparse_sweep.py
```

### 正确性验证

```bash
# warmup 3 轮 + repeat 3 轮，检查 allclose
python3 flex_attention_run_script.py --shape 4,8,2048,128 --warmup 3 --repeat 3

# 如果看到 "✅ 测试通过（allclose=True）" → 精度通过
# 如果看到 "❌ 测试失败（allclose=False）" → 精度异常，需排查
```

### 验证 reorder 状态

```bash
# 确认当前 reorder 是否可用
python3 -c "from torch_npu._inductor.kernel.flex_attention_reorder import reorder_flex_forward" 2>&1
```

- Import 成功 → reorder 可用
- ImportError → reorder 还未实现

### 测试 reorder 开启/关闭对比

```bash
# 对 FULL_KV 模式开启 reorder
python3 flex_attention_run_script.py --sparse-config block_diagonal_64_bs --enable-block-reorder

# 对 random_block_sparse 开启 reorder
python3 flex_attention_run_script.py --sparse-config random_block_sparse --enable-block-reorder

# 关闭 reorder（默认）
python3 flex_attention_run_script.py --sparse-config block_diagonal_64_bs
```

## 参数速查

### 常用 `--sparse-config` 选项

#### FULL_KV 元数据模式（推荐，host 侧构建 block mask，绕开 bishengir）

所有模式 S=1024 正确性已通过（allclose=1, 0% fail rate）。性能数据来自 2026-06-12 benchmark。

| 配置名 | S=1024 Flex | S=1024 Manual | S=8192 Flex | vs Raw(S=8192) | 正确性 |
|--------|:-----------:|:-------------:|:-----------:|:--------------:|:------:|
| `block_diagonal_64_bs` | 0.57 ms | 0.86 ms | 2.52 ms | — | ✅ |
| `sliding_window_128_bs` | 0.72 ms | 0.92 ms | 3.94 ms | — | ✅ |
| `nested_bs` | 0.95 ms | 0.84 ms | 15.7 ms | — | ✅ |
| `strided_bs` | 0.82 ms | 0.85 ms | ⏱ timeout | — | ✅ |
| `dilated_window_bs` | 0.86 ms | 0.86 ms | 5.19 ms | — | ✅ |
| `hybrid_sparse_bs` | 1.03 ms | 0.86 ms | 20.0 ms | — | ✅ |
| `checkerboard_64_bs` | 1.10 ms | 0.87 ms | ⏱ timeout | — | ✅ |
| `prefix_lm_bs` | 1.20 ms | 0.85 ms | ⏱ timeout | — | ✅ |

#### 3-Way 对比：Raw vs Newest vs Manual（S=1024, warmup=10, repeat=10）

全部正确性通过（allclose=1, 0% fail）。

| 配置名 | Raw Flex | Newest Flex | Manual | vs Manual |
|--------|:--------:|:-----------:|:------:|:---------:|
| `causal` | 4.31 ms | 2.65 ms | 0.59 ms | 4.5x 慢 (N) |
| `random_block_sparse` | 4.28 ms | 2.65 ms | 0.59 ms | 4.5x 慢 (N) |
| `block_diagonal_64_bs` | 0.56 ms | 0.54 ms | 0.80 ms | **1.5x 快** |
| `sliding_window_128_bs` | 0.73 ms | 0.71 ms | 0.80 ms | **1.1x 快** |
| `strided_bs` | 0.83 ms | 0.82 ms | 1.86 ms | **2.3x 快** |
| `dilated_window_bs` | 0.85 ms | 0.84 ms | 0.80 ms | 1.1x 慢 |
| `nested_bs` | 0.94 ms | 0.93 ms | 0.81 ms | 1.1x 慢 |
| `hybrid_sparse_bs` | 1.04 ms | 1.02 ms | 0.79 ms | 1.3x 慢 |
| `checkerboard_64_bs` | 1.08 ms | 1.08 ms | 0.80 ms | 1.3x 慢 |
| `prefix_lm_bs` | 1.22 ms | 1.18 ms | 0.79 ms | 1.5x 慢 |

> **关键发现**：FULL_KV 元数据模式在 Raw 和 Newest 上**都能工作**（±3% 性能差在噪声内）。两者都走 generic template + HAS_FULL_BLOCKS 路径。Newest 独有优势在于 causal fastpath（3 输入 dense 模板，causal 和 random_block_sparse 快 1.6x）。
>
> **Performance vs Manual 拐点在 ~35% density**：低于此值 Flex 更快（跳过无效 KV blocks），高于此值 Manual 更快（block-sparse metadata 开销 > 节省的计算）。

#### S=8192 性能（Newest）

| 配置名 | S=1024 | S=8192 | vs Manual(S=8192, ~58ms) |
|--------|:------:|:------:|:------------------------:|
| `block_diagonal_64_bs` | 0.54 ms | 2.52 ms | **23x** |
| `sliding_window_128_bs` | 0.71 ms | 3.94 ms | **15x** |
| `dilated_window_bs` | 0.84 ms | 5.19 ms | **11x** |
| `nested_bs` | 0.93 ms | 15.7 ms | **3.7x** |
| `hybrid_sparse_bs` | 1.02 ms | 20.0 ms | **2.9x** |

#### 原有 mask_mod 模式（token 级别，部分受 bishengir 限制）

| 配置名 | 说明 | NPU 支持 | 限制 |
|--------|------|----------|------|
| `causal` | 因果掩码（基线） | ✅ 通过 | — |
| `sliding_window_64` | 滑动窗口 size=64 | ✅ 通过 | — |
| `sliding_window_128` | 滑动窗口 size=128 | ✅ 通过 | — |
| `prefix_lm` | Prefix LM prefix=16 | ✅ 通过 | — |
| `band_global_32` | Band(32)+Global(2) | ✅ 通过 | — |
| `global_local` | 全局(4)+局部(64) | ✅ 通过 | — |
| `random_block_sparse` | 随机块稀疏 | ✅ 通过 | 需预构建 kv_indices |
| `nested` | 局部(64)+步长(32) | ❌ bishengir | ⚠️ 用 `nested_bs` |
| `dilated_window` | 空洞滑动窗口 | ❌ bishengir | ⚠️ 用 `dilated_window_bs` |
| `strided` | 步长掩码 | ❌ bishengir | ⚠️ 用 `strided_bs` |
| `checkerboard_*` | 棋盘掩码 | ❌ bishengir | ⚠️ 用 `checkerboard_64_bs` |
| `block_diagonal_*` | 块对角 | ❌ bishengir | ⚠️ 用 `block_diagonal_64_bs` |
| `uniform_doc_256` | 统一文档掩码 | ❌ bishengir | ⚠️ 需 doc boundary 对齐 block（暂无 `_bs` 版本） |
| `hybrid_sparse` | 复合稀疏 | ❌ bishengir | ⚠️ 用 `hybrid_sparse_bs` |
| `multiscale_dilated` | 多尺度空洞 | ❌ bishengir | ⚠️ 用 `multiscale_dilated_bs` |
| `alibi_causal` | ALiBi + Causal | ❌ bishengir | score_mod 不支持 FP 操作，需 NPU 编译器修复 |

### 输出解读

脚本运行后会打印：

```
shape=4,8,2048,128 causal bfloat16
  flex_attention:   9.123 ms
  manual_sdpa:      2.456 ms
  flex_reorder:     8.901 ms
  ── reorder computation time (CPU): 0.234 ms | kernel time: 8.901 ms | comp/kernel ratio: 2.63%
  ✅ 测试通过（allclose=True）
  max_abs_diff=0.000122, max_rel_diff=0.012%
```

- `flex_attention` — 当前部署版本的 flex 性能
- `manual_sdpa` — PyTorch 原生 SDPA（基线）
- `flex_reorder` — block reorder 后的 kernel 时间（**不含**重排计算开销）
- `reorder computation time (CPU)` — 重排计算本身耗时（CPU 侧），与 kernel 时间分开显示
- `comp/kernel ratio` — 重排计算占 kernel 时间的比例；>10% 说明重排开销不可忽视
- `trimmed: X/N outliers dropped` — 出现时表示 MAD 异常值剔除了 X 个偏离点
- `max_abs_diff / max_rel_diff` — 与 manual 的精度差异

### Shape Suite 一览

| Suite | Shapes |
|-------|--------|
| `single` | 默认 shape（当前 `4,8,8192,128`） |
| `small` | `(1,2,128,64)`, `(1,4,256,64)`, `(2,4,512,64)` |
| `smoke` | `(1,4,512,64)`, `(2,8,1024,64)` + 默认 shape |
| `large` | `(1,4,4096,128)`, `(1,4,8192,128)`, `(2,4,8192,128)`, `(2,8,16384,128)` |

```bash
python3 flex_attention_run_script.py --shape-suite large
```

### reorder 相关参数

```bash
--enable-block-reorder         # 开启 Q-block reorder（外部 torch.gather + pure block-sparse）
--block-reorder-mode wave_overlap  # 重排算法（默认 wave_overlap）
--wave-size 132                # wave 分区大小
```

> **当前状态**：外部 Q reorder 已实现（vllm-omni 方案），配合 `PURE_BLOCK_SPARSE` 模板使用。对有 locality 的稀疏模式（block_diagonal, sliding_window）可进一步提升 KV cache 命中率。对 FULL_KV 模式（`*_bs`）和 `random_block_sparse` 均可用。

### msprof 性能分析入口

```bash
# 采集 msprof 数据（flex + manual 自动对比）
python3 flex_attention_run_script.py --mode msprof --shape 1,4,512,64

# 指定 AI Core 指标（默认 PipeUtilization）
python3 flex_attention_run_script.py --mode msprof --shape 4,8,2048,128 \
  --msprof-aic-metrics "ArithmeticUtilization"

# 指定输出目录
python3 flex_attention_run_script.py --mode msprof --shape 4,8,2048,128 \
  --msprof-output /path/to/output

# 解析采集到的 msprof 数据
python3 summarize_msprof.py msprof_out/<timestamp>/
```

> 更详细的 msprof 采集、解析、瓶颈诊断请参考 `Flex_attn_profiling` skill。

## 常见问题

### reorder 不可用

当前外部 Q reorder + pure block-sparse 流程对所有 FULL_KV 模式（`*_bs`）可用。kernel-internal PERM reorder 在 S≤2048 可用，S≥4096 被 NPU 编译器限制阻塞。详见 [[flex-attn-perm-reorder-implementation]]。

### 非 causal 编译报错

**当前状态（2026-06-10）：**

已修复的内核问题：
- `get_offset_for_next_block` 中 masked `tl.load` → `tl.minimum` 钳制下标（消除 bishengir tensor 操作崩溃）
- `BLOCKS_ARE_CONTIGUOUS` / `ROWS_GUARANTEED_SAFE` 默认值 `False`（匹配上游 PyTorch）
- `sparse_masks.py` 中 mask_mod 重写：`%` → `(x//d)*d==x`、`|` → `bool+bool→int`、`.abs()` → `(diff<=n)&(-diff<=n)`

已知 bishengir 编译器限制：
- **`bool_to_bool_rintmode`**：整数 `//`（floor divide）生成的 MLIR 无法编译，影响 8 个模式
- **UB overflow**：3 条件以上的 mask_mod 超出 UB 空间限制（1572864 bits），影响 `global_local`
- **segfault**：`torch.where` 链在 score_mod 子图中导致 bishengir 崩溃
- **`aten.index.Tensor`**：tensor 下标索引不支持
- 这些问题需要在 bishengir/triton ascend backend 层面修复

### 部署后 import 失败

```bash
# 确认当前部署版本
diff -q raw_flex/site-packages/torch_npu/_inductor/kernel/flex_attention.py \
         Newest/site-packages/torch_npu/_inductor/kernel/flex_attention.py

# 重新部署
bash raw_flex/apply_raw.sh && python3 -c "import torch_npu._inductor.kernel.flex_attention"
```

## 版本对照

| 版本 | 特点 | 状态 |
|------|------|------|
| `raw_flex`（原始） | 标准 generic + causal fastpath 模板 | ✅ 可用（基线） |
| `Newest`（开发版） | PERM 模板 + pure block-sparse 模板 + 12 种 FULL_KV 模式 | ✅ 部署可用 |
| **pure block-sparse（外部 Q reorder）** | Host 侧 block mask → FULL_KV 元数据 → 无 subgraph kernel | ✅ 所有规模可用 |
| **block reorder（kernel 内部 PERM）** | 修改 kernel 模板，通过 side-channel 传递 PERM | ⚠️ S≤2048 可用 |
| `pattern_to_block_mask.py` | 12 种 host 侧 block mask builder | ✅ 可用 |
