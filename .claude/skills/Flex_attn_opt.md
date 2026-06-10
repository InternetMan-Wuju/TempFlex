---
name: Flex_attn_opt
description: NPU Flex Attention 项目快速上手 — 部署、测试 block reorder 正确性与性能
metadata:
  skill: true
---

# Flex Attention NPU 快速上手指南

## 项目目标

在 NPU（Ascend）上优化 `torch_npu/_inductor/kernel/flex_attention.py`，核心是把 `sparse-attn-source` 的 **block reorder（重排）** 技术用到 `torch_npu` 的 flex_attention 实现中，让 flex_attention 在各种稀疏模式下都能达到或接近 manual attention 的性能。

> ⚠️ **当前状态：block reorder 尚未正确实现。** `flex_attention_reorder.py` 有设计计划但代码未完成，测试脚本中 `_HAS_REORDER = False`。在此之前所有稀疏模式（包括 causal）都走通用 block-sparse Triton 模板，flex 明显慢于 manual。

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
  apply_raw.sh                                      ← 部署 raw 版本

Newest/                                            ← 开发版本（含重排设计）
  site-packages/torch_npu/_inductor/kernel/
    flex_attention_newest.py                        (2376 行，开发版本)
  apply_newest.sh                                   ← 部署 Newest 版本

flex_attention_run_script.py                       ← 单次 attention 性能/正确性测试
run_sparse_sweep.py                                ← 全量基准测试（多 shape × 多 mask）
sparse_masks.py                                    ← 18 种预定义稀疏 mask 配置
summarize_msprof.py                                ← msprof CSV 解析 + 对比报告生成

sparse_attention_report.md                         ← 最近一次性能报告
sparse_attention_after_report.md                   ← 优化后的性能报告
```

> ⚠️ **核心目标（block reorder）尚未完成。** `flex_attention_reorder.py` 有设计计划但代码未实现，测试脚本中 `_HAS_REORDER = False`。所有稀疏模式下 flex 目前都走通用 block-sparse 模板，慢于 manual。

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
# 默认 causal 掩码（flex vs manual 对比）
python3 flex_attention_run_script.py --shape 4,8,2048,128

# 非因果掩码
python3 flex_attention_run_script.py --shape 1,4,512,64 --sparse-config sliding_window_64

# 只看 flex（不对比 manual）
python3 flex_attention_run_script.py --shape 4,8,2048,128 --target flex

# 只测性能，不对比精度（更快）
python3 flex_attention_run_script.py --shape 4,8,2048,128 --no-compare

# 指定精度
python3 flex_attention_run_script.py --shape 4,8,2048,128 --dtype fp16
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
# 开启 reorder
python3 flex_attention_run_script.py --shape 4,8,2048,128 --enable-block-reorder

# 关闭 reorder（默认）
python3 flex_attention_run_script.py --shape 4,8,2048,128
```

## 参数速查

### 常用 `--sparse-config` 选项

| 配置名 | 说明 | NPU 支持 | 限制 |
|--------|------|----------|------|
| `causal` | 因果掩码（基线） | ✅ 通过 | — |
| `sliding_window_64` | 滑动窗口 size=64 | ✅ 通过 | — |
| `sliding_window_128` | 滑动窗口 size=128 | ✅ 通过 | — |
| `prefix_lm` | Prefix LM prefix=16 | ✅ 通过 | — |
| `band_global_32` | Band(32)+Global(2) | ✅ 通过 | — |
| `global_local` | 全局(4)+局部(64) | ❌ bishengir | UB overflow (1724416 > 1572864 bits) |
| `nested` | 局部(64)+步长(32) | ❌ bishengir | `//` 生成 bool_to_bool_rintmode |
| `dilated_window` | 空洞滑动窗口 | ❌ bishengir | `//` 生成 bool_to_bool_rintmode |
| `strided` | 步长掩码 | ❌ bishengir | `//` 生成 bool_to_bool_rintmode |
| `checkerboard_64` | 棋盘掩码 | ❌ bishengir | `//` 生成 bool_to_bool_rintmode |
| `block_diagonal_64` | 块对角掩码 | ❌ bishengir | `//` 生成 bool_to_bool_rintmode |
| `uniform_doc_256` | 统一文档掩码 | ❌ bishengir | `//` 生成 bool_to_bool_rintmode |
| `hybrid_sparse` | 复合稀疏 | ❌ bishengir | `//` 生成 bool_to_bool_rintmode |
| `multiscale_dilated` | 多尺度空洞 | ❌ bishengir | `//` 生成 bool_to_bool_rintmode |
| `random_block_sparse` | 随机块稀疏 | ⚠️ 未测试 | 使用 tensor 索引，可能不支持 |
| `alibi_causal` | ALiBi + Causal | ⚠️ 未测试 | score_mod 含 slopes 数组访问 |

### 输出解读

脚本运行后会打印：

```
shape=4,8,2048,128 causal bfloat16
  flex_attention:   9.123 ms
  manual_sdpa:      2.456 ms
  flex_reorder:     N/A
  ✅ 测试通过（allclose=True）
  max_abs_diff=0.000122, max_rel_diff=0.012%
```

- `flex_attention` — 当前部署版本的 flex 性能
- `manual_sdpa` — PyTorch 原生 SDPA（基线）
- `flex_reorder` — block reorder 性能（暂不可用，显示 N/A）
- `max_abs_diff / max_rel_diff` — 与 manual 的精度差异

### reorder 相关参数

```bash
--enable-block-reorder         # 开启 block reorder
--block-reorder-mode wave_overlap  # 重排算法（默认 wave_overlap）
--wave-size 128                # wave 分区大小
```

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

当前 `flex_attention_reorder.py` 尚未实现。如果看到 `_HAS_REORDER = False` 或 import 失败，说明 reorder 功能未就绪，flex 走通用 block-sparse 模板。

### 非 causal 编译报错

**当前状态（2026-06-10）：**

已修复的内核问题：
- `get_offset_for_next_block` 中 masked `tl.load` → `tl.minimum` 钳制下标（消除 bishengir tensor 操作崩溃）
- `BLOCKS_ARE_CONTIGUOUS` / `ROWS_GUARANTEED_SAFE` 默认值 `False`（匹配上游 PyTorch）
- `sparse_masks.py` 中 mask_mod 重写：`%` → `(x//d)*d==x`、`|` → `bool+bool→int`、`.abs()` → `(diff<=n)&(-diff<=n)`

已知 bishengir 编译器限制：
- **`bool_to_bool_rintmode`**：整数 `//`（floor divide）生成的 MLIR 无法编译，影响 8 个模式
- **UB overflow**：3 条件以上的 mask_mod 超出 UB 空间限制（1572864 bits），影响 `global_local`
- 这两个问题需要在 bishengir/triton ascend backend 层面修复

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
| `raw_flex`（原始） | 无优化，所有场景走通用 block-sparse Triton 模板 | ✅ 可用（基线） |
| `Newest`（开发版） | 新增 debug hook + reorder 相关改动 | ✅ 部署可用 |
| **block reorder**（sparse-attn-source） | 对 Q/KV blocks 做重排以提升连续访存与缓存命中 | ❌ 设计完成，代码未实现 |