---
name: Flex_attn_profiling
description: NPU 上 msprof 性能分析工作流 — 采集、解析、瓶颈诊断、调优建议
metadata:
  skill: true
---

# Flex Attention NPU 性能分析（msprof）

## 使用场景

- flex_attention 比 manual_sdpa 慢，你想知道**瓶颈在哪**
- 你想对比 reorder 开启/关闭时的 device 行为差异
- 你想看 AI Core 算力利用率（cube_utilization）、访存瓶颈（Memory）
- 你想分析 kernel 耗时分布，定位是**计算瓶颈**还是**调度/拷贝瓶颈**
- 你需要输出一份结构化的 flex vs manual profiling 对比报告

## 工作流总览

```
1. 采集 ──→  2. 解析导出  ──→  3. 对比分析  ──→  4. 诊断建议
                ↑
        采集后自动解析，若需自定义导出再手动
```

---

## 1. 采集阶段

### 一键采集（flex + manual 对比）

```bash
python3 flex_attention_run_script.py --mode msprof --shape 4,8,2048,128
```

### ⚠️ Reorder 性能测试：必须排除重排计算开销

当使用 `--enable-block-reorder` 测试性能时，**重排计算发生在 flex_attention 调用之前**，不应计入 kernel 执行时间。

**正确做法：只测量 flex_attention 的 kernel 执行时间，不包含 `compute_and_set_pending_perm` 的开销。**

`flex_attention_run_script.py` 已经正确处理了这一点：
- `compute_and_set_pending_perm()` 在 `time_runner` 之外调用（warmup 之前）
- `time_runner` 只测量编译后的 `flex_attention` kernel 执行时间
- msprof 采集同样只覆盖 `time_runner` 内的 kernel 执行

```bash
# 正确的 reorder 性能测试 —— msprof 只采 flex_attention kernel
python3 flex_attention_run_script.py --mode msprof \
  --shape 4,8,2048,128 --enable-block-reorder --target flex

# 对比：无 reorder 的 baseline
python3 flex_attention_run_script.py --mode msprof \
  --shape 4,8,2048,128 --target flex
```

**错误做法（会导致性能数据无效）：**
```bash
# ❌ 不要手写循环把 compute_and_set_pending_perm 放在计时范围内
# ❌ 不要在 warmup/repeat 循环内调用 reorder 计算
```

**验证 reorder 开销是否被排除：**
```bash
# 这两个数字应该非常接近（差异 < 2%）—— reorder 是纯 CPU 计算
python3 flex_attention_run_script.py --shape 4,8,2048,128 --enable-block-reorder 2>&1 | grep "avg:"
python3 flex_attention_run_script.py --shape 4,8,2048,128 2>&1 | grep "Flex Attention avg:"
```

这会：
1. 先用 flex 跑一次 Attention（子进程），msprof 自动采集
2. 再用 manual 跑一次，msprof 分开采集
3. 输出目录默认为 `msprof_out/<timestamp>/flex/` 和 `msprof_out/<timestamp>/manual/`

### 采集参数详解

**输出目录控制：**
```bash
# 指定根输出目录
--msprof-output /path/to/output

# 目录结构：
#   /path/to/output/flex/PROF_XXX/
#   /path/to/output/manual/PROF_XXX/
```

**AI Core 指标选择（`--msprof-aic-metrics`）：**

| 指标值 | 采集内容 | 适用场景 |
|--------|----------|----------|
| `PipeUtilization`（默认） | 流水线（Vector/Scalar/Cube/Mtx）各单元占比 | 日常性能评估 |
| `ArithmeticUtilization` | 算力利用率 | 算子是否为 compute-bound |
| `Memory` | 访存带宽、L1/L2 hit rate | 算子是否为 memory-bound |
| `MemoryL0` | L0 buffer 访存细节 | 细粒度访存分析 |
| `ResourceConflictRatio` | 资源冲突比率 | 排查 bank conflict |

```bash
# 采集算力利用率 + 访存指标（可同时传多个，用逗号分隔）
--msprof-aic-metrics "ArithmeticUtilization,Memory"
```

**额外 msprof 选项（`--msprof-option`）：**
```bash
# 例如开启 L2 缓存采样
--msprof-option "--l2=on"
# 或同时传多个
--msprof-option "--l2=on --task-memory=on"
```

**其他 benchmark 参数也会透传：**
```bash
# 非 causal 场景
python3 flex_attention_run_script.py --mode msprof \
  --shape 1,4,512,64 --sparse-config sliding_window_64

# 修改 warmup/repeat（影响 msprof scope 解析）
python3 flex_attention_run_script.py --mode msprof \
  --shape 4,8,2048,128 --warmup 5 --repeat 5

# 只采 flex
python3 flex_attention_run_script.py --mode msprof \
  --shape 4,8,2048,128 --target flex
```

### 采集后产出

每个 `PROF_XXX/` 目录下的核心文件：

```
PROF_XXX/
  device_0/
    sample.json                    ← 运行参数（--warmup/--repeat 等信息）
    mindstudio_profiler_output/
      op_statistic_*.csv           ← 算子类型级聚合统计
      op_summary_*.csv             ← 每个算子调用的详细信息
      task_time_*.csv              ← kernel/task 级执行时间
      api_statistic_*.csv          ← Host 侧 API 调用统计
      msprof_*.json                ← Timeline 数据（trace）
```

---

## 2. 解析与导出

### 自动解析（推荐步骤）

非 RC 场景下采集完成时 msprof 已自动解析并导出最小 model_id 的第 1 轮迭代数据。直接进入对比分析即可。

### 手动重新导出

如在 RC 场景，或需要导出其他迭代/模型的数据：

```bash
# 1. 先查询可用的模型号和迭代号
./msprof --query=on --output=<PROF_XXX_dir>

# 2. 按需导出
./msprof --export=on --output=<PROF_XXX_dir> \
  --model-id=<model_id> --iteration-id=<iteration_id>

# 导出 JSON 格式（默认 CSV）
./msprof --export=on --output=<PROF_XXX_dir> \
  --summary-format=json
```

### 用 summarize_msprof.py 生成可读报告

```bash
# 默认：解析 msprof_out/<最新目录>/ 下的 flex/ manual/ 对比
python3 summarize_msprof.py

# 指定目录
python3 summarize_msprof.py msprof_out/20260609_143022/

# 只看 flex 的凑最近的 PROF_* 目录
python3 summarize_msprof.py msprof_out/20260609_143022/flex/
```

**scope 模式控制：**

```bash
# auto（默认）：有 MSTX 标记时截取 repeat 窗口，否则估算
python3 summarize_msprof.py --scope auto

# full：使用全部采集数据（含 warmup）
python3 summarize_msprof.py --scope full

# mstx：强制使用 MSTX 标记窗口（没有则 fallback full）
python3 summarize_msprof.py --scope mstx

# repeat-tail：估算 warmup 后的稳态迭代
python3 summarize_msprof.py --scope repeat-tail \
  --warmup 5 --repeat 5
```

**其他参数：**

```bash
# 看 Top 20
python3 summarize_msprof.py --top 20

# 跳过过大的 CSV 文件（默认 200MB）
python3 summarize_msprof.py --max-csv-mb 500

# 不写入 result.log
python3 summarize_msprof.py --no-result-log
```

---

## 3. 报告解读

### Overview 表

```
target | profile    | scope | op total ms | task total ms | helper ms | AI CPU ms | top op type
-------+------------+-------+-------------+---------------+-----------+-----------+-------------
flex   | PROF_XXX_1 | mstx  | 12.345      | 11.100        | 3.200     | 0.800     | MatMul 6.200 ms
manual | PROF_XXX_2 | mstx  | 4.567       | 4.200         | 0.500     | 0.050     | BatchMatMul 3.100 ms
```

| 字段 | 含义 | 诊断参考 |
|------|------|----------|
| `op total ms` | 所有 device op 总耗时（含计算+搬运） | flex vs manual 第一对比指标 |
| `task total ms` | kernel 执行总耗时（不含搬运） | 接近 op total 说明搬运少 |
| `helper ms` | 辅助算子耗时（cast/reshape/transpose/arange 等） | 占比高说明 flex 生成了额外辅助算子 |
| `helper %` | `helper / op_total` | **>20% → 明显异常**，说明 kernel 融合不完全 |
| `AI CPU ms` | AI CPU 回退耗时 | **>0 → 可能触发了低效回退路径** |
| `top op type` | 最耗时的算子类型 | 看是否应该是 attention/matmul |

### Flex / Manual 比值表

```
metric       | flex ms  | manual ms | ratio
-------------+----------+-----------+-------
op total     | 12.345   | 4.567     | 2.70x
task total   | 11.100   | 4.200     | 2.64x
host api     | 1.800    | 0.600     | 3.00x
helper total | 3.200    | 0.500     | 6.40x
AI CPU total | 0.800    | 0.050     | 16.0x
```

**快速判读：**

| 模式 | 含义 |
|------|------|
| 整体 1.0x-1.1x | 性能可比，接近最优 |
| 整体 >1.5x + helper >2x | flex 产生了多余辅助算子，未有效融合 |
| 整体 >2x + helper 正常 | 通用 block-sparse 模板本身低效 |
| AI CPU 大幅偏高 | 部分逻辑回退到了 AI CPU（检查算子类型） |
| host api 大幅偏高 | 可能 kernel launch 过多（小 shape 常见） |

### Top Op Types 表

```
# | name      | count | total ms | share | extra
--+-----------+-------+----------+-------+--------------
1 | MatMul    | 8     | 6.200    | 50.2% | AI Core, avg=0.775 ms
2 | Cast      | 12    | 1.800    | 14.6% | AI Core, avg=0.150 ms
3 | Transpose | 6     | 1.400    | 11.3% | AI Core, avg=0.233 ms
```

- **share >80% 是 MatMul/Attention/Flash** → 计算集中在核心算子上，健康
- **前 3 里有 Cast/Transpose/Reshape/Arange** → helper ops 占比过高，融合不够
- **出现 AI CPU 行** → 某算子跑在了 AI CPU 而非 AI Core，需关注

### Top Kernels/Tasks 表

```
# | kernel_name                                      | count | total ms | share | extra
--+--------------------------------------------------+-------+----------+-------+--------------
1 | FusedAttentionKernel_1                           | 2     | 4.200    | 37.8% | AI Core
2 | tvm_matmul_kernel_1                              | 4     | 2.200    | 19.8% | AI Core
3 | tvm_softmax_kernel_1                             | 2     | 1.800    | 16.2% | AI Core
```

- 看 flex 侧是否有 **单一的 fused kernel** 吃掉大部分时间 → reorder 生效、kernel 融合成功
- 如果 flex 侧 top kernel 是多个小 kernel（matmul + softmax + transpose 分散）→ 通用模板未融合

### Top Host APIs 表

```
# | name              | count | total ms | share | extra
--+-------------------+-------+----------+-------+--------------
1 | cudaGraphLaunch   | 10    | 0.800    | 44.4% | runtime, avg=0.080 ms
2 | cudaKernelLaunch  | 24    | 0.500    | 27.8% | runtime, avg=0.021 ms
```

- flex 侧 kernel launch 数量 > manual → 通用模板产生了更多小 kernel
- 小 shape 场景（seq_len ≤ 512）下 launch 开销占比更大

---

## 4. 瓶颈诊断决策树

```
用户报告 flex 慢于 manual
│
├─ msprof 报告分析
│  │
│  ├─ flex/manual op_total ratio
│  │  ├─ 3x+ → 严重瓶颈，看 top op types 是什么
│  │  ├─ 1.5x - 3x → 中度瓶颈，检查 helper % 和 AI CPU
│  │  └─ 1.0x - 1.5x → 接近合理范围，检查 host API overhead
│  │
│  ├─ helper % 高 → 融合失败
│  │  常见：Cast/Transpose/Arange → 检查 score_mod/mask_mod 是否引入了这些操作
│  │
│  ├─ AI CPU > 0 → 回退路径
│  │  排查哪个 op 跑在 AI CPU，固定 dtype/shape 规避
│  │
│  └─ task 数量明显偏多
│      对比 flex 和 manual 的 kernel count
│      flex 的 kernel 数量 > manual 的 3x → 通用模板未融合
│
├─ 通用 block-sparse 模板本来就是瓶颈
│  这是当前 NPU Triton adapter 的限制：
│  NPU 不能有效利用 block sparsity 跳过 masked-out blocks，
│  导致通用模板退化为若干小 matmul 的拼接。
│
│  改进方向：
│  ├─ 实现 block reorder（提升缓存命中 + 连续访存）
│  └─ 增大 --block-m / --block-n（减少 kernel launch 次数）
│
└─ host API overhead 高 → 小 shape 场景常见
      kernel launch 数量多 → 尝试增大 block_m / block_n
      python3 flex_attention_run_script.py --shape 4,8,512,64 \
        --block-m 128 --block-n 128
```

---

## 5. 常见瓶颈模式

### 模式 A：通用 block-sparse 模板低效

**症状：** 非 causal 场景，flex 2-5x 慢于 manual

**当前 NPU 支持的模式（7 个）：**
- `causal` — ✅ 通过（基线）
- `sliding_window_64` / `sliding_window_128` — ✅ 通过
- `prefix_lm` — ✅ 通过
- `band_global_32` — ✅ 通过
- `global_local` — ✅ 通过（需 `BLOCK_M=32,BLOCK_N=32`）
- `random_block_sparse` — ✅ 通过（需预构建 kv_indices）

**不支持的模式（9 个）：**
- `nested`, `strided`, `dilated_window`, `checkerboard_64`, `hybrid_sparse`, `block_diagonal_64`, `uniform_doc_256`, `multiscale_dilated` — `bool_to_bool_rintmode`
- `alibi_causal` — `aten.index.Tensor` + FP score_mod

**msprof 表现：**
- op_total 高，helper % 正常或略高（10-15%）
- top op 为 MatMul（多个），不是单独的 FusedAttention
- task 数量多（每个 block 独立调度）

**分析：** NPU Triton adapter 不能有效利用 block sparsity 跳过 masked-out blocks，导致通用模板退化为若干小 matmul 的拼接。

**建议：** 开启 kernel-internal block reorder（`--enable-block-reorder`），观察 hit_rate 变化和 kernel 时间变化。

### 模式 E：Reorder 优化效果评估

**症状：** 开启了 `--enable-block-reorder`，想知道 reorder 是否生效

**判定标准：**
1. **hit_rate 变化**：看日志中 `Hit rate: X → Y`，如果 Y > X 说明重排改善了 KV block 连续性
2. **kernel 时间变化**：`Flex+wave_overlap` vs `Flex Attention` 的 avg ms 对比

**msprof 表现：**
- `Flex Attention` 和 `Flex+wave_overlap` 的时间差反映了 reorder 对 kernel 执行的影响
- 注意：重排计算（`compute_and_set_pending_perm`）在 CPU 上执行，**不在** msprof 采集范围内
- 如果 hit_rate 已经 100% 并且两个时间差异 < 2%：reorder 无额外收益（KV 访问已经连续）

```bash
# 完整的 reorder 效果评估流程
# 1. 先看 hit_rate 和基本时间
python3 flex_attention_run_script.py --shape 4,8,2048,128 \
  --enable-block-reorder --sparse-config global_local

# 2. 用 msprof 深入对比
python3 flex_attention_run_script.py --mode msprof \
  --shape 4,8,2048,128 --enable-block-reorder --target flex \
  --msprof-aic-metrics "Memory"
```

### 模式 B：fastpath 内辅助算子残留

**症状：** causal 场景，但 flex 仍然慢于 manual

**msprof 表现：**
- helper % 显著（>20%）
- top op types 含 Cast/Transpose/Arange

**分析：** 即使走了 fastpath，Triton 模板内仍有类型转换或形状变换操作。

**建议：** 检查 `compute_causal_flex_attention` 中是否有多余的 dtype cast，确认 Q/K/V 传入时已经是目标 dtype。

### 模式 C：AI CPU 回退

**症状：** msprof 报告中 AI CPU ms > 0

**msprof 表现：**
- AI CPU 列不为空
- 对应 op 在 Top Op Types 中标注了 AI CPU

**分析：** 某些算子（尤其是自定义 score_mod）不能完全下放到 AI Core，走 CPU 回退路径。

**建议：** 尽量使用 identity score_mod + 预定义 mask_mod。如需自定义，检查是否有 dtype 不匹配导致回退。

### 模式 D：小 shape launch 瓶颈

**症状：** 小 shape（seq_len ≤ 512）时 flex 慢，大 shape 正常

**msprof 表现：**
- host api 耗时占总时间比例高（>30%）
- task total 与 op total 接近

**分析：** kernel launch overhead 在小 shape 下被放大。

**建议：**
- 增大 `--block-m` / `--block-n`（如 128）
- 减少 warmup/repeat 次数
- 使用 `--static-compile` 避免重复编译

---

## 6. 端到端分析示例

```bash
# Step 1: 采集 msprof 数据
python3 flex_attention_run_script.py --mode msprof \
  --shape 4,8,2048,128 \
  --msprof-aic-metrics "ArithmeticUtilization,Memory"

# Step 2: 解析并生成报告
python3 summarize_msprof.py msprof_out/20260609_143022/

# Step 3: 查看快速判读
# 输出会在末尾的 "quick read" 部分给出第一印象

# Step 4: 如果 fastpath 未触发，检查原因
python3 flex_attention_run_script.py --shape 4,8,2048,128 2>&1 | grep debug

# Step 5: 根据诊断决策树定位瓶颈后，尝试修复或调整参数
```

你也可以在一次命令中快速获取性能概览和 msprof 报告：

```bash
# 先跑性能基准
python3 flex_attention_run_script.py --shape 4,8,2048,128

# 然后只采集 msprof（只分析 flex 侧）
python3 flex_attention_run_script.py --mode msprof \
  --shape 4,8,2048,128 --target flex
```