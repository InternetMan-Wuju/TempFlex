# NPU External Block Reorder 无效分析报告

> 日期: 2026-06-15 | 基于实测数据 + 架构对比分析

## 1. 现状

在 Ascend 910B NPU 上测试了 5 种 reorder 算法（wave_overlap, fiedler, fiedler_local, lexicographical, adaptive），覆盖：

| 维度 | 测试范围 |
|------|---------|
| 序列长度 | 32K, 49K, 65K, 80K, 98K |
| Pattern 类型 | 规则稀疏（8 种 + causal）+ 不规则随机 block-sparse（4 种密度 × 3 种子） |
| B/C 增益 | 全部在 **0.99x–1.02x** 范围内（测量噪音） |

### 算法确实生效了

在 S=81920, density=10.2%, n=640 blocks 上验证 perm 不是 identity：

```
算法            identity  原hit_rate  重排hit_rate  Δ
─────────────────────────────────────────────────────────
wave_overlap    False     0.2008      0.1885        -1.23%
fiedler         False     0.2008      0.1957        -0.51%
fiedler_local   False     0.2008      0.1939        -0.69%
```

三个算法都产生了非平凡排列（identity=False），但 KV 块局部性（相邻 Q block 之间共享的 KV column 比例）**反而全部下降**。这是 NPU 上 reorder 无效的直接量化证据。

## 2. 根因分析：GPU vs NPU 缓存架构根本差异

### 2.1 GPU 缓存架构（reorder 有效的基础）

NVIDIA H100 的内存层级：

```
HBM3 (80 GB, 3.35 TB/s)
    ↓ TMA 异步传输
L2 Cache (50 MB, ~200 cycles, 3.35 TB/s)
    ↓
L1/SMEM (228 KB/SM, ~30 cycles)
    ↓ 共享的 K/V tile 被多个 Q block 重复使用
Compute (Tensor Cores)
```

GPU 上 reorder 有效的核心机制：

1. **L1/SMEM 是用户可管理的 shared memory**。FlashAttention 显式地将 K/V tile 加载到 SMEM 中，然后对该 tile 对应的所有 Q block 做计算。
2. **Block reorder 将共享相同 K/V columns 的 Q rows 排在一起**。当多个连续 Q block 都访问同一批 KV blocks 时，K/V 数据一旦加载到 L1/SMEM 就可以被复用，不需要重新从 HBM 读取。
3. **HBM 带宽是瓶颈**。H100 的 3.35 TB/s 看起来高，但在大 model/long seq 下仍然紧张。减少 HBM → SMEM 的流量直接转化为加速。
4. **TMA (Tensor Memory Accelerator)** 支持异步预取，配合 block reorder 可以在计算当前 batch 时预取下一批。

**结论**：GPU 上 reorder 加速的本质 = **减少 HBM → L1/SMEM 的冗余 K/V 加载次数**。

### 2.2 NPU (Ascend 910B) 缓存架构

Da Vinci 架构的内存层级：

```
HBM2e (64 GB, ~1.2 TB/s)
    ↓
L2 统一缓冲区 (32 MB, 共享于 32 AI Cores)
    ↓
L1 Buffer (~1 MB/AI Core, scratchpad 模式)
    ↓
3D Cube Unit + Vector Unit
```

关键差异：

| 特性 | GPU H100 | NPU Ascend 910B | 对 reorder 的影响 |
|------|----------|-----------------|-------------------|
| L1 管理方式 | 用户可管理 SMEM + 硬件 cache | **Scratchpad（软件显式管理）** | 无透明缓存复用 |
| L2 大小 | 50 MB | 32 MB | 更小，更少缓存命中机会 |
| L2 特性 | 透明硬件 cache | 统一缓冲区（需显式管理） | 无自动缓存邻接数据 |
| HBM 带宽 | 3.35 TB/s | ~1.2 TB/s | 带宽更低，但非瓶颈 |
| 计算模型 | 线程级并行 (SIMT) | 数据流 (3D Cube + Vector) | 数据加载模式完全不同 |

### 2.3 为什么 NPU 上 hit rate 下降

实测数据表明 reorder 后 KV 块局部性反而降低了：

```
orig_hit_rate = 0.2008  (原生按序列顺序)
re_hit_rate   = 0.1885  (wave_overlap 重排后)
Δ = -1.23%
```

原因：

1. **NPU 的 flex_attention kernel 不依赖 L1/L2 缓存复用**。kernel 内部按 `full_kv_indices` 元数据显式指定每个 Q block 需要加载哪些 KV blocks。每次迭代从 HBM 加载独立的 K/V 数据，不存在「上一次加载的 KV 留在缓存中被下一轮复用」的机制。

2. **原生顺序本身就有自然局部性**。在 causal attention 中，相邻 token（Q block i 和 i+1）的 KV 可见范围几乎完全重叠（只差一个 block）。随机 sparse 时，这种自然邻接也存在——序列中相邻位置看到的 KV 集合高度重叠。

3. **Reorder 破坏了自然邻接**。重排后的 Q block 顺序中，相邻的 block 来自原始序列中不相邻的位置，它们的 KV 可见集合重叠度反而低于原生顺序。

4. **NPU 的 scratchpad L1 需要显式编程**。数据从 HBM → L2（32MB）→ L1（1MB）的传输由编译器（bishengir）静态调度，不存在 GPU 那种「数据留在 L1/SMEM 中直到被驱逐」的动态缓存行为。

### 2.4 图示对比

```
GPU H100 reorder 工作流 (有效):
┌─────────────────────────────────────────────────┐
│ 原始: Q₀ Q₁ Q₂ ...  → K/V 加载: A B A B A B ... │
│ Reorder后: Q₀ Q₂ Q₄ ... Q₁ Q₃ Q₅ ...           │
│            └─访问 KV集A──┘ └─访问 KV集B──┘      │
│ K/V 集A 一次加载到 SMEM → 多个 Q 复用 ✓          │
└─────────────────────────────────────────────────┘

NPU Ascend 910B reorder 工作流 (无效):
┌─────────────────────────────────────────────────┐
│ 原始: Q₀ Q₁ Q₂ ...                                │
│   Q₀→load KV[0..K₀], Q₁→load KV[0..K₁], ...     │
│   自然邻接: Qᵢ 和 Qᵢ₊₁ 的 KV 集高度重叠          │
│                                                   │
│ Reorder后: Q₅₉₉ Q₆₁₇ Q₆₀₈ ...                  │
│   每次加载独立的 KV blocks（显式full_kv_indices） │
│   无缓存复用 → 无加速                             │
│   邻接重叠反而更低 → hit_rate 下降                │
└─────────────────────────────────────────────────┘
```

## 追加记录：wave-based internal reorder 原型的 139 定位

日期：2026-06-16

本轮验证目标是绕开旧的 `program-level q_start from PERM buffer` 路线，尝试在 flex attention kernel 内实现 wave-based reorder。关键复现命令：

```bash
python3 /wyh/code/TempFlex/flex_attention_run_script.py \
  --shape 1,2,4096,128 \
  --sparse-config block_diagonal_64_bs \
  --target flex \
  --warmup 1 \
  --repeat 1 \
  --enable-block-reorder \
  --block-reorder-impl internal \
  --wave-size 2 \
  --no-causal-fastpath \
  --no-compare
```

已完成的隔离实验：

| 实验 | 结果 |
|------|------|
| PERM buffer 读取 `q_start` | `code 139` |
| wave template 内两个 Q tile body | `code 139` |
| 第二个 tile 只做 store | `code 139` |
| 第二个 tile no-op | `code 139` |
| single-tile debug，`q_start = wave_start` | `code 139` |
| single-tile debug，`q_start = wave_id` | `code 139` |
| 复用现有 pure block-sparse template 但走 reorder debug path | 后端表现不稳定，仍可出现 `code 139` |

结论：

`139` 是底层 native 崩溃，通常对应 `SIGSEGV`。在这些实验里，Python/Jinja 代码生成阶段已经通过，崩溃发生在 torch_npu/Triton/Ascend 后端编译或执行阶段。当前证据不支持继续在“单 Triton program 内多 Q tile / 动态 q_start”这条路径上投入。

后续建议：

1. 避免再把 PERM 写入 `FULL_KV_IDX` 或在 kernel 内动态读取 `q_start`。
2. 避免继续扩展当前 wave template 的内联双 tile body。
3. 如果继续做 internal reorder，优先尝试更保守的两阶段方案：保持单 tile kernel 稳定性，通过外层调度或分阶段 kernel 改变执行顺序。
4. 性能评估仍应聚焦非 causal 稀疏模式；causal 本身自然局部性强，不应作为 reorder 主收益目标。

## 追加记录：对齐专利实现后的 NPU 策略修正

日期：2026-06-16

重新阅读 `/wyh/code/TempFlex/sparse-attn-source/new-vllm-omni-for-sparse-attn-main/vllm_omni/diffusion/attention/backends/flash_attn.py` 后，专利/参考实现的实际执行链路是：

1. 根据 block mask 计算 `perm_all_heads`。
2. 用同一个 block-level perm 重排 Q blocks。
3. 用同一个 perm 重排 sparse mask rows / block-sparse metadata。
4. 调用稳定的 block-sparse attention kernel；kernel 内仍按连续 Q row 执行。
5. 对输出做 inverse reorder / scatter 回原始 Q block 顺序。

这说明 GPU 参考实现不是在 attention kernel 内动态读取 perm 来改变 `q_start`，而是外层重排输入和 metadata。NPU 当前应先实现并验证这个 patent-style external path，而不是继续推进 `program_id -> perm -> q_start` 或 wave template 内多 Q tile 的方案。

对应脚本调整：

- `flex_attention_run_script.py` 默认 `--block-reorder-impl external`。
- `reorder_80k_matrix.py` 默认测试 external，并把 baseline/reorder 拆成独立子进程。
- 新增 `--target reorder`，避免同一进程先编译 baseline 再编译 reorder 触发 NPU/Triton 139。
- 默认矩阵去掉 causal；causal 仅作为显式测试项，默认跳过 reorder。

小尺寸验证：

```bash
python3 /wyh/code/TempFlex/reorder_80k_matrix.py \
  --seq-lens 4096 \
  --configs block_diagonal_64_bs \
  --warmup 1 \
  --repeat 1 \
  --timeout 300
```

结果：baseline `0.442 ms`，external reorder `0.446 ms`，非 identity perm，均正常返回。该 smoke 只证明实现链路和进程隔离稳定，不代表 80K 性能结论。

## 追加记录：80K external reorder 首轮结果

日期：2026-06-16

为了避开普通 baseline 在长序列 pure block-sparse 路径上的 139，测试矩阵新增了 `identity_external` baseline：它使用与 patent-style reorder 相同的外层 Q/metadata gather 路径，但 perm 固定为 identity。这样可以把对比限定为“identity row order vs reordered row order”，避免 baseline/reorder 走不同编译路径。

最终保留的验证输出：

- `/wyh/code/TempFlex/docs/reorder_80k_strided_wave_kv_snake_inv_repeat8.md`

关键结果：

| 模式 | 长度 | baseline | reorder | 结论 |
|------|------|----------|---------|------|
| `sliding_window_128_bs` + `wave_overlap` | 81920 | 2.526 ms | 2.565 ms | 约 -1.5%，未达标 |
| `dilated_window_bs` + `wave_overlap` | 81920 | 3.407 ms | 3.427 ms | 约 -0.6%，未达标 |
| `block_diagonal_64_bs` + `wave_overlap` | 81920 | 1.686 ms | 1.651 ms | 单次 batch 约 +2.1%，但 repeat=8 确认未稳定复现 |
| `block_diagonal_64_bs` + `wave_overlap` | 65536 | 1.374 ms | 1.381 ms | 约 -0.5% |
| `block_diagonal_64_bs` + `wave_overlap` | 98304 | 1.930 ms | 1.937 ms | 约 -0.4% |
| `sliding_window_128_bs` + `fiedler` | 81920 | 2.542 ms | 2.550 ms | 约 -0.3% |
| `block_diagonal_64_bs` + `fiedler` | 81920 | 1.698 ms | 1.707 ms | 约 -0.5% |
| `lexicographical` | 81920 | 正常 | identity | 算法判定无需重排 |
| `sliding_window_128_bs` + `adaptive` | 81920 | 2.604 ms | identity | 修复 tuple 处理后算法判定无需重排 |
| `strided_bs` + `wave_overlap` + `snake_inv` KV order | 81920 | 140.632 ms | 140.256 ms | 最新 repeat=8 约 +0.27%，未稳定达标 |

不稳定模式：

- `strided_bs`、`nested_bs` 的 identity baseline 在 80K 仍会 139，但非 identity reorder 能跑，暂时不能计算可信 speedup。
- `hybrid_sparse_bs` 的 identity baseline 能跑，但 `wave_overlap` reorder 会 139。
- `adaptive` 原先返回 `(perm, reason)`，脚本已修复 tuple 处理；修复后 `sliding_window_128_bs@81920` 返回 identity。

当前结论：

external patent-style reorder 已经实现并能在 80K 跑通。单纯 row reorder 在多数稳定可对比模式上没有稳定达到 1%；补齐 KV column order 后，`strided_bs@81920` 使用 `wave_overlap + snake_inv` 曾出现过 1% 左右提升，但修复 NPU `IndexPut` 后最新 repeat=8 为 `1.0027x`。当前结论应调整为：该方向有 device 层改善信号，但还没有稳定达到 80K 1%。

## 追加记录：从专利实现继续可迁移的优化点

日期：2026-06-16

重新梳理 `sparse-attn-source` 中的专利/参考实现后，当前可迁移性排序如下：

| 参考实现分支 | NPU 当前可迁移性 | 判断 |
|--------------|------------------|------|
| `boundary_dp` KV order | 高 | 不改变 kernel 模板，只改变 FULL_KV metadata 中每个 wave 的 KV asc/desc 方向 |
| `edge_dp` KV order | 中 | 已按 CPU 侧实现迁移，避免 AICPU `scatter/indexput`，但目前不优于 `snake_inv` |
| `banded_union_wave` / `marginal_union_wave` | 中 | 目标函数适合搬：最小化 wave-level KV union；Python/CPU overhead 需要控制 |
| `auction_union_reorder_cuda + start_first` | 低到中 | 专利当前最强组合，但 CUDA extension 不能直接用于 NPU；可搬算法思想，不搬 CUDA kernel |
| `global_greedy` / `banded_window_greedy` | 低 | row-to-row overlap 不等于 FA kernel 真正受益的 wave-level KV union |
| Spectral / Fiedler / Morton | 低 | 更适合结构诊断，不适合作为 NPU 默认 solver |

已落地：`boundary_dp` 已移植到 `flex_attention_run_script.py::apply_kv_order_to_full_metadata`，CLI 支持 `--kv-order boundary_dp`，80K standalone 结果：

| 模式 | 长度 | baseline | reorder | 结论 |
|------|------|----------|---------|------|
| `strided_bs` + `wave_overlap` + `boundary_dp` | 81920 | 141.032 ms | 139.467 ms | `1.0112x` kernel-only speedup |

repeat=20 复测后，`boundary_dp` 的稳定收益没有保持 1%：

| KV order | 长度 | baseline | reorder | 结论 |
|----------|------|----------|---------|------|
| `snake_inv` | 81920 | 140.493 ms | 139.805 ms | `1.0049x` |
| `boundary_dp` | 81920 | 140.492 ms | 140.088 ms | `1.0029x` |
| `edge_dp` | 81920 | 140.666 ms | 140.173 ms | `1.0035x` |

这个结果说明 msprof 里观察到的“核心 kernel 对 KV order 敏感”仍然成立，但 `boundary_dp` 不能直接替代 `snake_inv` 作为默认策略。限制是当前 `boundary_dp` 在 CPU 上重建 bool mask 并做 DP，单次预处理约 29 ms；因此它现在是 kernel-order 实验候选，不是端到端生产实现。生产化方向应是：

1. 对固定/重复 sparse pattern 缓存 `perm + FULL_KV metadata`。
2. 把 selector 限制在 `strided_bs@65536+` 这类已验证收益模式，默认先选 `snake_inv`，保留 `boundary_dp` 为实验开关。
3. 后续尝试轻量 `banded_union_wave` row perm，把“wave-level KV union”目标搬进 NPU host 侧 permutation 生成器。
4. 继续避免 internal dynamic `q_start` 路线，保持单 tile kernel 稳定性。

已实现初版 selector：`flex_attention_run_script.py --npu-reorder-selector`。当前策略是只对 `strided_bs@65536+` 自动选择 `external + wave_overlap + snake_inv`，对 causal、短序列 `strided_bs`、`sliding_window_128_bs`、`dilated_window_bs`、`hybrid_sparse_bs` 以及其它未验证模式跳过 reorder。smoke 验证：

- `strided_bs@4096`：selector 跳过 reorder。
- `strided_bs@81920`：selector 自动选择 `wave_overlap + snake_inv` 并成功运行。

本轮继续迁移并筛选了两个 NPU 适配分支：

- `edge_dp`：已实现为 CPU 侧 KV edge-set DP，4K/32K/80K 均跑通；80K repeat=8 为 `1.0035x`，不优于 `snake_inv`，暂不进 selector 默认。
- `banded_union_wave` / `banded_union_fast`：已实现为 host 侧 row packing 研究模式；32K `banded_union_fast` kernel `23.112 ms`，慢于 `snake_inv`，且 CPU permutation 约 `893 ms`；80K 朴素 `banded_union_wave` 超过 2 分钟未完成后中断。结论是目标函数适合迁移，但必须做 bitset/Numba/C++ 或离线缓存，不能用当前 Python 循环直接进热路径。

## 追加记录：普通 no-reorder vs external reorder

日期：2026-06-16

之前的 80K reorder 表主要使用 `identity external` baseline。这个 baseline 的意义是让 baseline 和 reorder 走同一条 external Q/metadata 路径，只改变 row/KV order，便于做稳定 A/B；它不等于完全不开 reorder。

补充测试了真正不开 reorder 的普通 flex 路径：

| S | 普通 no-reorder (`--target flex`) | external reorder (`wave_overlap + snake_inv`) | 结论 |
|---:|----------------------------------:|-----------------------------------------------:|------|
| 81920 | `ERR(137)` | 139.404 ms | 普通 no-reorder 被 kill，无法直接比较 |
| 32768 | `ERR(139)` | 22.802 ms | 普通 no-reorder segfault，无法直接比较 |
| 16384 | 6.210 ms | 6.039 ms | reorder kernel-only `1.0283x` |

因此，当前 80K 下不能得到“普通 no-reorder vs reorder”的直接速度比；只能使用 identity external 做稳定 kernel A/B。16K 是目前能直接比较普通 no-reorder 与 external reorder 的最大已验证点，结果显示 reorder 的 flex_attention 执行时间约快 2.8%。

全模式 16K 对比已整理到 `docs/reorder_vs_no_reorder_16k.md`。大多数 FULL_KV 模式下 external reorder 路径快于普通 no-reorder 路径；`block_diagonal_64_bs` 略慢，`checkerboard_64_bs` 的普通 no-reorder 路径崩溃，`global_local_bs` 提升最大但可能包含模板路径差异。

补充 profiling 观察：

- `--mode profile-target` 原先没有加载 `--sparse-config`，导致 msprof 下 sparse/reorder profile 跑偏；已修复为复用 benchmark 的 sparse-config 解析逻辑。
- external reorder 原先在 NPU 上用 `IndexPut` 计算 inverse permutation，msprof 下会触发 AICPU 异常；已改为 CPU 计算 `inv_perm` 后拷到 NPU。
- `strided_bs@32768` 的 msprof 显示，`wave_overlap + snake_inv` 后核心 `triton_tem_fused_0` 平均从 `22.889 ms` 降到 `22.536 ms`，说明 KV order 确实改变了 device kernel 行为；但该改善尚未在 80K repeat=8 上稳定扩大到 1%。

## 3. 为什么专利（GPU）上 80K 有 1% 增益

专利 `adaptive_spectral_reorder` 的选择器逻辑：

```python
if row_cv < 0.03 and col_cv < 0.03:
    return identity, reason="uniform_mask"
```

这个阈值是**针对 GPU SM 架构的 NCU 数据校准的**。GPU 上：

- row_cv > 0.03 意味着 NNZ 分布不均匀 → 不同 SM 间负载不均衡
- reorder 通过 wave scheduling 均衡 SM 负载（greedy_reorder_cuda.py 的拍卖波分配算法）
- 80K+ 时 HBM 带宽压力足够大，1% 的 HBM 流量减少就能转化为 1% 的 wall-clock 加速

**在 NPU 上**：
- NPU 的 AI Core 间负载由 CANN runtime 调度，不受 block-level row order 影响
- HBM 带宽压力到 100K 也未达到瓶颈（单次 attention 的 KV 数据量约 GB 级，HBM 1.2 TB/s 绰绰有余）
- 关键是 **L1/L2 的 scratchpad 模型**——数据复用必须编译器在 compile time 静态分析决定，reorder 改变不了生成的 kernel 的数据加载模式

## 4. 结论

1. **External block reorder 在 NPU 上不是普遍有效优化**。多数规则 sparse pattern 上没有稳定收益，部分模式会回退或 139；但 `strided_bs` 上补齐 KV order 后有真实 kernel-only 正向信号。

2. **主要加速仍来自 PURE_BLOCK_SPARSE 模板**——跳过无效 KV blocks；reorder 目前只是额外的 kernel-order 微调，80K repeat=20 稳定收益约 `0.3%-0.5%`。

3. **普通 no-reorder 长序列路径不稳定**。`strided_bs@81920` 普通 `--target flex` 会 `ERR(137)`，`32768` 会 `ERR(139)`；因此 80K 当前只能用 `identity external` 做稳定 A/B，而不是直接和普通 no-reorder 比。

4. **kernel 内部动态 q_start / wave template 路线仍不建议继续硬推**。已多次复现 139；当前更稳的是 patent-style external Q/metadata reorder，加 selector，并尽量缓存 metadata。

5. **专利算法不能简单全量搬运**。`snake_inv`、`boundary_dp`、`edge_dp` 这类 KV order 可以低风险迁移；`banded_union/auction` 目标函数有价值，但朴素 Python host 实现开销太大，需要 bitset/Numba/C++ 或离线缓存。

## 5. 关于 32K vs 100K

之前 32K 不能跑是因为 Triton compile worker 找不到 `libtorch_npu.so`（`LD_LIBRARY_PATH` 未在 import torch 前设置）。修复后 100K 可以正常编译和运行——没有序列长度硬限制，内存也充足（60GB空闲）。现在唯一的限制是 32K 以上的路径 C 同进程双编译 crash（需 subprocess 隔离绕过，已实现）。
