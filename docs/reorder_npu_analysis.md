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

1. **External block reorder 在 NPU 上对规则/不规则 sparse pattern 均无效**——不是实现 bug，是 NPU 架构决定的。实测 hit rate 反而下降了。

2. **加速 100% 来自 PURE_BLOCK_SPARSE 模板**——跳过无效 KV blocks。

3. **唯一可能有价值的 reorder 路径是 kernel 内部实现**——修改 flex_attention kernel 模板中的 Q block 迭代循环的遍历顺序，让编译器在生成 kernel 时可以静态分析数据局部性并优化 L1 buffer 分配。

4. **专利的算法都是针对 GPU SM/SMEM 架构优化的**，移植到 NPU 需要重新设计数据加载策略而非简单搬运外部重排算法。

## 5. 关于 32K vs 100K

之前 32K 不能跑是因为 Triton compile worker 找不到 `libtorch_npu.so`（`LD_LIBRARY_PATH` 未在 import torch 前设置）。修复后 100K 可以正常编译和运行——没有序列长度硬限制，内存也充足（60GB空闲）。现在唯一的限制是 32K 以上的路径 C 同进程双编译 crash（需 subprocess 隔离绕过，已实现）。