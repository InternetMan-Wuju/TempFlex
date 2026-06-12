# NPU Flex Attention Block Reorder 设计

- **日期**: 2026-06-08
- **状态**: 已批准

## 1. 背景

在 NPU 上运行 Flex Attention 时，block-sparse mask 决定了每个 Q block 访问哪些 KV blocks。默认情况下，这些 KV blocks 的排列顺序可能不具备良好的空间局部性，导致 L2 cache 利用率低下。

GPU 侧的 vllm-omni 项目已经实现了 40+ 种 block reorder 算法（`sparse-attn-source/`），通过在 GPU 上对 KV block 的访问顺序进行重排，可显著提升 cache hit rate 和注意力计算速度。

本设计将这些算法中最核心的 `wave_overlap_reorder` 移植到 NPU，使其集成到 `torch_npu` 的 flex attention 基准测试架构中。

## 2. 架构

### 数据流

```
BlockMask (kv_indices, kv_num_blocks)
    │
    ├─ compute_block_hit_rate() → 基线 hit rate
    │
    ├─ rebuild_block_mask() → 稠密 mask (H, MB, NB)
    │
    ├─ wave_overlap_reorder(mask_float, wave_size) → perm (H, MB)
    │
    ├─ 按 perm 重排 kv_indices/kv_num_blocks → 重排后的 BlockMask
    │
    ├─ compute_block_hit_rate() → 重排后 hit rate
    │
    └─ flex_attention(重排后 BlockMask) → 计时
```

### 新增文件

```
Newest/site-packages/torch_npu/_inductor/kernel/
    flex_attention_reorder.py    # [新建] 所有 reorder 逻辑
```

### 修改文件

```
flex_attention_run_script.py     # 恢复 reorder 支持
Newest/apply_newest.sh           # 加入 reorder 模块部署
```

## 3. `flex_attention_reorder.py` 详细设计

### 3.1 `compute_block_hit_rate(kv_indices, kv_num_blocks) → tuple[float, float]`

计算 KV block 访问的空间局部性。

**输入**:
- `kv_indices`: `Tensor[B, H, n_blocks, max_blocks]` — 每个 Q block 的可访问 KV block 索引
- `kv_num_blocks`: `Tensor[B, H, n_blocks]` — 每个 Q block 的有效 KV block 数量

**输出**: `(baseline_hit_rate, reordered_hit_rate)` — 0~1 的 hit rate

**算法**: 对每个 Q block 行，遍历其访问的 KV blocks。如果相邻 KV block index 差 1（内存连续），则为 cache hit；否则 cache miss。

### 3.2 `rebuild_block_mask(kv_num_blocks, kv_indices, n_blocks_q, n_blocks_kv) → mask_float`

将稀疏 block mask 重建为稠密矩阵用于重排算法。

**输入**:
- `kv_num_blocks`, `kv_indices`: 同 3.1
- `n_blocks_q`: Q 维度总 block 数（NQ）
- `n_blocks_kv`: KV 维度总 block 数（NK）

**输出**: `mask_float: Tensor[1, NQ, NK]` — 0/1 浮点矩阵

### 3.3 `wave_overlap_reorder(mask_float, wave_size=132) → Tensor`

1D Spectral Sorting via Power Iteration。纯 PyTorch 实现，移植自 GPU 的 `wave_overlap_reorder()`（见 greedy_reorder_cuda.py:2441）。

**输入**:
- `mask_float`: `Tensor[H, MB, NB]` — 稠密 block mask
- `wave_size`: int — 每 wave 包含的 block 数量（默认 132）

**输出**: `perm: Tensor[H, MB]` — 每行的 block 置换索引

**算法**:
1. 幂迭代 10 轮求 mask 矩阵的 top-1 左奇异向量（Fiedler 向量近似）：
   - 随机初始化 V: `(H, NB, 1)`
   - `U = mask @ V` → 归一化
   - `V = mask.T @ U` → 归一化
2. 计算投影值: `proj = mask @ V` → `(H, MB)`
3. 按投影值排序: `perm = argsort(proj, dim=1)`

### 3.4 `ReorderInfo` 数据类

```python
@dataclass
class ReorderInfo:
    perm: Tensor                    # (H, MB) 行置换
    kv_indices: Tensor              # 重排后的 kv_indices
    kv_num_blocks: Tensor           # 重排后的 kv_num_blocks
    full_kv_indices: Tensor | None  # 重排后的 full_kv_indices
    full_kv_num_blocks: Tensor | None  # 重排后的 full_kv_num_blocks
    query: Tensor | None            # 重排后的 query（按 Q block 顺序）
    baseline_hit_rate: float        # 重排前 hit rate
    reordered_hit_rate: float       # 重排后 hit rate
    reorder_mode: str               # 使用的重排算法名称
```

### 3.5 `reorder_flex_forward(q, k, v, kv_num_blocks, kv_indices, ...) → ReorderInfo`

主入口函数。

**输入**:
- `q, k, v`: QKV 张量
- `kv_num_blocks, kv_indices`: 原始 block mask
- `full_kv_num_blocks, full_kv_indices`: 完整 block mask（可选）
- `BLOCK_SIZE`: block size
- `mode`: 重排算法名称（默认 `"wave_overlap"`）
- `wave_size`: int（默认 132）
- `verbose`: bool

**流程**:
1. 从 kv_indices 重建稠密 mask
2. 调用 wave_overlap_reorder 获得置换
3. 按 perm 重排 kv_indices/kv_num_blocks 的行
4. 重排 Q 的行（如果需要）
5. 计算重排前后 hit rate
6. 返回 ReorderInfo

### 3.6 `REORDER_REGISTRY`

```python
REORDER_REGISTRY = {
    "wave_overlap": wave_overlap_reorder,
}
```

后续可扩展: `"fiedler_wave"`, `"kv_sort_only"`, `"idf_wave"` 等。

## 4. 判断标准

### Correctness（精度）

| 指标 | 标准 | 数值（bfloat16） |
|------|------|-----------------|
| allclose | flex vs manual | rtol=2e-2, atol=2e-2 |
| max_rel_diff | 最大相对差异 | ≤ 3% ✅, > 3% ❌ |
| fail_ratio | 超 tol 的元素占比 | ≤ 1% ✅ |
| 重排 vs baseline | reorder vs 原始 flex | 同一 rtol/atol |

### Profile（性能）

| 指标 | 含义 |
|------|------|
| flex_ms | 原始 Flex Attention 平均耗时 |
| flex_reorder_ms | 重排后 Flex Attention 平均耗时 |
| manual_ms | Manual Attention 平均耗时 |
| speedup_vs_baseline | flex_ms / flex_reorder_ms（>1 = 更快） |
| speedup_vs_manual | manual_ms / flex_reorder_ms（>1 = 更快） |

### Hit Rate（新增）

| 指标 | 含义 | 评判 |
|------|------|------|
| baseline_hit_rate | 原始 block mask 的 cache hit rate | — |
| reordered_hit_rate | 重排后的 cache hit rate | — |
| hit_rate_improvement | reordered - baseline | > 5% ✅ |

### 综合成功标准

- ✅ **精度不下降**: allclose=True 且重排 vs baseline 也 allclose=True
- ✅ **Hit rate 提升**: reordered_hit_rate > baseline_hit_rate + 5%
- ✅ **性能提升**: flex_reorder_ms < flex_ms * 0.95（至少快 5%）
- ⚠️ **异常**: hit rate 提升但性能下降 → 分析重排本身的开销

## 5. 文件修改清单

### `flex_attention_reorder.py`（新建）

模块结构:
- `compute_block_hit_rate()`
- `rebuild_block_mask()`
- `wave_overlap_reorder()`
- `ReorderInfo` 数据类
- `reorder_flex_forward()`
- `REORDER_REGISTRY`

### `flex_attention_run_script.py`（修改）

- 加入 `from flex_attention_reorder import ...`
- 在 `run_benchmark()` 中恢复 `--enable-block-reorder` 分支
- 确保 `--block-reorder-mode` 支持 `wave_overlap`
- 保存/输出 reorder 相关的性能指标（hit rate, reorder_ms）

### `apply_newest.sh`（修改）

- 加入 `flex_attention_reorder.py` 的 cp 命令

## 6. 后续扩展（不在本轮范围内）

- `fiedler_wave_reorder`: 用 Fiedler 向量代替 top-1 奇异向量
- `kv_sort_only`: 仅对每个 Q block 内的 KV index 排序（简单基线）
- `idf_wave_reorder`: 用 IDF 加权修正低频率行的影响
- CUDA/C 扩展: 当前纯 PyTorch，不需 CUDA