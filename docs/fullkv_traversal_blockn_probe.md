# FULL_KV Traversal / BLOCK_N Probe

目的：验证 80K 当前瓶颈是否更接近 FULL_KV template traversal，而不是继续扩大 row reorder 搜索空间。

## 结论

- `BLOCK_N=128` 对 identity external 有明显速度信号：`hybrid_sparse_bs@81920` 从 `98.951 ms` 降到 `62.938 ms`。
- 但 `BLOCK_N=128` 不是可直接启用的 reorder 方案：`hybrid_sparse_bs@32768` reorder crash，`hybrid_sparse_bs@81920` reorder 只有 `0.9993x`，没有收益。
- 小 shape 输出对比显示 tile 改变本身可以保持数值一致：`hybrid_sparse_bs@4096` 的 `BLOCK_N=64` vs `BLOCK_N=128` identity external 输出 `allclose=True`，`max_abs=0.001953125`。
- 因此下一步应保留默认 `BLOCK_N=64` 的稳定路线，优化 `get_offset_for_next_block()` / FULL_KV metadata traversal，而不是直接把 `BLOCK_N=128` 加入 selector。

## Probe 结果

配置：`B=1, H=2, D=128`, `hybrid_sparse_bs`, `auction_union_exact_path + union_boundary_dp`, `warmup=3, repeat=5`。

| Seq Len | BLOCK_N | Identity ms | Reorder ms | Speedup | allclose | max_abs | 结论 |
|--------:|--------:|------------:|-----------:|--------:|----------|--------:|------|
| 32768 | 64 | 17.136 | 16.667 | 1.0281x | True | 0.000977 | reorder 有收益 |
| 81920 | 64 | 98.951 | CRASH | - | - | - | 80K 稳定性不足 |
| 32768 | 128 | 10.881 | CRASH | - | - | - | tile 变快但 reorder 不稳定 |
| 81920 | 128 | 62.938 | 62.981 | 0.9993x | True | 0.000488 | identity 明显变快，但 reorder 无收益 |

## Template 含义

当前 FULL_KV 路径默认 `SPARSE_KV_BLOCK_SIZE=128, BLOCK_N=64`，因此 `SPARSE_KV_MULTIPLE=2`。`forward_inner()` 每个 `BLOCK_N` step 都调用 `get_offset_for_next_block()`，而该 helper 会读取当前/下一个 `FULL_KV_IDX` 来计算 K/V pointer advance。

`BLOCK_N=128` 将 `SPARSE_KV_MULTIPLE` 变成 `1`，减少了一个 KV block 内的半块 traversal 过程，所以 identity external 速度大幅下降是合理信号。但它同时改变 matmul tile shape、register/UB 压力和编译路径，当前对 reorder 不够稳定。

## 下一步

1. 在默认 `BLOCK_N=64` 下专门优化 `get_offset_for_next_block()`，减少非 sparse-block 边界 step 的冗余 metadata load。
2. 考虑增加 FULL_KV 专用 traversal：外层按 `FULL_KV_IDX` block 遍历，内层静态处理两个 `BLOCK_N=64` subtile，只在 sparse block 边界读取 next block。
3. 80K msprof 继续用小 repeat/分段采集，避免 `exitCode:11` 中断导致没有 device summary。
