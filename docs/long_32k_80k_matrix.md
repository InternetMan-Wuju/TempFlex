# Long Sequence 32K/80K Matrix

- Shape: `B=1, H=2, S, D=128`
- Raw: 原长序列矩阵数据
- Newest without/with reorder: optimized/fallback fair A/B, `identity external` vs `optimized/fallback reorder external`, `warmup=3, repeat=5`

| Seq Len | Config | Raw | Newest (without reorder) | Newest (with reorder) | Speedup | allclose | 结论 |
|--------:|--------|----:|-------------------------:|----------------------:|--------:|----------|------|
| 32768 | `causal` | 50.059 | 47.401 | SKIP | - | - | causal 不启用 reorder; Newest/Raw 1.0561x |
| 32768 | `block_diagonal_64_bs` | CRASH | 0.859 | 0.846 | 1.0154x | True | 明确收益 |
| 32768 | `checkerboard_64_bs` | CRASH | 45.092 | 45.453 | 0.9921x | True | reorder 变慢 |
| 32768 | `sliding_window_128_bs` | CRASH | 1.214 | 1.189 | 1.0210x | True | 明确收益 |
| 32768 | `strided_bs` | CRASH | 23.227 | 22.766 | 1.0202x | True | 明确收益 |
| 32768 | `dilated_window_bs` | CRASH | 1.533 | 1.543 | 0.9935x | True | reorder 变慢 |
| 32768 | `nested_bs` | CRASH | 12.637 | 12.402 | 1.0189x | True | 明确收益 |
| 32768 | `hybrid_sparse_bs` | CRASH | 16.903 | 16.669 | 1.0140x | True | 明确收益 |
| 32768 | `global_local_bs` | CRASH | 2.567 | 2.553 | 1.0055x | True | 小幅收益 |
| 32768 | `multiscale_dilated_bs` | CRASH | 2.227 | 2.211 | 1.0072x | True | 小幅收益 |
| 32768 | `prefix_lm_bs` | CRASH | 45.955 | 45.428 | 1.0116x | True | 明确收益 |
| 32768 | `band_global_bs` | CRASH | 2.567 | 2.595 | 0.9892x | True | reorder 变慢 |
| 81920 | `causal` | ERR(-9) | ERR(-9) | SKIP | - | - | causal 不启用 reorder; 80K 原矩阵未跑通 |
| 81920 | `block_diagonal_64_bs` | ERR(-9) | 1.669 | 1.660 | 1.0054x | True | 小幅收益 |
| 81920 | `checkerboard_64_bs` | ERR(-9) | 277.973 | 279.488 | 0.9946x | True | reorder 变慢 |
| 81920 | `sliding_window_128_bs` | ERR(-9) | 2.547 | 2.531 | 1.0063x | True | 小幅收益 |
| 81920 | `strided_bs` | ERR(-9) | 140.296 | 140.328 | 0.9998x | True | 基本持平 |
| 81920 | `dilated_window_bs` | ERR(-9) | 3.409 | 3.422 | 0.9962x | True | reorder 变慢 |
| 81920 | `nested_bs` | ERR(-9) | 72.557 | 72.218 | 1.0047x | True | 小幅收益 |
| 81920 | `hybrid_sparse_bs` | ERR(-9) | 98.913 | 98.554 | 1.0036x | True | 小幅收益 |
| 81920 | `global_local_bs` | ERR(-9) | 5.995 | 5.979 | 1.0027x | True | 基本持平 |
| 81920 | `multiscale_dilated_bs` | ERR(-9) | 5.134 | 5.117 | 1.0033x | True | 小幅收益 |
| 81920 | `prefix_lm_bs` | ERR(-9) | 279.565 | 277.726 | 1.0066x | True | 小幅收益 |
| 81920 | `band_global_bs` | ERR(-9) | 5.982 | 5.994 | 0.9980x | True | 基本持平 |
