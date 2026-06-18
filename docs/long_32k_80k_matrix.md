# Long Sequence 32K/80K Matrix

- Shape: `B=1, H=2, S, D=128`
- Raw/Newest/Reorder: `warmup=3, repeat=5`

| Seq Len | Config | Raw | Newest (without reorder) | Newest (with reorder) | Speedup | allclose | 结论 |
|--------:|--------|----:|-------------------------:|----------------------:|--------:|----------|------|
| 32768 | `causal` | 50.059 | 47.401 | SKIP | - | - | causal 不启用 reorder; Newest/Raw 1.0561x |
| 32768 | `block_diagonal_64_bs` | CRASH | 0.862 | 0.848 | 1.0165x | True | 明确收益 |
| 32768 | `checkerboard_64_bs` | CRASH | 45.242 | 45.445 | 0.9955x | True | reorder 略慢 |
| 32768 | `sliding_window_128_bs` | CRASH | 1.204 | 1.204 | 1.0000x | True | 基本持平 |
| 32768 | `strided_bs` | CRASH | 23.404 | 22.949 | 1.0198x | True | 明确收益 |
| 32768 | `dilated_window_bs` | CRASH | 1.540 | 1.539 | 1.0006x | True | 基本持平 |
| 32768 | `nested_bs` | CRASH | 12.648 | 12.644 | 1.0003x | True | 基本持平 |
| 32768 | `hybrid_sparse_bs` | CRASH | 17.161 | 16.661 | 1.0300x | True | 明确收益 |
| 32768 | `global_local_bs` | CRASH | 2.566 | 2.572 | 0.9977x | True | 基本持平 |
| 32768 | `multiscale_dilated_bs` | CRASH | 2.257 | 2.227 | 1.0135x | True | 明确收益 |
| 32768 | `prefix_lm_bs` | CRASH | 46.196 | 45.232 | 1.0213x | True | 明确收益 |
| 32768 | `band_global_bs` | CRASH | 2.611 | 2.619 | 0.9969x | True | reorder 略慢 |
| 81920 | `causal` | ERR(-9) | ERR(-9) | SKIP | - | - | causal 不启用 reorder; 80K 原矩阵未跑通 |
| 81920 | `block_diagonal_64_bs` | ERR(-9) | 1.651 | 1.657 | 0.9964x | True | reorder 略慢 |
| 81920 | `checkerboard_64_bs` | ERR(-9) | 280.581 | 280.960 | 0.9987x | True | 基本持平 |
| 81920 | `sliding_window_128_bs` | ERR(-9) | 2.552 | 2.540 | 1.0047x | True | 小幅收益 |
| 81920 | `strided_bs` | ERR(-9) | 140.613 | 140.369 | 1.0017x | True | 基本持平 |
| 81920 | `dilated_window_bs` | ERR(-9) | 3.405 | 3.467 | 0.9821x | True | reorder 变慢 |
| 81920 | `nested_bs` | ERR(-9) | 72.539 | 72.044 | 1.0069x | True | 小幅收益 |
| 81920 | `hybrid_sparse_bs` | ERR(-9) | 98.857 | 98.357 | 1.0051x | True | 小幅收益 |
| 81920 | `global_local_bs` | ERR(-9) | 6.059 | 6.043 | 1.0026x | True | 基本持平 |
| 81920 | `multiscale_dilated_bs` | ERR(-9) | 5.112 | 5.149 | 0.9928x | True | reorder 略慢 |
| 81920 | `prefix_lm_bs` | ERR(-9) | 280.546 | 280.162 | 1.0014x | True | 基本持平 |
| 81920 | `band_global_bs` | ERR(-9) | 6.071 | 5.991 | 1.0134x | True | 明确收益 |
