# Optimized Fair Newest Reorder Matrix

- Method: `Newest without reorder = identity external`, `Newest with reorder = optimized external reorder`
- Shape: `B=1, H=2, D=128`
- Kernel block: `BLOCK_M=64, BLOCK_N=64`
- Timing: `warmup=3, repeat=5`

| Seq Len | Config | Mode | KV order | Without reorder | With reorder | Speedup | allclose | max_abs | Note |
|--------:|--------|------|----------|----------------:|-------------:|--------:|----------|--------:|------|
| 32768 | `causal` | `SKIP` | `-` | - | - | -x | - | - | skip: causal uses causal fastpath; reorder is not enabled |
| 32768 | `block_diagonal_64_bs` | `wave_overlap` | `snake_inv` | 0.859 | 0.846 | 1.0154x | True | 0.000000 | fallback: no validated auction win for this 32K config |
| 32768 | `checkerboard_64_bs` | `wave_overlap` | `snake_inv` | 45.092 | 45.453 | 0.9921x | True | 0.000244 | fallback: no validated auction win for this 32K config |
| 32768 | `sliding_window_128_bs` | `wave_overlap` | `snake_inv` | 1.214 | 1.189 | 1.0210x | True | 0.003906 | fallback: no validated auction win for this 32K config |
| 32768 | `strided_bs` | `auction_union_fast` | `snake_inv` | 23.227 | 22.766 | 1.0202x | True | 0.001953 | 32K optimized probe: auction_union_fast repeat=20 candidate |
| 32768 | `dilated_window_bs` | `wave_overlap` | `snake_inv` | 1.533 | 1.543 | 0.9935x | True | 0.003906 | fallback: no validated auction win for this 32K config |
| 32768 | `nested_bs` | `auction_union_fast` | `snake_inv` | 12.637 | 12.402 | 1.0189x | True | 0.001953 | 32K optimized probe: auction_union_fast repeat=20 candidate |
| 32768 | `hybrid_sparse_bs` | `auction_union_fast` | `union_boundary_dp` | 16.903 | 16.669 | 1.0140x | True | 0.000977 | 32K optimized probe: hybrid best KV orientation |
| 32768 | `global_local_bs` | `wave_overlap` | `snake_inv` | 2.567 | 2.553 | 1.0055x | True | 0.001953 | fallback: no validated auction win for this 32K config |
| 32768 | `multiscale_dilated_bs` | `wave_overlap` | `snake_inv` | 2.227 | 2.211 | 1.0072x | True | 0.001953 | fallback: no validated auction win for this 32K config |
| 32768 | `prefix_lm_bs` | `wave_overlap` | `snake_inv` | 45.955 | 45.428 | 1.0116x | True | 0.000488 | fallback: no validated auction win for this 32K config |
| 32768 | `band_global_bs` | `wave_overlap` | `snake_inv` | 2.567 | 2.595 | 0.9892x | True | 0.001953 | fallback: no validated auction win for this 32K config |
