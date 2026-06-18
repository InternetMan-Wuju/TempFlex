# Optimized Fair Newest Reorder Matrix

- Method: `Newest without reorder = identity external`, `Newest with reorder = optimized external reorder`
- Shape: `B=1, H=2, D=128`
- Kernel block: `BLOCK_M=64, BLOCK_N=64`
- Timing: `warmup=3, repeat=5`

| Seq Len | Config | Mode | KV order | Without reorder | With reorder | Speedup | allclose | max_abs | Note |
|--------:|--------|------|----------|----------------:|-------------:|--------:|----------|--------:|------|
| 16384 | `causal` | `SKIP` | `-` | - | - | -x | - | - | skip: causal uses causal fastpath; reorder is not enabled |
| 16384 | `block_diagonal_64_bs` | `wave_overlap` | `snake_inv` | 0.588 | 0.570 | 1.0316x | True | 0.000000 | fallback: no validated auction win for this 16K config |
| 16384 | `checkerboard_64_bs` | `wave_overlap` | `snake_inv` | 11.689 | 11.793 | 0.9912x | True | 0.000488 | fallback: no validated auction win for this 16K config |
| 16384 | `sliding_window_128_bs` | `wave_overlap` | `snake_inv` | 0.749 | 0.751 | 0.9973x | True | 0.003906 | fallback: no validated auction win for this 16K config |
| 16384 | `strided_bs` | `auction_union_fast` | `snake_inv` | 6.247 | 6.028 | 1.0363x | True | 0.001953 | 16K optimized probe: auction_union_fast on reorder-positive config |
| 16384 | `dilated_window_bs` | `wave_overlap` | `snake_inv` | 0.925 | 0.911 | 1.0154x | True | 0.003906 | fallback: no validated auction win for this 16K config |
| 16384 | `nested_bs` | `auction_union_fast` | `snake_inv` | 3.721 | 3.616 | 1.0290x | True | 0.001953 | 16K optimized probe: auction_union_fast on reorder-positive config |
| 16384 | `hybrid_sparse_bs` | `auction_union_fast` | `snake_inv` | 4.866 | 4.773 | 1.0195x | True | 0.001953 | 16K optimized probe: auction_union_fast on reorder-positive config |
| 16384 | `global_local_bs` | `wave_overlap` | `snake_inv` | 1.436 | 1.439 | 0.9979x | True | 0.001953 | fallback: no validated auction win for this 16K config |
| 16384 | `multiscale_dilated_bs` | `wave_overlap` | `snake_inv` | 1.274 | 1.265 | 1.0071x | True | 0.001953 | fallback: no validated auction win for this 16K config |
| 16384 | `prefix_lm_bs` | `auction_union_fast` | `snake_inv` | 12.043 | 11.678 | 1.0313x | True | 0.001953 | 16K optimized probe: auction_union_fast on reorder-positive config |
| 16384 | `band_global_bs` | `wave_overlap` | `snake_inv` | 1.429 | 1.439 | 0.9931x | True | 0.001953 | fallback: no validated auction win for this 16K config |
