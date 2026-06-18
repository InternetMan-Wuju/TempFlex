# Optimized Fair Newest Reorder Matrix: 80K

- Method: `Newest without reorder = identity external`, `Newest with reorder = optimized/fallback external reorder`
- Shape: `B=1, H=2, D=128`
- Kernel block: `BLOCK_M=64, BLOCK_N=64`
- Timing: `warmup=3, repeat=5`

| Seq Len | Config | Mode | KV order | Without reorder | With reorder | Speedup | allclose | max_abs | Note |
|--------:|--------|------|----------|----------------:|-------------:|--------:|----------|--------:|------|
| 81920 | `causal` | `SKIP` | `-` | - | - | -x | - | - | skip: causal uses causal fastpath; reorder is not enabled |
| 81920 | `block_diagonal_64_bs` | `wave_overlap` | `snake_inv` | 1.669 | 1.660 | 1.0054x | True | 0.000000 | fallback: no validated auction/exact-path win for this 80K config |
| 81920 | `checkerboard_64_bs` | `wave_overlap` | `snake_inv` | 277.973 | 279.488 | 0.9946x | True | 0.000244 | fallback: no validated auction/exact-path win for this 80K config |
| 81920 | `sliding_window_128_bs` | `wave_overlap` | `snake_inv` | 2.547 | 2.531 | 1.0063x | True | 0.003906 | fallback: no validated auction/exact-path win for this 80K config |
| 81920 | `strided_bs` | `auction_union_exact_path` | `snake_inv` | 140.296 | 140.328 | 0.9998x | True | 0.003906 | 80K rerun: exact path completed but no speedup |
| 81920 | `dilated_window_bs` | `wave_overlap` | `snake_inv` | 3.409 | 3.422 | 0.9962x | True | 0.003906 | 80K rerun: fallback wave_overlap completed but regressed |
| 81920 | `nested_bs` | `auction_union_exact_path` | `snake_inv` | 72.557 | 72.218 | 1.0047x | True | 0.000977 | 80K rerun: exact path completed with small speedup |
| 81920 | `hybrid_sparse_bs` | `wave_overlap` | `snake_inv` | 98.913 | 98.554 | 1.0036x | True | 0.001953 | 80K fallback: exact_path+union_boundary_dp still crashed; wave_overlap+snake_inv completed |
| 81920 | `global_local_bs` | `wave_overlap` | `snake_inv` | 5.995 | 5.979 | 1.0027x | True | 0.001953 | fallback: no validated auction/exact-path win for this 80K config |
| 81920 | `multiscale_dilated_bs` | `wave_overlap` | `snake_inv` | 5.134 | 5.117 | 1.0033x | True | 0.001953 | fallback: no validated auction/exact-path win for this 80K config |
| 81920 | `prefix_lm_bs` | `wave_overlap` | `snake_inv` | 279.565 | 277.726 | 1.0066x | True | 0.001953 | fallback: no validated auction/exact-path win for this 80K config |
| 81920 | `band_global_bs` | `auction_union_exact_path` | `snake_inv` | 5.982 | 5.994 | 0.9980x | True | 0.001953 | 80K optimized probe: exact path plus stable snake_inv |
