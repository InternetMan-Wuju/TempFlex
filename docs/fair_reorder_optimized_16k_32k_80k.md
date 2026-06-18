# Optimized Fair Newest Reorder Matrix: 16K / 32K / 80K

- Method: `Newest without reorder = identity external`, `Newest with reorder = optimized/fallback external reorder`
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
