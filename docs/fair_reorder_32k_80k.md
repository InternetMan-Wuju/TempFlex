# Long Reorder Correctness

- Method: identity external output vs wave_overlap reorder output
- Tolerance: `rtol=0.03, atol=0.03`

| Seq Len | Config | Identity | Reorder | allclose | max_abs | mean_abs | max_rel | Identity ms | Reorder ms |
|--------:|--------|----------|---------|----------|--------:|---------:|--------:|------------:|-----------:|
| 32768 | `block_diagonal_64_bs` | OK | OK | True | 0.000000 | 0.000000 | 0.0000 | 0.862 | 0.848 |
| 32768 | `checkerboard_64_bs` | OK | OK | True | 0.000244 | 0.000012 | 87.2537 | 45.242 | 45.445 |
| 32768 | `sliding_window_128_bs` | OK | OK | True | 0.003906 | 0.000059 | 297.4635 | 1.204 | 1.204 |
| 32768 | `strided_bs` | OK | OK | True | 0.001953 | 0.000021 | 286.1410 | 23.404 | 22.949 |
| 32768 | `dilated_window_bs` | OK | OK | True | 0.003906 | 0.000057 | 437.4832 | 1.540 | 1.539 |
| 32768 | `nested_bs` | OK | OK | True | 0.001953 | 0.000027 | 239.5045 | 12.648 | 12.644 |
| 32768 | `hybrid_sparse_bs` | OK | OK | True | 0.000488 | 0.000016 | 140.7787 | 17.161 | 16.661 |
| 32768 | `global_local_bs` | OK | OK | True | 0.001953 | 0.000047 | 337.5160 | 2.566 | 2.572 |
| 32768 | `multiscale_dilated_bs` | OK | OK | True | 0.001953 | 0.000050 | 285.6590 | 2.257 | 2.227 |
| 32768 | `prefix_lm_bs` | OK | OK | True | 0.000488 | 0.000010 | 85.1601 | 46.196 | 45.232 |
| 32768 | `band_global_bs` | OK | OK | True | 0.001953 | 0.000047 | 301.3250 | 2.611 | 2.619 |
| 81920 | `block_diagonal_64_bs` | OK | OK | True | 0.000000 | 0.000000 | 0.0000 | 1.651 | 1.657 |
| 81920 | `checkerboard_64_bs` | OK | OK | True | 0.000244 | 0.000009 | 64.0005 | 280.581 | 280.960 |
| 81920 | `sliding_window_128_bs` | OK | OK | True | 0.003906 | 0.000068 | 461.1015 | 2.552 | 2.540 |
| 81920 | `strided_bs` | OK | OK | True | 0.003906 | 0.000017 | 286.1410 | 140.613 | 140.369 |
| 81920 | `dilated_window_bs` | OK | OK | True | 0.003906 | 0.000065 | 316.2399 | 3.405 | 3.467 |
| 81920 | `nested_bs` | OK | OK | True | 0.001953 | 0.000021 | 244.0363 | 72.539 | 72.044 |
| 81920 | `hybrid_sparse_bs` | OK | OK | True | 0.001953 | 0.000019 | 207.2981 | 98.857 | 98.357 |
| 81920 | `global_local_bs` | OK | OK | True | 0.001953 | 0.000053 | 319.1714 | 6.059 | 6.043 |
| 81920 | `multiscale_dilated_bs` | OK | OK | True | 0.001953 | 0.000057 | 383.4581 | 5.112 | 5.149 |
| 81920 | `prefix_lm_bs` | OK | OK | True | 0.001953 | 0.000013 | 195.0469 | 280.546 | 280.162 |
| 81920 | `band_global_bs` | OK | OK | True | 0.001953 | 0.000053 | 252.2804 | 6.071 | 5.991 |
