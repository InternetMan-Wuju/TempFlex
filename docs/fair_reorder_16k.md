# Long Reorder Correctness

- Method: identity external output vs wave_overlap reorder output
- Tolerance: `rtol=0.03, atol=0.03`

| Seq Len | Config | Identity | Reorder | allclose | max_abs | mean_abs | max_rel | Identity ms | Reorder ms |
|--------:|--------|----------|---------|----------|--------:|---------:|--------:|------------:|-----------:|
| 16384 | `block_diagonal_64_bs` | OK | OK | True | 0.000000 | 0.000000 | 0.0000 | 0.567 | 0.576 |
| 16384 | `checkerboard_64_bs` | OK | OK | True | 0.000488 | 0.000032 | 121.6419 | 11.695 | 11.736 |
| 16384 | `sliding_window_128_bs` | OK | OK | True | 0.003906 | 0.000115 | 297.4635 | 0.735 | 0.730 |
| 16384 | `strided_bs` | OK | OK | True | 0.001953 | 0.000050 | 286.1410 | 6.223 | 6.030 |
| 16384 | `dilated_window_bs` | OK | OK | True | 0.003906 | 0.000110 | 316.2399 | 0.917 | 0.916 |
| 16384 | `nested_bs` | OK | OK | True | 0.001953 | 0.000062 | 239.5045 | 3.687 | 3.606 |
| 16384 | `hybrid_sparse_bs` | OK | OK | True | 0.001953 | 0.000056 | 208.4039 | 4.826 | 4.674 |
| 16384 | `global_local_bs` | OK | OK | True | 0.001953 | 0.000090 | 247.2811 | 1.422 | 1.416 |
| 16384 | `multiscale_dilated_bs` | OK | OK | True | 0.001953 | 0.000096 | 331.7995 | 1.264 | 1.260 |
| 16384 | `prefix_lm_bs` | OK | OK | True | 0.001953 | 0.000039 | 195.0469 | 11.952 | 11.628 |
| 16384 | `band_global_bs` | OK | OK | True | 0.001953 | 0.000091 | 301.3250 | 1.438 | 1.438 |
