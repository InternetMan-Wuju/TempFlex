# Fair Reorder Autotune

- Method: identity external output vs reorder external output
- Shape: `B=1, H=2, D=128`
- Timing: `warmup=5, repeat=20, trials=1`
- Pass threshold: speedup_mean >= `1.01` and allclose with max_abs <= `0.00390625`

| Seq Len | Config | Mode | KV order | Identity mean | Reorder mean | Speedup mean | Speedup median | allclose | max_abs | Pass |
|--------:|--------|------|----------|--------------:|-------------:|-------------:|---------------:|----------|--------:|------|
| 32768 | `band_global_bs` | `auction_union_fast` | `snake_inv` | 2.535 | 2.567 | 0.9875x | 0.9875x | True | 0.001953 | False |
| 32768 | `band_global_bs` | `auction_union_exact_path` | `snake_inv` | 2.552 | 2.559 | 0.9973x | 0.9973x | True | 0.001953 | False |
| 32768 | `hybrid_sparse_bs` | `auction_union_fast` | `snake_inv` | 16.935 | 16.636 | 1.0180x | 1.0180x | True | 0.001953 | True |
| 32768 | `hybrid_sparse_bs` | `auction_union_exact_path` | `snake_inv` | 16.908 | 16.650 | 1.0155x | 1.0155x | True | 0.001953 | True |
| 32768 | `nested_bs` | `auction_union_fast` | `snake_inv` | 12.600 | 12.435 | 1.0133x | 1.0133x | True | 0.001953 | True |
| 32768 | `nested_bs` | `auction_union_exact_path` | `snake_inv` | 12.601 | 12.439 | 1.0130x | 1.0130x | True | 0.001953 | True |
| 32768 | `strided_bs` | `auction_union_fast` | `snake_inv` | 23.202 | 22.820 | 1.0167x | 1.0167x | True | 0.001953 | True |
| 32768 | `strided_bs` | `auction_union_exact_path` | `snake_inv` | 23.208 | 22.826 | 1.0167x | 1.0167x | True | 0.001953 | True |
| 81920 | `band_global_bs` | `auction_union_fast` | `snake_inv` | 5.967 | 5.970 | 0.9995x | 0.9995x | True | 0.001953 | False |
| 81920 | `band_global_bs` | `auction_union_exact_path` | `snake_inv` | 5.958 | 5.973 | 0.9975x | 0.9975x | True | 0.001953 | False |
| 81920 | `hybrid_sparse_bs` | `auction_union_fast` | `snake_inv` | 98.917 | 98.320 | 1.0061x | 1.0061x | True | 0.001953 | False |
| 81920 | `hybrid_sparse_bs` | `auction_union_exact_path` | `snake_inv` | 98.961 | 98.188 | 1.0079x | 1.0079x | True | 0.000977 | False |
| 81920 | `nested_bs` | `auction_union_fast` | `snake_inv` | 72.640 | - | -x | -x | - | - | False |
| 81920 | `nested_bs` | `auction_union_exact_path` | `snake_inv` | 72.588 | 72.162 | 1.0059x | 1.0059x | True | 0.000977 | False |
| 81920 | `strided_bs` | `auction_union_fast` | `snake_inv` | 140.534 | 140.111 | 1.0030x | 1.0030x | True | 0.003906 | False |
| 81920 | `strided_bs` | `auction_union_exact_path` | `snake_inv` | 140.701 | 140.554 | 1.0010x | 1.0010x | True | 0.003906 | False |

## Selector candidates

- `hybrid_sparse_bs@32768`: `auction_union_fast` + `snake_inv` speedup_mean=1.0180x
- `hybrid_sparse_bs@32768`: `auction_union_exact_path` + `snake_inv` speedup_mean=1.0155x
- `nested_bs@32768`: `auction_union_fast` + `snake_inv` speedup_mean=1.0133x
- `nested_bs@32768`: `auction_union_exact_path` + `snake_inv` speedup_mean=1.0130x
- `strided_bs@32768`: `auction_union_fast` + `snake_inv` speedup_mean=1.0167x
- `strided_bs@32768`: `auction_union_exact_path` + `snake_inv` speedup_mean=1.0167x
