# Fair Reorder Autotune

- Method: identity external output vs reorder external output
- Shape: `B=1, H=2, D=128`
- Timing: `warmup=5, repeat=20, trials=1`
- Pass threshold: speedup_mean >= `1.01` and allclose with max_abs <= `0.00390625`

| Seq Len | Config | Mode | KV order | Identity mean | Reorder mean | Speedup mean | Speedup median | allclose | max_abs | Pass |
|--------:|--------|------|----------|--------------:|-------------:|-------------:|---------------:|----------|--------:|------|
| 32768 | `hybrid_sparse_bs` | `auction_union_fast` | `snake_inv` | 16.901 | 16.800 | 1.0060x | 1.0060x | True | 0.001953 | False |
| 32768 | `hybrid_sparse_bs` | `auction_union_fast` | `boundary_dp` | 16.944 | 16.664 | 1.0168x | 1.0168x | True | 0.001953 | True |
| 32768 | `hybrid_sparse_bs` | `auction_union_fast` | `union_boundary_dp` | 16.925 | 16.620 | 1.0184x | 1.0184x | True | 0.000977 | True |
| 32768 | `hybrid_sparse_bs` | `auction_union_exact_path` | `snake_inv` | 16.915 | 16.682 | 1.0140x | 1.0140x | True | 0.001953 | True |
| 32768 | `hybrid_sparse_bs` | `auction_union_exact_path` | `boundary_dp` | 16.887 | 16.660 | 1.0136x | 1.0136x | True | 0.001953 | True |
| 32768 | `hybrid_sparse_bs` | `auction_union_exact_path` | `union_boundary_dp` | 16.885 | 16.652 | 1.0140x | 1.0140x | True | 0.000977 | True |
| 81920 | `hybrid_sparse_bs` | `auction_union_fast` | `snake_inv` | 98.797 | - | -x | -x | - | - | False |
| 81920 | `hybrid_sparse_bs` | `auction_union_fast` | `boundary_dp` | 98.720 | - | -x | -x | - | - | False |
| 81920 | `hybrid_sparse_bs` | `auction_union_fast` | `union_boundary_dp` | 98.905 | - | -x | -x | - | - | False |
| 81920 | `hybrid_sparse_bs` | `auction_union_exact_path` | `snake_inv` | 98.902 | 98.164 | 1.0075x | 1.0075x | True | 0.000977 | False |
| 81920 | `hybrid_sparse_bs` | `auction_union_exact_path` | `boundary_dp` | 98.920 | 98.286 | 1.0065x | 1.0065x | True | 0.001953 | False |
| 81920 | `hybrid_sparse_bs` | `auction_union_exact_path` | `union_boundary_dp` | 98.847 | 98.167 | 1.0069x | 1.0069x | True | 0.000488 | False |

## Selector candidates

- `hybrid_sparse_bs@32768`: `auction_union_fast` + `boundary_dp` speedup_mean=1.0168x
- `hybrid_sparse_bs@32768`: `auction_union_fast` + `union_boundary_dp` speedup_mean=1.0184x
- `hybrid_sparse_bs@32768`: `auction_union_exact_path` + `snake_inv` speedup_mean=1.0140x
- `hybrid_sparse_bs@32768`: `auction_union_exact_path` + `boundary_dp` speedup_mean=1.0136x
- `hybrid_sparse_bs@32768`: `auction_union_exact_path` + `union_boundary_dp` speedup_mean=1.0140x
