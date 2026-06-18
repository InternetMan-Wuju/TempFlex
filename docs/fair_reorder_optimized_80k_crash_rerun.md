# Optimized Fair Newest Reorder Matrix

- Method: `Newest without reorder = identity external`, `Newest with reorder = optimized external reorder`
- Shape: `B=1, H=2, D=128`
- Kernel block: `BLOCK_M=64, BLOCK_N=64`
- Timing: `warmup=3, repeat=5`

| Seq Len | Config | Mode | KV order | Without reorder | With reorder | Speedup | allclose | max_abs | Note |
|--------:|--------|------|----------|----------------:|-------------:|--------:|----------|--------:|------|
| 81920 | `strided_bs` | `auction_union_exact_path` | `snake_inv` | 140.296 | 140.328 | 0.9998x | True | 0.003906 | 80K optimized probe: exact path plus stable snake_inv |
| 81920 | `dilated_window_bs` | `wave_overlap` | `snake_inv` | 3.409 | 3.422 | 0.9962x | True | 0.003906 | fallback: no validated auction/exact-path win for this 80K config |
| 81920 | `nested_bs` | `auction_union_exact_path` | `snake_inv` | 72.557 | 72.218 | 1.0047x | True | 0.000977 | 80K optimized probe: exact path plus stable snake_inv |
| 81920 | `hybrid_sparse_bs` | `auction_union_exact_path` | `union_boundary_dp` | 98.870 | - | -x | - | - | 80K optimized probe: exact path plus union_boundary_dp |
