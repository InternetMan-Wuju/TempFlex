# Fair Reorder Autotune

- Method: identity external output vs reorder external output
- Shape: `B=1, H=2, D=128`
- Kernel block: `BLOCK_M=64, BLOCK_N=64`
- Timing: `warmup=3, repeat=5, trials=1`
- Pass threshold: speedup_mean >= `1.01` and allclose with max_abs <= `0.00390625`

| Seq Len | Config | Mode | KV order | Identity mean | Reorder mean | Speedup mean | Speedup median | allclose | max_abs | Pass |
|--------:|--------|------|----------|--------------:|-------------:|-------------:|---------------:|----------|--------:|------|
| 81920 | `hybrid_sparse_bs` | `wave_overlap` | `snake_inv` | - | 98.949 | -x | -x | - | - | False |

## Selector candidates

- No candidate reached the pass threshold.
