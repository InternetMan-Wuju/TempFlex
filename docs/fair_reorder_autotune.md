# Fair Reorder Autotune

- Method: identity external output vs reorder external output
- Shape: `B=1, H=2, D=128`
- Timing: `warmup=5, repeat=20, trials=1`
- Pass threshold: speedup_mean >= `1.01` and allclose with max_abs <= `0.00390625`

| Seq Len | Config | Mode | KV order | Identity mean | Reorder mean | Speedup mean | Speedup median | allclose | max_abs | Pass |
|--------:|--------|------|----------|--------------:|-------------:|-------------:|---------------:|----------|--------:|------|
| 32768 | `band_global_bs` | `wave_overlap` | `snake_inv` | 2.544 | 2.554 | 0.9961x | 0.9961x | True | 0.001953 | False |
| 32768 | `band_global_bs` | `wave_union_fast` | `snake_inv` | 2.547 | 2.543 | 1.0016x | 1.0016x | True | 0.001953 | False |
| 32768 | `hybrid_sparse_bs` | `wave_overlap` | `snake_inv` | 16.852 | 16.665 | 1.0112x | 1.0112x | True | 0.000488 | True |
| 32768 | `hybrid_sparse_bs` | `wave_union_fast` | `snake_inv` | 16.893 | 16.647 | 1.0148x | 1.0148x | True | 0.001953 | True |
| 32768 | `nested_bs` | `wave_overlap` | `snake_inv` | 12.613 | 12.452 | 1.0129x | 1.0129x | True | 0.001953 | True |
| 32768 | `nested_bs` | `wave_union_fast` | `snake_inv` | 12.622 | 12.479 | 1.0115x | 1.0115x | True | 0.001953 | True |
| 32768 | `strided_bs` | `wave_overlap` | `snake_inv` | 23.186 | 22.810 | 1.0165x | 1.0165x | True | 0.001953 | True |
| 32768 | `strided_bs` | `wave_union_fast` | `snake_inv` | 23.177 | 22.898 | 1.0122x | 1.0122x | True | 0.001953 | True |
| 81920 | `band_global_bs` | `wave_overlap` | `snake_inv` | 6.022 | 5.978 | 1.0074x | 1.0074x | True | 0.001953 | False |
| 81920 | `band_global_bs` | `wave_union_fast` | `snake_inv` | 5.967 | 5.979 | 0.9980x | 0.9980x | True | 0.001953 | False |
| 81920 | `hybrid_sparse_bs` | `wave_overlap` | `snake_inv` | 98.863 | 98.431 | 1.0044x | 1.0044x | True | 0.001953 | False |
| 81920 | `hybrid_sparse_bs` | `wave_union_fast` | `snake_inv` | 98.890 | 98.499 | 1.0040x | 1.0040x | True | 0.001953 | False |
| 81920 | `nested_bs` | `wave_overlap` | `snake_inv` | 72.596 | 72.357 | 1.0033x | 1.0033x | True | 0.001953 | False |
| 81920 | `nested_bs` | `wave_union_fast` | `snake_inv` | 72.614 | 72.465 | 1.0021x | 1.0021x | True | 0.001953 | False |
| 81920 | `strided_bs` | `wave_overlap` | `snake_inv` | 140.629 | 140.003 | 1.0045x | 1.0045x | True | 0.003906 | False |
| 81920 | `strided_bs` | `wave_union_fast` | `snake_inv` | 140.533 | - | -x | -x | - | - | False |

## Selector candidates

- `hybrid_sparse_bs@32768`: `wave_overlap` + `snake_inv` speedup_mean=1.0112x
- `hybrid_sparse_bs@32768`: `wave_union_fast` + `snake_inv` speedup_mean=1.0148x
- `nested_bs@32768`: `wave_overlap` + `snake_inv` speedup_mean=1.0129x
- `nested_bs@32768`: `wave_union_fast` + `snake_inv` speedup_mean=1.0115x
- `strided_bs@32768`: `wave_overlap` + `snake_inv` speedup_mean=1.0165x
- `strided_bs@32768`: `wave_union_fast` + `snake_inv` speedup_mean=1.0122x
