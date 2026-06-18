# 32K Strided Reorder msprof Analysis

> Date: 2026-06-18  
> Shape: `B=1,H=2,S=32768,D=128`  
> Sparse config: `strided_bs`  
> Compare: `identity external` vs `wave_overlap + snake_inv external`  
> msprof metric: `PipeUtilization`

## Summary

For `strided_bs@32K`, reorder has a measurable kernel-side gain. The `time_runner` benchmark improves from `23.347 ms` to `22.925 ms`, or `1.0184x`. msprof agrees with the benchmark: the dominant fused kernel improves from `22.918 ms` avg to `22.550 ms` avg, or `1.0163x`.

This means reorder is useful for this 32K sparse mode. However, the 80K repeat=20 autotune did not keep a stable `>=1.01x`, so the current 80K bottleneck is not solved by row/KV permutation alone.

## Commands

Identity external:

```bash
python3 flex_attention_run_script.py --mode msprof --target reorder \
  --shape 1,2,32768,128 --sparse-config strided_bs --device auto --dtype bfloat16 \
  --warmup 3 --repeat 5 --no-compare --no-causal-fastpath \
  --enable-block-reorder --block-reorder-impl external \
  --block-reorder-mode identity --allow-identity-reorder --kv-order asc --wave-size 132 \
  --msprof-output msprof_out/strided32k_identity --msprof-aic-metrics PipeUtilization
```

Reorder external:

```bash
python3 flex_attention_run_script.py --mode msprof --target reorder \
  --shape 1,2,32768,128 --sparse-config strided_bs --device auto --dtype bfloat16 \
  --warmup 3 --repeat 5 --no-compare --no-causal-fastpath \
  --enable-block-reorder --block-reorder-impl external \
  --block-reorder-mode wave_overlap --kv-order snake_inv --wave-size 132 \
  --msprof-output msprof_out/strided32k_reorder --msprof-aic-metrics PipeUtilization
```

## Result

| Metric | Identity external | Reorder external | Speedup |
|---|---:|---:|---:|
| `time_runner` avg | 23.347 ms | 22.925 ms | 1.0184x |
| msprof op total | 190.4 ms | 186.9 ms | 1.0187x |
| msprof task total | 189.6 ms | 186.3 ms | 1.0177x |
| `triton_tem_fused_0` total | 183.3 ms | 180.4 ms | 1.0161x |
| `triton_tem_fused_0` avg | 22.918 ms | 22.550 ms | 1.0163x |
| helper total | 6.997 ms | 6.456 ms | 1.0838x |
| AI CPU total | 6.363 ms | 5.810 ms | 1.0952x |

## Bottleneck Reading

`triton_tem_fused_0` accounts for about `96%` of op time in both runs. Top fused kernels report high cube utilization: identity is around `97%-98%`, while reorder is around `98.5%-99.8%`.

The remaining helper/AI CPU work is mainly `Sort` and `IndexPut`:

| Op type | Identity | Reorder |
|---|---:|---:|
| `Sort` | 5.132 ms | 4.700 ms |
| `IndexPut` | 1.231 ms | 1.110 ms |
| `GatherElementsV2` | 0.312 ms | 0.309 ms |

These helpers are small relative to the fused kernel, but they still appear in the profile window. For end-to-end timing, metadata and reorder artifacts should be cached or moved fully outside the timed/profiled loop. For kernel-only reporting, the MSTX scope should be kept around the compiled flex call after external metadata has already been prepared.

## Optimization Direction

1. Keep `strided_bs@32K` in the secondary selector candidate list because the kernel profile confirms a real `~1.6%-1.8%` gain.
2. Do not enable 80K reorder by default yet; repeat=20 results show only `~1.004x` best stable gain.
3. Cache `perm`, inverse permutation, and reordered sparse metadata. This removes host-side reorder/metadata cost from practical repeated inference runs.
4. For 80K, prioritize FULL_KV / PURE_BLOCK_SPARSE template optimization. The dominant fused kernel is already cube-heavy, so stable `>=1%` likely needs lower kernel work or better template scheduling rather than more Python-side permutation search.
