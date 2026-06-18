# 32K Hybrid KV Orientation Template msprof Analysis

> Date: 2026-06-18  
> Shape: `B=1,H=2,S=32768,D=128`  
> Sparse config: `hybrid_sparse_bs`  
> Compare: `identity external + asc` vs `auction_union_fast + union_boundary_dp`  
> msprof metric: `PipeUtilization`

## Summary

`union_boundary_dp` improves the NPU FULL_KV/PURE_BLOCK_SPARSE fused kernel, not just host-side metadata behavior. The benchmark time improved from `17.083 ms` to `16.759 ms` (`1.0193x`), and msprof shows the dominant `triton_tem_fused_0` average improved from `16.644 ms` to `16.385 ms` (`1.0158x`).

The 80K identity external msprof run crashed with `exitCode:11` before device summary export, so this template-level profile uses the stable 32K case as the diagnostic anchor.

## Commands

Identity external:

```bash
python3 flex_attention_run_script.py --mode msprof --target reorder \
  --shape 1,2,32768,128 --sparse-config hybrid_sparse_bs \
  --device auto --dtype bfloat16 --warmup 3 --repeat 5 --no-compare \
  --no-causal-fastpath --enable-block-reorder --block-reorder-impl external \
  --block-reorder-mode identity --allow-identity-reorder --kv-order asc \
  --wave-size 132 --msprof-output msprof_out/hybrid32k_identity_template \
  --msprof-aic-metrics PipeUtilization
```

Reorder external:

```bash
python3 flex_attention_run_script.py --mode msprof --target reorder \
  --shape 1,2,32768,128 --sparse-config hybrid_sparse_bs \
  --device auto --dtype bfloat16 --warmup 3 --repeat 5 --no-compare \
  --no-causal-fastpath --enable-block-reorder --block-reorder-impl external \
  --block-reorder-mode auction_union_fast --kv-order union_boundary_dp \
  --wave-size 132 --msprof-output msprof_out/hybrid32k_union_template \
  --msprof-aic-metrics PipeUtilization
```

## Results

| Metric | Identity external | Reorder external | Speedup |
|---|---:|---:|---:|
| `time_runner` avg | 17.083 ms | 16.759 ms | 1.0193x |
| msprof op total | 139.9 ms | 137.9 ms | 1.0145x |
| msprof task total | 139.3 ms | 137.2 ms | 1.0153x |
| `triton_tem_fused_0` total | 133.2 ms | 131.1 ms | 1.0160x |
| `triton_tem_fused_0` avg | 16.644 ms | 16.385 ms | 1.0158x |
| helper total | 6.733 ms | 6.792 ms | 0.9913x |
| AI CPU total | 6.099 ms | 6.140 ms | 0.9933x |

## Template Interpretation

The FULL_KV path does consume metadata order inside the fused kernel. The template initializes from the first `FULL_KV_IDX` block, then `forward_inner` advances K/V pointers through `get_offset_for_next_block()`, which loads the next block index from `FULL_KV_IDX` whenever a sparse block boundary is crossed.

This means `union_boundary_dp` can improve fused-kernel locality by reducing cold K/V pointer jumps between adjacent metadata blocks. The profile supports this: helper time is flat, while `triton_tem_fused_0` improves by about `1.6%`.

## Next Template Work

1. Profile `hybrid_sparse_bs@81920` with a safer msprof recipe, for example `warmup=1, repeat=2`, or split identity/reorder into fresh output roots as done here.
2. Add `Memory` or `MemoryL0` metrics after PipeUtilization is stable, to verify whether the fused-kernel win maps to lower memory stalls or better cache behavior.
3. Inspect and tune `get_offset_for_next_block()` / FULL_KV traversal for long sequence stability. Current 80K results suggest the template responds to KV order, but the benefit is diluted or crash-prone at 80K.
4. Keep `union_boundary_dp` as a diagnostic and 32K candidate; do not enable it for 80K selector until repeat=20 reaches `>=1.01x` with allclose.
