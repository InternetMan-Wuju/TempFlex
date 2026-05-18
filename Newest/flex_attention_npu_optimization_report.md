# Flex Attention NPU Optimization Report

Date: 2026-05-18

This report records the optimization pass on
`torch_npu/_inductor/kernel/flex_attention.py`.

## 中文摘要

这轮优化的核心结论是：默认测试里的 Flex Attention 只是“identity score +
causal mask”，通用 Flex Triton 模板虽然确实融合成了 `triton_tem_fused_0`，
但单次约 9.1 ms，明显慢于手写 attention。对这个窄场景，直接降级/改写成
`aten.scaled_dot_product_attention(..., is_causal=True)` 会走 NPU
`FlashAttentionScore`，速度提升非常明显。

最终改动是在 `torch_npu/_inductor/kernel/flex_attention.py` 里加了一个保守
fastpath：只有在 score 图完全不改 score、mask 图严格是 `m >= n`、没有额外
buffer、不要 logsumexp、且 Q/K/V 是同 dtype/device 的 4D square non-GQA
布局时才触发。触发后生成代码不再发射通用 Triton Flex kernel，而是调用 Aten
SDPA；其它 Flex Attention 用法仍走原路径。

关键结果：

- baseline wall-clock：Flex `9.222 ms`，manual `2.375 ms`
- optimized wall-clock：Flex `0.517 ms`，manual `2.412 ms`
- Flex wall-clock 自身提升约 `17.8x`
- msprof Flex op total 从 `45.691 ms / 5 calls` 降到 `1.975 ms / 5 repeats`
- 优化后 Flex 按 device op total 约比 manual 快 `5.95x`

## Test Case

- Script: `/CYT_fileSys_2/Code1/flex_test/flex_attention2.py`
- Shape: `B=4, H=8, S=2048, D=128`
- Dtype/device: `bfloat16` on NPU
- Flex pattern: identity `score_mod`, causal mask `m >= n`
- Timing config used for the main comparisons: `warmup=3`, `repeat=5`
- Profiling scope: `summarize_msprof.py` used repeat-tail because the exported
  msprof trace did not contain usable MSTX repeat markers.

## Baseline

The current forced-recompile baseline did use the generated Flex Triton kernel,
not the older helper-op fallback pattern seen in earlier traces.

Generated code:

```text
torch_compile_debug/run_2026_05_18_14_18_31_101050-pid_47582/torchinductor/model__1_inference_3.0/output_code.py
```

Important generated constants:

```text
kernel: triton_tem_fused_0
BLOCK_M=64
BLOCK_N=64
PRESCALE_QK=False
num_stages=3
num_warps=4
grid=(32, 32, 1)
```

Benchmark:

```text
Flex:   9.222 ms
Manual: 2.375 ms
Correctness: allclose=True
```

msprof summary:

```text
log: /CYT_fileSys_2/Code1/flex_test/msprof_out/opt_baseline_current/result.log

Flex op total:   45.691 ms / 5 calls, avg 9.138 ms
Manual op total: 11.582 ms / 25 calls
Flex/manual op ratio: 3.95x slower
Top Flex op: triton_tem_fused_0, MIX_AIC, cube about 98%
```

Conclusion: the slow part was the generic Flex Triton template itself for this
plain causal-attention case.

## Parameter Iterations

The following variants were tried before changing the lowering:

| Change | Result | Conclusion |
| --- | ---: | --- |
| `BLOCK_M=128, BLOCK_N=64` | compile failed, UB overflow about 2.906 Mbits > 1.573 Mbits | not viable |
| `BLOCK_M=64, BLOCK_N=128` | compile failed, UB overflow about 2.373 Mbits > 1.573 Mbits | not viable |
| `BLOCK_M=32, BLOCK_N=64` | 10.522 ms | slower |
| `BLOCK_M=64, BLOCK_N=32` | 11.404 ms | slower |
| `PRESCALE_QK=True` | 9.263 ms | no useful gain |
| `num_stages=2` | 9.195 ms | noise-level gain |
| `num_stages=1` | 9.193 ms | noise-level gain |
| `num_warps=2` | 9.214 ms | noise-level gain |
| `num_warps=8` | 9.196 ms | noise-level gain |

Conclusion: tuning the generic template did not close the gap. The viable tile
settings were still around 9 ms, while native attention was much faster.

## Native Attention Probe

An eager `torch.nn.functional.scaled_dot_product_attention(..., is_causal=True)`
probe on the same shape was correct and much faster:

```text
Eager SDPA: about 0.420 ms, allclose=True
```

Other direct `torch_npu` attention APIs were also explored, but the simple calls
without a correct causal mask were either wrong for this test or required an
explicit mask. The reliable primitive for this case was PyTorch SDPA, which
dispatches to NPU FlashAttentionScore.

## Implemented Change

File changed:

```text
/CYT_fileSys_2/Code1/flex_test/Newest/site-packages/torch_npu/_inductor/kernel/flex_attention.py
/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/_inductor/kernel/flex_attention.py
```

The Flex Attention forward lowering now has a narrow causal SDPA fastpath. It
detects this exact pattern:

- no `score_mod_other_buffers`
- no `mask_mod_other_buffers`
- `OUTPUT_LOGSUMEXP=False`
- `ROWS_GUARANTEED_SAFE=True`
- `BLOCKS_ARE_CONTIGUOUS=True`
- score graph returns the original score unchanged
- mask graph is exactly `m >= n`
- query/key/value are 4D, same dtype, same device
- square non-GQA layout: same batch, heads, sequence length, and head dim

When those guards pass, the lowering emits:

```text
aten.scaled_dot_product_attention.default(
    query, key, value, None, 0.0, True,
    scale=<original scale>,
    enable_gqa=False,
)
```

The code uses `FallbackKernel.create(...)` so Inductor emits a normal Aten
fallback call instead of the generic Triton Flex template. The returned
logsumexp buffer is still created for the Flex Attention return contract, but
the generated inference graph DCE removes it when unused.

Debug marker when the fastpath is selected:

```text
[torch_npu flex_attention debug] 使用 torch_npu causal SDPA fastpath 替代通用 flex_attention Triton kernel
```

## Optimized Result

Generated code after the change:

```text
torch_compile_debug/run_2026_05_18_14_35_53_691438-pid_56123/torchinductor/model__1_inference_3.0/output_code.py
```

The generated model now calls:

```text
torch.ops.aten.scaled_dot_product_attention.default(
    arg0_1, arg1_1, arg2_1, None, 0.0, True,
    scale=0.08838834764831843,
    enable_gqa=False,
)
```

Benchmark:

```text
Flex:   0.517 ms
Manual: 2.412 ms
Correctness: allclose=True
```

msprof summary:

```text
log: /CYT_fileSys_2/Code1/flex_test/msprof_out/opt_sdpa_fastpath/result.log

Flex op total:   1.975 ms / 15 ops
Manual op total: 11.744 ms / 25 ops
Flex/manual op ratio: 0.17x, about 5.95x faster by device op total

Top Flex ops:
  FlashAttentionScore: 1.754 ms total / 5 calls, avg 0.351 ms
  Triu:                0.140 ms total / 5 calls
  OnesLike:            0.081 ms total / 5 calls
```

Performance changes:

| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| Wall-clock Flex benchmark | 9.222 ms | 0.517 ms | 17.8x faster |
| Wall-clock Flex vs manual | 3.88x slower | 4.66x faster | flipped from slower to faster |
| msprof Flex op total | 45.691 ms | 1.975 ms | 23.1x faster |
| msprof Flex task total | 36.549 ms | 3.610 ms | 10.1x faster |
| Optimized Flex vs manual op total | n/a | 1.975 ms vs 11.744 ms | 5.95x faster |

The optimized profile still contains small `Triu` and `OnesLike` calls. They
appear to come from causal mask setup inside the SDPA/NPU path. They are minor
relative to `FlashAttentionScore`.

## Correctness And Scope

This change is intentionally conservative. It does not try to optimize arbitrary
Flex Attention graphs. Any score modification, non-causal mask, extra captured
buffer, logsumexp output requirement, GQA layout, non-square sequence shape, or
dtype/device mismatch falls back to the original generic Flex Attention lowering.

One additional `warmup=10, repeat=10` confirmation run was attempted after the
main measurements, but the execution environment rejected the escalated NPU
command because of an approval/usage limit. The report therefore uses the
completed benchmark and msprof runs above.

## Reproduction Commands

Baseline benchmark:

```bash
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_flex_baseline_$(date +%s) \
python3 flex_attention2.py --mode benchmark --target both --device npu --warmup 3 --repeat 5
```

Baseline msprof:

```bash
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_msprof_baseline_$(date +%s) \
python3 flex_attention2.py --mode msprof --target both --device npu \
  --warmup 3 --repeat 5 \
  --msprof-output msprof_out/opt_baseline_current

python3 summarize_msprof.py msprof_out/opt_baseline_current --top 12
```

Optimized benchmark:

```bash
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_flex_sdpa_fastpath_$(date +%s) \
python3 flex_attention2.py --mode benchmark --target both --device npu --warmup 3 --repeat 5
```

Optimized msprof:

```bash
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_msprof_sdpa_fastpath_$(date +%s) \
python3 flex_attention2.py --mode msprof --target both --device npu \
  --warmup 3 --repeat 5 \
  --msprof-output msprof_out/opt_sdpa_fastpath

python3 summarize_msprof.py msprof_out/opt_sdpa_fastpath --top 12
```

## Suggested Next Steps

- Add a small regression test for the fastpath guard: identity score plus
  causal `m >= n` should lower to SDPA, while modified score/mask cases should
  stay on the generic Flex path.
- Investigate whether SDPA causal setup can avoid the small `Triu`/`OnesLike`
  overhead on this torch_npu version.
- If this is meant to become an upstreamable patch, gate it with a clear comment
  and keep the guard narrow unless broader mask equivalence checks are added.
