# Newest Patch Bundle

This directory is the portable patch bundle for the Flex Attention NPU debugging work from 2026-05-18.

It intentionally lives at repo top level instead of `aboutCodex/`, because `aboutCodex/` is gitignored and may contain personal Codex configuration.

## Install

Run from any directory:

```bash
bash /CYT_fileSys_2/Code1/flex_test/Newest/apply_newest.sh
```

The installer backs up current target files into:

```text
/CYT_fileSys_2/Code1/flex_test/Newest/backups/<timestamp>/
```

## Files

- `workspace/flex_attention2.py` -> `/CYT_fileSys_2/Code1/flex_test/flex_attention2.py`
- `site-packages/torch_npu/_inductor/kernel/flex_attention.py` -> `/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/_inductor/kernel/flex_attention.py`
- `site-packages/torch_npu/_inductor/npu_triton_heuristics.py` -> `/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/_inductor/npu_triton_heuristics.py`
- `site-packages/torch/_inductor/async_compile.py` -> `/usr/local/python3.11.14/lib/python3.11/site-packages/torch/_inductor/async_compile.py`

## `flex_attention2.py` Parameters

Default command:

```bash
python3 /CYT_fileSys_2/Code1/flex_test/flex_attention2.py
```

Default behavior:

- Runs `--mode benchmark --target both`.
- Uses shape `[B, H, S, D] = [4, 8, 2048, 128]`.
- Uses `--dtype bfloat16 --device auto`.
- Uses `--warmup 10 --repeat 10`.
- Uses Flex tile options `--block-m 64 --block-n 64`.
- Uses `--manual-mask precompute`.
- Uses static compile (`torch.compile(..., dynamic=False)`) on the NPU Flex path.

Common shape commands:

```bash
python3 flex_attention2.py --shape 1,4,512,64
python3 flex_attention2.py --shape 1,4,512,64 --shape 2x8x1024x64
python3 flex_attention2.py --shape-suite small
python3 flex_attention2.py --shape-suite smoke --max-shapes 2
```

Shape parameters:

- `--batch`, `--heads`, `--seq-len`, `--head-dim`: dimensions used by the default single-shape run.
- `--shape B,H,S,D` or `--shape BxHxSxD`: explicit shape; repeat the flag for a sweep.
- `--shape-suite single`: default suite, using the explicit dimension flags.
- `--shape-suite small`: quick small-shape checks.
- `--shape-suite smoke`: short representative sweep, including the default large shape.
- `--max-shapes N`: cap the number of attempted shapes.
- `--stop-on-shape-error`: stop at the first failed shape; otherwise multi-shape mode records the error and continues.

Execution and comparison:

- `--target both`: run Flex Attention and manual attention, then compare outputs.
- `--target flex`: run only Flex Attention.
- `--target manual`: run only the manual dense baseline.
- `--mode benchmark`: normal benchmark mode.
- `--mode profile-target`: profile a single target process; requires `--target flex` or `--target manual`.
- `--mode msprof`: run target process(es) through `msprof`.
- `--mode unittest`: run the built-in unittest cases.
- `--no-compare`: skip output comparison.
- `--rtol`, `--atol`, `--topk`: tune correctness reporting.

Compile and kernel tuning:

- `--static-compile`: default; uses `dynamic=False`.
- `--dynamic-compile`: request `dynamic=True`.
- `--allow-npu-dynamic-compile`: required before `--dynamic-compile` actually affects the NPU Flex path. Keep this for a later investigation because it is currently unstable.
- `--block-m`, `--block-n`: Flex Attention tile sizes.
- `--prescale-qk`: set `PRESCALE_QK=True`.
- `--num-warps`, `--num-stages`: override Triton kernel launch tuning.
- `--enable-gqa`: enable grouped-query attention.
- `--suppress-compile-errors`: let `torch.compile` fall back instead of failing on compilation errors.

Timing and profiling:

- `--warmup`: untimed iterations before measurement.
- `--repeat`: timed iterations; the script reports average latency.
- `--device`: `auto`, `npu`, `npu:0`, `cpu`, or `cuda`.
- `--dtype`: `bfloat16` by default; supports aliases such as `bf16`, `fp16`, and `fp32`.
- `--manual-mask precompute`: default; builds the manual dense mask outside the timed loop.
- `--manual-mask inside`: builds the manual dense mask during the manual attention call.
- `--mstx`: emit MSTX ranges.
- `--msprof-output`, `--msprof-aic-metrics`, `--msprof-option`: configure `msprof`.

## Notes

Read these files before continuing the investigation in a fresh Codex session:

- `flex_attention_npu_debug_notes.md`
- `flex_attention_npu_optimization_report.md`
