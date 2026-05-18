# Flex Attention NPU Debug Notes

Date: 2026-05-18

This note records the changes made during the Codex debugging session so the next Codex session can continue quickly.

This file is intentionally stored under `/CYT_fileSys_2/Code1/flex_test/Newest/`, not under `aboutCodex/`, because `aboutCodex/` is gitignored and may contain personal configuration.

## Goal

`flex_attention2.py` was crashing or unclear around NPU Inductor/Flex Attention compilation. We needed:

- `flex_attention2.py` to run directly.
- `--dynamic-compile` to avoid the unstable NPU dynamic compile path by default.
- A clear distinction between Flex Attention compilation and actual kernel runtime.
- A portable copy of modified files, because image-provided Python site-packages files may be refreshed.

## Main Findings

- `python3 flex_attention2.py` still enters `torch.compile` on the first compiled function call. It may reuse cache, but it still goes through Dynamo/Inductor setup.
- `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` forces cache bypass and is useful for checking whether source changes are active.
- `torch_npu/_inductor/kernel/flex_attention.py` is the Flex Attention lowering/template source, so Python prints there prove compilation/lowering is reached.
- Codex default `exec_command` sandbox may not expose the Ascend/NPU device nodes. In that sandbox, `torch.npu.is_available()` can print `False` with driver/hal errors even though the user's normal root shell prints `True`.
  - Verified on 2026-05-18: default sandbox returned `False`; rerunning the same command outside the sandbox returned `True`.
  - For real NPU checks, `msprof`, or `flex_attention2.py --device npu`, run outside the default sandbox / with escalated execution.
- Device-side `tl.device_print()` is not safe here:
  - Non-ASCII strings fail during Triton compilation.
  - ASCII `tl.device_print()` still crashed Ascend `triton-adapter-opt` for this generated flex attention kernel.
- The final timed loop calls the generated object from `AsyncCompile.triton(...).run(...)`, so runtime proof had to wrap the `.run` method returned from `torch._inductor.async_compile.AsyncCompile.triton`.
- Existing msprof traces under `msprof_out/flex_vs_manual` and `msprof_out/new_run` did not contain the expected `profile_repeat_*` MSTX markers in `msprof_*.json`, despite `--msproftx=on` and `--mstx-domain-include=flex_attention2`.
  - `summarize_msprof.py --scope auto` therefore fell back to the `repeat-tail` estimate: use the last `repeat` iterations after `warmup`.
  - This is good enough for a coarse steady-state device comparison, but not as strict as a real MSTX time window.
  - `api_statistic_*.csv` has no per-call timestamp, so host API totals remain full-profile and include setup/warmup/compile effects.
- Earlier traces under `msprof_out/flex_vs_manual` and `msprof_out/new_run` looked like helper/lowered ops (`SelectV2`, `Cast`, `SoftmaxV2`, `Mul`, `Sub`). A later forced-recompile baseline in `msprof_out/opt_baseline_current` did use the fused generated Triton kernel `triton_tem_fused_0`, but it was still slow: Flex device op total was 45.691 ms for 5 calls, about 3.95x manual attention.
- The 2026-05-18 optimization added a narrow causal SDPA fastpath for the exact identity-score + causal `m >= n` case. It lowers to `aten.scaled_dot_product_attention.default`, which dispatches to NPU `FlashAttentionScore`.
  - Wall-clock Flex benchmark improved from 9.222 ms to 0.517 ms.
  - msprof Flex op total improved from 45.691 ms to 1.975 ms for 5 repeat iterations.
  - Optimized Flex is about 5.95x faster than manual by device op total for the default test shape.
  - Full report: `/CYT_fileSys_2/Code1/flex_test/Newest/flex_attention_npu_optimization_report.md`

## Modified Files

Workspace file:

- `/CYT_fileSys_2/Code1/flex_test/flex_attention2.py`
- `/CYT_fileSys_2/Code1/flex_test/summarize_msprof.py`
- `/CYT_fileSys_2/Code1/flex_test/Newest/flex_attention_npu_optimization_report.md`

Image/runtime files:

- `/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/_inductor/kernel/flex_attention.py`
- `/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/_inductor/npu_triton_heuristics.py`
- `/usr/local/python3.11.14/lib/python3.11/site-packages/torch/_inductor/async_compile.py`

No `/vllm-workspace` file was directly modified in this session. The volatile image files patched here are Python site-packages files listed above.

## What Changed

### flex_attention2.py

- Delayed `torch_npu._inductor` import until NPU Flex Attention actually needs it.
- Added `--device auto`; default resolves to NPU when available, otherwise CPU.
- CPU Flex Attention path uses eager execution instead of `torch.compile`.
- NPU `--dynamic-compile` is ignored by default because the path is unstable; use `--allow-npu-dynamic-compile` to force it.
- Explicit NPU request gives a clearer runtime error if `torch.npu.is_available()` is false in the current process/container.
- `manualattention()` now has a `debug=False` flag and no longer prints during every warmup/repeat iteration by default, because that print polluted wall-clock benchmark timing.
- Added experiment-only knobs:
  - `--prescale-qk`
  - `--num-warps`
  - `--num-stages`
  These are passed through msprof child runs as well.

### summarize_msprof.py

- The script now writes the same summary it prints to `<root>/result.log` by default.
- Use `--result-log PATH` to override the log path.
- Use `--no-result-log` to print only to stdout.
- Generated logs from this session:
  - `/CYT_fileSys_2/Code1/flex_test/msprof_out/flex_vs_manual/result.log`
  - `/CYT_fileSys_2/Code1/flex_test/msprof_out/new_run/result.log`

### torch_npu/_inductor/kernel/flex_attention.py

- Added Python-side compile prints:
  - `正在编译 torch_npu flex_attention forward kernel ...`
  - `正在编译 torch_npu flex_attention backward kernel ...`
- Removed device-side `tl.device_print()` after it caused Triton/Ascend compilation failures.
- Left a host-side autotuner hook attempt in place; the final runtime proof comes from `async_compile.py`.
- Added a conservative causal SDPA fastpath in the forward lowering:
  - Requires identity `score_mod`.
  - Requires mask graph exactly equivalent to `m >= n`.
  - Requires no extra score/mask buffers and no logsumexp output request.
  - Requires 4D same-dtype, same-device, square non-GQA Q/K/V.
  - Emits `aten.scaled_dot_product_attention.default(..., is_causal=True, scale=scale, enable_gqa=False)` via `FallbackKernel`.
  - Falls back to the original generic Flex Triton path when any guard fails.

### torch_npu/_inductor/npu_triton_heuristics.py

- Added helper functions and a host-side print attempt in `NPUCachingAutotuner.run()`.
- This path did not catch the final timed-loop launch for the observed generated code, but the changes are harmless and kept for additional visibility if that path is used.

### torch/_inductor/async_compile.py

- Added a narrow wrapper around `AsyncCompile.triton(...)` results.
- It only activates when:
  - `device_str == "npu"`
  - the generated source looks like flex attention.
- It wraps the returned kernel object's `.run(...)` method and prints:
  - `正在运行 torch_npu flex_attention forward kernel run#N`
  - or `正在运行 torch_npu flex_attention backward kernel run#N`

## Portable Copies

Final copies are stored under:

- `/CYT_fileSys_2/Code1/flex_test/Newest/workspace/flex_attention2.py`
- `/CYT_fileSys_2/Code1/flex_test/Newest/site-packages/torch_npu/_inductor/kernel/flex_attention.py`
- `/CYT_fileSys_2/Code1/flex_test/Newest/site-packages/torch_npu/_inductor/npu_triton_heuristics.py`
- `/CYT_fileSys_2/Code1/flex_test/Newest/site-packages/torch/_inductor/async_compile.py`
- `/CYT_fileSys_2/Code1/flex_test/Newest/flex_attention_npu_optimization_report.md`

Use this script after a fresh image reset:

```bash
bash /CYT_fileSys_2/Code1/flex_test/Newest/apply_newest.sh
```

The script backs up target files under `Newest/backups/<timestamp>/` before replacing them.

## Test Commands

Normal run:

```bash
python3 flex_attention2.py
```

Force recompilation:

```bash
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_probe \
python3 flex_attention2.py
```

Expected debug markers:

```text
[torch_npu flex_attention debug] 正在编译 torch_npu flex_attention forward kernel ...
[torch_npu flex_attention debug] 正在运行 torch_npu flex_attention forward kernel run#1
```

Expected optimized fastpath marker for the default causal test:

```text
[torch_npu flex_attention debug] 使用 torch_npu causal SDPA fastpath 替代通用 flex_attention Triton kernel
```
