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

## Notes

Read `flex_attention_npu_debug_notes.md` in this same directory before continuing the investigation in a fresh Codex session.
