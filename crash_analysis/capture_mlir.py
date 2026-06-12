#!/usr/bin/env python3
"""Monkey-patch subprocess.run to capture .mlir before bishengir-compile runs."""

import os
import shutil
import subprocess as _sp

_original_run = _sp.run

def _patched_run(cmd, **kwargs):
    if isinstance(cmd, list):
        for arg in cmd:
            if isinstance(arg, str) and arg.endswith(".mlir") and os.path.isfile(arg):
                dst = os.path.join(os.path.dirname(__file__), "captured_kernel.mlir")
                shutil.copy2(arg, dst)
                print(f"\n[MLIR captured] {dst}", flush=True)
                break
    return _original_run(cmd, **kwargs)

_sp.run = _patched_run

import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))
from flex_attention_run_script import parse_args, run_shape_sweep, identity_score, causal_mask
from sparse_masks import get_sparse_config

if __name__ == "__main__":
    args = parse_args()
    torch._dynamo.config.suppress_errors = args.suppress_compile_errors

    if args.sparse_config:
        cfg = get_sparse_config(args.sparse_config)
        score_mod = cfg.get("score_mod", identity_score)
        mask_mod = cfg.get("mask_mod", causal_mask)
        kernel_opts = cfg.get("optimizations", {})
        print(f"[sparse-config] {args.sparse_config}: {cfg['description']}")
    else:
        score_mod, mask_mod, kernel_opts = identity_score, causal_mask, {}

    try:
        run_shape_sweep(args, score_mod=score_mod, mask_mod=mask_mod, kernel_options_extra=kernel_opts)
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        saved = os.path.join(os.path.dirname(__file__), "captured_kernel.mlir")
        if os.path.isfile(saved):
            print(f"[MLIR saved] {saved}")
        sys.exit(1)
