#!/usr/bin/env python3
"""Run the optimized fair Newest reorder matrix.

Fair means:
  Newest without reorder = identity external
  Newest with reorder    = optimized external reorder

Both sides use the same external Q/metadata path, so the speedup only compares
row/KV order changes. This script intentionally does not include Raw or Manual.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
MAIN_SCRIPT = ROOT / "flex_attention_run_script.py"
TMP_DIR = ROOT / "tmp_fair_reorder_optimized"

DEFAULT_SEQ_LENS = "16384,32768,81920"
DEFAULT_CONFIGS = (
    "causal,block_diagonal_64_bs,checkerboard_64_bs,sliding_window_128_bs,"
    "strided_bs,dilated_window_bs,nested_bs,hybrid_sparse_bs,global_local_bs,"
    "multiscale_dilated_bs,prefix_lm_bs,band_global_bs"
)


@dataclass
class RunResult:
    status: str
    ms: float | None


@dataclass
class OptimizedResult:
    seq_len: int
    sparse_config: str
    reorder_mode: str | None
    kv_order: str | None
    baseline_status: str
    reorder_status: str
    baseline_ms: float | None
    reorder_ms: float | None
    speedup: float | None
    max_abs_diff: float | None
    mean_abs_diff: float | None
    max_rel_diff: float | None
    allclose: bool | None
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimized fair Newest reorder matrix.")
    parser.add_argument("--seq-lens", default=DEFAULT_SEQ_LENS)
    parser.add_argument("--configs", default=DEFAULT_CONFIGS)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=2400)
    parser.add_argument("--wave-size", type=int, default=132)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rtol", type=float, default=0.03)
    parser.add_argument("--atol", type=float, default=0.03)
    parser.add_argument("--output-json", default="fair_reorder_optimized_16k_32k_80k.json")
    parser.add_argument("--output-md", default="docs/fair_reorder_optimized_16k_32k_80k.md")
    parser.add_argument("--keep-outputs", action="store_true")
    return parser.parse_args()


def csv_ints(spec: str) -> list[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def csv_strings(spec: str) -> list[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def find_ms(stdout: str) -> float | None:
    for line in stdout.splitlines():
        clean = strip_ansi(line)
        if "Flex+" in clean or "Flex Attention avg:" in clean:
            match = re.search(r"([0-9]+\.[0-9]+)\s*ms", clean)
            if match:
                return float(match.group(1))
    return None


def status(returncode: int) -> str:
    if returncode == 0:
        return "OK"
    if returncode == 124:
        return "TIMEOUT"
    if returncode in (139, -11):
        return "CRASH"
    return f"ERR({returncode})"


def choose_reorder(seq_len: int, config: str) -> tuple[str | None, str | None, str]:
    if config == "causal":
        return None, None, "skip: causal uses causal fastpath; reorder is not enabled"
    if seq_len >= 65536:
        if config in {"hybrid_sparse_bs"}:
            return (
                "auction_union_exact_path",
                "union_boundary_dp",
                "80K optimized probe: exact path plus union_boundary_dp",
            )
        if config in {"nested_bs", "strided_bs", "band_global_bs"}:
            return (
                "auction_union_exact_path",
                "snake_inv",
                "80K optimized probe: exact path plus stable snake_inv",
            )
        return (
            "wave_overlap",
            "snake_inv",
            "fallback: no validated auction/exact-path win for this 80K config",
        )
    if seq_len >= 32768:
        if config == "hybrid_sparse_bs":
            return (
                "auction_union_fast",
                "union_boundary_dp",
                "32K optimized probe: hybrid best KV orientation",
            )
        if config in {"nested_bs", "strided_bs"}:
            return (
                "auction_union_fast",
                "snake_inv",
                "32K optimized probe: auction_union_fast repeat=20 candidate",
            )
        return "wave_overlap", "snake_inv", "fallback: no validated auction win for this 32K config"
    if config in {"hybrid_sparse_bs", "nested_bs", "strided_bs", "prefix_lm_bs"}:
        return (
            "auction_union_fast",
            "snake_inv",
            "16K optimized probe: auction_union_fast on reorder-positive config",
        )
    return (
        "wave_overlap",
        "snake_inv",
        "fallback: no validated auction win for this 16K config",
    )


def env_for(seq_len: int, config: str, tag: str, mode: str | None, kv_order: str | None) -> dict[str, str]:
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = (
        "/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/lib:"
        "/usr/local/python3.11.14/lib:"
        + env.get("LD_LIBRARY_PATH", "")
    )
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}"
    mode_part = mode or "skip"
    kv_part = kv_order or "none"
    env["TORCHINDUCTOR_CACHE_DIR"] = (
        f"/tmp/torchinductor_fair_opt_{tag}_{config}_{seq_len}_{mode_part}_{kv_part}"
    )
    return env


def run_variant(
    args: argparse.Namespace,
    seq_len: int,
    config: str,
    tag: str,
    reorder_mode: str,
    kv_order: str,
    out_path: Path,
) -> RunResult:
    cmd = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--shape",
        f"{args.batch},{args.heads},{seq_len},{args.head_dim}",
        "--sparse-config",
        config,
        "--target",
        "reorder",
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--no-compare",
        "--no-causal-fastpath",
        "--enable-block-reorder",
        "--block-reorder-impl",
        "external",
        "--block-reorder-mode",
        "identity" if tag == "identity" else reorder_mode,
        "--wave-size",
        str(args.wave_size),
        "--block-m",
        str(args.block_m),
        "--block-n",
        str(args.block_n),
        "--kv-order",
        "asc" if tag == "identity" else kv_order,
        "--save-output",
        str(out_path),
    ]
    if tag == "identity":
        cmd.append("--allow-identity-reorder")
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env_for(seq_len, config, tag, reorder_mode, kv_order),
            text=True,
            capture_output=True,
            timeout=args.timeout,
        )
        return RunResult(status(proc.returncode), find_ms(proc.stdout or ""))
    except subprocess.TimeoutExpired as exc:
        return RunResult("TIMEOUT", find_ms(exc.stdout or ""))


def compare_tensors(a_path: Path, b_path: Path, rtol: float, atol: float) -> tuple[float, float, float, bool]:
    a = torch.load(a_path, map_location="cpu").float()
    b = torch.load(b_path, map_location="cpu").float()
    diff = (a - b).abs()
    rel = diff / b.abs().clamp_min(1e-6)
    return (
        float(diff.max().item()),
        float(diff.mean().item()),
        float(rel.max().item()),
        bool(torch.allclose(a, b, rtol=rtol, atol=atol)),
    )


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.{digits}f}"


def render_md(results: list[OptimizedResult], args: argparse.Namespace) -> str:
    lines = [
        "# Optimized Fair Newest Reorder Matrix",
        "",
        "- Method: `Newest without reorder = identity external`, `Newest with reorder = optimized external reorder`",
        f"- Shape: `B={args.batch}, H={args.heads}, D={args.head_dim}`",
        f"- Kernel block: `BLOCK_M={args.block_m}, BLOCK_N={args.block_n}`",
        f"- Timing: `warmup={args.warmup}, repeat={args.repeat}`",
        "",
        "| Seq Len | Config | Mode | KV order | Without reorder | With reorder | Speedup | allclose | max_abs | Note |",
        "|--------:|--------|------|----------|----------------:|-------------:|--------:|----------|--------:|------|",
    ]
    for r in results:
        lines.append(
            f"| {r.seq_len} | `{r.sparse_config}` | `{r.reorder_mode or 'SKIP'}` | "
            f"`{r.kv_order or '-'}` | {fmt(r.baseline_ms)} | {fmt(r.reorder_ms)} | "
            f"{fmt(r.speedup, 4)}x | {r.allclose if r.allclose is not None else '-'} | "
            f"{fmt(r.max_abs_diff, 6)} | {r.note} |"
        )
    return "\n".join(lines) + "\n"


def write_outputs(results: list[OptimizedResult], args: argparse.Namespace) -> None:
    (ROOT / args.output_json).write_text(
        json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    out_md = ROOT / args.output_md
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_md(results, args), encoding="utf-8")


def main() -> int:
    args = parse_args()
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    seq_lens = csv_ints(args.seq_lens)
    configs = csv_strings(args.configs)
    results: list[OptimizedResult] = []

    print("=== Optimized Fair Newest Reorder Matrix ===", flush=True)
    for seq_len in seq_lens:
        for config in configs:
            mode, kv_order, note = choose_reorder(seq_len, config)
            print(f"\n### S={seq_len} {config} mode={mode} kv={kv_order}", flush=True)
            if mode is None or kv_order is None:
                results.append(
                    OptimizedResult(
                        seq_len=seq_len,
                        sparse_config=config,
                        reorder_mode=None,
                        kv_order=None,
                        baseline_status="SKIP",
                        reorder_status="SKIP",
                        baseline_ms=None,
                        reorder_ms=None,
                        speedup=None,
                        max_abs_diff=None,
                        mean_abs_diff=None,
                        max_rel_diff=None,
                        allclose=None,
                        note=note,
                    )
                )
                write_outputs(results, args)
                print(f"  skip: {note}", flush=True)
                continue

            identity_path = TMP_DIR / f"{seq_len}_{config}_{mode}_{kv_order}_identity.pt"
            reorder_path = TMP_DIR / f"{seq_len}_{config}_{mode}_{kv_order}_reorder.pt"
            ident = run_variant(args, seq_len, config, "identity", mode, kv_order, identity_path)
            reorder = run_variant(args, seq_len, config, "reorder", mode, kv_order, reorder_path)
            speedup = (
                ident.ms / reorder.ms
                if ident.status == "OK" and reorder.status == "OK" and ident.ms is not None and reorder.ms
                else None
            )
            max_abs = mean_abs = max_rel = None
            close = None
            if ident.status == "OK" and reorder.status == "OK":
                max_abs, mean_abs, max_rel, close = compare_tensors(
                    identity_path,
                    reorder_path,
                    args.rtol,
                    args.atol,
                )
            print(
                f"  identity={ident.status} {ident.ms} reorder={reorder.status} {reorder.ms} "
                f"speedup={speedup} allclose={close} max_abs={max_abs}",
                flush=True,
            )
            results.append(
                OptimizedResult(
                    seq_len=seq_len,
                    sparse_config=config,
                    reorder_mode=mode,
                    kv_order=kv_order,
                    baseline_status=ident.status,
                    reorder_status=reorder.status,
                    baseline_ms=ident.ms if ident.status == "OK" else None,
                    reorder_ms=reorder.ms if reorder.status == "OK" else None,
                    speedup=speedup,
                    max_abs_diff=max_abs,
                    mean_abs_diff=mean_abs,
                    max_rel_diff=max_rel,
                    allclose=close,
                    note=note,
                )
            )
            write_outputs(results, args)
            time.sleep(1)

    if not args.keep_outputs:
        for path in TMP_DIR.glob("*.pt"):
            path.unlink(missing_ok=True)
    print(f"\nJSON written to {ROOT / args.output_json}")
    print(f"Markdown written to {ROOT / args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
