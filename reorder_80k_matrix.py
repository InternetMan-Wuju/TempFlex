#!/usr/bin/env python3
"""Long-sequence reorder benchmark matrix for Flex Attention on NPU.

This script focuses on the sparse patterns that are most relevant for reorder
evaluation at 80K+ sequence lengths. Each test case runs in a fresh subprocess
to avoid torch.compile / Inductor cache pollution between baseline and reorder
variants.

Default matrix:
  - Sequence lengths: 65536, 81920, 98304
  - Sparse configs: reorder-focused non-causal sparse modes
  - Variants:
      * baseline: identity external Q/metadata path
      * reorder : patent-style external Q/metadata reorder path

Typical usage:
  python3 reorder_80k_matrix.py
  python3 reorder_80k_matrix.py --seq-lens 81920 --configs sliding_window_128_bs,strided_bs
  python3 reorder_80k_matrix.py --baseline-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MAIN_SCRIPT = SCRIPT_DIR / "flex_attention_run_script.py"

DEFAULT_SEQ_LENS = [65536, 81920, 98304]
DEFAULT_CONFIGS = [
    "sliding_window_128_bs",
    "strided_bs",
    "nested_bs",
    "hybrid_sparse_bs",
    "dilated_window_bs",
    "block_diagonal_64_bs",
    "random_block_sparse",
]


@dataclass
class CaseResult:
    seq_len: int
    sparse_config: str
    variant: str
    batch: int
    heads: int
    head_dim: int
    returncode: int
    elapsed_s: float
    flex_ms: float | None
    reorder_ms: float | None
    reorder_comp_ms: float | None
    identity_perm: bool | None
    used_internal_perm_template: bool
    raw_stdout_tail: list[str]
    raw_stderr_tail: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 80K reorder-focused Flex Attention benchmark matrix.",
    )
    parser.add_argument(
        "--seq-lens",
        default=",".join(str(x) for x in DEFAULT_SEQ_LENS),
        help="Comma-separated sequence lengths. Default: 65536,81920,98304",
    )
    parser.add_argument(
        "--configs",
        default=",".join(DEFAULT_CONFIGS),
        help="Comma-separated sparse configs to test.",
    )
    parser.add_argument("--batch", type=int, default=1, help="Batch size.")
    parser.add_argument("--heads", type=int, default=2, help="Number of attention heads.")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension.")
    parser.add_argument("--device", default="auto", help="Target device.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations.")
    parser.add_argument("--repeat", type=int, default=8, help="Measured iterations.")
    parser.add_argument("--wave-size", type=int, default=132, help="Reorder wave size.")
    parser.add_argument("--dtype", default="bfloat16", help="Input dtype.")
    parser.add_argument(
        "--block-reorder-mode",
        default="wave_overlap",
        help="Reorder mode passed through to flex_attention_run_script.py.",
    )
    parser.add_argument(
        "--block-reorder-impl",
        choices=["internal", "external"],
        default="external",
        help="Reorder implementation to test. Default: external patent-style Q/metadata gather.",
    )
    parser.add_argument(
        "--baseline-kind",
        choices=["flex", "identity_external"],
        default="identity_external",
        help="Baseline variant. identity_external uses the same external path with identity perm for stable long-sequence A/B.",
    )
    parser.add_argument(
        "--kv-order",
        choices=["asc", "desc", "snake", "snake_inv", "boundary_dp", "edge_dp"],
        default="asc",
        help="KV column order for the reorder variant. Baseline stays asc.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Per-subprocess timeout in seconds.",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only the baseline variant.",
    )
    parser.add_argument(
        "--reorder-only",
        action="store_true",
        help="Run only the reorder variant.",
    )
    parser.add_argument(
        "--keep-causal-fastpath",
        action="store_true",
        help="Keep causal fastpath enabled even for reorder runs.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional JSON file path for machine-readable results.",
    )
    parser.add_argument(
        "--output-md",
        default=None,
        help="Optional Markdown summary path.",
    )
    args = parser.parse_args()
    if args.baseline_only and args.reorder_only:
        parser.error("--baseline-only and --reorder-only are mutually exclusive")
    return args


def parse_csv_ints(spec: str) -> list[int]:
    values = [part.strip() for part in spec.split(",") if part.strip()]
    return [int(x) for x in values]


def parse_csv_strings(spec: str) -> list[str]:
    return [part.strip() for part in spec.split(",") if part.strip()]


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def find_float_after(prefix: str, text: str) -> float | None:
    for line in text.splitlines():
        clean = strip_ansi(line)
        if prefix in clean:
            m = re.search(r"([0-9]+\.[0-9]+)\s*ms", clean)
            if m:
                return float(m.group(1))
    return None


def build_command(args: argparse.Namespace, seq_len: int, sparse_config: str, variant: str) -> list[str]:
    shape = f"{args.batch},{args.heads},{seq_len},{args.head_dim}"
    target = "reorder" if variant == "reorder" or args.baseline_kind == "identity_external" else "flex"
    cmd = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--shape", shape,
        "--sparse-config", sparse_config,
        "--target", target,
        "--device", args.device,
        "--dtype", args.dtype,
        "--warmup", str(args.warmup),
        "--repeat", str(args.repeat),
        "--no-compare",
    ]
    if not args.keep_causal_fastpath:
        cmd.append("--no-causal-fastpath")
    if variant == "baseline" and args.baseline_kind == "identity_external":
        cmd.extend(
            [
                "--enable-block-reorder",
                "--block-reorder-impl", "external",
                "--block-reorder-mode", "identity",
                "--allow-identity-reorder",
                "--wave-size", str(args.wave_size),
            ]
        )
    elif variant == "reorder":
        cmd.extend(
            [
                "--enable-block-reorder",
                "--block-reorder-impl", args.block_reorder_impl,
                "--block-reorder-mode", args.block_reorder_mode,
                "--wave-size", str(args.wave_size),
                "--kv-order", args.kv_order,
            ]
        )
    return cmd


def run_case(args: argparse.Namespace, seq_len: int, sparse_config: str, variant: str) -> CaseResult:
    cmd = build_command(args, seq_len, sparse_config, variant)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = (
        "/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/lib:"
        + env.get("LD_LIBRARY_PATH", "")
    )
    env["PYTHONPATH"] = f"{SCRIPT_DIR}:{env.get('PYTHONPATH', '')}"

    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            env=env,
        )
        elapsed = time.perf_counter() - started
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        identity_perm = None
        if "[reorder] Identity permutation" in stdout:
            identity_perm = True
        elif "[reorder] Non-identity permutation computed" in stdout:
            identity_perm = False
        return CaseResult(
            seq_len=seq_len,
            sparse_config=sparse_config,
            variant=variant,
            batch=args.batch,
            heads=args.heads,
            head_dim=args.head_dim,
            returncode=proc.returncode,
            elapsed_s=elapsed,
            flex_ms=find_float_after("Flex Attention avg:", stdout),
            reorder_ms=find_float_after("Flex+", stdout),
            reorder_comp_ms=find_float_after("reorder computation time", stdout),
            identity_perm=identity_perm,
            used_internal_perm_template="pure block-sparse + internal PERM template" in stdout,
            raw_stdout_tail=stdout.splitlines()[-12:],
            raw_stderr_tail=stderr.splitlines()[-12:],
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        return CaseResult(
            seq_len=seq_len,
            sparse_config=sparse_config,
            variant=variant,
            batch=args.batch,
            heads=args.heads,
            head_dim=args.head_dim,
            returncode=124,
            elapsed_s=elapsed,
            flex_ms=find_float_after("Flex Attention avg:", stdout),
            reorder_ms=find_float_after("Flex+", stdout),
            reorder_comp_ms=find_float_after("reorder computation time", stdout),
            identity_perm=None,
            used_internal_perm_template="pure block-sparse + internal PERM template" in stdout,
            raw_stdout_tail=stdout.splitlines()[-12:],
            raw_stderr_tail=stderr.splitlines()[-12:],
        )


def render_status(result: CaseResult) -> str:
    if result.returncode == 0:
        return "OK"
    if result.returncode == 124:
        return "TIMEOUT"
    if result.returncode == 139:
        return "CRASH"
    return f"ERR({result.returncode})"


def results_to_markdown(results: list[CaseResult]) -> str:
    by_key: dict[tuple[int, str], dict[str, CaseResult]] = {}
    for result in results:
        by_key.setdefault((result.seq_len, result.sparse_config), {})[result.variant] = result

    lines = [
        "# Reorder 80K Matrix",
        "",
        "| Seq Len | Config | Baseline | Reorder | Speedup | Notes |",
        "|--------:|--------|---------:|--------:|--------:|-------|",
    ]
    for (seq_len, config), variants in sorted(by_key.items()):
        base = variants.get("baseline")
        reo = variants.get("reorder")
        base_time = None
        if base:
            base_time = base.flex_ms if base.flex_ms is not None else base.reorder_ms
        base_ms = f"{base_time:.3f}" if base_time is not None else render_status(base) if base else "-"
        reo_ms = "-"
        speedup = "-"
        notes: list[str] = []
        if reo:
            if reo.reorder_ms is not None:
                reo_ms = f"{reo.reorder_ms:.3f}"
            elif reo.flex_ms is not None:
                reo_ms = f"{reo.flex_ms:.3f}"
            else:
                reo_ms = render_status(reo)
            if base_time and reo and reo.reorder_ms:
                speedup = f"{base_time / reo.reorder_ms:.4f}x"
            elif base_time and reo and reo.flex_ms:
                speedup = f"{base_time / reo.flex_ms:.4f}x"
            if reo.identity_perm is True:
                notes.append("identity")
            if reo.used_internal_perm_template:
                notes.append("internal_perm")
            if reo.returncode == 139:
                notes.append("crash")
        lines.append(
            f"| {seq_len} | {config} | {base_ms} | {reo_ms} | {speedup} | {', '.join(notes) if notes else '-'} |"
        )
    return "\n".join(lines) + "\n"


def print_case_summary(result: CaseResult) -> None:
    ms = result.reorder_ms if result.reorder_ms is not None else result.flex_ms
    ms_str = f"{ms:.3f} ms" if ms is not None else "N/A"
    perm_tag = ""
    if result.variant == "reorder":
        if result.identity_perm is True:
            perm_tag = " identity"
        elif result.identity_perm is False:
            perm_tag = " non-id"
    tmpl_tag = " internal-perm" if result.used_internal_perm_template else ""
    print(
        f"  [{result.variant}] {render_status(result):8s} {ms_str:>10s}"
        f"  ({result.elapsed_s:.0f}s){perm_tag}{tmpl_tag}"
    )
    if result.returncode != 0:
        for line in result.raw_stdout_tail[-4:]:
            print(f"    stdout: {line}")
        for line in result.raw_stderr_tail[-4:]:
            print(f"    stderr: {line}")


def main() -> int:
    args = parse_args()
    seq_lens = parse_csv_ints(args.seq_lens)
    configs = parse_csv_strings(args.configs)
    variants = ["baseline", "reorder"]
    if args.baseline_only:
        variants = ["baseline"]
    if args.reorder_only:
        variants = ["reorder"]

    results: list[CaseResult] = []
    print("=== Reorder 80K Matrix ===")
    print(
        f"configs={configs} seq_lens={seq_lens} "
        f"shape=({args.batch},{args.heads},S,{args.head_dim}) "
        f"baseline_kind={args.baseline_kind} reorder_impl={args.block_reorder_impl} "
        f"kv_order={args.kv_order}"
    )

    for seq_len in seq_lens:
        print(f"\n### S={seq_len}")
        for config in configs:
            print(f"- {config}")
            for variant in variants:
                result = run_case(args, seq_len, config, variant)
                results.append(result)
                print_case_summary(result)
                time.sleep(1)

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.write_text(
            json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nJSON written to {output_json}")

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.write_text(results_to_markdown(results), encoding="utf-8")
        print(f"Markdown written to {output_md}")

    print("\n=== Summary ===")
    print(f"{'seq_len':>8s}  {'config':24s}  {'variant':8s}  {'status':8s}  {'time_ms':>10s}")
    for result in results:
        ms = result.reorder_ms if result.reorder_ms is not None else result.flex_ms
        ms_str = f"{ms:.3f}" if ms is not None else "-"
        print(
            f"{result.seq_len:8d}  {result.sparse_config:24s}  {result.variant:8s}  "
            f"{render_status(result):8s}  {ms_str:>10s}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
