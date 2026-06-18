#!/usr/bin/env python3
"""Build a Raw/Newest/Reorder/Manual performance comparison table.

The matrix intentionally runs every measured cell in a fresh subprocess. Raw
and Newest are applied before their corresponding cells so Python modules and
torch.compile state do not mix across implementations.
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


ROOT = Path(__file__).resolve().parent
MAIN_SCRIPT = ROOT / "flex_attention_run_script.py"
APPLY_RAW = ROOT / "raw_flex" / "apply_raw.sh"
APPLY_NEWEST = ROOT / "Newest" / "apply_newest.sh"

DEFAULT_SEQ_LENS = [1024, 4096, 16384]
DEFAULT_CONFIGS = [
    "causal",
    "block_diagonal_64_bs",
    "checkerboard_64_bs",
    "sliding_window_128_bs",
    "strided_bs",
    "dilated_window_bs",
    "nested_bs",
    "hybrid_sparse_bs",
    "global_local_bs",
    "multiscale_dilated_bs",
    "prefix_lm_bs",
    "band_global_bs",
]


@dataclass
class CaseResult:
    seq_len: int
    sparse_config: str
    variant: str
    returncode: int
    elapsed_s: float
    flex_ms: float | None
    reorder_ms: float | None
    manual_ms: float | None
    skipped: bool
    skip_reason: str | None
    stdout_tail: list[str]
    stderr_tail: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare raw flex, Newest no-reorder, Newest reorder, and manual.",
    )
    parser.add_argument("--seq-lens", default=",".join(str(x) for x in DEFAULT_SEQ_LENS))
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--manual-warmup", type=int, default=1)
    parser.add_argument("--manual-repeat", type=int, default=1)
    parser.add_argument(
        "--max-manual-seq",
        type=int,
        default=2048,
        help="Skip manual dense attention above this sequence length.",
    )
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--wave-size", type=int, default=132)
    parser.add_argument("--block-reorder-mode", default="wave_overlap")
    parser.add_argument(
        "--kv-order",
        choices=["asc", "desc", "snake", "snake_inv", "boundary_dp", "edge_dp"],
        default="snake_inv",
    )
    parser.add_argument(
        "--keep-causal-fastpath",
        action="store_true",
        help="Keep causal fastpath enabled for flex cells.",
    )
    parser.add_argument(
        "--restore-newest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restore Newest implementation after the matrix finishes.",
    )
    parser.add_argument(
        "--include-manual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the Newest manual reference column.",
    )
    parser.add_argument(
        "--versions",
        default="raw,newest",
        help="Comma-separated implementation groups to run: raw,newest.",
    )
    parser.add_argument(
        "--newest-variants",
        default="flex,reorder,manual",
        help="Comma-separated Newest variants to run: flex,reorder,manual.",
    )
    parser.add_argument("--output-md", default="docs/raw_newest_reorder_manual_matrix.md")
    parser.add_argument("--output-json", default="raw_newest_reorder_manual_matrix.json")
    return parser.parse_args()


def parse_csv_ints(spec: str) -> list[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def parse_csv_strings(spec: str) -> list[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def find_float_after(prefix: str, text: str) -> float | None:
    for line in text.splitlines():
        clean = strip_ansi(line)
        if prefix in clean:
            match = re.search(r"([0-9]+\.[0-9]+)\s*ms", clean)
            if match:
                return float(match.group(1))
    return None


def apply_version(name: str) -> None:
    script = APPLY_RAW if name == "raw" else APPLY_NEWEST
    proc = subprocess.run(
        ["bash", str(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"apply {name} failed with code {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


def build_env(seq_len: int, config: str, variant: str) -> dict[str, str]:
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = (
        "/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/lib:"
        "/usr/local/python3.11.14/lib:"
        + env.get("LD_LIBRARY_PATH", "")
    )
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}"
    env["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/torchinductor_big_matrix_{variant}_{config}_{seq_len}"
    return env


def build_command(args: argparse.Namespace, seq_len: int, config: str, variant: str) -> list[str]:
    shape = f"{args.batch},{args.heads},{seq_len},{args.head_dim}"
    if variant == "newest_manual":
        target = "manual"
        warmup = args.manual_warmup
        repeat = args.manual_repeat
    elif variant == "newest_reorder":
        target = "reorder"
        warmup = args.warmup
        repeat = args.repeat
    else:
        target = "flex"
        warmup = args.warmup
        repeat = args.repeat

    cmd = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--shape", shape,
        "--sparse-config", config,
        "--target", target,
        "--device", args.device,
        "--dtype", args.dtype,
        "--warmup", str(warmup),
        "--repeat", str(repeat),
        "--no-compare",
    ]
    if not args.keep_causal_fastpath:
        cmd.append("--no-causal-fastpath")
    if variant == "newest_reorder":
        cmd.extend(
            [
                "--enable-block-reorder",
                "--block-reorder-impl",
                "external",
                "--block-reorder-mode",
                args.block_reorder_mode,
                "--wave-size",
                str(args.wave_size),
                "--kv-order",
                args.kv_order,
            ]
        )
    return cmd


def skip_reason(args: argparse.Namespace, seq_len: int, config: str, variant: str) -> str | None:
    if variant == "newest_manual" and seq_len > args.max_manual_seq:
        return f"manual skipped above S={args.max_manual_seq}"
    if variant == "newest_reorder" and config == "causal":
        return "causal uses no reorder; compare raw/newest flex instead"
    return None


def skipped_result(seq_len: int, config: str, variant: str, reason: str) -> CaseResult:
    return CaseResult(
        seq_len=seq_len,
        sparse_config=config,
        variant=variant,
        returncode=0,
        elapsed_s=0.0,
        flex_ms=None,
        reorder_ms=None,
        manual_ms=None,
        skipped=True,
        skip_reason=reason,
        stdout_tail=[],
        stderr_tail=[],
    )


def run_case(args: argparse.Namespace, seq_len: int, config: str, variant: str) -> CaseResult:
    reason = skip_reason(args, seq_len, config, variant)
    if reason:
        return skipped_result(seq_len, config, variant, reason)

    cmd = build_command(args, seq_len, config, variant)
    env = build_env(seq_len, config, variant)
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=args.timeout,
        )
        elapsed_s = time.perf_counter() - started
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        return CaseResult(
            seq_len=seq_len,
            sparse_config=config,
            variant=variant,
            returncode=proc.returncode,
            elapsed_s=elapsed_s,
            flex_ms=find_float_after("Flex Attention avg:", stdout),
            reorder_ms=find_float_after("Flex+", stdout),
            manual_ms=find_float_after("Manual Attention avg:", stdout),
            skipped=False,
            skip_reason=None,
            stdout_tail=stdout.splitlines()[-12:],
            stderr_tail=stderr.splitlines()[-12:],
        )
    except subprocess.TimeoutExpired as exc:
        elapsed_s = time.perf_counter() - started
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        return CaseResult(
            seq_len=seq_len,
            sparse_config=config,
            variant=variant,
            returncode=124,
            elapsed_s=elapsed_s,
            flex_ms=find_float_after("Flex Attention avg:", stdout),
            reorder_ms=find_float_after("Flex+", stdout),
            manual_ms=find_float_after("Manual Attention avg:", stdout),
            skipped=False,
            skip_reason=None,
            stdout_tail=stdout.splitlines()[-12:],
            stderr_tail=stderr.splitlines()[-12:],
        )


def status(result: CaseResult | None) -> str:
    if result is None:
        return "-"
    if result.skipped:
        return "SKIP"
    if result.returncode == 0:
        return "OK"
    if result.returncode == 124:
        return "TIMEOUT"
    if result.returncode in (139, -11):
        return "CRASH"
    return f"ERR({result.returncode})"


def cell(result: CaseResult | None) -> str:
    if result is None:
        return "-"
    if result.skipped:
        return "SKIP"
    if result.returncode != 0:
        return status(result)
    value = result.reorder_ms or result.flex_ms or result.manual_ms
    return f"{value:.3f}" if value is not None else "N/A"


def speedup(numerator: CaseResult | None, denominator: CaseResult | None) -> str:
    if numerator is None or denominator is None:
        return "-"
    lhs = numerator.flex_ms or numerator.reorder_ms or numerator.manual_ms
    rhs = denominator.reorder_ms or denominator.flex_ms or denominator.manual_ms
    if not lhs or not rhs:
        return "-"
    return f"{lhs / rhs:.4f}x"


def render_markdown(args: argparse.Namespace, results: list[CaseResult]) -> str:
    by_key: dict[tuple[int, str], dict[str, CaseResult]] = {}
    for result in results:
        by_key.setdefault((result.seq_len, result.sparse_config), {})[result.variant] = result

    lines = [
        "# Raw / Newest / Reorder / Manual Matrix",
        "",
        f"- Shape: `B={args.batch}, H={args.heads}, S, D={args.head_dim}`",
        f"- Flex warmup/repeat: `{args.warmup}/{args.repeat}`",
        f"- Manual warmup/repeat: `{args.manual_warmup}/{args.manual_repeat}`",
        f"- Manual skip threshold: `S>{args.max_manual_seq}`",
        f"- Reorder: `external + {args.block_reorder_mode} + kv_order={args.kv_order}`",
        "",
        "| Seq Len | Config | Raw flex | Newest flex | Newest reorder | Manual | Raw/Newest | Newest/Reorder | Manual/Newest | Notes |",
        "|--------:|--------|---------:|------------:|---------------:|-------:|-----------:|---------------:|--------------:|-------|",
    ]
    for seq_len, config in sorted(by_key):
        variants = by_key[(seq_len, config)]
        raw = variants.get("raw_flex")
        newest = variants.get("newest_flex")
        reorder = variants.get("newest_reorder")
        manual = variants.get("newest_manual")
        notes = []
        for label, result in (
            ("raw", raw),
            ("newest", newest),
            ("reorder", reorder),
            ("manual", manual),
        ):
            if result and result.returncode != 0:
                notes.append(f"{label}={status(result)}")
            if result and result.skipped and result.skip_reason:
                notes.append(f"{label}: {result.skip_reason}")
        lines.append(
            "| "
            f"{seq_len} | `{config}` | {cell(raw)} | {cell(newest)} | {cell(reorder)} | {cell(manual)} | "
            f"{speedup(raw, newest)} | {speedup(newest, reorder)} | {speedup(manual, newest)} | "
            f"{'; '.join(notes) if notes else '-'} |"
        )
    return "\n".join(lines) + "\n"


def print_case(result: CaseResult) -> None:
    if result.skipped:
        print(f"    [{result.variant}] SKIP  {result.skip_reason}", flush=True)
        return
    print(
        f"    [{result.variant}] {status(result):8s} {cell(result):>10s} ms "
        f"({result.elapsed_s:.0f}s)",
        flush=True,
    )
    if result.returncode != 0:
        for line in result.stdout_tail[-4:]:
            print(f"      stdout: {line}", flush=True)
        for line in result.stderr_tail[-4:]:
            print(f"      stderr: {line}", flush=True)


def main() -> int:
    args = parse_args()
    seq_lens = parse_csv_ints(args.seq_lens)
    configs = parse_csv_strings(args.configs)
    versions = parse_csv_strings(args.versions)
    newest_variants = parse_csv_strings(args.newest_variants)
    unknown_versions = sorted(set(versions) - {"raw", "newest"})
    if unknown_versions:
        raise ValueError(f"unknown --versions entries: {unknown_versions}")
    unknown_newest = sorted(set(newest_variants) - {"flex", "reorder", "manual"})
    if unknown_newest:
        raise ValueError(f"unknown --newest-variants entries: {unknown_newest}")
    results: list[CaseResult] = []

    print("=== Raw / Newest / Reorder / Manual Matrix ===", flush=True)
    print(
        f"configs={configs} seq_lens={seq_lens} "
        f"shape=({args.batch},{args.heads},S,{args.head_dim})",
        flush=True,
    )

    try:
        if "raw" in versions:
            print("\n### Applying raw", flush=True)
            apply_version("raw")
            for seq_len in seq_lens:
                print(f"\nS={seq_len}", flush=True)
                for config in configs:
                    print(f"  - {config}", flush=True)
                    result = run_case(args, seq_len, config, "raw_flex")
                    results.append(result)
                    print_case(result)
                    time.sleep(1)

        if "newest" in versions:
            print("\n### Applying Newest", flush=True)
            apply_version("newest")
            for seq_len in seq_lens:
                print(f"\nS={seq_len}", flush=True)
                for config in configs:
                    print(f"  - {config}", flush=True)
                    variants = []
                    if "flex" in newest_variants:
                        variants.append("newest_flex")
                    if "reorder" in newest_variants:
                        variants.append("newest_reorder")
                    if args.include_manual and "manual" in newest_variants:
                        variants.append("newest_manual")
                    for variant in variants:
                        result = run_case(args, seq_len, config, variant)
                        results.append(result)
                        print_case(result)
                        time.sleep(1)
    finally:
        if args.restore_newest:
            print("\n### Restoring Newest", flush=True)
            apply_version("newest")

    if args.output_json:
        output_json = (ROOT / args.output_json).resolve()
        output_json.write_text(
            json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nJSON written to {output_json}", flush=True)

    if args.output_md:
        output_md = (ROOT / args.output_md).resolve()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(args, results), encoding="utf-8")
        print(f"Markdown written to {output_md}", flush=True)

    print("\n=== Summary ===", flush=True)
    print(render_markdown(args, results), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
