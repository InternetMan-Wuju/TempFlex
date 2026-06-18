#!/usr/bin/env python3
"""Autotune fair external reorder variants for long-sequence Flex Attention.

The comparison is always:
  identity external output vs reorder external output

This keeps both sides on the same Q/metadata gather path and changes only the
row/KV order. Each variant runs in a fresh subprocess to avoid compile-cache
cross-talk on NPU.
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
TMP_DIR = ROOT / "tmp_fair_reorder_autotune"

DEFAULT_SEQ_LENS = "65536,81920"
DEFAULT_CONFIGS = "band_global_bs,hybrid_sparse_bs,nested_bs,strided_bs,sliding_window_128_bs,prefix_lm_bs"
DEFAULT_REORDER_MODES = "wave_overlap,wave_union_fast"
DEFAULT_KV_ORDERS = "asc,snake_inv,boundary_dp,edge_dp,union_boundary_dp"


@dataclass
class RunResult:
    status: str
    ms: float | None


@dataclass
class TrialStats:
    values: list[float]
    mean_ms: float | None
    median_ms: float | None
    min_ms: float | None
    max_ms: float | None
    std_ms: float | None


@dataclass
class AutotuneResult:
    seq_len: int
    sparse_config: str
    reorder_mode: str
    kv_order: str
    identity_statuses: list[str]
    reorder_statuses: list[str]
    identity: TrialStats
    reorder: TrialStats
    speedup_mean: float | None
    speedup_median: float | None
    max_abs_diff: float | None
    mean_abs_diff: float | None
    max_rel_diff: float | None
    allclose: bool | None
    pass_threshold: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fair A/B autotune for external reorder variants.")
    parser.add_argument("--seq-lens", default=DEFAULT_SEQ_LENS)
    parser.add_argument("--configs", default=DEFAULT_CONFIGS)
    parser.add_argument("--reorder-modes", default=DEFAULT_REORDER_MODES)
    parser.add_argument("--kv-orders", default=DEFAULT_KV_ORDERS)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=2400)
    parser.add_argument("--wave-size", type=int, default=132)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rtol", type=float, default=0.03)
    parser.add_argument("--atol", type=float, default=0.03)
    parser.add_argument("--speedup-threshold", type=float, default=1.01)
    parser.add_argument("--max-abs-threshold", type=float, default=0.00390625)
    parser.add_argument("--output-json", default="fair_reorder_autotune.json")
    parser.add_argument("--output-md", default="docs/fair_reorder_autotune.md")
    parser.add_argument(
        "--keep-outputs",
        action="store_true",
        help="Do not delete saved tensors in tmp_fair_reorder_autotune.",
    )
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


def env_for(seq_len: int, config: str, tag: str, mode: str, kv_order: str, trial: int) -> dict[str, str]:
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = (
        "/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/lib:"
        "/usr/local/python3.11.14/lib:"
        + env.get("LD_LIBRARY_PATH", "")
    )
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}"
    env["TORCHINDUCTOR_CACHE_DIR"] = (
        f"/tmp/torchinductor_fair_autotune_{tag}_{config}_{seq_len}_{mode}_{kv_order}_{trial}"
    )
    return env


def run_variant(
    args: argparse.Namespace,
    seq_len: int,
    config: str,
    tag: str,
    reorder_mode: str,
    kv_order: str,
    trial: int,
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
            env=env_for(seq_len, config, tag, reorder_mode, kv_order, trial),
            text=True,
            capture_output=True,
            timeout=args.timeout,
        )
        return RunResult(status(proc.returncode), find_ms(proc.stdout or ""))
    except subprocess.TimeoutExpired as exc:
        return RunResult("TIMEOUT", find_ms(exc.stdout or ""))


def stats(values: list[float]) -> TrialStats:
    if not values:
        return TrialStats([], None, None, None, None, None)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return TrialStats(
        values=values,
        mean_ms=float(statistics.mean(values)),
        median_ms=float(statistics.median(values)),
        min_ms=float(min(values)),
        max_ms=float(max(values)),
        std_ms=float(std),
    )


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


def render_md(results: list[AutotuneResult], args: argparse.Namespace) -> str:
    lines = [
        "# Fair Reorder Autotune",
        "",
        "- Method: identity external output vs reorder external output",
        f"- Shape: `B={args.batch}, H={args.heads}, D={args.head_dim}`",
        f"- Kernel block: `BLOCK_M={args.block_m}, BLOCK_N={args.block_n}`",
        f"- Timing: `warmup={args.warmup}, repeat={args.repeat}, trials={args.trials}`",
        f"- Pass threshold: speedup_mean >= `{args.speedup_threshold}` and allclose with max_abs <= `{args.max_abs_threshold}`",
        "",
        "| Seq Len | Config | Mode | KV order | Identity mean | Reorder mean | Speedup mean | Speedup median | allclose | max_abs | Pass |",
        "|--------:|--------|------|----------|--------------:|-------------:|-------------:|---------------:|----------|--------:|------|",
    ]
    for r in results:
        lines.append(
            f"| {r.seq_len} | `{r.sparse_config}` | `{r.reorder_mode}` | `{r.kv_order}` | "
            f"{fmt(r.identity.mean_ms)} | {fmt(r.reorder.mean_ms)} | {fmt(r.speedup_mean, 4)}x | "
            f"{fmt(r.speedup_median, 4)}x | {r.allclose if r.allclose is not None else '-'} | "
            f"{fmt(r.max_abs_diff, 6)} | {r.pass_threshold} |"
        )
    winners = [r for r in results if r.pass_threshold]
    lines.extend(["", "## Selector candidates", ""])
    if winners:
        for r in winners:
            lines.append(
                f"- `{r.sparse_config}@{r.seq_len}`: `{r.reorder_mode}` + `{r.kv_order}` "
                f"speedup_mean={r.speedup_mean:.4f}x"
            )
    else:
        lines.append("- No candidate reached the pass threshold.")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    seq_lens = csv_ints(args.seq_lens)
    configs = csv_strings(args.configs)
    reorder_modes = csv_strings(args.reorder_modes)
    kv_orders = csv_strings(args.kv_orders)

    results: list[AutotuneResult] = []
    print("=== Fair Reorder Autotune ===", flush=True)
    for seq_len in seq_lens:
        for config in configs:
            for mode in reorder_modes:
                for kv_order in kv_orders:
                    print(f"\n### S={seq_len} {config} mode={mode} kv={kv_order}", flush=True)
                    identity_statuses: list[str] = []
                    reorder_statuses: list[str] = []
                    identity_ms: list[float] = []
                    reorder_ms: list[float] = []
                    first_identity_path = None
                    first_reorder_path = None
                    for trial in range(args.trials):
                        identity_path = TMP_DIR / f"{seq_len}_{config}_{mode}_{kv_order}_t{trial}_identity.pt"
                        reorder_path = TMP_DIR / f"{seq_len}_{config}_{mode}_{kv_order}_t{trial}_reorder.pt"
                        ident = run_variant(args, seq_len, config, "identity", mode, kv_order, trial, identity_path)
                        reo = run_variant(args, seq_len, config, "reorder", mode, kv_order, trial, reorder_path)
                        identity_statuses.append(ident.status)
                        reorder_statuses.append(reo.status)
                        if ident.ms is not None and ident.status == "OK":
                            identity_ms.append(ident.ms)
                        if reo.ms is not None and reo.status == "OK":
                            reorder_ms.append(reo.ms)
                        if first_identity_path is None and ident.status == "OK" and reo.status == "OK":
                            first_identity_path = identity_path
                            first_reorder_path = reorder_path
                        print(f"  trial={trial} identity={ident.status} {ident.ms} reorder={reo.status} {reo.ms}", flush=True)
                        time.sleep(1)

                    ident_stats = stats(identity_ms)
                    reo_stats = stats(reorder_ms)
                    speedup_mean = (
                        ident_stats.mean_ms / reo_stats.mean_ms
                        if ident_stats.mean_ms is not None and reo_stats.mean_ms is not None
                        else None
                    )
                    speedup_median = (
                        ident_stats.median_ms / reo_stats.median_ms
                        if ident_stats.median_ms is not None and reo_stats.median_ms is not None
                        else None
                    )
                    max_abs = mean_abs = max_rel = None
                    close = None
                    if first_identity_path is not None and first_reorder_path is not None:
                        max_abs, mean_abs, max_rel, close = compare_tensors(
                            first_identity_path,
                            first_reorder_path,
                            args.rtol,
                            args.atol,
                        )
                    passed = bool(
                        speedup_mean is not None
                        and speedup_mean >= args.speedup_threshold
                        and close is True
                        and max_abs is not None
                        and max_abs <= args.max_abs_threshold
                    )
                    print(
                        f"  summary speedup_mean={speedup_mean} allclose={close} max_abs={max_abs} pass={passed}",
                        flush=True,
                    )
                    results.append(
                        AutotuneResult(
                            seq_len=seq_len,
                            sparse_config=config,
                            reorder_mode=mode,
                            kv_order=kv_order,
                            identity_statuses=identity_statuses,
                            reorder_statuses=reorder_statuses,
                            identity=ident_stats,
                            reorder=reo_stats,
                            speedup_mean=speedup_mean,
                            speedup_median=speedup_median,
                            max_abs_diff=max_abs,
                            mean_abs_diff=mean_abs,
                            max_rel_diff=max_rel,
                            allclose=close,
                            pass_threshold=passed,
                        )
                    )

    (ROOT / args.output_json).write_text(
        json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    out_md = ROOT / args.output_md
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_md(results, args), encoding="utf-8")
    if not args.keep_outputs:
        for path in TMP_DIR.glob("*.pt"):
            path.unlink(missing_ok=True)
    print(f"\nJSON written to {ROOT / args.output_json}")
    print(f"Markdown written to {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
