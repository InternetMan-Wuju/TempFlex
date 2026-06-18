#!/usr/bin/env python3
"""Long-sequence correctness check: identity external vs reordered external."""

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

import torch


ROOT = Path(__file__).resolve().parent
MAIN_SCRIPT = ROOT / "flex_attention_run_script.py"
TMP_DIR = ROOT / "tmp_long_correctness"

DEFAULT_CASES = [
    (32768, "block_diagonal_64_bs"),
    (32768, "sliding_window_128_bs"),
    (32768, "global_local_bs"),
    (81920, "block_diagonal_64_bs"),
    (81920, "sliding_window_128_bs"),
    (81920, "strided_bs"),
    (81920, "band_global_bs"),
]


@dataclass
class CorrectnessResult:
    seq_len: int
    sparse_config: str
    baseline_status: str
    reorder_status: str
    max_abs_diff: float | None
    mean_abs_diff: float | None
    max_rel_diff: float | None
    allclose: bool | None
    baseline_ms: float | None
    reorder_ms: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare identity external and reorder outputs.")
    parser.add_argument(
        "--cases",
        default=None,
        help="Comma-separated seq:config cases. Default uses built-in representative set.",
    )
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=0.03)
    parser.add_argument("--atol", type=float, default=0.03)
    parser.add_argument("--output-json", default="long_reorder_correctness.json")
    parser.add_argument("--output-md", default="docs/long_reorder_correctness.md")
    return parser.parse_args()


def parse_cases(spec: str | None) -> list[tuple[int, str]]:
    if not spec:
        return list(DEFAULT_CASES)
    cases: list[tuple[int, str]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        seq, config = item.split(":", 1)
        cases.append((int(seq), config))
    return cases


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


def env_for(seq_len: int, config: str, tag: str) -> dict[str, str]:
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = (
        "/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/lib:"
        "/usr/local/python3.11.14/lib:"
        + env.get("LD_LIBRARY_PATH", "")
    )
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}"
    env["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/torchinductor_long_correct_{tag}_{config}_{seq_len}"
    return env


def run_variant(args: argparse.Namespace, seq_len: int, config: str, tag: str, out_path: Path) -> tuple[str, float | None]:
    cmd = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--shape",
        f"1,2,{seq_len},128",
        "--sparse-config",
        config,
        "--target",
        "reorder",
        "--device",
        "auto",
        "--dtype",
        "bfloat16",
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
        "identity" if tag == "identity" else "wave_overlap",
        "--wave-size",
        "132",
        "--kv-order",
        "asc" if tag == "identity" else "snake_inv",
        "--save-output",
        str(out_path),
    ]
    if tag == "identity":
        cmd.append("--allow-identity-reorder")
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env_for(seq_len, config, tag),
            text=True,
            capture_output=True,
            timeout=args.timeout,
        )
        return status(proc.returncode), find_ms(proc.stdout or "")
    except subprocess.TimeoutExpired as exc:
        return "TIMEOUT", find_ms(exc.stdout or "")


def compare_tensors(a_path: Path, b_path: Path, rtol: float, atol: float) -> tuple[float, float, float, bool]:
    a = torch.load(a_path, map_location="cpu").float()
    b = torch.load(b_path, map_location="cpu").float()
    diff = (a - b).abs()
    denom = b.abs().clamp_min(1e-6)
    rel = diff / denom
    return (
        float(diff.max().item()),
        float(diff.mean().item()),
        float(rel.max().item()),
        bool(torch.allclose(a, b, rtol=rtol, atol=atol)),
    )


def render_md(results: list[CorrectnessResult], rtol: float, atol: float) -> str:
    lines = [
        "# Long Reorder Correctness",
        "",
        f"- Method: identity external output vs wave_overlap reorder output",
        f"- Tolerance: `rtol={rtol}, atol={atol}`",
        "",
        "| Seq Len | Config | Identity | Reorder | allclose | max_abs | mean_abs | max_rel | Identity ms | Reorder ms |",
        "|--------:|--------|----------|---------|----------|--------:|---------:|--------:|------------:|-----------:|",
    ]
    for r in results:
        if r.max_abs_diff is None:
            lines.append(
                f"| {r.seq_len} | `{r.sparse_config}` | {r.baseline_status} | {r.reorder_status} | "
                f"- | - | - | - | {r.baseline_ms if r.baseline_ms is not None else '-'} | {r.reorder_ms if r.reorder_ms is not None else '-'} |"
            )
        else:
            lines.append(
                f"| {r.seq_len} | `{r.sparse_config}` | {r.baseline_status} | {r.reorder_status} | "
                f"{r.allclose} | {r.max_abs_diff:.6f} | {r.mean_abs_diff:.6f} | {r.max_rel_diff:.4f} | "
                f"{r.baseline_ms:.3f} | {r.reorder_ms:.3f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    results: list[CorrectnessResult] = []
    print("=== Long Reorder Correctness ===", flush=True)
    for seq_len, config in parse_cases(args.cases):
        print(f"\n### S={seq_len} {config}", flush=True)
        identity_path = TMP_DIR / f"{seq_len}_{config}_identity.pt"
        reorder_path = TMP_DIR / f"{seq_len}_{config}_reorder.pt"
        base_status, base_ms = run_variant(args, seq_len, config, "identity", identity_path)
        print(f"  identity: {base_status} {base_ms}", flush=True)
        reorder_status, reorder_ms = run_variant(args, seq_len, config, "reorder", reorder_path)
        print(f"  reorder : {reorder_status} {reorder_ms}", flush=True)
        max_abs = mean_abs = max_rel = None
        close = None
        if base_status == "OK" and reorder_status == "OK":
            max_abs, mean_abs, max_rel, close = compare_tensors(
                identity_path,
                reorder_path,
                args.rtol,
                args.atol,
            )
            print(f"  compare : allclose={close} max_abs={max_abs:.6f} max_rel={max_rel:.4f}", flush=True)
        results.append(
            CorrectnessResult(
                seq_len=seq_len,
                sparse_config=config,
                baseline_status=base_status,
                reorder_status=reorder_status,
                max_abs_diff=max_abs,
                mean_abs_diff=mean_abs,
                max_rel_diff=max_rel,
                allclose=close,
                baseline_ms=base_ms,
                reorder_ms=reorder_ms,
            )
        )
        time.sleep(1)

    (ROOT / args.output_json).write_text(
        json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    out_md = ROOT / args.output_md
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_md(results, args.rtol, args.atol), encoding="utf-8")
    print(f"\nJSON written to {ROOT / args.output_json}")
    print(f"Markdown written to {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
