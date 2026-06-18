#!/usr/bin/env python3
"""Run manual reference attention for sparse configs in isolated subprocesses."""

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
    returncode: int
    elapsed_s: float
    manual_ms: float | None
    stdout_tail: list[str]
    stderr_tail: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual reference attention matrix.")
    parser.add_argument("--seq-lens", default=",".join(str(x) for x in DEFAULT_SEQ_LENS))
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output-json", default="manual_reference_matrix.json")
    parser.add_argument("--output-md", default="docs/manual_reference_matrix.md")
    return parser.parse_args()


def parse_csv_ints(spec: str) -> list[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def parse_csv_strings(spec: str) -> list[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def find_manual_ms(stdout: str) -> float | None:
    for line in stdout.splitlines():
        clean = strip_ansi(line)
        if "Manual Attention avg:" in clean:
            match = re.search(r"([0-9]+\.[0-9]+)\s*ms", clean)
            if match:
                return float(match.group(1))
    return None


def status(result: CaseResult | None) -> str:
    if result is None:
        return "-"
    if result.returncode == 0:
        return "OK"
    if result.returncode == 124:
        return "TIMEOUT"
    if result.returncode in (139, -11):
        return "CRASH"
    return f"ERR({result.returncode})"


def env_for(seq_len: int, config: str) -> dict[str, str]:
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = (
        "/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/lib:"
        "/usr/local/python3.11.14/lib:"
        + env.get("LD_LIBRARY_PATH", "")
    )
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}"
    env["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/torchinductor_manual_ref_{config}_{seq_len}"
    return env


def run_case(args: argparse.Namespace, seq_len: int, config: str) -> CaseResult:
    shape = f"{args.batch},{args.heads},{seq_len},{args.head_dim}"
    cmd = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--shape",
        shape,
        "--sparse-config",
        config,
        "--target",
        "manual",
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--no-compare",
    ]
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env_for(seq_len, config),
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
            returncode=proc.returncode,
            elapsed_s=elapsed_s,
            manual_ms=find_manual_ms(stdout),
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
            returncode=124,
            elapsed_s=elapsed_s,
            manual_ms=find_manual_ms(stdout),
            stdout_tail=stdout.splitlines()[-12:],
            stderr_tail=stderr.splitlines()[-12:],
        )


def render_markdown(args: argparse.Namespace, results: list[CaseResult]) -> str:
    by_key = {(r.seq_len, r.sparse_config): r for r in results}
    seq_lens = sorted({r.seq_len for r in results})
    configs = sorted({r.sparse_config for r in results})
    lines = [
        "# Manual Reference Attention Matrix",
        "",
        f"- Shape: `B={args.batch}, H={args.heads}, S, D={args.head_dim}`",
        f"- Warmup / Repeat: `{args.warmup}/{args.repeat}`",
        "- Command: `flex_attention_run_script.py --target manual --no-compare`",
        "",
        "| Config | " + " | ".join(f"S={s}" for s in seq_lens) + " | Notes |",
        "|--------|" + "|".join(":--:" for _ in seq_lens) + "|-------|",
    ]
    for config in configs:
        cells = []
        notes = []
        for seq_len in seq_lens:
            result = by_key.get((seq_len, config))
            if result and result.returncode == 0 and result.manual_ms is not None:
                cells.append(f"{result.manual_ms:.3f}")
            else:
                cells.append(status(result))
                if result:
                    notes.append(f"S={seq_len}:{status(result)}")
        lines.append(f"| `{config}` | " + " | ".join(cells) + f" | {'; '.join(notes) if notes else '-'} |")
    return "\n".join(lines) + "\n"


def print_case(result: CaseResult) -> None:
    ms = f"{result.manual_ms:.3f} ms" if result.manual_ms is not None else "N/A"
    print(
        f"  [{result.sparse_config}] S={result.seq_len:<6d} {status(result):8s} "
        f"{ms:>10s} ({result.elapsed_s:.0f}s)",
        flush=True,
    )
    if result.returncode != 0:
        for line in result.stdout_tail[-4:]:
            print(f"    stdout: {line}", flush=True)
        for line in result.stderr_tail[-4:]:
            print(f"    stderr: {line}", flush=True)


def main() -> int:
    args = parse_args()
    seq_lens = parse_csv_ints(args.seq_lens)
    configs = parse_csv_strings(args.configs)
    results: list[CaseResult] = []
    print("=== Manual Reference Attention Matrix ===", flush=True)
    print(
        f"configs={configs} seq_lens={seq_lens} "
        f"shape=({args.batch},{args.heads},S,{args.head_dim})",
        flush=True,
    )
    for seq_len in seq_lens:
        print(f"\n### S={seq_len}", flush=True)
        for config in configs:
            result = run_case(args, seq_len, config)
            results.append(result)
            print_case(result)
            time.sleep(1)

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

    print("\n=== Summary ===")
    print(render_markdown(args, results), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
