#!/usr/bin/env python3
"""Benchmark causal Flex Attention on raw_flex vs Newest.

Each case runs in a fresh subprocess after applying the requested
torch_npu flex_attention implementation. This keeps compile caches and
loaded Python modules from mixing raw/newest code in the same process.
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
APPLY = {
    "raw": ROOT / "raw_flex" / "apply_raw.sh",
    "newest": ROOT / "Newest" / "apply_newest.sh",
}

DEFAULT_SEQ_LENS = [4096, 8192, 16384, 32768, 65536, 81920]


@dataclass
class CaseResult:
    version: str
    seq_len: int
    batch: int
    heads: int
    head_dim: int
    returncode: int
    elapsed_s: float
    flex_ms: float | None
    used_causal_template: bool
    stdout_tail: list[str]
    stderr_tail: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raw vs Newest causal Flex Attention benchmark.")
    parser.add_argument(
        "--seq-lens",
        default=",".join(str(x) for x in DEFAULT_SEQ_LENS),
        help="Comma-separated sequence lengths.",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--versions",
        default="raw,newest",
        help="Comma-separated versions: raw,newest.",
    )
    parser.add_argument(
        "--no-restore-newest",
        action="store_true",
        help="Do not restore Newest after the benchmark.",
    )
    parser.add_argument("--output-md", default="docs/causal_raw_newest_perf.md")
    parser.add_argument("--output-json", default="causal_raw_newest_perf.json")
    return parser.parse_args()


def parse_csv_ints(spec: str) -> list[int]:
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def parse_csv_strings(spec: str) -> list[str]:
    return [part.strip() for part in spec.split(",") if part.strip()]


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def find_flex_ms(stdout: str) -> float | None:
    for line in stdout.splitlines():
        clean = strip_ansi(line)
        if "Flex Attention avg:" in clean:
            match = re.search(r"([0-9]+\.[0-9]+)\s*ms", clean)
            if match:
                return float(match.group(1))
    return None


def apply_version(version: str) -> None:
    script = APPLY[version]
    proc = subprocess.run(
        ["bash", str(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"apply {version} failed with code {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


def run_case(args: argparse.Namespace, version: str, seq_len: int) -> CaseResult:
    shape = f"{args.batch},{args.heads},{seq_len},{args.head_dim}"
    cmd = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--shape", shape,
        "--sparse-config", "causal",
        "--target", "flex",
        "--device", args.device,
        "--dtype", args.dtype,
        "--warmup", str(args.warmup),
        "--repeat", str(args.repeat),
        "--no-compare",
    ]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = (
        "/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/lib:"
        "/usr/local/python3.11.14/lib:"
        + env.get("LD_LIBRARY_PATH", "")
    )
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}"
    env["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/torchinductor_causal_{version}_{seq_len}"

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
            version=version,
            seq_len=seq_len,
            batch=args.batch,
            heads=args.heads,
            head_dim=args.head_dim,
            returncode=proc.returncode,
            elapsed_s=elapsed_s,
            flex_ms=find_flex_ms(stdout),
            used_causal_template=(
                "dense causal Flex Triton template" in stdout
                or "causal SDPA fastpath" in stdout
            ),
            stdout_tail=stdout.splitlines()[-12:],
            stderr_tail=stderr.splitlines()[-12:],
        )
    except subprocess.TimeoutExpired as exc:
        elapsed_s = time.perf_counter() - started
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        return CaseResult(
            version=version,
            seq_len=seq_len,
            batch=args.batch,
            heads=args.heads,
            head_dim=args.head_dim,
            returncode=124,
            elapsed_s=elapsed_s,
            flex_ms=find_flex_ms(stdout),
            used_causal_template=False,
            stdout_tail=stdout.splitlines()[-12:],
            stderr_tail=stderr.splitlines()[-12:],
        )


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


def render_markdown(results: list[CaseResult]) -> str:
    by_seq: dict[int, dict[str, CaseResult]] = {}
    for result in results:
        by_seq.setdefault(result.seq_len, {})[result.version] = result

    lines = [
        "# Causal Raw vs Newest Performance",
        "",
        "| Seq Len | Raw | Newest | Speedup | Newest Path | Notes |",
        "|--------:|----:|-------:|--------:|-------------|-------|",
    ]
    for seq_len in sorted(by_seq):
        raw = by_seq[seq_len].get("raw")
        newest = by_seq[seq_len].get("newest")
        raw_ms = f"{raw.flex_ms:.3f}" if raw and raw.flex_ms is not None else status(raw)
        newest_ms = f"{newest.flex_ms:.3f}" if newest and newest.flex_ms is not None else status(newest)
        if raw and newest and raw.flex_ms and newest.flex_ms:
            speedup = f"{raw.flex_ms / newest.flex_ms:.4f}x"
        else:
            speedup = "-"
        path = "causal_template" if newest and newest.used_causal_template else "-"
        notes = []
        if raw and raw.returncode != 0:
            notes.append(f"raw={status(raw)}")
        if newest and newest.returncode != 0:
            notes.append(f"newest={status(newest)}")
        lines.append(
            f"| {seq_len} | {raw_ms} | {newest_ms} | {speedup} | {path} | {', '.join(notes) if notes else '-'} |"
        )
    return "\n".join(lines) + "\n"


def print_case(result: CaseResult) -> None:
    ms = f"{result.flex_ms:.3f} ms" if result.flex_ms is not None else "N/A"
    path = " causal-template" if result.used_causal_template else ""
    print(
        f"  [{result.version}] S={result.seq_len:<6d} {status(result):8s} "
        f"{ms:>10s} ({result.elapsed_s:.0f}s){path}",
        flush=True,
    )
    if result.returncode != 0:
        for line in result.stdout_tail[-4:]:
            print(f"    stdout: {line}")
        for line in result.stderr_tail[-4:]:
            print(f"    stderr: {line}")


def main() -> int:
    args = parse_args()
    seq_lens = parse_csv_ints(args.seq_lens)
    versions = parse_csv_strings(args.versions)
    for version in versions:
        if version not in APPLY:
            raise ValueError(f"unknown version: {version}")

    results: list[CaseResult] = []
    print("=== Causal Raw vs Newest Matrix ===", flush=True)
    print(
        f"versions={versions} seq_lens={seq_lens} "
        f"shape=({args.batch},{args.heads},S,{args.head_dim})",
        flush=True,
    )

    try:
        for version in versions:
            print(f"\n### Applying {version}", flush=True)
            apply_version(version)
            for seq_len in seq_lens:
                result = run_case(args, version, seq_len)
                results.append(result)
                print_case(result)
                time.sleep(1)
    finally:
        if not args.no_restore_newest:
            print("\n### Restoring Newest", flush=True)
            apply_version("newest")

    if args.output_json:
        output_json = (ROOT / args.output_json).resolve()
        output_json.write_text(
            json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nJSON written to {output_json}")

    if args.output_md:
        output_md = (ROOT / args.output_md).resolve()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(results), encoding="utf-8")
        print(f"Markdown written to {output_md}")

    print("\n=== Summary ===")
    print(render_markdown(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
