#!/usr/bin/env python3
"""Render SVG bar charts from raw_newest_reorder_manual_matrix.json."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent
DEFAULT_JSON = ROOT / "raw_newest_reorder_manual_matrix.json"
DEFAULT_MANUAL_JSON = ROOT / "manual_reference_matrix.json"
DEFAULT_FAIR_16K_JSON = ROOT / "fair_reorder_16k.json"
DEFAULT_OUT = ROOT / "docs" / "reports"

VARIANTS = [
    ("raw_flex", "Raw", "#4C78A8"),
    ("newest_flex", "Newest (without reorder)", "#F58518"),
    ("newest_reorder", "Newest (with reorder)", "#54A24B"),
    ("newest_manual", "Manual", "#B279A2"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SVG charts for the performance matrix.")
    parser.add_argument("--input-json", default=str(DEFAULT_JSON))
    parser.add_argument(
        "--manual-json",
        default=str(DEFAULT_MANUAL_JSON),
        help="Optional manual-only matrix JSON used to override Newest manual cells.",
    )
    parser.add_argument(
        "--fair-16k-json",
        default=str(DEFAULT_FAIR_16K_JSON),
        help="Optional fair identity/reorder JSON used to override 16K Newest without/with reorder bars.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--seq-y-max",
        default="4096:3,16384:24",
        help="Optional comma-separated seq_len:y_max overrides, e.g. 16384:16,4096:2.",
    )
    return parser.parse_args()


def metric(result: dict) -> float | None:
    return result.get("reorder_ms") or result.get("flex_ms") or result.get("manual_ms")


def status(result: dict | None) -> str:
    if result is None:
        return "-"
    if result.get("skipped"):
        return "SKIP"
    rc = result.get("returncode")
    if rc == 0:
        return "OK"
    if rc == 124:
        return "TIMEOUT"
    if rc in (139, -11):
        return "CRASH"
    return f"ERR({rc})"


def nice_top(value: float) -> float:
    if value <= 0:
        return 1.0
    raw = value * 1.12
    power = 10 ** math.floor(math.log10(raw))
    scaled = raw / power
    if scaled <= 1.5:
        step = 1.5
    elif scaled <= 2:
        step = 2
    elif scaled <= 5:
        step = 5
    else:
        step = 10
    return step * power


def svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "middle", extra: str = "") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-family="DejaVu Sans, Arial, sans-serif" text-anchor="{anchor}" {extra}>'
        f"{escape(text)}</text>"
    )


def parse_seq_y_max(spec: str) -> dict[int, float]:
    result: dict[int, float] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        seq, value = item.split(":", 1)
        result[int(seq.strip())] = float(value.strip())
    return result


def render_seq_chart(
    seq_len: int,
    configs: list[str],
    rows: dict[tuple[int, str, str], dict],
    y_max_overrides: dict[int, float],
) -> str:
    variants = VARIANTS
    title = f"Raw / Newest Without Reorder / Newest With Reorder / Manual, S={seq_len}"
    if seq_len == 16384:
        variants = [
            ("raw_flex", "Raw", "#4C78A8"),
            ("newest_flex", "Newest (without reorder)", "#F58518"),
            ("newest_reorder", "Newest (with reorder)", "#54A24B"),
            ("newest_manual", "Manual", "#B279A2"),
        ]
        title = "Raw / Newest Without Reorder / Newest With Reorder / Manual, S=16384"

    width = 1480
    height = 720
    margin_left = 82
    margin_right = 30
    margin_top = 72
    margin_bottom = 190
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    group_w = plot_w / len(configs)
    bar_w = min(18, group_w / 6.2)
    gap = bar_w * 0.25

    values = []
    for config in configs:
        for variant, _, _ in variants:
            result = rows.get((seq_len, config, variant))
            value = metric(result) if result else None
            if value is not None and status(result) == "OK":
                values.append(float(value))
    y_top = y_max_overrides.get(seq_len, nice_top(max(values) if values else 1.0))

    def y(value: float) -> float:
        return margin_top + plot_h - (value / y_top) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffaf2"/>',
        svg_text(width / 2, 34, title, 22, "middle", 'font-weight="700" fill="#1f2a33"'),
        svg_text(width / 2, 58, f"Y axis: runtime in ms, max={y_top:g}. Missing bars are SKIP/CRASH/TIMEOUT.", 12, "middle", 'fill="#53606b"'),
    ]

    # Grid and y-axis labels.
    tick_count = 5
    for i in range(tick_count + 1):
        value = y_top * i / tick_count
        yy = y(value)
        parts.append(f'<line x1="{margin_left}" y1="{yy:.1f}" x2="{width - margin_right}" y2="{yy:.1f}" stroke="#e6ded2" stroke-width="1"/>')
        parts.append(svg_text(margin_left - 10, yy + 4, f"{value:.1f}", 11, "end", 'fill="#53606b"'))
    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#2f3942" stroke-width="1.2"/>')
    parts.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - margin_right}" y2="{margin_top + plot_h}" stroke="#2f3942" stroke-width="1.2"/>')
    parts.append(svg_text(22, margin_top + plot_h / 2, "ms", 13, "middle", 'fill="#2f3942" transform="rotate(-90 22 %.1f)"' % (margin_top + plot_h / 2)))

    # Legend.
    legend_x = margin_left
    legend_y = height - 28
    for i, (_, label, color) in enumerate(variants):
        x0 = legend_x + i * 245
        parts.append(f'<rect x="{x0}" y="{legend_y - 12}" width="16" height="16" rx="3" fill="{color}"/>')
        parts.append(svg_text(x0 + 24, legend_y + 1, label, 13, "start", 'fill="#1f2a33"'))

    # Bars.
    for ci, config in enumerate(configs):
        center = margin_left + group_w * (ci + 0.5)
        total_bars_w = len(variants) * bar_w + (len(variants) - 1) * gap
        start_x = center - total_bars_w / 2
        label = config.replace("_", " ")
        parts.append(svg_text(center, margin_top + plot_h + 24, label, 11, "end", f'fill="#28323b" transform="rotate(-38 {center:.1f} {margin_top + plot_h + 24:.1f})"'))
        for vi, (variant, short, color) in enumerate(variants):
            result = rows.get((seq_len, config, variant))
            value = metric(result) if result else None
            x0 = start_x + vi * (bar_w + gap)
            st = status(result)
            if value is not None and st == "OK":
                yy = y(float(value))
                bar_h = margin_top + plot_h - yy
                parts.append(f'<rect x="{x0:.1f}" y="{yy:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" rx="2"/>')
                if bar_h > 28:
                    parts.append(svg_text(x0 + bar_w / 2, yy + 14, f"{value:.2f}", 9, "middle", 'fill="white" font-weight="700"'))
                else:
                    parts.append(svg_text(x0 + bar_w / 2, yy - 4, f"{value:.2f}", 8, "middle", f'fill="{color}" font-weight="700"'))
            elif st != "-":
                parts.append(svg_text(x0 + bar_w / 2, margin_top + plot_h - 6 - vi * 12, st, 8, "middle", f'fill="{color}" font-weight="700"'))

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> int:
    args = parse_args()
    data = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    manual_path = Path(args.manual_json)
    manual_data = json.loads(manual_path.read_text(encoding="utf-8")) if manual_path.exists() else []
    fair_16k_path = Path(args.fair_16k_json)
    fair_16k_data = json.loads(fair_16k_path.read_text(encoding="utf-8")) if fair_16k_path.exists() else []
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_max_overrides = parse_seq_y_max(args.seq_y_max)

    seq_lens = sorted({int(item["seq_len"]) for item in data})
    configs = sorted({item["sparse_config"] for item in data})
    rows = {
        (int(item["seq_len"]), item["sparse_config"], item["variant"]): item
        for item in data
    }
    for item in manual_data:
        rows[(int(item["seq_len"]), item["sparse_config"], "newest_manual")] = {
            "seq_len": int(item["seq_len"]),
            "sparse_config": item["sparse_config"],
            "variant": "newest_manual",
            "returncode": item["returncode"],
            "manual_ms": item.get("manual_ms"),
            "flex_ms": None,
            "reorder_ms": None,
            "skipped": False,
        }
    for item in fair_16k_data:
        if int(item["seq_len"]) != 16384:
            continue
        config = item["sparse_config"]
        if item.get("baseline_status") == "OK" and item.get("baseline_ms") is not None:
            rows[(16384, config, "newest_flex")] = {
                "seq_len": 16384,
                "sparse_config": config,
                "variant": "newest_flex",
                "returncode": 0,
                "manual_ms": None,
                "flex_ms": item["baseline_ms"],
                "reorder_ms": None,
                "skipped": False,
            }
        if item.get("reorder_status") == "OK" and item.get("reorder_ms") is not None:
            rows[(16384, config, "newest_reorder")] = {
                "seq_len": 16384,
                "sparse_config": config,
                "variant": "newest_reorder",
                "returncode": 0,
                "manual_ms": None,
                "flex_ms": None,
                "reorder_ms": item["reorder_ms"],
                "skipped": False,
            }
    for seq_len in seq_lens:
        svg = render_seq_chart(seq_len, configs, rows, y_max_overrides)
        path = out_dir / f"raw_newest_reorder_manual_s{seq_len}.svg"
        path.write_text(svg, encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
