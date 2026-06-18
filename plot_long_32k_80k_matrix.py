#!/usr/bin/env python3
"""Render long-sequence SVG charts with fair identity/reorder A/B overlays."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent
DEFAULT_MATRIX_MD = ROOT / "docs" / "long_32k_80k_matrix.md"
DEFAULT_FAIR_JSON = ROOT / "fair_reorder_32k_80k.json"
DEFAULT_OUT = ROOT / "docs" / "reports"

VARIANTS = [
    ("raw", "Raw", "#4C78A8"),
    ("newest", "Newest (without reorder)", "#F58518"),
    ("reorder", "Newest (with reorder)", "#54A24B"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 32K/80K long-sequence SVG charts.")
    parser.add_argument("--matrix-md", default=str(DEFAULT_MATRIX_MD))
    parser.add_argument("--fair-json", default=str(DEFAULT_FAIR_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--seq-y-max",
        default="32768:72,81920:300",
        help="Optional comma-separated seq_len:y_max overrides.",
    )
    return parser.parse_args()


def parse_seq_y_max(spec: str) -> dict[int, float]:
    result: dict[int, float] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        seq, value = item.split(":", 1)
        result[int(seq.strip())] = float(value.strip())
    return result


def parse_cell(cell: str) -> dict[str, object]:
    text = cell.strip()
    try:
        return {"status": "OK", "value": float(text)}
    except ValueError:
        return {"status": text, "value": None}


def parse_matrix_md(path: Path) -> tuple[list[int], list[str], dict[tuple[int, str, str], dict[str, object]]]:
    seqs: list[int] = []
    configs: list[str] = []
    rows: dict[tuple[int, str, str], dict[str, object]] = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| "):
            continue
        if "Seq Len" in line or "---" in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 6:
            continue
        seq_len = int(cells[0])
        config = cells[1].strip("`")
        if seq_len not in seqs:
            seqs.append(seq_len)
        if config not in configs:
            configs.append(config)
        for variant, cell in zip(("raw", "newest", "reorder"), cells[2:5]):
            rows[(seq_len, config, variant)] = parse_cell(cell)
    return seqs, configs, rows


def apply_fair_overlay(rows: dict[tuple[int, str, str], dict[str, object]], fair_json: Path) -> None:
    if not fair_json.exists():
        return
    for item in json.loads(fair_json.read_text(encoding="utf-8")):
        seq_len = int(item["seq_len"])
        config = item["sparse_config"]
        baseline_status = str(item.get("baseline_status") or "-")
        reorder_status = str(item.get("reorder_status") or "-")
        rows[(seq_len, config, "newest")] = {
            "status": baseline_status,
            "value": item.get("baseline_ms") if baseline_status == "OK" else None,
        }
        rows[(seq_len, config, "reorder")] = {
            "status": reorder_status,
            "value": item.get("reorder_ms") if reorder_status == "OK" else None,
        }


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


def display_status(status: str) -> str:
    if status in ("", "-", "None"):
        return "-"
    if status.startswith("ERR") or status in ("SKIP", "CRASH", "TIMEOUT"):
        return status
    if re.fullmatch(r"-?\d+", status):
        return f"ERR({status})"
    return status


def render_seq_chart(
    seq_len: int,
    configs: list[str],
    rows: dict[tuple[int, str, str], dict[str, object]],
    y_max_overrides: dict[int, float],
) -> str:
    width = 1480
    height = 760
    margin_left = 88
    margin_right = 30
    margin_top = 76
    margin_bottom = 210
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    group_w = plot_w / len(configs)
    bar_w = min(18, group_w / 6.2)
    gap = bar_w * 0.25

    values: list[float] = []
    for config in configs:
        for variant, _, _ in VARIANTS:
            result = rows.get((seq_len, config, variant), {})
            if result.get("status") == "OK" and result.get("value") is not None:
                values.append(float(result["value"]))
    y_top = y_max_overrides.get(seq_len, nice_top(max(values) if values else 1.0))

    def y(value: float) -> float:
        return margin_top + plot_h - (value / y_top) * plot_h

    title = f"Raw / Newest Without Reorder / Newest With Reorder, S={seq_len}"
    subtitle = (
        f"Y axis: runtime in ms, max={y_top:g}. "
        "Non-causal orange/green use fair identity/reorder external A/B."
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffaf2"/>',
        svg_text(width / 2, 34, title, 22, "middle", 'font-weight="700" fill="#1f2a33"'),
        svg_text(width / 2, 58, subtitle, 12, "middle", 'fill="#53606b"'),
    ]

    tick_count = 5
    for i in range(tick_count + 1):
        value = y_top * i / tick_count
        yy = y(value)
        parts.append(
            f'<line x1="{margin_left}" y1="{yy:.1f}" x2="{width - margin_right}" y2="{yy:.1f}" '
            'stroke="#e6ded2" stroke-width="1"/>'
        )
        parts.append(svg_text(margin_left - 10, yy + 4, f"{value:.1f}", 11, "end", 'fill="#53606b"'))
    parts.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" '
        'stroke="#2f3942" stroke-width="1.2"/>'
    )
    parts.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - margin_right}" '
        f'y2="{margin_top + plot_h}" stroke="#2f3942" stroke-width="1.2"/>'
    )
    parts.append(
        svg_text(
            24,
            margin_top + plot_h / 2,
            "ms",
            13,
            "middle",
            'fill="#2f3942" transform="rotate(-90 24 %.1f)"' % (margin_top + plot_h / 2),
        )
    )

    legend_y = height - 28
    for i, (_, label, color) in enumerate(VARIANTS):
        x0 = margin_left + i * 245
        parts.append(f'<rect x="{x0}" y="{legend_y - 12}" width="16" height="16" rx="3" fill="{color}"/>')
        parts.append(svg_text(x0 + 24, legend_y + 1, label, 13, "start", 'fill="#1f2a33"'))

    for ci, config in enumerate(configs):
        center = margin_left + group_w * (ci + 0.5)
        total_bars_w = len(VARIANTS) * bar_w + (len(VARIANTS) - 1) * gap
        start_x = center - total_bars_w / 2
        label = config.replace("_", " ")
        parts.append(
            svg_text(
                center,
                margin_top + plot_h + 24,
                label,
                11,
                "end",
                f'fill="#28323b" transform="rotate(-38 {center:.1f} {margin_top + plot_h + 24:.1f})"',
            )
        )
        for vi, (variant, _, color) in enumerate(VARIANTS):
            result = rows.get((seq_len, config, variant), {"status": "-", "value": None})
            status = str(result.get("status") or "-")
            value = result.get("value")
            x0 = start_x + vi * (bar_w + gap)
            if status == "OK" and value is not None:
                number = float(value)
                yy = y(number)
                bar_h = margin_top + plot_h - yy
                parts.append(
                    f'<rect x="{x0:.1f}" y="{yy:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" '
                    f'fill="{color}" rx="2"/>'
                )
                value_label = f"{number:.1f}" if number >= 10 else f"{number:.2f}"
                if bar_h > 28:
                    parts.append(
                        svg_text(
                            x0 + bar_w / 2,
                            yy + 14,
                            value_label,
                            8,
                            "middle",
                            'fill="white" font-weight="700"',
                        )
                    )
                else:
                    parts.append(
                        svg_text(
                            x0 + bar_w / 2,
                            yy - 4,
                            value_label,
                            8,
                            "middle",
                            f'fill="{color}" font-weight="700"',
                        )
                    )
            else:
                label_y = margin_top + plot_h - 6 - vi * 12
                parts.append(
                    svg_text(
                        x0 + bar_w / 2,
                        label_y,
                        display_status(status),
                        8,
                        "middle",
                        f'fill="{color}" font-weight="700"',
                    )
                )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> int:
    args = parse_args()
    matrix_md = Path(args.matrix_md)
    fair_json = Path(args.fair_json)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_max_overrides = parse_seq_y_max(args.seq_y_max)

    seqs, configs, rows = parse_matrix_md(matrix_md)
    apply_fair_overlay(rows, fair_json)
    for seq_len in seqs:
        svg = render_seq_chart(seq_len, configs, rows, y_max_overrides)
        path = out_dir / f"long_32k_80k_s{seq_len}.svg"
        path.write_text(svg, encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
