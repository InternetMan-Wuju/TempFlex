#!/usr/bin/env python3
import argparse
import csv
import heapq
import json
import math
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path


HELPER_PATTERNS = (
    "arange",
    "cast",
    "contiguous",
    "copy",
    "equal",
    "fill",
    "gather",
    "greater",
    "index",
    "less",
    "pad",
    "range",
    "reshape",
    "scatter",
    "select",
    "slice",
    "sort",
    "squeeze",
    "tensormove",
    "transpose",
    "unsqueeze",
    "where",
)
ATTENTION_PATTERNS = (
    "attention",
    "batchmatmul",
    "bmm",
    "flash",
    "matmul",
    "softmax",
)


@dataclass
class Item:
    name: str
    time_us: float
    count: int = 1
    extra: str = ""


@dataclass
class ProfileSummary:
    label: str
    profile_dir: Path
    output_dir: Path | None = None
    op_stat_file: Path | None = None
    op_summary_file: Path | None = None
    task_file: Path | None = None
    api_file: Path | None = None
    trace_file: Path | None = None
    warnings: list[str] = field(default_factory=list)
    scope: str = "full"
    scope_source: str = "full profile"
    scope_window: tuple[float, float] | None = None
    warmup: int | None = None
    repeat: int | None = None

    op_total_us: float = 0.0
    op_count: int = 0
    op_types: list[Item] = field(default_factory=list)
    helper_us: float = 0.0
    attention_us: float = 0.0
    ai_cpu_us: float = 0.0

    task_total_us: float = 0.0
    task_count: int = 0
    task_kernels: list[Item] = field(default_factory=list)
    task_types: dict[str, float] = field(default_factory=dict)

    api_total_us: float = 0.0
    api_count: int = 0
    api_items: list[Item] = field(default_factory=list)

    op_invocations: list[Item] = field(default_factory=list)


def clean_header(name):
    return (name or "").strip().lstrip("\ufeff")


def clean_value(value):
    if value is None:
        return ""
    return str(value).strip().strip("\t")


def to_float(value, default=0.0):
    text = clean_value(value)
    if not text or text.upper() == "N/A":
        return default
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return default


def to_int(value, default=0):
    return int(to_float(value, default))


def ms(us):
    return us / 1000.0


def fmt_ms(us):
    value = ms(us)
    if math.isclose(value, 0.0, abs_tol=1e-9):
        return "0.000"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.3f}"


def fmt_pct(part, whole):
    if whole <= 0:
        return "n/a"
    return f"{part / whole * 100:.1f}%"


def short_path(path, root):
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def is_helper(name):
    lower = name.lower()
    return any(pattern in lower for pattern in HELPER_PATTERNS)


def is_attention(name):
    lower = name.lower()
    return any(pattern in lower for pattern in ATTENTION_PATTERNS)


def latest_file(directory, pattern):
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def find_output_dir(profile_dir):
    direct = profile_dir / "mindstudio_profiler_output"
    if direct.is_dir():
        return direct
    matches = list(profile_dir.glob("*/mindstudio_profiler_output"))
    return matches[0] if matches else None


def iter_csv_rows(path, max_bytes):
    if path is None:
        return
    size = path.stat().st_size
    if max_bytes is not None and size > max_bytes:
        raise ValueError(f"{path} is {size} bytes, above --max-csv-mb")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return
        reader.fieldnames = [clean_header(name) for name in reader.fieldnames]
        for row in reader:
            yield {clean_header(k): clean_value(v) for k, v in row.items()}


def read_csv_rows(path, max_bytes):
    if path is None:
        return []
    return list(iter_csv_rows(path, max_bytes))


def parse_sample_args(profile_dir):
    for sample_path in (
        profile_dir / "device_0" / "sample.json",
        profile_dir / "host" / "sample.json",
    ):
        if not sample_path.exists():
            continue
        try:
            data = json.loads(sample_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        params = data.get("app_parameters", "")
        try:
            tokens = shlex.split(params)
        except ValueError:
            tokens = params.split()
        parsed = {}
        idx = 0
        while idx < len(tokens):
            token = tokens[idx]
            if token.startswith("--"):
                if "=" in token:
                    key, value = token[2:].split("=", 1)
                    parsed[key] = value
                elif idx + 1 < len(tokens) and not tokens[idx + 1].startswith("--"):
                    parsed[token[2:]] = tokens[idx + 1]
                    idx += 1
                else:
                    parsed[token[2:]] = True
            idx += 1
        return parsed
    return {}


def int_arg(parsed_args, name, default):
    value = parsed_args.get(name, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def row_start_us(row, *keys):
    for key in keys:
        if key in row:
            return to_float(row.get(key), default=math.nan)
    return math.nan


def find_trace_window(trace_file, max_bytes):
    if trace_file is None or not trace_file.exists():
        return None
    size = trace_file.stat().st_size
    if max_bytes is not None and size > max_bytes:
        return None
    try:
        events = json.loads(trace_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    starts = []
    ends = []
    for event in events:
        name = str(event.get("name", ""))
        if "profile_repeat_loop" in name and event.get("dur") not in (None, ""):
            start = to_float(event.get("ts"), default=math.nan)
            dur = to_float(event.get("dur"), default=math.nan)
            if not math.isnan(start) and not math.isnan(dur):
                return (start, start + dur)
        if "profile_repeat_start" in name:
            starts.append(to_float(event.get("ts"), default=math.nan))
        if "profile_repeat_end" in name:
            ends.append(to_float(event.get("ts"), default=math.nan))

    starts = [value for value in starts if not math.isnan(value)]
    ends = [value for value in ends if not math.isnan(value)]
    if starts and ends:
        start = min(starts)
        valid_ends = [value for value in ends if value >= start]
        if valid_ends:
            return (start, max(valid_ends))
    return None


def select_rows_for_scope(rows, start_keys, summary, group_key=None):
    if not rows:
        return []
    sorted_rows = sorted(
        rows,
        key=lambda row: row_start_us(row, *start_keys),
    )
    if summary.scope_window is not None:
        start, end = summary.scope_window
        selected = [
            row for row in sorted_rows
            if start <= row_start_us(row, *start_keys) <= end
        ]
        if selected:
            return selected
        summary.warnings.append("MSTX window was found, but no CSV rows landed inside it; falling back to full rows.")
        return sorted_rows

    if summary.scope == "repeat-tail":
        warmup = summary.warmup if summary.warmup is not None else 10
        repeat = summary.repeat if summary.repeat is not None else 10
        total_iters = warmup + repeat
        if total_iters <= 0 or repeat <= 0:
            summary.warnings.append("cannot infer repeat-tail scope because warmup/repeat are invalid")
            return sorted_rows
        take = 0
        if group_key is not None:
            counts = {}
            for row in sorted_rows:
                key = group_key(row)
                counts[key] = counts.get(key, 0) + 1
            per_iter_rows = sum(count // total_iters for count in counts.values())
            take = repeat * per_iter_rows
        if take <= 0 or take > len(sorted_rows):
            take = max(1, math.ceil(len(sorted_rows) * repeat / total_iters))
        return sorted_rows[-take:]

    return sorted_rows


def top_push(heap, item, limit):
    if limit <= 0:
        return
    entry = (item.time_us, len(heap), item)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif item.time_us > heap[0][0]:
        heapq.heapreplace(heap, entry)


def heap_items_desc(heap):
    return [entry[2] for entry in sorted(heap, key=lambda x: x[0], reverse=True)]


def parse_op_statistic(summary, top_n, max_bytes):
    path = summary.op_stat_file
    if path is None:
        summary.warnings.append("missing op_statistic_*.csv")
        return

    top_heap = []
    try:
        for row in iter_csv_rows(path, max_bytes):
            op_type = row.get("OP Type", "UNKNOWN") or "UNKNOWN"
            core_type = row.get("Core Type", "")
            count = to_int(row.get("Count"))
            total_us = to_float(row.get("Total Time(us)"))
            ratio = row.get("Ratio(%)", "")
            summary.op_count += count
            summary.op_total_us += total_us

            name_for_classify = f"{op_type} {core_type}"
            if is_helper(name_for_classify):
                summary.helper_us += total_us
            if is_attention(name_for_classify):
                summary.attention_us += total_us
            if "AI_CPU" in core_type.upper():
                summary.ai_cpu_us += total_us

            item = Item(
                name=op_type,
                time_us=total_us,
                count=count,
                extra=f"{core_type}, avg={fmt_ms(to_float(row.get('Avg Time(us)')))} ms, ratio={ratio}%",
            )
            top_push(top_heap, item, top_n)
    except ValueError as exc:
        summary.warnings.append(str(exc))
        return

    summary.op_types = heap_items_desc(top_heap)


def parse_task_time(summary, top_n, max_bytes):
    path = summary.task_file
    if path is None:
        summary.warnings.append("missing task_time_*.csv")
        return

    try:
        rows = read_csv_rows(path, max_bytes)
    except ValueError as exc:
        summary.warnings.append(str(exc))
        return
    parse_task_rows(summary, rows, top_n)


def parse_task_rows(summary, rows, top_n):
    top_heap = []
    type_totals = {}
    for row in rows:
        kernel = row.get("kernel_name", "") or "N/A"
        kernel_type = row.get("kernel_type", "") or "UNKNOWN"
        duration_us = to_float(row.get("task_time(us)"))
        if duration_us <= 0:
            continue
        summary.task_count += 1
        summary.task_total_us += duration_us
        type_totals[kernel_type] = type_totals.get(kernel_type, 0.0) + duration_us
        top_push(top_heap, Item(kernel, duration_us, 1, kernel_type), top_n)

    summary.task_kernels = heap_items_desc(top_heap)
    summary.task_types = dict(sorted(type_totals.items(), key=lambda item: item[1], reverse=True))


def parse_api_statistic(summary, top_n, max_bytes):
    path = summary.api_file
    if path is None:
        summary.warnings.append("missing api_statistic_*.csv")
        return

    top_heap = []
    try:
        for row in iter_csv_rows(path, max_bytes):
            name = row.get("API Name", "") or "UNKNOWN"
            level = row.get("Level", "")
            total_us = to_float(row.get("Time(us)"))
            count = to_int(row.get("Count"))
            summary.api_total_us += total_us
            summary.api_count += count
            item = Item(
                name=name,
                time_us=total_us,
                count=count,
                extra=f"{level}, avg={fmt_ms(to_float(row.get('Avg(us)')))} ms",
            )
            top_push(top_heap, item, top_n)
    except ValueError as exc:
        summary.warnings.append(str(exc))
        return

    summary.api_items = heap_items_desc(top_heap)


def parse_op_summary(summary, top_n, max_bytes):
    path = summary.op_summary_file
    if path is None:
        summary.warnings.append("missing op_summary_*.csv")
        return

    try:
        rows = read_csv_rows(path, max_bytes)
    except ValueError as exc:
        summary.warnings.append(str(exc))
        return
    parse_op_summary_rows(summary, rows, top_n, aggregate_op_types=False)


def parse_op_summary_rows(summary, rows, top_n, aggregate_op_types):
    top_heap = []
    type_totals = {}
    type_counts = {}
    type_core = {}

    for row in rows:
        op_name = row.get("Op Name", "") or row.get("OP Type", "") or "UNKNOWN"
        op_type = row.get("OP Type", "") or "UNKNOWN"
        task_type = row.get("Task Type", "") or row.get("Core Type", "")
        duration_us = to_float(row.get("Task Duration(us)"))
        if duration_us <= 0:
            continue

        extra = op_type
        cube_util = row.get("cube_utilization(%)", "")
        if cube_util and cube_util.upper() != "N/A":
            extra = f"{extra}, cube={cube_util}%"
        top_push(top_heap, Item(op_name, duration_us, 1, extra), top_n)

        if aggregate_op_types:
            summary.op_count += 1
            summary.op_total_us += duration_us
            type_totals[op_type] = type_totals.get(op_type, 0.0) + duration_us
            type_counts[op_type] = type_counts.get(op_type, 0) + 1
            type_core.setdefault(op_type, task_type)

            name_for_classify = f"{op_type} {task_type}"
            if is_helper(name_for_classify):
                summary.helper_us += duration_us
            if is_attention(name_for_classify):
                summary.attention_us += duration_us
            if "AI_CPU" in task_type.upper():
                summary.ai_cpu_us += duration_us

    summary.op_invocations = heap_items_desc(top_heap)
    if aggregate_op_types:
        items = []
        for op_type, total_us in type_totals.items():
            count = type_counts[op_type]
            avg_us = total_us / count if count else 0.0
            items.append(
                Item(
                    op_type,
                    total_us,
                    count,
                    f"{type_core.get(op_type, '')}, avg={fmt_ms(avg_us)} ms",
                )
            )
        summary.op_types = sorted(items, key=lambda item: item.time_us, reverse=True)[:top_n]


def label_from_profile(profile_dir):
    parent = profile_dir.parent.name.lower()
    if parent in ("flex", "manual"):
        return parent
    name = profile_dir.name.lower()
    if "flex" in name:
        return "flex"
    if "manual" in name:
        return "manual"
    return profile_dir.parent.name or profile_dir.name


def discover_profiles(root, all_profiles=False):
    root = root.resolve()
    if root.name.startswith("PROF_"):
        return [(label_from_profile(root), root)]

    direct_output = find_output_dir(root)
    if direct_output is not None and root.name.startswith("PROF_"):
        return [(label_from_profile(root), root)]

    candidates = []
    for child in sorted(root.iterdir()) if root.is_dir() else []:
        if child.is_dir() and child.name.startswith("PROF_"):
            candidates.append((label_from_profile(child), child))
        elif child.is_dir():
            profs = sorted(child.glob("PROF_*"), key=lambda p: p.stat().st_mtime, reverse=True)
            if all_profiles:
                candidates.extend((child.name, prof) for prof in profs)
            elif profs:
                candidates.append((child.name, profs[0]))

    if candidates:
        return candidates

    nested = sorted(root.glob("*/*/PROF_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if all_profiles:
        return [(label_from_profile(path), path) for path in nested]

    latest_by_label = {}
    for path in nested:
        label = label_from_profile(path)
        if label not in latest_by_label:
            latest_by_label[label] = path
    return list(latest_by_label.items())


def analyze_profile(label, profile_dir, top_n, max_bytes, max_trace_bytes, scope, warmup, repeat):
    summary = ProfileSummary(label=label, profile_dir=profile_dir)
    summary.output_dir = find_output_dir(profile_dir)
    if summary.output_dir is None:
        summary.warnings.append("missing mindstudio_profiler_output")
        return summary

    summary.op_stat_file = latest_file(summary.output_dir, "op_statistic_*.csv")
    summary.op_summary_file = latest_file(summary.output_dir, "op_summary_*.csv")
    summary.task_file = latest_file(summary.output_dir, "task_time_*.csv")
    summary.api_file = latest_file(summary.output_dir, "api_statistic_*.csv")
    summary.trace_file = latest_file(summary.output_dir, "msprof_*.json")

    sample_args = parse_sample_args(profile_dir)
    summary.warmup = warmup if warmup is not None else int_arg(sample_args, "warmup", 10)
    summary.repeat = repeat if repeat is not None else int_arg(sample_args, "repeat", 10)

    requested_scope = scope
    if requested_scope in ("auto", "mstx"):
        summary.scope_window = find_trace_window(summary.trace_file, max_trace_bytes)
        if summary.scope_window is not None:
            summary.scope = "mstx"
            start, end = summary.scope_window
            summary.scope_source = f"MSTX repeat window {start:.3f}..{end:.3f} us"
        elif requested_scope == "mstx":
            summary.scope = "full"
            summary.scope_source = "full profile"
            summary.warnings.append("requested MSTX scope, but no profile_repeat_loop marker was found")
        else:
            summary.scope = "repeat-tail"
            summary.scope_source = (
                f"estimated repeat-tail: last {summary.repeat} steady-state iterations "
                f"from warmup={summary.warmup}, repeat={summary.repeat}"
            )
            summary.warnings.append("no MSTX repeat marker found; using repeat-tail estimate")
    elif requested_scope == "repeat-tail":
        summary.scope = "repeat-tail"
        summary.scope_source = (
            f"estimated repeat-tail: last {summary.repeat} steady-state iterations "
            f"from warmup={summary.warmup}, repeat={summary.repeat}"
        )
    else:
        summary.scope = "full"
        summary.scope_source = "full profile"

    if summary.scope == "full":
        parse_op_statistic(summary, top_n, max_bytes)
        parse_task_time(summary, top_n, max_bytes)
        parse_api_statistic(summary, top_n, max_bytes)
        parse_op_summary(summary, top_n, max_bytes)
        return summary

    try:
        op_rows = read_csv_rows(summary.op_summary_file, max_bytes)
        task_rows = read_csv_rows(summary.task_file, max_bytes)
    except ValueError as exc:
        summary.warnings.append(str(exc))
        op_rows = []
        task_rows = []

    op_rows = select_rows_for_scope(
        op_rows,
        ("Task Start Time(us)",),
        summary,
        group_key=lambda row: row.get("OP Type", "") or row.get("Op Name", ""),
    )
    task_rows = select_rows_for_scope(
        task_rows,
        ("task_start(us)",),
        summary,
        group_key=lambda row: row.get("kernel_name", "") or row.get("kernel_type", ""),
    )
    parse_op_summary_rows(summary, op_rows, top_n, aggregate_op_types=True)
    parse_task_rows(summary, task_rows, top_n)
    parse_api_statistic(summary, top_n, max_bytes)
    summary.warnings.append("host API table is still full-profile; exported api_statistic has no per-call timestamps")
    return summary


def print_table(headers, rows):
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))

    print(" | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row)))


def one_line_top(items):
    if not items:
        return "n/a"
    item = items[0]
    return f"{item.name} {fmt_ms(item.time_us)} ms"


def print_items(title, items, total_us, top_n):
    print(f"\n{title}")
    if not items:
        print("  n/a")
        return
    rows = []
    for idx, item in enumerate(items[:top_n], start=1):
        rows.append(
            [
                str(idx),
                item.name,
                item.count,
                fmt_ms(item.time_us),
                fmt_pct(item.time_us, total_us),
                item.extra,
            ]
        )
    print_table(["#", "name", "count", "total ms", "share", "extra"], rows)


def print_summary(root, summaries, top_n):
    print("# msprof readable summary")
    print(f"root: {root}")
    print("note: summary CSV files are used; trace JSON is only scanned for optional MSTX repeat markers.")

    overview_rows = []
    for summary in summaries:
        overview_rows.append(
            [
                summary.label,
                summary.profile_dir.name,
                summary.scope,
                fmt_ms(summary.op_total_us),
                summary.op_count,
                fmt_ms(summary.task_total_us),
                summary.task_count,
                fmt_ms(summary.helper_us),
                fmt_pct(summary.helper_us, summary.op_total_us),
                fmt_ms(summary.ai_cpu_us),
                one_line_top(summary.op_types),
            ]
        )

    print("\n## overview")
    print_table(
        [
            "target",
            "profile",
            "scope",
            "op total ms",
            "op calls",
            "task total ms",
            "tasks",
            "helper ms",
            "helper %",
            "AI CPU ms",
            "top op type",
        ],
        overview_rows,
    )

    by_label = {summary.label: summary for summary in summaries}
    if "flex" in by_label and "manual" in by_label:
        flex = by_label["flex"]
        manual = by_label["manual"]
        print("\n## flex / manual ratios")
        rows = []
        for name, flex_value, manual_value in (
            ("op total", flex.op_total_us, manual.op_total_us),
            ("task total", flex.task_total_us, manual.task_total_us),
            ("host api total", flex.api_total_us, manual.api_total_us),
            ("helper total", flex.helper_us, manual.helper_us),
            ("AI CPU total", flex.ai_cpu_us, manual.ai_cpu_us),
        ):
            ratio = "n/a" if manual_value <= 0 else f"{flex_value / manual_value:.2f}x"
            rows.append([name, fmt_ms(flex_value), fmt_ms(manual_value), ratio])
        print_table(["metric", "flex ms", "manual ms", "ratio"], rows)

    for summary in summaries:
        print(f"\n## {summary.label}: {summary.profile_dir.name}")
        print(f"scope: {summary.scope_source}")
        if summary.warnings:
            for warning in summary.warnings:
                print(f"warning: {warning}")

        print_items("Top op types by total device time", summary.op_types, summary.op_total_us, top_n)
        print_items("Top individual op invocations", summary.op_invocations, summary.op_total_us, top_n)
        print_items("Top kernels/tasks", summary.task_kernels, summary.task_total_us, top_n)
        print_items("Top host APIs", summary.api_items, summary.api_total_us, top_n)

        if summary.task_types:
            rows = [
                [task_type, fmt_ms(total), fmt_pct(total, summary.task_total_us)]
                for task_type, total in list(summary.task_types.items())[:top_n]
            ]
            print("\nTask type breakdown")
            print_table(["task type", "total ms", "share"], rows)

    print_hints(summaries)


def print_hints(summaries):
    by_label = {summary.label: summary for summary in summaries}
    if "flex" not in by_label or "manual" not in by_label:
        return

    flex = by_label["flex"]
    manual = by_label["manual"]
    print("\n## quick read")

    if manual.op_total_us > 0:
        ratio = flex.op_total_us / manual.op_total_us
        if ratio > 1.10:
            print(f"- Flex device op total is {ratio:.2f}x manual: the slowdown is visible on device, not just Python timing.")
        elif ratio < 0.90:
            print(f"- Flex device op total is {ratio:.2f}x manual: device kernels look faster than manual.")
        else:
            print(f"- Flex/manual device op totals are close ({ratio:.2f}x); check host API or synchronization overhead.")

    if flex.ai_cpu_us > 0:
        print(f"- Flex has {fmt_ms(flex.ai_cpu_us)} ms on AI CPU. Large AI CPU time often means fallback/helper work.")
    if flex.helper_us > manual.helper_us * 1.2 and flex.helper_us > 0:
        print(
            f"- Flex helper-like ops are higher ({fmt_ms(flex.helper_us)} ms vs "
            f"{fmt_ms(manual.helper_us)} ms). Look at Cast/Select/Index/Arange/Transpose rows."
        )

    flex_top_names = " ".join(item.name.lower() for item in flex.op_types[:5])
    if "attention" not in flex_top_names and "flash" not in flex_top_names:
        print("- Flex top op types do not obviously look like a fused attention kernel; it may be lowering into helper ops.")

    manual_top_names = " ".join(item.name.lower() for item in manual.op_types[:5])
    if "batchmatmul" in manual_top_names and "softmax" in manual_top_names:
        print("- Manual is mostly BatchMatMul/Softmax, which are mature dense kernels; it can beat an unfused/fallback Flex path.")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize msprof_out into human-readable comparison tables.")
    parser.add_argument("root", nargs="?", default="msprof_out/flex_vs_manual", help="msprof output root or a PROF_* dir")
    parser.add_argument("--top", type=int, default=10, help="number of top rows to print")
    parser.add_argument("--all", action="store_true", help="summarize all PROF_* dirs instead of latest per label")
    parser.add_argument(
        "--scope",
        choices=["auto", "full", "mstx", "repeat-tail"],
        default="auto",
        help="auto uses MSTX repeat markers when present, otherwise estimates the timed repeat tail.",
    )
    parser.add_argument("--warmup", type=int, default=None, help="override warmup count used by repeat-tail")
    parser.add_argument("--repeat", type=int, default=None, help="override repeat count used by repeat-tail")
    parser.add_argument(
        "--max-csv-mb",
        type=float,
        default=200.0,
        help="skip a CSV if it is larger than this many MiB",
    )
    parser.add_argument(
        "--max-json-mb",
        type=float,
        default=200.0,
        help="skip MSTX trace scan if JSON is larger than this many MiB",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"not found: {root}")

    profiles = discover_profiles(root, all_profiles=args.all)
    if not profiles:
        raise SystemExit(f"no PROF_* directories found under {root}")

    max_bytes = None if args.max_csv_mb <= 0 else int(args.max_csv_mb * 1024 * 1024)
    max_trace_bytes = None if args.max_json_mb <= 0 else int(args.max_json_mb * 1024 * 1024)
    summaries = [
        analyze_profile(
            label,
            profile_dir,
            top_n=args.top,
            max_bytes=max_bytes,
            max_trace_bytes=max_trace_bytes,
            scope=args.scope,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        for label, profile_dir in profiles
    ]
    summaries.sort(key=lambda summary: (summary.label, summary.profile_dir.name))
    print_summary(root, summaries, top_n=args.top)


if __name__ == "__main__":
    main()
