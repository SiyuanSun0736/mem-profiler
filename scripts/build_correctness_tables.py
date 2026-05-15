#!/usr/bin/env python3
"""Build citation-ready correctness tables from profiler and reference outputs."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys
from dataclasses import dataclass, field


EventTotals = dict[str, float]


PERF_EVENT_ALIASES = {
    "cache-misses": "cache-misses",
    "cache_misses": "cache-misses",
    "cache misses": "cache-misses",
    "cycles": "cycles",
    "instructions": "instructions",
    "dtlb-load-misses": "dtlb-load-misses",
    "dtlb_load_misses": "dtlb-load-misses",
    "itlb-load-misses": "itlb-load-misses",
    "itlb_load_misses": "itlb-load-misses",
    "page-faults": "page-faults",
    "page_faults": "page-faults",
    "minor-faults": "minor-faults",
    "minor_faults": "minor-faults",
    "major-faults": "major-faults",
    "major_faults": "major-faults",
}


MICROBENCH_EXPECTATIONS = {
    "high_llc_miss": "llc_load_misses",
    "high_dtlb_miss": "dtlb_misses",
    "high_page_fault": "minor_faults",
}


@dataclass
class TableArtifact:
    title: str
    stem: str
    columns: list[str]
    rows: list[dict[str, object]]
    notes: list[str] = field(default_factory=list)


def _parse_number(text: str) -> float | None:
    cleaned = text.strip()
    if not cleaned or cleaned.startswith("<"):
        return None
    cleaned = cleaned.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _canonical_perf_event(name: str) -> str:
    key = name.strip().lower().replace("_", "-")
    key = re.sub(r"\s+", "-", key)
    return PERF_EVENT_ALIASES.get(key, key)


def _read_jsonl(path: pathlib.Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(json.loads(stripped))
    return rows


def _sum_window_metrics(path: pathlib.Path) -> EventTotals:
    totals: EventTotals = {}
    for row in _read_jsonl(path):
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0.0) + float(value)
    return totals


def _parse_perf_stat(path: pathlib.Path) -> EventTotals:
    totals: EventTotals = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("---"):
            continue
        if "Performance counter stats" in stripped:
            continue

        value: float | None = None
        event_name = ""

        if "," in stripped:
            parts = [part.strip() for part in stripped.split(",")]
            if len(parts) >= 3:
                value = _parse_number(parts[0])
                event_name = parts[2]
        else:
            tokens = stripped.split()
            if len(tokens) >= 2:
                value = _parse_number(tokens[0])
                event_name = tokens[1]

        if value is None or not event_name:
            continue

        canonical = _canonical_perf_event(event_name)
        totals[canonical] = totals.get(canonical, 0.0) + value
    return totals


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _safe_rel_diff_pct(lhs: float, rhs: float) -> float | None:
    if rhs == 0:
        return None
    return abs(lhs - rhs) / abs(rhs) * 100.0


def _normalize_symbol(symbol: str) -> str:
    normalized = symbol.strip()
    normalized = re.sub(r"\+0x[0-9a-fA-F]+$", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _parse_perf_report(path: pathlib.Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    primary = re.compile(
        r"^\s*(?P<overhead>\d+(?:\.\d+)?)%\s+(?:(?P<comm>\S+)\s+)?"
        r"(?P<dso>\S+)\s+\[[^\]]+\]\s+(?P<symbol>.+?)\s*$"
    )
    fallback = re.compile(
        r"^\s*(?P<overhead>\d+(?:\.\d+)?)%\s+(?P<symbol>[^\[]+?)\s+\[(?P<dso>[^\]]+)\]\s*$"
    )

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        match = primary.match(line)
        if match is None:
            match = fallback.match(line)
        if match is None:
            continue

        overhead = _parse_number(match.group("overhead"))
        symbol = _normalize_symbol(match.group("symbol"))
        dso = match.groupdict().get("dso", "") or ""
        if overhead is None or not symbol:
            continue

        rows.append(
            {
                "symbol": symbol,
                "overhead_pct": overhead,
                "dso": dso.strip(),
            }
        )

    rows.sort(key=lambda row: float(row["overhead_pct"]), reverse=True)
    return rows


def _parse_proc_stat(path: pathlib.Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8").strip()
    close_idx = text.rfind(")")
    if close_idx < 0:
        raise ValueError(f"invalid /proc stat format: {path}")
    fields = text[close_idx + 2 :].split()
    if len(fields) <= 9:
        raise ValueError(f"short /proc stat format: {path}")
    return {
        "minor_faults": int(fields[7]),
        "major_faults": int(fields[9]),
    }


def _load_csv_rows(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _format_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _write_csv(path: pathlib.Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format_cell(row.get(column)) for column in columns})


def _render_markdown_table(columns: list[str], rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(_format_cell(row.get(column)) for column in columns) + " |"
        )
    return lines


def _write_markdown(path: pathlib.Path, artifact: TableArtifact) -> None:
    lines = [f"# {artifact.title}", ""]
    lines.extend(_render_markdown_table(artifact.columns, artifact.rows))
    if artifact.notes:
        lines.append("")
        lines.append("Notes:")
        for note in artifact.notes:
            lines.append(f"- {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_pmu_table(window_metrics: pathlib.Path, perf_stat: pathlib.Path) -> TableArtifact:
    totals = _sum_window_metrics(window_metrics)
    perf_totals = _parse_perf_stat(perf_stat)

    mappings = [
        ("cycles", "cycles", totals.get("cycles", 0.0), "cycles"),
        ("instructions", "instructions", totals.get("instructions", 0.0), "instructions"),
        (
            "cache-misses",
            "llc_load_misses + llc_store_misses",
            totals.get("llc_load_misses", 0.0) + totals.get("llc_store_misses", 0.0),
            "cache-misses",
        ),
        (
            "dTLB-load-misses",
            "dtlb_load_misses",
            totals.get("dtlb_load_misses", totals.get("dtlb_misses", 0.0)),
            "dtlb-load-misses",
        ),
        (
            "iTLB-load-misses",
            "itlb_load_misses",
            totals.get("itlb_load_misses", 0.0),
            "itlb-load-misses",
        ),
        (
            "page-faults",
            "minor_faults + major_faults",
            totals.get("minor_faults", 0.0) + totals.get("major_faults", 0.0),
            "page-faults",
        ),
        ("minor-faults", "minor_faults", totals.get("minor_faults", 0.0), "minor-faults"),
        ("major-faults", "major_faults", totals.get("major_faults", 0.0), "major-faults"),
    ]

    rows: list[dict[str, object]] = []
    for perf_metric, ebpf_metric, ebpf_total, perf_key in mappings:
        if perf_key not in perf_totals:
            continue
        perf_total = perf_totals[perf_key]
        rows.append(
            {
                "perf_metric": perf_metric,
                "ebpf_metric": ebpf_metric,
                "ebpf_total": round(ebpf_total, 4),
                "perf_total": round(perf_total, 4),
                "ratio_ebpf_perf": _safe_ratio(ebpf_total, perf_total),
                "rel_diff_pct": _safe_rel_diff_pct(ebpf_total, perf_total),
            }
        )

    return TableArtifact(
        title="PMU vs perf stat",
        stem="pmu_vs_perf_stat",
        columns=[
            "perf_metric",
            "ebpf_metric",
            "ebpf_total",
            "perf_total",
            "ratio_ebpf_perf",
            "rel_diff_pct",
        ],
        rows=rows,
        notes=[
            "cache-misses is compared against llc_load_misses + llc_store_misses and should be read as a coarse proxy match.",
            "dTLB and iTLB rows are emitted only when the perf stat file contains the corresponding events.",
        ],
    )


def _build_hotspot_summary(function_hotspot: pathlib.Path, perf_report: pathlib.Path, top_k: int) -> list[TableArtifact]:
    ebpf_rows = _load_csv_rows(function_hotspot)
    ebpf_rows.sort(key=lambda row: float(row.get("count", "0") or 0.0), reverse=True)
    perf_rows = _parse_perf_report(perf_report)

    perf_by_symbol = {
        _normalize_symbol(str(row["symbol"])): {**row, "rank_perf": idx + 1}
        for idx, row in enumerate(perf_rows)
    }

    detail_rows: list[dict[str, object]] = []
    for idx, row in enumerate(ebpf_rows[:top_k]):
        symbol = _normalize_symbol(row.get("func", ""))
        perf_row = perf_by_symbol.get(symbol)
        detail_rows.append(
            {
                "rank_ebpf": idx + 1,
                "symbol_ebpf": symbol,
                "count_ebpf": _parse_number(row.get("count", "")) or 0.0,
                "fraction_ebpf": _parse_number(row.get("fraction", "")) or 0.0,
                "rank_perf": perf_row.get("rank_perf") if perf_row else None,
                "overhead_perf_pct": perf_row.get("overhead_pct") if perf_row else None,
                "symbol_match": perf_row is not None,
                "source_file": row.get("source_file", ""),
                "source_line": row.get("source_line", ""),
            }
        )

    summary_rows: list[dict[str, object]] = []
    ebpf_symbols = [_normalize_symbol(row.get("func", "")) for row in ebpf_rows]
    perf_symbols = [_normalize_symbol(str(row["symbol"])) for row in perf_rows]
    max_k = min(top_k, len(ebpf_symbols), len(perf_symbols))
    for k in (1, 3, 5, 10):
        if k > max_k:
            continue
        ebpf_top = set(ebpf_symbols[:k])
        perf_top = set(perf_symbols[:k])
        overlap = len(ebpf_top & perf_top)
        summary_rows.append(
            {
                "top_k": k,
                "overlap_count": overlap,
                "overlap_ratio_pct": _safe_ratio(overlap, k) * 100.0 if k else None,
                "top1_match": ebpf_symbols[:1] == perf_symbols[:1],
            }
        )

    return [
        TableArtifact(
            title="Function hotspot vs perf report summary",
            stem="function_hotspot_vs_perf_report_summary",
            columns=["top_k", "overlap_count", "overlap_ratio_pct", "top1_match"],
            rows=summary_rows,
            notes=[
                "Overlap is computed on normalized symbols within the same Top-K cutoff.",
                "Use the detail table to cite exact function pairs and ranks.",
            ],
        ),
        TableArtifact(
            title="Function hotspot vs perf report detail",
            stem="function_hotspot_vs_perf_report_detail",
            columns=[
                "rank_ebpf",
                "symbol_ebpf",
                "count_ebpf",
                "fraction_ebpf",
                "rank_perf",
                "overhead_perf_pct",
                "symbol_match",
                "source_file",
                "source_line",
            ],
            rows=detail_rows,
            notes=[
                "rank_perf and overhead_perf_pct stay empty when the eBPF hotspot does not appear in the perf report Top-K list.",
            ],
        ),
    ]


def _build_fault_vs_proc_table(
    window_metrics: pathlib.Path,
    proc_before: pathlib.Path,
    proc_after: pathlib.Path,
) -> TableArtifact:
    totals = _sum_window_metrics(window_metrics)
    before = _parse_proc_stat(proc_before)
    after = _parse_proc_stat(proc_after)

    rows: list[dict[str, object]] = []
    for metric in ("minor_faults", "major_faults"):
        ebpf_total = totals.get(metric, 0.0)
        proc_delta = after[metric] - before[metric]
        rows.append(
            {
                "metric": metric,
                "ebpf_total": round(ebpf_total, 4),
                "proc_delta": proc_delta,
                "ratio_ebpf_proc": _safe_ratio(ebpf_total, proc_delta),
                "rel_diff_pct": _safe_rel_diff_pct(ebpf_total, proc_delta),
            }
        )

    rows.append(
        {
            "metric": "total_faults",
            "ebpf_total": round(totals.get("minor_faults", 0.0) + totals.get("major_faults", 0.0), 4),
            "proc_delta": (after["minor_faults"] - before["minor_faults"]) + (after["major_faults"] - before["major_faults"]),
            "ratio_ebpf_proc": _safe_ratio(
                totals.get("minor_faults", 0.0) + totals.get("major_faults", 0.0),
                (after["minor_faults"] - before["minor_faults"]) + (after["major_faults"] - before["major_faults"]),
            ),
            "rel_diff_pct": _safe_rel_diff_pct(
                totals.get("minor_faults", 0.0) + totals.get("major_faults", 0.0),
                (after["minor_faults"] - before["minor_faults"]) + (after["major_faults"] - before["major_faults"]),
            ),
        }
    )

    return TableArtifact(
        title="Fault count vs /proc/<pid>/stat",
        stem="fault_vs_proc_stat",
        columns=["metric", "ebpf_total", "proc_delta", "ratio_ebpf_proc", "rel_diff_pct"],
        rows=rows,
        notes=[
            "Use snapshots taken immediately before and after the same collection window.",
            "Minor and major faults correspond to /proc/<pid>/stat minflt and majflt deltas.",
        ],
    )


def _build_microbench_table(root: pathlib.Path) -> TableArtifact:
    baseline_dir = root / "baseline_sequential"
    baseline_totals = _sum_window_metrics(baseline_dir / "window_metrics.jsonl")

    rows: list[dict[str, object]] = []
    for scenario, metric in MICROBENCH_EXPECTATIONS.items():
        scenario_dir = root / scenario
        scenario_totals = _sum_window_metrics(scenario_dir / "window_metrics.jsonl")
        baseline_value = baseline_totals.get(metric, 0.0)
        scenario_value = scenario_totals.get(metric, 0.0)
        rows.append(
            {
                "scenario": scenario,
                "expected_metric": metric,
                "baseline_total": round(baseline_value, 4),
                "scenario_total": round(scenario_value, 4),
                "ratio_vs_baseline": _safe_ratio(scenario_value, baseline_value),
                "direction_ok": scenario_value > baseline_value,
            }
        )

    return TableArtifact(
        title="Micro benchmark direction check",
        stem="microbench_direction_check",
        columns=[
            "scenario",
            "expected_metric",
            "baseline_total",
            "scenario_total",
            "ratio_vs_baseline",
            "direction_ok",
        ],
        rows=rows,
        notes=[
            "This table is directional: scenario_total should exceed baseline_total for the expected metric.",
            "Use the /proc table when you need exact page-fault count agreement instead of directional validation.",
        ],
    )


def _write_combined_markdown(output_dir: pathlib.Path, artifacts: list[TableArtifact]) -> None:
    lines = ["# Correctness Tables", ""]
    for artifact in artifacts:
        lines.append(f"## {artifact.title}")
        lines.append("")
        lines.extend(_render_markdown_table(artifact.columns, artifact.rows))
        if artifact.notes:
            lines.append("")
            lines.append("Notes:")
            for note in artifact.notes:
                lines.append(f"- {note}")
        lines.append("")
    (output_dir / "correctness_tables.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build correctness comparison tables from profiler and reference outputs."
    )
    parser.add_argument("--pmu-window-metrics", type=pathlib.Path, help="window_metrics.jsonl used for PMU comparison")
    parser.add_argument("--perf-stat", type=pathlib.Path, help="perf stat output file (plain text or -x, CSV style)")
    parser.add_argument("--function-hotspot", type=pathlib.Path, help="function_hotspot_<metric>.csv from analysis/attribution.py")
    parser.add_argument("--perf-report", type=pathlib.Path, help="perf report --stdio output file")
    parser.add_argument("--fault-window-metrics", type=pathlib.Path, help="window_metrics.jsonl used for fault comparison")
    parser.add_argument("--proc-stat-before", type=pathlib.Path, help="/proc/<pid>/stat snapshot captured before collection")
    parser.add_argument("--proc-stat-after", type=pathlib.Path, help="/proc/<pid>/stat snapshot captured after collection")
    parser.add_argument("--microbench-root", type=pathlib.Path, help="results/micro_bench_<timestamp> root directory")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K cutoff used for hotspot alignment")
    parser.add_argument("--output", type=pathlib.Path, required=True, help="output directory for CSV and Markdown tables")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    artifacts: list[TableArtifact] = []

    if args.pmu_window_metrics and args.perf_stat:
        artifacts.append(_build_pmu_table(args.pmu_window_metrics, args.perf_stat))

    if args.function_hotspot and args.perf_report:
        artifacts.extend(_build_hotspot_summary(args.function_hotspot, args.perf_report, args.top_k))

    if args.fault_window_metrics and args.proc_stat_before and args.proc_stat_after:
        artifacts.append(
            _build_fault_vs_proc_table(
                args.fault_window_metrics,
                args.proc_stat_before,
                args.proc_stat_after,
            )
        )

    if args.microbench_root:
        artifacts.append(_build_microbench_table(args.microbench_root))

    if not artifacts:
        print("error: no complete input pairs were provided", file=sys.stderr)
        return 2

    for artifact in artifacts:
        _write_csv(args.output / f"{artifact.stem}.csv", artifact.columns, artifact.rows)
        _write_markdown(args.output / f"{artifact.stem}.md", artifact)

    _write_combined_markdown(args.output, artifacts)
    print(f"wrote {len(artifacts)} tables to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())