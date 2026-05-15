#!/usr/bin/env python3
"""Build citation-ready methodology validation tables.

This script turns raw experiment outputs under results/overhead_*,
results/stability_*, and results/sensitivity_* into CSV/Markdown tables with
 explicit thresholds, verdicts, and recommended default settings.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime


OVERHEAD_TARGET_PCT = 5.0
OVERHEAD_ACCEPTABLE_PCT = 10.0
STABILITY_TARGET_CV_PCT = 10.0
STABILITY_ACCEPTABLE_CV_PCT = 15.0
SENSITIVITY_TARGET_DRIFT_PCT = 10.0
SENSITIVITY_ACCEPTABLE_DRIFT_PCT = 20.0

DEFAULT_SAMPLE_RATE = 100
DEFAULT_WINDOW_SEC = 1.0
DEFAULT_PROBE_PROFILE = "probe_all"

OVERHEAD_PERF_EVENTS = ["cycles", "instructions", "cache-misses"]
STABILITY_METRICS = [
    "cycles",
    "instructions",
    "llc_load_misses",
    "llc_store_misses",
    "dtlb_misses",
    "minor_faults",
    "major_faults",
]
SENSITIVITY_METRICS = [
    "cycles",
    "instructions",
    "llc_load_misses",
    "dtlb_misses",
    "minor_faults",
]
PROBE_COVERAGE_KEYS = ["llc", "dtlb", "fault"]

PERF_EVENT_ALIASES = {
    "cache-misses": "cache-misses",
    "cache_misses": "cache-misses",
    "cache misses": "cache-misses",
    "cycles": "cycles",
    "instructions": "instructions",
}


@dataclass
class TableArtifact:
    title: str
    stem: str
    columns: list[str]
    rows: list[dict[str, object]]
    notes: list[str] = field(default_factory=list)


@dataclass
class RunSummary:
    label: str
    path: pathlib.Path
    duration_sec: float
    window_sec: float | None
    sample_rate: int | None
    enabled_probes: dict[str, bool]
    totals: dict[str, float]
    rates: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overhead-dir", type=pathlib.Path, help="results/overhead_* 目录")
    parser.add_argument("--stability-dir", type=pathlib.Path, help="results/stability_* 目录")
    parser.add_argument("--sensitivity-dir", type=pathlib.Path, help="results/sensitivity_* 目录")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="输出目录；会写出 CSV、Markdown 和总览摘要",
    )
    return parser.parse_args()


def _parse_number(text: str) -> float | None:
    cleaned = text.strip().replace(",", "")
    if not cleaned or cleaned.startswith("<"):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _canonical_perf_event(name: str) -> str:
    key = name.strip().lower().replace("_", "-")
    key = re.sub(r"\s+", "-", key)
    return PERF_EVENT_ALIASES.get(key, key)


def _safe_rel_diff_pct(lhs: float, rhs: float) -> float | None:
    if rhs == 0:
        return None
    return abs(lhs - rhs) / abs(rhs) * 100.0


def _load_jsonl(path: pathlib.Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        value = json.loads(stripped)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _load_run_metadata(path: pathlib.Path) -> dict[str, object]:
    merged: dict[str, object] = {}
    for record in _load_jsonl(path):
        if record.get("_record_type") == "run_end":
            if "end_ts_iso" in record:
                merged["end_ts_iso"] = record["end_ts_iso"]
            continue
        merged.update(record)
    return merged


def _parse_iso(ts: object) -> datetime | None:
    if not isinstance(ts, str) or not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def _sum_window_metrics(path: pathlib.Path) -> tuple[dict[str, float], int]:
    totals: dict[str, float] = {}
    window_ids: set[int] = set()
    for row in _load_jsonl(path):
        window_id = row.get("window_id")
        if isinstance(window_id, int):
            window_ids.add(window_id)
        for key, value in row.items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0.0) + float(value)
    return totals, len(window_ids)


def _summarize_run(run_dir: pathlib.Path, label: str | None = None) -> RunSummary:
    meta = _load_run_metadata(run_dir / "run_metadata.jsonl")
    totals, window_count = _sum_window_metrics(run_dir / "window_metrics.jsonl")
    start_ts = _parse_iso(meta.get("start_ts_iso"))
    end_ts = _parse_iso(meta.get("end_ts_iso"))

    duration_sec = 0.0
    if start_ts is not None and end_ts is not None and end_ts > start_ts:
        duration_sec = (end_ts - start_ts).total_seconds()
    else:
        window_sec = meta.get("window_sec")
        if isinstance(window_sec, (int, float)) and window_count > 0:
            duration_sec = float(window_sec) * window_count

    rates: dict[str, float] = {}
    if duration_sec > 0:
        for key, value in totals.items():
            rates[key] = value / duration_sec

    enabled_probes = meta.get("enabled_probes")
    if not isinstance(enabled_probes, dict):
        enabled_probes = {}

    sample_rate = meta.get("sample_rate")
    sample_rate_int = int(sample_rate) if isinstance(sample_rate, (int, float)) else None
    window_sec_value = meta.get("window_sec")
    window_sec_float = (
        float(window_sec_value) if isinstance(window_sec_value, (int, float)) else None
    )

    return RunSummary(
        label=label or run_dir.name,
        path=run_dir,
        duration_sec=duration_sec,
        window_sec=window_sec_float,
        sample_rate=sample_rate_int,
        enabled_probes={str(key): bool(value) for key, value in enabled_probes.items()},
        totals=totals,
        rates=rates,
    )


def _parse_perf_stat(path: pathlib.Path) -> dict[str, float]:
    totals: dict[str, float] = {}
    if not path.exists():
        return totals

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("---"):
            continue

        value: float | None = None
        event_name = ""

        if "," in line:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 3:
                value = _parse_number(parts[0])
                event_name = parts[2]
        else:
            tokens = line.split()
            if len(tokens) >= 2:
                value = _parse_number(tokens[0])
                event_name = tokens[1]

        if value is None or not event_name:
            continue

        canonical = _canonical_perf_event(event_name)
        totals[canonical] = totals.get(canonical, 0.0) + value

    return totals


def _parse_timing_csv(path: pathlib.Path) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = {}
    if not path.exists():
        return groups

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            group = row[0].strip().lower()
            if group in {"group", "组别"}:
                continue
            if len(row) < 3:
                continue
            elapsed_ms = _parse_number(row[2])
            if elapsed_ms is None:
                continue
            groups.setdefault(group, []).append(elapsed_ms)
    return groups


def _verdict_lower_better(
    value: float | None,
    target: float,
    acceptable: float,
) -> str:
    if value is None:
        return "n/a"
    if value <= target:
        return "pass"
    if value <= acceptable:
        return "caution"
    return "fail"


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


def _write_bundle_markdown(path: pathlib.Path, artifacts: list[TableArtifact]) -> None:
    lines = ["# 方法学验证摘要", ""]
    lines.append("本文件汇总开销、稳定性、参数敏感性三类实验的自动化对照表。")
    for artifact in artifacts:
        lines.extend(["", f"## {artifact.title}", ""])
        lines.extend(_render_markdown_table(artifact.columns, artifact.rows))
        if artifact.notes:
            lines.append("")
            for note in artifact.notes:
                lines.append(f"- {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _overhead_recommendation(verdict: str) -> str:
    if verdict == "pass":
        return "keep routine config as default"
    if verdict == "caution":
        return "acceptable for short runs; prefer lighter probes for long sessions"
    if verdict == "fail":
        return "raise sample-rate period or disable emit-events/LBR before default use"
    return "collect paired baseline/profiled timing first"


def _stability_recommendation(verdict: str) -> str:
    if verdict == "pass":
        return "can be used for absolute comparison"
    if verdict == "caution":
        return "report with CV and treat as semi-quantitative"
    if verdict == "fail":
        return "trend-only; increase repeat count or duration"
    return "not enough comparable runs"


def _build_overhead_artifact(root: pathlib.Path) -> TableArtifact:
    rows: list[dict[str, object]] = []
    notes: list[str] = [
        f"Target threshold: overhead <= {OVERHEAD_TARGET_PCT}% ; acceptable <= {OVERHEAD_ACCEPTABLE_PCT}%.",
        "Routine default should keep emit-events and LBR disabled; only enable them for one-off attribution runs.",
    ]

    timing_groups = _parse_timing_csv(root / "timing.csv")
    baseline = timing_groups.get("baseline", [])
    profiled = timing_groups.get("ebpf", []) or timing_groups.get("profiled", [])
    if baseline and profiled:
        baseline_mean = statistics.mean(baseline)
        baseline_std = statistics.stdev(baseline) if len(baseline) > 1 else 0.0
        profiled_mean = statistics.mean(profiled)
        profiled_std = statistics.stdev(profiled) if len(profiled) > 1 else 0.0
        delta_pct = ((profiled_mean - baseline_mean) / baseline_mean * 100.0) if baseline_mean else None
        verdict = _verdict_lower_better(delta_pct, OVERHEAD_TARGET_PCT, OVERHEAD_ACCEPTABLE_PCT)
        rows.append(
            {
                "metric": "wall_time",
                "unit": "ms",
                "baseline_value": baseline_mean,
                "profiled_value": profiled_mean,
                "baseline_std": baseline_std,
                "profiled_std": profiled_std,
                "delta_pct": delta_pct,
                "target_pct": OVERHEAD_TARGET_PCT,
                "acceptable_pct": OVERHEAD_ACCEPTABLE_PCT,
                "verdict": verdict,
                "recommendation": _overhead_recommendation(verdict),
            }
        )
    else:
        notes.append("timing.csv is missing baseline/ebpf paired samples; wall-time overhead row was skipped.")

    perf_baseline = _parse_perf_stat(root / "perf_stat_baseline.csv")
    perf_profiled = _parse_perf_stat(root / "perf_stat_ebpf.csv")
    if perf_baseline and perf_profiled:
        for event in OVERHEAD_PERF_EVENTS:
            baseline_value = perf_baseline.get(event)
            profiled_value = perf_profiled.get(event)
            if baseline_value is None or profiled_value is None:
                continue
            delta_pct = _safe_rel_diff_pct(profiled_value, baseline_value)
            verdict = _verdict_lower_better(delta_pct, OVERHEAD_TARGET_PCT, OVERHEAD_ACCEPTABLE_PCT)
            rows.append(
                {
                    "metric": event,
                    "unit": "count",
                    "baseline_value": baseline_value,
                    "profiled_value": profiled_value,
                    "baseline_std": None,
                    "profiled_std": None,
                    "delta_pct": delta_pct,
                    "target_pct": OVERHEAD_TARGET_PCT,
                    "acceptable_pct": OVERHEAD_ACCEPTABLE_PCT,
                    "verdict": verdict,
                    "recommendation": _overhead_recommendation(verdict),
                }
            )
    else:
        notes.append(
            "perf_stat_baseline.csv + perf_stat_ebpf.csv not both present; perf-event overhead rows were skipped."
        )

    if not rows:
        raise ValueError(f"no usable overhead inputs found under {root}")

    return TableArtifact(
        title="开销验证摘要",
        stem="overhead_summary",
        columns=[
            "metric",
            "unit",
            "baseline_value",
            "profiled_value",
            "baseline_std",
            "profiled_std",
            "delta_pct",
            "target_pct",
            "acceptable_pct",
            "verdict",
            "recommendation",
        ],
        rows=rows,
        notes=notes,
    )


def _build_stability_artifact(root: pathlib.Path) -> TableArtifact:
    run_dirs = sorted(path for path in root.glob("run_*") if path.is_dir())
    summaries = [
        _summarize_run(run_dir)
        for run_dir in run_dirs
        if (run_dir / "window_metrics.jsonl").exists()
    ]
    if not summaries:
        raise ValueError(f"no run_* directories with window_metrics.jsonl found under {root}")

    rows: list[dict[str, object]] = []
    for metric in STABILITY_METRICS:
        values = [summary.rates.get(metric) for summary in summaries if metric in summary.rates]
        values = [value for value in values if value is not None]
        if not values:
            continue
        mean_value = statistics.mean(values)
        std_value = statistics.stdev(values) if len(values) > 1 else 0.0
        cv_pct = (std_value / mean_value * 100.0) if mean_value else None
        verdict = _verdict_lower_better(cv_pct, STABILITY_TARGET_CV_PCT, STABILITY_ACCEPTABLE_CV_PCT)
        rows.append(
            {
                "metric": metric,
                "runs": len(values),
                "mean_rate_per_sec": mean_value,
                "std_rate_per_sec": std_value,
                "cv_pct": cv_pct,
                "target_cv_pct": STABILITY_TARGET_CV_PCT,
                "acceptable_cv_pct": STABILITY_ACCEPTABLE_CV_PCT,
                "verdict": verdict,
                "recommendation": _stability_recommendation(verdict),
            }
        )

    if not rows:
        raise ValueError(f"no stability metrics could be summarized under {root}")

    repeat_count = len(summaries)
    duration_values = sorted({summary.duration_sec for summary in summaries if summary.duration_sec > 0})
    notes = [
        f"Stable threshold: CV <= {STABILITY_TARGET_CV_PCT}% ; caution band <= {STABILITY_ACCEPTABLE_CV_PCT}%.",
        f"Current repeat count = {repeat_count}; thesis default should use at least 10 repeats.",
    ]
    if duration_values:
        notes.append(f"Observed run durations (sec): {', '.join(_format_cell(v) for v in duration_values)}.")

    return TableArtifact(
        title="稳定性验证摘要",
        stem="stability_summary",
        columns=[
            "metric",
            "runs",
            "mean_rate_per_sec",
            "std_rate_per_sec",
            "cv_pct",
            "target_cv_pct",
            "acceptable_cv_pct",
            "verdict",
            "recommendation",
        ],
        rows=rows,
        notes=notes,
    )


def _parse_samplerate_label(label: str) -> int | None:
    match = re.fullmatch(r"samplerate_(\d+)", label)
    if match is None:
        return None
    return int(match.group(1))


def _parse_window_label(label: str) -> float | None:
    match = re.fullmatch(r"window_(\d+(?:\.\d+)?)s", label)
    if match is None:
        return None
    return float(match.group(1))


def _pick_preferred_row(
    rows: list[dict[str, object]],
    preferred_value: float,
) -> str | None:
    for verdict_group in ("pass", "caution"):
        candidates = [row for row in rows if row.get("verdict") == verdict_group]
        if not candidates:
            continue
        candidates.sort(
            key=lambda row: (
                abs(float(row["setting_value"]) - preferred_value),
                float(row["setting_value"]),
            )
        )
        return str(candidates[0]["setting"])
    return None


def _build_sensitivity_drift_artifact(
    root: pathlib.Path,
    title: str,
    stem: str,
    parser: callable,
    reference_label: str,
    preferred_value: float,
) -> tuple[TableArtifact, str | None]:
    run_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    candidates: list[tuple[str, float, RunSummary]] = []
    for run_dir in run_dirs:
        setting_value = parser(run_dir.name)
        if setting_value is None:
            continue
        if not (run_dir / "window_metrics.jsonl").exists():
            continue
        candidates.append((run_dir.name, float(setting_value), _summarize_run(run_dir)))

    if not candidates:
        raise ValueError(f"no matching sensitivity runs found for {stem} under {root}")

    reference = next((summary for label, _, summary in candidates if label == reference_label), None)
    if reference is None:
        raise ValueError(f"missing reference run '{reference_label}' under {root}")

    rows: list[dict[str, object]] = []
    for label, setting_value, summary in sorted(candidates, key=lambda item: item[1]):
        comparable_metrics: list[str] = []
        drifts: list[float] = []
        for metric in SENSITIVITY_METRICS:
            ref_value = reference.rates.get(metric)
            sample_value = summary.rates.get(metric)
            if ref_value is None or sample_value is None or ref_value == 0:
                continue
            comparable_metrics.append(metric)
            drift = _safe_rel_diff_pct(sample_value, ref_value)
            if drift is not None:
                drifts.append(drift)
        median_drift = statistics.median(drifts) if drifts else None
        max_drift = max(drifts) if drifts else None
        verdict = _verdict_lower_better(
            max_drift,
            SENSITIVITY_TARGET_DRIFT_PCT,
            SENSITIVITY_ACCEPTABLE_DRIFT_PCT,
        )
        rows.append(
            {
                "setting": label,
                "setting_value": setting_value,
                "reference_setting": reference_label,
                "comparable_metrics": ",".join(comparable_metrics),
                "median_drift_pct": median_drift,
                "max_drift_pct": max_drift,
                "target_drift_pct": SENSITIVITY_TARGET_DRIFT_PCT,
                "acceptable_drift_pct": SENSITIVITY_ACCEPTABLE_DRIFT_PCT,
                "verdict": verdict,
                "recommended_default": False,
                "recommendation": "",
            }
        )

    recommended_label = _pick_preferred_row(rows, preferred_value)
    for row in rows:
        is_recommended = row["setting"] == recommended_label
        row["recommended_default"] = is_recommended
        if row["verdict"] == "n/a":
            row["recommendation"] = "target workload did not produce enough non-zero comparable metrics"
        elif is_recommended:
            row["recommendation"] = "recommended default"
        elif row["verdict"] == "pass":
            row["recommendation"] = "acceptable alternative"
        elif row["verdict"] == "caution":
            row["recommendation"] = "use only when trading fidelity for lower cost or smoother windows"
        else:
            row["recommendation"] = "do not use as default"

    notes = [
        f"Drift is computed on per-second rates against {reference_label}.",
        f"Target threshold: max drift <= {SENSITIVITY_TARGET_DRIFT_PCT}% ; acceptable <= {SENSITIVITY_ACCEPTABLE_DRIFT_PCT}%.",
    ]

    return (
        TableArtifact(
            title=title,
            stem=stem,
            columns=[
                "setting",
                "reference_setting",
                "comparable_metrics",
                "median_drift_pct",
                "max_drift_pct",
                "target_drift_pct",
                "acceptable_drift_pct",
                "verdict",
                "recommended_default",
                "recommendation",
            ],
            rows=rows,
            notes=notes,
        ),
        recommended_label,
    )


def _probe_profile_recommendation(label: str) -> str:
    if label == DEFAULT_PROBE_PROFILE:
        return "routine default; use for overview, hotspot, and feature extraction"
    if label == "probe_llc_only":
        return "specialized mode for LLC-only diagnosis"
    if label == "probe_fault_only":
        return "specialized mode for page-fault-focused diagnosis"
    return "specialized profile"


def _build_probe_artifact(root: pathlib.Path) -> tuple[TableArtifact, str]:
    run_dirs = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("probe_"))
    if not run_dirs:
        raise ValueError(f"no probe_* runs found under {root}")

    rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        summary = _summarize_run(run_dir)
        coverage = [summary.enabled_probes.get(key, False) for key in PROBE_COVERAGE_KEYS]
        coverage_ratio = sum(1 for value in coverage if value) / len(PROBE_COVERAGE_KEYS)
        missing = [key for key in PROBE_COVERAGE_KEYS if not summary.enabled_probes.get(key, False)]
        verdict = "default" if run_dir.name == DEFAULT_PROBE_PROFILE else "specialized"
        rows.append(
            {
                "setting": run_dir.name,
                "enabled_llc": summary.enabled_probes.get("llc", False),
                "enabled_dtlb": summary.enabled_probes.get("dtlb", False),
                "enabled_fault": summary.enabled_probes.get("fault", False),
                "coverage_ratio": coverage_ratio,
                "missing_core_probes": ",".join(missing),
                "verdict": verdict,
                "recommended_default": run_dir.name == DEFAULT_PROBE_PROFILE,
                "recommendation": _probe_profile_recommendation(run_dir.name),
            }
        )

    notes = [
        "Probe coverage threshold for the routine default is full llc + dtlb + fault coverage.",
        "Partial probe sets are diagnostic-only and should not replace the default collection profile in thesis tables.",
    ]

    return (
        TableArtifact(
            title="探针组合摘要",
            stem="sensitivity_probe_summary",
            columns=[
                "setting",
                "enabled_llc",
                "enabled_dtlb",
                "enabled_fault",
                "coverage_ratio",
                "missing_core_probes",
                "verdict",
                "recommended_default",
                "recommendation",
            ],
            rows=rows,
            notes=notes,
        ),
        DEFAULT_PROBE_PROFILE,
    )


def _format_probe_profile(label: str) -> str:
    if label == "probe_all":
        return "llc+dtlb+itlb+fault+mm_syscalls"
    if label == "probe_llc_only":
        return "llc only"
    if label == "probe_fault_only":
        return "fault only"
    return label


def _build_recommendation_artifact(
    overhead_present: bool,
    stability_present: bool,
    sample_rate_label: str | None,
    window_label: str | None,
    probe_label: str | None,
) -> TableArtifact:
    sample_rate_value = _parse_samplerate_label(sample_rate_label or "") or DEFAULT_SAMPLE_RATE
    window_value = _parse_window_label(window_label or "") or DEFAULT_WINDOW_SEC
    probe_value = probe_label or DEFAULT_PROBE_PROFILE
    routine_config = (
        f"window={window_value}s, sample_rate={sample_rate_value}, probes={_format_probe_profile(probe_value)}, "
        "emit-events=off, lbr=off"
    )

    rows: list[dict[str, object]] = []
    if overhead_present:
        rows.append(
            {
                "area": "overhead",
                "acceptance_rule": f"wall-time/perf delta <= {OVERHEAD_TARGET_PCT}% target, <= {OVERHEAD_ACCEPTABLE_PCT}% acceptable",
                "recommended_configuration": routine_config,
                "paper_wording": "只要默认配置保持在 5% 目标预算内，就可作为常规采集配置；超过 10% 时不应作为默认设置。",
            }
        )
    if stability_present:
        rows.append(
            {
                "area": "stability",
                "acceptance_rule": f"CV <= {STABILITY_TARGET_CV_PCT}% stable, <= {STABILITY_ACCEPTABLE_CV_PCT}% caution",
                "recommended_configuration": "repeat>=10, duration>=10s, report mean + std + CV",
                "paper_wording": "仅对 CV 不超过 10% 的指标做绝对量比较；其余指标只做趋势或热点排序解释。",
            }
        )
    rows.extend(
        [
            {
                "area": "sample_rate",
                "acceptance_rule": f"max drift <= {SENSITIVITY_TARGET_DRIFT_PCT}% vs samplerate_100",
                "recommended_configuration": f"sample_rate={sample_rate_value}",
                "paper_wording": "默认采样率取能通过 10% 漂移预算的保守配置；若需更低开销再提高 sample_rate period。",
            },
            {
                "area": "window",
                "acceptance_rule": f"max drift <= {SENSITIVITY_TARGET_DRIFT_PCT}% vs window_1.0s",
                "recommended_configuration": f"window={window_value}s",
                "paper_wording": "1.0s 作为默认窗口长度，兼顾时间定位能力与跨 run 稳定性。",
            },
            {
                "area": "probe_profile",
                "acceptance_rule": "default profile must keep llc + dtlb + fault all enabled",
                "recommended_configuration": f"{probe_value} ({_format_probe_profile(probe_value)})",
                "paper_wording": "probe_all 用于常规采集；精简探针组合只作为定向诊断模式，不替代正文默认配置。",
            },
            {
                "area": "deep_attribution",
                "acceptance_rule": "emit-events/LBR are diagnostic-only, not part of the low-overhead default",
                "recommended_configuration": "reuse routine config, then enable emit-events and optional LBR only for one-off attribution runs",
                "paper_wording": "函数级归因和 LBR 证据链属于深度诊断模式，应与低开销常规采集分开陈述。",
            },
        ]
    )

    return TableArtifact(
        title="推荐配置与论文口径",
        stem="methodology_recommendations",
        columns=["area", "acceptance_rule", "recommended_configuration", "paper_wording"],
        rows=rows,
        notes=[
            "This table fixes the thesis-facing thresholds and the default collection profile.",
            "If the current run does not contain one of the sensitivity references, the script falls back to window=1.0s and sample_rate=100.",
        ],
    )


def _write_artifact(output_dir: pathlib.Path, artifact: TableArtifact) -> None:
    _write_csv(output_dir / f"{artifact.stem}.csv", artifact.columns, artifact.rows)
    _write_markdown(output_dir / f"{artifact.stem}.md", artifact)


def main() -> None:
    args = parse_args()
    if not any([args.overhead_dir, args.stability_dir, args.sensitivity_dir]):
        sys.exit("[error] provide at least one of --overhead-dir / --stability-dir / --sensitivity-dir")

    args.output.mkdir(parents=True, exist_ok=True)
    artifacts: list[TableArtifact] = []

    overhead_present = False
    stability_present = False
    sample_rate_label: str | None = None
    window_label: str | None = None
    probe_label: str | None = None

    if args.overhead_dir is not None:
        artifact = _build_overhead_artifact(args.overhead_dir)
        _write_artifact(args.output, artifact)
        artifacts.append(artifact)
        overhead_present = True

    if args.stability_dir is not None:
        artifact = _build_stability_artifact(args.stability_dir)
        _write_artifact(args.output, artifact)
        artifacts.append(artifact)
        stability_present = True

    if args.sensitivity_dir is not None:
        sample_artifact, sample_rate_label = _build_sensitivity_drift_artifact(
            args.sensitivity_dir,
            title="采样率敏感性摘要",
            stem="sensitivity_samplerate_summary",
            parser=_parse_samplerate_label,
            reference_label="samplerate_100",
            preferred_value=float(DEFAULT_SAMPLE_RATE),
        )
        _write_artifact(args.output, sample_artifact)
        artifacts.append(sample_artifact)

        window_artifact, window_label = _build_sensitivity_drift_artifact(
            args.sensitivity_dir,
            title="窗口长度敏感性摘要",
            stem="sensitivity_window_summary",
            parser=_parse_window_label,
            reference_label="window_1.0s",
            preferred_value=DEFAULT_WINDOW_SEC,
        )
        _write_artifact(args.output, window_artifact)
        artifacts.append(window_artifact)

        probe_artifact, probe_label = _build_probe_artifact(args.sensitivity_dir)
        _write_artifact(args.output, probe_artifact)
        artifacts.append(probe_artifact)

    recommendation_artifact = _build_recommendation_artifact(
        overhead_present=overhead_present,
        stability_present=stability_present,
        sample_rate_label=sample_rate_label,
        window_label=window_label,
        probe_label=probe_label,
    )
    _write_artifact(args.output, recommendation_artifact)
    artifacts.append(recommendation_artifact)

    _write_bundle_markdown(args.output / "methodology_validation.md", artifacts)


if __name__ == "__main__":
    main()