#!/usr/bin/env python3
"""
audit_train_set_quality.py — 审计当前 train_set 与 curated manifests 的数据质量

输出：
  1. train_set/data_quality_audit.json
  2. docs/new-repo-plan/current-data-quality-audit.md

用途：
  - 固定当前 raw/curated/train_set 的真实口径
  - 拉出语义过滤与严格时间过滤的完整名单
  - 汇总 O2/O3 难例与 tie 密集区间，给后续补采或调阈值提供依据

用法：
  python scripts/audit_train_set_quality.py
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import pathlib
from collections import Counter, defaultdict
from typing import Any

from build_run_features import (  # noqa: PLC2701
    REPO_ROOT,
    VARIANTS,
    _load_manifest,
    _semantic_invalid_reasons,
    aggregate_run,
)


VARIANT_RANK = {variant: index for index, variant in enumerate(VARIANTS)}
DEFAULT_BASELINE = "O0"
DEFAULT_TIE_THRESHOLD = 0.05
DEFAULT_NEAR_TIE_THRESHOLD = 0.25
DEFAULT_MIN_ACTIVE_PIDS = 5
DEFAULT_MIN_ACTIVE_WINDOW_RATIO = 0.10
DEFAULT_MIN_CYCLES_PER_ITER = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=str(REPO_ROOT / "data" / "llvm_test_suite"),
        help="llvm-test-suite 数据根目录",
    )
    parser.add_argument(
        "--manifest-prefix",
        default="manifest_curated",
        help="要审计的 manifest 前缀（默认 manifest_curated）",
    )
    parser.add_argument(
        "--run-features-csv",
        default=str(REPO_ROOT / "train_set" / "run_features.csv"),
        help="run_features.csv 路径",
    )
    parser.add_argument(
        "--pairs-csv",
        default=str(REPO_ROOT / "train_set" / "pairs.csv"),
        help="pairs.csv 路径",
    )
    parser.add_argument(
        "--run-filter-summary",
        default=str(REPO_ROOT / "train_set" / "run_feature_filter_summary.json"),
        help="run_feature_filter_summary.json 路径",
    )
    parser.add_argument(
        "--time-filter-summary",
        default=str(REPO_ROOT / "train_set" / "time_score_filter_summary.json"),
        help="time_score_filter_summary.json 路径",
    )
    parser.add_argument(
        "--pairs-stats",
        default=str(REPO_ROOT / "train_set" / "pairs_stats.json"),
        help="pairs_stats.json 路径",
    )
    parser.add_argument(
        "--anchor-stats",
        default=str(REPO_ROOT / "train_set" / "anchor_set.stats.json"),
        help="anchor_set.stats.json 路径",
    )
    parser.add_argument(
        "--transformer-eval",
        default=str(REPO_ROOT / "train_set" / "model_transformer_eval.json"),
        help="model_transformer_eval.json 路径",
    )
    parser.add_argument(
        "--score-eval",
        default=str(REPO_ROOT / "train_set" / "score_eval.json"),
        help="score_eval.json 路径",
    )
    parser.add_argument(
        "--score-time-eval",
        default=str(REPO_ROOT / "train_set" / "score_time_eval.json"),
        help="score_time_eval.json 路径",
    )
    parser.add_argument(
        "--manifest-summary",
        default=str(REPO_ROOT / "data" / "llvm_test_suite" / "manifest_curated_summary.json"),
        help="manifest_curated_summary.json 路径",
    )
    parser.add_argument(
        "--json-output",
        default=str(REPO_ROOT / "train_set" / "data_quality_audit.json"),
        help="JSON 审计输出路径",
    )
    parser.add_argument(
        "--markdown-output",
        default=str(REPO_ROOT / "docs" / "new-repo-plan" / "current-data-quality-audit.md"),
        help="Markdown 审计输出路径",
    )
    parser.add_argument(
        "--baseline",
        default=DEFAULT_BASELINE,
        help="严格时间口径的基线 variant（默认 O0）",
    )
    parser.add_argument(
        "--min-active-pids",
        type=int,
        default=DEFAULT_MIN_ACTIVE_PIDS,
        help="语义过滤与严格时间过滤的最小 active_pid_count",
    )
    parser.add_argument(
        "--min-active-window-ratio",
        type=float,
        default=DEFAULT_MIN_ACTIVE_WINDOW_RATIO,
        help="严格时间过滤要求的最小 active_window_ratio",
    )
    parser.add_argument(
        "--min-cycles-per-iter",
        type=float,
        default=DEFAULT_MIN_CYCLES_PER_ITER,
        help="语义过滤要求的最小 cycles_per_iter",
    )
    parser.add_argument(
        "--tie-threshold",
        type=float,
        default=DEFAULT_TIE_THRESHOLD,
        help="tie 判定阈值（默认 0.05）",
    )
    parser.add_argument(
        "--near-tie-threshold",
        type=float,
        default=DEFAULT_NEAR_TIE_THRESHOLD,
        help="near-tie 判定阈值（默认 0.25）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Markdown 中每类样本展示的前 K 条记录",
    )
    return parser.parse_args()


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_csv_rows(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _relative_to_repo(path: pathlib.Path | str) -> str:
    path_obj = pathlib.Path(path)
    resolved = path_obj.resolve() if path_obj.is_absolute() else (REPO_ROOT / path_obj).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _resolve_output_dir(output_dir: str) -> pathlib.Path:
    path = pathlib.Path(output_dir)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _round_float(value: float, digits: int = 6) -> float:
    if math.isnan(value) or math.isinf(value):
        return value
    return round(value, digits)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[index])


def _format_float(value: float, digits: int = 4) -> str:
    if value is None:
        return "-"
    if math.isnan(value) or math.isinf(value):
        return str(value)
    return f"{value:.{digits}f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["（无）"]
    divider = ["---"] * len(headers)
    table = [f"| {' | '.join(headers)} |", f"| {' | '.join(divider)} |"]
    for row in rows:
        table.append(f"| {' | '.join(row)} |")
    return table


def _load_meta(meta_path: pathlib.Path) -> tuple[str, int | None]:
    run_id = "unknown"
    completion_count: int | None = None
    if not meta_path.exists():
        return run_id, completion_count

    for line in meta_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if run_id == "unknown" and "run_id" in record:
            run_id = str(record["run_id"])
        if "completion_count" in record:
            completion_count = _safe_int(record.get("completion_count"), default=completion_count or 0)
    return run_id, completion_count


def _compute_semantic_filter_audit(
    data_root: pathlib.Path,
    manifest_prefix: str,
    min_active_pids: int,
    min_cycles_per_iter: float,
) -> dict[str, Any]:
    filtered_runs: list[dict[str, Any]] = []
    counts_by_variant = {
        variant: {"seen": 0, "filtered": 0, "kept": 0}
        for variant in VARIANTS
    }
    reason_counts = Counter()
    n_seen = 0

    for variant in VARIANTS:
        manifest_path = data_root / f"{manifest_prefix}_{variant}.jsonl"
        if not manifest_path.exists():
            continue
        for entry in _load_manifest(manifest_path):
            counts_by_variant[variant]["seen"] += 1
            n_seen += 1

            output_dir = _resolve_output_dir(entry["output_dir"])
            windows = _read_jsonl(output_dir / "window_metrics.jsonl")
            run_id, completion_count = _load_meta(output_dir / "run_metadata.jsonl")
            feature_row = aggregate_run(windows, completion_count=completion_count)
            reasons = _semantic_invalid_reasons(
                feature_row,
                min_active_pids=min_active_pids,
                min_cycles_per_iter=min_cycles_per_iter,
            )
            if reasons:
                counts_by_variant[variant]["filtered"] += 1
                reason_counts.update(reasons)
                filtered_runs.append({
                    "program": entry["program"],
                    "variant": variant,
                    "run_id": run_id,
                    "completion_count": completion_count,
                    "active_pid_count": _safe_int(feature_row.get("active_pid_count")),
                    "window_count": _safe_int(feature_row.get("window_count")),
                    "active_window_count": _safe_int(feature_row.get("active_window_count")),
                    "active_window_ratio": _round_float(
                        _safe_div(
                            _safe_float(feature_row.get("active_window_count")),
                            _safe_float(feature_row.get("window_count")),
                        )
                    ),
                    "cycles_per_iter": _round_float(_safe_float(feature_row.get("cycles_per_iter"))),
                    "wall_time_sec": _round_float(_safe_float(feature_row.get("wall_time_sec"))),
                    "output_dir": _relative_to_repo(output_dir),
                    "reasons": reasons,
                })
            else:
                counts_by_variant[variant]["kept"] += 1

    filtered_runs.sort(key=lambda row: (VARIANT_RANK[row["variant"]], row["program"]))
    return {
        "n_seen": n_seen,
        "n_filtered": len(filtered_runs),
        "n_kept": n_seen - len(filtered_runs),
        "reasons": dict(reason_counts),
        "by_variant": counts_by_variant,
        "filtered_runs": filtered_runs,
    }


def _normalize_run_feature_row(
    row: dict[str, str],
    min_active_pids: int,
    min_active_window_ratio: float,
) -> dict[str, Any]:
    active_pid_count = _safe_int(row.get("active_pid_count"))
    window_count = _safe_int(row.get("window_count"))
    active_window_count = _safe_int(row.get("active_window_count"))
    active_window_ratio = _safe_div(active_window_count, window_count)

    reasons: list[str] = []
    if active_pid_count < min_active_pids:
        reasons.append("low_active_pid_count")
    if active_window_ratio < min_active_window_ratio:
        reasons.append("low_active_window_ratio")

    normalized = {
        "run_id": row.get("run_id", "unknown"),
        "program": row["program"],
        "variant": row["variant"],
        "wall_time_sec": _round_float(_safe_float(row.get("wall_time_sec"))),
        "window_count": window_count,
        "active_window_count": active_window_count,
        "active_window_ratio": _round_float(active_window_ratio),
        "active_pid_count": active_pid_count,
        "cycles_per_iter": _round_float(_safe_float(row.get("cycles_per_iter"))),
        "phase_ipc_ratio": _round_float(_safe_float(row.get("phase_ipc_ratio"))),
        "phase_llc_ratio": _round_float(_safe_float(row.get("phase_llc_ratio"))),
        "phase_fault_ratio": _round_float(_safe_float(row.get("phase_fault_ratio"))),
        "output_dir": _relative_to_repo(row.get("output_dir", "")),
        "time_score_input_ok": len(reasons) == 0,
        "time_score_invalid_reasons": reasons,
    }
    return normalized


def _compute_time_filter_audit(
    run_feature_rows: list[dict[str, Any]],
    baseline: str,
) -> dict[str, Any]:
    filtered_runs = [
        row for row in run_feature_rows
        if not row["time_score_input_ok"]
    ]
    filtered_runs.sort(key=lambda row: (VARIANT_RANK[row["variant"]], row["program"]))

    strict_baseline_programs = {
        row["program"]
        for row in run_feature_rows
        if row["variant"] == baseline and row["time_score_input_ok"]
    }
    missing_baseline_rows = []
    missing_baseline_map: dict[str, dict[str, Any]] = {}
    for row in run_feature_rows:
        if not row["time_score_input_ok"]:
            continue
        if row["program"] in strict_baseline_programs:
            continue
        missing_baseline_rows.append({
            "program": row["program"],
            "variant": row["variant"],
            "active_window_ratio": row["active_window_ratio"],
            "wall_time_sec": row["wall_time_sec"],
            "output_dir": row["output_dir"],
        })
        item = missing_baseline_map.setdefault(
            row["program"],
            {
                "program": row["program"],
                "present_variants": [],
                "strict_input_variants": [],
            },
        )
        item["strict_input_variants"].append(row["variant"])

    coverage_map: dict[str, set[str]] = defaultdict(set)
    for row in run_feature_rows:
        coverage_map[row["program"]].add(row["variant"])
    for item in missing_baseline_map.values():
        present_variants = sorted(coverage_map[item["program"]], key=VARIANT_RANK.get)
        item["present_variants"] = present_variants
        item["missing_variants"] = [variant for variant in VARIANTS if variant not in present_variants]
        item["strict_input_variants"] = sorted(
            set(item["strict_input_variants"]),
            key=VARIANT_RANK.get,
        )

    missing_baseline_programs = sorted(missing_baseline_map.values(), key=lambda row: row["program"])
    missing_baseline_rows.sort(key=lambda row: (VARIANT_RANK[row["variant"]], row["program"]))

    return {
        "n_input_filtered": len(filtered_runs),
        "n_input_ok": len(run_feature_rows) - len(filtered_runs),
        "filtered_runs": filtered_runs,
        "missing_strict_baseline_rows": missing_baseline_rows,
        "n_missing_strict_baseline_rows": len(missing_baseline_rows),
        "missing_strict_baseline_programs": missing_baseline_programs,
        "n_missing_strict_baseline_programs": len(missing_baseline_programs),
    }


def _compute_coverage_gaps(run_feature_rows: list[dict[str, Any]]) -> dict[str, Any]:
    coverage_map: dict[str, set[str]] = defaultdict(set)
    for row in run_feature_rows:
        coverage_map[row["program"]].add(row["variant"])

    incomplete_programs = []
    for program, present in sorted(coverage_map.items()):
        missing = [variant for variant in VARIANTS if variant not in present]
        if missing:
            incomplete_programs.append({
                "program": program,
                "present_variants": sorted(present, key=VARIANT_RANK.get),
                "missing_variants": missing,
            })

    return {
        "n_programs": len(coverage_map),
        "n_complete_programs": len(coverage_map) - len(incomplete_programs),
        "n_incomplete_programs": len(incomplete_programs),
        "incomplete_programs": incomplete_programs,
    }


def _build_pair_summary(
    pair_rows: list[dict[str, str]],
    transformer_eval: dict[str, Any],
    tie_threshold: float,
    near_tie_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    canonical_rows = [
        row for row in pair_rows
        if VARIANT_RANK[row["variant_i"]] < VARIANT_RANK[row["variant_j"]]
    ]
    by_pair: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in canonical_rows:
        key = f"{row['variant_i']}-{row['variant_j']}"
        by_pair[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    o2_o3_rows: list[dict[str, Any]] = []
    o2_o3_category_counts = Counter()
    per_pair_eval = transformer_eval.get("per_pair", {})

    for pair_name, rows in sorted(by_pair.items(), key=lambda item: item[0]):
        abs_log_ratios = [abs(_safe_float(row["log_ratio"])) for row in rows]
        label_counts = Counter(row["label_class"] for row in rows)
        metrics = per_pair_eval.get(pair_name, {})
        summary_rows.append({
            "pair": pair_name,
            "n_programs": len(rows),
            "tie_count": label_counts.get("tie", 0),
            "tie_rate": _round_float(label_counts.get("tie", 0) / len(rows)),
            "mean_abs_log_ratio": _round_float(sum(abs_log_ratios) / len(abs_log_ratios)),
            "median_abs_log_ratio": _round_float(_quantile(abs_log_ratios, 0.5)),
            "p90_abs_log_ratio": _round_float(_quantile(abs_log_ratios, 0.9)),
            "test_dir_acc": metrics.get("dir_acc"),
            "test_acc_3cls": metrics.get("acc_3cls"),
            "test_aux_tie_recall": metrics.get("aux_tie_recall"),
        })

        if pair_name != "O2-O3":
            continue

        for row in rows:
            abs_log_ratio = abs(_safe_float(row["log_ratio"]))
            if abs_log_ratio <= tie_threshold:
                bucket = "tie_threshold_candidates"
            elif abs_log_ratio <= near_tie_threshold:
                bucket = "near_tie_threshold_candidates"
            else:
                bucket = "sequence_feature_review_candidates"

            o2_o3_category_counts[bucket] += 1
            o2_o3_rows.append({
                "program": row["program"],
                "variant_i": row["variant_i"],
                "variant_j": row["variant_j"],
                "label_class": row["label_class"],
                "log_ratio": _round_float(_safe_float(row["log_ratio"])),
                "abs_log_ratio": _round_float(abs_log_ratio),
                "difficulty_bucket": bucket,
            })

    o2_o3_rows.sort(key=lambda row: (row["abs_log_ratio"], row["program"]))
    return summary_rows, o2_o3_rows, dict(o2_o3_category_counts)


def _attach_o2_o3_run_context(
    o2_o3_rows: list[dict[str, Any]],
    run_feature_index: dict[tuple[str, str], dict[str, Any]],
    min_active_window_ratio: float,
) -> list[dict[str, Any]]:
    contextual_rows: list[dict[str, Any]] = []
    for row in o2_o3_rows:
        program = row["program"]
        o2 = run_feature_index.get((program, "O2"))
        o3 = run_feature_index.get((program, "O3"))
        o2_ratio = _safe_float((o2 or {}).get("active_window_ratio"), default=float("nan"))
        o3_ratio = _safe_float((o3 or {}).get("active_window_ratio"), default=float("nan"))
        needs_repeat_timing = (
            o2 is None
            or o3 is None
            or o2_ratio < min_active_window_ratio
            or o3_ratio < min_active_window_ratio
        )
        contextual_rows.append({
            **row,
            "o2_active_window_ratio": _round_float(o2_ratio) if not math.isnan(o2_ratio) else None,
            "o3_active_window_ratio": _round_float(o3_ratio) if not math.isnan(o3_ratio) else None,
            "o2_phase_ipc_ratio": (o2 or {}).get("phase_ipc_ratio"),
            "o3_phase_ipc_ratio": (o3 or {}).get("phase_ipc_ratio"),
            "o2_phase_llc_ratio": (o2 or {}).get("phase_llc_ratio"),
            "o3_phase_llc_ratio": (o3 or {}).get("phase_llc_ratio"),
            "needs_repeat_timing": needs_repeat_timing,
            "action_bucket": (
                "repeat_timing_candidates" if needs_repeat_timing else row["difficulty_bucket"]
            ),
        })

    contextual_rows.sort(key=lambda row: (VARIANT_RANK["O2"], row["abs_log_ratio"], row["program"]))
    return contextual_rows


def _build_recommendations(
    semantic_filter: dict[str, Any],
    time_filter: dict[str, Any],
    coverage_gaps: dict[str, Any],
    o2_o3_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    action_counts = Counter(row["action_bucket"] for row in o2_o3_rows)
    return [
        {
            "priority": "P1",
            "title": "优先补 run_features 过滤口径与文档同步",
            "why": (
                f"当前 curated 账本是 145x4，但进入训练链路的只剩 {semantic_filter['n_kept']} runs；"
                f"语义过滤丢掉了 {semantic_filter['n_filtered']} runs，必须以过滤后口径描述模型结果。"
            ),
        },
        {
            "priority": "P2",
            "title": "优先补 strict-time 真值而不是继续扩样",
            "why": (
                f"严格时间口径先在输入阶段筛掉了 {time_filter['n_input_filtered']} runs，"
                f"另有 {time_filter['n_missing_strict_baseline_rows']} 条 run"
                f"（分布在 {time_filter['n_missing_strict_baseline_programs']} 个程序上）缺 strict O0 baseline。"
            ),
        },
        {
            "priority": "P3",
            "title": "把 O2/O3 难例拆成补采、调阈值、补时序特征三类处理",
            "why": (
                f"O2/O3 难例里 repeat-timing 候选 {action_counts.get('repeat_timing_candidates', 0)} 个，"
                f"tie/near-tie 阈值候选 {action_counts.get('tie_threshold_candidates', 0) + action_counts.get('near_tie_threshold_candidates', 0)} 个；"
                f"剩余 {action_counts.get('sequence_feature_review_candidates', 0)} 个更适合先查时序特征。"
            ),
        },
        {
            "priority": "P4",
            "title": "把 10 个缺失变体程序从主评估口径里单列",
            "why": (
                f"当前只有 {coverage_gaps['n_complete_programs']} 个程序保留了完整 O0/O1/O2/O3，"
                f"仍有 {coverage_gaps['n_incomplete_programs']} 个程序在过滤后缺变体。"
            ),
        },
    ]


def _build_markdown(
    audit: dict[str, Any],
    markdown_path: pathlib.Path,
    top_k: int,
) -> str:
    json_link = f"../../train_set/{audit['artifacts']['json_output_name']}"
    lines = [
        "# 当前数据质量审计",
        "",
        f"> 生成时间：{audit['generated_at']}  ",
        f"> 生成脚本：[scripts/audit_train_set_quality.py](../../scripts/audit_train_set_quality.py)  ",
        f"> 完整 JSON：[{audit['artifacts']['json_output_name']}]({json_link})",
        "",
        "## 1. 当前口径",
        "",
        f"1. 当前 raw/curated manifests 已是严格的 145x4，shared_program_count={audit['manifest_snapshot']['shared_program_count']}。",
        f"2. 当前训练链路使用的 run_features 是过滤后的子集：{audit['train_snapshot']['n_runs']} runs、{audit['train_snapshot']['n_pairs']} pairs、{audit['train_snapshot']['n_anchors']} anchors。",
        f"3. 当前过滤后仍只有 {audit['coverage_gaps']['n_complete_programs']} 个完整四变体程序，另有 {audit['coverage_gaps']['n_incomplete_programs']} 个程序缺至少一个 variant。",
        "",
        "## 2. 语义过滤完整名单",
        "",
        f"1. 当前 curated 账本共 {audit['semantic_filter']['n_seen']} runs，其中 {audit['semantic_filter']['n_filtered']} runs 被语义过滤，保留 {audit['semantic_filter']['n_kept']} runs。",
        f"2. 过滤原因里 `low_active_pid_count={audit['semantic_filter']['reasons'].get('low_active_pid_count', 0)}`，`nonpositive_cycles_per_iter={audit['semantic_filter']['reasons'].get('nonpositive_cycles_per_iter', 0)}`。",
        f"3. 完整名单在 [{audit['artifacts']['json_output_name']}]({json_link}) 的 `semantic_filter.filtered_runs`。",
        "",
    ]
    semantic_rows = [
        [
            row['variant'],
            row['program'],
            str(row['active_pid_count']),
            _format_float(row['cycles_per_iter'], 2),
            ", ".join(row['reasons']),
        ]
        for row in audit['semantic_filter']['filtered_runs'][:top_k]
    ]
    lines.extend(_markdown_table(["Variant", "Program", "active_pid_count", "cycles_per_iter", "reasons"], semantic_rows))
    lines.extend([
        "",
        "## 3. 严格时间过滤与缺失基线",
        "",
        f"1. 当前 run_features 里有 {audit['strict_time_filter']['n_input_filtered']} runs 进不了 strict-time 输入，主要原因都是低 `active_window_ratio`。",
        f"2. 另有 {audit['strict_time_filter']['n_missing_strict_baseline_rows']} 条 run"
        f"（分布在 {audit['strict_time_filter']['n_missing_strict_baseline_programs']} 个程序上）虽通过 strict 输入检查，但缺 strict O0 baseline。",
        f"3. 完整名单分别在 [{audit['artifacts']['json_output_name']}]({json_link}) 的 `strict_time_filter.filtered_runs`、`strict_time_filter.missing_strict_baseline_rows` 与 `strict_time_filter.missing_strict_baseline_programs`。",
        "",
    ])
    strict_rows = [
        [
            row['variant'],
            row['program'],
            _format_float(row['active_window_ratio'], 4),
            str(row['active_window_count']),
            str(row['window_count']),
            ", ".join(row['time_score_invalid_reasons']),
        ]
        for row in audit['strict_time_filter']['filtered_runs'][:top_k]
    ]
    lines.extend(_markdown_table(["Variant", "Program", "active_window_ratio", "active_windows", "windows", "reasons"], strict_rows))
    lines.extend([
        "",
        "### 3.1 缺失 strict O0 baseline 的程序",
        "",
    ])
    missing_baseline_rows = [
        [
            row['program'],
            ", ".join(row['present_variants']),
            ", ".join(row['strict_input_variants']),
            ", ".join(row['missing_variants']),
        ]
        for row in audit['strict_time_filter']['missing_strict_baseline_programs'][:top_k]
    ]
    lines.extend(_markdown_table(["Program", "present_variants", "strict_input_variants", "missing_variants"], missing_baseline_rows))
    lines.extend([
        "",
        "### 3.2 缺失 strict O0 baseline 的 run",
        "",
    ])
    missing_baseline_run_rows = [
        [
            row['variant'],
            row['program'],
            _format_float(row['active_window_ratio'], 4),
            _format_float(row['wall_time_sec'], 2),
        ]
        for row in audit['strict_time_filter']['missing_strict_baseline_rows'][:top_k]
    ]
    lines.extend(_markdown_table(["Variant", "Program", "active_window_ratio", "wall_time_sec"], missing_baseline_run_rows))
    lines.extend([
        "",
        "## 4. 过滤后覆盖缺口",
        "",
        f"1. 当前过滤后的 run_features 覆盖 {audit['coverage_gaps']['n_programs']} 个程序，其中完整四变体程序 {audit['coverage_gaps']['n_complete_programs']} 个。",
        f"2. 不完整程序共有 {audit['coverage_gaps']['n_incomplete_programs']} 个，完整名单在 [{audit['artifacts']['json_output_name']}]({json_link}) 的 `coverage_gaps.incomplete_programs`。",
        "",
    ])
    coverage_rows = [
        [
            row['program'],
            ", ".join(row['present_variants']),
            ", ".join(row['missing_variants']),
        ]
        for row in audit['coverage_gaps']['incomplete_programs'][:top_k]
    ]
    lines.extend(_markdown_table(["Program", "present_variants", "missing_variants"], coverage_rows))
    lines.extend([
        "",
        "## 5. O2/O3 难例与 tie 区间",
        "",
        f"1. O2/O3 是当前最难的近邻变体：test `acc_3cls={_format_float(audit['pair_difficulty']['transformer_test_metrics']['O2-O3']['acc_3cls'], 4)}`，`aux_tie_recall={_format_float(audit['pair_difficulty']['transformer_test_metrics']['O2-O3']['aux_tie_recall'], 4)}`。",
        f"2. 当前 O2/O3 全量程序里，repeat-timing 候选 {audit['pair_difficulty']['o2_o3_action_counts'].get('repeat_timing_candidates', 0)} 个，tie/near-tie 阈值候选 {audit['pair_difficulty']['o2_o3_action_counts'].get('tie_threshold_candidates', 0) + audit['pair_difficulty']['o2_o3_action_counts'].get('near_tie_threshold_candidates', 0)} 个。",
        f"3. 完整名单在 [{audit['artifacts']['json_output_name']}]({json_link}) 的 `pair_difficulty.o2_o3_programs`。",
        "",
    ])
    pair_summary_rows = [
        [
            row['pair'],
            str(row['n_programs']),
            _format_float(row['tie_rate'], 4),
            _format_float(row['median_abs_log_ratio'], 4),
            _format_float(row['test_acc_3cls'], 4),
            _format_float(row['test_aux_tie_recall'], 4),
        ]
        for row in audit['pair_difficulty']['pair_summary']
        if row['pair'] in {"O1-O2", "O1-O3", "O2-O3"}
    ]
    lines.extend(_markdown_table(["Pair", "n_programs", "tie_rate", "median_|log_ratio|", "test_acc_3cls", "test_aux_tie_recall"], pair_summary_rows))
    lines.extend([
        "",
        "### 5.1 O2/O3 样本分流建议",
        "",
    ])
    o2_o3_rows = [
        [
            row['program'],
            _format_float(row['abs_log_ratio'], 4),
            row['label_class'],
            row['action_bucket'],
            _format_float(row['o2_active_window_ratio'], 4) if row['o2_active_window_ratio'] is not None else "-",
            _format_float(row['o3_active_window_ratio'], 4) if row['o3_active_window_ratio'] is not None else "-",
        ]
        for row in audit['pair_difficulty']['o2_o3_programs'][:top_k]
    ]
    lines.extend(_markdown_table(["Program", "abs_log_ratio", "label", "action_bucket", "O2_active_ratio", "O3_active_ratio"], o2_o3_rows))
    lines.extend([
        "",
        "## 6. 建议的下一步",
        "",
    ])
    for item in audit['recommendations']:
        lines.append(f"1. {item['priority']} - {item['title']}：{item['why']}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    data_root = pathlib.Path(args.data_root)
    run_features_csv = pathlib.Path(args.run_features_csv)
    pairs_csv = pathlib.Path(args.pairs_csv)
    run_filter_summary_path = pathlib.Path(args.run_filter_summary)
    time_filter_summary_path = pathlib.Path(args.time_filter_summary)
    pairs_stats_path = pathlib.Path(args.pairs_stats)
    anchor_stats_path = pathlib.Path(args.anchor_stats)
    transformer_eval_path = pathlib.Path(args.transformer_eval)
    score_eval_path = pathlib.Path(args.score_eval)
    score_time_eval_path = pathlib.Path(args.score_time_eval)
    manifest_summary_path = pathlib.Path(args.manifest_summary)
    json_output_path = pathlib.Path(args.json_output)
    markdown_output_path = pathlib.Path(args.markdown_output)

    manifest_summary = _load_json(manifest_summary_path)
    run_filter_summary = _load_json(run_filter_summary_path)
    time_filter_summary = _load_json(time_filter_summary_path)
    pairs_stats = _load_json(pairs_stats_path)
    anchor_stats = _load_json(anchor_stats_path)
    transformer_eval = _load_json(transformer_eval_path)
    score_eval = _load_json(score_eval_path)
    score_time_eval = _load_json(score_time_eval_path)

    semantic_filter = _compute_semantic_filter_audit(
        data_root=data_root,
        manifest_prefix=args.manifest_prefix,
        min_active_pids=args.min_active_pids,
        min_cycles_per_iter=args.min_cycles_per_iter,
    )

    run_features_rows = [
        _normalize_run_feature_row(
            row,
            min_active_pids=args.min_active_pids,
            min_active_window_ratio=args.min_active_window_ratio,
        )
        for row in _load_csv_rows(run_features_csv)
    ]
    run_feature_index = {
        (row["program"], row["variant"]): row
        for row in run_features_rows
    }

    strict_time_filter = _compute_time_filter_audit(
        run_feature_rows=run_features_rows,
        baseline=args.baseline,
    )
    coverage_gaps = _compute_coverage_gaps(run_features_rows)

    pair_rows = _load_csv_rows(pairs_csv)
    pair_summary, o2_o3_rows, o2_o3_category_counts = _build_pair_summary(
        pair_rows=pair_rows,
        transformer_eval=transformer_eval,
        tie_threshold=args.tie_threshold,
        near_tie_threshold=args.near_tie_threshold,
    )
    o2_o3_programs = _attach_o2_o3_run_context(
        o2_o3_rows=o2_o3_rows,
        run_feature_index=run_feature_index,
        min_active_window_ratio=args.min_active_window_ratio,
    )

    semantic_consistency = {
        "matches_recorded_summary": (
            semantic_filter["n_seen"] == run_filter_summary.get("n_seen")
            and semantic_filter["n_kept"] == run_filter_summary.get("n_kept")
            and semantic_filter["n_filtered"] == run_filter_summary.get("n_filtered")
        ),
        "computed": {
            "n_seen": semantic_filter["n_seen"],
            "n_kept": semantic_filter["n_kept"],
            "n_filtered": semantic_filter["n_filtered"],
        },
        "recorded": {
            "n_seen": run_filter_summary.get("n_seen"),
            "n_kept": run_filter_summary.get("n_kept"),
            "n_filtered": run_filter_summary.get("n_filtered"),
        },
    }
    strict_time_consistency = {
        "matches_recorded_summary": (
            strict_time_filter["n_input_filtered"] == time_filter_summary.get("n_input_filtered")
            and strict_time_filter["n_missing_strict_baseline_rows"]
            == time_filter_summary.get("reasons", {}).get("missing_strict_baseline", 0)
        ),
        "computed": {
            "n_input_filtered": strict_time_filter["n_input_filtered"],
            "n_missing_strict_baseline_rows": strict_time_filter["n_missing_strict_baseline_rows"],
            "n_missing_strict_baseline_programs": strict_time_filter["n_missing_strict_baseline_programs"],
        },
        "recorded": {
            "n_input_filtered": time_filter_summary.get("n_input_filtered"),
            "n_missing_strict_baseline_rows": time_filter_summary.get("reasons", {}).get("missing_strict_baseline", 0),
        },
    }

    pair_test_metrics = {
        pair: transformer_eval.get("per_pair", {}).get(pair, {})
        for pair in ["O1-O2", "O1-O3", "O2-O3"]
    }

    audit = {
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "artifacts": {
            "json_output": _relative_to_repo(json_output_path),
            "json_output_name": json_output_path.name,
            "markdown_output": _relative_to_repo(markdown_output_path),
        },
        "manifest_snapshot": {
            "manifest_prefix": args.manifest_prefix,
            "expected_program_count": manifest_summary.get("expected_program_count"),
            "shared_program_count": manifest_summary.get("shared_program_count"),
            "variants": manifest_summary.get("variants", {}),
        },
        "train_snapshot": {
            "n_runs": len(run_features_rows),
            "n_pairs": pairs_stats.get("n_pairs"),
            "n_programs_in_pairs": pairs_stats.get("n_programs"),
            "n_anchors": anchor_stats.get("n_anchors"),
            "anchors_by_variant": anchor_stats.get("anchors_by_variant"),
            "score_eval": score_eval,
            "score_time_eval": score_time_eval,
            "transformer_test": transformer_eval.get("results", {}).get("test", {}),
        },
        "semantic_filter": {
            **semantic_filter,
            "recorded_summary": run_filter_summary,
            "consistency": semantic_consistency,
        },
        "strict_time_filter": {
            **strict_time_filter,
            "recorded_summary": time_filter_summary,
            "consistency": strict_time_consistency,
        },
        "coverage_gaps": coverage_gaps,
        "pair_difficulty": {
            "label_counts": pairs_stats.get("label_counts"),
            "pair_summary": pair_summary,
            "o2_o3_action_counts": dict(Counter(row["action_bucket"] for row in o2_o3_programs)),
            "o2_o3_initial_bucket_counts": o2_o3_category_counts,
            "o2_o3_programs": o2_o3_programs,
            "transformer_test_metrics": pair_test_metrics,
        },
    }
    audit["recommendations"] = _build_recommendations(
        semantic_filter=semantic_filter,
        time_filter=strict_time_filter,
        coverage_gaps=coverage_gaps,
        o2_o3_rows=o2_o3_programs,
    )

    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    markdown_output_path.write_text(_build_markdown(audit, markdown_output_path, top_k=args.top_k) + "\n")

    print(f"[ok] wrote {_relative_to_repo(json_output_path)}")
    print(f"[ok] wrote {_relative_to_repo(markdown_output_path)}")
    print(json.dumps({
        "semantic_filter": semantic_consistency,
        "strict_time_filter": strict_time_consistency,
        "coverage_gaps": {
            "n_complete_programs": coverage_gaps["n_complete_programs"],
            "n_incomplete_programs": coverage_gaps["n_incomplete_programs"],
        },
        "o2_o3_action_counts": audit["pair_difficulty"]["o2_o3_action_counts"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()