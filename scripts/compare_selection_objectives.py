#!/usr/bin/env python3
"""
compare_selection_objectives.py — score-first vs time-first 对比汇总
===================================================================

目标
----
1. 分别回放 `score_program.py` 的 score-first / time-first tuned 参数。
2. 对两套结果分别执行 `evaluate_score_vs_time.py`。
3. 生成一份机器可读 JSON 和一页 Markdown 对比表，便于选择默认口径。

输出
----
  train_set/objective_compare/
    scores_score_first.parquet
    score_eval_score_first.json
    score_time_eval_score_first.json
    scores_time_first.parquet
    score_eval_time_first.json
    score_time_eval_time_first.json

  train_set/score_selection_objective_comparison.json
  docs/new-repo-plan/08-score-selection-objective-comparison.md

用法
----
  python scripts/compare_selection_objectives.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from score_program import _is_reliable_tuned_best  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="并排回放 score-first 与 time-first 两套评分口径，并生成对比页",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default=None, help="透传给 score_program.py 的 --device")
    parser.add_argument(
        "--compare-dir",
        default="train_set/objective_compare",
        help="两套中间产物输出目录",
    )
    parser.add_argument(
        "--summary-json",
        default="train_set/score_selection_objective_comparison.json",
        help="最终对比 JSON 输出路径",
    )
    parser.add_argument(
        "--markdown",
        default="docs/new-repo-plan/08-score-selection-objective-comparison.md",
        help="最终 Markdown 对比页输出路径",
    )
    parser.add_argument(
        "--tuned-defaults-json",
        default="train_set/score_tune_fine_variant_best.json",
        help="评分层 fine tune 结果 JSON",
    )
    return parser.parse_args()


def _run_command(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return "NaN"
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_delta(value: float | None, better: str) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    sign = "+" if value >= 0 else ""
    suffix = " 更好" if (better == "higher" and value > 0) or (better == "lower" and value < 0) else ""
    return f"{sign}{value:.4f}{suffix}"


def _metric_row(name: str, score_value: float, time_value: float, better: str) -> dict[str, Any]:
    delta = float(score_value) - float(time_value)
    preferred = "score-first" if (delta > 0 and better == "higher") or (delta < 0 and better == "lower") else "time-first"
    if abs(delta) < 1e-12:
        preferred = "tie"
    return {
        "metric": name,
        "score_first": float(score_value),
        "time_first": float(time_value),
        "delta_score_minus_time": delta,
        "better": better,
        "preferred": preferred,
    }


def _build_reliability_table(tuned_data: dict[str, Any]) -> list[dict[str, Any]]:
    variants = [str(v) for v in tuned_data.get("variants", [])]
    tables: list[dict[str, Any]] = []
    for objective_key, section_key in (
        ("score_first", "best_for_score_by_variant"),
        ("time_first", "best_by_variant"),
    ):
        section = tuned_data.get(section_key, {})
        shared_best = (section.get("ALL") or {}).get("best")
        for variant in variants:
            payload = section.get(variant) or {}
            best = payload.get("best")
            if not isinstance(best, dict):
                continue
            reliable, reason = _is_reliable_tuned_best(best, shared_best)
            tables.append(
                {
                    "objective": objective_key,
                    "variant": variant,
                    "reliable": bool(reliable),
                    "reason": reason,
                    "n_score_valid": int(best.get("n_score_valid", 0) or 0),
                    "n_time_valid": int(best.get("n_time_valid", 0) or 0),
                    "score_corr": float(best["score_corr"]) if isinstance(best.get("score_corr"), (int, float)) and math.isfinite(float(best["score_corr"])) else None,
                    "time_corr": float(best["time_corr"]) if isinstance(best.get("time_corr"), (int, float)) and math.isfinite(float(best["time_corr"])) else None,
                    "tie_gate_threshold": float(best.get("tie_gate_threshold", 0.0) or 0.0),
                    "tie_shrink_power": float(best.get("tie_shrink_power", 0.0) or 0.0),
                    "tie_margin_weight_alpha": float(best.get("tie_margin_weight_alpha", 0.0) or 0.0),
                }
            )
    return tables


def _recommend_default(score_eval: dict[str, Any], score_time_eval: dict[str, Any], time_eval: dict[str, Any], time_time_eval: dict[str, Any]) -> dict[str, Any]:
    score_corr_gain = float(score_eval["corr_score_log"]) - float(time_eval["corr_score_log"])
    score_mae_gain = float(time_eval["mae_score_log"]) - float(score_eval["mae_score_log"])
    time_corr_gain = float(score_time_eval["corr_model_time"]) - float(time_time_eval["corr_model_time"])
    time_spearman_gain = float(score_time_eval["spearman_model"]) - float(time_time_eval["spearman_model"])

    prefer_score = (
        score_corr_gain > 0.0
        and score_mae_gain > 0.0
        and time_corr_gain > -0.01
        and time_spearman_gain > -0.01
    )
    if prefer_score:
        return {
            "recommended_default": "score-first",
            "reason": [
                f"proxy Pearson r 提升 {score_corr_gain:+.4f}",
                f"proxy MAE 改善 {score_mae_gain:+.4f}",
                f"strict time Pearson r 仅变化 {time_corr_gain:+.4f}",
                f"strict time Spearman 仅变化 {time_spearman_gain:+.4f}",
            ],
        }
    return {
        "recommended_default": "time-first",
        "reason": [
            f"strict time Pearson r 变化 {time_corr_gain:+.4f}",
            f"strict time Spearman 变化 {time_spearman_gain:+.4f}",
            f"proxy Pearson r 变化 {score_corr_gain:+.4f}",
            f"proxy MAE 改善 {score_mae_gain:+.4f}",
        ],
    }


def _write_markdown(
    path: pathlib.Path,
    summary: dict[str, Any],
    tuned_data: dict[str, Any],
    reliability_rows: list[dict[str, Any]],
) -> None:
    score_eval = summary["objectives"]["score_first"]["score_eval"]
    time_eval = summary["objectives"]["time_first"]["score_eval"]
    score_time_eval = summary["objectives"]["score_first"]["score_time_eval"]
    time_time_eval = summary["objectives"]["time_first"]["score_time_eval"]
    recommendation = summary["recommendation"]

    score_all = ((tuned_data.get("best_for_score_by_variant") or {}).get("ALL") or {}).get("best", {})
    time_all = ((tuned_data.get("best_by_variant") or {}).get("ALL") or {}).get("best", {})

    lines: list[str] = []
    lines.append("# score-first vs time-first 默认口径对比")
    lines.append("")
    lines.append(f"> 生成时间：{summary['generated_at']}  ")
    lines.append("> 生成脚本：scripts/compare_selection_objectives.py")
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append(f"当前建议默认口径：**{recommendation['recommended_default']}**。")
    lines.append("")
    for item in recommendation["reason"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 一页总表")
    lines.append("")
    lines.append("| 指标 | score-first | time-first | score-first - time-first | 更优口径 |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for row in summary["metrics"]:
        lines.append(
            f"| {row['metric']} | {_fmt_num(row['score_first'])} | {_fmt_num(row['time_first'])} | {_fmt_delta(row['delta_score_minus_time'], row['better'])} | {row['preferred']} |"
        )
    lines.append("")
    lines.append("## ALL 共享参数对比")
    lines.append("")
    lines.append("| 参数 | score-first | time-first |")
    lines.append("| --- | ---: | ---: |")
    for key in (
        "tie_gate_threshold",
        "tie_shrink_power",
        "tie_margin_weight_alpha",
        "min_anchor_quality",
        "anchor_outlier_mad_scale",
        "anchor_outlier_min_delta",
    ):
        lines.append(
            f"| {key} | {_fmt_num(score_all.get(key), 2)} | {_fmt_num(time_all.get(key), 2)} |"
        )
    lines.append("")
    lines.append("## Variant-local tuned 可靠性")
    lines.append("")
    lines.append("| 口径 | variant | reliable | reason | n_score_valid | n_time_valid | score_corr | time_corr | gate | shrink | alpha |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in reliability_rows:
        lines.append(
            "| {objective} | {variant} | {reliable} | {reason} | {n_score_valid} | {n_time_valid} | {score_corr} | {time_corr} | {gate} | {shrink} | {alpha} |".format(
                objective=row["objective"],
                variant=row["variant"],
                reliable="yes" if row["reliable"] else "no",
                reason=row["reason"],
                n_score_valid=row["n_score_valid"],
                n_time_valid=row["n_time_valid"],
                score_corr=_fmt_num(row["score_corr"]),
                time_corr=_fmt_num(row["time_corr"]),
                gate=_fmt_num(row["tie_gate_threshold"], 2),
                shrink=_fmt_num(row["tie_shrink_power"], 2),
                alpha=_fmt_num(row["tie_margin_weight_alpha"], 2),
            )
        )
    lines.append("")
    lines.append("## 解释")
    lines.append("")
    lines.append(
        "score-first 看的是单程序评分对 proxy 真值的恢复能力；time-first 看的是 strict 时间外部验证。当前这两套口径的时间指标差距很小，但 score-first 在 proxy 侧更稳，因此默认更适合作为主线口径。"
    )
    lines.append("")
    lines.append("## 复现命令")
    lines.append("")
    lines.append("```bash")
    lines.append("/home/ssy/mem-profiler/.venv/bin/python scripts/compare_selection_objectives.py --device cpu")
    lines.append("```")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    compare_dir = (REPO_ROOT / args.compare_dir).resolve()
    summary_json_path = (REPO_ROOT / args.summary_json).resolve()
    markdown_path = (REPO_ROOT / args.markdown).resolve()
    tuned_json_path = (REPO_ROOT / args.tuned_defaults_json).resolve()

    compare_dir.mkdir(parents=True, exist_ok=True)

    objectives = {
        "score_first": {
            "selection": "score",
            "scores": compare_dir / "scores_score_first.parquet",
            "score_eval": compare_dir / "score_eval_score_first.json",
            "score_time_eval": compare_dir / "score_time_eval_score_first.json",
        },
        "time_first": {
            "selection": "time",
            "scores": compare_dir / "scores_time_first.parquet",
            "score_eval": compare_dir / "score_eval_time_first.json",
            "score_time_eval": compare_dir / "score_time_eval_time_first.json",
        },
    }

    for payload in objectives.values():
        score_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "score_program.py"),
            "--tuned-selection-objective",
            payload["selection"],
            "--output",
            str(payload["scores"].relative_to(REPO_ROOT)),
            "--eval-output",
            str(payload["score_eval"].relative_to(REPO_ROOT)),
        ]
        if args.device:
            score_cmd.extend(["--device", args.device])
        _run_command(score_cmd)

        time_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "evaluate_score_vs_time.py"),
            "--scores",
            str(payload["scores"].relative_to(REPO_ROOT)),
            "--output",
            str(payload["score_time_eval"].relative_to(REPO_ROOT)),
        ]
        _run_command(time_cmd)

    tuned_data = _load_json(tuned_json_path)
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "objectives": {},
    }
    for name, payload in objectives.items():
        summary["objectives"][name] = {
            "selection": payload["selection"],
            "scores": str(payload["scores"].relative_to(REPO_ROOT)),
            "score_eval": _load_json(payload["score_eval"]),
            "score_time_eval": _load_json(payload["score_time_eval"]),
        }

    score_eval = summary["objectives"]["score_first"]["score_eval"]
    time_eval = summary["objectives"]["time_first"]["score_eval"]
    score_time_eval = summary["objectives"]["score_first"]["score_time_eval"]
    time_time_eval = summary["objectives"]["time_first"]["score_time_eval"]

    summary["metrics"] = [
        _metric_row("proxy.corr_score_log", score_eval["corr_score_log"], time_eval["corr_score_log"], "higher"),
        _metric_row("proxy.mae_score_log", score_eval["mae_score_log"], time_eval["mae_score_log"], "lower"),
        _metric_row("proxy.dir_accuracy", score_eval["dir_accuracy"], time_eval["dir_accuracy"], "higher"),
        _metric_row("proxy.band_accuracy", score_eval["band_accuracy"], time_eval["band_accuracy"], "higher"),
        _metric_row("time.corr_model_time", score_time_eval["corr_model_time"], time_time_eval["corr_model_time"], "higher"),
        _metric_row("time.spearman_model", score_time_eval["spearman_model"], time_time_eval["spearman_model"], "higher"),
        _metric_row("time.mae_model_time", score_time_eval["mae_model_time"], time_time_eval["mae_model_time"], "lower"),
        _metric_row("time.dir_acc_model", score_time_eval["dir_acc_model"], time_time_eval["dir_acc_model"], "higher"),
        _metric_row("time.band_acc_model", score_time_eval["band_acc_model"], time_time_eval["band_acc_model"], "higher"),
        _metric_row("coverage.n_valid_strict", score_time_eval["n_valid_strict"], time_time_eval["n_valid_strict"], "higher"),
    ]

    summary["recommendation"] = _recommend_default(score_eval, score_time_eval, time_eval, time_time_eval)
    reliability_rows = _build_reliability_table(tuned_data)
    summary["variant_reliability"] = reliability_rows

    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(markdown_path, summary, tuned_data, reliability_rows)

    print(f"[done] 对比 JSON: {summary_json_path}")
    print(f"[done] 对比页:   {markdown_path}")


if __name__ == "__main__":
    main()