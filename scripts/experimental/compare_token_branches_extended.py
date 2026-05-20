#!/usr/bin/env python3
"""
Aggregate four token-branch comparisons into a single side-by-side report.

Inputs are the baseline comparison JSONs produced by:
  - category-token branch
  - hotspot-token branch
  - fixed-work-token branch
  - fixed-work-token pooled branch
"""

from __future__ import annotations

import argparse
import json
import pathlib
from datetime import datetime, timezone
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _series_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return None if isinstance(value, float) and value != value else float(value)
    return float(value)


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _report_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate four token experiment branches into one report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--category-json", default="train_set/category_token_transformer/category_vs_baseline_comparison.json")
    parser.add_argument("--hotspot-json", default="train_set/hotspot_token_transformer/hotspot_vs_baseline_comparison.json")
    parser.add_argument("--fixed-work-json", default="train_set/fixed_work_token_transformer/fixed_work_vs_baseline_comparison.json")
    parser.add_argument("--fixed-work-pooled-json", default="train_set/fixed_work_token_pooled_transformer/fixed_work_pooled_vs_baseline_comparison.json")
    parser.add_argument("--output-dir", default="train_set/token_branch_comparison_extended")
    args = parser.parse_args()

    category_json = (REPO_ROOT / args.category_json).resolve()
    hotspot_json = (REPO_ROOT / args.hotspot_json).resolve()
    fixed_work_json = (REPO_ROOT / args.fixed_work_json).resolve()
    fixed_work_pooled_json = (REPO_ROOT / args.fixed_work_pooled_json).resolve()
    out_dir = (REPO_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    payloads = {
        "category": _load_json(category_json),
        "hotspot": _load_json(hotspot_json),
        "fixed_work": _load_json(fixed_work_json),
        "fixed_work_pooled": _load_json(fixed_work_pooled_json),
    }
    labels = {
        "category": "CategoryToken",
        "hotspot": "HotspotToken",
        "fixed_work": "FixedWorkToken",
        "fixed_work_pooled": "FixedWorkPooled",
    }

    overall_rows = []
    for key in ("category", "hotspot", "fixed_work", "fixed_work_pooled"):
        record = payloads[key]["overall_test"]
        overall_rows.append([
            labels[key],
            _fmt(_series_float(record["experiment"].get("mae"))),
            _fmt(_series_float(record["delta"].get("mae"))),
            _fmt(_series_float(record["experiment"].get("dir_acc"))),
            _fmt(_series_float(record["delta"].get("dir_acc"))),
            _fmt(_series_float(record["experiment"].get("acc_3cls"))),
            _fmt(_series_float(record["delta"].get("acc_3cls"))),
            _fmt(_series_float(record["experiment"].get("aux_tie_recall"))),
            _fmt(_series_float(record["delta"].get("aux_tie_recall"))),
        ])

    slice_rows = []
    for key in ("category", "hotspot", "fixed_work", "fixed_work_pooled"):
        focused = payloads[key]["focused_slices"]
        slice_rows.append([
            labels[key],
            _fmt(_series_float(focused["near_tie"]["experiment"].get("dir_acc"))),
            _fmt(_series_float(focused["near_tie"]["delta"].get("dir_acc"))),
            _fmt(_series_float(focused["O2-O3"]["experiment"].get("dir_acc"))),
            _fmt(_series_float(focused["O2-O3"]["delta"].get("dir_acc"))),
            _fmt(_series_float(focused["tie"]["experiment"].get("aux_tie_recall"))),
            _fmt(_series_float(focused["tie"]["delta"].get("aux_tie_recall"))),
        ])

    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "inputs": {
            "category": str(category_json.relative_to(REPO_ROOT)),
            "hotspot": str(hotspot_json.relative_to(REPO_ROOT)),
            "fixed_work": str(fixed_work_json.relative_to(REPO_ROOT)),
            "fixed_work_pooled": str(fixed_work_pooled_json.relative_to(REPO_ROOT)),
        },
        "overall": overall_rows,
        "focused": slice_rows,
    }
    json_path = out_dir / "token_branch_comparison_extended.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    lines = [
        "# Token Branch Comparison Extended",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Inputs",
        "",
        f"- category: {summary['inputs']['category']}",
        f"- hotspot: {summary['inputs']['hotspot']}",
        f"- fixed_work: {summary['inputs']['fixed_work']}",
        f"- fixed_work_pooled: {summary['inputs']['fixed_work_pooled']}",
        "",
        "## Overall Test",
        "",
        *_report_table(["Branch", "MAE", "Delta", "dir_acc", "Delta", "acc_3cls", "Delta", "tie_rec", "Delta"], overall_rows),
        "",
        "## Focused Slices",
        "",
        *_report_table(["Branch", "near_dir", "Delta", "O2-O3 dir", "Delta", "tie_rec", "Delta"], slice_rows),
    ]

    md_path = out_dir / "token_branch_comparison_extended.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"[ok] comparison json: {json_path}")
    print(f"[ok] comparison markdown: {md_path}")


if __name__ == "__main__":
    main()