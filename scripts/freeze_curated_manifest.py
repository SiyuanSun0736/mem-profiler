#!/usr/bin/env python3
"""
freeze_curated_manifest.py — 从最新 raw manifest 冻结可复现的 145x4 curated run list

选择规则：
  1. 按 variant 读取 manifest_bcc_<VARIANT>.jsonl
  2. 对每个 program，只保留 output_dir 时间戳最新的一条记录
  3. 仅接受同时存在 run_metadata.jsonl 和 window_metrics.jsonl 的 run
  4. 要求四个 variant 选出的 program 集合完全一致

输出：
  - data/llvm_test_suite/manifest_curated_O0.jsonl ... O3.jsonl
  - data/llvm_test_suite/manifest_curated_summary.json

用法：
    python scripts/freeze_curated_manifest.py
    python scripts/freeze_curated_manifest.py \
        --data-root data/llvm_test_suite \
        --input-prefix manifest_bcc \
        --output-prefix manifest_curated
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
VARIANTS = ["O0", "O1", "O2", "O3"]
REQUIRED_FILES = ("run_metadata.jsonl", "window_metrics.jsonl")
TIMESTAMP_RE = re.compile(r"_(\d{8}_\d{6})$")


def _load_manifest(manifest_path: pathlib.Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in manifest_path.read_text().splitlines()
        if line.strip()
    ]


def _relative_to_repo(path: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _extract_timestamp(output_dir: str) -> str | None:
    leaf = pathlib.PurePosixPath(output_dir).name
    match = TIMESTAMP_RE.search(leaf)
    if match is None:
        return None
    return match.group(1)


def _missing_required_files(output_dir: str) -> list[str]:
    run_dir = REPO_ROOT / output_dir
    return [name for name in REQUIRED_FILES if not (run_dir / name).is_file()]


def _select_curated_entries(entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    best_by_program: dict[str, tuple[tuple[str, int, str], dict[str, Any]]] = {}
    stats = {
        "manifest_records": len(entries),
        "valid_candidates": 0,
        "selected_records": 0,
        "dropped_records": 0,
        "missing_output_dir": 0,
        "invalid_timestamp": 0,
        "missing_required_files": 0,
    }

    for index, entry in enumerate(entries):
        program = str(entry.get("program", "")).strip()
        output_dir = str(entry.get("output_dir", "")).strip()

        if not program or not output_dir:
            stats["missing_output_dir"] += 1
            continue

        timestamp = _extract_timestamp(output_dir)
        if timestamp is None:
            stats["invalid_timestamp"] += 1
            continue

        missing_files = _missing_required_files(output_dir)
        if missing_files:
            stats["missing_required_files"] += 1
            continue

        stats["valid_candidates"] += 1
        sort_key = (timestamp, index, output_dir)
        current = best_by_program.get(program)
        if current is None or sort_key > current[0]:
            best_by_program[program] = (sort_key, entry)

    selected = [best_by_program[program][1] for program in sorted(best_by_program)]
    stats["selected_records"] = len(selected)
    stats["dropped_records"] = stats["valid_candidates"] - stats["selected_records"]
    return selected, stats


def _write_manifest(path: pathlib.Path, entries: list[dict[str, Any]]) -> None:
    content = "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n"
    path.write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 raw manifests 冻结可复现的 145x4 curated run list"
    )
    parser.add_argument(
        "--data-root",
        default="data/llvm_test_suite",
        help="数据根目录，默认 data/llvm_test_suite",
    )
    parser.add_argument(
        "--input-prefix",
        default="manifest_bcc",
        help="输入 manifest 前缀，默认 manifest_bcc",
    )
    parser.add_argument(
        "--output-prefix",
        default="manifest_curated",
        help="输出 manifest 前缀，默认 manifest_curated",
    )
    parser.add_argument(
        "--summary-name",
        default="manifest_curated_summary.json",
        help="汇总文件名，默认 manifest_curated_summary.json",
    )
    parser.add_argument(
        "--expected-program-count",
        type=int,
        default=145,
        help="期望每个 variant 的 program 数，默认 145",
    )
    args = parser.parse_args()

    data_root = (REPO_ROOT / args.data_root).resolve()
    if not data_root.exists():
        sys.exit(f"[错误] 数据目录不存在: {data_root}")

    selected_by_variant: dict[str, list[dict[str, Any]]] = {}
    program_sets: dict[str, set[str]] = {}
    summary: dict[str, Any] = {
        "data_root": _relative_to_repo(data_root),
        "input_prefix": args.input_prefix,
        "output_prefix": args.output_prefix,
        "selection_rule": "latest_complete_output_dir_per_program_variant",
        "required_files": list(REQUIRED_FILES),
        "expected_program_count": args.expected_program_count,
        "variants": {},
    }

    for variant in VARIANTS:
        manifest_path = data_root / f"{args.input_prefix}_{variant}.jsonl"
        if not manifest_path.exists():
            sys.exit(f"[错误] manifest 不存在: {manifest_path}")

        entries = _load_manifest(manifest_path)
        selected, stats = _select_curated_entries(entries)
        programs = {str(entry["program"]) for entry in selected}

        if len(programs) != args.expected_program_count:
            sys.exit(
                f"[错误] {variant} 冻结后 program 数为 {len(programs)}，"
                f"与期望 {args.expected_program_count} 不一致"
            )

        output_manifest = data_root / f"{args.output_prefix}_{variant}.jsonl"
        stats["output_manifest"] = _relative_to_repo(output_manifest)
        summary["variants"][variant] = stats
        selected_by_variant[variant] = selected
        program_sets[variant] = programs

    baseline_variant = VARIANTS[0]
    baseline_programs = program_sets[baseline_variant]
    for variant in VARIANTS[1:]:
        if program_sets[variant] != baseline_programs:
            only_left = sorted(baseline_programs - program_sets[variant])
            only_right = sorted(program_sets[variant] - baseline_programs)
            sys.exit(
                "[错误] curated run list 的 program 集合不一致: "
                f"{baseline_variant} only={only_left[:5]}, {variant} only={only_right[:5]}"
            )

    summary["shared_program_count"] = len(baseline_programs)

    for variant in VARIANTS:
        output_manifest = data_root / f"{args.output_prefix}_{variant}.jsonl"
        _write_manifest(output_manifest, selected_by_variant[variant])
        print(
            f"[ok] {variant}: {len(selected_by_variant[variant])} 条 -> "
            f"{_relative_to_repo(output_manifest)}",
            flush=True,
        )

    summary_path = data_root / args.summary_name
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"[ok] summary -> {_relative_to_repo(summary_path)}", flush=True)


if __name__ == "__main__":
    main()