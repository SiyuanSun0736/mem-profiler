#!/usr/bin/env python3
"""Deduplicate collected llvm-test-suite outputs for one variant and rebuild the manifest."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import sys
from collections import defaultdict
from typing import Any


REQUIRED_FILES = ("run_metadata.jsonl", "window_metrics.jsonl")
RUN_DIR_RE = re.compile(r"^(?P<bench>.+)_(?P<stamp>\d{8}_\d{6})(?:_retry(?P<retry>\d+))?$")


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _relative_to_project(path: pathlib.Path, project_root: pathlib.Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(project_root.resolve()))
    except ValueError:
        return str(resolved)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_run_stats(run_dir: pathlib.Path) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "completion_count": 0,
        "window_sec": None,
        "sample_rate": None,
    }

    for record in _load_jsonl(run_dir / "run_metadata.jsonl"):
        if stats["window_sec"] is None and "window_sec" in record:
            stats["window_sec"] = _coerce_float(record.get("window_sec"), 0.0)
        if stats["sample_rate"] is None and "sample_rate" in record:
            stats["sample_rate"] = _coerce_int(record.get("sample_rate"), 0)
        if record.get("_record_type") == "run_stats":
            stats["completion_count"] = _coerce_int(record.get("completion_count"), 0)

    return stats


def _compute_quality(run_dir: pathlib.Path) -> dict[str, Any]:
    match = RUN_DIR_RE.match(run_dir.name)
    if match is None:
        raise ValueError(f"invalid run directory name: {run_dir.name}")

    total_samples = 0
    window_lines = 0
    for record in _load_jsonl(run_dir / "window_metrics.jsonl"):
        total_samples += _coerce_int(record.get("samples"), 0)
        window_lines += 1

    run_stats = _extract_run_stats(run_dir)
    valid = all((run_dir / name).is_file() for name in REQUIRED_FILES)
    return {
        "program": match.group("bench"),
        "timestamp": match.group("stamp"),
        "retry": _coerce_int(match.group("retry"), 0),
        "run_dir": run_dir,
        "valid": valid,
        "total_samples": total_samples,
        "window_lines": window_lines,
        "completion_count": _coerce_int(run_stats.get("completion_count"), 0),
        "window_sec": run_stats.get("window_sec"),
        "sample_rate": run_stats.get("sample_rate"),
    }


def _quality_key(info: dict[str, Any]) -> tuple[Any, ...]:
    return (
        1 if info["valid"] else 0,
        info["total_samples"],
        info["window_lines"],
        info["completion_count"],
        info["timestamp"],
        info["retry"],
        info["run_dir"].name,
    )


def _first_test_file(test_subdir: pathlib.Path) -> pathlib.Path | None:
    candidates = sorted(test_subdir.glob("*.test"))
    if not candidates:
        return None
    return candidates[0]


def _parse_run_cmd(test_file: pathlib.Path, binary: pathlib.Path, test_data: pathlib.Path) -> str:
    run_raw = ""
    for line in test_file.read_text().splitlines():
        if line.startswith("RUN:"):
            run_raw = line[len("RUN:") :].strip()
            break

    if not run_raw:
        raise ValueError(f"no RUN line in {test_file}")

    cmd_part = run_raw
    if run_raw.startswith("cd %S ;"):
        cmd_part = run_raw[len("cd %S ;") :].strip()

    first_word = cmd_part.split(maxsplit=1)[0] if cmd_part.split() else ""
    if first_word.startswith("%S/"):
        bin_ref_name = first_word[len("%S/") :]
        cmd_part = cmd_part.replace(f"%S/{bin_ref_name}", str(binary.resolve()))

    return cmd_part.replace("%S", str(test_data.resolve()))


def _build_manifest_entry(
    info: dict[str, Any],
    variant: str,
    project_root: pathlib.Path,
    bin_dir: pathlib.Path,
    test_dir: pathlib.Path,
    by_run_dir: dict[str, dict[str, Any]],
    by_program: dict[str, dict[str, Any]],
    fallback_window_sec: float,
    fallback_duration_sec: float,
    fallback_sample_rate: int,
) -> dict[str, Any]:
    program = str(info["program"])
    run_dir = pathlib.Path(info["run_dir"])
    template = by_run_dir.get(run_dir.name) or by_program.get(program, {})

    binary_path = bin_dir / f"{program}_{variant}"
    test_subdir = test_dir / program
    test_file = _first_test_file(test_subdir)

    binary_value = _relative_to_project(binary_path, project_root)
    test_file_value = str(template.get("test_file", "")).strip()
    run_cmd_value = str(template.get("run_cmd", "")).strip()
    target_comm_value = str(template.get("target_comm", "")).strip()

    if test_file is not None:
        test_file_value = _relative_to_project(test_file, project_root)
        run_cmd_value = _parse_run_cmd(test_file, binary_path, test_subdir)
    elif not test_file_value:
        raise FileNotFoundError(f"missing .test file for program={program} in {test_subdir}")

    if not target_comm_value:
        target_comm_value = binary_path.name[:15]

    window_sec_value = info["window_sec"]
    if window_sec_value is None:
        window_sec_value = template.get("window_sec")
    sample_rate_value = info["sample_rate"]
    if sample_rate_value is None:
        sample_rate_value = template.get("sample_rate")

    return {
        "program": program,
        "variant": variant,
        "binary": binary_value,
        "test_file": test_file_value,
        "run_cmd": run_cmd_value,
        "target_comm": target_comm_value,
        "output_dir": _relative_to_project(run_dir, project_root),
        "window_sec": _coerce_float(window_sec_value, fallback_window_sec),
        "duration_sec": _coerce_float(template.get("duration_sec"), fallback_duration_sec),
        "sample_rate": _coerce_int(sample_rate_value, fallback_sample_rate),
        "completion_count": _coerce_int(info["completion_count"], 0),
    }


def _write_manifest(path: pathlib.Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries)
    path.write_text(content + ("\n" if content else ""))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deduplicate collected outputs for one llvm-test-suite variant"
    )
    parser.add_argument("--variant", required=True, help="variant name, e.g. O0")
    parser.add_argument("--project-root", required=True, help="repository root")
    parser.add_argument("--output-root", required=True, help="variant output root")
    parser.add_argument("--manifest", required=True, help="manifest path to rewrite")
    parser.add_argument("--bin-dir", required=True, help="compiled benchmark directory")
    parser.add_argument("--test-dir", required=True, help="benchmark .test directory")
    parser.add_argument("--window-sec", type=float, default=1.0, help="fallback window size")
    parser.add_argument("--duration-sec", type=float, default=60.0, help="fallback duration")
    parser.add_argument("--sample-rate", type=int, default=100, help="fallback sample rate")
    parser.add_argument("--dry-run", action="store_true", help="report only")
    args = parser.parse_args()

    project_root = pathlib.Path(args.project_root).resolve()
    output_root = pathlib.Path(args.output_root).resolve()
    manifest_path = pathlib.Path(args.manifest).resolve()
    bin_dir = pathlib.Path(args.bin_dir).resolve()
    test_dir = pathlib.Path(args.test_dir).resolve()

    if not output_root.is_dir():
        sys.exit(f"[错误] 输出目录不存在: {output_root}")

    existing_entries = _load_jsonl(manifest_path)
    by_run_dir: dict[str, dict[str, Any]] = {}
    by_program: dict[str, dict[str, Any]] = {}
    for entry in existing_entries:
        output_dir = str(entry.get("output_dir", "")).strip()
        if output_dir:
            by_run_dir[pathlib.PurePosixPath(output_dir).name] = entry
        program = str(entry.get("program", "")).strip()
        if program:
            by_program[program] = entry

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for child in sorted(output_root.iterdir()):
        if not child.is_dir():
            continue
        if RUN_DIR_RE.match(child.name) is None:
            continue
        info = _compute_quality(child)
        grouped[str(info["program"])].append(info)

    selected: list[dict[str, Any]] = []
    remove_dirs: list[pathlib.Path] = []
    skipped_invalid = 0

    for program in sorted(grouped):
        candidates = sorted(grouped[program], key=_quality_key, reverse=True)
        best = candidates[0]
        if best["valid"]:
            selected.append(best)
        else:
            skipped_invalid += 1

        remove_dirs.extend(pathlib.Path(item["run_dir"]) for item in candidates[1:])

    manifest_entries = [
        _build_manifest_entry(
            info=info,
            variant=args.variant,
            project_root=project_root,
            bin_dir=bin_dir,
            test_dir=test_dir,
            by_run_dir=by_run_dir,
            by_program=by_program,
            fallback_window_sec=args.window_sec,
            fallback_duration_sec=args.duration_sec,
            fallback_sample_rate=args.sample_rate,
        )
        for info in selected
    ]

    if not args.dry_run:
        for run_dir in remove_dirs:
            if run_dir.exists():
                shutil.rmtree(run_dir)
        _write_manifest(manifest_path, manifest_entries)

    duplicate_groups = sum(1 for candidates in grouped.values() if len(candidates) > 1)
    print(
        json.dumps(
            {
                "variant": args.variant,
                "output_root": _relative_to_project(output_root, project_root),
                "manifest": _relative_to_project(manifest_path, project_root),
                "program_count": len(grouped),
                "selected_programs": len(manifest_entries),
                "duplicate_groups": duplicate_groups,
                "removed_dirs": len(remove_dirs),
                "skipped_invalid_programs": skipped_invalid,
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()