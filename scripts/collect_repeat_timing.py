#!/usr/bin/env python3
"""
collect_repeat_timing.py — 为已采集 run 补充 fixed-work repeat timing sidecar
=============================================================================

目标
----
读取 llvm-test-suite 的 manifest_bcc_*.jsonl，重复执行其中记录的 run_cmd，
对每个 (program, variant) 生成更稳的 wall-time 真值，并写入对应 run 目录下的
repeat_timing.json。

输出 sidecar
-------------
每个 output_dir 下写出 repeat_timing.json，核心字段包括：
  - median_wall_time_sec
  - mad_wall_time_sec
  - success_count / failure_count
  - valid

当 valid=true 且 median_wall_time_sec > 0 时，下游 build_time_score_table.py
会优先使用该值构造 score_time。

用法
----
  python scripts/collect_repeat_timing.py
  python scripts/collect_repeat_timing.py --variant O2 --variant O3
  python scripts/collect_repeat_timing.py --program MiBench_network-dijkstra --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "llvm_test_suite"
DEFAULT_OUTPUT_NAME = "repeat_timing.json"
DEFAULT_FALLBACK_SIDECAR_ROOT = DEFAULT_DATA_ROOT / "repeat_timing_sidecars"
DEFAULT_WARMUP_COUNT = 3
DEFAULT_REPEAT_COUNT = 11
DEFAULT_TIMEOUT_SEC = 300.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="manifest_bcc_*.jsonl 路径；可重复传入，默认自动扫描 data/llvm_test_suite/manifest_bcc_*.jsonl",
    )
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="llvm-test-suite 派生数据根目录（用于自动扫描默认 manifest）",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help="写入每个 output_dir 的 sidecar 文件名",
    )
    parser.add_argument(
        "--fallback-sidecar-root",
        default=str(DEFAULT_FALLBACK_SIDECAR_ROOT),
        help="当 output_dir 不可写时，将 sidecar 镜像写入该根目录；目录结构保持与 data/llvm_test_suite 下的 output_dir 一致",
    )
    parser.add_argument(
        "--program",
        action="append",
        default=[],
        help="只处理指定 program；可重复传入",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="只处理指定 variant；可重复传入",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="最多处理多少条 manifest 记录；0 表示不限制",
    )
    parser.add_argument(
        "--warmup-count",
        type=int,
        default=DEFAULT_WARMUP_COUNT,
        help="正式计时前的预热次数",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=DEFAULT_REPEAT_COUNT,
        help="正式计时次数",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=DEFAULT_TIMEOUT_SEC,
        help="单次 run_cmd 的超时秒数",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若 output_dir 下已存在 repeat_timing.json，则覆盖重写",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将要处理的目标，不实际执行命令",
    )
    return parser.parse_args()


def _resolve_path(raw_path: str) -> pathlib.Path:
    path = pathlib.Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _relative_to_repo(path: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _nearest_existing_parent(path: pathlib.Path) -> pathlib.Path | None:
    current = path
    while not current.exists():
        parent = current.parent
        if parent == current:
            return None
        current = parent
    return current if current.is_dir() else current.parent


def _can_write_target(path: pathlib.Path) -> bool:
    if path.exists():
        return os.access(path, os.W_OK)
    parent = _nearest_existing_parent(path.parent)
    return parent is not None and os.access(parent, os.W_OK)


def _sidecar_subpath(output_dir: pathlib.Path, data_root: pathlib.Path) -> pathlib.Path:
    resolved_output_dir = output_dir.resolve()
    resolved_data_root = data_root.resolve()

    try:
        return resolved_output_dir.relative_to(resolved_data_root)
    except ValueError:
        pass

    try:
        return resolved_output_dir.relative_to(REPO_ROOT)
    except ValueError:
        pass

    anchor = ("data", "llvm_test_suite")
    parts = resolved_output_dir.parts
    for index in range(len(parts) - len(anchor) + 1):
        if parts[index:index + len(anchor)] == anchor:
            return pathlib.Path(*parts[index + len(anchor):])

    sanitized_parts = [part for part in parts if part and part != resolved_output_dir.anchor]
    return pathlib.Path("external", *sanitized_parts)


def _fallback_sidecar_path(
    output_dir: pathlib.Path,
    output_name: str,
    data_root: pathlib.Path,
    fallback_sidecar_root: pathlib.Path,
) -> pathlib.Path:
    return fallback_sidecar_root / _sidecar_subpath(output_dir, data_root) / output_name


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _default_manifests(data_root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(data_root.glob("manifest_bcc_*.jsonl"))


def _select_entries(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest_paths = [_resolve_path(path) for path in args.manifest]
    if not manifest_paths:
        manifest_paths = _default_manifests(_resolve_path(args.data_root))
    if not manifest_paths:
        raise FileNotFoundError("未找到任何 manifest_bcc_*.jsonl")

    selected_programs = set(args.program)
    selected_variants = set(args.variant)
    entries: list[dict[str, Any]] = []
    seen_output_dirs: set[str] = set()

    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest 不存在: {manifest_path}")
        for entry in _load_jsonl(manifest_path):
            program = str(entry.get("program", "")).strip()
            variant = str(entry.get("variant", "")).strip()
            output_dir = str(entry.get("output_dir", "")).strip()
            if not program or not variant or not output_dir:
                continue
            if selected_programs and program not in selected_programs:
                continue
            if selected_variants and variant not in selected_variants:
                continue
            if output_dir in seen_output_dirs:
                continue
            seen_output_dirs.add(output_dir)
            entries.append({
                **entry,
                "_manifest_path": str(manifest_path),
            })

    entries.sort(key=lambda item: (str(item.get("variant", "")), str(item.get("program", ""))))
    if args.limit and args.limit > 0:
        entries = entries[: args.limit]
    return entries


def _run_once(run_cmd: str, cwd: pathlib.Path, timeout_sec: float) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            ["bash", "-lc", run_cmd],
            cwd=str(cwd),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=timeout_sec,
        )
        elapsed_sec = time.perf_counter() - start
        return {
            "ok": completed.returncode == 0,
            "returncode": int(completed.returncode),
            "elapsed_sec": float(elapsed_sec),
            "timeout": False,
        }
    except subprocess.TimeoutExpired:
        elapsed_sec = time.perf_counter() - start
        return {
            "ok": False,
            "returncode": None,
            "elapsed_sec": float(elapsed_sec),
            "timeout": True,
        }


def _median_abs_dev(values: list[float], median: float) -> float:
    if not values:
        return float("nan")
    deviations = [abs(value - median) for value in values]
    return float(statistics.median(deviations))


def _build_sidecar_payload(
    entry: dict[str, Any],
    timings_sec: list[float],
    failure_count: int,
    timeout_count: int,
    warmup_failures: int,
    sidecar_path: pathlib.Path,
    sidecar_storage: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    median_wall_time_sec = float(statistics.median(timings_sec)) if timings_sec else float("nan")
    mean_wall_time_sec = float(statistics.mean(timings_sec)) if timings_sec else float("nan")
    stdev_wall_time_sec = (
        float(statistics.stdev(timings_sec))
        if len(timings_sec) > 1
        else (0.0 if timings_sec else float("nan"))
    )
    mad_wall_time_sec = _median_abs_dev(timings_sec, median_wall_time_sec)
    valid = (
        len(timings_sec) == args.repeat_count
        and args.repeat_count > 0
        and math.isfinite(median_wall_time_sec)
        and median_wall_time_sec > 0.0
    )

    test_file = _resolve_path(str(entry["test_file"]))
    output_dir = _resolve_path(str(entry["output_dir"]))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "time_source": "repeat_timing",
        "valid": bool(valid),
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "program": str(entry.get("program", "")),
        "variant": str(entry.get("variant", "")),
        "manifest_path": _relative_to_repo(_resolve_path(str(entry["_manifest_path"]))),
        "test_file": _relative_to_repo(test_file),
        "output_dir": _relative_to_repo(output_dir),
        "sidecar_path": _relative_to_repo(sidecar_path),
        "sidecar_storage": sidecar_storage,
        "cwd": _relative_to_repo(test_file.parent),
        "run_cmd": str(entry.get("run_cmd", "")),
        "warmup_count": int(args.warmup_count),
        "repeat_count": int(args.repeat_count),
        "timeout_sec": float(args.timeout_sec),
        "warmup_failures": int(warmup_failures),
        "success_count": int(len(timings_sec)),
        "failure_count": int(failure_count),
        "timeout_count": int(timeout_count),
        "timings_sec": [round(float(value), 6) for value in timings_sec],
        "median_wall_time_sec": round(median_wall_time_sec, 6) if math.isfinite(median_wall_time_sec) else None,
        "mean_wall_time_sec": round(mean_wall_time_sec, 6) if math.isfinite(mean_wall_time_sec) else None,
        "stdev_wall_time_sec": round(stdev_wall_time_sec, 6) if math.isfinite(stdev_wall_time_sec) else None,
        "mad_wall_time_sec": round(mad_wall_time_sec, 6) if math.isfinite(mad_wall_time_sec) else None,
        "min_wall_time_sec": round(min(timings_sec), 6) if timings_sec else None,
        "max_wall_time_sec": round(max(timings_sec), 6) if timings_sec else None,
    }
    return payload


def main() -> None:
    args = parse_args()
    data_root = _resolve_path(str(args.data_root))
    fallback_sidecar_root = _resolve_path(str(args.fallback_sidecar_root))
    entries = _select_entries(args)
    if not entries:
        print("[warn] 没有匹配到任何 manifest 记录", file=sys.stderr)
        sys.exit(1)

    print(f"[info] 选中 {len(entries)} 条 repeat timing 目标", flush=True)
    print(
        f"       warmup={args.warmup_count}  repeat={args.repeat_count}  timeout={args.timeout_sec:.1f}s",
        flush=True,
    )

    success_entries = 0
    skipped_entries = 0
    invalid_entries = 0
    fallback_entries = 0

    for index, entry in enumerate(entries, start=1):
        program = str(entry["program"])
        variant = str(entry["variant"])
        run_cmd = str(entry.get("run_cmd", "")).strip()
        test_file = _resolve_path(str(entry["test_file"]))
        output_dir = _resolve_path(str(entry["output_dir"]))
        primary_sidecar_path = output_dir / args.output_name
        fallback_sidecar_path = _fallback_sidecar_path(
            output_dir,
            args.output_name,
            data_root=data_root,
            fallback_sidecar_root=fallback_sidecar_root,
        )

        if not run_cmd:
            print(f"[warn] [{index}/{len(entries)}] {variant}/{program}: 缺少 run_cmd，跳过", flush=True)
            invalid_entries += 1
            continue

        if not test_file.exists():
            print(f"[warn] [{index}/{len(entries)}] {variant}/{program}: test_file 不存在: {test_file}", flush=True)
            invalid_entries += 1
            continue

        if not output_dir.exists():
            print(f"[warn] [{index}/{len(entries)}] {variant}/{program}: output_dir 不存在: {output_dir}", flush=True)
            invalid_entries += 1
            continue

        existing_sidecar_path = None
        if primary_sidecar_path.exists():
            existing_sidecar_path = primary_sidecar_path
        elif fallback_sidecar_path.exists():
            existing_sidecar_path = fallback_sidecar_path

        if existing_sidecar_path is not None and not args.overwrite:
            print(
                f"[skip] [{index}/{len(entries)}] {variant}/{program}: 已存在 {_relative_to_repo(existing_sidecar_path)}",
                flush=True,
            )
            skipped_entries += 1
            continue

        sidecar_path = primary_sidecar_path
        sidecar_storage = "output_dir"
        if not _can_write_target(primary_sidecar_path):
            sidecar_path = fallback_sidecar_path
            sidecar_storage = "fallback_sidecar_root"

        print(
            f"[info] [{index}/{len(entries)}] {variant}/{program} -> {_relative_to_repo(sidecar_path)}",
            flush=True,
        )
        if sidecar_storage == "fallback_sidecar_root":
            print(
                f"       output_dir 不可写，已回退到 {_relative_to_repo(fallback_sidecar_root)}",
                flush=True,
            )
        if args.dry_run:
            print(f"       cwd={_relative_to_repo(test_file.parent)}", flush=True)
            print(f"       cmd={run_cmd}", flush=True)
            skipped_entries += 1
            continue

        warmup_failures = 0
        for _ in range(args.warmup_count):
            warmup_result = _run_once(run_cmd, test_file.parent, args.timeout_sec)
            if not warmup_result["ok"]:
                warmup_failures += 1

        timings_sec: list[float] = []
        failure_count = 0
        timeout_count = 0
        for _ in range(args.repeat_count):
            result = _run_once(run_cmd, test_file.parent, args.timeout_sec)
            if result["ok"]:
                timings_sec.append(float(result["elapsed_sec"]))
            else:
                failure_count += 1
                timeout_count += int(bool(result.get("timeout", False)))

        payload = _build_sidecar_payload(
            entry,
            timings_sec=timings_sec,
            failure_count=failure_count,
            timeout_count=timeout_count,
            warmup_failures=warmup_failures,
            sidecar_path=sidecar_path,
            sidecar_storage=sidecar_storage,
            args=args,
        )
        try:
            sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            sidecar_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except PermissionError:
            if sidecar_path == fallback_sidecar_path:
                raise
            sidecar_path = fallback_sidecar_path
            sidecar_storage = "fallback_sidecar_root"
            payload["sidecar_path"] = _relative_to_repo(sidecar_path)
            payload["sidecar_storage"] = sidecar_storage
            sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            sidecar_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            print(
                f"[warn] output_dir 写入失败，已回退到 {_relative_to_repo(sidecar_path)}",
                flush=True,
            )

        if sidecar_storage == "fallback_sidecar_root":
            fallback_entries += 1

        if payload["valid"]:
            success_entries += 1
            print(
                f"[ok]   median={payload['median_wall_time_sec']:.6f}s  "
                f"mad={payload['mad_wall_time_sec']:.6f}s  n={payload['success_count']}",
                flush=True,
            )
        else:
            invalid_entries += 1
            print(
                f"[warn] 结果已写入但无效: success={payload['success_count']}/{payload['repeat_count']}  "
                f"failures={payload['failure_count']}  timeouts={payload['timeout_count']}",
                flush=True,
            )

    print()
    print("=" * 58)
    print("  repeat timing sidecar 采集完成")
    print("=" * 58)
    print(f"  valid   : {success_entries}")
    print(f"  skipped : {skipped_entries}")
    print(f"  invalid : {invalid_entries}")
    print(f"  fallback: {fallback_entries}")
    print("=" * 58)

    if invalid_entries > 0 and not args.dry_run:
        sys.exit(1)


if __name__ == "__main__":
    main()