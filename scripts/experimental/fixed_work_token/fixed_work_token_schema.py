#!/usr/bin/env python3

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
WORK_BUCKET_FEATURE_NAMES: tuple[str, ...] = (
    "bucket_midpoint",
    "instructions_share",
    "duration_share",
    "ipc",
    "llc_mpki_log1p",
    "dtlb_mpki_log1p",
    "itlb_mpki_log1p",
    "fault_per_ki_log1p",
    "mm_syscall_per_ki_log1p",
    "samples_per_ms_log1p",
)
NUM_BUCKETS_DEFAULT = 6


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den > 0 else default


def build_fixed_work_token_schema(num_buckets: int = NUM_BUCKETS_DEFAULT) -> dict[str, Any]:
    return {
        "num_buckets": int(num_buckets),
        "tokens_per_program": int(1 + num_buckets),
        "summary_token": {
            "token": "summary",
            "description": "Global run-level summary token built from the existing z-score run features.",
        },
        "work_bucket_tokens": {
            "token_prefix": "work_bucket",
            "count": int(num_buckets),
            "selection": "equal-work buckets built from cumulative instructions share within each run",
            "bucket_feature_names": list(WORK_BUCKET_FEATURE_NAMES),
            "progress_proxy": "cumulative instructions_share",
        },
    }


def _aggregate_run_windows(output_dir: str | pathlib.Path) -> pd.DataFrame:
    run_dir = pathlib.Path(output_dir)
    metrics_path = run_dir / "window_metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing window_metrics.jsonl: {metrics_path}")

    df = pd.read_json(metrics_path, lines=True)
    if df.empty:
        raise ValueError(f"empty window_metrics.jsonl: {metrics_path}")

    numeric_candidates = [
        "instructions",
        "cycles",
        "llc_load_misses",
        "dtlb_misses",
        "itlb_load_misses",
        "minor_faults",
        "major_faults",
        "mmap_calls",
        "munmap_calls",
        "mprotect_calls",
        "brk_calls",
        "samples",
    ]
    agg_map: dict[str, str] = {col: "sum" for col in numeric_candidates if col in df.columns}
    if "start_ns" in df.columns:
        agg_map["start_ns"] = "min"
    if "end_ns" in df.columns:
        agg_map["end_ns"] = "max"

    windows = (
        df.groupby("window_id")
        .agg(agg_map)
        .reset_index()
        .sort_values("window_id")
        .reset_index(drop=True)
    )

    if "start_ns" in windows.columns and "end_ns" in windows.columns:
        windows["duration_ms"] = (windows["end_ns"] - windows["start_ns"]) / 1e6
    else:
        windows["duration_ms"] = 1000.0

    instructions = windows.get("instructions", pd.Series(np.zeros(len(windows), dtype=float))).astype(float)
    cycles = windows.get("cycles", pd.Series(np.zeros(len(windows), dtype=float))).astype(float)
    faults = (
        windows.get("minor_faults", pd.Series(np.zeros(len(windows), dtype=float))).astype(float)
        + windows.get("major_faults", pd.Series(np.zeros(len(windows), dtype=float))).astype(float)
    )
    mm_syscalls = (
        windows.get("mmap_calls", pd.Series(np.zeros(len(windows), dtype=float))).astype(float)
        + windows.get("munmap_calls", pd.Series(np.zeros(len(windows), dtype=float))).astype(float)
        + windows.get("mprotect_calls", pd.Series(np.zeros(len(windows), dtype=float))).astype(float)
        + windows.get("brk_calls", pd.Series(np.zeros(len(windows), dtype=float))).astype(float)
    )
    samples = windows.get("samples", pd.Series(np.zeros(len(windows), dtype=float))).astype(float)
    duration_ms = windows["duration_ms"].astype(float)
    total_instructions = float(instructions.sum())

    windows["instructions_share"] = instructions / total_instructions if total_instructions > 0 else 0.0
    windows["ipc"] = np.divide(instructions, cycles, out=np.zeros_like(instructions), where=cycles > 0)
    windows["llc_mpki"] = np.divide(
        windows.get("llc_load_misses", pd.Series(np.zeros(len(windows), dtype=float))).astype(float) * 1000.0,
        instructions,
        out=np.zeros_like(instructions),
        where=instructions > 0,
    )
    windows["dtlb_mpki"] = np.divide(
        windows.get("dtlb_misses", pd.Series(np.zeros(len(windows), dtype=float))).astype(float) * 1000.0,
        instructions,
        out=np.zeros_like(instructions),
        where=instructions > 0,
    )
    windows["itlb_mpki"] = np.divide(
        windows.get("itlb_load_misses", pd.Series(np.zeros(len(windows), dtype=float))).astype(float) * 1000.0,
        instructions,
        out=np.zeros_like(instructions),
        where=instructions > 0,
    )
    windows["fault_per_ki"] = np.divide(
        faults * 1000.0,
        instructions,
        out=np.zeros_like(instructions),
        where=instructions > 0,
    )
    windows["mm_syscall_per_ki"] = np.divide(
        mm_syscalls * 1000.0,
        instructions,
        out=np.zeros_like(instructions),
        where=instructions > 0,
    )
    windows["samples_per_ms"] = np.divide(samples, duration_ms, out=np.zeros_like(duration_ms), where=duration_ms > 0)

    if total_instructions > 0:
        instr_arr = instructions.to_numpy(dtype=float)
        cum_end = np.cumsum(instr_arr) / total_instructions
        cum_start = np.concatenate(([0.0], cum_end[:-1]))
        windows["work_start_share"] = cum_start
        windows["work_end_share"] = cum_end
    else:
        n = max(len(windows), 1)
        windows["work_start_share"] = np.arange(n) / n
        windows["work_end_share"] = (np.arange(n) + 1) / n
    return windows


def build_fixed_work_tokens_for_run(
    output_dir: str | pathlib.Path,
    num_buckets: int = NUM_BUCKETS_DEFAULT,
) -> tuple[np.ndarray, dict[str, Any]]:
    windows = _aggregate_run_windows(output_dir)
    total_duration = float(windows["duration_ms"].sum())
    total_instructions = float(windows["instructions"].sum())

    token_matrix = np.zeros((num_buckets, len(WORK_BUCKET_FEATURE_NAMES)), dtype=np.float32)
    assigned_instruction_share = np.zeros(num_buckets, dtype=np.float32)

    for bucket_id in range(num_buckets):
        bucket_start = bucket_id / num_buckets
        bucket_end = (bucket_id + 1) / num_buckets
        bucket_mid = 0.5 * (bucket_start + bucket_end)

        accum = {
            "instructions": 0.0,
            "cycles": 0.0,
            "llc_load_misses": 0.0,
            "dtlb_misses": 0.0,
            "itlb_load_misses": 0.0,
            "faults": 0.0,
            "mm_syscalls": 0.0,
            "samples": 0.0,
            "duration_ms": 0.0,
            "instruction_share": 0.0,
        }

        for row in windows.itertuples(index=False):
            overlap = max(0.0, min(float(row.work_end_share), bucket_end) - max(float(row.work_start_share), bucket_start))
            if overlap <= 0:
                continue
            win_span = max(float(row.work_end_share) - float(row.work_start_share), 1e-12)
            frac = overlap / win_span
            accum["instructions"] += float(row.instructions) * frac
            accum["cycles"] += float(row.cycles) * frac
            accum["llc_load_misses"] += float(getattr(row, "llc_load_misses", 0.0)) * frac
            accum["dtlb_misses"] += float(getattr(row, "dtlb_misses", 0.0)) * frac
            accum["itlb_load_misses"] += float(getattr(row, "itlb_load_misses", 0.0)) * frac
            accum["faults"] += float(getattr(row, "minor_faults", 0.0) + getattr(row, "major_faults", 0.0)) * frac
            accum["mm_syscalls"] += float(
                getattr(row, "mmap_calls", 0.0)
                + getattr(row, "munmap_calls", 0.0)
                + getattr(row, "mprotect_calls", 0.0)
                + getattr(row, "brk_calls", 0.0)
            ) * frac
            accum["samples"] += float(getattr(row, "samples", 0.0)) * frac
            accum["duration_ms"] += float(getattr(row, "duration_ms", 0.0)) * frac
            accum["instruction_share"] += overlap

        assigned_instruction_share[bucket_id] = accum["instruction_share"]
        instr = accum["instructions"]
        cyc = accum["cycles"]
        duration_ms = accum["duration_ms"]
        ipc = _safe_div(instr, cyc)
        llc_mpki = _safe_div(accum["llc_load_misses"], instr) * 1000.0
        dtlb_mpki = _safe_div(accum["dtlb_misses"], instr) * 1000.0
        itlb_mpki = _safe_div(accum["itlb_load_misses"], instr) * 1000.0
        fault_per_ki = _safe_div(accum["faults"], instr) * 1000.0
        mm_per_ki = _safe_div(accum["mm_syscalls"], instr) * 1000.0
        samples_per_ms = _safe_div(accum["samples"], duration_ms)
        duration_share = _safe_div(duration_ms, total_duration)

        token_matrix[bucket_id, :] = np.array(
            [
                bucket_mid,
                accum["instruction_share"],
                duration_share,
                ipc,
                np.log1p(max(llc_mpki, 0.0)),
                np.log1p(max(dtlb_mpki, 0.0)),
                np.log1p(max(itlb_mpki, 0.0)),
                np.log1p(max(fault_per_ki, 0.0)),
                np.log1p(max(mm_per_ki, 0.0)),
                np.log1p(max(samples_per_ms, 0.0)),
            ],
            dtype=np.float32,
        )

    meta = {
        "window_count": int(len(windows)),
        "active_window_count": int((windows["instructions"] > 0).sum()),
        "num_buckets": int(num_buckets),
        "mean_assigned_instruction_share": float(assigned_instruction_share.mean()) if assigned_instruction_share.size else 0.0,
        "min_assigned_instruction_share": float(assigned_instruction_share.min()) if assigned_instruction_share.size else 0.0,
        "max_assigned_instruction_share": float(assigned_instruction_share.max()) if assigned_instruction_share.size else 0.0,
        "total_instructions": total_instructions,
    }
    return token_matrix, meta


def build_fixed_work_token_map(
    run_rows: pd.DataFrame,
    num_buckets: int = NUM_BUCKETS_DEFAULT,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, Any]]:
    required = {"program", "variant", "output_dir"}
    missing = sorted(required.difference(run_rows.columns))
    if missing:
        raise ValueError(f"run_rows missing columns: {', '.join(missing)}")

    token_map: dict[tuple[str, str], np.ndarray] = {}
    metas: list[dict[str, Any]] = []
    for row in run_rows[["program", "variant", "output_dir"]].itertuples(index=False):
        key = (str(row.program), str(row.variant))
        tokens, meta = build_fixed_work_tokens_for_run(row.output_dir, num_buckets=num_buckets)
        token_map[key] = tokens
        metas.append(meta)

    window_counts = [meta["window_count"] for meta in metas]
    active_counts = [meta["active_window_count"] for meta in metas]
    mean_instruction_share = [meta["mean_assigned_instruction_share"] for meta in metas]
    summary = {
        "n_runs": int(len(token_map)),
        "num_buckets": int(num_buckets),
        "bucket_feature_names": list(WORK_BUCKET_FEATURE_NAMES),
        "token_shape": [int(num_buckets), int(len(WORK_BUCKET_FEATURE_NAMES))],
        "window_count": {
            "min": int(min(window_counts, default=0)),
            "median": float(np.median(window_counts)) if window_counts else 0.0,
            "max": int(max(window_counts, default=0)),
        },
        "active_window_count": {
            "min": int(min(active_counts, default=0)),
            "median": float(np.median(active_counts)) if active_counts else 0.0,
            "max": int(max(active_counts, default=0)),
        },
        "mean_assigned_instruction_share": {
            "mean": float(np.mean(mean_instruction_share)) if mean_instruction_share else 0.0,
            "std": float(np.std(mean_instruction_share)) if mean_instruction_share else 0.0,
        },
    }
    return token_map, summary


def fit_fixed_work_token_scaler(
    token_map: dict[tuple[str, str], np.ndarray],
    train_keys: set[tuple[str, str]],
) -> dict[str, Any]:
    matrices = [token_map[key] for key in train_keys if key in token_map]
    if not matrices:
        raise ValueError("no fixed-work tokens available to fit scaler")
    mat = np.vstack(matrices).astype(np.float32)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    return {
        "feature_names": list(WORK_BUCKET_FEATURE_NAMES),
        "n_tokens": int(mat.shape[0]),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def apply_fixed_work_token_scaler(
    token_map: dict[tuple[str, str], np.ndarray],
    scaler: dict[str, Any],
) -> dict[tuple[str, str], np.ndarray]:
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    scaled: dict[tuple[str, str], np.ndarray] = {}
    for key, arr in token_map.items():
        scaled[key] = ((arr.astype(np.float32) - mean) / std).astype(np.float32)
    return scaled


def dump_schema_json(path: pathlib.Path, schema: dict[str, Any]) -> None:
    path.write_text(json.dumps(schema, indent=2, ensure_ascii=False))