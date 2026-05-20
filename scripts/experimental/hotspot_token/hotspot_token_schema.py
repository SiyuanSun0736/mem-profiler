#!/usr/bin/env python3

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
WINDOW_FEATURE_NAMES: tuple[str, ...] = (
    "rel_pos",
    "instructions_share",
    "ipc",
    "llc_mpki_log1p",
    "dtlb_mpki_log1p",
    "itlb_mpki_log1p",
    "fault_per_ki_log1p",
    "mm_syscall_per_ki_log1p",
    "samples_per_ms_log1p",
    "hotspot_score",
)
TOP_K_DEFAULT = 6


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den > 0 else default


def _positive_zscore(values: np.ndarray, active_mask: np.ndarray | None = None) -> np.ndarray:
    if values.size == 0:
        return values
    ref = values[active_mask] if active_mask is not None and active_mask.any() else values
    std = float(ref.std())
    if std <= 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    mean = float(ref.mean())
    return np.maximum((values - mean) / std, 0.0).astype(np.float32)


def _low_is_hot_zscore(values: np.ndarray, active_mask: np.ndarray | None = None) -> np.ndarray:
    if values.size == 0:
        return values
    ref = values[active_mask] if active_mask is not None and active_mask.any() else values
    std = float(ref.std())
    if std <= 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    mean = float(ref.mean())
    return np.maximum((mean - values) / std, 0.0).astype(np.float32)


def build_hotspot_token_schema(top_k: int = TOP_K_DEFAULT) -> dict[str, Any]:
    return {
        "top_k": int(top_k),
        "tokens_per_program": int(1 + top_k),
        "summary_token": {
            "token": "summary",
            "description": "Global run-level summary token built from the existing z-score run features.",
        },
        "hotspot_tokens": {
            "token_prefix": "hotspot_window",
            "count": int(top_k),
            "selection": "top-k windows ranked by memory-pressure hotspot score within each run",
            "window_feature_names": list(WINDOW_FEATURE_NAMES),
            "score_formula": (
                "score = sqrt(instructions_share) * (pos_z(llc_mpki) + pos_z(dtlb_mpki) + "
                "0.5*pos_z(itlb_mpki) + pos_z(fault_per_ki) + 0.5*pos_z(mm_syscall_per_ki) + "
                "0.75*pos_z(low_ipc))"
            ),
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
    agg_map: dict[str, str] = {
        col: "sum"
        for col in numeric_candidates
        if col in df.columns
    }
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
    duration_ms = windows["duration_ms"].astype(float).to_numpy()
    windows["samples_per_ms"] = np.divide(samples, duration_ms, out=np.zeros_like(duration_ms), where=duration_ms > 0)

    max_window_id = max(int(windows["window_id"].max()), 1)
    windows["rel_pos"] = windows["window_id"].astype(float) / float(max_window_id)

    active_mask = instructions.to_numpy(dtype=float) > 0
    score = _positive_zscore(windows["llc_mpki"].to_numpy(dtype=float), active_mask)
    score += _positive_zscore(windows["dtlb_mpki"].to_numpy(dtype=float), active_mask)
    score += 0.5 * _positive_zscore(windows["itlb_mpki"].to_numpy(dtype=float), active_mask)
    score += _positive_zscore(windows["fault_per_ki"].to_numpy(dtype=float), active_mask)
    score += 0.5 * _positive_zscore(windows["mm_syscall_per_ki"].to_numpy(dtype=float), active_mask)
    score += 0.75 * _low_is_hot_zscore(windows["ipc"].to_numpy(dtype=float), active_mask)

    importance = np.sqrt(np.clip(windows["instructions_share"].to_numpy(dtype=float), 0.0, None))
    score = score * np.where(active_mask, 0.25 + importance, 0.0)
    if float(score.max(initial=0.0)) <= 1e-12:
        score = windows["instructions_share"].to_numpy(dtype=float)
    windows["hotspot_score"] = score.astype(np.float32)
    return windows


def build_hotspot_tokens_for_run(
    output_dir: str | pathlib.Path,
    top_k: int = TOP_K_DEFAULT,
) -> tuple[np.ndarray, dict[str, Any]]:
    windows = _aggregate_run_windows(output_dir)
    active_windows = windows[windows["instructions"] > 0].copy()
    ranking_source = active_windows if not active_windows.empty else windows
    selected = (
        ranking_source
        .sort_values(["hotspot_score", "instructions_share", "window_id"], ascending=[False, False, True])
        .head(top_k)
        .reset_index(drop=True)
    )

    token_matrix = np.zeros((top_k, len(WINDOW_FEATURE_NAMES)), dtype=np.float32)
    for idx, row in selected.iterrows():
        token_matrix[idx, :] = np.array(
            [
                float(row["rel_pos"]),
                float(row["instructions_share"]),
                float(row["ipc"]),
                float(np.log1p(max(row["llc_mpki"], 0.0))),
                float(np.log1p(max(row["dtlb_mpki"], 0.0))),
                float(np.log1p(max(row["itlb_mpki"], 0.0))),
                float(np.log1p(max(row["fault_per_ki"], 0.0))),
                float(np.log1p(max(row["mm_syscall_per_ki"], 0.0))),
                float(np.log1p(max(row["samples_per_ms"], 0.0))),
                float(row["hotspot_score"]),
            ],
            dtype=np.float32,
        )

    meta = {
        "window_count": int(len(windows)),
        "active_window_count": int((windows["instructions"] > 0).sum()),
        "selected_window_count": int(len(selected)),
        "padding_token_count": int(max(0, top_k - len(selected))),
        "max_hotspot_score": float(windows["hotspot_score"].max()),
        "mean_hotspot_score": float(windows["hotspot_score"].mean()),
    }
    return token_matrix, meta


def build_hotspot_token_map(
    run_rows: pd.DataFrame,
    top_k: int = TOP_K_DEFAULT,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, Any]]:
    required = {"program", "variant", "output_dir"}
    missing = sorted(required.difference(run_rows.columns))
    if missing:
        raise ValueError(f"run_rows missing columns: {', '.join(missing)}")

    token_map: dict[tuple[str, str], np.ndarray] = {}
    metas: list[dict[str, Any]] = []
    for row in run_rows[["program", "variant", "output_dir"]].itertuples(index=False):
        key = (str(row.program), str(row.variant))
        tokens, meta = build_hotspot_tokens_for_run(row.output_dir, top_k=top_k)
        token_map[key] = tokens
        metas.append(meta)

    window_counts = [meta["window_count"] for meta in metas]
    active_counts = [meta["active_window_count"] for meta in metas]
    padding_counts = [meta["padding_token_count"] for meta in metas]
    summary = {
        "n_runs": int(len(token_map)),
        "top_k": int(top_k),
        "window_feature_names": list(WINDOW_FEATURE_NAMES),
        "token_shape": [int(top_k), int(len(WINDOW_FEATURE_NAMES))],
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
        "padding_token_count": {
            "mean": float(np.mean(padding_counts)) if padding_counts else 0.0,
            "max": int(max(padding_counts, default=0)),
        },
    }
    return token_map, summary


def fit_hotspot_token_scaler(
    token_map: dict[tuple[str, str], np.ndarray],
    train_keys: set[tuple[str, str]],
) -> dict[str, Any]:
    matrices: list[np.ndarray] = []
    for key in train_keys:
        arr = token_map[key]
        active_mask = ~np.all(np.isclose(arr, 0.0), axis=1)
        if active_mask.any():
            matrices.append(arr[active_mask])

    if not matrices:
        raise ValueError("no non-zero hotspot tokens available to fit scaler")

    mat = np.vstack(matrices).astype(np.float32)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    return {
        "feature_names": list(WINDOW_FEATURE_NAMES),
        "n_tokens": int(mat.shape[0]),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def apply_hotspot_token_scaler(
    token_map: dict[tuple[str, str], np.ndarray],
    scaler: dict[str, Any],
) -> dict[tuple[str, str], np.ndarray]:
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    scaled: dict[tuple[str, str], np.ndarray] = {}
    for key, arr in token_map.items():
        arr_f = arr.astype(np.float32)
        pad_mask = np.all(np.isclose(arr_f, 0.0), axis=1)
        arr_scaled = (arr_f - mean) / std
        arr_scaled[pad_mask] = 0.0
        scaled[key] = arr_scaled.astype(np.float32)
    return scaled


def dump_schema_json(path: pathlib.Path, schema: dict[str, Any]) -> None:
    path.write_text(json.dumps(schema, indent=2, ensure_ascii=False))