#!/usr/bin/env python3

from __future__ import annotations

import json
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_SCALER_PATH = REPO_ROOT / "train_set" / "feature_scaler.json"
ZERO_VAR_EPS = 1e-12

BASE_NON_TIME_COLS: list[str] = [
    "ipc", "cpi",
    "llc_load_miss_rate", "llc_store_miss_rate",
    "llc_mpki", "llc_store_mpki",
    "dtlb_miss_rate", "dtlb_mpki",
    "itlb_mpki",
    "fault_per_ki", "fault_per_ms",
    "minor_fault_ratio",
    "samples_per_ms",
    "win_ipc_mean", "win_ipc_std", "win_ipc_p95", "win_ipc_peak_share", "win_ipc_min",
    "win_llc_mpki_mean", "win_llc_mpki_std", "win_llc_mpki_p95",
    "win_llc_mpki_peak_share", "win_llc_mpki_min",
    "win_dtlb_mpki_mean", "win_dtlb_mpki_std", "win_dtlb_mpki_p95",
    "win_dtlb_mpki_peak_share", "win_dtlb_mpki_min",
    "win_itlb_mpki_mean", "win_itlb_mpki_std", "win_itlb_mpki_p95",
    "win_itlb_mpki_peak_share", "win_itlb_mpki_min",
    "win_fault_mean", "win_fault_std", "win_fault_p95",
    "win_fault_peak_share", "win_fault_min",
    "anon_fault_ratio",
    "file_fault_ratio",
    "write_fault_ratio",
    "instruction_fault_ratio",
    "mmap_per_ms",
    "munmap_per_ms",
    "brk_per_ms",
    "mm_syscall_per_ms",
    "mmap_bytes_per_ms",
    "warmup_ipc",
    "steady_ipc",
    "phase_ipc_ratio",
    "warmup_llc_mpki",
    "steady_llc_mpki",
    "phase_llc_ratio",
    "phase_fault_ratio",
]

# 当前已知的死特征：minor_fault_ratio 在现有快照中 std=0，保留在 run_features 表里，
# 但不再进入 pair / anchor / model 的输入列。
DEFAULT_DROPPED_INPUT_FEATURES = {"minor_fault_ratio"}


def _load_scaler(path: pathlib.Path) -> dict[str, dict]:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        return {}


def load_zero_variance_feature_cols(
    scaler_path: pathlib.Path | None = None,
) -> list[str]:
    path = scaler_path or DEFAULT_SCALER_PATH
    scaler = _load_scaler(path)
    zero_var: list[str] = []
    for col in BASE_NON_TIME_COLS:
        stats = scaler.get(col)
        if not stats:
            continue
        std = float(stats.get("std", 1.0) or 0.0)
        if std <= ZERO_VAR_EPS:
            zero_var.append(col)
    return zero_var


def get_non_time_cols(
    scaler_path: pathlib.Path | None = None,
) -> list[str]:
    dropped = set(DEFAULT_DROPPED_INPUT_FEATURES)
    dropped.update(load_zero_variance_feature_cols(scaler_path=scaler_path))
    return [col for col in BASE_NON_TIME_COLS if col not in dropped]


NON_TIME_COLS = get_non_time_cols()
DROPPED_INPUT_FEATURES = [col for col in BASE_NON_TIME_COLS if col not in NON_TIME_COLS]
