#!/usr/bin/env python3
"""
build_run_features.py — 从 BCC 采集数据构建运行级特征表并进行跨数据集标准化

流程：
  1. 读取全部 manifest 文件（O0/O1/O2/O3）
  2. 对每次 run，聚合 window_metrics.jsonl → 运行级特征
  3. 计算 MPKI 归一化特征与比率特征
  4. 提取时间窗分布特征（均值、标准差、P95、峰值份额、最小值）
  5. 输出 train_set/run_features.parquet 与 .csv（原始 MPKI 特征）
  6. 拟合跨数据集 z-score 标准化器
     - 偏态特征（MPKI 类）先做 log1p 再 z-score
     - 有界特征（IPC、miss rate 类）直接 z-score
  7. 输出 train_set/run_features_zscore.parquet 与 .csv
  8. 保存标准化参数到 train_set/feature_scaler.json

用法：
    python scripts/build_run_features.py
    python scripts/build_run_features.py \
        --data-root data/llvm_test_suite \
        --manifest-prefix manifest_curated \
    --min-active-pids 5 \
        --output train_set
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_MIN_ACTIVE_PIDS = 5
DEFAULT_MIN_CYCLES_PER_ITER = 0.0

VARIANTS = ["O0", "O1", "O2", "O3"]

# ── 需要跨窗口求和的原始计数字段 ─────────────────────────────────────────
SUM_FIELDS = [
    "cycles",
    "instructions",
    "llc_loads",
    "llc_load_misses",
    "llc_stores",
    "llc_store_misses",
    "dtlb_loads",
    "dtlb_load_misses",
    "dtlb_stores",
    "dtlb_store_misses",
    "dtlb_misses",
    "itlb_load_misses",
    "minor_faults",
    "major_faults",
    "samples",
    # fault 子类型
    "anon_faults",
    "file_faults",
    "shared_faults",
    "private_faults",
    "write_faults",
    "instruction_faults",
    # mm syscall 计数与字节
    "mmap_calls",
    "munmap_calls",
    "mprotect_calls",
    "brk_calls",
    "mmap_bytes",
    "munmap_bytes",
    "mprotect_bytes",
    "brk_growth_bytes",
    "brk_shrink_bytes",
]

# ── 偏态特征：先 log1p 再 z-score ─────────────────────────────────────────
# 这些特征在跨程序分布上高度右偏，直接 z-score 效果差。
LOG1P_FEATURES: frozenset[str] = frozenset({
    # 运行级 MPKI
    "llc_mpki",
    "llc_store_mpki",
    "dtlb_mpki",
    "itlb_mpki",
    "fault_per_ki",
    "fault_per_ms",
    # 窗口分布 MPKI
    "win_llc_mpki_mean",
    "win_llc_mpki_std",
    "win_llc_mpki_p95",
    "win_llc_mpki_peak_share",
    "win_dtlb_mpki_mean",
    "win_dtlb_mpki_std",
    "win_dtlb_mpki_p95",
    "win_dtlb_mpki_peak_share",
    "win_itlb_mpki_mean",
    "win_itlb_mpki_std",
    "win_itlb_mpki_p95",
    "win_itlb_mpki_peak_share",
    "win_fault_mean",
    "win_fault_std",
    "win_fault_p95",
    "win_fault_peak_share",
    # mm syscall 密度（计数/字节 per ms，高度右偏）
    "mmap_per_ms",
    "munmap_per_ms",
    "brk_per_ms",
    "mm_syscall_per_ms",
    "mmap_bytes_per_ms",
    # 阶段 LLC MPKI（与运行级 MPKI 分布一致）
    "warmup_llc_mpki",
    "steady_llc_mpki",
})

# ── 排除出标准化范围的列（时间真值、元数据、原始计数） ────────────────────
_EXCLUDE_EXACT: frozenset[str] = frozenset({
    "wall_time_sec",
    "wall_time_ms",
    "window_count",
    "active_window_count",
    "active_pid_count",     # 迭代次数代理，作为元数据保留，不参与特征标准化
    "cycles_per_iter",      # 用于标签定义，不作为模型输入特征
    "instructions_per_iter",
})
_EXCLUDE_PREFIXES: tuple[str, ...] = (
    "total_",
    "run_id",
    "program",
    "variant",
    "output_dir",
)


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    """安全除法：分母为 0 时返回 default。"""
    return num / den if den > 0 else default


def _window_stats(values: list[float]) -> dict[str, float]:
    """
    对一批窗口值计算分布统计量：
    mean, std, p95, peak_share (max/sum), min
    """
    if not values:
        return {"mean": 0.0, "std": 0.0, "p95": 0.0, "peak_share": 0.0, "min": 0.0}
    arr = np.array(values, dtype=float)
    total = arr.sum()
    return {
        "mean": float(arr.mean()),
        "std":  float(arr.std()),
        "p95":  float(np.percentile(arr, 95)),
        "peak_share": float(arr.max() / total) if total > 0 else 0.0,
        "min":  float(arr.min()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 每次 run 的特征聚合
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_run(windows: list[dict], completion_count: int | None = None) -> dict[str, Any]:
    """
    将该 run 的 window_metrics 列表聚合为单次运行特征字典。

    输出三类特征：
    A. 元信息（时间、窗口数、迭代次数代理）
    B. 原始计数总和（total_*）
    C. MPKI / 比率特征（模型主输入）
    D. 时间窗分布特征（win_*）
    E. 阶段特征（warmup / steady-state）

    迭代次数说明
    -----------
    采集脚本以 while true 循环运行程序，每次迭代产生独立 PID。
    因此当前运行内出现过的独立活跃 PID 数（有 instructions > 0 的 PID）
    可作为 completion_count 的代理，用于将 total_cycles 归一化为
    per-iteration 效率指标。这是用现有数据能做到的最近似估计；
    准确值需要在采集侧记录进程退出次数（见 run_metadata.jsonl）。
    """
    # ── A. 时间与窗口统计 ─────────────────────────────────────────────────
    start_ns = min(w["start_ns"] for w in windows)
    end_ns   = max(w["end_ns"]   for w in windows)
    wall_time_ms = (end_ns - start_ns) / 1e6
    window_count = len(windows)
    active_window_count = sum(1 for w in windows if w.get("instructions", 0) > 0)

    # ── 迭代次数代理：活跃 PID 数 ─────────────────────────────────────────
    # 采集脚本使用 while-true 循环，每次迭代独立 PID。
    # 统计在本次 run 中出现过且有 instructions > 0 的不同 PID 数作为代理值。
    # 若数据中无 pid 字段（极少数情况），退化为 1。
    active_pids: set = {
        w["pid"] for w in windows
        if w.get("instructions", 0) > 0 and "pid" in w
    }
    active_pid_count = max(len(active_pids), 1)

    feat: dict[str, Any] = {
        "wall_time_sec":        wall_time_ms / 1000.0,
        "wall_time_ms":         wall_time_ms,
        "window_count":         window_count,
        "active_window_count":  active_window_count,
        "active_pid_count":     active_pid_count,
    }

    # ── B. 原始计数总和 ───────────────────────────────────────────────────
    totals: dict[str, float] = {f: 0.0 for f in SUM_FIELDS}
    for w in windows:
        for f in SUM_FIELDS:
            totals[f] += w.get(f, 0)

    for f in SUM_FIELDS:
        feat[f"total_{f}"] = int(totals[f])

    # ── B2. 迭代归一化计数（per-iteration 效率代理） ─────────────────────
    # 将原始计数除以 active_pid_count，得到单次迭代的近似值。
    # 这是比 total_* 更合理的"固定工作量"代理，因为程序在 60s 内
    # 会循环执行 active_pid_count 次，不同 variant 迭代次数不同。
    iter_count = completion_count if (completion_count is not None and completion_count > 0) else active_pid_count
    feat["cycles_per_iter"]       = _safe_div(totals["cycles"],       iter_count)
    feat["instructions_per_iter"] = _safe_div(totals["instructions"], iter_count)

    # ── C. MPKI 归一化特征与比率特征 ────────────────────────────────────
    instr  = totals["instructions"]
    cycles = totals["cycles"]

    # 效率指标
    feat["ipc"] = _safe_div(instr, cycles)
    feat["cpi"] = _safe_div(cycles, instr)

    # LLC miss 率与 MPKI
    feat["llc_load_miss_rate"]  = _safe_div(totals["llc_load_misses"],  totals["llc_loads"])
    feat["llc_store_miss_rate"] = _safe_div(totals["llc_store_misses"], totals["llc_stores"])
    feat["llc_mpki"]            = _safe_div(totals["llc_load_misses"],  instr) * 1000
    feat["llc_store_mpki"]      = _safe_div(totals["llc_store_misses"], instr) * 1000

    # dTLB miss 率与 MPKI
    dtlb_accesses = totals["dtlb_loads"] + totals["dtlb_stores"]
    feat["dtlb_miss_rate"]  = _safe_div(totals["dtlb_misses"], dtlb_accesses)
    feat["dtlb_mpki"]       = _safe_div(totals["dtlb_misses"], instr) * 1000

    # iTLB MPKI
    feat["itlb_mpki"] = _safe_div(totals["itlb_load_misses"], instr) * 1000

    # Page fault 特征
    total_faults = totals["minor_faults"] + totals["major_faults"]
    feat["total_page_faults"] = int(total_faults)
    feat["fault_per_ki"]      = _safe_div(total_faults, instr) * 1000
    feat["fault_per_ms"]      = _safe_div(total_faults, wall_time_ms)
    feat["minor_fault_ratio"] = _safe_div(totals["minor_faults"], total_faults)

    # 采样覆盖率
    feat["samples_per_ms"] = _safe_div(totals["samples"], wall_time_ms)

    # ── C2. Fault 子类型比例特征 ─────────────────────────────────────────
    # 相对于 total_faults 的各子类型占比（bounded [0,1]，直接 z-score）
    feat["anon_fault_ratio"]        = _safe_div(totals["anon_faults"],        total_faults)
    feat["file_fault_ratio"]        = _safe_div(totals["file_faults"],        total_faults)
    feat["write_fault_ratio"]       = _safe_div(totals["write_faults"],       total_faults)
    feat["instruction_fault_ratio"] = _safe_div(totals["instruction_faults"], total_faults)

    # ── C3. MM syscall 密度特征 ──────────────────────────────────────────
    # 调用频率 per ms（log1p z-score，高度右偏）
    feat["mmap_per_ms"]       = _safe_div(totals["mmap_calls"],   wall_time_ms)
    feat["munmap_per_ms"]     = _safe_div(totals["munmap_calls"], wall_time_ms)
    feat["brk_per_ms"]        = _safe_div(totals["brk_calls"],    wall_time_ms)
    feat["mm_syscall_per_ms"] = _safe_div(
        totals["mmap_calls"] + totals["munmap_calls"]
        + totals["mprotect_calls"] + totals["brk_calls"],
        wall_time_ms,
    )
    # 分配字节速率
    feat["mmap_bytes_per_ms"] = _safe_div(totals["mmap_bytes"], wall_time_ms)

    # ── D. 时间窗分布特征 ─────────────────────────────────────────────────
    # win_ipc_*：每个窗口内的 IPC
    win_ipc = [
        _safe_div(w.get("instructions", 0), w.get("cycles", 0))
        for w in windows if w.get("cycles", 0) > 0
    ]
    for k, v in _window_stats(win_ipc).items():
        feat[f"win_ipc_{k}"] = v

    # win_llc_mpki_*：每个窗口内的 LLC load MPKI
    win_llc_mpki = [
        _safe_div(w.get("llc_load_misses", 0), w.get("instructions", 0)) * 1000
        for w in windows if w.get("instructions", 0) > 0
    ]
    for k, v in _window_stats(win_llc_mpki).items():
        feat[f"win_llc_mpki_{k}"] = v

    # win_dtlb_mpki_*：每个窗口内的 dTLB MPKI
    win_dtlb_mpki = [
        _safe_div(w.get("dtlb_misses", 0), w.get("instructions", 0)) * 1000
        for w in windows if w.get("instructions", 0) > 0
    ]
    for k, v in _window_stats(win_dtlb_mpki).items():
        feat[f"win_dtlb_mpki_{k}"] = v

    # win_itlb_mpki_*：每个窗口内的 iTLB MPKI
    win_itlb_mpki = [
        _safe_div(w.get("itlb_load_misses", 0), w.get("instructions", 0)) * 1000
        for w in windows if w.get("instructions", 0) > 0
    ]
    for k, v in _window_stats(win_itlb_mpki).items():
        feat[f"win_itlb_mpki_{k}"] = v

    # win_fault_*：每个窗口内的 page fault 原始计数
    win_fault = [
        float(w.get("minor_faults", 0) + w.get("major_faults", 0))
        for w in windows
    ]
    for k, v in _window_stats(win_fault).items():
        feat[f"win_fault_{k}"] = v

    # ── E. 阶段特征（warmup / steady-state）────────────────────────────
    # 将窗口按时间顺序三等分：前 1/4 为 warmup，中间 1/2 为 steady
    n_w = len(windows)
    n_warmup = max(1, n_w // 4)
    n_steady_end = max(n_warmup + 1, 3 * n_w // 4)
    warmup_wins  = windows[:n_warmup]
    steady_wins  = windows[n_warmup:n_steady_end]
    if not steady_wins:                  # 极少窗口时退化
        steady_wins = windows

    def _phase_ipc(wins: list[dict]) -> float:
        instr  = sum(w.get("instructions", 0) for w in wins)
        cyc    = sum(w.get("cycles", 0)       for w in wins)
        return _safe_div(instr, cyc)

    def _phase_llc_mpki(wins: list[dict]) -> float:
        instr = sum(w.get("instructions", 0)    for w in wins)
        miss  = sum(w.get("llc_load_misses", 0) for w in wins)
        return _safe_div(miss, instr) * 1000

    def _phase_fault_per_win(wins: list[dict]) -> float:
        faults = sum(w.get("minor_faults", 0) + w.get("major_faults", 0) for w in wins)
        return _safe_div(faults, len(wins))

    warmup_ipc       = _phase_ipc(warmup_wins)
    steady_ipc_val   = _phase_ipc(steady_wins)
    warmup_llc_mpki  = _phase_llc_mpki(warmup_wins)
    steady_llc_mpki  = _phase_llc_mpki(steady_wins)
    warmup_fault     = _phase_fault_per_win(warmup_wins)
    steady_fault     = _phase_fault_per_win(steady_wins)

    feat["warmup_ipc"]       = warmup_ipc
    feat["steady_ipc"]       = steady_ipc_val
    feat["phase_ipc_ratio"]  = _safe_div(warmup_ipc, steady_ipc_val, default=1.0)
    feat["warmup_llc_mpki"]  = warmup_llc_mpki
    feat["steady_llc_mpki"]  = steady_llc_mpki
    feat["phase_llc_ratio"]  = _safe_div(
        warmup_llc_mpki, max(steady_llc_mpki, 1e-9), default=1.0
    )
    feat["phase_fault_ratio"] = _safe_div(
        warmup_fault, max(steady_fault, 1e-9), default=1.0
    )

    return feat


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────────────────────

def _load_manifest(manifest_path: pathlib.Path) -> list[dict]:
    return [
        json.loads(l)
        for l in manifest_path.read_text().splitlines()
        if l.strip()
    ]


def _semantic_invalid_reasons(
    feat: dict[str, Any],
    min_active_pids: int,
    min_cycles_per_iter: float,
) -> list[str]:
    reasons: list[str] = []

    active_pid_count = int(feat.get("active_pid_count", 0) or 0)
    cycles_per_iter = float(feat.get("cycles_per_iter", 0.0) or 0.0)

    if active_pid_count < min_active_pids:
        reasons.append("low_active_pid_count")
    if cycles_per_iter <= min_cycles_per_iter:
        reasons.append("nonpositive_cycles_per_iter")

    return reasons


def build_run_features(
    data_root: pathlib.Path,
    manifest_prefix: str = "manifest_bcc",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """遍历所有 variant 的 manifest，聚合每条 run 的特征并返回过滤统计。"""
    return build_run_features_with_semantic_filter(
        data_root,
        manifest_prefix=manifest_prefix,
        min_active_pids=DEFAULT_MIN_ACTIVE_PIDS,
        min_cycles_per_iter=DEFAULT_MIN_CYCLES_PER_ITER,
    )


def build_run_features_with_semantic_filter(
    data_root: pathlib.Path,
    manifest_prefix: str = "manifest_bcc",
    min_active_pids: int = DEFAULT_MIN_ACTIVE_PIDS,
    min_cycles_per_iter: float = DEFAULT_MIN_CYCLES_PER_ITER,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """遍历所有 variant 的 manifest，聚合每条 run 的特征，过滤语义无效样本。"""
    records: list[dict] = []
    filter_stats: dict[str, Any] = {
        "manifest_prefix": manifest_prefix,
        "min_active_pids": int(min_active_pids),
        "min_cycles_per_iter": float(min_cycles_per_iter),
        "n_seen": 0,
        "n_kept": 0,
        "n_filtered": 0,
        "reasons": {
            "low_active_pid_count": 0,
            "nonpositive_cycles_per_iter": 0,
        },
        "by_variant": {
            variant: {"seen": 0, "kept": 0, "filtered": 0}
            for variant in VARIANTS
        },
        "examples": [],
    }

    for variant in VARIANTS:
        manifest_path = data_root / f"{manifest_prefix}_{variant}.jsonl"
        if not manifest_path.exists():
            print(f"[skip] manifest 不存在: {manifest_path}", flush=True)
            continue

        manifest = _load_manifest(manifest_path)
        print(f"[info] {variant}: {len(manifest)} 条运行记录", flush=True)

        for entry in manifest:
            filter_stats["n_seen"] += 1
            filter_stats["by_variant"][variant]["seen"] += 1

            program  = entry["program"]
            out_dir  = REPO_ROOT / entry["output_dir"]
            wm_path  = out_dir / "window_metrics.jsonl"
            meta_path = out_dir / "run_metadata.jsonl"

            if not wm_path.exists():
                print(f"  [warn] 缺少 window_metrics.jsonl: {wm_path}", flush=True)
                continue

            windows = [
                json.loads(l)
                for l in wm_path.read_text().splitlines()
                if l.strip()
            ]
            if not windows:
                print(f"  [warn] 空文件: {wm_path}", flush=True)
                continue

            # 提取 run_id 和 completion_count（扫描所有元数据记录）
            run_id = "unknown"
            completion_count: int | None = None
            if meta_path.exists():
                for line in meta_path.read_text().splitlines():
                    if line.strip():
                        try:
                            obj = json.loads(line)
                            if run_id == "unknown" and "run_id" in obj:
                                run_id = obj["run_id"]
                            if "completion_count" in obj:
                                completion_count = int(obj["completion_count"])
                        except (json.JSONDecodeError, ValueError):
                            continue

            feat = aggregate_run(windows, completion_count=completion_count)
            feat.update({
                "run_id":     run_id,
                "program":    program,
                "variant":    variant,
                "output_dir": str(out_dir),
            })

            invalid_reasons = _semantic_invalid_reasons(
                feat,
                min_active_pids=min_active_pids,
                min_cycles_per_iter=min_cycles_per_iter,
            )
            if invalid_reasons:
                filter_stats["n_filtered"] += 1
                filter_stats["by_variant"][variant]["filtered"] += 1
                for reason in invalid_reasons:
                    filter_stats["reasons"][reason] += 1
                if len(filter_stats["examples"]) < 10:
                    filter_stats["examples"].append({
                        "program": program,
                        "variant": variant,
                        "active_pid_count": int(feat.get("active_pid_count", 0) or 0),
                        "cycles_per_iter": float(feat.get("cycles_per_iter", 0.0) or 0.0),
                        "output_dir": str(out_dir.relative_to(REPO_ROOT)),
                        "reasons": invalid_reasons,
                    })
                continue

            filter_stats["n_kept"] += 1
            filter_stats["by_variant"][variant]["kept"] += 1
            records.append(feat)

    if not records:
        sys.exit("[错误] 未读取到任何运行记录，请检查 data_root 路径")

    df = pd.DataFrame(records)

    # 规范列顺序：元数据 → 时间/窗口信息 → 原始计数 → 特征 → 路径
    id_cols   = ["run_id", "program", "variant"]
    info_cols = ["wall_time_sec", "wall_time_ms", "window_count", "active_window_count"]
    total_cols = [c for c in df.columns if c.startswith("total_")]
    feat_cols = [
        c for c in df.columns
        if c not in id_cols + info_cols + total_cols + ["output_dir"]
    ]
    df = df[id_cols + info_cols + total_cols + feat_cols + ["output_dir"]]
    return df, filter_stats


# ─────────────────────────────────────────────────────────────────────────────
# Z-score 标准化
# ─────────────────────────────────────────────────────────────────────────────

def _get_normalize_cols(df: pd.DataFrame) -> list[str]:
    """返回需要进行 z-score 标准化的数值特征列。"""
    cols = []
    for col in df.columns:
        if col in _EXCLUDE_EXACT:
            continue
        if any(col.startswith(p) for p in _EXCLUDE_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def compute_zscore(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """
    跨数据集 z-score 标准化。

    策略：
    - LOG1P_FEATURES 中的偏态特征：clip(0) → log1p → z-score
    - 其余有界特征（IPC、miss rate 等）：直接 z-score

    返回：
    - df_z: 标准化后的 DataFrame（非特征列不变）
    - scaler: {feature_name: {transform, mean, std, n}}
    """
    norm_cols = _get_normalize_cols(df)
    df_z = df.copy()
    scaler: dict[str, dict] = {}

    for col in norm_cols:
        raw = df[col].astype(float)

        if col in LOG1P_FEATURES:
            transform = "log1p_zscore"
            values = raw.clip(lower=0.0).apply(math.log1p)
        else:
            transform = "zscore"
            values = raw

        mean = float(values.mean())
        std  = float(values.std(ddof=0))   # 总体标准差

        if std > 1e-12:
            df_z[col] = (values - mean) / std
        else:
            df_z[col] = 0.0                # 零方差列全部置 0

        scaler[col] = {
            "transform": transform,
            "mean":      mean,
            "std":       std,
            "n":         int(values.notna().sum()),
        }

    return df_z, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="构建运行级特征表并进行跨数据集 z-score 标准化"
    )
    parser.add_argument(
        "--data-root",
        default="data/llvm_test_suite",
        help="BCC 数据根目录（含 manifest_bcc_*.jsonl），默认 data/llvm_test_suite",
    )
    parser.add_argument(
        "--manifest-prefix",
        default="manifest_bcc",
        help="manifest 前缀，默认 manifest_bcc；可指定为 manifest_curated",
    )
    parser.add_argument(
        "--output",
        default="train_set",
        help="输出目录，默认 train_set",
    )
    parser.add_argument(
        "--min-active-pids",
        type=int,
        default=DEFAULT_MIN_ACTIVE_PIDS,
        help="最小有效 active_pid_count；低于该阈值的 run 会被过滤",
    )
    parser.add_argument(
        "--min-cycles-per-iter",
        type=float,
        default=DEFAULT_MIN_CYCLES_PER_ITER,
        help="最小有效 cycles_per_iter；小于等于该阈值的 run 会被过滤",
    )
    args = parser.parse_args()

    data_root = (REPO_ROOT / args.data_root).resolve()
    out_dir   = (REPO_ROOT / args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        sys.exit(f"[错误] 数据目录不存在: {data_root}")

    # ── Step 1. 聚合原始特征 ──────────────────────────────────────────────
    print("=" * 60, flush=True)
    print("Step 1: 聚合运行级特征（MPKI 归一化）", flush=True)
    print("=" * 60, flush=True)

    df, filter_stats = build_run_features_with_semantic_filter(
        data_root,
        manifest_prefix=args.manifest_prefix,
        min_active_pids=args.min_active_pids,
        min_cycles_per_iter=args.min_cycles_per_iter,
    )
    print(f"\n[info] 总计 {len(df)} 条运行记录，{len(df.columns)} 列", flush=True)

    raw_parquet = out_dir / "run_features.parquet"
    raw_csv     = out_dir / "run_features.csv"
    df.to_parquet(raw_parquet, index=False)
    df.to_csv(raw_csv, index=False)
    print(f"[ok]   {raw_parquet}", flush=True)
    print(f"[ok]   {raw_csv}", flush=True)

    # ── Step 2. 跨数据集 z-score 标准化 ──────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("Step 2: 跨数据集 z-score 标准化", flush=True)
    print("=" * 60, flush=True)
    print(
        "  偏态特征（MPKI 类）：log1p → z-score\n"
        "  有界特征（IPC/率类）：直接 z-score",
        flush=True,
    )

    df_z, scaler = compute_zscore(df)

    zscore_parquet = out_dir / "run_features_zscore.parquet"
    zscore_csv     = out_dir / "run_features_zscore.csv"
    df_z.to_parquet(zscore_parquet, index=False)
    df_z.to_csv(zscore_csv, index=False)
    print(f"[ok]   {zscore_parquet}", flush=True)
    print(f"[ok]   {zscore_csv}", flush=True)

    scaler_path = out_dir / "feature_scaler.json"
    scaler_path.write_text(json.dumps(scaler, indent=2, ensure_ascii=False))
    print(f"[ok]   {scaler_path}", flush=True)

    filter_summary_path = out_dir / "run_feature_filter_summary.json"
    filter_summary_path.write_text(json.dumps(filter_stats, indent=2, ensure_ascii=False))
    print(f"[ok]   {filter_summary_path}", flush=True)

    print("\n[info] 语义过滤摘要：", flush=True)
    print(
        f"  kept={filter_stats['n_kept']}  filtered={filter_stats['n_filtered']}  "
        f"min_active_pids={filter_stats['min_active_pids']}  "
        f"min_cycles_per_iter>{filter_stats['min_cycles_per_iter']}",
        flush=True,
    )
    for variant in VARIANTS:
        item = filter_stats["by_variant"][variant]
        print(
            f"  {variant}: seen={item['seen']} kept={item['kept']} filtered={item['filtered']}",
            flush=True,
        )
    if filter_stats["n_filtered"] > 0:
        print("  原因统计：", flush=True)
        for reason, count in filter_stats["reasons"].items():
            print(f"    {reason}: {count}", flush=True)

    # ── Step 3. 统计摘要 ──────────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("Step 3: 原始特征分布摘要（核心特征）", flush=True)
    print("=" * 60, flush=True)

    core_cols = [
        "ipc", "cpi",
        "llc_mpki", "llc_load_miss_rate",
        "dtlb_mpki", "dtlb_miss_rate",
        "itlb_mpki",
        "fault_per_ki", "fault_per_ms",
        "win_ipc_mean", "win_ipc_min",
        "win_llc_mpki_mean", "win_llc_mpki_peak_share",
        "win_dtlb_mpki_mean", "win_itlb_mpki_mean",
        "win_fault_mean", "win_fault_peak_share",
    ]
    available_core = [c for c in core_cols if c in df.columns]
    summary = df[available_core].describe().T[["mean", "std", "min", "50%", "max"]]
    summary.columns = pd.Index(["mean", "std", "min", "median", "max"])
    print(summary.to_string(), flush=True)

    # 零方差列警告
    zero_var_cols = [c for c, s in scaler.items() if s["std"] <= 1e-12]
    if zero_var_cols:
        print(f"\n[warn] {len(zero_var_cols)} 列方差为零（全程序值相同，已置 0）：")
        for c in zero_var_cols:
            print(f"  {c}")

    # 各 variant 样本数
    print("\n[info] 各 variant 样本数：")
    print(df["variant"].value_counts().sort_index().to_string())

    print(f"\n[done] 所有输出写入: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
