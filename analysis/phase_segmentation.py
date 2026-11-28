#!/usr/bin/env python3
"""
phase_segmentation.py — 程序运行周期阶段自动切分算法（对应论文 4.2 节）

基于 window_metrics.jsonl 的多维微架构与虚存时序特征，通过双段方差缩减（Change-Point）
与虚存预热衰减准则，自动识别：
  1. 初始化/预热期（init / warmup）：工作集分配与高密缺页阶段；
  2. 核心计算稳态期（steady）：访存与 IPC 进入稳态计算阶段；
  3. 收尾/释放期（teardown，可选）：尾窗指令骤降或集中释放阶段。

用法：
    python analysis/phase_segmentation.py --data results/xsbench_o2/ebpf --output results/xsbench_o2/phases
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PhaseSegment:
    phase: str
    start_window: int
    end_window: int
    num_windows: int
    duration_sec: float
    mean_ipc: float
    mean_llc_mpki: float
    mean_dtlb_mpki: float
    mean_fault_rate: float
    mean_mmap_mb_per_s: float
    total_instructions: int
    total_cycles: int
    total_minor_faults: int


def load_window_series(data_dir: pathlib.Path, target_pid: int = 0) -> pd.DataFrame:
    """加载并按 window_id 聚合 window_metrics.jsonl 时序。"""
    jsonl = data_dir / "window_metrics.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(f"找不到 {jsonl}")

    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if target_pid > 0 and "pid" in df.columns:
        df = df[df["pid"] == target_pid]
    if df.empty:
        return df

    sum_cols = [
        c for c in (
            "cycles", "instructions", "llc_load_misses", "llc_store_misses",
            "dtlb_misses", "itlb_load_misses", "minor_faults", "major_faults",
            "mmap_bytes", "munmap_bytes", "mmap_calls", "munmap_calls",
        )
        if c in df.columns
    ]
    agg_dict: dict[str, Any] = {c: "sum" for c in sum_cols}
    if "start_ns" in df.columns:
        agg_dict["start_ns"] = "min"
    if "end_ns" in df.columns:
        agg_dict["end_ns"] = "max"

    win = df.groupby("window_id", as_index=False).agg(agg_dict).sort_values("window_id").reset_index(drop=True)

    if "start_ns" in win.columns and "end_ns" in win.columns:
        win["duration_sec"] = ((win["end_ns"] - win["start_ns"]) / 1e9).clip(lower=1e-3)
    else:
        win["duration_sec"] = 1.0

    inst = win["instructions"].clip(lower=1) if "instructions" in win.columns else pd.Series(1, index=win.index)
    cyc = win["cycles"].clip(lower=1) if "cycles" in win.columns else pd.Series(1, index=win.index)
    llc_miss = (
        win.get("llc_load_misses", pd.Series(0, index=win.index))
        + win.get("llc_store_misses", pd.Series(0, index=win.index))
    )
    dtlb_miss = win.get("dtlb_misses", pd.Series(0, index=win.index))
    faults = (
        win.get("minor_faults", pd.Series(0, index=win.index))
        + win.get("major_faults", pd.Series(0, index=win.index))
    )
    mmap_bytes = win.get("mmap_bytes", pd.Series(0, index=win.index))

    win["ipc"] = win.get("instructions", pd.Series(0, index=win.index)) / cyc
    win["llc_mpki"] = 1000.0 * llc_miss / inst
    win["dtlb_mpki"] = 1000.0 * dtlb_miss / inst
    win["fault_rate"] = faults / win["duration_sec"]
    win["mmap_mb_per_s"] = (mmap_bytes / (1024.0 * 1024.0)) / win["duration_sec"]
    return win


def _build_feature_matrix(win: pd.DataFrame) -> np.ndarray:
    """构造用于变点检测的标准化多维特征矩阵。"""
    raw = np.column_stack([
        np.log1p(win["fault_rate"].to_numpy(dtype=float)),
        np.log1p(win["mmap_mb_per_s"].to_numpy(dtype=float)),
        win["ipc"].to_numpy(dtype=float),
        np.log1p(win["llc_mpki"].to_numpy(dtype=float)),
        np.log1p(win["dtlb_mpki"].to_numpy(dtype=float)),
    ])
    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    return (raw - mean) / std


def segment_phases(win: pd.DataFrame, min_effect_r2: float = 0.20) -> dict[str, Any]:
    """
    基于多维时序变点检测（双段方差缩减 + 虚存预热衰减验证）切分运行阶段。
    返回阶段列表、切分点及对齐模型特征列的阶段摘要指标。
    """
    n = len(win)
    if n == 0:
        return {"segments": [], "summary": {}, "change_point": None}

    if n < 3:
        seg = _summarize_segment("steady", win)
        return {
            "segments": [asdict(seg)],
            "change_point": None,
            "variance_reduction_r2": 0.0,
            "summary": _build_phase_summary([seg]),
        }

    # 1. 检测尾窗是否属于极低指令收尾阶段 (teardown)
    steady_end_idx = n
    if "instructions" in win.columns and n >= 4:
        med_inst = float(win["instructions"].iloc[:-1].median())
        if med_inst > 0 and float(win["instructions"].iloc[-1]) < 0.25 * med_inst:
            steady_end_idx = n - 1

    core = win.iloc[:steady_end_idx].reset_index(drop=True)
    m = len(core)
    best_split = 0
    best_r2 = 0.0

    if m >= 3:
        z = _build_feature_matrix(core)
        total_sse = float(((z - z.mean(axis=0)) ** 2).sum())
        if total_sse > 1e-9:
            for k in range(1, m):
                left = z[:k]
                right = z[k:]
                sse = float(((left - left.mean(axis=0)) ** 2).sum() + ((right - right.mean(axis=0)) ** 2).sum())
                r2 = 1.0 - (sse / total_sse)
                if r2 > best_r2:
                    best_r2 = r2
                    best_split = k

    # 2. 判定切分点是否满足显著性（避免纯稳态负载被强行切分）
    segments: list[PhaseSegment] = []
    has_warmup = False
    if best_split > 0 and best_r2 >= min_effect_r2:
        left_df = core.iloc[:best_split]
        right_df = core.iloc[best_split:]
        fault_ratio = (float(left_df["fault_rate"].mean()) + 1.0) / (float(right_df["fault_rate"].mean()) + 1.0)
        ipc_diff = abs(float(left_df["ipc"].mean()) - float(right_df["ipc"].mean()))
        llc_diff = abs(float(left_df["llc_mpki"].mean()) - float(right_df["llc_mpki"].mean()))
        # 预热期通常伴随更高缺页率或显著不同的 IPC / LLC MPKI 稳态
        if fault_ratio >= 1.5 or ipc_diff >= 0.08 or llc_diff >= 1.0 or best_r2 >= 0.45:
            has_warmup = True
            segments.append(_summarize_segment("init", left_df))
            segments.append(_summarize_segment("steady", right_df))

    if not has_warmup:
        segments.append(_summarize_segment("steady", core))

    if steady_end_idx < n:
        segments.append(_summarize_segment("teardown", win.iloc[steady_end_idx:]))

    return {
        "segments": [asdict(s) for s in segments],
        "change_point_window": int(core.iloc[best_split]["window_id"]) if has_warmup else None,
        "variance_reduction_r2": round(best_r2, 4),
        "summary": _build_phase_summary(segments),
    }


def _summarize_segment(name: str, df: pd.DataFrame) -> PhaseSegment:
    inst = int(df["instructions"].sum()) if "instructions" in df.columns else 0
    cyc = int(df["cycles"].sum()) if "cycles" in df.columns else 0
    faults = int(df["minor_faults"].sum() + df.get("major_faults", pd.Series(0, index=df.index)).sum())
    return PhaseSegment(
        phase=name,
        start_window=int(df["window_id"].iloc[0]),
        end_window=int(df["window_id"].iloc[-1]),
        num_windows=len(df),
        duration_sec=round(float(df["duration_sec"].sum()), 4),
        mean_ipc=round(float(inst / max(cyc, 1)), 4),
        mean_llc_mpki=round(float(df["llc_mpki"].mean()), 4),
        mean_dtlb_mpki=round(float(df["dtlb_mpki"].mean()), 4),
        mean_fault_rate=round(float(df["fault_rate"].mean()), 2),
        mean_mmap_mb_per_s=round(float(df["mmap_mb_per_s"].mean()), 4),
        total_instructions=inst,
        total_cycles=cyc,
        total_minor_faults=faults,
    )


def _build_phase_summary(segments: list[PhaseSegment]) -> dict[str, float]:
    by_name = {s.phase: s for s in segments}
    steady = by_name.get("steady") or segments[0]
    init = by_name.get("init") or steady
    return {
        "warmup_ipc": init.mean_ipc,
        "steady_ipc": steady.mean_ipc,
        "phase_ipc_ratio": round(init.mean_ipc / max(steady.mean_ipc, 1e-6), 4),
        "warmup_llc_mpki": init.mean_llc_mpki,
        "steady_llc_mpki": steady.mean_llc_mpki,
        "phase_llc_ratio": round((init.mean_llc_mpki + 1e-6) / max(steady.mean_llc_mpki + 1e-6, 1e-6), 4),
        "phase_fault_ratio": round((init.mean_fault_rate + 1.0) / max(steady.mean_fault_rate + 1.0, 1.0), 4),
    }


def write_markdown_report(result: dict[str, Any], out_path: pathlib.Path) -> None:
    lines = [
        "# 程序运行阶段自动切分报告",
        "",
        f"- 变点窗口 (`change_point_window`): `{result.get('change_point_window')}`",
        f"- 多维方差缩减解释度 ($R^2$): `{result.get('variance_reduction_r2')}`",
        "",
        "## 阶段切分明细",
        "",
        "| 阶段 (`phase`) | 窗口范围 | 时长 (s) | 平均 IPC | LLC MPKI | dTLB MPKI | 缺页率 (/s) | 映射速率 (MB/s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for s in result.get("segments", []):
        lines.append(
            f"| **{s['phase']}** | `w{s['start_window']} ~ w{s['end_window']}` ({s['num_windows']}w) | "
            f"{s['duration_sec']:.2f} | {s['mean_ipc']:.4f} | {s['mean_llc_mpki']:.2f} | "
            f"{s['mean_dtlb_mpki']:.2f} | {s['mean_fault_rate']:.1f} | {s['mean_mmap_mb_per_s']:.2f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="程序运行周期阶段自动切分（论文 4.2 节）")
    p.add_argument("--data", required=True, help="采集数据目录（包含 window_metrics.jsonl）")
    p.add_argument("--output", required=True, help="输出目录")
    p.add_argument("--pid", type=int, default=0, help="可选，过滤特定 PID")
    p.add_argument("--min-r2", type=float, default=0.20, help="变点方差缩减最小阈值（默认 0.20）")
    args = p.parse_args()

    data_dir = pathlib.Path(args.data)
    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    win = load_window_series(data_dir, target_pid=args.pid)
    if win.empty:
        sys.exit("[错误] 无有效窗口数据")

    res = segment_phases(win, min_effect_r2=args.min_r2)
    (out_dir / "phase_segmentation.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown_report(res, out_dir / "phase_segmentation.md")

    print(f"=== 阶段切分完成 (R2={res['variance_reduction_r2']}, change_point={res['change_point_window']}) ===")
    for s in res["segments"]:
        print(
            f"  [{s['phase']:8s}] w{s['start_window']}..w{s['end_window']} "
            f"IPC={s['mean_ipc']:.3f}, LLC_MPKI={s['mean_llc_mpki']:.2f}, fault/s={s['mean_fault_rate']:.1f}"
        )


if __name__ == "__main__":
    main()
