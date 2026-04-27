#!/usr/bin/env python3
"""
build_anchor_set.py — 构建参考锚点集（方案 A：参考锚点法）
============================================================

机制：Fixed Work · cycles_per_iter
  采集脚本以 while-true 循环运行程序，每次迭代产生独立 PID。
  使用 cycles_per_iter（= total_cycles / active_pid_count）而非 total_cycles，
  避免因 O3 迭代次数多于 O0 导致总量比较失真。

锚点定义
  对每个程序家族：
    - O0 变体作为基准锚点（baseline），S_O0 = 0
    - O3 变体作为"强优化"锚点（若可用），S_O3 = log(cpi_O0 / cpi_O3)
      其中 cpi_k = cycles_per_iter_k

标准化分数
  S_k = log(cpi_O0 / cpi_k)
    > 0 → k 比 O0 每次迭代用更少 cycles（优化更多）
    = 0 → 与 O0 持平
    < 0 → k 比 O0 更慢

输出
  train_set/anchor_set.parquet
    columns: program, variant, total_cycles, score_gt,
             anchor_role, [NON_TIME_COLS z-scored features...]

用法
  python scripts/build_anchor_set.py
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np
import pandas as pd

from feature_columns import DROPPED_INPUT_FEATURES, NON_TIME_COLS

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# 分数区间 → 档位
BAND_THRESHOLDS = [
    (0.75, "strong"),
    (0.50, "good"),
    (0.25, "medium"),
    (0.00, "poor"),
]


def _safe_div(numer: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return float(numer / denom)


def _anchor_quality_from_row(row_raw: pd.Series) -> tuple[float, float]:
    active_pid_count = int(row_raw.get("active_pid_count", 0) or 0)
    active_window_count = float(row_raw.get("active_window_count", 0) or 0.0)
    window_count = float(row_raw.get("window_count", 0) or 0.0)
    active_window_ratio = _safe_div(active_window_count, window_count)

    pid_score = min(1.0, math.log1p(max(active_pid_count, 0)) / math.log1p(64.0))
    window_score = math.sqrt(max(0.0, min(active_window_ratio, 1.0)))
    quality = max(0.0, min(pid_score * window_score, 1.0))
    return active_window_ratio, quality


def score_to_band(percentile: float) -> str:
    """将百分位 [0,1] 映射为优化档位。"""
    for thr, label in BAND_THRESHOLDS:
        if percentile >= thr:
            return label
    return "poor"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="构建参考锚点集（Fixed Work · 方案 A）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw",    default="train_set/run_features.parquet",
                        help="原始运行特征（含 cycles_per_iter）")
    parser.add_argument("--zscore", default="train_set/run_features_zscore.parquet",
                        help="Z-score 归一化特征（模型输入）")
    parser.add_argument("--output", default="train_set/anchor_set.parquet",
                        help="锚点集输出路径")
    parser.add_argument("--anchors", nargs="+", default=["O0", "O2", "O3"],
                        help="用作锚点的变体列表（第一个视为基准，S=0）")
    args = parser.parse_args()

    raw_path    = (REPO_ROOT / args.raw).resolve()
    zscore_path = (REPO_ROOT / args.zscore).resolve()
    out_path    = (REPO_ROOT / args.output).resolve()

    for p in (raw_path, zscore_path):
        if not p.exists():
            sys.exit(f"[错误] 找不到 {p}")

    # ── 读取数据 ──
    df_raw    = pd.read_parquet(raw_path)
    df_zscore = pd.read_parquet(zscore_path)
    if DROPPED_INPUT_FEATURES:
        print(f"[info] 已从锚点输入中剔除死特征: {', '.join(DROPPED_INPUT_FEATURES)}")

    # 建立 (program, variant) 快速查找
    raw_map = {
        (r["program"], r["variant"]): r
        for _, r in df_raw.iterrows()
    }
    z_map = {
        (r["program"], r["variant"]): r
        for _, r in df_zscore.iterrows()
    }

    baseline_variant = args.anchors[0]  # 通常是 O0
    anchor_variants  = args.anchors      # ["O0", "O3"]

    records: list[dict] = []
    programs_ok = 0
    programs_skip = 0

    programs = sorted(df_raw["program"].unique())
    for prog in programs:
        # 基准 cycles（O0）
        if (prog, baseline_variant) not in raw_map:
            programs_skip += 1
            continue

        cycles_base = max(float(raw_map[(prog, baseline_variant)].get("cycles_per_iter", 0))
                          or float(raw_map[(prog, baseline_variant)]["total_cycles"]), 1)

        for variant in anchor_variants:
            if (prog, variant) not in raw_map or (prog, variant) not in z_map:
                continue  # 该变体缺失，跳过

            row_raw = raw_map[(prog, variant)]
            row_z   = z_map[(prog, variant)]

            cycles_k  = max(float(row_raw.get("cycles_per_iter", 0))
                           or float(row_raw["total_cycles"]), 1)
            score_gt  = math.log(cycles_base / cycles_k)  # S_k = log(cpi_O0 / cpi_k)
            active_window_ratio, anchor_quality = _anchor_quality_from_row(row_raw)

            # anchor_role：第一个变体为 baseline，其余为 reference
            anchor_role = "baseline" if variant == baseline_variant else "reference"

            rec: dict = {
                "program":         prog,
                "variant":         variant,
                "cycles_per_iter": round(cycles_k, 1),
                "total_cycles":    int(row_raw.get("total_cycles", 0)),
                "active_pid_count": int(row_raw.get("active_pid_count", 1)),
                "window_count":    int(row_raw.get("window_count", 0) or 0),
                "active_window_count": int(row_raw.get("active_window_count", 0) or 0),
                "active_window_ratio": round(active_window_ratio, 6),
                "anchor_quality":  round(anchor_quality, 6),
                "score_gt":        round(score_gt, 6),
                "anchor_role":     anchor_role,
            }
            # 附上 z-score 特征（模型推理时需要）
            for col in NON_TIME_COLS:
                rec[col] = float(row_z.get(col, 0.0))

            records.append(rec)

        programs_ok += 1

    df_anchors = pd.DataFrame(records)

    # ── 计算归一化统计（仅基于 O0/O1/O2/O3 所有分数，用于后续 0-100 映射）──
    all_scores = df_anchors["score_gt"].values
    score_p5   = float(np.percentile(all_scores, 5))
    score_p95  = float(np.percentile(all_scores, 95))
    score_mean = float(all_scores.mean())
    score_std  = float(all_scores.std())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_anchors.to_parquet(out_path, index=False)

    # 保存归一化统计
    stats_path = out_path.with_suffix(".stats.json")
    stats = {
        "baseline_variant": baseline_variant,
        "anchor_variants":  anchor_variants,
        "n_programs":       programs_ok,
        "n_anchors":        len(df_anchors),
        "anchors_by_variant": {
            str(var): int((df_anchors["variant"] == var).sum())
            for var in anchor_variants
        },
        "anchor_quality_mean": round(float(df_anchors["anchor_quality"].mean()), 4),
        "anchor_quality_min":  round(float(df_anchors["anchor_quality"].min()), 4),
        "anchor_quality_max":  round(float(df_anchors["anchor_quality"].max()), 4),
        "score_min":        round(float(all_scores.min()), 4),
        "score_max":        round(float(all_scores.max()), 4),
        "score_p5":         round(score_p5,  4),
        "score_p95":        round(score_p95, 4),
        "score_mean":       round(score_mean, 4),
        "score_std":        round(score_std,  4),
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    # ── 打印摘要 ──
    print(f"\n{'='*60}")
    print(f"  锚点集构建完成（Fixed Work · 基准变体={baseline_variant}）")
    print(f"{'='*60}")
    print(f"  程序数: {programs_ok}  （跳过 {programs_skip}，缺少 {baseline_variant}）")
    print(f"  锚点总数: {len(df_anchors)}")
    print(f"  锚点质量: mean={stats['anchor_quality_mean']:.3f}  "
          f"min={stats['anchor_quality_min']:.3f}  max={stats['anchor_quality_max']:.3f}")
    print()
    print(f"  score_gt = log(cycles_{baseline_variant} / cycles_k)  [固定工作量对数时间比]")
    print(f"  分数分布：min={stats['score_min']:.3f}  max={stats['score_max']:.3f}  "
          f"mean={stats['score_mean']:.3f}  std={stats['score_std']:.3f}")
    print()

    # 按变体打印分数统计
    for var in anchor_variants:
        sub = df_anchors[df_anchors["variant"] == var]["score_gt"]
        if len(sub) == 0:
            continue
        print(f"  {var:4s}: n={len(sub):4d}  mean={sub.mean():+.3f}  "
              f"median={sub.median():+.3f}  min={sub.min():+.3f}  max={sub.max():+.3f}")

    print()
    print(f"[ok] 锚点集: {out_path}")
    print(f"[ok] 统计信息: {stats_path}")


if __name__ == "__main__":
    main()
