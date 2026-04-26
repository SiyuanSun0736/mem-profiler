#!/usr/bin/env python3
"""
evaluate_score_vs_time.py — 模型分数对时间评分的外部验证
=========================================================

验证维度
--------
1. 代理标签 vs 时间参考
     proxy label (score_gt) vs score_time
   衡量 cycles_per_iter 作为 wall-time 代理的可靠性

2. 模型分数 vs 时间参考
     model score (score_log) vs score_time
   衡量模型在真实时间维度的泛化性

指标
----
  mae_proxy_time    : MAE( score_gt,  score_time )
  corr_proxy_time   : Pearson r( score_gt, score_time )
  spearman_proxy    : Spearman ρ( score_gt, score_time )
  mae_model_time    : MAE( score_log, score_time )
  corr_model_time   : Pearson r( score_log, score_time )
  spearman_model    : Spearman ρ( score_log, score_time )
  band_acc_model    : 档位一致率（poor/medium/good/strong）以 score_time 为真值
  band_acc_proxy    : 档位一致率（以 score_time 为真值，score_gt 做预测）
  dir_acc_model     : 方向一致率：sign(score_log) == sign(score_time)
  dir_acc_proxy     : 方向一致率：sign(score_gt)  == sign(score_time)
  n_valid           : 参与计算的 (program, variant) 数量

档位分界（与 score_program.py 一致，对 score_time 做百分位归一化后划分）
  [0,25)  → poor
  [25,50) → medium
  [50,75) → good
  [75,100]→ strong

输出
----
  train_set/score_time_eval.json

用法
----
  python scripts/evaluate_score_vs_time.py
  python scripts/evaluate_score_vs_time.py \\
      --scores     train_set/scores.parquet \\
      --time-scores train_set/time_scores.parquet \\
      --output     train_set/score_time_eval.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

BAND_THRESHOLDS = [0, 25, 50, 75, 100]
BAND_LABELS = ["poor", "medium", "good", "strong"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scores",
        default=str(REPO_ROOT / "train_set" / "scores.parquet"),
        help="score_program.py 输出的 scores.parquet",
    )
    p.add_argument(
        "--time-scores",
        default=str(REPO_ROOT / "train_set" / "time_scores.parquet"),
        help="build_time_score_table.py 输出的 time_scores.parquet",
    )
    p.add_argument(
        "--output",
        default=str(REPO_ROOT / "train_set" / "score_time_eval.json"),
        help="输出 JSON 路径",
    )
    p.add_argument(
        "--min-active-pids",
        type=int,
        default=5,
        help="过滤 active_pid_count 过低的行（默认 5；这些行对应程序几乎未运行，指标不可靠）",
    )
    return p.parse_args()


def _to_band(scores_100: pd.Series) -> pd.Series:
    """将 0-100 分数映射为档位字符串。"""
    def _label(v: float) -> str:
        if v < 25:
            return "poor"
        elif v < 50:
            return "medium"
        elif v < 75:
            return "good"
        else:
            return "strong"

    return scores_100.apply(_label)


def _normalize_to_100(series: pd.Series, ref_min: float, ref_max: float) -> pd.Series:
    """使用给定的 ref 范围将 score_log 系列映射到 0-100。"""
    span = ref_max - ref_min
    if span < 1e-9:
        return pd.Series(50.0, index=series.index)
    return ((series - ref_min) / span * 100).clip(0, 100)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    r, _ = stats.pearsonr(a, b)
    return float(r)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    r, _ = stats.spearmanr(a, b)
    return float(r)


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _dir_acc(pred: np.ndarray, ref: np.ndarray) -> float:
    """sign 方向一致率（排除 ref ≈ 0 的行）。"""
    mask = np.abs(ref) > 1e-6
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(pred[mask]) == np.sign(ref[mask])))


def main() -> None:
    args = parse_args()
    scores_path = pathlib.Path(args.scores)
    time_scores_path = pathlib.Path(args.time_scores)
    output_path = pathlib.Path(args.output)

    # ── 检查输入 ───────────────────────────────────────────────────────────────
    for p in [scores_path, time_scores_path]:
        if not p.exists():
            print(f"[error] 输入文件不存在: {p}", file=sys.stderr)
            print("        请先运行 score_program.py 和 build_time_score_table.py",
                  file=sys.stderr)
            sys.exit(1)

    # ── 读取数据 ───────────────────────────────────────────────────────────────
    scores = pd.read_parquet(scores_path)
    ts = pd.read_parquet(time_scores_path)

    print(f"[info] scores.parquet      : {len(scores)} 行", flush=True)
    print(f"[info] time_scores.parquet : {len(ts)} 行", flush=True)

    # ── 合并 ───────────────────────────────────────────────────────────────────
    merged = scores.merge(
        ts[["program", "variant", "score_time", "time_per_iter", "active_pid_count"]],
        on=["program", "variant"],
        how="inner",
    )
    merged = merged.dropna(subset=["score_time", "score_log", "score_gt"])
    n_valid = len(merged)
    print(f"[info] 合并后有效行数: {n_valid}", flush=True)

    if n_valid < 2:
        print("[warn] 样本太少，无法计算相关性", file=sys.stderr)
        sys.exit(1)

    # ── 可靠性过滤 ─────────────────────────────────────────────────────────────
    # active_pid_count 过低表示该程序在采集窗口内几乎没有运行（可能崩溃/超慢），
    # 此时 cycles_per_iter 和 time_per_iter 均不可信。
    min_pids = args.min_active_pids
    reliable_mask = merged["active_pid_count"] >= min_pids
    n_excluded = int((~reliable_mask).sum())
    if n_excluded:
        excluded_progs = merged.loc[~reliable_mask, ["program", "variant", "active_pid_count"]]
        print(f"[warn] 过滤 {n_excluded} 行（active_pid_count < {min_pids}）:", flush=True)
        print(excluded_progs.to_string(index=False), flush=True)
    merged_clean = merged[reliable_mask].copy()
    n_clean = len(merged_clean)
    print(f"[info] 过滤后有效行数: {n_clean}", flush=True)

    # ── 计算 0-100 归一化（使用 score_gt 的全局分布作为参考尺度） ──────────────
    gt_vals = merged_clean["score_gt"].values
    ref_min = float(gt_vals.min())
    ref_max = float(gt_vals.max())

    merged_clean = merged_clean.copy()
    merged_clean["score_100_proxy"]  = _normalize_to_100(merged_clean["score_gt"],   ref_min, ref_max)
    merged_clean["score_100_model"]  = _normalize_to_100(merged_clean["score_log"],  ref_min, ref_max)
    merged_clean["score_100_time"]   = _normalize_to_100(merged_clean["score_time"], ref_min, ref_max)

    merged_clean["band_proxy"] = _to_band(merged_clean["score_100_proxy"])
    merged_clean["band_model"] = _to_band(merged_clean["score_100_model"])
    merged_clean["band_time"]  = _to_band(merged_clean["score_100_time"])

    # ── 计算指标 ───────────────────────────────────────────────────────────────
    gt   = merged_clean["score_gt"].values.astype(float)
    pred = merged_clean["score_log"].values.astype(float)
    time = merged_clean["score_time"].values.astype(float)

    # 同时对全量（含不可靠行）计算一次，便于对比
    gt_all   = merged["score_gt"].values.astype(float)
    pred_all = merged["score_log"].values.astype(float)
    time_all = merged["score_time"].values.astype(float)

    results: dict[str, object] = {
        "n_valid": n_valid,
        "n_excluded": n_excluded,
        "n_clean": n_clean,
        "min_active_pids_threshold": min_pids,
        # ── 过滤后（可靠行）──────────────────────────────────────────────
        # 代理标签 vs 时间参考
        "mae_proxy_time":   round(_mae(gt, time), 6),
        "corr_proxy_time":  round(_pearson(gt, time), 6),
        "spearman_proxy":   round(_spearman(gt, time), 6),
        "dir_acc_proxy":    round(_dir_acc(gt, time), 6),
        "band_acc_proxy":   round(
            float((merged_clean["band_proxy"] == merged_clean["band_time"]).mean()), 6
        ),
        # 模型分数 vs 时间参考
        "mae_model_time":   round(_mae(pred, time), 6),
        "corr_model_time":  round(_pearson(pred, time), 6),
        "spearman_model":   round(_spearman(pred, time), 6),
        "dir_acc_model":    round(_dir_acc(pred, time), 6),
        "band_acc_model":   round(
            float((merged_clean["band_model"] == merged_clean["band_time"]).mean()), 6
        ),
        # ── 全量（含不可靠行，供参考） ─────────────────────────────────
        "unfiltered": {
            "n": n_valid,
            "mae_proxy_time":  round(_mae(gt_all, time_all), 6),
            "corr_proxy_time": round(_pearson(gt_all, time_all), 6),
            "spearman_proxy":  round(_spearman(gt_all, time_all), 6),
            "mae_model_time":  round(_mae(pred_all, time_all), 6),
            "corr_model_time": round(_pearson(pred_all, time_all), 6),
            "spearman_model":  round(_spearman(pred_all, time_all), 6),
        },
        # 参考尺度
        "score_time_mean": round(float(np.mean(time)), 6),
        "score_time_std":  round(float(np.std(time)), 6),
        "ref_min_gt":      round(ref_min, 6),
        "ref_max_gt":      round(ref_max, 6),
    }

    # ── 打印摘要 ───────────────────────────────────────────────────────────────
    print()
    print("═" * 58)
    print("  时间评分外部验证结果")
    print(f"  （可靠行 {n_clean}/{n_valid}，已排除 active_pid_count < {min_pids} 的行）")
    print("═" * 58)
    print(f"  {'代理标签 (score_gt) vs score_time':36s}")
    print(f"    MAE          = {results['mae_proxy_time']:.4f}")
    print(f"    Pearson r    = {results['corr_proxy_time']:.4f}")
    print(f"    Spearman ρ   = {results['spearman_proxy']:.4f}")
    print(f"    方向一致率   = {results['dir_acc_proxy']:.4f}")
    print(f"    档位一致率   = {results['band_acc_proxy']:.4f}")
    print()
    print(f"  {'模型分数 (score_log) vs score_time':36s}")
    print(f"    MAE          = {results['mae_model_time']:.4f}")
    print(f"    Pearson r    = {results['corr_model_time']:.4f}")
    print(f"    Spearman ρ   = {results['spearman_model']:.4f}")
    print(f"    方向一致率   = {results['dir_acc_model']:.4f}")
    print(f"    档位一致率   = {results['band_acc_model']:.4f}")
    print()
    uf = results["unfiltered"]
    print("  全量（含不可靠行，供参考）:")
    print(f"    proxy: corr={uf['corr_proxy_time']:.4f}  spearman={uf['spearman_proxy']:.4f}")
    print(f"    model: corr={uf['corr_model_time']:.4f}  spearman={uf['spearman_model']:.4f}")
    print("═" * 58)

    # ── 写入文件 ───────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[done] 已写入 {output_path}", flush=True)


if __name__ == "__main__":
    main()
