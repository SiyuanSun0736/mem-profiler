#!/usr/bin/env python3
"""
evaluate_score_vs_time.py — 模型分数对时间评分的外部验证
=========================================================

验证维度
--------
1. 代理标签 vs 严格时间参考
      proxy label (score_gt) vs strict score_time
    衡量 cycles_per_iter 作为 wall-time 代理的可靠性

2. 模型分数 vs 严格时间参考
      model score (score_log) vs strict score_time
    衡量模型在真实时间维度的泛化性

3. loose vs strict 对照
    若 time_scores.parquet 同时提供 `score_time_loose` 与严格 `score_time`，
    则同时输出过滤前后对照指标，直接观察时间真值过滤是否提升一致性。

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
    n_valid           : 参与 loose 对照统计的 (program, variant) 数量
    n_clean           : 参与 strict 主统计的 (program, variant) 数量

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
        help="兼容旧版 time_scores.parquet 的回退过滤门槛；若文件已包含严格 score_time，则该参数不参与主流程",
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


def _build_metrics_block(df: pd.DataFrame, time_col: str) -> dict[str, float]:
    gt = df["score_gt"].values.astype(float)
    pred = df["score_log"].values.astype(float)
    time = df[time_col].values.astype(float)

    ref_min = float(gt.min())
    ref_max = float(gt.max())

    scored = df.copy()
    scored["score_100_proxy"] = _normalize_to_100(scored["score_gt"], ref_min, ref_max)
    scored["score_100_model"] = _normalize_to_100(scored["score_log"], ref_min, ref_max)
    scored["score_100_time"] = _normalize_to_100(scored[time_col], ref_min, ref_max)

    scored["band_proxy"] = _to_band(scored["score_100_proxy"])
    scored["band_model"] = _to_band(scored["score_100_model"])
    scored["band_time"] = _to_band(scored["score_100_time"])

    return {
        "mae_proxy_time": round(_mae(gt, time), 6),
        "corr_proxy_time": round(_pearson(gt, time), 6),
        "spearman_proxy": round(_spearman(gt, time), 6),
        "dir_acc_proxy": round(_dir_acc(gt, time), 6),
        "band_acc_proxy": round(
            float((scored["band_proxy"] == scored["band_time"]).mean()), 6
        ),
        "mae_model_time": round(_mae(pred, time), 6),
        "corr_model_time": round(_pearson(pred, time), 6),
        "spearman_model": round(_spearman(pred, time), 6),
        "dir_acc_model": round(_dir_acc(pred, time), 6),
        "band_acc_model": round(
            float((scored["band_model"] == scored["band_time"]).mean()), 6
        ),
        "score_time_mean": round(float(np.mean(time)), 6),
        "score_time_std": round(float(np.std(time)), 6),
        "ref_min_gt": round(ref_min, 6),
        "ref_max_gt": round(ref_max, 6),
    }


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
    merge_cols = [
        "program",
        "variant",
        "score_time",
        "time_per_iter",
        "active_pid_count",
        "active_window_ratio",
        "time_score_input_ok",
        "has_strict_baseline",
        "time_score_invalid_reasons",
    ]
    if "score_time_loose" in ts.columns:
        merge_cols.append("score_time_loose")

    merged = scores.merge(
        ts[[c for c in merge_cols if c in ts.columns]],
        on=["program", "variant"],
        how="inner",
    )

    loose_time_col = "score_time_loose" if "score_time_loose" in merged.columns else "score_time"
    merged_loose = merged.dropna(subset=[loose_time_col, "score_log", "score_gt"]).copy()
    n_valid = len(merged_loose)
    print(f"[info] loose 对照有效行数: {n_valid}", flush=True)

    if n_valid < 2:
        print("[warn] 样本太少，无法计算相关性", file=sys.stderr)
        sys.exit(1)

    if "score_time_loose" not in merged.columns:
        min_pids = args.min_active_pids
        strict_mask = merged_loose["active_pid_count"] >= min_pids
        merged_clean = merged_loose[strict_mask].copy()
        n_excluded = int((~strict_mask).sum())
        strict_reason_counts = {
            "low_active_pid_count": n_excluded,
            "low_active_window_ratio": 0,
            "missing_strict_baseline": 0,
        }
        strict_filter_mode = "legacy-active-pids-fallback"
        print(
            f"[warn] time_scores.parquet 缺少 strict 列，回退到 active_pid_count >= {min_pids} 过滤",
            flush=True,
        )
    else:
        merged_clean = merged.dropna(subset=["score_time", "score_log", "score_gt"]).copy()
        n_excluded = int(n_valid - len(merged_clean))
        strict_excluded = merged_loose[merged_loose["score_time"].isna()].copy()
        reasons = strict_excluded.get("time_score_invalid_reasons")
        strict_reason_counts = {
            "low_active_pid_count": int(
                reasons.fillna("").str.contains("low_active_pid_count").sum()
            ) if reasons is not None else 0,
            "low_active_window_ratio": int(
                reasons.fillna("").str.contains("low_active_window_ratio").sum()
            ) if reasons is not None else 0,
            "missing_strict_baseline": int(
                (
                    strict_excluded.get("time_score_input_ok", pd.Series(False, index=strict_excluded.index))
                    & ~strict_excluded.get("has_strict_baseline", pd.Series(False, index=strict_excluded.index))
                ).sum()
            ),
        }
        strict_filter_mode = "strict-time-score"

    n_clean = len(merged_clean)
    print(f"[info] strict 主统计有效行数: {n_clean}", flush=True)

    if n_clean < 2:
        print("[warn] strict 时间真值样本太少，无法计算主统计", file=sys.stderr)
        sys.exit(1)

    strict_metrics = _build_metrics_block(merged_clean, "score_time")
    loose_metrics = _build_metrics_block(merged_loose, loose_time_col)

    results: dict[str, object] = {
        "n_valid": n_valid,
        "n_valid_loose": n_valid,
        "n_excluded": n_excluded,
        "n_clean": n_clean,
        "n_valid_strict": n_clean,
        "strict_filter_mode": strict_filter_mode,
        "min_active_pids_threshold": args.min_active_pids,
        "loose_time_column": loose_time_col,
        "strict_filter": {
            "n_excluded_from_loose": n_excluded,
            "reasons": strict_reason_counts,
        },
        # ── strict 主统计 ────────────────────────────────────────────────
        "mae_proxy_time": strict_metrics["mae_proxy_time"],
        "corr_proxy_time": strict_metrics["corr_proxy_time"],
        "spearman_proxy": strict_metrics["spearman_proxy"],
        "dir_acc_proxy": strict_metrics["dir_acc_proxy"],
        "band_acc_proxy": strict_metrics["band_acc_proxy"],
        "mae_model_time": strict_metrics["mae_model_time"],
        "corr_model_time": strict_metrics["corr_model_time"],
        "spearman_model": strict_metrics["spearman_model"],
        "dir_acc_model": strict_metrics["dir_acc_model"],
        "band_acc_model": strict_metrics["band_acc_model"],
        # 参考尺度
        "score_time_mean": strict_metrics["score_time_mean"],
        "score_time_std": strict_metrics["score_time_std"],
        "ref_min_gt": strict_metrics["ref_min_gt"],
        "ref_max_gt": strict_metrics["ref_max_gt"],
        # ── loose 对照 ─────────────────────────────────────────────────
        "unfiltered": {
            "n": n_valid,
            "mae_proxy_time": loose_metrics["mae_proxy_time"],
            "corr_proxy_time": loose_metrics["corr_proxy_time"],
            "spearman_proxy": loose_metrics["spearman_proxy"],
            "dir_acc_proxy": loose_metrics["dir_acc_proxy"],
            "band_acc_proxy": loose_metrics["band_acc_proxy"],
            "mae_model_time": loose_metrics["mae_model_time"],
            "corr_model_time": loose_metrics["corr_model_time"],
            "spearman_model": loose_metrics["spearman_model"],
            "dir_acc_model": loose_metrics["dir_acc_model"],
            "band_acc_model": loose_metrics["band_acc_model"],
            "score_time_mean": loose_metrics["score_time_mean"],
            "score_time_std": loose_metrics["score_time_std"],
        },
    }

    # ── 打印摘要 ───────────────────────────────────────────────────────────────
    print()
    print("═" * 58)
    print("  时间评分外部验证结果")
    print(f"  （strict 主统计 {n_clean}/{n_valid}；mode={strict_filter_mode}）")
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
    print("  strict 过滤摘要:")
    print(f"    从 loose 排除 = {results['strict_filter']['n_excluded_from_loose']}")
    print(
        f"    reasons       = {results['strict_filter']['reasons']}",
    )
    print()
    uf = results["unfiltered"]
    print("  loose 对照（过滤前）:")
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
