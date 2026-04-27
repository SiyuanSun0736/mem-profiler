#!/usr/bin/env python3
"""
build_time_score_table.py — 基于 wall_time_sec 构建时间评分基准表
=================================================================

原理
----
采集脚本使用 while-true 循环，让短命程序持续重建。在 60 s 窗口内：

  time_per_iter(k)  =  wall_time_sec / active_pid_count
                     ≈  单次迭代平均挂钟时间（秒）

其中 active_pid_count 与 cycles_per_iter 的分母相同，保证两者量纲一致。

时间评分（参考 O0 基准）：
  score_time(k) = log( time_per_iter_O0 / time_per_iter_k )
                > 0 → 比 O0 快（优化有效）
                ≈ 0 → 与 O0 相当
                < 0 → 比 O0 慢（退化）

当程序缺少 O0 或 active_pid_count ≤ 0 时该程序从表中排除。

输出
----
  train_set/time_scores.parquet   — 含 time_per_iter, score_time 的评分表

用法
----
  python scripts/build_time_score_table.py
  python scripts/build_time_score_table.py --input train_set/run_features.parquet
  python scripts/build_time_score_table.py --baseline O0 --output train_set/time_scores.parquet
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

BASELINE_VARIANT = "O0"
DEFAULT_MIN_ACTIVE_PIDS = 5
DEFAULT_MIN_ACTIVE_WINDOW_RATIO = 0.10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        default=str(REPO_ROOT / "train_set" / "run_features.parquet"),
        help="run_features.parquet 路径",
    )
    p.add_argument(
        "--baseline",
        default=BASELINE_VARIANT,
        help="用作基线的变体名称（默认 O0）",
    )
    p.add_argument(
        "--output",
        default=str(REPO_ROOT / "train_set" / "time_scores.parquet"),
        help="输出 parquet 路径",
    )
    p.add_argument(
        "--summary-json",
        default=str(REPO_ROOT / "train_set" / "time_score_filter_summary.json"),
        help="输出严格时间真值过滤摘要 JSON 路径",
    )
    p.add_argument(
        "--min-active-pids",
        type=int,
        default=DEFAULT_MIN_ACTIVE_PIDS,
        help="构造严格时间真值时要求的最小 active_pid_count（默认 5）",
    )
    p.add_argument(
        "--min-active-window-ratio",
        type=float,
        default=DEFAULT_MIN_ACTIVE_WINDOW_RATIO,
        help="构造严格时间真值时要求的最小 active_window_count / window_count（默认 0.10）",
    )
    return p.parse_args()


def _safe_div(numer: float, denom: float) -> float:
    if denom <= 0:
        return float("nan")
    return float(numer / denom)


def _score_time_from_pair(t_base: float, t_variant: float) -> float:
    if not (math.isfinite(t_variant) and t_variant > 0 and math.isfinite(t_base) and t_base > 0):
        return float("nan")
    return math.log(t_base / t_variant)


def _strict_invalid_reasons(
    row: pd.Series,
    min_active_pids: int,
    min_active_window_ratio: float,
) -> list[str]:
    reasons: list[str] = []
    if int(row.get("active_pid_count", 0) or 0) < min_active_pids:
        reasons.append("low_active_pid_count")
    if float(row.get("active_window_ratio", 0.0) or 0.0) < min_active_window_ratio:
        reasons.append("low_active_window_ratio")
    return reasons


def build_time_scores(
    rf: pd.DataFrame,
    baseline: str,
    min_active_pids: int,
    min_active_window_ratio: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    required = {"program", "variant", "wall_time_sec", "active_pid_count"}
    missing = required - set(rf.columns)
    if missing:
        raise ValueError(f"run_features 缺少必要列: {missing}")

    rf = rf.copy()
    ratio_cols_present = {"window_count", "active_window_count"}.issubset(rf.columns)
    if ratio_cols_present:
        rf["active_window_ratio"] = rf.apply(
            lambda r: _safe_div(r["active_window_count"], r["window_count"]),
            axis=1,
        ).fillna(0.0)
    else:
        # 兼容旧版 run_features：若缺少窗口统计，则退化为不按该项过滤。
        rf["active_window_ratio"] = 1.0

    rf["time_per_iter"] = rf.apply(
        lambda r: _safe_div(r["wall_time_sec"], r["active_pid_count"]),
        axis=1,
    )

    loose_base_rows = rf[rf["variant"] == baseline][["program", "time_per_iter"]].copy()
    loose_base_rows = loose_base_rows.dropna(subset=["time_per_iter"])
    loose_base_rows = loose_base_rows.rename(columns={"time_per_iter": "time_per_iter_base_loose"})
    loose_base_map = loose_base_rows.groupby("program")["time_per_iter_base_loose"].mean()
    rf["has_loose_baseline"] = rf["program"].isin(set(loose_base_map.index))
    rf["time_per_iter_base_loose"] = rf["program"].map(loose_base_map)
    rf["score_time_loose"] = rf.apply(
        lambda row: _score_time_from_pair(row["time_per_iter_base_loose"], row["time_per_iter"]),
        axis=1,
    )

    rf["time_score_invalid_reasons"] = rf.apply(
        lambda row: _strict_invalid_reasons(
            row,
            min_active_pids=min_active_pids,
            min_active_window_ratio=min_active_window_ratio,
        ),
        axis=1,
    )
    rf["time_score_input_ok"] = rf["time_score_invalid_reasons"].apply(lambda reasons: len(reasons) == 0)

    strict_base_rows = rf[
        (rf["variant"] == baseline) & rf["time_score_input_ok"]
    ][["program", "time_per_iter"]].copy()
    strict_base_rows = strict_base_rows.dropna(subset=["time_per_iter"])
    strict_base_rows = strict_base_rows.rename(columns={"time_per_iter": "time_per_iter_base"})
    strict_base_map = strict_base_rows.groupby("program")["time_per_iter_base"].mean()

    rf["has_strict_baseline"] = rf["program"].isin(set(strict_base_map.index))
    rf["time_per_iter_base"] = rf["program"].map(strict_base_map)
    rf["score_time"] = rf.apply(
        lambda row: _score_time_from_pair(row["time_per_iter_base"], row["time_per_iter"])
        if row["time_score_input_ok"] and row["has_strict_baseline"]
        else float("nan"),
        axis=1,
    )
    rf["time_score_strict_ok"] = rf["score_time"].notna()
    rf["time_score_invalid_reasons"] = rf["time_score_invalid_reasons"].apply(
        lambda reasons: "|".join(reasons)
    )

    reason_counts = {
        "low_active_pid_count": int(
            (rf["active_pid_count"].fillna(0).astype(int) < min_active_pids).sum()
        ),
        "low_active_window_ratio": int((rf["active_window_ratio"] < min_active_window_ratio).sum()),
        "missing_strict_baseline": int(
            (rf["time_score_input_ok"] & ~rf["has_strict_baseline"]).sum()
        ),
    }

    summary: dict[str, object] = {
        "baseline": baseline,
        "min_active_pids": int(min_active_pids),
        "min_active_window_ratio": float(min_active_window_ratio),
        "active_window_ratio_available": bool(ratio_cols_present),
        "n_seen": int(len(rf)),
        "n_programs_seen": int(rf["program"].nunique()),
        "n_input_ok": int(rf["time_score_input_ok"].sum()),
        "n_input_filtered": int((~rf["time_score_input_ok"]).sum()),
        "n_programs_with_loose_baseline": int(len(loose_base_map)),
        "n_programs_with_strict_baseline": int(len(strict_base_map)),
        "n_valid_loose": int(rf["score_time_loose"].notna().sum()),
        "n_valid_strict": int(rf["score_time"].notna().sum()),
        "reasons": reason_counts,
        "by_variant": {},
    }

    example_cols = [
        "program",
        "variant",
        "active_pid_count",
        "active_window_ratio",
        "wall_time_sec",
        "output_dir",
        "time_score_invalid_reasons",
    ]
    available_example_cols = [c for c in example_cols if c in rf.columns]
    filtered_examples = rf[~rf["time_score_input_ok"]][available_example_cols].head(10).copy()
    if "active_window_ratio" in filtered_examples.columns:
        filtered_examples["active_window_ratio"] = filtered_examples["active_window_ratio"].round(6)
    if "wall_time_sec" in filtered_examples.columns:
        filtered_examples["wall_time_sec"] = filtered_examples["wall_time_sec"].round(6)
    summary["examples"] = filtered_examples.to_dict(orient="records")

    for variant, sub in rf.groupby("variant", sort=True):
        summary["by_variant"][str(variant)] = {
            "seen": int(len(sub)),
            "input_ok": int(sub["time_score_input_ok"].sum()),
            "input_filtered": int((~sub["time_score_input_ok"]).sum()),
            "valid_loose": int(sub["score_time_loose"].notna().sum()),
            "valid_strict": int(sub["score_time"].notna().sum()),
        }

    keep_cols = [
        "program",
        "variant",
        "wall_time_sec",
        "active_pid_count",
        "active_window_ratio",
        "time_per_iter",
        "time_per_iter_base_loose",
        "score_time_loose",
        "time_score_input_ok",
        "has_loose_baseline",
        "has_strict_baseline",
        "time_score_invalid_reasons",
        "time_per_iter_base",
        "score_time",
        "time_score_strict_ok",
    ]
    out = rf[[c for c in keep_cols if c in rf.columns]].copy()
    out = out.sort_values(["program", "variant"]).reset_index(drop=True)
    return out, summary


def main() -> None:
    args = parse_args()
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    summary_path = pathlib.Path(args.summary_json)

    if not input_path.exists():
        print(f"[error] 输入文件不存在: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[info] 读取 {input_path.name} ...", flush=True)
    rf = pd.read_parquet(input_path)
    print(f"       {len(rf)} 行，{rf['program'].nunique()} 个程序，"
          f"{rf['variant'].nunique()} 个变体", flush=True)

    out, summary = build_time_scores(
        rf,
        baseline=args.baseline,
        min_active_pids=args.min_active_pids,
        min_active_window_ratio=args.min_active_window_ratio,
    )

    n_valid_loose = int(out["score_time_loose"].notna().sum())
    n_valid = int(out["score_time"].notna().sum())
    n_total = len(out)
    n_prog  = out["program"].nunique()
    print(
        f"[info] 时间评分: loose={n_valid_loose}/{n_total}，strict={n_valid}/{n_total} 行有效"
        f"（{n_prog} 个程序）",
        flush=True,
    )
    print(
        f"       strict 门槛: active_pid_count >= {args.min_active_pids}, "
        f"active_window_ratio >= {args.min_active_window_ratio:.3f}",
        flush=True,
    )

    if n_valid > 0:
        valid = out.dropna(subset=["score_time"])
        print(f"       score_time 均值={valid['score_time'].mean():.4f}"
              f"  std={valid['score_time'].std():.4f}"
              f"  [min={valid['score_time'].min():.3f}, max={valid['score_time'].max():.3f}]",
              flush=True)

    print(
        f"[info] 严格过滤摘要: 输入过滤 {summary['n_input_filtered']} 行，"
        f"缺失严格基线 {summary['reasons']['missing_strict_baseline']} 行",
        flush=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    print(f"[done] 已写入 {output_path}", flush=True)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[done] 已写入 {summary_path}", flush=True)


if __name__ == "__main__":
    main()
