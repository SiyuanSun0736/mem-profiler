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
import math
import pathlib
import sys

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

BASELINE_VARIANT = "O0"


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
    return p.parse_args()


def build_time_scores(
    rf: pd.DataFrame,
    baseline: str,
) -> pd.DataFrame:
    required = {"program", "variant", "wall_time_sec", "active_pid_count"}
    missing = required - set(rf.columns)
    if missing:
        raise ValueError(f"run_features 缺少必要列: {missing}")

    # 计算 time_per_iter；active_pid_count ≤ 0 的行设为 NaN
    rf = rf.copy()
    rf["time_per_iter"] = rf.apply(
        lambda r: r["wall_time_sec"] / r["active_pid_count"]
        if r["active_pid_count"] > 0
        else float("nan"),
        axis=1,
    )

    # 构建 (program -> baseline time_per_iter) 映射
    base_rows = rf[rf["variant"] == baseline][["program", "time_per_iter"]].copy()
    base_rows = base_rows.dropna(subset=["time_per_iter"])
    base_rows = base_rows.rename(columns={"time_per_iter": "time_per_iter_base"})
    # 若同一 program 有多个 baseline row（通常只有一行），取均值
    base_map = base_rows.groupby("program")["time_per_iter_base"].mean()

    # 只保留有 baseline 的 program
    programs_with_base = set(base_map.index)
    rf = rf[rf["program"].isin(programs_with_base)].copy()

    rf["time_per_iter_base"] = rf["program"].map(base_map)

    # score_time = log( t_base / t_k )
    def _score_time(row: pd.Series) -> float:
        t_k = row["time_per_iter"]
        t_b = row["time_per_iter_base"]
        if not (math.isfinite(t_k) and t_k > 0 and math.isfinite(t_b) and t_b > 0):
            return float("nan")
        return math.log(t_b / t_k)

    rf["score_time"] = rf.apply(_score_time, axis=1)

    keep_cols = [
        "program",
        "variant",
        "wall_time_sec",
        "active_pid_count",
        "time_per_iter",
        "time_per_iter_base",
        "score_time",
    ]
    out = rf[[c for c in keep_cols if c in rf.columns]].copy()
    out = out.sort_values(["program", "variant"]).reset_index(drop=True)
    return out


def main() -> None:
    args = parse_args()
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)

    if not input_path.exists():
        print(f"[error] 输入文件不存在: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[info] 读取 {input_path.name} ...", flush=True)
    rf = pd.read_parquet(input_path)
    print(f"       {len(rf)} 行，{rf['program'].nunique()} 个程序，"
          f"{rf['variant'].nunique()} 个变体", flush=True)

    out = build_time_scores(rf, baseline=args.baseline)

    n_valid = out["score_time"].notna().sum()
    n_total = len(out)
    n_prog  = out["program"].nunique()
    print(f"[info] 时间评分: {n_valid}/{n_total} 行有效（{n_prog} 个程序）", flush=True)

    if n_valid > 0:
        valid = out.dropna(subset=["score_time"])
        print(f"       score_time 均值={valid['score_time'].mean():.4f}"
              f"  std={valid['score_time'].std():.4f}"
              f"  [min={valid['score_time'].min():.3f}, max={valid['score_time'].max():.3f}]",
              flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    print(f"[done] 已写入 {output_path}", flush=True)


if __name__ == "__main__":
    main()
