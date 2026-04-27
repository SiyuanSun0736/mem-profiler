#!/usr/bin/env python3
"""
build_pair_table.py — 构建成对样本表

从 run_features_zscore.parquet（模型输入）和 run_features.parquet（cycles 标签来源）：
  1. 枚举同一程序的全部 variant 对（O0 vs O1/O2/O3，O1 vs O2/O3，O2 vs O3）
  2. 正向对 + 反向对都加入（数据量翻倍）
  3. 以 cycles_per_iter 作为执行时间代理，计算 log(T_j/T_i) 作为回归标签

标签约定（与 04-model-plan.md 一致）：
  log_ratio = log(cycles_per_iter_j / cycles_per_iter_i)
    > +tie_thresh → i 更优（i 单次迭代用时更短）
    < -tie_thresh → j 更优（j 单次迭代用时更短）
    otherwise    → 持平

  注：cycles_per_iter = total_cycles / active_pid_count，是对单次迭代 cycles 的代理。
  采集脚本以 while-true 循环运行程序，每次迭代独立 PID，活跃 PID 数近似迭代次数。
  使用 per-iter 而非 total_cycles 的原因：O0 慢故迭代少，O3 快故迭代多，
  若不归一化则 total_cycles 可能出现 O3 > O0 的方向反转（O3 做了更多工作）。

用法：
    python scripts/build_pair_table.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import pathlib
import sys

import numpy as np
import pandas as pd

from feature_columns import DROPPED_INPUT_FEATURES, NON_TIME_COLS


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
VARIANTS = ["O0", "O1", "O2", "O3"]
_VARIANT_RANK = {v: i for i, v in enumerate(VARIANTS)}

TIE_THRESHOLD = 0.05  # |log(T_j/T_i)| < 0.05 视为持平

_CLASS_TO_INT = {"i_better": 0, "tie": 1, "j_better": 2}


def _label_class(log_ratio: float) -> str:
    if log_ratio > TIE_THRESHOLD:
        return "i_better"
    if log_ratio < -TIE_THRESHOLD:
        return "j_better"
    return "tie"


def build_pairs(
    df_z: pd.DataFrame,
    df_raw: pd.DataFrame,
    include_reverse: bool = True,
) -> pd.DataFrame:
    """
    构建成对样本。每条记录包含：
    - program, variant_i, variant_j, run_id_i, run_id_j
    - log_ratio, label_class, label_int
    - xi_{col}, xj_{col}, diff_{col} 各 len(NON_TIME_COLS) 列
    - variant_rank_diff: rank(j) - rank(i)（越大说明 j 优化程度名义上越高）
    """
    # 检查特征列是否存在
    missing = [c for c in NON_TIME_COLS if c not in df_z.columns]
    if missing:
        sys.exit(f"[错误] zscore 文件缺少特征列：{missing[:5]}...")

    # 构建快速查找字典
    z_map:   dict[tuple[str, str], pd.Series] = {}
    raw_map: dict[tuple[str, str], pd.Series] = {}
    for _, row in df_z.iterrows():
        z_map[(row["program"], row["variant"])] = row
    for _, row in df_raw.iterrows():
        raw_map[(row["program"], row["variant"])] = row

    programs = sorted(df_z["program"].unique())
    records: list[dict] = []

    for prog in programs:
        avail = [v for v in VARIANTS if (prog, v) in z_map and (prog, v) in raw_map]
        if len(avail) < 2:
            continue

        for vi, vj in itertools.combinations(avail, 2):
            xi_vals = np.array(
                [float(z_map[(prog, vi)].get(c, 0.0)) for c in NON_TIME_COLS],
                dtype=np.float32,
            )
            xj_vals = np.array(
                [float(z_map[(prog, vj)].get(c, 0.0)) for c in NON_TIME_COLS],
                dtype=np.float32,
            )

            cycles_i = max(float(raw_map[(prog, vi)].get("cycles_per_iter", 0)) or
                          float(raw_map[(prog, vi)].get("total_cycles", 1)), 1)
            cycles_j = max(float(raw_map[(prog, vj)].get("cycles_per_iter", 0)) or
                          float(raw_map[(prog, vj)].get("total_cycles", 1)), 1)
            log_ratio = math.log(cycles_j / cycles_i)

            def _make_row(
                xi: np.ndarray,
                xj: np.ndarray,
                lr: float,
                prog_: str,
                vi_: str,
                vj_: str,
            ) -> dict:
                cls = _label_class(lr)
                row: dict = {
                    "program":          prog_,
                    "variant_i":        vi_,
                    "variant_j":        vj_,
                    "run_id_i":         str(z_map[(prog_, vi_)].get("run_id", "")),
                    "run_id_j":         str(z_map[(prog_, vj_)].get("run_id", "")),
                    "log_ratio":        float(lr),
                    "label_class":      cls,
                    "label_int":        _CLASS_TO_INT[cls],
                    "variant_rank_diff": _VARIANT_RANK[vj_] - _VARIANT_RANK[vi_],
                }
                for k, col in enumerate(NON_TIME_COLS):
                    row[f"xi_{col}"]   = float(xi[k])
                    row[f"xj_{col}"]   = float(xj[k])
                    row[f"diff_{col}"] = float(xi[k] - xj[k])
                return row

            records.append(_make_row(xi_vals, xj_vals, log_ratio, prog, vi, vj))

            if include_reverse:
                records.append(
                    _make_row(xj_vals, xi_vals, -log_ratio, prog, vj, vi)
                )

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="构建成对样本表")
    parser.add_argument("--zscore", default="train_set/run_features_zscore.parquet")
    parser.add_argument("--raw",    default="train_set/run_features.parquet")
    parser.add_argument("--output", default="train_set")
    parser.add_argument("--no-reverse", action="store_true",
                        help="不生成反向对（默认生成）")
    args = parser.parse_args()

    zscore_path = (REPO_ROOT / args.zscore).resolve()
    raw_path    = (REPO_ROOT / args.raw).resolve()
    out_dir     = (REPO_ROOT / args.output).resolve()

    for p in [zscore_path, raw_path]:
        if not p.exists():
            sys.exit(f"[错误] 找不到 {p}，请先运行 build_run_features.py")

    df_z   = pd.read_parquet(zscore_path)
    df_raw = pd.read_parquet(raw_path)

    print(f"[info] z-score 特征：{len(df_z)} 条运行记录", flush=True)
    print(f"[info] 生成反向对：{not args.no_reverse}", flush=True)
    if DROPPED_INPUT_FEATURES:
        print(f"[info] 已从模型输入中剔除死特征: {', '.join(DROPPED_INPUT_FEATURES)}", flush=True)

    df_pairs = build_pairs(df_z, df_raw, include_reverse=not args.no_reverse)

    out_dir.mkdir(parents=True, exist_ok=True)
    df_pairs.to_parquet(out_dir / "pairs.parquet", index=False)
    df_pairs.to_csv(out_dir / "pairs.csv", index=False)

    # ── 统计摘要 ─────────────────────────────────────────────────────────
    n_total = len(df_pairs)
    label_counts = df_pairs["label_class"].value_counts().to_dict()

    # 各 variant 对的标签分布
    pair_label = (
        df_pairs.groupby(["variant_i", "variant_j", "label_class"])
        .size()
        .reset_index(name="count")
    )

    stats = {
        "n_pairs":       n_total,
        "n_programs":    int(df_pairs["program"].nunique()),
        "feature_dim":   len(NON_TIME_COLS),
        "input_dim":     len(NON_TIME_COLS) * 3,
        "tie_threshold": TIE_THRESHOLD,
        "label_counts":  {k: int(v) for k, v in label_counts.items()},
        "log_ratio_stats": {
            "mean":   round(float(df_pairs["log_ratio"].mean()), 4),
            "std":    round(float(df_pairs["log_ratio"].std()),  4),
            "min":    round(float(df_pairs["log_ratio"].min()),  4),
            "max":    round(float(df_pairs["log_ratio"].max()),  4),
            "p25":    round(float(df_pairs["log_ratio"].quantile(0.25)), 4),
            "median": round(float(df_pairs["log_ratio"].median()), 4),
            "p75":    round(float(df_pairs["log_ratio"].quantile(0.75)), 4),
        },
    }

    (out_dir / "pairs_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False)
    )

    print(f"\n[info] 共 {n_total} 条配对样本，覆盖 {stats['n_programs']} 个程序")
    print("[info] 标签分布：")
    for cls in ["i_better", "tie", "j_better"]:
        cnt = label_counts.get(cls, 0)
        print(f"  {cls:12s}: {cnt:5d}  ({100*cnt/n_total:.1f}%)")

    print("\n[info] log_ratio 分布：")
    lr = stats["log_ratio_stats"]
    print(f"  mean={lr['mean']}  std={lr['std']}  "
          f"[{lr['min']}, {lr['median']}, {lr['max']}]")

    print(f"\n[info] 各 variant 对的标签分布（前 12 行）：")
    print(pair_label.to_string(index=False))

    print(f"\n[ok] {out_dir / 'pairs.parquet'}")
    print(f"[ok] {out_dir / 'pairs.csv'}")
    print(f"[ok] {out_dir / 'pairs_stats.json'}")


if __name__ == "__main__":
    main()
