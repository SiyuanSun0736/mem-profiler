"""
hotspot.py — 热点识别与时间窗级分析

从 window_metrics.jsonl 聚合，按事件类型对函数/PID 排序，
输出热点摘要到 hotspot_summary.jsonl，并可选生成 CSV/图表。

用法：
    python analysis/hotspot.py \\
        --data data/run_001/ \\
        --output results/run_001/ \\
        [--top 20] [--metric llc_load_misses] [--pid 1234]
"""

import argparse
import json
import pathlib
import sys
import warnings
from itertools import combinations
from typing import Any

import pandas as pd


METRICS = [
    "llc_loads",
    "llc_load_misses",
    "llc_stores",
    "llc_store_misses",
    "dtlb_loads",
    "dtlb_load_misses",
    "dtlb_stores",
    "dtlb_store_misses",
    "dtlb_misses",
    "itlb_loads",
    "itlb_load_misses",
    "minor_faults",
    "major_faults",
    "lbr_samples",
    "lbr_entries",
    "samples",
]


def load_window_metrics(data_dir: pathlib.Path) -> pd.DataFrame:
    jsonl = data_dir / "window_metrics.jsonl"
    if not jsonl.exists():
        sys.exit(f"[错误] 找不到 {jsonl}")
    rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    return pd.DataFrame(rows)


def compute_pid_hotspot(
    df: pd.DataFrame,
    metric: str,
    top_n: int,
    target_pid: int | None,
) -> pd.DataFrame:
    """按 PID/TID 和 comm 聚合，返回按 metric 降序的 top_n 条记录。"""
    if target_pid:
        df = df[df["pid"] == target_pid]
    if metric not in df.columns:
        sys.exit(f"[错误] 指标 '{metric}' 不在数据中，可选: {METRICS}")

    group_cols = ["pid", "comm"]
    if "tid" in df.columns and df["tid"].notna().any():
        group_cols = ["pid", "tid", "comm"]

    agg = (
        df.groupby(group_cols)[[m for m in METRICS if m in df.columns]]
        .sum()
        .reset_index()
        .sort_values(metric, ascending=False)
        .head(top_n)
    )
    total = agg[metric].sum()
    agg["fraction"] = agg[metric] / total if total > 0 else 0.0
    return agg


def compute_time_series(
    df: pd.DataFrame,
    metric: str,
    pid: int | None,
) -> pd.DataFrame:
    """返回按 window_id 聚合的时间序列，便于绘制折线图。"""
    if pid:
        df = df[df["pid"] == pid]
    ts = (
        df.groupby("window_id")[metric]
        .sum()
        .reset_index()
        .rename(columns={metric: "value"})
    )
    ts["metric"] = metric
    return ts


# ── 时间窗级热点检测 ───────────────────────────────────────────────────────

HOTSPOT_METHODS = ("zscore", "iqr", "top_pct")


def compute_window_hotspots(
    df: pd.DataFrame,
    metric: str,
    method: str = "zscore",
    threshold: float = 2.0,
) -> pd.DataFrame:
    """
    识别指标异常高的时间窗口（热点窗口）。

    method:
      zscore  — z-score >= threshold 的窗口标记为热点（默认 threshold=2.0）
      iqr     — value > Q3 + threshold * IQR（默认 threshold=1.5）
      top_pct — value >= 第 (100-threshold) 百分位数（默认 threshold=10 → top 10%）

    返回 DataFrame（按 window_id 升序）：
      window_id, [start_ns, end_ns, duration_ms,] value, score, score_type, is_hot
    """
    if metric not in df.columns:
        sys.exit(f"[错误] 指标 '{metric}' 不在数据中，可选: {METRICS}")

    agg_dict: dict[str, Any] = {metric: "sum"}
    if "start_ns" in df.columns:
        agg_dict["start_ns"] = "min"
    if "end_ns" in df.columns:
        agg_dict["end_ns"] = "max"

    ts = (
        df.groupby("window_id")
        .agg(agg_dict)
        .reset_index()
        .rename(columns={metric: "value"})
        .sort_values("window_id")
    )

    if "start_ns" in ts.columns and "end_ns" in ts.columns:
        ts["duration_ms"] = (ts["end_ns"] - ts["start_ns"]) / 1e6

    if method == "zscore":
        mean = ts["value"].mean()
        std = ts["value"].std()
        ts["score"] = (ts["value"] - mean) / std if std > 0 else 0.0
        ts["is_hot"] = ts["score"] >= threshold
        ts["score_type"] = "z_score"
    elif method == "iqr":
        q1 = ts["value"].quantile(0.25)
        q3 = ts["value"].quantile(0.75)
        iqr = q3 - q1
        upper = q3 + threshold * iqr
        ts["score"] = (ts["value"] - q3) / iqr if iqr > 0 else 0.0
        ts["is_hot"] = ts["value"] > upper
        ts["score_type"] = "iqr_multiple"
    elif method == "top_pct":
        cutoff = ts["value"].quantile(1.0 - threshold / 100.0)
        ts["score"] = ts["value"].rank(pct=True)
        ts["is_hot"] = ts["value"] >= cutoff
        ts["score_type"] = "percentile_rank"
    else:
        sys.exit(f"[错误] 未知热点检测方法 '{method}'，可选: {HOTSPOT_METHODS}")

    return ts


def compute_window_attribution(
    df: pd.DataFrame,
    hot_window_ids: list[int],
    metric: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    对每个热点窗口，按 PID/comm 计算归因份额。

    返回 DataFrame（按 window_id 升序、count 降序，每窗口 top_n 条）：
      window_id, pid[, tid], comm, count, fraction
    """
    hot_df = df[df["window_id"].isin(hot_window_ids)]
    if hot_df.empty:
        return pd.DataFrame()

    group_cols = ["window_id", "pid", "comm"]
    if "tid" in df.columns and df["tid"].notna().any():
        group_cols = ["window_id", "pid", "tid", "comm"]

    agg = (
        hot_df.groupby(group_cols)[metric]
        .sum()
        .reset_index()
        .rename(columns={metric: "count"})
    )

    window_totals = agg.groupby("window_id")["count"].transform("sum")
    agg["fraction"] = agg["count"] / window_totals.where(window_totals > 0, 1.0)

    agg = (
        agg.sort_values(["window_id", "count"], ascending=[True, False])
        .groupby("window_id", group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return agg


def write_window_report(
    out_dir: pathlib.Path,
    run_id: str,
    ws: pd.DataFrame,
    wa: pd.DataFrame,
    metric: str,
) -> None:
    """
    写入 window_hotspots.jsonl / window_attribution.jsonl，
    并在控制台打印人可读的归因报告摘要。
    """
    # ── window_hotspots.jsonl ────────────────────────────────────────────
    hot_f = out_dir / "window_hotspots.jsonl"
    with open(hot_f, "w", encoding="utf-8") as fh:
        for _, row in ws.iterrows():
            rec: dict[str, Any] = {
                "schema_version": "1.0",
                "run_id":    run_id,
                "window_id": int(row["window_id"]),
                "metric":    metric,
                "value":     int(row["value"]),
                "score":     round(float(row["score"]), 4) if pd.notna(row["score"]) else None,
                "score_type": str(row["score_type"]),
                "is_hot":    bool(row["is_hot"]),
            }
            for ts_col in ("start_ns", "end_ns"):
                if ts_col in row.index and pd.notna(row[ts_col]):
                    rec[ts_col] = int(row[ts_col])
            if "duration_ms" in row.index and pd.notna(row["duration_ms"]):
                rec["duration_ms"] = round(float(row["duration_ms"]), 3)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[info] 窗口热点标注已写入 {hot_f}", flush=True)

    # ── window_attribution.jsonl ─────────────────────────────────────────
    if not wa.empty:
        attr_f = out_dir / "window_attribution.jsonl"
        with open(attr_f, "w", encoding="utf-8") as fh:
            for _, row in wa.iterrows():
                rec = {
                    "schema_version": "1.0",
                    "run_id":    run_id,
                    "window_id": int(row["window_id"]),
                    "metric":    metric,
                    "pid":       int(row["pid"]),
                    "comm":      str(row["comm"]),
                    "count":     int(row["count"]),
                    "fraction":  round(float(row["fraction"]), 4),
                }
                if "tid" in row.index and pd.notna(row["tid"]):
                    rec["tid"] = int(row["tid"])
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[info] 窗口归因已写入 {attr_f}", flush=True)

    # ── 控制台归因报告 ────────────────────────────────────────────────────
    hot_count = int(ws["is_hot"].sum())
    total     = len(ws)
    sep = "=" * 56
    print(f"\n{sep}")
    print(f"  时间窗热点归因报告   metric = {metric}")
    print(sep)
    print(f"  总窗口数   : {total}")
    print(f"  热点窗口数 : {hot_count}  ({100 * hot_count / total:.1f}%)")

    if hot_count == 0:
        print("  (未检测到热点窗口，可降低 --hotspot-threshold 重试)")
        print(sep + "\n")
        return

    hot_ws = ws[ws["is_hot"]].sort_values("value", ascending=False)
    disp_cols = ["window_id", "value", "score"]
    if "duration_ms" in hot_ws.columns and hot_ws["duration_ms"].notna().any():
        disp_cols.append("duration_ms")
    print(f"\n  Top 热点窗口 (按 value 降序):")
    print(hot_ws[disp_cols].head(10).to_string(index=False))

    if not wa.empty:
        per_win = int(wa.groupby("window_id").size().max())
        print(f"\n  热点窗口 PID 归因 (每窗口 Top {per_win}):")
        disp = ["window_id", "pid", "comm", "count", "fraction"]
        if "tid" in wa.columns:
            disp.insert(3, "tid")
        print(wa[disp].to_string(index=False))
    print(sep + "\n")


# ── 指标时序关系分析 ────────────────────────────────────────────────────────

def compute_metric_relations(
    df: pd.DataFrame,
    max_lag: int = 5,
    spike_zscore: float = 2.0,
) -> dict[str, Any]:
    """
    计算所有可用指标之间的时序关系，返回包含以下键的字典：

      ts          — 每窗口各指标汇总时间序列（window_id × metrics）
      corr        — 对称 Pearson 相关矩阵（pd.DataFrame）
      lagged      — 滞后互相关表（metric_a, metric_b, lag, correlation）
      co_spike    — 每窗口多指标同时热点标记（window_id, <m>_hot..., co_spike_count）
      pair_summary— 指标对摘要（按 |pearson_r| 降序）
      spike_zscore— 检测用 z-score 阈值（float）

    若可用指标少于 2 个（均为零或不存在）则返回空字典。
    """
    avail = [m for m in METRICS if m in df.columns and df[m].sum() > 0]
    if len(avail) < 2:
        return {}

    # 每窗口跨 PID 汇总
    ts_df = (
        df.groupby("window_id")[avail]
        .sum()
        .sort_index()
    )

    # 1. Pearson 相关矩阵
    corr_mat = ts_df.corr()

    # 2. 滞后互相关（metric_a, metric_b, lag ∈ [-max_lag, +max_lag]）
    n = len(ts_df)
    lagged_rows: list[dict] = []
    for a, b in combinations(avail, 2):
        sa = ts_df[a]
        sb = ts_df[b]
        for lag in range(-max_lag, max_lag + 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if lag == 0:
                    r = sa.corr(sb)
                elif lag > 0:
                    if lag >= n:
                        continue
                    r = sa.iloc[lag:].corr(sb.iloc[:-lag])
                else:
                    k = -lag
                    if k >= n:
                        continue
                    r = sa.iloc[:-k].corr(sb.iloc[k:])
            if pd.notna(r):
                lagged_rows.append({
                    "metric_a": a, "metric_b": b,
                    "lag": lag, "correlation": round(float(r), 4),
                })
    lagged_df = pd.DataFrame(lagged_rows)

    # 3. 联合热点检测（z-score 归一化后按阈值标记）
    std = ts_df.std().replace(0.0, 1.0)
    z_df = (ts_df - ts_df.mean()) / std
    hot_df = (z_df >= spike_zscore).astype(int)
    hot_df.columns = pd.Index([f"{m}_hot" for m in avail])
    hot_df["co_spike_count"] = hot_df.sum(axis=1)
    co_spike_df = hot_df.reset_index()

    # 4. 指标对汇总
    pair_rows: list[dict] = []
    for a, b in combinations(avail, 2):
        sub = lagged_df[(lagged_df["metric_a"] == a) & (lagged_df["metric_b"] == b)]
        if sub.empty:
            continue
        best = sub.loc[sub["correlation"].abs().idxmax()]
        co_count = int(
            (co_spike_df[f"{a}_hot"].astype(bool) & co_spike_df[f"{b}_hot"].astype(bool)).sum()
        )
        pair_rows.append({
            "metric_a":       a,
            "metric_b":       b,
            "pearson_r":      round(float(corr_mat.loc[a, b]), 4),
            "peak_lag":       int(best["lag"]),
            "peak_lag_corr":  round(float(best["correlation"]), 4),
            "co_spike_count": co_count,
        })
    pair_df = (
        pd.DataFrame(pair_rows)
        .sort_values("pearson_r", ascending=False, key=lambda s: s.abs())
    )

    return {
        "ts":           ts_df.reset_index(),
        "corr":         corr_mat,
        "lagged":       lagged_df,
        "co_spike":     co_spike_df,
        "pair_summary": pair_df,
        "spike_zscore": spike_zscore,
    }


def write_metric_relation_report(
    out_dir: pathlib.Path,
    run_id: str,
    mr: dict[str, Any],
) -> None:
    """
    写出指标时序关系分析结果：
      metric_lagged_corr.csv     — 所有指标对的滞后互相关序列
      co_spike_windows.csv       — 每窗口多指标热点标记
      metric_pair_summary.csv    — 指标对摘要（含 pearson_r / peak_lag）
      metric_relations.jsonl     — 结构化指标对摘要

    同时在控制台打印人可读报告。
    """
    if not mr:
        print("[info] 可用指标少于两个，跳过指标时序关系分析", flush=True)
        return

    mr["lagged"].to_csv(out_dir / "metric_lagged_corr.csv", index=False)
    mr["co_spike"].to_csv(out_dir / "co_spike_windows.csv", index=False)
    mr["pair_summary"].to_csv(out_dir / "metric_pair_summary.csv", index=False)

    rel_f = out_dir / "metric_relations.jsonl"
    with open(rel_f, "w", encoding="utf-8") as fh:
        for _, row in mr["pair_summary"].iterrows():
            rec: dict[str, Any] = {
                "schema_version": "1.0",
                "run_id":         run_id,
                "metric_a":       row["metric_a"],
                "metric_b":       row["metric_b"],
                "pearson_r":      float(row["pearson_r"]),
                "peak_lag":       int(row["peak_lag"]),
                "peak_lag_corr":  float(row["peak_lag_corr"]),
                "co_spike_count": int(row["co_spike_count"]),
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[info] 指标时序关系已写入 {rel_f}", flush=True)

    # ── 控制台报告 ────────────────────────────────────────────────────────
    sep = "=" * 60
    spike_thresh = mr.get("spike_zscore", 2.0)
    print(f"\n{sep}")
    print("  指标时序关系分析报告")
    print(sep)

    ps = mr["pair_summary"]
    print("\n  全指标对相关性（|pearson_r| 降序）:")
    print(ps[["metric_a", "metric_b", "pearson_r", "peak_lag",
               "peak_lag_corr", "co_spike_count"]].to_string(index=False))

    lagged_pairs = ps[ps["peak_lag"] != 0]
    if not lagged_pairs.empty:
        print("\n  存在滞后关系的指标对（peak_lag ≠ 0）:")
        print("  lag > 0 表示 metric_b 超前 metric_a；lag < 0 表示 metric_a 超前 metric_b")
        print(lagged_pairs[["metric_a", "metric_b", "pearson_r",
                              "peak_lag", "peak_lag_corr"]].head(10).to_string(index=False))

    co = mr["co_spike"]
    co_max = co["co_spike_count"].max()
    if co_max >= 2:
        multi_hot = co[co["co_spike_count"] >= 2]
        print(f"\n  多指标同时热点窗口（≥2 指标同时 z≥{spike_thresh}，共 {len(multi_hot)} 个窗口）:")
        print(multi_hot[["window_id", "co_spike_count"]].to_string(index=False))
    else:
        print(f"\n  未发现多指标同时热点窗口（z≥{spike_thresh}）")

    print(sep + "\n")


def write_hotspot_summary(
    out_dir: pathlib.Path,
    run_id: str,
    agg: pd.DataFrame,
    metric: str,
) -> None:
    out_f = out_dir / "hotspot_summary.jsonl"
    with open(out_f, "a", encoding="utf-8") as f:
        for _, row in agg.iterrows():
            rec = {
                "schema_version": "1.0",
                "run_id":         run_id,
                "pid":            int(row["pid"]),
                "comm":           row["comm"],
                "symbol":         row["comm"],   # P1：以进程名作为符号占位，P2 替换为函数名
                "dso":            "",
                "event_type":     metric,
                "count":          int(row[metric]),
                "fraction":       float(row["fraction"]),
            }
            if "tid" in row and not pd.isna(row["tid"]):
                rec["tid"] = int(row["tid"])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[info] 热点摘要已写入 {out_f}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="访存热点分析（PID 级 + 时间窗级）")
    p.add_argument("--data",   required=True, help="采集数据目录（含 window_metrics.jsonl）")
    p.add_argument("--output", required=True, help="输出目录")
    p.add_argument("--metric", default="llc_load_misses",
                   choices=METRICS, help="排序依据指标，默认 llc_load_misses")
    p.add_argument("--top",    type=int, default=20, help="显示前 N 名，默认 20")
    p.add_argument("--pid",    type=int, default=None, help="仅分析指定 PID")
    p.add_argument("--hotspot-method", default="zscore",
                   choices=HOTSPOT_METHODS,
                   help="热点窗口检测方法：zscore / iqr / top_pct，默认 zscore")
    p.add_argument("--hotspot-threshold", type=float, default=2.0,
                   help=("热点检测阈值：\n"
                         "  zscore  → z-score 下界（默认 2.0）\n"
                         "  iqr     → IQR 倍数（默认 1.5）\n"
                         "  top_pct → 热点占所有窗口的百分比（默认 10）"))
    args = p.parse_args()

    data_dir = pathlib.Path(args.data)
    out_dir  = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_window_metrics(data_dir)

    # 从 run_metadata.jsonl 提取 run_id（取最后一条有效记录）
    meta_f = data_dir / "run_metadata.jsonl"
    run_id = "unknown"
    if meta_f.exists():
        for line in meta_f.read_text().splitlines():
            try:
                obj = json.loads(line)
                if "run_id" in obj:
                    run_id = obj["run_id"]
            except json.JSONDecodeError:
                pass

    # ── P1：PID 级全量热点 ────────────────────────────────────────────────
    agg = compute_pid_hotspot(df, args.metric, args.top, args.pid)

    print(f"\n=== 热点 PID（按 {args.metric} 降序，Top {args.top}）===")
    display_cols = [col for col in ["pid", "tid", "comm", args.metric, "fraction"] if col in agg.columns]
    print(agg[display_cols].to_string(index=False))

    write_hotspot_summary(out_dir, run_id, agg, args.metric)

    csv_path = out_dir / f"hotspot_{args.metric}.csv"
    agg.to_csv(csv_path, index=False)
    print(f"[info] CSV 已保存至 {csv_path}")

    ts = compute_time_series(df, args.metric, args.pid)
    ts_path = out_dir / f"timeseries_{args.metric}.csv"
    ts.to_csv(ts_path, index=False)
    print(f"[info] 时间序列 CSV 已保存至 {ts_path}")

    # ── 时间窗级热点识别与归因 ────────────────────────────────────────────
    ws = compute_window_hotspots(
        df, args.metric,
        method=args.hotspot_method,
        threshold=args.hotspot_threshold,
    )
    hot_ids = ws[ws["is_hot"]]["window_id"].tolist()

    wa = (
        compute_window_attribution(df, hot_ids, args.metric, top_n=args.top)
        if hot_ids else pd.DataFrame()
    )

    write_window_report(out_dir, run_id, ws, wa, args.metric)

    ws.to_csv(out_dir / f"window_hotspots_{args.metric}.csv", index=False)
    print(f"[info] 窗口热点 CSV 已保存至 {out_dir / f'window_hotspots_{args.metric}.csv'}")

    if not wa.empty:
        wa.to_csv(out_dir / f"window_attribution_{args.metric}.csv", index=False)
        print(f"[info] 窗口归因 CSV 已保存至 {out_dir / f'window_attribution_{args.metric}.csv'}")

    # ── 指标时序关系分析 ───────────────────────────────────────────────────
    mr = compute_metric_relations(
        df if args.pid is None else df[df["pid"] == args.pid],
        max_lag=5,
        spike_zscore=args.hotspot_threshold,
    )
    write_metric_relation_report(out_dir, run_id, mr)


if __name__ == "__main__":
    main()
