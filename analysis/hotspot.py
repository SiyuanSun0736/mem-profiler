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

import pandas as pd


METRICS = [
    "llc_load_misses",
    "llc_store_misses",
    "dtlb_misses",
    "minor_faults",
    "major_faults",
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
    """按 PID 和 comm 聚合，返回按 metric 降序的 top_n 条记录。"""
    if target_pid:
        df = df[df["pid"] == target_pid]
    if metric not in df.columns:
        sys.exit(f"[错误] 指标 '{metric}' 不在数据中，可选: {METRICS}")

    agg = (
        df.groupby(["pid", "comm"])[METRICS]
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
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[info] 热点摘要已写入 {out_f}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="访存热点分析（PID 级，P1 阶段）")
    p.add_argument("--data",   required=True, help="采集数据目录（含 window_metrics.jsonl）")
    p.add_argument("--output", required=True, help="输出目录")
    p.add_argument("--metric", default="llc_load_misses",
                   choices=METRICS, help="排序依据指标，默认 llc_load_misses")
    p.add_argument("--top",    type=int, default=20, help="显示前 N 名，默认 20")
    p.add_argument("--pid",    type=int, default=None, help="仅分析指定 PID")
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

    agg = compute_pid_hotspot(df, args.metric, args.top, args.pid)

    # 控制台输出
    print(f"\n=== 热点 PID（按 {args.metric} 降序，Top {args.top}）===")
    print(agg[["pid", "comm", args.metric, "fraction"]].to_string(index=False))

    # 写入 hotspot_summary.jsonl
    write_hotspot_summary(out_dir, run_id, agg, args.metric)

    # 导出热点 CSV
    csv_path = out_dir / f"hotspot_{args.metric}.csv"
    agg.to_csv(csv_path, index=False)
    print(f"[info] CSV 已保存至 {csv_path}")

    # 时间序列 CSV
    ts = compute_time_series(df, args.metric, args.pid)
    ts_path = out_dir / f"timeseries_{args.metric}.csv"
    ts.to_csv(ts_path, index=False)
    print(f"[info] 时间序列 CSV 已保存至 {ts_path}")


if __name__ == "__main__":
    main()
