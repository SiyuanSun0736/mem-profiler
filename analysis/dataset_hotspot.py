"""
dataset_hotspot.py — 面向多次运行目录的数据集级热点识别

从一个数据根目录中自动发现多个 run 子目录，对每个 run 单独做时间窗热点检测，
然后汇总跨 run 的热点窗口排行和热点窗口归因结果。

用法：
    python analysis/dataset_hotspot.py \
        --data-root data/llvm_test_suite/bcc/O3-g \
        --output results/llvm_test_suite/aha_O3-g_hotspots \
        [--metric llc_load_misses] [--top 20]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import pandas as pd

try:
    from .hotspot import HOTSPOT_METHODS, METRICS, compute_window_attribution, compute_window_hotspots
except ImportError:
    from hotspot import HOTSPOT_METHODS, METRICS, compute_window_attribution, compute_window_hotspots


RUN_SUMMARY_COLUMNS = [
    "run_label",
    "run_id",
    "target_comm",
    "status",
    "skip_reason",
    "window_count",
    "hot_window_count",
    "hot_window_ratio",
    "metric",
    "metric_total",
    "max_window_value",
    "max_hot_score",
]

HOTSPOT_COLUMNS = [
    "run_label",
    "run_id",
    "target_comm",
    "metric",
    "window_id",
    "value",
    "score",
    "score_type",
    "run_metric_total",
    "window_share",
    "start_ns",
    "end_ns",
    "duration_ms",
    "top_pid",
    "top_tid",
    "top_comm",
    "top_count",
    "top_fraction",
]

ATTRIBUTION_COLUMNS = [
    "run_label",
    "run_id",
    "target_comm",
    "metric",
    "window_id",
    "pid",
    "tid",
    "comm",
    "count",
    "fraction",
]

ENTITY_COLUMNS = [
    "run_label",
    "run_id",
    "target_comm",
    "comm",
    "pid",
    "tid",
    "hot_window_hits",
    "total_count",
    "mean_fraction",
    "peak_count",
]

OVERVIEW_COLUMNS = [
    "metric",
    "analyzed_runs",
    "skipped_runs",
    "hotspot_window_count",
    "hotspot_run_count",
    "top_run_label",
    "top_window_id",
    "top_value",
    "top_score",
    "top_pid",
    "top_comm",
]


def discover_run_dirs(data_root: pathlib.Path) -> list[pathlib.Path]:
    if (data_root / "window_metrics.jsonl").exists():
        return [data_root]

    run_dirs = sorted({path.parent for path in data_root.rglob("window_metrics.jsonl")})
    if not run_dirs:
        sys.exit(f"[错误] 在 {data_root} 下没有找到任何 window_metrics.jsonl")
    return run_dirs


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def load_run_frame(run_dir: pathlib.Path) -> pd.DataFrame:
    rows = load_jsonl(run_dir / "window_metrics.jsonl")
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def load_run_metadata(run_dir: pathlib.Path) -> dict[str, Any]:
    rows = load_jsonl(run_dir / "run_metadata.jsonl")
    if not rows:
        return {}

    meta: dict[str, Any] = {}
    for row in rows:
        meta.update(row)
    return meta


def _relative_label(path: pathlib.Path, root: pathlib.Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def summarize_dataset_hotspots(
    data_root: pathlib.Path,
    metric: str,
    method: str,
    threshold: float,
    top_n: int,
    attribution_top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_dirs = discover_run_dirs(data_root)

    run_rows: list[dict[str, Any]] = []
    hotspot_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        metadata = load_run_metadata(run_dir)
        run_id = str(metadata.get("run_id", "unknown"))
        run_label = _relative_label(run_dir, data_root)
        df = load_run_frame(run_dir)
        if df.empty:
            run_rows.append({
                "run_label": run_label,
                "run_id": run_id,
                "target_comm": metadata.get("target_comm", ""),
                "status": "skipped",
                "skip_reason": "empty_window_metrics",
                "window_count": 0,
                "hot_window_count": 0,
                "hot_window_ratio": 0.0,
                "metric": metric,
                "metric_total": 0,
                "max_window_value": 0,
                "max_hot_score": 0.0,
            })
            continue
        if metric not in df.columns:
            print(f"[skip] {run_dir} 缺少指标 {metric}", flush=True)
            run_rows.append({
                "run_label": run_label,
                "run_id": run_id,
                "target_comm": metadata.get("target_comm", ""),
                "status": "skipped",
                "skip_reason": f"missing_metric:{metric}",
                "window_count": 0,
                "hot_window_count": 0,
                "hot_window_ratio": 0.0,
                "metric": metric,
                "metric_total": 0,
                "max_window_value": 0,
                "max_hot_score": 0.0,
            })
            continue

        target_comm = metadata.get("target_comm") or (
            df["comm"].dropna().iloc[0] if "comm" in df.columns and not df["comm"].dropna().empty else ""
        )

        windows = compute_window_hotspots(df, metric, method=method, threshold=threshold)
        hot_windows = windows[windows["is_hot"]].copy()
        hot_ids = hot_windows["window_id"].astype(int).tolist()
        attribution = (
            compute_window_attribution(df, hot_ids, metric, top_n=attribution_top_n)
            if hot_ids else pd.DataFrame()
        )
        if not attribution.empty:
            attribution = attribution[attribution["count"] > 0].copy()

        metric_total = int(windows["value"].sum()) if not windows.empty else 0
        max_value = int(windows["value"].max()) if not windows.empty else 0
        max_score = float(hot_windows["score"].max()) if not hot_windows.empty else 0.0

        run_rows.append({
            "run_label": run_label,
            "run_id": run_id,
            "target_comm": target_comm,
            "status": "analyzed",
            "skip_reason": "",
            "window_count": int(len(windows)),
            "hot_window_count": int(len(hot_windows)),
            "hot_window_ratio": round(float(len(hot_windows) / len(windows)), 4) if len(windows) else 0.0,
            "metric": metric,
            "metric_total": metric_total,
            "max_window_value": max_value,
            "max_hot_score": round(max_score, 4),
        })

        if hot_windows.empty:
            continue

        top_attr = pd.DataFrame()
        if not attribution.empty:
            top_attr = (
                attribution.sort_values(["window_id", "count"], ascending=[True, False])
                .groupby("window_id", group_keys=False)
                .head(1)
                .rename(columns={
                    "pid": "top_pid",
                    "tid": "top_tid",
                    "comm": "top_comm",
                    "count": "top_count",
                    "fraction": "top_fraction",
                })
            )

        merged = hot_windows.merge(top_attr, on="window_id", how="left") if not top_attr.empty else hot_windows
        for _, row in merged.sort_values(["score", "value"], ascending=[False, False]).iterrows():
            hotspot_rows.append({
                "run_label": run_label,
                "run_id": run_id,
                "target_comm": target_comm,
                "metric": metric,
                "window_id": int(row["window_id"]),
                "value": int(row["value"]),
                "score": round(float(row["score"]), 4) if pd.notna(row["score"]) else None,
                "score_type": str(row["score_type"]),
                "run_metric_total": metric_total,
                "window_share": round(float(row["value"] / metric_total), 4) if metric_total > 0 else 0.0,
                "start_ns": int(row["start_ns"]) if "start_ns" in row.index and pd.notna(row["start_ns"]) else None,
                "end_ns": int(row["end_ns"]) if "end_ns" in row.index and pd.notna(row["end_ns"]) else None,
                "duration_ms": round(float(row["duration_ms"]), 3) if "duration_ms" in row.index and pd.notna(row["duration_ms"]) else None,
                "top_pid": int(row["top_pid"]) if "top_pid" in row.index and pd.notna(row["top_pid"]) else None,
                "top_tid": int(row["top_tid"]) if "top_tid" in row.index and pd.notna(row["top_tid"]) else None,
                "top_comm": str(row["top_comm"]) if "top_comm" in row.index and pd.notna(row["top_comm"]) else None,
                "top_count": int(row["top_count"]) if "top_count" in row.index and pd.notna(row["top_count"]) else None,
                "top_fraction": round(float(row["top_fraction"]), 4) if "top_fraction" in row.index and pd.notna(row["top_fraction"]) else None,
            })

        if attribution.empty:
            continue

        for _, row in attribution.iterrows():
            attribution_rows.append({
                "run_label": run_label,
                "run_id": run_id,
                "target_comm": target_comm,
                "metric": metric,
                "window_id": int(row["window_id"]),
                "pid": int(row["pid"]),
                "tid": int(row["tid"]) if "tid" in row.index and pd.notna(row["tid"]) else None,
                "comm": str(row["comm"]),
                "count": int(row["count"]),
                "fraction": round(float(row["fraction"]), 4),
            })

    run_df = pd.DataFrame(run_rows, columns=RUN_SUMMARY_COLUMNS)
    if not run_df.empty:
        run_df = run_df.sort_values(["status", "hot_window_count", "max_hot_score"], ascending=[True, False, False])

    hotspot_df = pd.DataFrame(hotspot_rows, columns=HOTSPOT_COLUMNS)
    if not hotspot_df.empty:
        hotspot_df = hotspot_df.sort_values(["score", "value"], ascending=[False, False])

    attribution_df = pd.DataFrame(attribution_rows, columns=ATTRIBUTION_COLUMNS)
    if not attribution_df.empty:
        attribution_df = attribution_df.sort_values(["run_label", "window_id", "count"], ascending=[True, True, False])
    return run_df, hotspot_df, attribution_df


def build_entity_summary(attribution_df: pd.DataFrame) -> pd.DataFrame:
    if attribution_df.empty:
        return _empty_frame(ENTITY_COLUMNS)

    group_cols = ["run_label", "run_id", "target_comm", "comm", "pid"]
    if "tid" in attribution_df.columns and attribution_df["tid"].notna().any():
        group_cols.append("tid")

    summary = (
        attribution_df.groupby(group_cols)
        .agg(
            hot_window_hits=("window_id", "nunique"),
            total_count=("count", "sum"),
            mean_fraction=("fraction", "mean"),
            peak_count=("count", "max"),
        )
        .reset_index()
        .sort_values(["hot_window_hits", "total_count"], ascending=[False, False])
    )
    summary["mean_fraction"] = summary["mean_fraction"].round(4)
    for column in ENTITY_COLUMNS:
        if column not in summary.columns:
            summary[column] = None
    return summary[ENTITY_COLUMNS]


def build_metric_overview(
    metric: str,
    run_df: pd.DataFrame,
    hotspot_df: pd.DataFrame,
) -> dict[str, Any]:
    analyzed_runs = int((run_df["status"] == "analyzed").sum()) if not run_df.empty else 0
    skipped_runs = int((run_df["status"] == "skipped").sum()) if not run_df.empty else 0
    hotspot_run_count = int((run_df["hot_window_count"] > 0).sum()) if not run_df.empty else 0

    if hotspot_df.empty:
        return {
            "metric": metric,
            "analyzed_runs": analyzed_runs,
            "skipped_runs": skipped_runs,
            "hotspot_window_count": 0,
            "hotspot_run_count": hotspot_run_count,
            "top_run_label": None,
            "top_window_id": None,
            "top_value": None,
            "top_score": None,
            "top_pid": None,
            "top_comm": None,
        }

    top_row = hotspot_df.iloc[0]
    return {
        "metric": metric,
        "analyzed_runs": analyzed_runs,
        "skipped_runs": skipped_runs,
        "hotspot_window_count": int(len(hotspot_df)),
        "hotspot_run_count": hotspot_run_count,
        "top_run_label": top_row.get("run_label"),
        "top_window_id": int(top_row["window_id"]) if pd.notna(top_row.get("window_id")) else None,
        "top_value": int(top_row["value"]) if pd.notna(top_row.get("value")) else None,
        "top_score": round(float(top_row["score"]), 4) if pd.notna(top_row.get("score")) else None,
        "top_pid": int(top_row["top_pid"]) if pd.notna(top_row.get("top_pid")) else None,
        "top_comm": top_row.get("top_comm"),
    }


def write_jsonl(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    with out_path.open("w", encoding="utf-8") as fh:
        for record in df.where(pd.notna(df), None).to_dict(orient="records"):
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_metric_outputs(
    out_dir: pathlib.Path,
    metric: str,
    run_df: pd.DataFrame,
    hotspot_df: pd.DataFrame,
    attribution_df: pd.DataFrame,
    entity_df: pd.DataFrame,
    multi_metric: bool,
) -> None:
    suffix = f"_{metric}" if multi_metric else ""

    run_csv = out_dir / f"run_hotspot_summary{suffix}.csv"
    hotspot_csv = out_dir / f"dataset_hotspots_{metric}.csv"
    attribution_csv = out_dir / f"dataset_attribution_{metric}.csv"
    entity_csv = out_dir / f"entity_hotspots_{metric}.csv"

    run_df.to_csv(run_csv, index=False)
    hotspot_df.to_csv(hotspot_csv, index=False)
    attribution_df.to_csv(attribution_csv, index=False)
    entity_df.to_csv(entity_csv, index=False)

    write_jsonl(run_df, out_dir / f"run_hotspot_summary{suffix}.jsonl")
    write_jsonl(hotspot_df, out_dir / f"dataset_hotspots_{metric}.jsonl")
    write_jsonl(attribution_df, out_dir / f"dataset_attribution_{metric}.jsonl")
    write_jsonl(entity_df, out_dir / f"entity_hotspots_{metric}.jsonl")

    print(f"[info] {metric}: run 摘要已写入 {run_csv}")
    print(f"[info] {metric}: 热点窗口汇总已写入 {hotspot_csv}")
    print(f"[info] {metric}: 热点归因已写入 {attribution_csv}")
    print(f"[info] {metric}: 实体汇总已写入 {entity_csv}")


def print_report(run_df: pd.DataFrame, hotspot_df: pd.DataFrame, entity_df: pd.DataFrame, top_n: int) -> None:
    analyzed_runs = int((run_df["status"] == "analyzed").sum()) if "status" in run_df.columns else len(run_df)
    skipped_runs = int((run_df["status"] == "skipped").sum()) if "status" in run_df.columns else 0

    print("\n============================================================")
    print("  数据集级热点识别报告")
    print("============================================================")
    print(f"  run 数量       : {len(run_df)}")
    print(f"  已分析 run 数  : {analyzed_runs}")
    print(f"  跳过 run 数    : {skipped_runs}")
    print(f"  热点窗口总数   : {len(hotspot_df)}")

    if run_df.empty:
        print("  (没有可分析的 run)")
        print("============================================================\n")
        return

    print("\n  热点 run 摘要:")
    print(run_df.head(top_n).to_string(index=False))

    if not hotspot_df.empty:
        cols = [
            "run_label", "window_id", "value", "score", "window_share",
            "top_pid", "top_comm", "top_fraction",
        ]
        show_cols = [col for col in cols if col in hotspot_df.columns]
        print(f"\n  Top {min(top_n, len(hotspot_df))} 热点窗口:")
        print(hotspot_df[show_cols].head(top_n).to_string(index=False))

    if not entity_df.empty:
        cols = ["run_label", "pid", "tid", "comm", "hot_window_hits", "total_count", "mean_fraction"]
        show_cols = [col for col in cols if col in entity_df.columns]
        print(f"\n  Top {min(top_n, len(entity_df))} 热点归因实体:")
        print(entity_df[show_cols].head(top_n).to_string(index=False))

    print("============================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="对多次运行目录做批量热点识别")
    parser.add_argument("--data-root", required=True, help="数据根目录，目录下可包含多个 run 子目录")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--metric", default="llc_load_misses", choices=METRICS, help="热点检测指标")
    parser.add_argument("--all-metrics", action="store_true", help="一次处理所有预定义指标")
    parser.add_argument("--metrics", nargs="+", choices=METRICS, help="显式指定多个指标")
    parser.add_argument("--top", type=int, default=20, help="控制台和汇总输出展示前 N 条")
    parser.add_argument(
        "--attribution-top",
        type=int,
        default=5,
        help="每个热点窗口保留前 N 个归因实体",
    )
    parser.add_argument(
        "--hotspot-method",
        default="zscore",
        choices=HOTSPOT_METHODS,
        help="热点窗口检测方法：zscore / iqr / top_pct",
    )
    parser.add_argument(
        "--hotspot-threshold",
        type=float,
        default=2.0,
        help="热点检测阈值，与 analysis/hotspot.py 保持一致",
    )
    args = parser.parse_args()

    data_root = pathlib.Path(args.data_root)
    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.metrics:
        metrics_to_run = list(dict.fromkeys(args.metrics))
    elif args.all_metrics:
        metrics_to_run = list(METRICS)
    else:
        metrics_to_run = [args.metric]

    multi_metric = len(metrics_to_run) > 1
    overview_rows: list[dict[str, Any]] = []

    for metric in metrics_to_run:
        run_df, hotspot_df, attribution_df = summarize_dataset_hotspots(
            data_root=data_root,
            metric=metric,
            method=args.hotspot_method,
            threshold=args.hotspot_threshold,
            top_n=args.top,
            attribution_top_n=args.attribution_top,
        )
        entity_df = build_entity_summary(attribution_df)
        write_metric_outputs(
            out_dir=out_dir,
            metric=metric,
            run_df=run_df,
            hotspot_df=hotspot_df,
            attribution_df=attribution_df,
            entity_df=entity_df,
            multi_metric=multi_metric,
        )
        overview_rows.append(build_metric_overview(metric, run_df, hotspot_df))

        if not multi_metric:
            print_report(run_df, hotspot_df, entity_df, top_n=args.top)

    if multi_metric:
        overview_df = pd.DataFrame(overview_rows, columns=OVERVIEW_COLUMNS)
        overview_df = overview_df.sort_values(["hotspot_window_count", "top_score"], ascending=[False, False])
        overview_csv = out_dir / "metrics_overview.csv"
        overview_df.to_csv(overview_csv, index=False)
        write_jsonl(overview_df, out_dir / "metrics_overview.jsonl")
        print(f"[info] 指标总览已写入 {overview_csv}")
        print("\n============================================================")
        print("  多指标热点总览")
        print("============================================================")
        print(overview_df.to_string(index=False))
        print("============================================================\n")


if __name__ == "__main__":
    main()