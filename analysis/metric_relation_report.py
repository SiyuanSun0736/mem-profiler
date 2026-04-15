"""
metric_relation_report.py — 生成数据集级指标时序关系分析报告

默认读取 llvm-test-suite BCC 数据集目录，对每个 run 复用 hotspot.py
中的多指标时序关系分析逻辑，并输出跨 run 汇总、Markdown 摘要和可选图表。

用法：
    python analysis/metric_relation_report.py

    python analysis/metric_relation_report.py \
        --data-root data/llvm_test_suite/bcc/O3-g \
        --output results/llvm_test_suite/custom_metric_relations \
        --max-lag 8
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    from . import dataset_hotspot as dataset_utils
    from . import hotspot as hotspot_analysis
    from .report import METRIC_LABELS
except ImportError:
    import dataset_hotspot as dataset_utils
    import hotspot as hotspot_analysis
    from report import METRIC_LABELS


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "data/llvm_test_suite/bcc/O3-g"
DEFAULT_OUTPUT = REPO_ROOT / "results/llvm_test_suite/aha_O3-g_metric_relations"

RUN_COLUMNS = [
    "run_label",
    "run_id",
    "target_comm",
    "status",
    "skip_reason",
    "window_count",
    "available_metric_count",
    "pair_count",
    "nonzero_lag_pair_count",
    "multi_spike_window_count",
    "strongest_pair",
    "strongest_abs_pearson",
    "strongest_peak_lag",
]

PAIR_COLUMNS = [
    "run_label",
    "run_id",
    "target_comm",
    "metric_a",
    "metric_b",
    "pearson_r",
    "abs_pearson_r",
    "peak_lag",
    "peak_lag_corr",
    "abs_peak_lag_corr",
    "co_spike_count",
]

OVERVIEW_COLUMNS = [
    "metric_a",
    "metric_b",
    "run_count",
    "nonzero_lag_runs",
    "mean_abs_pearson",
    "max_abs_pearson",
    "mean_peak_lag_corr",
    "dominant_peak_lag",
    "mean_co_spike_count",
    "max_co_spike_count",
    "top_run_label",
    "top_abs_pearson",
    "top_peak_lag",
]

PALETTE = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#577590"]


def _setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.dpi": 150,
    })


def _relative_label(path: pathlib.Path, root: pathlib.Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


def _pair_label(metric_a: str, metric_b: str) -> str:
    return f"{_metric_label(metric_a)} vs {_metric_label(metric_b)}"


def _stringify(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    if df.empty:
        return ["(无数据)"]

    existing = [column for column in columns if column in df.columns]
    if not existing:
        return ["(无可展示列)"]

    header = "| " + " | ".join(existing) + " |"
    sep = "| " + " | ".join("---" for _ in existing) + " |"
    rows = []
    for _, row in df[existing].iterrows():
        rows.append("| " + " | ".join(_stringify(row[column]) for column in existing) + " |")
    return [header, sep, *rows]


def write_jsonl(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    with out_path.open("w", encoding="utf-8") as fh:
        for record in df.where(pd.notna(df), None).to_dict(orient="records"):
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize_dataset_metric_relations(
    data_root: pathlib.Path,
    max_lag: int,
    spike_zscore: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_dirs = dataset_utils.discover_run_dirs(data_root)

    run_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        metadata = dataset_utils.load_run_metadata(run_dir)
        run_id = str(metadata.get("run_id", "unknown"))
        run_label = _relative_label(run_dir, data_root)
        df = dataset_utils.load_run_frame(run_dir)

        if df.empty:
            run_rows.append({
                "run_label": run_label,
                "run_id": run_id,
                "target_comm": metadata.get("target_comm", ""),
                "status": "skipped",
                "skip_reason": "empty_window_metrics",
                "window_count": 0,
                "available_metric_count": 0,
                "pair_count": 0,
                "nonzero_lag_pair_count": 0,
                "multi_spike_window_count": 0,
                "strongest_pair": None,
                "strongest_abs_pearson": None,
                "strongest_peak_lag": None,
            })
            continue

        available_metrics = [
            metric
            for metric in hotspot_analysis.METRICS
            if metric in df.columns and float(df[metric].sum()) > 0
        ]

        if len(available_metrics) < 2:
            run_rows.append({
                "run_label": run_label,
                "run_id": run_id,
                "target_comm": metadata.get("target_comm", ""),
                "status": "skipped",
                "skip_reason": "insufficient_metrics",
                "window_count": int(df["window_id"].nunique()) if "window_id" in df.columns else 0,
                "available_metric_count": len(available_metrics),
                "pair_count": 0,
                "nonzero_lag_pair_count": 0,
                "multi_spike_window_count": 0,
                "strongest_pair": None,
                "strongest_abs_pearson": None,
                "strongest_peak_lag": None,
            })
            continue

        relations = hotspot_analysis.compute_metric_relations(
            df,
            max_lag=max_lag,
            spike_zscore=spike_zscore,
        )
        if not relations or relations["pair_summary"].empty:
            run_rows.append({
                "run_label": run_label,
                "run_id": run_id,
                "target_comm": metadata.get("target_comm", ""),
                "status": "skipped",
                "skip_reason": "empty_pair_summary",
                "window_count": int(df["window_id"].nunique()) if "window_id" in df.columns else 0,
                "available_metric_count": len(available_metrics),
                "pair_count": 0,
                "nonzero_lag_pair_count": 0,
                "multi_spike_window_count": 0,
                "strongest_pair": None,
                "strongest_abs_pearson": None,
                "strongest_peak_lag": None,
            })
            continue

        pair_df = relations["pair_summary"].copy()
        pair_df["abs_pearson_r"] = pair_df["pearson_r"].abs().round(4)
        pair_df["abs_peak_lag_corr"] = pair_df["peak_lag_corr"].abs().round(4)
        pair_df = pair_df.sort_values(["abs_pearson_r", "abs_peak_lag_corr"], ascending=[False, False])

        top_pair = pair_df.iloc[0]
        run_rows.append({
            "run_label": run_label,
            "run_id": run_id,
            "target_comm": metadata.get("target_comm", ""),
            "status": "analyzed",
            "skip_reason": "",
            "window_count": int(df["window_id"].nunique()) if "window_id" in df.columns else 0,
            "available_metric_count": len(available_metrics),
            "pair_count": int(len(pair_df)),
            "nonzero_lag_pair_count": int((pair_df["peak_lag"] != 0).sum()),
            "multi_spike_window_count": int((relations["co_spike"]["co_spike_count"] >= 2).sum()),
            "strongest_pair": f"{top_pair['metric_a']} vs {top_pair['metric_b']}",
            "strongest_abs_pearson": round(float(top_pair["abs_pearson_r"]), 4),
            "strongest_peak_lag": int(top_pair["peak_lag"]),
        })

        for _, row in pair_df.iterrows():
            pair_rows.append({
                "run_label": run_label,
                "run_id": run_id,
                "target_comm": metadata.get("target_comm", ""),
                "metric_a": row["metric_a"],
                "metric_b": row["metric_b"],
                "pearson_r": round(float(row["pearson_r"]), 4),
                "abs_pearson_r": round(float(row["abs_pearson_r"]), 4),
                "peak_lag": int(row["peak_lag"]),
                "peak_lag_corr": round(float(row["peak_lag_corr"]), 4),
                "abs_peak_lag_corr": round(float(row["abs_peak_lag_corr"]), 4),
                "co_spike_count": int(row["co_spike_count"]),
            })

    run_df = pd.DataFrame(run_rows, columns=RUN_COLUMNS)
    if not run_df.empty:
        run_df = run_df.sort_values(["status", "strongest_abs_pearson", "pair_count"], ascending=[True, False, False])

    pair_df = pd.DataFrame(pair_rows, columns=PAIR_COLUMNS)
    if not pair_df.empty:
        pair_df = pair_df.sort_values(["abs_pearson_r", "abs_peak_lag_corr"], ascending=[False, False])

    return run_df, pair_df


def build_pair_overview(pair_df: pd.DataFrame) -> pd.DataFrame:
    if pair_df.empty:
        return pd.DataFrame(columns=OVERVIEW_COLUMNS)

    rows: list[dict[str, Any]] = []
    for (metric_a, metric_b), sub in pair_df.groupby(["metric_a", "metric_b"], sort=False):
        strongest = sub.loc[sub["abs_pearson_r"].idxmax()]
        lag_counts = sub["peak_lag"].value_counts()
        dominant_peak_lag = int(lag_counts.index[0]) if not lag_counts.empty else 0
        rows.append({
            "metric_a": metric_a,
            "metric_b": metric_b,
            "run_count": int(sub["run_label"].nunique()),
            "nonzero_lag_runs": int((sub["peak_lag"] != 0).sum()),
            "mean_abs_pearson": round(float(sub["abs_pearson_r"].mean()), 4),
            "max_abs_pearson": round(float(sub["abs_pearson_r"].max()), 4),
            "mean_peak_lag_corr": round(float(sub["abs_peak_lag_corr"].mean()), 4),
            "dominant_peak_lag": dominant_peak_lag,
            "mean_co_spike_count": round(float(sub["co_spike_count"].mean()), 4),
            "max_co_spike_count": int(sub["co_spike_count"].max()),
            "top_run_label": strongest["run_label"],
            "top_abs_pearson": round(float(strongest["abs_pearson_r"]), 4),
            "top_peak_lag": int(strongest["peak_lag"]),
        })

    overview_df = pd.DataFrame(rows, columns=OVERVIEW_COLUMNS)
    if not overview_df.empty:
        overview_df = overview_df.sort_values(["mean_abs_pearson", "max_abs_pearson"], ascending=[False, False])
    return overview_df


def plot_pair_overview(overview_df: pd.DataFrame, out_dir: pathlib.Path, top_n: int) -> None:
    if overview_df.empty:
        return

    df = overview_df.head(top_n).copy().iloc[::-1]
    df["label"] = df.apply(lambda row: _pair_label(row["metric_a"], row["metric_b"]), axis=1)

    fig, ax = plt.subplots(figsize=(12, max(4, 0.55 * len(df))))
    bars = ax.barh(df["label"], df["mean_abs_pearson"], color=PALETTE[1], edgecolor="none")
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"  runs={int(row['run_count'])}, lag={int(row['dominant_peak_lag'])}",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Mean |Pearson r|")
    ax.set_title("Dataset-Level Metric Relation Strength", fontweight="bold")
    fig.tight_layout()
    out_path = out_dir / "metric_pair_overview.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[图表] {out_path}", flush=True)


def plot_co_spike_overview(overview_df: pd.DataFrame, out_dir: pathlib.Path, top_n: int) -> None:
    if overview_df.empty:
        return

    df = overview_df.sort_values(["mean_co_spike_count", "max_co_spike_count"], ascending=[False, False]).head(top_n).copy().iloc[::-1]
    df["label"] = df.apply(lambda row: _pair_label(row["metric_a"], row["metric_b"]), axis=1)

    fig, ax = plt.subplots(figsize=(12, max(4, 0.55 * len(df))))
    bars = ax.barh(df["label"], df["mean_co_spike_count"], color=PALETTE[4], edgecolor="none")
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"  max={int(row['max_co_spike_count'])}, top_run={row['top_run_label']}",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Mean Co-Spike Count")
    ax.set_title("Dataset-Level Co-Spike Overview", fontweight="bold")
    fig.tight_layout()
    out_path = out_dir / "metric_pair_co_spike_overview.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[图表] {out_path}", flush=True)


def write_markdown_report(
    out_dir: pathlib.Path,
    data_root: pathlib.Path,
    run_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    overview_df: pd.DataFrame,
    top_n: int,
    max_lag: int,
    spike_zscore: float,
) -> pathlib.Path:
    report_path = out_dir / "metric_relation_report.md"
    analyzed_runs = int((run_df["status"] == "analyzed").sum()) if not run_df.empty else 0
    skipped_runs = int((run_df["status"] == "skipped").sum()) if not run_df.empty else 0
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# 数据集级指标时序关系分析报告",
        "",
        f"- 数据目录: {data_root}",
        f"- 输出目录: {out_dir}",
        f"- 生成时间: {timestamp}",
        f"- 最大滞后窗口: {max_lag}",
        f"- 联合热点阈值: z >= {spike_zscore}",
        f"- run 总数: {len(run_df)}",
        f"- 已分析 run: {analyzed_runs}",
        f"- 跳过 run: {skipped_runs}",
        f"- 指标对记录数: {len(pair_df)}",
        "",
        f"## Top {min(top_n, len(overview_df))} 指标对总览",
        "",
    ]
    lines.extend(
        _markdown_table(
            overview_df.head(top_n),
            [
                "metric_a",
                "metric_b",
                "run_count",
                "mean_abs_pearson",
                "dominant_peak_lag",
                "mean_co_spike_count",
                "top_run_label",
            ],
        )
    )
    lines.extend([
        "",
        f"## Top {min(top_n, len(pair_df))} 单次运行指标对",
        "",
    ])
    lines.extend(
        _markdown_table(
            pair_df.head(top_n),
            [
                "run_label",
                "metric_a",
                "metric_b",
                "pearson_r",
                "peak_lag",
                "peak_lag_corr",
                "co_spike_count",
            ],
        )
    )
    lines.extend([
        "",
        f"## Top {min(top_n, len(run_df))} run 摘要",
        "",
    ])
    lines.extend(
        _markdown_table(
            run_df.head(top_n),
            [
                "run_label",
                "status",
                "available_metric_count",
                "pair_count",
                "strongest_pair",
                "strongest_abs_pearson",
                "strongest_peak_lag",
            ],
        )
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成默认指向 llvm-test-suite 数据集的指标时序关系报告")
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="数据根目录，默认使用 data/llvm_test_suite/bcc/O3-g",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="输出目录，默认写入 results/llvm_test_suite/aha_O3-g_metric_relations",
    )
    parser.add_argument("--max-lag", type=int, default=5, help="滞后互相关搜索的最大窗口偏移")
    parser.add_argument("--spike-zscore", type=float, default=2.0, help="联合热点阈值")
    parser.add_argument("--top", type=int, default=10, help="Markdown 和图表展示前 N 条")
    parser.add_argument("--skip-figures", action="store_true", help="只生成表格和 Markdown，不生成 PDF 图表")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root = pathlib.Path(args.data_root).expanduser().resolve()
    out_dir = pathlib.Path(args.output).expanduser().resolve()
    figures_dir = out_dir / "figures"

    if not data_root.exists():
        sys.exit(f"[错误] 数据目录不存在: {data_root}")

    out_dir.mkdir(parents=True, exist_ok=True)

    run_df, pair_df = summarize_dataset_metric_relations(
        data_root=data_root,
        max_lag=args.max_lag,
        spike_zscore=args.spike_zscore,
    )
    overview_df = build_pair_overview(pair_df)

    run_csv = out_dir / "run_metric_relation_summary.csv"
    pair_csv = out_dir / "dataset_metric_pairs.csv"
    overview_csv = out_dir / "metric_pair_overview.csv"

    run_df.to_csv(run_csv, index=False)
    pair_df.to_csv(pair_csv, index=False)
    overview_df.to_csv(overview_csv, index=False)

    write_jsonl(run_df, out_dir / "run_metric_relation_summary.jsonl")
    write_jsonl(pair_df, out_dir / "dataset_metric_pairs.jsonl")
    write_jsonl(overview_df, out_dir / "metric_pair_overview.jsonl")

    analyzed_runs = int((run_df["status"] == "analyzed").sum()) if not run_df.empty else 0
    skipped_runs = int((run_df["status"] == "skipped").sum()) if not run_df.empty else 0

    print(f"[info] run 级时序关系摘要已写入 {run_csv}")
    print(f"[info] 数据集级指标对明细已写入 {pair_csv}")
    print(f"[info] 指标对总览已写入 {overview_csv}")
    print("\n============================================================")
    print("  数据集级指标时序关系分析")
    print("============================================================")
    print(f"  run 总数       : {len(run_df)}")
    print(f"  已分析 run 数  : {analyzed_runs}")
    print(f"  跳过 run 数    : {skipped_runs}")
    print(f"  指标对记录数   : {len(pair_df)}")
    if not overview_df.empty:
        print(f"\n  Top {min(args.top, len(overview_df))} 指标对总览:")
        print(overview_df.head(args.top).to_string(index=False))
    print("============================================================\n")

    report_path = write_markdown_report(
        out_dir=out_dir,
        data_root=data_root,
        run_df=run_df,
        pair_df=pair_df,
        overview_df=overview_df,
        top_n=args.top,
        max_lag=args.max_lag,
        spike_zscore=args.spike_zscore,
    )
    print(f"[info] Markdown 报告已写入 {report_path}")

    if not args.skip_figures:
        figures_dir.mkdir(parents=True, exist_ok=True)
        _setup_style()
        plot_pair_overview(overview_df, figures_dir, top_n=args.top)
        plot_co_spike_overview(overview_df, figures_dir, top_n=args.top)
        print(f"[info] 图表输出目录: {figures_dir}")


if __name__ == "__main__":
    main()