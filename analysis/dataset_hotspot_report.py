"""
dataset_hotspot_report.py — 数据集级热点分析图表生成

从 analysis/dataset_hotspot.py 的输出目录读取 CSV，生成跨 run 热点窗口图、
归因实体图和多指标总览图。

用法：
    python analysis/dataset_hotspot_report.py \
        --results results/llvm_test_suite/aha_O3-g_hotspots \
        --output  results/llvm_test_suite/aha_O3-g_hotspots/figures \
        [--top 10]
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PALETTE = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#577590"]

try:
    from .report import METRIC_LABELS
except ImportError:
    try:
        from report import METRIC_LABELS
    except ImportError:
        METRIC_LABELS = {}


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


def _metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


def plot_metrics_overview(results_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    overview_f = results_dir / "metrics_overview.csv"
    if not overview_f.exists():
        print("[skip] 未找到 metrics_overview.csv，跳过指标总览图", flush=True)
        return

    df = pd.read_csv(overview_f)
    if df.empty:
        return

    df = df.sort_values(["hotspot_window_count", "top_score"], ascending=[False, False])
    labels = [_metric_label(metric) for metric in df["metric"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, 0.45 * len(df))))

    axes[0].barh(labels[::-1], df["hotspot_window_count"][::-1], color=PALETTE[1])
    axes[0].set_xlabel("Hot Window Count")
    axes[0].set_title("Hot Windows by Metric", fontweight="bold")

    score_series = df["top_score"].fillna(0)
    axes[1].barh(labels[::-1], score_series[::-1], color=PALETTE[4])
    axes[1].set_xlabel("Peak Hot Score")
    axes[1].set_title("Peak Hotspot Score by Metric", fontweight="bold")

    fig.tight_layout()
    out_path = out_dir / "metrics_overview.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[图表] {out_path}", flush=True)


def plot_dataset_hotspots(results_dir: pathlib.Path, out_dir: pathlib.Path, top_n: int) -> None:
    for csv_f in sorted(results_dir.glob("dataset_hotspots_*.csv")):
        metric = csv_f.stem.replace("dataset_hotspots_", "")
        df = pd.read_csv(csv_f)
        if df.empty:
            print(f"[skip] {metric} 无热点窗口，跳过热点图", flush=True)
            continue

        df = df.head(top_n).copy()
        df["label"] = df.apply(lambda row: f"{row['run_label']} / w{int(row['window_id'])}", axis=1)

        fig, ax = plt.subplots(figsize=(12, max(3.5, 0.5 * len(df))))
        bars = ax.barh(df["label"][::-1], df["value"][::-1], color=PALETTE[0])

        for bar, (_, row) in zip(bars, df.iloc[::-1].iterrows()):
            annotation = f"score={row['score']:.2f}"
            if pd.notna(row.get("top_pid")):
                annotation += f" | pid={int(row['top_pid'])}"
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"  {annotation}", va="center", fontsize=9)

        ax.set_xlabel(_metric_label(metric))
        ax.set_title(f"Cross-run Hot Windows — {_metric_label(metric)}", fontweight="bold")
        fig.tight_layout()
        out_path = out_dir / f"dataset_hotspots_{metric}.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[图表] {out_path}", flush=True)


def plot_entity_hotspots(results_dir: pathlib.Path, out_dir: pathlib.Path, top_n: int) -> None:
    for csv_f in sorted(results_dir.glob("entity_hotspots_*.csv")):
        metric = csv_f.stem.replace("entity_hotspots_", "")
        df = pd.read_csv(csv_f)
        if df.empty:
            print(f"[skip] {metric} 无热点归因实体，跳过归因图", flush=True)
            continue

        df = df.sort_values(["hot_window_hits", "total_count"], ascending=[False, False]).head(top_n).copy()

        def _label(row: pd.Series) -> str:
            pid_part = f"pid={int(row['pid'])}"
            tid_part = f", tid={int(row['tid'])}" if "tid" in row.index and pd.notna(row["tid"]) else ""
            return f"{row['run_label']} / {row['comm']} / {pid_part}{tid_part}"

        df["label"] = df.apply(_label, axis=1)

        fig, ax = plt.subplots(figsize=(13, max(3.5, 0.55 * len(df))))
        bars = ax.barh(df["label"][::-1], df["total_count"][::-1], color=PALETTE[3])

        for bar, (_, row) in zip(bars, df.iloc[::-1].iterrows()):
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f"  hits={int(row['hot_window_hits'])}, mean_fraction={row['mean_fraction']:.2f}",
                va="center",
                fontsize=9,
            )

        ax.set_xlabel(f"Total {_metric_label(metric)} in Hot Windows")
        ax.set_title(f"Hot Attribution Entities — {_metric_label(metric)}", fontweight="bold")
        fig.tight_layout()
        out_path = out_dir / f"entity_hotspots_{metric}.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[图表] {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="从批量热点分析结果生成图表")
    parser.add_argument("--results", required=True, help="analysis/dataset_hotspot.py 的输出目录")
    parser.add_argument("--output", required=True, help="图表输出目录")
    parser.add_argument("--top", type=int, default=10, help="每张图显示前 N 条")
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results)
    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    _setup_style()
    plot_metrics_overview(results_dir, out_dir)
    plot_dataset_hotspots(results_dir, out_dir, top_n=args.top)
    plot_entity_hotspots(results_dir, out_dir, top_n=args.top)


if __name__ == "__main__":
    main()