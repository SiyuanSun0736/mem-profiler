"""
report.py — 从分析结果生成论文级图表

生成图表列表：
  1. 时间序列折线图（多指标对比，window_id × metric_value）
  2. 热点 PID 条形图（按 llc_load_misses 等指标排序）
  3. 函数级热点水平条形图（attribution.py 输出）
  4. 指标相关性热力图（各指标之间的 Pearson 相关系数）

用法：
    python analysis/report.py \\
        --results results/run_001/ \\
        --output  results/run_001/figures/
"""

import argparse
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")   # 无 GUI 环境下使用 Agg 后端
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


PALETTE = ["#2d7dd2", "#e63946", "#06d6a0", "#ffd166", "#9b5de5", "#f77f00"]
METRICS  = ["llc_load_misses", "llc_store_misses", "dtlb_misses",
            "minor_faults", "major_faults"]
METRIC_LABELS = {
    "llc_load_misses":  "LLC Load Misses",
    "llc_store_misses": "LLC Store Misses",
    "dtlb_misses":      "dTLB Misses",
    "minor_faults":     "Minor Faults",
    "major_faults":     "Major Faults",
}


def _setup_style() -> None:
    plt.rcParams.update({
        "font.family":     "DejaVu Sans",
        "font.size":       11,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "figure.dpi":         150,
    })


def plot_time_series(
    results_dir: pathlib.Path,
    out_dir: pathlib.Path,
) -> None:
    """绘制各指标随时间窗变化的折线图。"""
    pattern = "timeseries_*.csv"
    files = list(results_dir.glob(pattern))
    if not files:
        print("[skip] 未找到 timeseries_*.csv，跳过时间序列图", flush=True)
        return

    fig, axes = plt.subplots(len(files), 1, figsize=(10, 3 * len(files)), sharex=True)
    if len(files) == 1:
        axes = [axes]

    for ax, csv_f in zip(axes, sorted(files)):
        metric = csv_f.stem.replace("timeseries_", "")
        df = pd.read_csv(csv_f)
        ax.plot(df["window_id"], df["value"], color=PALETTE[0], linewidth=1.5)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{x/1e3:.0f}K" if x >= 1000 else str(int(x))
        ))

    axes[-1].set_xlabel("Time Window ID")
    fig.suptitle("Memory Access Metrics Over Time", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = out_dir / "timeseries.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[图表] {out_path}", flush=True)


def plot_hotspot_bar(
    results_dir: pathlib.Path,
    out_dir: pathlib.Path,
    top_n: int = 10,
) -> None:
    """绘制热点 PID / 函数水平条形图。"""
    for fname, title_prefix in [
        ("hotspot_llc_load_misses.csv", "LLC Load Miss Hot PIDs"),
        ("function_hotspot_llc_load_misses.csv", "LLC Load Miss Hot Functions"),
    ]:
        csv_f = results_dir / fname
        if not csv_f.exists():
            continue
        df = pd.read_csv(csv_f).head(top_n)
        if df.empty:
            continue

        label_col = "func" if "func" in df.columns else "comm"
        count_col = "count" if "count" in df.columns else "llc_load_misses"

        fig, ax = plt.subplots(figsize=(9, max(3, 0.45 * len(df))))
        bars = ax.barh(
            df[label_col][::-1],
            df[count_col][::-1],
            color=PALETTE[0],
            edgecolor="none",
        )
        ax.set_xlabel("Event Count")
        ax.set_title(f"{title_prefix} (Top {top_n})", fontweight="bold")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{x/1e3:.0f}K" if x >= 1000 else str(int(x))
        ))
        fig.tight_layout()
        out_path = out_dir / f"{csv_f.stem}_bar.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[图表] {out_path}", flush=True)


def plot_correlation_heatmap(
    results_dir: pathlib.Path,
    out_dir: pathlib.Path,
) -> None:
    """绘制指标相关性热力图。"""
    # 尝试从上层数据目录找 window_metrics.jsonl
    candidates = list(results_dir.parent.rglob("window_metrics.jsonl"))
    if not candidates:
        print("[skip] 未找到 window_metrics.jsonl，跳过相关性热力图", flush=True)
        return

    df = pd.concat(
        [pd.DataFrame([json.loads(l) for l in f.read_text().splitlines() if l.strip()])
         for f in candidates],
        ignore_index=True,
    )
    cols = [c for c in METRICS if c in df.columns]
    if len(cols) < 2:
        return

    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    labels = [METRIC_LABELS.get(c, c) for c in cols]
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(corr.values[i, j]) > 0.5 else "black")
    ax.set_title("Metric Correlation (Pearson)", fontweight="bold")
    fig.tight_layout()
    out_path = out_dir / "metric_correlation.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[图表] {out_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="生成论文级访存分析图表")
    p.add_argument("--results", required=True, help="分析结果目录（含 CSV 文件）")
    p.add_argument("--output",  required=True, help="图表输出目录")
    p.add_argument("--top",     type=int, default=10, help="条形图显示前 N 名，默认 10")
    args = p.parse_args()

    results_dir = pathlib.Path(args.results)
    out_dir     = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    _setup_style()
    plot_time_series(results_dir, out_dir)
    plot_hotspot_bar(results_dir, out_dir, top_n=args.top)
    plot_correlation_heatmap(results_dir, out_dir)
    print(f"\n[info] 所有图表已保存至 {out_dir}", flush=True)


if __name__ == "__main__":
    main()
