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
METRICS  = [
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
    "anon_faults",
    "file_faults",
    "shared_faults",
    "private_faults",
    "write_faults",
    "instruction_faults",
    "mmap_calls",
    "munmap_calls",
    "mprotect_calls",
    "brk_calls",
    "mmap_bytes",
    "munmap_bytes",
    "mprotect_bytes",
    "brk_growth_bytes",
    "brk_shrink_bytes",
    "lbr_samples",
    "lbr_entries",
]
METRIC_LABELS = {
    "llc_loads":        "LLC Loads",
    "llc_load_misses":  "LLC Load Misses",
    "llc_stores":       "LLC Stores",
    "llc_store_misses": "LLC Store Misses",
    "dtlb_loads":       "dTLB Loads",
    "dtlb_load_misses": "dTLB Load Misses",
    "dtlb_stores":      "dTLB Stores",
    "dtlb_store_misses": "dTLB Store Misses",
    "dtlb_misses":      "dTLB Misses",
    "itlb_loads":       "iTLB Loads",
    "itlb_load_misses": "iTLB Load Misses",
    "minor_faults":     "Minor Faults",
    "major_faults":     "Major Faults",
    "anon_faults":      "Anonymous Faults",
    "file_faults":      "File-backed Faults",
    "shared_faults":    "Shared Faults",
    "private_faults":   "Private Faults",
    "write_faults":     "Write Faults",
    "instruction_faults": "Instruction Faults",
    "mmap_calls":       "mmap Calls",
    "munmap_calls":     "munmap Calls",
    "mprotect_calls":   "mprotect Calls",
    "brk_calls":        "brk Calls",
    "mmap_bytes":       "mmap Bytes",
    "munmap_bytes":     "munmap Bytes",
    "mprotect_bytes":   "mprotect Bytes",
    "brk_growth_bytes": "brk Growth Bytes",
    "brk_shrink_bytes": "brk Shrink Bytes",
    "lbr_samples":      "LBR Samples",
    "lbr_entries":      "LBR Entries",
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


def plot_window_hotspots(
    results_dir: pathlib.Path,
    out_dir: pathlib.Path,
) -> None:
    """
    绘制带热点窗口标注的时间序列图，并为每个热点指标绘制 PID 归因图。
    依赖 hotspot.py 输出的 window_hotspots_<metric>.csv
    以及（可选的）window_attribution_<metric>.csv。
    """
    hot_files = sorted(results_dir.glob("window_hotspots_*.csv"))
    if not hot_files:
        print("[skip] 未找到 window_hotspots_*.csv，跳过热点窗口图", flush=True)
        return

    for hot_f in hot_files:
        metric = hot_f.stem.replace("window_hotspots_", "")
        wh = pd.read_csv(hot_f)
        if wh.empty:
            continue

        # ── 时间序列 + 热点色带 ───────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(
            wh["window_id"], wh["value"],
            color=PALETTE[0], linewidth=1.5,
            label=METRIC_LABELS.get(metric, metric),
        )
        hot = wh[wh["is_hot"]]
        if not hot.empty:
            for wid in hot["window_id"]:
                ax.axvspan(wid - 0.5, wid + 0.5,
                           color=PALETTE[1], alpha=0.18, linewidth=0)
            ax.scatter(
                hot["window_id"], hot["value"],
                color=PALETTE[1], s=55, zorder=5,
                label=f"Hot Window (n={len(hot)})",
            )
        ax.set_xlabel("Time Window ID")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(
            f"Window Hotspot Detection — {METRIC_LABELS.get(metric, metric)}",
            fontweight="bold",
        )
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{x/1e3:.0f}K" if x >= 1000 else str(int(x))
        ))
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        out_path = out_dir / f"window_hotspots_{metric}.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[图表] {out_path}", flush=True)

        # ── PID 归因图 ────────────────────────────────────────────────────
        attr_f = results_dir / f"window_attribution_{metric}.csv"
        if attr_f.exists():
            attr = pd.read_csv(attr_f)
            if not attr.empty:
                _plot_window_attribution(attr, metric, out_dir)


def _plot_window_attribution(
    attr: pd.DataFrame,
    metric: str,
    out_dir: pathlib.Path,
) -> None:
    """
    为热点窗口绘制 PID 归因水平条形图。
    热点窗口数 <= 4 时每个窗口独立子图；否则汇总所有热点窗口。
    """
    attr = attr.copy()
    attr["label"] = attr.apply(
        lambda r: f"{r['comm']}({int(r['pid'])})", axis=1
    )
    hot_windows = sorted(attr["window_id"].unique())
    n = len(hot_windows)
    if n == 0:
        return

    if n <= 4:
        fig, axes = plt.subplots(
            1, n,
            figsize=(5 * n, max(3, 0.5 * attr["label"].nunique())),
            sharey=False,
        )
        if n == 1:
            axes = [axes]
        for ax, wid in zip(axes, hot_windows):
            wd = attr[attr["window_id"] == wid].sort_values("count", ascending=True)
            ax.barh(wd["label"], wd["count"], color=PALETTE[0], edgecolor="none")
            ax.set_xlabel("Count")
            ax.set_title(f"Window {wid}", fontsize=10, fontweight="bold")
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                lambda x, _: f"{x/1e3:.0f}K" if x >= 1000 else str(int(x))
            ))
        fig.suptitle(
            f"Hot Window PID Attribution — {METRIC_LABELS.get(metric, metric)}",
            fontweight="bold", fontsize=12,
        )
    else:
        agg = (
            attr.groupby("label")["count"]
            .sum()
            .reset_index()
            .sort_values("count", ascending=True)
            .tail(15)
        )
        fig, ax = plt.subplots(figsize=(9, max(3, 0.45 * len(agg))))
        ax.barh(agg["label"], agg["count"], color=PALETTE[0], edgecolor="none")
        ax.set_xlabel("Total Count Across Hot Windows")
        ax.set_title(
            f"Top PIDs in Hot Windows — {METRIC_LABELS.get(metric, metric)}",
            fontweight="bold",
        )
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{x/1e3:.0f}K" if x >= 1000 else str(int(x))
        ))

    fig.tight_layout()
    out_path = out_dir / f"window_attribution_{metric}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[图表] {out_path}", flush=True)


def plot_metric_relations(
    results_dir: pathlib.Path,
    out_dir: pathlib.Path,
) -> None:
    """
    绘制多指标时序关系图（三张子图）：
      1. Z-score 归一化多指标叠加折线图（metric_normalized_timeseries.pdf）
      2. Top-3 指标对的滞后互相关柱状图（metric_lagged_corr.pdf）
      3. 多指标联合热点矩阵（co_spike_heatmap.pdf）
    依赖 hotspot.py 生成的 metric_lagged_corr.csv / co_spike_windows.csv /
    metric_pair_summary.csv，以及可选的 window_metrics.jsonl。
    """
    lagged_f = results_dir / "metric_lagged_corr.csv"
    co_f     = results_dir / "co_spike_windows.csv"
    pair_f   = results_dir / "metric_pair_summary.csv"

    if not lagged_f.exists() and not co_f.exists():
        print("[skip] 未找到 metric_lagged_corr.csv / co_spike_windows.csv，"
              "跳过时序关系图", flush=True)
        return

    # ── 1. Z-score 归一化多指标叠加折线图 ─────────────────────────────────
    wm_candidates = list(results_dir.parent.rglob("window_metrics.jsonl"))
    if not wm_candidates:
        # 退后一层再搜一次（results 与 data 同级时）
        wm_candidates = list(results_dir.parent.parent.rglob("window_metrics.jsonl"))

    if wm_candidates:
        wm = pd.concat(
            [pd.DataFrame([json.loads(l)
                           for l in f.read_text().splitlines() if l.strip()])
             for f in wm_candidates],
            ignore_index=True,
        )
        avail = [m for m in METRICS if m in wm.columns and wm[m].sum() > 0]
        if len(avail) >= 2:
            ts = wm.groupby("window_id")[avail].sum().sort_index()
            std = ts.std().replace(0.0, 1.0)
            z = (ts - ts.mean()) / std

            fig, ax = plt.subplots(figsize=(12, 4))
            for i, col in enumerate(avail[:6]):
                ax.plot(
                    z.index, z[col],
                    color=PALETTE[i % len(PALETTE)],
                    linewidth=1.4, alpha=0.85,
                    label=METRIC_LABELS.get(col, col),
                )
            ax.axhline(2.0,  color="gray", linestyle="--", linewidth=0.8, alpha=0.55)
            ax.axhline(-2.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.55)
            ax.set_xlabel("Time Window ID")
            ax.set_ylabel("Z-score")
            ax.set_title("Normalized Multi-Metric Time Series", fontweight="bold")
            ax.legend(loc="upper right", fontsize=8, ncol=2)
            fig.tight_layout()
            out_path = out_dir / "metric_normalized_timeseries.pdf"
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            print(f"[图表] {out_path}", flush=True)

    # ── 2. 滞后互相关柱状图（top-3 指标对）─────────────────────────────────
    if lagged_f.exists() and pair_f.exists():
        lagged = pd.read_csv(lagged_f)
        pair   = pd.read_csv(pair_f)
        pair   = pair.sort_values("pearson_r", ascending=False,
                                  key=lambda s: s.abs()).head(3)

        if not lagged.empty and not pair.empty:
            n = len(pair)
            fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
            if n == 1:
                axes = [axes]

            for ax, (_, pr) in zip(axes, pair.iterrows()):
                a, b = pr["metric_a"], pr["metric_b"]
                sub = lagged[
                    (lagged["metric_a"] == a) & (lagged["metric_b"] == b)
                ].sort_values("lag")
                if sub.empty:
                    continue
                peak = int(pr["peak_lag"])
                colors = [
                    PALETTE[1] if int(row["lag"]) == peak else PALETTE[0]
                    for _, row in sub.iterrows()
                ]
                ax.bar(sub["lag"], sub["correlation"],
                       color=colors, edgecolor="none")
                ax.axhline(0, color="black", linewidth=0.7)
                ax.set_xlabel("Lag (windows)")
                ax.set_ylabel("Correlation")
                ax.set_title(
                    f"{METRIC_LABELS.get(a, a)}\nvs {METRIC_LABELS.get(b, b)}",
                    fontsize=9, fontweight="bold",
                )
                ax.annotate(
                    f"r={pr['pearson_r']:.2f}  peak_lag={peak}",
                    xy=(0.05, 0.93), xycoords="axes fraction",
                    fontsize=8, color="gray",
                )
            fig.suptitle("Lagged Cross-Correlation Between Metrics", fontweight="bold")
            fig.tight_layout()
            out_path = out_dir / "metric_lagged_corr.pdf"
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            print(f"[图表] {out_path}", flush=True)

    # ── 3. 联合热点矩阵 ────────────────────────────────────────────────────
    if co_f.exists():
        co_df = pd.read_csv(co_f)
        hot_cols = [c for c in co_df.columns if c.endswith("_hot")]
        if len(hot_cols) >= 2 and not co_df.empty:
            mat = co_df.set_index("window_id")[hot_cols].astype(float)
            mat.columns = pd.Index([
                METRIC_LABELS.get(c[:-4], c[:-4]) for c in hot_cols
            ])
            fig_h = max(2.5, 0.55 * len(hot_cols))
            fig_w = max(8,   0.22 * len(mat))
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            im = ax.imshow(mat.T, aspect="auto", cmap="Reds",
                           vmin=0, vmax=1, interpolation="nearest")
            plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Is Hot (z≥thr)")
            ax.set_yticks(range(len(mat.columns)))
            ax.set_yticklabels(mat.columns, fontsize=9)
            ax.set_xlabel("Window ID")
            ax.set_title("Co-Spike Heatmap (Window × Metric)", fontweight="bold")
            # 标注多指标同时热点窗口
            window_ids = list(mat.index)
            multi_ids = co_df[co_df["co_spike_count"] >= 2]["window_id"].tolist()
            for wid in multi_ids:
                if wid in window_ids:
                    pos = window_ids.index(wid)
                    ax.axvline(pos, color=PALETTE[1], alpha=0.65, linewidth=1.5)
            fig.tight_layout()
            out_path = out_dir / "co_spike_heatmap.pdf"
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
    plot_window_hotspots(results_dir, out_dir)
    plot_metric_relations(results_dir, out_dir)
    print(f"\n[info] 所有图表已保存至 {out_dir}", flush=True)


if __name__ == "__main__":
    main()
