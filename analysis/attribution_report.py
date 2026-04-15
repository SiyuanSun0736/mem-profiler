"""
attribution_report.py — 生成数据集级归因报告

默认读取 llvm-test-suite BCC 数据集目录，并复用现有热点/归因分析逻辑，
输出 CSV/JSONL、Markdown 摘要和可选图表。

用法：
    python analysis/attribution_report.py

    python analysis/attribution_report.py \
        --metric dtlb_misses \
        --data-root data/llvm_test_suite/bcc/O3-g \
        --output results/llvm_test_suite/aha_O3-g_attribution_report
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from datetime import datetime
from typing import Any

import pandas as pd

try:
    from . import dataset_hotspot as hotspot_analysis
    from . import dataset_hotspot_report as hotspot_plot
except ImportError:
    import dataset_hotspot as hotspot_analysis
    import dataset_hotspot_report as hotspot_plot


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "data/llvm_test_suite/bcc/O3-g"
DEFAULT_OUTPUT = REPO_ROOT / "results/llvm_test_suite/aha_O3-g_attribution_report"


def _cleanup_legacy_single_metric_outputs(out_dir: pathlib.Path) -> None:
    for file_name in ("run_hotspot_summary.csv", "run_hotspot_summary.jsonl"):
        legacy_path = out_dir / file_name
        if legacy_path.exists():
            legacy_path.unlink()


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
        values = [_stringify(row[column]) for column in existing]
        rows.append("| " + " | ".join(values) + " |")
    return [header, sep, *rows]


def _metric_summary_lines(metric: str, result: dict[str, pd.DataFrame], top_n: int) -> list[str]:
    run_df = result["run_df"]
    hotspot_df = result["hotspot_df"]
    entity_df = result["entity_df"]

    analyzed_runs = int((run_df["status"] == "analyzed").sum()) if not run_df.empty else 0
    skipped_runs = int((run_df["status"] == "skipped").sum()) if not run_df.empty else 0

    lines = [
        f"## 指标: {metric}",
        "",
        f"- run 总数: {len(run_df)}",
        f"- 已分析 run: {analyzed_runs}",
        f"- 跳过 run: {skipped_runs}",
        f"- 热点窗口总数: {len(hotspot_df)}",
        f"- 热点归因实体数: {len(entity_df)}",
        "",
        f"### Top {min(top_n, len(hotspot_df))} 热点窗口",
        "",
    ]
    lines.extend(
        _markdown_table(
            hotspot_df.head(top_n),
            [
                "run_label",
                "window_id",
                "value",
                "score",
                "window_share",
                "top_pid",
                "top_comm",
                "top_fraction",
            ],
        )
    )
    lines.extend([
        "",
        f"### Top {min(top_n, len(entity_df))} 热点归因实体",
        "",
    ])
    lines.extend(
        _markdown_table(
            entity_df.head(top_n),
            [
                "run_label",
                "comm",
                "pid",
                "tid",
                "hot_window_hits",
                "total_count",
                "mean_fraction",
            ],
        )
    )
    lines.append("")
    return lines


def write_markdown_report(
    out_dir: pathlib.Path,
    data_root: pathlib.Path,
    metrics_to_run: list[str],
    metric_results: dict[str, dict[str, pd.DataFrame]],
    overview_df: pd.DataFrame,
    top_n: int,
) -> pathlib.Path:
    report_path = out_dir / "attribution_report.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# 数据集级归因报告",
        "",
        f"- 数据目录: {data_root}",
        f"- 输出目录: {out_dir}",
        f"- 生成时间: {timestamp}",
        f"- 指标列表: {', '.join(metrics_to_run)}",
        "",
    ]

    if not overview_df.empty:
        lines.extend(["## 指标总览", ""])
        lines.extend(
            _markdown_table(
                overview_df,
                [
                    "metric",
                    "analyzed_runs",
                    "skipped_runs",
                    "hotspot_window_count",
                    "hotspot_run_count",
                    "top_run_label",
                    "top_window_id",
                    "top_score",
                    "top_pid",
                    "top_comm",
                ],
            )
        )
        lines.append("")

    for metric in metrics_to_run:
        lines.extend(_metric_summary_lines(metric, metric_results[metric], top_n=top_n))

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成默认指向 llvm-test-suite 数据集的归因报告")
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="数据根目录，默认使用 data/llvm_test_suite/bcc/O3-g",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="输出目录，默认写入 results/llvm_test_suite/aha_O3-g_attribution_report",
    )
    parser.add_argument(
        "--metric",
        choices=hotspot_analysis.METRICS,
        help="只处理单个指标；未指定时默认处理全部预定义指标",
    )
    parser.add_argument("--all-metrics", action="store_true", help="一次处理所有预定义指标")
    parser.add_argument("--metrics", nargs="+", choices=hotspot_analysis.METRICS, help="显式指定多个指标")
    parser.add_argument("--top", type=int, default=10, help="Markdown 和图表展示前 N 条")
    parser.add_argument(
        "--attribution-top",
        type=int,
        default=5,
        help="每个热点窗口保留前 N 个归因实体",
    )
    parser.add_argument(
        "--hotspot-method",
        default="zscore",
        choices=hotspot_analysis.HOTSPOT_METHODS,
        help="热点窗口检测方法",
    )
    parser.add_argument(
        "--hotspot-threshold",
        type=float,
        default=2.0,
        help="热点检测阈值，与 analysis/hotspot.py 保持一致",
    )
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

    if args.metrics:
        metrics_to_run = list(dict.fromkeys(args.metrics))
    elif args.metric and not args.all_metrics:
        metrics_to_run = [args.metric]
    else:
        metrics_to_run = list(hotspot_analysis.METRICS)

    multi_metric = len(metrics_to_run) > 1
    if multi_metric:
        _cleanup_legacy_single_metric_outputs(out_dir)

    overview_rows: list[dict[str, Any]] = []
    metric_results: dict[str, dict[str, pd.DataFrame]] = {}

    for metric in metrics_to_run:
        run_df, hotspot_df, attribution_df = hotspot_analysis.summarize_dataset_hotspots(
            data_root=data_root,
            metric=metric,
            method=args.hotspot_method,
            threshold=args.hotspot_threshold,
            top_n=args.top,
            attribution_top_n=args.attribution_top,
        )
        entity_df = hotspot_analysis.build_entity_summary(attribution_df)

        hotspot_analysis.write_metric_outputs(
            out_dir=out_dir,
            metric=metric,
            run_df=run_df,
            hotspot_df=hotspot_df,
            attribution_df=attribution_df,
            entity_df=entity_df,
            multi_metric=multi_metric,
        )

        overview_rows.append(hotspot_analysis.build_metric_overview(metric, run_df, hotspot_df))
        metric_results[metric] = {
            "run_df": run_df,
            "hotspot_df": hotspot_df,
            "entity_df": entity_df,
        }

        if not multi_metric:
            hotspot_analysis.print_report(run_df, hotspot_df, entity_df, top_n=args.top)

    overview_df = pd.DataFrame(overview_rows, columns=hotspot_analysis.OVERVIEW_COLUMNS)
    if not overview_df.empty:
        overview_df = overview_df.sort_values(["hotspot_window_count", "top_score"], ascending=[False, False])
        hotspot_analysis.write_jsonl(overview_df, out_dir / "metrics_overview.jsonl")
        overview_df.to_csv(out_dir / "metrics_overview.csv", index=False)
        if multi_metric:
            print("\n============================================================")
            print("  多指标热点总览")
            print("============================================================")
            print(overview_df.to_string(index=False))
            print("============================================================\n")

    report_path = write_markdown_report(
        out_dir=out_dir,
        data_root=data_root,
        metrics_to_run=metrics_to_run,
        metric_results=metric_results,
        overview_df=overview_df,
        top_n=args.top,
    )
    print(f"[info] Markdown 归因报告已写入 {report_path}")

    if not args.skip_figures:
        figures_dir.mkdir(parents=True, exist_ok=True)
        hotspot_plot._setup_style()
        hotspot_plot.plot_metrics_overview(out_dir, figures_dir)
        hotspot_plot.plot_dataset_hotspots(out_dir, figures_dir, top_n=args.top)
        hotspot_plot.plot_entity_hotspots(out_dir, figures_dir, top_n=args.top)
        print(f"[info] 图表输出目录: {figures_dir}")


if __name__ == "__main__":
    main()