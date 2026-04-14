"""
to_baseline.py — 将本库产出格式转换为 ebpf-mem-analyzer 可消费的输入格式

ebpf-mem-analyzer 基线仓库期望的输入格式：
  • 扁平 CSV，每行一条样本
  • 关键列：pid, comm, window_id, llc_misses, page_faults, ...

本脚本从 window_metrics.jsonl 读取，做最小化列映射后写出 baseline_input.csv。
需要根据 ebpf-mem-analyzer 的实际输入格式更新列映射（见 COLUMN_MAP）。

用法：
    python export/to_baseline.py \\
        --input  data/run_001/ \\
        --output /path/to/ebpf-mem-analyzer/data/new_input/ \\
        [--run-id <uuid>]
"""

import argparse
import json
import pathlib
import sys

import pandas as pd


# --------------------------------------------------------------------------
# 列名映射：本库列名 → ebpf-mem-analyzer 期望的列名
# 根据 ebpf-mem-analyzer 的实际接口更新此映射即可，无需修改其他代码。
# --------------------------------------------------------------------------
COLUMN_MAP: dict[str, str] = {
    "window_id":        "window_id",
    "pid":              "pid",
    "comm":             "comm",
    "llc_load_misses":  "llc_misses",   # 基线仓库合并了 load/store
    "dtlb_misses":      "dtlb_misses",
    "minor_faults":     "minor_faults",
    "major_faults":     "major_faults",
    "samples":          "total_samples",
}


def load_window_metrics(data_dir: pathlib.Path, run_id: str | None) -> pd.DataFrame:
    f = data_dir / "window_metrics.jsonl"
    if not f.exists():
        sys.exit(f"[错误] 找不到 {f}")
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    if run_id:
        df = df[df["run_id"] == run_id]
        if df.empty:
            sys.exit(f"[错误] run_id={run_id} 无匹配记录")
    return df


def convert(df: pd.DataFrame) -> pd.DataFrame:
    """应用列映射并补充派生列。"""
    # 合并 llc_load + store 为 llc_misses
    df = df.copy()
    df["llc_load_misses"] = df.get("llc_load_misses", 0).fillna(0).astype(int)
    df["llc_store_misses"] = df.get("llc_store_misses", 0).fillna(0).astype(int)
    df["llc_misses"] = df["llc_load_misses"] + df["llc_store_misses"]

    available = {src: dst for src, dst in COLUMN_MAP.items() if src in df.columns}
    out = df[list(available.keys())].rename(columns=available)

    # 计算时间窗时长（秒）
    if "start_ns" in df.columns and "end_ns" in df.columns:
        out["window_duration_sec"] = (
            (df["end_ns"] - df["start_ns"]) / 1e9
        ).round(4)

    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="将 ebpf-mem-profiler 产出转换为 ebpf-mem-analyzer 输入格式"
    )
    p.add_argument("--input",   required=True, help="本库数据目录（含 window_metrics.jsonl）")
    p.add_argument("--output",  required=True, help="目标输出目录")
    p.add_argument("--run-id",  default=None,  help="仅转换指定 run_id（默认全部）")
    args = p.parse_args()

    in_dir  = pathlib.Path(args.input)
    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_window_metrics(in_dir, args.run_id)
    df_out = convert(df_raw)

    out_path = out_dir / "baseline_input.csv"
    df_out.to_csv(out_path, index=False)
    print(f"[info] 转换完成：{len(df_out)} 行 → {out_path}")
    print(f"[info] 列: {list(df_out.columns)}")


if __name__ == "__main__":
    main()
