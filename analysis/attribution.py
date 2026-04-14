"""
attribution.py — 将采样地址归因到函数/基本块级别（P2 阶段）

依赖：
  • events.jsonl（由 src/collector.py 在 --emit-events 模式下生成）
  • 目标二进制文件（含 DWARF 调试信息效果最佳，无则 fallback 到符号表）
  • symbolize.py（addr2line 封装）

P1 阶段仅完成 PID 级聚合（见 hotspot.py）。
本脚本实现 P2 目标：函数级归因，输出 function_hotspot.jsonl。

用法：
    python analysis/attribution.py \\
        --data data/run_001/ \\
        --pid 1234 \\
        --binary /path/to/target_binary \\
        --output results/run_001/ \\
        [--top 30] [--metric llc_load_misses]
"""

import argparse
import json
import pathlib
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from symbolize import symbolize_addresses, read_maps


METRICS = [
    "llc_load_misses",
    "llc_store_misses",
    "dtlb_misses",
    "minor_faults",
    "major_faults",
    "lbr_samples",
]

EVENT_TYPE_NAMES = {
    1: "llc_load_misses",
    2: "llc_store_misses",
    3: "dtlb_misses",
    4: "minor_faults",
    5: "major_faults",
    6: "lbr_samples",
}


def load_events(data_dir: pathlib.Path) -> list[dict]:
    events_f = data_dir / "events.jsonl"
    if not events_f.exists():
        sys.exit(
            f"[错误] 找不到 {events_f}\n"
            "请在采集时加上 --emit-events 以生成逐事件记录。"
        )
    return [json.loads(l) for l in events_f.read_text().splitlines() if l.strip()]


def attribute_to_functions(
    events: list[dict],
    pid: int,
    metric: str,
    top_n: int,
) -> pd.DataFrame:
    """
    对目标 PID 的 events 按 IP 地址符号化，聚合到函数级，返回 top_n 热点。
    """
    # 过滤目标 PID 和指标对应事件类型
    target_etype = {v: k for k, v in EVENT_TYPE_NAMES.items()}.get(metric)
    filtered = [
        e for e in events
        if e.get("pid") == pid and e.get("event_type") == target_etype
    ]

    if not filtered:
        print(f"[警告] PID={pid} metric={metric} 无事件记录", flush=True)
        return pd.DataFrame()

    # 收集所有唯一 IP 地址
    unique_ips = list({int(e["ip"]) for e in filtered if e.get("ip")})
    if not unique_ips:
        return pd.DataFrame()

    # 读取进程内存布局并批量符号化
    maps = read_maps(pid)
    syms = symbolize_addresses(pid, unique_ips, maps=maps)
    ip_to_sym = dict(zip(unique_ips, syms))

    # 按函数名聚合计数
    func_counts: dict[str, dict] = defaultdict(lambda: {
        "count": 0, "dso": "", "source_file": "", "source_line": 0,
    })
    for ev in filtered:
        ip = int(ev.get("ip", 0))
        sym = ip_to_sym.get(ip)
        if sym:
            key = sym.func
            func_counts[key]["count"] += 1
            func_counts[key]["dso"]         = sym.dso
            func_counts[key]["source_file"] = sym.source_file
            func_counts[key]["source_line"] = sym.source_line

    rows = [{"func": k, **v} for k, v in func_counts.items()]
    df = pd.DataFrame(rows).sort_values("count", ascending=False).head(top_n)
    total = df["count"].sum()
    df["fraction"] = df["count"] / total if total > 0 else 0.0
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="函数级访存热点归因（P2 阶段）")
    p.add_argument("--data",    required=True, help="采集数据目录")
    p.add_argument("--pid",     type=int, required=True, help="目标 PID")
    p.add_argument("--binary",  type=str, default="",
                   help="目标二进制路径（含调试信息时符号化更准确，可省略）")
    p.add_argument("--output",  required=True, help="输出目录")
    p.add_argument("--metric",  default="llc_load_misses", choices=METRICS)
    p.add_argument("--top",     type=int, default=30)
    args = p.parse_args()

    data_dir = pathlib.Path(args.data)
    out_dir  = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = load_events(data_dir)
    df = attribute_to_functions(events, args.pid, args.metric, args.top)

    if df.empty:
        print("[info] 无函数级归因结果，退出。")
        return

    print(f"\n=== 函数级热点（PID={args.pid} metric={args.metric} Top {args.top}）===")
    print(df[["func", "count", "fraction", "source_file", "source_line"]].to_string(index=False))

    # 写入 function_hotspot.jsonl
    out_f = out_dir / "function_hotspot.jsonl"
    with open(out_f, "a", encoding="utf-8") as f:
        for _, row in df.iterrows():
            rec = {
                "schema_version": "1.0",
                "pid":            args.pid,
                "symbol":         row["func"],
                "dso":            row["dso"],
                "source_file":    row["source_file"],
                "source_line":    int(row["source_line"]),
                "event_type":     args.metric,
                "count":          int(row["count"]),
                "fraction":       float(row["fraction"]),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[info] 函数热点已写入 {out_f}")

    df.to_csv(out_dir / f"function_hotspot_{args.metric}.csv", index=False)


if __name__ == "__main__":
    main()
