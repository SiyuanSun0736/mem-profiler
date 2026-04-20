"""
loader.py — eBPF 采集器入口（BCC 原型阶段）

用法示例：
    # 需要 root 权限（或 CAP_BPF + CAP_PERFMON）
    sudo python src/loader.py --pid 1234 --window 1.0 --output data/run_001/
    sudo python src/loader.py --comm nginx --window 1.0 --output data/run_001/ --duration 60
    sudo python src/loader.py --pid 1234 --emit-events --output data/run_001/
"""

import argparse
import pathlib
import signal
import sys
import time

# 将 src/ 加入搜索路径，使 collector/filter/exporter 可直接 import
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from collector import Collector
from filter import list_pids_by_comm
from exporter import Exporter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="基于 eBPF 的细粒度进程访存事件采集器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--pid",  type=int, help="目标进程 PID")
    target.add_argument(
        "--comm",
        type=str,
        help="目标进程名（至多 15 字符）；将以全局 perf 采样 + eBPF comm 过滤方式跟踪当前和后续同名进程",
    )

    p.add_argument("--window",       type=float, default=1.0,
                   help="聚合时间窗（秒），默认 1.0")
    p.add_argument("--output",       type=str,   required=True,
                   help="输出目录，不存在时自动创建")
    p.add_argument("--duration",     type=int,   default=0,
                   help="采集总时长（秒），0 = 手动 Ctrl-C 停止")
    p.add_argument("--sample-rate",  type=int,   default=100,
                   help="采样型 probe 的 perf sample_period；当前主要影响 LBR 和 legacy BCC PMU backend，默认 100")
    p.add_argument(
        "--pmu-backend",
        type=str,
        default="auto",
        choices=["auto", "perf_event_open", "bcc"],
        help="PMU 指标后端：auto 优先 perf_event_open 真计数，失败时回退到 BCC sampling",
    )
    p.add_argument("--emit-events",  action="store_true",
                   help="同时向 ring buffer 写入逐事件记录（大采样量时会增加开销）")
    p.add_argument("--tid",          type=int, default=0,
                   help="仅追踪指定 TID（线程 ID）；设置后自动启用 per-TID 聚合")
    p.add_argument("--per-tid",      action="store_true",
                   help="按 TID 聚合窗口指标，而非按 PID 聚合")
    p.add_argument("--lbr",          action="store_true",
                   help="启用 LBR 分支栈采样，并将 LBR 条目写入 events.jsonl")
    p.add_argument("--no-llc",       action="store_true", help="禁用 LLC 访问/miss 采样（PMU 组 1）")
    p.add_argument("--no-dtlb",      action="store_true", help="禁用 dTLB 访问/miss 采样（PMU 组 2）")
    p.add_argument("--no-itlb",      action="store_true", help="禁用 iTLB 访问/miss 采样（PMU 组 3）")
    p.add_argument("--no-fault",     action="store_true", help="禁用 page fault 追踪（kprobe）")
    p.add_argument(
        "--no-fault-classification",
        action="store_true",
        help="禁用增强 page fault 分类（anon/file/shared/private/write/instruction）",
    )
    p.add_argument(
        "--no-mm-syscalls",
        action="store_true",
        help="禁用 mmap/munmap/mprotect/brk 追踪（默认开启）",
    )
    p.add_argument(
        "--track-children",
        action="store_true",
        help=(
            "追踪目标进程的子进程和子线程（仅对 --pid 模式有效）。\n"
            "PMU 真计数后端会轮询 /proc 并为匹配线程打开 perf_event_open 计数器。\n"
            "legacy BCC PMU backend 则依赖 perf_event inherit=1。\n"
            "LBR: 硬件限制不支持 inherit，轮切为 pid=-1 全局挂载 + eBPF child_pid_set 过滤。\n"
            "page fault kprobe: 已是全局探针，内核 child_pid_set 控制过滤。"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 解析目标 PID / comm
    target_pid: int = 0
    target_comm: str = ""
    if args.pid:
        target_pid = args.pid
    elif args.comm:
        target_comm = args.comm[:15]
        matched_pids = list_pids_by_comm(target_comm)
        if matched_pids:
            print(
                f"[info] --comm 模式将过滤 comm='{target_comm}'；当前匹配 {len(matched_pids)} 个 PID，perf 事件将以全局模式挂载",
                file=sys.stderr,
            )
        else:
            print(
                f"[警告] 当前未找到名为 '{target_comm}' 的进程；将继续等待后续同名进程并采集其事件",
                file=sys.stderr,
            )

    collector = Collector(
        target_pid=target_pid,
        target_tid=args.tid,
        target_comm=target_comm,
        window_sec=args.window,
        sample_rate=args.sample_rate,
        emit_events=args.emit_events or args.lbr,
        enable_llc=not args.no_llc,
        enable_dtlb=not args.no_dtlb,
        enable_itlb=not args.no_itlb,
        enable_fault=not args.no_fault,
        enable_fault_classification=(not args.no_fault) and (not args.no_fault_classification),
        enable_lbr=args.lbr,
        enable_mm_syscalls=not args.no_mm_syscalls,
        per_tid=args.per_tid,
        track_children=args.track_children,
        pmu_backend=args.pmu_backend,
    )

    exporter = None

    stop_flag = [False]

    def _on_signal(sig, frame):  # noqa: ANN001
        stop_flag[0] = True

    signal.signal(signal.SIGINT,  _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    print(f"[info] 开始采集 → {out_dir}  (Ctrl-C 停止)", flush=True)
    collector.start()

    exporter = Exporter(
        out_dir=out_dir,
        target_pid=target_pid,
        target_tid=args.tid,
        target_comm=target_comm,
        window_sec=args.window,
        sample_rate=args.sample_rate,
        emit_events=args.emit_events or args.lbr,
        enable_llc=not args.no_llc,
        enable_dtlb=not args.no_dtlb,
        enable_itlb=not args.no_itlb,
        enable_fault=not args.no_fault,
        enable_fault_classification=(not args.no_fault) and (not args.no_fault_classification),
        enable_lbr=args.lbr,
        enable_mm_syscalls=not args.no_mm_syscalls,
        aggregation_scope="per_tid" if (args.per_tid or args.tid > 0) else "per_pid",
        observations=collector.describe_observations(),
        collection_backend=collector.describe_collection_backend(),
    )

    deadline = (time.monotonic() + args.duration) if args.duration > 0 else float("inf")
    window_id = 0

    try:
        while not stop_flag[0] and time.monotonic() < deadline:
            time.sleep(args.window)
            snapshot = collector.drain_window(window_id)
            exporter.write_window(snapshot)
            window_id += 1
            print(
                f"[window {window_id:04d}] {len(snapshot.entries)} 条窗口记录 / {len(snapshot.events)} 条事件",
                flush=True,
            )
    finally:
        collector.stop()
        if exporter is not None:
            exporter.flush_and_close()
        print(
            f"[info] 采集结束，共 {window_id} 个时间窗，数据已保存至 {out_dir}",
            flush=True,
        )


if __name__ == "__main__":
    main()
