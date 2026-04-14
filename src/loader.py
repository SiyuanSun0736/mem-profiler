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
from filter import resolve_pid_by_comm
from exporter import Exporter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="基于 eBPF 的细粒度进程访存事件采集器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--pid",  type=int, help="目标进程 PID")
    target.add_argument("--comm", type=str, help="目标进程名（至多 15 字符）")

    p.add_argument("--window",       type=float, default=1.0,
                   help="聚合时间窗（秒），默认 1.0")
    p.add_argument("--output",       type=str,   required=True,
                   help="输出目录，不存在时自动创建")
    p.add_argument("--duration",     type=int,   default=0,
                   help="采集总时长（秒），0 = 手动 Ctrl-C 停止")
    p.add_argument("--sample-rate",  type=int,   default=100,
                   help="LLC/dTLB perf 采样率（每 N 次硬件事件触发一次 eBPF），默认 100")
    p.add_argument("--emit-events",  action="store_true",
                   help="同时向 ring buffer 写入逐事件记录（大采样量时会增加开销）")
    p.add_argument("--no-llc",       action="store_true", help="禁用 LLC miss 采样")
    p.add_argument("--no-dtlb",      action="store_true", help="禁用 dTLB miss 采样")
    p.add_argument("--no-fault",     action="store_true", help="禁用 page fault 追踪")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 解析目标 PID
    target_pid: int = 0
    target_comm: str = ""
    if args.pid:
        target_pid = args.pid
    elif args.comm:
        target_comm = args.comm
        target_pid = resolve_pid_by_comm(args.comm)
        if target_pid == 0:
            print(
                f"[警告] 未找到名为 '{args.comm}' 的进程，将采集所有进程事件",
                file=sys.stderr,
            )

    collector = Collector(
        target_pid=target_pid,
        window_sec=args.window,
        sample_rate=args.sample_rate,
        emit_events=args.emit_events,
        enable_llc=not args.no_llc,
        enable_dtlb=not args.no_dtlb,
        enable_fault=not args.no_fault,
    )

    exporter = Exporter(
        out_dir=out_dir,
        target_pid=target_pid,
        target_comm=target_comm,
        window_sec=args.window,
        sample_rate=args.sample_rate,
        enable_llc=not args.no_llc,
        enable_dtlb=not args.no_dtlb,
        enable_fault=not args.no_fault,
    )

    stop_flag = [False]

    def _on_signal(sig, frame):  # noqa: ANN001
        stop_flag[0] = True

    signal.signal(signal.SIGINT,  _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    print(f"[info] 开始采集 → {out_dir}  (Ctrl-C 停止)", flush=True)
    collector.start()

    deadline = (time.monotonic() + args.duration) if args.duration > 0 else float("inf")
    window_id = 0

    try:
        while not stop_flag[0] and time.monotonic() < deadline:
            time.sleep(args.window)
            snapshot = collector.drain_window(window_id)
            exporter.write_window(snapshot)
            window_id += 1
            print(
                f"[window {window_id:04d}] {len(snapshot.entries)} 条 PID 记录",
                flush=True,
            )
    finally:
        collector.stop()
        exporter.flush_and_close()
        print(
            f"[info] 采集结束，共 {window_id} 个时间窗，数据已保存至 {out_dir}",
            flush=True,
        )


if __name__ == "__main__":
    main()
