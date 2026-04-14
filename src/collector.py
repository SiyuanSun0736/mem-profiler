"""
collector.py — BCC 程序加载与周期性数据读取

负责：
  • 编译并加载 src/bcc_prog.c（BCC 原型版 eBPF 程序）
  • 绑定 perf_event（LLC miss、dTLB miss）和 kprobe（page fault）
  • 通过 drain_window() 读取 pid_stats PERCPU_HASH，计算差分后返回快照
"""

import ctypes
import pathlib
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

from observations import build_default_observations

_BCC_PROG_PATH = pathlib.Path(__file__).parent / "bcc_prog.c"


# ---- 数据结构（与 bcc_prog.c 中的 pid_mem_stats_t 字段顺序完全对应） ----

class _PidMemStatsCtype(ctypes.Structure):
    _fields_ = [
        ("llc_load_misses",  ctypes.c_uint64),
        ("llc_store_misses", ctypes.c_uint64),
        ("dtlb_misses",      ctypes.c_uint64),
        ("minor_faults",     ctypes.c_uint64),
        ("major_faults",     ctypes.c_uint64),
        ("samples",          ctypes.c_uint64),
        ("last_seen_ns",     ctypes.c_uint64),
        ("comm",             ctypes.c_char * 16),
    ]


@dataclass
class PidStats:
    pid: int
    comm: str
    llc_load_misses:  int = 0
    llc_store_misses: int = 0
    dtlb_misses:      int = 0
    minor_faults:     int = 0
    major_faults:     int = 0
    samples:          int = 0


@dataclass
class WindowSnapshot:
    """一个时间窗内所有被追踪 PID 的差分指标快照。"""
    window_id: int
    start_ns:  int
    end_ns:    int
    entries:   list[dict] = field(default_factory=list)

    def add(self, pid: int, delta: "PidStats") -> None:
        self.entries.append({
            "window_id":        self.window_id,
            "start_ns":         self.start_ns,
            "end_ns":           self.end_ns,
            "pid":              pid,
            "comm":             delta.comm,
            "llc_load_misses":  delta.llc_load_misses,
            "llc_store_misses": delta.llc_store_misses,
            "dtlb_misses":      delta.dtlb_misses,
            "minor_faults":     delta.minor_faults,
            "major_faults":     delta.major_faults,
            "samples":          delta.samples,
        })


class Collector:
    """
    BCC eBPF 采集器。

    参数
    ----
    target_pid  : 0 = 所有进程，非 0 = 仅追踪指定 PID
    window_sec  : 时间窗大小（秒），用于计算 start/end_ns
    sample_rate : perf 事件采样比（每 N 次硬件事件触发一次 eBPF handler）
    emit_events : 是否向 ring buffer 输出逐事件记录
    enable_*    : 分别控制三类探针的启用
    """

    def __init__(
        self,
        target_pid:   int   = 0,
        window_sec:   float = 1.0,
        sample_rate:  int   = 100,
        emit_events:  bool  = False,
        enable_llc:   bool  = True,
        enable_dtlb:  bool  = True,
        enable_fault: bool  = True,
    ) -> None:
        self.target_pid   = target_pid
        self.window_sec   = window_sec
        self.sample_rate  = sample_rate
        self.emit_events  = emit_events
        self.enable_llc   = enable_llc
        self.enable_dtlb  = enable_dtlb
        self.enable_fault = enable_fault
        self._observations = build_default_observations(
            sample_rate=sample_rate,
            enable_llc=enable_llc,
            enable_dtlb=enable_dtlb,
            enable_fault=enable_fault,
        )

        self._bpf: Optional[object] = None
        self._prev: dict[int, PidStats] = {}

    # ------------------------------------------------------------------

    def start(self) -> None:
        """编译并加载 eBPF 程序，绑定所有探针。"""
        try:
            from bcc import BPF, PerfType, PerfHWConfig
        except ImportError:
            sys.exit(
                "[错误] BCC 未安装。\n"
                "  Ubuntu/Debian: sudo apt install python3-bcc\n"
                "  或参考: https://github.com/iovisor/bcc/blob/master/INSTALL.md"
            )

        src = _BCC_PROG_PATH.read_text()
        self._bpf = BPF(text=src)
        bpf = self._bpf

        # 写入目标 PID 过滤值
        if self.target_pid:
            target_pid_map = bpf["target_pid_map"]
            target_pid_map[target_pid_map.Key(0)] = target_pid_map.Leaf(self.target_pid)

        # LLC load miss（CACHE_MISSES 是最通用的近似）
        if self.enable_llc:
            bpf.attach_perf_event(
                ev_type=PerfType.HARDWARE,
                ev_config=PerfHWConfig.CACHE_MISSES,
                fn_name=b"on_llc_load_miss",
                sample_period=self.sample_rate,
            )
            print(f"[probe] LLC load miss  sample_period={self.sample_rate}", flush=True)

        # dTLB miss（部分硬件用 RAW event，此处 fallback 到 CACHE_MISSES × 10）
        if self.enable_dtlb:
            try:
                bpf.attach_perf_event(
                    ev_type=PerfType.HARDWARE,
                    ev_config=PerfHWConfig.CACHE_MISSES,
                    fn_name=b"on_dtlb_miss",
                    sample_period=self.sample_rate * 10,
                )
                print("[probe] dTLB miss (fallback to CACHE_MISSES×10)", flush=True)
            except Exception as exc:
                print(f"[警告] dTLB perf_event 绑定失败 ({exc})，已跳过", flush=True)

        # page fault kprobe
        if self.enable_fault:
            bpf.attach_kprobe(
                event=b"handle_mm_fault",
                fn_name=b"on_page_fault",
            )
            print("[probe] handle_mm_fault kprobe", flush=True)

    def describe_observations(self) -> list[dict]:
        """返回本次运行的 observation 元数据，用于写入 run_metadata。"""
        return list(self._observations)

    # ------------------------------------------------------------------

    def drain_window(self, window_id: int) -> WindowSnapshot:
        """
        读取 pid_stats map 快照，与上一窗口做差分，清空 map，返回 WindowSnapshot。
        PERCPU_HASH 的每个值是长度为 num_cpu 的数组，需跨 CPU 求和。
        """
        now_ns = time.monotonic_ns()
        snap = WindowSnapshot(
            window_id=window_id,
            start_ns=now_ns - int(self.window_sec * 1e9),
            end_ns=now_ns,
        )

        if self._bpf is None:
            return snap

        raw_map = self._bpf["pid_stats"]
        current: dict[int, PidStats] = {}

        try:
            for k, cpu_vals in raw_map.items():
                pid = k.value
                agg = PidStats(pid=pid, comm="")
                for cv in cpu_vals:
                    agg.llc_load_misses  += cv.llc_load_misses
                    agg.llc_store_misses += cv.llc_store_misses
                    agg.dtlb_misses      += cv.dtlb_misses
                    agg.minor_faults     += cv.minor_faults
                    agg.major_faults     += cv.major_faults
                    agg.samples          += cv.samples
                    if cv.comm and not agg.comm:
                        agg.comm = cv.comm.decode("utf-8", errors="replace")

                current[pid] = agg

                prev = self._prev.get(pid)
                if prev:
                    delta = PidStats(
                        pid=pid,
                        comm=agg.comm,
                        llc_load_misses=  max(0, agg.llc_load_misses  - prev.llc_load_misses),
                        llc_store_misses= max(0, agg.llc_store_misses - prev.llc_store_misses),
                        dtlb_misses=      max(0, agg.dtlb_misses      - prev.dtlb_misses),
                        minor_faults=     max(0, agg.minor_faults     - prev.minor_faults),
                        major_faults=     max(0, agg.major_faults     - prev.major_faults),
                        samples=          max(0, agg.samples          - prev.samples),
                    )
                    snap.add(pid, delta)
                else:
                    snap.add(pid, agg)

        except Exception as exc:
            print(f"[警告] 读取 pid_stats 失败: {exc}", file=sys.stderr, flush=True)

        self._prev = current
        return snap

    # ------------------------------------------------------------------

    def stop(self) -> None:
        """卸载 eBPF 程序并释放资源。"""
        if self._bpf is not None:
            self._bpf.cleanup()
            self._bpf = None
            print("[info] eBPF 程序已卸载", flush=True)
