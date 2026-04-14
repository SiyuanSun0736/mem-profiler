"""
collector.py — BCC 程序加载与周期性数据读取

负责：
  • 编译并加载 src/bcc_prog.c（BCC 原型版 eBPF 程序）
  • 绑定更多 perf_event（LLC / dTLB / iTLB）和 kprobe（page fault）
  • 可选按 TID 聚合，或过滤指定 TID
  • 可选消费 ring buffer，导出逐事件与 LBR 分支栈信息
"""

import ctypes
import pathlib
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from observations import build_default_observations

_BCC_PROG_PATH = pathlib.Path(__file__).parent / "bcc_prog.c"
_MAX_LBR_ENTRIES = 8

_PERF_COUNT_HW_CACHE_LL = 2
_PERF_COUNT_HW_CACHE_DTLB = 3
_PERF_COUNT_HW_CACHE_ITLB = 4
_PERF_COUNT_HW_CACHE_OP_READ = 0
_PERF_COUNT_HW_CACHE_OP_WRITE = 1
_PERF_COUNT_HW_CACHE_RESULT_ACCESS = 0
_PERF_COUNT_HW_CACHE_RESULT_MISS = 1
_PERF_COUNT_HW_BRANCH_INSTRUCTIONS = 4
_PERF_SAMPLE_BRANCH_STACK = 1 << 11
_PERF_SAMPLE_BRANCH_USER = 1 << 0
_PERF_SAMPLE_BRANCH_ANY = 1 << 3


def _cache_config(cache: int, op: int, result: int) -> int:
    return cache | (op << 8) | (result << 16)


def _decode_comm(raw: bytes | ctypes.Array[ctypes.c_char]) -> str:
    return bytes(raw).split(b"\0", 1)[0].decode("utf-8", errors="replace")


class _EntityKeyCtype(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint32),
        ("tid", ctypes.c_uint32),
    ]


class _PidMemStatsCtype(ctypes.Structure):
    _fields_ = [
        ("llc_loads", ctypes.c_uint64),
        ("llc_load_misses", ctypes.c_uint64),
        ("llc_stores", ctypes.c_uint64),
        ("llc_store_misses", ctypes.c_uint64),
        ("dtlb_loads", ctypes.c_uint64),
        ("dtlb_load_misses", ctypes.c_uint64),
        ("dtlb_stores", ctypes.c_uint64),
        ("dtlb_store_misses", ctypes.c_uint64),
        ("dtlb_misses", ctypes.c_uint64),
        ("itlb_loads", ctypes.c_uint64),
        ("itlb_load_misses", ctypes.c_uint64),
        ("minor_faults", ctypes.c_uint64),
        ("major_faults", ctypes.c_uint64),
        ("lbr_samples", ctypes.c_uint64),
        ("lbr_entries", ctypes.c_uint64),
        ("samples", ctypes.c_uint64),
        ("last_seen_ns", ctypes.c_uint64),
        ("comm", ctypes.c_char * 16),
    ]


class _LbrEntryCtype(ctypes.Structure):
    _fields_ = [
        ("from_ip", ctypes.c_uint64),
        ("to_ip", ctypes.c_uint64),
        ("flags", ctypes.c_uint64),
    ]


class _MemEventCtype(ctypes.Structure):
    _fields_ = [
        ("ts_ns", ctypes.c_uint64),
        ("pid", ctypes.c_uint32),
        ("tid", ctypes.c_uint32),
        ("comm", ctypes.c_char * 16),
        ("event_type", ctypes.c_uint8),
        ("lbr_nr", ctypes.c_uint8),
        ("_pad0", ctypes.c_uint16),
        ("_pad1", ctypes.c_uint32),
        ("addr", ctypes.c_uint64),
        ("ip", ctypes.c_uint64),
        ("lbr", _LbrEntryCtype * _MAX_LBR_ENTRIES),
    ]


@dataclass
class PidStats:
    pid: int
    tid: int
    comm: str
    llc_loads: int = 0
    llc_load_misses: int = 0
    llc_stores: int = 0
    llc_store_misses: int = 0
    dtlb_loads: int = 0
    dtlb_load_misses: int = 0
    dtlb_stores: int = 0
    dtlb_store_misses: int = 0
    dtlb_misses: int = 0
    itlb_loads: int = 0
    itlb_load_misses: int = 0
    minor_faults: int = 0
    major_faults: int = 0
    lbr_samples: int = 0
    lbr_entries: int = 0
    samples: int = 0


@dataclass
class WindowSnapshot:
    """一个时间窗内所有被追踪实体的差分指标快照。"""

    window_id: int
    start_ns: int
    end_ns: int
    entries: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)

    def add(self, delta: "PidStats") -> None:
        row = {
            "window_id": self.window_id,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "entity_scope": "tid" if delta.tid else "pid",
            "pid": delta.pid,
            "comm": delta.comm,
            "llc_loads": delta.llc_loads,
            "llc_load_misses": delta.llc_load_misses,
            "llc_stores": delta.llc_stores,
            "llc_store_misses": delta.llc_store_misses,
            "dtlb_loads": delta.dtlb_loads,
            "dtlb_load_misses": delta.dtlb_load_misses,
            "dtlb_stores": delta.dtlb_stores,
            "dtlb_store_misses": delta.dtlb_store_misses,
            "dtlb_misses": delta.dtlb_misses,
            "itlb_loads": delta.itlb_loads,
            "itlb_load_misses": delta.itlb_load_misses,
            "minor_faults": delta.minor_faults,
            "major_faults": delta.major_faults,
            "lbr_samples": delta.lbr_samples,
            "lbr_entries": delta.lbr_entries,
            "samples": delta.samples,
        }
        if delta.tid:
            row["tid"] = delta.tid
        self.entries.append(row)


class Collector:
    """BCC eBPF 采集器。"""

    def __init__(
        self,
        target_pid: int = 0,
        target_tid: int = 0,
        window_sec: float = 1.0,
        sample_rate: int = 100,
        emit_events: bool = False,
        enable_llc: bool = True,
        enable_dtlb: bool = True,
        enable_itlb: bool = True,
        enable_fault: bool = True,
        enable_lbr: bool = False,
        per_tid: bool = False,
    ) -> None:
        self.target_pid = target_pid
        self.target_tid = target_tid
        self.window_sec = window_sec
        self.sample_rate = sample_rate
        self.emit_events = emit_events or enable_lbr
        self.enable_llc = enable_llc
        self.enable_dtlb = enable_dtlb
        self.enable_itlb = enable_itlb
        self.enable_fault = enable_fault
        self.enable_lbr = enable_lbr
        self.per_tid = per_tid or target_tid > 0
        self._observations = build_default_observations(
            sample_rate=sample_rate,
            enable_llc=enable_llc,
            enable_dtlb=enable_dtlb,
            enable_itlb=enable_itlb,
            enable_fault=enable_fault,
            enable_lbr=enable_lbr,
            scope="per_tid" if self.per_tid else "per_pid",
        )

        self._bpf: Optional[Any] = None
        self._prev: dict[tuple[int, int], PidStats] = {}
        self._pending_events: list[dict] = []
        self._events_open = False

    def _make_attr(
        self,
        perf_type: int,
        config: int,
        sample_period: int,
        sample_type: int = 0,
        branch_sample_type: int = 0,
    ) -> Any:
        from bcc.perf import Perf

        attr = Perf.perf_event_attr()
        attr.type = perf_type
        attr.config = config
        attr.sample_period = sample_period
        attr.sample_type = sample_type
        attr.branch_sample_type = branch_sample_type
        return attr

    def _attach_raw_event(self, attr: Any, fn_name: str, label: str) -> bool:
        """Attach a single independent perf event (no group)."""
        if self._bpf is None:
            return False
        pid = self.target_pid if self.target_pid else -1
        try:
            self._bpf.attach_perf_event_raw(attr=attr, fn_name=fn_name.encode(), pid=pid)
            print(f"[probe] {label} sample_period={attr.sample_period}", flush=True)
            return True
        except Exception as exc:
            print(f"[警告] {label} 绑定失败 ({exc})，已跳过", flush=True)
            return False

    def _attach_perf_event_group(
        self,
        events: list[tuple[Any, str, str]],
    ) -> None:
        """Attach a list of perf events as a single PMU group.

        ``events[0]`` is the group **leader**; it is opened with
        ``group_fd=-1``.  Every subsequent entry is a group **member**
        and is opened with ``group_fd=<per_cpu_leader_fd>``.

        Benefit: the kernel PMU scheduler keeps all events in the same
        group co-scheduled — they are enabled and disabled together.
        This prevents the temporal skew that would otherwise distort
        ratios (e.g. LLC-miss-rate = misses / accesses) when the PMU
        multiplexes more events than there are hardware counters.

        On any failure the method falls back to independent attachment.
        """
        if self._bpf is None or not events:
            return

        from bcc import BPF
        from bcc.utils import get_online_cpus

        bpf = self._bpf
        pid = self.target_pid if self.target_pid else -1

        # ── 1. Attach group leader (group_fd = -1) ──────────────────────
        l_attr, l_fn, l_label = events[0]
        try:
            bpf.attach_perf_event_raw(attr=l_attr, fn_name=l_fn.encode(), pid=pid)
            print(
                f"[probe] [grp-leader] {l_label}"
                f"  sample_period={l_attr.sample_period}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[警告] {l_label} (grp-leader) 绑定失败 ({exc})，"
                f"降级为独立模式",
                flush=True,
            )
            for attr, fn, label in events:
                self._attach_raw_event(attr, fn, label)
            return

        # ── 2. Retrieve per-CPU leader fds ──────────────────────────────
        #  attach_perf_event_raw stores {cpu: efd} in
        #  bpf.open_perf_events[(attr.type, attr.config)].
        leader_key = (l_attr.type, l_attr.config)
        per_cpu_leader_fds: dict[int, int] = dict(
            bpf.open_perf_events.get(leader_key, {})
        )

        if not per_cpu_leader_fds:
            print(
                f"[警告] {l_label}: 无法获取 per-CPU leader fds，"
                f"降级为独立模式",
                flush=True,
            )
            for attr, fn, label in events[1:]:
                self._attach_raw_event(attr, fn, label)
            return

        # ── 3. Attach each group member per CPU ─────────────────────────
        for f_attr, f_fn, f_label in events[1:]:
            try:
                fn_obj = bpf.load_func(f_fn.encode(), BPF.PERF_EVENT)
            except Exception as exc:
                print(f"[警告] {f_label}: load_func 失败 ({exc})，已跳过", flush=True)
                continue

            member_fds: dict[int, int] = {}
            for cpu, gfd in per_cpu_leader_fds.items():
                try:
                    fd = bpf._attach_perf_event_raw(fn_obj.fd, f_attr, pid, cpu, gfd)
                    member_fds[cpu] = fd
                except Exception as exc:
                    print(
                        f"[警告] {f_label} CPU{cpu} 绑定失败 ({exc})", flush=True
                    )

            if member_fds:
                # Register fds so BPF cleanup() handles them properly
                bpf.open_perf_events[(f_attr.type, f_attr.config)] = member_fds
                print(
                    f"[probe] [grp-member] {f_label}"
                    f"  ({len(member_fds)}/{len(per_cpu_leader_fds)} CPUs)",
                    flush=True,
                )
            else:
                print(f"[警告] {f_label}: 全部 CPU 绑定失败，已跳过", flush=True)

    def _open_event_stream(self) -> None:
        if self._bpf is None or not self.emit_events or self._events_open:
            return
        self._bpf["events_rb"].open_ring_buffer(self._handle_ring_event)
        self._events_open = True

    def _poll_events(self) -> None:
        if self._bpf is None or not self._events_open:
            return
        self._bpf.ring_buffer_consume()

    def _handle_ring_event(self, _ctx: Any, data: Any, _size: int) -> int:
        ev = ctypes.cast(data, ctypes.POINTER(_MemEventCtype)).contents
        row = {
            "ts_ns": int(ev.ts_ns),
            "pid": int(ev.pid),
            "tid": int(ev.tid),
            "comm": _decode_comm(ev.comm),
            "event_type": int(ev.event_type),
            "addr": int(ev.addr),
            "ip": int(ev.ip),
        }
        if ev.lbr_nr:
            row["lbr"] = [
                {
                    "from_ip": int(ev.lbr[idx].from_ip),
                    "to_ip": int(ev.lbr[idx].to_ip),
                    "flags": int(ev.lbr[idx].flags),
                }
                for idx in range(min(int(ev.lbr_nr), _MAX_LBR_ENTRIES))
            ]
        self._pending_events.append(row)
        return 0

    def start(self) -> None:
        """编译并加载 eBPF 程序，绑定所有探针。"""
        try:
            from bcc import BPF
            from bcc.perf import Perf
        except ImportError:
            sys.exit(
                "[错误] BCC 未安装。\n"
                "  Ubuntu/Debian: sudo apt install python3-bcc\n"
                "  或参考: https://github.com/iovisor/bcc/blob/master/INSTALL.md"
            )

        src = _BCC_PROG_PATH.read_text()
        self._bpf = BPF(text=src)
        bpf = self._bpf

        if self.target_pid:
            target_pid_map = bpf["target_pid_map"]
            target_pid_map[target_pid_map.Key(0)] = target_pid_map.Leaf(self.target_pid)
        if self.target_tid:
            target_tid_map = bpf["target_tid_map"]
            target_tid_map[target_tid_map.Key(0)] = target_tid_map.Leaf(self.target_tid)
        if self.per_tid:
            per_tid_map = bpf["per_tid_map"]
            per_tid_map[per_tid_map.Key(0)] = per_tid_map.Leaf(1)
        if self.emit_events:
            emit_events_map = bpf["emit_events_map"]
            emit_events_map[emit_events_map.Key(0)] = emit_events_map.Leaf(1)
            self._open_event_stream()

        if self.enable_llc:
            # PMU group: all four LLC counters are scheduled together.
            # The group leader (LLC load) fires the sample; every member
            # shares the same enable/disable window, so LLC-miss-rate
            # ratios are computed from counts measured over identical PMU
            # time slices — no multiplexing skew.
            self._attach_perf_event_group([
                (                                                   # ── leader
                    self._make_attr(
                        Perf.PERF_TYPE_HW_CACHE,
                        _cache_config(
                            _PERF_COUNT_HW_CACHE_LL,
                            _PERF_COUNT_HW_CACHE_OP_READ,
                            _PERF_COUNT_HW_CACHE_RESULT_ACCESS,
                        ),
                        self.sample_rate,
                    ),
                    "on_llc_load",
                    "LLC load",
                ),
                (                                                   # ── member
                    self._make_attr(
                        Perf.PERF_TYPE_HW_CACHE,
                        _cache_config(
                            _PERF_COUNT_HW_CACHE_LL,
                            _PERF_COUNT_HW_CACHE_OP_READ,
                            _PERF_COUNT_HW_CACHE_RESULT_MISS,
                        ),
                        self.sample_rate,
                    ),
                    "on_llc_load_miss",
                    "LLC load miss",
                ),
                (                                                   # ── member
                    self._make_attr(
                        Perf.PERF_TYPE_HW_CACHE,
                        _cache_config(
                            _PERF_COUNT_HW_CACHE_LL,
                            _PERF_COUNT_HW_CACHE_OP_WRITE,
                            _PERF_COUNT_HW_CACHE_RESULT_ACCESS,
                        ),
                        self.sample_rate,
                    ),
                    "on_llc_store",
                    "LLC store",
                ),
                (                                                   # ── member
                    self._make_attr(
                        Perf.PERF_TYPE_HW_CACHE,
                        _cache_config(
                            _PERF_COUNT_HW_CACHE_LL,
                            _PERF_COUNT_HW_CACHE_OP_WRITE,
                            _PERF_COUNT_HW_CACHE_RESULT_MISS,
                        ),
                        self.sample_rate,
                    ),
                    "on_llc_store_miss",
                    "LLC store miss",
                ),
            ])

        if self.enable_dtlb:
            # PMU hardware typically provides 4 programmable counters.
            # A single group must not exceed 4 events or the kernel will
            # refuse to schedule it ("event group too large").
            # ── dTLB PMU group (4 events, at the hardware limit) ───────
            self._attach_perf_event_group([
                (                                                   # ── leader
                    self._make_attr(
                        Perf.PERF_TYPE_HW_CACHE,
                        _cache_config(
                            _PERF_COUNT_HW_CACHE_DTLB,
                            _PERF_COUNT_HW_CACHE_OP_READ,
                            _PERF_COUNT_HW_CACHE_RESULT_ACCESS,
                        ),
                        self.sample_rate,
                    ),
                    "on_dtlb_load",
                    "dTLB load",
                ),
                (                                                   # ── member
                    self._make_attr(
                        Perf.PERF_TYPE_HW_CACHE,
                        _cache_config(
                            _PERF_COUNT_HW_CACHE_DTLB,
                            _PERF_COUNT_HW_CACHE_OP_READ,
                            _PERF_COUNT_HW_CACHE_RESULT_MISS,
                        ),
                        self.sample_rate,
                    ),
                    "on_dtlb_load_miss",
                    "dTLB load miss",
                ),
                (                                                   # ── member
                    self._make_attr(
                        Perf.PERF_TYPE_HW_CACHE,
                        _cache_config(
                            _PERF_COUNT_HW_CACHE_DTLB,
                            _PERF_COUNT_HW_CACHE_OP_WRITE,
                            _PERF_COUNT_HW_CACHE_RESULT_ACCESS,
                        ),
                        self.sample_rate,
                    ),
                    "on_dtlb_store",
                    "dTLB store",
                ),
                (                                                   # ── member
                    self._make_attr(
                        Perf.PERF_TYPE_HW_CACHE,
                        _cache_config(
                            _PERF_COUNT_HW_CACHE_DTLB,
                            _PERF_COUNT_HW_CACHE_OP_WRITE,
                            _PERF_COUNT_HW_CACHE_RESULT_MISS,
                        ),
                        self.sample_rate,
                    ),
                    "on_dtlb_store_miss",
                    "dTLB store miss",
                ),
            ])

        if self.enable_itlb:
            # ── iTLB PMU group (2 events) ───────────────────────────────
            self._attach_perf_event_group([
                (                                                   # ── leader
                    self._make_attr(
                        Perf.PERF_TYPE_HW_CACHE,
                        _cache_config(
                            _PERF_COUNT_HW_CACHE_ITLB,
                            _PERF_COUNT_HW_CACHE_OP_READ,
                            _PERF_COUNT_HW_CACHE_RESULT_ACCESS,
                        ),
                        self.sample_rate,
                    ),
                    "on_itlb_load",
                    "iTLB load",
                ),
                (                                                   # ── member
                    self._make_attr(
                        Perf.PERF_TYPE_HW_CACHE,
                        _cache_config(
                            _PERF_COUNT_HW_CACHE_ITLB,
                            _PERF_COUNT_HW_CACHE_OP_READ,
                            _PERF_COUNT_HW_CACHE_RESULT_MISS,
                        ),
                        self.sample_rate,
                    ),
                    "on_itlb_load_miss",
                    "iTLB load miss",
                ),
            ])

        if self.enable_lbr:
            # LBR requires a unique sample_type/branch_sample_type combination
            # and cannot be grouped with cache events (different perf_event
            # driver domains).  It is attached independently.
            self._attach_raw_event(
                self._make_attr(
                    Perf.PERF_TYPE_HARDWARE,
                    _PERF_COUNT_HW_BRANCH_INSTRUCTIONS,
                    self.sample_rate,
                    sample_type=_PERF_SAMPLE_BRANCH_STACK,
                    branch_sample_type=_PERF_SAMPLE_BRANCH_USER | _PERF_SAMPLE_BRANCH_ANY,
                ),
                "on_lbr_sample",
                "LBR branch stack",
            )

        if self.enable_fault:
            bpf.attach_kprobe(event=b"handle_mm_fault", fn_name=b"on_page_fault")
            print("[probe] handle_mm_fault kprobe", flush=True)

    def describe_observations(self) -> list[dict]:
        return list(self._observations)

    def drain_window(self, window_id: int) -> WindowSnapshot:
        """读取 pid_stats 快照并返回一个时间窗结果。"""
        self._poll_events()

        now_ns = time.monotonic_ns()
        snap = WindowSnapshot(
            window_id=window_id,
            start_ns=now_ns - int(self.window_sec * 1e9),
            end_ns=now_ns,
        )

        if self._bpf is None:
            return snap

        raw_map = self._bpf["pid_stats"]
        current: dict[tuple[int, int], PidStats] = {}

        try:
            for key_obj, cpu_vals in raw_map.items():
                pid = int(key_obj.pid)
                tid = int(key_obj.tid)
                entity_key = (pid, tid)
                agg = PidStats(pid=pid, tid=tid, comm="")

                for cv in cpu_vals:
                    agg.llc_loads += int(cv.llc_loads)
                    agg.llc_load_misses += int(cv.llc_load_misses)
                    agg.llc_stores += int(cv.llc_stores)
                    agg.llc_store_misses += int(cv.llc_store_misses)
                    agg.dtlb_loads += int(cv.dtlb_loads)
                    agg.dtlb_load_misses += int(cv.dtlb_load_misses)
                    agg.dtlb_stores += int(cv.dtlb_stores)
                    agg.dtlb_store_misses += int(cv.dtlb_store_misses)
                    agg.dtlb_misses += int(cv.dtlb_misses)
                    agg.itlb_loads += int(cv.itlb_loads)
                    agg.itlb_load_misses += int(cv.itlb_load_misses)
                    agg.minor_faults += int(cv.minor_faults)
                    agg.major_faults += int(cv.major_faults)
                    agg.lbr_samples += int(cv.lbr_samples)
                    agg.lbr_entries += int(cv.lbr_entries)
                    agg.samples += int(cv.samples)
                    if cv.comm and not agg.comm:
                        agg.comm = _decode_comm(cv.comm)

                current[entity_key] = agg
                prev = self._prev.get(entity_key)
                if prev is None:
                    snap.add(agg)
                    continue

                snap.add(
                    PidStats(
                        pid=pid,
                        tid=tid,
                        comm=agg.comm,
                        llc_loads=max(0, agg.llc_loads - prev.llc_loads),
                        llc_load_misses=max(0, agg.llc_load_misses - prev.llc_load_misses),
                        llc_stores=max(0, agg.llc_stores - prev.llc_stores),
                        llc_store_misses=max(0, agg.llc_store_misses - prev.llc_store_misses),
                        dtlb_loads=max(0, agg.dtlb_loads - prev.dtlb_loads),
                        dtlb_load_misses=max(0, agg.dtlb_load_misses - prev.dtlb_load_misses),
                        dtlb_stores=max(0, agg.dtlb_stores - prev.dtlb_stores),
                        dtlb_store_misses=max(0, agg.dtlb_store_misses - prev.dtlb_store_misses),
                        dtlb_misses=max(0, agg.dtlb_misses - prev.dtlb_misses),
                        itlb_loads=max(0, agg.itlb_loads - prev.itlb_loads),
                        itlb_load_misses=max(0, agg.itlb_load_misses - prev.itlb_load_misses),
                        minor_faults=max(0, agg.minor_faults - prev.minor_faults),
                        major_faults=max(0, agg.major_faults - prev.major_faults),
                        lbr_samples=max(0, agg.lbr_samples - prev.lbr_samples),
                        lbr_entries=max(0, agg.lbr_entries - prev.lbr_entries),
                        samples=max(0, agg.samples - prev.samples),
                    )
                )

        except Exception as exc:
            print(f"[警告] 读取 pid_stats 失败: {exc}", file=sys.stderr, flush=True)

        self._prev = current
        if self._pending_events:
            snap.events.extend(self._pending_events)
            self._pending_events = []
        return snap

    def stop(self) -> None:
        """卸载 eBPF 程序并释放资源。"""
        if self._bpf is not None:
            self._poll_events()
            self._bpf.cleanup()
            self._bpf = None
            print("[info] eBPF 程序已卸载", flush=True)
