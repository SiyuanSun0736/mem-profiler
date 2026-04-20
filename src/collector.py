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
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

# perf_event_attr.flags 中表示 inherit 的永久位置（kernel 规范：踤足 0=disabled, 1=inherit …）
_PERF_ATTR_FLAG_INHERIT: int = 1 << 1

from observations import build_default_observations
from perf_counter import PerfCounterBackend, PerfCounterSnapshot

_BCC_PROG_PATH = pathlib.Path(__file__).parent / "bcc_prog.c"
_MAX_LBR_ENTRIES = 8
_STALE_ENTITY_WINDOW_MULTIPLIER = 5.0
_MIN_STALE_ENTITY_TIMEOUT_NS = 1_000_000_000


def _expand_bcc_source(src_path: pathlib.Path) -> str:
    """Inline-expand local #include "..." so all BPF entry-point functions
    appear in the main source text.  BCC only recognises functions defined
    directly in the top-level text as programs; functions that arrive via
    #include'd headers are invisible to bpf_function_start()."""
    src_dir = src_path.parent
    seen: set[pathlib.Path] = {src_path.resolve()}

    def expand(text: str) -> str:
        def replacer(match: re.Match) -> str:
            fname = match.group(1)
            fpath = (src_dir / fname).resolve()
            if fpath in seen or not fpath.exists():
                return match.group(0)  # keep unknown / already-expanded
            seen.add(fpath)
            content = fpath.read_text()
            # Strip #pragma once so the inlined content doesn't break
            content = re.sub(r'^[ \t]*#pragma[ \t]+once[ \t]*\n',
                             '', content, flags=re.MULTILINE)
            return expand(content)   # recurse for nested local includes

        return re.sub(r'#include\s+"([^"]+)"', replacer, text)

    return expand(src_path.read_text())

_PERF_COUNT_HW_CACHE_LL = 2
_PERF_COUNT_HW_CACHE_DTLB = 3
_PERF_COUNT_HW_CACHE_ITLB = 4
_PERF_COUNT_HW_CACHE_OP_READ = 0
_PERF_COUNT_HW_CACHE_OP_WRITE = 1
_PERF_COUNT_HW_CACHE_RESULT_ACCESS = 0
_PERF_COUNT_HW_CACHE_RESULT_MISS = 1
_PERF_COUNT_HW_CPU_CYCLES = 0          # PERF_TYPE_HARDWARE CPU cycles
_PERF_COUNT_HW_INSTRUCTIONS = 1        # PERF_TYPE_HARDWARE retired instructions
_PERF_COUNT_HW_BRANCH_INSTRUCTIONS = 4
_PERF_COUNT_HW_CACHE_REFERENCES = 4   # PERF_TYPE_HARDWARE generic cache references
_PERF_COUNT_HW_CACHE_MISSES = 5       # PERF_TYPE_HARDWARE generic cache misses
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
        ("itlb_load_misses", ctypes.c_uint64),
        ("cycles", ctypes.c_uint64),
        ("instructions", ctypes.c_uint64),
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


class _TaskCommFilterCtype(ctypes.Structure):
    _fields_ = [("comm", ctypes.c_char * 16)]


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
    itlb_load_misses: int = 0
    cycles: int = 0
    instructions: int = 0
    minor_faults: int = 0
    major_faults: int = 0
    lbr_samples: int = 0
    lbr_entries: int = 0
    samples: int = 0
    last_seen_ns: int = 0


@dataclass
class WindowSnapshot:
    """一个时间窗内所有被追踪实体的差分指标快照。"""

    window_id: int
    start_ns: int
    end_ns: int
    entries: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)

    @staticmethod
    def _is_zero_delta(delta: "PidStats") -> bool:
        return all(
            value == 0
            for value in (
                delta.llc_loads,
                delta.llc_load_misses,
                delta.llc_stores,
                delta.llc_store_misses,
                delta.dtlb_loads,
                delta.dtlb_load_misses,
                delta.dtlb_stores,
                delta.dtlb_store_misses,
                delta.dtlb_misses,
                delta.itlb_load_misses,
                delta.cycles,
                delta.instructions,
                delta.minor_faults,
                delta.major_faults,
                delta.lbr_samples,
                delta.lbr_entries,
                delta.samples,
            )
        )

    def add(self, delta: "PidStats") -> None:
        if self._is_zero_delta(delta):
            return

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
            "itlb_load_misses": delta.itlb_load_misses,
            "cycles": delta.cycles,
            "instructions": delta.instructions,
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
        target_comm: str = "",
        window_sec: float = 1.0,
        sample_rate: int = 100,
        emit_events: bool = False,
        enable_llc: bool = True,
        enable_dtlb: bool = True,
        enable_itlb: bool = True,
        enable_fault: bool = True,
        enable_lbr: bool = False,
        per_tid: bool = False,
        track_children: bool = False,
        pmu_backend: str = "auto",
    ) -> None:
        self.target_pid = target_pid
        self.target_tid = target_tid
        self.target_comm = target_comm[:15]
        self.window_sec = window_sec
        self.sample_rate = sample_rate
        self.emit_events = emit_events or enable_lbr
        self.enable_llc = enable_llc
        self.enable_dtlb = enable_dtlb
        self.enable_itlb = enable_itlb
        self.enable_fault = enable_fault
        self.enable_lbr = enable_lbr
        self.per_tid = per_tid or target_tid > 0
        self.pmu_backend = pmu_backend
        self._use_bcc_pmu = pmu_backend == "bcc"
        self._llc_store_via_generic = False
        self._perf_backend: Optional[PerfCounterBackend] = None
        # track_children 只在指定了具体 target_pid 时才生效；
        # comm 模式本就是全局采样，子进程总能被内核防弹层捕获。
        self.track_children: bool = track_children and (target_pid > 0)
        self._observations: list[dict] = []
        self._refresh_observations()
        self._stale_entity_timeout_ns = max(
            int(self.window_sec * _STALE_ENTITY_WINDOW_MULTIPLIER * 1e9),
            _MIN_STALE_ENTITY_TIMEOUT_NS,
        )

        self._bpf: Optional[Any] = None
        self._prev: dict[tuple[int, int], PidStats] = {}
        self._pending_events: list[dict] = []
        self._events_open = False
        # 子进程追踪（track_children=True 时开启）
        self._tracked_child_pids: set[int] = set()
        self._child_monitor_lock: threading.Lock = threading.Lock()
        self._child_monitor_stop: Optional[threading.Event] = None
        self._child_monitor_thread: Optional[threading.Thread] = None

    def _refresh_observations(self) -> None:
        self._observations = build_default_observations(
            sample_rate=self.sample_rate,
            enable_llc=self.enable_llc,
            enable_dtlb=self.enable_dtlb,
            enable_itlb=self.enable_itlb,
            enable_fault=self.enable_fault,
            enable_lbr=self.enable_lbr,
            scope="per_tid" if self.per_tid else "per_pid",
            llc_store_via_generic=self._llc_store_via_generic,
            pmu_backend="bcc" if self._use_bcc_pmu else "perf_event_open",
        )

    def describe_collection_backend(self) -> str:
        if self._use_bcc_pmu:
            return "bcc"
        if self.enable_fault or self.enable_lbr:
            return "hybrid_perf_event_open_bcc"
        return "perf_event_open"

    def _make_attr(
        self,
        perf_type: int,
        config: int,
        sample_period: int,
        sample_type: int = 0,
        branch_sample_type: int = 0,
        inherit: bool = False,
    ) -> Any:
        from bcc.perf import Perf

        attr = Perf.perf_event_attr()
        attr.type = perf_type
        attr.config = config
        attr.sample_period = sample_period
        attr.sample_type = sample_type
        attr.branch_sample_type = branch_sample_type
        if inherit:
            # 设置 perf_event_attr.inherit 位（flags 第 1 位）。
            # BCC 的 ctypes 结构将位字段打包进the单个 flags u64字段；
            # 直接 attr.inherit = 1 在部分 BCC 版本下不会卸写 C 结构内存。
            attr.flags = getattr(attr, "flags", 0) | _PERF_ATTR_FLAG_INHERIT
        return attr

    def _attach_raw_event(
        self, attr: Any, fn_name: str, label: str,
        pid_override: Optional[int] = None,
    ) -> bool:
        """Attach a single independent perf event (no group).

        pid_override: 若提供，采用该 pid 而不是 self.target_pid。
            传入 -1 即表示全局挂载（所有进程）。
        """
        if self._bpf is None:
            return False
        if pid_override is not None:
            pid = pid_override
        else:
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
        inherit: bool = False,
        fallback_to_independent: bool = True,
    ) -> bool:
        """Attach a list of perf events as a single PMU group.

        ``events[0]`` is the group **leader**; it is opened with
        ``group_fd=-1``.  Every subsequent entry is a group **member**
        and is opened with ``group_fd=<per_cpu_leader_fd>``.

        Benefit: the kernel PMU scheduler keeps all events in the same
        group co-scheduled — they are enabled and disabled together.
        This prevents the temporal skew that would otherwise distort
        ratios (e.g. LLC-miss-rate = misses / accesses) when the PMU
        multiplexes more events than there are hardware counters.

        inherit=True: 为所有 attr 设置 inherit 位，使内核将事件继承给子进程。
            注意：LBR（PERF_SAMPLE_BRANCH_STACK）不支持 inherit，只应用于
            LLC / dTLB / iTLB 这类 PMU 计数器事件组。

        fallback_to_independent=False: 当 leader 绑定失败时不做独立降级，
            直接返回 False，由调用方决定 fallback 策略。

        Returns True if the leader was successfully attached (members may
        have partially failed), False if the leader itself failed.
        """
        if self._bpf is None or not events:
            return False

        if inherit:
            for attr, _, _ in events:
                attr.flags = getattr(attr, "flags", 0) | _PERF_ATTR_FLAG_INHERIT

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
                f"{'降级为独立模式' if fallback_to_independent else '跳过'}",
                flush=True,
            )
            if fallback_to_independent:
                for attr, fn, label in events:
                    self._attach_raw_event(attr, fn, label)
            return False

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
            return True

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

        return True

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

        src = _expand_bcc_source(_BCC_PROG_PATH)
        src_dir = str(_BCC_PROG_PATH.parent)
        self._bpf = BPF(text=src, cflags=[f"-I{src_dir}"])
        bpf = self._bpf

        if self.target_pid:
            target_pid_map = bpf["target_pid_map"]
            target_pid_map[target_pid_map.Key(0)] = target_pid_map.Leaf(self.target_pid)
        if self.target_tid:
            target_tid_map = bpf["target_tid_map"]
            target_tid_map[target_tid_map.Key(0)] = target_tid_map.Leaf(self.target_tid)
        if self.target_comm:
            target_comm_map = bpf["target_comm_map"]
            comm_filter = _TaskCommFilterCtype()
            comm_filter.comm = self.target_comm.encode("utf-8", errors="ignore")[:15]
            target_comm_map[target_comm_map.Key(0)] = comm_filter
        if self.per_tid:
            per_tid_map = bpf["per_tid_map"]
            per_tid_map[per_tid_map.Key(0)] = per_tid_map.Leaf(1)
        if self.emit_events:
            emit_events_map = bpf["emit_events_map"]
            emit_events_map[emit_events_map.Key(0)] = emit_events_map.Leaf(1)
            self._open_event_stream()

        if self.pmu_backend in {"auto", "perf_event_open"}:
            try:
                perf_backend = PerfCounterBackend(
                    target_pid=self.target_pid,
                    target_tid=self.target_tid,
                    target_comm=self.target_comm,
                    per_tid=self.per_tid,
                    track_children=self.track_children,
                    enable_llc=self.enable_llc,
                    enable_dtlb=self.enable_dtlb,
                    enable_itlb=self.enable_itlb,
                )
                perf_backend.start()
                self._perf_backend = perf_backend
                self._use_bcc_pmu = False
                self._llc_store_via_generic = perf_backend.uses_llc_store_proxy()
                print(
                    "[info] PMU backend = perf_event_open (time_enabled/time_running)",
                    flush=True,
                )
            except Exception as exc:
                if self._perf_backend is not None:
                    self._perf_backend.stop()
                    self._perf_backend = None
                if self.pmu_backend == "perf_event_open":
                    raise RuntimeError(f"perf_event_open PMU backend 初始化失败: {exc}") from exc
                self._use_bcc_pmu = True
                print(
                    f"[警告] perf_event_open PMU backend 初始化失败 ({exc})，降级为 BCC sampling",
                    flush=True,
                )
        else:
            self._use_bcc_pmu = True

        self._refresh_observations()

        if self._use_bcc_pmu:
            # ── cycles + instructions group ─────────────────────────────────
            # These are fundamental hardware counters used to derive IPC and
            # MPKI.  Grouped so the kernel keeps them co-scheduled and their
            # ratio remains meaningful under PMU multiplexing.
            self._attach_perf_event_group([
                (
                    self._make_attr(
                        Perf.PERF_TYPE_HARDWARE,
                        _PERF_COUNT_HW_CPU_CYCLES,
                        self.sample_rate,
                    ),
                    "on_cycles",
                    "CPU cycles",
                ),
                (
                    self._make_attr(
                        Perf.PERF_TYPE_HARDWARE,
                        _PERF_COUNT_HW_INSTRUCTIONS,
                        self.sample_rate,
                    ),
                    "on_instructions",
                    "hardware instructions",
                ),
            ], inherit=self.track_children)

        if self.enable_llc and self._use_bcc_pmu:
            # ── LLC read group: load (leader) + load miss (member) ──────
            # These events are universally supported for sampling.
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
            ], inherit=self.track_children)

            # ── LLC write group: store + store miss ──────────────────────
            # Some CPUs (e.g. Intel Skylake) don't support LL-write as a
            # perf sampling source.  We try the native HW_CACHE events
            # first; if the leader itself fails (EINVAL), fall back to
            # generic HARDWARE CACHE_REFERENCES / CACHE_MISSES events.
            write_ok = self._attach_perf_event_group([
                (                                                   # ── leader
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
            ], inherit=self.track_children, fallback_to_independent=False)

            if not write_ok:
                # Native LLC write sampling unsupported; use generic
                # CACHE_REFERENCES (all LLC accesses) and CACHE_MISSES
                # (all LLC misses) as proxy events.
                print(
                    "[info] LLC write 采样不受硬件支持，"
                    "降级为 generic cache-references / cache-misses",
                    flush=True,
                )
                self._llc_store_via_generic = True
                self._attach_perf_event_group([
                    (                                               # ── leader
                        self._make_attr(
                            Perf.PERF_TYPE_HARDWARE,
                            _PERF_COUNT_HW_CACHE_REFERENCES,
                            self.sample_rate,
                        ),
                        "on_llc_store",
                        "cache-references [LLC store proxy]",
                    ),
                    (                                               # ── member
                        self._make_attr(
                            Perf.PERF_TYPE_HARDWARE,
                            _PERF_COUNT_HW_CACHE_MISSES,
                            self.sample_rate,
                        ),
                        "on_llc_store_miss",
                        "cache-misses [LLC store miss proxy]",
                    ),
                ], inherit=self.track_children)
                self._refresh_observations()

        if self.enable_dtlb and self._use_bcc_pmu:
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
            ], inherit=self.track_children)

        if self.enable_itlb and self._use_bcc_pmu:
            # ── iTLB load miss (independent event) ─────────────────────
            # iTLB load access (HW_CACHE_RESULT_ACCESS) is not supported
            # on many Intel processors.  Attach only the miss counter as
            # an independent event so it is never blocked by a failing
            # group leader.
            self._attach_raw_event(
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
            )

        if self.enable_lbr:
            # LBR requires a unique sample_type/branch_sample_type combination
            # and cannot be grouped with cache events (different perf_event
            # driver domains).  It is attached independently.
            #
            # 硬件限制：内核明确拒绝 PERF_SAMPLE_BRANCH_STACK + inherit=1 的组合。
            # 当 track_children=True 时，改用 pid=-1（全局挂载），
            # 依赖 eBPF task_allowed() + child_pid_set 在内核层过滤子代。
            lbr_pid = -1 if self.track_children else None
            self._attach_raw_event(
                self._make_attr(
                    Perf.PERF_TYPE_HARDWARE,
                    _PERF_COUNT_HW_BRANCH_INSTRUCTIONS,
                    self.sample_rate,
                    sample_type=_PERF_SAMPLE_BRANCH_STACK,
                    branch_sample_type=_PERF_SAMPLE_BRANCH_USER | _PERF_SAMPLE_BRANCH_ANY,
                    # inherit 配置此处故意不开启：硬件不支持。
                ),
                "on_lbr_sample",
                "LBR branch stack",
                pid_override=lbr_pid,
            )
            if self.track_children:
                print("[probe] LBR 全局挂载（pid=-1），依赖 eBPF child_pid_set 过滤子进程",
                      flush=True)

        if self.enable_fault:
            bpf.attach_kprobe(event=b"handle_mm_fault", fn_name=b"on_page_fault")
            print("[probe] handle_mm_fault kprobe", flush=True)

        if self.track_children and (self.enable_fault or self.enable_lbr):
            print(f"[info] track_children 已开启，启动子进程监控线程", flush=True)
            self._start_child_monitor()

    def describe_observations(self) -> list[dict]:
        return list(self._observations)

    def _prune_stale_entities(
        self,
        raw_map: Any,
        current: dict[tuple[int, int], PidStats],
        now_ns: int,
    ) -> None:
        stale_before_ns = now_ns - self._stale_entity_timeout_ns
        stale_keys = [
            entity_key
            for entity_key, stats in current.items()
            if stats.last_seen_ns > 0 and stats.last_seen_ns <= stale_before_ns
        ]

        if not stale_keys:
            return

        for pid, tid in stale_keys:
            current.pop((pid, tid), None)
            self._prev.pop((pid, tid), None)
            try:
                del raw_map[raw_map.Key(pid=pid, tid=tid)]
            except Exception as exc:
                print(
                    f"[警告] 清理陈旧 pid_stats 条目失败: pid={pid} tid={tid} ({exc})",
                    file=sys.stderr,
                    flush=True,
                )

    def drain_window(self, window_id: int) -> WindowSnapshot:
        """读取 pid_stats 快照并返回一个时间窗结果。"""
        self._poll_events()

        now_ns = time.monotonic_ns()
        snap = WindowSnapshot(
            window_id=window_id,
            start_ns=now_ns - int(self.window_sec * 1e9),
            end_ns=now_ns,
        )

        if self._bpf is None and self._perf_backend is None:
            return snap

        raw_map = self._bpf["pid_stats"] if self._bpf is not None else None
        current: dict[tuple[int, int], PidStats] = {}

        if raw_map is not None:
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
                        agg.itlb_load_misses += int(cv.itlb_load_misses)
                        agg.cycles += int(cv.cycles)
                        agg.instructions += int(cv.instructions)
                        agg.minor_faults += int(cv.minor_faults)
                        agg.major_faults += int(cv.major_faults)
                        agg.lbr_samples += int(cv.lbr_samples)
                        agg.lbr_entries += int(cv.lbr_entries)
                        agg.samples += int(cv.samples)
                        agg.last_seen_ns = max(agg.last_seen_ns, int(cv.last_seen_ns))
                        if cv.comm and not agg.comm:
                            agg.comm = _decode_comm(cv.comm)

                    current[entity_key] = agg

            except Exception as exc:
                print(f"[警告] 读取 pid_stats 失败: {exc}", file=sys.stderr, flush=True)

        if self._perf_backend is not None:
            perf_current = self._perf_backend.read()
            for entity_key, perf_stats in perf_current.items():
                agg = current.get(entity_key)
                if agg is None:
                    agg = PidStats(
                        pid=perf_stats.pid,
                        tid=perf_stats.tid,
                        comm=perf_stats.comm,
                    )
                    current[entity_key] = agg
                elif perf_stats.comm and not agg.comm:
                    agg.comm = perf_stats.comm

                agg.llc_loads += perf_stats.llc_loads
                agg.llc_load_misses += perf_stats.llc_load_misses
                agg.llc_stores += perf_stats.llc_stores
                agg.llc_store_misses += perf_stats.llc_store_misses
                agg.dtlb_loads += perf_stats.dtlb_loads
                agg.dtlb_load_misses += perf_stats.dtlb_load_misses
                agg.dtlb_stores += perf_stats.dtlb_stores
                agg.dtlb_store_misses += perf_stats.dtlb_store_misses
                agg.dtlb_misses += perf_stats.dtlb_misses
                agg.itlb_load_misses += perf_stats.itlb_load_misses
                agg.cycles += perf_stats.cycles
                agg.instructions += perf_stats.instructions
                agg.last_seen_ns = max(agg.last_seen_ns, perf_stats.last_seen_ns)

        if raw_map is not None:
            self._prune_stale_entities(raw_map, current, now_ns)

        for entity_key, agg in current.items():
            prev = self._prev.get(entity_key)
            if prev is None:
                snap.add(agg)
                continue

            snap.add(
                PidStats(
                    pid=agg.pid,
                    tid=agg.tid,
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
                    itlb_load_misses=max(0, agg.itlb_load_misses - prev.itlb_load_misses),
                    cycles=max(0, agg.cycles - prev.cycles),
                    instructions=max(0, agg.instructions - prev.instructions),
                    minor_faults=max(0, agg.minor_faults - prev.minor_faults),
                    major_faults=max(0, agg.major_faults - prev.major_faults),
                    lbr_samples=max(0, agg.lbr_samples - prev.lbr_samples),
                    lbr_entries=max(0, agg.lbr_entries - prev.lbr_entries),
                    samples=max(0, agg.samples - prev.samples),
                )
            )

        self._prev = current
        if self._pending_events:
            snap.events.extend(self._pending_events)
            self._pending_events = []
        return snap

    def _start_child_monitor(self) -> None:
        """启动后台线程，轮询 /proc 发现根进程的子进程/线程，维护 child_pid_set。"""
        self._child_monitor_stop = threading.Event()
        self._child_monitor_thread = threading.Thread(
            target=self._child_monitor_loop,
            daemon=True,
            name="bcc-child-monitor",
        )
        self._child_monitor_thread.start()

    def _stop_child_monitor(self) -> None:
        if self._child_monitor_stop is not None:
            self._child_monitor_stop.set()
        if self._child_monitor_thread is not None:
            self._child_monitor_thread.join(timeout=2.0)
            self._child_monitor_thread = None
        self._child_monitor_stop = None

    def _child_monitor_loop(self) -> None:
        assert self._child_monitor_stop is not None
        while not self._child_monitor_stop.wait(timeout=0.5):
            try:
                self._refresh_child_pids()
            except Exception as exc:
                print(f"[child-monitor] 异常: {exc}", file=sys.stderr, flush=True)

    def _refresh_child_pids(self) -> None:
        """扫描 /proc 找出根进程的线程和直接子进程，同步到 child_pid_set BPF map。"""
        if self._bpf is None:
            return

        live: set[int] = set()

        # 1. 根进程的所有线程（/proc/<target_pid>/task/）
        task_dir = pathlib.Path(f"/proc/{self.target_pid}/task")
        try:
            for entry in task_dir.iterdir():
                if entry.name.isdigit():
                    live.add(int(entry.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass  # 根进程已退出

        # 2. 直接子进程（ppid == target_pid）及其线程
        for entry in pathlib.Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                status_text = (entry / "status").read_text()
            except (FileNotFoundError, PermissionError):
                continue
            try:
                ppid = next(
                    int(line.split()[1])
                    for line in status_text.splitlines()
                    if line.startswith("PPid:")
                )
            except StopIteration:
                continue
            if ppid != self.target_pid:
                continue
            child_pid = int(entry.name)
            live.add(child_pid)
            # 子进程自身的线程
            child_task = entry / "task"
            try:
                for t in child_task.iterdir():
                    if t.name.isdigit():
                        live.add(int(t.name))
            except (FileNotFoundError, PermissionError):
                pass

        # target_pid 本身由 target_pid_map 覆盖，child_pid_set 只存子代
        live.discard(self.target_pid)

        with self._child_monitor_lock:
            prev = self._tracked_child_pids
        new_pids  = live - prev
        gone_pids = prev - live

        child_map = self._bpf["child_pid_set"]
        for pid in new_pids:
            try:
                child_map[child_map.Key(pid)] = child_map.Leaf(1)
                print(f"[child-monitor] +PID {pid}", flush=True)
            except Exception as exc:
                print(f"[child-monitor] 添加 PID {pid} 失败: {exc}",
                      file=sys.stderr, flush=True)
        for pid in gone_pids:
            try:
                del child_map[child_map.Key(pid)]
            except Exception:
                pass

        with self._child_monitor_lock:
            self._tracked_child_pids = (prev | new_pids) - gone_pids

    def stop(self) -> None:
        """卸载 eBPF 程序并释放资源。"""
        self._stop_child_monitor()
        if self._perf_backend is not None:
            self._perf_backend.stop()
            self._perf_backend = None
        if self._bpf is not None:
            self._poll_events()
            self._bpf.cleanup()
            self._bpf = None
            print("[info] eBPF 程序已卸载", flush=True)
