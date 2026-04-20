"""
perf_counter.py -- perf_event_open-based PMU counting backend.

Opens per-thread counting fds, refreshes the tracked thread set from /proc,
and scales raw counts with time_enabled/time_running.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import os
import pathlib
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from bcc.perf import Perf

_PERF_COUNT_HW_CACHE_LL = 2
_PERF_COUNT_HW_CACHE_DTLB = 3
_PERF_COUNT_HW_CACHE_ITLB = 4
_PERF_COUNT_HW_CACHE_OP_READ = 0
_PERF_COUNT_HW_CACHE_OP_WRITE = 1
_PERF_COUNT_HW_CACHE_RESULT_ACCESS = 0
_PERF_COUNT_HW_CACHE_RESULT_MISS = 1
_PERF_COUNT_HW_CPU_CYCLES = 0
_PERF_COUNT_HW_INSTRUCTIONS = 1
_PERF_COUNT_HW_CACHE_REFERENCES = 4
_PERF_COUNT_HW_CACHE_MISSES = 5

_PERF_FORMAT_TOTAL_TIME_ENABLED = 1 << 0
_PERF_FORMAT_TOTAL_TIME_RUNNING = 1 << 1
_PERF_ATTR_FLAG_DISABLED = 1 << 0

_PERF_EVENT_IOC_ENABLE = 0x2400
_PERF_EVENT_IOC_DISABLE = 0x2401
_PERF_EVENT_IOC_RESET = 0x2403

_MONITOR_INTERVAL_SEC = 0.5
_SOFT_OPEN_ERRNOS = {
    errno.EBUSY,
    errno.EINVAL,
    errno.ENOENT,
    errno.ENODEV,
    errno.ENOSYS,
    errno.EOPNOTSUPP,
    errno.ESRCH,
}
_UNSUPPORTED_EVENT_ERRNOS = {
    errno.EINVAL,
    errno.ENODEV,
    errno.ENOSYS,
    errno.EOPNOTSUPP,
}


def _cache_config(cache: int, op: int, result: int) -> int:
    return cache | (op << 8) | (result << 16)


def _perf_event_open_nr() -> int:
    machine = os.uname().machine.lower()
    numbers = {
        "x86_64": 298,
        "amd64": 298,
        "aarch64": 241,
        "arm64": 241,
        "armv7l": 364,
        "i386": 336,
        "i686": 336,
        "riscv64": 241,
    }
    if machine not in numbers:
        raise RuntimeError(f"unsupported perf_event_open architecture: {machine}")
    return numbers[machine]


_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.syscall.restype = ctypes.c_long
_SYS_PERF_EVENT_OPEN = _perf_event_open_nr()


def _open_perf_event(attr: ctypes.Structure, pid: int) -> int:
    attr.size = ctypes.sizeof(attr)
    fd = _LIBC.syscall(
        _SYS_PERF_EVENT_OPEN,
        ctypes.byref(attr),
        pid,
        -1,
        -1,
        0,
    )
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return int(fd)


def _decode_comm(raw: str) -> str:
    return raw[:15]


def _read_comm(tid: int) -> str:
    try:
        return _decode_comm(pathlib.Path(f"/proc/{tid}/comm").read_text().strip())
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return ""


def _read_status_field(pid: int, field: str) -> Optional[int]:
    try:
        status_text = pathlib.Path(f"/proc/{pid}/status").read_text()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    prefix = f"{field}:"
    for line in status_text.splitlines():
        if line.startswith(prefix):
            try:
                return int(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


@dataclass
class PerfCounterSnapshot:
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
    last_seen_ns: int = 0


@dataclass(frozen=True)
class _ThreadTarget:
    pid: int
    tid: int
    comm: str


@dataclass
class _ThreadHandle:
    pid: int
    tid: int
    comm: str
    fds: dict[str, int] = field(default_factory=dict)

    def close(self) -> None:
        for fd in self.fds.values():
            try:
                fcntl.ioctl(fd, _PERF_EVENT_IOC_DISABLE, 0)
            except OSError:
                pass
            try:
                os.close(fd)
            except OSError:
                pass
        self.fds.clear()


class PerfCounterBackend:
    """Periodic perf_event_open reader for PMU counters."""

    def __init__(
        self,
        target_pid: int,
        target_tid: int,
        target_comm: str,
        per_tid: bool,
        track_children: bool,
        enable_llc: bool,
        enable_dtlb: bool,
        enable_itlb: bool,
    ) -> None:
        self.target_pid = target_pid
        self.target_tid = target_tid
        self.target_comm = target_comm[:15]
        self.per_tid = per_tid
        self.track_children = track_children and (target_pid > 0)
        self.enable_llc = enable_llc
        self.enable_dtlb = enable_dtlb
        self.enable_itlb = enable_itlb

        self._lock = threading.Lock()
        self._handles: dict[int, _ThreadHandle] = {}
        self._warned_open_failures: set[str] = set()
        self._llc_store_via_generic = False
        self._monitor_stop: Optional[threading.Event] = None
        self._monitor_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._detect_llc_store_proxy()
        discovered = self._refresh_entities()
        with self._lock:
            has_handles = bool(self._handles)
        if discovered and not has_handles:
            raise RuntimeError("discovered targets but failed to open any PMU counters")
        self._monitor_stop = threading.Event()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="perf-counter-monitor",
        )
        self._monitor_thread.start()

    def stop(self) -> None:
        if self._monitor_stop is not None:
            self._monitor_stop.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
        self._monitor_stop = None

        with self._lock:
            handles = list(self._handles.values())
            self._handles.clear()
        for handle in handles:
            handle.close()

    def uses_llc_store_proxy(self) -> bool:
        return self._llc_store_via_generic

    def read(self) -> dict[tuple[int, int], PerfCounterSnapshot]:
        self._refresh_entities()
        now_ns = time.monotonic_ns()

        with self._lock:
            handles = list(self._handles.values())
            broken_tids: list[int] = []
            aggregated: dict[tuple[int, int], PerfCounterSnapshot] = {}

            for handle in handles:
                try:
                    counts = self._read_thread_counts(handle)
                except OSError as exc:
                    broken_tids.append(handle.tid)
                    if exc.errno not in (errno.EBADF, errno.ENOENT, errno.ESRCH):
                        print(
                            f"[perf] 读取 TID {handle.tid} 计数失败: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                    continue

                entity_key = (handle.pid, handle.tid) if self.per_tid else (handle.pid, 0)
                snap = aggregated.get(entity_key)
                if snap is None:
                    snap = PerfCounterSnapshot(
                        pid=handle.pid,
                        tid=handle.tid if self.per_tid else 0,
                        comm=handle.comm,
                        last_seen_ns=now_ns,
                    )
                    aggregated[entity_key] = snap
                elif handle.comm and not snap.comm:
                    snap.comm = handle.comm

                snap.llc_loads += counts.get("llc_loads", 0)
                snap.llc_load_misses += counts.get("llc_load_misses", 0)
                snap.llc_stores += counts.get("llc_stores", 0)
                snap.llc_store_misses += counts.get("llc_store_misses", 0)
                snap.dtlb_loads += counts.get("dtlb_loads", 0)
                snap.dtlb_load_misses += counts.get("dtlb_load_misses", 0)
                snap.dtlb_stores += counts.get("dtlb_stores", 0)
                snap.dtlb_store_misses += counts.get("dtlb_store_misses", 0)
                snap.itlb_load_misses += counts.get("itlb_load_misses", 0)
                snap.cycles += counts.get("cycles", 0)
                snap.instructions += counts.get("instructions", 0)
                snap.last_seen_ns = now_ns

            if broken_tids:
                for tid in broken_tids:
                    handle = self._handles.pop(tid, None)
                    if handle is not None:
                        handle.close()

        for snap in aggregated.values():
            snap.dtlb_misses = snap.dtlb_load_misses + snap.dtlb_store_misses
        return aggregated

    def _monitor_loop(self) -> None:
        assert self._monitor_stop is not None
        while not self._monitor_stop.wait(timeout=_MONITOR_INTERVAL_SEC):
            try:
                self._refresh_entities()
            except Exception as exc:
                print(f"[perf] monitor 异常: {exc}", file=sys.stderr, flush=True)

    def _detect_llc_store_proxy(self) -> None:
        if not self.enable_llc:
            return
        native_specs = [
            (Perf.PERF_TYPE_HW_CACHE, _cache_config(_PERF_COUNT_HW_CACHE_LL, _PERF_COUNT_HW_CACHE_OP_WRITE, _PERF_COUNT_HW_CACHE_RESULT_ACCESS)),
            (Perf.PERF_TYPE_HW_CACHE, _cache_config(_PERF_COUNT_HW_CACHE_LL, _PERF_COUNT_HW_CACHE_OP_WRITE, _PERF_COUNT_HW_CACHE_RESULT_MISS)),
        ]
        for perf_type, config in native_specs:
            try:
                fd = self._open_counter_fd(perf_type, config, 0)
            except OSError as exc:
                if exc.errno in _UNSUPPORTED_EVENT_ERRNOS:
                    self._llc_store_via_generic = True
                    return
                raise
            else:
                os.close(fd)

    def _refresh_entities(self) -> int:
        live_targets = self._discover_targets()
        live_tids = set(live_targets)

        with self._lock:
            gone_tids = set(self._handles) - live_tids
            new_targets = [live_targets[tid] for tid in live_tids - set(self._handles)]

            for tid in live_tids & set(self._handles):
                handle = self._handles[tid]
                target = live_targets[tid]
                handle.comm = target.comm or handle.comm
                handle.pid = target.pid

            for tid in gone_tids:
                handle = self._handles.pop(tid, None)
                if handle is not None:
                    handle.close()

            for target in new_targets:
                handle = self._open_thread_handle(target)
                if handle is not None:
                    self._handles[target.tid] = handle
        return len(live_targets)

    def _discover_targets(self) -> dict[int, _ThreadTarget]:
        if self.target_tid > 0:
            return self._discover_single_tid(self.target_tid)
        if self.target_comm:
            return self._discover_comm_threads(self.target_comm)
        if self.target_pid > 0:
            return self._discover_pid_threads(self.target_pid)
        return {}

    def _discover_single_tid(self, tid: int) -> dict[int, _ThreadTarget]:
        tgid = _read_status_field(tid, "Tgid")
        if tgid is None:
            return {}
        return {tid: _ThreadTarget(pid=tgid, tid=tid, comm=_read_comm(tid))}

    def _discover_comm_threads(self, comm: str) -> dict[int, _ThreadTarget]:
        targets: dict[int, _ThreadTarget] = {}
        proc_root = pathlib.Path("/proc")
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                current_comm = _decode_comm((entry / "comm").read_text().strip())
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            if current_comm != comm:
                continue
            targets.update(self._discover_process_threads(int(entry.name)))
        return targets

    def _discover_pid_threads(self, root_pid: int) -> dict[int, _ThreadTarget]:
        targets = self._discover_process_threads(root_pid)
        if not self.track_children:
            return targets

        proc_root = pathlib.Path("/proc")
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            child_pid = int(entry.name)
            if child_pid == root_pid:
                continue
            ppid = _read_status_field(child_pid, "PPid")
            if ppid != root_pid:
                continue
            targets.update(self._discover_process_threads(child_pid))
        return targets

    def _discover_process_threads(self, pid: int) -> dict[int, _ThreadTarget]:
        targets: dict[int, _ThreadTarget] = {}
        task_dir = pathlib.Path(f"/proc/{pid}/task")
        try:
            entries = list(task_dir.iterdir())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            return targets
        for entry in entries:
            if not entry.name.isdigit():
                continue
            tid = int(entry.name)
            targets[tid] = _ThreadTarget(pid=pid, tid=tid, comm=_read_comm(tid))
        return targets

    def _open_thread_handle(self, target: _ThreadTarget) -> Optional[_ThreadHandle]:
        handle = _ThreadHandle(pid=target.pid, tid=target.tid, comm=target.comm)
        metric_specs = self._metric_specs()
        for metric, (perf_type, config) in metric_specs.items():
            try:
                handle.fds[metric] = self._open_counter_fd(perf_type, config, target.tid)
            except OSError as exc:
                if metric not in self._warned_open_failures:
                    print(
                        f"[perf] {metric} 打开失败，将跳过该计数器: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    self._warned_open_failures.add(metric)
                if exc.errno not in _SOFT_OPEN_ERRNOS:
                    handle.close()
                    raise
        if not handle.fds:
            return None
        return handle

    def _metric_specs(self) -> dict[str, tuple[int, int]]:
        specs = {
            "cycles": (Perf.PERF_TYPE_HARDWARE, _PERF_COUNT_HW_CPU_CYCLES),
            "instructions": (Perf.PERF_TYPE_HARDWARE, _PERF_COUNT_HW_INSTRUCTIONS),
        }
        if self.enable_llc:
            specs["llc_loads"] = (
                Perf.PERF_TYPE_HW_CACHE,
                _cache_config(_PERF_COUNT_HW_CACHE_LL, _PERF_COUNT_HW_CACHE_OP_READ, _PERF_COUNT_HW_CACHE_RESULT_ACCESS),
            )
            specs["llc_load_misses"] = (
                Perf.PERF_TYPE_HW_CACHE,
                _cache_config(_PERF_COUNT_HW_CACHE_LL, _PERF_COUNT_HW_CACHE_OP_READ, _PERF_COUNT_HW_CACHE_RESULT_MISS),
            )
            if self._llc_store_via_generic:
                specs["llc_stores"] = (Perf.PERF_TYPE_HARDWARE, _PERF_COUNT_HW_CACHE_REFERENCES)
                specs["llc_store_misses"] = (Perf.PERF_TYPE_HARDWARE, _PERF_COUNT_HW_CACHE_MISSES)
            else:
                specs["llc_stores"] = (
                    Perf.PERF_TYPE_HW_CACHE,
                    _cache_config(_PERF_COUNT_HW_CACHE_LL, _PERF_COUNT_HW_CACHE_OP_WRITE, _PERF_COUNT_HW_CACHE_RESULT_ACCESS),
                )
                specs["llc_store_misses"] = (
                    Perf.PERF_TYPE_HW_CACHE,
                    _cache_config(_PERF_COUNT_HW_CACHE_LL, _PERF_COUNT_HW_CACHE_OP_WRITE, _PERF_COUNT_HW_CACHE_RESULT_MISS),
                )
        if self.enable_dtlb:
            specs["dtlb_loads"] = (
                Perf.PERF_TYPE_HW_CACHE,
                _cache_config(_PERF_COUNT_HW_CACHE_DTLB, _PERF_COUNT_HW_CACHE_OP_READ, _PERF_COUNT_HW_CACHE_RESULT_ACCESS),
            )
            specs["dtlb_load_misses"] = (
                Perf.PERF_TYPE_HW_CACHE,
                _cache_config(_PERF_COUNT_HW_CACHE_DTLB, _PERF_COUNT_HW_CACHE_OP_READ, _PERF_COUNT_HW_CACHE_RESULT_MISS),
            )
            specs["dtlb_stores"] = (
                Perf.PERF_TYPE_HW_CACHE,
                _cache_config(_PERF_COUNT_HW_CACHE_DTLB, _PERF_COUNT_HW_CACHE_OP_WRITE, _PERF_COUNT_HW_CACHE_RESULT_ACCESS),
            )
            specs["dtlb_store_misses"] = (
                Perf.PERF_TYPE_HW_CACHE,
                _cache_config(_PERF_COUNT_HW_CACHE_DTLB, _PERF_COUNT_HW_CACHE_OP_WRITE, _PERF_COUNT_HW_CACHE_RESULT_MISS),
            )
        if self.enable_itlb:
            specs["itlb_load_misses"] = (
                Perf.PERF_TYPE_HW_CACHE,
                _cache_config(_PERF_COUNT_HW_CACHE_ITLB, _PERF_COUNT_HW_CACHE_OP_READ, _PERF_COUNT_HW_CACHE_RESULT_MISS),
            )
        return specs

    def _open_counter_fd(self, perf_type: int, config: int, pid: int) -> int:
        attr = Perf.perf_event_attr()
        attr.type = perf_type
        attr.config = config
        attr.sample_type = 0
        attr.read_format = _PERF_FORMAT_TOTAL_TIME_ENABLED | _PERF_FORMAT_TOTAL_TIME_RUNNING
        fd = _open_perf_event(attr, pid)
        fcntl.ioctl(fd, _PERF_EVENT_IOC_RESET, 0)
        fcntl.ioctl(fd, _PERF_EVENT_IOC_ENABLE, 0)
        return fd

    def _read_thread_counts(self, handle: _ThreadHandle) -> dict[str, int]:
        counts: dict[str, int] = {}
        for metric, fd in handle.fds.items():
            data = os.read(fd, 24)
            if len(data) != 24:
                raise OSError(errno.EIO, f"short perf read for {metric}")
            raw_value, time_enabled, time_running = struct.unpack("QQQ", data)
            if time_running == 0 or time_enabled == 0:
                counts[metric] = 0
                continue
            if time_running >= time_enabled:
                counts[metric] = int(raw_value)
                continue
            counts[metric] = int((raw_value * time_enabled + (time_running // 2)) // time_running)
        return counts