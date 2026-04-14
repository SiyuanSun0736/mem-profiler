"""
filter.py — PID/comm 过滤辅助工具

通过扫描 /proc 文件系统按进程名解析 PID。
"""

import os
import pathlib


def resolve_pid_by_comm(comm: str) -> int:
    """
    在 /proc 中查找第一个 comm 匹配的进程，返回其 PID；未找到返回 0。

    参数
    ----
    comm : 进程名，对应 /proc/<pid>/comm（内核截断至 15 字符）

    返回
    ----
    int : 第一个匹配的 PID，未找到时返回 0
    """
    target = comm[:15]  # 内核 TASK_COMM_LEN = 16，含 \0
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        comm_file = entry / "comm"
        try:
            if comm_file.read_text().strip() == target:
                return int(entry.name)
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
    return 0


def list_pids_by_comm(comm: str) -> list[int]:
    """
    返回所有 comm 匹配的 PID 列表（comm 可以是进程族，例如 worker 进程）。
    """
    target = comm[:15]
    pids: list[int] = []
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        comm_file = entry / "comm"
        try:
            if comm_file.read_text().strip() == target:
                pids.append(int(entry.name))
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
    return sorted(pids)
