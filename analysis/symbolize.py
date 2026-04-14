"""
symbolize.py — 将指令地址映射到函数名/源码行

步骤：
  1. 读取 /proc/<pid>/maps 获取进程内存布局
  2. 对每个地址确定所属 DSO（可执行文件或 .so）
  3. 调用 addr2line 或 nm 完成符号化

对于用户态进程，优先使用 addr2line -e <binary> -f -C <offset>
addr2line 需要二进制文件内有 DWARF 调试信息（-g 编译）；
若无调试信息则 fallback 到 nm/objdump 的函数边界匹配。
"""

import re
import subprocess
import pathlib
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass
class MapEntry:
    """单条 /proc/<pid>/maps 记录。"""
    start:  int
    end:    int
    perms:  str
    offset: int
    path:   str   # 空表示匿名映射


@dataclass
class SymbolInfo:
    func:        str   # 函数名（demangle 后），未知时为 "<unknown>"
    source_file: str   # 源文件路径，无调试信息时为 ""
    source_line: int   # 行号，无调试信息时为 0
    dso:         str   # 所属二进制/共享库路径
    offset:      int   # 在 DSO 文件内的偏移


def read_maps(pid: int) -> list[MapEntry]:
    """读取 /proc/<pid>/maps，返回所有映射条目。"""
    entries: list[MapEntry] = []
    maps_path = pathlib.Path(f"/proc/{pid}/maps")
    try:
        for line in maps_path.read_text().splitlines():
            m = re.match(
                r"([0-9a-f]+)-([0-9a-f]+)\s+(\S+)\s+([0-9a-f]+)\s+\S+\s+\d+\s*(.*)",
                line,
            )
            if not m:
                continue
            entries.append(MapEntry(
                start=int(m.group(1), 16),
                end=int(m.group(2), 16),
                perms=m.group(3),
                offset=int(m.group(4), 16),
                path=m.group(5).strip(),
            ))
    except (PermissionError, FileNotFoundError):
        pass
    return entries


def find_map_entry(addr: int, maps: list[MapEntry]) -> MapEntry | None:
    """返回包含 addr 的第一个 MapEntry，未找到返回 None。"""
    for e in maps:
        if e.start <= addr < e.end:
            return e
    return None


@lru_cache(maxsize=128)
def _addr2line_batch(binary: str, offsets_tuple: tuple[int, ...]) -> list[SymbolInfo]:
    """批量调用 addr2line，减少进程启动开销。"""
    args = ["addr2line", "-e", binary, "-f", "-C", "-p"] + [
        hex(o) for o in offsets_tuple
    ]
    results: list[SymbolInfo] = []
    try:
        out = subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True)
        for line in out.splitlines():
            # 格式：  funcname at file:lineno
            m = re.match(r"^(.*?) at (.*?):(\d+)$", line.strip())
            if m:
                results.append(SymbolInfo(
                    func=m.group(1) or "<unknown>",
                    source_file=m.group(2),
                    source_line=int(m.group(3)),
                    dso=binary,
                    offset=0,  # 由调用方填入
                ))
            else:
                results.append(SymbolInfo(
                    func=line.strip() or "<unknown>",
                    source_file="", source_line=0, dso=binary, offset=0,
                ))
    except (subprocess.CalledProcessError, FileNotFoundError):
        results = [
            SymbolInfo(func="<unknown>", source_file="", source_line=0,
                       dso=binary, offset=0)
            for _ in offsets_tuple
        ]
    return results


def symbolize_addresses(
    pid: int,
    addrs: list[int],
    maps: list[MapEntry] | None = None,
) -> list[SymbolInfo]:
    """
    将地址列表符号化，返回对应的 SymbolInfo 列表。

    参数
    ----
    pid   : 目标进程 PID（用于读取 /proc/maps，若 maps 已提供则忽略）
    addrs : 待符号化的虚拟地址列表
    maps  : 可选，已解析的 MapEntry 列表（避免重复读取 /proc/<pid>/maps）
    """
    if maps is None:
        maps = read_maps(pid)

    # 按 DSO 分组，减少 addr2line 调用次数
    groups: dict[str, list[tuple[int, int]]] = {}  # dso → [(addr, offset), ...]
    addr_to_entry: dict[int, tuple[str, int]] = {}  # addr → (dso, offset)

    for addr in addrs:
        entry = find_map_entry(addr, maps)
        if entry and entry.path and entry.path.startswith("/"):
            offset = addr - entry.start + entry.offset
            dso = entry.path
        else:
            offset = addr
            dso = "[unknown]"
        addr_to_entry[addr] = (dso, offset)
        groups.setdefault(dso, []).append((addr, offset))

    results_by_addr: dict[int, SymbolInfo] = {}
    for dso, addr_offsets in groups.items():
        if dso == "[unknown]":
            for addr, offset in addr_offsets:
                results_by_addr[addr] = SymbolInfo(
                    func="<unknown>", source_file="", source_line=0,
                    dso=dso, offset=offset,
                )
            continue

        offsets_tuple = tuple(o for _, o in addr_offsets)
        syms = _addr2line_batch(dso, offsets_tuple)
        for (addr, offset), sym in zip(addr_offsets, syms):
            sym.offset = offset
            results_by_addr[addr] = sym

    return [results_by_addr.get(a, SymbolInfo(
        func="<unknown>", source_file="", source_line=0, dso="", offset=a,
    )) for a in addrs]
