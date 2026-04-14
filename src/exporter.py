"""
exporter.py — 将采集结果写入 JSONL 文件

输出三类文件（见 docs/data_protocol.md）：
  • run_metadata.jsonl     — 每次运行一条元信息记录
  • window_metrics.jsonl   — 每个时间窗每个 PID 一条记录
  • events.jsonl           — 逐事件记录（仅在 emit_events=True 时生成，暂留接口）

所有文件使用 JSON Lines 格式（每行一个 JSON 对象），便于 pandas/jq 处理。
"""

import json
import pathlib
import platform
import socket
import uuid
from datetime import datetime, timezone
from typing import Optional

from collector import WindowSnapshot


class Exporter:
    """
    参数
    ----
    out_dir     : 输出目录（已由调用方创建）
    target_pid  : 采集目标 PID（0 = 全部进程）
    target_comm : 采集目标进程名
    window_sec  : 时间窗大小（秒）
    sample_rate : perf 采样率
    enable_*    : 各类探针是否启用
    """

    SCHEMA_VERSION = "1.0"

    def __init__(
        self,
        out_dir:      pathlib.Path,
        target_pid:   int   = 0,
        target_comm:  str   = "",
        window_sec:   float = 1.0,
        sample_rate:  int   = 100,
        enable_llc:   bool  = True,
        enable_dtlb:  bool  = True,
        enable_fault: bool  = True,
        observations: Optional[list[dict]] = None,
        collection_backend: str = "bcc",
    ) -> None:
        self._out   = out_dir
        self._run_id = str(uuid.uuid4())
        self._start_iso = datetime.now(timezone.utc).isoformat()

        # 打开输出文件
        self._meta_f   = open(out_dir / "run_metadata.jsonl",   "a", encoding="utf-8")
        self._window_f = open(out_dir / "window_metrics.jsonl", "a", encoding="utf-8")

        # 写入本次运行的元信息
        meta = {
            "schema_version": self.SCHEMA_VERSION,
            "run_id":         self._run_id,
            "start_ts_iso":   self._start_iso,
            "end_ts_iso":     None,
            "target_pid":     target_pid,
            "target_comm":    target_comm,
            "window_sec":     window_sec,
            "sample_rate":    sample_rate,
            "enabled_probes": {
                "llc":   enable_llc,
                "dtlb":  enable_dtlb,
                "fault": enable_fault,
            },
            "collection_backend": collection_backend,
            "observations": observations or [],
            "host_info": {
                "hostname":       socket.gethostname(),
                "kernel_version": platform.release(),
                "cpu_model":      _cpu_model(),
                "num_cpus":       _num_cpus(),
            },
        }
        self._meta_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        self._meta_f.flush()

    # ------------------------------------------------------------------

    def write_window(self, snap: WindowSnapshot) -> None:
        """将一个时间窗的所有 PID 记录追加写入 window_metrics.jsonl。"""
        for entry in snap.entries:
            row = {"schema_version": self.SCHEMA_VERSION, "run_id": self._run_id}
            row.update(entry)
            self._window_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._window_f.flush()

    # ------------------------------------------------------------------

    def flush_and_close(self) -> None:
        """更新 run_metadata 的 end_ts_iso，关闭所有文件。"""
        end_iso = datetime.now(timezone.utc).isoformat()

        # 追加一条 end 记录（简化处理；实际可用 patch-in-place）
        end_rec = {
            "schema_version": self.SCHEMA_VERSION,
            "run_id":         self._run_id,
            "end_ts_iso":     end_iso,
            "_record_type":   "run_end",
        }
        self._meta_f.write(json.dumps(end_rec, ensure_ascii=False) + "\n")

        self._meta_f.close()
        self._window_f.close()


# ---- 辅助函数 ----

def _cpu_model() -> str:
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def _num_cpus() -> int:
    try:
        return len([
            l for l in pathlib.Path("/proc/cpuinfo").read_text().splitlines()
            if l.startswith("processor")
        ])
    except OSError:
        return 0
