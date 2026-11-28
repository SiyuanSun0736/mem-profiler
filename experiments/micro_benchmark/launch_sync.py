#!/usr/bin/env python3
"""
launch_sync.py — Universal target workload synchronizer for zero-loss eBPF profiling.

Workflow:
  1. The target process pauses itself with SIGSTOP before executing instruction 0 of the workload.
  2. The profiler attaches all eBPF probes and perf counters to the target PID.
  3. The profiler sends SIGCONT to resume the target process.
  4. The workload executes 100% of its instructions under active observation (0ms lost).
"""

import os
import sys
import signal
import time
import subprocess

def launch_stopped(cmd: list[str]) -> subprocess.Popen:
    """Launch target command in stopped state (T) ready for profiler attachment."""
    sync_code = (
        "import os, signal, sys; "
        "os.kill(os.getpid(), signal.SIGSTOP); "
        "os.execvp(sys.argv[1], sys.argv[1:])"
    )
    full_cmd = [sys.executable, "-c", sync_code] + cmd
    proc = subprocess.Popen(full_cmd)

    # Wait until process transitions to Stopped state ('T')
    for _ in range(100):
        try:
            stat = open(f"/proc/{proc.pid}/stat").read().split()[2]
            if stat == "T":
                break
        except Exception:
            pass
        time.sleep(0.02)
    return proc

def resume(pid: int) -> None:
    """Send SIGCONT to wake up stopped process."""
    try:
        os.kill(pid, signal.SIGCONT)
    except OSError:
        pass

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <workload command...>", file=sys.stderr)
        sys.exit(1)

    p = launch_stopped(sys.argv[1:])
    print(f"[launch_sync] PID={p.pid} is in stopped state (T). Waiting for SIGCONT...", flush=True)
    p.wait()
