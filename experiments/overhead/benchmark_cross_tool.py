#!/usr/bin/env python3
"""
benchmark_cross_tool.py — Cross-tool performance overhead and data volume benchmark.

Evaluates 5 operational modes on a fixed-workload memory benchmark:
  1. Baseline (Uninstrumented native execution)
  2. perf stat (Hardware PMU counter aggregation)
  3. perf record (Standard callstack-based PMU event sampling: -e cache-misses -g)
  4. eBPF WindowSnapshot (Routine monitoring mode: window_metrics.jsonl)
  5. eBPF Event Attribution (Diagnostic mode: window_metrics.jsonl + events.jsonl)

Outputs summary tables in CSV and Markdown formats.
"""

import os
import sys
import time
import re
import json
import statistics
import subprocess
import pathlib

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
BENCH_BIN = ROOT_DIR / "experiments/micro_benchmark/bench_mem"
OUTPUT_DIR = ROOT_DIR / "results/cross_tool_benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ROUNDS = 80
REPEAT = 5

def parse_bench_time(stdout_text: str) -> float:
    m = re.search(r"in ([\d\.]+) ms", stdout_text)
    if m:
        return float(m.group(1))
    return 0.0

def run_baseline() -> tuple[float, float, int]:
    t0 = time.perf_counter()
    res = subprocess.run([str(BENCH_BIN), "-r", str(ROUNDS)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    t1 = time.perf_counter()
    wall_ms = (t1 - t0) * 1000.0
    internal_ms = parse_bench_time(res.stdout)
    return internal_ms, wall_ms, 0

def run_perf_stat() -> tuple[float, float, int]:
    stat_file = OUTPUT_DIR / "_perf_stat_tmp.csv"
    t0 = time.perf_counter()
    res = subprocess.run(["perf", "stat", "-x,", "-o", str(stat_file),
                          "-e", "cycles,instructions,cache-misses,dTLB-load-misses,page-faults",
                          str(BENCH_BIN), "-r", str(ROUNDS)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    t1 = time.perf_counter()
    wall_ms = (t1 - t0) * 1000.0
    internal_ms = parse_bench_time(res.stdout)
    sz = stat_file.stat().st_size if stat_file.exists() else 0
    if stat_file.exists():
        stat_file.unlink()
    return internal_ms, wall_ms, sz

def run_perf_record() -> tuple[float, float, int]:
    data_file = OUTPUT_DIR / "_perf_tmp.data"
    t0 = time.perf_counter()
    res = subprocess.run(["perf", "record", "-e", "cache-misses", "-g", "-o", str(data_file),
                          str(BENCH_BIN), "-r", str(ROUNDS)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    t1 = time.perf_counter()
    wall_ms = (t1 - t0) * 1000.0
    internal_ms = parse_bench_time(res.stdout)
    sz = data_file.stat().st_size if data_file.exists() else 0
    if data_file.exists():
        data_file.unlink()
    return internal_ms, wall_ms, sz

def run_ebpf(emit_events: bool = False) -> tuple[float, float, int]:
    out_run = OUTPUT_DIR / f"_ebpf_tmp_{'diag' if emit_events else 'routine'}"
    if out_run.exists():
        import shutil
        shutil.rmtree(out_run)
    out_run.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    p_bench = subprocess.Popen([str(BENCH_BIN), "-r", str(ROUNDS)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    loader_cmd = [
        sys.executable, str(ROOT_DIR / "src/loader.py"),
        "--pid", str(p_bench.pid),
        "--window", "1.0",
        "--duration", "0",
        "--output", str(out_run),
    ]
    if emit_events:
        loader_cmd.extend(["--emit-events", "--pmu-backend", "bcc"])
    else:
        loader_cmd.extend(["--pmu-backend", "auto"])

    p_loader = subprocess.Popen(loader_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    stdout, _ = p_bench.communicate()
    t1 = time.perf_counter()
    wall_ms = (t1 - t0) * 1000.0
    internal_ms = parse_bench_time(stdout)

    p_loader.terminate()
    try:
        p_loader.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        p_loader.kill()

    total_bytes = sum(f.stat().st_size for f in out_run.glob("*.jsonl"))
    return internal_ms, wall_ms, total_bytes

def main():
    print(f"=== Cross-Tool Performance & Storage Overhead Benchmark ===")
    print(f"Target Workload: bench_mem (-r {ROUNDS})")
    print(f"Repetitions: {REPEAT} per configuration\n")

    configs = [
        ("Baseline (Native)", run_baseline),
        ("perf stat", run_perf_stat),
        ("perf record (-g)", run_perf_record),
        ("eBPF (WindowSnapshot)", lambda: run_ebpf(emit_events=False)),
        ("eBPF (Event Attribution)", lambda: run_ebpf(emit_events=True)),
    ]

    results = []
    base_mean = None

    for name, runner in configs:
        print(f"[*] Running {name} ({REPEAT} iterations)...")
        internals, walls, sizes = [], [], []
        # 1 warm-up
        runner()
        for i in range(1, REPEAT + 1):
            int_ms, wall_ms, sz = runner()
            internals.append(int_ms)
            walls.append(wall_ms)
            sizes.append(sz)
            print(f"    Run {i}: internal={int_ms:.2f}ms, wall={wall_ms:.2f}ms, data={sz/1024:.1f}KB")

        mean_int = statistics.mean(internals)
        std_int = statistics.stdev(internals) if len(internals) > 1 else 0.0
        mean_size = statistics.mean(sizes)

        if base_mean is None:
            base_mean = mean_int
            overhead_pct = 0.0
        else:
            overhead_pct = ((mean_int - base_mean) / base_mean) * 100.0

        results.append({
            "config": name,
            "mean_ms": mean_int,
            "std_ms": std_int,
            "overhead_pct": overhead_pct,
            "mean_data_bytes": mean_size,
            "mean_data_kb": mean_size / 1024.0,
        })
        print(f"  -> Mean: {mean_int:.2f} ± {std_int:.2f} ms (Overhead: {overhead_pct:+.2f}%), Storage: {mean_size/1024.0:.1f} KB\n")

    # Output CSV and Markdown
    csv_path = OUTPUT_DIR / "cross_tool_overhead.csv"
    md_path = OUTPUT_DIR / "cross_tool_overhead.md"

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("config,mean_ms,std_ms,overhead_pct,mean_data_bytes,mean_data_kb\n")
        for r in results:
            f.write(f'"{r["config"]}",{r["mean_ms"]:.2f},{r["std_ms"]:.2f},{r["overhead_pct"]:.2f},{r["mean_data_bytes"]:.0f},{r["mean_data_kb"]:.2f}\n')

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 主流性能剖析工具端到端开销横向对比表\n\n")
        f.write("| 剖析方案 / 工具配置 | 运行时耗时 (ms) | 标准差 (ms) | 相对延时开销 (%) | 数据落盘体积 (KB) | 存储膨胀倍数 (相对 eBPF 常态) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        ebpf_routine_kb = next((r["mean_data_kb"] for r in results if "WindowSnapshot" in r["config"]), 1.0)
        for r in results:
            expansion = r["mean_data_kb"] / ebpf_routine_kb if ebpf_routine_kb > 0 else 1.0
            exp_str = f"{expansion:.1f}x" if r["mean_data_bytes"] > 0 else "-"
            f.write(f'| **{r["config"]}** | {r["mean_ms"]:.2f} | ±{r["std_ms"]:.2f} | **{r["overhead_pct"]:+.2f}%** | {r["mean_data_kb"]:.2f} KB | {exp_str} |\n')

    print(f"[+] Benchmark completed successfully!")
    print(f"    Saved: {csv_path}")
    print(f"    Saved: {md_path}")

if __name__ == "__main__":
    main()
