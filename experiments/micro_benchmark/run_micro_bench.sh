#!/usr/bin/env bash
# experiments/micro_benchmark/run_micro_bench.sh
#
# P3 方法学验证：微基准校验
#
# 目标：用已知访存行为的微基准程序校验 eBPF 采集结果的"方向正确性"：
#   1. 高 LLC miss 场景（随机大数组访问）→ LLC miss 计数应显著高于基准
#   2. 高 TLB miss 场景（大步长访问）    → dTLB miss 计数应高于基准
#   3. 高 page fault 场景（mmap 大块内存）→ minor fault 计数应高于基准
#
# 微基准程序用 Python（无需额外编译），当然换成 C 语言版效果更准确。
#
# 使用方法：
#   sudo bash experiments/micro_benchmark/run_micro_bench.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_TS=$(date +%Y%m%d_%H%M%S)
RESULTS_BASE="$ROOT_DIR/results/micro_bench_${RUN_TS}"
mkdir -p "$RESULTS_BASE"

DURATION=8
WINDOW=1.0

echo "=== 微基准校验 ==="
echo "结果目录: $RESULTS_BASE"

# ---- 微基准 Python 脚本（内联生成）----
MB_SCRIPT="$RESULTS_BASE/_bench.py"
cat > "$MB_SCRIPT" << 'PYEOF'
"""内联微基准：根据 BENCH_TYPE 环境变量选择访存模式。"""
import os, random, sys, time, mmap, array

bench = os.environ.get("BENCH_TYPE", "random")
SIZE  = int(os.environ.get("BENCH_SIZE", 64 * 1024 * 1024))  # 64 MB
ITERS = int(os.environ.get("BENCH_ITERS", 5_000_000))

if bench == "random":
    # 随机访问大数组 → 高 LLC miss
    buf = bytearray(SIZE)
    indices = [random.randrange(SIZE) for _ in range(ITERS)]
    s = 0
    for i in indices:
        s += buf[i]
    print(f"random_access checksum={s}")

elif bench == "stride":
    # 大步长访问 → 高 dTLB miss（4KB 步长跨页）
    PAGE = 4096
    buf = bytearray(SIZE)
    s, i = 0, 0
    for _ in range(ITERS):
        s += buf[i % SIZE]
        i += PAGE
    print(f"stride_access checksum={s}")

elif bench == "page_fault":
    # 逐页写新 mmap → 高 minor fault
    m = mmap.mmap(-1, SIZE)
    for off in range(0, SIZE, 4096):
        m[off] = 0x42
    m.close()
    print("page_fault_access done")

else:
    # 基准（顺序访问，低 miss 率）
    buf = bytearray(SIZE)
    s = sum(buf[i] for i in range(0, SIZE, 64))
    print(f"sequential checksum={s}")
PYEOF

run_bench() {
    local label="$1"
    local bench_type="$2"
    local out_dir="$RESULTS_BASE/$label"
    echo "[bench] $label  (BENCH_TYPE=$bench_type)"

    # 后台启动微基准，重复运行直到采集器结束
    (
        while true; do
            BENCH_TYPE="$bench_type" python3 "$MB_SCRIPT" &>/dev/null || true
        done
    ) &
    BENCH_PID=$!

    python3 "$ROOT_DIR/src/loader.py" \
        --pid "$BENCH_PID" \
        --window "$WINDOW" \
        --duration "$DURATION" \
        --output "$out_dir/" \
        2>/dev/null || true

    kill "$BENCH_PID" 2>/dev/null || true
    wait "$BENCH_PID" 2>/dev/null || true

    # 简单输出各指标总和
    python3 - <<PYEOF2
import json, pathlib
f = pathlib.Path("$out_dir/window_metrics.jsonl")
if f.exists():
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    for m in ["llc_load_misses","dtlb_misses","minor_faults","samples"]:
        total = sum(r.get(m,0) for r in rows)
        print(f"  {m:<22} = {total}")
PYEOF2
    echo ""
}

run_bench "baseline_sequential" "sequential"
run_bench "high_llc_miss"       "random"
run_bench "high_dtlb_miss"      "stride"
run_bench "high_page_fault"     "page_fault"

echo "=== 微基准校验完成 ==="
echo "预期：high_llc_miss > baseline (llc_load_misses)"
echo "预期：high_dtlb_miss > baseline (dtlb_misses)"
echo "预期：high_page_fault > baseline (minor_faults)"
