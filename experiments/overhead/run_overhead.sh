#!/usr/bin/env bash
# experiments/overhead/run_overhead.sh
#
# P3 方法学验证：采集开销测试
#
# 目标：量化 eBPF 程序对目标进程的性能影响，包括：
#   1. CPU overhead（使用 perf stat 比较有/无 eBPF 的 CPU cycles）
#   2. Latency impact（使用 benchmark 程序测量完成时间差异）
#   3. Measurement overhead（eBPF 自身消耗的 CPU 时间）
#
# 使用方法：
#   sudo bash experiments/overhead/run_overhead.sh
#
# 依赖：perf, python3, stress-ng（可选）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$ROOT_DIR/results/overhead_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

REPEAT=5           # 每组重复次数
DURATION=10        # 每次采集持续时间（秒）
WINDOW=1.0         # 时间窗大小（秒）

# 使用 stress-ng 作为受控负载，若不存在则 fallback 到 dd
if command -v stress-ng &>/dev/null; then
    LOAD_CMD="stress-ng --vm 1 --vm-bytes 256M --timeout ${DURATION}s --quiet"
    LOAD_NAME="stress-ng"
else
    LOAD_CMD="dd if=/dev/urandom of=/dev/null bs=1M count=$((DURATION * 50))"
    LOAD_NAME="dd"
fi

echo "=== 采集开销测试 ==="
echo "负载工具: $LOAD_NAME   重复次数: $REPEAT   单次时长: ${DURATION}s"
echo "结果目录: $RESULTS_DIR"
echo ""

# ---- 基准组（无 eBPF）----
echo "[1/3] 运行基准组（无 eBPF 采集器）..."
for i in $(seq 1 $REPEAT); do
    {
        START=$(date +%s%N)
        $LOAD_CMD 2>/dev/null || true
        END=$(date +%s%N)
        echo "baseline,$i,$(( (END - START) / 1000000 ))"
    } >> "$RESULTS_DIR/timing.csv"
    echo "  baseline run $i/$REPEAT done"
done

# ---- 采集组（有 eBPF）----
echo "[2/3] 运行采集组（eBPF 开启）..."
for i in $(seq 1 $REPEAT); do
    # 后台启动负载，取其 PID
    $LOAD_CMD 2>/dev/null &
    LOAD_PID=$!

    # 同时启动采集器
    START=$(date +%s%N)
    python3 "$ROOT_DIR/src/loader.py" \
        --pid "$LOAD_PID" \
        --window "$WINDOW" \
        --duration "$DURATION" \
        --output "$RESULTS_DIR/run_${i}/" \
        2>/dev/null || true

    wait "$LOAD_PID" 2>/dev/null || true
    END=$(date +%s%N)
    echo "ebpf,$i,$(( (END - START) / 1000000 ))" >> "$RESULTS_DIR/timing.csv"
    echo "  ebpf run $i/$REPEAT done"
done

# ---- perf stat 对比（单次）----
echo "[3/3] perf stat 对比..."
if command -v perf &>/dev/null; then
    echo "--- baseline ---" > "$RESULTS_DIR/perf_stat.txt"
    perf stat -e cycles,instructions,cache-misses \
        -- bash -c "$LOAD_CMD 2>/dev/null" 2>> "$RESULTS_DIR/perf_stat.txt" || true
else
    echo "perf 不可用，跳过 perf stat"
fi

# ---- 汇总 ----
echo ""
echo "=== 结果摘要 ==="
echo "timing.csv 内容："
echo "组别,次数,时间(ms)"
cat "$RESULTS_DIR/timing.csv"
echo ""
echo "请运行以下命令生成分析图表："
echo "  python3 $ROOT_DIR/analysis/report.py --results $RESULTS_DIR --output $RESULTS_DIR/figures/"
