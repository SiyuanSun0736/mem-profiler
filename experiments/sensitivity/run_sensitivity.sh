#!/usr/bin/env bash
# experiments/sensitivity/run_sensitivity.sh
#
# P3 方法学验证：参数敏感性测试
#
# 目标：评估关键参数变化对测量结果的影响：
#   1. sample_rate：不同采样率（50 / 100 / 500 / 1000）下的指标差异
#   2. window_sec：不同时间窗（0.5 / 1.0 / 2.0 / 5.0s）下的聚合结果
#   3. 指标覆盖度：只开 LLC vs 只开 fault vs 全开
#
# 使用方法：
#   sudo bash experiments/sensitivity/run_sensitivity.sh --pid <PID>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

TARGET_PID=0
TARGET_COMM=""
DURATION=10

while [[ $# -gt 0 ]]; do
    case $1 in
        --pid)  TARGET_PID="$2"; shift 2 ;;
        --comm) TARGET_COMM="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [[ $TARGET_PID -eq 0 && -z $TARGET_COMM ]]; then
    echo "用法: $0 --pid <PID> | --comm <进程名>"
    exit 1
fi

RUN_TS=$(date +%Y%m%d_%H%M%S)
RESULTS_BASE="$ROOT_DIR/results/sensitivity_${RUN_TS}"
mkdir -p "$RESULTS_BASE"

TARGET_ARG=""
[[ $TARGET_PID -ne 0 ]] && TARGET_ARG="--pid $TARGET_PID" || TARGET_ARG="--comm $TARGET_COMM"

run_one() {
    local label="$1"; shift
    local out_dir="$RESULTS_BASE/$label"
    echo "[run] $label"
    python3 "$ROOT_DIR/src/loader.py" \
        $TARGET_ARG --duration "$DURATION" --output "$out_dir/" "$@" 2>/dev/null || true
}

echo "=== 参数敏感性测试 ==="
echo "结果目录: $RESULTS_BASE"

# 1. 不同采样率
echo "--- 采样率扫描 ---"
for rate in 50 100 500 1000; do
    run_one "samplerate_${rate}" --window 1.0 --sample-rate "$rate"
done

# 2. 不同时间窗
echo "--- 时间窗扫描 ---"
for win in 0.5 1.0 2.0 5.0; do
    run_one "window_${win}s" --window "$win" --sample-rate 100
done

# 3. 探针组合
echo "--- 探针组合 ---"
run_one "probe_llc_only"   --window 1.0 --no-dtlb  --no-fault
run_one "probe_fault_only" --window 1.0 --no-llc   --no-dtlb
run_one "probe_all"        --window 1.0

echo ""
echo "=== 敏感性测试完成 ==="
echo "请对比 $RESULTS_BASE/ 下各子目录的 window_metrics.jsonl，分析参数影响。"
