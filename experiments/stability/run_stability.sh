#!/usr/bin/env bash
# experiments/stability/run_stability.sh
#
# P3 方法学验证：重复运行稳定性测试
#
# 目标：评估在相同条件下重复采集同一程序的结果波动程度，包括：
#   1. 各指标（LLC misses / page faults 等）的均值和标准差
#   2. 变异系数（CV = std / mean），CV < 10% 视为稳定
#   3. 不同重复次数下的收敛性曲线
#
# 使用方法：
#   sudo bash experiments/stability/run_stability.sh --pid <PID>
#   sudo bash experiments/stability/run_stability.sh --comm <name> --repeat 10

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 默认参数
TARGET_PID=0
TARGET_COMM=""
REPEAT=10
DURATION=10
WINDOW=1.0

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --pid)    TARGET_PID="$2";    shift 2 ;;
        --comm)   TARGET_COMM="$2";  shift 2 ;;
        --repeat) REPEAT="$2";       shift 2 ;;
        --duration) DURATION="$2";   shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

RUN_TS=$(date +%Y%m%d_%H%M%S)
RESULTS_BASE="$ROOT_DIR/results/stability_${RUN_TS}"
mkdir -p "$RESULTS_BASE"

echo "=== 重复运行稳定性测试 ==="
echo "目标: pid=${TARGET_PID:-all} comm=${TARGET_COMM:-any}"
echo "重复: $REPEAT 次   每次时长: ${DURATION}s   时间窗: ${WINDOW}s"
echo "结果目录: $RESULTS_BASE"
echo ""

for i in $(seq 1 $REPEAT); do
    RUN_DIR="$RESULTS_BASE/run_$(printf '%03d' $i)"
    echo "[run $i/$REPEAT] → $RUN_DIR"

    LOADER_ARGS="--window $WINDOW --duration $DURATION --output $RUN_DIR/"
    if [[ $TARGET_PID -ne 0 ]]; then
        LOADER_ARGS="$LOADER_ARGS --pid $TARGET_PID"
    elif [[ -n $TARGET_COMM ]]; then
        LOADER_ARGS="$LOADER_ARGS --comm $TARGET_COMM"
    else
        echo "[错误] 请指定 --pid 或 --comm" && exit 1
    fi

    python3 "$ROOT_DIR/src/loader.py" $LOADER_ARGS 2>/dev/null || true

    # 运行热点分析
    python3 "$ROOT_DIR/analysis/hotspot.py" \
        --data "$RUN_DIR/" \
        --output "$RUN_DIR/" \
        2>/dev/null || true

    sleep 1   # 避免连续运行互相干扰
done

python3 "$ROOT_DIR/scripts/build_methodology_tables.py" \
    --stability-dir "$RESULTS_BASE" \
    --output "$RESULTS_BASE"

echo ""
echo "自动生成的阈值化摘要："
echo "  $RESULTS_BASE/stability_summary.md"
echo "  $RESULTS_BASE/methodology_recommendations.md"
