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

# ---- 跨 run 统计（简单 Python one-liner）----
echo ""
echo "=== 统计各指标跨 run 均值和 CV ==="
python3 - <<'PYEOF'
import json, pathlib, statistics, sys, os

base = pathlib.Path(os.environ.get("RESULTS_BASE", "."))
metrics = ["llc_load_misses", "llc_store_misses", "dtlb_misses", "minor_faults", "major_faults"]

run_dirs = sorted(base.glob("run_*/"))
if not run_dirs:
    print("无 run 目录，跳过统计")
    sys.exit(0)

per_metric: dict[str, list[float]] = {m: [] for m in metrics}

for rdir in run_dirs:
    f = rdir / "window_metrics.jsonl"
    if not f.exists():
        continue
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    for m in metrics:
        total = sum(r.get(m, 0) for r in rows)
        per_metric[m].append(total)

print(f"{'指标':<22} {'均值':>12} {'标准差':>12} {'CV%':>8}")
print("-" * 58)
for m, vals in per_metric.items():
    if not vals:
        continue
    mean = statistics.mean(vals)
    std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
    cv   = (std / mean * 100) if mean > 0 else 0.0
    flag = " ⚠" if cv > 15 else ""
    print(f"{m:<22} {mean:>12.1f} {std:>12.1f} {cv:>7.1f}%{flag}")
PYEOF
export RESULTS_BASE="$RESULTS_BASE"
