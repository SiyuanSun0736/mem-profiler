#!/usr/bin/env bash
# experiments/correctness/run_reference_capture.sh
#
# 一键收集“eBPF 指标正确性对照”所需的参考材料：
#   1. eBPF 采集结果（window_metrics.jsonl / events.jsonl）
#   2. /proc/<pid>/stat 前后快照 + /proc/<pid>/maps 快照
#   3. perf stat 输出
#   4. perf record + perf report 输出
#   5. 可选：函数级归因与最终 Markdown / CSV 对照表

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"
ORIGINAL_ARGS=("$@")

if [[ -z "$PYTHON_BIN" ]]; then
    echo "[错误] 未找到 python3。" >&2
    exit 1
fi

if ! command -v perf >/dev/null 2>&1; then
    echo "[错误] 未找到 perf。" >&2
    exit 1
fi

usage() {
    cat <<'EOF'
用法：
  sudo bash experiments/correctness/run_reference_capture.sh [选项] -- <workload command...>

示例：
    sudo bash experiments/correctness/run_reference_capture.sh \
    --metric llc_load_misses \
    --duration 30 \
    --output results/correctness_xsbench \
        -- bash -lc 'exec ./bin/xsbench -g 10000 -p 400000'

选项：
  --output DIR              输出目录，默认 results/correctness_<timestamp>
  --duration SEC            每轮 workload 最长运行时间，默认 30
  --window SEC              eBPF 时间窗大小，默认 1.0
  --metric NAME             函数热点对照的目标指标，默认 llc_load_misses
  --top N                   函数热点 Top-N，默认 30
  --sample-rate N           loader.py 的 sample rate，默认 100
  --pmu-backend NAME        loader.py 的 PMU backend，默认 auto
  --perf-stat-events LIST   perf stat 事件列表，默认 cycles,instructions,cache-misses,dTLB-load-misses,iTLB-load-misses,page-faults,minor-faults,major-faults
  --perf-record-event NAME  perf record 主事件；未指定时按 metric 自动映射
  --no-track-children       不传递 --track-children 给 loader.py
  --skip-attribution        跳过 analysis/attribution.py
  --skip-table-build        跳过 scripts/build_correctness_tables.py
  --keep-perf-data          保留 perf.data；默认生成 perf report 后删除

说明：
    1. 该脚本会把同一 workload 命令重放 3 次：一次给 eBPF，一次给 perf stat，一次给 perf record。
  2. 为了保证三轮长度一致，workload 会被 timeout 限制在 --duration 内；若命令本身更短，会自然提前结束。
    3. eBPF 这一轮需要拿到真实 workload PID；如果你使用 shell 包装，请在命令里用 exec 替换外层 shell 进程。
    4. 若 workload 非常短，/proc/<pid>/stat 末态仍可能出现少量观测误差；这种场景建议把输入规模拉长。
EOF
}

metric_to_perf_event() {
    case "$1" in
        llc_load_misses|llc_store_misses)
            echo "cache-misses"
            ;;
        dtlb_misses)
            echo "dTLB-load-misses"
            ;;
        minor_faults)
            echo "minor-faults"
            ;;
        major_faults)
            echo "major-faults"
            ;;
        *)
            echo ""
            ;;
    esac
}

write_command_file() {
    local out_file="$1"
    shift
    : > "$out_file"
    printf '%q ' "$@" >> "$out_file"
    printf '\n' >> "$out_file"
}

wait_for_proc_file() {
    local pid="$1"
    local proc_name="$2"
    local attempt=0
    while [[ $attempt -lt 100 ]]; do
        if [[ -r "/proc/$pid/$proc_name" ]]; then
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            return 1
        fi
        sleep 0.05
        attempt=$((attempt + 1))
    done
    return 1
}

track_latest_proc_stat() {
    local pid="$1"
    local out_file="$2"
    while kill -0 "$pid" 2>/dev/null; do
        if [[ -r "/proc/$pid/stat" ]]; then
            cat "/proc/$pid/stat" > "$out_file" 2>/dev/null || true
        fi
        sleep 0.05
    done
}

launch_workload() {
    local stdout_log="$1"
    local stderr_log="$2"
    shift 2
    timeout --preserve-status "${DURATION}s" "$@" >"$stdout_log" 2>"$stderr_log" &
    echo $!
}

launch_workload_direct() {
    local stdout_log="$1"
    local stderr_log="$2"
    shift 2
    "$@" >"$stdout_log" 2>"$stderr_log" &
    echo $!
}

OUT_DIR=""
DURATION=30
WINDOW=1.0
METRIC="llc_load_misses"
TOP=30
SAMPLE_RATE=100
PMU_BACKEND="auto"
PERF_STAT_EVENTS="cycles,instructions,cache-misses,dTLB-load-misses,iTLB-load-misses,page-faults,minor-faults,major-faults"
PERF_RECORD_EVENT=""
TRACK_CHILDREN=1
SKIP_ATTRIBUTION=0
SKIP_TABLE_BUILD=0
KEEP_PERF_DATA=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)
            OUT_DIR="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --window)
            WINDOW="$2"
            shift 2
            ;;
        --metric)
            METRIC="$2"
            shift 2
            ;;
        --top)
            TOP="$2"
            shift 2
            ;;
        --sample-rate)
            SAMPLE_RATE="$2"
            shift 2
            ;;
        --pmu-backend)
            PMU_BACKEND="$2"
            shift 2
            ;;
        --perf-stat-events)
            PERF_STAT_EVENTS="$2"
            shift 2
            ;;
        --perf-record-event)
            PERF_RECORD_EVENT="$2"
            shift 2
            ;;
        --no-track-children)
            TRACK_CHILDREN=0
            shift
            ;;
        --skip-attribution)
            SKIP_ATTRIBUTION=1
            shift
            ;;
        --skip-table-build)
            SKIP_TABLE_BUILD=1
            shift
            ;;
        --keep-perf-data)
            KEEP_PERF_DATA=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "[错误] 未知参数: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
        exec sudo -E bash "$0" "${ORIGINAL_ARGS[@]}"
    fi
    echo "[错误] 该脚本需要 root 权限；请用 sudo 运行。" >&2
    exit 1
fi

if [[ $# -eq 0 ]]; then
    echo "[错误] 缺少 workload command；请在 -- 后提供。" >&2
    usage
    exit 1
fi

case "$METRIC" in
    llc_load_misses|llc_store_misses|dtlb_misses|minor_faults|major_faults)
        ;;
    *)
        echo "[错误] 当前脚本只支持 llc_load_misses / llc_store_misses / dtlb_misses / minor_faults / major_faults。" >&2
        exit 1
        ;;
esac

if [[ -z "$PERF_RECORD_EVENT" ]]; then
    PERF_RECORD_EVENT="$(metric_to_perf_event "$METRIC")"
fi

if [[ -z "$PERF_RECORD_EVENT" ]]; then
    echo "[错误] 无法为 metric=$METRIC 推断 perf record 事件，请显式传 --perf-record-event。" >&2
    exit 1
fi

if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="$ROOT_DIR/results/correctness_$(date +%Y%m%d_%H%M%S)"
elif [[ "$OUT_DIR" != /* ]]; then
    OUT_DIR="$ROOT_DIR/$OUT_DIR"
fi

WORKLOAD=("$@")

EBPF_DIR="$OUT_DIR/ebpf"
REF_DIR="$OUT_DIR/reference"
ATTR_DIR="$OUT_DIR/attribution"
TABLE_DIR="$OUT_DIR/tables"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$EBPF_DIR" "$REF_DIR" "$ATTR_DIR" "$TABLE_DIR" "$LOG_DIR"

write_command_file "$OUT_DIR/workload_command.txt" "${WORKLOAD[@]}"
cat > "$OUT_DIR/capture_config.env" <<EOF
DURATION=$DURATION
WINDOW=$WINDOW
METRIC=$METRIC
TOP=$TOP
SAMPLE_RATE=$SAMPLE_RATE
PMU_BACKEND=$PMU_BACKEND
PERF_STAT_EVENTS=$PERF_STAT_EVENTS
PERF_RECORD_EVENT=$PERF_RECORD_EVENT
TRACK_CHILDREN=$TRACK_CHILDREN
EOF

echo "=== 正确性参考材料采集 ==="
echo "输出目录: $OUT_DIR"
echo "目标 metric: $METRIC"
echo "perf record event: $PERF_RECORD_EVENT"
echo "workload: $(cat "$OUT_DIR/workload_command.txt")"
echo ""

echo "[1/5] eBPF + /proc 快照..."
EBPF_WORKLOAD_PID="$(launch_workload_direct "$LOG_DIR/ebpf_workload.stdout.log" "$LOG_DIR/ebpf_workload.stderr.log" "${WORKLOAD[@]}")"
echo "$EBPF_WORKLOAD_PID" > "$REF_DIR/target_pid.txt"

if ! wait_for_proc_file "$EBPF_WORKLOAD_PID" "stat"; then
    echo "[错误] workload PID=$EBPF_WORKLOAD_PID 启动过快或已退出，未能读取 /proc/$EBPF_WORKLOAD_PID/stat。" >&2
    wait "$EBPF_WORKLOAD_PID" 2>/dev/null || true
    exit 1
fi

cat "/proc/$EBPF_WORKLOAD_PID/stat" > "$REF_DIR/proc_stat_before.txt"
if wait_for_proc_file "$EBPF_WORKLOAD_PID" "maps"; then
    cat "/proc/$EBPF_WORKLOAD_PID/maps" > "$REF_DIR/proc_maps.txt" 2>/dev/null || true
fi

track_latest_proc_stat "$EBPF_WORKLOAD_PID" "$REF_DIR/proc_stat_after.txt" &
PROC_MONITOR_PID=$!

LOADER_CMD=(
    "$PYTHON_BIN" "$ROOT_DIR/src/loader.py"
    --pid "$EBPF_WORKLOAD_PID"
    --window "$WINDOW"
    --duration "$DURATION"
    --output "$EBPF_DIR"
    --sample-rate "$SAMPLE_RATE"
    --pmu-backend "$PMU_BACKEND"
    --emit-events
)
if [[ $TRACK_CHILDREN -eq 1 ]]; then
    LOADER_CMD+=(--track-children)
fi

"${LOADER_CMD[@]}" >"$LOG_DIR/loader.stdout.log" 2>"$LOG_DIR/loader.stderr.log" || true
if kill -0 "$EBPF_WORKLOAD_PID" 2>/dev/null; then
    kill -TERM "$EBPF_WORKLOAD_PID" 2>/dev/null || true
fi
wait "$EBPF_WORKLOAD_PID" 2>/dev/null || true
wait "$PROC_MONITOR_PID" 2>/dev/null || true

if [[ ! -s "$REF_DIR/proc_stat_after.txt" ]]; then
    echo "[警告] 未抓到 /proc/$EBPF_WORKLOAD_PID/stat 末态快照，fault 对照表可能无法生成。" >&2
fi

echo "[2/5] perf stat..."
if ! perf stat -x, -o "$REF_DIR/perf_stat.csv" -e "$PERF_STAT_EVENTS" -- \
    timeout --preserve-status "${DURATION}s" "${WORKLOAD[@]}" \
    >"$LOG_DIR/perf_stat_workload.stdout.log" 2>"$LOG_DIR/perf_stat_workload.stderr.log"; then
    echo "[警告] perf stat 返回非零退出码；保留已生成的输出继续后续流程。" >&2
fi

echo "[3/5] perf record + perf report..."
PERF_DATA="$REF_DIR/perf.data"
if perf record -o "$PERF_DATA" -e "$PERF_RECORD_EVENT" -g -- \
    timeout --preserve-status "${DURATION}s" "${WORKLOAD[@]}" \
    >"$LOG_DIR/perf_record_workload.stdout.log" 2>"$LOG_DIR/perf_record_workload.stderr.log"; then
    perf report -i "$PERF_DATA" --stdio --no-children --percent-limit 0 > "$REF_DIR/perf_report.txt"
else
    echo "[警告] perf record 返回非零退出码；跳过 perf report。" >&2
fi

echo "[4/5] 函数级归因..."
FUNCTION_HOTSPOT_CSV=""
if [[ $SKIP_ATTRIBUTION -eq 0 ]]; then
    if [[ -s "$EBPF_DIR/events.jsonl" && -s "$REF_DIR/proc_maps.txt" ]]; then
        "$PYTHON_BIN" "$ROOT_DIR/analysis/attribution.py" \
            --data "$EBPF_DIR" \
            --pid "$EBPF_WORKLOAD_PID" \
            --maps-file "$REF_DIR/proc_maps.txt" \
            --metric "$METRIC" \
            --top "$TOP" \
            --output "$ATTR_DIR" \
            >"$LOG_DIR/attribution.stdout.log" 2>"$LOG_DIR/attribution.stderr.log" || true
        FUNCTION_HOTSPOT_CSV="$ATTR_DIR/function_hotspot_${METRIC}.csv"
    else
        echo "[警告] 缺少 events.jsonl 或 proc_maps.txt，跳过函数级归因。" >&2
    fi
fi

echo "[5/5] 生成对照表..."
if [[ $SKIP_TABLE_BUILD -eq 0 ]]; then
    TABLE_ARGS=(
        --pmu-window-metrics "$EBPF_DIR/window_metrics.jsonl"
        --perf-stat "$REF_DIR/perf_stat.csv"
        --output "$TABLE_DIR"
    )

    if [[ -s "$EBPF_DIR/window_metrics.jsonl" && -s "$REF_DIR/proc_stat_before.txt" && -s "$REF_DIR/proc_stat_after.txt" ]]; then
        TABLE_ARGS+=(
            --fault-window-metrics "$EBPF_DIR/window_metrics.jsonl"
            --proc-stat-before "$REF_DIR/proc_stat_before.txt"
            --proc-stat-after "$REF_DIR/proc_stat_after.txt"
        )
    fi

    if [[ -n "$FUNCTION_HOTSPOT_CSV" && -s "$FUNCTION_HOTSPOT_CSV" && -s "$REF_DIR/perf_report.txt" ]]; then
        TABLE_ARGS+=(
            --function-hotspot "$FUNCTION_HOTSPOT_CSV"
            --perf-report "$REF_DIR/perf_report.txt"
        )
    fi

    "$PYTHON_BIN" "$ROOT_DIR/scripts/build_correctness_tables.py" "${TABLE_ARGS[@]}"
fi

if [[ $KEEP_PERF_DATA -eq 0 ]]; then
    rm -f "$PERF_DATA"
fi

echo ""
echo "=== 采集完成 ==="
echo "eBPF 输出:      $EBPF_DIR"
echo "参考材料:      $REF_DIR"
echo "归因结果:      $ATTR_DIR"
echo "对照表输出:    $TABLE_DIR"