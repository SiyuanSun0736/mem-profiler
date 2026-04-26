#!/bin/bash
# =============================================================================
# collect_single_bcc_testbench.sh — 单基准 BCC 一次性采集脚本
# =============================================================================
#
# 功能：
#   1. 读取 llvm-test-suite 的 .test 文件，展开 RUN: 命令。
#   2. 用 bash 持续循环执行该命令，保证短命 benchmark 能反复出现。
#   3. 调用 src/loader.py 进行一次定长采集，输出 JSONL 数据。
#
# 说明：
#   - 该脚本依赖 loader.py 的 --comm 过滤模式。perf 事件会全局挂载，
#     eBPF 在内核侧按 comm 过滤，因此能覆盖循环中不断重建 PID 的基准进程。
#   - comm 受 TASK_COMM_LEN 限制，只保留可执行文件 basename 的前 15 个字符。
#
# 示例：
#   sudo bash experiments/llvm_test_suite/collect_single_bcc_testbench.sh
#   sudo bash experiments/llvm_test_suite/collect_single_bcc_testbench.sh -s aha -v O3 -d 15
#   sudo bash experiments/llvm_test_suite/collect_single_bcc_testbench.sh \
#       -B data/llvm_test_suite/bin/O3/aha_O3 \
#       -T data/llvm_test_suite/test/O3/aha/aha.test \
#       -o data/llvm_test_suite/bcc/O3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/data/llvm_test_suite}"
VARIANT="${VARIANT:-O3}"
BENCH="${BENCH:-aha}"
WINDOW_SEC="${WINDOW_SEC:-1.0}"
DURATION_SEC="${DURATION_SEC:-60}"
SAMPLE_RATE="${SAMPLE_RATE:-100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATASET_ROOT/bcc/$VARIANT}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRYRUN="${DRYRUN:-0}"
EMIT_EVENTS="${EMIT_EVENTS:-0}"
ENABLE_LBR="${ENABLE_LBR:-0}"
PER_TID="${PER_TID:-0}"
EXTRA_LOADER_ARGS="${EXTRA_LOADER_ARGS:- --track-children}"

BIN_PATH="${BIN_PATH:-}"
TEST_PATH="${TEST_PATH:-}"

find_default_test_path() {
    local variant_dir="$1"
    local bench_name="$2"
    local bench_dir="$variant_dir/$bench_name"
    local resolved=""

    [[ -d "$bench_dir" ]] || return 0
    resolved=$(find "$bench_dir" -maxdepth 1 -name "*.test" | sort | head -1)
    if [[ -n "$resolved" ]]; then
        printf '%s\n' "$resolved"
    fi
    return 0
}

detect_available_variants() {
    local variant_dir variant
    [[ -d "$DATASET_ROOT/test" ]] || return 0

    for variant_dir in "$DATASET_ROOT/test"/*; do
        [[ -d "$variant_dir" ]] || continue
        variant="$(basename "$variant_dir")"
        [[ -d "$DATASET_ROOT/bin/$variant" ]] || continue
        printf '%s\n' "$variant"
    done
}

usage() {
    local detected
    detected="$(detect_available_variants | tr '\n' ' ' | sed 's/ $//')"

    cat <<EOF
用法: $0 [-v VARIANT] [-s BENCH] [-B BIN_PATH] [-T TEST_PATH] [-o OUTPUT_ROOT]
         [-O OUTPUT_DIR] [-w WINDOW_SEC] [-d DURATION_SEC] [-r SAMPLE_RATE]
         [-n] [-e] [-l] [-p]

选项:
  -v VARIANT       变体名称，默认 O3
  -s BENCH         基准名，默认 aha
  -B BIN_PATH      可执行文件绝对/相对路径
  -T TEST_PATH     .test 文件绝对/相对路径
  -o OUTPUT_ROOT   输出根目录，默认 data/llvm_test_suite/bcc/<VARIANT>
  -O OUTPUT_DIR    固定输出目录；指定后不再自动追加时间戳
    -w WINDOW_SEC    loader 时间窗秒数，默认 1.0
    -d DURATION_SEC  本次采集总时长秒数，默认 60
    -r SAMPLE_RATE   perf sample period（每 N 次事件触发一次），默认 100
  -n               干跑：仅打印展开后的命令和 loader 调用
  -e               打开 --emit-events
  -l               打开 --lbr（隐含 emit-events）
  -p               打开 --per-tid
  -h               显示帮助

默认输入:
    BIN_PATH  = data/llvm_test_suite/bin/<VARIANT>/<BENCH>_<VARIANT>
    TEST_PATH = data/llvm_test_suite/test/<VARIANT>/<BENCH>/ 目录中的首个 .test 文件
EOF

        [[ -n "$detected" ]] && echo "当前检测到的 VARIANT: $detected"
}

while getopts "v:s:B:T:o:O:w:d:r:nelph" opt; do
    case "$opt" in
        v) VARIANT="$OPTARG" ;;
        s) BENCH="$OPTARG" ;;
        B) BIN_PATH="$OPTARG" ;;
        T) TEST_PATH="$OPTARG" ;;
        o) OUTPUT_ROOT="$OPTARG" ;;
        O) OUTPUT_DIR="$OPTARG" ;;
        w) WINDOW_SEC="$OPTARG" ;;
        d) DURATION_SEC="$OPTARG" ;;
        r) SAMPLE_RATE="$OPTARG" ;;
        n) DRYRUN=1 ;;
        e) EMIT_EVENTS=1 ;;
        l) ENABLE_LBR=1; EMIT_EVENTS=1 ;;
        p) PER_TID=1 ;;
        h) usage; exit 0 ;;
        *) usage >&2; exit 1 ;;
    esac
done

[[ -z "$BIN_PATH" ]] && BIN_PATH="$DATASET_ROOT/bin/$VARIANT/${BENCH}_${VARIANT}"
if [[ -z "$TEST_PATH" ]]; then
    TEST_PATH="$(find_default_test_path "$DATASET_ROOT/test/$VARIANT" "$BENCH")"
fi

[[ -x "$BIN_PATH" ]] || { echo "Error: 可执行文件不存在或不可执行: $BIN_PATH" >&2; exit 1; }
[[ -f "$TEST_PATH" ]] || { echo "Error: .test 文件不存在: $TEST_PATH" >&2; exit 1; }
[[ -f "$PROJECT_ROOT/src/loader.py" ]] || { echo "Error: loader.py 不存在" >&2; exit 1; }

parse_run_cmd() {
    local test_file="$1"
    local binary="$2"
    local test_data="$3"
    local run_raw
    local cmd_part
    local first_word
    local bin_ref_name

    run_raw=$(grep '^RUN:' "$test_file" | head -1 | sed 's/^RUN: //') || return 1
    [[ -n "$run_raw" ]] || return 1

    cmd_part="$run_raw"
    if [[ "$run_raw" == "cd %S ;"* ]]; then
        cmd_part="${run_raw#cd %S ; }"
    fi

    first_word=$(echo "$cmd_part" | awk '{print $1}')
    if [[ "$first_word" == "%S/"* ]]; then
        bin_ref_name="${first_word#%S/}"
        cmd_part=$(printf '%s' "$cmd_part" | sed "s|%S/${bin_ref_name}|${binary}|g")
    fi

    printf '%s' "$cmd_part" | sed "s|%S|${test_data}|g"
}


TEST_DIR="$(cd "$(dirname "$TEST_PATH")" && pwd)"
BIN_PATH="$(cd "$(dirname "$BIN_PATH")" && pwd)/$(basename "$BIN_PATH")"
BENCH_CMD="$(parse_run_cmd "$TEST_PATH" "$BIN_PATH" "$TEST_DIR")" || {
    echo "Error: 未能从 $TEST_PATH 解析 RUN: 行" >&2
    exit 1
}

TARGET_COMM="$(basename "$BIN_PATH")"
TARGET_COMM="${TARGET_COMM:0:15}"
if [[ -n "$OUTPUT_DIR" ]]; then
    OUT_DIR="$OUTPUT_DIR"
else
    OUT_DIR="$OUTPUT_ROOT/${BENCH}_$(date +%Y%m%d_%H%M%S)"
fi

LOADER_CMD=(
    "$PYTHON_BIN" "$PROJECT_ROOT/src/loader.py"
    --comm "$TARGET_COMM"
    --window "$WINDOW_SEC"
    --duration "$DURATION_SEC"
    --sample-rate "$SAMPLE_RATE"
    --output "$OUT_DIR"
)
# 将 EXTRA_LOADER_ARGS 按 shell 单词拆分追加（支持多个 flag）
[[ -n "$EXTRA_LOADER_ARGS" ]] && { read -ra _xtra <<< "$EXTRA_LOADER_ARGS"; LOADER_CMD+=("${_xtra[@]}"); }

[[ "$EMIT_EVENTS" -eq 1 ]] && LOADER_CMD+=(--emit-events)
[[ "$ENABLE_LBR" -eq 1 ]] && LOADER_CMD+=(--lbr)
[[ "$PER_TID" -eq 1 ]] && LOADER_CMD+=(--per-tid)

echo "[info] BIN_PATH:    $BIN_PATH"
echo "[info] TEST_PATH:   $TEST_PATH"
echo "[info] TEST_DIR:    $TEST_DIR"
echo "[info] TARGET_COMM: $TARGET_COMM"
echo "[info] BENCH_CMD:   $BENCH_CMD"
echo "[info] OUTPUT_DIR:  $OUT_DIR"

if [[ "$DRYRUN" -eq 1 ]]; then
    echo "[DRYRUN] workload: setsid bash -lc 'cd \"$TEST_DIR\"; while true; do eval \"$BENCH_CMD\" >/dev/null 2>&1 || true; done'"
    printf '[DRYRUN] loader: '
    printf '%q ' "${LOADER_CMD[@]}"
    echo
    exit 0
fi

[[ "$EUID" -eq 0 ]] || {
    echo "Error: 请使用 sudo 运行该脚本，以便 loader.py 能挂载 perf/BPF 事件" >&2
    exit 1
}

mkdir -p "$OUTPUT_ROOT"

WORKLOAD_PID=""
LOADER_PID=""
_COUNT_FILE=""
_CLEANUP_DONE=0

wait_for_pid_exit() {
    local pid="$1"
    local max_checks="${2:-20}"
    local i

    for ((i = 0; i < max_checks; i++)); do
        if ! kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        sleep 0.1
    done
    return 1
}

wait_for_process_group_exit() {
    local pgid="$1"
    local max_checks="${2:-20}"
    local i

    for ((i = 0; i < max_checks; i++)); do
        if [[ -z "$(ps -o pid= -g "$pgid" 2>/dev/null | tr -d '[:space:]')" ]]; then
            return 0
        fi
        sleep 0.1
    done
    return 1
}

cleanup() {
    [[ "$_CLEANUP_DONE" -eq 1 ]] && return
    _CLEANUP_DONE=1

    if [[ -n "$LOADER_PID" ]] && kill -0 "$LOADER_PID" 2>/dev/null; then
        kill -s INT "$LOADER_PID" 2>/dev/null || true
        wait_for_pid_exit "$LOADER_PID" 20 || {
            kill -s TERM "$LOADER_PID" 2>/dev/null || true
            wait_for_pid_exit "$LOADER_PID" 20 || kill -s KILL "$LOADER_PID" 2>/dev/null || true
        }
        wait "$LOADER_PID" 2>/dev/null || true
    fi
    LOADER_PID=""

    if [[ -n "$WORKLOAD_PID" ]] && kill -0 "$WORKLOAD_PID" 2>/dev/null; then
        kill -s TERM -- "-$WORKLOAD_PID" 2>/dev/null || true
        wait_for_process_group_exit "$WORKLOAD_PID" 20 || {
            kill -s KILL -- "-$WORKLOAD_PID" 2>/dev/null || true
            wait_for_process_group_exit "$WORKLOAD_PID" 10 || true
        }
        wait "$WORKLOAD_PID" 2>/dev/null || true
    fi
    WORKLOAD_PID=""
}

trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM
trap cleanup EXIT

_COUNT_FILE=$(mktemp)
printf '0' > "$_COUNT_FILE"
RUN_DIR="$TEST_DIR" BENCH_CMD="$BENCH_CMD" COUNT_FILE="$_COUNT_FILE" setsid bash -lc '
    trap "exit 0" INT TERM
    cd "$RUN_DIR"
    _cnt=0
    while true; do
        eval "$BENCH_CMD" >/dev/null 2>&1 || true
        _cnt=$((_cnt + 1))
        printf '%d' "$_cnt" > "$COUNT_FILE"
    done
' &
WORKLOAD_PID=$!

for _ in $(seq 1 100); do
    if pgrep -x "$TARGET_COMM" >/dev/null 2>&1; then
        break
    fi
    sleep 0.1
done

echo "[info] workload loop pid=$WORKLOAD_PID，开始一次 BCC 采集"
loader_rc=0
"${LOADER_CMD[@]}" &
LOADER_PID=$!
wait "$LOADER_PID" || loader_rc=$?
LOADER_PID=""
cleanup

# 读取真实完成轮次并写入 run_metadata.jsonl
_completion_count=0
if [[ -n "$_COUNT_FILE" && -f "$_COUNT_FILE" ]]; then
    _val=$(cat "$_COUNT_FILE" 2>/dev/null)
    [[ "$_val" =~ ^[0-9]+$ ]] && _completion_count=$_val
    rm -f "$_COUNT_FILE"
fi
if [[ -f "$OUT_DIR/run_metadata.jsonl" ]]; then
    printf '{"completion_count": %d, "_record_type": "run_stats"}\n' "$_completion_count" \
        >> "$OUT_DIR/run_metadata.jsonl"
fi

if [[ "$loader_rc" -ne 0 ]]; then
    exit "$loader_rc"
fi

echo "[info] 采集完成 (完成 ${_completion_count} 轮)，输出目录: $OUT_DIR"