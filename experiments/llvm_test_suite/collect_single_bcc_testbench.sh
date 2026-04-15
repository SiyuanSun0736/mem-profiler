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
#   sudo bash experiments/llvm_test_suite/collect_single_bcc_testbench.sh -s aha -v O3-g -d 15
#   sudo bash experiments/llvm_test_suite/collect_single_bcc_testbench.sh \
#       -B data/llvm_test_suite/bin/O3-g/aha_O3-g \
#       -T data/llvm_test_suite/test/O3-g/aha/aha.test \
#       -o data/llvm_test_suite/bcc/O3-g

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/data/llvm_test_suite}"
VARIANT="${VARIANT:-O3-g}"
BENCH="${BENCH:-aha}"
WINDOW_SEC="${WINDOW_SEC:-1.0}"
DURATION_SEC="${DURATION_SEC:-10}"
SAMPLE_RATE="${SAMPLE_RATE:-100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATASET_ROOT/bcc/$VARIANT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRYRUN="${DRYRUN:-0}"
EMIT_EVENTS="${EMIT_EVENTS:-0}"
ENABLE_LBR="${ENABLE_LBR:-0}"
PER_TID="${PER_TID:-0}"
EXTRA_LOADER_ARGS="${EXTRA_LOADER_ARGS:- --track-children}"

BIN_PATH="${BIN_PATH:-}"
TEST_PATH="${TEST_PATH:-}"

usage() {
    cat <<EOF
用法: $0 [-v VARIANT] [-s BENCH] [-B BIN_PATH] [-T TEST_PATH] [-o OUTPUT_ROOT]
         [-w WINDOW_SEC] [-d DURATION_SEC] [-r SAMPLE_RATE] [-n] [-e] [-l] [-p]

选项:
  -v VARIANT       变体名称，默认 O3-g
  -s BENCH         基准名，默认 aha
  -B BIN_PATH      可执行文件绝对/相对路径
  -T TEST_PATH     .test 文件绝对/相对路径
  -o OUTPUT_ROOT   输出根目录，默认 data/llvm_test_suite/bcc/<VARIANT>
  -w WINDOW_SEC    loader 时间窗秒数，默认 1.0
  -d DURATION_SEC  本次采集总时长秒数，默认 10
  -r SAMPLE_RATE   perf sample rate，默认 100
  -n               干跑：仅打印展开后的命令和 loader 调用
  -e               打开 --emit-events
  -l               打开 --lbr（隐含 emit-events）
  -p               打开 --per-tid
  -h               显示帮助

默认输入:
  BIN_PATH  = data/llvm_test_suite/bin/<VARIANT>/<BENCH>_<VARIANT>
  TEST_PATH = data/llvm_test_suite/test/<VARIANT>/<BENCH>/<BENCH>.test
EOF
}

while getopts "v:s:B:T:o:w:d:r:nelph" opt; do
    case "$opt" in
        v) VARIANT="$OPTARG" ;;
        s) BENCH="$OPTARG" ;;
        B) BIN_PATH="$OPTARG" ;;
        T) TEST_PATH="$OPTARG" ;;
        o) OUTPUT_ROOT="$OPTARG" ;;
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
[[ -z "$TEST_PATH" ]] && TEST_PATH="$DATASET_ROOT/test/$VARIANT/$BENCH/${BENCH}.test"

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
OUT_DIR="$OUTPUT_ROOT/${BENCH}_$(date +%Y%m%d_%H%M%S)"

cd "$PROJECT_ROOT"
make all

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
cleanup() {
    if [[ -n "$WORKLOAD_PID" ]] && kill -0 "$WORKLOAD_PID" 2>/dev/null; then
        kill -- -"$WORKLOAD_PID" 2>/dev/null || true
        wait "$WORKLOAD_PID" 2>/dev/null || true
    fi
}

trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM
trap cleanup EXIT

RUN_DIR="$TEST_DIR" BENCH_CMD="$BENCH_CMD" setsid bash -lc '
    trap "exit 0" INT TERM
    cd "$RUN_DIR"
    while true; do
        eval "$BENCH_CMD" >/dev/null 2>&1 || true
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
"${LOADER_CMD[@]}"
echo "[info] 采集完成，输出目录: $OUT_DIR"