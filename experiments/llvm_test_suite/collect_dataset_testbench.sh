#!/bin/bash
# =============================================================================
# collect_dataset_testbench.sh — 基于预编译可执行文件 + .test 文件的 BCC 批量采集脚本
# =============================================================================
#
# 功能概述：
#   使用 data/llvm_test_suite/test/<VARIANT>/ 目录下的 .test 文件作为运行规格，
#   对 data/llvm_test_suite/bin/<VARIANT>/ 中预编译的可执行文件依次执行一次
#   BCC 采集。实际采集入口为 src/loader.py -> src/collector.py。
#
# 工作方式：
#   1. 解析 .test 文件中的首条 RUN: 命令。
#   2. 启动 benchmark 的无限循环工作负载，保证短命程序能持续重建。
#   3. 调用 collect_single_bcc_testbench.sh 对单个 benchmark 做一次定长采集。
#   4. 将输出目录与运行规格写入 manifest，便于后续分析脚本批量消费。
#
# 输出目录结构：
#   data/llvm_test_suite/
#   ├── bcc/<VARIANT>/                   BCC 采集结果根目录
#   │   ├── aha_20260419_220101/
#   │   │   ├── run_metadata.jsonl
#   │   │   ├── window_metrics.jsonl
#   │   │   └── events.jsonl            （仅 emit-events/lbr 时存在）
#   │   └── ...
#   └── manifest_bcc_<VARIANT>.jsonl    数据集清单（每行一条 JSON）
#
# 用法：
#   sudo bash experiments/llvm_test_suite/collect_dataset_testbench.sh
#   sudo bash experiments/llvm_test_suite/collect_dataset_testbench.sh -s aha
#   VARIANT=O2 bash experiments/llvm_test_suite/collect_dataset_testbench.sh -n
#
# 环境变量（可覆盖默认值）：
#   DATASET_ROOT      llvm-test-suite 派生数据根目录（默认 data/llvm_test_suite）
#   RUN_ROOT          运行期日志根目录（默认 results/llvm_test_suite）
#   LOG_DIR           脚本日志目录（默认 results/llvm_test_suite/log）
#   VARIANT           变体名称（默认 O3）
#   BIN_DIR           可执行文件目录（默认 data/llvm_test_suite/bin/<VARIANT>）
#   TEST_DIR          测试规格目录（默认 data/llvm_test_suite/test/<VARIANT>）
#   OUTPUT_ROOT       BCC 输出根目录（默认 data/llvm_test_suite/bcc/<VARIANT>）
#   MANIFEST          数据集清单路径（默认 data/llvm_test_suite/manifest_bcc_<VARIANT>.jsonl）
#   SINGLE_SCRIPT     单基准采集脚本路径（默认 experiments/llvm_test_suite/collect_single_bcc_testbench.sh）
#   DEDUP_SCRIPT      变体去重脚本路径（默认 experiments/llvm_test_suite/dedup_dataset_variant.py）
#   PYTHON_BIN        Python 可执行文件（默认 python3）
#   WINDOW_SEC        loader 时间窗大小，秒（默认 1.0）
#   DURATION_SEC      单次采集总时长，秒（默认 60）
#   SAMPLE_RATE       perf sample period（默认 100；每 N 次事件触发一次）
#   BENCH_FILTER      只采集指定基准名称（留空=全部）
#   OVERWRITE         1=总是重新采集，0=已有成功输出则跳过（默认 1）
#   RETRY_MAX         每个基准最多重试次数（默认 2）
#   DRYRUN            1=只解析和打印命令，不实际采集（默认 0）
#   DEDUP_AFTER_COLLECT 1=采集结束后自动去重并重建 manifest（默认 1）
#   EMIT_EVENTS       1=输出 events.jsonl（默认 0）
#   ENABLE_LBR        1=打开 --lbr（隐含 emit-events，默认 0）
#   PER_TID           1=按 TID 聚合（默认 0）
#   EXTRA_LOADER_ARGS 透传给 loader.py 的额外参数（默认 --track-children）
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/data/llvm_test_suite}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/results/llvm_test_suite}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/log}"
LOCK_DIR="${LOCK_DIR:-$RUN_ROOT/lock}"

VARIANT="${VARIANT:-O3}"
BIN_DIR="${BIN_DIR:-}"
TEST_DIR="${TEST_DIR:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
MANIFEST="${MANIFEST:-}"
SINGLE_SCRIPT="${SINGLE_SCRIPT:-$SCRIPT_DIR/collect_single_bcc_testbench.sh}"
DEDUP_SCRIPT="${DEDUP_SCRIPT:-$SCRIPT_DIR/dedup_dataset_variant.py}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

WINDOW_SEC="${WINDOW_SEC:-1.0}"
DURATION_SEC="${DURATION_SEC:-60}"
SAMPLE_RATE="${SAMPLE_RATE:-100}"
BENCH_FILTER="${BENCH_FILTER:-}"
OVERWRITE="${OVERWRITE:-1}"
RETRY_MAX="${RETRY_MAX:-2}"
DRYRUN="${DRYRUN:-0}"
DEDUP_AFTER_COLLECT="${DEDUP_AFTER_COLLECT:-1}"
EMIT_EVENTS="${EMIT_EVENTS:-0}"
ENABLE_LBR="${ENABLE_LBR:-0}"
PER_TID="${PER_TID:-0}"
EXTRA_LOADER_ARGS="${EXTRA_LOADER_ARGS:- --track-children}"

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

format_available_variants() {
    detect_available_variants | tr '\n' ' ' | sed 's/ $//'
}

usage() {
    local available_variants
    available_variants="$(format_available_variants)"

    cat <<EOF
用法: $0 [-v VARIANT] [-b BIN_DIR] [-t TEST_DIR] [-d OUTPUT_ROOT]
         [-w DURATION_SEC] [-i WINDOW_SEC] [-r SAMPLE_RATE] [-s BENCH]
         [-n] [-e] [-l] [-p]

选项:
  -v VARIANT       变体名称，默认 O3
  -b BIN_DIR       可执行文件目录
  -t TEST_DIR      测试规格目录
    -d OUTPUT_ROOT   BCC 输出根目录
    -w DURATION_SEC  单次采集总时长，秒（默认 60）
  -i WINDOW_SEC    loader 时间窗，秒（默认 1.0）
    -r SAMPLE_RATE   perf sample period（默认 100；每 N 次事件触发一次）
  -s BENCH         只采集指定 benchmark
  -n               DRYRUN 模式
  -e               打开 --emit-events
  -l               打开 --lbr（隐含 emit-events）
  -p               打开 --per-tid
  -h               显示帮助

示例:
  sudo bash $0
  sudo bash $0 -s aha
  sudo WINDOW_SEC=0.5 DURATION_SEC=20 bash $0 -s minisat
  bash $0 -n -s aha
EOF

    [[ -n "$available_variants" ]] && echo "当前检测到的 VARIANT: $available_variants"
    exit 0
}

while getopts "v:b:t:d:w:i:r:s:nelph" opt; do
    case "$opt" in
        v) VARIANT="$OPTARG" ;;
        b) BIN_DIR="$OPTARG" ;;
        t) TEST_DIR="$OPTARG" ;;
        d) OUTPUT_ROOT="$OPTARG" ;;
        w) DURATION_SEC="$OPTARG" ;;
        i) WINDOW_SEC="$OPTARG" ;;
        r) SAMPLE_RATE="$OPTARG" ;;
        s) BENCH_FILTER="$OPTARG" ;;
        n) DRYRUN=1 ;;
        e) EMIT_EVENTS=1 ;;
        l) ENABLE_LBR=1; EMIT_EVENTS=1 ;;
        p) PER_TID=1 ;;
        h) usage ;;
        *) echo "未知选项：-$OPTARG" >&2; exit 1 ;;
    esac
done

[[ -z "$BIN_DIR" ]] && BIN_DIR="$DATASET_ROOT/bin/$VARIANT"
[[ -z "$TEST_DIR" ]] && TEST_DIR="$DATASET_ROOT/test/$VARIANT"
[[ -z "$OUTPUT_ROOT" ]] && OUTPUT_ROOT="$DATASET_ROOT/bcc/$VARIANT"
[[ -z "$MANIFEST" ]] && MANIFEST="$DATASET_ROOT/manifest_bcc_${VARIANT}.jsonl"
AVAILABLE_VARIANTS="$(format_available_variants)"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()  { printf "${CYAN}[INFO]${NC}  %s\n" "$*"; }
pass()  { printf "${GREEN}[OK]${NC}    %s\n" "$*"; }
warn()  { printf "${YELLOW}[WARN]${NC}  %s\n" "$*"; }
err()   { printf "${RED}[SKIP]${NC}  %s\n" "$*"; }
bold()  { printf "${BOLD}%s${NC}\n" "$*"; }
retry() { printf "${YELLOW}[RETRY]${NC} %s\n" "$*"; }

COUNT_TOTAL=0
COUNT_OK=0
COUNT_SKIP=0
CURRENT_CHILD_PID=""
TEE_PID=""
LOCK_FILE=""
LOCK_FD=""
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

acquire_variant_lock() {
    command -v flock >/dev/null 2>&1 || {
        echo "Error: flock 不存在，无法启用采集互斥锁" >&2
        exit 1
    }

    mkdir -p "$LOCK_DIR"
    LOCK_FILE="$LOCK_DIR/collect_dataset_testbench_${VARIANT}.lock"
    exec {LOCK_FD}> "$LOCK_FILE"

    if ! flock -n "$LOCK_FD"; then
        echo "Error: VARIANT=$VARIANT 已有采集进程运行中，锁文件: $LOCK_FILE" >&2
        exit 1
    fi
}

release_variant_lock() {
    [[ -n "$LOCK_FD" ]] || return

    flock -u "$LOCK_FD" 2>/dev/null || true
    eval "exec ${LOCK_FD}>&-"
    LOCK_FD=""
}

cleanup() {
    [[ "$_CLEANUP_DONE" -eq 1 ]] && return
    _CLEANUP_DONE=1

    if [[ -n "$CURRENT_CHILD_PID" ]] && kill -0 "$CURRENT_CHILD_PID" 2>/dev/null; then
        kill -s INT "$CURRENT_CHILD_PID" 2>/dev/null || true
        wait_for_pid_exit "$CURRENT_CHILD_PID" 20 || {
            kill -s TERM "$CURRENT_CHILD_PID" 2>/dev/null || true
            wait_for_pid_exit "$CURRENT_CHILD_PID" 20 || kill -s KILL "$CURRENT_CHILD_PID" 2>/dev/null || true
        }
        wait "$CURRENT_CHILD_PID" 2>/dev/null || true
    fi
    CURRENT_CHILD_PID=""

    if [[ -n "$TEE_PID" ]]; then
        kill -s TERM "$TEE_PID" 2>/dev/null || true
        wait "$TEE_PID" 2>/dev/null || true
    fi
    TEE_PID=""

    release_variant_lock
}

trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM
trap cleanup EXIT

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

is_valid_dataset_dir() {
    local out_dir="$1"
    [[ -f "$out_dir/run_metadata.jsonl" && -f "$out_dir/window_metrics.jsonl" ]]
}

find_latest_dataset_dir() {
    local bench_name="$1"
    local latest
    latest=$(find "$OUTPUT_ROOT" -maxdepth 1 -mindepth 1 -type d -name "${bench_name}_*" | sort | tail -1)
    if [[ -n "$latest" ]]; then
        printf '%s\n' "$latest"
    fi
    return 0
}

run_single_bench() {
    local bench_name="$1"
    local binary="$2"
    local test_file="$3"
    local out_dir="$4"
    local single_cmd=(
        bash "$SINGLE_SCRIPT"
        -v "$VARIANT"
        -s "$bench_name"
        -B "$binary"
        -T "$test_file"
        -o "$OUTPUT_ROOT"
        -O "$out_dir"
        -w "$WINDOW_SEC"
        -d "$DURATION_SEC"
        -r "$SAMPLE_RATE"
    )

    [[ "$DRYRUN" -eq 1 ]] && single_cmd+=(-n)
    [[ "$EMIT_EVENTS" -eq 1 ]] && single_cmd+=(-e)
    [[ "$ENABLE_LBR" -eq 1 ]] && single_cmd+=(-l)
    [[ "$PER_TID" -eq 1 ]] && single_cmd+=(-p)

    PYTHON_BIN="$PYTHON_BIN" \
    EXTRA_LOADER_ARGS="$EXTRA_LOADER_ARGS" \
        "${single_cmd[@]}"
}

mkdir -p "$RUN_ROOT" "$LOG_DIR" "$OUTPUT_ROOT"
cd "$PROJECT_ROOT"

[[ -d "$BIN_DIR" ]] || {
    echo "Error: BIN_DIR 不存在: $BIN_DIR" >&2
    [[ -n "$AVAILABLE_VARIANTS" ]] && echo "Hint: 当前可用 VARIANT: $AVAILABLE_VARIANTS" >&2
    exit 1
}
[[ -d "$TEST_DIR" ]] || {
    echo "Error: TEST_DIR 不存在: $TEST_DIR" >&2
    [[ -n "$AVAILABLE_VARIANTS" ]] && echo "Hint: 当前可用 VARIANT: $AVAILABLE_VARIANTS" >&2
    exit 1
}
[[ -f "$SINGLE_SCRIPT" ]] || { echo "Error: 单基准脚本不存在: $SINGLE_SCRIPT" >&2; exit 1; }
[[ "$DRYRUN" -eq 1 || "$DEDUP_AFTER_COLLECT" -eq 0 || -f "$DEDUP_SCRIPT" ]] || {
    echo "Error: 去重脚本不存在: $DEDUP_SCRIPT" >&2
    exit 1
}
[[ -f "$PROJECT_ROOT/src/loader.py" ]] || { echo "Error: loader.py 不存在" >&2; exit 1; }

LOG_TS=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOG_DIR/collect_dataset_testbench_${VARIANT}_${LOG_TS}_$$.log"
exec > >(tee -a "$LOGFILE") 2>&1
TEE_PID=$!

if [[ "$DRYRUN" -eq 0 && "$EUID" -ne 0 ]]; then
    echo "Error: 请使用 sudo 运行该脚本，以便 loader.py 挂载 perf/BPF 事件" >&2
    exit 1
fi

if [[ "$DRYRUN" -eq 0 ]]; then
    acquire_variant_lock
fi

info "记录脚本输出到: ${LOGFILE#$PROJECT_ROOT/}"
[[ -n "$LOCK_FILE" ]] && info "采集锁:        ${LOCK_FILE#$PROJECT_ROOT/}"

bold "======================================================"
bold "     BCC 批量采集：$VARIANT 预编译基准"
bold "======================================================"
info "BIN_DIR:        $BIN_DIR"
info "TEST_DIR:       $TEST_DIR"
info "OUTPUT_ROOT:    $OUTPUT_ROOT"
info "WINDOW_SEC:     ${WINDOW_SEC}s"
info "DURATION_SEC:   ${DURATION_SEC}s"
info "SAMPLE_RATE:    $SAMPLE_RATE"
info "RETRY_MAX:      $RETRY_MAX"
info "EXTRA_ARGS:     ${EXTRA_LOADER_ARGS:-<none>}"
[[ "$DRYRUN" -eq 0 && "$DEDUP_AFTER_COLLECT" -eq 1 ]] && info "DEDUP_SCRIPT:   ${DEDUP_SCRIPT#$PROJECT_ROOT/}"
[[ -n "$BENCH_FILTER" ]] && warn "★ 单基准模式：仅处理 '$BENCH_FILTER' ★"
[[ "$DRYRUN" -eq 1 ]] && warn "★ DRYRUN 模式：仅解析命令，不执行采集 ★"
echo

[[ "$DRYRUN" -eq 0 ]] && : > "$MANIFEST"

for test_subdir in "$TEST_DIR"/*/; do
    [[ -d "$test_subdir" ]] || continue
    bench_name="$(basename "$test_subdir")"

    if [[ -n "$BENCH_FILTER" && "$bench_name" != "$BENCH_FILTER" ]]; then
        continue
    fi

    ((COUNT_TOTAL++)) || true
    info "[$COUNT_TOTAL] $bench_name"

    test_file=$(find "$test_subdir" -maxdepth 1 -name "*.test" | head -1)
    if [[ -z "$test_file" ]]; then
        err "$bench_name: 未找到 .test 文件，跳过"
        ((COUNT_SKIP++)) || true
        echo
        continue
    fi

    binary="$BIN_DIR/${bench_name}_${VARIANT}"
    if [[ ! -x "$binary" ]]; then
        err "$bench_name: 可执行文件不存在: $binary，跳过"
        ((COUNT_SKIP++)) || true
        echo
        continue
    fi

    bench_cmd=$(parse_run_cmd "$test_file" "$binary" "$test_subdir") || {
        err "$bench_name: .test 文件中未找到 RUN: 行，跳过"
        ((COUNT_SKIP++)) || true
        echo
        continue
    }

    latest_out_dir="$(find_latest_dataset_dir "$bench_name")"
    if [[ "$OVERWRITE" -eq 0 && -n "$latest_out_dir" ]] && is_valid_dataset_dir "$latest_out_dir"; then
        info "  已有有效输出，跳过: ${latest_out_dir#$PROJECT_ROOT/}"
        ((COUNT_OK++)) || true
        echo
        continue
    fi

    attempt=0
    collect_rc=0
    run_stamp="$(date +%Y%m%d_%H%M%S)"
    out_dir="$OUTPUT_ROOT/${bench_name}_${run_stamp}"

    while [[ "$attempt" -lt "$RETRY_MAX" ]]; do
        ((attempt++)) || true
        if [[ "$attempt" -gt 1 ]]; then
            retry "$bench_name: 第 $attempt 次重试"
            out_dir="$OUTPUT_ROOT/${bench_name}_${run_stamp}_retry${attempt}"
        fi

        info "  BENCH_CMD: $bench_cmd"
        info "  OUTPUT_DIR: ${out_dir#$PROJECT_ROOT/}"

        collect_rc=0
        run_single_bench "$bench_name" "$binary" "$test_file" "$out_dir" &
        CURRENT_CHILD_PID=$!
        wait "$CURRENT_CHILD_PID" || collect_rc=$?
        CURRENT_CHILD_PID=""
        if [[ "$collect_rc" -eq 0 ]]; then
            break
        fi
    done

    if [[ "$collect_rc" -ne 0 ]]; then
        err "$bench_name: $RETRY_MAX 次尝试后仍失败，跳过"
        ((COUNT_SKIP++)) || true
        echo
        continue
    fi

    target_comm="$(basename "$binary")"
    target_comm="${target_comm:0:15}"

    if [[ "$DRYRUN" -eq 0 ]]; then
        # 从 run_metadata.jsonl 读取真实完成轮次（由 collect_single_bcc_testbench.sh 写入）
        _completion_count=0
        if [[ -f "$out_dir/run_metadata.jsonl" ]]; then
            _val=$(grep '"_record_type".*"run_stats"' "$out_dir/run_metadata.jsonl" \
                   | grep -o '"completion_count":[[:space:]]*[0-9]*' \
                   | grep -o '[0-9]*$' | tail -1)
            [[ "$_val" =~ ^[0-9]+$ ]] && _completion_count=$_val
        fi

        local_bench_cmd_escaped="${bench_cmd//\"/\\\"}"
        printf '{"program":"%s","variant":"%s","binary":"%s","test_file":"%s","run_cmd":"%s","target_comm":"%s","output_dir":"%s","window_sec":%s,"duration_sec":%s,"sample_rate":%s,"completion_count":%d}\n' \
            "$bench_name" \
            "$VARIANT" \
            "${binary#$PROJECT_ROOT/}" \
            "${test_file#$PROJECT_ROOT/}" \
            "$local_bench_cmd_escaped" \
            "$target_comm" \
            "${out_dir#$PROJECT_ROOT/}" \
            "$WINDOW_SEC" \
            "$DURATION_SEC" \
            "$SAMPLE_RATE" \
            "$_completion_count" \
            >> "$MANIFEST"
    fi

    pass "$bench_name 完成"
    ((COUNT_OK++)) || true
    echo
done

if [[ "$DRYRUN" -eq 0 && "$DEDUP_AFTER_COLLECT" -eq 1 ]]; then
    info "开始自动去重并重建 manifest"
    "$PYTHON_BIN" "$DEDUP_SCRIPT" \
        --variant "$VARIANT" \
        --project-root "$PROJECT_ROOT" \
        --output-root "$OUTPUT_ROOT" \
        --manifest "$MANIFEST" \
        --bin-dir "$BIN_DIR" \
        --test-dir "$TEST_DIR" \
        --window-sec "$WINDOW_SEC" \
        --duration-sec "$DURATION_SEC" \
        --sample-rate "$SAMPLE_RATE" \
        || {
            echo "Error: 自动去重失败: $DEDUP_SCRIPT" >&2
            exit 1
        }
    echo
fi

bold "======================================================"
printf "  总计: %d  |  成功: %d  |  跳过: %d\n" \
    "$COUNT_TOTAL" "$COUNT_OK" "$COUNT_SKIP"
echo "  清单: $MANIFEST"
bold "======================================================"