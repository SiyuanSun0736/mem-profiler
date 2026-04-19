#!/bin/bash
# =============================================================================
# collect_dataset_all_variants.sh — 顺序采集 llvm-test-suite 全部变体
# =============================================================================
#
# 功能：
#   自动扫描 data/llvm_test_suite/bin/ 与 test/ 下的公共 VARIANT 目录，
#   依次调用 collect_dataset_testbench.sh 完成一次整批 BCC 采集。
#
# 用法：
#   sudo bash experiments/llvm_test_suite/collect_dataset_all_variants.sh
#   sudo bash experiments/llvm_test_suite/collect_dataset_all_variants.sh -v "O2 O3"
#   bash experiments/llvm_test_suite/collect_dataset_all_variants.sh -- -n
#   sudo DURATION_SEC=10 bash experiments/llvm_test_suite/collect_dataset_all_variants.sh -- -s aha
#
# 说明：
#   - 不传 -v 时，默认采集 data/llvm_test_suite/bin/<VARIANT> 与
#     data/llvm_test_suite/test/<VARIANT> 同时存在的全部变体。
#   - "--" 之后的参数会原样透传给 collect_dataset_testbench.sh。
#   - 本脚本会为每个变体单独设置 BIN_DIR/TEST_DIR/OUTPUT_ROOT/MANIFEST，
#     避免不同变体之间的输出相互覆盖。
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/data/llvm_test_suite}"
BIN_ROOT="${BIN_ROOT:-$DATASET_ROOT/bin}"
TEST_ROOT="${TEST_ROOT:-$DATASET_ROOT/test}"
BCC_ROOT="${BCC_ROOT:-$DATASET_ROOT/bcc}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$DATASET_ROOT}"
COLLECT_SCRIPT="${COLLECT_SCRIPT:-$SCRIPT_DIR/collect_dataset_testbench.sh}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info() { printf "${CYAN}[INFO]${NC}  %s\n" "$*"; }
pass() { printf "${GREEN}[OK]${NC}    %s\n" "$*"; }
err()  { printf "${RED}[FAIL]${NC}  %s\n" "$*"; }
bold() { printf "${BOLD}%s${NC}\n" "$*"; }

detect_available_variants() {
    local variant_dir variant
    [[ -d "$TEST_ROOT" ]] || return 0

    for variant_dir in "$TEST_ROOT"/*; do
        [[ -d "$variant_dir" ]] || continue
        variant="$(basename "$variant_dir")"
        [[ -d "$BIN_ROOT/$variant" ]] || continue
        printf '%s\n' "$variant"
    done
}

usage() {
    local detected
    detected="$(detect_available_variants | tr '\n' ' ' | sed 's/ $//')"

    cat <<EOF
用法: $0 [-v "O0 O1 O2 O3"] [-- collect_dataset_testbench.sh 的额外参数]

选项:
  -v VARIANTS     指定要采集的变体列表，使用空格分隔
  -h              显示帮助

示例:
  sudo bash $0
  sudo bash $0 -v "O2 O3"
  bash $0 -- -n
    sudo DURATION_SEC=10 bash $0 -- -s aha
EOF

    [[ -n "$detected" ]] && echo "当前检测到的 VARIANT: $detected"
}

VARIANT_SPEC="${VARIANTS:-}"
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--variants)
            [[ $# -ge 2 ]] || { echo "Error: $1 需要参数" >&2; exit 1; }
            VARIANT_SPEC="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            FORWARD_ARGS=("$@")
            break
            ;;
        *)
            echo "Error: 未知选项: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

[[ -f "$COLLECT_SCRIPT" ]] || { echo "Error: collect_dataset_testbench.sh 不存在: $COLLECT_SCRIPT" >&2; exit 1; }

declare -a VARIANT_LIST
if [[ -n "$VARIANT_SPEC" ]]; then
    read -r -a VARIANT_LIST <<< "$VARIANT_SPEC"
else
    mapfile -t VARIANT_LIST < <(detect_available_variants)
fi

[[ "${#VARIANT_LIST[@]}" -gt 0 ]] || {
    echo "Error: 未检测到可采集的 VARIANT，请检查 $BIN_ROOT 与 $TEST_ROOT" >&2
    exit 1
}

bold "======================================================"
bold "     BCC 批量采集：llvm-test-suite 全部变体"
bold "======================================================"
info "COLLECT_SCRIPT: ${COLLECT_SCRIPT#$PROJECT_ROOT/}"
info "DATASET_ROOT:    ${DATASET_ROOT#$PROJECT_ROOT/}"
info "VARIANTS:        ${VARIANT_LIST[*]}"
[[ "${#FORWARD_ARGS[@]}" -gt 0 ]] && info "透传参数:       ${FORWARD_ARGS[*]}"
echo

success_count=0
failed_count=0
CURRENT_CHILD_PID=""
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
}

trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM
trap cleanup EXIT

for variant in "${VARIANT_LIST[@]}"; do
    bin_dir="$BIN_ROOT/$variant"
    test_dir="$TEST_ROOT/$variant"
    output_root="$BCC_ROOT/$variant"
    manifest="$MANIFEST_ROOT/manifest_bcc_${variant}.jsonl"

    bold "───────────────── VARIANT $variant ─────────────────"

    if [[ ! -d "$bin_dir" ]]; then
        err "$variant: BIN_DIR 不存在: $bin_dir"
        ((failed_count++)) || true
        echo
        continue
    fi

    if [[ ! -d "$test_dir" ]]; then
        err "$variant: TEST_DIR 不存在: $test_dir"
        ((failed_count++)) || true
        echo
        continue
    fi

    collect_rc=0
    VARIANT="$variant" \
        BIN_DIR="$bin_dir" \
        TEST_DIR="$test_dir" \
        OUTPUT_ROOT="$output_root" \
        MANIFEST="$manifest" \
        bash "$COLLECT_SCRIPT" "${FORWARD_ARGS[@]}" &
    CURRENT_CHILD_PID=$!
    wait "$CURRENT_CHILD_PID" || collect_rc=$?
    CURRENT_CHILD_PID=""

    if [[ "$collect_rc" -eq 0 ]]; then
        pass "$variant 完成"
        ((success_count++)) || true
    else
        err "$variant 采集失败"
        ((failed_count++)) || true
    fi

    echo
done

bold "======================================================"
printf "  成功: %d  |  失败: %d\n" "$success_count" "$failed_count"
bold "======================================================"

if [[ "$failed_count" -gt 0 ]]; then
    exit 1
fi