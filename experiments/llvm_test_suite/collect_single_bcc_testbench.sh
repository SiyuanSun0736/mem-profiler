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
  -r SAMPLE_RATE   perf sample rate，默认 100
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

validate_collected_data() {
    local out_dir="$1"
    local target_comm="$2"
    local expect_events="$3"

    "$PYTHON_BIN" - "$PROJECT_ROOT" "$out_dir" "$target_comm" "$expect_events" <<'PYEOF'
import json
import pathlib
import sys

project_root = pathlib.Path(sys.argv[1])
output_dir = pathlib.Path(sys.argv[2])
target_comm = sys.argv[3]
expect_events = sys.argv[4] == "1"

activity_metrics = (
    "samples",
    "cycles",
    "instructions",
    "llc_loads",
    "llc_load_misses",
    "llc_stores",
    "llc_store_misses",
    "dtlb_loads",
    "dtlb_load_misses",
    "dtlb_stores",
    "dtlb_store_misses",
    "dtlb_misses",
    "itlb_loads",
    "itlb_load_misses",
    "minor_faults",
    "major_faults",
    "lbr_samples",
    "lbr_entries",
)


def fail(message: str) -> None:
    print(f"[error] 数据校验失败: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_jsonl(path: pathlib.Path) -> list[dict]:
    records: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    fail(f"{path.name}:{line_number} 不是合法 JSON: {exc.msg}")
                if not isinstance(record, dict):
                    fail(f"{path.name}:{line_number} 不是 JSON object")
                records.append(record)
    except OSError as exc:
        fail(f"无法读取 {path.name}: {exc}")
    return records


metadata_path = output_dir / "run_metadata.jsonl"
window_metrics_path = output_dir / "window_metrics.jsonl"
events_path = output_dir / "events.jsonl"

if not metadata_path.is_file():
    fail(f"缺少 {metadata_path.name}")
if not window_metrics_path.is_file():
    fail(f"缺少 {window_metrics_path.name}")
if expect_events and not events_path.is_file():
    fail("启用了 --emit-events/--lbr，但未生成 events.jsonl")

metadata_rows = load_jsonl(metadata_path)
window_rows = load_jsonl(window_metrics_path)

if not metadata_rows:
    fail("run_metadata.jsonl 为空")
if not window_rows:
    fail("window_metrics.jsonl 为空")

start_rows = [record for record in metadata_rows if record.get("_record_type") != "run_end"]
end_rows = [record for record in metadata_rows if record.get("_record_type") == "run_end"]

if len(start_rows) != 1:
    fail(f"run_metadata.jsonl 应包含 1 条 start record，实际为 {len(start_rows)}")
if len(end_rows) != 1:
    fail(f"run_metadata.jsonl 应包含 1 条 run_end record，实际为 {len(end_rows)}")

start_record = start_rows[0]
end_record = end_rows[0]
run_id = start_record.get("run_id")

if not run_id:
    fail("start record 缺少 run_id")
if end_record.get("run_id") != run_id:
    fail("start/end record 的 run_id 不一致")
if start_record.get("schema_version") != "1.0" or end_record.get("schema_version") != "1.0":
    fail("run_metadata 的 schema_version 非 1.0")
if start_record.get("end_ts_iso") is not None:
    fail("start record 的 end_ts_iso 应为 null")
if not start_record.get("start_ts_iso"):
    fail("start record 缺少 start_ts_iso")
if not end_record.get("end_ts_iso"):
    fail("run_end record 缺少 end_ts_iso")
if target_comm and start_record.get("target_comm") != target_comm:
    fail(
        "run_metadata.target_comm="
        f"{start_record.get('target_comm')!r} 与目标 comm={target_comm!r} 不一致"
    )

expected_entity_scope = "tid" if start_record.get("aggregation_scope") == "per_tid" else "pid"

try:
    import jsonschema
except ImportError:
    jsonschema = None

if jsonschema is not None:
    run_schema = json.loads(
        (project_root / "export/schema/run_metadata.schema.json").read_text(encoding="utf-8")
    )
    window_schema = json.loads(
        (project_root / "export/schema/window_metrics.schema.json").read_text(encoding="utf-8")
    )
    run_validator = jsonschema.Draft202012Validator(run_schema)
    window_validator = jsonschema.Draft202012Validator(window_schema)

    run_errors = sorted(run_validator.iter_errors(start_record), key=lambda error: list(error.path))
    if run_errors:
        fail(f"run_metadata start record 不符合 schema: {run_errors[0].message}")

    for row_index, row in enumerate(window_rows, 1):
        window_errors = sorted(
            window_validator.iter_errors(row),
            key=lambda error: list(error.path),
        )
        if window_errors:
            fail(
                "window_metrics.jsonl 第 "
                f"{row_index} 条记录不符合 schema: {window_errors[0].message}"
            )

for row_index, row in enumerate(window_rows, 1):
    if row.get("run_id") != run_id:
        fail(f"window_metrics.jsonl 第 {row_index} 条记录的 run_id 与 run_metadata 不一致")
    if row.get("entity_scope") != expected_entity_scope:
        fail(
            "window_metrics.jsonl 第 "
            f"{row_index} 条记录的 entity_scope={row.get('entity_scope')!r}，"
            f"期望 {expected_entity_scope!r}"
        )

    start_ns = row.get("start_ns")
    end_ns = row.get("end_ns")
    if not isinstance(start_ns, int) or not isinstance(end_ns, int) or end_ns <= start_ns:
        fail(f"window_metrics.jsonl 第 {row_index} 条记录的时间窗范围无效")
    if expected_entity_scope == "tid" and "tid" not in row:
        fail(f"window_metrics.jsonl 第 {row_index} 条 per_tid 记录缺少 tid 字段")

target_rows = [row for row in window_rows if row.get("comm") == target_comm] if target_comm else window_rows
if not target_rows:
    fail(f"window_metrics.jsonl 中没有 comm={target_comm!r} 的记录")

active_rows = [
    row for row in target_rows
    if any(int(row.get(metric_name, 0)) > 0 for metric_name in activity_metrics)
]
if not active_rows:
    fail("目标记录全部为 0，未采到有效样本")

window_ids = sorted({row["window_id"] for row in target_rows if isinstance(row.get("window_id"), int)})
if not window_ids:
    fail("目标记录缺少有效 window_id")

summary_totals = {
    "samples": sum(int(row.get("samples", 0)) for row in active_rows),
    "cycles": sum(int(row.get("cycles", 0)) for row in active_rows),
    "instructions": sum(int(row.get("instructions", 0)) for row in active_rows),
    "minor_faults": sum(int(row.get("minor_faults", 0)) for row in active_rows),
    "major_faults": sum(int(row.get("major_faults", 0)) for row in active_rows),
    "llc_load_misses": sum(int(row.get("llc_load_misses", 0)) for row in active_rows),
    "dtlb_misses": sum(int(row.get("dtlb_misses", 0)) for row in active_rows),
    "itlb_load_misses": sum(int(row.get("itlb_load_misses", 0)) for row in active_rows),
}
active_pids = sorted({row.get("pid") for row in active_rows if isinstance(row.get("pid"), int)})

print(
    "[info] 数据校验通过:"
    f" run_id={run_id}"
    f" target_rows={len(target_rows)}"
    f" active_rows={len(active_rows)}"
    f" windows={len(window_ids)}"
    f" active_pids={len(active_pids)}"
)
print(
    "[info] 数据摘要:"
    f" samples={summary_totals['samples']}"
    f" cycles={summary_totals['cycles']}"
    f" instructions={summary_totals['instructions']}"
    f" llc_load_misses={summary_totals['llc_load_misses']}"
    f" dtlb_misses={summary_totals['dtlb_misses']}"
    f" itlb_load_misses={summary_totals['itlb_load_misses']}"
    f" minor_faults={summary_totals['minor_faults']}"
    f" major_faults={summary_totals['major_faults']}"
)
PYEOF
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
loader_rc=0
"${LOADER_CMD[@]}" &
LOADER_PID=$!
wait "$LOADER_PID" || loader_rc=$?
LOADER_PID=""
cleanup

if [[ "$loader_rc" -ne 0 ]]; then
    exit "$loader_rc"
fi

echo "[info] 采集完成，开始校验输出数据"
validate_collected_data "$OUT_DIR" "$TARGET_COMM" "$EMIT_EVENTS"
echo "[info] 采集完成且数据有效，输出目录: $OUT_DIR"