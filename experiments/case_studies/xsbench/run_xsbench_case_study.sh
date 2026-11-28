#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CASE_DIR="$ROOT_DIR/experiments/case_studies/xsbench"
OUT_BASE="$ROOT_DIR/results/case_study_xsbench"
SYNC_LIB="$ROOT_DIR/experiments/micro_benchmark/libsync.so"

LOOKUPS="${LOOKUPS:-15000000}"
THREADS="${THREADS:-4}"
WINDOW="${WINDOW:-0.5}"

mkdir -p "$OUT_BASE"

for VAR in O0 O2 O3; do
    BIN="$CASE_DIR/bin/xsbench_${VAR}"
    VAR_OUT="$OUT_BASE/$VAR"
    EBPF_OUT="$VAR_OUT/ebpf"
    PHASE_OUT="$VAR_OUT/phases"
    HOTSPOT_OUT="$VAR_OUT/hotspot"
    ATTR_OUT="$VAR_OUT/attribution"
    mkdir -p "$EBPF_OUT" "$PHASE_OUT" "$HOTSPOT_OUT" "$ATTR_OUT"

    echo "=== [XSBench $VAR] 启动全链路采集 (threads=$THREADS, lookups=$LOOKUPS) ==="
    env LD_PRELOAD="$SYNC_LIB" SYNC_ATTACH=1 "$BIN" -t "$THREADS" -s large -m event -l "$LOOKUPS" \
        >"$VAR_OUT/workload.stdout.log" 2>"$VAR_OUT/workload.stderr.log" &
    WPID=$!

    for _ in $(seq 1 50); do
        st="$(awk '{print $3}' "/proc/$WPID/stat" 2>/dev/null || true)"
        [[ "$st" == "T" ]] && break
        sleep 0.02
    done
    cat "/proc/$WPID/maps" > "$VAR_OUT/proc_maps.txt"

    python3 "$ROOT_DIR/src/loader.py" \
        --pid "$WPID" \
        --window "$WINDOW" \
        --duration 60 \
        --output "$EBPF_OUT" \
        --sample-freq 4000 \
        --pmu-backend bcc \
        --track-children \
        --emit-events \
        --sync-target \
        >"$VAR_OUT/loader.stdout.log" 2>"$VAR_OUT/loader.stderr.log" || true

    wait "$WPID" 2>/dev/null || true

    echo "--- [XSBench $VAR] 1) 阶段自动切分 (phase_segmentation.py) ---"
    python3 "$ROOT_DIR/analysis/phase_segmentation.py" \
        --data "$EBPF_OUT" \
        --output "$PHASE_OUT"

    echo "--- [XSBench $VAR] 2) 热点与 Co-Spike 分析 (hotspot.py) ---"
    python3 "$ROOT_DIR/analysis/hotspot.py" \
        --data "$EBPF_OUT" \
        --output "$HOTSPOT_OUT" \
        --metric llc_store_misses \
        --top 10 || true

    echo "--- [XSBench $VAR] 3) 函数级源码归因 (attribution.py) ---"
    python3 "$ROOT_DIR/analysis/attribution.py" \
        --data "$EBPF_OUT" \
        --pid "$WPID" \
        --maps-file "$VAR_OUT/proc_maps.txt" \
        --metric llc_store_misses \
        --top 10 \
        --output "$ATTR_OUT" || true
done

echo "=== XSBench 全链路案例采集与分析完成: $OUT_BASE ==="
