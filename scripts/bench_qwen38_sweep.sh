#!/bin/bash
# Sweep the three Qwen3.8-27B launchers and capture a .bench for each.
#
# Each launcher runs `vllm serve` in the foreground, so this backgrounds it,
# waits for :8000 to answer, runs the bare `./llm_bench.py` (no args, matching
# how the rest of the repo's .bench files were taken), then tears the server
# down before the next model.
#
# This used to call ./api_bench.py, which now lives in archive/. llm_bench.py is
# NOT a drop-in for the .bench format. Bare invocation is now 12 runs / 1 warmup
# / --max-tokens 1024, against api_bench's 5 / 1 / 4096, and the summary is
# pooled throughput plus percentile rows instead of a mean/StdDev block. The
# three Qwen3.8 .bench files committed in 8ccfbae and 6d307ef came from
# api_bench and will not match anything produced from here on -- re-run all
# three together if you want a comparable set.
#
# The box's normal resident service (gemma-4-31B-it-INT8) is stopped first and
# restarted at the end, whether the sweep succeeds or not.
set -u

REPO=/src/llm-bench
SERVICE=gemma-4-31B-it-INT8.service
LOGDIR=/tmp/qwen38-sweep-$(date +%Y%m%d-%H%M%S)
READY_TIMEOUT=2400   # 30 min: the heretic BF16 build is ~55.6 GiB of weights
# Models to sweep: all three by default, or just the ones named on the argv.
# Naming a subset is how you resume a sweep that died partway without
# re-running the models that already produced a good .bench.
ALL_MODELS=(
    Qwen3.8-27B-Uncensored-INT8
    Qwen3.8-27B-Uncensored-INT8-mtp
    Qwen3.8-27B-heretic-ara
)
if (( $# )); then MODELS=("$@"); else MODELS=("${ALL_MODELS[@]}"); fi

mkdir -p "$LOGDIR"
cd "$REPO" || exit 1
echo "sweep: logs in $LOGDIR"

gpu_used_total() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        | awk '{s+=$1} END {print s+0}'
}

wait_gpu_clear() {
    local deadline=$((SECONDS + 300))
    while (( SECONDS < deadline )); do
        (( $(gpu_used_total) < 2000 )) && return 0
        sleep 5
    done
    echo "  WARN: GPUs still hold $(gpu_used_total) MiB after 300s"
    return 1
}

teardown_server() {
    pkill -f 'vllm serve' 2>/dev/null
    local deadline=$((SECONDS + 180))
    while (( SECONDS < deadline )); do
        pgrep -f 'vllm serve' >/dev/null || break
        sleep 3
    done
    pgrep -f 'vllm serve' >/dev/null && pkill -9 -f 'vllm serve' 2>/dev/null
    sleep 5
    wait_gpu_clear
}

restore_service() {
    echo "=== restoring $SERVICE ==="
    sudo systemctl start "$SERVICE" && echo "  started" || echo "  FAILED to start"
}
trap 'echo "sweep interrupted"; teardown_server; restore_service; exit 130' INT TERM

# --- stop the resident model -------------------------------------------------
echo "=== stopping $SERVICE ==="
sudo systemctl stop "$SERVICE" || { echo "FATAL: could not stop $SERVICE"; exit 1; }
teardown_server
echo "  GPU now at $(gpu_used_total) MiB"

# --- sweep -------------------------------------------------------------------
FAILED=()
for m in "${MODELS[@]}"; do
    echo "=== $m ==="
    if [[ ! -x "./$m" ]]; then
        echo "  SKIP: ./$m not executable"; FAILED+=("$m:not-executable"); continue
    fi

    nohup "./$m" > "$LOGDIR/$m.server.log" 2>&1 &
    srv=$!
    echo "  server pid $srv, waiting for :8000 (timeout ${READY_TIMEOUT}s)"

    ready=0
    deadline=$((SECONDS + READY_TIMEOUT))
    while (( SECONDS < deadline )); do
        if ! kill -0 "$srv" 2>/dev/null; then
            echo "  FATAL: server exited during load; tail of log:"
            tail -25 "$LOGDIR/$m.server.log" | sed 's/^/    /'
            break
        fi
        if [[ "$(curl -s -o /dev/null -w '%{http_code}' \
                 http://localhost:8000/v1/models 2>/dev/null)" == "200" ]]; then
            ready=1; break
        fi
        sleep 10
    done

    if (( ready )); then
        echo "  ready after ${SECONDS}s-ish, benching -> $m.bench"
        ./llm_bench.py > "$REPO/$m.bench" 2>&1
        rc=$?
        (( rc == 0 )) || { echo "  llm_bench exit $rc"; FAILED+=("$m:bench-rc-$rc"); }
        grep -c 'Running: 1 reqs' "$LOGDIR/$m.server.log" 2>/dev/null \
            | sed 's/^/  taint check, single-stream log samples: /'
        grep -cE 'Running: ([02-9]|[1-9][0-9]+) reqs' "$LOGDIR/$m.server.log" 2>/dev/null \
            | sed 's/^/  taint check, NON-single-stream samples: /'
        tail -12 "$REPO/$m.bench" | sed 's/^/    /'
    else
        echo "  FAILED: never became ready"
        FAILED+=("$m:never-ready")
        rm -f "$REPO/$m.bench"
    fi

    teardown_server
done

# --- restore -----------------------------------------------------------------
restore_service

echo "=== summary ==="
for m in "${MODELS[@]}"; do
    if [[ -s "$REPO/$m.bench" ]]; then
        printf '  %-34s %s\n' "$m" \
            "$(grep -A2 'Tokens/sec:' "$REPO/$m.bench" | awk '/Mean:/{print $2" t/s"}')"
    else
        printf '  %-34s NO BENCH\n' "$m"
    fi
done
(( ${#FAILED[@]} )) && { echo "  failures: ${FAILED[*]}"; echo "SWEEP-COMPLETE rc=1"; exit 1; }
echo "  all requested models benched"
echo "SWEEP-COMPLETE rc=0"
