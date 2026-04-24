#!/usr/bin/env bash
# Run the three dynamic-K/C comparison experiments in PARALLEL on a single machine.
#
# Each run gets:
#   - A unique run_id (datetime + tag) so MQTT topics and client IDs never collide
#   - Its own subset of GPUs (no GPU shared between runs)
#   - Permanent logs at <script_dir>/logs/<run_id>/  (preserved after cleanup)
#   - Temp JSON configs at /tmp/fl_run_<run_id>/     (auto-deleted on exit)
#
# Usage:
#   ./run_three_parallel.sh <TOTAL_CLIENT_NUM> <LOG_LEVEL> \
#       [GPUS_RUN1] [GPUS_RUN2] [GPUS_RUN3]
#
# Examples:
#   # 8-GPU machine: split evenly across 3 runs
#   ./run_three_parallel.sh 100 INFO  0,1,2  3,4,5  6,7
#
#   # 4-GPU machine: one GPU per run
#   ./run_three_parallel.sh 30 INFO  0  1  2,3
#
#   # Override GPU assignment via env vars
#   GPUS_BASELINE="0,1" GPUS_MAXITER="2,3" GPUS_ADAPTIVE="4,5" \
#       ./run_three_parallel.sh 50 INFO
#
# Stopping:
#   Ctrl+C kills all three launcher processes; each launcher terminates its own
#   aggregator + trainers, closes log files, and removes its temp config dir.
#
# Positional args:
#   $1  TOTAL_CLIENT_NUM   Trainers per run (default 100)
#   $2  LOG_LEVEL          fl_main.py log level (default INFO)
#   $3  GPUS_RUN1          GPU IDs for baseline run    (env: GPUS_BASELINE, default "0,1,2")
#   $4  GPUS_RUN2          GPU IDs for maxiter run     (env: GPUS_MAXITER,  default "3,4,5")
#   $5  GPUS_RUN3          GPU IDs for adaptive_k run  (env: GPUS_ADAPTIVE, default "6,7")

set -uo pipefail

TOTAL_CLIENT_NUM="${1:-100}"
LOG_LEVEL="${2:-INFO}"

# GPU sets: positional args > env vars > defaults
GPUS_BASELINE="${3:-${GPUS_BASELINE:-0,1,2}}"
GPUS_MAXITER="${4:-${GPUS_MAXITER:-3,4,5}}"
GPUS_ADAPTIVE="${5:-${GPUS_ADAPTIVE:-6,7}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$SCRIPT_DIR/launch_single_run.py"

REPO_PATH="${REPO_PATH:-$(cd "$SCRIPT_DIR/../../../../../.." && pwd)}"
export REPO_PATH

if [ -z "${FWDLLM_USER:-}" ]; then
    echo "[parallel] WARNING: FWDLLM_USER is not set."
fi

# Array of the actual python3 launcher PIDs (not sed or subshell PIDs).
declare -a LAUNCHER_PIDS=()
_CLEANING_UP=0

on_exit() {
    if [ "$_CLEANING_UP" = "1" ]; then return; fi
    _CLEANING_UP=1
    trap '' INT TERM
    echo ""
    echo "[parallel] Interrupt received — stopping all runs."
    # Send SIGTERM first; each launcher's signal handler cleans up its subtree.
    for pid in "${LAUNCHER_PIDS[@]:-}"; do
        [ -n "$pid" ] && kill -TERM "$pid" 2>/dev/null || true
    done
    echo "[parallel] Waiting up to 10s for launchers to clean up…"
    sleep 10
    # Force-kill any stragglers.
    for pid in "${LAUNCHER_PIDS[@]:-}"; do
        [ -n "$pid" ] && kill -KILL "$pid" 2>/dev/null || true
    done
    echo "[parallel] All runs stopped."
    exit 130
}
trap on_exit INT TERM

STAGGER_SECS="${STAGGER_SECS:-600}"   # 10-minute default; override with env var

echo "[parallel] Starting 3 staggered parallel runs."
echo "[parallel]   num_trainers  = $TOTAL_CLIENT_NUM"
echo "[parallel]   log_level     = $LOG_LEVEL"
echo "[parallel]   gpus baseline = $GPUS_BASELINE"
echo "[parallel]   gpus maxiter  = $GPUS_MAXITER"
echo "[parallel]   gpus adaptive = $GPUS_ADAPTIVE"
echo "[parallel]   logs dir      = $SCRIPT_DIR/logs/"
echo "[parallel]   stagger delay = ${STAGGER_SECS}s between runs"
echo ""

# Helper: print a countdown so it's obvious the script hasn't stalled.
stagger_sleep() {
    local secs="$1"
    local tag="$2"
    echo "[parallel] Waiting ${secs}s before launching ${tag}…"
    local remaining="$secs"
    while [ "$remaining" -gt 0 ]; do
        printf "\r[parallel]   %3ds remaining…" "$remaining"
        sleep 1
        remaining=$(( remaining - 1 ))
    done
    printf "\r[parallel]   Done waiting. Launching %s now.\n" "$tag"
}

# Launch each run in the background with a staggered start.
# -u flag: unbuffered Python output so progress appears immediately.
# The launcher prefixes every line with [<tag>] itself, so no sed needed.
# Capturing $! right after each background launch gives the python3 PID.

python3 -u "$LAUNCHER" \
    --tag          baseline \
    --agg-json     aggregator_async_base.json \
    --num-trainers "$TOTAL_CLIENT_NUM" \
    --gpus         "$GPUS_BASELINE" \
    --log-level    "$LOG_LEVEL" \
    &
LAUNCHER_PIDS[0]=$!
echo "[parallel] baseline launched (pid=${LAUNCHER_PIDS[0]})."

stagger_sleep "$STAGGER_SECS" "maxiter"

python3 -u "$LAUNCHER" \
    --tag          maxiter \
    --agg-json     aggregator_async_maxiter.json \
    --num-trainers "$TOTAL_CLIENT_NUM" \
    --gpus         "$GPUS_MAXITER" \
    --log-level    "$LOG_LEVEL" \
    &
LAUNCHER_PIDS[1]=$!
echo "[parallel] maxiter launched (pid=${LAUNCHER_PIDS[1]})."

stagger_sleep "$STAGGER_SECS" "adaptive_k"

python3 -u "$LAUNCHER" \
    --tag          adaptive_k \
    --agg-json     aggregator_async_dynk.json \
    --num-trainers "$TOTAL_CLIENT_NUM" \
    --gpus         "$GPUS_ADAPTIVE" \
    --log-level    "$LOG_LEVEL" \
    &
LAUNCHER_PIDS[2]=$!
echo "[parallel] adaptive_k launched (pid=${LAUNCHER_PIDS[2]})."
echo "[parallel] All 3 runs are now active."
echo "[parallel] Waiting for all runs to finish (Ctrl+C to abort all)…"
echo ""

wait "${LAUNCHER_PIDS[@]}"
echo "[parallel] All runs finished."
