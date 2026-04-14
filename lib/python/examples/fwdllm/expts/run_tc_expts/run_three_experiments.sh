#!/usr/bin/env bash
# Sequentially run the three dynamic-K/C comparison experiments:
#   1) baseline      : static K=5, C=15                (aggregator_async_base.json)
#   2) maxiter       : static K=5 + iter-cap bypass    (aggregator_async_maxiter.json)
#   3) adaptive_k    : adaptive K, fixed C             (aggregator_async_dynk.json)
#
# All K, C, N, learning_rate, max_iterations_per_data_id, and policy values
# are defined in the aggregator JSON configs above — NOT on the command line.
# Between each run we kill leftover trainer/aggregator processes and remove
# temp configs. Each run writes its own log files tagged with the experiment
# name so the three runs can be compared post-hoc.
#
# Usage:
#   ./run_three_experiments.sh <TOTAL_CLIENT_NUM> <LOG_LEVEL>
#
# Example:
#   ./run_three_experiments.sh 100 INFO
#
# Positional args:
#   $1 TOTAL_CLIENT_NUM   Number of trainer_X.json processes to spawn (default 100)
#   $2 LOG_LEVEL          Log level for fl_main.py (default INFO)

set -u

TOTAL_CLIENT_NUM=${1:-100}
LOG_LEVEL=${2:-INFO}
ENABLE_WATCHDOG="true"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_text_classification.sh"

if [ ! -x "$RUNNER" ]; then
    chmod +x "$RUNNER" 2>/dev/null || true
fi

# Run ordering and config mapping
declare -a RUN_TAGS=("baseline" "maxiter" "adaptive_k")
declare -a RUN_CFGS=(
    "aggregator_async_base.json"
    "aggregator_async_maxiter.json"
    "aggregator_async_dynk.json"
)

CURRENT_CHILD_PID=""

kill_stale_processes() {
    echo "[orchestrator] Killing stale fl_main.py processes..."
    pkill -f "${FWDLLM_USER}.*fl_main.py" 2>/dev/null || true
    # Also clean up anything else that might linger (accuracy-monitor watchdogs)
    sleep 2
    pkill -9 -f "${FWDLLM_USER}.*fl_main.py" 2>/dev/null || true
}

cleanup_temp_dirs() {
    echo "[orchestrator] Removing stray tmp_expanded_configs_* directories..."
    find "$(cd "$SCRIPT_DIR/../../../../.." && pwd)" -maxdepth 2 \
         -type d -name "tmp_expanded_configs_*" \
         -exec rm -rf {} + 2>/dev/null || true
}

on_exit() {
    echo "[orchestrator] Caught EXIT/INT/TERM — cleaning up..."
    if [ -n "$CURRENT_CHILD_PID" ] && kill -0 "$CURRENT_CHILD_PID" 2>/dev/null; then
        echo "[orchestrator] Killing current run (pid=$CURRENT_CHILD_PID)..."
        kill -TERM "$CURRENT_CHILD_PID" 2>/dev/null || true
        sleep 5
        kill -KILL "$CURRENT_CHILD_PID" 2>/dev/null || true
    fi
    kill_stale_processes
    cleanup_temp_dirs
    echo "[orchestrator] Cleanup complete."
}
trap on_exit EXIT INT TERM

if [ -z "${FWDLLM_USER:-}" ]; then
    echo "[orchestrator] ERROR: FWDLLM_USER is not set. Export it before running." >&2
    exit 1
fi

echo "[orchestrator] Starting 3-experiment sweep."
echo "[orchestrator]   total_client_num     = $TOTAL_CLIENT_NUM"
echo "[orchestrator]   LOG_LEVEL            = $LOG_LEVEL"
echo "[orchestrator]   (All other hyperparameters come from the aggregator JSONs.)"

SWEEP_START=$(date +%s)

for i in "${!RUN_TAGS[@]}"; do
    TAG="${RUN_TAGS[$i]}"
    CFG="${RUN_CFGS[$i]}"

    echo ""
    echo "=========================================================================="
    echo "[orchestrator] Run $((i+1))/3 | tag=${TAG} | config=${CFG}"
    echo "=========================================================================="

    # Ensure a clean slate before starting this run
    kill_stale_processes
    cleanup_temp_dirs
    sleep 10

    RUN_START=$(date +%s)

    bash "$RUNNER" \
        "$TOTAL_CLIENT_NUM" \
        "$LOG_LEVEL" \
        "$ENABLE_WATCHDOG" \
        "$CFG" \
        "$TAG" &
    CURRENT_CHILD_PID=$!

    echo "[orchestrator] Spawned run ${TAG} (pid=$CURRENT_CHILD_PID). Waiting..."
    wait "$CURRENT_CHILD_PID"
    RC=$?
    CURRENT_CHILD_PID=""

    RUN_END=$(date +%s)
    ELAPSED=$(( RUN_END - RUN_START ))
    printf "[orchestrator] Run %s finished: rc=%d, elapsed=%dh%dm%ds\n" \
        "$TAG" "$RC" \
        $(( ELAPSED / 3600 )) $(( (ELAPSED % 3600) / 60 )) $(( ELAPSED % 60 ))

    # Brief pause so logs flush before the next run starts
    kill_stale_processes
    cleanup_temp_dirs
    sleep 15
done

SWEEP_END=$(date +%s)
TOTAL_ELAPSED=$(( SWEEP_END - SWEEP_START ))
printf "[orchestrator] Sweep complete. Total elapsed: %dh%dm%ds\n" \
    $(( TOTAL_ELAPSED / 3600 )) $(( (TOTAL_ELAPSED % 3600) / 60 )) $(( TOTAL_ELAPSED % 60 ))
