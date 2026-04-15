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
#   ./run_three_experiments.sh <TOTAL_CLIENT_NUM> <LOG_LEVEL> [RUN_FILTER]
#
# Examples:
#   ./run_three_experiments.sh 100 INFO              # all three runs
#   ./run_three_experiments.sh 30  INFO adaptive_k   # only the adaptive_k run
#   ./run_three_experiments.sh 30  INFO maxiter      # only the maxiter run
#   ./run_three_experiments.sh 30  INFO baseline,maxiter   # baseline + maxiter
#
# Positional args:
#   $1 TOTAL_CLIENT_NUM   Number of trainer_X.json processes to spawn (default 100)
#   $2 LOG_LEVEL          Log level for fl_main.py (default INFO)
#   $3 RUN_FILTER         Optional comma-separated list of run tags to execute.
#                         Valid tags: baseline, maxiter, adaptive_k.
#                         Empty / unset = run all three in order.

set -u

TOTAL_CLIENT_NUM=${1:-100}
LOG_LEVEL=${2:-INFO}
RUN_FILTER=${3:-}
ENABLE_WATCHDOG="true"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_text_classification.sh"
# Repo root is 6 levels above SCRIPT_DIR (see run_text_classification.sh for
# the directory layout). Override REPO_PATH in the env if the script lives
# elsewhere (e.g. via a symlink).
REPO_PATH="${REPO_PATH:-$(cd "$SCRIPT_DIR/../../../../../.." && pwd)}"
export REPO_PATH   # propagate to the runner so both scripts agree

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
_CLEANING_UP=0

# Hard-kill every PyTorch/MQTT/aggregator process still matching our pattern.
# Used both between runs and inside the signal trap.
kill_stale_processes() {
    # SIGKILL directly — no sleep, no grace. Python PyTorch/MQTT processes
    # routinely ignore SIGTERM for 10s+ while they clean up GPU/sockets, so
    # we don't wait for them here.
    pkill -9 -f "${FWDLLM_USER}.*fl_main.py" 2>/dev/null || true
}

cleanup_temp_dirs() {
    # Temp dirs are created at $REPO_PATH/tmp_expanded_configs_*
    find "$REPO_PATH" -maxdepth 1 \
         -type d -name "tmp_expanded_configs_*" \
         -exec rm -rf {} + 2>/dev/null || true
}

on_exit() {
    # Reentrancy guard — repeated Ctrl+C must not relaunch the handler.
    if [ "$_CLEANING_UP" = "1" ]; then
        return
    fi
    _CLEANING_UP=1

    # Ignore further signals while we're tearing things down.
    trap '' INT TERM

    echo ""
    echo "[orchestrator] Signal received — hard-killing the current run and descendants."

    if [ -n "$CURRENT_CHILD_PID" ]; then
        # Negative PID sends to the entire process group. We launched each run
        # via setsid, so $CURRENT_CHILD_PID is also the PGID for that run.
        kill -KILL -- "-$CURRENT_CHILD_PID" 2>/dev/null || true
        # Also signal the lead process directly in case the group is gone.
        kill -KILL -- "$CURRENT_CHILD_PID" 2>/dev/null || true
    fi

    # Sweep any stragglers that slipped out of the group.
    kill_stale_processes
    cleanup_temp_dirs
    echo "[orchestrator] Cleanup complete."
    exit 130
}
trap on_exit INT TERM
# Keep EXIT trap lighter: just temp-dir cleanup if we exit normally.
trap 'cleanup_temp_dirs' EXIT

if [ -z "${FWDLLM_USER:-}" ]; then
    echo "[orchestrator] ERROR: FWDLLM_USER is not set. Export it before running." >&2
    exit 1
fi

# --- Resolve RUN_FILTER into a set of tags to execute ---
declare -A RUN_FILTER_SET=()
if [ -n "$RUN_FILTER" ]; then
    IFS=',' read -ra _filter_arr <<< "$RUN_FILTER"
    for tag in "${_filter_arr[@]}"; do
        # Trim whitespace
        tag="${tag## }"
        tag="${tag%% }"
        # Validate tag
        valid=0
        for known in "${RUN_TAGS[@]}"; do
            if [ "$tag" = "$known" ]; then
                valid=1
                break
            fi
        done
        if [ "$valid" != "1" ]; then
            echo "[orchestrator] ERROR: unknown run tag '$tag'. Valid: ${RUN_TAGS[*]}" >&2
            exit 2
        fi
        RUN_FILTER_SET[$tag]=1
    done
fi

echo "[orchestrator] Starting experiment sweep."
echo "[orchestrator]   total_client_num     = $TOTAL_CLIENT_NUM"
echo "[orchestrator]   LOG_LEVEL            = $LOG_LEVEL"
if [ ${#RUN_FILTER_SET[@]} -gt 0 ]; then
    echo "[orchestrator]   RUN_FILTER           = ${!RUN_FILTER_SET[*]}"
else
    echo "[orchestrator]   RUN_FILTER           = (none — running all ${#RUN_TAGS[@]} configs)"
fi
echo "[orchestrator]   (All other hyperparameters come from the aggregator JSONs.)"

SWEEP_START=$(date +%s)

for i in "${!RUN_TAGS[@]}"; do
    TAG="${RUN_TAGS[$i]}"
    # Apply the optional filter
    if [ ${#RUN_FILTER_SET[@]} -gt 0 ] && [ -z "${RUN_FILTER_SET[$TAG]:-}" ]; then
        echo "[orchestrator] Skipping run ${TAG} (not in RUN_FILTER)."
        continue
    fi
    CFG="${RUN_CFGS[$i]}"

    echo ""
    echo "=========================================================================="
    echo "[orchestrator] Run $((i+1))/3 | tag=${TAG} | config=${CFG}"
    echo "=========================================================================="

    # Ensure a clean slate before starting this run
    kill_stale_processes
    cleanup_temp_dirs
    sleep 5

    RUN_START=$(date +%s)

    # Launch the runner in its own session (== own process group). With
    # `setsid`, $CURRENT_CHILD_PID is also the PGID of every descendant it
    # spawns, so on Ctrl+C we can `kill -KILL -- -$PGID` the whole subtree in
    # one shot.
    setsid bash "$RUNNER" \
        "$TOTAL_CLIENT_NUM" \
        "$LOG_LEVEL" \
        "$ENABLE_WATCHDOG" \
        "$CFG" \
        "$TAG" </dev/null &
    CURRENT_CHILD_PID=$!

    echo "[orchestrator] Spawned run ${TAG} (pid=pgid=$CURRENT_CHILD_PID). Waiting..."
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
    sleep 5
done

SWEEP_END=$(date +%s)
TOTAL_ELAPSED=$(( SWEEP_END - SWEEP_START ))
printf "[orchestrator] Sweep complete. Total elapsed: %dh%dm%ds\n" \
    $(( TOTAL_ELAPSED / 3600 )) $(( (TOTAL_ELAPSED % 3600) / 60 )) $(( TOTAL_ELAPSED % 60 ))
