#!/usr/bin/env bash
# Ensure that you have set the FWDLLM_USER environment variable before
# running this script. Example:
#   conda env config vars set FWDLLM_USER=<your-folder-name>
#
# This script only runs the FedFwd pipeline. All training hyperparameters
# (K, C, learning_rate, data_loader_num_workers, partition_method, etc.) are
# read from the aggregator JSON config, NOT from CLI args — CLI only
# carries things the wrapper itself needs (logging, watchdog, which
# config to expand).
#
# Usage:
#   ./run_text_classification.sh <TOTAL_CLIENT_NUM> <LOG_LEVEL> [ENABLE_WATCHDOG] [AGG_JSON_NAME] [RUN_TAG]
#
# Positional args:
#   $1 TOTAL_CLIENT_NUM   Number of trainer_X.json processes to spawn (required)
#   $2 LOG_LEVEL          Log level passed to fl_main.py (required, e.g. INFO)
#   $3 ENABLE_WATCHDOG    "true" to enable error/accuracy watchdog (default: false)
#   $4 AGG_JSON_NAME      Aggregator config filename under json_scripts/  (default: aggregator.json)
#   $5 RUN_TAG            Optional tag appended to log filenames (used by the multi-run orchestrator)

if [ $# -lt 2 ]; then
    echo "Usage: $0 <TOTAL_CLIENT_NUM> <LOG_LEVEL> [ENABLE_WATCHDOG] [AGG_JSON_NAME] [RUN_TAG]" >&2
    exit 64
fi

total_client_num=$1
LOG_LEVEL=$2
ENABLE_WATCHDOG=${3:-false}
AGG_JSON_NAME=${4:-aggregator.json}
RUN_TAG=${5:-}

# --- Accuracy monitoring configuration ---
ACC_THRESHOLD=85          # Accuracy percentage (0-100) to monitor for
ACC_CONSEC_LIMIT=10       # Number of consecutive rounds above threshold before stopping the run

pkill -f "$FWDLLM_USER.*fl_main.py"
if [ $? -eq 0 ]; then
    echo "Successfully killed some processes."
else
    echo "No matching process found or kill failed."
fi
sleep 10  # Wait for the system to stabilize
nvidia-smi

C_LR=0.01
S_LR=0.1
ROUND=10
WORKER_NUM=1
model_type=distilbert
model_name=distilbert-base-uncased
# model_type=bert model_name=bert-base-uncased model_type=bert
# model_name=bert-large-uncased model_type=albert
# model_name=albert-base-v2 model_type=roberta-large
# model_name=roberta-large model_type=deberta
# model_name=microsoft/deberta-xlarge
train_batch_size=8
DATA_NAME=agnews
# fold_name=${model_type}_${DATA_NAME}

if [ $model_type = "distilbert" ];then
  peft_method=adapter
else
  peft_method=bitfit
fi

PARTITION_METHOD="niid_label_clients=100_alpha=1" # this is set in aggregator.json, this will be overwritten
if [ $DATA_NAME = "agnews" ];then
  max_seq_length=64  # this is set in aggregator.json, this will be overwritten
  frequency_of_the_test=1
elif [ $DATA_NAME = "20news" ];then
  max_seq_length=256
  frequency_of_the_test=1
elif [ $DATA_NAME = "yelp-p" ];then
  max_seq_length=256
  frequency_of_the_test=1
elif [ $DATA_NAME = "yahoo" ];then
  max_seq_length=256
  frequency_of_the_test=5
  PARTITION_METHOD="uniform_client_10000"
else
  max_seq_length=256
  frequency_of_the_test=1
fi


LOG_FILE="fedavg_transformer_tc.log"
CI=0

# --- Repo path auto-detection ---
# Derive REPO_PATH from this script's location. The runner lives at
#   <REPO>/lib/python/examples/fwdllm/expts/run_tc_expts/run_text_classification.sh
# so the repo root is six levels above SCRIPT_DIR. Set REPO_PATH in the
# environment to override (e.g. if this script is symlinked from elsewhere).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="${REPO_PATH:-$(cd "$SCRIPT_DIR/../../../../../.." && pwd)}"
echo "Using REPO_PATH=$REPO_PATH"

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

# --- FedFwd pipeline (only supported algorithm in this script) ---
LOG_DIR=./log/new
  # if [ -d "$LOG_DIR" ]; then rm -rf "$LOG_DIR" fi
  mkdir -p "$LOG_DIR"

  # Generate timestamp once
  RUN_TIMESTAMP=$(date +%d_%m_%H_%M)
  LOG_TAG_SUFFIX=""
  if [ -n "$RUN_TAG" ]; then
    LOG_TAG_SUFFIX="_${RUN_TAG}"
  fi

  # --- Expand the aggregator config early so we can parse its hyperparameters
  # into the log filename. This is the same expansion the aggregator process
  # will later read. ---
  AGG_SRC="$REPO_PATH/lib/python/examples/fwdllm/expts/run_tc_expts/json_scripts/${AGG_JSON_NAME}"
  if [ ! -f "$AGG_SRC" ]; then
    echo "ERROR: Aggregator config not found: $AGG_SRC"
    exit 2
  fi
  EXPANDED_TMP_DIR="${REPO_PATH}/tmp_expanded_configs_${RUN_TIMESTAMP}"
  mkdir -p "$EXPANDED_TMP_DIR"
  AGG_EXPANDED="$EXPANDED_TMP_DIR/aggregator_expanded.json"
  envsubst < "$AGG_SRC" > "$AGG_EXPANDED"
  echo "Wrote expanded aggregator config (source=${AGG_JSON_NAME}): $AGG_EXPANDED"

  # --- Parse key hyperparameters for the log filename ---
  # Produces eight space-separated tokens:
  #   K  C  N  MAXITER  POLICY  PARTSLUG  LR  DATA_LOADER_WORKERS
  # Any value absent from the JSON is replaced with "NA". Partition slug is
  # filesystem-safe (no '=' or ',').
  read CFG_K CFG_C CFG_N CFG_MAXITER CFG_POLICY CFG_PART CFG_LR CFG_DL_WORKERS < <(python -c "
import json
import re
with open('$AGG_EXPANDED') as f:
    c = json.load(f)
hp = c.get('hyperparameters', {}) or {}
sel = (c.get('selector', {}) or {}).get('kwargs', {}) or {}
K = sel.get('aggGoal', hp.get('aggGoal', 'NA'))
C = sel.get('c', 'NA')
N = sel.get('minInitialTrainers', 'NA')
mx = hp.get('max_iterations_per_data_id', 'none')
dynkc = sel.get('dynamic_kc', {}) or {}
policy = dynkc.get('policy', 'static') if dynkc.get('enabled', False) else 'static'
part = hp.get('partition_method', 'NA')
# Slug must preserve the alpha qualifier so heterogeneous partitions at
# different alpha values are distinguishable in log filenames.
# 'uniform'                        -> 'uniform'
# 'niid_label_clients=100_alpha=1' -> 'niidlabelclients100alpha1'
part_slug = re.sub(r'[^A-Za-z0-9]', '', str(part))[:32] or 'NA'
lr = hp.get('learning_rate', hp.get('lr', 'NA'))
dl_workers = hp.get('data_loader_num_workers', 'NA')
print(K, C, N, mx, policy, part_slug, lr, dl_workers)
")

  LOG_SUFFIX="fedFwd_${model_type}_${DATA_NAME}_lr${CFG_LR}_N${CFG_N}_K${CFG_K}_C${CFG_C}_maxIter${CFG_MAXITER}_pol-${CFG_POLICY}_part-${CFG_PART}_dlw${CFG_DL_WORKERS}${LOG_TAG_SUFFIX}_${RUN_TIMESTAMP}"
  AGG_LOG_FILE=$(readlink -f "$LOG_DIR/test_agg_${LOG_SUFFIX}.log")
  TRAINER_LOG_FILE=$(readlink -f "$LOG_DIR/test_trainer_${LOG_SUFFIX}.log")
  PARENT_PID=$$

  ACC_MONITOR_FILE=$(readlink -f "$LOG_DIR/accuracy_monitor_${LOG_SUFFIX}.log")  # overwritten each tick

  echo "Run config (from ${AGG_JSON_NAME}):  K=${CFG_K}  C=${CFG_C}  N=${CFG_N}  maxIter=${CFG_MAXITER}  policy=${CFG_POLICY}  part=${CFG_PART}  lr=${CFG_LR}  data_loader_workers=${CFG_DL_WORKERS}"
  SCRIPT_START_TIME=$(date +%s)
  _acc_consec_count=0
  _acc_last_seen_line=""  # dedup: only count each new eval result once
  _acc_last_grep_offset=0   # byte offset: attempt to resume grep from last occurrence

  # Function to check logs for errors and kill all processes if found
  check_errors() {
    # Find the first file that contains a real error (ignoring known
    # false positives)
    FOUND_ERR_FILE=$(grep -E -H "Error|Exception|Traceback" "$AGG_LOG_FILE" "$TRAINER_LOG_FILE" | \
                     grep -vE "Error Distribution Analysis|log_error_distribution" | \
                     head -n 1 | cut -d: -f1)     # Extracts the name that the first grep matches

    if [ ! -z "$FOUND_ERR_FILE" ]; then
      echo "--------------------------------------------------------"
      echo "ERROR DETECTED in $FOUND_ERR_FILE! Shutting down..."
      echo "--------------------------------------------------------"
      # Show the first few errors, excluding known false positives
      ERR_MSG=$(grep -E "Error|Exception|Traceback" "$FOUND_ERR_FILE" | \
                grep -vE "Error Distribution Analysis|log_error_distribution" | head -n 20)
      
      # Append termination message to both logs
      TERMINATION_MSG="Killed spawned processes due to error in $FOUND_ERR_FILE\n\n$ERR_MSG"
      echo -e "$TERMINATION_MSG" >> "$AGG_LOG_FILE"
      echo -e "$TERMINATION_MSG" >> "$TRAINER_LOG_FILE"
      echo -e "$TERMINATION_MSG"

      # Trigger the graceful cleanup and exit We kill the parent
      # process with TERM; the trap will handle the rest.
      kill -TERM $PARENT_PID
      exit 1
    fi
  }

  # Function to monitor model accuracy from the aggregator log. Tracks
  # consecutive rounds above ACC_THRESHOLD and triggers shutdown when
  # ACC_CONSEC_LIMIT is reached.
  check_accuracy() {
    [ -f "$AGG_LOG_FILE" ] || return

    # --- Smart tail-first grep --- The log is append-only and can be
    # very large (up to ~8 GB). We first try scanning only the last
    # 200 KB for the accuracy line, which covers many rounds without
    # reading the full file.  If nothing is found there (e.g. the file
    # is brand-new) we fall back to a full scan.
    ACC_LINE=$(tail -c 204800 "$AGG_LOG_FILE" 2>/dev/null | \
               grep -oE "'acc': [0-9]+\.?[0-9]*" | tail -n 1)
    if [ -z "$ACC_LINE" ]; then
      ACC_LINE=$(grep -oE "'acc': [0-9]+\.?[0-9]*" "$AGG_LOG_FILE" 2>/dev/null | tail -n 1)
    fi

    if [ -z "$ACC_LINE" ]; then
      # No eval result yet — write a waiting status so the file always
      # exists
      printf "Waiting for first eval result... | Runtime: %s\n" \
        "$(printf '%dh %dm %ds' $(( ($(date +%s) - SCRIPT_START_TIME)/3600 )) $(( (($(date +%s) - SCRIPT_START_TIME)%3600)/60 )) $(( ($(date +%s) - SCRIPT_START_TIME)%60 )))" \
        >> "$ACC_MONITOR_FILE"
      return
    fi

    # Dedup guard: if this is the same log line we saw last tick, the
    # training round hasn't produced a new eval result yet — skip
    # counter update but still refresh the runtime in the monitor
    # file.
    if [ "$ACC_LINE" = "$_acc_last_seen_line" ]; then
      NOW=$(date +%s)
      ELAPSED=$(( NOW - SCRIPT_START_TIME ))
      ELAPSED_FMT=$(printf '%dh %dm %ds' $(( ELAPSED/3600 )) $(( (ELAPSED%3600)/60 )) $(( ELAPSED%60 )))
      {
        echo "Acc     : ${ACC_PCT_LAST}% |  Threshold: ${ACC_THRESHOLD}%  |  Consecutive above: ${_acc_consec_count} / ${ACC_CONSEC_LIMIT} |  Runtime: ${ELAPSED_FMT} | No new eval"
      } >> "$ACC_MONITOR_FILE"
      return
    fi
    _acc_last_seen_line="$ACC_LINE"

    # Extract the raw fraction (e.g. 0.7505263...) and convert to
    # percentage
    ACC_RAW=$(echo "$ACC_LINE" | grep -oE "[0-9]+\.?[0-9]*$")
    ACC_PCT=$(awk "BEGIN { printf \"%.2f\", $ACC_RAW * 100 }")

    # Write status to the dedicated monitor file (overwrite, not
    # append). This keeps stdout clean — no repeated lines after a
    # 12-hour run.
    NOW=$(date +%s)
    ELAPSED=$(( NOW - SCRIPT_START_TIME ))
    ELAPSED_FMT=$(printf '%dh %dm %ds' $(( ELAPSED/3600 )) $(( (ELAPSED%3600)/60 )) $(( ELAPSED%60 )))
    ACC_PCT_LAST="$ACC_PCT"   # remember for dedup ticks
    {
      echo "Acc     : ${ACC_PCT}%  |  Threshold: ${ACC_THRESHOLD}%  |  Consecutive above: ${_acc_consec_count} / ${ACC_CONSEC_LIMIT} |  Runtime: ${ELAPSED_FMT}"
    } >> "$ACC_MONITOR_FILE"

    # Compare using awk (bash can't do float comparisons)
    IS_ABOVE=$(awk "BEGIN { print ($ACC_PCT >= $ACC_THRESHOLD) ? 1 : 0 }")

    if [ "$IS_ABOVE" -eq 1 ]; then
      _acc_consec_count=$(( _acc_consec_count + 1 ))
    else
      _acc_consec_count=0
    fi

    if [ "$_acc_consec_count" -ge "$ACC_CONSEC_LIMIT" ]; then
      NOW=$(date +%s)
      ELAPSED=$(( NOW - SCRIPT_START_TIME ))
      ELAPSED_FMT=$(printf '%dh %dm %ds' $(( ELAPSED/3600 )) $(( (ELAPSED%3600)/60 )) $(( ELAPSED%60 )))

      ACCURACY_MSG="[accuracy-monitor] ACC_CONSEC_LIMIT (${ACC_CONSEC_LIMIT}) reached."
      ACCURACY_MSG+="\n  Overall experiment time : ${ELAPSED_FMT}"
      ACCURACY_MSG+="\n  Consecutive rounds above ${ACC_THRESHOLD}% : ${_acc_consec_count}"
      ACCURACY_MSG+="\n  Last recorded accuracy   : ${ACC_PCT}%"
      ACCURACY_MSG+="\n  Triggering graceful shutdown..."

      echo -e "$ACCURACY_MSG"
      echo -e "$ACCURACY_MSG" >> "$AGG_LOG_FILE"
      echo -e "$ACCURACY_MSG" >> "$TRAINER_LOG_FILE"

      kill -TERM $PARENT_PID
      exit 0
    fi
  }

  # EXPANDED_TMP_DIR and AGG_EXPANDED were set up earlier so we could parse the
  # aggregator config for log naming.

  # Clean up expanded configs and background processes on exit.
  # Uses a reentrancy guard so repeated Ctrl+C can't re-enter this handler
  # while it's still running. Escalates to SIGKILL immediately because
  # PyTorch/MQTT children often ignore SIGTERM for many seconds.
  _CLEANING_UP=0
  cleanup() {
    if [ "$_CLEANING_UP" = "1" ]; then
      return
    fi
    _CLEANING_UP=1
    # Disable further trap firings during teardown.
    trap '' INT TERM
    echo "Cleaning up processes and temporary files..."

    # 1. Kill the watchdog first so it stops restarting us.
    if [ -n "$WATCHDOG_PID" ]; then
      kill -KILL "$WATCHDOG_PID" 2>/dev/null || true
    fi

    # 2. Hard-kill all trainer/aggregator processes. No grace period — these
    # are PyTorch+MQTT processes that ignore SIGTERM for 10–30s while
    # cleaning up GPU memory and broker sockets.
    pkill -9 -f "$FWDLLM_USER.*fl_main.py" 2>/dev/null || true

    # 3. Remove temp directory.
    if [ -n "$EXPANDED_TMP_DIR" ] && [ -d "$EXPANDED_TMP_DIR" ]; then
      rm -rf "$EXPANDED_TMP_DIR"
    fi

    # 4. Remove accuracy monitor status file.
    if [ -n "$ACC_MONITOR_FILE" ] && [ -f "$ACC_MONITOR_FILE" ]; then
      rm -f "$ACC_MONITOR_FILE"
    fi
  }
  trap cleanup EXIT
  trap 'cleanup; exit 130' INT TERM

  # AGG_EXPANDED is already populated from earlier (used for log naming).

  # Run aggregator/main.py once with logging
  python $REPO_PATH/lib/python/examples/fwdllm/aggregator/fl_main.py \
    --config "$AGG_EXPANDED" \
    --log_level "$LOG_LEVEL" \
    > "$AGG_LOG_FILE" 2>&1 &

  echo "started agg"

  sleep 10  # Give aggregator time to set up
  if [ "$ENABLE_WATCHDOG" = "true" ]; then
    check_errors
  fi

  echo -e "Log files to monitor: \n [Aggregator]: $AGG_LOG_FILE \n [Trainer]: $TRAINER_LOG_FILE \n [Accuracy]: $ACC_MONITOR_FILE\n"

  # --- GPU assignment ---
  # If CUDA_VISIBLE_DEVICES is set in the environment (e.g. "6,7"), use only
  # those physical GPUs. Inline `CUDA_VISIBLE_DEVICES=N python ...` overrides
  # the parent's CVD for the child, and N is interpreted as an absolute physical
  # device ID — NOT as an index into the parent's visible set. So we must map
  # the logical index (0..N-1) back to the physical GPU ID from _CVD_ARR before
  # spawning the child. Otherwise a parent CVD=6,7 would spawn children pinned
  # to physical GPUs 0,1, which is typically exactly what the user was avoiding.
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra _CVD_ARR <<< "$CUDA_VISIBLE_DEVICES"
    NUM_AVAIL_GPUS=${#_CVD_ARR[@]}
    echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (${NUM_AVAIL_GPUS} GPU(s))"
  else
    NUM_AVAIL_GPUS=8
    _CVD_ARR=(0 1 2 3 4 5 6 7)
    echo "CUDA_VISIBLE_DEVICES not set; defaulting to NUM_AVAIL_GPUS=${NUM_AVAIL_GPUS}"
  fi

  for X in $(seq 0 $(( total_client_num-1 )) )    # End value is inclusive
  do
    ASSIGN_TO_GPU=$(( X % NUM_AVAIL_GPUS ))
    PHYS_GPU="${_CVD_ARR[$ASSIGN_TO_GPU]}"
    TRAIN_SRC="$REPO_PATH/lib/python/examples/fwdllm/expts/run_tc_expts/json_scripts/trainer_${X}.json"
    TRAIN_EXPANDED="$EXPANDED_TMP_DIR/trainer_${X}_expanded.json"

    if [ -f "$TRAIN_SRC" ]; then
      envsubst < "$TRAIN_SRC" > "$TRAIN_EXPANDED"
      echo "  -> expanded trainer config: $TRAIN_EXPANDED"
      echo "Running client $X on physical GPU $PHYS_GPU (logical slot $ASSIGN_TO_GPU)"
      CUDA_VISIBLE_DEVICES="${PHYS_GPU}" python $REPO_PATH/lib/python/examples/fwdllm/trainer/fl_main.py \
        --config "$TRAIN_EXPANDED" \
        --log_level "$LOG_LEVEL" \
        >> "$TRAINER_LOG_FILE" 2>&1 &
      if [ "$ENABLE_WATCHDOG" = "true" ]; then
        check_errors
      fi
      sleep 8
    else
      echo "Trainer config not found, skipping: $TRAIN_SRC"
      fi
  done

  echo -e "Log files created: \n [Aggregator]: $AGG_LOG_FILE \n [Trainer]: $TRAINER_LOG_FILE"

  # Start background periodic check (every 30 seconds) The watchdog
  # will automatically exit if the parent process ($PARENT_PID) dies
  if [ "$ENABLE_WATCHDOG" = "true" ]; then
    echo "accuracy-monitor (appends on each tick): watch -n 10 cat ${ACC_MONITOR_FILE}"
    # Initialize the file immediately so it's always findable from the
    # start
    echo "accuracy-monitor starting up... ($(date '+%Y-%m-%d %H:%M:%S'))" > "$ACC_MONITOR_FILE"
    (
      while kill -0 $PARENT_PID 2>/dev/null; do   # Checks if the parent script is still alive
        check_errors
        check_accuracy
        sleep 30
      done
    ) &
    WATCHDOG_PID=$!
  fi

wait
# Clean up watchdog on normal exit
kill $WATCHDOG_PID 2>/dev/null