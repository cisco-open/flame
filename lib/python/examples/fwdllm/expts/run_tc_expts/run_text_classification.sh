#!/usr/bin/env bash
# Ensure that you have set the FWDLLM_USER environment variable before running this script
# Run to set as part of conda environment:
# conda env config vars set FWDLLM_USER=<your-folder-name>


client_num_per_round=$1
LR=$2
FL_ALG=$3
total_client_num=$4
LOG_LEVEL=$5
ENABLE_WATCHDOG=${6:-false}  # Set to "true" to enable error checking, defaults to "false"
# --- Accuracy monitoring configuration ---
ACC_THRESHOLD=80          # Accuracy percentage (0-100) to monitor for
ACC_CONSEC_LIMIT=20       # Number of consecutive rounds above threshold before stopping the run

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
# model_type=bert
# model_name=bert-base-uncased
# model_type=bert
# model_name=bert-large-uncased
# model_type=albert
# model_name=albert-base-v2
# model_type=roberta-large
# model_name=roberta-large
# model_type=deberta
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

REPO_PATH=/home/dgarg39/$FWDLLM_USER/flame
DATA_DIR=/home/dgarg39/$FWDLLM_USER/fednlp_data

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file
if [ $FL_ALG = "FedAvg" ];then
  mpirun -np $PROCESS_NUM -hostfile mpi_host_file \
  python -m fed_avg_main_tc.py \
    --gpu_mapping_file "gpu_mapping.yaml" \
    --gpu_mapping_key mapping_myMap \
    --client_num_per_round $client_num_per_round \
    --comm_round $ROUND \
    --ci $CI \
    --dataset "${DATA_NAME}" \
    --data_file "${DATA_DIR}/data_files/${DATA_NAME}_data.h5" \
    --partition_file "${DATA_DIR}/partition_files/${DATA_NAME}_partition.h5" \
    --partition_method $PARTITION_METHOD \
    --fl_algorithm $FL_ALG \
    --model_type $model_type\
    --model_name $model_name \
    --do_lower_case True \
    --train_batch_size $train_batch_size \
    --frequency_of_the_test $frequency_of_the_test \
    --eval_batch_size 32 \
    --max_seq_length $max_seq_length \
    --lr $C_LR \
    --server_lr $S_LR \
    --epochs 1 \
    --use_adapter True \
    --learning_rate $LR \
    > ./log/new/fedavg_${model_type}_${DATA_NAME}_lr${LR}_client_num_${client_num_per_round}.log 2>&1
elif [ $FL_ALG = FedSgd ];then
  mpirun -np $PROCESS_NUM -hostfile mpi_host_file \
  python -m fedavg_main_tc \
    --gpu_mapping_file "gpu_mapping.yaml" \
    --gpu_mapping_key mapping_myMap \
    --client_num_per_round $client_num_per_round \
    --comm_round $ROUND \
    --ci $CI \
    --dataset "${DATA_NAME}" \
    --data_file "${DATA_DIR}/data_files/${DATA_NAME}_data.h5" \
    --partition_file "${DATA_DIR}/partition_files/${DATA_NAME}_partition.h5" \
    --partition_method $PARTITION_METHOD \
    --fl_algorithm $FL_ALG \
    --model_type $model_type\
    --model_name $model_name \
    --frequency_of_the_test $frequency_of_the_test \
    --do_lower_case True \
    --train_batch_size $train_batch_size \
    --eval_batch_size 32 \
    --max_seq_length $max_seq_length \
    --lr $C_LR \
    --server_lr $S_LR \
    --epochs 1 \
    --use_adapter True \
    --learning_rate $LR \
    > ./log/new/fedsgd_${model_type}_${DATA_NAME}_lr${LR}_client_num_${client_num_per_round}_full.log 2>&1
else
  LOG_DIR=./log/new
  # if [ -d "$LOG_DIR" ]; then
  #   rm -rf "$LOG_DIR"
  # fi
  mkdir -p "$LOG_DIR"

  # Generate timestamp once
  RUN_TIMESTAMP=$(date +%d_%m_%H_%M)
  LOG_SUFFIX="fedFwd_${model_type}_${DATA_NAME}_lr${LR}_client_num_${client_num_per_round}_numerical_${RUN_TIMESTAMP}"
  AGG_LOG_FILE=$(readlink -f "$LOG_DIR/test_agg_${LOG_SUFFIX}.log")
  TRAINER_LOG_FILE=$(readlink -f "$LOG_DIR/test_trainer_${LOG_SUFFIX}.log")
  PARENT_PID=$$

  ACC_MONITOR_FILE=$(readlink -f "$LOG_DIR/accuracy_monitor_${LOG_SUFFIX}.log")  # overwritten each tick
  SCRIPT_START_TIME=$(date +%s)
  _acc_consec_count=0
  _acc_last_seen_line=""  # dedup: only count each new eval result once
  _acc_last_grep_offset=0   # byte offset: attempt to resume grep from last occurrence

  # Function to check logs for errors and kill all processes if found
  check_errors() {
    # Find the first file that contains a real error (ignoring known false positives)
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

      # Trigger the graceful cleanup and exit
      # We kill the parent process with TERM; the trap will handle the rest.
      kill -TERM $PARENT_PID
      exit 1
    fi
  }

  # Function to monitor model accuracy from the aggregator log.
  # Tracks consecutive rounds above ACC_THRESHOLD and triggers shutdown
  # when ACC_CONSEC_LIMIT is reached.
  check_accuracy() {
    [ -f "$AGG_LOG_FILE" ] || return

    # --- Smart tail-first grep ---
    # The log is append-only and can be very large (up to ~8 GB).
    # We first try scanning only the last 200 KB for the accuracy line, which
    # covers many rounds without reading the full file.  If nothing is found
    # there (e.g. the file is brand-new) we fall back to a full scan.
    ACC_LINE=$(tail -c 204800 "$AGG_LOG_FILE" 2>/dev/null | \
               grep -oE "'acc': [0-9]+\.?[0-9]*" | tail -n 1)
    if [ -z "$ACC_LINE" ]; then
      ACC_LINE=$(grep -oE "'acc': [0-9]+\.?[0-9]*" "$AGG_LOG_FILE" 2>/dev/null | tail -n 1)
    fi

    if [ -z "$ACC_LINE" ]; then
      # No eval result yet — write a waiting status so the file always exists
      printf "Waiting for first eval result... | Runtime: %s\n" \
        "$(printf '%dh %dm %ds' $(( ($(date +%s) - SCRIPT_START_TIME)/3600 )) $(( (($(date +%s) - SCRIPT_START_TIME)%3600)/60 )) $(( ($(date +%s) - SCRIPT_START_TIME)%60 )))" \
        >> "$ACC_MONITOR_FILE"
      return
    fi

    # Dedup guard: if this is the same log line we saw last tick, the training
    # round hasn't produced a new eval result yet — skip counter update but
    # still refresh the runtime in the monitor file.
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

    # Extract the raw fraction (e.g. 0.7505263...) and convert to percentage
    ACC_RAW=$(echo "$ACC_LINE" | grep -oE "[0-9]+\.?[0-9]*$")
    ACC_PCT=$(awk "BEGIN { printf \"%.2f\", $ACC_RAW * 100 }")

    # Write status to the dedicated monitor file (overwrite, not append).
    # This keeps stdout clean — no repeated lines after a 12-hour run.
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

  EXPANDED_TMP_DIR="${REPO_PATH}/tmp_expanded_configs_${RUN_TIMESTAMP}"
  mkdir -p "$EXPANDED_TMP_DIR"

  # Clean up expanded configs and background processes on exit
  cleanup() {
    echo "Cleaning up processes and temporary files..."
    # 1. Kill the watchdog first to prevent recursive calls
    if [ ! -z "$WATCHDOG_PID" ]; then
      kill $WATCHDOG_PID 2>/dev/null
    fi
    # 2. Kill all python trainer/aggregator processes
    pkill -f "$FWDLLM_USER.*fl_main.py"
    # 3. Remove temp directory
    if [ -d "$EXPANDED_TMP_DIR" ]; then
      echo "Removing temporary directory: $EXPANDED_TMP_DIR"
      rm -rf "$EXPANDED_TMP_DIR"
    fi
    # 4. Remove accuracy monitor status file
    if [ -f "$ACC_MONITOR_FILE" ]; then
      echo "Removing accuracy monitor file: $ACC_MONITOR_FILE"
      rm -f "$ACC_MONITOR_FILE"
    fi
  }
  # Trap common termination signals
  trap cleanup EXIT INT TERM    # cleanup function is called no matter how the script ends (normal exit, error, or manual termination)

  # substitute env variables in the temp files
  AGG_SRC="$REPO_PATH/lib/python/examples/fwdllm/expts/run_tc_expts/json_scripts/aggregator.json"
  AGG_EXPANDED="$EXPANDED_TMP_DIR/aggregator_expanded.json"
  envsubst < "$AGG_SRC" > "$AGG_EXPANDED"
  echo "Wrote expanded aggregator config: $AGG_EXPANDED"

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

  NUM_AVAIL_GPUS=8

  for X in $(seq 0 $(( total_client_num-1 )) )    # End value is inclusive
  do
    ASSIGN_TO_GPU=$(( X % NUM_AVAIL_GPUS ))
    TRAIN_SRC="$REPO_PATH/lib/python/examples/fwdllm/expts/run_tc_expts/json_scripts/trainer_${X}.json"
    TRAIN_EXPANDED="$EXPANDED_TMP_DIR/trainer_${X}_expanded.json"

    if [ -f "$TRAIN_SRC" ]; then
      envsubst < "$TRAIN_SRC" > "$TRAIN_EXPANDED"
      echo "  -> expanded trainer config: $TRAIN_EXPANDED"
      echo "Running client $X on GPU $ASSIGN_TO_GPU"
      CUDA_VISIBLE_DEVICES="${ASSIGN_TO_GPU}" python $REPO_PATH/lib/python/examples/fwdllm/trainer/fl_main.py \
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

  # Start background periodic check (every 30 seconds)
  # The watchdog will automatically exit if the parent process ($PARENT_PID) dies
  if [ "$ENABLE_WATCHDOG" = "true" ]; then
    echo "accuracy-monitor (appends on each tick): watch -n 10 cat ${ACC_MONITOR_FILE}"
    # Initialize the file immediately so it's always findable from the start
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
fi