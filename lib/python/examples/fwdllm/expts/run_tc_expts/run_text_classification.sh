# Ensure that you have set the FWDLLM_USER environment variable before running this script
# Run to set as part of conda environment:
# conda env config vars set FWDLLM_USER=<your-folder-name>


client_num_per_round=$1
LR=$2
FL_ALG=$3
total_client_num=$4
LOG_LEVEL=$5

pkill -f "gaurav.*/fl_main.py"
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

PARTITION_METHOD="uniform"
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
  AGG_LOG_FILE="$LOG_DIR/test_agg_${LOG_SUFFIX}.log"
  TRAINER_LOG_FILE="$LOG_DIR/test_trainer_${LOG_SUFFIX}.log"

  EXPANDED_TMP_DIR="${REPO_PATH}/tmp_expanded_configs_${RUN_TIMESTAMP}"
  mkdir -p "$EXPANDED_TMP_DIR"

  # Clean up expanded configs on exit (but keep log files!)
  cleanup() {
    echo "Cleaning up expanded JSONs: $EXPANDED_TMP_DIR"
    rm -rf "$EXPANDED_TMP_DIR"
  }
  trap cleanup EXIT

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
      sleep 8
    else
      echo "Trainer config not found, skipping: $TRAIN_SRC"
      fi
  done

  wait
fi