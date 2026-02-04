#!/bin/bash

# Dynamic conda initialization - works regardless of conda installation location
# This script detects conda from the current environment instead of hardcoded paths

# Initialize conda in this shell if not already done
if [ -z "$CONDA_PREFIX" ]; then
  # Try to find conda from common locations
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
  elif [ -n "$CONDA_EXE" ]; then
    # If conda executable is in path, derive the conda.sh location
    CONDA_BASE=$(dirname $(dirname "$CONDA_EXE"))
    source "$CONDA_BASE/etc/profile.d/conda.sh"
  else
    echo "Error: Cannot find conda. Please activate your conda environment before running this script."
    exit 1
  fi
fi

check_accuracy() {
  local log_file=$1
  local threshold=$2
  local accuracy_values
  accuracy_values=$(grep -oP 'test accuracy: [0-9]+/[0-9]+ \(\K[0-9]+\.[0-9]+' "$log_file" | tail -n 20)

  # Check if we have at least 20 accuracy values
  if [ $(echo "$accuracy_values" | wc -l) -lt 20 ]; then
    echo "Less than 20 accuracy values found."
    return 1  # Condition not met
  fi

  # Initialize total and count
  local total=0
  local count=0

  # Calculate the total of the last 20 accuracy values
  for value in $accuracy_values; do
    total=$(python -c "print($total + $value)")
    count=$((count + 1))
  done

  # Calculate the average
  local average=$(python -c "print($total / $count)")

  # Get the last accuracy value
  local last_value=$(echo "$accuracy_values" | tail -n 1)

  # Perform the comparison using Python
  result=$(python -c "print(float($average) >= float($threshold) and float($last_value) >= float($threshold))")

  # Check the result of the comparison
  if [ "$result" == "True" ]; then
    return 0  # Condition met
  else
    return 1  # Condition not met
  fi
}

# Function to terminate main.py
terminate_main_py() {
  pkill -f main.py
  pkill -f main_oort_agg.py
}

# Check for the correct number of arguments
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "$(date +'%Y-%m-%d %H:%M:%S') Usage: $0 <node-name> [conda-env-name]"
  echo "Example: $0 kaylee dg_flame"
  echo "If conda-env-name is not provided, will use currently active environment"
  exit 1
fi

node_name=$1
env_name=${2:-$CONDA_DEFAULT_ENV}  # Use second arg or fall back to current env

if [ -z "$env_name" ]; then
  echo "Error: No conda environment specified and no active environment detected"
  echo "Usage: $0 <node-name> [conda-env-name]"
  exit 1
fi

echo "$(date +'%Y-%m-%d %H:%M:%S') Using conda environment: ${env_name}"

# Fixed alpha value
alpha=0.1
threshold=0.70

aggType="fedavg"
selType="oort"
awareMode="oracular"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ASYNC_CIFAR10_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"

echo "$(date +'%Y-%m-%d %H:%M:%S') Async CIFAR10 directory: ${ASYNC_CIFAR10_DIR}"

# Availability traces to run in oracular mode
availability_traces=("syn0" "syn20" "syn50" "mobiperf")

for trace in "${availability_traces[@]}"; do
  echo "$(date +'%Y-%m-%d %H:%M:%S') Starting experiment for trace=${trace}, mode=${awareMode}, alpha=${alpha} on node=${node_name}..."
  start_time=$(date +%s)

  # Activate the specified conda environment
  conda activate "$env_name"
  
  # Kill any existing processes
  pkill -f main.py
  pkill -f main_oort_agg.py
  sleep 10
  echo "$(date +'%Y-%m-%d %H:%M:%S') Waited for cleanup to complete"

  # Set LD_LIBRARY_PATH dynamically based on current conda environment
  if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
    echo "$(date +'%Y-%m-%d %H:%M:%S') Set LD_LIBRARY_PATH to include: $CONDA_PREFIX/lib/"
  fi

  cd "$ASYNC_CIFAR10_DIR/aggregator"

  # Ensure log directories exist
  mkdir -p "$ASYNC_CIFAR10_DIR/eurosys26_expts/agg_logs"
  mkdir -p "$ASYNC_CIFAR10_DIR/eurosys26_expts/trainer_logs"

  timestamp=$(date +%d_%m_%H_%M)
  agg_log_file="$ASYNC_CIFAR10_DIR/eurosys26_expts/agg_logs/agg_${node_name}_${timestamp}_alpha${alpha}_cifar_70acc_${aggType}_${selType}_${awareMode}_${trace}.log"
  config_file="$ASYNC_CIFAR10_DIR/eurosys26_expts/configs/oort_n300_oracular_9may25_${trace}.json"
  wandb_run_name="agg_${node_name}_${timestamp}_alpha${alpha}_cifar_70acc_${aggType}_${selType}_${awareMode}_${trace}_c13_1.3k"

  echo "Created aggregator log file: ${agg_log_file}"
  python pytorch/main_oort_agg.py "$config_file" --log_to_wandb --wandb_run_name "$wandb_run_name" > "$agg_log_file" 2>&1 &
  agg_pid=$!
  echo "Aggregator PID: $agg_pid"
  sleep 15
  echo "$(date +'%Y-%m-%d %H:%M:%S') Waited after aggregator start"

  # Re-activate environment for trainers (in case of shell quirks)
  conda activate "$env_name"
  
  # Set LD_LIBRARY_PATH again
  if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
  fi
  
  cd "$ASYNC_CIFAR10_DIR/trainer/config_dir0.1_num300_traceFail_6d_3state_oort/"
  echo "Inside trainer folder for trace=${trace}"

  trainer_log_file="$ASYNC_CIFAR10_DIR/eurosys26_expts/trainer_logs/log_trainer_${node_name}_${timestamp}_${alpha}_${aggType}_${selType}_${awareMode}_${trace}.log"
  echo "Created trainer log file: ${trainer_log_file}"
  bash exec_300_trainers_2state.sh > "$trainer_log_file" 2>&1 &
  trainer_pid=$!
  echo "Trainer PID: $trainer_pid"
  echo "$(date +'%Y-%m-%d %H:%M:%S') All trainers successfully started"

  while true; do
    if check_accuracy "$agg_log_file" "$threshold"; then
      terminate_main_py
      sleep 30
      break
    else
      sleep 60
    fi
  done

  end_time=$(date +%s)
  elapsed_time=$((end_time - start_time))
  elapsed_human=$(printf '%02dh:%02dm:%02ds\n' $((elapsed_time/3600)) $((elapsed_time%3600/60)) $((elapsed_time%60)))
  echo "$(date +'%Y-%m-%d %H:%M:%S') Finished experiment for trace=${trace} in ${awareMode} mode on node=${node_name}. Time taken: ${elapsed_human}"
  echo "$(date +'%Y-%m-%d %H:%M:%S') Sleeping for a minute before next experiment"
  sleep 60
done

echo "$(date +'%Y-%m-%d %H:%M:%S') All experiments completed!"
