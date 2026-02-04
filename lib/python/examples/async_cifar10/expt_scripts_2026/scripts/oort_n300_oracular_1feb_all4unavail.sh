#!/bin/bash

# Dynamically detect and source conda Try common conda locations
if [ -n "$CONDA_EXE" ]; then
  # Conda is already initialized
  CONDA_BASE=$(dirname $(dirname $CONDA_EXE))
  source "$CONDA_BASE/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/coc/scratch/$USER/miniconda3/etc/profile.d/conda.sh" ]; then
  source "/coc/scratch/$USER/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/coc/scratch/$USER/anaconda3/etc/profile.d/conda.sh" ]; then
  source "/coc/scratch/$USER/anaconda3/etc/profile.d/conda.sh"
else
  echo "Error: Could not find conda installation"
  echo "Please ensure conda is initialized or set CONDA_EXE environment variable"
  exit 1
fi

# Global variables to track process IDs
AGG_PID=""
TRAINER_SCRIPT_PID=""
CLEANUP_DONE=false

# Comprehensive cleanup function
cleanup_all_processes() {
  if [ "$CLEANUP_DONE" = true ]; then
    return
  fi
  CLEANUP_DONE=true
  
  echo ""
  echo "$(date +'%Y-%m-%d %H:%M:%S') =========================================="
  echo "$(date +'%Y-%m-%d %H:%M:%S') CLEANUP: Terminating all processes..."
  echo "$(date +'%Y-%m-%d %H:%M:%S') =========================================="
  
  # Kill aggregator process
  if [ -n "$AGG_PID" ] && kill -0 "$AGG_PID" 2>/dev/null; then
    echo "$(date +'%Y-%m-%d %H:%M:%S') Killing aggregator (PID: $AGG_PID)"
    kill -TERM "$AGG_PID" 2>/dev/null || true
  fi
  
  # Kill trainer launch script and all its children
  if [ -n "$TRAINER_SCRIPT_PID" ] && kill -0 "$TRAINER_SCRIPT_PID" 2>/dev/null; then
    echo "$(date +'%Y-%m-%d %H:%M:%S') Killing trainer script (PID: $TRAINER_SCRIPT_PID) and all children"
    # Kill the entire process group
    pkill -P "$TRAINER_SCRIPT_PID" 2>/dev/null || true
    kill -TERM "$TRAINER_SCRIPT_PID" 2>/dev/null || true
  fi
  
  # Kill all main.py and main_oort_agg.py processes
  echo "$(date +'%Y-%m-%d %H:%M:%S') Killing all trainer and aggregator Python processes..."
  pkill -f "main.py" 2>/dev/null || true
  pkill -f "main_oort_agg.py" 2>/dev/null || true
  
  # Wait a bit for graceful termination
  sleep 2
  
  # Force kill any remaining processes
  echo "$(date +'%Y-%m-%d %H:%M:%S') Force killing any remaining processes..."
  pkill -9 -f "main.py" 2>/dev/null || true
  pkill -9 -f "main_oort_agg.py" 2>/dev/null || true
  
  # Wait for all processes to actually terminate
  local wait_count=0
  while pgrep -f "main.py\|main_oort_agg.py" > /dev/null && [ $wait_count -lt 10 ]; do
    echo "$(date +'%Y-%m-%d %H:%M:%S') Waiting for processes to terminate... ($wait_count/10)"
    sleep 1
    wait_count=$((wait_count + 1))
  done
  
  if pgrep -f "main.py\|main_oort_agg.py" > /dev/null; then
    echo "$(date +'%Y-%m-%d %H:%M:%S') Warning: Some processes may still be running"
    pgrep -af "main.py\|main_oort_agg.py"
  else
    echo "$(date +'%Y-%m-%d %H:%M:%S') All processes terminated successfully"
  fi
  
  echo "$(date +'%Y-%m-%d %H:%M:%S') =========================================="
}

# Set up trap to catch Ctrl+C (SIGINT) and other termination signals
trap 'echo ""; echo "Caught interrupt signal!"; cleanup_all_processes; exit 130' INT TERM

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

# Function to terminate main.py (replaced by cleanup_all_processes)
terminate_main_py() {
  cleanup_all_processes
}

# Check for the correct number of arguments
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "$(date +'%Y-%m-%d %H:%M:%S') Usage: $0 <node-name> [env-name]"
  echo "  node-name: Identifier for this node/experiment"
  echo "  env-name: Conda environment name (default: dg_flame)"
  exit 1
fi

node_name=$1
env_name=${2:-dg_flame}  # Default to dg_flame if not provided

# Fixed alpha value
alpha=0.1
threshold=0.70

aggType="fedavg"
selType="oort"
awareMode="oracular"

# Availability traces to run in oracular mode
availability_traces=("syn0" "syn20" "syn50" "mobiperf")

# Get the script's directory to make paths relative
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EXPT_BASE_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
ASYNC_CIFAR10_DIR="$( cd "$EXPT_BASE_DIR/.." && pwd )"

for trace in "${availability_traces[@]}"; do
  echo "$(date +'%Y-%m-%d %H:%M:%S') Starting experiment for trace=${trace}, mode=${awareMode}, alpha=${alpha} on node=${node_name}..."
  start_time=$(date +%s)
  
  # Reset cleanup flag for each trace
  CLEANUP_DONE=false

  conda activate "$env_name"
  if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment '$env_name'"
    exit 1
  fi
  
  # Initial cleanup
  echo "$(date +'%Y-%m-%d %H:%M:%S') Performing initial cleanup..."
  pkill -f main.py 2>/dev/null || true
  pkill -f main_oort_agg.py 2>/dev/null || true
  sleep 10
  echo "$(date +'%Y-%m-%d %H:%M:%S') Waited for cleanup to complete"

  # Dynamically set LD_LIBRARY_PATH based on current conda environment
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
  cd "$ASYNC_CIFAR10_DIR/aggregator"

  # Ensure log directories exist
  mkdir -p "$EXPT_BASE_DIR/agg_logs"
  mkdir -p "$EXPT_BASE_DIR/trainer_logs"

  timestamp=$(date +%d_%m_%H_%M)
  agg_log_file="$EXPT_BASE_DIR/agg_logs/agg_${node_name}_${timestamp}_alpha${alpha}_cifar_70acc_${aggType}_${selType}_${awareMode}_${trace}.log"
  config_file="$EXPT_BASE_DIR/configs/oort_n300_oracular_9may25_${trace}.json"
  wandb_run_name="agg_${node_name}_${timestamp}_alpha${alpha}_cifar_70acc_${aggType}_${selType}_${awareMode}_${trace}_c13_1.3k"

  echo "Created aggregator log file: ${agg_log_file}"
  python pytorch/main_oort_agg.py "$config_file" --log_to_wandb --wandb_run_name "$wandb_run_name" > "$agg_log_file" 2>&1 &
  AGG_PID=$!
  echo "Aggregator PID: $AGG_PID"
  sleep 15
  
  # Check if aggregator is still running
  if ! kill -0 "$AGG_PID" 2>/dev/null; then
    echo "$(date +'%Y-%m-%d %H:%M:%S') ERROR: Aggregator failed to start or crashed!"
    echo "$(date +'%Y-%m-%d %H:%M:%S') Check log file: $agg_log_file"
    tail -n 50 "$agg_log_file"
    cleanup_all_processes
    exit 1
  fi
  echo "$(date +'%Y-%m-%d %H:%M:%S') Waited after aggregator start"

  conda activate "$env_name"
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
  cd "$ASYNC_CIFAR10_DIR/trainer/config_dir0.1_num300_traceFail_6d_3state_oort/"
  echo "Inside trainer folder for trace=${trace}"

  trainer_log_file="$EXPT_BASE_DIR/trainer_logs/log_trainer_${node_name}_${timestamp}_${alpha}_${aggType}_${selType}_${awareMode}_${trace}.log"
  echo "Created trainer log file: ${trainer_log_file}"
  bash exec_300_trainers_2state.sh > "$trainer_log_file" 2>&1 &
  TRAINER_SCRIPT_PID=$!
  echo "Trainer script PID: $TRAINER_SCRIPT_PID"
  echo "$(date +'%Y-%m-%d %H:%M:%S') All trainers launch initiated"
  echo "$(date +'%Y-%m-%d %H:%M:%S') To terminate early, press Ctrl+C"

  while true; do
    # Check if aggregator is still running
    if ! kill -0 "$AGG_PID" 2>/dev/null; then
      echo "$(date +'%Y-%m-%d %H:%M:%S') ERROR: Aggregator process died unexpectedly!"
      echo "$(date +'%Y-%m-%d %H:%M:%S') Check log file: $agg_log_file"
      tail -n 50 "$agg_log_file"
      cleanup_all_processes
      exit 1
    fi
    
    if check_accuracy "$agg_log_file" "$threshold"; then
      echo "$(date +'%Y-%m-%d %H:%M:%S') Target accuracy reached!"
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
  
  # Clear PIDs for next iteration
  AGG_PID=""
  TRAINER_SCRIPT_PID=""
  
  echo "$(date +'%Y-%m-%d %H:%M:%S') Sleeping for a minute before next experiment"
  sleep 60
done

echo "$(date +'%Y-%m-%d %H:%M:%S') =========================================="
echo "$(date +'%Y-%m-%d %H:%M:%S') All experiments completed successfully!"
echo "$(date +'%Y-%m-%d %H:%M:%S') =========================================="
