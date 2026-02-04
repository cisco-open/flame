#!/bin/bash

# Source the bashrc file to ensure conda is set up
source /serenity/scratch/dgarg/anaconda3/etc/profile.d/conda.sh

# Function to check the last 20 accuracy values from the given log file against a threshold
check_accuracy() {
  local log_file=$1
  local threshold=$2
  local accuracy_values
  accuracy_values=$(grep -oP 'test accuracy: [0-9]+/[0-9]+ \(\K[0-9]+\.[0-9]+' "$log_file" | tail -n 20)

  # Check if we have at least 20 accuracy values
  if [ $(echo "$accuracy_values" | wc -l) -lt 20 ]; then
    return 1  # Condition not met
  fi

  # Calculate the average of the last 20 accuracy values
  local total=0
  local count=0
  for value in $accuracy_values; do
    total=$(echo "$total + $value" | bc -l)
    count=$((count + 1))
  done
  local average=$(echo "$total / $count" | bc -l)

  # Get the last accuracy value
  local last_value=$(echo "$accuracy_values" | tail -n 1)

  # Check if the average and the last accuracy value are both greater than or equal to the threshold
  if (( $(echo "$average >= $threshold" | bc -l) )) && (( $(echo "$last_value >= $threshold" | bc -l) )); then
    return 0  # Condition met
  else
    return 1  # Condition not met
  fi
}

# Function to terminate main.py
terminate_main_py() {
  pkill -f main.py
}

# Check for the correct number of arguments
if [ "$#" -ne 1 ]; then
  echo "$(date +'%Y-%m-%d %H:%M:%S') Usage: $0 <node-name>"
  exit 1
fi

node_name=$1

# List of baseline names
baseline_names=("sync_v2")

# Array of alpha values
alphas=(100 0.1)
threshold=0.70  # Define the accuracy threshold

# Loop through each baseline name
for baseline_name in "${baseline_names[@]}"; do

  # Determine the trainer directory suffix based on the baseline name
  if [[ $baseline_name == *"_v2" || $baseline_name == *"_v3" ]]; then
    trainer_dir_suffix="_48h_oort"
  else
    trainer_dir_suffix="_48h"
  fi

  # Loop through each alpha value
  for alpha in "${alphas[@]}"; do
    echo "$(date +'%Y-%m-%d %H:%M:%S') Starting experiment with alpha=${alpha} on node=${node_name} with baseline=${baseline_name}..."
    start_time=$(date +%s)

    # Start a new shell, activate conda environment, and clean all currently running processes
    conda activate dg_flame
    pkill -f main.py
    sleep 10  # Wait for the system to stabilize
    echo "$(date +'%Y-%m-%d %H:%M:%S') Waited for cleanup to complete"

    # Start the aggregator process with the correct configuration and log file name
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/serenity/scratch/dgarg/anaconda3/envs/dg_flame/lib/
    cd /home/dgarg39/flame/lib/python/examples/async_cifar10/aggregator
    agg_log_file="/home/dgarg39/flame/lib/python/examples/async_cifar10/aggregator/agg_${node_name}_$(date +%d_%m_%H_%M)_alpha${alpha}_cifar_70acc_oort_${baseline_name}.log"
    echo "Created aggregator log file: ${agg_log_file}"
    python pytorch/main.py fedbuff_config_final_expt_5sep24_${baseline_name}_felix.json --log_to_wandb --wandb_run_name agg_${node_name}_$(date +%d_%m_%H_%M)_alpha${alpha}_cifar_70acc_oort_${baseline_name} > "$agg_log_file" 2>&1 &
    sleep 15  # Wait for the aggregator to start
    echo "$(date +'%Y-%m-%d %H:%M:%S') Waited after aggregator start"

    # Start the trainers
    conda activate dg_flame
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/serenity/scratch/dgarg/anaconda3/envs/dg_flame/lib/
    cd /home/dgarg39/flame/lib/python/examples/async_cifar10/trainer
    cd config_dir${alpha}_num300_traceFail${trainer_dir_suffix}/
    trainer_log_file="/home/dgarg39/flame/lib/python/examples/async_cifar10/trainer/config_dir${alpha}_num300_traceFail${trainer_dir_suffix}/log_trainer_${node_name}_$(date +%d_%m_%H_%M)_${alpha}_num300_${baseline_name}.log"
    echo "Created trainer log file: ${trainer_log_file}"
    bash exec_300_trainers.sh > "$trainer_log_file" 2>&1 &
    echo "$(date +'%Y-%m-%d %H:%M:%S') All trainers successfully started"

    # Monitor the log file
    while true; do
      if check_accuracy "$agg_log_file" "$threshold"; then
        # If condition is met, terminate main.py and wait for 30 seconds
        terminate_main_py
        sleep 30  # Wait for all trainers and aggregator to stop
        break
      else
        # If not, check again after a minute
        sleep 60
      fi
    done

    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))
    elapsed_human=$(printf '%02dh:%02dm:%02ds\n' $((elapsed_time/3600)) $((elapsed_time%3600/60)) $((elapsed_time%60)))

    echo "$(date +'%Y-%m-%d %H:%M:%S') Finished experiment with alpha=${alpha} on node=${node_name} with baseline=${baseline_name}. Time taken: ${elapsed_human}"
    echo "$(date +'%Y-%m-%d %H:%M:%S') Sleeping for a minute, before next experiment"
    sleep 60
  done
done
