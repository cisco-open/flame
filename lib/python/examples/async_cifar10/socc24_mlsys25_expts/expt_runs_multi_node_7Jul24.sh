#!/bin/bash

# Function to check the last 20 accuracy values from log.log
check_accuracy() {
  local accuracy_values
  accuracy_values=$(tail -n 20 log.log | grep -oP 'test accuracy: [0-9]+/[0-9]+ \(\K[0-9]+\.[0-9]+')
  for value in $accuracy_values; do
    if (( $(echo "$value < 0.10" | bc -l) )); then
      return 1
    fi
  done
  return 0
}

# Function to terminate main.py on a given node
terminate_main_py() {
  local node=$1
  ssh dgarg39@$node.cc.gatech.edu "pkill -f main.py"
}

# Repeat the process until terminated manually
while true; do
  # Step 1: Start aggregator on JAYNE (local)
  conda activate dg_flame
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/serenity/scratch/dgarg/anaconda3/envs/dg_flame/lib/
  cd /home/dgarg39/flame/lib/python/examples/async_cifar10/aggregator
  python pytorch/main.py fedbuff_config_final_expt_7jul24.json --log_to_wandb --wandb_run_name agg_7jul_final_alpha100_cifar_80acc_fedbuff_client_avail > agg_7jul_final_alpha100_cifar_80acc_fedbuff_client_avail.log 2>&1 &

  # Step 2: Start trainers on JAYNE (local)
  sleep 10
  conda activate dg_flame
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/serenity/scratch/dgarg/anaconda3/envs/dg_flame/lib/
  cd /home/dgarg39/flame/lib/python/examples/async_cifar10/trainer
  cd config_dir100_num300_traceFail_48h/
  bash exec_100_trainers_jayne.sh > trainer_jayne_7jul_final_alpha100_cifar_80acc_fedbuff_client_avail.log 2>&1 &
  
  # Step 3: SSH into SHEPH node and run commands after 10 seconds
  sleep 10
  ssh dgarg39@shepherd.cc.gatech.edu "bash -s" << EOF
  conda activate dg_flame
  export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/serenity/scratch/dgarg/anaconda3/envs/dg_flame/lib/
  cd /home/dgarg39/flame/lib/python/examples/async_cifar10/trainer
  cd config_dir100_num300_traceFail_48h/
  bash exec_100_trainers_sheph.sh > trainer_sheph_7jul_final_alpha100_cifar_80acc_fedbuff_client_avail.log 2>&1 &
EOF

  # Step 4: SSH into Jayne node and run commands
  ssh dgarg39@jayne.cc.gatech.edu "bash -s" << EOF
  conda activate dg_flame
  export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/serenity/scratch/dgarg/anaconda3/envs/dg_flame/lib/
  cd /home/dgarg39/flame/lib/python/examples/async_cifar10/trainer
  cd config_dir100_num300_traceFail_48h/
  bash exec_100_trainers_jayne.sh > trainer_jayne_7jul_final_alpha100_cifar_80acc_fedbuff_client_avail.log 2>&1 &
EOF

  # Step 4: Monitor the log.log file
  while true; do
    if check_accuracy; then
      # If all last 20 accuracy values are >= 60%, terminate main.py on all nodes
      pkill -f main.py
      terminate_main_py shepherd
      terminate_main_py jayne
      break
    else
      # If not, check again after a minute
      sleep 60
    fi
  done

  # Sleep for 60 seconds before repeating the process for another dataset
  sleep 60
done