#!/bin/bash

# Validate command line argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <duration>"
    echo "Example: $0 90m"
    exit 1
fi

# Extract duration from command line argument
duration="$1"


echo "#### Starting FEDBUFF: [Alpha = 100, failure, aware using heartbeat, with fedbuff staleness]"
# Run aggregator command and wait for 30 seconds
cd aggregator/ && python pytorch/main.py fedbuff_config_test_expt_heartbeat_trackFail.json > agg_fedbuff_config_dir100_num100_fail_heartbeat_aware_10s_c6_k4_n10_31mar24.txt &
sleep 30
# Run Bash command as background process and wait for 90 minutes
cd trainer/config_dir100_num100_traceFailure_1.5h/ && bash exec_10_trainers.sh > logs_trainer_config_dir100_num100_fedbuff_fail_heartbeat_aware_10s_31mar24.txt &
sleep "$duration"
# kill all trainer and aggregator processes after 10mins of training
pkill -f main.py
echo "#### Completed FEDBUFF: [Alpha = 100, failure, aware using heartbeat, with fedbuff staleness]"
sleep 30
