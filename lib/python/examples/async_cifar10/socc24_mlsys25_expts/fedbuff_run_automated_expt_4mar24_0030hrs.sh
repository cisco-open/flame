#!/bin/bash

# Validate command line argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <duration>"
    echo "Example: $0 90m"
    exit 1
fi

# Extract duration from command line argument
duration="$1"

echo "#### Starting FEDBUFF: [Alpha = 100, noFailure, noStaleness]"
# Run aggregator command and wait for 30 seconds
cd aggregator/ && python pytorch/main.py fedbuff_config_large_expt_noStaleness.json > agg_fedbuff_config_dir100_num100_noFail_90min_c20_k10_n100_4mar24_noStaleness.txt &
sleep 30
# Run Bash command as background process and wait for 90 minutes
cd trainer/config_dir100_num100_noFail/ && bash exec_100_trainers.sh > logs_trainer_config_dir100_num100_fedbuff_noFail_4mar24_noStaleness.txt &
sleep "$duration"
# kill all trainer and aggregator processes after 90mins of training
pkill -f main.py
echo "#### Completed FEDBUFF: [Alpha = 100, noFailure]"
sleep 30


echo "#### Starting FEDBUFF: [Alpha = 100, failure, unaware, noStaleness]"
# Run aggregator command and wait for 30 seconds
cd aggregator/ && python pytorch/main.py fedbuff_config_large_expt_noStaleness.json > agg_fedbuff_config_dir100_num100_fail_unaware_90min_c20_k10_n100_4mar24_noStaleness.txt &
sleep 30
# Run Bash command as background process and wait for 90 minutes
cd trainer/config_dir100_num100_traceFailure_1.5h/ && bash exec_100_trainers.sh > logs_trainer_config_dir100_num100_fedbuff_fail_unaware_4mar24_noStaleness.txt &
sleep "$duration"
# kill all trainer and aggregator processes after 90mins of training
pkill -f main.py
echo "#### Completed FEDBUFF: [Alpha = 100, failure, unaware]"
sleep 30


echo "#### Starting FEDBUFF: [Alpha = 100, failure, aware, noStaleness]"
# Run aggregator command and wait for 30 seconds
cd aggregator/ && python pytorch/main.py fedbuff_config_large_expt_trackFail_noStaleness.json > agg_fedbuff_config_dir100_num100_fail_aware_90min_c20_k10_n100_4mar24_noStaleness.txt &
sleep 30
# Run Bash command as background process and wait for 90 minutes
cd trainer/config_dir100_num100_traceFailure_1.5h/ && bash exec_100_trainers.sh > logs_trainer_config_dir100_num100_fedbuff_fail_aware_4mar24_noStaleness.txt &
sleep "$duration"
# kill all trainer and aggregator processes after 90mins of training
pkill -f main.py
echo "#### Completed FEDBUFF: [Alpha = 100, failure, aware]"
sleep 30

echo "#### Starting FEDBUFF: [Alpha = 0.1, noFailure, noStaleness]"
# Run aggregator command and wait for 30 seconds
cd aggregator/ && python pytorch/main.py fedbuff_config_large_expt_noStaleness.json > agg_fedbuff_config_dir0.1_noFail_90min_c20_k10_n100_4mar24_noStaleness.txt &
sleep 30
# Run Bash command as background process and wait for 90 minutes
cd trainer/config_dir0.1_num100_noFail/ && bash exec_100_trainers.sh > logs_trainer_config_dir0.1_num100_fedbuff_noFail_4mar24_noStaleness.txt &
sleep "$duration"
# kill all trainer and aggregator processes after 90mins of training
pkill -f main.py
echo "#### Completed FEDBUFF: [Alpha = 0.1, noFailure]"
sleep 30


echo "#### Starting FEDBUFF: [Alpha = 0.1, failure, unaware, noStaleness]"
# Run aggregator command and wait for 30 seconds
cd aggregator/ && python pytorch/main.py fedbuff_config_large_expt_noStaleness.json > agg_fedbuff_config_dir0.1_fail_unaware_90min_c20_k10_n100_4mar24_noStaleness.txt &
sleep 30
# Run Bash command as background process and wait for 90 minutes
cd trainer/config_dir0.1_num100_traceFailure_1.5h/ && bash exec_100_trainers.sh > logs_trainer_config_dir0.1_num100_fedbuff_fail_unaware_4mar24_noStaleness.txt &
sleep "$duration"
# kill all trainer and aggregator processes after 90mins of training
pkill -f main.py
echo "#### Completed FEDBUFF: [Alpha = 0.1, failure, unaware]"
sleep 30


echo "#### Starting FEDBUFF: [Alpha = 0.1, failure, aware, noStaleness]"
# Run aggregator command and wait for 30 seconds
cd aggregator/ && python pytorch/main.py fedbuff_config_large_expt_trackFail_noStaleness.json > agg_fedbuff_config_dir0.1_fail_aware_90min_c20_k10_n100_4mar24_noStaleness.txt &
sleep 30
# Run Bash command as background process and wait for 90 minutes
cd trainer/config_dir0.1_num100_traceFailure_1.5h/ && bash exec_100_trainers.sh > logs_trainer_config_dir0.1_num100_fedbuff_fail_aware_4mar24_noStaleness.txt &
sleep "$duration"
# kill all trainer and aggregator processes after 90mins of training
pkill -f main.py
echo "#### Completed FEDBUFF: [Alpha = 0.1, failure, aware]"
sleep 30
