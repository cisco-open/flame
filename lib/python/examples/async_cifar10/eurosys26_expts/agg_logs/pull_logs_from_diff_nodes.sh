#!/bin/bash

# Base remote path
REMOTE_PATH="/home/dgarg39/flame/lib/python/examples/async_cifar10/eurosys26_expts/agg_logs"

# Array of [node filename] entries
declare -a FILES_TO_FETCH=(
    "shepherd agg_sheph_13_05_01_50_alpha0.1_cifar_70acc_TierFuse_TierSelect_TierTrack_syn_0.log"
    "shepherd agg_sheph_11_05_02_42_alpha0.1_cifar_70acc_fedbuff_oortAsync_oracular_syn0.log"
    "wash  agg_wash_11_05_02_42_alpha0.1_cifar_70acc_fedavg_oort_oracular_syn0.log"
    "shepherd agg_sheph_13_05_08_29_alpha0.1_cifar_70acc_TierFuse_TierSelect_TierTrack_syn_20.log"
    "jayne agg_jayne_11_05_15_20_alpha0.1_cifar_70acc_fedbuff_oortAsync_oracular_syn20.log"
    "wash  agg_wash_11_05_11_49_alpha0.1_cifar_70acc_fedavg_oort_oracular_syn20.log"
    "jayne agg_jayne_13_05_01_52_alpha0.1_cifar_70acc_TierFuse_TierSelect_TierTrack_syn_50.log"
    "shepherd agg_sheph_11_05_15_21_alpha0.1_cifar_70acc_fedbuff_oortAsync_oracular_syn50.log"
    "wash  agg_wash_11_05_20_20_alpha0.1_cifar_70acc_fedavg_oort_oracular_syn50.log"
    "wash  agg_wash_13_05_01_51_alpha0.1_cifar_70acc_TierFuse_TierSelect_TierTrack_mobiperf_3st.log"
    "jayne agg_jayne_12_05_00_04_alpha0.1_cifar_70acc_fedbuff_oortAsync_oracular_mobiperf.log"
    "wash  agg_wash_12_05_05_41_alpha0.1_cifar_70acc_fedavg_oort_oracular_mobiperf.log"
)

# Loop over files and fetch each using scp
for entry in "${FILES_TO_FETCH[@]}"; do
    read -r NODE FILE <<< "$entry"
    echo "Fetching $FILE from $NODE..."
    scp "dgarg39@${NODE}.cc.gatech.edu:${REMOTE_PATH}/${FILE}" .
done

echo "All files fetched successfully."