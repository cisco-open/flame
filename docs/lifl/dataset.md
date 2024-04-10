# FedScale Dataset Download Instructions

This document provides instructions on how to download datasets from FedScale and 
use `fedscale_dataset_partitioner.py` helper to partition the dataset to Flame trainers based on `client_data_mapping` in FedScale.

## Download FedScale Datasets

1. Navigate to the dataset directory and download the Femnist dataset from FedScale:
    ```bash
    cd ./third_party/fedscale_dataset/
    ./download.sh download femnist
    ```

## Partition the dataset

1. Navigate back to the home directory of flame repository and execute the helper:
    ```bash
    python ./lib/python/scripts/fedscale_dataset_partitioner.py --data_set femnist \
            --data_dir ./third_party/fedscale_dataset/data/femnist/ \
            --task femnist --model resnet152 --model_zoo none \
            --num_participants 3400
    ```

    The partitioned dataset will be saved to `/tmp/flame_dataset/`. 
    Flame trainers will load the dataset from `/tmp/flame_dataset/`.

## Download the Pre-partitioned Dataset (Metadata) from Google Drive

We also provided a pre-partitioned Femnist dataset with a total of 2,800 trainers.

    ```bash
    pip install gdown

    cd /tmp/
    gdown https://drive.google.com/uc?id=1tzpXjMe6VJKp3XxnfLa4--tHyfF-UTI8
    tar -xvf flame_dataset.tar
    ```