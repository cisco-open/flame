# FedScale Dataset Download Instructions

This document provides instructions on how to download datasets from FedScale and use `fedscale_dataset_partitioner.py` helper to partition the dataset to Flame trainers based on `client_data_mapping` in FedScale.

<!-- ## Download the Partitioned Dataset (Metadata) from Google Drive

1. Navigate to the data directory:
    ```bash
    cd /mydata/
    ```
2. Download and extract the dataset:
    ```bash
    gdown https://drive.google.com/uc?id=1tzpXjMe6VJKp3XxnfLa4--tHyfF-UTI8
    tar -xvf flame_dataset.tar
    ``` -->

## Download FedScale Datasets

1. Clone the FedScale repository:
    ```bash
    cd flame/
    git submodule update --init --recursive
    ```
    <!-- ```bash
    cd /mydata/
    git clone https://github.com/ShixiongQi/FedScale.git && cd /mydata/FedScale && git checkout dataset-partition
    ``` -->

### Download the Femnist Dataset

1. Navigate to the dataset directory and download the Femnist dataset:
    ```bash
    cd ./third_party/FedScale/benchmark/dataset
    ./download.sh download femnist
    ```

### Partition the Femnist Dataset

1. Navigate back to the `third_party` directory and execute the helper:
    ```bash
    python fedscale_dataset_partitioner.py --data_set femnist --data_dir ./FedScale/benchmark/dataset/data/femnist/ --task femnist --model resnet152 --model_zoo none --num_participants 3400
    ```

    The partitioned dataset will be saved to `/tmp/flame_dataset/`. Flame trainers will load the dataset from `/tmp/flame_dataset/`.

## Download the Pre-partitioned Dataset (Metadata) from Google Drive

We also provided a pre-partitioned Femnist dataset with a total of 2,800 trainers.

    ```bash
    cd /tmp/
    gdown https://drive.google.com/uc?id=1tzpXjMe6VJKp3XxnfLa4--tHyfF-UTI8
    tar -xvf flame_dataset.tar
    ```

<!-- ### Download the Reddit Dataset (Requires 25G Disk Space)

1. Navigate to the dataset directory and download the Reddit dataset:
    ```bash
    cd /mydata/FedScale/benchmark/dataset
    ./download.sh download reddit

    cd /mydata/FedScale/benchmark/dataset/data/reddit/
    wget https://huggingface.co/albert-base-v2/raw/main/config.json
    mv config.json albert-base-v2-config.json
    ```

### Download the Google Speech Dataset

1. Navigate to the dataset directory and download the Google Speech dataset:
    ```bash
    cd /mydata/FedScale/benchmark/dataset
    ./download.sh download speech
    ``` -->

<!-- # Download the partitioned dataset (metadata) from Google drive
```
cd /mydata/
gdown https://drive.google.com/uc?id=1tzpXjMe6VJKp3XxnfLa4--tHyfF-UTI8
tar -xvf flame_dataset.tar
```

# Download FedScale datasets
cd /mydata/
git clone https://github.com/ShixiongQi/FedScale.git && cd /mydata/FedScale && git checkout dataset-partition

# Next download the Reddit dataset (25G disk space)
cd /mydata/FedScale/benchmark/dataset
./download.sh download reddit
cd /mydata/FedScale/benchmark/dataset/data/reddit/
wget https://huggingface.co/albert-base-v2/raw/main/config.json
mv config.json albert-base-v2-config.json

# Next download the Google Speech dataset
cd /mydata/FedScale/benchmark/dataset
./download.sh download speech

# Next download the Femnist dataset
cd /mydata/FedScale/benchmark/dataset
./download.sh download femnist -->