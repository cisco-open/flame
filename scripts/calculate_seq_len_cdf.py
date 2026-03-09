import h5py
import numpy as np
from transformers import DistilBertTokenizer
from tqdm import tqdm
import sys

def calculate_stats(name, data):
    print(f"\n--- {name} Sequence Length Stats ---")
    print(f"Mean:   {np.mean(data):.2f}")
    print(f"Median: {np.median(data):.2f}")
    print(f"P90:    {np.percentile(data, 90):.2f}")
    print(f"P95:    {np.percentile(data, 95):.2f}")
    print(f"P99:    {np.percentile(data, 99):.2f}")
    print(f"Max:    {np.max(data)}")
    return np.percentile(data, 95)

def run_analysis(data_path, partition_path, model_name, method):
    print(f"Initializing Tokenizer: {model_name}")
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    
    print(f"Loading Data: {data_path}")
    data_file = h5py.File(data_path, "r")
    x_group = data_file["X"]
    
    # Aggregator Analysis: Individual sequence lengths
    agg_lengths = []
    for key in tqdm(x_group.keys(), desc="Aggregator (Global)"):
        text = x_group[key][()].decode("utf-8")
        agg_lengths.append(len(tokenizer.encode(text, add_special_tokens=True)))
    
    # Trainer Analysis: Max length per batch of 8
    print(f"Loading Partition: {partition_path}")
    partition_file = h5py.File(partition_path, "r")
    partition_data = partition_file[method]["partition_data"]
    
    trainer_batch_max_lengths = []
    batch_size = 8
    
    for client_idx in tqdm(partition_data.keys(), desc="Trainer (Partitions)"):
        indices = partition_data[client_idx]["train"][()]
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_lengths = [len(tokenizer.encode(x_group[str(idx)][()].decode("utf-8"), add_special_tokens=True)) for idx in batch_indices]
            trainer_batch_max_lengths.append(max(batch_lengths))
            
    data_file.close()
    partition_file.close()
    
    agg_p95 = calculate_stats("Aggregator (Individual)", agg_lengths)
    trainer_p95 = calculate_stats("Trainer (Max per Batch of 8)", trainer_batch_max_lengths)
    
    print(f"\nRecommended max_seq_length for 95% coverage:")
    print(f"Aggregator: {int(agg_p95)}")
    print(f"Trainer:    {int(trainer_p95)}")

if __name__ == "__main__":
    # Parameters based on your input
    DATA = "/Users/gaurav/Projects/fednlp_data/data_files/agnews_data.h5"
    PARTITION = "/Users/gaurav/Projects/fednlp_data/partition_files/agnews_partition.h5"
    MODEL = "distilbert-base-uncased"
    METHOD = "uniform"
    
    run_analysis(DATA, PARTITION, MODEL, METHOD)
