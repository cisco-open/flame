# Async CIFAR-10 Federated Learning

Asynchronous federated learning on CIFAR-10 with 300 trainers, demonstrating client selection strategies (Oort, FedBuff), availability tracking, and non-IID data distributions.

## Quick Start

```bash
# 1. Setup environment (automated)
# Option A: Auto-detect flame repo path
bash setup_env.sh my_flame_env

# Option B: Explicitly provide flame repo path (recommended)
bash setup_env.sh my_flame_env /home/user/flame

# 2. Check MQTT broker is running (contact admin if not)
systemctl is-active mosquitto || pgrep mosquitto

# 3. Run experiment (use your environment name from step 1)
conda activate my_flame_env
cd expt_scripts_2026/scripts
./oort_n300_oracular_1feb_all4unavail.sh my_node_name
```

Logs saved to: `eurosys26_expts/agg_logs/` and `eurosys26_expts/trainer_logs/`

## What This Example Does

Simulates a federated learning environment with:
- **1 Aggregator**: Central server coordinating training
- **300 Trainers**: Clients with local CIFAR-10 data partitions
- **MQTT Communication**: Message broker for parameter exchange
- **Smart Selection**: Oort strategy picks best trainers based on utility
- **Availability Traces**: Simulates real-world trainer unavailability (syn0/20/50, MobiPerf)
- **Non-IID Data**: Dirichlet sampling (alpha=0.1) creates heterogeneous data distributions

The experiment runs until reaching 70% test accuracy, testing 4 availability scenarios automatically.

## Directory Structure

```
async_cifar10/
├── setup_env.sh              # Automated environment setup
├── aggregator/               # Central server
│   ├── pytorch/main_oort_agg.py
│   └── *.json               # Aggregator configs
├── trainer/                  # Client trainers  
│   ├── pytorch/main.py
│   └── config_dir*/         # Pre-configured trainer sets
│       └── exec_*.sh        # Launch scripts
├── eurosys26_expts/
│   ├── scripts/             # Experiment runners ⭐
│   ├── configs/             # Experiment configs
│   ├── agg_logs/            # Output logs
│   └── trainer_logs/
└── data/                    # CIFAR-10 (auto-downloaded)
```

## Manual Setup

If `setup_env.sh` doesn't work for your system:

```bash
# 1. Create environment (replace 'my_flame_env' with your desired name)
conda create -n my_flame_env python=3.9 -y
conda activate my_flame_env

# 2. Install dependencies from root requirements.txt
cd /path/to/flame  # Navigate to flame root directory
pip install -r requirements.txt

# 3. Install Flame library
cd lib/python
pip install -e .

# 4. Set environment variable
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

# 5. Check MQTT broker status
systemctl is-active mosquitto 2>/dev/null || pgrep mosquitto
# If not running, contact your system administrator
```

**Note**: The root `requirements.txt` contains all necessary dependencies including:
- PyTorch, torchvision
- zstandard (for compression)
- wandb, sortedcontainers
- All Flame library dependencies

## Running Experiments

### Using Automated Script (Recommended)

```bash
cd eurosys26_expts/scripts
./oort_n300_oracular_10may_all4unavail.sh my_node_name
```

The script:
- Starts aggregator and 300 trainers
- Monitors accuracy, stops at 70%
- Runs 4 traces: syn0 (no failures), syn20, syn50, mobiperf
- Saves logs to `eurosys26_expts/{agg,trainer}_logs/`

### Manual Execution

**Terminal 1 (Aggregator)**:
```bash
conda activate my_flame_env  # Use your environment name
cd aggregator
python pytorch/main_oort_agg.py ../eurosys26_expts/configs/oort_n300_oracular_9may25_syn0.json
```

**Terminal 2 (Trainers)**:
```bash
conda activate my_flame_env  # Use your environment name
cd trainer/config_dir0.1_num300_traceFail_6d_3state_oort/
bash exec_300_trainers_2state.sh  # Distributes 300 trainers across 8 GPUs
```

**Single Trainer (Testing)**:
```bash
CUDA_VISIBLE_DEVICES=0 python ../pytorch/main.py --config trainer_1.json
```

## Key Configuration Parameters

**Aggregator config** (`eurosys26_expts/configs/*.json`):
```json
{
  "hyperparameters": {
    "aggGoal": 10,              // Trainer updates before global aggregation
    "trackTrainerAvail": {
      "type": "ORACULAR",       // ORACULAR (knows availability) or UNAWARE
      "trace": "avl_events_syn_0"  // syn0/20/50 or mobiperf
    }
  },
  "selector": {
    "sort": "oort",             // oort, random, fedbuff
    "kwargs": {"aggr_num": 10}  // Trainers selected per round
  }
}
```

**Trainer directories** (data distribution):
- `config_dir0.1_num300_*` - Highly non-IID (realistic)
- `config_dir1_num300_*` - Moderately non-IID
- `config_dir100_num300_*` - Nearly IID

## Monitoring

```bash
# Watch aggregator
tail -f eurosys26_expts/agg_logs/agg_*.log | grep "test accuracy"

# Watch trainers
tail -f eurosys26_expts/trainer_logs/log_trainer_*.log

# Weights & Biases (if configured)
# Visit: https://wandb.ai/your-username/ft-distr-ml
```

**Expected progress**: ~30% (round 50) → ~60% (round 200) → 70% (stop)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| MQTT connection failed | Check if mosquitto is running: `systemctl is-active mosquitto` or `pgrep mosquitto`. Contact admin if not running. |
| CUDA out of memory | Edit `exec_*_trainers.sh`, increase `NUM_AVAIL_GPUS` or reduce `batchSize` |
| Import error: `flame` | `cd ../../ && pip install -e .` |
| Processes hang | `pkill -f main.py && pkill -f main_oort_agg.py` |

## References

- [Oort Paper (OSDI'21)](https://www.usenix.org/conference/osdi21/presentation/lai)
- [FedBuff Paper](https://arxiv.org/abs/2106.06639)
- [Flame Documentation](https://github.com/cisco-open/flame/tree/main/docs)
