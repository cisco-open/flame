# Experiment Launcher - Quick Start Guide

This directory contains the programmatic experiment generation system for async_cifar10. Experiments are defined in compact YAML configs that reference metadata, rather than embedding full trainer configurations.

## Quick Start

### Running Experiments

```bash
# From async_cifar10 directory
cd /path/to/flame/lib/python/examples/async_cifar10

# Run a small test (5 trainers)
python3 launch/run_experiment.py experiments/configs/test_phase3_mini.yaml

# Run full-scale experiments (300 trainers, 4 experiments)
python3 launch/run_experiment.py experiments/configs/oort_n300_all4unavail.yaml
```

### Reproducing Experiments

Each experiment run creates an `execution_config.yaml` in its output directory with all parameters and metadata references needed for exact reproduction:

```bash
# Reproduce from execution config
python3 launch/reproduce.py experiments/run_20260203_223834_oort_n300_alpha0.1_syn0/execution_config.yaml
```

## Config File Structure

Experiment configs use **metadata references** (not embedded data) to keep files readable (~1-2KB):

```yaml
experiments:
  - name: oort_n300_alpha0.1_syn0
    description: "Oort with 300 trainers, alpha=0.1, synthetic unavailability"
    
    trainer:
      num_trainers: 300              # Number of trainers to spawn
      dataset:
        dirichlet_alpha: 0.1         # Data heterogeneity (0.1 = high, 100.0 = low)
      availability:
        mode: syn_0                  # Availability trace key from metadata
    
    aggregator:
      config_template: expt_scripts_2026/configs/oort_n300_oracular_9may25_syn0.json
      selector: oort                 # Selector algorithm
      tracking_mode: oracular        # Oracular or oblivious
      agg_goal: 10                   # Target trainers per aggregation round
    
    execution:
      num_gpus: 8                    # GPUs for trainer distribution
      sleep_between_spawns: 5.0      # Seconds between trainer spawns
      aggregator_warmup_time: 600    # Seconds before first aggregation
```

### Key Configuration Parameters

**Trainer Section:**
- `num_trainers`: Total trainers to spawn (typically 5 for tests, 300 for production)
- `start_id`: Optional, starting trainer ID (default: 1)
- `dataset.dirichlet_alpha`: Data heterogeneity
  - `0.1` = highly heterogeneous (realistic)
  - `1.0` = moderate heterogeneity
  - `10.0` = low heterogeneity
  - `100.0` = nearly IID
- `availability.mode`: References key in `metadata/availability_traces/*.yaml`
  - `syn_0` = always available (0% dropout)
  - `syn_20` = 20% synthetic dropout
  - `syn_50` = 50% synthetic dropout
  - `mobiperf_2st` = real-world MobiPerf traces

**Execution Timing (Critical for Success):**
- `sleep_between_spawns`: Delay between spawning each trainer (seconds)
- `aggregator_warmup_time`: How long aggregator waits before first round (seconds)

**Timing Formula:** To ensure all trainers join before first aggregation:
```
aggregator_warmup_time = 0.4 × sleep_between_spawns × num_trainers
```

Examples:
- 5 trainers: spawn=0.5s → warmup=1s
- 300 trainers: spawn=5.0s → warmup=600s

## Metadata System

Configs reference metadata files instead of embedding full data:

```
metadata/
├── trainer_registry.yaml                    # 300 trainers (intrinsic properties)
├── dataset_splits/
│   ├── cifar10_alpha0.1_n300.yaml          # Dirichlet alpha=0.1
│   ├── cifar10_alpha1.0_n300.yaml          # Dirichlet alpha=1.0
│   ├── cifar10_alpha10.0_n300.yaml         # Dirichlet alpha=10.0
│   └── cifar10_alpha100.0_n300.yaml        # Dirichlet alpha=100.0
└── availability_traces/
    ├── synthetic_traces.yaml                # syn_0, syn_20, syn_50
    └── mobiperf_traces.yaml                 # Real-world per-trainer traces
```

The launcher automatically loads metadata and generates full trainer configs at runtime.

## Output Structure

Each experiment run creates a timestamped directory:

```
experiments/run_20260203_223834_oort_n300_alpha0.1_syn0/
├── execution_config.yaml                    # Full reproducibility record
├── snapshot.yaml                            # Runtime state snapshot
├── aggregator_config.json                   # Generated aggregator config
├── 03_02_26_22_38_..._aggregator.log       # Aggregator logs
└── 03_02_26_22_38_..._trainers.log         # All trainer logs
```

**execution_config.yaml** contains:
- Git commit hash and branch
- Metadata file references and keys
- All experiment parameters
- Exact spawn commands used

Use this file to reproduce experiments exactly:
```bash
python3 launch/reproduce.py experiments/run_*/execution_config.yaml
```

## Running Multiple Experiments

Configs can define multiple experiments in batch:

```yaml
experiments:
  - name: exp1_syn0
    # ... config ...
  
  - name: exp2_syn20
    # ... config ...
  
  - name: exp3_syn50
    # ... config ...
```

The launcher runs them sequentially, each producing its own output directory.

## Optional Features

### Weights & Biases Logging

Add to your experiment config:

```yaml
aggregator:
  log_to_wandb: true
  wandb_project: "my-project"
  wandb_run_name: "oort_n300_alpha0.1_syn0"
```

### Custom Trainer ID Ranges

By default, trainers spawn with IDs [1, num_trainers]. To use a different range:

```yaml
trainer:
  num_trainers: 5
  start_id: 101  # Will spawn trainers 101-105
```

## Troubleshooting

**Issue: Aggregation completes with fewer trainers than expected**
- **Cause:** Aggregator started before trainers joined
- **Solution:** Increase `aggregator_warmup_time` using the formula:
  ```
  warmup = 0.4 × spawn_delay × num_trainers
  ```

**Issue: Losing trainers at startup**
- **Cause:** `sleep_between_spawns` too small for scale
- **Solution:** Use 5.0s for 300 trainers, 0.5s for small tests

**Issue: OOM errors with many trainers**
- **Cause:** Too many trainers per GPU
- **Solution:** Increase `num_gpus` in execution section

## Examples

See configs in this directory:
- `test_phase3_mini.yaml` - Small test (5 trainers, 2 GPUs)
- `oort_n300_all4unavail.yaml` - Full batch (4 × 300 trainers, 8 GPUs)

## Additional Tools

```bash
# Validate metadata integrity
python3 scripts/validate_metadata.py

# Extract/update metadata from existing configs (if needed)
python3 scripts/extract_metadata.py
```

## Migration Note

This system replaces 5,924 static JSON trainer configs with 7 YAML metadata files (99.9% reduction). The old methodology in `expt_scripts_2026/` remains available for backwards compatibility.
