# REFL Integration - Usage Guide

This guide explains how to use the REFL (Resource-Efficient Federated Learning) integration in Flame.

## Overview

The REFL integration provides three main components that can be used independently or together:

1. **Availability-Aware Selection** (`REFLOortSelector`) - Priority-based client selection using availability predictions
2. **Deadline-Based Aggregation** (`REFLFedAvg`) - Filters slow trainers and handles stale updates with various weighting strategies
3. **Availability Tracking** (`REFLAvailabilityTracker`) - Manages client availability traces and predictions

## Quick Start

### IMPORTANT: Use YAML Experiment Configs

The experiment runner expects YAML configuration files that reference JSON aggregator configs.

**Correct usage:**
```bash
cd /home/dgarg39/flame/lib/python/examples/async_cifar10

# Test with 5 trainers
python3 launch/run_experiment.py experiments/configs/refl_test_5trainers.yaml

# Run ablation study
python3 launch/run_experiment.py experiments/configs/refl_ablation_study.yaml

# Full scale with 300 trainers
python3 launch/run_experiment.py experiments/configs/refl_full_n300.yaml
```

**Note:** The YAML files reference JSON aggregator configs in `expt_scripts_2026/configs/refl_config_*.json`

### Basic REFL Configuration (JSON Aggregator Config)

The JSON aggregator configs define the Flame components:

```json
{
  "selector": {
    "sort": "refl_oort",
    "kwargs": {
      "aggr_num": 10,
      "avail_priority": 2,
      "availability_trace_file": "metadata/availability_traces/synthetic_traces.yaml"
    }
  },
  "optimizer": {
    "sort": "reflfedavg",
    "kwargs": {
      "deadline": 100,
      "stale_factor": -4
    }
  }
}
```

**See `expt_scripts_2026/configs/refl_config_*.json` for complete examples.**

## Component Configuration

### 1. REFLOortSelector

Priority-based client selection with availability awareness.

**Required Parameters:**
- `aggr_num`: Number of clients to select per round

**Optional Parameters:**
- `avail_priority`: Priority mode (default: 0)
  - `0`: No priority (standard Oort)
  - `1`: Fill mode - prioritize high-priority, fill from others
  - `2`: Strict mode - only select high-priority clients
- `availability_trace_file`: Path to availability trace file (required if avail_priority > 0)
- `avail_probability`: Prediction accuracy 0-1 (default: 1.0)
- `blacklist_rounds`: Max selections before blacklisting or -1 to disable (default: -1)
- `blacklist_max_len`: Max fraction of clients to blacklist (default: 0.3)
- `pacer_step`: Rounds between pacer adjustments or -1 to disable (default: 20)
- `pacer_delta`: Percentage adjustment per pacer step (default: 5)

**Oort UCB Parameters:**
- `exploration_factor`: Initial exploration rate (default: 0.9)
- `exploration_decay`: Decay rate per round (default: 0.98)
- `exploration_min`: Minimum exploration rate (default: 0.2)
- `round_threshold`: Speed filter percentile (default: 30)
- `alpha`: Staleness weight in utility (default: 2)

**Example:**
```json
{
  "selector": {
    "sort": "refl_oort",
    "kwargs": {
      "aggr_num": 30,
      "avail_priority": 2,
      "availability_trace_file": "metadata/availability_traces/mobiperf_traces.yaml",
      "blacklist_rounds": 50,
      "pacer_step": 20,
      "pacer_delta": 5,
      "exploration_factor": 0.9,
      "exploration_decay": 0.98,
      "exploration_min": 0.2,
      "round_threshold": 30
    }
  }
}
```

### 2. REFLFedAvg Optimizer

Deadline-based aggregation with stale update handling.

**Deadline Parameters:**
- `deadline`: Fixed deadline in seconds, or 0 for moving average (default: 0)
- `target_ratio`: Target percentile for moving average deadline (default: 0.8)
- `initial_deadline`: Initial deadline for moving average (default: 100.0)

**Staleness Parameters:**
- `stale_update`: Max rounds a stale update can remain cached, or -1 for no limit (default: -1)
- `stale_factor`: Staleness weighting strategy (default: 1)
  - `> 1`: Divide by constant factor
  - `1`: Equal weight (standard FedAvg)
  - `-1`: Average - divide by average staleness
  - `-2`: AdaSGD - divide by (staleness + 1)
  - `-3`: DynSGD - multiply by exp(-(staleness + 1))
  - `-4`: REFL - hybrid formula with utility
- `stale_beta`: REFL beta parameter, balances staleness vs utility (default: 0.9)
- `scale_coff`: REFL scaling coefficient (default: 10.0)

**Example:**
```json
{
  "optimizer": {
    "sort": "reflfedavg",
    "kwargs": {
      "deadline": 0,
      "stale_update": 5,
      "stale_factor": -4,
      "stale_beta": 0.9,
      "scale_coff": 10.0,
      "target_ratio": 0.8,
      "initial_deadline": 150.0
    }
  }
}
```

### 3. Availability Traces

Availability traces define when trainers are available/unavailable.

**Format (YAML):**
```yaml
trace_name:
  description: "Description of trace"
  traces:
    "1": {periods: [[start1, end1], [start2, end2]], duration: total_duration}
    "2": {periods: [[start1, end1], [start2, end2]], duration: total_duration}
```

**Provided Traces:**
- `synthetic_traces.yaml`:
  - `syn_0`: Always available (0% dropout)
  - `syn_20`: 20% synthetic dropout
  - `syn_50`: 50% synthetic dropout
- `mobiperf_traces.yaml`:
  - `mobiperf_2st`: Real-world MobiPerf traces (2-state)
  - `mobiperf_3st`: Real-world MobiPerf traces (3-state)

## Configuration Examples

### Example 1: Full REFL (All Features)

```json
{
  "selector": {
    "sort": "refl_oort",
    "kwargs": {
      "aggr_num": 10,
      "avail_priority": 2,
      "blacklist_rounds": 50,
      "pacer_step": 20,
      "availability_trace_file": "metadata/availability_traces/mobiperf_traces.yaml"
    }
  },
  "optimizer": {
    "sort": "reflfedavg",
    "kwargs": {
      "deadline": 100,
      "stale_update": 5,
      "stale_factor": -4,
      "stale_beta": 0.9
    }
  }
}
```

### Example 2: Availability-Aware Selection Only

```json
{
  "selector": {
    "sort": "refl_oort",
    "kwargs": {
      "aggr_num": 10,
      "avail_priority": 2,
      "availability_trace_file": "metadata/availability_traces/synthetic_traces.yaml"
    }
  },
  "optimizer": {
    "sort": "fedavg"
  }
}
```

### Example 3: Staleness Handling Only

```json
{
  "selector": {
    "sort": "random"
  },
  "optimizer": {
    "sort": "reflfedavg",
    "kwargs": {
      "deadline": 100,
      "stale_update": 5,
      "stale_factor": -4
    }
  }
}
```

### Example 4: Comparing Staleness Strategies

**Equal Weighting (Baseline):**
```json
{"optimizer": {"sort": "reflfedavg", "kwargs": {"stale_factor": 1}}}
```

**AdaSGD:**
```json
{"optimizer": {"sort": "reflfedavg", "kwargs": {"stale_factor": -2}}}
```

**DynSGD:**
```json
{"optimizer": {"sort": "reflfedavg", "kwargs": {"stale_factor": -3}}}
```

**REFL (Hybrid):**
```json
{"optimizer": {"sort": "reflfedavg", "kwargs": {"stale_factor": -4, "stale_beta": 0.9}}}
```

## Ablation Studies

Pre-configured ablation configs are provided in `aggregator/`:

1. **`refl_config_ablation_baseline.json`** - No REFL features (baseline)
2. **`refl_config_ablation_avail.json`** - Availability selection only
3. **`refl_config_ablation_staleness.json`** - Staleness handling only
4. **`refl_config_test.json`** - Full REFL (small scale)
5. **`refl_config_n300.json`** - Full REFL (300 trainers)

## Monitoring

### Selector Metrics

The REFLOortSelector logs:
- Number of priority vs. remaining clients per round
- Blacklisted clients count
- Pacer threshold adjustments
- Exploration/exploitation split

### Optimizer Metrics

The REFLFedAvg optimizer logs:
- Fast vs. slow trainer split
- Stale updates: cached, applied, discarded
- Moving average deadline (if adaptive)
- Staleness statistics

Use `get_statistics()` method for programmatic access:
```python
optimizer = aggregator.optimizer
stats = optimizer.get_statistics()
print(f"Stale cached: {stats['stale_cached_count']}")
print(f"Stale applied: {stats['stale_applied_total']}")
print(f"Moving avg deadline: {stats['moving_avg_deadline']:.2f}s")
```

## Troubleshooting

### Issue: No priority clients selected
**Cause:** Availability trace file not loaded or incorrect trace format  
**Solution:** Check trace file path and format. Ensure `avail_priority > 0` and `availability_trace_file` is set.

### Issue: All trainers marked as slow
**Cause:** Deadline too aggressive  
**Solution:** Increase `deadline` or use `deadline: 0` for adaptive moving average.

### Issue: Stale updates never applied
**Cause:** `stale_update` too restrictive or round duration too short  
**Solution:** Increase `stale_update` limit or check trainer/round timing.

### Issue: Blacklist too large
**Cause:** `blacklist_rounds` too low or `blacklist_max_len` too high  
**Solution:** Adjust `blacklist_rounds` upward or reduce `blacklist_max_len`.

## Performance Tuning

### For High Availability Environments (low dropout):
```json
{
  "avail_priority": 0,
  "deadline": 0,
  "stale_factor": 1
}
```
Use standard FedAvg since REFL overhead isn't justified.

### For Moderate Availability:
```json
{
  "avail_priority": 1,
  "deadline": 0,
  "stale_factor": -4
}
```
Use priority fill mode and adaptive deadline.

### For Low Availability (high dropout):
```json
{
  "avail_priority": 2,
  "deadline": 0,
  "stale_update": 10,
  "stale_factor": -4
}
```
Strict priority mode, cache stale updates longer.

## Comparison with Felix/FedBuff

To compare REFL against Felix (FedBuff + AsyncOort):

**REFL Config:**
```json
{
  "selector": {"sort": "refl_oort", "kwargs": {"aggr_num": 10, "avail_priority": 2}},
  "optimizer": {"sort": "reflfedavg", "kwargs": {"deadline": 100, "stale_factor": -4}}
}
```

**Felix Config:**
```json
{
  "selector": {"sort": "async_oort", "kwargs": {"aggr_num": 10}},
  "optimizer": {"sort": "fedbuff", "kwargs": {"use_oort_lr": true}}
}
```

Ensure identical:
- Dataset splits (same Dirichlet alpha)
- Availability traces
- Hyperparameters (batch size, learning rate, etc.)
- Number of trainers and rounds

## References

- **REFL Paper:** https://arxiv.org/abs/2111.01108
- **REFL EuroSys'23:** ACM EuroSys 2023
- **Integration Plan:** `experiments/REFL_INTEGRATION_PLAN.md`
- **Flame Documentation:** `../../README.md`
