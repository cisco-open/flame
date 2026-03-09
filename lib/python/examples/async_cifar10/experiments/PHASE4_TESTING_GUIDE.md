# Phase 4: REFL Testing & Validation Guide

**Date:** February 6, 2026  
**Status:** Testing Phase  
**Goal:** Validate REFL implementation and compare with Felix

---

## Overview

Phase 4 focuses on testing, debugging, and validating the REFL implementation completed in Phases 1-3. This includes unit tests, integration tests, REFL validation, and head-to-head comparison with Felix.

---

## Quick Start

### 1. Small-Scale Test (5 Trainers)

Start with a minimal test to verify basic functionality:

```bash
cd /home/dgarg39/flame/lib/python/examples/async_cifar10

# Run small test
python3 launch/run_experiment.py experiments/configs/refl_test_5trainers.yaml
```

**What to check:**
- ✓ Aggregator starts without errors
- ✓ 5 trainers spawn successfully
- ✓ REFLOortSelector is used (check logs)
- ✓ REFLFedAvg optimizer is used
- ✓ Training progresses normally
- ✓ No Python exceptions in logs

**Expected output:**
```
======================================================================
EXPERIMENT BATCH RUNNER
======================================================================

Loaded 1 experiments from experiments/configs/refl_test_5trainers.yaml

======================================================================
[1/1] RUNNING EXPERIMENT: refl_test_5trainers_syn0
======================================================================
...
```

### 2. Ablation Study (100 Trainers each)

Run all four ablation experiments sequentially:

```bash
python3 launch/run_experiment.py experiments/configs/refl_ablation_study.yaml
```

**Configurations tested:**
1. **Baseline:** Standard Oort + FedAvg (no REFL)
2. **Availability:** REFL priority selection only
3. **Staleness:** REFL staleness handling only
4. **Full REFL:** All features enabled

**What to compare:**
- Convergence speed (rounds to target accuracy)
- Final accuracy
- Training time per round
- Number of stale updates used
- Fairness metrics

### 3. Full-Scale Test (300 Trainers)

After initial tests pass, run full-scale REFL:

```bash
python3 launch/run_experiment.py experiments/configs/refl_full_n300.yaml
```

**Requirements:**
- Availability traces for 300 trainers (see "Trace Generation" below)
- Dataset splits for 300 trainers (already exists)
- 8 GPUs recommended

---

## Current Issues to Resolve

### Issue 1: Aggregator Config Parameters

The JSON configs in `expt_scripts_2026/configs/refl_config_*.json` may need adjustment:

**Check these parameters:**
- `aggr_num`: Should match `agg_goal` in YAML config
- Trace file path: verify `availability_trace_file` points to correct location
- Hyperparameters: ensure `rounds`, `batchSize`, `learningRate` are appropriate

### Issue 2: Availability Trace Generation

Current traces may not cover all 300 trainers. Need to:

```bash
# Check current trace coverage
python3 -c "import yaml; print(len(yaml.safe_load(open('metadata/availability_traces/synthetic_traces.yaml'))))"
```

If <300, extend traces or create mappings (see "Trace Generation" section below).

### Issue 3: Top Aggregator Integration

The `syncfl/top_aggregator.py` needs to provide round timing metadata:

**Required changes:**
- Track `round_start_time` and `round_end_time`
- Compute `round_duration`
- Pass to optimizer in `TrainResult` objects

**Location:** `lib/python/flame/mode/horizontal/syncfl/top_aggregator.py`

### Issue 4: Selector/Optimizer Registration

Verify the REFL components are properly registered:

```bash
cd /home/dgarg39/flame/lib/python
python3 -c "from flame.selectors import selector_provider; print('refl_oort' in selector_provider._classes)"
python3 -c "from flame.optimizers import optimizer_provider; print('reflfedavg' in optimizer_provider._classes)"
```

Both should print `True`.

---

## Testing Checklist

### ✓ Unit Tests (Not Yet Started)

Create tests for core REFL components:

**Test availability tracker:**
```python
# test_refl_tracker.py
from flame.availability.refl_tracker import REFLAvailabilityTracker

def test_load_traces():
    tracker = REFLAvailabilityTracker("metadata/availability_traces/synthetic_traces.yaml")
    assert tracker.is_available("trainer_1", 0.0)

def test_priority_computation():
    # Test priority = true_prob * ucb_score
    pass

def test_split_by_priority():
    # Test priority-based sorting and splitting
    pass
```

**Test staleness weighting:**
```python
# test_reflfedavg.py
from flame.optimizer.reflfedavg import REFLFedAvg

def test_equal_weighting():
    # stale_factor = 1
    pass

def test_adasgd_weighting():
    # stale_factor = -2: weight = 1/(staleness+1)
    pass

def test_refl_weighting():
    # stale_factor = -4: hybrid formula
    pass
```

**Test deadline filtering:**
```python
def test_fixed_deadline():
    # Filter updates > deadline
    pass

def test_moving_average_deadline():
    # Adaptive deadline at target_ratio percentile
    pass
```

### ✓ Integration Tests (Ready to Run)

1. **Test with 5 trainers (syn_0):**
   ```bash
   python3 launch/run_experiment.py experiments/configs/refl_test_5trainers.yaml
   ```

2. **Test with different availability modes:**
   - `syn_0`: Always available
   - `syn_20`: 20% unavailability
   - `syn_50`: 50% unavailability
   - `mobiperf_2st`: Realistic traces

3. **Test all staleness strategies:**
   - Modify `stale_factor` in JSON config: 1, -1, -2, -3, -4
   - Compare convergence and fairness

### ✓ REFL Validation (Next Step)

Reproduce REFL's published results:

**Target metrics (from REFL paper):**
- CIFAR-10, α=0.1, 100 clients
- Convergence to ~70% accuracy
- 2-3x speedup vs FedAvg
- Better fairness (lower variance across clients)

**How to validate:**
1. Run REFL with their exact hyperparameters
2. Compare convergence curves
3. Check resource efficiency (compute + communication)
4. Measure fairness metrics (client accuracy variance)

---

## Availability Trace Generation

### Current State

Traces exist for synthetic scenarios and MobiPerf:
- `metadata/availability_traces/synthetic_traces.yaml`: Synthetic traces
- `metadata/availability_traces/mobiperf_traces.yaml`: Real-world traces from MobiPerf dataset

### Extending to 300 Trainers

If traces don't cover 300 trainers, two options:

**Option 1: Duplicate traces**
```python
# expand_traces.py
import yaml

with open('metadata/availability_traces/synthetic_traces.yaml') as f:
    traces = yaml.safe_load(f)

# Duplicate to reach 300
expanded = {}
for i in range(1, 301):
    source_id = ((i - 1) % len(traces)) + 1
    expanded[f'trainer_{i}'] = traces[f'trainer_{source_id}']

with open('metadata/availability_traces/synthetic_traces_n300.yaml', 'w') as f:
    yaml.dump(expanded, f)
```

**Option 2: Generate synthetic traces**
```python
# Using REFL's trace generator
import numpy as np

def generate_2state_trace(num_trainers, duration, mean_avail=0.8):
    """Generate 2-state availability traces."""
    traces = {}
    for i in range(1, num_trainers + 1):
        # Random availability pattern
        trace = np.random.rand(duration) < mean_avail
        timestamps = list(range(duration))
        traces[f'trainer_{i}'] = {
            'times': timestamps,
            'availability': trace.tolist()
        }
    return traces
```

---

## Syncfl Top Aggregator Modifications

The top aggregator needs to track round timing and pass it to the optimizer.

**File:** `lib/python/flame/mode/horizontal/syncfl/top_aggregator.py`

**Required changes:**

```python
# In the training loop
round_start_time = time.time()

# ... selection and training ...

round_end_time = time.time()
round_duration = round_end_time - round_start_time

# When creating TrainResult objects
for trainer_id, weights in trainer_weights.items():
    result = TrainResult(
        trainer_id=trainer_id,
        weights=weights,
        # ... other fields ...
        completion_time=time.time(),
        round_duration=round_duration,
        staleness=0,  # Will be computed by optimizer
    )
```

---

## Debugging Tips

### Check Aggregator Logs

Aggregator logs are saved to `expt_scripts_2026/agg_logs/`:

```bash
# View most recent aggregator log
tail -f expt_scripts_2026/agg_logs/*_aggregator.log
```

**What to look for:**
- "Using selector: refl_oort"
- "Using optimizer: reflfedavg"
- "Selected trainers: [...]"
- "Applied stale updates: X, cached: Y"

### Check Trainer Logs

Trainer logs are in `expt_scripts_2026/trainer_logs/`:

```bash
# Check if trainers are running
ls expt_scripts_2026/trainer_logs/

# View specific trainer
tail -f expt_scripts_2026/trainer_logs/trainer_1.log
```

### Common Errors

**ImportError: No module named 'flame.availability'**
- The availability module isn't in Python path
- Ensure you're running from the correct directory

**KeyError: 'refl_oort' not found**
- Selector not registered
- Check `flame/selectors.py` registration

**FileNotFoundError: availability trace file**
- Trace file path incorrect in JSON config
- Use relative path from example directory

**ValueError: Not enough trainers available**
- Priority selection can't find enough available trainers
- Lower `avail_priority` (try 0 or 1)
- Check trace file has correct trainer IDs

---

## Performance Metrics to Track

### Convergence Metrics
- **Rounds to target accuracy** (e.g., 70% test accuracy)
- **Final accuracy** after fixed rounds (e.g., 100 rounds)
- **Convergence curve** (accuracy vs rounds)

### Resource Efficiency
- **Training time per round** (lower is better)
- **Communication cost** (bytes transferred)
- **Compute cost** (GPU hours)

### Fairness Metrics
- **Client accuracy variance** (lower is better)
- **Minimum client accuracy** (higher is better)
- **Jain's fairness index**

### REFL-Specific Metrics
- **Stale update statistics**:
  - Number cached per round
  - Number applied per round
  - Number discarded (too stale)
- **Priority selection**:
  - High priority selections vs low priority
  - Blacklist length over time
- **Pacer adjustments**:
  - `round_threshold` evolution
  - Effect on selection diversity

---

## Next Steps

### Immediate (Week 1)
1. ✅ Create YAML experiment configs (DONE)
2. ⏳ Run small-scale test (5 trainers)
3. ⏳ Debug any import/registration errors
4. ⏳ Verify selector and optimizer are used

### Short-term (Week 2)
1. ⏳ Add round timing to top aggregator
2. ⏳ Create unit tests for REFL components
3. ⏳ Run ablation study (100 trainers)
4. ⏳ Extend availability traces to 300 trainers

### Medium-term (Week 3-4)
1. ⏳ Run full-scale REFL (300 trainers)
2. ⏳ Reproduce REFL's published results
3. ⏳ Compare against Felix on identical workloads
4. ⏳ Document performance differences

---

## Comparison with Felix

### Experimental Setup

For a fair comparison, use **identical** configurations:

**Dataset:**
- CIFAR-10, α=0.1 (same Dirichlet split)
- Same train/test splits for all 300 trainers

**Availability:**
- Same trace file
- Same trainer-to-trace mappings

**Hyperparameters:**
- Same batch size, learning rate, epochs per round
- Same number of rounds

**Selection:**
- Same `agg_goal` (number of trainers per round)
- Compare: Felix vs REFL vs Oort baseline

### Run Comparison

```bash
# 1. Baseline: Standard Oort
python3 launch/run_experiment.py experiments/configs/refl_ablation_study.yaml

# 2. Felix (existing)
python3 launch/run_experiment.py experiments/configs/<felix_config>.yaml

# 3. Full REFL
python3 launch/run_experiment.py experiments/configs/refl_full_n300.yaml
```

### Analysis

Compare across all three:
- Convergence curves
- Resource efficiency
- Fairness metrics
- Robustness to availability changes

**Expected outcome:**
- Felix: Best handling of asynchrony
- REFL: Better fairness and resource efficiency
- Oort: Baseline performance

---

## Success Criteria (Updated)

### Phase 4 Complete When:

- [ ] Small-scale test runs without errors
- [ ] All REFL components working (selector + optimizer)
- [ ] Ablation study shows independent feature effects
- [ ] Full-scale experiment (300 trainers) runs successfully
- [ ] Convergence matches expected REFL behavior
- [ ] Resource efficiency gains validated
- [ ] Fairness improvements demonstrated
- [ ] Head-to-head comparison with Felix completed
- [ ] Documentation updated with findings

---

## Useful Commands

```bash
# Navigate to async_cifar10
cd /home/dgarg39/flame/lib/python/examples/async_cifar10

# Run experiment
python3 launch/run_experiment.py experiments/configs/<config>.yaml

# Check experiment results
ls experiments/

# View logs
tail -f expt_scripts_2026/agg_logs/*_aggregator.log

# Kill all flame processes (if needed)
pkill -f "flame"

# Check available configs
ls experiments/configs/refl*.yaml
ls expt_scripts_2026/configs/refl*.json
```

---

## Contact & Questions

For issues or questions about Phase 4 testing, refer to:
- [REFL_INTEGRATION_PLAN.md](REFL_INTEGRATION_PLAN.md) - Overall integration plan
- [aggregator/REFL_README.md](#) - Configuration guide (if it exists in aggregator/)
- REFL paper: "REFL: Resource-Efficient Federated Learning"
