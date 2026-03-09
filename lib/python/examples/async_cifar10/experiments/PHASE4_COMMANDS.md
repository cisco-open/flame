# Phase 4: Testing Commands & Setup

**Date:** February 6, 2026  
**Status:** Ready to Test  

---

## What is Phase 4?

Phase 4 is the **Testing & Validation** phase of the REFL integration. The implementation (Phases 1-3) is complete, and now we need to:

1. ✅ **Test the implementation** - Verify components work correctly
2. ✅ **Debug issues** - Fix any runtime errors
3. ✅ **Validate results** - Compare against REFL's published results
4. ✅ **Compare with Felix** - Head-to-head performance comparison

---

## What Was Wrong Before?

### The Issue

You ran:
```bash
python3 launch/run_experiment.py aggregator/refl_config_test.json
```

And got:
```
Loaded 0 experiments from refl_config_test.json
```

### The Problem

The experiment runner (`launch/run_experiment.py`) expects a **YAML experiment configuration file**, not a JSON aggregator config directly.

**Two-level configuration system:**
1. **YAML experiment config** - Defines experiment parameters (which trainers, dataset, execution settings)
2. **JSON aggregator config** - Defines Flame components (selector, optimizer, hyperparameters)

The YAML file **references** the JSON file.

---

## What Was Fixed?

### 1. Moved JSON Configs to Correct Location

```bash
# Before: aggregator/refl_config_test.json
# After:  expt_scripts_2026/configs/refl_config_test.json
```

All REFL JSON configs now in: `expt_scripts_2026/configs/`

### 2. Created YAML Experiment Configs

Created three YAML configs in `experiments/configs/`:

- **refl_test_5trainers.yaml** - Small test (5 trainers)
- **refl_full_n300.yaml** - Full scale (300 trainers)
- **refl_ablation_study.yaml** - Ablation study (4 experiments)

### 3. Fixed aggr_num Parameter

Changed `refl_config_test.json`:
- Before: `"aggr_num": 10`
- After: `"aggr_num": 3` (select 3 out of 5 trainers)

### 4. Updated Documentation

- Updated [REFL_README.md](../aggregator/REFL_README.md) with correct commands
- Created [PHASE4_TESTING_GUIDE.md](PHASE4_TESTING_GUIDE.md) with comprehensive testing guide

---

## Correct Commands for Phase 4

### Step 1: Small Test (Start Here!)

```bash
cd /home/dgarg39/flame/lib/python/examples/async_cifar10

# Run small test with 5 trainers
python3 launch/run_experiment.py experiments/configs/refl_test_5trainers.yaml
```

**Expected output:**
```
======================================================================
EXPERIMENT BATCH RUNNER
======================================================================

Loaded 1 experiments from experiments/configs/refl_test_5trainers.yaml

======================================================================
[1/1] RUNNING EXPERIMENT: refl_test_5trainers_syn0
======================================================================

[1/6] Setting up experiment directory...
  ✓ Created: experiments/run_<timestamp>_refl_oort_n5_oracular_alpha0p1_syn0

[2/6] Initializing spawners...
  ✓ Spawners initialized

[3/6] Starting aggregator...
  Spawning aggregator with config: expt_scripts_2026/configs/refl_config_test.json
  ...
```

### Step 2: Check for Errors

Monitor the aggregator log:
```bash
tail -f expt_scripts_2026/agg_logs/*_aggregator.log
```

**Look for:**
- ✅ "Using selector: refl_oort"
- ✅ "Using optimizer: reflfedavg"
- ✅ "Selected trainers: [...]"
- ❌ Any Python exceptions or errors

### Step 3: Run Ablation Study

After small test passes:
```bash
python3 launch/run_experiment.py experiments/configs/refl_ablation_study.yaml
```

This runs **4 experiments sequentially:**
1. Baseline (no REFL)
2. Availability only
3. Staleness only
4. Full REFL

### Step 4: Full Scale Test

After ablations pass:
```bash
python3 launch/run_experiment.py experiments/configs/refl_full_n300.yaml
```

---

## Expected Behavior

### What Should Happen

1. **Experiment loads** - YAML parsed successfully
2. **Aggregator starts** - JSON config loaded, REFL components registered
3. **Trainers spawn** - 5 (or N) trainers start training
4. **Selection happens** - REFLOortSelector picks high-priority trainers
5. **Aggregation happens** - REFLFedAvg filters by deadline and applies staleness weighting
6. **Training converges** - Accuracy improves over rounds

### What to Check

**In aggregator logs:**
```
INFO - Initializing selector: refl_oort
INFO - Initializing optimizer: reflfedavg
INFO - Round 1: Selected trainers: [1, 3, 5]
INFO - Round 1: Applied 0 stale updates, cached 0
INFO - Round 1: Test accuracy: 0.35
INFO - Round 2: Selected trainers: [2, 4, 5]
INFO - Round 2: Applied 0 stale updates, cached 0
INFO - Round 2: Test accuracy: 0.42
...
```

**In trainer logs:**
```
INFO - Trainer 1 starting
INFO - Round 1: Training epoch 1/1
INFO - Round 1: Sending weights to aggregator
INFO - Round 2: Training epoch 1/1
...
```

---

## Common Issues & Fixes

### Issue 1: ImportError for flame.availability

**Error:**
```
ImportError: No module named 'flame.availability'
```

**Fix:**
```bash
# Make sure you're in the correct directory
cd /home/dgarg39/flame/lib/python/examples/async_cifar10

# Check Python path
python3 -c "import sys; print('\n'.join(sys.path))"
```

### Issue 2: Selector not found

**Error:**
```
KeyError: 'refl_oort' not found in selector registry
```

**Fix:**
```bash
# Verify registration
cd /home/dgarg39/flame/lib/python
python3 -c "from flame.selectors import selector_provider; print('refl_oort' in selector_provider._classes)"
# Should print: True

# If False, check flame/selectors.py registration
```

### Issue 3: Availability trace file not found

**Error:**
```
FileNotFoundError: metadata/availability_traces/synthetic_traces.yaml
```

**Fix:**
```bash
# Check trace file exists
ls metadata/availability_traces/synthetic_traces.yaml

# Verify path in JSON config is relative to async_cifar10 directory
```

### Issue 4: Not enough available trainers

**Error:**
```
ValueError: Not enough trainers available for selection
```

**Fix:**
- Lower `avail_priority` to 1 or 0 in JSON config
- Or use `syn_0` availability mode (always available)

---

## File Structure

```
async_cifar10/
├── experiments/
│   ├── configs/
│   │   ├── refl_test_5trainers.yaml         ← YAML experiment configs
│   │   ├── refl_full_n300.yaml
│   │   └── refl_ablation_study.yaml
│   ├── REFL_INTEGRATION_PLAN.md             ← Overall plan
│   ├── PHASE4_TESTING_GUIDE.md              ← Detailed testing guide
│   └── PHASE4_COMMANDS.md                   ← This file
│
├── expt_scripts_2026/
│   ├── configs/
│   │   ├── refl_config_test.json            ← JSON aggregator configs
│   │   ├── refl_config_n300.json
│   │   ├── refl_config_ablation_baseline.json
│   │   ├── refl_config_ablation_avail.json
│   │   └── refl_config_ablation_staleness.json
│   ├── agg_logs/                            ← Aggregator logs
│   └── trainer_logs/                        ← Trainer logs
│
├── aggregator/
│   └── REFL_README.md                       ← Configuration reference
│
└── launch/
    └── run_experiment.py                    ← Experiment runner
```

---

## Next Steps

1. **Run small test:**
   ```bash
   cd /home/dgarg39/flame/lib/python/examples/async_cifar10
   python3 launch/run_experiment.py experiments/configs/refl_test_5trainers.yaml
   ```

2. **Check logs for errors:**
   ```bash
   tail -f expt_scripts_2026/agg_logs/*_aggregator.log
   ```

3. **Debug any issues** (see "Common Issues & Fixes" above)

4. **If successful**, proceed to ablation study and full-scale tests

5. **Document results** and compare with Felix

---

## Quick Reference

| Task | Command |
|------|---------|
| Small test (5 trainers) | `python3 launch/run_experiment.py experiments/configs/refl_test_5trainers.yaml` |
| Ablation study (4x100 trainers) | `python3 launch/run_experiment.py experiments/configs/refl_ablation_study.yaml` |
| Full scale (300 trainers) | `python3 launch/run_experiment.py experiments/configs/refl_full_n300.yaml` |
| View aggregator logs | `tail -f expt_scripts_2026/agg_logs/*_aggregator.log` |
| View trainer logs | `ls expt_scripts_2026/trainer_logs/` |
| Check experiments | `ls experiments/run_*/` |
| Kill all processes | `pkill -f "flame"` |

---

## Documentation

- **[REFL_INTEGRATION_PLAN.md](REFL_INTEGRATION_PLAN.md)** - Overall integration plan with checkpoint
- **[PHASE4_TESTING_GUIDE.md](PHASE4_TESTING_GUIDE.md)** - Comprehensive testing guide
- **[../aggregator/REFL_README.md](../aggregator/REFL_README.md)** - Configuration reference

---

**Ready to test! Start with the small test (5 trainers) and let me know what happens.**
