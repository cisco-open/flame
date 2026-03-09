# REFL Implementation Q&A

**Date:** February 6, 2026

---

## Q1: Do we need changes to aggregator/trainer code files?

**Answer: NO - No changes needed to main aggregator/trainer code.**

### Why Not?

Flame's architecture provides clean abstractions:
- **Selectors** are plug-and-play via `selector_provider` 
- **Optimizers** are plug-and-play via `optimizer_provider`
- Main aggregator/trainer code is agnostic to which selector/optimizer is used

### What Was Needed?

Only **registration and extension**:
1. ✅ Register `refl_oort` in `flame/selectors.py`
2. ✅ Register `reflfedavg` in `flame/optimizers.py`
3. ✅ Extend `TrainResult` with optional timing fields (backward compatible)
4. ✅ Fix JSON config format (add `taskid`, proper channel structure)

### Migration Plan Adjustments

The MIGRATION_PLAN.md showed the new spawner/launcher system which **already works** with REFL:
- YAML experiment configs specify which selector/optimizer to use
- JSON aggregator configs get loaded normally
- Trainers spawn with correct configurations
- **No code changes needed** in aggregator/trainer main files

---

## Q2: How does availability tracking work in REFL?

**Answer: REFL uses ORACULAR traces - perfect future knowledge of availability.**

### REFL's Approach (third_party/REFL/core/helper/client.py)

```python
class Client:
    def __init__(self, hostId, clientId, speed, traces=None):
        self.traces = traces  # Availability trace data
        
    def isActive(self, cur_time):
        """Check if client is available at current time"""
        if self.traces is None:
            return True  # Always available if no trace
            
        # Wrap time within trace duration
        norm_time = cur_time % self.traces['finish_time']
        
        # Check if within active window
        if (self.traces['active'][i] <= norm_time <= 
            self.traces['inactive'][i]):
            return True
        return False
```

### Trace Format

```python
# Example trace for one client
{
    'duration': 211625,
    'finish_time': 518400,  # Loop period (6 days in seconds)
    'active': [12788, 100044, 188992, ...],     # Start times
    'inactive': [65881, 133574, 208292, ...],   # End times  
    'model': 'CPH1801'  # Device model
}
```

**Key points:**
- `active[i]` to `inactive[i]` = one availability window
- Time wraps: `current_time % finish_time`
- From MobiPerf dataset (real mobile device traces)

### Can REFL Function Without Traces?

**NO - REFL fundamentally requires traces.** From the code:
- If `traces is None`, client is always available (trivial case)
- Priority computation requires trace data
- Deadline filtering needs availability predictions
- This is **intentional** - REFL assumes oracular knowledge for research purposes

### Our Implementation (flame/availability/refl_tracker.py)

```python
class REFLAvailabilityTracker:
    def __init__(self, trace_file):
        """Load traces - pickle or YAML format"""
        self.traces = self.load_traces(trace_file)
    
    def is_available(self, trainer_id, current_time):
        """Check if trainer available now"""
        # Same logic as REFL's Client.isActive()
        
    def get_priority(self, trainer_id, round_start, deadline):
        """Compute priority based on near-term availability"""
        # Returns 0 (unavailable), 1 (partially), or 2 (fully available)
```

**Supported formats:**
- REFL pickle files (from third_party/REFL)
- Flame YAML traces (metadata/availability_traces/*.yaml)

### Is This Realistic?

**For research: Yes** - Allows controlled comparison
**For production: No** - Real systems don't have perfect future knowledge

**Alternative approaches:**
- **Felix**: No traces needed, handles true unpredictability
- **Predictive REFL**: Could add ML-based availability prediction (future work)
- **Trace replay**: Use historical data to simulate realistic scenarios

---

## Q3: Simplify the integration plan?

**Done! ✅**

### What Changed

**Before:** 1,196 lines with detailed implementation code in sections 1-12

**After:** ~270 lines focused on:
- ✅ **Design decisions** - Why we made specific choices
- ✅ **Key concepts** - Availability tracking, priority selection, staleness
- ✅ **Configuration** - How to use the system
- ✅ **Status tracking** - What's done, what's remaining
- ✅ **Quick reference** - Files created, commands to run

**Removed:**
- Detailed code examples for each component
- Step-by-step implementation instructions
- Redundant explanations of already-implemented features
- Speculative future work not relevant to current testing

### What Remains

1. **Executive Summary** - High-level overview
2. **Key Design Decisions** - Why we chose specific approaches
3. **Implementation Status** - Phase 1-3 complete, Phase 4 in progress
4. **Core Design Choices** - Availability tracking, selection, staleness
5. **Configuration Structure** - How configs work
6. **Remaining Work** - What's needed for Phase 4
7. **Key Takeaways** - REFL's strengths and limitations
8. **Appendices** - File lists, REFL vs Felix comparison

---

## Summary of Fixes (Feb 6, 2026)

### Issue: KeyError 'taskid'

**Root Cause:** JSON configs used simplified format, but Flame requires full schema

**Fix Applied:**
```json
{
  "taskid": "experiment_id",  // ✅ Added
  "backend": "mqtt",           // ✅ Added
  "brokers": [...],            // ✅ Added
  "groupAssociation": {...},   // ✅ Added
  "channels": [                // ✅ Fixed structure
    {
      "name": "param-channel",
      "pair": ["trainer", "aggregator"],
      "funcTags": {...}
    }
  ],
  "hyperparameters": {
    "aggGoal": 3,              // ✅ Changed from rounds/epochs only
    "trackTrainerAvail": {...} // ✅ Added availability tracking config
  },
  "selector": {...},
  "optimizer": {...}
}
```

**Files Fixed:**
- ✅ `refl_config_test.json`
- ✅ `refl_config_n300.json`
- ✅ `refl_config_ablation_baseline.json`
- ✅ `refl_config_ablation_avail.json`
- ✅ `refl_config_ablation_staleness.json`

---

## Next Steps

1. **Test again** with fixed configs:
   ```bash
   cd /home/dgarg39/flame/lib/python/examples/async_cifar10
   python3 launch/run_experiment.py experiments/configs/refl_test_5trainers.yaml
   ```

2. **Check logs** for successful initialization:
   ```bash
   tail -f expt_scripts_2026/agg_logs/*_aggregator.log
   ```

3. **Verify** REFL components loaded:
   - "Using selector: refl_oort"
   - "Using optimizer: reflfedavg"

4. **If successful**, proceed to ablation study

---

## Key Points to Remember

1. **No code changes** needed - selectors/optimizers are modular
2. **Traces required** - REFL cannot function without oracular availability data
3. **Synchronous FL** - REFL is not asynchronous like Felix
4. **Config format matters** - Must match Flame's schema exactly
5. **Two-level configs** - YAML experiment → JSON aggregator
