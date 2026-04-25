# Dynamic K and C for FwdLLM: Design Document

**Author:** Dhruv Garg  
**Date:** 2026-04-11  
**Branch:** `dg/dyn_k_c_asyncOORT`

---

## Table of Contents

1. [Background and Motivation](#1-background-and-motivation)
2. [Current Architecture](#2-current-architecture)
3. [Answer: Is FwdLLM Constrained to SyncFL?](#3-answer-is-fwdllm-constrained-to-syncfl)
4. [Design Goals and Non-Goals](#4-design-goals-and-non-goals)
5. [High-Level Architecture](#5-high-level-architecture)
6. [Component Design](#6-component-design)
   - 6.1 [DynamicKCPolicy (Abstract Base)](#61-dynamickcpolicy-abstract-base)
   - 6.2 [DynamicKCController](#62-dynamickccontroller)
   - 6.3 [Policy Implementations Summary](#63-policy-implementations-summary)
   - 6.4 [Policy Factory](#64-policy-factory)
   - 6.5 [Configurable Task Eligibility States](#65-configurable-task-eligibility-states)
   - 6.6 [Eligible Ends Count as a Policy Metric](#66-eligible-ends-count-as-a-policy-metric)
7. [Aggregator Implementation Notes](#7-aggregator-implementation-notes)
8. [Phase 2: Async FwdLLM — Dynamic K and C](#8-phase-2-async-fwdllm--dynamic-k-and-c)
9. [Config Schema](#9-config-schema)
10. [Logging and Observability](#10-logging-and-observability)
11. [Testing Plan](#11-testing-plan)
12. [Known Risks and Mitigations](#12-known-risks-and-mitigations)
13. [File Change Summary](#13-file-change-summary)

---

## 1. Background and Motivation

### 1.1 Current Static Behavior

FwdLLM currently uses fixed values for two critical hyperparameters throughout the entire training run:

| Parameter | Symbol | Where Set | Default (`aggregator.json`) |
|-----------|--------|-----------|---------------------------|
| Aggregation Goal | **K** | `config.hyperparameters.aggGoal` → `self._agg_goal` in `fwdllm_aggregator.py:219` | 10 |
| Concurrency | **C** | `config.selector.kwargs["c"]` → `self.c` in `async_oort.py:81` | 30 |

Additionally, the set of trainer ends that are *eligible* for a given task is hardcoded in the selector:

| Task | Eligible Availability States | Code Location |
|------|------------------------------|---------------|
| `train` | `{AVL_TRAIN, None}` | `async_oort.py:1449–1455` |
| `eval` | `{AVL_EVAL, AVL_TRAIN, None}` | `async_oort.py:1457–1487` |

- **K** controls how many gradient updates must be collected before the aggregator performs a variance check and (if variance passes) updates the model weights.
- **C** controls how many trainers are maintained in the active pool at any given time (async FL only).
- **Eligible states** determine which trainers can receive a given task type — this is currently fixed and selector-specific, with no way to override via config.

### 1.2 Why Each of These Is Suboptimal

**K is too rigid:**
- Early: gradient variance is high; too few aggregated gradients may not produce a stable update → K should be larger.
- Late (near convergence): gradients are consistent → K can shrink for faster iteration.
- Post-rejection: after a variance rejection, increasing K temporarily may help collect more gradient diversity before the next attempt.

**C is too rigid:**
- Early: few trainers have utility estimates; over-selecting wastes compute on stale low-quality gradients.
- Mid/late: when training is stable, more parallel trainers = higher throughput with no quality penalty.
- Critically: C is set blind to the *actual number of eligible trainers* at any moment. If only 12 trainers are `AVL_TRAIN` but C=30, the selector cannot fill the concurrency target anyway — C is effectively wasted headroom. A dynamic C can track how many trainers are actually eligible and right-size itself.

**Hardcoded eligible states are wrong for FwdLLM:**

Classic FL: trainers are either training or evaluating — two distinct roles, hence separate `AVL_TRAIN` / `AVL_EVAL` states.

FwdLLM (federated fine-tuning with forward gradients): the training computation and evaluation computation are structurally similar (both are forward passes). A device in `AVL_EVAL` state is *capable* of running the training task too. Restricting training task assignment to `AVL_TRAIN` only (as the current hardcode does) throws away perfectly capable trainers and starves the aggregator of gradient updates. For FwdLLM, `train` should be eligible for both `AVL_TRAIN` and `AVL_EVAL`.

This config should be a first-class parameter, not a code change.

---

## 2. Current Architecture

### 2.1 Inheritance Hierarchy

```
flame.mode.role.Role
  └─ flame.mode.horizontal.syncfl.top_aggregator.TopAggregator
       └─ flame.mode.horizontal.asyncfl.top_aggregator.TopAggregator  (AsyncTopAgg)
            └─ flame.mode.horizontal.syncfl.fwdllm_aggregator.TopAggregator  (FwdLLM TopAgg)
                 └─ examples.fwdllm.aggregator.FedSgdAggregator.FedSGDAggregator
```

### 2.2 Sync vs Async Routing

`compose()` in `fwdllm_aggregator.py:1468` reads `config.selector.kwargs.get("is_async")` and sets `self.is_async`. Every call to `_distribute_weights()` (line 1451) and `_aggregate_weights()` (line 1459) branches on `self.is_async`.

### 2.3 K — Where It Lives

| Location | Line | Usage |
|----------|------|-------|
| `fwdllm_aggregator.py` | 219 | `self._agg_goal = config.hyperparameters.aggregation_goal or 1` — set once |
| `_aggregate_grads_async()` | 614 | `if self._agg_goal_cnt < self._agg_goal: return` |
| `_aggregate_grads_async()` | 621 | `if self._agg_goal_cnt == self._agg_goal: _process_aggregation_goal_met()` |
| `sync_collect_and_accumulate_grads()` | 913 | `if self._agg_goal_cnt >= self._agg_goal: break` |
| `_process_aggregation_goal_met()` | 865 | `self._updates_in_queue -= self._agg_goal` |
| `compose()` asyncfl_loop | 1504 | `Loop(loop_check_fn=lambda: self._agg_goal_cnt == self._agg_goal)` |

### 2.4 C — Where It Lives

| Location | Line | Usage |
|----------|------|-------|
| `async_oort.py __init__()` | 81 | `self.c = kwargs["c"]` — set once |
| `select()` train path | 259 | `concurrency = min(len(ends), self.c)` |
| `select()` eval path | 268 | `concurrency = min(len(ends), self.c + curr_round_eval_slots_left)` |
| `_handle_send_state()` | 1301 | `extra = max(0, concurrency - len(selected_ends))` |

### 2.5 Eligible States Filtering — Where It Lives

The filtering loop in `async_oort.py:1430–1493` (inside `_handle_send_state()`) populates `filtered_ends` from `ends` based on hardcoded logic:

```python
# HARDCODED TODAY (async_oort.py:1449–1487)
if task_to_perform == "train" and (
    curr_end_id_avl_state in (TrainerAvailState.AVL_TRAIN.value, None)
):
    filtered_ends[end_id] = ends[end_id]        # AVL_TRAIN only for train
elif (
    self.check_three_state_avl                   # always True (line 158)
    and task_to_perform == "eval"
    and (curr_end_id_avl_state in (AVL_EVAL, AVL_TRAIN, None))
):
    filtered_ends[end_id] = ends[end_id]        # both states for eval
```

`check_three_state_avl` is hardcoded `True` at `async_oort.py:158` — there is no config path to change it. The same pattern appears in `async_random.py:79`.

---

## 3. Answer: Is FwdLLM Constrained to SyncFL?

**No.** `FedSGDAggregator` inherits from `fwdllm_aggregator.py::TopAggregator`, which extends `AsyncTopAgg`. It already has both `_aggregate_grads_sync()` and `_aggregate_grads_async()`. Routing is by `"is_async"` in `selector.kwargs` — `aggregator.json` already sets `"is_async": true`. Setting `"is_async": false` (or omitting it, since it defaults to `False`) engages the sync path. The class's presence under `syncfl/` is a historical naming artifact.

---

## 4. Design Goals and Non-Goals

### Goals

1. **Dynamic K (both phases):** Allow `_agg_goal` to change during training, driven by a configurable policy.
2. **Dynamic C (Phase 2, async):** Allow concurrency `self.c` to change during training in async FL.
3. **Eligible-ends-aware policies:** Policies can read how many ends are currently eligible per task (train/eval) and factor that into K/C decisions.
4. **Configurable task eligibility states:** Which `avl_state` values are eligible for `train` vs `eval` tasks must be a config parameter in `aggregator.json`, not a code change. Default behavior must match current hardcoded logic.
5. **Selector independence:** Dynamic K/C operates above the selector layer. Any selector (`async_oort`, `async_random`, `fedbuff`) can be paired with the controller without modification — except for reading the two new config keys (`dynamic_c` channel property and `task_eligible_states`).
6. **Backward compatibility:** When the `dynamic_kc` config block is absent or `enabled: false`, and when `task_eligible_states` is absent, all behavior must be identical to today.
7. **Observable:** Every K/C change and every eligible-state mapping is logged.
8. **Pluggable policies:** New policies can be added without touching aggregator or selector core code.

### Non-Goals

- Joint K+C co-optimization (each is controlled independently).
- Changes to the OORT utility computation.
- Online/learned policy parameters.

---

## 5. High-Level Architecture

```
                    ┌──────────────────────────────────────────────────────┐
                    │              FwdLLM TopAggregator                    │
                    │                                                      │
  config reads      │  self._agg_goal  ◄─────────────────────────────┐    │
  task_eligible_    │  (current K)                                    │    │
  states once at ──►│                                                 │    │
  init              │  After _process_aggregation_goal_met():         │    │
                    │    metrics = _build_dynamic_kc_metrics()        │    │
                    │      ├─ var_pass_rate, var_last                 │    │
                    │      ├─ avg_staleness, p75_staleness            │    │
                    │      ├─ n_eligible_train  ◄── computed from     │    │
                    │      └─ n_eligible_eval      task_eligible_     │    │
                    │                               states + ends()   │    │
                    │    new_k, new_c = controller.step(metrics)      │    │
                    │    self._agg_goal = new_k  ──────────────────── ┘    │
                    │    channel.set_property("dynamic_c", new_c)          │
                    │                            │                         │
                    └────────────────────────────│─────────────────────────┘
                                                 │ channel property
                                                 ▼
                    ┌──────────────────────────────────────────────────────┐
                    │              AsyncOortSelector                       │
                    │                                                      │
                    │  __init__(): self._task_eligible_states =            │
                    │    kwargs.get("task_eligible_states", DEFAULTS)      │
                    │                                                      │
                    │  _handle_send_state():                               │
                    │    filtered_ends = {ends eligible for task}          │
                    │    using self._task_eligible_states[task_to_perform] │
                    │    (replaces hardcoded AVL_TRAIN/AVL_EVAL checks)    │
                    │                                                      │
                    │  select():                                           │
                    │    effective_c = channel_props.get("dynamic_c",      │
                    │                                    self.c)           │
                    │    concurrency = min(len(ends), effective_c)         │
                    └──────────────────────────────────────────────────────┘
                                                 │
                                                 │  DynamicKCController
                                  ┌──────────────┴─────────────────┐
                                  │  policy: DynamicKCPolicy        │
                                  │  k: int, c: int                 │
                                  │  bounds, history, cooldown      │
                                  └─────────────────────────────────┘
```

**Core design decisions:**

1. `DynamicKCController` lives in the aggregator — it has the broadest view (variance, staleness, model version, eligible counts).
2. C is communicated to the selector via `channel.set_property("dynamic_c", new_c)` — no direct coupling between aggregator and selector objects.
3. `task_eligible_states` is read from `selector.kwargs` in `__init__()` by both the selector (for filtering) and the aggregator (for counting eligible ends in metrics). Both read from the same config key, ensuring consistency without runtime coupling.
4. `self.c` in the selector is **never mutated**. `dynamic_c` from `channel_props` is an override. When absent, `self.c` is used.
5. `self._agg_goal` in the aggregator **is mutated** by the controller after each aggregation. The `asyncfl_loop` lambda captures `self` by reference so it automatically uses the updated value.

---

## 6. Component Design

### 6.1 `DynamicKCPolicy` (Abstract Base)

**New file:** `lib/python/flame/selector/dynamic_kc_policy.py`

```python
from abc import ABC, abstractmethod
from typing import Optional

class DynamicKCPolicy(ABC):

    @abstractmethod
    def compute_new_k(self, current_k: int, metrics: dict) -> Optional[int]:
        """Return a new K value, or None to keep current."""

    @abstractmethod
    def compute_new_c(self, current_c: int, metrics: dict) -> Optional[int]:
        """Return a new C value, or None to keep current."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging."""
```

The `metrics` dict passed to policies contains (all keys are optional; policies check with `.get()`):

| Key | Type | Source | Notes |
|-----|------|--------|-------|
| `var_pass_rate` | float [0,1] | Aggregator | Rolling fraction of aggs where variance check passed |
| `var_last` | float | Aggregator | Variance from most recent aggregation |
| `avg_staleness` | float | Aggregator | Mean staleness of updates in last K-window |
| `p75_staleness` | float | Aggregator | 75th-pct staleness |
| `model_version` | int | Aggregator | Current model version |
| `n_aggs_completed` | int | Aggregator | Total aggregations since run start |
| `var_threshold` | float | Aggregator | FedSGDAggregator's current variance threshold |
| `n_eligible_train` | int | Aggregator | Ends currently eligible for `train` task (see §6.6) |
| `n_eligible_eval` | int | Aggregator | Ends currently eligible for `eval` task (see §6.6) |

`n_eligible_train` and `n_eligible_eval` are the two new metrics added in this revision. They are computed by the aggregator using the same `task_eligible_states` config that the selector uses for filtering (§6.5), so both are always consistent.

### 6.2 `DynamicKCController`

**New file:** `lib/python/flame/selector/dynamic_kc_controller.py`

```python
import logging
from collections import deque
from typing import Optional, Tuple
from flame.selector.dynamic_kc_policy import DynamicKCPolicy

logger = logging.getLogger(__name__)

class DynamicKCController:
    """
    Controls dynamic K (aggregation goal) and C (concurrency).

    Lives in the aggregator. Called via step(metrics) after every aggregation
    (or every update_every_n_aggs aggregations). Clamps results to [min, max].
    """

    def __init__(
        self,
        policy: DynamicKCPolicy,
        k_init: int,
        c_init: int,
        k_min: int,
        k_max: int,
        c_min: int,
        c_max: int,
        update_every_n_aggs: int = 1,
        history_window: int = 20,
    ):
        self.policy = policy
        self.k = k_init
        self.c = c_init
        self.k_min = k_min
        self.k_max = k_max
        self.c_min = c_min
        self.c_max = c_max
        self.update_every_n_aggs = update_every_n_aggs

        self._agg_counter = 0
        self._total_updates = 0
        self._history = deque(maxlen=history_window)
        self._k_history = [(0, k_init)]
        self._c_history = [(0, c_init)]

    def step(self, metrics: dict) -> Tuple[int, int]:
        """
        Called after each aggregation. Returns (current_k, current_c),
        possibly updated.
        """
        self._agg_counter += 1
        self._total_updates += 1
        self._history.append(metrics)

        if self._agg_counter < self.update_every_n_aggs:
            return self.k, self.c

        self._agg_counter = 0

        new_k = self.policy.compute_new_k(self.k, metrics)
        new_c = self.policy.compute_new_c(self.c, metrics)

        if new_k is not None:
            clamped = max(self.k_min, min(self.k_max, new_k))
            if clamped != self.k:
                logger.info(
                    f"[DynamicKC] K: {self.k} → {clamped} "
                    f"(policy={self.policy.name()}, metrics={_fmt(metrics)})"
                )
                self.k = clamped
                self._k_history.append((self._total_updates, self.k))

        if new_c is not None:
            clamped = max(self.c_min, min(self.c_max, new_c))
            if clamped != self.c:
                logger.info(
                    f"[DynamicKC] C: {self.c} → {clamped} "
                    f"(policy={self.policy.name()}, metrics={_fmt(metrics)})"
                )
                self.c = clamped
                self._c_history.append((self._total_updates, self.c))

        return self.k, self.c

    def get_k(self) -> int: return self.k
    def get_c(self) -> int: return self.c

    def summary(self) -> dict:
        return {
            "k": self.k, "c": self.c,
            "total_updates": self._total_updates,
            "k_changes": len(self._k_history) - 1,
            "c_changes": len(self._c_history) - 1,
            "k_history_last5": list(self._k_history[-5:]),
            "c_history_last5": list(self._c_history[-5:]),
        }

def _fmt(metrics: dict) -> str:
    """Format key metrics for logging (truncated for readability)."""
    keys = ["var_pass_rate", "var_last", "avg_staleness", "n_eligible_train",
            "n_eligible_eval", "model_version"]
    return {k: round(metrics[k], 3) if isinstance(metrics.get(k), float)
            else metrics.get(k) for k in keys if k in metrics}
```

### 6.3 Policy Implementations Summary

| Class | K | C | Key Metric(s) | Use Case |
|-------|---|---|---------------|----------|
| `NoOpPolicy` | ✗ | ✗ | — | Baseline / testing |
| `VarianceBasedPolicy` | ✓ | ✗ | `var_pass_rate` | Adapt K to gradient quality |
| `StalenessBasedPolicy` | ✗ | ✓ | `avg_staleness` | Adapt C to staleness regime |
| `EligibleEndsBasedPolicy` | ✗ | ✓ | `n_eligible_train` | Right-size C to actual eligible pool |
| `StepSchedulePolicy` | ✓ | ✗ | `n_aggs_completed` | Curriculum K decay |
| `CompositePolicy` | ✓/✗ | ✓/✗ | (delegates) | Combine orthogonal policies |

See `lib/python/flame/selector/dynamic_kc_policy.py` for full implementations.

### 6.4 Policy Factory

```python
# In dynamic_kc_policy.py

def build_policy(policy_name: str, policy_kwargs: dict) -> DynamicKCPolicy:
    _registry = {
        "noop":                  NoOpPolicy,
        "variance_based":        VarianceBasedPolicy,
        "staleness_based":       StalenessBasedPolicy,
        "eligible_ends_based":   EligibleEndsBasedPolicy,
        "step_schedule":         StepSchedulePolicy,
        "composite":             CompositePolicy,
    }
    cls = _registry.get(policy_name)
    if cls is None:
        raise ValueError(f"Unknown dynamic_kc policy: '{policy_name}'. "
                         f"Valid: {list(_registry)}")
    return cls(**policy_kwargs)
```

For `composite`, `policy_kwargs["sub_policies"]` is a list of `{"name": ..., "kwargs": {...}}` dicts, each recursively instantiated via `build_policy`.

### 6.5 Configurable Task Eligibility States

#### 6.5.1 The Problem

The current filtering loop in `async_oort.py:1430–1493` hardcodes which `avl_state` values allow an end to receive a given task. This is wrong for FwdLLM:

| Scenario | Desired `train`-eligible states | Currently supported? |
|----------|--------------------------------|----------------------|
| Classic FL | `{AVL_TRAIN}` | ✓ (hardcoded) |
| FwdLLM | `{AVL_TRAIN, AVL_EVAL}` | ✗ (requires code change) |
| Future: eval-only pretraining | `{AVL_EVAL}` | ✗ |

#### 6.5.2 Config Key: `task_eligible_states`

Add to `selector.kwargs` in `aggregator.json`:

```json
"task_eligible_states": {
    "train": ["AVL_TRAIN"],
    "eval":  ["AVL_EVAL", "AVL_TRAIN"]
}
```

Rules:
- Only `AVL_TRAIN`, `AVL_EVAL`, `UN_AVL` are valid state values (from `TrainerAvailState` enum).
- `None` (no state set on the end) is **always treated as eligible** regardless of this config, matching the existing behavior for trainers that don't send heartbeats.
- The key `task_eligible_states` is **optional**. When absent, behavior defaults to the current hardcode (shown above).
- The `check_three_state_avl` boolean is **deprecated** (it was always hardcoded `True` anyway). The new `task_eligible_states` replaces its role completely and explicitly.

#### 6.5.3 Changes to `async_oort.py`

**In `__init__()`**, after existing attribute setup:

```python
# Default matches current hardcoded behavior
_DEFAULT_TASK_ELIGIBLE_STATES = {
    "train": [TrainerAvailState.AVL_TRAIN.value],
    "eval":  [TrainerAvailState.AVL_EVAL.value, TrainerAvailState.AVL_TRAIN.value],
}

raw = kwargs.get("task_eligible_states", _DEFAULT_TASK_ELIGIBLE_STATES)
# Validate
for task, states in raw.items():
    for s in states:
        if s not in {v.value for v in TrainerAvailState}:
            raise ValueError(
                f"task_eligible_states['{task}'] contains unknown state '{s}'. "
                f"Valid: {[v.value for v in TrainerAvailState]}"
            )
self._task_eligible_states: dict[str, list[str]] = raw
logger.info(f"[TaskEligibility] task_eligible_states = {self._task_eligible_states}")
```

**Replace the hardcoded filtering block** (lines 1436–1493) with:

```python
# BEFORE (hardcoded):
if task_to_perform == "train" and (
    curr_end_id_avl_state in (TrainerAvailState.AVL_TRAIN.value, None)
):
    filtered_ends[end_id] = ends[end_id]
    count_avl_train += 1
elif (
    self.check_three_state_avl and task_to_perform == "eval"
    and curr_end_id_avl_state in (AVL_EVAL, AVL_TRAIN, None)
):
    filtered_ends[end_id] = ends[end_id]
    count_avl_eval += 1
else:
    count_ineligible += 1

# AFTER (configurable):
eligible_states_for_task = self._task_eligible_states.get(task_to_perform, [])
if (
    curr_end_id_avl_state is None                       # no state = always eligible
    or curr_end_id_avl_state in eligible_states_for_task
):
    filtered_ends[end_id] = ends[end_id]
    if task_to_perform == "train":
        count_avl_train += 1
    else:
        count_avl_eval += 1
    logger.debug(
        f"Adding end {end_id} to filtered_ends: task={task_to_perform}, "
        f"avl_state={curr_end_id_avl_state}, "
        f"eligible_states={eligible_states_for_task}"
    )
else:
    count_ineligible += 1
    logger.debug(
        f"Skipping end {end_id}: task={task_to_perform}, "
        f"avl_state={curr_end_id_avl_state} not in {eligible_states_for_task}"
    )
```

This is a **pure replacement** — the logic table is identical to current defaults when `task_eligible_states` is absent. The `check_three_state_avl` attribute can remain for backward compat but is no longer consulted by this loop.

The same change applies to `async_random.py`'s analogous block at lines ~562–589 for consistency.

#### 6.5.4 FwdLLM Config for Training on Both States

```json
"task_eligible_states": {
    "train": ["AVL_TRAIN", "AVL_EVAL"],
    "eval":  ["AVL_EVAL", "AVL_TRAIN"]
}
```

With this, ends in `AVL_EVAL` state are eligible to receive `train` tasks. `EligibleEndsBasedPolicy` will then correctly see a larger `n_eligible_train` pool and can scale C up accordingly — a natural interaction between the two new features.

### 6.6 Eligible Ends Count as a Policy Metric

#### 6.6.1 What It Is

`n_eligible_train` and `n_eligible_eval` are counts of ends (across all currently known ends) whose `avl_state` makes them eligible for the respective task according to the `task_eligible_states` config.

These are computed by the aggregator in `_build_dynamic_kc_metrics()` and surfaced as metrics to the `DynamicKCController`. Policies can then use them to make capacity-aware decisions:

- **C too large for eligible pool**: If `n_eligible_train < C`, the selector cannot fill the concurrency target regardless of K/C. Decreasing C to match `n_eligible_train` removes wasted headroom.
- **Eligible pool has grown**: If `n_eligible_train >> C` (e.g., more devices came online, or `task_eligible_states` was expanded to include `AVL_EVAL`), increasing C exploits the available parallelism.
- **Eval bottleneck**: If `n_eligible_eval` is very small relative to the eval goal, the system is bottlenecked on evaluation — can reduce `evalGoalFactor` or decrease K temporarily.

#### 6.6.2 Where It's Computed

In `_build_dynamic_kc_metrics()` in `fwdllm_aggregator.py`, after existing metrics:

```python
# Read the same task_eligible_states that the selector uses
task_eligible_states = self.config.selector.kwargs.get(
    "task_eligible_states",
    {
        "train": [TrainerAvailState.AVL_TRAIN.value],
        "eval":  [TrainerAvailState.AVL_EVAL.value, TrainerAvailState.AVL_TRAIN.value],
    }
)

channel = self.cm.get_by_tag(TAG_DISTRIBUTE)
if channel:
    all_ends = channel.ends()          # dict[end_id, End]
    for task_name, allowed_states in task_eligible_states.items():
        count = 0
        for end in all_ends.values():
            avl = end.get_property(PROP_AVL_STATE)
            # None avl_state = always eligible (matches selector behavior)
            if avl is None or avl in allowed_states:
                count += 1
        metrics[f"n_eligible_{task_name}"] = count
```

**Key property:** Because both the selector and the aggregator read from `config.selector.kwargs["task_eligible_states"]` (with the same defaults when absent), `n_eligible_train` computed by the aggregator is always consistent with the `filtered_ends` count the selector would compute. There is no runtime coupling required.

---

## 7. Aggregator Implementation Notes

### 7.1 Controller initialisation (`fwdllm_aggregator.py::internal_init`)

The `dynamic_kc` block is read from `selector.kwargs`. When `enabled: true`, a `DynamicKCController` is instantiated with the configured policy and bounds. Three counters are added: `_n_aggs_completed`, `_var_pass_count`, `_var_total_count`.

### 7.2 Hook in `_process_aggregation_goal_met`

After each aggregation goal is met, `_build_dynamic_kc_metrics(channel)` collects `var_pass_rate`, `avg_staleness`, and `n_eligible_train` metrics, then `controller.step(metrics)` returns `(new_k, new_c)`. `self._agg_goal` is updated in-place; C is propagated to the selector via `channel.set_property("dynamic_c", new_c)`. See `fwdllm_aggregator.py` for the full implementation.

---

## 8. Phase 2: Async FwdLLM — Dynamic K and C

Phase 2 builds on Phase 1. The aggregator-side changes are the same (the hook in `_process_aggregation_goal_met()` fires in both sync and async paths). The additional work is teaching `AsyncOortSelector` to consume `dynamic_c`.

### 8.1 Changes to `async_oort.py`

`select()` reads `dynamic_c` from `channel_props` (falling back to `self.c` when absent) and uses it as `effective_c` for both train and eval concurrency. The async receive gate uses `>=` instead of `==` to prevent missed triggers when K decreases mid-run (`_agg_goal_cnt` is reset to 0 immediately inside `_process_aggregation_goal_met`).

### 8.2 Interaction Between K, C, and Eligible Ends Count (Async)

These three axes interact in the async path:

| Scenario | Recommended Policy Response |
|----------|-----------------------------|
| `n_eligible_train < C` | `EligibleEndsBasedPolicy` decreases C to match eligible pool size |
| `n_eligible_train` grows (e.g., FwdLLM expands `train` states to `AVL_EVAL`) | `EligibleEndsBasedPolicy` increases C to exploit new capacity |
| `var_pass_rate` is high | `VarianceBasedPolicy` decreases K (aggregating more than needed) |
| `avg_staleness > threshold` | `StalenessBasedPolicy` decreases C (fewer parallel trainers = fresher updates) |

To avoid instability, `VarianceBasedPolicy` controls K only and `StalenessBasedPolicy`/`EligibleEndsBasedPolicy` control C only. Use `CompositePolicy` to combine them. The controller applies K and C changes sequentially in the same step — but since the policies are orthogonal, there is no conflict.

### 8.3 The `asyncfl_loop` and Dynamic K

```python
asyncfl_loop = Loop(loop_check_fn=lambda: self._agg_goal_cnt == self._agg_goal)
```

The lambda captures `self` by reference, so `self._agg_goal` is evaluated fresh on each loop check — the loop automatically uses the updated value. The `>=` fix (§8.1) also guards the async receive gate.

---

## 9. Config Schema

### 10.1 Async FwdLLM (recommended production config)

```json
"selector": {
    "sort": "async_oort",
    "kwargs": {
        "c": 30,
        "aggGoal": 10,
        "evalGoalFactor": 0,
        "selectType": "default",
        "roundNudgeType": "last_train",
        "minInitialTrainers": 50,
        "k": 10,
        "is_async": true,

        "task_eligible_states": {
            "//": "FwdLLM: AVL_EVAL ends can also train",
            "train": ["AVL_TRAIN", "AVL_EVAL"],
            "eval":  ["AVL_EVAL", "AVL_TRAIN"]
        },

        "dynamic_kc": {
            "enabled": true,
            "policy": "composite",
            "policy_kwargs": {
                "sub_policies": [
                    {
                        "name": "variance_based",
                        "kwargs": {
                            "high_threshold": 0.8,
                            "low_threshold": 0.3,
                            "k_step": 2,
                            "window": 10
                        }
                    },
                    {
                        "name": "eligible_ends_based",
                        "kwargs": {
                            "headroom_factor": 1.5,
                            "undercommit_factor": 0.8,
                            "c_step": 5,
                            "window": 5
                        }
                    }
                ]
            },
            "k_min": 5,
            "k_max": 30,
            "c_min": 10,
            "c_max": 80,
            "update_every_n_aggs": 5
        }
    }
}
```

### 10.2 Classic Sync FL (Phase 1, baseline for testing)

```json
"selector": {
    "sort": "async_oort",
    "kwargs": {
        "is_async": false,

        "task_eligible_states": {
            "train": ["AVL_TRAIN"],
            "eval":  ["AVL_EVAL", "AVL_TRAIN"]
        },

        "dynamic_kc": {
            "enabled": true,
            "policy": "step_schedule",
            "policy_kwargs": {"k_step": 2, "n_aggs_per_step": 30, "k_floor": 5},
            "k_min": 5,
            "k_max": 30,
            "c_min": 1,
            "c_max": 1,
            "update_every_n_aggs": 1
        }
    }
}
```

### 10.3 Backward-Compatible (No Dynamic K/C, Classic Eligible States)

```json
"selector": {
    "sort": "async_oort",
    "kwargs": {
        "c": 30,
        "aggGoal": 10,
        "is_async": true
    }
}
```

When `dynamic_kc` is absent: controller is `None`, no K/C changes ever happen.  
When `task_eligible_states` is absent: filtering defaults to current hardcoded logic.  
Existing experiments are **completely unaffected**.

---

## 10. Logging and Observability

### Log lines emitted (all at `INFO` level)

**Controller init:**
```
[DynamicKC] Controller initialized. K_init=10, C_init=30, policy=composite[variance_based,eligible_ends_based], k_bounds=[5,30], c_bounds=[10,80]
```

**Task eligibility at selector init:**
```
[TaskEligibility] task_eligible_states = {'train': ['AVL_TRAIN', 'AVL_EVAL'], 'eval': ['AVL_EVAL', 'AVL_TRAIN']}
```

**K change:**
```
[DynamicKC] K: 10 → 8 (policy=variance_based, metrics={'var_pass_rate': 0.83, 'var_last': 0.041, 'model_version': 47, 'n_eligible_train': 38})
[DynamicKC] _agg_goal: 10 → 8 at model_version=47
```

**C change:**
```
[DynamicKC] C: 30 → 35 (policy=eligible_ends_based, metrics={'n_eligible_train': 52, 'n_eligible_eval': 20, 'model_version': 52})
[DynamicKC] Propagated dynamic_c=35 to channel.
```

**Selector using dynamic C:**
```
[DynamicKC] Using dynamic_c=35 (static self.c=30)
```

**Eligible ends count at filter time:**
```
Filtered ends created. count_avl_train: 32, count_avl_eval: 6, count_ineligible: 12
```

**Summary (every 10 aggs):**
```
[DynamicKC] Summary: {'k': 7, 'c': 35, 'total_updates': 50, 'k_changes': 2, 'c_changes': 3, 'k_history_last5': [(0,10),(20,8),(40,7)], 'c_history_last5': [(0,30),(25,35)]}
```

---

## 11. Testing Plan

### 11.1 Unit Tests: Policies

**File:** `lib/python/tests/selector/test_dynamic_kc_policy.py`

| Test | What It Checks |
|------|---------------|
| `test_noop_always_returns_none` | NoOp → None for any metrics |
| `test_variance_increases_k_when_noisy` | `var_pass_rate=0.2` → K + step |
| `test_variance_decreases_k_when_stable` | `var_pass_rate=0.9` → K − step |
| `test_variance_no_change_in_midrange` | `var_pass_rate=0.55` → None |
| `test_variance_waits_for_window` | < window steps → None even if rate is extreme |
| `test_staleness_decreases_c_when_stale` | `avg_staleness=5.0` → C − step |
| `test_staleness_increases_c_when_fresh` | `avg_staleness=0.4` → C + step |
| `test_eligible_ends_increases_c_when_pool_large` | `n_eligible_train = C * 2` → C + step |
| `test_eligible_ends_decreases_c_when_pool_small` | `n_eligible_train = C * 0.5` → `int(avg_eligible)` |
| `test_eligible_ends_no_change_in_range` | `n_eligible_train = C * 1.1` → None |
| `test_eligible_ends_missing_metric` | `n_eligible_train` absent → None |
| `test_step_schedule_fires_at_n_aggs` | At n=50,100,150 → K − step |
| `test_composite_variance_wins_for_k` | Composite → variance controls K |
| `test_composite_eligible_wins_for_c` | Composite → eligible_ends controls C |
| `test_composite_first_wins` | Two policies both returning non-None for K → first one wins |

### 11.2 Unit Tests: Controller

**File:** `lib/python/tests/selector/test_dynamic_kc_controller.py`

| Test | What It Checks |
|------|---------------|
| `test_k_clamped_to_bounds` | K never < k_min, never > k_max |
| `test_c_clamped_to_bounds` | C never < c_min, never > c_max |
| `test_update_every_n_not_earlier` | Policy called only every N steps |
| `test_noop_leaves_k_c_unchanged` | NoOp policy → k and c never change |
| `test_summary_keys` | summary() returns expected dict keys |
| `test_history_logged` | k_history and c_history grow on each change |

### 11.3 Unit Tests: Task Eligibility

**File:** `lib/python/tests/selector/test_task_eligible_states.py`

| Test | What It Checks |
|------|---------------|
| `test_default_train_only_avl_train` | Default: `train` admits only `AVL_TRAIN` (+ None) |
| `test_default_eval_admits_both` | Default: `eval` admits `AVL_EVAL` + `AVL_TRAIN` |
| `test_fwdllm_train_admits_avl_eval` | `task_eligible_states.train = [AVL_TRAIN, AVL_EVAL]` |
| `test_none_state_always_eligible` | End with `avl_state=None` always passes regardless of config |
| `test_un_avl_never_eligible` | `UN_AVL` is never in any default eligible list |
| `test_invalid_state_raises` | Unknown state string in config → `ValueError` at init |
| `test_n_eligible_metric_matches_filter` | Aggregator's `n_eligible_train` equals `len(filtered_ends)` computed by selector with same config |

The last test is the critical integration invariant: the metric the aggregator computes must agree with what the selector will actually do. Run this as a parameterized test over several `task_eligible_states` configs.

### 11.4 Integration Test: Aggregator Dry-Run

1. Instantiate `FedSGDAggregator` with a mock config including `dynamic_kc` + `task_eligible_states`.
2. Populate mock ends with various `avl_state` values.
3. Call `_process_aggregation_goal_met()` several times with varying variance.
4. Assert `self._agg_goal` changes according to the policy.
5. Assert `channel.get_property("dynamic_c")` reflects the controller's C.

### 11.5 End-to-End Validation

Run `run_text_classification.sh` with three configs:

| Config | `task_eligible_states` | `dynamic_kc` | Expected Outcome |
|--------|------------------------|-------------|-----------------|
| A (baseline) | absent (default) | absent | Identical to current behavior |
| B (eligibility only) | `train: [AVL_TRAIN, AVL_EVAL]` | absent | More eligible trainers, faster convergence |
| C (full) | `train: [AVL_TRAIN, AVL_EVAL]` | enabled, composite | Dynamic K + C, highest throughput |

Compare: aggregations to 80% accuracy, final accuracy at round 300, K/C trace (from log parsing).

---

## 12. Known Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| K decreases too aggressively | Medium | `k_min` bound; `window ≥ 10` smooths noise; `update_every_n_aggs ≥ 5` |
| K oscillates up/down each step | Low-Medium | Rolling window in policies; optionally add a cooldown counter to controller |
| C exceeds actual eligible pool | Low | `EligibleEndsBasedPolicy` detects `n_eligible_train < C` and snaps C down |
| `dynamic_c` stale if channel reinitializes | Low | `dynamic_c` absent → selector falls back to `self.c`; first distribution after init uses static `self.c` until first controller step |
| `asyncfl_loop` condition misfires if K decreases mid-run | Low | `>=` guard in async gate (§8.1) prevents missed trigger |
| `task_eligible_states` misconfigured silently | Medium | Validate state strings at `__init__()` against `TrainerAvailState` enum; `ValueError` at startup before any training begins |
| FwdLLM `AVL_EVAL` ends assigned training tasks but don't support them at trainer level | Medium | This is a trainer-side concern — the trainer must handle both tasks; verify `FedSGDTrainer` responds correctly to `task_to_perform="train"` regardless of its own `avl_state`. (It does, since `avl_state` is a scheduler signal, not a capability flag.) |
| `n_eligible_train` metric inconsistent with `len(filtered_ends)` | Low | Both use the same `task_eligible_states` config with the same defaults. Validated by unit test §11.3 row 7. |

---

## 13. File Change Summary

### New Files

| File | Purpose |
|------|---------|
| `lib/python/flame/selector/dynamic_kc_policy.py` | Abstract base + all policy implementations (`NoOp`, `VarianceBased`, `StalenessBased`, `EligibleEndsBased`, `StepSchedule`, `Composite`) + `build_policy()` factory |
| `lib/python/flame/selector/dynamic_kc_controller.py` | `DynamicKCController` class |
| `lib/python/tests/selector/test_dynamic_kc_policy.py` | Policy unit tests |
| `lib/python/tests/selector/test_dynamic_kc_controller.py` | Controller unit tests |
| `lib/python/tests/selector/test_task_eligible_states.py` | Eligibility state config + metric consistency tests |

### Modified Files

| File | Changes |
|------|---------|
| `lib/python/flame/mode/horizontal/syncfl/fwdllm_aggregator.py` | `internal_init()`: add controller init + new counter attrs; `_process_aggregation_goal_met()`: add controller hook; new `_build_dynamic_kc_metrics()` (incl. eligible-ends computation); `_reset_agg_goal_variables()`: reset variance counters; async gate: `== → >=` |
| `lib/python/flame/selector/async_oort.py` | `__init__()`: read + validate `task_eligible_states`, store as `self._task_eligible_states`; `_handle_send_state()` filtering block: replace hardcoded state checks with configurable lookup; `select()`: read `dynamic_c` from `channel_props`; `enforce_min_start()`: accept `effective_c` param |
| `lib/python/flame/selector/async_random.py` | Same filtering block change as `async_oort.py` for consistency |
| `lib/python/examples/fwdllm/expts/run_tc_expts/json_scripts/aggregator.json` | Add `task_eligible_states` and `dynamic_kc` blocks to `selector.kwargs` for FwdLLM experiment configs |

### Files That Do NOT Need Changes

| File | Reason |
|------|--------|
| `oort.py` (sync oort selector) | Phase 1 dynamic K touches aggregator only |
| `syncfl/top_aggregator.py` | No changes needed |
| `asyncfl/top_aggregator.py` | No changes needed |
| `FedSgdAggregator.py` | `aggregate()` is K/C-agnostic; all routing is in parent |
| Trainer files | Trainers are unaffected by K and C; `avl_state` is a scheduler signal, not a capability constraint |

---

## Appendix A: Async Data Flow with Dynamic K, C, and Eligible States

```
Round N ─────────────────────────────────────────────────────────────────────────────────
 _distribute_weights_async():
   channel.ends(VAL_CH_STATE_SEND, "train")
     → AsyncOortSelector.select(ends, channel_props={..., "dynamic_c": C_n}, task="train")
         effective_c = channel_props["dynamic_c"]   = C_n
         _handle_send_state():
           filtered_ends = {e for e in ends if
             e.avl_state is None OR
             e.avl_state in self._task_eligible_states["train"]}
           feasible_extra = min(effective_c - len(selected_ends), len(filtered_ends))
           candidates = oort_select(filtered_ends, feasible_extra)
   → send weights to candidates

 ... trainers train asynchronously ...

 _aggregate_grads_async() [called per trainer message]:
   recv 1 message; _agg_goal_cnt += 1
   if _agg_goal_cnt >= K_n:
     _process_aggregation_goal_met(is_async=True)
       FedSGDAggregator.aggregate()    [variance check + model update]
       _model_version += 1
       _agg_goal_cnt = 0

       # Dynamic K/C update
       metrics = _build_dynamic_kc_metrics()
         metrics["n_eligible_train"] = count ends with avl ∈ task_eligible_states["train"]
         metrics["var_pass_rate"]    = _var_pass_count / _var_total_count
         metrics["avg_staleness"]    = mean(_per_round_staleness_list)
       new_k, new_c = controller.step(metrics)
       self._agg_goal = new_k                 # K_n+1
       channel.set_property("dynamic_c", new_c)  # C_n+1

Round N+1 ───────────────────────────────────────────────────────────────────────────────
 selector reads "dynamic_c" = C_n+1 from channel_props
 asyncfl_loop uses self._agg_goal = K_n+1
```

