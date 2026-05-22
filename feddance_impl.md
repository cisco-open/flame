# Technical Specification: FedDance Implementation

This document provides the technical requirements and architectural specification for reproducing **FedDance**, a participant selection framework for Federated Learning (FL) in dynamic environments.

---

## 1. Problem Formulation and Scope

### 1.1 Problem Statement
The goal is to solve a distributed optimization problem:  
$$\min_{w} F(w) = \sum_{m=1}^{M} p_m F_m(w)$$  
where $M$ is the total device pool, $p_m$ is the data weight of device $m$, and $F_m(w)$ is the local loss. The system must select a subset $N$ of participants per round to maximize global model accuracy and convergence speed under two constraints:
1.  **Dynamic Availability:** Devices are intermittently online ($D_m(v) \in \{0, 1\}$).
2.  **Training Dynamics:** Marginal returns of frequently selected devices diminish over time.

### 1.2 Inputs and Outputs
*   **Inputs (Server):**
    *   Binary availability logs for all $M$ devices ($D_m(v)$).
    *   Feedback from participants: Cumulative average training loss ($L_m$) and local training accuracy ($a_m$).
*   **Outputs (Server):**
    *   Participant set $\mathcal{S}$ for the current round $r$, where $|\mathcal{S}| = N$.

### 1.3 Assumptions

#### 1.3.1 Assumptions about Problem/Data
*   **Non-IID Distribution:** Data is partitioned using a Dirichlet distribution (default $\alpha=0.1$) to simulate heterogeneity.
*   **Unbiased Gradients:** Stochastic gradients from mini-batches are unbiased estimators of local gradients.
*   **Bounded Variance/Gradients:** Expected squared norm and variance of stochastic gradients are uniformly bounded.
*   **Objective Properties:** The loss function $F_m(w)$ is $\mu$-convex and $L$-smooth.

#### 1.3.2 Assumptions made by Implementation
*   **Poisson Distribution:** Device availability follows a Poisson process, making the future window a representative statistical sample of the device's pattern.
*   **Loss as Gradient Proxy:** Local training loss is a computationally cheap proxy for the gradient norm, indicating the importance of a device's data.
*   **Check-in Log Sufficiency:** The server can accurately track binary check-ins without needing explicit "offline" signals from devices.

---

## 2. Core Architecture and Component Responsibilities

### 2.1 Global Coordinator (Central Server)
The server maintains long-running processes for three calculation modules:

1.  **Device Availability Predictor:**
    *   **Responsibility:** Forecasts the probability $V_m(r)$ that a device will be active at least once in the next $K$ rounds.
    *   **Logic:** Updates arrival rate $\lambda_m$ using a sliding history window $K_h$.
2.  **Device Importance Estimator:**
    *   **Responsibility:** Quantifies the value of a device's data distribution using training loss $I_m(r)$.
    *   **Logic:** Uses the cumulative average loss of mini-batches from the *previous* round of engagement.
3.  **Accuracy Increment Calculator:**
    *   **Responsibility:** Measures the marginal return of a participant by tracking accuracy changes across its last $\beta$ engagements.
4.  **Participant Selector:**
    *   **Responsibility:** Integrates the three factors into a final Utility score $U_m$ and applies a Multi-Armed Bandit (MAB) style confidence bound for exploration.

### 2.2 Local Controller (Distributed Devices)
1.  **Device Status Tracer:** Reports binary availability (check-in) to the server.
2.  **Runtime Monitor:**
    *   **At each local iteration:** Records training loss.
    *   **At end of round:** Records local training accuracy.
    *   **Aggregation:** Computes the average loss across all $\tau$ local steps to send back as metadata with the model update.

---

## 3. Mathematical Objectives and Logic

### 3.1 Utility Factors
*   **Availability ($V_m$):** 
    $$V_m(r) = 1 - \exp(-\lambda_m(r) \cdot K)$$
    Where $\lambda_m(r)$ is the arrival rate over history window $K_h$:  
    $$\lambda_m(r) = \frac{\sum_{v=r-K_h}^{r-1} D_m(v)}{K_h}$$
*   **Importance ($I_m$):** Average loss over $\tau$ local steps in the last engaged round:
    $$I_m(r) = \frac{1}{\tau |\xi_m|} \sum_{u=(r-1)\cdot \tau}^{r\cdot \tau - 1} \sum_{\xi \in \xi_m^{(u)}} f(w_m^{(t)}, \xi)$$
*   **Accuracy Increment ($A_m$):** Slope of accuracy over $\beta$ rounds:
    $$A_m(r) = \frac{1}{\beta - 1} (a_{m,\beta}(r) - a_{m,1}(r))$$

### 3.2 Integrated Utility with Exploration
The selector chooses devices with the **top-N** scores based on:
$$U_m(r) = \left( V_m(r) \cdot I_m(r) \cdot A_m(r) \right) \cdot \left( 1 + \frac{\log_{10}(R+1)}{10(1 + J_m)} \right)$$
*   $R$: Global round counter.
*   $J_m$: The index of the last round in which device $m$ was involved.

---

## 4. Implementation Details

### 4.1 Data Flow and Training Loop
1.  **Check-in:** Server updates $\lambda_m(r)$ for all online devices $P$.
2.  **Cold Start Handling:** For devices never selected before, use the average $I$ and $A$ from the *previous* round's participants.
3.  **Selection:** Calculate $U_m$ for all $m \in P$ and pick top-$N$.
4.  **Local Training:** Participants execute $\tau$ steps of SGD. Runtime Monitor collects loss and final accuracy.
5.  **Feedback:** Participants upload model weights + average loss + final accuracy.
6.  **Global Update:** Server aggregates weights and updates utility state variables.

### 4.2 Hyperparameters (Explicitly Stated)
| Parameter | Value | Description |
| :--- | :--- | :--- |
| **$N$** | 10 (or up to 50) | Number of participants per round |
| **$K$** | 5 | Future availability prediction window |
| **$K_h$** | 50 | History window for arrival rate $\lambda$ |
| **$\beta$** | 5 | Window for accuracy increment calculation |
| **$\alpha$** | 0.1 | Dirichlet parameter for Non-IID data partitioning |
| **$\tau$** | 1 or 4 | Local update steps (dataset dependent; see Table 1) |
| **Pool Size** | 1000 | Candidate devices |

### 4.3 Implementation-Sensitive Details
*   **Availability Threshold:** Predictor classifies a device as "available" for binary metrics if the probability $V_m > 0.5$ (only used for evaluation of the predictor, not the selection itself).
*   **Byproduct Loss:** Loss must be collected *during* training to avoid a separate forward pass over the dataset, ensuring $\mathcal{O}(1)$ overhead.
*   **Isolation:** Clients should be implemented using `torch.multiprocessing` to ensure resource isolation.

---

## 5. Summary of Reproducibility Factors

### 5.1 Minimum Viable Baseline (MVB)
1.  Implement the Poisson arrival rate formula using a simple global list of binary check-ins.
2.  Modify the standard FL aggregation to return average loss and final training accuracy.
3.  Implement the base Utility function $U_m = V_m \cdot I_m \cdot A_m$ (excluding exploration) and select top-$N$.

### 5.2 Full-Fidelity Requirements
1.  **Exploration Term:** Must include the MAB-inspired confidence bound (log term in Eq 8) to prevent starvation.
2.  **Cold-Start Logic:** Must implement the "average substitution" for $I_m$ and $A_m$ for new devices.
3.  **Availability Traces:** Use the specific real-world trace data (battery/WLAN) rather than random noise to simulate $D_m(v)$.

### 5.3 Implementation Risks and Pitfalls
*   **Stale Metrics:** If a device isn't selected for many rounds, its $I_m$ and $A_m$ become stale. The exploration term is critical to refresh these.
*   **Negative Accuracy Increments:** The paper calculates accuracy *increments*. In early stages or with highly non-IID data, accuracy might fluctuate. Ensure the logic handles $A_m \leq 0$ without breaking the multiplication (Reasonable Inference: The paper likely takes absolute increments or clips them).
*   **Communication Overhead:** Unlike REFL, FedDance avoids pre-selection queries. Ensure metadata (loss/accuracy) is bundled with the model weights to maintain this efficiency.

### 5.4 Recommended Implementation Order
1.  **Telemetry:** Update client-side monitors to collect loss/accuracy and server-side to track check-in history.
2.  **Predictor:** Implement the Poisson-based $V_m$ logic.
3.  **Basic Utility:** Integrate $V_m, I_m, A_m$ into the selection loop.
4.  **Refinement:** Add exploration terms and cold-start logic.
5.  **Environment:** Set up the Dirichlet data partitioner and availability trace emulator.

### 5.5 Unanswered Questions
*   **Handling $A_m \leq 0$:** If a device's accuracy decreases, $A_m$ becomes negative, which would flip the sign of the entire utility $U_m$ or make it negative. The paper does not specify if they use `ReLU(A_m)` or an absolute value.
*   **Aggregation Weighting:** While the selection is $U_m$-based, the paper mentions "general averaging schemes" for aggregation but doesn't explicitly state if $q_m$ in Eq 10 is simple $1/N$ or weighted by utility/data size.
*   **Initialization of $\lambda_m$:** For the very first round ($r=0$), the arrival rate has no history. The paper doesn't specify the initial prior for $\lambda_m$.

---

## 6. Phased Implementation Plan in Flame Codebase

This section lays out a concrete, phased plan to port FedDance into the Flame repository while reusing as much existing scaffolding as possible (Oort, REFL_Oort, REFL availability tracker). Each phase is independently testable and ends in a runnable, regression-safe state.

### 6.0 Locked Decisions (user-confirmed)

These choices are settled and drive the rest of the plan:

1. **Cleanup scope:** Full §7 cleanup is in. Phase 0 strips debug-trainer IDs, formalizes `AbstractSelector` lifecycle hooks, removes `hasattr` guards, deletes dead code, dedupes imports, **and** starts a real selector test suite. FedDance lands on a clean foundation.
2. **Test layout:** New `lib/python/tests/conftest.py` with shared fixtures (mock `End`, mock `channel_props`). Selector tests stay under `tests/selector/` but draw from the shared conftest. Pytest runner.
3. **Parent class:** `FedDanceSelector(AbstractSelector)` — composition over inheritance. No subclassing `OortSelector`. We reimplement the small bits of utility-list/top-N logic FedDance needs rather than risk inheriting Oort's UCB term.
4. **Negative $A_m$ handling:** Default `"relu"` — clamp to `max(A_m, 1e-6)`. Still configurable via `negative_accuracy_handling` kwarg (`"relu" | "abs" | "raw"`), but `"relu"` is the default and what we test against.

### 6.1 Mapping FedDance Concepts to Flame Primitives

| FedDance concept                | Flame primitive (existing)                                 | Action                                                  |
| :------------------------------ | :--------------------------------------------------------- | :------------------------------------------------------ |
| Participant Selector            | `AbstractSelector` (`selector/__init__.py`)                | New subclass `FedDanceSelector`                         |
| Importance $I_m$ (loss proxy)   | `MessageType.STAT_UTILITY` + `Trainer.oort_loss()`         | Reuse as-is; treat `stat_utility` as $I_m$              |
| Accuracy Increment $A_m$        | (none)                                                     | Add `MessageType.LOCAL_ACCURACY`; trainer reports $a_m$ |
| Availability $V_m$ (Poisson)    | `REFLAvailabilityTracker` (trace ingestion)                | New `FedDancePredictor` consumes binary check-ins       |
| Real-world traces $D_m(v)$      | `availability/refl_tracker.py` (pickle/YAML loaders)       | Reuse loader; binarize to check-in events               |
| Top-N + UCB exploration         | `OortSelector.calculate_total_utility` + exploration term  | Re-implement Eq. 8 inline; keep `selected_ends` plumbing |
| Aggregation                     | `OptimizerType.FEDAVG` (`optimizer/fedavg.py`)             | Reuse unchanged (FedDance is selector-only per §5)      |
| In-flight tracking / overcommit | `selected_ends`, `_cleanup_recvd_ends`, `ordered_updates_recv_ends` | Reuse pattern from `OortSelector`                     |
| Config plumbing                 | `SelectorType` enum + `selector_provider.register(...)`    | Add `FEDDANCE = "feddance"`                              |

**Parity rule:** any logic shared with Oort/REFL_Oort (selected_ends bookkeeping, exploration scaling, cleanup hooks, blocklist semantics) should call the same helpers — do not fork.

### 6.2 Phase 0 — Telemetry Plumbing (foundation, no behavior change)

**Goal:** Get $a_m$ on the wire without changing any existing selector's behavior. This is a pure additive change that other selectors (Oort, REFL_Oort, Async_Oort, FedBuff, Random) ignore.

1. **Add wire field.** In `lib/python/flame/mode/message.py`:
   ```python
   LOCAL_ACCURACY = 33  # local training accuracy reported by trainer (FedDance A_m)
   ```
2. **Trainer-side capture.** In `lib/python/flame/mode/horizontal/syncfl/trainer.py`:
   * Add `self._local_accuracy: float = 0.0` next to `self._stat_utility`.
   * In the training loop (where loss is computed), accumulate correct-predictions / total-predictions over $\tau$ local steps.
   * Reset via a `reset_local_accuracy()` analogous to `reset_stat_utility()`.
   * In `_send_weights()`, add `MessageType.LOCAL_ACCURACY: self._local_accuracy` to the outgoing dict.
3. **Aggregator-side capture.** In `lib/python/flame/mode/horizontal/syncfl/top_aggregator.py::_aggregate_weights()`:
   * Mirror the `STAT_UTILITY` handling block — store as a new end property `PROP_LOCAL_ACCURACY` via `channel.set_end_property(end, PROP_LOCAL_ACCURACY, msg[MessageType.LOCAL_ACCURACY])`.
   * Define `PROP_LOCAL_ACCURACY = "local_accuracy"` constant near `PROP_STAT_UTILITY`.

**Validation:** Run an existing Oort experiment; confirm no behavior delta. Inspect logs for `LOCAL_ACCURACY` field presence.

**Files touched:** `mode/message.py`, `mode/horizontal/syncfl/trainer.py`, `mode/horizontal/syncfl/top_aggregator.py`.

### 6.3 Phase 1 — Availability Predictor Module

**Goal:** Standalone, unit-testable Poisson predictor that ingests binary check-ins and emits $V_m(r) = 1 - e^{-\lambda_m K}$.

1. Create `lib/python/flame/availability/feddance_predictor.py`:
   * Class `FedDancePredictor` with:
     * `__init__(history_window=50, prediction_window=5, lambda_init=0.0)` (per §4.2: $K_h{=}50$, $K{=}5$).
     * `record_checkin(end_id: str, round_num: int)` — appends to a per-device sliding deque of length $K_h$.
     * `lambda_m(end_id, round_num)` — sum of check-ins in `[round - K_h, round - 1]` divided by $K_h$.
     * `V_m(end_id, round_num)` — Poisson formula from §3.1.
     * Optional `record_from_trace(trace: dict)` adapter that reuses `REFLAvailabilityTracker`'s loaders to seed history from pickle/YAML traces (§5.2 full-fidelity item 3).
2. Cold-start prior for $\lambda_m$ (resolves Unanswered Question §5.5): default to the **population mean $\lambda$** from observed history; if none yet, return $V_m = 0.5$ so unseen devices are not aggressively de-prioritized. Make this configurable via `lambda_cold_start_strategy` kwarg.

**Validation:** Unit tests with synthetic check-in series (always-on, always-off, periodic). Assert $V_m \approx 1.0$ for always-on, $\approx 0$ for always-off, monotonic in $\lambda_m$.

**Files touched:** new `availability/feddance_predictor.py`; possibly extend `availability/__init__.py` to export it.

### 6.4 Phase 2 — Minimum Viable Baseline (MVB) Selector

**Goal:** A working `FedDanceSelector` that implements §5.1 MVB: $U_m = V_m \cdot I_m \cdot A_m$, top-N, no exploration term yet.

1. **Enum + registration.**
   * `config.py`: add `FEDDANCE = "feddance"` to `SelectorType`.
   * `selectors.py`: import `FedDanceSelector`; `selector_provider.register(SelectorType.FEDDANCE, FedDanceSelector)`.
2. **Create `lib/python/flame/selector/feddance.py`** subclassing `AbstractSelector` (NOT `OortSelector`, to avoid inheriting Oort's UCB term unintentionally; we re-use specific helpers by composition):
   * Constructor reads kwargs: `aggr_num` (N), `history_window` (K_h), `prediction_window` (K), `accuracy_window` (β), `availability_trace_file`, `trainer_registry_file`, plus optional `overcommitment` (default 1.0 — sync FedDance does not require overcommitment per §4.1, but allow it for parity with Oort plumbing).
   * Initializes:
     * `self.predictor = FedDancePredictor(...)`
     * `self.accuracy_history: dict[end_id, deque(maxlen=β)]`
     * `self.last_engaged_round: dict[end_id, int]` — for $J_m$ and for the §4.1 "previous round of engagement" rule.
     * `self.last_loss: dict[end_id, float]` — cumulative-avg loss reported by trainer in its last engaged round.
     * `self.selected_ends`, `self.ordered_updates_recv_ends` — same plumbing as `OortSelector` (copy the patterns at `selector/oort.py:91-97`, `selector/oort.py:695-770`).
   * `select(ends, channel_props, trainer_unavail_list, task_to_perform, **kwargs)`:
     1. Filter eligible ends (exclude in-flight, exclude `trainer_unavail_list`, exclude blocklist) — mirror `OortSelector.select` lines 212-245.
     2. For each present end_id, record check-in: `self.predictor.record_checkin(end_id, round)`.
     3. Compute per-end $V_m, I_m, A_m$:
        * $V_m$ from predictor.
        * $I_m$: pull from `ends[end_id].get_property(PROP_STAT_UTILITY)`; if `None`, mark as cold-start.
        * $A_m$: slope from `accuracy_history[end_id]` if `len >= 2` else cold-start.
     4. Cold-start substitution (§5.2 item 2): replace cold-start $I_m, A_m$ with the **mean** $I, A$ of the previous round's participants (cache `prev_round_mean_I`, `prev_round_mean_A` after every successful aggregation).
     5. $U_m = V_m \cdot I_m \cdot A_m$; pick top-$N$.
     6. Update `self.selected_ends`, increment `PROP_SELECTED_COUNT`, set `PROP_LAST_SELECTED_ROUND` (J_m).
3. **Aggregator integration for $A_m$ updates.** When `_aggregate_weights()` receives `LOCAL_ACCURACY`, the selector needs to be notified to push the value into its `accuracy_history`. Two options:
   * **(Preferred, mirrors REFL_Oort pattern)** Add an `on_update_received(end_id, msg)` hook the aggregator calls in `_aggregate_weights` (we already have a similar pattern at [top_aggregator.py:285-292](lib/python/flame/mode/horizontal/syncfl/top_aggregator.py#L285-L292) where it does `channel._selector.ordered_updates_recv_ends.append(end)`). Extend that block to also call `channel._selector.record_round_result(end_id, accuracy, loss, round)` if the method exists (use `hasattr` to keep it opt-in and non-breaking for other selectors).
   * **(Alt)** Have the selector read end properties lazily inside `select()`. Simpler but couples selector to property timing. Prefer option 1.
4. **Compute and cache previous-round means** in the aggregator hook so cold-start substitution works on round $r+1$.

**Validation:** Run with 10 trainers, MVB only. Verify: (a) every selected end's $U_m$ logged; (b) cold-start ends use mean substitution on round 1; (c) selection set stable across two `select()` calls in same round (distribute then aggregate) — same caching pattern as [refl_oort.py:157-159](lib/python/flame/selector/refl_oort.py#L157-L159).

**Files touched:** `config.py`, `selectors.py`, new `selector/feddance.py`, `mode/horizontal/syncfl/top_aggregator.py` (hook only).

### 6.5 Phase 3 — Full-Fidelity Utility (Exploration Term)

**Goal:** Add the MAB-style log term from Eq. 8 — required by §5.2 item 1.

1. In `FedDanceSelector.select()`, after MVB $U_m$ is computed, multiply by:
   ```
   1 + log10(R + 1) / (10 * (1 + J_m))
   ```
   where `R = round_num` and `J_m = self.last_engaged_round.get(end_id, 0)`.
2. **Resolve §5.5 ambiguity for $A_m \leq 0$.** Make the policy configurable via `negative_accuracy_handling` kwarg:
   * `"relu"` (default, safest): clamp $A_m \to \max(A_m, \epsilon)$ with a small $\epsilon$ (e.g. `1e-6`) so $U_m$ never collapses or flips sign.
   * `"abs"`: take $|A_m|$ — keeps magnitude info.
   * `"raw"`: pass through unmodified (paper-faithful but risky).
   Log which strategy is active.
3. Track `last_engaged_round[end_id] = round_num` only **after** an update is received (in the `on_update_received` hook), not at selection time, so $J_m$ reflects actual engagement, not just selection (paper text: "the last round in which device $m$ was involved").

**Validation:** With a small pool, verify that a device which has been idle for many rounds gets its utility lifted by the log term, and that no $U_m$ goes negative under default `relu` mode.

**Files touched:** `selector/feddance.py`.

### 6.6 Phase 4 — Availability Trace Integration

**Goal:** Drive $D_m(v)$ from real-world traces, not random availability — §5.2 item 3.

1. `FedDancePredictor.record_from_trace(trace_dict)` walks each device's trace, computes a binary check-in per round bucket, and seeds the sliding window.
2. `FedDanceSelector.__init__` calls this once at startup using paths from kwargs (`availability_trace_file`, `trainer_registry_file`) — same kwargs already used by `REFLOortSelector` so we can reuse existing trace YAMLs under `examples/async_cifar10/metadata/availability_traces/`.
3. `trainer_unavail_list` passed into `select()` already lists currently-offline devices (computed by `top_aggregator.get_curr_unavail_trainers()` from trace events). Use it to:
   * Exclude offline devices from selection (consistent with REFL_Oort).
   * Still record their *prior* check-ins via the predictor so $V_m$ reflects history.

**Files touched:** `selector/feddance.py`, `availability/feddance_predictor.py`.

### 6.7 Phase 5 — Example Config & Reproducibility Polish

**Goal:** A runnable example that mirrors paper hyperparameters (§4.2) and a smoke test.

1. Add `lib/python/examples/feddance_cifar10/` (mirroring `async_cifar10/` structure):
   * `aggregator_config.json` with:
     ```json
     "selector": {
       "sort": "feddance",
       "kwargs": {
         "aggr_num": 10,
         "history_window": 50,
         "prediction_window": 5,
         "accuracy_window": 5,
         "negative_accuracy_handling": "relu",
         "availability_trace_file": "metadata/availability_traces/synthetic_traces.yaml",
         "trainer_registry_file": "metadata/trainer_registry.yaml"
       }
     },
     "optimizer": { "sort": "fedavg", "kwargs": {} }
     ```
   * `hyperparameters.rounds`, `epochs=1`, `aggGoal=10`, Dirichlet $\alpha=0.1$ (set via datasampler kwargs).
2. Add a post-processing parity check script that compares per-round selection diversity vs. an Oort baseline run.
3. Update `README` / docs (only if the user asks).

**Files touched:** new files under `examples/feddance_cifar10/`.

### 6.8 Cross-Phase: Parity & Anti-Drift Checklist

To ensure "utmost similarity and parity" with existing selector machinery:

- [ ] `selected_ends` is a `set`, initialized in `__init__`, populated like [oort.py:311-315](lib/python/flame/selector/oort.py#L311-L315).
- [ ] `_cleanup_recvd_ends(ends)` implemented identically in shape to [oort.py:695-770](lib/python/flame/selector/oort.py#L695-L770).
- [ ] `ordered_updates_recv_ends` populated by the aggregator's existing hook at [top_aggregator.py:285-292](lib/python/flame/mode/horizontal/syncfl/top_aggregator.py#L285-L292) (no new hook plumbing required there).
- [ ] `select()` is idempotent within a round (distribute/aggregate both call it) — cache `newly_selected_this_round` exactly like [refl_oort.py:157-159](lib/python/flame/selector/refl_oort.py#L157-L159).
- [ ] `PROP_*` constants reused from `selector/oort.py` where possible; introduce `PROP_LOCAL_ACCURACY`, `PROP_LAMBDA_M`, `PROP_V_M` only where new state is needed.
- [ ] All FedDance-specific kwargs documented in the selector docstring and default-safe (so an under-specified config doesn't crash — falls back to MVB behavior).
- [ ] Logging conventions match: `[FEDDANCE_SELECT]`, `[FEDDANCE_DEBUG]`, `[FEDDANCE_FILTER]` prefixes mirroring Oort/REFL_Oort styles.
- [ ] No changes to Oort/REFL_Oort/AsyncOort/FedBuff selectors; FedDance lives alongside them.

### 6.9 Risks & Open Items to Confirm Before Coding

1. **β=5 window vs. fast-changing accuracy early on.** First $\beta - 1$ rounds will have insufficient $A_m$ history — confirm we handle this via cold-start substitution (we do).
2. **Cold-start mean from previous round** assumes the previous round actually produced metrics. For round $r=1$ specifically, fall back to `I_m = 1.0`, `A_m = 1.0` (or a configurable prior) — document explicitly.
3. **Sync vs async parity.** All references in `top_aggregator.py` are syncfl. If the user later wants async FedDance, we'd subclass for `async_oort.py`-style version-state handling — out of scope for this plan.
4. **Aggregator weighting (§5.5 unanswered).** Keep FedAvg as-is (data-size weighted). If the paper meant uniform $1/N$, that's a one-line change in the optimizer config later.

### 6.10 Recommended Build Order (Executable Checklist)

1. Phase 0 — telemetry plumbing → run any existing experiment to confirm no regression.
2. Phase 1 — predictor + unit tests.
3. Phase 2 — MVB selector → smoke test with N=10, pool=100, traces optional.
4. Phase 3 — exploration term + negative-A handling → comparison run against Phase 2 to confirm exploration lifts long-idle devices.
5. Phase 4 — trace integration → run with `synthetic_traces.yaml` from existing REFL example.
6. Phase 5 — example config + parity check vs. Oort baseline.

---

## 7. Codebase Audit: Sub-Optimal Patterns To Fix Inline

While planning FedDance, an audit of `selector/oort.py`, `selector/refl_oort.py`, and `mode/horizontal/syncfl/top_aggregator.py` surfaced patterns that should be cleaned up *now*, before we extend them. Inheriting these into FedDance would propagate the smell. Each fix lives in the phase noted; nothing here is speculative refactor — every item is touched by FedDance work anyway.

### 7.1 Issue: Hardcoded Debug Trainer IDs in Hot Paths

**Where:** `selector/oort.py:294-299`, `selector/refl_oort.py:186-202, 230-248, 363-386`, `mode/horizontal/syncfl/top_aggregator.py:294-299, 387-405`.

**Symptom:** Blocks like `test_trainer_id = "505f9fc483cf4df68a2409257b5fad7d3c580389"` and `if end == test_trainer_id: logger.warning(...)` are baked into production selection/aggregation paths. They were clearly added to chase a specific bug and never removed.

**Fix (Phase 0):** Strip all `[DEBUG_389_*]`, `[DEBUG_411_*]`, `[TRACK_SELECT]`, `[BUG_FOUND]` blocks from `top_aggregator.py`, `oort.py`, `refl_oort.py`. Keep the structural log lines (e.g. `[FILTER_DEBUG]`, `[CLEANUP_DEBUG]`, `[SELECTION_SUMMARY]`) but at `debug` level rather than `info`/`warning`.

**Why now:** We'd otherwise copy this pattern into `feddance.py`. Better to land FedDance with no debug-trainer hardcoding from day one.

### 7.2 Issue: Brittle Private-Attribute Coupling Between Aggregator and Selector

**Where:** `top_aggregator.py:285-292` and `top_aggregator.py:339-347`.

**Symptom:**
```python
if hasattr(channel, "_selector") and hasattr(channel._selector, "ordered_updates_recv_ends"):
    channel._selector.ordered_updates_recv_ends.append(end)
...
if hasattr(channel, "_selector") and hasattr(channel._selector, "_cleanup_recvd_ends"):
    channel._selector._cleanup_recvd_ends(channel._ends)
```
This reaches into private attributes (`_selector`, `_ends`) and gates on `hasattr` — an implicit duck-typed contract. Any selector that wants to participate has to expose the same private surface. FedDance would need the same plumbing; doing so would deepen the smell.

**Fix (Phase 0, prerequisite to Phase 2):** Promote this to a real lifecycle contract on `AbstractSelector`:

```python
# selector/__init__.py
class AbstractSelector(ABC):
    def __init__(self, **kwargs):
        ...
        self.selected_ends: set[str] = set()
        self.ordered_updates_recv_ends: list[str] = []

    def on_update_received(self, end_id: str, msg: dict, round_num: int) -> None:
        """Hook called by aggregator when a trainer update arrives.
        Default: record into ordered_updates_recv_ends for in-flight bookkeeping.
        Subclasses may override to extract per-update metrics (e.g. FedDance
        records LOCAL_ACCURACY into accuracy_history)."""
        self.ordered_updates_recv_ends.append(end_id)

    def on_round_completed(self, ends: dict, round_num: int) -> None:
        """Hook called by aggregator after aggregation; selectors free in-flight ends."""
        # Default: pop everyone in ordered_updates_recv_ends out of selected_ends
        for end_id in self.ordered_updates_recv_ends:
            self.selected_ends.discard(end_id)
        self.ordered_updates_recv_ends.clear()
```

Then `top_aggregator._aggregate_weights()` simplifies to:
```python
if channel.selector is not None:
    channel.selector.on_update_received(end, msg, self._round)
...
if channel.selector is not None:
    channel.selector.on_round_completed(channel._ends, self._round)
```

And `OortSelector._cleanup_recvd_ends` becomes a `super().on_round_completed(...)` override that adds its own logging.

**Side effect:** Expose `channel.selector` as a public property if not already (check `channel.py` during Phase 0). This removes ~5 lines of `hasattr` boilerplate from every future selector.

**Why now:** FedDance needs its own `on_update_received` to capture `LOCAL_ACCURACY` and feed `accuracy_history`. Without the hook, we'd add yet another `hasattr` block in `_aggregate_weights`.

### 7.3 Issue: Defensive `hasattr(self, 'selected_ends')` Proliferation

**Where:** `selector/oort.py:96-97, 169, 204, 212, 314, 328, 718`, `selector/refl_oort.py:94-101, 157, 169, 201, 328, 336`.

**Symptom:** Both selectors guard `self.selected_ends` accesses with `hasattr` "just in case." But `AbstractSelector.__init__` already initializes it ([selector/__init__.py:36](lib/python/flame/selector/__init__.py#L36)). The guards exist because at some point the attribute initialization was uncertain, and the guards were never removed.

**Fix (Phase 0):** Remove all `hasattr(self, 'selected_ends')` checks. Ensure `AbstractSelector.__init__` is always called by subclasses via `super().__init__(**kwargs)` — it is, in all current selectors. Add a test in the new selector test suite asserting the invariant.

**Why now:** FedDance would inherit this defensive pattern; we want a clean baseline.

### 7.4 Issue: Latent Bug in `OortSelector.remove_from_selected_ends`

**Where:** `selector/oort.py:772-796`.

**Symptom:**
```python
def remove_from_selected_ends(self, ends, end_id):
    selected_ends = self.selected_ends[self.requester]  # TypeError: set is not subscriptable
```
`self.selected_ends` is a `set` (line 97), not a dict, so this raises `TypeError` if ever called. Reference search shows no callers in-tree — dead code.

**Fix (Phase 0):** Delete the method. If a real use case emerges, re-add against the actual set semantics.

### 7.5 Issue: Duplicate Imports

**Where:** `selector/oort.py:23, 25` — `import numpy as np` twice.

**Fix (Phase 0):** Delete the duplicate.

### 7.6 Issue: No End-to-End Selector Tests

**Where:** `lib/python/tests/selector/` has 4 policy-level unit tests (dynamic_kc_*, adaptive_k_*, task_eligible_states) but **zero** tests exercising the full `AbstractSelector.select()` contract for Random, Default, FedBuff, Oort, AsyncOort, or REFL_Oort.

**Symptom:** A change to `AbstractSelector` (e.g. the hook contract in §7.2) has no automated safety net. Selector regressions are caught only at experiment runtime.

**Fix (Phase 0 — establish the suite; Phase 2 — add FedDance tests):** Create `lib/python/tests/selector/test_selector_contract.py` with a parameterized test class that exercises every registered selector against:

1. **Empty ends** → `select()` returns `{}` without raising.
2. **`num_ends == num_to_select`** → returns all ends.
3. **`num_ends > num_to_select`** → returns exactly `num_to_select` ends.
4. **Idempotency within a round** → second `select()` with same `round` returns same set.
5. **`selected_ends` is always a `set` of strings** post-call.
6. **`on_update_received` hook (after §7.2 lands)** populates `ordered_updates_recv_ends`.
7. **`on_round_completed` hook** clears in-flight set.

Then add per-selector tests:

- `test_random_selector.py` — sampling diversity, seed reproducibility.
- `test_default_selector.py` — returns all ends as-is.
- `test_oort_selector.py` — cold-start random selection on round 1; utility-based selection on round 2+; exploration factor decay.
- `test_refl_oort_selector.py` — priority-fill mode with mock availability tracker.
- `test_feddance_selector.py` (added in Phase 2) — see §6.4.

Use the existing pattern (mock heavy deps via `__new__` + `with patch.object(Cls, "__init__", lambda self, **kw: None)`) for fast tests, plus a small integration test that constructs a real selector with fake `End` objects.

**Why now:** Phase 0 changes (`AbstractSelector` hook contract, hasattr cleanup, debug-id stripping) **need** safety nets before they ship. Writing the contract suite first protects every selector simultaneously, not just FedDance.

### 7.7 Issue: Inconsistent `task_to_perform` Parameter Passing

**Where:** `selector/oort.py:163-169` declares `task_to_perform` as the 4th positional parameter; `selector/refl_oort.py:110-117` does the same; but `AbstractSelector.select()` signature is `select(self, ends, channel_props)` — only 2 positional args. Subclasses extended the signature without updating the base.

**Fix (Phase 0):** Update `AbstractSelector.select()` signature to formally include `task_to_perform: str = "train"` and `trainer_unavail_list: list = None` as keyword-only args with documented defaults. Subclasses that don't need them ignore via `**kwargs`. This makes the contract explicit and lets new selectors (FedDance) know what they receive.

### 7.8 Summary of Phase 0 Expansion

Phase 0 (originally just telemetry) now also covers a one-time hygiene pass. Concrete file changes in Phase 0:

| File | Change |
| :--- | :--- |
| `mode/message.py` | Add `LOCAL_ACCURACY = 33` |
| `mode/horizontal/syncfl/trainer.py` | Track and send `_local_accuracy` |
| `mode/horizontal/syncfl/top_aggregator.py` | (a) Read `LOCAL_ACCURACY` to end property; (b) replace private-attr coupling at lines 285-292 and 339-347 with `selector.on_update_received` / `on_round_completed` calls; (c) strip `[DEBUG_389_*]` blocks. |
| `selector/__init__.py` (`AbstractSelector`) | (a) Initialize `ordered_updates_recv_ends`; (b) add `on_update_received` and `on_round_completed` no-op hooks; (c) formalize `select()` signature with `task_to_perform`, `trainer_unavail_list` kwargs. |
| `selector/oort.py` | (a) Delete duplicate `numpy` import; (b) delete `remove_from_selected_ends`; (c) remove all `hasattr(self, 'selected_ends')` guards; (d) override `on_round_completed` to call base + log; (e) strip `[DEBUG_389_*]`. |
| `selector/refl_oort.py` | (a) Remove `hasattr` guards; (b) strip `[DEBUG_389_*]` and `[TRACK_SELECT]` blocks; (c) override `on_update_received` if it needs accuracy/loss capture (it doesn't — keep base default). |
| `tests/selector/test_selector_contract.py` | New: parameterized contract tests for all registered selectors. |
| `tests/selector/test_oort_selector.py` | New: Oort-specific tests. |
| `tests/selector/test_refl_oort_selector.py` | New: REFL_Oort-specific tests. |
| `tests/selector/test_random_selector.py` | New: Random selector tests. |
| `tests/selector/conftest.py` | New: shared fixtures (mock `End`, mock `channel_props`). |

After this, Phase 1–5 work as originally described, with FedDance overriding `on_update_received` cleanly instead of inheriting a duck-typed mess.

### 7.9 Explicit Non-Goals for This Cleanup

To keep the diff scoped:

- **Not refactoring** `AsyncOortSelector` (1966 lines) — too risky without async test coverage. Leave its `hasattr` patterns alone for now; just adopt the new `AbstractSelector` hooks where it already touches the aggregator path.
- **Not modifying** `FedBuffSelector`, `AsyncRandomSelector`, `DefaultSelector` beyond the `AbstractSelector` signature update — they're not on the FedDance critical path.
- **Not changing** the trainer-side `oort_loss` mechanism — `_stat_utility` remains the loss proxy.
- **Not introducing** a new optimizer for FedDance — FedAvg stays.

---

## 8. Config Format Decision: JSON Runtime + Optional YAML Descriptor

### 8.1 What exists today

- **Sync example (`examples/cifar10/`)**: only JSON runtime configs (`aggregator/config.json`, `trainer/config.json`). No YAML.
- **Async example (`examples/async_cifar10/`)**:
  - Runtime: JSON (under `experiments/run_*/aggregator_config.json` etc.)
  - Launcher-level descriptors: YAML (under `experiments/configs/`, `expt_scripts_2026/`) — these are *not* loaded by aggregator/trainer; they're consumed by `launch/run_experiment.py` which generates the JSON.

### 8.2 Decision for FedDance

- **Phase 5 deliverable:** `examples/feddance_cifar10/` with **JSON runtime configs only**, mirroring the sync `examples/cifar10/` layout (`aggregator/config.json`, `trainer/config.json`). This is the minimal runnable example and matches the sync nature of FedDance per §1.2.
- **Optional Phase 5b** (skip unless requested): if grid sweeps are desired, add a YAML experiment descriptor under `examples/feddance_cifar10/experiments/configs/feddance_baseline.yaml` and wire it into the existing `async_cifar10/launch/` tooling (or a new sync launcher). This is *additive* and not required for the baseline.