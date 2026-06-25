# High-Fidelity Simulator for FwdLLM — Staged Parity Plan

Pickup-ready design for porting the **high-fidelity sim + parity checker** to the `fwdllm` example
across the **fluxtune / fwdllm / fwdllm++** baselines.

**Prerequisites (read first):** [async_cifar10/PARITY.md](../async_cifar10/PARITY.md) — the proven
methodology this plan reuses wholesale (the ladder §1–§2, role/tier tags, dependency gating, the
landed sim mechanisms in §3, the workflow policy + run-length budget). This doc only states the
**fwdllm deltas**; everything not redefined here is inherited from PARITY.md unchanged.

**Scope of this round:** 100% trainer availability for ALL trainers. Unavailability is deferred and
follows [async_cifar10/UNAVAILABILITY_DESIGN.md](../async_cifar10/UNAVAILABILITY_DESIGN.md) as its
template (per-baseline source of truth, withhold-then-deliver); §F.7 keeps the `track_trainer_avail`
seam wired-but-inert so it turns on later without rework. Assumes the launcher-script port (same shape
as async_cifar10's `debug_run.sh` + `expt_scripts_2026/*.yaml`) has landed.

**Decisions locked (kickoff):**
- A trusted **real** wall-clock run is the reference per baseline (as in async_cifar10). Parity ≠
  tuning sim to real; real must pass `validate_real` admissibility first (§F.6).
- The parity checker **extends the shared `async_cifar10/scripts/parity` engine** — fwdllm rungs are
  *added*, the engine is reused.
- The fluxtune / fwdllm / fwdllm++ → config mapping is **user-supplied** (§C, the one blocking gap).

---

## §A  Already done & directly reusable (start here)

The high-fidelity sim is a **flame-core** capability. FwdLLM's aggregator inherits **AsyncTopAgg**, so
most machinery is already *on the class* — the work is **wiring it into fwdllm's gradient loop and
trainer**, not re-implementing it.

### A.1  Aggregator spine (inherited, baseline-agnostic)
`Role → syncfl.TopAggregator → asyncfl.TopAggregator (AsyncTopAgg) → syncfl.fwdllm_aggregator →
examples/fwdllm FedSGDAggregator`. Already on the instance (from `asyncfl/top_aggregator.py`):

| Mechanism | Status for fwdllm |
|---|---|
| Virtual clock `_vclock`, `_advance_sim_clock` (`vclock = max(vclock, sct)`) | inherited; **NOT yet driven** by the grad loop |
| §3.drain sct-ordered drain `_sim_recv_min` | inherited; **bypassed** (fwdllm uses raw `recv_fifo`) |
| §3.resid one-in-flight residence `_sim_hold_busy_slots` | inherited; **bypassed** |
| §4.5/§4.9 sct-gated pool exclusion / carry-over (oort overlay) | available if a baseline uses oort selection |

### A.2  Selector + duration signal (already wired)
- §S.dur duration helper `client_duration.py::real_client_task_train_duration` (intrinsic
  `WALL_SEND_TS − WALL_RECV_TS`) is **already called** in `fwdllm_aggregator._process_single_trainer_message`
  (~line 713) to set `PROP_CLIENT_TASK_TRAIN_DURATION`. No port needed.
- Oort selector with faithful §S.pacer two-branch pacer, §S.temporal UCB, D1–D8 fixes — all in
  `flame/selector/oort.py`, usable verbatim if a baseline selects via oort.
- Dynamic-KC controller/policy (`flame/selector/dynamic_kc_{controller,policy}.py`) is **already
  integrated** (`_build_dynamic_kc_metrics`, `_dynamic_kc_controller.step()`). See
  [docs/dynamic_kc_design.md](docs/dynamic_kc_design.md).

### A.3  Parity tooling (reuse wholesale, extend with fwdllm rungs)
- Checker engine `async_cifar10/scripts/parity/{checks.py,report.py}` + `parity_check.py` — reuse,
  add fwdllm rungs (§D). Launchers (`debug_run.sh` + `*.yaml`), plotting
  (`scripts/analysis/analyze_run.py`), and the PARITY.md workflow policy all carry over unchanged.

### A.4  NOT reusable as-is — the actual work (§B, §E)
1. **Trainer sim path.** `FedSgdTrainer` emulates delay with real `time.sleep` (`_emulate_training_delay`,
   `FedSgdTrainer.py:459`); it does NOT stamp a modeled completion (`SIM_COMPLETION_TS` /
   `SIM_CLIENT_TASK_TRAIN_DURATION_S`) or `WALL_SEND_TS`/`WALL_RECV_TS` the way async_cifar10's
   `trainer/pytorch/main.py` does. **Port required.**
2. **Aggregator grad loop is off the virtual clock.** `_aggregate_grads_async` (~line 627) calls
   `channel.recv_fifo(...,1)` directly and commits on wall arrival — never consults `_sim_recv_min`,
   never advances `_vclock`, never holds in-flight slots. **The single biggest structural port.**
3. **Endogenous commit cadence** — variance-gated dynamic-K, not fixed-K (§B): a new emergent layer the
   async_cifar10 ladder does not model.

---

## §B  How FwdLLM differs structurally (the crux for fidelity)

async_cifar10 commits a model version every fixed **K** updates. FwdLLM does forward-gradient,
variance-controlled, **dynamic-K** aggregation. Four differences drive every new design choice:

- **B.1 Aggregates GRADIENTS (JVPs), not weights.** Trainers send forward-gradient estimates
  (`GRADIENTS` + `GRADIENTS_FOR_VAR_CHECK`); the agg accumulates into `grad_pool` and applies a
  server-LR SGD step at commit. Gradient **values** depend on real GPU compute (run for real in sim) →
  **mode-invariant** given identical input+perturbation seed. What differs across modes is **which
  gradients arrive, in what order, against which model version** — i.e. clock+selection+ordering
  fidelity. This is what makes the async_cifar10 ladder applicable.
- **B.2 Commit cadence is ENDOGENOUS (variance-gated dynamic-K).** At each `_agg_goal` (=K) boundary,
  `aggregate()`: `var = calculate_var(...)`; `var ≤ var_threshold` → **commit** (server-LR step, eval,
  `data_id += 1`, `model_version += 1`, clear `cached_v`); else **roll back**, push grads to `cached_v`,
  `iteration_per_data_id += 1`, **retry the same data_id**. `max_iter_per_data_id` force-commits despite
  failed variance. So updates-per-`model_version` is a **random variable** of the gradient-variance
  trajectory — **the `model_version` clock is not a fixed function of update count**, the central new
  thing parity must reproduce. A round is the outer loop over data bins (`data_id`); `_round += 1` only
  when `data_id == total_data_bins`.
- **B.3 Dynamic K and C.** After each agg-goal cycle `DynamicKCController.step(metrics)` may change **K**
  (`_agg_goal`) and **C** (concurrency) from observed metrics (var-pass ratio, eligible-ends). Parity
  must reproduce the **K/C trajectory**, not a scalar.
- **B.4 Eval per-variance-pass.** `eval_model()` runs on **every** committed data_id, not a fixed
  schedule. The eval modeled-delay must be stamped (async_cifar10 dead-end: stale-eval `sct`
  past-dating). FwdLLM eval is a forward pass (≈ train), so the eval delay model is closer to train than
  async_cifar10's 20×-faster eval.

> **Crux:** in async_cifar10 clock and commit count are loosely coupled; in FwdLLM the commit
> (model-version) cadence is a *feedback function of gradient variance over the accumulated pool*. The
> sim must reproduce not just **when** updates arrive (clock) but the **variance trajectory** gating
> each commit. Variance is mode-invariant **iff** the contributing set+order+model-version of gradients
> matches — reducing fwdllm parity back to clock+selection+ordering parity, plus a new
> **variance-cadence** verification layer.

---

## §C  Baseline matrix — **OPEN (blocking; user to supply)**

fluxtune / fwdllm / fwdllm++ are **not** in code — they map to config knobs. Filling this is the one
gate for the per-baseline plan (mirrors async_cifar10's felix/oort/refl/feddance source-of-truth).
Candidate differentiating knobs:

| Knob | Where | Meaning |
|---|---|---|
| `var_control` | hyperparameters | variance-gated dynamic-K on/off |
| `var_threshold` | hyperparameters | commit gate |
| `perturbation_sampling` | hyperparameters | forward-grad perturbation draw |
| `select_perturbation_using_jvp` | hyperparameters | JVP-guided perturbation selection |
| `is_async` | selector.kwargs | async grad loop vs sync barrier |
| `selector.sort` | selector | `async_random` vs `async_oort` |
| `dynamic_kc` | selector.kwargs | dynamic K/C vs static |
| `max_iter_per_data_id` | hyperparameters | force-commit cap |

**TO FILL (replace `?` with the mapping):**

| baseline | var_control | selector | dynamic_kc | perturbation / jvp-select | async | notes |
|---|---|---|---|---|---|---|
| **fluxtune** | ? | ? | ? | ? | ? | ? |
| **fwdllm** | ? | ? | ? | ? | ? | ? |
| **fwdllm++** | ? | ? | ? | ? | ? | ? |

This mapping decides: whether Stage 4 (oort selection) runs, whether Stage 5 (sync barrier) runs, and
the V1/V4 distributions (`max_iter_per_data_id`, `var_threshold` defaults). Parity is staged **one
baseline at a time** (PARITY.md workflow rule 3).

---

## §D  The parity ladder, adapted to FwdLLM

Reuse the async_cifar10 ladder (PARITY.md §1–§2): same stages, role tags, tiers, dependency gating
(lowest broken rung with sound inputs = ROOT).

### D.1  Reused verbatim
Stage 0 TC1/K10; Stage 1 P3/K1/K7; Stage 2 A1 (trivial under 100% avail) / A2; Stage 3 S3/4, A2c, and
Sx/Sd/S2/S1 **only if the baseline uses oort selection** (skip for `async_random`); Stage 4 T2,
per-phase timing, K6, T_mqtt; Stage 8 C1; Stage 9 budget/stop.

### D.2  MODIFIED rungs (re-defined for variance-gated dynamic-K)
| ID | async_cifar10 meaning | FwdLLM redefinition |
|---|---|---|
| **K3a** | K-th fastest async commit | advance of `model_version` **per committed data_id** (clock delta between successive **variance passes**) |
| **K3b** | overhead residual ≈ 0 | same, measured on the **variance-pass** boundary |
| **K2** | model versions / vsec | **committed data_ids / vsec** (throughput of *successful* variance passes) |
| **U3** | version gap at commit | gap of each contributing gradient vs `model_version` at the cycle it lands — spread across **multiple iterations per data_id** |
| **K8 / U2** | @ matched model_version | @ matched **data_id** (the meaningful progress axis) |

### D.3  NEW fwdllm-specific rungs (the variance-cadence layer — the prize)
Append-only, with deps so the engine can localize.

**Stage 6′ — Variance-gated aggregation cadence** (dep Stage 5 ordering + Stage 1 clock)

| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| **V1** | iterations-per-data_id dist (realized dynamic K) | MECHANISM/DIST | # accumulation cycles to pass variance ⇒ contributing set/order diverged | U5,U4 |
| **V2** | per-cycle `var` trajectory (at each agg-goal) | MECHANISM/DIST | variance *signal* diverges with matched inputs ⇒ grad-pool composition/order differs | V1 |
| **V3** | `cached_v` pool size over time | MECHANISM/DIAG | rollback/cache bookkeeping diverges | V1 |
| **V4** | force-commit frequency (`max_iter_per_data_id` bypass rate) | MECHANISM/DIST | cap hit at a different rate ⇒ chronic variance divergence | V1 |
| **V5** | variance-pass ratio per window | EMERGENT/DIST | rollup feeding DynamicKC | V1,V2 |

**Stage 3′ — Dynamic K/C trajectory** (dep Stage 3 selection + V5)

| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| **DK1** | K (`_agg_goal`) trajectory | MECHANISM/DIST | DynamicKC sees different metrics ⇒ K diverges (feeds back into cadence) | V5 |
| **DK2** | C (`dynamic_c`) trajectory | MECHANISM/DIST | concurrency target diverges | V5,S3/4 |
| **DK3** | eligible-ends-count metric fed to policy | CONTROL/DIST | the policy *input* differs (fix input, not policy) | A2 |

**Stage 7′ — Forward-gradient quality**

| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| **G1** | per-update grad/JVP norm or SNR dist | EMERGENT/DIST | grad *quality* diverges (should be ≈ mode-invariant; FAIL = perturbation seed/order leaked) | S2,T_gpu |
| **G2** | grad_pool size at commit (realized contributions) | EMERGENT/DIST | rollup of V1×K | V1,DK1 |

> **Decomposition example:** `K2`(throughput)✗ but `K3a`(per-pass advance)✓ → clock fine, commit count
> diverged → walk to V1/V5. `V1`✗ + `V2`✓-given-matched-input → the *inputs* to variance differ → walk
> to U5/S2. `V2`✗ with V1 inputs matched → a true grad-pool accumulation-order bug. Never fix K2/V5
> directly (emergent) — localize the lowest variance-cadence rung with sound inputs.

---

## §E  Staged implementation + validation

Each stage states exit criteria + min run length (PARITY.md budget table). Land config-gated (flag-off
⇒ byte-identical) + test-guarded.

### Stage 0 — Telemetry coverage + real admissibility (gate)
Port/confirm fwdllm emission for every field the ladder reads: `vclock_now`, `sim_completion_ts`/`sct`,
`model_version`, `data_id`, `iteration_per_data_id`, `var`, `var_threshold`, `_agg_goal`(K),
`dynamic_c`(C), `agg_goal_count`, per-commit `train_duration`, `WALL_SEND_TS`/`WALL_RECV_TS`, grad/JVP
norm. Implement **TC1** for fwdllm; **partition eval vs train commits** keyed on the variance-pass eval
(async_cifar10 dead-end: eval commits polluting `agg_rounds` broke monotone/staleness). Run
`validate_real`-equivalent admissibility (concurrency ≤ c, no double-dispatch, one-in-flight by
construction). **Min:** 5–10 min smoke. **Exit:** TC1 all-present both modes; real admissible; report
JSON with no SKIPs.

### Stage 1 — Trainer sim path (modeled completion, no wall sleep)
Add `simulated` mode to `FedSgdTrainer`: replace `_emulate_training_delay` (real `time.sleep`) with
stamping `_sim_completion_ts = sim_send_ts + max(real_gpu_s, modeled_budget_D) + sim_completion_leg_s`;
send `SIM_COMPLETION_TS` + `SIM_CLIENT_TASK_TRAIN_DURATION_S` + `WALL_SEND_TS`/`WALL_RECV_TS` (client
stamps for §S.dur). Stamp a **per-eval** `sct` (dead-end: reusing the last train `sct` past-dates every
eval). Eval delay ≈ train (forward pass), **not** the 20× speedup — CONFIRM the factor per baseline.
**Min:** 10 min. **Exit:** P3 (modeled D) matches; K6 advancing; T2 matched; no real `time.sleep` on
the sim path.

### Stage 2 — Drive the virtual clock from the grad loop (THE structural port)
Replace the raw `channel.recv_fifo(...,1)` in `_aggregate_grads_async` with the inherited
**`_sim_recv_min`** path (sct-ordered reorder buffer + §3.drain) so a fast trainer's grad buffers as a
future and grads are consumed in completion order. Wire **`_sim_hold_busy_slots`** (§3.resid) for
one-in-flight-per-trainer. Note: fwdllm processes **one message per `_aggregate_grads_async` call**, so
design the slot release on the **agg-goal boundary** (where `_per_agg_trainer_list` clears), NOT
per-message — the hold/release points differ from asyncfl's batch `_aggregate_weights`. Keep
`simCommitOverheadSeconds=0` (overhead on the vclock is a hard dead-end). **Risk:** a data_id can span
**many** agg-goal cycles (rollback path) — the slot-hold and sct-buffer must survive rollbacks without
leaking or double-committing a grad; over-instrument `inflight_residence` + per-cycle buffer occupancy.
**Min:** 45 min. **Exit:** K1 monotone; K3a/K3b (per variance-pass) ≈ 0 residual; U5 inter-arrival;
one-in-flight overlap ≈ real (target 0%); no past-dating (commit_gap ≈ 0).

### Stage 3 — Variance-cadence parity (the fwdllm prize)
Implement V1–V5, G1–G2 (§D.3). Likely root families (confirm via the lowest broken rung):
- *Contributing-set/order divergence* (U5/S2 upstream) → V1 diverges → K2. Fix is upstream; V-rungs
  only **localize**.
- *Grad-pool accumulation order* (`cached_v` carry-over, `grad_pool.append` order) → V2 diverges with
  matched inputs → a true sim bug in the accumulation bookkeeping.
- *Force-commit rate* (V4) divergence → chronic variance divergence, not a separate bug.
**Min:** 90 min – 2 h (variance-feedback-compounding, like the async clock residuals). **Exit:** V1/V2/V5
PASS; K2 (committed-data_ids/vsec) PASS; DK1/DK2 track (if dynamic_kc on). **DynamicKC coupling:**
validate **DK3** (policy *input*) before DK1/DK2 — a diverging input means fix the metric, not the
policy (CONTROL-before-MECHANISM).

### Stage 4 — Selection fidelity (only oort-selecting baselines)
If any baseline uses `async_oort`, validate A2c/Sx/Sd/S2 + the §S.pacer/§S.temporal/§S.dur stack (all
landed in `flame/selector/oort.py`). For `async_random` this stage is inert (S-rungs WARN/skip).
**Min:** 45 min – 3 h (a selection-mix residual can surface late — refl K2 only at 3 h). **Exit:** A2c
PASS, Sd binding real≈sim, no pacer ratchet.

### Stage 5 — Sync-path parity (only if a baseline runs sync `_aggregate_grads_sync`)
If any baseline is strict-sync, give the sync grad barrier the barrier-anchored visibility-lag
treatment (§6.u6). Streaming (async) baselines commit each grad at its own sct → leave per-message
correct. **Min:** 45 min. **Exit:** U6 barrier-anchored lag real≈sim; no per-message past-dating.

### Stage 6 — Convergence sign-off
C1 accuracy, C2 loss, K8/U2 terminal-state @ matched data_id. **Min:** full (3–4 h+). **Exit:** curves
within tolerance at matched data_id; K8/U2 rel within bar.

### Per-stage discipline (from PARITY.md)
Smoke (5 min) before any multi-hour run; one mechanism per run round when a fix could perturb another
baseline; re-run **real** only when the real path changes (sim-only validates against the stored real
dir); over-instrument first. Land config-gated + a pytest guard; run the fwdllm test subset + the shared
selector/mode/sim/parity suites green before declaring a stage done.

---

## §F  Key design decisions (precise)

1. **Sim does real forward-grad compute, charges modeled time.** GPU runs the real JVP; the agg stamps
   `sct = sim_send_ts + max(real_gpu_s, D) + leg`. Makes gradient values mode-invariant (§B.1).
   **Never** put overhead on the virtual clock (`vclock = max(vclock, sct)` only).
2. **Progress axis is `data_id` (committed variance passes), not raw update count.** All throughput/
   terminal rungs (K2, K8, U2) re-keyed to committed data_ids. Update count per data_id is the dynamic-K
   random variable V1 validates — an output to match, not an input to assume.
3. **Variance is the new emergent gate; localize it, never tune it.** A `var`/`var_pass_ratio`
   divergence is *always* downstream of a contributing-set/order/accumulation divergence (V-rungs gate
   on U5/S2/V1). Treat V5/DK1/DK2 as EMERGENT — walk down. `var_threshold` is a config knob, **not** a
   parity lever.
4. **Slot residence must survive variance rollbacks.** A data_id can span many agg-goal cycles, so
   `_sim_hold_busy_slots` release keys on the **agg-goal boundary** (`_per_agg_trainer_list` clear), and
   the sct reorder buffer must not strand a grad across a rollback. The fwdllm-specific extension of
   §3.resid and the most likely subtle leak.
5. **DynamicKC: validate the input before the policy.** DK3 (eligible-ends count, var-pass metrics fed
   to `DynamicKCController.step`) is a CONTROL rung; a diverging *output* (DK1/DK2) with matched *input*
   is the rare genuine policy bug, else fix the input. Controller is shared/baseline-agnostic — do not
   fork it per baseline.
6. **Real is the reference only after admissibility.** A real↔sim gap has **two** fix directions. The
   §S.dur helper is already wired — keep it; if a selection-mix residual appears, check whether the
   **real** input is the divergent side (PARITY.md: cleaning real can flip the divergence onto sim).
7. **100% availability now, keep the seam.** `track_trainer_avail` stays wired but inert (`enabled:
   False`); A1/A2 near-trivial. Do NOT delete the eligibility plumbing — unavailability is the next
   workstream, following [UNAVAILABILITY_DESIGN.md](../async_cifar10/UNAVAILABILITY_DESIGN.md) (the
   fwdllm side adds: which baselines are availability-aware, the variance-cadence interaction with a
   withheld/late grad, and whether a withheld grad rolls into `cached_v`).

---

## §G  Open questions / risks / dead-ends

**Open (toward Stage 0):**
- §C baseline→knob mapping (**blocking the per-baseline matrix**).
- Eval-delay factor for fwdllm (forward pass ≈ train; what factor vs train budget?).
- Which baselines are async vs sync (decides Stage 5).
- Does any baseline use `async_oort` (decides Stage 4 scope) or all `async_random`?
- `max_iter_per_data_id` / `var_threshold` defaults per baseline (drive V1/V4).

**Risks specific to fwdllm:**
- **Variance-cadence is feedback-compounding** — a tiny per-cycle grad-pool order difference compounds
  into a different iterations-per-data_id (like the async clock residuals that surfaced only at 3 h).
  Bin V1/V2 by run-fraction to separate a *constant* mix bias from a *compounding* feedback loop.
- **`cached_v` carry-over** is stateful across cycles; a divergence looks like a variance bug but is
  bookkeeping (V3 DIAG localizes it).
- **One-message-per-call grad loop** vs asyncfl's batch — slot-hold/release wiring is genuinely
  different; do not copy async_cifar10's release points blindly.

**Pre-emptive dead-ends (from PARITY.md):** overhead > 0 on the vclock; prediction-only gates that never
block; tuning a scalar redispatch/overhead knob instead of fixing the mechanism; letting eval commits
into the train/agg stream; expressing "busy" via the unavailable list. **fwdllm-new:** do **not** tune
`var_threshold`/`max_iter_per_data_id` to force cadence parity — those are baseline-defining knobs; a
cadence gap is an upstream set/order/clock divergence.

---

## §H  Status

*(empty — first entry after Stage 0 smoke. Record per-baseline run dirs, lowest broken rung, root
hypothesis, score X/N, JSON path; keep the run-length budget keyed. One section, updated in place.)*
