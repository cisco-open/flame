# High-Fidelity Simulator for FwdLLM -- Design & Staged Build Plan

**Design-only.** This document is the plan for building a high-fidelity **simulated-clock** runner
for the `fwdllm` example (FedFwd / forward-gradient FL) that reaches **real<->sim parity** across the
**fluxtune / fwdllm / fwdllm++** baselines, at **100% availability (syn_0)** and under
**unavailability (syn_20, syn_50, mobiperf)**. The implementation is a **future branch/PR**; nothing
here is built yet.

**Prerequisites (read first):**
- [async_cifar10/PARITY.md](../async_cifar10/PARITY.md) -- the parity methodology (the ladder §1, roles/
  tiers/dependency-gating, workflow policy, run-length budget, landed sim mechanisms §3). **fwdllm's
  own rung catalog is PARITY.md §F** (modified rungs, the variance-cadence layer). This doc references
  §F for rung *definitions* and does not duplicate them; it states the fwdllm **build deltas**.
- [async_cifar10/UNAVAILABILITY_DESIGN.md](../async_cifar10/UNAVAILABILITY_DESIGN.md) -- the availability
  substrate this reuses wholesale (ClientAvailability mixin, trace-read effect path, the two flag axes,
  send-gate/deliver-late-stale, two-ledger discipline, starvation self-termination, the ground-truth
  fidelity rungs A6/A7/A8/K11). fwdllm inherits it via the aggregator class chain (§A).

**Starting reality (why this is real work, not a re-label):** FedFwd has **no simulated-clock support
today** -- every launcher yaml is `time_mode: real`, and the trainer emulates delay with a real
`time.sleep` (`trainer/forward_training/FedSgdTrainer.py:481` `_emulate_training_delay`, called at
`:523`; explicit note `trainer/main.py:57-78`). The aggregator machinery for a virtual clock is
*inherited but not driven* (§A). Adding the sim clock + wiring unavailability into the
variance-gated gradient loop is the build.

**Decisions locked (kickoff):**
- A trusted **real** wall-clock run is the reference per baseline (as in async_cifar10). Parity != tuning
  sim to real; real must pass `validate_real` admissibility first.
- The parity checker **extends the shared `async_cifar10/scripts/parity` engine** -- fwdllm rungs are
  *added* (PARITY.md §F is their catalog), the engine is reused.
- The fluxtune / fwdllm / fwdllm++ -> config mapping is **resolved** from the landed launcher configs
  (§C); it is no longer a blocking gap.

---

## §A  Already on the class (inherited) vs. NOT yet wired

The virtual clock is a **flame-core** capability. fwdllm's aggregator is
`examples/fwdllm/aggregator/FedSgdAggregator.py` -> `flame/mode/horizontal/syncfl/fwdllm_aggregator.py::
TopAggregator(AsyncTopAgg)` -> `asyncfl.TopAggregator` -> `syncfl.TopAggregator(ClientAvailability, Role)`.
So both the vclock machinery **and** the availability substrate are already **on the instance** -- the
work is **wiring them into fwdllm's gradient loop + trainer**, not re-implementing them.

### A.1  Aggregator spine (inherited, baseline-agnostic)
| Mechanism (from `asyncfl/top_aggregator.py`) | Status for fwdllm |
|---|---|
| Virtual clock `_vclock`, `_advance_sim_clock` (`vclock = max(vclock, sct)`) | inherited; **NOT yet driven** by the grad loop |
| §3.drain sct-ordered drain `_sim_recv_min` | inherited; **bypassed** (fwdllm's `_aggregate_grads_async` uses raw `recv_fifo`) |
| §3.resid one-in-flight residence `_sim_hold_busy_slots` | inherited; **bypassed** |
| §4.5/§4.9 sct-gated pool exclusion / carry-over (oort overlay) | available if a baseline selects via oort (fluxtune) |
| `ClientAvailability` substrate (trace-read gate, two ledgers, proactive evict, starvation advance) | inherited; **NOT yet wired** into the fwdllm grad loop |

### A.2  Selector + duration signal (already wired)
- §S.dur duration helper `client_duration.py::real_client_task_train_duration` (intrinsic
  `WALL_SEND_TS - WALL_RECV_TS`) is already called in `fwdllm_aggregator._process_single_trainer_message`
  (`:745`) to set `PROP_CLIENT_TASK_TRAIN_DURATION`. No port needed.
- Oort selector (faithful §S.pacer two-branch pacer, §S.temporal UCB, D1-D8 fixes) is in
  `flame/selector/oort.py`, usable verbatim by fluxtune (`async_oort`).
- Dynamic-KC controller/policy (`flame/selector/dynamic_kc_{controller,policy}.py`) is already integrated
  (`_build_dynamic_kc_metrics`, `_dynamic_kc_controller.step()`); fluxtune leaves it **disabled** (fixed
  K/C) per its config. See [docs/dynamic_kc_design.md](docs/dynamic_kc_design.md).
- Analysis: `scripts/analysis/analyze_run.py` is already fwdllm-aware via `telemetry_manifest.yaml`
  (data_id/iteration_per_data_id progress hierarchy; per-category populate/partial status documented).

### A.3  NOT reusable as-is -- the actual work
1. **Trainer sim path.** `FedSgdTrainer._emulate_training_delay` (`:481`) uses real `time.sleep`; it does
   NOT stamp a modeled completion (`SIM_COMPLETION_TS` / `SIM_CLIENT_TASK_TRAIN_DURATION_S`) or
   `WALL_SEND_TS`/`WALL_RECV_TS` the way async_cifar10's `trainer/pytorch/main.py` does. **Port required.**
2. **Aggregator grad loop is off the virtual clock.** `_aggregate_grads_async` (`fwdllm_aggregator.py:693`)
   calls `channel.recv_fifo(...,1)` directly and commits on wall arrival -- never consults `_sim_recv_min`,
   never advances `_vclock`, never holds in-flight slots. **The single biggest structural port.**
3. **Endogenous commit cadence** -- variance-gated dynamic-K, not fixed-K (PARITY.md §F.1): a new emergent
   layer the async_cifar10 ladder does not model.
4. **Availability telemetry.** The fwdllm trainer emits **no `EVENT_AVAIL_CHANGE`**
   (`telemetry_manifest.yaml`: availability = *partial*), so A6/A7/A8 ground-truth rungs can't run.
   The `avail_change` / `agg_belief_change` / `send_gate_wait` builders exist
   (`flame/telemetry/events.py:225/521`, `task_send` fields) and just need emitting from the fwdllm
   trainer + aggregator.

---

## §B  How FwdLLM differs structurally

Full detail (the crux that makes the ladder applicable, and the variance-cadence rung layer it forces) is
**PARITY.md §F.1**. In one line: fwdllm aggregates **gradients** (JVPs) not weights, its commit cadence is
**endogenous** (variance-gated dynamic-K), and its progress axis is **`data_id`** (committed variance
passes), not raw update count. Gradient *values* are mode-invariant given identical input+perturbation
seed, so fwdllm parity reduces to clock+selection+ordering parity **plus** a new variance-cadence layer.

Concrete anchors (`flame/mode/horizontal/syncfl/fwdllm_aggregator.py`): `aggregate()` var gate +
rollback/`cached_v` at each `_agg_goal` boundary; `total_data_bins=150` (`:271`); force-commit cap
`_max_iter_per_data_id` (`:276`, config key `max_iterations_per_data_id`); `_reselect_each_iteration`
(`:295`, per-iteration reselection for fwdllm++); sync path `_aggregate_grads_sync` (`:1354`).

---

## §C  Baseline matrix -- RESOLVED (from the landed launcher configs)

Read from `expt_scripts/{fluxtune,fwdllm,fwdllm_plus}_n10_smoke.yaml`. Canonical copy + taxonomy mapping
is **PARITY.md §F.2**; repeated here for the build plan.

| baseline | sync/async | selector | agg | tracking_mode / avail | reselection | K (agg_goal) | dynamic_kc |
|---|---|---|---|---|---|---|---|
| **fluxtune** | async | `async_oort` | fedbuff (+server LR, JVP) | `client_notify` (3-tier, mobiperf_3st_50) | -- | 3 | disabled (fixed K/C) |
| **fwdllm** | sync | `random` | fedavg | `default` (unaware) | per-round | 10 (=c; all selected required) | -- |
| **fwdllm_plus** | sync | `random` | fedavg | `oracular` (`_metadata`, mobiperf_2st) | per-iteration (`reselect_each_iteration=True`) | 2 | -- |

**Availability-taxonomy mapping** (UNAVAILABILITY_DESIGN.md two axes):
- **fwdllm** -- unaware at selection (`avail_select_filter=False`), reactive-90s in-flight
  (`proactive_inflight_evict=False`).
- **fwdllm_plus** -- aware at selection via `oracular` trace read (`avail_select_filter=True`),
  reactive-90s in-flight.
- **fluxtune** -- async, aware via `client_notify` (message-transport), reactive-90s in-flight.
  `client_notify` is async_cifar10's **deferred Stage-H** tracking model -> **decision D1**.

This decides: Stage 3 oort rungs run **only** for fluxtune; Stage 5 sync-barrier rungs run for
**fwdllm/fwdllm_plus**; V1/V4 distributions are shaped by each baseline's `max_iterations_per_data_id` /
`var_threshold`. Parity is staged **one baseline at a time** (PARITY.md workflow rule).

---

## §D  The parity ladder for fwdllm

**Rung definitions live in PARITY.md §F** (modified rungs §F.3; variance-cadence / dynamic-K/C /
forward-grad rungs §F.4). This section states only which rungs apply and how they gate.

- **Reused verbatim** (PARITY.md §2): Stage 0 TC1/K10; Stage 1 P3/K1/K7; Stage 2 A1/A2 (+ A3 time-base
  CONTROL, + A6/A7/A8 ground-truth once telemetry is ported); Stage 3 S3/4, A2c and the oort stack
  (Sx/Sd/S2) **only for fluxtune**; Stage 4 T2, per-phase timing, K6, T_mqtt; Stage 8 C1; Stage 9 budget/stop.
- **Modified** for variance-gated dynamic-K: K3a/K3b/K2/U3/K8/U2 -> PARITY.md §F.3 (all re-keyed to the
  **variance-pass boundary** / committed **data_id**).
- **New** -- the variance-cadence layer (the prize): V1-V5 (Stage 6'), DK1-DK3 (Stage 3'), G1-G2 (Stage 7')
  -> PARITY.md §F.4. Localize down, never fix an EMERGENT rung directly; `var_threshold` /
  `max_iterations_per_data_id` are config knobs, not parity levers.
- **Availability** rungs (A1-A5, A6/A7/A8/K11, withheld_delivery, abandon_timeout, starvation_advance,
  eligible_pool_reduction) are inherited from the async_cifar10 substrate and apply once §Stage-Avail
  (below) wires the effect path + telemetry.

---

## §E  Staged implementation + validation

Each stage: config-gated (flag-off => byte-identical), test-guarded, exit criteria + min run length
(PARITY.md budget table). Land one mechanism per run round. **Smoke (5 min) before any multi-hour run.**

### Stage 0 -- Telemetry coverage + real admissibility (gate)
Confirm fwdllm emits every field the ladder reads: `vclock_now`, `sim_completion_ts`/`sct`,
`model_version`, `data_id`, `iteration_per_data_id`, `var`, `var_threshold`, `_agg_goal`(K),
`dynamic_c`(C), `agg_goal_count`, per-commit `train_duration`, `WALL_SEND_TS`/`WALL_RECV_TS`, grad/JVP
norm, **and `avail_change`/`agg_belief_change`/`send_gate_wait`** (§A.3 item 4). Implement **TC1** for
fwdllm; **partition eval vs train commits** on the variance-pass eval (async_cifar10 dead-end: eval commits
polluting `agg_rounds` broke monotone/staleness). Run `validate_real` admissibility. **Min:** 5-10 min
smoke. **Exit:** TC1 all-present both modes; real admissible; report JSON no SKIPs.

### Stage 1 -- Trainer sim path (modeled completion, no wall sleep)
Add a `simulated` mode to `FedSgdTrainer`: replace `_emulate_training_delay` (real `time.sleep`) with
stamping `_sim_completion_ts = sim_send_ts + max(real_gpu_s, modeled_budget_D) + sim_completion_leg_s`;
send `SIM_COMPLETION_TS` + `SIM_CLIENT_TASK_TRAIN_DURATION_S` + `WALL_SEND_TS`/`WALL_RECV_TS`. Stamp a
**per-eval** `sct` (dead-end: reusing the last train `sct` past-dates every eval); eval delay ~= train
(forward pass), **not** async_cifar10's 20x speedup -- **decision D4**. **Min:** 10 min. **Exit:** P3 matches;
K6 advancing; T2 matched; no real `time.sleep` on the sim path.

### Stage 2 -- Drive the virtual clock from the grad loop (THE structural port)
Replace the raw `channel.recv_fifo(...,1)` in `_aggregate_grads_async` (`:693`) with the inherited
**`_sim_recv_min`** path (sct-ordered reorder buffer + §3.drain) so a fast trainer's grad buffers as a
future and grads are consumed in completion order; wire **`_sim_hold_busy_slots`** (§3.resid) for
one-in-flight-per-trainer. **fwdllm processes one message per `_aggregate_grads_async` call**, so key the
slot release on the **agg-goal boundary** (where `_per_agg_trainer_list` clears), NOT per-message -- the
hold/release points differ from asyncfl's batch `_aggregate_weights`. Keep `simCommitOverheadSeconds=0`
(overhead on the vclock is a hard dead-end). **Risk:** a data_id can span **many** agg-goal cycles
(rollback path) -- slot-hold + sct-buffer must survive rollbacks without leaking or double-committing a
grad; over-instrument `inflight_residence` + per-cycle buffer occupancy. **Min:** 45 min. **Exit:** K1
monotone; K3a/K3b (per variance-pass) ~= 0 residual; U5 inter-arrival; one-in-flight overlap ~= real;
no past-dating (commit_gap ~= 0).

### Stage Avail -- Wire unavailability into the grad loop (syn_20/50/mobiperf)
Wire the inherited `ClientAvailability` effect path into fwdllm: send-time gate (real) /
`delivery_ts = max(sct, next_avail)` buffering (sim); the two ledgers; per-baseline in-flight timing
(reactive-90s for all three fwdllm baselines; no proactive-evict baseline here); starvation vclock-advance
under scarcity; and per-baseline `avail_select_filter` (fwdllm off, fwdllm_plus/fluxtune on). Emit the
availability telemetry (Stage 0) so A6/A7/A8/K11 run. **Interaction risk (decision D3):** a withheld/late
grad meets the variance gate -- does it roll into `cached_v` on a rollback, and does a late grad against an
old `model_version` inflate the variance signal? This is genuinely new vs async_cifar10 (which commits
weights, not a variance-gated pool). **Min:** 45-90 min. **Exit:** A1-A5 + A6/A7/A8 PASS; withheld-then-
delivered (not dropped); self-stops (`"stopping run"`, no `SIM_WALL_CEILING`); syn_0 byte-identical gate
ON vs OFF.

### Stage 3 -- Variance-cadence parity (the fwdllm prize)
Implement V1-V5, G1-G2, DK1-DK3 (PARITY.md §F.4). Likely roots (confirm via the lowest broken rung):
contributing-set/order divergence (U5/S2 upstream) -> V1 -> K2; grad-pool accumulation order
(`cached_v` carry-over, `grad_pool.append` order) -> V2 with matched inputs = a true sim bug; force-commit
rate (V4) = chronic variance divergence, not a separate bug. **DynamicKC coupling:** validate **DK3**
(policy *input*) before DK1/DK2 (fluxtune leaves dynamic_kc off, so DK is inert there; relevant only if a
baseline enables it). **Min:** 90 min - 2 h (variance-feedback-compounding). **Exit:** V1/V2/V5 PASS; K2
(committed-data_ids/vsec) PASS; DK tracks if enabled.

### Stage 4 -- Selection fidelity (fluxtune only, `async_oort`)
Validate A2c/Sx/Sd/S2 + the §S.pacer/§S.temporal/§S.dur stack (all landed in `flame/selector/oort.py`).
For the two `random`-selector baselines this stage is inert (S-rungs WARN/skip). **Min:** 45 min - 3 h (a
selection-mix residual can surface late). **Exit:** A2c PASS, Sd binding real~=sim, no pacer ratchet.

### Stage 5 -- Sync-path parity (fwdllm / fwdllm_plus, `_aggregate_grads_sync`)
Give the sync grad barrier the barrier-anchored visibility-lag treatment (§6.u6). fluxtune streams (async),
commits each grad at its own sct -> leave per-message correct. **Min:** 45 min. **Exit:** U6 barrier-
anchored lag real~=sim; no per-message past-dating.

### Stage 6 -- Convergence sign-off
C1 accuracy, C2 loss, K8/U2 terminal-state @ matched **data_id**. **Min:** full (3-4 h+). **Exit:** curves
within tolerance at matched data_id; K8/U2 rel within bar.

---

## §F  Key design decisions (open -- resolved at implementation)

- **D1 -- tracking-mode strategy.** fluxtune's landed config uses `tracking_mode: client_notify`, which
  UNAVAILABILITY_DESIGN.md treats as the **deferred Stage-H** message-transport model (async_cifar10 v1 is
  all `trace_read`). Options: (a) implement `client_notify` first-class for fwdllm now (larger scope, but
  it's baked into fluxtune's config and can't just be ignored), or (b) map fluxtune onto `trace_read` for a
  v1 parity pass and treat `client_notify` as Stage H. **Lean (a)** -- fluxtune cannot run its intended
  taxonomy without it. fwdllm_plus's `oracular` and fwdllm's unaware paths are already `trace_read`-shaped.
- **D2 -- availability telemetry port.** The fwdllm trainer emits no `EVENT_AVAIL_CHANGE`; the aggregator
  emits no `agg_belief_change`/`send_gate_wait`. All three builders exist in `flame/telemetry/events.py`
  -- port emission (trainer state machine + aggregator belief hooks) so A6/A7/A8/K11 light up. Prereq for
  Stage Avail's exit.
- **D3 -- variance-cadence x withheld/late grads.** Does a withheld grad roll into `cached_v` on rollback?
  Does a late grad against a stale `model_version` distort the `var` signal (and thus dynamic-K)? New to
  fwdllm; over-instrument in Stage Avail before Stage 3.
- **D4 -- eval-delay factor.** FedFwd eval is a forward pass (~= train cost), unlike async_cifar10's
  ~20x-faster eval. Confirm the factor per baseline before Stage 1's per-eval `sct` stamp.

### Locked principles (from async_cifar10, carried over)
1. **Sim does real forward-grad compute, charges modeled time.** GPU runs the real JVP; the agg stamps
   `sct = sim_send_ts + max(real_gpu_s, D) + leg`. Makes gradient values mode-invariant.
   **Never** put overhead on the virtual clock (`vclock = max(vclock, sct)` only).
2. **Progress axis is `data_id`** (committed variance passes). Update count per data_id is the dynamic-K
   random variable V1 validates -- an output to match, not an input to assume.
3. **Variance is an emergent gate; localize, never tune it.** V-rungs gate on U5/S2/V1. `var_threshold` is
   a baseline-defining config knob.
4. **Slot residence must survive variance rollbacks.** Release keys on the agg-goal boundary; the sct
   reorder buffer must not strand a grad across a rollback (the fwdllm-specific extension of §3.resid).
5. **DynamicKC: validate the input before the policy.** DK3 (CONTROL) before DK1/DK2 (MECHANISM). The
   controller is shared/baseline-agnostic -- do not fork it per baseline.
6. **Real is the reference only after admissibility.** A real<->sim gap has two fix directions; if a
   selection-mix residual appears, check whether the **real** input is the divergent side.
7. **syn_0 byte-identical gate OFF; unavailability config-gated.** Reuse the async_cifar10 flag axes and
   the `simUnavailability` gate; do not delete eligibility plumbing.

---

## §G  Open questions / risks / dead-ends

**Open (toward Stage 0):**
- D1-D4 above.
- Which real datasets/traces to use for the fwdllm reference runs (agnews H5 is the smoke default).

**Risks specific to fwdllm:**
- **Variance-cadence is feedback-compounding** -- a tiny per-cycle grad-pool order difference compounds into
  a different iterations-per-data_id (like the async clock residuals that surfaced only at 3 h). Bin V1/V2
  by run-fraction to separate a *constant* mix bias from a *compounding* feedback loop.
- **`cached_v` carry-over** is stateful across cycles; a divergence looks like a variance bug but is
  bookkeeping (V3 DIAG localizes it).
- **One-message-per-call grad loop** vs asyncfl's batch -- slot-hold/release wiring is genuinely different;
  do not copy async_cifar10's release points blindly.
- **Withheld grad x variance gate** (D3) -- a stale late grad may distort `var` and the dynamic-K decision.

**Pre-emptive dead-ends (from PARITY.md / UNAVAILABILITY_DESIGN.md):** overhead > 0 on the vclock;
prediction-only gates that never block; tuning a scalar redispatch/overhead knob instead of fixing the
mechanism; letting eval commits into the train/agg stream; expressing "busy" via the unavailable list.
**fwdllm-new:** do **not** tune `var_threshold`/`max_iterations_per_data_id` to force cadence parity --
those are baseline-defining knobs; a cadence gap is an upstream set/order/clock divergence.

---

## §H  Status

*(empty -- first entry after Stage 0 smoke on the implementation branch. Record per-baseline run dirs,
lowest broken rung, root hypothesis, score X/N, JSON path; keep the run-length budget keyed. One section,
updated in place.)*
