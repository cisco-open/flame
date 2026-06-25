# Sim Unavailability — Design & Staged Plan

**Status:** design settled (all open questions resolved Jun 25); implementation NOT started.
Pickup-ready reference for adding trainer **unavailability** to the high-fidelity async_cifar10
sim — and the template for the same feature in fwdllm ([simulate_fwdllm.md](../fwdllm/simulate_fwdllm.md) §7).

**Prerequisites (read first):** [PARITY.md](PARITY.md) §1–§2 (the causal ladder, role/tier tags,
dependency gating) and its §3 mechanism reference (`_vclock`, §3.drain, §3.resid, §4.5/§4.9, §S.dur).
This feature extends that ladder; every new rung and mechanism below assumes that vocabulary.

**Goal:** let trainers drop in/out of `AVL_TRAIN` / `AVL_EVAL` / `UN_AVL` per their traces inside
the sim, emitting correct client-side notifications on the **virtual clock**, without (a) breaking
the parity already won at 100% availability, (b) a per-tick MQTT broadcast storm, or (c)
frozen-clock deadlocks.

---

## 0. What already exists (design *with* the grain)

The tree already has **three** availability paths; the 100%-avail runs left them dormant
(`trainer_event_dict`/`trainer_unavail_durations` default `None`). Reconcile these, don't add a fourth.

| # | Path | Where | Time-base | Drives |
|---|---|---|---|---|
| **A. Agg pull (event trace)** | `get_curr_unavail_trainers()` binary-searches each trainer's `trainer_event_dict` (SortedDict `ts→state`) | syncfl/oort/asyncfl top_agg | `_vclock.now` (sim) / `time.time()−agg_start` (real) | `channel.set_curr_unavailable_trainers` at selection |
| **B. Agg pull (duration windows)** | `oracular_trainer_avail_check(end)` tests `trainer_unavail_durations[end]` `(start,dur)` | `asyncfl/top_aggregator.py:1264` | same | per-pick veto |
| **C. Trainer push (notifications)** | `check_and_update_state_avl` pops trace events → `channel.update_trainer_state` → backend → `Channel.update_state` (evicts `UN_AVL` / `AVL_TRAIN→AVL_EVAL` from `selected_ends`) | `trainer/pytorch/main.py` + `channel.py:1056` | `_sim_now()` = last dispatch `_sim_send_ts` (sim) / wall (real) | MQTT message |

**Two anchoring facts:**
1. **A/B already key on `_vclock.now` in sim** (comment at `asyncfl/top_aggregator.py:1264`:
   *"wall-clock would barely advance vs the sim timeline, so every unavailability window would be
   missed"*). Aggregator-pull on the virtual clock = the no-comms, deterministic, never-freezes path.
2. **C has a frozen-clock defect in sim:** `_sim_now()` returns `_sim_send_ts`, which only updates
   when the trainer is *dispatched*. An unselected trainer never advances → never pops events → never
   notifies; a trainer that goes `UN_AVL` can't be selected → can't advance → **stuck `UN_AVL`
   forever**. So C cannot be the source of truth for *selection* in sim until this is fixed.

---

## 1. Core decisions (all resolved)

### The source of truth is PER-BASELINE — the asymmetry IS the design
- **Availability-UNAWARE (oort, refl):** real trainers go unavailable and **do not notify**; the
  oracular aggregator **reads the traces**. So **A/B are the FAITHFUL model here, not a shortcut** —
  real oort has no notification channel. The oracle gates **new selection only** (Q-new-1): it never
  proactively evicts an in-flight trainer (it has no way to know), it just waits for the withheld
  update to arrive on return.
- **Availability-AWARE (felix, fluxtune):** the aggregator does **not** read traces; it is informed
  by **trainer→agg notification events** that **must take effect immediately** at the aggregator and
  factor into selection AND `selected_ends`. So **C is the faithful model here**, with the
  frozen-clock defect fixed.
- **One trace + one `state_at(trainer, vclock)` resolver + one effect path** shared by both, so
  "what the agg believes" and "what the trainer is" cannot disagree. The only per-baseline difference
  is *how the agg learns* a transition (oracle read vs event) and *whether* it acts on it (`awareness`
  flag). **agg awareness** (does it factor availability in?) is orthogonal to **how it learns**.

### Stop-gap for the aware path (Q4)
Model the aware event as **instant oracular reflection at the agg** (agg reads the shared trace on
the vclock at the transition instant, zero lag — behaviorally equal to instant push for selection).
**End goal = the true `avl_*` trainer→agg message.** Architecture must keep the
state-resolution + effect logic identical so the future swap changes only *transport*, not *effect*
(extensibility is a hard requirement). Per-tick broadcast (agg pings everyone each step) is
**rejected** (comms-heavy, induces sub-optimal decisions).

### Mid-flight unavailability = COMPLETE-then-WITHHOLD-then-DELIVER (not a lost update)
When a trainer goes `UN_AVL` mid-compute it **finishes local compute, withholds the result, and
delivers it (now stale) on return to an available state.** Delayed delivery, not cancellation:
- Sim holds the completed update at `sct` but gates *delivery* on the next `AVL_*` window:
  **`delivery_ts = max(sct, next_avail_ts)`**; it commits as a **stale** contribution (feeds
  Stage-5/6 staleness, not a lost-update path).
- **AWARE agg:** on the `UN_AVL` event, **frees the slot** (replacement selectable) AND **tracks the
  pending withheld delivery separately** (two ledgers — see Challenge 4).
- **UNAWARE agg:** no replacement; just waits for the delayed delivery.
- Late withheld update on return (Q-new-2): **async aware (felix) = accept-stale** through the
  baseline's existing staleness path; **sync (feddance) = reject if staleness exceeds the baseline's
  (lower) tolerance** — reuse the existing threshold, do NOT invent a new scalar.
- **Distinct from "busy"** (PARITY.md dead-end: do NOT route busy→`UN_AVL`). Busy = `AVL_*` but
  occupied (hold slot, returns on time). Withheld = result exists, delivery deferred. Separate states.

### Timing, comms, determinism
- **Transitions are continuous / event-scheduled at the exact transition vclock**, reflected at the
  aware agg **immediately (lag = 0 first cut)** — not sampled-at-selection (a mid-round transition
  must take effect mid-round). The clock-advance/commit path must consult the **next transition time**
  so it can't skip a mid-window change: clamp the advance to `min(next_sct, next_transition_ts)`.
- If lag is ever modeled it lives on the **vclock** (per the §S.dur lesson: selector-fed quantities
  must be intrinsic, never wall-contaminated).
- Oracular-pull is deterministic given trace+clock. The event path stays deterministic too (first cut
  = instant oracular reflection); the true-message path must order events by vclock with a defined
  tie-break to preserve `SEED=1234` real+sim parity (Challenge 6).
- **Everything config-gated, default OFF** ⇒ byte-identical to today's 46/46 scoreboard. Per-baseline
  `availability_aware: bool` + `availability_trace: <name>` + master `sim_unavailability` gate.

### Trace representation
- Collapse to **one event-trace representation** (`AVL_TRAIN/AVL_EVAL/UN_AVL`, strictly more
  expressive than duration-windows; derive windows if a path still needs them). **Prefer 3-state
  traces**; 2-state ({avail, unavail}) is allowed but **limited-utility for aware baselines**
  (felix/fluxtune act on the `AVL_TRAIN↔AVL_EVAL` task-type split a 2-state trace collapses; oort is
  indifferent). Surface trace granularity in telemetry.
- Single-source the trace + resolver like `client_duration.py` was single-sourced in §S.dur; both the
  trainer side (transitions/telemetry) and agg side (oracular driver) read the identical object.
  Traces (`mobiperf_2st/3st`, `syn_0`=100%, `syn_20`, `syn_50`) cover all n=300, loaded identically
  both sides (Q-new-3).

---

## 2. First-principles factors (the why behind each decision)

- **F1 Clock authority.** One monotonic vclock owns "now"; every availability decision is indexed by
  it, never wall, never a per-trainer frozen clock. The trace is **sim-seconds since run start**; sim
  (vclock) and real (wall-elapsed) must index the SAME windows — this is parity rung **A3** (the REFL
  HIGH-1 hazard), a CONTROL for the whole feature.
- **F2 Source of truth** — per-baseline (§1). Both models share one trace + one resolver + one effect.
- **F3 Event semantics (what a transition CAUSES).** `→UN_AVL` aware: excluded from selection AND
  freed from `selected_ends`/`all_selected` (`channel.update_state`), in-flight update withheld not
  discarded. `→UN_AVL` unaware: agg waits (no proactive free). `AVL_TRAIN→AVL_EVAL`: train-pool
  removal, eval-eligible only. `→AVL_TRAIN`: re-enters pool + triggers withheld delivery. The driver
  must produce these effects deterministically at the transition instant, not merely filter the next
  selection.
- **F4 Withhold-then-deliver** (§1). Return-stage fates: on-time / straggler-hold / withheld-then-
  delivered. NO permanent cancellation. `_sim_hold_busy_slots` and oort `pending_after`/carry-over
  must keep a withheld end accounted until its delayed delivery commits.
- **F5 Comms** — unaware: zero (oracular pull); aware: bounded by # real transitions (instant
  reflection first cut), per-tick broadcast rejected.
- **F6 Timing** — continuous/event-scheduled, lag 0 first cut (§1).
- **F7 Regression surface (won mechanisms).** §3.resid `_sim_hold_busy_slots` (aware frees slot +
  tracks pending; unaware holds until delivery); §4.5 `pending_after` / §4.9 carry-over (withheld end
  must NOT re-enter pool during the down window, but its delivery must still commit, stale); §S.pacer
  /§S.dur/A2c selector inputs (fewer candidates move the `pref` percentile — expect A2/S2/A2c shift;
  score only genuinely-eligible trainers); per-baseline return stages each gain a "trainer vanished"
  branch. All config-gated ⇒ default-off keeps byte-identity.
- **F8 Determinism** — oracular-pull seed-stable; event path ordered by vclock (Challenge 6).
- **F9 Trace** (§1).
- **F10 Starvation.** `oort/top_aggregator` already has a `max_retries` wait-retry for too-few-
  available; under real unavailability it fires. In sim the wait must **advance the vclock** (jump to
  next availability event / next in-flight `delivery_ts`), not spin on wall (Stage F).

---

## 3. Concepts to keep crisp (naming discipline, PARITY.md)

- **availability state** (`AVL_TRAIN/AVL_EVAL/UN_AVL`) × **busy/occupied?** × **has in-flight update?**
  — three orthogonal axes, never conflate (the busy→UN_AVL dead-end).
- **transition instant** (vclock the state changes) vs **observation instant** (vclock the agg acts);
  lag = observation − transition (0 if continuous).
- Return fates: on-time / straggler-hold / withheld-then-delivered (stale). No cancellation.
- **agg awareness** ⊥ **how it learns** (oracle read vs event message).
- `_sim_now()` must stop meaning "last dispatch ts" — availability reads the **global vclock** (or is
  fully agg-driven so the trainer never needs `_sim_now` for it).

---

## 4. New parity rungs (Stage 2 = Availability; extend the ladder)

Each new mechanism leaves the finest-grained check that localizes it (PARITY.md Growth rule).
- **A1 avail_composition** (exists, trivial at 100%): now match per-state counts over the run, binned.
- **A3 trace_time_base_consistency** `[NEW]` (CONTROL/DIST, dep K3): same trace → same windows both
  modes. **Hard gate — do not read A1/A2/A4 until A3 passes** (the REFL HIGH-1 / Challenge 2 lesson).
- **A4 per_trainer_duty_cycle** `[NEW]` (MECHANISM/DIST, dep A3): on/off fraction per trainer matches.
- **transition_effect** `[NEW]`: counts of UN_AVL slot-frees (aware), withheld-then-delivered updates,
  AVL_TRAIN→AVL_EVAL demotions; sim vs real.
- **withheld_delivery** `[NEW]`: dist of `delivery_ts − sct` (down-window delay) + resulting staleness
  (cross-checks F4 against Stage-5/6 U3).
- **observation_lag** `[NEW]`: transition→effect lag (tests F6; **must be ≈0** first cut for aware).
- **eligible_pool_reduction** `[NEW]`: A2 (`num_eligible`) tracks real's reduction, not just at 100%.
- **Regression guard:** re-run the syn_0 90-min all-baseline parity with availability OFF → current
  scoreboard byte-for-byte (config-gating proof).
- **Ramp:** `syn_0` (regression) → **`syn_20` (first validation target)** → `syn_50` → `mobiperf_*`.
  Shortest run per effect (run-length budget table); reserve long runs for C1/C2. First mechanism pass
  = 45-min "one rung" budget, 5-min smoke first.

---

## 5. Staged implementation + testing plan

One mechanism per stage, each gated by its own tests + a syn_0 byte-identity regression + (where it
changes dynamics) a short syn_20 run. All config-gated, default OFF. Build the **unaware** (oort,
pure pull) path before the **aware** (felix, event+eviction+withhold) path — strictly simpler, shares
the substrate. Context-free names (`_ts`/`_time_s`, `_round`).

### Stage A — Substrate: one trace, one resolver, one clock (NO behavior change)
- **A.1** New `flame/availability/trace.py` (mirrors `client_duration.py`): `load_trace(trainer_id) →
  SortedDict[ts→state]` and `state_at(trace, t) → TrainerAvailState` (the binary-search currently
  inlined in `get_curr_unavail_trainers` / `oracular_trainer_avail_check` / trainer
  `check_and_update_state_avl`). Replace all three call sites; collapse `trainer_unavail_durations`
  onto the event-trace.
- **A.2** Config surface: per-baseline `availability_aware`, `availability_trace`, master
  `sim_unavailability` (default False). `syn_0` ⇒ no UN_AVL events ⇒ inert even when on.
- **A.3** Fix `_sim_now()` frozen clock (F1/F8): trainer availability reads the **global vclock**, not
  `_sim_send_ts`. Route: make availability fully agg-driven; agg stamps the current vclock onto every
  message the trainer receives, trainer uses that as "now" for its notification telemetry.
- **A.4** Telemetry (on the vclock): `avail_change` (move off wall), `agg_observed_state` (per-trainer
  belief at each selection), trace granularity per run.
- **Tests:** resolver determinism + parity with the old inlined searches; frozen-clock deadlock cannot
  recur; `sim_unavailability=False` ⇒ byte-identical. **Exit:** syn_0 90-min all-baseline parity holds
  the scoreboard byte-for-byte.

### Stage B — A3 time-base CONTROL (gate for everything above it)
- **B.1** A3 `trace_time_base_consistency` (CONTROL/DIST, dep K3): resolved on/off windows align
  between modes within tolerance. **B.2** A4 `per_trainer_duty_cycle` (dep A3).
- **Tests + a 5-min syn_20 smoke** to populate A3/A4 (checker-side, validates instantly vs stored
  dirs). **Exit:** A3 PASS on a syn_20 smoke for oort.

### Stage C — UNAWARE oracular-pull driver (oort, refl)
- **C.1** Activate `get_curr_unavail_trainers()` for oort/refl via the Stage-A resolver on
  `_vclock.now` → `channel.set_curr_unavailable_trainers`. **Gates new selection only** (Q-new-1).
- **C.2** Withheld-then-deliver for the pull path: an in-flight trainer entering `UN_AVL` is **not**
  evicted; its modeled update is held and delivered at `delivery_ts = max(sct, next_avail_ts)`,
  committing **stale**. Held end must NOT re-enter the pool during the down window (extend §4.5
  `pending_after`: exclude on `vclock < delivery_ts`, not just `< sct`).
- **C.3** Ordering: drain/commit order must key on **`delivery_ts`** for held ends (a withheld delivery
  has `delivery_ts > sct`), or past-dating reappears (Challenge 1).
- **Tests:** unaware never frees the in-flight slot; withheld delivers at `max(sct,next_avail)`,
  commits stale; held end excluded until delivery; determinism.
- **Validation:** syn_20, 45-min, oort+refl. Read A1, A2 (eligible-pool reduction), A4, withheld-delay
  dist + staleness (U3). **Exit:** A1/A2/A3/A4 PASS, no new past-dating (U6), K2/K3b hold vs a syn_20
  real reference.

### Stage D — AWARE immediate-event driver (felix; fluxtune if in scope)
- **D.1** Instant reflection: at every selection AND at each transition vclock the agg recomputes
  per-trainer state from the resolver (stop-gap for the push message); effect path identical to a real
  `avl_*` message (future swap = transport only).
- **D.2** Continuous/event-scheduled timing (F6): advancing the vclock toward the next commit, clamp to
  `min(next_sct, next_transition_ts)` so a mid-window `UN_AVL` takes effect before the in-flight `sct`.
- **D.3** UN_AVL eviction + reselect: free the slot from `selected_ends`/`all_selected` (reuse
  `channel.update_state` logic), make a replacement selectable, AND register the withheld delivery
  separately (`pending_withheld[end]=delivery_ts`). Reconcile with §3.resid `_sim_hold_busy_slots` —
  slot freed *and* pending delivery tracked, no leak, no double-count (Challenge 4).
- **D.4** `AVL_TRAIN↔AVL_EVAL`: task-type eligibility change via the resolver; only meaningful where
  the baseline dispatches eval (sync oort dispatches 0 — felix-relevant; flag inert baselines,
  Challenge 8).
- **D.5** Late withheld update = accept-stale (async) through felix's existing staleness path (no new
  accept path; Q-new-2 async branch).
- **Tests:** UN_AVL frees slot + tracks pending (no leak); replacement selectable; event-scheduled
  clamp fires a mid-window transition before the straggler's sct; AVL_EVAL gates task type; late update
  commits stale.
- **Validation:** syn_20, 45-min, felix. A1/A2/A3/A4, transition-effect counts vs real, observation_lag
  ≈0, withheld-delivery, U3, U6. **Exit:** mechanism rungs PASS at syn_20; then a full-length run to
  re-confirm felix C1/C2 under unavailability.

### Stage E — SYNC baselines + staleness-gated rejection (feddance)
- **E.1** Apply C/D to the sync path (feddance is availability-aware via its predictor; barrier
  re-selects the cohort each round).
- **E.2** Staleness rejection: a late withheld update exceeding feddance's tolerance is **dropped**
  (baseline's existing rule, no new threshold). Changes round composition → expect K8/U2 movement;
  validate it's faithful (Challenge 9).
- **E.3** §6.u6 barrier-anchor: unavailability changes which K form the barrier; the barrier-anchored
  U6 lag must be computed over the *actually contributing* cohort.
- **Tests + syn_20 45-min feddance.** **Exit:** A-rungs + U3/U6 + K8 PASS.

### Stage F — Starvation / clock-advance under scarcity (F10)
- **F.1** In the `max_retries` wait-retry, when no one is selectable, **advance the vclock to the next
  availability event (or next in-flight `delivery_ts`)** rather than wall-sleeping. Clamp to the
  nearest of {next transition, next `delivery_ts`, next `sct`}; guard against an all-unavailable window
  spinning forever (Challenge 10).
- **Validation:** syn_50, 45-min (heavier unavailability triggers scarcity), all baselines. **Exit:**
  no stalls; K1 monotone; round cadence faithful at syn_50.

### Stage G — Ladder integration + ramp + sign-off
- **G.1** Land all new rungs in `scripts/parity/{checks.py,report.py}` with deps (A1 now enforced, A3,
  A4, transition_effect, withheld_delivery, observation_lag, eligible_pool_reduction). Append-only.
- **G.2** Ramp: syn_0 → syn_20 → syn_50 → mobiperf_*. **G.3** Per-baseline sign-off, now *with*
  availability.

### Stage H (FUTURE) — true `avl_*` message transport
Swap the aware-path instant-oracular reflection for real trainer→agg `avl_*` messages, processed
immediately, **without changing the effect logic** (D.1 was built for this). Preserve determinism by
ordering events on the vclock with a defined tie-break (Challenge 6).

---

## 6. Challenges / land-mines

1. **Ordering must key on `delivery_ts`, not `sct`, for withheld updates** (high risk of
   re-introducing past-dating). §3.drain's min-`sct` gate and §4.9 carry-over assume `commit order ==
   sct order`; a withheld delivery commits at `max(sct,next_avail) > sct`. Re-validate U6/U3 after C/D.
2. **A3 time-base drift is the silent killer.** If sim vclock and real wall-elapsed advance at
   different rates, the same trace makes a trainer unavailable at different real moments → every higher
   rung diverges and mislocalizes. Hard CONTROL gate; do not read A1/A2/A4 until A3 passes.
3. **Two-tolerance trap on the eligible pool (A2 vs S3/4).** With availability ON,
   `eligible = candidates − in_flight − unavailable`; a small gap can fail A2's tight KS while S3/4
   in_flight passes. Decompose the channel first; don't chase A2 as a separate bug.
4. **Busy vs unavailable vs withheld = three distinct non-pool states** (slot-leak / N≈300 ramp). The
   §3.resid dead-end (busy→UN_AVL) ramped in-flight to ~300. Aware UN_AVL **frees** the slot but still
   **tracks** the pending delivery — slot accounting and delivery accounting are separate ledgers;
   conflating them leaks slots or double-commits.
5. **Stop-gap fidelity to real felix may not be lag-free.** Real felix uses MQTT notifications with
   real delivery+processing lag; the stop-gap models lag 0. If real's lag is non-trivial,
   `observation_lag` parity (sim 0 vs real >0) diverges and selection contexts won't match. Measure
   real's notification lag early; if material, model it on the vclock or accelerate Stage H (the §S.dur
   lesson: don't let a wall lag contaminate a vclock-indexed decision).
6. **Determinism / event tie-break at a shared vclock instant.** Transition, commit, and selection
   events can coincide. Define a total order (e.g. transitions < commits < selections, then by
   trainer_id) so `SEED=1234` real+sim parity + exact rungs stay enforceable.
7. **Compound states with existing carry-over.** An oort §4.9 carried-over straggler that ALSO goes
   UN_AVL, or a §3.resid held slot whose trainer flips AVL_TRAIN→AVL_EVAL, are real cases. Enumerate
   the (avail_state × occupied × in-flight) cross-product and assert each cell in tests.
8. **AVL_EVAL may be inert for some baselines** (sync oort dispatches 0 eval). A 3-state trace's eval
   windows then do nothing; report which baselines exercise the eval split (ties to the 2-state
   limited-utility flag, F9).
9. **Staleness-rejection on sync changes round composition (feddance).** Dropping over-stale late
   updates shifts K8/U2/round-count — expect movement, validate it's faithful, reuse the existing
   threshold (no new scalar).
10. **Scarcity clock-advance must not stall or fast-forward past events** (F10/Stage F). Clamp the jump
    to the nearest of {next transition, next `delivery_ts`, next `sct`} and terminate on an
    all-unavailable window. Guard K1 monotone + the K5 failsafe ceiling.
11. **Regression discipline.** Every stage re-runs the syn_0 90-min all-baseline parity and must hold
    the scoreboard byte-for-byte before its syn_20 validation counts. A stage perturbing another
    baseline serializes (one baseline per run round).

---

## 7. Open follow-ups (note here as work lands)

- **Q-new-2 sync confirmation:** verify feddance's existing staleness threshold is the right rejection
  gate on a real syn_20 run before E.2 (don't assume the async tolerance transfers).
- **Real notification lag (Challenge 5):** measure on a real felix run early; decides whether lag 0 is
  admissible or Stage H must move up.
- *(Append new open questions/info needs here as the substrate lands — keep this the single ledger.)*
