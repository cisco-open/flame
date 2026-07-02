# Sim Unavailability -- Design & Remaining Work

Design + status for modeling client **unavailability** in the FLAME FL simulator. The mechanism is
built and landed across all six async_cifar10 baselines; **felix is confirmed at full parity**. This
doc now carries mostly **open items + next-steps**; landed history is collapsed to pointers (full detail
in git). The fwdllm port has its own plan in
[../fwdllm/simulate_fwdllm.md](../fwdllm/simulate_fwdllm.md) (parity rungs in
[PARITY.md §F](PARITY.md)).

---

## Status (Jul 2) -- felix CONFIRMED, fedbuff blocked, cross-baseline validation pending

- **felix: FULL PARITY, 62/62 enforced checks** (live n=300 syn_50, `--runtime-s 3150`). Clock/timebase,
  availability composition, selection, dispatch/training phases, staleness, and outcomes all PASS; real
  self-stops cleanly (`"stopping run"`, no external kill, no `SIM_WALL_CEILING`). This is the reference
  baseline the project set out to validate -- essentially PR-ready on its own.
- **fedbuff: NOT yet trustworthy.** Its run from the same campaign was **confounded** by a concurrent
  felix-real run: fedbuff-real's per-round pacing collapsed ~3 min after felix-real launched (5-10s/round
  -> 40-80s+/round) while felix-real stayed steady. The two ran on **separate nodes** with flat RAM / 0%
  GPU on fedbuff's node, so it is **not** local CPU/GPU/RAM contention -- leading suspects are a shared
  MQTT broker under ~600 concurrent real trainer connections, a shared network path, or shared
  storage/telemetry-write contention. Not yet root-caused.

### Next steps (this workstream)
1. **fedbuff shared-resource investigation** -- root-cause what collapsed fedbuff-real's pacing when
   felix-real started despite separate nodes (broker / network / storage). Do this before a blind re-run.
2. **fedbuff isolated re-run** (+ sim) once understood, or at minimum confirmed no other n=300 real run is
   active anywhere in the shared environment during its window; re-check parity.
3. **PR scoping** -- felix is PR-ready (62/62, clean self-stop). Decide whether to scope the PR to felix
   now and follow up on fedbuff, or hold for both per the standing "land baselines together" agreement.

### Next steps (parallel branch -- can proceed independently, merge fixes back as they land)
Exhaustive cross-baseline validation is **not** a blocker for landing the substrate. On a parallel branch,
run the full parity campaign for **oort / oort_star / refl / feddance** across **syn_0 / syn_20 / syn_50 /
mobiperf**, sim+real, at n=300; resolve the still-open §7 rows; fixes merge back into `dg-fork-main` as
they come. See "Open items" for the ordered gates (legacy `trackTrainerAvail` cleanup, mobiperf live
exercise) and PARITY.md's run-length budget.

---

## Preamble -- what this is

**Goal.** Model client *unavailability* (devices dropping offline mid-training) so a fast **simulated**
run (virtual clock, no real sleeps) reproduces what a **real** run (wall-clock, MQTT, true delays) does --
**sim/real parity** -- for every baseline, **config-gated and default-OFF** (byte-identical to today when off).

**What was built (v1).** A shared availability substrate (`flame/availability/trace.py` +
`ClientAvailability`) mixed into the syncfl base and inherited by asyncfl/oort, so all baselines share one
trace-read effect path:
- **Send-time gate, deliver-late-stale.** A trainer that goes UN_AVL mid-flight *keeps computing*; its
  upload is gated at send-time (real) / buffered to `delivery_ts = max(sct, next_avail)` (sim) and
  committed later as a stale update. Nothing is cancelled or dropped.
- **Two ledgers, never conflated.** Slot ledger (90s vclock *abandon* frees the in-flight slot) + delivery
  ledger (`pending_withheld[end]=delivery_ts`, commits through the existing staleness gate).
- **Proactive in-flight eviction** -- **felix only** (the one fully-aware baseline): frees a slot the trace
  shows UN_AVL at the next selection boundary, no 90s wait.
- **Starvation / vclock-advance under scarcity.** When the eligible pool is too small to start a round, sim
  advances the vclock to the next availability transition instead of spinning (self-terminating, B2.0.2).
- **Absolute (vs. ground-truth-trace) fidelity checks** on top of the relative (real-vs-sim) ones: A6
  (trainer state), A7 (aggregator belief, per selection/commit checkpoint), A8 (send-gate wait), K11
  (commit promptness). Relative checks answer "do the two modes agree?"; these answer "is either one
  *correct*?" -- shared `scripts/parity/ground_truth.py`, one canonical trainer<->aggregator time origin.
- **Parity ladder** (`scripts/parity/`) -- availability rungs A1/A3/A4/A4dur/A5/A6/A7/A8/K11,
  withheld_delivery, abandon_timeout, starvation_advance, eligible_pool_reduction.

**Two orthogonal axes per baseline (keep separate).**
1. **Knowledge at selection** (`avail_select_filter`): does the selector read the trace to avoid
   *selecting* currently-UN_AVL trainers? aware = yes, unaware = select blind.
2. **In-flight slot-free timing** (`proactive_inflight_evict`): when a *dispatched* trainer goes UN_AVL
   mid-round, free its slot at the next boundary (proactive, felix only) or wait the 90s vclock abandon
   (reactive-90s, everyone else). Aware-at-selection != in-flight eviction.

The knowledge *model* is **trace-read** for all v1 baselines; message-transport (`client_notify`) and
predictive models are **Stage H** (future).

---

## Working agreement (standing -- read every session)

1. **Common first, one baseline first.** Land shared/library changes once, drive a single reference
   baseline; don't fan out until it behaves.
2. **Short runs to debug, long runs to confirm.** Gate on unit tests + syn_0 byte-identity + shortest
   syn_20 smoke. Long runs confirm; never find first bugs.
3. **Local deterministic tests before runs.** Prefer a synthetic-trace pytest that exhibits the bug over a
   long run that hunts for it.
4. **Keep this doc crisp.** Completed stages: mechanism + where it lives + exit (2-3 lines). Full detail
   only for active/next; dead-ends in §6.
5. **No stale content.** The moment a section is superseded -- a bug fixed, a task landed, a prediction
   resolved, a next-step taken -- collapse it to a 2-3 line note (or delete) **in the same edit**.
6. **Whole-doc crisp pass on every edit.** Re-read top to bottom and push down anything the new result also
   supersedes. The top of the doc carries only **open issues** + **next steps**; run commands, wall-time
   estimates, and per-run hypothesis tables for finished work belong in raw logs.

---

## Baseline matrix (CANONICAL)

| baseline | sync/async | agg base / entry | knowledge @ selection (`avail_select_filter`) | in-flight slot-free (`proactive_inflight_evict`) | config-gate (as run) |
|---|---|---|---|---|---|
| **felix** | **async** | `asyncfl` (<- syncfl) / `main_asyncfl_agg.py` | aware | **proactive** (felix only) | `simUnavailability` |
| **fedbuff** | **async** | `asyncfl` / `main_asyncfl_agg.py` | unaware | reactive-90s | `simUnavailability` |
| **oort** | **sync** | `oort/top_aggregator` / `main_oort_sync_agg.py` | unaware | reactive-90s | `simUnavailability`+ |
| **oort_star** | **sync** | `oort/top_aggregator` / `main_oort_sync_agg.py` | aware | reactive-90s | `simUnavailability`+ |
| **refl** | **sync** | `syncfl` FedAvg / `main_fedavg_agg.py` | aware | reactive-90s | `simUnavailability`+ |
| **feddance** | **sync** | `syncfl` FedAvg / `main_fedavg_agg.py` | aware | reactive-90s | `simUnavailability` |

+ oort/oort_star/refl's `baselines.yaml` catalog entry still carries the legacy `trackTrainerAvail:
{enabled: True, type: ORACULAR}` block (pre-dates this project). In every `debug_run.sh --trace`-launched
run the substitution sets `simUnavailability=True` for them too, so they run the same modern path. The
legacy block only matters if the parity YAML is loaded *without* that substitution -- untested territory,
and the reason it isn't cleaned up yet (removing it risks silently disabling availability in that path).
See Open item 1.

**Notes.** (1) `ClientAvailability` lives in `flame/availability/client_availability.py`, mixed into
`syncfl/top_aggregator.py` (`class TopAggregator(ClientAvailability, Role)`); asyncfl/oort extend it -- all
six share the substrate. (2) trace-read is v1; the knowledge model becomes message-transport / predictive
in Stage H, but the select-filter / in-flight-evict *behavior* is unchanged. (3) **felix is the only
baseline that de-selects an in-flight trainer** when it goes UN_AVL; the aware-at-selection-only baselines
still hit the 90s abandon for mid-round drop-offs.

### Flag reference
- `avail_select_filter: bool` -- selector excludes currently-UN_AVL trainers from the **selection** pool
  (`get_curr_task_ineligible_trainers`). ON: felix/oort_star/refl/feddance. OFF: oort/fedbuff.
- `proactive_inflight_evict: bool` -- gates `_sim_evict_unavail_inflight` (in-flight boundary eviction).
  ON: **felix only**. OFF: everyone else (reactive-90s).
- `tracking_mode` -- knowledge-model axis: `trace_read` (v1, live) | `client_notify` (Stage H) |
  `predictive` (future). Replaces the `oracular` value at concept/log level (YAML field *value* compat kept).

---

## v1 core decisions (resolved -- durable reference)

- **Knowledge model:** trace-read for all; one shared trace + `state_at(trainer, vclock)` + one effect path.
- **Mid-flight UN_AVL = compute-completes, gate the send, deliver-late (stale).** Real: gate at send-time.
  Sim: buffer at `delivery_ts = max(sct, next_avail_ts)`.
- **Two ledgers, never conflated:** slot ledger (frees `selected_ends`) + delivery ledger
  (`pending_withheld`, commits stale through the staleness gate). Order commits by `(delivery_ts, end_id)`,
  never `sct`.
- **Busy != unavailable != withheld** -- three distinct non-pool states. Never route busy->UN_AVL.
- **All availability time on the vclock in sim.** Never wall, never a frozen per-trainer clock.
- **Config-gated, default OFF** => byte-identical. `simUnavailability` is the gate for all 6 baselines as
  run (see Baseline matrix + note).
- **availability state** (`AVL_TRAIN/AVL_EVAL/UN_AVL`) x **busy?** x **has in-flight update?** are
  orthogonal. `syn_0/20/50` are 2-state (no AVL_EVAL); `_trace_has_avl_eval` guard collapses D.2 for them.

---

## Parity rungs (availability tier -- what exists)

- **A1** `avail_composition` (per-state counts, binned). **A3** `trace_time_base_consistency` -- CONTROL
  hard gate (dep K3). **A4** `per_trainer_duty_cycle`. **A4dur** duration-weighted TVD
  (`mean_err<=0.05`, `frac_within_tol(0.10)>=0.95`). **A5** `state_timeline_agreement` -- per-(trainer,t)
  exact match. All five are **relative** (real vs sim).
- **A6** `trainer_trace_fidelity`, **A7** `agg_belief_fidelity` (tagged `selection`/`commit`), **A8**
  `send_gate_wait_fidelity` (real-mode only) -- **absolute** (vs. ground-truth trace), independently per
  mode. A7 is mechanism-agnostic (trace_read now, client_notify/predictive later).
- **K11** `commit_promptness` (INV) -- per-event: actual commit vs. earliest-legally-committable time,
  generic over gate reason; primary promptness gate. `withheld_delivery` / `abandon_timeout` stay as
  secondary distributional diagnostics.
- **eligible_pool_reduction** (`Aa`, HELD), **observation_lag** (live in v1 trace_read via A7),
  **starvation_advance** (vclock jumps under scarcity). Calibrate HELD rungs at mobiperf.
- **Ramp:** syn_0 -> syn_20 -> syn_50 -> mobiperf_*.

---

## Challenges / land-mines (durable -- consult before the fwdllm port)

Most are resolved; the still-open ones (3, 15) matter for cross-baseline validation and the port.

1. Ordering on `delivery_ts`, not `sct` -- resolved (U6/U3 validated).
2. A3 time-base drift -- resolved; hard CONTROL gate; 90s abandon re-clocked to vclock.
3. **A2 two-tolerance trap (OPEN, watch).** Bimodal sim vs smoother real -> KS shape artifact; means match;
   improving with run length (0.437->0.338). Expect <=0.2 at n=300/3h -- confirm at T5.
4-14. Resolved: busy/unavail/withheld three ledgers (4); real send-gate fidelity (5); determinism via
   `(delivery_ts,end_id)` (6); compound straggler x UN_AVL (7); AVL_EVAL inert for oort + `_trace_has_avl_eval`
   guard (8); staleness-on-sync cohort movement (9); scarcity advance skips no events (10); syn_0
   byte-identity discipline (11); library mixin spans examples, never example-local (12); empty per-task
   pool corrupting `selected_ends` -- fixed by keying cleanup off `connected_ends` in all 3 selectors,
   **still needs live mobiperf_3st exercise** (13); scarcity threshold via F.2 unified pattern (14).
15. **Per-baseline in-flight accounting + scenario sizing (OPEN, per-baseline).** In-flight is NOT constant:
    oort (sync, over-selects) `in_flight ~= overcommitment*agg_goal - completed`; felix/fedbuff (async)
    concurrency-bound, can exceed agg_goal; refl/feddance (sync FedAvg) clear `selected_ends` each round,
    feddance returns *partial* selections so `eligible ~= (1-unavail)*n`. Manage in-flight per baseline; do
    NOT assume "sync has no in-flight term." syn_50 caps ~43% unavail, so feddance's straddle window is
    narrow (n~19) -- size `n ~= threshold / (1-unavail_frac)`.
16-18. Resolved: real syncfl recv-barrier bounded `timeout=min(90s,budget)` (B2.0.1, 16); sim starvation
   self-termination `>`->`>=` budget check (B2.0.2, 17); real-mode trace-clock join-ramp re-anchor
   (B2.0.3, 18) -- correct, but was masking item 20.
19. Resolved (Batch 4): oort/felix `K6 sim_send_ts` -- sim `Trainer._sim_now()` froze at last dispatch;
   fixed via due-ts stamping + EOT final wake-up. K6/A6 PASS on the felix n=300 run.
20. Resolved (Batch 3 T3.1a): `debug_run.sh` never wired the trainer's own `client_notify.trace`, so real
   trainers ran their send-gate against always-available `syn_0` regardless of `--trace` (sim was
   aggregator-driven, so unaffected). Fixed + regression-tested
   (`tests/launch/test_debug_run_trace_substitution.py`). This was the true cause behind the feddance A3/K3b
   symptoms earlier blamed on clock-origin.

---

## Dead-ends (settled -- do not retry)

- **busy -> UN_AVL routing** -- three distinct states.
- **Frozen per-trainer clock** (`_sim_now()` = last-dispatch ts) -- stuck UN_AVL forever; read `_vclock.now`.
- **Wall-clock in sim** for selection gate / 90s abandon -- wall barely advances vs vclock.
- **Per-tick MQTT broadcast** -- comms storm; v1 = trace-read pull (zero comms).
- **Ordering withheld commits by `sct`** -- past-dating; order by `(delivery_ts, end_id)`.
- **Forking withhold/abandon per stack** -- single shared `ClientAvailability`.
- **A4 bare transition fraction** -- brittle in trace-read mode; replaced by A4dur + Aa.
- **D.2 excluding AVL_TRAIN from eval on 2-state traces** -- empty eval pool wiped `selected_ends`; fixed by
  `_trace_has_avl_eval` guard.
- **Subtracting starvation vclock-jumps from the budget** -- breaks parity (real polls scarcity on wall
  budget; sim vclock jump consumes virtual budget symmetrically). Size `--runtime-s` accordingly.

---

## Known parity failures (non-blocking -- resolve at T5 with long-run data)

Resolved this project (full detail in git / §7 history): A4dur (all baselines, real selection missing
`vclock_now` -> fixed, `mean_err=0.0`); feddance A3 (item 20); K6/A6/A7-commit/asyncfl-TIMEOUT (Batch 4,
confirmed on felix n=300). Still open, for the parallel-branch campaign:

| Check | Baseline | Status |
|---|---|---|
| A2 `eligibility` KS | oort | 0.437->0.338 (1.5h->3h); FAILs @ syn_20 n=300 short. Bimodal-vs-smooth shape; means match. Investigate with A4dur. |
| A2 `eligibility` KS | feddance | FAILs @ syn_20 n=300 smoke; earlier "clears at n=300" (n=25) not confirmed -- re-open. |
| K3b `overhead_residual` | oort | rel~=0.116; run-length sensitive; P3 gates at n=300. |
| P3 `trainer_speed` | oort | ratio=1.153 (tol 1.15); marginal tail at n=300; gates K3b. |
| `throughput` | oort | FAILs @ syn_20+syn_50 n=300 smoke; baseline-specific, lower priority. |
| `avail_composition`/`commit_visibility`/`total_commits` | fedbuff | FAILs @ syn_20 n=300 smoke -- but see fedbuff confound (Status); re-check after isolated re-run. |
| C2 `loss` | feddance | avg_diff~=0.16 (few eval pts); early-training noise at alpha=0.1; K8/C1/utility PASS. |
| U5 `inter-arrival` rho | feddance | 0.659->0.381 (syn_20->50); watch at mobiperf. |

---

## Open items -- pick up in order

1. **Legacy `trackTrainerAvail` cleanup (oort/oort_star/refl) -- flagged, NOT done.** Their `baselines.yaml`
   entries still carry `trackTrainerAvail: {enabled: True, type: ORACULAR}`. In every `debug_run.sh
   --trace` run this resolves to the same `simUnavailability=True` path (a substitution quirk, not a static
   value). Do **not** just zero `enabled`/`type` -- that risks silently disabling availability for any
   invocation not going through the substitution script. Safe path: (a) add `simUnavailability: true`
   statically wherever their trace is set; (b) only then zero the legacy block; (c) verify via an actual
   generation + short real run (no pytest coverage of `debug_run.sh`'s generator).
2. **Minimal mobiperf live exercise before "mature enough to port."** The entire mobiperf (3-state,
   AVL_EVAL) path is untested live so far (only syn_0/20/50, which are 2-state and collapse AVL_EVAL away).
   Challenge 13's fix explicitly still needs live exercise at mobiperf_3st. ~30-45 min for one async
   (felix) + one sync (refl or oort_star) baseline exercises AVL_EVAL + the empty-pool cleanup live and runs
   Batch 3's checks against a 3-state trace.
3. **PR workflow.** felix is confirmed (62/62). Gated for a felix+fedbuff PR on the fedbuff isolated re-run +
   concurrent-run investigation (Status). The fwdllm learnings are now carried into
   [../fwdllm/simulate_fwdllm.md](../fwdllm/simulate_fwdllm.md) and [PARITY.md §F](PARITY.md); the design
   decisions kept (this doc's v1 decisions / land-mines) and rejected (Dead-ends) are the durable record.

### Minimal bar before porting fwdllm (not a full T5 gate)
Don't gate the port on the full 3h x 6-baseline x 4-trace campaign (that's for paper-quality numbers). Bar:
(1) Batch 3 (T3.0-T3.5) at least through felix -- **done** (A4dur confirmed; item 20 fixed; absolute
fidelity checks live). (2) mobiperf_2st for one async + one sync baseline (~30-45 min) to exercise AVL_EVAL
+ Challenge 13's cleanup live for the first time. (3) The full syn_50/mobiperf 6-baseline sweep is
separable -- parallel branch, not a pre-port blocker.

---

## Stage H (future -- out of scope)

Two independent knowledge-model upgrades, both replacing `trace_read` on the `tracking_mode` axis; the
effect logic (select-filter / in-flight-evict) is unchanged -- only how the agg learns state changes:
- **H.1 Message-transport** (`client_notify` ON for aware baselines): trainers push avl-state changes over
  MQTT instead of the agg reading the trace + a continuous/event-scheduled vclock clamp. Re-measure
  `observation_lag` (must be ~0) once live. (fwdllm's fluxtune baseline already configures `client_notify`
  -- see simulate_fwdllm.md D1.)
- **H.2 Predictive**: a learned/heuristic availability model (no trace read or message push). Not designed yet.

---

## History (collapsed -- full detail in git)

- **A-G, C.6, D, E, F.2, B2.0.x, T0-T5, Batch 2/3/4** all landed. Substrate + A3 time-base CONTROL,
  send-gate/deliver-late, two-ledger, proactive evict (felix), syncfl path, starvation self-termination,
  the absolute ground-truth fidelity checks (Batch 3 T3.0-T3.5: A6/A7/A8/K11 + `ground_truth.py` + shared
  canonical time origin), and Batch 4's four gating fixes (asyncfl real self-stop, sim trainer-clock freeze
  / K6 / A6, A7-commit checker `max_gap_s`) are all in code + tests, confirmed on the felix n=300 run.
  Per-task mechanism/file/exit detail lives in commit history; the durable decisions are in "v1 core
  decisions", "Challenges", and "Dead-ends" above.
