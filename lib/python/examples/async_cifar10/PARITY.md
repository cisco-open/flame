# Real / Sim Parity — Methodical Causal Ladder

Living document for the async_cifar10 parity checker.
Kept in sync with `scripts/parity/checks.py` (check functions),
`scripts/parity/report.py` (stage grouping + verdict), and the pytest suite.

**Real/sim comparator — give the two run dirs, get a report JSON:**
```bash
cd lib/python/examples/async_cifar10
# single baseline: point at the real + sim run dirs
PYTHONIOENCODING=utf-8 python scripts/parity_check.py \
  --real experiments/<real_run_dir> \
  --sim  experiments/<sim_run_dir> \
  --agg-goal 10 --budget-s <runtime_s> --json-out experiments/parity_<baseline>_<tag>.json

# all baselines at once: auto-discovers the latest real/sim pair per tag
PYTHONIOENCODING=utf-8 python scripts/parity_check.py --batch \
  --experiments-dir experiments --baselines felix oort refl feddance \
  --agg-goal 10 --budget-s <runtime_s> --json-out experiments/parity.json
```
`--budget-s` = the run's `--runtime-s`. Add `--lenient` to demote DIST fails to
warnings; prints a stage-grouped report + root-cause banner.
Per-run plots: `python ../../../scripts/analysis/analyze_run.py <run>/telemetry`.

**Launch runs** (node-agnostic; any baselines/mode/duration on any machine):
```bash
bash scripts/debug_run.sh --baselines 'oort refl' --runtime-s 3600 --mode both
```
Reads `expt_scripts_2026/felix_oort_refl_feddance_alpha0.1_parity.yaml` (every
baseline × sim/real), seeds real+sim identically (`SEED=1234`, `SEED=none` to
disable), and applies the per-baseline sim fixes. Split across machines by
passing different `--baselines`.

**Readiness/regression tests** (no cluster; run under lib/python with `dg_flame`):
`pytest tests/mode/ tests/selector/test_oort_selector.py tests/sim/
examples/async_cifar10/scripts/parity/` — guards baseline wiring, the in-memory
cache, serialize-once, sim ordering (barrier/residence/carry-over), the overhead
model, deterministic seeding, and every checker rung. Last green: **146 pass / 7 skip**.

---

## Status (Jun 17)

The **base-algorithm fidelity pass** (D1–D6 + structural fixes, all config-driven,
defaulting to the Oort paper) has landed and been exercised on oort and refl. The
historic root causes are dead; what remains is one structural sim fix (oort
carry-over, implemented) plus a clean **seeded** run to separate genuine
divergence from stochastic path drift.

**Next run is ready to launch:** seeded oort/refl/felix pairs, with oort's §4.9
carry-over fix ON. Implementation is complete (believed-vs-actual telemetry+plots
validated; debug_run.sh node-agnostic). Nothing blocks the launch.

### What's landed (validated signal — good and bad)

**oort.** D1 unsorted-`pref` bug FIXED + validated (`Sd` preferred-duration
real .507 / sim .498, was a .346 gap). `system_util` — the Jun-15 root — RESOLVED
(`Sx` KS .035, was .132). Clock/throughput/accuracy all green.
*Remaining root = in-flight carry-over* (`Sr`): real carries ~3.3 stragglers
in-flight/round, sim drains to 0.15 → under-counts sim concurrency (`S3/4` 13.2 vs
16.3) and inflates real's null-utility selections. **Fix §4.9 implemented + enabled,
unvalidated** (needs this run).

**refl.** §4.5 `sct`-gated pool exclusion VALIDATED — every emergent check green
(loss, throughput, staleness, accuracy). *Remaining = participation (`S2`) +
preferred_duration (`Sd`)*, both **stochastic-class**: speed-CLASS selection rates
match; only which-individual-within-class drifts on the now-stochastic weighted
exploit. Open: gate as stochastic, or chase as genuine — **the seeded run decides**.

**felix.** REGRESSED under the paper defaults: sim over-overlaps (advance 2.1 vs
real 4.1, staleness 17 vs 2.8). Real stayed healthy → sim-side sensitivity. The
sync-Oort paper knobs changed the selected mix and the async overlap model (tuned
for the old mix) no longer spaces completions. There is **no async Oort reference**
(`third_party/Oort` is sync-only). Open: revert felix's async knobs to the
parity values, vs keep paper defaults and re-tune the sim overlap model. A seeded
felix isolates mix-driven (recovers) from clock-model (won't) collapse.

**feddance.** No overcommit (in_flight 10/10) → not a residence/carry-over case.
Divergence is **selection-mix** in `feddance_I` (last-round loss): path-dependent
stochastic accumulation, same class as §5. Likely a **checker reclassification**
(P3/K3 → DIST for stochastic selectors), not a sim bug — confirm after seeding.

**Cross-cutting:**
- **Seeding.** Dedicated per-selector RNG (`AbstractSelector._rng`/`_pyrng`,
  seeded from `hyperparameters.seed`, threaded as `_seed`), insulated from the
  process-global RNG. Selection is now a pure function of (state, seed) →
  comparable across runs/modes. Check `Sdet` (`decision_determinism`) reports
  `eligible/decision/chosen_match_frac` and splits the diagnosis (seeding worked /
  RNG desync / genuine input divergence). **Read `Sdet` first on the next run.**
- **Believed-vs-actual utility** telemetry (`EVENT_UTILITY_BELIEF`, emitted at the
  stat-utility return-overwrite in oort/asyncfl/syncfl) + 4 live plots in
  analyze_run.py (validated). Audit finding: the "low believed_I" is two non-bugs —
  it's the D2-*normalized* reward (raw stat-utility is healthy, P50/P90 ≈ 40–55/80–98);
  every exploit pick has non-null utility; the nulls are explore picks + carried-over
  in-flight (a 4th symptom of the oort carry-over root, not a defect).
- **Checker corrections** (principled, guarded, append-only): `P3` trainer_speed →
  integer-grid KS (sub-second wall-capture jitter excluded by the virtual clock);
  `A2b`/`A2c` eligible/selected speed → `training_delay_s` metadata pool (real
  leaves duration None for non-completers → observed pool sampled only fast
  completers); `Sr` residence rung; throughput family → one 5% bar; stochastic
  selectors → gated P1/S1/utility.
- **Run length.** 1 h reaches steady state for `Sdet` + all mechanism checks
  (clock, throughput, staleness, P3, selection, AND participation — seeding makes
  a real participation gap show immediately, so it's no longer horizon-gated).
  Convergence C1/C2 is the exception: the gap grows with training, so a sub-2h PASS
  is downgraded to `[??] LOWC` (one-sided; a FAIL still surfaces). **Loop on 1 h
  reading Sdet + mechanisms; reserve one 3–4 h run for final convergence sign-off.**

### Next run — hypothesis & what to watch

Launch **seeded** oort + refl + felix pairs (feddance optional; likely checker-only).
1 h fast loop first.

| baseline | change this run | hypothesis | watch |
|---|---|---|---|
| **oort** | seeded + §4.9 carry-over ON | carry-over closes the last structural gap; seeding turns mix-quantities into genuine signal | `Sr` PASS (in_flight_after 0.15→~3.3); `S3/4`→~16; committed_fresh→~10; null-util frac→real; `Sdet` decision_match≈1; A2c re-judged; believed-vs-actual plots |
| **refl** | seeded (no code change) | participation/`Sd` gaps collapse if stochastic; persist if genuine | `Sdet`; `S2` participation; `Sd`; emergent stays green |
| **felix** | seeded (no code change) | isolates how much advance-collapse is mix (recovers) vs clock-model (won't) | `Sdet`; K3 advance; staleness; overlap |

### Next steps (branching on outcome)

- **oort `Sr` passes & `Sdet` decision_match≈1** → carry-over fix confirmed; the
  residual raw-utility magnitude / A2c bias are now genuine selector signal — judge
  them, don't defer. Move to a 3–4 h convergence sign-off.
- **oort `Sr` still fails** → re-read the carry-over hold (is it firing? check
  `[SIM_CARRYOVER]` / `inflight_residence.in_flight_after`); the straggler may be
  stale-rejected before the hold engages.
- **`Sdet` decision_match≈1 but chosen_match≪1** → a selector still hits the global
  RNG / seed not threaded — fix upstream, don't trust the mix verdicts.
- **`Sdet` decision_match≪1** → candidate set/utilities differ *before* the draw;
  drop to `eligible_match` to see if it's the SET or the VALUES (availability/
  ordering vs utility), fix that input.
- **refl participation matches seeded** → gate `S2`/`Sd` as stochastic-class
  (mechanism is proven correct). **Still diverges seeded** → genuine input
  divergence; chase via `Sdet`.
- **felix recovers under seeding** → it was selection-mix; keep paper defaults.
  **Still collapses** → it's the async overlap model; either revert the async knobs
  to the parity values or re-tune the overlap model (no async reference exists, so
  anchor to the real run's spacing).
- **feddance `mean_I` horizon-bias ruled clean** → reclassify P3/K3 to DIST for
  stochastic selectors; no sim change, no re-run.

### Dead ends — do NOT retry

- Overhead > 0 on the virtual clock (masks & drifts; clock must `= max(vclock, sct)`).
- Prediction-only gates with no real blocking (never fire).
- `version_at(sct)` staleness relabel (fedbuff consumes the *real* number; inert).
- Adding `mqtt_fetch` (~57 s) to `sct` (not version-relevant; inflates staleness ~6×).
- `simRedispatchGapSeconds=0` for felix (sim over-overlaps; the gap is a real mechanism).
- Tuning sim to a *wrong* real, or any scalar fudge where a mechanism is called for.
- Re-chasing: GPU contention (overrun 0), SEND_TIMEOUT (0×), MQTT drops (0), the
  felix post-compute leg as a "bug" (it's serial-aggregator scheduling), per-trainer
  exact-set/identity on a stochastic streaming selector (path-dependent by nature).

---

## §1  Philosophy: the parity ladder

An FL run is a **pipeline**. Each round flows through the same stages in both
real and sim mode:

```
clock/time-base → availability → selection → dispatch+training
   → update-return+ordering → aggregation → utility → emergent outcomes
```

Parity must hold at *every* stage. If it breaks at stage N, every stage above N
also diverges — but those upper failures are **consequences, not bugs**. The job
of the checker is to find the **lowest broken rung**: the earliest stage whose
own inputs are sound but whose output diverges. That stage holds the root cause.

This replaces the old "severity-ordered symptom list." Severity tells you what
hurts; the ladder tells you *why*, and does so automatically.

### Three roles every check plays

Tag each check with the role it serves in localization:

- **CONTROL** — confirms an *input* to a stage is identical across modes
  (e.g. trainer_speed_s, training_budget_s, telemetry coverage). A failing
  control means the sim's inputs differ; fix the input model, not the stage.
- **MECHANISM** — confirms a *single transformation* inside a stage is modeled
  (e.g. per-commit overhead, inter-round overlap, availability time-base). A
  failing mechanism with passing controls is a *localized* bug — the prize.
- **EMERGENT** — an aggregate outcome (throughput, terminal state, convergence,
  utility). These are what we ultimately care about, but they never localize on
  their own; they only tell you *something* below them broke.

Debugging rule: an EMERGENT failure is a prompt to walk *down* the ladder to the
mechanism/control checks beneath it. Never fix an emergent symptom directly.

### Two-axis classification

Every check has two orthogonal labels:

- **STAGE** (0–9 below): where in the causal pipeline it sits. Determines
  ordering and dependency.
- **TIER** (enforcement strictness, unchanged from today):
  - `INV`  — sim-mode invariant; FAIL is always a hard FAIL.
  - `EXACT`— must match within tight tolerance; hard FAIL.
  - `DIST` — distributional match; FAIL unless `--lenient`.
  - `DIAG` — diagnostic only; never FAILs (informational), but feeds root-cause.

STAGE drives diagnosis; TIER drives the pass/fail verdict. They are independent.

### Dependency gating (the part that makes checks build on each other)

Each check declares its **upstream prerequisites** — the checks whose passing is
required for this check to be *meaningful*. The verdict engine then:

1. Walks rungs bottom-up.
2. Finds the lowest stage with an enforced FAIL whose upstreams all PASS →
   labels it **ROOT-CAUSE**.
3. Tags every higher enforced FAIL whose upstream chain contains a failed check
   as **DOWNSTREAM (of <root>)**, demoted from the headline failure list.

Result: one run prints "ROOT-CAUSE: stage-1 overhead residual (K3b); 7 downstream
failures suppressed" instead of nine equally-loud FAILs you have to triage by hand.

`deps` must name the *strongest causal link*, not a generic base. In particular
TC1 (coverage) is **not** a universal ancestor — it gates only K10, because a
missing field makes a downstream check SKIP (handled locally), not FAIL. Wiring
every check to depend on TC1 would wrongly demote independent failures (e.g. a
real trainer_speed gap) to "downstream" whenever any *unrelated* field is absent.

### Growth rule

Every time a parity bug is root-caused, leave behind the **most fine-grained
check that would have localized it to the responsible mechanism**, placed at its
causal stage with its upstream dependencies declared. Checks are append-only:
never delete one to "clean up." A check that is currently redundant becomes a
regression guard the next time the simulator changes.

When a single coarse check can be split into independent mechanisms, **split it**
— one assertion per mechanism. A blob KS over six timing phases tells you "timing
is off"; six per-phase KS checks tell you "the MQTT-fetch phase is off, the rest
match." Always prefer the latter.

---

## §2  The ladder

Stages run foundational → emergent. Within a stage, controls/mechanisms precede
the emergent rollup. `[NEW]` = to implement; everything else exists in checks.py.
"Isolates" = the one thing this check tells you when it fails *and its upstreams
pass*. "Dep" = upstream prerequisites.

### Stage 0 — Telemetry coverage  *(gate for everything)*
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| TC1 `[NEW]` | Field coverage matrix | CONTROL/INV | A field a downstream check reads is missing/sparse in one mode — explains every downstream SKIP at once | — |
| K10 | vclock_now present (sim) | CONTROL/INV | Sim path never stamps vclock (sync aggregator today) | TC1 |

> TC1 generalizes K10: for *each* field consumed downstream (vclock_now,
> staleness, trainer_speed_s, avail_composition, num_eligible, the phase fields,
> stat_utility, sim_send_ts), report presence count + density per mode. One table
> turns "9 mysterious SKIPs" into "these 3 fields are absent in sim."

### Stage 1 — Clock / time-base  *(the foundation; most parity bugs live here)*
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| K1 | vclock monotone (sim) | MECHANISM/INV | vclock goes backwards | K10 |
| K7 | sim_rate in [0.01,100] | MECHANISM/INV | vclock/wall absurd | K10 |
| P3 | trainer_speed_s distribution | CONTROL/DIST | The *input* to the clock model differs (speed model itself wrong) | — |
| K3a `[NEW]` | Modeled-compute advance | MECHANISM/EXACT | sim Δvclock vs the speed order-statistic the round-close formula *should* produce (K-th fastest in-flight for async; max-of-K for sync) — tests the advance **formula** with overhead excluded | P3,K1 |
| K3b `[NEW]` | Overhead residual | MECHANISM/EXACT | `real_advance − sim_advance` per round ≈ 0 — the missing per-commit MQTT/dispatch overhead (CRITICAL-1). Pass once `sim_commit_overhead_s` is modeled | K3a |
| K4 | Overlap factor | MECHANISM/DIAG | Sim doesn't model inter-round async pipelining | P3,K1 |
| K3 | Per-round advance distribution | EMERGENT/EXACT | Sum of K3a+K3b+K4 diverges (rollup) | K3a,K3b,K4 |
| K2 | Rounds-per-virtual-second | EMERGENT/EXACT | Throughput diverges (rollup) | K3 |

> The decomposition is the whole point. Today K3 (per_round_advance) lumps
> formula + overhead + overlap into one FAIL. Split it: if **P3 passes, K3a
> passes, K3b fails, K4 passes** → the bug is *pure missing overhead*, nothing
> else. That single sentence is what CRITICAL-1 took a paragraph of prose to say.
> K3b is also cross-validated at Stage 4 (mqtt_fetch phase): real overhead seen
> at the trainer level should equal K3b residual × agg_goal.

### Stage 2 — Availability  *(indexed by the clock — so gated on Stage 1)*
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| A1 | avail_composition parity | MECHANISM/DIST | Per-state available counts diverge | — |
| A2 | num_eligible / num_candidates | MECHANISM/DIST | Eligible-set size diverges | A1 |
| A3 `[NEW]` | Trace time-base consistency | CONTROL/DIST | Availability trace indexed by *different* clocks (sim=vclock, real=wall) — the REFL HIGH-1 bug. Compare each trainer's first/last-available time mapped through its mode's clock | K3 |
| A4 `[NEW]` | Per-trainer duty-cycle | MECHANISM/DIST | A trainer's on/off fraction differs even when set sizes match; needs avail_change events | A3 |

> A2's failure on REFL is *downstream* of the clock (sim runs at vclock_rate
> 0.274 → hits different trace windows). A3 makes that explicit: it fails only
> when the time-base mapping itself is wrong, so A2-fail + A3-pass = "fix the
> clock first," A2-fail + A3-fail = "fix the trace lookup."

### Stage 3 — Selection  *(given the eligible set)*
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| S3/4 | num_chosen / in_flight / effective_c | MECHANISM/DIST | Selector picks a different count | A2 |
| A2c `[Jun14c]` | selected-vs-pool speed bias | MECHANISM/DIST | Selector's revealed *speed preference* (`bias=selected−pool`) diverges with the pool matched → **selector-scoring** (oort), vs pool itself diverging → **composition** (A2b, refl) | A2b |
| Sx `[Jun14c]` | selector score-term localize | DIAG | *Which* utility-score term drives a mix split (oort believed_I/temporal/system_util; feddance V/I/A/U) — pinpoints e.g. oort `system_util` | A2b |
| Sd `[Jun16]` | preferred-duration penalty bind | MECHANISM/DIST | Oort speed-penalty **binding frequency** per round (≥1 selected w/ `system_util<1`) + reconstructed `pref` median — the D1 unsorted-`pref` guard (caught: real 80 % vs sim 46 %). Works on pre-instrumentation runs (reconstructs `pref=dur·√system_util`). | A2b |
| S2 | Participation frequency | EMERGENT/DIST | Per-trainer chosen-count diverges | S3/4 |
| S1 | Per-round Jaccard | DIAG | Exact set identity (gated WARN for stochastic selectors) | A2 |

### Stage 4 — Dispatch & training  *(per-trainer timing; the overhead source)*
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| T2 `[NEW]` | training_budget_s distribution | CONTROL/DIST | The *input* to the speed model differs | — |
| T_pre `[NEW]` | pre_train_s phase | MECHANISM/DIST | one phase | — |
| T_w2g `[NEW]` | weights_to_gpu_s phase | MECHANISM/DIST | one phase | — |
| T_gpu `[NEW]` | gpu_compute_s phase | MECHANISM/DIST | one phase | T2 |
| T_mqtt `[NEW]` | mqtt_fetch_s phase | MECHANISM/DIST | per-commit MQTT overhead at trainer level (cross-checks K3b) | — |
| T_w2r `[NEW]` | weights_to_ram_s phase | MECHANISM/DIST | one phase | — |
| T_post `[NEW]` | post_train_s phase | MECHANISM/DIST | one phase | — |
| T3 | GPU budget respected | MECHANISM/INV | real GPU time overruns modeled budget | T2 |
| K6 | sim_send_ts correctness | CONTROL/INV | sim dispatch timestamps not stamped/advancing | K10 |

> Today `trainer_phase` is one DIAG blob. Split into one DIST sub-check per phase
> so the report says exactly which phase diverges. Keep the blob's combined table
> in the report for at-a-glance reading, but each phase asserts independently.

### Stage 5 — Update return & ordering
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| U5 | Inter-arrival order (Spearman) | MECHANISM/DIST (gated WARN) | Arrival rank within a round diverges | K3,S3/4 |
| U4 | agg_goal_count cycles 1..K | MECHANISM/INV | Lost/double-counted update per round | — |

### Stage 6 — Aggregation
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| U3 | Staleness distribution | MECHANISM/DIST | Staleness diverges (async: directly downstream of clock under-charge) | K3,U5 |
| P1 | Aggregation sequence | EMERGENT/DIST (gated for stochastic) | Per-round contributing set diverges | S2,U5 |

### Stage 7 — Statistical utility
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| F1-3 | Per-trainer utility distributions | EMERGENT/DIST | Utility diverges (downstream of selection+training+staleness) | S2,T_gpu,U3 |

### Stage 8 — Emergent outcomes  *(the headline numbers; depend on ~everything)*
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| K8 | Terminal-state parity at matched V | EMERGENT/EXACT | rounds/trainers at matched virtual budget diverge | K2,S2 |
| U2 | Total commits at matched V | EMERGENT/EXACT | commit count at V diverges | K2,U4 |
| C1 | Accuracy curve by FL round | EMERGENT/DIST | accuracy diverges | F1-3,K8 |
| C2 `[NEW]` | Loss curve by FL round | EMERGENT/DIST | loss diverges (tracked separately from acc) | F1-3,K8 |

### Stage 9 — Budget / stop sanity  *(meta; orthogonal to causal chain)*
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| K9 | Stopped by budget, not rounds cap | INV (WARN) | Comparison truncated by `rounds` cap | — |
| K5 | Failsafe ceiling | INV | Sim wall overshoot > 20% of budget | — |

---

## §3  Mechanism reference — implemented sim fixes

The simulator does **real GPU compute** but stamps a *modeled* completion time
`sct` (it does not sleep the trainer's wall budget). Parity work is making the
sim's clock, ordering, and availability behave as the real pipeline would at that
`sct`. The validated mechanisms below are config-gated and guarded; the
**Overriding principle**: parity ≠ goal, a *correct* simulator is — real is the
reference only after `validate_real` shows it admissible (done: concurrency
28.8/c30, double-dispatch 0); never tune sim to a wrong real.

### Clock & ordering (felix async stack, validated)
- **Overhead → 0** (`simCommitOverheadSeconds=0`): the clock TRACKS completions
  (`vclock = max(vclock, sct)`) instead of being a pure overhead ramp.
- **Drain by physical READINESS, not predicted completion** (`_sim_recv_min`):
  admit any in-flight end whose message has physically arrived into the reorder
  buffer, so slow trainers buffer as futures and commit in `sct` order (staleness
  7.2 → 3.5). Probing the LIVE in-flight set (`recv_fifo` pops min-`sct`) reorders
  the past-dated tail.
- **`realDistributeSettleSeconds=0`**: removes a real-only 2×`sleep(0.1)`/commit so
  real holds ~c computing instead of being artificially slowed (advance 4.1,
  staleness 2.8).
- **`simRedispatchGapSeconds`** (post-commit re-dispatch leg, slot-held by cooling):
  spaces completions without counting toward the committed update's staleness.
  `0.6` zeroed the over-advance → throughput family green. *Residual:* a
  gap↔staleness coupling means one knob can't hit both advance and staleness; felix
  is HELD pending a buffer-aging investigation, not a scalar.

### §4.5  refl — `sct`-gated pool exclusion (`simInflightResidence`, validated)
A trainer that has physically sent but is modeled as still computing (`vclock < sct`)
must NOT re-enter the eligible pool — in real it is busy. In `oort/top_aggregator.
_distribute_weights`, the still-computing set (`_sim_buffer.pending_after(vclock)`)
is added to `trainer_unavail_list` (the *unavailable* path, NOT `selected_ends` —
which would re-dispatch and reset `sct`); released at `vclock ≥ sct` (budget ≤ ~56s).
Fixed refl's pool composition (A2b 12.4 → ~6.5 = real), flipping all emergent checks
green. Guard: `test_virtual_clock.py::test_pending_after_*`,
`test_sync_sim_ordering.py::TestSimInflightResidence`.

### §4.9  oort — `sct`-gated carry-over (`simInflightCarryover`, implemented, UNVALIDATED)
The sync-oort aggregator over-selects (×1.3) and closes a round at agg_goal=10,
leaving the ~3 slowest still computing. In **real** they stay in `selected_ends`
in-flight across rounds (`in_flight_after` 3.3); in **sim** the update arrives at
once, gets stale-rejected (prior `MODEL_VERSION`), and frees its slot → sim drains
to 0.15. **Distinct from §4.5**: §4.5 gates pool *re-entry*; §4.9 gates the
*cleanup/commit*. In `oort/top_aggregator._oort_sim_recv`: a prior-round straggler
(`_round − MODEL_VERSION > 0`) with `sct > vclock_round_start` is held (not yielded,
not clock-advanced), re-buffered so it stays in `selected_ends` (carried in-flight),
and commits a few rounds later once the clock passes its `sct`. Enabled for the oort
sim block. Guard: `test_sync_sim_ordering.py::TestSimInflightCarryover`.

### §5  Checker corrections (stochastic / observability classes)
Once the sim *dynamics* match, some residual FAILs were the checker enforcing exact
identity on quantities a stochastic / in-memory simulator cannot reproduce
(diagnostic tell: byte-identical across runs despite large dynamics changes). All
are principled, guarded, append-only — a future *deterministic* selector still gets
exact enforcement via `DETERMINISTIC_SELECTORS`:
- **P1 aggregation_sequence** → WARN for stochastic selectors (exact per-round set
  identity unattainable; S2 participation is the enforced invariant).
- **F1-3 utility** → enforce the POOLED KS (per-trainer KS=1.0 was mechanical for
  n≤2 samples; means were identical).
- **phase_mqtt_fetch** → DIAG (in-mem cache wall time, deliberately off the virtual
  clock).
- **trainer_speed / eligible_speed / selection_bias** → integer-grid / metadata-pool
  (see Status → checker corrections).

### Discrepancy ledger — flame vs reference Oort (per-baseline)
flame has ONE `OortSelector` inherited by both the `oort` baseline (should match
standalone Oort, `third_party/Oort`) and `refl` (should match the REFL fork,
`third_party/REFL`). The two references differ on defaults, so each baseline's
knobs are config-driven (`selector.kwargs`), defaulting to the Oort paper
(`scoring.OORT_PAPER_DEFAULTS`) with refl overriding to the fork.

| # | discrepancy | resolution |
|---|---|---|
| D1 | `pref` not sorted | FIXED (sort added) — was a port bug; validated on oort |
| D2 | stat-utility not normalized/clipped | FIXED (`scoring.oort_normalize_reward`, config `normalize_reward`/`clip_bound`) |
| D3 | `round_threshold` | config-driven: oort/felix=10 (paper), refl=30 (fork) |
| D4 | `cut_off_util` + cutoff-index | FIXED: config (0.7 paper / 0.05 refl); index now thresholds the exploit-boundary score (was inert) |
| D5 | temporal time-base | **DEFERRED** — flame uses `last_selected_round`; refs use round-last-UPDATED. Needs an aggregator-stamped prop; subtle, high blast radius. Flagged in code. |
| D6 | `clip_bound` | config-driven: 0.98 paper / 0.9 fork |
| S | refl exploitation | FIXED: was deterministic top-k; now the fork's cut_off_util-augmented utility-weighted `np.random.choice` |
