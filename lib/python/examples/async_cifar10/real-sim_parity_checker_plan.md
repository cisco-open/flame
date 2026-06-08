# Real ↔ Sim Parity Checker — Rigorous Fidelity Plan

**Status:** proposal
**Author:** parity-checker overhaul
**Motivating runs:**
- sim  = `experiments/run_20260606_172711_dbg_felix_n300_alpha0.1_syn0_stream_sim`
- real = `experiments/run_20260606_182638_dbg_felix_n300_alpha0.1_syn0_stream_real`

Both: Felix (`async_oort` + `fedbuff`), n=300, α=0.1, syn_0, agg_goal=10,
`max_runtime_s = sim_wall_ceiling_s = 10800s (3h)`, `min_trainers_to_start=290`.

---

## 0. TL;DR

The simulator and the real run are **not doing the same work**, and the current
parity scripts do not catch it. In the same 3-hour budget:

| | FL rounds | wall elapsed | final vclock | virtual-s **per round** |
|---|---:|---:|---:|---:|
| **real** | **673** | 10506 s (2.92 h) | n/a (= wall) | **15.6 s** |
| **sim**  | **410** | 3191 s (0.89 h) | 10806 s (≈3 h) | **26.4 s** |

The trainer **speed model is identical** — `max(trainer_speed_s)` per round is
~28 s in *both* runs. The divergence is purely in **how the simulator advances
its virtual clock per round**:

- **Sim** charges ≈ `max(trainer_speed_s)` of the committing cohort (~26–28 s)
  for every FL round, as if rounds were **serialized**.
- **Real** closes consecutive rounds ~15.6 s apart because asynchronous trainers
  **overlap**: while the 10 trainers that close round *N* finish, other trainers
  are already mid-training, so the *next* round's commits land far sooner than a
  full ~28 s training time later.

Net effect: sim **over-charges virtual time ~70% per round**, so it exhausts the
3 h virtual budget in 410 rounds while real fits 673. **Every downstream parity
metric computed over the overlapping rounds 1..410 can still "match"** — and they
largely do — which is exactly why the current checker reports green while the two
runs are doing materially different amounts of work.

This document (a) explains why the existing checks are blind to this,
(b) specifies an **exhaustive** check battery that only passes on a near-replicative
logical + throughput match, and (c) consolidates the eight scattered scripts into
**one holistic checker** with a single report, deprecating the rest.

> Scope note: the checker's job is to **detect and localize** fidelity gaps like
> the vclock-overcharge above, not to fix the simulator's clock model. The fix to
> the sim's per-round advance is tracked separately; this plan makes the gap
> impossible to miss.

---

## 1. Why the current checks miss it

Current assets (to be consolidated — see §4):

| file | role | blind spot |
|---|---|---|
| `scripts/parity_checks.py` | canonical pure checks, imported by pytest | compares only **overlapping rounds**; no throughput / rounds-per-virtual-second invariant |
| `scripts/compare_parity.py` | CLI report wrapping the above + 3 extra checks | same; "convergence" check has a copy-paste self-compare bug (line 295) |
| `scripts/sanity_check_real_sim.py` | T1–T7 single-run sanity | **T2 codifies the bug as correct** (expects vclock advance ≈ max(speed)); T4 "wall_speedup" looks great precisely *because* sim overcharges then stops early |
| `scripts/compare_clock_parity.py` | per-round virtual-time deviation | actually **would** surface this (it prints per-round vt deviation) but it is **not a gate**, not in `run_all_parity`, and prints WARN text only |
| `scripts/analyze_*.py` (5) | per-run diagnostic plots | single-run, not parity gates |

Three concrete failure modes:

1. **Overlapping-rounds-only comparison.** `selection`, `staleness`,
   `participation`, `utility`, `aggregation_sequence` are all computed on
   `set(real_rounds) & set(sim_rounds)`. Sim stops at 410; real reaches 673. The
   checker silently ignores rounds 411..673 and the throughput gap that produced
   them. There is **no check on total work done per unit virtual time**.

2. **The clock check encodes the bug as the spec.** `t2_vclock_accounting`
   asserts per-round vclock advance ≈ `max(trainer_speed_s)` and only *downgrades
   to WARN* for asyncfl. In this data sim advance (26.4 s) ≈ max speed (27.6 s) →
   T2 is *happy*. But real advance is 15.6 s. T2 is testing the wrong invariant:
   it should test that **sim's per-round advance matches real's per-round wall
   advance**, not that it matches the slowest trainer.

3. **`wall_speedup` is a vanity metric here.** T4 reports sim "faster than real"
   (good!) — but that speedup is partly an artifact: sim burns the virtual budget
   in fewer rounds, so it *finishes sooner*. Speed without fidelity is
   meaningless; there is no companion check that the **same number of rounds /
   commits** happened for the **same virtual budget**.

---

## 1B. Second pair: REFL (sync) — additional, *different* parity defects

Runs:
- sim  = `experiments/run_20260606_172726_dbg_refl_n300_alpha0.1_syn0_stream_sim`
- real = `experiments/run_20260606_180320_dbg_refl_n300_alpha0.1_syn0_stream_real`

REFL = `refl_oort` selector + `refl` optimizer, sync, `aggr_num=10`,
overcommit `num_chosen=13` (1.3×), `stale_update=5` (discard updates >5 rounds old),
deadline=100.

| | FL rounds | wall | final vclock | virtual-s/round | max(speed)/round | staleness>5 |
|---|---:|---:|---:|---:|---:|---:|
| refl **real** | **1000 (capped)** | 3036 s (0.84 h) | n/a (=wall) | 3.04 s | 11.1 s | 0 / 10000 |
| refl **sim**  | **1000 (capped)** | 1790 s (0.50 h) | **None** | n/a | 7.7 s | 0 / 10000 |

Four findings:

1. **Round-cap early termination (the user's fix).** Both refl runs stop at exactly
   **1000 rounds**, *before* the 3 h budget — so the comparison is truncated by the
   `rounds` cap, not the wall/vclock budget. **Fixed**: `rounds: 1000 → 20000` in
   `expt_scripts_2026/felix_oort_refl_feddance_alpha0.1_OVERNIGHT_node{1,2}.yaml`
   (canonical source, covers overnight + debug) plus an explicit `h["rounds"]=20000`
   guard in `scripts/debug_run.sh` (non-smoke branch, alongside the `max_runtime_s`
   override). New checker check **K9** below makes truncation-by-cap an explicit WARN
   so it can never silently distort a comparison again.

2. **The sync aggregator emits NO `vclock_now` at all** (neither sim nor real).
   Felix (async) agg_round events carry `vclock_now` + `sim_completion_ts_recv`; the
   REFL/sync path carries neither. Consequence: the entire §3.H clock/throughput
   battery (K1–K3, K7) **cannot run on refl sim** — the single most important fidelity
   dimension is unobservable for sync baselines. **Required fix**: stamp `vclock_now`
   on agg_round events in the syncfl sim path too (and `sim_completion_ts_recv` where
   applicable). Until then the checker must **FAIL-LOUD (not silently SKIP)** when a
   sim run lacks vclock telemetry — see check **K10**.

3. **`trainer_speed_s` divergence is larger for refl than felix.** refl sim speeds are
   quantized integers (`2,3,5,5,7,9,…`, mean max 7.7 s) vs real measured floats
   (mean max 11.1 s) — a ~30% under-model of the committed cohort's time. Felix
   matched closely (27.6 vs 28.3). So the sim's per-trainer time model is *baseline-
   sensitive*: it under-charges the fast cohort that REFL's overcommit-then-take-
   fastest-10 selection produces. Check **P3** (`trainer_speed_s` parity) will WARN/FAIL
   on refl and PASS on felix, localizing this.

4. **Selection structure IS in parity** (good news, both baselines): sim vs real
   `concurrency / effective_c / num_chosen / in_flight / num_eligible` all match —
   felix (conc 10, in_flight ~11 both), refl (chosen 13, in_flight ~65, eligible
   ~245 both). And **staleness discard works**: refl `max staleness = 5`, 0 updates
   `>5` in both modes. So the defects are isolated to (clock telemetry, per-trainer
   time model), *not* selection/eligibility/staleness logic.

### Why REFL proceeds faster than Felix (and why the premise was inverted)

Observed real throughput: **refl 3.04 s/round vs felix 15.6 s/round** (5× faster) —
the *opposite* of the expectation that async-felix (agg_goal 10, "concurrency 30")
should outrun sync-refl (agg_goal 10, overcommit 13). The telemetry shows why:

| | configured | **actual in_flight** | num_chosen/round | s/round |
|---|---|---:|---:|---:|
| felix (async) | `async_oort c=10` | **~11** (median 10) | ~0–1 incremental | 15.6 |
| refl (sync)   | `refl_oort aggr_num=10`, 1.3× | **~65** (median 65, max 80) | 13 | 3.04 |

- **Felix is NOT running at concurrency 30** — it runs at **c=10**, so it keeps only
  ~10 tasks outstanding. With agg_goal=10 and only ~10 in flight, a round can't close
  until essentially *all* in-flight finish → round ≈ slowest-of-10 (~28 s, 15.6 s with
  overlap). There is no spare concurrency to pipeline the next round.
- **Refl keeps ~65 in flight** (sync dispatches large cohorts + overcommit accumulate
  across rounds) and aggregates the **fastest 10 of a deep pool** → the 10th-fastest
  lands quickly → 3 s/round with heavy overlap.

So the throughput inversion is fully explained by **in-flight depth: ~10 (felix) vs
~65 (refl)**, and it reproduces identically in sim and real. **Actionable for the
experiment owner (not the checker):** if felix is intended to run at concurrency 30,
its selector `c`/concurrency must be raised from 10 → 30 — that deepens felix's
in-flight pool and is what would make felix faster than refl as expected. This is an
*experiment-config* finding, tracked as a next step (§7), separate from the sim↔real
fidelity work.

---

## 2. Design principles for the new checker

1. **One report, one exit code.** A single module + CLI emits a structured report
   (stdout table + `--json-out`) and exits non-zero on any enforced FAIL.
2. **Pure core, thin CLI.** All logic lives in importable, stdlib-only functions
   (preserve the existing `parity_checks.py` contract so both pytest suites keep
   working — see §4.3). The CLI only formats.
3. **Three tiers per check:**
   - **EXACT** — must match bit-for-bit (deterministic logical quantities).
   - **DISTRIBUTIONAL** — match within KS / mean tolerance (stochastic-but-stationary quantities).
   - **INVARIANT** — single-run sanity that must hold regardless of the other run.
   Every check declares its tier and its enforce/WARN gate explicitly.
4. **Compare the whole run, not the intersection.** Throughput and budget-accounting
   checks compare *terminal* state (total rounds, total commits, final vclock) and
   *rate* (rounds per virtual-second), not just the overlapping prefix.
5. **Localize, don't just fail.** Every FAIL prints the first diverging
   round/commit with a context window (extend `first_divergence`).
6. **Selector-aware gating.** Stochastic selectors are gated to participation
   *frequency* parity, not per-round set identity (already handled; keep it).

---

## 3. The exhaustive check battery

Organized along the FL lifecycle the user asked for: **availability → selection →
training → updates received & ordering → update processing → statistical utility →
aggregation → clock/throughput**. Each row: ID, tier, what it asserts, the
telemetry it reads, the gate.

### 3.A Availability / eligibility

| ID | tier | assertion | source | gate |
|---|---|---|---|---|
| A1 | DIST | per-round `avail_composition` (AVL_TRAIN/AVL_EVAL/UN_AVL/UNKNOWN) counts match | selection events | KS≤0.1 per class; mean diff ≤ 2 trainers |
| A2 | EXACT | `num_candidates` and `num_eligible` per round match | selection events | exact for matched rounds; **report coverage** (how many rounds matched) |
| A3 | INV | `min_trainers_to_start` join barrier honored: round 1 selection sees ≥ N eligible in both | selection events | both ≥ 290 |
| A4 | DIST | per-trainer availability duty-cycle (fraction of rounds a trainer is AVL_*) matches | selection events | max per-trainer abs diff ≤ 0.1 |

> A1–A4 close the gap that early sim runs selected from a half-filled pool. The
> barrier fix exists; these checks *prove* it stayed fixed and that the syn_0
> trace is replayed identically.

### 3.B Selection

| ID | tier | assertion | source | gate |
|---|---|---|---|---|
| S1 | EXACT* | per-round chosen **set** identical | selection (`chosen`) | EXACT for `DETERMINISTIC_SELECTORS`; **gated→WARN** for stochastic (join-order RNG) |
| S2 | DIST | per-trainer participation **frequency** parity (the enforced invariant for stochastic selectors) | agg_round `contributing_trainers` | avg abs diff ≤ 10, max ≤ 25 |
| S3 | EXACT | `num_chosen` per round matches; and `num_chosen/agg_goal ≤ 1.6` (both modes) | selection | exact match + ratio bound |
| S4 | DIST | `effective_c` / `concurrency` / `in_flight` per round match | selection | KS ≤ 0.2 |
| S5 | DIST | `exploration_factor` trajectory (Oort) matches | selection | mean abs diff ≤ 0.05 |

`*` S1 is enforced only for selectors registered in `DETERMINISTIC_SELECTORS`
(currently empty by design — see the docstring in `parity_checks.py`). Keep that
gating; surface Jaccard as signal.

### 3.C Training (per-trainer compute model)

| ID | tier | assertion | source | gate |
|---|---|---|---|---|
| T1 | DIST | per-trainer `real_gpu_time_s` distribution matches | trainer_round | KS ≤ 0.2 per trainer |
| T2 | DIST | per-trainer `training_budget_s` (D) matches | trainer_round | mean abs diff ≤ 0.5 s |
| T3 | INV | GPU budget respected: frac(rounds with gpu>budget) ≤ 0.25 (each mode) | trainer_round | mean overrun frac ≤ 0.25 |
| T4 | EXACT | `gpu_compute_s == real_gpu_time_s` (same measurement) | trainer_round | err ≤ 1e-6 |
| T5 | DIST | dataset size per trainer (drives D) matches → proves identical split replay | trainer_round / registry | exact per trainer |
| T6 | DIST | per-trainer **round_duration** (`sim_round_duration_s`) matches; expected mean diff ≈ gpu_time | trainer_round | KS ≤ 0.2 |

### 3.D Updates received & ordering  ← **the gap this overhaul is built around**

| ID | tier | assertion | source | gate |
|---|---|---|---|---|
| U1 | **EXACT** | **commit sequence parity**: the mode-agnostic logical sequence of committed updates (one entry per update in `(round, agg_goal_count)` order, keyed by `end`+`round`+`staleness`) agrees on the shared prefix; `first_divergence` localizes the first mismatch with context | agg_round | first divergence index == None on shared prefix |
| U2 | **EXACT** | **total commits parity for matched virtual budget**: # agg_round events up to virtual time V (= min of the two final vclocks) must match within tolerance | agg_round + vclock | abs diff ≤ 2% of commits |
| U3 | DIST | per-round **staleness** distribution matches; all staleness ≥ 0 (both) | agg_round | KS ≤ 0.2, mean diff ≤ 1.0, all non-negative |
| U4 | EXACT | `agg_goal_count` cycles 1..agg_goal cleanly within each round (no lost/double-counted update) | agg_round | no round exceeds agg_goal |
| U5 | DIST | inter-arrival ordering: the **rank order** in which trainers' updates arrive within a round matches (Spearman ρ on arrival index) | agg_round | mean ρ ≥ 0.7 |

### 3.E Update processing / aggregation

| ID | tier | assertion | source | gate |
|---|---|---|---|---|
| P1 | DIST | per-round set of `contributing_trainers` matches | agg_round | exact-set-match frac ≥ 0.7 (gated to WARN for stochastic, like S1) |
| P2 | DIST | aggregation **weights** / `agg_rate` per update match (fedbuff staleness-scaled) | agg_round | mean abs diff ≤ 1% |
| P3 | DIST | `trainer_speed_s` per round matches (the input to the clock model) | agg_round | KS ≤ 0.1  *(this is the control: it WILL pass — see §0 — proving the divergence is in the clock model, not the speed model)* |

### 3.F Statistical utility

| ID | tier | assertion | source | gate |
|---|---|---|---|---|
| F1 | DIST | per-trainer `stat_utility` distribution matches | agg_round | max KS ≤ 0.2 |
| F2 | DIST | per-trainer mean utility diff small | agg_round | avg ≤ tolerance |
| F3 | DIST | utility **disparity ratio** (max/min across trainers) matches | agg_round | abs diff ≤ 10% |

### 3.G Convergence

| ID | tier | assertion | source | gate |
|---|---|---|---|---|
| C1 | DIST | accuracy curve aligned **by FL round** (not by wall) matches | agg_eval | avg abs acc diff ≤ 0.05 |
| C2 | DIST | loss curve by round matches | agg_eval | avg abs diff ≤ tolerance |
| C3 | **FIX** | repair the self-compare bug: current `compare_parity.check_convergence` sets `sc = curve(real["agg_evals"])` then overwrites — confirm sim curve is actually used | agg_eval | n/a (correctness fix) |

### 3.H Clock & throughput  ← **the new enforced core**

| ID | tier | assertion | source | gate |
|---|---|---|---|---|
| K1 | INV | sim vclock monotone non-decreasing | agg_round (sim) | strictly non-decreasing |
| K2 | **EXACT** | **rounds-per-virtual-second parity**: `rounds / final_vclock` (sim) vs `rounds / wall_elapsed` (real) | agg_round | **abs rel diff ≤ 10%** — *this is the headline gate; it FAILs the motivating run (15.6 vs 26.4 s/round, 69% off)* |
| K3 | **EXACT** | **per-round virtual-advance distribution parity**: sim Δvclock/round vs real Δwall/round | agg_round | KS ≤ 0.2 AND mean diff ≤ 15% — *FAILs the motivating run* |
| K4 | **DIAGNOSTIC** | **overlap factor**: `mean(max(trainer_speed_s)) / mean(per-round advance)`. Real ≈ 28/15.6 ≈ 1.8 (healthy async overlap). Sim ≈ 28/26.4 ≈ 1.06 (no overlap). Report both; **FAIL if abs diff in overlap factor > 0.3** | agg_round | localizes the bug to "sim does not model inter-round overlap" |
| K5 | INV | failsafe sanity: sim wall must not overshoot `sim_wall_ceiling_s` by >20%; record whether `SIM_WALL_CEILING` / `WALL_CLOCK_FAILSAFE` fired | agg log | overshoot ≤ 20% |
| K6 | INV | sim mode: `task_recv.sim_send_ts` non-null & increasing after round 1; real mode: null | trainer task_recv | no violations |
| K7 | INV | sim_rate (`vclock/wall`) in sane range [0.01, 100] | agg_round (sim) | in range |
| K8 | DIST | **terminal-state parity**: at matched virtual budget V, both reached comparable FL-round count, total commits, total unique trainers used | agg_round | rounds within 10%, trainers within 5% |
| K9 | INV | **stopped-by-budget, not by cap**: neither run hit the `rounds` cap before `max_runtime_s` — else the comparison is truncated and downstream metrics are biased | agg_round + config | WARN if `max_round == rounds_cap` and wall/vclock < budget |
| K10 | INV | **vclock telemetry present**: a sim run's agg_round events must carry `vclock_now` (sync path currently omits it). FAIL-LOUD rather than silently SKIP K1–K3/K7 | agg_round (sim) | FAIL if sim run has zero `vclock_now` stamps |

> **K2/K3/K4 are the checks that would have caught the 410-vs-673 regression on
> day one.** They compare the *rate* and *shape* of virtual-time progress, not the
> per-round logical content. K4 in particular names the mechanism (missing overlap
> modeling) so the next engineer goes straight to the sim clock advance code.

### Overall verdict rule

- Any **EXACT** or **INVARIANT** FAIL → overall FAIL (exit 1).
- **DIST** FAIL → overall FAIL unless `--lenient`; WARN-gated checks never fail
  unless `--strict`.
- Report prints, per check: `tier`, `status`, the metric, the gate, and (on FAIL)
  the localized first-divergence context.

---

## 4. Consolidation: one holistic checker, deprecate the rest

### 4.1 Target layout

```
scripts/
  parity/
    __init__.py
    checks.py        # pure functions: loaders + all §3 checks (stdlib only)
    report.py        # formatting: stdout table + json + plots
    cli.py           # argparse entry; `python -m scripts.parity.cli ...`
  parity_check.py    # thin shim → scripts.parity.cli.main()  (stable CLI path)
```

`checks.py` **absorbs and supersedes**:
- `parity_checks.py` (keep every public name as a re-export for the tests — §4.3)
- `compare_parity.py` (the 9 CLI checks → §3 rows, convergence bug fixed)
- `compare_clock_parity.py` (per-round vt → K2/K3/K4)
- `sanity_check_real_sim.py` (T1–T7 → K1/K5/K6/K7 + T3/T4 invariants; **drop the
  bug-encoding T2**, replace with K3)

The five `analyze_*.py` plot scripts are **diagnostic, single-run, not gates**.
They stay for now but are invoked *through* the consolidated checker behind a
`--diagnostics` flag (so there's one entry point). Mark them deprecated-as-standalone
in their module docstrings; fold their plots into `report.py` opportunistically.

### 4.2 Single command, single output

```
python -m scripts.parity.cli \
    --real experiments/run_..._real \
    --sim  experiments/run_..._sim \
    --agg-goal 10 \
    --json-out parity.json --plot-out parity.png \
    [--strict] [--lenient] [--diagnostics]
```

- One stdout report grouped by §3 section (A–H), each line tier-tagged.
- One `parity.json` with every metric (machine-readable, for CI / dashboards).
- One `parity.png` multi-panel: virtual-time trajectory, per-round advance
  (real vs sim), overlap-factor bars, GPU-vs-budget, convergence-by-round.
- Exit code per §3 verdict rule.

`compare_overnight.sh` is rewritten to call this one command per (baseline) pair
and to drop its separate `compare_parity.py` invocation.

### 4.2b Multi-baseline invocation (test parity per selector as we converge)

A single batch entry point auto-discovers the latest sim/real run dir per baseline
and runs the checker for each, so we can re-verify parity for **every** selector as
the simulator gets closer to fidelity:

```
python -m scripts.parity.cli --batch \
    --experiments-dir experiments \
    --baselines felix oort refl feddance \
    [--agg-goal 10] [--strict] [--json-out parity_<baseline>.json]
# → one report per baseline + a roll-up table:
#   baseline | rounds(real/sim) | s/round(real/sim) | K2 | K3 | P3 | overall
```

This is the "way of invoking real-sim parity for different baselines" — the same
gates run for each, and the roll-up makes it obvious which selectors are in parity
(felix today: clock FAIL; refl today: K10 vclock-missing FAIL) and which converge as
fixes land. Baseline→(selector, sync/async, agg_goal, overcommit) metadata is read
from the run's `aggregator_config.json`/`execution_config.yaml`, so no per-baseline
code branches are needed in the checker.

### 4.2c Per-selector test layer

Beyond the cross-mode parity (which is selector-agnostic), add a **per-selector**
test module so each selector's *own* contract is pinned independently of real↔sim:

`tests/mode/test_selector_invariants.py`, parametrized over the registered selectors:

| selector | extra invariants to assert |
|---|---|
| `async_oort` (felix) | in_flight ≈ `c` (catches the c=10-vs-intended-30 drift); num_chosen incremental (0–1 typical); agg_goal cycle 1..N per round |
| `refl_oort` (refl) | num_chosen == round(1.3·aggr_num); **staleness ≤ `stale_update` (no committed update older than 5)**; blacklist honored for `blacklist_rounds`; pacer step/delta applied |
| `oort` (sync oort) | num_chosen == agg_goal; deterministic exploit set given seed |
| `feddance` | (define from selector kwargs) |

Each runs against a single run dir (sim or real) — they are *invariants*, not parity,
so they catch misconfiguration (e.g. felix's concurrency) and selector-logic
regressions that a sim↔real diff would miss when *both* sides share the same bug.

### 4.3 Preserve the pytest contract (do not break tests)

`tests/mode/test_parity_checks.py` and `tests/mode/test_real_sim_e2e_parity.py`
import `parity_checks` and call: `selection_parity`, `aggregation_sequence_parity`,
`staleness_parity`, `participation_parity`, `sim_send_ts_ok`, `gpu_budget_ok`,
`sim_commit_order_monotone`, `agg_goal_cycles_ok`, `commit_sequence`,
`first_divergence`, `run_all_parity`, plus `short/jaccard/ks_stat/mean_std`,
loaders, and `DETERMINISTIC_SELECTORS`.

Migration without breakage:
1. Move bodies into `scripts/parity/checks.py`.
2. Leave `scripts/parity_checks.py` as a re-export shim
   (`from scripts.parity.checks import *`) **or** keep `parity_checks.py` as the
   canonical module and have `parity/checks.py` import from it — pick one home,
   keep the import path the tests use alive.
3. **Extend `run_all_parity`** to include the new K2/K3/K4/U2/K8 throughput checks
   and add matching e2e test assertions (these are the assertions that would have
   failed CI on the motivating run).
4. Add unit tests for the new checks in `test_parity_checks.py` with tiny
   synthetic fixtures (one where rates match → pass, one with the 410-vs-673
   shape → fail).

### 4.4 Files to remove (after the shim + tests are green)

| remove | replaced by |
|---|---|
| `scripts/compare_parity.py` | `scripts/parity/` (CLI + checks) |
| `scripts/compare_clock_parity.py` | K1–K8 in `checks.py`, plot in `report.py` |
| `scripts/sanity_check_real_sim.py` | K/T invariants in `checks.py` (minus buggy T2) |
| `scripts/__pycache__/*` for the above | — |

`scripts/parity_checks.py` is **kept** (or converted to a shim) to preserve the
test import path. The `analyze_*.py` scripts are **kept** but demoted to
`--diagnostics` sub-invocations.

---

## 5. Implementation order

1. **Scaffold** `scripts/parity/` and move `parity_checks.py` bodies into
   `checks.py`; add the re-export shim; run both pytest suites → green
   (no behavior change yet).
2. **Add §3.H clock/throughput checks** (K2/K3/K4/K8, U2) to `checks.py` and
   `run_all_parity`; add unit + e2e tests; confirm the suite **FAILs** on the
   motivating sim run and **passes** on a synthetic matched pair.
3. **Fold in** the remaining §3 checks (A, S4/S5, P2/P3/U5, F3) from the old CLIs;
   add **K9** (cap-truncation) and **K10** (vclock-telemetry-present) — the latter is
   needed so refl/sync sim runs FAIL loudly instead of silently skipping the clock
   battery.
4. **Build `report.py` + `cli.py`** single output; fix the convergence
   self-compare bug (C3). Add `--batch` multi-baseline mode (§4.2b).
5. **Add per-selector test layer** (`tests/mode/test_selector_invariants.py`, §4.2c).
6. **Rewrite `compare_overnight.sh`** to the single command; demote `analyze_*`
   behind `--diagnostics`.
7. **Delete** the superseded files (§4.4); update any docs/HANDOFF references.
8. **Run** the new checker (`--batch`) on felix + refl pairs; paste the FAILing
   K2/K3/K4 (felix) and K10 (refl) output into the respective fix tickets as repros.

## 5B. Next steps (sim fidelity + experiment config — outside the checker)

These are surfaced by the checker but fixed elsewhere; the checker's job is to keep
catching them until they're resolved, then prove parity holds:

1. **Sim clock overlap model (felix).** The sim charges ≈ max committed
   `trainer_speed_s` per round (no inter-round overlap), inflating virtual-s/round
   26.4 vs real 15.6. Fix the per-round virtual-advance to account for the in-flight
   pipeline; re-run until K2/K3/K4 pass.
2. **Syncfl sim vclock telemetry (refl).** The sync aggregator path emits no
   `vclock_now` / `sim_completion_ts_recv`. Add the stamps so the clock battery is
   observable for sync baselines; K10 gates this.
3. **Sim per-trainer time model for fast cohorts (refl).** Integer-quantized,
   ~30% under real for refl's fastest-10 cohort. Tighten the speed model / quantization;
   P3 gates this.
4. **Felix concurrency config (DONE).** Felix was running at `c=10` (in_flight ~11),
   not the intended 30 (eurosys26 paper configs use 30). Set `async_oort c: 10 → 30`
   in all n300 α0.1 felix configs: `OVERNIGHT_node1` (both variants), `CONTROL_node1`,
   `SIMULATED_node1`, `STREAMING_node1`, `felix_n300_alpha0.1_syn0_STREAMING`. (n10/n48
   smoke/parity configs keep their scaled c=8/12.) The `test_selector_invariants`
   async_oort case (`in_flight ≈ c`) will assert this going forward. Expectation after
   re-run: felix in_flight ~30, deeper pipeline, faster rounds — felix should now
   out-pace refl as designed.
5. **Round cap → 20000 (DONE).** `rounds: 1000 → 20000` in the OVERNIGHT base yamls +
   `debug_run.sh` guard, so the 3 h budget binds instead of the round count.

## 6. Acceptance criteria

- Single command produces one stdout report + `parity.json` + `parity.png`.
- On the motivating pair: **K2, K3, K4, U2, K8 FAIL** and the report localizes the
  cause to "sim per-round virtual advance ≈ max(trainer_speed_s); no inter-round
  overlap modeled — real overlap factor 1.8 vs sim 1.06."
- P3 (`trainer_speed_s` parity) and T1/T2 (gpu time / budget) **PASS**, proving the
  trainer model is faithful and isolating the defect to the clock advance.
- Both existing pytest suites pass against the consolidated module.
- A matched, faithful real/sim pair passes **all** enforced checks.
- The four superseded scripts are removed; only `scripts/parity/` (+ kept
  `parity_checks.py` shim + demoted `analyze_*`) remain.
- `--batch` runs felix + refl (+ oort/feddance) and emits a roll-up; today it must
  report **felix: clock FAIL**, **refl: K10 vclock-missing FAIL** — and after the §5B
  fixes, both reach all-green.
- Per-selector invariants pass for each baseline's run (felix in_flight≈c, refl
  staleness≤5 + chosen==13, etc.).
