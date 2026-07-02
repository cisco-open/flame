# Plan: Felix Streaming‑Misprioritization Experiment (async_cifar10, n=50)

> **Status (updated 2026‑06‑11): IMPLEMENTED.** See the as‑built docs
> [EXPERIMENT_felix_streaming.md](EXPERIMENT_felix_streaming.md) and
> [IMPLEMENTATION_felix_streaming.md](IMPLEMENTATION_felix_streaming.md) — they
> supersede this plan. Two design changes vs the original plan below:
> 1. **Oracle is per‑baseline and online**, not a single global selector. Each
>    baseline `B` is paired with `B_oracle` (same selector) via an **aggregator‑side
>    true‑utility injector** (`aggregator/pytorch/oracle_utility.py` + a hook in the
>    sync/async/oort `_distribute_weights`) — every selector becomes oracular with no
>    selector changes; `B` vs `B_oracle` time‑to‑acc is that baseline's staleness tax.
>    (The global greedy `OracleSelector` remains as an optional, unused ceiling.)
> 2. **Phase 1 = uniform streaming only, parallelized across 4 nodes** (one
>    `B + B_oracle` pair per node); staggered streaming is phase 2. Pool all nodes'
>    runs on one box, then `run_felix_streaming.sh analyze`.
> The one piece not yet run end‑to‑end is the online injector — pilot it before the
> production sweep; the offline counterfactual is the validated fallback.

---

> Original plan (for reference):

## Context

We want to show, on the FLAME simulator, that **client utility is non‑stationary under
streaming data**, that **baselines prioritize on stale utilities and therefore
mis‑select**, and that **Felix's eval‑driven utility refresh tracks an oracle best** —
yielding the fastest time‑to‑accuracy among practical baselines. The async_cifar10 example
already has most of the machinery: deterministic prefix‑streaming, a trainer‑side
`util_counterfactual` (true utility under the current model), an **offline checkpoint‑replay
oracle** (`scripts/analysis/oracle_misselection.py`) that produces per‑round top‑k overlap /
mis‑selection / regret, and a rich telemetry + plotting pipeline
(`scripts/analysis/analyze_run.py`).

The deliverable is **two markdown files** plus the **code changes** that make the
experiment runnable:

1. **Experiment‑design doc** — `lib/python/examples/async_cifar10/docs/EXPERIMENT_felix_streaming.md`
   (claims, setup, justification, plot placeholders).
2. **Implementation‑plan doc** — `lib/python/examples/async_cifar10/docs/IMPLEMENTATION_felix_streaming.md`
   (thorough engineering plan: structure + hints, no exact code).

Confirmed decisions (from clarifying Q&A):
- **Streaming**: run **both** uniform (current single `full_after_s`) and **staggered**
  per‑client onset/rate as an ablation.
- **Oracle**: use the **offline replay** oracle for disparity metrics **and** add an
  **online `OracleSelector` run** (true top‑K each round) as the Claim‑3 performance ceiling.
- **Stopping**: stop after **20 consecutive evals ≥ 60%** test acc (reset on any dip), with a
  **max‑rounds / max‑runtime safety cap** so non‑converging arms still terminate.

Fixed knobs for all arms: CIFAR‑10, **n=50**, **Dirichlet α=0.1** (most heterogeneous),
**availability `syn_0`** (100%), **aggGoal=10** (sync ⇒ `aggr_num=10`; async felix ⇒
`aggGoal=10`), **`time_mode: simulated`** (virtual‑clock time‑to‑acc), checkpoints +
`util_counterfactual` ON.

---

## Deliverable 1 — Experiment‑design doc (content outline)

> File: `docs/EXPERIMENT_felix_streaming.md`. Written for a reader who will run it and read
> the figures. Each `![...]( )` is a placeholder to drop a generated PDF/PNG.

**1. Motivation / thesis.** Production FL clients *stream* data; systems score clients on a
*static, whole‑dataset* notion of utility computed the last time they trained. Two opposing
forces churn the true priority continuously: (i) new local data raises a client's loss →
utility ↑; (ii) the global model learning from a client lowers its loss → utility ↓. Stale
beliefs therefore decay, and selection mis‑prioritizes.

**2. Claims (verbatim mapping).**
- **Claim 1 — streaming makes utility dynamic; systems ignore it (qualitative + evidence).**
  Show a streaming client's true utility rise as data arrives and *dip after it participates*;
  show a low‑utility client whose utility keeps rising (data accruing) yet is rarely picked.
- **Claim 2 — stale prioritization degrades time‑to‑accuracy.** Build the oracle (true utility
  of all 50 clients each round → true top‑K). Show different clients are top‑20 in different
  ingest regions; measure how far each baseline's picks drift from the oracle over time (do any
  catch up?); correlate disparity with time‑to‑60%.
- **Claim 3 — Felix's eval selector does its job.** With the oracle as ceiling, Felix is (i)
  fastest‑to‑accuracy among practical baselines and (ii) closest to the oracle in selection.

**3. Experimental setup (table).** dataset / n=50 / α=0.1 / syn_0 / aggGoal=10 / streaming
{uniform, staggered} / target 60% / stop=20 consecutive evals / sim mode / felix `c=10`,
oracle `c=50`. **Arms**: `feddance`, `oort`, `refl`, `felix`, `oracle` × {uniform, staggered}
= 10 runs; offline oracle replay applied to the 4 practical baselines.

**4. Why this design is sound (justification — one paragraph each).**
- α=0.1 maximizes inter‑client utility divergence → prioritization matters most.
- `syn_0` (100% available) isolates the *prioritization* effect from availability churn; it
  also neutralizes REFL's availability machinery so the comparison is fair (all baselines reduce
  to their utility‑ranking core).
- n=50 is small enough to run the `c=50` eval‑all oracle cheaply and dodge MQTT‑scale issues,
  yet large enough that top‑10‑of‑50 selection is meaningful.
- aggGoal=10 fixed across baselines controls aggregation granularity / effective parallelism.
- Streaming makes utility non‑stationary, which is the precondition the thesis needs.
- "20 consecutive evals ≥ 60%" gives a stable time‑to‑acc (not a lucky spike); the cap
  guarantees termination.

**5. Plot placeholders.**
- Claim 1: `![Utility trajectory + participation dips]( )`, `![Rising‑but‑unpicked client]( )`,
  `![True‑utility heatmap (clients×time) with top‑K churn]( )`.
- Claim 2: `![Oracle top‑20 membership over ingest regions]( )`,
  `![Selection disparity vs oracle over rounds (overlap / mis‑selection / regret), per baseline]( )`,
  `![Mean disparity vs time‑to‑60% scatter]( )`.
- Claim 3: `![Accuracy vs sim‑time, all arms incl. oracle]( )`,
  `![Time‑to‑60% bars + gap to oracle]( )`, `![Overlap‑with‑oracle bars per baseline]( )`,
  `![Uniform vs staggered ablation]( )`.

**6. How to reproduce.** Commands to launch the multi‑arm YAML, run oracle replay, generate
figures (see Verification below), `dg_flame` conda env.

---

## Deliverable 2 — Implementation‑plan doc (mirrors the code‑changes section below)

> File: `docs/IMPLEMENTATION_felix_streaming.md`. Same content as **Code changes** below, written
> as a standalone engineering checklist with file/function anchors and acceptance criteria per
> item. No exact code — structure + hints.

---

## Code changes

### A. Target‑accuracy stopping rule (core new feature)
- **Config** — `lib/python/flame/config.py`: add hyperparameters
  `target_accuracy` (alias `targetAccuracy`, float, default `None` = disabled),
  `stable_evals_above_target` (alias, int, default 20). Reuse the existing `rounds` and
  `max_experiment_runtime_s` as the safety cap (already honored in sim via the virtual clock).
- **Aggregator** — the eval result funnels through `_eval_emit(round, loss, acc)` in
  `lib/python/flame/mode/horizontal/syncfl/top_aggregator.py`
  (shared by sync + async stacks). Add a `_check_target_stop(acc)` called from `_eval_emit`:
  maintain `self._consecutive_above_target` (init 0); on `acc >= target_accuracy` increment,
  else reset to 0; when it reaches `stable_evals_above_target` set `self._work_done = True`.
  The async loop already exits on `_work_done`
  (`asyncfl/top_aggregator.py:1160`).
- **Thread‑safety hint**: async `evaluate()` runs the eval in a daemon thread and only one eval
  is in flight at a time (`_eval_inflight` guard in `main_asyncfl_agg.py`),
  so the counter is only touched from one thread; setting the `_work_done` bool is GIL‑atomic.
- **Cadence hint**: keep `evalEveryNRounds` modest (e.g. 10) so 20 evals ≈ 200 rounds, not tens
  of thousands. Document the interaction in both docs.
- **Acceptance**: a low target (e.g. 0.2) fires the stop after exactly 20 consecutive evals and
  the run exits cleanly; a non‑converging arm hits the cap.

### B. Staggered per‑client streaming (ablation)
- **Trainer** — `trainer/pytorch/main.py`:
  extend the `data_streaming` config block (parsed ~L222) with an optional `stagger` sub‑dict:
  `enabled`, `onset_max_s`, `rate_jitter` (and optional `min_visible`). Derive **per‑client**
  `onset_s` and `span_s` deterministically from the existing `trainer_id` seed (already used at
  L454). Generalize `_visible_sample_count()` (`main.py:479`):
  replace `frac = sim_now / full_after_s` with
  `frac = clamp((sim_now - onset_s) / span_s, 0, 1)` (≤0 before onset → `min_visible`/0).
- **Oracle must mirror exactly** — `scripts/analysis/oracle_misselection.py`:
  `visible_count()` (L203) and `_dig_full_after_s` (L137) must reproduce the *same* per‑client
  onset/span from the *same* `sha256(task_id)` seed. Add a `_dig_stagger()` reading the stagger
  block from `execution_config.yaml` (already snapshotted + parsed by `resolve_config`, L89).
  This is the single most error‑prone coupling — call it out in the doc.
- **Telemetry hint**: the `util_disparity` event already carries `visible_samples/total_samples/
  elapsed_s`; ensure the per‑client `onset_s` is recoverable (either emit it or let the oracle
  read it from config) so `estimate_full_after_s` self‑calibration still works.
- **Acceptance**: two runs with identical seeds produce identical per‑round `visible_samples`;
  oracle‑reconstructed visible counts match trainer telemetry exactly.

### C. Online `OracleSelector` (Claim‑3 performance ceiling)
- **New selector** — `lib/python/flame/selector/oracle.py`, `OracleSelector` extending
  `AsyncOortSelector` (`async_oort.py`) to reuse its async/concurrency machinery. Override the
  ranking key: instead of stale `PROP_STAT_UTILITY`, rank candidates by a **fresh true utility**
  computed on the *current* model. Realize this by evaluating **all** clients each round
  (`c = n = 50`, eval‑all via `evalGoalFactor`) so every client reports its `util_counterfactual`
  true utility, surfaced to the selector as a new `PROP_TRUE_UTILITY`
  (`selector/properties.py`). Select the top `aggGoal=10` by that value for training.
- **Trainer/agg plumbing hint**: `util_counterfactual` already computes true utility under the
  current model; add a `MessageType` field to report it and have the async aggregator set
  `PROP_TRUE_UTILITY` (same place it sets `PROP_STAT_UTILITY`, ~L393 in
  `asyncfl/top_aggregator.py`).
- **Baseline entry** — add `oracle:` to `_metadata/baselines.yaml` (async stack,
  `selector.sort: oracle`, eval‑all settings).
- **Documented fallback**: if the dedicated selector is deferred, approximate the ceiling by
  running **Felix with `c=50` + full eval** (utilities ≈ fresh). Recommend the real selector for
  a clean *true* top‑K.
- **Acceptance**: `OracleSelector`'s chosen set has ≈1.0 top‑k overlap with the offline oracle's
  true top‑k.

### D. n=50 dataset split + experiment configs
- **Split**: generate `_metadata/dataset_splits/cifar10_alpha0.1_n50.yaml` (schema:
  `{dataset_name, dirichlet_alpha, num_trainers, trainer_data_splits: {trainer_NNN: [idx,...]}}`).
  Locate/reuse the Dirichlet partitioner (`scripts/extract_metadata.py`
  `_extract_dataset_splits`, L119, defines the schema); if no standalone generator exists, add a
  tiny `np.random.dirichlet`‑based helper. Note: small runs *can* fall back to the first 50
  trainers of the n300 split (oracle resolver L171‑172), but that gives n300‑sized partitions —
  prefer a dedicated n=50 split so each client holds ~1/50 of CIFAR‑10.
- **Experiment YAML(s)**: base on the multi‑arm overnight file
  `expt_scripts_2026/felix_oort_refl_feddance_alpha0.1_OVERNIGHT_node1.yaml`.
  Create `n50_alpha0.1_syn0_stream.yaml` with one experiment per arm × streaming condition. Set:
  `num_trainers: 50`, `dirichlet_alpha: 0.1`, `availability.mode: syn_0`, `time_mode: simulated`,
  `data_streaming.enabled: 'True'` (+ `stagger` block for the staggered variant),
  `util_counterfactual.enabled: 'True'`, `checkpoint.enabled: 'True'` + `every_n_rounds`
  (required for offline oracle), `aggGoal/aggr_num: 10`, `targetAccuracy: 0.60`,
  `stableEvalsAboveTarget: 20`, `evalEveryNRounds: 10`, `rounds`/`max_experiment_runtime_s` cap. felix
  `selector.kwargs.c: 10`; oracle arm `c: 50`.

### E. Analysis + claim figures
- **Per‑run oracle**: run `oracle_misselection.py --run-dir <run>` for each of the 4 practical
  baselines → `oracle_utility.csv`, `oracle_misselection.csv`, `oracle_counterfactual.csv`.
- **New figure driver** — `scripts/analysis/felix_streaming_figures.py`: consume all arms'
  telemetry + oracle CSVs and emit the claim figures into a shared `figures/` dir. Reuse
  `plot_helpers.py` (`binned_line`, `cdf_multi`, `scatter_diag`). Figures:
  - Claim 1: per‑client true‑utility trajectory with participation markers + data‑unlock overlay
    (from `oracle_utility.csv` + `agg_round` participation); true‑utility heatmap + top‑K‑churn.
  - Claim 2: oracle top‑20 membership stream graph; disparity (overlap / mis‑selection / regret)
    vs round per baseline; mean‑disparity‑vs‑time‑to‑60% scatter.
  - Claim 3: accuracy‑vs‑sim‑time multi‑arm; time‑to‑60% bars; overlap‑with‑oracle bars;
    uniform‑vs‑staggered ablation.
- **Time‑to‑acc extraction hint**: parse the `[ASYNC_EVAL]` / agg‑eval telemetry for the
  first/stable crossing of 60% in `vclock` time.

### F. Run orchestration
- A driver shell script (pattern: `scripts/overnight_run.sh`) to launch all arms × conditions
  in sim mode, then run oracle replay + figure generation, under the `dg_flame` env.

---

## Verification (end‑to‑end)

1. **Stopping smoke**: n=10, `targetAccuracy: 0.2`, `evalEveryNRounds: 5` — confirm the run
   stops after 20 consecutive evals ≥ 0.2 and exits; confirm a high target hits the cap.
2. **Streaming determinism**: two identical‑seed staggered runs → identical per‑round
   `visible_samples`; `oracle_misselection.py` reconstructed visible counts match telemetry.
3. **Oracle correctness**: `OracleSelector` chosen set vs offline‑oracle true top‑k overlap ≈ 1.0.
4. **Checkpoints/CSVs present**: each baseline run has `checkpoints/round_*.pt`; oracle replay
   writes the three CSVs.
5. **Full sweep**: launch `n50_alpha0.1_syn0_stream.yaml` (10 runs); generate figures; sanity‑check
   the thesis — Felix time‑to‑60% < other baselines and > oracle; Felix overlap‑with‑oracle is
   highest among the four; disparity correlates with time‑to‑60%.
6. Each figure script run is idempotent and writes into `figures/`; spot‑check one figure per claim.

---

## Key code anchors (for fast resume)

| Concern | Location |
| --- | --- |
| Selectors | `lib/python/flame/selector/{async_oort,oort,refl_oort,feddance}.py` |
| Selector scoring | `lib/python/flame/selector/scoring.py`, `properties.py` |
| Async aggregator (stack) | `lib/python/examples/async_cifar10/aggregator/pytorch/main_asyncfl_agg.py` |
| Stop hook (shared) | `lib/python/flame/mode/horizontal/syncfl/top_aggregator.py` `_eval_emit`; work loop `asyncfl/top_aggregator.py:1160` |
| Streaming | `trainer/pytorch/main.py:479` `_visible_sample_count`, `:489` `_rebuild_stream_loader` |
| Offline oracle | `scripts/analysis/oracle_misselection.py` (`visible_count` L203, `resolve_config` L89) |
| Plot helpers | `scripts/analysis/plot_helpers.py`; `scripts/analysis/analyze_run.py` |
| Baselines catalog | `lib/python/examples/_metadata/baselines.yaml` |
| Example multi‑arm YAML | `expt_scripts_2026/felix_oort_refl_feddance_alpha0.1_OVERNIGHT_node1.yaml` |
| Dataset splits | `lib/python/examples/_metadata/dataset_splits/cifar10_alpha*_n300.yaml` (need n50) |
