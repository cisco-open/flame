# Implementation — Felix Streaming Misprioritization Experiment

Engineering record for the experiment in [EXPERIMENT_felix_streaming.md](EXPERIMENT_felix_streaming.md).
Status: **implemented on branch `dg/fix_sim_fidelity`** (this doc doubles as the
change map + acceptance criteria). Paths are repo-relative.

---

## ▶ RESUME HERE (checkpoint — 2026‑06‑11)

**Where we are:** all code, configs, and docs are written and **statically
validated** (every touched `.py` compiles; the 4 per‑node YAMLs parse with
injection on the `*_oracle` arm only; the stagger formula is byte‑identical across
trainer / offline oracle / online injector). **Nothing has been run** — no smoke
test, no sweep. Nothing committed (branch `dg/fix_sim_fidelity`). Env: `dg_flame`
conda. All commands below run from `lib/python/examples/async_cifar10/`.

**Design recap:** 8 arms = `{felix, oort, refl, feddance} × {B, B_oracle}`, uniform
streaming, n=50, α=0.1, syn_0, aggGoal=10, sim mode, stop at 20 consecutive evals
≥ 60%. `B_oracle` = same selector as `B` + aggregator‑side true‑utility injection
(no selector changes). Split across 4 nodes, one `B + B_oracle` pair each.

**Do these in order:**

1. **Smoke test (one GPU box) — validate the online injector end‑to‑end.** This is
   the only piece not yet run; do it first.
   ```bash
   scripts/run_felix_streaming.sh smoke
   ```
   Pass criteria: aggregator log shows `[ORACLE_INJECT] ... set 50/50 utilities`
   each round on `*_oracle` arms; the target‑stop fires (`[TARGET_STOP]`);
   checkpoints land in each `experiments/run_*smoke*/checkpoints/`; the trailing
   `oracle_misselection.py` + `felix_streaming_figures.py` finish without error.
   If the injector misbehaves, the **offline counterfactual is the validated
   fallback** (run the baselines without injection; per‑baseline disparity still
   comes from `oracle_misselection.py`).

2. **Calibrate the streaming horizon** from the smoke/pilot. In
   `scripts/gen_n50_experiment.py` set `HORIZON_S` (and caps `ROUNDS_CAP`,
   `MAX_RUNTIME_S`) so data is still unlocking through most of training (check the
   per‑round `vclock_now` span vs `full_after_s`); re‑run the generator.

3. **Production sweep — on each node i (1..4), with that node's GPU count:**
   ```bash
   FELIX_NUM_GPUS=<gpus on this node> scripts/run_felix_streaming.sh node <i>
   ```
   node1=felix, node2=feddance, node3=oort, node4=refl (each runs `B` then
   `B_oracle`). Writes `experiments/run_*` (oracle dirs suffixed `_node<i>`).

4. **Pool + analyze on ONE box** once all nodes finish:
   ```bash
   # rsync every node's experiments/run_*_n50_alpha0.1_* into one experiments/
   scripts/run_felix_streaming.sh analyze     # oracle replay (per baseline) + figures
   ```
   Figures → `experiments/figures/`; drop them into the placeholders in
   [EXPERIMENT_felix_streaming.md](EXPERIMENT_felix_streaming.md).

5. **Phase 2 (later): staggered streaming.** Set `STAGGER_CONDITIONS = [False, True]`
   in `scripts/gen_n50_experiment.py`, regenerate, re‑run.

**Open follow‑ups** (non‑blocking): 4 named figures still stubbed (see §E); the
global `OracleSelector`/`baselines.yaml oracle` entry is unused (optional ceiling,
safe to delete); per‑round central eval cost is `n` forward passes (lower
`sample_size` if it dominates). Full detail: §C (injector), Verification checklist,
Follow‑ups at the bottom.

---

## Overview of changes

| # | Concern | Files |
| --- | --- | --- |
| A | Target-accuracy stop rule | `lib/python/flame/config.py`, `lib/python/flame/mode/horizontal/syncfl/top_aggregator.py` |
| B | Staggered per-client streaming | `lib/python/examples/async_cifar10/trainer/pytorch/main.py`, `scripts/analysis/oracle_misselection.py` |
| C | **Per-baseline online oracle** (aggregator-side true-utility injection) | `aggregator/pytorch/oracle_utility.py` (new); hooks in `syncfl/`, `asyncfl/`, `oort/top_aggregator.py`; mixin in the 3 example aggregators (`main_asyncfl_agg.py`, `main_oort_sync_agg.py`, `main_fedavg_agg.py`) |
| C′ | (optional, unused by default) global greedy `OracleSelector` | `lib/python/flame/selector/oracle.py`, `selectors.py`, `config.py`, `baselines.yaml` |
| D | n=50 split + per-node YAMLs | `scripts/gen_dirichlet_split.py`, `scripts/gen_n50_experiment.py`, `_metadata/dataset_splits/cifar10_alpha0.1_n50.yaml`, `expt_scripts_2026/n50_alpha0.1_syn0_stream_unif_{sim,node1..4}.yaml` |
| E | Cross-arm figures | `scripts/analysis/felix_streaming_figures.py` |
| F | Orchestration (4-node) | `lib/python/examples/async_cifar10/scripts/run_felix_streaming.sh` |

---

## A. Target-accuracy stop rule

**Config** (`config.py`, `Hyperparameters`): added
`target_accuracy` (`targetAccuracy`, float, default `None` = disabled) and
`stable_evals_above_target` (`stableEvalsAboveTarget`, int, default 20). The
existing `rounds` and `max_experiment_runtime_s` remain the safety cap (the latter is already
honored against the **virtual** clock in sim mode at `increment_round`).

**Aggregator** (`syncfl/top_aggregator.py`, shared by sync + async stacks):
- `__init__` initializes `_target_accuracy`, `_stable_evals_above_target`,
  `_consecutive_above_target = 0`.
- `_eval_emit(round, loss, acc)` now calls **`_check_target_stop(round, acc)`**:
  increments the consecutive counter when `acc >= target`, resets to 0 otherwise,
  and sets `self._work_done = True` once the counter reaches the threshold. The
  async work loop already exits on `_work_done` (`asyncfl/top_aggregator.py:1210`).

**Why it's thread-safe**: the async stack runs `evaluate()` in a daemon thread but
guards with `_eval_inflight` so only one eval is ever in flight; the counter is
touched by a single thread at a time and the `_work_done` bool store is GIL-atomic.

**Cadence note**: keep `evalEveryNRounds` modest (10 here) so 20 evals ≈ 200
rounds rather than tens of thousands.

**Acceptance**: smoke run with `targetAccuracy=0.2, stableEvalsAboveTarget=2`
terminates shortly after two consecutive evals ≥ 0.2; a high target falls through
to the `rounds`/`max_experiment_runtime_s` cap.

## B. Staggered per-client streaming

**Trainer** (`trainer/pytorch/main.py`):
- Module-level **`_stagger_params(trainer_id, onset_max_s, base_span_s, rate_jitter)`**
  derives a per-client `(onset_s, span_s)` deterministically from
  `sha256(f"{trainer_id}:stagger")` (two disjoint 32-bit slices → `u1, u2`):
  `onset = onset_max_s·u1`, `span = base_span·(1 + rate_jitter·(2u2−1))`, floored
  at `base_span/4`.
- The `data_streaming` config block now parses an optional `stagger` sub-dict
  (`enabled`, `onset_max_s`, `rate_jitter`, `min_visible`); `load_data` sets
  `self._stream_onset_s/_stream_span_s` when enabled (uniform = onset 0,
  span `full_after_s`).
- **`_visible_sample_count`** generalized to
  `frac = clamp((sim_now − onset)/span, 0, 1)`, floored at `min_visible`.

**Oracle mirror** (`scripts/analysis/oracle_misselection.py`) — *the critical
coupling*: a byte-for-byte copy of **`stagger_params`**, a generalized
**`visible_count(sim_now, onset_s, span_s, total, min_visible)`**, a
**`_dig_stagger`** that reads the `stagger` block out of `execution_config.yaml`
(via `resolve_config`), and per-trainer `onset_s/span_s` computed once before the
checkpoint loop. If you change the schedule formula in the trainer, change it here
too or the reconstructed visible prefixes drift.

**Acceptance**: two identical-seed staggered runs produce identical per-round
`visible_samples`; the oracle's reconstructed visible counts match the trainer's
`util_disparity` telemetry.

## C. Per-baseline online oracle (aggregator-side true-utility injection)

**Mechanism (one hook, no selector changes).** Each baseline `B` is paired with
`B_oracle` = the *same* selector, run with `oracle_utility_injection.enabled=True`.
Each round, *before* selection, the aggregator overwrites every candidate's
`PROP_STAT_UTILITY` (and, for feddance, `PROP_LOCAL_ACCURACY`) with the **true**
current value. Because OORT/REFL/FedDance/Felix all rank on that property, they all
become oracular with zero selector edits, and `B_oracle` is a genuine alternative
trajectory whose `time-to-60%` gap to `B` is B's staleness tax.

**Provider** (`aggregator/pytorch/oracle_utility.py`): `OracleUtilityProvider`
lazily builds `task_id → {arrival_global_idx, total, onset_s, span_s}` from
`trainer_registry.yaml` + the Dirichlet split (resolved via
`FLAME_TELEMETRY_DIR → run_dir/snapshot.yaml → metadata_location`), loads the CIFAR
pool once, then each round computes `I_m = N·√(mean(loss²))` on each candidate's
currently-unlocked prefix under `agg.model` at `agg._vclock.now`. Formulas
(`_stagger_params`, `_visible_count`, oort utility) are byte-identical to
`oracle_misselection.py`/trainer `main.py`. `OracleInjectMixin` exposes
`_init_oracle_util(data_root)` + the `_inject_oracle_utilities` override. Injection
is wrapped in try/except — it must never break training.

**Wiring**:
- Framework base `_inject_oracle_utilities(channel, task)` = no-op in
  `syncfl/top_aggregator.py`; **called before selection** in `_distribute_weights`
  of all three stacks: `syncfl/` (fedavg/feddance), `asyncfl/` (felix), and
  `oort/top_aggregator.py` (oort/refl, which has its own `_distribute_weights`).
- The 3 example aggregators add `OracleInjectMixin` as the first base and call
  `self._init_oracle_util(<example>/data)` in `initialize()`.
- Config: `oracle_utility_injection` (`enabled`, `alpha`, `num_trainers`,
  `sample_size`, `inject_accuracy`) + `data_streaming` ride on the **aggregator**
  hyperparameters via pydantic `Extra.allow` (no `config.py` change).

**C′ (optional, not used by default)**: `OracleSelector` (`flame/selector/oracle.py`,
`SelectorType.ORACLE`, `oracle` baseline) — a global greedy true-top-K selector,
kept as an absolute ceiling but not part of the per-baseline node runs.

**Acceptance** (needs a pilot run): each `B_oracle`'s offline mis-selection ≈ 0
(it already selects the true top-K); `[ORACLE_INJECT]` logs show `set 50/50`
utilities each round; `B_oracle` reaches 60% no slower than `B`.

## D. n=50 split + experiment YAML

**Split generator** (`scripts/gen_dirichlet_split.py`): per-class Dirichlet(α) over
N trainers, deterministic in `--seed`, writes the launcher schema
(`dataset_name, dirichlet_alpha, num_trainers, total_samples, trainer_data_splits:
{trainer_NNN: [...]}`) that `spawner.MetadataLoader.get_dataset_split` expects.
Generated `cifar10_alpha0.1_n50.yaml` (50000 samples; min 28 / max 3310 / mean
1000 — strongly non-IID). `trainer_registry.yaml` already covers `trainer_001..300`.

**Experiment generator** (`scripts/gen_n50_experiment.py`): emits a combined
`..._unif_sim.yaml` (all 8 arms) **and 4 per-node files** `..._unif_node{1..4}.yaml`,
each = `{B, B_oracle}` for that node's baseline (node1 felix, node2 feddance,
node3 oort, node4 refl). `NUM_GPUS` env sets the per-node GPU count. 8 arms =
`{felix, oort, refl, feddance} × {B, B_oracle}` (uniform; phase 2 adds staggered).
`B_oracle` = same selector as B + the `oracle_utility_injection`/`data_streaming`
blocks on the aggregator config. Each arm: `num_trainers=50`,
`dirichlet_alpha=0.1`, `availability.mode=syn_0`, `time_mode=simulated`,
`data_streaming` (uniform `full_after_s=10800`; staggered adds the `stagger`
block), `util_counterfactual` on, `checkpoint` on (offline oracle needs it),
`aggGoal/aggr_num=10`, `targetAccuracy=0.60`, `stableEvalsAboveTarget=20`,
`evalEveryNRounds=10`, `rounds=20000`/`max_experiment_runtime_s=12600` cap. felix `c=10`;
`B_oracle` keeps B's own selector kwargs. Structure mirrors the working
`felix_oort_refl_feddance_alpha0.1.yaml`.

**Tuning note**: `full_after_s`, `onset_max_s`, and the caps are first-pass values.
Calibrate against the pilot's per-round virtual-clock span so data is still
arriving through most of training (otherwise the dynamic-utility regime ends before
60% is reached and the effect washes out).

## E. Cross-arm figures

`scripts/analysis/felix_streaming_figures.py` discovers `run_*` dirs under an
experiments root, maps each to `baseline/variant/condition` (variant ∈ {base,
oracle}), and emits the claim figures into `experiments/figures/` (reusing
`plot_helpers`):
- **Claim 3** — accuracy vs sim-time with each baseline next to its oracle (`B*`),
  time-to-target bars, and a **per-baseline staleness-tax bar** = `T(base) −
  T(oracle)`. Sim-time x-axis joins `agg_eval` (acc by round) to
  `agg_round.vclock_now` (round → virtual time).
- **Claim 2** — mis-selection rate and utility regret vs round per baseline (from
  `oracle_misselection.csv`); mean-disparity vs time-to-target scatter.
- **Claim 1** — true-utility trajectories for the highest-variance clients and a
  clients×round true-utility heatmap (from `oracle_utility.csv`).

`time_to_target(rounds, accs, vclock, target, stable_k)` returns the first
(round, sim-time) of a `stable_k`-long run ≥ target (k=1 for first-crossing bars).
The figure functions degrade to `no_data_plot` when inputs are missing, so the
driver runs before all arms finish. *Still TODO* (placeholders referenced in the
design doc but not yet emitted): `claim1_rising_unpicked`,
`claim2_oracle_topk_membership`, `claim3_overlap_with_oracle`,
`claim3_uniform_vs_staggered` — add as follow-up plot functions.

## F. Orchestration (4 nodes)

`scripts/run_felix_streaming.sh {smoke | node <1-4> | run | analyze}` — robust
`dg_flame` conda activation, regenerates split + YAMLs (`NUM_GPUS` from
`FELIX_NUM_GPUS`):
- `node <i>` — launch that node's `{B + B_oracle}` YAML (the production path).
- `analyze` — pool step: `oracle_misselection.py` on each baseline run dir (skips
  `*_oracle_*`), then `felix_streaming_figures.py`. Run on one box after rsync-ing
  all nodes' `experiments/run_*` together.
- `smoke` — all 8 arms on one node, 6 rounds / target 0.2 / 2-eval stop.
- `run` — single-node fallback (all 8 arms here).

## Verification checklist

1. **Imports/parse** — all edited Python compiles; provider imports; mixin override
   resolves; stagger formula parity (trainer = oracle script = provider). ✅ done.
2. **Stop smoke** — `run_felix_streaming.sh smoke` terminates via the target stop,
   writes checkpoints, runs oracle replay + figures without error. ⏳ needs cluster.
3. **Injection pilot** — on a smoke run, `[ORACLE_INJECT]` logs `set 50/50` each
   round and the run doesn't error. ⏳ needs cluster.
4. **Streaming determinism** (phase 2) — identical seeds → identical per-round
   `visible_samples`; oracle reconstruction matches.
5. **Oracle correctness** — each `B_oracle`'s offline mis-selection ≈ 0.
6. **Full sweep** — per-baseline staleness tax > 0; Felix's tax smallest; Felix
   fastest among practical baselines; disparity correlates with time-to-60%.

## Follow-ups / known gaps

- **Pilot the online injector** (items 2–3) before the production sweep — it's the
  one piece not yet run end-to-end. The offline counterfactual is the validated
  fallback if the injector needs debugging.
- Calibrate streaming horizon + caps from the pilot (Section D note).
- Emit the four remaining named figures (Section E): `claim1_rising_unpicked`,
  `claim2_oracle_topk_membership`, `claim3_overlap_with_oracle`,
  `claim3_uniform_vs_staggered`.
- Per-round central eval cost: the oracle aggregator does `n` forward passes/round;
  if it dominates, raise `oracle_utility_injection.every_n`-style gating (not yet
  implemented) or lower `sample_size`.
- Phase 2: staggered streaming (`STAGGER_CONDITIONS=[False,True]`).
