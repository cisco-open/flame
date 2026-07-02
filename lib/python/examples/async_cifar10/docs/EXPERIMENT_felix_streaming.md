> ⚠️ **STATUS: paused / not validated against the current code.** This streaming-
> misprioritization experiment (n=50, the per-baseline online **oracle** selector
> `flame/selector/oracle.py`, the aggregator-side `oracle_utility.py` injection, the
> `gen_*`/`run_felix_streaming.sh` tooling, and `scripts/analysis/felix_streaming_figures.py`)
> was developed for a while and then set aside; the **same branch** was subsequently
> used for the real/sim fidelity work (see `PARITY.md`). It is shipped here for
> continuity but has **not** been re-run against the fidelity-era changes, so the
> oracle/n50 path may need updating or fixing before use. The fidelity work does not
> depend on it (the oracle injection is a guarded no-op unless explicitly enabled).

# Felix Update 11-Jun-26 — Streaming Misprioritization Experiment

> Does stale, whole-dataset client prioritization cost you time-to-accuracy under
> streaming data, and does Felix's eval-driven utility refresh recover it?
>
> Setup at a glance: **CIFAR-10 · n=50 · Dirichlet α=0.1 (most heterogeneous) ·
> `syn_0` 100% availability · aggGoal=10 · simulator (virtual clock) ·
> **uniform streaming (phase 1; staggered = phase 2)** · stop at 20 consecutive
> evals ≥ 60% test accuracy.**
>
> Arms (phase 1): each baseline `B ∈ {felix, oort, refl, feddance}` paired with its
> **per-baseline online oracle `B_oracle`** (same selector as B, fed *true* utilities
> each round) = **8 runs**, split across **4 nodes** (one `B + B_oracle` pair per node).
> Run per node with `FELIX_NUM_GPUS=<n> scripts/run_felix_streaming.sh node <1-4>`,
> then pool and `scripts/run_felix_streaming.sh analyze`.

---

## 1. Motivation / thesis

Production FL clients **stream** data: their local datasets grow over time. Yet
selection systems score each client by a **static, whole-dataset** notion of
utility, computed the last time that client trained, on the data it had unlocked
back then. Two opposing forces churn the *true* priority continuously:

1. **New local data raises a client's loss → utility ↑.** Fresh, unseen samples
   are exactly what the global model has not yet fit.
2. **The global model learning from a client lowers that client's loss →
   utility ↓.** Right after a client participates, the model has absorbed its
   data, so its marginal value drops.

So a client's utility is a moving target — it rises while data accrues and dips
each time the client is used. A selector that ranks on stale beliefs therefore
**misprioritizes**: it keeps picking clients whose value it already harvested,
and ignores clients quietly accumulating high-value data. The cost is paid in
**time-to-accuracy**.

## 2. Claims

### Claim 1 — streaming makes utility dynamic; systems don't model it
*Argued qualitatively, backed with measured evidence.* Client utility over the
entire dataset is not close to reality. We show, from the oracle's true per-round
utility (`oracle_utility.csv`):
- a streaming client's true utility **rising as data arrives** and **dipping right
  after it participates**; and
- a client with **low** utility whose value **keeps rising** (data accruing) yet
  is **rarely picked** — the priority inversion the systems miss.

**Figures**
- ![Per-client true-utility trajectories with participation dips](figures/claim1_utility_trajectories.pdf)
- ![Rising-but-unpicked client vs the dynamic top-K threshold](figures/claim1_rising_unpicked.pdf)
- ![True-utility heatmap (clients × round) — top-K set churns](figures/claim1_utility_heatmap.pdf)

### Claim 2 — stale prioritization degrades time-to-accuracy
We build the **oracle**: every client's *true current* utility under the *current*
model each round → the true top-K. Two complementary realizations (Section 4):
an **offline checkpoint replay** for exact disparity metrics, and an **online
oracle run** that always selects the true top-K (the performance ceiling).

We then show:
- different clients are **top-20 in different regions of data ingest** (the oracle
  membership is non-stationary);
- how far each baseline's picks **drift from the oracle over the run** — and
  whether any baseline **catches up** (it should not while data keeps streaming);
- that this disparity **correlates with time-to-60%**: more misselection → slower.

Disparity is measured as **mis-selection rate** = 1 − top-K overlap with the
oracle, and **utility regret** = (oracle top-K true utility − selected true
utility), both per round (`oracle_misselection.csv`).

**Figures**
- ![Oracle top-20 membership across ingest regions](figures/claim2_oracle_topk_membership.pdf)
- ![Selection disparity vs oracle over rounds, per baseline (uniform)](figures/claim2_misselection_vs_round_unif.pdf)
- ![Selection disparity vs oracle over rounds, per baseline (staggered)](figures/claim2_misselection_vs_round_stag.pdf)
- ![Utility regret vs oracle over rounds](figures/claim2_regret_vs_round_unif.pdf)
- ![Mean disparity vs time-to-60% (the key correlation)](figures/claim2_disparity_vs_ttt.pdf)

### Claim 3 — Felix with the eval selector does its job
Each baseline `B` is run alongside `B_oracle` (B's exact policy, fed *true* utilities
each round). Two readings:
- **Per-baseline staleness tax** = `time-to-60%(B) − time-to-60%(B_oracle)`: how much
  time B loses to stale prioritization. Felix's tax should be the **smallest** (its
  eval selector already partially refreshes utilities), feddance/oort/refl's larger.
- **Felix vs the others**: among the practical baselines, Felix is fastest to
  time-to-accuracy and its picks are closest to the true top-K (Claim-2 disparity).

**Figures**
- ![Accuracy vs sim-time, each baseline vs its oracle* (uniform)](figures/claim3_accuracy_vs_simtime_unif.pdf)
- ![Time-to-60%, baseline vs oracle (uniform)](figures/claim3_time_to_target_unif.pdf)
- ![Per-baseline staleness tax = T(base) − T(oracle)](figures/claim3_staleness_tax_unif.pdf)
- ![Mean overlap-with-oracle per baseline](figures/claim3_overlap_with_oracle.pdf)
- (phase 2) ![Uniform vs staggered ablation (Felix advantage grows)](figures/claim3_uniform_vs_staggered.pdf)

## 3. Experimental setup

| Knob | Value | Notes |
| --- | --- | --- |
| Dataset | CIFAR-10 | 50k train images |
| Trainers `n` | 50 | small enough for cheap central oracle eval; large enough for top-10-of-50 |
| Heterogeneity | Dirichlet **α=0.1** | most heterogeneous; split `cifar10_alpha0.1_n50.yaml` (min 28 / max 3310 / mean 1000 samples) |
| Availability | **`syn_0`** (100%) | isolates prioritization from availability churn |
| Aggregation goal | **10** | sync ⇒ `aggr_num=10`; async ⇒ `aggGoal=10` |
| Concurrency `c` | felix **10** | `B_oracle` keeps B's own `c`/`aggr_num`; injection refreshes *all* candidates regardless of `c` |
| Streaming | **uniform (phase 1)** | one `full_after_s=10800`. Staggered (per-client onset≤5400 s, ±50% span) = phase 2 |
| Time mode | **simulated** | virtual-clock time-to-accuracy, deterministic, fast |
| Target accuracy | **60%** | |
| Stop rule | **20 consecutive evals ≥ 60%** | resets on any dip; `rounds`/`max_experiment_runtime_s` safety cap |
| Eval cadence | every **10** rounds | 20 evals ≈ 200 sustained rounds |

**Arms (8).** `{felix, oort, refl, feddance} × {B, B_oracle}`. Each `B_oracle` uses
the **same selector** as `B`, plus aggregator-side true-utility injection. The 4
practical baselines also get the offline oracle replay for the per-round disparity
metrics.

## 4. The oracle (per baseline, two realizations)

The oracle answers: *what would **this** selector do, and how fast would it
converge, if it knew every client's true current utility at each selection step?*

- **Per-baseline online oracle `B_oracle`** — the same selector as `B`, but each
  round, *before* selection, the aggregator overwrites every candidate's
  `PROP_STAT_UTILITY` (and, for feddance, local accuracy) with the **true** current
  value: it reconstructs each client's currently-unlocked data prefix
  (deterministic from the Dirichlet split + streaming schedule) and forward-passes
  the *current* global model — `I_m = N·√(mean(loss²))`. Every selector then ranks
  oracularly with **zero selector changes**, and `B_oracle` becomes a genuine
  alternative training trajectory. The contrast `B` vs `B_oracle` (especially
  `time-to-60%`) is *that baseline's* staleness tax. (Implemented as an aggregator
  hook + `aggregator/pytorch/oracle_utility.py`; the aggregator is allowed to
  reconstruct data because it is an oracle, not a deployable policy.)
- **Offline checkpoint replay** — `scripts/analysis/oracle_misselection.py`
  reconstructs the same true utility from each baseline's checkpoints and joins it
  to the *believed* utility logged in `EVENT_SELECTION.per_trainer`, giving per-round
  top-K overlap / mis-selection rate / utility regret **along B's own trajectory**.
  It uses the identical utility definition, so it cross-checks the online oracle
  (B_oracle's mis-selection should be ≈ 0).

## 4b. Distributed execution & pooling (4 nodes)

Phase-1 wall-clock is parallelized so each node runs one `B + B_oracle` pair
sequentially (the oracle's per-round central eval is the long pole; pairing keeps
all nodes balanced):

| Node | HW | YAML | Arms |
| --- | --- | --- | --- |
| node1 | 128c/500GB | `expt_scripts_2026/...unif_node1.yaml` | felix + felix_oracle |
| node2 | 128c/500GB | `...unif_node2.yaml` | feddance + feddance_oracle |
| node3 | 96c/250GB | `...unif_node3.yaml` | oort + oort_oracle |
| node4 | 96c/250GB | `...unif_node4.yaml` | refl + refl_oracle |

**On each node** (the split + YAMLs are regenerated locally with that node's GPU count):
```bash
cd lib/python/examples/async_cifar10
FELIX_NUM_GPUS=<gpus on this node> scripts/run_felix_streaming.sh node <1-4>
```
Each node writes its runs under `experiments/run_*` (arm names are globally unique;
oracle runs are suffixed `_node<i>`).

**Pooling + analysis (on one box once all 4 nodes finish):**
1. `rsync` every node's `experiments/run_*_n50_alpha0.1_*` dir into a single
   `experiments/` on the analysis box (keep checkpoints + telemetry).
2. `cd lib/python/examples/async_cifar10 && scripts/run_felix_streaming.sh analyze`
   — runs `oracle_misselection.py` on each baseline run (per-baseline disparity)
   and `felix_streaming_figures.py` to emit all claim figures into
   `experiments/figures/`.
3. (Sanity) confirm each `B_oracle`'s mis-selection ≈ 0 vs the offline replay.

## 5. Why this design is sound

- **α=0.1 maximizes inter-client utility divergence**, so *which* clients you pick
  matters most — the regime where misprioritization is costly.
- **100% availability (`syn_0`)** removes availability churn, so observed
  differences are attributable to *utility staleness*, not who happens to be
  online. It also neutralizes REFL's availability machinery, reducing every
  baseline to its utility-ranking core — a fair comparison.
- **n=50** keeps the aggregator's per-round central oracle eval (one forward pass
  per candidate) cheap and dodges MQTT-scale issues, yet is large enough that
  top-10-of-50 selection is meaningful.
- **aggGoal=10 fixed** across baselines controls aggregation granularity /
  effective parallelism, removing it as a confound.
- **Streaming makes utility non-stationary** — the precondition the thesis needs.
  Phase 2's **uniform vs staggered** ablation will show the effect is not an
  artifact of a single global unlock clock: staggering (clients' data arriving in
  different windows) should *amplify* the top-K churn and the baselines' disparity.
- **"20 consecutive evals ≥ 60%"** yields a stable time-to-accuracy (not a lucky
  spike); the `rounds`/`max_experiment_runtime_s` cap guarantees non-converging arms still
  terminate.
- **Simulated mode** gives a clean virtual-clock time axis, determinism, and speed.

## 6. How to reproduce

```bash
cd lib/python/examples/async_cifar10

# --- quick single-node sanity pass (all 8 arms, tiny) ---
scripts/run_felix_streaming.sh smoke

# --- 4-node production run (phase 1) ---
# On each node i (1..4), with that node's GPU count:
FELIX_NUM_GPUS=<gpus> scripts/run_felix_streaming.sh node <i>
# -> writes experiments/run_*  (baseline + baseline_oracle for that node)

# --- pool + analyze on ONE box after all nodes finish ---
# rsync every node's experiments/run_*_n50_alpha0.1_* into one experiments/, then:
scripts/run_felix_streaming.sh analyze
#   = oracle_misselection.py per baseline run  +  felix_streaming_figures.py
```

Manual equivalent of `analyze`:
```bash
for d in experiments/run_*_n50_alpha0.1_*; do            # skip *_oracle_* if you like
  python ../../../scripts/analysis/oracle_misselection.py --run-dir "$d"; done
python ../../../scripts/analysis/felix_streaming_figures.py --runs-root experiments --target 0.60
```

Figures land in `experiments/figures/`; drop them into the placeholders above.
All commands assume the `dg_flame` conda env. **Phase 2 (staggered)**: set
`STAGGER_CONDITIONS = [False, True]` in `scripts/gen_n50_experiment.py` and re-run.
