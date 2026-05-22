# Migrating an example to the YAML launcher + shared metadata

This is the reference playbook for moving an example off per-trainer JSON configs
and ad-hoc shell scripts onto the `flame.launch` YAML launcher with the shared
`examples/_metadata/` bundle. `async_cifar10/` is the worked reference
implementation; mirror it.

- **How to *run*** the launcher and the experiment-YAML shape: see
  [`examples/README.md`](README.md).
- **Why** we moved off thousands of JSON files: see
  [`async_cifar10/MIGRATION_PLAN.md`](async_cifar10/MIGRATION_PLAN.md).
- This doc: **what to change** (trainer + aggregator + metadata) to migrate a new
  example, and what legacy to delete.

Backward compatibility is intentionally not preserved. The launcher is the only
supported run path; legacy JSON/shell entrypoints are decommissioned (see
[Legacy decommission](#legacy-decommission)).

---

## 1. The shared metadata bundle (`examples/_metadata/`)

One bundle, reused across examples. The launcher injects per-trainer data from it
so no example stores per-trainer JSON anymore.

| File | Holds | Lookup key (in `flame/launch/spawner.py`) |
|------|-------|--------------------------------------------|
| `trainer_registry.yaml` | `trainer_NNN` → `{trainer_id, task_id, training_delay_s, speed_class, mobiperf_device_id}` | `trainer_{id:03d}` |
| `dataset_splits/<dataset>_alpha<a>_n<N>.yaml` | `trainer_data_splits.trainer_NNN` → list of sample indices | `f"{dataset}_alpha{alpha}_n{N}"` |
| `availability_traces/synthetic_traces.yaml` | `traces.syn_{0,20,50}` → `[[ts,state],...]` | `traces[trace_name]` |
| `availability_traces/mobiperf_traces.yaml` | `traces.device_NNN.states_{2st,3st_50,3st_75}` | `device_{id:03d}` |
| `baselines.yaml` | baseline catalog (selector/optimizer/stack/tracking) | `baselines[<name>]` |
| `aggregator_base.json` | generic aggregator config template (`PLACEHOLDER` selector/optimizer/job) | — |

`async_cifar10/metadata` is a symlink to `../_metadata`, so an example points at
the shared bundle by symlinking (or via `experiment.metadata.dir`).

### Adding a new dataset's data

1. **Dataset splits.** Generate `dataset_splits/<dataset>_alpha<a>_n<N>.yaml` with
   a `trainer_data_splits` map (`trainer_001 … trainer_NNN` → index lists).
   Reuse the existing split-generation script; do **not** hand-write splits.
2. **Trainer population.** If the new example needs a different `N` or device
   timing, extend `trainer_registry.yaml` (or add a parallel registry and point
   `experiment.metadata.registry` at it). The registry is dataset-agnostic —
   reuse it when the device population/timing is the same.
3. **Availability traces.** Reuse `synthetic_traces.yaml` / `mobiperf_traces.yaml`
   as-is — availability is independent of the dataset. Only add traces if you
   need new patterns.

So data, availability, and timing are **shared**; usually only a new
`dataset_splits/*.yaml` is required per dataset.

---

## 2. Aggregator-side changes

1. **One entrypoint per stack**, named for the stack (see async_cifar10):
   - `main_asyncfl_agg.py` → `flame.mode.horizontal.asyncfl.top_aggregator`
   - `main_oort_sync_agg.py` → `flame.mode.horizontal.oort.top_aggregator`
   - `main_fedavg_agg.py` → `flame.mode.horizontal.top_aggregator` (base syncfl)

   Delete legacy `main*.py` that don't map to a stack you run.
2. **Config intake via `load_config_from_argv()`** (`flame/launch/cli.py`). The
   launcher spawns aggregators with `--config-json`; manual runs can use
   `--config <file>`. Do not parse a positional config path.
3. **Gate wandb behind `--log_to_wandb`.** No module-level `wandb.init()` (it
   runs on import and blocks non-wandb runs). Mirror `main_fedavg_agg.py`'s
   `initialize_wandb()` + `self.log_to_wandb` pattern.
4. **Availability comes from `_metadata`**, not per-trainer JSON. For oracular
   stacks, `read_trainer_unavailability(trace)` reads
   `_metadata/{trainer_registry.yaml, availability_traces/*}`. Use
   `track_trainer_avail.get('trace', '<unset>')` — the key is absent when
   tracking is disabled.

## 3. Trainer-side changes

The trainer is spawned with `--config-json`; everything per-trainer arrives in
`hyperparameters`. The trainer must read (not hardcode):

- `trainer_indices_list` — dataset sample indices (from `dataset_splits`).
- `training_delay_s` — per-trainer delay (from registry).
- `avl_events_<trace>` — availability events for the active trace.
- `client_notify.{enabled,trace}` — availability-aware notify; `trace` is always
  present (the launcher defaults it to `syn_0`).

Trainer entrypoint also uses `load_config_from_argv()`. One trainer `main.py` is
typically enough across stacks.

---

## 4. Baseline catalog (`baselines.yaml`)

A baseline names the selector + optimizer + aggregator stack + tracking in one
place; an experiment YAML says `baseline: <name>` and overrides only specifics.

```yaml
<name>:
  description: ...
  example:
    aggregator_main: aggregator/pytorch/main_<stack>_agg.py   # owns the stack
  aggregator:        # deep-merged into the aggregator JSON
    selector: {sort: ..., kwargs: {...}}
    optimizer: {sort: ..., kwargs: {...}}
    hyperparameters: {trackTrainerAvail: {...}}
  trainer:           # deep-merged into each trainer config
    hyperparameters: {client_notify: {...}}
```

The launcher resolves `aggregator_main` from the baseline (an experiment may not
override it) and **fails fast on a selector/stack mismatch** — async selectors
(`async_oort`, `async_random`, `fedbuff`) must run on the asyncfl stack;
everything else on a sync stack (`flame/launch/runner.py:_validate_stack`).

---

## 5. Migration checklist for a new example

1. Symlink `<example>/metadata → ../_metadata` (or set `experiment.metadata.dir`).
2. Add `dataset_splits/<dataset>_alpha<a>_n<N>.yaml`; reuse registry + traces.
3. Add `<example>/configs/trainer_base.yaml` (static per-example trainer
   template; per-trainer fields are injected by the launcher).
4. Provide one aggregator entrypoint per stack you run (§2); delete the rest.
5. Make trainer + aggregator entrypoints intake `--config-json` and read all
   per-trainer values from `hyperparameters` (§2–§3).
6. Add the example's baselines to `baselines.yaml` with `example.aggregator_main`.
7. Add `expt_scripts/<baseline>_n10_*_smoke.yaml` experiment YAMLs.
8. Validate without spawning: load the YAML, resolve the baseline, build the
   aggregator config, and run `_validate_stack` (see the dry-run snippet in the
   commit history / `runner` API). Then run a 10-trainer smoke test.
9. Delete the example's legacy JSON config dirs and mark its shell scripts
   deprecated (§Legacy decommission).

---

## Legacy decommission

When migrating, remove what the launcher path doesn't use, so deployment isn't
confusing. Capture the removals here or in the example's README.

Remove / mark deprecated:
- Per-trainer JSON config directories (replaced by `dataset_splits` + launcher).
- Aggregator entrypoints that don't map to a running stack.
- Module-level `wandb.init()` and positional-config argument parsing.
- Hand-rolled launch shell scripts → add a `DEPRECATED.md` per scripts dir
  pointing at the launcher (see async_cifar10's deprecated dirs).

Keep (still current):
- `--config <file>` manual run path for single-process debugging.
- Historical experiment shell scripts *only* as deprecated-marked records.

---

## Status of other examples

| Example | Status |
|---------|--------|
| `async_cifar10` | Migrated (reference). 5 baselines: felix, fedbuff, fedavg, oort, refl. |
| `feddance_cifar10` | Has launcher YAMLs; FedDance baseline blockers tracked in `async_cifar10/FEDDANCE_TODO.md`. |
| `async_google_speech` | **TODO** — migrate per this guide (needs google-speech dataset splits + per-stack aggregator entrypoints). |
| `fwdllm` | **TODO** — migrate per this guide. |
