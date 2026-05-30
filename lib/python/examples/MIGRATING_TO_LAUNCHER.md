# Migrating an example to the YAML launcher + shared metadata

This is the reference playbook for moving an example off per-trainer JSON configs
and ad-hoc shell scripts onto the `flame.launch` YAML launcher with the shared
`examples/_metadata/` bundle. `async_cifar10/` is the worked reference
implementation; mirror it.

- **How to *run*** the launcher and the experiment-YAML shape: see
  [`examples/README.md`](README.md).
- **Why** we moved off thousands of JSON files: see
  [`async_cifar10/MIGRATION_PLAN.md`](async_cifar10/MIGRATION_PLAN.md) (historical).
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

### What the launcher injects automatically per trainer

`flame/launch/spawner.py:ConfigGenerator.generate_trainer_config()` merges
these layers in order (last wins for leaf values):

1. `configs/trainer_base.yaml` — static per-example template
2. `baseline.trainer` dict from `baselines.yaml` — selector/optimizer/notify defaults
3. Per-trainer metadata from `_metadata/`:
   - `taskid` (from registry `task_id`)
   - `training_delay_s` (from registry, 4–18 s range for the 300-trainer pool)
   - `trainer_indices_list` (from dataset splits; for non-CIFAR examples see §6)
   - **All five availability traces pre-injected**: `avl_events_syn_0`, `avl_events_syn_20`,
     `avl_events_syn_50`, `avl_events_mobiperf_2st`, `avl_events_mobiperf_3st_50`,
     `avl_events_mobiperf_3st_75` — trainer selects at runtime via
     `client_notify.trace`
4. `experiment.trainer.config_overrides` — experiment-level deep-merge
5. Dotted-key overrides: `job.id`, `job.name`, `hyperparameters.*`

All five traces are pre-injected so the trainer never has to be re-spawned to
switch availability modes within a batch.

### Adding a new dataset's data

1. **Index-style splits** (e.g., CIFAR-10). Generate
   `dataset_splits/<dataset>_alpha<a>_n<N>.yaml` with a `trainer_data_splits`
   map (`trainer_001 … trainer_NNN` → list of sample indices). Reuse the
   existing split-generation script; do **not** hand-write splits.
2. **Path-style datasets** (e.g., NLP datasets from H5 partitions). The shared
   `dataset_splits` YAML cannot hold per-trainer index lists; instead, store
   `data_file_path`, `partition_file_path`, and `client_idx` directly in
   `trainer_base.yaml` as placeholders, and inject them via
   `experiment.trainer.config_overrides.hyperparameters`. See §6 for the
   fwdllm-specific guidance.
3. **Trainer population.** If the new example needs a different `N` or device
   timing, extend `trainer_registry.yaml` (or add a parallel registry and point
   `experiment.metadata.registry` at it). The registry is dataset-agnostic —
   reuse it when the device population/timing is the same.
4. **Availability traces.** Reuse `synthetic_traces.yaml` / `mobiperf_traces.yaml`
   as-is — availability is independent of the dataset. Only add traces if you
   need new patterns.

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

   ```python
   from flame.launch.cli import load_config_from_argv
   config = load_config_from_argv()
   ```

3. **Gate wandb behind `--log_to_wandb`.** No module-level `wandb.init()` (it
   runs on import and blocks non-wandb runs). Mirror `main_fedavg_agg.py`'s
   `initialize_wandb()` + `self.log_to_wandb` pattern.

4. **Availability comes from `_metadata`**, not per-trainer JSON. For oracular
   stacks, `read_trainer_unavailability(trace)` reads
   `_metadata/{trainer_registry.yaml, availability_traces/*}`. Use
   `track_trainer_avail.get('trace', '<unset>')` — the key is absent when
   tracking is disabled.

5. **Telemetry**: the launcher sets `FLAME_TELEMETRY_DIR` before spawning; the
   aggregator should write structured JSONL events there. Typical events:
   `selection` (selected trainer IDs, scores, round) and `aggregation`
   (aggregation time, staleness stats). See §5 for the full telemetry contract.

---

## 3. Trainer-side changes

The trainer is spawned with `--config-json`; everything per-trainer arrives in
`hyperparameters`. The trainer must read (not hardcode):

- `trainer_indices_list` — dataset sample indices (from `dataset_splits`).
  For path-style datasets, use `data_file_path` + `partition_file_path` +
  `client_idx` instead (§6).
- `training_delay_s` — per-trainer delay (from registry).
- `training_delay_enabled` — bool gate on delay (from config_overrides or
  trainer_base.yaml default).
- `avl_events_<trace>` — all five trace types pre-injected; trainer reads
  `client_notify.trace` to pick the active one at runtime.
- `client_notify.{enabled,trace}` — availability-aware notify; `trace` is always
  present (launcher defaults it to `syn_0`).

Trainer entrypoint uses `load_config_from_argv()`:

```python
from flame.launch.cli import load_config_from_argv
config = load_config_from_argv()
```

One trainer `main.py` is typically enough across stacks.

### Time mode (real vs simulated)

The launcher passes `--time_mode real|simulated` as a CLI-only arg (not in config
JSON). The trainer reads it from `sys.argv` alongside the config. Two modes:

- **real**: trainer sleeps `training_delay_s` at true wall-clock pace. Availability
  transitions happen at real-time intervals. Use for wall-clock benchmarks.
- **simulated**: no sleeps; the trainer reports its completion time back to the
  aggregator so the aggregator can order updates by a *virtual clock*. Availability
  states are evaluated against that virtual clock. Use for fast iteration.

Set `time_mode` in the experiment YAML under `trainer.time_mode`. The spawner
passes it as `--time_mode` to each trainer subprocess.

### Data streaming (optional)

By default a trainer loads its **entire** partition at init and reuses it on every
selection. Real devices instead generate data continuously, so a client selected
early should train on *less* data than one selected late. `async_cifar10/main.py`
is the reference implementation — mirror it when the example wants realistic data
growth.

- **Config** (on `hyperparameters`, disabled by default in `trainer_base.yaml`):
  ```yaml
  data_streaming:
    enabled: "False"
    full_data_available_after_s: 0   # sim-seconds until 100% data is visible
  ```
  Enable per experiment under `trainer.config_overrides.hyperparameters.data_streaming`.
- **Algorithm**: at `load_data()` retain the full pool and a one-time shuffle seeded
  by `trainer_id`. A helper exposes a growing prefix:
  `visible = floor(min(1, sim_elapsed / X) * total)`, floored at 1 sample; once
  `sim_elapsed >= X` the full pool stays visible. `sim_elapsed` is driven by the
  sim clock (real-mode: wall-clock elapsed; simulated-mode: aggregator-stamped
  virtual time).
- **Where**: the loader is rebuilt at the top of `train()` and `evaluate()` so
  the visible subset (and the `dataset_size` reported to the aggregator) reflects
  the current time. All a no-op when `enabled: "False"`.

### Utility counterfactual (optional telemetry)

Emit a comparison of streamed-prefix statistical utility vs full-pool utility to
measure the data-availability impact on selection bias.

```yaml
util_counterfactual:
  enabled: "True"
  every_n_rounds: 1
  sample_size: 256
```

Enable under `trainer.config_overrides.hyperparameters.util_counterfactual`.
The trainer emits `util_disparity` JSONL events; the post-run analyzer produces
streamed-vs-full disparity plots.

### Memory profiler (optional)

`async_cifar10/trainer/pytorch/memory_profiler.py` provides a `MemoryProfiler`
class that logs RSS, Python object counts, and torch tensor sizes (CPU + GPU)
every N rounds. Import and instantiate it in the trainer `__init__`; call
`log_memory_before_round()` at the top of the training loop. Useful for catching
leaks in long runs.

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

## 5. Telemetry and post-run analysis

Telemetry is **auto-enabled** for any run launched via the YAML runner: the runner
sets `FLAME_TELEMETRY_DIR=<exp_dir>/telemetry/` before spawning processes; any
child process that imports `flame.telemetry` (or checks the env var) will write
to that directory. No broker or extra setup required.

### JSONL event contract

Each process writes one file: `trainer_<id>.jsonl` or `aggregator.jsonl`. Each
line is a JSON object with at least `{"event": "<name>", "ts": <iso8601>, ...}`.

**Trainer events** (implement these for useful plots):

| Event | Required fields | Purpose |
|-------|-----------------|---------|
| `trainer_round` | `round`, `loss`, `accuracy`, `train_time_s`, `wait_time_s` | Per-round training stats |
| `avail_change` | `round`, `old_state`, `new_state` | State-machine transitions |
| `util_disparity` | `round`, `streamed_util`, `full_util` | Data-streaming disparity |

**Aggregator events**:

| Event | Required fields | Purpose |
|-------|-----------------|---------|
| `selection` | `round`, `selected_ids`, `scores` | Which trainers were chosen |
| `aggregation` | `round`, `agg_time_s`, `staleness_stats` | Aggregation timing |

### Post-run analysis scripts

The runner calls analysis best-effort after experiment completion. Two scripts are
available in `async_cifar10/scripts/`:

- **`analyze_send_recv_lag.py`**: parses `[SEND_RECV_LAG]` log entries from the
  aggregator log; reports per-trainer median/p95/max lag. Run manually:
  ```bash
  python scripts/analyze_send_recv_lag.py <exp_dir>/*_aggregator.log [--warn-threshold-s 5.0]
  ```
- **`compare_parity.py`**: compares two experiment runs (e.g., real vs simulated
  time_mode) for trajectory parity.

For full telemetry plot generation, point a post-processing script at the
`telemetry/` directory. The JSONL files are append-only and line-buffered, so
`tail -f` works during a live run.

### Output directory structure

```
experiments/run_YYYYMMDD_HHMMSS_<name>/
  aggregator_config.json             # merged aggregator config used
  execution_config.yaml              # compact metadata refs + spawn commands
  snapshot.yaml                      # git info + metadata SHA256 checksums
  YYYYMMDD_HHMMSS_*_aggregator.log   # aggregator stdout/stderr
  YYYYMMDD_HHMMSS_*_trainers.log     # all trainers combined (line-buffered)
  YYYYMMDD_HHMMSS_*_resources.log    # RAM/GPU monitoring (if enabled)
  telemetry/
    aggregator.jsonl
    trainer_1.jsonl … trainer_N.jsonl
  plots/                             # generated by post-run analysis (if run)
```

---

## 6. fwdllm-specific migration notes

fwdllm differs from async_cifar10 in two key areas: (a) its dataset is stored as
H5 partitions on disk (not index lists injected by the spawner), and (b) its
existing trainer/aggregator already use `flame.config.Config` and have partial
availability support. This makes the port narrower than a greenfield migration.

### What's already compatible

- Trainer (`trainer/fl_main.py`) and aggregator (`aggregator/fl_main.py`) both
  use `flame.config.Config`. Only the config-loading call needs to change.
- Availability notification thread and `client_notify` struct are already present.
- Oracular tracking (`track_trainer_avail`) is already implemented in
  `FedSGDAggregator.py`.

### What must change

**1. Config intake** (trainer and aggregator)

Replace:
```python
parser.add_argument("--config", required=True)
config = Config(args.config)
```
with:
```python
from flame.launch.cli import load_config_from_argv
config = load_config_from_argv()
```
Also add `--time_mode` as a side-channel CLI arg (not in config JSON) if you want
simulated mode support:
```python
import argparse, sys
_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--time_mode", default="real")
_known, _ = _p.parse_known_args()
time_mode = _known.time_mode
```

**2. trainer_base.yaml for fwdllm**

Create `fwdllm/configs/trainer_base.yaml`. Unlike async_cifar10, the dataset
fields cannot come from `_metadata/dataset_splits/` (H5 paths are not index
lists). Put the dataset fields in the base template with placeholder comments, and
let experiments override them via `config_overrides.hyperparameters`:

```yaml
# fwdllm/configs/trainer_base.yaml (minimal skeleton)
backend: mqtt
brokers:
  - host: localhost
    port: 1883
channels:
  # ... same channel block as current trainer JSON configs
hyperparameters:
  # Dataset — override per experiment or per baseline
  dataset: agnews                          # override in experiment YAML
  data_file_path: PLACEHOLDER              # inject via config_overrides
  partition_file_path: PLACEHOLDER         # inject via config_overrides
  partition_method: niid_label_clients=100_alpha=1
  # Model
  model_type: distilbert
  model_name: distilbert-base-uncased
  max_seq_length: 64
  peft_method: adapter
  # FL
  fl_algorithm: FedFwd
  epochs: 1
  comm_round: 3000
  learning_rate: 0.01
  server_lr: 0.1
  train_batch_size: 32
  eval_batch_size: 32
  forward_mode: "True"
  use_adapter: "True"
  freeze_layers: "True"
  fp16: "True"
  # Per-trainer fields — injected by spawner
  client_idx: 0                            # injected: (trainer_id - 1) % 100
  training_delay_s: 4.0                   # injected from registry
  training_delay_enabled: "False"
  training_delay_factor: "10"
  # Availability
  wait_until_next_avl: "True"
  client_notify:
    enabled: "False"
    trace: syn_0
  # avl_events_* — injected by spawner for all five trace types
```

**3. Per-trainer `client_idx` injection**

fwdllm uses `client_idx` (0–99) to select the trainer's data partition from the
H5 file. The spawner injects `taskid` and `training_delay_s` automatically from
the registry. For `client_idx`, add it to `experiment.trainer.config_overrides`
or implement a custom injection hook in a thin wrapper around `TrainerSpawner`.

Simplest approach: add `client_idx` as a dotted-key override per trainer. In the
experiment YAML, set a formula under `config_overrides.hyperparameters.client_idx`
that the spawner evaluates to `(trainer_id - 1) % num_clients`. If the spawner
doesn't support per-trainer formula evaluation, inject it via a small wrapper
script that calls the spawner with per-trainer overrides.

**4. Baselines in `_metadata/baselines.yaml`**

Add fwdllm baselines. Minimal entries:

```yaml
fedfwd_async_oort:
  description: FedFwd + async_oort selector + fedbuff optimizer (asyncfl stack)
  example:
    aggregator_main: aggregator/main_asyncfl_agg.py
  aggregator:
    selector:
      sort: async_oort
      kwargs:
        evalGoalFactor: 1.0
    optimizer:
      sort: fedbuff
      kwargs: {}
    hyperparameters:
      trackTrainerAvail:
        enabled: "False"
        type: "NA"
  trainer:
    hyperparameters:
      client_notify:
        enabled: "True"
      fl_algorithm: FedFwd
      forward_mode: "True"

fedfwd_oort_oracular:
  description: FedFwd + oort selector + fedavg optimizer + oracular tracking (syncfl stack)
  example:
    aggregator_main: aggregator/main_oort_sync_agg.py
  aggregator:
    selector:
      sort: oort
      kwargs: {}
    optimizer:
      sort: fedavg
      kwargs: {}
    hyperparameters:
      trackTrainerAvail:
        enabled: "True"
        type: oracular
  trainer:
    hyperparameters:
      client_notify:
        enabled: "False"
      fl_algorithm: FedFwd
```

**5. Availability traces** — reuse `synthetic_traces.yaml` and
`mobiperf_traces.yaml` as-is. fwdllm currently embeds trace arrays inline in
trainer JSONs; the launcher will inject them from `_metadata/` instead. No new
trace files needed for the standard syn_0/syn_20/syn_50/mobiperf variants.

**6. Aggregator entrypoints**

fwdllm currently has a single `aggregator/fl_main.py`. Split into per-stack
files matching async_cifar10's pattern, then register each in `baselines.yaml`:

| Stack | File | Baseline |
|-------|------|---------|
| asyncfl | `aggregator/main_asyncfl_agg.py` | fedfwd_async_oort |
| syncfl (oort) | `aggregator/main_oort_sync_agg.py` | fedfwd_oort_oracular |

Each file only changes the `TopAggregator` import; all FL logic stays in
`FedSGDAggregator.py`.

**7. Wandb gating**

fwdllm aggregator likely calls `wandb.init()` unconditionally. Gate it behind
`--log_to_wandb` following `main_fedavg_agg.py`'s `initialize_wandb()` pattern.
The launcher sets `log_to_wandb: false` by default; override in the experiment
YAML if needed.

**8. Telemetry**

fwdllm does not yet emit structured JSONL telemetry. Add `trainer_round` events
at minimum to get loss/accuracy curves from the post-run analyzer. The env var
`FLAME_TELEMETRY_DIR` is set by the launcher; check it and open a JSONL file:

```python
import json, os, pathlib
_tel_dir = os.environ.get("FLAME_TELEMETRY_DIR")
_tel_file = open(pathlib.Path(_tel_dir) / f"trainer_{trainer_id}.jsonl", "a") if _tel_dir else None

def _emit(event, **fields):
    if _tel_file:
        _tel_file.write(json.dumps({"event": event, **fields}) + "\n")
        _tel_file.flush()
```

---

## 7. Migration checklist for a new example

1. Symlink `<example>/metadata → ../_metadata` (or set `experiment.metadata.dir`).
2. Add `dataset_splits/<dataset>_alpha<a>_n<N>.yaml`; reuse registry + traces.
   For path-style datasets (fwdllm), skip this step and inject paths via
   `config_overrides` instead (§6).
3. Add `<example>/configs/trainer_base.yaml` (static per-example trainer
   template; per-trainer fields are injected by the launcher).
4. Provide one aggregator entrypoint per stack you run (§2); delete the rest.
5. Make trainer + aggregator entrypoints use `load_config_from_argv()` and read
   all per-trainer values from `hyperparameters` (§2–§3).
6. Add the example's baselines to `baselines.yaml` with `example.aggregator_main`.
7. Add `expt_scripts/<baseline>_n10_*_smoke.yaml` experiment YAMLs.
8. Validate without spawning: load the YAML, resolve the baseline, build the
   aggregator config, and run `_validate_stack`. Then run a 10-trainer smoke test.
9. (Optional) Add `trainer_round` / `avail_change` JSONL telemetry events to
   enable post-run analysis plots (§5).
10. (Optional) Port the `data_streaming` block into `trainer_base.yaml` + the
    trainer `main.py` if the example wants streamed data growth (§3 Data streaming).
11. (Optional) Add `util_counterfactual` support if comparing streamed vs full
    utility matters for this example (§3 Utility counterfactual).
12. Delete the example's legacy JSON config dirs and mark its shell scripts
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
| `async_cifar10` | Migrated (reference). 5 baselines: felix, fedbuff, fedavg, oort, refl. Includes telemetry, streaming, time_mode, memory profiler. |
| `feddance_cifar10` | Has launcher YAMLs; FedDance baseline blockers tracked in `async_cifar10/FEDDANCE_TODO.md`. |
| `async_google_speech` | **TODO** — migrate per this guide (needs google-speech dataset splits + per-stack aggregator entrypoints). |
| `fwdllm` | **TODO** — see §6 for fwdllm-specific steps. Main blockers: (1) config intake switch to `load_config_from_argv()`; (2) per-trainer `client_idx` injection; (3) split aggregator into per-stack entrypoints; (4) add baselines to `baselines.yaml`; (5) gate wandb; (6) add JSONL telemetry. Dataset splits are H5-path-based, not index-list-based — use `config_overrides` for data paths rather than `_metadata/dataset_splits/`. |
