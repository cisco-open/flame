# Trainer Config Migration Plan

Pause point: 2026-05-20. Pick up here.

This document captures the plan to migrate per-trainer JSON configs into the
shared YAML metadata bundle used by `flame.launch`, with strict verification
before any deletion.

---

## Status snapshot

| Step | State | Notes |
| :--- | :--- | :--- |
| Survey JSON variation in `cifar10/` and `async_cifar10/` | done | findings below |
| Locked design decisions (scope, verification, deletion, script location) | done | §2 |
| Migration script (`lib/python/scripts/migrate_trainer_configs.py`) | **partial, uncommitted in working tree** | needs review + completion (§4) |
| Verification script (`lib/python/scripts/verify_migration.py`) | not started | spec in §5 |
| Migration plan YAML (`examples/cifar10/_migration_plan.yaml`) | not generated | §6 |
| Migrated metadata (`examples/_metadata/...`) | not generated | §6 |
| Verification run | not started | §6 |
| Stage JSONs in `_pending_deletion/` | not started | §7 |
| `examples/README.md` for launcher usage | not started | §8 |

The two earlier commits (FedDance baseline, generic launcher) on `dg/feddance_impl`
are unchanged and unrelated to this migration. Resume work on the same branch.

---

## 1. Survey findings (recorded for reference)

### cifar10/trainer/ — simple
- 16 sub-locations: 15 `configN/` dirs (each with 10 `trainer_*.json`) + 2
  root-level files (`config.json`, `config2.json`).
- 150 trainer JSONs total in the `configN/` dirs.
- **Only `taskid` and `hyperparameters.trainer_indices_list` vary per trainer.**
- Cross-dir variation: `realm` differs (`default/us` for `config10..15`,
  `default/us/west` for `config1..9`).
- Constants across all 150 JSONs: `rounds=5`, `job.id="622a358619ab59012eabeefb"`,
  `batchSize=32`, `learningRate=0.01`, etc.
- The 10 `taskid` values are stable across all 15 config dirs (trainer N has the
  same `taskid` in every dir).
- Root-level `config.json` (`taskid=...370`, indices = `[1..10]`) and
  `config2.json` (`taskid=...371`, no indices) are template/test files, not part
  of the dataset-heterogeneity migration. Leave them alone for now.

### async_cifar10/trainer/ — large, more shapes
- 26 config dirs, 5,935 JSON files.
- **11 distinct varying-field signatures**:

| Signature (top-level / hyperparameters) | Dirs |
| :--- | :--- |
| `taskid` + `realm` / `trainer_indices_list` | 1 (300 files) |
| `taskid` + `realm` / `failure_durations_s`, `trainer_indices_list`, `training_delay_s` | 2 (200) |
| `taskid` + `realm` / `failure_durations_s`, `trainer_indices_list` | 1 (100) |
| `taskid` / `failure_durations_s`, `trainer_indices_list`, `training_delay_s` | 7 (2100) |
| `taskid` / 4× `avl_events_*`, `client_notify`, `trainer_indices_list`, `training_delay_s` | 4 (1240) |
| `taskid` / 5× `avl_events_*`, `client_notify`, `trainer_indices_list`, `training_delay_s`, `wait_until_next_avl` (mobiperf set) | 1 (310) |
| `taskid` + `realm` / — | 3 (300) |
| `taskid` + `channels` + `groupAssociation` + `realm` / `failure_durations_s`, `heartbeats`, `learningRate`, `trainer_indices_list`, `training_delay_s` | 2 (224) |
| `taskid` + `realm` / `failure_durations_s` | 1 (110) |
| `taskid` / `three_state_avl_event_ts`, `trainer_indices_list`, `training_delay_s`, `two_state_unavl_durations_s` | 1 (310) |
| `taskid` / 5× `avl_events_*` (syn variant), `client_notify`, `trainer_indices_list`, `training_delay_s`, `wait_until_next_avl` | 3 (930) |

- Some `avl_events_*` fields may already be byte-identical to what the
  shared `examples/_metadata/availability_traces/` files contain. Verify
  before deciding whether those JSONs are reducible or need their own
  trace files.
- **Out of scope for this round.** Tackle async_cifar10 only after cifar10
  proves the toolchain end-to-end.

---

## 2. Locked decisions

These came from the AskUserQuestion responses on 2026-05-19:

1. **Scope this round: cifar10 only.** Prove the toolchain. Defer async_cifar10.
2. **Verification: strict byte-equivalence.** Reconstruct every original JSON
   from the new metadata + plan, serialize with identical formatting, byte-
   compare. Any mismatch = fail.
3. **Deletion: stage for manual review.** Move verified-redundant JSONs to a
   `_pending_deletion/` dir. Commit the move. User spot-checks. A follow-up
   commit deletes. Reversible.
4. **Script location: `lib/python/scripts/`.** Run as
   `python -m scripts.migrate_trainer_configs` and
   `python -m scripts.verify_migration`.

---

## 3. Target schema for the migration output

Two files written under `examples/_metadata/`:

- **`trainer_registry_<namespace>.yaml`** — one entry per logical trainer
  (1..10 for cifar10), keyed `trainer_001`..`trainer_010`, holding
  `task_id` and `trainer_id`. Mirrors the existing
  `trainer_registry.yaml` schema used by `flame.launch.spawner.MetadataLoader`.
- **`dataset_splits/<dataset>_<namespace>_<configN>.yaml`** × 15 — each holds
  `trainer_data_splits: { trainer_001: [..indices..], ... }`. Same schema as
  the existing `cifar10_alpha*_n300.yaml` files.

One control file written under the example dir:

- **`examples/cifar10/_migration_plan.yaml`** — the contract used by the
  verifier. For each source `configN/`, it records:
  - `dir_overrides`: top-level constants (full snapshot of `realm`, `channels`,
    etc., regardless of whether they match trainer_base.yaml — guarantees the
    verifier doesn't rely on `trainer_base.yaml` matching legacy values).
  - `dir_hp_overrides`: hyperparameters constant within the dir (rounds,
    batchSize, learningRate).
  - `dataset_split_file`: the path to the dataset_split YAML for this dir.
  - `trainer_ids`: ordered list of trainer numbers seen in the dir.
  - `per_trainer_extras_in_split`: bool; True only if per-trainer variation
    extends beyond `trainer_indices_list` (false for cifar10).

This is **lossless** by construction: every key in every source JSON is captured
in one of: `trainer_registry_<ns>.yaml` (taskid), `dataset_splits/<>.yaml`
(indices), or `_migration_plan.yaml::directories[].dir_*overrides` (everything
else).

---

## 4. Migration script (partial, in working tree)

Path: `lib/python/scripts/migrate_trainer_configs.py` (uncommitted).

What it already does:
- Loads every `configN/trainer_*.json` under `<example>/trainer/`.
- Classifies keys into (constant-in-dir, varying-in-dir) at top level + inside
  `hyperparameters`.
- Validates `taskid` is stable per logical trainer across all dirs.
- Writes the trainer registry, one dataset_split YAML per config dir, and the
  migration plan YAML.
- CLI:
  ```bash
  python -m scripts.migrate_trainer_configs \
      --example lib/python/examples/cifar10 \
      --metadata-out lib/python/examples/_metadata \
      --plan-out lib/python/examples/cifar10/_migration_plan.yaml \
      --namespace sync \
      --dataset-name cifar10
  ```

What still needs review / completion:
- **Test the script** by dry-running on cifar10 and inspecting the output YAMLs.
- **Edge case**: confirm the cifar10 root-level `config.json` / `config2.json`
  files are correctly ignored (they have no `configN/` parent dir).
- **Idempotency**: rerunning should overwrite previous outputs cleanly, not
  append. (Current code: `yaml.safe_dump` to a fresh open — fine.)
- **Unit test**: add `lib/python/tests/scripts/test_migrate.py` that runs the
  script on a synthetic mini cifar10 layout (3 trainers, 2 configs) and asserts
  the output structure.

---

## 5. Verification script (to write tomorrow)

Path: `lib/python/scripts/verify_migration.py`. Spec:

```python
def reconstruct_trainer_json(
    plan: dict,           # parsed _migration_plan.yaml
    registry: dict,       # parsed trainer_registry_<ns>.yaml
    config_dir_name: str, # e.g. "config1"
    trainer_id: int,      # e.g. 1
) -> dict:
    """Produce a dict identical to the original trainer_N.json."""
    out = {}
    dir_entry = next(d for d in plan["directories"] if d["name"] == config_dir_name)
    # 1. Start from dir_overrides (which already include shared top-level keys).
    out.update(deepcopy(dir_entry["dir_overrides"]))
    # 2. Inject taskid from registry.
    out["taskid"] = registry["trainers"][f"trainer_{trainer_id:03d}"]["task_id"]
    # 3. Inject hyperparameters: dir_hp_overrides + trainer_indices_list from split.
    out["hyperparameters"] = deepcopy(dir_entry["dir_hp_overrides"])
    split = yaml.safe_load(open(repo_root / dir_entry["dataset_split_file"]))
    out["hyperparameters"]["trainer_indices_list"] = (
        split["trainer_data_splits"][f"trainer_{trainer_id:03d}"]
    )
    return out
```

Then byte-compare:

```python
def verify_file(original_path: Path, reconstructed: dict) -> tuple[bool, str]:
    """Compare reconstructed against the on-disk original byte-for-byte."""
    original_bytes = original_path.read_bytes()
    # Match the source files' formatting: indent=4, sort_keys=False, no
    # trailing newline trimming. Confirm formatting by inspecting one source
    # file before committing the verifier (json.dumps default differs).
    reconstructed_bytes = (json.dumps(reconstructed, indent=4) + "\n").encode()
    if original_bytes == reconstructed_bytes:
        return True, ""
    return False, _diff(original_bytes, reconstructed_bytes)
```

**Critical pre-step**: open one of the source files in binary mode and inspect
the exact serialization style (key order in source? trailing newline? indent
size? unicode escape handling?). The reconstruction must match exactly.

Run mode:
```bash
python -m scripts.verify_migration \
    --plan lib/python/examples/cifar10/_migration_plan.yaml \
    --report lib/python/examples/cifar10/_migration_report.json
```

The report lists every source file with `pass | fail` and a unified diff
fragment for any failure. The script exits non-zero on any failure.

---

## 6. Execution order (tomorrow)

1. Inspect the byte-level format of `lib/python/examples/cifar10/trainer/config1/trainer_1.json`:
   - Key order (is it stable? sorted? insertion order?)
   - Indent size (4 spaces? tabs?)
   - Trailing newline?
   - Unicode escapes (`ensure_ascii`)?
2. Update `verify_migration.py`'s reconstruction to match that exact formatting.
3. Run the migration:
   ```bash
   cd /home/dgarg39/flame
   python -m scripts.migrate_trainer_configs \
       --example lib/python/examples/cifar10 \
       --metadata-out lib/python/examples/_metadata \
       --plan-out lib/python/examples/cifar10/_migration_plan.yaml \
       --namespace sync \
       --dataset-name cifar10
   ```
4. Run the verifier; require **100% pass on all 150 files**.
5. Inspect a couple of files in `_metadata/dataset_splits/cifar10_sync_config*.yaml`
   manually as a sanity check.
6. If anything fails, fix the migration/verifier, **do not delete anything**.

---

## 7. Staging deletion (after verification passes)

Once `verify_migration.py` reports 100% pass:

```bash
mkdir -p lib/python/examples/cifar10/trainer/_pending_deletion
git mv lib/python/examples/cifar10/trainer/config{1..15} \
       lib/python/examples/cifar10/trainer/_pending_deletion/
```

Commit as:
> Migrate cifar10 trainer JSONs to shared metadata (staged for deletion)

User spot-checks `_pending_deletion/` against the new YAMLs. When confirmed,
a follow-up commit does `git rm -r _pending_deletion/`.

The root-level `config.json` and `config2.json` stay in place (not part of
the migration).

---

## 8. examples/README.md (parallel deliverable)

Independent of the migration; can be written first thing tomorrow as a warm-up.

Outline:
- **What this dir is**: federated learning examples sharing a launcher and metadata.
- **Layout**:
  - `_metadata/`: shared trainer registry, dataset splits, availability traces.
  - `<example>/aggregator/pytorch/main.py`, `<example>/trainer/pytorch/main.py`:
    entry points; accept `--config` (path) or `--config-json` (string).
  - `<example>/configs/trainer_base.yaml`: per-example trainer config template.
  - `<example>/experiments/configs/*.yaml`: experiment descriptors consumed by
    the launcher.
- **Running an experiment**:
  ```bash
  python -m flame.launch.run_experiment \
      lib/python/examples/feddance_cifar10/experiments/configs/smoke_10trainer.yaml
  ```
- **What the launcher does**: parses the experiment YAML → builds per-trainer
  JSON in memory via `ConfigGenerator` → spawns aggregator + N trainers, each
  via `--config-json`.
- **Adding a new example**: 4-step recipe (create `aggregator/pytorch/main.py`
  using `load_config_from_argv`, same for trainer, add `configs/trainer_base.yaml`,
  add an experiment YAML).
- **Pointer to the metadata schema**: `_metadata/trainer_registry*.yaml` and
  `_metadata/dataset_splits/*.yaml` formats.

---

## 9. Open questions to keep in mind

- The cifar10 `trainer_indices_list` arrays in `configN/` dirs have *varying
  lengths* across trainers within the same dir (e.g. config10: lengths 4803,
  4841, ..., 5236). Each `configN` represents a different Dirichlet sampling
  draw. Naming convention `cifar10_sync_configN.yaml` is fine for now but
  doesn't encode the Dirichlet α — we don't know it from the JSONs. Leave as
  `configN` opaque labels; if α is later identified from notebooks or git
  history, rename then.
- After migration, the spawner reads dataset splits via
  `MetadataLoader.get_dataset_split(alpha, trainer_id, dataset_name, num_trainers)`
  which builds key `f"{dataset_name}_alpha{alpha}_n{num_trainers}"`. The
  migrated files use the schema `{dataset_name}_{namespace}_{configN}` —
  doesn't match the spawner's key-build. **Tomorrow's decision**: either
  rename migrated splits to the alpha-encoded schema (need α), or extend
  `MetadataLoader` to accept an explicit `split_key` parameter. The verifier
  doesn't go through `MetadataLoader`, so this is only a concern when the
  user later wants to *run* a legacy config as a new experiment.

---

## Resume command for tomorrow

```bash
cd /home/dgarg39/flame
git status
# Should show: lib/python/scripts/migrate_trainer_configs.py (untracked)
#              migration.md (untracked)
cat migration.md  # this document
```
