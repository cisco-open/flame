# fwdllm legacy-code deletion candidates (separate PR, after `launcher-script-fwdllm` merges)

This is the durable tracking doc for what to delete once the launcher
migration PR merges. `PR_CLEANUP_PLAN.md` (the working notes for that PR)
has already been removed from the branch — its still-relevant content is
folded into `../MIGRATING_TO_LAUNCHER.md` (see `MIGRATION_TO_LAUNCHER_FWDLLM.md`
for the current status pointer). This file is the one thing from that
cleanup that survives, so keep it up to date and refer back to it when
opening the deletion PR.

Do NOT delete anything below in the `launcher-script-fwdllm` PR itself — do
it as a follow-up PR once that one is merged and the launcher path has had a
chance to prove itself in real use.

## Confirmed safe to delete

Legacy MPI/JSON launch path, superseded by `flame.launch` + the YAML
experiments in `expt_scripts/`. Nothing in the new path imports any of these.

- `expts/run_tc_expts/json_scripts/` (155 files: 5 legacy aggregator `*.json`
  variants + `trainer_0.json..trainer_149.json`). Includes
  `aggregator_async_base.json`, `aggregator_async_dynk.json`,
  `aggregator_async_maxiter.json` — confirmed no 5th baseline is needed for
  these; they're superseded exploratory variants, not preserved
  functionality (only `aggregator.json` → `fluxtune_dynkc` and
  `aggregator_dynamic_kc.json` → `fluxtune` were carried forward).
- `expts/run_tc_expts/run_three_experiments.sh`
- `expts/run_tc_expts/run_three_parallel.sh`
- `expts/run_tc_expts/run_text_classification.sh`
- `expts/run_tc_expts/launch_single_run.py`
- `expts/run_tc_expts/gpu_mapping.yaml`
- `expts/run_tc_expts/mpi_host_file`
- `expts/run_tc_expts/fedavg_main_tc.py` (not imported by anything else,
  unlike its sibling `initializer.py` below)
- `aggregator/fl_main.py`, `trainer/fl_main.py` — confirmed dead via a
  repo-wide grep pass: the only references left are inside the
  already-doomed `expts/run_tc_expts/` scripts above (`run_three_*.sh`,
  `run_text_classification.sh`, `launch_single_run.py`) plus one comment in
  `main_fedfwd_agg.py`. Both files use the old argparse+MPI-era
  `Config(args.config)` wiring; the live entrypoints
  (`aggregator/main_fedfwd_agg.py`, `trainer/main.py`) use
  `flame.launch.cli.load_config_from_argv()` instead and already import
  every FL class `fl_main.py` used. Same origin commit
  (`d29a2f7f`, 2025-01-31), last touched together in `8449fac5`
  (2026-05-06) — kept in lockstep by copy/paste until the launcher
  migration updated only `main_fedfwd_agg.py`/`main.py`'s `__main__` block.

## Do NOT delete — confirmed still-live dependencies of the new path

- `expts/initializer.py` — imported directly by `aggregator/main_fedfwd_agg.py`
  and `trainer/main.py` (the new launcher entrypoints), not just the legacy
  `fedavg_main_tc.py`/`fl_main.py`.

## Not code, no action needed

`expts/run_tc_expts/cache_dir/` is already gitignored (253M local data
cache, untracked).
