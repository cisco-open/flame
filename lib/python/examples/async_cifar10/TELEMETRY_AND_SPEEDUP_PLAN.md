# Plan: Trainer/Aggregator Telemetry + Virtual-Clock Speedup (async_cifar10)

## Context

Two related needs in the `async_cifar10` FL example. **Order of work: Task 1 (telemetry) first, then Task 2 (speedup) later.**

1. **Telemetry & visualization.** Today we only see loss/accuracy/round-time. We can't see *how the selector chooses* (utility vs. speed tradeoff, what it deems available/eligible, how it uses current resources) or *how the aggregator aggregates* (staleness, agg-goal progress, participation). The existing `scripts/plotters/` is a manual, multi-step, hardcoded-path pipeline tuned for an old "fwdllm" log schema; it scrapes regexes out of free-text logs and is not run automatically post-run. We want robust, structured telemetry that supports apples-to-apples comparison **across selector implementations** and auto-generated plots after a run.

2. **Speedup that actually speeds up.** `speedup_factor=2` did not yield ~2x wall-clock. Root cause: slow-trainer dynamics are simulated with a literal `time.sleep()` after the GPU finishes, and several fixed waits/polls ignore `speedup_factor` entirely. We want to **keep the dynamics** (slow trainers delay their updates; selection still trades off utility vs. speed; staleness still accrues) while **eliminating real wall-clock spent sleeping**. The chosen approach is a **virtual (simulated) clock**: trainers do real GPU compute but report a *simulated completion time* instead of sleeping; the aggregator orders/commits updates by simulated time.

**Scope / reuse principle:** implement for `async_cifar10` first, but place all shared logic in the `flame` library base classes (aggregator / selector / trainer mixins) so other examples inherit it without duplication. The example only wires config + example-specific hooks.

---

# Task 1 — Structured telemetry + plots (do first)

### Structured emission (library layer, generic)
- New **`flame/telemetry/` module**: a lightweight `TelemetryWriter` that appends **JSONL event records** to a per-run file (one schema, typed events). Inject via base aggregator/selector/trainer so every example emits the same schema → cross-selector comparison is free, no regex scraping.
- **Event types to emit** (most state already exists; we structure it, not recompute):
  - *Selector*: per-`select()` — candidates, eligible set, availability composition (counts of `AVL_TRAIN`/`AVL_EVAL`/`UN_AVL` from `PROP_AVL_STATE`), per-trainer utility/speed used, chosen set, explore-vs-exploit split. Hook the base `AbstractSelector` ([selector/__init__.py](../../flame/selector/__init__.py)) + each concrete selector's decision point.
  - *Aggregator*: per-round loss/accuracy/round-time (already logged), plus staleness distribution, agg-goal progress, per-trainer participation, in-flight count (async).
  - *Trainer*: per-round real GPU time vs. `sim_round_duration`, wait time, availability state, samples visible (streaming). This is also the data that proves the Task 2 speedup works.
  - *Streaming utility disparity (new)*: with data streaming enabled, emit at fine resolution the trainer's **statistical utility on the currently-unlocked prefix** (the real, time-T value) **alongside a counterfactual utility computed over the full dataset** as if it were unlocked from the start, plus their ratio and the visible-sample fraction. Reuse the existing utility computation (Oort `I_m` / `reset_stat_utility` / `fetch_statistical_utility`) and the streaming machinery (`_visible_sample_count`, `_rebuild_stream_loader`, [trainer/pytorch/main.py:428](trainer/pytorch/main.py#L428)) — build a full-pool loader for the counterfactual forward pass. **Configurable** (`util_counterfactual: {mode, every_n_rounds, sample_size}`), **default subsample + every-N-rounds**, gated off otherwise, so the extra forward pass is opt-in. This surfaces the hypothesized gap between real-world streamed utility and the full-dataset assumption.
- Keep `wandb` as-is for live dashboards; JSONL is the source of truth for offline/paper plots.

### Plotting (fresh module, reuse helpers)
- New **`scripts/analysis/analyze_run.py`** (or `flame/telemetry/plots.py`): a single `analyze_run.py <run_dir>` that reads the JSONL and emits a PNG bundle + a small summary. **Reuse the existing matplotlib helpers** rather than rewriting: CDF/percentile rendering from `scripts/plotters/cdf_plot.py` (`generateCDF`), multi-run band/interpolation logic from `scripts/plotters/comparative_plotter.py`, and stacked-bar (train vs. stall) from `scripts/plotters/stall_stacked_bar_plot.py`.
- **Plots:** loss/accuracy/round-time over time; selector availability composition over rounds (stacked); utility-vs-speed scatter of selected vs. eligible; selection-frequency/fairness per trainer; staleness CDF; agg-goal/in-flight timeline; trainer real-GPU vs. simulated-time stacked bar (the speedup evidence); **streamed-vs-full utility disparity over time** (per-trainer real vs. counterfactual utility + ratio). A `--compare run_dirs...` mode overlays selectors.
- **Auto post-run:** invoke `analyze_run.py` from the example's launch/teardown so plots land in `<run_dir>/plots/` automatically.

---

# Task 2 — Virtual-clock speedup (do later)

## Implementation status (2026-05-28)

Implemented and unit-verified (156 tests green incl. `tests/sim` + `tests/mode`):
- `speedup_factor` removed everywhere; replaced by `time_mode: real|simulated` (default `simulated`) plumbed through experiment_config → spawner (`--time_mode`) → trainer, and into the aggregator config (`hyperparameters.time_mode`). YAML `speedup_factor` lines stripped from ~20 configs.
- Contract: `MessageType.SIM_SEND_TS / SIM_COMPLETION_TS / SIM_ROUND_DURATION`, `PROP_SIM_SEND_TS / PROP_SIM_COMPLETION_TS`, and pure `flame/sim/virtual_clock.py` (`VirtualClock`, `SimReorderBuffer`, `sim_ordered_ends`).
- Trainer: `real` sleeps `D`; `simulated` skips sleeps, reports `sim_completion_ts`/`D`, evaluates availability + streaming against sim-time (`_sim_now`), and skips (rather than busy-waits) when unavailable in sim mode.
- Async aggregator: stamps `sim_send_ts` on distribute; `simulated` commits via `_sim_recv_min` (reorder buffer → ascending `sim_completion_ts`, advances `T_v`) and sources `PROP_ROUND_DURATION` from `D`; `real` keeps arrival-order FIFO. Bounded-recv guard retained.

## Update (2026-06-01): unified timing model + GPU contention tracking

**Unified `max(gpu, D)` round duration** ([trainer/pytorch/main.py](trainer/pytorch/main.py), [asyncfl/top_aggregator.py](../../flame/mode/horizontal/asyncfl/top_aggregator.py)):
- Previously both modes used a fixed `D` for round duration. Now `sim_round_duration = max(gpu_time, D)` in all modes: no contention → `D` (correct device speed); GPU overrun → `gpu > D` (OORT correctly deprioritizes the contended trainer). `sim_completion_ts = sim_send_ts + max(gpu, D)` mirrors real `recv_ts − sent_ts = max(gpu, D)` exactly.
- Real mode sleep: `time.sleep(max(0, D − gpu))` — absorbs remaining budget. No sleep on overrun.
- Aggregator: `PROP_ROUND_DURATION = timedelta(SIM_ROUND_DURATION)` in sim mode; `recv_ts − sent_ts` in real mode. Both yield `max(gpu, D)`.

**Budget tracking + overrun detection** (`MessageType.TRAINING_BUDGET_S = 37`):
- Trainer sends `training_delay_s` (D) as `TRAINING_BUDGET_S` on every update (both modes).
- Aggregator: `[TIMING_OVERRUN_AGG]` warning when virtual elapsed ≤ 0 (sim) or wall lag > budget (real). Trainer: `[TIMING_OVERRUN]` warning when `gpu > D` with advice to reduce trainers-per-GPU or add GPUs.

**OORT crash fix** ([selector/async_oort.py](../../flame/selector/async_oort.py)):
- Root cause: `timedelta(0)` round duration → `system_utility = 0` for all trainers → `np.random.choice(replace=False, p=[...])` fails when all probabilities are zero. Fixed at the source (overrun now yields `gpu > D > 0`, never zero duration) plus a defensive zero-probability filter in `sample_by_util`.

**Compare-parity enhancements** ([scripts/compare_parity.py](scripts/compare_parity.py)):
- Check 9: GPU contention analysis — per-trainer overrun fraction vs budget; FAIL >25%, WARN >10%; graceful on old runs without `training_budget_s`.
- `--plot-out PATH`: 2-panel timing sanity plot — per-trainer mean GPU bar + budget marker (top), round-by-round deviation mean/max (bottom). Overrun zone highlighted in red.

Remaining / known gaps:
- **Sync aggregator**: init + `sim_send_ts` stamping done, but the `first_k`-by-`sim_completion_ts` selection and sim-sourced round duration are **not** implemented (sync still uses arrival order in sim mode). The primary target (felix) is async; sync (oort/refl/fedavg) sim-ordering is a follow-up.
- **End-to-end two-run verification** (Q1: `real` vs `simulated` per-round parity + speedup) needs a GPU/MQTT run — in progress (initial runs done; parity investigation ongoing).
- **Availability in sim mode** uses the trainer's own trace evaluated at the stamped sim-time; correlated-trace / oracular-at-`T_v` selection nuances (and the client-notify-vs-oracular question) want runtime validation.
- `examples/fwdllm` still hardcodes `speedup_factor=1.0` (separate example; left as-is).

## Why virtual-clock is the right call (correctness of reordering)

Concern: the async aggregator pops updates FIFO **by physical arrival** (`channel.recv_fifo`, [channel.py:430](../../flame/channel.py#L430)) — if we stop sleeping, all trainers finish ~together and arrival order no longer reflects simulated speed, corrupting staleness/ordering. Verified against both loops:

- **Async** ([asyncfl/top_aggregator.py:176](../../flame/mode/horizontal/asyncfl/top_aggregator.py#L176)): pops **one** update per iteration via `recv_fifo(..., 1)`; staleness = `agg_round − trainer_version`, so commit order *does* matter.
- **Sync** ([syncfl/top_aggregator.py:231](../../flame/mode/horizontal/syncfl/top_aggregator.py#L231)): waits for `first_k` then aggregates via a weighted average (`optimizer.do`), which is **order-independent**; only *which* k complete and the round duration matter.

**Decoupling constraint (important):** the aggregator must **not** hold a global profile of client runtimes — that is unrealistic FL. Instead it acts only on information **trainers report about themselves**. A trainer's simulated round duration *is* a local quantity (its own `training_delay_s`, eval delay, streaming reveal — all known to the trainer at start). So:

- **Upfront ETA announce:** when a trainer is selected and begins (real) compute, it immediately sends a cheap control message announcing `sim_completion_ts = sim_send_ts + its own modeled duration`. This is local, observation-based reporting (like Oort/FedScale clients reporting expected speed), not aggregator coupling. Because the announcement is sent at *start* and is tiny, it arrives well before the heavy weight update — so by the time any update lands, the aggregator already knows every in-flight trainer's reported ETA.
- **Async commit rule:** the aggregator orders in-flight ends by reported `sim_completion_ts`, picks the **smallest**, and does a *targeted* blocking `channel.recv(end_id)` ([channel.py:384](../../flame/channel.py#L384)) on exactly that end. It commits it and advances the virtual clock `T_v` to that ETA. Ordering is by reported sim time, **not physical arrival**, so GPU-contention jitter (different trainers sharing a GPU finishing out of sim-order) cannot mis-order updates.
- **Bounded, correct waiting:** the only wait is physical — blocking until the earliest-ETA update actually arrives. If a simulated-faster trainer is physically slow due to contention, the aggregator waits real compute time for it (unavoidable for correctness; this is the "wait a bit" — bounded by one trainer's real compute, **never** by simulated delays). A real-time budget on the targeted recv maps onto the existing `SEND_TIMEOUT_WAIT_S` path so a trainer that announces then goes unavailable doesn't block forever (drop/stale → advance).
- **Sync:** the k committed updates = the k selected trainers with smallest reported `sim_completion_ts`; round duration = the k-th smallest. Aggregation result is unchanged (order-independent); we stop sleeping and set round wall-time from virtual time.

This is more efficient than scaling sleeps: wall-clock is bounded by *real GPU time only*, not by the modeled delays, so large `training_delay_s` values cost nothing — and it keeps the aggregator decoupled from any client profile.

## Resolved design questions (from review)

**Two clean modes, one logical timeline, no `speedup_factor`.** There is a single logical simulated timeline driven by each trainer's modeled round duration `D` (= `training_delay_s` + eval). The two modes only differ in how they *realize* that timeline:
- **`real` (unsimulated)** — how training behaves in the real world. The trainer **sleeps** `D` so a slow device genuinely takes that long; wall-clock **is** the sim-timeline (1:1). Decisions use **actual arrival order** and the **real recv−send delta**. Authentic baseline.
- **`simulated`** — the same dynamics enacted faster: **no sleep**. The GPU does its real (fast) work, the trainer reports `sim_completion_ts = sim_send_ts + D`, and the aggregator advances a **virtual clock** `T_v`. Wall-clock = real GPU only.

`speedup_factor` is **removed entirely** from both modes (it never worked end-to-end and is unnecessary: `real` runs at true pace, `simulated` is bounded by GPU compute). Sleep exists **only** in `real` mode — it is not reintroduced anywhere else.

**Q1 — Two-run verification and the equivalence expectation.** Run the same experiment in `real` and `simulated` (fixed seed). Both read availability / data-visibility / speed / ordering off the *same logical times*, so they're expected to produce the **same per-(model-version) selection sets, staleness, and loss/accuracy** — "same per round/model-version, not per wall-clock-second" — with `simulated` wall-clock ≪ `real`. Equivalence is **exact in a deterministic scripted scenario** (well-separated `D`, controlled/mocked GPU time → arrival order == sim-completion order) and **within tolerance in a live smoke** (the only divergence source is real-GPU jitter reordering two trainers whose `D` are very close). Verify both: scripted parity test + smoke per-round parity & speedup.

**Q2 — Trainer speed / system-utility source.** In `simulated` mode the wall-clock recv−send delta collapses to ~GPU time and misrepresents speed, so `PROP_ROUND_DURATION` is sourced from `SIM_ROUND_DURATION = max(gpu, D)`. In `real` mode the real delta `recv_ts − sent_ts = max(gpu, D)` (sleep absorbs remaining budget). Both modes deliver the same quantity to OORT: correct device speed when no contention, true overrun cost when GPU is shared. (The current `recv−send` path is *already* wrong under the old `speedup_factor>1`; removing speedup + this fix resolve it.)

**Q3 — Dual timestamps (real + simulated) for lineage.** Keep both on every event record and in the aggregator's `_track_trainer_version_duration_s` (`real_ts` + `sim_ts`, send & recv), so lineage and wall-clock behavior stay reconstructable. Telemetry events gain `sim_ts`/`sim_completion_ts` alongside the existing real `ts`.

**Q4 — Availability & data-streaming timing without `speedup_factor`.** Both are expressed in **sim-seconds** and evaluated against the logical sim-time: in `real` mode that equals wall-clock-since-start (1:1, no scaling — replaces today's `real_elapsed × speedup_factor` at [main.py:263](trainer/pytorch/main.py#L263)/[main.py:434](trainer/pytorch/main.py#L434)); in `simulated` mode against the virtual clock stamped on each task (`sim_send_ts = T_v`), so the trainer needs no free-running clock. **Smoke-test impact (expected, and fine):** switch the telemetry smoke to `time_mode: simulated`, delete `speedup_factor`, and keep `full_data_available_after_s` / availability horizons in sim-seconds (e.g. a 600s = 10-min ramp in sim-time).

### Shared library layer (generic, reused by all examples)
- **A new `flame/sim/virtual_clock.py` mixin** holding `T_v`, `advance(ts)`, and a per-in-flight-end map of **trainer-reported** `sim_completion_ts`. Aggregators compose this in; examples inherit. The aggregator stores only what trainers announce — no client profile.
- **Message contract** in [flame/common/constants.py](../../flame/common/constants.py): (a) a lightweight **ETA-announce control message** (`MessageType.SIM_COMPLETION_TS`) the trainer sends at start of compute; (b) the same field echoed on the final weight update for validation. Backward-compatible: absence ⇒ real-time behavior (non-sim examples untouched).
- **New selector property** `PROP_SIM_COMPLETION_TS` in [selector/properties.py](../../flame/selector/properties.py), populated from the trainer's announcement; aggregator records `sim_send_ts` at distribute-time (reuse existing `PROP_ROUND_START_TIME` / `_track_trainer_version_duration_s`).

### Trainer changes ([trainer/pytorch/main.py](trainer/pytorch/main.py))
- Compute the modeled round duration `D` (`training_delay_s` + eval) locally. **`real` mode:** sleep `D` after GPU work ([main.py:571](trainer/pytorch/main.py#L571), [main.py:708](trainer/pytorch/main.py#L708)) — keep it. **`simulated` mode:** skip the sleep; instead **announce** `sim_completion_ts = sim_send_ts + D` (where `sim_send_ts` is the sim-time the aggregator stamped on the task) and echo `D`/`sim_completion_ts` on the outgoing update.
- Drop `speedup_factor` from the timing path entirely. Evaluate availability and data-visibility against the logical sim-time: wall-clock-since-start in `real` mode, the task's stamped sim-time in `simulated` mode (replaces `real_elapsed × speedup_factor` in `_visible_sample_count`/`check_and_update_state_avl`). In `simulated` mode the fixed real-time waits ([main.py:485](trainer/pytorch/main.py#L485), [main.py:668](trainer/pytorch/main.py#L668), polling/heartbeat threads, 20s fallback at [main.py:295](trainer/pytorch/main.py#L295)) must not gate progress.

### Aggregator changes
- **Async** ([asyncfl/top_aggregator.py](../../flame/mode/horizontal/asyncfl/top_aggregator.py)): in `simulated` mode, consume ETA-announce messages into an in-flight ETA map; replace the single `recv_fifo(...,1)` pop with "select in-flight end with min `sim_completion_ts` → targeted `channel.recv(end_id)` → commit → advance `T_v`"; set `PROP_ROUND_DURATION = sim_completion_ts − sim_send_ts` (Q2); stamp each distributed task with `sim_send_ts = T_v`. In `real` mode keep today's arrival-ordered FIFO recv and real recv−send delta. Both keep the `RECV_TIMEOUT_WAIT_S` guard. Staleness keys off commit order (virtual-clock order in `simulated`, arrival order in `real`).
- **Sync** ([syncfl/top_aggregator.py](../../flame/mode/horizontal/syncfl/top_aggregator.py)): `simulated` mode commits the `first_k` smallest-`sim_completion_ts` responders with round duration from virtual time; `real` mode unchanged. Aggregation math is order-independent (weighted average) either way.
- Gate behind `time_mode: real|simulated`. `real` is the authentic, unchanged-decision baseline; `simulated` reproduces it via the virtual clock and is expected to match per-round (Q1).

### `time_mode` plumbing (new config; replaces `speedup_factor`)
- **Config field:** add `time_mode: str = "simulated"` to `TrainerConfig` in [experiment_config.py](../../flame/launch/experiment_config.py) (default `simulated`; `real` is opt-in for verification). Load it in the same place `speedup_factor` was read (`load_experiment_config`, ~L190).
- **To the trainer:** the trainer reads `time_mode` from argv. Replace the trainer's `--speedup_factor` arg ([trainer/pytorch/main.py](trainer/pytorch/main.py), `main()`) with `--time_mode`; `TrainerSpawner` ([spawner.py](../../flame/launch/spawner.py)) passes `--time_mode <mode>` instead of `--speedup_factor` (drop that field from the spawner too). `runner.py` passes `time_mode=exp.trainer.time_mode`.
- **To the aggregator:** the aggregator also needs `time_mode` (virtual-clock vs arrival path). Thread it into the aggregator config `hyperparameters.time_mode` via the runner's aggregator `config_overrides` (same mechanism as `agg_goal`), read it in `asyncfl/syncfl top_aggregator.__init__`.
- **Snapshot/exec-config:** drop the `speedup_factor` keys from [snapshot.py:92](../../flame/launch/snapshot.py#L92) and [execution_config_generator.py:118](../../flame/launch/execution_config_generator.py#L118); record `time_mode` instead.

### `speedup_factor` removal inventory (delete all occurrences)
Code (remove field/arg/reads and the `/ speedup_factor` or `* speedup_factor` arithmetic — the timing now uses sim-time per the two modes):
- [experiment_config.py](../../flame/launch/experiment_config.py): `TrainerConfig.speedup_factor` field + the `.get("speedup_factor", ...)` read.
- [spawner.py](../../flame/launch/spawner.py): ctor param `speedup_factor`, `self.speedup_factor`, and the `--speedup_factor` cmd args.
- [runner.py](../../flame/launch/runner.py): `speedup_factor=exp.trainer.speedup_factor` kwarg.
- [snapshot.py](../../flame/launch/snapshot.py), [execution_config_generator.py](../../flame/launch/execution_config_generator.py): the recorded `speedup_factor` keys.
- [trainer/pytorch/main.py](trainer/pytorch/main.py): ctor param, `self.speedup_factor`, the `--speedup_factor` argparse arg, and every `/ speedup_factor` (avail event ts ~L283, training delay sleep ~L718, eval delay sleep ~L855) and `* speedup_factor` (`_visible_sample_count` sim_elapsed ~L461, util-disparity elapsed ~L712). Replace with the two-mode sim-time logic; remove the stale `TODO(DG): revisit speedup_factor coupling` note.
- YAMLs: delete the `speedup_factor: 1.0` lines across `expt_scripts_2026/*.yaml` and `experiments/configs/*.yaml` (≈25 files; a leftover line is harmless — the loader uses per-field `.get()` — but remove for cleanliness). Add `time_mode:` where a non-default is wanted.
- Out of scope but for consistency: `examples/fwdllm/.../FedSgdTrainer.py` hardcodes `speedup_factor=1.0` + an eval-delay divide; leave or clean separately (it's a different example, always 1.0, so behavior is unaffected).

### Smoke-test YAML update
[felix_n10_alpha100_syn20_telemetry_smoke.yaml](expt_scripts_2026/felix_n10_alpha100_syn20_telemetry_smoke.yaml): remove `speedup_factor: 60.0`; add `time_mode: simulated`; set `data_streaming.full_data_available_after_s` back to **sim-seconds** (e.g. `600` for a 10-min *sim-time* ramp). Availability stays `syn_20`. For the Q1 two-run verification, add a tiny companion run/YAML (small `D`, few rounds) so a `time_mode: real` run finishes quickly.

### Implementation order (suggested)
1. `flame/sim/virtual_clock.py` mixin + `MessageType.SIM_COMPLETION_TS` + `PROP_SIM_COMPLETION_TS`; add `time_mode` config plumbing; rip out `speedup_factor` (inventory above).
2. Trainer: modeled `D`, announce `sim_completion_ts` (simulated) / sleep `D` (real); sim-time availability + streaming; dual timestamps in telemetry.
3. Async aggregator: virtual-clock ordering + targeted recv + `PROP_ROUND_DURATION` from sim duration (simulated); keep arrival path (real). Then sync.
4. Tests: `tests/sim` (virtual clock), `tests/mode` (scripted real-vs-sim parity + arrival-order independence).
5. Smoke YAML update; run the two-run verification (Q1).

---

## Discovered during Task 1 testing (carry into Task 2)

Running the telemetry smoke (`felix_n10_alpha100_syn20_telemetry_smoke.yaml`, async_oort/fedbuff) surfaced several real bugs. Some were fixed inline to unblock telemetry validation; the deeper ones are flagged for Task 2.

**Already fixed (keep, but revisit under the virtual clock):**
- **`speedup_factor` never reached the trainer.** The launcher built the trainer command without `--speedup_factor`/`--battery_threshold`, so the trainer always ran at `1.0` regardless of YAML — almost certainly why "speedup_factor=2 gave no 2× speedup". A stop-gap wiring fix was committed (spawner passes `--speedup_factor`), but **Task 2 supersedes it entirely: `speedup_factor` is removed in favor of the `real`/`simulated` two-mode design** (see "Resolved design questions" + the removal inventory). The `--battery_threshold` wiring stays.
- **Aggregator hung forever on a quiet in-flight trainer.** `recv_fifo` had no timeout; when every in-flight end went silent (all unavailable, or a stale ghost) the async aggregator blocked indefinitely. Band-aided with `recv_fifo(timeout=...)` + `RECV_TIMEOUT_WAIT_S=30` in asyncfl `_aggregate_weights`, plus a ghost filter (`channel.has`) and `selected_ends` cleanup in `async_oort._cleanup_removed_ends`. **The virtual-clock redesign should replace this band-aid** with the targeted, ETA-ordered receive (which has principled per-end completion times and timeouts).
- UTF-8 stdio in the launcher (latin-1 locales crashed on status glyphs) and an async `channel.ends()==None` guard.

**Open issues for Task 2:**
- **In-flight accounting leaks (`freed=0`).** Telemetry showed a persistent `in_flight=1` ghost every round: an end gets selected, never returns an update, and is never freed from `selected_ends` (channel cleanup logged `in_flight_after=1, freed=0` each round). The proper fix is explicit in-flight↔availability reconciliation: on a client-notify `AVL_TRAIN→UN_AVL` transition (felix) drop the end from in-flight immediately; for non-notify baselines (fedbuff) a real timeout must drop it — and a returning trainer's late update should be explicitly accepted-or-discarded by version/staleness checks. This is core to the async correctness work.
- **Correlated availability is unrealistic.** All trainers share one `syn_20` "pattern" trace, so they go UN_AVL/AVL in lockstep (the whole system stalls together when the pattern is down). Decorrelate with per-trainer phase offsets (or per-trainer traces) so availability is independent — this also removes the global-stall failure mode at low speedup.
- **Model not learning in the smoke (accuracy flat at ~0.10 = random, loss pinned at ln 10).** Even after data fully unlocked, 100 async rounds produced no learning. Likely a hyperparameter/optimizer issue (client LR `0.001`, fedbuff server LR/`use_oort_lr`, delta-weight scaling) compounded by a very small per-trainer partition (~164 samples in this split). Needs a convergence sanity pass (validate felix actually learns on a known-good config) before drawing conclusions from streamed-vs-full utility plots. Telemetry is correct — it is what surfaced this.
- **`np.str_` end-ids leak into participation/selection dicts** (from `np.random.choice` in oort sampling). Harmless today (subclass of `str`) but a smell; normalize to `str` to avoid subtle set/dict-key surprises.

---

## Tests (enhance pytest; cover simulated + real modes)

Existing selector tests live in `lib/python/tests/selector/` with `lib/python/tests/conftest.py`. Add:
- **`tests/telemetry/test_event_schema.py`** (Task 1): schema/round-trip + that each selector emits the required event fields (parametrized over selectors for cross-comparability). Include a streaming-utility case: with a known prefix vs. full pool, assert the counterfactual utility ≥/≠ the streamed utility as expected and that disparity events are emitted at the configured cadence.
- **`tests/sim/test_virtual_clock.py`** (Task 2): virtual-clock advance/ordering unit tests; assert min-`sim_completion_ts` selection reproduces a hand-computed staleness sequence.
- **`tests/mode/test_async_aggregation_ordering.py`** (Task 2): feed a scripted set of in-flight ends with known sim durations + shuffled physical arrival; assert commit order and per-update staleness are **identical** to a reference computed by virtual time, and **independent of arrival order** (the jitter guarantee).
- **`tests/mode/test_sync_aggregation_equivalence.py`** (Task 2): assert sync aggregate output is identical in `simulated` vs. `real` mode (order-independence guard / regression).
- **Mode parity / regression** (Task 2): parametrize a tiny end-to-end run over `time_mode ∈ {simulated, real}`; assert correctness parity (final weights/accuracy within tolerance) and that `simulated` wall-clock < `real` (performance guard).

---

## Critical files

| Area | File |
|---|---|
| Async agg loop | `lib/python/flame/mode/horizontal/asyncfl/top_aggregator.py` |
| Sync agg loop | `lib/python/flame/mode/horizontal/syncfl/top_aggregator.py` |
| Channel recv | `lib/python/flame/channel.py` (`recv` L384, `recv_fifo` L430) |
| Selector base/props | `lib/python/flame/selector/__init__.py`, `lib/python/flame/selector/properties.py` |
| Trainer | `lib/python/examples/async_cifar10/trainer/pytorch/main.py` |
| Msg types | `lib/python/flame/common/constants.py` |
| New: telemetry (Task 1) | `lib/python/flame/telemetry/` |
| New: analysis (Task 1) | `scripts/analysis/analyze_run.py` (reuses `scripts/plotters/` helpers) |
| New: virtual clock (Task 2) | `lib/python/flame/sim/virtual_clock.py` |

---

## Verification

**Task 1**
1. Run produces `<run_dir>/events.jsonl` and `<run_dir>/plots/*.png` automatically; eyeball selector availability-composition and streamed-vs-full utility-disparity plots.
2. `pytest lib/python/tests/telemetry` green.
3. Cross-selector compare: run two selectors (e.g. async_oort vs. feddance), `analyze_run.py --compare` overlays utility/speed/staleness/accuracy.

**Task 2** (the two-run equivalence is the core check — see Q1)
4. `pytest lib/python/tests/sim lib/python/tests/mode` green: scripted deterministic scenario (fixed durations + availability + seed) yields **identical** commit order, staleness, selection sets, and resulting weights in `real` vs. `simulated` mode, and independent of physical arrival order.
5. **Two smoke runs, same seed** — one `real`, one `simulated`: assert **per-(model-version) selection sets and loss/accuracy match** (within RNG tolerance), and `simulated` wall-clock ≪ `real`. This is the "same selection and learning over time (per round, not per second)" verification. Use a **short / small-`D` / few-round** scenario so the `real` run (true pace, no speedup) finishes in reasonable wall-clock.

---

## Parity smoke results (2026-05-30, pre-pending-commit fix)

Runs: `run_20260530_130644_felix_n10_parity_REAL` vs `run_20260530_171846_felix_n10_parity_SIMULATED`
Config: `felix_n10_parity_real_vs_sim.yaml` — 10 trainers, `syn_0` availability, 100 rounds, `agg_goal=5`, `c=8`.

| Check | Result | Notes |
|---|---|---|
| 1. Selection parity (Jaccard) | **WARN** — 0% exact, mean J=0.477 | Unseeded OORT RNG + participation skew |
| 2. Statistical utility (KS) | **FAIL** — max KS=0.806 | Slow trainers (D≥16s) have n_sim≤2 vs n_real≥32 |
| 3. Aggregation sequence | **FAIL** — 1% exact | Directly downstream of participation skew |
| 4. Staleness distribution | **WARN** — real mean=0.99, sim mean=2.14, diff=1.15 | Sim over-accumulates staleness |
| 5. Participation counts | **FAIL** — avg diff=31, max=47 | Slow trainers starved in sim; fast trainers dominate |
| 6. Convergence (acc/loss) | **PASS** — avg acc diff=0.040 | Both curves still flat (~random) at 100 rounds |

**Root cause:** before the `_sim_pending_commit` fix, slow trainers (D=16–18s) were released back into selection at round end even though their buffer entry was uncommitted. OORT's system-utility then penalised them (high staleness → poor speed score), causing fast trainers (D=4–13s) to monopolise selection. Trainers 0371/0374/0377 (D=16/18/17s) appeared only 2/2/1 times in sim vs 36/32/32 in real.

**Fix applied:** `_sim_pending_commit` — a trainer with an uncommitted `_sim_buffer` entry stays in `all_selected` (invisible to the selector) until `pop_min` commits that entry in `_sim_recv_min`. This enforces the real-mode invariant (one in-flight update per trainer at a time) in simulation.

**Expected improvement in next run:** staleness mean should drop to ~1.0 (matching real), participation skew should narrow, and Jaccard should improve toward ≥0.7. Selection will not be bit-for-bit identical because OORT's RNG is unseeded; statistical distributions across trainers should match.
6. Regression: `real`-mode decisions unchanged vs. pre-Task-2 baseline (same seed). Note a convergence prerequisite from Task 1 findings — confirm felix actually learns on a known-good config before reading equivalence into accuracy curves.
