# Telemetry Plotting Overhaul — Plan

Companion to [PARITY.md](PARITY.md). PARITY.md is about *correctness* of the sim
(does it match real?); this doc is about *legibility and speed* of the post-run
analysis (`scripts/analysis/analyze_run.py` + `plot_helpers.py`).

Goal in one line: fewer, denser, more decision-relevant figures; one pass over the
2.2 GB aggregator log instead of six; and three new thematic deep-dives —
**availability**, **selection (why)**, **aggregation (rate / staleness / cadence)**.

Run: `python ../../../scripts/analysis/analyze_run.py <run>/telemetry`.
Existing outputs live in `experiments/run_*/plots/{performance,sanity,selection,insights,system}/`.

---

## §0 Performance — make `analyze_run.py` fast

This is the "also faster" half of the request and the highest-leverage change.

### 0.1 Single-pass aggregator-log parser  *(biggest win)*
The aggregator log for a felix run is **2.2 GB / 9.7 M lines**. Today it is read
**six independent times**, each a full scan:

| pass | function | file:line |
|---|---|---|
| 1 | `parse_send_recv_lags` (`[SEND_RECV_LAG]`, `[TIMING_OVERRUN_AGG]`) | [analyze_run.py:258](../../../scripts/analysis/analyze_run.py#L258) |
| 2 | `_parse_lag_decomp` (`[LAG_DECOMP]`) | [analyze_run.py:298](../../../scripts/analysis/analyze_run.py#L298) |
| 3 | `_parse_overrun_excesses` | [analyze_run.py:328](../../../scripts/analysis/analyze_run.py#L328) |
| 4 | `_parse_sim_barrier` (`[SIM_BARRIER]`) | [analyze_run.py:347](../../../scripts/analysis/analyze_run.py#L347) |
| 5 | `_parse_agg_commit_timing` (`[AGG_COMMIT_TIMING]`) | [analyze_run.py:380](../../../scripts/analysis/analyze_run.py#L380) |
| 6 | `_SEND_RE` dispatch parse + per-round lag re-parse | [analyze_run.py:1411](../../../scripts/analysis/analyze_run.py#L1411), [:1734](../../../scripts/analysis/analyze_run.py#L1734) |

**Action:** one `parse_agg_log(tdir)` that compiles all the `[TAG]` regexes once,
loops the file a single time, dispatches per line, and returns a dataclass/dict of
every series (`wall_lags`, `lag_decomp{...}`, `overrun_excesses`, `sim_barrier`,
`agg_commit`, `sends_by_round`, `rd_lags`). Memoize per `tdir`. Expected ~6× less
IO on the dominant cost. All six existing parsers become thin accessors so nothing
downstream changes.

Cheap pre-filter: `if "[" not in line: continue` before regex (most lines have no
tag). Even better, gate on a single substring set check.

### 0.2 Stream the JSONL instead of materializing all of it
`load_events` ([:74](../../../scripts/analysis/analyze_run.py#L74)) builds one list
of **every** record across 300 trainer files + the aggregator JSONL. The selection
stream alone is 149 754 events each carrying a `per_trainer` dict of up to 300
entries — hundreds of MB of Python objects held for the whole run.

**Action:**
- Split loading by role/event. Most extractors only need one event type; give them
  an iterator (`iter_events(tdir, event=...)`) so we never hold all selection
  `per_trainer` blobs at once.
- For the per-round aggregates we recompute repeatedly (e.g. `trainer_rounds_by_round`,
  `accuracy_by_round`), compute once into small dicts and pass them down.
- `per_trainer` is the memory hog. When only `chosen`/counts are needed, skip
  decoding it (parse with a streaming filter, or have the emitter write a slim
  `selection_lite` alongside — see §5).

### 0.3 Stop re-loading checkpoints with torch every run
`_weight_change_norm` ([:521](../../../scripts/analysis/analyze_run.py#L521)) `torch.load`s
**every** `round_*.pt`. For long runs that is hundreds of GPU-snapshot loads on the
CPU path. **Action:** cache the computed `(round, l2)` series to
`analysis/weight_change_norm.csv`; recompute only for checkpoints newer than the
cache. Skip entirely unless `--weights` is passed.

### 0.4 Down-sample before plotting, not in matplotlib
Several series have one point per round over >4000 rounds (lag-over-rounds,
staleness-over-rounds, queue-depth). Rendering 4000+ vertices per line as PDF is
slow and unreadable. Adopt a shared `binned_line` helper (§1) that reduces to
~200 x-bins with a chosen reducer (mean / p50 / p99) *before* handing to mpl.

### 0.5 Parallelize the five plot groups
`analyze()` ([:1824](../../../scripts/analysis/analyze_run.py#L1824)) runs the groups
serially. They are independent and mostly IO/CPU bound → run under a
`ProcessPoolExecutor` (matplotlib Agg is fork-safe). Lower priority than 0.1–0.2.

### 0.6 Target
A felix sim run analysis should drop from "minutes" to well under a minute, with
the log read exactly once. Add a `--quick` flag that skips checkpoint norms and
caps per-trainer figures, for use inside the parity loop.

---

## §1 Shared helper additions (`plot_helpers.py`)

Most asks below reduce to two missing primitives. Add them once:

- **`binned_line(x, y, nbins=200, reducer="mean"|"p50"|"p99", band=None)`** — bucket
  x into ~200 ranges, reduce y per bucket, optional min/max or IQR band. This is the
  single fix for *every* "too many scatter dots" / "too noisy line" / "render is
  slow" complaint (insights scatters, `runtime_agg_vs_trainer`, queue-depth,
  send-recv-lag, vclock advance). Replaces ad-hoc `_smooth` ([:958](../../../scripts/analysis/analyze_run.py#L958)).
- **`cdf_plot`/`cdf_multi`** already exist ([:384](../../../scripts/analysis/analyze_run.py#L384)) —
  reuse them everywhere a "distribution over rounds is unreadable" complaint appears
  instead of inventing new chart types.

Everything else is a call-site change, not a new helper.

---

## §2 Per-category fixes (addressing every item)

### insights/
- **`loss_vs_visible`, `update_norm_vs_visible`, `utility_vs_visible`**
  ([:1245](../../../scripts/analysis/analyze_run.py#L1245)): currently `scatter_plot`
  over every trainer-round. → `binned_line` over the x (visible fraction), reducer
  `mean` with a p10–p90 band. Dense scatter becomes one readable trend line per plot.
- **disparity plot** (`util_disparity_ratio`, [:1260](../../../scripts/analysis/analyze_run.py#L1260)):
  *What it means* — `utility_streamed / utility_full`: under streaming a trainer only
  sees a prefix of its data, so its **believed** statistical utility (computed on the
  visible prefix) can differ from the utility on the full set. Ratio 1.0 = streaming
  hides nothing; <1 = the selector under-values trainers whose data hasn't unlocked
  yet. **Make it actionable**: it is currently a free-floating ratio. Connect it to
  selection by overlaying, on the same round axis, the *mis-selection rate* (already
  computed in `insights`) — i.e. "when disparity is high, do we mis-select?". Add a
  second panel: scatter/binned of per-trainer `utility_ratio` vs that trainer's
  selection frequency, to show whether disparity actually changed who got picked.

### performance/
- **`global_weight_change_norm`** ([:548](../../../scripts/analysis/analyze_run.py#L548)):
  *Meaning* — `||w_r − w_{r-1}||₂`, the L2 step the global model takes per checkpoint.
  *Expected shape*: **large early, decaying toward zero** as the model converges; a
  flat or rising tail signals non-convergence / churn / too-high effective LR or high
  staleness re-introducing old gradients. **Make it informative**: (a) keep log-y;
  (b) overlay test-accuracy on a secondary axis (`dual_axis_line`) so the reader sees
  "steps shrink as accuracy plateaus"; (c) annotate the round where the step norm
  crosses below, say, 10% of its peak (a convergence marker). This turns a bare curve
  into a convergence diagnostic.

### sanity/
- **`runtime_agg_vs_trainer`** ([:679](../../../scripts/analysis/analyze_run.py#L679)):
  scatter of every (trainer-reported, agg-observed) pair. → bin by trainer-reported x
  (~50 bins) and plot a **single p99 marker per bin** (the user's ask), plus the y=x
  diagonal. Keep the overhead CDF that already follows it. One dot per x-bin instead
  of tens of thousands.
- **`selection_count_consistency`** ([:764](../../../scripts/analysis/analyze_run.py#L764)):
  chosen vs contributing per round, very noisy over 4000 rounds. → `binned_line`
  (mean per ~200 round-bins) for both series; drop `clip_outliers` hack. Add a faint
  raw line at low alpha if detail is wanted.

### selection/
- **`availability_composition`** ([:1058](../../../scripts/analysis/analyze_run.py#L1058)):
  today a `stacked_area`; under syn_0 it is one solid block (every trainer is
  `AVL_TRAIN` always — confirmed: only states seen are `AVL_TRAIN` + a little
  `UNKNOWN`). → switch to a **line per state** (`line_plot` with one series per
  availability state). For syn_0 this is honestly one flat line at n=300; the plot
  should say so rather than render a blue wall. The interesting version only appears
  under a dynamic availability trace (non-syn_0) — see §3 Availability.
- **`eval_vs_train_selections`** ([:930](../../../scripts/analysis/analyze_run.py#L930)):
  *Why it looked empty* — felix **does** select eval (confirmed: 1625 rounds with 10
  eval-chosen, 459 with 8, …) but **70 682** eval-selection events have `chosen=[]`,
  and eval rounds are sparse relative to 4230 train rounds, so on a full-width stacked
  area the eval band is a few invisible spikes. → (a) plot **eval as its own
  smoothed rate line** (eval selections per 50-round bin) on a second axis, not
  stacked under train; (b) annotate eval cadence (mean rounds between eval bursts).
  This makes the eval selector visibly active.
- **`exploration_factor`** ([:919](../../../scripts/analysis/analyze_run.py#L919)):
  decays to ~0 within tens/hundreds of rounds but x spans >1k. → (a) **clip x to the
  round where exploration first hits ~0** (plus a small margin); (b) plot **two
  lines — explore fraction and exploit fraction (= 1 − explore)** of each round's
  picks. (c) **Drop `selection_coverage`** ([:887](../../../scripts/analysis/analyze_run.py#L887))
  if it adds nothing beyond the Lorenz curve — keep `selection_fairness_lorenz` as the
  single coverage/fairness figure (but see fairness fix below).
- **Availability histogram → "what drives selection" group** (the macro ask): the
  goal is *visually see which factor correlates with being picked*. With the rich
  `per_trainer` keys (`believed_I`, `system_util`, `temporal`, `speed_s`,
  `last_train_round`, `last_eval_round`) we can build a **selection-attribution
  panel** (see §3 Selection-why). Replace the bare availability histogram with: a
  small-multiples of "picked vs not-picked" distributions for each factor, so the
  separation (or lack of it) is the visual.
- **`selected_speed_utility_over_rounds`** ([:942](../../../scripts/analysis/analyze_run.py#L942)):
  *Macro data* = "what the selector believed about its picks each round." Make it
  readable: keep the smoothed dual line but add the **believed-vs-actual sim-delay
  CDF** as the headline (the gap the user cares about). *Why believed < actual for
  refl but the reverse in felix*: `believed_I` is the utility the selector recorded
  at pick time; `actual` is the trainer's realized `stat_utility` that round. The two
  diverge for different reasons per baseline:
  - **refl** has *no eval selector*, so belief is staler/coarser; here belief
    *under-estimates* (believed < actual) because refl's utility proxy is conservative
    and it happens to pick trainers that realize higher utility than predicted.
  - **felix** *does* run an eval selector intended to tighten belief, but the eval
    utility feeds back with lag/staleness, so during fast async churn belief
    *over-estimates* (believed > actual). The eval selector improves *ranking*
    correlation (`Im_staleness_rankcorr`) more than it fixes the *level*.
  **Is it logged correctly?** Partly suspect: `per_trainer.utility`/`speed_s` are
  often `null` (confirmed in samples) — the populated field is `believed_I`. The
  current code falls back `believed_I → utility`, so when `believed_I` is absent it
  silently plots `None`-gappy series. **Action:** (a) assert/measure `believed_I`
  coverage and surface it in the title (`n=… of … picks had a believed value`);
  (b) plot believed and actual as **paired CDFs of sim-delay** (the user's
  suggestion) so the level gap and its sign are unambiguous per baseline; (c) add a
  signed `believed − actual` histogram with the mean annotated, so "refl negative,
  felix positive" is a one-glance fact.
- **Per-client utility over time (aggregate across 300 trainers)** (the macro ask):
  per-trainer lines are hopeless at n=300. Proposed group (see §3 Selection-why):
  - a **utility heatmap** (trainer × round-bin, color = mean believed utility),
    trainers sorted by mean utility — shows drift and stratification at a glance;
  - **utility-percentile bands over rounds** (p10/p50/p90 of the believed-utility
    distribution per round-bin) — one figure summarizing 300 trajectories;
  - to relate utility→selection→training: overlay, per round-bin, the **mean utility
    of *picked* vs *pool*** (already have `selected_vs_pool_utility`, [:851](../../../scripts/analysis/analyze_run.py#L851))
    against accuracy gain — closing the loop the user wants. For cross-baseline
    comparison, emit these as the *same* binned figure so felix/refl/oort/feddance
    overlay cleanly.
- **Fairness must include eval selections** (explicit correction): the Lorenz/Gini
  fairness ([:895](../../../scripts/analysis/analyze_run.py#L895)) counts only
  `chosen` from *train* selections via `freq`. **Action:** build `freq` from **both
  train and eval** selection events (eval participation is still participation), and
  label the figure "train+eval selection fairness." Optionally show two Lorenz curves
  (train-only vs train+eval) so the eval contribution to fairness is visible.
- **`trainer_state_fraction_per_trainer`** ([:1187](../../../scripts/analysis/analyze_run.py#L1187)):
  *Rename* — "state" is misleading (state changes constantly); call it
  **`trainer_time_allocation_per_trainer`** ("fraction of rounds spent in each
  activity"). Categories the user wants, made explicit and colored:
  `idle_train` (available-for-train, not picked), `idle_eval` (available-for-eval,
  not picked), `train`, `eval`, `unavail`. Today the code only has
  `train/eval/idle/unavailable` ([:1153](../../../scripts/analysis/analyze_run.py#L1153)) —
  split `idle` into `idle_train`/`idle_eval` using the trainer's availability state.
  Annotate each segment's % only on the aggregate bar (keep per-trainer bar
  text-free to avoid crowding, per the user).

### system/
- **`comm_message_accounting`, `comm_per_round_train_vs_eval`** ([:1567](../../../scripts/analysis/analyze_run.py#L1567)):
  hard to read as per-round stacked areas over 4000 rounds. → bin to ~200 round-bins
  (sum per bin) and/or switch the message-accounting one to **cumulative** lines
  (cumulative sent / returned / discarded) — monotone curves read far better than a
  noisy per-round area, and the discarded gap is the signal.
- **`compute_time_by_task_cdf`** ([:1664](../../../scripts/analysis/analyze_run.py#L1664)):
  *Why no eval* — it groups by `task_to_perform`, but eval trainer-rounds either
  don't emit `real_gpu_time_s` or aren't tagged `eval` in `trainer_round` (eval runs
  off the critical path in a daemon thread per PARITY §3a). **Action:** confirm eval
  forward-passes emit a `trainer_round` (or a dedicated `eval_round`) with
  `real_gpu_time_s` + `task_to_perform="eval"`; if not, add it (§5). Then this CDF
  gets its eval series. Additionally produce **separate per-task histograms with both
  a count axis and an MB axis** (train up+down = 2×model, eval down = 1×model) so
  compute and communication per task sit side by side.
- **`mqtt_delivery_accounting`** ([:1433](../../../scripts/analysis/analyze_run.py#L1433)):
  missed messages are invisible on cumulative dispatched-vs-received lines (the gap
  ≈ in-flight is the healthy baseline). **Action:** add a **drops callout**: compute
  `final_gap − steady_state_inflight`; if >0, render a red annotation + a small
  histogram of per-round `dispatched − received` deltas (a spike = a drop event). The
  plot should *shout* when messages are actually lost, not require eyeballing two
  near-identical curves.
- **`queue_depth_over_rounds`** ([:1690](../../../scripts/analysis/analyze_run.py#L1690)):
  raw per-round line is jagged. → `binned_line` mean+p99 band per round-bin, **plus a
  queue-depth CDF** (what fraction of rounds had queue ≥ k). The CDF answers "is the
  buffer usually shallow with rare spikes?" better than the time series.
- **`send_recv_lag_over_rounds` is getting worse over time** ([:1720](../../../scripts/analysis/analyze_run.py#L1720)):
  *Likely root cause* — this is the **same buffer-backup mechanism PARITY §3 traced
  for staleness**. `wall_lag` per update is dominated by `queue_wait_s` (MQTT-arrival
  → aggregator-dequeue). In sim, all in-flight updates drain into `_sim_buffer` in one
  barrier but are popped one-per-commit over K rounds; an update with staleness S has
  `queue_wait_s ≈ S × wall_per_round`. As the run proceeds and the reorder buffer
  backs up (overhead vs sct-frontier mismatch, PARITY §3b), staleness — and therefore
  `queue_wait_s`, and therefore `wall_lag` — **grows monotonically**. So the upward
  drift is a *symptom of the overhead mis-tune*, not an independent bug; it should
  flatten once felix overhead=0.315 lands (PARITY §4/#1). **Action on the plot:**
  decompose the over-rounds lag into its `[LAG_DECOMP]` components (we already parse
  them) as a stacked binned area, so the rising part is visibly the `queue_wait`
  component — turning "lag is getting worse, why?" into "queue_wait is the riser,
  which = staleness, which = overhead." Add a guard note tying it to PARITY §3.
- **`barrier_wait`, `speedup_factor`, `trainer_time_split`** (hard to read,
  [:1490](../../../scripts/analysis/analyze_run.py#L1490), [:1467](../../../scripts/analysis/analyze_run.py#L1467), [:1620](../../../scripts/analysis/analyze_run.py#L1620)):
  apply the same binned-line treatment; for `trainer_time_split` keep the whole-run
  stacked bar ([:1656](../../../scripts/analysis/analyze_run.py#L1656)) as the headline
  and demote the over-rounds area to a CDF (already present) — drop the noisy area.
- **`vclock_adv`, `sim_delay` need CDFs** ([:1514](../../../scripts/analysis/analyze_run.py#L1514)):
  add `cdf_plot` companions for the per-round vclock advance and the per-trainer
  sim-delay (sim_round_duration − gpu). The CDF is the parity-relevant view (matches
  how PARITY checks K3/U3 think) and renders instantly.

---

## §3 New thematic deep-dives (the "more thorough plots" ask)

The user wants *more* on **availability**, **selection (and why)**, and
**aggregation (rate, frequency, staleness)**. Three new sub-modules, each a small
set of dense figures, gated to skip cleanly when the data is degenerate (e.g. syn_0
availability).

### 3a Availability (`plots/availability/`)
syn_0 is static, so these only light up under a real availability trace; design for
that now.
- **Availability composition over time** — line per state (`AVL_TRAIN`, `AVL_EVAL`,
  `UN_AVL`), count on y. (Replaces the stacked block.)
- **Per-trainer duty-cycle CDF** — fraction of run each trainer was available
  (parity check A4 lives here too).
- **Availability churn rate** — `avail_change` events per round-bin (how dynamic is
  the trace?).
- **Available-vs-selected funnel per round-bin** — candidates → eligible → chosen,
  three binned lines, showing where the population is lost (already have
  `num_candidates`/`num_eligible`/`num_chosen` on every selection event).

### 3b Selection — *why* (`plots/selection/why/`)
This is the heart of the user's request: *visually see which factor drove selection.*
Use `per_trainer` factor fields (`believed_I`, `system_util`, `temporal`, `speed_s`,
plus `last_train_round`/`last_eval_round` for recency).
- **Picked-vs-pool factor separation** — for each factor, overlaid CDFs of the value
  among *picked* vs *not-picked* trainers (per round, pooled). Big horizontal gap =
  that factor drove selection; overlapping = it didn't. One small-multiple per factor
  → a single "what the selector actually optimized" figure.
- **Factor → selection correlation over time** — per round-bin, Spearman corr between
  each factor and the chosen indicator (`_spearman` already exists, [:401](../../../scripts/analysis/analyze_run.py#L401)).
  Lines that rise/fall show the selector shifting emphasis (e.g. exploration→exploit).
- **Utility distribution evolution** — percentile bands (p10/p50/p90) of believed
  utility per round-bin + a trainer×round-bin utility heatmap (sorted by mean). The
  aggregate-across-300 view the user asked for.
- **Cross-baseline overlay**: emit the picked-vs-pool and percentile-band figures with
  identical axes so felix/refl/oort/feddance can be composited (extends
  `compare_streaming`, [:1846](../../../scripts/analysis/analyze_run.py#L1846)).

### 3c Aggregation — rate / cadence / staleness (`plots/aggregation/`)
The user wants "what rate did an update get aggregated, how frequently, what
staleness." Most inputs already exist on `agg_round` (`staleness`, `agg_goal_count`,
`updates_in_queue`, `contributing_trainers`) plus the new `commit_gap_s`/`buf_depth`/
`residence_rounds`/`inflight` from PARITY §3a.
- **Commit cadence** — commits per virtual-second (and per round) over time;
  binned. Answers "how fast are updates landing."
- **Staleness over time + CDF** — already have both ([:1700](../../../scripts/analysis/analyze_run.py#L1700));
  add a **staleness vs trainer-speed** binned scatter (slow trainers should be the
  stale ones — confirms the mechanism).
- **Reorder-buffer health** — `commit_gap_s` and `buf_depth` over round-bins
  (PARITY §3 telemetry); >0 and rising = backup. This is the direct visual for the
  felix overhead bug.
- **Update residence time** — CDF of `residence_rounds` (rounds an update waited in
  the buffer before committing). Ties staleness to the buffer mechanic.
- **Contribution recency** — histogram of `round − last_train_round` for committed
  updates (how stale, in wall terms, were the models that landed).

---

## §4 Conceptual answers (collected, for the doc body)

| Question | Answer |
|---|---|
| What is the disparity plot? | `utility_streamed/utility_full`; streaming hides part of a trainer's data so believed utility ≠ full-data utility. Ratio<1 ⇒ selector under-values not-yet-unlocked trainers. Connect to mis-selection rate to show impact. |
| Should `global_wt_change` rise or fall? | **Fall** (decay toward 0) as the model converges. Flat/rising tail = non-convergence, churn, or staleness re-injecting old gradients. Overlay accuracy to read it as a convergence diagnostic. |
| Why is `eval_vs_train_selections` empty? | Eval selections happen but most carry `chosen=[]` and eval rounds are sparse vs 4230 train rounds → invisible under a full-width stacked area. Plot eval as its own smoothed rate line. |
| Why does `believed < actual` (refl) but `believed > actual` (felix)? | refl has no eval selector → conservative belief that under-estimates realized utility. felix's eval selector feeds back with lag during async churn → belief over-estimates the level even while improving rank correlation. Also `believed_I` is sometimes null (logging gap) — surface coverage. |
| Why does `send_recv_lag` worsen over time? | It's the buffer-backup symptom from PARITY §3: `queue_wait_s ≈ staleness × wall_per_round`; as the reorder buffer backs up (overhead vs sct-frontier mismatch) staleness and thus lag rise monotonically. Should flatten with felix overhead=0.315. Decompose the over-rounds lag to make `queue_wait` the visible riser. |

---

## §5 Telemetry additions needed (small)

1. **Eval trainer-rounds**: ensure eval forward passes emit a record with
   `real_gpu_time_s` + `task_to_perform="eval"` so `compute_time_by_task_cdf` and the
   time-allocation split get their eval data. (Eval runs in a daemon thread per
   PARITY §3a — make sure it still telemetry-emits.)
2. **`believed_I` coverage**: guarantee `per_trainer[*].believed_I` is set for picked
   trainers (today often null → gappy believed-vs-actual plots).
3. **Optional `selection_lite`** event (round, task, num chosen, chosen ids, factor
   summary) without the full 300-entry `per_trainer` blob — lets §0.2 skip the heavy
   decode for count-only plots. Keep the full event for the why-plots.
4. **`idle_train` vs `idle_eval`**: derivable from existing availability state; no new
   field strictly needed, just split in the analyzer.

Everything in §3c (`commit_gap_s`, `buf_depth`, `residence_rounds`, `inflight`) already
exists per PARITY §3a.

---

## §6 Phased work plan & status

**Phase 1 — speed + shared primitives.  ✓ DONE**
- ✓ §0.1 single-pass `parse_agg_log()` (memoized) — the six log parsers are now
  thin accessors; the 2.2 GB log is read once. Validated: felix sim run ~2m35s
  end-to-end (was multiple full log scans).
- ✓ §1 `binned_line` helper (`plot_helpers.py`).
- ✓ Applied to: insights scatters (loss/update_norm/utility vs visible),
  `runtime_agg_vs_trainer` (P99/bin), `selection_count_consistency`, `queue_depth`
  (+CDF), `send_recv_lag_over_rounds`.
- ✓ CDF companion for per-round vclock advance (`sim_vclock_advance_cdf`); sim-delay
  CDF already existed (`trainer_time_breakdown_cdf`).
- ◻ Deferred: §0.2 streamed JSONL, §0.3 checkpoint cache, §0.5 parallel groups,
  `--quick`. (The single-pass log parser was the dominant win; these are follow-ups.)

**Phase 2 — correctness/legibility on existing plots.  ✓ DONE (core)**
- ✓ Fairness now explicitly train+eval, with a train-only comparison curve; dropped
  `selection_coverage` (Lorenz is the single fairness figure).
- ✓ Renamed `trainer_state_fraction_*` → `trainer_time_allocation_*`; idle split into
  `idle_train`/`idle_eval`; unused categories auto-dropped; % only on aggregate bar.
- ✓ `availability_composition` → lines (honest single flat line under syn_0).
- ✓ `eval_vs_train_selections` → eval/train rate lines (binned) — eval selector now
  visible.
- ✓ `exploration_factor` → explore/exploit lines, x clipped to the decay→0 round.
- ✓ `global_weight_change_norm` → accuracy overlay (nearest eval) + convergence marker.
- ✓ `mqtt` drops callout + per-round delivery-delta histogram.
- ✓ believed-vs-actual: signed `believed−actual` gap histogram (mean/sign annotated) +
  `believed_I` coverage in titles.
- ✓ `util_disparity_ratio` overlaid with mis-selection rate (impact on selection).
- ✓ comm per-round + message accounting → cumulative lines.
- ◻ Deferred (needs §5.1 telemetry): `compute_time_by_task` eval series + per-task
  MB/count histograms; per-round LAG_DECOMP stacked area for send-recv-lag.

**Phase 3 — new deep-dives (§3).  ✓ DONE (core), some plots deferred**
- ✓ `plots/aggregation/`: commit cadence, staleness-vs-speed, buffer health
  (commit_gap_s/buf_depth + CDF), residence-time CDF. Async-only plots gate out for
  sync baselines.
- ✓ `plots/selection/why/`: picked-vs-pool factor separation CDFs (believed_I,
  system_util, temporal, speed_s — refl logs these too), picked-utility p10/p50/p90
  bands over rounds.
- ✓ `plots/availability/`: candidates→eligible→chosen funnel (always on); duty-cycle
  CDF + churn rate (gated to dynamic traces; static → explicit no-data note).
- ◻ Deferred: factor→selection Spearman-over-time, utility heatmap (trainer×round-bin),
  cross-baseline overlays via `compare_streaming`.

**Phase 4 — pruning.  ◻ TODO**
- Cut to the ~8 parity-relevant figures for the parity loop (PARITY §4/#6) behind
  `--quick`; keep the full set for deep dives.

---

### Decisions (resolved)
- **Location**: new top-level subdirs `plots/availability/`, `plots/aggregation/`,
  `plots/selection/why/` — distinct audience. ✓
- **Availability validation**: no dynamic-availability run yet → build §3a **gated to
  skip cleanly** under static (syn_0) availability; validate later against a real trace.
- **`--quick` default**: TBD (revisit at Phase 4); default stays full set for now.
