# `analyze_run.py` — onboarding a new example

`analyze_run.py` reads the JSONL telemetry a run writes under
`<run>/telemetry/` and produces PDFs under `<run>/plots/`, grouped into 8
categories:

| category | question it answers | needs |
|---|---|---|
| `performance/` | did it learn? (accuracy, loss) | `agg_eval` |
| `sanity/` | is the run healthy & as configured? | `trainer_round` |
| `selection/` | who was picked? | `selection` |
| `selection/why/` | why were they picked? | `selection` with per-trainer factor fields (`believed_I`/`system_util`/`temporal`/`speed_s`) |
| `insights/` | mis-selection & data-unlock audit | `util_disparity` + oracle-utility logs (oort-family selectors only) |
| `system/` | resources, timing, communication | `trainer_round`, `agg_round` |
| `availability/` | selection funnel & churn | `selection`, `avail_change` |
| `aggregation/` | commit cadence, staleness vs. speed | `agg_round` |

It was originally written for `async_cifar10` and generalized only
informally, so two things it assumed were true for that example are **not**
true in general:

1. **`round` is the only progress unit, and it's fine-grained** (advances
   every aggregation). Not true for fwdllm-family examples, where `round`
   only advances once every sub-unit (`data_id`, `total_data_bins=150`
   of them) finishes — a run can sit at `round == 1` for hundreds of events.
2. **The model has one fixed, full-model size for comm-cost plots.** Not true
   for gradient/perturbation-based schemes (e.g. FedFwd) that only ever put a
   trainable subset on the wire.

Both are handled by a per-example, **optional** manifest — an example with no
manifest gets exactly `async_cifar10`'s original behavior, unchanged.

## The manifest: `examples/<name>/telemetry_manifest.yaml`

`configure_from_manifest()` walks up from a run's `telemetry_dir` looking for
`examples/<name>/telemetry_manifest.yaml` and, if found, applies it. Three
fields, all optional:

```yaml
model_param_count: 450340   # trainable params actually sent over the wire —
                             # not necessarily the full model (see fwdllm's
                             # manifest for the FedFwd rationale). Overridden
                             # by --model-params on the CLI, never the reverse.

progress_hierarchy:          # sub-round fields, ordered major->minor, each
  - field: data_id            # with a `bound` used as that level's mixed-
    bound: 150                # radix multiplier (must exceed the field's max
  - field: iteration_per_data_id   # real value, or ordering breaks across
    bound: 15                      # round boundaries). Consumed by
                                    # progress_key() — see its docstring for
                                    # which call sites it's safe to use on
                                    # (single-stream groupings only; never to
                                    # join against an event type that doesn't
                                    # carry the same fields, e.g. selection).

event_categories:            # declare what this example's telemetry actually
  performance: populated     # populates, and why (populated | partial |
  ...                        # not_populated). write_summary() cross-checks
                              # this against which plots/<category>/ dirs
                              # actually got files and flags drift:
                              #   MISMATCH — declared populated, nothing written
                              #   DRIFT    — declared not_populated, something written
                              # See lib/python/examples/fwdllm/telemetry_manifest.yaml
                              # for a fully worked example with reasoning
                              # comments per category.
```

See `lib/python/tests/analysis/test_manifest.py` for the loader's exact
fallback behavior (absent manifest, malformed YAML, CLI-override guard).

## Checklist: wiring telemetry for a new example

Worked example throughout: fwdllm (`lib/python/examples/fwdllm/`), which went
through this exact process — see
`lib/python/examples/fwdllm/MIGRATION_TO_LAUNCHER_FWDLLM.md` Part 5 for the
full investigation, evidence, and fixes referenced below.

1. **Aggregator must emit at least `agg_eval` and `agg_round` once per
   aggregation cycle.** Without this, `plots/performance/` and
   `plots/aggregation/` are structurally impossible no matter what the
   analyzer does — this was fwdllm's biggest gap (`fwdllm_aggregator.py`
   emitted zero telemetry of any kind before P5.2). Wrap every emit in
   `if telemetry.is_enabled(): try: ... except Exception: logger.debug(...)`
   — telemetry must never be able to break training.
2. **The real FL-selection algorithm must call `emit_selection(...)`.**
   Check the actual selector class the aggregator's `selector.sort` config
   uses (not just any selector in the codebase) — sibling selectors are not
   a reliable reference. fwdllm/fwdllm_plus use `selector/random.py`, which
   had never called `emit_selection` at all (P5.5); every `selection` event
   that existed for those baselines came from the trainer's own trivial
   1-candidate channel selector, not the real "who got picked" decision.
3. **If progress isn't a plain, every-aggregation `round`, declare a
   `progress_hierarchy`** in the manifest instead of hardcoding sub-round
   fields into the analyzer (`progress_key()`'s `_DEFAULT_PROGRESS_HIERARCHY`
   fallback is fwdllm's original shape, kept only for backward compatibility
   with callers that pass no `telemetry_dir` context — new examples should
   declare their own, not rely on it).
4. **If comm-cost plots should reflect less than the full model**, set
   `model_param_count` to what's actually transmitted, measured from a real
   run's log line (a trainable-param-count log, a `state_dict()` size dump,
   etc.) — not guessed.
5. **Declare `event_categories`** honestly, including `not_populated` for
   categories that structurally can't be populated (a streaming-only insight,
   an oort-family-only mis-selection audit, etc.) — an undeclared category is
   a silent gap someone has to rediscover; a declared `not_populated` is a
   documented, expected outcome.
6. **Run `analyze_run.py` against a real (or synthetic, matching the
   telemetry shape the instrumented code actually emits) smoke run** and read
   `plots/summary.txt`'s `manifest event_categories check` section. Every
   line should read `ok: ...` — a `MISMATCH` or `DRIFT` line means the
   manifest and reality disagree; fix whichever one is wrong.
7. **Add regression tests exercising the real emission code path** (not
   mocked out) — see `lib/python/tests/mode/test_fwdllm_agg_telemetry.py` and
   `lib/python/tests/selector/test_random_selection_telemetry.py` for the
   fwdllm pattern: minimal real objects (e.g. a real `torch.nn.Linear(1,1)`
   model) plus fakes only for the parts genuinely external to what's being
   tested.

## Gotchas found while onboarding fwdllm

- **A trainer-role `selection` event's `"selector"` field is a
  channel-implementation artifact, not the real FL selector.** Every role
  (aggregator *and* trainer) constructs its own local `Selector` for its
  channel; a trainer's channel to its aggregator always has exactly 1
  candidate, so whatever selector class the example's `trainer_base.yaml`
  hardcodes as a placeholder trivially "selects" that one peer regardless of
  its kwargs. This is launcher-wide, not fwdllm-specific (checked
  `async_cifar10` too — same pattern). Read which selector algorithm a run
  really used from the **aggregator**'s config/telemetry, never the
  trainer's.
- **A stale in-memory reselection cache can silently exclude a departed
  trainer forever** if a per-round accumulation path doesn't call the real
  selector every cycle (fwdllm's `_round_selected_ends` gap, fixed via
  `_prune_departed_from_round_cache` — see the migration doc's Part 3). Not
  an analysis-tooling gap, but the kind of thing that produces telemetry that
  *looks* fine (events keep flowing) while the run itself has silently
  stalled for one trainer's slot — worth an explicit check on any new
  example with its own reselection/caching logic.
