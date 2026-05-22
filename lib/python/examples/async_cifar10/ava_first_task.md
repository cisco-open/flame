# First task: split the trainer delay into compute + RTT, then make RTT time-varying

This builds on the async_cifar10 example. For environment setup and how to run a
baseline, follow the [README](README.md) first — make sure you can run the felix
(or any) smoke test end-to-end before starting here.

The goal is two incremental changes to how a trainer's per-round delay is
modeled:

1. **(Task A)** Replace the single `training_delay_s` with two components:
   `computation_time_ms` and `rtt_communication_time_ms`.
2. **(Task B)** Make `rtt_communication_time_ms` vary over wall-clock time, and
   have the trainer pick its current RTT based on time elapsed since it started.

You should not need to touch the aggregator or the launcher core.

---

## Background: how the delay works today

The per-trainer delay flows metadata → config → trainer:

1. **Source of truth** — `examples/_metadata/trainer_registry.yaml`. Each entry
   has `training_delay_s` (plus `speed_class`, `mobiperf_device_id`):

   ```yaml
   trainer_001:
     trainer_id: 1
     task_id: 505f9fc4...
     training_delay_s: "4.0"
     speed_class: fast
   ```

2. **Injection** — `flame/launch/spawner.py` copies the registry value into each
   trainer's generated config (`config["hyperparameters"]["training_delay_s"] =
   trainer_meta["training_delay_s"]`, near line 130).

3. **Schema** — `flame/config.py`, `Hyperparameters` class: `training_delay_s`
   (alias `trainingDelaySeconds`), `training_delay_enabled`.

4. **Application** — `trainer/pytorch/main.py`: read at `__init__`
   (`self.training_delay_s = ...`), applied in the train path with
   `time.sleep(self.training_delay_s / self.speedup_factor)` (~line 514) and a
   scaled version in the eval path (~line 647).

`speedup_factor` divides the delay so smoke tests run faster — preserve that.

---

## Task A — split into computation_time_ms + rtt_communication_time_ms

Conceptually: `delay = computation_time + rtt_communication_time`. Keep the two
separate so RTT can later vary independently of compute.

Suggested subtasks:

- [ ] **Registry.** Add `computation_time_ms` and `rtt_communication_time_ms` to
  each entry in `trainer_registry.yaml`. Decide migration: derive from the
  existing `training_delay_s` (e.g. a fixed compute/RTT split) so behavior is
  unchanged at first. Keep `training_delay_s` temporarily for comparison, or
  remove it once parity is confirmed.
  - Helper: write a one-off script that reads the registry, splits each
    `training_delay_s` into the two fields, and writes it back. Don't hand-edit
    300 entries.
- [ ] **Schema.** Add `computation_time_ms` and `rtt_communication_time_ms`
  fields to `Hyperparameters` in `flame/config.py` (mirror the existing
  `training_delay_s` field: `Optional`, with alias if you want camelCase).
- [ ] **Injection.** In `flame/launch/spawner.py`, inject the two new fields from
  `trainer_meta` into the generated trainer config (alongside or instead of
  `training_delay_s`).
- [ ] **Trainer.** In `trainer/pytorch/main.py`, read the two fields and replace
  the single sleep with `sleep((computation_time_ms + rtt_communication_time_ms)
  / 1000 / speedup_factor)`. Mind the **ms → s** conversion. Update the eval-path
  delay the same way.
- [ ] **Parity check.** Run the felix smoke test; confirm per-round timing and
  the trainer's logged delay match the pre-change behavior (sum of the two ==
  old `training_delay_s`).

---

## Task B — make RTT time-varying

Now let `rtt_communication_time_ms` change as the experiment progresses, and have
each trainer choose its current RTT from the time elapsed since it started.

Suggested subtasks:

- [ ] **Decide the RTT(t) model.** Pick a simple, documented function of elapsed
  seconds `t`, e.g. a step schedule (`t < 60s → base`, else `2×base`) or a
  smooth one (`base * (1 + amplitude * sin(2π t / period))`). Keep compute time
  constant in this task.
- [ ] **Parameterize via the registry (recommended).** Instead of a single
  `rtt_communication_time_ms`, store the parameters of RTT(t) per trainer
  (e.g. `rtt_base_ms`, `rtt_amplitude`, `rtt_period_s`). Thread them through the
  schema (`flame/config.py`) and injection (`spawner.py`) the same way as
  Task A. Avoids hardcoding the model in the trainer.
- [ ] **Track elapsed time in the trainer.** In `trainer/pytorch/main.py`, record
  a start timestamp once (`self.start_time = time.time()` in `__init__` or on
  first round), then each round compute `elapsed = time.time() - self.start_time`
  and `rtt = rtt_of(elapsed, params)`. Use that `rtt` in the sleep instead of a
  static value.
- [ ] **Log it.** Log `elapsed`, the chosen `rtt_communication_time_ms`, and the
  compute time each round so you can plot RTT vs time from the trainer log.
- [ ] **Verify.** Run a smoke test and confirm from the logs that RTT changes
  over time per your model, while compute time stays fixed. A short script that
  greps the trainer log and plots RTT vs elapsed is a good sanity check.

---

## Notes & gotchas

- Units: registry/config are in **ms**; `time.sleep` takes **seconds**. Convert.
- Always divide the sleep by `speedup_factor` so smoke tests stay fast.
- There are two delay sites in the trainer (train path and eval path). Update
  both, keeping their existing ratio.
- The aggregator measures round duration (Oort utility uses it). Time-varying
  RTT will change measured round durations — expected; note it when comparing
  selector behavior across baselines.
- Keep the registry change reversible (a script, not manual edits) so you can
  regenerate if the split ratio or RTT model changes.
