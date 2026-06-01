# First task: make training delay time-varying (sinusoidal model)

This builds on the async_cifar10 example. Follow [README](README.md) first and
confirm you can run the felix smoke test end-to-end before starting here.

Today each trainer has a fixed `training_delay_s` (D) that never changes. Real
devices vary: a phone on charge trains faster, a busy device trains slower. The
goal is to make D vary over wall-clock time using a sinusoidal model, so each
trainer's effective budget oscillates around its configured mean value.

The smoke test to target is the **real-mode** felix smoke:
```bash
python -m flame.launch.run_experiment \
    lib/python/examples/async_cifar10/expt_scripts_2026/felix_n10_alpha100_syn0_smoke.yaml
```
(Set `time_mode: real` in the YAML. Simulation mode is still being verified; do
not target it for this task.)

---

## The model

```
D(t) = clamp(mean_D + amplitude × sin(2π × t / period_s), min=0.5)
```

where:
- `mean_D` = `training_delay_s` from the registry (the existing per-trainer value)
- `amplitude` = `mean_D × amplitude_fraction` (default `0.2` → ±20% of mean)
- `period_s` = oscillation period in seconds (default `120s`)
- `min` = hard floor of `0.5s` (never less than half a second)
- `t` = wall-clock seconds elapsed since the trainer started

With the default values a trainer with `mean_D = 4s` oscillates between `3.2s`
and `4.8s`; a trainer with `mean_D = 16s` oscillates between `12.8s` and `19.2s`.

Both min and max stay within 20% of the mean, satisfying the original spec.

---

## Background: how training_delay_s works today

1. **Registry** — `examples/_metadata/trainer_registry.yaml`. Each entry has a
   `training_delay_s` value (e.g. `"4.0"` for fast, `"16.0"` for very_slow).

2. **Injection** — `flame/launch/spawner.py` (~line 130):
   ```python
   config["hyperparameters"]["training_delay_s"] = trainer_meta["training_delay_s"]
   ```

3. **Schema** — `flame/config.py`, `Hyperparameters`:
   `training_delay_s` (alias `trainingDelaySeconds`), `training_delay_enabled`.

4. **Application** — `trainer/pytorch/main.py`: reads `self.training_delay_s` at
   `__init__`; uses it as the budget D each round in the timing block (~line 696):
   ```python
   _modeled_delay_s = self.training_delay_s if self.training_delay_enabled else 0.0
   _remaining_time  = max(0.0, _modeled_delay_s - _real_gpu_time_s)
   ```

Your task is to make `_modeled_delay_s` time-varying instead of fixed.

---

## Task A — Add variation config fields to the schema

- [ ] **`flame/config.py`.** Add a nested config block to `Hyperparameters`:
  ```python
  training_delay_variation: Optional[Dict] = None
  # Expected keys: enabled (bool), period_s (float), amplitude_fraction (float)
  # Defaults: enabled=False, period_s=120, amplitude_fraction=0.2
  ```
  Keep it optional so all existing configs without it continue to work unchanged.

- [ ] **Smoke-test YAML.** In
  `expt_scripts_2026/felix_n10_alpha100_syn0_smoke.yaml` (or a copy), add under
  `trainer.hyperparameters`:
  ```yaml
  training_delay_variation:
    enabled: true
    period_s: 120
    amplitude_fraction: 0.2
  ```

---

## Task B — Implement time-varying delay in the trainer

- [ ] **Read config at `__init__`** (`trainer/pytorch/main.py`). Parse
  `config.hyperparameters.training_delay_variation` and store:
  ```python
  _var = config.hyperparameters.training_delay_variation or {}
  self.delay_variation_enabled   = str(_var.get("enabled", "False")).lower() == "true"
  self.delay_variation_period_s  = float(_var.get("period_s", 120))
  self.delay_variation_amplitude = float(_var.get("amplitude_fraction", 0.2)) * self.training_delay_s
  ```

- [ ] **Record start time.** Add `self._experiment_start_time = time.time()` in
  `__init__` (set once; never reset per round).

- [ ] **Compute D(t) each round.** Add a helper method:
  ```python
  def _effective_delay_s(self) -> float:
      if not self.delay_variation_enabled:
          return self.training_delay_s
      t = time.time() - self._experiment_start_time
      raw = self.training_delay_s + self.delay_variation_amplitude * math.sin(
          2 * math.pi * t / self.delay_variation_period_s
      )
      return max(0.5, raw)
  ```

- [ ] **Use it in the timing block.** Replace the fixed read with the helper in
  the `train()` method (~line 696):
  ```python
  _modeled_delay_s = self._effective_delay_s() if self.training_delay_enabled else 0.0
  ```
  No other lines need to change — `_remaining_time`, overrun logic, and
  `sim_round_duration` all derive from `_modeled_delay_s`.

- [ ] **Log it.** Add `effective_delay_s` and `delay_phase_t` to the telemetry
  `extra={}` dict so you can plot D(t) vs round from the trainer JSONL.

---

## Task C — Verify

- [ ] Run the real-mode smoke test (10 trainers, `time_mode: real`, variation
  enabled). Confirm from trainer logs that `[TRAIN_CYCLE]` lines show varying
  `budget=` values over rounds for each trainer.

- [ ] Write a short script that reads `trainer_N.jsonl` files and plots
  `effective_delay_s` vs round number (or elapsed time) for all trainers. You
  should see sinusoidal oscillation around each trainer's `mean_D`.

- [ ] Confirm that with `enabled: false` (or config absent) the behavior is
  identical to the original fixed-delay path.

- [ ] Confirm the 0.5s floor: set `amplitude_fraction: 1.5` for a trainer with
  `mean_D = 0.3s`; verify logged `effective_delay_s` never drops below 0.5s.

---

## Notes & gotchas

- `math.sin` is already imported in `trainer/pytorch/main.py`.
- The variation parameters are **per-trainer-globally-fixed** (read once from
  config), but `_effective_delay_s()` returns a **per-call varying value** because
  `t = time.time() - start` changes each call. This is intentional.
- In sim mode `self._experiment_start_time` (wall-clock) is not the same as
  sim-time. For now scope this task to `time_mode: real` only. Add a
  `TODO: sim-time support` comment if you want to mark the gap.
- The `training_delay_s` in the registry stays as the **mean** value; you are not
  changing the registry. Variation is additive on top of that mean.
- Eval path also uses `training_delay_s` (search for `eval_delay` or a second
  sleep in `evaluate()`). Check if it should also vary — for this task, leave it
  fixed and note the inconsistency.
