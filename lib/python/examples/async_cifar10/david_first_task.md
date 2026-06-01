# First task: add time-varying trainer location (lat/long) from a mobility trace

This builds on the async_cifar10 example. Follow [README](README.md) first and
confirm you can run the felix smoke test end-to-end before starting here.

The goal is to give each trainer a location identity that moves over time — like a
real mobile device. The trainer should read a per-device lat/long trace, interpolate
its current coordinates based on wall-clock time elapsed since it started, and emit
location as structured telemetry each round.

The smoke test to target is the **real-mode** felix smoke:
```bash
python -m flame.launch.run_experiment \
    lib/python/examples/async_cifar10/expt_scripts_2026/felix_n10_alpha100_syn0_smoke.yaml
```
(Set `time_mode: real` in the YAML. Simulation mode is still being verified; do
not target it for this task.)

---

## Background: how trainer metadata flows today

1. **Registry** — `examples/_metadata/trainer_registry.yaml`. Each entry already
   has a `mobiperf_device_id` (e.g. `device_001`). This is the logical handle
   that links a trainer to a device profile.

2. **Spawner injection** — `flame/launch/spawner.py` (~line 118-160) reads
   `trainer_meta` from the registry and injects into each trainer's config:
   `training_delay_s`, `trainer_indices_list`, all availability traces. You will
   follow the same pattern to inject a location trace.

3. **Schema** — `flame/config.py`, `Hyperparameters` class defines what fields the
   trainer config carries. You will add a new field here.

4. **Trainer** — `trainer/pytorch/main.py`. Reads from `config.hyperparameters` at
   `__init__`. Emits structured telemetry via `telemetry.emit(ev, **fields)` in the
   `train()` method (~line 721). You will add location reading and emission here.

5. **Telemetry events** — `flame/telemetry/events.py` defines typed event builders
   (e.g. `build_trainer_round`). You can add a field to `build_trainer_round` or
   add it to the `extra={}` dict — pick whichever keeps the schema clean.

---

## Task A — Create a per-device location trace and wire it to the registry

Each trainer needs a sequence of `(elapsed_s, lat, lon)` waypoints that represent
where the device is at a given number of seconds after the experiment starts.

Suggested subtasks:

- [ ] **Design the trace format.** A simple YAML or JSON list per device works:
  ```yaml
  device_001:
    - {elapsed_s: 0,    lat: 37.7749, lon: -122.4194}   # San Francisco
    - {elapsed_s: 300,  lat: 37.3861, lon: -122.0839}   # Palo Alto
    - {elapsed_s: 600,  lat: 37.3382, lon: -121.8863}   # San Jose
  ```
  Keep the data small (3-5 waypoints per device) for now. You can use any
  geographically plausible coordinates — real-world city pairs work well.
  For the 10 smoke-test trainers (device_001 … device_010), 10 entries suffice.

- [ ] **Write a generator script.** Add a script under
  `examples/_metadata/` (e.g. `generate_location_traces.py`) that produces a
  `location_traces.yaml` file in the same directory. Use random or hand-chosen
  routes; document what you chose and why. Check it in alongside the generated
  file so it can be re-run if routes need changing.

- [ ] **Add to registry.** No need to embed full trace data in
  `trainer_registry.yaml`; the spawner already loads traces separately. Store the
  mapping `trainer_id → device_id` (already there as `mobiperf_device_id`) and
  let the spawner resolve it.

---

## Task B — Inject the location trace via the spawner

Follow the same pattern as availability traces (~line 136-158 in `spawner.py`).

- [ ] **`ExperimentMetadata`.** In `flame/launch/metadata.py` (or wherever
  `get_mobiperf_trace` is implemented), add a `get_location_trace(trainer_id)`
  method that loads the device's waypoint list from `location_traces.yaml`.

- [ ] **`spawner.py`.** After the mobility trace injection block, add:
  ```python
  config["hyperparameters"]["location_trace"] = self.metadata.get_location_trace(trainer_id)
  ```

- [ ] **`flame/config.py`.** Add `location_trace: Optional[List[Dict]] = None`
  to `Hyperparameters` (or `Optional[Any]` if the schema is flexible). Keep the
  field optional so existing configs without it still load.

---

## Task C — Implement time-varying location in the trainer

- [ ] **Read trace at `__init__`.** Parse `config.hyperparameters.location_trace`
  into a list of `(elapsed_s, lat, lon)` tuples. Store as `self.location_trace`.
  If absent or empty, set `self.location_trace = None`.

- [ ] **Interpolate current location.** Write a small helper
  `_current_location(elapsed_s)` that linearly interpolates `(lat, lon)` from the
  waypoint list. Use the last waypoint once past the final entry (no extrapolation).
  ```python
  def _current_location(self, elapsed_s: float):
      if not self.location_trace:
          return None, None
      # find bounding waypoints and interpolate
      ...
  ```

- [ ] **Emit each round.** In `train()`, after computing `_cycle_start`, compute:
  ```python
  elapsed = time.time() - self._experiment_start_time
  lat, lon = self._current_location(elapsed)
  ```
  Add `"lat": lat, "lon": lon` to the `extra={}` dict in the `build_trainer_round`
  telemetry call (~line 741). Log it at DEBUG level too.

- [ ] **`_experiment_start_time`.** Set `self._experiment_start_time = time.time()`
  once in `__init__` (or on the first train call). Don't reset it per round.

---

## Task D — Verify

- [ ] Run the real-mode smoke test (`time_mode: real`, 10 trainers). Confirm the
  trainer JSONL telemetry files contain `lat`/`lon` fields in `trainer_round`
  events.
- [ ] Write a short script (or add to `compare_parity.py` as a helper) that reads
  the trainer telemetry and plots each trainer's path (lat vs lon) over time. Two
  trainers on the same device_id should trace identical paths.
- [ ] Confirm `lat`/`lon` are `None` for trainers whose config has no
  `location_trace` (graceful no-op).

---

## Notes & gotchas

- The trainer already has `_sim_now()` which returns wall-clock-since-start in real
  mode. You can reuse that instead of a separate `_experiment_start_time` if it
  already tracks what you need.
- The location feature should be **read-only** — it must not affect selection,
  training, or aggregation. It is purely telemetry/metadata.
- Keep trace files small (< 50 KB total). If you want more realistic movement,
  sample from publicly available GPS datasets; just keep the format consistent.
- The spawner runs inside the launcher process; the trainer runs as a subprocess.
  Data passes via the generated config JSON — it must be JSON-serializable (list of
  dicts, no tuples at the top level).
