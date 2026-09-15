## FMoW on Flame — Federated Learning over Satellite Imagery

This example ports **fMoW** (Functional Map of the World) onto Flame's federated
learning stack, using a satellite-constellation framing: each trainer is a LEO
satellite that only "sees" (and can train on) the subset of images it has
physically captured so far in simulated time, rather than a static per-client
data partition.

### What is FMoW?

fMoW is a satellite-image scene-classification dataset: RGB chips labeled with
one of 62 land-use/scene categories (`airport`, `race_track`,
`amusement_park`, ...), split into `train` / `val` / `test` / `seq`. This port
uses the WILDS-preprocessed release (fMoW-rgb v1.1), which is a flat directory
of already-resized 224x224 PNGs, `images/rgb_img_{i}.png`, where `i` is simply
the row's position in `rgb_metadata.csv` — **not** the descriptive
`train/<category>/<category>_<id>/...` path that the CSV's `img_path` column
happens to contain (that column is vestigial in this distribution; see
"Bugs found and fixed" below).

### The satellite framing

Instead of a Dirichlet or index-based split, data availability here is physical. `captures.py` uses the `geodetic.npz` LEO constellation to simulate every timestep and track when a satellite comes within a specified `radius` (km) of an image. Each trainer only trains on images captured up to the current simulated time, with its dataset growing over rounds rather than being fixed.

Captures are given as `events`, where an event contains the `timestep` the event occurred on and the `image` captured. 

### Layout

```
configs/
  fmow_config.yaml     FMoW specific configuration
  trainer_base.yaml    Flame trainer hyperparameters template
                       (adds fmow_config_path, reuses satellite_coordinates_path)

dependencies/
  model.py             DenseNet161 (ImageNet-pretrained), group normalized and
                       freeze_layers leaves only denseblock4/norm5/classifier trainable
  fmow_dataset.py      Transforms dataset into Tensor arrays and allows for selection
                       by split

setup/
  download_fmow_dataset.sh   downloads + extracts fMoW-rgb v1.1 (~54GB)
  config.py                  FMoW configuration schema
  captures.py                Builds captures.npz based on geodetic.npz (events + per-satellite offsets)
  generate_ground_stations.py       Generates ground_stations.yaml
  generate_satellite_availability.py Generates satellite visibility events
  setup_fmow.py              Downloads data, schedules captures, and generates optional availability inputs

trainer/pytorch/main.py               PyTorchFMoWTrainer
aggregator/pytorch/main_fedavg_agg.py PyTorchFMoWAggregator

exports/fmow_fedavg_smoke_n10.yaml    FMoW smoketest:
                                      baseline=fedavg, 10 trainers, 50 rounds
```

`model.py`, `fmow_dataset.py`, and `config.py` are symlinked into both
`trainer/pytorch/` and `aggregator/pytorch/` (mirroring the existing
`metadata -> ../_metadata` convention), so both roles import the same code
with a plain top-level `import model` / `from fmow_dataset import ...` rather
than using `sys.path`, since Python auto-adds a directly-run script's own directory
to `sys.path`. The one thing symlinks can't solve is locating the plain data
file `fmow_config.yaml` itself, which is why `fmow_config_path` exists as a
hyperparameter (set once in `trainer_base.yaml` for the trainer, and again
under the experiment YAML's `aggregator.config_overrides.hyperparameters`,
since the aggregator's config comes from a completely separate merge chain
that never reads `trainer_base.yaml`).

### Preparing the inputs

See [Getting started with setup](docs/getting-started.md) for prerequisites and
commands, and the [FMoW configuration reference](docs/config-reference.md) for
all fields, defaults, and optional sections.

After preparing inputs, use the [data coverage script](docs/data-coverage.md) to
estimate unique captured and reachable training images without running training.

From the repository root, run:

```bash
python lib/python/examples/fmow/setup/setup_fmow.py all \
  --config lib/python/examples/fmow/configs/fmow_config.yaml
```

This downloads the dataset if needed, schedules image captures, then generates
ground stations and availability when `ground_stations` is configured:

```yaml
ground_stations:
  num_stations: 8
  seed: 0  # Optional; defaults to 0.
  elevation_angle: 10.0
  min_window: 90.0

# Optional: override the default output path.
availability:
  trace_path: lib/python/examples/fmow/metadata/availability_traces/satellite_traces.yaml
```

Ground stations are written to `satellites.leo_dir/ground_stations.yaml`.
Availability reads that file and `satellites.leo_dir/ecef.npz`, then writes to
`availability.trace_path`. Orbital files must already exist; setup does not
generate satellite orbits. Capture generation also requires `geodetic.npz`.

Omit `ground_stations` to skip both generation steps and preserve existing files.
Within that section, omitted fields default to eight stations, seed `0`, elevation
`10.0` degrees, and minimum window `90.0` seconds.
Omit `availability` or its `trace_path` to use the default
`lib/python/examples/fmow/metadata/availability_traces/satellite_traces.yaml`.
That path controls both generation output and runtime input; it does not enable
generation by itself. Configured generation steps overwrite their outputs.
Use `download`, `captures`, `ground-stations`, or `availability` instead of `all`
to run one step. With no command, setup defaults to `all`. The individual
generator scripts now live under `setup/` and still accept their CLI arguments.

An elevation threshold of `-90` and `min_window: 0` accepts every satellite
throughout the input samples. The generator still marks satellites unavailable
at the end of the trace. These settings do not make the trace infinite.

### How a round works

**Trainer** (`PyTorchFMoWTrainer`, one per satellite): `initialize()` loads
`fmow_config.yaml` and builds the model; `load_data()` loads this satellite's
slice of `captures.npz`, admits whatever's been captured as of the current
simulated time into `self.image_buffer`, and builds the train loader from it.
Every `train()` call re-admits newly captured images and rebuilds the loader
before training, so the dataset a satellite trains on grows as simulated time advances.

**Aggregator** (`PyTorchFMoWAggregator`): standard FedAvg `TopAggregator` —
`random` selector, `fedavg` optimizer, evaluates against the fMoW `test` split.

### Bugs found and fixed

**Framework-level** (`flame/selector/default.py`, affects every example using
`selector.sort: default`, not just FMoW):
- `DefaultSelector`'s initializer didn't accept `**kwargs`, so it broke the moment `channel_manager.py` started unconditionally injecting a `_seed` kwarg into every selector's constructor. Fixed to accept and forward `**kwargs` to `AbstractSelector`, matching every other selector class.
- `DefaultSelector` was also missing `_cleanup_recvd_ends` and `_cleanup_send_ends` methods which seemed to be required by `channel.py` and `trainer.py`.

**FMoW-specific:**
- `_rebuild_train_loader()` built its `DataLoader` with `shuffle=True`, and shuffling crashed whenever a satellite's capture buffer is still empty. Trainers having an empty dataset is specific to the way FMoW captures images over time. Fixed by gating `shuffle=len(indices) > 0`, and by skipping the epoch loop entirely in `train()` when there's nothing captured yet that round.
