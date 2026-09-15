# Getting started with FMoW setup

[`setup_fmow.py`](../setup/setup_fmow.py) prepares the dataset, image capture
schedule, ground stations, and satellite availability. It does not generate
satellite orbits or launch a training experiment.

See the [configuration reference](config-reference.md) for every supported
field and its default.

## 1. Prepare the environment and orbital inputs

Run from the Flame repository root. Setup needs Python with `numpy` and
`PyYAML`; capture generation also needs `pandas` and `scipy`. Downloading the
dataset uses `bash`, `curl`, and `tar`, with network access and disk space for
the roughly 54 GB archive plus the extracted dataset.

Place the following inputs in the directory configured as `satellites.leo_dir`:

| Input | Required for | Contents |
| --- | --- | --- |
| `geodetic.npz` | Capture generation | `coords` with shape `(timesteps, satellites, 2)` containing latitude/longitude in degrees; `time_s`; and `sat_names`. |
| `ecef.npz` | Availability generation | `ecef_km` with shape `(timesteps, satellites, 3)` containing Earth-centered coordinates in kilometers; and `time_s`. |

Both arrays must describe the same satellites in the same order.
Setup generates captures and availability for all satellites in the orbital
inputs. The coverage script alone accepts `--num-satellites 200` to analyze
indices 0 through 199, or defaults to the trainer-registry count when omitted. Satellite
index `0` becomes `trainer_001`, index `1` becomes `trainer_002`, and so on.

The current availability generator uses sample indices as seconds and truncates
`min_window` to an integer sample count. Supply one-second samples starting at
time zero. It reads `time_s` but does not use it to convert availability event
timestamps. Capture generation does use `time_s`.

Existing `captures.npz` is not a replacement for `geodetic.npz` when running
`all`: the capture step always regenerates it.

## 2. Configure setup

Edit [`configs/fmow_config.yaml`](../configs/fmow_config.yaml). Check the dataset
and orbital paths, capture radius, and ground-station settings.

Include `ground_stations` to generate both stations and availability during
`all`. Its `num_stations`, `seed`, `elevation_angle`, and `min_window` fields
all have defaults. `availability.trace_path` optionally overrides where the
trace is written and later read.

The repository's `fmow/metadata` is a symlink to `examples/_metadata`, so writes
through that path update the shared metadata directory.

## 3. Run setup

```bash
python lib/python/examples/fmow/setup/setup_fmow.py all
```

To use another configuration:

```bash
python lib/python/examples/fmow/setup/setup_fmow.py all --config /path/to/fmow_config.yaml
```

The steps run in this order:

| Step | What it does | Output and existing-file behavior |
| --- | --- | --- |
| Download | Downloads and extracts FMoW if the dataset directory does not appear populated. | Skips when `rgb_metadata.csv`, `country_code_mapping.csv`, and an `images/` directory exist. This is a presence check, not a full integrity check. |
| Captures | Uses training-image coordinates and `geodetic.npz` to record each image's first capture by each satellite. | Overwrites `satellites.leo_dir/captures.npz`. |
| Ground stations | Generates station coordinates from the configured count and seed. | Overwrites `satellites.leo_dir/ground_stations.yaml` if `ground_stations` is configured. |
| Availability | Reads the station file and `ecef.npz`, then calculates usable visibility intervals. | Overwrites `availability.trace_path` if `ground_stations` is configured. |

A failed step stops the command; earlier completed outputs remain. Setup does
not skip generation just because its output files already exist.

## Run only the steps needed

Replace `all` with one of these commands. Each accepts `--config PATH`.

| Command | Prerequisites and behavior |
| --- | --- |
| `download` | Downloads the dataset if needed. No orbital files required. |
| `captures` | Requires `geodetic.npz` and the dataset's `rgb_metadata.csv` with `split`, `lat`, and `lon` columns. Does not download the dataset first. |
| `ground-stations` | Requires `ground_stations` settings. Generates only the station file; no orbital files required. |
| `availability` | Requires `ground_stations` settings, an existing `ground_stations.yaml`, and `ecef.npz`. Uses the station file without regenerating it. |

With no command, setup defaults to `all`.

If captures already exist and only station connectivity needs preparing:

```bash
python lib/python/examples/fmow/setup/setup_fmow.py ground-stations
python lib/python/examples/fmow/setup/setup_fmow.py availability
```

For manually specified ground stations, keep the desired file and run only
`availability`. Its coordinates come from that file; the configured station
count and seed only affect the `ground-stations` command.

## After setup

The generated captures contain `events`, `offsets`, and `sat_names`. The generated
availability YAML contains per-trainer `[time, state]` events using `AVL_TRAIN`
and `UN_AVL`. Availability ends with an unavailable event at the trace horizon,
even with `elevation_angle: -90` and `min_window: 0`.

To train, use an experiment YAML such as
[`fmow_fedavg_n200.yaml`](../exports/fmow_fedavg_n200.yaml), which selects satellite
availability. The training environment, trainer registry, and Flame runtime
configuration are separate prerequisites; setup does not create them. Ensure
the experiment's trainer and aggregator point to the same FMoW config and that
each selected satellite has a trace and registry entry.
