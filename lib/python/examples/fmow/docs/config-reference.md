# FMoW configuration reference

[`fmow_config.yaml`](../configs/fmow_config.yaml) describes FMoW input locations,
capture settings, and ground-station visibility settings. Both setup and the
FMoW training roles read it. The experiment YAML separately selects the
availability source and sets trainer count, aggregation, GPUs, and training
hyperparameters. Setup generates inputs for all orbital satellites. Coverage alone accepts
`--num-satellites`, defaulting to the number of trainers in FMoW's registry. The experiment
trainer count should match; the launcher is not modified by setup.

For setup commands and prerequisites, see [Getting started](getting-started.md).

## Complete example

```yaml
dataset:
  root_dir: lib/python/examples/fmow/data/fmow
  num_classes: 62
  image_size: 224

satellites:
  leo_dir: lib/python/examples/fmow/metadata/leo

capture:
  radius: 15.0

ground_stations:
  num_stations: 8
  seed: 0
  elevation_angle: 10.0
  min_window: 90.0

availability:
  trace_path: lib/python/examples/fmow/metadata/availability_traces/satellite_traces.yaml
```

Dataset, orbital, and availability paths resolve to absolute paths when the
config loads. Relative values are anchored to the Flame repository root,
regardless of the working directory or YAML location. Absolute paths and `~`
paths are also supported. An explicitly supplied `--config` path follows normal
command-line path rules.

## Fields and defaults

| Field | Default | Meaning |
| --- | --- | --- |
| `dataset.root_dir` | `lib/python/examples/fmow/data/fmow` | Dataset download destination and runtime dataset location. |
| `dataset.num_classes` | `62` | Number of output classes for the model. Does not change the downloaded dataset. |
| `dataset.image_size` | `224` | Image transform size used at runtime. Does not change visibility or capture generation. |
| `satellites.leo_dir` | `lib/python/examples/fmow/metadata/leo` | Contains orbital inputs and generated capture and ground-station files. |
| `capture.radius` | `15.0` | Capture radius in kilometers. Controls which images enter each satellite's capture schedule, not ground-station connectivity. |
| `ground_stations.num_stations` | `8` | Number of randomly generated stations. Use a positive integer. |
| `ground_stations.seed` | `0` | Random seed for station placement. Optional even when `ground_stations` is present. |
| `ground_stations.elevation_angle` | `10.0` | Minimum elevation, in degrees, for a satellite to be visible from a station. |
| `ground_stations.min_window` | `90.0` | Required remaining visibility window, in seconds, at a station. Use `0` to disable this cutoff. |
| `availability.trace_path` | `lib/python/examples/fmow/metadata/availability_traces/satellite_traces.yaml` | Output file for generated availability and input file for the runtime satellite reader. |

Station coordinates are sampled uniformly over the sphere. A satellite is
available if at least one station has a qualifying visibility window.
`min_window` shortens the usable end of each station's visible interval; it does
not merely discard short passes. With the default value, the final 90 seconds
of a visible interval are not eligible for starting a task.

## Optional sections

- Omitted fields use the defaults above. An empty config can be written as `{}`.
- Omitting `ground_stations`, or setting it to `null`, skips both station and
  availability generation during `all`. Existing files are left alone.
- `ground_stations: {}` enables both generation steps using all defaults.
- Omitting `availability`, or just its `trace_path`, uses the default trace path.
  Specifying a trace path alone does not enable generation.
- Changing `satellites.leo_dir` does not change the default availability output
  path. Override `availability.trace_path` separately if needed.

For example, this enables generation with eight stations, seed zero, and
visibility unrestricted by angle or minimum window:

```yaml
ground_stations:
  elevation_angle: -90
  min_window: 0
```

This makes satellites eligible throughout the sampled orbital data, but the
generator still emits an unavailable event at the end of the trace. It does not
produce an infinite always-available schedule. Increasing the elevation threshold
reduces visibility; an infinite positive threshold makes no satellites visible.

## How runtime uses this config

The trainer and aggregator receive the config location through their
`fmow_config_path` hyperparameter. When satellite availability is selected in the
experiment, both read `availability.trace_path` from this config.

Setup does not change the experiment's selected availability mode or start
training. If using a custom config file or orbital directory, also align the
runtime `fmow_config_path` and trainer `satellite_coordinates_path` settings.
The latter is a separate Flame hyperparameter and is not automatically updated
by `satellites.leo_dir`.
