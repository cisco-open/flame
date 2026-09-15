# Estimate training-data coverage

[`scripts/data_coverage.py`](../scripts/data_coverage.py) reads the inputs prepared
by setup and estimates unique captured and reachable FMoW training images.
It accepts `--config` and an optional `--num-satellites`. Satellite count defaults
to the number of trainers in the current FMoW registry; simulation duration comes
from the orbital files.

## Workflow

From the repository root:

```bash
python lib/python/examples/fmow/setup/setup_fmow.py all --config lib/python/examples/fmow/configs/fmow_config.yaml
python lib/python/examples/fmow/scripts/data_coverage.py --config lib/python/examples/fmow/configs/fmow_config.yaml
```

Both commands use the default FMoW config when `--config` is omitted.
For a different scenario, change its config, run setup, then run coverage against
that same config. Use separate output directories to preserve multiple scenarios.

Setup generates data for all input satellites. For coverage of the current
200-trainer experiment, select that subset when analyzing:

```bash
python lib/python/examples/fmow/setup/setup_fmow.py all
python lib/python/examples/fmow/scripts/data_coverage.py --num-satellites 200
```

This selects indices 0 through 199 even when orbital files contain 300 satellites.
There is no satellite-count field in `fmow_config`. Without the parameter, coverage
counts entries in `fmow/metadata/trainer_registry.yaml`, currently 300.
The registry's declared `num_trainers` header is not used; actual entries are counted.
Inputs must contain at least the requested number of satellites. A smaller file
causes an error instead of silently changing the count.
The experiment YAML still needs a matching trainer count; coverage does not read
or modify the experiment YAML.

## Inputs

Python needs `numpy` and `PyYAML`. The config identifies:

| File | Purpose |
| --- | --- |
| `satellites.leo_dir/captures.npz` | Saved image capture times for each satellite. |
| `satellites.leo_dir/geodetic.npz` | Satellite ordering and the full orbital timeline. |
| `availability.trace_path` | Saved ground-station availability events. The default path applies when omitted. |
| `dataset.root_dir/rgb_metadata.csv` | All training-image IDs and the denominator for percentages. |

Ground stations affect coverage through the generated availability trace. The
script does not recalculate visibility, generate station positions, or change
metadata. After changing capture radius or ground-station settings, rerun setup
before analyzing. Include `ground_stations` in the config to enable regeneration
of stations and availability; otherwise setup preserves those existing files.

The script reports the saved trace description. Captures do not record their
generation radius, so it cannot verify that existing captures match the latest
config. Pointing to a new config without regenerating files does not create a
new scenario.

## Results

The table reports every 30 minutes and at the final orbital time, currently
three hours. The denominator is every CSV row labeled `train`.

| Column | Meaning |
| --- | --- |
| Captured | Unique training images captured by the selected satellites before the reporting time. |
| Reachable | Unique captured images with a usable connection on at least one capturing satellite at or after capture, but before the reporting time. |
| No contact yet | Captured images with no qualifying connection yet on any capturing satellite. |

Satellites retain images captured while disconnected. Duplicate captures count
once globally. A later capture by another satellite can provide an earlier
connection than the first capturing satellite. Events exactly at a reporting
boundary count in the following interval.

Reachable coverage is an optimistic upper bound on global-model participation.
It assumes no training or transmission delay and does not model selection or
aggregation. It does not measure accepted updates or model accuracy. The script
prints results without writing output files or running a training experiment.
