"""Estimate unique captured and ground-station-reachable FMoW training data.

Reads existing captures. Reachability is an opportunity upper bound, not a
prediction of which training updates a selector or aggregator will accept.
"""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import yaml

FMOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FMOW_DIR / "setup"))
from config import load_config
REGISTRY_PATH = FMOW_DIR / "metadata/trainer_registry.yaml"


def resolve_num_satellites(num_satellites: int | None = None) -> int:
    """Use an explicit coverage count, or the number of trainer registry entries."""
    if num_satellites is None:
        with REGISTRY_PATH.open() as f:
            num_satellites = len(yaml.safe_load(f)["trainers"])
    if num_satellites <= 0:
        raise ValueError("The satellite count must be positive")
    return num_satellites


def next_contact_times(capture_times: np.ndarray, events: list) -> np.ndarray:
    """Earliest AVL_TRAIN time at or after each capture; infinity if none.

    A transition applies at its timestamp. An image captured while disconnected
    remains eligible for a later contact. These satellite traces have two states.
    """
    times = np.array([event[0] for event in events], dtype=float)
    states = np.array([event[1] for event in events])
    if not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError("Availability events must be nonempty and strictly time ordered")
    if not np.isfinite(times).all() or not np.isin(states, ["AVL_TRAIN", "UN_AVL"]).all():
        raise ValueError("Satellite traces require finite times and AVL_TRAIN/UN_AVL states")
    starts = times[states == "AVL_TRAIN"]
    next_indices = np.searchsorted(starts, capture_times, side="left")
    result = np.full(len(capture_times), np.inf)
    has_next = next_indices < len(starts)
    result[has_next] = starts[next_indices[has_next]]
    current = np.searchsorted(times, capture_times, side="right") - 1
    connected = (current >= 0) & (states[np.maximum(current, 0)] == "AVL_TRAIN")
    result[connected] = capture_times[connected]
    return result


def coverage_times(events, offsets, satellite_indices, traces, training_rows):
    """Reduce all satellite copies to the earliest capture/contact per image."""
    captured = np.full(len(training_rows), np.inf)
    reachable = np.full(len(training_rows), np.inf)
    for index in satellite_indices:
        local = events[int(offsets[index]):int(offsets[index + 1])]
        image_ids = local[:, 1].astype(np.int64)
        if np.any(image_ids < 0) or np.any(image_ids >= len(training_rows)):
            raise ValueError("Capture image IDs are outside rgb_metadata.csv")
        if not training_rows[image_ids].all():
            raise ValueError("Capture file contains images outside the training split")
        times = local[:, 0].astype(float)
        if np.any(times < 0) or not np.isfinite(times).all():
            raise ValueError("Capture times must be finite and nonnegative")
        contacts = next_contact_times(times, traces[index])
        np.minimum.at(captured, image_ids, times)
        np.minimum.at(reachable, image_ids, contacts)
    return captured[training_rows], reachable[training_rows]


def coverage_rows(captured, reachable, checkpoints):
    """Count unique training images strictly before each elapsed-time boundary."""
    total = len(captured)
    for time_s in checkpoints:
        n_captured = int(np.count_nonzero(captured < time_s))
        n_reachable = int(np.count_nonzero(reachable < time_s))
        yield {
            "time_s": float(time_s),
            "captured_images": n_captured,
            "captured_percent": 100 * n_captured / total,
            "reachable_images": n_reachable,
            "reachable_percent": 100 * n_reachable / total,
            "captured_without_contact_images": n_captured - n_reachable,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=FMOW_DIR / "configs/fmow_config.yaml")
    parser.add_argument("--num-satellites", type=int, help="Default: number of trainers in FMoW's registry")
    args = parser.parse_args()
    config = load_config(args.config)
    leo_dir = Path(config.satellites.leo_dir)
    with np.load(leo_dir / "captures.npz") as captures:
        events, offsets = captures["events"], captures["offsets"]
        names = captures["sat_names"]
    with np.load(leo_dir / "geodetic.npz") as geo:
        orbit_times = geo["time_s"]
        if not np.array_equal(names, geo["sat_names"][:len(names)]):
            parser.error("Capture and geodetic satellite ordering does not match")
    if len(orbit_times) < 2 or np.any(np.diff(orbit_times) <= 0):
        parser.error("Orbital timestamps must be increasing and contain at least two samples")
    horizon = float(orbit_times[-1] + (orbit_times[-1] - orbit_times[-2]))
    if not np.isfinite(horizon) or horizon <= 0:
        parser.error("Orbital duration must be finite and positive")
    count = resolve_num_satellites(args.num_satellites)
    if not 0 < count <= len(offsets) - 1:
        parser.error("Requested satellite count exceeds captures.npz; regenerate setup or pass --num-satellites")
    indices = range(count)
    with open(config.availability.trace_path) as f:
        saved = yaml.safe_load(f)
    traces = {i: saved["trainers"][f"trainer_{i + 1:03d}"] for i in indices}
    source = f"{config.availability.trace_path}: {saved.get('description', '')}"

    metadata_path = Path(config.dataset.root_dir) / "rgb_metadata.csv"
    with metadata_path.open(newline="") as f:
        training_rows = np.array([row["split"] == "train" for row in csv.DictReader(f)])
    if not training_rows.any():
        parser.error("rgb_metadata.csv has no training rows")
    captured, reachable = coverage_times(events, offsets, indices, traces, training_rows)
    checkpoints = np.append(np.arange(1800, horizon, 1800), horizon)
    rows = list(coverage_rows(captured, reachable, checkpoints))
    print(f"Training images: {len(captured):,}; satellite indices: {indices.start}..{indices.stop - 1}")
    print(f"Connectivity: {source}")
    print(f"Captures: {leo_dir / 'captures.npz'} (saved events; generation radius is not recorded)")
    print("Run setup_fmow all with this config after changing capture or ground-station settings.")
    print("Reachable coverage is an upper bound: no selection, training duration, or aggregation is modeled.")
    print("Counts include events strictly before each elapsed-time boundary; input files are not modified.\n")
    print(f"{'Hours':>7} {'Captured':>12} {'% train':>9} {'Reachable':>12} {'% train':>9} {'No contact yet':>16}")
    for row in rows:
        print(f"{row['time_s'] / 3600:7.2f} {row['captured_images']:12,d} {row['captured_percent']:9.2f} "
              f"{row['reachable_images']:12,d} {row['reachable_percent']:9.2f} {row['captured_without_contact_images']:16,d}")


if __name__ == "__main__":
    main()
