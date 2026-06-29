#!/usr/bin/env python3
"""Plot round vs effective_delay_s from trainer telemetry JSONL files.

Groups data by trainer_id (from task_recv events) across all trainer_*.jsonl files,
then plots a single line per unique trainer_id.
"""

import glob
import json
import os
import sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

TELEMETRY_DIR = (
    "/home/dgarg39/flame/lib/python/examples/async_cifar10/"
    "experiments/run_20260621_182051_felix_n10_alpha100_syn20_smoke/telemetry/"
)


def short_id(trainer_id):
    """Return a short, human-friendly label for a trainer_id."""
    return trainer_id[-6:] if len(trainer_id) > 6 else trainer_id


def main():
    pattern = os.path.join(TELEMETRY_DIR, "trainer_*.jsonl")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No trainer_*.jsonl files found in {TELEMETRY_DIR}")
        sys.exit(1)

    print(f"Found {len(files)} trainer files")

    # First pass: discover trainer_id for each file (from task_recv events)
    # Also collect trainer_round data keyed by end_id
    file_to_trainer_id = {}  # filepath -> trainer_id
    endid_to_delay = defaultdict(list)  # end_id -> [(round, delay), ...]

    for filepath in files:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                event = record.get("event")

                # Discover trainer_id from task_recv events
                if event == "task_recv" and "trainer_id" in record:
                    file_to_trainer_id[filepath] = record["trainer_id"]

                # Collect trainer_round data by end_id
                if event == "trainer_round" and "effective_delay_s" in record:
                    end_id = record.get("end_id") or "unknown"
                    endid_to_delay[end_id].append((record["round"], record["effective_delay_s"]))

    # Build mapping: end_id -> trainer_id via file association
    # Each trainer file has a consistent end_id (same as the filename hash)
    # Find which end_ids belong to which file
    endid_to_file = {}
    for filepath in files:
        with open(filepath, "r") as f:
            first_line = f.readline().strip()
            if first_line:
                record = json.loads(first_line)
                end_id = record.get("end_id") or "unknown"
                endid_to_file[end_id] = filepath

    # Now group by trainer_id
    trainer_data = defaultdict(list)  # trainer_id -> [(round, delay), ...]
    for end_id, data in endid_to_delay.items():
        filepath = endid_to_file.get(end_id)
        if filepath and filepath in file_to_trainer_id:
            tid = file_to_trainer_id[filepath]
        else:
            tid = end_id  # fallback to end_id
        trainer_data[tid].extend(data)

    # Sort each trainer's data by round
    for tid in trainer_data:
        trainer_data[tid].sort(key=lambda x: x[0])

    print(f"Found {len(trainer_data)} unique trainer_ids")

    fig, ax = plt.subplots(figsize=(12, 6))

    for tid in sorted(trainer_data.keys()):
        data = trainer_data[tid]
        if not data:
            continue
        rounds = [d[0] for d in data]
        delays = [d[1] for d in data]
        label = short_id(tid)
        print(f"  {label}: {len(data)} data points (rounds {min(rounds)}-{max(rounds)})")
        ax.plot(rounds, delays, marker="o", markersize=3, label=f"{tid} ({label})")

    ax.set_xlabel("Round")
    ax.set_ylabel("Effective Delay (s)")
    ax.set_title("Round vs Effective Delay (per trainer_id)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(os.path.dirname(TELEMETRY_DIR), "trainer_round_vs_delay.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()