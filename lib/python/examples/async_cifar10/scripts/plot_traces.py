# plot_traces.py works relative to the directory it needs to be used in
# For example, to plot the traces of the us_border experiment, you would
# go into the us_border directory in the terminal and run:
# python ../../scripts/plot_traces.py

import json
import random
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Function used from compare_parity.py
def load_trainer_jsonl_dir(telemetry_dir: Optional[str]) -> dict:
    """Load all trainer_*.jsonl files from a telemetry dir.

    Returns {short_id: {"task_recv": [...], "trainer_round": [...]}}
    """
    if not telemetry_dir:
        return {}
    d = Path(telemetry_dir)
    result = {}
    for f in sorted(d.glob("trainer_*.jsonl")):
        short_id = f.stem[-4:]  # last 4 hex chars
        task_recv_evs, trainer_round_evs = [], []
        with open(f) as fp:
            for line in fp:
                try:
                    e = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                ev = e.get("event")
                if ev == "task_recv":
                    task_recv_evs.append(e)
                elif ev == "trainer_round":
                    trainer_round_evs.append(e)
        result[short_id] = {
            "task_recv": task_recv_evs,
            "trainer_round": trainer_round_evs,
        }
    return result


def main():
    dir = Path.cwd() / "telemetry"
    if not dir.is_dir():
        print(f"{dir} not found")
        return

    telemetry = load_trainer_jsonl_dir(str(dir))
    if not telemetry:
        return

    plt.figure(figsize=[10, 6])

    for i, (tid, d) in enumerate(telemetry.items()):
        events = d.get("trainer_round", [])
        offset_x = random.uniform(-0.1, 0.1)
        offset_y = random.uniform(-0.1, 0.1)
        x = [e["lon"] + offset_x for e in events]
        y = [e["lat"] + offset_y for e in events]

        plt.plot(
            x,
            y,
            marker="*",
            markersize=1,
            # color=line_color,
            alpha=0.7,
            label=f"Trainer {tid}",
        )

    plt.title("Location Traces")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()

    plots_dir = Path.cwd() / "plots"
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / "location_traces.png", dpi=150)


if __name__ == "__main__":
    main()
