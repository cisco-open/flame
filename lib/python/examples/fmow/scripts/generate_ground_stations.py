# Generate ground stations for availability trace

import argparse
import numpy as np
import yaml

from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "metadata" / "leo"
FILENAME = "ground_stations.yaml"

def generate_ground_stations(num_stations: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)

    lon = rng.uniform(-180.0, 180.0, size=num_stations)
    lat = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=num_stations)))

    stations = {
        f"gs_{i:02d}": {
            "lat": round(float(lat[i]), 4),
            "lon": round(float(lon[i]), 4)
        }
        for i in range(num_stations)
    }

    return {
        "description": (
            f"Synthetic ground station locations (random uniform-distribution, "
            f"seed={seed})"
        ),
        "seed": seed,
        "num_stations": num_stations,
        "stations": stations
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-stations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dir", type=Path, default=OUT_DIR)
    parser.add_argument("--filename", type=Path, default=FILENAME)
    args = parser.parse_args()

    data = generate_ground_stations(args.num_stations, args.seed)

    args.dir.mkdir(parents=True, exist_ok=True)
    with open(args.dir / args.filename, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"Wrote {args.num_stations} ground stations (seed={args.seed}) to {args.dir / args.filename}")

if __name__ == "__main__":
    main()