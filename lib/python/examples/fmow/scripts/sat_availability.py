import argparse
import numpy as np
import yaml

from pathlib import Path

EARTH_RADIUS_KM = 6371.0
FMOW_DIR = Path(__file__).resolve().parent.parent
METADATA_DIR = FMOW_DIR / "metadata"
LEO_DIR = METADATA_DIR / "leo"
OUTPUT_DIR = METADATA_DIR / "availability_traces" 
FILENAME = "satellite_traces.yaml"
GS_FILE = "ground_stations.yaml"

def geo_to_ecef(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.array([
        EARTH_RADIUS_KM * np.cos(lat) * np.cos(lon),
        EARTH_RADIUS_KM * np.cos(lat) * np.sin(lon),
        EARTH_RADIUS_KM * np.sin(lat)
    ])

def find_runs(available: np.ndarray):
    # Find runs of availability and record timesteps where they start/end
    padded = np.concatenate(([False], available, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1) - 1
    return list(zip(starts, ends))

def elevation_angles_deg(station_ecef: np.ndarray, sat_ecef: np.ndarray) -> np.ndarray:
    # Calculate availability for every satellite based on a single ground station
    d = sat_ecef - station_ecef
    d_norm = np.linalg.norm(d, axis=-1)
    up = station_ecef / np.linalg.norm(station_ecef)
    sin_elev = np.sum(d*up, axis=-1) / d_norm
    return np.degrees(np.arcsin(np.clip(sin_elev, -1.0, 1.0)))

def find_satellite_events(
    visible: np.ndarray, min_window: float
) -> np.ndarray:
    events = np.zeros_like(visible)

    for sat_idx in range(visible.shape[1]):
        for start, end in find_runs(visible[:, sat_idx]):
            # Find all times which are visible for at 
            # least min_window
            avail_end = end - int(min_window)
            if avail_end < start:
                continue
            events[start:avail_end+1, sat_idx] = True

    return events
    

def convert_to_trace(
    visible: np.ndarray
) -> list:
    events = [(0, "UN_AVL")]
    for start, end in find_runs(visible):
        events.append((int(start), "AVL_TRAIN"))
        events.append((int(end + 1), "UN_AVL"))


    if len(events) > 1 and events[1][0] == events[0][0]:
        events = events[1:]
    return events

def generate_trace(
    ground_stations: dict, sat_ecef: np.ndarray,
    min_elevation: float, min_window: float
) -> tuple[dict, dict]:
    num_timesteps, num_satellites, _ = sat_ecef.shape

    visible = np.zeros((num_timesteps, num_satellites), dtype=bool)
    for station in ground_stations.values():
        station["ecef"] = geo_to_ecef(station["lat"], station["lon"])
        elevation = elevation_angles_deg(station["ecef"], sat_ecef)
        station["visible"] = elevation >= min_elevation
        visible |= find_satellite_events(station["visible"], min_window)

    satellite_events = {}
    for sat_idx in range(num_satellites):
        satellite_events[sat_idx] = convert_to_trace(
            visible[:, sat_idx],
        )
        
    eligible_over_time = visible.sum(axis=1)
    never_eligible = sum(
        1 for events in satellite_events.values()
        if not any(state == "AVL_TRAIN" for _, state in events)
    )

    stats = {
        "num_satellites": num_satellites,
        "eligible_min": int(eligible_over_time.min()),
        "eligible_max": int(eligible_over_time.max()),
        "eligible_mean": float(eligible_over_time.mean()),
        "ghost_satellites": never_eligible
    }
    return satellite_events, stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gs-file", type=Path, default=LEO_DIR / GS_FILE)
    parser.add_argument("--elevation-angle", type=float, default=10.0)
    parser.add_argument("--min-window", type=float, default=90.0)
    parser.add_argument("--dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--filename", type=Path, default=FILENAME)
    args = parser.parse_args()

    with open(args.gs_file) as f:
        gs_data = yaml.safe_load(f)
    ground_stations = gs_data["stations"]

    ecef = np.load(LEO_DIR / "ecef.npz")
    sat_ecef, _ = ecef["ecef_km"], ecef["time_s"]

    satellite_events, stats = generate_trace(
        ground_stations, sat_ecef, args.elevation_angle, args.min_window
    )

    print(f"Ground stations: {len(ground_stations)}, satellites: {stats['num_satellites']}")
    print(f"Eligible at once: min={stats['eligible_min']}, mean={stats['eligible_mean']}, max={stats['eligible_max']}")
    print(f"Satellites with zero usable passes: {stats['ghost_satellites']}")

    trainers = {}
    for sat_idx, event in enumerate(satellite_events.values()):
        events = satellite_events.get(sat_idx, [(0, "UN_AVL")])
        trainer_key = f"trainer_{(sat_idx + 1):03d}"
        trainers[trainer_key] = [list(e) for e in events]
        
    output = {
        "description": (
            f"Geometric availability trace: {len(ground_stations)} ground stations, "
            f"min_elevation_deg={args.elevation_angle}, min_window_s={args.min_window}"
        ),
        "generated_from": {
            # "ground_stations": str(args.gs_file),
            "min_elevation_angle": args.elevation_angle,
            "min_window": args.min_window,
        },
        "trainers": trainers,
    }
    
    out_file = args.dir / args.filename
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        yaml.safe_dump(output, f, sort_keys=False, default_flow_style=None)

    print(f"Wrote geometric availability trace to {out_file}")
        

if __name__ == "__main__":
    main()