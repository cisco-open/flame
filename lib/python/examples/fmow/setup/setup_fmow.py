import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import load_config
from generate_ground_stations import write_ground_stations
from generate_satellite_availability import write_satellite_availability

_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "fmow_config.yaml"
_DOWNLOAD_SCRIPT = Path(__file__).parent / "download_fmow_dataset.sh"

def download_fmow(config: dict) -> None:
    data_dir = Path(config.dataset.root_dir)
    if (data_dir / "rgb_metadata.csv").exists() and \
        (data_dir / "country_code_mapping.csv").exists() and \
        (data_dir / "images").is_dir():
        print(f"[setup_fmow] FMoW dataset already present at {data_dir}, skipped download")
        return
    
    print(f"Downloading FMoW dataset (~54GB), this could take a while...")
    subprocess.run(["bash", str(_DOWNLOAD_SCRIPT), str(data_dir)], check=True)

def main():
    parser = argparse.ArgumentParser(description="Prepare FMoW data and satellite inputs.")
    parser.add_argument(
        "command", nargs="?", default="all",
        choices=["all", "download", "captures", "ground-stations", "availability"],
    )
    parser.add_argument("--config", type=Path, default=_CONFIG_PATH)
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command in ("all", "download"):
        download_fmow(config)
    if args.command in ("all", "captures"):
        from captures import schedule_image_capture
        schedule_image_capture(config)

    leo_dir = Path(config.satellites.leo_dir)
    gs_file = leo_dir / "ground_stations.yaml"
    if args.command in ("all", "ground-stations"):
        if config.ground_stations is not None:
            write_ground_stations(
                config.ground_stations.num_stations, config.ground_stations.seed, gs_file,
            )
        else:
            print("[setup_fmow] No ground_stations settings; skipped ground station generation")

    if args.command in ("all", "availability"):
        generation = config.ground_stations
        if generation is not None:
            write_satellite_availability(
                gs_file, leo_dir / "ecef.npz", Path(config.availability.trace_path),
                generation.elevation_angle, generation.min_window,
            )
        else:
            print("[setup_fmow] No ground_stations settings; skipped trace generation")

if __name__ == "__main__":
    main()
