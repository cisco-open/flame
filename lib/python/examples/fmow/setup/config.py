from dataclasses import dataclass, field
from pathlib import Path
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[5]

@dataclass
class DatasetConfig:
    root_dir: str = "lib/python/examples/fmow/data/fmow"
    num_classes: int = 62
    image_size: int = 224

@dataclass
class SatellitesConfig:
    leo_dir: str = "lib/python/examples/fmow/metadata/leo"

@dataclass
class CaptureConfig:
    radius: float = 15.0

@dataclass
class GroundStationsConfig:
    num_stations: int = 8
    seed: int = 0
    elevation_angle: float = 10.0
    min_window: float = 90.0

@dataclass
class AvailabilityConfig:
    trace_path: str = "lib/python/examples/fmow/metadata/availability_traces/satellite_traces.yaml"

@dataclass
class FMoWConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    satellites: SatellitesConfig = field(default_factory=SatellitesConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    availability: AvailabilityConfig = field(default_factory=AvailabilityConfig)
    ground_stations: GroundStationsConfig | None = None

    def __post_init__(self):
        """Anchor input/output locations to the repository, not the process cwd."""
        self.dataset.root_dir = str((_REPO_ROOT / Path(self.dataset.root_dir).expanduser()).resolve())
        self.satellites.leo_dir = str((_REPO_ROOT / Path(self.satellites.leo_dir).expanduser()).resolve())
        self.availability.trace_path = str((_REPO_ROOT / Path(self.availability.trace_path).expanduser()).resolve())

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FMoWConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        
        dataset_data = data.get("dataset", {})
        satellites_data = data.get("satellites", {})
        capture_data = data.get("capture", {})
        availability_data = data.get("availability") or {}
        ground_stations_data = data.get("ground_stations")

        return cls(
            dataset=DatasetConfig(**dataset_data)if dataset_data else DatasetConfig(),
            satellites=SatellitesConfig(**satellites_data) if satellites_data else SatellitesConfig(),
            capture=CaptureConfig(**capture_data) if capture_data else CaptureConfig(),
            availability=AvailabilityConfig(**availability_data),
            ground_stations=GroundStationsConfig(**ground_stations_data) if ground_stations_data is not None else None,
        )

def load_config(path: str | Path) -> FMoWConfig:
    return FMoWConfig.from_yaml(path)