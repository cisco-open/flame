"""Read generated satellite availability for FMoW trainers and aggregators."""

from pathlib import Path

import yaml
from sortedcontainers import SortedDict


class SatelliteAvailability:
    """Load a satellite trace once and expose Flame's two event formats.

    Trace timestamps are seconds from simulation start. Satellite index 0 maps
    to ``trainer_001``. Events are read as generated, including any final
    unavailable event; this reader does not extend or repeat the schedule.
    """

    def __init__(self, trace_path: str | Path):
        with open(trace_path, encoding="utf-8") as f:
            self._events = yaml.safe_load(f)["trainers"]

    def events_for_satellite(self, satellite_index: int) -> list:
        """Return a fresh event list for the trainer to consume with pop(0).

        Missing satellites raise KeyError rather than becoming available by
        default. Copy each event too, so callers cannot modify the loaded trace.
        """
        trainer_key = f"trainer_{satellite_index + 1:03d}"
        return [[float(ts), state] for ts, state in self._events[trainer_key]]

    def events_by_endpoint(self, registry_path: str | Path) -> dict:
        """Return endpoint task_id -> SortedDict(timestamp -> state).

        Include satellites present in the trace, resolving their endpoint IDs
        through the existing trainer registry. The registry may contain other
        trainers; a traced satellite missing from the registry raises KeyError.
        """
        with open(registry_path, encoding="utf-8") as f:
            registry = yaml.safe_load(f)["trainers"]

        return {
            registry[trainer_key]["task_id"]: SortedDict(
                (float(ts), state) for ts, state in events
            )
            for trainer_key, events in self._events.items()
        }
