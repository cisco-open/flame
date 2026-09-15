"""Check optional setup generation without downloading the FMoW dataset."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch
from contextlib import chdir

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup_fmow
from config import FMoWConfig, load_config


class SetupFMoWTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.leo = self.root / "orbits"
        self.leo.mkdir()
        self.trace = self.root / "traces" / "custom.yaml"
        self.config = self.root / "config.yaml"
        self.settings = {
            "satellites": {"leo_dir": str(self.leo)},
            "availability": {"trace_path": str(self.trace)},
        }
        np.savez(
            self.leo / "ecef.npz",
            ecef_km=np.tile([[7000., 0., 0.], [-7000., 0., 0.]], (4, 1, 1)),
            time_s=np.arange(4),
        )

    def run_setup(self, command="all"):
        self.config.write_text(yaml.safe_dump(self.settings))
        captures = ModuleType("captures")
        captures.schedule_image_capture = Mock()
        with patch.dict(sys.modules, {"captures": captures}), \
             patch.object(setup_fmow, "download_fmow") as download, \
             patch.object(sys, "argv", ["setup_fmow", command, "--config", str(self.config)]):
            setup_fmow.main()
        return download, captures.schedule_image_capture

    def test_all_generates_stations_before_trace_at_configured_paths(self):
        self.settings["ground_stations"] = {
            "num_stations": 2, "seed": 7,
            "elevation_angle": -90, "min_window": 0,
        }
        download, captures = self.run_setup()
        download.assert_called_once()
        captures.assert_called_once()
        stations = yaml.safe_load((self.leo / "ground_stations.yaml").read_text())
        self.assertEqual(stations["num_stations"], 2)
        self.assertEqual(stations["seed"], 7)
        trace = yaml.safe_load(self.trace.read_text())
        self.assertEqual(trace["trainers"], {
            "trainer_001": [[0, "AVL_TRAIN"], [4, "UN_AVL"]],
            "trainer_002": [[0, "AVL_TRAIN"], [4, "UN_AVL"]],
        })

    def test_omitted_generation_preserves_existing_files(self):
        station_path = self.leo / "ground_stations.yaml"
        station_path.write_text("existing stations")
        self.trace.parent.mkdir()
        self.trace.write_text("existing trace")
        self.run_setup()
        self.assertEqual(station_path.read_text(), "existing stations")
        self.assertEqual(self.trace.read_text(), "existing trace")
        config = load_config(self.config)
        self.assertIsNone(config.ground_stations)

    def test_availability_only_uses_existing_stations_and_configured_angle(self):
        station_path = self.leo / "ground_stations.yaml"
        station_path.write_text("stations:\n  fixed: {lat: 0, lon: 0}\n")
        self.settings["ground_stations"] = {
            "elevation_angle": 10, "min_window": 1,
        }
        download, captures = self.run_setup("availability")
        download.assert_not_called()
        captures.assert_not_called()
        self.assertEqual(station_path.read_text(), "stations:\n  fixed: {lat: 0, lon: 0}\n")
        trace = yaml.safe_load(self.trace.read_text())["trainers"]
        self.assertEqual(trace["trainer_001"], [[0, "AVL_TRAIN"], [3, "UN_AVL"]])
        self.assertEqual(trace["trainer_002"], [[0, "UN_AVL"]])

    def test_optional_seed_and_output_path_use_defaults(self):
        self.settings.pop("availability")
        self.settings["ground_stations"] = {"num_stations": 2}
        with patch.object(setup_fmow, "write_satellite_availability") as write_trace:
            self.run_setup()
        stations = yaml.safe_load((self.leo / "ground_stations.yaml").read_text())
        self.assertEqual(stations["seed"], 0)
        write_trace.assert_called_once_with(
            self.leo / "ground_stations.yaml", self.leo / "ecef.npz",
            Path(FMoWConfig().availability.trace_path), 10.0, 90.0,
        )

    def test_paths_do_not_depend_on_working_directory(self):
        config_path = Path(setup_fmow._CONFIG_PATH).resolve()
        expected = load_config(config_path)
        with chdir(self.root):
            actual = load_config(config_path)
        self.assertEqual(actual, expected)
        for location in [actual.dataset.root_dir, actual.satellites.leo_dir,
                         actual.availability.trace_path]:
            self.assertTrue(Path(location).is_absolute())

    def test_absolute_custom_paths_are_preserved(self):
        self.config.write_text(yaml.safe_dump(self.settings))
        with chdir(self.root):
            config = load_config(self.config)
        self.assertEqual(Path(config.satellites.leo_dir), self.leo.resolve())
        self.assertEqual(Path(config.availability.trace_path), self.trace.resolve())


if __name__ == "__main__":
    unittest.main()
