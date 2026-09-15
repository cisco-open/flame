"""Coverage accounting across captures, disconnected periods, and replicas."""

from pathlib import Path
import sys
import unittest
import tempfile
import io
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_coverage import coverage_rows, coverage_times, next_contact_times
from data_coverage import main
import data_coverage


class CoverageTests(unittest.TestCase):
    def test_default_count_reads_registry_and_override_takes_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = Path(directory) / "registry.yaml"
            registry.write_text("trainers:\n  trainer_001: {}\n  trainer_002: {}\n")
            with patch.object(data_coverage, "REGISTRY_PATH", registry):
                self.assertEqual(data_coverage.resolve_num_satellites(), 2)
                self.assertEqual(data_coverage.resolve_num_satellites(1), 1)

    def test_parameter_selects_satellites_and_config_selects_saved_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "captures.npz", events=[[0, 0], [0, 1]],
                     offsets=[0, 1, 2], sat_names=["a", "b"])
            np.savez(root / "geodetic.npz", time_s=[0, 1], sat_names=["a", "b", "c"])
            (root / "rgb_metadata.csv").write_text("split\ntrain\ntrain\n")
            (root / "trace.yaml").write_text("trainers:\n  trainer_001: [[0, UN_AVL]]\n")
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump({
                "dataset": {"root_dir": str(root)},
                "satellites": {"leo_dir": str(root)},
                "availability": {"trace_path": str(root / "trace.yaml")},
            }))
            output = io.StringIO()
            with patch.object(sys, "argv", ["data_coverage", "--config", str(config), "--num-satellites", "1"]), redirect_stdout(output):
                main()
            self.assertIn("satellite indices: 0..0", output.getvalue())
            self.assertIn("50.00", output.getvalue())
            self.assertNotIn("100.00", output.getvalue())

    def test_contacts_respect_transitions_and_keep_disconnected_captures(self):
        events = [[0, "UN_AVL"], [10, "AVL_TRAIN"], [20, "UN_AVL"],
                  [30, "AVL_TRAIN"], [40, "UN_AVL"]]
        actual = next_contact_times(np.array([0, 9, 10, 19, 20, 29, 30, 40]), events)
        np.testing.assert_array_equal(actual, [10, 10, 10, 19, 30, 30, 30, np.inf])

    def test_replicas_count_once_and_later_capture_can_reach_first(self):
        events = np.array([[1, 0], [2, 1], [3, 0], [4, 2]])
        traces = {0: [[0, "UN_AVL"], [10, "AVL_TRAIN"]],
                  1: [[0, "AVL_TRAIN"]]}
        captured, reachable = coverage_times(
            events, np.array([0, 2, 4]), range(2), traces,
            np.array([True, True, True, True, False]),
        )
        np.testing.assert_array_equal(captured, [1, 2, 4, np.inf])
        np.testing.assert_array_equal(reachable, [3, 10, 4, np.inf])
        rows = list(coverage_rows(captured, reachable, [4, 10, 11]))
        self.assertEqual(rows[0]["captured_images"], 2)
        self.assertEqual(rows[0]["reachable_images"], 1)
        self.assertEqual(rows[1]["reachable_percent"], 50)
        self.assertEqual(rows[2]["reachable_percent"], 75)

    def test_selected_subset_excludes_other_satellites(self):
        captured, reachable = coverage_times(
            np.array([[1, 0], [2, 1]]), np.array([0, 1, 2]), [1],
            {1: [[0, "UN_AVL"]]}, np.array([True, True]),
        )
        np.testing.assert_array_equal(captured, [np.inf, 2])
        self.assertTrue(np.isinf(reachable).all())

    def test_nontraining_capture_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside the training split"):
            coverage_times(np.array([[1, 1]]), np.array([0, 1]), [0],
                           {0: [[0, "AVL_TRAIN"]]}, np.array([True, False]))


if __name__ == "__main__":
    unittest.main()
