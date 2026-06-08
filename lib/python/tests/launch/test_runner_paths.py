# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Tests for ExperimentRunner path resolution (no subprocess spawning)."""

from pathlib import Path

import pytest

from flame.launch.experiment_config import (
    ExampleConfig,
    ExperimentConfig,
    MetadataPaths,
)
from flame.launch.runner import ExperimentRunner


@pytest.fixture
def fake_example_dir(tmp_path):
    ex = tmp_path / "ex"
    (ex / "trainer" / "pytorch").mkdir(parents=True)
    (ex / "aggregator" / "pytorch").mkdir(parents=True)
    (ex / "configs").mkdir()
    return ex


class TestPathResolution:
    def test_defaults(self, fake_example_dir):
        runner = ExperimentRunner(fake_example_dir)
        exp = ExperimentConfig(name="x")
        paths = runner._resolve_example_paths(exp)
        assert paths["example_dir"] == fake_example_dir.resolve()
        assert paths["trainer_main"] == fake_example_dir / "trainer/pytorch/main.py"
        assert paths["trainer_base"] == fake_example_dir / "configs/trainer_base.yaml"
        # aggregator_main is resolved separately (baseline-owned, default fallback)
        # and is no longer part of _resolve_example_paths.
        assert "aggregator_main" not in paths
        assert (
            runner._resolve_aggregator_main(exp, None) == "aggregator/pytorch/main.py"
        )

    def test_example_overrides(self, fake_example_dir):
        runner = ExperimentRunner(fake_example_dir)
        exp = ExperimentConfig(
            name="x",
            example=ExampleConfig(
                trainer_main="trainer/pytorch/main_v2.py",
                aggregator_main="aggregator/pytorch/agg_v2.py",
            ),
        )
        paths = runner._resolve_example_paths(exp)
        assert paths["trainer_main"].name == "main_v2.py"
        # With no baseline owning it, the example-level override wins.
        assert (
            runner._resolve_aggregator_main(exp, None)
            == "aggregator/pytorch/agg_v2.py"
        )

    def test_aggregator_main_baseline_owns(self, fake_example_dir):
        """When a baseline declares example.aggregator_main, it wins."""
        runner = ExperimentRunner(fake_example_dir)
        exp = ExperimentConfig(name="x")
        baseline = {"example": {"aggregator_main": "aggregator/pytorch/oort_agg.py"}}
        assert (
            runner._resolve_aggregator_main(exp, baseline)
            == "aggregator/pytorch/oort_agg.py"
        )

    def test_aggregator_main_baseline_conflict_raises(self, fake_example_dir):
        """A baseline-owned aggregator_main plus an experiment-level override is
        ambiguous and must fail fast."""
        runner = ExperimentRunner(fake_example_dir)
        exp = ExperimentConfig(
            name="x",
            example=ExampleConfig(aggregator_main="aggregator/pytorch/override.py"),
        )
        baseline = {"example": {"aggregator_main": "aggregator/pytorch/oort_agg.py"}}
        with pytest.raises(ValueError):
            runner._resolve_aggregator_main(exp, baseline)

    def test_metadata_dir_override(self, fake_example_dir, tmp_path):
        shared = tmp_path / "shared_meta"
        shared.mkdir()
        runner = ExperimentRunner(fake_example_dir)
        exp = ExperimentConfig(name="x", metadata=MetadataPaths(dir=str(shared)))
        paths = runner._resolve_example_paths(exp)
        assert paths["metadata_dir"] == shared.resolve()
        assert paths["registry_path"] == (shared.resolve() / "trainer_registry.yaml")

    def test_metadata_dir_at_constructor(self, fake_example_dir, tmp_path):
        shared = tmp_path / "shared_meta"
        shared.mkdir()
        runner = ExperimentRunner(fake_example_dir, metadata_dir=shared)
        exp = ExperimentConfig(name="x")
        paths = runner._resolve_example_paths(exp)
        assert paths["metadata_dir"] == shared.resolve()
