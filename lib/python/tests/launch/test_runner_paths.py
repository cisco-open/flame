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
        assert paths["aggregator_main"] == fake_example_dir / "aggregator/pytorch/main.py"
        assert paths["trainer_base"] == fake_example_dir / "configs/trainer_base.yaml"

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
        assert paths["aggregator_main"].name == "agg_v2.py"

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
