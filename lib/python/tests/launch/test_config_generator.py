# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Tests for ConfigGenerator with the real shared metadata bundle."""

from pathlib import Path

import pytest
import yaml


SHARED_METADATA = Path(__file__).resolve().parents[2] / "examples" / "_metadata"
TRAINER_BASE = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "feddance_cifar10"
    / "configs"
    / "trainer_base.yaml"
)


pytestmark = pytest.mark.skipif(
    not SHARED_METADATA.is_dir() or not TRAINER_BASE.is_file(),
    reason="shared metadata or feddance_cifar10 trainer_base not present",
)


@pytest.fixture
def loader():
    from flame.launch.spawner import MetadataLoader

    return MetadataLoader(SHARED_METADATA)


@pytest.fixture
def gen(loader):
    from flame.launch.spawner import ConfigGenerator

    return ConfigGenerator(loader, TRAINER_BASE)


class TestMetadataLoader:
    def test_registry_loaded(self, loader):
        assert len(loader.trainer_registry) > 0

    def test_dataset_splits_loaded(self, loader):
        assert len(loader.dataset_splits) > 0


class TestConfigGenerator:
    def test_generate_basic(self, gen):
        cfg = gen.generate_trainer_config(
            trainer_id=1, alpha=0.1, availability_mode="syn_0"
        )
        assert "taskid" in cfg
        assert isinstance(cfg["taskid"], str) and len(cfg["taskid"]) > 0
        assert cfg["hyperparameters"]["trainer_indices_list"]
        assert cfg["selector"]["sort"] == "feddance"

    def test_overrides_applied(self, gen):
        cfg = gen.generate_trainer_config(
            trainer_id=2,
            alpha=0.1,
            availability_mode="syn_0",
            **{"job.id": "experiment-xyz", "hyperparameters.batchSize": 64},
        )
        assert cfg["job"]["id"] == "experiment-xyz"
        assert cfg["hyperparameters"]["batchSize"] == 64
