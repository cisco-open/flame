# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Tests for flame.launch.experiment_config."""

import textwrap
from pathlib import Path

import pytest

from flame.launch.experiment_config import (
    ExampleConfig,
    ExperimentBatch,
    MetadataPaths,
    load_experiment_config,
)


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "exp.yaml"
    p.write_text(textwrap.dedent(content))
    return p


class TestExperimentConfig:
    def test_minimal_yaml(self, tmp_path):
        path = _write(tmp_path, """\
            experiments:
              - name: tiny
                trainer:
                  num_trainers: 3
                aggregator:
                  config_template: agg.json
        """)
        batch = load_experiment_config(path)
        assert len(batch.experiments) == 1
        e = batch.experiments[0]
        assert e.name == "tiny"
        assert e.trainer.num_trainers == 3
        assert isinstance(e.example, ExampleConfig)
        assert isinstance(e.metadata, MetadataPaths)

    def test_example_and_metadata_sections(self, tmp_path):
        path = _write(tmp_path, """\
            experiments:
              - name: with_overrides
                example:
                  trainer_main: trainer/pytorch/custom_main.py
                  trainer_base: configs/custom_base.yaml
                metadata:
                  dir: examples/_metadata
                  registry: shared_registry.yaml
                trainer:
                  num_trainers: 5
                aggregator:
                  config_template: agg.json
        """)
        batch = load_experiment_config(path)
        e = batch.experiments[0]
        assert e.example.trainer_main == "trainer/pytorch/custom_main.py"
        assert e.example.trainer_base == "configs/custom_base.yaml"
        assert e.metadata.dir == "examples/_metadata"
        assert e.metadata.registry == "shared_registry.yaml"

    def test_defaults_when_sections_absent(self, tmp_path):
        path = _write(tmp_path, """\
            experiments:
              - name: defaults
                trainer:
                  num_trainers: 1
                aggregator:
                  config_template: agg.json
        """)
        batch = load_experiment_config(path)
        e = batch.experiments[0]
        assert e.example.trainer_main == "trainer/pytorch/main.py"
        assert e.example.aggregator_main == "aggregator/pytorch/main.py"
        assert e.example.trainer_base == "configs/trainer_base.yaml"

    def test_batch_of_experiments(self, tmp_path):
        path = _write(tmp_path, """\
            experiments:
              - name: a
                trainer:
                  num_trainers: 1
                aggregator:
                  config_template: a.json
              - name: b
                trainer:
                  num_trainers: 2
                aggregator:
                  config_template: b.json
        """)
        batch = load_experiment_config(path)
        assert [e.name for e in batch.experiments] == ["a", "b"]


class TestLogPrefix:
    def test_includes_selector_and_n(self, tmp_path):
        path = _write(tmp_path, """\
            experiments:
              - name: x
                trainer:
                  num_trainers: 42
                aggregator:
                  config_template: a.json
                  selector: feddance
                  tracking_mode: oracular
        """)
        e = load_experiment_config(path).experiments[0]
        prefix = e.get_log_prefix(timestamp_str="00_00_00_00_00")
        assert "feddance" in prefix
        assert "n42" in prefix
        assert "oracular" in prefix
