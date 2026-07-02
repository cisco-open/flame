# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Tests for fwdllm_aggregator.read_trainer_unavailability() -- it must read
availability traces from a `_metadata` bundle (registry + traces), not from
the legacy json_scripts/trainer_*.json directory."""

import glob as glob_module

import yaml

from flame.mode.horizontal.syncfl.fwdllm_aggregator import TopAggregator


def _write_metadata(tmp_path):
    metadata_dir = tmp_path / "_metadata"
    (metadata_dir / "availability_traces").mkdir(parents=True)

    registry = {
        "trainers": {
            "trainer_001": {
                "trainer_id": 1,
                "task_id": "task-aaa",
            },
            "trainer_002": {
                "trainer_id": 2,
                "task_id": "task-bbb",
            },
        }
    }
    (metadata_dir / "trainer_registry.yaml").write_text(yaml.safe_dump(registry))

    mobiperf_traces = {
        "traces": {
            "device_001": {"states_2st": [[0, "AVL_TRAIN"], [100, "UN_AVL"]]},
            "device_002": {"states_2st": [[0, "UN_AVL"], [50, "AVL_TRAIN"]]},
        }
    }
    (metadata_dir / "availability_traces" / "mobiperf_traces.yaml").write_text(
        yaml.safe_dump(mobiperf_traces)
    )
    return metadata_dir


def test_reads_from_metadata_bundle(tmp_path):
    metadata_dir = _write_metadata(tmp_path)

    result = TopAggregator.read_trainer_unavailability(
        None, trace="mobiperf_2st", metadata_dir=metadata_dir
    )

    assert set(result.keys()) == {"task-aaa", "task-bbb"}
    assert dict(result["task-aaa"]) == {0: "AVL_TRAIN", 100: "UN_AVL"}
    assert dict(result["task-bbb"]) == {0: "UN_AVL", 50: "AVL_TRAIN"}


def test_does_not_read_json_scripts(tmp_path, monkeypatch):
    metadata_dir = _write_metadata(tmp_path)

    def fail_glob(*args, **kwargs):
        raise AssertionError("must not glob legacy json_scripts/ directory")

    monkeypatch.setattr(glob_module, "glob", fail_glob)

    result = TopAggregator.read_trainer_unavailability(
        None, trace="mobiperf_2st", metadata_dir=metadata_dir
    )
    assert len(result) == 2
