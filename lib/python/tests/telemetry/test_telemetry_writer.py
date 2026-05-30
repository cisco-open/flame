# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Tests for the process-global telemetry writer."""

import json
from datetime import timedelta

import pytest

from flame import telemetry


@pytest.fixture(autouse=True)
def _reset_telemetry():
    """Ensure each test starts/ends with telemetry disabled."""
    telemetry.shutdown()
    yield
    telemetry.shutdown()


def _read_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


class TestDisabledByDefault:
    def test_emit_is_noop_when_unconfigured(self, monkeypatch):
        monkeypatch.delenv(telemetry.ENV_DIR, raising=False)
        assert telemetry.is_enabled() is False
        # must not raise
        telemetry.emit("selection", round=1)

    def test_configure_returns_none_without_dir(self, monkeypatch):
        monkeypatch.delenv(telemetry.ENV_DIR, raising=False)
        assert telemetry.configure(role="trainer", end_id="1") is None
        assert telemetry.is_enabled() is False


class TestWriting:
    def test_configure_and_emit_roundtrip(self, tmp_path):
        w = telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        assert w is not None
        assert telemetry.is_enabled()
        telemetry.emit("agg_eval", round=3, **{"test-accuracy": 0.5})
        telemetry.shutdown()

        recs = _read_jsonl(tmp_path / "aggregator.jsonl")
        assert len(recs) == 1
        r = recs[0]
        assert r["event"] == "agg_eval"
        assert r["role"] == "aggregator"
        assert r["round"] == 3
        assert r["test-accuracy"] == 0.5
        assert "ts" in r

    def test_end_id_in_filename_and_record(self, tmp_path):
        telemetry.configure(role="trainer", end_id="t7", run_dir=str(tmp_path))
        telemetry.emit("trainer_round", round=1, real_gpu_time_s=2.0)
        telemetry.shutdown()
        recs = _read_jsonl(tmp_path / "trainer_t7.jsonl")
        assert recs[0]["end_id"] == "t7"

    def test_env_var_configures_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv(telemetry.ENV_DIR, str(tmp_path))
        w = telemetry.configure(role="aggregator")
        assert w is not None
        assert telemetry.get_run_dir() == str(tmp_path)

    def test_serializes_timedelta_set_and_numpy(self, tmp_path):
        import numpy as np

        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        telemetry.emit(
            "agg_round",
            round=1,
            duration=timedelta(seconds=2.5),
            chosen={"b", "a"},
            util=np.float64(1.25),
        )
        telemetry.shutdown()
        recs = _read_jsonl(tmp_path / "aggregator.jsonl")
        r = recs[0]
        assert r["duration"] == 2.5
        assert sorted(r["chosen"]) == ["a", "b"]
        assert r["util"] == 1.25
