# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Logic-level tests for migrate_async_cifar10 (lookups, reconstruction shape)."""

import pytest

from scripts.migrate_async_cifar10 import (
    ALPHA01_TRACE_FIELDS,
    _build_template,
    _diff,
    _lookup_mobiperf,
    _lookup_synthetic,
    _stringify_trace,
    _strip_for_diff,
    _trainer_key,
)


class TestStringifyTrace:
    def test_empty(self):
        assert _stringify_trace([]) == "[]"

    def test_single_event(self):
        assert _stringify_trace([[0, "AVL_TRAIN"]]) == "[(0, 'AVL_TRAIN')]"

    def test_multi_events_match_python_repr(self):
        lol = [[0, "AVL_TRAIN"], [300, "UN_AVL"]]
        assert _stringify_trace(lol) == "[(0, 'AVL_TRAIN'), (300, 'UN_AVL')]"

    def test_roundtrip_via_repr(self):
        import ast
        original = [[0, "A"], [10, "B"], [20, "C"]]
        s = _stringify_trace(original)
        parsed = ast.literal_eval(s)
        assert [list(t) for t in parsed] == original


class TestTrainerKey:
    def test_padding(self):
        assert _trainer_key(1) == "trainer_001"
        assert _trainer_key(99) == "trainer_099"
        assert _trainer_key(300) == "trainer_300"


class TestLookupMobiperf:
    def test_lookup(self):
        mobiperf = {"traces": {"device_001": {"states_2st": [[0, "X"]]}}}
        assert _lookup_mobiperf(mobiperf, 1, "states_2st") == [[0, "X"]]


class TestLookupSynthetic:
    def test_uniform_fallback(self):
        synthetic = {"traces": {"syn_0": {"pattern": [[0, "A"]]}}}
        assert _lookup_synthetic(synthetic, "syn_0", 5) == [[0, "A"]]

    def test_per_trainer_preferred(self):
        synthetic = {"traces": {"syn_20": {
            "pattern": [[0, "UNIFORM"]],
            "per_trainer": {"n300": {"trainer_005": [[0, "SPECIFIC"]]}},
        }}}
        assert _lookup_synthetic(synthetic, "syn_20", 5) == [[0, "SPECIFIC"]]

    def test_per_trainer_falls_back_when_trainer_missing(self):
        synthetic = {"traces": {"syn_20": {
            "pattern": [[0, "UNIFORM"]],
            "per_trainer": {"n300": {"trainer_001": [[0, "SPECIFIC"]]}},
        }}}
        # trainer_002 not present in per_trainer -> fall back to pattern
        assert _lookup_synthetic(synthetic, "syn_20", 2) == [[0, "UNIFORM"]]


class TestBuildTemplate:
    def test_replaces_per_trainer_fields_with_placeholders(self):
        sample = {
            "taskid": "abc",
            "backend": "mqtt",
            "hyperparameters": {
                "batchSize": 32,
                "training_delay_s": "4.0",
                "trainer_indices_list": [1, 2, 3],
                "avl_events_syn_0": "[(0, 'AVL_TRAIN')]",
            },
            "role": "trainer",
        }
        t = _build_template(sample)
        assert t["taskid"] == "__TASKID__"
        assert t["backend"] == "mqtt"
        assert t["hyperparameters"]["batchSize"] == 32
        assert t["hyperparameters"]["training_delay_s"] is None
        assert t["hyperparameters"]["trainer_indices_list"] is None
        assert t["hyperparameters"]["avl_events_syn_0"] is None
        assert t["role"] == "trainer"

    def test_preserves_key_order(self):
        sample = {"c": 1, "a": 2, "b": 3, "taskid": "x"}
        t = _build_template(sample)
        # taskid stays at its original position
        assert list(t.keys()) == ["c", "a", "b", "taskid"]


class TestDiff:
    def test_no_diff(self):
        assert _diff({"a": 1}, {"a": 1}) == {}

    def test_value_diff(self):
        assert _diff({"a": 2}, {"a": 1}) == {"a": 2}

    def test_nested_diff(self):
        a = {"hp": {"x": 1, "y": 2}, "k": "v"}
        b = {"hp": {"x": 1, "y": 3}, "k": "v"}
        assert _diff(a, b) == {"hp": {"y": 2}}

    def test_missing_key_in_b(self):
        assert _diff({"a": 1, "b": 2}, {"a": 1}) == {"b": 2}


class TestStripForDiff:
    def test_drops_taskid_and_per_trainer_hp(self):
        d = {
            "taskid": "x",
            "backend": "mqtt",
            "hyperparameters": {
                "batchSize": 32,
                "training_delay_s": "4.0",
                "trainer_indices_list": [1],
                "avl_events_syn_0": "...",
                "use_oort_loss_fn": "True",
            },
        }
        s = _strip_for_diff(d)
        assert "taskid" not in s
        assert s["backend"] == "mqtt"
        assert s["hyperparameters"] == {
            "batchSize": 32,
            "use_oort_loss_fn": "True",
        }


class TestAlpha01TraceFieldsConsistent:
    def test_all_avl_keys_have_lookups(self):
        for k in ALPHA01_TRACE_FIELDS:
            kind, sub = ALPHA01_TRACE_FIELDS[k]
            assert kind in ("mobiperf", "synthetic")
            assert isinstance(sub, str) and sub
