# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Tests for baseline catalog + deep-merge + provenance tracking."""

from pathlib import Path

import pytest
import yaml

from flame.launch.baselines import (
    deep_merge,
    format_provenance,
    load_baselines,
    merge_with_provenance,
)


class TestDeepMerge:
    def test_overlay_wins_for_scalars(self):
        assert deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_keys_unique_to_base_preserved(self):
        assert deep_merge({"a": 1, "b": 2}, {"a": 9}) == {"a": 9, "b": 2}

    def test_keys_unique_to_overlay_added(self):
        assert deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_nested_dicts_merged(self):
        base = {"hp": {"x": 1, "y": 2}, "k": "v"}
        overlay = {"hp": {"y": 9, "z": 3}}
        assert deep_merge(base, overlay) == {"hp": {"x": 1, "y": 9, "z": 3}, "k": "v"}

    def test_lists_replaced_not_merged(self):
        assert deep_merge({"a": [1, 2]}, {"a": [3]}) == {"a": [3]}

    def test_does_not_mutate_inputs(self):
        base = {"a": {"b": 1}}
        overlay = {"a": {"c": 2}}
        result = deep_merge(base, overlay)
        result["a"]["b"] = 999
        assert base == {"a": {"b": 1}}
        assert overlay == {"a": {"c": 2}}


class TestMergeWithProvenance:
    def test_records_layer_per_leaf(self):
        merged, prov = merge_with_provenance([
            ("base", {"a": 1, "b": 2}),
            ("overlay", {"b": 9, "c": 3}),
        ])
        assert merged == {"a": 1, "b": 9, "c": 3}
        assert prov == {"a": "base", "b": "overlay", "c": "overlay"}

    def test_nested_provenance_uses_dotted_paths(self):
        merged, prov = merge_with_provenance([
            ("base", {"hp": {"x": 1, "y": 2}}),
            ("overlay", {"hp": {"y": 9}}),
        ])
        assert merged == {"hp": {"x": 1, "y": 9}}
        assert prov == {"hp.x": "base", "hp.y": "overlay"}

    def test_skips_empty_layers(self):
        merged, prov = merge_with_provenance([
            ("base", {"a": 1}),
            ("empty", {}),
            ("other", None),
        ])
        assert merged == {"a": 1}
        assert prov == {"a": "base"}


class TestFormatProvenance:
    def test_empty_provenance_shows_blank_message(self):
        out = format_provenance("aggregator", {})
        assert "aggregator" in out
        assert "no merged fields" in out

    def test_groups_by_layer(self):
        prov = {"a": "L1", "b.c": "L1", "d": "L2"}
        out = format_provenance("agg", prov)
        assert "from L1" in out
        assert "from L2" in out
        assert "- a" in out
        assert "- b.c" in out
        assert "- d" in out

    def test_truncates_long_lists(self):
        prov = {f"k{i}": "L" for i in range(100)}
        out = format_provenance("agg", prov, max_paths=10)
        assert "more" in out


class TestLoadBaselines:
    def test_missing_file_returns_empty(self, tmp_path):
        assert load_baselines(tmp_path) == {}

    def test_loads_baselines_dict(self, tmp_path):
        (tmp_path / "baselines.yaml").write_text(yaml.safe_dump({
            "baselines": {
                "felix": {"description": "F", "aggregator": {"x": 1}},
                "refl":  {"description": "R", "aggregator": {"y": 2}},
            }
        }))
        b = load_baselines(tmp_path)
        assert set(b) == {"felix", "refl"}
        assert b["felix"]["aggregator"] == {"x": 1}


class TestSharedBaselinesYaml:
    """Verify the actual shipped baselines.yaml parses and has expected entries."""

    def test_felix_present_and_well_formed(self):
        shared = (
            Path(__file__).resolve().parents[2]
            / "examples" / "_metadata"
        )
        if not (shared / "baselines.yaml").is_file():
            pytest.skip("shared baselines.yaml not present in this checkout")
        b = load_baselines(shared)
        assert "felix" in b
        felix = b["felix"]
        # Felix invariants
        assert felix["aggregator"]["selector"]["sort"] == "async_oort"
        assert felix["aggregator"]["selector"]["kwargs"]["evalGoalFactor"] == 1.0
        assert felix["aggregator"]["optimizer"]["sort"] == "fedbuff"
        assert (
            felix["aggregator"]["optimizer"]["kwargs"]["agg_rate_conf"]["type"]
            == "new"
        )
        assert (
            felix["aggregator"]["hyperparameters"]["trackTrainerAvail"]["enabled"]
            == "False"
        )
        assert (
            felix["trainer"]["hyperparameters"]["client_notify"]["enabled"]
            == "True"
        )

    def test_other_baselines_present(self):
        shared = (
            Path(__file__).resolve().parents[2]
            / "examples" / "_metadata"
        )
        if not (shared / "baselines.yaml").is_file():
            pytest.skip("shared baselines.yaml not present in this checkout")
        b = load_baselines(shared)
        assert "refl" in b
        assert "feddance" in b
        assert "oort" in b
