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
        # §3i: felix overhead stays OFF the clock (0); the per-trainer cycle leg is
        # modeled in the trainer sct instead. Guards the two against re-coupling.
        assert (
            float(felix["aggregator"]["hyperparameters"]["simCommitOverheadSeconds"])
            == 0.0
        )
        assert (
            float(felix["trainer"]["hyperparameters"]["simCompletionLegSeconds"]) > 0.0
        )

    def test_completion_leg_alias_maps_to_field(self):
        """simCompletionLegSeconds (baselines.yaml) must populate the trainer
        Hyperparameters.sim_completion_leg_s field the example trainer reads."""
        from flame.config import Hyperparameters
        base = {"rounds": 1, "epochs": 1}
        hp = Hyperparameters(**base, **{"simCompletionLegSeconds": 1.6})
        assert hp.sim_completion_leg_s == 1.6
        # default is off when unset
        assert Hyperparameters(**base).sim_completion_leg_s == 0.0

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


class TestFwdllmBaselines:
    """fwdllm/fwdllm_plus/fluxtune/fluxtune_dynkc replace the retired
    fedfwd_async_random_dynkc/fedfwd_oracular."""

    @pytest.fixture
    def baselines(self):
        shared = Path(__file__).resolve().parents[2] / "examples" / "_metadata"
        if not (shared / "baselines.yaml").is_file():
            pytest.skip("shared baselines.yaml not present in this checkout")
        return load_baselines(shared)

    def test_retired_keys_are_gone(self, baselines):
        assert "fedfwd_async_random_dynkc" not in baselines
        assert "fedfwd_oracular" not in baselines

    def test_all_four_present(self, baselines):
        for name in ("fwdllm", "fwdllm_plus", "fluxtune", "fluxtune_dynkc"):
            assert name in baselines

    def test_fwdllm_is_sync_random_fedavg_unaware(self, baselines):
        b = baselines["fwdllm"]["aggregator"]
        assert b["selector"]["sort"] == "random"
        assert b["selector"]["kwargs"]["is_async"] is False
        assert b["optimizer"]["sort"] == "fedavg"
        assert b["hyperparameters"]["trackTrainerAvail"]["enabled"] == "False"
        assert b["hyperparameters"]["reselect_each_iteration"] is False
        assert (
            baselines["fwdllm"]["trainer"]["hyperparameters"]["client_notify"]["enabled"]
            == "False"
        )

    def test_fwdllm_plus_is_sync_random_fedavg_oracular(self, baselines):
        b = baselines["fwdllm_plus"]["aggregator"]
        assert b["selector"]["sort"] == "random"
        assert b["selector"]["kwargs"]["is_async"] is False
        assert b["optimizer"]["sort"] == "fedavg"
        assert b["hyperparameters"]["trackTrainerAvail"]["enabled"] == "True"
        assert b["hyperparameters"]["trackTrainerAvail"]["type"] == "ORACULAR"
        assert b["hyperparameters"]["reselect_each_iteration"] is True

    def test_fluxtune_is_async_oort_fedbuff_with_explicit_lr(self, baselines):
        b = baselines["fluxtune"]["aggregator"]
        assert b["selector"]["sort"] == "async_oort"
        assert b["selector"]["kwargs"]["is_async"] is True
        assert b["optimizer"]["sort"] == "fedbuff"
        assert b["optimizer"]["kwargs"]["learning_rate"] == 0.075
        assert b["hyperparameters"]["trackTrainerAvail"]["enabled"] == "False"
        trainer_hp = baselines["fluxtune"]["trainer"]["hyperparameters"]
        assert trainer_hp["select_perturbation_using_jvp"] is True
        assert trainer_hp["client_notify"]["enabled"] == "True"
        assert "3st" in trainer_hp["client_notify"]["trace"]

    def test_fluxtune_dynamic_kc_is_config_driven_and_off_by_default(self, baselines):
        """Owner clarification: unlike felix (always fixed K/C), fluxtune's
        K/C policy must be config-driven -- dynamic_kc defaults to disabled
        (fixed K/C, like felix) but exposes enabled + policy so it can be
        flipped on per experiment without forking the selector."""
        dkc = baselines["fluxtune"]["aggregator"]["selector"]["kwargs"]["dynamic_kc"]
        assert dkc["enabled"] is False
        assert "policy" in dkc

    def test_fluxtune_dynkc_preserves_legacy_production_default(self, baselines):
        """Frozen parity artifact: must keep the exact selector/optimizer
        shape (including the known dataset_name copy-paste bug) of the
        retired fedfwd_async_random_dynkc, just renamed."""
        b = baselines["fluxtune_dynkc"]["aggregator"]
        assert b["selector"]["sort"] == "async_random"
        assert b["selector"]["kwargs"]["dynamic_kc"]["enabled"] is True
        assert b["optimizer"]["sort"] == "fedbuff"
        assert "learning_rate" not in b["optimizer"]["kwargs"]
        assert b["optimizer"]["kwargs"]["dataset_name"] == "google-speech"
