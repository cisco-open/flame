# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Config-gen wiring tests: verify 6-baseline matrix HP after merge.

These tests catch mis-wiring before cluster time by simulating the runner's
baseline.aggregator + experiment.config_overrides merge for each sim experiment
in the parity YAML and asserting the correct availability flags.

No subprocess spawning, no GPU, no MQTT — pure YAML/dict tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from flame.launch.baselines import deep_merge, load_baselines

# ── Paths ─────────────────────────────────────────────────────────────────────
_EXAMPLES = Path(__file__).parents[2] / "examples"
_METADATA_DIR = _EXAMPLES / "_metadata"
_PARITY_YAML = (
    _EXAMPLES
    / "async_cifar10/expt_scripts_2026/felix_oort_refl_feddance_alpha0.1_parity.yaml"
)


@pytest.fixture(scope="module")
def baselines():
    return load_baselines(_METADATA_DIR)


@pytest.fixture(scope="module")
def parity_experiments():
    with open(_PARITY_YAML) as f:
        data = yaml.safe_load(f)
        return data.get("experiments", data) if isinstance(data, dict) else data


def _sim_exp(parity_experiments, name: str) -> dict:
    """Return the sim experiment with the given name."""
    for exp in parity_experiments:
        if exp["name"] == name:
            return exp
    raise KeyError(f"experiment {name!r} not in parity YAML")


def _merged_hp(baselines, parity_experiments, exp_name: str, baseline_name: str) -> dict:
    """Merge baseline.aggregator.hyperparameters with experiment config_overrides HP."""
    bl = baselines[baseline_name]
    bl_hp = (bl.get("aggregator") or {}).get("hyperparameters") or {}
    exp = _sim_exp(parity_experiments, exp_name)
    exp_hp = (
        (exp.get("aggregator") or {})
        .get("config_overrides", {})
        .get("hyperparameters") or {}
    )
    return deep_merge(bl_hp, exp_hp)


# ── Catalog presence ───────────────────────────────────────────────────────────

class TestBaselineCatalogPresence:
    def test_all_six_baselines_in_catalog(self, baselines):
        required = {"felix", "oort", "oort_star", "refl", "feddance", "fedbuff"}
        missing = required - set(baselines)
        assert not missing, f"Missing from baselines.yaml: {missing}"

    def test_oort_star_uses_oort_selector(self, baselines):
        assert baselines["oort_star"]["aggregator"]["selector"]["sort"] == "oort"

    def test_oort_star_uses_fedavg_optimizer(self, baselines):
        assert baselines["oort_star"]["aggregator"]["optimizer"]["sort"] == "fedavg"

    def test_oort_star_has_oracular_tracking(self, baselines):
        hp = baselines["oort_star"]["aggregator"]["hyperparameters"]
        tta = hp["trackTrainerAvail"]
        assert tta["enabled"] == "True"
        assert tta["type"] == "ORACULAR"

    def test_all_six_baselines_in_parity_yaml(self, parity_experiments):
        present = {e["baseline"] for e in parity_experiments if "baseline" in e}
        required = {"felix", "oort", "oort_star", "refl", "feddance", "fedbuff"}
        missing = required - present
        assert not missing, f"Baselines not in parity YAML: {missing}"


# ── Per-baseline HP verification ───────────────────────────────────────────────

class TestFelixWiring:
    def test_avail_select_filter_on(self, baselines, parity_experiments):
        hp = _merged_hp(baselines, parity_experiments,
                        "felix_n300_alpha0.1_syn0_stream_sim", "felix")
        assert hp["avail_select_filter"] == "True"

    def test_proactive_inflight_evict_on(self, baselines, parity_experiments):
        hp = _merged_hp(baselines, parity_experiments,
                        "felix_n300_alpha0.1_syn0_stream_sim", "felix")
        assert hp["proactive_inflight_evict"] == "True"

    def test_sim_unavailability_on(self, baselines, parity_experiments):
        hp = _merged_hp(baselines, parity_experiments,
                        "felix_n300_alpha0.1_syn0_stream_sim", "felix")
        assert hp.get("sim_unavailability") == "True"

    def test_trackTrainerAvail_disabled(self, baselines, parity_experiments):
        hp = _merged_hp(baselines, parity_experiments,
                        "felix_n300_alpha0.1_syn0_stream_sim", "felix")
        tta = hp.get("trackTrainerAvail") or {}
        assert str(tta.get("enabled", "False")).strip().lower() != "true", (
            "felix uses client_notify, not oracular"
        )


class TestOortWiring:
    def test_avail_select_filter_off(self, baselines, parity_experiments):
        hp = _merged_hp(baselines, parity_experiments,
                        "oort_n300_alpha0.1_syn0_stream_sim", "oort")
        assert hp["avail_select_filter"] == "False"

    def test_proactive_inflight_evict_off(self, baselines, parity_experiments):
        hp = _merged_hp(baselines, parity_experiments,
                        "oort_n300_alpha0.1_syn0_stream_sim", "oort")
        assert hp["proactive_inflight_evict"] == "False"

    def test_trackTrainerAvail_enabled(self, baselines, parity_experiments):
        hp = _merged_hp(baselines, parity_experiments,
                        "oort_n300_alpha0.1_syn0_stream_sim", "oort")
        tta = hp.get("trackTrainerAvail") or {}
        assert str(tta.get("enabled", "False")).strip().lower() == "true"
        assert tta.get("type", "").upper() == "ORACULAR"

    def test_trace_name_set(self, baselines, parity_experiments):
        hp = _merged_hp(baselines, parity_experiments,
                        "oort_n300_alpha0.1_syn0_stream_sim", "oort")
        tta = hp.get("trackTrainerAvail") or {}
        assert tta.get("trace"), "oort must have trackTrainerAvail.trace for starvation-advance"


class TestOortStarWiring:
    def test_avail_select_filter_on(self, baselines, parity_experiments):
        hp = _merged_hp(baselines, parity_experiments,
                        "oort_star_n300_alpha0.1_syn0_stream_sim", "oort_star")
        assert hp["avail_select_filter"] == "True"

    def test_proactive_inflight_evict_off(self, baselines, parity_experiments):
        hp = _merged_hp(baselines, parity_experiments,
                        "oort_star_n300_alpha0.1_syn0_stream_sim", "oort_star")
        assert hp["proactive_inflight_evict"] == "False"

    def test_trackTrainerAvail_enabled(self, baselines, parity_experiments):
        hp = _merged_hp(baselines, parity_experiments,
                        "oort_star_n300_alpha0.1_syn0_stream_sim", "oort_star")
        tta = hp.get("trackTrainerAvail") or {}
        assert str(tta.get("enabled", "False")).strip().lower() == "true"
        assert tta.get("type", "").upper() == "ORACULAR"

    def test_differs_from_oort_only_in_filter(self, baselines, parity_experiments):
        oort_hp = _merged_hp(baselines, parity_experiments,
                             "oort_n300_alpha0.1_syn0_stream_sim", "oort")
        ostar_hp = _merged_hp(baselines, parity_experiments,
                              "oort_star_n300_alpha0.1_syn0_stream_sim", "oort_star")
        # avail_select_filter is the only flag that differs
        assert oort_hp["avail_select_filter"] == "False"
        assert ostar_hp["avail_select_filter"] == "True"
        assert oort_hp["proactive_inflight_evict"] == ostar_hp["proactive_inflight_evict"]
        oort_tta = oort_hp.get("trackTrainerAvail", {})
        ostar_tta = ostar_hp.get("trackTrainerAvail", {})
        assert oort_tta.get("enabled") == ostar_tta.get("enabled")
        assert oort_tta.get("type") == ostar_tta.get("type")


class TestReflWiring:
    def test_avail_select_filter_on(self, baselines, parity_experiments):
        hp = _merged_hp(baselines, parity_experiments,
                        "refl_n300_alpha0.1_syn0_stream_sim", "refl")
        assert hp["avail_select_filter"] == "True"

    def test_proactive_inflight_evict_off(self, baselines, parity_experiments):
        hp = _merged_hp(baselines, parity_experiments,
                        "refl_n300_alpha0.1_syn0_stream_sim", "refl")
        assert hp["proactive_inflight_evict"] == "False"

    def test_trackTrainerAvail_enabled(self, baselines, parity_experiments):
        hp = _merged_hp(baselines, parity_experiments,
                        "refl_n300_alpha0.1_syn0_stream_sim", "refl")
        tta = hp.get("trackTrainerAvail") or {}
        assert str(tta.get("enabled", "False")).strip().lower() == "true"
        assert tta.get("type", "").upper() == "ORACULAR"


class TestFedDanceWiring:
    def test_avail_select_filter_on(self, baselines, parity_experiments):
        hp = _merged_hp(baselines, parity_experiments,
                        "feddance_n300_alpha0.1_syn0_stream_sim", "feddance")
        assert hp["avail_select_filter"] == "True"

    def test_proactive_inflight_evict_off(self, baselines, parity_experiments):
        hp = _merged_hp(baselines, parity_experiments,
                        "feddance_n300_alpha0.1_syn0_stream_sim", "feddance")
        assert hp["proactive_inflight_evict"] == "False"

    def test_trackTrainerAvail_disabled(self, baselines, parity_experiments):
        hp = _merged_hp(baselines, parity_experiments,
                        "feddance_n300_alpha0.1_syn0_stream_sim", "feddance")
        tta = hp.get("trackTrainerAvail") or {}
        assert str(tta.get("enabled", "False")).strip().lower() != "true", (
            "feddance has its own check-in predictor, not oracular"
        )


class TestFedBuffWiring:
    def test_avail_select_filter_off(self, baselines, parity_experiments):
        hp = _merged_hp(baselines, parity_experiments,
                        "fedbuff_n300_alpha0.1_syn0_stream_sim", "fedbuff")
        assert hp["avail_select_filter"] == "False"

    def test_proactive_inflight_evict_off(self, baselines, parity_experiments):
        hp = _merged_hp(baselines, parity_experiments,
                        "fedbuff_n300_alpha0.1_syn0_stream_sim", "fedbuff")
        assert hp["proactive_inflight_evict"] == "False"

    def test_sim_unavailability_on(self, baselines, parity_experiments):
        hp = _merged_hp(baselines, parity_experiments,
                        "fedbuff_n300_alpha0.1_syn0_stream_sim", "fedbuff")
        assert hp.get("sim_unavailability") == "True", (
            "fedbuff must activate the availability gate via sim_unavailability "
            "(unaware baseline: gate loads trace but does not filter selection pool)"
        )

    def test_availability_trace_set(self, baselines, parity_experiments):
        hp = _merged_hp(baselines, parity_experiments,
                        "fedbuff_n300_alpha0.1_syn0_stream_sim", "fedbuff")
        has_trace = (
            hp.get("availability_trace")
            or (hp.get("client_notify") or {}).get("trace")
        )
        assert has_trace, (
            "fedbuff must have availability_trace (or client_notify.trace) so "
            "_init_availability can load the syn_0 trace without warning"
        )

    def test_trackTrainerAvail_disabled(self, baselines, parity_experiments):
        hp = _merged_hp(baselines, parity_experiments,
                        "fedbuff_n300_alpha0.1_syn0_stream_sim", "fedbuff")
        tta = hp.get("trackTrainerAvail") or {}
        assert str(tta.get("enabled", "False")).strip().lower() != "true", (
            "fedbuff should not use oracular tracking"
        )
