# Copyright 2023 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for AdaptiveKVarTrackingPolicy."""

import pytest

from flame.selector.dynamic_kc_policy import (
    AdaptiveKVarTrackingPolicy,
    build_policy,
)


def _metrics(var_last, iter_in_data_id, target=15, var_threshold=0.2, data_id=0):
    return {
        "var_last": var_last,
        "var_threshold": var_threshold,
        "iteration_per_data_id": iter_in_data_id,
        "target_iter_per_data_id": target,
        "data_id": data_id,
    }


class TestAdaptiveKRegistration:
    def test_builds_via_factory(self):
        p = build_policy("adaptive_k_var_tracking", {"target_iter_per_data_id": 10})
        assert isinstance(p, AdaptiveKVarTrackingPolicy)
        assert p.target_iter_per_data_id == 10

    def test_name(self):
        assert AdaptiveKVarTrackingPolicy().name() == "adaptive_k_var_tracking"


class TestAdaptiveKEarlyPhase:
    """progress < 0.33 → always return None."""

    def test_no_change_in_early_phase_even_if_ratio_bad(self):
        p = AdaptiveKVarTrackingPolicy(target_iter_per_data_id=15, k_step=2)
        # iter 0 → progress = 1/15 ≈ 0.066
        assert p.compute_new_k(10, _metrics(var_last=5.0, iter_in_data_id=0)) is None
        # iter 4 → progress = 5/15 ≈ 0.33 — still early
        assert p.compute_new_k(10, _metrics(var_last=5.0, iter_in_data_id=3)) is None


class TestAdaptiveKMidPhase:
    """0.33 ≤ progress < 0.75."""

    def test_high_ratio_with_non_improving_trend_increases_k(self):
        p = AdaptiveKVarTrackingPolicy(
            target_iter_per_data_id=15, k_step=2, window=6
        )
        # Prime trend with constant ratios (trend = 0)
        for iter_ in range(5, 9):
            p.compute_new_k(10, _metrics(var_last=2.0, iter_in_data_id=iter_))
        # iter 9 → progress = 10/15 ≈ 0.667. ratio = 2.0/0.2 = 10 (> 1.5) — increase
        result = p.compute_new_k(10, _metrics(var_last=2.0, iter_in_data_id=9))
        assert result == 12

    def test_low_ratio_decreases_k(self):
        p = AdaptiveKVarTrackingPolicy(target_iter_per_data_id=15, k_step=2)
        # iter 7 → progress = 8/15 ≈ 0.533; ratio = 0.04/0.2 = 0.2 (< 0.5) — decrease
        result = p.compute_new_k(10, _metrics(var_last=0.04, iter_in_data_id=7))
        assert result == 8

    def test_mid_ratio_no_change(self):
        p = AdaptiveKVarTrackingPolicy(target_iter_per_data_id=15, k_step=2)
        # ratio = 0.2/0.2 = 1.0 — in no-op band
        result = p.compute_new_k(10, _metrics(var_last=0.2, iter_in_data_id=7))
        assert result is None


class TestAdaptiveKLatePhase:
    """progress ≥ 0.75."""

    def test_ratio_above_one_doubles_step(self):
        p = AdaptiveKVarTrackingPolicy(target_iter_per_data_id=15, k_step=2)
        # iter 11 → progress = 12/15 = 0.8; ratio = 0.4/0.2 = 2.0 > 1.0
        result = p.compute_new_k(10, _metrics(var_last=0.4, iter_in_data_id=11))
        assert result == 14  # 10 + 2*2

    def test_very_low_ratio_late_decreases(self):
        p = AdaptiveKVarTrackingPolicy(target_iter_per_data_id=15, k_step=2)
        # ratio = 0.02/0.2 = 0.1 < 0.4
        result = p.compute_new_k(10, _metrics(var_last=0.02, iter_in_data_id=12))
        assert result == 8

    def test_mid_ratio_late_no_change(self):
        p = AdaptiveKVarTrackingPolicy(target_iter_per_data_id=15, k_step=2)
        # ratio = 0.15/0.2 = 0.75 — between 0.4 and 1.0, no change
        result = p.compute_new_k(10, _metrics(var_last=0.15, iter_in_data_id=12))
        assert result is None


class TestAdaptiveKDataIdReset:
    def test_window_resets_on_new_data_id(self):
        p = AdaptiveKVarTrackingPolicy(target_iter_per_data_id=15, k_step=2, window=6)
        # Prime window under data_id=0
        for iter_ in range(0, 6):
            p.compute_new_k(10, _metrics(var_last=1.0, iter_in_data_id=iter_, data_id=0))
        assert len(p._ratio_history) == 6

        # Transition to data_id=1 — history should clear on next call
        p.compute_new_k(10, _metrics(var_last=1.0, iter_in_data_id=0, data_id=1))
        # One new entry after the reset
        assert len(p._ratio_history) == 1
        assert p._last_data_id == 1


class TestAdaptiveKMissingMetrics:
    def test_missing_var_last_returns_none(self):
        p = AdaptiveKVarTrackingPolicy()
        m = _metrics(var_last=1.0, iter_in_data_id=10)
        m["var_last"] = None
        assert p.compute_new_k(10, m) is None

    def test_zero_threshold_returns_none(self):
        p = AdaptiveKVarTrackingPolicy()
        m = _metrics(var_last=1.0, iter_in_data_id=10, var_threshold=0)
        assert p.compute_new_k(10, m) is None

    def test_missing_target_returns_none(self):
        p = AdaptiveKVarTrackingPolicy(target_iter_per_data_id=0)
        m = _metrics(var_last=1.0, iter_in_data_id=10)
        m["target_iter_per_data_id"] = None
        # target_iter_per_data_id=0 (ctor default) and metric None → both falsy
        assert p.compute_new_k(10, m) is None


class TestAdaptiveKDoesNotTouchC:
    def test_compute_new_c_always_none(self):
        p = AdaptiveKVarTrackingPolicy()
        assert p.compute_new_c(
            15, _metrics(var_last=1.0, iter_in_data_id=10)
        ) is None
