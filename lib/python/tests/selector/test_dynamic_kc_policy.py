# Copyright 2023 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for dynamic_kc_policy.py"""

import pytest

from flame.selector.dynamic_kc_policy import (
    NoOpPolicy,
    VarianceBasedPolicy,
    StalenessBasedPolicy,
    EligibleEndsBasedPolicy,
    StepSchedulePolicy,
    CompositePolicy,
    build_policy,
)


# ---------------------------------------------------------------------------
# NoOpPolicy
# ---------------------------------------------------------------------------


class TestNoOpPolicy:
    def setup_method(self):
        self.policy = NoOpPolicy()

    def test_compute_new_k_returns_none(self):
        assert self.policy.compute_new_k(10, {"var_pass_rate": 0.9}) is None

    def test_compute_new_c_returns_none(self):
        assert self.policy.compute_new_c(30, {"avg_staleness": 5.0}) is None

    def test_name(self):
        assert self.policy.name() == "noop"


# ---------------------------------------------------------------------------
# VarianceBasedPolicy
# ---------------------------------------------------------------------------


class TestVarianceBasedPolicy:
    def setup_method(self):
        self.policy = VarianceBasedPolicy(
            high_threshold=0.8,
            low_threshold=0.3,
            k_step=2,
            window=5,
        )

    def _fill_window(self, rate: float):
        """Feed a constant rate until the window is full."""
        for _ in range(5):
            result = self.policy.compute_new_k(10, {"var_pass_rate": rate})
        return result

    def test_returns_none_before_window_fills(self):
        # Only 4 values — window is 5, so result is None
        for i in range(4):
            result = self.policy.compute_new_k(10, {"var_pass_rate": 0.9})
        assert result is None

    def test_high_pass_rate_decreases_k(self):
        result = self._fill_window(0.9)  # > high_threshold=0.8
        assert result == 8  # 10 - 2

    def test_low_pass_rate_increases_k(self):
        result = self._fill_window(0.1)  # < low_threshold=0.3
        assert result == 12  # 10 + 2

    def test_mid_pass_rate_no_change(self):
        result = self._fill_window(0.5)  # in [0.3, 0.8]
        assert result is None

    def test_missing_var_pass_rate_returns_none(self):
        result = self.policy.compute_new_k(10, {})
        assert result is None

    def test_compute_new_c_always_none(self):
        assert self.policy.compute_new_c(30, {"var_pass_rate": 0.9}) is None

    def test_name(self):
        assert self.policy.name() == "variance_based"


# ---------------------------------------------------------------------------
# StalenessBasedPolicy
# ---------------------------------------------------------------------------


class TestStalenessBasedPolicy:
    def setup_method(self):
        self.policy = StalenessBasedPolicy(
            stale_threshold=3.0,
            fresh_threshold=1.0,
            c_step=5,
            window=5,
        )

    def _fill_window(self, staleness: float):
        for _ in range(5):
            result = self.policy.compute_new_c(30, {"avg_staleness": staleness})
        return result

    def test_returns_none_before_window_fills(self):
        for _ in range(4):
            result = self.policy.compute_new_c(30, {"avg_staleness": 5.0})
        assert result is None

    def test_high_staleness_decreases_c(self):
        result = self._fill_window(4.0)  # > stale_threshold=3.0
        assert result == 25  # 30 - 5

    def test_low_staleness_increases_c(self):
        result = self._fill_window(0.5)  # < fresh_threshold=1.0
        assert result == 35  # 30 + 5

    def test_mid_staleness_no_change(self):
        result = self._fill_window(2.0)  # in [1.0, 3.0]
        assert result is None

    def test_missing_avg_staleness_returns_none(self):
        result = self.policy.compute_new_c(30, {})
        assert result is None

    def test_compute_new_k_always_none(self):
        assert self.policy.compute_new_k(10, {"avg_staleness": 5.0}) is None

    def test_name(self):
        assert self.policy.name() == "staleness_based"


# ---------------------------------------------------------------------------
# EligibleEndsBasedPolicy
# ---------------------------------------------------------------------------


class TestEligibleEndsBasedPolicy:
    def setup_method(self):
        self.policy = EligibleEndsBasedPolicy(
            headroom_factor=1.5,
            undercommit_factor=0.8,
            c_step=5,
            window=3,
        )

    def _fill_window(self, n_eligible: int):
        for _ in range(3):
            result = self.policy.compute_new_c(20, {"n_eligible_train": n_eligible})
        return result

    def test_returns_none_before_window_fills(self):
        for _ in range(2):
            result = self.policy.compute_new_c(20, {"n_eligible_train": 100})
        assert result is None

    def test_large_pool_increases_c(self):
        # avg_eligible=50 > 20 * 1.5 = 30 → increase
        result = self._fill_window(50)
        assert result == 25  # 20 + 5

    def test_small_pool_snaps_c_down(self):
        # avg_eligible=10 < 20 * 0.8 = 16 → snap down to int(10)=10
        result = self._fill_window(10)
        assert result == 10

    def test_mid_pool_no_change(self):
        # avg_eligible=20, 20 < 30 and 20 > 16 → no change
        result = self._fill_window(20)
        assert result is None

    def test_snap_floor_is_1(self):
        # avg_eligible=0 → snap down to max(1, 0) = 1
        result = self._fill_window(0)
        assert result == 1

    def test_missing_n_eligible_returns_none(self):
        result = self.policy.compute_new_c(20, {})
        assert result is None

    def test_compute_new_k_always_none(self):
        assert self.policy.compute_new_k(10, {"n_eligible_train": 100}) is None

    def test_name(self):
        assert self.policy.name() == "eligible_ends_based"


# ---------------------------------------------------------------------------
# StepSchedulePolicy
# ---------------------------------------------------------------------------


class TestStepSchedulePolicy:
    def setup_method(self):
        self.policy = StepSchedulePolicy(k_step=2, n_aggs_per_step=10, k_floor=5)

    def test_no_change_before_step_boundary(self):
        for n in range(1, 10):
            result = self.policy.compute_new_k(20, {"n_aggs_completed": n})
            assert result is None, f"Expected None at n={n}"

    def test_decreases_at_step_boundary(self):
        result = self.policy.compute_new_k(20, {"n_aggs_completed": 10})
        assert result == 18  # 20 - 2

    def test_does_not_go_below_floor(self):
        result = self.policy.compute_new_k(5, {"n_aggs_completed": 10})
        assert result is None  # already at floor, no change

    def test_multiple_steps(self):
        k = 20
        for step in [10, 20, 30]:
            r = self.policy.compute_new_k(k, {"n_aggs_completed": step})
            if r is not None:
                k = r
        assert k == 14  # 20 → 18 → 16 → 14

    def test_compute_new_c_always_none(self):
        assert self.policy.compute_new_c(30, {"n_aggs_completed": 10}) is None

    def test_name(self):
        assert self.policy.name() == "step_schedule"


# ---------------------------------------------------------------------------
# CompositePolicy
# ---------------------------------------------------------------------------


class TestCompositePolicy:
    def test_first_matching_sub_policy_wins_for_k(self):
        policy = CompositePolicy(
            sub_policies=[
                {"name": "variance_based", "kwargs": {"window": 1, "k_step": 3}},
                {"name": "step_schedule", "kwargs": {"k_step": 5, "n_aggs_per_step": 1, "k_floor": 1}},
            ]
        )
        # variance_based with window=1 will return on first call if rate is extreme
        result = policy.compute_new_k(10, {"var_pass_rate": 0.95, "n_aggs_completed": 1})
        assert result == 7  # variance_based fires first: 10 - 3

    def test_second_policy_fires_when_first_returns_none(self):
        policy = CompositePolicy(
            sub_policies=[
                {"name": "variance_based", "kwargs": {"window": 10}},  # needs 10 calls
                {"name": "step_schedule", "kwargs": {"k_step": 2, "n_aggs_per_step": 1, "k_floor": 1}},
            ]
        )
        # variance_based won't fire (window=10), step_schedule fires at n_aggs_completed=1
        result = policy.compute_new_k(10, {"var_pass_rate": 0.5, "n_aggs_completed": 1})
        assert result == 8  # step_schedule: 10 - 2

    def test_c_delegated_to_staleness_policy(self):
        policy = CompositePolicy(
            sub_policies=[
                {"name": "staleness_based", "kwargs": {"window": 1, "c_step": 5}},
            ]
        )
        result = policy.compute_new_c(30, {"avg_staleness": 5.0})
        assert result == 25  # 30 - 5

    def test_name_format(self):
        policy = CompositePolicy(
            sub_policies=[
                {"name": "variance_based"},
                {"name": "staleness_based"},
            ]
        )
        assert "variance_based" in policy.name()
        assert "staleness_based" in policy.name()
        assert "composite" in policy.name()


# ---------------------------------------------------------------------------
# build_policy factory
# ---------------------------------------------------------------------------


class TestBuildPolicy:
    def test_builds_noop(self):
        p = build_policy("noop", {})
        assert isinstance(p, NoOpPolicy)

    def test_builds_variance_based_with_kwargs(self):
        p = build_policy("variance_based", {"k_step": 4, "window": 5})
        assert isinstance(p, VarianceBasedPolicy)
        assert p.k_step == 4

    def test_builds_staleness_based(self):
        p = build_policy("staleness_based", {})
        assert isinstance(p, StalenessBasedPolicy)

    def test_builds_eligible_ends_based(self):
        p = build_policy("eligible_ends_based", {})
        assert isinstance(p, EligibleEndsBasedPolicy)

    def test_builds_step_schedule(self):
        p = build_policy("step_schedule", {})
        assert isinstance(p, StepSchedulePolicy)

    def test_raises_on_unknown_policy(self):
        with pytest.raises(ValueError, match="Unknown dynamic_kc policy"):
            build_policy("nonexistent_policy", {})
