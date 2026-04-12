# Copyright 2023 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for dynamic_kc_controller.py"""

import pytest

from flame.selector.dynamic_kc_controller import DynamicKCController
from flame.selector.dynamic_kc_policy import NoOpPolicy, VarianceBasedPolicy


def _make_controller(policy=None, k_init=10, c_init=30, k_min=1, k_max=50, c_min=1, c_max=100, update_every_n_aggs=1):
    if policy is None:
        policy = NoOpPolicy()
    return DynamicKCController(
        policy=policy,
        k_init=k_init,
        c_init=c_init,
        k_min=k_min,
        k_max=k_max,
        c_min=c_min,
        c_max=c_max,
        update_every_n_aggs=update_every_n_aggs,
    )


class TestDynamicKCControllerInit:
    def test_initial_values(self):
        ctrl = _make_controller(k_init=10, c_init=30)
        assert ctrl.get_k() == 10
        assert ctrl.get_c() == 30

    def test_history_initialized(self):
        ctrl = _make_controller()
        assert len(ctrl._k_history) == 1
        assert ctrl._k_history[0] == (0, 10)
        assert ctrl._c_history[0] == (0, 30)


class TestDynamicKCControllerStep:
    def test_noop_policy_does_not_change_kc(self):
        ctrl = _make_controller(policy=NoOpPolicy())
        k, c = ctrl.step({"var_pass_rate": 0.9})
        assert k == 10
        assert c == 30

    def test_step_increments_counters(self):
        ctrl = _make_controller()
        ctrl.step({})
        assert ctrl._total_updates == 1

    def test_update_every_n_aggs_skips_policy(self):
        policy = VarianceBasedPolicy(window=1, high_threshold=0.5, k_step=2)
        ctrl = _make_controller(policy=policy, update_every_n_aggs=3)
        # First two calls: policy should NOT be invoked (counter < 3)
        k1, _ = ctrl.step({"var_pass_rate": 0.9})
        k2, _ = ctrl.step({"var_pass_rate": 0.9})
        assert k1 == 10 and k2 == 10  # unchanged
        # Third call: policy fires
        k3, _ = ctrl.step({"var_pass_rate": 0.9})
        assert k3 == 8  # 10 - 2

    def test_k_is_clamped_to_max(self):
        policy = VarianceBasedPolicy(window=1, low_threshold=0.5, k_step=100)
        ctrl = _make_controller(policy=policy, k_init=10, k_max=20)
        k, _ = ctrl.step({"var_pass_rate": 0.1})  # below low → increase by 100
        assert k == 20  # clamped to k_max

    def test_k_is_clamped_to_min(self):
        policy = VarianceBasedPolicy(window=1, high_threshold=0.5, k_step=100)
        ctrl = _make_controller(policy=policy, k_init=10, k_min=5)
        k, _ = ctrl.step({"var_pass_rate": 0.9})  # above high → decrease by 100
        assert k == 5  # clamped to k_min

    def test_k_history_updated_on_change(self):
        policy = VarianceBasedPolicy(window=1, high_threshold=0.5, k_step=2)
        ctrl = _make_controller(policy=policy, k_init=10)
        ctrl.step({"var_pass_rate": 0.9})  # decrease K
        assert len(ctrl._k_history) == 2
        assert ctrl._k_history[-1][1] == 8

    def test_k_history_not_updated_when_unchanged(self):
        policy = NoOpPolicy()
        ctrl = _make_controller(policy=policy)
        ctrl.step({})
        assert len(ctrl._k_history) == 1  # no change

    def test_returns_current_kc_when_policy_returns_none(self):
        ctrl = _make_controller(policy=NoOpPolicy(), k_init=15, c_init=45)
        k, c = ctrl.step({})
        assert k == 15
        assert c == 45

    def test_multiple_steps_accumulate_history(self):
        policy = VarianceBasedPolicy(window=1, high_threshold=0.5, k_step=2)
        ctrl = _make_controller(policy=policy, k_init=20, k_min=1)
        for _ in range(3):
            ctrl.step({"var_pass_rate": 0.9})  # each decreases K by 2
        assert ctrl.get_k() == 14  # 20 → 18 → 16 → 14
        assert len(ctrl._k_history) == 4  # init + 3 changes


class TestDynamicKCControllerSummary:
    def test_summary_contains_required_keys(self):
        ctrl = _make_controller()
        ctrl.step({})
        summary = ctrl.summary()
        for key in ["k", "c", "total_updates", "k_changes", "c_changes", "k_history_last5", "c_history_last5", "policy"]:
            assert key in summary

    def test_summary_total_updates_matches_step_count(self):
        ctrl = _make_controller()
        for _ in range(5):
            ctrl.step({})
        assert ctrl.summary()["total_updates"] == 5

    def test_summary_policy_name(self):
        ctrl = _make_controller(policy=NoOpPolicy())
        assert ctrl.summary()["policy"] == "noop"
