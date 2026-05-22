# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FedDancePredictor tests."""

import math

import pytest

from flame.availability.feddance_predictor import FedDancePredictor


class TestInit:
    def test_defaults(self):
        p = FedDancePredictor()
        assert p.history_window == 50
        assert p.prediction_window == 5

    def test_rejects_nonpositive_windows(self):
        with pytest.raises(ValueError):
            FedDancePredictor(history_window=0)
        with pytest.raises(ValueError):
            FedDancePredictor(prediction_window=-1)


class TestColdStart:
    def test_unseen_end_returns_cold_start_lambda(self):
        p = FedDancePredictor(lambda_cold_start=0.3)
        assert p.lambda_m("never_seen", 10) == 0.3

    def test_unseen_end_V_m_uses_cold_start(self):
        p = FedDancePredictor(prediction_window=5, lambda_cold_start=0.0)
        assert p.V_m("never_seen", 10) == pytest.approx(0.0)

    def test_is_tracked(self):
        p = FedDancePredictor()
        assert not p.is_tracked("x")
        p.record_checkin("x", 1)
        assert p.is_tracked("x")


class TestAlwaysAvailable:
    def test_lambda_approaches_one(self):
        p = FedDancePredictor(history_window=50, prediction_window=5)
        for r in range(50):
            p.record_checkin("alice", r)
        # round=50: count in [0, 50) = 50
        assert p.lambda_m("alice", 50) == 1.0
        assert p.V_m("alice", 50) == pytest.approx(1 - math.exp(-5))


class TestNeverAvailable:
    def test_no_checkins_yields_zero(self):
        p = FedDancePredictor()
        p.record_checkin("bob", 0)  # one stale checkin
        # round=100: history window is [50, 100), bob has only round 0
        assert p.lambda_m("bob", 100) == 0.0
        assert p.V_m("bob", 100) == 0.0


class TestPartialAvailability:
    def test_half_present(self):
        p = FedDancePredictor(history_window=50, prediction_window=5)
        for r in range(0, 50, 2):
            p.record_checkin("carol", r)
        # 25 check-ins in window of 50
        assert p.lambda_m("carol", 50) == pytest.approx(0.5)
        expected = 1 - math.exp(-0.5 * 5)
        assert p.V_m("carol", 50) == pytest.approx(expected)


class TestSlidingWindow:
    def test_old_checkins_evicted(self):
        p = FedDancePredictor(history_window=10, prediction_window=5)
        for r in range(20):
            p.record_checkin("dan", r)
        # Only rounds 10-19 remain in the deque
        assert p.lambda_m("dan", 20) == 1.0
        # At round 30, window is [20, 30) — none of dan's rounds fall there
        assert p.lambda_m("dan", 30) == 0.0


class TestSeedHistory:
    def test_seed_populates_window(self):
        p = FedDancePredictor(history_window=10, prediction_window=5)
        p.seed_history("eve", rounds_present=list(range(0, 10)))
        assert p.lambda_m("eve", 10) == 1.0

    def test_seed_truncates_to_window(self):
        p = FedDancePredictor(history_window=5)
        p.seed_history("frank", rounds_present=list(range(0, 20)))
        # Only the last 5 rounds (15-19) retained
        assert p.lambda_m("frank", 20) == 1.0


class TestDuplicateCheckin:
    def test_same_round_recorded_once(self):
        p = FedDancePredictor(history_window=10)
        p.record_checkin("g", 5)
        p.record_checkin("g", 5)
        p.record_checkin("g", 5)
        assert p.lambda_m("g", 10) == pytest.approx(1 / 10)
