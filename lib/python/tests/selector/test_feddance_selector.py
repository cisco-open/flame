# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FedDance selector tests."""

import math

import pytest

from flame.mode.message import MessageType
from flame.selector.feddance import (
    FedDanceSelector,
    PROP_LOCAL_ACCURACY,
    PROP_STAT_UTILITY,
    PROP_U,
)


@pytest.fixture
def feddance():
    return FedDanceSelector(
        aggr_num=3,
        history_window=10,
        prediction_window=5,
        accuracy_window=5,
    )


class TestInit:
    def test_defaults(self, feddance):
        assert feddance.aggr_num == 3
        assert feddance.history_window == 10
        assert feddance.prediction_window == 5
        assert feddance.accuracy_window == 5
        assert feddance.negative_accuracy_handling == "relu"
        assert isinstance(feddance.selected_ends, set)
        assert isinstance(feddance.ordered_updates_recv_ends, list)

    def test_missing_aggr_num_raises(self):
        with pytest.raises(KeyError):
            FedDanceSelector()

    def test_invalid_neg_a_handling_raises(self):
        with pytest.raises(ValueError):
            FedDanceSelector(aggr_num=3, negative_accuracy_handling="bogus")


class TestColdStart:
    def test_first_round_picks_top_n_via_cold_start(
        self, feddance, make_ends, channel_props
    ):
        ends = make_ends(count=10, prefix="t")
        result = feddance.select(
            ends, channel_props, trainer_unavail_list=[], task_to_perform="train"
        )
        assert len(result) == feddance.num_of_ends
        for end_id in result:
            assert end_id in ends


class TestUnavailFiltering:
    def test_unavail_excluded(self, feddance, make_ends, channel_props):
        ends = make_ends(count=6, prefix="t")
        result = feddance.select(
            ends, channel_props, trainer_unavail_list=["t0", "t1"], task_to_perform="train"
        )
        assert "t0" not in result
        assert "t1" not in result


class TestIdempotentWithinRound:
    def test_same_round_returns_same_picks(self, feddance, make_ends, channel_props):
        ends = make_ends(count=10, prefix="t")
        r1 = feddance.select(
            ends, channel_props, trainer_unavail_list=[], task_to_perform="train"
        )
        r2 = feddance.select(
            ends, channel_props, trainer_unavail_list=[], task_to_perform="train"
        )
        assert set(r1.keys()) == set(r2.keys())


class TestOnUpdateReceived:
    def test_captures_loss_and_accuracy(self, feddance):
        feddance.on_update_received(
            "t1",
            {MessageType.STAT_UTILITY: 0.7, MessageType.LOCAL_ACCURACY: 0.9},
            round_num=1,
        )
        assert feddance.last_loss["t1"] == pytest.approx(0.7)
        assert feddance.accuracy_history["t1"][-1] == pytest.approx(0.9)
        assert feddance.last_engaged_round["t1"] == 1
        assert "t1" in feddance.ordered_updates_recv_ends

    def test_missing_fields_ok(self, feddance):
        feddance.on_update_received("t1", {}, round_num=1)
        assert "t1" not in feddance.last_loss
        assert "t1" not in feddance.accuracy_history
        assert feddance.last_engaged_round["t1"] == 1


class TestOnRoundCompleted:
    def test_updates_prev_round_means(self, feddance, make_ends):
        feddance.on_update_received(
            "a", {MessageType.STAT_UTILITY: 1.0, MessageType.LOCAL_ACCURACY: 0.5}, 1
        )
        feddance.on_update_received(
            "b", {MessageType.STAT_UTILITY: 3.0, MessageType.LOCAL_ACCURACY: 0.9}, 1
        )
        feddance.selected_ends.update(["a", "b"])
        feddance.on_round_completed(make_ends(["a", "b"]), 1)

        assert feddance.prev_round_mean_I == pytest.approx(2.0)
        assert feddance.prev_round_mean_A == pytest.approx(0.7)
        assert feddance.selected_ends == set()
        assert feddance.ordered_updates_recv_ends == []


class TestAccuracySlope:
    def test_two_points_gives_slope(self, feddance):
        feddance.accuracy_history["t"] = __import__("collections").deque(
            [0.5, 0.7], maxlen=5
        )
        assert feddance._accuracy_slope("t") == pytest.approx(0.2)

    def test_single_point_returns_none(self, feddance):
        feddance.accuracy_history["t"] = __import__("collections").deque(
            [0.5], maxlen=5
        )
        assert feddance._accuracy_slope("t") is None

    def test_unseen_end_returns_none(self, feddance):
        assert feddance._accuracy_slope("nope") is None


class TestNegativeAccuracyHandling:
    def test_relu_clamps_to_eps(self, feddance):
        assert feddance._handle_negative_a(-0.5) == pytest.approx(1e-6)

    def test_abs_takes_absolute(self):
        s = FedDanceSelector(aggr_num=3, negative_accuracy_handling="abs")
        assert s._handle_negative_a(-0.3) == pytest.approx(0.3)

    def test_raw_passes_through(self):
        s = FedDanceSelector(aggr_num=3, negative_accuracy_handling="raw")
        assert s._handle_negative_a(-0.3) == pytest.approx(-0.3)


class TestUtilityFormula:
    def test_utility_records_properties(
        self, feddance, make_ends, channel_props
    ):
        ends = make_ends(
            ["a", "b", "c"],
            **{PROP_STAT_UTILITY: 1.0, PROP_LOCAL_ACCURACY: 0.5},
        )
        feddance.select(
            ends, channel_props, trainer_unavail_list=[], task_to_perform="train"
        )
        for eid in ends:
            assert ends[eid].get_property(PROP_U) is not None


class TestPredictorTracksCheckins:
    def test_seen_ends_become_tracked(self, feddance, make_ends, channel_props):
        ends = make_ends(count=5, prefix="t")
        feddance.select(
            ends, channel_props, trainer_unavail_list=[], task_to_perform="train"
        )
        for eid in ends:
            assert feddance.predictor.is_tracked(eid)


class TestInflightExcluded:
    def test_inflight_not_reselected_next_round(
        self, feddance, make_ends, channel_props
    ):
        ends = make_ends(count=10, prefix="t")
        feddance.select(
            ends, channel_props, trainer_unavail_list=[], task_to_perform="train"
        )
        first = set(feddance.selected_ends)

        channel_props["round"] = 2
        feddance.newly_selected_this_round = set()
        r2 = feddance.select(
            ends, channel_props, trainer_unavail_list=[], task_to_perform="train"
        )
        for end_id in r2:
            assert end_id not in first


class TestSelectorRegistration:
    def test_enum_value_present(self):
        from flame.config import SelectorType
        assert SelectorType.FEDDANCE.value == "feddance"
