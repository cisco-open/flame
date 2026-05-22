# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Oort selector tests."""

from datetime import timedelta

import pytest

from flame.selector.oort import OortSelector


@pytest.fixture
def oort():
    return OortSelector(aggr_num=3)


class TestOortInit:
    def test_defaults(self, oort):
        assert oort.aggr_num == 3
        assert oort.num_of_ends == int(3 * 1.3)
        assert isinstance(oort.selected_ends, set)
        assert oort.ordered_updates_recv_ends == []
        assert 0.0 < oort.exploration_factor <= 1.0


class TestOortColdStart:
    def test_first_round_random_selection(self, oort, make_ends, channel_props):
        ends = make_ends(count=10, prefix="t")
        result = oort.select(
            ends, channel_props, trainer_unavail_list=[], task_to_perform="train"
        )
        assert len(result) == oort.num_of_ends
        for end_id in result:
            assert end_id in ends

    def test_in_flight_excluded_next_round(
        self, oort, make_ends, channel_props
    ):
        ends = make_ends(count=10, prefix="t")
        oort.select(
            ends, channel_props, trainer_unavail_list=[], task_to_perform="train"
        )
        first_picked = set(oort.selected_ends)

        channel_props["round"] = 2
        result = oort.select(
            ends, channel_props, trainer_unavail_list=[], task_to_perform="train"
        )
        for end_id in result:
            assert end_id not in first_picked or end_id in oort.selected_ends


class TestOortIdempotentWithinRound:
    def test_same_round_returns_same_set(self, oort, make_ends, channel_props):
        ends = make_ends(count=10, prefix="t")
        r1 = oort.select(
            ends, channel_props, trainer_unavail_list=[], task_to_perform="train"
        )
        r2 = oort.select(
            ends, channel_props, trainer_unavail_list=[], task_to_perform="train"
        )
        assert set(r1.keys()) == set(r2.keys())


class TestOortCleanup:
    def test_cleanup_recvd_ends_clears_inflight(self, oort, make_ends):
        oort.selected_ends.update(["a", "b", "c"])
        oort.ordered_updates_recv_ends = ["a", "b"]
        oort._cleanup_recvd_ends(make_ends(["a", "b", "c"]))
        assert oort.selected_ends == {"c"}
        assert oort.ordered_updates_recv_ends == []
