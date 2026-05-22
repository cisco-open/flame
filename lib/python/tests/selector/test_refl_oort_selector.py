# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""REFL_Oort selector tests."""

import pytest

from flame.selector.refl_oort import REFLOortSelector


@pytest.fixture
def refl_oort():
    return REFLOortSelector(aggr_num=3, avail_priority=0)


class TestREFLOortInit:
    def test_defaults(self, refl_oort):
        assert refl_oort.aggr_num == 3
        assert refl_oort.avail_priority == 0
        assert refl_oort.avail_tracker is None
        assert isinstance(refl_oort.selected_ends, set)


class TestREFLOortFiltersUnavail:
    def test_unavail_excluded(self, refl_oort, make_ends, channel_props):
        ends = make_ends(count=6, prefix="t")
        unavail = ["t0", "t1"]
        result = refl_oort.select(
            ends, channel_props, trainer_unavail_list=unavail, task_to_perform="train"
        )
        for end_id in result:
            assert end_id not in unavail


class TestREFLOortInflightExcluded:
    def test_inflight_not_reselected_next_round(
        self, refl_oort, make_ends, channel_props
    ):
        ends = make_ends(count=10, prefix="t")
        refl_oort.select(
            ends, channel_props, trainer_unavail_list=[], task_to_perform="train"
        )
        first = set(refl_oort.selected_ends)

        channel_props["round"] = 2
        refl_oort.newly_selected_this_round = set()
        result = refl_oort.select(
            ends, channel_props, trainer_unavail_list=[], task_to_perform="train"
        )
        for end_id in result:
            assert end_id not in first
