# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for sync, set-based selectors.

Random and async selectors use dict-keyed selected_ends and are tested
separately. This module covers Default, Oort, REFL_Oort, and FedDance
(once registered).
"""

import pytest

from flame.selector import AbstractSelector


def _build_default():
    from flame.selector.default import DefaultSelector
    return DefaultSelector()


def _build_oort():
    from flame.selector.oort import OortSelector
    return OortSelector(aggr_num=3)


def _build_refl_oort():
    from flame.selector.refl_oort import REFLOortSelector
    return REFLOortSelector(aggr_num=3, avail_priority=0)


BUILDERS = [
    pytest.param(_build_default, id="default"),
    pytest.param(_build_oort, id="oort"),
    pytest.param(_build_refl_oort, id="refl_oort"),
]


@pytest.fixture(params=BUILDERS)
def selector(request):
    return request.param()


class TestAbstractSelectorContract:
    def test_is_abstract_selector(self, selector):
        assert isinstance(selector, AbstractSelector)

    def test_selected_ends_is_set(self, selector):
        assert isinstance(selector.selected_ends, set)

    def test_ordered_updates_recv_ends_is_list(self, selector):
        assert isinstance(selector.ordered_updates_recv_ends, list)


class TestLifecycleHooks:
    def test_on_update_received_appends(self, selector):
        selector.on_update_received("end1", {}, 1)
        assert "end1" in selector.ordered_updates_recv_ends

    def test_on_round_completed_frees_received(self, selector, make_ends):
        selector.selected_ends.update(["e1", "e2"])
        selector.ordered_updates_recv_ends = ["e1"]
        selector.on_round_completed(make_ends(["e1", "e2"]), 1)
        assert "e1" not in selector.selected_ends
        assert "e2" in selector.selected_ends
        assert selector.ordered_updates_recv_ends == []

    def test_on_round_completed_with_nothing_pending(self, selector, make_ends):
        selector.on_round_completed(make_ends(["a"]), 1)
        assert selector.selected_ends == set()


class TestSelectEmptyEnds:
    def test_returns_empty_dict(self, selector, channel_props):
        result = selector.select(
            {}, channel_props, trainer_unavail_list=[], task_to_perform="train"
        )
        assert result == {}


class TestSelectReturnsValidSubset:
    def test_returned_ends_are_subset(self, selector, make_ends, channel_props):
        ends = make_ends(count=10, prefix="t")
        result = selector.select(
            ends, channel_props, trainer_unavail_list=[], task_to_perform="train"
        )
        for end_id in result:
            assert end_id in ends
