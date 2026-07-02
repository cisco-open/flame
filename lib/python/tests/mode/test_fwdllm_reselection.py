# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Tests for the `reselect_each_iteration` selection-granularity gate:
per-round (False) selects once and reuses the same trainer set for the
whole round; per-iteration (True, default) re-invokes the selector every
call."""

import time

from flame.config import TrainerAvailState
from flame.mode.horizontal.syncfl.fwdllm_aggregator import (
    ROUND_CACHE_STUCK_TIMEOUT_S,
    TopAggregator,
)


class _FakeSelector:
    """Stand-in for RandomSelector exposing only `selected_ends`."""

    def __init__(self):
        self.selected_ends = set()


class _FakeChannel:
    """Records each `ends()` call and returns the next canned selection.

    Also models end departure: `_removed` mimics a fully disconnected end
    (as `channel.remove()` would leave it -- `has()` returns False);
    `_unavail` mimics an end that explicitly reported `UN_AVL` but is
    still connected (as `channel.update_state()` would leave it -- `has()`
    still True, but its avl-state property reads `UN_AVL`).
    """

    def __init__(self, selections):
        self._selections = list(selections)
        self.calls = 0
        self._selector = _FakeSelector()
        self._removed = set()
        self._unavail = set()

    def ends(self, state, task_to_perform):
        self.calls += 1
        return self._selections[min(self.calls - 1, len(self._selections) - 1)]

    def has(self, end_id):
        return end_id not in self._removed

    def get_end_property(self, end_id, key):
        if end_id in self._unavail:
            return TrainerAvailState.UN_AVL
        return None


class _FakeAggregator:
    """Minimal stand-in exposing only the state
    `_select_ends_respecting_reselect_gate` touches."""

    def __init__(self, reselect_each_iteration, agg_goal=None):
        self._reselect_each_iteration = reselect_each_iteration
        self._round_selected_ends = None
        self._round_selected_ends_round = None
        self._round_cache_activity_ts = {}
        self._round = 0
        if agg_goal is not None:
            self._agg_goal = agg_goal

    select = TopAggregator._select_ends_respecting_reselect_gate
    _rearm_recv_eligibility = staticmethod(TopAggregator._rearm_recv_eligibility)
    _prune_departed_from_round_cache = (
        TopAggregator._prune_departed_from_round_cache
    )


def _drive_two_databins_two_iterations(agg, channel):
    """2 databins x 2 iterations each, within one round."""
    for _databin in range(2):
        for _iteration in range(2):
            agg.select(channel, "train")


class TestReselectGate:
    def test_per_round_selects_once_then_again_after_rollover(self):
        agg = _FakeAggregator(reselect_each_iteration=False)
        channel = _FakeChannel(selections=[["t1", "t2"]])

        _drive_two_databins_two_iterations(agg, channel)
        assert channel.calls == 1
        assert agg.select(channel, "train") == ["t1", "t2"]
        assert channel.calls == 1

        # Round rollover: advance self._round, as
        # _process_aggregation_goal_met does at the databin-wraparound point.
        agg._round += 1
        channel._selections = [["t3", "t4"]]
        channel.calls = 0
        ends = agg.select(channel, "train")
        assert ends == ["t3", "t4"]
        assert channel.calls == 1

        # Still cached for the rest of the new round.
        agg.select(channel, "train")
        assert channel.calls == 1

    def test_per_iteration_selects_every_call(self):
        agg = _FakeAggregator(reselect_each_iteration=True)
        channel = _FakeChannel(selections=[["t1"], ["t2"], ["t3"], ["t4"]])

        _drive_two_databins_two_iterations(agg, channel)
        assert channel.calls == 4

    def test_per_round_does_not_cache_empty_selection(self):
        """An empty/None selection (no trainers joined yet) must not be
        cached as the round's selection -- retry on the next call."""
        agg = _FakeAggregator(reselect_each_iteration=False)
        channel = _FakeChannel(selections=[None, None, ["t1"]])

        assert agg.select(channel, "train") is None
        assert agg.select(channel, "train") is None
        assert agg.select(channel, "train") == ["t1"]
        assert channel.calls == 3

        # Now cached -- a 4th call must not invoke the selector again.
        assert agg.select(channel, "train") == ["t1"]
        assert channel.calls == 3

    def test_per_round_accumulates_partial_selections_until_agg_goal(self):
        """Must keep merging in newly-selected trainers until the cache
        reaches `_agg_goal`, not freeze on the first partial result."""
        agg = _FakeAggregator(reselect_each_iteration=False, agg_goal=3)
        channel = _FakeChannel(selections=[["t1"], ["t2"], ["t3"]])

        assert agg.select(channel, "train") == ["t1"]
        assert agg.select(channel, "train") == ["t1", "t2"]
        assert agg.select(channel, "train") == ["t1", "t2", "t3"]
        assert channel.calls == 3

        # Cache has reached agg_goal -- further calls must not re-query.
        assert agg.select(channel, "train") == ["t1", "t2", "t3"]
        assert channel.calls == 3

    def test_cache_hit_rearms_selector_recv_eligibility(self):
        """Regression test (2026-06-28 live-run hang): a cache hit must
        re-arm selected_ends, or it drains to empty after one pass and
        the receive side permanently stalls."""
        agg = _FakeAggregator(reselect_each_iteration=False, agg_goal=2)
        channel = _FakeChannel(selections=[["t1", "t2"]])

        ends = agg.select(channel, "train")
        assert ends == ["t1", "t2"]
        assert channel._selector.selected_ends == {"t1", "t2"}

        # Simulate cleanup_recvd_end draining both after iteration 0.
        channel._selector.selected_ends.clear()
        assert channel._selector.selected_ends == set()

        ends = agg.select(channel, "train")
        assert ends == ["t1", "t2"]
        assert channel.calls == 1  # still cache-hit, no re-query
        assert channel._selector.selected_ends == {"t1", "t2"}

    def test_accumulate_path_also_rearms_selector_recv_eligibility(self):
        """The accumulate-until-agg_goal path re-arms too, so behavior
        doesn't depend on which branch is taken."""
        agg = _FakeAggregator(reselect_each_iteration=False, agg_goal=2)
        channel = _FakeChannel(selections=[["t1"], ["t2"]])

        agg.select(channel, "train")
        assert channel._selector.selected_ends == {"t1"}

        agg.select(channel, "train")
        assert channel._selector.selected_ends == {"t1", "t2"}


class TestStaleCachePruning:
    """Regression tests for the `_round_selected_ends` stale-cache gap:
    a cached-but-departed end must be pruned so the cache-size check stops
    reporting "full" and the round can backfill the freed slot, instead of
    stalling forever on a contribution that can never arrive."""

    def test_disconnected_end_pruned_and_backfilled(self):
        agg = _FakeAggregator(reselect_each_iteration=False, agg_goal=2)
        channel = _FakeChannel(selections=[["t1", "t2"], ["t3"]])

        assert agg.select(channel, "train") == ["t1", "t2"]
        assert channel.calls == 1

        # t1 fully disconnects (channel.remove() -- has() now False).
        channel._removed.add("t1")

        ends = agg.select(channel, "train")
        assert ends == ["t2", "t3"]
        assert channel.calls == 2  # re-queried the selector to backfill

        # Now cached again at agg_goal -- no further re-query.
        assert agg.select(channel, "train") == ["t2", "t3"]
        assert channel.calls == 2

    def test_un_avl_end_pruned_and_backfilled(self):
        agg = _FakeAggregator(reselect_each_iteration=False, agg_goal=2)
        channel = _FakeChannel(selections=[["t1", "t2"], ["t3"]])

        assert agg.select(channel, "train") == ["t1", "t2"]
        assert channel.calls == 1

        # t1 stays connected but reports UN_AVL (channel.update_state()).
        channel._unavail.add("t1")

        ends = agg.select(channel, "train")
        assert ends == ["t2", "t3"]
        assert channel.calls == 2

        assert agg.select(channel, "train") == ["t2", "t3"]
        assert channel.calls == 2

    def test_still_present_end_not_pruned(self):
        """Control case: no departure means no pruning and no re-query."""
        agg = _FakeAggregator(reselect_each_iteration=False, agg_goal=2)
        channel = _FakeChannel(selections=[["t1", "t2"]])

        assert agg.select(channel, "train") == ["t1", "t2"]
        assert agg.select(channel, "train") == ["t1", "t2"]
        assert channel.calls == 1


class TestStuckCachePruning:
    """Regression tests for the round-cache scale gap found on a real n=100
    run (see examples/MIGRATING_TO_LAUNCHER.md §9): a cached end that's
    still formally connected (not disconnected, not UN_AVL) but has gone
    ROUND_CACHE_STUCK_TIMEOUT_S without a real accepted contribution --
    e.g. one that never finished receiving its initial weights -- must
    also be pruned/backfilled, not just an explicitly departed one
    (TestStaleCachePruning above)."""

    def test_stuck_end_pruned_and_backfilled(self):
        agg = _FakeAggregator(reselect_each_iteration=False, agg_goal=2)
        channel = _FakeChannel(selections=[["t1", "t2"], ["t3"]])

        assert agg.select(channel, "train") == ["t1", "t2"]
        assert channel.calls == 1

        # t1 is still connected and AVL_TRAIN (neither departure check in
        # TestStaleCachePruning would catch it), but has gone well past the
        # stuck-timeout without a real accepted contribution.
        agg._round_cache_activity_ts["t1"] = time.time() - (ROUND_CACHE_STUCK_TIMEOUT_S + 10)

        ends = agg.select(channel, "train")
        assert ends == ["t2", "t3"]
        assert channel.calls == 2  # re-queried the selector to backfill
        assert "t1" not in agg._round_cache_activity_ts  # cleaned up on prune

        assert agg.select(channel, "train") == ["t2", "t3"]
        assert channel.calls == 2

    def test_recently_active_end_not_pruned(self):
        """An end within the timeout window (even if not the most recent to
        contribute) must not be pruned -- only genuinely stuck members
        should churn the cache."""
        agg = _FakeAggregator(reselect_each_iteration=False, agg_goal=2)
        channel = _FakeChannel(selections=[["t1", "t2"]])

        assert agg.select(channel, "train") == ["t1", "t2"]
        agg._round_cache_activity_ts["t1"] = time.time() - (ROUND_CACHE_STUCK_TIMEOUT_S - 30)

        assert agg.select(channel, "train") == ["t1", "t2"]
        assert channel.calls == 1  # no re-query -- nothing pruned

    def test_freshly_cached_end_not_immediately_pruned(self):
        """Control: an end that just entered the cache (activity_ts == now,
        set by the accumulate path itself) must not be immediately treated
        as stuck on the very next call."""
        agg = _FakeAggregator(reselect_each_iteration=False, agg_goal=2)
        channel = _FakeChannel(selections=[["t1", "t2"]])

        assert agg.select(channel, "train") == ["t1", "t2"]
        assert set(agg._round_cache_activity_ts.keys()) == {"t1", "t2"}

        assert agg.select(channel, "train") == ["t1", "t2"]
        assert channel.calls == 1
