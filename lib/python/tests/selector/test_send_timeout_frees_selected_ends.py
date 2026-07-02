# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Regression: the 90s SEND_TIMEOUT_WAIT_S abandon in async_oort.py/fedbuff.py
must free the end from selected_ends, not just all_selected -- recv_ends
derives from selected_ends, so leaving it there hangs recv_fifo forever
(UNAVAILABILITY_DESIGN.md, Open A)."""

import time

import pytest


class _StopAfterAbandon(Exception):
    """Raised from a stubbed pacer() to inspect state right after the abandon
    block runs, without driving async_oort's heavier downstream selection
    (same technique as TestCoolingHoldsConcurrency in test_async_sim_ordering.py)."""


def _boom():
    raise _StopAfterAbandon()


class TestAsyncOortSendTimeoutFreesSelectedEnds:
    @staticmethod
    def _stub_selector():
        from flame.selector.async_oort import AsyncOortSelector

        sel = AsyncOortSelector.__new__(AsyncOortSelector)
        sel.requester = "agg"
        sel.selected_ends = {"agg": {"stale"}}
        sel.all_selected = {"stale": time.time() - 100}  # past 90s SEND_TIMEOUT_WAIT_S
        sel.ordered_updates_recv_ends = []
        sel.track_trainer_timeouts = {}
        sel.pacer = _boom
        return sel

    def test_stale_end_leaves_both_all_selected_and_selected_ends(self, make_ends):
        sel = self._stub_selector()
        ends = make_ends(["stale", "fresh"])

        with pytest.raises(_StopAfterAbandon):
            sel._handle_send_state(
                ends=ends,
                concurrency=2,
                channel_props={"round": 1},
                trainer_unavail_list=[],
                task_to_perform="train",
                agg_version_state=(1, 0, 0),
                trainer_version_states={},
            )

        assert "stale" not in sel.all_selected
        assert "stale" not in sel.selected_ends["agg"]

    def test_fresh_end_untouched(self, make_ends):
        sel = self._stub_selector()
        sel.all_selected["fresh"] = time.time()  # well within the 90s window
        sel.selected_ends["agg"].add("fresh")
        ends = make_ends(["stale", "fresh", "third"])

        with pytest.raises(_StopAfterAbandon):
            sel._handle_send_state(
                ends=ends,
                concurrency=3,  # 2 already selected -> extra=1, past the extra==0 short-circuit
                channel_props={"round": 1},
                trainer_unavail_list=[],
                task_to_perform="train",
                agg_version_state=(1, 0, 0),
                trainer_version_states={},
            )

        assert "fresh" in sel.all_selected
        assert "fresh" in sel.selected_ends["agg"]


class TestFedBuffSendTimeoutFreesSelectedEnds:
    @staticmethod
    def _stub_selector():
        from flame.selector.fedbuff import FedBuffSelector

        sel = FedBuffSelector(c=2, aggGoal=1)
        sel.requester = "agg"
        sel.selected_ends = {"agg": {"stale"}}
        sel.all_selected = {"stale": time.time() - 100}  # past 90s SEND_TIMEOUT_WAIT_S
        return sel

    def test_stale_end_leaves_both_all_selected_and_selected_ends(self, make_ends):
        sel = self._stub_selector()
        ends = make_ends(["stale", "fresh"])

        # concurrency=1 with 1 already selected -> extra=0, fixed before the
        # abandon loop runs, so the freed slot can't be immediately re-filled
        # by the candidate loop below (a separate, correct RNG-dependent path).
        sel._handle_send_state(ends=ends, concurrency=1, connected_ends=ends)

        assert "stale" not in sel.all_selected
        assert "stale" not in sel.selected_ends["agg"]
