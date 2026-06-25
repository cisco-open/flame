# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""One-in-flight-per-trainer invariant (simInflightResidence) for the felix
async stack.

Real keeps a trainer out of selection (VAL_CH_STATE_SEND) from the moment it is
dispatched until its update returns AND is aggregated — measured 0% overlapping
in-flight intervals over a full run. Sim, where a trainer is freed instantly
(no train sleep), re-dispatched a still-in-flight fast trainer 13.9% of the time.

A BUSY trainer (compute task outstanding) is AVL_TRAIN/AVL_EVAL but temporarily
occupied; it must hold its concurrency slot in the selector's ``selected_ends``
(concurrency budgeted as ``extra = c - len(selected_ends)``), NOT be marked
UN_AVL (which is for trainers that cannot participate at all). Marking busy
trainers unavailable frees their slot and the selector over-selects toward N
(in-flight ballooned to ~300, clock crawled). These tests pin that
``_sim_hold_busy_slots`` holds busy trainers via the slot — and never via the
unavailable list — with the flag widening the held set from "already buffered" to
"all dispatched-but-not-committed (train+eval)".
"""

import types

import pytest

import flame.mode.horizontal.asyncfl.top_aggregator as async_mod
from flame.end import KEY_END_STATE, VAL_END_STATE_NONE
from tests.mode.test_async_staggered_redispatch import _DistChannel, _make_dist_agg


@pytest.fixture(autouse=True)
def _identity_weights(monkeypatch):
    # Avoid needing a live ML framework: weights_to_device is identity here.
    monkeypatch.setattr(async_mod, "weights_to_device", lambda w, d: w)


class _FakeEnd:
    def __init__(self):
        self._props = {}

    def set_property(self, key, value):
        self._props[key] = value

    def get_property(self, key):
        return self._props.get(key)


class _HoldChannel:
    """Minimal channel for _sim_hold_busy_slots: a selector + ends map."""

    def __init__(self, all_ends, selected, all_selected, requester="agg"):
        self._ends = {e: _FakeEnd() for e in all_ends}
        self._selector = types.SimpleNamespace(
            requester=requester,
            selected_ends={requester: set(selected)},
            all_selected=dict(all_selected),
        )

    def has(self, end):
        return end in self._ends


def _make_hold_agg(buffered, inflight_expected, residence):
    from tests.mode.test_async_sim_ordering import _make_agg

    agg = _make_agg()
    agg.simulated = True
    agg._round = 5
    agg._sim_inflight_residence = residence
    agg._sim_pending_commit = set()
    agg._sim_inflight_expected = dict(inflight_expected)
    # Fake reorder buffer exposing only pending_ends().
    agg._sim_buffer = types.SimpleNamespace(pending_ends=lambda: list(buffered))
    return agg


def _slot(channel):
    return channel._selector.selected_ends["agg"]


class TestHoldBusySlots:
    def test_residence_holds_full_outstanding_set_in_slot(self):
        # e1 buffered; e2 dispatched-but-not-yet-buffered; e3 free.
        ch = _HoldChannel(["e1", "e2", "e3"], selected=[], all_selected={})
        agg = _make_hold_agg(
            buffered=["e1"], inflight_expected={"e1": 110.0, "e2": 120.0}, residence=True
        )
        agg._sim_hold_busy_slots(ch)
        # Both busy trainers hold a concurrency slot; e3 (free) does not.
        assert _slot(ch) == {"e1", "e2"}
        assert set(ch._selector.all_selected) == {"e1", "e2"}
        assert agg._sim_pending_commit == {"e1", "e2"}

    def test_flag_off_holds_only_buffered(self):
        ch = _HoldChannel(["e1", "e2"], selected=[], all_selected={})
        agg = _make_hold_agg(
            buffered=["e1"], inflight_expected={"e1": 110.0, "e2": 120.0}, residence=False
        )
        agg._sim_hold_busy_slots(ch)
        # e2 (dispatched, not buffered) is NOT held when the flag is off.
        assert _slot(ch) == {"e1"}
        assert agg._sim_pending_commit == {"e1"}

    def test_eval_outstanding_is_held(self):
        # _sim_inflight_expected mixes train+eval dispatches; both are busy.
        ch = _HoldChannel(["ev1", "tr1"], selected=[], all_selected={})
        agg = _make_hold_agg(
            buffered=[], inflight_expected={"ev1": 130.0, "tr1": 125.0}, residence=True
        )
        agg._sim_hold_busy_slots(ch)
        assert _slot(ch) == {"ev1", "tr1"}

    def test_committed_trainer_releases_its_slot(self):
        # e1 was holding a slot last round; it committed (gone from buffer AND
        # popped from _sim_inflight_expected) -> released so the pool can refill it.
        ch = _HoldChannel(
            ["e1", "e2"], selected=["e1", "e2"], all_selected={"e1": 1.0, "e2": 1.0}
        )
        agg = _make_hold_agg(
            buffered=["e2"], inflight_expected={"e2": 120.0}, residence=True
        )
        agg._sim_hold_busy_slots(ch)
        assert _slot(ch) == {"e2"}                       # e1 released, e2 still busy
        assert set(ch._selector.all_selected) == {"e2"}
        assert ch._ends["e1"].get_property(KEY_END_STATE) == VAL_END_STATE_NONE

    def test_nothing_busy_is_noop(self):
        ch = _HoldChannel(["e1"], selected=[], all_selected={})
        agg = _make_hold_agg(buffered=[], inflight_expected={}, residence=True)
        agg._sim_hold_busy_slots(ch)
        assert _slot(ch) == set()
        assert agg._sim_pending_commit == set()


class TestBusyNeverMarkedUnavailable:
    """The semantic correction: busy (AVL_TRAIN/AVL_EVAL) trainers must never go on
    the unavailable (UN_AVL) list — _distribute_weights must not add in-flight
    trainers there, regardless of the residence flag."""

    @pytest.mark.parametrize("residence", [True, False])
    def test_distribute_does_not_mark_inflight_unavailable(self, residence):
        ch = _DistChannel(["e2"])
        agg = _make_dist_agg(ch, staggered=False)
        agg._sim_inflight_residence = residence
        agg._sim_inflight_expected = {"e1": 150.0, "e3": 160.0}
        agg._distribute_weights("tag", "train")
        assert ch.unavail == []

    def test_distribute_does_not_touch_inflight_entries(self):
        ch = _DistChannel(["e2"])
        agg = _make_dist_agg(ch, staggered=False)
        agg._sim_inflight_residence = True
        agg._sim_inflight_expected = {"e1": 150.0}
        agg._distribute_weights("tag", "train")
        assert agg._sim_inflight_expected["e1"] == 150.0  # only a commit pops it
        assert "e1" not in ch.sent                        # not re-dispatched
