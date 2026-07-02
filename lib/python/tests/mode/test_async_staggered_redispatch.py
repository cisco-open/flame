# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Event-driven (staggered) re-dispatch for the felix async aggregator.

The round boundary used to re-dispatch the whole freed cohort at one frozen
round-start vclock, collapsing the per-trainer completion stagger real keeps.
With ``simStaggeredRedispatch`` on, each TRAIN dispatch is stamped at the vclock
at which its slot freed (a prior commit), so ``sct = sim_send_ts + compute``
regains the spread. These tests pin: (1) the FIFO pop helper, (2) the commit-side
free-slot push, (3) per-end staggered ``sim_send_ts`` on the train path, and the
three invariants the fix must preserve — eval/flag-off/real all stay batched, and
sync stays batched by construction.
"""

import inspect
import time
import types
from collections import deque

import pytest

from flame.mode.horizontal.asyncfl import top_aggregator as async_mod
from flame.mode.horizontal.asyncfl.top_aggregator import TopAggregator
from flame.mode.message import MessageType
from flame.sim import SimReorderBuffer, VirtualClock
from flame.selector.properties import PROP_SIM_SEND_TS

# Reuse the fake channel / drain harness from the receive-ordering test.
from tests.mode.test_async_sim_ordering import FakeChannel, _ConcreteAgg, _make_agg, _drain


# ── 1. _pop_free_slot_ts (pure FIFO helper) ──────────────────────────

class TestPopFreeSlotTs:
    def _agg(self, stamps):
        agg = _ConcreteAgg.__new__(_ConcreteAgg)
        agg._sim_free_slot_ts = deque(stamps, maxlen=128)
        return agg

    def test_fifo_oldest_first(self):
        agg = self._agg([10.0, 12.0, 14.0])
        assert agg._pop_free_slot_ts(100.0) == 10.0
        assert agg._pop_free_slot_ts(100.0) == 12.0
        assert agg._pop_free_slot_ts(100.0) == 14.0

    def test_empty_falls_back_to_now(self):
        agg = self._agg([])
        assert agg._pop_free_slot_ts(42.0) == 42.0

    def test_clamps_to_now(self):
        # A stamp can never legitimately exceed the live vclock; clamp defensively.
        agg = self._agg([200.0])
        assert agg._pop_free_slot_ts(100.0) == 100.0


# ── 2. commit-side free-slot push (via the real _sim_recv_min) ────────

def _make_stag_agg(staggered):
    agg = _make_agg()
    agg._sim_staggered_redispatch = staggered
    agg._sim_free_slot_ts = deque(maxlen=128)
    agg._sim_last_commit_sct = {}
    return agg


class TestCommitPushesFreeSlot:
    def test_train_commit_pushes_advanced_vclock(self):
        durations = {"t1": 10.0, "t2": 5.0, "t3": 25.0}
        agg = _make_stag_agg(staggered=True)
        channel = FakeChannel(set(durations), list(durations.items()))
        committed, t_v = _drain(agg, channel)
        # One stamp per committed train update, each == the vclock after that
        # commit, i.e. the ascending committed completion times.
        assert list(agg._sim_free_slot_ts) == [5.0, 10.0, 25.0]
        # last-commit sct recorded per end for held_s telemetry.
        assert agg._sim_last_commit_sct == {"t1": 10.0, "t2": 5.0, "t3": 25.0}

    def test_flag_off_pushes_nothing(self):
        durations = {"t1": 10.0, "t2": 5.0}
        agg = _make_stag_agg(staggered=False)
        channel = FakeChannel(set(durations), list(durations.items()))
        _drain(agg, channel)
        assert list(agg._sim_free_slot_ts) == []          # no stagger bookkeeping
        assert agg._sim_last_commit_sct == {"t1": 10.0, "t2": 5.0}  # cheap, always on


# ── 3. _distribute_weights per-end stamping ──────────────────────────

class _DistChannel:
    """Minimal channel for _distribute_weights: records every SIM_SEND_TS."""

    def __init__(self, send_ends):
        self._send_ends = list(send_ends)
        self.properties = {}
        self.props = {}      # end -> {key: value}
        self.sent = {}       # end -> payload
        self._selector = types.SimpleNamespace()

    def await_join(self):
        pass

    def ends(self, state, task=None, agg_version_state=None, trainer_version_states=None):
        return list(self._send_ends)

    def dumps(self, msg):
        return dict(msg)

    def send_payload(self, end, payload):
        self.sent[end] = payload

    def set_end_property(self, end, key, value):
        self.props.setdefault(end, {})[key] = value

    def set_curr_unavailable_trainers(self, trainer_unavail_list=None):
        self.unavail = list(trainer_unavail_list or [])

    def has(self, end):
        return end in self._send_ends


def _make_dist_agg(channel, *, staggered, simulated=True, free_slots=()):
    agg = _ConcreteAgg.__new__(_ConcreteAgg)
    agg.simulated = simulated
    agg.agg_start_time_ts = time.time()
    agg._round = 5
    agg._vclock = VirtualClock()
    agg._vclock.advance(100.0)
    agg._sim_staggered_redispatch = staggered
    agg._sim_inflight_residence = False
    agg._sim_free_slot_ts = deque(free_slots, maxlen=128)
    agg._sim_last_commit_sct = {}
    agg._sim_inflight_expected = {}
    agg._sim_trainer_budget = {}
    agg._sim_budget_min = 12.0
    agg._sim_redispatch_gap_s = 0.0
    agg._sim_cooldown_until = {}
    agg._real_distribute_settle_s = 0.0
    agg.trainer_event_dict = None
    agg._track_trainer_version_duration_s = {}
    agg.cm = types.SimpleNamespace(get_by_tag=lambda tag: channel)
    # Stub the selection-adjacent helpers _distribute_weights calls.
    agg._await_min_trainers = lambda ch: None
    agg._update_weights = lambda: None
    agg._inject_oracle_utilities = lambda ch, task: None
    agg.weights = {}
    return agg


def _sent_ts(channel):
    return [channel.props[e][PROP_SIM_SEND_TS] for e in channel._send_ends]


@pytest.fixture(autouse=True)
def _identity_weights(monkeypatch):
    # Avoid needing a live ML framework: weights_to_device is identity here.
    monkeypatch.setattr(async_mod, "weights_to_device", lambda w, d: w)


class TestDistributeStagger:
    def test_train_staggered_assigns_freed_slot_stamps_fifo(self):
        ends = ["e1", "e2", "e3"]
        ch = _DistChannel(ends)
        agg = _make_dist_agg(ch, staggered=True, free_slots=[91.0, 95.0, 98.0])
        agg._distribute_weights("tag", "train")
        assert _sent_ts(ch) == [91.0, 95.0, 98.0]          # distinct, spread
        # gate-expected completion uses each end's own staggered stamp.
        assert agg._sim_inflight_expected["e1"] == pytest.approx(91.0 + 12.0)
        assert agg._sim_inflight_expected["e3"] == pytest.approx(98.0 + 12.0)
        # the per-end payload carries that end's stamp.
        assert ch.sent["e2"][MessageType.SIM_SEND_TS] == 95.0

    def test_train_staggered_falls_back_to_now_when_queue_drains(self):
        ends = ["e1", "e2", "e3"]
        ch = _DistChannel(ends)
        agg = _make_dist_agg(ch, staggered=True, free_slots=[91.0])  # only one stamp
        agg._distribute_weights("tag", "train")
        assert _sent_ts(ch) == [91.0, 100.0, 100.0]        # rest ride the frontier

    def test_flag_off_train_is_batched(self):
        ends = ["e1", "e2", "e3"]
        ch = _DistChannel(ends)
        agg = _make_dist_agg(ch, staggered=False, free_slots=[91.0, 95.0])
        agg._distribute_weights("tag", "train")
        assert _sent_ts(ch) == [100.0, 100.0, 100.0]       # one frozen frontier
        assert list(agg._sim_free_slot_ts) == [91.0, 95.0]  # queue untouched

    def test_eval_is_batched_even_with_flag_on(self):
        ends = ["e1", "e2", "e3"]
        ch = _DistChannel(ends)
        agg = _make_dist_agg(ch, staggered=True, free_slots=[91.0, 95.0])
        agg._distribute_weights("tag", "eval")
        assert _sent_ts(ch) == [100.0, 100.0, 100.0]       # eval is not staggered
        assert list(agg._sim_free_slot_ts) == [91.0, 95.0]  # FIFO reserved for train

    def test_real_mode_unaffected(self):
        ends = ["e1", "e2"]
        ch = _DistChannel(ends)
        agg = _make_dist_agg(ch, staggered=True, simulated=False, free_slots=[91.0])
        agg._distribute_weights("tag", "train")
        # real never stamps PROP_SIM_SEND_TS.
        assert all(PROP_SIM_SEND_TS not in ch.props.get(e, {}) for e in ends)


# ── 4. sync stays batched (regression guard, by construction) ─────────

class TestSyncStaysBatched:
    @pytest.mark.parametrize(
        "module",
        ["flame.mode.horizontal.syncfl.top_aggregator",
         "flame.mode.horizontal.oort.top_aggregator"],
    )
    def test_sync_distribute_does_not_stagger(self, module):
        import importlib
        mod = importlib.import_module(module)
        src = inspect.getsource(mod.TopAggregator._distribute_weights)
        # A sync round IS a synchronized cohort (barrier): one shared sim_send_ts,
        # never the async per-slot stagger.
        assert "_sim_send_ts = self._vclock.now" in src
        assert "_sim_staggered_redispatch" not in src
