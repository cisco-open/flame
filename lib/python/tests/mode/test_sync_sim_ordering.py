# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Sync-aggregator simulated-mode receive ordering.

Exercises the real ``TopAggregator._sync_sim_recv_first_k`` (syncfl) against a
fake channel to prove that, regardless of physical arrival order, the sync
aggregator commits the ``first_k`` updates with the SMALLEST sim_completion_ts
(the k that would physically finish first in real mode) and advances its virtual
clock to the k-th smallest. This is the sync analogue of the async ordering
guarantee and is what makes simulated mode decision-equivalent to real mode for
the sync baselines (fedavg / oort / refl / feddance).
"""

import itertools
from collections import defaultdict

import pytest

from flame.mode.horizontal.syncfl.top_aggregator import TopAggregator
from flame.mode.message import MessageType
from flame.selector.properties import PROP_SIM_SEND_TS, PROP_ROUND_DURATION
from flame.sim import VirtualClock


class FakeSyncChannel:
    """Delivers one update per selected end (with a sim_completion_ts) in a
    fixed physical arrival order; ``recv_fifo`` drains all ready ones then
    signals a timeout with (None, ...)."""

    def __init__(self, scts, arrival_order, round_durations=None, sim_send_ts=0.0):
        self._scts = dict(scts)
        self._rd = dict(round_durations or {})
        self._queue = list(arrival_order)
        self._props = defaultdict(dict)
        for e in self._scts:
            self._props[e][PROP_SIM_SEND_TS] = sim_send_ts

    def has(self, e):
        return e in self._scts

    def ends(self, *a):
        return list(self._scts)

    def get_end_property(self, end, key):
        return self._props[end].get(key)

    def set_end_property(self, end, key, val):
        self._props[end][key] = val

    def recv_fifo(self, end_ids, first_k=0, timeout=None):
        ids = set(end_ids)
        i = 0
        while i < len(self._queue):
            e = self._queue[i]
            if e in ids:
                self._queue.pop(i)
                msg = {MessageType.WEIGHTS: f"w_{e}",
                       MessageType.SIM_COMPLETION_TS: self._scts[e]}
                if e in self._rd:
                    msg[MessageType.SIM_ROUND_DURATION] = self._rd[e]
                yield (msg, (e, None))
            else:
                i += 1
        yield (None, ("", None))  # nothing more ready this pass


class _ConcreteSyncAgg(TopAggregator):
    def check_and_sleep(self):
        pass

    def evaluate(self):
        pass

    def initialize(self):
        pass

    def load_data(self):
        pass

    def train(self):
        pass


def _make_agg():
    agg = _ConcreteSyncAgg.__new__(_ConcreteSyncAgg)
    agg._vclock = VirtualClock()
    agg.simulated = True
    return agg


def _committed_ends(agg, channel, first_k):
    out = agg._sync_sim_recv_first_k(channel, channel.ends(), first_k)
    return [md[0] for _msg, md in out]


SCTS = {"t1": 10.0, "t2": 5.0, "t3": 25.0, "t4": 15.0, "t5": 8.0}
# ascending by sct: t2(5), t5(8), t1(10), t4(15), t3(25)


class TestSyncSimRecvFirstK:
    def test_commits_k_smallest_completion(self):
        agg = _make_agg()
        ch = FakeSyncChannel(SCTS, arrival_order=["t3", "t1", "t4", "t2", "t5"])
        committed = _committed_ends(agg, ch, first_k=3)
        assert committed == ["t2", "t5", "t1"]  # 3 smallest sct, ascending
        assert agg._vclock.now == 10.0  # advanced to the k-th smallest

    def test_independent_of_arrival_order(self):
        expected = ["t2", "t5", "t1"]
        for arrival in itertools.permutations(SCTS):
            agg = _make_agg()
            ch = FakeSyncChannel(SCTS, list(arrival))
            assert _committed_ends(agg, ch, first_k=3) == expected

    def test_k_equals_all(self):
        agg = _make_agg()
        ch = FakeSyncChannel(SCTS, list(SCTS))
        committed = _committed_ends(agg, ch, first_k=5)
        assert committed == ["t2", "t5", "t1", "t4", "t3"]
        assert agg._vclock.now == 25.0

    def test_round_duration_set_from_sim_round_duration(self):
        agg = _make_agg()
        rd = {"t2": 5.0, "t5": 8.0, "t1": 10.0, "t3": 25.0, "t4": 15.0}
        ch = FakeSyncChannel(SCTS, list(SCTS), round_durations=rd, sim_send_ts=0.0)
        agg._sync_sim_recv_first_k(ch, ch.ends(), first_k=2)
        # committed t2,t5 get PROP_ROUND_DURATION from SIM_ROUND_DURATION
        assert ch.get_end_property("t2", PROP_ROUND_DURATION).total_seconds() == 5.0
        assert ch.get_end_property("t5", PROP_ROUND_DURATION).total_seconds() == 8.0

    def test_fewer_responders_than_k(self):
        # Only 2 of the 5 selected ever respond. Real mode's recv_fifo(first_k=3)
        # would block forever; the sim barrier drains what's ready (the recv_fifo
        # returns (None,...) once nothing more is queued) and commits the 2
        # smallest by sct. No fixed deadline needed — the grace is the dead-end
        # ceiling and the fake signals "no more ready" immediately.
        agg = _make_agg()
        ch = FakeSyncChannel(SCTS, arrival_order=["t1", "t2"])  # only 2 queued
        committed = _committed_ends(agg, ch, first_k=3)
        assert committed == ["t2", "t1"]  # the 2 smallest sct, ascending
        assert agg._vclock.now == 10.0


class TestSimInflightResidence:
    """PARITY §4.5: in sim the oort/refl aggregator marks trainers still computing
    in sim time (buffered sct > vclock) as UNAVAILABLE for selection, so a slow
    client stays out of the eligible pool until ``vclock >= sct`` — matching real,
    where it is genuinely busy. Excluded via the unavailable list (NOT selected_ends,
    which would re-dispatch it). Guards the buffer-driven hold/release invariant the
    aggregator snippet relies on (`SimReorderBuffer.pending_after` merged into the
    trainer_unavail_list)."""

    def _unavail(self, buf, base_unavail, vclock_now):
        # Mirror of oort/top_aggregator._distribute_weights §4.5 snippet.
        held = buf.pending_after(vclock_now)
        merged = list(set(base_unavail) | held) if held else list(base_unavail)
        return held, merged

    def test_slow_trainer_held_then_released(self):
        from flame.sim import SimReorderBuffer

        buf = SimReorderBuffer()
        buf.add("fast", 4.0)     # already complete at vclock 5
        buf.add("slow", 30.0)    # still computing at vclock 5

        # Round N (vclock=5): the slow trainer is held unavailable; the fast one is
        # not (it is available to commit, not still computing). Base unavail merged.
        held, merged = self._unavail(buf, base_unavail=["pre"], vclock_now=5.0)
        assert held == {"slow"}
        assert "slow" in merged          # excluded from selection / pool
        assert "fast" not in merged      # available, not held
        assert "pre" in merged           # additive merge, not overwrite

        # Once the clock passes its sct (it commits), it is no longer held.
        buf.discard("slow")  # committed -> leaves the buffer
        held2, merged2 = self._unavail(buf, base_unavail=["pre"], vclock_now=35.0)
        assert held2 == set() and merged2 == ["pre"]

    def test_noop_when_nothing_still_computing(self):
        from flame.sim import SimReorderBuffer

        buf = SimReorderBuffer()
        buf.add("a", 2.0)
        held, merged = self._unavail(buf, base_unavail=["x"], vclock_now=10.0)
        assert held == set() and merged == ["x"]  # untouched


class TestSimInflightCarryover:
    """PARITY §4.9: the sim oort/refl aggregator must CARRY a prior-round straggler
    that is still computing at this round's start (modeled sct > vclock_round_start)
    rather than deliver its physically-instant update and stale-clean it. Without the
    carry-over gate, sim in-flight drains to ~0 while real holds ~3 (overcommit). This
    drives the REAL ``OortTopAggregator._oort_sim_recv`` generator off a pre-loaded
    buffer to prove the gate holds the still-computing straggler and delivers the
    fresh + already-completed ends in ascending sct, advancing the clock only to the
    delivered ones."""

    def _make_oort_agg(self, carryover, round_num, vclock_start):
        from flame.mode.horizontal.oort.top_aggregator import (
            TopAggregator as OortTopAggregator,
        )
        from flame.sim import SimReorderBuffer

        class _ConcreteOortAgg(OortTopAggregator):
            check_and_sleep = evaluate = initialize = load_data = train = (
                lambda self: None
            )

        class _HP:
            def __init__(self, c):
                self.sim_inflight_carryover = c

        class _Cfg:
            def __init__(self, c):
                self.hyperparameters = _HP(c)

        agg = _ConcreteOortAgg.__new__(_ConcreteOortAgg)
        agg._vclock = VirtualClock()
        agg._vclock.advance(vclock_start)
        agg.simulated = True
        agg._round = round_num
        agg.config = _Cfg(carryover)
        agg._sim_buffer = SimReorderBuffer()
        return agg

    def _load(self, agg, items):
        # items: list of (end, sct, model_version)
        for end, sct, mv in items:
            msg = {MessageType.WEIGHTS: f"w_{end}",
                   MessageType.SIM_COMPLETION_TS: sct,
                   MessageType.SIM_ROUND_DURATION: sct,
                   MessageType.MODEL_VERSION: mv}
            agg._sim_buffer.add(end, sct, (msg, (end, None)))

    def _drive(self, agg):
        # buffer pre-loaded; pass the same ends so to_probe is empty (no recv_fifo)
        ch = FakeSyncChannel({}, arrival_order=[])
        ends = list(agg._sim_buffer.pending_ends())
        return [md[0] for _msg, md in agg._oort_sim_recv(ch, ends)]

    # round 2 starting at vclock 5: "done" already completed (sct 4), "fresh" is this
    # round (sct 8), "slow" is a prior-round straggler still computing (sct 30).
    ITEMS = [("done", 4.0, 1), ("fresh", 8.0, 2), ("slow", 30.0, 1)]

    def test_carryover_holds_still_computing_straggler(self):
        agg = self._make_oort_agg(carryover=True, round_num=2, vclock_start=5.0)
        self._load(agg, self.ITEMS)
        committed = self._drive(agg)
        assert committed == ["done", "fresh"]            # ascending sct, slow held
        assert agg._vclock.now == 8.0                    # clock not advanced to 30
        assert agg._sim_buffer.has("slow")               # carried in-flight
        assert not agg._sim_buffer.has("fresh")          # delivered, left buffer

    def test_off_by_default_drains_straggler(self):
        agg = self._make_oort_agg(carryover=False, round_num=2, vclock_start=5.0)
        self._load(agg, self.ITEMS)
        committed = self._drive(agg)
        assert committed == ["done", "fresh", "slow"]    # all delivered (drained)
        assert agg._vclock.now == 30.0
        assert not agg._sim_buffer.has("slow")

    def test_carryover_releases_once_clock_passes_sct(self):
        # A later round starts past the straggler's sct -> it is delivered, not held.
        agg = self._make_oort_agg(carryover=True, round_num=4, vclock_start=35.0)
        self._load(agg, [("slow", 30.0, 1)])
        committed = self._drive(agg)
        assert committed == ["slow"]                     # released & committed (stale)
        assert not agg._sim_buffer.has("slow")
