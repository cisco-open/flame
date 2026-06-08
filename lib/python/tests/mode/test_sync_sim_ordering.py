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

    def test_fewer_responders_than_k(self, monkeypatch):
        # Only 2 of the 5 selected ever respond. Real mode's recv_fifo(first_k=3)
        # would block forever; sim waits to the deadline then commits what it has
        # (the 2 smallest by sct). Shrink the deadline so the test is fast.
        import flame.mode.horizontal.syncfl.top_aggregator as agg_mod
        monkeypatch.setattr(agg_mod, "SYNC_SIM_RECV_DEADLINE_S", 0.3)
        agg = _make_agg()
        ch = FakeSyncChannel(SCTS, arrival_order=["t1", "t2"])  # only 2 queued
        committed = _committed_ends(agg, ch, first_k=3)
        assert committed == ["t2", "t1"]  # the 2 smallest sct, ascending
        assert agg._vclock.now == 10.0
