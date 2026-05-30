# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Aggregator-level test for simulated-mode receive ordering.

Exercises the real ``TopAggregator._sim_recv_min`` against a fake channel to
prove that, regardless of the physical arrival order, the async aggregator
commits in-flight updates in ascending virtual-completion order and advances
its virtual clock accordingly.
"""

import itertools

import pytest

from flame.mode.horizontal.asyncfl.top_aggregator import TopAggregator
from flame.mode.message import MessageType
from flame.sim import SimReorderBuffer, VirtualClock


class FakeChannel:
    """Minimal channel: queued (end -> msg) delivered in a fixed arrival order.

    Uses a list-based queue that is consumed eagerly (not via a generator)
    so that calling ``next(recv_fifo(...))`` removes the message immediately.
    Mirrors the one-message-per-call pattern _sim_recv_min uses.
    """

    def __init__(self, inflight, arrival_order):
        self._inflight = set(inflight)
        self._queue = list(arrival_order)  # list of (end_id, sct)

    def has(self, end_id):
        return end_id in self._inflight

    def ends(self, state=None):
        return list(self._inflight)

    def recv_fifo(self, end_ids, first_k=0, timeout=None):
        # Find and remove the first queued message whose end is in end_ids,
        # then yield it. The eager pop means next() leaves the queue consistent.
        end_ids = set(end_ids)
        for i, (end_id, sct) in enumerate(self._queue):
            if end_id in end_ids:
                self._queue.pop(i)
                yield (
                    {MessageType.WEIGHTS: f"w_{end_id}",
                     MessageType.SIM_COMPLETION_TS: sct},
                    (end_id, None),
                )
                return
        # Nothing found: yield nothing (caller gets StopIteration on next())
        return
        yield  # make this a generator


class _ConcreteAgg(TopAggregator):
    """Concrete stub so we can instantiate without the abstract methods."""

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
    """A TopAggregator with only the state _sim_recv_min needs."""
    agg = _ConcreteAgg.__new__(_ConcreteAgg)
    agg._vclock = VirtualClock()
    agg._sim_buffer = SimReorderBuffer()
    agg._sim_committed = set()
    return agg


def _drain(agg, channel):
    """Repeatedly call _sim_recv_min until all messages committed.

    Each call: fill buffer from all currently-receivable ends (one probe each),
    then pop and commit the minimum. The test provides instantaneous delivery
    (no real timeout needed), so we keep calling until the queue and buffer
    are both empty.
    """
    committed = []
    max_iters = (len(channel.ends()) + 1) * 3
    for _ in range(max_iters):
        recv_ends = [e for e in channel.ends() if channel.has(e)]
        msg, (end, _) = agg._sim_recv_min(channel, recv_ends)
        if msg is not None:
            committed.append((end, msg[MessageType.SIM_COMPLETION_TS]))
        if not channel._queue and not agg._sim_buffer.pending_ends():
            break
    return committed, agg._vclock.now


class TestSimRecvMin:
    def test_commits_in_completion_order(self):
        durations = {"t1": 10.0, "t2": 5.0, "t3": 25.0, "t4": 15.0}
        expected = [("t2", 5.0), ("t1", 10.0), ("t4", 15.0), ("t3", 25.0)]
        agg = _make_agg()
        # arrival order deliberately scrambled vs completion order
        channel = FakeChannel(
            inflight=set(durations),
            arrival_order=[("t3", 25.0), ("t1", 10.0), ("t4", 15.0), ("t2", 5.0)],
        )
        committed, t_v = _drain(agg, channel)
        assert committed == expected
        assert t_v == 25.0  # advanced to the last committed completion

    def test_independent_of_arrival_order(self):
        durations = {"a": 3.0, "b": 1.0, "c": 2.0, "d": 4.0}
        expected = [("b", 1.0), ("c", 2.0), ("a", 3.0), ("d", 4.0)]
        for arrival in itertools.permutations(durations.items()):
            agg = _make_agg()
            channel = FakeChannel(set(durations), list(arrival))
            committed, _ = _drain(agg, channel)
            assert committed == expected

    def test_virtual_clock_monotone(self):
        durations = {"x": 8.0, "y": 2.0, "z": 5.0}
        agg = _make_agg()
        channel = FakeChannel(set(durations), list(durations.items()))
        committed, t_v = _drain(agg, channel)
        times = [c for _, c in committed]
        assert times == sorted(times)  # committed in nondecreasing sim time
        assert t_v == max(times)
