# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""sct-ordered DIRECT drain ingestion (simSctOrderedDrain) for the async stack.

The async sim clock blew up (staleness ~15 vs real ~3, only ~1.6 of 10
commits/round advancing the clock) because in-flight updates were ingested
through the recv_fifo streamer, whose background task + shared queue could
strand a *delivered* update out of the reorder buffer's view — the clock then
lapped the stranded lower-sct update and committed it past-dated. The fix
ingests by draining each in-flight end's rx queue DIRECTLY
(``channel.drain_ready``), so the buffer is always a COMPLETE snapshot of
arrived in-flight updates and the existing min-sct gate commits in true
completion order (commit_gap ≈ 0).

These tests pin two things:
  1. the ``End`` readiness primitives that make a directly-drained message
     impossible to miss (``get_ready_nowait`` / peek-aware ``is_rxq_empty``);
  2. that ``_sim_recv_min`` with the flag ON commits strictly in sct order with
     zero past-dating regardless of physical arrival order, and produces the
     SAME commit sequence as the (already-correct on a reliable channel) flag-off
     path — i.e. the change fixes ingestion reliability without altering the
     committed semantics.
"""

import asyncio
import itertools
import threading

import cloudpickle
import pytest

from flame.channel import Channel
from flame.end import End, KEY_END_STATE, VAL_END_STATE_RECVD
from flame.mode.horizontal.asyncfl.top_aggregator import TopAggregator
from flame.mode.message import MessageType
from flame.sim import SimReorderBuffer, VirtualClock


# --------------------------------------------------------------------------- #
# 1. End readiness primitives (the anti-stranding guard)
# --------------------------------------------------------------------------- #
class TestEndReadyPrimitives:
    def test_get_ready_nowait_drains_fifo_then_none(self):
        e = End("x")
        e.rxq.put_nowait(b"a")
        e.rxq.put_nowait(b"b")
        assert e.get_ready_nowait() == b"a"
        assert e.get_ready_nowait() == b"b"
        assert e.get_ready_nowait() is None  # empty → None, never blocks

    def test_get_ready_nowait_returns_peek_buf_first(self):
        # A prior peek() parks the message in peek_buf and leaves rxq empty; the
        # ready-drain must still surface it (else a peeked message is stranded).
        e = End("x")
        e.peek_buf = b"peeked"
        e.rxq.put_nowait(b"queued")
        assert e.get_ready_nowait() == b"peeked"
        assert e.peek_buf is None
        assert e.get_ready_nowait() == b"queued"

    def test_is_rxq_empty_honors_peek_buf(self):
        e = End("x")
        assert e.is_rxq_empty() is True
        e.peek_buf = b"peeked"          # message pending in peek_buf only
        assert e.is_rxq_empty() is False  # must NOT report "nothing ready"
        e.peek_buf = None
        e.rxq.put_nowait(b"q")
        assert e.is_rxq_empty() is False


# --------------------------------------------------------------------------- #
# 1b. The REAL channel.drain_ready over real End queues + a threaded backend
# loop — exercises the loop-threading, the pull-on-loop / decode-off-loop split,
# and peek_buf handling that the FakeChannel can't.
# --------------------------------------------------------------------------- #
class _LoopBackend:
    """Minimal backend: a real asyncio loop on a background thread (drain_ready
    schedules its non-blocking pull there via run_async) + cleanup bookkeeping."""

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self.cleaned = []

    def loop(self):
        return self._loop

    def set_cleanup_ready(self, end_id):
        self.cleaned.append(end_id)

    def stop(self):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2)


def _make_channel(end_ids):
    ch = Channel.__new__(Channel)
    ch._name = "t"
    ch._backend = _LoopBackend()
    ch._ends = {e: End(e) for e in end_ids}
    return ch


def _put(ch, end_id, msg):
    # Enqueue a (pickled_payload, timestamp) on the End's rx queue FROM the loop
    # thread (asyncio.Queue is not cross-thread safe), mirroring how the backend
    # delivers a reassembled message.
    payload = (cloudpickle.dumps(msg), None)
    fut = asyncio.run_coroutine_threadsafe(
        _put_coro(ch._ends[end_id], payload), ch._backend.loop()
    )
    fut.result(timeout=2)


async def _put_coro(end, payload):
    end.rxq.put_nowait(payload)


class TestRealChannelDrainReady:
    def test_drains_all_decoded_and_marks_recvd(self):
        ch = _make_channel(["a", "b", "c"])
        try:
            _put(ch, "a", {MessageType.SIM_COMPLETION_TS: 5.0})
            _put(ch, "b", {MessageType.SIM_COMPLETION_TS: 3.0})
            _put(ch, "c", {MessageType.SIM_COMPLETION_TS: 9.0})
            out = ch.drain_ready(["a", "b", "c"], timeout=0.5)
            got = {md[0]: msg[MessageType.SIM_COMPLETION_TS] for msg, md in out}
            assert got == {"a": 5.0, "b": 3.0, "c": 9.0}  # all decoded, none stranded
            for e in ("a", "b", "c"):
                assert ch._ends[e].get_property(KEY_END_STATE) == VAL_END_STATE_RECVD
                assert e in ch._backend.cleaned
            # queues fully drained
            assert ch.drain_ready(["a", "b", "c"], timeout=0) == []
        finally:
            ch._backend.stop()

    def test_surfaces_peek_buffered_message(self):
        # A message parked in peek_buf (rxq empty) must still be drained — the
        # exact stranding the readiness fix targets.
        ch = _make_channel(["a"])
        try:
            ch._ends["a"].peek_buf = cloudpickle.dumps(
                {MessageType.SIM_COMPLETION_TS: 7.0}
            ), None
            out = ch.drain_ready(["a"], timeout=0)
            assert len(out) == 1
            assert out[0][0][MessageType.SIM_COMPLETION_TS] == 7.0
        finally:
            ch._backend.stop()

    def test_timeout_returns_empty_when_nothing_ready(self):
        ch = _make_channel(["a"])
        try:
            assert ch.drain_ready(["a"], timeout=0.05) == []
        finally:
            ch._backend.stop()


# --------------------------------------------------------------------------- #
# Fake channel whose drain_ready models the streamer-free direct drain:
# returns every ready (msg, metadata) for the given ends in arrival order,
# removing them from the queue. (A reliable, complete ingestion — exactly what
# the real channel.drain_ready guarantees and the real recv_fifo did not.)
# --------------------------------------------------------------------------- #
class _FakeEnd:
    def __init__(self, ready_fn=None):
        self._props = {}
        self._ready_fn = ready_fn

    def get_property(self, key):
        return self._props.get(key)

    def set_property(self, key, value):
        self._props[key] = value

    def is_rxq_empty(self):
        return True if self._ready_fn is None else not self._ready_fn()


class FakeChannel:
    def __init__(self, inflight, arrival_order):
        self._inflight = set(inflight)
        self._queue = list(arrival_order)  # list of (end_id, sct)
        self._ends = {
            e: _FakeEnd(ready_fn=(lambda e=e: any(q[0] == e for q in self._queue)))
            for e in self._inflight
        }

    def has(self, end_id):
        return end_id in self._inflight

    def ends(self, state=None):
        return list(self._inflight)

    # recv_fifo kept so the SAME channel can drive the flag-off path for the
    # equivalence test; mirrors the set-wide one-call drain of the real API.
    def recv_fifo(self, end_ids, first_k=0, timeout=None):
        ids = set(end_ids)
        i = 0
        while i < len(self._queue):
            end_id, sct = self._queue[i]
            if end_id in ids:
                self._queue.pop(i)
                yield (
                    {MessageType.WEIGHTS: f"w_{end_id}",
                     MessageType.SIM_COMPLETION_TS: sct,
                     MessageType.TRAINING_BUDGET_S: sct},
                    (end_id, None),
                )
            else:
                i += 1
        yield (None, ("", None))

    def drain_ready(self, end_ids, timeout=None):
        ids = set(end_ids)
        out, rest = [], []
        for end_id, sct in self._queue:
            if end_id in ids:
                self._ends[end_id].set_property(KEY_END_STATE, VAL_END_STATE_RECVD)
                out.append((
                    {MessageType.WEIGHTS: f"w_{end_id}",
                     MessageType.SIM_COMPLETION_TS: sct,
                     MessageType.TRAINING_BUDGET_S: sct},
                    (end_id, None),
                ))
            else:
                rest.append((end_id, sct))
        self._queue = rest
        return out


class _ConcreteAgg(TopAggregator):
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


def _make_agg(sct_ordered_drain):
    agg = _ConcreteAgg.__new__(_ConcreteAgg)
    agg._vclock = VirtualClock()
    agg._sim_buffer = SimReorderBuffer()
    agg._sim_committed = set()
    agg._sim_pending_commit = set()
    agg._sim_inflight_expected = {}
    agg._sim_trainer_budget = {}
    agg._sim_budget_min = 12.0
    agg._sim_budget_running_mean = 12.0
    agg._sim_budget_n = 0
    agg._sim_sct_ordered_drain = sct_ordered_drain
    return agg


def _drain(agg, channel):
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


# --------------------------------------------------------------------------- #
# 2. _sim_recv_min with the drain flag ON
# --------------------------------------------------------------------------- #
class TestSctOrderedDrainCommit:
    SCENARIO = {"t1": 10.0, "t2": 5.0, "t3": 25.0, "t4": 15.0}
    COMPLETION_ORDER = [("t2", 5.0), ("t1", 10.0), ("t4", 15.0), ("t3", 25.0)]

    def test_commits_in_sct_order_any_arrival(self):
        for perm in itertools.permutations(self.SCENARIO.items()):
            agg = _make_agg(sct_ordered_drain=True)
            channel = FakeChannel(set(self.SCENARIO), list(perm))
            committed, t_v = _drain(agg, channel)
            assert committed == self.COMPLETION_ORDER
            assert t_v == 25.0

    def test_no_past_dating_clock_never_exceeds_committed_sct(self):
        # The whole point: every commit lands at its own sct, never lapped — so
        # the per-commit gap (vclock_at_commit - sct) is ~0 for all commits.
        agg = _make_agg(sct_ordered_drain=True)
        channel = FakeChannel(set(self.SCENARIO),
                              [("t3", 25.0), ("t1", 10.0), ("t4", 15.0), ("t2", 5.0)])
        clock_at_commit = []
        for _ in range(len(self.SCENARIO)):
            msg, (end, _) = agg._sim_recv_min(channel, list(channel.ends()))
            clock_at_commit.append((msg[MessageType.SIM_COMPLETION_TS], agg._vclock.now))
        for sct, vclock in clock_at_commit:
            assert vclock == sct  # committed exactly at completion, no past-dating

    def test_flag_on_equals_flag_off_on_reliable_channel(self):
        # On a channel that delivers reliably, the direct-drain path must commit
        # the SAME sequence + clock as the legacy recv_fifo path — the change is
        # ingestion reliability, not committed semantics.
        for perm in itertools.permutations(self.SCENARIO.items()):
            arrival = list(perm)
            off = _make_agg(sct_ordered_drain=False)
            on = _make_agg(sct_ordered_drain=True)
            c_off, tv_off = _drain(off, FakeChannel(set(self.SCENARIO), list(arrival)))
            c_on, tv_on = _drain(on, FakeChannel(set(self.SCENARIO), list(arrival)))
            assert c_on == c_off == self.COMPLETION_ORDER
            assert tv_on == tv_off == 25.0
