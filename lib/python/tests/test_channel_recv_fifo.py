# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the streaming ``Channel.recv_fifo`` receive path.

These guard against the active-task leak that stalled streaming async runs (and
would equally stall simulator mode, which probes ``recv_fifo`` per-end with a
short timeout).

The bug: ``Channel._streamer_for_recv_fifo`` tracks one "active task" per end
while awaiting that end's message. The streamer is fire-and-forget and outlives
the ``recv_fifo`` caller. An end whose trainer never sends (slow / dropped /
unavailable) used to block on ``End.get()`` with no timeout, leaving the end
permanently in ``_active_recv_fifo_tasks``. That end was then skipped
("already has active task") on every future receive, so its updates were never
consumed -> the aggregator stalled with a monotonically growing
``active_tasks`` count.

Two layers are covered:

* ``TestStreamerCleanup`` drives ``_streamer_for_recv_fifo`` directly (fast,
  in-loop) to assert the active set always drains.
* ``TestRecvFifoSimPattern`` drives the *real* public ``recv_fifo`` generator
  over a background event loop, mirroring exactly how simulator mode
  (``TopAggregator._sim_recv_min``) probes one end at a time with a small
  timeout. It also proves a late message is not lost — the property simulator
  mode relies on.
"""

import asyncio
import threading
import time

import cloudpickle

from flame.channel import Channel
from flame.end import KEY_END_STATE
from flame.mode.message import MessageType


class FakeMetricCollector:
    def accumulate(self, *args, **kwargs):
        pass


class FakeEnd:
    """Minimal End stand-in backed by an asyncio.Queue.

    ``get()`` blocks until a payload is available, mirroring the real End so a
    quiet end (one that never gets a payload) blocks forever absent a timeout.
    """

    def __init__(self):
        self._rxq = None
        self.properties = {}

    @property
    def rxq(self):
        if self._rxq is None:
            self._rxq = asyncio.Queue()
        return self._rxq

    async def get(self):
        return await self.rxq.get()

    def qsize(self):
        return self.rxq.qsize()

    def set_property(self, key, value):
        self.properties[key] = value

    def get_property(self, key):
        return self.properties.get(key)


# --------------------------------------------------------------------------- #
# Layer 1: direct _streamer_for_recv_fifo cleanup (fast, single asyncio.run)
# --------------------------------------------------------------------------- #
class TestStreamerCleanup:
    @staticmethod
    def _make_bare_channel(end_ids):
        ch = Channel.__new__(Channel)
        ch._name = "test"
        ch._ends = {eid: FakeEnd() for eid in end_ids}
        ch._active_recv_fifo_tasks = set()
        ch._rx_queue = None  # created inside the scenario loop
        ch.mc = FakeMetricCollector()
        return ch

    def test_quiet_end_is_released_on_timeout(self):
        async def scenario():
            ch = self._make_bare_channel(["quiet"])
            ch._rx_queue = asyncio.Queue()
            await ch._streamer_for_recv_fifo(["quiet"], timeout=0.1)
            return ch

        ch = asyncio.run(scenario())
        assert ch._active_recv_fifo_tasks == set(), "active task leaked for quiet end"
        assert ch._rx_queue.empty(), "timed-out end must not enqueue a non-message"

    def test_delivered_message_flows_and_releases(self):
        async def scenario():
            ch = self._make_bare_channel(["live"])
            ch._rx_queue = asyncio.Queue()
            ch._ends["live"].rxq.put_nowait((b"payload", "ts"))
            await ch._streamer_for_recv_fifo(["live"], timeout=1.0)
            return ch

        ch = asyncio.run(scenario())
        assert ch._active_recv_fifo_tasks == set()
        end_id, payload = ch._rx_queue.get_nowait()
        assert end_id == "live"
        assert payload == (b"payload", "ts")

    def test_mixed_live_and_quiet_ends_all_released(self):
        async def scenario():
            end_ids = ["live", "quiet1", "quiet2"]
            ch = self._make_bare_channel(end_ids)
            ch._rx_queue = asyncio.Queue()
            ch._ends["live"].rxq.put_nowait((b"data", "ts"))
            await ch._streamer_for_recv_fifo(end_ids, timeout=0.1)
            return ch

        ch = asyncio.run(scenario())
        assert ch._active_recv_fifo_tasks == set(), "active tasks leaked"
        delivered = []
        while not ch._rx_queue.empty():
            delivered.append(ch._rx_queue.get_nowait())
        assert len(delivered) == 1
        assert delivered[0][0] == "live"

    def test_repeated_rounds_do_not_accumulate_active_tasks(self):
        async def scenario():
            end_ids = [f"end{i}" for i in range(5)]
            ch = self._make_bare_channel(end_ids)
            ch._rx_queue = asyncio.Queue()
            for _ in range(10):
                await ch._streamer_for_recv_fifo(end_ids, timeout=0.05)
                assert len(ch._active_recv_fifo_tasks) == 0
            return ch

        ch = asyncio.run(scenario())
        assert ch._active_recv_fifo_tasks == set()

    def test_legacy_blocking_mode_still_delivers(self):
        """timeout=None must preserve legacy blocking behavior for sync callers:
        the end is awaited until it delivers, then released."""

        async def scenario():
            ch = self._make_bare_channel(["live"])
            ch._rx_queue = asyncio.Queue()
            ch._ends["live"].rxq.put_nowait((b"x", "ts"))
            await ch._streamer_for_recv_fifo(["live"], timeout=None)
            return ch

        ch = asyncio.run(scenario())
        assert ch._active_recv_fifo_tasks == set()
        assert ch._rx_queue.get_nowait()[0] == "live"


# --------------------------------------------------------------------------- #
# Layer 2: full recv_fifo over a real background loop (simulator probe pattern)
# --------------------------------------------------------------------------- #
class FakeBackend:
    def __init__(self, loop):
        self._loop = loop

    def loop(self):
        return self._loop

    def set_cleanup_ready(self, end_id):
        pass


class TestRecvFifoSimPattern:
    """Exercise the real ``recv_fifo`` generator exactly as ``_sim_recv_min``
    does: ``next(channel.recv_fifo([e], 1, timeout=...))``."""

    @staticmethod
    def _start_loop():
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        return loop, thread

    @staticmethod
    def _stop_loop(loop, thread):
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()

    @staticmethod
    def _make_channel(loop, end_ids):
        ch = Channel.__new__(Channel)
        ch._name = "test"
        ch._backend = FakeBackend(loop)
        ch._ends = {eid: FakeEnd() for eid in end_ids}
        ch._active_recv_fifo_tasks = set()
        # rx_queue is awaited on the loop thread; create it there to bind it.
        fut = asyncio.run_coroutine_threadsafe(_make_queue(), loop)
        ch._rx_queue = fut.result(timeout=2)
        ch.mc = FakeMetricCollector()
        return ch

    @staticmethod
    def _inject(loop, end, msg_dict, ts="t"):
        """Deliver a message to an end's rx queue from the loop thread."""
        payload = (cloudpickle.dumps(msg_dict), ts)

        def _put():
            end.rxq.put_nowait(payload)

        loop.call_soon_threadsafe(_put)

    @staticmethod
    def _probe(ch, end_id, timeout):
        return next(ch.recv_fifo([end_id], 1, timeout=timeout))

    @staticmethod
    def _wait_drained(ch, deadline_s=2.0):
        end = time.time() + deadline_s
        while time.time() < end and ch._active_recv_fifo_tasks:
            time.sleep(0.01)

    def test_quiet_probe_returns_none_and_releases(self):
        loop, thread = self._start_loop()
        try:
            ch = self._make_channel(loop, ["quiet"])
            msg, (end, _) = self._probe(ch, "quiet", timeout=0.2)
            assert msg is None
            self._wait_drained(ch)
            assert ch._active_recv_fifo_tasks == set(), "active task leaked in sim probe"
        finally:
            self._stop_loop(loop, thread)

    def test_live_probe_delivers_and_releases(self):
        loop, thread = self._start_loop()
        try:
            ch = self._make_channel(loop, ["live"])
            self._inject(loop, ch._ends["live"], {MessageType.MODEL_VERSION: 7})
            msg, (end, _) = self._probe(ch, "live", timeout=1.0)
            assert end == "live"
            assert msg[MessageType.MODEL_VERSION] == 7
            self._wait_drained(ch)
            assert ch._active_recv_fifo_tasks == set()
            assert ch._ends["live"].get_property(KEY_END_STATE) is not None
        finally:
            self._stop_loop(loop, thread)

    def test_late_message_after_timeout_is_not_lost(self):
        """The simulator-critical property: an end that misses its probe window
        keeps its message buffered, and a later probe of that end delivers it.
        (Validates that wait_for cancellation does not drop a queued payload.)"""
        loop, thread = self._start_loop()
        try:
            ch = self._make_channel(loop, ["e"])

            # First probe: end is quiet -> times out, returns None, releases.
            msg, _ = self._probe(ch, "e", timeout=0.2)
            assert msg is None
            self._wait_drained(ch)
            assert ch._active_recv_fifo_tasks == set()

            # Message arrives late (after the probe gave up).
            self._inject(loop, ch._ends["e"], {MessageType.MODEL_VERSION: 3})
            time.sleep(0.05)

            # Re-probe: the buffered message must be delivered, not lost.
            msg2, (end, _) = self._probe(ch, "e", timeout=1.0)
            assert end == "e"
            assert msg2[MessageType.MODEL_VERSION] == 3
            self._wait_drained(ch)
            assert ch._active_recv_fifo_tasks == set()
        finally:
            self._stop_loop(loop, thread)

    def test_repeated_quiet_probes_do_not_accumulate(self):
        """Many rounds of probing quiet ends must not grow the active set —
        this is the exact stuck-run signature."""
        loop, thread = self._start_loop()
        try:
            ends = [f"e{i}" for i in range(4)]
            ch = self._make_channel(loop, ends)
            for _ in range(6):
                for e in ends:
                    msg, _ = self._probe(ch, e, timeout=0.05)
                    assert msg is None
            self._wait_drained(ch)
            assert ch._active_recv_fifo_tasks == set()
        finally:
            self._stop_loop(loop, thread)


async def _make_queue():
    return asyncio.Queue()
