# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""fwdllm_aggregator.py's _aggregate_grads_async (fluxtune/async) and
sync_collect_and_accumulate_grads (fwdllm/fwdllm_plus/sync) both called
channel.recv_fifo() with no timeout -- which defaults to blocking forever
(see channel.recv_fifo's docstring). If a selected/in-flight trainer went
quiet, the call never returned, the composer loop never cycled back to
_distribute_weights, and _check_early_stop_conditions() (which enforces
--max-runtime-s/--max-data-id) never got a chance to run -- confirmed via a
real fluxtune run (run_20260701_225428_fluxtune_n10_smoke) that hung 11+
minutes past its 600s budget until manually killed. This covers the fix:
both call sites now pass timeout=RECV_TIMEOUT_WAIT_S, matching the pattern
asyncfl/top_aggregator.py already uses for the identical reason.
"""

from flame.mode.horizontal.asyncfl.top_aggregator import RECV_TIMEOUT_WAIT_S
from flame.mode.horizontal.syncfl.fwdllm_aggregator import TopAggregator


class _FakeChannel:
    """Records every recv_fifo() call's args; ends() returns a fixed pool.
    recv_fifo always yields a single (None, ...) -- callers under test only
    need to observe the call args and return cleanly, not process a real
    message."""

    def __init__(self, ends_result):
        self._ends_result = ends_result
        self.recv_fifo_calls = []

    def ends(self, state=None):
        return self._ends_result

    def recv_fifo(self, end_ids, first_k, timeout=None):
        self.recv_fifo_calls.append(
            {"end_ids": end_ids, "first_k": first_k, "timeout": timeout}
        )
        yield None, ("", None)


class _FakeChannelManager:
    def __init__(self, channel):
        self._channel = channel

    def get_by_tag(self, tag):
        return self._channel


class TestAggregateGradsAsyncBoundedRecv:
    """fluxtune's path (is_async=True)."""

    class _FakeAggregator:
        _aggregate_grads_async = TopAggregator._aggregate_grads_async

        def __init__(self, channel):
            self.cm = _FakeChannelManager(channel)

    def test_recv_fifo_called_with_recv_timeout_wait_s(self):
        channel = _FakeChannel(ends_result=["t1", "t2"])
        agg = self._FakeAggregator(channel)

        agg._aggregate_grads_async("aggregate")

        assert len(channel.recv_fifo_calls) == 1
        assert channel.recv_fifo_calls[0]["timeout"] == RECV_TIMEOUT_WAIT_S
        assert channel.recv_fifo_calls[0]["first_k"] == 1


class TestSyncCollectBoundedRecv:
    """fwdllm/fwdllm_plus's path (is_async=False)."""

    class _FakeAggregator:
        sync_collect_and_accumulate_grads = (
            TopAggregator.sync_collect_and_accumulate_grads
        )

        def __init__(self, channel, agg_goal=3):
            self._agg_goal = agg_goal
            self.ends_not_selected_yet = False
            self._round = 1
            self.data_id = 0
            self.iteration_per_data_id = 0

    def test_recv_fifo_called_with_recv_timeout_wait_s(self):
        channel = _FakeChannel(ends_result=["t1", "t2", "t3"])
        agg = self._FakeAggregator(channel)

        agg.sync_collect_and_accumulate_grads("aggregate", channel)

        assert len(channel.recv_fifo_calls) == 1
        assert channel.recv_fifo_calls[0]["timeout"] == RECV_TIMEOUT_WAIT_S
