# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Smoke-test stop bar: a run should stop once `self.data_id` reaches
`hyperparameters.max_data_id_progress`, or once `hyperparameters.max_runtime_s`
of wall time has elapsed since the aggregator started -- whichever comes
first. Both caps are independent of `rounds`/`total_data_bins` and are
checked from `_distribute_weights`, which runs on every composer tick on
both the sync and async paths -- unlike the rounds-based stop in
`_process_aggregation_goal_met`, this also fires when an aggregation goal is
never met.

Also covers `_async_inner_loop_done`: the async/hybrid compose path's inner
`asyncfl_loop` must also exit on `_work_done`, not just an agg-goal match --
see `TestAsyncInnerLoopExitsOnWorkDone` below."""

import time
from types import SimpleNamespace

from flame.mode.horizontal.syncfl.fwdllm_aggregator import TopAggregator


class _FakeAggregator:
    """Minimal stand-in exposing only the state
    `_check_early_stop_conditions`/`_distribute_weights` touch."""

    def __init__(
        self,
        data_id=0,
        max_data_id_progress=None,
        max_runtime_s=None,
        agg_start_time_ts=None,
        is_async=False,
    ):
        self._work_done = False
        self.data_id = data_id
        self.is_async = is_async
        self.agg_start_time_ts = (
            agg_start_time_ts if agg_start_time_ts is not None else time.time()
        )
        self.config = SimpleNamespace(
            hyperparameters=SimpleNamespace(
                max_data_id_progress=max_data_id_progress,
                max_runtime_s=max_runtime_s,
            )
        )

    def _distribute_weights_sync(self, tag, task_to_perform="train"):
        pass

    def _distribute_weights_async(self, tag, task_to_perform="train"):
        pass

    check = TopAggregator._check_early_stop_conditions
    _check_early_stop_conditions = TopAggregator._check_early_stop_conditions
    distribute = TopAggregator._distribute_weights


class TestEarlyStopConditions:
    def test_unset_caps_never_stop(self):
        agg = _FakeAggregator(data_id=999)
        agg.check()
        assert agg._work_done is False

    def test_max_data_id_progress_stops_once_reached(self):
        agg = _FakeAggregator(data_id=10, max_data_id_progress=10)
        agg.check()
        assert agg._work_done is True

    def test_max_data_id_progress_not_yet_reached(self):
        agg = _FakeAggregator(data_id=9, max_data_id_progress=10)
        agg.check()
        assert agg._work_done is False

    def test_max_runtime_s_stops_once_elapsed(self):
        agg = _FakeAggregator(
            max_runtime_s=1.0, agg_start_time_ts=time.time() - 2.0
        )
        agg.check()
        assert agg._work_done is True

    def test_max_runtime_s_not_yet_elapsed(self):
        agg = _FakeAggregator(max_runtime_s=600.0, agg_start_time_ts=time.time())
        agg.check()
        assert agg._work_done is False

    def test_whichever_fires_first_data_id_before_runtime(self):
        """data_id cap reached well before the runtime cap -- data_id should
        be the one that trips, and it should win even though both are set."""
        agg = _FakeAggregator(
            data_id=10,
            max_data_id_progress=10,
            max_runtime_s=600.0,
            agg_start_time_ts=time.time(),
        )
        agg.check()
        assert agg._work_done is True

    def test_whichever_fires_first_runtime_before_data_id(self):
        agg = _FakeAggregator(
            data_id=1,
            max_data_id_progress=10,
            max_runtime_s=1.0,
            agg_start_time_ts=time.time() - 2.0,
        )
        agg.check()
        assert agg._work_done is True

    def test_already_work_done_is_a_noop(self):
        """Once stopped, the check must not re-derive or overwrite state."""
        agg = _FakeAggregator(data_id=5, max_data_id_progress=10)
        agg._work_done = True
        agg.check()
        assert agg._work_done is True

    def test_distribute_weights_checks_stop_conditions_on_sync_path(self):
        agg = _FakeAggregator(data_id=10, max_data_id_progress=10, is_async=False)
        agg.distribute("tag", "train")
        assert agg._work_done is True

    def test_distribute_weights_checks_stop_conditions_on_async_path(self):
        agg = _FakeAggregator(data_id=10, max_data_id_progress=10, is_async=True)
        agg.distribute("tag", "train")
        assert agg._work_done is True


class _FakeAsyncLoopAggregator:
    """Minimal stand-in exposing only the state
    `_async_inner_loop_done` touches."""

    def __init__(self, agg_goal_cnt, agg_goal, work_done):
        self._agg_goal_cnt = agg_goal_cnt
        self._agg_goal = agg_goal
        self._work_done = work_done

    inner_loop_done = TopAggregator._async_inner_loop_done


class TestAsyncInnerLoopExitsOnWorkDone:
    """Reproduces a live smoke-test finding: `_check_early_stop_conditions()`
    can set `_work_done`, but the process keeps running until the launcher's
    external watchdog force-kills it, because the async/hybrid compose
    path's inner `asyncfl_loop` only checked `_agg_goal_cnt == _agg_goal`,
    never `_work_done` -- so it kept spinning, and the outer loop (which
    does check `_work_done`) never got a chance to observe it."""

    def test_exits_on_work_done_even_if_agg_goal_not_met(self):
        agg = _FakeAsyncLoopAggregator(agg_goal_cnt=0, agg_goal=3, work_done=True)
        assert agg.inner_loop_done() is True

    def test_exits_on_agg_goal_met_even_if_work_not_done(self):
        agg = _FakeAsyncLoopAggregator(agg_goal_cnt=3, agg_goal=3, work_done=False)
        assert agg.inner_loop_done() is True

    def test_does_not_exit_while_neither_condition_holds(self):
        agg = _FakeAsyncLoopAggregator(agg_goal_cnt=1, agg_goal=3, work_done=False)
        assert agg.inner_loop_done() is False
