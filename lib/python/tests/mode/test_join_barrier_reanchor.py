# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Real-mode agg_start_time_ts re-anchor at the join barrier.

`agg_start_time_ts` is stamped at aggregator __init__, before the trainer
cohort joins. In sim mode this is harmless (_avail_now() reads _vclock.now,
which only starts advancing at round-0 selection). In real mode _avail_now()
reads time.time() - agg_start_time_ts, so without a re-anchor the join wait
(real OS process spawn/connect — ~300s wall at n=300) is silently baked into
every subsequent availability-trace read: real ends up reading the trace
~300s ahead of where sim/ground-truth says it should be (root-caused via
feddance syn_50 A3 — see UNAVAILABILITY_DESIGN.md).

`_mark_join_barrier_done` fixes this by re-anchoring agg_start_time_ts to
"now" (real mode only) at the moment the join barrier resolves, whatever
that wait actually took — self-correcting if the join wait duration changes.
"""

import time
import types

from flame.mode.horizontal.syncfl.top_aggregator import TopAggregator


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


def _make_agg(*, simulated: bool, min_trainers_to_start=None, n_joined=0):
    agg = _ConcreteAgg.__new__(_ConcreteAgg)
    agg.simulated = simulated
    agg.config = types.SimpleNamespace(
        hyperparameters=types.SimpleNamespace(
            min_trainers_to_start=min_trainers_to_start,
            min_trainers_join_timeout_s=5.0,
        )
    )
    agg.agg_start_time_ts = time.time()
    channel = types.SimpleNamespace(_ends={f"t{i}": None for i in range(n_joined)})
    return agg, channel


class TestMarkJoinBarrierDone:
    def test_real_mode_reanchors_agg_start(self):
        agg, _ = _make_agg(simulated=False)
        original = agg.agg_start_time_ts
        time.sleep(0.02)
        agg._mark_join_barrier_done()
        assert agg._join_barrier_done is True
        assert agg.agg_start_time_ts > original

    def test_sim_mode_leaves_agg_start_untouched(self):
        agg, _ = _make_agg(simulated=True)
        original = agg.agg_start_time_ts
        time.sleep(0.02)
        agg._mark_join_barrier_done()
        assert agg._join_barrier_done is True
        assert agg.agg_start_time_ts == original


class TestAwaitMinTrainersReanchors:
    def test_real_mode_barrier_disabled_still_reanchors(self):
        # min_trainers_to_start unset → barrier is a no-op wait, but the join
        # (channel.await_join(), called by the caller before this) still cost
        # real wall-clock time, so agg_start_time_ts must still move.
        agg, channel = _make_agg(simulated=False, min_trainers_to_start=None)
        original = agg.agg_start_time_ts
        time.sleep(0.02)
        agg._await_min_trainers(channel)
        assert agg.agg_start_time_ts > original

    def test_real_mode_barrier_satisfied_reanchors_after_wait(self):
        agg, channel = _make_agg(
            simulated=False, min_trainers_to_start=3, n_joined=3
        )
        original = agg.agg_start_time_ts
        time.sleep(0.02)
        agg._await_min_trainers(channel)
        assert agg._join_barrier_done is True
        assert agg.agg_start_time_ts > original

    def test_sim_mode_never_reanchors(self):
        agg, channel = _make_agg(
            simulated=True, min_trainers_to_start=3, n_joined=3
        )
        original = agg.agg_start_time_ts
        time.sleep(0.02)
        agg._await_min_trainers(channel)
        assert agg.agg_start_time_ts == original

    def test_one_shot_does_not_reanchor_twice(self):
        agg, channel = _make_agg(
            simulated=False, min_trainers_to_start=3, n_joined=3
        )
        agg._await_min_trainers(channel)
        anchored = agg.agg_start_time_ts
        time.sleep(0.02)
        agg._await_min_trainers(channel)  # second call: already done, no-op
        assert agg.agg_start_time_ts == anchored
