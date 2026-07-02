# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""An end must contribute at most once per aggregation cycle in the generic
asyncfl TopAggregator (flame/mode/horizontal/asyncfl/top_aggregator.py).

Unlike fwdllm_aggregator.TopAggregator (which has a dedicated
_per_agg_trainer_list guard, see test_fwdllm_duplicate_contribution.py), the
generic asyncfl aggregator had no equivalent check. That was safe only as
long as the selector's SEND_TIMEOUT_WAIT_S reclaim was itself broken (see
../../examples/MIGRATING_TO_LAUNCHER.md's aggregator gotchas): a stuck trainer's slot never actually
reopened, so it could never be reselected while its original response was
still in flight. Fixing that reclaim (async_oort.py/async_random.py/
fedbuff.py) makes a genuine duplicate-contribution newly reachable, so this
guard (_agg_cycle_contributed_ends) closes the same gap here.
"""

from types import SimpleNamespace

import torch

from flame.mode.horizontal.asyncfl.top_aggregator import TopAggregator
from flame.mode.message import MessageType


class _FakeSelector:
    def __init__(self):
        self.ordered_updates_recv_ends = []


class _FakeChannel:
    """Yields a canned sequence of (msg, (end_id, timestamp)) pairs, one per
    recv_fifo call, mirroring the real one-message-per-call contract."""

    def __init__(self, messages):
        self._messages = list(messages)
        self._end_props = {}
        self.cleaned_up = []
        self._selector = _FakeSelector()

    def ends(self, state=None):
        return ["t1"]

    def has(self, end_id):
        return True

    def recv_fifo(self, recv_ends, first_k=0, timeout=None):
        yield self._messages.pop(0)

    def set_end_property(self, end_id, key, value):
        self._end_props[(end_id, key)] = value

    def get_end_property(self, end_id, key):
        return self._end_props.get((end_id, key))

    def cleanup_provided_ends(self, end_ids):
        self.cleaned_up.append(end_ids)


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
    """A TopAggregator with only the state _aggregate_weights touches,
    configured so agg_goal is never reached (avoids needing to stub the
    optimizer.scale_add_agg_weights/_update_model finalization path)."""
    agg = _ConcreteAgg.__new__(_ConcreteAgg)
    agg.simulated = False
    agg._agg_goal = 5  # high enough that 2 messages never trip finalization
    agg._agg_goal_cnt = 0
    agg._agg_goal_weights = None
    agg._agg_cycle_contributed_ends = set()
    agg._updates_in_queue = 0
    agg._updates_recevied = {}
    agg._per_round_update_list = []
    agg._per_trainer_staleness_track = {}
    agg._round_update_values = {"staleness": [], "stat_utility": [], "trainer_speed": []}
    agg._round = 1
    agg.reject_stale_updates = "False"
    agg._track_trainer_version_duration_s = {}
    agg.cache = {}
    agg.model = torch.nn.Linear(1, 1)
    agg.optimizer = SimpleNamespace(do=lambda agg_w, cache, total, version, staleness_factor: agg_w)
    return agg


def _msg_for(end_id, version=1):
    return (
        {
            MessageType.WEIGHTS: {"weight": torch.zeros(1, 1)},
            MessageType.MODEL_VERSION: version,
            MessageType.DATASET_SIZE: 10,
        },
        (end_id, 0.0),
    )


class TestAsyncflDuplicateContributionGuard:
    def test_second_contribution_from_same_end_same_cycle_is_ignored(self):
        agg = _make_agg()
        channel = _FakeChannel([_msg_for("t1"), _msg_for("t1")])
        agg.cm = SimpleNamespace(get_by_tag=lambda tag: channel)

        agg._aggregate_weights("param-channel")
        assert "t1" in agg.cache
        assert agg._agg_goal_cnt == 1
        assert channel.cleaned_up == []

        # Second message from the SAME end within the same (unreset) cycle
        # must be rejected, not double-counted.
        agg._aggregate_weights("param-channel")
        assert agg._agg_goal_cnt == 1
        assert channel.cleaned_up == ["t1"]

    def test_different_ends_both_counted(self):
        agg = _make_agg()
        channel = _FakeChannel([_msg_for("t1"), _msg_for("t2")])
        agg.cm = SimpleNamespace(get_by_tag=lambda tag: channel)

        agg._aggregate_weights("param-channel")
        agg._aggregate_weights("param-channel")

        assert agg._agg_goal_cnt == 2
        assert channel.cleaned_up == []

    def test_reset_agg_goal_variables_clears_guard_for_next_cycle(self):
        agg = _make_agg()
        agg._agg_cycle_contributed_ends = {"t1"}
        agg._reset_agg_goal_variables()
        assert agg._agg_cycle_contributed_ends == set()
