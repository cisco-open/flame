# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""fwdllm's TopAggregator extends the asyncfl base (not syncfl's), which
has no rounds-based stop condition of its own. self._work_done was never
set anywhere in fwdllm_aggregator.py, so the composer loop
(Loop(loop_check_fn=lambda: self._work_done)) never exited regardless of
hyperparameters.rounds -- every fwdllm/fwdllm_plus/fluxtune run (sync
path) ran forever until manually killed or the launcher's watchdog
force-terminated it. _process_aggregation_goal_met must set
self._work_done once self._round exceeds hyperparameters.rounds, on the
same round-rollover branch that already advances self._round."""

from types import SimpleNamespace
from unittest.mock import patch

from flame.mode.horizontal.syncfl.fwdllm_aggregator import TopAggregator


class _FakeChannel:
    def __init__(self):
        self.properties = {}

    def get_end_property(self, end, prop):
        return None

    def set_property(self, name, value):
        self.properties[name] = value

    def cleanup_recvd_ends(self):
        pass


class _FakeAggregator:
    """Minimal stand-in exposing only the state
    `_process_aggregation_goal_met` touches, with the heavy model/aggregate
    machinery stubbed out."""

    def __init__(self, round_, rounds, total_data_bins=150, var_good_enough=True):
        self._agg_goal = 2
        self._agg_goal_cnt = 2
        self._per_agg_trainer_list = []
        self._model_version_unique_trainers = set()
        self._model_version_trainer_stats = {
            "train_duration": [],
            "partial_stat_utility": [],
        }
        self.grad_pool = []
        self.grad = []
        self.model = None
        self._max_iter_per_data_id = None
        self.iteration_per_data_id = 0
        self.var = 0.0
        self.var_threshold = 0.3
        self.var_good_enough = var_good_enough
        self.data_id = total_data_bins - 1
        self.total_data_bins = total_data_bins
        self._is_model_updated = False
        self._model_version = 0
        self._round = round_
        self._updates_in_queue = 2
        self._updates_received = {}
        self._n_aggs_completed = 0
        self._var_total_count = 0
        self._var_pass_count = 0
        self._dynamic_kc_controller = None
        self.config = SimpleNamespace(
            hyperparameters=SimpleNamespace(
                inc_model_version_per_data_id=True, rounds=rounds
            )
        )

    def add_local_trained_result(self, *args, **kwargs):
        pass

    def aggregate(self, round_id):
        pass

    def eval_model(self):
        return {"eval_loss": 0.0}, None, None

    def _log_and_reset_model_version_stats(self):
        pass

    process = TopAggregator._process_aggregation_goal_met


@patch(
    "flame.mode.horizontal.syncfl.fwdllm_aggregator.fc.make_functional_with_buffers",
    return_value=(None, [], None),
)
class TestRoundsBasedTermination:
    def test_work_done_set_once_rounds_exhausted(self, _mock_ffb):
        agg = _FakeAggregator(round_=50, rounds=50)
        channel = _FakeChannel()

        agg.process("tag", channel)

        assert agg._round == 51
        assert agg._work_done is True

    def test_work_done_not_set_while_rounds_remain(self, _mock_ffb):
        agg = _FakeAggregator(round_=10, rounds=50)
        channel = _FakeChannel()

        agg.process("tag", channel)

        assert agg._round == 11
        assert agg._work_done is False

    def test_work_done_not_touched_when_round_not_complete(self, _mock_ffb):
        """data_id hasn't reached total_data_bins yet -- the round-rollover
        branch (and therefore the rounds check) must not run at all."""
        agg = _FakeAggregator(round_=49, rounds=50, total_data_bins=150)
        agg.data_id = 5  # not the last data bin
        channel = _FakeChannel()

        agg.process("tag", channel)

        assert agg._round == 49
        assert not hasattr(agg, "_work_done")

    def test_work_done_not_touched_when_variance_check_fails(self, _mock_ffb):
        """var_good_enough=False takes the retry branch, which never
        reaches the round-rollover code at all."""
        agg = _FakeAggregator(round_=49, rounds=50, var_good_enough=False)
        channel = _FakeChannel()

        agg.process("tag", channel)

        assert agg._round == 49
        assert not hasattr(agg, "_work_done")
