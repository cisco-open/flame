# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""fwdllm's aggregator emitted zero telemetry (see
../../examples/MIGRATING_TO_LAUNCHER.md's telemetry gotchas) -- unlike
asyncfl/top_aggregator.py, it never called telemetry.emit() for agg_eval/
agg_round, so plots/performance/ and plots/insights/ were structurally empty
for every fwdllm-family baseline regardless of what the analyzer did.

This covers the fix: _process_aggregation_goal_met now emits agg_eval (right
after eval_model(), only on the variance-check-passed path) and agg_round
(once per completed aggregation cycle, on both the passed and failed paths)
when telemetry is enabled, and stays a true no-op (no emit call at all) when
it isn't.
"""

from datetime import timedelta

import torch

from flame import telemetry
from flame.mode.horizontal.syncfl.fwdllm_aggregator import (
    PROP_ROUND_START_TIME,
    PROP_STAT_UTILITY,
    TopAggregator,
)
from flame.mode.message import MessageType


class _FakeChannel:
    def __init__(self, durations=None, utilities=None):
        self._durations = durations or {}
        self._utilities = utilities or {}
        self.cleaned_up_rounds = []
        self.round_prop = None

    def get_end_property(self, end, key):
        if key == "round_duration":
            return self._durations.get(end)
        if key == "stat_utility":
            return self._utilities.get(end)
        return None

    def set_property(self, key, value):
        if key == "round":
            self.round_prop = value

    def cleanup_recvd_ends(self):
        self.cleaned_up_rounds.append("cleaned")


class _FakeHyperparameters:
    inc_model_version_per_data_id = True
    rounds = 1000


class _FakeConfig:
    hyperparameters = _FakeHyperparameters()


class _FakeAggregator:
    """Binds the real _process_aggregation_goal_met onto a minimal stand-in.

    Only stubs the heavy ML plumbing (aggregate/eval_model/model
    functionalization) that method touches but which is irrelevant to the
    telemetry-emission logic under test -- everything else (round/data_id
    bookkeeping, telemetry field construction) runs for real.
    """

    _process_aggregation_goal_met = TopAggregator._process_aggregation_goal_met

    def __init__(self, contributors, var_good_enough, staleness_map=None,
                 total_data_bins=150):
        self._per_agg_trainer_list = list(contributors)
        self._agg_goal_cnt = len(contributors)
        self._agg_goal = len(contributors) or 1
        self._model_version_unique_trainers = set()
        self._model_version_trainer_stats = {
            "train_duration": [], "partial_stat_utility": [],
        }
        self._trainer_last_model_version = staleness_map or {}
        self._model_version = 5
        self._round = 1
        self.data_id = 3
        self.iteration_per_data_id = 0
        self.total_data_bins = total_data_bins
        self._updates_in_queue = len(contributors)
        self._updates_received = {c: 1 for c in contributors}
        self._max_iter_per_data_id = None
        self._dynamic_kc_controller = None
        self._n_aggs_completed = 0
        self._var_pass_count = 0
        self._var_total_count = 0
        self.var = 0.1
        self.var_good_enough = var_good_enough
        self.var_threshold = 0.3
        self.grad_pool = []
        self.grad = [torch.zeros(1)]
        self.model = torch.nn.Linear(1, 1)
        self.config = _FakeConfig()
        self._eval_result = {"eval_loss": 0.42, "acc": 0.9, "mcc": 0.5}

        # Stubs for the heavy ML machinery this method calls but which this
        # test doesn't need to exercise for real.
        import functorch as fc
        self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(
            self.model
        )

    def aggregate(self, round_num):
        pass  # normally sets self.var/self.var_good_enough as a side effect

    def add_local_trained_result(self, *a, **k):
        pass

    def eval_model(self):
        return dict(self._eval_result), None, []

    def _log_and_reset_model_version_stats(self):
        pass


class TestAggEvalTelemetry:
    def test_emitted_on_variance_pass_with_expected_fields(self, tmp_path):
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _FakeAggregator(contributors=["t1"], var_good_enough=True)
            channel = _FakeChannel(
                durations={"t1": timedelta(seconds=5)},
                utilities={"t1": 1.5},
            )

            agg._process_aggregation_goal_met(tag="aggregate", channel=channel)

            lines = (tmp_path / "aggregator.jsonl").read_text().splitlines()
            import json
            events = [json.loads(l) for l in lines]
            evals = [e for e in events if e["event"] == "agg_eval"]
            assert len(evals) == 1
            assert evals[0]["test-loss"] == 0.42
            assert evals[0]["test-accuracy"] == 0.9
            # data_id snapshot must be the pre-increment value (the data_id
            # that was actually evaluated), not the post-increment one.
            assert evals[0]["data_id"] == 3
            assert evals[0]["round"] == 1
        finally:
            telemetry.shutdown()

    def test_not_emitted_on_variance_fail(self, tmp_path):
        """eval_model() only runs on the variance-check-passed path, so no
        agg_eval event should appear when the check fails."""
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _FakeAggregator(contributors=["t1"], var_good_enough=False)
            channel = _FakeChannel()

            agg._process_aggregation_goal_met(tag="aggregate", channel=channel)

            import json
            lines = (tmp_path / "aggregator.jsonl").read_text().splitlines()
            events = [json.loads(l) for l in lines]
            assert not [e for e in events if e["event"] == "agg_eval"]
        finally:
            telemetry.shutdown()

    def test_noop_when_telemetry_disabled(self, tmp_path):
        """No configure() call -- telemetry stays disabled; must not raise
        and must not write anything."""
        assert not telemetry.is_enabled()
        agg = _FakeAggregator(contributors=["t1"], var_good_enough=True)
        channel = _FakeChannel(durations={"t1": timedelta(seconds=5)})

        agg._process_aggregation_goal_met(tag="aggregate", channel=channel)

        assert not (tmp_path / "aggregator.jsonl").exists()


class TestAggRoundTelemetry:
    def test_emitted_every_cycle_regardless_of_variance_outcome(self, tmp_path):
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _FakeAggregator(contributors=["t1", "t2"], var_good_enough=False)
            channel = _FakeChannel(
                durations={"t1": timedelta(seconds=5), "t2": timedelta(seconds=7)},
                utilities={"t1": 1.0, "t2": 2.0},
            )

            agg._process_aggregation_goal_met(tag="aggregate", channel=channel)

            import json
            lines = (tmp_path / "aggregator.jsonl").read_text().splitlines()
            events = [json.loads(l) for l in lines]
            rounds = [e for e in events if e["event"] == "agg_round"]
            assert len(rounds) == 1
            r = rounds[0]
            assert sorted(r["contributing_trainers"]) == ["t1", "t2"]
            assert sorted(r["trainer_speed_s"]) == [5.0, 7.0]
            assert sorted(r["stat_utility"]) == [1.0, 2.0]
        finally:
            telemetry.shutdown()

    def test_staleness_computed_against_pre_cycle_model_version(self, tmp_path):
        """staleness = the model version this cycle aggregated against, minus
        each contributor's last-known trained-on version -- captured BEFORE
        self._model_version potentially advances later in the same call."""
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _FakeAggregator(
                contributors=["t1"], var_good_enough=True,
                staleness_map={"t1": 3},  # agg's _model_version starts at 5
            )
            channel = _FakeChannel(
                durations={"t1": timedelta(seconds=1)}, utilities={"t1": 0.1}
            )

            agg._process_aggregation_goal_met(tag="aggregate", channel=channel)

            import json
            lines = (tmp_path / "aggregator.jsonl").read_text().splitlines()
            events = [json.loads(l) for l in lines]
            rounds = [e for e in events if e["event"] == "agg_round"]
            assert rounds[0]["staleness"] == [2]  # 5 - 3, not 6 - 3
        finally:
            telemetry.shutdown()

    def test_contributor_list_captured_before_reset(self, tmp_path):
        """_per_agg_trainer_list is cleared at the end of this method --
        agg_round's contributing_trainers must reflect this cycle's
        contributors, not the post-reset empty list."""
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _FakeAggregator(contributors=["a", "b", "c"], var_good_enough=False)
            channel = _FakeChannel()

            agg._process_aggregation_goal_met(tag="aggregate", channel=channel)

            assert agg._per_agg_trainer_list == []  # confirms the real reset ran
            import json
            lines = (tmp_path / "aggregator.jsonl").read_text().splitlines()
            events = [json.loads(l) for l in lines]
            rounds = [e for e in events if e["event"] == "agg_round"]
            assert sorted(rounds[0]["contributing_trainers"]) == ["a", "b", "c"]
        finally:
            telemetry.shutdown()

    def test_noop_when_telemetry_disabled(self, tmp_path):
        assert not telemetry.is_enabled()
        agg = _FakeAggregator(contributors=["t1"], var_good_enough=False)
        channel = _FakeChannel()

        agg._process_aggregation_goal_met(tag="aggregate", channel=channel)

        assert not (tmp_path / "aggregator.jsonl").exists()

    def test_agg_observed_s_keyed_by_end_id(self, tmp_path):
        """agg_observed_s reuses the same PROP_ROUND_DURATION values already
        read for trainer_speed_s, but as an {end_id: seconds} dict -- lets
        analyze_run.py's runtime_agg_vs_trainer/runtime_overhead_* plots work
        for fwdllm too (Part 6 follow-on to P5.2)."""
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _FakeAggregator(contributors=["t1", "t2"], var_good_enough=False)
            channel = _FakeChannel(
                durations={"t1": timedelta(seconds=5), "t2": timedelta(seconds=7)},
            )

            agg._process_aggregation_goal_met(tag="aggregate", channel=channel)

            import json
            lines = (tmp_path / "aggregator.jsonl").read_text().splitlines()
            events = [json.loads(l) for l in lines]
            rounds = [e for e in events if e["event"] == "agg_round"]
            assert rounds[0]["agg_observed_s"] == {"t1": 5.0, "t2": 7.0}
        finally:
            telemetry.shutdown()


class _UtilityFakeChannel:
    """Generic fake for _process_single_trainer_message's channel calls --
    stores per-end properties in a dict, doesn't care about specific PROP_*
    identities beyond PROP_STAT_UTILITY/PROP_ROUND_START_TIME (both read by
    the code path under test)."""

    class _Selector:
        def __init__(self):
            self.ordered_updates_recv_ends = []

    def __init__(self, stat_utility=None):
        self._stat_utility = dict(stat_utility or {})
        self._selector = self._Selector()

    def get_end_property(self, end, key):
        if key == PROP_STAT_UTILITY:
            return self._stat_utility.get(end)
        if key == PROP_ROUND_START_TIME:
            return None  # skip PROP_ROUND_DURATION computation, irrelevant here
        return None

    def set_end_property(self, end, key, value):
        if key == PROP_STAT_UTILITY:
            self._stat_utility[end] = value

    def set_property(self, key, value):
        pass

    def cleanup_recvd_end(self, end):
        pass

    def cleanup_provided_ends(self, end):
        pass


class _UtilityFakeAggregator:
    """Minimal stand-in exposing only the state
    _process_single_trainer_message's STAT_UTILITY/utility_belief branch
    touches -- the GRADIENTS branch is stubbed out (aggregate_grads_from_
    trainers is a no-op) since it's irrelevant to the telemetry under test."""

    process = TopAggregator._process_single_trainer_message

    def __init__(self, model_version=5, data_id=3, iteration_per_data_id=0,
                 is_async=False):
        self._per_agg_trainer_list = []
        self._trainer_last_model_version = {}
        self._updates_received = {}
        self._updates_in_queue = 0
        self._agg_goal_cnt = 0
        self._round_cache_activity_ts = {}
        self._model_version = model_version
        self.data_id = data_id
        self.iteration_per_data_id = iteration_per_data_id
        self.is_async = is_async
        self._round = 1
        self.grad_pool = []

    def aggregate_grads_from_trainers(self, *args, **kwargs):
        pass


def _msg(model_version=5, stat_utility=0.7):
    # GRADIENTS/GRADIENTS_FOR_VAR_CHECK go through _calculate_hash() (log-only,
    # unrelated to the telemetry under test) which calls .detach() on them --
    # must be real tensors, not plain lists.
    return {
        MessageType.MODEL_VERSION: model_version,
        MessageType.GRADIENTS: torch.zeros(1),
        MessageType.GRADIENTS_FOR_VAR_CHECK: torch.zeros(1),
        MessageType.STAT_UTILITY: stat_utility,
    }


class TestUtilityBeliefTelemetry:
    """fwdllm_aggregator.py never emitted utility_belief -- only
    asyncfl/top_aggregator.py did -- so selected_utility_believed_vs_actual*/
    selected_utility_belief_gap* were structurally impossible for fwdllm-
    family baselines regardless of selector (see
    ../../examples/MIGRATING_TO_LAUNCHER.md §9). Covers the fix in
    _process_single_trainer_message's STAT_UTILITY branch."""

    def test_emits_believed_and_actual(self, tmp_path):
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _UtilityFakeAggregator()
            channel = _UtilityFakeChannel(stat_utility={"t1": 0.3})  # prior belief

            agg.process(channel, _msg(stat_utility=0.9), "t1", timestamp=0)

            import json
            events = [json.loads(l) for l in
                      (tmp_path / "aggregator.jsonl").read_text().splitlines()]
            ub = [e for e in events if e["event"] == "utility_belief"]
            assert len(ub) == 1
            assert ub[0]["believed"] == 0.3
            assert ub[0]["actual"] == 0.9
            assert ub[0]["end_id"] == "t1"
        finally:
            telemetry.shutdown()

    def test_believed_none_on_first_ever_return(self, tmp_path):
        """No prior PROP_STAT_UTILITY for this end -- believed must be None,
        not a crash or a fabricated 0."""
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _UtilityFakeAggregator()
            channel = _UtilityFakeChannel()

            agg.process(channel, _msg(stat_utility=0.5), "t1", timestamp=0)

            import json
            events = [json.loads(l) for l in
                      (tmp_path / "aggregator.jsonl").read_text().splitlines()]
            ub = [e for e in events if e["event"] == "utility_belief"]
            assert ub[0]["believed"] is None
            assert ub[0]["actual"] == 0.5
        finally:
            telemetry.shutdown()

    def test_staleness_uses_model_version_not_round(self, tmp_path):
        """fwdllm's round can sit at 1 for an entire run -- staleness must be
        computed against self._model_version (the cycle-advancing quantity,
        matching agg_round's own staleness convention from P5.2), not round."""
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _UtilityFakeAggregator(model_version=8)
            channel = _UtilityFakeChannel()

            agg.process(channel, _msg(model_version=5, stat_utility=0.5), "t1", timestamp=0)

            import json
            events = [json.loads(l) for l in
                      (tmp_path / "aggregator.jsonl").read_text().splitlines()]
            ub = [e for e in events if e["event"] == "utility_belief"]
            assert ub[0]["staleness"] == 3  # 8 - 5
        finally:
            telemetry.shutdown()

    def test_carries_data_id_and_iteration_for_progress_key(self, tmp_path):
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _UtilityFakeAggregator(data_id=42, iteration_per_data_id=2)
            channel = _UtilityFakeChannel()

            agg.process(channel, _msg(stat_utility=0.5), "t1", timestamp=0)

            import json
            events = [json.loads(l) for l in
                      (tmp_path / "aggregator.jsonl").read_text().splitlines()]
            ub = [e for e in events if e["event"] == "utility_belief"]
            assert ub[0]["data_id"] == 42
            assert ub[0]["iteration_per_data_id"] == 2
        finally:
            telemetry.shutdown()

    def test_noop_when_telemetry_disabled(self, tmp_path):
        assert not telemetry.is_enabled()
        agg = _UtilityFakeAggregator()
        channel = _UtilityFakeChannel()

        agg.process(channel, _msg(stat_utility=0.5), "t1", timestamp=0)

        assert not (tmp_path / "aggregator.jsonl").exists()


class TestRoundCacheActivityResetOnContribution:
    """A real accepted contribution must restart the round-cache
    stuck-timeout clock (see TestStuckCachePruning in
    test_fwdllm_reselection.py) -- otherwise a trainer that eventually does
    respond, just slowly, would still get evicted next time the cache is
    checked."""

    def test_accepted_contribution_updates_activity_ts(self):
        agg = _UtilityFakeAggregator()
        channel = _UtilityFakeChannel()
        agg._round_cache_activity_ts["t1"] = 0.0  # ancient/never-set

        agg.process(channel, _msg(stat_utility=0.5), "t1", timestamp=0)

        assert agg._round_cache_activity_ts["t1"] > 0.0
