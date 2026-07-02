# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""RandomSelector (fwdllm's/fwdllm_plus's real aggregator-side FL selector)
never called emit_selection at all -- confirmed via audit: every "selection"
event that existed for those baselines came exclusively from the trainer's
own trivial 1-candidate channel selector (a channel-implementation artifact,
see ../../examples/MIGRATING_TO_LAUNCHER.md's telemetry gotchas), never from
the real selection decision made here. This covers the fix: select()'s SEND
branch now emits a real selection event once it actually picks candidates.
"""

import json
from datetime import timedelta

from flame import telemetry
from flame.channel import KEY_CH_SELECT_REQUESTER, KEY_CH_STATE, VAL_CH_STATE_SEND
from flame.selector.random import RandomSelector


def _make_selector(**overrides):
    kwargs = {"is_async": False, "k": 5, "c": 3}
    kwargs.update(overrides)
    return RandomSelector(**kwargs)


class TestRandomSelectorEmitsSelectionTelemetry:
    def test_emits_selection_event_on_new_pick(self, tmp_path, make_ends):
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            sel = _make_selector()
            ends = make_ends(
                count=5, stat_utility=1.0, round_duration=timedelta(seconds=2)
            )
            channel_props = {
                "round": 1,
                KEY_CH_STATE: VAL_CH_STATE_SEND,
                KEY_CH_SELECT_REQUESTER: "agg1",
            }

            result = sel.select(
                ends, channel_props, trainer_unavail_list=[], task_to_perform="train"
            )

            assert len(result) == 3  # min(len(ends), c)
            events = [
                json.loads(l)
                for l in (tmp_path / "aggregator.jsonl").read_text().splitlines()
            ]
            sels = [e for e in events if e["event"] == "selection"]
            assert len(sels) == 1
            assert sels[0]["selector"] == "RandomSelector"
            assert sels[0]["task"] == "train"
            assert sels[0]["round"] == 1
            assert sorted(sels[0]["chosen"]) == sorted(result.keys())
        finally:
            telemetry.shutdown()

    def test_eval_task_recorded_distinctly_from_train(self, tmp_path, make_ends):
        """random.py's select() takes task_to_perform -- must not hardcode
        "train" the way fedbuff.py's emit_selection call does."""
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            sel = _make_selector()
            ends = make_ends(
                count=5, stat_utility=1.0, round_duration=timedelta(seconds=2)
            )
            channel_props = {
                "round": 1,
                KEY_CH_STATE: VAL_CH_STATE_SEND,
                KEY_CH_SELECT_REQUESTER: "agg1",
            }

            sel.select(
                ends, channel_props, trainer_unavail_list=[], task_to_perform="eval"
            )

            events = [
                json.loads(l)
                for l in (tmp_path / "aggregator.jsonl").read_text().splitlines()
            ]
            sels = [e for e in events if e["event"] == "selection"]
            assert len(sels) == 1
            assert sels[0]["task"] == "eval"
        finally:
            telemetry.shutdown()

    def test_no_event_when_selection_short_circuits(self, tmp_path, make_ends):
        """When every candidate is unavailable, avl_candidates can't meet
        concurrency and the method returns {} early -- no selection actually
        happened, so no event should be emitted."""
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            sel = _make_selector(c=2)
            ends = make_ends(count=2, round_duration=timedelta(seconds=2))
            channel_props = {
                "round": 1,
                KEY_CH_STATE: VAL_CH_STATE_SEND,
                KEY_CH_SELECT_REQUESTER: "agg1",
            }

            result = sel.select(
                ends, channel_props, trainer_unavail_list=list(ends.keys()),
                task_to_perform="train",
            )

            assert result == {}
            assert not (tmp_path / "aggregator.jsonl").exists() or not [
                json.loads(l)
                for l in (tmp_path / "aggregator.jsonl").read_text().splitlines()
                if json.loads(l)["event"] == "selection"
            ]
        finally:
            telemetry.shutdown()

    def test_noop_when_telemetry_disabled(self, tmp_path, make_ends):
        assert not telemetry.is_enabled()
        sel = _make_selector()
        ends = make_ends(
            count=5, stat_utility=1.0, round_duration=timedelta(seconds=2)
        )
        channel_props = {
            "round": 1,
            KEY_CH_STATE: VAL_CH_STATE_SEND,
            KEY_CH_SELECT_REQUESTER: "agg1",
        }

        # must not raise, and must not write anything
        result = sel.select(
            ends, channel_props, trainer_unavail_list=[], task_to_perform="train"
        )

        assert len(result) == 3
        assert not (tmp_path / "aggregator.jsonl").exists()


class TestRandomSelectorAggVersionStatePassthrough:
    """fwdllm_aggregator.py threads (model_version, data_id, iteration_id)
    through channel.ends(agg_version_state=...) -> select()'s **kwargs (see
    ../../examples/MIGRATING_TO_LAUNCHER.md §9). Attaching data_id/
    iteration_per_data_id to the emitted selection event lets analyze_run.py's
    progress_key() place it on the same fine-grained axis as trainer_round/
    agg_round/agg_eval, instead of collapsing onto fwdllm's coarse `round`.
    """

    def test_agg_version_state_attaches_data_id_and_iteration(self, tmp_path, make_ends):
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            sel = _make_selector()
            ends = make_ends(
                count=5, stat_utility=1.0, round_duration=timedelta(seconds=2)
            )
            channel_props = {
                "round": 1,
                KEY_CH_STATE: VAL_CH_STATE_SEND,
                KEY_CH_SELECT_REQUESTER: "agg1",
            }

            sel.select(
                ends, channel_props, trainer_unavail_list=[], task_to_perform="train",
                agg_version_state=(7, 42, 3),  # (model_version, data_id, iteration_id)
            )

            events = [
                json.loads(l)
                for l in (tmp_path / "aggregator.jsonl").read_text().splitlines()
            ]
            sels = [e for e in events if e["event"] == "selection"]
            assert len(sels) == 1
            assert sels[0]["data_id"] == 42
            assert sels[0]["iteration_per_data_id"] == 3
        finally:
            telemetry.shutdown()

    def test_no_agg_version_state_omits_data_id(self, tmp_path, make_ends):
        """async_cifar10's fedavg baseline also uses RandomSelector but never
        passes agg_version_state -- must not crash, and must not fabricate
        data_id/iteration_per_data_id fields."""
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            sel = _make_selector()
            ends = make_ends(
                count=5, stat_utility=1.0, round_duration=timedelta(seconds=2)
            )
            channel_props = {
                "round": 1,
                KEY_CH_STATE: VAL_CH_STATE_SEND,
                KEY_CH_SELECT_REQUESTER: "agg1",
            }

            sel.select(
                ends, channel_props, trainer_unavail_list=[], task_to_perform="train",
            )

            events = [
                json.loads(l)
                for l in (tmp_path / "aggregator.jsonl").read_text().splitlines()
            ]
            sels = [e for e in events if e["event"] == "selection"]
            assert len(sels) == 1
            assert "data_id" not in sels[0]
            assert "iteration_per_data_id" not in sels[0]
        finally:
            telemetry.shutdown()
