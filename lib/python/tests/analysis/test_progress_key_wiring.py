# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for wiring progress_key() into accuracy_by_round/
loss_by_round/cumulative_comm_by_round (see
../../examples/MIGRATING_TO_LAUNCHER.md §5's manifest/progress_key
mechanism). fwdllm's `round` can sit at 1 for an entire run (round only
advances once all data bins finish), which previously collapsed every
agg_eval/selection event onto one x-value for these three functions
specifically. async_cifar10 records carry no data_id, so progress_key()
is a no-op there -- these tests must not change its behavior.
"""

import os
import sys

import pytest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..",
                 "scripts", "analysis"),
)

import analyze_run as ar  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_module_globals():
    saved = list(ar._PROGRESS_HIERARCHY)
    yield
    ar._PROGRESS_HIERARCHY = saved


def _use_fwdllm_hierarchy():
    """Matches lib/python/examples/fwdllm/telemetry_manifest.yaml's declared
    hierarchy, without needing a manifest file on disk."""
    ar._PROGRESS_HIERARCHY = [
        {"field": "data_id", "bound": 150},
        {"field": "iteration_per_data_id", "bound": 15},
    ]


class TestAccuracyLossByRoundFolding:
    def test_async_cifar10_no_data_id_unchanged(self):
        records = [
            {"event": "agg_eval", "round": 1, "test-accuracy": 0.5},
            {"event": "agg_eval", "round": 2, "test-accuracy": 0.6},
        ]
        acc = ar.accuracy_by_round(records)
        assert acc == {1: 0.5, 2: 0.6}

    def test_fwdllm_collapse_is_fixed(self):
        """The concrete bug: 30 agg_eval events all at round==1 (data_id
        advancing 0..29) collapsed onto one bucket under plain round."""
        _use_fwdllm_hierarchy()
        records = [
            {"event": "agg_eval", "round": 1, "data_id": i,
             "iteration_per_data_id": 0, "test-accuracy": 0.5 + i * 0.01}
            for i in range(30)
        ]
        acc = ar.accuracy_by_round(records)
        assert len(acc) == 30  # was 1 under plain round

    def test_loss_by_round_same_fold(self):
        _use_fwdllm_hierarchy()
        records = [
            {"event": "agg_eval", "round": 1, "data_id": i,
             "iteration_per_data_id": 0, "test-loss": 1.0 - i * 0.01}
            for i in range(5)
        ]
        assert len(ar.loss_by_round(records)) == 5


class TestCumulativeCommByRoundFolding:
    def test_async_cifar10_no_data_id_unchanged(self):
        records = [
            {"event": "selection", "round": 1, "task": "train", "chosen": ["a", "b"]},
            {"event": "selection", "round": 2, "task": "train", "chosen": ["a"]},
        ]
        rounds, cum = ar.cumulative_comm_by_round(records)
        assert rounds == [1, 2]
        assert cum[0] < cum[1]  # monotonically increasing

    def test_fwdllm_selection_events_fold_data_id(self):
        """Real aggregator-side selection events (selector/random.py, P5.6)
        carry data_id/iteration_per_data_id when the aggregator threads
        agg_version_state through channel.ends() -- must spread across
        distinct buckets instead of collapsing onto round==1."""
        _use_fwdllm_hierarchy()
        records = [
            {"event": "selection", "round": 1, "task": "train", "chosen": ["a"],
             "data_id": i, "iteration_per_data_id": 0}
            for i in range(10)
        ]
        rounds, cum = ar.cumulative_comm_by_round(records)
        assert len(rounds) == 10

    def test_mixed_folded_and_plain_events_dont_collide(self):
        """Trainer-side placeholder-selector events (P5.1) don't carry
        data_id and degrade to a plain-round key; must coexist with the
        far-larger folded keys from real aggregator-side events without
        overwriting each other."""
        _use_fwdllm_hierarchy()
        records = [
            # real aggregator-side event, folded key
            {"event": "selection", "round": 1, "task": "train", "chosen": ["a"],
             "data_id": 0, "iteration_per_data_id": 0},
            # trainer-side placeholder event, no data_id -> plain round key (1)
            {"event": "selection", "round": 1, "task": "train", "chosen": ["agg1"]},
        ]
        rounds, cum = ar.cumulative_comm_by_round(records)
        assert len(rounds) == 2  # both buckets present, no collision
        assert len(cum) == 2

    def test_round_zero_warmup_still_excluded(self):
        """The round-0 exclusion is checked on the raw round field, not
        progress_key -- must still exclude genuine pre-training warmup even
        though data_id is folded into the bucket key."""
        _use_fwdllm_hierarchy()
        records = [
            {"event": "selection", "round": 0, "task": "train", "chosen": ["a"] * 100,
             "data_id": 0, "iteration_per_data_id": 0},
            {"event": "selection", "round": 1, "task": "train", "chosen": ["a"],
             "data_id": 0, "iteration_per_data_id": 0},
        ]
        rounds, cum = ar.cumulative_comm_by_round(records)
        assert len(rounds) == 1  # only the round>=1 event counted


class TestCommVsAccuracySeries:
    def test_fwdllm_join_stays_consistent_after_folding(self):
        """Both accuracy_by_round and cumulative_comm_by_round now key by
        progress_key -- the <= cumulative join in comm_vs_accuracy_series
        must still line up (same axis on both sides) and produce one point
        per real eval, not one collapsed point."""
        _use_fwdllm_hierarchy()
        records = []
        for i in range(5):
            records.append({"event": "selection", "round": 1, "task": "train",
                             "chosen": ["a"], "data_id": i, "iteration_per_data_id": 0})
            records.append({"event": "agg_eval", "round": 1, "data_id": i,
                             "iteration_per_data_id": 0, "test-accuracy": 0.5 + i * 0.05})
        xs, ys = ar.comm_vs_accuracy_series(records)
        assert len(xs) == 5
        assert len(ys) == 5
        # comm is cumulative -- non-decreasing as accuracy points advance
        assert xs == sorted(xs)
