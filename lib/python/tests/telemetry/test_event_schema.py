# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Schema tests for telemetry event builders and selector emission.

The cross-selector comparison feature relies on every selector emitting the
same selection-event schema, so these tests assert required fields are present
and consistent regardless of which selector produced them.
"""

import json
from datetime import timedelta

import pytest

from flame import telemetry
from flame.selector.properties import (
    PROP_AVL_STATE,
    PROP_ROUND_DURATION,
    PROP_STAT_UTILITY,
)
from flame.telemetry.events import (
    EVENT_SELECTION,
    EVENT_UTIL_DISPARITY,
    build_agg_round,
    build_selection,
    build_trainer_round,
    build_util_disparity,
)


@pytest.fixture(autouse=True)
def _reset_telemetry():
    telemetry.shutdown()
    yield
    telemetry.shutdown()


def _read(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --- builder schema --------------------------------------------------------


class TestBuilders:
    def test_selection_required_fields(self):
        ev, f = build_selection(
            round_num=2,
            task="train",
            selector="OortSelector",
            num_candidates=10,
            num_eligible=7,
            avail_composition={"AVL_TRAIN": 7, "UN_AVL": 3},
            chosen=["a", "b"],
            in_flight=2,
        )
        assert ev == EVENT_SELECTION
        for key in (
            "round", "task", "selector", "num_candidates", "num_eligible",
            "avail_composition", "chosen", "num_chosen", "in_flight",
        ):
            assert key in f
        assert f["num_chosen"] == 2

    def test_trainer_round_drops_none(self):
        ev, f = build_trainer_round(round_num=1, real_gpu_time_s=3.0)
        assert "real_gpu_time_s" in f
        assert "wait_time_s" not in f  # None omitted

    def test_agg_round_lists(self):
        ev, f = build_agg_round(round_num=5, staleness=[0, 1, 2], agg_goal=3)
        assert f["staleness"] == [0, 1, 2]
        assert f["agg_goal"] == 3

    def test_util_disparity_ratio_and_fraction(self):
        ev, f = build_util_disparity(
            round_num=1,
            elapsed_s=10.0,
            visible_samples=50,
            total_samples=200,
            utility_streamed=4.0,
            utility_full=8.0,
        )
        assert ev == EVENT_UTIL_DISPARITY
        assert f["visible_fraction"] == pytest.approx(0.25)
        assert f["utility_ratio"] == pytest.approx(0.5)

    def test_util_disparity_handles_zero_full(self):
        _, f = build_util_disparity(
            round_num=1, elapsed_s=1.0, visible_samples=1, total_samples=0,
            utility_streamed=1.0, utility_full=0.0,
        )
        assert f["utility_ratio"] is None
        assert f["visible_fraction"] is None


# --- streaming-utility disparity direction ---------------------------------


class TestUtilDisparityDirection:
    def test_full_pool_higher_utility_gives_ratio_below_one(self):
        """Oort utility = N * sqrt(mean(loss^2)). With comparable mean squared
        loss, the full pool (larger N) yields higher raw utility than the
        streamed prefix, so streamed/full < 1 -- the hypothesized disparity."""
        import math

        mean_sq = 0.5
        n_streamed, n_full = 50, 200
        util_streamed = n_streamed * math.sqrt(mean_sq)
        util_full = n_full * math.sqrt(mean_sq)
        _, f = build_util_disparity(
            round_num=1, elapsed_s=5.0,
            visible_samples=n_streamed, total_samples=n_full,
            utility_streamed=util_streamed, utility_full=util_full,
        )
        assert f["utility_ratio"] < 1.0


# --- selector emission produces a consistent schema ------------------------


def _make_oort():
    from flame.selector.oort import OortSelector

    return OortSelector(aggr_num=3)


def _make_feddance():
    from flame.selector.feddance import FedDanceSelector

    return FedDanceSelector(aggr_num=3)


@pytest.mark.parametrize("factory", [_make_oort, _make_feddance])
def test_selector_emits_selection_event(factory, make_ends, channel_props, tmp_path):
    telemetry.configure(role="aggregator", run_dir=str(tmp_path))
    selector = factory()

    ends = make_ends(count=10, prefix="t")
    # tag availability so composition is meaningful. Leave stat-utility unset
    # so both selectors take their first-round cold-start path (the realistic
    # round-1 behavior); the selection event is emitted either way.
    for i, end in enumerate(ends.values()):
        end.set_property(PROP_AVL_STATE, "AVL_TRAIN" if i < 7 else "UN_AVL")
        end.set_property(PROP_ROUND_DURATION, timedelta(seconds=i + 1))

    selector.select(
        ends, channel_props, trainer_unavail_list=[], task_to_perform="train"
    )
    telemetry.shutdown()

    recs = [r for r in _read(tmp_path / "aggregator.jsonl") if r["event"] == EVENT_SELECTION]
    assert recs, "expected at least one selection event"
    r = recs[-1]
    # required, selector-agnostic schema
    for key in (
        "round", "task", "selector", "num_candidates", "num_eligible",
        "avail_composition", "chosen", "num_chosen", "in_flight",
    ):
        assert key in r, f"{key} missing for {r['selector']}"
    assert r["num_candidates"] == 10
    # availability composition was derived from end properties
    assert r["avail_composition"].get("AVL_TRAIN", 0) == 7
    assert r["avail_composition"].get("UN_AVL", 0) == 3
