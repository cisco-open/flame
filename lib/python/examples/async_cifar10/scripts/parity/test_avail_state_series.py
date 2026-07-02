# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the C.6.2 shared trainer_state_series resolver."""

from __future__ import annotations

import os
import sys

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from parity.avail_state_series import (  # noqa: E402
    build_trainer_state_series,
    run_span,
    state_fractions,
    total_variation_distance,
)


def _sel(round_num, ts, vclock_now, per_trainer):
    return {
        "event": "selection", "task": "train", "round": round_num, "ts": ts,
        "vclock_now": vclock_now, "per_trainer": per_trainer,
    }


def test_build_series_sim_uses_vclock():
    events = [
        _sel(1, 100.0, 0.0, {"t1": {"avl_state": "AVL_TRAIN"}}),
        _sel(2, 110.0, 600.0, {"t1": {"avl_state": "UN_AVL"}}),
        _sel(3, 120.0, 900.0, {"t1": {"avl_state": "AVL_TRAIN"}}),
    ]
    series = build_trainer_state_series(events, mode="sim")
    assert series["t1"] == [
        (0.0, "AVL_TRAIN"), (600.0, "UN_AVL"), (900.0, "AVL_TRAIN"),
    ]


def test_build_series_real_uses_ts_minus_t0():
    events = [
        _sel(1, 1000.0, None, {"t1": {"avl_state": "AVL_TRAIN"}}),
        _sel(2, 1300.0, None, {"t1": {"avl_state": "UN_AVL"}}),
    ]
    series = build_trainer_state_series(events, mode="real")
    assert series["t1"] == [(0.0, "AVL_TRAIN"), (300.0, "UN_AVL")]


def test_build_series_real_prefers_vclock_now_over_ts_minus_t0():
    # Real-mode first selection event fires well after the true run start
    # (e.g. ~300s of MQTT join-ramp for n=300) — ts - t0 would understate
    # elapsed time relative to the trace's agg_start-anchored origin. Once
    # vclock_now is stamped on real events too (ClientAvailability._avail_now()),
    # it must win over the ts - t0 fallback.
    events = [
        _sel(1, 1000.0, 600.0, {"t1": {"avl_state": "AVL_TRAIN"}}),
        _sel(2, 1300.0, 900.0, {"t1": {"avl_state": "UN_AVL"}}),
    ]
    series = build_trainer_state_series(events, mode="real")
    assert series["t1"] == [(600.0, "AVL_TRAIN"), (900.0, "UN_AVL")]


def test_build_series_skips_unknown_state():
    events = [_sel(1, 0.0, 0.0, {"t1": {"avl_state": "UNKNOWN"}})]
    series = build_trainer_state_series(events, mode="sim")
    assert series == {}


def test_build_series_same_t_last_write_wins():
    events = [
        _sel(1, 0.0, 100.0, {"t1": {"avl_state": "AVL_TRAIN"}}),
        _sel(2, 0.0, 100.0, {"t1": {"avl_state": "UN_AVL"}}),
    ]
    series = build_trainer_state_series(events, mode="sim")
    assert series["t1"] == [(100.0, "UN_AVL")]


def test_run_span():
    series = {"t1": [(0.0, "AVL_TRAIN"), (900.0, "UN_AVL")],
              "t2": [(0.0, "AVL_TRAIN"), (500.0, "UN_AVL")]}
    assert run_span(series) == 900.0


def test_state_fractions_dwell_time():
    # AVL_TRAIN for 600s, then UN_AVL for the remaining 300s of a 900s span.
    series = {"t1": [(0.0, "AVL_TRAIN"), (600.0, "UN_AVL"), (900.0, "UN_AVL")]}
    fracs = state_fractions(series)
    assert fracs["t1"]["AVL_TRAIN"] == 600.0 / 900.0
    assert fracs["t1"]["UN_AVL"] == 300.0 / 900.0


def test_state_fractions_credits_trailing_segment_to_t_end():
    series = {"t1": [(0.0, "AVL_TRAIN"), (600.0, "UN_AVL")]}
    fracs = state_fractions(series, t_end=900.0)
    assert fracs["t1"]["AVL_TRAIN"] == 600.0 / 900.0
    assert fracs["t1"]["UN_AVL"] == 300.0 / 900.0


def test_state_fractions_omits_single_sample_trainers():
    series = {"t1": [(0.0, "AVL_TRAIN")]}
    assert state_fractions(series) == {}


def test_total_variation_distance_identical_is_zero():
    a = {"AVL_TRAIN": 0.7, "UN_AVL": 0.3}
    assert total_variation_distance(a, dict(a)) == 0.0


def test_total_variation_distance_disjoint_is_one():
    a = {"AVL_TRAIN": 1.0}
    b = {"UN_AVL": 1.0}
    assert total_variation_distance(a, b) == 1.0


def test_total_variation_distance_partial():
    a = {"AVL_TRAIN": 0.6, "UN_AVL": 0.4}
    b = {"AVL_TRAIN": 0.5, "UN_AVL": 0.5}
    assert abs(total_variation_distance(a, b) - 0.1) < 1e-9
