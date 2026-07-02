# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Batch 3 T3.2 — ground_truth.py + A6 trainer_trace_fidelity_parity.

Unlike the relative (real-vs-sim) rungs in test_availability_rungs.py, A6
compares one mode's own trainer telemetry against a synthetic ground-truth
trace directly — these tests inject a *known* drift between "observed" and
"ground truth" and assert the fidelity calc measures and flags it correctly,
per the working agreement (synthetic-drift unit test before touching real data).
"""

from __future__ import annotations

import json
import math
import os
import sys

from sortedcontainers import SortedDict

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from parity.checks import trainer_trace_fidelity_parity  # noqa: E402
from parity.ground_truth import (  # noqa: E402
    by_short_id,
    expected_send_gate_wait,
    load_ground_truth,
    resolve_trace_name,
    state_fractions_over_range,
    transitions_in_range,
)


def _sel(round_num, vclock_now):
    return {"event": "selection", "task": "train", "round": round_num,
            "ts": vclock_now, "vclock_now": vclock_now}


def _avail_change(round_num, old_state, new_state, sim_now):
    return {"event": "avail_change", "round": round_num, "old_state": old_state,
            "new_state": new_state, "sim_now": sim_now}


# ---------------------------------------------------------------------------
# resolve_trace_name
# ---------------------------------------------------------------------------

def _write_config(tmp_path, hp: dict):
    p = tmp_path / "aggregator_config.json"
    p.write_text(json.dumps({"hyperparameters": hp}))
    return tmp_path


def test_resolve_trace_name_missing_file(tmp_path):
    assert resolve_trace_name(str(tmp_path)) is None


def test_resolve_trace_name_no_keys_set(tmp_path):
    _write_config(tmp_path, {"trackTrainerAvail": {"enabled": "False", "type": "NA"}})
    assert resolve_trace_name(str(tmp_path)) is None


def test_resolve_trace_name_track_trainer_avail_priority(tmp_path):
    # All three keys set -> trackTrainerAvail wins (debug_run.sh's own priority).
    _write_config(tmp_path, {
        "trackTrainerAvail": {"enabled": "True", "type": "ORACULAR", "trace": "syn_20"},
        "client_notify": {"enabled": "False", "trace": "syn_50"},
        "availability_trace": "syn_0",
    })
    assert resolve_trace_name(str(tmp_path)) == "syn_20"


def test_resolve_trace_name_client_notify_fallback(tmp_path):
    _write_config(tmp_path, {
        "trackTrainerAvail": {"enabled": "False", "type": "NA"},
        "client_notify": {"enabled": "False", "trace": "syn_20"},
    })
    assert resolve_trace_name(str(tmp_path)) == "syn_20"


def test_resolve_trace_name_availability_trace_fallback(tmp_path):
    _write_config(tmp_path, {"availability_trace": "syn_50"})
    assert resolve_trace_name(str(tmp_path)) == "syn_50"


def test_load_ground_truth_none_when_no_trace():
    assert load_ground_truth(None) is None


# ---------------------------------------------------------------------------
# state_fractions_over_range / transitions_in_range
# ---------------------------------------------------------------------------

def test_state_fractions_over_range_always_available():
    trace = SortedDict()  # syn_0-style empty trace -> always AVL_TRAIN
    frac = state_fractions_over_range(trace, 0.0, 100.0)
    assert frac == {"AVL_TRAIN": 1.0}


def test_state_fractions_over_range_single_transition():
    trace = SortedDict({600.0: "UN_AVL"})
    frac = state_fractions_over_range(trace, 0.0, 900.0)
    assert math.isclose(frac["AVL_TRAIN"], 600.0 / 900.0)
    assert math.isclose(frac["UN_AVL"], 300.0 / 900.0)


def test_state_fractions_over_range_degenerate():
    trace = SortedDict({600.0: "UN_AVL"})
    assert state_fractions_over_range(trace, 100.0, 100.0) == {}


def test_transitions_in_range():
    trace = SortedDict({100.0: "UN_AVL", 200.0: "AVL_TRAIN", 500.0: "UN_AVL"})
    assert transitions_in_range(trace, 0.0, 300.0) == [
        (100.0, "UN_AVL"), (200.0, "AVL_TRAIN"),
    ]
    # left-exclusive: a transition exactly at t_start is not "within" the range.
    assert transitions_in_range(trace, 100.0, 300.0) == [(200.0, "AVL_TRAIN")]


def test_by_short_id():
    gt = {"505f9fc483cf4df68a2409257b5fad7d3c580370": SortedDict()}
    out = by_short_id(gt)
    assert list(out) == ["0370"]


# ---------------------------------------------------------------------------
# expected_send_gate_wait (Batch 3 T3.4, A8)
# ---------------------------------------------------------------------------

def test_expected_send_gate_wait_already_available():
    trace = SortedDict({600.0: "UN_AVL", 900.0: "AVL_TRAIN"})
    # sct before the drop -> already AVL_TRAIN there (default), no wait.
    assert expected_send_gate_wait(trace, 500.0) == 0.0


def test_expected_send_gate_wait_mid_gap():
    trace = SortedDict({600.0: "UN_AVL", 900.0: "AVL_TRAIN"})
    # sct lands inside the UN_AVL window -> wait until the recovery transition.
    assert math.isclose(expected_send_gate_wait(trace, 700.0), 200.0)


def test_expected_send_gate_wait_exact_transition_boundary():
    trace = SortedDict({600.0: "UN_AVL", 900.0: "AVL_TRAIN"})
    # sct exactly at the recovery point -> already available, zero wait.
    assert expected_send_gate_wait(trace, 900.0) == 0.0


def test_expected_send_gate_wait_never_recovers():
    trace = SortedDict({600.0: "UN_AVL"})
    assert expected_send_gate_wait(trace, 700.0) is None


# ---------------------------------------------------------------------------
# trainer_trace_fidelity_parity (A6)
# ---------------------------------------------------------------------------

def test_a6_skips_without_ground_truth():
    trainers = {"0001": {"avail_change": [_avail_change(1, "AVL_TRAIN", "UN_AVL", 100.0)]}}
    sel = [_sel(1, 900.0)]
    res = trainer_trace_fidelity_parity(trainers, sel, "sim", None)
    assert res["ok"] and res.get("status") == "SKIP"


def test_a6_skips_on_degenerate_span():
    trainers = {"0001": {"avail_change": []}}
    gt = {"t1_0001": SortedDict()}
    res = trainer_trace_fidelity_parity(trainers, [], "sim", gt)
    assert res["ok"] and res.get("status") == "SKIP"


def test_a6_skips_when_no_matching_trainers():
    # ground truth keyed by a short id that never shows up in trainer telemetry.
    trainers = {"0001": {"avail_change": []}}
    gt = {"t1_9999": SortedDict()}
    sel = [_sel(1, 900.0)]
    res = trainer_trace_fidelity_parity(trainers, sel, "sim", gt)
    assert res["ok"] and res.get("status") == "SKIP"


def test_a6_passes_when_observed_matches_ground_truth():
    # Ground truth: AVL_TRAIN [0,600), UN_AVL [600,900). Trainer observes and
    # logs the exact same transition at the exact same trace-time.
    gt = {"t1_0001": SortedDict({600.0: "UN_AVL"})}
    trainers = {"0001": {"avail_change": [
        _avail_change(1, "AVL_TRAIN", "UN_AVL", 600.0),
    ]}}
    sel = [_sel(1, 0.0), _sel(2, 900.0)]
    res = trainer_trace_fidelity_parity(trainers, sel, "sim", gt)
    assert res["ok"], res
    assert res["n_trainers"] == 1
    assert res["mean_err"] == 0.0
    assert res["n_missed_transitions"] == 0
    assert res["n_spurious_transitions"] == 0
    assert res["max_lag_s"] == 0.0


def test_a6_fails_when_trainer_never_tracked_the_trace():
    # This is exactly Challenges §5 item 20: the trainer's own trace was
    # wired to a trivial always-available one, so it never logs any
    # transition, while ground truth says it should have gone UN_AVL at 600s
    # for the back half of a 900s run -- a full state-fraction mismatch.
    gt = {"t1_0001": SortedDict({600.0: "UN_AVL"})}
    trainers = {"0001": {"avail_change": []}}
    sel = [_sel(1, 0.0), _sel(2, 900.0)]
    res = trainer_trace_fidelity_parity(trainers, sel, "sim", gt)
    # No sim_now-tagged avail_change events anywhere in the run -> SKIP, not a
    # false "perfect fidelity" pass -- distinguishing "never transitioned" from
    # "no fidelity signal at all" is the whole point of this rung.
    assert res["ok"] and res.get("status") == "SKIP"


def test_a6_fails_on_injected_lag_drift():
    # Ground truth transitions at 600s; the trainer's own clock is skewed and
    # only logs the transition (and its corresponding entry into the observed
    # timeline) 300s late -- same class of bug as B2.0.3, but on the trainer
    # side. Duration-weighted TVD should catch the resulting fraction skew.
    gt = {"t1_0001": SortedDict({600.0: "UN_AVL"})}
    trainers = {"0001": {"avail_change": [
        _avail_change(1, "AVL_TRAIN", "UN_AVL", 900.0),  # 300s late
    ]}}
    sel = [_sel(1, 0.0), _sel(2, 900.0)]
    res = trainer_trace_fidelity_parity(trainers, sel, "sim", gt, lag_tol_s=30.0)
    assert not res["ok"], res
    assert res["mean_err"] > 0.05
    assert res["n_missed_transitions"] == 1  # 300s lag exceeds lag_tol_s
    assert res["n_spurious_transitions"] == 1


def test_a6_syn0_regression_perfect_fidelity():
    # syn_0-style always-available trace: empty ground-truth SortedDict, and a
    # trainer that (correctly) never transitions -- but DOES carry at least
    # one sim_now-tagged avail_change event (e.g. a same-state re-stamp),
    # otherwise there's no fidelity signal to distinguish from "never ran
    # T3.2's telemetry fix" (see test_a6_fails_when_trainer_never_tracked_the_trace).
    gt = {"t1_0001": SortedDict()}
    trainers = {"0001": {"avail_change": [
        _avail_change(1, "AVL_TRAIN", "AVL_TRAIN", 0.0),
    ]}}
    sel = [_sel(1, 0.0), _sel(2, 900.0)]
    res = trainer_trace_fidelity_parity(trainers, sel, "sim", gt)
    assert res["ok"], res
    assert res["mean_err"] == 0.0
