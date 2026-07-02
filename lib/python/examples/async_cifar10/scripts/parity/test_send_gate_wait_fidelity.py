# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Batch 3 T3.4 — A8 send_gate_wait_fidelity_parity.

Same synthetic-drift unit-test pattern as T3.2/T3.3 (test_ground_truth.py /
test_agg_belief_fidelity.py), applied to the trainer-side [SEND_GATE] wait
duration: does the observed send_gate_wait_s match what the raw trace says
the wait SHOULD have been, given send_gate_sct (the trainer's own
trace-time-basis clock sampled right before the gate check)?
"""

from __future__ import annotations

import os
import sys

from sortedcontainers import SortedDict

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from parity.checks import send_gate_wait_fidelity_parity  # noqa: E402


def _task_send(round_num, send_gate_wait_s, send_gate_sct):
    return {
        "event": "task_send", "round": round_num, "task_to_perform": "train",
        "send_gate_wait_s": send_gate_wait_s, "send_gate_sct": send_gate_sct,
    }


def test_a8_skips_without_ground_truth():
    trainers = {"0001": {"task_send": [_task_send(1, 200.0, 700.0)]}}
    res = send_gate_wait_fidelity_parity(trainers, None)
    assert res["ok"] and res.get("status") == "SKIP"


def test_a8_skips_when_no_matching_trainer():
    gt = {"t1_9999": SortedDict({600.0: "UN_AVL"})}
    trainers = {"0001": {"task_send": [_task_send(1, 200.0, 700.0)]}}
    res = send_gate_wait_fidelity_parity(trainers, gt)
    assert res.get("status") == "SKIP"


def test_a8_skips_when_no_gate_fields_present():
    # sim telemetry (or telemetry predating T3.4): send_gate_wait_s/sct absent.
    gt = {"t1_0001": SortedDict({600.0: "UN_AVL"})}
    trainers = {"0001": {"task_send": [
        {"event": "task_send", "round": 1, "task_to_perform": "train"},
    ]}}
    res = send_gate_wait_fidelity_parity(trainers, gt)
    assert res.get("status") == "SKIP"


def test_a8_passes_when_observed_matches_ground_truth():
    # Ground truth: AVL_TRAIN [0,600), UN_AVL [600,900), AVL_TRAIN [900,inf).
    # Trainer completed compute at sct=700 (mid-outage) -> should wait 200s.
    gt = {"t1_0001": SortedDict({600.0: "UN_AVL", 900.0: "AVL_TRAIN"})}
    trainers = {"0001": {"task_send": [_task_send(1, 200.0, 700.0)]}}
    res = send_gate_wait_fidelity_parity(trainers, gt)
    assert res["ok"], res
    assert res["mean_err_s"] == 0.0
    assert res["n_scored"] == 1


def test_a8_zero_wait_when_already_available():
    gt = {"t1_0001": SortedDict({600.0: "UN_AVL", 900.0: "AVL_TRAIN"})}
    trainers = {"0001": {"task_send": [_task_send(1, 0.0, 100.0)]}}
    res = send_gate_wait_fidelity_parity(trainers, gt)
    assert res["ok"], res
    assert res["mean_err_s"] == 0.0


def test_a8_fails_on_injected_wait_drift():
    # A promptness bug: observed wait is 300s short of what ground truth says.
    gt = {"t1_0001": SortedDict({600.0: "UN_AVL", 900.0: "AVL_TRAIN"})}
    trainers = {"0001": {"task_send": [_task_send(1, 50.0, 700.0)]}}  # expect 200.0
    res = send_gate_wait_fidelity_parity(trainers, gt, mean_tol_s=10.0)
    assert not res["ok"], res
    assert res["mean_err_s"] == 150.0


def test_a8_excludes_uncomparable_never_recovers_events():
    # Trace never recovers after sct -> excluded from scoring, not penalized.
    gt = {"t1_0001": SortedDict({600.0: "UN_AVL"})}
    trainers = {"0001": {"task_send": [
        _task_send(1, 50.0, 700.0),   # uncomparable (no recovery in trace)
        _task_send(2, 0.0, 100.0),    # already available -> comparable, err=0
    ]}}
    res = send_gate_wait_fidelity_parity(trainers, gt)
    assert res["ok"], res
    assert res["n_events"] == 2
    assert res["n_scored"] == 1
    assert res["n_uncomparable"] == 1


def test_a8_population_rollup_across_multiple_events():
    gt = {"t1_0001": SortedDict({600.0: "UN_AVL", 900.0: "AVL_TRAIN"})}
    trainers = {"0001": {"task_send": [
        _task_send(1, 200.0, 700.0),  # exact match, err=0
        _task_send(2, 190.0, 700.0),  # err=10
    ]}}
    res = send_gate_wait_fidelity_parity(trainers, gt, mean_tol_s=10.0,
                                         within_tau_s=15.0)
    assert res["n_scored"] == 2
    assert res["mean_err_s"] == 5.0
    assert res["ok"]
