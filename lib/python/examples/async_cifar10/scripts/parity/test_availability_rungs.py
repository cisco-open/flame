# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Stage C parity rungs: withheld_delivery / abandon_timeout / eligible_pool_reduction.

These cover the run-independent (structural) logic of the new availability rungs
and the loader fields that feed them. Tolerance calibration vs a real syn_20
reference is deferred to an actual run (these only assert what is true regardless
of run data: ordering invariants, the vclock wall-leak guard, and SKIP/empty
handling).
"""

from __future__ import annotations

import json
import os
import sys

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from parity.checks import (  # noqa: E402
    abandon_timeout_parity,
    commit_promptness_parity,
    duration_duty_cycle_parity,
    eligible_pool_reduction_parity,
    load_agg_jsonl,
    load_trainer_jsonl_dir,
    starvation_advance_parity,
    withheld_delivery_parity,
)


def _sel(round_num, nc, ne):
    return {"event": "selection", "task": "train", "round": round_num,
            "ts": float(round_num), "num_candidates": nc, "num_eligible": ne}


# ---------------------------------------------------------------------------
# withheld_delivery
# ---------------------------------------------------------------------------

def test_withheld_delivery_skips_when_empty():
    res = withheld_delivery_parity({}, {"withheld_deliveries": []})
    assert res["ok"] and res.get("status") == "SKIP"


def test_withheld_delivery_passes_well_formed():
    evs = [
        {"end_id": "t1", "sct": 150.0, "delivery_ts": 200.0, "delay_s": 50.0,
         "staleness": 3, "accepted": True},
        {"end_id": "t2", "sct": 260.0, "delivery_ts": 260.0, "delay_s": 0.0,
         "staleness": 0, "accepted": True},
    ]
    res = withheld_delivery_parity({}, {"withheld_deliveries": evs})
    assert res["ok"], res
    assert res["n_withheld"] == 2
    assert res["accept_frac"] == 1.0
    assert res["mean_delay_s"] == 25.0


def test_withheld_delivery_fails_on_pre_completion_delivery():
    # delivery_ts < sct would be past-dating — must fail loudly.
    evs = [{"end_id": "t1", "sct": 200.0, "delivery_ts": 150.0, "staleness": 1}]
    res = withheld_delivery_parity({}, {"withheld_deliveries": evs})
    assert not res["ok"] and res["violations"]


def test_withheld_delivery_fails_on_negative_staleness():
    evs = [{"end_id": "t1", "sct": 100.0, "delivery_ts": 120.0, "staleness": -2}]
    res = withheld_delivery_parity({}, {"withheld_deliveries": evs})
    assert not res["ok"]


# ---------------------------------------------------------------------------
# commit_promptness (K11, Batch 3 T3.5)
# ---------------------------------------------------------------------------

def test_commit_promptness_skips_when_empty():
    res = commit_promptness_parity({"withheld_deliveries": []})
    assert res["ok"] and res.get("status") == "SKIP"


def test_commit_promptness_skips_when_no_actual_commit_ts():
    # Telemetry predating T3.5: delivery_ts present, actual_commit_ts absent.
    evs = [{"end_id": "t1", "sct": 100.0, "delivery_ts": 150.0}]
    res = commit_promptness_parity({"withheld_deliveries": evs})
    assert res.get("status") == "SKIP"


def test_commit_promptness_passes_near_zero_slack():
    evs = [
        {"end_id": "t1", "delivery_ts": 200.0, "actual_commit_ts": 200.0},
        {"end_id": "t2", "delivery_ts": 300.0, "actual_commit_ts": 300.4},
    ]
    res = commit_promptness_parity({"withheld_deliveries": evs})
    assert res["ok"], res
    assert res["n_events"] == 2
    assert res["n_early_violations"] == 0
    assert res["n_late_violations"] == 0


def test_commit_promptness_fails_on_early_violation():
    # Committed BEFORE its legal delivery_ts -- a hard correctness bug.
    evs = [{"end_id": "t1", "delivery_ts": 200.0, "actual_commit_ts": 150.0}]
    res = commit_promptness_parity({"withheld_deliveries": evs})
    assert not res["ok"], res
    assert res["n_early_violations"] == 1
    assert res["early_violations"] == [{"end": "t1", "slack_s": -50.0}]
    assert res["n_late_violations"] == 0


def test_commit_promptness_fails_on_late_violation():
    # Held far longer than delivery_ts required -- a promptness/scheduling bug.
    evs = [{"end_id": "t1", "delivery_ts": 200.0, "actual_commit_ts": 260.0}]
    res = commit_promptness_parity({"withheld_deliveries": evs}, late_slack_tol_s=30.0)
    assert not res["ok"], res
    assert res["n_late_violations"] == 1
    assert res["late_violations"] == [{"end": "t1", "slack_s": 60.0}]
    assert res["n_early_violations"] == 0


def test_commit_promptness_early_and_late_scored_independently():
    evs = [
        {"end_id": "t1", "delivery_ts": 200.0, "actual_commit_ts": 150.0},  # early
        {"end_id": "t2", "delivery_ts": 200.0, "actual_commit_ts": 260.0},  # late
        {"end_id": "t3", "delivery_ts": 200.0, "actual_commit_ts": 200.5},  # fine
    ]
    res = commit_promptness_parity({"withheld_deliveries": evs}, late_slack_tol_s=30.0)
    assert not res["ok"]
    assert res["n_events"] == 3
    assert res["n_early_violations"] == 1
    assert res["n_late_violations"] == 1


def test_commit_promptness_ignores_events_missing_delivery_ts():
    evs = [{"end_id": "t1", "actual_commit_ts": 200.0}]
    res = commit_promptness_parity({"withheld_deliveries": evs})
    assert res.get("status") == "SKIP"


# ---------------------------------------------------------------------------
# abandon_timeout  (vclock wall-leak control)
# ---------------------------------------------------------------------------

def test_abandon_skips_when_empty():
    res = abandon_timeout_parity({}, {"abandon_timeouts": []})
    assert res["ok"] and res.get("status") == "SKIP"


def test_abandon_passes_vclock_scale_ages():
    evs = [{"end_id": "t1", "sim_send_ts": 100.0, "vclock_now": 195.0, "age_s": 95.0},
           {"end_id": "t2", "sim_send_ts": 50.0, "vclock_now": 200.0, "age_s": 150.0}]
    res = abandon_timeout_parity({}, {"abandon_timeouts": evs})
    assert res["ok"], res
    assert res["n_abandon"] == 2 and res["mean_age_s"] >= 90


def test_abandon_fails_on_wall_clock_leak():
    # epoch-scale age => the deadline was measured on the wall, not the vclock.
    evs = [{"end_id": "t1", "sim_send_ts": 0.0, "vclock_now": 1.75e9, "age_s": 1.75e9}]
    res = abandon_timeout_parity({}, {"abandon_timeouts": evs})
    assert not res["ok"] and res["wall_leak_ends"] == ["t1"]
    assert "WALL-CLOCK LEAK" in res.get("note", "")


def test_abandon_fails_below_threshold():
    evs = [{"end_id": "t1", "sim_send_ts": 100.0, "vclock_now": 150.0, "age_s": 50.0}]
    res = abandon_timeout_parity({}, {"abandon_timeouts": evs})
    assert not res["ok"] and res["below_threshold"]


def test_abandon_age_derived_when_missing():
    evs = [{"end_id": "t1", "sim_send_ts": 100.0, "vclock_now": 195.0}]  # no age_s
    res = abandon_timeout_parity({}, {"abandon_timeouts": evs})
    assert res["ok"] and res["max_age_s"] == 95.0


# ---------------------------------------------------------------------------
# eligible_pool_reduction
# ---------------------------------------------------------------------------

def test_eligible_pool_reduction_skips_without_fields():
    real = {"selection_train": [{"event": "selection", "round": 1, "ts": 1.0}]}
    sim = {"selection_train": [{"event": "selection", "round": 1, "ts": 1.0}]}
    res = eligible_pool_reduction_parity(real, sim)
    assert res["ok"] and res.get("status") == "SKIP"


def test_eligible_pool_reduction_passes_when_matched():
    real = {"selection_train": [_sel(1, 300, 240), _sel(2, 300, 250)]}  # red 60,50
    sim = {"selection_train": [_sel(1, 300, 245), _sel(2, 300, 248)]}   # red 55,52
    res = eligible_pool_reduction_parity(real, sim)
    assert res["ok"], res


def test_eligible_pool_reduction_fails_when_divergent():
    real = {"selection_train": [_sel(1, 300, 290), _sel(2, 300, 292)]}  # red ~9
    sim = {"selection_train": [_sel(1, 300, 200), _sel(2, 300, 210)]}   # red ~95
    res = eligible_pool_reduction_parity(real, sim)
    assert not res["ok"]


# ---------------------------------------------------------------------------
# duty_cycle_duration (A4dur, C.6.3)
# ---------------------------------------------------------------------------

def _sel_avl(round_num, ts, vclock_now, per_trainer):
    return {"event": "selection", "task": "train", "round": round_num, "ts": ts,
            "vclock_now": vclock_now, "per_trainer": per_trainer}


def test_duty_cycle_duration_skips_without_avl_state():
    real = {"selection_train": [{"event": "selection", "round": 1, "ts": 1.0,
                                  "per_trainer": {"t1": {}}}]}
    sim = {"selection_train": [{"event": "selection", "round": 1, "ts": 1.0,
                                 "vclock_now": 0.0, "per_trainer": {"t1": {}}}]}
    res = duration_duty_cycle_parity(real, sim)
    assert res["ok"] and res.get("status") == "SKIP"


def test_duty_cycle_duration_passes_when_matched():
    # t1: AVL_TRAIN for [0,600), UN_AVL for [600,900) in both modes.
    real = {"selection_train": [
        _sel_avl(1, 1000.0, None, {"t1": {"avl_state": "AVL_TRAIN"}}),
        _sel_avl(2, 1600.0, None, {"t1": {"avl_state": "UN_AVL"}}),
        _sel_avl(3, 1900.0, None, {"t1": {"avl_state": "UN_AVL"}}),
    ]}
    sim = {"selection_train": [
        _sel_avl(1, 0.0, 0.0, {"t1": {"avl_state": "AVL_TRAIN"}}),
        _sel_avl(2, 0.0, 600.0, {"t1": {"avl_state": "UN_AVL"}}),
        _sel_avl(3, 0.0, 900.0, {"t1": {"avl_state": "UN_AVL"}}),
    ]}
    res = duration_duty_cycle_parity(real, sim)
    assert res["ok"], res
    assert res["n_trainers"] == 1
    assert res["mean_err"] == 0.0


def test_duty_cycle_duration_fails_when_divergent():
    # real: AVL_TRAIN the whole span. sim: UN_AVL the whole span -> TVD = 1.0.
    real = {"selection_train": [
        _sel_avl(1, 1000.0, None, {"t1": {"avl_state": "AVL_TRAIN"}}),
        _sel_avl(2, 1900.0, None, {"t1": {"avl_state": "AVL_TRAIN"}}),
    ]}
    sim = {"selection_train": [
        _sel_avl(1, 0.0, 0.0, {"t1": {"avl_state": "UN_AVL"}}),
        _sel_avl(2, 0.0, 900.0, {"t1": {"avl_state": "UN_AVL"}}),
    ]}
    res = duration_duty_cycle_parity(real, sim)
    assert not res["ok"]
    assert res["mean_err"] == 1.0


# ---------------------------------------------------------------------------
# loaders surface the new events
# ---------------------------------------------------------------------------

def test_load_agg_jsonl_surfaces_new_events(tmp_path):
    p = tmp_path / "aggregator_x.jsonl"
    lines = [
        {"event": "selection", "task": "train", "round": 1, "ts": 1.0,
         "num_candidates": 300, "num_eligible": 250},
        {"event": "withheld_delivery", "round": 2, "ts": 2.0, "end_id": "t1",
         "sct": 10.0, "delivery_ts": 20.0, "delay_s": 10.0, "staleness": 1},
        {"event": "abandon_timeout", "round": 3, "ts": 3.0, "end_id": "t2",
         "sim_send_ts": 5.0, "vclock_now": 100.0, "age_s": 95.0},
    ]
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    agg = load_agg_jsonl(str(p))
    assert len(agg["withheld_deliveries"]) == 1
    assert len(agg["abandon_timeouts"]) == 1
    assert agg["withheld_deliveries"][0]["end_id"] == "t1"


# ---------------------------------------------------------------------------
# starvation_advance
# ---------------------------------------------------------------------------

def _agg_round(vclock_now):
    return {"event": "agg_round", "vclock_now": vclock_now}


def _sim_with_avail(rounds, withheld=True):
    """Build a sim dict with gate-active marker and given agg_round list."""
    return {
        "agg_rounds": rounds,
        "withheld_deliveries": [{"end_id": "t0"}] if withheld else [],
        "abandon_timeouts": [],
        "selection_train": [],
    }


def test_starvation_advance_skips_without_avail():
    # no withheld/abandon/avail_composition → gate off → SKIP
    sim = {"agg_rounds": [_agg_round(10.0), _agg_round(20.0), _agg_round(30.0)],
           "withheld_deliveries": [], "abandon_timeouts": [], "selection_train": []}
    res = starvation_advance_parity({}, sim)
    assert res["ok"] and res.get("status") == "SKIP"


def test_starvation_advance_skips_with_no_agg_rounds():
    sim = _sim_with_avail([])
    res = starvation_advance_parity({}, sim)
    assert res["ok"] and res.get("status") == "SKIP"


def test_starvation_advance_no_jumps():
    sim = _sim_with_avail([_agg_round(float(i * 100)) for i in range(10)])
    res = starvation_advance_parity({}, sim)
    assert res["ok"] and res["n_starvation_jumps"] == 0


def test_starvation_advance_detects_jump():
    # 8 normal 100s gaps then one 3000s gap (30× mean)
    vclocks = [float(i * 100) for i in range(8)] + [800.0 + 3000.0]
    sim = _sim_with_avail([_agg_round(v) for v in vclocks])
    res = starvation_advance_parity({}, sim, jump_factor=5.0)
    assert res["ok"] and res["n_starvation_jumps"] >= 1
    assert res["max_jump_s"] > 1000.0


def test_load_trainer_jsonl_dir_surfaces_avail_change(tmp_path):
    p = tmp_path / "trainer_0001.jsonl"
    lines = [
        {"event": "task_recv", "round": 1},
        {"event": "avail_change", "round": 1, "old_state": "AVL_TRAIN",
         "new_state": "UN_AVL"},
    ]
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    tr = load_trainer_jsonl_dir(str(tmp_path))
    assert "0001" in tr
    assert len(tr["0001"]["avail_change"]) == 1
    assert tr["0001"]["avail_change"][0]["new_state"] == "UN_AVL"
