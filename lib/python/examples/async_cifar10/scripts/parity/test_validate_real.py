"""Unit tests for the §4.0 real-correctness validator's sound concurrency basis.

The aggregator's in_flight/contributing_trainers cannot measure true concurrency
or re-dispatch (see validate_real docstring); these tests pin the task_send
[wall_recv_ts, wall_send_ts] interval logic that replaced them.

Run:  python -m pytest scripts/parity/test_validate_real.py -q
"""

from __future__ import annotations

import os
import sys

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from parity.validate_real import (  # noqa: E402
    _train_intervals,
    _concurrency_stats,
    _gpu_work_s,
    check_concurrency,
    speedup_ceiling,
)


def _send(recv, send, ttp="train"):
    return {"event": "task_send", "task_to_perform": ttp,
            "wall_recv_ts": recv, "wall_send_ts": send}


def test_overlap_across_trainers_is_concurrency_not_breach():
    # Two trainers both busy [0,10] -> peak concurrency 2, NO double-dispatch.
    trainer = {"0001": {"task_send": [_send(0, 10)]},
               "0002": {"task_send": [_send(0, 10)]}}
    tiv = _train_intervals(trainer)
    assert tiv["overlap_breach"] == 0
    assert _concurrency_stats(tiv["intervals"])["peak"] == 2


def test_same_trainer_overlap_is_double_dispatch_breach():
    # One trainer with two overlapping intervals = corrupt/double-dispatch.
    trainer = {"0001": {"task_send": [_send(0, 10), _send(5, 15)]}}
    tiv = _train_intervals(trainer)
    assert tiv["overlap_breach"] == 1


def test_serial_same_trainer_is_clean():
    trainer = {"0001": {"task_send": [_send(0, 10), _send(10, 20)]}}
    tiv = _train_intervals(trainer)
    assert tiv["overlap_breach"] == 0
    assert _concurrency_stats(tiv["intervals"])["peak"] == 1


def test_eval_tasks_excluded_from_train_concurrency():
    trainer = {"0001": {"task_send": [_send(0, 10, ttp="eval"), _send(0, 10)]}}
    tiv = _train_intervals(trainer)
    assert len(tiv["intervals"]) == 1  # only the train task


def test_bad_timestamps_dropped():
    trainer = {"0001": {"task_send": [_send(None, 10), _send(10, 5), _send(0, 8)]}}
    tiv = _train_intervals(trainer)
    assert tiv["dropped"] == 2
    assert len(tiv["intervals"]) == 1


def test_time_weighted_mean_concurrency():
    # [0,10] alone for 0-5, then [5,10] joins -> mean = (5*1 + 5*2)/10 = 1.5
    stats = _concurrency_stats([(0, 10), (5, 10)])
    assert stats["peak"] == 2
    assert stats["mean"] == 1.5


def test_clean_telemetry_passes_concurrency():
    agg = {"selection_train": [{"num_chosen": 3, "num_eligible": 10, "chosen": ["a", "b", "c"]}],
           "agg_rounds": []}
    tiv = _train_intervals({"0001": {"task_send": [_send(0, 10)]},
                            "0002": {"task_send": [_send(0, 10)]}})
    r = check_concurrency(agg, tiv)
    assert r["ok"] is True
    assert r["peak_concurrency"] == 2
    assert r["double_dispatch_overlaps"] == 0


def test_eligible_violation_fails():
    agg = {"selection_train": [{"num_chosen": 12, "num_eligible": 10, "chosen": list("abcdefghijkl")}],
           "agg_rounds": []}
    tiv = _train_intervals({"0001": {"task_send": [_send(0, 10)]}})
    r = check_concurrency(agg, tiv)
    assert r["eligible_violations"] == 1
    assert r["ok"] is False


def test_gpu_work_excludes_eval():
    trainer = {"0001": {"trainer_round": [
        {"task_to_perform": "train", "gpu_compute_s": 2.0},
        {"task_to_perform": "eval", "gpu_compute_s": 9.0},
        {"task_to_perform": "train", "gpu_compute_s": 3.0},
    ]}}
    assert _gpu_work_s(trainer) == 5.0


def test_speedup_ceiling_arithmetic():
    # virtual 1000s over wall 100s -> 10x intrinsic. GPU work 40s at concurrency
    # 2 -> floor 20s -> pct_of_floor 20% -> ceiling 50x.
    agg = {"agg_rounds": [
        {"ts": 0.0, "vclock_now": 0.0},
        {"ts": 100.0, "vclock_now": 1000.0},
    ]}
    trainer = {"a": {"trainer_round": [{"task_to_perform": "train", "gpu_compute_s": 40.0}]}}
    tiv = {"intervals": [(0, 10), (0, 10)]}  # mean concurrency 2 over [0,10]
    sp = speedup_ceiling(agg, trainer, tiv)
    assert sp["intrinsic_speedup_x"] == 10.0
    assert sp["mean_concurrency"] == 2.0
    assert sp["compute_floor_wall_s"] == 20.0
    assert sp["pct_of_floor"] == 20.0
    assert sp["ceiling_speedup_x"] == 50.0


def test_speedup_real_run_has_unit_intrinsic():
    # No vclock (real) -> virtual == wall -> intrinsic 1x.
    agg = {"agg_rounds": [{"ts": 0.0}, {"ts": 50.0}]}
    sp = speedup_ceiling(agg, {}, {"intervals": []})
    assert sp["intrinsic_speedup_x"] == 1.0
    assert sp["compute_floor_wall_s"] is None  # no concurrency data


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
