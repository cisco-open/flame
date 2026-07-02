# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Batch 3 T3.3 — A7 agg_belief_fidelity_parity.

Same synthetic-drift unit-test pattern as T3.2's test_ground_truth.py, applied
to the aggregator's BELIEF (not the trainer's own avail_change self-report).
Two independent checkpoints (selection / commit) are tested separately since
that's the whole point of tagging them apart -- an unaware baseline (oort/
fedbuff) only ever gets a meaningful commit-checkpoint score.
"""

from __future__ import annotations

import os
import sys

from sortedcontainers import SortedDict

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from parity.checks import agg_belief_fidelity_parity  # noqa: E402


def _sel(round_num, vclock_now, per_trainer):
    return {"event": "selection", "task": "train", "round": round_num,
            "ts": vclock_now, "vclock_now": vclock_now, "per_trainer": per_trainer}


def _belief(round_num, end_id, state, observed_at, checkpoint="commit"):
    return {"event": "agg_belief_change", "round": round_num, "end_id": end_id,
            "state": state, "observed_at": observed_at, "checkpoint": checkpoint}


def test_a7_skips_without_ground_truth():
    agg = {"selection_train": [_sel(1, 0.0, {}), _sel(2, 900.0, {})],
           "agg_belief_changes": []}
    res = agg_belief_fidelity_parity(agg, "sim", None)
    assert res["selection"]["ok"] and res["selection"].get("status") == "SKIP"
    assert res["commit"]["ok"] and res["commit"].get("status") == "SKIP"


def test_a7_skips_on_degenerate_span():
    agg = {"selection_train": [], "agg_belief_changes": []}
    gt = {"t1_0001": SortedDict()}
    res = agg_belief_fidelity_parity(agg, "sim", gt)
    assert res["selection"].get("status") == "SKIP"
    assert res["commit"].get("status") == "SKIP"


def test_a7_selection_checkpoint_passes_when_matched():
    # Ground truth: AVL_TRAIN [0,600), UN_AVL [600,900). Aggregator's own
    # selection-time belief (per_trainer.avl_state) tracks it exactly.
    gt = {"t1_0001": SortedDict({600.0: "UN_AVL"})}
    agg = {
        "selection_train": [
            _sel(1, 0.0, {"0001": {"avl_state": "AVL_TRAIN"}}),
            _sel(2, 600.0, {"0001": {"avl_state": "UN_AVL"}}),
            _sel(3, 900.0, {"0001": {"avl_state": "UN_AVL"}}),
        ],
        "agg_belief_changes": [],
    }
    res = agg_belief_fidelity_parity(agg, "sim", gt)
    assert res["selection"]["ok"], res["selection"]
    assert res["selection"]["mean_err"] == 0.0
    # No commit-checkpoint telemetry at all in this run -> SKIP, not a false pass.
    assert res["commit"].get("status") == "SKIP"


def test_a7_selection_checkpoint_fails_when_diverged():
    gt = {"t1_0001": SortedDict({600.0: "UN_AVL"})}
    agg = {
        "selection_train": [
            _sel(1, 0.0, {"0001": {"avl_state": "AVL_TRAIN"}}),
            _sel(2, 900.0, {"0001": {"avl_state": "AVL_TRAIN"}}),  # never saw UN_AVL
        ],
        "agg_belief_changes": [],
    }
    res = agg_belief_fidelity_parity(agg, "sim", gt)
    assert not res["selection"]["ok"], res["selection"]
    assert res["selection"]["mean_err"] > 0.0


def test_a7_commit_checkpoint_passes_when_matched():
    # Aggregator's own commit-time belief (agg_belief_change, from
    # _record_commit_belief) reads the same state ground truth has at that
    # instant -- meaningful even for an unaware baseline with no selection
    # filter at all (selection_train carries no per_trainer avl_state here).
    gt = {"t1_0001": SortedDict({600.0: "UN_AVL"})}
    agg = {
        "selection_train": [_sel(1, 0.0, {}), _sel(2, 900.0, {})],
        "agg_belief_changes": [
            _belief(1, "0001", "AVL_TRAIN", 100.0, checkpoint="commit"),
            _belief(2, "0001", "UN_AVL", 600.0, checkpoint="commit"),
        ],
    }
    res = agg_belief_fidelity_parity(agg, "sim", gt)
    assert res["commit"]["ok"], res["commit"]
    assert res["commit"]["mean_err"] == 0.0
    # No per_trainer.avl_state anywhere -> selection checkpoint has no signal.
    assert res["selection"].get("status") == "SKIP"


def test_a7_commit_checkpoint_fails_on_belief_wrong_at_its_own_instant():
    # Ground truth transitions at 600s; the aggregator's LAST commit lands at
    # 900 but still claims AVL_TRAIN -- i.e. a belief-recording bug where the
    # recorded state doesn't even match ground truth AT its own claimed
    # instant (unlike the interior-gap case below, where both endpoints are
    # individually correct and only the un-observed middle differs). This
    # must still fail post-interior-gap-fix: `max_gap_s` only excuses
    # UN-observed periods, not a wrong reading at an observed one.
    gt = {"t1_0001": SortedDict({600.0: "UN_AVL"})}
    agg = {
        "selection_train": [_sel(1, 0.0, {}), _sel(2, 1000.0, {})],
        "agg_belief_changes": [
            _belief(1, "0001", "AVL_TRAIN", 100.0, checkpoint="commit"),
            _belief(2, "0001", "AVL_TRAIN", 900.0, checkpoint="commit"),  # still wrong
        ],
    }
    res = agg_belief_fidelity_parity(agg, "sim", gt, lag_tol_s=30.0)
    assert not res["commit"]["ok"], res["commit"]
    assert res["commit"]["mean_err"] > 0.05


def test_a7_commit_checkpoint_ignores_selection_events():
    # A selection-checkpoint agg_belief_change (if one were ever emitted)
    # must not leak into the commit-checkpoint score.
    gt = {"t1_0001": SortedDict()}
    agg = {
        "selection_train": [_sel(1, 0.0, {}), _sel(2, 900.0, {})],
        "agg_belief_changes": [
            _belief(1, "0001", "UN_AVL", 100.0, checkpoint="selection"),
        ],
    }
    res = agg_belief_fidelity_parity(agg, "sim", gt)
    assert res["commit"].get("status") == "SKIP"


def test_a7_commit_checkpoint_does_not_extrapolate_past_last_commit():
    # Batch 4 finding (UNAVAILABILITY_DESIGN.md): the trainer's last commit
    # lands at 550 while still AVL_TRAIN (correctly so -- ground truth is
    # AVL_TRAIN up to 600), then it goes UN_AVL at 600 and never commits
    # again for the rest of the 900s span. Extrapolating "still AVL_TRAIN"
    # across [550, 900) would blame ~300/900 of the span on a belief that
    # was simply never re-sampled, not wrong -- the window must truncate to
    # [t_start, last observed t] instead.
    gt = {"t1_0001": SortedDict({600.0: "UN_AVL"})}
    agg = {
        "selection_train": [_sel(1, 0.0, {}), _sel(2, 900.0, {})],
        "agg_belief_changes": [
            _belief(1, "0001", "AVL_TRAIN", 100.0, checkpoint="commit"),
            _belief(2, "0001", "AVL_TRAIN", 550.0, checkpoint="commit"),
        ],
    }
    res = agg_belief_fidelity_parity(agg, "sim", gt)
    assert res["commit"]["ok"], res["commit"]
    assert res["commit"]["mean_err"] == 0.0
    # The untruncated window would have counted the 600s ground-truth
    # transition as "missed" too (it falls after the last observation) --
    # truncation excludes it from the diagnostic count as well.
    assert res["commit"]["n_missed_transitions"] == 0



def test_a7_commit_checkpoint_does_not_extrapolate_across_interior_gap():
    # Batch 4 live-run finding (UNAVAILABILITY_DESIGN.md, felix n=300 syn_50):
    # both commits are individually CORRECT -- t=100 reads AVL_TRAIN (true,
    # ground truth is AVL_TRAIN on [0,200)), t=590 reads AVL_TRAIN (true,
    # ground truth is AVL_TRAIN on [400,600)) -- but the trace dips to UN_AVL
    # on [200,400) in between, with no commit to observe it. Holding the
    # first commit's belief all the way to the second (old behavior) would
    # blame ~200/490s of that gap on "wrong belief", when neither commit was
    # ever wrong at its own instant -- same class of over-penalization the
    # tail fix (extrapolate_tail=False) already exempts, just mid-run instead
    # of at the end.
    gt = {"t1_0001": SortedDict({200.0: "UN_AVL", 400.0: "AVL_TRAIN"})}
    agg = {
        "selection_train": [_sel(1, 0.0, {}), _sel(2, 600.0, {})],
        "agg_belief_changes": [
            _belief(1, "0001", "AVL_TRAIN", 100.0, checkpoint="commit"),
            _belief(2, "0001", "AVL_TRAIN", 590.0, checkpoint="commit"),
        ],
    }
    res = agg_belief_fidelity_parity(agg, "sim", gt, lag_tol_s=30.0)
    assert res["commit"]["ok"], res["commit"]
    assert res["commit"]["mean_err"] == 0.0


def test_a7_real_and_sim_scored_independently():
    gt = {"t1_0001": SortedDict({600.0: "UN_AVL"})}
    real_agg = {
        "selection_train": [_sel(1, 0.0, {}), _sel(2, 900.0, {})],
        "agg_belief_changes": [
            _belief(1, "0001", "AVL_TRAIN", 100.0),
            _belief(2, "0001", "AVL_TRAIN", 700.0),  # real never saw UN_AVL
        ],
    }
    sim_agg = {
        "selection_train": [_sel(1, 0.0, {}), _sel(2, 900.0, {})],
        "agg_belief_changes": [
            _belief(1, "0001", "AVL_TRAIN", 100.0),
            _belief(2, "0001", "UN_AVL", 600.0),
        ],
    }
    real_res = agg_belief_fidelity_parity(real_agg, "real", gt)
    sim_res = agg_belief_fidelity_parity(sim_agg, "sim", gt)
    assert not real_res["commit"]["ok"]
    assert sim_res["commit"]["ok"]
