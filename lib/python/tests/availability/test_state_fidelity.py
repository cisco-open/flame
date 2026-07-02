# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""T4 — State-fidelity tests for ClientAvailability.

Deterministic, no cluster. Drives tiny synthetic traces through the mixin's
selection-filter and delivery-ledger methods:

  T-state-exact      state_at() matches the trace at every AVL/UN_AVL boundary
  T-eval-pool        AVL_EVAL trainers excluded from train, included in eval
  T-withhold-deliver mid-flight UN_AVL → withheld, then delivered-stale
  T-aware-vs-reactive avail_select_filter ON vs OFF; proactive_inflight_evict
  T-starvation-sync  exactly one vclock advance; self-terminates (B2.0.2 regression)
"""

import math
import types

import pytest
from sortedcontainers import SortedDict

from flame.availability.client_availability import ClientAvailability
from flame.availability.trace import next_avail_after, state_at
from flame.config import TrainerAvailState


# ---------------------------------------------------------------------------
# Trace builders
# ---------------------------------------------------------------------------

def _trace(*pairs):
    d = SortedDict()
    for ts, state in pairs:
        d[float(ts)] = state
    return d


# Standard 2-state: down [300, 600), recovers at 600.
_T1 = _trace((0, "AVL_TRAIN"), (300, "UN_AVL"), (600, "AVL_TRAIN"))

# 3-state: eval-only window [200, 400).
_T2_EVAL = _trace((0, "AVL_TRAIN"), (200, "AVL_EVAL"), (400, "AVL_TRAIN"))

# Never recovers after going down.
_NEVER = _trace((0, "AVL_TRAIN"), (300, "UN_AVL"))


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _VClock:
    def __init__(self, now):
        self._now = float(now)

    @property
    def now(self):
        return self._now

    def advance(self, ts):
        if float(ts) > self._now:
            self._now = float(ts)


class _FakeEnd:
    def __init__(self):
        self.props = {}

    def set_property(self, k, v):
        self.props[k] = v

    def get_property(self, k):
        return self.props.get(k)


class _FlatSelector:
    """oort/refl/feddance shape: flat set, no all_selected."""

    def __init__(self):
        self.selected_ends = set()

    def add(self, end):
        self.selected_ends.add(end)


class _Channel:
    def __init__(self, selector, ends):
        self._selector = selector
        self._ends = {e: _FakeEnd() for e in ends}
        self._props = {}

    def has(self, end):
        return end in self._ends

    def set_end_property(self, end, key, value):
        self._props[(end, key)] = value

    def get_end_property(self, end, key):
        return self._props.get((end, key))


class _Harness(ClientAvailability):
    """Minimal ClientAvailability host for state-fidelity tests."""

    def __init__(
        self,
        trainer_event_dict,
        vclock_now=0.0,
        budget=1800.0,
        avail_select_filter=True,
        proactive_inflight_evict=False,
        simulated=True,
    ):
        self.trainer_event_dict = trainer_event_dict
        self._trace_has_avl_eval = bool(trainer_event_dict) and any(
            TrainerAvailState.AVL_EVAL in trace.values()
            for trace in trainer_event_dict.values()
        )
        self.pending_withheld = {}
        self._sim_withheld_payload = {}
        self._sim_withheld_delivering = {}
        self.avail_select_filter = avail_select_filter
        self.proactive_inflight_evict = proactive_inflight_evict
        self._vclock = _VClock(vclock_now)
        self._work_done = False
        self._round = 1
        self.simulated = simulated
        self.agg_start_time_ts = 0.0
        hp = types.SimpleNamespace(
            max_experiment_runtime_s=budget,
            aggregation_goal=5,
        )
        self.config = types.SimpleNamespace(hyperparameters=hp)

    def _avail_now(self):
        return self._vclock.now


# ===========================================================================
# T-state-exact
# ===========================================================================

class TestStateExact:
    """state_at() matches the 2-state trace at every boundary, both directions."""

    def _st(self, t):
        return state_at(_T1, t)

    def test_pre_window_avl_train(self):
        assert self._st(0.0) == TrainerAvailState.AVL_TRAIN
        assert self._st(299.9) == TrainerAvailState.AVL_TRAIN

    def test_at_down_boundary_unavl(self):
        assert self._st(300.0) == TrainerAvailState.UN_AVL

    def test_inside_down_window_unavl(self):
        assert self._st(450.0) == TrainerAvailState.UN_AVL
        assert self._st(599.9) == TrainerAvailState.UN_AVL

    def test_at_recovery_boundary_avl_train(self):
        assert self._st(600.0) == TrainerAvailState.AVL_TRAIN

    def test_post_recovery_avl_train(self):
        assert self._st(900.0) == TrainerAvailState.AVL_TRAIN

    def test_get_curr_unavail_includes_down_trainer(self):
        h = _Harness({"t1": _T1}, vclock_now=450)
        assert "t1" in h.get_curr_unavail_trainers()

    def test_get_curr_unavail_empty_pre_window(self):
        h = _Harness({"t1": _T1}, vclock_now=100)
        assert h.get_curr_unavail_trainers() == []

    def test_get_curr_unavail_empty_post_recovery(self):
        h = _Harness({"t1": _T1}, vclock_now=700)
        assert h.get_curr_unavail_trainers() == []

    def test_ineligible_matches_unavail_for_2state_train(self):
        """On a 2-state trace, ineligible('train') == unavail (no AVL_EVAL present)."""
        h = _Harness({"t1": _T1}, vclock_now=450)
        assert h.get_curr_task_ineligible_trainers("train") == h.get_curr_unavail_trainers()

    def test_state_at_empty_trace_default_avl(self):
        """Empty SortedDict (syn_0) → always AVL_TRAIN."""
        assert state_at(SortedDict(), 9999.0) == TrainerAvailState.AVL_TRAIN


# ===========================================================================
# T-eval-pool
# ===========================================================================

class TestEvalPool:
    """3-state trace: AVL_EVAL trainer excluded from train, included in eval."""

    def _harness(self, t):
        return _Harness(
            {"t1": _T2_EVAL, "t2": _T1},
            vclock_now=t,
        )

    def test_eval_trainer_excluded_from_train(self):
        """t1 is AVL_EVAL at t=250 → excluded from 'train' dispatch."""
        h = self._harness(250)
        ineligible = h.get_curr_task_ineligible_trainers("train")
        assert "t1" in ineligible
        assert "t2" not in ineligible  # t2 is AVL_TRAIN at 250

    def test_eval_trainer_not_excluded_from_eval(self):
        """t1 is AVL_EVAL at t=250 → NOT excluded from 'eval' dispatch."""
        h = self._harness(250)
        assert "t1" not in h.get_curr_task_ineligible_trainers("eval")

    def test_unavl_trainer_excluded_from_both(self):
        """t2 is UN_AVL at t=400 → excluded from train AND eval."""
        h = self._harness(400)
        assert "t2" in h.get_curr_task_ineligible_trainers("train")
        assert "t2" in h.get_curr_task_ineligible_trainers("eval")

    def test_avl_train_trainer_eligible_for_train_not_eval(self):
        """t2 at t=100 (AVL_TRAIN) in a 3-state trace: eligible for train only.
        In a 3-state environment, only AVL_EVAL trainers participate in eval;
        AVL_TRAIN trainers are excluded from eval dispatch (not UN_AVL, just wrong role).
        """
        h = self._harness(100)
        assert "t2" not in h.get_curr_task_ineligible_trainers("train")
        # AVL_TRAIN is the excluded_state for "eval" when _trace_has_avl_eval=True
        assert "t2" in h.get_curr_task_ineligible_trainers("eval")

    def test_trace_has_avl_eval_detected(self):
        """_trace_has_avl_eval is True when any trace contains AVL_EVAL."""
        h = self._harness(0)
        assert h._trace_has_avl_eval is True

    def test_trace_has_avl_eval_false_for_2state(self):
        """2-state traces (AVL_TRAIN/UN_AVL only) → _trace_has_avl_eval=False."""
        h = _Harness({"t1": _T1})
        assert h._trace_has_avl_eval is False


# ===========================================================================
# T-withhold-deliver
# ===========================================================================

class TestWithholdDeliver:
    """mid-flight UN_AVL → withheld then delivered-stale; delivery_ts is correct."""

    def test_delivery_ts_when_down_at_sct(self):
        """Trainer is UN_AVL at sct=400 → delivery waits for recovery at 600."""
        h = _Harness({"t1": _T1})
        assert h.compute_delivery_ts("t1", sct=400.0) == 600.0

    def test_delivery_ts_immediate_when_avl_at_sct(self):
        """Trainer is AVL_TRAIN at sct=100 → immediate delivery (= sct)."""
        h = _Harness({"t1": _T1})
        assert h.compute_delivery_ts("t1", sct=100.0) == 100.0

    def test_delivery_ts_inf_when_never_recovers(self):
        """Trace never recovers → math.inf (Challenge 10 guard)."""
        h = _Harness({"t1": _NEVER})
        assert h.compute_delivery_ts("t1", sct=400.0) == math.inf

    def test_delivery_ts_at_recovery_boundary(self):
        """sct exactly at down boundary (300) → delivery at 600 (UN_AVL at 300)."""
        h = _Harness({"t1": _T1})
        assert h.compute_delivery_ts("t1", sct=300.0) == 600.0

    def test_free_stalled_slot_populates_ledger(self):
        """free_stalled_slot registers delivery_ts in pending_withheld."""
        h = _Harness({"t1": _T1}, vclock_now=0.0)
        sel = _FlatSelector()
        sel.add("t1")
        ch = _Channel(sel, ["t1"])
        delivery_ts = h.free_stalled_slot(ch, "t1", reason="test", sct=400.0)
        assert delivery_ts == 600.0
        assert h.pending_withheld["t1"] == 600.0

    def test_withheld_held_before_delivery_ts(self):
        """withheld_held_ends() includes end when vclock < delivery_ts."""
        h = _Harness({"t1": _T1}, vclock_now=500)
        h.pending_withheld["t1"] = 600.0
        assert "t1" in h.withheld_held_ends()

    def test_withheld_not_held_at_delivery_ts(self):
        """withheld_held_ends() is empty when vclock == delivery_ts."""
        h = _Harness({"t1": _T1}, vclock_now=600)
        h.pending_withheld["t1"] = 600.0
        assert "t1" not in h.withheld_held_ends()

    def test_withheld_not_held_after_delivery_ts(self):
        """withheld_held_ends() is empty when vclock > delivery_ts."""
        h = _Harness({"t1": _T1}, vclock_now=700)
        h.pending_withheld["t1"] = 600.0
        assert "t1" not in h.withheld_held_ends()

    def test_ready_withheld_ordered_by_delivery_ts(self):
        """ready_withheld() returns (end, dts) sorted ascending by dts."""
        h = _Harness({"t1": _T1, "t2": _T1}, vclock_now=700)
        h.pending_withheld = {"t1": 600.0, "t2": 650.0}
        due = h.ready_withheld()
        assert [e for e, _ in due] == ["t1", "t2"]

    def test_free_stalled_slot_noop_when_gate_off(self):
        """Gate off (trainer_event_dict=None) → free_stalled_slot is a no-op."""
        h = _Harness(None, vclock_now=0.0)
        sel = _FlatSelector()
        ch = _Channel(sel, [])
        result = h.free_stalled_slot(ch, "t1", reason="test", sct=100.0)
        assert result is None
        assert h.pending_withheld == {}

    def test_commit_withheld_removes_from_ledger(self):
        """commit_withheld() pops the end from pending_withheld."""
        h = _Harness({"t1": _T1})
        h.pending_withheld["t1"] = 600.0
        h.commit_withheld("t1")
        assert "t1" not in h.pending_withheld


# ===========================================================================
# T-aware-vs-reactive
# ===========================================================================

class TestAwareVsReactive:
    """avail_select_filter ON vs OFF; proactive_inflight_evict divergence."""

    def test_select_filter_on_excludes_unavl(self):
        h = _Harness({"t1": _T1}, vclock_now=450, avail_select_filter=True)
        assert "t1" in h.get_curr_unavail_trainers()

    def test_select_filter_off_returns_empty(self):
        """Unaware baselines (oort/fedbuff): selection filter bypassed → empty."""
        h = _Harness({"t1": _T1}, vclock_now=450, avail_select_filter=False)
        assert h.get_curr_unavail_trainers() == []

    def test_select_filter_off_ineligible_empty_both_tasks(self):
        """Unaware: ineligible list always [] for both train and eval."""
        h = _Harness({"t1": _T1}, vclock_now=450, avail_select_filter=False)
        assert h.get_curr_task_ineligible_trainers("train") == []
        assert h.get_curr_task_ineligible_trainers("eval") == []

    def test_proactive_evict_on_frees_slot_at_boundary(self):
        """proactive_inflight_evict=True: in-flight UN_AVL trainer is evicted."""
        h = _Harness(
            {"t1": _T1}, vclock_now=450,
            avail_select_filter=True, proactive_inflight_evict=True,
        )
        sel = _FlatSelector()
        sel.add("t1")  # mark t1 as in-flight
        ch = _Channel(sel, ["t1"])
        h._sim_evict_unavail_inflight(ch)
        # Slot freed: t1 removed from selected_ends
        assert "t1" not in sel.selected_ends
        # Delivery ledger populated
        assert "t1" in h.pending_withheld
        assert h.pending_withheld["t1"] == 600.0

    def test_proactive_evict_off_leaves_slot(self):
        """proactive_inflight_evict=False: in-flight trainer slot untouched."""
        h = _Harness(
            {"t1": _T1}, vclock_now=450,
            avail_select_filter=True, proactive_inflight_evict=False,
        )
        sel = _FlatSelector()
        sel.add("t1")
        ch = _Channel(sel, ["t1"])
        h._sim_evict_unavail_inflight(ch)
        # Slot NOT freed
        assert "t1" in sel.selected_ends
        assert "t1" not in h.pending_withheld

    def test_proactive_evict_ignores_avl_trainer(self):
        """proactive_inflight_evict=True: in-flight AVL trainer NOT evicted."""
        h = _Harness(
            {"t1": _T1}, vclock_now=100,  # t1 is AVL_TRAIN at t=100
            avail_select_filter=True, proactive_inflight_evict=True,
        )
        sel = _FlatSelector()
        sel.add("t1")
        ch = _Channel(sel, ["t1"])
        h._sim_evict_unavail_inflight(ch)
        assert "t1" in sel.selected_ends  # slot kept
        assert "t1" not in h.pending_withheld


# ===========================================================================
# T-starvation-sync
# ===========================================================================

class TestStarvationSync:
    """Scripted scarcity → exactly one vclock advance; self-terminates (B2.0.2)."""

    _EXHAUST = _trace((0, "AVL_TRAIN"), (500, "UN_AVL"))
    _RECOVER = _trace((0, "AVL_TRAIN"), (500, "UN_AVL"), (800, "AVL_TRAIN"))

    def _run_branch(self, h):
        """Mirror the starvation sim branch: advance or set work_done."""
        _nxt = h._next_avail_vclock()
        _budget = float(
            getattr(h.config.hyperparameters, "max_experiment_runtime_s", float("inf"))
        )
        if _nxt is not None and _nxt > h._vclock.now and h._vclock.now < _budget:
            h._vclock.advance(_nxt)
        else:
            h._work_done = True
        return _nxt

    def test_single_advance_to_recovery_time(self):
        """One starvation call → vclock jumps exactly to the recovery transition."""
        h = _Harness({"t1": self._EXHAUST, "t2": self._RECOVER}, vclock_now=600)
        _nxt = self._run_branch(h)
        assert _nxt == 800.0
        assert h._vclock.now == 800.0
        assert h._work_done is False

    def test_terminates_when_no_recovery(self):
        """All trainers exhausted → work_done=True on first call."""
        h = _Harness({"t1": self._EXHAUST, "t2": self._EXHAUST}, vclock_now=600)
        _nxt = self._run_branch(h)
        assert _nxt is None
        assert h._work_done is True

    def test_terminates_at_budget(self):
        """vclock == budget → work_done=True (the B2.0.2 fix: >= not >)."""
        h = _Harness({"t1": self._RECOVER}, vclock_now=1800, budget=1800)
        self._run_branch(h)
        assert h._work_done is True

    def test_terminates_past_budget(self):
        """vclock > budget → work_done=True."""
        h = _Harness({"t1": self._RECOVER}, vclock_now=1850, budget=1800)
        self._run_branch(h)
        assert h._work_done is True

    def test_next_avail_vclock_minimum_across_trainers(self):
        """Returns the EARLIEST next recovery when multiple trainers are down."""
        early = _trace((0, "AVL_TRAIN"), (500, "UN_AVL"), (700, "AVL_TRAIN"))
        late  = _trace((0, "AVL_TRAIN"), (500, "UN_AVL"), (900, "AVL_TRAIN"))
        h = _Harness({"t1": early, "t2": late}, vclock_now=600)
        assert h._next_avail_vclock() == 700.0

    def test_next_avail_vclock_includes_pending_withheld(self):
        """A withheld delivery_ts is a candidate too (re-enters pool earlier)."""
        # Both trainers exhausted, but pending_withheld shows one due at 650.
        h = _Harness({"t1": self._EXHAUST, "t2": self._EXHAUST}, vclock_now=600)
        h.pending_withheld["t1"] = 650.0
        nxt = h._next_avail_vclock()
        assert nxt == 650.0

    def test_next_avail_vclock_none_when_gate_off(self):
        """Gate off (trainer_event_dict=None) → None (no trace to scan)."""
        h = _Harness(None, vclock_now=100)
        assert h._next_avail_vclock() is None

    def test_k1_monotone_vclock_never_goes_back(self):
        """vclock.advance() is strictly monotone — calling it with past value is no-op."""
        h = _Harness({"t1": self._RECOVER}, vclock_now=850)
        h._vclock.advance(800.0)  # past value
        assert h._vclock.now == 850.0  # unchanged
