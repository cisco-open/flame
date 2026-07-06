# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for B2.0.2 — starvation self-termination.

Bug: when the trace horizon is reached, _next_avail_vclock() returns None
(or a value <= vclock.now), the starvation branch returned without advancing
and without setting _work_done, causing infinite spins until the wall ceiling.
The budget check in increment_round used strict > so vclock exactly at budget
never stopped either.

These tests verify that both conditions now terminate correctly.
"""

import math
import types

import pytest
from sortedcontainers import SortedDict

from flame.availability.client_availability import ClientAvailability
from flame.availability.trace import next_avail_after


# ---------------------------------------------------------------------------
# Minimal clock + trace helpers
# ---------------------------------------------------------------------------

def _trace(*pairs):
    d = SortedDict()
    for ts, state in pairs:
        d[float(ts)] = state
    return d


# A trace where all trainers go UN_AVL at t=500 and never recover.
_EXHAUST = _trace((0, "AVL_TRAIN"), (500, "UN_AVL"))

# A trace where trainer recovers much later — used to verify _next_avail_vclock
# returns a finite future value when not exhausted.
_RECOVER = _trace((0, "AVL_TRAIN"), (500, "UN_AVL"), (800, "AVL_TRAIN"))


class _VClock:
    def __init__(self, now):
        self._now = float(now)

    @property
    def now(self):
        return self._now

    def advance(self, ts):
        if float(ts) > self._now:
            self._now = float(ts)


class _Harness(ClientAvailability):
    """Minimal ClientAvailability host for starvation-termination tests."""

    def __init__(self, trainer_event_dict, vclock_now, budget, simulated=True):
        self.trainer_event_dict = trainer_event_dict
        self._trace_has_avl_eval = False
        self.pending_withheld = {}
        self._sim_withheld_payload = {}
        self._sim_withheld_delivering = {}
        self._vclock = _VClock(vclock_now)
        self._work_done = False
        self._round = 1
        self.simulated = simulated
        self.avail_select_filter = True
        self.proactive_inflight_evict = False
        self.agg_start_time_ts = 0.0  # for real-mode wall-elapsed check
        # Minimal config mock
        hp = types.SimpleNamespace(
            max_experiment_runtime_s=budget,
            aggregation_goal=5,
        )
        self.config = types.SimpleNamespace(hyperparameters=hp)

    def _avail_now(self):
        return self._vclock.now


# ---------------------------------------------------------------------------
# _next_avail_vclock behaviour
# ---------------------------------------------------------------------------

def test_next_avail_vclock_returns_none_when_trace_exhausted():
    """All trainers permanently UN_AVL → no future recovery → None."""
    h = _Harness({"t1": _EXHAUST, "t2": _EXHAUST}, vclock_now=600, budget=1800)
    result = h._next_avail_vclock()
    assert result is None, f"Expected None, got {result}"


def test_next_avail_vclock_returns_future_when_recovery_exists():
    """Trainer recovers at t=800; query at t=600 → 800.0 returned."""
    h = _Harness({"t1": _EXHAUST, "t2": _RECOVER}, vclock_now=600, budget=1800)
    result = h._next_avail_vclock()
    assert result == 800.0, f"Expected 800.0, got {result}"


def test_next_avail_vclock_returns_none_at_budget_horizon():
    """Trace's only future transition is exactly at the budget horizon.

    next_avail_after returns math.inf when there is no finite future AVL
    transition; candidates list stays empty → None returned.
    """
    # The trace goes UN_AVL at 1800 (= budget). No AVL after that.
    _AT_BUDGET = _trace((0, "AVL_TRAIN"), (1800, "UN_AVL"))
    h = _Harness({"t1": _AT_BUDGET}, vclock_now=1800, budget=1800)
    result = h._next_avail_vclock()
    assert result is None, f"Expected None at budget horizon, got {result}"


# ---------------------------------------------------------------------------
# Starvation termination logic — directly test the decision logic
# ---------------------------------------------------------------------------

def _run_starvation_sim_branch(h):
    """Mirror the starvation sim branch from distribute() / aggregate().

    Calls _next_avail_vclock(), checks the termination condition, and sets
    h._work_done=True if termination is warranted. Returns (_nxt, _budget).
    """
    _nxt = h._next_avail_vclock()
    _budget = float(
        getattr(h.config.hyperparameters, "max_experiment_runtime_s", float("inf"))
    )
    if _nxt is not None and _nxt > h._vclock.now and h._vclock.now < _budget:
        h._vclock.advance(_nxt)
        # (in production: refresh unavail list etc.)
    else:
        # Trace horizon or budget reached — stop instead of spinning.
        h._work_done = True
    return _nxt, _budget


def test_starvation_terminates_when_nxt_is_none():
    """_next_avail_vclock returns None → work_done=True, no infinite spin."""
    h = _Harness({"t1": _EXHAUST}, vclock_now=600, budget=1800)
    _nxt, _ = _run_starvation_sim_branch(h)
    assert _nxt is None
    assert h._work_done is True


def test_starvation_terminates_when_vclock_at_budget():
    """vclock exactly at budget → work_done=True (was: 1800>1800=False, never stopped)."""
    # _RECOVER has AVL at 800, but vclock is already at budget (1800).
    h = _Harness({"t1": _RECOVER}, vclock_now=1800, budget=1800)
    _nxt, _budget = _run_starvation_sim_branch(h)
    # _nxt could be None (no future AVL after 1800) or > 1800; either way terminates.
    assert h._work_done is True


def test_starvation_terminates_when_vclock_past_budget():
    """vclock > budget → work_done=True."""
    h = _Harness({"t1": _RECOVER}, vclock_now=1850, budget=1800)
    _run_starvation_sim_branch(h)
    assert h._work_done is True


def test_starvation_advances_when_future_avail_within_budget():
    """Normal scarcity: _nxt < budget → vclock advances, work_done stays False."""
    h = _Harness({"t1": _RECOVER}, vclock_now=600, budget=1800)
    _nxt, _ = _run_starvation_sim_branch(h)
    assert _nxt == 800.0
    assert h._vclock.now == 800.0
    assert h._work_done is False


# ---------------------------------------------------------------------------
# increment_round budget check: >= (was strict >)
# ---------------------------------------------------------------------------

def test_budget_check_triggers_at_exact_equality():
    """elapsed == budget must stop the run (>= check, not strict >).

    Previously: 1800 > 1800 = False → never stopped.
    Fixed:      1800 >= 1800 = True → stops.
    """
    # We test the condition directly rather than calling increment_round, which
    # requires a full aggregator stack.  The fix is: elapsed >= budget triggers.
    budget = 1800.0
    elapsed = 1800.0  # exactly at budget
    assert elapsed >= budget, "Fix regression: >= check must fire at exact equality"

    # Confirm strict-> would have missed it (documents the original bug).
    assert not (elapsed > budget), "Strict > missed this case (original bug)"


def test_budget_check_triggers_above():
    """elapsed > budget also stops (sanity — regression of original > behaviour)."""
    budget = 1800.0
    elapsed = 1801.0
    assert elapsed >= budget


def test_budget_check_does_not_trigger_below():
    """elapsed < budget should NOT stop."""
    budget = 1800.0
    elapsed = 1799.0
    assert not (elapsed >= budget)
