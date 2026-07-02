# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Ground-truth trace lookups — shared infra for the Batch 3 absolute-fidelity
rungs (A6 trainer_trace_fidelity, A7 agg_belief_fidelity, A8 send_gate_wait_fidelity).

Deliberately thin (see UNAVAILABILITY_DESIGN.md Batch 3 ▶ Implementation
phases → Phase 2): reuses flame.availability.trace's load_trace/state_at/
read_trainer_unavailability almost as-is — no need to reimplement trace
loading, it already exists in exactly the shape needed. New code here is:
  (a) a trace-*name* resolver — given a run dir, which of 3 possible config
      keys holds the trace name actually used (varies by baseline, mirrors
      debug_run.sh's own 3-way substitution branch), and
  (b) a duration-weighted, time-range query wrapper around state_at, since the
      checkers need "fraction of [t_start, t_end] spent in each state", not a
      single point lookup.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

from sortedcontainers import SortedDict

from flame.availability.trace import (
    next_avail_after,
    read_trainer_unavailability,
    state_at,
)
from flame.config import TrainerAvailState

_AVL_STATES = frozenset({TrainerAvailState.AVL_TRAIN, TrainerAvailState.AVL_EVAL})


def resolve_trace_name(run_dir: str) -> Optional[str]:
    """Read aggregator_config.json in run_dir and return the configured trace name.

    Priority mirrors debug_run.sh's own ``--trace`` substitution branch (see
    UNAVAILABILITY_DESIGN.md Baseline matrix ⁺ note): oort/oort_star/refl write
    trackTrainerAvail.trace; felix/oort_star (HP-level client_notify path)
    write client_notify.trace; feddance/fedbuff write availability_trace. Try
    in that order so a run with more than one key set (e.g. a stale YAML
    default alongside the live debug_run.sh override) resolves to the one
    debug_run.sh itself would have picked, not an arbitrary one.

    Returns None if the config file is missing/unreadable or no key is set
    (gate off / syn_0-style always-available).
    """
    path = Path(run_dir) / "aggregator_config.json"
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    hp = cfg.get("hyperparameters", cfg)

    track = hp.get("trackTrainerAvail") or {}
    trace = track.get("trace")
    if trace:
        return trace

    client_notify = hp.get("client_notify") or {}
    trace = client_notify.get("trace")
    if trace:
        return trace

    return hp.get("availability_trace") or None


def load_ground_truth(
    trace_name: Optional[str], base_dir: Optional[str] = None
) -> Optional[dict]:
    """{task_id: SortedDict[ts -> state_str]} for every registered trainer.

    Thin re-export of flame.availability.trace.read_trainer_unavailability —
    the canonical per-trainer trace loader, already used aggregator-side.
    Returns None when trace_name is falsy or the registry/trace can't be read.
    """
    return read_trainer_unavailability(trace_name, base_dir=base_dir)


def by_short_id(ground_truth: Optional[dict]) -> dict:
    """Re-key {task_id: trace} to {task_id[-4:]: trace}.

    Matches the short-id convention scripts/parity/checks.py uses for trainer
    telemetry (load_trainer_jsonl_dir keys on f.stem[-4:]; short() truncates
    the same way elsewhere in this module) — lets A6/A7 join ground truth
    directly against per-trainer telemetry dicts without a separate registry
    lookup at call time.
    """
    return {task_id[-4:]: trace for task_id, trace in (ground_truth or {}).items()}


def state_fractions_over_range(
    trace: SortedDict, t_start: float, t_end: float
) -> dict:
    """Duration-weighted {state: fraction_of[t_start, t_end]}, read directly
    off the raw ground-truth trace.

    Exact dwell-time integration over the trace's own transition points
    within the range (not sampled/binned) — the ground-truth counterpart to
    avail_state_series.state_fractions(), which does the same integration
    over an *observed* (telemetry-derived) series instead.
    """
    if t_end <= t_start:
        return {}
    boundaries = [t_start]
    boundaries.extend(trace.irange(t_start, t_end, inclusive=(False, False)))
    boundaries.append(t_end)
    durations: dict = {}
    for a, b in zip(boundaries, boundaries[1:]):
        if b <= a:
            continue
        s = state_at(trace, a).value
        durations[s] = durations.get(s, 0.0) + (b - a)
    total = sum(durations.values())
    if total <= 0:
        return {}
    return {s: d / total for s, d in durations.items()}


def transitions_in_range(trace: SortedDict, t_start: float, t_end: float) -> list:
    """[(ts, state_str), ...] ground-truth transitions within (t_start, t_end]."""
    return [
        (ts, trace[ts]) for ts in trace.irange(t_start, t_end, inclusive=(False, True))
    ]


def expected_send_gate_wait(trace: SortedDict, sct: float) -> Optional[float]:
    """Ground-truth-expected [SEND_GATE] wait (Batch 3 T3.4, A8): the trainer
    became ready to send at ``sct`` (its own trace-time-basis clock) -- how
    long should it have had to wait, per the raw trace, before the gate
    releases it?

    0 if the trace already shows the trainer AVL_* at ``sct`` (no reason to
    wait), else the gap to the next AVL_* transition. Same state-check-first
    logic as ``ClientAvailability.compute_delivery_ts`` (client_availability.py,
    the sim-side send-gate) -- next_avail_after(trace, sct) alone is NOT
    enough, since it always returns the *next* transition even when the
    trainer is already available right now -- expressed as a wait DURATION
    (this - sct) rather than an absolute delivery time.

    Returns None if the trace never recovers after ``sct``
    (``next_avail_after`` == inf) -- an unanswerable "would wait forever"
    case; the caller excludes it from scoring instead of treating it as a
    numeric error.
    """
    if state_at(trace, sct) in _AVL_STATES:
        return 0.0
    nxt = next_avail_after(trace, sct)
    if math.isinf(nxt):
        return None
    return max(0.0, nxt - sct)
