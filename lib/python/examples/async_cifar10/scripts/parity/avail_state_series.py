# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Shared per-trainer availability-state series builder (Stage C.6.2).

Single resolver, imported by both checks.py (Aa / A4dur / observation_lag rungs)
and analyze_run.py (the four plots re-pointed in C.6.4), so the forward-fill
semantics never diverge between the checker and the plotter (Challenge 12
discipline — one function, not a duplicated copy in each consumer).

Reads `per_trainer[end_id]["avl_state"]` on each `selection` event (C.6.1).
Time-base: both modes prefer `vclock_now`, stamped on the event via
`ClientAvailability._avail_now()` (sim: `_vclock.now`; real: wall-elapsed since
`agg_start_time_ts`) — the same shared origin the rest of the availability
substrate uses. Telemetry recorded before real-mode `vclock_now` stamping was
added falls back to `ts - t0` (t0 = the run's first selection event ts), which
is a *biased* estimate: real's first selection event fires only once enough
trainers have joined over MQTT (~300s for n=300), so it under-counts elapsed
time relative to the trace's true origin and skews any duration-weighted
comparison (A4dur) for trainers whose state changes mid-run. Kept only for
backward-compat with old runs — new telemetry should always carry vclock_now.
"""

from __future__ import annotations

from typing import Optional


def _event_time(e: dict, mode: str, t0: float) -> Optional[float]:
    vclock = e.get("vclock_now")
    if vclock is not None:
        return vclock
    if mode == "sim":
        return None
    ts = e.get("ts")
    return None if ts is None else ts - t0


def build_trainer_state_series(
    selection_events: list, mode: str
) -> dict[str, list]:
    """{end_id: [(t, avl_state), ...]} forward-fill series, sorted by t.

    mode: "sim" or "real", both preferring t = vclock_now, falling back to
    ts - t0 only for real-mode telemetry recorded before vclock_now was
    stamped on real selection events. One sample per
    end_id per selection event it appears as a candidate in (whether or not
    selected) — `avail_composition`/`per_trainer` already cover every
    candidate in the pool, not just the chosen subset. Consecutive samples at
    the same t collapse to the last write (same-instant events, e.g. carried
    pacer state).
    """
    real_ts = [e.get("ts") for e in selection_events if e.get("ts") is not None]
    t0 = min(real_ts) if (mode == "real" and real_ts) else 0.0

    series: dict[str, list] = {}
    for e in sorted(
        selection_events, key=lambda x: (x.get("round", 0), x.get("ts", 0.0))
    ):
        t = _event_time(e, mode, t0)
        if t is None:
            continue
        for end_id, cand in (e.get("per_trainer") or {}).items():
            state = cand.get("avl_state")
            if state is None or state == "UNKNOWN":
                continue
            pts = series.setdefault(end_id, [])
            if pts and pts[-1][0] == t:
                pts[-1] = (t, state)
            else:
                pts.append((t, state))
    return series


def build_observed_timeline_from_avail_change(avail_change_events: list) -> list:
    """Sorted [(sim_now, new_state), ...] from one trainer's own avail_change telemetry.

    ``sim_now`` (Batch 3 T3.2) is the trainer's own trace-time-basis clock at
    the moment it applied the transition — sim: virtual-clock seconds; real:
    wall-elapsed since the shared AGG_START_TS origin (T3.0). Distinct from
    the record's ``ts`` (always wall time.time(), meaningless against a trace
    indexed in trace-seconds). Events recorded before ``sim_now`` existed are
    dropped; an entirely-empty result means "no fidelity signal for this
    trainer" (old telemetry), not "trainer never transitioned" — the A6
    caller must treat empty-with-no-events differently only if it also has no
    sim_now-tagged events at all across the whole run (checked once, not per
    trainer).
    """
    pts: list = []
    for e in sorted(
        avail_change_events, key=lambda x: (x.get("round", 0), x.get("ts", 0.0))
    ):
        t = e.get("sim_now")
        state = e.get("new_state")
        if t is None or state is None:
            continue
        if pts and pts[-1][0] == t:
            pts[-1] = (t, state)
        else:
            pts.append((t, state))
    return pts


def build_observed_timeline_from_agg_belief(events: list) -> list:
    """Sorted [(observed_at, state), ...] from one trainer's own
    agg_belief_change telemetry (Batch 3 T3.3), already filtered by the
    caller to a single end_id and checkpoint ("selection" | "commit").

    ``observed_at`` is the trace-time-basis clock the belief was read at
    (vclock seconds / wall-elapsed since the shared origin) -- same role as
    avail_change's ``sim_now`` in build_observed_timeline_from_avail_change,
    just a different telemetry stream (aggregator belief vs. trainer's own
    self-report).
    """
    pts: list = []
    for e in sorted(
        events, key=lambda x: (x.get("round", 0), x.get("observed_at", 0.0))
    ):
        t = e.get("observed_at")
        state = e.get("state")
        if t is None or state is None:
            continue
        if pts and pts[-1][0] == t:
            pts[-1] = (t, state)
        else:
            pts.append((t, state))
    return pts


def selection_run_span(selection_events: list, mode: str) -> float:
    """Max observed time across all selection events.

    Unlike run_span() (which needs a built per-trainer avl_state series),
    this only needs the events' own ts/vclock_now -- usable by checks that
    just need the run's overall time horizon and nothing about per-trainer
    avail state (e.g. A6, which reads its own per-trainer state from
    avail_change telemetry instead). Same t = vclock_now / (ts - t0)
    time-base convention as build_trainer_state_series.
    """
    real_ts = [e.get("ts") for e in selection_events if e.get("ts") is not None]
    t0 = min(real_ts) if (mode == "real" and real_ts) else 0.0
    times = [
        t for t in (_event_time(e, mode, t0) for e in selection_events)
        if t is not None
    ]
    return max(times) if times else 0.0


def run_span(series: dict) -> float:
    """Max observed t across all trainers — the run's own time horizon."""
    return max((pts[-1][0] for pts in series.values() if pts), default=0.0)


def state_fractions(series: dict, t_end: Optional[float] = None) -> dict:
    """Per-trainer {state: fraction_of_span} from a forward-filled series.

    Dwell-time integration: each sample's state holds until the next sample
    (or `t_end` for the trailing segment, default = the trainer's own last
    sample — i.e. no tail credited beyond its last observation). Trainers
    with < 2 samples are omitted (no observed dwell to integrate). Per-trainer
    vectors sum to 1.
    """
    out: dict = {}
    for end_id, pts in series.items():
        if len(pts) < 2:
            continue
        durations: dict = {}
        for (t_a, s_a), (t_b, _) in zip(pts, pts[1:]):
            durations[s_a] = durations.get(s_a, 0.0) + max(0.0, t_b - t_a)
        last_t, last_s = pts[-1]
        end_t = t_end if t_end is not None else last_t
        if end_t > last_t:
            durations[last_s] = durations.get(last_s, 0.0) + (end_t - last_t)
        total = sum(durations.values())
        if total <= 0:
            continue
        out[end_id] = {s: d / total for s, d in durations.items()}
    return out


def total_variation_distance(a: dict, b: dict) -> float:
    """0.5 * Σ_s |a_s - b_s| over the union of states; 0 if identical."""
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)
