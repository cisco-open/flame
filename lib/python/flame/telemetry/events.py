# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Telemetry event types and typed builders.

Centralizing the event names + field schemas here is what makes runs
comparable across selectors / aggregators / examples. Emitters should call the
``build_*`` helpers (or :func:`flame.telemetry.emit` with these constants) so
every run produces the same columns.
"""

from __future__ import annotations

from typing import Any, Optional

# ---- Event type constants -------------------------------------------------

EVENT_RUN_META = "run_meta"          # one-time run identity / config snapshot
EVENT_SELECTION = "selection"        # selector decision for a round
EVENT_AGG_EVAL = "agg_eval"          # aggregator test loss/accuracy
EVENT_AGG_ROUND = "agg_round"        # aggregation step: staleness/agg-goal/participation
EVENT_TRAINER_ROUND = "trainer_round"  # per-round trainer timing/availability
EVENT_UTIL_DISPARITY = "util_disparity"  # streamed-prefix vs full-pool utility
EVENT_AVAIL_CHANGE = "avail_change"  # trainer availability state transition
EVENT_TASK_RECV = "task_recv"        # trainer received a task from aggregator
EVENT_TASK_SEND = "task_send"        # trainer finished & sent the update back
EVENT_INFLIGHT_RESIDENCE = "inflight_residence"  # per-round in-flight drain accounting (oort sync)
EVENT_UTILITY_BELIEF = "utility_belief"  # believed (at selection) vs actual (at return) client utility
EVENT_DISPATCH = "dispatch"          # per-dispatch re-dispatch-stagger validation (felix)
EVENT_WITHHELD_DELIVERY = "withheld_delivery"  # late stale delivery of a send-gated update
EVENT_ABANDON_TIMEOUT = "abandon_timeout"      # 90s vclock slot-free of a stalled trainer
EVENT_AGG_BELIEF_CHANGE = "agg_belief_change"   # aggregator's belief about a trainer's avail state

KNOWN_EVENTS = frozenset(
    {
        EVENT_RUN_META,
        EVENT_SELECTION,
        EVENT_AGG_EVAL,
        EVENT_AGG_ROUND,
        EVENT_TRAINER_ROUND,
        EVENT_UTIL_DISPARITY,
        EVENT_AVAIL_CHANGE,
        EVENT_TASK_RECV,
        EVENT_TASK_SEND,
        EVENT_INFLIGHT_RESIDENCE,
        EVENT_UTILITY_BELIEF,
        EVENT_DISPATCH,
        EVENT_WITHHELD_DELIVERY,
        EVENT_ABANDON_TIMEOUT,
        EVENT_AGG_BELIEF_CHANGE,
    }
)


# ---- Typed builders -------------------------------------------------------
# Each returns (event_type, fields_dict). Callers do:
#   telemetry.emit(*build_selection(...))  -> emit(event, **fields)
# but emit() takes (event, **fields), so callers use:
#   ev, f = build_selection(...); telemetry.emit(ev, **f)


def build_selection(
    *,
    round_num: int,
    task: str,
    selector: str,
    num_candidates: int,
    num_eligible: int,
    avail_composition: dict[str, int],
    chosen: list[str],
    in_flight: int,
    per_trainer: Optional[dict[str, dict[str, Any]]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> tuple[str, dict[str, Any]]:
    """Selector decision record.

    avail_composition: counts keyed by availability-state name
        (e.g. {"AVL_TRAIN": 30, "AVL_EVAL": 5, "UN_AVL": 65}).
    per_trainer: optional {end_id: {"utility": .., "speed_s": .., "selected": bool}}.
    extra: selector-specific fields (e.g. explore/exploit split, cutoff).
    """
    fields: dict[str, Any] = {
        "round": round_num,
        "task": task,
        "selector": selector,
        "num_candidates": num_candidates,
        "num_eligible": num_eligible,
        "avail_composition": avail_composition,
        "chosen": list(chosen),
        "num_chosen": len(chosen),
        "in_flight": in_flight,
    }
    if per_trainer is not None:
        fields["per_trainer"] = per_trainer
    if extra:
        fields.update(extra)
    return EVENT_SELECTION, fields


def build_agg_eval(
    *, round_num: int, metrics: dict[str, float]
) -> tuple[str, dict[str, Any]]:
    """Aggregator evaluation metrics (loss/accuracy/...)."""
    fields = {"round": round_num}
    fields.update(metrics)
    return EVENT_AGG_EVAL, fields


def build_agg_round(
    *,
    round_num: int,
    agg_goal: Optional[int] = None,
    agg_goal_count: Optional[int] = None,
    in_flight: Optional[int] = None,
    updates_in_queue: Optional[int] = None,
    staleness: Optional[list[float]] = None,
    stat_utility: Optional[list[float]] = None,
    trainer_speed_s: Optional[list[float]] = None,
    contributing_trainers: Optional[list[str]] = None,
    agg_observed_s: Optional[dict[str, float]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> tuple[str, dict[str, Any]]:
    """Aggregation-step record (one per completed aggregation).

    agg_observed_s: {end_id -> wall seconds the aggregator observed between
        sending the model and receiving/processing that trainer's update}. Lets
        the analyzer compare aggregator-side turnaround to the trainer-reported
        time (overhead sanity check).
    """
    fields: dict[str, Any] = {"round": round_num}
    for k, v in (
        ("agg_goal", agg_goal),
        ("agg_goal_count", agg_goal_count),
        ("in_flight", in_flight),
        ("updates_in_queue", updates_in_queue),
        ("staleness", staleness),
        ("stat_utility", stat_utility),
        ("trainer_speed_s", trainer_speed_s),
        ("contributing_trainers", contributing_trainers),
        ("agg_observed_s", agg_observed_s),
    ):
        if v is not None:
            fields[k] = v
    if extra:
        fields.update(extra)
    return EVENT_AGG_ROUND, fields


def build_trainer_round(
    *,
    round_num: int,
    real_gpu_time_s: float,
    sim_round_duration_s: Optional[float] = None,
    wait_time_s: Optional[float] = None,
    avail_state: Optional[str] = None,
    visible_samples: Optional[int] = None,
    total_samples: Optional[int] = None,
    dataset_size: Optional[int] = None,
    stat_utility: Optional[float] = None,
    final_loss: Optional[float] = None,
    delta_weight_l2: Optional[float] = None,
    extra: Optional[dict[str, Any]] = None,
) -> tuple[str, dict[str, Any]]:
    """Per-round trainer timing/availability record.

    delta_weight_l2: L2 norm of the model update (||trained - received global||)
        the trainer uploads -- lets analysis relate update magnitude to the
        amount of unlocked data under streaming.
    """
    fields: dict[str, Any] = {
        "round": round_num,
        "real_gpu_time_s": real_gpu_time_s,
    }
    for k, v in (
        ("sim_round_duration_s", sim_round_duration_s),
        ("wait_time_s", wait_time_s),
        ("avail_state", avail_state),
        ("visible_samples", visible_samples),
        ("total_samples", total_samples),
        ("dataset_size", dataset_size),
        ("stat_utility", stat_utility),
        ("final_loss", final_loss),
        ("delta_weight_l2", delta_weight_l2),
    ):
        if v is not None:
            fields[k] = v
    if extra:
        fields.update(extra)
    return EVENT_TRAINER_ROUND, fields


def build_util_disparity(
    *,
    round_num: int,
    elapsed_s: float,
    visible_samples: int,
    total_samples: int,
    utility_streamed: float,
    utility_full: float,
    sample_size_used: Optional[int] = None,
) -> tuple[str, dict[str, Any]]:
    """Streamed-prefix vs full-dataset statistical-utility comparison."""
    visible_fraction = (
        visible_samples / total_samples if total_samples else None
    )
    ratio = (
        utility_streamed / utility_full
        if utility_full not in (0, None)
        else None
    )
    fields: dict[str, Any] = {
        "round": round_num,
        "elapsed_s": elapsed_s,
        "visible_samples": visible_samples,
        "total_samples": total_samples,
        "visible_fraction": visible_fraction,
        "utility_streamed": utility_streamed,
        "utility_full": utility_full,
        "utility_ratio": ratio,
    }
    if sample_size_used is not None:
        fields["sample_size_used"] = sample_size_used
    return EVENT_UTIL_DISPARITY, fields


def build_avail_change(
    *,
    round_num: Optional[int],
    old_state: str,
    new_state: str,
    sim_now: Optional[float] = None,
) -> tuple[str, dict[str, Any]]:
    """Trainer availability state transition.

    sim_now: the trainer's own trace-time-basis clock (``_sim_now()``) at the
    moment the transition was applied — sim: virtual-clock seconds; real:
    wall-elapsed since the shared AGG_START_TS origin (Batch 3 T3.0). Distinct
    from the record's own wall ``ts`` (always epoch time.time(), meaningless
    against a trace indexed in trace-seconds). Needed for A6
    (trainer_trace_fidelity, Batch 3 T3.2) to compare *when* a trainer applied
    a transition against ground truth, not just *that* it eventually did.
    Optional/back-compat: None for telemetry recorded before this field existed.
    """
    return EVENT_AVAIL_CHANGE, {
        "round": round_num,
        "old_state": old_state,
        "new_state": new_state,
        "sim_now": sim_now,
    }


def build_task_recv(
    *,
    round_num: int,
    trainer_id: str,
    time_mode: str,
    sim_send_ts: Optional[float] = None,
    avl_state: Optional[str] = None,
) -> tuple[str, dict[str, Any]]:
    """Trainer received a task (weights) from the aggregator.

    sim_send_ts: the virtual clock value stamped by the aggregator (sim mode only).
    Emitting None in real mode for both sim_send_ts and vclock makes the per-round
    trainer state directly comparable between real and sim telemetry.
    """
    fields: dict[str, Any] = {
        "round": round_num,
        "trainer_id": trainer_id,
        "time_mode": time_mode,
        "sim_send_ts": sim_send_ts,
        "avl_state": avl_state,
    }
    return EVENT_TASK_RECV, fields


def build_task_send(
    *,
    round_num: int,
    trainer_id: str,
    task_to_perform: Optional[str],
    wall_recv_ts: Optional[float],
    wall_send_ts: float,
    time_mode: str,
    send_gate_wait_s: Optional[float] = None,
    send_gate_sct: Optional[float] = None,
) -> tuple[str, dict[str, Any]]:
    """Trainer finished a task and sent the update back to the aggregator.

    Unlike trainer_round (emitted inside train(), BEFORE the real-mode budget
    sleep), this fires from _send_weights — AFTER the sleep and the upload — so
    ``[wall_recv_ts, wall_send_ts]`` brackets the trainer's true busy/in-flight
    window in real mode.  That interval is the sound basis for real concurrency
    in validate_real: trainer_round's own ts cannot bracket it.

    send_gate_wait_s / send_gate_sct (Batch 3 T3.4, real mode only): the
    [SEND_GATE] wait loop also runs strictly after trainer_round is emitted
    (train() -> put()/_send_weights, per the tasklet composition), so — like
    wall_send_ts above — task_send, not trainer_round, is the event that can
    actually carry it. send_gate_sct is the trainer's own _sim_now() sampled
    right before the gate check (same trace-time-basis clock T3.2's
    avail_change.sim_now uses); send_gate_wait_s is the wall-time actually
    spent blocked in the loop (0.0 when the gate never engaged). Always None
    in sim mode (the gate is a real-mode-only mechanism).
    """
    return EVENT_TASK_SEND, {
        "round": round_num,
        "trainer_id": trainer_id,
        "task_to_perform": task_to_perform,
        "wall_recv_ts": wall_recv_ts,
        "wall_send_ts": wall_send_ts,
        "time_mode": time_mode,
        "send_gate_wait_s": send_gate_wait_s,
        "send_gate_sct": send_gate_sct,
    }


def build_inflight_residence(
    *,
    round_num: int,
    time_mode: str,
    in_flight_before: int,
    in_flight_after: int,
    newly_selected: Optional[int] = None,
    committed_fresh: Optional[int] = None,
    cleaned: Optional[int] = None,
    stale_rejected: Optional[int] = None,
    residence_rounds: Optional[list[int]] = None,
    carried_over_ages: Optional[list[int]] = None,
    residence_staleness: Optional[list[int]] = None,
    residence_was_fresh: Optional[list[bool]] = None,
) -> tuple[str, dict[str, Any]]:
    """Per-round in-flight drain accounting for the oort sync aggregator.

    Localizes the in-flight RESIDENCE divergence (real holds ~15.6 in-flight, sim
    drains to the designed ~13): a straggler occupies ``selected_ends`` from selection
    until it is cleaned. ``residence_rounds`` = (current_round − entry_round) for each
    trainer cleaned this round; ``carried_over_ages`` = ages of those still in-flight
    AFTER cleanup. Comparing sim vs real residence distributions shows whether sim
    evicts stragglers a round too early (the eviction-timing fine-tune). ``time_mode``
    = "sim"|"real" so the two are directly comparable.

    ``residence_staleness`` / ``residence_was_fresh`` are PAIRED 1:1 with
    ``residence_rounds`` (same order, same cleaned ends): the commit staleness
    (``round − trained_version``) and the fresh-vs-stale-reject class of each cleaned
    end. They decompose the residence-distribution SHAPE gap (refl A2: real peaks at
    residence=3, sim flatter) by commit class — i.e. whether sim under-holds the
    fresh-committed body or the stale-carryover tail.
    """
    fields: dict[str, Any] = {
        "round": round_num,
        "time_mode": time_mode,
        "in_flight_before": in_flight_before,
        "in_flight_after": in_flight_after,
    }
    for k, v in (
        ("newly_selected", newly_selected),
        ("committed_fresh", committed_fresh),
        ("cleaned", cleaned),
        ("stale_rejected", stale_rejected),
        ("residence_rounds", residence_rounds),
        ("carried_over_ages", carried_over_ages),
        ("residence_staleness", residence_staleness),
        ("residence_was_fresh", residence_was_fresh),
    ):
        if v is not None:
            fields[k] = v
    return EVENT_INFLIGHT_RESIDENCE, fields


def build_dispatch(
    *,
    round_num: int,
    end_id: str,
    task: str,
    time_mode: str,
    sim_send_ts: Optional[float] = None,
    redispatch_stagger_s: Optional[float] = None,
    held_s: Optional[float] = None,
    staggered: Optional[bool] = None,
) -> tuple[str, dict[str, Any]]:
    """Per-dispatch re-dispatch-stagger validation (felix event-driven re-dispatch).

    ``redispatch_stagger_s`` = this end's ``sim_send_ts`` minus the cohort minimum
    in the same distribute call: 0 for the legacy round-boundary batch (all share
    one frozen vclock), spread across the round's advance once event-driven
    re-dispatch is on. ``held_s`` = vclock minus this end's PRIOR commit sct = how
    long (virtual seconds) it sat held since it last completed before being
    re-dispatched; the boundary backlog shows large held_s, continuous re-dispatch
    drives it toward 0. Sim-only fields; lets the run confirm the cohort next-sct
    spread recovers real's ~3.85s before reading K2/K3b/U3.
    """
    fields: dict[str, Any] = {
        "round": round_num,
        "end_id": end_id,
        "task": task,
        "time_mode": time_mode,
    }
    for k, v in (
        ("sim_send_ts", sim_send_ts),
        ("redispatch_stagger_s", redispatch_stagger_s),
        ("held_s", held_s),
        ("staggered", staggered),
    ):
        if v is not None:
            fields[k] = v
    return EVENT_DISPATCH, fields


def build_utility_belief(
    *,
    round_num: int,
    end_id: str,
    believed: Optional[float],
    actual: Optional[float],
    staleness: Optional[int] = None,
    time_mode: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> tuple[str, dict[str, Any]]:
    """Believed-vs-actual client statistical utility, per returning trainer.

    ``believed`` = the utility the selector held for this client when it was selected
    (``PROP_STAT_UTILITY`` *before* this return overwrites it — the value from the
    client's previous return, i.e. STALE by ``staleness`` rounds). ``actual`` = the
    fresh Oort statistical utility the client computed this round and reports on return
    (``MessageType.STAT_UTILITY``). Both are the SAME quantity (Oort stat-utility), so
    ``believed − actual`` is the pure staleness error in the selector's belief — the
    quantity the "believed vs actual utility" plot needs. Emitted for EVERY baseline
    (every client reports stat-utility on return, even non-utility selectors), so the
    plot compares felix/eval-refreshed beliefs against the stale-utility baselines.
    ``believed`` is None on a client's first-ever return (no prior belief)."""
    fields: dict[str, Any] = {
        "round": round_num,
        "end_id": end_id,
        "believed": believed,
        "actual": actual,
    }
    if staleness is not None:
        fields["staleness"] = staleness
    if time_mode is not None:
        fields["time_mode"] = time_mode
    if extra:
        fields.update(extra)
    return EVENT_UTILITY_BELIEF, fields


def build_withheld_delivery(
    *,
    round_num: int,
    end_id: str,
    sct: float,
    delivery_ts: float,
    staleness: Optional[int] = None,
    accepted: Optional[bool] = None,
    time_mode: str = "sim",
    actual_commit_ts: Optional[float] = None,
) -> tuple[str, dict[str, Any]]:
    """A send-gated update committing late (stale) at its ``delivery_ts``.

    ``delivery_ts − sct`` is the down-window delay the completed update waited
    while its trainer was ``UN_AVL`` (the compute-completes / gate-the-send /
    deliver-late model). ``staleness`` = the current round minus the update's
    ``MODEL_VERSION``; ``accepted`` records the staleness-gate outcome (async
    fedbuff always accepts; sync feddance may reject over tolerance — Stage E).
    Backs the ``withheld_delivery`` parity rung.

    actual_commit_ts (Batch 3 T3.5, K11): the aggregator's own clock
    (``_avail_now()`` — vclock in sim, wall-elapsed in real) sampled at the
    instant this update actually commits, i.e. AFTER the caller's
    ``_advance_sim_clock``/equivalent for this specific update, not before —
    the whole point is to catch reinjection-polling lag between ``delivery_ts``
    (the earliest legal commit time — already exactly what ``delivery_ts`` is,
    no separate ``earliest_legally_committable_time`` bookkeeping needed for
    this single-gate-type v1) and when the update actually lands.
    ``commit_slack_s = actual_commit_ts - delivery_ts`` is derived by the
    caller-facing K11 check, not stored here, to keep this builder a thin
    field-carrier like its siblings.
    """
    fields: dict[str, Any] = {
        "round": round_num,
        "end_id": end_id,
        "sct": sct,
        "delivery_ts": delivery_ts,
        "delay_s": float(delivery_ts) - float(sct),
        "time_mode": time_mode,
    }
    if staleness is not None:
        fields["staleness"] = staleness
    if accepted is not None:
        fields["accepted"] = accepted
    if actual_commit_ts is not None:
        fields["actual_commit_ts"] = float(actual_commit_ts)
    return EVENT_WITHHELD_DELIVERY, fields


def build_abandon_timeout(
    *,
    round_num: int,
    end_id: str,
    sim_send_ts: float,
    vclock_now: float,
    time_mode: str = "sim",
    reason: str = "abandon_90s_vclock",
) -> tuple[str, dict[str, Any]]:
    """A stalled in-flight trainer freed by a slot-free trigger.

    ``reason`` distinguishes C.3 90s-vclock abandons from D.1 aware boundary
    evictions. ``vclock_now − sim_send_ts`` is the in-flight age at trigger.
    Backs the ``abandon_timeout`` parity rung, which fails loudly if the
    deadline is measured on the wall instead of the vclock.
    """
    return EVENT_ABANDON_TIMEOUT, {
        "round": round_num,
        "end_id": end_id,
        "sim_send_ts": sim_send_ts,
        "vclock_now": vclock_now,
        "age_s": float(vclock_now) - float(sim_send_ts),
        "time_mode": time_mode,
        "reason": reason,
    }


def build_agg_belief_change(
    *,
    round_num: int,
    end_id: str,
    state: str,
    observed_at: float,
    checkpoint: str,
    source: str = "trace_read",
) -> tuple[str, dict[str, Any]]:
    """The aggregator's belief about `end_id`'s availability state (Batch 3 T3.3).

    ``checkpoint`` distinguishes WHERE the aggregator's view was read:
    ``"selection"`` (pre-round eligibility read) or ``"commit"`` (state at an
    update's completion time, whether or not the mechanism actually gates on
    it — real mode never gates here, sim's send-gate does; recording both
    lets A7 compare the aggregator's belief against ground truth regardless).
    ``source`` is the knowledge MECHANISM the belief came from — ``trace_read``
    (v1, live) today, ``client_notify``/``predictive`` later (Stage H) — kept
    mechanism-agnostic so a future populator just passes a different source
    without any telemetry/checker/plot change. ``observed_at`` is the
    trace-time-basis clock (vclock seconds / wall-elapsed since the shared
    origin) at which the belief was read, NOT the record's own wall ``ts``.
    """
    return EVENT_AGG_BELIEF_CHANGE, {
        "round": round_num,
        "end_id": end_id,
        "state": state,
        "observed_at": observed_at,
        "checkpoint": checkpoint,
        "source": source,
    }
