"""Validate that a REAL run is internally correct *before* it is used as
the parity reference.

Parity is not the goal — a *correct* simulator is.  Real is the reference only
after it obeys its own invariants; otherwise matching sim to real just fits sim
to a bug.  This module asserts the invariants the parity ladder
assumes hold on the real side:

  concurrency  — true in-flight overlap (peak + time-weighted mean); no trainer
                 is busy on two overlapping tasks (double-dispatch guard)
  selection    — chosen-list is self-consistent (len==num_chosen); chosen<=eligible
  aggregation  — agg_goal_count cycles 1..K (no lost/double update); staleness
                 is non-negative; report the holding/advance identity

**The sound basis for concurrency is the `task_send` event, not the aggregator's
selection fields.**  The aggregator's `in_flight` == `num_chosen` (a per-round
selected count, NOT true concurrency) and `effective_c` is absent.  `trainer_round`
can't help either: it is emitted inside train(), *before* the real-mode budget
sleep, so its ts ~= task-start, not completion.  `task_send` fires from
_send_weights AFTER the sleep + upload, carrying [wall_recv_ts, wall_send_ts] that
brackets the trainer's true busy window.  Concurrency = max/mean overlap of those
intervals; a double-dispatch = two overlapping intervals for one trainer (the
trainer loop is serial, so this is exactly 0 unless telemetry is corrupt).

There is no configured static concurrency cap (oort/refl/feddance concurrency is
emergent via overcommitment + availability), so we report peak/mean concurrency
(the "real holds ~c computing" number) rather than asserting
`in_flight <= c`.  This assumes the run carries the task_send event (every run
from this code onward does).

Usage:
    python scripts/parity_check.py --validate-real <real_run_dir>
    # or directly:
    python -m scripts.parity.validate_real <real_run_dir>

Exit 0 = all enforced invariants hold; exit 1 = at least one violated.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from parity.checks import load_run_dir, agg_goal_cycles_ok  # noqa: E402


# Allow a tiny fraction of telemetry edge cases (interleaving/late events) before
# calling an invariant violated — these are hard assertions, not distributions.
_VIOLATION_FRAC_TOL = 0.005


def _train_intervals(trainer: dict) -> dict:
    """Per-trainer in-flight intervals [wall_recv_ts, wall_send_ts] for *train*
    tasks, from the task_send event.  Returns the flat interval list plus a count
    of intra-trainer overlaps (a double-dispatch — impossible in the serial
    trainer loop, so non-zero means corrupt telemetry, not a real breach)."""
    intervals: list = []      # (start, end) across all trainers
    overlap_breach = 0        # one trainer busy on two overlapping tasks
    dropped = 0               # task_send with missing/invalid timestamps
    for _sid, d in trainer.items():
        spans = []
        for e in d.get("task_send", []):
            if e.get("task_to_perform") != "train":
                continue
            r, s = e.get("wall_recv_ts"), e.get("wall_send_ts")
            if r is None or s is None or s < r:
                dropped += 1
                continue
            spans.append((float(r), float(s)))
        spans.sort()
        prev_end = None
        for r, s in spans:
            if prev_end is not None and r < prev_end:
                overlap_breach += 1
            prev_end = s if prev_end is None else max(prev_end, s)
            intervals.append((r, s))
    return {
        "intervals": intervals,
        "overlap_breach": overlap_breach,
        "dropped": dropped,
    }


def _concurrency_stats(intervals: list) -> dict:
    """Peak and time-weighted mean of the concurrency step function."""
    if not intervals:
        return {"peak": 0, "mean": 0.0, "span_s": 0.0}
    pts = sorted(
        [(s, 1) for s, _ in intervals] + [(e, -1) for _, e in intervals],
        key=lambda x: (x[0], x[1]),  # release (-1) before acquire (+1) at equal ts
    )
    cur = peak = 0
    area = 0.0
    last_t = pts[0][0]
    for t, delta in pts:
        area += cur * (t - last_t)
        last_t = t
        cur += delta
        peak = max(peak, cur)
    span = pts[-1][0] - pts[0][0]
    return {"peak": peak, "mean": round(area / span, 2) if span > 0 else 0.0,
            "span_s": round(span, 1)}


def _gpu_work_s(trainer: dict) -> float:
    """Total real GPU compute across all train rounds — the irreducible work."""
    total = 0.0
    for _sid, d in trainer.items():
        for e in d.get("trainer_round", []):
            if e.get("task_to_perform") == "eval":
                continue
            g = e.get("gpu_compute_s")
            if g is not None:
                total += float(g)
    return total


def speedup_ceiling(agg: dict, trainer: dict, tiv: dict) -> dict:
    """How much speedup we got and how much headroom remains (informational).

    intrinsic_speedup = virtual_s / wall_s — virtual time simulated per wall
    second (= the run's speedup vs real-time; ~1 for a real run).
    compute_floor_s = total GPU work / mean concurrency — the minimum wall a
    perfectly-efficient run could take (GPU-bound).  pct_of_floor = floor/wall:
    100% means we're at the GPU floor; <100% means overhead (telemetry, logging,
    recv loop, aggregation) dominates and there's headroom.  ceiling_speedup =
    virtual_s / floor = the best achievable speedup if overhead were zero.
    """
    ar = agg["agg_rounds"]
    ts = [e.get("ts") for e in ar if e.get("ts") is not None]
    wall_s = (max(ts) - min(ts)) if len(ts) > 1 else 0.0
    vclocks = [e.get("vclock_now") for e in ar if e.get("vclock_now") is not None]
    virtual_s = max(vclocks) if vclocks else wall_s  # real run: virtual == wall
    gpu_work = _gpu_work_s(trainer)
    mean_conc = _concurrency_stats(tiv["intervals"])["mean"]
    floor_s = (gpu_work / mean_conc) if mean_conc > 0 else None
    return {
        "wall_s": round(wall_s, 1),
        "virtual_s": round(virtual_s, 1),
        "intrinsic_speedup_x": round(virtual_s / wall_s, 2) if wall_s > 0 else None,
        "gpu_work_s": round(gpu_work, 1),
        "mean_concurrency": mean_conc,
        "compute_floor_wall_s": round(floor_s, 1) if floor_s else None,
        "pct_of_floor": round(100 * floor_s / wall_s, 1) if (floor_s and wall_s > 0) else None,
        "ceiling_speedup_x": round(virtual_s / floor_s, 1) if floor_s else None,
    }


def _infer_agg_goal(agg: dict) -> int:
    for e in agg["agg_rounds"]:
        g = e.get("agg_goal")
        if g:
            return int(g)
    return 0


def check_concurrency(agg: dict, tiv: dict) -> dict:
    """True in-flight concurrency (peak + mean) + no double-dispatch + chosen<=eligible."""
    sel = agg["selection_train"]
    elig_viol = sum(
        1 for e in sel
        if e.get("num_eligible") is not None and e.get("num_chosen") is not None
        and e["num_chosen"] > e["num_eligible"]
    )
    stats = _concurrency_stats(tiv["intervals"])
    # Enforce: never select more than eligible, and no trainer doubly in-flight.
    # Concurrency peak/mean is reported (no configured cap to assert against).
    ok = elig_viol == 0 and tiv["overlap_breach"] == 0
    return {
        "ok": ok,
        "peak_concurrency": stats["peak"],
        "mean_concurrency": stats["mean"],
        "measured_over_s": stats["span_s"],
        "double_dispatch_overlaps": tiv["overlap_breach"],
        "eligible_violations": elig_viol,
        "n_train_intervals": len(tiv["intervals"]),
        "dropped_bad_ts": tiv["dropped"],
    }


def check_selection(agg: dict) -> dict:
    """chosen-list self-consistency (len == num_chosen) + chosen <= eligible.

    The re-dispatch-while-in-flight invariant is enforced in check_concurrency as
    the double-dispatch overlap test (the sound, timestamp-based form).  The old
    contributing_trainers-based redispatch metric was a bookkeeping artifact (it
    never cleared overcommit-discarded completions) and is removed.
    """
    sel = agg["selection_train"]
    len_mismatch = sum(
        1 for e in sel
        if e.get("num_chosen") is not None and len(e.get("chosen") or []) != e["num_chosen"]
    )
    n_sel = len(sel)
    chosen_trainers = {str(t) for e in sel for t in (e.get("chosen") or [])}
    ok = len_mismatch / max(n_sel, 1) <= _VIOLATION_FRAC_TOL
    return {
        "ok": ok,
        "len_mismatches": len_mismatch,
        "n_chosen_trainers": len(chosen_trainers),
        "n_selections": n_sel,
    }


def check_aggregation(agg: dict) -> dict:
    """agg_goal cycles 1..K; staleness non-negative; holding/advance identity."""
    agg_goal = _infer_agg_goal(agg)
    cyc = agg_goal_cycles_ok(agg, agg_goal)

    neg = 0
    n_stale = 0
    stale_sum = 0.0
    for e in agg["agg_rounds"]:
        for s in (e.get("staleness") or []):
            if s is None:
                continue
            n_stale += 1
            stale_sum += s
            if s < 0:
                neg += 1
    mean_stale = (stale_sum / n_stale) if n_stale else None

    # Diagnostic identity: mean staleness ~ holding / advance. advance = mean
    # Δts between consecutive FL rounds (wall, real mode).
    by_round = {}
    for e in agg["agg_rounds"]:
        r = e.get("round")
        if r is not None:
            by_round[r] = max(by_round.get(r, 0.0), e.get("ts", 0.0))
    rounds = sorted(by_round)
    advances = [by_round[rounds[i]] - by_round[rounds[i - 1]]
                for i in range(1, len(rounds))
                if by_round[rounds[i]] - by_round[rounds[i - 1]] > 0]
    mean_adv = (sum(advances) / len(advances)) if advances else None
    implied_holding = (mean_stale * mean_adv) if (mean_stale and mean_adv) else None

    ok = cyc.get("ok", True) and neg == 0
    return {
        "ok": ok,
        "agg_goal": agg_goal,
        "rounds_over_goal": len(cyc.get("rounds_over_goal", []) or []),
        "negative_staleness": neg,
        "mean_staleness": round(mean_stale, 3) if mean_stale is not None else None,
        "mean_advance_s": round(mean_adv, 3) if mean_adv is not None else None,
        "implied_holding_s": round(implied_holding, 2) if implied_holding else None,
    }


def validate_real(run_dir: str) -> bool:
    agg, trainer = load_run_dir(run_dir)
    tiv = _train_intervals(trainer)
    label = Path(run_dir.rstrip("/")).name
    print("=" * 78)
    print(f"  REAL-CORRECTNESS VALIDATION  {label}")
    print(f"  ({len(agg['agg_rounds'])} agg_round, {len(agg['selection_train'])} selection, "
          f"{len(trainer)} trainers, {len(tiv['intervals'])} train intervals)")
    print("=" * 78)

    results = {
        "concurrency": check_concurrency(agg, tiv),
        "selection": check_selection(agg),
        "aggregation": check_aggregation(agg),
    }
    all_ok = True
    for name, r in results.items():
        ok = r.pop("ok")
        all_ok = all_ok and ok
        tag = "[OK]  " if ok else "[FAIL]"
        print(f"\n  {tag} {name}")
        for k, v in r.items():
            print(f"          {k}: {v}")

    # Speedup / headroom — informational, does not affect admissibility.
    sp = speedup_ceiling(agg, trainer, tiv)
    print("\n  [INFO] speedup / headroom")
    for k, v in sp.items():
        print(f"          {k}: {v}")
    if sp["pct_of_floor"] is not None and sp["pct_of_floor"] < 50:
        print("          -> overhead-bound (telemetry/logging/recv/agg dominate); "
              f"~{round(100 / max(sp['pct_of_floor'], 1e-9), 1)}x wall headroom to the GPU floor.")

    print("\n" + "=" * 78)
    print("  REAL IS ADMISSIBLE as the parity reference."
          if all_ok else
          "  REAL VIOLATES an invariant — fix real before matching sim.")
    print("=" * 78)
    return all_ok


def main(argv=None) -> None:
    # Robust to a latin-1 stdout (e.g. invoked without PYTHONIOENCODING=utf-8): the
    # report prints a few non-ASCII glyphs (-> arrow, em dash, §). Reconfigure rather
    # than chase each char.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: validate_real.py <real_run_dir>", file=sys.stderr)
        sys.exit(2)
    ok = validate_real(argv[0])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
