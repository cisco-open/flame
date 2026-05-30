#!/usr/bin/env python
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Post-run telemetry analyzer for FLAME experiments.

Reads the JSONL event streams written by :mod:`flame.telemetry` (one file per
process in a run's telemetry directory) and emits a bundle of PNG plots plus a
short text summary. Schema-driven, so it works identically across selectors /
aggregators -- enabling apples-to-apples comparison.

Usage
-----
    python analyze_run.py <telemetry_dir> [--out <plots_dir>]
    python analyze_run.py --compare <dir1> <dir2> ... [--labels a b ...]

``<telemetry_dir>`` contains ``aggregator_*.jsonl`` and ``trainer_*.jsonl``.
Default output is ``<telemetry_dir>/../plots``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_helpers as ph  # noqa: E402

# Event-type names (kept in sync with flame/telemetry/events.py). Imported from
# the package when available, else hardcoded so the analyzer also runs stand-alone.
try:
    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib", "python"
        ),
    )
    from flame.telemetry.events import (  # noqa: E402
        EVENT_AGG_EVAL,
        EVENT_AGG_ROUND,
        EVENT_AVAIL_CHANGE,
        EVENT_SELECTION,
        EVENT_TRAINER_ROUND,
        EVENT_UTIL_DISPARITY,
    )
except Exception:  # pragma: no cover - fallback for standalone use
    EVENT_SELECTION = "selection"
    EVENT_AGG_EVAL = "agg_eval"
    EVENT_AGG_ROUND = "agg_round"
    EVENT_TRAINER_ROUND = "trainer_round"
    EVENT_UTIL_DISPARITY = "util_disparity"
    EVENT_AVAIL_CHANGE = "avail_change"


def load_events(telemetry_dir: str) -> list[dict]:
    """Load all JSONL records from a telemetry directory."""
    records: list[dict] = []
    for path in sorted(glob.glob(os.path.join(telemetry_dir, "*.jsonl"))):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # skip a torn final line from a killed process
    return records


def by_event(records: list[dict], event: str) -> list[dict]:
    return [r for r in records if r.get("event") == event]


# --- individual plots -------------------------------------------------------


def plot_accuracy_loss(records, out_dir) -> list[str]:
    rows = by_event(records, EVENT_AGG_EVAL)
    if not rows:
        return []
    rows.sort(key=lambda r: r.get("round", 0))
    rounds = [r.get("round") for r in rows]
    saved = []
    # accept either "test-accuracy"/"test-loss" or generic keys
    acc = [r.get("test-accuracy") for r in rows]
    loss = [r.get("test-loss") for r in rows]
    if any(a is not None for a in acc):
        rs = [r for r, a in zip(rounds, acc) if a is not None]
        av = [a for a in acc if a is not None]
        p = ph.line_plot(
            {"test-accuracy": (rs, av)}, "round", "accuracy",
            "Aggregator test accuracy over rounds", out_dir, "accuracy_over_rounds.png",
        )
        if p:
            saved.append(p)
    if any(x is not None for x in loss):
        rs = [r for r, x in zip(rounds, loss) if x is not None]
        lv = [x for x in loss if x is not None]
        p = ph.line_plot(
            {"test-loss": (rs, lv)}, "round", "loss",
            "Aggregator test loss over rounds", out_dir, "loss_over_rounds.png",
        )
        if p:
            saved.append(p)
    return saved


def plot_staleness(records, out_dir) -> list[str]:
    rows = by_event(records, EVENT_AGG_ROUND)
    if not rows:
        return []
    saved = []
    stale_vals = []
    for r in rows:
        for s in (r.get("staleness") or []):
            stale_vals.append(s)
    if stale_vals:
        p = ph.cdf_plot(
            stale_vals, "update staleness (rounds)", "Update staleness CDF",
            out_dir, "staleness_cdf.png",
        )
        if p:
            saved.append(p)
    # in-flight / queue timeline (by event arrival order)
    inflight = [(i, r.get("updates_in_queue")) for i, r in enumerate(rows)
                if r.get("updates_in_queue") is not None]
    if inflight:
        xs = [i for i, _ in inflight]
        ys = [v for _, v in inflight]
        p = ph.line_plot(
            {"updates_in_queue": (xs, ys)}, "aggregation step", "updates in queue",
            "Async queue depth over aggregation steps", out_dir, "queue_depth.png",
        )
        if p:
            saved.append(p)
    return saved


def plot_avail_composition(records, out_dir) -> list[str]:
    rows = by_event(records, EVENT_SELECTION)
    if not rows:
        return []
    rows.sort(key=lambda r: r.get("round", 0))
    # one composition per round (last selection in that round wins)
    per_round: dict[int, dict] = {}
    for r in rows:
        comp = r.get("avail_composition")
        if comp:
            per_round[r.get("round", 0)] = comp
    if not per_round:
        return []
    rounds = sorted(per_round.keys())
    state_names = sorted({s for c in per_round.values() for s in c.keys()})
    series = {s: [per_round[rd].get(s, 0) for rd in rounds] for s in state_names}
    p = ph.stacked_area(
        rounds, series, "round", "trainer count",
        "Availability composition over rounds (selector view)",
        out_dir, "availability_composition.png",
    )
    return [p] if p else []


def plot_utility_speed_scatter(records, out_dir) -> list[str]:
    rows = by_event(records, EVENT_SELECTION)
    xs, ys, sel = [], [], []
    for r in rows:
        per = r.get("per_trainer") or {}
        for _, info in per.items():
            u = info.get("utility")
            s = info.get("speed_s")
            if u is None or s is None:
                continue
            xs.append(s)
            ys.append(u)
            sel.append(bool(info.get("selected")))
    if not xs:
        return []
    p = ph.scatter_plot(
        xs, ys, sel, "round duration / speed (s)", "statistical utility",
        "Utility vs speed: selected vs eligible", out_dir, "utility_vs_speed.png",
    )
    return [p] if p else []


def plot_selection_frequency(records, out_dir) -> list[str]:
    rows = by_event(records, EVENT_SELECTION)
    counter: Counter = Counter()
    for r in rows:
        for eid in (r.get("chosen") or []):
            counter[str(eid)] += 1
    if not counter:
        return []
    items = counter.most_common()
    cats = [k for k, _ in items]
    vals = [v for _, v in items]
    p = ph.bar_plot(
        cats, vals, "times selected", "Selection frequency per trainer (fairness)",
        out_dir, "selection_frequency.png",
    )
    return [p] if p else []


def plot_trainer_time_breakdown(records, out_dir) -> list[str]:
    rows = by_event(records, EVENT_TRAINER_ROUND)
    if not rows:
        return []
    agg: dict[str, dict[str, list]] = defaultdict(
        lambda: {"gpu": [], "sim": [], "wait": []}
    )
    for r in rows:
        tid = str(r.get("end_id", "?"))
        agg[tid]["gpu"].append(r.get("real_gpu_time_s") or 0.0)
        agg[tid]["sim"].append(r.get("sim_round_duration_s") or 0.0)
        agg[tid]["wait"].append(r.get("wait_time_s") or 0.0)
    cats = sorted(agg.keys())

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    segments = {
        "real_gpu_time_s": [mean(agg[c]["gpu"]) for c in cats],
        "sim_round_duration_s": [mean(agg[c]["sim"]) for c in cats],
        "wait_time_s": [mean(agg[c]["wait"]) for c in cats],
    }
    p = ph.stacked_bar(
        cats, segments, "mean seconds per round",
        "Trainer time breakdown: real GPU vs simulated delay vs wait",
        out_dir, "trainer_time_breakdown.png",
    )
    return [p] if p else []


def plot_util_disparity(records, out_dir) -> list[str]:
    rows = by_event(records, EVENT_UTIL_DISPARITY)
    if not rows:
        return []
    saved = []
    # ratio over elapsed time, one series per trainer
    by_trainer: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_trainer[str(r.get("end_id", "?"))].append(r)
    ratio_series = {}
    streamed_series = {}
    full_series = {}
    for tid, rs in by_trainer.items():
        rs.sort(key=lambda r: r.get("elapsed_s", 0))
        xs = [r.get("elapsed_s") for r in rs]
        ratio_series[tid] = (xs, [r.get("utility_ratio") for r in rs])
        streamed_series["%s streamed" % tid] = (xs, [r.get("utility_streamed") for r in rs])
        full_series["%s full" % tid] = (xs, [r.get("utility_full") for r in rs])
    p = ph.line_plot(
        ratio_series, "elapsed sim time (s)", "streamed / full utility ratio",
        "Streamed-vs-full utility ratio over time (1.0 = no disparity)",
        out_dir, "util_disparity_ratio.png",
    )
    if p:
        saved.append(p)
    # absolute streamed vs full (combine; can be busy with many trainers)
    combined = {}
    combined.update(streamed_series)
    combined.update(full_series)
    p = ph.line_plot(
        combined, "elapsed sim time (s)", "statistical utility",
        "Streamed-prefix vs full-dataset utility over time",
        out_dir, "util_disparity_absolute.png",
    )
    if p:
        saved.append(p)
    return saved


def write_summary(records, out_dir, telemetry_dir) -> str:
    counts = Counter(r.get("event") for r in records)
    n_trainers = len({r.get("end_id") for r in records if r.get("role") == "trainer"})
    lines = [
        "Telemetry summary for: %s" % telemetry_dir,
        "total events: %d" % len(records),
        "trainers seen: %d" % n_trainers,
        "event counts:",
    ]
    for ev, c in counts.most_common():
        lines.append("  %-16s %d" % (ev, c))
    text = "\n".join(lines) + "\n"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "summary.txt")
    with open(path, "w") as fh:
        fh.write(text)
    return path


def analyze(telemetry_dir: str, out_dir: Optional[str] = None) -> list[str]:
    records = load_events(telemetry_dir)
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(telemetry_dir)), "plots")
    if not records:
        print("no telemetry events found in %s" % telemetry_dir)
        return []
    saved: list[str] = []
    plotters = [
        plot_accuracy_loss,
        plot_staleness,
        plot_avail_composition,
        plot_utility_speed_scatter,
        plot_selection_frequency,
        plot_trainer_time_breakdown,
        plot_util_disparity,
    ]
    for fn in plotters:
        try:
            saved.extend(fn(records, out_dir))
        except Exception as e:  # one bad plot must not stop the rest
            print("  (plot %s failed: %s)" % (fn.__name__, e))
    saved.append(write_summary(records, out_dir, telemetry_dir))
    print("wrote %d artifact(s) to %s" % (len(saved), out_dir))
    for p in saved:
        print("  %s" % p)
    return saved


def compare(dirs: list[str], labels: Optional[list[str]], out_dir: str) -> list[str]:
    """Overlay accuracy / staleness across runs (one per selector)."""
    if labels is None or len(labels) != len(dirs):
        labels = [os.path.basename(os.path.dirname(os.path.abspath(d))) or d for d in dirs]
    acc_series = {}
    stale_series_vals = {}
    for label, d in zip(labels, dirs):
        recs = load_events(d)
        ev = by_event(recs, EVENT_AGG_EVAL)
        ev.sort(key=lambda r: r.get("round", 0))
        rs = [r.get("round") for r in ev if r.get("test-accuracy") is not None]
        av = [r.get("test-accuracy") for r in ev if r.get("test-accuracy") is not None]
        if rs:
            acc_series[label] = (rs, av)
        stale = [s for r in by_event(recs, EVENT_AGG_ROUND) for s in (r.get("staleness") or [])]
        if stale:
            stale_series_vals[label] = stale
    saved = []
    p = ph.line_plot(
        acc_series, "round", "accuracy",
        "Accuracy comparison across runs", out_dir, "compare_accuracy.png",
    )
    if p:
        saved.append(p)
    # overlay staleness CDFs by plotting each as its own line
    if stale_series_vals:
        import numpy as np

        series = {}
        for label, vals in stale_series_vals.items():
            arr = np.sort(np.asarray(vals, dtype=float))
            y = np.arange(1, len(arr) + 1) / len(arr)
            series[label] = (arr, y)
        p = ph.line_plot(
            series, "staleness (rounds)", "CDF",
            "Staleness CDF comparison across runs", out_dir, "compare_staleness_cdf.png",
        )
        if p:
            saved.append(p)
    print("wrote %d comparison artifact(s) to %s" % (len(saved), out_dir))
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("telemetry_dir", nargs="?", help="run telemetry directory")
    parser.add_argument("--out", help="output plots directory")
    parser.add_argument("--compare", nargs="+", help="telemetry dirs to compare")
    parser.add_argument("--labels", nargs="+", help="labels for --compare dirs")
    args = parser.parse_args()

    if args.compare:
        out = args.out or "compare_plots"
        compare(args.compare, args.labels, out)
        return
    if not args.telemetry_dir:
        parser.error("provide a telemetry_dir or --compare dirs...")
    analyze(args.telemetry_dir, args.out)


if __name__ == "__main__":
    main()
