#!/usr/bin/env python
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Cross-arm figures for the Felix streaming-misprioritization experiment.

Consumes the telemetry + offline-oracle CSVs of every arm under a runs root and
emits the claim figures (see docs/EXPERIMENT_felix_streaming.md):

  Claim 1  per-client true-utility trajectories (+ participation markers),
           true-utility heatmap (clients x round).
  Claim 2  selection disparity-from-oracle over rounds (mis-selection / regret),
           mean-disparity vs time-to-60% scatter.
  Claim 3  accuracy vs sim-time / round (all arms), time-to-60% bars,
           overlap-with-oracle, uniform-vs-staggered ablation.

Prereqs per run dir: telemetry/*.jsonl and (for the 4 practical baselines)
analysis/oracle_{utility,misselection}.csv produced by oracle_misselection.py.

    python felix_streaming_figures.py --runs-root <experiments_dir> \
        [--out <experiments_dir>/figures] [--target 0.60]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import plot_helpers as ph

TARGET_DEFAULT = 0.60
BASELINES = ("felix", "oort", "refl", "feddance")


# --- run discovery + parsing ------------------------------------------------

def parse_arm(run_dir: str):
    """(baseline, variant, cond) from a run dir name like
    run_<ts>_<baseline>[_oracle]_n50_..._stream_<unif|stag>_sim[_nodeK].
    variant in {base, oracle}. None if not ours."""
    name = os.path.basename(run_dir.rstrip("/"))
    cond = "stag" if "stag" in name else ("unif" if "unif" in name else "na")
    variant = "oracle" if "_oracle_" in name else "base"
    for b in BASELINES:
        if f"_{b}_" in name:
            return b, variant, cond
    return None, None, None


def discover_runs(runs_root: str):
    """arm_key 'baseline/variant/cond' -> latest run dir (telemetry present)."""
    runs = {}
    for d in sorted(glob.glob(os.path.join(runs_root, "run_*"))):
        if not os.path.isdir(os.path.join(d, "telemetry")):
            continue
        b, variant, cond = parse_arm(d)
        if not b:
            continue
        runs[f"{b}/{variant}/{cond}"] = d  # sorted() => newest timestamp wins
    return runs


def _iter_events(telemetry_dir: str, event: str):
    for path in glob.glob(os.path.join(telemetry_dir, "*.jsonl")):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("event") == event:
                    yield r


def load_eval_curve(run_dir):
    """(rounds, accs) sorted by round, plus round->vclock map from agg_round."""
    tdir = os.path.join(run_dir, "telemetry")
    by_round = {}
    for r in _iter_events(tdir, "agg_eval"):
        acc = r.get("test-accuracy", r.get("test_accuracy"))
        if acc is not None:
            by_round[int(r.get("round", 0))] = float(acc)
    vclock = {}
    for r in _iter_events(tdir, "agg_round"):
        v = r.get("vclock_now")
        if v is not None:
            vclock[int(r.get("round", 0))] = float(v)
    rounds = sorted(by_round)
    return rounds, [by_round[x] for x in rounds], vclock


def nearest_vclock(vclock, rnd):
    if not vclock:
        return None
    le = [x for x in vclock if x <= rnd]
    return vclock[max(le)] if le else vclock[min(vclock)]


def time_to_target(rounds, accs, vclock, target, stable_k=1):
    """First round (and its sim-time) where acc>=target for stable_k consecutive
    evals. Returns (round, sim_time) or (None, None)."""
    run = 0
    for i, a in enumerate(accs):
        run = run + 1 if a >= target else 0
        if run >= stable_k:
            r0 = rounds[i - stable_k + 1]
            return r0, nearest_vclock(vclock, r0)
    return None, None


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _f(row, key):
    v = row.get(key)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# --- figures ----------------------------------------------------------------

def fig_accuracy(runs, out, target):
    """Claim 3: accuracy vs sim-time (baseline vs its oracle) + the staleness tax."""
    for cond in ("unif", "stag"):
        s_time, ttt = {}, {}   # ttt[(baseline, variant)] = sim-time to target
        for b in BASELINES:
            for variant in ("base", "oracle"):
                rd = runs.get(f"{b}/{variant}/{cond}")
                if not rd:
                    continue
                rounds, accs, vclock = load_eval_curve(rd)
                if not rounds:
                    continue
                label = b if variant == "base" else f"{b}*"  # * = oracular
                s_time[label] = ([nearest_vclock(vclock, r) or r for r in rounds], accs)
                _, t0 = time_to_target(rounds, accs, vclock, target, stable_k=1)
                if t0 is not None:
                    ttt[(b, variant)] = t0
        ph.binned_line(s_time, "sim-time (s)", "test accuracy",
                       f"Accuracy vs sim-time, baseline vs oracle* ({cond})", out,
                       f"claim3_accuracy_vs_simtime_{cond}", target=target, nbins=120)
        if ttt:
            cats = [f"{b}/{v}" for (b, v) in ttt]
            ph.bar_plot(cats, [ttt[k] for k in ttt], "sim-time to target (s)",
                        f"Time-to-{int(target * 100)}%, baseline vs oracle ({cond})",
                        out, f"claim3_time_to_target_{cond}")
        # staleness tax per baseline = T_base - T_oracle (>0 => staleness cost time)
        tax_b, tax_v = [], []
        for b in BASELINES:
            tb, to = ttt.get((b, "base")), ttt.get((b, "oracle"))
            if tb is not None and to is not None:
                tax_b.append(b)
                tax_v.append(tb - to)
        if tax_b:
            ph.bar_plot(tax_b, tax_v, "time-to-target tax (s): base - oracle",
                        f"Per-baseline staleness tax ({cond})", out,
                        f"claim3_staleness_tax_{cond}")


def fig_disparity(runs, out):
    """Claim 2: mis-selection / regret vs round for the 4 practical baselines."""
    for cond in ("unif", "stag"):
        miss, regret = {}, {}
        for b in ("felix", "oort", "refl", "feddance"):
            rd = runs.get(f"{b}/base/{cond}")
            if not rd:
                continue
            rows = [r for r in load_csv(os.path.join(rd, "analysis",
                    "oracle_misselection.csv")) if r.get("task") == "train"]
            xs = [_f(r, "round") for r in rows]
            for store, key in ((miss, "misselection_rate"), (regret, "utility_regret")):
                pair = [(x, _f(r, key)) for x, r in zip(xs, rows)]
                pair = [(x, y) for x, y in pair if x is not None and y is not None]
                if pair:
                    store[b] = ([p[0] for p in pair], [p[1] for p in pair])
        ph.binned_line(miss, "round", "mis-selection rate (1 - top-K overlap)",
                       f"Selection disparity vs oracle ({cond})", out,
                       f"claim2_misselection_vs_round_{cond}", nbins=100)
        ph.binned_line(regret, "round", "utility regret",
                       f"Utility regret vs oracle ({cond})", out,
                       f"claim2_regret_vs_round_{cond}", nbins=100)


def fig_disparity_vs_ttt(runs, out, target):
    """Claim 2: mean mis-selection vs time-to-target scatter (baseline x cond)."""
    xs, ys, groups = [], [], []
    for cond in ("unif", "stag"):
        for b in ("felix", "oort", "refl", "feddance"):
            rd = runs.get(f"{b}/base/{cond}")
            if not rd:
                continue
            rounds, accs, vclock = load_eval_curve(rd)
            _, t0 = time_to_target(rounds, accs, vclock, target, stable_k=1)
            rows = [r for r in load_csv(os.path.join(rd, "analysis",
                    "oracle_misselection.csv")) if r.get("task") == "train"]
            vals = [v for v in (_f(r, "misselection_rate") for r in rows) if v is not None]
            if t0 is None or not vals:
                continue
            xs.append(sum(vals) / len(vals))
            ys.append(t0)
            groups.append(f"{b}_{cond}")
    if xs:
        ph.scatter_plot(xs, ys, groups, "mean mis-selection rate",
                        "sim-time to target (s)", "Disparity vs time-to-target",
                        out, "claim2_disparity_vs_ttt")


def fig_utility_trajectories(runs, out, n_clients=6):
    """Claim 1: true-utility trajectories + heatmap from the best-resolved arm."""
    rd = None
    for key in ("felix/base/unif", "oort/base/unif", "felix/base/stag", "oort/base/stag"):
        cand = runs.get(key)
        if cand and os.path.exists(os.path.join(cand, "analysis", "oracle_utility.csv")):
            rd = cand
            break
    if not rd:
        ph.no_data_plot("True-utility trajectories (no oracle_utility.csv)",
                        out, "claim1_utility_trajectories")
        return
    rows = load_csv(os.path.join(rd, "analysis", "oracle_utility.csv"))
    by_end = defaultdict(list)  # end_id -> [(round, true, selected)]
    for r in rows:
        rnd, tru = _f(r, "round"), _f(r, "true")
        if rnd is None or tru is None:
            continue
        by_end[r["end_id"]].append((rnd, tru, _f(r, "selected") or 0.0))

    def var(seq):
        ys = [t for _, t, _ in seq]
        m = sum(ys) / len(ys)
        return sum((y - m) ** 2 for y in ys) / len(ys)

    picks = sorted(by_end, key=lambda e: var(by_end[e]), reverse=True)[:n_clients]
    series = {}
    for e in picks:
        seq = sorted(by_end[e])
        series[e[:8]] = ([s[0] for s in seq], [s[1] for s in seq])
    ph.binned_line(series, "round", "true utility (oracle)",
                   "Claim 1: per-client true-utility trajectories", out,
                   "claim1_utility_trajectories", nbins=200)

    all_rounds = sorted({rr for seq in by_end.values() for rr, _, _ in seq})
    if all_rounds and picks:
        ridx = {r: i for i, r in enumerate(all_rounds)}
        mat = []
        for e in picks:
            row = [float("nan")] * len(all_rounds)
            for rr, t, _ in by_end[e]:
                row[ridx[rr]] = t
            mat.append(row)
        ph.heatmap(mat, "round", "client", "Claim 1: true-utility heatmap",
                   out, "claim1_utility_heatmap",
                   yticklabels=[e[:8] for e in picks], cbar_label="true utility")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", required=True,
                    help="experiments dir holding run_* subdirs")
    ap.add_argument("--out")
    ap.add_argument("--target", type=float, default=TARGET_DEFAULT)
    args = ap.parse_args()

    out = args.out or os.path.join(args.runs_root, "figures")
    os.makedirs(out, exist_ok=True)
    runs = discover_runs(args.runs_root)
    if not runs:
        raise SystemExit(f"no matching run_* dirs under {args.runs_root}")
    print("arms found:")
    for k, v in sorted(runs.items()):
        print(f"  {k:18s} {os.path.basename(v)}")

    fig_accuracy(runs, out, args.target)
    fig_disparity(runs, out)
    fig_disparity_vs_ttt(runs, out, args.target)
    fig_utility_trajectories(runs, out)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
