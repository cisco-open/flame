#!/usr/bin/env python3
"""Trainer state-count dynamics over rounds/time.

For each selection event we have:
  avail_composition  - {AVL_TRAIN, AVL_EVAL, UN_AVL, UNKNOWN} counts
  in_flight          - trainers with an outstanding task
  num_chosen         - trainers selected this round

From agg_round events:
  agg_goal / agg_goal_count - target vs actual committed updates

Plot A (per-round line chart):
  - AVL_TRAIN, AVL_EVAL, UN_AVL, in_flight over rounds
  - num_chosen overlaid as step-chart
  - secondary y: agg_goal vs agg_goal_count (from agg_round)

Plot B (stacked area: availability breakdown as fraction of n_trainers)

Plot C (per-round idle count = AVL_TRAIN - in_flight, aggregated P50/P90)

Supports multi-run overlay for cross-baseline comparison.

Usage:
    python analyze_dynamics_timeline.py <run_dir> [run_dir2 ...]
    python analyze_dynamics_timeline.py --run felix_sim=<dir> --run refl_real=<dir>
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

STATES = ["AVL_TRAIN", "AVL_EVAL", "UN_AVL", "UNKNOWN"]
_PALETTE = {
    "AVL_TRAIN": "#4e79a7",
    "AVL_EVAL":  "#76b7b2",
    "UN_AVL":    "#e15759",
    "UNKNOWN":   "#bab0ac",
    "in_flight": "#f28e2b",
    "idle":      "#59a14f",
    "chosen":    "#9467bd",
}


def load_agg_events(run_dir: str) -> list[dict]:
    events = []
    for fpath in glob.glob(os.path.join(run_dir, "telemetry", "aggregator_*.jsonl")):
        with open(fpath, errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


def load_trainer_events(run_dir: str) -> list[dict]:
    events = []
    for fpath in glob.glob(os.path.join(run_dir, "telemetry", "trainer_*.jsonl")):
        with open(fpath, errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    if e.get("event") in ("avail_change", "task_recv"):
                        events.append(e)
                except json.JSONDecodeError:
                    pass
    return events


def extract_per_round(agg_events: list[dict]) -> dict[int, dict]:
    """Build per-round dict from selection + agg_round events."""
    by_round: dict[int, dict] = defaultdict(dict)

    for ev in agg_events:
        r = ev.get("round")
        if r is None:
            continue
        et = ev.get("event")
        if et == "selection":
            comp = ev.get("avail_composition") or {}
            by_round[r]["avl_train"] = comp.get("AVL_TRAIN", 0)
            by_round[r]["avl_eval"] = comp.get("AVL_EVAL", 0)
            by_round[r]["un_avl"] = comp.get("UN_AVL", 0)
            by_round[r]["unknown"] = comp.get("UNKNOWN", 0)
            by_round[r]["in_flight"] = ev.get("in_flight", 0)
            by_round[r]["num_chosen"] = ev.get("num_chosen", 0)
            by_round[r]["ts_sel"] = ev.get("ts")
            by_round[r]["vclock"] = ev.get("vclock_now")
        elif et == "agg_round":
            by_round[r]["agg_goal"] = ev.get("agg_goal", 0)
            by_round[r]["agg_goal_count"] = ev.get("agg_goal_count", 0)
            by_round[r]["ts_agg"] = ev.get("ts")
            by_round[r].setdefault("vclock", ev.get("vclock_now"))

    return dict(by_round)


def report(label: str, by_round: dict[int, dict]) -> None:
    rounds = sorted(by_round)
    if not rounds:
        print(f"  {label}: no per-round data")
        return

    n_total = max(
        (by_round[r].get("avl_train", 0) + by_round[r].get("avl_eval", 0) +
         by_round[r].get("un_avl", 0) + by_round[r].get("unknown", 0))
        for r in rounds
    )

    print(f"\n{'='*72}")
    print(f"  {label}  -  {len(rounds)} rounds  n_trainers~={n_total}")
    print(f"{'='*72}")

    # Summary stats
    in_flight_vals = [by_round[r].get("in_flight", 0) for r in rounds]
    idle_vals = [
        max(0, by_round[r].get("avl_train", 0) - by_round[r].get("in_flight", 0))
        for r in rounds
    ]
    unavl_vals = [by_round[r].get("un_avl", 0) for r in rounds]
    chosen_vals = [by_round[r].get("num_chosen", 0) for r in rounds]

    def _pct(lst, p):
        sv = sorted(lst)
        if not sv:
            return 0
        return sv[max(0, min(len(sv)-1, int(len(sv)*p/100)))]

    def _med(lst):
        return _pct(lst, 50)

    print(f"  in_flight:  median={_med(in_flight_vals):.0f}  p90={_pct(in_flight_vals, 90):.0f}  max={max(in_flight_vals):.0f}")
    print(f"  idle:       median={_med(idle_vals):.0f}  p90={_pct(idle_vals, 90):.0f}")
    print(f"  unavail:    median={_med(unavl_vals):.0f}  p90={_pct(unavl_vals, 90):.0f}")
    print(f"  chosen:     median={_med(chosen_vals):.0f}  p90={_pct(chosen_vals, 90):.0f}  max={max(chosen_vals):.0f}")

    # Agg goal vs count
    goal_vals = [by_round[r].get("agg_goal", 0) for r in rounds if "agg_goal" in by_round[r]]
    count_vals = [by_round[r].get("agg_goal_count", 0) for r in rounds if "agg_goal_count" in by_round[r]]
    if goal_vals and count_vals:
        undercount = sum(1 for g, c in zip(goal_vals, count_vals) if c < g)
        print(f"  agg_goal:   median={_med(goal_vals):.0f}  "
              f"rounds where count<goal: {undercount}/{len(goal_vals)}")


def plot(specs: list[tuple[str, str, dict[int, dict]]], out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots.", file=sys.stderr)
        return

    os.makedirs(out_dir, exist_ok=True)

    nruns = len(specs)
    ls_cycle = ["-", "--", "-.", ":"]
    alpha_base = 0.85 if nruns == 1 else 0.65

    # -- Plot A: AVL_TRAIN, UN_AVL, in_flight, chosen per round -----------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=False)

    for ri, (label, run_dir, by_round) in enumerate(specs):
        ls = ls_cycle[ri % len(ls_cycle)]
        rounds = sorted(by_round)
        if not rounds:
            continue

        avl  = [by_round[r].get("avl_train", 0) for r in rounds]
        unavl = [by_round[r].get("un_avl", 0) for r in rounds]
        inf  = [by_round[r].get("in_flight", 0) for r in rounds]
        idle = [max(0, a - f) for a, f in zip(avl, inf)]
        chosen = [by_round[r].get("num_chosen", 0) for r in rounds]

        ax1.plot(rounds, avl, ls=ls, color=_PALETTE["AVL_TRAIN"], alpha=alpha_base,
                 lw=1.2, label=f"{label} AVL_TRAIN")
        ax1.plot(rounds, unavl, ls=ls, color=_PALETTE["UN_AVL"], alpha=alpha_base,
                 lw=1.0, label=f"{label} UN_AVL")
        ax1.plot(rounds, inf, ls=ls, color=_PALETTE["in_flight"], alpha=alpha_base,
                 lw=1.2, label=f"{label} in_flight")
        ax1.plot(rounds, idle, ls=ls, color=_PALETTE["idle"], alpha=alpha_base * 0.7,
                 lw=0.8, label=f"{label} idle")
        ax1.step(rounds, chosen, where="post", ls=ls, color=_PALETTE["chosen"],
                 alpha=alpha_base * 0.8, lw=1.0, label=f"{label} num_chosen")

    ax1.set_xlabel("Round")
    ax1.set_ylabel("Count (trainers)")
    ax1.set_title("Trainer state counts per round")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(alpha=0.25)

    # -- Plot A bottom: agg_goal vs agg_goal_count -------------------------
    for ri, (label, run_dir, by_round) in enumerate(specs):
        ls = ls_cycle[ri % len(ls_cycle)]
        rounds = sorted(by_round)
        rg = [r for r in rounds if "agg_goal" in by_round[r]]
        if not rg:
            continue
        goals = [by_round[r]["agg_goal"] for r in rg]
        counts = [by_round[r].get("agg_goal_count", 0) for r in rg]
        ax2.step(rg, goals, where="post", ls=ls, color="#4e79a7", alpha=0.7,
                 lw=1.2, label=f"{label} agg_goal")
        ax2.plot(rg, counts, ls=ls, color="#e15759", alpha=0.7,
                 lw=1.0, marker=".", markersize=2, label=f"{label} agg_goal_count")

    ax2.set_xlabel("Round")
    ax2.set_ylabel("Count")
    ax2.set_title("agg_goal vs agg_goal_count per round")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    out = os.path.join(out_dir, "dynamics_timeline.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot A: {out}")

    # -- Plot B: stacked area - availability breakdown ---------------------
    if nruns == 1:
        label, run_dir, by_round = specs[0]
        rounds = sorted(by_round)
        if rounds:
            fig2, ax = plt.subplots(figsize=(14, 5))
            avl   = [by_round[r].get("avl_train", 0) for r in rounds]
            aeval = [by_round[r].get("avl_eval", 0) for r in rounds]
            unavl = [by_round[r].get("un_avl", 0) for r in rounds]
            unk   = [by_round[r].get("unknown", 0) for r in rounds]
            ax.stackplot(
                rounds,
                [avl, aeval, unavl, unk],
                labels=["AVL_TRAIN", "AVL_EVAL", "UN_AVL", "UNKNOWN"],
                colors=[_PALETTE["AVL_TRAIN"], _PALETTE["AVL_EVAL"],
                        _PALETTE["UN_AVL"], _PALETTE["UNKNOWN"]],
                alpha=0.75,
            )
            ax.set_xlabel("Round")
            ax.set_ylabel("Trainers")
            ax.set_title(f"Availability breakdown (stacked) - {label}")
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(alpha=0.25)
            out2 = os.path.join(out_dir, "dynamics_avail_stack.png")
            fig2.tight_layout()
            fig2.savefig(out2, dpi=120, bbox_inches="tight")
            plt.close(fig2)
            print(f"  Plot B: {out2}")

    # -- Plot C: idle count CDF per run ------------------------------------
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    from plotters._annot import annotate_percentiles, flush_percentile_table
    for ri, (label, run_dir, by_round) in enumerate(specs):
        rounds = sorted(by_round)
        idle_vals = [
            max(0, by_round[r].get("avl_train", 0) - by_round[r].get("in_flight", 0))
            for r in rounds
        ]
        if not idle_vals:
            continue
        sv = sorted(idle_vals)
        n = len(sv)
        ys = [(i + 1) / n for i in range(n)]
        # pick a color per run
        run_colors = ["#4e79a7", "#e15759", "#59a14f", "#f28e2b"]
        c = run_colors[ri % len(run_colors)]
        ls = ls_cycle[ri % len(ls_cycle)]
        ax3.plot(sv, ys, ls=ls, color=c, lw=1.5, label=label)
        annotate_percentiles(ax3, idle_vals, color=c, label=label, below=True)
    ax3.set_xlabel("Idle trainers (AVL_TRAIN - in_flight)")
    ax3.set_ylabel("CDF")
    ax3.set_title("Idle trainer count CDF\n(P50/P90/P99 in table below)")
    ax3.set_ylim(0, 1.05)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    flush_percentile_table(ax3)
    out3 = os.path.join(out_dir, "dynamics_idle_cdf.png")
    fig3.tight_layout()
    fig3.savefig(out3, dpi=120, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Plot C: {out3}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="*")
    parser.add_argument("--run", action="append", default=[], metavar="LABEL=DIR")
    parser.add_argument("--out-dir", metavar="DIR", default=None)
    args = parser.parse_args()

    specs_raw: list[tuple[str, str]] = []
    for d in args.run_dirs:
        specs_raw.append((os.path.basename(d.rstrip("/")), d))
    for item in args.run:
        if "=" not in item:
            parser.error(f"--run expects label=path, got {item!r}")
        lbl, path = item.split("=", 1)
        specs_raw.append((lbl, path))

    if not specs_raw:
        parser.print_help()
        sys.exit(1)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(specs_raw[0][1], "plots")

    specs_full = []
    for label, run_dir in specs_raw:
        print(f"Loading {run_dir} ({label})...")
        agg = load_agg_events(run_dir)
        print(f"  {len(agg)} agg events")
        by_round = extract_per_round(agg)
        report(label, by_round)
        specs_full.append((label, run_dir, by_round))

    plot(specs_full, out_dir)


if __name__ == "__main__":
    main()
