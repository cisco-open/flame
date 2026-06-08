#!/usr/bin/env python3
"""Analyze per-phase trainer timing from trainer_round telemetry.

Reads telemetry/trainer_*.jsonl from one or more run directories and produces:
  Plot A: stacked bar — CPU-bound phases averaged per round  (+ P50/P90/P99)
  Plot B: stacked bar — GPU-bound phases averaged per round  (+ P50/P90/P99)
  Plot C: per-trainer heatmap of total round time (localize spikes)
  Plot D: CDF per CPU phase across all trainer-round observations
  Plot E: CDF per GPU phase across all trainer-round observations

Output PNGs land in <run_dir>/plots/.

Usage:
    python analyze_trainer_phases.py <run_dir> [run_dir2 ...]
    python analyze_trainer_phases.py --compare run_a run_b --labels oort refl
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

# Allow importing from the plotters package whether the script is run directly
# or via python -m.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from plotters._annot import annotate_percentiles, flush_percentile_table

# ── phase taxonomy ──────────────────────────────────────────────────────────
CPU_PHASES = [
    "mqtt_fetch_s",
    "weights_to_ram_s",
    "pre_train_s",
    "post_train_s",
    "post_cpu_s",
    "mqtt_send_s",
]
GPU_PHASES = [
    "weights_to_gpu_s",
    "gpu_compute_s",
    "weights_from_gpu_s",
]
WALL_PHASES = ["sleep_s"]
ALL_PHASES = CPU_PHASES + GPU_PHASES + WALL_PHASES

CPU_COLORS = ["#4e79a7", "#76b7b2", "#59a14f", "#f28e2b", "#9c755f", "#bab0ac"]
GPU_COLORS = ["#e15759", "#ff9da7", "#edc948"]


def load_events(run_dir: str) -> list[dict]:
    pattern = os.path.join(run_dir, "telemetry", "trainer_*.jsonl")
    files = glob.glob(pattern)
    if not files:
        print(f"  WARNING: no trainer telemetry found in {run_dir}/telemetry/", file=sys.stderr)
        return []
    events = []
    for fpath in files:
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("event") == "trainer_round":
                    events.append(obj)
    return events


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = max(0, min(len(sorted_vals) - 1, int(len(sorted_vals) * p / 100)))
    return sorted_vals[idx]


def aggregate_phases(events: list[dict]) -> dict[int, dict[str, dict]]:
    """Return {round -> {phase -> {"mean", "p50", "p90", "p99", "max"}}}."""
    by_round: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for ev in events:
        r = ev.get("round", 0)
        for ph in ALL_PHASES:
            v = ev.get(ph)
            if v is not None:
                by_round[r][ph].append(float(v))

    result = {}
    for r, phases in sorted(by_round.items()):
        result[r] = {}
        for ph, vals in phases.items():
            sv = sorted(vals)
            result[r][ph] = {
                "mean": sum(sv) / len(sv),
                "p50": _pct(sv, 50),
                "p90": _pct(sv, 90),
                "p99": _pct(sv, 99),
                "max": sv[-1],
                "vals": sv,
            }
    return result


def collect_phase_vals(events: list[dict]) -> dict[str, list[float]]:
    """Flat list of all observed values per phase (for CDFs)."""
    out: dict[str, list[float]] = defaultdict(list)
    for ev in events:
        for ph in ALL_PHASES:
            v = ev.get(ph)
            if v is not None:
                out[ph].append(float(v))
    return dict(out)


def _heatmap_data(events: list[dict]) -> tuple[list, list, list[list]]:
    total_s_by: dict[str, dict[int, float]] = defaultdict(dict)
    for ev in events:
        r = ev.get("round", 0)
        tid = str(ev.get("end_id", ev.get("trainer_id", "?")))
        total = sum(float(ev.get(ph, 0.0) or 0.0) for ph in ALL_PHASES)
        if total == 0.0:
            total = float(ev.get("real_gpu_time_s", 0.0) or 0.0)
        total_s_by[tid][r] = total
    trainers = sorted(total_s_by.keys())
    rounds = sorted({r for t in total_s_by.values() for r in t})
    matrix = [
        [total_s_by[t].get(r, float("nan")) for r in rounds]
        for t in trainers
    ]
    return rounds, trainers, matrix


def _cdf_xy(vals):
    sv = sorted(v for v in vals if v is not None)
    n = len(sv)
    if not n:
        return [], []
    return sv, [(i + 1) / n for i in range(n)]


def plot_run(run_dir: str, label: str, agg: dict[int, dict[str, dict]],
             events: list[dict], plot_heatmap: bool) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not installed; skipping plots.", file=sys.stderr)
        return

    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    rounds = sorted(agg.keys())
    if not rounds:
        print(f"  No rounds to plot for {label}.", file=sys.stderr)
        return

    phase_vals = collect_phase_vals(events)

    def _series(phase_list):
        return {
            ph: np.array([agg[r].get(ph, {}).get("mean", 0.0) for r in rounds])
            for ph in phase_list
        }

    # ── Plot A: CPU stacked bar + total P50/P90 band ──────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    cpu_s = _series(CPU_PHASES)
    bottom = np.zeros(len(rounds))
    for ph, color in zip(CPU_PHASES, CPU_COLORS):
        vals = cpu_s[ph]
        if vals.sum() > 0:
            ax.bar(rounds, vals, bottom=bottom, label=ph, color=color, width=0.8)
            bottom += vals
    # P90 line of total CPU time across all rounds
    all_cpu_totals = [
        sum(float(ev.get(ph, 0.0) or 0.0) for ph in CPU_PHASES) for ev in events
    ]
    if all_cpu_totals:
        p90_cpu = _pct(sorted(all_cpu_totals), 90)
        ax.axhline(p90_cpu, color="black", ls="--", lw=1, label=f"P90 total CPU={p90_cpu:.2f}s")
        p50_cpu = _pct(sorted(all_cpu_totals), 50)
        ax.axhline(p50_cpu, color="gray", ls=":", lw=1, label=f"P50 total CPU={p50_cpu:.2f}s")
    ax.set_xlabel("Round")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"CPU phases per round (mean) — {label}")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xlim(left=min(rounds) - 0.5)
    fig.tight_layout()
    out = os.path.join(run_dir, "plots", f"trainer_phases_cpu_{label}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Wrote {out}")

    # ── Plot B: GPU stacked bar ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    gpu_s = _series(GPU_PHASES)
    bottom = np.zeros(len(rounds))
    for ph, color in zip(GPU_PHASES, GPU_COLORS):
        vals = gpu_s[ph]
        if vals.sum() > 0:
            ax.bar(rounds, vals, bottom=bottom, label=ph, color=color, width=0.8)
            bottom += vals
    all_gpu_totals = [
        sum(float(ev.get(ph, 0.0) or 0.0) for ph in GPU_PHASES) for ev in events
    ]
    if all_gpu_totals:
        p90_gpu = _pct(sorted(all_gpu_totals), 90)
        ax.axhline(p90_gpu, color="black", ls="--", lw=1, label=f"P90 total GPU={p90_gpu:.2f}s")
        p50_gpu = _pct(sorted(all_gpu_totals), 50)
        ax.axhline(p50_gpu, color="gray", ls=":", lw=1, label=f"P50 total GPU={p50_gpu:.2f}s")
    ax.set_xlabel("Round")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"GPU phases per round (mean) — {label}")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xlim(left=min(rounds) - 0.5)
    fig.tight_layout()
    out = os.path.join(run_dir, "plots", f"trainer_phases_gpu_{label}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Wrote {out}")

    # ── Plot C: per-trainer heatmap ───────────────────────────────────────
    if plot_heatmap and events:
        hm_rounds, hm_trainers, matrix = _heatmap_data(events)
        if hm_rounds and hm_trainers:
            mat = __import__("numpy").array(matrix, dtype=float)
            fig, ax = plt.subplots(figsize=(max(8, len(hm_rounds) * 0.15),
                                             max(4, len(hm_trainers) * 0.08)))
            im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", interpolation="nearest")
            ax.set_xlabel("Round")
            ax.set_ylabel("Trainer")
            ax.set_title(f"Total round time heatmap — {label}")
            step = max(1, len(hm_rounds) // 20)
            ax.set_xticks(range(0, len(hm_rounds), step))
            ax.set_xticklabels([str(hm_rounds[i]) for i in range(0, len(hm_rounds), step)], fontsize=7)
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, label="Total round time (s)")
            fig.tight_layout()
            out = os.path.join(run_dir, "plots", f"trainer_phases_heatmap_{label}.png")
            fig.savefig(out, dpi=120)
            plt.close(fig)
            print(f"  Wrote {out}")

    # ── Plot D: CDF per CPU phase ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for ph, color in zip(CPU_PHASES, CPU_COLORS):
        vals = phase_vals.get(ph, [])
        if not vals:
            continue
        xs, ys = _cdf_xy(vals)
        ax.plot(xs, ys, color=color, label=ph, lw=1.5)
        annotate_percentiles(ax, vals, color=color, label=ph, below=True)
    ax.set_xlabel("Phase duration (s)")
    ax.set_ylabel("CDF")
    ax.set_title(f"CPU phase CDF — {label}")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_ylim(0, 1.05)
    flush_percentile_table(ax)
    fig.tight_layout()
    out = os.path.join(run_dir, "plots", f"trainer_phases_cdf_cpu_{label}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")

    # ── Plot E: CDF per GPU phase ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for ph, color in zip(GPU_PHASES, GPU_COLORS):
        vals = phase_vals.get(ph, [])
        if not vals:
            continue
        xs, ys = _cdf_xy(vals)
        ax.plot(xs, ys, color=color, label=ph, lw=1.5)
        annotate_percentiles(ax, vals, color=color, label=ph, below=True)
    ax.set_xlabel("Phase duration (s)")
    ax.set_ylabel("CDF")
    ax.set_title(f"GPU phase CDF — {label}")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_ylim(0, 1.05)
    flush_percentile_table(ax)
    fig.tight_layout()
    out = os.path.join(run_dir, "plots", f"trainer_phases_cdf_gpu_{label}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")


def print_summary(label: str, agg: dict[int, dict[str, dict]]) -> None:
    print(f"\n{'='*72}")
    print(f"  {label}  —  {len(agg)} rounds")
    print(f"  {'Phase':<22} {'mean(s)':>9} {'p50':>7} {'p90':>7} {'p99':>7} {'max':>7}")
    print(f"  {'-'*22} {'-'*9} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    acc: dict[str, list[float]] = defaultdict(list)
    for r_data in agg.values():
        for ph, stats in r_data.items():
            acc[ph].extend(stats["vals"])
    for ph in ALL_PHASES:
        vals = acc.get(ph, [])
        if not vals:
            continue
        sv = sorted(vals)
        mn = sum(sv) / len(sv)
        p50 = _pct(sv, 50)
        p90 = _pct(sv, 90)
        p99 = _pct(sv, 99)
        mx = sv[-1]
        group = "CPU " if ph in CPU_PHASES else ("GPU " if ph in GPU_PHASES else "wall")
        print(f"  [{group}] {ph:<18} {mn:>9.3f} {p50:>7.3f} {p90:>7.3f} {p99:>7.3f} {mx:>7.3f}")
    print(f"{'='*72}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze per-phase trainer timing from trainer_round telemetry"
    )
    parser.add_argument("run_dirs", nargs="*", help="Run directories to analyze")
    parser.add_argument("--compare", nargs="+", metavar="RUN_DIR",
                        help="Compare two or more run directories side by side")
    parser.add_argument("--labels", nargs="+", metavar="LABEL",
                        help="Short labels for the compared runs (same order as --compare)")
    parser.add_argument("--heatmap", action="store_true", default=True)
    parser.add_argument("--no-heatmap", dest="heatmap", action="store_false")
    args = parser.parse_args()

    runs = args.compare or args.run_dirs
    if not runs:
        parser.print_help()
        sys.exit(1)

    labels = args.labels or [os.path.basename(r.rstrip("/")) for r in runs]
    if len(labels) < len(runs):
        labels += [os.path.basename(r.rstrip("/")) for r in runs[len(labels):]]

    for run_dir, label in zip(runs, labels):
        print(f"\nLoading {run_dir} ({label})…")
        events = load_events(run_dir)
        print(f"  {len(events)} trainer_round events")
        if not events:
            continue
        agg = aggregate_phases(events)
        print_summary(label, agg)
        plot_run(run_dir, label, agg, events, args.heatmap)


if __name__ == "__main__":
    main()
