#!/usr/bin/env python3
"""Parse [PLACEMENT] log lines from trainer logs and report CPU/GPU pinning balance.

Each trainer process logs at start-up:
    [PLACEMENT] trainer=<id> gpu=<CUDA_VISIBLE_DEVICES> cpu_cores=<list>

This script reads those lines from trainer log files in a run directory,
builds histograms of how many trainers landed on each core / GPU, and flags
imbalance (any core or GPU receiving 2× the median assignment count).

Outputs:
  stdout:  per-core and per-GPU assignment table with imbalance flags
  PNG:     <run_dir>/plots/pinning_balance.png  — two side-by-side histograms

Usage:
    python analyze_pinning.py <run_dir> [run_dir2 ...]
    python analyze_pinning.py --run label1=<dir1> --run label2=<dir2>
"""

import argparse
import glob
import os
import re
import sys
from collections import Counter, defaultdict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from plotters._annot import annotate_percentiles, flush_percentile_table

_RE_PLACEMENT = re.compile(
    r"\[PLACEMENT\]\s+trainer=(\S+)\s+gpu=(\S+)\s+cpu_cores=(\[.*?\])"
)


def _parse_core_list(s: str) -> list[int]:
    s = s.strip()
    if not s or s == "[]":
        return []
    try:
        import ast
        return list(ast.literal_eval(s))
    except Exception:
        nums = re.findall(r"\d+", s)
        return [int(x) for x in nums]


def load_placements(run_dir: str) -> list[dict]:
    """Return list of {trainer_id, gpu, cpu_cores} dicts from log files."""
    records = []
    seen_trainers: set[str] = set()

    # Check trainer log files — may be in run_dir directly or a logs/ subdir
    patterns = [
        os.path.join(run_dir, "*trainer*.log"),
        os.path.join(run_dir, "logs", "*trainer*.log"),
        os.path.join(run_dir, "*trainer*.out"),
        os.path.join(run_dir, "logs", "*trainer*.out"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))

    # Also scan any *.log in the run dir
    if not files:
        files = glob.glob(os.path.join(run_dir, "*.log"))

    for fpath in sorted(files):
        with open(fpath, errors="replace") as f:
            for line in f:
                m = _RE_PLACEMENT.search(line)
                if not m:
                    continue
                tid = m.group(1)
                gpu = m.group(2)
                cores = _parse_core_list(m.group(3))
                if tid in seen_trainers:
                    continue  # take only first occurrence (startup log)
                seen_trainers.add(tid)
                records.append({"trainer_id": tid, "gpu": gpu, "cpu_cores": cores})

    return records


def report(label: str, records: list[dict]) -> dict:
    """Print placement stats and return summary dicts for plotting."""
    n = len(records)
    print(f"\n{'='*70}")
    print(f"  {label}  —  {n} trainer placement records")
    print(f"{'='*70}")

    if not records:
        print("  No [PLACEMENT] lines found — run the experiment with cpu_pinning=on.")
        return {}

    # GPU distribution
    gpu_counter: Counter = Counter()
    for r in records:
        gpu_counter[r["gpu"]] += 1

    print(f"\n  GPU distribution ({len(gpu_counter)} GPUs):")
    gpu_vals = sorted(gpu_counter.values())
    gpu_median = gpu_vals[len(gpu_vals) // 2] if gpu_vals else 1
    for gpu_id, cnt in sorted(gpu_counter.items(), key=lambda x: x[0]):
        flag = " <<< IMBALANCE" if cnt > 2 * gpu_median else ""
        print(f"    GPU {gpu_id:>3}: {cnt:>4} trainers{flag}")

    # CPU core distribution
    core_counter: Counter = Counter()
    for r in records:
        for c in r["cpu_cores"]:
            core_counter[c] += 1

    print(f"\n  CPU core distribution ({len(core_counter)} cores used by {n} trainers):")
    if core_counter:
        core_vals = sorted(core_counter.values())
        core_median = core_vals[len(core_vals) // 2] if core_vals else 1
        imbalanced_cores = []
        # Print a compact table (at most 64 cores per row)
        cores_sorted = sorted(core_counter.keys())
        # Summary line
        print(f"    min_trainers_per_core={core_vals[0]}  "
              f"median={core_median}  "
              f"max={core_vals[-1]}")
        for c in cores_sorted:
            cnt = core_counter[c]
            if cnt > 2 * core_median:
                imbalanced_cores.append((c, cnt))
        if imbalanced_cores:
            print(f"\n  IMBALANCED CORES (> 2× median={core_median}):")
            for c, cnt in imbalanced_cores:
                print(f"    core {c:>3}: {cnt} trainers")
        else:
            print(f"    All {len(core_counter)} cores balanced (all within 2× median={core_median})")

    return {
        "gpu_counter": dict(gpu_counter),
        "core_counter": dict(core_counter),
        "n": n,
    }


def plot(label: str, summary: dict, run_dir: str) -> None:
    if not summary:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    gpu_counter = summary.get("gpu_counter", {})
    core_counter = summary.get("core_counter", {})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: GPU bar chart
    ax = axes[0]
    if gpu_counter:
        gpus = sorted(gpu_counter.keys())
        counts = [gpu_counter[g] for g in gpus]
        ax.bar(gpus, counts, color="#4e79a7", alpha=0.85)
        ax.axhline(sum(counts) / len(counts), color="red", ls="--", lw=1,
                   label=f"mean={sum(counts)/len(counts):.1f}")
        ax.set_xlabel("GPU (CUDA_VISIBLE_DEVICES)")
        ax.set_ylabel("Number of trainers")
        ax.set_title(f"GPU assignment balance — {label}")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No GPU data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"GPU assignment balance — {label}")

    # Right: CPU core histogram
    ax2 = axes[1]
    if core_counter:
        cores = sorted(core_counter.keys())
        counts = [core_counter[c] for c in cores]
        ax2.bar(cores, counts, color="#e15759", alpha=0.85, width=0.8)
        mean_c = sum(counts) / len(counts)
        ax2.axhline(mean_c, color="black", ls="--", lw=1, label=f"mean={mean_c:.1f}")
        ax2.set_xlabel("CPU core index")
        ax2.set_ylabel("Number of trainers pinned")
        ax2.set_title(f"CPU core pinning balance — {label}")
        ax2.legend(fontsize=8)

        # Annotate percentiles on the distribution of trainers-per-core
        # (this goes in a text box, not on the bar chart axes)
        all_counts_vals = list(core_counter.values())
        sv = sorted(float(v) for v in all_counts_vals)
        def _pct(sv, p):
            if not sv:
                return float("nan")
            idx = max(0, min(len(sv)-1, int(len(sv)*p/100)))
            return sv[idx]
        p50, p90, p99 = _pct(sv, 50), _pct(sv, 90), _pct(sv, 99)
        ax2.text(0.02, 0.97,
                 f"trainers/core: P50={p50:.0f}  P90={p90:.0f}  P99={p99:.0f}",
                 transform=ax2.transAxes, fontsize=7, va="top",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
    else:
        ax2.text(0.5, 0.5, "No CPU core data", ha="center", va="center",
                 transform=ax2.transAxes)
        ax2.set_title(f"CPU core pinning balance — {label}")

    fig.tight_layout()
    out_dir = os.path.join(run_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"pinning_balance_{label}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="*", help="Run directories")
    parser.add_argument("--run", action="append", default=[], metavar="LABEL=DIR",
                        help="Labeled run (repeatable)")
    args = parser.parse_args()

    specs: list[tuple[str, str]] = []
    for d in args.run_dirs:
        specs.append((os.path.basename(d.rstrip("/")), d))
    for item in args.run:
        if "=" not in item:
            parser.error(f"--run expects label=path, got {item!r}")
        label, path = item.split("=", 1)
        specs.append((label, path))

    if not specs:
        parser.print_help()
        sys.exit(1)

    for label, run_dir in specs:
        records = load_placements(run_dir)
        summary = report(label, records)
        plot(label, summary, run_dir)


if __name__ == "__main__":
    main()
