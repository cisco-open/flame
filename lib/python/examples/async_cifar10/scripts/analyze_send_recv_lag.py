"""
Post-processing script: parse SEND_RECV_LAG entries from an aggregator log and
report per-trainer lag statistics. Raises a warning if any trainer's median lag
exceeds a configurable threshold. Optionally emits a CDF plot.

Usage:
    python analyze_send_recv_lag.py <aggregator_log> [--warn-threshold-s 5.0] [--plot]
"""

import argparse
import os
import re
import sys
from collections import defaultdict
import statistics

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

LAG_RE = re.compile(
    r"\[SEND_RECV_LAG\] end=(\S+) version=(\d+) wall_lag_s=([0-9.]+)"
)


def parse_lags(log_path: str) -> dict[str, list[float]]:
    lags: dict[str, list[float]] = defaultdict(list)
    with open(log_path) as f:
        for line in f:
            m = LAG_RE.search(line)
            if m:
                end_id, _version, lag_s = m.group(1), m.group(2), float(m.group(3))
                lags[end_id].append(lag_s)
    return dict(lags)


def report(lags: dict[str, list[float]], warn_threshold_s: float) -> None:
    if not lags:
        print("No SEND_RECV_LAG entries found — was the run built with instrumentation?")
        return

    all_lags = [v for vals in lags.values() for v in vals]
    sv = sorted(all_lags)
    n = len(sv)
    print(f"\n{'='*60}")
    print(f"Global  n={n:4d}  "
          f"min={sv[0]:.2f}s  "
          f"p50={sv[n//2]:.2f}s  "
          f"p90={sv[int(n*0.90)]:.2f}s  "
          f"p99={sv[int(n*0.99)]:.2f}s  "
          f"max={sv[-1]:.2f}s")
    print(f"{'='*60}")

    warnings = []
    for end_id, vals in sorted(lags.items()):
        sv2 = sorted(vals)
        n2 = len(sv2)
        med = sv2[n2 // 2]
        p95 = sv2[int(n2 * 0.95)] if n2 >= 20 else sv2[-1]
        short_id = end_id[-8:]
        print(f"  trainer ...{short_id}  n={n2:4d}  "
              f"min={sv2[0]:.2f}s  median={med:.2f}s  "
              f"p95={p95:.2f}s  max={sv2[-1]:.2f}s")
        if med > warn_threshold_s:
            warnings.append((short_id, med))

    if warnings:
        print(f"\n{'!'*60}")
        print("WARNING: the following trainers have median lag above "
              f"{warn_threshold_s}s — possible MQTT broker backlog or slow "
              "model delivery:")
        for short_id, med in warnings:
            print(f"  trainer ...{short_id}  median={med:.2f}s")
        print("Consider increasing the inter-send sleep (0.2s → 0.5s) or "
              "checking broker load.")
        print(f"{'!'*60}")
        sys.exit(1)


def plot_cdf(lags: dict[str, list[float]], log_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from plotters._annot import annotate_percentiles, flush_percentile_table
    except ImportError:
        print("matplotlib not installed; skipping CDF plot.", file=sys.stderr)
        return

    all_lags = sorted(v for vals in lags.values() for v in vals)
    if not all_lags:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    n = len(all_lags)
    xs = all_lags
    ys = [(i + 1) / n for i in range(n)]
    color = "#4e79a7"
    ax.plot(xs, ys, color=color, lw=1.5, label="wall_lag_s (all trainers)")
    annotate_percentiles(ax, all_lags, color=color, label="wall_lag_s", below=True)
    flush_percentile_table(ax)
    ax.set_xlabel("wall_lag_s (agg→trainer→agg round trip, seconds)")
    ax.set_ylabel("CDF")
    ax.set_title(f"Send-recv lag CDF\n{os.path.basename(log_path)}")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    run_dir = os.path.dirname(os.path.dirname(log_path))
    out_dir = os.path.join(run_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "send_recv_lag_cdf.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", help="Path to aggregator log file")
    parser.add_argument(
        "--warn-threshold-s", type=float, default=5.0,
        help="Warn if any trainer's median lag exceeds this (seconds)"
    )
    parser.add_argument("--plot", action="store_true", help="Emit CDF plot PNG")
    args = parser.parse_args()
    lags = parse_lags(args.log)
    report(lags, args.warn_threshold_s)
    if args.plot:
        plot_cdf(lags, args.log)


if __name__ == "__main__":
    main()
