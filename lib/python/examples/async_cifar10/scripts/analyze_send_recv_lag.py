"""
Post-processing script: parse SEND_RECV_LAG entries from an aggregator log and
report per-trainer lag statistics. Raises a warning if any trainer's median lag
exceeds a configurable threshold.

Usage:
    python analyze_send_recv_lag.py <aggregator_log> [--warn-threshold-s 5.0]
"""

import argparse
import re
import sys
from collections import defaultdict
import statistics


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
    print(f"\n{'='*60}")
    print(f"Global  n={len(all_lags):4d}  "
          f"min={min(all_lags):.2f}s  "
          f"median={statistics.median(all_lags):.2f}s  "
          f"p95={sorted(all_lags)[int(len(all_lags)*0.95)]:.2f}s  "
          f"max={max(all_lags):.2f}s")
    print(f"{'='*60}")

    warnings = []
    for end_id, vals in sorted(lags.items()):
        med = statistics.median(vals)
        p95 = sorted(vals)[int(len(vals) * 0.95)] if len(vals) >= 20 else max(vals)
        short_id = end_id[-8:]
        print(f"  trainer ...{short_id}  n={len(vals):4d}  "
              f"min={min(vals):.2f}s  median={med:.2f}s  "
              f"p95={p95:.2f}s  max={max(vals):.2f}s")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", help="Path to aggregator log file")
    parser.add_argument(
        "--warn-threshold-s", type=float, default=5.0,
        help="Warn if any trainer's median lag exceeds this (seconds)"
    )
    args = parser.parse_args()
    lags = parse_lags(args.log)
    report(lags, args.warn_threshold_s)


if __name__ == "__main__":
    main()
