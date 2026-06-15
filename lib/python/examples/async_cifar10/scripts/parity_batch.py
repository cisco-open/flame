#!/usr/bin/env python
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Per-baseline real/sim parity batch.

Discovers, for each requested baseline, the latest real + sim run pair whose
timestamp is at/after ``--since``, runs the parity comparator, prints the
stage-grouped report, and writes a date-time-stamped JSON per baseline.

Run dirs are named  run_<YYYYMMDD>_<HHMMSS>_dbg_<baseline>_..._stream_{real,sim}
(smoke dirs `_dbg_smoke_<baseline>_` are excluded automatically).

Usage:
    cd lib/python/examples/async_cifar10
    python scripts/parity_batch.py \
        --since 20260608_230000 \
        --baselines felix oort refl feddance \
        --agg-goal 10 --budget-s 12600
    # -> experiments/parity_<baseline>_<now>.json  (one per baseline)
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from parity.checks import load_run_dir, run_all_parity, first_divergence  # noqa: E402
from parity.report import print_report, write_json  # noqa: E402

_TS_RE = re.compile(r"run_(\d{8})_(\d{6})_")


def _stamp14(s: str) -> str:
    """Normalize a --since string to 14 digits YYYYMMDDHHMMSS (zero-padded)."""
    digits = re.sub(r"\D", "", s)
    return (digits + "0" * 14)[:14]


def _dir_stamp14(d: str):
    m = _TS_RE.search(os.path.basename(d))
    return (m.group(1) + m.group(2)) if m else None


def _latest_after(experiments_dir: str, baseline: str, mode: str, since14: str):
    """Latest run dir for (baseline, mode) with timestamp >= since14, or None.

    Matches `_dbg_<baseline>_` literally so `_dbg_smoke_<baseline>_` is excluded.
    """
    pat = os.path.join(experiments_dir, f"run_*_dbg_{baseline}_*stream_{mode}")
    hits = []
    for d in glob.glob(pat):
        s = _dir_stamp14(d)
        if s and s >= since14:
            hits.append((s, d))
    hits.sort()
    return hits[-1][1] if hits else None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", required=True,
                    help="earliest run stamp, e.g. 20260608_230000 (only later runs)")
    ap.add_argument("--baselines", nargs="+", required=True)
    ap.add_argument("--experiments-dir", default="experiments")
    ap.add_argument("--agg-goal", type=int, default=10)
    ap.add_argument("--budget-s", type=float, default=None,
                    help="= the run's --runtime-s (enables K5/K9)")
    ap.add_argument("--out-dir", default=None, help="default: --experiments-dir")
    ap.add_argument("--lenient", action="store_true",
                    help="demote DIST fails to warnings")
    args = ap.parse_args()

    since14 = _stamp14(args.since)
    out_dir = args.out_dir or args.experiments_dir
    os.makedirs(out_dir, exist_ok=True)
    nowstamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {}
    for b in args.baselines:
        real = _latest_after(args.experiments_dir, b, "real", since14)
        sim = _latest_after(args.experiments_dir, b, "sim", since14)
        if not real or not sim:
            print(f"[{b}] SKIP: real={os.path.basename(real) if real else None} "
                  f"sim={os.path.basename(sim) if sim else None} "
                  f"(need both at/after --since {args.since})")
            summary[b] = None
            continue
        print(f"\n{'#' * 72}\n# {b}\n#   real = {os.path.basename(real)}"
              f"\n#   sim  = {os.path.basename(sim)}\n{'#' * 72}")
        real_agg, real_tr = load_run_dir(real)
        sim_agg, sim_tr = load_run_dir(sim)
        res = run_all_parity(real_agg, sim_agg, real_tr, sim_tr,
                             agg_goal=args.agg_goal, budget_s=args.budget_s)
        fd = first_divergence(real_agg, sim_agg)
        fd["ok"] = True
        fd["tier"] = "DIAG"
        res["first_divergence_summary"] = fd
        passed = print_report(res, lenient=args.lenient,
                              real_label=os.path.basename(real),
                              sim_label=os.path.basename(sim))
        out = os.path.join(out_dir, f"parity_{b}_{nowstamp}.json")
        write_json(res, out, extra={"baseline": b, "real_dir": real,
                                    "sim_dir": sim, "agg_goal": args.agg_goal})
        summary[b] = {"passed": passed, "json": out}

    print(f"\n{'=' * 72}\n  PARITY BATCH SUMMARY (since {args.since})\n{'=' * 72}")
    for b in args.baselines:
        s = summary.get(b)
        if s is None:
            print(f"  {b:<10} SKIP (missing real/sim pair)")
        else:
            print(f"  {b:<10} {'PASS' if s['passed'] else 'FAIL'}  ->  {s['json']}")
    print(f"{'=' * 72}\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
