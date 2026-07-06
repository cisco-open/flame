"""Parity checker CLI.

Single command, single output:

    python -m scripts.parity.cli \\
        --real experiments/run_..._real \\
        --sim  experiments/run_..._sim \\
        --agg-goal 10 \\
        --json-out parity.json --plot-out parity.png \\
        [--strict] [--lenient] [--diagnostics] \\
        [--rounds-cap N] [--budget-s S]

    # Auto-discover and compare all baselines:
    python -m scripts.parity.cli --batch \\
        --experiments-dir experiments \\
        --baselines felix refl oort feddance \\
        [--agg-goal 10] [--json-out parity_<baseline>.json]

Exit 0 = passed; exit 1 = one or more enforced checks failed.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path


def _find_run_dirs(experiments_dir: str, baseline_tag: str) -> tuple:
    """Return (real_dir, sim_dir) by looking for latest real/sim pair matching baseline_tag."""
    pattern_real = os.path.join(experiments_dir, f"*{baseline_tag}*real*")
    pattern_sim = os.path.join(experiments_dir, f"*{baseline_tag}*sim*")
    reals = sorted(glob.glob(pattern_real))
    sims = sorted(glob.glob(pattern_sim))
    if not reals:
        raise FileNotFoundError(f"No real run dir matching *{baseline_tag}*real* in {experiments_dir}")
    if not sims:
        raise FileNotFoundError(f"No sim run dir matching *{baseline_tag}*sim* in {experiments_dir}")
    return reals[-1], sims[-1]


def _run_pair(real_dir: str, sim_dir: str,
              agg_goal: int, rounds_cap, budget_s,
              strict: bool, lenient: bool,
              json_out, plot_out,
              real_label: str = "", sim_label: str = "") -> bool:
    """Load, check, and report one real/sim pair.  Returns True if passed."""
    # Import here to avoid circular import issues when run as __main__
    import sys as _sys
    # Ensure scripts/ is on path so parity.checks can be imported
    _script_dir = str(Path(__file__).resolve().parents[1])
    if _script_dir not in _sys.path:
        _sys.path.insert(0, _script_dir)

    from parity.checks import (
        load_run_dir, run_all_parity, first_divergence,
    )
    from parity.ground_truth import load_ground_truth, resolve_trace_name
    from parity.report import print_report, write_json, write_plot

    real_label = real_label or os.path.basename(real_dir.rstrip("/"))
    sim_label = sim_label or os.path.basename(sim_dir.rstrip("/"))

    print(f"[parity] Loading real: {real_label}")
    real_agg, real_trainers = load_run_dir(real_dir)
    print(f"[parity] Loading sim:  {sim_label}")
    sim_agg, sim_trainers = load_run_dir(sim_dir)

    real_ground_truth = load_ground_truth(resolve_trace_name(real_dir))
    sim_ground_truth = load_ground_truth(resolve_trace_name(sim_dir))

    print(f"[parity]   real: {len(real_agg['agg_rounds'])} agg_round events, "
          f"{len(real_agg['selection_train'])} selection events, "
          f"{len(real_agg['agg_evals'])} eval events, "
          f"{len(real_trainers)} trainer files")
    print(f"[parity]   sim:  {len(sim_agg['agg_rounds'])} agg_round events, "
          f"{len(sim_agg['selection_train'])} selection events, "
          f"{len(sim_agg['agg_evals'])} eval events, "
          f"{len(sim_trainers)} trainer files")

    results = run_all_parity(
        real_agg, sim_agg, real_trainers, sim_trainers,
        agg_goal=agg_goal,
        rounds_cap=rounds_cap,
        budget_s=budget_s,
        real_ground_truth=real_ground_truth,
        sim_ground_truth=sim_ground_truth,
    )

    # Add first_divergence as a diagnostic summary entry (always ok — index=0 is expected for async)
    fd = first_divergence(real_agg, sim_agg)
    fd["ok"] = True
    fd["tier"] = "DIAG"
    results["first_divergence_summary"] = fd

    passed = print_report(results, strict=strict, lenient=lenient,
                          real_label=real_label, sim_label=sim_label)

    if json_out:
        from parity.report import write_json as _wj
        from parity.checks import verdict_summary
        _wj(results, json_out, extra={
            "real_dir": real_dir, "sim_dir": sim_dir, "agg_goal": agg_goal,
            "summary": verdict_summary(results, strict=strict, lenient=lenient),
        })
    if plot_out:
        write_plot(results, plot_out, real_label=real_label, sim_label=sim_label)

    return passed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ── single-pair mode ──
    parser.add_argument("--real", metavar="DIR",
                        help="Real run directory (contains telemetry/)")
    parser.add_argument("--sim", metavar="DIR",
                        help="Simulated run directory (contains telemetry/)")
    parser.add_argument("--agg-goal", type=int, default=0,
                        help="Aggregation goal (enables U4 agg_goal_cycles check)")
    parser.add_argument("--rounds-cap", type=int, default=None,
                        help="rounds cap from config (enables K9 truncation check)")
    parser.add_argument("--budget-s", type=float, default=None,
                        help="max_experiment_runtime_s / sim_wall_ceiling_s (enables K5/K9)")
    parser.add_argument("--strict", action="store_true",
                        help="Treat WARN as FAIL")
    parser.add_argument("--lenient", action="store_true",
                        help="Treat DIST FAIL as WARN")
    parser.add_argument("--json-out", metavar="PATH", default=None,
                        help="Write full results JSON to this path")
    parser.add_argument("--plot-out", metavar="PATH", default=None,
                        help="Write summary PNG to this path")
    parser.add_argument("--diagnostics", action="store_true",
                        help="(reserved) Run diagnostic single-run analysis scripts")
    # ── real-correctness validation ──
    parser.add_argument("--validate-real", metavar="DIR", default=None,
                        help="Validate a real run's own invariants (concurrency/"
                             "selection/aggregation) before using it as reference")
    # ── batch mode ──
    parser.add_argument("--batch", action="store_true",
                        help="Auto-discover sim/real pairs per baseline and run all")
    parser.add_argument("--experiments-dir", metavar="DIR", default="experiments",
                        help="Top-level experiments directory for --batch")
    parser.add_argument("--baselines", nargs="+", default=[],
                        metavar="TAG",
                        help="Baseline tags to discover (e.g. felix refl oort)")
    args = parser.parse_args()

    # ── ensure scripts/ is importable ────────────────────────────────────────
    _scripts_dir = str(Path(__file__).resolve().parents[1])
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)

    # ── real-correctness validation ───────────────────────────────────
    if args.validate_real:
        from parity.validate_real import validate_real
        sys.exit(0 if validate_real(args.validate_real) else 1)

    # ── batch mode ───────────────────────────────────────────────────────────
    if args.batch:
        if not args.baselines:
            parser.error("--batch requires --baselines tag1 tag2 ...")
        from parity.report import roll_up_table

        batch_results: dict = {}
        all_passed = True
        for tag in args.baselines:
            try:
                real_dir, sim_dir = _find_run_dirs(args.experiments_dir, tag)
            except FileNotFoundError as e:
                print(f"[parity] SKIP {tag}: {e}")
                continue
            json_out = (f"parity_{tag}.json" if args.json_out is None
                        else args.json_out.replace(".json", f"_{tag}.json"))
            plot_out = (None if args.plot_out is None
                        else args.plot_out.replace(".png", f"_{tag}.png"))
            print(f"\n{'#'*72}")
            print(f"# Baseline: {tag}")
            print(f"#   real={os.path.basename(real_dir)}")
            print(f"#   sim ={os.path.basename(sim_dir)}")
            print(f"{'#'*72}")
            try:
                ok = _run_pair(
                    real_dir, sim_dir,
                    agg_goal=args.agg_goal,
                    rounds_cap=args.rounds_cap,
                    budget_s=args.budget_s,
                    strict=args.strict,
                    lenient=args.lenient,
                    json_out=json_out,
                    plot_out=plot_out,
                )
                # Collect for roll-up (re-load results from JSON if written)
                import json as _json
                if json_out and os.path.exists(json_out):
                    with open(json_out) as f:
                        batch_results[tag] = {"results": _json.load(f)}
                else:
                    batch_results[tag] = {}
                if not ok:
                    all_passed = False
            except Exception as exc:
                print(f"[parity] ERROR {tag}: {exc}")
                all_passed = False

        roll_up_table(batch_results)
        sys.exit(0 if all_passed else 1)

    # ── single-pair mode ─────────────────────────────────────────────────────
    if not args.real or not args.sim:
        parser.error("Provide --real and --sim (or use --batch mode)")

    ok = _run_pair(
        args.real, args.sim,
        agg_goal=args.agg_goal,
        rounds_cap=args.rounds_cap,
        budget_s=args.budget_s,
        strict=args.strict,
        lenient=args.lenient,
        json_out=args.json_out,
        plot_out=args.plot_out,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
