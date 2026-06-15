"""Canonical real/sim parity checks — re-export shim.

All logic has moved to ``scripts/parity/checks.py``.  This file re-exports
every public name so that existing callers (pytest suite, compare_parity.py,
etc.) continue to work without modification.

See ``parity/checks.py`` for the full §3 battery including the new §3.H
clock/throughput checks (K1–K10) that catch the 410-vs-673 rounds regression.
"""

from __future__ import annotations

import sys
import os

# Make the parity sub-package importable when this file's directory is on sys.path
# (standard when running scripts directly or when tests add scripts/ to sys.path).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from parity.checks import (  # noqa: F401, E402
    # helpers
    short,
    jaccard,
    mean_std,
    ks_stat,
    spearman_rho,
    # loaders
    load_agg_jsonl,
    load_trainer_jsonl_dir,
    load_run_dir,
    # selection constants
    DETERMINISTIC_SELECTORS,
    _selector_name,
    # §3.B selection
    selection_parity,
    # §3.D updates
    aggregation_sequence_parity,
    staleness_parity,
    commit_sequence,
    first_divergence,
    agg_goal_cycles_ok,
    inter_arrival_order_parity,
    # §3.E processing
    participation_parity,
    trainer_speed_parity,
    # §3.F utility
    utility_parity,
    # §3.G convergence
    convergence_parity,
    # §3.C sim invariants
    sim_send_ts_ok,
    gpu_budget_ok,
    trainer_phase_parity,
    # Stage 0 / 1 / 2 / 4 / 8 additions (causal ladder)
    field_coverage,
    modeled_compute_advance,
    overhead_residual,
    avail_timebase_parity,
    duty_cycle_parity,
    training_budget_parity,
    trainer_phase_split,
    convergence_loss_parity,
    avail_composition_parity,
    eligibility_parity,
    eligible_speed_composition_parity,
    selection_detail_parity,
    # registry / verdict helpers
    CHECK_META,
    check_stage,
    check_role,
    # §3.H clock
    vclock_telemetry_present,
    sim_commit_order_monotone,
    sim_rate_ok,
    failsafe_ok,
    throughput_parity,
    per_round_advance_parity,
    overlap_factor,
    total_commits_parity,
    terminal_state_parity,
    budget_not_cap,
    # overall
    run_all_parity,
    overall_verdict,
)
