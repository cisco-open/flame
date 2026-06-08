# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Opt-in end-to-end real/sim parity check.

This is NOT run in the default suite (it needs two completed runs — a real and a
simulated one of the *same seeded* scenario). Point it at the run dirs and run:

    FLAME_E2E_REAL_DIR=experiments/run_..._real \\
    FLAME_E2E_SIM_DIR=experiments/run_..._sim \\
    FLAME_E2E_AGG_GOAL=5 \\
    pytest lib/python/tests/mode/test_real_sim_e2e_parity.py -v

It asserts the parity battery from parity_checks.py: exact-matchable invariants
(sim_send_ts, virtual-clock monotonicity, agg_goal cycles, staleness ≥ 0, GPU
budget respected) are required; distributional quantities (selection Jaccard,
staleness KS) must meet the seeded-run tolerances. Generate the run pair with a
seeded config (see expt_scripts_2026/felix_n*_parity_seeded_{real,sim}.yaml).
"""

import glob
import os
import pathlib
import sys

import pytest

_SCRIPTS = (
    pathlib.Path(__file__).resolve().parents[2]
    / "examples" / "async_cifar10" / "scripts"
)
sys.path.insert(0, str(_SCRIPTS))
import parity_checks as pc  # noqa: E402

REAL_DIR = os.environ.get("FLAME_E2E_REAL_DIR")
SIM_DIR = os.environ.get("FLAME_E2E_SIM_DIR")
AGG_GOAL = int(os.environ.get("FLAME_E2E_AGG_GOAL", "0") or 0)

pytestmark = pytest.mark.skipif(
    not (REAL_DIR and SIM_DIR),
    reason="set FLAME_E2E_REAL_DIR and FLAME_E2E_SIM_DIR to two run dirs to enable",
)


def _agg_jsonl(run_dir):
    hits = glob.glob(os.path.join(run_dir, "telemetry", "aggregator_*.jsonl"))
    assert hits, f"no aggregator telemetry under {run_dir}/telemetry"
    return hits[0]


@pytest.fixture(scope="module")
def results():
    real_agg = pc.load_agg_jsonl(_agg_jsonl(REAL_DIR))
    sim_agg = pc.load_agg_jsonl(_agg_jsonl(SIM_DIR))
    real_tr = pc.load_trainer_jsonl_dir(os.path.join(REAL_DIR, "telemetry"))
    sim_tr = pc.load_trainer_jsonl_dir(os.path.join(SIM_DIR, "telemetry"))
    return pc.run_all_parity(real_agg, sim_agg, real_tr, sim_tr, agg_goal=AGG_GOAL)


# ── exact-matchable invariants (must hold) ──────────────────────────────────

def test_sim_send_ts(results):
    assert results["sim_send_ts"]["ok"], results["sim_send_ts"]


def test_sim_commit_order_monotone(results):
    assert results["sim_commit_monotone"]["ok"], results["sim_commit_monotone"]


def test_staleness_nonnegative_and_close(results):
    assert results["staleness"]["ok"], results["staleness"]


def test_gpu_budget_respected_sim(results):
    assert results["gpu_budget_sim"]["ok"], results["gpu_budget_sim"]


@pytest.mark.skipif(AGG_GOAL <= 0, reason="set FLAME_E2E_AGG_GOAL to enable")
def test_agg_goal_cycles(results):
    assert results["agg_goal_cycles_sim"]["ok"], results["agg_goal_cycles_sim"]
    assert results["agg_goal_cycles_real"]["ok"], results["agg_goal_cycles_real"]


# ── distributional parity (seeded-run tolerances) ───────────────────────────

def test_selection_parity(results):
    # Enforced only for deterministic selectors; for stochastic ones (all shipped
    # selectors today) this is gated to a report — exact per-round selection can't
    # match across real/sim (join-order-dependent candidate ordering + RNG), so
    # participation-frequency parity below is the enforced selection invariant.
    # See parity_checks.DETERMINISTIC_SELECTORS for the full rationale.
    assert results["selection"]["ok"], results["selection"]


def test_participation_parity(results):
    assert results["participation"]["ok"], results["participation"]
