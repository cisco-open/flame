# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the canonical real/sim parity checks (parity_checks.py).

Runs in the default suite on synthetic telemetry — no MQTT/GPU — so the parity
logic itself is verified independently of any live run. The opt-in end-to-end
check (test_real_sim_e2e_parity.py) reuses the same functions on real runs.
"""

import pathlib
import sys

import pytest

# parity_checks lives with the async_cifar10 example scripts.
_SCRIPTS = (
    pathlib.Path(__file__).resolve().parents[2]
    / "examples" / "async_cifar10" / "scripts"
)
sys.path.insert(0, str(_SCRIPTS))
import parity_checks as pc  # noqa: E402


def _agg(selection=None, agg_rounds=None, agg_evals=None):
    return {
        "selection_train": selection or [],
        "agg_rounds": agg_rounds or [],
        "agg_evals": agg_evals or [],
    }


def _sel(round_, chosen, ts=0.0):
    return {"event": "selection", "task": "train", "round": round_,
            "ts": ts, "chosen": chosen}


def _round(round_, contributing, staleness, agg_goal_count=1, vclock=None, ts=0.0):
    e = {"event": "agg_round", "round": round_, "ts": ts,
         "contributing_trainers": contributing, "staleness": staleness,
         "agg_goal_count": agg_goal_count}
    if vclock is not None:
        e["vclock_now"] = vclock
    return e


class TestSelectionParity:
    def test_identical_is_exact(self):
        a = _agg(selection=[_sel(1, ["x", "y"]), _sel(2, ["y", "z"])])
        r = pc.selection_parity(a, a)
        assert r["ok"] and r["mean_jaccard"] == 1.0 and r["exact_match_frac"] == 1.0

    def test_disjoint_fails(self):
        real = _agg(selection=[_sel(1, ["a", "b"])])
        sim = _agg(selection=[_sel(1, ["c", "d"])])
        r = pc.selection_parity(real, sim)
        assert not r["ok"] and r["mean_jaccard"] == 0.0


class TestStalenessParity:
    def test_matching_ok(self):
        real = _agg(agg_rounds=[_round(1, ["a"], [0]), _round(1, ["b"], [1])])
        sim = _agg(agg_rounds=[_round(1, ["a"], [0]), _round(1, ["b"], [1])])
        r = pc.staleness_parity(real, sim)
        assert r["ok"] and r["real_mean"] == r["sim_mean"]

    def test_negative_staleness_fails(self):
        real = _agg(agg_rounds=[_round(1, ["a"], [0])])
        sim = _agg(agg_rounds=[_round(1, ["a"], [-1])])  # impossible in either mode
        r = pc.staleness_parity(real, sim)
        assert not r["ok"] and r["all_nonnegative"] is False


class TestSimSendTs:
    def test_real_null_sim_increasing_ok(self):
        real_tr = {"aa": {"task_recv": [{"round": 2, "sim_send_ts": None}]}}
        sim_tr = {"aa": {"task_recv": [
            {"round": 2, "sim_send_ts": 5.0}, {"round": 3, "sim_send_ts": 11.0}]}}
        assert pc.sim_send_ts_ok(real_tr, sim_tr)["ok"]

    def test_sim_null_fails(self):
        sim_tr = {"aa": {"task_recv": [{"round": 2, "sim_send_ts": None}]}}
        assert not pc.sim_send_ts_ok({}, sim_tr)["ok"]


class TestGpuBudget:
    def test_within_budget_ok(self):
        tr = {"aa": {"trainer_round": [
            {"real_gpu_time_s": 1.0, "training_budget_s": 5.0},
            {"real_gpu_time_s": 2.0, "training_budget_s": 5.0}]}}
        assert pc.gpu_budget_ok(tr)["ok"]

    def test_overrun_fails(self):
        tr = {"aa": {"trainer_round": [
            {"real_gpu_time_s": 9.0, "training_budget_s": 5.0},
            {"real_gpu_time_s": 8.0, "training_budget_s": 5.0}]}}
        r = pc.gpu_budget_ok(tr)
        assert not r["ok"] and r["mean_overrun_frac"] == 1.0


class TestSimInvariants:
    def test_commit_order_monotone(self):
        ok = _agg(agg_rounds=[_round(1, ["a"], [0], vclock=5.0),
                              _round(1, ["b"], [0], vclock=10.0)])
        assert pc.sim_commit_order_monotone(ok)["ok"]
        bad = _agg(agg_rounds=[_round(1, ["a"], [0], vclock=10.0),
                               _round(1, ["b"], [0], vclock=5.0)])
        assert not pc.sim_commit_order_monotone(bad)["ok"]

    def test_agg_goal_cycles(self):
        ok = _agg(agg_rounds=[_round(1, ["a"], [0], agg_goal_count=1),
                              _round(1, ["b"], [0], agg_goal_count=2)])
        assert pc.agg_goal_cycles_ok(ok, agg_goal=2)["ok"]
        bad = _agg(agg_rounds=[_round(1, ["a"], [0], agg_goal_count=3)])
        assert not pc.agg_goal_cycles_ok(bad, agg_goal=2)["ok"]


class TestLogicalSequence:
    def test_commit_sequence_order(self):
        a = _agg(agg_rounds=[
            _round(2, ["b"], [1], agg_goal_count=2),
            _round(2, ["a"], [0], agg_goal_count=1),  # earlier in logical order
        ])
        seq = pc.commit_sequence(a)
        assert [s["end"] for s in seq] == ["a", "b"]  # sorted by agg_goal_count

    def test_first_divergence_localizes(self):
        real = _agg(agg_rounds=[_round(1, ["a"], [0], 1), _round(1, ["b"], [0], 2)])
        sim = _agg(agg_rounds=[_round(1, ["a"], [0], 1), _round(1, ["c"], [5], 2)])
        d = pc.first_divergence(real, sim)
        assert d["index"] == 1  # first mismatch at the 2nd committed update

    def test_first_divergence_none_when_equal(self):
        a = _agg(agg_rounds=[_round(1, ["a"], [0], 1), _round(1, ["b"], [0], 2)])
        assert pc.first_divergence(a, a)["index"] is None


def test_run_all_parity_smoke():
    a = _agg(selection=[_sel(1, ["a", "b"])],
             agg_rounds=[_round(1, ["a"], [0], vclock=1.0)])
    tr = {"aa": {"task_recv": [], "trainer_round": []}}
    res = pc.run_all_parity(a, a, tr, tr, agg_goal=2)
    # Identical inputs ⇒ no parity check fails. field_coverage may flag the
    # deliberately-sparse fixture's missing telemetry fields (a coverage signal,
    # not a parity divergence), and data-less checks SKIP — both are excluded.
    for name, v in res.items():
        if name == "field_coverage" or v.get("status") == "SKIP":
            continue
        assert v["ok"], f"{name}: {v}"


# ── §3.H clock / throughput new checks ──────────────────────────────────────

def _round_speed(round_, contributing, staleness, agg_goal_count=1,
                 vclock=None, ts=0.0, speed=None):
    """Helper: agg_round event with optional vclock_now and trainer_speed_s."""
    e = _round(round_, contributing, staleness, agg_goal_count, vclock, ts)
    if speed is not None:
        e["trainer_speed_s"] = speed if isinstance(speed, list) else [speed]
    return e


class TestVclockTelemetryPresent:
    def test_present_passes(self):
        sim = _agg(agg_rounds=[_round(1, ["a"], [0], vclock=10.0)])
        r = pc.vclock_telemetry_present(sim)
        assert r["ok"]

    def test_absent_fails(self):
        sim = _agg(agg_rounds=[_round(1, ["a"], [0])])  # no vclock_now
        r = pc.vclock_telemetry_present(sim)
        assert not r["ok"] and "ZERO vclock_now" in r["note"]


class TestThroughputParity:
    def _make_matched_pair(self):
        """Build real + sim where both do ~10 rounds in 100s of their respective time."""
        # real: 10 rounds, wall 0..100 s (10 s/round)
        real_rounds = [
            _round(r, ["a"], [0], ts=float(r * 10))
            for r in range(1, 11)
        ]
        real = _agg(agg_rounds=real_rounds)
        # sim: 10 rounds, vclock 0..100 s (10 s/round virtual)
        sim_rounds = [
            _round(r, ["a"], [0], vclock=float(r * 10), ts=float(r * 1))
            for r in range(1, 11)
        ]
        sim = _agg(agg_rounds=sim_rounds)
        return real, sim

    def test_matched_passes(self):
        real, sim = self._make_matched_pair()
        r = pc.throughput_parity(real, sim, tol_rel=0.10)
        assert r["ok"], r

    def test_diverged_fails(self):
        """Motivating case: sim does 41 rounds in 410 s vclock; real does 67 in 670 s wall."""
        # sim: 41 rounds, 26 s/round virtual → 41/1066 throughput
        sim_rounds = [_round(r, ["a"], [0], vclock=float(r * 26), ts=float(r))
                      for r in range(1, 42)]
        # real: 67 rounds, 15.6 s/round wall → 67/1045 throughput
        real_rounds = [_round(r, ["a"], [0], ts=float(r * 15.6))
                       for r in range(1, 68)]
        real = _agg(agg_rounds=real_rounds)
        sim = _agg(agg_rounds=sim_rounds)
        r = pc.throughput_parity(real, sim, tol_rel=0.10)
        assert not r["ok"], r

    def test_no_vclock_fails(self):
        real = _agg(agg_rounds=[_round(1, ["a"], [0], ts=10.0)])
        sim = _agg(agg_rounds=[_round(1, ["a"], [0])])  # no vclock
        r = pc.throughput_parity(real, sim)
        assert not r["ok"] and "K10" in r.get("note", "")


class TestPerRoundAdvanceParity:
    def test_matched_passes(self):
        # Both advance 10s per round
        real = _agg(agg_rounds=[_round(r, ["a"], [0], ts=float(r * 10))
                                  for r in range(1, 11)])
        sim = _agg(agg_rounds=[_round(r, ["a"], [0], vclock=float(r * 10), ts=float(r))
                                 for r in range(1, 11)])
        r = pc.per_round_advance_parity(real, sim, ks_tol=0.2, mean_tol_rel=0.15)
        assert r["ok"], r

    def test_diverged_fails(self):
        # sim: 26 s/round vclock; real: 15 s/round wall → 73% relative diff
        real = _agg(agg_rounds=[_round(r, ["a"], [0], ts=float(r * 15))
                                  for r in range(1, 11)])
        sim = _agg(agg_rounds=[_round(r, ["a"], [0], vclock=float(r * 26), ts=float(r))
                                 for r in range(1, 11)])
        r = pc.per_round_advance_parity(real, sim, ks_tol=0.2, mean_tol_rel=0.15)
        assert not r["ok"], r


class TestOverlapFactor:
    def test_matched_passes(self):
        # Both: speed=20s, advance=15s → overlap ≈ 1.33 for both
        real = _agg(agg_rounds=[
            _round_speed(r, ["a"], [0], ts=float(r * 15), speed=20.0)
            for r in range(1, 11)
        ])
        sim = _agg(agg_rounds=[
            _round_speed(r, ["a"], [0], vclock=float(r * 15), ts=float(r), speed=20.0)
            for r in range(1, 11)
        ])
        r = pc.overlap_factor(real, sim, tol=0.3)
        assert r["ok"], r

    def test_no_overlap_sim_fails(self):
        # sim: speed≈advance (1.0 overlap); real: speed=20, advance=12 (1.67 overlap)
        real = _agg(agg_rounds=[
            _round_speed(r, ["a"], [0], ts=float(r * 12), speed=20.0)
            for r in range(1, 11)
        ])
        sim = _agg(agg_rounds=[
            _round_speed(r, ["a"], [0], vclock=float(r * 20), ts=float(r), speed=20.0)
            for r in range(1, 11)
        ])
        r = pc.overlap_factor(real, sim, tol=0.3)
        assert not r["ok"], r


class TestTotalCommitsParity:
    def test_matched_passes(self):
        # Both: 5 commits spanning 0-40s of their respective time
        # real: ts = 0,10,20,30,40  (wall_elapsed=40); sim: vclock = 0,10,20,30,40
        real = _agg(agg_rounds=[_round(r, ["a"], [0], ts=float((r - 1) * 10))
                                  for r in range(1, 6)])
        sim = _agg(agg_rounds=[_round(r, ["a"], [0], vclock=float((r - 1) * 10),
                                       ts=float(r))
                                 for r in range(1, 6)])
        r = pc.total_commits_parity(real, sim, tol_rel=0.05)
        assert r["ok"], r

    def test_diverged_fails(self):
        # sim: 3 commits, vclock 0,10,20 → final_vclock=20
        # real: 6 commits at ts 0,5,10,15,20,25 → wall_elapsed=25, V=min(20,25)=20
        # n_sim: vclock ≤ 20 → 3; n_real: ts-t0=0,5,10,15,20,25 ≤ 20 → 5 → diverged
        real = _agg(agg_rounds=[_round(r, ["a"], [0], ts=float((r - 1) * 5))
                                  for r in range(1, 7)])
        sim = _agg(agg_rounds=[_round(r, ["a"], [0], vclock=float((r - 1) * 10),
                                       ts=float(r))
                                 for r in range(1, 4)])
        r = pc.total_commits_parity(real, sim, tol_rel=0.02)
        assert not r["ok"], r


class TestTerminalStateParity:
    def test_matched_passes(self):
        # Both modes: 10 rounds on a normalized 0..90s timeline → rel_diff 0.
        # (real wall normalizes by t0=min(ts); sim vclock is used directly, so the
        # sim vclocks must span 0..90 to match real's normalized 0..90 — otherwise
        # the V=min cutoff drops sim's last round and yields a spurious 10% edge.)
        real = _agg(agg_rounds=[_round(r, ["a", "b"], [0, 0], ts=float(r * 10))
                                  for r in range(1, 11)])
        sim = _agg(agg_rounds=[_round(r, ["a", "b"], [0, 0], vclock=float((r - 1) * 10),
                                       ts=float(r))
                                 for r in range(1, 11)])
        r = pc.terminal_state_parity(real, sim)
        assert r["ok"], r

    def test_diverged_fails(self):
        # sim: 5 rounds in 130s vclock; real: 10 rounds in 100s wall
        # V = min(130, 100) = 100; sim has 4 rounds ≤100, real has 10
        real = _agg(agg_rounds=[_round(r, ["a"], [0], ts=float(r * 10))
                                  for r in range(1, 11)])
        sim = _agg(agg_rounds=[_round(r, ["a"], [0], vclock=float(r * 26),
                                       ts=float(r))
                                 for r in range(1, 6)])
        r = pc.terminal_state_parity(real, sim, rounds_tol=0.10)
        assert not r["ok"], r


class TestTrainerSpeedParity:
    def test_identical_passes(self):
        rounds_with_speed = [
            _round_speed(r, ["a"], [0], speed=28.0) for r in range(1, 6)
        ]
        agg = _agg(agg_rounds=rounds_with_speed)
        r = pc.trainer_speed_parity(agg, agg, ks_tol=0.1)
        assert r["ok"]

    def test_out_of_support_tail_fails(self):
        # Jun-16 support-guard semantics: P3 fails when sim produces speeds BEYOND
        # real's support (the genuine speed-model bug — oort's old 56s→sim tail).
        real_rounds = [_round_speed(r, ["a"], [0], speed=11.0) for r in range(1, 11)]
        sim_rounds = [_round_speed(r, ["a"], [0], speed=56.0) for r in range(1, 11)]
        real = _agg(agg_rounds=real_rounds)
        sim = _agg(agg_rounds=sim_rounds)
        r = pc.trainer_speed_parity(real, sim)
        assert not r["ok"]
        assert r["support_ratio"] > 1.0 + r["support_tol"]

    def test_faster_sim_within_support_defers_to_mix(self):
        # sim faster than real, same support direction (wall-capture / faster
        # selection mix) is NOT a speed-model bug — P3 passes, A2c owns the mix.
        real_rounds = [_round_speed(r, ["a"], [0], speed=11.0) for r in range(1, 11)]
        sim_rounds = [_round_speed(r, ["a"], [0], speed=7.0) for r in range(1, 11)]
        r = pc.trainer_speed_parity(_agg(agg_rounds=real_rounds),
                                    _agg(agg_rounds=sim_rounds))
        assert r["ok"], r


class TestBudgetNotCap:
    def test_no_cap_provided_skips(self):
        a = _agg(agg_rounds=[_round(100, ["a"], [0], ts=100.0)])
        r = pc.budget_not_cap(a, a)
        assert r["ok"]

    def test_under_cap_passes(self):
        a = _agg(agg_rounds=[_round(50, ["a"], [0], ts=100.0)])
        r = pc.budget_not_cap(a, a, rounds_cap=1000)
        assert r["ok"]

    def test_hit_cap_warns(self):
        a = _agg(agg_rounds=[_round(1000, ["a"], [0], ts=100.0)])
        r = pc.budget_not_cap(a, a, rounds_cap=1000, budget_s=10800.0)
        assert not r["ok"] and r.get("warnings")


class TestRunAllParityExtended:
    """Smoke test: run_all_parity returns a result for every expected new key."""

    _NEW_KEYS = [
        "vclock_telemetry", "throughput", "per_round_advance",
        "overlap_factor", "total_commits", "terminal_state",
        "trainer_speed",
    ]

    def test_new_keys_present(self):
        a = _agg(
            selection=[_sel(1, ["a", "b"])],
            agg_rounds=[
                _round_speed(r, ["a"], [0], vclock=float(r * 10),
                              ts=float(r), speed=12.0)
                for r in range(1, 6)
            ],
        )
        tr = {"aa": {"task_recv": [], "trainer_round": []}}
        res = pc.run_all_parity(a, a, tr, tr, agg_goal=2)
        for key in self._NEW_KEYS:
            assert key in res, f"missing key: {key}"

    def test_overall_verdict_fails_on_k2(self):
        """A pair where sim overcharges virtual time should fail overall verdict."""
        real = _agg(agg_rounds=[_round(r, ["a"], [0], ts=float(r * 15))
                                  for r in range(1, 68)])
        sim = _agg(agg_rounds=[_round(r, ["a"], [0], vclock=float(r * 26),
                                       ts=float(r))
                                 for r in range(1, 42)])
        tr: dict = {}
        res = pc.run_all_parity(real, sim, tr, tr)
        passed, roots, downstream, _warnings = pc.overall_verdict(res)
        assert not passed
        failures = set(roots) | set(downstream)
        assert "throughput" in failures or "per_round_advance" in failures


def _sel_pool(round_, speeds, ts=0.0):
    """selection event with a per_trainer pool carrying speed_s (for A2b)."""
    per_trainer = {f"t{i:03d}": {"speed_s": sp, "utility": None, "selected": i < 10}
                   for i, sp in enumerate(speeds)}
    return {"event": "selection", "task": "train", "round": round_, "ts": ts,
            "num_candidates": len(speeds), "num_eligible": len(speeds),
            "per_trainer": per_trainer}


class TestEligibleSpeedComposition:
    """A2b: eligible-pool speed-composition parity (catches what A2's count misses)."""

    def test_matched_pool_passes(self):
        pool = [3.0, 5.0, 8.0, 12.0, 20.0] * 6
        real = _agg(selection=[_sel_pool(r, pool) for r in range(1, 6)])
        sim = _agg(selection=[_sel_pool(r, pool) for r in range(1, 6)])
        res = pc.eligible_speed_composition_parity(real, sim)
        assert res["ok"], res

    def test_diverged_pool_fails(self):
        # Same eligible-set SIZE (30) in both, but sim pool is slow-skewed (the refl
        # signature: slow clients re-enter sim's pool). A2b must FAIL on composition.
        fast = [2.0, 3.0, 4.0, 5.0, 6.0] * 6   # real: fast-skewed pool
        slow = [10.0, 12.0, 14.0, 18.0, 22.0] * 6  # sim: slow-skewed pool
        real = _agg(selection=[_sel_pool(r, fast) for r in range(1, 6)])
        sim = _agg(selection=[_sel_pool(r, slow) for r in range(1, 6)])
        res = pc.eligible_speed_composition_parity(real, sim)
        assert not res["ok"], res
        assert res["real_mean_pool_speed_s"] < res["sim_mean_pool_speed_s"]

    def test_no_per_trainer_skips(self):
        real = _agg(selection=[_sel(1, ["a"])])
        sim = _agg(selection=[_sel(1, ["a"])])
        res = pc.eligible_speed_composition_parity(real, sim)
        assert res["ok"] and res.get("status") == "SKIP", res


class TestInflightResidenceEvent:
    """The oort in-flight residence telemetry builder (PARITY §4.x fine-tuning)."""

    def test_builder_shape(self):
        from flame.telemetry.events import build_inflight_residence, EVENT_INFLIGHT_RESIDENCE
        ev, f = build_inflight_residence(
            round_num=7, time_mode="sim", in_flight_before=16, in_flight_after=13,
            committed_fresh=10, cleaned=3, stale_rejected=0,
            residence_rounds=[0, 1, 2], carried_over_ages=[0, 0, 1])
        assert ev == EVENT_INFLIGHT_RESIDENCE
        assert f["time_mode"] == "sim" and f["in_flight_before"] == 16
        assert f["residence_rounds"] == [0, 1, 2]
        # optional fields omitted when None
        ev2, f2 = build_inflight_residence(
            round_num=1, time_mode="real", in_flight_before=13, in_flight_after=13)
        assert "residence_rounds" not in f2 and "cleaned" not in f2

    def test_builder_paired_commit_class(self):
        """residence_staleness / residence_was_fresh ride alongside residence_rounds
        (paired 1:1) to decompose the residence-shape gap by commit class (refl A2)."""
        from flame.telemetry.events import build_inflight_residence
        ev, f = build_inflight_residence(
            round_num=7, time_mode="real", in_flight_before=16, in_flight_after=13,
            residence_rounds=[3, 3, 5], residence_staleness=[0, 2, 4],
            residence_was_fresh=[True, False, False])
        assert f["residence_staleness"] == [0, 2, 4]
        assert f["residence_was_fresh"] == [True, False, False]
        assert len(f["residence_staleness"]) == len(f["residence_rounds"])
        # omitted when not supplied
        _, f2 = build_inflight_residence(
            round_num=1, time_mode="sim", in_flight_before=13, in_flight_after=13,
            residence_rounds=[1])
        assert "residence_staleness" not in f2
