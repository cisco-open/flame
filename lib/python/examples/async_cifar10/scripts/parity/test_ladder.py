"""Fault-injection tests for the parity causal ladder.

Each test builds a synthetic real/sim pair, injects a *single* mechanism fault,
and asserts the verdict names exactly that rung as ROOT-CAUSE with the rest
demoted to downstream.  This is the regression net that keeps the ladder
self-reinforcing (PARITY.md §3d).

Run:  python -m pytest scripts/parity/test_ladder.py -q
  or:  python scripts/parity/test_ladder.py
"""

from __future__ import annotations

import os
import sys

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from parity.checks import run_all_parity, overall_verdict  # noqa: E402

AGG_GOAL = 4
TRAINERS = ["0001", "0002", "0003", "0004"]
SPEEDS = [1.0, 2.0, 3.0, 4.0]
STALENESS = [0, 1, 0, 1]
UTILS = [10.0, 20.0, 30.0, 40.0]
PHASES = {
    "pre_train_s": 0.01, "weights_to_gpu_s": 0.5, "gpu_compute_s": 8.0,
    "mqtt_fetch_s": 1.0, "weights_to_ram_s": 0.3, "post_train_s": 0.01,
}


def _build_mode(n_rounds: int, advance: float, *, with_vclock: bool) -> tuple:
    """Build (agg, trainers) for one mode.

    real mode (with_vclock=False): progress carried by ts = (r-1)*advance.
    sim  mode (with_vclock=True):  progress carried by vclock_now=(r-1)*advance;
                                   ts is a separate fast wall clock.
    """
    agg_rounds, selection_train, agg_evals = [], [], []
    for r in range(1, n_rounds + 1):
        prog = (r - 1) * advance
        e = {
            "event": "agg_round", "round": r, "agg_goal_count": AGG_GOAL,
            "trainer_speed_s": list(SPEEDS),
            "contributing_trainers": list(TRAINERS),
            "staleness": list(STALENESS),
            "stat_utility": list(UTILS),
        }
        if with_vclock:
            e["vclock_now"] = prog
            e["ts"] = (r - 1) * 1.0  # fast independent wall clock
        else:
            e["ts"] = prog
        agg_rounds.append(e)
        selection_train.append({
            "event": "selection", "task": "train", "round": r,
            "ts": e["ts"], "chosen": list(TRAINERS),
            "avail_composition": {"TRAIN": 10}, "num_eligible": 10,
            "num_candidates": 10, "num_chosen": AGG_GOAL, "in_flight": AGG_GOAL,
            "effective_c": AGG_GOAL, "selector": "OortSelector",
        })
    for r in range(1, n_rounds + 1, max(1, n_rounds // 5)):
        agg_evals.append({"event": "agg_eval", "round": r,
                          "test-accuracy": 0.5 + r * 0.001,
                          "test-loss": 2.0 - r * 0.001})

    trainers: dict = {}
    for tid in TRAINERS:
        task_recv, trainer_round = [], []
        for r in range(1, n_rounds + 1):
            tr = {"event": "trainer_round", "round": r,
                  "real_gpu_time_s": PHASES["gpu_compute_s"],
                  "training_budget_s": PHASES["gpu_compute_s"] + 1.0}
            tr.update(PHASES)
            trainer_round.append(tr)
            task_recv.append({
                "event": "task_recv", "round": r,
                "sim_send_ts": ((r - 1) * advance if with_vclock else None),
            })
        trainers[tid] = {"task_recv": task_recv, "trainer_round": trainer_round}

    agg = {"agg_rounds": agg_rounds, "selection_train": selection_train,
           "agg_evals": agg_evals}
    return agg, trainers


def _verdict(real_agg, sim_agg, real_tr, sim_tr):
    results = run_all_parity(real_agg, sim_agg, real_tr, sim_tr,
                             agg_goal=AGG_GOAL)
    return results, overall_verdict(results)


# ───────────────────────────── tests ──────────────────────────────────────

def test_clean_pair_passes():
    """Identical content + matched time-base ⇒ no enforced failures."""
    real_agg, real_tr = _build_mode(20, advance=10.0, with_vclock=False)
    sim_agg, sim_tr = _build_mode(20, advance=10.0, with_vclock=True)
    results, (passed, roots, downstream, warnings) = _verdict(
        real_agg, sim_agg, real_tr, sim_tr)
    assert passed, (
        f"clean pair should pass; roots={roots} downstream={downstream}")
    assert not roots and not downstream


def test_overhead_residual_is_root():
    """Sim under-charges the clock (no per-commit overhead) ⇒ Stage-1
    overhead_residual is the root; throughput/per_round_advance demote to
    downstream; the speed control (P3) still passes."""
    real_agg, real_tr = _build_mode(20, advance=10.0, with_vclock=False)
    sim_agg, sim_tr = _build_mode(20, advance=3.0, with_vclock=True)
    results, (passed, roots, downstream, warnings) = _verdict(
        real_agg, sim_agg, real_tr, sim_tr)
    assert not passed
    assert "overhead_residual" in roots, f"roots={roots}"
    assert results["trainer_speed"]["ok"], "P3 control must stay green"
    # emergent clock checks are consequences, not independent roots
    assert "per_round_advance" in downstream
    assert "throughput" in downstream
    assert "per_round_advance" not in roots and "throughput" not in roots


def test_missing_vclock_is_stage0_root():
    """Sim path never stamps vclock_now ⇒ Stage-0 coverage is the root and the
    Stage-1 clock checks SKIP (not FAIL)."""
    real_agg, real_tr = _build_mode(20, advance=10.0, with_vclock=False)
    sim_agg, sim_tr = _build_mode(20, advance=10.0, with_vclock=False)  # no vclock
    results, (passed, roots, downstream, warnings) = _verdict(
        real_agg, sim_agg, real_tr, sim_tr)
    assert not passed
    assert roots and roots[0] in ("field_coverage", "vclock_telemetry"), \
        f"expected stage-0 root, got {roots}"
    # downstream clock checks must SKIP, never silently FAIL
    assert results["overhead_residual"].get("status") == "SKIP"
    assert results["throughput"].get("note", "").startswith("K10:") or \
        results["throughput"].get("status") == "SKIP"


def test_phase_split_localizes_single_phase():
    """A divergence confined to mqtt_fetch flags only that phase, not the rest."""
    real_agg, real_tr = _build_mode(20, advance=10.0, with_vclock=False)
    sim_agg, sim_tr = _build_mode(20, advance=10.0, with_vclock=True)
    # blow up only the mqtt_fetch phase in sim
    for d in sim_tr.values():
        for e in d["trainer_round"]:
            e["mqtt_fetch_s"] = 50.0
    results, _ = _verdict(real_agg, sim_agg, real_tr, sim_tr)
    assert not results["phase_mqtt_fetch"]["ok"], "mqtt phase should fail"
    for other in ("phase_pre_train", "phase_gpu_compute", "phase_weights_to_gpu"):
        assert results[other]["ok"], f"{other} should stay green"


def test_avail_timebase_detects_trajectory_shift():
    """A systematic eligible-count shift in sim trips A3."""
    real_agg, real_tr = _build_mode(20, advance=10.0, with_vclock=False)
    sim_agg, sim_tr = _build_mode(20, advance=10.0, with_vclock=True)
    for i, e in enumerate(sim_agg["selection_train"]):
        e["num_eligible"] = 10 if i < 10 else 2  # second half collapses
    results, _ = _verdict(real_agg, sim_agg, real_tr, sim_tr)
    assert not results["avail_timebase"]["ok"], "A3 should catch the shift"


def test_eligibility_pointmass_passes_on_mean():
    """A2: real num_eligible is a constant point-mass (300), sim 298.9 ± tiny.
    KS saturates to ~1 but the means match — must PASS on the mean (PARITY.md §3i),
    not raise a spurious FAIL."""
    from parity.checks import eligibility_parity
    import random
    rng = random.Random(0)
    real = {"selection_train": [{"num_eligible": 300, "num_candidates": 300}
                                for _ in range(500)]}
    # sim: occasionally one fewer eligible -> mean 298.9, tiny variance
    sim = {"selection_train": [{"num_eligible": 300 - (1 if rng.random() < 0.5 else 0),
                                "num_candidates": 300 - (1 if rng.random() < 0.5 else 0)}
                               for _ in range(500)]}
    res = eligibility_parity(real, sim)
    assert res["ks_eligible"] is not None and res["ks_eligible"] > 0.2, "KS must be high"
    assert res["ok"], "point-mass with matching means must pass"
    assert "point-mass" in res.get("note", ""), "note should explain the rescue"


def test_eligibility_real_divergence_still_fails():
    """A2 guard: a genuine eligible-set divergence (different means, real spread)
    must STILL FAIL — the point-mass rescue must not mask it."""
    from parity.checks import eligibility_parity
    import random
    rng = random.Random(1)
    real = {"selection_train": [{"num_eligible": rng.randint(250, 300),
                                 "num_candidates": rng.randint(250, 300)}
                                for _ in range(500)]}
    sim = {"selection_train": [{"num_eligible": rng.randint(100, 150),
                                "num_candidates": rng.randint(100, 150)}
                               for _ in range(500)]}
    res = eligibility_parity(real, sim)
    assert not res["ok"], "a real eligible-set divergence must not be rescued"


def _sel_events(rows):
    """rows: list of (selected_bool, speed_s, extra_dict) -> one selection event."""
    pt = {}
    for i, (seld, sp, extra) in enumerate(rows):
        e = {"speed_s": sp, "selected": seld}
        e.update(extra or {})
        pt[f"t{i}"] = e
    return {"selection_train": [{"event": "selection", "task": "train",
                                 "round": 1, "ts": 0.0, "per_trainer": pt}]}


def test_selection_bias_localizes_selector_vs_pool():
    """A2c: pool matches but the selector's revealed speed preference (bias =
    selected-mean − pool-mean) diverges ⇒ selector-scoring case (oort), distinct
    from a pool-composition case (refl, where the pool itself diverges). The pool
    (all per_trainer) is IDENTICAL across modes; only the `selected` flags differ."""
    from parity.checks import selection_speed_bias_parity
    POOL = [3.0, 4.0, 5.0, 10.0, 15.0, 20.0]            # same candidates both modes
    real_rows = [(sp in (3.0, 4.0), sp, {}) for sp in POOL] * 30   # real picks fast
    sim_rows = [(sp in (10.0, 15.0), sp, {}) for sp in POOL] * 30  # sim picks ~average
    res = selection_speed_bias_parity(_sel_events(real_rows), _sel_events(sim_rows))
    assert abs(res["real_pool_mean_s"] - res["sim_pool_mean_s"]) < 0.01  # pools match
    assert res["real_bias_s"] < -3.0, res              # real exploits (picks fast)
    assert res["sim_bias_s"] > res["real_bias_s"] + 3.0  # sim's preference is flatter


def test_selector_score_localizes_worst_term():
    """The score-localizer pinpoints the single diverging utility-score term
    (here system_util) and leaves the matching term (believed_I) alone."""
    from parity.checks import selector_score_parity
    # believed_I distribution is IDENTICAL across modes (KS≈0); only system_util shifts.
    bi = [66.0, 68.0, 70.0, 72.0, 74.0]
    real = _sel_events([(True, 5.0, {"believed_I": b, "system_util": 0.70}) for b in bi] * 20)
    sim = _sel_events([(True, 5.0, {"believed_I": b, "system_util": 0.95}) for b in bi] * 20)
    res = selector_score_parity(real, sim)
    assert res["worst_component"] == "system_util", res
    assert res["per_component"]["believed_I"]["ks"] < 0.01     # matched term
    assert res["per_component"]["system_util"]["ks"] > 0.9     # the diverging term


def test_selection_checks_skip_without_per_trainer():
    """Both new checks SKIP cleanly when selection telemetry has no per_trainer
    (non-instrumented selectors) — never a spurious failure."""
    from parity.checks import (selection_speed_bias_parity, selector_score_parity,
                               preferred_duration_parity)
    empty = {"selection_train": [{"event": "selection", "task": "train",
                                  "round": 1, "ts": 0.0}]}
    for fn in (selection_speed_bias_parity, selector_score_parity,
               preferred_duration_parity):
        res = fn(empty, empty)
        assert res["ok"] and res.get("status") == "SKIP", (fn.__name__, res)


def _binding_rounds(frac_binding, n=100):
    """n selection rounds; a `frac_binding` fraction have a speed-penalized
    selected trainer (system_util<1), the rest are non-binding (system_util==1)."""
    events = []
    for r in range(n):
        binds = r < int(frac_binding * n)
        su = 0.5 if binds else 1.0
        pt = {"t0": {"speed_s": 10.0, "selected": True, "system_util": su},
              "t1": {"speed_s": 6.0, "selected": True, "system_util": 1.0}}
        events.append({"event": "selection", "task": "train", "round": r,
                       "ts": float(r), "per_trainer": pt})
    return {"selection_train": events}


def test_preferred_duration_detects_binding_frequency_gap():
    """Sd: the Jun-15 oort sort bug left system_util only ~.13 KS off but flipped
    the PENALTY BINDING FREQUENCY hard (real ~80%/round vs sim ~46%). This check
    catches that gap directly — the growth-rule guard for the unsorted-`pref` bug."""
    from parity.checks import preferred_duration_parity
    real = _binding_rounds(0.80)
    sim = _binding_rounds(0.45)
    res = preferred_duration_parity(real, sim)
    assert not res["ok"], res                                  # gap is flagged
    assert abs(res["real_frac_binding"] - 0.80) < 0.02, res
    assert abs(res["sim_frac_binding"] - 0.45) < 0.02, res
    assert res["frac_diff"] > 0.2, res
    # a MATCHED binding frequency passes
    res2 = preferred_duration_parity(_binding_rounds(0.78), _binding_rounds(0.80))
    assert res2["ok"], res2


def _residence_events(carry, fresh=10, stale=6, res_rounds=0.2, n=100):
    """n rounds of inflight_residence telemetry with a given mean carry-over."""
    return {"residence": [
        {"event": "inflight_residence", "round": r, "ts": float(r),
         "in_flight_after": carry, "committed_fresh": fresh,
         "stale_rejected": stale,
         "residence_rounds": [1] * int(round(res_rounds * 10)) + [0] * (10 - int(round(res_rounds * 10)))}
        for r in range(n)]}


def test_residence_detects_straggler_drain():
    """Sr: the oort sync stack over-selects, so real CARRIES ~3 stragglers in-flight
    each round while a naive sim cleans up the instantly-arrived updates and DRAINS
    to ~0. This structural carry-over gap (invariant to selection mix) is the §4.5-
    class root the residence rung makes first-class."""
    from parity.checks import inflight_residence_parity
    real = _residence_events(carry=3.3)
    sim = _residence_events(carry=0.15)
    res = inflight_residence_parity(real, sim)
    assert not res["ok"], res                          # drain-vs-carry flagged
    assert res["real_inflight_after"] > 3.0
    assert res["sim_inflight_after"] < 0.5
    assert res["rel_diff_carry"] > 0.3
    # matched carry-over passes
    res2 = inflight_residence_parity(_residence_events(3.2), _residence_events(3.3))
    assert res2["ok"], res2
    # no-overcommit (nothing to carry in either mode) is trivially matched, not a fail
    res3 = inflight_residence_parity(_residence_events(0.05), _residence_events(0.0))
    assert res3["ok"], res3
    # async stack / no residence telemetry → clean SKIP, never a spurious failure
    res4 = inflight_residence_parity({"residence": []}, {"residence": []})
    assert res4["ok"] and res4.get("status") == "SKIP", res4


def test_convergence_low_confidence_on_short_runs():
    """A convergence PASS under a sub-2h budget is downgraded to LOW_CONF (the
    curves haven't diverged yet); a genuine FAIL still surfaces regardless of
    horizon; a long-run PASS is a confident pass."""
    from parity.checks import convergence_loss_parity, SHORT_RUN_CONFIDENCE_S

    def evs(losses):
        return {"agg_evals": [{"round": i, "test-loss": v}
                              for i, v in enumerate(losses)]}

    match = evs([1.0, 0.8, 0.6])        # real
    near = evs([1.0, 0.81, 0.61])       # sim within tol -> PASS
    far = evs([1.0, 1.4, 1.9])          # sim far -> FAIL

    short = SHORT_RUN_CONFIDENCE_S - 1
    long = SHORT_RUN_CONFIDENCE_S + 1

    # short + pass -> ok stays True but flagged low-confidence
    r = convergence_loss_parity(match, near, budget_s=short)
    assert r["ok"] and r.get("low_confidence") and r.get("status") == "LOW_CONF"
    # long + pass -> confident pass, no flag
    r = convergence_loss_parity(match, near, budget_s=long)
    assert r["ok"] and not r.get("low_confidence")
    # short + genuine divergence -> still FAILS, not masked
    r = convergence_loss_parity(match, far, budget_s=short)
    assert not r["ok"] and not r.get("low_confidence")
    # no budget given -> unchanged (legacy)
    r = convergence_loss_parity(match, near, budget_s=None)
    assert r["ok"] and not r.get("low_confidence")


def test_trainer_speed_tolerates_wall_capture_jitter():
    """P3 compares at the modeled integer-second grid: sim reports exact integer
    training-delays, real carries sub-second wall-capture jitter on top. The raw
    KS would FAIL on that jitter alone; the grid KS must PASS. But a *systematic*
    overhead (real shifted a full grid cell) must still FAIL."""
    from parity.checks import trainer_speed_parity
    import random
    random.seed(0)
    base = [float(random.choice([2, 3, 5, 6, 8, 9])) for _ in range(20000)]
    sim = {"agg_rounds": [{"trainer_speed_s": list(base)}]}
    # real = same integer grid + small positive capture jitter (sleep/settle leg)
    jittered = [v + random.uniform(0.02, 0.09) for v in base]
    real = {"agg_rounds": [{"trainer_speed_s": jittered}]}
    res = trainer_speed_parity(real, sim)
    assert res["raw_ks_stat"] > 0.1, res          # raw sub-second KS would fail
    assert res["ok"], res                          # grid KS passes
    assert res["ks_stat"] <= 0.1, res
    # a genuine ~+1s systematic speed-model offset is NOT absorbed
    shifted = [v + 1.0 for v in base]
    res_bad = trainer_speed_parity({"agg_rounds": [{"trainer_speed_s": shifted}]}, sim)
    assert not res_bad["ok"], res_bad


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
