"""Parity checks library — importable single source of truth.

Extends and supersedes ``parity_checks.py`` (which remains a re-export shim for
backward compatibility with the pytest suite and CLI scripts that import from it).

Full §3 battery from real-sim_parity_checker_plan.md:
  §3.A  Availability / eligibility    (A1–A4)
  §3.B  Selection                     (S1–S5)
  §3.C  Training                      (T1–T6)
  §3.D  Updates received & ordering   (U1–U5)
  §3.E  Update processing             (P1–P3)
  §3.F  Statistical utility           (F1–F3)
  §3.G  Convergence                   (C1–C3, self-compare bug fixed)
  §3.H  Clock & throughput            (K1–K10)  ← the new enforced core

All functions are stdlib-only so they run in the default pytest environment.
"""

from __future__ import annotations

import collections
import glob
import json
import math
import os
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════════════
# §0  Helpers
# ═══════════════════════════════════════════════════════════════════

def short(end_id: str) -> str:
    return end_id[-4:] if end_id else "None"


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def mean_std(vals: list) -> tuple:
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    v = sum((x - m) ** 2 for x in vals) / len(vals)
    return m, math.sqrt(v)


def ks_stat(a: list, b: list) -> float:
    """Two-sample Kolmogorov–Smirnov statistic (no scipy needed)."""
    if not a or not b:
        return float("nan")
    combined = sorted(set(a + b))
    na, nb = len(a), len(b)
    sa, sb = sorted(a), sorted(b)
    ia = ib = 0
    d = 0.0
    for v in combined:
        while ia < na and sa[ia] <= v:
            ia += 1
        while ib < nb and sb[ib] <= v:
            ib += 1
        d = max(d, abs(ia / na - ib / nb))
    return d


def spearman_rho(a: list, b: list) -> float:
    """Spearman rank correlation (no scipy needed)."""
    n = min(len(a), len(b))
    if n < 2:
        return float("nan")
    a, b = a[:n], b[:n]

    def _ranks(xs):
        sorted_idx = sorted(range(len(xs)), key=lambda i: xs[i])
        ranks = [0.0] * len(xs)
        for rank, idx in enumerate(sorted_idx):
            ranks[idx] = float(rank + 1)
        return ranks

    ra, rb = _ranks(a), _ranks(b)
    d2 = sum((ra[i] - rb[i]) ** 2 for i in range(n))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


# ═══════════════════════════════════════════════════════════════════
# §1  Loaders
# ═══════════════════════════════════════════════════════════════════

def load_agg_jsonl(path: str) -> dict:
    """Parse an aggregator telemetry JSONL into typed, sorted lists."""
    selection_train: list = []
    agg_rounds: list = []
    agg_evals: list = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            ev = e.get("event")
            if ev == "selection" and e.get("task") == "train":
                selection_train.append(e)
            elif ev == "agg_round":
                agg_rounds.append(e)
            elif ev == "agg_eval":
                agg_evals.append(e)
    selection_train.sort(key=lambda x: (x["round"], x["ts"]))
    agg_rounds.sort(key=lambda x: (x["round"], x.get("agg_goal_count", 0), x["ts"]))
    agg_evals.sort(key=lambda x: x["round"])
    return {
        "selection_train": selection_train,
        "agg_rounds": agg_rounds,
        "agg_evals": agg_evals,
    }


def load_trainer_jsonl_dir(telemetry_dir: Optional[str]) -> dict:
    """Load all trainer_*.jsonl from a telemetry dir.

    Returns {short_id: {"task_recv": [...], "trainer_round": [...]}}.
    """
    if not telemetry_dir:
        return {}
    d = Path(telemetry_dir)
    result: dict = {}
    for f in sorted(d.glob("trainer_*.jsonl")):
        short_id = f.stem[-4:]
        task_recv_evs, trainer_round_evs = [], []
        with open(f) as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ev = e.get("event")
                if ev == "task_recv":
                    task_recv_evs.append(e)
                elif ev == "trainer_round":
                    trainer_round_evs.append(e)
        result[short_id] = {
            "task_recv": task_recv_evs,
            "trainer_round": trainer_round_evs,
        }
    return result


def load_run_dir(run_dir: str) -> tuple:
    """Load aggregator + trainer telemetry from a run directory.

    Returns (agg_data, trainer_data).
    """
    telemetry_dir = os.path.join(run_dir, "telemetry")
    agg_files = sorted(glob.glob(os.path.join(telemetry_dir, "aggregator_*.jsonl")))
    if not agg_files:
        raise FileNotFoundError(f"No aggregator_*.jsonl in {telemetry_dir}")
    if len(agg_files) == 1:
        agg_data = load_agg_jsonl(agg_files[0])
    else:
        merged: dict = {"selection_train": [], "agg_rounds": [], "agg_evals": []}
        for f in agg_files:
            d = load_agg_jsonl(f)
            for k in merged:
                merged[k].extend(d[k])
        merged["selection_train"].sort(key=lambda x: (x["round"], x["ts"]))
        merged["agg_rounds"].sort(
            key=lambda x: (x["round"], x.get("agg_goal_count", 0), x["ts"]))
        merged["agg_evals"].sort(key=lambda x: x["round"])
        agg_data = merged
    trainer_data = load_trainer_jsonl_dir(telemetry_dir)
    return agg_data, trainer_data


# ═══════════════════════════════════════════════════════════════════
# §2  Internal helpers for per-round timeline extraction
# ═══════════════════════════════════════════════════════════════════

def _per_round_last_event(agg_rounds: list) -> dict:
    """Group agg_round events by FL round; keep last event per round (by ts)."""
    by_round: dict = {}
    for e in agg_rounds:
        r = e.get("round")
        if r is None:
            continue
        if r not in by_round or e.get("ts", 0) > by_round[r].get("ts", 0):
            by_round[r] = e
    return by_round


def _per_round_max_speed(agg_rounds: list) -> dict:
    """Per FL round: max trainer_speed_s across all commits in that round."""
    out: dict = {}
    for e in agg_rounds:
        r = e.get("round")
        if r is None:
            continue
        speeds = e.get("trainer_speed_s") or []
        if speeds:
            out[r] = max(out.get(r, 0.0), max(speeds))
    return out


def _per_round_advances(agg_rounds: list, use_vclock: bool) -> list:
    """Compute per-FL-round time advances.

    use_vclock=True:  Δvclock_now between consecutive rounds (sim mode).
    use_vclock=False: Δts (wall) between consecutive rounds (real mode).
    Returns list of positive advances.
    """
    by_round = _per_round_last_event(agg_rounds)
    rounds_sorted = sorted(by_round.keys())
    if len(rounds_sorted) < 2:
        return []
    advances = []
    for i in range(1, len(rounds_sorted)):
        e_prev = by_round[rounds_sorted[i - 1]]
        e_curr = by_round[rounds_sorted[i]]
        if use_vclock:
            v_prev = e_prev.get("vclock_now")
            v_curr = e_curr.get("vclock_now")
            if v_prev is None or v_curr is None:
                continue
            adv = v_curr - v_prev
        else:
            t_prev = e_prev.get("ts")
            t_curr = e_curr.get("ts")
            if t_prev is None or t_curr is None:
                continue
            adv = t_curr - t_prev
        if adv > 0:
            advances.append(adv)
    return advances


# ═══════════════════════════════════════════════════════════════════
# §3.B  Selection  (S1–S5)
# ═══════════════════════════════════════════════════════════════════

def _by_round_selection(selection_train: list) -> dict:
    out: dict = {}
    for e in selection_train:
        out.setdefault(e["round"], set()).update(e.get("chosen", []))
    return out


# Selectors whose per-round SET selection is a deterministic function of
# (candidate set, seed).  Currently empty — every shipped selector samples
# from a join-order-dependent candidate list, making exact per-round set
# identity unattainable across real/sim.  participation_parity is the
# enforced selection invariant for stochastic selectors.
DETERMINISTIC_SELECTORS: set = set()


def _selector_name(*loaded: dict) -> str:
    """Selector class name from selection telemetry; '' if unknown."""
    for d in loaded:
        for e in d.get("selection_train", []):
            name = e.get("selector")
            if name:
                return name
    return ""


def selection_parity(real: dict, sim: dict, max_rounds: Optional[int] = None,
                     warn_jaccard: float = 0.7) -> dict:
    """S1/S2: Per-round selection overlap (Jaccard).

    Enforced only for DETERMINISTIC_SELECTORS; gated to WARN for stochastic
    selectors (participation_parity is the enforced invariant for those).
    """
    r = _by_round_selection(real["selection_train"])
    s = _by_round_selection(sim["selection_train"])
    rounds = sorted(set(r) & set(s))
    if max_rounds is not None:
        rounds = [x for x in rounds if x <= max_rounds]
    js, exact = [], 0
    for rd in rounds:
        j = jaccard(r[rd], s[rd])
        js.append(j)
        if j == 1.0:
            exact += 1
    mean_j = sum(js) / len(js) if js else float("nan")
    selector = _selector_name(real, sim)
    gated = bool(selector) and selector not in DETERMINISTIC_SELECTORS
    enforced_ok = (not js) or mean_j >= warn_jaccard
    return {
        "ok": True if gated else enforced_ok,
        "tier": "DIST",
        "gated": gated,
        "selector": selector or None,
        "rounds_compared": len(rounds),
        "mean_jaccard": round(mean_j, 3) if js else None,
        "exact_match_frac": round(exact / len(js), 3) if js else None,
    }


# ═══════════════════════════════════════════════════════════════════
# §3.D  Updates received & ordering  (U1–U5)
# ═══════════════════════════════════════════════════════════════════

def aggregation_sequence_parity(real: dict, sim: dict,
                                 max_rounds: Optional[int] = None) -> dict:
    """P1 / U1: Per-round set of contributing trainers matches across modes."""
    def by_round(agg_rounds):
        out: dict = {}
        for e in agg_rounds:
            out.setdefault(e["round"], set()).update(e.get("contributing_trainers", []))
        return out

    r, s = by_round(real["agg_rounds"]), by_round(sim["agg_rounds"])
    rounds = sorted(set(r) & set(s))
    if max_rounds is not None:
        rounds = [x for x in rounds if x <= max_rounds]
    matches = sum(1 for rd in rounds if r[rd] == s[rd])
    return {
        "ok": (not rounds) or matches == len(rounds),
        "tier": "DIST",
        "rounds_compared": len(rounds),
        "exact_set_match_frac": round(matches / len(rounds), 3) if rounds else None,
    }


def staleness_parity(real: dict, sim: dict, warn_ks: float = 0.2,
                     warn_mean_diff: float = 1.0) -> dict:
    """U3: Staleness distributions match within tolerance."""
    def vals(agg_rounds):
        out = []
        for e in agg_rounds:
            out.extend(e.get("staleness", []))
        return out

    rv, sv = vals(real["agg_rounds"]), vals(sim["agg_rounds"])
    rm, _ = mean_std(rv)
    sm, _ = mean_std(sv)
    ks = ks_stat(rv, sv)
    mean_diff = abs(rm - sm) if not (math.isnan(rm) or math.isnan(sm)) else float("nan")
    ok = True
    if not math.isnan(ks):
        ok = ks <= warn_ks and (math.isnan(mean_diff) or mean_diff <= warn_mean_diff)
    nonneg = all(v >= 0 for v in rv + sv)
    return {
        "ok": ok and nonneg,
        "tier": "DIST",
        "real_mean": round(rm, 3) if not math.isnan(rm) else None,
        "sim_mean": round(sm, 3) if not math.isnan(sm) else None,
        "ks_stat": round(ks, 3) if not math.isnan(ks) else None,
        "all_nonnegative": nonneg,
    }


def commit_sequence(agg: dict) -> list:
    """U1 helper: mode-agnostic logical sequence of committed updates.

    One entry per aggregated update in (round, agg_goal_count) order.
    """
    evs = sorted(agg["agg_rounds"],
                 key=lambda e: (e["round"], e.get("agg_goal_count", 0)))
    seq = []
    for e in evs:
        ends = e.get("contributing_trainers", [])
        stales = e.get("staleness", [])
        for i, end in enumerate(ends):
            seq.append({
                "round": e["round"],
                "end": short(end),
                "staleness": stales[i] if i < len(stales) else None,
            })
    return seq


def first_divergence(real_agg: dict, sim_agg: dict, ctx: int = 2) -> dict:
    """U1: First index where real vs sim commit sequences differ (by end+round).

    index=None means sequences agree on the shared prefix.
    """
    rs, ss = commit_sequence(real_agg), commit_sequence(sim_agg)
    for i in range(min(len(rs), len(ss))):
        if (rs[i]["end"], rs[i]["round"]) != (ss[i]["end"], ss[i]["round"]):
            lo = max(0, i - ctx)
            return {"index": i, "real": rs[lo:i + ctx + 1],
                    "sim": ss[lo:i + ctx + 1],
                    "real_len": len(rs), "sim_len": len(ss)}
    return {"index": None, "real_len": len(rs), "sim_len": len(ss)}


def agg_goal_cycles_ok(agg: dict, agg_goal: int) -> dict:
    """U4: agg_goal_count within each round cycles 1..agg_goal (no lost/double-counted update)."""
    if agg_goal <= 0:
        return {"ok": True, "tier": "EXACT", "note": "agg_goal unknown"}
    bad_rounds = []
    by_round: dict = {}
    for e in agg["agg_rounds"]:
        by_round.setdefault(e["round"], []).append(e.get("agg_goal_count"))
    for rd, counts in by_round.items():
        present = [c for c in counts if c is not None]
        if present and max(present) > agg_goal:
            bad_rounds.append(rd)
    return {"ok": not bad_rounds, "tier": "EXACT",
            "rounds_over_goal": bad_rounds}


def inter_arrival_order_parity(real: dict, sim: dict,
                                min_rho: float = 0.7) -> dict:
    """U5: Rank order in which trainers' updates arrive within a round (Spearman ρ).

    For each FL round present in both, compute Spearman ρ between real and sim
    per-trainer arrival rank.  Mean ρ across rounds is the metric.
    """
    def arrival_ranks(agg_rounds):
        by_round: dict = {}
        for e in agg_rounds:
            r = e.get("round")
            if r is None:
                continue
            for t in e.get("contributing_trainers", []):
                by_round.setdefault(r, []).append(t)
        return by_round

    r_arr = arrival_ranks(real["agg_rounds"])
    s_arr = arrival_ranks(sim["agg_rounds"])
    common = sorted(set(r_arr) & set(s_arr))
    rhos = []
    for rd in common:
        rt, st = r_arr[rd], s_arr[rd]
        all_t = list(dict.fromkeys(rt + st))
        r_idx = [rt.index(t) if t in rt else len(rt) for t in all_t]
        s_idx = [st.index(t) if t in st else len(st) for t in all_t]
        rho = spearman_rho(r_idx, s_idx)
        if not math.isnan(rho):
            rhos.append(rho)
    mean_rho = sum(rhos) / len(rhos) if rhos else float("nan")
    ok = math.isnan(mean_rho) or mean_rho >= min_rho
    return {
        "ok": ok,
        "tier": "DIST",
        "mean_spearman_rho": round(mean_rho, 3) if not math.isnan(mean_rho) else None,
        "n_rounds": len(rhos),
        "min_rho": min_rho,
    }


# ═══════════════════════════════════════════════════════════════════
# §3.E  Update processing  (P1–P3)
# ═══════════════════════════════════════════════════════════════════

def participation_parity(real: dict, sim: dict, warn_avg_diff: float = 10.0) -> dict:
    """S2 / P1: Per-trainer participation counts match within tolerance."""
    def counts(agg_rounds):
        c = collections.Counter()
        for e in agg_rounds:
            for t in e.get("contributing_trainers", []):
                c[t] += 1
        return c

    rc, sc = counts(real["agg_rounds"]), counts(sim["agg_rounds"])
    trainers = set(rc) | set(sc)
    diffs = [abs(rc.get(t, 0) - sc.get(t, 0)) for t in trainers]
    avg = sum(diffs) / len(diffs) if diffs else 0.0
    return {
        "ok": avg <= warn_avg_diff,
        "tier": "DIST",
        "avg_diff": round(avg, 2),
        "max_diff": max(diffs) if diffs else 0,
    }


def trainer_speed_parity(real: dict, sim: dict, ks_tol: float = 0.1) -> dict:
    """P3: trainer_speed_s distributions match (control: proves speed model is identical).

    If this PASSES while K2/K3/K4 FAIL, the divergence is isolated to the
    sim clock advance model, not the trainer time model.
    """
    def all_speeds(agg_rounds):
        vals = []
        for e in agg_rounds:
            vals.extend(e.get("trainer_speed_s", []) or [])
        return [float(v) for v in vals if v is not None]

    real_speeds = all_speeds(real["agg_rounds"])
    sim_speeds = all_speeds(sim["agg_rounds"])
    if not real_speeds or not sim_speeds:
        return {"ok": True, "tier": "DIST", "status": "SKIP",
                "note": "no trainer_speed_s in telemetry"}
    ks = ks_stat(real_speeds, sim_speeds)
    real_mean, _ = mean_std(real_speeds)
    sim_mean, _ = mean_std(sim_speeds)
    ok = not math.isnan(ks) and ks <= ks_tol
    return {
        "ok": ok,
        "tier": "DIST",
        "ks_stat": round(ks, 3) if not math.isnan(ks) else None,
        "ks_tol": ks_tol,
        "real_mean_speed_s": round(real_mean, 2),
        "sim_mean_speed_s": round(sim_mean, 2),
        "real_max_speed_s": round(max(real_speeds), 2),
        "sim_max_speed_s": round(max(sim_speeds), 2),
        "n_real": len(real_speeds),
        "n_sim": len(sim_speeds),
    }


# ═══════════════════════════════════════════════════════════════════
# §3.F  Statistical utility  (F1–F3)
# ═══════════════════════════════════════════════════════════════════

def utility_parity(real: dict, sim: dict, max_ks: float = 0.2) -> dict:
    """F1/F2/F3: Per-trainer stat_utility distributions match."""
    def per_trainer_utils(agg_rounds):
        d: dict = collections.defaultdict(list)
        for e in agg_rounds:
            for t, u in zip(e.get("contributing_trainers", []),
                            e.get("stat_utility", [])):
                if u is not None:
                    d[t].append(u)
        return d

    r_utils = per_trainer_utils(real["agg_rounds"])
    s_utils = per_trainer_utils(sim["agg_rounds"])
    all_trainers = sorted(set(r_utils) | set(s_utils))
    ks_stats, mean_diffs = [], []
    for t in all_trainers:
        ru = r_utils.get(t, [])
        su = s_utils.get(t, [])
        ks = ks_stat(ru, su)
        rm, _ = mean_std(ru)
        sm, _ = mean_std(su)
        diff = abs(rm - sm) if not (math.isnan(rm) or math.isnan(sm)) else float("nan")
        if not math.isnan(ks):
            ks_stats.append(ks)
        if not math.isnan(diff):
            mean_diffs.append(diff)
    max_ks_val = max(ks_stats) if ks_stats else float("nan")
    avg_mean_diff = sum(mean_diffs) / len(mean_diffs) if mean_diffs else float("nan")
    ok = math.isnan(max_ks_val) or max_ks_val <= max_ks
    return {
        "ok": ok,
        "tier": "DIST",
        "max_ks_stat": round(max_ks_val, 3) if not math.isnan(max_ks_val) else None,
        "avg_mean_utility_diff": round(avg_mean_diff, 2) if not math.isnan(avg_mean_diff) else None,
        "n_trainers": len(all_trainers),
        "max_ks_tol": max_ks,
    }


# ═══════════════════════════════════════════════════════════════════
# §3.G  Convergence  (C1–C3)
# ═══════════════════════════════════════════════════════════════════

def convergence_parity(real: dict, sim: dict,
                        acc_tol: float = 0.05) -> dict:
    """C1/C2: Accuracy and loss curves aligned by FL round.

    C3 fix: the original compare_parity.py had a self-compare bug where
    sc was assigned from real["agg_evals"] before being overwritten with
    sim["agg_evals"].  This implementation uses sim directly.
    """
    def curve(agg_evals):
        return {e["round"]: {"acc": e.get("test-accuracy"), "loss": e.get("test-loss")}
                for e in agg_evals}

    rc = curve(real["agg_evals"])
    sc = curve(sim["agg_evals"])  # fix: no intermediate real assignment
    rounds = sorted(set(rc) & set(sc))
    if not rounds:
        return {"ok": True, "tier": "DIST", "status": "SKIP",
                "note": "no overlapping eval rounds"}
    acc_diffs, loss_diffs = [], []
    for r in rounds:
        ra, rl = rc[r].get("acc"), rc[r].get("loss")
        sa, sl = sc[r].get("acc"), sc[r].get("loss")
        if ra is not None and sa is not None:
            acc_diffs.append(abs(ra - sa))
        if rl is not None and sl is not None:
            loss_diffs.append(abs(rl - sl))
    avg_acc = sum(acc_diffs) / len(acc_diffs) if acc_diffs else float("nan")
    avg_loss = sum(loss_diffs) / len(loss_diffs) if loss_diffs else float("nan")
    ok = math.isnan(avg_acc) or avg_acc <= acc_tol
    return {
        "ok": ok,
        "tier": "DIST",
        "eval_rounds_compared": len(rounds),
        "avg_accuracy_diff": round(avg_acc, 4) if not math.isnan(avg_acc) else None,
        "avg_loss_diff": round(avg_loss, 4) if not math.isnan(avg_loss) else None,
        "acc_tol": acc_tol,
    }


# ═══════════════════════════════════════════════════════════════════
# §3.H  Clock & throughput  (K1–K10)  ← THE NEW ENFORCED CORE
# ═══════════════════════════════════════════════════════════════════

def vclock_telemetry_present(sim: dict) -> dict:
    """K10 [INV]: sim agg_round events must carry vclock_now.

    FAIL-LOUD when absent (sync path currently omits it) so K1–K3/K7 cannot
    silently skip on sync-baseline sim runs.
    """
    n_total = len(sim["agg_rounds"])
    n_with = sum(1 for e in sim["agg_rounds"] if e.get("vclock_now") is not None)
    if n_with == 0:
        return {
            "ok": False,
            "tier": "INV",
            "note": (
                "sim has ZERO vclock_now stamps on agg_round events. "
                "The sync aggregator path does not emit vclock_now. "
                "Fix: stamp vclock_now on agg_round events in the syncfl sim path. "
                "K1-K3/K7 CANNOT RUN on this sim run."
            ),
            "n_total_events": n_total,
            "n_with_vclock": 0,
        }
    return {
        "ok": True,
        "tier": "INV",
        "n_total_events": n_total,
        "n_with_vclock": n_with,
        "frac_with_vclock": round(n_with / n_total, 3) if n_total else 0,
    }


def sim_commit_order_monotone(sim: dict) -> dict:
    """K1 [INV]: sim vclock_now on agg_round events must be non-decreasing."""
    seq = [e.get("vclock_now") for e in sim["agg_rounds"]
           if e.get("vclock_now") is not None]
    if not seq:
        return {"ok": True, "tier": "INV", "status": "SKIP",
                "note": "no vclock_now stamps — K10 should catch this",
                "n_stamped": 0, "monotone": True}
    monotone = all(seq[i] <= seq[i + 1] + 1e-9 for i in range(len(seq) - 1))
    return {"ok": monotone, "tier": "INV", "n_stamped": len(seq),
            "monotone": monotone}


def sim_rate_ok(sim: dict, min_rate: float = 0.01, max_rate: float = 100.0) -> dict:
    """K7 [INV]: sim_rate = vclock / wall_sim must be in sane range [0.01, 100]."""
    vclock_vals = [e.get("vclock_now") for e in sim["agg_rounds"]
                   if e.get("vclock_now") is not None]
    ts_vals = [e["ts"] for e in sim["agg_rounds"] if e.get("ts") is not None]
    if not vclock_vals:
        return {"ok": True, "tier": "INV", "status": "SKIP",
                "note": "no vclock_now — K10 should catch this"}
    if len(ts_vals) < 2:
        return {"ok": True, "tier": "INV", "status": "SKIP",
                "note": "insufficient ts data"}
    final_vclock = max(vclock_vals)
    wall_elapsed = max(ts_vals) - min(ts_vals)
    if wall_elapsed <= 0:
        return {"ok": False, "tier": "INV", "note": "zero wall elapsed"}
    sim_rate = final_vclock / wall_elapsed
    ok = min_rate <= sim_rate <= max_rate
    return {
        "ok": ok,
        "tier": "INV",
        "sim_rate": round(sim_rate, 4),
        "final_vclock_s": round(final_vclock, 1),
        "wall_elapsed_s": round(wall_elapsed, 1),
        "range": [min_rate, max_rate],
    }


def failsafe_ok(sim: dict, budget_s: Optional[float] = None,
                max_overshoot: float = 0.20) -> dict:
    """K5 [INV]: sim wall must not overshoot sim_wall_ceiling_s by > 20%."""
    rounds = [e for e in sim["agg_rounds"] if e.get("event") == "agg_round"]
    all_evs = sim.get("_all_events", sim["agg_rounds"])  # agg_rounds used as proxy
    if len(rounds) < 2:
        return {"ok": True, "tier": "INV", "status": "SKIP",
                "note": "fewer than 2 agg_round events"}
    wall_elapsed = rounds[-1]["ts"] - rounds[0]["ts"]
    failsafe_fired = any(
        "SIM_WALL_CEILING" in str(e.get("stop_reason", "")) or
        "WALL_CLOCK_FAILSAFE" in str(e.get("stop_reason", ""))
        for e in all_evs
    )
    if budget_s is None:
        vclock_final = rounds[-1].get("vclock_now")
        if not vclock_final:
            return {"ok": True, "tier": "INV", "status": "SKIP",
                    "note": "no budget_s and no vclock_now — cannot compute overshoot"}
        budget_s = vclock_final
    overshoot = (wall_elapsed - budget_s) / budget_s if budget_s > 0 else 0
    ok = overshoot <= max_overshoot
    return {
        "ok": ok,
        "tier": "INV",
        "wall_elapsed_s": round(wall_elapsed, 1),
        "budget_s": round(budget_s, 1),
        "overshoot_frac": round(overshoot, 3),
        "failsafe_fired": failsafe_fired,
        "max_overshoot": max_overshoot,
    }


def throughput_parity(real: dict, sim: dict, tol_rel: float = 0.10) -> dict:
    """K2 [EXACT]: rounds-per-virtual-second parity.

    sim_throughput  = total_sim_rounds / final_vclock_sim
    real_throughput = total_real_rounds / wall_elapsed_real

    On the motivating Felix run (410 vs 673 rounds in the same 3 h budget)
    rel_diff ≈ 40% → FAIL (gate ≤ 10%).
    """
    sim_vclock_vals = [e.get("vclock_now") for e in sim["agg_rounds"]
                       if e.get("vclock_now") is not None]
    if not sim_vclock_vals:
        return {"ok": False, "tier": "EXACT",
                "note": "K10: no vclock_now in sim agg_round events — cannot compute throughput"}
    final_vclock = max(sim_vclock_vals)
    sim_by_round = _per_round_last_event(sim["agg_rounds"])
    n_sim_rounds = len(sim_by_round)
    sim_throughput = n_sim_rounds / final_vclock if final_vclock > 0 else 0.0

    real_ts = [e["ts"] for e in real["agg_rounds"] if e.get("ts") is not None]
    if not real_ts or len(real_ts) < 2:
        return {"ok": True, "tier": "EXACT", "status": "SKIP",
                "note": "insufficient real ts data (< 2 agg_round events)"}
    wall_elapsed = max(real_ts) - min(real_ts)
    real_by_round = _per_round_last_event(real["agg_rounds"])
    n_real_rounds = len(real_by_round)
    real_throughput = n_real_rounds / wall_elapsed if wall_elapsed > 0 else 0.0

    if wall_elapsed <= 0 or real_throughput == 0 or sim_throughput == 0:
        return {"ok": True, "tier": "EXACT", "status": "SKIP",
                "note": "zero wall elapsed or throughput — run too short to measure"}
    rel_diff = abs(sim_throughput - real_throughput) / max(sim_throughput, real_throughput)
    ok = rel_diff <= tol_rel
    sim_s_per_round = final_vclock / n_sim_rounds if n_sim_rounds else 0
    real_s_per_round = wall_elapsed / n_real_rounds if n_real_rounds else 0
    return {
        "ok": ok,
        "tier": "EXACT",
        "sim_rounds": n_sim_rounds,
        "real_rounds": n_real_rounds,
        "final_vclock_s": round(final_vclock, 1),
        "real_wall_elapsed_s": round(wall_elapsed, 1),
        "sim_s_per_round": round(sim_s_per_round, 2),
        "real_s_per_round": round(real_s_per_round, 2),
        "rel_diff": round(rel_diff, 3),
        "tol": tol_rel,
    }


def per_round_advance_parity(real: dict, sim: dict,
                              ks_tol: float = 0.2,
                              mean_tol_rel: float = 0.15) -> dict:
    """K3 [EXACT]: per-round virtual-advance distribution parity.

    sim Δvclock/round vs real Δwall/round — KS ≤ 0.2 AND mean diff ≤ 15%.
    On the motivating run (sim ≈ 26.4 s/round, real ≈ 15.6 s/round) → FAIL.
    """
    sim_adv = _per_round_advances(sim["agg_rounds"], use_vclock=True)
    real_adv = _per_round_advances(real["agg_rounds"], use_vclock=False)
    if not sim_adv:
        has_vclock = any(e.get("vclock_now") is not None for e in sim["agg_rounds"])
        note = ("K10: no vclock_now advances in sim agg_round events" if not has_vclock
                else "fewer than 2 sim rounds — run too short to measure advances")
        return {"ok": True, "tier": "EXACT", "status": "SKIP", "note": note}
    if not real_adv:
        return {"ok": True, "tier": "EXACT", "status": "SKIP",
                "note": "fewer than 2 real rounds — run too short to measure advances"}
    ks = ks_stat(sim_adv, real_adv)
    sim_mean, _ = mean_std(sim_adv)
    real_mean, _ = mean_std(real_adv)
    mean_rel_diff = (abs(sim_mean - real_mean) / max(sim_mean, real_mean)
                     if max(sim_mean, real_mean) > 0 else 0.0)
    ok = ks <= ks_tol and mean_rel_diff <= mean_tol_rel
    return {
        "ok": ok,
        "tier": "EXACT",
        "sim_mean_advance_s": round(sim_mean, 2),
        "real_mean_advance_s": round(real_mean, 2),
        "mean_rel_diff": round(mean_rel_diff, 3),
        "ks_stat": round(ks, 3),
        "ks_tol": ks_tol,
        "mean_tol_rel": mean_tol_rel,
        "n_sim_rounds": len(sim_adv),
        "n_real_rounds": len(real_adv),
    }


def overlap_factor(real: dict, sim: dict, tol: float = 0.3) -> dict:
    """K4 [DIAG]: async overlap factor diagnostic.

    overlap = mean(max_trainer_speed) / mean(per_round_advance).
    Real ≈ 1.8 (healthy async overlap); sim ≈ 1.06 (no inter-round overlap).
    FAIL if |sim_overlap - real_overlap| > 0.3 — localizes the bug to
    "sim does not model inter-round overlap".
    """
    sim_adv = _per_round_advances(sim["agg_rounds"], use_vclock=True)
    real_adv = _per_round_advances(real["agg_rounds"], use_vclock=False)
    sim_speeds = _per_round_max_speed(sim["agg_rounds"])
    real_speeds = _per_round_max_speed(real["agg_rounds"])
    if not sim_adv or not real_adv:
        return {"ok": True, "tier": "DIAG", "status": "SKIP",
                "note": "insufficient advance data (K10 may be blocking)"}
    sim_mean_adv = sum(sim_adv) / len(sim_adv)
    real_mean_adv = sum(real_adv) / len(real_adv)
    sim_mean_speed = (sum(sim_speeds.values()) / len(sim_speeds)
                      if sim_speeds else float("nan"))
    real_mean_speed = (sum(real_speeds.values()) / len(real_speeds)
                       if real_speeds else float("nan"))
    if sim_mean_adv == 0 or real_mean_adv == 0:
        return {"ok": False, "tier": "DIAG", "note": "zero advance in one mode"}
    sim_ov = sim_mean_speed / sim_mean_adv if not math.isnan(sim_mean_speed) else float("nan")
    real_ov = real_mean_speed / real_mean_adv if not math.isnan(real_mean_speed) else float("nan")
    if math.isnan(sim_ov) or math.isnan(real_ov):
        return {"ok": True, "tier": "DIAG", "status": "SKIP",
                "note": "no trainer_speed_s telemetry"}
    abs_diff = abs(sim_ov - real_ov)
    ok = abs_diff <= tol
    return {
        "ok": ok,
        "tier": "DIAG",
        "sim_overlap_factor": round(sim_ov, 3),
        "real_overlap_factor": round(real_ov, 3),
        "abs_diff": round(abs_diff, 3),
        "tol": tol,
        "sim_mean_speed_s": round(sim_mean_speed, 2) if not math.isnan(sim_mean_speed) else None,
        "real_mean_speed_s": round(real_mean_speed, 2) if not math.isnan(real_mean_speed) else None,
        "sim_mean_advance_s": round(sim_mean_adv, 2),
        "real_mean_advance_s": round(real_mean_adv, 2),
        "interpretation": (
            f"sim: {sim_mean_speed:.1f}s speed / {sim_mean_adv:.1f}s advance = "
            f"{sim_ov:.2f}x overlap; "
            f"real: {real_mean_speed:.1f}s speed / {real_mean_adv:.1f}s advance = "
            f"{real_ov:.2f}x overlap. "
            f"1.0 = no inter-round overlap; higher = more async pipelining."
        ),
    }


def total_commits_parity(real: dict, sim: dict, tol_rel: float = 0.02) -> dict:
    """U2 [EXACT]: total commits at matched virtual budget V = min(final_vclock, final_wall).

    abs diff ≤ 2% of commits.
    """
    sim_vclock_vals = [e.get("vclock_now") for e in sim["agg_rounds"]
                       if e.get("vclock_now") is not None]
    if not sim_vclock_vals:
        return {"ok": False, "tier": "EXACT",
                "note": "K10: no vclock_now in sim events"}
    final_sim_vclock = max(sim_vclock_vals)
    real_ts = [e["ts"] for e in real["agg_rounds"] if e.get("ts") is not None]
    if not real_ts:
        return {"ok": False, "tier": "EXACT", "note": "no ts in real events"}
    real_t0 = min(real_ts)
    final_real_wall = max(real_ts) - real_t0
    V = min(final_sim_vclock, final_real_wall)
    if V <= 0:
        return {"ok": True, "tier": "EXACT", "status": "SKIP",
                "note": "matched virtual budget V ≤ 0 — run too short to measure"}
    n_sim = sum(1 for e in sim["agg_rounds"] if (e.get("vclock_now") or 0) <= V + 1e-9)
    n_real = sum(1 for e in real["agg_rounds"]
                 if e.get("ts") is not None and (e["ts"] - real_t0) <= V + 1e-9)
    if max(n_sim, n_real, 1) == 0:
        return {"ok": True, "tier": "EXACT", "note": "no commits in V window"}
    rel_diff = abs(n_sim - n_real) / max(n_sim, n_real)
    ok = rel_diff <= tol_rel
    return {
        "ok": ok,
        "tier": "EXACT",
        "matched_virtual_budget_s": round(V, 1),
        "n_sim_commits": n_sim,
        "n_real_commits": n_real,
        "rel_diff": round(rel_diff, 4),
        "tol": tol_rel,
    }


def terminal_state_parity(real: dict, sim: dict,
                           rounds_tol: float = 0.10,
                           trainers_tol: float = 0.05) -> dict:
    """K8 [EXACT]: at matched virtual budget V, both modes have comparable FL-round count.

    rounds within 10%, unique trainers within 5%.
    """
    sim_vclock_vals = [e.get("vclock_now") for e in sim["agg_rounds"]
                       if e.get("vclock_now") is not None]
    if not sim_vclock_vals:
        return {"ok": False, "tier": "EXACT",
                "note": "K10: no vclock_now in sim events — cannot compute terminal state parity"}
    final_sim_vclock = max(sim_vclock_vals)
    real_ts_all = [e["ts"] for e in real["agg_rounds"] if e.get("ts") is not None]
    if not real_ts_all:
        return {"ok": False, "tier": "EXACT", "note": "no ts in real events"}
    real_t0 = min(real_ts_all)
    final_real_wall = max(real_ts_all) - real_t0
    V = min(final_sim_vclock, final_real_wall)
    if V <= 0:
        return {"ok": True, "tier": "EXACT", "status": "SKIP",
                "note": "matched virtual budget V ≤ 0 — run too short to measure"}

    sim_by_round = _per_round_last_event(sim["agg_rounds"])
    real_by_round = _per_round_last_event(real["agg_rounds"])

    sim_rounds_at_V = {r for r, e in sim_by_round.items()
                       if (e.get("vclock_now") or 0) <= V + 1e-9}
    real_rounds_at_V = {r for r, e in real_by_round.items()
                        if e.get("ts") is not None and (e["ts"] - real_t0) <= V + 1e-9}

    def _trainers(agg_rounds, round_set):
        ts = set()
        for e in agg_rounds:
            if e.get("round") in round_set:
                ts.update(e.get("contributing_trainers", []))
        return ts

    sim_trainers = _trainers(sim["agg_rounds"], sim_rounds_at_V)
    real_trainers = _trainers(real["agg_rounds"], real_rounds_at_V)
    n_sr, n_rr = len(sim_rounds_at_V), len(real_rounds_at_V)
    n_st, n_rt = len(sim_trainers), len(real_trainers)
    rounds_rel_diff = abs(n_sr - n_rr) / max(n_sr, n_rr, 1)
    trainers_rel_diff = abs(n_st - n_rt) / max(n_st, n_rt, 1)
    ok = rounds_rel_diff <= rounds_tol and trainers_rel_diff <= trainers_tol
    return {
        "ok": ok,
        "tier": "EXACT",
        "matched_virtual_budget_s": round(V, 1),
        "sim_rounds_at_V": n_sr,
        "real_rounds_at_V": n_rr,
        "rounds_rel_diff": round(rounds_rel_diff, 3),
        "rounds_tol": rounds_tol,
        "sim_trainers_at_V": n_st,
        "real_trainers_at_V": n_rt,
        "trainers_rel_diff": round(trainers_rel_diff, 3),
        "trainers_tol": trainers_tol,
    }


def budget_not_cap(real: dict, sim: dict,
                   rounds_cap: Optional[int] = None,
                   budget_s: Optional[float] = None) -> dict:
    """K9 [INV]: neither run should have stopped due to a rounds cap before budget.

    WARN if max_round == rounds_cap and wall/vclock < budget.
    """
    real_max_round = max((e.get("round", 0) for e in real["agg_rounds"]), default=0)
    sim_max_round = max((e.get("round", 0) for e in sim["agg_rounds"]), default=0)
    result: dict = {
        "ok": True,
        "tier": "INV",
        "real_max_round": real_max_round,
        "sim_max_round": sim_max_round,
    }
    if rounds_cap is None:
        result["note"] = "rounds_cap not provided — K9 skipped"
        return result
    warnings = []
    for mode, max_r, agg_r in [
        ("real", real_max_round, real["agg_rounds"]),
        ("sim", sim_max_round, sim["agg_rounds"]),
    ]:
        if max_r >= rounds_cap:
            # Check if wall/vclock also exhausted budget
            if mode == "sim":
                vclock_vals = [e.get("vclock_now") for e in agg_r
                               if e.get("vclock_now") is not None]
                t_used = max(vclock_vals) if vclock_vals else None
            else:
                ts_vals = [e["ts"] for e in agg_r if e.get("ts") is not None]
                t_used = (max(ts_vals) - min(ts_vals)) if len(ts_vals) >= 2 else None
            budget_used = t_used is not None and budget_s is not None
            if not budget_used or (budget_s is not None and t_used is not None
                                   and t_used < budget_s * 0.95):
                warnings.append(
                    f"{mode} hit rounds_cap={rounds_cap} (max_round={max_r}) "
                    f"before exhausting budget — comparison is truncated. "
                    f"Increase `rounds` config to let budget bind."
                )
    if warnings:
        result["ok"] = False  # treated as WARN in verdict rule
        result["warnings"] = warnings
    return result


# ═══════════════════════════════════════════════════════════════════
# §3.C  Sim-mode invariants  (K6 / T3 / T4)
# ═══════════════════════════════════════════════════════════════════

def sim_send_ts_ok(real_trainers: dict, sim_trainers: dict) -> dict:
    """K6 [INV]: real task_recv.sim_send_ts null; sim non-null and increasing (>0 after r1)."""
    issues = []
    for tid, data in real_trainers.items():
        bad = [e["sim_send_ts"] for e in data.get("task_recv", [])
               if e.get("sim_send_ts") is not None]
        if bad:
            issues.append(f"real/{tid}: unexpected non-null sim_send_ts {bad[:3]}")
    for tid, data in sim_trainers.items():
        evs = [e for e in data.get("task_recv", []) if e.get("round", 0) > 1]
        if not evs:
            continue
        vals = [e.get("sim_send_ts") for e in evs]
        if any(v is None for v in vals):
            issues.append(f"sim/{tid}: null sim_send_ts (vclock stamp missing)")
        elif not any((v or 0) > 0 for v in vals):
            issues.append(f"sim/{tid}: all sim_send_ts==0 (vclock not advancing)")
    return {"ok": not issues, "tier": "INV", "issues": issues}


def gpu_budget_ok(trainers: dict, warn_overrun_frac: float = 0.25) -> dict:
    """T3 [INV]: fraction of rounds where real_gpu_time_s exceeded the modeled budget."""
    fracs = []
    for _tid, d in trainers.items():
        evs = [e for e in d.get("trainer_round", [])
               if "real_gpu_time_s" in e and e.get("training_budget_s", 0) > 0]
        if not evs:
            continue
        over = sum(1 for e in evs if e["real_gpu_time_s"] > e["training_budget_s"])
        fracs.append(over / len(evs))
    if not fracs:
        return {"ok": True, "tier": "INV",
                "note": "no training_budget_s telemetry", "mean_overrun_frac": None}
    mean_frac = sum(fracs) / len(fracs)
    return {
        "ok": mean_frac <= warn_overrun_frac,
        "tier": "INV",
        "mean_overrun_frac": round(mean_frac, 3),
        "trainers_with_any_overrun": int(sum(1 for f in fracs if f > 0)),
    }


# ═══════════════════════════════════════════════════════════════════
# §4  Consolidated run_all_parity (extended)
# ═══════════════════════════════════════════════════════════════════

def run_all_parity(real_agg: dict, sim_agg: dict,
                   real_trainers: dict, sim_trainers: dict,
                   agg_goal: int = 0,
                   max_rounds: Optional[int] = None,
                   rounds_cap: Optional[int] = None,
                   budget_s: Optional[float] = None) -> dict:
    """Run the full parity + invariant battery; returns {name: result_dict}.

    Extended from the original to include the §3.H clock/throughput checks
    (K2/K3/K4/K8/K10/U2/P3) alongside the original checks.
    Backward-compatible: callers that pass only real_agg/sim_agg/trainers/agg_goal
    still get the original results plus the new ones.
    """
    results: dict = {}

    # ── §3.H: clock gate — check K10 first so downstream clock checks can SKIP ──
    results["vclock_telemetry"] = vclock_telemetry_present(sim_agg)

    # ── §3.H: clock invariants ──
    results["sim_commit_monotone"] = sim_commit_order_monotone(sim_agg)
    results["sim_rate"] = sim_rate_ok(sim_agg)

    # ── §3.H: parity checks (headline new checks) ──
    results["throughput"] = throughput_parity(real_agg, sim_agg)
    results["per_round_advance"] = per_round_advance_parity(real_agg, sim_agg)
    results["overlap_factor"] = overlap_factor(real_agg, sim_agg)
    results["total_commits"] = total_commits_parity(real_agg, sim_agg)
    results["terminal_state"] = terminal_state_parity(real_agg, sim_agg)
    results["budget_not_cap"] = budget_not_cap(
        real_agg, sim_agg, rounds_cap=rounds_cap, budget_s=budget_s)

    # ── §3.B: selection ──
    results["selection"] = selection_parity(real_agg, sim_agg, max_rounds)

    # ── §3.D: updates ──
    results["aggregation_sequence"] = aggregation_sequence_parity(
        real_agg, sim_agg, max_rounds)
    results["staleness"] = staleness_parity(real_agg, sim_agg)
    results["inter_arrival_order"] = inter_arrival_order_parity(real_agg, sim_agg)

    # ── §3.E: processing ──
    results["participation"] = participation_parity(real_agg, sim_agg)
    results["trainer_speed"] = trainer_speed_parity(real_agg, sim_agg)

    # ── §3.F: utility ──
    results["utility"] = utility_parity(real_agg, sim_agg)

    # ── §3.G: convergence ──
    results["convergence"] = convergence_parity(real_agg, sim_agg)

    # ── §3.C: sim-mode trainer invariants ──
    results["sim_send_ts"] = sim_send_ts_ok(real_trainers, sim_trainers)
    results["gpu_budget_real"] = gpu_budget_ok(real_trainers)
    results["gpu_budget_sim"] = gpu_budget_ok(sim_trainers)

    if agg_goal:
        results["agg_goal_cycles_real"] = agg_goal_cycles_ok(real_agg, agg_goal)
        results["agg_goal_cycles_sim"] = agg_goal_cycles_ok(sim_agg, agg_goal)

    return results


# ═══════════════════════════════════════════════════════════════════
# §5  Overall verdict
# ═══════════════════════════════════════════════════════════════════

_WARN_ONLY_CHECKS = {"budget_not_cap", "overlap_factor", "inter_arrival_order"}
_DIST_CHECKS = {"selection", "aggregation_sequence", "staleness", "participation",
                "trainer_speed", "utility", "convergence", "per_round_advance",
                "inter_arrival_order"}


def overall_verdict(results: dict, strict: bool = False,
                    lenient: bool = False) -> tuple:
    """Return (passed: bool, failures: list[str], warnings: list[str]).

    Enforcement:
      EXACT/INV FAIL  → always overall FAIL
      DIST FAIL       → FAIL unless --lenient
      WARN-only       → never FAIL unless --strict
      DIAG FAIL       → always a warning (never FAIL unless --strict)
    """
    failures, warnings = [], []
    for name, res in results.items():
        if res.get("ok"):
            continue
        tier = res.get("tier", "DIST")
        if name in _WARN_ONLY_CHECKS or tier == "DIAG":
            if strict:
                failures.append(name)
            else:
                warnings.append(name)
        elif tier in ("EXACT", "INV"):
            failures.append(name)
        elif tier == "DIST":
            if lenient:
                warnings.append(name)
            else:
                failures.append(name)
        else:
            failures.append(name)
    return not bool(failures), failures, warnings
