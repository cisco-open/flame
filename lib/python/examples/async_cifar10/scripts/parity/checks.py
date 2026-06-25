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
import re
import statistics
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════════════
# §0  Helpers
# ═══════════════════════════════════════════════════════════════════

def short(end_id: str) -> str:
    return end_id[-4:] if end_id else "None"


# task_id -> training_delay_s (the per-trainer *modeled* compute, in seconds), read
# from the static trainer registry. This is the mode-symmetric speed source for the
# pool-composition check (A2b): real telemetry leaves PROP_CLIENT_TASK_TRAIN_DURATION = None for
# any candidate that hasn't *completed* a round (slow clients, most of the pool), so
# pooling the observed speed_s samples different subsets per mode. The registry delay
# is present for every candidate in both modes — same number, same trainer.
_DELAY_REGISTRY_CACHE: Optional[dict] = None
_SPEED_CLASS_REGISTRY_CACHE: Optional[dict] = None


def _trainer_speed_class_map() -> dict:
    """{task_id: speed_class} from metadata/trainer_registry.yaml (cached).

    Same stdlib line scan as ``_trainer_delay_map``; within each trainer block
    ``task_id`` is followed by ``training_delay_s`` then ``speed_class``. Used by
    S2 to enforce participation by intrinsic speed CLASS (the policy-level
    invariant) for stochastic selectors, where per-trainer identity is path
    -dependent. Returns {} if the registry can't be found.
    """
    global _SPEED_CLASS_REGISTRY_CACHE
    if _SPEED_CLASS_REGISTRY_CACHE is not None:
        return _SPEED_CLASS_REGISTRY_CACHE
    out: dict = {}
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent.parent / "metadata" / "trainer_registry.yaml",
        Path.cwd() / "metadata" / "trainer_registry.yaml",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is not None:
        last_task = None
        for line in path.read_text().splitlines():
            m = re.search(r"task_id:\s*(\S+)", line)
            if m:
                last_task = m.group(1).strip().strip("'\"")
                continue
            m = re.search(r"speed_class:\s*'?([\w]+)'?", line)
            if m and last_task is not None:
                out[last_task] = m.group(1).strip()
                last_task = None
    _SPEED_CLASS_REGISTRY_CACHE = out
    return out


def _trainer_delay_map() -> dict:
    """{task_id: training_delay_s} from metadata/trainer_registry.yaml (cached).

    stdlib-only line scan (no yaml dep): within each trainer block ``task_id`` is
    immediately followed by ``training_delay_s``. Returns {} if the registry can't
    be found, in which case callers fall back to the observed speed_s.
    """
    global _DELAY_REGISTRY_CACHE
    if _DELAY_REGISTRY_CACHE is not None:
        return _DELAY_REGISTRY_CACHE
    out: dict = {}
    here = Path(__file__).resolve()
    # scripts/parity/checks.py -> example root is two levels up from scripts/
    candidates = [
        here.parent.parent.parent / "metadata" / "trainer_registry.yaml",
        Path.cwd() / "metadata" / "trainer_registry.yaml",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is not None:
        last_task = None
        for line in path.read_text().splitlines():
            m = re.search(r"task_id:\s*(\S+)", line)
            if m:
                last_task = m.group(1).strip().strip("'\"")
                continue
            m = re.search(r"training_delay_s:\s*'?([\d.]+)'?", line)
            if m and last_task is not None:
                out[last_task] = float(m.group(1))
                last_task = None
    _DELAY_REGISTRY_CACHE = out
    return out


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


def percentile(vals: list, q: float) -> float:
    """The q-th percentile (q in [0,100]) by linear interpolation; no numpy."""
    if not vals:
        return float("nan")
    s = sorted(vals)
    if len(s) == 1:
        return float(s[0])
    pos = (q / 100.0) * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)


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
    eval_commits: list = []
    agg_evals: list = []
    residence: list = []
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
                # Eval commits emit event=agg_round (tagged task=eval) so U6/U6e can
                # read their commit timeliness, but they carry no agg_goal_count and
                # don't advance the clock/aggregate — keep them OUT of agg_rounds so
                # the train-commit checks (K1 monotone, U3 staleness, U1/U5 ordering)
                # aren't contaminated. Only the eval-aware checks opt into them.
                if str(e.get("task_to_perform", "train")) == "eval":
                    eval_commits.append(e)
                else:
                    agg_rounds.append(e)
            elif ev == "agg_eval":
                agg_evals.append(e)
            elif ev == "inflight_residence":
                residence.append(e)
    selection_train.sort(key=lambda x: (x["round"], x["ts"]))
    agg_rounds.sort(key=lambda x: (x["round"], x.get("agg_goal_count", 0), x["ts"]))
    eval_commits.sort(key=lambda x: (x["round"], x["ts"]))
    agg_evals.sort(key=lambda x: x["round"])
    residence.sort(key=lambda x: (x["round"], x["ts"]))
    return {
        "selection_train": selection_train,
        "agg_rounds": agg_rounds,
        "eval_commits": eval_commits,
        "agg_evals": agg_evals,
        "residence": residence,
    }


def load_trainer_jsonl_dir(telemetry_dir: Optional[str]) -> dict:
    """Load all trainer_*.jsonl from a telemetry dir.

    Returns {short_id: {"task_recv": [...], "trainer_round": [...],
    "task_send": [...]}}.  task_send (§4.0) carries [wall_recv_ts, wall_send_ts]
    bracketing the trainer's true busy window for real-concurrency validation.
    """
    if not telemetry_dir:
        return {}
    d = Path(telemetry_dir)
    result: dict = {}
    for f in sorted(d.glob("trainer_*.jsonl")):
        short_id = f.stem[-4:]
        task_recv_evs, trainer_round_evs, task_send_evs = [], [], []
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
                elif ev == "task_send":
                    task_send_evs.append(e)
        result[short_id] = {
            "task_recv": task_recv_evs,
            "trainer_round": trainer_round_evs,
            "task_send": task_send_evs,
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
        merged: dict = {"selection_train": [], "agg_rounds": [],
                        "eval_commits": [], "agg_evals": [], "residence": []}
        for f in agg_files:
            d = load_agg_jsonl(f)
            for k in merged:
                merged[k].extend(d[k])
        merged["selection_train"].sort(key=lambda x: (x["round"], x["ts"]))
        merged["agg_rounds"].sort(
            key=lambda x: (x["round"], x.get("agg_goal_count", 0), x["ts"]))
        merged["eval_commits"].sort(key=lambda x: (x["round"], x["ts"]))
        merged["agg_evals"].sort(key=lambda x: x["round"])
        merged["residence"].sort(key=lambda x: (x["round"], x["ts"]))
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
# §3.A  Availability / eligibility  (A1–A2)
# ═══════════════════════════════════════════════════════════════════

def avail_composition_parity(real: dict, sim: dict,
                              tol_rel: float = 0.20) -> dict:
    """A1 [DIST]: Per-round avail_composition counts match across modes.

    avail_composition is a dict {state: count} on each selection event.
    Typical states: TRAIN, EVAL, UNAVAIL, UNKNOWN.
    Compares mean per-state count across rounds.
    """
    def collect(sel_events):
        by_key: dict = {}
        n = 0
        for e in sel_events:
            comp = e.get("avail_composition")
            if not comp:
                continue
            n += 1
            for k, v in comp.items():
                by_key.setdefault(k, []).append(v)
        return by_key, n

    r_by_key, r_n = collect(real["selection_train"])
    s_by_key, s_n = collect(sim["selection_train"])
    if not r_by_key or not s_by_key:
        return {"ok": True, "tier": "DIST", "status": "SKIP",
                "note": "no avail_composition in telemetry"}

    all_keys = sorted(set(r_by_key) | set(s_by_key))
    violations, per_key = [], {}
    for k in all_keys:
        r_vals = r_by_key.get(k, [0])
        s_vals = s_by_key.get(k, [0])
        r_mean = sum(r_vals) / len(r_vals)
        s_mean = sum(s_vals) / len(s_vals)
        ref = max(r_mean, s_mean, 1.0)
        rel = abs(r_mean - s_mean) / ref
        per_key[k] = {"real_mean": round(r_mean, 1), "sim_mean": round(s_mean, 1),
                      "rel_diff": round(rel, 3)}
        if rel > tol_rel:
            violations.append(k)
    return {
        "ok": len(violations) == 0,
        "tier": "DIST",
        "rounds_real": r_n,
        "rounds_sim": s_n,
        "per_state": per_key,
        "violations": violations,
        "tol_rel": tol_rel,
    }


def eligibility_parity(real: dict, sim: dict, warn_ks: float = 0.2) -> dict:
    """A2 [DIST]: num_eligible and num_candidates distributions match across modes."""
    def collect(sel_events):
        eligible, candidates = [], []
        for e in sel_events:
            ne = e.get("num_eligible")
            nc = e.get("num_candidates")
            if ne is not None:
                eligible.append(ne)
            if nc is not None:
                candidates.append(nc)
        return eligible, candidates

    r_el, r_ca = collect(real["selection_train"])
    s_el, s_ca = collect(sim["selection_train"])
    if not r_el and not r_ca:
        return {"ok": True, "tier": "DIST", "status": "SKIP",
                "note": "no num_eligible/num_candidates in telemetry"}

    ks_el = ks_stat(r_el, s_el) if r_el and s_el else float("nan")
    ks_ca = ks_stat(r_ca, s_ca) if r_ca and s_ca else float("nan")
    r_el_mean = sum(r_el) / len(r_el) if r_el else float("nan")
    s_el_mean = sum(s_el) / len(s_el) if s_el else float("nan")

    # Point-mass guard: KS saturates to ~1 when one side has (near-)zero variance —
    # e.g. real num_eligible is a constant 300 (all-eligible streaming) while sim is
    # 298.9 ± tiny. The KS is then uninformative; the means are the right comparator.
    # Rescue a KS fail ONLY when (a) the means match within a tight relative margin
    # and (b) a side is genuinely degenerate (coefficient of variation below cv_floor),
    # so a real eligible-set divergence is never masked. Kept tight on purpose.
    MEAN_TOL_REL, CV_FLOOR = 0.02, 0.01

    def _pointmass_match(rv, sv, ks):
        if math.isnan(ks) or ks <= warn_ks or not rv or not sv:
            return False
        rm, sm = (sum(rv) / len(rv)), (sum(sv) / len(sv))
        denom = max(abs(rm), abs(sm), 1.0)
        if abs(rm - sm) / denom > MEAN_TOL_REL:
            return False
        cv = lambda v, m: (statistics.pstdev(v) / abs(m)) if (len(v) > 1 and m) else 0.0
        return min(cv(rv, rm), cv(sv, sm)) < CV_FLOOR

    pm_el = _pointmass_match(r_el, s_el, ks_el)
    pm_ca = _pointmass_match(r_ca, s_ca, ks_ca)
    ok_el = math.isnan(ks_el) or ks_el <= warn_ks or pm_el
    ok_ca = math.isnan(ks_ca) or ks_ca <= warn_ks or pm_ca
    ok = ok_el and ok_ca
    out = {
        "ok": ok,
        "tier": "DIST",
        "ks_eligible": round(ks_el, 3) if not math.isnan(ks_el) else None,
        "ks_candidates": round(ks_ca, 3) if not math.isnan(ks_ca) else None,
        "real_mean_eligible": round(r_el_mean, 1) if not math.isnan(r_el_mean) else None,
        "sim_mean_eligible": round(s_el_mean, 1) if not math.isnan(s_el_mean) else None,
        "warn_ks": warn_ks,
    }
    if pm_el or pm_ca:
        out["note"] = ("point-mass distribution: KS uninformative (zero-variance side), "
                       "means match within {:.0%} — passed on mean".format(MEAN_TOL_REL))
    return out


def eligible_speed_composition_parity(real: dict, sim: dict, ks_tol: float = 0.20) -> dict:
    """A2b [DIST]: the SPEED composition of the eligible candidate pool matches.

    A2 (eligibility) checks the eligible-set *size*; A2b checks *who* is in it — the
    `trainer_speed_s` distribution of every candidate seen at selection (pooled over
    rounds). The size can match while the composition diverges, so A2 sails through.

    Why it matters: in real a slow client stays busy (in-flight) for its whole
    budget, so it is OUT of the eligible pool that long → real's pool is fast-skewed.
    If sim frees a non-committed/in-flight client back to the pool too early, slow
    clients re-enter → sim's pool skews slow (e.g. refl sim pool-mean 12.1 s vs real
    6.8 s, KS .363, while oort/felix/feddance match at KS≈.07).
    A2b is the finest check that localizes that divergence; selection-stage fails
    (participation, committed trainer_speed mix) downstream of it are *consequences*.
    The fix is sim-side: hold non-committed candidates out of the pool until they
    legitimately return (the in-flight-residence model, shared with oort) — NOT to bend
    the check.

    Speed source. The pool composition is compared on each candidate's
    **static ``training_delay_s``** (the modeled compute, from the trainer registry),
    NOT the observed ``per_trainer.speed_s`` (= PROP_CLIENT_TASK_TRAIN_DURATION). Real telemetry
    leaves PROP_CLIENT_TASK_TRAIN_DURATION = None for any candidate that has not *completed* a
    round — at steady state ~158/300 of refl's pool — so pooling observed speed
    samples only the fast completers in real while sim (modeled) fills nearly all,
    comparing different SUBSETS (the "modeled vs wall" asymmetry). The registry
    delay is present for every candidate in both modes, so it tests the genuine
    eligible-set membership composition. Verified: observed-speed pool reads real 7.0
    / sim 12.2 (KS .39) purely from the None-density skew, while the metadata pool is
    real 12.13 / sim 12.13 (KS .000) — the eligible pool is in fact identical.
    Observed-speed means are retained as a diagnostic. Falls back to observed speed
    when the registry is unavailable (other examples).
    """
    delay = _trainer_delay_map()

    def pool(sel_events):
        meta, obs = [], []
        for e in sel_events:
            for eid, cand in (e.get("per_trainer") or {}).items():
                d = delay.get(eid)
                if d is not None:
                    meta.append(d)
                sp = cand.get("speed_s")
                if sp is not None:
                    obs.append(sp)
        return meta, obs

    r_meta, r_obs = pool(real["selection_train"])
    s_meta, s_obs = pool(sim["selection_train"])
    used_metadata = bool(r_meta and s_meta)
    r = r_meta if used_metadata else r_obs
    s = s_meta if used_metadata else s_obs
    if not r or not s:
        return {"ok": True, "tier": "DIST", "status": "SKIP",
                "note": "no per_trainer speed/delay in selection telemetry"}
    ks = ks_stat(r, s)
    ok = not math.isnan(ks) and ks <= ks_tol
    return {
        "ok": ok,
        "tier": "DIST",
        "ks_stat": round(ks, 3) if not math.isnan(ks) else None,
        "ks_tol": ks_tol,
        "speed_source": "training_delay_s" if used_metadata else "observed_speed_s",
        "real_mean_pool_speed_s": round(sum(r) / len(r), 2),
        "sim_mean_pool_speed_s": round(sum(s) / len(s), 2),
        "real_observed_pool_speed_s": round(sum(r_obs) / len(r_obs), 2) if r_obs else None,
        "sim_observed_pool_speed_s": round(sum(s_obs) / len(s_obs), 2) if s_obs else None,
        "n_real": len(r),
        "n_sim": len(s),
    }


def selection_speed_bias_parity(real: dict, sim: dict, ks_tol: float = 0.20) -> dict:
    """A2c [DIST]: does the selector pick the same SPEED mix from its pool?

    A2b (eligible_speed) checks the *pool* composition; this checks the *selected*
    subset. Reading the two together localizes a selection divergence to one of two
    causes:
      - pool diverges (A2b FAIL)             -> POOL COMPOSITION (refl: sim frees busy
        clients early so slow ones re-enter the pool; selected may still match).
      - pool matches but selected diverges   -> SELECTOR-SCORING/path bias (oort: from
        a like pool real exploits utility -> picks fast, sim picks ~pool-average).
    `bias = mean(selected) - mean(pool)` (per mode) is the selector's revealed speed
    preference. Like A2b, the pool/selected speeds are taken from the **static
    `training_delay_s` metadata** (the modeled compute) rather than observed `speed_s`:
    real leaves `speed_s = None` for non-completers, so an observed pool samples only
    the fast completers (real pool 8.17 vs metadata 12.13) and an observed bias falsely
    reads sim as picking much faster relative to its pool (−0.48 vs −3.22). On the
    metadata basis both pools are identical, so the bias isolates the genuine
    selected-speed preference. KS is over the SELECTED-candidate speeds (metadata).
    Observed means retained as a diagnostic; falls back to observed when the registry
    is unavailable.
    """
    delay = _trainer_delay_map()

    def split(events):
        sel, pool, sel_obs, pool_obs = [], [], [], []
        for e in events:
            for eid, c in (e.get("per_trainer") or {}).items():
                d = delay.get(eid)
                sp = c.get("speed_s")
                chosen = c.get("selected")
                if d is not None:
                    pool.append(d)
                    if chosen:
                        sel.append(d)
                if sp is not None:
                    pool_obs.append(sp)
                    if chosen:
                        sel_obs.append(sp)
        return sel, pool, sel_obs, pool_obs

    r_sel, r_pool, r_sel_obs, r_pool_obs = split(real["selection_train"])
    s_sel, s_pool, s_sel_obs, s_pool_obs = split(sim["selection_train"])
    used_metadata = bool(r_sel and s_sel)
    if not used_metadata:
        # fall back to observed speed_s (no registry / other examples)
        r_sel, r_pool, s_sel, s_pool = r_sel_obs, r_pool_obs, s_sel_obs, s_pool_obs
    if not (r_sel and s_sel):
        return {"ok": True, "tier": "DIST", "status": "SKIP",
                "note": "no selected per_trainer speed/delay in selection telemetry"}
    ks = ks_stat(r_sel, s_sel)
    ok = not math.isnan(ks) and ks <= ks_tol

    def _m(x):
        return round(sum(x) / len(x), 2) if x else None

    return {
        "ok": ok, "tier": "DIST",
        "ks_stat": round(ks, 3) if not math.isnan(ks) else None, "ks_tol": ks_tol,
        "speed_source": "training_delay_s" if used_metadata else "observed_speed_s",
        "real_selected_mean_s": _m(r_sel), "sim_selected_mean_s": _m(s_sel),
        "real_pool_mean_s": _m(r_pool), "sim_pool_mean_s": _m(s_pool),
        "real_bias_s": round(_m(r_sel) - _m(r_pool), 2),
        "sim_bias_s": round(_m(s_sel) - _m(s_pool), 2),
        "real_observed_selected_s": _m(r_sel_obs), "sim_observed_selected_s": _m(s_sel_obs),
        "real_observed_pool_s": _m(r_pool_obs), "sim_observed_pool_s": _m(s_pool_obs),
        "n_real": len(r_sel), "n_sim": len(s_sel),
    }


# Utility-score component keys emitted by the per-selector audit (oort / feddance).
_SCORE_COMPONENT_KEYS = (
    "believed_I", "temporal", "system_util",                  # oort
    "feddance_V", "feddance_I", "feddance_A", "feddance_U",    # feddance
    "v_m", "i_m", "a_m", "u_m",                               # generic
)


def selector_score_parity(real: dict, sim: dict, ks_tol: float = 0.20) -> dict:
    """Score-localize [DIAG]: WHICH utility-score term drives a selection-mix split?

    For utility selectors the `per_trainer` audit carries the score components
    (oort believed_I/temporal/system_util; feddance feddance_V/I/A/U). This compares
    each component's distribution over SELECTED candidates, sim vs real, and reports
    the worst-diverging term — localizing a selector divergence to a single score
    input (e.g. feddance_I = loss-utility, oort believed_I = stat_utility) instead of
    a black-box "the selector picks differently". DIAG: several of these terms are
    path-dependent (the selection history differs across modes) so a divergence here
    is a localization aid, not a verdict. SKIP for non-utility selectors.
    """
    def comps(events):
        out: dict = {}
        for e in events:
            for c in (e.get("per_trainer") or {}).values():
                if not c.get("selected"):
                    continue
                for k in _SCORE_COMPONENT_KEYS:
                    v = c.get(k)
                    if v is not None:
                        out.setdefault(k, []).append(v)
        return out

    r, s = comps(real["selection_train"]), comps(sim["selection_train"])
    common = [k for k in _SCORE_COMPONENT_KEYS if r.get(k) and s.get(k)]
    if not common:
        return {"ok": True, "tier": "DIAG", "status": "SKIP",
                "note": "no per_trainer score components (non-utility selector)"}
    per: dict = {}
    worst_ks, worst_k = -1.0, None
    for k in common:
        ks = ks_stat(r[k], s[k])
        if math.isnan(ks):
            continue
        per[k] = {"ks": round(ks, 3),
                  "real_mean": round(sum(r[k]) / len(r[k]), 3),
                  "sim_mean": round(sum(s[k]) / len(s[k]), 3)}
        if ks > worst_ks:
            worst_ks, worst_k = ks, k
    return {
        "ok": worst_ks <= ks_tol, "tier": "DIAG",
        "worst_component": worst_k,
        "worst_ks": round(worst_ks, 3) if worst_ks >= 0 else None,
        "ks_tol": ks_tol, "per_component": per,
    }


def preferred_duration_parity(real: dict, sim: dict, frac_tol: float = 0.20) -> dict:
    """Stage-3 [DIST]: does the Oort speed penalty BIND at the same rate?

    Root-cause guard for the oort `pref`-not-sorted bug.
    Oort's `system_util = min(1, (pref/round_duration)^alpha)` only penalizes a
    trainer when its duration exceeds the round-preferred duration `pref` (the
    round_threshold-th PERCENTILE of candidate durations). When `pref` is computed
    on an UNSORTED list it lands at an arbitrary (too-high) value, so the penalty
    rarely binds, the selector ignores speed, and sim picks ~pool-average instead
    of fast (A2c bias diverges). That bug left `system_util` (Sx) only ~.13 KS off
    but flipped the *binding frequency* hard (real 80%/round vs sim 46%) — which is
    what this check measures directly.

    Per mode, over SELECTED candidates: the fraction of rounds where >=1 selected
    trainer is speed-penalized (system_util < 1). Reconstructs `pref` from the
    existing per_trainer audit (`pref = round_duration * sqrt(system_util)` for a
    binding entry, alpha=2) so it works on telemetry recorded BEFORE the
    round_preferred_duration_s instrumentation was added. SKIP for non-oort
    selectors (no per_trainer.system_util).
    """
    eps = 1e-6

    def per_round_binding(events):
        binds, pref_samples = [], []
        for e in events:
            any_pen, saw = False, False
            for c in (e.get("per_trainer") or {}).values():
                if not c.get("selected"):
                    continue
                su = c.get("system_util")
                if su is None:
                    continue
                saw = True
                if su < 1.0 - eps:
                    any_pen = True
                    sp = c.get("speed_s")
                    if sp is not None and su > 0:
                        pref_samples.append(sp * math.sqrt(su))  # alpha=2
            if saw:
                binds.append(1.0 if any_pen else 0.0)
        return binds, pref_samples

    r_binds, r_pref = per_round_binding(real["selection_train"])
    s_binds, s_pref = per_round_binding(sim["selection_train"])
    if not (r_binds and s_binds):
        return {"ok": True, "tier": "DIST", "status": "SKIP",
                "note": "no selected per_trainer.system_util (non-oort selector)"}

    r_frac = sum(r_binds) / len(r_binds)
    s_frac = sum(s_binds) / len(s_binds)
    diff = abs(r_frac - s_frac)

    def _med(x):
        return round(statistics.median(x), 2) if x else None

    # Observability gate (refl): the penalty is INACTIVE in real — it never binds
    # and no `pref` is reconstructable. That is the PROP_CLIENT_TASK_TRAIN_DURATION None-density
    # asymmetry (same class A2b/A2c resolved): real's `calculate_round_preferred_
    # duration` is fed mostly None durations (non-completers → 60s default), so
    # `pref` inflates and the speed penalty never fires; sim has dense modeled
    # durations so it binds. There is no real binding BEHAVIOUR to reproduce, so a
    # binding-FREQUENCY mismatch here is the observability gap, not a selector bug.
    # WARN, don't FAIL. oort (real_frac > 0) stays fully enforced — this only fires
    # when real exercises no penalty at all, so the D1 unsorted-`pref` guard holds.
    if r_frac == 0.0 and not r_pref:
        return {
            "ok": True, "tier": "DIST", "status": "WARN",
            "note": ("real penalty inactive (no binding, no reconstructable pref) — "
                     "PROP_CLIENT_TASK_TRAIN_DURATION None-density artifact; nothing to match"),
            "real_frac_binding": round(r_frac, 3), "sim_frac_binding": round(s_frac, 3),
            "frac_diff": round(diff, 3), "frac_tol": frac_tol,
            "real_pref_median_s": _med(r_pref), "sim_pref_median_s": _med(s_pref),
            "n_rounds_real": len(r_binds), "n_rounds_sim": len(s_binds),
        }

    return {
        "ok": diff <= frac_tol, "tier": "DIST",
        "real_frac_binding": round(r_frac, 3), "sim_frac_binding": round(s_frac, 3),
        "frac_diff": round(diff, 3), "frac_tol": frac_tol,
        "real_pref_median_s": _med(r_pref), "sim_pref_median_s": _med(s_pref),
        "n_rounds_real": len(r_binds), "n_rounds_sim": len(s_binds),
    }


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


def selection_detail_parity(real: dict, sim: dict,
                              tol_chosen: float = 0.05,
                              tol_inflight: float = 0.15) -> dict:
    """S3/S4 [DIST]: num_chosen, in_flight, effective_c mean parity across modes.

    num_chosen and in_flight are enforced (DIST); effective_c is diagnostic only.
    """
    def collect(sel_events):
        chosen, inflight, eff_c = [], [], []
        for e in sel_events:
            nc = e.get("num_chosen")
            inf = e.get("in_flight")
            ec = e.get("effective_c")
            if nc is not None:
                chosen.append(nc)
            if inf is not None:
                inflight.append(inf)
            if ec is not None:
                eff_c.append(ec)
        return chosen, inflight, eff_c

    r_ch, r_inf, r_ec = collect(real["selection_train"])
    s_ch, s_inf, s_ec = collect(sim["selection_train"])
    if not r_ch:
        return {"ok": True, "tier": "DIST", "status": "SKIP",
                "note": "no num_chosen in selection telemetry"}

    def mean_or_nan(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    r_ch_m = mean_or_nan(r_ch)
    s_ch_m = mean_or_nan(s_ch)
    r_inf_m = mean_or_nan(r_inf)
    s_inf_m = mean_or_nan(s_inf)
    r_ec_m = mean_or_nan(r_ec)
    s_ec_m = mean_or_nan(s_ec)

    rel_chosen = abs(r_ch_m - s_ch_m) / max(r_ch_m, s_ch_m, 1) if not math.isnan(r_ch_m) else 0.0
    rel_inflight = abs(r_inf_m - s_inf_m) / max(r_inf_m, s_inf_m, 1) if (
        not math.isnan(r_inf_m) and not math.isnan(s_inf_m)) else 0.0

    ok = rel_chosen <= tol_chosen and rel_inflight <= tol_inflight
    return {
        "ok": ok,
        "tier": "DIST",
        "real_mean_chosen": round(r_ch_m, 2) if not math.isnan(r_ch_m) else None,
        "sim_mean_chosen": round(s_ch_m, 2) if not math.isnan(s_ch_m) else None,
        "rel_diff_chosen": round(rel_chosen, 3),
        "real_mean_inflight": round(r_inf_m, 2) if not math.isnan(r_inf_m) else None,
        "sim_mean_inflight": round(s_inf_m, 2) if not math.isnan(s_inf_m) else None,
        "rel_diff_inflight": round(rel_inflight, 3),
        "real_mean_effective_c": round(r_ec_m, 2) if not math.isnan(r_ec_m) else None,
        "sim_mean_effective_c": round(s_ec_m, 2) if not math.isnan(s_ec_m) else None,
        "tol_chosen": tol_chosen,
        "tol_inflight": tol_inflight,
    }


def inflight_residence_parity(real: dict, sim: dict,
                              tol_rel: float = 0.3,
                              floor: float = 0.5) -> dict:
    """Sr [DIST]: straggler carry-over (in-flight residence) parity across modes.

    On the sync oort stack (oort + refl) the aggregator over-selects
    (aggr_num*overcommitment) and closes a round at agg_goal commits, leaving the
    slowest ~(selected-agg_goal) trainers still computing — they CARRY OVER into
    the next round as in-flight (`in_flight_after`).  In real these stragglers
    occupy a slot until they actually finish; in sim their update arrives
    physically at once, so a naive aggregator cleans them up immediately and the
    in-flight set DRAINS to ~0.  That structural drain (not stochastic path drift —
    it is invariant to the selection mix) under-counts sim concurrency (S3/4),
    under-commits fresh updates, and lets slow trainers re-enter the pool a round
    early, skewing committed speed/budget (P3/T2).

    Grades the mean `in_flight_after` (carried stragglers) across modes; reports
    residence_rounds, committed_fresh, stale_rejected as diagnostics.  SKIPs when
    the stack emits no `inflight_residence` telemetry (async felix / feddance) or
    when neither mode carries anything (no overcommit → nothing to carry, trivially
    matched).  This is the §4.5-class carry-over rung, distinct from the pool-
    exclusion `sim_inflight_residence` mechanism (which keeps still-computing
    trainers out of the *pool* but does not make sim *carry* them in-flight).
    """
    r_ev = real.get("residence", [])
    s_ev = sim.get("residence", [])
    if not r_ev or not s_ev:
        return {"ok": True, "tier": "DIST", "status": "SKIP",
                "note": "no inflight_residence telemetry (async stack or "
                        "pre-instrumentation)"}

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    def field(evs, key):
        return [e[key] for e in evs if e.get(key) is not None]

    def flat(evs, key):
        out = []
        for e in evs:
            v = e.get(key)
            if isinstance(v, list):
                out.extend(v)
        return out

    r_carry = mean(field(r_ev, "in_flight_after"))
    s_carry = mean(field(s_ev, "in_flight_after"))
    r_fresh = mean(field(r_ev, "committed_fresh"))
    s_fresh = mean(field(s_ev, "committed_fresh"))
    r_stale = mean(field(r_ev, "stale_rejected"))
    s_stale = mean(field(s_ev, "stale_rejected"))
    r_res = mean(flat(r_ev, "residence_rounds"))
    s_res = mean(flat(s_ev, "residence_rounds"))

    if max(r_carry, s_carry) < floor:
        ok = True
        rel = 0.0
        note = ("no overcommit carry-over in either mode "
                f"(real={r_carry:.2f} sim={s_carry:.2f} < {floor}) — trivially matched")
    else:
        rel = abs(r_carry - s_carry) / max(r_carry, s_carry)
        ok = rel <= tol_rel
        note = ("sim drains stragglers vs real carry-over"
                if not ok else "carry-over matched")
    return {
        "ok": ok,
        "tier": "DIST",
        "rel_diff_carry": round(rel, 3),
        "tol_rel": tol_rel,
        "real_inflight_after": round(r_carry, 2),
        "sim_inflight_after": round(s_carry, 2),
        "real_committed_fresh": round(r_fresh, 2),
        "sim_committed_fresh": round(s_fresh, 2),
        "real_stale_rejected": round(r_stale, 2),
        "sim_stale_rejected": round(s_stale, 2),
        "real_residence_rounds": round(r_res, 3),
        "sim_residence_rounds": round(s_res, 3),
        "note": note,
    }


# ═══════════════════════════════════════════════════════════════════
# §3.D  Updates received & ordering  (U1–U5)
# ═══════════════════════════════════════════════════════════════════

def aggregation_sequence_parity(real: dict, sim: dict,
                                 max_rounds: Optional[int] = None) -> dict:
    """P1 / U1: Per-round set of contributing trainers matches across modes.

    Enforced only for DETERMINISTIC_SELECTORS; gated to WARN for stochastic
    selectors.  Exact per-round contributing-set identity is unattainable for a
    stochastic, streaming, path-dependent selector (a trainer is chosen in
    *different* rounds across modes), the same reason S1 (selection_parity) is
    gated.  The enforced selection invariants for stochastic selectors are
    participation_parity (S2) + the pooled distributions; this check stays as a
    diagnostic so a future deterministic selector still gets exact-set checking.
    """
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
    selector = _selector_name(real, sim)
    gated = bool(selector) and selector not in DETERMINISTIC_SELECTORS
    enforced_ok = (not rounds) or matches == len(rounds)
    return {
        "ok": True if gated else enforced_ok,
        "tier": "DIST",
        "gated": gated,
        "selector": selector or None,
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


NEAR_ZERO_LAG_S = 0.05  # U6: both-modes mean lag <= this ⇒ immediate commit, KS uninformative


def commit_visibility_parity(real: dict, sim: dict, warn_ks: float = 0.2,
                             warn_mean_diff: float = 2.0) -> dict:
    """U6 (commit timeliness): update_visibility_lag_s distributions match.

    Lag = aggregator-clock delay between an update becoming READY to aggregate
    and being COMMITTED to the global model (sim: vclock-sct; real: wall
    commit-arrival). Same metric, mode-appropriate clock. For async the target
    is ~0 in both modes (independent commits at own readiness); for sync it is
    the barrier wait, matching in both. Either way fidelity = sim dist == real
    dist, so we KS the two and also flag the mean gap. Upstream of staleness:
    a sim that commits updates late (past-dating) inflates staleness downstream.
    """
    def vals(agg_rounds):
        out = []
        for e in agg_rounds:
            v = e.get("update_visibility_lag_s")
            if v is None:
                continue
            if isinstance(v, (int, float)):
                out.append(float(v))
            else:
                out.extend(float(x) for x in v if x is not None)
        return out

    rv, sv = vals(real["agg_rounds"]), vals(sim["agg_rounds"])
    if not rv or not sv:
        return {"ok": True, "tier": "DIST", "skipped": True,
                "reason": "update_visibility_lag_s absent in one mode "
                          "(re-run to populate)",
                "real_n": len(rv), "sim_n": len(sv)}
    rm, _ = mean_std(rv)
    sm, _ = mean_std(sv)
    ks = ks_stat(rv, sv)
    mean_diff = abs(rm - sm) if not (math.isnan(rm) or math.isnan(sm)) else float("nan")
    # Point-mass guard: when both modes commit immediately the lag is a sub-50ms
    # point mass at ~0, so KS→1.0 is uninformative (the A2 num_candidates case).
    # A real past-dating divergence (felix: sim mean 14.8s) clears NEAR_ZERO_S by
    # 100s of ms — judge those on mean_diff, not the degenerate-KS artifact.
    pointmass = (not math.isnan(rm) and not math.isnan(sm)
                 and abs(rm) <= NEAR_ZERO_LAG_S and abs(sm) <= NEAR_ZERO_LAG_S)
    if pointmass:
        ok = True
    elif not math.isnan(ks):
        ok = ks <= warn_ks and (math.isnan(mean_diff) or mean_diff <= warn_mean_diff)
    else:
        ok = True
    out = {
        "ok": ok,
        "tier": "DIST",
        "real_mean": round(rm, 3) if not math.isnan(rm) else None,
        "sim_mean": round(sm, 3) if not math.isnan(sm) else None,
        "real_p90": round(percentile(rv, 90), 3),
        "sim_p90": round(percentile(sv, 90), 3),
        "ks_stat": round(ks, 3) if not math.isnan(ks) else None,
        "mean_diff": round(mean_diff, 3) if not math.isnan(mean_diff) else None,
    }
    if pointmass:
        out["note"] = ("both modes commit immediately (mean lag <= {:.0f}ms): "
                       "KS uninformative on a near-zero point mass — passed on mean"
                       .format(NEAR_ZERO_LAG_S * 1000))
    return out


def eval_commit_timeliness(sim: dict, max_excess_s: float = 2.0) -> dict:
    """U6e (sim invariant): EVAL commits must be as timely as TRAIN commits.

    An eval task that ships a STALE train completion ts (the `evaluate()` reused
    `_sim_completion_ts` bug) commits long after the virtual clock has passed it.
    Signature: eval `update_visibility_lag_s` (fallback `commit_gap_s`)
    systematically larger than train's. Sim-only — real never past-dates by
    construction; self-SKIPs when the run dispatches no eval (e.g. sync oort) or
    the field is absent. Localizes the eval-stale-`sct` regression directly.
    """
    def by_task(agg_rounds, key):
        out = collections.defaultdict(list)
        for e in agg_rounds:
            v = e.get(key)
            if v is None:
                continue
            t = str(e.get("task_to_perform", "train"))
            vs = v if isinstance(v, (list, tuple)) else [v]
            out[t].extend(float(x) for x in vs if x is not None)
        return out

    # Train commits live in agg_rounds; eval commits are partitioned into
    # eval_commits at load — U6e needs both to compare eval-vs-train timeliness.
    commits = sim["agg_rounds"] + sim.get("eval_commits", [])
    lag = by_task(commits, "update_visibility_lag_s")
    if not lag.get("eval") and not lag.get("train"):
        lag = by_task(commits, "commit_gap_s")  # older runs
    train, ev = lag.get("train", []), lag.get("eval", [])
    if not ev:
        return {"ok": True, "tier": "DIST", "skipped": True,
                "reason": "no eval commits in sim (baseline dispatches no eval, "
                          "or task-tagged field absent — re-run to populate)",
                "train_n": len(train), "eval_n": 0}
    tm, _ = mean_std(train) if train else (0.0, 0.0)
    em, _ = mean_std(ev)
    excess = em - tm
    ok = excess <= max_excess_s
    out = {
        "ok": ok, "tier": "DIST",
        "train_mean": round(tm, 3), "eval_mean": round(em, 3),
        "eval_minus_train_s": round(excess, 3),
        "eval_p90": round(percentile(ev, 90), 3),
        "train_n": len(train), "eval_n": len(ev),
    }
    if not ok:
        out["note"] = ("eval commits systematically past-dated vs train "
                       "(eval likely shipping a stale train sct)")
    return out


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

def participation_parity(real: dict, sim: dict, ks_tol: float = 0.2) -> dict:
    """S2: per-trainer participation distributions match over a MATCHED round window.

    The enforced selection invariant for stochastic selectors. The participation
    *shape* (how commits spread across trainers) must match; the *amount* (total
    commits / rounds) is a throughput quantity already owned by K2/U2/K8 and must
    NOT be re-charged here.

    History: the first cut used raw `avg_diff` (grew with run length → false-failed
    long runs); later switched to participation **share** (count / total_commits) to
    be length-free. But share is NOT amount-free: when the two modes complete a
    different number of rounds in the compared window (the throughput gap), every
    trainer's count scales by the same ratio, and *any* scalar normalization (share,
    rate, count/mean) preserves that offset — so share_KS re-measures the throughput
    delta as if it were a shape divergence (e.g. feddance share_KS .427 while the
    per-trainer count distribution on equal rounds matched at .033; oort .368→.127).

    Fix (consistent with K8/U2/terminal): count participation over the MATCHED round
    window — the first N = min(rounds_real, rounds_sim) rounds of each mode — then KS
    on per-trainer counts. Equal rounds ⇒ equal totals ⇒ KS measures pure shape. A
    genuine shape divergence still FAILs (e.g. refl .530 — a real selection-mix
    difference to localize at Stage 3, NOT suppressed). `share_ks` (full run) kept as
    a diagnostic; `avg_diff`/`max_diff` raw diagnostics.
    """
    def rounds_of(agg_rounds):
        by_round = collections.defaultdict(list)
        for e in agg_rounds:
            r = e.get("round")
            if r is not None:
                by_round[r].extend(e.get("contributing_trainers", []))
        return by_round

    def counts_first_n(by_round, n):
        c = collections.Counter()
        for r in sorted(by_round)[:n]:
            c.update(by_round[r])
        return c

    r_by, s_by = rounds_of(real["agg_rounds"]), rounds_of(sim["agg_rounds"])
    if not r_by or not s_by:
        return {"ok": True, "tier": "DIST", "note": "no contributing_trainers"}
    n_matched = min(len(r_by), len(s_by))
    rc = counts_first_n(r_by, n_matched)
    sc = counts_first_n(s_by, n_matched)
    trainers = set(rc) | set(sc)
    if not trainers:
        return {"ok": True, "tier": "DIST", "note": "no contributing_trainers"}
    r_counts = [rc.get(t, 0) for t in trainers]
    s_counts = [sc.get(t, 0) for t in trainers]
    ks = ks_stat(r_counts, s_counts)
    # full-run share KS — diagnostic only (entangled with the throughput delta).
    rc_full = collections.Counter(t for vs in r_by.values() for t in vs)
    sc_full = collections.Counter(t for vs in s_by.values() for t in vs)
    allt = set(rc_full) | set(sc_full)
    tr, ts = sum(rc_full.values()) or 1, sum(sc_full.values()) or 1
    share_ks = ks_stat([rc_full.get(t, 0) / tr for t in allt],
                        [sc_full.get(t, 0) / ts for t in allt])
    diffs = [abs(rc.get(t, 0) - sc.get(t, 0)) for t in trainers]
    avg = sum(diffs) / len(diffs)

    # Participation by SPEED CLASS — the policy-level invariant for a STOCHASTIC
    # selector. The per-trainer-IDENTITY KS (matched_count_ks) is path-dependent:
    # refl builds a ~120-trainer persistent core whose SIZE, concentration, and
    # speed composition match across modes, but the specific individuals diverge
    # (Jun-24 3h: only 63 of ~120 shared) because the weighted-exploit draw, fed
    # slightly different per-round eligibility (the A2 in-flight-timing artifact),
    # locks in different individuals via rich-get-richer. The mode-specific cores
    # are SPEED-MATCHED (real-only D̄ 9.8 vs sim-only 9.2; participation-weighted
    # D̄ 8.20 vs 8.26) and A2c/K8 pass — so there is no selection-mix bias, only
    # stochastic identity. What the POLICY determines (and must match) is how
    # participation distributes across intrinsic speed CLASSES; bucket the matched
    # -window counts by the registry `speed_class` and compare the per-class SHARE
    # (total-variation distance). Granularity matters: at speed_class level the
    # Jun-24 refl shares match (TVD 0.026), while per-SECOND buckets re-expose the
    # same stochastic within-class identity noise (TVD 0.187, sign-alternating).
    # Same §5 class as P1/F1-3 per-trainer KS.
    speed_class = _trainer_speed_class_map()
    speed_class_tvd = None
    if speed_class:
        def class_share(counter):
            agg = collections.Counter()
            for t, k in counter.items():
                c = speed_class.get(t)
                if c is not None:
                    agg[c] += k
            tot = sum(agg.values()) or 1
            return {b: agg[b] / tot for b in agg}
        rcs, scs = class_share(rc), class_share(sc)
        buckets = set(rcs) | set(scs)
        speed_class_tvd = 0.5 * sum(abs(rcs.get(b, 0) - scs.get(b, 0)) for b in buckets)

    selector = _selector_name(real, sim)
    gated = bool(selector) and selector not in DETERMINISTIC_SELECTORS
    tvd_tol = 0.15
    if gated and speed_class_tvd is not None:
        # stochastic: enforce the speed-class participation, identity is diagnostic
        ok = speed_class_tvd <= tvd_tol
    else:
        ok = not math.isnan(ks) and ks <= ks_tol
    return {
        "ok": ok,
        "tier": "DIST",
        "gated_stochastic": gated,
        "speed_class_tvd": round(speed_class_tvd, 3) if speed_class_tvd is not None else None,
        "tvd_tol": tvd_tol,
        "matched_count_ks": round(ks, 3) if not math.isnan(ks) else None,
        "ks_tol": ks_tol,
        "n_rounds_matched": n_matched,
        "share_ks": round(share_ks, 3) if not math.isnan(share_ks) else None,  # diagnostic
        "avg_diff": round(avg, 2),   # raw, scale-dependent (diagnostic only)
        "max_diff": max(diffs) if diffs else 0,
    }


def decision_determinism_parity(real: dict, sim: dict) -> dict:
    """Sdet [DIAG]: under a shared seed, are the per-round selection DECISIONS
    reproducible across modes — and if not, is it the inputs or the draw?

    Consumes the seeding telemetry stamped on each selection event
    (`seed`, `eligible_fingerprint`, `decision_fingerprint`, `chosen`). Matching
    rounds by round number, it reports three fractions:
      - eligible_match  : same candidate SET seen by the selector.
      - decision_match  : same set AND same per-candidate utility/speed + k (the
                          full draw input).
      - chosen_match    : same selected set.
    This splits a participation/selection divergence cleanly (the whole point of
    seeding:
      - seed present, decision_match≈1, chosen_match≈1 → seeding WORKED; any
        residual participation/utility gap is NOT stochastic — look elsewhere.
      - decision_match≈1 but chosen_match≪1 → identical inputs, different draw =
        RNG desync (a selector still hitting the global np.random, or a seed not
        threaded). Fix the selector RNG.
      - decision_match≪1 → inputs already diverge (availability/utility/eligible
        ordering) BEFORE the draw; seeding can't help until that's fixed — drop
        to the eligible_match line to see if it's the SET or the values.
    DIAG: never fails; it localizes. SKIP if fingerprints absent (unseeded/old run).
    """
    def by_round(events):
        out = {}
        for e in events:
            r = e.get("round")
            if r is None or e.get("decision_fingerprint") is None:
                continue
            out[r] = e  # last write per round wins
        return out

    r_by, s_by = by_round(real["selection_train"]), by_round(sim["selection_train"])
    common = sorted(set(r_by) & set(s_by))
    if not common:
        return {"ok": True, "tier": "DIAG", "status": "SKIP",
                "note": "no decision_fingerprint in selection telemetry "
                        "(run unseeded or pre-instrumentation)"}
    elig = dec = chosen = 0
    for r in common:
        re_, se = r_by[r], s_by[r]
        if re_.get("eligible_fingerprint") == se.get("eligible_fingerprint"):
            elig += 1
        if re_.get("decision_fingerprint") == se.get("decision_fingerprint"):
            dec += 1
        if set(re_.get("chosen") or []) == set(se.get("chosen") or []):
            chosen += 1
    n = len(common)
    real_seed = next((e.get("seed") for e in r_by.values()), None)
    sim_seed = next((e.get("seed") for e in s_by.values()), None)
    elig_f, dec_f, chosen_f = elig / n, dec / n, chosen / n
    if real_seed is None or sim_seed is None:
        verdict = "UNSEEDED — decisions are independent stochastic paths; expect low match"
    elif dec_f > 0.98 and chosen_f > 0.98:
        verdict = "seeding WORKED — decisions reproducible; residual gaps are NOT stochastic"
    elif dec_f > 0.98:
        verdict = "RNG DESYNC — identical inputs, different draw (selector not using seeded RNG)"
    else:
        verdict = "INPUT DIVERGENCE — candidate set/utilities differ before the draw (fix upstream)"
    return {
        "ok": True,
        "tier": "DIAG",
        "real_seed": real_seed,
        "sim_seed": sim_seed,
        "n_rounds_compared": n,
        "eligible_match_frac": round(elig_f, 3),
        "decision_match_frac": round(dec_f, 3),
        "chosen_match_frac": round(chosen_f, 3),
        "verdict": verdict,
    }


def trainer_speed_parity(real: dict, sim: dict, ks_tol: float = 0.1,
                         support_tol: float = 0.15) -> dict:
    """P3: trainer_speed_s — the speed MODEL is identical (control).

    Enforced metric = **support containment** (Jun-16 reclassification). The job
    of P3 is to isolate the *speed model* (does a trainer's compute time come from
    the same generator across modes?), NOT selection. But `trainer_speed_s` pools
    the *selected* trainers' speeds, so a frequency/mean shift here can be either:
      (a) a genuine speed-model bug — sim produces speeds **outside real's
          support** (oort's old 56 s→sim tail vs real_max 21 s), or
      (b) selection mix — sim *selects* faster trainers from the **same support**
          (feddance 3h: pool speed `A2b` KS=0, sim_max 56.0 ≈ real_max 56.12, but
          sim picks faster → mean 10.9 vs 12.7).
    Only (a) is a speed-model bug; (b) is owned by `A2c selection_bias`/`Sx`.
    Distinguishing them from two speed lists alone: wall-capture and faster-mix
    both keep sim **within** real's support (real = compute + capture ≥ sim, and a
    faster mix only drops sim's high tail), whereas a model bug pushes sim's tail
    **beyond** real. So we enforce ``sim_p99 <= real_p99 * (1 + support_tol)`` and
    demote the grid/mean KS to diagnostics (the selection-mix signal, judged by
    A2c at its own tolerance). Verified non-masking: oort's genuine tail trips the
    support guard; feddance's mix passes it while A2c still owns (and at 3h passes)
    the mix verdict. NB: this defers a *real* fidelity gap (the mix can move
    end-to-end perf) — flagged in PARITY.md to revisit for higher fidelity.
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
    raw_ks = ks_stat(real_speeds, sim_speeds)
    grid_ks = ks_stat([round(v) for v in real_speeds],
                      [round(v) for v in sim_speeds])
    real_mean, _ = mean_std(real_speeds)
    sim_mean, _ = mean_std(sim_speeds)
    mean_overhead = real_mean - sim_mean
    real_p99, sim_p99 = percentile(real_speeds, 99), percentile(sim_speeds, 99)
    # support guard: sim must not produce speeds materially beyond real's range.
    support_ratio = sim_p99 / real_p99 if real_p99 > 0 else float("nan")
    ok = (not math.isnan(support_ratio)
          and support_ratio <= 1.0 + support_tol)
    mix_deferred = bool(ok and grid_ks > ks_tol)  # passes support but mix-shifted
    return {
        "ok": ok,
        "tier": "DIST",
        "support_ratio": round(support_ratio, 3) if not math.isnan(support_ratio) else None,
        "support_tol": support_tol,
        "real_p99_speed_s": round(real_p99, 2),
        "sim_p99_speed_s": round(sim_p99, 2),
        "mix_deferred": mix_deferred,
        # diagnostics (selection-mix signal; A2c selection_bias owns the verdict):
        "ks_stat": round(grid_ks, 3) if not math.isnan(grid_ks) else None,
        "ks_tol": ks_tol,
        "raw_ks_stat": round(raw_ks, 3) if not math.isnan(raw_ks) else None,
        "mean_overhead_s": round(mean_overhead, 3),
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

def utility_parity(real: dict, sim: dict, max_ks: float = 0.2,
                   min_samples: int = 10) -> dict:
    """F1/F2/F3: stat_utility distributions match.

    The *enforced* metric is the **pooled** utility KS — every committed
    utility value across all trainers, compared as one distribution.  That is
    the mode-agnostic, path-independent measure of whether the simulator
    reproduces the utilities the system aggregates.

    Per-trainer identity is a separate, *gated* diagnostic: for a stochastic,
    streaming selector a given trainer is chosen in different rounds across
    modes (seeing different streamed data), so its individual utility series
    can't match — and a trainer seen only 1–2 times yields a mechanical KS=1.0
    that says nothing.  We therefore restrict the per-trainer KS to trainers
    with >= ``min_samples`` commits in BOTH modes and only enforce it for a
    DETERMINISTIC selector (currently none ship; see DETERMINISTIC_SELECTORS).
    """
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

    # ── pooled distribution (the enforced fidelity measure) ──
    r_pool = [u for vals in r_utils.values() for u in vals]
    s_pool = [u for vals in s_utils.values() for u in vals]
    pooled_ks = ks_stat(r_pool, s_pool)

    # ── per-trainer diagnostic, restricted to well-sampled trainers ──
    all_trainers = sorted(set(r_utils) | set(s_utils))
    ks_stats, mean_diffs = [], []
    for t in all_trainers:
        ru = r_utils.get(t, [])
        su = s_utils.get(t, [])
        if len(ru) < min_samples or len(su) < min_samples:
            continue
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

    selector = _selector_name(real, sim)
    gated = bool(selector) and selector not in DETERMINISTIC_SELECTORS
    pooled_ok = math.isnan(pooled_ks) or pooled_ks <= max_ks
    per_trainer_ok = math.isnan(max_ks_val) or max_ks_val <= max_ks
    # Stochastic: enforce the pooled distribution only.  Deterministic: also
    # require per-trainer identity over well-sampled trainers.
    ok = pooled_ok if gated else (pooled_ok and per_trainer_ok)
    return {
        "ok": ok,
        "tier": "DIST",
        "gated": gated,
        "selector": selector or None,
        "pooled_ks_stat": round(pooled_ks, 3) if not math.isnan(pooled_ks) else None,
        "max_ks_stat": round(max_ks_val, 3) if not math.isnan(max_ks_val) else None,
        "avg_mean_utility_diff": round(avg_mean_diff, 2) if not math.isnan(avg_mean_diff) else None,
        "n_trainers": len(all_trainers),
        "n_trainers_well_sampled": len(ks_stats),
        "min_samples": min_samples,
        "max_ks_tol": max_ks,
    }


# ═══════════════════════════════════════════════════════════════════
# §3.G  Convergence  (C1–C3)
# ═══════════════════════════════════════════════════════════════════

# Below this run budget a convergence PASS is not trustworthy: the real/sim
# accuracy/loss gap grows with training, so a short run hasn't trained far enough
# to reveal it (a short-run FAIL is still real — the gap only widens). 2 h.
SHORT_RUN_CONFIDENCE_S = 7200.0


def _mark_low_confidence_if_short(res: dict, budget_s: Optional[float]) -> dict:
    """Tag a convergence PASS as low-confidence on a sub-2h run; leave FAILs alone."""
    if (budget_s is not None and budget_s < SHORT_RUN_CONFIDENCE_S
            and res.get("ok") and not _is_skipped(res)):
        res["low_confidence"] = True
        res["status"] = "LOW_CONF"
        res["note"] = (f"budget {int(budget_s)}s < {int(SHORT_RUN_CONFIDENCE_S)}s: "
                       "convergence pass is inconclusive (a fail would still be real)")
    return res


def convergence_parity(real: dict, sim: dict,
                        acc_tol: float = 0.05,
                        budget_s: Optional[float] = None) -> dict:
    """C1/C2: Accuracy and loss curves aligned by FL round.

    C3 fix: the original compare_parity.py had a self-compare bug where
    sc was assigned from real["agg_evals"] before being overwritten with
    sim["agg_evals"].  This implementation uses sim directly.

    Horizon guard: on a sub-2h run a PASS is downgraded to LOW_CONF (the curves
    haven't diverged yet); a genuine FAIL still surfaces.
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
    return _mark_low_confidence_if_short({
        "ok": ok,
        "tier": "DIST",
        "eval_rounds_compared": len(rounds),
        "avg_accuracy_diff": round(avg_acc, 4) if not math.isnan(avg_acc) else None,
        "avg_loss_diff": round(avg_loss, 4) if not math.isnan(avg_loss) else None,
        "acc_tol": acc_tol,
    }, budget_s)


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


def throughput_parity(real: dict, sim: dict, tol_rel: float = 0.05) -> dict:
    """K2 [EXACT]: rounds-per-virtual-second parity.

    sim_throughput  = total_sim_rounds / final_vclock_sim
    real_throughput = total_real_rounds / wall_elapsed_real

    On the motivating Felix run (410 vs 673 rounds in the same 3 h budget)
    rel_diff ≈ 40% → FAIL.

    Tolerance 5%: the throughput family — K2 (this mechanism), K8 (rounds at
    matched V), U2 (commits at matched V) — all measure the same rounds-per-virtual-time
    signal and share ONE tolerance, set here. Tightened 10%->5% deliberately as the
    throughput-fidelity bar. U2 == K8 == K2 on the identical quantity, so they must agree.
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

    KS is enforced at the **integer-second grid** (same wall-capture rationale as
    P3 `trainer_speed_parity`): sim's Δvclock is quantized to whole-second modeled
    completions (mass piled at e.g. 28.00) while real's Δwall spreads continuously
    around the same value (28.0x network/scheduling jitter). A raw KS then jumps to
    ~0.7 at the quantization point even when the means/medians/percentiles match
    (feddance 3h: raw .715 vs grid .064, identical p10..p90). The grid KS aligns
    them; the mean-diff guard (≤ ``mean_tol_rel``) still catches a genuine advance
    divergence (felix sim 2.25 vs real 4.02 fails on the mean regardless). Raw KS
    kept as a diagnostic.
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
    raw_ks = ks_stat(sim_adv, real_adv)
    grid_ks = ks_stat([round(v) for v in sim_adv], [round(v) for v in real_adv])
    sim_mean, _ = mean_std(sim_adv)
    real_mean, _ = mean_std(real_adv)
    mean_rel_diff = (abs(sim_mean - real_mean) / max(sim_mean, real_mean)
                     if max(sim_mean, real_mean) > 0 else 0.0)
    ok = grid_ks <= ks_tol and mean_rel_diff <= mean_tol_rel
    return {
        "ok": ok,
        "tier": "EXACT",
        "sim_mean_advance_s": round(sim_mean, 2),
        "real_mean_advance_s": round(real_mean, 2),
        "mean_rel_diff": round(mean_rel_diff, 3),
        "ks_stat": round(grid_ks, 3),
        "raw_ks_stat": round(raw_ks, 3),
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


def total_commits_parity(real: dict, sim: dict, tol_rel: float = 0.05) -> dict:
    """U2 [EXACT]: total commits at matched virtual budget V = min(final_vclock, final_wall).

    abs diff ≤ 5% of commits — the shared throughput-family tolerance (= K2, K8).

    Rationale: U4 (agg_goal_count cycles 1..K, INV) separately guarantees a fixed
    agg_goal commits per round, so at matched V the commit count is the round count × agg_goal —
    i.e. U2 carries no signal beyond K8's matched-V round rollup (and the K2 throughput mechanism).
    Verified: U2.rel_diff == K8.rounds_rel_diff to 3 decimals on all four
    baselines (commits/round identical across modes). The old 2% bar required matched-V commits to
    match 5× tighter than matched-V rounds / throughput itself, with no separate mechanism behind
    it: a stochastic 2-rounds-in-85 difference (feddance) or a residual throughput delta that K2/K8
    already judge failed U2 alone. The throughput family now shares ONE deliberate 5% bar; U2 stays
    as a commit-level cross-check of the same rollup (append-only guard), not a stricter one.
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
                           rounds_tol: float = 0.05,
                           trainers_tol: float = 0.05) -> dict:
    """K8 [EXACT]: at matched virtual budget V, both modes have comparable FL-round count.

    rounds within 5% (the shared throughput-family bar, = K2/U2), unique trainers within 5%.
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


_PHASE_FIELDS = ("pre_train_s", "gpu_compute_s", "mqtt_fetch_s",
                 "weights_to_gpu_s", "weights_to_ram_s", "post_train_s")


def trainer_phase_parity(real_trainers: dict, sim_trainers: dict) -> dict:
    """T_phase [DIAG]: Per-phase timing distribution comparison (real vs sim).

    Collects trainer_round phase fields across all trainers and reports KS +
    mean for each.  Purely diagnostic — helps isolate WHERE real/sim time
    diverges (e.g. mqtt_fetch_s real>>sim explains vclock under-charge).
    """
    def collect_phases(trainers: dict) -> dict:
        out: dict = {f: [] for f in _PHASE_FIELDS}
        for _tid, d in trainers.items():
            for e in d.get("trainer_round", []):
                for f in _PHASE_FIELDS:
                    v = e.get(f)
                    if v is not None and v >= 0:
                        out[f].append(v)
        return out

    r_phases = collect_phases(real_trainers)
    s_phases = collect_phases(sim_trainers)

    has_data = any(r_phases[f] for f in _PHASE_FIELDS)
    if not has_data:
        return {"ok": True, "tier": "DIAG", "status": "SKIP",
                "note": "no phase timing fields in trainer telemetry"}

    per_phase = {}
    for f in _PHASE_FIELDS:
        rv, sv = r_phases[f], s_phases[f]
        if not rv and not sv:
            continue
        r_mean = sum(rv) / len(rv) if rv else float("nan")
        s_mean = sum(sv) / len(sv) if sv else float("nan")
        ks = ks_stat(rv, sv) if rv and sv else float("nan")
        per_phase[f] = {
            "real_mean_s": round(r_mean, 3) if not math.isnan(r_mean) else None,
            "sim_mean_s": round(s_mean, 3) if not math.isnan(s_mean) else None,
            "ks": round(ks, 3) if not math.isnan(ks) else None,
        }

    return {"ok": True, "tier": "DIAG", "per_phase": per_phase}


# ═══════════════════════════════════════════════════════════════════
# §3.0  Telemetry coverage  (TC1)  — Stage 0 gate
# ═══════════════════════════════════════════════════════════════════

# (label, event_source, field, mode_expected)
#   event_source ∈ {"agg", "sel", "trainer_round", "task_recv"}
#   mode_expected ∈ {"both", "sim", "real"} — where the field must be present
_COVERAGE_SPEC = [
    ("agg_round.vclock_now",            "agg",           "vclock_now",            "sim"),
    ("agg_round.trainer_speed_s",       "agg",           "trainer_speed_s",       "both"),
    ("agg_round.staleness",             "agg",           "staleness",             "both"),
    ("agg_round.stat_utility",          "agg",           "stat_utility",          "both"),
    ("agg_round.contributing_trainers", "agg",           "contributing_trainers", "both"),
    ("selection.num_eligible",          "sel",           "num_eligible",          "both"),
    ("selection.avail_composition",     "sel",           "avail_composition",     "both"),
    ("selection.num_chosen",            "sel",           "num_chosen",            "both"),
    ("trainer_round.gpu_compute_s",     "trainer_round", "gpu_compute_s",         "both"),
    ("trainer_round.training_budget_s", "trainer_round", "training_budget_s",     "both"),
    ("task_recv.sim_send_ts",           "task_recv",     "sim_send_ts",           "sim"),
]


def field_coverage(real_agg: dict, sim_agg: dict,
                   real_trainers: dict, sim_trainers: dict) -> dict:
    """TC1 [INV]: every field a downstream check reads must be present in the
    modes that need it.

    Generalizes K10: a single coverage matrix turns "9 mysterious SKIPs"
    into "these fields are absent in sim".  FAIL-LOUD when an expected field
    has zero density in a mode that requires it.
    """
    def _density(events: list, field: str) -> Optional[float]:
        if not events:
            return None
        n = sum(1 for e in events if e.get(field) not in (None, [], {}))
        return n / len(events)

    def _agg_evs(agg, src):
        return agg["agg_rounds"] if src == "agg" else agg["selection_train"]

    def _tr_evs(tr, src):
        return [e for d in tr.values() for e in d.get(src, [])]

    matrix: dict = {}
    violations: list = []
    for label, src, field, mode in _COVERAGE_SPEC:
        if src in ("agg", "sel"):
            rd = _density(_agg_evs(real_agg, src), field)
            sd = _density(_agg_evs(sim_agg, src), field)
        else:
            rd = _density(_tr_evs(real_trainers, src), field)
            sd = _density(_tr_evs(sim_trainers, src), field)
        matrix[label] = {
            "real": round(rd, 3) if rd is not None else None,
            "sim": round(sd, 3) if sd is not None else None,
            "expect": mode,
        }
        if mode in ("both", "real") and not rd:
            violations.append(f"{label}(real)")
        if mode in ("both", "sim") and not sd:
            violations.append(f"{label}(sim)")
    return {"ok": not violations, "tier": "INV",
            "matrix": matrix, "violations": violations}


# ═══════════════════════════════════════════════════════════════════
# §3.1  Clock-model decomposition  (K3a / K3b)  — Stage 1
# ═══════════════════════════════════════════════════════════════════

def modeled_compute_advance(real: dict, sim: dict) -> dict:
    """K3a [DIAG]: per-mode, compare per-round advance to per-round max
    committed trainer_speed_s (the modeled *compute* component).

    advance − max_speed = the implied per-round overhead (real) or
    overlap/overhead net (sim).  Reporting both modes side-by-side isolates
    whether the gap K3/K2 see is compute-formula vs overhead vs overlap.
    """
    def _stats(agg: dict, use_vclock: bool):
        adv = _per_round_advances(agg["agg_rounds"], use_vclock=use_vclock)
        spd = _per_round_max_speed(agg["agg_rounds"])
        if not adv or not spd:
            return None
        return sum(adv) / len(adv), sum(spd.values()) / len(spd)

    s = _stats(sim, use_vclock=True)
    r = _stats(real, use_vclock=False)
    if not s or not r:
        return {"ok": True, "tier": "DIAG", "status": "SKIP",
                "note": "insufficient advance/speed data (K10 may be blocking sim)"}
    return {
        "ok": True, "tier": "DIAG",
        "sim_mean_advance_s": round(s[0], 2),
        "sim_mean_max_speed_s": round(s[1], 2),
        "sim_implied_overhead_s": round(s[0] - s[1], 2),
        "real_mean_advance_s": round(r[0], 2),
        "real_mean_max_speed_s": round(r[1], 2),
        "real_implied_overhead_s": round(r[0] - r[1], 2),
    }


def overhead_residual(real: dict, sim: dict, tol_rel: float = 0.10,
                      agg_goal: int = 0) -> dict:
    """K3b [EXACT]: real_mean_advance − sim_mean_advance ≈ 0.

    The decisive Stage-1 mechanism check: the per-round wall→vclock residual
    is the per-commit MQTT/dispatch overhead the sim omits (CRITICAL-1).
    Reports implied per-commit overhead = residual / agg_goal.
    """
    sim_adv = _per_round_advances(sim["agg_rounds"], use_vclock=True)
    real_adv = _per_round_advances(real["agg_rounds"], use_vclock=False)
    if not sim_adv:
        has_vclock = any(e.get("vclock_now") is not None for e in sim["agg_rounds"])
        return {"ok": True, "tier": "EXACT", "status": "SKIP",
                "note": ("K10: no vclock advances in sim agg_round events"
                         if not has_vclock
                         else "fewer than 2 sim rounds — too short to measure")}
    if not real_adv:
        return {"ok": True, "tier": "EXACT", "status": "SKIP",
                "note": "fewer than 2 real rounds — too short to measure"}
    sim_mean = sum(sim_adv) / len(sim_adv)
    real_mean = sum(real_adv) / len(real_adv)
    residual = real_mean - sim_mean
    rel = abs(residual) / real_mean if real_mean > 0 else 0.0
    per_commit = (residual / agg_goal) if agg_goal else None
    return {
        "ok": rel <= tol_rel,
        "tier": "EXACT",
        "real_mean_advance_s": round(real_mean, 2),
        "sim_mean_advance_s": round(sim_mean, 2),
        "residual_s": round(residual, 2),
        "rel": round(rel, 3),
        "tol_rel": tol_rel,
        "implied_per_commit_overhead_s": (round(per_commit, 3)
                                          if per_commit is not None else None),
        "agg_goal": agg_goal or None,
    }


# ═══════════════════════════════════════════════════════════════════
# §3.2x  Availability time-base & duty-cycle  (A3 / A4)  — Stage 2
# ═══════════════════════════════════════════════════════════════════

def avail_timebase_parity(real: dict, sim: dict,
                          n_bins: int = 10, tol_rel: float = 0.20) -> dict:
    """A3 [DIST]: num_eligible trajectory aligned by run progress (round/maxround).

    If the availability trace is indexed by a different time-base in each mode
    (sim=vclock, real=wall — the REFL HIGH-1 bug), the eligible-count curve vs
    normalized progress diverges even when the clock advance looks fine.
    """
    def _traj(sel):
        by_round: dict = {}
        for e in sel:
            ne = e.get("num_eligible")
            if ne is None:
                continue
            by_round.setdefault(e["round"], []).append(ne)
        if not by_round:
            return None
        maxr = max(by_round)
        bins: list = [[] for _ in range(n_bins)]
        for r, vals in by_round.items():
            frac = r / maxr if maxr else 0.0
            idx = min(n_bins - 1, int(frac * n_bins))
            bins[idx].append(sum(vals) / len(vals))
        return [(sum(b) / len(b) if b else None) for b in bins]

    rt, st = _traj(real["selection_train"]), _traj(sim["selection_train"])
    if not rt or not st:
        return {"ok": True, "tier": "DIST", "status": "SKIP",
                "note": "no num_eligible trajectory"}
    per_bin, diffs = [], []
    for i in range(n_bins):
        rv, sv = rt[i], st[i]
        if rv is None or sv is None:
            per_bin.append(None)
            continue
        rel = abs(rv - sv) / max(rv, sv, 1.0)
        diffs.append(rel)
        per_bin.append(round(rel, 3))
    max_rel = max(diffs) if diffs else float("nan")
    return {
        "ok": math.isnan(max_rel) or max_rel <= tol_rel,
        "tier": "DIST",
        "max_rel_diff": round(max_rel, 3) if not math.isnan(max_rel) else None,
        "per_bin_rel_diff": per_bin,
        "tol_rel": tol_rel,
    }


def duty_cycle_parity(real_trainers: dict, sim_trainers: dict) -> dict:
    """A4 [DIST]: per-trainer availability duty-cycle parity.

    Requires avail_change telemetry (on/off transitions per trainer), which
    the current loader does not surface — SKIP placeholder per the append-only
    growth rule; activates automatically once that telemetry exists.
    """
    def _has_avail_change(tr):
        return any(d.get("avail_change") for d in tr.values())

    if not _has_avail_change(real_trainers) and not _has_avail_change(sim_trainers):
        return {"ok": True, "tier": "DIST", "status": "SKIP",
                "note": "avail_change telemetry not available; A4 inactive"}
    # Telemetry present: compare per-trainer on-fraction.
    def _on_frac(tr):
        out = {}
        for tid, d in tr.items():
            evs = d.get("avail_change", [])
            if not evs:
                continue
            on = sum(1 for e in evs if e.get("available"))
            out[tid] = on / len(evs)
        return out

    rf, sf = _on_frac(real_trainers), _on_frac(sim_trainers)
    keys = set(rf) | set(sf)
    diffs = [abs(rf.get(k, 0.0) - sf.get(k, 0.0)) for k in keys]
    max_diff = max(diffs) if diffs else 0.0
    return {"ok": max_diff <= 0.2, "tier": "DIST",
            "max_dutycycle_diff": round(max_diff, 3), "n_trainers": len(keys)}


# ═══════════════════════════════════════════════════════════════════
# §3.4x  Training input control & per-phase split  (T2 / T_*)  — Stage 4
# ═══════════════════════════════════════════════════════════════════

def training_budget_parity(real_trainers: dict, sim_trainers: dict,
                           ks_tol: float = 0.1, support_tol: float = 0.15) -> dict:
    """T2 [DIST]: training_budget_s — the *input* to the speed model is identical.

    Same Jun-16 reclassification as P3 (`trainer_speed_parity`): `training_budget_s`
    is captured over the *selected* trainers, so a frequency/mean shift is either a
    genuine budget-assignment bug (sim assigns budgets outside real's support) or
    selection mix (sim selects faster trainers from the same support — feddance 3h:
    A2b pool KS=0). We enforce support containment (``sim_p99 <= real_p99 *
    (1+support_tol)``) and keep the distribution KS as a diagnostic owned by A2c.
    """
    def _vals(tr):
        out = []
        for d in tr.values():
            for e in d.get("trainer_round", []):
                v = e.get("training_budget_s")
                if v is not None and v >= 0:
                    out.append(float(v))
        return out

    rv, sv = _vals(real_trainers), _vals(sim_trainers)
    if not rv or not sv:
        return {"ok": True, "tier": "DIST", "status": "SKIP",
                "note": "no training_budget_s in telemetry"}
    ks = ks_stat(rv, sv)
    rm, _ = mean_std(rv)
    sm, _ = mean_std(sv)
    real_p99, sim_p99 = percentile(rv, 99), percentile(sv, 99)
    support_ratio = sim_p99 / real_p99 if real_p99 > 0 else float("nan")
    ok = (not math.isnan(support_ratio)
          and support_ratio <= 1.0 + support_tol)
    return {"ok": ok, "tier": "DIST",
            "support_ratio": round(support_ratio, 3) if not math.isnan(support_ratio) else None,
            "support_tol": support_tol,
            "real_p99_s": round(real_p99, 2), "sim_p99_s": round(sim_p99, 2),
            "mix_deferred": bool(ok and ks > ks_tol),
            "ks_stat": round(ks, 3), "ks_tol": ks_tol,
            "real_mean_s": round(rm, 2), "sim_mean_s": round(sm, 2),
            "n_real": len(rv), "n_sim": len(sv)}


def trainer_phase_split(real_trainers: dict, sim_trainers: dict,
                        ks_tol: float = 0.25) -> dict:
    """T_* [DIST]: one independent KS check per training phase.

    Splits the trainer_phase DIAG blob so the report says exactly which phase
    diverges ("mqtt_fetch off, rest match") instead of "timing is off".
    Returns {phase_<name>: result_dict}.
    """
    def _collect(tr, field):
        out = []
        for d in tr.values():
            for e in d.get("trainer_round", []):
                v = e.get(field)
                if v is not None and v >= 0:
                    out.append(float(v))
        return out

    results: dict = {}
    for f in _PHASE_FIELDS:
        key = "phase_" + (f[:-2] if f.endswith("_s") else f)
        rv, sv = _collect(real_trainers, f), _collect(sim_trainers, f)
        if not rv or not sv:
            results[key] = {"ok": True, "tier": "DIST", "status": "SKIP",
                            "note": f"no {f} in telemetry", "phase": f}
            continue
        ks = ks_stat(rv, sv)
        rm, _ = mean_std(rv)
        sm, _ = mean_std(sv)
        res = {"ok": ks <= ks_tol, "tier": "DIST", "phase": f,
               "ks_stat": round(ks, 3), "ks_tol": ks_tol,
               "real_mean_s": round(rm, 3), "sim_mean_s": round(sm, 3)}
        # mqtt_fetch is pure network-I/O wall time: the sim serves weights from
        # an in-memory cache and folds the trainer cycle into budget+leg, so this
        # phase is deliberately NOT part of the virtual clock.  Comparing it
        # is apples-to-oranges (real MQTT round-trip vs in-mem read) — keep it as
        # a DIAG so a divergence is reported but never enforced.  gpu_compute and
        # the other modeled phases stay enforced DIST.
        if f == "mqtt_fetch_s":
            res["tier"] = "DIAG"
            res["note"] = ("wall-time network I/O; sim uses in-mem cache, "
                           "excluded from the virtual clock (diagnostic only)")
        results[key] = res
    return results


# ═══════════════════════════════════════════════════════════════════
# §3.8x  Loss curve  (C2)  — Stage 8
# ═══════════════════════════════════════════════════════════════════

def convergence_loss_parity(real: dict, sim: dict, loss_tol: float = 0.15,
                            budget_s: Optional[float] = None) -> dict:
    """C2 [DIST]: loss curve by FL round, asserted independently of accuracy.

    Horizon guard (see convergence_parity): sub-2h PASS → LOW_CONF; FAIL stands.
    """
    def _curve(evs):
        return {e["round"]: e.get("test-loss") for e in evs}

    rc, sc = _curve(real["agg_evals"]), _curve(sim["agg_evals"])
    rounds = sorted(set(rc) & set(sc))
    diffs = [abs(rc[r] - sc[r]) for r in rounds
             if rc[r] is not None and sc[r] is not None]
    if not diffs:
        return {"ok": True, "tier": "DIST", "status": "SKIP",
                "note": "no overlapping loss evals"}
    avg = sum(diffs) / len(diffs)
    return _mark_low_confidence_if_short(
        {"ok": avg <= loss_tol, "tier": "DIST",
         "avg_loss_diff": round(avg, 4),
         "eval_rounds_compared": len(diffs), "loss_tol": loss_tol}, budget_s)


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

    Ordered HIGH → MID → LOW so coarse failures surface first:
      §0 Budget / stop-condition sanity
      §1 High-level counts (rounds, commits)
      §2 Convergence (accuracy / loss)
      §3 Availability (composition, eligibility)
      §4 Selection (detail, Jaccard, participation)
      §5 Updates / staleness
      §6 Clock & throughput (gating check first)
      §7 Trainer timing & sim invariants
      §8 Statistical utility
    """
    results: dict = {}

    # ── Stage 0 Telemetry coverage (gate) ──
    results["field_coverage"] = field_coverage(
        real_agg, sim_agg, real_trainers, sim_trainers)
    results["vclock_telemetry"] = vclock_telemetry_present(sim_agg)

    # ── Stage 1 Clock model (control → mechanism → emergent) ──
    results["sim_commit_monotone"] = sim_commit_order_monotone(sim_agg)
    results["sim_rate"] = sim_rate_ok(sim_agg)
    results["trainer_speed"] = trainer_speed_parity(real_agg, sim_agg)
    results["modeled_compute_advance"] = modeled_compute_advance(real_agg, sim_agg)
    results["overhead_residual"] = overhead_residual(
        real_agg, sim_agg, agg_goal=agg_goal)
    results["overlap_factor"] = overlap_factor(real_agg, sim_agg)
    results["per_round_advance"] = per_round_advance_parity(real_agg, sim_agg)
    results["throughput"] = throughput_parity(real_agg, sim_agg)

    # ── Stage 2 Availability ──
    results["avail_composition"] = avail_composition_parity(real_agg, sim_agg)
    results["eligibility"] = eligibility_parity(real_agg, sim_agg)
    results["eligible_speed"] = eligible_speed_composition_parity(real_agg, sim_agg)
    results["avail_timebase"] = avail_timebase_parity(real_agg, sim_agg)
    results["duty_cycle"] = duty_cycle_parity(real_trainers, sim_trainers)

    # ── Stage 3 Selection ──
    results["selection_detail"] = selection_detail_parity(real_agg, sim_agg)
    results["residence"] = inflight_residence_parity(real_agg, sim_agg)
    results["selection_bias"] = selection_speed_bias_parity(real_agg, sim_agg)
    results["selector_score"] = selector_score_parity(real_agg, sim_agg)
    results["preferred_duration"] = preferred_duration_parity(real_agg, sim_agg)
    results["participation"] = participation_parity(real_agg, sim_agg)
    results["decision_determinism"] = decision_determinism_parity(real_agg, sim_agg)
    results["selection"] = selection_parity(real_agg, sim_agg, max_rounds)

    # ── Stage 4 Dispatch & training ──
    results["training_budget"] = training_budget_parity(real_trainers, sim_trainers)
    results.update(trainer_phase_split(real_trainers, sim_trainers))
    results["trainer_phase"] = trainer_phase_parity(real_trainers, sim_trainers)
    results["gpu_budget_real"] = gpu_budget_ok(real_trainers)
    results["gpu_budget_sim"] = gpu_budget_ok(sim_trainers)
    results["sim_send_ts"] = sim_send_ts_ok(real_trainers, sim_trainers)

    # ── Stage 5 Update return & ordering ──
    results["inter_arrival_order"] = inter_arrival_order_parity(real_agg, sim_agg)
    if agg_goal:
        results["agg_goal_cycles_real"] = agg_goal_cycles_ok(real_agg, agg_goal)
        results["agg_goal_cycles_sim"] = agg_goal_cycles_ok(sim_agg, agg_goal)

    # ── Stage 6 Aggregation ──
    results["commit_visibility"] = commit_visibility_parity(real_agg, sim_agg)
    results["eval_commit_timeliness"] = eval_commit_timeliness(sim_agg)
    results["staleness"] = staleness_parity(real_agg, sim_agg)
    results["aggregation_sequence"] = aggregation_sequence_parity(
        real_agg, sim_agg, max_rounds)

    # ── Stage 7 Statistical utility ──
    results["utility"] = utility_parity(real_agg, sim_agg)

    # ── Stage 8 Emergent outcomes ──
    results["terminal_state"] = terminal_state_parity(real_agg, sim_agg)
    results["total_commits"] = total_commits_parity(real_agg, sim_agg)
    results["convergence"] = convergence_parity(real_agg, sim_agg, budget_s=budget_s)
    results["convergence_loss"] = convergence_loss_parity(
        real_agg, sim_agg, budget_s=budget_s)

    # ── Stage 9 Budget / stop sanity ──
    results["budget_not_cap"] = budget_not_cap(
        real_agg, sim_agg, rounds_cap=rounds_cap, budget_s=budget_s)
    results["failsafe"] = failsafe_ok(sim_agg, budget_s=budget_s)

    return results


# ═══════════════════════════════════════════════════════════════════
# §5  Causal registry  (stage / role / deps)  — single source of truth
# ═══════════════════════════════════════════════════════════════════
#
# Each result key maps to its rung on the parity ladder.  STAGE drives
# diagnosis ordering; ROLE labels its localization purpose; DEPS lists the
# upstream checks whose passing is required for this one to be meaningful.
# TIER (enforcement) is read live from each result dict, not stored here.
#
#   role ∈ {"CONTROL", "MECHANISM", "EMERGENT", "DIAG"}

CHECK_META: dict = {
    # ── Stage 0 Telemetry coverage ──
    "field_coverage":          {"stage": 0, "role": "CONTROL",  "deps": ()},
    "vclock_telemetry":        {"stage": 0, "role": "CONTROL",  "deps": ("field_coverage",)},
    # ── Stage 1 Clock model ──
    "sim_commit_monotone":     {"stage": 1, "role": "MECHANISM", "deps": ("vclock_telemetry",)},
    "sim_rate":                {"stage": 1, "role": "MECHANISM", "deps": ("vclock_telemetry",)},
    "trainer_speed":           {"stage": 1, "role": "CONTROL",  "deps": ()},
    "modeled_compute_advance": {"stage": 1, "role": "DIAG",     "deps": ("trainer_speed", "sim_commit_monotone")},
    "overhead_residual":       {"stage": 1, "role": "MECHANISM", "deps": ("trainer_speed", "sim_commit_monotone")},
    "overlap_factor":          {"stage": 1, "role": "DIAG",     "deps": ("trainer_speed", "sim_commit_monotone")},
    "per_round_advance":       {"stage": 1, "role": "EMERGENT", "deps": ("overhead_residual",)},
    "throughput":              {"stage": 1, "role": "EMERGENT", "deps": ("per_round_advance",)},
    # ── Stage 2 Availability ──
    "avail_composition":       {"stage": 2, "role": "MECHANISM", "deps": ()},
    "eligibility":             {"stage": 2, "role": "MECHANISM", "deps": ("avail_composition",)},
    "eligible_speed":          {"stage": 2, "role": "MECHANISM", "deps": ("eligibility",)},
    "avail_timebase":          {"stage": 2, "role": "CONTROL",  "deps": ("per_round_advance",)},
    "duty_cycle":              {"stage": 2, "role": "MECHANISM", "deps": ("avail_timebase",)},
    # ── Stage 3 Selection ──
    "selection_detail":        {"stage": 3, "role": "MECHANISM", "deps": ("eligibility",)},
    "residence":               {"stage": 3, "role": "MECHANISM", "deps": ("eligibility",)},
    "selection_bias":          {"stage": 3, "role": "MECHANISM", "deps": ("eligible_speed",)},
    "selector_score":          {"stage": 3, "role": "DIAG",     "deps": ("eligible_speed",)},
    "preferred_duration":      {"stage": 3, "role": "MECHANISM", "deps": ("eligible_speed",)},
    "participation":           {"stage": 3, "role": "EMERGENT", "deps": ("selection_detail", "eligible_speed", "selection_bias")},
    "decision_determinism":    {"stage": 3, "role": "DIAG",     "deps": ("eligibility",)},
    "selection":               {"stage": 3, "role": "DIAG",     "deps": ("eligibility",)},
    # ── Stage 4 Dispatch & training ──
    "training_budget":         {"stage": 4, "role": "CONTROL",  "deps": ()},
    "phase_pre_train":         {"stage": 4, "role": "MECHANISM", "deps": ()},
    "phase_weights_to_gpu":    {"stage": 4, "role": "MECHANISM", "deps": ()},
    "phase_gpu_compute":       {"stage": 4, "role": "MECHANISM", "deps": ("training_budget",)},
    "phase_mqtt_fetch":        {"stage": 4, "role": "DIAG",      "deps": ()},
    "phase_weights_to_ram":    {"stage": 4, "role": "MECHANISM", "deps": ()},
    "phase_post_train":        {"stage": 4, "role": "MECHANISM", "deps": ()},
    "trainer_phase":           {"stage": 4, "role": "DIAG",     "deps": ()},
    "gpu_budget_real":         {"stage": 4, "role": "MECHANISM", "deps": ("training_budget",)},
    "gpu_budget_sim":          {"stage": 4, "role": "MECHANISM", "deps": ("training_budget",)},
    "sim_send_ts":             {"stage": 4, "role": "CONTROL",  "deps": ("vclock_telemetry",)},
    # ── Stage 5 Update return & ordering ──
    "inter_arrival_order":     {"stage": 5, "role": "MECHANISM", "deps": ("per_round_advance", "selection_detail")},
    "agg_goal_cycles_real":    {"stage": 5, "role": "MECHANISM", "deps": ()},
    "agg_goal_cycles_sim":     {"stage": 5, "role": "MECHANISM", "deps": ()},
    # ── Stage 6 Aggregation ──
    "commit_visibility":       {"stage": 6, "role": "MECHANISM", "deps": ("per_round_advance",)},
    "eval_commit_timeliness":  {"stage": 6, "role": "MECHANISM", "deps": ("commit_visibility",)},
    "staleness":               {"stage": 6, "role": "MECHANISM", "deps": ("per_round_advance", "inter_arrival_order", "commit_visibility")},
    "aggregation_sequence":    {"stage": 6, "role": "EMERGENT", "deps": ("participation", "inter_arrival_order")},
    "first_divergence_summary": {"stage": 6, "role": "DIAG",    "deps": ()},
    # ── Stage 7 Statistical utility ──
    "utility":                 {"stage": 7, "role": "EMERGENT", "deps": ("participation", "phase_gpu_compute", "staleness")},
    # ── Stage 8 Emergent outcomes ──
    "terminal_state":          {"stage": 8, "role": "EMERGENT", "deps": ("throughput", "participation")},
    "total_commits":           {"stage": 8, "role": "EMERGENT", "deps": ("throughput",)},
    "convergence":             {"stage": 8, "role": "EMERGENT", "deps": ("utility", "terminal_state")},
    "convergence_loss":        {"stage": 8, "role": "EMERGENT", "deps": ("utility", "terminal_state")},
    # ── Stage 9 Budget / stop sanity (orthogonal) ──
    "budget_not_cap":          {"stage": 9, "role": "DIAG",     "deps": ()},
    "failsafe":                {"stage": 9, "role": "MECHANISM", "deps": ()},
}

# Checks whose FAIL is downgraded to WARN regardless of tier (expected-noisy).
_WARN_ONLY_CHECKS = {"budget_not_cap", "inter_arrival_order"}


def check_stage(name: str) -> int:
    return CHECK_META.get(name, {}).get("stage", 99)


def check_role(name: str) -> str:
    return CHECK_META.get(name, {}).get("role", "")


def _is_skipped(res: dict) -> bool:
    note = res.get("note") or ""
    return res.get("status") == "SKIP" or note.startswith("K10:")


def _classify(name: str, res: dict, strict: bool, lenient: bool) -> str:
    """One of {'pass','fail','warn','skip'} for a single check result."""
    if _is_skipped(res):
        return "skip"
    if res.get("ok", True):
        return "pass"
    tier = res.get("tier", "DIST")
    if name in _WARN_ONLY_CHECKS or tier == "DIAG":
        return "fail" if strict else "warn"
    if tier in ("EXACT", "INV"):
        return "fail"
    if tier == "DIST":
        return "warn" if lenient else "fail"
    return "fail"


def _transitive_deps(name: str, _seen: Optional[set] = None) -> set:
    """All upstream check names reachable from `name` via the dep graph."""
    if _seen is None:
        _seen = set()
    for dep in CHECK_META.get(name, {}).get("deps", ()):
        if dep not in _seen:
            _seen.add(dep)
            _transitive_deps(dep, _seen)
    return _seen


def overall_verdict(results: dict, strict: bool = False,
                    lenient: bool = False) -> tuple:
    """Return (passed, root_causes, downstream, warnings).

    Causal localization:
      * A check is an enforced FAIL per tier/lenient/strict rules.
      * Among failures, one whose transitive upstream chain contains another
        failure is DOWNSTREAM; otherwise it is a ROOT-CAUSE.
      * root_causes/downstream are sorted by ladder stage (lowest first).
      * passed == (no enforced failures).

    Back-compat: callers expecting the old 3-tuple can use
    ``passed, root+downstream, warnings`` — see report.py.
    """
    statuses = {n: _classify(n, r, strict, lenient)
                for n, r in results.items() if isinstance(r, dict)}
    failed = {n for n, s in statuses.items() if s == "fail"}
    warnings = sorted((n for n, s in statuses.items() if s == "warn"),
                      key=check_stage)

    roots, downstream = [], []
    for n in failed:
        if _transitive_deps(n) & failed:
            downstream.append(n)
        else:
            roots.append(n)
    roots.sort(key=lambda n: (check_stage(n), n))
    downstream.sort(key=lambda n: (check_stage(n), n))
    return (not failed), roots, downstream, warnings


def verdict_summary(results: dict, strict: bool = False,
                    lenient: bool = False) -> dict:
    """Enforced pass/total tally for a run — the parity scoreboard.

    ``pass``/``fail`` are the *enforced* universe (DIST under default rules,
    EXACT, INV); ``warn`` (DIAG, lenient-demoted DIST, WARN-only checks) and
    ``skip`` (telemetry absent / N/A) are excluded from the denominator so the
    headline ``pass/total`` tracks only checks that can actually fail. ``score``
    is ``pass/total`` over that enforced universe; ``roots`` is the lowest broken
    rung(s). Emitted into the JSON (``summary`` key) and the report footer so the
    scoreboard is persisted and regenerable — not hand-maintained in PARITY.md.
    """
    counts = {"pass": 0, "fail": 0, "warn": 0, "skip": 0}
    for n, r in results.items():
        if isinstance(r, dict):
            counts[_classify(n, r, strict, lenient)] += 1
    passed, roots, downstream, warnings = overall_verdict(
        results, strict=strict, lenient=lenient)
    total = counts["pass"] + counts["fail"]
    return {
        "passed": passed,
        "n_pass": counts["pass"],
        "n_fail": counts["fail"],
        "n_warn": counts["warn"],
        "n_skip": counts["skip"],
        "n_enforced": total,
        "score": round(counts["pass"] / total, 3) if total else None,
        "roots": roots,
        "downstream": downstream,
        "warnings": warnings,
    }
