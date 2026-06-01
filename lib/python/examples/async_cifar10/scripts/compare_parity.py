"""
Parity comparator: real run vs simulated run.

Checks:
  1. Selection parity        — per round: same trainers selected?
  2. Statistical utility     — per-trainer utility distributions match?
  3. Aggregation sequence    — same contributing-trainer order per round?
  4. Staleness               — distributions match?
  5. Participation counts    — each trainer used similar # of times?
  6. Convergence             — accuracy / loss from agg_eval events
  7. Round duration parity   — per-trainer sim_round_duration_s distributions match?
  8. sim_send_ts presence    — sim mode: vclock stamps non-zero; real mode: null
  9. GPU contention          — actual GPU time vs budget per trainer; detect overruns

Usage:
    python compare_parity.py \\
        --real  experiments/run_.../telemetry/aggregator_*.jsonl \\
        --sim   experiments/run_.../telemetry/aggregator_*.jsonl \\
        [--real-trainer-dir experiments/run_.../telemetry/] \\
        [--sim-trainer-dir  experiments/run_.../telemetry/] \\
        [--rounds 100] [--strict] [--plot-out timing.png] [--json-out results.json]

Exit 0 = all checks passed (or within tolerance)
Exit 1 = one or more checks failed
"""

import argparse
import collections
import json
import math
import sys
from pathlib import Path
from typing import Optional


# ── helpers ──────────────────────────────────────────────────────────────────

def short(end_id: str) -> str:
    return end_id[-4:] if end_id else "None"


def load_trainer_jsonl_dir(telemetry_dir: Optional[str]) -> dict:
    """Load all trainer_*.jsonl files from a telemetry dir.

    Returns {short_id: {"task_recv": [...], "trainer_round": [...]}}
    """
    if not telemetry_dir:
        return {}
    d = Path(telemetry_dir)
    result = {}
    for f in sorted(d.glob("trainer_*.jsonl")):
        short_id = f.stem[-4:]  # last 4 hex chars
        task_recv_evs, trainer_round_evs = [], []
        with open(f) as fp:
            for line in fp:
                try:
                    e = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                ev = e.get("event")
                if ev == "task_recv":
                    task_recv_evs.append(e)
                elif ev == "trainer_round":
                    trainer_round_evs.append(e)
        result[short_id] = {"task_recv": task_recv_evs, "trainer_round": trainer_round_evs}
    return result


def load_agg_jsonl(path: str) -> dict:
    """Parse an aggregator telemetry JSONL into typed lists."""
    selection_train: list[dict] = []   # task=train selection events
    agg_rounds: list[dict] = []        # per-update aggregation events
    agg_evals: list[dict] = []         # per-round evaluation events

    with open(path) as f:
        for line in f:
            e = json.loads(line.strip())
            ev = e.get("event")
            if ev == "selection" and e.get("task") == "train":
                selection_train.append(e)
            elif ev == "agg_round":
                agg_rounds.append(e)
            elif ev == "agg_eval":
                agg_evals.append(e)

    # Sort by round then ts
    selection_train.sort(key=lambda x: (x["round"], x["ts"]))
    agg_rounds.sort(key=lambda x: (x["round"], x.get("agg_goal_count", 0), x["ts"]))
    agg_evals.sort(key=lambda x: x["round"])

    return {
        "selection_train": selection_train,
        "agg_rounds": agg_rounds,
        "agg_evals": agg_evals,
    }


def ks_stat(a: list[float], b: list[float]) -> float:
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


def mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    v = sum((x - m) ** 2 for x in vals) / len(vals)
    return m, math.sqrt(v)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# ── check functions ───────────────────────────────────────────────────────────

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


def check_selection_parity(real: dict, sim: dict, max_rounds: Optional[int]) -> dict:
    """Check 1: per-round selection — same trainers chosen?"""
    # group by outer round
    def by_round(sel_events):
        d = collections.defaultdict(set)
        for e in sel_events:
            r = e["round"]
            d[r].update(e.get("chosen", []))
        return d

    r_sel = by_round(real["selection_train"])
    s_sel = by_round(sim["selection_train"])

    rounds = sorted(set(r_sel) & set(s_sel))
    if max_rounds:
        rounds = [r for r in rounds if r <= max_rounds]

    jaccards = []
    exact_match = 0
    per_round_detail = []
    for r in rounds:
        rs, ss = r_sel[r], s_sel[r]
        j = jaccard(rs, ss)
        jaccards.append(j)
        exact = rs == ss
        if exact:
            exact_match += 1
        per_round_detail.append((r, rs, ss, j, exact))

    mean_j = sum(jaccards) / len(jaccards) if jaccards else float("nan")
    exact_pct = exact_match / len(rounds) * 100 if rounds else 0.0

    status = PASS if mean_j >= 0.7 else (WARN if mean_j >= 0.4 else FAIL)

    # Find rounds with lowest Jaccard
    worst = sorted(per_round_detail, key=lambda x: x[3])[:5]

    return {
        "status": status,
        "rounds_compared": len(rounds),
        "exact_match_pct": round(exact_pct, 1),
        "mean_jaccard": round(mean_j, 3),
        "worst_rounds": [
            {"round": r, "real": sorted(short(t) for t in rs),
             "sim": sorted(short(t) for t in ss), "jaccard": round(j, 3)}
            for r, rs, ss, j, _ in worst
        ],
    }


def check_utility_parity(real: dict, sim: dict) -> dict:
    """Check 2: per-trainer stat_utility distributions match?"""
    def per_trainer_utils(agg_rounds):
        d = collections.defaultdict(list)
        for e in agg_rounds:
            for t, u in zip(e.get("contributing_trainers", []), e.get("stat_utility", [])):
                if u is not None:
                    d[t].append(u)
        return d

    r_utils = per_trainer_utils(real["agg_rounds"])
    s_utils = per_trainer_utils(sim["agg_rounds"])

    all_trainers = sorted(set(r_utils) | set(s_utils))
    rows = []
    ks_stats = []
    mean_diffs = []
    for t in all_trainers:
        ru = r_utils.get(t, [])
        su = s_utils.get(t, [])
        ks = ks_stat(ru, su)
        rm, rs_ = mean_std(ru)
        sm, ss_ = mean_std(su)
        diff = abs(rm - sm) if not math.isnan(rm) and not math.isnan(sm) else float("nan")
        rows.append({
            "trainer": short(t),
            "real_n": len(ru), "sim_n": len(su),
            "real_mean": round(rm, 2), "sim_mean": round(sm, 2),
            "mean_diff": round(diff, 2),
            "ks_stat": round(ks, 3) if not math.isnan(ks) else None,
        })
        if not math.isnan(ks):
            ks_stats.append(ks)
        if not math.isnan(diff):
            mean_diffs.append(diff)

    max_ks = max(ks_stats) if ks_stats else float("nan")
    avg_mean_diff = sum(mean_diffs) / len(mean_diffs) if mean_diffs else float("nan")

    # KS < 0.2 = distributions match well; < 0.3 = acceptable
    status = PASS if max_ks < 0.2 else (WARN if max_ks < 0.35 else FAIL)

    return {
        "status": status,
        "max_ks_stat": round(max_ks, 3) if not math.isnan(max_ks) else None,
        "avg_mean_utility_diff": round(avg_mean_diff, 2) if not math.isnan(avg_mean_diff) else None,
        "per_trainer": rows,
    }


def check_aggregation_sequence(real: dict, sim: dict, max_rounds: Optional[int]) -> dict:
    """Check 3: per-round aggregation sequence — same contributing trainers?"""
    def by_round(agg_rounds):
        d = collections.defaultdict(list)
        for e in agg_rounds:
            r = e["round"]
            if max_rounds and r > max_rounds:
                continue
            t = e["contributing_trainers"][0] if e["contributing_trainers"] else None
            d[r].append(t)
        return d

    r_seq = by_round(real["agg_rounds"])
    s_seq = by_round(sim["agg_rounds"])

    rounds = sorted(set(r_seq) & set(s_seq))
    exact_rounds = 0
    set_match_rounds = 0
    per_round_detail = []
    for r in rounds:
        rs, ss = r_seq[r], s_seq[r]
        exact = rs == ss
        set_match = set(t for t in rs if t) == set(t for t in ss if t)
        if exact:
            exact_rounds += 1
        if set_match:
            set_match_rounds += 1
        if not exact:
            per_round_detail.append({
                "round": r,
                "real_seq": [short(t) for t in rs],
                "sim_seq": [short(t) for t in ss],
                "set_match": set_match,
            })

    exact_pct = exact_rounds / len(rounds) * 100 if rounds else 0.0
    set_pct = set_match_rounds / len(rounds) * 100 if rounds else 0.0

    status = PASS if set_pct >= 70 else (WARN if set_pct >= 40 else FAIL)

    return {
        "status": status,
        "rounds_compared": len(rounds),
        "exact_sequence_match_pct": round(exact_pct, 1),
        "set_match_pct": round(set_pct, 1),
        "mismatched_rounds_sample": per_round_detail[:10],
    }


def check_staleness(real: dict, sim: dict) -> dict:
    """Check 4: staleness distributions match?"""
    def get_staleness(agg_rounds):
        vals = []
        for e in agg_rounds:
            vals.extend(e.get("staleness", []))
        return vals

    rs = get_staleness(real["agg_rounds"])
    ss = get_staleness(sim["agg_rounds"])

    r_counts = collections.Counter(rs)
    s_counts = collections.Counter(ss)
    all_k = sorted(set(r_counts) | set(s_counts))

    ks = ks_stat([float(x) for x in rs], [float(x) for x in ss])
    r_mean, _ = mean_std([float(x) for x in rs])
    s_mean, _ = mean_std([float(x) for x in ss])

    status = PASS if abs(r_mean - s_mean) < 1.0 else (WARN if abs(r_mean - s_mean) < 2.0 else FAIL)

    return {
        "status": status,
        "real_mean_staleness": round(r_mean, 3),
        "sim_mean_staleness": round(s_mean, 3),
        "mean_staleness_diff": round(abs(r_mean - s_mean), 3),
        "ks_stat": round(ks, 3) if not math.isnan(ks) else None,
        "real_distribution": {k: r_counts.get(k, 0) for k in all_k},
        "sim_distribution": {k: s_counts.get(k, 0) for k in all_k},
    }


def check_participation(real: dict, sim: dict) -> dict:
    """Check 5: per-trainer participation counts match?"""
    def counts(agg_rounds):
        c = collections.Counter()
        for e in agg_rounds:
            for t in e.get("contributing_trainers", []):
                c[t] += 1
        return c

    rc = counts(real["agg_rounds"])
    sc = counts(sim["agg_rounds"])
    trainers = sorted(set(rc) | set(sc))

    rows = []
    diffs = []
    for t in trainers:
        r, s = rc.get(t, 0), sc.get(t, 0)
        d = abs(r - s)
        diffs.append(d)
        rows.append({"trainer": short(t), "real": r, "sim": s, "abs_diff": d})

    rows.sort(key=lambda x: -x["abs_diff"])
    avg_diff = sum(diffs) / len(diffs) if diffs else 0.0
    max_diff = max(diffs) if diffs else 0

    total_r = sum(rc.values())
    total_s = sum(sc.values())

    status = PASS if avg_diff < 10 else (WARN if avg_diff < 25 else FAIL)

    return {
        "status": status,
        "total_updates_real": total_r,
        "total_updates_sim": total_s,
        "avg_participation_diff": round(avg_diff, 1),
        "max_participation_diff": max_diff,
        "per_trainer": rows,
    }


def check_convergence(real: dict, sim: dict) -> dict:
    """Check 6: accuracy/loss curves."""
    def curve(agg_evals):
        return {e["round"]: {"acc": e.get("test-accuracy"), "loss": e.get("test-loss")}
                for e in agg_evals}

    rc = curve(real["agg_evals"])
    sc = curve(real["agg_evals"])  # same source intentional for self-check
    # use actual sim curve
    sc = curve(sim["agg_evals"])

    rounds = sorted(set(rc) & set(sc))
    if not rounds:
        return {"status": WARN, "note": "no overlapping eval rounds"}

    acc_diffs = []
    loss_diffs = []
    rows = []
    for r in rounds:
        ra, rl = rc[r].get("acc"), rc[r].get("loss")
        sa, sl = sc[r].get("acc"), sc[r].get("loss")
        if ra is not None and sa is not None:
            acc_diffs.append(abs(ra - sa))
        if rl is not None and sl is not None:
            loss_diffs.append(abs(rl - sl))
        rows.append({
            "round": r,
            "real_acc": round(ra, 4) if ra is not None else None,
            "sim_acc": round(sa, 4) if sa is not None else None,
            "real_loss": round(rl, 4) if rl is not None else None,
            "sim_loss": round(sl, 4) if sl is not None else None,
        })

    avg_acc_diff = sum(acc_diffs) / len(acc_diffs) if acc_diffs else float("nan")
    avg_loss_diff = sum(loss_diffs) / len(loss_diffs) if loss_diffs else float("nan")

    # Within 5pp accuracy = PASS; within 10pp = WARN
    status = PASS
    if not math.isnan(avg_acc_diff):
        status = PASS if avg_acc_diff < 0.05 else (WARN if avg_acc_diff < 0.10 else FAIL)

    return {
        "status": status,
        "eval_rounds_compared": len(rounds),
        "avg_accuracy_diff": round(avg_acc_diff, 4) if not math.isnan(avg_acc_diff) else None,
        "avg_loss_diff": round(avg_loss_diff, 4) if not math.isnan(avg_loss_diff) else None,
        "curve": rows,
    }


def check_round_duration_parity(real_trainers: dict, sim_trainers: dict) -> dict:
    """Check 7: per-trainer sim_round_duration_s distributions match?

    Sim mode: sim_round_duration_s = remaining_time = max(0, D - gpu_time) ≈ D - gpu.
    Real mode: sim_round_duration_s = gpu + remaining_time = max(gpu, D) = D (no overrun).
    Expected mean diff ≈ gpu_time (small, ~0.5s for fast trainers).
    Large diffs indicate overruns or dataset-size divergence.
    """
    all_ids = sorted(set(real_trainers) | set(sim_trainers))
    if not all_ids:
        return {"status": WARN, "note": "no trainer telemetry dirs provided"}

    rows, ks_stats, mean_diffs = [], [], []
    for tid in all_ids:
        r_evs = real_trainers.get(tid, {}).get("trainer_round", [])
        s_evs = sim_trainers.get(tid, {}).get("trainer_round", [])
        r_durs = [e["sim_round_duration_s"] for e in r_evs if "sim_round_duration_s" in e]
        s_durs = [e["sim_round_duration_s"] for e in s_evs if "sim_round_duration_s" in e]
        ks = ks_stat(r_durs, s_durs)
        rm, _ = mean_std(r_durs)
        sm, _ = mean_std(s_durs)
        diff = abs(rm - sm) if not (math.isnan(rm) or math.isnan(sm)) else float("nan")
        rows.append({"trainer": tid, "real_mean_s": round(rm, 3), "sim_mean_s": round(sm, 3),
                     "mean_diff_s": round(diff, 3), "ks_stat": round(ks, 3) if not math.isnan(ks) else None,
                     "n_real": len(r_durs), "n_sim": len(s_durs)})
        if not math.isnan(ks):
            ks_stats.append(ks)
        if not math.isnan(diff):
            mean_diffs.append(diff)

    max_ks = max(ks_stats) if ks_stats else float("nan")
    avg_diff = sum(mean_diffs) / len(mean_diffs) if mean_diffs else float("nan")
    status = PASS if (math.isnan(max_ks) or max_ks < 0.2) else (WARN if max_ks < 0.4 else FAIL)
    return {"status": status, "max_ks_stat": round(max_ks, 3) if not math.isnan(max_ks) else None,
            "avg_mean_diff_s": round(avg_diff, 3) if not math.isnan(avg_diff) else None,
            "per_trainer": rows}


def check_sim_send_ts(real_trainers: dict, sim_trainers: dict) -> dict:
    """Check 8: sim_send_ts correctness.

    Real mode:  task_recv.sim_send_ts should be None (aggregator doesn't stamp vclock there).
    Sim mode:   task_recv.sim_send_ts must be non-None and must increase over rounds,
                confirming the SIM_SEND_TS fix is in effect.  A flat 0.0 across all rounds
                indicates the bug is still present (trainer using _sim_now()=0 as base).
    """
    issues = []

    # real: all sim_send_ts should be null
    for tid, data in real_trainers.items():
        non_null = [e["sim_send_ts"] for e in data.get("task_recv", []) if e.get("sim_send_ts") is not None]
        if non_null:
            issues.append(f"real/{tid}: unexpected non-null sim_send_ts values: {non_null[:3]}")

    # sim: sim_send_ts should be non-null and non-trivially non-zero after round 1
    sim_ok, sim_all_zero, sim_missing = 0, 0, 0
    for tid, data in sim_trainers.items():
        evs = [e for e in data.get("task_recv", []) if e.get("round", 0) > 1]
        if not evs:
            sim_missing += 1
            continue
        vals = [e.get("sim_send_ts") for e in evs]
        nulls = [v for v in vals if v is None]
        zeros = [v for v in vals if v is not None and v == 0.0]
        non_zeros = [v for v in vals if v is not None and v > 0.0]
        if nulls:
            issues.append(f"sim/{tid}: {len(nulls)} null sim_send_ts values — SIM_SEND_TS fix may not be active")
            sim_all_zero += 1
        elif not non_zeros:
            issues.append(f"sim/{tid}: all sim_send_ts==0.0 (rounds>1) — vclock not advancing")
            sim_all_zero += 1
        else:
            sim_ok += 1

    status = PASS if not issues else FAIL
    return {"status": status, "issues": issues, "sim_trainers_ok": sim_ok,
            "sim_trainers_flat_zero": sim_all_zero, "sim_trainers_no_data": sim_missing}


def check_gpu_contention(real_trainers: dict, sim_trainers: dict) -> dict:
    """Check 9: GPU contention — does actual GPU time respect the per-trainer budget?

    Reads training_budget_s and real_gpu_time_s from trainer_round events.
    Overrun fraction = rounds where gpu_time > budget.
    FAIL if mean overrun fraction > 25%; WARN if > 10%.
    Old runs without training_budget_s in telemetry are skipped with WARN.
    """
    def _extract(trainers):
        stats = {}
        for tid, d in trainers.items():
            evs = [e for e in d.get("trainer_round", [])
                   if "real_gpu_time_s" in e and e.get("training_budget_s", 0) > 0]
            if not evs:
                continue
            gpu = [e["real_gpu_time_s"] for e in evs]
            bgt = [e["training_budget_s"] for e in evs]
            overruns = [g > b for g, b in zip(gpu, bgt)]
            stats[tid] = {
                "n": len(evs),
                "budget_s": round(bgt[0], 2),
                "mean_gpu_s": round(sum(gpu) / len(gpu), 3),
                "max_gpu_s": round(max(gpu), 3),
                "overrun_frac": round(sum(overruns) / len(overruns), 3),
                "overran_rounds": int(sum(overruns)),
            }
        return stats

    real_st = _extract(real_trainers)
    sim_st  = _extract(sim_trainers)

    if not real_st and not sim_st:
        return {"status": WARN,
                "note": "no training_budget_s in telemetry (old run without timing model changes)"}

    def _agg(st):
        if not st:
            return None
        fracs = [v["overrun_frac"] for v in st.values()]
        return {
            "n_trainers": len(st),
            "mean_overrun_frac": round(sum(fracs) / len(fracs), 3),
            "max_overrun_frac": round(max(fracs), 3),
            "trainers_with_any_overrun": int(sum(1 for f in fracs if f > 0)),
        }

    real_agg = _agg(real_st)
    sim_agg  = _agg(sim_st)

    worst = max(
        (real_agg or {}).get("mean_overrun_frac", 0),
        (sim_agg  or {}).get("mean_overrun_frac", 0),
    )
    status = FAIL if worst > 0.25 else (WARN if worst > 0.10 else PASS)

    all_tids = sorted(set(real_st) | set(sim_st))
    rows = []
    for tid in all_tids:
        r, s = real_st.get(tid, {}), sim_st.get(tid, {})
        rows.append({
            "trainer": tid,
            "budget_s":          (r or s).get("budget_s"),
            "real_mean_gpu_s":   r.get("mean_gpu_s"),
            "real_max_gpu_s":    r.get("max_gpu_s"),
            "real_overrun_frac": r.get("overrun_frac"),
            "sim_mean_gpu_s":    s.get("mean_gpu_s"),
            "sim_max_gpu_s":     s.get("max_gpu_s"),
            "sim_overrun_frac":  s.get("overrun_frac"),
        })
    rows.sort(key=lambda x: max(
        x.get("real_overrun_frac") or 0,
        x.get("sim_overrun_frac") or 0,
    ), reverse=True)

    return {
        "status":      status,
        "real_summary": real_agg,
        "sim_summary":  sim_agg,
        "per_trainer":  rows,
    }


def plot_timing_sanity(real_trainers: dict, sim_trainers: dict, out_path: str) -> None:
    """Plot actual GPU time vs expected budget per trainer per round.

    Layout (2 rows × N_modes cols):
      Top:    per-trainer bar (mean GPU time) + red budget marker per trainer
      Bottom: round-by-round deviation (GPU − budget); positive = contention

    If contention is present the bars will exceed the red markers and the
    bottom panel will show positive spikes.  If everything is healthy the
    bars stay well below the red markers and the bottom panel stays negative.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not available; skipping timing plot")
        return

    def _extract(trainers):
        data = {}
        for tid, d in trainers.items():
            evs = sorted(
                [e for e in d.get("trainer_round", [])
                 if "real_gpu_time_s" in e and e.get("training_budget_s", 0) > 0],
                key=lambda x: x.get("round", 0),
            )
            if not evs:
                continue
            data[tid] = {
                "rounds": [e["round"] for e in evs],
                "gpu":    [e["real_gpu_time_s"] for e in evs],
                "budget": evs[0]["training_budget_s"],
            }
        return data

    real_data = _extract(real_trainers) if real_trainers else {}
    sim_data  = _extract(sim_trainers)  if sim_trainers  else {}

    modes = [(lbl, d) for lbl, d in [("Real", real_data), ("Sim", sim_data)] if d]
    if not modes:
        print("[plot] No training_budget_s data found; skipping timing plot")
        return

    ncols = len(modes)
    fig, axes = plt.subplots(2, ncols, figsize=(7 * ncols, 9), squeeze=False)
    fig.suptitle("GPU Training Time vs Budget  (Contention Sanity Check)", fontsize=13)
    cmap = plt.cm.tab10

    for col, (label, data) in enumerate(modes):
        tids = sorted(data.keys(), key=lambda t: data[t]["budget"])
        colors = {t: cmap(i % 10) for i, t in enumerate(tids)}
        xs = list(range(len(tids)))

        # ── top: per-trainer mean GPU bar + budget marker ──────────────────
        ax_t = axes[0][col]
        mean_gpus = [sum(data[t]["gpu"]) / len(data[t]["gpu"]) for t in tids]
        budgets   = [data[t]["budget"] for t in tids]

        ax_t.bar(xs, mean_gpus, color=[colors[t] for t in tids], alpha=0.75, zorder=2,
                 label="mean GPU time")
        for x, b in zip(xs, budgets):
            ax_t.plot([x - 0.38, x + 0.38], [b, b], color="red", linewidth=2.5, zorder=3)
        # dummy line for legend
        ax_t.plot([], [], color="red", linewidth=2.5, label="budget (D)")

        for x, g, b in zip(xs, mean_gpus, budgets):
            if g > b:
                ax_t.annotate("OVER", xy=(x, g), ha="center", va="bottom",
                              color="red", fontsize=7, fontweight="bold")

        ax_t.set_xticks(xs)
        ax_t.set_xticklabels([f"...{t}" for t in tids], rotation=45, ha="right", fontsize=8)
        ax_t.set_ylabel("seconds")
        ax_t.set_title(f"{label}: mean GPU time vs budget", fontsize=10)
        ax_t.set_ylim(bottom=0)
        ax_t.legend(fontsize=8)
        ax_t.grid(axis="y", alpha=0.3)

        # ── bottom: round-by-round deviation (mean + max across trainers) ──
        ax_b = axes[1][col]
        devs_per_round: dict = collections.defaultdict(list)
        for t in tids:
            bgt = data[t]["budget"]
            for r, g in zip(data[t]["rounds"], data[t]["gpu"]):
                devs_per_round[r].append(g - bgt)

        if devs_per_round:
            rds = sorted(devs_per_round.keys())
            means = [sum(devs_per_round[r]) / len(devs_per_round[r]) for r in rds]
            maxes = [max(devs_per_round[r]) for r in rds]

            ax_b.fill_between(rds, means, maxes, alpha=0.20, color="orange")
            ax_b.plot(rds, means, color="steelblue", linewidth=1.5, label="mean deviation")
            ax_b.plot(rds, maxes, color="darkorange", linewidth=1.0, alpha=0.8,
                      label="max deviation")
            ax_b.fill_between(rds, 0, maxes,
                              where=[m > 0 for m in maxes],
                              alpha=0.10, color="red")
            ax_b.axhline(0, color="red", linestyle="--", linewidth=1.5,
                         label="budget boundary (0 = on budget)")

        ax_b.set_xlabel("round")
        ax_b.set_ylabel("GPU time − budget (s)")
        ax_b.set_title(f"{label}: GPU contention deviation per round", fontsize=10)
        ax_b.legend(fontsize=8, loc="upper left")
        ax_b.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[plot] Timing sanity plot → {out_path}")


# ── reporting ─────────────────────────────────────────────────────────────────

def _status_icon(s: str) -> str:
    return {"PASS": "[OK]", "WARN": "[!!]", "FAIL": "[XX]"}.get(s, "[??]")


def print_report(results: dict, strict: bool) -> bool:
    overall_ok = True
    print(f"\n{'='*70}")
    print("  PARITY REPORT: real vs simulated")
    print(f"{'='*70}\n")

    checks = [
        ("1. Selection parity (trainer sets per round)", "selection"),
        ("2. Statistical utility distributions",         "utility"),
        ("3. Aggregation sequence (per-round order)",    "aggregation"),
        ("4. Staleness distribution",                    "staleness"),
        ("5. Trainer participation counts",              "participation"),
        ("6. Convergence (accuracy / loss)",             "convergence"),
        ("7. Round duration parity (sim_round_duration_s)", "round_duration"),
        ("8. sim_send_ts presence / correctness",        "sim_send_ts"),
        ("9. GPU contention (actual GPU time vs budget)", "gpu_contention"),
    ]

    for title, key in checks:
        r = results.get(key, {})
        status = r.get("status", "UNKNOWN")
        icon = _status_icon(status)
        print(f"  {icon} [{status:4s}] {title}")

        if key == "selection":
            print(f"         rounds compared: {r['rounds_compared']}")
            print(f"         exact match:     {r['exact_match_pct']}%")
            print(f"         mean Jaccard:    {r['mean_jaccard']}  (1.0 = identical sets)")
            if r.get("worst_rounds"):
                print("         worst rounds:")
                for w in r["worst_rounds"][:3]:
                    print(f"           round {w['round']:3d}: J={w['jaccard']:.3f} "
                          f"real={w['real']} sim={w['sim']}")

        elif key == "utility":
            print(f"         max KS stat:         {r.get('max_ks_stat')}  (0=identical, <0.2=good)")
            print(f"         avg mean-util diff:  {r.get('avg_mean_utility_diff')}")
            print("         per trainer:")
            for row in r.get("per_trainer", []):
                print(f"           ...{row['trainer']}: real_mean={row['real_mean']} "
                      f"sim_mean={row['sim_mean']} diff={row['mean_diff']} "
                      f"KS={row['ks_stat']}  (n_real={row['real_n']} n_sim={row['sim_n']})")

        elif key == "aggregation":
            print(f"         rounds compared:       {r['rounds_compared']}")
            print(f"         exact-sequence match:  {r['exact_sequence_match_pct']}%")
            print(f"         set match:             {r['set_match_pct']}%  (same trainers, any order)")
            if r.get("mismatched_rounds_sample"):
                print("         sample mismatches:")
                for m in r["mismatched_rounds_sample"][:3]:
                    print(f"           round {m['round']:3d}: real={m['real_seq']} "
                          f"sim={m['sim_seq']}  set_match={m['set_match']}")

        elif key == "staleness":
            print(f"         real mean: {r.get('real_mean_staleness')}  "
                  f"sim mean: {r.get('sim_mean_staleness')}  "
                  f"diff: {r.get('mean_staleness_diff')}")
            print(f"         real dist: {r.get('real_distribution')}")
            print(f"         sim  dist: {r.get('sim_distribution')}")

        elif key == "participation":
            print(f"         total updates  real: {r.get('total_updates_real')}  "
                  f"sim: {r.get('total_updates_sim')}")
            print(f"         avg participation diff per trainer: {r.get('avg_participation_diff')}")
            print(f"         max participation diff: {r.get('max_participation_diff')}")
            print("         per trainer (sorted by diff):")
            for row in r.get("per_trainer", []):
                print(f"           ...{row['trainer']}: real={row['real']:3d}  sim={row['sim']:3d}  "
                      f"diff={row['abs_diff']}")

        elif key == "convergence":
            print(f"         eval rounds compared: {r.get('eval_rounds_compared')}")
            print(f"         avg accuracy diff:    {r.get('avg_accuracy_diff')}")
            print(f"         avg loss diff:        {r.get('avg_loss_diff')}")
            if r.get("curve"):
                print("         curve (sample):")
                for row in r["curve"][:5]:
                    print(f"           round {row['round']:3d}: "
                          f"acc real={row['real_acc']} sim={row['sim_acc']}  "
                          f"loss real={row['real_loss']} sim={row['sim_loss']}")

        elif key == "round_duration":
            note = r.get("note")
            if note:
                print(f"         note: {note}")
            else:
                print(f"         max KS stat:      {r.get('max_ks_stat')}  (<0.2=good)")
                print(f"         avg mean diff (s): {r.get('avg_mean_diff_s')}")
                for row in r.get("per_trainer", []):
                    print(f"           ...{row['trainer']}: real={row['real_mean_s']}s "
                          f"sim={row['sim_mean_s']}s diff={row['mean_diff_s']}s "
                          f"KS={row['ks_stat']}  (n_real={row['n_real']} n_sim={row['n_sim']})")

        elif key == "sim_send_ts":
            print(f"         sim trainers OK (non-zero vclock): {r.get('sim_trainers_ok')}")
            print(f"         sim trainers flat-zero (bug present): {r.get('sim_trainers_flat_zero')}")
            print(f"         sim trainers no task_recv data: {r.get('sim_trainers_no_data')}")
            for issue in r.get("issues", [])[:5]:
                print(f"           [!] {issue}")

        elif key == "gpu_contention":
            note = r.get("note")
            if note:
                print(f"         note: {note}")
            else:
                ra, sa = r.get("real_summary"), r.get("sim_summary")
                if ra:
                    print(f"         real: {ra['n_trainers']} trainers, "
                          f"mean_overrun_frac={ra['mean_overrun_frac']:.1%}  "
                          f"max_overrun_frac={ra['max_overrun_frac']:.1%}  "
                          f"trainers_with_overruns={ra['trainers_with_any_overrun']}")
                if sa:
                    print(f"         sim:  {sa['n_trainers']} trainers, "
                          f"mean_overrun_frac={sa['mean_overrun_frac']:.1%}  "
                          f"max_overrun_frac={sa['max_overrun_frac']:.1%}  "
                          f"trainers_with_overruns={sa['trainers_with_any_overrun']}")
                print("         per trainer (sorted by worst overrun fraction):")
                for row in r.get("per_trainer", []):
                    r_str = (f"real: gpu={row['real_mean_gpu_s']}s max={row['real_max_gpu_s']}s "
                             f"overrun={row['real_overrun_frac']:.1%}"
                             if row.get("real_mean_gpu_s") is not None else "real: n/a")
                    s_str = (f"sim: gpu={row['sim_mean_gpu_s']}s max={row['sim_max_gpu_s']}s "
                             f"overrun={row['sim_overrun_frac']:.1%}"
                             if row.get("sim_mean_gpu_s") is not None else "sim: n/a")
                    print(f"           ...{row['trainer']} D={row['budget_s']}s | {r_str} | {s_str}")

        if status == "FAIL" or (strict and status == "WARN"):
            overall_ok = False
        print()

    verdict = "ALL CHECKS PASSED" if overall_ok else "ONE OR MORE CHECKS FAILED"
    print(f"{'='*70}")
    print(f"  {_status_icon('PASS' if overall_ok else 'FAIL')} {verdict}")
    print(f"{'='*70}\n")
    return overall_ok


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare real vs simulated run parity")
    parser.add_argument("--real", required=True, help="Path to real aggregator JSONL")
    parser.add_argument("--sim",  required=True, help="Path to simulated aggregator JSONL")
    parser.add_argument("--real-trainer-dir", default=None,
                        help="Dir containing real trainer_*.jsonl files (for checks 7-8)")
    parser.add_argument("--sim-trainer-dir",  default=None,
                        help="Dir containing sim trainer_*.jsonl files (for checks 7-8)")
    parser.add_argument("--rounds", type=int, default=None,
                        help="Only compare up to this round number")
    parser.add_argument("--strict", action="store_true",
                        help="Treat WARN as FAIL")
    parser.add_argument("--json-out", default=None,
                        help="Write full results as JSON to this path")
    parser.add_argument("--plot-out", default=None,
                        help="Write timing sanity plot (GPU vs budget) to this PNG path")
    args = parser.parse_args()

    print(f"Loading real: {args.real}")
    real = load_agg_jsonl(args.real)
    print(f"Loading sim:  {args.sim}")
    sim  = load_agg_jsonl(args.sim)

    print(f"  real: {len(real['agg_rounds'])} agg_round events, "
          f"{len(real['selection_train'])} train-selection events, "
          f"{len(real['agg_evals'])} eval events")
    print(f"  sim:  {len(sim['agg_rounds'])} agg_round events, "
          f"{len(sim['selection_train'])} train-selection events, "
          f"{len(sim['agg_evals'])} eval events")

    real_trainers = load_trainer_jsonl_dir(args.real_trainer_dir)
    sim_trainers  = load_trainer_jsonl_dir(args.sim_trainer_dir)
    if real_trainers or sim_trainers:
        print(f"  real trainer files: {len(real_trainers)}, sim trainer files: {len(sim_trainers)}")

    results = {
        "selection":     check_selection_parity(real, sim, args.rounds),
        "utility":       check_utility_parity(real, sim),
        "aggregation":   check_aggregation_sequence(real, sim, args.rounds),
        "staleness":     check_staleness(real, sim),
        "participation": check_participation(real, sim),
        "convergence":   check_convergence(real, sim),
        "round_duration":  check_round_duration_parity(real_trainers, sim_trainers),
        "sim_send_ts":     check_sim_send_ts(real_trainers, sim_trainers),
        "gpu_contention":  check_gpu_contention(real_trainers, sim_trainers),
    }

    ok = print_report(results, args.strict)

    if args.plot_out:
        plot_timing_sanity(real_trainers, sim_trainers, args.plot_out)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Full results written to {args.json_out}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
