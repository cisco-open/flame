"""
Parity comparator: real run vs simulated run.

Checks:
  1. Selection parity     — per round: same trainers selected? same order?
  2. Statistical utility  — per-trainer utility distributions match?
  3. Aggregation sequence — same contributing-trainer order per round?
  4. Staleness            — distributions match?
  5. Participation counts — each trainer used similar # of times?
  6. Accuracy / loss      — convergence curves from agg_eval events

Usage:
    python compare_parity.py \\
        --real  experiments/run_20260529_101200.../telemetry/aggregator_*.jsonl \\
        --sim   experiments/run_20260529_152224.../telemetry/aggregator_*.jsonl \\
        [--rounds 100] [--strict]

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
    parser.add_argument("--rounds", type=int, default=None,
                        help="Only compare up to this round number")
    parser.add_argument("--strict", action="store_true",
                        help="Treat WARN as FAIL")
    parser.add_argument("--json-out", default=None,
                        help="Write full results as JSON to this path")
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

    results = {
        "selection":     check_selection_parity(real, sim, args.rounds),
        "utility":       check_utility_parity(real, sim),
        "aggregation":   check_aggregation_sequence(real, sim, args.rounds),
        "staleness":     check_staleness(real, sim),
        "participation": check_participation(real, sim),
        "convergence":   check_convergence(real, sim),
    }

    ok = print_report(results, args.strict)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Full results written to {args.json_out}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
