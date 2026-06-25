#!/usr/bin/env python3
"""H1 (residence leak) vs H2 (scorer-on-drifted-pool) discriminator for oort.

Reconstructs, per round, the eligible-vs-in-flight partition from stored
selection telemetry (chosen - (explore_ids|exploit_ids) = in_flight_at_entry),
then decomposes each speed bucket's net selection rate into two channels:

    selection_rate_per_round = eligible_fraction * P(selected_new | eligible)

H1 signature: the sim/real gap lives in eligible_fraction (slow trainers are
              eligible too often in sim because they are under-held in-flight).
H2 signature: eligible_fraction matches, but P(selected|eligible) is higher in
              sim (the scorer picks slower from the same menu).
"""
import json
import sys
import statistics
from collections import defaultdict


def load_selection_events(path):
    evs = []
    with open(path) as fh:
        for line in fh:
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("event") == "selection" and ev.get("task") == "train":
                evs.append(ev)
    evs.sort(key=lambda e: e.get("round", 0))
    return evs


def analyze(path, label):
    evs = load_selection_events(path)
    # 1) per-trainer intrinsic speed = median of measured speed_s across rounds
    speed_samples = defaultdict(list)
    for ev in evs:
        for tid, info in ev.get("per_trainer", {}).items():
            s = info.get("speed_s")
            if s is not None:
                speed_samples[tid].append(float(s))
    intrinsic = {t: statistics.median(v) for t, v in speed_samples.items() if v}

    # 2) per-trainer round-level counters
    n_rounds = 0
    elig_rounds = defaultdict(int)      # rounds the trainer was eligible
    inflight_rounds = defaultdict(int)  # rounds held in-flight at entry
    sel_new = defaultdict(int)          # times newly selected (explore|exploit)
    seen = set()

    for ev in evs:
        n_rounds += 1
        pt = ev.get("per_trainer", {})
        candidates = set(pt.keys())
        chosen = set(ev.get("chosen", []))
        newly = set(ev.get("explore_ids", []) or []) | set(ev.get("exploit_ids", []) or [])
        inflight_entry = chosen - newly
        eligible = candidates - inflight_entry
        for t in candidates:
            seen.add(t)
        for t in eligible:
            elig_rounds[t] += 1
        for t in inflight_entry:
            inflight_rounds[t] += 1
        for t in newly:
            sel_new[t] += 1

    # 3) bucket by intrinsic speed
    def bucket(t):
        s = intrinsic.get(t)
        if s is None:
            return None
        if s < 5.0:
            return "fast(<5s)"
        if s < 10.0:
            return "mid(5-10s)"
        return "slow(>=10s)"

    buckets = ["fast(<5s)", "mid(5-10s)", "slow(>=10s)"]
    agg = {b: defaultdict(float) for b in buckets}
    counts = {b: 0 for b in buckets}
    for t in seen:
        b = bucket(t)
        if b is None:
            continue
        counts[b] += 1
        agg[b]["elig_rounds"] += elig_rounds[t]
        agg[b]["inflight_rounds"] += inflight_rounds[t]
        agg[b]["sel_new"] += sel_new[t]

    print(f"\n===== {label}   (rounds={n_rounds}, trainers_seen={len(seen)}) =====")
    hdr = f"{'bucket':12s} {'n':>4s} {'elig_frac':>9s} {'inflt_frac':>10s} {'sel/round':>9s} {'P(sel|elig)':>11s}"
    print(hdr)
    out = {}
    for b in buckets:
        n = counts[b]
        if n == 0:
            continue
        # per-trainer-per-round averages
        elig_frac = agg[b]["elig_rounds"] / (n * n_rounds)
        inflt_frac = agg[b]["inflight_rounds"] / (n * n_rounds)
        sel_per_round = agg[b]["sel_new"] / (n * n_rounds)
        p_sel_given_elig = (agg[b]["sel_new"] / agg[b]["elig_rounds"]) if agg[b]["elig_rounds"] else 0.0
        out[b] = dict(n=n, elig_frac=elig_frac, inflt_frac=inflt_frac,
                      sel_per_round=sel_per_round, p_sel_given_elig=p_sel_given_elig)
        print(f"{b:12s} {n:4d} {elig_frac:9.4f} {inflt_frac:10.4f} "
              f"{sel_per_round:9.5f} {p_sel_given_elig:11.5f}")
    return out, n_rounds


def main():
    sim_path, real_path = sys.argv[1], sys.argv[2]
    sim, sim_r = analyze(sim_path, "SIM")
    real, real_r = analyze(real_path, "REAL")

    print("\n===== DECOMPOSITION: where does the slow-trainer gap live? =====")
    print("selection_rate = eligible_fraction * P(selected|eligible)\n")
    print(f"{'bucket':12s} {'metric':18s} {'SIM':>10s} {'REAL':>10s} {'sim/real':>9s}")
    for b in ["fast(<5s)", "mid(5-10s)", "slow(>=10s)"]:
        if b not in sim or b not in real:
            continue
        for m in ("elig_frac", "p_sel_given_elig", "sel_per_round", "inflt_frac"):
            sv, rv = sim[b][m], real[b][m]
            ratio = (sv / rv) if rv else float("nan")
            print(f"{b:12s} {m:18s} {sv:10.5f} {rv:10.5f} {ratio:9.2f}")
        print()

    # fast/slow selection-rate ratio cross-check (doc cited real 6.3x vs sim 2.8x)
    def fs_ratio(d):
        f = d.get("fast(<5s)", {}).get("sel_per_round", 0)
        s = d.get("slow(>=10s)", {}).get("sel_per_round", 0)
        return (f / s) if s else float("nan")
    print(f"fast/slow sel-rate ratio:  SIM={fs_ratio(sim):.2f}  REAL={fs_ratio(real):.2f}  "
          f"(doc cited sim~2.8 real~6.3)")


if __name__ == "__main__":
    main()
