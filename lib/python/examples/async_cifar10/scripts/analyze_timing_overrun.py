"""Visualize trainer-iteration timing overruns and attribute the excess to the
trainer vs. the aggregator/transport.

For every committed update we reconstruct, per (trainer, model_version):

    budget D            trainer's modeled round time (training_budget_s)
    trainer_busy        pre_train_s + real_gpu_time_s + post_train_s
                        (the trainer's own active wall time, excludes the
                         modeled real-mode sleep)
    wall_lag            aggregator send -> recv (SEND_RECV_LAG in the agg log)
    transport           wall_lag - trainer_busy - modeled_sleep
                        (down-leg + up-leg + aggregator receive-queue)

Two deviations from the "stipulated" time D:
    trainer_dev = trainer_busy - D     excess the *trainer* itself added
    agg_dev     = wall_lag    - D      excess the *aggregator* observed

If agg_dev tracks trainer_dev, the excess is the trainer's own per-round work;
if agg_dev >> trainer_dev, the gap is transport / aggregator-side queueing.

Outputs (per the request):
  * a histogram of how many responses came back late, bucketed by lateness,
    drawn for both the trainer-side and the aggregator-side deviation;
  * a CDF of the *extent* of the deviation (seconds over budget), trainer-side
    vs aggregator-side vs transport.

Multiple runs can be overlaid (e.g. with-fix vs without-fix, real vs simulated):

    python analyze_timing_overrun.py \
        --run "real_nofix=/path/run_a" --run "real_fix=/path/run_b" \
        --out /tmp/timing_overrun.png

A single positional run_dir is also accepted.
"""

import argparse
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os as _os, sys as _sys
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _SCRIPT_DIR not in _sys.path:
    _sys.path.insert(0, _SCRIPT_DIR)
from plotters._annot import annotate_percentiles, flush_percentile_table

# Lateness buckets (seconds over budget) for the count histogram.
BUCKETS = [(-1e9, 0), (0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 60), (60, 1e9)]
BUCKET_LABELS = ["on time", "0-1", "1-2", "2-5", "5-10", "10-20", "20-60", "60+"]

_RE_SEND = re.compile(
    r"^(\S+ \S+) .*sending weights to (\w+) model_version=(\d+) task=train"
)
_RE_LAG = re.compile(
    r"^(\S+ \S+) .*\[SEND_RECV_LAG\] end=(\w+) version=(\d+) wall_lag_s=([0-9.]+)"
)
_RE_OVERRUN_AGG = re.compile(r"\[TIMING_OVERRUN_AGG\]")
_RE_LAG_HIGH = re.compile(r"\[SEND_RECV_LAG_HIGH\]")


def _log_ts(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S,%f").timestamp()


def _find_agg_log(run_dir):
    cands = glob.glob(os.path.join(run_dir, "*aggregator.log"))
    return cands[0] if cands else None


def parse_agg_log(log_path):
    """sent[end][ver]=epoch, wall_lag[end][ver]=s, plus the two warning counts."""
    sent = defaultdict(dict)
    wall_lag = defaultdict(dict)
    n_overrun_agg = n_lag_high = 0
    if not log_path or not os.path.exists(log_path):
        return sent, wall_lag, n_overrun_agg, n_lag_high
    with open(log_path, errors="replace") as f:
        for line in f:
            m = _RE_SEND.search(line)
            if m:
                sent[m.group(2)][int(m.group(3))] = _log_ts(m.group(1))
                continue
            m = _RE_LAG.search(line)
            if m:
                wall_lag[m.group(2)][int(m.group(3))] = float(m.group(4))
                continue
            if _RE_OVERRUN_AGG.search(line):
                n_overrun_agg += 1
            elif _RE_LAG_HIGH.search(line):
                n_lag_high += 1
    return sent, wall_lag, n_overrun_agg, n_lag_high


def parse_trainer_telemetry(run_dir):
    """rounds[end][ver] = {D, gpu, pre, post, rem, busy, time_mode}."""
    rounds = defaultdict(dict)
    tdir = os.path.join(run_dir, "telemetry")
    for fp in glob.glob(os.path.join(tdir, "trainer_*.jsonl")):
        with open(fp, errors="replace") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if e.get("event") != "trainer_round":
                    continue
                end = e.get("trainer_id") or e.get("end_id")
                ver = e.get("round")
                gpu = float(e.get("real_gpu_time_s", 0.0) or 0.0)
                pre = float(e.get("pre_train_s", 0.0) or 0.0)
                post = float(e.get("post_train_s", 0.0) or 0.0)
                rounds[end][ver] = {
                    "D": float(e.get("training_budget_s", 0.0) or 0.0),
                    "gpu": gpu,
                    "pre": pre,
                    "post": post,
                    "rem": float(e.get("remaining_time_s", 0.0) or 0.0),
                    "busy": pre + gpu + post,
                    "time_mode": e.get("time_mode", "real"),
                }
    return rounds


def build_records(run_dir):
    """One record per matched (trainer, version) update."""
    sent, wall_lag, n_overrun_agg, n_lag_high = parse_agg_log(_find_agg_log(run_dir))
    rounds = parse_trainer_telemetry(run_dir)
    recs = []
    for end, vers in rounds.items():
        for ver, r in vers.items():
            D = r["D"]
            if D <= 0:
                continue
            busy = r["busy"]
            rec = {
                "trainer_dev": busy - D,
                "trainer_busy": busy,
                "D": D,
                "gpu": r["gpu"],
                "pre": r["pre"],
                "post": r["post"],
                "time_mode": r["time_mode"],
            }
            wl = wall_lag.get(end, {}).get(ver)
            if wl is not None:
                rec["wall_lag"] = wl
                rec["agg_dev"] = wl - D
                rec["transport"] = max(0.0, wl - busy - r["rem"])
            recs.append(rec)
    meta = {
        "n_overrun_agg": n_overrun_agg,
        "n_lag_high": n_lag_high,
        "n_records": len(recs),
        "n_with_agg": sum(1 for r in recs if "agg_dev" in r),
    }
    return recs, meta


def _bucket_counts(devs):
    counts = [0] * len(BUCKETS)
    for d in devs:
        for i, (lo, hi) in enumerate(BUCKETS):
            if lo <= d < hi:
                counts[i] += 1
                break
    return counts


def _cdf_xy(vals):
    vals = sorted(vals)
    n = len(vals)
    if n == 0:
        return [], []
    ys = [(i + 1) / n for i in range(n)]
    return vals, ys


def _summary(label, recs, meta):
    tdev = [r["trainer_dev"] for r in recs]
    adev = [r["agg_dev"] for r in recs if "agg_dev" in r]
    trans = [r["transport"] for r in recs if "transport" in r]
    n = len(tdev)
    t_late = sum(1 for d in tdev if d > 0)
    a_late = sum(1 for d in adev if d > 0)

    def pct(a, q):
        if not a:
            return float("nan")
        s = sorted(a)
        return s[min(len(s) - 1, int(len(s) * q))]

    print(f"\n=== {label} ===")
    mode = recs[0]["time_mode"] if recs else "?"
    print(f"  time_mode={mode}  matched_updates={n}  (with agg wall-lag: {meta['n_with_agg']})")
    print(f"  agg-log warnings: TIMING_OVERRUN_AGG={meta['n_overrun_agg']}  SEND_RECV_LAG_HIGH={meta['n_lag_high']}")
    if n:
        print(f"  trainer-side late: {t_late}/{n} ({100*t_late/n:.0f}%)  "
              f"dev med={st.median(tdev):.2f}s p90={pct(tdev,0.9):.2f}s p99={pct(tdev,0.99):.2f}s max={max(tdev):.2f}s")
    if adev:
        print(f"  agg-observed late: {a_late}/{len(adev)} ({100*a_late/len(adev):.0f}%)  "
              f"dev med={st.median(adev):.2f}s p90={pct(adev,0.9):.2f}s p99={pct(adev,0.99):.2f}s max={max(adev):.2f}s")
    if trans:
        print(f"  transport (down+up+aggq): med={st.median(trans):.2f}s p90={pct(trans,0.9):.2f}s max={max(trans):.2f}s")
    if recs:
        gpu = [r["gpu"] for r in recs]; pre = [r["pre"] for r in recs]; post = [r["post"] for r in recs]
        print(f"  trainer phases (med): pre={st.median(pre):.2f}s gpu={st.median(gpu):.2f}s post={st.median(post):.2f}s")


def plot(runs, out_path):
    """runs: list of (label, recs, meta). Two panels: count histogram + CDF."""
    fig, (axh, axc) = plt.subplots(1, 2, figsize=(15, 6))

    x = range(len(BUCKET_LABELS))
    nruns = len(runs)
    # group width so multiple runs sit side by side
    span = 0.8
    for ri, (label, recs, _meta) in enumerate(runs):
        tdev = [r["trainer_dev"] for r in recs]
        adev = [r["agg_dev"] for r in recs if "agg_dev" in r]
        tc = _bucket_counts(tdev)
        ac = _bucket_counts(adev)
        w = span / (2 * max(1, nruns))
        off = -span / 2 + ri * 2 * w
        axh.bar([i + off for i in x], tc, width=w, label=f"{label} · trainer",
                alpha=0.85)
        axh.bar([i + off + w for i in x], ac, width=w, label=f"{label} · agg",
                alpha=0.55, hatch="//")
    axh.set_xticks(list(x))
    axh.set_xticklabels(BUCKET_LABELS, rotation=30)
    axh.set_xlabel("lateness over budget D (s)")
    axh.set_ylabel("number of responses")
    axh.set_title("Responses by lateness bucket\n(trainer-attributable vs aggregator-observed)")
    axh.legend(fontsize=8)
    axh.grid(axis="y", alpha=0.3)

    for label, recs, _meta in runs:
        tdev = [max(0.0, r["trainer_dev"]) for r in recs]
        adev = [max(0.0, r["agg_dev"]) for r in recs if "agg_dev" in r]
        trans = [r["transport"] for r in recs if "transport" in r]
        _colors = {"trainer": "#4e79a7", "agg": "#e15759", "transport": "#59a14f"}
        for series, name, style in [
            (tdev, "trainer", "-"),
            (adev, "agg", "--"),
            (trans, "transport", ":"),
        ]:
            xs, ys = _cdf_xy(series)
            if xs:
                c = _colors.get(name, "gray")
                axc.plot(xs, ys, style, color=c, label=f"{label} · {name}")
                annotate_percentiles(axc, series, color=c,
                                     label=f"{label}·{name}", below=True)
    axc.set_xlabel("deviation from expected time D (s over budget)")
    axc.set_ylabel("cumulative fraction of responses")
    axc.set_title("CDF of timing-deviation extent\n(P50/P90/P99 in table below)")
    axc.set_ylim(0, 1.02)
    axc.grid(alpha=0.3)
    axc.legend(fontsize=8)
    flush_percentile_table(axc)
    # log-x helps when a few outliers dominate; guard against all-zero
    try:
        axc.set_xscale("symlog", linthresh=1.0)
    except Exception:
        pass

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"\nWrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", help="single run directory")
    ap.add_argument("--run", action="append", default=[],
                    help="labeled run as label=/path/to/run_dir (repeatable, for overlay)")
    ap.add_argument("--out", default=None, help="output PNG path")
    args = ap.parse_args()

    specs = []
    if args.run_dir:
        specs.append((os.path.basename(args.run_dir.rstrip("/")), args.run_dir))
    for item in args.run:
        if "=" not in item:
            ap.error(f"--run expects label=path, got {item!r}")
        label, path = item.split("=", 1)
        specs.append((label, path))
    if not specs:
        ap.error("provide a run_dir or at least one --run label=path")

    runs = []
    for label, path in specs:
        recs, meta = build_records(path)
        _summary(label, recs, meta)
        runs.append((label, recs, meta))

    out = args.out or os.path.join(specs[0][1], "plots", "timing_overrun.png")
    plot(runs, out)


if __name__ == "__main__":
    main()
