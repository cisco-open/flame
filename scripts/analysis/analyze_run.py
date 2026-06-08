#!/usr/bin/env python
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Post-run telemetry analyzer for FLAME experiments.

Reads the JSONL event streams written by :mod:`flame.telemetry` (one file per
process in a run's telemetry directory) and emits PDF plots grouped by class:

  plots/performance/  did it learn? (accuracy, loss, convergence)
  plots/sanity/       is the run healthy & as configured? (expected-vs-actual
                      runtime/utility, overruns, rejections, data unlock)
  plots/selection/    who was picked? (coverage, selected-vs-pool, participation)
  plots/insights/     why? (staleness audit, mis-selection, data-unlock effects)
  plots/system/       resources & communication

Every figure is config-stamped (selector/alpha/n/availability/streaming/...).

Usage
-----
    python analyze_run.py <telemetry_dir> [--out <plots_dir>]
    python analyze_run.py --compare-streaming <d1> <d2> ... [--labels ...]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_helpers as ph  # noqa: E402

try:
    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib", "python"
        ),
    )
    from flame.telemetry.events import (  # noqa: E402
        EVENT_AGG_EVAL,
        EVENT_AGG_ROUND,
        EVENT_AVAIL_CHANGE,
        EVENT_SELECTION,
        EVENT_TRAINER_ROUND,
        EVENT_UTIL_DISPARITY,
    )
except Exception:  # pragma: no cover
    EVENT_SELECTION = "selection"
    EVENT_AGG_EVAL = "agg_eval"
    EVENT_AGG_ROUND = "agg_round"
    EVENT_TRAINER_ROUND = "trainer_round"
    EVENT_UTIL_DISPARITY = "util_disparity"
    EVENT_AVAIL_CHANGE = "avail_change"


# Communication is reported in MEGABYTES. One model = MODEL_PARAM_COUNT fp32
# params; a train round moves it down (agg->trainer) AND up (trainer->agg) = 2x,
# an eval round moves it down only = 1x. MODEL_PARAM_COUNT is the async_cifar10
# `Net` (3 conv + 3 fc; conv1 3->64, conv2 64->128, conv3 128->256, fc 1024->128->
# 256->10) = 537,610 params -> ~2.15 MB/model. Override via --model-params if the
# Net changes; the value only scales the comm axes linearly.
MODEL_PARAM_COUNT = 537610
BYTES_PER_PARAM = 4
MODEL_MB = MODEL_PARAM_COUNT * BYTES_PER_PARAM / 1e6


def load_events(telemetry_dir: str) -> list[dict]:
    """Load all JSONL records from a telemetry directory."""
    records: list[dict] = []
    for path in sorted(glob.glob(os.path.join(telemetry_dir, "*.jsonl"))):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def by_event(records, event):
    return [r for r in records if r.get("event") == event]


def _sub(out_root, cls):
    return os.path.join(out_root, cls)


# ==========================================================================
# data extractors
# ==========================================================================


def _visible_fraction(r):
    vs, ts = r.get("visible_samples"), r.get("total_samples")
    if vs is None or not ts:
        return None
    return vs / ts


def accuracy_by_round(records):
    out = {}
    for r in by_event(records, EVENT_AGG_EVAL):
        a = r.get("test-accuracy")
        if a is not None:
            out[int(r.get("round", 0))] = a
    return out


def loss_by_round(records):
    out = {}
    for r in by_event(records, EVENT_AGG_EVAL):
        a = r.get("test-loss")
        if a is not None:
            out[int(r.get("round", 0))] = a
    return out


def sim_time_by_round(records):
    """round -> cumulative max sim_completion_ts (sim wall time)."""
    best = {}
    for r in by_event(records, EVENT_TRAINER_ROUND):
        rd = int(r.get("round", 0))
        sc = r.get("sim_completion_ts")
        if sc is None:
            sc = r.get("sim_round_duration_s")
        if sc is None:
            continue
        best[rd] = max(best.get(rd, 0.0), float(sc))
    out, run = {}, 0.0
    for rd in sorted(best):
        run = max(run, best[rd])
        out[rd] = run
    return out


def cumulative_comm_by_round(records):
    """Cumulative comm in MEGABYTES (train=2x model, eval=1x model, per chosen)."""
    per_round = defaultdict(float)
    for r in by_event(records, EVENT_SELECTION):
        rd = int(r.get("round", 0))
        n = len(r.get("chosen") or [])
        eq = 2.0 * n if r.get("task", "train") == "train" else 1.0 * n
        per_round[rd] += eq * MODEL_MB
    rounds = sorted(per_round)
    cum, run = [], 0.0
    for rd in rounds:
        run += per_round[rd]
        cum.append(run)
    return rounds, cum


def comm_breakdown_by_round(records):
    """Per-round communication split by task AND direction, in MB, plus message
    and discard counts — the apples-to-apples comm picture (#17).

    down (agg->trainer) = the full model to every CHOSEN end (selection events).
    up   (trainer->agg) for TRAIN = the full model back from each COMMITTED end
        (agg_round.contributing_trainers); for EVAL only a tiny stat-utility
        scalar comes back (~0 MB, so eval up is omitted from MB but counted as a
        message). discarded(train) = chosen - committed = sent-but-unused this
        round (overcommitment slack / stragglers).
    Returns {round: {train_down_mb, train_up_mb, eval_down_mb,
                     msgs_down, msgs_up, discarded}}."""
    chosen = defaultdict(lambda: {"train": 0, "eval": 0})
    for s in by_event(records, EVENT_SELECTION):
        chosen[int(s.get("round", 0))][s.get("task", "train")] += len(s.get("chosen") or [])
    committed = defaultdict(int)  # train updates returned (per round)
    for a in by_event(records, EVENT_AGG_ROUND):
        committed[int(a.get("round", 0))] += len(a.get("contributing_trainers") or [])
    out = {}
    for rd in sorted(set(chosen) | set(committed)):
        ctr = chosen[rd]["train"]; cev = chosen[rd]["eval"]; up = committed[rd]
        out[rd] = {
            "train_down_mb": ctr * MODEL_MB,
            "train_up_mb": up * MODEL_MB,
            "eval_down_mb": cev * MODEL_MB,
            "msgs_down": ctr + cev,
            "msgs_up": up + cev,  # eval returns a (tiny) utility message too
            "discarded": max(0, ctr - up),
        }
    return out


def comm_vs_accuracy_series(records):
    rounds, cum = cumulative_comm_by_round(records)
    cum_at = dict(zip(rounds, cum))
    acc = accuracy_by_round(records)
    xs, ys, running = [], [], 0.0
    for rd in sorted(acc):
        ups = [c for r, c in cum_at.items() if r <= rd]
        running = max(ups) if ups else running
        xs.append(running)
        ys.append(acc[rd])
    return xs, ys


def time_to_target(records, target):
    evs = sorted(by_event(records, EVENT_AGG_EVAL), key=lambda r: r.get("round", 0))
    start_ts = min((r.get("ts") for r in records if r.get("ts")), default=None)
    sim_map = sim_time_by_round(records)
    for r in evs:
        a = r.get("test-accuracy")
        if a is not None and a >= target:
            rd = int(r.get("round", 0))
            wall = (r.get("ts") - start_ts) if (r.get("ts") and start_ts) else None
            return {"round": rd, "wall_s": wall, "sim_s": sim_map.get(rd),
                    "accuracy": a, "reached": True}
    return {"round": None, "wall_s": None, "sim_s": None,
            "accuracy": None, "reached": False}


def load_oracle_misselection(telemetry_dir):
    p = os.path.join(os.path.dirname(os.path.abspath(telemetry_dir)),
                     "analysis", "oracle_misselection.csv")
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def load_oracle_utility(telemetry_dir):
    p = os.path.join(os.path.dirname(os.path.abspath(telemetry_dir)),
                     "analysis", "oracle_utility.csv")
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def load_oracle_counterfactual(telemetry_dir):
    p = os.path.join(os.path.dirname(os.path.abspath(telemetry_dir)),
                     "analysis", "oracle_counterfactual.csv")
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


_LAG_RE = re.compile(r"\[SEND_RECV_LAG\] end=(\S+) version=(\d+) wall_lag_s=([0-9.]+)")
_OVERRUN_RE = re.compile(r"\[TIMING_OVERRUN_AGG\]")


def parse_send_recv_lags(telemetry_dir: str) -> tuple[list[float], int]:
    """Parse SEND_RECV_LAG and TIMING_OVERRUN_AGG lines from the aggregator log.

    Returns (wall_lags, overrun_count). Returns ([], 0) if no log is found or
    no matching lines exist — callers should handle the empty case gracefully.
    """
    run_dir = os.path.dirname(os.path.abspath(telemetry_dir))
    logs = glob.glob(os.path.join(run_dir, "*aggregator*.log"))
    if not logs:
        return [], 0
    lags: list[float] = []
    overruns = 0
    with open(logs[0]) as fh:
        for line in fh:
            m = _LAG_RE.search(line)
            if m:
                lags.append(float(m.group(3)))
            if _OVERRUN_RE.search(line):
                overruns += 1
    return lags, overruns


_OVERRUN_EXCESS_RE = re.compile(
    r"\[TIMING_OVERRUN_AGG\].*excess=([0-9.]+)s"
)

# [LAG_DECOMP] replaces the old [MQTT_DELIVERY_LAG] with a 6-component breakdown.
# Any field can be '-' when the required trainer-side timestamp was absent.
_LAG_DECOMP_RE = re.compile(
    r"\[LAG_DECOMP\] end=(\S+) version=(\d+) "
    r"wall_lag_s=([0-9.-]+) "
    r"agg_to_trainer_s=([0-9.-]+|-) "
    r"compute_s=([0-9.-]+|-) "
    r"post_wait_s=([0-9.-]+|-) "
    r"mqtt_lag_s=([0-9.-]+|-) "
    r"queue_wait_s=([0-9.-]+|-) "
    r"process_s=([0-9.-]+|-)"
)


def _parse_lag_decomp(telemetry_dir: str) -> dict[str, list[float]]:
    """Parse [LAG_DECOMP] lines from the aggregator log.

    Returns a dict keyed by component name; each value is a list of floats
    (one per update where that component was available). '-' entries are dropped.
    Keys: wall_lag_s, agg_to_trainer_s, compute_s, post_wait_s,
          mqtt_lag_s, queue_wait_s, process_s
    """
    run_dir = os.path.dirname(os.path.abspath(telemetry_dir))
    logs = glob.glob(os.path.join(run_dir, "*aggregator*.log"))
    result: dict[str, list[float]] = {
        k: [] for k in ("wall_lag_s", "agg_to_trainer_s", "compute_s",
                         "post_wait_s", "mqtt_lag_s", "queue_wait_s", "process_s")
    }
    if not logs:
        return result
    keys = ("wall_lag_s", "agg_to_trainer_s", "compute_s",
            "post_wait_s", "mqtt_lag_s", "queue_wait_s", "process_s")
    with open(logs[0]) as fh:
        for line in fh:
            m = _LAG_DECOMP_RE.search(line)
            if not m:
                continue
            for i, k in enumerate(keys):
                v = m.group(i + 3)
                if v != "-":
                    result[k].append(float(v))
    return result


def _parse_overrun_excesses(telemetry_dir: str) -> list[float]:
    """Parse overrun excess (wall_lag - budget) from [TIMING_OVERRUN_AGG] lines."""
    run_dir = os.path.dirname(os.path.abspath(telemetry_dir))
    logs = glob.glob(os.path.join(run_dir, "*aggregator*.log"))
    if not logs:
        return []
    excesses: list[float] = []
    with open(logs[0]) as fh:
        for line in fh:
            m = _OVERRUN_EXCESS_RE.search(line)
            if m:
                excesses.append(float(m.group(1)))
    return excesses


def _spearman(a, b):
    import numpy as np
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a.argsort().argsort(), b.argsort().argsort())[0, 1])


def im_staleness_by_round(telemetry_dir):
    rows = [r for r in load_oracle_utility(telemetry_dir) if r.get("task") == "train"]
    by_round = defaultdict(lambda: {"bel": [], "tru": []})
    for r in rows:
        try:
            b = float(r.get("believed")); t = float(r.get("true"))
        except (TypeError, ValueError):
            continue
        rd = int(float(r["round"]))
        by_round[rd]["bel"].append(b); by_round[rd]["tru"].append(t)
    rounds, corr, ngap = [], [], []
    for rd in sorted(by_round):
        bel, tru = by_round[rd]["bel"], by_round[rd]["tru"]
        if len(bel) < 3:
            continue
        mt = sum(tru) / len(tru)
        rounds.append(rd)
        corr.append(_spearman(bel, tru))
        ngap.append((sum(abs(b - t) for b, t in zip(bel, tru)) / len(bel) / mt)
                    if mt else None)
    return rounds, corr, ngap


def _floats(rows, key):
    out = []
    for r in rows:
        try:
            out.append(float(r.get(key)))
        except (TypeError, ValueError):
            out.append(None)
    return out


def trainer_rounds_by_round(records):
    out = defaultdict(list)
    for r in by_event(records, EVENT_TRAINER_ROUND):
        out[int(r.get("round", 0))].append(r)
    return out


# ==========================================================================
# performance/
# ==========================================================================


def perf_plots(records, out, stamp, tdir):
    d = _sub(out, "performance"); saved = []
    acc = accuracy_by_round(records)
    loss = loss_by_round(records)
    sim_map = sim_time_by_round(records)
    if acc:
        rs = sorted(acc)
        p = ph.line_plot({"test accuracy": (rs, [acc[r] for r in rs])},
                         "round", "test accuracy", "Test accuracy over rounds",
                         d, "accuracy_over_rounds.pdf", stamp=stamp, target=0.6)
        if p: saved.append(p)
        # vs sim-time (fair axis for async vs sync)
        xs = [sim_map.get(r) for r in rs]
        if any(x is not None for x in xs):
            pairs = [(x, acc[r]) for x, r in zip(xs, rs) if x is not None]
            p = ph.line_plot({"test accuracy": ([a for a, _ in pairs], [b for _, b in pairs])},
                             "sim-time (s)", "test accuracy",
                             "Test accuracy over sim-time", d,
                             "accuracy_over_simtime.pdf", stamp=stamp, target=0.6)
            if p: saved.append(p)
        # accuracy gain per eval (bars) + overall accuracy (secondary axis) so
        # both the per-eval delta and the absolute trajectory are visible.
        gains = [acc[rs[i]] - acc[rs[i - 1]] for i in range(1, len(rs))]
        if gains:
            p = ph.signed_bar_line(rs[1:], gains, [acc[r] for r in rs[1:]],
                                   "round", "Δ accuracy / eval", "test accuracy",
                                   "Accuracy gain per eval (bars) + overall accuracy (line)",
                                   d, "accuracy_gain_per_eval.pdf", stamp=stamp)
            if p: saved.append(p)
    if loss:
        rs = sorted(loss)
        p = ph.line_plot({"test loss": (rs, [loss[r] for r in rs])},
                         "round", "test loss", "Test loss over rounds",
                         d, "loss_over_rounds.pdf", stamp=stamp)
        if p: saved.append(p)
    # accuracy vs cumulative data unlocked
    trbr = trainer_rounds_by_round(records)
    unlocked = {}
    for rd, lst in trbr.items():
        fr = [(_visible_fraction(r)) for r in lst]
        fr = [x for x in fr if x is not None]
        if fr:
            unlocked[rd] = sum(fr) / len(fr)
    if acc and unlocked:
        rs = [r for r in sorted(acc) if r in unlocked]
        if rs:
            p = ph.line_plot({"accuracy": ([unlocked[r] for r in rs], [acc[r] for r in rs])},
                             "mean visible fraction (data unlocked)", "test accuracy",
                             "Accuracy vs amount of data unlocked", d,
                             "accuracy_vs_data_unlocked.pdf", stamp=stamp)
            if p: saved.append(p)
    # global weight-change norm from checkpoints
    saved += _weight_change_norm(tdir, d, stamp)
    return saved


def _weight_change_norm(tdir, d, stamp):
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(tdir)), "checkpoints")
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "round_*.pt")))
    if len(paths) < 2:
        return []
    try:
        import torch
    except Exception:
        return []
    rounds, norms, prev = [], [], None
    for p in paths:
        try:
            blob = torch.load(p, map_location="cpu")
        except Exception:
            continue
        sd = blob.get("state_dict", {})
        if prev is not None:
            sq = 0.0
            for k, v in sd.items():
                if k in prev and torch.is_floating_point(v):
                    diff = (v.float() - prev[k].float())
                    sq += float(torch.sum(diff * diff).item())
            rounds.append(int(blob.get("round", 0)))
            norms.append(sq ** 0.5)
        prev = {k: v.float() for k, v in sd.items() if torch.is_floating_point(v)}
    if not rounds:
        return []
    p = ph.line_plot({"||w_r - w_(r-1)||": (rounds, norms)}, "round",
                     "global weight-change L2", "Global model change per checkpoint",
                     d, "global_weight_change_norm.pdf", stamp=stamp, logy=True)
    return [p] if p else []


# ==========================================================================
# sanity/
# ==========================================================================


def sanity_plots(records, out, stamp, tdir):
    d = _sub(out, "sanity"); saved = []
    tr = by_event(records, EVENT_TRAINER_ROUND)

    # expected vs actual runtime
    exp, act, grp = [], [], []
    for r in tr:
        e = r.get("training_budget_s"); a = r.get("real_gpu_time_s")
        if e is None or a is None:
            continue
        exp.append(e); act.append(a)
        grp.append("late" if r.get("overran") else ("early" if a < e else "ontime"))
    if exp:
        p = ph.scatter_diag(exp, act, "expected runtime (budget s)",
                            "actual GPU time (s)",
                            "Trainer runtime: expected vs actual", d,
                            "trainer_runtime_expected_vs_actual.pdf", stamp=stamp,
                            groups=grp, group_labels=[("early", "early", "#2ca02c"),
                                                      ("ontime", "on-time", "#1f77b4"),
                                                      ("late", "late/overran", "#d62728")])
        if p: saved.append(p)
        resid = [a - e for e, a in zip(exp, act)]
        p = ph.hist_plot(resid, "actual - expected runtime (s)",
                         "Trainer runtime residual (>0 = slower than budget)", d,
                         "trainer_runtime_residual_hist.pdf", stamp=stamp, vline=0.0)
        if p: saved.append(p)
        # Response-lateness CDF. NOTE the residual above compares *GPU* time to
        # budget, so "early" just means the GPU finished early — the trainer
        # still SLEEPS to fill the budget. The true response time is
        # max(gpu, D), so lateness = response - budget is 0 (on-time) or >0
        # (overrun); it is never negative. This CDF shows the lateness extent.
        resp_dev = [r.get("sim_round_duration_s") - r.get("training_budget_s")
                    for r in tr if r.get("sim_round_duration_s") is not None
                    and r.get("training_budget_s") is not None]
        if resp_dev:
            p = ph.cdf_plot(resp_dev, "lateness = response - budget (s); 0 = on-time",
                            "Trainer response-lateness CDF (sleeps fill budget; never early)",
                            d, "trainer_response_lateness_cdf.pdf", stamp=stamp)
            if p: saved.append(p)

    # per-trainer late-FRACTION histogram (replaces a per-trainer bar that was
    # unreadable at 300 trainers). One value per trainer = its overran-rounds /
    # total-rounds; the hist annotates aggregate P50/P90/P99 across trainers.
    counts = defaultdict(lambda: {"late": 0, "total": 0})
    for r in tr:
        e = r.get("training_budget_s"); a = r.get("real_gpu_time_s")
        if e is None or a is None:
            continue
        tid = str(r.get("end_id", "?"))[-3:]
        counts[tid]["total"] += 1
        if r.get("overran"):
            counts[tid]["late"] += 1
    if counts:
        late_frac = [counts[t]["late"] / counts[t]["total"]
                     for t in counts if counts[t]["total"]]
        p = ph.hist_plot(late_frac, "per-trainer late fraction (overran rounds / total)",
                         "Per-trainer overrun fraction (aggregate P50/P90/P99)", d,
                         "trainer_late_fraction_hist.pdf", stamp=stamp, vline=None)
        if p: saved.append(p)

    # overrun rate over rounds
    trbr = trainer_rounds_by_round(records)
    rs = sorted(trbr)
    orr = [sum(1 for r in trbr[rd] if r.get("overran")) / len(trbr[rd]) for rd in rs]
    if rs:
        p = ph.line_plot({"overrun fraction": (rs, orr)}, "round",
                         "fraction of trainers over budget",
                         "Overrun rate over rounds (GPU contention)", d,
                         "overrun_rate_over_rounds.pdf", stamp=stamp)
        if p: saved.append(p)

    # Budget slack CDF: remaining_time_s is the sleep a trainer takes to fill its
    # training budget when the GPU finished early (remaining = budget - gpu > 0).
    # A fat tail here means many trainers are sleeping long = budget is too loose.
    # Zero means the trainer overran (no sleep). Plotting per-mode (real vs sim)
    # helps identify whether the budget is well-calibrated.
    slack_by_mode = defaultdict(list)
    for r in tr:
        rt = r.get("remaining_time_s")
        if rt is None:
            continue
        mode = str(r.get("time_mode", "unknown"))
        slack_by_mode[mode].append(float(rt))
    all_slack = [v for vals in slack_by_mode.values() for v in vals]
    if all_slack:
        if len(slack_by_mode) > 1:
            p = ph.cdf_multi({f"{k} (n={len(v)})": v for k, v in slack_by_mode.items()},
                             "remaining_time_s (sleep to fill budget; 0 = overrun)",
                             "Budget slack CDF by time mode (remaining_time_s)", d,
                             "budget_slack_cdf.pdf", stamp=stamp)
        else:
            p = ph.cdf_plot(all_slack,
                            "remaining_time_s (sleep to fill budget; 0 = overrun)",
                            "Budget slack CDF (remaining_time_s; 0 = overrun/on-time)", d,
                            "budget_slack_cdf.pdf", stamp=stamp)
        if p: saved.append(p)

    # aggregator-observed vs trainer-reported response time (overhead sanity):
    # if a trainer returns in 10s but the aggregator only processes it at 14s,
    # the 4s overhead shows up here (points above the diagonal).
    agg_obs = {}
    for r in by_event(records, EVENT_AGG_ROUND):
        rd = int(r.get("round", 0))
        for eid, sec in (r.get("agg_observed_s") or {}).items():
            agg_obs[(rd, str(eid))] = sec
    # trainer-reported response per trainer -> {round: duration}; match each
    # aggregator-observed value to that trainer's nearest round (robust to the
    # aggregator-round vs trained-model-version offset under staleness).
    tr_rep = defaultdict(dict)
    for r in tr:
        if r.get("sim_round_duration_s") is not None:
            tr_rep[str(r.get("end_id"))][int(r.get("round", 0))] = r["sim_round_duration_s"]
    xs, ys = [], []
    for (rd, eid), ao in agg_obs.items():
        cand = tr_rep.get(eid)
        if not cand:
            continue
        nearest = min(cand, key=lambda x: abs(x - rd))
        xs.append(cand[nearest]); ys.append(ao)
    if xs:
        p = ph.scatter_diag(xs, ys, "trainer-reported response (s)",
                            "aggregator-observed (s)",
                            "Aggregator-observed vs trainer-reported response", d,
                            "runtime_agg_vs_trainer.pdf", stamp=stamp)
        if p: saved.append(p)
        overhead = [y - x for x, y in zip(xs, ys)]
        p = ph.hist_plot(overhead,
                         "aggregator overhead = observed - reported (s)",
                         "Aggregator processing/network overhead", d,
                         "runtime_overhead_hist.pdf", stamp=stamp, vline=0.0)
        if p: saved.append(p)
        # CDF of the same overhead — reads the tail (P90/P99) at a glance.
        p = ph.cdf_plot(overhead, "aggregator overhead = observed - reported (s)",
                        "Aggregator overhead CDF", d,
                        "runtime_overhead_cdf.pdf", stamp=stamp)
        if p: saved.append(p)

    # expected vs actual utility (believed vs true)
    urows = [r for r in load_oracle_utility(tdir) if r.get("task") == "train"]
    bel, tru, vf = [], [], []
    for r in urows:
        try:
            b = float(r.get("believed")); t = float(r.get("true"))
        except (TypeError, ValueError):
            continue
        bel.append(b); tru.append(t)
    if bel:
        p = ph.scatter_diag(bel, tru, "believed utility (at selection)",
                            "true utility (oracle)",
                            "Trainer utility: believed vs true", d,
                            "utility_expected_vs_actual.pdf", stamp=stamp)
        if p: saved.append(p)
    # utility discrepancy over rounds
    rounds, corr, ngap = im_staleness_by_round(tdir)
    if rounds:
        p = ph.line_plot({"normalized |believed-true|": (rounds, ngap)}, "round",
                         "normalized utility discrepancy",
                         "Utility staleness magnitude over rounds", d,
                         "utility_discrepancy_over_rounds.pdf", stamp=stamp)
        if p: saved.append(p)

    # stale-update rejections (new telemetry; skip if absent)
    ar = by_event(records, EVENT_AGG_ROUND)
    rej = [(int(r.get("round", 0)), r.get("updates_accepted"), r.get("updates_rejected_stale"))
           for r in ar if r.get("updates_rejected_stale") is not None]
    if rej:
        rej.sort()
        xs = [r for r, _, _ in rej]
        p = ph.stacked_area(xs, {"accepted": [a or 0 for _, a, _ in rej],
                                 "rejected (stale)": [s or 0 for _, _, s in rej]},
                            "round", "updates", "Accepted vs stale-rejected updates",
                            d, "stale_update_rejections.pdf", stamp=stamp)
        if p: saved.append(p)

    # data unlock curve (banded)
    band = {}
    for rd in rs:
        fr = [(_visible_fraction(r)) for r in trbr[rd]]
        fr = [x for x in fr if x is not None]
        if fr:
            band[rd] = (sum(fr) / len(fr), min(fr), max(fr))
    if band:
        bx = sorted(band)
        p = ph.banded_line(bx, [band[r][0] for r in bx], [band[r][1] for r in bx],
                           [band[r][2] for r in bx], "round",
                           "visible fraction (data unlocked)",
                           "Data unlock over the run (mean / min-max)", d,
                           "data_unlock_curve.pdf", stamp=stamp)
        if p: saved.append(p)

    # selection-count consistency
    sel = by_event(records, EVENT_SELECTION)
    sel_by_round = defaultdict(int)
    for s in sel:
        if s.get("task", "train") == "train":
            sel_by_round[int(s.get("round", 0))] += len(s.get("chosen") or [])
    contrib = {int(r.get("round", 0)): len(r.get("contributing_trainers") or [])
               for r in ar}
    if sel_by_round:
        rr = sorted(set(sel_by_round) | set(contrib))
        series = {"chosen (train)": (rr, [sel_by_round.get(r, 0) for r in rr])}
        if contrib:
            series["contributing"] = (rr, [contrib.get(r, 0) for r in rr])
        p = ph.line_plot(series, "round", "trainer count",
                         "Selection/aggregation count consistency", d,
                         "selection_count_consistency.pdf", stamp=stamp)
        if p: saved.append(p)

    # Pre-train setup time CDF: time from train() entry to GPU compute start
    # (data loader rebuild, availability check). Large values mean GPU queue
    # contention — many co-located processes waiting to start the kernel.
    pre_train_vals = [r.get("pre_train_s") for r in by_event(records, EVENT_TRAINER_ROUND)
                      if r.get("pre_train_s") is not None]
    p = ph.cdf_plot(pre_train_vals, "pre_train_s (setup → GPU start)",
                    f"Pre-train setup time CDF (n={len(pre_train_vals)})",
                    d, "pre_train_s_cdf.pdf", stamp=stamp)
    if p: saved.append(p)

    # GPU compute time CDF: actual on-device time. Compared with budget this
    # directly shows how often the GPU overshoots its modelled training_delay_s.
    gpu_vals = [r.get("real_gpu_time_s") for r in by_event(records, EVENT_TRAINER_ROUND)
                if r.get("real_gpu_time_s") is not None
                and str(r.get("task_to_perform", "train")) == "train"]
    p = ph.cdf_plot(gpu_vals, "real_gpu_time_s",
                    f"GPU compute time CDF — train rounds (n={len(gpu_vals)})",
                    d, "gpu_compute_cdf.pdf", stamp=stamp)
    if p: saved.append(p)

    # SEND_RECV_LAG CDF: wall-clock time between aggregator sending a task and
    # receiving the gradient back. Instrumented in BOTH sync and async aggregators
    # so this plot is present for every baseline. When no events are found the
    # plot is produced as a "no data" placeholder — never silently absent.
    wall_lags, overrun_cnt = parse_send_recv_lags(tdir)
    lag_title_suffix = (f"n={len(wall_lags)}, timing_overruns={overrun_cnt}"
                        if wall_lags else "no events in agg log")
    p = ph.cdf_plot(wall_lags,
                    "wall_lag_s (agg send → grad recv)",
                    f"Send-recv lag CDF ({lag_title_suffix})",
                    d, "send_recv_lag_cdf.pdf", stamp=stamp)
    if p: saved.append(p)
    p = ph.hist_plot(wall_lags,
                     "wall_lag_s (agg send → grad recv)",
                     f"Send-recv lag histogram ({lag_title_suffix})",
                     d, "send_recv_lag_hist.pdf", stamp=stamp, vline=None)
    if p: saved.append(p)

    # Overrun magnitude CDF: for every update where wall_lag > budget, what is
    # the excess (wall_lag - budget)?  Separates "slightly late" from "stuck for
    # minutes". Parsed from [TIMING_OVERRUN_AGG] lines.
    overrun_excesses = _parse_overrun_excesses(tdir)
    p = ph.cdf_plot(overrun_excesses,
                    "excess = wall_lag - budget (s)",
                    f"Timing-overrun excess CDF (n={len(overrun_excesses)}; "
                    f"0 = no overruns)",
                    d, "timing_overrun_excess_cdf.pdf", stamp=stamp)
    if p: saved.append(p)
    return saved


# ==========================================================================
# selection/
# ==========================================================================


def selection_plots(records, out, stamp, tdir):
    d = _sub(out, "selection"); saved = []
    sel = by_event(records, EVENT_SELECTION)

    # selected vs pool true utility (from oracle)
    urows = [r for r in load_oracle_utility(tdir) if r.get("task") == "train"]
    byr = defaultdict(lambda: {"sel": [], "all": []})
    for r in urows:
        try:
            t = float(r.get("true"))
        except (TypeError, ValueError):
            continue
        rd = int(float(r["round"]))
        byr[rd]["all"].append(t)
        if str(r.get("selected")) in ("1", "True", "true"):
            byr[rd]["sel"].append(t)
    rr = sorted(byr)
    if rr:
        sel_u = [(sum(byr[r]["sel"]) / len(byr[r]["sel"])) if byr[r]["sel"] else None for r in rr]
        all_u = [(sum(byr[r]["all"]) / len(byr[r]["all"])) if byr[r]["all"] else None for r in rr]
        p = ph.line_plot({"selected": (rr, sel_u), "candidate pool": (rr, all_u)},
                         "round", "mean true utility",
                         "True utility: selected vs candidate pool", d,
                         "selected_vs_pool_utility.pdf", stamp=stamp)
        if p: saved.append(p)

    # cumulative unique selected + Gini
    freq = Counter()
    cum_unique, seen, xs = [], set(), []
    for s in sorted(sel, key=lambda r: r.get("round", 0)):
        for c in (s.get("chosen") or []):
            freq[str(c)] += 1
            seen.add(str(c))
        xs.append(int(s.get("round", 0)))
        cum_unique.append(len(seen))
    if xs:
        import numpy as np
        def gini(counts):
            a = np.sort(np.asarray(counts, float))
            if a.sum() == 0:
                return 0.0
            n = len(a)
            return float((2 * np.arange(1, n + 1) - n - 1).dot(a) / (n * a.sum()))
        # gini computed over running frequency snapshots is heavy; report final + curve of unique
        p = ph.line_plot({"cumulative unique selected": (xs, cum_unique)}, "round",
                         "unique trainers selected",
                         "Selection coverage (cumulative unique, Gini=%.2f)" % gini(list(freq.values())),
                         d, "selection_coverage.pdf", stamp=stamp)
        if p: saved.append(p)
    if freq:
        # Lorenz curve of selection inequality (replaces a per-trainer bar that
        # was unreadable at 300 trainers). Level 1: overall. Level 2: split by
        # availability — trainers that were EVER unavailable vs always-available
        # (degenerate/absent for syn_0, where everyone is always available).
        ever_unavail = {
            str(r.get("end_id")) for r in by_event(records, EVENT_AVAIL_CHANGE)
            if str(r.get("new_state", "")).lower() not in ("", "avl_train", "available")
        }
        series = {"all trainers": list(freq.values())}
        if ever_unavail and any(t in ever_unavail for t in freq):
            series["always-available"] = [v for t, v in freq.items() if t not in ever_unavail]
            series["ever-unavailable"] = [v for t, v in freq.items() if t in ever_unavail]
        p = ph.lorenz_plot(series,
                           "Selection fairness (Lorenz; lower Gini = more equal)", d,
                           "selection_fairness_lorenz.pdf", stamp=stamp)
        if p: saved.append(p)

    # exploration factor over rounds
    ef = [(int(s.get("round", 0)), s.get("exploration_factor")) for s in sel
          if s.get("exploration_factor") is not None]
    if ef:
        ef = sorted(set(ef))
        p = ph.line_plot({"exploration factor": ([r for r, _ in ef], [v for _, v in ef])},
                         "round", "exploration factor",
                         "Exploration factor decay", d,
                         "exploration_factor_over_rounds.pdf", stamp=stamp)
        if p: saved.append(p)

    # eval vs train selections per round (felix)
    et = defaultdict(lambda: {"train": 0, "eval": 0})
    for s in sel:
        et[int(s.get("round", 0))][s.get("task", "train")] += len(s.get("chosen") or [])
    if any(v["eval"] for v in et.values()):
        rr = sorted(et)
        p = ph.stacked_area(rr, {"train": [et[r]["train"] for r in rr],
                                 "eval": [et[r]["eval"] for r in rr]},
                            "round", "selections", "Train vs eval selections per round",
                            d, "eval_vs_train_selections.pdf", stamp=stamp)
        if p: saved.append(p)

    # per-round average BELIEVED speed & utility of the clients actually picked
    # (what the selector "saw" about its picks each round)
    spd, utl = {}, {}
    for s in sel:
        if s.get("task", "train") != "train":
            continue
        pt = s.get("per_trainer") or {}
        chosen = [str(c) for c in (s.get("chosen") or [])]
        sp = [pt[c].get("speed_s") for c in chosen if c in pt and pt[c].get("speed_s") is not None]
        uu = [pt[c].get("believed_I", pt[c].get("utility")) for c in chosen
              if c in pt and pt[c].get("believed_I", pt[c].get("utility")) is not None]
        rd = int(s.get("round", 0))
        if sp:
            spd[rd] = sum(sp) / len(sp)
        if uu:
            utl[rd] = sum(uu) / len(uu)
    both = sorted(set(spd) & set(utl))
    if both:
        p = ph.dual_axis_line(both, [spd[r] for r in both], [utl[r] for r in both],
                              "round", "avg speed of picked (s)",
                              "avg believed utility of picked",
                              "Picked clients: avg speed & utility per round", d,
                              "selected_speed_utility_over_rounds.pdf", stamp=stamp)
        if p: saved.append(p)
    elif utl:  # e.g. FedDance has no speed factor
        rr = sorted(utl)
        p = ph.line_plot({"avg believed utility of picked": (rr, [utl[r] for r in rr])},
                         "round", "avg believed utility of picked",
                         "Picked clients: avg utility per round", d,
                         "selected_speed_utility_over_rounds.pdf", stamp=stamp)
        if p: saved.append(p)

    # #14: separate CDFs of the picked clients' believed SPEED and believed
    # UTILITY (over all picks), plus expected-vs-actual utility — the believed
    # value at selection vs the trainer's actual stat_utility that round.
    all_sp, all_bel = [], []
    exp_act = {"believed (at selection)": [], "actual (trainer stat_utility)": []}
    actual_util = {}  # (round, end3) -> stat_utility
    for r in by_event(records, EVENT_TRAINER_ROUND):
        if r.get("stat_utility") is not None:
            actual_util[(int(r.get("round", 0)), str(r.get("end_id", ""))[-3:])] = float(r["stat_utility"])
    for s in sel:
        if s.get("task", "train") != "train":
            continue
        pt = s.get("per_trainer") or {}
        rd = int(s.get("round", 0))
        for c in (s.get("chosen") or []):
            info = pt.get(str(c)) or {}
            if info.get("speed_s") is not None:
                all_sp.append(float(info["speed_s"]))
            bel = info.get("believed_I", info.get("utility"))
            if bel is not None:
                all_bel.append(float(bel))
                act = actual_util.get((rd, str(c)[-3:]))
                if act is not None:
                    exp_act["believed (at selection)"].append(float(bel))
                    exp_act["actual (trainer stat_utility)"].append(act)
    cdfs = {}
    if all_sp:
        cdfs["speed (s)"] = all_sp
    if cdfs:
        p = ph.cdf_multi(cdfs, "value", "Picked-client speed distribution (CDF)", d,
                         "selected_speed_cdf.pdf", stamp=stamp)
        if p: saved.append(p)
    if all_bel:
        p = ph.cdf_multi({"believed utility": all_bel}, "utility",
                         "Picked-client utility distribution (CDF)", d,
                         "selected_utility_cdf.pdf", stamp=stamp)
        if p: saved.append(p)
    if exp_act["actual (trainer stat_utility)"]:
        p = ph.cdf_multi(exp_act, "utility",
                         "Picked-client utility: expected (believed) vs actual", d,
                         "selected_utility_expected_vs_actual_cdf.pdf", stamp=stamp)
        if p: saved.append(p)

    # participation heatmap (trainer x round: 0 idle, 1 eval-selected, 2 trained)
    saved += _participation_heatmap(records, d, stamp)

    # availability composition
    per_round = {}
    for r in sel:
        comp = r.get("avail_composition")
        if comp:
            per_round[int(r.get("round", 0))] = comp
    if per_round:
        rr = sorted(per_round)
        states = sorted({s for c in per_round.values() for s in c})
        series = {s: [per_round[r].get(s, 0) for r in rr] for s in states}
        p = ph.stacked_area(rr, series, "round", "trainer count",
                            "Availability composition over rounds", d,
                            "availability_composition.pdf", stamp=stamp)
        if p: saved.append(p)
    return saved


def _participation_heatmap(records, d, stamp):
    import numpy as np
    trained = defaultdict(set)  # round -> trainers (trained)
    for r in by_event(records, EVENT_TRAINER_ROUND):
        trained[int(r.get("round", 0))].add(str(r.get("end_id")))
    evalsel = defaultdict(set)
    for s in by_event(records, EVENT_SELECTION):
        if s.get("task") == "eval":
            for c in (s.get("chosen") or []):
                evalsel[int(s.get("round", 0))].add(str(c))
    trainers = sorted({t for s in trained.values() for t in s}
                      | {t for s in evalsel.values() for t in s})
    rounds = sorted(set(trained) | set(evalsel))
    if not trainers or not rounds:
        return []
    # Per-trainer availability per round (forward-filled from avail_change), so
    # idle cells can distinguish "available, not picked" from "unavailable".
    unavail = defaultdict(dict)  # trainer -> {round: True if UN_AVL}
    ac = defaultdict(list)
    for r in by_event(records, EVENT_AVAIL_CHANGE):
        ac[str(r.get("end_id"))].append((int(r.get("round", 0)),
                                         str(r.get("new_state", ""))))
    for t, evs in ac.items():
        evs.sort()
        cur = None
        for rd, st in evs:
            cur = st
            unavail[t][rd] = ("UN_AVL" in cur)
    idx = {t: i for i, t in enumerate(trainers)}
    ridx = {r: j for j, r in enumerate(rounds)}
    # 0 not-selected, 1 eval, 2 train, 3 unavailable (not selected),
    # 4 selected-but-unavailable. Availability forward-filled across rounds.
    m = np.zeros((len(trainers), len(rounds)))
    for t in trainers:
        last_un = False
        for r in rounds:
            if r in unavail.get(t, {}):
                last_un = unavail[t][r]
            sel = (t in evalsel.get(r, set())) or (t in trained.get(r, set()))
            if t in trained.get(r, set()):
                v = 4 if last_un else 2
            elif t in evalsel.get(r, set()):
                v = 4 if last_un else 1
            else:
                v = 3 if last_un else 0
            m[idx[t], ridx[r]] = v
    yl = [t[-3:] for t in trainers] if len(trainers) <= 40 else None
    p = ph.heatmap(m, "round", "trainer", "Participation per trainer x round", d,
                   "participation_heatmap.pdf", stamp=stamp, yticklabels=yl,
                   discrete=[(0, "not selected", "#cfcfcf"),
                             (1, "eval", "#9ecae1"),
                             (2, "train", "#a1d99b"),
                             (3, "unavailable", "#fcae91"),
                             (4, "selected but unavail", "#fd8d3c")])
    return [p] if p else []


# ==========================================================================
# insights/
# ==========================================================================


def insights_plots(records, out, stamp, tdir):
    d = _sub(out, "insights"); saved = []
    # I_m staleness
    rounds, corr, ngap = im_staleness_by_round(tdir)
    if rounds:
        p = ph.line_plot({"rank-corr(believed,true)": (rounds, corr)}, "round",
                         "rank correlation",
                         "I_m staleness: belief-vs-truth ranking (1=perfect)", d,
                         "Im_staleness_rankcorr.pdf", stamp=stamp)
        if p: saved.append(p)
    # mis-selection (generic oracle)
    mrows = [r for r in load_oracle_misselection(tdir) if r.get("task") == "train"]
    if mrows:
        mrows.sort(key=lambda r: float(r.get("round", 0)))
        xs = _floats(mrows, "round")
        p = ph.line_plot({"mis-selection rate": (xs, _floats(mrows, "misselection_rate")),
                          "utility regret": (xs, _floats(mrows, "utility_regret"))},
                         "round", "value",
                         "Mis-selection vs oracle top-k", d,
                         "misselection_over_rounds.pdf", stamp=stamp)
        if p: saved.append(p)
    # counterfactual (self-relative)
    crows = load_oracle_counterfactual(tdir)
    if crows:
        crows.sort(key=lambda r: float(r.get("round", 0)))
        xs = _floats(crows, "round")
        p = ph.line_plot({"cf mis-selection": (xs, _floats(crows, "cf_misselection_rate"))},
                         "round", "counterfactual mis-selection rate",
                         "Self-relative mis-selection (own true-factor pick)", d,
                         "cf_misselection_over_rounds.pdf", stamp=stamp)
        if p: saved.append(p)
        p = ph.line_plot({"cf regret": (xs, _floats(crows, "cf_utility_regret"))},
                         "round", "counterfactual regret (own true-score)",
                         "Self-relative regret (true-score forfeited to stale factors)",
                         d, "cf_regret_over_rounds.pdf", stamp=stamp)
        if p: saved.append(p)
    # data-unlock effects (vs visible fraction)
    tr = by_event(records, EVENT_TRAINER_ROUND)
    for key, ylab, fname, title in [
        ("delta_weight_l2", "update L2 norm", "update_norm_vs_visible.pdf",
         "Update magnitude vs unlocked data"),
        ("final_loss", "final training loss", "loss_vs_visible.pdf",
         "Training loss vs unlocked data"),
        ("stat_utility", "statistical utility", "utility_vs_visible.pdf",
         "Statistical utility vs unlocked data"),
    ]:
        xy = [(_visible_fraction(r), r.get(key)) for r in tr]
        xy = [(a, b) for a, b in xy if a is not None and b is not None]
        if xy:
            p = ph.scatter_plot([a for a, _ in xy], [b for _, b in xy], None,
                                "visible fraction", ylab, title, d, fname, stamp=stamp)
            if p: saved.append(p)
    # util disparity streamed vs full
    ud = by_event(records, EVENT_UTIL_DISPARITY)
    if ud:
        byr = defaultdict(list)
        for r in ud:
            v = r.get("utility_ratio")
            if v is not None:
                byr[int(r.get("round", 0))].append(v)
        rr = sorted(byr)
        if rr:
            p = ph.line_plot({"streamed/full ratio": (rr, [sum(byr[r]) / len(byr[r]) for r in rr])},
                             "round", "streamed / full utility ratio",
                             "Streamed-vs-full utility ratio (1=no disparity)", d,
                             "util_disparity_ratio.pdf", stamp=stamp, target=1.0)
            if p: saved.append(p)
    return saved


# ==========================================================================
# resource monitor parsing
# ==========================================================================

_RE_RESOURCE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"         # timestamp
    r".*?RAM:\s*([\d.]+)GB / ([\d.]+)GB \(([\d.]+)%\)"  # RAM
    r".*?Swap:\s*([\d.]+)GB \(([\d.]+)%\)"               # Swap
)
_RE_GPU = re.compile(
    r"GPU(\d+):\s*([\d.]+)/([\d.]+)GB \(([\d.]+)%\) Util:(\d+)%"
)


def parse_resource_log(run_dir: str) -> list[dict]:
    """Parse the *_resources.log in run_dir.  Returns a list of dicts with
    keys: timestamp_s (epoch float), ram_gb, ram_pct, swap_gb, swap_pct,
    and gpu_N_mem_gb / gpu_N_mem_pct / gpu_N_util_pct per GPU index N."""
    candidates = glob.glob(os.path.join(run_dir, "*_resources.log"))
    if not candidates:
        return []
    rows = []
    with open(candidates[0], encoding="utf-8") as fh:
        for line in fh:
            m = _RE_RESOURCE.match(line)
            if not m:
                continue
            ts_str, ram_gb, ram_total_gb, ram_pct, swap_gb, swap_pct = m.groups()
            from datetime import datetime as _dt
            ts = _dt.strptime(ts_str, "%Y-%m-%d %H:%M:%S").timestamp()
            row: dict = {
                "timestamp_s": ts,
                "ram_gb": float(ram_gb),
                "ram_pct": float(ram_pct),
                "swap_gb": float(swap_gb),
                "swap_pct": float(swap_pct),
            }
            for gm in _RE_GPU.finditer(line):
                idx, mem_gb, mem_total_gb, mem_pct, util_pct = gm.groups()
                n = int(idx)
                row[f"gpu_{n}_mem_gb"] = float(mem_gb)
                row[f"gpu_{n}_mem_pct"] = float(mem_pct)
                row[f"gpu_{n}_util_pct"] = float(util_pct)
            rows.append(row)
    return rows


def resource_plots(out: str, stamp: str, run_dir: str) -> list[str]:
    """Generate RAM and GPU resource usage plots from the run's *_resources.log."""
    rows = parse_resource_log(run_dir)
    if not rows:
        return []
    d = _sub(out, "system")
    saved = []

    t0 = rows[0]["timestamp_s"]
    times = [(r["timestamp_s"] - t0) / 60.0 for r in rows]  # minutes since start

    # RAM over time
    ram_vals = [r["ram_gb"] for r in rows]
    swap_vals = [r["swap_gb"] for r in rows]
    ram_series = {"RAM used (GB)": (times, ram_vals)}
    if any(s > 0.05 for s in swap_vals):
        ram_series["Swap used (GB)"] = (times, swap_vals)
    p = ph.line_plot(ram_series, "time (min)", "GB",
                     "RAM usage over time", d, "resource_ram_over_time.pdf",
                     stamp=stamp)
    if p:
        saved.append(p)

    # GPU utilization over time (one series per GPU)
    gpu_indices = sorted({int(k.split("_")[1]) for k in rows[0] if k.startswith("gpu_") and k.endswith("_util_pct")})
    if gpu_indices:
        util_series = {}
        for n in gpu_indices:
            key = f"gpu_{n}_util_pct"
            vals = [r.get(key, 0.0) for r in rows]
            if any(v > 0 for v in vals):
                util_series[f"GPU{n} util%"] = (times, vals)
        if not util_series:
            # all zeros — include one flat series so the plot is visible
            util_series["GPU util% (all 0)"] = (times, [0.0] * len(times))
        p = ph.line_plot(util_series, "time (min)", "util (%)",
                         "GPU utilization over time", d, "resource_gpu_util_over_time.pdf",
                         stamp=stamp)
        if p:
            saved.append(p)

        # GPU memory over time
        mem_series = {}
        for n in gpu_indices:
            key = f"gpu_{n}_mem_gb"
            vals = [r.get(key, 0.0) for r in rows]
            mem_series[f"GPU{n} mem (GB)"] = (times, vals)
        p = ph.line_plot(mem_series, "time (min)", "memory (GB)",
                         "GPU memory usage over time", d, "resource_gpu_mem_over_time.pdf",
                         stamp=stamp)
        if p:
            saved.append(p)

    return saved


# ==========================================================================
# system/
# ==========================================================================


def system_plots(records, out, stamp, tdir):
    d = _sub(out, "system"); saved = []
    xs, ys = comm_vs_accuracy_series(records)
    if xs:
        p = ph.line_plot({"accuracy": (xs, ys)},
                         "cumulative comm (MB)", "test accuracy",
                         f"Communication cost vs accuracy (model={MODEL_MB:.2f} MB)", d,
                         "comm_vs_accuracy.pdf", stamp=stamp)
        if p: saved.append(p)
    # per-round comm split by task AND direction (MB): train down / train up /
    # eval down (eval up is a tiny utility scalar, ~0 MB). #17.
    cb = comm_breakdown_by_round(records)
    if cb:
        rr = sorted(cb)
        td = [cb[r]["train_down_mb"] for r in rr]
        tu = [cb[r]["train_up_mb"] for r in rr]
        ed = [cb[r]["eval_down_mb"] for r in rr]
        tot = sum(td) + sum(tu) + sum(ed)
        p = ph.stacked_area(rr, {"train down (agg->tr)": td,
                                 "train up (tr->agg)": tu,
                                 "eval down (agg->tr)": ed},
                            "round", "comm (MB)",
                            f"Per-round comm by task+direction "
                            f"(total {tot:.0f} MB: train {sum(td) + sum(tu):.0f}, eval {sum(ed):.0f})",
                            d, "comm_per_round_train_vs_eval.pdf", stamp=stamp)
        if p: saved.append(p)
        # message counts: sent (down), returned (up), discarded (sent-but-unused)
        p = ph.stacked_area(rr, {"returned (up)": [cb[r]["msgs_up"] for r in rr],
                                 "discarded (sent, unused)": [cb[r]["discarded"] for r in rr]},
                            "round", "messages / round",
                            f"Message accounting (down {sum(cb[r]['msgs_down'] for r in rr)}, "
                            f"up {sum(cb[r]['msgs_up'] for r in rr)}, "
                            f"discarded {sum(cb[r]['discarded'] for r in rr)})",
                            d, "comm_message_accounting.pdf", stamp=stamp)
        if p: saved.append(p)
    # trainer time breakdown (mean per trainer)
    agg = defaultdict(lambda: {"gpu": [], "sim": [], "wait": []})
    for r in by_event(records, EVENT_TRAINER_ROUND):
        tid = str(r.get("end_id", "?"))[-3:]
        agg[tid]["gpu"].append(r.get("real_gpu_time_s") or 0.0)
        agg[tid]["sim"].append(r.get("sim_round_duration_s") or 0.0)
        agg[tid]["wait"].append(r.get("wait_time_s") or 0.0)
    if agg:
        cats = sorted(agg)
        mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
        segs = {"gpu": [mean(agg[c]["gpu"]) for c in cats],
                "sim-delay": [max(0.0, mean(agg[c]["sim"]) - mean(agg[c]["gpu"])) for c in cats],
                "wait": [mean(agg[c]["wait"]) for c in cats]}
        p = ph.stacked_bar(cats, segs, "mean seconds / round",
                           "Trainer time breakdown (gpu / sim-delay / wait)", d,
                           "trainer_time_breakdown.pdf", stamp=stamp)
        if p: saved.append(p)
    # aggregate round-time split across ALL trainers per round (setup / GPU /
    # post-cleanup / modeled sleep) — the system-level view of where a round's
    # wall time goes. Uses pre_train_s/post_train_s (telemetry); absent fields
    # default to 0 so older runs degrade gracefully.
    split_rd = defaultdict(lambda: defaultdict(list))
    for r in by_event(records, EVENT_TRAINER_ROUND):
        rd = int(r.get("round", 0))
        split_rd[rd]["pre (setup)"].append(r.get("pre_train_s") or 0.0)
        split_rd[rd]["gpu compute"].append(r.get("real_gpu_time_s") or 0.0)
        split_rd[rd]["post (cleanup)"].append(r.get("post_train_s") or 0.0)
        split_rd[rd]["sleep (budget)"].append(r.get("remaining_time_s") or 0.0)
    if split_rd:
        rr = sorted(split_rd)
        _m = lambda xs: sum(xs) / len(xs) if xs else 0.0
        series = {k: [_m(split_rd[r][k]) for r in rr]
                  for k in ("pre (setup)", "gpu compute", "post (cleanup)", "sleep (budget)")}
        p = ph.stacked_area(rr, series, "round",
                            "mean seconds / round (across trainers)",
                            "Trainer round-time split (mean across trainers)", d,
                            "trainer_time_split_over_rounds.pdf", stamp=stamp)
        if p: saved.append(p)
        # #19: whole-run overall split as a single stacked bar (the area above,
        # collapsed over all rounds) — one glanceable "where did time go" summary.
        allv = defaultdict(list)
        for rd in rr:
            for k in series:
                allv[k].extend(split_rd[rd][k])
        overall = {k: [_m(allv[k])] for k in series}
        p = ph.stacked_bar(["whole run"], overall, "mean seconds / round",
                           "Trainer round-time split (whole-run mean)", d,
                           "trainer_time_split_overall.pdf", stamp=stamp)
        if p: saved.append(p)
    # #13: trainer compute time split by task (train vs eval) — per-round GPU
    # seconds grouped by task_to_perform, aggregate distribution as a CDF (train
    # and eval cost differently; eval is forward-only).
    comp_by_task = defaultdict(list)
    for r in by_event(records, EVENT_TRAINER_ROUND):
        g = r.get("real_gpu_time_s")
        if g is not None:
            comp_by_task[str(r.get("task_to_perform", "train"))].append(float(g))
    series_ct = {f"{k} (n={len(v)}, mean={sum(v) / len(v):.2f}s)": sorted(v)
                 for k, v in comp_by_task.items() if v}
    if series_ct:
        p = ph.cdf_multi(series_ct, "trainer GPU compute (s)",
                         "Trainer compute time by task (train vs eval)", d,
                         "compute_time_by_task_cdf.pdf", stamp=stamp)
        if p: saved.append(p)
    # queue depth
    inflight = [(int(r.get("round", 0)), r.get("updates_in_queue")) for r in
                by_event(records, EVENT_AGG_ROUND) if r.get("updates_in_queue") is not None]
    if inflight:
        inflight.sort()
        p = ph.line_plot({"updates in queue": ([r for r, _ in inflight],
                                               [v for _, v in inflight])},
                         "round", "updates in queue", "Async queue depth over rounds",
                         d, "queue_depth_over_rounds.pdf", stamp=stamp)
        if p: saved.append(p)

    # Send-recv lag over rounds: median wall_lag per round, parsed from the
    # aggregator log. Now instrumented in BOTH sync and async aggregators, so
    # this plot exists for all baselines (placeholder when no events).
    wall_lags, _overruns = parse_send_recv_lags(tdir)
    rd_lags: dict[int, list[float]] = defaultdict(list)
    if wall_lags:
        run_dir = os.path.dirname(os.path.abspath(tdir))
        agg_logs = glob.glob(os.path.join(run_dir, "*aggregator*.log"))
        _lag_re = re.compile(
            r"increment_round.*round (\d+)"
        )
        _lag_entry_re = re.compile(
            r"\[SEND_RECV_LAG\] end=\S+ version=\d+ wall_lag_s=([0-9.]+)"
        )
        if agg_logs:
            current_round = 0
            with open(agg_logs[0]) as fh:
                for line in fh:
                    rm = _lag_re.search(line)
                    if rm:
                        current_round = int(rm.group(1))
                    lm = _lag_entry_re.search(line)
                    if lm:
                        rd_lags[current_round].append(float(lm.group(1)))
    _m = lambda xs: sum(xs) / len(xs) if xs else 0.0
    lag_series = {}
    if rd_lags:
        rr_lag = sorted(rd_lags)
        lag_series["median wall_lag_s"] = (rr_lag, [_m(rd_lags[r]) for r in rr_lag])
    p = ph.line_plot(
        lag_series,
        "round", "median wall_lag_s (send→recv)",
        f"Send-recv lag over rounds (n={len(wall_lags)}, overruns={_overruns})",
        d, "send_recv_lag_over_rounds.pdf", stamp=stamp)
    if p: saved.append(p)

    # Full lag decomposition from [LAG_DECOMP]: 6 components per update.
    # (i) agg_to_trainer: agg channel.send → trainer channel.recv (network delivery out)
    # (ii) compute: max(gpu_time, training_delay_s) — modeled round cost
    # (iii) post_wait: trainer channel.recv + compute → trainer channel.send (overhead + sleep)
    # (iv) mqtt_lag: trainer channel.send → MQTT arrival at agg (network delivery back)
    # (v+vi) queue_wait: MQTT arrival → aggregator dequeues (sim buffer wait; ~0 real)
    # (vii) process: aggregator per-message overhead (property sets, logging)
    decomp = _parse_lag_decomp(tdir)
    _n_decomp = len(decomp["wall_lag_s"])

    # Individual CDFs for each component
    _comp_labels = {
        "agg_to_trainer_s": "agg→trainer delivery (s)",
        "compute_s":         "trainer compute max(gpu,D) (s)",
        "post_wait_s":       "post-compute wait (s)",
        "mqtt_lag_s":        "trainer→agg MQTT delivery (s)",
        "queue_wait_s":      "sim buffer / queue wait (s)",
        "process_s":         "agg per-msg processing (s)",
    }
    for key, xlabel in _comp_labels.items():
        vals = decomp[key]
        if vals:
            p = ph.cdf_plot(vals, xlabel,
                            f"Lag component: {key} CDF (n={len(vals)})",
                            d, f"lag_decomp_{key}_cdf.pdf", stamp=stamp)
            if p: saved.append(p)

    # Multi-series CDF: all components + total for visual comparison.
    decomp_series: dict[str, list[float]] = {}
    if decomp["wall_lag_s"]:
        decomp_series[f"wall_lag_s (n={_n_decomp})"] = sorted(decomp["wall_lag_s"])
    for key, xlabel in _comp_labels.items():
        if decomp[key]:
            decomp_series[f"{key} (n={len(decomp[key])})"] = sorted(decomp[key])
    p = ph.cdf_multi(decomp_series,
                     "seconds",
                     "Wall-lag 6-component decomposition CDF",
                     d, "wall_lag_decomposition_cdf.pdf", stamp=stamp)
    if p: saved.append(p)

    return saved


# ==========================================================================
# orchestration
# ==========================================================================


def write_summary(records, out, tdir):
    counts = Counter(r.get("event") for r in records)
    n_tr = len({r.get("end_id") for r in records if r.get("role") == "trainer"})
    lines = [f"Telemetry summary: {tdir}", f"config: {ph.config_stamp(os.path.dirname(os.path.abspath(tdir)))}",
             f"total events: {len(records)}", f"trainers seen: {n_tr}", "event counts:"]
    for ev, c in counts.most_common():
        lines.append(f"  {ev:16} {c}")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "summary.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def analyze(telemetry_dir, out_dir=None):
    records = load_events(telemetry_dir)
    run_dir = os.path.dirname(os.path.abspath(telemetry_dir))
    if out_dir is None:
        out_dir = os.path.join(run_dir, "plots")
    if not records:
        print("no telemetry events found in %s" % telemetry_dir)
        return []
    stamp = ph.config_stamp(run_dir)
    saved = []
    for fn in (perf_plots, sanity_plots, selection_plots, insights_plots, system_plots):
        try:
            saved.extend(fn(records, out_dir, stamp, telemetry_dir))
        except Exception as e:
            print("  (%s failed: %s)" % (fn.__name__, e))
    try:
        saved.extend(resource_plots(out_dir, stamp, run_dir))
    except Exception as e:
        print("  (resource_plots failed: %s)" % e)
    saved.append(write_summary(records, out_dir, telemetry_dir))
    print("wrote %d artifact(s) under %s" % (len(saved), out_dir))
    for p in saved:
        print("  %s" % p)
    return saved


# ==========================================================================
# cross-run comparison (kept; not the per-run focus)
# ==========================================================================


def compare_streaming(dirs, labels, out_dir, target=0.6):
    if labels is None or len(labels) != len(dirs):
        labels = [os.path.basename(os.path.dirname(os.path.abspath(d))) or d for d in dirs]
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    acc_series, comm_acc, true_util, missel, regret = {}, {}, {}, {}, {}
    util_ratio, cf_miss, cf_reg, ttt_rows = {}, {}, {}, []
    corr_series, gap_series = {}, {}
    for label, d in zip(labels, dirs):
        recs = load_events(d)
        acc = accuracy_by_round(recs)
        if acc:
            rs = sorted(acc); acc_series[label] = (rs, [acc[r] for r in rs])
        xs, ys = comm_vs_accuracy_series(recs)
        if xs:
            comm_acc[label] = (xs, ys)
        ttt_rows.append({"label": label, **time_to_target(recs, target)})
        ud = by_event(recs, EVENT_UTIL_DISPARITY)
        if ud:
            byr = defaultdict(list)
            for r in ud:
                if r.get("utility_ratio") is not None:
                    byr[int(r.get("round", 0))].append(r["utility_ratio"])
            rs = sorted(byr)
            util_ratio[label] = (rs, [sum(byr[r]) / len(byr[r]) for r in rs])
        orows = [r for r in load_oracle_misselection(d) if r.get("task") == "train"]
        if orows:
            orows.sort(key=lambda r: float(r.get("round", 0)))
            ox = _floats(orows, "round")
            true_util[label] = (ox, _floats(orows, "mean_true_selected"))
            missel[label] = (ox, _floats(orows, "misselection_rate"))
            regret[label] = (ox, _floats(orows, "utility_regret"))
        rounds, corr, ngap = im_staleness_by_round(d)
        if rounds:
            corr_series[label] = (rounds, corr); gap_series[label] = (rounds, ngap)
        crows = load_oracle_counterfactual(d)
        if crows:
            crows.sort(key=lambda r: float(r.get("round", 0)))
            cx = _floats(crows, "round")
            cf_miss[label] = (cx, _floats(crows, "cf_misselection_rate"))
            cf_reg[label] = (cx, _floats(crows, "cf_utility_regret"))

    def _l(series, xl, yl, title, fname, **kw):
        p = ph.line_plot(series, xl, yl, title, out_dir, fname, **kw)
        if p: saved.append(p)

    _l(acc_series, "round", "test accuracy", "Accuracy across runs", "compare_accuracy.pdf", target=target)
    _l(comm_acc, "cumulative comm (MB)", "test accuracy", "Comm vs accuracy", "compare_comm_vs_accuracy.pdf")
    _l(true_util, "round", "mean true utility of selected", "Selected-set true utility", "compare_true_utility_selected.pdf")
    _l(missel, "round", "mis-selection rate", "Mis-selection (oracle top-k)", "compare_misselection.pdf")
    _l(regret, "round", "utility regret", "Selection regret (oracle)", "compare_utility_regret.pdf")
    _l(util_ratio, "round", "streamed/full ratio", "Streamed-vs-full utility ratio", "compare_util_disparity.pdf")
    _l(corr_series, "round", "rank-corr(believed,true)", "I_m staleness ranking", "compare_Im_rankcorr.pdf")
    _l(gap_series, "round", "norm |believed-true|", "I_m staleness magnitude", "compare_Im_gap.pdf")
    _l(cf_miss, "round", "cf mis-selection rate", "Self-relative mis-selection", "compare_cf_misselection.pdf")
    _l(cf_reg, "round", "cf regret (own true-score)", "Self-relative regret", "compare_cf_regret.pdf")

    ttt_path = os.path.join(out_dir, "compare_time_to_target.csv")
    with open(ttt_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["label", "reached", "round", "wall_s", "sim_s", "accuracy"])
        w.writeheader()
        for r in ttt_rows:
            w.writerow({k: r.get(k) for k in ["label", "reached", "round", "wall_s", "sim_s", "accuracy"]})
    saved.append(ttt_path)
    cats = [r["label"] for r in ttt_rows if r.get("round") is not None]
    if cats:
        p = ph.bar_plot(cats, [r["round"] for r in ttt_rows if r.get("round") is not None],
                        "rounds to target", "Rounds to %.0f%% accuracy" % (target * 100),
                        out_dir, "compare_time_to_target_rounds.pdf")
        if p: saved.append(p)
    print("wrote %d comparison artifact(s) to %s" % (len(saved), out_dir))
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("telemetry_dir", nargs="?", help="run telemetry directory")
    parser.add_argument("--out", help="output plots directory")
    parser.add_argument("--compare-streaming", nargs="+",
                        help="telemetry dirs for cross-run comparison")
    parser.add_argument("--labels", nargs="+", help="labels for --compare-streaming dirs")
    parser.add_argument("--target", type=float, default=0.6)
    parser.add_argument("--model-params", type=int, default=None,
                        help="param count for MB comm conversion (default: async_cifar10 Net)")
    args = parser.parse_args()
    if args.model_params:
        global MODEL_PARAM_COUNT, MODEL_MB
        MODEL_PARAM_COUNT = args.model_params
        MODEL_MB = MODEL_PARAM_COUNT * BYTES_PER_PARAM / 1e6
    if args.compare_streaming:
        compare_streaming(args.compare_streaming, args.labels, args.out or "compare_plots", args.target)
        return
    if not args.telemetry_dir:
        parser.error("provide a telemetry_dir or --compare-streaming dirs...")
    analyze(args.telemetry_dir, args.out)


if __name__ == "__main__":
    main()
