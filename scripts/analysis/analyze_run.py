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
        EVENT_UTILITY_BELIEF,
    )
except Exception:  # pragma: no cover
    EVENT_SELECTION = "selection"
    EVENT_AGG_EVAL = "agg_eval"
    EVENT_AGG_ROUND = "agg_round"
    EVENT_TRAINER_ROUND = "trainer_round"
    EVENT_UTIL_DISPARITY = "util_disparity"
    EVENT_AVAIL_CHANGE = "avail_change"
    EVENT_UTILITY_BELIEF = "utility_belief"


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


def _sub(out_root, *cls):
    return os.path.join(out_root, *cls)


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
    """Cumulative comm in MEGABYTES (train=2x model, eval=1x model, per chosen).

    Round 0 is the pre-training selection warmup (the selector retries while
    trainers join — tens of thousands of selection events that are not real
    model dispatches), so it is excluded to avoid a spurious comm spike that
    dwarfs every real round.
    """
    per_round = defaultdict(float)
    for r in by_event(records, EVENT_SELECTION):
        rd = int(r.get("round", 0))
        if rd < 1:
            continue
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
        rd = int(s.get("round", 0))
        if rd < 1:  # skip round-0 selection warmup (see cumulative_comm_by_round)
            continue
        chosen[rd][s.get("task", "train")] += len(s.get("chosen") or [])
    committed = defaultdict(int)  # train updates returned (per round)
    for a in by_event(records, EVENT_AGG_ROUND):
        rd = int(a.get("round", 0))
        if rd < 1:
            continue
        committed[rd] += len(a.get("contributing_trainers") or [])
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


# ── single-pass aggregator-log parser ─────────────────────────────────────
# The aggregator log is multi-GB (millions of lines). Every tagged series below
# used to be parsed in its own full scan (6+ passes). parse_agg_log() reads the
# file ONCE, gating each line on a cheap substring before any regex, and caches
# the result per telemetry_dir. The old parser names are kept as thin accessors
# so callers are unchanged.

_LAG_RE = re.compile(r"\[SEND_RECV_LAG\] end=(\S+) version=(\d+) wall_lag_s=([0-9.]+)")
_OVERRUN_EXCESS_RE = re.compile(r"\[TIMING_OVERRUN_AGG\].*excess=([0-9.]+)s")
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
_SIM_BARRIER_ROUND_RE = re.compile(r"\[SIM_BARRIER\].*?round=(\d+)")
_SIM_BARRIER_WAIT_RE = re.compile(r"\[SIM_BARRIER\].*?barrier_wait_s=([0-9.]+)")
_AGG_CACHE_RE = re.compile(r"\[AGG_COMMIT_TIMING\].*?cache_store_s=([0-9.]+)")
_AGG_OPT_RE = re.compile(r"\[AGG_COMMIT_TIMING\].*?optimizer_s=([0-9.]+)")
_SEND_RE = re.compile(r"sending weights to \S+ (?:with )?model_version[=:] ?(\d+)")
# Dispatch count from the INFO roll-up (per-send line is DEBUG); SEND_TIMEOUT = lost model.
_DISTRIBUTE_TIMING_RE = re.compile(r"\[DISTRIBUTE_TIMING\] round=(\d+) n_sends=(\d+)")
_SEND_TIMEOUT_RE = re.compile(r"Removing end \S+ from self\.all_selected since")

_LAG_DECOMP_KEYS = ("wall_lag_s", "agg_to_trainer_s", "compute_s",
                    "post_wait_s", "mqtt_lag_s", "queue_wait_s", "process_s")

_AGG_LOG_CACHE: dict[str, dict] = {}


def parse_agg_log(telemetry_dir: str) -> dict:
    """Single-pass scan of the aggregator log; returns every tagged series.

    Cached per telemetry_dir. Keys:
      wall_lags            list[float]  ([SEND_RECV_LAG] wall_lag_s)
      lag_by_round         {round: [wall_lag_s]}  (version == round)
      overrun_count        int
      overrun_excesses     list[float]
      lag_decomp           {component: [float]}   (6 components, '-' dropped)
      sim_barrier_waits    list[float]
      sim_barrier_by_round {round: [wait_s]}
      agg_cache_store      list[float]
      agg_optimizer        list[float]
      sends_by_round       {round: count}  (weight dispatches)
    """
    run_dir = os.path.dirname(os.path.abspath(telemetry_dir))
    if run_dir in _AGG_LOG_CACHE:
        return _AGG_LOG_CACHE[run_dir]
    out = {
        "wall_lags": [], "lag_by_round": defaultdict(list),
        "overrun_count": 0, "overrun_excesses": [],
        "lag_decomp": {k: [] for k in _LAG_DECOMP_KEYS},
        "sim_barrier_waits": [], "sim_barrier_by_round": defaultdict(list),
        "agg_cache_store": [], "agg_optimizer": [],
        "sends_by_round": defaultdict(int),
        "dispatched_by_round": defaultdict(int),
        "send_timeouts": 0,
    }
    logs = glob.glob(os.path.join(run_dir, "*aggregator*.log"))
    if not logs:
        _AGG_LOG_CACHE[run_dir] = out
        return out
    with open(logs[0]) as fh:
        for line in fh:
            # Cheap gate: dispatch lines carry "sending weights"; everything else
            # we parse is bracket-tagged. Skip the rest before any regex.
            tagged = "[" in line
            if (not tagged and "sending weights" not in line
                    and "from self.all_selected since" not in line):
                continue
            if "[SEND_RECV_LAG]" in line:
                m = _LAG_RE.search(line)
                if m:
                    out["wall_lags"].append(float(m.group(3)))
                    out["lag_by_round"][int(m.group(2))].append(float(m.group(3)))
            elif "[LAG_DECOMP]" in line:
                m = _LAG_DECOMP_RE.search(line)
                if m:
                    for i, k in enumerate(_LAG_DECOMP_KEYS):
                        v = m.group(i + 3)
                        if v != "-":
                            out["lag_decomp"][k].append(float(v))
            elif "[TIMING_OVERRUN_AGG]" in line:
                out["overrun_count"] += 1
                m = _OVERRUN_EXCESS_RE.search(line)
                if m:
                    out["overrun_excesses"].append(float(m.group(1)))
            elif "[SIM_BARRIER]" in line:
                mw = _SIM_BARRIER_WAIT_RE.search(line)
                if mw:
                    w = float(mw.group(1))
                    out["sim_barrier_waits"].append(w)
                    mr = _SIM_BARRIER_ROUND_RE.search(line)
                    if mr:
                        out["sim_barrier_by_round"][int(mr.group(1))].append(w)
            elif "[AGG_COMMIT_TIMING]" in line:
                m1 = _AGG_CACHE_RE.search(line)
                if m1:
                    out["agg_cache_store"].append(float(m1.group(1)))
                m2 = _AGG_OPT_RE.search(line)
                if m2:
                    out["agg_optimizer"].append(float(m2.group(1)))
            elif "[DISTRIBUTE_TIMING]" in line:
                m = _DISTRIBUTE_TIMING_RE.search(line)
                if m:
                    out["dispatched_by_round"][int(m.group(1))] += int(m.group(2))
            elif "sending weights" in line:
                m = _SEND_RE.search(line)
                if m:
                    out["sends_by_round"][int(m.group(1))] += 1
            elif "from self.all_selected since" in line:
                out["send_timeouts"] += 1
    _AGG_LOG_CACHE[run_dir] = out
    return out


def parse_send_recv_lags(telemetry_dir: str) -> tuple[list[float], int]:
    """(wall_lags, overrun_count) — accessor over parse_agg_log."""
    a = parse_agg_log(telemetry_dir)
    return a["wall_lags"], a["overrun_count"]


def _parse_lag_decomp(telemetry_dir: str) -> dict[str, list[float]]:
    """[LAG_DECOMP] 6-component breakdown — accessor over parse_agg_log."""
    return parse_agg_log(telemetry_dir)["lag_decomp"]


def _parse_overrun_excesses(telemetry_dir: str) -> list[float]:
    """[TIMING_OVERRUN_AGG] excess (wall_lag - budget) — accessor."""
    return parse_agg_log(telemetry_dir)["overrun_excesses"]


def _parse_sim_barrier(telemetry_dir: str):
    """(waits, by_round) for [SIM_BARRIER] — accessor over parse_agg_log."""
    a = parse_agg_log(telemetry_dir)
    return a["sim_barrier_waits"], a["sim_barrier_by_round"]


def _parse_agg_commit_timing(telemetry_dir: str):
    """(cache_store, optimizer) for [AGG_COMMIT_TIMING] — accessor."""
    a = parse_agg_log(telemetry_dir)
    return a["agg_cache_store"], a["agg_optimizer"]


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
            # Plot accuracy AS A FUNCTION OF data unlocked: sort points by the
            # x value (visible fraction) so the line is a proper relationship
            # curve, not connected in round order. Under streaming, visible
            # fraction rises monotonically with time, so this is essentially
            # accuracy-over-time re-indexed by data availability; sorting just
            # guarantees a left-to-right curve even where the per-round mean
            # fraction plateaus or dips slightly.
            pts = sorted((unlocked[r], acc[r]) for r in rs)
            xs_u = [x for x, _ in pts]
            ys_a = [y for _, y in pts]
            p = ph.line_plot({"accuracy": (xs_u, ys_a)},
                             "mean visible fraction (data unlocked), sorted ascending",
                             "test accuracy",
                             "Accuracy vs amount of data unlocked", d,
                             "accuracy_vs_data_unlocked.pdf", stamp=stamp)
            if p: saved.append(p)
    # global weight-change norm from checkpoints
    saved += _weight_change_norm(tdir, d, stamp, acc)
    return saved


def _weight_change_norm(tdir, d, stamp, acc=None):
    """L2 step of the global model per checkpoint, ||w_r − w_{r-1}||.

    Expected to DECAY toward 0 as the model converges; a flat/rising tail flags
    non-convergence or staleness re-injecting old gradients. We overlay test
    accuracy (nearest eval per checkpoint round) and mark the convergence round
    where the step norm first drops below 10% of its peak, turning a bare curve
    into a convergence diagnostic.
    """
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
    # convergence marker: first round where the step drops below 10% of peak
    peak = max(norms) if norms else 0.0
    conv_round = next((r for r, nrm in zip(rounds, norms) if nrm < 0.1 * peak), None)
    if acc:
        # nearest-eval accuracy per checkpoint round → secondary axis
        acc_rounds = sorted(acc)
        def _nearest_acc(r):
            rr = min(acc_rounds, key=lambda x: abs(x - r))
            return acc[rr]
        title = "Global model change per checkpoint + accuracy (step should decay)"
        if conv_round is not None:
            title += f" — conv@{conv_round}"
        p = ph.dual_axis_line(rounds, norms, [_nearest_acc(r) for r in rounds],
                              "round", "‖w_r − w_(r-1)‖ (L2)", "test accuracy",
                              title, d, "global_weight_change_norm.pdf", stamp=stamp)
    else:
        title = "Global model change per checkpoint (step should decay to 0)"
        if conv_round is not None:
            title += f" — conv@{conv_round}"
        p = ph.line_plot({"‖w_r − w_(r-1)‖": (rounds, norms)}, "round",
                         "global weight-change L2", title, d,
                         "global_weight_change_norm.pdf", stamp=stamp, logy=True)
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
        # Was a scatter of tens of thousands of (reported, observed) pairs. Bin by
        # trainer-reported x and plot the P99 observed per bin (the tail the user
        # cares about — worst-case aggregator turnaround at a given response time).
        p = ph.binned_line({"agg-observed P99": (xs, ys)},
                           "trainer-reported response (s)",
                           "aggregator-observed (s)",
                           "Aggregator-observed vs trainer-reported (P99/bin)", d,
                           "runtime_agg_vs_trainer.pdf", stamp=stamp,
                           nbins=50, reducer="p99")
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

    # Believed-vs-actual client utility from LIVE telemetry (utility_belief event),
    # available for EVERY baseline: believed = what the selector held at selection
    # (PROP_STAT_UTILITY, stale); actual = the fresh stat-utility the client reports
    # on return. The gap is the selector's belief staleness — small for an eval-
    # refreshed selector (felix), larger for stale-utility baselines (oort/refl/
    # feddance). This is the live counterpart to the oracle plot above; for the
    # oracle replay see oracle_utility.py / oracle_misselection.py.
    ub = by_event(records, EVENT_UTILITY_BELIEF)
    lb, la, lgap, lrounds = [], [], [], []
    for r in ub:
        b, a = r.get("believed"), r.get("actual")
        if b is None or a is None:
            continue
        try:
            b = float(b); a = float(a)
        except (TypeError, ValueError):
            continue
        lb.append(b); la.append(a); lgap.append(abs(b - a))
        lrounds.append((int(r.get("round", 0)), abs(b - a)))
    if lb:
        p = ph.cdf_multi({"believed (at selection)": lb, "actual (at return)": la},
                         "client statistical utility",
                         "Selected utility: believed vs actual (live)", d,
                         "selected_utility_believed_vs_actual_cdf.pdf", stamp=stamp)
        if p: saved.append(p)
        p = ph.cdf_plot(lgap, "|believed - actual| utility (belief staleness error)",
                        "Selector belief staleness CDF", d,
                        "selected_utility_belief_gap_cdf.pdf", stamp=stamp)
        if p: saved.append(p)
        p = ph.scatter_diag(lb, la, "believed utility (at selection)",
                            "actual utility (at return)",
                            "Client utility: believed vs actual (live)", d,
                            "selected_utility_believed_vs_actual.pdf", stamp=stamp)
        if p: saved.append(p)
        # mean gap binned over rounds (does belief staleness grow over the run?)
        gap_by_round = {}
        for rd, g in lrounds:
            gap_by_round.setdefault(rd, []).append(g)
        gx = sorted(gap_by_round)
        gy = [sum(gap_by_round[r]) / len(gap_by_round[r]) for r in gx]
        p = ph.line_plot({"mean |believed - actual|": (gx, gy)}, "round",
                         "belief staleness error (utility)",
                         "Belief staleness over the run (live)", d,
                         "selected_utility_belief_gap_over_rounds.pdf", stamp=stamp)
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
        # Drop round 0 (pre-training selection warmup: tens of thousands of
        # retry events that otherwise dominate the y-axis).
        if s.get("task", "train") == "train" and int(s.get("round", 0)) >= 1:
            sel_by_round[int(s.get("round", 0))] += len(s.get("chosen") or [])
    # Sum across agg_round events per round: async emits one event per commit,
    # so a dict-comprehension would overwrite and show 1 instead of agg_goal.
    contrib = defaultdict(int)
    for r in ar:
        rd = int(r.get("round", 0))
        if rd >= 1:
            contrib[rd] += len(r.get("contributing_trainers") or [])
    if sel_by_round:
        rr = sorted(set(sel_by_round) | set(contrib))
        # Binned (mean/bin) so the chosen-vs-contributing relationship is smooth
        # over thousands of rounds instead of a jagged per-round line.
        series = {"chosen (train)": (rr, [sel_by_round.get(r, 0) for r in rr])}
        if contrib:
            series["contributing"] = (rr, [contrib.get(r, 0) for r in rr])
        p = ph.binned_line(series, "round", "trainer count",
                           "Selection/aggregation count consistency (round-0 warmup excluded)",
                           d, "selection_count_consistency.pdf", stamp=stamp,
                           nbins=200, reducer="mean")
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

    # Total trainer population (for coverage % and Lorenz padding): every
    # trainer that ever reported a round or an availability change, unioned with
    # everyone ever selected.
    population = ({str(r.get("end_id")) for r in by_event(records, EVENT_TRAINER_ROUND)}
                 | {str(r.get("end_id")) for r in by_event(records, EVENT_AVAIL_CHANGE)})
    population.discard("None")
    population.discard("")

    # Selection frequency per trainer, counting BOTH train AND eval selections
    # (eval participation is still participation — the fairness metric must include
    # it). freq_all = train+eval; freq_train = train only, so the Lorenz can show
    # how much eval participation evens out the distribution. The old
    # selection_coverage dual-axis curve is dropped: the Lorenz below is the single
    # coverage/fairness figure.
    freq_all, freq_train = Counter(), Counter()
    for s in sel:
        is_train = s.get("task", "train") == "train"
        for c in (s.get("chosen") or []):
            freq_all[str(c)] += 1
            if is_train:
                freq_train[str(c)] += 1
    population |= set(freq_all)
    if freq_all:
        # Pad to the FULL population so never-selected trainers count as zeros —
        # otherwise n is just the count of ever-selected trainers and the
        # Lorenz/Gini understate real inequality.
        n_pop = len(population) or 1
        ever_train = len([t for t in population if freq_train.get(t, 0)])
        ever_any = len([t for t in population if freq_all.get(t, 0)])
        series = {
            f"train+eval ({ever_any}/{n_pop} ever picked)":
                [freq_all.get(t, 0) for t in population],
            f"train only ({ever_train}/{n_pop} ever trained)":
                [freq_train.get(t, 0) for t in population],
        }
        p = ph.lorenz_plot(series,
                           f"Selection fairness (Lorenz; n={n_pop} population, "
                           f"train+eval vs train-only; lower Gini = more equal)", d,
                           "selection_fairness_lorenz.pdf", stamp=stamp)
        if p: saved.append(p)

    # exploration vs exploitation over rounds. The exploration_factor (epsilon)
    # decays to ~0 within tens/hundreds of rounds, so the run-length x-axis hides
    # the action. Clip x to just past where it first reaches ~0, and plot BOTH the
    # explore fraction and its exploit complement (= 1 - explore). This subsumes
    # the old selection_coverage curve, which is now dropped in favor of the
    # single Lorenz fairness figure below.
    ef = sorted({(int(s.get("round", 0)), float(s.get("exploration_factor")))
                 for s in sel if s.get("exploration_factor") is not None})
    if ef:
        rs = [r for r, _ in ef]; ev = [v for _, v in ef]
        # first round where exploration has effectively decayed to 0 (+10% margin)
        eps = 1e-3
        zero_idx = next((i for i, v in enumerate(ev) if v <= eps), len(ev) - 1)
        cut = rs[min(len(rs) - 1, int(zero_idx * 1.1) + 1)]
        rs_c = [r for r in rs if r <= cut]; ev_c = ev[:len(rs_c)]
        p = ph.line_plot(
            {"exploration": (rs_c, ev_c),
             "exploitation (1−explore)": (rs_c, [1.0 - v for v in ev_c])},
            "round", "fraction",
            f"Exploration vs exploitation (x clipped at decay→0, round {cut})", d,
            "exploration_factor_over_rounds.pdf", stamp=stamp)
        if p: saved.append(p)

    # Eval selections: felix DOES run an eval selector, but most eval-selection
    # events carry chosen=[] and eval bursts are sparse vs train rounds, so a
    # full-width stacked area renders them invisible. Plot the eval selection RATE
    # as its own smoothed line (binned mean of eval-chosen per round) so the eval
    # selector is visibly active; train is on a second binned line for context.
    et = defaultdict(lambda: {"train": 0, "eval": 0})
    for s in sel:
        et[int(s.get("round", 0))][s.get("task", "train")] += len(s.get("chosen") or [])
    if any(v["eval"] for v in et.values()):
        rr = sorted(r for r in et if r >= 1)
        p = ph.binned_line(
            {"eval selections/round": (rr, [et[r]["eval"] for r in rr]),
             "train selections/round": (rr, [et[r]["train"] for r in rr])},
            "round", "selections", "Eval vs train selection rate (mean/bin)",
            d, "eval_vs_train_selections.pdf", stamp=stamp, nbins=150, reducer="mean")
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
    def _smooth(ys, w=15):
        """Centered rolling mean so the noisy per-round trace is legible."""
        import numpy as np
        a = np.asarray(ys, float)
        if len(a) < 3:
            return a
        w = min(w, len(a) if len(a) % 2 else len(a) - 1)
        w = max(3, w | 1)  # odd window
        kern = np.ones(w) / w
        return np.convolve(a, kern, mode="same")

    both = sorted(set(spd) & set(utl))
    if both:
        p = ph.dual_axis_line(both, _smooth([spd[r] for r in both]),
                              _smooth([utl[r] for r in both]),
                              "round", "avg speed of picked (s, smoothed)",
                              "avg believed utility of picked (smoothed)",
                              "Picked clients: avg speed & utility per round "
                              "(rolling mean, w=15)", d,
                              "selected_speed_utility_over_rounds.pdf", stamp=stamp)
        if p: saved.append(p)
        # CDF equivalent: distribution of the per-round averages (far easier to
        # read than the raw jagged time series).
        p = ph.cdf_multi({"avg speed of picked (s)": [spd[r] for r in both],
                          "avg believed utility of picked": [utl[r] for r in both]},
                         "per-round average value",
                         "Picked clients: per-round avg speed & utility (CDF)", d,
                         "selected_speed_utility_cdf.pdf", stamp=stamp)
        if p: saved.append(p)
    elif utl:  # e.g. FedDance has no speed factor
        rr = sorted(utl)
        p = ph.line_plot({"avg believed utility of picked": (rr, _smooth([utl[r] for r in rr]))},
                         "round", "avg believed utility of picked (smoothed)",
                         "Picked clients: avg utility per round (rolling mean, w=15)", d,
                         "selected_speed_utility_over_rounds.pdf", stamp=stamp)
        if p: saved.append(p)
        p = ph.cdf_plot([utl[r] for r in rr], "per-round avg believed utility",
                        "Picked clients: per-round avg utility (CDF)", d,
                        "selected_speed_utility_cdf.pdf", stamp=stamp)
        if p: saved.append(p)

    # #14: separate CDFs of the picked clients' believed SPEED and believed
    # UTILITY (over all picks), plus expected-vs-actual utility — the believed
    # value at selection vs the trainer's actual stat_utility that round.
    all_sp, all_bel = [], []
    exp_act = {"believed (at selection)": [], "actual (trainer stat_utility)": []}
    n_picks = 0; n_with_bel = 0  # believed_I coverage (often null → gappy plots)
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
            n_picks += 1
            info = pt.get(str(c)) or {}
            if info.get("speed_s") is not None:
                all_sp.append(float(info["speed_s"]))
            bel = info.get("believed_I", info.get("utility"))
            if bel is not None:
                n_with_bel += 1
                all_bel.append(float(bel))
                act = actual_util.get((rd, str(c)[-3:]))
                if act is not None:
                    exp_act["believed (at selection)"].append(float(bel))
                    exp_act["actual (trainer stat_utility)"].append(act)
    cov = f"believed_I on {n_with_bel}/{n_picks} picks" if n_picks else "no picks"
    cdfs = {}
    if all_sp:
        cdfs["speed (s)"] = all_sp
    if cdfs:
        p = ph.cdf_multi(cdfs, "value", "Picked-client speed distribution (CDF)", d,
                         "selected_speed_cdf.pdf", stamp=stamp)
        if p: saved.append(p)
    if all_bel:
        p = ph.cdf_multi({"believed utility": all_bel}, "utility",
                         f"Picked-client utility distribution (CDF; {cov})", d,
                         "selected_utility_cdf.pdf", stamp=stamp)
        if p: saved.append(p)
    if exp_act["actual (trainer stat_utility)"]:
        p = ph.cdf_multi(exp_act, "utility",
                         f"Picked-client utility: believed vs actual ({cov})", d,
                         "selected_utility_expected_vs_actual_cdf.pdf", stamp=stamp)
        if p: saved.append(p)
        # Signed believed − actual gap: the SIGN is the cross-baseline story
        # (refl under-estimates → negative mean; felix over-estimates → positive).
        gap = [b - a for b, a in zip(exp_act["believed (at selection)"],
                                     exp_act["actual (trainer stat_utility)"])]
        import numpy as _np
        mg = float(_np.mean(gap)) if gap else 0.0
        sign = "over-estimates" if mg > 0 else "under-estimates"
        p = ph.hist_plot(gap, "believed − actual utility",
                         f"Belief error of selector (mean={mg:+.3f} → {sign}; <0 conservative)",
                         d, "selected_utility_belief_gap_hist.pdf", stamp=stamp, vline=0.0)
        if p: saved.append(p)

    # participation heatmap (trainer x round: 0 idle, 1 eval-selected, 2 trained)
    saved += _participation_heatmap(records, d, stamp)
    # per-trainer & aggregate state-fraction plots (available / train / eval /
    # idle / unavailable)
    saved += _state_fraction_plots(records, d, stamp)

    # availability composition: one LINE per availability state (was a stacked
    # area that, under static availability like syn_0, renders as one solid block).
    # Lines make a flat single-state trace honest and a dynamic trace readable.
    per_round = {}
    for r in sel:
        comp = r.get("avail_composition")
        if comp:
            per_round[int(r.get("round", 0))] = comp
    if per_round:
        rr = sorted(per_round)
        states = sorted({s for c in per_round.values() for s in c})
        note = " (static: one state — see availability/ for dynamics)" \
            if len(states) == 1 else ""
        series = {s: (rr, [per_round[r].get(s, 0) for r in rr]) for s in states}
        p = ph.line_plot(series, "round", "trainer count",
                         f"Availability composition over rounds{note}", d,
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
                   discrete=[(0, "not selected", "#ededed"),
                             (1, "eval", "#3182bd"),
                             (2, "train", "#31a354"),
                             (3, "unavailable", "#fdae6b"),
                             (4, "selected but unavail", "#e6550d")])
    return [p] if p else []


def _state_fraction_plots(records, d, stamp):
    """Per-trainer and aggregate fraction-of-time in each ACTIVITY.

    Renamed from "state" (state changes constantly; this is a time-allocation
    breakdown). Exclusive per (trainer, round) activities:
      train, eval, idle_train (avail-for-train, not picked),
      idle_eval (avail-for-eval, not picked), unavail.
    'available' fraction = train+eval+idle_train+idle_eval. Produces:
      • aggregate single stacked bar (population mean) — the headline split;
      • per-trainer horizontal stacked bar (every trainer's split);
      • across-trainer CDFs of fraction-available and fraction-training.
    Files are trainer_time_allocation_* (was trainer_state_fraction_*).
    """
    import numpy as np
    trained = defaultdict(set)
    for r in by_event(records, EVENT_TRAINER_ROUND):
        trained[int(r.get("round", 0))].add(str(r.get("end_id")))
    evalsel = defaultdict(set)
    for s in by_event(records, EVENT_SELECTION):
        if s.get("task") == "eval":
            for c in (s.get("chosen") or []):
                evalsel[int(s.get("round", 0))].add(str(c))
    # availability forward-fill: track the full state string so idle can be split
    # into idle_train (AVL_TRAIN) vs idle_eval (AVL_EVAL). UN_AVL = unavailable.
    ac = defaultdict(list)
    for r in by_event(records, EVENT_AVAIL_CHANGE):
        ac[str(r.get("end_id"))].append((int(r.get("round", 0)),
                                         str(r.get("new_state", ""))))
    trainers = sorted({t for s in trained.values() for t in s}
                      | {t for s in evalsel.values() for t in s}
                      | set(ac))
    rounds = sorted(set(trained) | set(evalsel))
    rounds = [r for r in rounds if r >= 1]
    if not trainers or not rounds:
        return []
    # per-trainer exclusive counts
    cats = ["train", "eval", "idle_train", "idle_eval", "unavail"]
    counts = {t: {c: 0 for c in cats} for t in trainers}
    avail_count = {t: 0 for t in trainers}
    for t in trainers:
        evs = sorted(ac.get(t, []))
        ei, state = 0, ""
        for r in rounds:
            while ei < len(evs) and evs[ei][0] <= r:
                state = evs[ei][1]; ei += 1
            un = "UN_AVL" in state
            if not un:
                avail_count[t] += 1
            if t in trained.get(r, set()):
                counts[t]["train"] += 1
            elif t in evalsel.get(r, set()):
                counts[t]["eval"] += 1
            elif un:
                counts[t]["unavail"] += 1
            elif "AVL_EVAL" in state:
                counts[t]["idle_eval"] += 1
            else:  # AVL_TRAIN (or unknown/default) and not picked
                counts[t]["idle_train"] += 1
    R = len(rounds)
    saved = []
    seg_colors = {"train": "#31a354", "eval": "#3182bd",
                  "idle_train": "#bdbdbd", "idle_eval": "#9e9ac8",
                  "unavail": "#e6550d"}
    # Drop activities that never occur (e.g. idle_eval under static availability)
    # so the legend/colors stay uncrowded.
    cats = [c for c in cats if any(counts[t][c] for t in trainers)]
    # 1) aggregate population-mean split (single stacked bar, % annotated)
    agg = {c: [float(np.mean([counts[t][c] / R for t in trainers]))] for c in cats}
    p = ph.stacked_bar(["population mean"], agg, "fraction of rounds",
                       "Trainer time allocation — population mean (per round)", d,
                       "trainer_time_allocation_aggregate.pdf", stamp=stamp,
                       annotate=True, colors=seg_colors)
    if p: saved.append(p)
    # 2) per-trainer split (horizontal stacked bar; sorted by training fraction;
    #    no per-segment text to avoid crowding 300 bars)
    order = sorted(trainers, key=lambda t: counts[t]["train"] / R, reverse=True)
    yl = [t[-3:] for t in order]
    segs = {c: [counts[t][c] / R for t in order] for c in cats}
    p = ph.stacked_bar(yl, segs, "fraction of rounds",
                       "Trainer time allocation — per trainer (sorted by train fraction)",
                       d, "trainer_time_allocation_per_trainer.pdf", stamp=stamp,
                       horizontal=True, colors=seg_colors)
    if p: saved.append(p)
    # 3) across-trainer CDFs: fraction-available & fraction-training
    p = ph.cdf_multi({"fraction available": [avail_count[t] / R for t in trainers],
                      "fraction training": [counts[t]["train"] / R for t in trainers]},
                     "fraction of rounds (per trainer)",
                     "Across-trainer distribution of availability & training", d,
                     "trainer_time_allocation_cdf.pdf", stamp=stamp)
    if p: saved.append(p)
    return saved


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
    # data-unlock effects (vs visible fraction). These were dense scatters over
    # every trainer-round; bin over the x (visible fraction) into a mean trend
    # line + P10–P90 band so the relationship is legible (and renders fast).
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
            p = ph.binned_line({ylab: ([a for a, _ in xy], [b for _, b in xy])},
                               "visible fraction", ylab, title, d, fname,
                               stamp=stamp, nbins=60, reducer="mean", band=True)
            if p: saved.append(p)
    # util disparity streamed vs full, CONNECTED to selection impact. The ratio
    # = utility_streamed/utility_full: under streaming a trainer's believed utility
    # (on the visible prefix) differs from its full-data utility; ratio<1 means the
    # selector under-values not-yet-unlocked trainers. To make it actionable we
    # overlay the mis-selection rate on a second axis — "when disparity is high, do
    # we mis-select?" — aligning on the rounds both series share.
    ud = by_event(records, EVENT_UTIL_DISPARITY)
    if ud:
        byr = defaultdict(list)
        for r in ud:
            v = r.get("utility_ratio")
            if v is not None:
                byr[int(r.get("round", 0))].append(v)
        rr = sorted(byr)
        if rr:
            ratio = {r: sum(byr[r]) / len(byr[r]) for r in rr}
            miss = {int(float(m["round"])): float(m["misselection_rate"])
                    for m in load_oracle_misselection(tdir)
                    if m.get("task") == "train" and m.get("misselection_rate") not in (None, "")}
            common = [r for r in rr if r in miss]
            if len(common) >= 3:
                p = ph.dual_axis_line(
                    common, [ratio[r] for r in common], [miss[r] for r in common],
                    "round", "streamed / full utility ratio (1=no disparity)",
                    "mis-selection rate",
                    "Utility disparity vs mis-selection (does disparity drive bad picks?)",
                    d, "util_disparity_ratio.pdf", stamp=stamp)
            else:
                p = ph.line_plot({"streamed/full ratio": (rr, [ratio[r] for r in rr])},
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
    times = [r["timestamp_s"] - t0 for r in rows]  # SECONDS since start
    # Note: x-resolution is bounded by the resource sampler interval
    # (execution.monitoring.check_interval_seconds); plotting in seconds just
    # stops the minute-bucketing from hiding within-minute variation.
    sample_dt = (times[1] - times[0]) if len(times) > 1 else 0
    xlab = f"time (s, sampled every ~{sample_dt:.0f}s)" if sample_dt else "time (s)"

    # RAM over time — GB and %
    ram_vals = [r["ram_gb"] for r in rows]
    swap_vals = [r["swap_gb"] for r in rows]
    ram_series = {"RAM used (GB)": (times, ram_vals)}
    if any(s > 0.05 for s in swap_vals):
        ram_series["Swap used (GB)"] = (times, swap_vals)
    p = ph.line_plot(ram_series, xlab, "GB", "RAM usage over time", d,
                     "resource_ram_over_time.pdf", stamp=stamp)
    if p: saved.append(p)
    ram_pct_series = {"RAM used (%)": (times, [r["ram_pct"] for r in rows])}
    if any(r.get("swap_pct", 0) > 0.05 for r in rows):
        ram_pct_series["Swap used (%)"] = (times, [r.get("swap_pct", 0) for r in rows])
    p = ph.line_plot(ram_pct_series, xlab, "% of total",
                     "RAM usage over time (%)", d, "resource_ram_pct_over_time.pdf",
                     stamp=stamp)
    if p: saved.append(p)

    # GPU utilization over time (one series per GPU; already a %)
    gpu_indices = sorted({int(k.split("_")[1]) for k in rows[0] if k.startswith("gpu_") and k.endswith("_util_pct")})
    if gpu_indices:
        util_series = {}
        for n in gpu_indices:
            vals = [r.get(f"gpu_{n}_util_pct", 0.0) for r in rows]
            if any(v > 0 for v in vals):
                util_series[f"GPU{n} util%"] = (times, vals)
        if not util_series:
            util_series["GPU util% (all 0)"] = (times, [0.0] * len(times))
        p = ph.line_plot(util_series, xlab, "util (%)",
                         "GPU utilization over time", d, "resource_gpu_util_over_time.pdf",
                         stamp=stamp)
        if p: saved.append(p)

        # GPU memory over time — GB and %
        mem_series = {f"GPU{n} mem (GB)": (times, [r.get(f"gpu_{n}_mem_gb", 0.0) for r in rows])
                      for n in gpu_indices}
        p = ph.line_plot(mem_series, xlab, "memory (GB)",
                         "GPU memory usage over time", d, "resource_gpu_mem_over_time.pdf",
                         stamp=stamp)
        if p: saved.append(p)
        mem_pct_series = {f"GPU{n} mem%": (times, [r.get(f"gpu_{n}_mem_pct", 0.0) for r in rows])
                          for n in gpu_indices}
        if any(any(v > 0 for v in s[1]) for s in mem_pct_series.values()):
            p = ph.line_plot(mem_pct_series, xlab, "% of total",
                             "GPU memory usage over time (%)", d,
                             "resource_gpu_mem_pct_over_time.pdf", stamp=stamp)
            if p: saved.append(p)

    return saved


# ==========================================================================
# system/
# ==========================================================================


def mqtt_delivery_plots(records, out, stamp, tdir):
    """MQTT drop sanity (both modes): per-round dispatched − received hovers at 0 when
    healthy, sustained-positive on a broker drop. Cross-checked vs SEND_TIMEOUTs (the
    aggregator's own lost-model count). Dispatched from [DISTRIBUTE_TIMING], received
    from task_recv events."""
    import numpy as _np
    d = _sub(out, "system"); saved = []
    agg = parse_agg_log(tdir)
    dispatched_by_round = agg["dispatched_by_round"] or agg["sends_by_round"]
    send_timeouts = agg.get("send_timeouts", 0)
    if not dispatched_by_round:
        return saved  # no agg log → nothing to check
    recvs_by_round: dict[int, int] = defaultdict(int)
    for r in records:
        if r.get("event") == "task_recv":
            rd = r.get("round")
            if rd is not None:
                recvs_by_round[int(rd)] += 1
    rounds = sorted(r for r in (set(dispatched_by_round) | set(recvs_by_round)) if r >= 1)
    cs = cr = 0
    xs, deltas = [], []
    for r in rounds:
        nd = dispatched_by_round.get(r, 0); nr = recvs_by_round.get(r, 0)
        cs += nd; cr += nr
        xs.append(r); deltas.append(nd - nr)
    # Per-round delta differences away the constant in-flight offset, so it hovers at 0;
    # a broker drop shows as a sustained positive run.
    run_pos = 0; max_run = 0  # longest run of positive (undelivered) deltas
    for v in deltas:
        run_pos = run_pos + 1 if v > 0 else 0
        max_run = max(max_run, run_pos)
    p = ph.line_plot(
        {"dispatched − received (per round)": (xs, deltas)},
        "round", "messages dropped over MQTT (per round)",
        f"MQTT drops per round — hovers at 0 = none "
        f"(SEND_TIMEOUTs={send_timeouts}, net={cs - cr}; dispatched={cs} received={cr})",
        d, "mqtt_drops_over_rounds.pdf", stamp=stamp)
    if p: saved.append(p)
    p = ph.cdf_plot(
        deltas, "per-round (dispatched − received)",
        f"MQTT drops CDF — mass at 0 = no drops (SEND_TIMEOUTs={send_timeouts})",
        d, "mqtt_drops_cdf.pdf", stamp=stamp)
    if p: saved.append(p)
    # Definitive drop verdict: SEND_TIMEOUT is unambiguous; a long unbroken run of
    # positive deltas is the heuristic backstop for slow/partial loss.
    if send_timeouts > 0 or max_run >= 10:
        p = ph.no_data_plot(
            f"MQTT message drops suspected: SEND_TIMEOUTs={send_timeouts}, "
            f"longest undelivered run={max_run} rounds",
            d, "mqtt_drops_callout.pdf", stamp=stamp,
            note=f"dispatched={cs} received={cr} net={cs - cr}; a healthy run is flat 0")
        if p: saved.append(p)
    return saved


def sim_speedup_plots(records, out, stamp, tdir):
    """Simulator-only debug plots (real runs get explicit no-data placeholders):
      • sim_speedup_factor_over_time — vclock/wall (1.0 = real-equivalent, >1 faster)
      • sim_barrier_wait_cdf / _over_rounds — the recv barrier wall
      • sim_vclock_advance_decomp_over_rounds — Δvclock vs modeled compute (K3b)
    These are the levers for debugging the speedup fix and the clock-tier parity
    checks (K2/K3/K3b/sim_rate).
    """
    d = _sub(out, "system"); saved = []
    ar = by_event(records, EVENT_AGG_ROUND)
    is_sim = any(r.get("vclock_now") is not None for r in ar)

    def _nd(fn, title, note):
        p = ph.no_data_plot(title, d, fn, stamp=stamp, note=note)
        if p:
            saved.append(p)

    # ── speedup factor over time: vclock / wall ──
    if is_sim:
        pairs = sorted((r["ts"], r["vclock_now"]) for r in ar
                       if r.get("ts") is not None and r.get("vclock_now") is not None)
        if len(pairs) >= 2:
            t0 = pairs[0][0]
            xs, ys = [], []
            for t, v in pairs[1:]:  # skip first (wall≈0 → div blow-up)
                dt = t - t0
                if dt > 0:
                    xs.append(dt); ys.append(v / dt)
            p = ph.line_plot(
                {"sim speedup (vclock/wall)": (xs, ys)}, "wall time (s)",
                "speedup factor (virtual-s / wall-s)",
                "Sim speedup over real (1.0 = real-equivalent; >1 = faster than real)",
                d, "sim_speedup_factor_over_time.pdf", stamp=stamp, target=1.0)
            if p:
                saved.append(p)
    else:
        _nd("sim_speedup_factor_over_time.pdf",
            "Sim speedup over real (real run: N/A)",
            "real mode has no virtual clock; speedup is defined for sim runs only")

    # ── barrier wait: CDF + over rounds ──
    waits, by_round = _parse_sim_barrier(tdir)
    if waits:
        p = ph.cdf_plot(waits, "barrier_wait_s (sim recv set-drain)",
                        f"Sim recv barrier wait CDF (n={len(waits)})", d,
                        "sim_barrier_wait_cdf.pdf", stamp=stamp)
        if p:
            saved.append(p)
        rr = sorted(r for r in by_round if r >= 1)
        if rr:
            _m = lambda xs: sum(xs) / len(xs) if xs else 0.0
            p = ph.line_plot(
                {"mean barrier_wait_s": (rr, [_m(by_round[r]) for r in rr])},
                "round", "barrier_wait_s",
                "Sim recv barrier wait over rounds (target ≈ wall_lag = compute+mqtt)",
                d, "sim_barrier_wait_over_rounds.pdf", stamp=stamp, clip_outliers=True)
            if p:
                saved.append(p)
    else:
        note = "[SIM_BARRIER] is emitted only by the simulated recv path"
        _nd("sim_barrier_wait_cdf.pdf", "Sim recv barrier wait CDF (real run: N/A)", note)
        _nd("sim_barrier_wait_over_rounds.pdf",
            "Sim recv barrier wait over rounds (real run: N/A)", note)

    # ── per-round vclock advance vs modeled compute (debugs K3b overhead) ──
    if is_sim:
        by_v, by_spd = {}, {}
        for r in ar:
            rd = r.get("round")
            if rd is None or rd < 1:
                continue
            if r.get("vclock_now") is not None:
                if rd not in by_v or r.get("ts", 0) > by_v[rd][1]:
                    by_v[rd] = (r["vclock_now"], r.get("ts", 0))
            sp = r.get("trainer_speed_s") or []
            if sp:
                by_spd[rd] = max(by_spd.get(rd, 0.0), max(sp))
        rr = sorted(by_v)
        if len(rr) >= 2:
            xr, adv, comp = [], [], []
            for i in range(1, len(rr)):
                xr.append(rr[i])
                adv.append(max(0.0, by_v[rr[i]][0] - by_v[rr[i - 1]][0]))
                comp.append(by_spd.get(rr[i], float("nan")))
            p = ph.line_plot(
                {"Δvclock/round (actual advance)": (xr, adv),
                 "max committed speed (modeled compute)": (xr, comp)},
                "round", "seconds",
                "Sim per-round vclock advance vs modeled compute "
                "(gap = per-commit overhead + overlap)",
                d, "sim_vclock_advance_decomp_over_rounds.pdf", stamp=stamp,
                clip_outliers=True)
            if p:
                saved.append(p)
            # CDF of per-round vclock advance (the parity-relevant K3 view; renders
            # instantly and reads the distribution shape the line plot hides).
            p = ph.cdf_plot(adv, "Δvclock per round (s)",
                            f"Sim per-round vclock advance CDF (n={len(adv)})", d,
                            "sim_vclock_advance_cdf.pdf", stamp=stamp)
            if p:
                saved.append(p)
    else:
        _nd("sim_vclock_advance_decomp_over_rounds.pdf",
            "Sim per-round vclock advance (real run: N/A)",
            "virtual-clock advance is defined for sim runs only")
    return saved


def system_plots(records, out, stamp, tdir):
    d = _sub(out, "system"); saved = []
    xs, ys = comm_vs_accuracy_series(records)
    if xs:
        # Auto-scale the comm axis to GB once values get large (>1 GB) for
        # readability; otherwise keep MB.
        unit, scale = ("MB", 1.0)
        if xs and max(xs) >= 1000.0:
            unit, scale = ("GB", 1000.0)
        p = ph.line_plot({"accuracy": ([x / scale for x in xs], ys)},
                         f"cumulative comm ({unit})", "test accuracy",
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
        # CUMULATIVE comm by direction (monotone lines read far better than a noisy
        # per-round area over thousands of rounds; the slopes are the per-round rate).
        def _cum(vals):
            out, run = [], 0.0
            for v in vals:
                run += v; out.append(run)
            return out
        p = ph.line_plot({"train down (agg→tr)": (rr, _cum(td)),
                          "train up (tr→agg)": (rr, _cum(tu)),
                          "eval down (agg→tr)": (rr, _cum(ed))},
                         "round", "cumulative comm (MB)",
                         f"Cumulative comm by task+direction "
                         f"(total {tot:.0f} MB: train {sum(td) + sum(tu):.0f}, eval {sum(ed):.0f})",
                         d, "comm_per_round_train_vs_eval.pdf", stamp=stamp)
        if p: saved.append(p)
        # message accounting, CUMULATIVE: returned vs discarded (sent-but-unused).
        # The growing discarded gap is the signal (overcommitment slack/stragglers).
        p = ph.line_plot({"returned (up)": (rr, _cum([cb[r]["msgs_up"] for r in rr])),
                          "discarded (sent, unused)": (rr, _cum([cb[r]["discarded"] for r in rr]))},
                         "round", "cumulative messages",
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
        # CDF companion: the per-trainer-round bars above are unreadable at 300
        # trainers, so also show the distribution of each component across all
        # trainer-rounds (annotated P50/P90/P99).
        gpu_all = [v for c in cats for v in agg[c]["gpu"]]
        sim_all = [max(0.0, s - g) for c in cats
                   for s, g in zip(agg[c]["sim"], agg[c]["gpu"])]
        wait_all = [v for c in cats for v in agg[c]["wait"]]
        p = ph.cdf_multi({"gpu": gpu_all, "sim-delay": sim_all, "wait": wait_all},
                         "seconds / trainer-round",
                         "Trainer time breakdown (CDF across trainer-rounds)", d,
                         "trainer_time_breakdown_cdf.pdf", stamp=stamp)
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
        _comp_keys = ("pre (setup)", "gpu compute", "post (cleanup)", "sleep (budget)")
        _m = lambda xs: sum(xs) / len(xs) if xs else 0.0
        series = {k: [_m(split_rd[r][k]) for r in rr] for k in _comp_keys}
        p = ph.stacked_area(rr, series, "round",
                            "mean seconds / round (across trainers)",
                            "Trainer round-time split (mean across trainers)", d,
                            "trainer_time_split_over_rounds.pdf", stamp=stamp)
        if p: saved.append(p)
        # CDF companion of the over-rounds split: distribution of each component
        # across all trainer-rounds (annotated), readable where the area plot is not.
        allv = defaultdict(list)
        for rd in rr:
            for k in _comp_keys:
                allv[k].extend(split_rd[rd][k])
        p = ph.cdf_multi({k: allv[k] for k in _comp_keys if any(allv[k])},
                         "seconds / trainer-round",
                         "Trainer round-time split (CDF across trainer-rounds)", d,
                         "trainer_time_split_over_rounds_cdf.pdf", stamp=stamp)
        if p: saved.append(p)
        # #19: whole-run overall split as a single stacked bar (the area above,
        # collapsed over all rounds) — one glanceable "where did time go" summary,
        # with exact per-segment seconds annotated.
        overall = {k: [_m(allv[k])] for k in _comp_keys}
        p = ph.stacked_bar(["whole run"], overall, "mean seconds / round",
                           "Trainer round-time split (whole-run mean)", d,
                           "trainer_time_split_overall.pdf", stamp=stamp,
                           annotate=True)
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
    # aggregate per-commit cost: cache store (in-memory) vs optimizer (floor).
    _cs, _opt = _parse_agg_commit_timing(tdir)
    if _cs or _opt:
        series = {}
        if _cs:
            series[f"cache_store_s (n={len(_cs)})"] = _cs
        if _opt:
            series[f"optimizer_s (n={len(_opt)})"] = _opt
        p = ph.cdf_multi(series, "seconds per commit/round",
                         "Aggregate commit cost: cache store vs optimizer", d,
                         "agg_commit_timing_cdf.pdf", stamp=stamp)
        if p: saved.append(p)

    # queue depth: binned mean+P99 band over rounds (was a jagged per-round line)
    # plus a CDF (what fraction of rounds had queue >= k — reads the spike tail).
    inflight = [(int(r.get("round", 0)), r.get("updates_in_queue")) for r in
                by_event(records, EVENT_AGG_ROUND) if r.get("updates_in_queue") is not None]
    if inflight:
        inflight.sort()
        qx = [r for r, _ in inflight]; qy = [v for _, v in inflight]
        p = ph.binned_line({"updates in queue": (qx, qy)}, "round",
                           "updates in queue", "Async queue depth over rounds (P50/bin + band)",
                           d, "queue_depth_over_rounds.pdf", stamp=stamp,
                           nbins=200, reducer="p50", band=True)
        if p: saved.append(p)
        p = ph.cdf_plot(qy, "updates in queue",
                        f"Async queue depth CDF (n={len(qy)} rounds)", d,
                        "queue_depth_cdf.pdf", stamp=stamp)
        if p: saved.append(p)

    # staleness over rounds + CDF (both modes; debugs U3 staleness parity).
    stale_by_round = defaultdict(list); all_stale = []
    for r in by_event(records, EVENT_AGG_ROUND):
        for s in (r.get("staleness") or []):
            if s is not None:
                stale_by_round[int(r.get("round", 0))].append(float(s))
                all_stale.append(float(s))
    if all_stale:
        p = ph.cdf_plot(all_stale, "staleness (rounds behind)",
                        f"Update staleness CDF (n={len(all_stale)})", d,
                        "staleness_cdf.pdf", stamp=stamp)
        if p: saved.append(p)
        rr = sorted(r for r in stale_by_round if r >= 1)
        _m = lambda xs: sum(xs) / len(xs) if xs else 0.0
        p = ph.line_plot(
            {"mean staleness": (rr, [_m(stale_by_round[r]) for r in rr])},
            "round", "staleness (rounds behind)", "Update staleness over rounds",
            d, "staleness_over_rounds.pdf", stamp=stamp, clip_outliers=True)
        if p: saved.append(p)

    # commit-visibility lag (U6): per-update delay between an update becoming
    # ready to aggregate and being committed, in the aggregator's own clock
    # (sim: vclock-sct; real: wall commit-arrival). Sim past-dating shows up as a
    # large/growing lag here BEFORE it propagates into staleness; the over-rounds
    # series exposes the felix clock-jump pattern a single CDF would smear.
    # Split train vs eval (different lines): eval reusing a stale train sct shows
    # up as an eval-only lag blowup the combined series would smear. For baselines
    # that dispatch no eval (e.g. sync oort) the eval series is simply absent.
    vis_by_round = defaultdict(lambda: defaultdict(list)); all_vis = defaultdict(list)
    for r in by_event(records, EVENT_AGG_ROUND):
        v = r.get("update_visibility_lag_s")
        if v is None:
            continue
        v = v if isinstance(v, list) else [v]
        task = str(r.get("task_to_perform", "train"))
        for x in v:
            if x is not None:
                vis_by_round[task][int(r.get("round", 0))].append(float(x))
                all_vis[task].append(float(x))
    if any(all_vis.values()):
        series_cdf = {f"{t} (n={len(xs)}, mean={sum(xs) / len(xs):.2f}s)": sorted(xs)
                      for t, xs in sorted(all_vis.items()) if xs}
        p = ph.cdf_multi(series_cdf, "commit visibility lag (s, own clock)",
                         "Commit-visibility lag by task (train vs eval)", d,
                         "commit_visibility_lag_cdf.pdf", stamp=stamp)
        if p: saved.append(p)
        series_line = {}
        for t, rmap in sorted(vis_by_round.items()):
            vr = sorted(rr for rr in rmap if rr >= 1)
            if vr:
                series_line[f"P50 {t} lag"] = (vr, [_m(rmap[rr]) for rr in vr])
        if series_line:
            p = ph.binned_line(
                series_line, "round", "commit visibility lag (s)",
                "Commit-visibility lag over rounds by task (P50/bin)",
                d, "commit_visibility_lag_over_rounds.pdf", stamp=stamp,
                nbins=200, reducer="p50", band=True)
            if p: saved.append(p)

    # Send-recv lag over rounds (binned). The round is the model `version`
    # stamped on each [SEND_RECV_LAG] line (version=N == round N), parsed once in
    # parse_agg_log. Instrumented in BOTH sync and async aggregators, so this
    # plot exists for all baselines (placeholder when no events). The upward
    # drift seen on felix is the buffer-backup symptom: wall_lag is
    # dominated by queue_wait ≈ staleness × wall/round, which rises as the reorder
    # buffer backs up. We overlay queue_wait (from [LAG_DECOMP]) so the riser is
    # visibly that component, not an independent regression.
    _agg = parse_agg_log(tdir)
    wall_lags, _overruns = _agg["wall_lags"], _agg["overrun_count"]
    rd_lags = _agg["lag_by_round"]
    lag_raw = {}  # series -> (xs, ys) raw per-update, binned by binned_line
    if rd_lags:
        rx, ry = [], []
        for r in sorted(rd_lags):
            if r >= 1:
                for v in rd_lags[r]:
                    rx.append(r); ry.append(v)
        if rx:
            lag_raw["wall_lag_s (send→recv)"] = (rx, ry)
    # queue_wait component (same x-base would need per-round; approximate by
    # spreading the decomp queue_wait over the same round span is not exact, so we
    # only overlay wall_lag here and leave the component split to the LAG_DECOMP
    # CDFs below + the per-round decomposition in the aggregation deep-dive).
    p = ph.binned_line(
        lag_raw, "round", "wall_lag_s",
        f"Send-recv lag over rounds — P50/bin "
        f"(n={len(wall_lags)}, overruns={_overruns}; rise=queue_wait/staleness)",
        d, "send_recv_lag_over_rounds.pdf", stamp=stamp, reducer="p50", band=True)
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
# availability/  — deep-dive (lights up under a dynamic availability trace;
# gated to skip cleanly under static availability like syn_0)
# ==========================================================================


def availability_plots(records, out, stamp, tdir):
    d = _sub(out, "availability"); saved = []
    sel = by_event(records, EVENT_SELECTION)
    ac = by_event(records, EVENT_AVAIL_CHANGE)

    # 1) candidates → eligible → chosen funnel (binned over rounds). Always
    # meaningful: shows where the population is lost between availability and pick.
    fx, cand, elig, chos = [], [], [], []
    for s in sel:
        if s.get("task", "train") != "train":
            continue
        rd = int(s.get("round", 0))
        if rd < 1:
            continue
        nc, ne, nch = s.get("num_candidates"), s.get("num_eligible"), s.get("num_chosen")
        if nc is None:
            continue
        fx.append(rd); cand.append(nc)
        elig.append(ne if ne is not None else 0)
        chos.append(nch if nch is not None else 0)
    if fx:
        p = ph.binned_line(
            {"candidates": (fx, cand), "eligible": (fx, elig), "chosen": (fx, chos)},
            "round", "trainer count",
            "Availability→selection funnel (candidates→eligible→chosen, mean/bin)",
            d, "selection_funnel_over_rounds.pdf", stamp=stamp, nbins=150, reducer="mean")
        if p: saved.append(p)

    # Determine if availability is DYNAMIC. syn_0 emits one AVL_TRAIN per trainer
    # at startup and never changes → duty-cycle / churn plots are degenerate.
    by_trainer = defaultdict(list)
    states_seen = set()
    for r in ac:
        by_trainer[str(r.get("end_id"))].append((int(r.get("round", 0)),
                                                  str(r.get("new_state", ""))))
        states_seen.add(str(r.get("new_state", "")))
    has_unavail = any("UN_AVL" in s for s in states_seen)
    dynamic = has_unavail or any(len(v) > 1 for v in by_trainer.values())
    if not dynamic:
        p = ph.no_data_plot(
            "Availability is static (no UN_AVL transitions)",
            d, "availability_dynamics.pdf", stamp=stamp,
            note=f"{len(by_trainer)} trainers, states={sorted(states_seen)} — "
                 f"duty-cycle/churn need a dynamic trace (non-syn_0)")
        if p: saved.append(p)
        return saved

    # 2) per-trainer duty cycle (fraction of the run available). Forward-fill the
    # state across the round span; available = not UN_AVL.
    rounds = sorted({rd for evs in by_trainer.values() for rd, _ in evs})
    rmax = max(rounds) if rounds else 0
    duty = []
    churn_by_round = defaultdict(int)
    for t, evs in by_trainer.items():
        evs = sorted(evs)
        for rd, _ in evs:
            churn_by_round[rd] += 1
        # integrate availability over [0, rmax]
        avail_rounds, cur_r, cur_un = 0, 0, False
        for rd, st in evs:
            if not cur_un:
                avail_rounds += max(0, rd - cur_r)
            cur_r, cur_un = rd, ("UN_AVL" in st)
        if not cur_un:
            avail_rounds += max(0, rmax - cur_r)
        duty.append(avail_rounds / rmax if rmax else 0.0)
    p = ph.cdf_plot(duty, "fraction of run available (per trainer)",
                    f"Per-trainer duty-cycle CDF (n={len(duty)})", d,
                    "duty_cycle_cdf.pdf", stamp=stamp)
    if p: saved.append(p)

    # 3) availability churn rate over rounds (avail_change events / round-bin)
    cr = sorted(churn_by_round)
    p = ph.binned_line({"avail_change events": (cr, [churn_by_round[r] for r in cr])},
                       "round", "transitions", "Availability churn rate (events/bin)",
                       d, "availability_churn_over_rounds.pdf", stamp=stamp,
                       nbins=150, reducer="sum")
    if p: saved.append(p)
    return saved


# ==========================================================================
# selection/why/  — deep-dive: which factor drove selection?
# ==========================================================================


def selection_why_plots(records, out, stamp, tdir):
    d = _sub(out, "selection", "why"); saved = []
    sel = by_event(records, EVENT_SELECTION)

    # Factor fields carried in per_trainer (felix-style selectors). For each, we
    # separate the value distribution among PICKED vs NOT-PICKED trainers: a wide
    # horizontal gap means that factor drove selection; overlap means it didn't.
    factor_keys = ("believed_I", "system_util", "temporal", "speed_s")
    picked = {k: [] for k in factor_keys}
    pool = {k: [] for k in factor_keys}
    CAP = 200_000  # bound memory/time on the 150k-event per_trainer stream
    # utility percentile bands over rounds (believed_I among picked)
    util_x, util_y = [], []
    for s in sel:
        if s.get("task", "train") != "train":
            continue
        pt = s.get("per_trainer") or {}
        rd = int(s.get("round", 0))
        for tid, info in pt.items():
            is_sel = bool(info.get("selected"))
            for k in factor_keys:
                v = info.get(k)
                if v is None:
                    continue
                bucket = picked[k] if is_sel else pool[k]
                if len(bucket) < CAP:
                    bucket.append(float(v))
            if is_sel:
                bi = info.get("believed_I", info.get("utility"))
                if bi is not None and rd >= 1:
                    util_x.append(rd); util_y.append(float(bi))
    any_factor = False
    for k in factor_keys:
        if picked[k] and pool[k]:
            any_factor = True
            p = ph.cdf_multi({f"picked (n={len(picked[k])})": picked[k],
                              f"not picked (n={len(pool[k])})": pool[k]},
                             k, f"Selection driver: {k} — picked vs pool "
                             f"(wide gap = drove selection)", d,
                             f"factor_separation_{k}.pdf", stamp=stamp)
            if p: saved.append(p)
    if not any_factor:
        p = ph.no_data_plot(
            "No per-trainer factor fields (believed_I/system_util/temporal/speed_s)",
            d, "factor_separation.pdf", stamp=stamp,
            note="this selector does not log per-trainer factors (e.g. refl/oort)")
        if p: saved.append(p)

    # utility percentile bands over round-bins: p10/p50/p90 of believed utility
    # of picked clients — the across-population utility trajectory in one figure.
    if util_x:
        saved += _utility_bands(util_x, util_y, d, stamp)
    return saved


def _utility_bands(xs, ys, d, stamp, nbins=150):
    import numpy as np
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    if xs.size < 3:
        return []
    edges = np.linspace(xs.min(), xs.max(), nbins + 1)
    idx = np.clip(np.digitize(xs, edges) - 1, 0, nbins - 1)
    cx, p10, p50, p90 = [], [], [], []
    for b in range(nbins):
        sel = ys[idx == b]
        if sel.size == 0:
            continue
        cx.append(0.5 * (edges[b] + edges[b + 1]))
        p10.append(float(np.quantile(sel, 0.1)))
        p50.append(float(np.quantile(sel, 0.5)))
        p90.append(float(np.quantile(sel, 0.9)))
    if not cx:
        return []
    p = ph.banded_line(cx, p50, p10, p90, "round",
                       "believed utility (picked)",
                       "Picked-client believed-utility p10/p50/p90 over rounds",
                       d, "picked_utility_bands_over_rounds.pdf", stamp=stamp)
    return [p] if p else []


# ==========================================================================
# aggregation/  — deep-dive: rate / cadence / staleness / buffer health
# ==========================================================================


def aggregation_plots(records, out, stamp, tdir):
    d = _sub(out, "aggregation"); saved = []
    ar = by_event(records, EVENT_AGG_ROUND)
    if not ar:
        return saved

    # 1) commit cadence: commits per round-bin (async emits one agg_round per
    # commit). Shows how fast updates are landing over the run.
    commits_by_round = defaultdict(int)
    for r in ar:
        rd = int(r.get("round", 0))
        if rd >= 1:
            commits_by_round[rd] += 1
    cr = sorted(commits_by_round)
    if cr:
        p = ph.binned_line({"commits/round": (cr, [commits_by_round[r] for r in cr])},
                           "round", "commits", "Commit cadence (commits/round, mean/bin)",
                           d, "commit_cadence_over_rounds.pdf", stamp=stamp,
                           nbins=150, reducer="mean")
        if p: saved.append(p)

    # 2) staleness vs trainer-speed (per commit): slow trainers should be the
    # stale ones — confirms the staleness mechanism. Both are single-element lists.
    sp_x, st_y = [], []
    for r in ar:
        st = r.get("staleness") or []; sp = r.get("trainer_speed_s") or []
        for a, b in zip(sp, st):
            if a is not None and b is not None:
                sp_x.append(float(a)); st_y.append(float(b))
    if sp_x:
        p = ph.binned_line({"staleness vs speed": (sp_x, st_y)},
                           "trainer speed (s)", "staleness (rounds behind)",
                           "Staleness vs trainer speed (slow→stale? mean/bin)", d,
                           "staleness_vs_speed.pdf", stamp=stamp, nbins=60, reducer="mean")
        if p: saved.append(p)

    # 3) reorder-buffer health: commit_gap_s (vclock − sct; >0 = buffer backed up)
    # and buf_depth over round-bins. The direct visual for the felix overhead bug
    # A rising commit_gap_s = updates committing long after completion.
    # Split commit_gap_s train vs eval (eval past-dating from a stale sct is the
    # bug this catches); buf_depth is buffer-global so it stays a single line.
    gap_by_task = defaultdict(lambda: ([], []))  # task -> (rounds, gaps)
    dx, depth_v = [], []
    for r in ar:
        rd = int(r.get("round", 0))
        if rd < 1:
            continue
        if r.get("commit_gap_s") is not None:
            t = str(r.get("task_to_perform", "train"))
            gx_t, gv_t = gap_by_task[t]
            gx_t.append(rd); gv_t.append(float(r["commit_gap_s"]))
        if r.get("buf_depth") is not None:
            dx.append(rd); depth_v.append(float(r["buf_depth"]))
    if any(gx for gx, _ in gap_by_task.values()):
        line_series = {f"commit_gap_s {t}": (gx, gv)
                       for t, (gx, gv) in sorted(gap_by_task.items()) if gx}
        if dx:
            line_series["buf_depth"] = (dx, depth_v)
        p = ph.binned_line(line_series,
                           "round", "value", "Reorder-buffer health by task "
                           "(commit_gap_s>0 & rising = backup)", d,
                           "buffer_health_over_rounds.pdf", stamp=stamp,
                           nbins=150, reducer="p50")
        if p: saved.append(p)
        cdf_series = {f"{t} (n={len(gv)})": sorted(gv)
                      for t, (gx, gv) in sorted(gap_by_task.items()) if gv}
        p = ph.cdf_multi(cdf_series, "commit_gap_s (vclock − sct)",
                        "Commit-gap CDF by task (0 = no buffer backup)", d,
                        "commit_gap_cdf.pdf", stamp=stamp)
        if p: saved.append(p)

    # 4) update residence time: rounds an update waited in the buffer before
    # committing — ties staleness to the buffer mechanic.
    resid = [int(r["residence_rounds"]) for r in ar
             if r.get("residence_rounds") is not None]
    if resid:
        p = ph.cdf_plot(resid, "residence (rounds in buffer)",
                        f"Update residence-time CDF (n={len(resid)})", d,
                        "residence_rounds_cdf.pdf", stamp=stamp)
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
    for fn in (perf_plots, sanity_plots, selection_plots, insights_plots,
               system_plots, sim_speedup_plots, mqtt_delivery_plots,
               availability_plots, selection_why_plots, aggregation_plots):
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
