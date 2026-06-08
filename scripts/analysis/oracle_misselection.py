#!/usr/bin/env python
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Offline oracle: ground-truth utility + mis-selection metrics for a FLAME run.

Selectors like OORT/REFL rank clients by a *stale* statistical utility (computed
the last time a client trained, on the data unlocked back then). Under streaming
data each client's true utility drifts away from that belief. This script
reconstructs the **true current utility** of every candidate trainer at each
saved checkpoint -- under the *current* global model, on the *currently-unlocked*
prefix of that trainer's deterministic data stream -- and joins it with the
selector's logged *believed* utility (``EVENT_SELECTION.per_trainer``) to quantify
mis-selection.

Everything is reconstructed deterministically from the run's artifacts, so the
oracle is identical across baselines (fair apples-to-apples):
  * per-trainer data partition  -> dataset split YAML (Dirichlet)
  * per-trainer arrival order    -> randperm seeded by sha256(task_id) (main.py)
  * visible prefix at round R     -> floor(min(1, sim_now/full_after_s) * total)
  * statistical utility           -> N * sqrt(mean(per_sample_loss^2))  (Oort)

Requires per-round model checkpoints (``<run>/checkpoints/round_*.pt``), produced
by the aggregator when ``checkpoint.enabled`` is set.

Usage
-----
    python oracle_misselection.py --run-dir <experiment_run_dir> [options]

Outputs (under ``<run>/analysis/``):
    oracle_utility.csv          per (round, trainer): believed/true/true_full, selected, visible_fraction
    oracle_misselection.csv     per round: gap, top-k overlap, regret, mean-true-utility-of-selected
    oracle_summary.json         run-level rollup
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
from collections import defaultdict
from typing import Optional

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- model: must mirror the trainer's Net exactly so state_dict keys match -----
# (lib/python/examples/async_cifar10/trainer/pytorch/main.py:Net)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
EVENT_SELECTION = "selection"


# --- config resolution ------------------------------------------------------


def _load_yaml(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def resolve_config(run_dir: str, args) -> dict:
    """Resolve dataset/alpha/streaming/metadata from the run artifacts + CLI."""
    exec_cfg_path = os.path.join(run_dir, "execution_config.yaml")
    snap_path = os.path.join(run_dir, "snapshot.yaml")
    exec_cfg = _load_yaml(exec_cfg_path) if os.path.exists(exec_cfg_path) else {}
    snap = _load_yaml(snap_path) if os.path.exists(snap_path) else {}

    exp = exec_cfg.get("experiment", {}) or snap.get("experiment", {})
    tr = exp.get("trainer", {})
    ds = tr.get("dataset", {})

    dataset = args.dataset or ds.get("name", "cifar10")
    alpha = args.alpha if args.alpha is not None else (
        ds.get("alpha", ds.get("dirichlet_alpha"))
    )
    num_trainers = args.num_trainers or tr.get("num_trainers")

    # metadata dir: snapshot records absolute location; else fall back to repo
    meta_dir = args.metadata_dir or snap.get("metadata_location")
    if not meta_dir:
        # repo default
        here = os.path.dirname(os.path.abspath(__file__))
        meta_dir = os.path.join(
            here, "..", "..", "lib", "python", "examples", "_metadata"
        )
    meta_dir = os.path.abspath(meta_dir)

    # streaming horizon: CLI wins, else dig hyperparameters out of the configs
    full_after_s = args.full_after_s
    if full_after_s is None:
        full_after_s = _dig_full_after_s(tr) or 0.0

    # sample size for utility (parity with the live counterfactual)
    sample_size = args.sample_size
    if sample_size is None:
        sample_size = _dig_sample_size(tr) or 256

    return {
        "dataset": dataset,
        "alpha": alpha,
        "num_trainers": int(num_trainers) if num_trainers else None,
        "metadata_dir": meta_dir,
        "full_after_s": float(full_after_s),
        "sample_size": int(sample_size) if sample_size else None,
        "split_key": (exec_cfg.get("metadata_refs", {}) or {}).get("dataset_split_key"),
    }


def _dig_full_after_s(trainer_cfg: dict) -> Optional[float]:
    hp = trainer_cfg.get("hyperparameters") or {}
    ds = hp.get("data_streaming") or {}
    if str(ds.get("enabled", "False")) != "True":
        return 0.0
    v = ds.get("full_data_available_after_s")
    return float(v) if v is not None else None


def _dig_sample_size(trainer_cfg: dict) -> Optional[int]:
    hp = trainer_cfg.get("hyperparameters") or {}
    uc = hp.get("util_counterfactual") or {}
    v = uc.get("sample_size")
    return int(v) if v not in (None, "None", "") else None


# --- trainer table: end_id(task_id) -> indices + seed -----------------------


def build_trainer_table(cfg: dict) -> dict:
    """Map telemetry end_id (== task_id) -> {indices, seed, trainer_key}.

    The shuffle seed mirrors main.py: sha256(str(task_id)) % 2**31. The data
    partition comes from the Dirichlet split keyed by trainer_NNN.
    """
    meta = cfg["metadata_dir"]
    registry = _load_yaml(os.path.join(meta, "trainer_registry.yaml"))["trainers"]

    # Resolve the split file: exact key, then n<num_trainers>, then n300.
    # Small runs (e.g. n10 smoke) reuse the first N trainers of the n300 split.
    ds_dir = os.path.join(meta, "dataset_splits")
    candidates = []
    if cfg.get("split_key"):
        candidates.append(f"{cfg['split_key']}.yaml")
    candidates.append(f"{cfg['dataset']}_alpha{cfg['alpha']}_n{cfg['num_trainers']}.yaml")
    candidates.append(f"{cfg['dataset']}_alpha{cfg['alpha']}_n300.yaml")
    split_path = next(
        (os.path.join(ds_dir, c) for c in candidates
         if os.path.exists(os.path.join(ds_dir, c))),
        None,
    )
    if split_path is None:
        raise SystemExit(
            f"no dataset split file found in {ds_dir} (tried {candidates})"
        )
    print(f"using dataset split: {os.path.basename(split_path)}")
    splits = _load_yaml(split_path)["trainer_data_splits"]

    table: dict[str, dict] = {}
    for trainer_key, info in registry.items():
        task_id = str(info["task_id"])
        if trainer_key not in splits:
            continue
        indices = list(splits[trainer_key])
        seed = int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % (2 ** 31)
        table[task_id] = {
            "trainer_key": trainer_key,
            "indices": indices,
            "seed": seed,
        }
    return table


# --- streaming + utility (mirror main.py) -----------------------------------


def visible_count(sim_now: float, full_after_s: float, total: int) -> int:
    """Replicate main.py:_visible_sample_count."""
    if full_after_s <= 0:
        return total
    frac = min(1.0, sim_now / full_after_s)
    n = math.floor(frac * total)
    return min(total, max(1, n))


def oort_utility(model, data, targets, norm_n, device, sample_size=None):
    """N * sqrt(mean(loss^2)); mirrors main.py:_oort_utility."""
    u, _ = oort_utility_acc(model, data, targets, norm_n, device, sample_size)
    return u


def oort_utility_acc(model, data, targets, norm_n, device, sample_size=None):
    """Return (Oort utility, top-1 accuracy) from one forward pass.

    Utility = N*sqrt(mean(loss^2)) (mirrors main.py:_oort_utility); accuracy is
    the local top-1 used as FedDance's true A_m source (slope across checkpoints).
    """
    n = data.shape[0]
    if n == 0:
        return 0.0, 0.0
    if sample_size is not None and n > sample_size:
        sel = torch.randperm(n)[:sample_size]
        data = data[sel]
        targets = targets[sel]
    criterion = nn.CrossEntropyLoss(reduction="none")
    model.eval()
    with torch.no_grad():
        tg = targets.to(device)
        out = model(data.to(device))
        per_sample = criterion(out, tg)
        sumsq = torch.square(per_sample).sum().item()
        acc = (out.argmax(dim=-1) == tg).float().mean().item()
    n_used = data.shape[0]
    util = norm_n * math.sqrt(sumsq / n_used) if n_used > 0 else 0.0
    return util, acc


# --- telemetry loading ------------------------------------------------------


def load_selection_events(telemetry_dir: str) -> list[dict]:
    rows: list[dict] = []
    for path in glob.glob(os.path.join(telemetry_dir, "*.jsonl")):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("event") == EVENT_SELECTION:
                    rows.append(r)
    return rows


def estimate_full_after_s(telemetry_dir: str) -> float:
    """Infer the streaming horizon from util_disparity telemetry (self-calibrate).

    visible = floor(min(1, elapsed/T) * total)  =>  T ~= elapsed * total / visible
    for events where 0 < visible < total. Returns 0.0 if no streaming is observed
    (all visible == total). Same sim clock as the checkpoints' sim_time_s.
    """
    ests = []
    saw_partial = False
    for path in glob.glob(os.path.join(telemetry_dir, "*.jsonl")):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("event") != "util_disparity":
                    continue
                vis = r.get("visible_samples")
                tot = r.get("total_samples")
                el = r.get("elapsed_s")
                if not vis or not tot or el is None:
                    continue
                if vis < tot and vis > 0 and el > 0:
                    saw_partial = True
                    ests.append(el * tot / vis)
    if not saw_partial or not ests:
        return 0.0
    ests.sort()
    return ests[len(ests) // 2]  # median


def load_checkpoints(ckpt_dir: str, every: int, limit: Optional[int]) -> list[dict]:
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "round_*.pt")))
    metas = []
    for i, p in enumerate(paths):
        if every > 1 and (i % every != 0):
            continue
        metas.append(p)
    if limit:
        metas = metas[:limit]
    return metas


# --- data pool: full CIFAR-10 train, normalized, on CPU ---------------------


def build_image_pool(data_root: str):
    """Return (images[N,3,32,32] float normalized, targets[N]) for CIFAR-10 train.

    Deterministic eval-style transform (ToTensor + Normalize, no augmentation):
    the oracle measures data-intrinsic utility under the current model, so the
    ground truth is reproducible. (The live believed-utility used train-time
    augmentation; absolute scales differ slightly but rankings/gaps are stable.)
    """
    from torchvision.datasets import CIFAR10

    ds = CIFAR10(data_root, train=True, download=True)
    imgs = torch.from_numpy(ds.data).float().div_(255.0)  # [N,32,32,3]
    imgs = imgs.permute(0, 3, 1, 2).contiguous()  # [N,3,32,32]
    mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
    imgs = (imgs - mean) / std
    targets = torch.tensor(ds.targets, dtype=torch.long)
    return imgs, targets


# --- main -------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", help="output dir (default <run>/analysis)")
    ap.add_argument("--metadata-dir")
    ap.add_argument("--dataset")
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--num-trainers", type=int)
    ap.add_argument("--full-after-s", type=float,
                    help="streaming horizon (sim-sec); 0 = no streaming")
    ap.add_argument("--sample-size", type=int,
                    help="per-trainer utility subsample (default from config/256)")
    ap.add_argument("--data-root",
                    default="/home/dgarg39/flame/lib/python/examples/async_cifar10/data")
    ap.add_argument("--every-n-checkpoints", type=int, default=1,
                    help="use every Nth checkpoint to bound compute")
    ap.add_argument("--max-checkpoints", type=int)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    telemetry_dir = os.path.join(run_dir, "telemetry")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    out_dir = args.out or os.path.join(run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(ckpt_dir):
        raise SystemExit(
            f"no checkpoints/ in {run_dir}; rerun with checkpoint.enabled=True"
        )

    cfg = resolve_config(run_dir, args)
    # Self-calibrate the streaming horizon from telemetry when not given/found.
    if args.full_after_s is None and not cfg["full_after_s"]:
        est = estimate_full_after_s(telemetry_dir)
        if est:
            cfg["full_after_s"] = est
            print(f"estimated full_after_s from util_disparity telemetry: {est:.1f}")
    print("resolved config:", json.dumps(cfg, indent=2))
    table = build_trainer_table(cfg)
    print(f"trainer table: {len(table)} trainers (registry x split)")

    selections = load_selection_events(telemetry_dir)

    # Restrict the oracle to trainers that actually participated in THIS run
    # (keeps compute bounded for small runs; full n300 runs use all of them).
    participants = set()
    for sel in selections:
        participants.update(str(t) for t in (sel.get("per_trainer") or {}).keys())
        participants.update(str(t) for t in (sel.get("chosen") or []))
    for path in glob.glob(os.path.join(telemetry_dir, "trainer_*.jsonl")):
        eid = os.path.basename(path)[len("trainer_"):-len(".jsonl")]
        participants.add(eid)
    if participants:
        table = {t: v for t, v in table.items() if t in participants}
        print(f"restricted to {len(table)} participating trainers")
    # believed utility + selected set per (round, task) -> use train-task rows for
    # the headline selection-quality metrics (train is what consumes utility).
    print(f"selection events: {len(selections)}")

    ckpts = load_checkpoints(ckpt_dir, args.every_n_checkpoints, args.max_checkpoints)
    print(f"checkpoints to evaluate: {len(ckpts)}")
    if not ckpts:
        raise SystemExit("no checkpoints matched")

    device = torch.device(args.device)
    imgs, targets = build_image_pool(args.data_root)

    # Pre-shuffle each trainer's order once (deterministic).
    for tid, info in table.items():
        order = torch.randperm(
            len(info["indices"]),
            generator=torch.Generator().manual_seed(info["seed"]),
        )
        # global sample indices in arrival order
        gidx = torch.tensor(info["indices"], dtype=torch.long)[order]
        info["arrival_global_idx"] = gidx
        info["total"] = len(info["indices"])

    full_after_s = cfg["full_after_s"]
    sample_size = cfg["sample_size"]

    # true utility per (round, trainer)
    # round -> {end_id: {"true": .., "true_full": .., "visible_fraction": ..}}
    true_by_round: dict[int, dict[str, dict]] = {}
    for path in ckpts:
        blob = torch.load(path, map_location="cpu")
        rnd = int(blob["round"])
        sim_now = float(blob.get("sim_time_s", 0.0))
        model = Net().to(device)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        per_t: dict[str, dict] = {}
        for tid, info in table.items():
            total = info["total"]
            vis_n = visible_count(sim_now, full_after_s, total)
            gidx = info["arrival_global_idx"]
            vis_g = gidx[:vis_n]
            u_stream, acc_stream = oort_utility_acc(
                model, imgs[vis_g], targets[vis_g], norm_n=vis_n,
                device=device, sample_size=sample_size,
            )
            u_full = oort_utility(
                model, imgs[gidx], targets[gidx], norm_n=total,
                device=device, sample_size=sample_size,
            )
            per_t[tid] = {
                "true": u_stream,
                "true_full": u_full,
                "acc": acc_stream,
                "visible_fraction": (vis_n / total) if total else None,
            }
        true_by_round[rnd] = per_t
        print(f"  round {rnd}: sim_now={sim_now:.1f} evaluated {len(per_t)} trainers")

    avail_rounds = sorted(true_by_round.keys())

    def nearest_oracle_round(r: int) -> int:
        # match a selection round to the closest checkpoint round <= r (else min)
        le = [x for x in avail_rounds if x <= r]
        return max(le) if le else min(avail_rounds)

    # true A_m (FedDance accuracy-increment) = local-accuracy slope between
    # consecutive checkpoints, per trainer. orr -> {end_id: slope}.
    true_A_by_round: dict[int, dict[str, float]] = {}
    for i, rnd in enumerate(avail_rounds):
        prev = avail_rounds[i - 1] if i > 0 else None
        per_t = {}
        for tid, d in true_by_round[rnd].items():
            a_now = d.get("acc", 0.0)
            if prev is not None and tid in true_by_round[prev]:
                a_prev = true_by_round[prev][tid].get("acc", a_now)
                gap = max(1, rnd - prev)
            else:
                a_prev, gap = a_now, 1
            per_t[tid] = (a_now - a_prev) / gap
        true_A_by_round[rnd] = per_t

    # --- join with selections + write per (round,trainer) rows ---------------
    util_rows = []  # round, end_id, believed, true, true_full, selected, visible_fraction, task
    per_round_metrics = []  # round, task, k, gap, topk_overlap, regret, mean_true_selected
    cf_metrics = []  # per-selector counterfactual replay (believed-vs-true-factor top-k)

    for sel in selections:
        rnd = int(sel.get("round", 0))
        task = sel.get("task", "train")
        chosen = set(str(c) for c in (sel.get("chosen") or []))
        per_trainer = sel.get("per_trainer") or {}
        if not per_trainer:
            continue
        orr = nearest_oracle_round(rnd)
        truth = true_by_round.get(orr, {})

        # candidate pool = trainers the selector actually scored this round
        candidates = [str(t) for t in per_trainer.keys() if str(t) in truth]
        if not candidates:
            continue

        believed = {str(t): (per_trainer[t].get("utility")) for t in per_trainer}

        for t in candidates:
            util_rows.append({
                "round": rnd,
                "task": task,
                "end_id": t,
                "believed": believed.get(t),
                "true": truth[t]["true"],
                "true_full": truth[t]["true_full"],
                "selected": int(t in chosen),
                "visible_fraction": truth[t]["visible_fraction"],
            })

        sel_in_pool = [t for t in candidates if t in chosen]
        k = len(sel_in_pool)
        if k == 0:
            continue
        # oracle top-k by true utility within the candidate pool
        ranked = sorted(candidates, key=lambda t: truth[t]["true"], reverse=True)
        oracle_topk = set(ranked[:k])
        overlap = len(set(sel_in_pool) & oracle_topk) / k
        mean_true_selected = sum(truth[t]["true"] for t in sel_in_pool) / k
        mean_true_oracle = sum(truth[t]["true"] for t in oracle_topk) / k
        regret = mean_true_oracle - mean_true_selected
        # believed-minus-true gap on selected set (where believed known)
        gaps = [
            (believed[t] - truth[t]["true"])
            for t in sel_in_pool
            if believed.get(t) is not None
        ]
        gap = (sum(gaps) / len(gaps)) if gaps else None

        per_round_metrics.append({
            "round": rnd,
            "task": task,
            "k": k,
            "believed_minus_true_gap": gap,
            "topk_overlap": overlap,
            "misselection_rate": 1.0 - overlap,
            "utility_regret": regret,
            "mean_true_selected": mean_true_selected,
            "mean_true_oracle_topk": mean_true_oracle,
        })

        # --- counterfactual replay (self-relative): re-run THIS selector's
        # scoring formula with believed factors vs. true factors substituted
        # (I_m for all; A_m also for FedDance), top-k each, measure the gap. This
        # is the staleness penalty *within* the selector's own algorithm.
        selector = str(sel.get("selector", ""))
        is_feddance = "feddance" in selector.lower()
        bel_score, tru_score = {}, {}
        for t in candidates:
            pt = per_trainer.get(t) or {}
            true_I = truth[t]["true"]
            if is_feddance:
                V = pt.get("feddance_V"); I = pt.get("feddance_I")
                A = pt.get("feddance_A"); U = pt.get("feddance_U")
                if None in (V, I, A, U) or I == 0 or A == 0:
                    continue
                true_A = (true_A_by_round.get(orr, {}) or {}).get(t, A)
                bel_score[t] = U
                # U = V*I*A*MAB -> substitute I->true_I, A->true_A via ratios
                tru_score[t] = U * (true_I / I) * (true_A / A if A else 1.0)
            else:  # oort / refl / felix family: (I + temporal) * system_util
                bI = pt.get("believed_I")
                temporal = pt.get("temporal")
                su = pt.get("system_util")
                if bI is None or temporal is None or su is None:
                    continue
                bel_score[t] = (bI + temporal) * su
                tru_score[t] = (true_I + temporal) * su  # speed ~identity here
        kk = min(k, len(bel_score))
        if kk >= 1:
            bel_topk = set(sorted(bel_score, key=bel_score.get, reverse=True)[:kk])
            tru_topk = set(sorted(tru_score, key=tru_score.get, reverse=True)[:kk])
            cf_overlap = len(bel_topk & tru_topk) / kk
            # Regret in the selector's OWN true-score units (>=0): true-score the
            # selector forfeited by ranking with stale factors instead of true.
            cf_regret = (
                sum(tru_score[t] for t in tru_topk) / kk
                - sum(tru_score[t] for t in bel_topk) / kk
            )
            cf_metrics.append({
                "round": rnd,
                "selector": selector,
                "k": kk,
                "cf_misselection_rate": 1.0 - cf_overlap,
                "cf_topk_overlap": cf_overlap,
                "cf_utility_regret": cf_regret,
            })

    # --- write outputs -------------------------------------------------------
    _write_csv(os.path.join(out_dir, "oracle_utility.csv"), util_rows,
               ["round", "task", "end_id", "believed", "true", "true_full",
                "selected", "visible_fraction"])
    _write_csv(os.path.join(out_dir, "oracle_misselection.csv"), per_round_metrics,
               ["round", "task", "k", "believed_minus_true_gap", "topk_overlap",
                "misselection_rate", "utility_regret", "mean_true_selected",
                "mean_true_oracle_topk"])
    _write_csv(os.path.join(out_dir, "oracle_counterfactual.csv"), cf_metrics,
               ["round", "selector", "k", "cf_misselection_rate",
                "cf_topk_overlap", "cf_utility_regret"])

    train_metrics = [m for m in per_round_metrics if m["task"] == "train"]
    summary = {
        "run_dir": run_dir,
        "config": cfg,
        "n_checkpoints": len(ckpts),
        "n_selection_rounds_scored": len(per_round_metrics),
        "mean_misselection_rate_train": _safe_mean(
            [m["misselection_rate"] for m in train_metrics]),
        "mean_utility_regret_train": _safe_mean(
            [m["utility_regret"] for m in train_metrics]),
        "mean_true_selected_train": _safe_mean(
            [m["mean_true_selected"] for m in train_metrics]),
        # self-relative counterfactual (this selector with stale vs true factors)
        "mean_cf_misselection_rate": _safe_mean(
            [m["cf_misselection_rate"] for m in cf_metrics]),
        "mean_cf_utility_regret": _safe_mean(
            [m["cf_utility_regret"] for m in cf_metrics]),
        "n_cf_rounds": len(cf_metrics),
    }
    with open(os.path.join(out_dir, "oracle_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("wrote:")
    for f in ("oracle_utility.csv", "oracle_misselection.csv", "oracle_summary.json"):
        print("  ", os.path.join(out_dir, f))
    print("summary:", json.dumps(summary, indent=2))


def _safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def _write_csv(path, rows, header):
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in header})


if __name__ == "__main__":
    main()
