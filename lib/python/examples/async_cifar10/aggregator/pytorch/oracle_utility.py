# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Online oracle utility injection (per-baseline oracle).

Turns ANY baseline into "that baseline fed true utilities at each selection step":
before each training selection, the aggregator overwrites every candidate's
``PROP_STAT_UTILITY`` (and, for FedDance, ``PROP_LOCAL_ACCURACY``) with its TRUE
current value -- computed centrally from the current global model on the
candidate's currently-unlocked data prefix. The selector (OORT/REFL/FedDance/Felix)
then ranks oracularly with ZERO selector changes, and the run diverges into the
counterfactual trajectory B' (vs the stale-utility run B).

It is the online twin of scripts/analysis/oracle_misselection.py: same data
partition (Dirichlet split), same deterministic arrival order (sha256(task_id)),
same streaming schedule (uniform or staggered), same utility
``I_m = N*sqrt(mean(loss^2))``. Keep the formulas in sync with that script and with
trainer/pytorch/main.py:_stagger_params.

Gated by hyperparameters.oracle_utility_injection.enabled == "True". The aggregator
is *allowed* to reconstruct trainer data here precisely because it is an oracle /
upper-bound, not a deployable policy.
"""
from __future__ import annotations

import glob
import hashlib
import logging
import math
import os

import torch
import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
PROP_STAT_UTILITY = "stat_utility"
PROP_LOCAL_ACCURACY = "local_accuracy"


# --- formulas (mirror oracle_misselection.py / trainer main.py) -------------

def _stagger_params(trainer_id, onset_max_s, base_span_s, rate_jitter):
    h = hashlib.sha256(f"{trainer_id}:stagger".encode()).hexdigest()
    u1 = int(h[0:8], 16) / 0xFFFFFFFF
    u2 = int(h[8:16], 16) / 0xFFFFFFFF
    onset_s = onset_max_s * u1
    span_s = base_span_s * (1.0 + rate_jitter * (2.0 * u2 - 1.0))
    return onset_s, max(base_span_s / 4.0, span_s)


def _visible_count(sim_now, onset_s, span_s, total, min_visible=1):
    if span_s <= 0:
        return total
    frac = min(1.0, max(0.0, (sim_now - onset_s) / span_s))
    return min(total, max(min_visible, math.floor(frac * total)))


def _oort_utility_acc(model, data, targets, norm_n, device, sample_size=None):
    n = data.shape[0]
    if n == 0:
        return 0.0, 0.0
    if sample_size is not None and n > sample_size:
        sel = torch.randperm(n)[:sample_size]
        data, targets = data[sel], targets[sel]
    crit = nn.CrossEntropyLoss(reduction="none")
    model.eval()
    with torch.no_grad():
        tg = targets.to(device)
        out = model(data.to(device))
        per = crit(out, tg)
        sumsq = torch.square(per).sum().item()
        acc = (out.argmax(dim=-1) == tg).float().mean().item()
    nu = data.shape[0]
    return (norm_n * math.sqrt(sumsq / nu) if nu else 0.0), acc


class OracleUtilityProvider:
    """Lazily builds the trainer->data table + CIFAR pool, then injects per round."""

    def __init__(self, config, data_root):
        hp = getattr(config, "hyperparameters", None)
        oi = (getattr(hp, "oracle_utility_injection", None) or {}) if hp else {}
        self.enabled = str(oi.get("enabled", "False")) == "True"
        self.sample_size = int(oi.get("sample_size", 256) or 256)
        self.inject_accuracy = str(oi.get("inject_accuracy", "False")) == "True"
        # alpha / num_trainers select the Dirichlet split file (the aggregator
        # config doesn't otherwise carry them); set by the experiment generator.
        self.alpha = oi.get("alpha", 0.1)
        self.num_trainers = int(oi.get("num_trainers", 50) or 50)
        ds = (getattr(hp, "data_streaming", None) or {}) if hp else {}
        self.full_after_s = float(ds.get("full_data_available_after_s", 0) or 0)
        stg = ds.get("stagger") or {}
        self.stg_on = str(stg.get("enabled", "False")) == "True" and self.full_after_s > 0
        self.onset_max_s = float(stg.get("onset_max_s", 0.0) or 0.0)
        self.rate_jitter = float(stg.get("rate_jitter", 0.0) or 0.0)
        self.min_visible = int(stg.get("min_visible", 1) or 1)
        self.data_root = data_root
        self._table = None        # task_id -> {arrival_global_idx, total, onset_s, span_s}
        self._imgs = self._targets = None
        if self.enabled:
            logger.info(
                f"[ORACLE_INJECT] enabled (full_after_s={self.full_after_s}, "
                f"stagger={'on' if self.stg_on else 'off'}, "
                f"inject_accuracy={self.inject_accuracy})"
            )

    # --- lazy setup ---------------------------------------------------------
    def _metadata_dir(self):
        td = os.environ.get("FLAME_TELEMETRY_DIR")
        run_dir = os.path.dirname(td.rstrip("/")) if td else None
        snap = os.path.join(run_dir, "snapshot.yaml") if run_dir else None
        if snap and os.path.exists(snap):
            loc = (yaml.safe_load(open(snap)) or {}).get("metadata_location")
            if loc and os.path.isdir(loc):
                return loc
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(here, "..", "..", "..", "_metadata"))

    def _build_table(self, alpha, num_trainers, dataset="cifar10"):
        meta = self._metadata_dir()
        registry = yaml.safe_load(open(os.path.join(meta, "trainer_registry.yaml")))["trainers"]
        ds_dir = os.path.join(meta, "dataset_splits")
        for cand in (f"{dataset}_alpha{alpha}_n{num_trainers}.yaml",
                     f"{dataset}_alpha{alpha}_n300.yaml"):
            p = os.path.join(ds_dir, cand)
            if os.path.exists(p):
                splits = yaml.safe_load(open(p))["trainer_data_splits"]
                break
        else:
            raise FileNotFoundError(f"no dataset split in {ds_dir} for alpha={alpha}")
        table = {}
        for tkey, info in registry.items():
            if tkey not in splits:
                continue
            tid = str(info["task_id"])
            idx = list(splits[tkey])
            seed = int(hashlib.sha256(tid.encode()).hexdigest(), 16) % (2 ** 31)
            order = torch.randperm(len(idx), generator=torch.Generator().manual_seed(seed))
            gidx = torch.tensor(idx, dtype=torch.long)[order]
            if self.stg_on:
                onset, span = _stagger_params(tid, self.onset_max_s,
                                              self.full_after_s, self.rate_jitter)
            else:
                onset, span = 0.0, self.full_after_s
            table[tid] = {"arrival_global_idx": gidx, "total": len(idx),
                          "onset_s": onset, "span_s": span}
        return table

    def _build_pool(self):
        from torchvision.datasets import CIFAR10
        ds = CIFAR10(self.data_root, train=True, download=True)
        imgs = torch.from_numpy(ds.data).float().div_(255.0).permute(0, 3, 1, 2).contiguous()
        mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
        return (imgs - mean) / std, torch.tensor(ds.targets, dtype=torch.long)

    def _ensure(self, alpha, num_trainers):
        if self._table is None:
            self._table = self._build_table(alpha, num_trainers)
            self._imgs, self._targets = self._build_pool()
            logger.info(f"[ORACLE_INJECT] table built: {len(self._table)} trainers")

    # --- per-round injection ------------------------------------------------
    def inject(self, agg, channel, end_ids, task_to_perform):
        """Set true PROP_STAT_UTILITY (+accuracy) for the given candidate end_ids."""
        if not self.enabled or task_to_perform != "train":
            return
        try:
            self._ensure(self.alpha, self.num_trainers)
            sim_now = float(getattr(agg, "_vclock", None).now) if getattr(
                agg, "simulated", False) and getattr(agg, "_vclock", None) else 0.0
            device = agg.device
            model = agg.model
            n_set = 0
            for eid in end_ids:
                info = self._table.get(str(eid))
                if info is None:
                    continue
                vis = _visible_count(sim_now, info["onset_s"], info["span_s"],
                                     info["total"], self.min_visible)
                g = info["arrival_global_idx"][:vis]
                util, acc = _oort_utility_acc(
                    model, self._imgs[g], self._targets[g], norm_n=vis,
                    device=device, sample_size=self.sample_size)
                channel.set_end_property(str(eid), PROP_STAT_UTILITY, util)
                if self.inject_accuracy:
                    channel.set_end_property(str(eid), PROP_LOCAL_ACCURACY, acc)
                n_set += 1
            logger.debug(f"[ORACLE_INJECT] round={getattr(agg,'_round','?')} "
                         f"sim_now={sim_now:.0f} set {n_set}/{len(end_ids)} utilities")
        except Exception as e:  # injection must never break training
            logger.warning(f"[ORACLE_INJECT] failed (non-fatal): {e}")


class OracleInjectMixin:
    """Mix into a CIFAR aggregator to enable per-baseline online oracle injection.

    Call ``self._init_oracle_util(data_root)`` in ``initialize()``; the framework
    aggregator calls ``_inject_oracle_utilities`` before each selection. Place the
    mixin BEFORE the framework TopAggregator in the bases so this override wins.
    """

    def _init_oracle_util(self, data_root):
        self._oracle_util = OracleUtilityProvider(self.config, data_root)

    def _inject_oracle_utilities(self, channel, task_to_perform):
        prov = getattr(self, "_oracle_util", None)
        if prov is None or not prov.enabled:
            return
        prov.inject(self, channel, list(channel.ends() or []), task_to_perform)
