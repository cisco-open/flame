# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Pure, side-effect-free selector scoring helpers.

These encode the *formula* each selector uses to turn per-candidate factor values
into a score. They are imported by both the live selectors AND the offline
staleness audit (``scripts/analysis/oracle_misselection.py``) so the
counterfactual "what would this selector pick with true/fresh factors" replay
never drifts from the real algorithm: the live path computes scores from believed
factor values, the audit re-computes them with true values substituted, using the
exact same functions here. No ``End``/channel state -- explicit args only.
"""

from __future__ import annotations

import math


# Canonical Oort hyperparameters = the paper's standalone Oort defaults
# (third_party/Oort/training/argParser.py). The `oort` and `felix` baselines use
# these; the `refl` baseline overrides a subset via selector.kwargs to match the
# REFL fork (third_party/REFL/core/argParser.py): round_threshold 30, clip_bound
# 0.9, cut_off_util 0.05, exploration_decay 0.98, exploration_min 0.3.
OORT_PAPER_DEFAULTS = {
    "round_threshold": 10.0,    # argParser.py:52
    "round_penalty": 2.0,       # :53  (== system_util exponent `alpha`)
    "clip_bound": 0.98,         # :56  (reward clip percentile, get_norm)
    "cut_off_util": 0.7,        # :105 (exploitation pool breadth factor)
    "pacer_step": 20,           # :20
    "pacer_delta": 5.0,         # :19
    "exploration_factor": 0.9,  # :24
    "exploration_decay": 0.95,  # :25
    "exploration_min": 0.2,     # :104
}


# ---- OORT / AsyncOort (Felix) -------------------------------------------------
# Final score = (stat_util + temporal_uncertainty) * system_utility


def oort_temporal_uncertainty(round_num: int, last_selected_round) -> float:
    """UCB-style exploration bonus; 0 if no history / undefined."""
    if not last_selected_round or last_selected_round <= 0 or round_num <= 0:
        return 0.0
    try:
        return math.sqrt(0.1 * math.log(round_num) / last_selected_round)
    except (ValueError, ZeroDivisionError):
        return 0.0


def oort_system_utility(
    round_duration_s, preferred_duration_s, alpha: float
) -> float:
    """Speed penalty: 1 if at/under preferred duration, else (pref/dur)^alpha."""
    if round_duration_s is None or preferred_duration_s is None:
        return 1.0
    if round_duration_s <= preferred_duration_s:
        return 1.0
    if round_duration_s <= 0:
        return 1.0
    return math.pow(preferred_duration_s / round_duration_s, alpha)


def oort_norm_stats(rewards: list, clip_bound: float = 0.95, thres: float = 1e-4):
    """Reference Oort ``get_norm`` (oort/oort.py:394): returns ``(min, range,
    clip_value)`` over the candidate reward list, used to normalize+clip the
    statistical reward into ~[0,1] before adding the temporal term. ``min`` is
    deflated by 0.999 and ``range`` floored at ``thres`` exactly as upstream."""
    if not rewards:
        return 0.0, thres, float("inf")
    s = sorted(rewards)
    clip_value = s[min(int(len(s) * clip_bound), len(s) - 1)]
    _min = s[0] * 0.999
    _range = max(s[-1] - _min, thres)
    return _min, _range, clip_value


def oort_normalize_reward(raw: float, min_: float, range_: float, clip_value: float) -> float:
    """Clip the raw reward at ``clip_value`` then min-max normalize to ~[0,1]
    (reference Oort score calc, oort.py:292-295). With no normalization the raw
    reward (~70) dwarfs the temporal/UCB term (~0.05), making exploration inert."""
    creward = min(raw, clip_value)
    return (creward - min_) / range_


def oort_combine_score(stat_util: float, temporal: float, system_util: float) -> float:
    """The exact combination used by OortSelector / AsyncOortSelector.

    ``stat_util`` is expected NORMALIZED (see ``oort_normalize_reward``) so the
    additive ``temporal`` term is meaningful, matching reference Oort."""
    return (stat_util + temporal) * system_util


def oort_total_utility(
    *,
    stat_util: float,
    round_duration_s,
    preferred_duration_s,
    alpha: float,
    round_num: int,
    last_selected_round,
) -> float:
    """End-to-end OORT/Felix per-candidate score from raw factor values."""
    temporal = oort_temporal_uncertainty(round_num, last_selected_round)
    system_util = oort_system_utility(round_duration_s, preferred_duration_s, alpha)
    return oort_combine_score(stat_util, temporal, system_util)


# ---- FedDance -----------------------------------------------------------------
# Final score U_m = (V_m * I_m * A_m) * (1 + log10(R+1) / (10*(1 + J_m)))


def feddance_mab(round_num: int, last_engaged_round) -> float:
    return 1.0 + (
        math.log10(round_num + 1) / (10.0 * (1.0 + (last_engaged_round or 0)))
    )


def feddance_utility(
    *, V: float, I: float, A: float, round_num: int, last_engaged_round
) -> float:
    return (V * I * A) * feddance_mab(round_num, last_engaged_round)


# ---- shared ranking -----------------------------------------------------------


def topk_by_score(scores: dict, k: int) -> list:
    """Deterministic top-k end_ids by score (desc); ties by id for stability."""
    if k <= 0:
        return []
    ordered = sorted(scores.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    return [eid for eid, _ in ordered[:k]]
