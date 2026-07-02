# Copyright 2024 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Shared availability trace resolver — single source of truth for agg and trainer.

state_at / next_avail_after replace the three inlined bisect_right copies in
get_curr_unavail_trainers (main_oort_sync_agg.py:311),
_trace_read_avail_check (asyncfl/top_aggregator.py), and
check_and_update_state_avl (trainer/pytorch/main.py:336).
"""

import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from sortedcontainers import SortedDict

from flame.config import TrainerAvailState

logger = logging.getLogger(__name__)

_DEFAULT_TRACE_DIR = (
    Path(__file__).resolve().parents[2] / "examples/_metadata/availability_traces"
)
_METADATA_DIR = Path(__file__).resolve().parents[2] / "examples/_metadata"

_MOBIPERF_SUBS: dict = {
    "mobiperf_2st": "states_2st",
    "mobiperf_3st_50": "states_3st_50",
    "mobiperf_3st_75": "states_3st_75",
}

_AVL_STATE_VALUES = frozenset(
    {TrainerAvailState.AVL_TRAIN.value, TrainerAvailState.AVL_EVAL.value}
)


# ---------------------------------------------------------------------------
# Internal YAML cache — loaded once per (trace_dir) call site, never reloaded.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=16)
def _raw_mobiperf(trace_dir: str) -> dict:
    # encoding="utf-8" explicit: open() otherwise falls back to the node's
    # locale-preferred encoding, which mis-decodes any non-ASCII byte on a
    # non-UTF-8 locale (e.g. C/POSIX) and yaml.safe_load then rejects the
    # resulting control chars — bit us once already on the sibling parity
    # YAML (see debug_run.sh).
    with open(Path(trace_dir) / "mobiperf_traces.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["traces"]


@lru_cache(maxsize=16)
def _raw_synthetic(trace_dir: str) -> dict:
    with open(Path(trace_dir) / "synthetic_traces.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["traces"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _canonical_trace_name(trace_name: str) -> str:
    """Normalize a configured trace name to a canonical store key.

    Legacy configs (trackTrainerAvail.trace) name traces with an ``avl_events_``
    prefix, e.g. ``avl_events_syn_20`` / ``avl_events_mobiperf_2st``; newer ones
    use the bare canonical name (``syn_20``). Strip the prefix so both resolve to
    the same SortedDict — without this the legacy oort/refl JSON configs load 0
    traces and the gate silently turns OFF.
    """
    if trace_name and trace_name.startswith("avl_events_"):
        return trace_name[len("avl_events_"):]
    return trace_name


def load_trace(
    trace_name: str,
    trainer_key: str,
    *,
    base_dir: Optional[str] = None,
) -> SortedDict:
    """Return per-trainer availability as SortedDict[ts_s → state_str].

    trace_name: canonical name — syn_0 / syn_20 / syn_50 / mobiperf_2st /
        mobiperf_3st_50 / mobiperf_3st_75.
    trainer_key: registry key ('trainer_001'). For mobiperf the numeric suffix
        is used to derive the device key ('device_001').
    base_dir: override for the trace store root; default = examples/_metadata/
        availability_traces/.

    syn_0 / all-available traces return an empty SortedDict (always AVL_TRAIN).
    """
    trace_dir = str(Path(base_dir) if base_dir else _DEFAULT_TRACE_DIR)
    trace_name = _canonical_trace_name(trace_name)

    if trace_name in _MOBIPERF_SUBS:
        sub = _MOBIPERF_SUBS[trace_name]
        raw = _raw_mobiperf(trace_dir)
        num = trainer_key.split("_")[-1]
        device_key = f"device_{num}"
        events = raw[device_key][sub]
    elif trace_name.startswith("syn_"):
        raw = _raw_synthetic(trace_dir)
        if trace_name not in raw:
            raise KeyError(f"trace {trace_name!r} not found in synthetic_traces.yaml")
        entry = raw[trace_name]
        per_trainer = entry.get("per_trainer", {}).get("n300", {})
        pattern = entry.get("pattern", [])
        events = per_trainer.get(trainer_key) or pattern
    else:
        raise KeyError(f"unsupported trace name: {trace_name!r}")

    result = SortedDict()
    for ts, state in events:
        result[float(ts)] = state
    return result


def state_at(trace: SortedDict, t: float) -> TrainerAvailState:
    """Trainer availability at time t.

    bisect_right(t)-1: the last event whose timestamp ≤ t.
    Returns AVL_TRAIN (safe default) when no event has fired yet or trace is empty.
    """
    if not trace:
        return TrainerAvailState.AVL_TRAIN
    idx = trace.bisect_right(t) - 1
    if idx < 0:
        return TrainerAvailState.AVL_TRAIN
    state_str = trace.peekitem(idx)[1]
    try:
        return TrainerAvailState(state_str)
    except ValueError:
        return TrainerAvailState.AVL_TRAIN


def next_avail_after(trace: SortedDict, t: float) -> float:
    """Smallest ts > t whose state ∈ {AVL_TRAIN, AVL_EVAL}.

    Returns math.inf if the trace never recovers (caller must guard — Challenge 10).
    """
    idx = trace.bisect_right(t)
    for i in range(idx, len(trace)):
        ts, state = trace.peekitem(i)
        if state in _AVL_STATE_VALUES:
            return float(ts)
    return math.inf


def read_trainer_unavailability(
    trace: Optional[str],
    base_dir: Optional[str] = None,
) -> Optional[dict]:
    """Build task_id → SortedDict[ts_s → state_str] from the canonical store.

    Free function (Batch 3 T3.2 Phase 2): extracted from
    ClientAvailability.read_trainer_unavailability so scripts/parity/
    ground_truth.py can call it directly without instantiating the mixin
    class. That method now delegates here — same behavior, one implementation.

    Reads examples/_metadata/trainer_registry.yaml once; individual trace
    SortedDicts are built via load_trace() which caches the raw YAML.
    Returns None on fatal errors (caller treats None as gate-off).
    """
    if not trace:
        return None

    registry_path = _METADATA_DIR / "trainer_registry.yaml"
    try:
        with open(registry_path, encoding="utf-8") as f:
            registry = yaml.safe_load(f)["trainers"]
    except FileNotFoundError:
        logger.error(f"[AVAIL] trainer registry not found: {registry_path}")
        return None

    trainer_events_dict: dict = {}
    errors = 0
    for tk, meta in registry.items():
        task_id = meta["task_id"]
        try:
            trainer_events_dict[task_id] = load_trace(trace, tk, base_dir=base_dir)
        except (KeyError, FileNotFoundError) as exc:
            logger.warning(f"[AVAIL] skipping {tk}: {exc}")
            errors += 1

    if errors:
        logger.warning(
            f"[AVAIL] {errors}/{len(registry)} trainers had missing trace data"
        )
    logger.info(
        f"[AVAIL] loaded {len(trainer_events_dict)} trainer traces (trace={trace!r})"
    )
    return trainer_events_dict or None
