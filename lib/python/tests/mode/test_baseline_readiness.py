# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Readiness guards for every baseline + the sim-speedup changes.

Fast, no-cluster checks so regressions in baseline wiring or the shared
aggregator changes (in-memory cache, serialize-once sends) are caught in CI:

  * every baseline in baselines.yaml resolves its selector + optimizer,
  * MemCache is a drop-in for the diskcache API the optimizers use, and a real
    fedavg aggregation runs on it and frees entries,
  * a model message survives serialize-once (dumps -> loads) losslessly.

The sim-recv barrier ordering for all three stacks is covered by
test_async_sim_ordering / test_sync_sim_ordering / test_sim_barrier.
"""

from __future__ import annotations

import pathlib

import cloudpickle
import pytest

# Baseline -> (selector sort, optimizer sort), mirrors baselines.yaml. Loaded
# from the catalog when present so this stays in sync automatically.
_FALLBACK = {
    "felix": ("async_oort", "fedbuff"),
    "fedbuff": ("fedbuff", "fedbuff"),
    "refl": ("refl_oort", "refl"),
    "oort": ("oort", "fedavg"),
    "feddance": ("feddance", "fedavg"),
    "fedavg": ("random", "fedavg"),
}


def _baseline_sorts():
    shared = (pathlib.Path(__file__).resolve().parents[2]
              / "examples" / "_metadata" / "baselines.yaml")
    if not shared.is_file():
        return _FALLBACK
    import yaml
    cat = (yaml.safe_load(shared.read_text()) or {}).get("baselines", {})
    out = {}
    for name, b in cat.items():
        agg = b.get("aggregator", {})
        sel = (agg.get("selector", {}) or {}).get("sort")
        opt = (agg.get("optimizer", {}) or {}).get("sort")
        if sel and opt:
            out[name] = (sel, opt)
    return out or _FALLBACK


@pytest.mark.parametrize("baseline", sorted(_baseline_sorts()))
def test_baseline_selector_and_optimizer_registered(baseline):
    """Each baseline's selector + optimizer must be registered (else it can't run).

    Checks registration (not instantiation, which needs per-baseline kwargs).
    """
    from flame.selectors import selector_provider
    from flame.optimizers import optimizer_provider
    sel_sort, opt_sort = _baseline_sorts()[baseline]
    assert sel_sort in selector_provider._objects, f"selector {sel_sort} not registered"
    assert opt_sort in optimizer_provider._objects, f"optimizer {opt_sort} not registered"


def test_memcache_dropin_and_frees():
    """MemCache supports the diskcache API the optimizers call, and pop frees."""
    from flame.mode.horizontal.syncfl.top_aggregator import MemCache
    c = MemCache()
    c["a"] = 1; c["b"] = 2
    c.reset("size_limit", 1)          # diskcache config -> no-op
    assert set(c.iterkeys()) == {"a", "b"}
    assert c.pop("a") == 1 and len(c) == 1   # consumed entry freed
    assert c["b"] == 2


def test_fedavg_aggregation_on_memcache():
    """A real fedavg.do over a MemCache aggregates and empties the cache
    (the in-memory swap on the live aggregation path, used by every sync baseline)."""
    torch = pytest.importorskip("torch")
    from flame.mode.horizontal.syncfl.top_aggregator import MemCache
    from flame.optimizers import optimizer_provider
    from flame.optimizer.train_result import TrainResult

    opt = optimizer_provider.get("fedavg")
    base = {"w": torch.zeros(3)}
    cache = MemCache()
    cache["t1"] = TrainResult(weights={"w": torch.ones(3)}, count=5, end_id="t1")
    cache["t2"] = TrainResult(weights={"w": torch.ones(3) * 3}, count=5, end_id="t2")
    agg = opt.do(base, cache, total=10)
    # weighted mean: 0.5*1 + 0.5*3 = 2
    assert agg is not None and torch.allclose(agg["w"], torch.full((3,), 2.0))
    assert len(cache) == 0, "optimizer must pop/free consumed updates"


def test_serialize_once_roundtrip():
    """The model message survives serialize-once (dumps once, reused per end)
    losslessly -- send_payload delivers exactly what send would have."""
    torch = pytest.importorskip("torch")
    msg = {"WEIGHTS": {"w": torch.ones(4)}, "ROUND": 7, "SIM_SEND_TS": 12.5}
    restored = cloudpickle.loads(cloudpickle.dumps(msg))
    assert restored["ROUND"] == 7 and restored["SIM_SEND_TS"] == 12.5
    assert torch.allclose(restored["WEIGHTS"]["w"], msg["WEIGHTS"]["w"])
