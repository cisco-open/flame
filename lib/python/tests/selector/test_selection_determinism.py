# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Selector determinism under seeding (real/sim parity prerequisite).

Each selector now owns a DEDICATED RNG (``self._rng`` / ``self._pyrng``) seeded
at construction from ``config.hyperparameters.seed`` (threaded as the reserved
``_seed`` kwarg by the channel manager). Drawing from the dedicated RNG rather
than the process-global ``np.random`` / ``random`` makes selection a pure
function of (state, seed) AND insulates it from any other np.random consumer in
the process — so a residual real/sim divergence under a shared seed is a genuine
input divergence, not RNG desync. These tests assert that contract on the two
selectors the felix baseline uses (FedBuff for train, Oort for utility):

  * same seed + same state  -> identical selection (reproducible),
  * the selection actually consumes the RNG (distinct seeds can differ), so the
    seeding is meaningful rather than a no-op.

Without this, real and simulated runs can never match selection even with
identical logical state -- which is exactly what the 2026-05-30 parity smoke
(mean Jaccard 0.48) showed.
"""

import random

import numpy as np
import pytest

from flame.channel import (
    KEY_CH_SELECT_REQUESTER,
    KEY_CH_STATE,
    VAL_CH_STATE_SEND,
)
from flame.selector.fedbuff import FedBuffSelector
from flame.selector.oort import OortSelector


def _seed_all(seed: int) -> None:
    """Mirror the aggregator's Phase-0 seeding of the process-global RNGs."""
    np.random.seed(seed)
    random.seed(seed)


def _fedbuff_chosen(seed, make_ends, n=20, c=4):
    """One fresh FedBuff send-state selection under the given seed.

    Seed is threaded via the dedicated-RNG kwarg (``_seed``), as the channel
    manager does — NOT the process-global RNG.
    """
    sel = FedBuffSelector(_seed=seed, c=c, aggGoal=2)
    ends = make_ends(count=n, prefix="t")
    cp = {
        KEY_CH_STATE: VAL_CH_STATE_SEND,
        KEY_CH_SELECT_REQUESTER: "agg",
        "round": 1,
    }
    result = sel.select(ends, cp, trainer_unavail_list=[])
    return frozenset(result)


def _oort_chosen(seed, make_ends, n=20, k=3):
    """One fresh Oort first-round (random) selection under the given seed."""
    sel = OortSelector(_seed=seed, aggr_num=k)
    ends = make_ends(count=n, prefix="t")
    result = sel.select(
        ends, {"round": 1, "cur_time": 0.0},
        trainer_unavail_list=[], task_to_perform="train",
    )
    return frozenset(result)


class TestFedBuffDeterminism:
    def test_same_seed_reproducible(self, make_ends):
        assert _fedbuff_chosen(42, make_ends) == _fedbuff_chosen(42, make_ends)

    def test_subset_of_candidates(self, make_ends):
        chosen = _fedbuff_chosen(42, make_ends, n=20, c=4)
        assert len(chosen) == 4  # sampled a strict subset -> RNG was exercised

    def test_distinct_seeds_can_differ(self, make_ends):
        # With 20 candidates choose-4, distinct seeds almost surely vary; assert
        # at least two outcomes across several seeds (guards the RNG is live).
        outcomes = {_fedbuff_chosen(s, make_ends) for s in range(12)}
        assert len(outcomes) > 1


class TestOortDeterminism:
    def test_same_seed_reproducible(self, make_ends):
        assert _oort_chosen(7, make_ends) == _oort_chosen(7, make_ends)

    def test_distinct_seeds_can_differ(self, make_ends):
        outcomes = {_oort_chosen(s, make_ends) for s in range(12)}
        assert len(outcomes) > 1


class TestSeedingContract:
    """The exact mechanism the aggregator relies on: re-seeding both global RNGs
    makes a downstream draw reproducible."""

    def test_numpy_and_random_reseed_reproduces(self):
        _seed_all(123)
        a = (np.random.rand(5).tolist(), [random.random() for _ in range(5)])
        _seed_all(123)
        b = (np.random.rand(5).tolist(), [random.random() for _ in range(5)])
        assert a == b


class TestDedicatedRngInsulation:
    """The dedicated per-selector RNG must be insulated from the process-global
    RNG, so other np.random/random consumers in the aggregator can't desync
    selection between real and sim (the whole point of self._rng/_pyrng)."""

    def test_global_rng_perturbation_does_not_affect_selection(self, make_ends):
        first = _oort_chosen(7, make_ends)
        # Perturb the global RNGs arbitrarily (simulating other consumers doing a
        # DIFFERENT amount of work between selections in real vs sim).
        np.random.seed(999)
        _ = np.random.rand(37)
        random.seed(123)
        _ = [random.random() for _ in range(11)]
        second = _oort_chosen(7, make_ends)
        assert first == second

    def test_unseeded_is_still_stochastic(self, make_ends):
        # _seed=None (legacy) -> independent RNG each construction -> may differ.
        outcomes = {
            frozenset(
                OortSelector(aggr_num=3).select(
                    make_ends(count=20, prefix="t"),
                    {"round": 1, "cur_time": 0.0},
                    trainer_unavail_list=[], task_to_perform="train",
                )
            )
            for _ in range(12)
        }
        assert len(outcomes) > 1


from flame.selector import AbstractSelector


class _Mini(AbstractSelector):
    """Minimal concrete selector to exercise the base-class RNG contract."""

    def select(self, ends, channel_props):
        return {}


def _np_seq(sel, n=8):
    return sel._rng.rand(n).tolist()


def _py_seq(sel, n=8):
    return [sel._pyrng.random() for _ in range(n)]


class TestDedicatedRngContract:
    """Base-class guarantees the dedicated RNGs uphold, independent of any
    selector's selection logic."""

    def test_same_seed_same_sequence(self):
        a, b = _Mini(_seed=7), _Mini(_seed=7)
        assert _np_seq(a) == _np_seq(b)
        assert _py_seq(a) == _py_seq(b)

    def test_distinct_seeds_differ(self):
        assert _np_seq(_Mini(_seed=7)) != _np_seq(_Mini(_seed=8))
        assert _py_seq(_Mini(_seed=7)) != _py_seq(_Mini(_seed=8))

    def test_none_seed_is_unseeded_and_independent(self):
        assert _Mini(_seed=None)._seed is None
        # two independent unseeded RNGs almost surely differ
        assert _np_seq(_Mini(_seed=None)) != _np_seq(_Mini(_seed=None))

    def test_construction_does_not_touch_global_rng(self):
        # Seeding a selector must not perturb the process-global RNG state.
        random.seed(0)
        before = [random.random() for _ in range(3)]
        random.seed(0)
        _Mini(_seed=123)  # constructing a seeded selector in between
        after = [random.random() for _ in range(3)]
        assert before == after

    @pytest.mark.parametrize("seed", [None, 0, 1234])
    def test_seed_recorded_and_rngs_present(self, seed):
        sel = _Mini(_seed=seed)
        assert sel._seed == seed
        assert isinstance(_np_seq(sel), list) and isinstance(_py_seq(sel), list)
