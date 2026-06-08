# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Selector determinism under seeding (real/sim parity prerequisite).

The selector runs in the aggregator process and draws from the process-global
``np.random`` / ``random`` RNGs. Phase 0 of the parity work seeds those RNGs at
aggregator init so selection becomes a pure function of (state, seed). These
tests assert that contract directly on the two selectors the felix baseline
uses (FedBuff for train, Oort for utility):

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
    """One fresh FedBuff send-state selection under the given seed."""
    _seed_all(seed)
    sel = FedBuffSelector(c=c, aggGoal=2)
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
    _seed_all(seed)
    sel = OortSelector(aggr_num=k)
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
