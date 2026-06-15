# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the per-commit virtual-clock overhead model.

`sim_commit_overhead_s` charges the agg→trainer→agg MQTT/dispatch latency the
max(gpu, D) timing model omits (PARITY.md §5 CRITICAL-1 / MEDIUM-1). It is
applied via TopAggregator._advance_sim_clock, shared by the syncfl/oort/asyncfl
commit paths. Default 0.0 must be a no-op (identical to the old advance(sct)).
"""

import pytest

from flame.config import Hyperparameters
from flame.mode.horizontal.asyncfl.top_aggregator import TopAggregator
from flame.sim import VirtualClock


class _Agg(TopAggregator):
    """Concrete stub so we can instantiate without the abstract methods."""

    def check_and_sleep(self):
        pass

    def evaluate(self):
        pass

    def initialize(self):
        pass

    def load_data(self):
        pass

    def train(self):
        pass


def _mk(overhead=None):
    a = _Agg.__new__(_Agg)
    a._vclock = VirtualClock()
    if overhead is not None:
        a._sim_commit_overhead_s = overhead
    return a


class TestAdvanceSimClock:
    def test_zero_overhead_is_plain_advance(self):
        """Default 0 → behaves exactly like vclock.advance(sct)."""
        a = _mk(0.0)
        for sct in (5.0, 5.0, 5.0, 5.0):
            a._advance_sim_clock(sct)
        assert a._vclock.now == 5.0

    def test_missing_attr_defaults_to_zero(self):
        """Instantiation paths that bypass __init__ must not crash."""
        a = _mk(overhead=None)  # _sim_commit_overhead_s unset
        a._advance_sim_clock(7.0)
        assert a._vclock.now == 7.0

    def test_clustered_completions_charge_k_times(self):
        """When completions cluster, K commits add ~K*overhead (serialized)."""
        a = _mk(1.0)
        for sct in (5.0, 5.0, 5.0, 5.0):
            a._advance_sim_clock(sct)
        assert a._vclock.now == pytest.approx(5.0 + 4 * 1.0)  # 9.0

    def test_spread_completions_charge_once_per_gap(self):
        """When compute already spreads completions, overhead overlaps → +1."""
        a = _mk(1.0)
        for sct in (10.0, 20.0, 30.0, 40.0):
            a._advance_sim_clock(sct)
        assert a._vclock.now == pytest.approx(40.0 + 1.0)  # 41.0

    def test_monotone_under_overhead(self):
        a = _mk(0.5)
        prev = 0.0
        for sct in (3.0, 1.0, 2.0, 8.0, 2.0):
            a._advance_sim_clock(sct)
            assert a._vclock.now >= prev
            prev = a._vclock.now


class TestConfigField:
    def test_default_is_zero(self):
        hp = Hyperparameters(rounds=1, epochs=1)
        assert hp.sim_commit_overhead_s == 0.0

    def test_alias_parses(self):
        hp = Hyperparameters(rounds=1, epochs=1, simCommitOverheadSeconds=0.58)
        assert hp.sim_commit_overhead_s == pytest.approx(0.58)
