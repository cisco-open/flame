# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the simulated-time primitives.

These validate the ordering logic that makes simulated mode reproduce real
mode's per-round behavior: a monotone virtual clock, deterministic
completion-time ordering, and an arrival-order-independent reorder buffer.
"""

import random

import pytest

from flame.sim import SimReorderBuffer, VirtualClock, sim_ordered_ends


class TestVirtualClock:
    def test_starts_at_zero(self):
        assert VirtualClock().now == 0.0

    def test_advances_forward_only(self):
        c = VirtualClock()
        assert c.advance(5.0) == 5.0
        assert c.now == 5.0
        # a past timestamp must not move the clock backward
        assert c.advance(3.0) == 5.0
        assert c.now == 5.0
        assert c.advance(7.5) == 7.5

    def test_reset(self):
        c = VirtualClock()
        c.advance(10)
        c.reset()
        assert c.now == 0.0

    def test_advance_equal_ts_is_noop(self):
        c = VirtualClock()
        c.advance(5.0)
        assert c.advance(5.0) == 5.0  # equal ts must not regress or double-count
        assert c.now == 5.0


class TestSimOrderedEnds:
    def test_ascending_by_completion(self):
        order = sim_ordered_ends({"a": 30.0, "b": 10.0, "c": 20.0})
        assert order == ["b", "c", "a"]

    def test_ties_broken_by_end_id_deterministically(self):
        m = {"t3": 5.0, "t1": 5.0, "t2": 5.0}
        assert sim_ordered_ends(m) == ["t1", "t2", "t3"]

    def test_independent_of_insertion_order(self):
        base = {"a": 3.0, "b": 1.0, "c": 2.0}
        items = list(base.items())
        for _ in range(20):
            random.shuffle(items)
            assert sim_ordered_ends(dict(items)) == ["b", "c", "a"]


class TestSimReorderBuffer:
    def test_pop_min_orders_by_completion(self):
        buf = SimReorderBuffer()
        buf.add("a", 30.0, "pa")
        buf.add("b", 10.0, "pb")
        buf.add("c", 20.0, "pc")
        assert buf.pop_min() == ("b", 10.0, "pb")
        assert buf.pop_min() == ("c", 20.0, "pc")
        assert buf.pop_min() == ("a", 30.0, "pa")
        assert buf.pop_min() is None

    def test_pop_order_independent_of_add_order(self):
        durations = {"t0": 4.0, "t1": 1.0, "t2": 3.0, "t3": 2.0}
        expected = ["t1", "t3", "t2", "t0"]  # ascending by completion
        for _ in range(20):
            buf = SimReorderBuffer()
            ends = list(durations)
            random.shuffle(ends)  # simulate arbitrary physical arrival order
            for e in ends:
                buf.add(e, durations[e], None)
            popped = []
            while len(buf):
                popped.append(buf.pop_min()[0])
            assert popped == expected

    def test_has_pending_discard(self):
        buf = SimReorderBuffer()
        buf.add("a", 1.0)
        assert buf.has("a") and not buf.has("b")
        assert buf.pending_ends() == {"a"}
        buf.discard("a")
        assert not buf.has("a")
        assert len(buf) == 0

    def test_peek_min_ts(self):
        buf = SimReorderBuffer()
        assert buf.peek_min_ts() is None
        buf.add("a", 5.0)
        buf.add("b", 2.0)
        assert buf.peek_min_ts() == 2.0
        # peek does not remove
        assert len(buf) == 2

    def test_ties_broken_by_end_id(self):
        buf = SimReorderBuffer()
        buf.add("t2", 1.0)
        buf.add("t1", 1.0)
        assert buf.pop_min()[0] == "t1"

    def test_add_same_end_overwrites(self):
        # The aggregator re-probes an end across fill passes; a second add for
        # the same end must replace (not duplicate) its buffered entry.
        buf = SimReorderBuffer()
        buf.add("a", 5.0, "old")
        buf.add("a", 8.0, "new")
        assert len(buf) == 1
        assert buf.pop_min() == ("a", 8.0, "new")

    def test_clear_drops_all(self):
        buf = SimReorderBuffer()
        buf.add("a", 1.0)
        buf.add("b", 2.0)
        buf.clear()
        assert len(buf) == 0 and buf.pop_min() is None

    def test_pending_after_returns_future_completions(self):
        # PARITY §4.5: ends whose sct > vclock are "still computing".
        buf = SimReorderBuffer()
        buf.add("done", 5.0)
        buf.add("slow1", 12.0)
        buf.add("slow2", 30.0)
        assert buf.pending_after(10.0) == {"slow1", "slow2"}
        # boundary: sct == vclock is NOT still-computing (already available).
        assert buf.pending_after(12.0) == {"slow2"}
        # once vclock passes all sct, nothing is held (they all commit).
        assert buf.pending_after(99.0) == set()
        assert SimReorderBuffer().pending_after(0.0) == set()


class TestStalenessReconstruction:
    """A scripted scenario: committing buffered updates in completion order and
    advancing a virtual clock reproduces a hand-computed staleness sequence,
    independent of the order updates were buffered (arrival order)."""

    def test_staleness_matches_reference_independent_of_arrival(self):
        # trainer -> (model_version_when_sent, modeled completion time)
        # all sent at version 1; they complete at different sim times.
        scenario = {
            "t1": (1, 10.0),
            "t2": (1, 5.0),
            "t3": (1, 25.0),
            "t4": (1, 15.0),
        }
        agg_goal = 2  # model version increments every 2 commits

        def run(arrival_order):
            clock = VirtualClock()
            buf = SimReorderBuffer()
            for e in arrival_order:
                buf.add(e, scenario[e][1], scenario[e][0])
            model_version = 1
            committed = 0
            staleness_seq = []
            while len(buf):
                end, ts, sent_version = buf.pop_min()
                clock.advance(ts)
                staleness_seq.append((end, model_version - sent_version))
                committed += 1
                if committed % agg_goal == 0:
                    model_version += 1
            return staleness_seq

        # commit order is always by completion time: t2(5),t1(10),t4(15),t3(25)
        # versions: t2,t1 at v1 (staleness 0); then v->2; t4,t3 at v2 (staleness 1)
        reference = [("t2", 0), ("t1", 0), ("t4", 1), ("t3", 1)]
        import itertools

        for perm in itertools.permutations(scenario.keys()):
            assert run(list(perm)) == reference
