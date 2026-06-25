# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Aggregator-level test for simulated-mode receive ordering.

Exercises the real ``TopAggregator._sim_recv_min`` against a fake channel to
prove that, regardless of the physical arrival order, the async aggregator
commits in-flight updates in ascending virtual-completion order and advances
its virtual clock accordingly.
"""

import itertools

import pytest

from flame.mode.horizontal.asyncfl.top_aggregator import TopAggregator
from flame.mode.message import MessageType
from flame.sim import SimReorderBuffer, VirtualClock


class _FakeEnd:
    """Minimal end with the get/set_property surface _sim_recv_min touches
    (it resets buffered ends' KEY_END_STATE so the selector keeps their slot).

    `is_rxq_empty()` models physical readiness (§3j): the drain pulls any
    in-flight end whose rxq is non-empty regardless of modeled completion."""

    def __init__(self, ready_fn=None):
        self._props = {}
        self._ready_fn = ready_fn  # callable -> True if a message has arrived

    def get_property(self, key):
        return self._props.get(key)

    def set_property(self, key, value):
        self._props[key] = value

    def is_rxq_empty(self):
        return True if self._ready_fn is None else not self._ready_fn()


class FakeChannel:
    """Minimal channel: queued (end -> msg) delivered in a fixed arrival order.

    Uses a list-based queue that is consumed eagerly (not via a generator)
    so that calling ``next(recv_fifo(...))`` removes the message immediately.
    Mirrors the one-message-per-call pattern _sim_recv_min uses.
    """

    def __init__(self, inflight, arrival_order):
        self._inflight = set(inflight)
        self._queue = list(arrival_order)  # list of (end_id, sct)
        # An end's rxq is "non-empty" iff it has a message still queued — models
        # physical arrival (§3j drains ready in-flight ends regardless of exp).
        self._ends = {
            e: _FakeEnd(ready_fn=(lambda e=e: any(q[0] == e for q in self._queue)))
            for e in self._inflight
        }

    def has(self, end_id):
        return end_id in self._inflight

    def ends(self, state=None):
        return list(self._inflight)

    def recv_fifo(self, end_ids, first_k=0, timeout=None):
        # Model the real recv_fifo: drain ALL currently-ready messages whose end
        # is in end_ids (FIFO across the set) in a single call, then signal "no
        # more ready" with (None, ...). The barrier drain relies on this set-wide
        # behavior; yielding one-at-a-time would misrepresent the real API.
        ids = set(end_ids)
        i = 0
        while i < len(self._queue):
            end_id, sct = self._queue[i]
            if end_id in ids:
                self._queue.pop(i)
                yield (
                    {MessageType.WEIGHTS: f"w_{end_id}",
                     MessageType.SIM_COMPLETION_TS: sct,
                     MessageType.TRAINING_BUDGET_S: sct},
                    (end_id, None),
                )
            else:
                i += 1
        yield (None, ("", None))  # nothing more ready this pass


class _ConcreteAgg(TopAggregator):
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


def _make_agg():
    """A TopAggregator with only the state _sim_recv_min needs."""
    agg = _ConcreteAgg.__new__(_ConcreteAgg)
    agg._vclock = VirtualClock()
    agg._sim_buffer = SimReorderBuffer()
    agg._sim_committed = set()
    # _sim_recv_min consults _sim_pending_commit to release cross-round-blocked
    # ends; the real __init__ sets it, which __new__ bypasses here.
    agg._sim_pending_commit = set()
    # Virtual-completion gate state (real __init__ sets these; __new__ bypasses).
    agg._sim_inflight_expected = {}
    agg._sim_trainer_budget = {}
    agg._sim_budget_min = 12.0
    agg._sim_budget_running_mean = 12.0
    agg._sim_budget_n = 0
    return agg


def _drain(agg, channel):
    """Repeatedly call _sim_recv_min until all messages committed.

    Each call: fill buffer from all currently-receivable ends (one probe each),
    then pop and commit the minimum. The test provides instantaneous delivery
    (no real timeout needed), so we keep calling until the queue and buffer
    are both empty.
    """
    committed = []
    max_iters = (len(channel.ends()) + 1) * 3
    for _ in range(max_iters):
        recv_ends = [e for e in channel.ends() if channel.has(e)]
        msg, (end, _) = agg._sim_recv_min(channel, recv_ends)
        if msg is not None:
            committed.append((end, msg[MessageType.SIM_COMPLETION_TS]))
        if not channel._queue and not agg._sim_buffer.pending_ends():
            break
    return committed, agg._vclock.now


class TestSimRecvMin:
    def test_commits_in_completion_order(self):
        durations = {"t1": 10.0, "t2": 5.0, "t3": 25.0, "t4": 15.0}
        expected = [("t2", 5.0), ("t1", 10.0), ("t4", 15.0), ("t3", 25.0)]
        agg = _make_agg()
        # arrival order deliberately scrambled vs completion order
        channel = FakeChannel(
            inflight=set(durations),
            arrival_order=[("t3", 25.0), ("t1", 10.0), ("t4", 15.0), ("t2", 5.0)],
        )
        committed, t_v = _drain(agg, channel)
        assert committed == expected
        assert t_v == 25.0  # advanced to the last committed completion

    def test_independent_of_arrival_order(self):
        durations = {"a": 3.0, "b": 1.0, "c": 2.0, "d": 4.0}
        expected = [("b", 1.0), ("c", 2.0), ("a", 3.0), ("d", 4.0)]
        for arrival in itertools.permutations(durations.items()):
            agg = _make_agg()
            channel = FakeChannel(set(durations), list(arrival))
            committed, _ = _drain(agg, channel)
            assert committed == expected

    def test_virtual_clock_monotone(self):
        durations = {"x": 8.0, "y": 2.0, "z": 5.0}
        agg = _make_agg()
        channel = FakeChannel(set(durations), list(durations.items()))
        committed, t_v = _drain(agg, channel)
        times = [c for _, c in committed]
        assert times == sorted(times)  # committed in nondecreasing sim time
        assert t_v == max(times)


def _real_path_drain(arrival_order):
    """Reference model of the REAL aggregator commit order: FIFO by physical
    arrival (mirrors channel.recv_fifo popping one per iteration)."""
    return [(end, sct) for end, sct in arrival_order]


def _sim_path_drain(arrival_order):
    """SIM commit order via the real TopAggregator._sim_recv_min over a fake
    channel with the given physical arrival order."""
    durations = {e: s for e, s in arrival_order}
    agg = _make_agg()
    channel = FakeChannel(set(durations), list(arrival_order))
    committed, _ = _drain(agg, channel)
    return committed


def _staleness_seq(commit_seq, sent_version, agg_goal):
    """Staleness (agg_round - sent_version) per commit, with model version
    incrementing every agg_goal commits — the async aggregator's bookkeeping."""
    version = 1
    out = []
    for i, (end, _sct) in enumerate(commit_seq):
        out.append((end, version - sent_version[end]))
        if (i + 1) % agg_goal == 0:
            version += 1
    return out


class TestRealVsSimPathEquivalence:
    """The core parity guarantee: in a low-contention scenario where physical
    arrival order already equals completion order, the simulated commit path
    reproduces the real (arrival-ordered) path EXACTLY — same commit sequence
    and same staleness. Under scrambled arrival (the GPU-contention/jitter case)
    the sim path still follows completion order while the real path follows
    arrival; this is precisely the non-determinism sim removes."""

    SCENARIO = {"t1": 10.0, "t2": 5.0, "t3": 25.0, "t4": 15.0}
    SENT_VERSION = {"t1": 1, "t2": 1, "t3": 1, "t4": 1}
    COMPLETION_ORDER = [("t2", 5.0), ("t1", 10.0), ("t4", 15.0), ("t3", 25.0)]

    def test_sorted_arrival_real_equals_sim(self):
        # arrival already in completion order (well-separated D, no contention)
        arrival = list(self.COMPLETION_ORDER)
        real = _real_path_drain(arrival)
        sim = _sim_path_drain(arrival)
        assert sim == real == self.COMPLETION_ORDER
        # staleness sequences match exactly too
        assert _staleness_seq(sim, self.SENT_VERSION, 2) == _staleness_seq(
            real, self.SENT_VERSION, 2
        )

    def test_scrambled_arrival_sim_recovers_completion_order(self):
        import itertools

        for perm in itertools.permutations(self.SCENARIO.items()):
            arrival = list(perm)
            sim = _sim_path_drain(arrival)
            # sim always recovers completion order regardless of arrival jitter
            assert sim == self.COMPLETION_ORDER
            # the real path commits in arrival order — only equal to sim when
            # arrival already was completion order (documents the divergence).
            if arrival != self.COMPLETION_ORDER:
                assert _real_path_drain(arrival) != sim


    def test_sim_staleness_independent_of_arrival(self):
        # staleness is a function of completion order, not arrival order — so
        # sim yields one canonical staleness sequence under any arrival jitter.
        import itertools

        ref = _staleness_seq(self.COMPLETION_ORDER, self.SENT_VERSION, 2)
        for perm in itertools.permutations(self.SCENARIO.items()):
            sim = _sim_path_drain(list(perm))
            assert _staleness_seq(sim, self.SENT_VERSION, 2) == ref


class TestRedispatchGap:
    """§3k: the post-commit re-dispatch gap records a per-trainer cooldown
    (sct + gap) at commit time. _distribute_weights consumes it to hold a
    just-committed end out of selection until vclock passes the cooldown, so the
    end returns with a fresher model_version and the gap does NOT inflate the
    committing update's staleness (which is already set by the pre-commit sct).
    """

    def test_gap_off_records_no_cooldown(self):
        # Default (gap unset / 0): the mechanism is inert — no cooldown bookkeeping.
        durations = {"t1": 10.0, "t2": 5.0}
        agg = _make_agg()
        channel = FakeChannel(set(durations), [("t2", 5.0), ("t1", 10.0)])
        _drain(agg, channel)
        assert not getattr(agg, "_sim_cooldown_until", {})

    def test_gap_on_records_cooldown_at_sct_plus_gap(self):
        durations = {"t1": 10.0, "t2": 5.0, "t3": 25.0}
        gap = 1.5
        agg = _make_agg()
        agg._sim_redispatch_gap_s = gap
        agg._sim_cooldown_until = {}
        channel = FakeChannel(set(durations),
                              [("t3", 25.0), ("t2", 5.0), ("t1", 10.0)])
        committed, _ = _drain(agg, channel)
        # every committed end gets cooldown = its own sct + gap
        assert agg._sim_cooldown_until == {
            end: sct + gap for end, sct in committed
        }
        # cooldown is strictly in the future of each commit's sct (gap > 0)
        for end, sct in committed:
            assert agg._sim_cooldown_until[end] == pytest.approx(sct + gap)

    def test_gap_does_not_change_commit_order_or_clock(self):
        # The gap is a post-commit scheduling effect: it must not perturb the
        # in-cycle commit order or the virtual clock advance.
        durations = {"a": 3.0, "b": 1.0, "c": 2.0}
        agg_off = _make_agg()
        ch_off = FakeChannel(set(durations), [("a", 3.0), ("b", 1.0), ("c", 2.0)])
        committed_off, tv_off = _drain(agg_off, ch_off)
        agg_on = _make_agg()
        agg_on._sim_redispatch_gap_s = 1.0
        agg_on._sim_cooldown_until = {}
        ch_on = FakeChannel(set(durations), [("a", 3.0), ("b", 1.0), ("c", 2.0)])
        committed_on, tv_on = _drain(agg_on, ch_on)
        assert committed_on == committed_off
        assert tv_on == tv_off


class TestCoolingHoldsConcurrency:
    """§3L: cooling trainers (post-commit re-dispatch limbo) must occupy a
    concurrency slot so the idle pool can't refill it — otherwise the redispatch
    gap is inert (computing concurrency pinned at c) and advance/staleness miss
    parity. The selector subtracts ``sim_cooling_count`` from the free-slot budget.
    """

    @staticmethod
    def _stub_selector():
        from flame.selector.async_oort import AsyncOortSelector

        sel = AsyncOortSelector.__new__(AsyncOortSelector)
        sel.requester = "agg"
        sel.selected_ends = {"agg": set()}  # no in-flight
        sel.all_selected = {}
        return sel

    def _call(self, sel, concurrency, cooling_count):
        ends = {f"t{i}": _FakeEnd() for i in range(5)}
        return sel._handle_send_state(
            ends=ends,
            concurrency=concurrency,
            channel_props={"round": 1, "sim_cooling_count": cooling_count},
            trainer_unavail_list=[],
            task_to_perform="train",
            agg_version_state=(1, 0, 0),
            trainer_version_states={},
        )

    def test_full_cooling_holds_all_slots_no_refill(self):
        # 2 free slots fully consumed by 2 cooling ends -> extra == 0 -> no dispatch.
        sel = self._stub_selector()
        assert self._call(sel, concurrency=2, cooling_count=2) == {}

    def test_zero_cooling_proceeds_past_shortcircuit(self):
        # With no cooling and free slots, selection must NOT short-circuit at
        # extra == 0; a pacer tripwire (hit only past the short-circuit) proves it.
        sel = self._stub_selector()

        class _Tripwire(Exception):
            pass

        def _boom():
            raise _Tripwire()

        sel.pacer = _boom
        with pytest.raises(_Tripwire):
            self._call(sel, concurrency=2, cooling_count=0)


class TestGateProbesLiveInflight:
    """§3g: the gate must probe the LIVE in-flight set (_sim_inflight_expected),
    not just the recv_ends snapshot taken once per cycle. Otherwise it holds the
    clock for the earliest-expected straggler but never probes it (it's absent
    from the stale snapshot), spins, and commits past it — the past-dated commit
    that decouples version from the clock and drifts staleness."""

    def test_commits_earliest_inflight_absent_from_recv_snapshot(self):
        agg = _make_agg()
        # T has the earliest modeled completion but is NOT in the recv_ends
        # snapshot we pass (mimics a trainer that entered RECV after the snapshot
        # was taken at the top of _aggregate_weights).
        agg._sim_inflight_expected = {"A": 10.0, "B": 15.0, "T": 1.0}
        channel = FakeChannel(
            inflight={"A", "B", "T"},
            arrival_order=[("A", 10.0), ("B", 15.0), ("T", 1.0)],
        )
        msg, (end, _) = agg._sim_recv_min(channel, ["A", "B"])  # snapshot omits T
        # The earliest completion is committed first and drives the clock — the
        # gate pulled T in via the live in-flight set instead of lapping it.
        assert end == "T"
        assert msg[MessageType.SIM_COMPLETION_TS] == 1.0
        assert agg._vclock.now == 1.0

    def test_does_not_block_on_future_inflight(self):
        # An in-flight trainer expected far in the FUTURE (beyond the buffered
        # minimum) must NOT be waited for — we commit the ready earliest instead.
        agg = _make_agg()
        agg._sim_inflight_expected = {"A": 2.0, "FUT": 999.0}
        channel = FakeChannel(
            inflight={"A", "FUT"},
            arrival_order=[("A", 2.0)],  # FUT has not arrived (and shouldn't block)
        )
        msg, (end, _) = agg._sim_recv_min(channel, ["A"])
        assert end == "A"
        assert agg._vclock.now == 2.0

    def test_drains_ready_inflight_above_ceiling(self):
        # §3j: a SLOW trainer whose modeled exp is far future (above the probe
        # ceiling) but whose message has ALREADY ARRIVED must be drained NOW, so it
        # buffers as a future and commits in sct-order — not drained-in late and
        # committed past-dated (the residence~0, gap~45 staleness-tail signature).
        #
        # A pre-buffered entry makes the ceiling FINITE (buffered_min + slack); with
        # an empty buffer the ceiling is +inf and everything drains regardless, so
        # this non-empty-buffer setup is what isolates the §3j readiness admit.
        agg = _make_agg()
        agg._sim_inflight_expected = {"SLOW": 999.0}
        agg._sim_buffer.add(
            "A", 2.0,
            ({MessageType.WEIGHTS: "w_A", MessageType.SIM_COMPLETION_TS: 2.0}, ("A", None)),
        )
        channel = FakeChannel(inflight={"SLOW"}, arrival_order=[("SLOW", 5.0)])
        # recv_ends empty; buffer min = 2.0 → ceiling = 4.0 < SLOW.exp (999), so the
        # exp bound alone would NOT admit SLOW — only its rxq readiness does.
        msg, (end, _) = agg._sim_recv_min(channel, [])
        assert end == "A"                      # earliest still commits first
        assert agg._sim_buffer.has("SLOW")     # SLOW was DRAINED, not lapped
        assert agg._sim_buffer.peek_min_ts() == 5.0  # buffered as a future


class TestClockJumpClamp:
    """The arrival-based gate is inert in sim — real-GPU compute is ~0.4s wall, so
    every in-flight trainer is already buffered (inflight_tracked==buf_depth) and
    the gate's not-yet-arrived hold never fires (gate_holds=0). A forced commit of
    a far-future straggler then jumps the clock past the fresh fast cohort still
    in flight, past-dating it on arrival (72% of commits in the 2.5h felix run,
    fresh source dominant). The clamp re-bases the hold onto modeled completion:
    each commit advances the clock at most to the earliest in-flight FUTURE exp
    (+slack), INCLUDING buffered ends (which the gate excludes) — so a low-exp
    straggler still pins the jump."""

    def _buffered(self, agg, end, sct):
        agg._sim_buffer.add(
            end, sct,
            ({MessageType.WEIGHTS: f"w_{end}",
              MessageType.SIM_COMPLETION_TS: sct}, (end, None)),
        )

    def test_clamp_caps_jump_to_earliest_inflight_future(self):
        agg = _make_agg()
        agg._sim_clock_jump_clamp = True
        # A is the ready minimum (sct=100); S is in flight with a low modeled
        # completion estimate (exp=8, a lower bound). Committing A must not advance
        # the clock past S's possible completion — else a later-landing S is
        # past-dated. Both already buffered → the gate sees no stuck end, so only
        # the clamp can hold the clock here.
        agg._sim_inflight_expected = {"A": 100.0, "S": 8.0}
        self._buffered(agg, "A", 100.0)
        self._buffered(agg, "S", 500.0)
        channel = FakeChannel(inflight=set(), arrival_order=[])
        msg, (end, _) = agg._sim_recv_min(channel, [])
        assert end == "A"                       # earliest sct still commits first
        assert agg._vclock.now == 10.0          # S.exp(8) + slack(2), NOT 100

    def test_disabled_clamp_laps_to_committed_sct(self):
        agg = _make_agg()
        agg._sim_clock_jump_clamp = False
        agg._sim_inflight_expected = {"A": 100.0, "S": 8.0}
        self._buffered(agg, "A", 100.0)
        self._buffered(agg, "S", 500.0)
        channel = FakeChannel(inflight=set(), arrival_order=[])
        msg, (end, _) = agg._sim_recv_min(channel, [])
        assert end == "A"
        assert agg._vclock.now == 100.0         # old behavior: jumps to sct, laps S

    def test_clamp_never_advances_backwards(self):
        # No in-flight FUTURE (all exp <= vclock): clamp is inert, clock advances
        # normally to the committed sct (a jump with nothing to lap is safe).
        agg = _make_agg()
        agg._sim_clock_jump_clamp = True
        agg._vclock.advance(50.0)
        agg._sim_inflight_expected = {"A": 200.0, "OLD": 5.0}  # OLD exp < vclock
        self._buffered(agg, "A", 200.0)
        channel = FakeChannel(inflight=set(), arrival_order=[])
        msg, (end, _) = agg._sim_recv_min(channel, [])
        assert end == "A"
        assert agg._vclock.now == 200.0         # OLD (exp<=vclock) does not pin


class TestExpectedCompletionLowerBound:
    """The gate's expected completion must be a LOWER BOUND on sct, so the clock
    never laps a not-yet-seen trainer (the past-dating seed). The unseen-trainer
    default is the running MINIMUM observed budget, not the mean (which overshoots
    fast trainers: gate_holds=0 over a full felix run, 74% commits past-dated)."""

    def test_budget_min_tracks_minimum_below_mean(self):
        agg = _make_agg()
        # commit a fast (2s) and slow (25s) trainer; min must follow the fastest.
        channel = FakeChannel({"fast", "slow"}, [("fast", 2.0), ("slow", 25.0)])
        _drain(agg, channel)
        assert agg._sim_budget_min == 2.0
        assert agg._sim_budget_min < agg._sim_budget_running_mean  # min, not mean

    def test_unseen_default_is_lower_bound_not_mean(self):
        # After observing a fast trainer, an UNSEEN trainer dispatched now must get
        # expected = send + min (a true lower bound), never the larger mean — else
        # the clock laps the unseen trainer when it actually finishes earlier.
        agg = _make_agg()
        _drain(agg, FakeChannel({"fast"}, [("fast", 2.0)]))
        unseen_budget = agg._sim_trainer_budget.get("NEW", agg._sim_budget_min)
        assert unseen_budget == agg._sim_budget_min == 2.0
        assert unseen_budget <= agg._sim_budget_running_mean
