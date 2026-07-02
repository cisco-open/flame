# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Sync-aggregator simulated-mode receive ordering.

Exercises the real ``TopAggregator._sync_sim_recv_first_k`` (syncfl) against a
fake channel to prove that, regardless of physical arrival order, the sync
aggregator commits the ``first_k`` updates with the SMALLEST sim_completion_ts
(the k that would physically finish first in real mode) and advances its virtual
clock to the k-th smallest. This is the sync analogue of the async ordering
guarantee and is what makes simulated mode decision-equivalent to real mode for
the sync baselines (fedavg / oort / refl / feddance).
"""

import itertools
from collections import defaultdict

import pytest

from flame.mode.horizontal.syncfl.top_aggregator import TopAggregator
from flame.mode.message import MessageType
from flame.selector.properties import (
    PROP_SIM_SEND_TS,
    PROP_CLIENT_TASK_TRAIN_DURATION,
    PROP_STAT_UTILITY,
)
from flame.sim import VirtualClock, SimReorderBuffer


class FakeSyncChannel:
    """Delivers one update per selected end (with a sim_completion_ts) in a
    fixed physical arrival order; ``recv_fifo`` drains all ready ones then
    signals a timeout with (None, ...)."""

    def __init__(self, scts, arrival_order, round_durations=None, sim_send_ts=0.0):
        self._scts = dict(scts)
        self._rd = dict(round_durations or {})
        self._queue = list(arrival_order)
        self._props = defaultdict(dict)
        for e in self._scts:
            self._props[e][PROP_SIM_SEND_TS] = sim_send_ts

    def has(self, e):
        return e in self._scts

    def ends(self, *a):
        return list(self._scts)

    def get_end_property(self, end, key):
        return self._props[end].get(key)

    def set_end_property(self, end, key, val):
        self._props[end][key] = val

    def recv_fifo(self, end_ids, first_k=0, timeout=None):
        ids = set(end_ids)
        i = 0
        while i < len(self._queue):
            e = self._queue[i]
            if e in ids:
                self._queue.pop(i)
                msg = {MessageType.WEIGHTS: f"w_{e}",
                       MessageType.SIM_COMPLETION_TS: self._scts[e]}
                if e in self._rd:
                    msg[MessageType.SIM_CLIENT_TASK_TRAIN_DURATION_S] = self._rd[e]
                yield (msg, (e, None))
            else:
                i += 1
        yield (None, ("", None))  # nothing more ready this pass


class _ConcreteSyncAgg(TopAggregator):
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
    agg = _ConcreteSyncAgg.__new__(_ConcreteSyncAgg)
    agg._vclock = VirtualClock()
    agg.simulated = True
    # Gate-off availability state. Production sets _sim_buffer in __init__ and the
    # ledgers in _init_availability; __new__ bypasses both, so set them here.
    # _sync_sim_recv_first_k references self._sim_buffer directly; trainer_event_dict
    # =None + empty pending_withheld keep the ClientAvailability helpers no-op, so the
    # sim-ordering logic is exercised in isolation (byte-identical to gate OFF).
    agg._sim_buffer = SimReorderBuffer()
    agg.trainer_event_dict = None
    agg.pending_withheld = {}
    return agg


def _committed_ends(agg, channel, first_k):
    out = agg._sync_sim_recv_first_k(channel, channel.ends(), first_k)
    return [md[0] for _msg, md in out]


SCTS = {"t1": 10.0, "t2": 5.0, "t3": 25.0, "t4": 15.0, "t5": 8.0}
# ascending by sct: t2(5), t5(8), t1(10), t4(15), t3(25)


class TestSyncSimRecvFirstK:
    def test_commits_k_smallest_completion(self):
        agg = _make_agg()
        ch = FakeSyncChannel(SCTS, arrival_order=["t3", "t1", "t4", "t2", "t5"])
        committed = _committed_ends(agg, ch, first_k=3)
        assert committed == ["t2", "t5", "t1"]  # 3 smallest sct, ascending
        assert agg._vclock.now == 10.0  # advanced to the k-th smallest

    def test_independent_of_arrival_order(self):
        expected = ["t2", "t5", "t1"]
        for arrival in itertools.permutations(SCTS):
            agg = _make_agg()
            ch = FakeSyncChannel(SCTS, list(arrival))
            assert _committed_ends(agg, ch, first_k=3) == expected

    def test_k_equals_all(self):
        agg = _make_agg()
        ch = FakeSyncChannel(SCTS, list(SCTS))
        committed = _committed_ends(agg, ch, first_k=5)
        assert committed == ["t2", "t5", "t1", "t4", "t3"]
        assert agg._vclock.now == 25.0

    def test_round_duration_set_from_sim_round_duration(self):
        agg = _make_agg()
        rd = {"t2": 5.0, "t5": 8.0, "t1": 10.0, "t3": 25.0, "t4": 15.0}
        ch = FakeSyncChannel(SCTS, list(SCTS), round_durations=rd, sim_send_ts=0.0)
        agg._sync_sim_recv_first_k(ch, ch.ends(), first_k=2)
        # committed t2,t5 get PROP_CLIENT_TASK_TRAIN_DURATION from SIM_CLIENT_TASK_TRAIN_DURATION_S
        assert ch.get_end_property("t2", PROP_CLIENT_TASK_TRAIN_DURATION).total_seconds() == 5.0
        assert ch.get_end_property("t5", PROP_CLIENT_TASK_TRAIN_DURATION).total_seconds() == 8.0

    def test_fewer_responders_than_k(self):
        # Only 2 of the 5 selected ever respond. Real mode's recv_fifo(first_k=3)
        # would block forever; the sim barrier drains what's ready (the recv_fifo
        # returns (None,...) once nothing more is queued) and commits the 2
        # smallest by sct. No fixed deadline needed — the grace is the dead-end
        # ceiling and the fake signals "no more ready" immediately.
        agg = _make_agg()
        ch = FakeSyncChannel(SCTS, arrival_order=["t1", "t2"])  # only 2 queued
        committed = _committed_ends(agg, ch, first_k=3)
        assert committed == ["t2", "t1"]  # the 2 smallest sct, ascending
        assert agg._vclock.now == 10.0


class TestSimInflightResidence:
    """PARITY §4.5: in sim the oort/refl aggregator marks trainers still computing
    in sim time (buffered sct > vclock) as UNAVAILABLE for selection, so a slow
    client stays out of the eligible pool until ``vclock >= sct`` — matching real,
    where it is genuinely busy. Excluded via the unavailable list (NOT selected_ends,
    which would re-dispatch it). Guards the buffer-driven hold/release invariant the
    aggregator snippet relies on (`SimReorderBuffer.pending_after` merged into the
    trainer_unavail_list)."""

    def _unavail(self, buf, base_unavail, vclock_now):
        # Mirror of oort/top_aggregator._distribute_weights §4.5 snippet.
        held = buf.pending_after(vclock_now)
        merged = list(set(base_unavail) | held) if held else list(base_unavail)
        return held, merged

    def test_slow_trainer_held_then_released(self):
        from flame.sim import SimReorderBuffer

        buf = SimReorderBuffer()
        buf.add("fast", 4.0)     # already complete at vclock 5
        buf.add("slow", 30.0)    # still computing at vclock 5

        # Round N (vclock=5): the slow trainer is held unavailable; the fast one is
        # not (it is available to commit, not still computing). Base unavail merged.
        held, merged = self._unavail(buf, base_unavail=["pre"], vclock_now=5.0)
        assert held == {"slow"}
        assert "slow" in merged          # excluded from selection / pool
        assert "fast" not in merged      # available, not held
        assert "pre" in merged           # additive merge, not overwrite

        # Once the clock passes its sct (it commits), it is no longer held.
        buf.discard("slow")  # committed -> leaves the buffer
        held2, merged2 = self._unavail(buf, base_unavail=["pre"], vclock_now=35.0)
        assert held2 == set() and merged2 == ["pre"]

    def test_noop_when_nothing_still_computing(self):
        from flame.sim import SimReorderBuffer

        buf = SimReorderBuffer()
        buf.add("a", 2.0)
        held, merged = self._unavail(buf, base_unavail=["x"], vclock_now=10.0)
        assert held == set() and merged == ["x"]  # untouched


class TestSimInflightCarryover:
    """PARITY §4.9: the sim oort/refl aggregator must CARRY a prior-round straggler
    that is still computing at this round's start (modeled sct > vclock_round_start)
    rather than deliver its physically-instant update and stale-clean it. Without the
    carry-over gate, sim in-flight drains to ~0 while real holds ~3 (overcommit). This
    drives the REAL ``OortTopAggregator._oort_sim_recv`` generator off a pre-loaded
    buffer to prove the gate holds the still-computing straggler and delivers the
    fresh + already-completed ends in ascending sct, advancing the clock only to the
    delivered ones."""

    def _make_oort_agg(self, carryover, round_num, vclock_start):
        from flame.mode.horizontal.oort.top_aggregator import (
            TopAggregator as OortTopAggregator,
        )
        from flame.sim import SimReorderBuffer

        class _ConcreteOortAgg(OortTopAggregator):
            check_and_sleep = evaluate = initialize = load_data = train = (
                lambda self: None
            )

        class _HP:
            def __init__(self, c):
                self.sim_inflight_carryover = c

        class _Cfg:
            def __init__(self, c):
                self.hyperparameters = _HP(c)

        agg = _ConcreteOortAgg.__new__(_ConcreteOortAgg)
        agg._vclock = VirtualClock()
        agg._vclock.advance(vclock_start)
        agg.simulated = True
        agg._round = round_num
        agg.config = _Cfg(carryover)
        agg._sim_buffer = SimReorderBuffer()
        return agg

    def _load(self, agg, items):
        # items: list of (end, sct, model_version)
        for end, sct, mv in items:
            msg = {MessageType.WEIGHTS: f"w_{end}",
                   MessageType.SIM_COMPLETION_TS: sct,
                   MessageType.SIM_CLIENT_TASK_TRAIN_DURATION_S: sct,
                   MessageType.MODEL_VERSION: mv}
            agg._sim_buffer.add(end, sct, (msg, (end, None)))

    def _drive(self, agg):
        # buffer pre-loaded; pass the same ends so to_probe is empty (no recv_fifo)
        ch = FakeSyncChannel({}, arrival_order=[])
        ends = list(agg._sim_buffer.pending_ends())
        return [md[0] for _msg, md in agg._oort_sim_recv(ch, ends)]

    # round 2 starting at vclock 5: "done" already completed (sct 4), "fresh" is this
    # round (sct 8), "slow" is a prior-round straggler still computing (sct 30).
    ITEMS = [("done", 4.0, 1), ("fresh", 8.0, 2), ("slow", 30.0, 1)]

    def test_carryover_holds_still_computing_straggler(self):
        agg = self._make_oort_agg(carryover=True, round_num=2, vclock_start=5.0)
        self._load(agg, self.ITEMS)
        committed = self._drive(agg)
        assert committed == ["done", "fresh"]            # ascending sct, slow held
        assert agg._vclock.now == 8.0                    # clock not advanced to 30
        assert agg._sim_buffer.has("slow")               # carried in-flight
        assert not agg._sim_buffer.has("fresh")          # delivered, left buffer

    def test_off_by_default_drains_straggler(self):
        agg = self._make_oort_agg(carryover=False, round_num=2, vclock_start=5.0)
        self._load(agg, self.ITEMS)
        committed = self._drive(agg)
        assert committed == ["done", "fresh", "slow"]    # all delivered (drained)
        assert agg._vclock.now == 30.0
        assert not agg._sim_buffer.has("slow")

    def test_carryover_releases_once_clock_passes_sct(self):
        # A later round starts past the straggler's sct -> it is delivered, not held.
        agg = self._make_oort_agg(carryover=True, round_num=4, vclock_start=35.0)
        self._load(agg, [("slow", 30.0, 1)])
        committed = self._drive(agg)
        assert committed == ["slow"]                     # released & committed (stale)
        assert not agg._sim_buffer.has("slow")

    def test_pinned_threshold_holds_straggler_across_retry(self):
        # Regression for the §4.9 decay: the block-for-K-fresh retry loop re-enters
        # _oort_sim_recv AFTER an earlier pass advanced the clock. The carry-over
        # threshold must stay pinned to the ROUND START (set by _aggregate_weights),
        # not re-read the advanced self._vclock.now — otherwise a straggler whose sct
        # falls between round-start and the advanced clock is committed instead of
        # carried (in_flight_after decayed 4.3 -> 0 over the run). Here round 2 started
        # at vclock 5; a prior pass advanced the clock to 8; a prior-round straggler
        # (sct 7) is still computing as of round start and must be HELD.
        agg = self._make_oort_agg(carryover=True, round_num=2, vclock_start=5.0)
        agg._round_start_vclock = 5.0          # pinned by _aggregate_weights at entry
        agg._vclock.advance(8.0)               # an earlier pass advanced now -> 8.0
        self._load(agg, [("straggler", 7.0, 1)])
        committed = self._drive(agg)
        assert committed == []                 # sct 7 > pinned start 5 -> held
        assert agg._sim_buffer.has("straggler")  # carried, not drained
        assert agg._vclock.now == 8.0          # untouched (a per-call `now` would commit it)


class TestStaleTrainerPropsRecorded:
    """Correctness fix (oort + refl): a trainer that was selected and computed must
    have its speed (PROP_CLIENT_TASK_TRAIN_DURATION) and statistical utility (PROP_STAT_UTILITY)
    recorded into the selector's memory EVEN when its update arrives stale and is
    dropped from aggregation. Otherwise the selector reads PROP_STAT_UTILITY=None as
    'unexplored' and re-selects the same slow trainers forever (kept real's mix
    artificially broad vs sim). See PARITY.md 'Real is the reference, but VERIFY'."""

    def _make_agg(self, simulated):
        from datetime import datetime
        from flame.mode.horizontal.oort.top_aggregator import (
            TopAggregator as OortTopAggregator,
        )

        class _ConcreteOortAgg(OortTopAggregator):
            check_and_sleep = evaluate = initialize = load_data = train = (
                lambda self: None
            )

        agg = _ConcreteOortAgg.__new__(_ConcreteOortAgg)
        agg._vclock = VirtualClock()
        agg.simulated = simulated
        agg._round = 10
        return agg

    def test_real_records_intrinsic_excluding_delivery_lag(self):
        """Real duration = WALL_SEND_TS - WALL_RECV_TS (the client's intrinsic
        compute+sleep, both client stamps). The dispatch->recv delivery lag (the agg
        stamps `dispatch` at selection, but a slow client held one-in-flight receives
        the weights later) MUST be excluded — folding it into WALL_SEND - dispatch
        inflated slow-trainer durations ~8s, raised the selector's preferred-duration
        percentile, and made real under-penalize slow clients vs sim (the K2 root)."""
        from datetime import datetime, timedelta

        agg = self._make_agg(simulated=False)
        ch = FakeSyncChannel({}, arrival_order=[])
        dispatch_ts = datetime(2026, 1, 1, 0, 0, 0)
        # agg dispatched at t=0; client received weights 8s later (delivery lag),
        # computed 12s, sent at t=20. Recorded duration must be 12 (intrinsic), not
        # 20 (dispatch-anchored). The +6s aggregator read-wait (recv at +26) is also
        # excluded by construction since we anchor on the two client stamps.
        wall_recv_ts = dispatch_ts.timestamp() + 8.0
        wall_send_ts = dispatch_ts.timestamp() + 20.0
        recv_ts = dispatch_ts + timedelta(seconds=26.0)
        agg._oort_sent_version_ts = {"slow": {7: dispatch_ts}}
        msg = {
            MessageType.MODEL_VERSION: 7,
            MessageType.STAT_UTILITY: 4.2,
            MessageType.WALL_RECV_TS: wall_recv_ts,
            MessageType.WALL_SEND_TS: wall_send_ts,
        }

        agg._record_returned_trainer_props(ch, "slow", msg, recv_ts)

        assert ch.get_end_property("slow", PROP_STAT_UTILITY) == 4.2
        assert ch.get_end_property("slow", PROP_CLIENT_TASK_TRAIN_DURATION).total_seconds() == 12.0

    def test_real_falls_back_to_wall_send_minus_dispatch_without_wall_recv(self):
        """Without WALL_RECV_TS the intrinsic measure is unavailable; fall back to
        WALL_SEND_TS - dispatch (the prior-fix path, still server-wait-free on the
        recv side)."""
        from datetime import datetime, timedelta

        agg = self._make_agg(simulated=False)
        ch = FakeSyncChannel({}, arrival_order=[])
        send_ts = datetime(2026, 1, 1, 0, 0, 0)
        wall_send_ts = send_ts.timestamp() + 12.0
        recv_ts = send_ts + timedelta(seconds=18.0)
        agg._oort_sent_version_ts = {"slow": {7: send_ts}}
        msg = {
            MessageType.MODEL_VERSION: 7,
            MessageType.STAT_UTILITY: 4.2,
            MessageType.WALL_SEND_TS: wall_send_ts,
        }

        agg._record_returned_trainer_props(ch, "slow", msg, recv_ts)

        assert ch.get_end_property("slow", PROP_STAT_UTILITY) == 4.2
        assert ch.get_end_property("slow", PROP_CLIENT_TASK_TRAIN_DURATION).total_seconds() == 12.0

    def test_real_falls_back_to_recv_minus_dispatch_without_wall_send(self):
        from datetime import datetime, timedelta

        agg = self._make_agg(simulated=False)
        ch = FakeSyncChannel({}, arrival_order=[])
        send_ts = datetime(2026, 1, 1, 0, 0, 0)
        recv_ts = send_ts + timedelta(seconds=18.0)
        agg._oort_sent_version_ts = {"slow": {7: send_ts}}
        msg = {MessageType.MODEL_VERSION: 7, MessageType.STAT_UTILITY: 4.2}

        agg._record_returned_trainer_props(ch, "slow", msg, recv_ts)

        assert ch.get_end_property("slow", PROP_STAT_UTILITY) == 4.2
        assert ch.get_end_property("slow", PROP_CLIENT_TASK_TRAIN_DURATION).total_seconds() == 18.0

    def test_sim_records_utility_and_duration_for_stale(self):
        agg = self._make_agg(simulated=True)
        ch = FakeSyncChannel({}, arrival_order=[])
        msg = {
            MessageType.MODEL_VERSION: 7,
            MessageType.STAT_UTILITY: 4.2,
            MessageType.SIM_CLIENT_TASK_TRAIN_DURATION_S: 18.0,
        }

        agg._record_returned_trainer_props(ch, "slow", msg, None)

        assert ch.get_end_property("slow", PROP_STAT_UTILITY) == 4.2
        assert ch.get_end_property("slow", PROP_CLIENT_TASK_TRAIN_DURATION).total_seconds() == 18.0

    def test_no_utility_field_is_safe(self):
        agg = self._make_agg(simulated=True)
        ch = FakeSyncChannel({}, arrival_order=[])
        msg = {MessageType.MODEL_VERSION: 7, MessageType.SIM_CLIENT_TASK_TRAIN_DURATION_S: 5.0}

        agg._record_returned_trainer_props(ch, "x", msg, None)

        assert ch.get_end_property("x", PROP_STAT_UTILITY) is None
        assert ch.get_end_property("x", PROP_CLIENT_TASK_TRAIN_DURATION).total_seconds() == 5.0


class TestStaleRejectRecordsPropsIntegration:
    """Integration regression: drive the REAL OortTopAggregator._aggregate_weights
    stale-reject path end to end (oort + refl share this stack) and assert the
    rejected trainer's speed + utility ARE recorded into the selector's memory while
    the stale update is NOT aggregated. This FAILS on the pre-fix code, which
    `continue`d before recording — leaving the trainer PROP_STAT_UTILITY=None
    (selector reads as unexplored, re-picks forever). Complements the unit-level
    TestStaleTrainerPropsRecorded by guarding the call-site wiring."""

    def _build_agg(self):
        from datetime import datetime, timedelta
        from flame.mode.horizontal.oort.top_aggregator import (
            TopAggregator as OortTopAggregator,
        )

        sent_ts = datetime(2026, 1, 1)
        # Client computed 12s then sent; the aggregator READ the stale straggler only
        # at +18s (6s of server-side read-wait while it sat unread until this round
        # drained the buffer). The recorded client task-train duration must be the 12s
        # the client took, not the 18s read latency.
        wall_send_ts = sent_ts.timestamp() + 12.0
        recv_ts = sent_ts + timedelta(seconds=18.0)

        class _Selector:
            def __init__(self):
                self.selected_ends = {"slow"}
                self.ordered_updates_recv_ends = []

        class _Chan:
            def __init__(self, selector):
                self._selector = selector
                self._props = defaultdict(dict)
                self._delivered = False

            def get_end_property(self, end, key):
                return self._props[end].get(key)

            def set_end_property(self, end, key, val):
                self._props[end][key] = val

            def recv_fifo(self, end_ids, first_k=0, timeout=None):
                # Deliver one STALE update (version 7 < round 11) exactly once.
                if not self._delivered and "slow" in set(end_ids):
                    self._delivered = True
                    msg = {
                        MessageType.MODEL_VERSION: 7,
                        MessageType.STAT_UTILITY: 4.2,
                        MessageType.WEIGHTS: "w",
                        MessageType.WALL_SEND_TS: wall_send_ts,
                    }
                    yield (msg, ("slow", recv_ts))
                yield (None, ("", None))  # nothing more ready

            def cleanup_recvd_ends(self):
                for e in self._selector.ordered_updates_recv_ends:
                    self._selector.selected_ends.discard(e)
                self._selector.ordered_updates_recv_ends = []

        class _CM:
            def __init__(self, chan):
                self._chan = chan

            def get_by_tag(self, tag):
                return self._chan

        class _Optimizer:
            # No stale_update_max attr -> standard oort: reject ALL stale updates.
            def do(self, weights, cache, total=0):
                return weights

        class _SelCfg:
            kwargs = {"aggr_num": 10}

        class _HP:
            # Real-recv timeout block reads these (getattr-with-default in prod);
            # max_experiment_runtime_s=None ⇒ recv timeout falls back to the stall
            # default, no budget cap. (Was missing → AttributeError after the
            # oort/syncfl real-recv timeout landed.)
            trainer_recv_wall_timeout_s = 90.0
            max_experiment_runtime_s = None

        class _Config:
            selector = _SelCfg()
            hyperparameters = _HP()

        class _ConcreteOortAgg(OortTopAggregator):
            def check_and_sleep(self): pass
            def evaluate(self): pass
            def initialize(self): pass
            def load_data(self): pass
            def train(self): pass
            def _compute_aggregator_stats(self): pass
            def _reset_aggregator_stats(self): pass
            def _update_model(self): pass

        agg = _ConcreteOortAgg.__new__(_ConcreteOortAgg)
        agg.simulated = False
        agg._round = 11
        sel = _Selector()
        chan = _Chan(sel)
        agg.cm = _CM(chan)
        agg.config = _Config()
        agg.optimizer = _Optimizer()
        agg.weights = "W"
        agg.cache = {}
        agg._updates_recevied = {}
        agg._oort_sent_version_ts = {"slow": {7: sent_ts}}
        return agg, chan, sel

    def test_stale_reject_records_speed_and_utility_but_not_aggregated(self):
        agg, chan, sel = self._build_agg()

        agg._aggregate_weights("tag")

        # The fix: a stale-but-returned trainer's speed + utility are recorded into
        # the selector's memory (pre-fix: both None -> stuck "unexplored" forever).
        assert chan.get_end_property("slow", PROP_STAT_UTILITY) == 4.2
        rd = chan.get_end_property("slow", PROP_CLIENT_TASK_TRAIN_DURATION)
        # client task-train duration = WALL_SEND_TS - dispatch (12s), excluding the
        # 6s aggregator read-wait that recv_ts would have added.
        assert rd is not None and rd.total_seconds() == 12.0
        # ...while the stale update itself was NOT aggregated (dropped from the model)
        # and the trainer was cleaned out of the in-flight set.
        assert agg.cache == {}
        assert "slow" not in sel.selected_ends


# --- U6 real barrier-anchored visibility lag (feddance/fedavg sync barrier) ----
from flame.mode.horizontal.syncfl.top_aggregator import TopAggregator as _SyncAgg


def test_barrier_anchored_lags_basic():
    """lag_i = max_completion - completion_i: the slowest (barrier-setter) has
    lag 0, early finishers wait the gap. This is the within-round spread real's
    per-message `now-arrival` metric was blind to (feddance U6, sim 15.5 vs 0.02)."""
    durs = [13.0, 11.0, 2.0, 56.0]  # 56 = slowest = barrier
    lags = _SyncAgg._barrier_anchored_lags(durs)
    assert lags == [43.0, 45.0, 54.0, 0.0]
    assert min(lags) == 0.0  # barrier-setter always 0 (proves not a past-dating shift)


def test_barrier_anchored_lags_none_safe():
    # Missing WALL_SEND_TS -> None passes through; barrier ignores Nones.
    assert _SyncAgg._barrier_anchored_lags([None, 5.0, 20.0]) == [None, 15.0, 0.0]
    assert _SyncAgg._barrier_anchored_lags([None, None]) == [None, None]
    assert _SyncAgg._barrier_anchored_lags([]) == []


def test_barrier_anchored_lags_matches_sim_spread():
    """Mean of the anchored lags equals (max - mean) of completions — the same
    quantity sim reports as vclock-sct (barrier - sct)."""
    import numpy as np
    durs = [2.0, 5.0, 7.0, 9.0, 13.0, 14.0, 16.0, 17.0, 25.0, 27.0]
    lags = _SyncAgg._barrier_anchored_lags(durs)
    assert abs(np.mean(lags) - (max(durs) - np.mean(durs))) < 1e-9
