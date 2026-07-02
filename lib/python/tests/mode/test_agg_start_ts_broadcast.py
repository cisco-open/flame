# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Batch 3 T3.0: broadcast the aggregator's trace-read origin to trainers.

Real mode's `_avail_now()`/`_sim_now()`-equivalent needs a SHARED origin
between aggregator and trainer, exactly like sim's shared `_vclock` instance.
Before this, a real-mode trainer had no way to know `agg_start_time_ts` (the
join-barrier-reanchored origin, see test_join_barrier_reanchor.py) at all, so
any trainer-side wall-clock trace lookup would have had to derive its own
origin (e.g. its own process-start time) -- reintroducing a per-trainer
join-ramp-style skew, the same class of bug as B2.0.3 just moved to the
trainer side. `MessageType.AGG_START_TS` closes that: the aggregator stamps
it on every real-mode dispatch (mirroring how `SIM_SEND_TS` is stamped for
sim), and the trainer caches it, using it as the origin for `_sim_now()`'s
real-mode branch instead of `trainer_start_ts`.
"""

from __future__ import annotations

import types

import pytest

from flame.mode.message import MessageType


# ── Minimal channel mocks (mirrors tests/mode/test_async_staggered_redispatch.py) ──

class _DistChannel:
    """Records every dispatched payload/property, no real transport."""

    def __init__(self, ends):
        self._ends = {e: None for e in ends}
        self.properties = {}
        self.props = {}
        self.sent = {}
        self._selector = types.SimpleNamespace(selected_ends=set())

    def await_join(self):
        pass

    def ends(self, state, task=None, **kw):
        return list(self._ends.keys())

    def dumps(self, msg):
        return dict(msg)

    def send_payload(self, end, payload):
        self.sent[end] = payload

    def set_end_property(self, end, key, value):
        self.props.setdefault(end, {})[key] = value

    def set_curr_unavailable_trainers(self, trainer_unavail_list=None):
        self.unavail = list(trainer_unavail_list or [])


def _stub_common(agg, channel, *, simulated):
    agg.simulated = simulated
    agg.agg_start_time_ts = 1_700_000_000.0
    agg._round = 3
    agg.trainer_event_dict = None  # gate off: skip the oracular unavailable-list path
    agg.cm = types.SimpleNamespace(get_by_tag=lambda tag: channel)
    agg._await_min_trainers = lambda ch: None
    agg._update_weights = lambda: None
    agg._inject_oracle_utilities = lambda ch, task: None
    agg._avail_stamp_end_states = lambda ch: None
    agg._avail_now = lambda: 0.0
    agg._sim_abandon_stalled = lambda ch: None
    agg._sim_evict_unavail_inflight = lambda ch: None
    agg.datasampler = types.SimpleNamespace(get_metadata=lambda r, e: {})
    agg.weights = {}


def _make_syncfl_agg(channel, *, simulated):
    from flame.mode.horizontal.syncfl.top_aggregator import TopAggregator

    class _Concrete(TopAggregator):
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

    agg = _Concrete.__new__(_Concrete)
    _stub_common(agg, channel, simulated=simulated)
    agg.config = types.SimpleNamespace(
        hyperparameters=types.SimpleNamespace(aggregation_goal=1)
    )
    return agg


def _make_oort_agg(channel, *, simulated):
    from flame.mode.horizontal.oort.top_aggregator import TopAggregator

    class _Concrete(TopAggregator):
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

    agg = _Concrete.__new__(_Concrete)
    _stub_common(agg, channel, simulated=simulated)
    agg.config = types.SimpleNamespace(
        selector=types.SimpleNamespace(kwargs={"aggr_num": 1}),
        hyperparameters=types.SimpleNamespace(sim_inflight_residence=False),
    )
    return agg


@pytest.fixture(autouse=True)
def _identity_weights(monkeypatch):
    import flame.mode.horizontal.syncfl.top_aggregator as syncfl_mod
    import flame.mode.horizontal.oort.top_aggregator as oort_mod

    monkeypatch.setattr(syncfl_mod, "weights_to_device", lambda w, d: w)
    monkeypatch.setattr(oort_mod, "weights_to_device", lambda w, d: w)


class TestAggregatorBroadcastsOrigin:
    """Every real-mode dispatch carries AGG_START_TS; sim never does."""

    def test_syncfl_real_mode_stamps_agg_start_ts(self):
        ch = _DistChannel(["e1", "e2"])
        agg = _make_syncfl_agg(ch, simulated=False)
        agg._distribute_weights("tag", "train")
        assert ch.sent["e1"][MessageType.AGG_START_TS] == agg.agg_start_time_ts
        assert ch.sent["e2"][MessageType.AGG_START_TS] == agg.agg_start_time_ts
        assert MessageType.SIM_SEND_TS not in ch.sent["e1"]

    def test_syncfl_sim_mode_never_stamps_agg_start_ts(self):
        from flame.sim import VirtualClock

        ch = _DistChannel(["e1", "e2"])
        agg = _make_syncfl_agg(ch, simulated=True)
        agg._vclock = VirtualClock()
        agg._vclock.advance(42.0)
        agg._distribute_weights("tag", "train")
        assert MessageType.AGG_START_TS not in ch.sent["e1"]
        assert ch.sent["e1"][MessageType.SIM_SEND_TS] == 42.0

    def test_oort_real_mode_stamps_agg_start_ts(self):
        ch = _DistChannel(["e1", "e2"])
        agg = _make_oort_agg(ch, simulated=False)
        agg._distribute_weights("tag", "train")
        assert ch.sent["e1"][MessageType.AGG_START_TS] == agg.agg_start_time_ts
        assert MessageType.SIM_SEND_TS not in ch.sent["e1"]

    def test_oort_sim_mode_never_stamps_agg_start_ts(self):
        from flame.sim import VirtualClock

        ch = _DistChannel(["e1", "e2"])
        agg = _make_oort_agg(ch, simulated=True)
        agg._vclock = VirtualClock()
        agg._vclock.advance(17.0)
        agg._distribute_weights("tag", "train")
        assert MessageType.AGG_START_TS not in ch.sent["e1"]
        assert ch.sent["e1"][MessageType.SIM_SEND_TS] == 17.0

    def test_asyncfl_real_mode_stamps_agg_start_ts(self):
        # asyncfl's staggered path is sim-only (_staggered requires
        # self.simulated); real mode always takes the shared/batched branch,
        # so it's enough to exercise that one.
        from flame.mode.horizontal.asyncfl.top_aggregator import TopAggregator
        from flame.sim import VirtualClock

        class _Concrete(TopAggregator):
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

        ch = _DistChannel(["e1", "e2"])
        agg = _Concrete.__new__(_Concrete)
        _stub_common(agg, ch, simulated=False)
        agg._vclock = VirtualClock()
        agg._sim_staggered_redispatch = False
        agg._sim_inflight_residence = False
        agg._sim_free_slot_ts = __import__("collections").deque(maxlen=128)
        agg._sim_last_commit_sct = {}
        agg._sim_inflight_expected = {}
        agg._sim_trainer_budget = {}
        agg._sim_budget_min = 12.0
        agg._sim_redispatch_gap_s = 0.0
        agg._sim_cooldown_until = {}
        agg._real_distribute_settle_s = 0.0
        agg._track_trainer_version_duration_s = {}
        agg.weights = {}
        agg._distribute_weights("tag", "train")
        assert ch.sent["e1"][MessageType.AGG_START_TS] == agg.agg_start_time_ts


class TestTrainerCachesOrigin:
    """syncfl/trainer.py's message handling caches AGG_START_TS (real mode)."""

    def _make_trainer(self):
        from flame.mode.horizontal.syncfl.trainer import Trainer

        class _Concrete(Trainer):
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

        t = _Concrete.__new__(_Concrete)
        t.trainer_id = "trainer-under-test"
        t._round = 0
        t._work_done = False
        t.task_to_perform = "train"
        t._agg_start_origin = None
        return t

    def _fetch(self, trainer, msg):
        """Drive _fetch_weights with a minimal channel (no WEIGHTS payload,
        so the torch-dependent model-loading branch is never entered)."""
        selector = types.SimpleNamespace(ordered_updates_recv_ends=[])
        channel = types.SimpleNamespace(
            _selector=selector,
            await_join=lambda: None,
            one_end=lambda state: "agg-end",
            recv=lambda end: (msg, None),
            cleanup_recvd_ends=lambda: None,
        )
        trainer.cm = types.SimpleNamespace(get_by_tag=lambda tag: channel)
        trainer._fetch_weights("tag")

    def test_caches_agg_start_ts_from_message(self):
        t = self._make_trainer()
        origin = 1_700_000_123.456
        self._fetch(t, {MessageType.ROUND: 1, MessageType.AGG_START_TS: origin})
        assert t._agg_start_origin == origin

    def test_no_agg_start_ts_in_message_leaves_cache_untouched(self):
        t = self._make_trainer()
        t._agg_start_origin = 111.0
        self._fetch(t, {MessageType.ROUND: 1})
        assert t._agg_start_origin == 111.0  # unchanged, not reset to None

    def test_re_caches_on_every_dispatch(self):
        t = self._make_trainer()
        self._fetch(t, {MessageType.ROUND: 1, MessageType.AGG_START_TS: 100.0})
        assert t._agg_start_origin == 100.0
        self._fetch(t, {MessageType.ROUND: 2, MessageType.AGG_START_TS: 100.0})
        assert t._agg_start_origin == 100.0


# PyTorchCifar10Trainer._sim_now()'s real-mode branch is example-specific code
# (examples/async_cifar10/trainer/pytorch/main.py), not core flame -- see
# examples/async_cifar10/trainer/pytorch/test_agg_start_ts_sim_now.py.
