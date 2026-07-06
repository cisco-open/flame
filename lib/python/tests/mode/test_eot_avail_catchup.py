# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Batch 4 finding 2 (UNAVAILABILITY_DESIGN.md): a sim-mode trainer that goes
quiet (correctly withheld/evicted as UN_AVL, so never dispatched again) has a
frozen `_sim_now()` and so can never catch its `avail_change` telemetry up to
later trace transitions on its own. `inform_end_of_training`'s existing
`channel.broadcast(...)` already reaches every connected end regardless of
dispatch state, so it now piggybacks the aggregator's final `_avail_now()`
(gated on sim mode + the availability feature being on, so the broadcast
payload is byte-identical when off) -- one last wake-up letting
`_refresh_avl_state()` flush any queued transitions before the trainer exits.

Two halves, mirroring test_agg_start_ts_broadcast.py's structure:
  * aggregator side: inform_end_of_training's broadcast payload.
  * trainer side: _fetch_weights calls _refresh_avl_state() on EOT receipt.
"""

from __future__ import annotations

import types

import pytest

from flame.mode.message import MessageType


# ---- Aggregator side -------------------------------------------------------

class _BroadcastChannel:
    """Records the last broadcast payload, no real transport."""

    def __init__(self):
        self.broadcasts = []

    def broadcast(self, payload):
        self.broadcasts.append(dict(payload))


def _make_syncfl_agg(*, simulated, gate_on, vclock_now=42.0):
    from flame.mode.horizontal.syncfl.top_aggregator import TopAggregator
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

    agg = _Concrete.__new__(_Concrete)
    agg.simulated = simulated
    agg._work_done = True
    agg.trainer_event_dict = {"e1": object()} if gate_on else None
    agg.agg_start_time_ts = 1_700_000_000.0
    channel = _BroadcastChannel()
    agg.cm = types.SimpleNamespace(get_by_tag=lambda tag: channel)
    agg.dist_tag = "tag"
    if simulated:
        agg._vclock = VirtualClock()
        agg._vclock.advance(vclock_now)
    return agg, channel


class TestInformEndOfTrainingCarriesFinalClock:
    def test_sim_mode_gate_on_carries_sim_send_ts(self):
        agg, ch = _make_syncfl_agg(simulated=True, gate_on=True, vclock_now=77.0)
        agg.inform_end_of_training()
        assert ch.broadcasts[0][MessageType.SIM_SEND_TS] == 77.0
        assert ch.broadcasts[0][MessageType.EOT] is True

    def test_sim_mode_gate_off_omits_sim_send_ts(self):
        """Byte-identical broadcast payload when the availability feature is off."""
        agg, ch = _make_syncfl_agg(simulated=True, gate_on=False)
        agg.inform_end_of_training()
        assert MessageType.SIM_SEND_TS not in ch.broadcasts[0]
        assert ch.broadcasts[0] == {MessageType.EOT: True}

    def test_real_mode_omits_sim_send_ts(self):
        """Real mode's clock never freezes -- no catch-up broadcast needed."""
        agg, ch = _make_syncfl_agg(simulated=False, gate_on=True)
        agg.inform_end_of_training()
        assert MessageType.SIM_SEND_TS not in ch.broadcasts[0]
        assert ch.broadcasts[0] == {MessageType.EOT: True}


# ---- Trainer side -----------------------------------------------------------

class TestTrainerFlushesCatchupOnEot:
    def _make_trainer(self, *, with_refresh_hook):
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
        if with_refresh_hook:
            t._refresh_calls = 0

            def _refresh():
                t._refresh_calls += 1

            t._refresh_avl_state = _refresh
        return t

    def _fetch(self, trainer, msg):
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

    def test_eot_triggers_refresh_when_hook_present(self):
        t = self._make_trainer(with_refresh_hook=True)
        self._fetch(t, {MessageType.EOT: True, MessageType.SIM_SEND_TS: 900.0})
        assert t._work_done is True
        assert t._refresh_calls == 1

    def test_eot_without_hook_is_safe_noop(self):
        """Not every Trainer subclass defines _refresh_avl_state (example-
        specific) -- the hasattr guard must not raise for those."""
        t = self._make_trainer(with_refresh_hook=False)
        assert not hasattr(t, "_refresh_avl_state")
        self._fetch(t, {MessageType.EOT: True})  # must not raise
        assert t._work_done is True

    def test_non_eot_message_does_not_trigger_refresh(self):
        t = self._make_trainer(with_refresh_hook=True)
        self._fetch(t, {MessageType.ROUND: 3})
        assert t._refresh_calls == 0
