# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Batch 3 T3.4: trainer-side [SEND_GATE] delay decomposition telemetry.

`_send_weights`'s UN_AVL wait loop (syncfl/trainer.py) is the actual
enforcement point of the real-mode send-time gate (T3.1a/T3.1b landed the
mechanism; this task instruments it). Two new fields on the `task_send`
event -- not `trainer_round` (task/train() emits trainer_round BEFORE
put()/_send_weights runs, per the tasklet composition in `Trainer.compose`,
so `_phase_times` merged there can never see this loop's timing for the
same round; task_send already exists for exactly this ordering reason, see
`build_task_send`'s own docstring re: wall_send_ts):
  - `send_gate_sct`: the trainer's own trace-time-basis clock (`_sim_now()`)
    sampled right before the gate check, real mode only, regardless of
    whether the gate actually engages.
  - `send_gate_wait_s`: wall-time actually spent blocked in the loop (0.0
    when the gate never engaged), real mode only; always None in sim mode.
"""

import types

import pytest

from flame.config import TrainerAvailState


class _FakeClock:
    """Deterministic time.time()/time.sleep() so wait duration is exact."""

    def __init__(self, start=1_000.0):
        self.t = start

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s


def _make_trainer():
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
    t._round = 5
    t._phase_times = {}
    t.task_to_perform = "eval"  # skips the torch/cloudpickle weight-send path
    t._stat_utility = 0.5
    t._local_accuracy = 0.0
    t._updates_returned_upto_round = 0
    t.wait_until_next_avl = "True"
    return t


def _make_channel():
    return types.SimpleNamespace(
        await_join=lambda: None,
        one_end=lambda state: "agg-end",
        send=lambda end, msg: None,
        _selector=types.SimpleNamespace(_cleanup_send_ends=lambda: None),
    )


@pytest.fixture
def captured_events(monkeypatch):
    """Capture every telemetry.emit() call made by syncfl/trainer.py."""
    import flame.mode.horizontal.syncfl.trainer as trainer_mod

    events: list = []
    monkeypatch.setattr(trainer_mod.telemetry, "is_enabled", lambda: True)
    monkeypatch.setattr(
        trainer_mod.telemetry, "emit",
        lambda ev, **fields: events.append((ev, fields)),
    )
    return events


def _task_send_fields(events):
    for ev, fields in events:
        if ev == "task_send":
            return fields
    raise AssertionError("no task_send event emitted")


class TestSendGateWaitRealMode:
    def test_waits_and_stamps_observed_duration(self, monkeypatch, captured_events):
        import flame.mode.horizontal.syncfl.trainer as trainer_mod

        t = _make_trainer()
        t.simulated = False
        t.avl_state = TrainerAvailState.UN_AVL
        t._sim_now = lambda: 500.0
        t.cm = types.SimpleNamespace(get_by_tag=lambda tag: _make_channel())

        clock = _FakeClock()
        calls = {"n": 0}

        def _sleep(s):
            clock.sleep(s)
            calls["n"] += 1
            if calls["n"] >= 3:
                t.avl_state = TrainerAvailState.AVL_TRAIN

        monkeypatch.setattr(trainer_mod.time, "time", clock.time)
        monkeypatch.setattr(trainer_mod.time, "sleep", _sleep)

        t._send_weights("tag")

        fields = _task_send_fields(captured_events)
        assert fields["send_gate_sct"] == 500.0
        assert fields["send_gate_wait_s"] == pytest.approx(3.0)
        assert calls["n"] == 3

    def test_no_wait_when_already_available(self, monkeypatch, captured_events):
        import flame.mode.horizontal.syncfl.trainer as trainer_mod

        t = _make_trainer()
        t.simulated = False
        t.avl_state = TrainerAvailState.AVL_TRAIN
        t._sim_now = lambda: 42.0
        t.cm = types.SimpleNamespace(get_by_tag=lambda tag: _make_channel())

        def _sleep(_s):
            raise AssertionError("sleep should never be called when already available")

        monkeypatch.setattr(trainer_mod.time, "sleep", _sleep)

        t._send_weights("tag")

        fields = _task_send_fields(captured_events)
        assert fields["send_gate_sct"] == 42.0
        assert fields["send_gate_wait_s"] == 0.0

    def test_missing_sim_now_leaves_sct_none(self, monkeypatch, captured_events):
        """Defensive fallback: a base-class trainer with no _sim_now() override
        (not every example wires one) must not raise -- sct is simply absent."""
        t = _make_trainer()
        t.simulated = False
        t.avl_state = TrainerAvailState.AVL_TRAIN
        assert not hasattr(t, "_sim_now")
        t.cm = types.SimpleNamespace(get_by_tag=lambda tag: _make_channel())

        t._send_weights("tag")

        fields = _task_send_fields(captured_events)
        assert fields["send_gate_sct"] is None
        assert fields["send_gate_wait_s"] == 0.0


class TestSendGateWaitSimMode:
    def test_sim_mode_never_stamps_gate_fields(self, monkeypatch, captured_events):
        import flame.mode.horizontal.syncfl.trainer as trainer_mod

        t = _make_trainer()
        t.simulated = True
        t.avl_state = TrainerAvailState.UN_AVL  # would gate in real mode; sim ignores it
        t._sim_now = lambda: 999.0
        t.cm = types.SimpleNamespace(get_by_tag=lambda tag: _make_channel())

        def _sleep(_s):
            raise AssertionError("sim mode must never enter the wall-sleep gate loop")

        monkeypatch.setattr(trainer_mod.time, "sleep", _sleep)

        t._send_weights("tag")

        fields = _task_send_fields(captured_events)
        assert fields["send_gate_sct"] is None
        assert fields["send_gate_wait_s"] is None
