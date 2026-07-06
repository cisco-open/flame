# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Batch 3 T3.1b: `_refresh_avl_state()` transitions correctly in real mode,
anchored to the T3.0-broadcast origin -- one code path for both modes now,
not a sim-only gated wrapper plus a separate always-on background thread.

Regression target: `_refresh_avl_for_sim` (the old name) was a real no-op in
real mode (`if not self.simulated: return`). It turned out not to be why real
trainers never tracked the trace live (that was `debug_run.sh` never wiring
`client_notify.trace` -- Challenges §5 item 20 / T3.1a), but it was still
real dead code. This test would have failed under the old implementation.
"""

from flame.config import TrainerAvailState
from trainer.pytorch.main import PyTorchCifar10Trainer


def _make_trainer(*, simulated, agg_start_origin, now, events):
    t = PyTorchCifar10Trainer.__new__(PyTorchCifar10Trainer)
    t.simulated = simulated
    t._sim_send_ts = now if simulated else None
    t._agg_start_origin = agg_start_origin
    t.trainer_start_ts = 0.0
    t.trainer_id = "trainer-under-test"
    t._round = 0
    t.cm = object()  # only needs to be truthy/non-None
    t.state_avl_event_ts = list(events)
    t.avl_state = TrainerAvailState.AVL_TRAIN
    t.client_notify = {"enabled": "False"}
    return t


class TestRefreshAvlStateRealMode:
    def test_pops_a_due_transition(self):
        import time as _time

        now = _time.time()
        origin = now - 30.0  # 30s of trace-elapsed time
        t = _make_trainer(
            simulated=False,
            agg_start_origin=origin,
            now=None,
            events=[(20.0, "UN_AVL"), (60.0, "AVL_TRAIN")],
        )
        t._refresh_avl_state()
        assert t.avl_state == TrainerAvailState.UN_AVL
        assert t.state_avl_event_ts == [(60.0, "AVL_TRAIN")]  # only the due one popped

    def test_catches_up_multiple_missed_transitions(self):
        import time as _time

        now = _time.time()
        origin = now - 100.0  # far past every scheduled transition below
        t = _make_trainer(
            simulated=False,
            agg_start_origin=origin,
            now=None,
            events=[(10.0, "UN_AVL"), (20.0, "AVL_TRAIN"), (30.0, "UN_AVL")],
        )
        t._refresh_avl_state()
        assert t.avl_state == TrainerAvailState.UN_AVL  # last of the three due events
        assert t.state_avl_event_ts == []

    def test_no_due_transition_leaves_state_untouched(self):
        import time as _time

        now = _time.time()
        origin = now - 5.0  # only 5s of trace-elapsed time
        t = _make_trainer(
            simulated=False,
            agg_start_origin=origin,
            now=None,
            events=[(600.0, "UN_AVL")],
        )
        t._refresh_avl_state()
        assert t.avl_state == TrainerAvailState.AVL_TRAIN
        assert t.state_avl_event_ts == [(600.0, "UN_AVL")]

    def test_real_mode_is_not_a_no_op_regression_guard(self):
        """The historical bug: real mode never called through to
        check_and_update_state_avl at all. Assert it's reachable now."""
        import time as _time

        now = _time.time()
        t = _make_trainer(
            simulated=False,
            agg_start_origin=now - 1000.0,
            now=None,
            events=[(1.0, "UN_AVL")],
        )
        assert t.avl_state == TrainerAvailState.AVL_TRAIN
        t._refresh_avl_state()
        assert t.avl_state != TrainerAvailState.AVL_TRAIN, (
            "real-mode _refresh_avl_state() must not be a no-op"
        )
