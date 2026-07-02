# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Batch 4 finding 2: `avail_change.sim_now` must be the transition's own
scheduled trace-time, not `_sim_now()` at processing time.

Regression target: sim-mode `_sim_now()` returns the frozen `_sim_send_ts`
from the trainer's last dispatch. A trainer that goes quiet (correctly
withheld/evicted as UN_AVL, so no longer dispatched) has a stale `_sim_now()`
for as long as it stays quiet -- if a *later* catch-up call stamped
`sim_now=self._sim_now()`, every transition flushed in that catch-up would be
mis-recorded at the stale/late instant instead of its true trace time. Using
the transition's own `state_avl_event_ts[0][0]` (which the trainer already
has locally, loaded from the trace at init) instead is correct regardless of
when the catch-up actually happens. See UNAVAILABILITY_DESIGN.md, Batch 4
finding 2, and test_refresh_avl_state.py for the sibling real-mode coverage.
"""

from flame.config import TrainerAvailState
from trainer.pytorch.main import PyTorchCifar10Trainer


def _make_trainer(*, simulated, sim_send_ts, events):
    t = PyTorchCifar10Trainer.__new__(PyTorchCifar10Trainer)
    t.simulated = simulated
    t._sim_send_ts = sim_send_ts
    t._agg_start_origin = None
    t.trainer_start_ts = 0.0
    t.trainer_id = "trainer-under-test"
    t._round = 7
    t.cm = object()  # only needs to be truthy/non-None
    t.state_avl_event_ts = list(events)
    t.avl_state = TrainerAvailState.AVL_TRAIN
    t.client_notify = {"enabled": "False"}
    return t


class TestAvailChangeSimNowIsDueTs:
    def test_sim_now_uses_transition_due_ts_not_processing_time(self, monkeypatch):
        """Catch-up long after the transition was due still records its TRUE
        trace time, not the (much later) instant it happened to be noticed."""
        import trainer.pytorch.main as main_mod

        captured = []
        monkeypatch.setattr(main_mod.telemetry, "is_enabled", lambda: True)
        monkeypatch.setattr(
            main_mod.telemetry, "emit",
            lambda ev, **fields: captured.append(fields),
        )

        # Transition was due at trace-time 600.0; the trainer only gets
        # around to processing it at sim_send_ts=890.0 (a very late catch-up,
        # e.g. the Batch 4 finding-1 EOT wake-up after a long idle stretch).
        t = _make_trainer(
            simulated=True, sim_send_ts=890.0,
            events=[(600.0, "UN_AVL")],
        )
        t.check_and_update_state_avl()

        assert t.avl_state == TrainerAvailState.UN_AVL
        assert len(captured) == 1
        assert captured[0]["sim_now"] == 600.0  # the transition's own due time
        assert captured[0]["sim_now"] != 890.0  # NOT processing-time

    def test_multi_transition_catchup_each_gets_its_own_due_ts(self, monkeypatch):
        """A burst catch-up (several missed transitions popped in sequence)
        stamps each event with ITS OWN due time, not the burst's shared
        processing time -- this is what lets duration-weighted fidelity
        checks (A6) reconstruct the correct timeline even when every
        transition is only discovered in one late catch-up call."""
        import trainer.pytorch.main as main_mod

        captured = []
        monkeypatch.setattr(main_mod.telemetry, "is_enabled", lambda: True)
        monkeypatch.setattr(
            main_mod.telemetry, "emit",
            lambda ev, **fields: captured.append(fields),
        )

        t = _make_trainer(
            simulated=True, sim_send_ts=901.5,
            events=[(600.0, "UN_AVL"), (750.0, "AVL_TRAIN"), (800.0, "UN_AVL")],
        )
        while t.state_avl_event_ts and t._sim_now() >= t.state_avl_event_ts[0][0]:
            t.check_and_update_state_avl()

        assert t.avl_state == TrainerAvailState.UN_AVL
        assert [c["sim_now"] for c in captured] == [600.0, 750.0, 800.0]

    def test_real_mode_unaffected_still_uses_due_ts(self, monkeypatch):
        """Real mode's clock never freezes, but the fix applies uniformly --
        due_ts is at least as correct as (and simpler than) re-deriving it
        from a live wall-clock read at processing time."""
        import trainer.pytorch.main as main_mod

        captured = []
        monkeypatch.setattr(main_mod.telemetry, "is_enabled", lambda: True)
        monkeypatch.setattr(
            main_mod.telemetry, "emit",
            lambda ev, **fields: captured.append(fields),
        )

        import time

        t = _make_trainer(simulated=False, sim_send_ts=None, events=[(30.0, "UN_AVL")])
        t._agg_start_origin = time.time() - 40.0  # 40s of trace-elapsed real time
        t.check_and_update_state_avl()

        assert t.avl_state == TrainerAvailState.UN_AVL
        assert captured[0]["sim_now"] == 30.0
