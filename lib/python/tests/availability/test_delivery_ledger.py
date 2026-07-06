# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Stage C substrate: delivery ledger + slot-free hook on ClientAvailability.

These exercise the pure delivery-ledger logic (compute_delivery_ts, the
pending_withheld ordering/residence/commit helpers, and free_stalled_slot's
two-ledger effect) in isolation from the full sim aggregator. They assert the
three Stage C correctness invariants at the substrate level:
  1. no double-count / slot freed exactly once,
  2. a still-down trainer is excluded until its delivery_ts (never re-selected),
  3. withheld delivery commits in delivery_ts order (no past-dating).
And the gate-off no-op that preserves byte-identity.
"""

from sortedcontainers import SortedDict

from flame.availability.client_availability import ClientAvailability


def _trace(*pairs):
    d = SortedDict()
    for ts, state in pairs:
        d[float(ts)] = state
    return d


class _FakeSelector:
    def __init__(self):
        self.requester = "agg"
        self.all_selected = {}
        self.selected_ends = {"agg": set()}


class _FakeEnd:
    def __init__(self):
        self.props = {}

    def set_property(self, k, v):
        self.props[k] = v


class _FakeChannel:
    def __init__(self, ends):
        self._selector = _FakeSelector()
        self._ends = {e: _FakeEnd() for e in ends}

    def has(self, end):
        return end in self._ends


class _Harness(ClientAvailability):
    """Minimal mixin host with a controllable clock and gate state."""

    def __init__(self, trainer_event_dict=None, now=0.0):
        self.trainer_event_dict = trainer_event_dict
        self.pending_withheld = {}
        self._now = now

    def _avail_now(self):  # override the vclock/wall source
        return self._now


# down window: [100, 200); recovers (AVL_TRAIN) at 200.
_DOWN = _trace((0, "AVL_TRAIN"), (100, "UN_AVL"), (200, "AVL_TRAIN"))


def test_compute_delivery_ts_in_down_window():
    h = _Harness({"t1": _DOWN}, now=130)
    # update completed at sct=150 while down; delivers when AVL again (200).
    assert h.compute_delivery_ts("t1", sct=150) == 200.0


def test_compute_delivery_ts_clamped_to_sct():
    # next_avail (200) is earlier than a late sct (260): delivery cannot
    # precede completion, so delivery_ts = sct.
    h = _Harness({"t1": _DOWN}, now=130)
    assert h.compute_delivery_ts("t1", sct=260) == 260.0


def test_compute_delivery_ts_never_recovers_is_inf():
    never = _trace((0, "AVL_TRAIN"), (100, "UN_AVL"))
    h = _Harness({"t1": never}, now=130)
    assert h.compute_delivery_ts("t1", sct=150) == float("inf")


def test_compute_delivery_ts_gate_off_returns_sct():
    h = _Harness(trainer_event_dict=None, now=130)
    assert h.compute_delivery_ts("t1", sct=150) == 150.0


def test_free_stalled_slot_frees_slot_and_registers_delivery():
    h = _Harness({"t1": _DOWN}, now=150)
    ch = _FakeChannel(["t1"])
    ch._selector.all_selected["t1"] = 12345.0
    ch._selector.selected_ends["agg"].add("t1")

    dts = h.free_stalled_slot(ch, "t1", reason="abandon_90s", sct=150)

    assert dts == 200.0
    # slot ledger: released
    assert "t1" not in ch._selector.all_selected
    assert "t1" not in ch._selector.selected_ends["agg"]
    assert ch._ends["t1"].props["state"] == "none"
    # delivery ledger: registered
    assert h.pending_withheld == {"t1": 200.0}


def test_free_stalled_slot_gate_off_is_noop():
    h = _Harness(trainer_event_dict=None, now=150)
    ch = _FakeChannel(["t1"])
    ch._selector.all_selected["t1"] = 12345.0
    assert h.free_stalled_slot(ch, "t1", reason="x", sct=150) is None
    # nothing freed, nothing registered (byte-identity preserved)
    assert ch._selector.all_selected == {"t1": 12345.0}
    assert h.pending_withheld == {}


def test_withheld_held_ends_excludes_until_delivery():
    h = _Harness({"t1": _DOWN}, now=150)
    h.pending_withheld = {"t1": 200.0, "t2": 180.0}
    # before either delivery_ts: both held (invariant 2)
    h._now = 170
    assert h.withheld_held_ends() == {"t1", "t2"}
    # past t2's delivery_ts: only t1 still held
    h._now = 190
    assert h.withheld_held_ends() == {"t1"}
    # past both: none held
    h._now = 250
    assert h.withheld_held_ends() == set()


def test_ready_withheld_orders_by_delivery_ts_then_id():
    h = _Harness({"t1": _DOWN}, now=300)
    h.pending_withheld = {"t3": 200.0, "t1": 200.0, "t2": 150.0}
    # all due at now=300; order by (delivery_ts, end_id): t2,150 < t1,200 < t3,200
    assert h.ready_withheld() == [("t2", 150.0), ("t1", 200.0), ("t3", 200.0)]


def test_ready_withheld_excludes_future():
    h = _Harness({"t1": _DOWN}, now=190)
    h.pending_withheld = {"t1": 200.0, "t2": 150.0}
    assert h.ready_withheld() == [("t2", 150.0)]


def test_commit_withheld_pops():
    h = _Harness({"t1": _DOWN}, now=200)
    h.pending_withheld = {"t1": 200.0}
    assert h.commit_withheld("t1") == 200.0
    assert h.pending_withheld == {}
    assert h.commit_withheld("t1") is None  # idempotent: no double-count
