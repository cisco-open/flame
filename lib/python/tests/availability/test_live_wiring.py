# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Stage C live-wiring: the shared commit-loop primitives in ClientAvailability.

The send-gate withhold / late re-commit / vclock-abandon EFFECT is library-level
(both the asyncfl _sim_recv_min pop site and the oort _sim_drain_buffer pop loop
call the same mixin methods). These exercise that shared core directly against a
real SimReorderBuffer and both selector shapes:

  * asyncfl selectors: selected_ends = dict[requester -> set], plus all_selected
  * oort/refl/feddance: selected_ends = flat set, no all_selected

They assert the three Stage C invariants at the wiring level:
  1. no double-count — an abandoned-then-arrived update commits exactly once,
  2. a still-down trainer is excluded until delivery_ts (held in the ledger),
  3. withheld delivery re-injects in delivery_ts order (no past-dating).
Plus the gate-off byte-identity no-op and the Challenge-7 composition contract
(why oort must gate carry-over BEFORE the send-gate).
"""

import math

from sortedcontainers import SortedDict

from flame.availability.client_availability import ClientAvailability
from flame.config import TrainerAvailState
from flame.selector.properties import PROP_AVL_STATE, PROP_SIM_SEND_TS
from flame.sim import SimReorderBuffer


def _trace(*pairs):
    d = SortedDict()
    for ts, state in pairs:
        d[float(ts)] = state
    return d


# down window [100,200); recovers (AVL_TRAIN) at 200.
_DOWN = _trace((0, "AVL_TRAIN"), (100, "UN_AVL"), (200, "AVL_TRAIN"))
# late down window [300,400); used for future-sct composition.
_LATE_DOWN = _trace((0, "AVL_TRAIN"), (300, "UN_AVL"), (400, "AVL_TRAIN"))
# never recovers after going down.
_NEVER = _trace((0, "AVL_TRAIN"), (100, "UN_AVL"))


class _FakeEnd:
    def __init__(self):
        self.props = {}

    def set_property(self, k, v):
        self.props[k] = v

    def get_property(self, k):
        return self.props.get(k)


class _AsyncSelector:
    """fedbuff/async_oort shape: dict-by-requester + all_selected."""

    def __init__(self):
        self.requester = "agg"
        self.all_selected = {}
        self.selected_ends = {"agg": set()}

    def add(self, end):
        self.all_selected[end] = 0.0
        self.selected_ends["agg"].add(end)

    def holds(self, end):
        return end in self.selected_ends["agg"]


class _OortSelector:
    """oort/refl/feddance shape: flat set, no all_selected."""

    def __init__(self):
        self.selected_ends = set()

    def add(self, end):
        self.selected_ends.add(end)

    def holds(self, end):
        return end in self.selected_ends


class _Channel:
    def __init__(self, selector, ends):
        self._selector = selector
        self._ends = {e: _FakeEnd() for e in ends}
        self._props = {}

    def has(self, end):
        return end in self._ends

    def get_end_property(self, end, key):
        return self._props.get((end, key))

    def set_end_property(self, end, key, value):
        self._props[(end, key)] = value


class _Harness(ClientAvailability):
    """Minimal mixin host with a controllable clock + a real reorder buffer."""

    def __init__(self, trainer_event_dict=None, now=0.0, inflight_tracker=False):
        self.trainer_event_dict = trainer_event_dict
        # Mirrors _init_availability's derivation (production sets this once at
        # load time, not per-call) — kept in sync here so harness tests exercise
        # the same guard real runs do.
        self._trace_has_avl_eval = bool(trainer_event_dict) and any(
            TrainerAvailState.AVL_EVAL in trace.values()
            for trace in trainer_event_dict.values()
        )
        self.pending_withheld = {}
        self._sim_withheld_payload = {}
        self._sim_withheld_delivering = {}
        self._sim_buffer = SimReorderBuffer()
        self._sim_committed = set()
        self._round = 5
        self._now = now
        # asyncfl tracks an in-flight gate dict; oort does not.
        if inflight_tracker:
            self._sim_inflight_expected = {}

    def _avail_now(self):
        return self._now


def _payload(tag):
    # opaque to the helpers (only telemetry reads MODEL_VERSION, off in tests).
    return ({"tag": tag}, (tag, None))


# ---------------------------------------------------------------------------
# _sim_withhold_if_unavail — the shared per-update send-gate primitive
# ---------------------------------------------------------------------------

def test_withhold_if_unavail_holds_unavailable_at_sct():
    h = _Harness({"t1": _DOWN}, now=130, inflight_tracker=True)
    sel = _AsyncSelector(); sel.add("t1")
    ch = _Channel(sel, ["t1"])
    h._sim_inflight_expected["t1"] = 999.0
    p = _payload("t1")

    # completed at sct=150 while UN_AVL -> held, slot freed, ledger registered.
    assert h._sim_withhold_if_unavail(ch, "t1", 150.0, p) is True
    assert h.pending_withheld == {"t1": 200.0}
    assert h._sim_withheld_payload["t1"] == (150.0, p)
    assert not sel.holds("t1")                      # slot freed
    assert "t1" not in h._sim_inflight_expected     # gate tracker cleared


def test_withhold_if_unavail_commits_when_available():
    h = _Harness({"t1": _DOWN}, now=260)
    sel = _AsyncSelector(); sel.add("t1")
    ch = _Channel(sel, ["t1"])
    # sct=260 is past the down window -> available -> committable, no withhold.
    assert h._sim_withhold_if_unavail(ch, "t1", 260.0, _payload("t1")) is False
    assert h.pending_withheld == {}


def test_withhold_if_unavail_gate_off_is_noop():
    h = _Harness(trainer_event_dict=None, now=150)
    sel = _AsyncSelector(); sel.add("t1")
    ch = _Channel(sel, ["t1"])
    assert h._sim_withhold_if_unavail(ch, "t1", 150.0, _payload("t1")) is False
    assert sel.holds("t1")          # nothing freed
    assert h.pending_withheld == {}


def test_withhold_if_unavail_never_recovers_drops_payload():
    h = _Harness({"t1": _NEVER}, now=130)
    sel = _OortSelector(); sel.add("t1")
    ch = _Channel(sel, ["t1"])
    # undeliverable in-window: slot still freed + ledger registered (inf), but no
    # payload stashed (Challenge 10 edge).
    assert h._sim_withhold_if_unavail(ch, "t1", 150.0, _payload("t1")) is True
    assert h.pending_withheld == {"t1": math.inf}
    assert "t1" not in h._sim_withheld_payload
    assert not sel.holds("t1")


def test_withhold_does_not_consult_round_start():
    # Contract that makes oort's carry-over-first ordering load-bearing
    # (Challenge 7): the send-gate is a PURE per-update decision keyed on the
    # trainer's state at sct — it would withhold a still-computing future-sct
    # UN_AVL straggler if called directly. So the oort loop MUST apply carry-over
    # (still-computing) before this, or it would free a slot that is still busy.
    h = _Harness({"t1": _LATE_DOWN}, now=50)
    sel = _OortSelector(); sel.add("t1")
    ch = _Channel(sel, ["t1"])
    # sct=350 is in the future (still computing) AND UN_AVL at 350.
    assert h._sim_withhold_if_unavail(ch, "t1", 350.0, _payload("t1")) is True


# ---------------------------------------------------------------------------
# _sim_pop_committable — asyncfl loop
# ---------------------------------------------------------------------------

def test_pop_committable_skips_withheld_returns_next():
    h = _Harness({"t1": _DOWN, "t2": _DOWN}, now=130, inflight_tracker=True)
    sel = _AsyncSelector(); sel.add("t1"); sel.add("t2")
    ch = _Channel(sel, ["t1", "t2"])
    # t1 completes UN_AVL at 150 (held); t2 completes available at 260 (commit).
    h._sim_buffer.add("t1", 150.0, _payload("t1"))
    h._sim_buffer.add("t2", 260.0, _payload("t2"))

    popped = h._sim_pop_committable(ch)
    assert popped is not None and popped[0] == "t2"
    assert "t1" in h.pending_withheld and not sel.holds("t1")
    # t2 is the only committable; buffer now empty.
    assert h._sim_pop_committable(ch) is None


def test_pop_committable_gate_off_single_pop():
    h = _Harness(trainer_event_dict=None, now=0)
    sel = _AsyncSelector(); sel.add("a")
    ch = _Channel(sel, ["a"])
    h._sim_buffer.add("a", 10.0, _payload("a"))
    popped = h._sim_pop_committable(ch)
    assert popped[0] == "a" and popped[1] == 10.0
    assert h.pending_withheld == {}


# ---------------------------------------------------------------------------
# _sim_reinject_ready_withheld — ordering, residence, slot-only drop
# ---------------------------------------------------------------------------

def test_reinject_adds_due_in_order_marks_delivering():
    h = _Harness({"t1": _DOWN}, now=200)
    h.pending_withheld = {"t1": 200.0, "t2": 150.0}
    p1, p2 = _payload("t1"), _payload("t2")
    h._sim_withheld_payload = {"t1": (150.0, p1), "t2": (120.0, p2)}

    h._sim_reinject_ready_withheld()
    # both due at now=200 -> both re-injected, ledger emptied, marked delivering.
    assert h.pending_withheld == {}
    assert h._sim_buffer.has("t1") and h._sim_buffer.has("t2")
    assert h._sim_withheld_delivering["t1"] == (150.0, 200.0)
    assert h._sim_withheld_delivering["t2"] == (120.0, 150.0)
    # re-injected keyed at delivery_ts (commit order), so t2(150) pops before t1(200).
    assert h._sim_buffer.pop_min()[0] == "t2"


def test_reinject_excludes_future():
    h = _Harness({"t1": _DOWN}, now=190)
    h.pending_withheld = {"t1": 200.0}
    h._sim_withheld_payload = {"t1": (150.0, _payload("t1"))}
    h._sim_reinject_ready_withheld()
    # not yet due (200 > 190): still held, not in buffer.
    assert h.pending_withheld == {"t1": 200.0}
    assert not h._sim_buffer.has("t1")


def test_reinject_keeps_slot_only_entry_until_payload_arrives():
    # Eviction registered an ESTIMATED delivery_ts before the trainer's update
    # had physically completed. A reinject tick that lands after the estimate
    # but before the payload shows up must NOT drop the ledger entry (Next
    # actions §2 under-emission bug) — it stays registered, still excluded via
    # withheld_held_ends() at the original dts is moot once now > dts, but the
    # ledger entry itself must survive for the payload to be recognized later.
    h = _Harness({"t1": _DOWN}, now=250)
    h.pending_withheld = {"t1": 200.0}  # no payload yet
    h._sim_reinject_ready_withheld()
    assert h.pending_withheld == {"t1": 200.0}       # NOT dropped
    assert not h._sim_buffer.has("t1")                # nothing to deliver yet
    assert "t1" not in h._sim_withheld_delivering

    # the physical update now arrives (actual sct=250, past the original
    # estimate) via the normal pop path -> recognized as still-withheld,
    # delivery_ts bumped to the real completion time (no past-dating).
    sel = _OortSelector()
    ch = _Channel(sel, ["t1"])
    p = _payload("t1")
    h._sim_buffer.add("t1", 250.0, p)
    assert h._sim_pop_committable(ch) is None         # held, not committed
    assert h.pending_withheld == {"t1": 250.0}        # bumped from the estimate
    assert h._sim_withheld_payload["t1"] == (250.0, p)

    h._sim_reinject_ready_withheld()
    assert h.pending_withheld == {}
    assert h._sim_buffer.has("t1")
    assert h._sim_withheld_delivering["t1"] == (250.0, 250.0)


# ---------------------------------------------------------------------------
# Invariant 1 — abandoned-then-arrived commits exactly once
# ---------------------------------------------------------------------------

def test_invariant1_abandoned_then_arrived_commits_once():
    h = _Harness({"t1": _DOWN}, now=150, inflight_tracker=True)
    sel = _AsyncSelector()
    ch = _Channel(sel, ["t1"])
    # abandon already registered the ledger (slot-only, no payload yet).
    h.pending_withheld = {"t1": 200.0}
    h._sim_inflight_expected["t1"] = 999.0

    # the physical update now arrives in the buffer.
    h._sim_buffer.add("t1", 155.0, _payload("t1"))
    # pop: end is in pending_withheld -> stash payload, do NOT re-register/commit.
    assert h._sim_pop_committable(ch) is None
    assert h.pending_withheld == {"t1": 200.0}          # unchanged (no double-reg)
    assert h._sim_withheld_payload["t1"][0] == 155.0    # arrived payload stashed
    assert "t1" not in h._sim_inflight_expected

    # at delivery_ts, reinject delivers it; pop commits exactly once.
    h._now = 200
    h._sim_reinject_ready_withheld()
    popped = h._sim_pop_committable(ch)
    assert popped is not None and popped[0] == "t1"
    assert h.pending_withheld == {}                      # fully drained
    assert h._sim_pop_committable(ch) is None            # no second commit


# ---------------------------------------------------------------------------
# C.3 — vclock abandon (both selector shapes)
# ---------------------------------------------------------------------------

def _setup_abandon(selector_cls, inflight_tracker):
    h = _Harness({"t1": _DOWN}, now=300, inflight_tracker=inflight_tracker)
    sel = selector_cls(); sel.add("t1")
    ch = _Channel(sel, ["t1"])
    if inflight_tracker:
        h._sim_inflight_expected["t1"] = 12.0
    return h, sel, ch


def test_abandon_fires_past_90s_async_shape():
    h, sel, ch = _setup_abandon(_AsyncSelector, inflight_tracker=True)
    ch.set_end_property("t1", PROP_SIM_SEND_TS, 300 - 100)  # age 100 > 90
    h._sim_abandon_stalled(ch)
    assert not sel.holds("t1")                  # slot freed -> replacement selectable
    assert "t1" in h.pending_withheld           # delivery ledger registered
    assert "t1" not in h._sim_inflight_expected


def test_abandon_fires_past_90s_oort_shape():
    h, sel, ch = _setup_abandon(_OortSelector, inflight_tracker=False)
    ch.set_end_property("t1", PROP_SIM_SEND_TS, 300 - 95)   # age 95 > 90
    h._sim_abandon_stalled(ch)
    assert not sel.holds("t1")
    assert "t1" in h.pending_withheld


def test_abandon_does_not_fire_under_90s():
    h, sel, ch = _setup_abandon(_OortSelector, inflight_tracker=False)
    ch.set_end_property("t1", PROP_SIM_SEND_TS, 300 - 50)   # age 50 <= 90
    h._sim_abandon_stalled(ch)
    assert sel.holds("t1")                      # still in-flight
    assert h.pending_withheld == {}


def test_abandon_skips_buffered_committed_and_withheld():
    h, sel, ch = _setup_abandon(_OortSelector, inflight_tracker=False)
    ch.set_end_property("t1", PROP_SIM_SEND_TS, 300 - 100)
    # already arrived in the buffer -> not stalled.
    h._sim_buffer.add("t1", 250.0, _payload("t1"))
    h._sim_abandon_stalled(ch)
    assert sel.holds("t1") and h.pending_withheld == {}


def test_abandon_gate_off_is_noop():
    h = _Harness(trainer_event_dict=None, now=300)
    sel = _OortSelector(); sel.add("t1")
    ch = _Channel(sel, ["t1"])
    ch.set_end_property("t1", PROP_SIM_SEND_TS, 0.0)
    h._sim_abandon_stalled(ch)
    assert sel.holds("t1") and h.pending_withheld == {}


# ---------------------------------------------------------------------------
# _sim_evict_unavail_inflight — Stage D.1 proactive boundary eviction
# ---------------------------------------------------------------------------
# Uses _DOWN: AVL_TRAIN[0,100) → UN_AVL[100,200) → AVL_TRAIN[200,∞)


def _harness_aware(now, ends=("t1",), selector_cls=_OortSelector):
    """Harness with proactive_inflight_evict=True, vclock at `now`."""
    h = _Harness({"t1": _DOWN, "t2": _DOWN}, now=now)
    h.proactive_inflight_evict = True
    sel = selector_cls()
    for e in ends:
        sel.add(e)
    ch = _Channel(sel, list(ends))
    return h, sel, ch


def test_evict_frees_slot_for_unavail_trainer_oort():
    # vclock=150 → t1 is UN_AVL [100,200); should be evicted immediately.
    h, sel, ch = _harness_aware(150, ("t1",), _OortSelector)
    h._sim_evict_unavail_inflight(ch)
    assert not sel.holds("t1")
    assert "t1" in h.pending_withheld
    assert h.pending_withheld["t1"] == 200.0  # next AVL_TRAIN from trace


def test_evict_frees_slot_for_unavail_trainer_async():
    h, sel, ch = _harness_aware(150, ("t1",), _AsyncSelector)
    h._sim_evict_unavail_inflight(ch)
    assert not sel.holds("t1")
    assert h.pending_withheld["t1"] == 200.0


def test_evict_leaves_available_trainer():
    # vclock=50 → t1 is AVL_TRAIN; no eviction.
    h, sel, ch = _harness_aware(50, ("t1",), _OortSelector)
    h._sim_evict_unavail_inflight(ch)
    assert sel.holds("t1")
    assert h.pending_withheld == {}


def test_evict_skips_buffered_trainer():
    # t1 is UN_AVL at vclock=150 but its update already arrived in the buffer
    # → not stalled → eviction skipped.
    h, sel, ch = _harness_aware(150, ("t1",), _OortSelector)
    h._sim_buffer.add("t1", 120.0, _payload("t1"))
    h._sim_evict_unavail_inflight(ch)
    assert sel.holds("t1")
    assert h.pending_withheld == {}


def test_evict_skips_already_withheld():
    # Invariant 1: t1 already in pending_withheld → no double-count.
    h, sel, ch = _harness_aware(150, ("t1",), _OortSelector)
    h.pending_withheld["t1"] = 200.0
    h._sim_evict_unavail_inflight(ch)
    assert sel.holds("t1")             # nothing changed
    assert h.pending_withheld == {"t1": 200.0}


def test_evict_noop_when_awareness_false():
    # proactive_inflight_evict=False → falls through to 90s abandon; evict is inert.
    h, sel, ch = _harness_aware(150, ("t1",), _OortSelector)
    h.proactive_inflight_evict = False
    h._sim_evict_unavail_inflight(ch)
    assert sel.holds("t1")
    assert h.pending_withheld == {}


def test_evict_noop_when_gate_off():
    h = _Harness(trainer_event_dict=None, now=150)
    h.proactive_inflight_evict = True
    sel = _OortSelector(); sel.add("t1")
    ch = _Channel(sel, ["t1"])
    h._sim_evict_unavail_inflight(ch)
    assert sel.holds("t1")
    assert h.pending_withheld == {}


def test_evict_multiple_trainers_partial():
    # t1 in UN_AVL[100,200), t2 has a different trace where it's AVL at vclock=150.
    # Only t1 should be evicted.
    _avl_trace = _trace((0, "AVL_TRAIN"))
    h = _Harness({"t1": _DOWN, "t2": _avl_trace}, now=150)
    h.proactive_inflight_evict = True
    sel = _OortSelector(); sel.add("t1"); sel.add("t2")
    ch = _Channel(sel, ["t1", "t2"])
    h._sim_evict_unavail_inflight(ch)
    assert not sel.holds("t1")
    assert sel.holds("t2")
    assert "t1" in h.pending_withheld
    assert "t2" not in h.pending_withheld


# ---------------------------------------------------------------------------
# _avail_stamp_end_states — avail_composition/avl_state blind-spot fix
# ---------------------------------------------------------------------------

def test_stamp_end_states_writes_every_known_end():
    # t1 in-flight UN_AVL[100,200) (e.g. just D.1-evicted); t2 plain AVL_TRAIN.
    # Both are stamped, including the in-flight/evicted one — the whole point
    # is this no longer depends on being in a "fresh candidate" subset.
    h = _Harness({"t1": _DOWN, "t2": _trace((0, "AVL_TRAIN"))}, now=150)
    sel = _OortSelector(); sel.add("t1")
    ch = _Channel(sel, ["t1", "t2"])
    h._avail_stamp_end_states(ch)
    assert ch.get_end_property("t1", PROP_AVL_STATE) == TrainerAvailState.UN_AVL
    assert ch.get_end_property("t2", PROP_AVL_STATE) == TrainerAvailState.AVL_TRAIN


def test_stamp_end_states_defaults_avl_train_when_no_trace():
    h = _Harness({"t1": _DOWN}, now=150)
    sel = _OortSelector()
    ch = _Channel(sel, ["t1", "t2"])  # t2 has no per-trainer trace entry
    h._avail_stamp_end_states(ch)
    assert ch.get_end_property("t2", PROP_AVL_STATE) == TrainerAvailState.AVL_TRAIN


def test_stamp_end_states_noop_when_gate_off():
    h = _Harness(trainer_event_dict=None, now=150)
    sel = _OortSelector()
    ch = _Channel(sel, ["t1"])
    h._avail_stamp_end_states(ch)
    assert ch.get_end_property("t1", PROP_AVL_STATE) is None


# ---------------------------------------------------------------------------
# get_curr_task_ineligible_trainers — D.2 task-type (AVL_TRAIN/AVL_EVAL) gate
# ---------------------------------------------------------------------------

_TRAIN_THEN_EVAL = _trace((0, "AVL_TRAIN"), (100, "AVL_EVAL"))


def test_task_ineligible_excludes_un_avl_for_both_tasks():
    h = _Harness({"t1": _DOWN}, now=150)  # UN_AVL[100,200)
    assert h.get_curr_task_ineligible_trainers("train") == ["t1"]
    assert h.get_curr_task_ineligible_trainers("eval") == ["t1"]


def test_task_ineligible_excludes_avl_eval_from_train_dispatch():
    h = _Harness({"t1": _TRAIN_THEN_EVAL}, now=150)  # AVL_EVAL at t=150
    assert h.get_curr_task_ineligible_trainers("train") == ["t1"]
    assert h.get_curr_task_ineligible_trainers("eval") == []


def test_task_ineligible_excludes_avl_train_from_eval_dispatch():
    h = _Harness({"t1": _TRAIN_THEN_EVAL}, now=50)  # AVL_TRAIN at t=50
    assert h.get_curr_task_ineligible_trainers("eval") == ["t1"]
    assert h.get_curr_task_ineligible_trainers("train") == []


def test_task_ineligible_unknown_task_matches_unavail_only():
    h = _Harness({"t1": _TRAIN_THEN_EVAL}, now=150)  # AVL_EVAL
    assert h.get_curr_task_ineligible_trainers("htbt") == []


def test_task_ineligible_gate_off_is_noop():
    h = _Harness(trainer_event_dict=None, now=150)
    assert h.get_curr_task_ineligible_trainers("train") == []
    assert h.get_curr_task_ineligible_trainers("eval") == []


def test_task_ineligible_2state_trace_does_not_exclude_avl_train_from_eval():
    # Regression (found via a real felix syn_0 hang, Jun 28): syn_0/syn_20 are
    # 2-state (AVL_TRAIN/UN_AVL only, no AVL_EVAL ever). Excluding AVL_TRAIN
    # from eval on such a trace makes eval's eligible pool PERMANENTLY EMPTY,
    # which corrupts the selector's shared in-flight tracking for train too
    # (async_oort.py/fedbuff.py _handle_send_state's "invalid prior selection"
    # cleanup wipes selected_ends when ends={}) — the aggregator stops reading
    # completed train responses entirely. _trace_has_avl_eval must gate this off
    # for a trace that never produces AVL_EVAL for ANY trainer.
    h = _Harness({"t1": _DOWN, "t2": _trace((0, "AVL_TRAIN"))}, now=50)  # all AVL_TRAIN
    assert h._trace_has_avl_eval is False
    assert h.get_curr_task_ineligible_trainers("eval") == []
    assert h.get_curr_task_ineligible_trainers("train") == []
