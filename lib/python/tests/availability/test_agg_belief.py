# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Batch 3 T3.3 — aggregator belief-tracking hooks on ClientAvailability.

_record_avail_belief / _record_commit_belief are mechanism-agnostic telemetry
hooks (agg_belief_change), not behavior — these tests assert the emission
schema and the commit-checkpoint gate-off/no-trace no-ops, plus that
_sim_withhold_if_unavail (the sim commit checkpoint) calls through to it.
"""

import json

import pytest
from sortedcontainers import SortedDict

from flame import telemetry
from flame.availability.client_availability import ClientAvailability
from flame.telemetry.events import EVENT_AGG_BELIEF_CHANGE, build_agg_belief_change


@pytest.fixture(autouse=True)
def _reset_telemetry():
    telemetry.shutdown()
    yield
    telemetry.shutdown()


def _trace(*pairs):
    d = SortedDict()
    for ts, state in pairs:
        d[float(ts)] = state
    return d


class _Harness(ClientAvailability):
    def __init__(self, trainer_event_dict=None, now=0.0, simulated=True):
        self.trainer_event_dict = trainer_event_dict
        self.pending_withheld = {}
        self._sim_withheld_payload = {}
        self._sim_withheld_delivering = {}
        self._now = now
        self.simulated = simulated
        self._round = 3

    def _avail_now(self):
        return self._now


# ---------------------------------------------------------------------------
# build_agg_belief_change schema
# ---------------------------------------------------------------------------

def test_build_agg_belief_change_schema():
    ev, f = build_agg_belief_change(
        round_num=5, end_id="t1", state="UN_AVL", observed_at=120.0,
        checkpoint="commit", source="trace_read",
    )
    assert ev == EVENT_AGG_BELIEF_CHANGE
    for key in ("round", "end_id", "state", "observed_at", "checkpoint", "source"):
        assert key in f
    assert f["checkpoint"] == "commit"
    assert f["source"] == "trace_read"


# ---------------------------------------------------------------------------
# _record_avail_belief
# ---------------------------------------------------------------------------

def test_record_avail_belief_noop_when_telemetry_disabled():
    h = _Harness()
    h._record_avail_belief("t1", "AVL_TRAIN", observed_at=0.0, checkpoint="selection")
    # No exception, no writer configured -- nothing to assert beyond "didn't crash".


def test_record_avail_belief_emits_when_enabled(tmp_path):
    telemetry.configure(role="aggregator", run_dir=str(tmp_path))
    h = _Harness()
    h._record_avail_belief("t1", "UN_AVL", observed_at=42.0, checkpoint="commit",
                          source="trace_read")
    telemetry.shutdown()
    path = tmp_path / "aggregator.jsonl"
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    belief = [l for l in lines if l["event"] == EVENT_AGG_BELIEF_CHANGE]
    assert len(belief) == 1
    assert belief[0]["end_id"] == "t1"
    assert belief[0]["state"] == "UN_AVL"
    assert belief[0]["observed_at"] == 42.0
    assert belief[0]["checkpoint"] == "commit"
    assert belief[0]["round"] == 3


# ---------------------------------------------------------------------------
# _record_commit_belief
# ---------------------------------------------------------------------------

def test_record_commit_belief_noop_gate_off(tmp_path):
    telemetry.configure(role="aggregator", run_dir=str(tmp_path))
    h = _Harness(trainer_event_dict=None)
    h._record_commit_belief("t1", sct=100.0)
    telemetry.shutdown()
    lines = (tmp_path / "aggregator.jsonl").read_text().splitlines()
    assert not any(json.loads(l)["event"] == EVENT_AGG_BELIEF_CHANGE for l in lines if l.strip())


def test_record_commit_belief_noop_missing_trace(tmp_path):
    telemetry.configure(role="aggregator", run_dir=str(tmp_path))
    h = _Harness(trainer_event_dict={"t2": _trace((0, "AVL_TRAIN"))})
    h._record_commit_belief("t1", sct=100.0)  # t1 not in trainer_event_dict
    telemetry.shutdown()
    lines = (tmp_path / "aggregator.jsonl").read_text().splitlines()
    assert not any(json.loads(l)["event"] == EVENT_AGG_BELIEF_CHANGE for l in lines if l.strip())


def test_record_commit_belief_reads_state_at_sct(tmp_path):
    telemetry.configure(role="aggregator", run_dir=str(tmp_path))
    trace = _trace((0, "AVL_TRAIN"), (600, "UN_AVL"))
    h = _Harness(trainer_event_dict={"t1": trace})
    h._record_commit_belief("t1", sct=700.0)  # after the 600s transition
    telemetry.shutdown()
    lines = [json.loads(l) for l in (tmp_path / "aggregator.jsonl").read_text().splitlines() if l.strip()]
    belief = [l for l in lines if l["event"] == EVENT_AGG_BELIEF_CHANGE][0]
    assert belief["state"] == "UN_AVL"
    assert belief["observed_at"] == 700.0
    assert belief["checkpoint"] == "commit"


def test_record_commit_belief_defaults_to_avail_now(tmp_path):
    telemetry.configure(role="aggregator", run_dir=str(tmp_path))
    trace = _trace((0, "AVL_TRAIN"))
    h = _Harness(trainer_event_dict={"t1": trace}, now=55.0)
    h._record_commit_belief("t1")  # sct=None -> _avail_now()
    telemetry.shutdown()
    lines = [json.loads(l) for l in (tmp_path / "aggregator.jsonl").read_text().splitlines() if l.strip()]
    belief = [l for l in lines if l["event"] == EVENT_AGG_BELIEF_CHANGE][0]
    assert belief["observed_at"] == 55.0


# ---------------------------------------------------------------------------
# _sim_withhold_if_unavail calls through to _record_commit_belief
# ---------------------------------------------------------------------------

def test_sim_withhold_if_unavail_records_commit_belief(monkeypatch):
    h = _Harness(trainer_event_dict={"t1": _trace((0, "AVL_TRAIN"))})
    calls = []
    monkeypatch.setattr(h, "_record_commit_belief", lambda end, sct=None: calls.append((end, sct)))
    # Available the whole time -> gate returns False (commits now), but the
    # belief must still have been recorded before that decision.
    result = h._sim_withhold_if_unavail(channel=None, end="t1", sct=10.0, msgmd=None)
    assert result is False
    assert calls == [("t1", 10.0)]


def test_sim_withhold_if_unavail_gate_off_never_records_belief(monkeypatch):
    h = _Harness(trainer_event_dict=None)
    calls = []
    monkeypatch.setattr(h, "_record_commit_belief", lambda end, sct=None: calls.append((end, sct)))
    result = h._sim_withhold_if_unavail(channel=None, end="t1", sct=10.0, msgmd=None)
    assert result is False
    assert calls == []


# ---------------------------------------------------------------------------
# _emit_withheld_delivery (Batch 3 T3.5 — K11 commit_promptness)
# ---------------------------------------------------------------------------

def test_emit_withheld_delivery_stamps_actual_commit_ts(tmp_path):
    """actual_commit_ts must reflect the CALLER's clock at emission time
    (_avail_now(), sampled fresh -- not orig_sct/delivery_ts, which are just
    passed through). The caller is responsible for advancing its clock for
    THIS commit before calling (see syncfl/oort top_aggregator.py's Batch 3
    T3.5 reorder) -- this test only asserts the harness's current _avail_now()
    ends up on the emitted event, not the reorder itself (that's covered by
    the existing top_aggregator withheld-commit-loop tests continuing to pass
    unchanged, since the reorder touches only telemetry timing, not state)."""
    telemetry.configure(role="aggregator", run_dir=str(tmp_path))
    h = _Harness(now=305.5)
    h._emit_withheld_delivery("t1", {}, orig_sct=100.0, delivery_ts=300.0)
    telemetry.shutdown()
    lines = [json.loads(l) for l in (tmp_path / "aggregator.jsonl").read_text().splitlines() if l.strip()]
    ev = [l for l in lines if l["event"] == "withheld_delivery"][0]
    assert ev["sct"] == 100.0
    assert ev["delivery_ts"] == 300.0
    assert ev["actual_commit_ts"] == 305.5


def test_emit_withheld_delivery_noop_when_telemetry_disabled():
    h = _Harness(now=100.0)
    h._emit_withheld_delivery("t1", {}, orig_sct=0.0, delivery_ts=50.0)  # must not raise
