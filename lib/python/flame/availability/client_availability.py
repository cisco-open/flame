# Copyright 2024 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""ClientAvailability — library-level oracular availability for all aggregators.

Mixed into flame/mode/horizontal/syncfl/top_aggregator.TopAggregator (the
common ancestor of the oort, asyncfl, syncfl, and fwdllm stacks). All four
aggregators inherit read_trainer_unavailability, get_curr_unavail_trainers,
_avail_now, _init_availability, and the dormant free_stalled_slot hook.

This consolidates the three duplicated read_trainer_unavailability copies that
previously lived in main_oort_sync_agg.py:173, main_asyncfl_agg.py:158, and
fwdllm_aggregator.py:481, and the duplicated get_curr_unavail_trainers in
main_oort_sync_agg.py:298 (which used wall-time in sim — now corrected to
use _vclock.now via _avail_now()).

Stage A: substrate only — no behavior change when sim_unavailability=False
(the default). _init_availability sets trainer_event_dict=None when the gate
is off, preserving byte-identical output on syn_0 runs.
Stage C will activate free_stalled_slot and wire the pending_withheld ledger.
"""

import logging
import math
import time
from typing import Optional

from flame import telemetry
from flame.availability.trace import load_trace, next_avail_after, state_at
from flame.availability.trace import (
    read_trainer_unavailability as _read_trainer_unavailability,
)
from flame.config import TrainerAvailState
from flame.mode.message import MessageType
from flame.selector.properties import PROP_AVL_STATE, PROP_SIM_SEND_TS
from flame.telemetry.events import (
    build_abandon_timeout,
    build_agg_belief_change,
    build_withheld_delivery,
)

logger = logging.getLogger(__name__)

_AVL_STATES = frozenset(
    {TrainerAvailState.AVL_TRAIN, TrainerAvailState.AVL_EVAL}
)

# Re-clock of the selector's wall-based abandon (SEND_TIMEOUT_WAIT_S) onto the
# vclock. Heuristic basis (PARITY.md / design §1): train takes ≤60s, so an
# in-flight trainer dispatched > 90 vclock-seconds ago is assumed offline.
_AVAIL_ABANDON_TIMEOUT_S = 90.0


class ClientAvailability:
    """Oracular availability substrate for TopAggregator subclasses.

    Depends on attributes set by syncfl TopAggregator.internal_init():
        self.simulated       (bool)
        self._vclock         (VirtualClock)
        self.agg_start_time_ts  (float, epoch seconds)
        self.config          (Config)
    These are all present before initialize() runs, so _init_availability
    may be called from internal_init() or initialize().
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_availability(self, config) -> None:
        """Populate trainer_event_dict from config; no-op when gate is off.

        Master gate: sim_unavailability (default False). Also accepts the
        legacy track_trainer_avail["enabled"]=True path so existing configs
        keep working without adding the new flag.

        Sets:
            self.trainer_event_dict     — dict[task_id → SortedDict] or None
            self.avail_select_filter    — bool (exclude UN_AVL from selection pool)
            self.proactive_inflight_evict — bool (Stage D proactive boundary eviction)
            self.pending_withheld       — dict[end → delivery_ts] (Stage C)
        """
        hp = config.hyperparameters
        self.trainer_event_dict: Optional[dict] = None
        # avail_select_filter: whether UN_AVL trainers are excluded from the
        # selection pool (get_curr_task_ineligible_trainers). True for aware
        # baselines (felix/oort_star/refl/feddance); False for unaware (oort/fedbuff).
        self.avail_select_filter: bool = bool(
            getattr(hp, "avail_select_filter", True)
        )
        # proactive_inflight_evict: whether in-flight slots are freed at the next
        # selection boundary when the trace shows UN_AVL (felix only).
        # NOTE: `_legacy_aware` fallback is dead in practice -- the pydantic
        # field always exists (default=None), so getattr never falls back to
        # it; kept as the correct safe default if that ever changes.
        _legacy_aware = bool(getattr(hp, "availability_aware", False))
        self.proactive_inflight_evict: bool = bool(
            getattr(hp, "proactive_inflight_evict", _legacy_aware)
        )
        self.pending_withheld: dict = {}
        # C.2 send-time withhold ledgers (shared by asyncfl + oort commit loops):
        #   _sim_withheld_payload   end -> (orig_sct, (msg, metadata)) held update
        #   _sim_withheld_delivering end -> (orig_sct, delivery_ts) being re-injected
        # Initialized here (before the gate check) so the commit-loop helpers can
        # reference them unconditionally; they stay empty when the gate is off.
        if not hasattr(self, "_sim_withheld_payload"):
            self._sim_withheld_payload: dict = {}
        if not hasattr(self, "_sim_withheld_delivering"):
            self._sim_withheld_delivering: dict = {}

        sim_unavail = bool(getattr(hp, "sim_unavailability", False))
        track = getattr(hp, "track_trainer_avail", None) or {}
        legacy_enabled = str(track.get("enabled", "False")).strip().lower() == "true"

        if not sim_unavail and not legacy_enabled:
            return

        if sim_unavail:
            client_notify = getattr(hp, "client_notify", None) or {}
            trace_name = (
                client_notify.get("trace")
                or getattr(hp, "availability_trace", None)
                or track.get("trace")
            )
        else:
            # Legacy path: only activate for ORACULAR type
            if str(track.get("type", "")).upper() != "ORACULAR":
                return
            trace_name = track.get("trace")

        if not trace_name:
            logger.warning("[AVAIL] availability enabled but no trace name configured")
            return

        trace_dir = getattr(hp, "availability_trace_dir", None)
        self.trainer_event_dict = self.read_trainer_unavailability(
            trace=trace_name, base_dir=trace_dir
        )
        # D.2 guard: does this trace ever produce AVL_EVAL anywhere, for any
        # trainer? syn_0/syn_20/syn_50 are 2-state (AVL_TRAIN/UN_AVL only) —
        # "the AVL_TRAIN<->AVL_EVAL split a 2-state trace collapses" (§1 Trace
        # representation). Computed once here, not per-call (would be an
        # O(trainers) scan of every SortedDict on every selection boundary).
        self._trace_has_avl_eval: bool = bool(self.trainer_event_dict) and any(
            TrainerAvailState.AVL_EVAL in trace.values()
            for trace in self.trainer_event_dict.values()
        )

    # ------------------------------------------------------------------
    # Time source — single call site for "what time is it on the trace timeline"
    # ------------------------------------------------------------------

    def _avail_now(self) -> float:
        """Current time on the availability timeline.

        sim:  self._vclock.now  (sim-seconds since experiment start)
        real: wall-elapsed since agg_start_time_ts

        Never uses a per-trainer frozen clock (_sim_send_ts) — see PARITY.md §S.dur
        and the REFL HIGH-1 frozen-clock root cause.
        """
        if getattr(self, "simulated", False):
            return float(self._vclock.now)
        return time.time() - self.agg_start_time_ts

    # ------------------------------------------------------------------
    # Trace loading (replaces three duplicated copies)
    # ------------------------------------------------------------------

    def read_trainer_unavailability(
        self,
        trace: Optional[str] = None,
        base_dir: Optional[str] = None,
    ) -> Optional[dict]:
        """Build task_id → SortedDict[ts_s → state_str] from the canonical store.

        Thin wrapper -- the implementation is a free function in
        flame.availability.trace so scripts/parity/ground_truth.py can call
        it without instantiating this mixin. Kept here for existing callers
        (self.read_trainer_unavailability(...) in _init_availability).
        """
        return _read_trainer_unavailability(trace, base_dir=base_dir)

    # ------------------------------------------------------------------
    # Oracular selection gate
    # ------------------------------------------------------------------

    def get_curr_unavail_trainers(self) -> list:
        """Trainers in UN_AVL state per trace-read at _avail_now().

        Returns [] when trainer_event_dict is None (gate off) or
        avail_select_filter is False (unaware baselines: oort/fedbuff).
        Byte-identical to today's behavior when sim_unavailability=False.

        Replaces the inlined bisect_right loop formerly duplicated in:
            syncfl/top_aggregator.py:1163
            main_oort_sync_agg.py:298  (wall-time bug now corrected)
        """
        if self.trainer_event_dict is None:
            return []
        if not getattr(self, "avail_select_filter", True):
            return []

        now = self._avail_now()
        unavail = [
            tid
            for tid, trace in self.trainer_event_dict.items()
            if state_at(trace, now) == TrainerAvailState.UN_AVL
        ]
        logger.info(
            f"[TRACE_READ] unavail={len(unavail)}/{len(self.trainer_event_dict)} "
            f"@ t={now:.1f}s"
        )
        return unavail

    def get_curr_task_ineligible_trainers(self, task: str) -> list:
        """D.2: UN_AVL + task-type-ineligible trainers per the trace-read.

        Extends get_curr_unavail_trainers with the task-type partition:
        AVL_TRAIN-only trainers are excluded from "eval" dispatch and vice
        versa; UN_AVL is excluded from both. Returns [] when the gate is off
        or avail_select_filter is False. The `_trace_has_avl_eval` guard is
        load-bearing: without it, a 2-state trace (nobody ever AVL_EVAL)
        makes "eval"'s eligible pool permanently empty, and an empty pool fed
        into the selector's shared-`selected_ends` cleanup (Challenge 13)
        wipes out train's in-flight tracking too — found via a real felix
        syn_0 hang (zero AGG_RECV_WEIGHTS despite all trainers sending).
        """
        if self.trainer_event_dict is None:
            return []
        if not getattr(self, "avail_select_filter", True):
            return []
        has_eval = getattr(self, "_trace_has_avl_eval", False)
        excluded_state = (
            TrainerAvailState.AVL_EVAL if task == "train"
            else TrainerAvailState.AVL_TRAIN if (task == "eval" and has_eval)
            else None
        )
        now = self._avail_now()
        ineligible = [
            tid
            for tid, trace in self.trainer_event_dict.items()
            if state_at(trace, now) == TrainerAvailState.UN_AVL
            or (excluded_state is not None and state_at(trace, now) == excluded_state)
        ]
        logger.info(
            f"[TRACE_READ] task={task!r} ineligible="
            f"{len(ineligible)}/{len(self.trainer_event_dict)} @ t={now:.1f}s"
        )
        return ineligible

    def _avail_stamp_end_states(self, channel) -> None:
        """C.6.1 follow-up: write each known end's CURRENT oracular state onto
        PROP_AVL_STATE before selection, so emit_selection's avail_composition /
        per_trainer["avl_state"] reflect reality instead of staying all-UNKNOWN.

        PROP_AVL_STATE (flame.selector.properties) was never written anywhere in
        the v1 oracular path — only the legacy client_notify push (channel.py
        update_state -> PROP_END_AVL_STATE, a DIFFERENT property, off in v1) set
        an end's availability property. Without this, avail_composition's
        `end.get_property(PROP_AVL_STATE)` read None for every end, every round.

        Stamps EVERY known end (channel._ends), not just the eligible/candidate
        subset — this is what makes an in-flight trainer D.1/C.3 just evicted
        show up as UN_AVL in the very next selection's telemetry, instead of
        being structurally invisible to it. No-op when the gate is off
        (trainer_event_dict is None) ⇒ byte-identical at sim_unavailability=False.
        """
        if self.trainer_event_dict is None:
            return
        now = self._avail_now()
        for end_id in list(channel._ends.keys()):
            trace = self.trainer_event_dict.get(end_id)
            state = state_at(trace, now) if trace else TrainerAvailState.AVL_TRAIN
            channel.set_end_property(end_id, PROP_AVL_STATE, state)
        # No separate agg_belief_change emission for the "selection" checkpoint:
        # PROP_AVL_STATE (stamped above) is already read by emit_selection into
        # per_trainer[end]["avl_state"] on every selection event, so A7's
        # selection score reads that directly. Only "commit" (below) is new.

    # ------------------------------------------------------------------
    # Aggregator belief tracking (Batch 3 T3.3) — mechanism-agnostic hook
    # ------------------------------------------------------------------

    def _record_avail_belief(
        self, end: str, state: str, *, observed_at: float,
        checkpoint: str, source: str = "trace_read",
    ) -> None:
        """Persist the aggregator's BELIEF about `end`'s availability state.

        Tagged by ``checkpoint`` ("selection" | "commit") and knowledge-source
        ``source`` (the mechanism populating this belief — "trace_read" today,
        "client_notify"/"predictive" later per Stage H). Mechanism-agnostic by
        design: a future populator calls this with a different `source`, no
        telemetry/checker/plot change needed. Emits `agg_belief_change`;
        no-op when telemetry is disabled.
        """
        if not telemetry.is_enabled():
            return
        ev, f = build_agg_belief_change(
            round_num=getattr(self, "_round", -1), end_id=end, state=str(state),
            observed_at=float(observed_at), checkpoint=checkpoint, source=source,
        )
        telemetry.emit(ev, **f)

    def _record_commit_belief(self, end: str, sct: Optional[float] = None) -> None:
        """T3.3 commit checkpoint: record the trace-read belief at an update's
        completion instant — one shared call site for both modes and all
        three aggregator stacks (asyncfl/oort/syncfl), rather than each
        re-deriving `state_at()` independently.

        sim: called from `_sim_withhold_if_unavail`, so this reads the exact
        state that gate's own withhold decision is based on. real: called
        from each stack's real receive loop; the aggregator never gates on
        this in real mode (enforcement is trainer-side, T3.1a/T3.1b's
        send-gate) — recording it anyway gives A7 a same-instant
        ground-truth comparison point for every baseline, including unaware
        ones (oort/fedbuff) that don't filter at selection.

        No-op when the gate is off (trainer_event_dict is None), or on a
        bare/partially-initialized aggregator that never ran
        _init_availability (mirrors _sim_withhold_if_unavail's own guard, so
        this is safe to call from any real-mode commit loop unconditionally),
        or when `end` has no trace entry (e.g. a registry mismatch).
        """
        trainer_event_dict = getattr(self, "trainer_event_dict", None)
        if trainer_event_dict is None:
            return
        trace = trainer_event_dict.get(end)
        if trace is None:
            return
        now = float(sct) if sct is not None else self._avail_now()
        state = state_at(trace, now)
        self._record_avail_belief(end, state.value, observed_at=now, checkpoint="commit")

    # ------------------------------------------------------------------
    # Delivery ledger (Stage C) — withheld update bookkeeping
    # ------------------------------------------------------------------
    #
    # Two ledgers, never one (Challenge 4 / invariant 1):
    #   * slot ledger    — in-flight count; freed here (free_stalled_slot).
    #   * delivery ledger — pending_withheld[end] = delivery_ts; the completed
    #                       update is HELD, not discarded, and committed later
    #                       (stale) by the live commit loop.
    # The same effect path serves all three triggers (90s vclock abandon, aware
    # boundary eviction, and the future avl_* message) — only the trigger differs.

    def compute_delivery_ts(self, end: str, sct: float) -> float:
        """When a withheld update from `end` becomes deliverable.

        The earliest time >= sct at which the trainer is AVL_* again: a
        completed-but-withheld update cannot deliver before it finished
        computing (`sct`) nor while the trainer is unreachable. If the trainer
        is already available at `sct` (it recovered before completing, or never
        went down) delivery is immediate (= sct); otherwise it waits for the
        next AVL_* window. Returns math.inf when the trace never recovers — the
        caller must guard (Challenge 10); the update is undeliverable in-window.
        """
        if self.trainer_event_dict is None:
            return float(sct)
        trace = self.trainer_event_dict.get(end)
        if not trace:
            return float(sct)
        ref = float(sct)
        if state_at(trace, ref) in _AVL_STATES:
            return ref
        nxt = next_avail_after(trace, ref)
        if nxt == math.inf:
            return math.inf
        return max(ref, float(nxt))

    def free_stalled_slot(
        self, channel, end: str, *, reason: str, sct: Optional[float] = None
    ) -> Optional[float]:
        """Free an in-flight slot and register its pending withheld delivery.

        Triggered by (a) the 90s vclock abandon for everyone (C.3) and (b) the
        aware boundary eviction for availability_aware baselines (Stage D); Stage H
        adds an avl_* message trigger without changing this effect logic — the
        abstraction exists for exactly that swap.

        1. Remove `end` from the selector slot ledger (selected_ends / all_selected).
        2. Compute delivery_ts = compute_delivery_ts(end, sct) (delivery ledger).
        3. Register pending_withheld[end] = delivery_ts.

        No-op (returns None) when the gate is off (trainer_event_dict is None),
        preserving byte-identity. Returns the registered delivery_ts otherwise.
        """
        if self.trainer_event_dict is None:
            logger.debug(
                f"[AVAIL] free_stalled_slot({end!r}) — gate off, no-op"
            )
            return None

        # 1. Slot ledger: release the concurrency slot so a replacement is
        #    selectable. Mirrors the abandon path's removal (random.py:215).
        #    Robust to BOTH selector shapes: asyncfl (fedbuff/async_oort/
        #    async_random) keep selected_ends as a dict keyed by requester plus an
        #    all_selected dict; oort/refl/feddance keep selected_ends as a flat set
        #    and have no all_selected. Drop `end` from whichever is present.
        self._avail_free_slot_ledger(channel, end)
        # Drop it from the asyncfl gate's in-flight tracker too (no-op on oort,
        # which tracks in-flight purely via selected_ends) — "free the slot" is one
        # effect across both stacks.
        self._avail_drop_inflight(end)

        # 2/3. Delivery ledger: hold the completed update until delivery_ts.
        if sct is None:
            sct = self._avail_now()
        delivery_ts = self.compute_delivery_ts(end, sct)
        self.pending_withheld[end] = delivery_ts
        logger.info(
            f"[AVAIL] free_stalled_slot({end!r}, reason={reason!r}) "
            f"sct={float(sct):.1f} delivery_ts={delivery_ts:.1f}"
        )
        return delivery_ts

    def withheld_held_ends(self, now: Optional[float] = None) -> set:
        """Ends whose withheld update has NOT yet reached its delivery_ts.

        These must stay out of the eligible pool (invariant 2: a still-down
        trainer is never re-selected) until vclock >= delivery_ts — the §4.5
        residence exclusion extended from `sct` to `delivery_ts`. Unioned into
        the unavailable list by the selection driver.
        """
        if not self.pending_withheld:
            return set()
        if now is None:
            now = self._avail_now()
        return {end for end, dts in self.pending_withheld.items() if dts > now}

    def ready_withheld(self, now: Optional[float] = None) -> list:
        """Withheld ends due for delivery (delivery_ts <= now), commit order.

        Ordered by (delivery_ts, end_id) so the late stale commits replay
        deterministically (Challenge 1/6 — past-dating is avoided because a
        withheld update commits at delivery_ts > sct, never at sct).
        """
        if not self.pending_withheld:
            return []
        if now is None:
            now = self._avail_now()
        due = [
            (end, dts) for end, dts in self.pending_withheld.items() if dts <= now
        ]
        due.sort(key=lambda kv: (kv[1], str(kv[0])))
        return due

    def commit_withheld(self, end: str) -> Optional[float]:
        """Pop `end` from the delivery ledger once its late update has committed."""
        return self.pending_withheld.pop(end, None)

    # ------------------------------------------------------------------
    # Slot-ledger helpers (robust to both selector shapes)
    # ------------------------------------------------------------------

    def _avail_free_slot_ledger(self, channel, end: str) -> None:
        """Drop `end` from the selector slot ledger + reset its end state.

        asyncfl selectors keep selected_ends as dict[requester -> set] + an
        all_selected dict; oort/refl/feddance keep selected_ends as a flat set
        and have no all_selected. Handle whichever is present.
        """
        sel = getattr(channel, "_selector", None)
        if sel is not None:
            all_selected = getattr(sel, "all_selected", None)
            if isinstance(all_selected, dict):
                all_selected.pop(end, None)
            selected_ends = getattr(sel, "selected_ends", None)
            if isinstance(selected_ends, dict):
                requester = getattr(sel, "requester", None)
                if requester in selected_ends:
                    selected_ends[requester].discard(end)
            elif isinstance(selected_ends, set):
                selected_ends.discard(end)
        if channel.has(end):
            from flame.end import KEY_END_STATE, VAL_END_STATE_NONE

            channel._ends[end].set_property(KEY_END_STATE, VAL_END_STATE_NONE)

    def _avail_drop_inflight(self, end: str) -> None:
        """Remove `end` from the asyncfl gate's in-flight tracker (no-op on oort)."""
        ie = getattr(self, "_sim_inflight_expected", None)
        if isinstance(ie, dict):
            ie.pop(end, None)

    def _avail_inflight_ends(self, channel) -> set:
        """In-flight (slot-ledger) ends, generic over both selector shapes."""
        sel = getattr(channel, "_selector", None)
        if sel is None:
            return set()
        se = getattr(sel, "selected_ends", None)
        if isinstance(se, dict):
            out: set = set()
            for v in se.values():
                out |= set(v)
            return out
        if isinstance(se, (set, list, tuple)):
            return set(se)
        return set()

    # ------------------------------------------------------------------
    # Shared commit-loop wiring (C.2 / C.3) — one logic for asyncfl + oort
    # ------------------------------------------------------------------
    #
    # Both stacks own a different sim commit loop (asyncfl _sim_recv_min's single
    # pop site; oort _sim_drain_buffer's generator pop loop), but the send-gate /
    # late-recommit / vclock-abandon EFFECT is identical, so it lives here and each
    # loop just calls in. Depends only on self._sim_buffer + the ledgers; never on
    # a stack-specific attribute (the gate tracker is reached via the guarded
    # _avail_drop_inflight hook). Off ⇒ pending_withheld stays empty ⇒ no-op.

    def _sim_reinject_ready_withheld(self) -> None:
        """C.2: re-inject withheld updates whose delivery_ts has arrived.

        Call before each pop. For every ledger entry due at the current vclock,
        re-add the held payload to the reorder buffer at delivery_ts so it
        commits stale through the normal path, then pop the ledger. A
        slot-only entry (registered at eviction time, before the update
        physically completed) carries no payload yet — it STAYS registered
        rather than being dropped; once the payload arrives, `_sim_withhold_
        if_unavail` recognizes it's still in `pending_withheld`, stashes it,
        and bumps `delivery_ts` to the real completion time so the next call
        here delivers it. No-op when the ledger is empty.
        """
        if not getattr(self, "pending_withheld", None):
            return
        buf = getattr(self, "_sim_buffer", None)
        if buf is None:
            return
        for end, dts in self.ready_withheld(self._avail_now()):
            payload = self._sim_withheld_payload.pop(end, None)
            if payload is None:
                continue  # no payload yet; stays registered until it arrives
            self.commit_withheld(end)
            orig_sct, msgmd = payload
            buf.add(end, float(dts), msgmd)
            self._sim_withheld_delivering[end] = (float(orig_sct), float(dts))
            logger.info(
                f"[WITHHELD_REINJECT] end={str(end)[-4:]} "
                f"sct={float(orig_sct):.1f} delivery_ts={float(dts):.1f}"
            )

    def _sim_withhold_if_unavail(self, channel, end, sct, msgmd) -> bool:
        """Per-update send-gate: True if this completed update is HELD, else False.

        The single-update primitive both commit loops share. When True the caller
        must skip the update — it has been removed from the buffer/slot accounting
        (held in the delivery ledger, delivered stale at delivery_ts). When False
        the update is committable now (trainer available at completion, or gate
        off). The caller applies any stack-specific gate (oort's still-computing
        carry-over) BEFORE this — a still-computing future-sct straggler has not
        reached its send-gate yet (Challenge 7), so carry-over wins.

        Returns False unchanged when sim_unavailability is off (compute_delivery_ts
        ⇒ sct), preserving byte-identity.
        """
        # Gate off (or a bare aggregator that never ran _init_availability): the
        # update is always committable and no ledger is touched. Keeps the shared
        # primitive safe to call from any partially-initialized commit loop.
        if getattr(self, "trainer_event_dict", None) is None:
            return False
        # T3.3 commit-checkpoint belief: read once here, covering both branches
        # below (fresh commit and re-registered/re-stashed) with one call.
        self._record_commit_belief(end, sct)
        # invariant 1: never re-register / double-count an end whose slot was
        # already freed (C.3 abandon). Its arrived payload is stashed so the
        # reinject delivers it at the registered delivery_ts.
        if end in self.pending_withheld:
            self._sim_withheld_payload[end] = (float(sct), msgmd)
            self._avail_drop_inflight(end)
            # delivery_ts was an ESTIMATE made at eviction time (before this
            # update finished computing); bump to the real completion-based
            # value if later -- never earlier (invariant 2: never release a
            # still-down trainer before its registered window).
            self.pending_withheld[end] = max(
                self.pending_withheld[end], self.compute_delivery_ts(end, float(sct))
            )
            return True
        dts = self.compute_delivery_ts(end, sct)
        if dts <= sct:
            return False  # available at completion (or gate off) — commit now
        # withhold: trainer is UN_AVL at sct; hold the completed update.
        if dts == math.inf:
            # trace never recovers in-window: the update is undeliverable.
            # free_stalled_slot still registers the ledger (end stays excluded);
            # drop the payload (acceptable v1 edge, Challenge 10).
            logger.info(
                f"[WITHHELD_LOST] end={str(end)[-4:]} sct={float(sct):.1f} "
                f"trace never recovers"
            )
        else:
            self._sim_withheld_payload[end] = (float(sct), msgmd)
        self.free_stalled_slot(
            channel, end, reason="send_gate_withhold", sct=float(sct)
        )
        return True

    def _sim_pop_committable(self, channel):
        """C.2 (asyncfl): pop the smallest buffered update that is committable now.

        Loops over _sim_withhold_if_unavail, skipping send-gated updates and
        popping the next-smallest committable one. Returns (end, sct, (msg,
        metadata)) or None (buffer drained). The vclock is NOT advanced for a
        withheld pop — only the committed update drives the clock.

        Off (or trainer available at sct) ⇒ a single pop identical to the prior
        pop_min(); reinject a no-op. Byte-identical when sim_unavailability is off.
        """
        buf = self._sim_buffer
        while True:
            popped = buf.pop_min()
            if popped is None:
                return None
            end, sct, msgmd = popped
            if self._sim_withhold_if_unavail(channel, end, sct, msgmd):
                continue
            return popped

    def _sim_take_withheld_delivering(self, end: str) -> Optional[tuple]:
        """Pop (orig_sct, delivery_ts) if `end`'s commit is a late withheld delivery.

        The commit body calls this to recognize a re-injected stale delivery (vs a
        fresh/straggler commit) so it can emit the withheld_delivery rung and tag
        the "withheld" past-dating bucket. Returns None for an ordinary commit.
        """
        d = getattr(self, "_sim_withheld_delivering", None)
        if not d:
            return None
        return d.pop(end, None)

    def _emit_withheld_delivery(self, end, msg, orig_sct, delivery_ts) -> None:
        """Emit the withheld_delivery rung for a late stale commit (best-effort).

        Stamps ``actual_commit_ts`` via ``_avail_now()`` — call this AFTER
        advancing the clock (``_advance_sim_clock``) for this update, kept
        as one uniform rule across all three stacks (see UNAVAILABILITY_
        DESIGN.md's T3.5 for why syncfl/oort's drain loops were reordered to
        match asyncfl, despite it being a provable no-op there today).
        """
        if not telemetry.is_enabled():
            return
        mv = msg.get(MessageType.MODEL_VERSION) if isinstance(msg, dict) else None
        ev, f = build_withheld_delivery(
            round_num=self._round, end_id=end,
            sct=float(orig_sct), delivery_ts=float(delivery_ts),
            staleness=(self._round - int(mv)) if mv is not None else None,
            accepted=True, time_mode="sim",
            actual_commit_ts=self._avail_now(),
        )
        telemetry.emit(ev, **f)

    def _sim_abandon_stalled(self, channel) -> None:
        """C.3: free in-flight slots stalled past the 90s vclock deadline.

        Re-clocks the selector's wall-based abandon (inert in sim) onto the vclock.
        A trainer dispatched > 90 vclock-seconds ago whose update has neither
        buffered nor committed is assumed offline: free its slot (a replacement
        becomes selectable) and register the delivery ledger. If its update later
        physically arrives it is reconciled by _sim_pop_committable (payload stash)
        / _sim_reinject_ready_withheld.

        Slot ledger ⊥ delivery ledger (Challenge 4): the slot is freed here, the
        completed update is NOT discarded — it still commits (stale) and is
        accept/reject-gated by the baseline's existing staleness rule. No-op when
        the gate is off (trainer_event_dict is None) ⇒ byte-identical.
        """
        if getattr(self, "trainer_event_dict", None) is None:
            return
        inflight = self._avail_inflight_ends(channel)
        if not inflight:
            return
        now = self._avail_now()
        buf = getattr(self, "_sim_buffer", None)
        committed = getattr(self, "_sim_committed", set())
        for end in list(inflight):
            if buf is not None and buf.has(end):
                continue  # already arrived — not stalled
            if end in committed or end in self.pending_withheld:
                continue  # invariant 1: already committed / abandoned
            sst = channel.get_end_property(end, PROP_SIM_SEND_TS)
            if sst is None:
                continue
            if now - float(sst) <= _AVAIL_ABANDON_TIMEOUT_S:
                continue
            self.free_stalled_slot(
                channel, end, reason="abandon_90s_vclock", sct=now
            )
            logger.info(
                f"[ABANDON_90S] end={str(end)[-4:]} sim_send_ts={float(sst):.1f} "
                f"vclock={now:.1f} age={now - float(sst):.1f}s"
            )
            if telemetry.is_enabled():
                ev, f = build_abandon_timeout(
                    round_num=getattr(self, "_round", -1), end_id=end,
                    sim_send_ts=float(sst), vclock_now=now, time_mode="sim",
                    reason="abandon_90s_vclock",
                )
                telemetry.emit(ev, **f)

    def _sim_evict_unavail_inflight(self, channel) -> None:
        """D.1: Proactively free in-flight slots for trainers now showing UN_AVL.

        For availability_aware baselines the oracular trace read at the selection
        boundary is authoritative — no need to wait for the 90s vclock deadline
        (C.3). A trainer that transitioned to UN_AVL since it was dispatched has
        its slot freed immediately so a replacement is selectable this round.

        Same effect as free_stalled_slot (slot ledger freed + delivery ledger
        registered) — only the trigger differs from C.3. No-op when the gate is
        off (trainer_event_dict is None) or proactive_inflight_evict is False (unaware
        and aware-at-selection-only baselines stay on the C.3 90s path).
        """
        if not getattr(self, "proactive_inflight_evict", False):
            return
        if getattr(self, "trainer_event_dict", None) is None:
            return
        inflight = self._avail_inflight_ends(channel)
        if not inflight:
            return
        now = self._avail_now()
        buf = getattr(self, "_sim_buffer", None)
        committed = getattr(self, "_sim_committed", set())
        for end in list(inflight):
            if buf is not None and buf.has(end):
                continue  # update already arrived in buffer — not stalled
            if end in committed or end in self.pending_withheld:
                continue  # invariant 1: already committed / registered
            trace = self.trainer_event_dict.get(end)
            if not trace:
                continue
            if state_at(trace, now) != TrainerAvailState.UN_AVL:
                continue  # still available — leave the slot
            self.free_stalled_slot(
                channel, end, reason="aware_boundary_eviction", sct=now
            )
            logger.info(
                f"[AWARE_EVICT] end={str(end)[-4:]} vclock={now:.1f} "
                f"state=UN_AVL — proactive boundary eviction"
            )
            if telemetry.is_enabled():
                sst = channel.get_end_property(end, PROP_SIM_SEND_TS)
                ev, f = build_abandon_timeout(
                    round_num=getattr(self, "_round", -1), end_id=end,
                    sim_send_ts=float(sst) if sst is not None else now,
                    vclock_now=now, time_mode="sim",
                    reason="aware_boundary_eviction",
                )
                telemetry.emit(ev, **f)

    def _next_avail_vclock(self) -> Optional[float]:
        """Stage F: earliest vclock at which any trainer next becomes selectable.

        Returns the minimum of:
        - next AVL_* transition for each currently-UN_AVL trainer's trace
        - earliest pending withheld delivery_ts (withheld end re-enters pool then)

        Used by starvation clock-advance: when no trainers are selectable, the
        sim advances the vclock here rather than wall-sleeping, so the outer
        retry immediately sees the newly-available cohort. Returns None when the
        gate is off or no future availability exists in any trace.
        """
        if not getattr(self, "trainer_event_dict", None):
            return None
        now = self._avail_now()
        candidates: list = []
        for trace in self.trainer_event_dict.values():
            nxt = next_avail_after(trace, now)
            if nxt != math.inf:
                candidates.append(nxt)
        pw = getattr(self, "pending_withheld", None)
        if pw:
            candidates.extend(pw.values())
        return min(candidates) if candidates else None
