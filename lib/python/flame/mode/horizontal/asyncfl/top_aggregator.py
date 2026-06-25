# Copyright 2023 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License"); you
# may not use this file except in compliance with the License. You may
# obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""Asynchronous horizontal FL top level aggregator."""

import logging
import time
from collections import deque
from datetime import datetime, timedelta

import numpy as np
from flame.channel import VAL_CH_STATE_HTBT_RECV, VAL_CH_STATE_RECV, VAL_CH_STATE_SEND
from flame.end import KEY_END_STATE, VAL_END_STATE_NONE
from flame.common.constants import DeviceType
from flame.common.util import (
    materialize_weights,
    weights_to_device,
    weights_to_model_device,
)
from flame.mode.composer import CloneComposer
from flame.mode.horizontal.syncfl.top_aggregator import (
    TAG_AGGREGATE,
    TAG_DISTRIBUTE,
    TAG_HEARTBEAT,
)
from flame.mode.horizontal.syncfl.top_aggregator import TopAggregator as SyncTopAgg
from flame.mode.message import MessageType
from flame.mode.horizontal.client_duration import real_client_task_train_duration
from flame.mode.tasklet import Loop, Tasklet
from flame.optimizer.train_result import TrainResult
from flame import telemetry
from flame.telemetry.events import build_agg_round, build_dispatch, build_utility_belief
from flame.sim import SimReorderBuffer
from flame.selector.properties import PROP_SIM_SEND_TS, PROP_SIM_COMPLETION_TS
from flame.selector.oort import (
    PROP_DATASET_SIZE,
    PROP_LAST_SELECTED_ROUND,
    PROP_LAST_EVAL_ROUND,
    PROP_CLIENT_TASK_TRAIN_DURATION,
    PROP_ROUND_START_TIME,
    PROP_STAT_UTILITY,
    PROP_UPDATE_COUNT,
)

logger = logging.getLogger(__name__)

SEND_TIMEOUT_WAIT_S = 90  # 90 seconds timeout
# Real MQTT delivery overhead (agg→trainer + trainer→agg) expected in both real
# and sim (localhost). Added to budget_s before firing [TIMING_OVERRUN_AGG].
_NETWORK_SLACK_S = 2.0

# Max wall-clock to block on one async receive before skipping the cycle and
# re-selecting; guards against hanging when all in-flight trainers go quiet.
RECV_TIMEOUT_WAIT_S = 30

# Virtual-completion gate: each pass eagerly drains ready updates (grace-bounded
# recv_fifo also waits, capturing messages that reassemble during the pass) and
# then holds the commit while an in-flight trainer still stuck in the rxq is
# EXPECTED to complete earlier than the buffered minimum. Bounded by both the
# RECV_TIMEOUT_WAIT_S deadline and this pass cap so it never spins.
_SIM_GATE_MAX_PASSES = 64
# Gate slack: don't hold the commit for an in-flight trainer expected to complete
# only marginally earlier (absorbs budget-estimate noise).
_SIM_ORDER_SLACK_S = 2.0


class TopAggregator(SyncTopAgg):
    """Asynchronous top level Aggregator implements an ML aggregation
    role."""

    def internal_init(self) -> None:
        """Initialize internal state for role."""
        logger.info("Calling internal init for SYNC from ASYNC")
        super().internal_init()

        self._agg_goal_cnt = 0
        self._agg_goal_weights = None
        self._agg_goal = self.config.hyperparameters.aggregation_goal or 1

        self._updates_in_queue = 0
        self._updates_recevied = {}
        self._trainer_participation_in_round_count = {}
        self._trainer_participation_in_round = {}
        self._per_round_update_list = []
        self._aggregator_staleness_track_rounds = []
        self._aggregator_round_avg_staleness = []
        self._per_trainer_staleness_track = {}
        self._track_trainer_version_duration_s = {}

        self._sim_buffer = SimReorderBuffer()
        self._sim_committed: set = set()
        self._sim_pending_commit: set = set()
        self._sim_enqueue_round = {}  # end -> round it entered the reorder buffer
        # Virtual-completion gate: the aggregator's record of each in-flight trainer's
        # EXPECTED completion = dispatch vclock + its MODELED budget. Lets _sim_recv_min hold
        # the clock at the earliest expected completion so it can't race past an update that
        # virtually completed but whose message isn't drained yet. The budget is learned from
        # the contention-free TRAINING_BUDGET_S (not SIM_CLIENT_TASK_TRAIN_DURATION_S, which
        # GPU contention inflates): it's a true LOWER BOUND on sct, so clamping never overshoots.
        self._sim_inflight_expected: dict = {}   # end -> expected sim_completion_ts
        self._sim_trainer_budget: dict = {}      # end -> last observed TRAINING_BUDGET_S
        # Unseen-trainer floor for the gate's expected completion: a running MINIMUM,
        # so expected stays a true lower bound on sct and the clock never laps a
        # not-yet-seen (often fast) trainer (the past-dating seed). Mean would overshoot.
        self._sim_budget_min: float = 12.0
        self._sim_budget_running_mean: float = 12.0
        self._sim_budget_n: int = 0

        # Past-dating source attribution: which seed produced each past-dated commit
        # (sct < vclock by > slack), so a pacing fix can target the dominant one. Sources:
        #   fresh      — dispatched <=1 round ago, lapped before its update landed
        #   redispatch — re-dispatched with a new sct already below the advanced clock
        #   straggler  — first commit, dispatched >1 round ago (genuinely slow in-flight)
        #   round1     — startup transient (current round <= 1)
        self._sim_pastdated_by_source: dict = {}  # source -> [count, gap_cum]
        self._sim_commit_count: dict = {}          # end -> times committed (re-dispatch tell)

        # post-commit re-dispatch gap: end -> vclock before which it stays out of selection
        # (last commit sct + sim_redispatch_gap_s). Models finish->re-dispatch latency, which
        # spaces completions but does NOT count toward staleness (set by the pre-commit hold).
        self._sim_cooldown_until: dict = {}
        _gap = getattr(self.config.hyperparameters, "sim_redispatch_gap_s", 0.0)
        self._sim_redispatch_gap_s: float = float(_gap) if _gap is not None else 0.0

        # Clock-jump clamp. The arrival gate above is inert in sim (real-GPU compute ~0.4s
        # wall, so every in-flight trainer is already buffered → gate_holds=0); a forced commit
        # of a far-future straggler then jumps vclock past the fresh fast cohort, past-dating
        # them. The clamp caps each commit's advance at the earliest MODELED completion of any
        # still-in-flight FUTURE (exp > vclock) trainer, so the clock creeps with that cohort.
        # Excludes exp <= vclock (due/abandoned) so a lost entry can't pin the clock.
        _clamp = getattr(self.config.hyperparameters, "sim_clock_jump_clamp", True)
        self._sim_clock_jump_clamp: bool = bool(_clamp) if _clamp is not None else True

        # Event-driven re-dispatch (async only; FALSIFIED, kept off — PARITY.md §3.evt).
        # Re-stamps each freed slot's refill at the vclock it FREED (not the shared round-start
        # frontier) to regain the per-trainer completion stagger the round boundary collapses.
        # Backdating bounded by one round's advance (~4s) << min compute (~12s), so no commit
        # past-dates at dispatch; MODEL_VERSION stays self._round. Default off ⇒ byte-identical.
        _stag = getattr(self.config.hyperparameters, "sim_staggered_redispatch", False)
        self._sim_staggered_redispatch: bool = bool(_stag) if _stag is not None else False

        # sct-ordered ingestion (async only; §3.drain). When on, _sim_recv_min fills the reorder
        # buffer by draining each in-flight end's rx queue directly (channel.drain_ready) instead
        # of via the recv_fifo streamer, whose background task + shared queue could strand a
        # delivered update out of the buffer's view and let the clock lap it (past-dating). A
        # COMPLETE buffer lets the min-sct gate commit in true completion order. Default off ⇒
        # recv_fifo path. Supersedes staggered re-dispatch, so the two aren't enabled together.
        _drain = getattr(self.config.hyperparameters, "sim_sct_ordered_drain", False)
        self._sim_sct_ordered_drain: bool = bool(_drain) if _drain is not None else False
        # One-in-flight-per-trainer invariant (§3.resid, felix async). A trainer with an update
        # still outstanding must NOT be re-selected — real keeps it out of VAL_CH_STATE_SEND
        # until its update returns and is aggregated. Default off ⇒ unchanged selection.
        _resid = getattr(self.config.hyperparameters, "sim_inflight_residence", False)
        self._sim_inflight_residence: bool = bool(_resid) if _resid is not None else False
        # FIFO of vclocks at which a train-commit freed a slot; popped oldest-first to stamp
        # the trainer that refills that slot. Bounded (≈ concurrency in steady state; trimmed
        # so a transient imbalance can't make a stamp arbitrarily stale).
        self._sim_free_slot_ts: deque = deque(maxlen=128)
        self._sim_last_commit_sct: dict = {}  # end -> its last commit sct (held_s telem)

        # Real-mode settle sleep before selection (0 = compute-bound).
        _settle = getattr(self.config.hyperparameters, "real_distribute_settle_s", 0.1)
        self._real_distribute_settle_s: float = float(_settle) if _settle is not None else 0.1

        self._prev_distribute_weights_success = False

        self._per_trainer_last_heartbeat_ts = {}
        if "heartbeat_freq_s" in self.config.hyperparameters.track_trainer_avail:
            self._trainer_heartbeat_freq_s = (
                self.config.hyperparameters.track_trainer_avail["heartbeat_freq_s"]
            )
        else:
            self._trainer_heartbeat_freq_s = 99999

        if "max_allowed_miss_heartbeats" in self.config.hyperparameters.track_trainer_avail:
            self._trainer_max_miss_heartbeats = (
                self.config.hyperparameters.track_trainer_avail["max_allowed_miss_heartbeats"]
            )
        else:
            self._trainer_max_miss_heartbeats = 99999

        self.all_trainers = set()

    def _reset_agg_goal_variables(self):
        logger.debug("##### reset agg goal variables")
        # reset agg goal count
        self._agg_goal_cnt = 0

        # reset agg goal weights
        self._agg_goal_weights = None
        logger.debug(
            f"##### reset _agg_goal_cnt:{self._agg_goal_cnt}, _agg_goal_weights: "
            f"{self._agg_goal_weights}"
        )

        if self.simulated:
            self._sim_committed.clear()

    # TODO: (DG) Need to update or delete, not used right now
    def _read_heartbeat(self, tag: str) -> None:
        """Receive trainer heartbeat messaages asynchronously.

        This method is overriden from one in synchronous top
        aggregator (..top_aggregator).
        """
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug("No channel found")
            return

        logger.debug(f"Channel {channel} found for tag {tag}")
        # receive heartbeat message from trainers
        msg, metadata = next(channel.recv_fifo(channel.ends(VAL_CH_STATE_HTBT_RECV), 1))
        end, _ = metadata
        if not msg:
            logger.debug(f"No data from {end}; skipping it")
            return

        logger.debug(f"received heartbeat from {end}, will process further")
        self._process_trainer_heartbeat(msg=msg, end=end)

    def _process_trainer_heartbeat(self, msg, end) -> None:
        if MessageType.HEARTBEAT in msg:
            heartbeat_timestamp = msg[MessageType.HEARTBEAT]
            logger.debug(
                f"received heartbeat from {end} "
                f"with timestamp {heartbeat_timestamp} "
                f"at current time: {time.time()}"
            )

            # Add trainer to global_trainer set Used only to check
            # unavailable trainers later
            if end not in self.all_trainers:
                self.all_trainers.add(end)
                logger.debug(f"Added end {end} to all_trainers set")

            # Add trainer to heartbeat dict if it isnt there Add only
            # most recent heartbeat timestamp as value Discard stale
            # heartbeats if received.
            if end not in self._per_trainer_last_heartbeat_ts.keys():
                self._per_trainer_last_heartbeat_ts[end] = heartbeat_timestamp
                logger.debug(
                    f"Added first timestamp for trainer {end} "
                    f"with timestamp {heartbeat_timestamp}"
                )
            elif heartbeat_timestamp > self._per_trainer_last_heartbeat_ts[end]:
                logger.debug(
                    f"Will update timestamp for trainer {end} "
                    f" (current={self._per_trainer_last_heartbeat_ts[end]})"
                    f" with new timestamp {heartbeat_timestamp}"
                )
                self._per_trainer_last_heartbeat_ts[end] = heartbeat_timestamp
            else:
                logger.debug(
                    f"the heartbeat for {end} with timestamp "
                    f"{heartbeat_timestamp} was stale"
                )
        else:
            logger.warning(f"Got invalid {msg} while processing heartbeat")

    @staticmethod
    def _sim_end_has_ready_msg(channel, end) -> bool:
        """True if `end`'s rx queue holds a message (non-blocking readiness check).

        Used by _sim_recv_min to drain a physically-arrived in-flight update
        regardless of its modeled completion time, so slow trainers are buffered as
        futures rather than drained-in late and committed past-dated."""
        try:
            e = channel._ends.get(end)
            return e is not None and not e.is_rxq_empty()
        except Exception:
            return False

    def _sim_recv_min(self, channel, recv_ends):
        """Barrier: drain the in-flight set, then commit the smallest
        sim_completion_ts. The virtual clock advances TO each committed completion
        (vclock = max(vclock, sct) in _advance_sim_clock); with
        sim_commit_overhead_s = 0 the clock therefore tracks completions rather
        than a per-commit overhead ramp. The overhead-on-clock was the root
        cause: it ran the clock ahead of completions (overhead_cum dominated the
        clock), inflating and drifting staleness. Periodic [SIM_CLOCK_DIAG]
        verifies the clock is now sct-driven."""
        if not hasattr(self, "_sim_inflight_expected"):  # bare-init guard (tests)
            self._sim_inflight_expected = {}
            self._sim_trainer_budget = {}
            self._sim_budget_running_mean = 12.0
            self._sim_budget_n = 0
            self._sim_budget_min = 12.0
            self._sim_pastdated_by_source = {}
            self._sim_commit_count = {}
        barrier_t0 = time.time()
        deadline = barrier_t0 + RECV_TIMEOUT_WAIT_S
        drained_all = True
        probed = 0
        # Eager drain + virtual-completion gate: each pass pulls every ready update
        # into the buffer (grace-bounded recv_fifo also WAITS, capturing messages
        # that reassemble during the pass). Then the gate holds the commit while an
        # in-flight trainer whose update is still stuck in the rxq is EXPECTED to
        # complete earlier than the buffered minimum — so the clock can't race past
        # a virtually-completed-but-undelivered update (the straggler source).
        def _ingest(msg, metadata):
            # Buffer one received update into the sct-ordered reorder buffer,
            # keyed by its actual sender + sct.
            actual_end = metadata[0]
            sct = msg.get(MessageType.SIM_COMPLETION_TS)
            if sct is None:
                sct = self._vclock.now
            # Tripwire (#3): a trainer should be in-flight (hence buffered)
            # at most once; re-adding overwrites a prior update of its.
            if self._sim_buffer.has(actual_end):
                self._sim_dupadd = getattr(self, "_sim_dupadd", 0) + 1
            self._sim_buffer.add(actual_end, float(sct), (msg, metadata))
            if not hasattr(self, "_sim_enqueue_round"):
                self._sim_enqueue_round = {}
            self._sim_enqueue_round.setdefault(actual_end, getattr(self, "_round", 0))

        for _pass in range(_SIM_GATE_MAX_PASSES):
            # Ingest arrived in-flight updates into the sct-ordered buffer. The
            # probe set is the recv_ends snapshot (taken once upstream in
            # _aggregate_weights) UNION the LIVE in-flight set — a stale snapshot
            # alone misses ends that entered RECV after it, which is what let the
            # gate spin on an earliest-expected straggler it never probed and then
            # commit past it (past-dated, staleness drift).
            grace = self._sim_recv_grace_s()
            live_inflight = [
                e for e in set(recv_ends) | set(self._sim_inflight_expected)
                if channel.has(e)
                and not self._sim_buffer.has(e) and e not in self._sim_committed
            ]
            # ends still pending ingestion this pass — drives the "nothing left to
            # commit and nothing in flight" loop-exit below (per ingestion path).
            _pending_ends = live_inflight
            if getattr(self, "_sim_sct_ordered_drain", False):
                # sct-faithful ingestion: drain each live in-flight end's rx queue
                # DIRECTLY (no recv_fifo streamer), so the buffer is a COMPLETE
                # snapshot of every arrived in-flight update — the streamer could
                # strand a delivered update out of the buffer's view and let the
                # clock lap it (commit past-dated). Already-buffered/committed ends
                # are excluded above, so a HOLD pass waits (drain_ready's poll)
                # only on the genuinely-not-yet-arrived earlier-sct straggler.
                if live_inflight:
                    probed = max(probed, len(live_inflight))
                    for msg, metadata in channel.drain_ready(live_inflight, timeout=grace):
                        _ingest(msg, metadata)
                    drained_all = all(
                        self._sim_buffer.has(e) or e in self._sim_committed
                        for e in live_inflight
                    )
            else:
                # Legacy recv_fifo ingestion (default). An in-flight end is probed
                # if it is physically READY (non-empty rxq) — drain it regardless of
                # `exp` so a slow trainer whose message already arrived buffers as a
                # FUTURE instead of being drained-in late and committed past-dated —
                # OR its modeled `exp` is at/before the buffered minimum (+slack), so
                # the gate can still wait (recv_fifo grace) for an expected-soon
                # straggler whose fragments are mid-reassembly.
                _bmin = self._sim_buffer.peek_min_ts()
                _probe_ceiling = (
                    _bmin + _SIM_ORDER_SLACK_S if _bmin is not None else float("inf")
                )
                to_probe = [e for e in recv_ends
                            if not self._sim_buffer.has(e) and e not in self._sim_committed]
                _seen = set(to_probe)
                to_probe += [
                    e for e, exp in self._sim_inflight_expected.items()
                    if e not in _seen and channel.has(e)
                    and not self._sim_buffer.has(e) and e not in self._sim_committed
                    and (self._sim_end_has_ready_msg(channel, e) or exp <= _probe_ceiling)
                ]
                _pending_ends = to_probe
                if to_probe:
                    probed = max(probed, len(to_probe))
                    for msg, metadata in channel.recv_fifo(
                        to_probe, first_k=len(to_probe), timeout=grace
                    ):
                        if msg is None:  # no more ready (grace expired or set drained)
                            break
                        _ingest(msg, metadata)
                    drained_all = all(self._sim_buffer.has(e) for e in to_probe)
            # Gate: earliest expected completion among un-drained in-flight trainers.
            buffered_min = self._sim_buffer.peek_min_ts()
            _stuck_end, min_stuck = None, None
            for e, exp in self._sim_inflight_expected.items():
                if self._sim_buffer.has(e) or e in self._sim_committed:
                    continue  # already drained into the buffer, or committed
                if min_stuck is None or exp < min_stuck:
                    min_stuck, _stuck_end = exp, e
            earlier_stuck = (buffered_min is not None and min_stuck is not None
                             and min_stuck + _SIM_ORDER_SLACK_S < buffered_min)
            if buffered_min is None and not _pending_ends:
                break  # nothing to commit and nothing in flight
            if not earlier_stuck:
                break  # the buffered minimum is the true next completion
            if time.time() >= deadline:
                # A stuck trainer never arrived within the failsafe; stop waiting
                # for it (treat as lost so it can't block future commits too) and
                # commit the buffered min.
                self._sim_gate_failsafe = getattr(self, "_sim_gate_failsafe", 0) + 1
                self._sim_inflight_expected.pop(_stuck_end, None)
                break
            # else: HOLD — an in-flight trainer is expected to complete before the
            # buffered min, so committing now would lap it (the past-dating source).
            # Loop to keep draining/waiting for that earlier-expected stuck trainer.
            self._sim_gate_holds = getattr(self, "_sim_gate_holds", 0) + 1
        barrier_wait = time.time() - barrier_t0
        self._note_sim_fill(barrier_wait, drained_all)

        # Pop the minimum regardless of recv_ends membership so buffered updates
        # are not lost when an end is cleaned up before its commit.
        popped = self._sim_buffer.pop_min()
        if popped is None:
            return None, ("", datetime.now())
        _end, sct, (m, md) = popped
        # Clamp the clock-jump to the earliest in-flight FUTURE modeled completion
        # so a far-future straggler commit can't lap the fresh fast cohort. Only
        # exp > vclock counts (an already-due/abandoned end never pins the clock);
        # never advance backwards. Committing a straggler "early" (vclock < sct) is
        # the intended in-flight residence, not a past-dating.
        _advance_to = sct
        if getattr(self, "_sim_clock_jump_clamp", True):
            _now = self._vclock.now
            _min_future_exp = None
            for e, exp in self._sim_inflight_expected.items():
                if e == _end or e in self._sim_committed:
                    continue
                if exp > _now and (_min_future_exp is None or exp < _min_future_exp):
                    _min_future_exp = exp
            if _min_future_exp is not None:
                _advance_to = max(_now, min(sct, _min_future_exp + _SIM_ORDER_SLACK_S))
        self._advance_sim_clock(_advance_to)
        # Re-dispatch tell: captured BEFORE the add, since _sim_committed already
        # holds _end on a second commit. Drives the past-dating source breakdown.
        _was_recommit = _end in self._sim_committed
        if not hasattr(self, "_sim_commit_count"):
            self._sim_commit_count = {}
        self._sim_commit_count[_end] = self._sim_commit_count.get(_end, 0) + 1
        self._sim_committed.add(_end)
        # start this end's post-commit re-dispatch cooldown. Held out of
        # selection (in _distribute_weights) until vclock >= sct + gap, so it
        # returns with a fresher model_version -- the gap spaces completions
        # without counting toward this update's (already-recorded) staleness.
        _gap = getattr(self, "_sim_redispatch_gap_s", 0.0)
        if _gap > 0.0:
            if not hasattr(self, "_sim_cooldown_until"):
                self._sim_cooldown_until = {}
            self._sim_cooldown_until[_end] = sct + _gap
        # Event-driven re-dispatch bookkeeping: record this commit's sct (held_s
        # telemetry) and, for a TRAIN commit, push the just-advanced vclock as the
        # freed-slot stamp. The trainer that refills this slot rides this vclock
        # (popped FIFO in _distribute_weights) instead of the round-start frontier,
        # so train dispatches ≈ train commits keep the FIFO balanced and fresh.
        if not hasattr(self, "_sim_last_commit_sct"):
            self._sim_last_commit_sct = {}
        self._sim_last_commit_sct[_end] = sct
        if getattr(self, "_sim_staggered_redispatch", False):
            _is_train = isinstance(m, dict) and (
                MessageType.WEIGHTS in m or MessageType.WEIGHTS_BYTES in m
            )
            if _is_train:
                if not hasattr(self, "_sim_free_slot_ts"):
                    self._sim_free_slot_ts = deque(maxlen=128)
                self._sim_free_slot_ts.append(self._vclock.now)
        # Gate bookkeeping: this trainer is no longer in flight; learn its MODELED
        # budget (running mean refines the default for trainers not yet observed).
        # learn from TRAINING_BUDGET_S (contention-free modeled delay), NOT
        # SIM_CLIENT_TASK_TRAIN_DURATION_S (= max(gpu, budget), contention-inflated). The modeled
        # budget is the stable lower bound the gate needs so it fires on genuine
        # stragglers instead of being pushed into the future by a GPU spike.
        self._sim_inflight_expected.pop(_end, None)
        _budget = m.get(MessageType.TRAINING_BUDGET_S) if isinstance(m, dict) else None
        if _budget is None and isinstance(m, dict):  # fallback for older messages
            _budget = m.get(MessageType.SIM_CLIENT_TASK_TRAIN_DURATION_S)
        if _budget is not None:
            self._sim_trainer_budget[_end] = float(_budget)
            self._sim_budget_n += 1
            self._sim_budget_running_mean += (float(_budget) - self._sim_budget_running_mean) / self._sim_budget_n
            self._sim_budget_min = min(self._sim_budget_min, float(_budget))
        _commit_gap = self._vclock.now - sct
        # a "past-dated" commit is one the clock already lapped
        # (sct < vclock by more than the gate slack) — exactly what inflates
        # version-vs-clock and drifts staleness. The predictor fix should drive
        # this count and the cumulative past-dating toward zero.
        if _commit_gap > _SIM_ORDER_SLACK_S:
            self._sim_pastdated_commits = getattr(self, "_sim_pastdated_commits", 0) + 1
            self._sim_pastdated_gap_cum = getattr(self, "_sim_pastdated_gap_cum", 0.0) + _commit_gap
            self._sim_pastdated_gap_max = max(getattr(self, "_sim_pastdated_gap_max", 0.0), _commit_gap)
            # Attribute the past-dating to its seed so the pacing fix can target the
            # dominant one (see _sim_pastdated_by_source init). round_lag = how many
            # rounds ago this update was trained (current round - its MODEL_VERSION).
            _mv = m.get(MessageType.MODEL_VERSION) if isinstance(m, dict) else None
            _round_lag = (self._round - int(_mv)) if _mv is not None else None
            if self._round <= 1:
                _src = "round1"
            elif _was_recommit:
                _src = "redispatch"
            elif _round_lag is not None and _round_lag <= 1:
                _src = "fresh"
            else:
                _src = "straggler"
            if not hasattr(self, "_sim_pastdated_by_source"):
                self._sim_pastdated_by_source = {}
            _agg = self._sim_pastdated_by_source.setdefault(_src, [0, 0.0])
            _agg[0] += 1
            _agg[1] += _commit_gap
        logger.info(  # [SIM_BARRIER]: barrier_wait_s should track wall_lag
            f"[SIM_BARRIER] round={getattr(self, '_round', -1)} end={_end[-4:]} "
            f"barrier_wait_s={barrier_wait:.3f} probed={probed} "
            f"buf_depth={len(self._sim_buffer)} sct={sct:.1f} "
            f"T_v={self._vclock.now:.1f} commit_gap_s={_commit_gap:.1f}"
        )
        # ── clock / completion-spacing diagnostics ──────────
        # With overhead=0 the clock should be sct-driven: overhead_cum ~ 0 and
        # sct_adv_cum ~ vclock. buf_past (sct<=vclock, already completed) vs
        # buf_future (sct>vclock, not yet completed but physically arrived early):
        # a buffer full of FUTURE completions, committed early, is the remaining
        # throughput/dispatch-spacing question (advance ~3.0 vs real 4.5).
        self._sim_diag_n = getattr(self, "_sim_diag_n", 0) + 1
        if self._sim_diag_n % 500 == 0:
            _now = self._vclock.now
            _scts = [ts for ts, _ in self._sim_buffer._items.values()]
            _past = sum(1 for s in _scts if s <= _now)
            _bmin = self._sim_buffer.peek_min_ts()
            _lead = (_now - _bmin) if _bmin is not None else 0.0
            _pd_src = " ".join(
                f"{k}={v[0]}/{v[1]:.0f}s"
                for k, v in sorted(getattr(self, "_sim_pastdated_by_source", {}).items())
            ) or "none"
            logger.info(
                f"[SIM_CLOCK_DIAG] commits={self._sim_diag_n} "
                f"round={getattr(self, '_round', -1)} vclock={_now:.0f} "
                f"overhead_cum={getattr(self, '_sim_overhead_cum', 0.0):.0f} "
                f"sct_adv_cum={getattr(self, '_sim_sct_adv_cum', 0.0):.0f} "
                f"vclock_lead_over_buf={_lead:.1f} buf_depth={len(_scts)} "
                f"buf_past={_past} buf_future={len(_scts) - _past} "
                f"commit_gap_s={_commit_gap:.1f} barrier_wait_s={barrier_wait:.2f} "
                f"inflight_tracked={len(self._sim_inflight_expected)} "
                f"gate_holds={getattr(self, '_sim_gate_holds', 0)} "
                f"gate_failsafe={getattr(self, '_sim_gate_failsafe', 0)} "
                f"pastdated_commits={getattr(self, '_sim_pastdated_commits', 0)} "
                f"pastdated_gap_cum={getattr(self, '_sim_pastdated_gap_cum', 0.0):.0f} "
                f"pastdated_gap_max={getattr(self, '_sim_pastdated_gap_max', 0.0):.0f} "
                f"pastdated_by_source=[{_pd_src}] "
                f"budget_mean={getattr(self, '_sim_budget_running_mean', 0.0):.1f} "
                f"dup_buffer_adds={getattr(self, '_sim_dupadd', 0)}"
            )
        # recv_fifo marks every delivered end RECVD, but we only COMMITTED the
        # popped one — the rest are buffered yet still in-flight. _handle_recv_state
        # strips RECVD ends from selected_ends (freeing their concurrency slot),
        # which would let the selector over-select to N. Reset the still-buffered
        # ends back to NONE so they keep their in-flight slot until they commit;
        # to_probe already skips them via _sim_buffer.has(), so they aren't re-recv'd.
        for _buf_end in self._sim_buffer.pending_ends():
            if channel.has(_buf_end):
                channel._ends[_buf_end].set_property(KEY_END_STATE, VAL_END_STATE_NONE)
        # Release trainer that was blocked waiting for this cross-round commit.
        # Free its concurrency slot too (selected_ends), now that it committed,
        # so the next selection can refill it — the slot was held since round end.
        if _end in self._sim_pending_commit:
            self._sim_pending_commit.discard(_end)
            sel = channel._selector
            if _end in sel.all_selected:
                del sel.all_selected[_end]
            if sel.requester in sel.selected_ends:
                sel.selected_ends[sel.requester].discard(_end)
            if channel.has(_end):
                channel._ends[_end].set_property(KEY_END_STATE, VAL_END_STATE_NONE)
            logger.info(f"[SIM_PENDING_COMMIT] released {_end[-4:]} sct={sct:.1f}")
        return m, md

    def _aggregate_weights(self, tag: str) -> None:
        """Aggregate local model weights asynchronously.

        This method is overriden from one in synchronous top
        aggregator (..top_aggregator).
        """
        logger.debug(f"[AGG_START] Agg weights inside top_aggregator asyncfl, current model_version={self._round}")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug("No channel found")
            return

        # Filter to live ends; drop ghosts that left after selection to avoid
        # blocking recv_fifo on an empty queue.
        recv_ends = channel.ends(VAL_CH_STATE_RECV)
        if recv_ends:
            recv_ends = [e for e in recv_ends if channel.has(e)]
        if not recv_ends:
            if self.simulated and len(self._sim_buffer) > 0:
                recv_ends = []  # buffer still has entries to drain — don't block
            else:
                logger.debug(f"[AGG_RECV] no live recv ends (round={self._round}); skipping")
                time.sleep(0.5)
                return
        if self.simulated:
            msg, metadata = self._sim_recv_min(channel, recv_ends)
        else:
            msg, metadata = next(
                channel.recv_fifo(recv_ends, 1, timeout=RECV_TIMEOUT_WAIT_S)
            )
        end, _ = metadata
        if not msg:
            logger.debug(f"[AGG_RECV] No data from {end}; skipping it, agg_model_version={self._round}")
            return
        _t_msg_start = datetime.now()  # start of per-message processing (vii)

        # NOTE: Only 2 types of messages are expected here: (i) model
        # updates after task_to_perform=TRAIN with weights or (ii)
        # statistical utility updates after task_to_perform=EVAL with
        # info on stat_utility. Else, throw an error.

        # Case #1: Message after task_to_perform=TRAIN. This will
        # contain stat_utility too but will processed later. A train update may
        # carry weights as raw bytes (WEIGHTS_BYTES, lazy-deserialize) instead of
        # a live tensor — both mean "this is a model update", so check for either;
        # otherwise a train update (which also has STAT_UTILITY) would misroute to
        # the eval branch below.
        if MessageType.WEIGHTS in msg or MessageType.WEIGHTS_BYTES in msg:
            logger.debug(
                f"[AGG_RECV_WEIGHTS] received model updates from {end} "
                f"with trainer_model_version={msg[MessageType.MODEL_VERSION]}, "
                f"agg_current_version={self._round}"
            )

            channel.set_end_property(
                end, PROP_LAST_SELECTED_ROUND, msg[MessageType.MODEL_VERSION]
            )

            channel.set_end_property(
                end, PROP_LAST_EVAL_ROUND, msg[MessageType.MODEL_VERSION]
            )
            round_start_time_tup = channel.get_end_property(end, PROP_ROUND_START_TIME)
            end = metadata[0]
            timestamp = metadata[1]
            logger.debug(
                f"round_start_time_tup={round_start_time_tup} end={end} ts={timestamp}"
            )

            # TODO: (DG) Also set the end property for task=eval done
            # at timestamp=current.

        # Case #2: Message after task_to_perform=EVAL
        elif MessageType.STAT_UTILITY in msg:
            logger.debug(
                f"[AGG_RECV_EVAL] received eval message from {end}, "
                f"with stat_utility={msg[MessageType.STAT_UTILITY]}, "
                f"trainer_model_version={msg[MessageType.MODEL_VERSION]}, "
                f"agg_current_version={self._round}"
            )

            channel.set_end_property(
                end, PROP_STAT_UTILITY, msg[MessageType.STAT_UTILITY]
            )

            # Set last eval round to be used later for the ranking
            channel.set_end_property(
                end, PROP_LAST_EVAL_ROUND, msg[MessageType.MODEL_VERSION]
            )

            # TODO: (DG) Also set the end property for task=eval done
            # at timestamp=current.

            # add trainer to list of ends that have replied with eval updates
            # (async_oort tracks these for its round-end cleanup; other async
            # selectors e.g. fedbuff don't define the list — skip for them).
            if hasattr(channel._selector, "trainer_eval_recv_ends"):
                channel._selector.trainer_eval_recv_ends.append(end)
                logger.debug(
                    f"After appending {end} to trainer_eval_recv_ends: "
                    f"{channel._selector.trainer_eval_recv_ends}"
                )

            # Remove end from selected_ends and set its state to none
            # so that it can be selected for training in this round.
            logger.debug(
                f"Eval done, will remove end {end} from selected_ends and all_selected "
                f"to allow re-selection in same round for train"
            )
            channel._selector.remove_from_selected_ends(channel._ends, end)
            channel._selector._cleanup_removed_ends(end)

            # Eval-commit timeliness telemetry (mirror of the train branch below):
            # an eval task must commit at its OWN modeled completion, not a stale
            # one. Emitting commit_gap_s/update_visibility_lag_s tagged task=eval
            # lets the analyzer/checker catch eval past-dating (the stale-sct bug)
            # separately from train. sim-only fields are None in real mode.
            if telemetry.is_enabled():
                _sct_eval = msg.get(MessageType.SIM_COMPLETION_TS)
                _ts_eval = metadata[1] if len(metadata) > 1 else None
                _commit_gap_eval = (
                    (self._vclock.now - float(_sct_eval))
                    if (self.simulated and _sct_eval is not None) else None
                )
                _ready_e, _committed_e, _vis_lag_e = self._update_visibility_lag(
                    _sct_eval, _ts_eval
                )
                _mv_eval = msg.get(MessageType.MODEL_VERSION)
                _stale_eval = (self._round - int(_mv_eval)) if _mv_eval is not None else None
                ev, fields = build_agg_round(
                    round_num=self._round,
                    staleness=[_stale_eval] if _stale_eval is not None else None,
                    contributing_trainers=[end],
                    extra={
                        "task_to_perform": "eval",
                        "sim_completion_ts_recv": float(_sct_eval) if _sct_eval is not None else None,
                        "vclock_now": self._vclock.now if self.simulated else None,
                        "commit_gap_s": _commit_gap_eval,
                        "update_ready_ts": _ready_e,
                        "update_committed_ts": _committed_e,
                        "update_visibility_lag_s": [_vis_lag_e] if _vis_lag_e is not None else [],
                    },
                )
                telemetry.emit(ev, **fields)

            return

        # Else, throw an error and return
        else:
            logger.error(
                f"Invalid message received from {end} in aggregate_weights: {msg}"
            )
            return

        if self.reject_stale_updates == "True":
            logger.debug("Check trainer model version, disallow stale updates")
            if MessageType.MODEL_VERSION in msg:
                version = msg[MessageType.MODEL_VERSION]

            if version != self._round:
                logger.info(
                    f"Rejecting trainer update of version {version}, "
                    f"agg self._round: {self._round}. Will return."
                )
                return

        # update _track_trainer_version_duration_s to capture training
        # time
        if end not in self._track_trainer_version_duration_s.keys():
            logger.error(
                f"{end} not found in _track_trainer_version_duration_s "
                f"during aggregation"
            )
        else:
            # Use the MQTT arrival timestamp (captured when the message first landed
            # in the per-trainer rxq) so that wall_lag_s measures actual
            # send→receive latency, not commit latency. In sim mode the reorder
            # buffer delays commit by several real seconds after MQTT delivery;
            # using datetime.now() here would inflate the lag measurement by the
            # entire buffer-wait duration and fire false SEND_RECV_LAG_HIGH alerts.
            recv_wts_ts = timestamp if isinstance(timestamp, datetime) else datetime.now()
            recv_wts_version = msg[MessageType.MODEL_VERSION]

            # check0- verify that this recvd version was sent to
            # trainer
            if (
                recv_wts_version
                in self._track_trainer_version_duration_s[end][
                    "sent_wts_version_ts"
                ].keys()
            ):
                sent_wts_ts = self._track_trainer_version_duration_s[end][
                    "sent_wts_version_ts"
                ][recv_wts_version]
                # check1- sent_wts should have happened before current
                # time. Else, handle error
                if recv_wts_ts <= sent_wts_ts:
                    logger.error(
                        f"Trainer: {end}. Recv wts {recv_wts_ts} happened "
                        f"before send wts: {sent_wts_ts} "
                        f"for version {recv_wts_version}"
                    )

                # check2- recv_wts should not have happend for this
                # version before. Else, handle error
                if (
                    recv_wts_version
                    in self._track_trainer_version_duration_s[end][
                        "recv_wts_version_ts"
                    ].keys()
                ):
                    logger.error(
                        f"Trainer: {end}. Recv wts {recv_wts_ts} has already "
                        f"occured for version: {recv_wts_version}"
                    )

                # Process the recv_wts_ts and update training time
                self._track_trainer_version_duration_s[end]["recv_wts_version_ts"][
                    recv_wts_version
                ] = recv_wts_ts

                wall_lag_s = (recv_wts_ts - sent_wts_ts).total_seconds()
                logger.info(
                    f"[SEND_RECV_LAG] end={end} version={recv_wts_version} "
                    f"wall_lag_s={wall_lag_s:.3f}"
                )
                # Full per-message lag decomposition into 6 components.
                _wst = msg.get(MessageType.WALL_SEND_TS)   # trainer send (float unix)
                _wrt = msg.get(MessageType.WALL_RECV_TS)   # trainer recv of agg weights (float unix)
                _rcs = msg.get(MessageType.CLIENT_TASK_TRAIN_COMPUTE_S) # modeled compute duration (float s)
                _agg_sent_unix = sent_wts_ts.timestamp() if hasattr(sent_wts_ts, "timestamp") else None
                _agg_recv_unix = recv_wts_ts.timestamp() if hasattr(recv_wts_ts, "timestamp") else None
                _agg_to_trainer = f"{float(_wrt) - _agg_sent_unix:.3f}" if (_wrt and _agg_sent_unix) else "-"
                _compute = f"{float(_rcs):.3f}" if _rcs is not None else "-"
                _post_wait = f"{float(_wst) - float(_wrt) - float(_rcs):.3f}" if (_wst and _wrt and _rcs is not None) else "-"
                _mqtt_lag = f"{_agg_recv_unix - float(_wst):.3f}" if (_wst and _agg_recv_unix) else "-"
                _queue_wait = f"{(_t_msg_start - recv_wts_ts).total_seconds():.3f}"
                _process = f"{(datetime.now() - _t_msg_start).total_seconds():.3f}"
                logger.info(
                    f"[LAG_DECOMP] end={end} version={recv_wts_version} "
                    f"wall_lag_s={wall_lag_s:.3f} "
                    f"agg_to_trainer_s={_agg_to_trainer} "
                    f"compute_s={_compute} "
                    f"post_wait_s={_post_wait} "
                    f"mqtt_lag_s={_mqtt_lag} "
                    f"queue_wait_s={_queue_wait} "
                    f"process_s={_process}"
                )

                _budget_s = float(msg.get(MessageType.TRAINING_BUDGET_S, 0.0))
                if _budget_s > 0:
                    if self.simulated:
                        # sim overrun: modeled compute exceeded budget (GPU contention).
                        # Use SIM_CLIENT_TASK_TRAIN_DURATION_S (= max(gpu, D), pure compute) — NOT
                        # SIM_COMPLETION_TS - SIM_SEND_TS, which now also includes the
                        # post-compute completion leg and is not an overrun signal.
                        _virt_elapsed = float(msg.get(MessageType.SIM_CLIENT_TASK_TRAIN_DURATION_S, 0.0))
                        if _virt_elapsed > 0.0:
                            if _virt_elapsed > _budget_s:
                                logger.warning(
                                    f"[TIMING_OVERRUN_AGG] {end[-4:]} ver={recv_wts_version} "
                                    f"budget={_budget_s:.1f}s overrun: "
                                    f"virtual_elapsed={_virt_elapsed:.2f}s "
                                    f"(excess={_virt_elapsed - _budget_s:.2f}s). "
                                    f"Reduce trainers-per-GPU or add GPUs."
                                )
                    else:
                        if wall_lag_s > _budget_s + _NETWORK_SLACK_S:
                            logger.warning(
                                f"[TIMING_OVERRUN_AGG] {end[-4:]} ver={recv_wts_version} "
                                f"budget={_budget_s:.1f}s+slack={_NETWORK_SLACK_S:.1f}s "
                                f"overrun: wall_lag={wall_lag_s:.2f}s "
                                f"(excess={wall_lag_s - _budget_s - _NETWORK_SLACK_S:.2f}s). "
                                f"Reduce trainers-per-GPU or add GPUs."
                            )

            # TODO: (DG) Can pass a flag for this later.
            allow_updates_more_than_timeout_old = True

            if ((recv_wts_ts - sent_wts_ts).total_seconds() > SEND_TIMEOUT_WAIT_S) and (
                not allow_updates_more_than_timeout_old
            ):
                # NOTE: (DG) Timeout means that an update returns with
                # latency of [timeout, infinty). While some updates
                # might be less stale, most could be very stale.
                # Instead of cherry-picking which updates to keep and
                # which to discard, we will discard all such delayed
                # updates.
                time_staleness_s = (
                    recv_wts_ts - sent_wts_ts
                ).total_seconds() - SEND_TIMEOUT_WAIT_S
                logger.info(
                    f"Update from end {end} arrived more "
                    f"than {SEND_TIMEOUT_WAIT_S} seconds after last send. "
                    f"Update is stale by time {time_staleness_s} over the "
                    f"timeout and will be discarded."
                )

                # TODO: (DG) NEEDS TESTING. Sanity check is that it
                # should not come here with ClientNotify enabled. But
                # when it did come with ClientNotify and Train->Eval
                # calling reset_end_state_to_none, it caused issues.

                # Currently, the end is now in recvd state and will be
                # removed from selected_ends in handle_recv_state in
                # the next iteration. To add the getter through
                # recv_fifo again, we will (i) remove the end from
                # selected_ends, and (ii) set the end state to none.
                logger.debug(
                    f"Attempting to remove end {end} from selected_ends and "
                    f"re-setting its channel state"
                )
                channel._selector.remove_from_selected_ends(channel._ends, end)
                channel._selector.reset_end_state_to_none(channel._ends, end)
                channel._selector._cleanup_removed_ends(end)
                return
            # NOTE: (DG) Previously had a version equality check here
            # for version sent and version received. It was supposed
            # to be equal for syncfl and help discard incorrect round
            # messages. For asyncfl too it should be equal. However it
            # is possible that after leave/join of a trainer between
            # two rounds, a new round version is sent to the trainer,
            # while it sends back the previous version sent to it.
            # This is also a valid update since it is just the
            # previous one (and there are checks on the trainer side
            # to avoid redundant updates).
            else:
                # NOTE: total_training_time_s is approximate. It only
                # captures training time for those send_wt and recv_wt
                # that complete. Timeouts are not included in this
                # time and can be observed separately.
                curr_cumulative_training_s = self._track_trainer_version_duration_s[
                    end
                ]["total_training_time_s"]
                # Both modes: the client's INTRINSIC task-train duration = max(gpu, D),
                # excluding server-side waits (§S.dur). Real anchors on the two CLIENT stamps
                # (WALL_SEND - WALL_RECV); an agg-anchored span (recv - dispatch) folds in
                # read-wait + delivery lag and inflates slow-trainer trainer_speed telemetry.
                if self.simulated:
                    round_duration_td = timedelta(
                        seconds=float(msg.get(MessageType.SIM_CLIENT_TASK_TRAIN_DURATION_S, 0.0))
                    )
                else:
                    round_duration_td = real_client_task_train_duration(
                        msg, sent_wts_ts, recv_wts_ts
                    )
                    if round_duration_td is None:  # no client stamps -> prior behavior
                        round_duration_td = recv_wts_ts - sent_wts_ts
                curr_round_time_s = round_duration_td.total_seconds()
                new_cumulative_training_s = (
                    curr_cumulative_training_s + curr_round_time_s
                )
                self._track_trainer_version_duration_s[end][
                    "total_training_time_s"
                ] = new_cumulative_training_s
                logger.debug(
                    f"Updated training time record for {end}, details: "
                    f"{self._track_trainer_version_duration_s[end]}"
                )

                # Following the relaxation in asyncFL to not check for
                # model version equality at the aggregator, we do the
                # same for asyncoort too. We will set the end property
                # without doing the equality check. Round duration can
                # be calculated based on send and recv time for that
                # version to that trainer.
                logger.debug(
                    f"Setting channel property {PROP_CLIENT_TASK_TRAIN_DURATION} for "
                    f"end {end} with duration {round_duration_td}"
                )
                channel.set_end_property(
                    end, PROP_CLIENT_TASK_TRAIN_DURATION, round_duration_td
                )

        channel._selector.ordered_updates_recv_ends.append(end)
        self._updates_in_queue += 1
        self._per_round_update_list.append(end)
        if end not in self._updates_recevied:
            self._updates_recevied[end] = 1
        else:
            self._updates_recevied[end] += 1

        # Process the weights and send to optimizer. Lazy-deserialize: restore
        # the tensor from WEIGHTS_BYTES (only paid for this committed update);
        # default None so an eval-only/malformed message can't UnboundLocalError.
        weights = None
        if materialize_weights(msg) is not None:
            weights = weights_to_model_device(msg[MessageType.WEIGHTS], self.model)

        if MessageType.DATASET_SIZE in msg:
            count = msg[MessageType.DATASET_SIZE]
            channel.set_end_property(
                end, PROP_DATASET_SIZE, msg[MessageType.DATASET_SIZE]
            )

        if MessageType.MODEL_VERSION in msg:
            version = msg[MessageType.MODEL_VERSION]

        stat_utility = 0  # default
        if MessageType.STAT_UTILITY in msg:
            # Believed (PROP_STAT_UTILITY before overwrite) vs actual (incoming)
            # client utility — the staleness of the selector's belief
            # (believed-vs-actual; emitted for every baseline).
            if telemetry.is_enabled():
                _believed = channel.get_end_property(end, PROP_STAT_UTILITY)
                _mv = msg.get(MessageType.MODEL_VERSION)
                ev, f = build_utility_belief(
                    round_num=self._round,
                    end_id=end,
                    believed=float(_believed) if _believed is not None else None,
                    actual=float(msg[MessageType.STAT_UTILITY]),
                    staleness=(self._round - _mv) if _mv is not None else None,
                    time_mode="sim" if self.simulated else "real",
                )
                telemetry.emit(ev, **f)
            channel.set_end_property(
                end, PROP_STAT_UTILITY, msg[MessageType.STAT_UTILITY]
            )
            stat_utility = msg[MessageType.STAT_UTILITY]

        logger.debug(
            f"Received weights from {end}. It was trained on model version {version}, with {count} samples. Returned stat utility {stat_utility}"
        )

        if (
            weights is not None and count > 0
        ):  # SC_TS: count = 0 means no data (it was trained on!), so ignore!
            tres = TrainResult(weights, count, version, stat_utility)
            _cs0 = time.time()
            self.cache[end] = tres   # in-memory (MemCache)
            self._agg_cache_store_s = time.time() - _cs0
            logger.debug(f"received {len(self.cache)} trainer updates in cache")
            update_staleness_val = self._round - tres.version
            logger.debug(
                f"Received update from {end}. agg_version: {self._round}, trainer version: {tres.version}, update_staleness_val: {update_staleness_val}"
            )

            # Populate round statistics vars
            self._round_update_values["staleness"].append(update_staleness_val)
            self._round_update_values["stat_utility"].append(stat_utility)
            _round_dur = channel.get_end_property(end_id=end, key=PROP_CLIENT_TASK_TRAIN_DURATION)
            _trainer_speed_s = _round_dur.total_seconds() if _round_dur is not None else 0.0
            self._round_update_values["trainer_speed"].append(_trainer_speed_s)

            if telemetry.is_enabled():
                _sct_recv = msg.get(MessageType.SIM_COMPLETION_TS)
                # Buffer health (sim): commit_gap_s = how far the vclock has run
                # PAST this update's completion ts (>0 => reorder buffer backed up,
                # the staleness-inflation signature); residence_rounds = rounds it
                # sat buffered. inflight = concurrent in-flight (both modes).
                _enq_round = self._sim_enqueue_round.pop(end, self._round) if self.simulated else None
                _commit_gap_s = (self._vclock.now - float(_sct_recv)) if (self.simulated and _sct_recv is not None) else None
                # update_visibility_lag_s: aggregator-clock delay between when an
                # update became READY to aggregate and when it was COMMITTED into
                # the global model — same metric, mode-appropriate clock. Real:
                # wall(commit) - wall(MQTT arrival); should be ~0 (timely by
                # construction). Sim: vclock(commit) - sct(ready) = how far the
                # virtual clock ran past this update's modeled completion (the
                # past-dating signature). Fidelity = sim dist matches real dist.
                _ready_ts, _committed_ts, _vis_lag_s = self._update_visibility_lag(
                    _sct_recv, timestamp
                )
                ev, fields = build_agg_round(
                    round_num=self._round,
                    agg_goal=self._agg_goal,
                    agg_goal_count=self._agg_goal_cnt,
                    updates_in_queue=self._updates_in_queue,
                    staleness=[update_staleness_val],
                    stat_utility=[stat_utility],
                    trainer_speed_s=[_trainer_speed_s],
                    contributing_trainers=[end],
                    agg_observed_s={end: _trainer_speed_s},
                    extra={
                        "task_to_perform": "train",
                        "sim_completion_ts_recv": float(_sct_recv) if _sct_recv is not None else None,
                        "vclock_now": self._vclock.now if self.simulated else None,
                        "commit_gap_s": _commit_gap_s,
                        "update_ready_ts": _ready_ts,
                        "update_committed_ts": _committed_ts,
                        "update_visibility_lag_s": [_vis_lag_s] if _vis_lag_s is not None else [],
                        "buf_depth": len(self._sim_buffer) if self.simulated else None,
                        "residence_rounds": (self._round - _enq_round) if _enq_round is not None else None,
                        "inflight": self._updates_in_queue,
                    },
                )
                telemetry.emit(ev, **fields)

            # capture per trainer staleness
            if end in self._per_trainer_staleness_track.keys():
                logger.debug(f"found {end} in dict")
                self._per_trainer_staleness_track[end].append(update_staleness_val)
                logger.debug(
                    f"updated _per_trainer_staleness_track "
                    f"{self._per_trainer_staleness_track}"
                )
            else:
                logger.debug(f"NEW Entry {end} in dict")
                self._per_trainer_staleness_track[end] = []
                logger.debug(
                    f"created new list entry in dict _per_trainer_staleness_track "
                    f"{self._per_trainer_staleness_track}"
                )
                self._per_trainer_staleness_track[end].append(update_staleness_val)
                logger.debug(
                    f"updated _per_trainer_staleness_track "
                    f"{self._per_trainer_staleness_track}"
                )

            # staleness_alpha = 0.3 staleness_factor = staleness_alpha
            # * (1 / (self._round - tres.version + 1))

            # DG-FIX: check trainer version, discard if stale if
            # (tres.version == (self._round - 1)) or ((tres.version ==
            # self._round)):

            # if tres.version == self._round: logger.debug("proceeding
            #     to agg weights") self._agg_goal_weights =
            #     self.optimizer.do( self._agg_goal_weights,
            #         self.cache, total=count, version=self._round,
            #         staleness_factor=staleness_factor, ) # increment
            #         agg goal count self._agg_goal_cnt += 1 else:
            #         logger.debug("stale update from worker,
            #         discarding") return

            logger.debug("proceeding to agg weights")
            _opt0 = time.time()
            self._agg_goal_weights = self.optimizer.do(
                self._agg_goal_weights,
                self.cache,
                total=count,
                version=self._round,
                staleness_factor=0.0,
            )
            # [AGG_COMMIT_TIMING] per-commit aggregate cost: cache store (was disk
            # IO, now in-memory) + optimizer; complements SIM_BARRIER/DISTRIBUTE.
            logger.info(
                f"[AGG_COMMIT_TIMING] round={self._round} "
                f"cache_store_s={getattr(self, '_agg_cache_store_s', 0.0):.4f} "
                f"optimizer_s={time.time() - _opt0:.4f}"
            )
            # increment agg goal count
            self._agg_goal_cnt += 1

        if self._agg_goal_cnt < self._agg_goal:
            logger.debug(
                f"agg_goal_cnt={self._agg_goal_cnt} < agg_goal={self._agg_goal}, waiting for more"
            )
            channel.set_end_property(end, PROP_UPDATE_COUNT, self._updates_recevied[end])
            return

        if self._agg_goal_weights is None:
            logger.debug("failed model aggregation")
            time.sleep(1)
            return

        # set global weights, by adding scaled aggregated weights with
        # aggregation goal
        if self._agg_goal_cnt == self._agg_goal:
            logger.debug(
                f"agg_goal={self._agg_goal} reached, round={self._round}"
            )
            for trainer_update in self._per_round_update_list:
                if (
                    trainer_update
                    not in self._trainer_participation_in_round_count.keys()
                ):
                    self._trainer_participation_in_round_count[trainer_update] = 1
                    self._trainer_participation_in_round[trainer_update] = [
                        0
                    ] * 20000  # assuming max 20K rounds
                    self._trainer_participation_in_round[trainer_update][
                        self._round - 1
                    ] = 1
                else:
                    self._trainer_participation_in_round_count[trainer_update] += 1
                    self._trainer_participation_in_round[trainer_update][
                        self._round - 1
                    ] = 1

        self.weights = self.optimizer.scale_add_agg_weights(
            self.weights, self._agg_goal_weights, self._agg_goal
        )

        # update model with global weights
        self._update_model()

        # decrement counter since updates consumed from queue
        self._updates_in_queue -= self._agg_goal

        logger.debug(
            f"====== aggregation finished for round {self._round}, "
            f"self._agg_goal_cnt: {self._agg_goal_cnt}, self._updates_recevied: "
            f"{self._updates_recevied}, self._trainer_participation_in_round_count: "
            f"{self._trainer_participation_in_round_count}"
        )
        logger.debug(
            f"After round: {self._round}, remaining _updates_in_queue: "
            f"{self._updates_in_queue}"
        )

        if self._round % 100 == 0:
            logger.info(
                f"top agg staleness list after round {self._round} is "
                f"{self._aggregator_round_avg_staleness}"
            )
            logger.debug(
                f"top agg trainer participation in rounds, after round "
                f"{self._round} is {self._trainer_participation_in_round}"
            )

        self._compute_aggregator_stats()
        if self._round % 5 == 0:
            logger.debug(f"_agg_training_stats: {self._agg_training_stats}")
        self._reset_aggregator_stats()

        # per trainer analytics
        if self._round % 100 == 0:
            for k, v in self._per_trainer_staleness_track.items():
                trainer_staleness_arr = np.array(v)
                logger.debug(
                    f"Trainer {k} staleness info. Min {np.min(trainer_staleness_arr)}, "
                    f"Max {np.max(trainer_staleness_arr)}, "
                    f"Avg {np.mean(trainer_staleness_arr)}, "
                    f"P50 {np.median(trainer_staleness_arr)}, "
                    f"P90 {np.percentile(trainer_staleness_arr, 90)}, "
                    f"P99 {np.percentile(trainer_staleness_arr, 99)}"
                )

        total_training_time_all_trainers = 0
        for k, v in self._track_trainer_version_duration_s.items():
            total_training_time_all_trainers += v["total_training_time_s"]
        avg_training_time = total_training_time_all_trainers / len(
            self._track_trainer_version_duration_s
        )
        logger.debug(
            f"Avg training time {avg_training_time} across "
            f"{len(self._track_trainer_version_duration_s)} trainers"
        )

        logger.debug("Agg goal reached, so resetting trainer end states in the channel")
        channel.cleanup_recvd_ends()

        if self.simulated:
            self._sim_hold_busy_slots(channel)

    def oracular_trainer_avail_check(self, end: str) -> bool:
        logger.debug("In oracular_trainer_avail_check")

        picked_trainer_is_available = True

        if end in self.trainer_unavail_durations.keys():
            # aggregator seconds from start, on the trace's timeline: virtual
            # clock in simulated mode (wall-clock would barely advance vs the
            # sim timeline, so every unavailability window would be missed),
            # wall-clock in real mode. Mirrors the trainer-side _sim_now() path.
            agg_time_since_start_s = (
                self._vclock.now if self.simulated
                else time.time() - self.agg_start_time_ts
            )

            curr_trainer_unavail_list = self.trainer_unavail_durations[end]

            # iterate through unavailability list First, check if the
            # current time is within any failure window

            for start_time, duration in curr_trainer_unavail_list:
                if start_time <= agg_time_since_start_s < start_time + duration:
                    logger.debug(
                        f"### Trainer {end} attempted to be picked in failed " f"state."
                    )
                    picked_trainer_is_available = False
                    return picked_trainer_is_available
                else:
                    logger.debug(f"### Trainer {end} is available.")
                    picked_trainer_is_available = True

            # Remove entries that occurred in the past
            updated_trainer_unavail_list = [
                (start_time, duration)
                for start_time, duration in curr_trainer_unavail_list
                if (start_time + duration) >= agg_time_since_start_s
            ]

            # Remove end from trainer_unavail_durations if list is
            # empty TODO: Check if deletion is happening properly
            if len(updated_trainer_unavail_list) == 0:
                logger.info(
                    f"### Trainer {end} will no longer fail, removing from "
                    f"trainer_unavail_durations"
                )
                del self.trainer_unavail_durations[end]
            else:
                self.trainer_unavail_durations[end] = updated_trainer_unavail_list
        else:
            logger.debug(
                f"No info on end {end} in self.trainer_unavail_durations"
                f", returning TRUE (default)"
            )
        return picked_trainer_is_available

    def hearbeat_trainer_avail_check(self, end: str) -> bool:
        picked_trainer_is_available = True
        last_acceptable_heartbeat_ts = time.time() - (
            self._trainer_max_miss_heartbeats * self._trainer_heartbeat_freq_s
        )

        # return True if: heartbeat was received from trainer and it
        # is within last_acceptable_heartbeat_ts

        # return False if: if end isnt in heartbeat dict, means that
        # the trainer hasn't given a heartbeat in a while and was
        # removed based on last_acceptable_heartbeat_ts

        # NOTE: During agg init, it might have registered a trainer,
        # but not received heartbeat in such a scenario, we return
        # True so that agg is able to send init_weights to trainer and
        # start the training process this is when trainer not in
        # all_trainers and not in dict

        if (end not in self._per_trainer_last_heartbeat_ts.keys()) and (
            end not in self.all_trainers
        ):
            picked_trainer_is_available = True
            logger.debug(
                f"Might be trainer init(), trainer {end} hasnt sent any"
                f" heartbeats yet, but we return True"
            )
        elif end not in self._per_trainer_last_heartbeat_ts.keys():
            picked_trainer_is_available = False
            logger.debug(f"Trainer {end} was already marked unavailable")
        elif self._per_trainer_last_heartbeat_ts[end] < last_acceptable_heartbeat_ts:
            del self._per_trainer_last_heartbeat_ts[end]
            picked_trainer_is_available = False
            logger.debug(
                f"Trainer {end} missed max_allowed_heartbeats, " f"marked unavailable"
            )
        elif self._per_trainer_last_heartbeat_ts[end] >= last_acceptable_heartbeat_ts:
            picked_trainer_is_available = True
            logger.debug(f"Trainer {end} is available")
        else:
            logger.error(f"Availability check failed, trainer {end}, returning True")

        return picked_trainer_is_available

    def get_unavailable_trainers(self) -> list:
        # Works only for heartbeat based right now TODO: (DG) Extend
        # for other trainer_avail_checks too
        current_unavailable_trainers = [
            end
            for end in self.all_trainers
            if end not in self._per_trainer_last_heartbeat_ts.keys()
        ]
        return current_unavailable_trainers

    def check_trainer_availability(self, end: str) -> bool:
        picked_trainer_is_available = True
        if self.track_trainer_avail["enabled"] == "False":
            return True
        elif self.track_trainer_avail["type"] == "ORACULAR":
            picked_trainer_is_available = self.oracular_trainer_avail_check(end)
        elif self.track_trainer_avail["type"] == "HEARTBEAT":
            picked_trainer_is_available = self.hearbeat_trainer_avail_check(end)

        return picked_trainer_is_available

    def _pop_free_slot_ts(self, round_now):
        """Oldest freed-slot vclock (FIFO) to stamp a re-dispatch, else round_now.

        A stamp is the vclock at which a prior train commit freed a slot; it is
        always <= the live vclock (monotone), clamped defensively. Empty queue =
        cold start (round 1) or a transient with no held slot ⇒ the live frontier."""
        q = getattr(self, "_sim_free_slot_ts", None)
        if not q:
            return round_now
        ts = float(q.popleft())
        return min(ts, float(round_now)) if round_now is not None else ts

    def _sim_hold_busy_slots(self, channel) -> None:
        """Hold BUSY trainers (a compute task still outstanding) in their
        concurrency slot until their update commits.

        Sim analog of real, where the channel keeps a dispatched trainer out of
        VAL_CH_STATE_SEND until its update returns AND is aggregated (one in-flight
        update per trainer). Busy != UN_AVL: a busy trainer is AVL_TRAIN/AVL_EVAL but
        temporarily occupied, so it holds a slot in the selector's selected_ends
        (concurrency is budgeted as extra = c - len(selected_ends)) — it does NOT go
        on the unavailable list, which is for trainers that cannot participate at all.
        cleanup_recvd_ends frees a trainer the instant its message arrives (instant in
        sim); this re-blocks it so the slot is released only on commit (_sim_recv_min).
        Dropping a busy slot frees a phantom the selector refills with a NEW trainer,
        so in-flight grows toward N each round (the over-selection bug).

        Default holds the already-buffered set; with sim_inflight_residence on it holds
        the FULL dispatched-but-not-committed set (_sim_inflight_expected, train+eval),
        so a trainer whose update has not yet drained into the buffer also can't be
        re-selected mid-flight — closing the overlap tail."""
        sel = channel._selector
        requester = sel.requester
        pending_in_buffer = set(self._sim_buffer.pending_ends())
        held = pending_in_buffer
        if self._sim_inflight_residence:
            held = pending_in_buffer | set(self._sim_inflight_expected)

        # Release trainers no longer busy (committed, or — residence off — dispatched
        # with no buffer entry yet); they refill next round's fill pass.
        for end_id in [e for e in list(sel.all_selected.keys()) if e not in held]:
            del sel.all_selected[end_id]
            if channel.has(end_id):
                channel._ends[end_id].set_property(KEY_END_STATE, VAL_END_STATE_NONE)
            if requester in sel.selected_ends and end_id in sel.selected_ends[requester]:
                sel.selected_ends[requester].discard(end_id)

        # Block every busy trainer from re-selection, KEEPING its slot.
        for end_id in held:
            self._sim_pending_commit.add(end_id)
            if end_id not in sel.all_selected:
                sel.all_selected[end_id] = time.time()
            if requester in sel.selected_ends:
                sel.selected_ends[requester].add(end_id)
        if held:
            logger.debug(
                f"[SIM_PENDING] round={self._round} held {len(held)} busy "
                f"trainer(s) (buffered={len(pending_in_buffer)}): "
                f"{[e[-4:] for e in held]}"
            )

    def _distribute_weights(self, tag: str, task_to_perform: str = "train") -> None:
        """Distribute a global model in asynchronous FL fashion."""
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found for tag {tag}")
            return

        channel.await_join()
        # wait for the configured cohort so real/sim select from the same pool
        self._await_min_trainers(channel)
        self._update_weights()

        if not self.simulated and self._real_distribute_settle_s > 0.0:
            # Settle channel state before selection (real only); 0 removes this brake.
            time.sleep(self._real_distribute_settle_s)

        if self.trainer_event_dict is not None:
            curr_unavail_trainer_list = self.get_curr_unavail_trainers()
        else:
            curr_unavail_trainer_list = []

        # exclude ends in their post-commit cooldown (until vclock >= sct + gap);
        # prune expired entries.
        _gap = getattr(self, "_sim_redispatch_gap_s", 0.0)
        _cd = getattr(self, "_sim_cooldown_until", None)
        _cooling = []
        if self.simulated and _gap > 0.0 and _cd:
            _now = self._vclock.now
            _cooling = [e for e, t in self._sim_cooldown_until.items() if t > _now]
            for e in list(self._sim_cooldown_until):
                if self._sim_cooldown_until[e] <= _now:
                    del self._sim_cooldown_until[e]
            if _cooling:
                curr_unavail_trainer_list = list(
                    set(curr_unavail_trainer_list) | set(_cooling)
                )
                logger.debug(
                    f"[SIM_REDISPATCH_GAP] round={self._round} cooling={len(_cooling)} "
                    f"gap={self._sim_redispatch_gap_s:.2f}s vclock={_now:.1f}"
                )
        if self.simulated:
            # expose cooling count so the selector holds those slots (no refill).
            channel.properties["sim_cooling_count"] = len(_cooling)

        # One-in-flight-per-trainer is enforced by _sim_hold_busy_slots (commit-side):
        # a busy trainer holds its concurrency slot in selected_ends until its update
        # commits, NOT the unavailable list (which is for UN_AVL trainers that can't
        # participate at all). Marking busy trainers unavailable here frees their slot
        # and over-selects toward N — see _sim_hold_busy_slots.
        channel.set_curr_unavailable_trainers(
            trainer_unavail_list=curr_unavail_trainer_list
        )

        # Expose current vclock to selector so it can attach it to selection events.
        if self.simulated:
            channel.properties["vclock_now"] = self._vclock.now

        # Per-baseline online oracle: refresh candidate stat-utility to true values
        # before selection. No-op unless oracle_utility_injection is enabled.
        self._inject_oracle_utilities(channel, task_to_perform)

        ends = channel.ends(VAL_CH_STATE_SEND, task_to_perform)
        if not ends:
            logger.debug(f"No trainers found for tag {tag}")
            return

        # Filter to ends we actually dispatch to (skip already-sent-this-round).
        _send_ends = []
        for end in list(ends):
            if end in self._track_trainer_version_duration_s:
                sent_versions = self._track_trainer_version_duration_s[end]["sent_wts_version_ts"]
                recv_versions = self._track_trainer_version_duration_s[end]["recv_wts_version_ts"]
                if self._round in sent_versions and self._round not in recv_versions:
                    logger.warning(
                        f"[SELECTION_CHECK] Skipping {end}: already sent model_version={self._round} "
                        f"but no response received yet."
                    )
                    continue
                unreturned = [v for v in sent_versions if v not in recv_versions]
                if unreturned:
                    logger.warning(
                        f"[SELECTION_CHECK] {end} has {len(unreturned)} unreturned versions: {unreturned}"
                    )
            _send_ends.append(end)

        _round_now = self._vclock.now if self.simulated else None
        # Event-driven re-dispatch: stagger each TRAIN dispatch by the vclock at
        # which its slot freed (a prior commit), so the round-boundary cohort no
        # longer collapses to one frozen frontier. Off / eval / real ⇒ one shared
        # round_now stamp and a single serialized payload (byte-identical to before).
        _staggered = (
            self.simulated
            and getattr(self, "_sim_staggered_redispatch", False)
            and task_to_perform == "train"
        )
        # Per-end sim_send_ts: popped freed-slot stamp (staggered) else round_now.
        _end_send_ts = {}
        for end in _send_ends:
            _end_send_ts[end] = (
                self._pop_free_slot_ts(_round_now) if _staggered else _round_now
            )
        _cohort_min = min(_end_send_ts.values()) if (_staggered and _end_send_ts) else None

        # Same model goes to every recipient; serialize once unless staggered (each
        # carries its own SIM_SEND_TS so the payload must be rebuilt per end — sim
        # weights are small and dwarfed by the real GPU compute these runs do).
        base_msg = {
            MessageType.WEIGHTS: weights_to_device(self.weights, DeviceType.CPU),
            MessageType.ROUND: self._round,
            MessageType.MODEL_VERSION: self._round,
            MessageType.TASK_TO_PERFORM: task_to_perform,
        }
        _shared_payload = None
        if not _staggered:
            if self.simulated:
                base_msg[MessageType.SIM_SEND_TS] = _round_now
            _shared_payload = channel.dumps(base_msg)

        _send_t0 = time.time()  # [DISTRIBUTE_TIMING]
        for end in _send_ends:
            logger.debug(
                f"sending weights to {end} model_version={self._round} task={task_to_perform}"
            )
            channel.set_end_property(
                end, PROP_ROUND_START_TIME, (self._round, datetime.now())
            )
            if self.simulated:
                _sst = _end_send_ts[end]
                channel.set_end_property(end, PROP_SIM_SEND_TS, _sst)
                # Expected completion = dispatch vclock + a lower-bound budget (own
                # observed, else the running min), so the gate never laps this trainer.
                _budget = self._sim_trainer_budget.get(end, self._sim_budget_min)
                self._sim_inflight_expected[end] = _sst + _budget
                if _staggered:
                    _m = dict(base_msg)
                    _m[MessageType.SIM_SEND_TS] = _sst
                    payload = channel.dumps(_m)
                else:
                    payload = _shared_payload
                if telemetry.is_enabled():
                    _prior = self._sim_last_commit_sct.get(end)
                    ev, f = build_dispatch(
                        round_num=self._round, end_id=end, task=task_to_perform,
                        time_mode="sim", sim_send_ts=float(_sst),
                        redispatch_stagger_s=(float(_sst - _cohort_min)
                                              if _cohort_min is not None else 0.0),
                        held_s=(float(_round_now - _prior) if _prior is not None else None),
                        staggered=_staggered,
                    )
                    telemetry.emit(ev, **f)
            else:
                payload = _shared_payload
            channel.send_payload(end, payload)

            if end not in self._track_trainer_version_duration_s:
                self._track_trainer_version_duration_s[end] = {
                    "last_send_wts_ts": -1,
                    "sent_wts_version_ts": {},
                    "recv_wts_version_ts": {},
                    "total_training_time_s": -1,
                }

            self._track_trainer_version_duration_s[end]["sent_wts_version_ts"][
                self._round
            ] = datetime.now()

        if _send_ends:
            logger.info(
                f"[DISTRIBUTE_TIMING] round={self._round} n_sends={len(_send_ends)} "
                f"staggered={_staggered} "
                f"send_wall_s={time.time() - _send_t0:.3f}"
            )

    def compose(self) -> None:
        """Compose role with tasklets."""
        super().compose()

        with CloneComposer(self.composer) as _:
            task_internal_init = Tasklet("internal_init", self.internal_init)

            task_reset_agg_goal_vars = Tasklet(
                "reset_agg_goal_vars", self._reset_agg_goal_variables
            )

            # Created separate put tasklets for train and eval
            task_put_train = Tasklet("distribute", self.put, TAG_DISTRIBUTE, "train")

            task_put_eval = Tasklet("distribute", self.put, TAG_DISTRIBUTE, "eval")

            # TODO: (DG) Update later, task_get_weights gets both
            # weights from train and eval tasks. Will create a cleaner
            # separation later.
            task_get_weights = Tasklet("aggregate", self.get, TAG_AGGREGATE)

            # task_get_heartbeat = Tasklet("heartbeat", self.get,
            # TAG_HEARTBEAT)

        c = self.composer
        c.unlink()

        loop = Loop(loop_check_fn=lambda: self._work_done)
        # create a loop object for asyncfl to manage concurrency as
        # well as aggregation goal
        asyncfl_loop = Loop(loop_check_fn=lambda: self._agg_goal_cnt == self._agg_goal)

        # chain them again with new tasklets introduced in this class
        (
            task_internal_init
            >> c.tasklet("load_data")
            >> c.tasklet("initialize")
            >> loop(
                task_reset_agg_goal_vars
                # >> asyncfl_loop(task_put >> task_get_weights >>
                # >> task_get_heartbeat)
                >> asyncfl_loop(task_put_train >> task_put_eval >> task_get_weights)
                >> c.tasklet("train")
                >> c.tasklet("evaluate")
                >> c.tasklet("analysis")
                >> c.tasklet("save_metrics")
                >> c.tasklet("checkpoint")
                >> c.tasklet("inc_round")
            )
            >> c.tasklet("inform_end_of_training")
            >> c.tasklet("save_params")
            >> c.tasklet("save_model")
        )

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the top level
        aggregator role."""
        return [TAG_DISTRIBUTE, TAG_AGGREGATE, TAG_HEARTBEAT]
