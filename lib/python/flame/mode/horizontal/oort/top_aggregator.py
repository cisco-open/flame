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
"""Oort horizontal FL top level aggregator."""

import logging
import time
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Tuple

from flame.sim import SimReorderBuffer

from flame.channel import VAL_CH_STATE_SEND
from flame.common.constants import DeviceType
from flame.common.util import (
    materialize_weights,
    weights_to_device,
    weights_to_model_device,
)
from flame.mode.message import MessageType
from flame.optimizer.train_result import TrainResult
from flame.selector.oort import (
    PROP_LAST_SELECTED_ROUND,
    PROP_ROUND_DURATION,
    PROP_ROUND_START_TIME,
    PROP_STAT_UTILITY,
    PROP_LAST_EVAL_ROUND,
)
from flame.selector.properties import PROP_SIM_SEND_TS

from ..top_aggregator import TopAggregator as BaseTopAggregator
from flame import telemetry
from flame.telemetry.events import (
    build_agg_round,
    build_inflight_residence,
    build_utility_belief,
)

logger = logging.getLogger(__name__)

# Real MQTT delivery overhead (agg→trainer + trainer→agg) expected in both real
# and sim (localhost). Added to budget_s before firing [TIMING_OVERRUN_AGG].
_NETWORK_SLACK_S = 2.0


class TopAggregator(BaseTopAggregator):
    """Oort Top level Aggregator implements an ML aggregation role."""

    def _oort_sim_recv(self, channel, end_ids):
        """Simulated recv for the oort stack, with cross-round straggler carry.

        Probes not-yet-buffered in-flight ends into a PERSISTENT reorder buffer,
        then yields buffered updates in ascending sim_completion_ts (advancing the
        virtual clock and stamping each end's PROP_ROUND_DURATION from
        SIM_ROUND_DURATION). It is a GENERATOR so the caller's existing
        "stop once aggr_num *accepted*" loop drives consumption — exactly mirroring
        real mode, where recv_fifo keeps delivering (and the loop cleans stale
        stragglers along the way) until aggr_num fresh updates land.

        The buffer persists across rounds: an overcommitment straggler (selected
        but not in this round's top by completion time) stays buffered and yields
        in a LATER round as a stale update, matching real (where it arrives late).
        Carrying it (vs the syncfl helper's per-call local buffer that drops it)
        is what makes sim staleness match real for REFL (which accepts stale), and
        keeps in-flight bounded — the straggler is finally cleaned instead of
        accumulating in selected_ends and inflating per-round selection. A popped
        end leaves the buffer, so a later re-selection re-probes its fresh update.
        Updates the caller doesn't consume (it broke early) stay buffered."""
        if not hasattr(self, "_sim_buffer"):
            self._sim_buffer = SimReorderBuffer()
        buf = self._sim_buffer
        # Barrier: drain the whole un-buffered set in one recv_fifo pass; the
        # yield-loop below commits in ascending sim_completion_ts order.
        to_probe = [e for e in end_ids if not buf.has(e)]
        barrier_t0 = time.time()
        drained_all = True
        if to_probe:
            grace = self._sim_recv_grace_s()
            for msg, md in channel.recv_fifo(
                to_probe, first_k=len(to_probe), timeout=grace
            ):
                if not msg:  # no more ready (grace expired or set drained)
                    break
                actual_end = md[0]
                sct = msg.get(MessageType.SIM_COMPLETION_TS)
                sct = float(sct) if sct is not None else self._vclock.now
                buf.add(actual_end, sct, (msg, md))
            drained_all = all(buf.has(e) for e in to_probe)
        barrier_wait = time.time() - barrier_t0
        if to_probe:
            self._note_sim_fill(barrier_wait, drained_all)
            logger.info(
                f"[SIM_BARRIER] round={getattr(self, '_round', -1)} probed={len(to_probe)} "
                f"barrier_wait_s={barrier_wait:.3f} buf_depth={len(buf)}"
            )

        # Carry-over gate: a prior-round straggler whose modeled
        # completion sct is still in the future at THIS round's start is STILL
        # COMPUTING — in real its update has not arrived, so it occupies its slot
        # (in-flight) rather than being delivered and stale-cleaned. Sim delivers it
        # physically at once; without the gate it is popped, stale-rejected, and freed
        # → in-flight drains to ~0 while real carries ~3 (overcommit). When on, hold
        # such stragglers in the buffer (and thus in selected_ends) until a later round
        # starts with vclock >= sct. Fresh (this-round) ends are always delivered; a
        # prior straggler that has already completed (sct <= round_start) is delivered
        # and stale-committed exactly as before. Default off.
        _hp = getattr(getattr(self, "config", None), "hyperparameters", None)
        carryover = bool(getattr(_hp, "sim_inflight_carryover", False))
        vclock_round_start = self._vclock.now
        held_over: list = []
        while True:
            popped = buf.pop_min()
            if popped is None:
                break
            end, sct, (msg, md) = popped
            if carryover:
                _tr = msg.get(MessageType.MODEL_VERSION, 0)
                if (self._round - _tr) > 0 and sct > vclock_round_start:
                    held_over.append((end, sct, (msg, md)))
                    continue
            self._advance_sim_clock(sct)
            _srd = msg.get(MessageType.SIM_ROUND_DURATION)
            _sst = channel.get_end_property(end, PROP_SIM_SEND_TS)
            if _srd is not None:
                channel.set_end_property(
                    end, PROP_ROUND_DURATION, timedelta(seconds=float(_srd))
                )
            elif _sst is not None:
                channel.set_end_property(
                    end, PROP_ROUND_DURATION,
                    timedelta(seconds=max(0.0, sct - float(_sst))),
                )
            yield msg, md
        # Re-buffer the still-computing stragglers so they carry to the next round
        # (occupying their in-flight slot) and commit once vclock reaches their sct.
        for _e, _sct, _payload in held_over:
            buf.add(_e, _sct, _payload)

    def _aggregate_weights(self, tag: str) -> None:
        """
        Aggregate local model weights, accepting K trainers out of
        1.3K clients selected from selector. Moreover, trainers' round
        duration is measured, which determine the system utility of
        trainers for Oort algorithm.

        This method is overriden from one in horizontal top aggregator
        (..top_aggregator).
        """
        channel = self.cm.get_by_tag(tag)
        if not channel:
            return

        total = 0

        # receive local model parameters from trainers terminate
        # aggregating when received weights from k ends (k *
        # overcommitment is selected for training with Oort)

        # CRITICAL: Use ALL in-flight trainers (selected_ends), not just newly selected
        # This allows us to receive stale updates from slow trainers selected in previous rounds
        # while also receiving fresh updates from trainers selected in this round.
        # This is essential for SyncFL with overcommitment where we select more than we wait for.
        if hasattr(channel._selector, 'selected_ends'):
            end_ids = list(channel._selector.selected_ends)
            logger.info(
                f"[AGGREGATE] Round {self._round}: Listening to ALL in-flight trainers. "
                f"Total in-flight={len(end_ids)} (includes old stragglers + newly selected)"
            )
        else:
            # Fallback to channel.ends() if selected_ends doesn't exist
            end_ids = channel.ends()
            logger.warning(
                f"[AGGREGATE] Round {self._round}: selected_ends not found, using channel.ends(). "
                f"Stale updates may not be consumed!"
            )

        # In-flight residence tracking: record the round each trainer
        # entered the in-flight set so cleanup can emit per-straggler residence. A
        # carryover straggler keeps its earlier entry round (setdefault); a
        # newly-selected one gets the current round.
        if not hasattr(self, "_inflight_entry_round"):
            self._inflight_entry_round = {}
        for _e in end_ids:
            self._inflight_entry_round.setdefault(_e, self._round)

        configured_aggr_num = self.config.selector.kwargs.get("aggr_num", 10)
        aggr_num = min(configured_aggr_num, len(end_ids))
        
        # CRITICAL: Log mismatch between configured and actual aggregation count
        if len(end_ids) < configured_aggr_num:
            logger.warning(
                f"[AGGREGATE] Round {self._round}: Aggregating with FEWER trainers than configured! "
                f"in_flight={len(end_ids)} < configured_aggr_num={configured_aggr_num}. "
                f"Will wait for {aggr_num} updates."
            )
        else:
            logger.info(
                f"[AGGREGATE] Round {self._round}: Waiting for {aggr_num} updates "
                f"from {len(end_ids)} in-flight trainers"
            )

        received_end_count = 0

        # simulated: commit the aggr_num updates with the smallest
        # sim_completion_ts (the k that would physically finish first in real),
        # reordering away physical arrival jitter and advancing the virtual
        # clock. Uses a persistent buffer so overcommitment stragglers carry
        # across rounds and commit late as stale (mirroring real). real: receive
        # by physical FIFO arrival (authentic baseline).
        if self.simulated:
            _recv = self._oort_sim_recv(channel, end_ids)
        else:
            _recv = channel.recv_fifo(end_ids, aggr_num)

        for msg, metadata in _recv:
            end, _ = metadata

            if not msg:
                logger.info(f"[MSG_SKIP] No data from ...{end[-8:]}; skipping it")
                continue

            # Calculate staleness
            trainer_round = msg.get(MessageType.MODEL_VERSION, 0)
            staleness = self._round - trainer_round

            # Check if optimizer supports REFL staleness (has stale_update_max attribute)
            stale_update_max = getattr(self.optimizer, 'stale_update_max', None)
            
            # If staleness > 0, check if we should accept or reject based on REFL config
            if staleness > 0:
                    # CRITICAL FIX: Even if we reject stale message, we MUST clean up the trainer
                    # from in-flight tracking. Otherwise, with overcommitment, trainers that don't
                    # make the top-K will be permanently stuck in selected_ends.
                    # This fixes the issue where 177 trainers accumulate over 400 rounds.
                    should_reject = False
                    
                    if stale_update_max is not None:
                        # REFL mode: check against stale_update_max threshold
                        if stale_update_max >= 0 and staleness > stale_update_max:
                            logger.info(
                                f"[MSG_SKIP] Message from ...{end[-8:]} TOO STALE: "
                                f"staleness={staleness} > stale_update_max={stale_update_max}; skipping it"
                            )
                            should_reject = True
                        else:
                            logger.info(
                                f"[MSG_ACCEPT_STALE] Stale message from ...{end[-8:]}, "
                                f"staleness={staleness} <= stale_update_max={stale_update_max}, "
                                f"accepting with weight degradation"
                            )
                    else:
                        # Non-REFL mode (standard Oort): reject all stale messages
                        logger.info(
                            f"[MSG_SKIP] Stale message from ...{end[-8:]}, "
                            f"expected_round={self._round}, got_round={trainer_round}; skipping it"
                        )
                        should_reject = True
                    
                    # Clean up trainer from in-flight set even if rejecting the update
                    if should_reject:
                        # Check if trainer is currently in selected_ends (in-flight)
                        is_in_flight = end in getattr(channel._selector, 'selected_ends', set())
                        channel._selector.ordered_updates_recv_ends.append(end)
                        logger.info(
                            f"[CLEANUP_STALE] Added stale trainer ...{end[-8:]} to cleanup queue. "
                            f"trainer_round={trainer_round}, current_round={self._round}, staleness={staleness}, "
                            f"currently_in_flight={is_in_flight}, cleanup_queue_size={len(channel._selector.ordered_updates_recv_ends)}"
                        )
                        # Track specific trainer ID for detailed debugging
                        test_id = '505f9fc483cf4df68a2409257b5fad7d3c580411'
                        if end == test_id:
                            logger.info(
                                f"[TRACK_411] Round {self._round}: Trainer 411 marked for cleanup. "
                                f"Was in_flight={is_in_flight}, staleness={staleness}"
                            )
                        continue

            total = self._handle_weights_msg(msg, metadata, channel, total)

            if end not in self._updates_recevied.keys():
                self._updates_recevied[end] = 1
            else:
                self._updates_recevied[end] += 1
            
            # CRITICAL: Notify selector that this trainer has returned its update
            # This prevents the selector from re-selecting this trainer in the next round
            # before it has returned its update (key for SyncFL with overcommitment)
            channel._selector.ordered_updates_recv_ends.append(end)
            
            logger.info(f"[MSG_ACCEPTED] Message from ...{end[-8:]} accepted, received_end_count={received_end_count + 1}/{aggr_num}")

            # remove end_id if it sends a valid message with correct
            # round info break the for loop if k valid messages arrive
            received_end_count += 1
            # Only remove if end is in end_ids (stale messages from previous rounds won't be)
            if end in end_ids:
                end_ids.remove(end)
            if received_end_count == aggr_num:
                break

        # running the second loop to aggregate up to aggr_num updates
        # from trainers. Real mode only: the sim path above already returned the
        # aggr_num smallest-sct updates in one shot (re-probing would block).
        while not self.simulated and received_end_count < aggr_num:
            for msg, metadata in channel.recv_fifo(end_ids, 1):
                end, _ = metadata

                if not msg:
                    logger.info(f"[MSG_SKIP] (loop2) No data from ...{end[-8:]}; skipping it")
                    continue

                # Calculate staleness
                trainer_round = msg.get(MessageType.MODEL_VERSION, 0)
                staleness = self._round - trainer_round

                # Check if optimizer supports REFL staleness (has stale_update_max attribute)
                stale_update_max = getattr(self.optimizer, 'stale_update_max', None)
                
                # If staleness > 0, check if we should accept or reject based on REFL config
                if staleness > 0:
                    # CRITICAL FIX: Even if we reject stale message, we MUST clean up the trainer
                    # from in-flight tracking. Otherwise, with overcommitment, trainers that don't
                    # make the top-K will be permanently stuck in selected_ends.
                    should_reject = False
                    
                    if stale_update_max is not None:
                        # REFL mode: check against stale_update_max threshold
                        if stale_update_max >= 0 and staleness > stale_update_max:
                            logger.info(
                                f"[MSG_SKIP] (loop2) Message from ...{end[-8:]} TOO STALE: "
                                f"staleness={staleness} > stale_update_max={stale_update_max}; skipping it"
                            )
                            should_reject = True
                        else:
                            logger.info(
                                f"[MSG_ACCEPT_STALE] (loop2) Stale message from ...{end[-8:]}, "
                                f"staleness={staleness} <= stale_update_max={stale_update_max}, "
                                f"accepting with weight degradation"
                            )
                    else:
                        # Non-REFL mode (standard Oort): reject all stale messages
                        logger.info(
                            f"[MSG_SKIP] (loop2) Stale message from ...{end[-8:]}, "
                            f"expected_round={self._round}, got_round={trainer_round}; skipping it"
                        )
                        should_reject = True
                    
                    # Clean up trainer from in-flight set even if rejecting the update
                    if should_reject:
                        # Check if trainer is currently in selected_ends (in-flight)
                        is_in_flight = end in getattr(channel._selector, 'selected_ends', set())
                        channel._selector.ordered_updates_recv_ends.append(end)
                        logger.info(
                            f"[CLEANUP_STALE] (loop2) Added stale trainer ...{end[-8:]} to cleanup queue. "
                            f"trainer_round={trainer_round}, current_round={self._round}, staleness={staleness}, "
                            f"currently_in_flight={is_in_flight}, cleanup_queue_size={len(channel._selector.ordered_updates_recv_ends)}"
                        )
                        # Track specific trainer ID for detailed debugging
                        test_id = '505f9fc483cf4df68a2409257b5fad7d3c580411'
                        if end == test_id:
                            logger.info(
                                f"[TRACK_411] Round {self._round}: (loop2) Trainer 411 marked for cleanup. "
                                f"Was in_flight={is_in_flight}, staleness={staleness}"
                            )
                        continue

                total = self._handle_weights_msg(msg, metadata, channel, total)
                
                # CRITICAL: Notify selector that this trainer has returned its update
                channel._selector.ordered_updates_recv_ends.append(end)
                
                logger.info(f"[MSG_ACCEPTED] (loop2) Message from ...{end[-8:]} accepted, received_end_count={received_end_count + 1}/{aggr_num}")

                # remove end_id if it sends a valid message with
                # correct round info break the for loop if k valid
                # messages arrive
                received_end_count += 1
                # Only remove if end is in end_ids (stale messages from previous rounds won't be)
                if end in end_ids:
                    end_ids.remove(end)
                if received_end_count == aggr_num:
                    break

        logger.debug(f"received {len(self.cache)} trainer updates in cache")

        # Aggregation-round telemetry (the OORT stack overrides _aggregate_weights
        # and otherwise emits none). Emit BEFORE optimizer.do, which consumes the
        # cache. Read from the cached TrainResult objects (staleness / stat_utility
        # / round_duration), and agg_observed_s = aggregator-side send->recv wall.
        if telemetry.is_enabled():
            contrib = list(self.cache)
            stale, sutil, speeds, agg_obs = [], [], [], {}
            for eid in contrib:
                tres = self.cache[eid]
                if getattr(tres, "staleness", None) is not None:
                    stale.append(tres.staleness)
                if getattr(tres, "stat_utility", None) is not None:
                    sutil.append(tres.stat_utility)
                rd = getattr(tres, "round_duration", None)
                if rd is not None:
                    speeds.append(rd)
                    agg_obs[eid] = rd
            ev, fields = build_agg_round(
                round_num=self._round,
                agg_goal=aggr_num,
                agg_goal_count=received_end_count,
                updates_in_queue=len(getattr(channel._selector, "selected_ends", []) or []),
                staleness=stale,
                stat_utility=sutil,
                trainer_speed_s=speeds,
                contributing_trainers=contrib,
                agg_observed_s=agg_obs or None,
                extra={"vclock_now": self._vclock.now if self.simulated else None},
            )
            telemetry.emit(ev, **fields)

        # optimizer conducts optimization (in this case, aggregation)
        _opt0 = time.time()
        global_weights = self.optimizer.do(
            deepcopy(self.weights), self.cache, total=total
        )
        # [AGG_COMMIT_TIMING] per-round aggregate cost (cache store in-memory +
        # optimizer + weight deepcopy).
        logger.info(
            f"[AGG_COMMIT_TIMING] round={self._round} "
            f"cache_store_s={getattr(self, '_agg_cache_store_s', 0.0):.4f} "
            f"optimizer_s={time.time() - _opt0:.4f}"
        )
        self._agg_cache_store_s = 0.0
        if global_weights is None:
            logger.debug("failed model aggregation")
            time.sleep(1)
            return

        self._compute_aggregator_stats()
        if self._round % 5 == 0:
            logger.info(f"_agg_training_stats: {self._agg_training_stats}")
        self._reset_aggregator_stats()

        # set global weights
        self.weights = global_weights

        # update model with global weights
        self._update_model()

        # CRITICAL: Clean up trainers who returned updates, freeing them from in-flight set
        # This must happen immediately after aggregation to prevent race condition where
        # trainers finishing after agg_goal but before next select() remain incorrectly
        # marked as busy, allowing them to be sent new work while still processing old work
        num_to_cleanup = len(channel._selector.ordered_updates_recv_ends)
        cleanup_list = channel._selector.ordered_updates_recv_ends.copy()
        in_flight_before = len(getattr(channel._selector, 'selected_ends', set()))
        
        # Check if specific test trainer is in cleanup list
        test_id = '505f9fc483cf4df68a2409257b5fad7d3c580411'
        test_in_cleanup = test_id in cleanup_list
        test_in_flight_before = test_id in getattr(channel._selector, 'selected_ends', set())
        
        logger.info(
            f"[CLEANUP_INVOKE] Round {self._round}: About to cleanup {num_to_cleanup} trainers. "
            f"in_flight_before={in_flight_before}"
        )
        if test_in_cleanup or test_in_flight_before:
            logger.info(
                f"[TRACK_411] Round {self._round}: Before cleanup - in_cleanup_list={test_in_cleanup}, "
                f"in_flight={test_in_flight_before}"
            )
        
        channel.cleanup_recvd_ends()
        
        in_flight_after = len(getattr(channel._selector, 'selected_ends', set()))
        test_in_flight_after = test_id in getattr(channel._selector, 'selected_ends', set())
        
        logger.info(
            f"[CLEANUP_COMPLETE] Round {self._round}: Freed {num_to_cleanup} trainers. "
            f"in_flight: {in_flight_before} -> {in_flight_after} (delta={in_flight_before - in_flight_after})"
        )
        if test_in_cleanup or test_in_flight_before:
            logger.info(
                f"[TRACK_411] Round {self._round}: After cleanup - in_flight={test_in_flight_after}, "
                f"successfully_freed={test_in_flight_before and not test_in_flight_after}"
            )

        # [INFLIGHT_RESIDENCE] per-straggler residence telemetry. residence
        # = rounds a cleaned trainer spent in selected_ends; carried_over_ages = ages of
        # those still in-flight. Comparing sim vs real residence distributions reveals
        # whether sim evicts stragglers a round too early (sim in-flight 13.4 vs real 15.6).
        _entry = getattr(self, "_inflight_entry_round", {})
        _resid = [self._round - _entry.pop(_e, self._round) for _e in cleanup_list]
        if telemetry.is_enabled():
            _remaining = getattr(channel._selector, "selected_ends", set()) or set()
            _ages = [self._round - _entry.get(_e, self._round) for _e in _remaining]
            ev, fields = build_inflight_residence(
                round_num=self._round,
                time_mode="sim" if self.simulated else "real",
                in_flight_before=in_flight_before,
                in_flight_after=in_flight_after,
                committed_fresh=received_end_count,
                cleaned=num_to_cleanup,
                stale_rejected=max(0, num_to_cleanup - received_end_count),
                residence_rounds=_resid,
                carried_over_ages=_ages,
            )
            telemetry.emit(ev, **fields)

        logger.info(
            f"====== aggregation finished for round {self._round}, "
            f"self._updates_recevied: "
            f"{self._updates_recevied}"
        )

    def _distribute_weights(self, tag: str, task_to_perform: str = "train") -> None:
        """
        Distribute local model weights to 1.3K clients, where K is the
        number of desired trainers to select. Moreover, measure the
        start time of a round to for round duration measurement on
        each trainers for measuring their system utility that is
        required for Oort algorithm.

        This method is overriden from one in horizontal top aggregator
        (..top_aggregator).
        """
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found for tag {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()
        # then wait for the configured cohort so real/sim select from the same pool
        self._await_min_trainers(channel)

        # Get desired number of trainers
        aggr_num = self.config.selector.kwargs.get("aggr_num", 10)
        overcommitment = getattr(channel._selector, 'overcommitment', 1.3)
        desired_selection = int(aggr_num * overcommitment)
        
        # Configuration for wait-retry mechanism
        max_retries = self.config.selector.kwargs.get('max_selection_retries', 5)
        retry_wait_seconds = self.config.selector.kwargs.get('selection_retry_wait', 2.0)
        min_trainers_ratio = self.config.selector.kwargs.get('min_trainers_ratio', 0.5)  # At least 50% of aggr_num
        
        # CRITICAL: If min_trainers_ratio >= overcommitment, wait for FULL desired_selection
        # This prevents sending weights to partial sets that may never respond
        if min_trainers_ratio >= overcommitment:
            min_required_trainers = desired_selection  # Wait for all 13 trainers
            logger.info(
                f"[DISTRIBUTE] Round {self._round}: Strict mode - will wait for FULL desired_selection={desired_selection} trainers"
            )
        else:
            min_required_trainers = max(1, int(aggr_num * min_trainers_ratio))
        
        logger.info(
            f"[DISTRIBUTE] Round {self._round}: Desired selection={desired_selection} "
            f"(aggr_num={aggr_num}, overcommit={overcommitment}), "
            f"min_required={min_required_trainers}"
        )

        # before distributing weights, update it from global model
        self._update_weights()

        # Per-baseline online oracle: overwrite candidate stat-utility with true
        # current values before the selector ranks. No-op unless enabled.
        self._inject_oracle_utilities(channel, task_to_perform)

        # before invoking channel.ends() to select, set the
        # trainer_unavail if it isn't None
        if self.trainer_event_dict is not None:
            curr_unavail_trainer_list = self.get_curr_unavail_trainers()
        else:
            curr_unavail_trainer_list = []

        # [SIM_RESIDENCE] Mark trainers that are STILL COMPUTING in
        # sim time as unavailable for this selection. In sim a dispatched trainer's
        # update arrives physically at once, so it can re-enter the eligible pool
        # before its modeled completion `sct`; real keeps it busy (out of the pool)
        # for its whole compute. A buffered end with `sct > vclock` is exactly such
        # a straggler. Excluding it via the unavailable list (NOT selected_ends —
        # that would re-dispatch it) keeps sim's eligible pool from carrying the slow
        # tail, matching real's pool composition (refl A2b 12.41->~6.5). Bounded:
        # released once `vclock >= sct` (the buffer pops & commits it). Default off.
        if self.simulated and getattr(
            self.config.hyperparameters, "sim_inflight_residence", False
        ):
            _buf = getattr(self, "_sim_buffer", None)
            if _buf is not None:
                _held = _buf.pending_after(self._vclock.now)
                if _held:
                    curr_unavail_trainer_list = list(
                        set(curr_unavail_trainer_list) | _held
                    )
                    logger.info(
                        f"[SIM_RESIDENCE] round={self._round} held {len(_held)} "
                        f"still-computing trainers out of selection "
                        f"(vclock={self._vclock.now:.1f})"
                    )

        channel.set_curr_unavailable_trainers(
            trainer_unavail_list=curr_unavail_trainer_list
        )

        logger.debug(
            f"Sending weights to trainers with task_to_perform = {task_to_perform}"
        )
        
        # CRITICAL FIX: Implement wait-retry mechanism for trainer selection
        # If insufficient trainers are available, wait and retry instead of proceeding
        selected_ends = None
        retry_count = 0
        
        while retry_count <= max_retries:
            # Get currently available trainer ends
            all_ends = list(channel._ends.keys())
            
            # Get unavailable trainers
            if self.trainer_event_dict is not None:
                unavail_trainers = set(self.get_curr_unavail_trainers())
            else:
                unavail_trainers = set()
            
            # Get in-flight trainers (already selected, waiting for updates)
            in_flight_trainers = getattr(channel._selector, 'selected_ends', set())
            if not isinstance(in_flight_trainers, set):
                in_flight_trainers = set(in_flight_trainers) if in_flight_trainers else set()
            
            # Calculate eligible trainers
            eligible_trainers = [
                end for end in all_ends
                if end not in unavail_trainers and end not in in_flight_trainers
            ]
            
            num_eligible = len(eligible_trainers)
            
            logger.info(
                f"[DISTRIBUTE] Round {self._round}, Attempt {retry_count + 1}/{max_retries + 1}: "
                f"total_ends={len(all_ends)}, unavailable={len(unavail_trainers)}, "
                f"in_flight={len(in_flight_trainers)}, eligible={num_eligible}, "
                f"required={min_required_trainers}"
            )
            
            # Check if we have enough eligible trainers
            if num_eligible >= min_required_trainers:
                logger.info(
                    f"[DISTRIBUTE] Round {self._round}: Sufficient trainers available "
                    f"({num_eligible} >= {min_required_trainers}), proceeding with selection. "
                    f"Will select min({desired_selection}, {num_eligible}) trainers."
                )
                break
            else:
                # Insufficient trainers - log warning
                logger.warning(
                    f"[DISTRIBUTE] Round {self._round}, Attempt {retry_count + 1}: "
                    f"INSUFFICIENT trainers! eligible={num_eligible} < required={min_required_trainers}. "
                    f"Breakdown: total={len(all_ends)}, unavail={len(unavail_trainers)}, "
                    f"in_flight={len(in_flight_trainers)}"
                )
                
                if retry_count < max_retries:
                    logger.warning(
                        f"[DISTRIBUTE] Waiting {retry_wait_seconds}s before retry {retry_count + 2}/{max_retries + 1}..."
                    )
                    time.sleep(retry_wait_seconds)
                    retry_count += 1
                    # Update unavailability list before retry
                    if self.trainer_event_dict is not None:
                        curr_unavail_trainer_list = self.get_curr_unavail_trainers()
                        channel.set_curr_unavailable_trainers(
                            trainer_unavail_list=curr_unavail_trainer_list
                        )
                else:
                    # Max retries exceeded - proceed with warning
                    logger.error(
                        f"[DISTRIBUTE] Round {self._round}: Max retries ({max_retries}) exceeded. "
                        f"Proceeding with ONLY {num_eligible} trainers (required: {min_required_trainers}, "
                        f"desired: {desired_selection}). THIS MAY IMPACT TRAINING QUALITY!"
                    )
                    break

        # Now perform the actual selection
        selected_ends = channel.ends(VAL_CH_STATE_SEND, task_to_perform)
        
        if not selected_ends or len(selected_ends) == 0:
            logger.error(
                f"[DISTRIBUTE] Round {self._round}: No trainers selected! "
                f"Cannot proceed with weight distribution."
            )
            return
        
        logger.info(
            f"[DISTRIBUTE] Round {self._round}: Selected {len(selected_ends)} trainers. "
            f"Will aggregate when {min(aggr_num, len(selected_ends))} updates received."
        )

        # Same model goes to every recipient this round; build + serialize once.
        _sim_send_ts = self._vclock.now if self.simulated else None
        msg = {
            MessageType.WEIGHTS: weights_to_device(self.weights, DeviceType.CPU),
            MessageType.ROUND: self._round,
            MessageType.MODEL_VERSION: self._round,
            MessageType.TASK_TO_PERFORM: task_to_perform,
        }
        if self.simulated:
            msg[MessageType.SIM_SEND_TS] = _sim_send_ts
        _payload = channel.dumps(msg)
        _send_t0 = time.time()
        for end in selected_ends:
            logger.info(
                f"sending weights to {end} with model_version: {self._round} for task: {task_to_perform}"
            )
            _send_ts = datetime.now()
            channel.set_end_property(
                end, PROP_ROUND_START_TIME, (self._round, _send_ts)
            )
            # Per-version send timestamp so stale-update SEND_RECV_LAG can use the
            # original send time for version N even after the round advances.
            if not hasattr(self, "_oort_sent_version_ts"):
                self._oort_sent_version_ts: dict = {}
            self._oort_sent_version_ts.setdefault(end, {})[self._round] = _send_ts
            if self.simulated:
                channel.set_end_property(end, PROP_SIM_SEND_TS, _sim_send_ts)
            channel.send_payload(end, _payload)
        if selected_ends:
            logger.info(
                f"[DISTRIBUTE_TIMING] round={self._round} n_sends={len(selected_ends)} "
                f"send_wall_s={time.time() - _send_t0:.3f}"
            )

    def _handle_weights_msg(
        self, msg: Any, metadata: Tuple[str, datetime], channel: Any, total: int
    ) -> int:
        end = metadata[0]
        timestamp = metadata[1]
        _t_msg_start = datetime.now()  # start of per-message processing (vii)

        logger.info(f"[MSG_PROCESSING] Processing message from end ...{end[-8:]}, round={self._round}, msg_version={msg.get(MessageType.MODEL_VERSION, 'N/A')}")
        logger.debug(f"received data from {end}")

        # calculate round duration for this end, if the round number
        # information is identical with round_start_time. In simulated mode the
        # sim recv path already set PROP_ROUND_DURATION from SIM_ROUND_DURATION;
        # the physical wall-clock delta here is ~0 (no sleeps), so don't clobber.
        round_start_time_tup = channel.get_end_property(end, PROP_ROUND_START_TIME)
        if not self.simulated and round_start_time_tup[0] == msg[MessageType.MODEL_VERSION]:
            channel.set_end_property(
                end, PROP_ROUND_DURATION, timestamp - round_start_time_tup[1]
            )

        # Per-version send-time lookup so stale updates (version N arriving in
        # round M > N) use the correct send timestamp for version N, not the
        # overwritten PROP_ROUND_START_TIME from the later re-selection.
        _msg_version = msg.get(MessageType.MODEL_VERSION, self._round)
        _sent_version_ts = getattr(self, "_oort_sent_version_ts", {})
        _sent_ts = _sent_version_ts.get(end, {}).get(_msg_version)
        if _sent_ts is None and isinstance(round_start_time_tup, tuple):
            # Fallback: if we somehow lack the per-version entry (e.g. process
            # resumed mid-run), use PROP_ROUND_START_TIME only when the version
            # matches so we don't measure the wrong round's lag.
            if round_start_time_tup[0] == _msg_version:
                _sent_ts = round_start_time_tup[1]
        if _sent_ts is not None:
            _recv_ts = timestamp if isinstance(timestamp, datetime) else datetime.now()
            wall_lag_s = (_recv_ts - _sent_ts).total_seconds()
            logger.info(
                f"[SEND_RECV_LAG] end={end} version={_msg_version} "
                f"wall_lag_s={wall_lag_s:.3f}"
            )
            # Full per-message lag decomposition into 6 components.
            _wst = msg.get(MessageType.WALL_SEND_TS)   # trainer send (float unix)
            _wrt = msg.get(MessageType.WALL_RECV_TS)   # trainer recv of agg weights (float unix)
            _rcs = msg.get(MessageType.ROUND_COMPUTE_S) # modeled compute duration (float s)
            _agg_sent_unix = _sent_ts.timestamp() if hasattr(_sent_ts, "timestamp") else None
            _agg_recv_unix = _recv_ts.timestamp() if hasattr(_recv_ts, "timestamp") else None
            _agg_to_trainer = f"{float(_wrt) - _agg_sent_unix:.3f}" if (_wrt and _agg_sent_unix) else "-"
            _compute = f"{float(_rcs):.3f}" if _rcs is not None else "-"
            _post_wait = f"{float(_wst) - float(_wrt) - float(_rcs):.3f}" if (_wst and _wrt and _rcs is not None) else "-"
            _mqtt_lag = f"{_agg_recv_unix - float(_wst):.3f}" if (_wst and _agg_recv_unix) else "-"
            _queue_wait = f"{(_t_msg_start - _recv_ts).total_seconds():.3f}"
            _process = f"{(datetime.now() - _t_msg_start).total_seconds():.3f}"
            logger.info(
                f"[LAG_DECOMP] end={end} version={_msg_version} "
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
                    # sim overrun: virtual round duration > budget.
                    # ROUND_COMPUTE_S = max(gpu, D) = SIM_ROUND_DURATION.
                    if _rcs is not None and float(_rcs) > _budget_s:
                        logger.warning(
                            f"[TIMING_OVERRUN_AGG] {end[-4:]} ver={_msg_version} "
                            f"budget={_budget_s:.1f}s overrun: "
                            f"virtual_elapsed={float(_rcs):.2f}s "
                            f"(excess={float(_rcs) - _budget_s:.2f}s). "
                            f"Reduce trainers-per-GPU or add GPUs."
                        )
                else:
                    if wall_lag_s > _budget_s + _NETWORK_SLACK_S:
                        logger.warning(
                            f"[TIMING_OVERRUN_AGG] {end[-4:]} ver={_msg_version} "
                            f"budget={_budget_s:.1f}s+slack={_NETWORK_SLACK_S:.1f}s "
                            f"overrun: wall_lag={wall_lag_s:.2f}s "
                            f"(excess={wall_lag_s - _budget_s - _NETWORK_SLACK_S:.2f}s). "
                            f"Reduce trainers-per-GPU or add GPUs."
                        )

        # Lazy-deserialize: restore the tensor from WEIGHTS_BYTES (only paid for
        # this committed update). weights defaults to None so an eval-only or
        # malformed message can never UnboundLocalError at the `weights is not
        # None` check below.
        weights = None
        if materialize_weights(msg) is not None:
            weights = weights_to_model_device(msg[MessageType.WEIGHTS], self.model)

        if MessageType.DATASET_SIZE in msg:
            count = msg[MessageType.DATASET_SIZE]

        if MessageType.STAT_UTILITY in msg:
            # Believed-vs-actual utility telemetry: the PROP_STAT_UTILITY held NOW
            # (before this return overwrites it) is what the selector BELIEVED at
            # selection (stale by `staleness` rounds); the incoming value is the
            # ACTUAL fresh utility. Emit before overwriting. (believed-vs-actual)
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
            logger.info(
                f"End {end} sent a message with utility {msg[MessageType.STAT_UTILITY]}"
            )

        trainer_model_version = 0  # default
        if MessageType.MODEL_VERSION in msg:
            channel.set_end_property(
                end, PROP_LAST_SELECTED_ROUND, msg[MessageType.MODEL_VERSION]
            )
            trainer_model_version = msg[MessageType.MODEL_VERSION]
            logger.info(
                f"End {end} sent a model update version {msg[MessageType.MODEL_VERSION]}, while current model version {self._round}"
            )

        # Set last eval round for the trainer since training also
        # means that eval was done for the same round.
        channel.set_end_property(end, PROP_LAST_EVAL_ROUND, trainer_model_version)

        stat_utility = 0  # default
        if MessageType.STAT_UTILITY in msg:
            channel.set_end_property(
                end, PROP_STAT_UTILITY, msg[MessageType.STAT_UTILITY]
            )
            stat_utility = msg[MessageType.STAT_UTILITY]

        logger.debug(f"{end}'s parameters trained with {count} samples")

        if weights is not None and count > 0:
            total += count
            
            # Calculate staleness BEFORE creating TrainResult
            update_staleness_val = self._round - trainer_model_version
            
            # Get round duration if available
            round_duration_obj = channel.get_end_property(end, PROP_ROUND_DURATION)
            round_duration_seconds = None
            if round_duration_obj:
                round_duration_seconds = round_duration_obj.total_seconds()
            
            # Create TrainResult with all REFL-required fields
            tres = TrainResult(
                weights=weights,
                count=count,
                version=trainer_model_version,
                stat_utility=stat_utility,
                staleness=update_staleness_val,
                round_duration=round_duration_seconds,
                end_id=end
            )
            
            _cs0 = time.time()
            self.cache[end] = tres   # in-memory (MemCache)
            self._agg_cache_store_s = (
                getattr(self, "_agg_cache_store_s", 0.0) + time.time() - _cs0)

            logger.debug(
                f"Created TrainResult for {end}: staleness={update_staleness_val}, "
                f"stat_utility={stat_utility}, round_duration={round_duration_seconds}"
            )

            # Populate round statistics vars
            self._round_update_values["staleness"].append(update_staleness_val)
            self._round_update_values["stat_utility"].append(stat_utility)
            # Only append trainer_speed if round_duration is available
            if round_duration_seconds is not None:
                self._round_update_values["trainer_speed"].append(round_duration_seconds)
            else:
                logger.debug(
                    f"Skipping trainer_speed for {end} - round_duration is None "
                    f"(stale update: msg_version={trainer_model_version}, current_round={self._round})"
                )

        return total
