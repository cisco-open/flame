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
from datetime import datetime, timedelta

import numpy as np
from flame.channel import VAL_CH_STATE_HTBT_RECV, VAL_CH_STATE_RECV, VAL_CH_STATE_SEND
from flame.end import KEY_END_STATE, VAL_END_STATE_NONE
from flame.common.constants import DeviceType
from flame.common.util import weights_to_device, weights_to_model_device
from flame.mode.composer import CloneComposer
from flame.mode.horizontal.syncfl.top_aggregator import (
    TAG_AGGREGATE,
    TAG_DISTRIBUTE,
    TAG_HEARTBEAT,
)
from flame.mode.horizontal.syncfl.top_aggregator import TopAggregator as SyncTopAgg
from flame.mode.message import MessageType
from flame.mode.tasklet import Loop, Tasklet
from flame.optimizer.train_result import TrainResult
from flame import telemetry
from flame.telemetry.events import build_agg_round
from flame.sim import SimReorderBuffer
from flame.selector.properties import PROP_SIM_SEND_TS, PROP_SIM_COMPLETION_TS
from flame.selector.oort import (
    PROP_DATASET_SIZE,
    PROP_LAST_SELECTED_ROUND,
    PROP_LAST_EVAL_ROUND,
    PROP_ROUND_DURATION,
    PROP_ROUND_START_TIME,
    PROP_STAT_UTILITY,
    PROP_UPDATE_COUNT,
)

logger = logging.getLogger(__name__)

SEND_TIMEOUT_WAIT_S = 90  # 90 seconds timeout

# Max wall-clock to block on one async receive before skipping the cycle and
# re-selecting; guards against hanging when all in-flight trainers go quiet.
RECV_TIMEOUT_WAIT_S = 30


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

    # Short per-end probe timeout: how long we wait for one trainer's MQTT
    # message before moving on. Not the training wait — just queue delivery jitter.
    SIM_RECV_FILL_TIMEOUT_S = 0.5

    def _sim_recv_min(self, channel, recv_ends):
        """Fill the reorder buffer from un-buffered, un-committed ends; pop and
        commit the entry with the smallest sim_completion_ts."""
        to_probe = [
            e for e in recv_ends
            if not self._sim_buffer.has(e) and e not in self._sim_committed
        ]
        for e in to_probe:
            msg, metadata = next(
                channel.recv_fifo([e], 1, timeout=self.SIM_RECV_FILL_TIMEOUT_S)
            )
            if msg is not None:
                # metadata[0] is the actual sender; may differ from probed end
                # if a stale recv task delivered a different end's message first.
                actual_end = metadata[0]
                sct = msg.get(MessageType.SIM_COMPLETION_TS)
                if sct is None:
                    sct = self._vclock.now
                self._sim_buffer.add(actual_end, float(sct), (msg, metadata))

        # Pop the minimum regardless of recv_ends membership so buffered updates
        # are not lost when an end is cleaned up before its commit.
        popped = self._sim_buffer.pop_min()
        if popped is None:
            time.sleep(0.5)
            return None, ("", datetime.now())
        _end, sct, (m, md) = popped
        self._vclock.advance(sct)
        self._sim_committed.add(_end)
        logger.debug(
            f"[SIM_RECV] committed end={_end[-4:]} sct={sct:.1f} "
            f"T_v={self._vclock.now:.1f} buf={len(self._sim_buffer)}"
        )
        # Release trainer that was blocked waiting for this cross-round commit.
        if _end in self._sim_pending_commit:
            self._sim_pending_commit.discard(_end)
            sel = channel._selector
            if _end in sel.all_selected:
                del sel.all_selected[_end]
            if channel.has(_end):
                channel._ends[_end].set_property(KEY_END_STATE, VAL_END_STATE_NONE)
            logger.info(f"[SIM_PENDING_COMMIT] released {_end[-4:]} sct={sct:.1f}")
        return m, md

    def _aggregate_weights(self, tag: str) -> None:
        """Aggregate local model weights asynchronously.

        This method is overriden from one in synchronous top
        aggregator (..top_aggregator).
        """
        logger.info(f"[AGG_START] Agg weights inside top_aggregator asyncfl, current model_version={self._round}")
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

        # NOTE: Only 2 types of messages are expected here: (i) model
        # updates after task_to_perform=TRAIN with weights or (ii)
        # statistical utility updates after task_to_perform=EVAL with
        # info on stat_utility. Else, throw an error.

        # Case #1: Message after task_to_perform=TRAIN. This will
        # contain stat_utility too but will processed later.
        if MessageType.WEIGHTS in msg:
            logger.info(
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
            logger.info(
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

            # add trainer to list of ends that have replied with eval
            # updates capture telemetry on trainer participation in
            # rounds
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
            recv_wts_ts = datetime.now()
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
                _lag_warn_threshold_s = 30.0 if not self.simulated else 10.0
                if wall_lag_s > _lag_warn_threshold_s:
                    logger.warning(
                        f"[SEND_RECV_LAG_HIGH] end={end} version={recv_wts_version} "
                        f"wall_lag_s={wall_lag_s:.1f}s — possible MQTT backlog"
                    )

                _budget_s = float(msg.get(MessageType.TRAINING_BUDGET_S, 0.0))
                if _budget_s > 0:
                    if self.simulated:
                        _sst = channel.get_end_property(end, PROP_SIM_SEND_TS)
                        _sct = msg.get(MessageType.SIM_COMPLETION_TS)
                        if _sst is not None and _sct is not None:
                            _virt_elapsed = float(_sct) - float(_sst)
                            if _virt_elapsed <= 0.0:
                                logger.warning(
                                    f"[TIMING_OVERRUN_AGG] {end[-4:]} ver={recv_wts_version} "
                                    f"budget={_budget_s:.1f}s overrun: virtual_elapsed={_virt_elapsed:.2f}s. "
                                    f"Reduce trainers-per-GPU or add GPUs."
                                )
                    else:
                        if wall_lag_s > _budget_s:
                            logger.warning(
                                f"[TIMING_OVERRUN_AGG] {end[-4:]} ver={recv_wts_version} "
                                f"budget={_budget_s:.1f}s overrun: wall_lag={wall_lag_s:.2f}s "
                                f"(excess={wall_lag_s - _budget_s:.2f}s). "
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
                # Both modes: round_duration = max(gpu, D); mirrors recv_ts-sent_ts for OORT utility.
                if self.simulated:
                    round_duration_td = timedelta(
                        seconds=float(msg.get(MessageType.SIM_ROUND_DURATION, 0.0))
                    )
                else:
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
                    f"Setting channel property {PROP_ROUND_DURATION} for "
                    f"end {end} with duration {round_duration_td}"
                )
                channel.set_end_property(
                    end, PROP_ROUND_DURATION, round_duration_td
                )

        channel._selector.ordered_updates_recv_ends.append(end)
        self._updates_in_queue += 1
        self._per_round_update_list.append(end)
        if end not in self._updates_recevied:
            self._updates_recevied[end] = 1
        else:
            self._updates_recevied[end] += 1

        # Process the weights and send to optimizer
        if MessageType.WEIGHTS in msg:
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
            channel.set_end_property(
                end, PROP_STAT_UTILITY, msg[MessageType.STAT_UTILITY]
            )
            stat_utility = msg[MessageType.STAT_UTILITY]

        logger.info(
            f"Received weights from {end}. It was trained on model version {version}, with {count} samples. Returned stat utility {stat_utility}"
        )

        if (
            weights is not None and count > 0
        ):  # SC_TS: count = 0 means no data (it was trained on!), so ignore!
            tres = TrainResult(weights, count, version, stat_utility)
            # save training result from trainer in a disk cache
            self.cache[end] = tres
            logger.debug(f"received {len(self.cache)} trainer updates in cache")
            update_staleness_val = self._round - tres.version
            logger.info(
                f"Received update from {end}. agg_version: {self._round}, trainer version: {tres.version}, update_staleness_val: {update_staleness_val}"
            )

            # Populate round statistics vars
            self._round_update_values["staleness"].append(update_staleness_val)
            self._round_update_values["stat_utility"].append(stat_utility)
            _round_dur = channel.get_end_property(end_id=end, key=PROP_ROUND_DURATION)
            _trainer_speed_s = _round_dur.total_seconds() if _round_dur is not None else 0.0
            self._round_update_values["trainer_speed"].append(_trainer_speed_s)

            if telemetry.is_enabled():
                _sct_recv = msg.get(MessageType.SIM_COMPLETION_TS)
                ev, fields = build_agg_round(
                    round_num=self._round,
                    agg_goal=self._agg_goal,
                    agg_goal_count=self._agg_goal_cnt,
                    updates_in_queue=self._updates_in_queue,
                    staleness=[update_staleness_val],
                    stat_utility=[stat_utility],
                    trainer_speed_s=[_trainer_speed_s],
                    contributing_trainers=[end],
                    extra={
                        "sim_completion_ts_recv": float(_sct_recv) if _sct_recv is not None else None,
                        "vclock_now": self._vclock.now if self.simulated else None,
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

            logger.info("proceeding to agg weights")
            self._agg_goal_weights = self.optimizer.do(
                self._agg_goal_weights,
                self.cache,
                total=count,
                version=self._round,
                staleness_factor=0.0,
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
            logger.info(
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

        logger.info(
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
            logger.info(f"_agg_training_stats: {self._agg_training_stats}")
        self._reset_aggregator_stats()

        # per trainer analytics
        if self._round % 100 == 0:
            for k, v in self._per_trainer_staleness_track.items():
                trainer_staleness_arr = np.array(v)
                logger.info(
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
        logger.info(
            f"Avg training time {avg_training_time} across "
            f"{len(self._track_trainer_version_duration_s)} trainers"
        )

        logger.debug("Agg goal reached, so resetting trainer end states in the channel")
        channel.cleanup_recvd_ends()

        if self.simulated:
            sel = channel._selector
            requester = sel.requester
            pending_in_buffer = set(self._sim_buffer.pending_ends())

            # Release all_selected trainers with no buffer entry yet (GPU still
            # running — rare). They'll be probed next round's fill pass.
            for end_id in [e for e in list(sel.all_selected.keys()) if e not in pending_in_buffer]:
                del sel.all_selected[end_id]
                if channel.has(end_id):
                    channel._ends[end_id].set_property(KEY_END_STATE, VAL_END_STATE_NONE)
                if requester in sel.selected_ends and end_id in sel.selected_ends[requester]:
                    sel.selected_ends[requester].discard(end_id)

            # Block every trainer with a pending buffer entry from re-selection.
            # cleanup_recvd_ends may have freed them early; re-block here so the
            # real-mode invariant holds: one in-flight update per trainer at a time.
            for end_id in pending_in_buffer:
                self._sim_pending_commit.add(end_id)
                if end_id not in sel.all_selected:
                    sel.all_selected[end_id] = time.time()
                if requester in sel.selected_ends and end_id in sel.selected_ends[requester]:
                    sel.selected_ends[requester].discard(end_id)
            if pending_in_buffer:
                logger.info(
                    f"[SIM_PENDING] round={self._round} blocked {len(pending_in_buffer)} "
                    f"trainer(s) pending buffer commit: {[e[-4:] for e in pending_in_buffer]}"
                )

    def oracular_trainer_avail_check(self, end: str) -> bool:
        logger.debug("In oracular_trainer_avail_check")

        picked_trainer_is_available = True

        if end in self.trainer_unavail_durations.keys():
            # get aggregator seconds from start
            agg_time_since_start_s = time.time() - self.agg_start_time_ts

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

    def _distribute_weights(self, tag: str, task_to_perform: str = "train") -> None:
        """Distribute a global model in asynchronous FL fashion."""
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found for tag {tag}")
            return

        channel.await_join()
        self._update_weights()

        # Brief pause so channel state settles before selector runs.
        time.sleep(0.1)

        if self.trainer_event_dict is not None:
            curr_unavail_trainer_list = self.get_curr_unavail_trainers()
            channel.set_curr_unavailable_trainers(
                trainer_unavail_list=curr_unavail_trainer_list
            )
        else:
            channel.set_curr_unavailable_trainers(trainer_unavail_list=[])

        # Expose current vclock to selector so it can attach it to selection events.
        if self.simulated:
            channel.properties["vclock_now"] = self._vclock.now

        ends = channel.ends(VAL_CH_STATE_SEND, task_to_perform)
        if not ends:
            logger.debug(f"No trainers found for tag {tag}")
            return

        ends_list = list(ends)
        for idx, end in enumerate(ends_list):
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

            logger.info(
                f"sending weights to {end} model_version={self._round} task={task_to_perform}"
            )
            channel.set_end_property(
                end, PROP_ROUND_START_TIME, (self._round, datetime.now())
            )

            msg = {
                MessageType.WEIGHTS: weights_to_device(self.weights, DeviceType.CPU),
                MessageType.ROUND: self._round,
                MessageType.MODEL_VERSION: self._round,
                MessageType.TASK_TO_PERFORM: task_to_perform,
            }
            # Stamp virtual send-time so trainer can compute sim_completion_ts correctly.
            # Without this, _sim_send_ts stays None on the trainer → _sim_now()=0 →
            # sim_completion_ts = D for all trainers regardless of when task was dispatched,
            # collapsing the reorder buffer to sort by tiny GPU-time differences only.
            if self.simulated:
                sim_send_ts = self._vclock.now
                msg[MessageType.SIM_SEND_TS] = sim_send_ts
                channel.set_end_property(end, PROP_SIM_SEND_TS, sim_send_ts)
                logger.debug(
                    f"[SIM_SEND] end={end[-4:]} round={self._round} sim_send_ts={sim_send_ts:.2f}"
                )

            channel.send(end, msg)

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

            # Stagger sends: shorter interval in sim (no real sleeps) vs real mode.
            if idx < len(ends_list) - 1:
                time.sleep(0.2 if self.simulated else 0.5)

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
