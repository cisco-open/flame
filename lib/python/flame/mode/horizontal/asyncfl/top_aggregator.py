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
from datetime import datetime

import numpy as np
from flame.channel import VAL_CH_STATE_HTBT_RECV, VAL_CH_STATE_RECV, VAL_CH_STATE_SEND
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

        # check if distribute_weights was successful
        self._prev_distribute_weights_success = False

        # variables related to checking trainer availability
        self._per_trainer_last_heartbeat_ts = {}
        if "heartbeat_freq_s" in self.config.hyperparameters.track_trainer_avail.keys():
            self._trainer_heartbeat_freq_s = (
                self.config.hyperparameters.track_trainer_avail["heartbeat_freq_s"]
            )
        else:
            self._trainer_heartbeat_freq_s = 99999

        if (
            "max_allowed_miss_heartbeats"
            in self.config.hyperparameters.track_trainer_avail.keys()
        ):
            self._trainer_max_miss_heartbeats = (
                self.config.hyperparameters.track_trainer_avail[
                    "max_allowed_miss_heartbeats"
                ]
            )
        else:
            self._trainer_max_miss_heartbeats = 99999

        # maintain a set of all trainers that have sent heartbeats
        # previously
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

    def _aggregate_weights(self, tag: str) -> None:
        """Aggregate local model weights asynchronously.

        This method is overriden from one in synchronous top
        aggregator (..top_aggregator).
        """
        logger.info("Agg weights inside top_aggregator asyncfl")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug("No channel found")
            return

        logger.debug(f"Channel {channel} found for tag {tag}")
        # receive local model parameters from a trainer who arrives
        # first NOTE: (DG) Right now, the leave notifications also
        # cause a message to be processed and yield (None,None) from
        # recv_fifo().
        msg, metadata = next(channel.recv_fifo(channel.ends(VAL_CH_STATE_RECV), 1))
        end, _ = metadata
        if not msg:
            logger.debug(f"No data from {end}; skipping it")
            return

        # NOTE: Only 2 types of messages are expected here: (i) model
        # updates after task_to_perform=TRAIN with weights or (ii)
        # statistical utility updates after task_to_perform=EVAL with
        # info on stat_utility. Else, throw an error.

        # Case #1: Message after task_to_perform=TRAIN. This will
        # contain stat_utility too but will processed later.
        if MessageType.WEIGHTS in msg:
            logger.info(
                f"received model updates from {end} "
                f"with model version {msg[MessageType.MODEL_VERSION]}"
            )

            # For OORT selector NOTE: (DG) Last selected round should
            # have ideally been set in distribute weights. But it was
            # here in the old oort code and ive kept it. Instead of
            # PROP_LAST_SELECTED_ROUND, it should have been
            # PROP_LAST_UPDATE_RECVD_ROUND.
            channel.set_end_property(
                end, PROP_LAST_SELECTED_ROUND, msg[MessageType.MODEL_VERSION]
            )

            # Set last eval round for the trainer since training also
            # means that eval was done for the same round.
            channel.set_end_property(
                end, PROP_LAST_EVAL_ROUND, msg[MessageType.MODEL_VERSION]
            )
            # calculate round duration for this end, if the round
            # number information is identical with round_start_time
            logger.debug(
                f"Getting channel property {PROP_ROUND_START_TIME} for " f"end {end}"
            )
            round_start_time_tup = channel.get_end_property(end, PROP_ROUND_START_TIME)
            end = metadata[0]
            timestamp = metadata[1]
            logger.debug(
                f"Returned round_start_time_tup: {round_start_time_tup} for "
                f"end {end} and timestamp {timestamp}"
            )

            # TODO: (DG) Also set the end property for task=eval done
            # at timestamp=current.

        # Case #2: Message after task_to_perform=EVAL
        elif MessageType.STAT_UTILITY in msg:
            logger.info(
                f"received eval message {msg} in agg_weights from {end}, "
                f"with stat_utility {msg[MessageType.STAT_UTILITY]} after "
                f"round {msg[MessageType.MODEL_VERSION]}. Updating end property"
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
                curr_round_time_s = (recv_wts_ts - sent_wts_ts).total_seconds()
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
                    f"end {end} with duration "
                    f"{recv_wts_ts - sent_wts_ts}"
                )
                channel.set_end_property(
                    end, PROP_ROUND_DURATION, recv_wts_ts - sent_wts_ts
                )

        # capture telemetry on trainer participation in rounds
        channel._selector.ordered_updates_recv_ends.append(end)
        logger.debug(
            f"After appending {end} to ordered_updates_recv_ends: "
            f"{channel._selector.ordered_updates_recv_ends}"
        )

        self._updates_in_queue += 1

        self._per_round_update_list.append(end) #SC_TS: PER Round! In Async FwdLLM -> we need to decide whether per data bin or per iteration!

        # SC_TS: what is even the diff between this and  self._updates_in_queue += 1??!!
        if end not in self._updates_recevied.keys():
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

        stat_utility = 0        # default
        if MessageType.STAT_UTILITY in msg:
            channel.set_end_property(
                end, PROP_STAT_UTILITY, msg[MessageType.STAT_UTILITY]
            )
            stat_utility = msg[MessageType.STAT_UTILITY]

        logger.info(
            f"Received weights from {end}. It was trained on model version {version}, with {count} samples. Returned stat utility {stat_utility}"
        )

        if weights is not None and count > 0: #SC_TS: count = 0 means no data (it was trained on!), so ignore!
            tres = TrainResult(weights, count, version, stat_utility)
            # save training result from trainer in a disk cache
            self.cache[end] = tres
            logger.debug(f"received {len(self.cache)} trainer updates in cache")
            update_staleness_val = self._round - tres.version
            logger.info(f"Received update from {end}. agg_version: {self._round}, trainer version: {tres.version}, update_staleness_val: {update_staleness_val}")
            
            # Populate round statistics vars
            self._round_update_values["staleness"].append(update_staleness_val)
            self._round_update_values["stat_utility"].append(stat_utility)
            self._round_update_values["trainer_speed"].append(channel.get_end_property(end_id=end, key=PROP_ROUND_DURATION).total_seconds())

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
            # SC_TS: append weights to this list, till agg goal reached! 
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
            # didn't reach the aggregation goal; return
            logger.debug(f"didn't reach agg goal. _agg_goal_cnt: {self._agg_goal_cnt} while _agg_goal is {self._agg_goal}")

            # Set trainer participation count property here to be used
            # later in selection.
            channel.set_end_property(
                end, PROP_UPDATE_COUNT, self._updates_recevied[end]
            )
            return

        if self._agg_goal_weights is None:
            logger.debug("failed model aggregation")
            time.sleep(1)
            return

        # set global weights, by adding scaled aggregated weights with
        # aggregation goal
        if self._agg_goal_cnt == self._agg_goal:
            logger.info(f"reached agg goal since _agg_goal_cnt: {self._agg_goal_cnt} and _agg_goal is: {self._agg_goal}")
            logger.debug(
                f"Reached agg_goal {self._agg_goal}, "
                f"current _updates_in_queue: {self._updates_in_queue}, "
                f"current round before agg: {self._round}"
            )

            # update per-trainer participation in round agg
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

        # Computing rate: Not used anywhere right now rate = 1 /
        # math.sqrt(1 + self._round - tres.version) logger.debug(f"
        # rate at top_agg: {rate}")

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
        """Distribute a global model in asynchronous FL fashion.

        This method is overridden from one in synchronous top
        aggregator (..top_aggregator).
        """
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found for tag {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        # before distributing weights, update it from global model
        self._update_weights()

        # busy wait for 0.1 seconds before proceeding. This is to wait
        # on distribute_weights to let the system state get updated
        # before selector is invoked again
        logger.debug(f"Starting busy wait at time {time.time()}")
        time.sleep(0.1)
        logger.debug(f"Ended busy wait at time {time.time()}")

        # before invoking channel.ends() to select, set the
        # trainer_unavail if it isn't None
        if self.trainer_event_dict is not None:
            curr_unavail_trainer_list = self.get_curr_unavail_trainers()
            channel.set_curr_unavailable_trainers(
                trainer_unavail_list=curr_unavail_trainer_list
            )
            logger.debug(
                f"Passed curr_unavail_trainer_list: "
                f"{curr_unavail_trainer_list} to channel"
            )
        else:
            # Handling the case for oort's selector since it expects 3
            # arguments
            channel.set_curr_unavailable_trainers(trainer_unavail_list=[])

        # check if there are any ends to send weights to

        logger.debug(
            f"Sending weights to trainers with task_to_perform = {task_to_perform}"
        )
        ends = channel.ends(VAL_CH_STATE_SEND, task_to_perform)
        # NRL TODO: else will take care of randomly selecting x
        # trainers for "eval only" operation
        if not ends:
            logger.debug(
                f"No trainers found for tag {tag}, will "
                f"move to get() for fetch weights from trainers"
            )
            return

        # send out global model parameters to trainers
        for end in ends:
            # Send shouldn't be allowed if already sent to a trainer
            # in that same round
            logger.info(
                f"sending weights to {end} with model_version: {self._round} for task: {task_to_perform}"
            )

            # setting start time for OORT TODO: (DG) round_start_time
            # for all trainers in the same round may not be the same
            logger.debug(
                f"Setting channel property {PROP_ROUND_START_TIME} for "
                f"end {end}. For round {self._round} at time: {datetime.now()}"
            )
            channel.set_end_property(
                end, PROP_ROUND_START_TIME, (self._round, datetime.now())
            )

            # we use _round to indicate a model version
            channel.send(
                end,
                {
                    MessageType.WEIGHTS: weights_to_device(
                        self.weights, DeviceType.CPU
                    ),
                    MessageType.ROUND: self._round,
                    MessageType.MODEL_VERSION: self._round,
                    MessageType.TASK_TO_PERFORM: task_to_perform,
                },
            )

            # Update send_time in training_duration_s
            if end not in self._track_trainer_version_duration_s.keys():
                logger.debug(
                    f"{end} not in _track_trainer_version_duration_s, " f"will add"
                )
                self._track_trainer_version_duration_s[end] = dict()
                self._track_trainer_version_duration_s[end]["last_send_wts_ts"] = -1

                # sent_wts_version_ts, recv_wts_version_ts is a dict
                # of version sent/recv and its timestamp. This will be
                # primarily used by AsyncOORT selector since it needs
                # round_duration times. TODO: (DG) Right now the dict
                # maintains ALL sent/recv versions and timestamps for
                # all trainers. For thousands of trainers it might
                # incur memory-bloat. Can optimize to retain just the
                # versions and timestamps of those that were sent but
                # not received back for the trainer.
                self._track_trainer_version_duration_s[end]["sent_wts_version_ts"] = {}
                self._track_trainer_version_duration_s[end]["recv_wts_version_ts"] = {}
                self._track_trainer_version_duration_s[end][
                    "total_training_time_s"
                ] = -1

            # Update sent_wts_version_ts with version and timestamp
            self._track_trainer_version_duration_s[end]["sent_wts_version_ts"][
                self._round
            ] = datetime.now()

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
