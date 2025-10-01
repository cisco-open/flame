# Copyright 2023 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#
# SPDX-License-Identifier: Apache-2.0
"""SyncFL horizontal FL top level aggregator for FwdLLM."""

import gc
import logging
import psutil
import time
from datetime import datetime
import sklearn
import numpy as np
from flame.channel import VAL_CH_STATE_HTBT_RECV, VAL_CH_STATE_RECV, VAL_CH_STATE_SEND
from flame.common.constants import DeviceType
from flame.common.util import weights_to_device, weights_to_model_device
from flame.mode.composer import CloneComposer
import pickle
from flame.mode.horizontal.syncfl.top_aggregator import (
    TAG_AGGREGATE,
    TAG_DISTRIBUTE,
    TAG_HEARTBEAT,
)
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
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
import functorch as fc
import torch

from torch.nn import CrossEntropyLoss

logger = logging.getLogger(__name__)

PROP_ROUND_END_TIME = "round_end_time"

SEND_TIMEOUT_WAIT_S = 90  # 90 seconds timeout


class TopAggregator(SyncTopAgg):
    """Asynchronous top level Aggregator implements an ML aggregation
    role."""

    def internal_init(self) -> None:
        """Initialize internal state for role."""
        super().internal_init()

        self._agg_goal_cnt = 0
        self._agg_goal_weights = None
        self._agg_goal = self.config.hyperparameters.aggregation_goal or 1

        self._updates_in_queue = 0
        self._updates_recevied = {}
        self._trainer_participation_in_round_count = {}
        self._trainer_participation_in_round = {}
        self._per_round_update_list = []
        self._per_round_staleness_list = []
        self._aggregator_staleness_track_rounds = []
        self._aggregator_round_avg_staleness = []
        self._per_trainer_staleness_track = {}
        self._track_trainer_version_duration_s = {}

        # check if distribute_weights was successful
        self._prev_distribute_weights_success = False

        self.data_id = 0
        self.total_data_bins = 150

        self.grad_pool = []
        self.var = None
        self.ends_not_selected_yet = False
        self.iteration_per_data_id = 0
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

        # maintain a set of all trainers that have sent heartbeats previously
        self.all_trainers = set()
        try:
            self.minInitialTrainers = self.config.selector.kwargs.get("minInitialTrainers")
            assert self.minInitialTrainers is not None
        except (KeyError, AssertionError):
            raise KeyError("minInitialTrainers must be specified in selector config & must not be None for determinism")
        logger.info("finished init for sync agg")

    def pause_execution(self):
        time.sleep(1)
        return

    def log_memory(self, tag, device):
        # GPU memory
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)

        # CPU memory
        process = psutil.Process()
        cpu_memory = process.memory_info().rss  # in bytes

        logging.info(
            f"[MEM:{tag}] "
            f"GPU Allocated: {allocated/1e6:.2f} MB | "
            f"GPU Reserved: {reserved/1e6:.2f} MB | "
            f"CPU Memory: {cpu_memory/1e6:.2f} MB | "
            f"Device: {device}, aggregator"
        )

    def print_trainable_params_stats(self, location=""):
        total_params = 0
        trainable_params = 0
        total_size = 0.0
        trainable_size = 0.0

        for param in self.model.parameters():
            numel = param.numel()
            size_MB = numel * param.element_size() / 1e6

            total_params += numel
            total_size += size_MB

            if param.requires_grad:
                trainable_params += numel
                trainable_size += size_MB

        fraction = trainable_params / total_params if total_params > 0 else 0
        loc_str = f"[{location}] " if location else ""

        print(
            f"{loc_str}Trainable params: {trainable_params:,} / {total_params:,} "
            f"({fraction:.2%}), Size: {trainable_size:.2f} MB / {total_size:.2f} MB"
        )

    def get_trainable_param_state_dict(self):
        return {
            name: param.detach().cpu()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def print_param_dict_stats(self, param_dict, location=""):
        total_params = 0
        total_size = 0.0

        for tensor in param_dict.values():
            numel = tensor.numel()
            size_MB = numel * tensor.element_size() / 1e6

            total_params += numel
            total_size += size_MB

        loc_str = f"[{location}] " if location else ""
        print(
            f"{loc_str}Param dict stats — Total params: {total_params:,}, Size: {total_size:.2f} MB"
        )

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

        This method is overriden from one in synchronous top aggregator
        (..top_aggregator).
        """
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.info("No channel found")
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

            # Add trainer to global_trainer set Used only to check unavailable
            # trainers later
            if end not in self.all_trainers:
                self.all_trainers.add(end)
                logger.debug(f"Added end {end} to all_trainers set")

            # Add trainer to heartbeat dict if it isnt there Add only most
            # recent heartbeat timestamp as value Discard stale heartbeats if
            # received.
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
                logger.info(
                    f"the heartbeat for {end} with timestamp "
                    f"{heartbeat_timestamp} was stale"
                )
        else:
            logger.warning(f"Got invalid {msg} while processing heartbeat")

    def _aggregate_weights(self, tag: str) -> None:
        """Aggregate local model weights asynchronously.

        This method is overriden from one in synchronous top aggregator
        (..top_aggregator).
        """
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.info("No channel found")
            return

        logger.debug(f"Channel {channel} found for tag {tag}")
        # receive local model parameters from a trainer who arrives first NOTE:
        # (DG) Right now, the leave notifications also cause a message to be
        # processed and yield (None,None) from recv_fifo().
        msg, metadata = next(channel.recv_fifo(channel.ends(VAL_CH_STATE_RECV), 1))
        end, _ = metadata
        if not msg:
            logger.debug(f"No data from {end}; skipping it")
            return

        # NOTE: Only 2 types of messages are expected here: (i) model updates
        # after task_to_perform=TRAIN with weights or (ii) statistical utility
        # updates after task_to_perform=EVAL with info on stat_utility. Else,
        # throw an error.

        # Case #1: Message after task_to_perform=TRAIN. This will contain
        # stat_utility too but will processed later.
        if MessageType.WEIGHTS in msg:
            logger.info(
                f"received model updates from {end} "
                f"with model version {msg[MessageType.MODEL_VERSION]}"
            )

            # For OORT selector NOTE: (DG) Last selected round should have
            # ideally been set in distribute weights. But it was here in the old
            # oort code and ive kept it. Instead of PROP_LAST_SELECTED_ROUND, it
            # should have been PROP_LAST_UPDATE_RECVD_ROUND.
            channel.set_end_property(
                end, PROP_LAST_SELECTED_ROUND, msg[MessageType.MODEL_VERSION]
            )

            # Set last eval round for the trainer since training also means that
            # eval was done for the same round.
            channel.set_end_property(
                end, PROP_LAST_EVAL_ROUND, msg[MessageType.MODEL_VERSION]
            )
            # calculate round duration for this end, if the round number
            # information is identical with round_start_time
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

            # TODO: (DG) Also set the end property for task=eval done at
            # timestamp=current.

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

            # TODO: (DG) Also set the end property for task=eval done at
            # timestamp=current.

            # add trainer to list of ends that have replied with eval updates
            # capture telemetry on trainer participation in rounds
            channel._selector.trainer_eval_recv_ends.append(end)
            logger.debug(
                f"After appending {end} to trainer_eval_recv_ends: "
                f"{channel._selector.trainer_eval_recv_ends}"
            )

            # Remove end from selected_ends and set its state to none so that it
            # can be selected for training in this round.
            logger.info(
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

        # update _track_trainer_version_duration_s to capture training time
        if end not in self._track_trainer_version_duration_s.keys():
            logger.error(
                f"{end} not found in _track_trainer_version_duration_s "
                f"during aggregation"
            )
        else:
            recv_wts_ts = datetime.now()
            recv_wts_version = msg[MessageType.MODEL_VERSION]

            # check0- verify that this recvd version was sent to trainer
            if (
                recv_wts_version
                in self._track_trainer_version_duration_s[end][
                    "sent_wts_version_ts"
                ].keys()
            ):
                sent_wts_ts = self._track_trainer_version_duration_s[end][
                    "sent_wts_version_ts"
                ][recv_wts_version]
                # check1- sent_wts should have happened before current time.
                # Else, handle error
                if recv_wts_ts <= sent_wts_ts:
                    logger.error(
                        f"Trainer: {end}. Recv wts {recv_wts_ts} happened "
                        f"before send wts: {sent_wts_ts} "
                        f"for version {recv_wts_version}"
                    )

                # check2- recv_wts should not have happend for this version
                # before. Else, handle error
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
                # NOTE: (DG) Timeout means that an update returns with latency
                # of [timeout, infinty). While some updates might be less stale,
                # most could be very stale. Instead of cherry-picking which
                # updates to keep and which to discard, we will discard all such
                # delayed updates.
                time_staleness_s = (
                    recv_wts_ts - sent_wts_ts
                ).total_seconds() - SEND_TIMEOUT_WAIT_S
                logger.info(
                    f"Update from end {end} arrived more "
                    f"than {SEND_TIMEOUT_WAIT_S} seconds after last send. "
                    f"Update is stale by time {time_staleness_s} over the "
                    f"timeout and will be discarded."
                )

                # TODO: (DG) NEEDS TESTING. Sanity check is that it should not
                # come here with ClientNotify enabled. But when it did come with
                # ClientNotify and Train->Eval calling reset_end_state_to_none,
                # it caused issues.

                # Currently, the end is now in recvd state and will be removed
                # from selected_ends in handle_recv_state in the next iteration.
                # To add the getter through recv_fifo again, we will (i) remove
                # the end from selected_ends, and (ii) set the end state to
                # none.
                logger.info(
                    f"Attempting to remove end {end} from selected_ends and "
                    f"re-setting its channel state"
                )
                channel._selector.remove_from_selected_ends(channel._ends, end)
                channel._selector.reset_end_state_to_none(channel._ends, end)
                channel._selector._cleanup_removed_ends(end)
                return
            # NOTE: (DG) Previously had a version equality check here for
            # version sent and version received. It was supposed to be equal for
            # syncfl and help discard incorrect round messages. For asyncfl too
            # it should be equal. However it is possible that after leave/join
            # of a trainer between two rounds, a new round version is sent to
            # the trainer, while it sends back the previous version sent to it.
            # This is also a valid update since it is just the previous one (and
            # there are checks on the trainer side to avoid redundant updates).
            else:
                # NOTE: total_training_time_s is approximate. It only captures
                # training time for those send_wt and recv_wt that complete.
                # Timeouts are not included in this time and can be observed
                # separately.
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

                # Following the relaxation in asyncFL to not check for model
                # version equality at the aggregator, we do the same for
                # asyncoort too. We will set the end property without doing the
                # equality check. Round duration can be calculated based on send
                # and recv time for that version to that trainer.
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

        self._per_round_update_list.append(end)

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

        if MessageType.STAT_UTILITY in msg:
            channel.set_end_property(
                end, PROP_STAT_UTILITY, msg[MessageType.STAT_UTILITY]
            )
            stat_utility = msg[MessageType.STAT_UTILITY]

        logger.info(
            f"Received weights from {end}. It was trained on model version {version}, with {count} samples. Returned stat utility {stat_utility}"
        )

        if weights is not None and count > 0:
            tres = TrainResult(weights, count, version, stat_utility)
            # save training result from trainer in a disk cache
            self.cache[end] = tres
            logger.debug(f"received {len(self.cache)} trainer updates in cache")
            logger.debug(f"agg_version: {self._round}, trainer version: {tres.version}")
            update_staleness_val = self._round - tres.version
            logger.debug(f"update_staleness_val: {update_staleness_val}")
            self._per_round_staleness_list.append(update_staleness_val)

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

            # DG-FIX: check trainer version, discard if stale if (tres.version
            # == (self._round - 1)) or ((tres.version == self._round)):

            # if tres.version == self._round: logger.debug("proceeding to agg
            #     weights") self._agg_goal_weights = self.optimizer.do(
            #     self._agg_goal_weights, self.cache, total=count,
            #         version=self._round, staleness_factor=staleness_factor, )
            #         # increment agg goal count self._agg_goal_cnt += 1 else:
            #         logger.debug("stale update from worker, discarding")
            #         return

            logger.debug("proceeding to agg weights")
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
            logger.debug("didn't reach agg goal")
            logger.debug(f" current: {self._agg_goal_cnt}; agg goal: {self._agg_goal}")

            # Set trainer participation count property here to be used later in
            # selection.
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
            logger.debug("reached agg goal")
            logger.debug(f" current: {self._agg_goal_cnt}; agg goal: {self._agg_goal}")
            logger.info(
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

            # update staleness list for aggregator
            self._aggregator_staleness_track_rounds.append(
                self._per_round_staleness_list
            )

            self._aggregator_round_avg_staleness.append(
                np.mean(np.array(self._per_round_staleness_list))
            )

            self._per_round_update_list = []
            self._per_round_staleness_list = []

        # Computing rate: Not used anywhere right now rate = 1 / math.sqrt(1 +
        # self._round - tres.version) logger.debug(f" rate at top_agg: {rate}")

        self.weights = self.optimizer.scale_add_agg_weights(
            self.weights, self._agg_goal_weights, self._agg_goal
        )

        # update model with global weights
        self._update_model()

        # decrement counter since updates consumed from queue
        self._updates_in_queue -= self._agg_goal

        logger.debug(f"aggregation finished for round {self._round}")
        logger.info(
            f"====== aggregation finished for round {self._round}, "
            f"self._agg_goal_cnt: {self._agg_goal_cnt}, self._updates_recevied: "
            f"{self._updates_recevied}, self._trainer_participation_in_round_count: "
            f"{self._trainer_participation_in_round_count}"
        )
        logger.info(
            f"After round: {self._round}, remaining _updates_in_queue: "
            f"{self._updates_in_queue}"
        )

        if self._round % 100 == 0:
            logger.debug(
                f"top agg staleness list after round {self._round} is "
                f"{self._aggregator_round_avg_staleness}"
            )
            logger.debug(
                f"top agg trainer participation in rounds, after round "
                f"{self._round} is {self._trainer_participation_in_round}"
            )

        # print out data on staleness for aggregator, per round unroll the list
        # of lists into a numpy array, get the avg
        agg_staleness_arr = np.hstack(self._aggregator_staleness_track_rounds)
        logger.info(f"==== aggregator avg staleness: {np.mean(agg_staleness_arr)}")

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

    def aggregate_grads_from_trainers(self, trainer_grad):
        self.print_trainable_params_stats(
            location="[start,aggregate_grads_from_trainers()]"
        )
        all_zero = all(torch.allclose(g, torch.zeros_like(g)) for g in self.grad)
        logger.info(f"Are all grads zero initially? {all_zero}")

        self.log_memory("start aggregate_grads_from_trainers", self.device)

        # logger.info(f"len(self.model.named_parameters()):
        # {len(self.model.named_parameters())}, len(self.params):
        # {len(self.params)}") self.grad.to(DeviceType.CPU)
        # trainer_grad.to(DeviceType.CPU)
        np = self.model.named_parameters()

        for i, (name, param) in enumerate(np):  # Assuming self.params is a dict
            if param.requires_grad:
                if name in trainer_grad:
                    grad_device = self.grad[i].device
                    trainer_grad[name] = trainer_grad[name].to(grad_device)
                    # Ensure the layer name exists in trainer_grad

                    self.grad[i].add_(trainer_grad[name])
                else:
                    logger.warning(f"Gradient for {name} not found in trainer_grad.")

        self.log_memory("end aggregate_grads_from_trainers", self.device)
        self.print_trainable_params_stats(
            location="[end,aggregate_grads_from_trainers()]"
        )

    def aggregate_grad_pool(self, grad_list):
        self.print_trainable_params_stats(location="[start,aggregate_grad_pool()]")
        if len(grad_list) == 0:
            self.print_trainable_params_stats(location="[end,aggregate_grad_pool()]")
            return None
        if len(grad_list) == 1:
            self.print_trainable_params_stats(location="[end,aggregate_grad_pool()]")
            return grad_list[0]
        else:
            grad = grad_list[0]
            for id, k in enumerate(grad):
                for i in range(0, len(grad_list)):
                    if i == 0:
                        grad[id] = grad_list[i][id]
                    else:
                        grad[id] += grad_list[i][id]
            self.print_trainable_params_stats(location="[end,aggregate_grad_pool()]")
            return grad

    def _aggregate_grads_async(self, tag: str) -> None:
        """Aggregate trainer gradients asynchronously."""
        logger.info("starting aggregate_grads_async")
        if self.ends_not_selected_yet:
            logger.info("no ends selected yet")
            return

        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.info("No channel found")
            return

        logger.debug(f"Channel {channel} found for tag {tag}")
        # receive local model parameters from a trainer who arrives first NOTE:
        # (DG) Right now, the leave notifications also cause a message to be
        # processed and yield (None,None) from recv_fifo().
        if channel.ends(VAL_CH_STATE_RECV) is None:
            logger.info("no ends yet")
            return
        msg, metadata = next(channel.recv_fifo(channel.ends(VAL_CH_STATE_RECV), 1))
        end, _ = metadata
        if not msg:
            logger.debug(f"No data from {end}; skipping it")
            return

        if MessageType.GRADIENTS in msg and MessageType.GRADIENTS_FOR_VAR_CHECK in msg:
            logger.info(
                f"received gradients from {end} "
                f"with model version {msg[MessageType.MODEL_VERSION]}"
            )

            # For OORT selector NOTE: (DG) Last selected round should have
            # ideally been set in distribute weights. But it was here in the old
            # oort code and ive kept it. Instead of PROP_LAST_SELECTED_ROUND, it
            # should have been PROP_LAST_UPDATE_RECVD_ROUND.
            channel.set_end_property(
                end, PROP_LAST_SELECTED_ROUND, msg[MessageType.MODEL_VERSION]
            )

            # Set last eval round for the trainer since training also means that
            # eval was done for the same round.
            channel.set_end_property(
                end, PROP_LAST_EVAL_ROUND, msg[MessageType.MODEL_VERSION]
            )
            # calculate round duration for this end, if the round number
            # information is identical with round_start_time
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

            # TODO: (DG) Also set the end property for task=eval done at
            # timestamp=current.

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

        # update _track_trainer_version_duration_s to capture training time
        if end not in self._track_trainer_version_duration_s.keys():
            logger.error(
                f"{end} not found in _track_trainer_version_duration_s "
                f"during aggregation"
            )
        else:
            recv_wts_ts = datetime.now()
            recv_wts_version = msg[MessageType.MODEL_VERSION]

            # check0- verify that this recvd version was sent to trainer
            if (
                recv_wts_version
                in self._track_trainer_version_duration_s[end][
                    "sent_wts_version_ts"
                ].keys()
            ):
                sent_wts_ts = self._track_trainer_version_duration_s[end][
                    "sent_wts_version_ts"
                ][recv_wts_version]
                # check1- sent_wts should have happened before current time.
                # Else, handle error
                if recv_wts_ts <= sent_wts_ts:
                    logger.error(
                        f"Trainer: {end}. Recv wts {recv_wts_ts} happened "
                        f"before send wts: {sent_wts_ts} "
                        f"for version {recv_wts_version}"
                    )

                # check2- recv_wts should not have happend for this version
                # before. Else, handle error
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
                # NOTE: (DG) Timeout means that an update returns with latency
                # of [timeout, infinty). While some updates might be less stale,
                # most could be very stale. Instead of cherry-picking which
                # updates to keep and which to discard, we will discard all such
                # delayed updates.
                time_staleness_s = (
                    recv_wts_ts - sent_wts_ts
                ).total_seconds() - SEND_TIMEOUT_WAIT_S
                logger.info(
                    f"Update from end {end} arrived more "
                    f"than {SEND_TIMEOUT_WAIT_S} seconds after last send. "
                    f"Update is stale by time {time_staleness_s} over the "
                    f"timeout and will be discarded."
                )

                # TODO: (DG) NEEDS TESTING. Sanity check is that it should not
                # come here with ClientNotify enabled. But when it did come with
                # ClientNotify and Train->Eval calling reset_end_state_to_none,
                # it caused issues.

                # Currently, the end is now in recvd state and will be removed
                # from selected_ends in handle_recv_state in the next iteration.
                # To add the getter through recv_fifo again, we will (i) remove
                # the end from selected_ends, and (ii) set the end state to
                # none.
                logger.info(
                    f"Attempting to remove end {end} from selected_ends and "
                    f"re-setting its channel state"
                )
                channel._selector.remove_from_selected_ends(channel._ends, end)
                channel._selector.reset_end_state_to_none(channel._ends, end)
                channel._selector._cleanup_removed_ends(end)
                return
            # NOTE: (DG) Previously had a version equality check here for
            # version sent and version received. It was supposed to be equal for
            # syncfl and help discard incorrect round messages. For asyncfl too
            # it should be equal. However it is possible that after leave/join
            # of a trainer between two rounds, a new round version is sent to
            # the trainer, while it sends back the previous version sent to it.
            # This is also a valid update since it is just the previous one (and
            # there are checks on the trainer side to avoid redundant updates).
            else:
                # NOTE: total_training_time_s is approximate. It only captures
                # training time for those send_wt and recv_wt that complete.
                # Timeouts are not included in this time and can be observed
                # separately.
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

                # Following the relaxation in asyncFL to not check for model
                # version equality at the aggregator, we do the same for
                # asyncoort too. We will set the end property without doing the
                # equality check. Round duration can be calculated based on send
                # and recv time for that version to that trainer.
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

        self._updates_in_queue += 1

        self._per_round_update_list.append(end)

        if end not in self._updates_recevied.keys():
            self._updates_recevied[end] = 1
        else:
            self._updates_recevied[end] += 1

        # Process the gradients
        if MessageType.GRADIENTS in msg:
            # weights = weights_to_model_device(msg[MessageType.WEIGHTS],
            # self.model)
            trainer_gradients = msg[MessageType.GRADIENTS]
            self.aggregate_grads_from_trainers(trainer_gradients)

        if MessageType.GRADIENTS_FOR_VAR_CHECK in msg:
            logger.info(
                f"received GRADIENTS_FOR_VAR_CHECK, {len(msg[MessageType.GRADIENTS_FOR_VAR_CHECK])}"
            )
            self.grad_for_var_check_list.append(
                msg[MessageType.GRADIENTS_FOR_VAR_CHECK]
            )

        if MessageType.DATASET_SIZE in msg:
            count = msg[MessageType.DATASET_SIZE]
            channel.set_end_property(
                end, PROP_DATASET_SIZE, msg[MessageType.DATASET_SIZE]
            )

        if MessageType.MODEL_VERSION in msg:
            version = msg[MessageType.MODEL_VERSION]

        if MessageType.STAT_UTILITY in msg:
            channel.set_end_property(
                end, PROP_STAT_UTILITY, msg[MessageType.STAT_UTILITY]
            )
            stat_utility = msg[MessageType.STAT_UTILITY]

        logger.info(
            f"Received grads from {end}. It was trained on model version {version}, with {count} samples"
        )

        logger.info("proceeding to agg weights")

        self._agg_goal_cnt += 1
        logger.info(
            f"self._agg_goal_cnt = {self._agg_goal_cnt}, self._agg_goal = {self._agg_goal}"
        )

        if self._agg_goal_cnt < self._agg_goal:
            # didn't reach the aggregation goal; return
            logger.info("didn't reach agg goal")
            logger.debug(f" current: {self._agg_goal_cnt}; agg goal: {self._agg_goal}")

            # Set trainer participation count property here to be used later in
            # selection.
            channel.set_end_property(
                end, PROP_UPDATE_COUNT, self._updates_recevied[end]
            )
            channel.cleanup_recvd_ends()
            return

        # if self._agg_goal_weights is None: logger.debug("failed model
        #     aggregation") time.sleep(1) return

        # set global weights, by adding scaled aggregated weights with
        # aggregation goal

        if self._agg_goal_cnt == self._agg_goal:
            logger.info("reached agg goal")
            logger.debug(f" current: {self._agg_goal_cnt}; agg goal: {self._agg_goal}")
            logger.info(
                f"Reached agg_goal {self._agg_goal}, "
                f"current _updates_in_queue: {self._updates_in_queue}, "
                f"current round before agg: {self._round}"
            )
            # NRL commenting for now update per-trainer participation in round
            # agg for trainer_update in self._per_round_update_list: if (
            # trainer_update not in
            #     self._trainer_participation_in_round_count.keys() ):
            #         self._trainer_participation_in_round_count[trainer_update]
            #         = 1 self._trainer_participation_in_round[trainer_update] =
            #     [ 0 ] * 20000  # assuming max 20K rounds
            #         self._trainer_participation_in_round[trainer_update][
            #         self._round - 1 ] = 1 else:
            #             self._trainer_participation_in_round_count[trainer_update]
            #         += 1 self._trainer_participation_in_round[trainer_update][
            #         self._round - 1 ] = 1

            # # update staleness list for aggregator
            # self._aggregator_staleness_track_rounds.append(
            #     self._per_round_staleness_list )

            # self._aggregator_round_avg_staleness.append(
            #     np.mean(np.array(self._per_round_staleness_list)) )

            # self._per_round_update_list = [] self._per_round_staleness_list =
            # []

            # Computing rate: Not used anywhere right now rate = 1 / math.sqrt(1
            # + self._round - tres.version) logger.debug(f" rate at top_agg:
            # {rate}")

            # NRL not aggregating for now

            # self.weights = self.optimizer.scale_add_agg_weights( self.weights,
            #     self._agg_goal_weights, self._agg_goal )

            # # update model with global weights
            # self._update_model()

            logger.info("calling aggregate for fwdllm (Async)")
            self.grad_pool.append(self.grad)
            self.add_local_trained_result(0, self.grad, self._agg_goal_cnt)
            self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(
                self.model
            )
            self.grad = [torch.zeros_like(p) for p in self.params]

            self.aggregate(self._round)
            self._agg_goal_cnt = 0
            # decrement counter since updates consumed from queue
            self._updates_in_queue -= self._agg_goal

            round_to_print = self._round
            data_id_to_print = self.data_id

            if self.var_good_enough:

                # evaluate model to calculate loss
                result, _, _ = self.eval_model()
                logger.info(f"eval loss = {result['eval_loss']}")
                self.data_id += 1
                self.iteration_per_data_id = 0
                # TODO: need to replace it with per end property
                if self.data_id == self.total_data_bins:
                    logger.info("incrementing round number now ")
                    self._round += 1
                    self.data_id = 0
                    channel.set_property("round", self._round)

            else:
                self.iteration_per_data_id += 1

        logger.debug(f"aggregation finished for round {round_to_print}")
        logger.info(
            f"====== aggregation finished for round {round_to_print}, data id: {data_id_to_print}, "
            f"self._agg_goal_cnt: {self._agg_goal_cnt}, self._updates_recevied: "
            f"{self._updates_recevied}, self._trainer_participation_in_round_count: "
            f"{self._trainer_participation_in_round_count}"
        )

        logger.info(
            f"After round: {round_to_print}, remaining _updates_in_queue: "
            f"{self._updates_in_queue}"
        )
        # NRL commenting for now if self._round % 100 == 0: logger.debug( f"top
        # agg staleness list after round {self._round} is "
        #     f"{self._aggregator_round_avg_staleness}" ) logger.debug( f"top
        #         agg trainer participation in rounds, after round "
        #         f"{self._round} is {self._trainer_participation_in_round}" )

        # # print out data on staleness for aggregator, per round unroll
        # # the list of lists into a numpy array, get the avg
        # agg_staleness_arr = np.hstack(self._aggregator_staleness_track_rounds)
        # logger.info(f"==== aggregator avg staleness:
        # {np.mean(agg_staleness_arr)}")

        # # per trainer analytics
        # if self._round % 100 == 0: for k, v in
        #     self._per_trainer_staleness_track.items(): trainer_staleness_arr =
        #         np.array(v) logger.info( f"Trainer {k} staleness info. Min
        #         {np.min(trainer_staleness_arr)}, " f"Max
        #             {np.max(trainer_staleness_arr)}, " f"Avg
        #             {np.mean(trainer_staleness_arr)}, " f"P50
        #             {np.median(trainer_staleness_arr)}, " f"P90
        #             {np.percentile(trainer_staleness_arr, 90)}, " f"P99
        #             {np.percentile(trainer_staleness_arr, 99)}" )

        # total_training_time_all_trainers = 0 for k, v in
        # self._track_trainer_version_duration_s.items():
        #     total_training_time_all_trainers += v["total_training_time_s"]
        # avg_training_time = total_training_time_all_trainers / len(
        #     self._track_trainer_version_duration_s ) logger.info( f"Avg
        # training time {avg_training_time} across "
        # f"{len(self._track_trainer_version_duration_s)} trainers" )
        if self.var_good_enough:
            channel._selector.remove_from_selected_ends(channel._ends, end)
        logger.debug("Agg goal reached, so resetting trainer end states in the channel")
        channel.cleanup_recvd_ends()

    def _aggregate_grads_sync(self, tag: str) -> None:
        """Aggregate trainer gradients synchronously."""
        logger.info("starting aggregate_grads_sync")
        self.log_memory("start _aggregate_grads_sync", self.device)
        self.print_trainable_params_stats(location="[start,_aggregate_grads_sync()]")
        if self.ends_not_selected_yet:
            logger.info("no ends selected yet")
            return

        channel = self.cm.get_by_tag(tag)
        if not channel:
            return

        logger.debug(f"Channel {channel} found for tag {tag}")
        # receive local model parameters from a trainer who arrives first NOTE:
        # (DG) Right now, the leave notifications also cause a message to be
        # processed and yield (None,None) from recv_fifo().
        if channel.ends(VAL_CH_STATE_RECV) is None:
            logger.info("no ends yet")
            return

        total = 0

        # receive local model parameters from trainers
        for msg, metadata in channel.recv_fifo(channel.ends()):
            end, timestamp = metadata
            if not msg:
                logger.info(f"No data from {end}; skipping it")
                continue

            if (
                MessageType.GRADIENTS in msg
                and MessageType.GRADIENTS_FOR_VAR_CHECK in msg
            ):
                logger.info(
                    f"received gradients from {end} "
                    f"with model version {msg[MessageType.MODEL_VERSION]}"
                )
                self._agg_goal_cnt += 1

                # For OORT selector NOTE: (DG) Last selected round should have
                # ideally been set in distribute weights. But it was here in the
                # old oort code and ive kept it. Instead of
                # PROP_LAST_SELECTED_ROUND, it should have been
                # PROP_LAST_UPDATE_RECVD_ROUND.
                channel.set_end_property(
                    end, PROP_LAST_SELECTED_ROUND, msg[MessageType.MODEL_VERSION]
                )

                # Set last eval round for the trainer since training also means
                # that eval was done for the same round.
                channel.set_end_property(
                    end, PROP_LAST_EVAL_ROUND, msg[MessageType.MODEL_VERSION]
                )
                # calculate round duration for this end, if the round number
                # information is identical with round_start_time
                logger.debug(
                    f"Getting channel property {PROP_ROUND_START_TIME} for "
                    f"end {end}"
                )
                round_start_time_tup = channel.get_end_property(
                    end, PROP_ROUND_START_TIME
                )
                end = metadata[0]
                timestamp = metadata[1]
                logger.debug(
                    f"Returned round_start_time_tup: {round_start_time_tup} for "
                    f"end {end} and timestamp {timestamp}"
                )

                # TODO: (DG) Also set the end property for task=eval done at
                # timestamp=current.

            else:
                logger.error(
                    f"Invalid message received from {end} in aggregate_weights: {msg}"
                )
                return

            logger.debug(f"received data from {end}")
            channel.set_end_property(end, PROP_ROUND_END_TIME, (round, timestamp))

            # logger.debug(f"received message in agg_grads_sync {msg} from {end}")
            # capture telemetry on trainer participation in rounds
            channel._selector.ordered_updates_recv_ends.append(end)
            self._updates_in_queue += 1
            self._per_round_update_list.append(end)

            if end not in self._updates_recevied.keys():
                self._updates_recevied[end] = 1
            else:
                self._updates_recevied[end] += 1

            # Process the gradients
            if MessageType.GRADIENTS in msg:
                # weights = weights_to_model_device(msg[MessageType.WEIGHTS],
                # self.model)
                trainer_gradients = msg[MessageType.GRADIENTS]
                self.aggregate_grads_from_trainers(trainer_gradients)

            if MessageType.GRADIENTS_FOR_VAR_CHECK in msg:
                logger.info(
                    f"received GRADIENTS_FOR_VAR_CHECK, {len(msg[MessageType.GRADIENTS_FOR_VAR_CHECK])}"
                )
                self.grad_for_var_check_list.append(
                    msg[MessageType.GRADIENTS_FOR_VAR_CHECK]
                )

            if MessageType.DATASET_SIZE in msg:
                count = msg[MessageType.DATASET_SIZE]
                channel.set_end_property(
                    end, PROP_DATASET_SIZE, msg[MessageType.DATASET_SIZE]
                )

            if MessageType.MODEL_VERSION in msg:
                version = msg[MessageType.MODEL_VERSION]

            if MessageType.STAT_UTILITY in msg:
                channel.set_end_property(
                    end, PROP_STAT_UTILITY, msg[MessageType.STAT_UTILITY]
                )
                stat_utility = msg[MessageType.STAT_UTILITY]

            logger.info(
                f"Received grads from {end}. It was trained on model version {version}, with {count} samples"
            )

            if self._agg_goal_cnt == self._agg_goal:
                logger.info(
                    f"Reached agg_goal of {self._agg_goal} since agg_goal_count is {self._agg_goal_cnt}. Breaking from for loop, proceeding to aggregate."
                )
                break

        logger.debug(f"received {len(self.cache)} trainer updates in cache")

        # Proceed to aggregating gradients
        # logger.info("calling aggregate for fwdllm (Sync)")
        self.grad_pool.append(self.grad)
        self.print_trainable_params_stats(
            location="[agg_start,_aggregate_grads_sync()]"
        )
        self.add_local_trained_result(0, self.grad, self._agg_goal_cnt)
        self.print_trainable_params_stats(
            location="[after_add_local,_aggregate_grads_sync()]"
        )
        self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(
            self.model
        )
        self.grad = [torch.zeros_like(p) for p in self.params]

        self.aggregate(self._round)
        self.print_trainable_params_stats(
            location="[after_aggregate(),_aggregate_grads_sync()]"
        )
        self._agg_goal_cnt = 0
        # decrement counter since updates consumed from queue
        self._updates_in_queue -= self._agg_goal

        round_to_print = self._round
        data_id_to_print = self.data_id

        if self.var_good_enough:
            # evaluate model to calculate loss
            result, _, _ = self.eval_model()
            logger.info(f"eval loss = {result['eval_loss']}")
            self.data_id += 1
            self.iteration_per_data_id = 0
            # TODO: need to replace it with per end property
            if self.data_id == self.total_data_bins:
                logger.info("incrementing round number now ")
                self._round += 1
                self.data_id = 0
                channel.set_property("round", self._round)
        else:
            self.iteration_per_data_id += 1

        logger.debug(f"aggregation finished for round {round_to_print}")
        logger.info(
            f"====== aggregation finished for round {round_to_print}, data id: {data_id_to_print}, "
            f"self._agg_goal_cnt: {self._agg_goal_cnt}, self._updates_recevied: "
            f"{self._updates_recevied}, self._trainer_participation_in_round_count: "
            f"{self._trainer_participation_in_round_count}"
        )

        logger.info(
            f"After round: {round_to_print}, remaining _updates_in_queue: "
            f"{self._updates_in_queue}"
        )

        self.log_memory("end _aggregate_grads_sync", self.device)

    def eval_model(self, epoch=0, global_step=0, device=None):
        if not device:
            device = self.device

        logger.info(f"device inside eval_model() is set to: {device}")
        self.log_memory("start eval_model", self.device)

        results = {}

        eval_loss = 0.0
        nb_eval_steps = 0
        n_batches = len(self.test_global)
        test_sample_len = len(self.test_global.dataset)
        preds = np.empty((test_sample_len, self.num_labels))

        logger.info(
            f"Created n_batches: {n_batches}, test_sample_len: {test_sample_len} and preds.shape: {preds.shape}, location of model: {next(self.model.parameters()).device}"
        )

        out_label_ids = np.empty(test_sample_len)
        # Move model to device before performing the eval
        self.model.to(device)
        self.model.eval()
        self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(
            self.model
        )

        for i, batch in enumerate(self.test_global):
            with torch.no_grad():
                batch = tuple(t for t in batch)
                x = batch[1].to(device)
                labels = batch[4].to(device)

                output = self.model(x)
                logits = output[0]

                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                eval_loss += loss.item()

            nb_eval_steps += 1
            start_index = self.args.eval_batch_size * i

            end_index = (
                start_index + self.args.eval_batch_size
                if i != (n_batches - 1)
                else test_sample_len
            )
            preds[start_index:end_index] = logits.detach().cpu().numpy()
            out_label_ids[start_index:end_index] = labels.detach().cpu().numpy()

        eval_loss = eval_loss / nb_eval_steps

        model_outputs = preds
        preds = np.argmax(preds, axis=1)
        result, wrong = self.compute_metrics(
            preds, out_label_ids, self.test_global.examples
        )
        result["eval_loss"] = eval_loss
        results.update(result)

        # self.results.update(result)
        logging.info(f"results after eval are: {results}, len(wrong) is: {len(wrong)}")

        # TODO: Check if model needs to be moved back to cpu? Do we need to keep
        # moving the model between CPU and GPU repeatedly?
        del x, labels, output, logits, loss
        torch.cuda.empty_cache()
        gc.collect()

        self.log_memory("end eval_model", self.device)

        return result, model_outputs, wrong

    def compute_metrics(self, preds, labels, eval_examples=None):
        assert len(preds) == len(labels)
        self.log_memory("start compute_metrics", self.device)

        extra_metrics = {}
        extra_metrics["acc"] = sklearn.metrics.accuracy_score(labels, preds)
        mismatched = labels != preds

        if eval_examples:
            wrong = [i for (i, v) in zip(eval_examples, mismatched) if v.any()]
        else:
            wrong = ["NA"]

        mcc = matthews_corrcoef(labels, preds)

        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

        self.log_memory("end compute_metrics", self.device)

        return (
            {**{"mcc": mcc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}, **extra_metrics},
            wrong,
        )

    def oracular_trainer_avail_check(self, end: str) -> bool:
        logger.debug("In oracular_trainer_avail_check")

        picked_trainer_is_available = True

        if end in self.trainer_unavail_durations.keys():
            # get aggregator seconds from start
            agg_time_since_start_s = time.time() - self.agg_start_time_ts

            curr_trainer_unavail_list = self.trainer_unavail_durations[end]

            # iterate through unavailability list First, check if the current
            # time is within any failure window

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

            # Remove end from trainer_unavail_durations if list is empty TODO:
            # Check if deletion is happening properly
            if len(updated_trainer_unavail_list) == 0:
                logger.debug(
                    f"### Trainer {end} will no longer fail, removing from "
                    f"trainer_unavail_durations"
                )
                del self.trainer_unavail_durations[end]
            else:
                self.trainer_unavail_durations[end] = updated_trainer_unavail_list
        else:
            logger.info(
                f"No info on end {end} in self.trainer_unavail_durations"
                f", returning TRUE (default)"
            )
        return picked_trainer_is_available

    def hearbeat_trainer_avail_check(self, end: str) -> bool:
        picked_trainer_is_available = True
        last_acceptable_heartbeat_ts = time.time() - (
            self._trainer_max_miss_heartbeats * self._trainer_heartbeat_freq_s
        )

        # return True if: heartbeat was received from trainer and it is within
        # last_acceptable_heartbeat_ts

        # return False if: if end isnt in heartbeat dict, means that the trainer
        # hasn't given a heartbeat in a while and was removed based on
        # last_acceptable_heartbeat_ts

        # NOTE: During agg init, it might have registered a trainer, but not
        # received heartbeat in such a scenario, we return True so that agg is
        # able to send init_weights to trainer and start the training process
        # this is when trainer not in all_trainers and not in dict

        if (end not in self._per_trainer_last_heartbeat_ts.keys()) and (
            end not in self.all_trainers
        ):
            picked_trainer_is_available = True
            logger.info(
                f"Might be trainer init(), trainer {end} hasnt sent any"
                f" heartbeats yet, but we return True"
            )
        elif end not in self._per_trainer_last_heartbeat_ts.keys():
            picked_trainer_is_available = False
            logger.debug(f"Trainer {end} was already marked unavailable")
        elif self._per_trainer_last_heartbeat_ts[end] < last_acceptable_heartbeat_ts:
            del self._per_trainer_last_heartbeat_ts[end]
            picked_trainer_is_available = False
            logger.info(
                f"Trainer {end} missed max_allowed_heartbeats, " f"marked unavailable"
            )
        elif self._per_trainer_last_heartbeat_ts[end] >= last_acceptable_heartbeat_ts:
            picked_trainer_is_available = True
            logger.debug(f"Trainer {end} is available")
        else:
            logger.error(f"Availability check failed, trainer {end}, returning True")

        return picked_trainer_is_available

    def get_unavailable_trainers(self) -> list:
        # Works only for heartbeat based right now TODO: (DG) Extend for other
        # trainer_avail_checks too
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
        """Distribute a global model in synchronous FL fashion - for FwdLLM.
        This method actually sends either gradients or calc_more_var to
        trainers, not the actual model weights.

        This method is overridden from one in synchronous top aggregator
        (..top_aggregator).
        """

        logger.info(f"Device for agg: {next(self.model.parameters()).device}")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found for tag {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()
        global_model_params = self.get_global_model_params()
        self.weights = global_model_params  # TODO: check this, not sure where self.weights is initialised
        # before distributing weights, update it from global model
        # self._update_weights()

        # busy wait for 0.1 seconds before proceeding. This is to wait on
        # distribute_weights to let the system state get updated before selector
        # is invoked again

        logger.debug(f"Starting busy wait at time {time.time()}")
        time.sleep(0.1)
        logger.debug(f"Ended busy wait at time {time.time()}")

        # before invoking channel.ends() to select, set the trainer_unavail if
        # it isn't None if self.trainer_unavail_durations is not None:
        # curr_unavail_trainer_list = self.get_curr_unavail_trainers()
        #     channel.set_curr_unavailable_trainers(
        #     trainer_unavail_list=curr_unavail_trainer_list )
        #         logger.debug(f"Passed curr_unavail_trainer_list: "
        #     f"{curr_unavail_trainer_list} to channel") else: # Handling the
        #     case for oort's selector since it expects 3 # arguments
        #                  channel.set_curr_unavailable_trainers(trainer_unavail_list=[])

        # check if there are any ends to send weights to

        # logger.info( f"Sending weights to trainers with task_to_perform =
        #     {task_to_perform}" )
        ends = channel.ends(VAL_CH_STATE_SEND, task_to_perform)
        logger.info(f"ends: {ends}")
        if ends is None:
            self.ends_not_selected_yet = True
        else:
            self.ends_not_selected_yet = False
        # NRL TODO: else will take care of randomly selecting x trainers for
        # "eval only" operation
        if not ends:
            logger.debug(
                f"No trainers found for tag {tag}, will "
                f"move to get() for fetch weights from trainers"
            )
            return
        if self.var:
            logger.info(
                f"self.var = {self.var}, self.var_threshold = {self.var_threshold}"
            )
        if self.var_good_enough == True:
            logger.info(
                "Will send new weights to ends since variance is less than threshold"
            )
        else:
            logger.info(
                "Sending variance = bad to trainers since variance is greater than threshold"
            )

        # send out global model parameters to trainers
        self.print_trainable_params_stats(location="[populate_params, _distr_weights]")
        trainable_params = self.get_trainable_param_state_dict()
        self.print_param_dict_stats(trainable_params, location="After filtering")
        shared_weights = weights_to_device(trainable_params, DeviceType.CPU)

        shared_grad_pool = self.aggregate_grad_pool(self.grad_pool)

        shared_grad_pool_trainable = []
        if shared_grad_pool == None:
            shared_grad_pool_trainable = None
        else:
            idx = 0
            for param in self.model.parameters():
                if param.requires_grad:
                    shared_grad_pool_trainable.append(shared_grad_pool[idx].clone())
                idx += 1

        for end in ends:
            # setting start time for OORT TODO: (DG) round_start_time for all
            # trainers in the same round may not be the same            
            logger.debug(
                f"Setting channel property {PROP_ROUND_START_TIME} for "
                f"end {end}. For round {self._round} at time: {datetime.now()}"
            )
            channel.set_end_property(
                end, PROP_ROUND_START_TIME, (self._round, datetime.now())
            )

            # we use _round to indicate a model version
            # logger.info(f"sending data id: {self.data_id}")
            payload = None
            if self.var_good_enough == True:
                logger.info(
                    f"sending weights to {end} with model_version: {self._round}, data_id: {self.data_id} for task: {task_to_perform}"
                )
                
                payload = {
                    MessageType.WEIGHTS: shared_weights,
                    MessageType.GRAD_POOL: shared_grad_pool_trainable,
                    MessageType.ROUND: self._round,
                    MessageType.MODEL_VERSION: self._round,
                    MessageType.TASK_TO_PERFORM: task_to_perform,
                    MessageType.DATA_ID: self.data_id,
                    MessageType.ITERATION_PER_DATA_ID: self.iteration_per_data_id,
                }
                sizes_mb = {
                    key.name if hasattr(key, "name") else str(key): len(
                        pickle.dumps(value)
                    )
                    / (1024 * 1024)
                    for key, value in payload.items()
                }
                total_size_mb = sum(sizes_mb.values())

                logger.info(
                    f"[DEBUG] Payload size breakdown for {end}: "
                    + ", ".join([f"{k}: {v:.2f} MB" for k, v in sizes_mb.items()])
                    + f", Total: {total_size_mb:.2f} MB"
                )
                
                channel.send(end, payload)
                # Added a 1 second sleep so as to not overwhelm mqtt and cuda
                time.sleep(1)

                self.grad_pool = []
                self.grad_for_var_check_list = []
            else:
                logger.info(
                    f"sending var = bad to {end} with model_version: {self._round}, data_id: {self.data_id} for task: {task_to_perform}"
                )
                payload = {
                    MessageType.VAR: "bad",
                    MessageType.ROUND: self._round,
                    MessageType.MODEL_VERSION: self._round,
                    MessageType.TASK_TO_PERFORM: task_to_perform,
                    MessageType.DATA_ID: self.data_id,
                    MessageType.ITERATION_PER_DATA_ID: self.iteration_per_data_id,
                }
                msg_bytes = pickle.dumps(payload)
                logger.info(
                    f"[DEBUG] Payload size for {end}: {len(msg_bytes) / (1024 * 1024):.2f} MB"
                )
                channel.send(end, payload)
                # Added a 0.5 second sleep so as to not overwhelm mqtt
                # time.sleep(0.5)
            del payload
            gc.collect()

            # Update send_time in training_duration_s
            if end not in self._track_trainer_version_duration_s.keys():
                logger.debug(
                    f"{end} not in _track_trainer_version_duration_s, " f"will add"
                )
                self._track_trainer_version_duration_s[end] = dict()
                self._track_trainer_version_duration_s[end]["last_send_wts_ts"] = -1

                # sent_wts_version_ts, recv_wts_version_ts is a dict of version
                # sent/recv and its timestamp. This will be primarily used by
                # AsyncOORT selector since it needs round_duration times. TODO:
                # (DG) Right now the dict maintains ALL sent/recv versions and
                # timestamps for all trainers. For thousands of trainers it
                # might incur memory-bloat. Can optimize to retain just the
                # versions and timestamps of those that were sent but not
                # received back for the trainer.
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
            task_pause_exec = Tasklet("pause_exec", self.pause_execution)

            task_reset_agg_goal_vars = Tasklet(
                "reset_agg_goal_vars", self._reset_agg_goal_variables
            )

            # Created separate put tasklets for train and eval
            task_put_train = Tasklet("distribute", self.put, TAG_DISTRIBUTE, "train")

            task_put_eval = Tasklet("distribute", self.put, TAG_DISTRIBUTE, "eval")

            # TODO: (DG) Update later, task_get_weights gets both weights from
            # train and eval tasks. Will create a cleaner separation later.
            task_get_weights = Tasklet("aggregate", self.get, TAG_AGGREGATE)

            task_get_heartbeat = Tasklet("heartbeat", self.get, TAG_HEARTBEAT)
            task_init = Tasklet("initialize", self.initialize)

            task_aggregate_grads_sync = Tasklet(
                "aggregate", self._aggregate_grads_sync, TAG_AGGREGATE
            )

        c = self.composer
        c.unlink()

        loop = Loop(loop_check_fn=lambda: self._work_done)
        # create a loop object for asyncfl to manage concurrency as well as
        # aggregation goal asyncfl_loop = Loop(loop_check_fn=lambda:
        # self._agg_goal_cnt == self._agg_goal)

        # chain them again with new tasklets introduced in this class
        (
            task_internal_init
            >> task_init
            >> loop(
                # task_reset_agg_goal_vars
                task_put_train
                # >> asyncfl_loop(task_put >> task_get_weights >>
                # >> task_get_heartbeat
                >> task_aggregate_grads_sync
            )
            # >> c.tasklet("load_data") c.tasklet("initialize")
            # >> task_get_heartbeat task_put_train c.tasklet("heartbeat") loop(
            # >> task_reset_agg_goal_vars # >> asyncfl_loop(task_put >>
            # >> task_get_weights >> c.tasklet("heartbeat") ) >>
            # >> asyncfl_loop(task_put_train >> task_put_eval >>
            # >> task_get_weights) >> c.tasklet("train") >>
            #     c.tasklet("evaluate") >> c.tasklet("analysis") >>
            #     c.tasklet("save_metrics") >> c.tasklet("inc_round") )
            # >> c.tasklet("inform_end_of_training") c.tasklet("save_params")
            # c.tasklet("save_model")
        )

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the top level
        aggregator role."""
        return [TAG_DISTRIBUTE, TAG_AGGREGATE, TAG_HEARTBEAT]
