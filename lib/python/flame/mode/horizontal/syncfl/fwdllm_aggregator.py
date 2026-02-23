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
"""Aysnc and SyncFL horizontal FL top level aggregator for FwdLLM."""

# TODO: Shift is_async param to hyperparameters
import gc
import logging
import psutil
import time
from datetime import datetime
import sklearn
import numpy as np
import ast
import os
import json
from sortedcontainers import SortedDict
from flame.channel import VAL_CH_STATE_HTBT_RECV, VAL_CH_STATE_RECV, VAL_CH_STATE_SEND
from flame.common.constants import DeviceType
from flame.common.util import weights_to_device, weights_to_model_device
from flame.config import OptimizerType
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
from flame.mode.horizontal.asyncfl.top_aggregator import TopAggregator as AsyncTopAgg
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
import glob

from torch.nn import CrossEntropyLoss
import flame.monitor.runtime
from flame.monitor.runtime import FwdLLMStage, timer_decorator


logger = logging.getLogger(__name__)

PROP_ROUND_END_TIME = "round_end_time"

SEND_TIMEOUT_WAIT_S = 90  # 90 seconds timeout


@timer_decorator
def recv_fifo_wrapper(channel, ends):
    logger.debug("Entering recv_fifo_wrapper generator loop")
    for msg, metadata in channel.recv_fifo(ends):
        logger.debug(f"Yielding msg from {metadata}")
        yield msg, metadata
    logger.debug("Exiting recv_fifo_wrapper")

class TopAggregator(AsyncTopAgg):
    """Top level Aggregator implements an ML aggregation
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

        # Dictionary to store trainer state: Key = trainer_id, Value = model_version, data_id, iteration_id
        self._trainer_state_dict = {}

        # check if distribute_weights was successful
        self._prev_distribute_weights_success = False

        self.data_id = 0
        self.total_data_bins = 150
        self._is_model_updated = False
        self._model_version = 0
        self.grad_pool = []
        self.var = None
        self.ends_not_selected_yet = False
        self.iteration_per_data_id = 0
        self._optimizer_sort_value = self.config.optimizer.sort
        OPTIMIZERS_SUPPORTING_GRAD_AGGREGATION = (OptimizerType.FEDBUFF,)
        self._weighted_aggregation_enabled = (
            self._optimizer_sort_value in OPTIMIZERS_SUPPORTING_GRAD_AGGREGATION
        )
        if not self._weighted_aggregation_enabled:
            logger.info(
                f"Setting rate=1.0 for all updates because optimizer.sort is "
                f"{self._optimizer_sort_value}; weighted aggregation only supported by {OPTIMIZERS_SUPPORTING_GRAD_AGGREGATION}."
            )
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

        logger.info(f"Experiment set to run in is_async: {self.is_async}")
        # maintain a set of all trainers that have sent heartbeats previously
        self.all_trainers = set()
        try:
            self.minInitialTrainers = self.config.selector.kwargs.get(
                "minInitialTrainers"
            )
            assert self.minInitialTrainers is not None
        except (KeyError, AssertionError):
            raise KeyError(
                "minInitialTrainers must be specified in selector config & must not be None for determinism"
            )
        self.trainer_unavail_durations = None
        self._cached_test_data = None
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

    def read_trainer_unavailability(self, trace=None) -> None:
        logger.info(f"Came to read_trainer_unavailability, trace: {trace}")
        trainer_events_dict = {}

        # TODO(Aishwwarya): Set path to read JSON files without 'aish_test' after Twisha's PR merge
        files_path = "/home/dgarg39/aish_test/flame/lib/python/examples/fwdllm/expts/run_tc_expts/json_scripts"

        dirname = os.path.dirname(__file__)
        search_pattern = os.path.join(dirname, files_path, "trainer_*.json")
        json_files = glob.glob(search_pattern)

        if not json_files:
            logger.warning(f"No JSON files found matching pattern: {search_pattern}")
            return {}

        logger.info(f"Found {len(json_files)} JSON files to process.")

        for file_path in json_files:
            with open(file_path) as f:
                trainer_json = json.load(f)
                curr_trainer_id = trainer_json["taskid"]
                event_list = ast.literal_eval(trainer_json["hyperparameters"][trace])

                # SortedDict for efficient timestamp lookup
                state_dict = SortedDict()

                # Process the events
                for timestamp, event_name in event_list:
                    state_dict[timestamp] = event_name

                trainer_events_dict[curr_trainer_id] = state_dict
                logger.info(f"Completed file read for {file_path}")

        logger.info("Completed reading all trainer unavailability from files")
        return trainer_events_dict

    def aggregate_grads_from_trainers(
        self,
        trainer_grad,
        version_for_rate: int,
        stat_utility: float = 0.0,
        grad_for_var_check=None,
    ):
        """Aggregate a single trainer's gradients into self.grad.

        All incoming tensors are scaled by `rate` before accumulation.
        If `grad_for_var_check` is provided (list of tensors), it is scaled by the
        same `rate` and appended to `self.grad_for_var_check_list` for variance checks.
        """
        # logger.info(f"trainer grad in {trainer_grad}")
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

        # rate = scale * alpha(staleness) + (1 - scale) * beta(stat_utility)
        # alpha: polynomial decay in staleness; beta: polynomial_upshift
        if not self._weighted_aggregation_enabled:
            rate = 1.0
        else:
            staleness_val = self._model_version - version_for_rate
            try:
                scale_val = self.optimizer.agg_rate_conf["scale"]
                a_exp_val = self.optimizer.agg_rate_conf["a_exp"]
                b_exp_val = self.optimizer.agg_rate_conf["b_exp"]
                rate = self.optimizer.weight_factor(
                    scale=scale_val,
                    staleness=staleness_val,
                    a_exp=a_exp_val,
                    loss=stat_utility,
                    b_exp=b_exp_val,
                    alpha_type="polynomial",
                    beta_type="polynomial_upshift",
                )
                if rate != 1.0:
                    logger.info(
                        f"Weighted received gradients by rate: {rate} with staleness: {staleness_val}, stat utility: {stat_utility}"
                    )
            except Exception as e:
                logger.warning(
                    f"Falling back to neutral rate due to error in weight_factor: {e}"
                )
                rate = 1.0

        for i, (name, param) in enumerate(np):  # Assuming self.params is a dict
            if param.requires_grad:
                if name in trainer_grad:
                    grad_device = self.grad[i].device
                    trainer_grad[name] = trainer_grad[name].to(grad_device)
                    # Ensure the layer name exists in trainer_grad

                    # Apply scalar rate: g'_i = rate * g_i
                    self.grad[i].add_(trainer_grad[name] * rate)
                else:
                    logger.warning(f"Gradient for {name} not found in trainer_grad.")

        # Also accumulate var-check gradients with the same rate if provided.
        # Assumption: grad_for_var_check is an iterable of tensors.
        if grad_for_var_check is not None:
            stacked = torch.stack(list(grad_for_var_check))
            self.grad_for_var_check_list.append(stacked * rate)

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
        """
        Aggregate local model GRADIENTS asynchronously for FwdLLM.

        This method is overridden from AsyncTopAgg.
        It receives gradients, aggregates them until _agg_goal is met,
        then performs FwdLLM variance check and model update.
        """
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.info("No channel found")
            return

        if channel.ends(VAL_CH_STATE_RECV) is None:
            logger.info("no ends yet")
            return
        time.sleep(0.1)  # Slight delay to allow messages to arrive

        msg, metadata = next(channel.recv_fifo(channel.ends(VAL_CH_STATE_RECV), 1))
        end, _ = metadata
        if not msg:
            logger.debug(f"No data from {end}; skipping it")
            return

        if MessageType.GRADIENTS in msg and MessageType.GRADIENTS_FOR_VAR_CHECK in msg:
            logger.info(
                f"Received gradients from {end} "
                f"with model version {msg[MessageType.MODEL_VERSION]}"
            )
            channel.set_end_property(
                end, PROP_LAST_SELECTED_ROUND, msg[MessageType.MODEL_VERSION]
            )
            channel.set_end_property(
                end, PROP_LAST_EVAL_ROUND, msg[MessageType.MODEL_VERSION]
            )
            # receiving stat_utility for every update from trainer
            if MessageType.STAT_UTILITY in msg:
                logger.info(
                    f"received stat_utility from {end} "
                    f"msg[MessageType.STAT_UTILITY] = {msg[MessageType.STAT_UTILITY]}"
                )
                channel.set_end_property(
                    end, PROP_STAT_UTILITY, msg[MessageType.STAT_UTILITY]
                )
        elif MessageType.STAT_UTILITY in msg:
            logger.info(
                f"Received eval-only message from {end}, "
                f"stat_utility {msg[MessageType.STAT_UTILITY]}"
            )
            channel.set_end_property(
                end, PROP_STAT_UTILITY, msg[MessageType.STAT_UTILITY]
            )
            channel.set_end_property(
                end, PROP_LAST_EVAL_ROUND, msg[MessageType.MODEL_VERSION]
            )
            channel._selector.trainer_eval_recv_ends.append(end)
            channel._selector.remove_from_selected_ends(channel._ends, end)
            channel._selector._cleanup_removed_ends(end)
            return
        else:
            logger.error(
                f"Invalid message received from {end} in aggregate_weights: {msg}"
            )
            return

        # TODO: Check if we want to discard after putting in the queue
        if self.reject_stale_updates == True:
            logger.info("Check trainer model version, disallow stale updates")
            if MessageType.MODEL_VERSION in msg:
                version = msg[MessageType.MODEL_VERSION]
                logger.info(
                    f"Model version aggregator: {self._model_version}, Model version trainer: {version}"
                )

            if version != self._model_version:
                logger.info(
                    f"Rejecting stale update with staleness: {self._model_version-version}. Trainer update version: {version}, "
                    f" self._model_version: {self._model_version}"
                )
                channel.cleanup_provided_ends(end)
                return

        channel._selector.ordered_updates_recv_ends.append(end)
        self._updates_in_queue += 1

        if MessageType.GRADIENTS in msg:
            trainer_gradients = msg[MessageType.GRADIENTS]
            version_for_rate = msg[MessageType.MODEL_VERSION]

            grad_for_var_check = (
                msg[MessageType.GRADIENTS_FOR_VAR_CHECK]
                if MessageType.GRADIENTS_FOR_VAR_CHECK in msg
                else None
            )

            self.aggregate_grads_from_trainers(
                trainer_gradients,
                version_for_rate=version_for_rate,
                stat_utility=channel.get_end_property(end, PROP_STAT_UTILITY),
                grad_for_var_check=grad_for_var_check,
            )

            # del trainer_gradients # Free memory

        if MessageType.GRADIENTS_FOR_VAR_CHECK in msg:
            self.grad_for_var_check_list.append(
                msg[MessageType.GRADIENTS_FOR_VAR_CHECK]
            )

        if MessageType.DATASET_SIZE in msg:
            count = msg[MessageType.DATASET_SIZE]
            channel.set_end_property(end, PROP_DATASET_SIZE, count)

        logger.info(f"Received and processed grads from {end}.")

        self._agg_goal_cnt += 1

        if self._agg_goal_cnt < self._agg_goal:
            logger.info(f"Agg goal not met. Have {self._agg_goal_cnt}/{self._agg_goal}")
            channel.set_end_property(
                end, PROP_UPDATE_COUNT, self._updates_recevied.get(end, 0) + 1
            )
            return

        if self._agg_goal_cnt == self._agg_goal:
            logger.info(
                f"Aggregation goal {self._agg_goal} reached. Performing FwdLLM aggregation."
            )

            self.grad_pool.append(self.grad)
            self.add_local_trained_result(
                0, self.grad, self._agg_goal_cnt
            )  # Assuming 0 is ok

            self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(
                self.model
            )
            self.grad = [torch.zeros_like(p) for p in self.params]

            self.aggregate(self._round)  # This sets self.var and self.var_good_enough

            if self.var_good_enough:

                logger.info(
                    f"Variance check PASSED. Evaluating model and advancing data_id."
                )
                self.iteration_per_data_id += 1  # This is iter 1 for the new data_id.
                result, _, _ = self.eval_model()
                logger.info(
                    f"Round {self._round}, Data ID {self.data_id} Eval Loss: {result['eval_loss']}"
                )
                self.data_id += 1
                self.iteration_per_data_id = 0  # Reset iteration count
                self._is_model_updated = True
                self._model_version += 1

                if self.data_id == self.total_data_bins:
                    logger.info(
                        f"All data bins complete. Incrementing round to {self._round + 1}"
                    )
                    self._round += 1
                    self.data_id = 0
                    channel.set_property(
                        "round", self._round
                    )  # Update channel property

            else:
                logger.info(
                    f"Variance check FAILED. Retrying on same data_id {self.data_id}."
                )
                self.iteration_per_data_id += 1
                self._is_model_updated = False

            self._updates_in_queue -= self._agg_goal
            self._agg_goal_cnt = 0  # Reset for the next batch
            # ASYNC: Clean up ends that just sent data
            logger.debug(
                "Agg goal reached, so resetting trainer end states in the channel"
            )
            channel.cleanup_recvd_ends()

    @timer_decorator
    def collect_and_accumulate_grads(self, tag, channel):
        """Aggregate trainer gradients synchronously, with timing and stage metadata."""
        # Create FwdLLMStage for timing/metrics logging
        self.fwd_llm_stage = FwdLLMStage(self._round, self.data_id, self.iteration_per_data_id, trainer_id=None)
        
        recv_ends = channel.ends()
        if self.ends_not_selected_yet and len(recv_ends) == 0:
            logger.info("no ends selected yet")
            return

        num_min_req = self._agg_goal  # change hardcoding, set it to aggGoal
        logger.info(f"Total ends: {len(recv_ends)}, required : {num_min_req}")
        num_min_req = min(num_min_req, len(recv_ends))
        if self.ends_not_selected_yet:
            # this is inefficient, but it will work
            # can improve this by tracking how many clients need to be freed up
            # If weights were not distributed in this iteration, async read messages from
            # one trainer until required trainers are available to distribute weights to
            # while maintaining concurrency.
            # This is best effort sync aggregation, if agg goal is not met, we default to async.
            logger.info(f"We are waiting to clear up queue")
            num_min_req = min(num_min_req, 1)

        for msg, metadata in channel.recv_fifo(channel.ends()):
            end, timestamp = metadata
            if not msg:
                logger.info(f"No data from {end}; skipping it")
                continue

            if MessageType.MODEL_VERSION in msg:
                version = msg[MessageType.MODEL_VERSION]

                if self.reject_stale_updates == True:
                    if version != self._model_version:
                        logger.info(
                            f"Rejecting trainer update from {end} of version {version}, "
                            f"agg self._model_version: {self._model_version}. Will return."
                        )
                        channel.cleanup_recvd_end(end)
                        # channel._selector.ordered_updates_recv_ends.append(end)
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
                trainer_gradients = msg[MessageType.GRADIENTS]
                version_for_rate = msg[MessageType.MODEL_VERSION]

                grad_for_var_check = (
                    msg[MessageType.GRADIENTS_FOR_VAR_CHECK]
                    if MessageType.GRADIENTS_FOR_VAR_CHECK in msg
                    else None
                )

                self.aggregate_grads_from_trainers(
                    trainer_gradients,
                    version_for_rate=version_for_rate,
                    stat_utility=channel.get_end_property(end, PROP_STAT_UTILITY),
                    grad_for_var_check=grad_for_var_check,
                )

            if MessageType.DATASET_SIZE in msg:
                count = msg[MessageType.DATASET_SIZE]
                channel.set_end_property(
                    end, PROP_DATASET_SIZE, msg[MessageType.DATASET_SIZE]
                )

            if MessageType.STAT_UTILITY in msg:
                logger.info(
                    f"received stat_utility from {end} "
                    f"msg[MessageType.STAT_UTILITY] {msg[MessageType.STAT_UTILITY]}"
                )
                channel.set_end_property(
                    end, PROP_STAT_UTILITY, msg[MessageType.STAT_UTILITY]
                )

            logger.info(
                f"Received grads from {end}. It was trained on model version {version}, with {count} samples"
            )

            if self._agg_goal_cnt == self._agg_goal:
                logger.info(
                    f"Reached agg_goal of {self._agg_goal} since agg_goal_count is {self._agg_goal_cnt}. Breaking from for loop, proceeding to aggregate."
                )
                break

        # second loop to poll more if needed
        # TODO(Aishwwarya): This stalls indefinitely when we have distributed weights but there are no more update to read.
        num_freed = 0  # if at least one is freed, exit loop
        while self._agg_goal_cnt < self._agg_goal and not self.ends_not_selected_yet:
            for msg, metadata in channel.recv_fifo(channel.ends(), 1):
                end, timestamp = metadata
                if not msg:
                    logger.info(f"No data from {end}; skipping it")
                    continue

                if MessageType.MODEL_VERSION in msg:
                    version = msg[MessageType.MODEL_VERSION]

                    if self.reject_stale_updates == True:
                        if version != self._model_version:
                            logger.info(
                                f"Rejecting trainer update from {end} of version {version}, "
                                f"agg self._model_version: {self._model_version}. Will return."
                            )
                            # num_freed += 1
                            channel.cleanup_recvd_end(end)
                            # channel._selector.ordered_updates_recv_ends.append(end)
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

    @timer_decorator
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

        # receive local model parameters from trainers
        self.collect_and_accumulate_grads(tag, channel)

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

        if self._agg_goal_cnt < self._agg_goal:
            # we enter this only if we have not distributed weights to enough clients
            # and want to free up resources

            # skip the aggregation
            # do not clean up the ends that we did not aggregate on yet
            # we are already cleaning up the ends that gave back stale results
            # channel.cleanup_recvd_ends()
            logger.info(f"did not reach agg goal, not aggregating")
            return

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
            self._is_model_updated = True
            # TODO: need to replace it with per end property
            if self.data_id == self.total_data_bins:
                logger.info("incrementing round number now ")
                self._round += 1
                self.data_id = 0
                channel.set_property("round", self._round)

            if self.config.hyperparameters.inc_model_version_per_data_id:
                self._model_version += 1
                logger.info(
                    f"incrementing model version to {self._model_version} now, round id: {self._round}"
                )
            else:
                self._model_version = self._round
            logger.info(f"Model version updated to: {self._model_version}")

        else:
            self.iteration_per_data_id += 1
            self._is_model_updated = False

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

        self.fwd_llm_stage = FwdLLMStage(
            self._round, self.data_id, self.iteration_per_data_id
        )
        channel.cleanup_recvd_ends()

        self.log_memory("end _aggregate_grads_sync", self.device)

    @timer_decorator
    def _force_cuda_memory_cleanup(self):
        torch.cuda.empty_cache()
        gc.collect()

    @timer_decorator
    def invoke_gc(self, payload):
        del payload
        gc.collect()

    @timer_decorator
    def eval_model(self, epoch=0, global_step=0, device=None):
        if not device:
            device = self.device

        logger.info(f"device inside eval_model() is set to: {device}")
        self.log_memory("start eval_model", self.device)

        results = {}

        eval_loss_total = torch.tensor(0.0, device=device)
        num_eval_steps = 0
        test_sample_len = len(self.test_global.dataset)
        
        # Move model to device before performing the eval
        self.model.to(device)
        self.model.eval()
        self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(
            self.model
        )

        # One-time GPU data transfer for caching test data
        if not hasattr(self, "_cached_test_data") or self._cached_test_data is None:
            logger.info("One-time GPU data transfer for evaluation dataset")
            self._cached_test_data = [t.to(device) for t in self.test_global.dataset.tensors]

        input_ids_all = self._cached_test_data[1]
        labels_all = self._cached_test_data[4]

        # Accumulate predictions on GPU
        preds_gpu = torch.empty((test_sample_len, self.num_labels), device=device)
        out_label_ids_gpu = torch.empty(test_sample_len, dtype=labels_all.dtype, device=device)

        batch_size = self.args.eval_batch_size
        loss_fct = CrossEntropyLoss()

        from torch.cuda.amp import autocast
        import contextlib
        autocast_cm = autocast() if self.args.fp16 else contextlib.nullcontext()
        with torch.no_grad(), autocast_cm:
            for batch_start_idx in range(0, test_sample_len, batch_size):
                batch_end_idx = min(batch_start_idx + batch_size, test_sample_len)
                
                x = input_ids_all[batch_start_idx:batch_end_idx]
                labels = labels_all[batch_start_idx:batch_end_idx]

                output = self.model(x)
                if hasattr(output, "logits"):
                    logits = output.logits
                elif isinstance(output, (tuple, list)):
                    logits = output[0]
                else:
                    logits = output

                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                eval_loss_total += loss

                preds_gpu[batch_start_idx:batch_end_idx] = logits
                out_label_ids_gpu[batch_start_idx:batch_end_idx] = labels
                num_eval_steps += 1

        # Move to CPU only once at the end
        eval_loss = (eval_loss_total / num_eval_steps).item()
        preds = preds_gpu.cpu().numpy()
        out_label_ids = out_label_ids_gpu.cpu().numpy()

        logger.info(
            f"# of batches: {num_eval_steps} with (batch_size, seq_len): {input_ids_all.shape}. test_sample_len: {test_sample_len}, preds.shape: {preds.shape}, location of model: {next(self.model.parameters()).device}"
        )

        model_outputs = preds
        preds_argmax = np.argmax(preds, axis=1)
        result, wrong = self.compute_metrics(
            preds_argmax, out_label_ids, self.test_global.examples
        )
        result["eval_loss"] = eval_loss
        results.update(result)

        # self.results.update(result)
        logging.info(
            f"results after eval are: {results}, len(wrong) is: {len(wrong)}, 'data_id_iterations': {self.iteration_per_data_id}"
        )

        # TODO: Check if model needs to be moved back to cpu? Do we need to keep
        # moving the model between CPU and GPU repeatedly?

        # Can delete x, labels, output, logits, loss in case we run out of memory
        self._force_cuda_memory_cleanup()

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

    def _distribute_weights_sync(
        self, tag: str, task_to_perform: str = "train"
    ) -> None:
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

        if self.trainer_event_dict is not None:
            curr_unavail_trainer_list = self.get_curr_unavail_trainers()
            channel.set_curr_unavailable_trainers(
                trainer_unavail_list=curr_unavail_trainer_list
            )
            logger.debug(
                f"Passed curr_unavail_trainer_list: "
                f"{curr_unavail_trainer_list} to channel"
            )

        else:  # Handling the case for oort's selector since it expects 3 # arguments
            channel.set_curr_unavailable_trainers(trainer_unavail_list=[])

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
                    f"sending weights to {end} with model_version: {self._model_version}, data_id: {self.data_id} for task: {task_to_perform}"
                )
                
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

                payload = {
                    MessageType.WEIGHTS: shared_weights,
                    MessageType.GRAD_POOL: shared_grad_pool_trainable,
                    MessageType.ROUND: self._round,
                    MessageType.MODEL_VERSION: self._model_version,
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
                    f"sending var = bad to {end} with model_version: {self._model_version}, round: {self._round}, data_id: {self.data_id} for task: {task_to_perform}"
                )
                payload = {
                    MessageType.VAR: "bad",
                    MessageType.ROUND: self._round,
                    MessageType.MODEL_VERSION: self._model_version,
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
            self.invoke_gc(payload)

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
                self._model_version
            ] = datetime.now()

    def _distribute_weights_async(
        self, tag: str, task_to_perform: str = "train"
    ) -> None:
        """
        Distribute a global model in asynchronous FL fashion - for FwdLLM.
        This method actually sends either gradients or calc_more_var to
        trainers, not the actual model weights.

        This method is overridden from one in asynchronous top aggregator
        (..top_aggregator).
        """
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found for tag {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()
        global_model_params = self.get_global_model_params()
        self.weights = global_model_params
        logger.debug(f"Starting busy wait at time {time.time()}")
        time.sleep(0.1)
        logger.debug(f"Ended busy wait at time {time.time()}")
        if self.trainer_event_dict is not None:
            curr_unavail_trainer_list = self.get_curr_unavail_trainers()
            channel.set_curr_unavailable_trainers(
                trainer_unavail_list=curr_unavail_trainer_list
            )
            logger.info(
                f"Passed curr_unavail_trainer_list: "
                f"{curr_unavail_trainer_list} to channel"
            )
        else:
            # Handling the case for oort's selector since it expects 3
            # arguments
            channel.set_curr_unavailable_trainers(trainer_unavail_list=[])

        logger.debug(
            f"Sending weights to trainers with task_to_perform = {task_to_perform}"
        )

        # check if there are any ends to send weights to
        self._curr_agg_version = (
            self._model_version,
            self.data_id,
            self.iteration_per_data_id,
        )
        logger.debug(
            f"Aggregator version state (model_version, data_id, iteration_id): {self._curr_agg_version}"
        )
        ends = channel.ends(
            state=VAL_CH_STATE_SEND,
            task_to_perform=task_to_perform,
            agg_version_state=self._curr_agg_version,
            trainer_version_states=self._trainer_state_dict,
        )
        logger.info(f"ends: {ends}")
        # TODO: check in agg_weights if ends is None
        if ends is None:
            self.ends_not_selected_yet = True
        else:
            self.ends_not_selected_yet = False
        if not ends:
            logger.debug(
                f"No trainers found for tag {tag}, will "
                f"move to get() for fetch weights from trainers"
            )
            return

        logger.info(f"Distributing tasks to {len(ends)} selected trainers...")

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
        if self.var_good_enough:
            logger.info(
                f"sending weights to {ends} with model_version: {self._model_version}, round: {self._round}, data_id: {self.data_id} for task: {task_to_perform}"
            )
            logger.info(
                "Variance is GOOD. Preparing and sending new model weights and grad_pool."
            )
            self.print_trainable_params_stats(
                location="[populate_params, _distr_weights]"
            )
            trainable_params = self.get_trainable_param_state_dict()
            shared_weights = weights_to_device(trainable_params, DeviceType.CPU)

            shared_grad_pool = self.aggregate_grad_pool(self.grad_pool)

            shared_grad_pool_trainable = []
            if shared_grad_pool is None:
                shared_grad_pool_trainable = None
            else:
                idx = 0
                for param in self.model.parameters():
                    if param.requires_grad:
                        shared_grad_pool_trainable.append(shared_grad_pool[idx].clone())
                    idx += 1

            # Clear pools after model has been updated and we move to the next round!
            if self._is_model_updated:
                self.grad_pool = []  # update when model version is updated!
                self.grad_for_var_check_list = []  # Update once agg goal met
                self._is_model_updated = False

            payload = {
                MessageType.WEIGHTS: shared_weights,
                MessageType.GRAD_POOL: shared_grad_pool_trainable,
                MessageType.ROUND: self._round,
                MessageType.MODEL_VERSION: self._model_version,
                MessageType.TASK_TO_PERFORM: task_to_perform,
                MessageType.DATA_ID: self.data_id,
                MessageType.ITERATION_PER_DATA_ID: self.iteration_per_data_id,
            }
            # Clean up memory
            # del shared_weights
            # del shared_grad_pool
            # del shared_grad_pool_trainable

        else:
            logger.info(
                f"sending var = bad to {ends} with model_version: {self._model_version}, round: {self._round}, data_id: {self.data_id} for task: {task_to_perform}"
            )
            logger.info("Variance is BAD. Sending request for more variance checks.")
            payload = {
                MessageType.VAR: "bad",
                MessageType.ROUND: self._round,
                MessageType.MODEL_VERSION: self._model_version,
                MessageType.TASK_TO_PERFORM: task_to_perform,
                MessageType.DATA_ID: self.data_id,
                MessageType.ITERATION_PER_DATA_ID: self.iteration_per_data_id,
            }

        # ASYNC SEND LOOP (from AsyncTopAgg)
        for end in ends:
            # Updated the trainer state dict
            self._trainer_state_dict[end] = (
                self._model_version,
                self.data_id,
                self.iteration_per_data_id,
            )

            logger.info(
                f"Sending payload to {end} with model_version: {self._model_version}, "
                f"data_id: {self.data_id}, iter: {self.iteration_per_data_id}"
            )

            # Set OORT property (from AsyncTopAgg)
            channel.set_end_property(
                end, PROP_ROUND_START_TIME, (self._round, datetime.now())
            )

            # Send the payload
            channel.send(end, payload)

            # Track send time (from AsyncTopAgg)
            if end not in self._track_trainer_version_duration_s.keys():
                self._track_trainer_version_duration_s[end] = {
                    "last_send_wts_ts": -1,
                    "sent_wts_version_ts": {},
                    "recv_wts_version_ts": {},
                    "total_training_time_s": 0,  # Initialize to 0
                }
            self._track_trainer_version_duration_s[end]["sent_wts_version_ts"][
                self._model_version
            ] = datetime.now()

        # Clean up the large payload object
        del payload
        gc.collect()

    def _distribute_weights(self, tag: str, task_to_perform: str = "train") -> None:
        if self.is_async:
            logger.info("Inside distribute of async")
            self._distribute_weights_async(tag, task_to_perform)
        else:
            logger.info("Inside distribute of sync")
            self._distribute_weights_sync(tag, task_to_perform)

    def _aggregate_weights(self, tag: str) -> None:
        if self.is_async:
            logger.info("Inside async aggregator")
            self._aggregate_grads_async(tag)
        else:
            logger.info("Inside sync aggregator")
            self._aggregate_grads_sync(tag)

    # TODO: Cleanup compose loop
    def compose(self) -> None:
        """Compose role with tasklets."""
        logger.info(f"Fetch is_async value from config:")
        if self.config.selector.kwargs.get("is_async") is not None:
            self.is_async = self.config.selector.kwargs.get("is_async")
        else:
            self.is_async = False

        if self.is_async:
            super().compose()
            with CloneComposer(self.composer) as _:
                task_internal_init = Tasklet("internal_init", self.internal_init)

                task_reset_agg_goal_vars = Tasklet(
                    "reset_agg_goal_vars", self._reset_agg_goal_variables
                )

                # Created separate put tasklets for train and eval
                task_put_train = Tasklet(
                    "distribute", self.put, TAG_DISTRIBUTE, "train"
                )

                task_get_weights = Tasklet(
                    "aggregate", self._aggregate_weights, TAG_AGGREGATE
                )

                # task_get_heartbeat = Tasklet("heartbeat", self.get,
                # TAG_HEARTBEAT)
                task_init = Tasklet("initialize", self.initialize)

            c = self.composer
            c.unlink()

            loop = Loop(loop_check_fn=lambda: self._work_done)
            # create a loop object for asyncfl to manage concurrency as
            # well as aggregation goal
            asyncfl_loop = Loop(
                loop_check_fn=lambda: self._agg_goal_cnt == self._agg_goal
            )
            logger.info("Hybrid compose")

            # chain them again with new tasklets introduced in this class
            (
                task_internal_init
                >> task_init
                >> loop(
                    task_reset_agg_goal_vars
                    # >> asyncfl_loop(task_put >> task_get_weights >>
                    >> asyncfl_loop(task_put_train >> task_get_weights)
                    >> c.tasklet("analysis")
                    >> c.tasklet("save_metrics")
                )
                >> c.tasklet("inform_end_of_training")
            )
        else:
            logger.info("Sync loop")
            super().compose()

            with CloneComposer(self.composer) as _:
                task_internal_init = Tasklet("internal_init", self.internal_init)
                task_pause_exec = Tasklet("pause_exec", self.pause_execution)

                task_reset_agg_goal_vars = Tasklet(
                    "reset_agg_goal_vars", self._reset_agg_goal_variables
                )

                # Created separate put tasklets for train and eval
                task_put_train = Tasklet(
                    "distribute", self.put, TAG_DISTRIBUTE, "train"
                )

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
