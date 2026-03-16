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
import torch.nn.functional as F
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
import math


logger = logging.getLogger(__name__)

PROP_ROUND_END_TIME = "round_end_time"

SEND_TIMEOUT_WAIT_S = 90  # 90 seconds timeout

import hashlib


def _calculate_hash(tensor):
    if tensor is None:
        return ""

    """Calculate a hash for a tensor for logging."""
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def log_error_distribution(probs, labels):
    """
    Analyzes error distribution across 4 classes and logs via logging.info.
    probs: ndarray [N, 4] - Softmax probabilities
    labels: ndarray [N] - Integer ground truth
    """
    # 1. Prediction and Error Masking
    if hasattr(probs, "detach"):
        probs = probs.detach().cpu().numpy()
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()

    # 2. Handle the float labels safely
    actual_labels = np.round(labels).astype(int)
    preds = np.argmax(probs, axis=1)

    wrong_mask = preds != actual_labels

    if not np.any(wrong_mask):
        logging.info("Accuracy is 100%.")
        return

    # 3. Filter for wrong predictions (Now safely NumPy)
    wrong_probs = probs[wrong_mask]
    logging.info(f"wrong probs len = {len(wrong_probs)}")

    # Now this will work perfectly
    confidences = np.max(wrong_probs, axis=1)

    # Margin calculation
    sorted_wrong = np.sort(wrong_probs, axis=1)
    margins = sorted_wrong[:, -1] - sorted_wrong[:, -2]

    # 3. Binning Logic (0.0 to 1.0)
    bins = np.linspace(0, 1.0, 11)
    margin_bins = np.digitize(margins, bins) - 1

    logging.debug("=== Error Distribution Analysis (Incorrect Predictions Only) ===")
    logging.debug(
        f"{'Margin Bin':<12} | {'Count':<8} | {'Avg Confidence':<15} | {'Max Confidence'}"
    )
    logging.debug("-" * 65)

    for i in range(len(bins) - 1):
        mask = margin_bins == i
        count = np.sum(mask)
        bin_label = f"{bins[i]:.1f}-{bins[i+1]:.1f}"

        if count > 0:
            avg_conf = np.mean(confidences[mask])
            max_conf = np.max(confidences[mask])
            logging.debug(
                f"{bin_label:<12} | {count:<8} | {avg_conf:<15.4f} | {max_conf:.4f}"
            )
        else:
            logging.debug(f"{bin_label:<12} | 0        | -               | -")

    # 4. Summary Statistics for "Confidently Wrong" samples
    high_margin_count = np.sum(margins > 0.5)
    logging.debug(
        f"Summary: {high_margin_count} errors have a margin > 0.5 (Confidently Wrong)."
    )


def log_margin_distribution(probs):
    # 2. Get the Top 2 values for every sample
    #    values shape: [N, 2], indices shape: [N, 2]
    top2_values, top2_indices = torch.topk(probs, k=2, dim=1)

    # 3. Calculate the Margin (Gap)
    #    Column 0 is the Winner, Column 1 is the Runner-up
    margins = top2_values[:, 0] - top2_values[:, 1]
    margins = margins.numpy()

    # 4. Define your "Indecision Zone"
    #    Samples where the gap between winner and loser is tiny (< 0.1)
    indecisive_count = np.sum(margins < 0.1)

    logging.debug(f"\n--- INDECISION REPORT ---")
    logging.debug(f"  Total Samples: {len(margins)}")
    logging.debug(
        f"  Samples with Margin < 0.1: {indecisive_count} ({(indecisive_count/len(margins))*100:.1f}%)"
    )
    logging.debug(f"  Avg Margin: {np.mean(margins):.4f}")

    # 5. (Optional) Histogram the margins to see the spread
    hist, bin_edges = np.histogram(margins, bins=10, range=(0.0, 1.0))
    logging.debug(f"  Margin Distribution: {hist}")
    logging.debug("------------------------------------------\n")


def compute_metrics_with_logging(probs, preds, out_label_ids, examples):

    logging.debug(f"'Hash' |  'Prob'  | 'Pred' | 'Actual'")

    for i, batch in enumerate(examples):
        batch = tuple(t.to("cpu") for t in batch)
        for j, example in enumerate(batch[1]):

            pred = preds[i * 8 + j]
            actual = out_label_ids[i * 8 + j]
            prob = probs[i * 8 + j]

            # 2. Print the row
            # We slice the hash to [:10] for better readability in the console
            logging.debug(
                f"{_calculate_hash(example)}... | {prob} | {pred} | {actual} "
            )

    return


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

        current_dir = os.path.dirname(os.path.abspath(__file__))
        files_path = os.path.join(
            current_dir, "../../../../examples/fwdllm/expts/run_tc_expts/json_scripts"
        )
        search_pattern = os.path.join(files_path, "trainer_*.json")
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
        format_hash = lambda d: {k: _calculate_hash(v)[:8] for k, v in d.items()}
        logger.debug(f"Trainer grad received {format_hash(trainer_grad)}")
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
            logger.debug(f"no weighted aggregation, rate = 1.0")
            rate = 1.0
        else:
            staleness_val = self._model_version - version_for_rate

            if self.optimizer.agg_rate_conf["type"] == "old":
                rate = 1 / math.sqrt(1 + staleness_val)  # As per the Fedbuff paper

            elif self.optimizer.agg_rate_conf["type"] == "new":
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
                except Exception as e:
                    logger.warning(
                        f"Falling back to neutral rate due to error in weight_factor: {e}"
                    )
                    rate = 1.0

        if rate != 1.0:
            logger.info(
                f"Weighted received gradients by rate: {rate} with staleness: {staleness_val}, stat utility: {stat_utility}"
            )

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
        # time.sleep(0.1)  # Slight delay to allow messages to arrive

        msg, metadata = next(channel.recv_fifo(channel.ends(VAL_CH_STATE_RECV), 1))
        end, timestamp = metadata
        if not msg:
            logger.debug(f"No data from {end}; skipping it")
            return

        # Use new extracted helper
        if not self._process_single_trainer_message(channel, msg, end, timestamp):
            return

        logger.info(f"Received and processed grads from {end}.")

        if self._agg_goal_cnt < self._agg_goal:
            logger.info(f"Agg goal not met. Have {self._agg_goal_cnt}/{self._agg_goal}")
            channel.set_end_property(
                end, PROP_UPDATE_COUNT, self._updates_recevied.get(end, 0) + 1
            )
            return

        if self._agg_goal_cnt == self._agg_goal:
            self._process_aggregation_goal_met(tag, channel, is_async=True)

    @timer_decorator
    def _process_single_trainer_message(self, channel, msg, end, timestamp):
        if MessageType.MODEL_VERSION in msg:
            version = msg[MessageType.MODEL_VERSION]
            if version != self._model_version:
                logger.info(
                    f"Received grad with staleness={self._model_version-version}."
                )
            if self.reject_stale_updates == True:
                if version != self._model_version:
                    logger.info(
                        f"Rejecting trainer update from {end} of version {version}, "
                        f"agg self._model_version: {self._model_version}. Will return."
                    )
                    if self.is_async:
                        channel.cleanup_provided_ends(end)
                    else:
                        channel.cleanup_recvd_end(end)
                    return False

        if MessageType.GRADIENTS in msg and MessageType.GRADIENTS_FOR_VAR_CHECK in msg:
            logger.info(
                f"received gradients from {end} "
                f"with model version {msg[MessageType.MODEL_VERSION]}"
            )
            self._agg_goal_cnt += 1

            channel.set_end_property(
                end, PROP_LAST_SELECTED_ROUND, msg[MessageType.MODEL_VERSION]
            )
            channel.set_end_property(
                end, PROP_LAST_EVAL_ROUND, msg[MessageType.MODEL_VERSION]
            )
            logger.debug(
                f"Getting channel property {PROP_ROUND_START_TIME} for " f"end {end}"
            )
            round_start_time_tup = channel.get_end_property(end, PROP_ROUND_START_TIME)
            logger.debug(
                f"Returned round_start_time_tup: {round_start_time_tup} for "
                f"end {end} and timestamp {timestamp}"
            )
            if round_start_time_tup is not None:
                sent_ts = round_start_time_tup[1]
                round_duration = timestamp - sent_ts
                channel.set_end_property(end, PROP_ROUND_DURATION, round_duration)
                logger.info(
                    f"Set PROP_ROUND_DURATION for {end}: {round_duration.total_seconds():.3f}s"
                )
        else:
            logger.error(
                f"Invalid message received from {end} in aggregate_weights: {msg}"
            )
            return False

        logger.debug(f"received data from {end}")
        channel.set_end_property(end, PROP_ROUND_END_TIME, (self._round, timestamp))

        channel._selector.ordered_updates_recv_ends.append(end)
        self._updates_in_queue += 1
        self._per_round_update_list.append(end)

        if end not in self._updates_recevied.keys():
            self._updates_recevied[end] = 1
        else:
            self._updates_recevied[end] += 1

        if MessageType.GRADIENTS in msg:
            trainer_gradients = msg[MessageType.GRADIENTS]
            version_for_rate = msg[MessageType.MODEL_VERSION]
            grad_for_var_check = (
                msg[MessageType.GRADIENTS_FOR_VAR_CHECK]
                if MessageType.GRADIENTS_FOR_VAR_CHECK in msg
                else None
            )
            logger.debug(
                f"Calling aggregate_grads_for_trainers with grad_for_var_check: {_calculate_hash(grad_for_var_check)}"
            )
            self.aggregate_grads_from_trainers(
                trainer_gradients,
                version_for_rate=version_for_rate,
                stat_utility=channel.get_end_property(end, PROP_STAT_UTILITY),
                grad_for_var_check=grad_for_var_check,
            )

            # del trainer_gradients # Free memory

        # This will add to the var check list twice, it is already added once
        # within self.aggregate_grads_from_trainers
        # if MessageType.GRADIENTS_FOR_VAR_CHECK in msg:
        #     self.grad_for_var_check_list.append(
        #         msg[MessageType.GRADIENTS_FOR_VAR_CHECK]
        #     )

        count = 0
        if MessageType.DATASET_SIZE in msg:
            count = msg[MessageType.DATASET_SIZE]
            channel.set_end_property(end, PROP_DATASET_SIZE, count)

        if MessageType.STAT_UTILITY in msg:
            logger.info(
                f"received stat_utility from {end} "
                f"msg[MessageType.STAT_UTILITY] {msg[MessageType.STAT_UTILITY]}"
            )
            channel.set_end_property(
                end, PROP_STAT_UTILITY, msg[MessageType.STAT_UTILITY]
            )
            logger.info(f"grad_pool already has: {len(self.grad_pool)}")

        version = msg.get(MessageType.MODEL_VERSION, "unknown")
        logger.info(
            f"Received grads from {end}. It was trained on model version {version}, with {count} samples"
        )
        channel.remove_from_selected_ends(end)
        return True

    @timer_decorator
    def _process_aggregation_goal_met(self, tag, channel, is_async=False):
        logger.info(
            f"Aggregation goal {self._agg_goal} reached. Performing FwdLLM aggregation."
        )

        self.grad_pool.append(self.grad)
        format_hash = lambda d: [_calculate_hash(v) for v in d]
        logger.debug(
            f"self.grad when agg goal met - length : {len(self.grad)} - hash :  {format_hash(self.grad)}"
        )

        self.add_local_trained_result(0, self.grad, self._agg_goal_cnt)

        self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(
            self.model
        )
        self.grad = [torch.zeros_like(p) for p in self.params]

        self.aggregate(self._round)

        if self.var_good_enough:
            logger.info(
                f"Variance check PASSED. Evaluating model and advancing data_id."
            )
            self.iteration_per_data_id += 1
            result, _, _ = self.eval_model()
            logger.info(
                f"Round {self._round}, Data ID {self.data_id} Eval Loss: {result['eval_loss']}"
            )
            self.data_id += 1
            self.iteration_per_data_id = 0
            self._is_model_updated = True

            if self.config.hyperparameters.inc_model_version_per_data_id:
                self._model_version += 1
            else:
                self._model_version = self._round

            if self.data_id == self.total_data_bins:
                logger.info(
                    f"All data bins complete. Incrementing round to {self._round + 1}"
                )
                self._round += 1
                self.data_id = 0
                channel.set_property("round", self._round)

        else:
            logger.info(
                f"Variance check FAILED. Retrying on same data_id {self.data_id}."
            )
            self.iteration_per_data_id += 1
            self._is_model_updated = False

        self._updates_in_queue -= self._agg_goal
        self._agg_goal_cnt = 0

        self.fwd_llm_stage = FwdLLMStage(
            self._round, self.data_id, self.iteration_per_data_id
        )

        if is_async:
            logger.debug(
                "Agg goal reached, so resetting trainer end states in the channel"
            )
        channel.cleanup_recvd_ends()

        # Centralized cleanup
        # self._force_cuda_memory_cleanup()

    @timer_decorator
    def sync_collect_and_accumulate_grads(self, tag, channel):
        """Aggregate trainer gradients synchronously, with timing and stage metadata."""
        self.fwd_llm_stage = FwdLLMStage(
            self._round, self.data_id, self.iteration_per_data_id, trainer_id=None
        )

        recv_ends = channel.ends()

        num_min_req = self._agg_goal  # change hardcoding, set it to aggGoal
        logger.info(f"Total ends: {len(recv_ends)}, required : {num_min_req}")
        num_min_req = min(num_min_req, len(recv_ends))
        if self.ends_not_selected_yet:
            logger.info(f"We are waiting to clear up queue")
            num_min_req = min(num_min_req, 1)

        for msg, metadata in channel.recv_fifo(channel.ends(), num_min_req):
            end, timestamp = metadata
            if not msg:
                logger.info(f"No data from {end}; skipping it")
                continue

            self._process_single_trainer_message(channel, msg, end, timestamp)

            if self._agg_goal_cnt >= self._agg_goal:
                logger.info(
                    f"Reached agg_goal of {self._agg_goal} since agg_goal_count is {self._agg_goal_cnt}. Breaking from for loop, proceeding to aggregate."
                )
                break

        # Second loop

    @timer_decorator
    def _aggregate_grads_sync(self, tag: str) -> None:
        """Aggregate trainer gradients synchronously."""
        logger.info("starting aggregate_grads_sync")
        self.log_memory("start _aggregate_grads_sync", self.device)
        self.print_trainable_params_stats(location="[start,_aggregate_grads_sync()]")

        channel = self.cm.get_by_tag(tag)
        if not channel:
            return

        logger.debug(f"Channel {channel} found for tag {tag}")
        if channel.ends(VAL_CH_STATE_RECV) is None:
            logger.info("no ends yet")
            return

        # receive local model parameters from trainers
        self.sync_collect_and_accumulate_grads(tag, channel)

        logger.debug(f"received {len(self.cache)} trainer updates in cache")

        if self._agg_goal_cnt < self._agg_goal:
            logger.info(f"did not reach agg goal, not aggregating")
            return

        self._process_aggregation_goal_met(tag, channel, is_async=False)
        self.log_memory("end _aggregate_grads_sync", self.device)

    @timer_decorator
    def _force_cuda_memory_cleanup(self):
        torch.cuda.empty_cache()
        gc.collect()

    @timer_decorator
    def invoke_gc(self):
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
            self._cached_test_data = [
                t.to(device) for t in self.test_global.dataset.tensors
            ]

        input_ids_all = self._cached_test_data[1]
        labels_all = self._cached_test_data[4]

        # Accumulate predictions on GPU
        preds_gpu = torch.empty((test_sample_len, self.num_labels), device=device)
        out_label_ids_gpu = torch.empty(
            test_sample_len, dtype=labels_all.dtype, device=device
        )

        batch_size = self.args.eval_batch_size
        loss_fct = CrossEntropyLoss()

        from torch.cuda.amp import autocast
        import contextlib

        autocast_cm = autocast() if self.args.fp16 else contextlib.nullcontext()
        if not self.args.fp16:
            logging.warning(f"Autocast is disabled: {self.args.fp16}")

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

        # Uncomment the below to log the prediction stats
        probs = F.softmax(torch.tensor(preds), dim=1)  # Shape: [N, 4]
        log_margin_distribution(probs)
        compute_metrics_with_logging(probs, preds, out_label_ids, self.test_global)
        log_error_distribution(probs, out_label_ids)

        result["eval_loss"] = eval_loss
        results.update(result)

        # self.results.update(result)
        logging.info(
            f"results after eval are: {results}, len(wrong) is: {len(wrong)}, 'data_id_iterations': {self.iteration_per_data_id}"
        )

        # TODO: Check if model needs to be moved back to cpu? Do we need to keep
        # moving the model between CPU and GPU repeatedly?

        # Can delete x, labels, output, logits, loss in case we run into any memory issues
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

    @timer_decorator
    def _prepare_distribution_payload(self, task_to_perform: str):
        if self.var:
            logger.info(
                f"self.var = {self.var}, self.var_threshold = {self.var_threshold}"
            )

        if not self.var_good_enough:
            logger.info(
                "Sending variance = bad to trainers since variance is greater than threshold"
            )
            logger.info("Variance is BAD. Sending request for more samples.")
            return {
                MessageType.VAR: "bad",
                MessageType.ROUND: self._round,
                MessageType.MODEL_VERSION: self._model_version,
                MessageType.TASK_TO_PERFORM: task_to_perform,
                MessageType.DATA_ID: self.data_id,
                MessageType.ITERATION_PER_DATA_ID: self.iteration_per_data_id,
            }

        logger.info(
            "Will send new weights to ends since variance is less than threshold"
        )
        logger.info("Variance is GOOD. Preparing new model weights and grad_pool.")

        self.print_trainable_params_stats(location="[_prepare_distribution_payload]")
        trainable_params = self.get_trainable_param_state_dict()

        shared_weights = weights_to_device(
            trainable_params, DeviceType.CPU
        )  # Need to move to CPU for sending over MQTT

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

        payload = {
            MessageType.WEIGHTS: shared_weights,
            MessageType.GRAD_POOL: shared_grad_pool_trainable,
            MessageType.ROUND: self._round,
            MessageType.MODEL_VERSION: self._model_version,
            MessageType.TASK_TO_PERFORM: task_to_perform,
            MessageType.DATA_ID: self.data_id,
            MessageType.ITERATION_PER_DATA_ID: self.iteration_per_data_id,
        }

        return payload

    def _update_state_after_payload_prepared(self):
        """Update state after preparing payload.
        Reset grad pools if the model was updated.
        """
        if self._is_model_updated:
            self.grad_pool = []
            self.grad_for_var_check_list = []
            self._is_model_updated = False

    @timer_decorator
    def _distribute_weights_sync(
        self, tag: str, task_to_perform: str = "train"
    ) -> None:
        """Distribute a global model in synchronous FL fashion - for FwdLLM.
        This method actually sends either gradients or calc_more_var to
        trainers, not the actual model weights.

        This method is overridden from one in synchronous top aggregator
        """

        logger.info(f"Device for agg: {next(self.model.parameters()).device}")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found for tag {tag}")
            return

        channel.await_join()
        global_model_params = self.get_global_model_params()
        format_hash = lambda d: {k: _calculate_hash(v)[:8] for k, v in d.items()}
        logging.info(
            f"Model distributed to clients (Hashed): {format_hash(global_model_params)}"
        )
        self.weights = global_model_params

        logger.debug(f"Starting busy wait at time {time.time()}")
        time.sleep(0.1)
        logger.debug(f"Ended busy wait at time {time.time()}")

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
            channel.set_curr_unavailable_trainers(trainer_unavail_list=[])

        ends = channel.ends(VAL_CH_STATE_SEND, task_to_perform)
        logger.info(f"ends: {ends}")
        if ends is None or len(ends) >= self._agg_goal:
            self.ends_not_selected_yet = True
        else:
            self.ends_not_selected_yet = False

        if not ends:
            logger.debug(
                f"No trainers found for tag {tag}, will "
                f"move to get() for fetch weights from trainers"
            )
            return

        payload = self._prepare_distribution_payload(task_to_perform)
        self._update_state_after_payload_prepared()

        for end in ends:
            logger.debug(
                f"Setting channel property {PROP_ROUND_START_TIME} for "
                f"end {end}. For round {self._round} at time: {datetime.now()}"
            )
            channel.set_end_property(
                end, PROP_ROUND_START_TIME, (self._round, datetime.now())
            )

            if self.var_good_enough:
                logger.info(
                    f"sending weights to {end} with model_version: {self._model_version}, data_id: {self.data_id} for task: {task_to_perform}"
                )

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
            else:
                logger.info(
                    f"sending var = bad to {end} with model_version: {self._model_version}, round: {self._round}, data_id: {self.data_id} for task: {task_to_perform}"
                )

                msg_bytes = pickle.dumps(payload)
                logger.info(
                    f"[DEBUG] Payload size for {end}: {len(msg_bytes) / (1024 * 1024):.2f} MB"
                )

            channel.send(end, payload)
            logger.info(f"Sent weights to {end}")
            # self.invoke_gc()

    @timer_decorator
    def _distribute_weights_async(
        self, tag: str, task_to_perform: str = "train"
    ) -> None:
        """Distribute a global model in asynchronous FL fashion - for FwdLLM."""
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

        payload = self._prepare_distribution_payload(task_to_perform)
        self._update_state_after_payload_prepared()

        if self.var_good_enough:
            logger.info(
                f"Async: sending weights to {ends} with model_version: {self._model_version}, round: {self._round}, data_id: {self.data_id} for task: {task_to_perform}"
            )

        for end in ends:
            logger.debug(
                f"Setting channel property {PROP_ROUND_START_TIME} for "
                f"end {end}. For round {self._round} at time: {datetime.now()}"
            )
            channel.set_end_property(
                end, PROP_ROUND_START_TIME, (self._round, datetime.now())
            )
            channel.send(end, payload)
        logger.info(f"Sent weights to all ends")

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
