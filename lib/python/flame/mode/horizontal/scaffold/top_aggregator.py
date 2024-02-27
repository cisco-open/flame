# Copyright 2023 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""SCAFFOLD horizontal FL top level aggregator."""

import logging
import time
from copy import deepcopy
from datetime import datetime

from diskcache import Cache
from flame.common.constants import (
    PROP_ROUND_END_TIME,
    PROP_ROUND_START_TIME,
    DeviceType,
    TrainState,
)
from flame.common.util import (
    MLFramework,
    get_ml_framework_in_use,
    weights_to_device,
    weights_to_model_device,
)
from flame.mode.composer import Composer
from flame.mode.horizontal.syncfl.top_aggregator import TAG_AGGREGATE, TAG_DISTRIBUTE
from flame.mode.horizontal.syncfl.top_aggregator import (
    TopAggregator as BaseTopAggregator,
)
from flame.mode.message import MessageType
from flame.mode.tasklet import Loop, Tasklet
from flame.optimizer.train_result import TrainResult

logger = logging.getLogger(__name__)

TAG_GET_DATATSET_SIZE = "getDatasetSize"


class TopAggregator(BaseTopAggregator):
    """Top level Aggregator implements an ML aggregation role."""

    def internal_init(self) -> None:
        """Initialize internal state for role."""
        ml_framework_in_use = get_ml_framework_in_use()

        # only support pytorch
        if ml_framework_in_use != MLFramework.PYTORCH:
            raise NotImplementedError(
                "supported ml framework not found; "
                f"supported frameworks (for feddyn) are: {[MLFramework.PYTORCH.name.lower()]}"
            )

        super().internal_init()

        # second cache is added for CONTROL_WEIGHTS messages (same shape as the model)
        self.control_cache = Cache()
        self.control_cache.reset("size_limit", 1e15)
        self.control_cache.reset("cull_limit", 0)

    def _aggregate_weights(self, tag: str) -> None:
        channel = self.cm.get_by_tag(tag)
        if not channel:
            return

        total = 0

        # receive local model parameters from trainers
        for msg, metadata in channel.recv_fifo(channel.ends()):
            end, timestamp = metadata
            if not msg:
                logger.debug(f"No data from {end}; skipping it")
                continue

            logger.debug(f"received data from {end}")
            channel.set_end_property(end, PROP_ROUND_END_TIME, (round, timestamp))

            if MessageType.WEIGHTS in msg:
                weights = weights_to_model_device(msg[MessageType.WEIGHTS], self.model)

            if MessageType.CONTROL_WEIGHTS in msg:
                c_weights = weights_to_model_device(
                    msg[MessageType.CONTROL_WEIGHTS], self.model
                )

            if MessageType.DATASET_SIZE in msg:
                count = msg[MessageType.DATASET_SIZE]

            if MessageType.DATASAMPLER_METADATA in msg:
                self.datasampler.handle_metadata_from_trainer(
                    msg[MessageType.DATASAMPLER_METADATA],
                    end,
                    channel,
                )

            logger.debug(f"{end}'s parameters trained with {count} samples")

            if weights is not None and count > 0:
                total += count
                tres = TrainResult(weights, count)
                # save training result from trainer in a disk cache
                self.cache[end] = tres
                # save c_weights result from the trainer
                c_tres = TrainResult(c_weights)
                self.control_cache[end] = c_tres

        logger.debug(f"received {len(self.cache)} trainer updates in cache")

        # optimizer conducts optimization (in this case, aggregation)
        global_weights = self.optimizer.do(
            deepcopy(self.weights),
            self.cache,
            total=total,
            num_trainers=len(channel.ends()),
            control_cache=self.control_cache,
        )
        if global_weights is None:
            logger.debug("failed model aggregation")
            time.sleep(1)
            return

        # set global weights
        self.weights = global_weights

        # update model with global weights
        self._update_model()

    def get_dataset_size(self, tag: str) -> None:
        """Get all dataset sizes of trainers prior to training on the dataset."""
        logger.debug("calling get_dataset_size")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            return

        self.dataset_sizes = dict()

        # receive dataset size from all trainers
        all_ends = channel.all_ends()
        for msg, metadata in channel.recv_fifo(all_ends):
            end, _ = metadata
            if not msg:
                logger.debug(f"No data from {end}; skipping it")
                continue

            logger.debug(f"received data from {end}")
            if MessageType.DATASET_SIZE in msg:
                self.dataset_sizes[end] = msg[MessageType.DATASET_SIZE]

        # record all active trainers along with the sizes of their datasets
        self.optimizer.save_state(TrainState.PRE, dataset_sizes=self.dataset_sizes)
        logger.debug(f"dataset sizes: {self.dataset_sizes}")
        logger.debug("exiting get_dataset_size")

    def _distribute_weights(self, tag: str) -> None:
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found for tag {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        # before distributing weights, update it from global model
        self._update_weights()

        # if c_glob variable not defined in optimizer, fill with 0s
        self.optimizer.save_state(TrainState.PRE, glob_weights=self.weights)

        selected_ends = channel.ends()
        datasampler_metadata = self.datasampler.get_metadata(self._round, selected_ends)

        # send out global model parameters to trainers
        for end in selected_ends:
            logger.debug(f"sending weights to {end}")
            channel.send(
                end,
                {
                    MessageType.WEIGHTS: weights_to_device(
                        self.weights, DeviceType.CPU
                    ),
                    MessageType.CLIENT_WEIGHT: self.optimizer.weight_dict.get(end, 1),
                    MessageType.CONTROL_WEIGHTS: weights_to_device(
                        self.optimizer.c_glob, DeviceType.CPU
                    ),
                    MessageType.ROUND: self._round,
                    MessageType.DATASAMPLER_METADATA: datasampler_metadata,
                },
            )
            # register round start time on each end for round duration measurement.
            channel.set_end_property(
                end, PROP_ROUND_START_TIME, (round, datetime.now())
            )

    def compose(self) -> None:
        """Compose role with tasklets."""
        with Composer() as composer:
            self.composer = composer

            task_internal_init = Tasklet("internal_init", self.internal_init)

            task_init = Tasklet("initialize", self.initialize)

            task_load_data = Tasklet("load_data", self.load_data)

            task_get_dataset = Tasklet(
                "get_dataset_size", self.get_dataset_size, TAG_GET_DATATSET_SIZE
            )

            task_put = Tasklet("distribute", self.put, TAG_DISTRIBUTE)

            task_get = Tasklet("aggregate", self.get, TAG_AGGREGATE)

            task_train = Tasklet("train", self.train)

            task_eval = Tasklet("evaluate", self.evaluate)

            task_analysis = Tasklet("analysis", self.run_analysis)

            task_save_metrics = Tasklet("save_metrics", self.save_metrics)

            task_increment_round = Tasklet("inc_round", self.increment_round)

            task_end_of_training = Tasklet(
                "inform_end_of_training", self.inform_end_of_training
            )

            task_save_params = Tasklet("save_params", self.save_params)

            task_save_model = Tasklet("save_model", self.save_model)

        # create a loop object with loop exit condition function
        loop = Loop(loop_check_fn=lambda: self._work_done)
        (
            task_internal_init
            >> task_load_data
            >> task_init
            >> loop(
                task_get_dataset
                >> task_put
                >> task_get
                >> task_train
                >> task_eval
                >> task_analysis
                >> task_save_metrics
                >> task_increment_round
            )
            >> task_end_of_training
            >> task_save_params
            >> task_save_model
        )

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the top level aggregator role."""
        return [TAG_DISTRIBUTE, TAG_AGGREGATE, TAG_GET_DATATSET_SIZE]
