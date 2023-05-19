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
"""FedDyn horizontal FL top level aggregator."""

import logging
from copy import deepcopy
import time

from flame.common.util import (MLFramework, get_ml_framework_in_use,
                            weights_to_device, weights_to_model_device)
from flame.common.constants import (DeviceType, TrainState)
from flame.optimizer.train_result import TrainResult
from flame.mode.composer import Composer
from flame.mode.message import MessageType
from flame.mode.tasklet import Loop, Tasklet
from flame.mode.horizontal.top_aggregator import TopAggregator as BaseTopAggregator

logger = logging.getLogger(__name__)

TAG_DISTRIBUTE = 'distribute'
TAG_AGGREGATE = 'aggregate'
TAG_GET_DATATSET_SIZE = 'getDatasetSize'

PROP_ROUND_END_TIME = "round_end_time"

class TopAggregator(BaseTopAggregator):
    """FedDyn Top level Aggregator implements an ML aggregation role."""

    def internal_init(self) -> None:
        """Initialize internal state for role."""
        ml_framework_in_use = get_ml_framework_in_use()
        
        # only support pytorch
        if ml_framework_in_use != MLFramework.PYTORCH:
            raise NotImplementedError(
                "supported ml framework not found; "
                f"supported frameworks (for feddyn) are: {[MLFramework.PYTORCH.name.lower()]}")
        
        super().internal_init()

    def get(self, tag: str) -> None:
        """Get data from remote role(s)."""
        if tag == TAG_AGGREGATE:
            self._aggregate_weights(tag)
        elif tag == TAG_GET_DATATSET_SIZE:
            self.get_dataset_size(tag)
    
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

        logger.debug(f"received {len(self.cache)} trainer updates in cache")

        # optimizer conducts optimization (in this case, aggregation)
        global_weights = self.optimizer.do(
            deepcopy(self.cld_weights),
            self.cache,
            total=total,
            num_trainers=len(channel.ends()),
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
        
        # record all active trainers
        self.optimizer.save_state(TrainState.PRE, active_ends = all_ends)
        logger.debug(f"dataset sizes: {self.dataset_sizes}")
        logger.debug("exiting get_dataset_size")

    def _distribute_weights(self, tag: str) -> None:
        logger.debug("calling _distribute_weights")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found for tag {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        # before distributing weights, update it from global model
        self._update_weights()
        
        total_samples = sum(self.dataset_sizes.values())
        num_trainers = len(self.dataset_sizes)
        weight_dict = {end:(self.dataset_sizes[end]/total_samples) * num_trainers for end in self.dataset_sizes}
        
        logger.debug(f"weight_dict: {weight_dict}")
        
        self.cld_weights = self.optimizer.cld_model
        if self.cld_weights == None:
            self.cld_weights = self.weights

        # send out global model parameters to trainers
        for end in channel.ends():
            logger.debug(f"sending weights to {end}")
            channel.send(end, {
                MessageType.WEIGHTS: weights_to_device(self.cld_weights, DeviceType.CPU),
                MessageType.ROUND: self._round,
                MessageType.ALPHA_ADPT: self.optimizer.alpha / weight_dict.get(end, 1)
            })

    def compose(self) -> None:
        """Compose role with tasklets."""
        with Composer() as composer:
            self.composer = composer

            task_internal_init = Tasklet("internal_init", self.internal_init)

            task_init = Tasklet("initialize", self.initialize)

            task_load_data = Tasklet("load_data", self.load_data)

            task_get_dataset = Tasklet("get_dataset_size", self.get, TAG_GET_DATATSET_SIZE)

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
        task_internal_init >> task_load_data >> task_init >> loop(
            task_get_dataset >> task_put >> task_get >> task_train >> 
            task_eval >> task_analysis >> task_save_metrics >> 
            task_increment_round
        ) >> task_end_of_training >> task_save_params >> task_save_model

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the top level aggregator role."""
        return [TAG_DISTRIBUTE, TAG_AGGREGATE, TAG_GET_DATATSET_SIZE]
