# Copyright 2022 Cisco Systems, Inc. and its affiliates
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
"""horizontal FL top level aggregator."""

import logging
import time
from copy import deepcopy
from datetime import datetime

from diskcache import Cache
from flame.channel_manager import ChannelManager
from flame.common.constants import DeviceType
from flame.common.custom_abcmeta import ABCMeta, abstract_attribute
from flame.common.util import (
    MLFramework,
    get_ml_framework_in_use,
    mlflow_runname,
    valid_frameworks,
    weights_to_device,
    weights_to_model_device,
)
from flame.config import Config
from flame.mode.composer import Composer
from flame.mode.message import MessageType
from flame.mode.role import Role
from flame.mode.tasklet import Loop, Tasklet
from flame.optimizer.train_result import TrainResult
from flame.optimizers import optimizer_provider
from flame.plugin import PluginManager, PluginType
from flame.registries import registry_provider
from flame.datasamplers import datasampler_provider

logger = logging.getLogger(__name__)

TAG_DISTRIBUTE = "distribute"
TAG_AGGREGATE = "aggregate"
PROP_ROUND_START_TIME = "round_start_time"
PROP_ROUND_END_TIME = "round_end_time"


class TopAggregator(Role, metaclass=ABCMeta):
    """Top level Aggregator implements an ML aggregation role."""
    
    @abstract_attribute
    def config(self) -> Config:
        """Abstract attribute for config object."""

    @abstract_attribute
    def model(self):
        """Abstract attribute for model object."""

    @abstract_attribute
    def dataset(self):
        """
        Abstract attribute for datset.

        dataset's type is Dataset (in flame/dataset.py).
        """

    def internal_init(self) -> None:
        """Initialize internal state for role."""
        # global variable for plugin manager
        self.plugin_manager = PluginManager()

        self.cm = ChannelManager()
        self.cm(self.config)
        self.cm.join_all()

        self.registry_client = registry_provider.get(self.config.registry.sort)
        # initialize registry client
        self.registry_client(self.config.registry.uri, self.config.job.job_id)

        base_model = self.config.base_model
        if base_model and base_model.name != "" and base_model.version > 0:
            self.model = self.registry_client.load_model(
                base_model.name, base_model.version
            )

        self.registry_client.setup_run(mlflow_runname(self.config))
        self.metrics = dict()

        # disk cache is used for saving memory in case model is large
        self.cache = Cache()
        self.optimizer = optimizer_provider.get(
            self.config.optimizer.sort, **self.config.optimizer.kwargs
        )

        self.datasampler = datasampler_provider.get(
            self.config.datasampler.sort, **self.config.datasampler.kwargs
        ).aggregator_data_sampler
        
        self._round = 1
        self._rounds = 1
        self._rounds = self.config.hyperparameters.rounds
        self._work_done = False

        self.framework = get_ml_framework_in_use()
        if self.framework == MLFramework.UNKNOWN:
            raise NotImplementedError(
                "supported ml framework not found; "
                f"supported frameworks are: {valid_frameworks}"
            )

    def get(self, tag: str) -> None:
        """Get data from remote role(s)."""
        if tag == TAG_AGGREGATE:
            self._aggregate_weights(tag)

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

        # optimizer conducts optimization (in this case, aggregation)
        global_weights = self.optimizer.do(
            deepcopy(self.weights),
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

    def put(self, tag: str) -> None:
        """Set data to remote role(s)."""
        if tag == TAG_DISTRIBUTE:
            self.dist_tag = tag
            self._distribute_weights(tag)

    def _distribute_weights(self, tag: str) -> None:
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found for tag {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        # before distributing weights, update it from global model
        self._update_weights()

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
                    MessageType.ROUND: self._round,
                    MessageType.DATASAMPLER_METADATA: datasampler_metadata,
                },
            )
            # register round start time on each end for round duration measurement.
            channel.set_end_property(
                end, PROP_ROUND_START_TIME, (round, datetime.now())
            )

    def inform_end_of_training(self) -> None:
        """Inform all the trainers that the training is finished."""
        channel = self.cm.get_by_tag(self.dist_tag)
        if not channel:
            logger.debug(f"channel not found for tag {self.dist_tag}")
            return

        channel.broadcast({MessageType.EOT: self._work_done})
        logger.debug("done broadcasting end-of-training")

    def run_analysis(self):
        """Run analysis plugins and update results to metrics."""
        logger.debug("running analyzer plugins")

        plugins = self.plugin_manager.get_plugins(PluginType.ANALYZER)
        for plugin in plugins:
            # get callback function and call it
            func = plugin.callback()
            metrics = func(self.model, self.dataset)
            if not metrics:
                continue

            self.update_metrics(metrics)

    def save_metrics(self):
        """Save metrics in a model registry."""
        # update metrics with metrics from metric collector
        self.metrics = self.metrics | self.mc.get()
        logger.debug(f"saving metrics: {self.metrics}")
        if self.metrics:
            self.registry_client.save_metrics(self._round - 1, self.metrics)
            logger.debug("saving metrics done")

    def increment_round(self):
        """Increment the round counter."""
        logger.debug(f"Incrementing current round: {self._round}")
        logger.debug(f"Total rounds: {self._rounds}")
        self._round += 1
        self._work_done = self._round > self._rounds

        channel = self.cm.get_by_tag(self.dist_tag)
        if not channel:
            logger.debug(f"channel not found for tag {self.dist_tag}")
            return

        logger.debug(f"Incremented round to {self._round}")
        # set necessary properties to help channel decide how to select ends
        channel.set_property("round", self._round)

    def save_params(self):
        """Save hyperparamets in a model registry."""
        if self.config.hyperparameters:
            self.registry_client.save_params(self.config.hyperparameters)

    def save_model(self):
        """Save model in a model registry."""
        if self.model:
            model_name = f"{self.config.job.name}-{self.config.job.job_id}"
            self.registry_client.save_model(model_name, self.model)

    def update_metrics(self, metrics: dict[str, float]):
        """Update metrics."""
        self.metrics = self.metrics | metrics

    def _update_model(self):
        if self.framework == MLFramework.PYTORCH:
            self.model.load_state_dict(self.weights)
        elif self.framework == MLFramework.TENSORFLOW:
            self.model.set_weights(self.weights)

    def _update_weights(self):
        if self.framework == MLFramework.PYTORCH:
            self.weights = self.model.state_dict()
        elif self.framework == MLFramework.TENSORFLOW:
            self.weights = self.model.get_weights()

    def compose(self) -> None:
        """Compose role with tasklets."""
        with Composer() as composer:
            self.composer = composer

            task_internal_init = Tasklet("internal_init", self.internal_init)

            task_init = Tasklet("initialize", self.initialize)

            task_load_data = Tasklet("load_data", self.load_data)

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
                task_put
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

    def run(self) -> None:
        """Run role."""
        self.composer.run()

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the top level aggregator role."""
        return [TAG_DISTRIBUTE, TAG_AGGREGATE]
