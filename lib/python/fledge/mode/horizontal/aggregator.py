# Copyright (c) 2021 Cisco Systems, Inc. and its affiliates
# All rights reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""horizontal FL aggregator."""

import logging
import time

from diskcache import Cache

from ...channel_manager import ChannelManager
from ...common.custom_abcmeta import ABCMeta, abstract_attribute
from ...optimizer.train_result import TrainResult
from ...optimizers import optimizer_provider
from ...plugin import PluginManager, PluginType
from ...registries import registry_provider
from ..composer import Composer
from ..role import Role
from ..tasklet import Loop, Tasklet

logger = logging.getLogger(__name__)

TAG_DISTRIBUTE = 'distribute'
TAG_AGGREGATE = 'aggregate'


class Aggregator(Role, metaclass=ABCMeta):
    """Aggregator implements an ML aggregation role."""

    @abstract_attribute
    def weights(self):
        """Abstract attribute for model weights."""

    @abstract_attribute
    def config(self):
        """Abstract attribute for config object."""

    @abstract_attribute
    def metrics(self):
        """
        Abstract attribute for metrics object.

        metrics must be the form of dict(str, float).
        """

    @abstract_attribute
    def model(self):
        """Abstract attribute for model object."""

    @abstract_attribute
    def dataset(self):
        """
        Abstract attribute for datset.

        dataset's type is Dataset (in fledge/dataset.py).
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
                base_model.name, base_model.version)

        self.registry_client.setup_run()

        # disk cache is used for saving memory in case model is large
        self.cache = Cache()
        self.optimizer = optimizer_provider.get(self.config.optimizer)

        self._round = 1
        self._rounds = 1
        if 'rounds' in self.config.hyperparameters:
            self._rounds = self.config.hyperparameters['rounds']
        self._work_done = False

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
        for end in channel.ends():
            msg = channel.recv(end)
            if not msg:
                logger.debug(f"No data received from {end}")
                continue

            weights = msg[0]
            count = msg[1]

            total += count
            logger.debug(f"{end}'s parameters trained with {count} samples")

            tres = TrainResult(weights, count)
            # save training result from trainer in a disk cache
            self.cache[end] = tres

        # optimizer conducts optimization (in this case, aggregation)
        global_weights = self.optimizer.do(self.cache, total)
        if global_weights is None:
            logger.debug("failed model aggregation")
            time.sleep(1)
            return

        # set global weights
        self.weights = global_weights

    def put(self, tag: str) -> None:
        """Set data to remote role(s)."""
        if tag == TAG_DISTRIBUTE:
            self._distribute_weights(tag)

    def _distribute_weights(self, tag: str) -> None:
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found for tag {tag}")
            return

        while len(channel.ends()) == 0:
            logger.debug("no end found in the channel")
            time.sleep(1)

        self._work_done = (self._round >= self._rounds)

        # send out global model parameters to trainers
        for end in channel.ends():
            logger.debug(f"sending weights to {end}")
            channel.send(end, (self._work_done, self.weights))

        self._round += 1

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

            # merge metrics with the existing metrics
            self.metrics = self.metrics | metrics

    def save_metrics(self):
        """Save metrics in a model registry."""
        logger.debug(f"saving metrics: {self.metrics}")
        if self.metrics:
            self.registry_client.save_metrics(self._round - 1, self.metrics)
            logger.debug("saving metrics done")

    def save_params(self):
        """Save hyperparamets in a model registry."""
        if self.config.hyperparameters:
            self.registry_client.save_params(self.config.hyperparameters)

    def save_model(self):
        """Save model in a model registry."""
        if self.model:
            model_name = f"{self.config.job.name}-{self.config.job.job_id}"
            self.registry_client.save_model(model_name, self.model)

    def compose(self) -> None:
        """Compose role with tasklets."""
        with Composer() as composer:
            self.composer = composer

            task_internal_init = Tasklet(self.internal_init)

            task_init = Tasklet(self.initialize)

            task_load_data = Tasklet(self.load_data)

            task_put = Tasklet(self.put, TAG_DISTRIBUTE)

            task_get = Tasklet(self.get, TAG_AGGREGATE)

            task_train = Tasklet(self.train)

            task_eval = Tasklet(self.evaluate)

            task_analysis = Tasklet(self.run_analysis)

            task_save_metrics = Tasklet(self.save_metrics)

            task_save_params = Tasklet(self.save_params)

            task_save_model = Tasklet(self.save_model)

        # create a loop object with loop exit condition function
        loop = Loop(loop_check_fn=lambda: self._work_done)
        task_internal_init >> task_init >> loop(
            task_load_data >> task_put >> task_get >> task_train >> task_eval
            >> task_analysis >> task_save_metrics
        ) >> task_save_params >> task_save_model

    def run(self) -> None:
        """Run role."""
        self.composer.run()

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the aggregator role."""
        return [TAG_DISTRIBUTE, TAG_AGGREGATE]
