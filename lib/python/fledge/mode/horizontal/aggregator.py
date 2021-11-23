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

from ...channel_manager import ChannelManager
from ...common.custom_abcmeta import ABCMeta, abstract_attribute
from ...registries import registry_provider
from ..composer import Composer
from ..role import Role
from ..tasklet import Tasklet, loop

logger = logging.getLogger(__name__)

TAG_DISTRIBUTE = 'distribute'
TAG_AGGREGATE = 'aggregate'


class Aggregator(Role, metaclass=ABCMeta):
    """Aggregator implements an ML aggregation role."""

    #
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

    def internal_init(self) -> None:
        """Initialize internal state for role."""
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

        self.registry_client.setup_run()

        self._epoch = 1
        self._epochs = 1

        if 'rounds' in self.config.hyperparameters:
            self._epochs = self.config.hyperparameters['rounds']

    def get(self, tag: str) -> None:
        """Get data from remote role(s)."""
        if tag == TAG_AGGREGATE:
            self._aggregate_weights(tag)

    def _aggregate_weights(self, tag: str) -> None:
        channel = self.cm.get_by_tag(tag)
        if not channel:
            return

        ##############################################################
        # TODO: this aggregation part should be modularized
        #       by making an algorithm pluggable.
        #
        #       The following is the implementation of FedSGD algo.
        ##############################################################
        total = 0
        weights_array = []
        # receive local model parameters from trainers
        for end in channel.ends():
            msg = channel.recv(end)
            if not msg:
                logger.debug(f"No data received from {end}")
                continue

            weights = msg[0]
            count = msg[1]
            total += count
            weights_array.append((weights, count))
            logger.debug(f"{end}'s parameters trained with {count} samples")

        if len(weights_array) == 0 or total == 0:
            logger.debug("no local model parameters are obtained")
            time.sleep(1)
            return

        count = weights_array[0][1]
        rate = count / total
        global_weights = [weight * rate for weight in weights_array[0][0]]

        for weights, count in weights_array[1:]:
            rate = count / total

            for idx in range(len(weights)):
                global_weights[idx] += weights[idx] * rate
        ##############################################################

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

        self._work_done = (self._epoch >= self._epochs)

        # send out global model parameters to trainers
        for end in channel.ends():
            logger.debug(f"sending weights to {end}")
            channel.send(end, (self._work_done, self.weights))

        self._epoch += 1

    def save_metrics(self):
        """Save metrics in a model registry."""
        logger.debug(f"saving metrics: {self.metrics}")
        if self.metrics:
            self.registry_client.save_metrics(self._epoch - 1, self.metrics)
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

            task_save_metrics = Tasklet(
                self.save_metrics, loop_check_func=lambda: self._work_done
            )

            task_save_params = Tasklet(self.save_params)

            task_save_model = Tasklet(self.save_model)

        task_internal_init >> task_init >> loop(
            task_load_data >> task_put >> task_get >> task_train >> task_eval >>
            task_save_metrics
        ) >> task_save_params >> task_save_model

    def run(self) -> None:
        """Run role."""
        self.composer.run()

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the aggregator role."""
        return [TAG_DISTRIBUTE, TAG_AGGREGATE]
