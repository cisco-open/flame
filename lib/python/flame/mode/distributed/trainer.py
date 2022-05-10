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

"""distributed FL trainer."""

import logging

from diskcache import Cache

from ...channel_manager import ChannelManager
from ...common.custom_abcmeta import ABCMeta, abstract_attribute
from ...common.util import (MLFramework, get_ml_framework_in_use,
                            valid_frameworks,  mlflow_runname)
from ...optimizer.train_result import TrainResult
from ...optimizers import optimizer_provider
from ...registries import registry_provider
from ..composer import Composer
from ..role import Role
from ..message import MessageType
from ..tasklet import Loop, Tasklet

logger = logging.getLogger(__name__)

TAG_RECEIVE = 'receive'
TAG_SEND = 'send'

class Trainer(Role, metaclass=ABCMeta):
    """Trainer implements an ML training role."""

    @abstract_attribute
    def config(self):
        """Abstract attribute for config object."""

    @abstract_attribute
    def dataset_size(self):
        """Abstract attribute for size of dataset used to train."""

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
                base_model.name, base_model.version)

        self.registry_client.setup_run(mlflow_runname(self.config))
        self.metrics = dict()

        self.cache = Cache()
        self.total_data_points = 0
        self.optimizer = optimizer_provider.get(self.config.optimizer.sort,
                                                **self.config.optimizer.kwargs)

        self._round = 1
        self._rounds = self.config.hyperparameters['rounds']
        self._work_done = False

        self._lead_trainer = None

        self.framework = get_ml_framework_in_use()
        if self.framework == MLFramework.UNKNOWN:
            raise NotImplementedError(
                "supported ml framework not found; "
                f"supported frameworks are: {valid_frameworks}")

    def get(self, tag: str) -> None:
        """Get data from remote role(s)."""
        if tag == TAG_RECEIVE:
            self._receive_weights(tag)

    def _receive_weights(self, tag: str) -> None:
        logger.debug("calling _receive_weights")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"[_receive_weights] channel not found with tag {tag}")
            return

        self.total_data_points = 0
        for end in channel.ends():
            dict = channel.recv(end)
            if not dict:
                logger.debug(f"No data received from {end}")
                continue

            for k, v in dict.items():
                if k == MessageType.WEIGHTS:
                    weights = v
                elif k == MessageType.DATASET_SIZE:
                    count = v
                elif k == MessageType.ROUND and v > self._round:
                    self._round = v

            self.total_data_points += count
            logger.debug(f"{end}'s parameters trained with {count} samples")

            if weights is not None:
                tres = TrainResult(weights, count)
                self.cache[end] = tres

    def _aggregate_weights(self) -> None:
        logger.debug("aggregating weights from all trainers")
        global_weights = self.optimizer.do(self.cache, self.total_data_points)
        if global_weights is None:
            logger.debug("failed model aggregation")
            return

        # set global weights
        self.weights = global_weights
        # update model with global weights
        self._update_model()

    def put(self, tag: str) -> None:
        """Set data to remote role(s)."""
        if tag == TAG_SEND:
            self.select_tag = tag
            self._send_weights(tag)

    def _send_weights(self, tag: str) -> None:
        logger.debug("calling _send_weights")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"[_send_weights] channel not found with {tag}")
            return
        
        self._lead_trainer = sorted(channel.ends())[0]

        self._update_weights()
        channel.broadcast({MessageType.WEIGHTS: self.weights if self._round != 1 else None, 
                           MessageType.DATASET_SIZE: self.dataset_size, 
                           MessageType.ROUND: self._round})
        logger.debug("broadcasting weights done")

    def save_metrics(self):
        """Save metrics in a model registry."""
        logger.debug(f"saving metrics: {self.metrics}")
        if self.metrics:
            self.registry_client.save_metrics(self._round, self.metrics)
            logger.debug("saving metrics done")

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

    def increment_round(self):
        """Increment the round counter."""
        logger.debug(f"Incrementing current round: {self._round}")

        self._round += 1
        self._work_done = (self._round > self._rounds)

    def save_params(self):
        """Save hyperparamets in a model registry."""

        channel = self.cm.get_by_tag(self.select_tag)

        if self.config.hyperparameters and channel.get_backend_id() == self._lead_trainer:
            self.registry_client.save_params(self.config.hyperparameters)

    def save_model(self):
        """Save model in a model registry."""

        channel = self.cm.get_by_tag(self.select_tag)

        if self.model and channel.get_backend_id() == self._lead_trainer:
            model_name = f"{self.config.job.name}-{self.config.job.job_id}"
            self.registry_client.save_model(model_name, self.model)

    def compose(self) -> None:
        """Compose role with tasklets."""
        with Composer() as composer:
            self.composer = composer

            task_internal_init = Tasklet(self.internal_init)

            task_load_data = Tasklet(self.load_data)

            task_init = Tasklet(self.initialize)

            task_receive = Tasklet(self.get, TAG_RECEIVE)

            task_train = Tasklet(self.train)

            task_eval = Tasklet(self.evaluate)

            task_send = Tasklet(self.put, TAG_SEND)

            task_aggregate = Tasklet(self._aggregate_weights)

            task_increment_round = Tasklet(self.increment_round)

            task_save_metrics = Tasklet(self.save_metrics)

            task_save_params = Tasklet(self.save_params)

            task_save_model = Tasklet(self.save_model)

            # create a loop object with loop exit condition function
            loop = Loop(loop_check_fn=lambda: self._work_done)
            task_internal_init >> task_load_data >> task_init >> loop(
                task_send >> task_receive >> task_aggregate 
                >> task_train >> task_eval >> task_save_metrics 
                >> task_increment_round) >> task_save_params >> task_save_model

    def run(self) -> None:
        """Run role."""
        self.composer.run()

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the trainer role."""
        return [TAG_RECEIVE, TAG_SEND]
