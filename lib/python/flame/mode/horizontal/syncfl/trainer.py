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
"""horizontal FL trainer."""

import logging

from flame.channel import VAL_CH_STATE_RECV, VAL_CH_STATE_SEND
from flame.channel_manager import ChannelManager
from flame.common.constants import DeviceType, TrainerState
from flame.common.custom_abcmeta import ABCMeta, abstract_attribute
from flame.common.util import (
    MLFramework,
    delta_weights_pytorch,
    delta_weights_tensorflow,
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
from flame.optimizers import optimizer_provider
from flame.registries import registry_provider
from flame.datasamplers import datasampler_provider

logger = logging.getLogger(__name__)

TAG_FETCH = "fetch"
TAG_UPLOAD = "upload"


class Trainer(Role, metaclass=ABCMeta):
    """Trainer implements an ML training role."""

    @abstract_attribute
    def config(self) -> Config:
        """Abstract attribute for config object."""

    @abstract_attribute
    def model(self):
        """Abstract attribute for model object."""

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

        self.registry_client.setup_run(mlflow_runname(self.config))
        self.metrics = dict()

        # needed for trainer-side optimization algorithms such as fedprox
        temp_opt = optimizer_provider.get(
            self.config.optimizer.sort, **self.config.optimizer.kwargs
        )
        self.regularizer = temp_opt.regularizer

        self.datasampler = datasampler_provider.get(
            self.config.datasampler.sort, **self.config.datasampler.kwargs
        ).trainer_data_sampler

        self._round = 1
        self._work_done = False

        self.framework = get_ml_framework_in_use()
        if self.framework == MLFramework.UNKNOWN:
            raise NotImplementedError(
                "supported ml framework not found; "
                f"supported frameworks are: {valid_frameworks}"
            )

        if self.framework == MLFramework.PYTORCH:
            self._delta_weights_fn = delta_weights_pytorch

        elif self.framework == MLFramework.TENSORFLOW:
            self._delta_weights_fn = delta_weights_tensorflow

    def get(self, tag: str) -> None:
        """Get data from remote role(s)."""
        if tag == TAG_FETCH:
            self._fetch_weights(tag)

    def _fetch_weights(self, tag: str) -> None:
        logger.debug("calling _fetch_weights")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"[_fetch_weights] channel not found with tag {tag}")
            return

        # this call waits for at least one peer joins this channel
        channel.await_join()

        # one aggregator is sufficient
        end = channel.one_end(VAL_CH_STATE_RECV)
        msg, _ = channel.recv(end)

        if MessageType.WEIGHTS in msg:
            self.weights = weights_to_model_device(msg[MessageType.WEIGHTS], self.model)
            self._update_model()

        if MessageType.EOT in msg:
            self._work_done = msg[MessageType.EOT]

        if MessageType.ROUND in msg:
            self._round = msg[MessageType.ROUND]

        if MessageType.DATASAMPLER_METADATA in msg:
            self.datasampler.handle_metadata_from_aggregator(
                msg[MessageType.DATASAMPLER_METADATA]
            )

        self.regularizer.save_state(TrainerState.PRE_TRAIN, glob_model=self.model)
        logger.debug(f"work_done: {self._work_done}, round: {self._round}")

    def put(self, tag: str) -> None:
        """Set data to remote role(s)."""
        if tag == TAG_UPLOAD:
            self._send_weights(tag)

    def _send_weights(self, tag: str) -> None:
        logger.debug("calling _send_weights")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"[_send_weights] channel not found with {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        # one aggregator is sufficient
        end = channel.one_end(VAL_CH_STATE_SEND)

        self._update_weights()
        self.regularizer.save_state(TrainerState.POST_TRAIN, loc_model=self.model)

        delta_weights = self._delta_weights_fn(self.weights, self.prev_weights)

        # send delta_weights to regularizer
        self.regularizer.update()

        msg = {
            MessageType.WEIGHTS: weights_to_device(delta_weights, DeviceType.CPU),
            MessageType.DATASET_SIZE: self.dataset_size,
            MessageType.MODEL_VERSION: self._round,
            MessageType.DATASAMPLER_METADATA: self.datasampler.get_metadata(),
        }
        channel.send(end, msg)
        logger.debug("sending weights done")

    def save_metrics(self):
        """Save metrics in a model registry."""
        logger.debug(f"saving metrics: {self.metrics}")
        if self.metrics:
            self.registry_client.save_metrics(self._round - 1, self.metrics)
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
        # save weights before updating it
        self.prev_weights = self.weights

        if self.framework == MLFramework.PYTORCH:
            self.weights = self.model.state_dict()
        elif self.framework == MLFramework.TENSORFLOW:
            self.weights = self.model.get_weights()

    def compose(self) -> None:
        """Compose role with tasklets."""
        with Composer() as composer:
            self.composer = composer

            task_internal_init = Tasklet("", self.internal_init)

            task_load_data = Tasklet("", self.load_data)

            task_init = Tasklet("", self.initialize)

            task_get = Tasklet("", self.get, TAG_FETCH)

            task_train = Tasklet("", self.train)

            task_eval = Tasklet("", self.evaluate)

            task_put = Tasklet("", self.put, TAG_UPLOAD)

            task_save_metrics = Tasklet("", self.save_metrics)

            # create a loop object with loop exit condition function
            loop = Loop(loop_check_fn=lambda: self._work_done)
            (
                task_internal_init
                >> task_load_data
                >> task_init
                >> loop(
                    task_get >> task_train >> task_eval >> task_put >> task_save_metrics
                )
            )

    def run(self) -> None:
        """Run role."""
        self.composer.run()

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the trainer role."""
        return [TAG_FETCH, TAG_UPLOAD]
