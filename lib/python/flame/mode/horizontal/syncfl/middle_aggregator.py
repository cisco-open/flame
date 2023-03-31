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
"""honrizontal FL middle level aggregator."""

import logging
import time
from copy import deepcopy

from diskcache import Cache
from flame.channel_manager import ChannelManager
from flame.common.custom_abcmeta import ABCMeta, abstract_attribute
from flame.common.util import (
    MLFramework,
    delta_weights_pytorch,
    delta_weights_tensorflow,
    get_ml_framework_in_use,
    valid_frameworks,
)
from flame.config import Config
from flame.mode.composer import Composer
from flame.mode.message import MessageType
from flame.mode.role import Role
from flame.mode.tasklet import Loop, Tasklet
from flame.optimizer.train_result import TrainResult
from flame.optimizers import optimizer_provider
from flame.plugin import PluginManager

logger = logging.getLogger(__name__)

TAG_DISTRIBUTE = "distribute"
TAG_AGGREGATE = "aggregate"
TAG_FETCH = "fetch"
TAG_UPLOAD = "upload"

# 60 second wait time until a trainer appears in a channel
WAIT_TIME_FOR_TRAINER = 60


class MiddleAggregator(Role, metaclass=ABCMeta):
    """Middle level aggregator.

    It acts as a proxy between top level aggregator and trainer.
    """

    @abstract_attribute
    def config(self) -> Config:
        """Abstract attribute for config object."""

    def internal_init(self) -> None:
        """Initialize internal state for role."""
        # global variable for plugin manager
        self.plugin_manager = PluginManager()

        self.cm = ChannelManager()
        self.cm(self.config)
        self.cm.join_all()

        self.optimizer = optimizer_provider.get(
            self.config.optimizer.sort, **self.config.optimizer.kwargs
        )

        self._round = 1
        self._work_done = False

        self.cache = Cache()
        self.dataset_size = 0

        # save distribute tag in an instance variable
        self.dist_tag = TAG_DISTRIBUTE

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
        if tag == TAG_AGGREGATE:
            self._aggregate_weights(tag)

    def put(self, tag: str) -> None:
        """Set data to remote role(s)."""
        if tag == TAG_UPLOAD:
            self._send_weights(tag)
        if tag == TAG_DISTRIBUTE:
            self._distribute_weights(tag)

    def _fetch_weights(self, tag: str) -> None:
        logger.debug("calling _fetch_weights")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"[_fetch_weights] channel not found with tag {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        # one aggregator is sufficient
        end = channel.one_end()
        msg, _ = channel.recv(end)

        if MessageType.WEIGHTS in msg:
            self.weights = msg[MessageType.WEIGHTS]

        if MessageType.EOT in msg:
            self._work_done = msg[MessageType.EOT]

        if MessageType.ROUND in msg:
            self._round = msg[MessageType.ROUND]

    def _distribute_weights(self, tag: str) -> None:
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found for tag {tag}")
            return

        # this call waits for at least one peer to join this channel
        self.trainer_no_show = channel.await_join(WAIT_TIME_FOR_TRAINER)
        if self.trainer_no_show:
            logger.debug("channel await join timeouted")
            # send dummy weights to unblock top aggregator
            self._send_dummy_weights(TAG_UPLOAD)
            return

        for end in channel.ends():
            logger.debug(f"sending weights to {end}")
            channel.send(
                end, {MessageType.WEIGHTS: self.weights, MessageType.ROUND: self._round}
            )

    def _aggregate_weights(self, tag: str) -> None:
        channel = self.cm.get_by_tag(tag)
        if not channel:
            return

        total = 0
        # receive local model parameters from trainers
        for msg, metadata in channel.recv_fifo(channel.ends()):
            end, _ = metadata
            if not msg:
                logger.debug(f"No data from {end}; skipping it")
                continue

            if MessageType.WEIGHTS in msg:
                weights = msg[MessageType.WEIGHTS]

            if MessageType.DATASET_SIZE in msg:
                count = msg[MessageType.DATASET_SIZE]

            logger.debug(f"{end}'s parameters trained with {count} samples")

            if weights is not None and count > 0:
                total += count
                tres = TrainResult(weights, count)
                # save training result from trainer in a disk cache
                self.cache[end] = tres

        # optimizer conducts optimization (in this case, aggregation)
        global_weights = self.optimizer.do(
            deepcopy(self.weights), self.cache, total=total
        )
        if global_weights is None:
            logger.debug("failed model aggregation")
            time.sleep(1)
            return

        # save global weights before updating it
        self.prev_weights = self.weights

        # set global weights
        self.weights = global_weights
        self.dataset_size = total

    def _send_weights(self, tag: str) -> None:
        logger.debug("calling _send_weights")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found with {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        # one aggregator is sufficient
        end = channel.one_end()

        delta_weights = self._delta_weights_fn(self.weights, self.prev_weights)

        msg = {
            MessageType.WEIGHTS: delta_weights,
            MessageType.DATASET_SIZE: self.dataset_size,
            MessageType.MODEL_VERSION: self._round,
        }
        channel.send(end, msg)
        logger.debug("sending weights done")

    def _send_dummy_weights(self, tag: str) -> None:
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found with {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        # one aggregator is sufficient
        end = channel.one_end()

        dummy_msg = {MessageType.WEIGHTS: None, MessageType.DATASET_SIZE: 0}
        channel.send(end, dummy_msg)
        logger.debug("sending dummy weights done")

    def update_round(self):
        """Update the round counter."""
        logger.debug(f"Update current round: {self._round}")

        channel = self.cm.get_by_tag(self.dist_tag)
        if not channel:
            logger.debug(f"channel not found for tag {self.dist_tag}")
            return

        # set necessary properties to help channel decide how to select ends
        channel.set_property("round", self._round)

    def inform_end_of_training(self) -> None:
        """Inform all the trainers that the training is finished."""
        logger.debug("inform end of training")

        channel = self.cm.get_by_tag(self.dist_tag)
        if not channel:
            logger.debug(f"channel not found for tag {self.dist_tag}")
            return

        channel.broadcast({MessageType.EOT: self._work_done})

    def compose(self) -> None:
        """Compose role with tasklets."""
        with Composer() as composer:
            self.composer = composer

            task_internal_init = Tasklet(self.internal_init)

            task_init = Tasklet(self.initialize)

            task_load_data = Tasklet(self.load_data)

            task_put_dist = Tasklet(self.put, TAG_DISTRIBUTE)
            task_put_dist.set_continue_fn(cont_fn=lambda: self.trainer_no_show)

            task_put_upload = Tasklet(self.put, TAG_UPLOAD)

            task_get_aggr = Tasklet(self.get, TAG_AGGREGATE)

            task_get_fetch = Tasklet(self.get, TAG_FETCH)

            task_eval = Tasklet(self.evaluate)

            task_update_round = Tasklet(self.update_round)

            task_end_of_training = Tasklet(self.inform_end_of_training)

        # create a loop object with loop exit condition function
        loop = Loop(loop_check_fn=lambda: self._work_done)
        (
            task_internal_init
            >> task_load_data
            >> task_init
            >> loop(
                task_get_fetch
                >> task_put_dist
                >> task_get_aggr
                >> task_put_upload
                >> task_eval
                >> task_update_round
            )
            >> task_end_of_training
        )

    def run(self) -> None:
        """Run role."""
        self.composer.run()

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the middle level aggregator role."""
        return [TAG_DISTRIBUTE, TAG_AGGREGATE, TAG_FETCH, TAG_UPLOAD]
