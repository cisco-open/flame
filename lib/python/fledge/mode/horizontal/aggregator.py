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

    def internal_init(self) -> None:
        """Initialize internal state for role."""
        self.cm = ChannelManager()
        self.cm(self.config)
        self.cm.join_all()

        self._round = 1
        self._repeats = 1

        if 'rounds' in self.config.hyperparameters:
            self._repeats = self.config.hyperparameters['rounds']

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

        self._work_done = (self._round >= self._repeats)

        # send out global model parameters to trainers
        for end in channel.ends():
            logger.debug(f"sending weights to {end}")
            channel.send(end, (self._work_done, self.weights))

        self._round += 1

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

            task_eval = Tasklet(
                self.evaluate, loop_check_func=lambda: self._work_done
            )

            task_internal_init >> task_init >> loop(
                task_load_data >> task_put >> task_get >> task_train >>
                task_eval
            )

    def run(self) -> None:
        """Run role."""
        self.composer.run()
