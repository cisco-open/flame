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

import logging
from abc import ABCMeta

from flame.common.constants import DeviceType
from flame.common.util import weights_to_device, weights_to_model_device
from flame.mode.composer import CloneComposer
from flame.mode.horizontal.syncfl.trainer import TAG_FETCH, TAG_UPLOAD
from flame.mode.horizontal.syncfl.trainer import Trainer as BaseTrainer
from flame.mode.message import MessageType
from flame.mode.tasklet import Tasklet

logger = logging.getLogger(__name__)

TAG_COORDINATE = "coordinate"


class Trainer(BaseTrainer, metaclass=ABCMeta):
    def _get_aggregator(self):
        logger.debug("calling _get_aggregator")
        channel = self.cm.get_by_tag(TAG_COORDINATE)
        if not channel:
            logger.debug(f"channel not found with tag {TAG_COORDINATE}")
            return

        channel.await_join()

        end = channel.one_end()
        msg, _ = channel.recv(end)

        self._work_done = msg[MessageType.EOT]
        if self._work_done:
            logger.debug("work is done")
            return

        if MessageType.COORDINATED_ENDS in msg:
            self.aggregator_id = msg[MessageType.COORDINATED_ENDS]
        logger.debug("exited _get_aggregator")

    def _fetch_weights(self, tag: str) -> None:
        logger.debug("calling _fetch_weights")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found with tag {tag}")
            return

        # this call waits for at least one peer joins this channel
        channel.await_join()

        msg, _ = channel.recv(self.aggregator_id)

        if MessageType.WEIGHTS in msg:
            self.weights = weights_to_model_device(
                msg[MessageType.WEIGHTS], self.model
            )
            self._update_model()

        if MessageType.EOT in msg:
            self._work_done = msg[MessageType.EOT]

        if MessageType.ROUND in msg:
            self._round = msg[MessageType.ROUND]

        logger.debug(f"work_done: {self._work_done}, round: {self._round}")

    def _send_weights(self, tag: str) -> None:
        logger.debug("calling _send_weights")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"[_send_weights] channel not found with {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        self._update_weights()

        delta_weights = self._delta_weights_fn(self.weights, self.prev_weights)

        msg = {
            MessageType.WEIGHTS: weights_to_device(
                delta_weights, DeviceType.CPU
            ),
            MessageType.DATASET_SIZE: self.dataset_size,
            MessageType.MODEL_VERSION: self._round,
        }
        channel.send(self.aggregator_id, msg)
        logger.debug("sending weights done")

    def compose(self) -> None:
        super().compose()

        with CloneComposer(self.composer) as composer:
            self.composer = composer

            task_get_aggregator = Tasklet("", self._get_aggregator)

        self.composer.get_tasklet("fetch").insert_before(task_get_aggregator)

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the trainer role."""
        return [TAG_FETCH, TAG_UPLOAD, TAG_COORDINATE]
