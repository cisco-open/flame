# Copyright 2024 Cisco Systems, Inc. and its affiliates
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

from flame.mode.composer import CloneComposer
from flame.mode.horizontal.syncfl.middle_aggregator import (
    TAG_AGGREGATE,
    TAG_DISTRIBUTE,
    TAG_FETCH,
    TAG_UPLOAD,
)
from flame.mode.horizontal.syncfl.middle_aggregator import (
    MiddleAggregator as BaseLeafAggregator,
)
from flame.mode.message import MessageType
from flame.mode.tasklet import Tasklet

logger = logging.getLogger(__name__)

TAG_COORDINATE = "coordinate"


class LeafAggregator(BaseLeafAggregator):
    def get_channel(self, tag: str):
        """Return channel of a given tag when it is ready to use."""
        channel = self.cm.get_by_tag(tag)
        if not channel:
            raise ValueError(f"channel not found for tag {tag}")

        channel.await_join()

        return channel

    def _liveness_check(self) -> None:
        logger.debug("calling _liveness_check")

        channel = self.get_channel(TAG_COORDINATE)

        end = channel.one_end()
        msg, _ = channel.recv(end)

        if MessageType.EOT in msg:
            self._work_done = msg[MessageType.EOT]

        if self._work_done:
            logger.debug("work is done")
            return

        if MessageType.META_INFO_REQ not in msg:
            raise ValueError("no meta info req message")

        # here middle aggregator can send some useful meta info to coordinator
        # meta information can be overhead during the previous round
        #
        # TODO: implement the logic
        logger.debug("sending meta info response")
        channel.send(end, {MessageType.META_INFO_RES: "some useful info"})
        logger.debug("sent meta info response")

        logger.debug("exited _liveness_check")

    def _get_trainers(self) -> None:
        logger.debug("getting trainers from coordinator")

        channel = self.get_channel(TAG_COORDINATE)

        end = channel.one_end()
        msg, _ = channel.recv(end)
        logger.debug(f"received msg = {msg} from {end}")

        if MessageType.COORDINATED_ENDS not in msg:
            raise ValueError("no coordinated ends message")

        self._work_done = msg[MessageType.EOT]
        if self._work_done:
            logger.debug("work is done")
            return

        dist_channel = self.cm.get_by_tag(TAG_DISTRIBUTE)
        # overide distribute channel's ends method
        dist_channel.ends = lambda: msg[MessageType.COORDINATED_ENDS]

        logger.debug("received trainers from coordinator")

    def _handle_no_trainer(self):
        channel = self.cm.get_by_tag(TAG_DISTRIBUTE)

        self.no_trainer = False
        if len(channel.ends()) == 0:
            logger.debug("no trainers found")
            self.no_trainer = True
            return

    def _get_mid_aggregator(self):
        logger.debug("calling _get_mid_aggregator")
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
            self.mid_aggregator_id = msg[MessageType.COORDINATED_ENDS]
        logger.debug("exited _get_mid_aggregator")

    def _fetch_weights(self, tag: str) -> None:
        logger.debug("calling _fetch_weights")

        self.fetch_success = False
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found with tag {tag}")
            return

        # this call waits for at least one peer joins this channel
        channel.await_join()

        msg, _ = channel.recv(self.mid_aggregator_id)

        if MessageType.WEIGHTS in msg:
            self.weights = msg[MessageType.WEIGHTS]

        if MessageType.EOT in msg:
            self._work_done = msg[MessageType.EOT]

        if MessageType.ROUND in msg:
            self._round = msg[MessageType.ROUND]

        self.fetch_success = True
        logger.debug(f"work_done: {self._work_done}, round: {self._round}")

    def _send_weights(self, tag: str) -> None:
        logger.debug("calling _send_weights")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"[_send_weights] channel not found with {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        delta_weights = self._delta_weights_fn(self.weights, self.prev_weights)

        msg = {
            MessageType.WEIGHTS: delta_weights,
            MessageType.DATASET_SIZE: self.dataset_size,
            MessageType.MODEL_VERSION: self._round,
        }
        channel.send(self.mid_aggregator_id, msg)
        logger.debug("sending weights done")

    def compose(self) -> None:
        """Compose role with tasklets."""
        super().compose()

        with CloneComposer(self.composer) as composer:
            self.composer = composer

            task_liveness_check = Tasklet("liveness_check", self._liveness_check)

            task_get_mid_aggregator = Tasklet("get_mid_aggregator", self._get_mid_aggregator)

            task_get_trainers = Tasklet("get_trainers", self._get_trainers)

            task_no_trainer = Tasklet("handle_no_trainer", self._handle_no_trainer)

            task_no_trainer.set_continue_fn(cont_fn=lambda: self.no_trainer)

        self.composer.get_tasklet("fetch").insert_before(task_get_mid_aggregator)
        task_get_mid_aggregator.insert_before(task_no_trainer)
        task_no_trainer.insert_before(task_get_trainers)
        task_get_trainers.insert_before(task_liveness_check)

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the top level
        aggregator role.
        """
        return [
            TAG_AGGREGATE,
            TAG_DISTRIBUTE,
            TAG_FETCH,
            TAG_UPLOAD,
            TAG_COORDINATE,
        ]
