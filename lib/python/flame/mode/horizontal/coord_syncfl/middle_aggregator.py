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

from flame.mode.composer import Composer
from flame.mode.horizontal.syncfl.middle_aggregator import (
    TAG_AGGREGATE,
    TAG_DISTRIBUTE,
    TAG_FETCH,
    TAG_UPLOAD,
)
from flame.mode.horizontal.syncfl.middle_aggregator import (
    MiddleAggregator as BaseMiddleAggregator,
)
from flame.mode.message import MessageType
from flame.mode.tasklet import Loop, Tasklet

logger = logging.getLogger(__name__)

TAG_COORDINATE = "coordinate"


class MiddleAggregator(BaseMiddleAggregator):
    def get_channel(self, tag: str):
        """Return channel of a given tag when it is ready to use."""
        channel = self.cm.get_by_tag(tag)
        if not channel:
            raise ValueError(f"channel not found for tag {tag}")

        channel.await_join()

        return channel

    def _get_trainers(self) -> None:
        logger.debug("getting trainers from coordinator")

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

    def compose(self) -> None:
        """Compose role with tasklets."""
        with Composer() as composer:
            self.composer = composer

            task_internal_init = Tasklet("", self.internal_init)

            task_init = Tasklet("", self.initialize)

            task_load_data = Tasklet("", self.load_data)

            task_get_trainers = Tasklet("", self._get_trainers)

            task_no_trainer = Tasklet("", self._handle_no_trainer)
            task_no_trainer.set_continue_fn(cont_fn=lambda: self.no_trainer)

            task_put_dist = Tasklet("", self.put, TAG_DISTRIBUTE)

            task_put_upload = Tasklet("", self.put, TAG_UPLOAD)

            task_get_aggr = Tasklet("", self.get, TAG_AGGREGATE)

            task_get_fetch = Tasklet("", self.get, TAG_FETCH)

            task_eval = Tasklet("", self.evaluate)

            task_update_round = Tasklet("", self.update_round)

            task_end_of_training = Tasklet("", self.inform_end_of_training)

        loop = Loop(loop_check_fn=lambda: self._work_done)
        (
            task_internal_init
            >> task_load_data
            >> task_init
            >> loop(
                task_get_trainers
                >> task_no_trainer
                >> task_get_fetch
                >> task_put_dist
                >> task_get_aggr
                >> task_put_upload
                >> task_eval
                >> task_update_round
            )
            >> task_end_of_training
        )

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the top level aggregator role."""
        return [TAG_AGGREGATE, TAG_DISTRIBUTE, TAG_FETCH, TAG_UPLOAD, TAG_COORDINATE]
