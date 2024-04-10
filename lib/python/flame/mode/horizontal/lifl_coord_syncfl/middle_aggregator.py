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
    MiddleAggregator as BaseMiddleAggregator,
)
from flame.mode.message import MessageType
from flame.mode.tasklet import Tasklet

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

    def _liveness_check(self) -> None:
        logger.debug("calling _liveness_check")

        channel = self.get_channel(TAG_COORDINATE)

        end = channel.one_end()
        msg, _ = channel.recv(end)

        if MessageType.EOT in msg:
            self._work_done = msg[MessageType.EOT]

        if self._work_done:
            logger.debug("work is done")
            self.no_leaf_agg = True # stop mid agg
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

    def _get_leaf_aggs(self) -> None:
        logger.debug("getting leaf aggs from coordinator")

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

        logger.debug("received leaf aggregators from coordinator")

    def _handle_no_leaf_agg(self):
        channel = self.cm.get_by_tag(TAG_DISTRIBUTE)

        self.no_leaf_agg = False
        if len(channel.ends()) == 0:
            logger.debug("no leaf aggs found")
            self.no_leaf_agg = True
            return

    def compose(self) -> None:
        """Compose role with tasklets."""
        super().compose()

        with CloneComposer(self.composer) as composer:
            self.composer = composer

            task_liveness_check = Tasklet("liveness_check", self._liveness_check)

            task_get_leaf_aggs = Tasklet("get_leaf_aggs", self._get_leaf_aggs)

            task_no_leaf_agg = Tasklet("handle_no_leaf_agg", self._handle_no_leaf_agg)

            task_no_leaf_agg.set_continue_fn(cont_fn=lambda: self.no_leaf_agg)

        self.composer.get_tasklet("fetch").insert_before(task_no_leaf_agg)
        task_no_leaf_agg.insert_before(task_get_leaf_aggs)
        task_get_leaf_aggs.insert_before(task_liveness_check)

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
