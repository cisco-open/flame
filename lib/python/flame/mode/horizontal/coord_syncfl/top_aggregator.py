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

from flame.mode.composer import CloneComposer
from flame.mode.horizontal.syncfl.top_aggregator import TAG_AGGREGATE, TAG_DISTRIBUTE
from flame.mode.horizontal.syncfl.top_aggregator import (
    TopAggregator as BaseTopAggregator,
)
from flame.mode.message import MessageType
from flame.mode.tasklet import Tasklet

logger = logging.getLogger(__name__)

TAG_COORDINATE = "coordinate"  # coordinate with the coordinator


class TopAggregator(BaseTopAggregator):
    def get_channel(self, tag: str):
        """Return channel of a given tag when it is ready to use."""
        channel = self.cm.get_by_tag(tag)
        if not channel:
            raise ValueError(f"channel not found for tag {tag}")

        channel.await_join()

        return channel

    def get_coordinated_ends(self):
        """Receive the ends of middle aggregators."""
        logger.debug("calling get_coordinate_ends()")
        channel = self.get_channel(TAG_COORDINATE)

        end = channel.one_end()
        msg, _ = channel.recv(end)
        logger.debug(f"received message = {msg} from {end}")

        self._work_done = msg[MessageType.EOT]
        if self._work_done:
            logger.debug("work is done")
            return

        dist_channel = self.cm.get_by_tag(TAG_DISTRIBUTE)
        # overide distribute channel's ends method
        dist_channel.ends = lambda: msg[MessageType.COORDINATED_ENDS]

        logger.debug("exited get_coordinate_ends()")

    def compose(self) -> None:
        """Compose role with tasklets."""
        super().compose()

        with CloneComposer(self.composer) as composer:
            self.composer = composer

            task_get_coord_ends = Tasklet("get_coord_ends", self.get_coordinated_ends)


        self.composer.get_tasklet("distribute").insert_before(task_get_coord_ends)
        self.composer.get_tasklet("inform_end_of_training").remove()

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the top level aggregator role."""
        return [TAG_AGGREGATE, TAG_DISTRIBUTE, TAG_COORDINATE]
